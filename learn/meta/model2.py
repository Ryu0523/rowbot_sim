#!/usr/bin/env python3
"""
The step-2 inference model (learn/meta/PRIOR_DERIVATION.md, decisions D3
and D4): predict the distribution of the next H observed errors
e_{k..k+H-1} under a planned command sequence, from the history.

Future errors come in three levels (D4):

    c    the boat's RULES (fixed or slow): a low-dimensional latent code.
         Uncertainty about c is the only thing probing can reduce.
    h    the current state of hidden processes, read from the recent
         history by the encoder
    nu   innovations: waves not yet met, spray, slam timing, noise

    e_{k+j} = D(g_{k+j}, x_{k+j}, j; c) + nu_{k+j}

  bank     a learned stable linear filter bank on the measured inputs
           (diagonal complex modes, lambda = exp(-exp(nu) + i theta), poles
           scheduled by slow signals; a learned nonlinearity before it). It
           carries lags, memory and wave-locked phase. Over the horizon it
           is driven by what the MPC knows: the model's own rollout under
           the plan, and no waves beyond the current step.
  decoder  D, nonlinear in c (FiLM). kind='linear' is the old basis form
           Phi(.) . c, kept as a baseline only.
  encoder  a transformer over up to 250 history tokens (inputs, and the
           observed errors with random observation noise), with a CLS
           summary.
  c-flow   conditional rectified flow: p(c | history).
  nu-flow  conditional rectified flow over the whole H x 5 innovation
           trajectory, given the history summary, c and the plan. It
           generates any shape, including 'near zero plus a rare large
           spike' (slams), in asinh space.

Training (two stages):
  A  auto-decoding. For every training checkpoint, a free code c* and the
     bank + decoder are fitted to the operator's relabelled RULE targets:
     the own future and donor segments, 125 steps of warm-up then H steps.
     Donors 10-12 are held out of the loss, as a check of generalisation
     across plans.
  B  encoder + c-flow on c* | history prefix, and nu-flow on
     nu* = e_observed - D(c*) along the own future, trained jointly.
"""
import math
import os
import pickle
import time

import numpy as np
import torch
from torch import nn

C, D_S, D_X, H, WARM = 5, 26, 11, 20, 125
D_U = 2                 # the command applied DURING the step (thrust, nozzle)
D_IN = D_S + D_U + 2    # + waves-available flag, wave-known-this-step flag
PATIENCE = 3            # validations without improvement before stopping
N_FIT = 10              # donors 1..9 fit the codes; 10..12 held out; the own
                        # future (0) never enters the code fit
MAX_HIST = 250


def device():
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.35)
        return torch.device("cuda")
    return torch.device("cpu")


def mlp(n_in, hidden, n_out, act=nn.SiLU):
    layers, n = [], n_in
    for h_ in hidden:
        layers += [nn.Linear(n, h_), act()]
        n = h_
    layers.append(nn.Linear(n, n_out))
    return nn.Sequential(*layers)


def t_embed(tau, n=32):
    f = torch.exp(torch.linspace(0, math.log(1000.0), n // 2,
                                 device=tau.device))
    a = tau[..., None] * f
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


# ------------------------------------------------------------------ data
class Split:
    """A packed split on the device. Inputs normalised by the library's
    mu / sd, errors by the TRAINING per-channel sd (pass `stats`)."""

    def __init__(self, cache, split, dev, stats=None, targets=True):
        d = np.load(os.path.join(cache, f"{split}.npz"))
        self.meta = pickle.load(open(os.path.join(cache,
                                                  f"{split}_meta.pkl"), "rb"))
        lib = np.load(os.path.join(cache, "lib.npz"))
        # the library normalises the operator's 28 inputs: S's 26, then U
        mu28, sd28 = lib["mu"].astype(np.float32), lib["sd"].astype(
            np.float32)
        s_mu, s_sd = mu28[:D_S], sd28[:D_S]
        u_mu, u_sd = mu28[D_S:D_S + D_U], sd28[D_S:D_S + D_U]
        n, T, _ = d["S"].shape
        self.n, self.T = n, T
        valid = np.arange(T)[None] < d["len"][:, None]
        S = ((d["S"] - s_mu) / s_sd) * valid[..., None]
        E = d["E"]
        if stats is None:
            stats = dict(s_mu=s_mu, s_sd=s_sd,
                         e_sd=E[valid].std(0).astype(np.float32) + 1e-6)
        self.stats = stats
        e_sd = stats["e_sd"]
        self.S = torch.tensor(S, dtype=torch.float32, device=dev)
        # commands applied during each step, on the scale of S's actuator
        # columns (S[k, 5:7] is the actuator state BEFORE step k)
        self.U = torch.tensor(((d["U"] - u_mu) / u_sd)
                              * valid[..., None], dtype=torch.float32,
                              device=dev)
        self.E = torch.tensor(E / e_sd * valid[..., None],
                              dtype=torch.float32, device=dev)
        self.len = torch.tensor(d["len"], device=dev)
        self.CK = torch.tensor(d["CK"].astype(np.int64), device=dev)
        self.n_ck = self.CK.shape[1]
        self.has_rule = False
        if targets:
            t = np.load(os.path.join(cache, f"{split}_targets.npz"))
            self.Y = torch.tensor(t["SEG_Y"] / e_sd, dtype=torch.float32,
                                  device=dev)
            self.XH = torch.tensor((t["SEG_XH"] - s_mu[:D_X]) / s_sd[:D_X],
                                   dtype=torch.float32, device=dev)
            self.REF = torch.tensor(t["SEG_REF"].astype(np.int64),
                                    device=dev)
            seeds = np.array([m["seed"] for m in self.meta])
            if "seeds" in t.files and not np.array_equal(t["seeds"], seeds):
                raise RuntimeError(f"{split}: targets do not match the pack")
            code = self.meta[0].get("code") if self.meta else None
            if "code" in t.files and code and str(t["code"]) != code:
                raise RuntimeError(f"{split}: targets built by other code")
            self.has_rule = bool(np.abs(t["SEG_Y"]).sum() > 0)
        self.pairs = torch.nonzero(self.CK >= 0)          # (P, 2): i, c
        self.info = np.array([m["informative"] for m in self.meta])
        self.style = [m["style"] for m in self.meta]
        self.dev = dev


def seg_inputs(D, ii, cc, mm, wave_off):
    """Bank inputs (B, WARM + H, D_IN) and valid mask (B, WARM + H) for
    segments (ii, cc, mm): WARM steps of the referenced episode's recorded
    inputs and commands, then the horizon: the model's own rollout (XH),
    the true waves at j = 0 only, and the planned commands (the recorded
    ones). wave_off (B,) bool: the waves group removed.
    Layout: [S (26), U (2), waves available, wave known this step]."""
    ref = D.REF[ii, cc, mm]
    j, t0 = ref[:, 0], ref[:, 1]
    B = len(ii)
    ar = torch.arange(WARM, device=D.dev)[None]
    idx = t0[:, None] - WARM + ar
    ok = idx >= 0
    idc = idx.clamp(min=0)
    warm = torch.cat([D.S[j[:, None], idc], D.U[j[:, None], idc]],
                     -1) * ok[..., None]
    hz = torch.zeros(B, H, D_S + D_U, device=D.dev)
    hz[..., :D_X] = D.XH[ii, cc, mm]
    hz[:, 0, D_X:D_S] = D.S[j, t0, D_X:]
    hz[..., D_S:] = D.U[j[:, None], t0[:, None] + torch.arange(
        H, device=D.dev)[None]]
    seq = torch.cat([warm, hz], 1)
    known = torch.zeros(B, WARM + H, 1, device=D.dev)
    known[:, :WARM + 1] = 1.0
    avail = (~wave_off).float()[:, None, None].expand(B, WARM + H, 1)
    seq = torch.cat([seq, avail, known * avail], -1)
    seq[..., D_X:D_S] = seq[..., D_X:D_S] * avail
    mask = torch.cat([ok, torch.ones(B, H, dtype=torch.bool,
                                     device=D.dev)], 1)
    return seq, mask


# ----------------------------------------------------------------- model
class Bank(nn.Module):
    """Learned LNL filter bank: v = lin(s) + mlp(s), then n_m diagonal
    complex modes x' = lambda_k x + gamma B v (gamma = sqrt(1 - |lambda|^2)
    keeps the state scale), lambda_k scheduled by slow inputs low-passed at
    5 s. Initial poles cover the prior's ranges: real modes tau 0.12-30 s,
    resonant modes 0.3-10 rad/s with zeta 0.1."""

    SLOW = (0, 3, 4, 5)

    def __init__(self, n_v=16, n_m=64, dt=0.24):
        super().__init__()
        self.lin = nn.Linear(D_IN, n_v)
        self.mlp = mlp(D_IN, (64,), n_v)
        n_r = n_m // 2
        tau = torch.exp(torch.linspace(math.log(0.12), math.log(30.0), n_r))
        wn = torch.exp(torch.linspace(math.log(0.3), math.log(10.0),
                                      n_m - n_r))
        zeta = 0.1
        log_rate = torch.cat([torch.log(dt / tau),
                              torch.log(zeta * wn * dt)])
        theta = torch.cat([torch.zeros(n_r),
                           wn * dt * math.sqrt(1 - zeta ** 2)])
        self.log_rate = nn.Parameter(log_rate)
        self.theta = nn.Parameter(theta)
        self.Br = nn.Parameter(torch.randn(n_m, n_v) / math.sqrt(n_v))
        self.Bi = nn.Parameter(torch.randn(n_m, n_v) / math.sqrt(n_v))
        self.kap = nn.Parameter(torch.zeros(n_m, len(self.SLOW)))
        self.rho = nn.Parameter(torch.zeros(n_m, len(self.SLOW)))
        self.n_m, self.n_v, self.dt = n_m, n_v, dt
        self.a_slow = math.exp(-dt / 5.0)
        self.d_out = 2 * n_m + 3 * n_v

    def forward(self, seq, mask):
        """seq (B, T, D_IN), mask (B, T) -> features (B, T, 2 n_m + 3 n_v):
        the modes, and the bank input v delayed by 1-3 steps (a delay line,
        as the operator prior delays its filter inputs by up to 3 steps).

        x_k = lambda_k x_{k-1} + gamma_k B v_k, x_{-1} = 0, computed by a
        parallel (Hillis-Steele) prefix scan over the pairs (lambda, input):
        log2(T) vectorised steps instead of T sequential ones. Inputs before
        a history's first valid step are zero, so its state stays zero
        there."""
        B, T, _ = seq.shape
        dev = seq.device
        v = (self.lin(seq) + self.mlp(seq)) * mask[..., None]
        ur, ui = v @ self.Br.T, v @ self.Bi.T
        sl = seq[..., list(self.SLOW)] * mask[..., None]
        k = torch.arange(T, device=dev)
        dk = (k[:, None] - k[None]).clamp(min=0).float()
        Km = (1 - self.a_slow) * self.a_slow ** dk * (k[:, None] >= k[None])
        sb = torch.einsum("kj,bjs->bks", Km, sl)
        rate = torch.exp(self.log_rate + sb @ self.kap.T)
        th = self.theta * torch.exp(sb @ self.rho.T)
        r = torch.exp(-rate)
        g = torch.sqrt(torch.clamp(1 - r * r, min=1e-6))
        ar, ai = r * torch.cos(th), r * torch.sin(th)
        br, bi = g * ur, g * ui
        d = 1
        while d < T:
            pr = ar[:, d:] * br[:, :-d] - ai[:, d:] * bi[:, :-d]
            pi = ar[:, d:] * bi[:, :-d] + ai[:, d:] * br[:, :-d]
            nar = ar[:, d:] * ar[:, :-d] - ai[:, d:] * ai[:, :-d]
            nai = ar[:, d:] * ai[:, :-d] + ai[:, d:] * ar[:, :-d]
            br = torch.cat([br[:, :d], br[:, d:] + pr], 1)
            bi = torch.cat([bi[:, :d], bi[:, d:] + pi], 1)
            ar = torch.cat([ar[:, :d], nar], 1)
            ai = torch.cat([ai[:, :d], nai], 1)
            d *= 2
        lags = [torch.cat([torch.zeros_like(v[:, :l]), v[:, :-l]], 1)
                for l in (1, 2, 3)]
        return torch.cat([br, bi] + lags, -1)


class Decoder(nn.Module):
    """D(g, s, j; c). 'film': an MLP whose hidden layers are scaled and
    shifted by c. 'linear': Phi(g, s, j) . c per channel (baseline)."""

    def __init__(self, c_dim, d_g, kind="film", hidden=256, K=16):
        super().__init__()
        self.kind, self.c_dim, self.K = kind, c_dim, K
        self.jemb = nn.Embedding(H, 16)
        d_in = d_g + D_IN + 16
        if kind == "film":
            self.l1 = nn.Linear(d_in, hidden)
            self.l2 = nn.Linear(hidden, hidden)
            self.l3 = nn.Linear(hidden, hidden)
            self.out = nn.Linear(hidden, C)
            self.film = mlp(c_dim, (256,), 6 * hidden)
            self.hidden = hidden
        else:
            assert c_dim == C * K
            self.phi = mlp(d_in, (hidden, hidden), C * K)
        self.norm = nn.LayerNorm(d_g)

    def forward(self, g, s, c):
        """g (B, H, d_g), s (B, H, D_IN), c (B, c_dim) or (B, S, c_dim)
        -> (B, H, C) or (B, S, H, C)."""
        B = g.shape[0]
        j = self.jemb(torch.arange(H, device=g.device))[None].expand(B, H,
                                                                     16)
        x = torch.cat([self.norm(g), s, j], -1)
        multi = c.dim() == 3
        if self.kind == "linear":
            Phi = self.phi(x).view(B, H, C, self.K)
            cc = c.view(*c.shape[:-1], C, self.K)
            if multi:
                return torch.einsum("bhck,bsck->bshc", Phi, cc)
            return torch.einsum("bhck,bck->bhc", Phi, cc)
        f = self.film(c)
        if multi:
            x = x[:, None].expand(B, c.shape[1], H, x.shape[-1])
            f = f[:, :, None]
        else:
            f = f[:, None]
        g1, b1, g2, b2, g3, b3 = f.chunk(6, -1)
        act = nn.functional.silu
        h_ = act(self.l1(x) * (1 + g1) + b1)
        h_ = act(self.l2(h_) * (1 + g2) + b2)
        h_ = act(self.l3(h_) * (1 + g3) + b3)
        return self.out(h_)


class Encoder(nn.Module):
    def __init__(self, d=128, layers=4, heads=4):
        super().__init__()
        self.inp = mlp(D_IN + C, (d,), d)
        self.pos = nn.Parameter(torch.zeros(MAX_HIST + 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        lay = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(lay, layers,
                                        enable_nested_tensor=False)
        self.out = nn.LayerNorm(d)
        self.d = d

    def forward(self, tok, mask):
        """tok (B, L, D_IN + C), mask (B, L) True = valid; token L-1 is the
        most recent. -> (B, d)."""
        B, L, _ = tok.shape
        h_ = self.inp(tok) + self.pos[1:L + 1].flip(0)[None]
        h_ = torch.cat([self.cls.expand(B, 1, self.d) + self.pos[:1], h_], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool,
                                     device=tok.device), ~mask], 1)
        return self.out(self.tf(h_, src_key_padding_mask=pad)[:, 0])


class CFlow(nn.Module):
    def __init__(self, c_dim, d_ctx=128, hidden=(512, 512, 512)):
        super().__init__()
        self.net = mlp(c_dim + 32 + d_ctx, hidden, c_dim)
        self.c_dim = c_dim

    def forward(self, z, tau, ctx):
        return self.net(torch.cat([z, t_embed(tau), ctx], -1))

    @torch.no_grad()
    def sample(self, ctx, n, steps=24, gen=None):
        B = ctx.shape[0]
        cx = ctx.repeat_interleave(n, 0)
        z = torch.randn(B * n, self.c_dim, device=ctx.device, generator=gen)
        for i in range(steps):
            tau = torch.full((B * n,), i / steps, device=ctx.device)
            z = z + self(z, tau, cx) / steps
        return z.view(B, n, self.c_dim)


NU_SCALE = 0.5          # asinh space: nu = NU_SCALE sinh(y)


class NuFlow(nn.Module):
    """Rectified flow over the (H, C) innovation trajectory in asinh space,
    a transformer over the H horizon tokens. Per-step condition: the
    horizon inputs, the flags, D's prediction and j; global: flow time,
    the history summary and the code."""

    def __init__(self, c_dim, d_ctx=128, d=128, layers=3, heads=4):
        super().__init__()
        self.tok = nn.Linear(C + D_IN + C + 16, d)
        self.glob = mlp(32 + d_ctx + c_dim, (d,), d)
        self.jemb = nn.Embedding(H, 16)
        lay = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(lay, layers,
                                        enable_nested_tensor=False)
        self.out = nn.Linear(d, C)

    def forward(self, y, tau, ctx, cz, feat):
        """y (B, H, C), tau (B,), ctx (B, d_ctx), cz (B, c_dim), feat
        (B, H, D_IN + C) -> velocity (B, H, C)."""
        B = y.shape[0]
        j = self.jemb(torch.arange(H, device=y.device))[None].expand(B, H,
                                                                     16)
        h_ = self.tok(torch.cat([y, feat, j], -1))
        h_ = h_ + self.glob(torch.cat([t_embed(tau), ctx, cz], -1))[:, None]
        return self.out(self.tf(h_))

    @torch.no_grad()
    def sample(self, ctx, cz, feat, steps=24, gen=None):
        B = ctx.shape[0]
        y = torch.randn(B, H, C, device=ctx.device, generator=gen)
        for i in range(steps):
            tau = torch.full((B,), i / steps, device=ctx.device)
            y = y + self(y, tau, ctx, cz, feat) / steps
        return NU_SCALE * torch.sinh(y)


# ----------------------------------------------------------- utilities
def history_tokens(D, ii, end, Hl, wave_off, obs_sd=None, gen=None):
    """Tokens (B, MAX_HIST, D_IN + C) of the last Hl steps before `end`
    (right-aligned: the most recent is the last token), and the mask.
    Errors enter as asinh of the normalised error (+ observation noise)."""
    B = len(ii)
    ar = torch.arange(MAX_HIST, device=D.dev)[None]
    idx = end[:, None] - MAX_HIST + ar
    mask = (ar >= MAX_HIST - Hl[:, None]) & (idx >= 0)
    idc = idx.clamp(min=0)
    s = D.S[ii[:, None], idc]
    u_ = D.U[ii[:, None], idc]
    e = D.E[ii[:, None], idc]
    if obs_sd is not None:
        e = e + obs_sd[:, None, :] * torch.randn(e.shape, device=D.dev,
                                                  generator=gen)
    avail = (~wave_off).float()[:, None, None].expand(B, MAX_HIST, 1)
    s = s.clone()
    s[..., D_X:] = s[..., D_X:] * avail
    tok = torch.cat([s, u_, avail, avail, torch.asinh(e)], -1)         * mask[..., None]
    return tok, mask


def obs_noise_sd(B, e_sd, dev, gen=None):
    """Observation noise per episode: white, sd LogU[1e-3, 0.03] x A_REF,
    in normalised units (PRIOR_DERIVATION G12)."""
    a_ref = torch.tensor([4.0, 4.0, 2.0, 12.0, 3.0], device=dev)
    u = torch.rand(B, 1, device=dev, generator=gen)
    f = torch.exp(math.log(1e-3) + u * (math.log(0.03) - math.log(1e-3)))
    return f * a_ref / torch.tensor(e_sd, device=dev)


class Model(nn.Module):
    def __init__(self, kind="film", c_dim=None, n_m=64):
        super().__init__()
        c_dim = c_dim or (32 if kind == "film" else C * 16)
        self.kind, self.c_dim = kind, c_dim
        self.bank = Bank(n_m=n_m)
        self.dec = Decoder(c_dim, self.bank.d_out, kind)
        self.enc = Encoder()
        self.cflow = CFlow(c_dim)
        self.nuflow = NuFlow(c_dim)
        # True: the model never sees the wave elevations (the main design:
        # only the boat's states, the commands and the history of the
        # unexpected response; any unmeasured cause shows up there)
        self.waves_off = False

    def woff(self, n, dev):
        return torch.full((n,), bool(self.waves_off), dtype=torch.bool,
                          device=dev)

    def seg_decode(self, D, ii, cc, mm, c, wave_off):
        seq, mask = seg_inputs(D, ii, cc, mm, wave_off)
        g = self.bank(seq, mask)[:, WARM:]
        return self.dec(g, seq[:, WARM:], c), seq[:, WARM:]


# -------------------------------------------------------------- stage A
def train_decoder(D, kind="film", steps=30000, batch=48, lr=1e-3,
                  lr_code=1e-2, p_wave_off=0.15, log=print, seed=0):
    dev = D.dev
    torch.manual_seed(seed)
    model = Model(kind).to(dev)
    codes = nn.Parameter(0.01 * torch.randn(D.n, D.n_ck, model.c_dim,
                                            device=dev))
    net = list(model.bank.parameters()) + list(model.dec.parameters())
    opt = torch.optim.Adam([dict(params=net, lr=lr),
                            dict(params=[codes], lr=lr_code)])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr, lr_code], total_steps=steps, pct_start=0.05)
    g = torch.Generator(device="cpu").manual_seed(seed)
    M = D.REF.shape[2]
    # a FIXED validation set of checkpoints, scored by ratios of sums over
    # it (sum of squared error / sum of squared target): one batch's
    # weighted mean jumped with single large events (0.88 -> 1.46 in the
    # preliminary run) and could not tell over-fitting from noise
    gv = torch.Generator(device="cpu").manual_seed(seed + 7)
    val = D.pairs[torch.randperm(len(D.pairs), generator=gv)[:256]]
    best = dict(held=float("inf"), it=-1)

    @torch.no_grad()
    def validate():
        num = torch.zeros(2, C, device=dev)
        den = torch.zeros(2, C, device=dev)
        for s0 in range(0, len(val), 32):
            r = val[s0:s0 + 32]
            b = len(r)
            ii = r[:, 0].repeat_interleave(M)
            cc = r[:, 1].repeat_interleave(M)
            mm = torch.arange(M, device=dev).repeat(b)
            wo = torch.full((b * M,), p_wave_off >= 1.0, dtype=torch.bool,
                            device=dev)
            pred, _ = model.seg_decode(D, ii, cc, mm, codes[ii, cc], wo)
            y = D.Y[ii, cc, mm]
            for k, m_ in enumerate(((mm >= 1) & (mm < N_FIT), mm >= N_FIT)):
                num[k] += ((pred - y) ** 2)[m_].sum((0, 1))
                den[k] += (y ** 2)[m_].sum((0, 1))
        return (num / den.clamp(min=1e-9)).cpu().numpy()

    t0 = time.time()
    for it in range(steps):
        r = D.pairs[torch.randint(0, len(D.pairs), (batch,), generator=g)]
        ii = r[:, 0].repeat_interleave(M)
        cc = r[:, 1].repeat_interleave(M)
        mm = torch.arange(M, device=dev).repeat(batch)
        woff = (torch.rand(batch, generator=g) < p_wave_off).to(dev) \
            .repeat_interleave(M)
        c = codes[ii, cc]
        pred, _ = model.seg_decode(D, ii, cc, mm, c, woff)
        y = D.Y[ii, cc, mm]
        # relative error: every checkpoint counts the same whatever its
        # operator's amplitude (the 30x amplitude span otherwise lets the
        # loudest operators decide the fit)
        scale = (y ** 2).mean((1, 2)).view(batch, M)[:, 1:N_FIT].mean(1)
        w = (1.0 / scale.clamp(min=1e-2)).repeat_interleave(M)
        se = ((pred - y) ** 2).mean((1, 2)) * w
        # the code is fitted on donors 1..9 only: never on the own future,
        # whose realised waves and innovations belong to nu (D4)
        fit = (mm >= 1) & (mm < N_FIT)
        held_m = mm >= N_FIT
        loss = se[fit].mean() + 1e-3 * (codes[r[:, 0], r[:, 1]] ** 2).sum(
            -1).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net, 1.0)
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == steps - 1:
            v = validate()
            held = float(v[1].mean())
            log(f"    decoder[{kind}] {it:6d}: validation (error/target "
                f"power) fitted donors {v[0].mean():.3f}, held-out donors "
                f"{held:.3f} [" + " ".join(f"{x:.2f}" for x in v[1])
                + f"]  ({time.time() - t0:.0f} s)")
            if it > steps // 4 and held >= best["held"]:
                best["bad"] = best.get("bad", 0) + 1
                if best["bad"] >= PATIENCE:
                    log(f"    decoder: held-out donors not better for "
                        f"{PATIENCE} checks, stopping at step {it}")
                    break
            if it > steps // 4 and held < best["held"]:
                best["bad"] = 0
                best.update(held=held, it=it, net={
                    k_: v_.detach().clone() for k_, v_ in
                    list(model.bank.state_dict().items())
                    + [("dec." + k2, v2) for k2, v2 in
                       model.dec.state_dict().items()]},
                    codes=codes.detach().clone())
    if best["it"] >= 0 and best["it"] != it:
        # keep the step whose held-out donors were predicted best
        net = best["net"]
        model.bank.load_state_dict({k_: v_ for k_, v_ in net.items()
                                    if not k_.startswith("dec.")})
        model.dec.load_state_dict({k_[4:]: v_ for k_, v_ in net.items()
                                   if k_.startswith("dec.")})
        log(f"    decoder: restored step {best['it']} (held-out "
            f"{best['held']:.3f})")
        return model, best["codes"]
    return model, codes.detach()


def fit_codes(model, D, steps=400, lr=3e-2, batch=64, log=None):
    """Test-time auto-decoding with the network frozen: the ORACLE code of
    each checkpoint of a split, fitted on donors 1..9 (a diagnostic of
    the decoder's capacity on unseen operators, never part of the method)."""
    dev = D.dev
    M = D.REF.shape[2]
    codes = torch.zeros(D.n, D.n_ck, model.c_dim, device=dev)
    for p in model.parameters():
        p.requires_grad_(False)
    for s0 in range(0, len(D.pairs), batch):
        r = D.pairs[s0:s0 + batch]
        b = len(r)
        cz = torch.zeros(b, model.c_dim, device=dev, requires_grad=True)
        opt = torch.optim.Adam([cz], lr)
        ii = r[:, 0].repeat_interleave(M)
        cc = r[:, 1].repeat_interleave(M)
        mm = torch.arange(M, device=dev).repeat(b)
        woff = model.woff(b * M, dev)
        seq, mask = seg_inputs(D, ii, cc, mm, woff)
        with torch.no_grad():
            gb = model.bank(seq, mask)[:, WARM:]
        y = D.Y[ii, cc, mm]
        fit = ((mm >= 1) & (mm < N_FIT)).view(b, M)
        scale = (y ** 2).mean((1, 2)).view(b, M)[:, 1:N_FIT].mean(1)
        w = 1.0 / scale.clamp(min=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps,
                                                           eta_min=1e-3)
        for _ in range(steps):
            pred = model.dec(gb, seq[:, WARM:], cz.repeat_interleave(M, 0))
            se = ((pred - y) ** 2).mean((1, 2)).view(b, M) * w[:, None]
            loss = (se * fit).sum() / fit.sum() + 1e-3 * (cz ** 2).sum(
                -1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        codes[r[:, 0], r[:, 1]] = cz.detach()
    for p in model.parameters():
        p.requires_grad_(True)
    return codes


# -------------------------------------------------------------- stage B
def train_posterior(model, D, codes, steps=20000, batch=128, lr=3e-4,
                    p_wave_off=0.15, log=print, seed=1):
    dev = D.dev
    zv = codes[D.pairs[:, 0], D.pairs[:, 1]]
    z_mu, z_sd = zv.mean(0), zv.std(0) + 1e-4
    params = (list(model.enc.parameters()) + list(model.cflow.parameters())
              + list(model.nuflow.parameters()))
    opt = torch.optim.AdamW(params, lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=steps,
                                                pct_start=0.05)
    g = torch.Generator(device="cpu").manual_seed(seed)
    # 3% of the episodes held out of stage B: their checkpoints score the
    # two flows' losses every 1000 steps (fixed noise), for early stopping
    gv = torch.Generator(device="cpu").manual_seed(seed + 11)
    ep_val = torch.randperm(D.n, generator=gv)[:max(1, D.n // 33)].to(dev)
    is_val = (D.pairs[:, 0:1] == ep_val[None]).any(1)
    train_pairs, val_pairs = D.pairs[~is_val], D.pairs[is_val][:512]
    best = dict(loss=float("inf"), it=-1, bad=0)

    def losses(r, gen, obs=True):
        ii, cc = r[:, 0], r[:, 1]
        b = len(r)
        end = D.CK[ii, cc]
        u = torch.rand(b, generator=gen).to(dev)
        short = torch.rand(b, generator=gen).to(dev) < 0.25
        Hl = torch.where(short, (u * 17).long(),
                         (u * (torch.clamp(end, max=MAX_HIST) + 1)).long())
        Hl = torch.minimum(Hl, end)
        woff = (torch.rand(b, generator=gen) < p_wave_off).to(dev)
        tok, hm = history_tokens(D, ii, end, Hl, woff, obs_noise_sd(
            b, D.stats["e_sd"], dev, torch.Generator(device=dev).manual_seed(
                int(torch.randint(0, 2 ** 31, (1,), generator=gen))))
            if obs else None)
        ctx = model.enc(tok, hm)
        z1 = (codes[ii, cc] - z_mu) / z_sd
        z0 = torch.randn(z1.shape, generator=gen).to(dev)
        tau = torch.rand(b, generator=gen).to(dev)
        zt = (1 - tau[:, None]) * z0 + tau[:, None] * z1
        l_c = ((model.cflow(zt, tau, ctx) - (z1 - z0)) ** 2).mean()
        mm = torch.zeros(b, dtype=torch.long, device=dev)
        with torch.no_grad():
            dm, sh = model.seg_decode(D, ii, cc, mm, codes[ii, cc], woff)
        ar = torch.arange(H, device=dev)[None]
        e_fut = D.E[ii[:, None], end[:, None] + ar]
        y1 = torch.asinh((e_fut - dm) / NU_SCALE)
        y0 = torch.randn(y1.shape, generator=gen).to(dev)
        tn = torch.rand(b, generator=gen).to(dev)
        yt = (1 - tn[:, None, None]) * y0 + tn[:, None, None] * y1
        l_n = ((model.nuflow(yt, tn, ctx, z1, torch.cat([sh, dm], -1))
                - (y1 - y0)) ** 2).mean()
        return l_c, l_n

    @torch.no_grad()
    def validate():
        gen = torch.Generator(device="cpu").manual_seed(12345)
        lc, ln_ = [], []
        for s0 in range(0, len(val_pairs), 128):
            a, b = losses(val_pairs[s0:s0 + 128], gen)
            lc.append(a.item())
            ln_.append(b.item())
        return float(np.mean(lc)), float(np.mean(ln_))

    t0 = time.time()
    for it in range(steps):
        r = train_pairs[torch.randint(0, len(train_pairs), (batch,),
                                      generator=g)]
        ii, cc = r[:, 0], r[:, 1]
        end = D.CK[ii, cc]
        u = torch.rand(batch, generator=g).to(dev)
        short = torch.rand(batch, generator=g).to(dev) < 0.25
        Hl = torch.where(short, (u * 17).long(),
                         (u * (torch.clamp(end, max=MAX_HIST) + 1)).long())
        Hl = torch.minimum(Hl, end)
        woff = (torch.rand(batch, generator=g) < p_wave_off).to(dev)
        tok, hm = history_tokens(D, ii, end, Hl, woff,
                                 obs_noise_sd(batch, D.stats["e_sd"], dev))
        ctx = model.enc(tok, hm)
        z1 = (codes[ii, cc] - z_mu) / z_sd
        z0 = torch.randn_like(z1)
        tau = torch.rand(batch, device=dev)
        zt = (1 - tau[:, None]) * z0 + tau[:, None] * z1
        loss_c = ((model.cflow(zt, tau, ctx) - (z1 - z0)) ** 2).mean()
        # innovations along the own future, given the true code
        mm = torch.zeros(batch, dtype=torch.long, device=dev)
        with torch.no_grad():
            dm, sh = model.seg_decode(D, ii, cc, mm, codes[ii, cc], woff)
        ar = torch.arange(H, device=dev)[None]
        e_fut = D.E[ii[:, None], end[:, None] + ar]
        y1 = torch.asinh((e_fut - dm) / NU_SCALE)
        y0 = torch.randn_like(y1)
        tn = torch.rand(batch, device=dev)
        yt = (1 - tn[:, None, None]) * y0 + tn[:, None, None] * y1
        feat = torch.cat([sh, dm], -1)
        loss_n = ((model.nuflow(yt, tn, ctx, z1, feat) - (y1 - y0))
                  ** 2).mean()
        loss = loss_c + loss_n
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == steps - 1:
            vc, vn = validate()
            log(f"    posterior {it:6d}: c-flow {loss_c.item():.4f}  "
                f"nu-flow {loss_n.item():.4f} | held-out episodes: c-flow "
                f"{vc:.4f}  nu-flow {vn:.4f}  ({time.time() - t0:.0f} s)")
            v = vc + vn
            if it > steps // 5:
                if v < best["loss"]:
                    best.update(loss=v, it=it, bad=0, state={
                        k_: {n_: x.detach().clone() for n_, x in
                             getattr(model, k_).state_dict().items()}
                        for k_ in ("enc", "cflow", "nuflow")})
                else:
                    best["bad"] += 1
                    if best["bad"] >= PATIENCE:
                        log(f"    posterior: held-out loss not better for "
                            f"{PATIENCE} checks, stopping at step {it}")
                        break
    if best["it"] >= 0 and best["it"] != it:
        for k_, sd in best["state"].items():
            getattr(model, k_).load_state_dict(sd)
        log(f"    posterior: restored step {best['it']} (held-out "
            f"{best['loss']:.4f})")
    return (z_mu, z_sd)


# ----------------------------------------------------------- prediction
@torch.no_grad()
def predict(model, zst, D, ii, cc, Hl, n_c=32, n_nu=4, wave_off=None,
            gen=None):
    """Posterior predictive over the own-future horizon of checkpoints
    (ii, cc) after a history of Hl steps: code samples (B, n_c, c_dim),
    the decoder's prediction per code (B, n_c, H, C), and samples of e
    (B, n_c * n_nu, H, C). Normalised units."""
    dev = D.dev
    B = len(ii)
    z_mu, z_sd = zst
    wave_off = model.woff(B, dev) if wave_off is None else wave_off
    end = D.CK[ii, cc]
    tok, hm = history_tokens(D, ii, end, Hl, wave_off)
    ctx = model.enc(tok, hm)
    zs = model.cflow.sample(ctx, n_c, gen=gen)                 # (B, n_c, cd)
    cs = zs * z_sd + z_mu
    seq, mask = seg_inputs(D, ii, cc, torch.zeros_like(ii), wave_off)
    gb = model.bank(seq, mask)[:, WARM:]
    sh = seq[:, WARM:]
    dm = model.dec(gb, sh, cs)                                 # (B,n_c,H,C)
    ctx_r = ctx.repeat_interleave(n_c * n_nu, 0)
    zr = zs.reshape(B * n_c, -1).repeat_interleave(n_nu, 0)
    dmr = dm.reshape(B * n_c, H, C).repeat_interleave(n_nu, 0)
    shr = sh.repeat_interleave(n_c * n_nu, 0)
    nu = model.nuflow.sample(ctx_r, zr, torch.cat([shr, dmr], -1), gen=gen)
    es = (dmr + nu).view(B, n_c * n_nu, H, C)
    return cs, dm, es


@torch.no_grad()
def decode_codes(model, D, ii, cc, mm, cs):
    """D for codes cs (B, S, c_dim) on segments (ii, cc, mm) -> (B, S, H, C)."""
    seq, mask = seg_inputs(D, ii, cc, mm, model.woff(len(ii), D.dev))
    gb = model.bank(seq, mask)[:, WARM:]
    return model.dec(gb, seq[:, WARM:], cs)


# ------------------------------------------------------------ top level
def _ckpt(cache, tag):
    return os.path.join(cache, f"model2{tag}.pt")


def train_all(cache, args, log=print):
    dev = device()
    kind = getattr(args, "kind", "film") or "film"
    tag = getattr(args, "tag", "") or ""
    smoke = getattr(args, "smoke", False)
    D = Split(cache, "train", dev)
    log(f"train: {D.n} episodes, {len(D.pairs)} checkpoints; e sd "
        f"{np.round(D.stats['e_sd'], 3)}")
    sa, sb = (300, 200) if smoke else (getattr(args, "steps_a", 20000),
                                       getattr(args, "steps_b", 15000))
    waves_off = (getattr(args, "waves", "off") or "off") == "off"
    pw = 1.0 if waves_off else 0.15
    log(f"model: decoder {kind}, wave elevations "
        f"{'NOT used' if waves_off else 'used (15% of samples without)'}")
    model, codes = train_decoder(D, kind=kind, steps=sa, log=log,
                                 p_wave_off=pw)
    model.waves_off = waves_off
    zst = train_posterior(model, D, codes, steps=sb, log=log, p_wave_off=pw)
    torch.save(dict(model=model.state_dict(), kind=kind, c_dim=model.c_dim,
                    zst=[z.cpu() for z in zst], stats=D.stats,
                    codes=codes.cpu(), waves_off=waves_off),
               _ckpt(cache, tag))
    log(f"saved {_ckpt(cache, tag)}")


def load(cache, tag, dev):
    ck = torch.load(_ckpt(cache, tag), weights_only=False)
    model = Model(ck["kind"], ck["c_dim"]).to(dev)
    model.load_state_dict(ck["model"])
    model.waves_off = ck.get("waves_off", False)
    model.eval()
    return model, [z.to(dev) for z in ck["zst"]], ck["stats"]


def crps_samples(s, y):
    """CRPS per element from samples s (B, S, ...) and truth y (B, ...)."""
    a = (s - y[:, None]).abs().mean(1)
    ss, _ = torch.sort(s, 1)
    S = s.shape[1]
    w = (2 * torch.arange(1, S + 1, device=s.device) - S - 1).view(
        1, S, *([1] * (s.dim() - 2)))
    b = (w * ss).sum(1) / (S * S)
    return a - b


def gauss_crps(sig, y):
    """CRPS of N(0, sig^2) at y (the climatological reference)."""
    z = y / sig
    pdf = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return sig * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


@torch.no_grad()
def evaluate(model, zst, D, Hs=(0, 8, 32, 64, 120, 240), batch=16,
             q_tail=None, oracle=None, gen_seed=0):
    """Rows per (episode, checkpoint, H), per horizon step and channel:
      own future (actual observed errors, noise included):
        e      the errors themselves (for the split's own climatology)
        se_*   squared error of the predictive mean (flow), zero, and
               persistence (mean of the last 4 errors)
        cov90  90% interval coverage (samples of e)
        crps   CRPS of the samples
        tail_p / tail_y  predicted probability / occurrence of |e| > q
      counterfactual (source splits: donor segments, the rule truth),
      per donor group: 'in' = donors 1..9 (the oracle code was fitted on
      them), 'held' = donors 10..12 (never fitted):
        cse_flow_* / cse_zero_* / cse_oracle_*  squared error of D's mean
               over code samples / zero / the test-time fitted code
        ccov_*  coverage of the rule truth by the code samples' spread"""
    dev = D.dev
    gen = torch.Generator(device=dev).manual_seed(gen_seed)
    rows = []
    M = D.REF.shape[2]
    q = torch.tensor(q_tail, device=dev) if q_tail is not None else None
    groups = dict(inn=torch.arange(1, N_FIT, device=dev),
                  held=torch.arange(N_FIT, M, device=dev))
    for s0 in range(0, len(D.pairs), batch):
        r = D.pairs[s0:s0 + batch]
        ii, cc = r[:, 0], r[:, 1]
        nb = len(ii)
        end = D.CK[ii, cc]
        ar = torch.arange(H, device=dev)[None]
        e_fut = D.E[ii[:, None], end[:, None] + ar]
        e_last = torch.stack([D.E[ii, (end - k).clamp(min=0)]
                              for k in range(1, 5)], 1).mean(1)
        for Hh in Hs:
            Hl = torch.minimum(torch.full_like(end, Hh), end)
            cs, dm, es = predict(model, zst, D, ii, cc, Hl, gen=gen)
            mean = es.mean(1)
            lo = torch.quantile(es, 0.05, dim=1)
            hi = torch.quantile(es, 0.95, dim=1)
            cov = ((e_fut >= lo) & (e_fut <= hi)).float()
            crps = crps_samples(es, e_fut)
            pers = e_last[:, None].expand_as(e_fut) if Hh > 0 else                 torch.zeros_like(e_fut)
            if q is not None:
                tp = (es.abs() > q).float().mean(1)
                ty = (e_fut.abs() > q).float()
            cf = {}
            if D.has_rule:
                for gname, mm in groups.items():
                    G = len(mm)
                    ib = ii.repeat_interleave(G)
                    cb = cc.repeat_interleave(G)
                    mb = mm.repeat(nb)
                    dd = decode_codes(model, D, ib, cb, mb,
                                      cs.repeat_interleave(G, 0))
                    y = D.Y[ib, cb, mb]
                    clo = torch.quantile(dd, 0.05, dim=1)
                    chi = torch.quantile(dd, 0.95, dim=1)
                    part = {f"cse_flow_{gname}": (dd.mean(1) - y) ** 2,
                            f"cse_zero_{gname}": y ** 2,
                            f"ccov_{gname}": ((y >= clo) & (y <= chi))
                            .float()}
                    if oracle is not None:
                        do = decode_codes(model, D, ib, cb, mb,
                                          oracle[ib, cb][:, None])[:, 0]
                        part[f"cse_oracle_{gname}"] = (do - y) ** 2
                    for k_, v in part.items():
                        cf[k_] = v.view(nb, G, H, C).mean(1)
            for b_ in range(nb):
                row = dict(ep=int(ii[b_]), ck=int(cc[b_]), H=int(Hh),
                           info=bool(D.info[int(ii[b_])]))
                for nm, v in (("e", e_fut), ("se_flow", (mean - e_fut) ** 2),
                              ("se_zero", e_fut ** 2),
                              ("se_pers", (pers - e_fut) ** 2),
                              ("cov90", cov), ("crps", crps)):
                    row[nm] = v[b_].cpu().numpy()          # (H, C)
                if q is not None:
                    row["tail_p"] = tp[b_].cpu().numpy()
                    row["tail_y"] = ty[b_].cpu().numpy()
                for k_, v in cf.items():
                    row[k_] = v[b_].cpu().numpy()
                rows.append(row)
    return rows


def _gauss_crps_np(sig, y):
    z = y / sig
    from math import erf, pi, sqrt
    cdf = 0.5 * (1 + np.vectorize(erf)(z / sqrt(2)))
    pdf = np.exp(-0.5 * z * z) / sqrt(2 * pi)
    return sig * (z * (2 * cdf - 1) + 2 * pdf - 1 / sqrt(pi))


def summarise(rows, Hs=(0, 8, 32, 64, 120, 240)):
    """Text table. Skill = MSE / zero MSE (1 = no correction, 0 = perfect),
    per channel, for the own future (all steps; and j = 0 / 1-4 / 5-19);
    90% coverage; CRPS skill against the SPLIT'S OWN climatology (a zero-mean
    Gaussian with the split's rms per channel), per channel; tail Brier
    skill; counterfactual skill on the held-out donors (flow, and the oracle
    code on held-out and on in-sample donors)."""
    out = []
    ch = "surge sway yaw heave pitch"
    E = np.array([r["e"] for r in rows if r["H"] == rows[0]["H"]])
    rms = np.sqrt((E ** 2).mean((0, 1))) + 1e-12            # (C,)
    for Hh in Hs:
        rs = [r for r in rows if r["H"] == Hh]
        if not rs:
            continue

        def S(k):
            return np.sum([r[k] for r in rs], 0)          # (H, C)
        z, f, p = S("se_zero"), S("se_flow"), S("se_pers")
        blk = [(0, 1), (1, 5), (5, H)]
        line = f"  H {Hh:>3}: own-future skill [{ch}] " + " ".join(
            f"{x:.2f}" for x in f.sum(0) / z.sum(0))
        line += " | j0/j1-4/j5-19 " + " / ".join(
            f"{(f[a:b].sum() / z[a:b].sum()):.2f}" for a, b in blk)
        line += f" | persist {p.sum() / z.sum():.2f}"
        line += f" | cov90 {np.mean([r['cov90'].mean() for r in rs]):.2f}"
        Ee = np.array([r["e"] for r in rs])
        c0 = _gauss_crps_np(rms, Ee).sum((0, 1))
        c1 = np.array([r["crps"] for r in rs]).sum((0, 1))
        line += " | CRPS skill " + " ".join(f"{x:.2f}" for x in c1 / c0)
        if "tail_p" in rs[0]:
            tp = np.array([r["tail_p"] for r in rs])
            ty = np.array([r["tail_y"] for r in rs])
            b0 = ((ty.mean() - ty) ** 2).mean()
            bs = (f"{((tp - ty) ** 2).mean() / b0:.2f}" if ty.sum() >= 20
                  else "n/a (too few)")
            line += (f" | tail rate {ty.mean():.3f} pred {tp.mean():.3f} "
                     f"Brier {bs}")
        if "cse_flow_held" in rs[0]:
            cz = S("cse_zero_held").sum(0)
            line += "\n         counterfactual (held-out donors): flow " \
                + " ".join(f"{x:.2f}" for x in S("cse_flow_held").sum(0) / cz)
            line += f" cov90 {np.mean([r['ccov_held'].mean() for r in rs]):.2f}"
            if "cse_oracle_held" in rs[0]:
                line += " | oracle code: held-out " + " ".join(
                    f"{x:.2f}" for x in S("cse_oracle_held").sum(0) / cz)
                line += ", in-sample " + " ".join(
                    f"{x:.2f}" for x in S("cse_oracle_inn").sum(0)
                    / S("cse_zero_inn").sum(0))
        out.append(line)
    return "\n".join(out)


def eval_all(cache, args, log=print):
    dev = device()
    tag = getattr(args, "tag", "") or ""
    model, zst, stats = load(cache, tag, dev)
    Dtr = Split(cache, "train", dev, stats=stats, targets=False)
    M = np.arange(Dtr.T)[None] < Dtr.len.cpu().numpy()[:, None]
    q = np.quantile(np.abs(Dtr.E.cpu().numpy()[M]), 0.99, axis=0)
    del Dtr
    torch.cuda.empty_cache()
    res = {}
    for split in ("A", "B", "C"):
        D = Split(cache, split, dev, stats=stats)
        oracle = fit_codes(model, D) if D.has_rule else None
        rows = evaluate(model, zst, D, q_tail=q, oracle=oracle)
        res[split] = rows
        log(f"eval {split} ({D.n} episodes):\n" + summarise(rows))
        del D
        torch.cuda.empty_cache()
    pickle.dump(res, open(os.path.join(cache, f"eval2{tag}.pkl"), "wb"))
