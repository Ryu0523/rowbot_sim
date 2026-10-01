#!/usr/bin/env python3
"""
Inferring the model's error from history: a learned basis, a transformer
history encoder and a flow-matching posterior.

What is inferred is the MPC's one-step error, per control step and per
channel (surge, sway, yaw, heave, pitch velocity; data.py's Y): 'what the
boat did minus what the model said'. That is also exactly the form in
which an MPC can use it -- a velocity correction added in every rollout
step -- and it is observable on the water, so the same quantity is learned
in the source and tested in the target.

    y_c(x) = phi_c(x) . z_c        c = 1..5 channels, phi: R^11 -> R^(5 x K)

  basis    phi is a network trained so that, across many random residual
           functions, each episode's errors are well fitted by SOME
           coefficients z (ridge regression, solved in closed form inside
           the loss). It replaces the hand-chosen features of learn/adapt/.
  encoder  a transformer reads the history tokens (data.py's O) and a CLS
           token summarises it; an empty history is just the CLS token.
  flow     conditional flow matching (rectified flow) over z: a velocity
           field v(z_tau, tau, context) transports N(0, I) to the posterior
           of the coefficients given the history. Trained on pairs
           (history window ending at a checkpoint, coefficients fitted to
           the function's TRUTH over the whole input box at that moment),
           windows from empty to the whole past. Box-wide targets (not a
           fit to the trajectory) are what make a steady history yield a
           posterior that stays broad about speeds and headings the boat
           never ran -- the first review found the trajectory-fit target
           taught a sharp posterior there. Sampling: Euler.

GPU if available (the laptop's 3080 Ti), memory capped per process.
"""
import math

import numpy as np
import torch
from torch import nn

C, K = 5, 16
D_X = 11          # basis inputs: the residual's 9 inputs + heave, pitch
D_TOK = 19        # encoder tokens (data.py O)
LAM = 0.1         # ridge regularisation (in normalised units)


def device():
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.2)   # <= ~3.2 GB VRAM
        return torch.device("cuda")
    return torch.device("cpu")


def mlp(n_in, hidden, n_out, act=nn.SiLU):
    layers, n = [], n_in
    for h in hidden:
        layers += [nn.Linear(n, h), act()]
        n = h
    layers.append(nn.Linear(n, n_out))
    return nn.Sequential(*layers)


class Basis(nn.Module):
    def __init__(self, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(D_X, hidden, C * K)

    def forward(self, x):                     # (..., D_X) -> (..., C, K)
        return self.net(x).view(*x.shape[:-1], C, K)


def ridge(Phi, Y, mask, lam=LAM):
    """Closed-form ridge per episode and channel.
    Phi (B, N, C, K), Y (B, N, C), mask (B, N) -> z (B, C, K)."""
    m = mask[..., None, None].float()
    P = Phi * m
    A = torch.einsum("bnck,bncj->bckj", P, P)
    A = A + lam * torch.eye(K, device=Phi.device)
    rhs = torch.einsum("bnck,bnc->bck", P, Y * mask[..., None].float())
    return torch.linalg.solve(A, rhs[..., None])[..., 0]


class Encoder(nn.Module):
    def __init__(self, d=128, layers=3, heads=4, t_max=512):
        super().__init__()
        self.inp = mlp(D_TOK, (d,), d)
        self.pos = nn.Parameter(torch.zeros(t_max + 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        lay = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(lay, layers)
        self.out = nn.LayerNorm(d)
        self.d = d

    def forward(self, tok, mask):
        """tok (B, T, D_TOK), mask (B, T) True = valid -> (B, d)."""
        B, T, _ = tok.shape
        h = self.inp(tok) + self.pos[1:T + 1]
        h = torch.cat([self.cls.expand(B, 1, self.d) + self.pos[:1], h], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool,
                                     device=tok.device), ~mask], 1)
        return self.out(self.tf(h, src_key_padding_mask=pad)[:, 0])


def t_embed(tau, n=32):
    f = torch.exp(torch.linspace(0, math.log(1000.0), n // 2,
                                 device=tau.device))
    a = tau[:, None] * f[None]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class Flow(nn.Module):
    def __init__(self, d_ctx=128, hidden=(512, 512, 512)):
        super().__init__()
        self.net = mlp(C * K + 32 + d_ctx, hidden, C * K)

    def forward(self, z, tau, ctx):
        return self.net(torch.cat([z, t_embed(tau), ctx], -1))

    @torch.no_grad()
    def sample(self, ctx, n, steps=24):
        """ctx (B, d) -> samples (B, n, C*K) of normalised coefficients."""
        B = ctx.shape[0]
        c = ctx.repeat_interleave(n, 0)
        z = torch.randn(B * n, C * K, device=ctx.device)
        for i in range(steps):
            tau = torch.full((B * n,), i / steps, device=ctx.device)
            z = z + self(z, tau, c) / steps
        return z.view(B, n, C * K)


# ------------------------------------------------------------- the data
N_PROBE = 128
N_CK = 3


class Episodes:
    """Padded tensors of a list of data.episode() results, normalised by
    the TRAINING y_std (pass it for test splits)."""

    def __init__(self, eps, dev, y_std=None):
        n_in = len(eps)
        eps = [e for e in eps if e["finite"] and len(e["Y"]) > 20]
        self.n_in, self.n_dropped = n_in, n_in - len(eps)
        T = max(len(e["Y"]) for e in eps)
        n = len(eps)
        self.n, self.T = n, T
        X = np.zeros((n, T, D_X), np.float32)
        O = np.zeros((n, T, D_TOK), np.float32)
        Y = np.zeros((n, T, C), np.float32)
        M = np.zeros((n, T), bool)
        P = np.zeros((n, N_PROBE, D_X), np.float32)
        YP = np.zeros((n, N_CK, N_PROBE, C), np.float32)
        CK = np.zeros((n, N_CK), np.int64)
        HP = np.zeros((n, N_CK), bool)           # has probe truth
        for i, e in enumerate(eps):
            L = len(e["Y"])
            X[i, :L, :9] = e["X"]
            X[i, :L, 9:] = e["O"][:, 9:11]
            O[i, :L] = e["O"]
            Y[i, :L] = e["Y"]
            M[i, :L] = True
            if e.get("P") is not None and len(e["ck"]):
                k = min(len(e["ck"]), N_CK)
                P[i] = e["P"]
                YP[i, :k] = e["YP"][:k]
                CK[i, :k] = e["ck"][:k]
                HP[i, :k] = True
        self.y_std = (Y[M].std(0) + 1e-6) if y_std is None else y_std
        self.y_rms_raw = np.sqrt((Y[M] ** 2).mean(0))
        O[..., -C:] = O[..., -C:] / self.y_std        # tokens: normalised Y
        self.X = torch.tensor(X, device=dev)
        self.O = torch.tensor(O, device=dev)
        self.Y = torch.tensor(Y / self.y_std, device=dev)
        self.M = torch.tensor(M, device=dev)
        self.P = torch.tensor(P, device=dev)
        self.YP = torch.tensor(YP / self.y_std, device=dev)
        self.CK = torch.tensor(CK, device=dev)
        self.HP = torch.tensor(HP, device=dev)
        self.len = self.M.sum(1)
        self.info = np.array([e["informative"] for e in eps])
        self.style = [e["style"] for e in eps]


def memory_ok(style):
    """A hysteresis function whose relay really carried memory here: it
    switched at least twice, spent 10-90% of the time on, and dwelt longer
    than a few wave encounters between switches."""
    return (style is not None and style.get("hysteresis")
            and style.get("toggles", 0) >= 2
            and 0.1 < style.get("on_frac", 0) < 0.9
            and style.get("dwell_s", 0) > 3.0)


# ------------------------------------------------------------- training
def train_basis(D, steps=3000, batch=64, lr=1e-3, log=print):
    """Across episodes, fit coefficients on part of the data and score on
    the rest, in two ways at once: along the trajectory (fit on 5-50% of
    the steps) and over the input box (fit on 25-75% of one checkpoint's
    probes) -- the basis must span functions everywhere, not only where
    boats happened to go."""
    dev = D.X.device
    basis = Basis().to(dev)
    opt = torch.optim.Adam(basis.parameters(), lr)
    g = torch.Generator(device="cpu").manual_seed(0)
    for it in range(steps):
        idx = torch.randint(0, D.n, (batch,), generator=g).to(dev)
        X, Y, M = D.X[idx], D.Y[idx], D.M[idx]
        frac = 0.05 + 0.45 * torch.rand(batch, 1, device=dev)
        fit = (torch.rand(M.shape, device=dev) < frac) & M
        ev = M & ~fit
        Phi = basis(X)
        z = ridge(Phi, Y, fit)
        pred = torch.einsum("bnck,bck->bnc", Phi, z)
        loss = (((pred - Y) ** 2).mean(-1) * ev).sum() / ev.sum()
        k = torch.randint(0, N_CK, (batch,), generator=g).to(dev)
        hp = D.HP[idx, k]
        if hp.any():
            PX, PY = D.P[idx][hp], D.YP[idx, k][hp]
            Pp = basis(PX)
            f2 = 0.25 + 0.5 * torch.rand(len(PX), 1, device=dev)
            pf = torch.rand(PX.shape[:2], device=dev) < f2
            zp = ridge(Pp, PY, pf)
            pp = torch.einsum("bnck,bck->bnc", Pp, zp)
            loss = loss + ((((pp - PY) ** 2).mean(-1) * ~pf).sum()
                           / (~pf).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == steps - 1:
            log(f"    basis {it:5d}: held-out loss {loss.item():.4f}")
    return basis


@torch.no_grad()
def probe_targets(basis, D, lam=LAM):
    """Coefficients fitted to each checkpoint's probe truth over the whole
    input box: the function as it stood then, not as the trajectory
    happened to sample it. (n, N_CK, C*K); rows without truth are 0."""
    out = torch.zeros(D.n, N_CK, C * K, device=D.X.device)
    for i in range(0, D.n, 128):
        sl = slice(i, i + 128)
        Pp = basis(D.P[sl])
        for k in range(N_CK):
            m = torch.ones(Pp.shape[:2], dtype=torch.bool, device=D.X.device)
            out[sl, k] = ridge(Pp, D.YP[sl, k], m, lam).flatten(1)
    return out


def history_mask(D, idx, end, H):
    ar = torch.arange(D.T, device=D.X.device)[None]
    return (ar < end[:, None]) & (ar >= (end - H)[:, None])


def noisy_history(O, gen, lo=0.01, hi=3.0):
    """Augmentation: the history's error channels plus noise of random
    strength (log-uniform, per episode, spread per channel) and random time
    correlation (AR(1), rho 0-0.95) -- errors that no function of the
    boat's state can predict, as the real gap has (split C: ~30% of it
    even the future window's own fit cannot explain). The targets stay the
    clean function, so the network must learn to tell structure from noise
    rather than read every wiggle as a function."""
    B, T, _ = O.shape
    dev = O.device
    sig = torch.exp(torch.empty(B, 1, 1).uniform_(np.log(lo), np.log(hi),
                                                  generator=gen)).to(dev)
    sig = sig * torch.exp(0.5 * torch.randn(B, 1, C, generator=gen)).to(dev)
    rho = torch.empty(B, 1).uniform_(0.0, 0.95, generator=gen).to(dev)
    w = torch.randn(B, T, C, generator=gen).to(dev)
    e = torch.empty_like(w)
    e[:, 0] = w[:, 0]
    a = torch.sqrt(1 - rho ** 2)
    for t in range(1, T):
        e[:, t] = rho * e[:, t - 1] + a * w[:, t]
    O = O.clone()
    O[..., -C:] = O[..., -C:] + sig * e
    return O


def train_flow(D, Zk, steps=6000, batch=128, lr=3e-4, log=print,
               augment=False):
    """Pairs (history window ending at a checkpoint, coefficients of the
    function at that checkpoint). Windows from empty to the whole past; a
    quarter of them 0-16 steps, so a short history must give a broad
    posterior. Rectified flow over the normalised coefficients."""
    dev = D.X.device
    valid = D.HP
    Zv = Zk[valid]
    z_mu, z_sd = Zv.mean(0), Zv.std(0) + 1e-6
    enc, flow = Encoder().to(dev), Flow().to(dev)
    params = list(enc.parameters()) + list(flow.parameters())
    opt = torch.optim.AdamW(params, lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device="cpu").manual_seed(1)
    rows = torch.nonzero(valid)                           # (episode, k)
    for it in range(steps):
        r = rows[torch.randint(0, len(rows), (batch,), generator=g).to(dev)]
        idx, k = r[:, 0], r[:, 1]
        end = D.CK[idx, k]
        u = torch.rand(batch, generator=g).to(dev)
        short = torch.rand(batch, generator=g).to(dev) < 0.25
        H = torch.where(short, (u * 17).long(), (u * (end + 1)).long())
        H = torch.minimum(H, end)
        O = D.O[idx]
        if augment and bool(torch.rand(1, generator=g) < 0.75):
            O = noisy_history(O, g)
        ctx = enc(O, history_mask(D, idx, end, H))
        z1 = (Zk[idx, k] - z_mu) / z_sd
        z0 = torch.randn_like(z1)
        tau = torch.rand(batch, device=dev)
        zt = (1 - tau[:, None]) * z0 + tau[:, None] * z1
        loss = ((flow(zt, tau, ctx) - (z1 - z0)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == steps - 1:
            log(f"    flow {it:5d}: loss {loss.item():.4f}")
    return enc, flow, (z_mu, z_sd)


# ------------------------------------------------ a Gaussian baseline
class GaussBLR:
    """Bayesian linear regression on the SAME learned basis with a Gaussian
    prior fitted to the training functions' coefficients (mean and full
    covariance) and per-channel noise from the training data: what the
    flow must beat to show that a non-Gaussian posterior earns its keep."""

    def __init__(self, Zk, valid, noise):
        Zv = Zk[valid]
        self.mu = Zv.mean(0)
        S = torch.cov(Zv.T) + 1e-4 * torch.eye(Zv.shape[1], device=Zv.device)
        self.P0 = torch.linalg.inv(S)
        # floored: a channel that is rarely active in training would
        # otherwise get ~zero noise and an ill-conditioned, exploding fit
        self.noise = noise.clamp(min=1e-2)                # (C,), normalised

    @torch.no_grad()
    def posterior(self, Phi, Y, mask):
        """Phi (B, N, C, K), Y (B, N, C), mask (B, N) -> mean, cov."""
        B = Phi.shape[0]
        m = mask[..., None, None].float()
        Pw = Phi * m / self.noise.sqrt()[None, None, :, None]
        A = torch.zeros(B, C * K, C * K, device=Phi.device)
        rhs = torch.zeros(B, C * K, device=Phi.device)
        for c in range(C):
            sl = slice(c * K, (c + 1) * K)
            A[:, sl, sl] = torch.einsum("bnk,bnj->bkj", Pw[:, :, c],
                                        Pw[:, :, c])
            rhs[:, sl] = torch.einsum("bnk,bn->bk", Pw[:, :, c],
                                      Y[:, :, c] * mask.float()
                                      / self.noise[c].sqrt())
        Prec = A + self.P0
        cov = torch.linalg.inv(Prec)
        cov = 0.5 * (cov + cov.transpose(1, 2))
        mean = torch.einsum("bij,bj->bi", cov, rhs + self.P0 @ self.mu)
        return mean, cov

    def sample(self, mean, cov, n):
        L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(
            cov.shape[-1], device=cov.device))
        e = torch.randn(mean.shape[0], n, mean.shape[1], device=mean.device)
        return mean[:, None] + torch.einsum("bij,bsj->bsi", L, e)


@torch.no_grad()
def traj_noise(basis, D):
    """Per-channel variance of the trajectory errors around the nearest
    checkpoint's probe fit (the GaussBLR's observation noise)."""
    Zk = probe_targets(basis, D)
    Phi_all, res = [], []
    for i in range(0, D.n, 128):
        sl = slice(i, i + 128)
        Phi = basis(D.X[sl])
        z = Zk[sl, 1].view(-1, C, K)
        pred = torch.einsum("bnck,bck->bnc", Phi, z)
        e = (pred - D.Y[sl]) ** 2 * D.M[sl][..., None]
        res.append(e.sum((0, 1)))
    return torch.stack(res).sum(0) / D.M.sum()


# ------------------------------------------------------------ evaluation
def _interval(ps, lo_q=0.05, hi_q=0.95):
    """Order-statistic interval of S samples along dim 1."""
    s, _ = torch.sort(ps, dim=1)
    S = s.shape[1]
    return s[:, int(lo_q * S)], s[:, int(np.ceil(hi_q * S)) - 1]


@torch.no_grad()
def evaluate_probes(basis, enc, flow, zst, blr, D, Hs=(0, 8, 32, 64, 120),
                    n_samp=256):
    """The function over the whole input box, after a history window of H
    steps ending at a checkpoint. Per episode, checkpoint and H: squared
    error per channel at the probes for each method (summed so skill can be
    formed per split and subset), the 90% coverage of the probe truth and
    the interval width -- near (first half of the probes) and box
    (second half) separately."""
    dev = D.X.device
    z_mu, z_sd = zst
    Zk = probe_targets(basis, D)
    rows = []
    for k in range(N_CK):
        has = torch.nonzero(D.HP[:, k])[:, 0]
        for H in Hs:
            for i in range(0, len(has), 32):
                b = has[i:i + 32]
                end = D.CK[b, k]
                Hb = torch.minimum(torch.full_like(end, H), end)
                hm = history_mask(D, b, end, Hb)
                PhiT = basis(D.X[b])
                PhiP = basis(D.P[b])
                truth = D.YP[b, k]
                preds = {"zero": torch.zeros_like(truth)}
                zr = ridge(PhiT, D.Y[b], hm)
                preds["ridge"] = torch.einsum("bnck,bck->bnc", PhiP, zr)
                mean, cov = blr.posterior(PhiT, D.Y[b], hm)
                zs_b = blr.sample(mean, cov, n_samp).view(len(b), n_samp,
                                                         C, K)
                ps_b = torch.einsum("bnck,bsck->bsnc", PhiP, zs_b)
                preds["blr"] = torch.einsum("bnck,bck->bnc", PhiP,
                                            mean.view(-1, C, K))
                ctx = enc(D.O[b], hm)
                zs = flow.sample(ctx, n_samp) * z_sd + z_mu
                ps = torch.einsum("bnck,bsck->bsnc", PhiP,
                                  zs.view(len(b), n_samp, C, K))
                preds["flow"] = ps.mean(1)
                preds["oracle"] = torch.einsum(
                    "bnck,bck->bnc", PhiP, Zk[b, k].view(-1, C, K))
                cov_w = {}
                for nm, s in (("flow", ps), ("blr", ps_b)):
                    lo, hi = _interval(s)
                    inside = ((truth >= lo) & (truth <= hi)).float()
                    cov_w[nm] = (inside, (hi - lo))
                half = N_PROBE // 2
                for j, e_i in enumerate(b.tolist()):
                    r = dict(ep=e_i, k=k, H=H, info=bool(D.info[e_i]),
                             mem=memory_ok(D.style[e_i]))
                    for region, sl in (("near", slice(0, half)),
                                       ("box", slice(half, None))):
                        for nm, pr in preds.items():
                            r[f"se_{nm}_{region}"] = ((pr[j, sl] - truth[j, sl])
                                                      ** 2).sum(0).cpu().numpy()
                        for nm, (ins, w) in cov_w.items():
                            r[f"cov_{nm}_{region}"] = float(ins[j, sl].mean())
                            r[f"wid_{nm}_{region}"] = float(w[j, sl].mean())
                    rows.append(r)
    return rows


@torch.no_grad()
def evaluate_traj(basis, enc, flow, zst, blr, D, Ls=(0, 8, 32, 128, 240),
                  horizon=40, n_samp=64):
    """The next `horizon` steps' observed errors after a prefix of L steps.
    Per episode: squared error per channel summed over the window, for
    zero, persistence (last 4 / 16 steps, prefix mean), ridge on the prefix
    and on its last 40 steps, the Gaussian baseline, the flow's posterior
    mean, and the window's own fit (an in-sample ceiling)."""
    dev = D.X.device
    z_mu, z_sd = zst
    ar = torch.arange(D.T, device=dev)[None]
    rows = []
    for L in Ls:
        ok = torch.nonzero(D.len >= L + horizon)[:, 0]
        for i in range(0, len(ok), 64):
            b = ok[i:i + 64]
            X, Y = D.X[b], D.Y[b]
            pre = (ar < L).expand(len(b), -1)
            fut = (ar >= L) & (ar < L + horizon)
            fut = fut.expand(len(b), -1)
            Phi = basis(X)
            P = {"zero": torch.zeros_like(Y)}
            for nm, n_last in (("last4", 4), ("last16", 16), ("mean", 10 ** 6)):
                w = (ar >= L - n_last) & (ar < L)
                w = w.expand(len(b), -1)
                cnt = w.sum(1).clamp(min=1)[:, None]
                mu = (Y * w[..., None]).sum(1) / cnt
                if L == 0:
                    mu = torch.zeros_like(mu)
                P[nm] = mu[:, None, :].expand_as(Y)
            zr = ridge(Phi, Y, pre)
            P["ridge"] = torch.einsum("bnck,bck->bnc", Phi, zr)
            w40 = ((ar >= L - 40) & (ar < L)).expand(len(b), -1)
            P["ridge40"] = torch.einsum("bnck,bck->bnc", Phi,
                                        ridge(Phi, Y, w40))
            mean, _ = blr.posterior(Phi, Y, pre)
            P["blr"] = torch.einsum("bnck,bck->bnc", Phi, mean.view(-1, C, K))
            ctx = enc(D.O[b], pre)
            zs = flow.sample(ctx, n_samp) * z_sd + z_mu
            P["flow"] = torch.einsum("bnck,bsck->bsnc", Phi,
                                     zs.view(len(b), n_samp, C, K)).mean(1)
            P["window"] = torch.einsum("bnck,bck->bnc", Phi,
                                       ridge(Phi, Y, fut))
            for j, e_i in enumerate(b.tolist()):
                r = dict(ep=e_i, L=L, info=bool(D.info[e_i]),
                         mem=memory_ok(D.style[e_i]))
                f = fut[j]
                for nm, pr in P.items():
                    r[f"se_{nm}"] = ((pr[j, f] - Y[j, f]) ** 2).sum(0) \
                        .cpu().numpy()
                rows.append(r)
    return rows
