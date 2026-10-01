#!/usr/bin/env python3
"""
The one-step error model (DEFECTS M7): learn

    p( e0_t | everything recorded before step t, the state x_t, the command
         applied during step t )

and get multi-step predictions by rolling the MPC's own reduced model
forward, sampling e0 step by step (the chain rule; the physics model, not
the network, carries the error into the state). e0 is the error against the
model the MPC rolls out (no waves, ideal actuators; data3.py), 7 channels.

Only the boat's own states, the commands and the error history are inputs
(D6). No latent 'rules' code: the conditional distribution is the target
(PFN argument: trained on draws from the prior, the network approximates
the posterior predictive, whatever the causes).

  net     a causal transformer over step tokens
          [state x_t (11), command u_t (2), asinh(e_{t-1}) (10), first flag];
          the output at t may use tokens <= t only
  head    a conditional flow (rectified flow matching) over the 10-dim e_t
          in asinh space, conditioned on the transformer output at t

Recorded commands are fine for one-step training: in the recorded episodes
each command depends only on what was observable at that time, which is in
the conditioning, so p(e0_t | history, u_t) is causal. (Multi-step
conditioning on recorded commands would not be: later commands reacted to
later errors.)

Rollout-robust fine-tuning (DEFECTS M9): rollout_core is the ONE rollout
code path (KV cache, per-row moments, gradients optional; Model0T is the
MPC's model in torch), used by train_rollout (V0: the FM loss on recorded
and wide-plan branch windows; V1: plus an energy score of the model's own
24-step rollouts) and by eval_multi_step.
"""
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

C7, D_X, D_U = 10, 11, 2       # C7: all error channels (data3.N_E)
D_TOK = D_X + D_U + C7 + 1
W_CTX = 256
HB = 24
KAPPA = 0.5                       # asinh scale of the targets


def device(frac=0.35):
    """The GPU with a per-process memory cap (the rollout-training phases
    raise it to 0.5), else the CPU."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac)
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
class Data3:
    """A packed split with its e0, on the device. Inputs normalised by the
    library's mu / sd (the operator's 28 inputs: S's first 11, then U),
    e0 by the TRAINING per-channel sd (pass `stats`)."""

    def __init__(self, cache, split, dev, stats=None):
        d = np.load(os.path.join(cache, f"{split}.npz"))
        e0 = np.load(os.path.join(cache, f"{split}_e0.npz"))["E0"]
        lib = np.load(os.path.join(cache, "lib.npz"))
        mu, sd = lib["mu"].astype(np.float32), lib["sd"].astype(np.float32)
        n, T = d["U"].shape[:2]
        if e0.shape[:2] != (n, T):      # an e0 left over from another pack
            raise RuntimeError(f"{split}_e0.npz {e0.shape[:2]} does not match "
                               f"{split}.npz {(n, T)}; rerun step 3's e0")
        L = d["len"]
        # e0 exists for t < len - 1 (packed splits: the last step has no
        # next state) or t < len (block splits store the final state)
        n_e0 = L - 1 if d["XS"].shape[1] == T else L
        valid = np.arange(T)[None] < n_e0[:, None]
        if stats is None:
            stats = dict(e_sd=e0[valid].std(0).astype(np.float32) + 1e-6,
                         x_mu=mu[:D_X], x_sd=sd[:D_X], u_mu=mu[26:28],
                         u_sd=sd[26:28])
        self.stats = stats
        X = (d["S"][..., :D_X] - stats["x_mu"]) / stats["x_sd"]
        U = (d["U"] - stats["u_mu"]) / stats["u_sd"]
        self.X = torch.tensor(X * (np.arange(T)[None] < L[:, None])[..., None],
                              dtype=torch.float32, device=dev)
        self.U = torch.tensor(U, dtype=torch.float32, device=dev)
        self.E = torch.tensor(e0 / stats["e_sd"] * valid[..., None],
                              dtype=torch.float32, device=dev)
        self.valid = torch.tensor(valid, device=dev)
        self.len = torch.tensor(L, device=dev)
        self.n, self.T = n, T
        self.XS = d["XS"]
        self.Uraw = d["U"]
        self.dev = dev


def make_tokens(X, U, E_prev, first):
    """(B, L, D_TOK) from states, commands, previous errors (normalised)."""
    return torch.cat([X, U, torch.asinh(E_prev / KAPPA), first[..., None]],
                     -1)


def window_tokens(D, ii, a, L, obs_sd=None, gen=None):
    """Tokens of windows [a, a + L) of episodes ii; the previous error of
    the first token is e0_{a-1} (0 at an episode's start)."""
    ar = torch.arange(L, device=D.dev)[None]
    t = a[:, None] + ar
    tc = t.clamp(max=D.T - 1)
    X, U = D.X[ii[:, None], tc], D.U[ii[:, None], tc]
    tp = (t - 1).clamp(min=0, max=D.T - 1)
    Ep = D.E[ii[:, None], tp] * (t >= 1)[..., None]
    if obs_sd is not None:
        # drawn on the generator's device (the CPU for train(), as before)
        Ep = Ep + obs_sd[:, None, :] * torch.randn(
            Ep.shape, generator=gen,
            device=gen.device if gen is not None else "cpu").to(D.dev)
    first = (ar == 0).float().expand_as(t)
    tgt = D.E[ii[:, None], tc]
    ok = D.valid[ii[:, None], tc] & (t < D.T)
    return make_tokens(X, U, Ep, first), tgt, ok


# ----------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, d=192, layers=6, heads=6):
        super().__init__()
        self.inp = mlp(D_TOK, (d,), d)
        self.pos = nn.Parameter(torch.zeros(W_CTX, d))
        lay = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(lay, layers,
                                        enable_nested_tensor=False)
        self.out = nn.LayerNorm(d)
        self.head = mlp(C7 + 32 + d, (512, 512, 512), C7)
        self.d = d

    def encode(self, tok):
        """Causal: output at t uses tokens <= t. tok (B, L <= W_CTX, D_TOK)."""
        B, L, _ = tok.shape
        h = self.inp(tok) + self.pos[:L][None]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=tok.device),
                          1)
        return self.out(self.tf(h, mask=mask, is_causal=True))

    def velocity(self, y, tau, h):
        return self.head(torch.cat([y, t_embed(tau), h], -1))

    @torch.no_grad()
    def sample(self, h, n, steps=24, gen=None):
        """h (B, d) -> samples of e0 (B, n, C7), normalised units."""
        B = h.shape[0]
        hh = h.repeat_interleave(n, 0)
        y = torch.randn(B * n, C7, device=h.device, generator=gen)
        for i in range(steps):
            tau = torch.full((B * n,), i / steps, device=h.device)
            y = y + self.velocity(y, tau, hh) / steps
        return (KAPPA * torch.sinh(y)).view(B, n, C7)

    def sample_grad(self, h, y0, steps=24):
        """sample() with explicit base noise y0 (..., C7) at conditions h
        (..., d), differentiable (the same Euler grid tau = i / steps), for
        rollout training and common random numbers. The head's first layer
        acts on cat([y, t_embed(tau), h]), so its h and tau parts are
        computed once per call instead of once per Euler step (the same
        function up to rounding; ~a third fewer kernels in the 576 head
        calls of a 24-step rollout)."""
        lin0 = self.head[0]
        W = lin0.weight
        a_h = F.linear(h, W[:, C7 + 32:], lin0.bias)
        taus = torch.arange(steps, device=h.device, dtype=h.dtype) / steps
        a_t = F.linear(t_embed(taus), W[:, C7:C7 + 32])     # (steps, 512)
        y = y0
        for i in range(steps):
            v = self.head[1:](F.linear(y, W[:, :C7]) + a_h + a_t[i])
            y = y + v / steps
        return KAPPA * torch.sinh(y)


# -------------------------------------------------------------- training
def obs_sd(B, dev, gen=None):
    """Observation noise on the history's errors: per window, white, sd
    LogU[0.01, 0.1] of the error's normalised scale, jittered per channel."""
    u = torch.rand(B, 1, generator=gen).to(dev)
    base = torch.exp(math.log(0.01) + u * (math.log(0.1) - math.log(0.01)))
    return base * torch.exp(0.3 * torch.randn(B, C7, generator=gen).to(dev))


def val_episodes(n, seed=0):
    """The episodes train() keeps out of training (its validation set), by
    exactly its draw, so the later stages (branches, train_rollout) hold
    out the same episodes."""
    gv = torch.Generator(device="cpu").manual_seed(seed + 5)
    return torch.randperm(n, generator=gv)[:max(2, n // 33)]


def split_pools(n, seed=0):
    """(training episodes, validation episodes) as index tensors."""
    is_val = torch.zeros(n, dtype=torch.bool)
    is_val[val_episodes(n, seed)] = True
    return torch.nonzero(~is_val)[:, 0], torch.nonzero(is_val)[:, 0]


def fm_draw(pool, b, gen, Lmax, L, dev):
    """b recorded windows of length L from the episodes in pool: random
    starts, a quarter of them at the episode's start."""
    ii = pool[torch.randint(0, len(pool), (b,), generator=gen)]
    span = (Lmax[ii] - 1 - L).clamp(min=0)
    a = (torch.rand(b, generator=gen) * (span + 1).float()).long()
    a = torch.where(torch.rand(b, generator=gen) < 0.25,
                    torch.zeros_like(a), a)
    return ii.to(dev), a.to(dev)


def fm_loss_h(net, h, tgt, ok, gen):
    """Flow-matching loss of the head at transformer outputs h (B, L, d)
    against targets tgt (B, L, C7) where ok."""
    y1 = torch.asinh(tgt / KAPPA)
    y0 = torch.randn(y1.shape, generator=gen).to(h.device)
    tau = torch.rand(y1.shape[:2], generator=gen).to(h.device)
    yt = (1 - tau[..., None]) * y0 + tau[..., None] * y1
    v = net.velocity(yt, tau, h)
    se = ((v - (y1 - y0)) ** 2).mean(-1)
    return (se * ok).sum() / ok.sum().clamp(min=1)


def fm_loss(net, D, ii, a, L, gen, noise=True):
    """The teacher-forced FM loss on recorded windows [a, a + L)."""
    tok, tgt, ok = window_tokens(D, ii, a, L, obs_sd(len(ii), D.dev, gen)
                                 if noise else None, gen)
    return fm_loss_h(net, net.encode(tok), tgt, ok, gen)


@torch.no_grad()
def fm_validate(net, D, va, L, Lmax):
    """FM loss on 8 x 32 recorded windows of the validation episodes, with a
    fixed stream (the same windows and noise at every check)."""
    gen = torch.Generator(device="cpu").manual_seed(4321)
    ls = []
    for _ in range(8):
        ii, a = fm_draw(va, 32, gen, Lmax, L, D.dev)
        ls.append(fm_loss(net, D, ii, a, L, gen).item())
    return float(np.mean(ls))


def train(D, steps=20000, batch=48, L=W_CTX, lr=3e-4, log=print, seed=0,
          patience=3, net=None, pools=None, resume=None):
    """Teacher-forced FM training on D's recorded windows. net: start from
    this network (fine-tuning) instead of a fresh one; pools: (training,
    validation) episode index tensors instead of split_pools(D.n); resume:
    a file path to save the training state to every 1000 steps and to
    continue from when it exists (learn/meta/train_resume.py)."""
    dev = D.dev
    torch.manual_seed(seed)
    if net is None:
        net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    # warm-up >= 2 steps (OneCycleLR divides by zero below that; only
    # short test runs, < 40 steps, are affected)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=max(0.05, 2.0 / steps))
    g = torch.Generator(device="cpu").manual_seed(seed)
    tr, va = split_pools(D.n, seed) if pools is None else pools
    Lmax = D.len.cpu()

    best = dict(v=float("inf"), it=-1, bad=0)
    from learn.meta.train_resume import Resume
    rs = Resume(resume, dict(kind="model3", steps=steps, batch=batch, L=L,
                             lr=lr, seed=seed, n=int(D.n)), log=log)
    start = rs.load(net, opt, sched, g, best)
    t0 = time.time()
    it = start - 1
    for it in range(start, steps):
        ii, a = fm_draw(tr, batch, g, Lmax, L, dev)
        loss = fm_loss(net, D, ii, a, L, g)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == steps - 1:
            v = fm_validate(net, D, va, L, Lmax)
            log(f"    one-step model {it:6d}: train {loss.item():.4f}  "
                f"episodes kept out of training {v:.4f}  "
                f"({time.time() - t0:.0f} s)")
            if it > steps // 5:
                if v < best["v"]:
                    best.update(v=v, it=it, bad=0, state={
                        k_: x.detach().clone() for k_, x in
                        net.state_dict().items()})
                else:
                    best["bad"] += 1
                    if best["bad"] >= patience:
                        log(f"    not better for {patience} checks, "
                            f"stopping at step {it}")
                        break
        rs.save(it, net, opt, sched, g, best)
    if best["it"] >= 0 and best["it"] != it:
        net.load_state_dict(best["state"])
        log(f"    restored step {best['it']} ({best['v']:.4f})")
    rs.done()
    return net


# ------------------------------------------------------------ evaluation
def crps_samples(s, y):
    """Fair (unbiased in the sample count) CRPS from samples s (B, S, ...)
    at truth y (B, ...)."""
    a = (s - y[:, None]).abs().mean(1)
    ss, _ = torch.sort(s, 1)
    S = s.shape[1]
    w = (2 * torch.arange(1, S + 1, device=s.device) - S - 1).view(
        1, S, *([1] * (s.dim() - 2)))
    return a - (w * ss).sum(1) / (S * (S - 1))


def gauss_crps_np(sig, y):
    from math import erf, pi, sqrt
    z = y / sig
    cdf = 0.5 * (1 + np.vectorize(erf)(z / sqrt(2)))
    pdf = np.exp(-0.5 * z * z) / sqrt(2 * pi)
    return sig * (z * (2 * cdf - 1) + 2 * pdf - 1 / sqrt(pi))


class Tally:
    """Sums per channel (and per horizon step) for the summary table."""

    def __init__(self, shape):
        z = lambda: np.zeros(shape)   # noqa: E731
        self.se, self.se0, self.cov, self.crps, self.n = z(), z(), z(), z(), 0
        self.ys = []
        self.tail_p, self.tail_y = [], []

    def add(self, s, y, q=None, ref=None):
        """s (B, S, ...) samples, y (B, ...) truth; ref (B, ...) the
        reference prediction the skill is relative to (default 0).
        Coverage is the randomised PIT's share inside [0.05, 0.95]: exactly
        0.90 for a calibrated forecaster at any sample count (quantiles of
        few samples cover less). The squared error of the sample mean has
        the sample-mean variance removed (unbiased in the sample count)."""
        S = s.shape[1]
        m = s.mean(1)
        ref = torch.zeros_like(y) if ref is None else ref
        self.se += ((m - y) ** 2 - s.var(1, unbiased=True) / S).sum(
            0).cpu().numpy()
        self.se0 += ((ref - y) ** 2).sum(0).cpu().numpy()
        r = (s < y.unsqueeze(1)).sum(1).float()
        pit = (r + torch.rand_like(r)) / (S + 1)
        self.cov += ((pit >= 0.05) & (pit <= 0.95)).float().sum(
            0).cpu().numpy()
        self.crps += crps_samples(s, y).sum(0).cpu().numpy()
        self.n += y.shape[0]
        self.ys.append(y.cpu().numpy())
        if q is not None:
            self.tail_p.append((s.abs() > q).float().mean(1).cpu().numpy())
            self.tail_y.append((y.abs() > q).float().cpu().numpy())

    def summary(self, chans=5):
        Y = np.concatenate(self.ys)
        rms = np.sqrt((Y ** 2).mean(0)) + 1e-12
        c0 = gauss_crps_np(rms, Y).sum(0)
        out = dict(skill=self.se / np.maximum(self.se0, 1e-12),
                   cov=self.cov / self.n, crps=self.crps / c0)
        if self.tail_p:
            tp, ty = np.concatenate(self.tail_p), np.concatenate(self.tail_y)
            b0 = ((ty.mean(0) - ty) ** 2).mean(0)
            out["tail"] = (ty.mean(0), tp.mean(0),
                           ((tp - ty) ** 2).mean(0) / np.maximum(b0, 1e-12),
                           ty.sum(0))
        return out


@torch.no_grad()
def eval_one_step(net, D, q, n_samp=64, start=40, batch=16, gen_seed=0):
    """Every step t >= start of every episode, with the recorded history
    before it (up to W_CTX steps)."""
    dev = D.dev
    gen = torch.Generator(device=dev).manual_seed(gen_seed)
    tal = Tally((C7,))
    L = W_CTX
    for s0 in range(0, D.n, batch):
        ii = torch.arange(s0, min(D.n, s0 + batch), device=dev)
        Tn = int(D.len[ii].max())
        # window 1: [0, L) scores start..L-1; window 2: [Tn - L, Tn) scores
        # L..Tn-1 (with >= Tn - 2L + L history)
        for a0, lo, hi in ((0, start, L), (max(0, Tn - L), L, Tn)):
            if hi <= lo:
                continue
            a = torch.full_like(ii, a0)
            tok, tgt, ok = window_tokens(D, ii, a, L)
            h = net.encode(tok)
            pos = torch.arange(L, device=dev) + a0
            sel = ok & (pos[None] >= lo) & (pos[None] < hi)
            hs, ys = h[sel], tgt[sel]
            for c0 in range(0, len(hs), 2048):
                s = net.sample(hs[c0:c0 + 2048], n_samp, gen=gen)
                tal.add(s, ys[c0:c0 + 2048], q)
    return tal.summary()


def state_tokens(st10, act, U_now, stats, env):
    """Normalised state inputs (N, 11) from reduced states (N, 10) and the
    actuator fractions (N, 2) -- as residuals.inputs_of + zr."""
    p = env["red"].p
    x, y, u, z, zd, th, thd, psi, r, v = st10.T
    X = np.stack([u / env["u_ref"] - 1.0, v, r * env["L"] / env["u_ref"],
                  np.cos(psi), np.sin(psi), act[:, 0], act[:, 1], zd, thd,
                  (z - p.get("z0", 0.0)) / 0.2,
                  (th - p.get("th0", 0.0)) / 0.05], 1)
    return (X - stats["x_mu"]) / stats["x_sd"]


# ------------------------------------------- torch rollout (KV cache)
SC_ES = (2, 9, 8, 3, 5, 7)       # reduced columns u, v, r, z, th, psi
# the error channel behind each ES state feature (u, v, r, z, th, psi, act
# thrust, act nozzle), for the scale floor
ES_E_CH = (0, 1, 2, 5, 6, 7, 8, 9)
ES_SCALE_FLOOR = 0.1


def plant_to_reduced_t(xs14):
    """data2.plant_to_reduced for tensors: (..., 14) -> (..., 10)."""
    return xs14[..., [0, 1, 6, 2, 8, 4, 10, 5, 11, 7]]


def state_tokens_t(st10, act, stats, env):
    """state_tokens for tensors (same formula, same stats): reduced states
    (..., 10) and actuator fractions (..., 2), float64 -> normalised
    inputs (..., 11), float64."""
    p = env["red"].p
    x, y, u, z, zd, th, thd, psi, r, v = st10.unbind(-1)
    X = torch.stack([u / env["u_ref"] - 1.0, v, r * env["L"] / env["u_ref"],
                     torch.cos(psi), torch.sin(psi), act[..., 0], act[..., 1],
                     zd, thd, (z - p.get("z0", 0.0)) / 0.2,
                     (th - p.get("th0", 0.0)) / 0.05], -1)
    mu = torch.as_tensor(stats["x_mu"], dtype=X.dtype, device=X.device)
    sd = torch.as_tensor(stats["x_sd"], dtype=X.dtype, device=X.device)
    return (X - mu) / sd


class Model0T:
    """data3.model0_step in torch float64, differentiable: ReducedModel.step
    with no waves over env["sub"] substeps of env["dt"]. With eta = 0 every
    wave term vanishes exactly (numpy adds 0.0): the heave / pitch
    equilibria are the running attitude z0 / th0 and the wave sway / yaw
    terms are 0. The heave and pitch transitions are red._phi's (constants
    here); the waterjet or rudder branch as in the numpy code."""

    def __init__(self, env):
        red = env["red"]
        self.p = {k_: float(v) for k_, v in red.p.items()
                  if isinstance(v, (int, float, np.integer, np.floating))}
        p = self.p
        self.dt, self.sub = float(env["dt"]), int(env["sub"])
        self.t_max, self.rud_max = float(env["t_max"]), float(env["rud_max"])
        # heave and pitch are linear and decoupled with eta = 0: the sub
        # substeps are one application of the transition's power (the same
        # map up to rounding, far fewer kernels)
        self.Pz = [[float(a) for a in row] for row in np.linalg.matrix_power(
            red._phi(p["wn_heave"], p["z_heave"], self.dt), self.sub)]
        self.Pp = [[float(a) for a in row] for row in np.linalg.matrix_power(
            red._phi(p["wn_pitch"], p["z_pitch"], self.dt), self.sub)]
        self.z_eq, self.th_eq = p.get("z0", 0.0), p.get("th0", 0.0)
        self.jet = p.get("steer_jet", 0.0) > 0.5
        self.ay = float(np.exp(-self.dt / p["tau_r"]))

    def __call__(self, s, U):
        """One control step: s (..., 10), U (..., 2) command fractions
        (broadcastable)."""
        p, dt, Pz, Pp = self.p, self.dt, self.Pz, self.Pp
        thrust, rudder = U[..., 0] * self.t_max, U[..., 1] * self.rud_max
        x, y, u, z, zd, th, thd, psi, r, v = s.unbind(-1)
        ez, ep_ = z - self.z_eq, th - self.th_eq
        nz = Pz[0][0] * ez + Pz[0][1] * zd
        nzd = Pz[1][0] * ez + Pz[1][1] * zd
        nth = Pp[0][0] * ep_ + Pp[0][1] * thd
        nthd = Pp[1][0] * ep_ + Pp[1][1] * thd
        rud_eff = rudder.clamp(-p["rud_stall"], p["rud_stall"])
        if self.jet:
            # the jet's side force and yaw-rate target are fixed over the
            # control step (commands held)
            lift = p["k_jet_side"] * thrust.clamp(min=0.0) * torch.sin(rud_eff)
            r_tgt = p["k_nomoto_f"] * lift
        for _ in range(self.sub):
            ud = (thrust - p["k_drag"] * u * u.abs()) / p["m_surge"]
            if not self.jet:
                lift = p["k_lift"] * u * u.abs() * rud_eff
            vd = (lift - p["k_lin_sway"] * v - p["k_sway"] * v * v.abs()
                  - p["m_coriolis"] * u * r) / p["m_sway"]
            if self.jet:
                r_new = r_tgt + (r - r_tgt) * self.ay
            else:
                r_new = (p["k_nomoto"] * rudder
                         + (r - p["k_nomoto"] * rudder) * self.ay)
            u = u + ud * dt
            v = v + vd * dt
            psi = psi + 0.5 * (r + r_new) * dt
            r = r_new
            c_, s_ = torch.cos(psi), torch.sin(psi)
            x = x + (u * c_ - v * s_) * dt
            y = y + (u * s_ + v * c_) * dt
        return torch.stack([x, y, u, nz + self.z_eq, nzd, nth + self.th_eq,
                            nthd, psi, r, v], -1)


_M0T = {}


def model0t(env):
    """The Model0T of an env (built once)."""
    key = id(env["red"])
    if key not in _M0T:
        _M0T[key] = Model0T(env)
    return _M0T[key]


def _qkv(lay, x, d):
    sa = lay.self_attn
    nh = sa.num_heads
    q, k_, v = F.linear(lay.norm1(x), sa.in_proj_weight,
                        sa.in_proj_bias).split(d, -1)
    return q, k_, v, nh, d // nh


def kv_history(net, tok):
    """encode()'s square causal pass over history tokens (B, L, D_TOK),
    layer by layer by hand with the same parameters (norm-first layers:
    x += SA(LN1 x), x += FF(LN2 x) with FF = linear1 -> ReLU -> linear2;
    heads split from in_proj; final LayerNorm), keeping every layer's keys
    and values. Returns ([(K, V) (B, heads, L, d_head)] per layer, outputs
    (B, L, d)). Causal, so the keys / values at t depend on tokens <= t."""
    B, L, _ = tok.shape
    d = net.d
    x = net.inp(tok) + net.pos[:L][None]
    kv = []
    for lay in net.tf.layers:
        q, k_, v, nh, hd = _qkv(lay, x, d)
        q, k_, v = (t.view(B, L, nh, hd).transpose(1, 2) for t in (q, k_, v))
        kv.append((k_, v))
        # square and causal: is_causal's top-left alignment is right here
        o = F.scaled_dot_product_attention(q, k_, v, is_causal=True)
        x = x + lay.self_attn.out_proj(o.transpose(1, 2).reshape(B, L, d))
        x = x + lay.linear2(lay.activation(lay.linear1(lay.norm2(x))))
    return kv, net.out(x)


def kv_step(net, tok, pos, hist, hide, cache):
    """One generated token per sequence, attending to its row's history
    and to its own sequence's generated tokens so far.

    tok (B, ..., D_TOK) and pos (B, ..., d) (broadcastable) the token and
    its position embedding; hist kv_history's (K, V) per layer (B, heads,
    NH, d_head), shared by all sequences of a row (broadcast in the
    products, never copied); hide (B, NH) True for history keys the row
    must not see (its recorded future when k < NH); cache per layer
    ([keys], [values]) of the generated tokens, appended to here (a list,
    no in-place writes, no concatenation with the history cache).
    A single query attends to all visible keys; SDPA with is_causal=True
    would be top-left aligned and let it see key 0 only, hence by hand.
    Returns the output (B, ..., d)."""
    d = net.d
    x = net.inp(tok) + pos
    lead = x.shape[:-1]
    B, nb = lead[0], len(lead) - 1
    hmask = hide.view(B, *([1] * (nb + 1)), hide.shape[-1])
    for li, lay in enumerate(net.tf.layers):
        q, k_, v, nh, hd = _qkv(lay, x, d)
        q, k_, v = (t.reshape(*lead, nh, hd) for t in (q, k_, v))
        ck, cv = cache[li]
        ck.append(k_)
        cv.append(v)
        Kh, Vh = hist[li]
        NH = Kh.shape[2]
        sc = hd ** -0.5
        s_h = torch.einsum("b...hd,bhtd->b...ht", q, Kh) * sc
        s_h = s_h.masked_fill(hmask, float("-inf"))
        Kg, Vg = torch.stack(ck, -2), torch.stack(cv, -2)   # (..., h, j+1, dh)
        s_g = (q.unsqueeze(-2) @ Kg.transpose(-1, -2)).squeeze(-2) * sc
        w = torch.softmax(torch.cat([s_h, s_g], -1), -1)
        o = (torch.einsum("b...ht,bhtd->b...hd", w[..., :NH], Vh)
             + (w[..., NH:].unsqueeze(-2) @ Vg).squeeze(-2))
        x = x + lay.self_attn.out_proj(o.reshape(*lead, d))
        x = x + lay.linear2(lay.activation(lay.linear1(lay.norm2(x))))
    return net.out(x)


def _damp(x, f):
    """x in the forward pass (exactly), its gradient times f."""
    xd = x.detach()
    return xd + f * (x - xd)


def rollout_core(net, D, ii, k, plans, xs_k, env, n_samp, base=None,
                 gen=None, grad=False, obs=None, fb_damp=None, ode_steps=24,
                 steer=None):
    """Multi-step prediction, the ONE code path for rollout training (grad)
    and evaluation (no grad).

    Rows are moments b (episode ii[b], moment k[b]; k may differ between
    rows), plans (B, P, H, 2) command fractions per moment, xs_k (B, 14) the
    plant state at k; n_samp = S samples per (moment, plan). Returns e
    samples (B, P, S, H, 10) normalised, the rolled reduced states after
    each step (B, P, S, H, 10) float64 and the actuator positions after
    each step (B, P, S, H, 2).

    History: the window [a, a + NH), a = max(0, k - NH), NH = W_CTX - HB,
    tokens from window_tokens (the first flag at the window's position 0,
    its e_prev = e_{a-1} or 0), padded to the FIXED length NH; Lh = k - a.
    Recorded tokens at positions >= Lh (the recorded future when k < NH)
    are hidden from every generated token; history positions < Lh attend
    causally, so nothing an output depends on sees them. One history pass
    per moment, shared by its P plans and S samples. obs (B, 10): optional
    observation noise on the RECORDED e_prev (history and token 0), as
    window_tokens(obs_sd=...) adds it in training.

    Token j of a sequence sits at position Lh + j. Conventions as rollout():
    token 0 carries the TRUE actuator position xs_k[12:14] / max and the
    recorded e_{k-1}; token j >= 1 the actuator column clip(U_{j-1} +
    gap_{j-1} * dtc) (the command just taken plus its sampled gap) and
    e_prev = sample j - 1; the command columns hold U_j; the state columns
    come from the rolled state, s_{j+1} = Model0T(s_j, U_j) + e[:8] * dtc on
    the E7 columns (float64).

    Flow sampling: base noise (B, P, S, H, 10) ~ N(0, I), drawn fresh from
    gen (on gen's device) unless passed as `base` (common random numbers,
    tests); ode_steps Euler steps (never fewer than 24). grad=True keeps
    gradients through the history pass, the sampler, the generated keys /
    values, e_prev, the actuator columns and the rolled states; fb_damp
    (off by default) scales the gradient through the fed-back e_prev and
    actuator columns by that factor per step.

    steer (None by default: bit-identical to the rollouts above) is a
    state-feedback hook on the plan: steer(j, sr_j, U_j) receives the rolled
    reduced state BEFORE step j (B, P, S, 10) and the plan's command U_j
    (B, P, 1, 2) and returns the command actually applied during step j
    (B, P, S, 2), float64 (learn/meta/mpc_learned.SteerT: the nozzle column
    replaced by the heading autopilot, as on the boat). That command is
    what the token's command columns, Model0T and the actuator column see."""
    assert ode_steps >= 24, "the flow needs its 24-step grid"
    with torch.set_grad_enabled(grad):
        return _rollout_core(net, D, ii, k, plans, xs_k, env, n_samp, base,
                             gen, obs, fb_damp, ode_steps, steer)


def _rollout_core(net, D, ii, k, plans, xs_k, env, S, base, gen, obs,
                  fb_damp, ode_steps, steer=None):
    from learn.meta.data3 import E7
    dev, st = D.dev, D.stats
    iit = torch.as_tensor(ii, device=dev).long()
    kt = torch.as_tensor(k, device=dev).long()
    pl = torch.as_tensor(plans, dtype=torch.float64, device=dev)
    B, P, Hh, _ = pl.shape
    NH = W_CTX - HB
    assert Hh <= HB
    gdev = gen.device if gen is not None else dev
    a = (kt - NH).clamp(min=0)
    Lh = kt - a
    tok_h, _, _ = window_tokens(D, iit, a, NH, obs_sd=obs, gen=gen)
    hist, _ = kv_history(net, tok_h)
    hide = torch.arange(NH, device=dev)[None] >= Lh[:, None]
    e_prev = D.E[iit, (kt - 1).clamp(min=0)] * (kt >= 1)[:, None]
    if obs is not None:
        e_prev = e_prev + obs * torch.randn(e_prev.shape, generator=gen,
                                            device=gdev).to(dev)
    e_prev = e_prev[:, None, None].expand(B, P, S, C7)
    xs = torch.as_tensor(xs_k, dtype=torch.float64, device=dev)
    amax = torch.tensor([env["t_max"], env["rud_max"]], dtype=torch.float64,
                        device=dev)
    act = (xs[:, 12:14] / amax)[:, None, None].expand(B, P, S, 2)
    sr = plant_to_reduced_t(xs)[:, None, None].expand(B, P, S, 10)
    if base is None:
        base = torch.randn((B, P, S, Hh, C7), generator=gen, device=gdev)
    base = base.to(dev)
    u_mu = torch.as_tensor(st["u_mu"], dtype=torch.float64, device=dev)
    u_sd = torch.as_tensor(st["u_sd"], dtype=torch.float64, device=dev)
    Un = ((pl - u_mu) / u_sd).float()
    e_sd = torch.as_tensor(st["e_sd"], device=dev)
    sel = torch.zeros(len(E7), 10, dtype=torch.float64, device=dev)
    sel[torch.arange(len(E7)), torch.tensor(E7)] = 1.0
    lo = torch.tensor([0.0, -1.0], dtype=torch.float64, device=dev)
    hi = torch.tensor([1.0, 1.0], dtype=torch.float64, device=dev)
    dtc = env["dt"] * env["sub"]
    m0 = model0t(env)
    n_st = len(E7)
    cache = [([], []) for _ in net.tf.layers]
    outs, states, acts = [], [], []
    for j in range(Hh):
        X = state_tokens_t(sr, act, st, env).float()
        first = ((Lh + j) == 0).float().view(B, 1, 1).expand(B, P, S)
        Uj = pl[:, :, None, j]
        if steer is None:
            Un_j = Un[:, :, None, j].expand(B, P, S, 2)
        else:
            # the command applied during step j, per sample (feedback on
            # the rolled state), normalised as Un
            Uj = steer(j, sr, Uj)
            Un_j = ((Uj - u_mu) / u_sd).float()
        tok = make_tokens(X, Un_j, e_prev, first)
        h = kv_step(net, tok, net.pos[Lh + j].view(B, 1, 1, -1), hist, hide,
                    cache)
        e = net.sample_grad(h, base[:, :, :, j], ode_steps)
        e_raw = (e * e_sd).double()
        sr = m0(sr, Uj) + (e_raw[..., :n_st] * dtc) @ sel
        act = torch.maximum(torch.minimum(
            Uj + e_raw[..., n_st:n_st + 2] * dtc, hi), lo)
        outs.append(e)
        states.append(sr)
        acts.append(act)
        e_prev = e
        if fb_damp is not None:
            e_prev, act = _damp(e_prev, fb_damp), _damp(act, fb_damp)
    return torch.stack(outs, 3), torch.stack(states, 3), torch.stack(acts, 3)


@torch.no_grad()
def rollout_kv(net, D, ii, k, plans, xs_k, env, n_samp=16, gen=None):
    """rollout() through rollout_core, with its signature and outputs
    (plans (B, H, 2), P = 1; rows may have different k): e samples (B,
    n_samp, H, 10) normalised, rolled reduced states (B, n_samp, H, 10) as
    numpy."""
    pl = torch.as_tensor(np.asarray(plans), dtype=torch.float64)[:, None]
    e, s, _ = rollout_core(net, D, ii, k, pl, xs_k, env, n_samp, gen=gen)
    return e[:, 0], s[:, 0].cpu().numpy()


@torch.no_grad()
def rollout(net, D, ii, k, plans, xs_k, env, n_samp=16, gen=None,
            ctx=W_CTX - HB, ode_steps=24):
    """Multi-step prediction: B moments (episode ii[b], step k[b]), plans
    (B, H, 2) command fractions, the plant state at k (B, 14). Returns
    samples of e (B, n_samp, H, 10), normalised units, and the rolled-out
    reduced states after each step (B, n_samp, H, 10)."""
    from learn.meta.data2 import plant_to_reduced
    from learn.meta.data3 import E7, model0_step
    dev = D.dev
    st = D.stats
    B, Hh, _ = plans.shape
    kt = torch.as_tensor(k, device=dev)
    iit = torch.as_tensor(ii, device=dev)
    a = (kt - ctx).clamp(min=0)
    assert bool((kt == kt[0]).all()), "rollout rows must share one moment k"
    Lh = int(kt[0] - a[0])
    # history tokens [a, k): the recorded steps before the moment only
    tok_h, _, _ = window_tokens(D, iit, a, Lh)
    nh = (kt - a)                                     # history length/row
    sr = np.repeat(plant_to_reduced(xs_k), n_samp, 0)
    Ur = np.repeat(plans, n_samp, 0)
    act = np.repeat(np.asarray(xs_k[:, 12:14]) / np.array(
        [env["t_max"], env["rud_max"]]), n_samp, 0)
    e_prev = D.E[iit, (kt - 1).clamp(min=0)] * (kt >= 1)[:, None]
    e_prev = e_prev.repeat_interleave(n_samp, 0)
    toks = tok_h.repeat_interleave(n_samp, 0)
    nhr = nh.repeat_interleave(n_samp, 0)
    out = torch.zeros(B * n_samp, Hh, C7, device=dev)
    states = np.zeros((B * n_samp, Hh, 10))
    dtc = env["dt"] * env["sub"]
    n_st = len(E7)
    for j in range(Hh):
        X = torch.tensor(state_tokens(sr, act, Ur[:, j], st, env),
                         dtype=torch.float32, device=dev)
        Un = torch.tensor((Ur[:, j] - st["u_mu"]) / st["u_sd"],
                          dtype=torch.float32, device=dev)
        first = ((nhr + j) == 0).float()
        new = make_tokens(X[:, None], Un[:, None], e_prev[:, None],
                          first[:, None])
        toks = torch.cat([toks, new], 1)
        # all rows share one k (asserted above), so every row's sequence is
        # exactly its history [a, k) plus the j + 1 generated tokens, at
        # positions 0 .. Lh + j <= W_CTX - 1: nothing to pad or hide.
        # (rollout_core does the same with per-row k and a KV cache; this
        # re-encoding version is kept as the reference for the tests)
        h = net.encode(toks[:, -W_CTX:])[:, -1]
        e = net.sample(h, 1, steps=ode_steps, gen=gen)[:, 0]
        out[:, j] = e
        e_prev = e
        e_raw = e.cpu().numpy() * st["e_sd"]
        nxt = model0_step(env, sr, Ur[:, j])
        nxt[:, list(E7)] += e_raw[:, :n_st] * dtc
        sr = nxt
        states[:, j] = sr
        # the actuator positions as the network was trained on them: the
        # command plus its predicted trailing part (no lag model)
        act = np.clip(Ur[:, j] + e_raw[:, n_st:n_st + 2] * dtc,
                      [0.0, -1.0], [1.0, 1.0])
    return (out.view(B, n_samp, Hh, C7),
            states.reshape(B, n_samp, Hh, 10))


# ------------------------------------------------ rollout training (M9)
class Branches:
    """A wide-plan branch file (data3.build_tbranches: ep, k, U, XS, E0,
    wide) on the device, with what the losses need: the branch tokens
    (token j: the TRUE state and actuator position before step j, the
    command U_j), errors in normalised units, the e = 0 rollout of every
    plan from s_k (Model0T) on the ES state features (u, v, r, z, th, psi,
    actuator thrust / nozzle) and the TRUE deviation from it."""

    def __init__(self, br, D, env):
        dev, st = D.dev, D.stats
        f64 = dict(dtype=torch.float64, device=dev)
        # backstop for files written before build_tbranches filtered: a
        # non-finite moment would make the ES scale NaN (every step skipped)
        keep = np.isfinite(br["XS"]).all((1, 2, 3)) & \
            np.isfinite(br["E0"]).all((1, 2, 3))
        if not keep.all():
            n0 = keep.size
            br = {k_: (np.asarray(v)[keep] if np.ndim(v) and len(v) == n0
                       else v) for k_, v in dict(br).items()}
            print(f"Branches: dropped {n0 - int(keep.sum())} of {n0} moments "
                  f"with non-finite XS / E0", flush=True)
            if not keep.any():
                raise RuntimeError("Branches: all moments non-finite")
        self.n_dropped = int((~keep).sum())
        N, P, Hh, _ = br["U"].shape
        self.N, self.P, self.H = N, P, Hh
        self.ep = torch.as_tensor(br["ep"], device=dev).long()
        self.k = torch.as_tensor(br["k"], device=dev).long()
        self.U = torch.as_tensor(br["U"], **f64)
        XS = torch.as_tensor(br["XS"], **f64)
        self.xs0 = XS[:, 0, 0]                  # all plans start here
        amax = torch.tensor([env["t_max"], env["rud_max"]], **f64)
        act = XS[..., 12:14] / amax
        red = plant_to_reduced_t(XS)
        self.Xtok = state_tokens_t(red[:, :, :Hh], act[:, :, :Hh], st,
                                   env).float()
        u_mu = torch.as_tensor(st["u_mu"], **f64)
        u_sd = torch.as_tensor(st["u_sd"], **f64)
        self.Un = ((self.U - u_mu) / u_sd).float()
        self.E = torch.as_tensor(br["E0"] / st["e_sd"], dtype=torch.float32,
                                 device=dev)
        wide = br["wide"] if "wide" in br else np.zeros((N, P), bool)
        self.wide = torch.as_tensor(wide, device=dev).bool()
        m0 = model0t(env)
        sr = plant_to_reduced_t(self.xs0)[:, None].expand(N, P, 10)
        ref = []
        for j in range(Hh):
            sr = m0(sr, self.U[:, :, j])
            # with e = 0 the actuator sits at the command
            ref.append(torch.cat([sr[..., list(SC_ES)], self.U[:, :, j]], -1))
        self.ref = torch.stack(ref, 2)          # (N, P, H, 8)
        truth = torch.cat([red[:, :, 1:, list(SC_ES)], act[:, :, 1:]], -1)
        self.dev_true = truth - self.ref


def fm_branch_loss(net, D, br, rows, pp, gen, noise=True):
    """The FM loss on branch windows (moments rows, plans pp): the recorded
    tokens [a, k), a = max(0, k + H - W_CTX), then the branch's H tokens
    (true state and actuator position, command U_j, e_prev = the recorded
    e_{k-1} for j = 0 and the branch's own e_{j-1} after), scored on the H
    branch positions against the branch's errors. Observation noise as in
    fm_loss, on every e_prev of the window."""
    dev, Hh = D.dev, br.H
    ii, k = br.ep[rows], br.k[rows]
    b = len(rows)
    a = (k + Hh - W_CTX).clamp(min=0)
    Lr = k - a
    osd = obs_sd(b, dev, gen) if noise else None
    tok_r, _, _ = window_tokens(D, ii, a, W_CTX - Hh, osd, gen)
    Eb = br.E[rows, pp]                                  # (b, H, C7)
    e0 = D.E[ii, (k - 1).clamp(min=0)] * (k >= 1)[:, None]
    ep = torch.cat([e0[:, None], Eb[:, :-1]], 1)
    if noise:
        ep = ep + osd[:, None] * torch.randn(ep.shape, generator=gen).to(dev)
    idx = Lr[:, None] + torch.arange(Hh, device=dev)[None]
    tok_b = make_tokens(br.Xtok[rows, pp], br.Un[rows, pp], ep,
                        (idx == 0).float())
    # the recorded window, then room for the branch; the branch overwrites
    # [Lr, Lr + H): anything after it is invisible (causal)
    tok = torch.cat([tok_r, torch.zeros(b, Hh, D_TOK, device=dev)], 1)
    tok = tok.scatter(1, idx[..., None].expand(-1, -1, D_TOK), tok_b)
    h = net.encode(tok).gather(1, idx[..., None].expand(-1, -1, net.d))
    return fm_loss_h(net, h, Eb, torch.ones_like(idx, dtype=torch.bool), gen)


def energy_score(x, y):
    """Fair energy score per group: samples x (R, S, n, G), truth y (R, n,
    G) -> (R, G), with the norm over the n features of each group:
    mean_s |x_s - y| - sum_{s != s'} |x_s - x_s'| / (2 S (S - 1)).
    sqrt(. + 1e-12) keeps the gradient finite at 0."""
    S = x.shape[1]
    d_xy = torch.sqrt(((x - y[:, None]) ** 2).sum(-2) + 1e-12).mean(1)
    d_xx = torch.sqrt(((x[:, :, None] - x[:, None]) ** 2).sum(-2) + 1e-12)
    off = 1.0 - torch.eye(S, device=x.device, dtype=x.dtype)
    return d_xy - (d_xx * off[None, :, :, None]).sum((1, 2)) / (
        2 * S * (S - 1))


def es_value(br, rows, pp, e, sr, act, scale):
    """Energy score of rollouts against their branch truths, summed over
    the feature groups, per (moment, plan): rows (b,), pp (b, np), e (b,
    np, S, H, 10), sr (b, np, S, H, 10), act (b, np, S, H, 2) -> (b, np).

    e groups, per error channel c: [a_{1..H,c}, cumsum_j a_{j,c} / sqrt(H)],
    a = asinh(e_norm / KAPPA) (the running sum is what the state
    integrates). State groups, per channel of (u, v, r, z, th, psi,
    actuator thrust, nozzle): the H-sequence of the deviation from the e = 0
    rollout / scale (H, 8), the per-step sd of the TRUE deviation over the
    training branches."""
    b, n_p, S, Hh, _ = e.shape

    def fe(x):
        a = torch.asinh(x / KAPPA)
        return torch.cat([a, a.cumsum(-2) / math.sqrt(Hh)], -2)

    y_e = fe(br.E[rows[:, None], pp]).reshape(b * n_p, 2 * Hh, C7)
    x_e = fe(e).reshape(b * n_p, S, 2 * Hh, C7)
    ref = br.ref[rows[:, None], pp][:, :, None]
    x_s = ((torch.cat([sr[..., list(SC_ES)], act], -1) - ref) / scale
           ).float().reshape(b * n_p, S, Hh, -1)
    y_s = (br.dev_true[rows[:, None], pp] / scale).float().reshape(
        b * n_p, Hh, -1)
    es = energy_score(x_e, y_e).sum(-1) + energy_score(x_s, y_s).sum(-1)
    return es.view(b, n_p)


def _lr_factor(it, steps, warm):
    """Linear warm-up, then cosine down to 10% at `steps`."""
    if it < warm:
        return (it + 1) / warm
    x = (it - warm) / max(1, steps - warm)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, x)))


def train_rollout(net, D, tb, vb, env, es_weight=1.0, steps=3000, lr=1e-4,
                  warmup=150, check=250, patience=4, n_rec=36, n_br=12,
                  n_mom=16, n_plan=2, n_samp=8, fb_damp=None, seed=0,
                  log=print, log_every=25):
    """Fine-tune the one-step model (from model3.pt) for rollouts (DEFECTS
    M9). es_weight 0: V0 'coverage' (the teacher-forced FM loss on recorded
    AND wide-plan branch windows); es_weight 1: V1 'rollout score' (plus a
    sequence-level energy score of its own 24-step rollouts against the
    true continuations under the same plan).

    Why V1 is statistically right: the truth and a rollout both obey
    s_{j+1} = model0(s_j, u_j) + e_j * dtc on the 8 E7 columns and act_{j+1}
    = u_j + gap_j * dtc (e0_from defines e so; no heading wrap; the actuator
    clip never acts on truths), so the conditional joint law of the e
    sequence given (history before k, plan) fixes the joint law of the
    rolled states and actuator positions (x, y aside). Energy scores are
    proper, so the TRUE conditional joint minimises them. A per-step
    corrective target (Data-as-Demonstrator) would teach cancelling
    legitimate random draws, and oracle one-step labels on the model's own
    path (DAgger) would teach that what the model's own samples imply about
    hidden things (actuator rate, operator gains, slow drift, coloured-noise
    state) does not persist; neither is built.

    tb / vb: training / validation branch dicts (train_tbranches.npz,
    train_vbranches.npz). One step: L_FM = the window-weighted mean of the
    FM loss on n_rec recorded windows (training episodes, observation
    noise) and n_br branch windows; backward, clip 1.0, stash. V1 also:
    L_ES on n_mom moments x n_plan of their plans x n_samp samples
    (rollout_core with grad and observation noise), backward, clip 1.0 on
    its own; grad = g_FM + es_weight * g_ES (both bounded, so es_weight is
    the ratio of update sizes). A step with a non-finite loss or gradient
    is skipped. AdamW, warm-up then cosine to 10%.

    Validation at step 0 (= the initial model) and every `check` steps,
    with common random numbers (fixed branches, fixed base noise per row):
    ES on vb (also split by plans with / without a full-range step), FM on
    the validation episodes' recorded windows and on vb's branch windows.
    V1 keeps the lowest validation ES whose recorded-window FM is <= 1.02 x
    its step-0 value, counting an improvement only when it beats the best
    by > 2 paired standard errors over moments; V0 keeps the lowest
    combined FM. Stops after `patience` checks without improvement and
    restores the best (possibly step 0). Returns (net, info)."""
    dev = D.dev
    params = [p for p in net.parameters()]
    g = torch.Generator(device="cpu").manual_seed(seed + 11)
    tr, va = split_pools(D.n)                 # model3.pt's held-out split
    Lmax = D.len.cpu()
    TB, VB = Branches(tb, D, env), Branches(vb, D, env)
    va_set = set(va.tolist())
    assert not (set(TB.ep.tolist()) & va_set), "training branches in val"
    assert set(VB.ep.tolist()) <= va_set, "validation branches not in val"
    # (H, 8), stored with the model. Floored at ES_SCALE_FLOOR x the
    # channel's one-step error scale e_sd * dtc: where the truth is
    # (nearly) deterministic -- the nozzle has caught up by the late steps
    # of most branches, sd ~1e-8 -- the raw sd would blow any sample spread
    # up by 1e8 and the ES would be nothing but that channel
    dtc = env["dt"] * env["sub"]
    esd = torch.as_tensor(D.stats["e_sd"], dtype=torch.float64, device=dev)
    floor = ES_SCALE_FLOOR * esd[list(ES_E_CH)] * dtc
    scale = torch.maximum(TB.dev_true.std((0, 1)), floor)
    assert torch.isfinite(scale).all(), "ES scale non-finite"
    w_rec = n_rec / (n_rec + n_br)
    opt = torch.optim.AdamW(params, lr, weight_decay=1e-4)
    warm = min(warmup, max(1, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: _lr_factor(it, steps, warm))
    base_v = torch.randn((VB.N, VB.P, n_samp, VB.H, C7),
                         generator=torch.Generator().manual_seed(seed + 99))

    @torch.no_grad()
    def validate():
        net.eval()
        es = torch.zeros(VB.N, VB.P, device=dev)
        allp = torch.arange(VB.P, device=dev)
        for s0 in range(0, VB.N, 32):
            r = torch.arange(s0, min(VB.N, s0 + 32), device=dev)
            e, sr, act = rollout_core(net, D, VB.ep[r], VB.k[r], VB.U[r],
                                      VB.xs0[r], env, n_samp,
                                      base=base_v[r.cpu()])
            es[r] = es_value(VB, r, allp.expand(len(r), -1), e, sr, act,
                             scale)
        fm_rec = fm_validate(net, D, va, W_CTX, Lmax)
        gen = torch.Generator(device="cpu").manual_seed(4322)
        flat = torch.arange(VB.N * VB.P, device=dev)
        tot = 0.0
        for s0 in range(0, len(flat), 32):
            fl = flat[s0:s0 + 32]
            tot += fm_branch_loss(net, D, VB, fl // VB.P, fl % VB.P,
                                  gen).item() * len(fl)
        fm_br = tot / len(flat)
        net.train()
        wd = VB.wide
        return dict(es_rows=es.mean(1), es=es.mean().item(),
                    es_wide=es[wd].mean().item() if wd.any() else float("nan"),
                    es_narrow=es[~wd].mean().item(), fm_rec=fm_rec,
                    fm_br=fm_br, fm=w_rec * fm_rec + (1 - w_rec) * fm_br)

    def show(it, v, tag=""):
        log(f"    check {it:5d}: val ES {v['es']:.4f} (full-range plans "
            f"{v['es_wide']:.4f}, others {v['es_narrow']:.4f}) | val FM "
            f"recorded {v['fm_rec']:.4f}, branch {v['fm_br']:.4f}, combined "
            f"{v['fm']:.4f}{tag}")

    net.train()
    v0 = validate()
    show(0, v0, "  (initial model)")
    snap = lambda: {k_: x.detach().clone()  # noqa: E731
                    for k_, x in net.state_dict().items()}
    best = dict(it=0, v=v0, state=snap(), bad=0)
    hist = [dict(it=0, **{k_: x for k_, x in v0.items() if k_ != "es_rows"})]
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    n_skip, t_steps, it = 0, 0.0, 0
    for it in range(1, steps + 1):
        t1 = time.time()
        ii, a = fm_draw(tr, n_rec, g, Lmax, W_CTX, dev)
        l_rec = fm_loss(net, D, ii, a, W_CTX, g)
        rows = torch.randint(0, TB.N, (n_br,), generator=g).to(dev)
        pp = torch.randint(0, TB.P, (n_br,), generator=g).to(dev)
        l_br = fm_branch_loss(net, D, TB, rows, pp, g)
        l_fm = w_rec * l_rec + (1 - w_rec) * l_br
        opt.zero_grad(set_to_none=True)
        l_fm.backward()
        g_fm = float(nn.utils.clip_grad_norm_(params, 1.0))
        l_es, g_es = float("nan"), 0.0
        if es_weight > 0:
            stash = [None if p.grad is None else p.grad.detach().clone()
                     for p in params]
            opt.zero_grad(set_to_none=True)
            rows = torch.randperm(TB.N, generator=g)[:n_mom].to(dev)
            pp = torch.rand(len(rows), TB.P, generator=g).argsort(1)[
                :, :n_plan].to(dev)
            e, sr, act = rollout_core(
                net, D, TB.ep[rows], TB.k[rows], TB.U[rows[:, None], pp],
                TB.xs0[rows], env, n_samp, gen=g, grad=True,
                obs=obs_sd(len(rows), dev, g), fb_damp=fb_damp)
            loss_es = es_value(TB, rows, pp, e, sr, act, scale).mean()
            loss_es.backward()
            del e, sr, act
            l_es = float(loss_es)
            g_es = float(nn.utils.clip_grad_norm_(params, 1.0))
            for p, s in zip(params, stash):
                if s is None:
                    p.grad = None if p.grad is None else es_weight * p.grad
                elif p.grad is None:
                    p.grad = s
                else:
                    p.grad = s + es_weight * p.grad
        finite = all(math.isfinite(x) for x in (float(l_fm), g_fm, g_es)) \
            and (es_weight == 0 or math.isfinite(l_es))
        if finite:
            opt.step()
        else:
            n_skip += 1
            opt.zero_grad(set_to_none=True)
            log(f"    step {it}: non-finite (FM {float(l_fm):.4g}, ES "
                f"{l_es:.4g}, grad norms {g_fm:.4g} / {g_es:.4g}), skipped")
        sched.step()
        t_steps += time.time() - t1
        if it % log_every == 0 or it == 1:
            mem = (torch.cuda.max_memory_allocated() / 2 ** 30
                   if dev.type == "cuda" else 0.0)
            log(f"    step {it:5d}: FM {float(l_fm):.4f} (recorded "
                f"{float(l_rec):.4f}, branch {float(l_br):.4f}) ES "
                f"{l_es:.4f} | grad norms FM {g_fm:.3f} ES {g_es:.3f} | lr "
                f"{sched.get_last_lr()[0]:.2e} | {t_steps / it:.2f} s/step, "
                f"GPU peak {mem:.2f} GB")
        if it % check == 0 or it == steps:
            v = validate()
            if es_weight > 0:
                ok_fm = v["fm_rec"] <= 1.02 * v0["fm_rec"]
                dd = v["es_rows"] - best["v"]["es_rows"]
                se = (dd.std() / math.sqrt(len(dd))).item() if len(dd) > 1 \
                    else 0.0
                better = ok_fm and dd.mean().item() < -2 * se
                why = (f"  (dES vs best {dd.mean().item():+.4f} +- {se:.4f}"
                       f"{'' if ok_fm else ', recorded FM > 1.02 x start'})")
            else:
                better = v["fm"] < best["v"]["fm"]
                why = ""
            show(it, v, why + ("  best" if better else ""))
            hist.append(dict(it=it, **{k_: x for k_, x in v.items()
                                       if k_ != "es_rows"}))
            if better:
                best.update(it=it, v=v, state=snap(), bad=0)
            else:
                best["bad"] += 1
                if best["bad"] >= patience:
                    log(f"    not better for {patience} checks, stopping at "
                        f"step {it}")
                    break
    if best["it"] != it:
        net.load_state_dict(best["state"])
    log(f"    kept step {best['it']} (val ES {best['v']['es']:.4f}, FM "
        f"combined {best['v']['fm']:.4f}); {n_skip} steps skipped")
    mem = (torch.cuda.max_memory_allocated() / 2 ** 30
           if dev.type == "cuda" else 0.0)
    return net, dict(scale=scale.cpu().numpy(), best_it=best["it"],
                     hist=hist, n_skip=n_skip, s_per_step=t_steps / max(1, it),
                     gpu_peak_gb=mem, es_weight=es_weight)


@torch.no_grad()
def eval_multi_step(net, D, br, env, q, n_samp=16, batch=48, gen_seed=0,
                    mask=None):
    """br: dict(ep, k, U (NB, P, H, 2), XS (NB, P, H + 1, 14), E0 (NB, P, H,
    10)) of closed-loop truths under plans fixed in advance (all plans of a
    moment start from the same state). Rollouts by rollout_core, rows
    grouped as moments x plans (one history pass per moment); a batch may
    mix moments k, since every row's history ends at its own k. mask (NB,
    P): score only these (moment, plan) pairs, regrouped one plan per row."""
    if mask is not None:
        nb, pp = np.nonzero(mask)
        br = dict(ep=br["ep"][nb], k=br["k"][nb], U=br["U"][nb, pp][:, None],
                  XS=br["XS"][nb, pp][:, None], E0=br["E0"][nb, pp][:, None])
    dev = D.dev
    gen = torch.Generator(device=dev).manual_seed(gen_seed)
    NB, P, Hh, _ = br["U"].shape
    from learn.meta.data2 import plant_to_reduced
    from learn.meta.data3 import model0_step
    tal = Tally((Hh, C7))
    # rolled-out STATES: u, v, r, z, theta, psi (reduced columns), scored
    # against the true states; the reference is the MPC's own rollout
    # without any error model (e = 0)
    SC = SC_ES
    tal_s = Tally((Hh, len(SC)))
    U = br["U"].reshape(NB * P, Hh, 2)
    XS0 = br["XS"][:, :, 0].reshape(NB * P, 14)
    XT = plant_to_reduced(br["XS"][:, :, 1:].reshape(-1, 14)).reshape(
        NB * P, Hh, 10)
    E0 = br["E0"].reshape(NB * P, Hh, C7) / D.stats["e_sd"]
    # scale of each state channel: sd of the true deviation from the
    # error-free rollout (so the table reads in comparable units)
    ref_all = np.zeros_like(XT)
    for s0 in range(0, NB * P, 4096):
        sr = plant_to_reduced(XS0[s0:s0 + 4096].astype(float))
        for j in range(Hh):
            sr = model0_step(env, sr, U[s0:s0 + 4096, j].astype(float))
            ref_all[s0:s0 + 4096, j] = sr
    ssc = (XT - ref_all)[..., list(SC)].reshape(-1, len(SC)).std(0) + 1e-9
    ssc_t = torch.tensor(ssc, device=dev)
    f = lambda a: torch.as_tensor(  # noqa: E731
        a, device=dev)[..., list(SC)].div(ssc_t).float()
    per = max(1, batch // P)                       # moments per batch
    for s0 in range(0, NB, per):
        mb = np.arange(s0, min(NB, s0 + per))
        sl = (mb[:, None] * P + np.arange(P)[None]).ravel()   # moment-plans
        e, sts, _ = rollout_core(net, D, br["ep"][mb], br["k"][mb],
                                 br["U"][mb].astype(float),
                                 br["XS"][mb, 0, 0].astype(float), env,
                                 n_samp, gen=gen)
        s = e.reshape(len(sl), n_samp, Hh, C7)
        tal.add(s, torch.tensor(E0[sl], dtype=torch.float32, device=dev), q)
        tal_s.add(f(sts.reshape(len(sl), n_samp, Hh, 10)), f(XT[sl]),
                  ref=f(ref_all[sl]))
    out = tal.summary()
    out["states"] = tal_s.summary(chans=len(SC))
    return out


def fmt_one(res, label):
    ch = "surge sway yaw heave pitch"
    s = res["skill"][:5]
    line = (f"  {label}: skill [{ch}] " + " ".join(f"{x:.2f}" for x in s)
            + " | cov90 " + " ".join(f"{x:.2f}" for x in res["cov"][:5])
            + " | CRPS skill " + " ".join(f"{x:.2f}" for x in
                                          res["crps"][:5]))
    if "tail" in res:
        rate, pred, brier, cnt = res["tail"]
        line += (" | large errors: rate " + " ".join(
            f"{x:.3f}" for x in rate[:5]) + ", predicted " + " ".join(
            f"{x:.3f}" for x in pred[:5]))
    return line


def fmt_multi(res, label):
    blocks = [(0, 1), (1, 5), (5, HB)]
    ch = "surge sway yaw heave pitch"
    lines = [f"  {label}: errors, per channel [{ch}], by horizon step "
             f"(skill = MSE / MSE of predicting 0):"]
    for a, b in blocks:
        s = res["skill"][a:b, :5].mean(0)
        c = res["cov"][a:b, :5].mean(0)
        cr = res["crps"][a:b, :5].mean(0)
        lines.append(f"    steps {a}-{b - 1}: skill " + " ".join(
            f"{x:.2f}" for x in s) + " | cov90 " + " ".join(
            f"{x:.2f}" for x in c) + " | CRPS skill " + " ".join(
            f"{x:.2f}" for x in cr))
    if "states" in res:
        S = res["states"]
        lines.append("  rolled-out states [u v r z theta psi] (skill = MSE /"
                     " MSE of the MPC's rollout without an error model):")
        for a, b in blocks:
            lines.append(f"    steps {a}-{b - 1}: skill " + " ".join(
                f"{x:.2f}" for x in S["skill"][a:b].mean(0)) + " | cov90 "
                + " ".join(f"{x:.2f}" for x in S["cov"][a:b].mean(0)))
    return "\n".join(lines)
