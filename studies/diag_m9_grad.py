#!/usr/bin/env python3
"""
Why did V1's rollout score make the model worse (DEFECTS M9)? Three looks
at model3.pt, on the training / validation branches:

  1  which feature groups make up the validation energy score (10 error
     channels, 8 state channels), for plans with / without a full-range step
  2  the rollout-score gradient over 12 fixed training batches, four ways:
       full    as in training (gradients through the fed-back samples)
       cut     the generated tokens' inputs (state, actuator, e_prev)
               detached: each step's output is trained for its own
               (detached) context only; the loss still integrates the
               samples into the states
       cut_e   cut, error-channel groups only
       cut_s   cut, state groups only
     per way: mean norm of one batch's gradient, norm of the 12-batch mean,
     their ratio (how much of one batch's gradient is signal), and the
     cosine with the other ways and with the FM gradient
  3  does a small step along each way's mean direction lower the validation
     score? (96 fixed validation moments x 4 plans, fixed samples)

    python studies/diag_m9_grad.py [--cache-name meta3]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from learn.meta import model3 as M
from learn.meta import relabel

ap = argparse.ArgumentParser()
ap.add_argument("--cache-name", default="meta2",
                help="cache under studies/_cache (meta3: M10)")
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache",
                 ap.parse_args().cache_name)
dev = M.device(0.5)
ck = torch.load(os.path.join(C, "model3.pt"), weights_only=False)
net = M.Net().to(dev)
net.load_state_dict(ck["net"])
theta0 = {k: v.detach().clone() for k, v in net.state_dict().items()}
env = relabel._env()
D = M.Data3(C, "train", dev, stats=ck["stats"])
TB = M.Branches(dict(np.load(os.path.join(C, "train_tbranches.npz"))), D, env)
VB = M.Branches(dict(np.load(os.path.join(C, "train_vbranches.npz"))), D, env)
dtc = env["dt"] * env["sub"]
esd = torch.as_tensor(D.stats["e_sd"], dtype=torch.float64, device=dev)
scale = torch.maximum(TB.dev_true.std((0, 1)),
                      M.ES_SCALE_FLOOR * esd[list(M.ES_E_CH)] * dtc)
GROUPS = [f"e:{c}" for c in ("u", "v", "r", "zd", "thd", "z", "th", "psi",
                             "thr", "noz")] + \
    [f"s:{c}" for c in ("u", "v", "r", "z", "th", "psi", "thr", "noz")]


def es_groups(br, rows, pp, e, sr, act):
    """model3.es_value, but per feature group: (b, np, 18)."""
    b, n_p, S, Hh, _ = e.shape

    def fe(x):
        a = torch.asinh(x / M.KAPPA)
        return torch.cat([a, a.cumsum(-2) / math.sqrt(Hh)], -2)

    y_e = fe(br.E[rows[:, None], pp]).reshape(b * n_p, 2 * Hh, M.C7)
    x_e = fe(e).reshape(b * n_p, S, 2 * Hh, M.C7)
    ref = br.ref[rows[:, None], pp][:, :, None]
    x_s = ((torch.cat([sr[..., list(M.SC_ES)], act], -1) - ref) / scale
           ).float().reshape(b * n_p, S, Hh, -1)
    y_s = (br.dev_true[rows[:, None], pp] / scale).float().reshape(
        b * n_p, Hh, -1)
    return torch.cat([M.energy_score(x_e, y_e), M.energy_score(x_s, y_s)],
                     -1).view(b, n_p, -1)


def rollout_cut(ii, k, plans, xs_k, S, gen, obs):
    """model3._rollout_core with the generated tokens' inputs detached."""
    from learn.meta.data3 import E7
    st = D.stats
    iit, kt = ii.long(), k.long()
    pl = plans.to(torch.float64)
    B, P, Hh, _ = pl.shape
    NH = M.W_CTX - M.HB
    a = (kt - NH).clamp(min=0)
    Lh = kt - a
    tok_h, _, _ = M.window_tokens(D, iit, a, NH, obs_sd=obs, gen=gen)
    hist, _ = M.kv_history(net, tok_h)
    hide = torch.arange(NH, device=dev)[None] >= Lh[:, None]
    e_prev = D.E[iit, (kt - 1).clamp(min=0)] * (kt >= 1)[:, None]
    e_prev = e_prev + obs * torch.randn(e_prev.shape, generator=gen).to(dev)
    e_prev = e_prev[:, None, None].expand(B, P, S, M.C7)
    xs = xs_k.to(torch.float64)
    amax = torch.tensor([env["t_max"], env["rud_max"]], dtype=torch.float64,
                        device=dev)
    act = (xs[:, 12:14] / amax)[:, None, None].expand(B, P, S, 2)
    sr = M.plant_to_reduced_t(xs)[:, None, None].expand(B, P, S, 10)
    base = torch.randn((B, P, S, Hh, M.C7), generator=gen).to(dev)
    u_mu = torch.as_tensor(st["u_mu"], dtype=torch.float64, device=dev)
    u_sd = torch.as_tensor(st["u_sd"], dtype=torch.float64, device=dev)
    Un = ((pl - u_mu) / u_sd).float()
    e_sd = torch.as_tensor(st["e_sd"], device=dev)
    sel = torch.zeros(len(E7), 10, dtype=torch.float64, device=dev)
    sel[torch.arange(len(E7)), torch.tensor(E7)] = 1.0
    lo = torch.tensor([0.0, -1.0], dtype=torch.float64, device=dev)
    hi = torch.tensor([1.0, 1.0], dtype=torch.float64, device=dev)
    m0 = M.model0t(env)
    cache = [([], []) for _ in net.tf.layers]
    outs, states, acts = [], [], []
    for j in range(Hh):
        X = M.state_tokens_t(sr.detach(), act.detach(), st, env).float()
        first = ((Lh + j) == 0).float().view(B, 1, 1).expand(B, P, S)
        tok = M.make_tokens(X, Un[:, :, None, j].expand(B, P, S, 2),
                            e_prev.detach(), first)
        h = M.kv_step(net, tok, net.pos[Lh + j].view(B, 1, 1, -1), hist,
                      hide, cache)
        e = net.sample_grad(h, base[:, :, :, j], 24)
        e_raw = (e * e_sd).double()
        Uj = pl[:, :, None, j]
        sr = m0(sr, Uj) + (e_raw[..., :len(E7)] * dtc) @ sel
        act = torch.maximum(torch.minimum(
            Uj + e_raw[..., len(E7):len(E7) + 2] * dtc, hi), lo)
        outs.append(e)
        states.append(sr)
        acts.append(act)
        e_prev = e
    return torch.stack(outs, 3), torch.stack(states, 3), torch.stack(acts, 3)


params = list(net.parameters())


def flat_grad():
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p))
                      .reshape(-1) for p in params]).detach().clone()


# ------------------------------------------------ 1: score decomposition
print("1  validation energy score by feature group (model3.pt)")
net.eval()
es = torch.zeros(VB.N, VB.P, len(GROUPS), device=dev)
base_v = torch.randn((VB.N, VB.P, 8, VB.H, M.C7),
                     generator=torch.Generator().manual_seed(99))
allp = torch.arange(VB.P, device=dev)
with torch.no_grad():
    for s0 in range(0, VB.N, 32):
        r = torch.arange(s0, min(VB.N, s0 + 32), device=dev)
        e, sr, act = M.rollout_core(net, D, VB.ep[r], VB.k[r], VB.U[r],
                                    VB.xs0[r], env, 8, base=base_v[r.cpu()])
        es[r] = es_groups(VB, r, allp.expand(len(r), -1), e, sr, act)
wd = VB.wide
print(f"   total {es.sum(-1).mean():.2f} (full-range plans "
      f"{es[wd].sum(-1).mean():.2f}, others {es[~wd].sum(-1).mean():.2f})")
for gi, nm in enumerate(GROUPS):
    print(f"   {nm:<6} {es[..., gi].mean():6.3f}  full-range "
          f"{es[wd][:, gi].mean():6.3f}  others {es[~wd][:, gi].mean():6.3f}")

# ------------------------------------------------ 2: gradient study
print("\n2  rollout-score gradient over 12 fixed training batches")
net.train()
MODES = ("full", "cut", "cut_e", "cut_s")
G = {m: [] for m in MODES}
G["fm"] = []
tr, va = M.split_pools(D.n)
Lmax = D.len.cpu()
for bi in range(12):
    for m in MODES:
        g = torch.Generator().manual_seed(1000 + bi)
        rows = torch.randperm(TB.N, generator=g)[:16].to(dev)
        pp = torch.rand(len(rows), TB.P, generator=g).argsort(1)[:, :2].to(dev)
        obs = M.obs_sd(len(rows), dev, g)
        net.zero_grad(set_to_none=True)
        if m == "full":
            e, sr, act = M.rollout_core(
                net, D, TB.ep[rows], TB.k[rows], TB.U[rows[:, None], pp],
                TB.xs0[rows], env, 8, gen=g, grad=True, obs=obs)
        else:
            e, sr, act = rollout_cut(TB.ep[rows], TB.k[rows],
                                     TB.U[rows[:, None], pp], TB.xs0[rows],
                                     8, g, obs)
        gr = es_groups(TB, rows, pp, e, sr, act)
        sl = {"cut_e": slice(0, 10), "cut_s": slice(10, 18)}.get(
            m, slice(0, 18))
        gr[..., sl].sum(-1).mean().backward()
        G[m].append(flat_grad())
        del e, sr, act, gr
    g = torch.Generator().manual_seed(2000 + bi)
    net.zero_grad(set_to_none=True)
    ii, a = M.fm_draw(tr, 36, g, Lmax, M.W_CTX, dev)
    M.fm_loss(net, D, ii, a, M.W_CTX, g).backward()
    G["fm"].append(flat_grad())
    print(f"   batch {bi + 1}/12 done", flush=True)
mean = {m: torch.stack(v).mean(0) for m, v in G.items()}
print(f"   {'way':<6} {'|g| per batch':>14} {'|mean g|':>10} {'ratio':>7}")
for m, v in G.items():
    nb = torch.stack([x.norm() for x in v]).mean().item()
    print(f"   {m:<6} {nb:14.2f} {mean[m].norm().item():10.2f} "
          f"{mean[m].norm().item() / nb:7.3f}")
cos = lambda a, b: (a @ b / (a.norm() * b.norm() + 1e-30)).item()  # noqa
names = list(G)
print("   cosine of the mean gradients: " + ", ".join(
    f"{a}/{b} {cos(mean[a], mean[b]):+.2f}" for i, a in enumerate(names)
    for b in names[i + 1:]))
# split halves: is the mean direction reproducible?
for m in MODES:
    h1 = torch.stack(G[m][:6]).mean(0)
    h2 = torch.stack(G[m][6:]).mean(0)
    print(f"   {m}: cosine between the means of batches 1-6 and 7-12 "
          f"{cos(h1, h2):+.2f}")

# ------------------------------------------------ 3: line search
print("\n3  validation score after a step along each mean direction "
      "(96 validation moments x 4 plans, fixed samples)")
sub = torch.randperm(VB.N, generator=torch.Generator().manual_seed(5))[:96]
sub = sub.to(dev)
base_s = torch.randn((96, VB.P, 8, VB.H, M.C7),
                     generator=torch.Generator().manual_seed(7))


@torch.no_grad()
def val_sub():
    net.eval()
    tot = []
    for s0 in range(0, 96, 32):
        r = sub[s0:s0 + 32]
        e, sr, act = M.rollout_core(net, D, VB.ep[r], VB.k[r], VB.U[r],
                                    VB.xs0[r], env, 8,
                                    base=base_s[s0:s0 + 32])
        tot.append(es_groups(VB, r, allp.expand(len(r), -1), e, sr,
                             act).sum(-1))
    net.train()
    return torch.cat(tot).mean().item()


def set_theta(d, eta):
    net.load_state_dict(theta0)
    with torch.no_grad():
        off = 0
        for p in params:
            n = p.numel()
            p -= eta * d[off:off + n].view_as(p)
            off += n


v0 = val_sub()
print(f"   start {v0:.3f}")
n_par = sum(p.numel() for p in params)
for m in MODES + ("fm",):
    rms = torch.stack(G[m]).pow(2).mean(0).sqrt() + 1e-12
    for lab, d in (("plain", mean[m]), ("per-parameter", mean[m] / rms)):
        d = d / d.norm()
        out = []
        for step_rms in (1e-4, 3e-4, 1e-3):
            set_theta(d, step_rms * math.sqrt(n_par))
            out.append(val_sub() - v0)
        print(f"   {m:<6} {lab:<13} change at step size (per-parameter "
              f"rms 1e-4 / 3e-4 / 1e-3): " + " ".join(f"{x:+.3f}" for x in out))
net.load_state_dict(theta0)
