#!/usr/bin/env python3
"""
Follow-up to diag_m9_grad.py: the validation rollout score of model3.pt
jumped from ~55 to >110 after one step of per-parameter size 1e-4 along
ANY mean gradient, the FM one included. Which feature groups blow up, from
what step size on, and what do the samples look like there?

  a  mean gradients (8 fixed training batches) of the rollout score (full
     chain) and of the FM loss, per-parameter normalised like Adam, plus a
     random direction of the same per-parameter size
  b  steps of per-parameter size 1e-6 .. 1e-4 along each: total and
     per-group validation score (96 moments x 4 plans, fixed samples)
  c  per error channel: the largest |e| of the samples and the share
     beyond 10 training sd, at model3.pt and after the 1e-4 FM step
  d  the one-step FM validation loss after the same steps (recorded
     windows of the validation episodes), to compare the sensitivities

    python studies/diag_m9_sharp.py [--cache-name meta3]
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
params = list(net.parameters())
n_par = sum(p.numel() for p in params)
tr, va = M.split_pools(D.n)
Lmax = D.len.cpu()


def es_groups(br, rows, pp, e, sr, act):
    """model3.es_value per feature group: (b, np, 18)."""
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


def flat_grad():
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p))
                      .reshape(-1) for p in params]).detach().clone()


# ---------------------------------------------------- a: mean gradients
net.train()
GE, GF = [], []
for bi in range(8):
    g = torch.Generator().manual_seed(1000 + bi)
    rows = torch.randperm(TB.N, generator=g)[:16].to(dev)
    pp = torch.rand(len(rows), TB.P, generator=g).argsort(1)[:, :2].to(dev)
    net.zero_grad(set_to_none=True)
    e, sr, act = M.rollout_core(net, D, TB.ep[rows], TB.k[rows],
                                TB.U[rows[:, None], pp], TB.xs0[rows], env,
                                8, gen=g, grad=True,
                                obs=M.obs_sd(len(rows), dev, g))
    es_groups(TB, rows, pp, e, sr, act).sum(-1).mean().backward()
    GE.append(flat_grad())
    del e, sr, act
    g = torch.Generator().manual_seed(2000 + bi)
    net.zero_grad(set_to_none=True)
    ii, a = M.fm_draw(tr, 36, g, Lmax, M.W_CTX, dev)
    M.fm_loss(net, D, ii, a, M.W_CTX, g).backward()
    GF.append(flat_grad())
DIRS = {}
for nm, Gs in (("rollout score", GE), ("FM", GF)):
    Gs = torch.stack(Gs)
    d = Gs.mean(0) / (Gs.pow(2).mean(0).sqrt() + 1e-12)
    DIRS[nm] = d / d.pow(2).mean().sqrt()          # per-parameter rms 1
r = torch.randn(n_par, generator=torch.Generator().manual_seed(3)).to(dev)
DIRS["random"] = r / r.pow(2).mean().sqrt()
del GE, GF

# ---------------------------------------------------- b, c: steps
sub = torch.randperm(VB.N, generator=torch.Generator().manual_seed(5))[:96]
sub = sub.to(dev)
base_s = torch.randn((96, VB.P, 8, VB.H, M.C7),
                     generator=torch.Generator().manual_seed(7))
allp = torch.arange(VB.P, device=dev)


@torch.no_grad()
def val_sub():
    net.eval()
    tot, es_all = [], []
    for s0 in range(0, 96, 32):
        rr = sub[s0:s0 + 32]
        e, sr, act = M.rollout_core(net, D, VB.ep[rr], VB.k[rr], VB.U[rr],
                                    VB.xs0[rr], env, 8,
                                    base=base_s[s0:s0 + 32])
        tot.append(es_groups(VB, rr, allp.expand(len(rr), -1), e, sr, act))
        es_all.append(e.reshape(-1, M.C7))
    net.train()
    return torch.cat(tot).mean((0, 1)), torch.cat(es_all)


def set_theta(d, size):
    net.load_state_dict(theta0)
    with torch.no_grad():
        off = 0
        for p in params:
            n = p.numel()
            p -= size * d[off:off + n].view_as(p)
            off += n


g0, e0 = val_sub()
fm0 = M.fm_validate(net, D, va, M.W_CTX, Lmax)
print(f"model3.pt: validation rollout score {g0.sum():.3f}, FM validation "
      f"{fm0:.4f}")
SIZES = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
e_blow = None
for nm, d in DIRS.items():
    print(f"\nalong the {nm} direction (per-parameter step size):")
    for sz in SIZES:
        set_theta(d, sz)
        gs, es_ = val_sub()
        fm = M.fm_validate(net, D, va, M.W_CTX, Lmax)
        dg = gs - g0
        top = torch.argsort(dg.abs(), descending=True)[:4]
        print(f"   {sz:.0e}: score {gs.sum():8.3f} ({dg.sum():+8.3f}), FM "
              f"{fm:.4f} ({fm - fm0:+.4f}); largest changes " + ", ".join(
                  f"{GROUPS[i]} {dg[i]:+.2f}" for i in top.tolist()))
        if nm == "FM" and sz == 1e-4:
            e_blow = es_
net.load_state_dict(theta0)

print("\nsamples per error channel (normalised units): largest |e|, share "
      "beyond 10 sd -- model3.pt / after the 1e-4 FM step")
for c in range(M.C7):
    a0, a1 = e0[:, c].abs(), e_blow[:, c].abs()
    print(f"   {GROUPS[c]:<6} max {a0.max():9.2f} / {a1.max():9.2f}   "
          f">10 sd {(a0 > 10).float().mean():.5f} / "
          f"{(a1 > 10).float().mean():.5f}")
