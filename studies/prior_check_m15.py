#!/usr/bin/env python3
"""
Prior predictive check of the operator family (DEFECTS M15): per-episode
descriptive statistics of the one-step error e0, computed the same way on
episodes drawn from the prior (the source splits train / A of a cache) and on
the target (C, Cb). For each statistic: where the target's episodes fall in
the prior's per-episode distribution (percentiles), i.e. whether the prior
puts its mass where the target is. Statistics are per episode (a pooled
correlation of a sign-symmetric prior is ~0 by construction and says
nothing). Descriptive only: no statement about information or ceilings.

  rms_c        error size per channel (surge, sway, yaw, heave, pitch)
  sync_sy      same-step correlation of sway and yaw errors (a force at a
               point moves both)
  sync_hp      same-step correlation of heave-rate and pitch-rate errors
  damp_c       correlation of a channel's error with its own velocity
               (surge: speed minus its episode mean; sway v; yaw r; heave w;
               pitch q): negative = the error acts like extra damping
  slope_th     least-squares slope of the pitch-rate error on the pitch angle
               (per episode, in error sd per angle sd)
  slope_u      same on the speed
  ac1_c        lag-1 autocorrelation per channel

    python studies/prior_check_m15.py [--cache meta3]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="meta3")
ap.add_argument("--n-prior", type=int, default=2000)
args = ap.parse_args()
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache", args.cache)
CH = ("surge", "sway", "yaw", "heave", "pitch")
# plant state columns (14): x y z phi th psi u v w p q r thr noz
VEL = (6, 7, 11, 8, 10)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-30))


def slope(y, x):
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / (x @ x + 1e-30) * (x.std() + 1e-30) / (y.std() + 1e-30))


def stats_of(split, n_max):
    d = np.load(os.path.join(C, f"{split}.npz"))
    e0 = np.load(os.path.join(C, f"{split}_e0.npz"))["E0"]
    L = d["len"]
    T = d["U"].shape[1]
    n_e = L - 1 if d["XS"].shape[1] == T else L
    rows = []
    for i in range(min(n_max, len(n_e))):
        n = int(n_e[i])
        if n < 100:
            continue
        e = e0[i, 40:n, :5].astype(float)
        xs = d["XS"][i, 40:n].astype(float)
        if not np.isfinite(e).all() or e.std(0).min() < 1e-9:
            continue
        r = {}
        for c, nm in enumerate(CH):
            r[f"rms_{nm}"] = float(np.sqrt((e[:, c] ** 2).mean()))
            r[f"ac1_{nm}"] = corr(e[1:, c], e[:-1, c])
            r[f"damp_{nm}"] = corr(e[:, c], xs[:, VEL[c]])
        r["sync_sy"] = corr(e[:, 1], e[:, 2])
        r["sync_hp"] = corr(e[:, 3], e[:, 4])
        r["slope_th"] = slope(e[:, 4], xs[:, 4])
        r["slope_u"] = slope(e[:, 4], xs[:, 6])
        rows.append(r)
    return rows


prior = stats_of("train", args.n_prior) + stats_of("A", 300)
target = stats_of("C", 64) + stats_of("Cb", 64)
keys = list(prior[0])
print(f"prior predictive check ({args.cache}): {len(prior)} prior episodes, "
      f"{len(target)} target episodes")
print(f"  {'statistic':<11} {'prior 5% / 50% / 95%':>24}   {'target median':>13}"
      f"   target episodes' median percentile in the prior (share below 5% / "
      f"above 95%)")
for k in keys:
    p = np.array([r[k] for r in prior])
    t = np.array([r[k] for r in target])
    q = np.quantile(p, [0.05, 0.5, 0.95])
    pct = np.array([(p < v).mean() for v in t]) * 100
    print(f"  {k:<11} {q[0]:+8.3f} {q[1]:+8.3f} {q[2]:+8.3f}   {np.median(t):+13.3f}"
          f"   {np.median(pct):5.1f}%  ({(pct < 5).mean():.2f} / "
          f"{(pct > 95).mean():.2f})")
