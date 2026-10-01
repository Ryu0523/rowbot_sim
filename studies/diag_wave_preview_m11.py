#!/usr/bin/env python3
"""
Are the 15 elevations at the MPC's stations (the hull footprint, at the
moment of prediction) the right wave information for the target's one-step
errors, or is the wave the hull meets DURING the step (the water ahead of
the bow at planing speed) what matters? Descriptive (DEFECTS M11): the
target episodes' seas are rebuilt from their meta (checked against the
recorded elevations), and the one-step errors of pitch rate, yaw rate and
sway (step t -> t + 1) are correlated with

  now     wave features under the hull at step t (what the 15 points give)
  mid     the same half a step later, at the midpoint of the hull's path
  next    at step t + 1, where the hull really is then
  ahead   centreline elevations 1, 2, 4, 8 m ahead of the bow at step t

Features: longitudinal slope (bow minus stern centre over their distance),
transverse slope (starboard minus port, mean over stations), twist
(transverse slope weighted by station position), mean elevation.

    python studies/diag_wave_preview_m11.py
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from learn.meta import relabel

C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache", "meta2")
env = relabel._env()
dtc = env["dt"] * env["sub"]
x_st, y_off = np.asarray(env["x_st"]), np.asarray(env["y_off"])
AHEAD = (1.0, 2.0, 4.0, 8.0)


def elev(sea, X, Y, t):
    """Elevation of one SeaState at points X, Y (any shape) at time t (same
    shape or scalar)."""
    ph = sea.k * (X[..., None] * np.cos(sea.th) + Y[..., None] * np.sin(sea.th)) \
        - sea.w * np.asarray(t)[..., None] + sea.phi
    return (sea.a * np.cos(ph)).sum(-1)


def grid(x, y, psi, xs, ys):
    """World coordinates of hull-frame points (xs along, ys across) for
    poses (n,): (n, len(xs), len(ys))."""
    c, s = np.cos(psi)[:, None, None], np.sin(psi)[:, None, None]
    X = x[:, None, None] + xs[None, :, None] * c - ys[None, None, :] * s
    Y = y[:, None, None] + xs[None, :, None] * s + ys[None, None, :] * c
    return X, Y


def feats(E3):
    """E3 (n, 5, 3) elevations at the stations -> dict of features (n,)."""
    cen = E3[:, :, 1]
    tsl = (E3[:, :, 2] - E3[:, :, 0])
    xc = x_st - x_st.mean()
    return dict(long=(cen[:, -1] - cen[:, 0]) / (x_st[-1] - x_st[0]),
                trans=tsl.mean(1), twist=(tsl * xc).sum(1) / (xc ** 2).sum(),
                mean=cen.mean(1))


rows = {}
for split in ("C", "Cb"):
    d = np.load(os.path.join(C, f"{split}.npz"))
    e0 = np.load(os.path.join(C, f"{split}_e0.npz"))["E0"]
    meta = pickle.load(open(os.path.join(C, f"{split}_meta.pkl"), "rb"))
    L = d["len"]
    T = d["U"].shape[1]
    n_e = L - 1 if d["XS"].shape[1] == T else L
    acc = {k: [] for k in ("thd", "yaw", "sway")}
    F = {}
    chk = []
    for i in range(len(L)):
        n = int(n_e[i])
        if n < 80:
            continue
        sea = relabel._sea(meta[i])
        t_idx = np.arange(40, n - 1)
        XS = d["XS"][i].astype(float)
        tt = t_idx * dtc
        x, y, psi = XS[t_idx, 0], XS[t_idx, 1], XS[t_idx, 5]
        X, Y = grid(x, y, psi, x_st, y_off)
        now = elev(sea, X, Y, tt[:, None, None])
        rec = d["S"][i, t_idx, 11:26].reshape(-1, 5, 3)
        chk.append(np.abs(now - rec).max() / (np.abs(rec).max() + 1e-9))
        x1, y1, p1 = XS[t_idx + 1, 0], XS[t_idx + 1, 1], XS[t_idx + 1, 5]
        Xm, Ym = grid(0.5 * (x + x1), 0.5 * (y + y1), psi + 0.5 * np.angle(
            np.exp(1j * (p1 - psi))), x_st, y_off)
        mid = elev(sea, Xm, Ym, (tt + 0.5 * dtc)[:, None, None])
        Xn, Yn = grid(x1, y1, p1, x_st, y_off)
        nxt = elev(sea, Xn, Yn, (tt + dtc)[:, None, None])
        Xa, Ya = grid(x, y, psi, x_st[-1] + np.array(AHEAD), np.zeros(1))
        ahead = elev(sea, Xa, Ya, tt[:, None, None])[:, :, 0]
        for when, E3 in (("now", now), ("mid", mid), ("next", nxt)):
            for k, v in feats(E3).items():
                F.setdefault(f"{when}:{k}", []).append(v)
        for j, a in enumerate(AHEAD):
            F.setdefault(f"ahead {a:g} m", []).append(ahead[:, j])
        acc["thd"].append(e0[i, t_idx, 4])
        acc["yaw"].append(e0[i, t_idx, 2])
        acc["sway"].append(e0[i, t_idx, 1])
    print(f"{split}: rebuilt sea vs recorded elevations, worst relative "
          f"error {max(chk):.2e}")
    rows[split] = ({k: np.concatenate(v) for k, v in acc.items()},
                   {k: np.concatenate(v) for k, v in F.items()})


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-30))


for split, (A, F) in rows.items():
    print(f"\n{split}: correlation of the one-step error (step t -> t+1) with")
    names = [k for k in F]
    print("   " + " " * 16 + "   pitch   yaw  sway")
    for k in names:
        print(f"   {k:<16} " + " ".join(f"{corr(A[c], F[k]):+6.2f}"
                                       for c in ("thd", "yaw", "sway")))
