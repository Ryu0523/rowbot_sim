#!/usr/bin/env python3
"""
Which speeds and relative wave directions the learned error models were
trained on: the meta7 ('cat', the cat_p_* rows of studies/final_eval.py)
packs, read from disk (XS = plant state per control step, len = valid
steps, meta = each episode's sea). Speed u = XS[..., 6]; heading psi =
XS[..., 5]; the sea's theta0 is the PROPAGATION direction (sim/wavefield:
phase k (x cos th + y sin th) - w t; learn/repro/task.py runs head seas
with the track along +x and theta0 = pi), so the waves come from
theta0 + pi and the relative direction (0 head, 180 following) is
|wrap(theta0 + pi - psi)|. Encounter period per step from the episode's
Tp (deep water, peak component) and the step's velocity along the wave
direction; the MGN 328 band of studies/safety_events.py ([0.7, 1.15] c).

Light: XS of the train split is 190 MB (float32), read alone from the
.npz (no other array is loaded).

Output: studies/_cache/wave_regimes/training_coverage.json, printed table.

    .venv/Scripts/python.exe -m studies.wave_regime_training_coverage
"""
import json
import math
import os
import pickle
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np   # noqa: E402

CACHE = os.path.join(HERE, "_cache", "meta7")
OUT = os.path.join(HERE, "_cache", "wave_regimes")
KN = 0.514444
G = 9.81
HORIZON = 24 * 0.24


def load(split, names):
    path = os.path.join(CACHE, f"{split}.npz")
    with np.load(path) as z:
        arr = {n: z[n] for n in names}
    meta = pickle.load(open(os.path.join(CACHE, f"{split}_meta.pkl"), "rb"))
    return arr, meta


def coverage(split):
    arr, meta = load(split, ("XS", "len"))
    XS, ln = arr["XS"], arr["len"]
    n, T, _ = XS.shape
    valid = np.arange(T)[None, :] < ln[:, None]
    tp = np.array([m["sea"]["tp"] for m in meta], float)
    hs = np.array([m["sea"]["hs"] for m in meta], float)
    th0 = np.array([m["sea"]["theta0"] for m in meta], float)
    n_dir = np.array([m["sea"].get("n_dir", 0) for m in meta], int)
    world = sorted({m.get("world", "?") for m in meta})
    u = XS[..., 6].astype(float)
    v = XS[..., 7].astype(float)
    psi = XS[..., 5].astype(float)
    rel = np.abs((th0[:, None] + np.pi - psi + np.pi) % (2 * np.pi) - np.pi)
    mu = psi - th0[:, None]                    # heading minus propagation
    u_w = u * np.cos(mu) - v * np.sin(mu)      # speed along the waves
    w = 2 * np.pi / tp[:, None]
    k = w * w / G
    c = G / w
    we = np.abs(w - k * u_w)
    te = np.where(we > 1e-9, 2 * np.pi / np.maximum(we, 1e-9), np.inf)
    U = u[valid] / KN
    R = np.degrees(rel[valid])
    UW = (u_w / c)[valid]
    TE = te[valid]
    out = dict(split=split, world=world, episodes=int(n),
               steps=int(valid.sum()),
               tp=[float(tp.min()), float(tp.max())],
               hs=[float(hs.min()), float(hs.max())],
               long_crested_share=float(np.mean(n_dir == 1)),
               speed_kn_pct={str(q): float(np.percentile(U, q))
                             for q in (0, 0.1, 1, 5, 25, 50, 75, 95, 99,
                                       99.9, 100)},
               speed_share={f"<{a}": float(np.mean(U < a))
                            for a in (5, 8, 10, 12, 14, 16)})
    edges = [0, 22.5, 67.5, 112.5, 157.5, 180.01]
    names = ["head 0-22", "bow 22-67", "beam 67-112", "stern-q 112-157",
             "following 157-180"]
    out["rel_dir_share"] = {nm: float(np.mean((R >= a) & (R < b)))
                            for nm, a, b in zip(names, edges[:-1],
                                                edges[1:])}
    out["mgn328_share"] = float(np.mean((UW >= 0.7) & (UW <= 1.15)))
    out["uw_over_c_pct"] = {str(q): float(np.percentile(UW, q))
                            for q in (0.1, 1, 50, 99, 99.9, 100)}
    out["Te_gt_horizon_share"] = float(np.mean(TE > HORIZON))
    out["Te_gt_20s_share"] = float(np.mean(TE > 20.0))
    # joint: following/stern-quartering AND near the crest speed AND slow
    fol = R >= 112.5
    out["following_and_mgn"] = float(np.mean(fol & (UW >= 0.7)
                                             & (UW <= 1.15)))
    out["below_12kn_any_dir"] = float(np.mean(U < 12))
    # speed by direction bin
    out["speed_by_dir_pct"] = {
        nm: [float(np.percentile(U[(R >= a) & (R < b)], q))
             for q in (1, 50, 99)] if np.any((R >= a) & (R < b)) else None
        for nm, a, b in zip(names, edges[:-1], edges[1:])}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for split in ("train", "C"):
        if not os.path.exists(os.path.join(CACHE, f"{split}.npz")):
            print(f"{split}: missing")
            continue
        with zipfile.ZipFile(os.path.join(CACHE, f"{split}.npz")) as z:
            if "XS.npy" not in z.namelist():
                print(f"{split}: no XS")
                continue
        r = coverage(split)
        res[split] = r
        print(f"\n{split} ({r['world']}): {r['episodes']} episodes, "
              f"{r['steps']} steps; Tp {r['tp'][0]:.1f}-{r['tp'][1]:.1f} s, "
              f"Hs {r['hs'][0]:.2f}-{r['hs'][1]:.2f} m, long-crested "
              f"{r['long_crested_share']:.2f}")
        print("  speed kn percentiles: " + ", ".join(
            f"p{q} {v:.1f}" for q, v in r["speed_kn_pct"].items()))
        print("  share of steps below: " + ", ".join(
            f"{a} kn {v:.4f}" for a, v in r["speed_share"].items()))
        print("  relative direction: " + ", ".join(
            f"{a} {v:.3f}" for a, v in r["rel_dir_share"].items()))
        print("  speed (kn) p1/p50/p99 by direction: " + "; ".join(
            f"{a} {'/'.join(f'{x:.1f}' for x in v)}" if v else f"{a} -"
            for a, v in r["speed_by_dir_pct"].items()))
        print(f"  speed along waves / c percentiles: " + ", ".join(
            f"p{q} {v:.2f}" for q, v in r["uw_over_c_pct"].items()))
        print(f"  MGN 328 band share {r['mgn328_share']:.4f} "
              f"(following/stern-quartering and in band "
              f"{r['following_and_mgn']:.4f}); T_e > horizon "
              f"{r['Te_gt_horizon_share']:.3f}; T_e > 20 s "
              f"{r['Te_gt_20s_share']:.4f}")
    with open(os.path.join(OUT, "training_coverage.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nsaved {os.path.join(OUT, 'training_coverage.json')}")


if __name__ == "__main__":
    main()
