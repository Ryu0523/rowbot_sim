#!/usr/bin/env python3
"""
Diagnostics for the preview-value study (studies/preview_value.py):

A. Are impacts tied to the height of the wave being met? From runs with the
   pose recorded (preview runs; those with no trigger are constant-speed
   runs), the elevation the bow met is rebuilt, cut into zero-crossing
   waves, and each filtered CG-acceleration peak is assigned to the wave
   under the bow at its time. Reported: the distribution of H / Hs for the
   waves that produced a peak >= 3 g (and >= 2 g) against all waves; the
   hit rate (fraction of waves above a height threshold that produced a
   >= 3 g peak) and the coverage (fraction of >= 3 g peaks that came from
   waves above the threshold), for several thresholds; the same for the
   preceding wave and for the pair (max of the two).
B. Does the speed fall in time? For triggered runs, the speed at the
   trigger onset and 2 / 4 / 6 s later, the lead to the dangerous crest at
   onset, and the speed when that crest arrives.

    python -m studies.preview_diag [run name glob]
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from studies.bow_camera import Surface, POSE_DT     # noqa: E402

RUNS = os.path.join(ROOT, "studies", "_cache", "preview", "runs")
KN = 0.514444


def bow_series(meta, z):
    surf = Surface(z["sea_a"], z["sea_w"], z["sea_th"], z["sea_phi"])
    pose = np.asarray(z["pose"], float)
    dt = float(meta["dt"])
    t = dt * (1 + 5 * np.arange(len(pose)))
    x_st = float(meta["x_station"])
    xb = pose[:, 0] + x_st * np.cos(pose[:, 5])
    yb = pose[:, 1] + x_st * np.sin(pose[:, 5])
    return t, surf.eta(xb, yb, t), pose


def zero_crossing_waves(t, e):
    """[(t_start, t_end, H, t_crest)] between consecutive up-crossings."""
    up = np.flatnonzero((e[:-1] < 0) & (e[1:] >= 0)) + 1
    out = []
    for i0, i1 in zip(up[:-1], up[1:]):
        seg = e[i0:i1]
        out.append((t[i0], t[i1], float(seg.max() - seg.min()),
                    t[i0 + int(np.argmax(seg))]))
    return out


def peaks_with_times(acc, dt, level):
    """Times and values of positive-excursion peaks >= level of the series
    `acc` (one per excursion above zero)."""
    pos = acc > 0
    edges = np.flatnonzero(np.diff(pos.astype(int)))
    starts = edges[pos[edges + 1]] + 1 if len(edges) else np.array([], int)
    out = []
    for st in starts:
        en = st + int(np.argmax(~pos[st:])) if (~pos[st:]).any() else len(acc)
        j = st + int(np.argmax(acc[st:en]))
        if acc[j] >= level:
            out.append((dt * (j + 1), float(acc[j])))
    return out


def part_a(names, settle=60.0):
    rows_all, rows_imp2, rows_imp3 = [], [], []
    pair_all, pair_imp3 = [], []
    n_waves = 0
    for nm in names:
        with open(os.path.join(RUNS, nm + ".json")) as f:
            meta = json.load(f)
        z = np.load(os.path.join(RUNS, nm + ".npz"))
        hs = float(meta["hs_eff"])
        t, e, pose = bow_series(meta, z)
        waves = [w for w in zero_crossing_waves(t, e) if w[0] > settle]
        n_waves += len(waves)
        H = np.array([w[2] for w in waves]) / hs
        Hprev = np.r_[np.nan, H[:-1]]
        rows_all.extend(H.tolist())
        pair_all.extend(np.fmax(H, Hprev).tolist())
        acc = np.asarray(z["acc_f"], float)[:, 0]
        dt = float(meta["dt"])
        starts = np.array([w[0] for w in waves])
        for lev, store, pstore in ((2.0, rows_imp2, None),
                                   (3.0, rows_imp3, pair_imp3)):
            for tp, a in peaks_with_times(acc, dt, lev):
                if tp <= settle:
                    continue
                # the wave under the bow: the bow meets wave i between
                # starts[i] and starts[i+1]; the CG is x_st behind, so the
                # CG feels it a little later -- assign by the bow time
                # minus the travel of the bow-CG distance
                u = float(np.interp(tp, dt * (1 + 5 * np.arange(len(pose))),
                                    pose[:, 6]))
                lag = float(meta["x_station"]) / max(u + 7.8, 1.0)
                i = int(np.searchsorted(starts, tp - lag)) - 1
                if 0 <= i < len(H):
                    store.append(H[i])
                    if pstore is not None:
                        pstore.append(max(H[i], H[i - 1] if i > 0 else 0.0))
    A, I2, I3 = (np.asarray(v) for v in (rows_all, rows_imp2, rows_imp3))
    PA, PI3 = np.asarray(pair_all), np.asarray(pair_imp3)
    print(f"\nA. {len(names)} runs, {n_waves} waves, {len(I2)} peaks >= 2 g, "
          f"{len(I3)} peaks >= 3 g")
    print("   H/Hs quantiles        all waves: "
          + " ".join(f"{q:.2f}" for q in np.quantile(A, [0.5, 0.75, 0.9,
                                                        0.97])))
    if len(I2):
        print("   H/Hs quantiles   waves w/ >=2 g: "
              + " ".join(f"{q:.2f}" for q in np.quantile(I2, [0.5, 0.75,
                                                             0.9, 0.97])))
    if len(I3):
        print("   H/Hs quantiles   waves w/ >=3 g: "
              + " ".join(f"{q:.2f}" for q in np.quantile(I3, [0.5, 0.75,
                                                             0.9, 0.97])))
    print("   threshold  frac of waves above  hit rate (>=3 g | above)  "
          "coverage (above | >=3 g)   [pair = max(this, previous)]")
    for thr in (0.8, 1.0, 1.2, 1.3, 1.5):
        above = A > thr
        hit = (I3 > thr).sum() / max(above.sum(), 1)
        cov = (I3 > thr).mean() if len(I3) else float("nan")
        pabove = PA > thr
        phit = (PI3 > thr).sum() / max(pabove.sum(), 1)
        pcov = (PI3 > thr).mean() if len(PI3) else float("nan")
        print(f"   {thr:4.1f}       {above.mean():5.2f}             "
              f"{hit:5.2f}                 {cov:5.2f}          "
              f"[pair {pabove.mean():.2f} / {phit:.2f} / {pcov:.2f}]")


def part_b(names):
    print("\nB. speed response after trigger onsets")
    for nm in names:
        with open(os.path.join(RUNS, nm + ".json")) as f:
            meta = json.load(f)
        z = np.load(os.path.join(RUNS, nm + ".npz"))
        lg = np.asarray(z["trig_log"], float)       # t, trig, metric, lead, v
        if lg[:, 1].sum() == 0:
            continue
        t_slow, u = np.asarray(z["t_slow"], float), np.asarray(z["u"], float)
        on = np.flatnonzero((lg[1:, 1] > 0) & (lg[:-1, 1] == 0)) + 1
        on = on[lg[on, 0] > 60.0]
        if len(on) == 0:
            continue
        t_on = lg[on, 0]
        lead = lg[on, 3]
        u_on = np.interp(t_on, t_slow, u) / KN
        d = {k: np.interp(t_on + k, t_slow, u) / KN - u_on for k in (2, 4, 6)}
        u_crest = np.interp(t_on + lead, t_slow, u) / KN
        print(f"   {nm}: {len(on)} onsets; lead median {np.median(lead):.1f} s"
              f" (10-90% {np.percentile(lead, 10):.1f}-"
              f"{np.percentile(lead, 90):.1f}); speed at onset "
              f"{u_on.mean():.1f} kn; change after 2/4/6 s "
              f"{d[2].mean():+.1f}/{d[4].mean():+.1f}/{d[6].mean():+.1f} kn; "
              f"speed when the crest arrives {u_crest.mean():.1f} kn "
              f"(10-90% {np.percentile(u_crest, 10):.1f}-"
              f"{np.percentile(u_crest, 90):.1f})")


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "*"
    names = [os.path.basename(p)[:-5] for p in
             sorted(glob.glob(os.path.join(RUNS, pat + ".json")))]
    const = []
    for nm in names:
        with open(os.path.join(RUNS, nm + ".json")) as f:
            m = json.load(f)
        if m["stats"] and m["stats"]["n_slow"] == 0:
            const.append(nm)
    print("constant-speed (untriggered) runs:", const)
    if const:
        part_a(const)
    part_b(names)
