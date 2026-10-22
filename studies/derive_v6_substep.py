#!/usr/bin/env python3
"""Check V6 of learn/meta/HULL_FORCE_AUDIT_REVIEW_2026-10-09.md (section 4;
HULL_FORCE_DERIVATION section 3.5, check C6): is the "peak acceleration"
statistic a property of the integrator? Integration step and a fixed output
low-pass filter on the Fridsma harness.

Cases (studies/exp_planing.py harness, exactly as exp_sim_redesign2 uses it:
fridsma_hull / fridsma71_hull with tow="cg", MODEL scale -- beam 9 in, hull
1.143 m, speeds in m/s model scale, the plant stepped at model-scale dt):
  F69B3   Fridsma 1969 model B, regular head waves lam/L = 3, H = 0.111 b
          (the resonance case). Statistic (exp_sim_redesign2.rw_job): the
          mean over encounter cycles of the per-cycle maximum upward
          acceleration at the CG and at the bow gauge (0.1 L aft of the
          stem), in g, over the second half of a 16 s run.
  F71c30  Fridsma 1971 condition 30 (V/sqrt(L) 6, LCG 64 %, H1/3 = 0.444 b),
          irregular head seas, 40 components, SEED FIXED (seed = 30, the
          harness's seed_k 0) so the wave realisation is the same for every
          dt. Statistic (f71_job): mean of the positive excursion peaks of
          the CG and bow accelerations (Fridsma's "mean of all positive
          peaks"), first 10 % of the 60 s run dropped. Also A1/10 (mean of
          the highest tenth of the excursion peaks) and the maximum.
Configurations: ZR (Wagner sections) and ZR+MLM.

Time step. The harness runs at model scale, dt 0.004 s. The derivation's
full-scale steps 0.02, 0.01, 0.005, 0.0025 s are converted with the Froude
time scale sqrt(b_full / b_model) = sqrt(2.025 / 0.2286) = 2.976 (the
ledger's "model 0.004 s = full 0.012 s" used 3): model dt 0.00672, 0.00336,
0.00168, 0.00084 s.

Filter. 2nd-order Butterworth, forward-backward (scipy.signal.filtfilt),
cut-off 20 Hz full scale = 59.5 Hz model scale; second cut-off 10 Hz full =
29.8 Hz model. Applied to the CG and bow acceleration series before any
statistic.

Run: python -m studies.derive_v6_substep
Outputs: studies/_cache/derive/v6_substep.json / .txt, series in
studies/_cache/derive/v6_series/*.npz
"""
import json
import math
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "studies", "_cache", "derive")
SER = os.path.join(OUT, "v6_series")
os.makedirs(SER, exist_ok=True)

G = 9.81
B_FULL, B_MODEL = 2.025, 9 * 0.0254
SCALE_T = math.sqrt(B_FULL / B_MODEL)             # Froude time scale 2.976
DT_FULL = (0.02, 0.01, 0.005, 0.0025)
FC_FULL = (20.0, 10.0)                             # Hz, full scale
CASES = ("F69B3", "F71c30")
CFGS = ("ZR", "ZRM")


def lowpass(y, dt, fc_model):
    from scipy.signal import butter, filtfilt
    fs = 1.0 / dt
    wn = fc_model / (0.5 * fs)
    if wn >= 0.999:
        return y.copy()           # cut-off above Nyquist: nothing to remove
    b, a = butter(2, wn)
    return filtfilt(b, a, y, axis=0)


def stats_f69(q, dt, om_e):
    """q: (n, 2) CG and bow acceleration (g) over the analysed window."""
    te = 2 * np.pi / om_e
    m = max(int(round(te / dt)), 1)
    cyc = [q[j:j + m] for j in range(0, len(q) - m + 1, m)]
    pk = np.array([[c[:, 0].max(), c[:, 1].max()] for c in cyc])
    return dict(cg_cycmean=float(pk[:, 0].mean()),
                bow_cycmean=float(pk[:, 1].mean()),
                cg_max=float(q[:, 0].max()), bow_max=float(q[:, 1].max()),
                n_cyc=len(cyc))


def stats_f71(q):
    from studies.exp_planing import excursion_peaks
    out = {}
    for j, nm in ((0, "cg"), (1, "bow")):
        pk = np.sort(excursion_peaks(q[:, j]))[::-1]
        n10 = max(1, len(pk) // 10)
        out[nm + "_mean"] = float(pk.mean())
        out[nm + "_a110"] = float(pk[:n10].mean())
        out[nm + "_max"] = float(q[:, j].max())
        out[nm + "_n"] = int(len(pk))
    return out


def job(args):
    case, cfg, dt_full = args
    from studies.exp_sim_redesign2 import make_plant, ZRM
    from sim.planing_zarnick import ZOPT_ZR
    kind = ZOPT_ZR if cfg == "ZR" else ZRM
    dt = dt_full / SCALE_T
    if case == "F69B3":
        from studies.exp_planing import fridsma_hull
        from sim.test_vessel import Monochromatic
        hull, V = fridsma_hull("B", tow="cg")
        T = 16.0
        lam = 3.0
        H = 0.111 * hull.b
        k = 2 * np.pi / (lam * hull.L)
        a = 0.5 * H
        om = np.sqrt(G * k)
        sea = Monochromatic(om, a, theta0=np.pi)
        om_e = om + k * V
    else:
        from studies.exp_planing import fridsma71_hull
        from sim.wavefield import SeaState
        hull, V, hs = fridsma71_hull(30, tow="cg")
        T = 60.0
        tp = 15.63 * np.sqrt(hs / G)
        sea = SeaState(hs, tp, gamma=1.0, theta0=np.pi, n_freq=40, n_dir=1,
                       seed=30)
        om_e = None
    p = make_plant(kind, hull, sea, dt, V)
    s = np.zeros(14)
    s[6] = V
    s[2], s[4] = p.running_attitude(V)
    xa = hull.L - hull.lcg - 0.1 * hull.L
    xb = p.x[-1]
    n = int(round(T / dt))
    acc = np.empty((n, 3))
    t0 = time.perf_counter()
    for i in range(n):
        s = p.step(s, i * dt, 0.0, 0.0)
        s[6] = V
        ac, ab = p.last_cg_acc, p.last_bow_acc
        bw = (ac + (ab - ac) * xa / xb) / G
        acc[i] = ac / G, bw, bw * math.cos(p.Th_rest + s[4])
    wall = time.perf_counter() - t0
    if not np.all(np.isfinite(acc)):
        return dict(case=case, cfg=cfg, dt_full=dt_full, bad=True)
    np.savez_compressed(os.path.join(SER, f"{case}_{cfg}_dt{dt_full}.npz"),
                        acc=acc.astype(np.float32), dt=dt, V=V)
    win = acc[n // 2:] if case == "F69B3" else acc[n // 10:]
    res = dict(case=case, cfg=cfg, dt_full=dt_full, dt_model=dt, n_steps=n,
               wall_s=wall, V=float(V), raw=None, filt={})
    for tag, fc in (("raw", None),) + tuple((f"lp{int(f)}", f)
                                            for f in FC_FULL):
        q = win[:, :2] if fc is None else lowpass(win[:, :2], dt,
                                                   fc * SCALE_T)
        st = stats_f69(q, dt, om_e) if case == "F69B3" else stats_f71(q)
        if fc is None:
            res["raw"] = st
        else:
            res["filt"][tag] = dict(st, fc_full=fc, fc_model=fc * SCALE_T)
    return res


def report(rows):
    L = []
    L.append(f"V6: Fridsma harness at MODEL scale; dt_model = dt_full / "
             f"{SCALE_T:.3f}; filter 2nd-order Butterworth filtfilt, cut-off "
             f"20 Hz full = {20 * SCALE_T:.1f} Hz model, 10 Hz full = "
             f"{10 * SCALE_T:.1f} Hz model")
    by = {(r["case"], r["cfg"], r["dt_full"]): r for r in rows
          if isinstance(r, dict) and not r.get("bad")}
    keys = {"F69B3": [("cg_cycmean", "CG mean of per-cycle peaks"),
                      ("bow_cycmean", "bow mean of per-cycle peaks"),
                      ("cg_max", "CG max"), ("bow_max", "bow max")],
            "F71c30": [("cg_mean", "CG mean of positive peaks"),
                       ("bow_mean", "bow mean of positive peaks"),
                       ("cg_a110", "CG A1/10"), ("bow_a110", "bow A1/10"),
                       ("cg_max", "CG max"), ("bow_max", "bow max")]}
    conv = {}
    for case in CASES:
        for cfg in CFGS:
            L.append("")
            L.append(f"{case} {cfg}: dt_full " + "  ".join(
                f"{d}" for d in DT_FULL) + "   (model dt " + ", ".join(
                f"{d / SCALE_T:.5f}" for d in DT_FULL) + ")")
            walls = [by.get((case, cfg, d), {}).get("wall_s") for d in DT_FULL]
            L.append("  compute s   " + "  ".join(
                f"{w:6.0f}" if w else "   nan" for w in walls))
            if case == "F71c30":
                ns = [by.get((case, cfg, d), {}).get("raw", {}).get("cg_n")
                      for d in DT_FULL]
                L.append("  n peaks CG  " + "  ".join(
                    f"{x:6d}" if x else "   nan" for x in ns))
            for key, name in keys[case]:
                for tag in ("raw", "lp20", "lp10"):
                    vals = []
                    for d in DT_FULL:
                        r = by.get((case, cfg, d))
                        if r is None:
                            vals.append(float("nan"))
                        elif tag == "raw":
                            vals.append(r["raw"][key])
                        else:
                            vals.append(r["filt"][tag][key])
                    vals = np.array(vals)
                    ref = vals[-1]
                    rng = (vals.max() - vals.min()) / ref * 100 if ref else 0
                    # dt at which the statistic is within 2 % of the finest
                    ok = [d for d, v in zip(DT_FULL, vals)
                          if abs(v / ref - 1) <= 0.02]
                    dconv = max(ok) if ok else None
                    conv[(case, cfg, key, tag)] = dict(
                        vals=vals.tolist(), range_pct=float(rng),
                        dt_conv_full=dconv,
                        coarse_vs_fine_pct=float((vals[0] / ref - 1) * 100))
                    L.append(f"  {name:<30s} {tag:<5s} " + "  ".join(
                        f"{v:6.3f}" for v in vals) +
                        f"   spread {rng:5.1f} %  coarse/fine "
                        f"{(vals[0] / ref - 1) * 100:+5.1f} %  2 %-converged "
                        f"from dt_full {dconv}")
    L.append("")
    L.append("summary: how much the statistic moves over dt_full 0.02 -> "
             "0.0025 s (spread / finest value), raw vs filtered 20 Hz")
    for case in CASES:
        for cfg in CFGS:
            parts = []
            for key, name in keys[case]:
                a = conv[(case, cfg, key, "raw")]
                b = conv[(case, cfg, key, "lp20")]
                parts.append(f"{key} raw {a['range_pct']:.1f} % / lp20 "
                             f"{b['range_pct']:.1f} %")
            L.append(f"  {case} {cfg}: " + "; ".join(parts))
    return "\n".join(L), conv


if __name__ == "__main__":
    from studies.parallel_runner import run_parallel
    jobs = [(c, k, d) for c in CASES for k in CFGS for d in DT_FULL]
    jobs.sort(key=lambda j: j[2])        # finest (longest) first
    t0 = time.time()
    rows = run_parallel(job, jobs, gb_per_job=0.2, reserve_gb=1.5,
                        max_procs=8)
    rows = [r if isinstance(r, dict) else dict(err=repr(r), case=j[0],
                                                cfg=j[1], dt_full=j[2])
            for j, r in zip(jobs, rows)]
    txt, conv = report(rows)
    txt += f"\nwall {time.time() - t0:.0f} s"
    print(txt)
    open(os.path.join(OUT, "v6_substep.txt"), "w").write(txt + "\n")
    json.dump(dict(rows=rows, conv={"|".join(map(str, k)): v
                                    for k, v in conv.items()},
                   scale_t=SCALE_T, wall_s=time.time() - t0),
              open(os.path.join(OUT, "v6_substep.json"), "w"), indent=1,
              default=float)
