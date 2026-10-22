#!/usr/bin/env python3
"""
TIMING of the bow burials in following / quartering seas above the wave
speed (the hazard the 'ease off over the crest' rules of
studies/wave_regime_oracle.py act on): how long after the bow passes a crest
(the bow place ph of studies/wave_regime_oracle.wave_at crossing 0 upward,
i.e. the bow moving from the back of the wave ahead onto its front face)
does a burial start, compared with the jet's thrust response (0.1 s delay +
0.6 s first-order lag, sim/planing_vessel.py:129-133)?

Constant throttle runs built exactly as studies/wave_regime_oracle.run_one
(the same make_mission, start speed, autopilot, SafeMode); per burial onset
(studies/safety_events bow-wet spells deeper than BURY_FRAC f0 or longer
than WET_LONG): the time since the last upward crest crossing at the bow and
the bow place at the onset; the encounter period at the bow (zero
up-crossings of the surface there).

  python -m studies.wave_regime_oracle_timing --workers 2
Output: studies/_cache/wave_regimes/oracle_timing.txt, oracle_timing.pkl
"""
import argparse
import math
import os
import pickle
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np   # noqa: E402

OUT = os.path.join(HERE, "_cache", "wave_regimes")
PI = math.pi


def run(args):
    cell, f, seed, T = args
    from studies import final_eval as FE
    from studies import safety_events as se
    from studies.wave_phase_oracle import SafeMode
    from studies import wave_regime_oracle as O
    c = O.ALL_CELLS[cell]
    sd = dict(O.SEAS[c["sea"]])
    sd["theta0"] = float(O.wrap(PI - math.radians(c["chi"])))
    m = FE.make_mission(seed, 0, T, sea=sd)
    pl = m.plant
    m.s = pl.initial_state(O.u_calm(f) * O.KN)
    m.s[5] = m.phi
    safe = SafeMode(m.dt * m.sub)
    while not m.done():
        thrust = f * m.t_max
        rud = m.ep._steer(m.s, thrust)
        thrust, rud, _ = safe(m.s[3], thrust, rud)
        m.advance(thrust, rud)
        if not m.finite:
            break
    traj = pl.trajectory()
    dt = float(traj["meta"]["dt"])
    n0 = int(round(O.T_SKIP / dt))
    cut = {kk: (v[n0:] if isinstance(v, np.ndarray) and v.shape[:1]
                == traj["t"].shape else v) for kk, v in traj.items()}
    raw = se.safety_events(cut, k=2.0, k_report=(1.5, 2.5),
                           track=float(m.phi))
    tt, ss = cut["t"], cut["s"]
    x_b = float(pl.sec.x_bow)
    ph, _ = O.traj_phase(m.sea, ss, tt, x_b)
    hb = se.bow_height(cut)
    wet = O.runs(hb < 0.0)
    f0 = raw["bow"]["f0"]
    bury = [a for a, b in wet if hb[a:b].min() < -se.BURY_FRAC * f0
            or (b - a) * dt >= se.WET_LONG - 1e-9]
    dph = np.diff(ph)
    up = np.nonzero((ph[:-1] < 0.0) & (ph[1:] >= 0.0)
                    & (np.abs(dph) < PI / 2))[0] + 1
    since = []
    for a in bury:
        prev = up[up <= a]
        since.append(float((a - prev[-1]) * dt) if prev.size else np.nan)
    # encounter period at the bow: surface zero up-crossings
    cth, sth = np.cos(m.sea.th), np.sin(m.sea.th)
    X = ss[:, 0] + x_b * np.cos(ss[:, 5])
    Y = ss[:, 1] + x_b * np.sin(ss[:, 5])
    eta = np.zeros(len(tt))
    for i0 in range(0, len(tt), 2000):
        sl = slice(i0, i0 + 2000)
        p = (m.sea.k[None, :] * (X[sl, None] * cth + Y[sl, None] * sth)
             - m.sea.w[None, :] * tt[sl, None] + m.sea.phi[None, :])
        eta[sl] = (m.sea.a[None, :] * np.cos(p)).sum(1)
    e0 = eta - eta.mean()
    zc = np.nonzero((e0[:-1] < 0) & (e0[1:] >= 0))[0]
    Tz = float(np.mean(np.diff(zc)) * dt) if zc.size > 1 else np.nan
    crest_gap = float(np.median(np.diff(up)) * dt) if up.size > 1 else np.nan
    return dict(cell=cell, f=f, seed=seed,
                kn=float(np.mean(ss[:, 6])) / O.KN,
                n_bury=len(bury), since=np.array(since),
                ph_bury=ph[bury] if bury else np.zeros(0),
                Tz_bow=Tz, crest_gap=crest_gap, n_up=int(up.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", default="0,1,2,3")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",") if x]
    jobs = [(cell, f, s, 120.0) for cell in ("tp5_180", "tp5_135")
            for f in (0.3, 0.45, 0.6) for s in seeds]
    t0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            res = pool.map(run, jobs)
    else:
        res = list(map(run, jobs))
    pickle.dump(res, open(os.path.join(OUT, "oracle_timing.pkl"), "wb"))
    L = [f"BOW-BURIAL TIMING ({len(res)} constant-throttle runs, "
         f"{time.time() - t0:.0f} s; studies/wave_regime_oracle_timing.py)."
         " since = time from the bow's last upward crest crossing to the "
         "burial onset (s); Tz = mean zero-up-crossing period of the surface"
         " at the bow (s); crest gap = median time between upward crest "
         "crossings at the bow (s). Jet: 0.1 s delay + 0.6 s lag -> 63 % of "
         "a thrust step after 0.7 s, 86 % after 1.3 s, 95 % after 1.9 s."]
    for cell in ("tp5_180", "tp5_135"):
        for f in (0.3, 0.45, 0.6):
            rr = [r for r in res if r["cell"] == cell and r["f"] == f]
            s = np.concatenate([r["since"] for r in rr])
            s = s[np.isfinite(s)]
            phb = np.concatenate([r["ph_bury"] for r in rr])
            L.append(
                f"  {cell} f {f:.2f}: {np.mean([r['kn'] for r in rr]):.1f} kn,"
                f" burials {sum(r['n_bury'] for r in rr)}; since crest: "
                + (f"median {np.median(s):.2f} s, quartiles "
                   f"{np.percentile(s, 25):.2f}-{np.percentile(s, 75):.2f}, "
                   f"min {s.min():.2f}, share < 0.7 s {np.mean(s < 0.7):.2f},"
                   f" < 1.3 s {np.mean(s < 1.3):.2f}" if s.size else "none")
                + f"; ph at onset (circular mean) "
                f"{math.degrees(np.angle(np.mean(np.exp(1j * phb)))) if phb.size else float('nan'):+.0f} deg"
                f"; Tz {np.nanmean([r['Tz_bow'] for r in rr]):.2f} s, crest "
                f"gap {np.nanmean([r['crest_gap'] for r in rr]):.2f} s")
    txt = "\n".join(L)
    open(os.path.join(OUT, "oracle_timing.txt"), "w",
         encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
