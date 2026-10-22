#!/usr/bin/env python3
"""
What sets off the large, fast heading excursions ("yaw flicks") seen in
the regime scan (studies/wave_regime_scan.py), and are they tied to
individual waves?

Re-runs selected scan missions exactly as the scan does (same builder,
same start speed, autopilot, SafeMode; the scan's run_one is not reused
because it does not return the trajectory) and, on the plant-step
trajectory after 20 s:
  - flick onsets: |yaw rate| first exceeding R_FLICK deg/s (runs >= 0.1 s
    apart);
  - for each onset: the nearest impact peak (studies/safety_events
    acc_peaks, k = 2, >= 2 g) and the bow keel immersion (plant
    last_rel_bow + draft_bow: keel depth of the bow station) max within
    +-0.5 s, against the run's 99th percentile; the Heun substep count;
  - chance level: the share of all rows within 0.5 s of a >= 2 g peak (the
    probability that a random instant would be 'near an impact');
  - the yaw moment terms at the onset row, from the plant's own deriv
    split into: Munk (slender-body) moment, crossflow, Froude-Krylov wave
    moment, jet, by re-evaluating deriv with each term's inputs (exact for
    the jet and Froude-Krylov parts, which deriv adds separately).
  - the CG wave phase (no flips) at the onset.

  python -m studies.wave_regime_events
Output: studies/_cache/wave_regimes/events.txt, events.pkl
"""
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
import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
OUT = os.path.join(HERE, "_cache", "wave_regimes")
KN = 0.514444
PI = math.pi
R_FLICK = 20.0          # deg/s
T_SKIP = 20.0


def yaw_terms(pl, s, t):
    """Yaw moment (N m) of the plant at state s, split by source, using
    the same expressions as PlaningVessel.deriv (sim/planing_vessel.py
    lines 299-320 and the jet at 327-348)."""
    x, dx, rho = pl.x, pl.dx, pl.rho
    u, v, r = s[6], s[7], s[11]
    psi = s[5]
    cps, sps = math.cos(psi), math.sin(psi)
    X, Y = s[0] + x * cps, s[1] + x * sps
    e, et, ett, ex, ey = pl._waves(X, Y, t)
    s_tr = -ex * sps + ey * cps
    _, aux = pl.deriv(s, t)
    d = aux["d"]
    A = pl._section(d)[0]
    ab = pl._a_b(u)
    Ysec = -rho * 9.81 * ab * A * s_tr
    N_fk = float((x * Ysec).sum() * dx)
    m22 = 0.5 * PI * rho * np.maximum(d, 0.0) ** 2
    vl = v + x * r
    Ylift = -abs(u) * m22[0] * vl[0]
    N_transom = float(x[0] * Ylift)
    N_munk = float(-abs(u) * (m22 * vl).sum() * dx)
    Ycf = -0.5 * rho * pl.cd_lat * np.maximum(d, 0.0) * vl * np.abs(vl)
    N_cf = float((x * Ycf).sum() * dx)
    fx, fy = pl.prop.forces(s[12], s[13], u)
    N_jet = float(pl.thrusters[0][0] * fy)
    return dict(fk=N_fk, transom=N_transom, munk=N_munk, cf=N_cf,
                jet=N_jet, d_bow=float(d[-1]),
                m22_sum=float(m22.sum() * dx), Izz=float(pl.Mtot[5, 5]),
                a66_inst=float((m22 * x * x).sum() * dx),
                a66_design=float(pl.Mtot[5, 5] - pl.Iz),
                lw=float((d > 0).sum() * dx))


def run_traj(sea_name, chi, f, seed, T=120.0):
    from studies import final_eval as FE
    from studies.wave_phase_oracle import SafeMode
    from studies.wave_regime_scan import SEAS, U_CALM, wrap
    sd = dict(SEAS[sea_name])
    sd["theta0"] = float(wrap(PI - math.radians(chi)))
    m = FE.make_mission(seed, 0, T, sea=sd)
    pl = m.plant
    m.s = pl.initial_state(U_CALM[f] * KN)
    m.s[5] = m.phi
    safe = SafeMode(m.dt * m.sub)
    while not m.done():
        thrust = f * m.t_max
        rud = m.ep._steer(m.s, thrust)
        thrust, rud, _ = safe(m.s[3], thrust, rud)
        m.advance(thrust, rud)
        if not m.finite or getattr(pl, "capsized", False):
            break
    return m, pl.trajectory()


def analyse(key, T=120.0):
    from studies import safety_events as se
    sea_name, chi, f, seed = key
    t0 = time.perf_counter()
    m, tr = run_traj(sea_name, chi, f, seed, T)
    pl = m.plant
    dt = float(tr["meta"]["dt"])
    n0 = int(round(T_SKIP / dt))
    t, s = tr["t"][n0:], tr["s"][n0:]
    a_cg = tr["a_cg"][n0:]
    keel_bow = tr["rel_bow"][n0:] + tr["meta"]["draft_bow"]
    nsub = tr["n_sub"][n0:]
    idx, pk = se.acc_peaks(a_cg, dt, 2.0)
    imp = idx[pk >= 2.0]
    near = np.zeros(len(t), bool)
    w = int(round(0.5 / dt))
    for i in imp:
        near[max(i - w, 0):i + w + 1] = True
    chance = float(near.mean())
    rdeg = np.degrees(np.abs(s[:, 11]))
    above = rdeg > R_FLICK
    ons = np.nonzero(above[1:] & ~above[:-1])[0] + 1
    # onsets at least 0.1 s apart
    keep, last = [], -10 ** 9
    for i in ons:
        if i - last >= int(round(0.1 / dt)):
            keep.append(i)
        last = i
    ons = np.array(keep, int)
    kb99 = float(np.percentile(keel_bow, 99))
    rows = []
    for i in ons:
        lo, hi = max(i - w, 0), min(i + w + 1, len(t))
        dist = (np.min(np.abs(imp - i)) * dt) if imp.size else np.inf
        terms = yaw_terms(pl, s[i], float(t[i]))
        rows.append(dict(t=float(t[i]), near_imp=bool(near[i]),
                         dist_imp=float(dist),
                         keel_bow_max=float(keel_bow[lo:hi].max()),
                         over99=bool(keel_bow[lo:hi].max() >= kb99),
                         nsub=int(nsub[lo:hi].max()),
                         r_peak=float(rdeg[i:min(i + int(2 / dt), len(t))].max()),
                         e_deg=float(np.degrees(wrap_(s[i, 5] - m.phi))),
                         v=float(s[i, 7]), u=float(s[i, 6]),
                         terms=terms))
    # the same yaw-moment split at the running state (median over the run
    # of |terms|, every 25th row), for scale
    base = [yaw_terms(pl, s[j], float(t[j])) for j in range(0, len(t), 25)]
    med = {k: float(np.median([abs(b[k]) for b in base]))
           for k in ("fk", "transom", "munk", "cf", "jet", "d_bow",
                     "m22_sum", "lw")}
    out = dict(key=key, kn=float(np.mean(s[:, 6]) / KN), n_onsets=len(ons),
               n_imp=int(imp.size), chance=chance,
               share_near=float(np.mean([r["near_imp"] for r in rows]))
               if rows else float("nan"),
               share_over99=float(np.mean([r["over99"] for r in rows]))
               if rows else float("nan"),
               keel_bow_p50=float(np.median(keel_bow)), keel_bow_p99=kb99,
               rows=rows, base_median=med, Izz=float(pl.Mtot[5, 5]),
               t_run=time.perf_counter() - t0)
    return out


def wrap_(a):
    return (a + PI) % (2 * PI) - PI


def main():
    keys = [("tp5", 0, 0.6, 0), ("tp5", 0, 1.0, 0), ("tp5", 0, 0.6, 1),
            ("tp5", 45, 0.6, 0), ("tp5", 180, 0.6, 0), ("tp8", 180, 0.6, 0),
            ("tp5", 135, 0.6, 0), ("tp5", 180, 0.2, 0), ("tp5", 90, 0.6, 0)]
    if len(sys.argv) > 1:
        keys = [tuple(int(x) if i in (1, 3) else (float(x) if i == 2 else x)
                      for i, x in enumerate(k.split(":")))
                for k in sys.argv[1:]]
    L = []
    P = L.append
    P(f"YAW FLICKS: onsets of |yaw rate| > {R_FLICK} deg/s after 20 s; "
      "near = within 0.5 s of a >= 2 g CG impact peak (k = 2); chance = "
      "share of all instants that are that close to such a peak; over99 = "
      "the bow keel immersion within +-0.5 s reaches the run's 99th "
      "percentile. Yaw-moment split (kN m) at the onset vs the run's median"
      " |value|: Munk (slender-body, -U int m22 v_l dx), transom lift "
      "(x_T Y), crossflow, Froude-Krylov wave moment, jet; m22 = int m22 dx"
      " (kg/m * m = kg) against the CONSTANT yaw inertia Mtot[5,5].")
    res = []
    for k in keys:
        o = analyse(k)
        res.append(o)
        P("")
        P(f"{k}: {o['kn']:.1f} kn, onsets {o['n_onsets']}, >=2 g peaks "
          f"{o['n_imp']}, near-impact {o['share_near']:.2f} (chance "
          f"{o['chance']:.2f}), bow immersion at its 99th pct "
          f"{o['share_over99']:.2f} (chance ~0.01 x window); keel at bow "
          f"median {o['keel_bow_p50']:.2f} m, p99 {o['keel_bow_p99']:.2f} m;"
          f" Izz(total, constant) {o['Izz']:.0f} kg m2 ({o['t_run']:.0f} s)")
        b = o["base_median"]
        P(f"   run median |N| kN m: munk {b['munk'] / 1e3:.2f}, transom "
          f"{b['transom'] / 1e3:.2f}, cf {b['cf'] / 1e3:.2f}, fk "
          f"{b['fk'] / 1e3:.2f}, jet {b['jet'] / 1e3:.2f}; m22 {b['m22_sum']:.0f}"
          f" kg, d_bow {b['d_bow']:.2f} m, wetted length {b['lw']:.2f} m")
        for r in o["rows"][:12]:
            tm = r["terms"]
            P(f"   t {r['t']:6.2f}: r_peak {r['r_peak']:5.1f} deg/s, hdg "
              f"{r['e_deg']:+5.1f}, v {r['v']:+.2f} m/s, impact "
              f"{'near' if r['near_imp'] else 'far '} ({r['dist_imp']:.2f} s),"
              f" keel_bow max {r['keel_bow_max']:.2f} m, nsub {r['nsub']}; "
              f"N munk {tm['munk'] / 1e3:+6.2f} transom "
              f"{tm['transom'] / 1e3:+6.2f} cf {tm['cf'] / 1e3:+6.2f} fk "
              f"{tm['fk'] / 1e3:+6.2f} jet {tm['jet'] / 1e3:+6.2f}; m22 "
              f"{tm['m22_sum']:.0f} kg, yaw added inertia now "
              f"{tm['a66_inst']:.0f} vs used {tm['a66_design']:.0f} kg m2, "
              f"lw {tm['lw']:.2f} m")
    txt = "\n".join(L)
    with open(os.path.join(OUT, "events.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    with open(os.path.join(OUT, "events.pkl"), "wb") as fh:
        pickle.dump(res, fh)
    print(txt)


if __name__ == "__main__":
    main()
