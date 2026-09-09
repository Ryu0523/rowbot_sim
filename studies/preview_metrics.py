#!/usr/bin/env python3
"""
Does preview help ANY of the criteria, or only the one it was tested on?

M7 measured preview against bow acceleration and a counted slam rate, and the
second of those turned out to be noise (DEFECTS.md A22). Before concluding that
preview is worthless, it is worth checking whether it helps something else --
relative motion, deck wetness, fatigue, added resistance -- since those are
better-conditioned statistics and a real effect would show up in them first.

Every metric, three preview strategies, against a constant-thrust baseline
interpolated to the SAME mean speed, with a standard error on every number.

Run: python -m studies.preview_metrics
"""
import json
import numpy as np

from hydro import bem
from sim.env import Episode, FREEBOARD
from sim.vessel import NonlinearVessel
from sim.test_vessel import Monochromatic
from sim.wavefield import SeaState
from sim import seakeeping
from control.reduced import ReducedModel
from control.mpc import MPPIController, PreviewProvider

DB = "hydro_wigley_10m.npz"
G = 9.81
SEEDS = list(range(16))     # 10 left the group case unresolved at +-2%
T_END = 260.0
N_DIR = 1
HS, TP = 3.25, 9.7
# Down to 3 kN. The group-preview case settles at 3.62 m/s, below the 3.86 m/s
# the old 5 kN floor produced -- and np.interp CLAMPS outside its range, so that
# case was silently compared against a baseline running 0.24 m/s faster than it.
# Since slowing down improves nearly every metric, that flattered it. Same class
# of error as an optimum sitting on the edge of its bracket.
THRUSTS = (3000, 4000, 5000, 7000, 9000, 11000)

CASES = [("no preview", 0.0, 24, 0.5),
         ("short 8 s", 8.0, 24, 0.5),
         ("group 90 s", 90.0, 90, 1.0)]

KEYS = ("acc_rms", "acc_p99", "fatigue", "rvm_rms", "rvv_rms",
        "slam_impulse", "slam_p99", "slam_rate_ochi", "wet_rate")


def mpc_run(db, red, seed, t_preview, horizon, dt_ctrl):
    ep = Episode(db, red, hs=HS, tp=TP, seed=seed, t_preview=t_preview,
                 u_ref=4.5, n_dir=N_DIR, use_rudder=False, dt_ctrl=dt_ctrl)
    ep.ctrl = MPPIController(red, PreviewProvider(ep.sea, t_preview, 0.0, seed),
                             dt_ctrl=dt_ctrl, u_ref=4.5, seed=seed,
                             horizon=horizon, n_samples=192, use_rudder=False)
    return ep.run(T_END)


def baseline(db):
    out = []
    for th in THRUSTS:
        rows = []
        for sd in SEEDS:
            sea = SeaState(HS, TP, n_freq=32, n_dir=N_DIR, seed=sd)
            v = NonlinearVessel(db, sea, L=db.L,
                                B=float(db.attrs.get("B", 2.5)),
                                T=float(db.attrs.get("T", 0.8)), dt=0.05)
            s = v.initial_state(2.0)
            n = int(T_END / 0.05)
            acc = np.empty(n); spd = np.empty(n)
            rel = np.empty(n); sf = np.empty(n)
            t = 0.0
            for i in range(n):
                s = v.step(s, t, th, 0.0, 0.05)
                acc[i] = v.last_bow_acc; spd[i] = s[6]
                rel[i] = v.last_rel_bow; sf[i] = v.last_slam_force
                t += 0.05
            r = seakeeping.summarise(rel, acc / G, 0.05, v.T,
                                     FREEBOARD, v.v_slam)
            r.update(u_mean=float(np.mean(spd[n // 4:])),
                     slam_impulse=float(sf.sum() * 0.05 / (t / 60.0) / 1e3),
                     slam_p99=float(np.percentile(sf, 99) / 1e3))
            rows.append(r)
        m = {k: float(np.mean([r[k] for r in rows]))
             for k in KEYS + ("u_mean",)}
        m.update({k + "_se": float(np.std([r[k] for r in rows], ddof=1)
                                   / np.sqrt(len(rows)))
                  for k in KEYS + ("u_mean",)})
        out.append(m)
        print(f"    {th/1000:>5.0f} kN -> u {m['u_mean']:.2f}")
    return out


def interp(base, key, u):
    """Baseline value at speed u. Refuses to extrapolate.

    np.interp clamps silently outside its range, which turns an out-of-bracket
    comparison into a wrong number rather than an error. Every metric here
    improves as the vessel slows, so comparing a slow controller against a
    clamped (faster) baseline manufactures an improvement out of nothing.
    """
    us = np.array([b["u_mean"] for b in base])
    o = np.argsort(us)
    if not (us.min() - 1e-9 <= u <= us.max() + 1e-9):
        raise ValueError(
            f"speed {u:.2f} m/s is outside the baseline bracket "
            f"[{us.min():.2f}, {us.max():.2f}] -- widen THRUSTS")
    return (float(np.interp(u, us[o], np.array([b[key] for b in base])[o])),
            float(np.mean([b[key + "_se"] for b in base])))


def main():
    db = bem.load(DB)
    red = ReducedModel.identify(
        NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L), db)
    print(f"{len(SEEDS)} seeds x {T_END:.0f} s, long-crested head seas")
    print("  constant-thrust baseline:")
    base = baseline(db)

    rows = []
    for label, tprev, hor, dtc in CASES:
        got = [mpc_run(db, red, sd, tprev, hor, dtc) for sd in SEEDS]
        m = {k: float(np.mean([g[k] for g in got])) for k in KEYS + ("u_mean",)}
        m.update({k + "_se": float(np.std([g[k] for g in got], ddof=1)
                                   / np.sqrt(len(got)))
                  for k in KEYS + ("u_mean",)})
        m["label"] = label
        rows.append(m)

    print(f"\n  delta vs constant thrust AT EQUAL SPEED, per metric")
    print(f"  {'metric':<16}" + "".join(f"{c[0]:>22}" for c in CASES))
    n_res = 0
    for k in KEYS:
        line = f"  {k:<16}"
        for m in rows:
            b, bse = interp(base, k, m["u_mean"])
            d = 100 * (m[k] - b) / max(abs(b), 1e-9)
            dse = 100 * np.sqrt(m[k + "_se"] ** 2 + bse ** 2) / max(abs(b), 1e-9)
            res = abs(d) > 2 * dse
            n_res += res and d < 0
            line += f"{d:>+13.1f} ±{dse:<4.1f}{'*' if res else ' '}"
        print(line)
    print(f"\n  speeds: " + "  ".join(f"{m['label']} {m['u_mean']:.2f} m/s"
                                      for m in rows))
    print("  * = exceeds twice its own standard error")
    print(f"\n  metrics IMPROVED beyond noise by any preview strategy: {n_res}"
          f" of {len(KEYS)*len(CASES)}")
    json.dump(dict(base=base, rows=rows), open("preview_metrics.json", "w"),
              indent=1)
    return rows


if __name__ == "__main__":
    main()
