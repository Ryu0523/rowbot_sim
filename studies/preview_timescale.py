#!/usr/bin/env python3
"""
Preview was measured over the wrong timescale.

M7 swept t_preview from 0 to 12 s and found nothing that survives an error bar
(see DEFECTS.md A22). That is not a surprise once two numbers are put side by
side:

    surge time constant   tau_u = 6.3 s
    encounter period      T_e   = 7.5 s

To re-time an individual wave encounter, the actuator has to move inside one,
so it needs tau << T_e -- call it T_e/4. The only actuator this vessel has is
3.4x too slow for that. No amount of preview fixes an actuator that cannot act:
preview of 2 s and of 12 s produced identical results because NEITHER was
actionable. The MPC horizon is also 24 x 0.5 = 12 s, so 12 s of preview was the
most the controller could even represent.

But there is a second timescale in a seaway, and thrust IS fast enough for it.
Waves arrive in GROUPS -- five to ten waves, so 50-100 s at Tp = 9.7 s -- and
tau_u = 6.3 s is eight to sixteen times shorter than that. A vessel cannot dodge
one wave, but it can slow down before a group and speed up after it.

So this study asks the question M7 should have asked: not "how far ahead", but
"WHICH timescale". It compares

    short : horizon 12 s, preview  8 s   (what M7 tested)
    group : horizon 90 s, preview 90 s   (the group scale)

against the same constant-thrust baseline, at equal speed, with 12 seeds and a
standard error on every number.

A negative result here is worth as much as a positive one: it would mean the
binding constraint is actuator bandwidth, not sensing range -- and that buying a
drone to see further would have bought nothing.

Run: python -m studies.preview_timescale
"""
import json
import numpy as np

from hydro import bem
from sim.env import Episode
from sim.vessel import NonlinearVessel
from sim.test_vessel import Monochromatic
from sim.wavefield import SeaState
from control.reduced import ReducedModel
from control.mpc import MPPIController, PreviewProvider

DB = "hydro_wigley_10m.npz"
G = 9.81
SEEDS = list(range(12))
T_END = 300.0          # long enough to contain several wave groups
N_DIR = 1              # long-crested head seas, as in M7
HS, TP = 3.25, 9.7

CASES = [
    ("no preview      ", 0.0, 24, 0.5),
    ("short  (M7)     ", 8.0, 24, 0.5),
    ("group  (90 s)   ", 90.0, 90, 1.0),
]


def mpc_run(db, red, seed, t_preview, horizon, dt_ctrl):
    ep = Episode(db, red, hs=HS, tp=TP, seed=seed, t_preview=t_preview,
                 u_ref=4.5, n_dir=N_DIR, use_rudder=False, dt_ctrl=dt_ctrl)
    # rebuild the controller with the horizon this case is about; Episode does
    # not expose it, and the horizon is the whole point here
    ep.ctrl = MPPIController(red, PreviewProvider(ep.sea, t_preview, 0.0, seed),
                             dt_ctrl=dt_ctrl, u_ref=4.5, seed=seed,
                             horizon=horizon, n_samples=192, use_rudder=False)
    return ep.run(T_END)


def baseline(db, thrusts=(5000, 7000, 9000, 11000)):
    out = []
    for th in thrusts:
        rows = []
        for sd in SEEDS:
            sea = SeaState(HS, TP, n_freq=32, n_dir=N_DIR, seed=sd)
            v = NonlinearVessel(db, sea, L=db.L,
                                B=float(db.attrs.get("B", 2.5)),
                                T=float(db.attrs.get("T", 0.8)), dt=0.05)
            s = v.initial_state(2.0)
            n = int(T_END / 0.05)
            acc, spd, sf = np.empty(n), np.empty(n), np.empty(n)
            t = 0.0
            for i in range(n):
                s = v.step(s, t, th, 0.0, 0.05)
                acc[i] = v.last_bow_acc
                spd[i] = s[6]
                sf[i] = v.last_slam_force
                t += 0.05
            rows.append(dict(
                u_mean=float(np.mean(spd[n // 4:])),
                acc_p99=float(np.percentile(np.abs(acc), 99) / G),
                acc_rms=float(np.sqrt(np.mean(acc ** 2)) / G),
                slam_impulse=float(sf.sum() * 0.05 / (t / 60.0) / 1e3)))
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        m.update({k + "_se": float(np.std([r[k] for r in rows], ddof=1)
                                   / np.sqrt(len(rows))) for k in rows[0]})
        out.append(m)
        print(f"    thrust {th:6.0f} N -> u {m['u_mean']:.2f}  "
              f"acc_p99 {m['acc_p99']:.3f}  slamImp {m['slam_impulse']:.0f}")
    return out


def interp(base, key, u):
    us = np.array([b["u_mean"] for b in base])
    o = np.argsort(us)
    return float(np.interp(u, us[o], np.array([b[key] for b in base])[o])), \
        float(np.mean([b[key + "_se"] for b in base]))


def main():
    db = bem.load(DB)
    red = ReducedModel.identify(
        NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L), db)
    tau_u = red.p["tau_u"]
    we = 2 * np.pi / TP + (2 * np.pi / TP) ** 2 * 4.5 / G
    print(f"tau_u = {tau_u:.2f} s   T_e = {2*np.pi/we:.2f} s   "
          f"ratio {tau_u*we/(2*np.pi):.2f}")
    print(f"{len(SEEDS)} seeds x {T_END:.0f} s, long-crested head seas\n")

    print("  constant-thrust baseline:")
    base = baseline(db)

    keys = ("acc_p99", "acc_rms", "slam_impulse")
    print(f"\n  {'case':<18}{'u':>6}{'acc_p99':>9}{'+-':>6}"
          f"{'d%':>7}{'+-':>6}{'res':>5}"
          f"{'slamImp':>9}{'+-':>6}{'d%':>7}{'+-':>6}{'res':>5}")
    rows = []
    for label, tprev, hor, dtc in CASES:
        got = [mpc_run(db, red, sd, tprev, hor, dtc) for sd in SEEDS]
        m = {k: float(np.mean([g[k] for g in got]))
             for k in keys + ("u_mean",)}
        m.update({k + "_se": float(np.std([g[k] for g in got], ddof=1)
                                   / np.sqrt(len(got)))
                  for k in keys + ("u_mean",)})
        m["label"], m["t_preview"], m["horizon"] = label, tprev, hor * dtc
        rows.append(m)

        cells = ""
        for k in ("acc_p99", "slam_impulse"):
            b, bse = interp(base, k, m["u_mean"])
            d = 100 * (m[k] - b) / max(abs(b), 1e-9)
            dse = 100 * np.sqrt(m[k + "_se"] ** 2 + bse ** 2) / max(abs(b), 1e-9)
            fmt = "9.3f" if k == "acc_p99" else "9.0f"
            cells += (f"{m[k]:{fmt}}{m[k+'_se']:>6.2f}{d:>7.1f}{dse:>6.1f}"
                      f"{('YES' if abs(d) > 2*dse else 'no'):>5}")
        print(f"  {label:<18}{m['u_mean']:>6.2f}{cells}")

    json.dump(dict(base=base, rows=rows, tau_u=tau_u),
              open("preview_timescale.json", "w"), indent=1)

    print("\n  " + "-" * 66)
    print("  'res' = does the change exceed twice its own standard error.")
    print("  If the group case is also 'no', the limit is the ACTUATOR, not the")
    print("  sensor: 6.3 s of surge lag cannot be argued with by seeing further,")
    print("  and the hardware question becomes trim tabs, not a taller mast.")
    return base, rows


if __name__ == "__main__":
    main()
