#!/usr/bin/env python3
"""
Which uncalibrated placeholder actually endangers a conclusion?

`DEFECTS.md` section C lists five groups of numbers that have the right shape
but no measurement behind them, and flags roll damping as "especially
dangerous" because potential flow leaves roll almost undamped, so the
placeholder decides the entire roll resonance. That warning is correct in
general. It is not evidence about THIS project's results.

The useful question is narrower: over the range each number could plausibly
take, how far do the reported metrics move? A parameter that could be wrong by
4x and still moves the answer by 1% is not a risk to any conclusion, however
uncertain it is. One that moves the answer by 30% has to be measured before
anything downstream is quoted.

So each placeholder is swept over a wide bracket and the result is reported as
an ELASTICITY, d ln(metric) / d ln(parameter): the percentage the metric moves
per percent the parameter moves. That is the number that says which towing-tank
test to buy first, which is the only decision this table has to support.

Run: python -m studies.placeholder_sensitivity
"""
import numpy as np

from hydro import bem
from sim.env import Episode
from sim.vessel import NonlinearVessel
from sim.test_vessel import Monochromatic
from control.reduced import ReducedModel

DB = "hydro_wigley_10m.npz"
G = 9.81
SEEDS = (0, 1)
T_END = 100.0

# name -> (how to apply a multiplier, bracket). The bracket is how wrong the
# number could plausibly be, not a uniform convention: roll damping gets the
# widest because it is the least constrained by anything we have.
VISC_DOF = dict(surge=0, sway=1, heave=2, roll=3, pitch=4, yaw=5)


def apply(plant, name, mult):
    if name in VISC_DOF:
        plant.visc = plant.visc.copy()
        plant.visc[VISC_DOF[name]] *= mult
    elif name == "rudder lift slope":
        plant.rudder.cl_alpha *= mult
    elif name == "thrust lag tau":
        plant.prop.tau *= mult
    elif name == "thrust rate limit":
        plant.prop.rate_max *= mult
    elif name == "propeller radius (ventilation)":
        plant.prop.r_prop *= mult
    elif name == "slam pile-up factor":
        plant.slam.dma = plant.slam.dma * mult
    else:
        raise KeyError(name)


SWEEP = [
    ("roll damping", "roll", (0.25, 4.0)),
    ("surge drag (speed envelope)", "surge", (0.6, 1.7)),
    ("heave damping", "heave", (0.5, 2.0)),
    ("pitch damping", "pitch", (0.5, 2.0)),
    ("sway damping", "sway", (0.5, 2.0)),
    ("yaw damping", "yaw", (0.5, 2.0)),
    ("rudder lift slope", "rudder lift slope", (0.7, 1.4)),
    ("thrust lag tau", "thrust lag tau", (0.5, 2.0)),
    ("thrust rate limit", "thrust rate limit", (0.5, 2.0)),
    ("propeller radius (ventilation)", "propeller radius (ventilation)",
     (0.7, 1.4)),
    ("slam pile-up factor", "slam pile-up factor", (0.4, 1.0)),
]

METRICS = ("rms_acc", "p99_acc", "slams", "speed", "heave", "pitch")


def run(db, red, mult_name=None, mult=1.0, hs=3.25, tp=9.7):
    out = []
    for sd in SEEDS:
        ep = Episode(db, red, hs=hs, tp=tp, seed=sd, t_preview=0.0,
                     u_ref=4.5, n_freq=24, n_dir=5,
                     use_rudder=False, autopilot=True)
        if mult_name is not None:
            apply(ep.plant, mult_name, mult)
        s = ep.plant.initial_state(ep.u_ref * 0.8)
        t = 0.0
        acc, hv, pt, sp = [], [], [], []
        for _ in range(int(T_END / ep.dt_ctrl)):
            thr, rud = ep.ctrl.to_actuator(
                ep.ctrl(ep.reduced.from_plant_state(s), t))
            for _ in range(ep.sub):
                s = ep.plant.step(s, t, thr, rud, ep.dt)
                t += ep.dt
                acc.append(abs(ep.plant.last_bow_acc) / G)
                hv.append(s[2]); pt.append(s[4]); sp.append(s[6])
        a = np.array(acc)
        out.append(dict(rms_acc=float(np.sqrt((a ** 2).mean())),
                        p99_acc=float(np.percentile(a, 99)),
                        slams=float(ep.plant.slam_count),
                        speed=float(np.mean(sp)),
                        heave=float(np.std(hv)),
                        pitch=float(np.degrees(np.std(pt)))))
    return {k: float(np.mean([o[k] for o in out])) for k in METRICS}


def main():
    db = bem.load(DB)
    red = ReducedModel.identify(
        NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L), db)

    base = run(db, red)
    print("baseline: " + "  ".join(f"{k}={v:.3f}" for k, v in base.items()))
    print("\nelasticity  d ln(metric) / d ln(parameter), over the bracket shown")
    print(f"  {'placeholder':<32}{'bracket':>12}"
          + "".join(f"{m:>9}" for m in METRICS))

    worst = []
    for label, key, (lo, hi) in SWEEP:
        rl = run(db, red, key, lo)
        rh = run(db, red, key, hi)
        span = np.log(hi / lo)
        e = {}
        for m in METRICS:
            a, b = rl[m], rh[m]
            if min(abs(a), abs(b)) < 1e-9:          # counts can be zero
                e[m] = float("nan") if abs(a - b) < 1e-9 else np.inf
            else:
                e[m] = float(np.log(b / a) / span)
        worst.append((label, max(abs(v) for v in e.values()
                                 if np.isfinite(v)) if any(
            np.isfinite(v) for v in e.values()) else 0.0))
        print(f"  {label:<32}{f'x{lo:g}-{hi:g}':>12}"
              + "".join(f"{e[m]:>9.2f}" if np.isfinite(e[m]) else f"{'-':>9}"
                        for m in METRICS))

    print("\n  " + "-" * 60)
    worst.sort(key=lambda r: -r[1])
    print("  ranked by worst-case elasticity -- buy the tank tests in this order:")
    for label, v in worst:
        bar = "#" * int(round(min(v, 1.0) * 30))
        print(f"    {label:<32}{v:>6.2f}  {bar}")
    print("\n  An elasticity of 0.1 means a parameter wrong by a factor of two")
    print("  moves that metric by about 7%. Below ~0.05 the placeholder cannot")
    print("  change a conclusion and only its ORDER of magnitude matters.")
    return worst


if __name__ == "__main__":
    main()
