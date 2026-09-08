#!/usr/bin/env python3
"""
How much does the slamming IMPACT LOAD change, now that it is fed back?

DEFECTS.md listed "slamming detected but not loaded" as a carried limitation.
This measures what removing it was worth, because the answer is not obvious in
either direction and the two candidate answers point opposite ways:

  * The IMPULSE is small. A slam transfers the momentum of a modest sectional
    added mass at a few m/s -- tens of kg times a couple of m/s against a nine
    tonne hull. So the trajectory should barely move.

  * The FORCE is large and brief, and bow acceleration is a derivative. It is
    also the single most heavily weighted term in the tuned objective
    (w_acc = 8.8, the largest of the seven). So the metric being optimised
    could change a great deal even while the motion does not.

If the second holds, the limitation mattered for the OBJECTIVE rather than for
the DYNAMICS -- which would mean every weight tuned without it was tuned
against a mis-stated cost. That is worth knowing before trusting M6.

Run: python -m studies.slam_effect
"""
import numpy as np

from hydro import bem
from sim.env import Episode
from sim.vessel import NonlinearVessel
from sim.test_vessel import Monochromatic
from control.reduced import ReducedModel

DB = "hydro_wigley_10m.npz"
G = 9.81


def run(db, red, seed, hs, tp, slam_load, t_end=120.0, t_preview=0.0):
    ep = Episode(db, red, hs=hs, tp=tp, seed=seed, t_preview=t_preview,
                 u_ref=4.5, n_freq=24, n_dir=5,
                 use_rudder=False, autopilot=True)
    ep.plant.use_slam_load = slam_load
    ep.plant.slam_count = 0

    s = ep.plant.initial_state(ep.u_ref * 0.8)
    t = 0.0
    acc, hv, pt, sp, fsl = [], [], [], [], []
    for _ in range(int(t_end / ep.dt_ctrl)):
        sr = ep.reduced.from_plant_state(s)
        thr, rud = ep.ctrl.to_actuator(ep.ctrl(sr, t))
        for _ in range(ep.sub):
            s = ep.plant.step(s, t, thr, rud, ep.dt)
            t += ep.dt
            acc.append(ep.plant.last_bow_acc / G)
            hv.append(s[2]); pt.append(s[4]); sp.append(s[6])
            if slam_load:
                fsl.append(ep.plant.slam_load(
                    s[:6], s[6:12], ep.plant.wave_forces(s[:6], t)[1], t)[0])
    acc = np.abs(np.array(acc))
    return dict(peak_acc=float(acc.max()),
                p99_acc=float(np.percentile(acc, 99)),
                rms_acc=float(np.sqrt(np.mean(acc ** 2))),
                heave=float(np.std(hv)), pitch=float(np.degrees(np.std(pt))),
                speed=float(np.mean(sp)), slams=int(ep.plant.slam_count),
                peak_F=float(np.max(fsl)) / 1e3 if fsl else 0.0)


def main(seeds=(0, 1), sea_states=((3.25, 9.7), (1.88, 8.0))):
    db = bem.load(DB)
    red = ReducedModel.identify(
        NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L), db)
    disp = float(db.M[0, 0]) * G / 1e3          # displacement weight, kN

    print(f"Wigley 10 m, displacement {disp:.0f} kN\n")
    print(f"  {'sea':>12} {'seed':>4} {'slam load':>10} {'peak a_bow':>11}"
          f"{'p99':>7}{'rms':>7}{'heave':>7}{'pitch':>7}{'speed':>7}"
          f"{'slams':>6}{'peak F':>8}")
    rows = {}
    for hs, tp in sea_states:
        for sd in seeds:
            for on in (False, True):
                r = run(db, red, sd, hs, tp, on)
                rows[(hs, sd, on)] = r
                print(f"  Hs{hs:>5.2f} Tp{tp:>4.1f} {sd:>4} "
                      f"{'ON' if on else 'off':>10}"
                      f"{r['peak_acc']:>10.3f}g{r['p99_acc']:>7.3f}"
                      f"{r['rms_acc']:>7.3f}{r['heave']:>7.3f}"
                      f"{r['pitch']:>7.2f}{r['speed']:>7.2f}"
                      f"{r['slams']:>6}"
                      + (f"{r['peak_F']:>7.1f}kN" if on else f"{'-':>8}"))

    def ratio(key):
        a = [rows[k]["" + key] for k in rows if not k[2]]
        b = [rows[k]["" + key] for k in rows if k[2]]
        return float(np.mean(b) / max(np.mean(a), 1e-12))

    print("\n  " + "-" * 62)
    print("  effect of feeding the impact load back (ON / off):")
    for k, name in (("peak_acc", "peak bow acceleration"),
                    ("p99_acc", "99th-percentile bow acceleration"),
                    ("rms_acc", "rms bow acceleration"),
                    ("heave", "heave std"), ("pitch", "pitch std"),
                    ("speed", "mean speed")):
        print(f"    {name:<34} x{ratio(k):.2f}")
    pkF = np.mean([rows[k]["peak_F"] for k in rows if k[2]])
    print(f"    peak impact force                  {pkF:.1f} kN "
          f"= {100*pkF/disp:.0f}% of displacement")

    print("\n  " + "-" * 62)
    a_mot = max(abs(ratio("heave") - 1), abs(ratio("pitch") - 1))
    a_met = abs(ratio("peak_acc") - 1)
    print(f"  motion changed by  {a_mot:.1%}")
    print(f"  metric changed by  {a_met:.1%}")
    if a_met > 3 * max(a_mot, 0.01):
        print("  => the limitation was in the OBJECTIVE, not the DYNAMICS:")
        print("     the trajectory is nearly unchanged, but the quantity the")
        print("     tuner was minimising is not the one the vessel feels.")
    return rows


if __name__ == "__main__":
    main()
