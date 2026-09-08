#!/usr/bin/env python3
"""
Is a sensor that measures the SEA STATE worth anything?

Wave-by-wave preview turned out to be worthless on this vessel, because the
only actuator is 3.4x slower than the encounter period (DEFECTS.md A23). But
that result also showed what DOES work: choosing a lower mean speed. And every
study so far has been handed Hs and Tp as constants. A real vessel is not.

So the sensing question was aimed at the wrong quantity too. Not "what will the
next wave do", which cannot be acted on, but "what sea am I in", which sets the
one control that matters. This study measures what that knowledge is worth.

Two parts:

  1. THE HULL IS ALREADY A WAVE BUOY. In the contouring regime the vessel
     follows the surface, so its own heave should carry the wave amplitude with
     no forward-looking sensor at all. If sigma_heave ~ Hs/4, a sea-state
     sensor is redundant for measuring the sea the vessel is ALREADY IN, and
     the only thing an external sensor can add is advance notice of a sea it
     has not reached yet.

  2. THE VALUE OF KNOWING. For each sea state find the thrust that minimises
     the operator objective, then compare against a single compromise thrust
     held across all of them. The gap is the entire value of sea-state
     knowledge, however it is obtained.

Run: python -m studies.sea_state_value
"""
import numpy as np

from hydro import bem
from sim.env import score, FREEBOARD
from sim.vessel import NonlinearVessel
from sim.wavefield import SeaState
from sim import seakeeping

DB = "hydro_wigley_10m.npz"
G = 9.81
SEEDS = (0, 1, 2, 3, 4, 5)
T_END = 220.0
# The first bracket started at 3 kN and every sea state picked the bottom of
# it, which makes a "value of knowing" of 0% meaningless -- an optimum on the
# boundary is the bracket's answer, not the physics'. Extended down until the
# speed penalty in `score` turns the curve back up and the optimum is interior.
THRUSTS = (1500, 2000, 2500, 3000, 3500, 4500, 6000, 8000)
SEA_STATES = [(1.88, 8.0), (2.50, 8.8), (3.25, 9.7), (4.00, 10.5)]
U_REF = 4.5


def run(db, hs, tp, thrust, seed):
    sea = SeaState(hs, tp, n_freq=32, n_dir=1, seed=seed)
    v = NonlinearVessel(db, sea, L=db.L, B=float(db.attrs.get("B", 2.5)),
                        T=float(db.attrs.get("T", 0.8)), dt=0.05)
    s = v.initial_state(2.0)
    n = int(T_END / 0.05)
    acc = np.empty(n); spd = np.empty(n); rel = np.empty(n); hv = np.empty(n)
    t = 0.0
    for i in range(n):
        s = v.step(s, t, thrust, 0.0, 0.05)
        acc[i] = v.last_bow_acc; spd[i] = s[6]
        rel[i] = v.last_rel_bow; hv[i] = s[2]
        t += 0.05
    m = seakeeping.summarise(rel, acc / G, 0.05, v.T, FREEBOARD, v.v_slam)
    m.update(u_mean=float(np.mean(spd[n // 4:])), cross_rms=0.0, finite=True,
             slam_per_min=float(v.slam_count / (t / 60.0)),
             heave_std=float(np.std(hv[n // 4:])))
    # read the wave period back out of the vessel's OWN heave, then undo the
    # Doppler shift with the speed it already knows
    x = hv[n // 4:] - hv[n // 4:].mean()
    F = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    f = np.fft.rfftfreq(x.size, 0.05)
    we = 2 * np.pi * f[int(np.argmax(F[1:])) + 1]
    u = m["u_mean"]
    m["tp_est"] = float(2 * np.pi / ((-1 + np.sqrt(1 + 4 * (u / G) * we))
                                     / (2 * u / G))) if we > 0 else np.nan
    return m


def main():
    db = bem.load(DB)

    print("1. can the hull read the sea it is already in?\n")
    print(f"  {'Hs':>6}{'Tp':>6}{'sigma_wave':>12}{'sigma_heave':>13}"
          f"{'ratio':>8}{'Tp est':>9}{'err':>7}")
    for hs, tp in SEA_STATES:
        r = [run(db, hs, tp, 7000, sd) for sd in SEEDS]
        sw = hs / 4.0
        sh = float(np.mean([x["heave_std"] for x in r]))
        te = float(np.mean([x["tp_est"] for x in r]))
        print(f"  {hs:>6.2f}{tp:>6.1f}{sw:>12.3f}{sh:>13.3f}{sh/sw:>8.3f}"
              f"{te:>9.2f}{100*(te-tp)/tp:>6.0f}%")

    print("\n2. what is it worth to know?\n")
    print(f"  {'Hs':>6}{'Tp':>6}  " + "".join(f"{t/1000:>9.1f}k"
                                              for t in THRUSTS)
          + f"{'best':>9}{'u*':>7}")
    tab = {}
    for hs, tp in SEA_STATES:
        js = []
        for th in THRUSTS:
            r = [run(db, hs, tp, th, sd) for sd in SEEDS]
            js.append(float(np.mean([score(x, U_REF) for x in r])))
            tab[(hs, th)] = (js[-1], float(np.mean([x["u_mean"] for x in r])))
        i = int(np.argmin(js))
        print(f"  {hs:>6.2f}{tp:>6.1f}  " + "".join(f"{j:>10.3f}" for j in js)
              + f"{THRUSTS[i]/1000:>8.1f}k{tab[(hs,THRUSTS[i])][1]:>7.2f}")

    # one thrust for all seas, chosen to minimise the mean objective
    fixed = [(th, float(np.mean([tab[(hs, th)][0] for hs, _ in SEA_STATES])))
             for th in THRUSTS]
    best_fixed = min(fixed, key=lambda z: z[1])
    adaptive = float(np.mean([min(tab[(hs, th)][0] for th in THRUSTS)
                              for hs, _ in SEA_STATES]))

    print(f"\n  one fixed thrust for every sea  : {best_fixed[1]:.4f} "
          f"(at {best_fixed[0]/1000:.0f} kN)")
    print(f"  thrust chosen per sea state     : {adaptive:.4f}")
    gain = 100 * (best_fixed[1] - adaptive) / max(best_fixed[1], 1e-9)
    print(f"  value of knowing the sea state  : {gain:.1f}%")

    print("\n  " + "-" * 62)
    print("  Compare with what wave-by-wave preview bought: nothing that")
    print("  exceeded its own error bar, at any horizon from 0 to 90 s.")
    return gain


if __name__ == "__main__":
    main()
