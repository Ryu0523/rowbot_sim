#!/usr/bin/env python3
"""
Heading as a control channel -- the lever this project locked out.

Every preview study so far ran with `use_rudder=False`. That was defensible at
the time (rudder lift makes sway, and the reduced model has no sway state, so
the controller could not see the drift it caused) but it means the strongest
lever was never measured. Two numbers say how strong:

    encounter period, head to following seas :  7.59 -> 13.42 s   (1.77x)
    encounter period, thrust +-20%           :  7.49 ->  7.12 s   (1.05x)

Heading has roughly FIFTEEN TIMES the authority over encounter frequency that
thrust has. And the rudder is twice as quick: tau_r = 3.35 s against tau_u =
6.30 s.

There is a second effect that may matter more. Bow acceleration is dominated by
pitch through the L/2 lever, and pitch is driven by the wave SLOPE ALONG THE
HULL. Turn off head seas and that slope collapses -- at beam seas the wave
varies across the 2.5 m beam instead of the 10 m length. So heading attacks the
dominant term directly, where thrust only nudges the frequency at which it
arrives.

THE CATCH, and why this is a routing question rather than a free lunch: turning
off the wind costs progress. If the destination lies dead into the waves,
heading at an angle delta off it makes good only u.cos(delta). Sailing at 135
degrees to the waves throws away 29% of every metre travelled, so the vessel
must go correspondingly faster -- which is the trade this study measures.

Everything is therefore compared at equal VELOCITY MADE GOOD toward a
destination dead upwind, not at equal speed. That is the only comparison that
means anything for a vessel that has somewhere to be.

Run: python -m studies.heading_choice
"""
import json
import numpy as np

from hydro import bem
from sim.env import FREEBOARD
from sim.vessel import NonlinearVessel
from sim.wavefield import SeaState
from sim import seakeeping

DB = "hydro_wigley_10m.npz"
G = 9.81
SEEDS = list(range(8))
T_END = 220.0
HS, TP = 3.25, 9.7
DT = 0.05
# 3.0 was unreachable: head seas at the lowest thrust already make 3.08 m/s
# good, so the reference column was never populated and every other column
# raised a KeyError. The target has to sit inside EVERY heading's range.
VMG_TARGET = 3.2        # m/s toward a destination dead into the waves

# relative heading in degrees: 180 = head seas, 90 = beam.
# Below 120 the speed needed to hold the VMG target leaves the thrust envelope.
# 120 deg cannot hold the VMG target inside the thrust envelope -- holding
# 3.2 m/s of progress at 60 deg off the waves needs 6.4 m/s through the water.
HEADINGS = (180, 165, 150, 135)
THRUSTS = (2000, 3000, 4500, 6000, 8000, 10000, 12000)

KEYS = ("acc_rms", "acc_p99", "fatigue", "rvm_rms", "rvv_rms",
        "slam_impulse", "slam_rate_ochi", "wet_rate", "roll_rms")


def run(db, seed, mu_deg, thrust):
    """The vessel always sails along +x; the SEA is rotated instead, which is
    equivalent and avoids needing the rudder to hold an off-head heading."""
    sea = SeaState(HS, TP, n_freq=32, n_dir=1, seed=seed,
                   theta0=np.radians(mu_deg))
    v = NonlinearVessel(db, sea, L=db.L, B=float(db.attrs.get("B", 2.5)),
                        T=float(db.attrs.get("T", 0.8)), dt=DT)
    s = v.initial_state(2.0)
    n = int(T_END / DT)
    acc = np.empty(n); spd = np.empty(n); rel = np.empty(n)
    sf = np.empty(n); rol = np.empty(n)
    t = 0.0
    for i in range(n):
        s = v.step(s, t, thrust, 0.0, DT)
        acc[i] = v.last_bow_acc; spd[i] = s[6]
        rel[i] = v.last_rel_bow; sf[i] = v.last_slam_force
        rol[i] = s[3]
        t += DT
    m = seakeeping.summarise(rel, acc / G, DT, v.T, FREEBOARD, v.v_slam)
    u = float(np.mean(spd[n // 4:]))
    m.update(u_mean=u,
             vmg=u * abs(np.cos(np.radians(mu_deg))),
             roll_rms=float(np.degrees(np.std(rol[n // 4:]))),
             slam_impulse=float(sf.sum() * DT / (t / 60.0) / 1e3))
    return m


def agg(rows):
    m = {k: float(np.mean([r[k] for r in rows]))
         for k in KEYS + ("u_mean", "vmg")}
    m.update({k + "_se": float(np.std([r[k] for r in rows], ddof=1)
                               / np.sqrt(len(rows)))
              for k in KEYS + ("u_mean", "vmg")})
    return m


def main():
    db = bem.load(DB)
    print(f"{len(SEEDS)} seeds x {T_END:.0f} s, long-crested SS5, "
          f"compared at VMG = {VMG_TARGET} m/s toward a destination dead upwind\n")

    table = {}
    for mu in HEADINGS:
        curve = []
        for th in THRUSTS:
            a = agg([run(db, sd, mu, th) for sd in SEEDS])
            curve.append(a)
        table[mu] = curve
        v = [c["vmg"] for c in curve]
        print(f"  {mu:>3}deg  VMG range {min(v):.2f} - {max(v):.2f} m/s"
              + ("" if max(v) >= VMG_TARGET else "   <-- CANNOT REACH TARGET"))

    print(f"\n  every metric at VMG = {VMG_TARGET} m/s, "
          f"relative to head seas (%)\n")
    print(f"  {'metric':<16}" + "".join(f"{str(m)+'d':>12}" for m in HEADINGS))
    ref = {}
    out_rows = []
    for k in KEYS:
        line = f"  {k:<16}"
        for mu in HEADINGS:
            c = table[mu]
            vs = np.array([x["vmg"] for x in c])
            ys = np.array([x[k] for x in c])
            o = np.argsort(vs)
            if VMG_TARGET > vs.max() or VMG_TARGET < vs.min():
                line += f"{'n/a':>12}"
                continue
            val = float(np.interp(VMG_TARGET, vs[o], ys[o]))
            se = float(np.mean([x[k + "_se"] for x in c]))
            if mu == HEADINGS[0]:
                ref[k] = (val, se)
                line += f"{val:>12.3f}"
            else:
                r, rse = ref[k]
                d = 100 * (val - r) / max(abs(r), 1e-9)
                dse = 100 * np.sqrt(se ** 2 + rse ** 2) / max(abs(r), 1e-9)
                line += f"{d:>+10.1f}{'*' if abs(d) > 2*dse else ' '} "
                out_rows.append((k, mu, d, dse))
        print(line)

    print("\n  first column is the absolute value in head seas; the rest are")
    print("  percentage changes from it. * = beyond twice its standard error.")

    wins = [(k, mu, d) for k, mu, d, dse in out_rows
            if d < 0 and abs(d) > 2 * dse]
    print(f"\n  resolved IMPROVEMENTS from turning off head seas: {len(wins)}")
    if wins:
        best = min(wins, key=lambda r: r[2])
        print(f"  largest: {best[0]} {best[2]:+.1f}% at {best[1]} deg")
        print("\n  => heading is a real lever, and it costs only distance made")
        print("     good -- which is a ROUTING decision, not a sensing one. The")
        print("     wave direction it needs is a slow statistic the hull already")
        print("     measures from its own motion.")
    else:
        print("\n  => even heading does nothing at equal VMG. The extra speed")
        print("     needed to hold progress cancels the gentler encounter.")
    json.dump({str(k): v for k, v in table.items()},
              open("heading_choice.json", "w"), indent=1, default=float)
    return table


if __name__ == "__main__":
    main()
