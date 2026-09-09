#!/usr/bin/env python3
"""
Is there exploitable information in the wave field beyond eta_bar and slope?

The controller compresses the whole previewed sea into two scalars per step:
the mean elevation under the hull, and its slope along the hull. That is nearly
lossless for the INSTANTANEOUS force -- a rigid body in a long wave responds
only to the 0th and 1st moments of pressure over its length, and there is
nothing for curvature to push on.

It is badly lossy in one specific place: the MPC's horizon is 12 s, and
anything past that is discarded outright. But a seaway has structure on a much
longer timescale. Waves arrive in GROUPS of 50-100 s, and the surge time
constant is 6.3 s -- eight to sixteen times shorter. The vessel cannot dodge a
wave, but it is easily fast enough to be slower during a big group.

That is precisely the feature a learned policy could use and this MPC
structurally cannot represent. Before building an RL pipeline to look for it,
this asks the cheaper question: IS THE INFORMATION THERE AT ALL?

A hand-built scheduler, no optimiser and no learning:

    look ahead T seconds along the track, measure how rough it is,
    and back off the throttle in proportion.

If a five-line controller can beat constant thrust at equal speed, the feature
is real and RL is worth building. If it cannot, the information is not there,
and no amount of feature engineering or policy capacity will conjure it -- which
is a far cheaper thing to learn now than after training runs.

Run: python -m studies.envelope_feature
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
SEEDS = list(range(16))
T_END = 300.0
HS, TP = 3.25, 9.7
DT = 0.05
DT_CTRL = 0.5

# thrust settings for the fixed-throttle reference curve
THRUSTS = (3000, 4000, 5000, 7000, 9000, 11000)

# (look-ahead seconds, gain). gain 0 reproduces constant thrust exactly, which
# is the control that proves the harness itself adds nothing.
CASES = [("constant (gain 0)", 40.0, 0.0),
         ("look 20 s, gain 0.5", 20.0, 0.5),
         ("look 40 s, gain 0.5", 40.0, 0.5),
         ("look 40 s, gain 1.0", 40.0, 1.0),
         ("look 70 s, gain 1.0", 70.0, 1.0),
         ("look 40 s, gain 2.0", 40.0, 2.0)]

KEYS = ("acc_rms", "acc_p99", "fatigue", "rvm_rms", "rvv_rms",
        "slam_impulse", "slam_p99", "slam_rate_ochi", "wet_rate")


def roughness_ahead(sea, x0, y0, t0, u, t_look, n=40):
    """RMS wave elevation over the water the vessel is about to cross.

    This is the whole 'deep feature': one number, summarising a stretch of sea
    the MPC's 12 s horizon never sees. It needs no phase resolution -- only how
    big the next group is -- which is also why it would be a far easier thing to
    measure with real hardware than wave-by-wave preview.
    """
    dt = t_look / n
    ts = t0 + np.arange(1, n + 1) * dt
    xs = x0 + u * (ts - t0)
    # each sample is taken where the vessel will be AND when it gets there --
    # the sea moves too, so evaluating the whole track at t0 would be a
    # snapshot of water the vessel never meets
    e = np.array([sea.eta(np.array([xs[i]]), np.array([y0]), ts[i])[0]
                  for i in range(n)])
    return float(np.sqrt(np.mean(e ** 2)))


def run(db, seed, t_look, gain, thrust0):
    sea = SeaState(HS, TP, n_freq=32, n_dir=1, seed=seed)
    v = NonlinearVessel(db, sea, L=db.L, B=float(db.attrs.get("B", 2.5)),
                        T=float(db.attrs.get("T", 0.8)), dt=DT)
    s = v.initial_state(2.0)
    n = int(T_END / DT)
    sub = int(round(DT_CTRL / DT))
    acc = np.empty(n); spd = np.empty(n); rel = np.empty(n); sf = np.empty(n)
    ref = HS / 4.0                       # the sea's own rms, known to the boat
    t = 0.0
    thrust = thrust0
    for i in range(n):
        if i % sub == 0 and gain != 0.0:
            r = roughness_ahead(sea, s[0], s[1], t, max(s[6], 1.0), t_look)
            thrust = thrust0 * (1.0 - gain * (r / ref - 1.0))
            thrust = float(np.clip(thrust, 0.15 * thrust0, 1.6 * thrust0))
            thrust = min(thrust, 12000.0)
        s = v.step(s, t, thrust, 0.0, DT)
        acc[i] = v.last_bow_acc; spd[i] = s[6]
        rel[i] = v.last_rel_bow; sf[i] = v.last_slam_force
        t += DT
    m = seakeeping.summarise(rel, acc / G, DT, v.T, FREEBOARD, v.v_slam)
    m.update(u_mean=float(np.mean(spd[n // 4:])),
             slam_impulse=float(sf.sum() * DT / (t / 60.0) / 1e3),
             slam_p99=float(np.percentile(sf, 99) / 1e3))
    return m


def aggregate(rows):
    m = {k: float(np.mean([r[k] for r in rows])) for k in KEYS + ("u_mean",)}
    m.update({k + "_se": float(np.std([r[k] for r in rows], ddof=1)
                               / np.sqrt(len(rows))) for k in KEYS + ("u_mean",)})
    return m


def interp(base, key, u):
    us = np.array([b["u_mean"] for b in base])
    o = np.argsort(us)
    if not (us.min() - 1e-9 <= u <= us.max() + 1e-9):
        raise ValueError(f"speed {u:.2f} outside baseline bracket "
                         f"[{us.min():.2f}, {us.max():.2f}]")
    return (float(np.interp(u, us[o], np.array([b[key] for b in base])[o])),
            float(np.mean([b[key + "_se"] for b in base])))


def main():
    db = bem.load(DB)
    print(f"{len(SEEDS)} seeds x {T_END:.0f} s, long-crested head seas")
    print("  fixed-throttle reference curve:")
    base = []
    for th in THRUSTS:
        b = aggregate([run(db, sd, 0.0, 0.0, th) for sd in SEEDS])
        base.append(b)
        print(f"    {th/1000:>5.1f} kN -> u {b['u_mean']:.2f}")

    rows = []
    for label, t_look, gain in CASES:
        m = aggregate([run(db, sd, t_look, gain, 7000.0) for sd in SEEDS])
        m["label"] = label
        rows.append(m)

    print("\n  delta vs fixed throttle AT EQUAL SPEED")
    print(f"  {'metric':<16}" + "".join(f"{c[0][:19]:>21}" for c in CASES[:3]))
    n_better = 0
    for k in KEYS:
        line = f"  {k:<16}"
        for m in rows[:3]:
            b, bse = interp(base, k, m["u_mean"])
            d = 100 * (m[k] - b) / max(abs(b), 1e-9)
            dse = 100 * np.sqrt(m[k + "_se"] ** 2 + bse ** 2) / max(abs(b), 1e-9)
            line += f"{d:>+12.1f} ±{dse:<4.1f}{'*' if abs(d) > 2*dse else ' '}"
        print(line)

    print(f"\n  {'case':<22}{'u':>6}" + "".join(f"{k[:9]:>11}" for k in
                                               ("acc_rms", "rvm_rms",
                                                "slam_impulse", "wet_rate")))
    for m in rows:
        cells = ""
        for k in ("acc_rms", "rvm_rms", "slam_impulse", "wet_rate"):
            b, bse = interp(base, k, m["u_mean"])
            d = 100 * (m[k] - b) / max(abs(b), 1e-9)
            dse = 100 * np.sqrt(m[k + "_se"] ** 2 + bse ** 2) / max(abs(b), 1e-9)
            res = abs(d) > 2 * dse
            n_better += res and d < 0
            cells += f"{d:>+9.1f}{'*' if res else ' '} "
        print(f"  {m['label']:<22}{m['u_mean']:>6.2f}{cells}")

    print("\n  * = exceeds twice its own standard error")
    print(f"  resolved IMPROVEMENTS across all cases: {n_better}")
    if n_better == 0:
        print("\n  => the group envelope carries no usable signal for this vessel.")
        print("     A learned policy would be searching an empty room: the")
        print("     information is not in the input, so no encoder recovers it.")
    else:
        print("\n  => there IS exploitable structure beyond the 12 s horizon.")
        print("     A five-line scheduler found it, so a learned policy with")
        print("     richer features is worth building.")
    json.dump(dict(base=base, rows=rows), open("envelope_feature.json", "w"),
              indent=1, default=float)
    return rows


if __name__ == "__main__":
    main()
