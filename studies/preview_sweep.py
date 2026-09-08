#!/usr/bin/env python3
"""
M7 -- what is wave preview worth, on the verified model?

This repeats the Phase 0 study on the real thing: nonlinear Cummins plant with
identified actuators and added resistance, instead of the invented second-order
oscillator. The Phase 0 answer -- a knee around 4 s, with the gain in slamming
rather than in RMS motion -- was always provisional on exactly this.

The design that mattered in Phase 0 still matters here: a controller can always
reduce motion by going slower, so preview is only worth something if it beats
the no-information baseline AT THE SAME SPEED. The baseline is therefore a
sweep of constant thrust settings, and preview has to sit below that curve.

Cross-reference for the sensing decision: hydrodynamic memory measured in M2 is
2.2-2.9 s, so any preview horizon below that is fighting the hull's own inertia
rather than the sea.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hydro import bem
from sim.env import Episode
from sim.vessel import NonlinearVessel
from sim.test_vessel import Monochromatic
from control.reduced import ReducedModel

DB = "hydro_wigley_10m.npz"
HORIZONS = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0]
# Twelve, not three. The slam COUNT needed far more than this to resolve
# anything (see DEFECTS.md A22); the continuous severity metrics do not, but
# the seed spread is what puts an error bar on every number here, and three
# seeds cannot estimate a spread.
SEEDS = list(range(12))
T_END = 200.0
U_REF = 4.5
N_DIR = 1          # long-crested head seas -- see the note below

# Scoped to LONG-CRESTED HEAD SEAS on purpose. Short-crested seas excite sway,
# and the MPC's reduced model has no sway state, so it can only chase the
# resulting drift with heading. That confounds the speed/motion trade-off this
# study is about. It also surfaced a real coupling worth carrying forward:
# rudder lift scales with u^2, so slowing down for ride quality costs steering
# authority. Adding sway to the reduced model is the fix, and belongs with the
# heading-control work rather than here.
#
# The rudder is LOCKED for the same reason, and it is not optional: rudder lift
# makes sway as well as yaw, the reduced model has no sway state, so the
# controller cannot see the drift it is causing. Left free it produced 51 m of
# cross-track in long-crested head seas, where there is no lateral wave
# excitation at all -- the drift was entirely self-inflicted.


def constant_thrust_baseline(db, thrusts=(3000, 5000, 7000, 9000, 11000),
                             seeds=SEEDS, t_end=T_END, hs=3.25, tp=9.7):
    from sim.wavefield import SeaState
    out = []
    for th in thrusts:
        rows = []
        for sd in seeds:
            sea = SeaState(hs, tp, n_freq=32, n_dir=N_DIR, seed=sd)
            v = NonlinearVessel(db, sea, L=db.L,
                                B=float(db.attrs.get("B", 2.5)),
                                T=float(db.attrs.get("T", 0.8)), dt=0.05)
            s = v.initial_state(2.0)
            n = int(t_end / 0.05)
            acc, spd, sf = np.empty(n), np.empty(n), np.empty(n)
            t = 0.0
            for i in range(n):
                s = v.step(s, t, th, 0.0, 0.05)
                acc[i] = v.last_bow_acc
                spd[i] = s[6]
                sf[i] = v.last_slam_force
                t += 0.05
            rows.append(dict(u_mean=float(np.mean(spd[n // 4:])),
                             acc_p99=float(np.percentile(np.abs(acc), 99) / 9.81),
                             acc_rms=float(np.sqrt(np.mean(acc ** 2)) / 9.81),
                             slam_per_min=v.slam_count / (t / 60.0),
                             slam_p99=float(np.percentile(sf, 99) / 1e3),
                             slam_impulse=float(sf.sum() * 0.05
                                                / (t / 60.0) / 1e3)))
        out.append({k: float(np.mean([r[k] for r in rows])) for k in rows[0]})
        out[-1].update({k + "_se": float(np.std([r[k] for r in rows], ddof=1)
                                         / np.sqrt(len(rows))) for k in rows[0]})
        print(f"  thrust {th:6.0f} N -> u {out[-1]['u_mean']:.2f} m/s, "
              f"acc_p99 {out[-1]['acc_p99']:.3f} g, "
              f"slam {out[-1]['slam_per_min']:.1f}/min")
    return out


def main():
    db = bem.load(DB)
    plant = NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L)
    print("identifying reduced model for the MPC ...")
    red = ReducedModel.identify(plant, db)

    print("\nno-information baseline (constant thrust):")
    base = constant_thrust_baseline(db)

    print("\npreview sweep (MPC, oracle preview):")
    print(f"  {'T_prev':>7}{'u_mean':>9}{'acc_rms':>9}{'acc_p99':>9}"
          f"{'slam/min':>10}{'slamP99':>9}{'slamImp':>9}{'cross':>8}")
    rows = []
    for tp_ in HORIZONS:
        got = []
        for sd in SEEDS:
            ep = Episode(db, red, seed=sd, t_preview=tp_, u_ref=U_REF,
                         n_dir=N_DIR, use_rudder=False)
            got.append(ep.run(T_END))
        keys = ("u_mean", "acc_rms", "acc_p99", "slam_per_min",
                "cross_rms", "slam_p99", "slam_impulse")
        m = {k: float(np.mean([g[k] for g in got])) for k in keys}
        # standard error across seeds: without it none of these numbers can be
        # compared to another, which is exactly how the first version of this
        # study reported a slamming result that was one sigma of counting noise
        m.update({k + "_se": float(np.std([g[k] for g in got], ddof=1)
                                   / np.sqrt(len(got))) for k in keys})
        m["t_preview"] = tp_
        rows.append(m)
        print(f"  {tp_:>7.1f}{m['u_mean']:>9.2f}{m['acc_rms']:>9.3f}"
              f"{m['acc_p99']:>9.3f}{m['slam_per_min']:>10.2f}"
              f"{m['slam_p99']:>9.1f}{m['slam_impulse']:>9.1f}"
              f"{m['cross_rms']:>8.2f}")

    json.dump(dict(baseline=base, preview=rows),
              open("preview_sweep.json", "w"), indent=1)
    _plot(base, rows)
    _readout(base, rows)
    return base, rows


def _interp_baseline(base, key, u):
    us = np.array([b["u_mean"] for b in base])
    ys = np.array([b[key] for b in base])
    o = np.argsort(us)
    return float(np.interp(u, us[o], ys[o]))


def _readout(base, rows):
    print("\n  value of preview AT EQUAL SPEED (vs constant-thrust baseline)")
    print("  every delta carries +-1 standard error over the seed set; a delta")
    print("  smaller than its own error bar is not a result.\n")
    for key, unit in (("acc_p99", "g"), ("slam_impulse", "kN.s/min"),
                      ("slam_p99", "kN"), ("slam_per_min", "/min")):
        print(f"  {key} [{unit}]")
        print(f"    {'T_prev':>7}{'u':>7}{'value':>10}{'+-':>7}"
              f"{'baseline':>10}{'delta%':>9}{'+-':>7}{'resolved':>10}")
        for r in rows:
            b = _interp_baseline(base, key, r["u_mean"])
            v, se = r[key], r.get(key + "_se", 0.0)
            # the baseline is interpolated between thrust settings, so give it
            # the mean standard error of the bracket rather than pretending it
            # is exact
            bse = float(np.mean([x.get(key + "_se", 0.0) for x in base]))
            d = 100 * (v - b) / max(abs(b), 1e-9)
            dse = 100 * np.sqrt(se ** 2 + bse ** 2) / max(abs(b), 1e-9)
            print(f"    {r['t_preview']:>7.1f}{r['u_mean']:>7.2f}{v:>10.3f}"
                  f"{se:>7.3f}{b:>10.3f}{d:>9.1f}{dse:>7.1f}"
                  f"{('yes' if abs(d) > 2 * dse else 'no'):>10}")
        print()


def _plot(base, rows):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    bu = [b["u_mean"] for b in base]
    cm = plt.cm.viridis(np.linspace(0, 1, len(rows)))
    for i, (key, lab) in enumerate([("acc_p99", "bow accel 99th pct (g)"),
                                    ("slam_per_min", "slams per minute")]):
        ax[i].plot(bu, [b[key] for b in base], "k-o", lw=1.8, ms=5,
                   label="no information (constant thrust)")
        for c, r in zip(cm, rows):
            ax[i].scatter([r["u_mean"]], [r[key]], s=80, color=c,
                          edgecolor="k", zorder=3,
                          label=f"preview {r['t_preview']:g} s")
        ax[i].set_xlabel("mean speed (m/s)  $\\rightarrow$ better")
        ax[i].set_ylabel(lab + "  $\\downarrow$ better")
        ax[i].grid(alpha=.3)
    ax[0].legend(fontsize=7)
    h = [r["t_preview"] for r in rows]
    ax[2].plot(h, [r["slam_per_min"] for r in rows], "-o", c="tab:purple",
               label="slams/min")
    ax[2].plot(h, [r["acc_p99"] for r in rows], "-s", c="tab:red",
               label="accel p99 (g)")
    ax[2].set_xlabel("preview horizon (s)")
    ax[2].axvspan(2.2, 2.9, color="tab:green", alpha=.15)
    ax[2].annotate("hydrodynamic\nmemory (M2)", (2.9, ax[2].get_ylim()[1] * .7),
                   fontsize=8, color="tab:green")
    ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)
    fig.suptitle("M7: value of wave preview on the verified nonlinear model "
                 "(SS5, Wigley 10 m)", fontsize=12)
    fig.tight_layout()
    fig.savefig("fig11_m7_preview.png", dpi=140)
    print("\n  figure: fig11_m7_preview.png")


if __name__ == "__main__":
    main()
