#!/usr/bin/env python3
"""
Is the controller actually USING the preview?

A null result on the preview sweep has two very different explanations:

  (a) preview genuinely has little value for this hull, or
  (b) the controller cannot exploit it.

They look identical in the metrics, so this separates them by looking at the
COMMAND rather than the outcome. If the thrust command has the same statistics
with and without preview, the information never reached the actuator and the
sweep measured the controller, not the physics.

Three signatures are checked:
  * command variability -- a preview-exploiting controller modulates thrust more
  * correlation with the wave AHEAD of the vessel, at the preview horizon
  * lead time of peak correlation -- it should be positive and grow with the
    horizon if the controller is anticipating rather than reacting
"""
import numpy as np

from hydro import bem
from sim.env import Episode
from sim.vessel import NonlinearVessel
from sim.test_vessel import Monochromatic
from control.reduced import ReducedModel

DB = "hydro_wigley_10m.npz"
G = 9.81


def analyse(db, red, t_preview, seed=0, t_end=200.0, u_ref=4.5):
    ep = Episode(db, red, seed=seed, t_preview=t_preview, u_ref=u_ref,
                 n_dir=1, use_rudder=False)
    m = ep.run(t_end, trace=True)
    tr = m["trace"]
    t, x, thrust = tr[:, 0], tr[:, 1], tr[:, 6]
    th = thrust / 12000.0

    # wave elevation AT the vessel, as a reference signal
    eta_here = np.array([ep.sea.eta(np.atleast_1d(x[i]), np.zeros(1), t[i])[0]
                         for i in range(0, len(t), 4)])
    thr_s = th[::4] - np.mean(th[::4])
    eta_s = eta_here - np.mean(eta_here)
    dt = t[4] - t[0]

    # cross-correlation: positive lag = thrust LEADS the wave (anticipation)
    n = len(thr_s)
    lags = np.arange(-int(20 / dt), int(20 / dt) + 1)
    denom = np.std(thr_s) * np.std(eta_s) * n
    cc = np.array([np.sum(np.roll(thr_s, -k) * eta_s) / max(denom, 1e-12)
                   for k in lags])
    best = lags[int(np.argmax(np.abs(cc)))] * dt
    return dict(t_preview=t_preview, thrust_std=float(np.std(th)),
                thrust_range=float(np.ptp(th)), lead_s=float(best),
                peak_corr=float(np.max(np.abs(cc))),
                u_mean=m["u_mean"], acc_p99=m["acc_p99"])


def main():
    db = bem.load(DB)
    red = ReducedModel.identify(
        NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L), db)
    print("Does the thrust command change when preview is available?\n")
    print(f"  {'T_prev':>7}{'thrust std':>12}{'range':>9}{'peak |corr|':>13}"
          f"{'lead s':>9}{'u':>7}{'acc_p99':>9}")
    rows = []
    for tp_ in (0.0, 4.0, 8.0):
        r = analyse(db, red, tp_)
        rows.append(r)
        print(f"  {tp_:>7.1f}{r['thrust_std']:>12.4f}{r['thrust_range']:>9.3f}"
              f"{r['peak_corr']:>13.3f}{r['lead_s']:>9.1f}"
              f"{r['u_mean']:>7.2f}{r['acc_p99']:>9.3f}")

    base, prev = rows[0], rows[-1]
    ratio = prev["thrust_std"] / max(base["thrust_std"], 1e-9)
    print(f"\n  thrust variability with 8 s preview / without = {ratio:.2f}x")
    used = ratio > 1.3 or abs(prev["lead_s"]) > abs(base["lead_s"]) + 1.0
    if used:
        print("  -> the controller IS modulating on the preview. A null result "
              "in the sweep is then about the physics, not the plumbing.")
    else:
        print("  -> the command barely changes. The preview is NOT reaching "
              "the actuator, so the sweep measured the controller, not the "
              "value of preview. Fix the controller before concluding "
              "anything about sensing.")
    return rows


if __name__ == "__main__":
    main()
