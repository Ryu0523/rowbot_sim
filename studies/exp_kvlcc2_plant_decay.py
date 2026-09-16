#!/usr/bin/env python3
"""
Experiment: the PLANT against KVLCC2's measured roll decays, end to end.

exp_kvlcc2_roll_prediction.py predicted KVLCC2's roll damping in the
frequency domain (BEM wave part + simplified Ikeda): 1.33-1.65x the measured
equivalent damping at 0 kn, 0.84-0.94x at 15.5 kn. That is the method on
paper. This is the method as the simulation runs it: the time-domain plant --
radiation memory, nonlinear hydrostatics, whichever roll-damping model the
Hull selects -- released from each run's measured first extremum, and the
SAME equation fitted to its output as to the measurement
(studies.exp_kvlcc2_rolldecay.fit).

Roll damping beyond the BEM's wave part, three ways:
  placeholder   the 10 m USV's quadratic coefficient, Froude-scaled -- what
                roll_damping="auto" used to fall back to, KVLCC2's C_M being
                outside the simplified Ikeda range (0.9-0.99)
  ikeda         simplified Ikeda although out of range -- what "auto" is now
  decay fit     the run's own fitted zeta_1 and k2. NOT a prediction: it
                checks that a decay test entered in the Hull
                (roll_damping=dict(zeta1=, k2=, T=)) comes back out of the
                plant unchanged

At 15.5 kn the model is towed: surge held at the measured speed, sway and yaw
held as well (how SSPA restrained the model is not in the release).

Anchor: the SSPA time series (Mendeley, CC BY 4.0). Everything else is model.

Run: python -m studies.exp_kvlcc2_plant_decay
"""
import numpy as np

from studies.exp_kvlcc2_rolldecay import RUNS, decay_window, fit, load, zeta_eq

KNOT = 0.514444
SCALE = 68.0


def release(h, db, phi0, U, t):
    """The plant released from roll phi0 at speed U, sampled at times t."""
    from sim.forces import Wind
    from sim.test_vessel import Monochromatic
    captive = ("surge", "sway", "yaw") if U > 0 else None
    v = h.plant(Monochromatic(1.0, 0.0), db=db, wind=Wind(), captive=captive)
    s = v.initial_state(U)
    s[3] = phi0
    n = int(np.ceil(t[-1] / v.dt)) + 1
    tt = np.arange(n) * v.dt
    phi = np.empty(n)
    for i in range(n):
        phi[i] = s[3]
        s = v.step(s, tt[i], 0.0, 0.0)
    return np.interp(t, tt, phi), v.roll_source


def main():
    import warnings
    warnings.filterwarnings("ignore")
    from hydro import hulls
    from sim import config
    h = hulls.kvlcc2_68()
    _, db = config.load(h)
    print("\nEXPERIMENT -- the time-domain plant against KVLCC2's measured "
          "roll decays (SSPA 1:68)\n")
    out = {}
    for run, kn in RUNS.items():
        t, phi = decay_window(*load(run))
        meas = fit(t, phi)
        U = kn * KNOT / np.sqrt(SCALE)
        res = {"measured": dict(meas, rms_vs_meas=0.0)}
        for lab, mode in (("placeholder", "placeholder"), ("ikeda", "ikeda"),
                          ("decay fit", dict(zeta1=meas["z1"], k2=meas["k2"],
                                             T=meas["T"]))):
            h.roll_damping = mode
            ph, src = release(h, db, phi[0], U, t)
            r = fit(t, ph)
            r["rms_vs_meas"] = float(np.sqrt(np.mean((ph - phi) ** 2)))
            r["source"] = src
            res[lab] = r
        out[run] = res
        m5 = zeta_eq(meas, 5)
        print(f"  run {run}, {kn:.1f} kn (model {U:.3f} m/s), released at "
              f"{np.degrees(phi[0]):+.2f} deg, {t[-1]:.0f} s")
        print(f"    {'':<13}{'T s':>7}{'zeta_1':>9}{'k2':>8}{'z_eq 2':>8}"
              f"{'z_eq 5':>8}{'z_eq 9':>8}{'/meas@5':>9}{'rms deg':>9}")
        for lab, r in res.items():
            print(f"    {lab:<13}{r['T']:>7.3f}{r['z1']:>9.4f}{r['k2']:>8.3f}"
                  f"{zeta_eq(r, 2):>8.4f}{zeta_eq(r, 5):>8.4f}"
                  f"{zeta_eq(r, 9):>8.4f}{zeta_eq(r, 5) / m5:>9.2f}"
                  f"{np.degrees(r['rms_vs_meas']):>9.3f}")
        print()
    print("  rms deg: the plant's time series against the measured one, "
          "sample by sample\n  (a period error alone makes it grow along "
          "the record).")
    return out


if __name__ == "__main__":
    main()
