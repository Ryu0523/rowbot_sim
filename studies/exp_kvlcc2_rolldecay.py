#!/usr/bin/env python3
"""
Experiment: roll decay of KVLCC2, measured -- what the model must reproduce.

THE DATA
SSPA model tests of KVLCC2 at scale 1:68, released by Alexandersson and
Kjellberg (Chalmers) on Mendeley Data under CC BY 4.0
(data/external/kvlcc2_rolldecay). Three roll decays, roll angle phi(t):
  21337, 21338   0 kn
  21340          15.5 kn full scale (0.967 m/s model)
Loading condition, full scale (model_test_parameters.csv): Lpp 320 m, B 58 m,
T 20.8 m, KG 18.6 m, GM 5.73 m, kxx 23.2 m (0.4 B), volume 312653 m3.
Bilge keels: the BKL/BKB columns are EMPTY. Whether the model had them is not
recorded in the release, and that is stated wherever it matters.

HOW THE NUMBERS ARE EXTRACTED
The damping here is about 1% of critical: a few percent of amplitude lost per
cycle. Per-half-cycle logarithmic decrements, the textbook reduction, are
dominated by noise at that level -- the first version of this file used them
and got the two 0 kn repeats disagreeing by a factor of two in damping, and
with opposite trends in amplitude. That was the ruler, not the ship.

So the whole decay is fitted instead, by simulation, to

    phi'' + 2 zeta_1 w0 phi' + k2 phi'|phi'| + w0^2 (phi - phi_0) = 0

with w0, zeta_1, k2, the offset phi_0 and the initial state all free
(least squares on the time series). The equivalent linear damping at
amplitude phi_a is then

    zeta_eq(phi_a) = zeta_1 + (4 / 3pi) k2 phi_a

-- the same linear-plus-quadratic form the vessel model uses. The two 0 kn
runs are repeats of one test, so their agreement is the measure of how well
this is known.

Run: python -m studies.exp_kvlcc2_rolldecay
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "external", "kvlcc2_rolldecay")
SCALE = 68.0
RUNS = {"21337": 0.0, "21338": 0.0, "21340": 15.5}


def load(run):
    import pandas as pd
    d = pd.read_csv(os.path.join(DATA, f"model_test_{run}.csv"))
    return d["time"].values, d["phi"].values


def decay_window(t, phi):
    """From the first extremum after the excitation to the end of the record."""
    i0 = int(np.argmax(np.abs(phi)))
    return t[i0:] - t[i0], phi[i0:]


def simulate(p, t):
    from scipy.integrate import solve_ivp
    w0, z1, k2, off, x0, v0 = p

    def f(_, y):
        return [y[1], -2 * z1 * w0 * y[1] - k2 * y[1] * abs(y[1])
                - w0 * w0 * (y[0] - off)]
    s = solve_ivp(f, (t[0], t[-1]), [x0, v0], t_eval=t, rtol=1e-8,
                  atol=1e-10, max_step=0.05)
    return s.y[0] if s.success else np.full_like(t, np.nan)


def fit(t, phi):
    from scipy.optimize import least_squares
    # starting point from zero crossings
    s = np.sign(phi - np.median(phi))
    zc = np.where(s[1:] * s[:-1] < 0)[0]
    T0 = 2.0 * float(np.median(np.diff(t[zc])))
    p0 = [2 * np.pi / T0, 0.01, 0.01, float(np.median(phi)), phi[0], 0.0]
    lo = [0.5 * p0[0], 0.0, 0.0, -0.05, -0.5, -1.0]
    hi = [1.5 * p0[0], 0.2, 5.0, 0.05, 0.5, 1.0]
    r = least_squares(lambda p: simulate(p, t) - phi, p0, bounds=(lo, hi),
                      x_scale=[0.1, 0.01, 0.1, 0.01, 0.1, 0.1])
    rms = float(np.sqrt(np.mean(r.fun ** 2)))
    return dict(w0=r.x[0], T=2 * np.pi / r.x[0], z1=r.x[1], k2=r.x[2],
                offset=r.x[3], rms=rms, amp0=float(np.abs(phi[0])))


def zeta_eq(r, amp_deg):
    return r["z1"] + 4.0 / (3.0 * np.pi) * r["k2"] * np.radians(amp_deg)


def main():
    print("\nEXPERIMENT -- KVLCC2 roll decay, measured (SSPA, 1:68), "
          "fitted by simulation\n")
    out = {}
    print(f"  {'run':<7}{'kn':>5}{'T model s':>11}{'T full s':>10}"
          f"{'zeta_1':>9}{'k2':>8}{'z_eq 2deg':>10}{'z_eq 5deg':>10}"
          f"{'z_eq 9deg':>10}{'fit rms deg':>12}")
    for run, kn in RUNS.items():
        t, phi = decay_window(*load(run))
        r = fit(t, phi)
        out[run] = r
        print(f"  {run:<7}{kn:>5.1f}{r['T']:>11.3f}{r['T']*np.sqrt(SCALE):>10.2f}"
              f"{r['z1']:>9.4f}{r['k2']:>8.3f}{zeta_eq(r, 2):>10.4f}"
              f"{zeta_eq(r, 5):>10.4f}{zeta_eq(r, 9):>10.4f}"
              f"{np.degrees(r['rms']):>12.3f}")
    a, b = out["21337"], out["21338"]
    print(f"\n  repeatability of the two 0 kn runs:")
    for amp in (2, 5, 9):
        za, zb = zeta_eq(a, amp), zeta_eq(b, amp)
        print(f"    zeta_eq at {amp} deg: {za:.4f} vs {zb:.4f}  "
              f"({abs(za - zb) / (0.5 * (za + zb)):.0%} apart)")
    print(f"    natural period: {a['T']:.3f} vs {b['T']:.3f} s")
    s = out["21340"]
    print(f"\n  forward speed 15.5 kn raises zeta_eq at 5 deg from "
          f"{0.5*(zeta_eq(a,5)+zeta_eq(b,5)):.4f} to {zeta_eq(s,5):.4f}")
    return out


if __name__ == "__main__":
    main()
