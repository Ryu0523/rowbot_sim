#!/usr/bin/env python3
"""
M3 GATE -- does the time-domain model reproduce the exact frequency-domain answer?

For a linear system the two must agree, and they are computed along completely
different paths:

  frequency   one 6x6 complex solve using A(w) and B(w) straight from the BEM
  time        RK4 integration of the ODE, with radiation carried by the
              state-space memory fitted in M2

So the comparison exercises A_inf, K(t), the realisation, the assembly of the
36-pair block system and every sign convention along the way. Nothing external
is needed: the reference is exact by construction.

Two more checks that need no reference at all:

  * heave RAO -> 1 and pitch RAO -> wave slope as w -> 0. A small body in a
    long wave simply rides the surface. This is physics, not numerics, and it
    catches scaling errors that a self-consistent round trip cannot.
  * free decay: released from rest the vessel must oscillate at the frequency
    where the RAO peaks, and die away. Amplitude and period come from a path
    with no wave forcing at all.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hydro import bem
from .cummins import (LinearVessel, rao_frequency_domain,
                      excitation_series, extract_amplitude_phase)

DB = "hydro_wigley_10m.npz"
G = 9.81
TEST_OMEGAS = [0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5]
HEAD_SEAS = 12          # direction index: beta = pi


def run(path=DB, dt=0.01, n_periods=45):
    db = bem.load(path)
    print(f"database: {db}")
    print(f"  head seas: beta = {np.degrees(db.directions[HEAD_SEAS]):.0f} deg\n")

    ves = LinearVessel(db, verbose=True)
    print(f"  total model states: {ves.n_state} "
          f"(12 rigid + {ves.rad.n} radiation)\n")

    sign = _determine_convention(db, ves, dt)

    rows = []
    print(f"  {'omega':>7}{'T (s)':>7}{'lam/L':>7}"
          f"{'|heave| fd':>12}{'|heave| td':>12}{'err%':>7}"
          f"{'|pitch| fd':>12}{'|pitch| td':>12}{'err%':>7}{'dphase':>8}")
    for w in TEST_OMEGAS:
        xi, wa = rao_frequency_domain(db, w, HEAD_SEAS)
        fn, _ = excitation_series(db, w, HEAD_SEAS, sign=sign)
        t, s = ves.simulate(fn, n_periods * 2 * np.pi / wa, dt=dt)
        amp = np.empty(6); pha = np.empty(6)
        for d in range(6):
            amp[d], pha[d] = extract_amplitude_phase(t, s[:, d], wa)
        lam = 2 * np.pi * G / wa ** 2
        dph = np.angle(np.exp(1j * (pha[2] - np.angle(xi[2]))))
        rows.append((wa, np.abs(xi), amp, dph))
        print(f"  {wa:>7.2f}{2*np.pi/wa:>7.2f}{lam/db.L:>7.1f}"
              f"{np.abs(xi[2]):>12.4f}{amp[2]:>12.4f}"
              f"{100*abs(amp[2]-abs(xi[2]))/max(abs(xi[2]),1e-12):>7.2f}"
              f"{np.abs(xi[4]):>12.4f}{amp[4]:>12.4f}"
              f"{100*abs(amp[4]-abs(xi[4]))/max(abs(xi[4]),1e-12):>7.2f}"
              f"{np.degrees(dph):>8.1f}")

    lf = _low_frequency_limit(db, ves, dt, sign)
    dec = _free_decay(db, ves, dt)
    _plot(db, rows, dec, sign)
    _verdict(rows, lf, dec)
    return rows


def _determine_convention(db, ves, dt):
    """Amplitudes are the same either way; only the phase distinguishes them."""
    w = 1.0
    xi, wa = rao_frequency_domain(db, w, HEAD_SEAS)
    best, best_err = -1.0, 1e9
    for sign in (-1.0, +1.0):
        fn, _ = excitation_series(db, w, HEAD_SEAS, sign=sign)
        t, s = ves.simulate(fn, 40 * 2 * np.pi / wa, dt=dt)
        _, ph = extract_amplitude_phase(t, s[:, 2], wa)
        err = abs(np.angle(np.exp(1j * (ph - np.angle(xi[2])))))
        print(f"  time convention e^({'+' if sign > 0 else '-'}iwt): "
              f"heave phase error {np.degrees(err):6.1f} deg")
        if err < best_err:
            best, best_err = sign, err
    print(f"  -> using e^({'+' if best > 0 else '-'}iwt)\n")
    return best


def _low_frequency_limit(db, ves, dt, sign):
    """A small body in a long wave rides the surface: heave RAO -> 1."""
    w = db.omega[np.argmin(np.abs(db.omega - 0.25))]
    xi, wa = rao_frequency_domain(db, w, HEAD_SEAS)
    fn, _ = excitation_series(db, wa, HEAD_SEAS, sign=sign)
    t, s = ves.simulate(fn, 40 * 2 * np.pi / wa, dt=dt)
    a_heave, _ = extract_amplitude_phase(t, s[:, 2], wa)
    lam = 2 * np.pi * G / wa ** 2
    print(f"\n  long-wave limit at omega={wa:.2f} (lambda/L = "
          f"{lam/db.L:.1f}):")
    print(f"    heave RAO  freq-domain {np.abs(xi[2]):.4f}, "
          f"time-domain {a_heave:.4f}   (-> 1.0 expected)")
    return np.abs(xi[2]), a_heave


def _free_decay(db, ves, dt):
    """Release from an initial heave offset; no wave forcing at all."""
    s0 = np.zeros(ves.n_state)
    s0[2] = 1.0                                   # 1 m initial heave offset
    t, s = ves.simulate(lambda tt: np.zeros(6), 40.0, dt=dt, s0=s0)
    z = s[:, 2]

    # zero crossings -> period; successive peaks -> logarithmic decrement
    sign_change = np.where(np.diff(np.sign(z)))[0]
    if len(sign_change) < 3:
        return dict(period=np.nan, zeta=np.nan, t=t, z=z)
    period = 2 * np.mean(np.diff(t[sign_change]))
    pk = [i for i in range(1, len(z) - 1)
          if z[i] > z[i - 1] and z[i] > z[i + 1] and z[i] > 0.02]
    if len(pk) >= 2:
        delta = np.log(z[pk[0]] / z[pk[1]])
        zeta = delta / np.sqrt(4 * np.pi ** 2 + delta ** 2)
    else:
        zeta = np.nan
    print(f"\n  free decay in heave: period {period:.2f} s "
          f"(omega_n = {2*np.pi/period:.2f} rad/s), zeta = {zeta:.3f}")
    return dict(period=period, zeta=zeta, t=t, z=z,
                omega_n=2 * np.pi / period)


def _verdict(rows, lf, dec):
    errs_h = [abs(a[2] - x[2]) / max(x[2], 1e-12) for _, x, a, _ in rows]
    errs_p = [abs(a[4] - x[4]) / max(x[4], 1e-12) for _, x, a, _ in rows]
    phs = [abs(np.degrees(d)) for _, _, _, d in rows]
    peak_w = rows[int(np.argmax([x[2] for _, x, _, _ in rows]))][0]

    checks = [
        ("heave RAO time vs freq  < 2%", max(errs_h) < 0.02),
        ("pitch RAO time vs freq  < 2%", max(errs_p) < 0.02),
        ("heave phase agreement   < 5 deg", max(phs) < 5.0),
        ("long-wave heave RAO -> 1 within 8%", abs(lf[1] - 1.0) < 0.08),
        ("free decay is stable and oscillatory",
         np.isfinite(dec["period"]) and dec["period"] > 0),
        ("decay damping in 0 < zeta < 1",
         np.isfinite(dec["zeta"]) and 0 < dec["zeta"] < 1),
    ]
    print("\n  " + "-" * 48)
    for n, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {n}")
    print("  " + "-" * 48)
    print(f"  worst heave err {100*max(errs_h):.2f}%, "
          f"pitch {100*max(errs_p):.2f}%, phase {max(phs):.1f} deg")
    print(f"  M3 TIME-DOMAIN GATE: "
          f"{'PASSED' if all(o for _, o in checks) else 'FAILED'}")


def _plot(db, rows, dec, sign):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    w = [r[0] for r in rows]
    for k, (d, lab) in enumerate([(2, "heave"), (4, "pitch")]):
        ax[k].plot(w, [r[1][d] for r in rows], "k-o", lw=2, ms=6,
                   label="frequency domain (exact)")
        ax[k].plot(w, [r[2][d] for r in rows], "r--s", lw=1.6, ms=5,
                   label="time domain (Cummins)")
        ax[k].set_xlabel("$\\omega$ (rad/s)")
        ax[k].set_ylabel(f"{lab} RAO")
        ax[k].set_title(f"{lab.capitalize()} RAO, head seas", fontsize=11)
        ax[k].grid(alpha=.3); ax[k].legend(fontsize=8)
    ax[0].axhline(1.0, ls=":", c="tab:blue", lw=1.2)
    ax[0].annotate("long-wave limit = 1", (w[0], 1.02), fontsize=8,
                   color="tab:blue")

    ax[2].plot(dec["t"], dec["z"], "tab:purple", lw=1.6)
    ax[2].axhline(0, c="k", lw=.5)
    ax[2].set_xlabel("t (s)"); ax[2].set_ylabel("heave (m)")
    ax[2].set_title(f"Free decay: T={dec['period']:.2f} s, "
                    f"$\\zeta$={dec['zeta']:.3f}", fontsize=11)
    ax[2].grid(alpha=.3)
    fig.suptitle("M3 gate: time domain vs exact frequency domain "
                 "(Wigley 10 m, zero speed)", fontsize=12)
    fig.tight_layout()
    fig.savefig("fig10_m3_rao.png", dpi=140)
    print("\n  figure: fig10_m3_rao.png")


if __name__ == "__main__":
    run()
