#!/usr/bin/env python3
"""
Are the six viscous damping placeholders anywhere near reality?

There is no towing-tank dataset for a 10 m Wigley USV's roll damping and there
is not going to be one. A placeholder can still be checked, though, by turning
it into a quantity the literature reports: the DAMPING RATIO at resonance,
zeta, the fraction of critical damping. That number is dimensionless and
comparable across hulls -- as long as it is compared at a comparable frequency,
which is where this file's first answer went wrong.

TWO CONVERSIONS

Potential part, from the BEM. Radiation damping B(w) is energy carried away by
the waves the hull itself makes:

    zeta_pot = B(w_n) / (2 w_n (M + A(w_n)))

Viscous part. The model uses quadratic damping F = b |x'| x', which has no
damping ratio of its own -- it depends on amplitude. Harmonic balance gives
b_eq = (8/3pi) b a w, hence

    zeta_visc = (4 / 3 pi) b a / (M + A)

so it must be quoted AT an amplitude, and it is.

THE MISTAKE THIS FILE MADE FIRST, because it is the same one twice

The first version flagged this vessel's roll damping as "above published" against
a 3-5% range for bare hulls. That range is real, and it is for SHIPS, and it does
not transfer -- not because ships are bigger, but because they roll SLOWLY. Roll
radiation damping is strongly frequency dependent, and this project's own BEM
table shows it:

    w = 0.7 rad/s (T = 9 s, ship-like)      B44 =     2 N m s/rad
    w = 3.4 rad/s (T = 1.8 s, this vessel)  B44 =  5820 N m s/rad

A big ship rolls at the far left of that curve, where potential flow really does
give it almost nothing, which is why "roll is undamped in potential flow" became
a rule of thumb. A 10 m craft rolls near the PEAK. The rule of thumb is a
statement about ship roll periods, not about roll.

That matters for this codebase directly: the comment on `visc` in
`sim/vessel.py` says potential flow "leaves roll essentially undamped, so this
is not optional". Measured below, the placeholder supplies about a fifth of
this vessel's roll damping, not all of it.

> An external anchor used outside the regime it was measured in is not an
> external anchor. It is a number with a citation attached.

Run: python -m studies.damping_check
"""
import numpy as np

from hydro import bem
from hydro.inertia import (wigley_inertia, with_gyradius, K_ROLL_REAL,
                           K_PITCH_REAL, K_YAW_REAL)

VISC = np.array([280.0, 4000.0, 3000.0, 2500.0, 9000.0, 6000.0])
NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")
L, B, T = 10.0, 2.5, 0.8


def resonance(db, M, i):
    """Natural frequency solved consistently -- added mass depends on it."""
    C = db.C[i, i]
    w = np.sqrt(C / (M[i, i] + db.A[-1, i, i]))
    for _ in range(60):
        w = np.sqrt(C / (M[i, i] + np.interp(w, db.omega, db.A[:, i, i])))
    A = float(np.interp(w, db.omega, db.A[:, i, i]))
    Bd = float(np.interp(w, db.omega, db.B[:, i, i]))
    return float(w), A, Bd


def main():
    db = bem.load("hydro_wigley_10m.npz")
    M_geo, _ = wigley_inertia(L, B, T, -T / 3)

    # ---- why the ship rule of thumb does not transfer ---------------------
    print("\n  roll radiation damping is a strong function of ROLL PERIOD.")
    print("  Same hull, same BEM, different natural period:\n")
    print(f"  {'T_roll':>8}{'w':>8}{'B44':>10}{'A44':>9}"
          f"{'zeta_pot':>11}   who rolls there")
    for Tr, who in ((12.0, "a large ship"), (6.0, "a coaster"),
                    (3.0, "a large launch"), (2.0, "this vessel, realistic k"),
                    (1.66, "this vessel, bare-hull k")):
        w = 2 * np.pi / Tr
        A = float(np.interp(w, db.omega, db.A[:, 3, 3]))
        Bd = float(np.interp(w, db.omega, db.B[:, 3, 3]))
        # the inertia that WOULD put resonance there, given C44
        Mt = db.C[3, 3] / w ** 2
        print(f"  {Tr:>7.1f}s{w:>8.2f}{Bd:>10.0f}{A:>9.0f}"
              f"{Bd/(2*w*Mt):>10.1%}   {who}")
    print("\n  The 3-5% rule for bare hulls describes the top row. This vessel")
    print("  is four rows down, near the peak of B44. Same physics, different")
    print("  place on the curve.\n")

    # ---- each dof, under both mass-distribution assumptions ---------------
    amps = {"heave": (0.25, 0.5, 1.0), "roll": (5.0, 10.0, 20.0),
            "pitch": (2.0, 5.0, 10.0)}
    unit = {"heave": "m", "roll": "deg", "pitch": "deg"}
    cases = (("bare-hull gyradius", M_geo),
             ("realistic gyradius", with_gyradius(
                 M_geo, L, B, K_ROLL_REAL[0], K_PITCH_REAL[0], K_YAW_REAL[0])))

    spans = {}
    for i, nm in enumerate(NAMES):
        if db.C[i, i] <= 0:
            continue
        print(f"  {nm.upper()}")
        for lab, M in cases:
            w, A, Bd = resonance(db, M, i)
            Mt = M[i, i] + A
            z_pot = Bd / (2 * w * Mt)
            mid = amps[nm][1]
            a_si = np.radians(mid) if unit[nm] == "deg" else mid
            z_v = (4 / (3 * np.pi)) * VISC[i] * a_si / Mt
            print(f"    {lab:<20} T_n {2*np.pi/w:>5.2f} s   "
                  f"zeta_pot {z_pot:>6.1%}   "
                  f"zeta_visc(at {mid:g} {unit[nm]}) {z_v:>6.1%}   "
                  f"placeholder is {z_v/(z_pot+z_v):>5.0%} of the total")
        # amplitude dependence, on the realistic mass distribution
        w, A, Bd = resonance(db, cases[1][1], i)
        Mt = cases[1][1][i, i] + A
        z_pot = Bd / (2 * w * Mt)
        span = []
        for a in amps[nm]:
            a_si = np.radians(a) if unit[nm] == "deg" else a
            span.append(z_pot + (4 / (3 * np.pi)) * VISC[i] * a_si / Mt)
        spans[nm] = span
        print(f"    total zeta over {amps[nm][0]:g}-{amps[nm][-1]:g} "
              f"{unit[nm]}: {span[0]:.1%} .. {span[-1]:.1%}\n")

    print("  What this does and does not settle:")
    print("    - it settles that the roll placeholder is NOT carrying the whole")
    print("      model, which the code comment claims. It is about a fifth.")
    print("    - it does not settle whether B44 itself is right. Radiation was")
    print("      verified against Hulme's sphere and against analytic limits;")
    print("      roll on a wide shallow hull is the least-covered corner of")
    print("      that verification, and no Wigley experiment covers it either")
    print("      -- those are head-seas tests, so they give heave and pitch.")
    print("    - a free-decay test remains the only way to close it, and it")
    print("      now has a specific prediction to falsify: total zeta of")
    print(f"      roughly {spans['roll'][0]:.0%}-{spans['roll'][-1]:.0%} "
          f"at 5-20 degrees of roll, with a")
    print("      natural period near 2 s.")


if __name__ == "__main__":
    main()
