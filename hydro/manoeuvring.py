#!/usr/bin/env python3
"""
Linear manoeuvring derivatives of a new hull from its main dimensions.

For a vessel with no captive-model test and no turning trial, the standard
first estimate is the regression of Clarke, Gedling and Hine (1983) over
rotating-arm and PMM tests of merchant hulls:

    Y'_v = -pi (T/L)^2 (1 + 0.40 C_B B/T)
    Y'_r = -pi (T/L)^2 (-1/2 + 2.2 B/L - 0.080 B/T)
    N'_v = -pi (T/L)^2 (1/2 + 2.4 T/L)
    N'_r = -pi (T/L)^2 (1/4 + 0.039 B/T - 0.56 B/L)

non-dimensionalised by 1/2 rho L^2 U (force per sway velocity) and by L^3,
L^4 for the rest. Transcribed from Fossen's MSS (CRAFT/SHIP/models/clarke83.m,
copy in data/external/mss), and verify() evaluates that file's own lines to
check the transcription rather than trusting it.

HOW NonlinearVessel USES THEM. Its hull force is Y = -1/2 rho L^2 |u| v Y'_v
and N = -1/2 rho L^4 |u| r N'_r, in magnitudes. Y'_r and N'_v are NOT used:
the plant carries the ideal-fluid part of the sway-yaw coupling separately, as
the Munk moment inside the added-mass Coriolis term (DEFECTS E11), and Clarke's
N'_v is a measured total with Munk inside it. Adding it would count Munk
twice. What that leaves is the imbalance E11 describes: the full potential
Munk moment against a hull whose viscous stabilising lift is only in N'_r.

VALIDITY. Merchant hulls, a first estimate. A small planing or semi-planing
craft is outside what the regression was fitted to. Replace with the vessel's
own values (Hull.Yv_prime, Hull.Nr_prime) as soon as a turning trial or a PMM
test exists.

Run: python -m hydro.manoeuvring
"""
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSS_CLARKE = os.path.join(HERE, "data", "external", "mss", "clarke83.m")


def clarke_1983(L, B, T, C_B):
    """Non-dimensional linear derivatives, signed in the usual convention."""
    S = np.pi * (T / L) ** 2
    return dict(
        Yv=-S * (1.0 + 0.40 * C_B * B / T),
        Yr=-S * (-0.5 + 2.2 * B / L - 0.080 * B / T),
        Nv=-S * (0.5 + 2.4 * T / L),
        Nr=-S * (0.25 + 0.039 * B / T - 0.56 * B / L),
        Yvdot=-S * (1.0 + 0.16 * C_B * B / T - 5.1 * (B / L) ** 2),
        Yrdot=-S * (0.67 * B / L - 0.0033 * (B / T) ** 2),
        Nvdot=-S * (1.1 * B / L - 0.041 * B / T),
        Nrdot=-S * (1.0 / 12 + 0.017 * C_B * B / T - 0.33 * B / L))


def verify(verbose=True):
    """Evaluate clarke83.m's own formula lines and compare."""
    say = print if verbose else (lambda *a, **k: None)
    if not os.path.exists(MSS_CLARKE):
        say("  clarke83.m not found; skipped")
        return True
    src = open(MSS_CLARKE, encoding="utf-8", errors="replace").read()
    names = ("Yvdot", "Yrdot", "Nvdot", "Nrdot", "Yv", "Yr", "Nv", "Nr")
    ok = True
    for L, B, T, Cb in ((10.0, 2.5, 0.8, 4 / 9), (320.0, 58.0, 20.8, 0.81),
                        (230.0, 32.2, 10.8, 0.65)):
        env = dict(L=L, B=B, T=T, Cb=Cb, pi=np.pi)
        env["S"] = np.pi * (T / L) ** 2
        mine = clarke_1983(L, B, T, Cb)
        for nm in names:
            m = re.search(rf"^{nm}\s*=\s*(.+?);", src, flags=re.M)
            if not m:
                say(f"  {nm}: line not found in clarke83.m")
                ok = False
                continue
            val = eval(m.group(1).replace("^", "**"), {"__builtins__": {}}, env)
            good = abs(val - mine[nm]) <= 1e-12 * max(1.0, abs(val))
            ok &= good
            if not good:
                say(f"  {nm} for L/B {L/B:.2f}: clarke83.m {val:.6g}, here "
                    f"{mine[nm]:.6g}  MISMATCH")
    say(f"  Clarke (1983) against MSS clarke83.m, 8 derivatives x 3 hulls: "
        f"{'identical' if ok else 'MISMATCH'}")
    return ok


def main():
    print("\nCLARKE et al. (1983) linear derivatives\n")
    ok = verify()
    print(f"\n  {'hull':<22}{'Y_v':>9}{'N_r':>9}{'(plant uses the magnitudes)':>30}")
    for lab, L, B, T, Cb in (("Wigley USV 10 m", 10.0, 2.5, 0.8, 4 / 9),
                             ("KVLCC2", 320.0, 58.0, 20.8, 0.8098),
                             ("KCS", 230.0, 32.2, 10.8, 0.6505)):
        d = clarke_1983(L, B, T, Cb)
        print(f"  {lab:<22}{d['Yv']:>9.4f}{d['Nr']:>9.5f}")
    print("\n  For comparison, the Wigley USV's own identified values are "
          "0.025 and 0.0060\n  (studies/manoeuvring_identify.py, published "
          "ranges intersected with checks).")
    return ok


if __name__ == "__main__":
    main()
