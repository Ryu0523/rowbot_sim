#!/usr/bin/env python3
"""
M4 GATE (part 1) -- does the nonlinear strip model reduce to the linear one?

The nonlinear Froude-Krylov integral is only trustworthy if, for motions small
enough that linear theory holds, it reproduces the linear hydrostatic
stiffness that the BEM computed independently:

    C33 = rho g A_wp   = rho g (2/3) L B
    C55 = rho g I_wp   = rho g B L^3 / 30
    C35 = 0            by fore-aft symmetry

Those are exact for a Wigley hull, and the BEM has its own numbers for them,
so this is a three-way check: analytic vs BEM vs the strip integral.

It also pins the pitch sign convention. Both signs give a plausible-looking
moment; only one gives a POSITIVE C55 (a restoring moment). Determining it
here means the vessel model never has to assume it.
"""
import numpy as np

from hydro import bem
from .sections import WigleySections, G, RHO

DB = "hydro_wigley_10m.npz"


def analytic_stiffness(L, B, T, rho=RHO, g=G):
    return dict(C33=rho * g * (2.0 / 3.0) * L * B,
                C55=rho * g * B * L ** 3 / 30.0,
                volume=4.0 / 9.0 * L * B * T)


def numeric_stiffness(sec, eps_z=1e-4, eps_p=1e-5, sign_pitch=-1.0):
    """Central-difference the nonlinear force about the equilibrium."""
    zero = np.zeros(sec.n)

    def F(z, p):
        return sec.fk_restoring(zero, z, p, sign_pitch)[:2]

    fz_p, _ = F(+eps_z, 0.0)
    fz_m, _ = F(-eps_z, 0.0)
    _, mp_p = F(0.0, +eps_p)
    _, mp_m = F(0.0, -eps_p)
    _, mz_p = F(+eps_z, 0.0)
    _, mz_m = F(-eps_z, 0.0)
    # force = +buoyancy, restoring stiffness is -d(force)/d(displacement)
    return dict(C33=-(fz_p - fz_m) / (2 * eps_z),
                C55=-(mp_p - mp_m) / (2 * eps_p),
                C35=-(mz_p - mz_m) / (2 * eps_z))


def main(n_stations=(21, 41, 81, 161)):
    L, B, T = 10.0, 2.5, 0.8
    ana = analytic_stiffness(L, B, T)
    db = bem.load(DB)
    bem_C33, bem_C55 = db.C[2, 2], db.C[4, 4]

    print(f"Wigley L={L} B={B} T={T}")
    print(f"  analytic  V={ana['volume']:.4f} m^3  "
          f"C33={ana['C33']:.0f}  C55={ana['C55']:.0f}")
    print(f"  BEM                      "
          f"C33={bem_C33:.0f}  C55={bem_C55:.0f}   "
          f"(err {100*(bem_C33-ana['C33'])/ana['C33']:+.2f}% / "
          f"{100*(bem_C55-ana['C55'])/ana['C55']:+.2f}%)\n")

    # pitch sign: only one choice gives a restoring (positive) C55
    signs = {}
    for sp in (-1.0, +1.0):
        s = numeric_stiffness(WigleySections(L, B, T, 81), sign_pitch=sp)
        signs[sp] = s["C55"]
        print(f"  sign_pitch = {sp:+.0f}: C55 = {s['C55']:+.0f}")
    sign_pitch = max(signs, key=signs.get)
    print(f"  -> restoring requires sign_pitch = {sign_pitch:+.0f}\n")

    print(f"  {'stations':>9}{'V err%':>9}{'C33 err%':>10}{'C55 err%':>10}"
          f"{'C35/C33':>10}")
    res = {}
    for n in n_stations:
        sec = WigleySections(L, B, T, n)
        num = numeric_stiffness(sec, sign_pitch=sign_pitch)
        res[n] = num
        print(f"  {n:>9}{100*(sec.volume()-ana['volume'])/ana['volume']:>9.3f}"
              f"{100*(num['C33']-ana['C33'])/ana['C33']:>10.3f}"
              f"{100*(num['C55']-ana['C55'])/ana['C55']:>10.3f}"
              f"{num['C35']/num['C33']:>10.2e}")

    fine = res[n_stations[-1]]
    sec = WigleySections(L, B, T, n_stations[-1])
    checks = [
        ("volume matches analytic within 0.1%",
         abs(sec.volume() - ana["volume"]) / ana["volume"] < 1e-3),
        ("C33 matches analytic within 0.5%",
         abs(fine["C33"] - ana["C33"]) / ana["C33"] < 5e-3),
        ("C55 matches analytic within 0.5%",
         abs(fine["C55"] - ana["C55"]) / ana["C55"] < 5e-3),
        ("C35 negligible by symmetry",
         abs(fine["C35"] / fine["C33"]) < 1e-6),
        ("C33 agrees with BEM within 1%",
         abs(fine["C33"] - bem_C33) / bem_C33 < 1e-2),
        ("C55 agrees with BEM within 2%",
         abs(fine["C55"] - bem_C55) / bem_C55 < 2e-2),
        ("emerged section gives zero area",
         sec.area(np.full(sec.n, -T - 0.1)).max() == 0.0),
        ("deeper immersion gives more buoyancy",
         sec.area(np.full(sec.n, 0.3)).sum() > sec.area(np.zeros(sec.n)).sum()),
    ]
    print("\n  " + "-" * 48)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    print("  " + "-" * 48)
    ok = all(o for _, o in checks)
    print(f"  M4a NONLINEAR-FK LINEAR-LIMIT GATE: "
          f"{'PASSED' if ok else 'FAILED'}")
    print(f"  -> vessel model must use sign_pitch = {sign_pitch:+.0f}")
    return sign_pitch, ok


if __name__ == "__main__":
    main()
