#!/usr/bin/env python3
"""
Design-phase checks for the any-speed plant (read-only; nothing tracked is
changed). Reuses the rebuilt captive equations of studies/anyspeed_audit.py.

  1. Hydrostatic identity at rest: the plant's bottom term -rho g sum A tau dx,
     the taper term rho g sum dQ/dx|_d dx (absent in the plant), and the
     wetted-transom face force rho g Q_T must sum to zero on a closed hull.
  2. Displacement ("sunk") branch with the complete D/Dt(m_a V) (the audit's
     dmx term), continued from 1 kn upward: does it stay stable past the
     speed where the plant jumps to the planing branch (bistability)?
  3. Sensitivity of the equilibrium at 10-16 kn to the buoyancy factor a_b
     (1.0 = closed-body hydrostatics, 0.5 = Zarnick planing value), with the
     dmx term on: how load-bearing the a_b closure is in the transition.

Output: studies/_cache/anyspeed/design_check.txt
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "studies"))

import anyspeed_audit as AU  # noqa: E402

OUT = os.path.join(HERE, "studies", "_cache", "anyspeed", "design_check.txt")
KN, G = AU.KN, AU.G
L = []


def out(s=""):
    print(s)
    L.append(s)


def part1(p):
    SP = AU.p_SP(p)
    rg = p.rho * G
    T = AU.terms(p, 0.0, 0.0, 0.0, 0.0, 0.0)
    d, A, tau = T["d"], T["A"], T["tau"]
    x, dx = p.x, p.dx
    bottom = -(rg * A * tau).sum() * dx
    # dQ/dx at fixed d: finite difference of Q(d, yc(x)) in yc only
    Q = np.array([AU.q_first_moment(p, di, yi) for di, yi in zip(d, p.yc)])
    h = 1e-6
    Qp = np.array([AU.q_first_moment(p, di, yi + h) for di, yi in zip(d, p.yc)])
    dQdyc = (Qp - Q) / h
    ycp = np.gradient(p.yc, x)
    taper = (rg * dQdyc * ycp).sum() * dx
    # transom face: keel depth extrapolated to the transom edge
    dT = d[0] + (d[0] - d[1]) * 0.5
    QT = AU.q_first_moment(p, dT, p.yc[0])
    face = rg * QT
    out("1  HYDROSTATIC IDENTITY AT REST (closed hull: sum must be 0)")
    out(f"   bottom (plant Xp at rest)  -rho g sum A tau dx   {bottom:+8.1f} N")
    out(f"   taper  (absent)            +rho g sum dQ/dx|_d   {taper:+8.1f} N")
    out(f"   transom face (absent)      +rho g Q_T (d_T {dT:.3f} m) "
        f"{face:+8.1f} N")
    out(f"   sum                                              "
        f"{bottom + taper + face:+8.1f} N   (W = {p.m * G:.0f} N)")
    out(f"   ratio of residual to W: {(bottom + taper + face) / (p.m * G):+.2e}"
        "  (strip discretisation of the taper; exact in the limit)")
    out()


def lin_modes(p, u, sol, extra):
    def f4(v):
        wd, qd, *_ = AU.captive_rhs(p, v[0], v[1], v[2], v[3], u, extra=extra)
        return [v[2], v[3], wd, qd]
    J = AU.jac(f4, [sol[0], sol[1], 0.0, 0.0], [1e-6, 1e-7, 1e-6, 1e-7])
    md, lam = AU.modes(J)
    return md, float(np.max(lam.real))


def part2(p):
    SP = AU.p_SP(p)
    out("2  DISPLACEMENT BRANCH WITH THE COMPLETE D/Dt(m_a V), continued "
        "upward (a_b blend of the plant kept)")
    out("   kn   z_eq     trim   resid   max Re(lambda)  modes T_n/zeta"
        "              plant branch at same speed (running_attitude trim)")
    guess = (0.0, 0.0)
    for kn in (1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 25):
        u = kn * KN
        sol, r = AU.equilibrium(p, u, guess, extra=("dmx",), only=True)
        if r > 1e-6:
            out(f"   {kn:3d}  branch lost (resid {r:.1e})")
            break
        guess = tuple(sol)
        md, mre = lin_modes(p, u, sol, ("dmx",))
        snap = p.running_attitude(u)
        ms = "  ".join(f"{m[2]:5.2f}/{m[1]:+.3f}" for m in md[:2])
        out(f"   {kn:3d} {sol[0]:+.4f}  {np.degrees(SP * (p.Th_rest + sol[1])):6.2f}"
            f"  {r:.0e}   {mre:+8.3f}      {ms:30s}"
            f"  {np.degrees(SP * (p.Th_rest + snap[1])):5.2f}")
    out()


def part3(p):
    SP = AU.p_SP(p)
    out("3  a_b SENSITIVITY IN THE TRANSITION, complete D/Dt on, planing-"
        "branch equilibrium (start from running_attitude)")
    out("   kn   a_b    z_eq     trim   buoy/W   max Re(lambda)")
    ab0 = p.hull.a_b
    orig = p._a_b
    for kn in (10, 12, 14, 16, 20):
        u = kn * KN
        snap = p.running_attitude(u)
        for ab in (1.0, 0.75, 0.5):
            p._a_b = (lambda uu, ab=ab: ab)
            sol, r = AU.equilibrium(p, u, snap, extra=("dmx",))
            md, mre = lin_modes(p, u, sol, ("dmx",))
            *_, T = AU.captive_rhs(p, sol[0], sol[1], 0, 0, u,
                                   extra=("dmx",))
            bs = T["buoy"].sum() * p.dx / (p.m * G)
            out(f"   {kn:3d}  {ab:4.2f}  {sol[0]:+.4f}  "
                f"{np.degrees(SP * (p.Th_rest + sol[1])):6.2f}   {bs:5.2f}"
                f"   {mre:+8.3f}   (resid {r:.0e})")
        p._a_b = orig
    out()


def part4(p):
    """Does the sunk branch the audit found at 10-25 kn (z < 0, trim
    2.4-3.0 deg, unstable) survive the complete D/Dt? Newton from the audit's
    own sunk equilibria, with and without the term."""
    from scipy.optimize import fsolve
    SP = AU.p_SP(p)
    out("4  SUNK BRANCH (audit section C, 10-25 kn) WITH AND WITHOUT THE "
        "COMPLETE D/Dt, Newton from the audit's sunk equilibria")
    sunk = {10: (-0.0700, 2.39), 12: (-0.1099, 2.76), 14: (-0.1331, 3.02),
            16: (-0.1290, 3.00), 20: (-0.1192, 2.98), 25: (-0.1045, 2.94)}
    for kn, (z0, tr) in sunk.items():
        u = kn * KN
        th0 = np.radians(tr) / SP - p.Th_rest
        for extra in ((), ("dmx",)):
            def res(v, extra=extra):
                wd, qd, *_ = AU.captive_rhs(p, v[0], v[1], 0.0, 0.0, u,
                                            extra=extra)
                return [wd, qd]
            sol, _, _, _ = fsolve(res, [z0, th0], full_output=True,
                                  xtol=1e-12)
            r = np.abs(res(sol)).max()
            _, mre = lin_modes(p, u, sol, extra)
            out(f"   {kn:3d} kn  {'with term' if extra else 'v0 terms ':9s}"
                f"  z {sol[0]:+.4f}  trim "
                f"{np.degrees(SP * (p.Th_rest + sol[1])):6.2f}  resid {r:.0e}"
                f"  max Re(lambda) {mre:+.3f}")
    out("   -> with the term the low-trim sunk branch is gone: Newton lands on"
        " the planing branch (10 kn) or on bow-down buried states (unstable)")
    out()


def main():
    h, p = AU.build(captive_u=None)
    if "--sunk" in sys.argv:
        part4(p)
        path = OUT.replace(".txt", "_sunk.txt")
    else:
        part1(p)
        part2(p)
        part3(p)
        part4(p)
        path = OUT
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
