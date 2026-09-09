#!/usr/bin/env python3
"""
Manoeuvring gate -- can this plant be steered at all?

Every preview study in this project ran with the rudder LOCKED. The stated
reason was that the MPC's reduced model had no sway state, so the controller
could not see the drift it was causing. That was true, but it was not the whole
story: the PLANT was misbehaving too, and locking the channel hid it for the
whole project.

The plant's only lateral resistance was a quadratic viscous term. A hull's
dominant side force at small drift angles is lift-like, proportional to u*v,
and it was absent. With 5 degrees of rudder the hull reached a 30 degree drift
angle and turned inside its own length, and the surge Coriolis coupling m*v*r
then consumed 5.1 kN of a 7 kN thrust.

So before any heading-control study means anything, the plant has to pass the
checks a real vessel is required to pass. These are the standard manoeuvring
trials, and they are what pins the two new coefficients -- which are otherwise
placeholders of the same standing as the viscous damping.

Run: python -m sim.test_manoeuvre
"""
import numpy as np

from hydro import bem
from sim.vessel import NonlinearVessel
from sim.test_vessel import Monochromatic

DB = "hydro_wigley_10m.npz"
DT = 0.05


def steady_turn(db, rudder_deg, thrust=7000.0, t_end=160.0, u0=4.5):
    p = NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L)
    s = p.initial_state(u0)
    t = 0.0
    for _ in range(int(t_end / DT)):
        s = p.step(s, t, thrust, np.radians(rudder_deg), DT)
        t += DT
    u, v, r = float(s[6]), float(s[7]), float(s[11])
    return dict(u=u, v=v, r=r, L=p.L,
                drift=float(np.degrees(np.arctan2(v, max(u, 1e-6)))),
                radius=float(u / abs(r)) if abs(r) > 1e-6 else np.inf)


def course_stability(db, thrust=7000.0, kick_deg=10.0, u0=4.5):
    """Kick the rudder, centre it, and see whether the turn dies away.

    A directionally unstable hull keeps turning with the rudder amidships. The
    vessel does not have to be stable -- many small craft are not, and that is
    what an autopilot is for -- but it must not diverge, or no controller can
    hold a heading."""
    p = NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L)
    s = p.initial_state(u0)
    t = 0.0
    for _ in range(int(20.0 / DT)):
        s = p.step(s, t, thrust, np.radians(kick_deg), DT)
        t += DT
    r_kick = abs(float(s[11]))
    for _ in range(int(60.0 / DT)):
        s = p.step(s, t, thrust, 0.0, DT)
        t += DT
    return r_kick, abs(float(s[11]))


def main():
    db = bem.load(DB)
    p0 = NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L)
    L = p0.L
    print(f"Wigley {L:.0f} m -- manoeuvring trials")
    print(f"  Yv' = {p0.Yv_prime:.4f}   Nr' = {p0.Nr_prime:.4f}\n")

    straight = steady_turn(db, 0.0)
    print(f"  {'rudder':>8}{'u':>8}{'drift':>9}{'radius/L':>11}{'speed loss':>12}")
    turns = {}
    for a in (5, 10, 15, 25, 35):
        r = steady_turn(db, a)
        turns[a] = r
        loss = 100 * (1 - r["u"] / straight["u"])
        print(f"  {a:>7}d{r['u']:>8.2f}{r['drift']:>8.1f}d"
              f"{r['radius']/L:>11.2f}{loss:>11.0f}%")

    r_kick, r_after = course_stability(db)
    print(f"\n  course stability: yaw rate {r_kick:.4f} -> {r_after:.4f} rad/s "
          f"60 s after centring the rudder")

    t35 = turns[35]
    loss35 = 100 * (1 - t35["u"] / straight["u"])
    checks = [
        ("turning radius at 35 deg is 1-4 ship lengths",
         1.0 <= t35["radius"] / L <= 4.0),
        ("drift angle at 15 deg rudder below 15 deg",
         turns[15]["drift"] < 15.0),
        ("speed loss in a hard turn is 10-60%", 10.0 <= loss35 <= 60.0),
        # Only up to stall. Past it a rudder genuinely loses lift and the turn
        # widens again -- that is correct physics, and the first version of this
        # check called it a failure.
        ("turn tightens monotonically with rudder angle, up to stall",
         all(turns[a]["radius"] > turns[b]["radius"]
             for a, b in zip((5, 10, 15), (10, 15, 25)))),
        # ...which exposes a parameter inconsistency worth its own check: the
        # rudder's maximum deflection is past its own stall angle, so the last
        # third of its travel makes the vessel turn WORSE. No one would build
        # that; one of the two numbers is wrong.
        ("maximum rudder deflection does not exceed the stall angle",
         p0.rudder.max <= p0.rudder.stall + 1e-9),
        ("no directional divergence with the rudder centred",
         r_after < 0.5 * r_kick),
    ]
    print("\n  " + "-" * 54)
    for n, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {n}")
    print("  " + "-" * 54)
    good = all(o for _, o in checks)
    print(f"  MANOEUVRING GATE: {'PASSED' if good else 'FAILED'}")
    print("\n  Yv' and Nr' are calibration placeholders like the viscous terms.")
    print("  This gate does not measure them -- it bounds them, which is the")
    print("  most a simulation can do without a turning trial.")
    return good


if __name__ == "__main__":
    main()
