#!/usr/bin/env python3
"""
The published methods, computed for THIS hull -- and only where I can do it
without inventing coefficients.

"Run all the experiments" cannot mean what it sounds like. A towing tank cannot
be run from here. What the published work divides into is two different things,
and the difference decides what is possible:

  A CLOSED-FORM METHOD fitted to experiments. Whicker and Fehlner's lift-slope
    curve, Wagner's water-entry solution, Pierson-Moskowitz. These can be
    evaluated for this hull right now, and that is a real result: it either
    agrees with a placeholder or it does not.

  A DATASET. Journee's Wigley RAOs, SIMMAN's PMM runs, Ikeda's roll-damping
    database, Blendermann's wind coefficients, the Wageningen open-water
    polynomials. These are numbers someone measured. They have to be obtained.
    Reproducing them from memory would produce something that looks like
    evidence and is not, which is the failure this whole line of work has been
    about.

So this file computes the first kind and reports the second kind as pending,
with what each would settle. Two checks are done here in full.

Run: python -m studies.external_methods
"""
import numpy as np

RHO = 1025.0
G = 9.81


# --------------------------------------------------- 1. rudder lift slope
def whicker_fehlner(ar, sweep=0.0):
    """Lift-curve slope of a low-aspect-ratio all-movable control surface.

        dCL/dalpha = 1.8 pi AR / (1.8 + cos(L) sqrt(AR^2/cos^4(L) + 4))

    This is the standard form for rudders, fitted to free-stream measurements.
    A rudder is NOT a wing: at AR below about 2 the slope is a fraction of the
    2D value of 2 pi, and using 2 pi -- or any thin-aerofoil number -- gives a
    rudder several times too powerful.

    Note what the formula asks for: the EFFECTIVE aspect ratio. A rudder whose
    root sits against a hull or a fixed horn sees its own mirror image, so the
    effective aspect ratio is close to twice the geometric one. That factor is
    the difference between 2.1 and 3.3 per radian here.

    Delegates to the one implementation, in `sim/actuators.py`, which the
    rudder now uses to compute its own slope. Two copies of a formula drift.
    """
    from sim.actuators import Rudder
    return Rudder.whicker_fehlner(ar, sweep)


def check_rudder():
    from sim.actuators import Rudder
    r = Rudder()
    chord = r.area / r.span
    ar_geo = r.span ** 2 / r.area
    print("  1. RUDDER LIFT SLOPE -- Whicker & Fehlner, closed form\n")
    print(f"     span {r.span:.2f} m, area {r.area:.3f} m2, "
          f"chord {chord:.3f} m, geometric AR {ar_geo:.2f}")
    print(f"     {'effective AR':>16}{'dCL/dalpha':>14}   case")
    for ar, case in ((ar_geo, "free stream, no end plate"),
                     (1.5 * ar_geo, "partial ground effect at the root"),
                     (2.0 * ar_geo, "root against the hull: full mirror image")):
        print(f"     {ar:>16.2f}{whicker_fehlner(ar):>14.2f}   {case}")
    lo, hi = whicker_fehlner(ar_geo), whicker_fehlner(2 * ar_geo)
    # The value the model USED to hard-code. Compared here rather than the
    # rudder's current cl_alpha, because the rudder now computes its slope
    # from this very formula -- comparing it to itself would pass trivially.
    old = 3.50
    inside = lo <= old <= hi
    print(f"\n     the model hard-coded cl_alpha = {old:.2f} /rad")
    if inside:
        print(f"     -> inside the {lo:.2f}-{hi:.2f} band.\n")
    else:
        side = "ABOVE" if old > hi else "BELOW"
        edge = hi if old > hi else lo
        print(f"     -> {abs(old/edge-1):.0%} {side} the {lo:.2f}-{hi:.2f} "
              f"band -- close, but outside every case")
        print(f"        the formula covers, including the most favourable.")
        print(f"        The first draft of this file printed 'survives' here.")
        print(f"        That conclusion was written before the number was")
        print(f"        computed, and the number disagreed with it.")
        print(f"\n     The rudder now COMPUTES its slope from its own blade:")
        print(f"        {r.cl_alpha:.2f} /rad at effective AR = "
              f"{r.ar_factor:.1f} x geometric.")
        print(f"     Why 3.5 was probably chosen: it stood in for the propeller")
        print(f"     slipstream over the rudder, which the model does not have.")
        print(f"     That raises dynamic pressure, not lift slope, so it is now")
        print(f"     listed as a missing term instead of absorbed into this one.")
        print(f"     The same propeller race is what justifies the 38 deg stall")
        print(f"     angle -- the model cited the race without modelling it.\n")
    print(f"     NOT settled by this: the stall angle. The model raised it to")
    print(f"     38 deg to sit outside the 35 deg stops, justified by the")
    print(f"     propeller race. Whicker & Fehlner also give a stall relation")
    print(f"     and it is a free-stream one, so it would not settle the")
    print(f"     in-race case anyway -- that needs Molland and Turnock.\n")
    return inside


# ------------------------------------------------ 2. slamming, Wagner exact
def check_slam():
    """The pile-up factor against Wagner's exact wedge solution.

    For a 2D wedge of deadrise beta entering at speed V, Wagner's flat-plate
    expansion gives a wetted half-width c = (pi/2) d / tan(beta) at penetration
    d -- larger than the geometric half-width b = d / tan(beta) by exactly
    pi/2, because the water piles up the sides.

    The model does not model the pile-up geometrically. It uses the GEOMETRIC
    half-width and multiplies the resulting force by PILE_UP = (pi/2)^2. Those
    are the same thing, and this is the proof rather than the assertion:

        m_a(Wagner)  = (pi/2) rho c^2 = (pi/2) rho (pi/2)^2 d^2 / tan^2 beta
        d m_a / dd   = pi^3 rho d / (4 tan^2 beta)

        m_a(geometric) = (pi/2) rho b^2 = (pi/2) rho d^2 / tan^2 beta
        d m_a / dd     = pi rho d / tan^2 beta
        times (pi/2)^2 = pi^3 rho d / (4 tan^2 beta)          <- identical

    Verified numerically below over a range of deadrise angles, because an
    algebraic identity that has been transcribed into code is no longer an
    algebraic identity.
    """
    from sim.sections import SlamLoad
    print("  2. SLAMMING PILE-UP -- Wagner, exact\n")
    print(f"     {'deadrise':>10}{'d':>7}{'Wagner dm/dd':>15}"
          f"{'geometric x (pi/2)^2':>22}{'error':>9}")
    worst = 0.0
    for beta_deg in (10.0, 20.0, 30.0, 45.0):
        beta = np.radians(beta_deg)
        for d in (0.05, 0.20):
            wagner = np.pi ** 3 * RHO * d / (4 * np.tan(beta) ** 2)
            b = d / np.tan(beta)
            geo = SlamLoad.PILE_UP * np.pi * RHO * d / np.tan(beta) ** 2
            e = abs(geo - wagner) / wagner
            worst = max(worst, e)
            print(f"     {beta_deg:>9.0f}d{d:>7.2f}{wagner:>15.1f}"
                  f"{geo:>22.1f}{e:>9.2e}")
    print(f"\n     PILE_UP = (pi/2)^2 = {SlamLoad.PILE_UP:.5f}, worst error "
          f"{worst:.1e}\n")
    print(f"     What this does NOT do: validate the model against water-entry")
    print(f"     EXPERIMENTS. Wagner's solution itself over-predicts peak")
    print(f"     pressure for small deadrise. The measured drop tests are in")
    print(f"     Zhao, Faltinsen & Aarsnes (21st Symp. Naval Hydrodynamics):")
    print(f"     a 30 deg wedge and a bow-flare section at MARINTEK, force and")
    print(f"     pressure time histories -- free to read, but in figures only,")
    print(f"     so using them means digitising curves. (Zhao & Faltinsen 1993,")
    print(f"     cited here before, is the numerical method, not the tests.)\n")
    return worst < 1e-9


# ------------------------------------------------------- 3. the rest, honestly
PENDING = [
    ("Wageningen B-series open-water", "dataset (polynomial fit, published)",
     "K_T(J), K_Q(J)",
     "the model has the right FORM (T = rho n^2 D^4 K_T(J), K_T linear in J) "
     "with kt0 = 0.45 and j0 = 0.75 as placeholders. The published polynomial "
     "would replace two numbers, not the structure."),
    ("Whicker & Fehlner stall relation", "closed form, free stream",
     "rudder stall angle",
     "computed above for the lift slope only. The stall angle in a propeller "
     "race is a different problem and needs Molland & Turnock."),
    ("Molland & Turnock rudder-propeller", "dataset (wind tunnel)",
     "flow-straightening gamma, in-race stall",
     "gamma = 0.45 is the MMG typical value and it directly scales the rudder "
     "inflow correction, which moved the turning circle by 42%."),
    ("Blendermann / Isherwood wind", "dataset (wind tunnel)",
     "C_X(g), C_Y(g), C_N(g)",
     "cx = 0.7, cy = 0.9 and a 0.5 m yaw lever are placeholders with the "
     "right order. The real C_N(g) peaks near 40 deg and is not a constant "
     "lever, which is the part the shape would fix."),
    ("Ikeda roll damping", "semi-empirical method + database",
     "eddy and lift components of roll damping",
     "lower priority than it looked: studies/damping_check.py shows the BEM "
     "already supplies ~4/5 of this vessel's roll damping, because a 10 m "
     "hull rolls near the peak of B44 rather than at a ship's period."),
    ("Holtrop-Mennen resistance", "regression on ~300 model tests",
     "calm-water surge resistance",
     "280 N/(m/s)^2 sets the whole speed envelope. Holtrop is fitted on "
     "merchant hulls with L/B 5-8; this hull is L/B 4.0, so it would "
     "extrapolate."),
    ("Journee / ITTC Wigley seakeeping", "dataset (towing tank)",
     "heave and pitch RAOs, several Froude numbers",
     "the single highest-value item: it tests mesh, BEM, radiation memory and "
     "the nonlinear FK correction end to end, against measurement, on exactly "
     "this hull form. Covers 46% of this project's bow-acceleration variance "
     "(studies/rao_relevance.py)."),
    ("SIMMAN 2008/2014/2020", "dataset (free-running + PMM)",
     "Y_v, N_r, Y_r, N_v and rudder forces, separated",
     "the only thing that can separate Yv', Nr' and the rudder inflow angle. "
     "studies/manoeuvring_identify.py leaves a factor of 1.8 of uncertainty "
     "in turning radius that nothing internal can close."),
    ("Gothenburg / Tokyo CFD workshops", "dataset (EFD for 5415, KCS, KVLCC)",
     "6-DOF seakeeping and roll decay",
     "the only public roll-decay data reachable. Different hulls, so it "
     "validates the METHOD rather than this vessel."),
    ("free-decay test on the real hull", "measurement, not yet possible",
     "all six viscous damping placeholders",
     "has a falsifiable prediction waiting for it: total roll zeta 10-13% "
     "over 5-20 deg, natural period near 2 s."),
]


def report():
    print("  3. EVERYTHING ELSE, and why it is not computed here\n")
    print(f"  {'source':<34}{'kind':<38}{'settles'}")
    for name, kind, settles, _ in PENDING:
        print(f"  {name:<34}{kind:<38}{settles}")
    print()
    for name, _, _, note in PENDING:
        print(f"    {name}")
        for line in _wrap(note, 68):
            print(f"        {line}")
    print()


def _wrap(s, n):
    out, cur = [], ""
    for w in s.split():
        if len(cur) + len(w) + 1 > n:
            out.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        out.append(cur)
    return out


def main():
    print("\nPUBLISHED METHODS APPLIED TO THIS HULL\n")
    ok_r = check_rudder()
    ok_s = check_slam()
    report()
    print(f"  computed here and passing: rudder lift slope {ok_r}, "
          f"Wagner pile-up {ok_s}")
    print(f"  pending on data acquisition: {len(PENDING)} items above.")


if __name__ == "__main__":
    main()
