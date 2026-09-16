#!/usr/bin/env python3
"""
Joint identification of Yv', Nr' -- run the other way round.

WHAT WAS WRONG WITH THE OLD PROCEDURE

Yv' and Nr' were chosen so that the turning circle came out plausible. The
turning circle was then presented as evidence that the model manoeuvres
plausibly. That is circular, and the circularity had teeth: the values absorbed
the rudder inflow angle and the Munk moment, both of which were missing, so
"calibrated" meant "tuned to compensate for two absent terms".

Adding those terms makes it worse, not better, unless the derivatives are
redone -- which is why `DEFECTS.md` D4 said they had to move together.

THE PROCEDURE HERE

Invert it. Set Yv' and Nr' from PUBLISHED ranges of the non-dimensional
derivatives, and then let the turning circle be an OUTPUT -- a prediction a
turning trial can falsify, rather than a target that was fitted.

    Y_v' = Y_v / (0.5 rho L^2 U)      typical published |Y_v'|  0.010 - 0.040
    N_r' = N_r / (0.5 rho L^4 U)      typical published |N_r'|  0.002 - 0.008

Those ranges are for merchant hulls, and this hull is not one -- L/B = 4.0 and
C_B = 0.44 against the L/B 5-8, C_B 0.5-0.85 the regressions were fitted over.
So the ranges are being extrapolated, and that is stated rather than hidden.
Clarke, Gedling and Hine's regression gives these derivatives from L/B, B/T and
C_B and is the right tool for pinning them; it is also fitted on the same
merchant-ship population, so it would extrapolate too.

WHAT COMES OUT

Not a number. A FEASIBLE REGION: the pairs that satisfy every constraint that
can be checked without a trial. A single value is then picked from the middle of
it, and the spread of turning circles across the region is the honest error bar
on every heading result in this project.

Run: python -m studies.manoeuvring_identify
"""
import numpy as np

from hydro import bem
from sim.vessel import NonlinearVessel
from sim.test_vessel import Monochromatic
import sim.test_manoeuvre as TM

YV_RANGE = (0.010, 0.040)
NR_RANGE = (0.002, 0.008)
L_OVER_B, C_B = 4.0, 0.444


def trial(db, yv, nr, rudder_deg=35.0):
    old = NonlinearVessel.__init__

    def patched(self, *a, **kw):
        old(self, *a, **kw)
        self.Yv_prime, self.Nr_prime = yv, nr
    NonlinearVessel.__init__ = patched
    try:
        z = TM.steady_turn(db, rudder_deg)
        z15 = TM.steady_turn(db, 15.0)
        kick, after = TM.course_stability(db)
    finally:
        NonlinearVessel.__init__ = old
    return dict(radius=z["radius"] / z["L"], drift=z["drift"], u=z["u"],
                drift15=z15["drift"], stable=after < 0.5 * kick,
                decay=after / max(kick, 1e-9))


def own_checks(r):
    """Constraints that do not borrow anything from a ship standard."""
    return (r["radius"] >= 1.0                  # sanity floor, mine
            and r["drift15"] < 15.0
            and r["drift"] < 20.0
            and r["stable"])


def imo(r):
    """IMO MSC.137(76): tactical diameter <= 5 L. Written for ships."""
    return 2.2 * r["radius"] <= 5.0


def feasible(r):
    return own_checks(r) and imo(r)


def main():
    db = bem.load("hydro_wigley_10m.npz")
    print(f"\n  hull: L/B {L_OVER_B:.1f}, C_B {C_B:.3f} -- outside the L/B 5-8,")
    print(f"  C_B 0.5-0.85 population the published derivative ranges were")
    print(f"  fitted on. Every row below is an extrapolation.\n")

    yvs = np.linspace(*YV_RANGE, 5)
    nrs = np.linspace(*NR_RANGE, 7)
    print("  35 deg turning radius in ship lengths.")
    print("    x = fails a check of this project's own")
    print("    i = passes those, fails only the borrowed IMO bound\n")
    print(f"  {'':>10}" + "".join(f"{n:>9.4f}" for n in nrs) + "   <- Nr'")
    grid = {}
    for yv in yvs:
        row = f"  {yv:>10.3f}"
        for nr in nrs:
            r = trial(db, yv, nr)
            grid[(yv, nr)] = r
            mark = (" " if feasible(r) else
                    "i" if own_checks(r) else "x")
            row += f"{r['radius']:>8.2f}{mark}" 
        print(row)
    print(f"  ^ Yv'\n")

    own = {k: v for k, v in grid.items() if own_checks(v)}
    good = {k: v for k, v in grid.items() if feasible(v)}
    ro = [v["radius"] for v in own.values()]
    print(f"  passing this project's own checks: {len(own)}/{len(grid)} pairs, "
          f"radius {min(ro):.2f} - {max(ro):.2f} L")
    print(f"  also passing the borrowed IMO bound: {len(good)}/{len(grid)}")
    print(f"  -> the IMO bound removes {len(own)-len(good)} otherwise-acceptable")
    print(f"     pairs, and it is a ship standard applied to a 10 m boat. It is")
    print(f"     doing real work here, so it is worth knowing that.\n")
    if not good:
        print("  NOTHING in the published range satisfies the constraints.")
        print("  That would be a real finding -- it would mean the hull terms")
        print("  and the new rudder/Munk terms are inconsistent -- but it is")
        print("  not what happens here.")
        return
    rad = [v["radius"] for v in good.values()]
    print(f"  {len(good)} of {len(grid)} pairs are feasible.")
    print(f"  turning radius across the feasible region: "
          f"{min(rad):.2f} - {max(rad):.2f} L  "
          f"(a factor of {max(rad)/min(rad):.1f})")

    # centre of the feasible region, by rank rather than by value, so the
    # choice does not depend on how the grid was spaced
    ys = sorted({k[0] for k in good}); ns = sorted({k[1] for k in good})
    yv_c, nr_c = ys[len(ys) // 2], ns[len(ns) // 2]
    while (yv_c, nr_c) not in good:               # centre may fall in a hole
        yv_c, nr_c = min(good, key=lambda k: (k[0] - yv_c) ** 2
                         + ((k[1] - nr_c) * 5) ** 2)
    c = good[(yv_c, nr_c)]
    print(f"\n  centre of the feasible region:  Yv' = {yv_c:.3f}, "
          f"Nr' = {nr_c:.4f}")
    print(f"    35 deg radius {c['radius']:.2f} L, drift {c['drift']:.1f} deg, "
          f"speed {c['u']:.2f} m/s")
    print(f"    yaw rate decays to {c['decay']:.1%} of the kick in 60 s")

    cur = NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L)
    print(f"\n  currently shipped:              Yv' = {cur.Yv_prime:.3f}, "
          f"Nr' = {cur.Nr_prime:.4f}")
    r0 = trial(db, cur.Yv_prime, cur.Nr_prime)
    print(f"    35 deg radius {r0['radius']:.2f} L, drift {r0['drift']:.1f} deg"
          f"   {'feasible' if feasible(r0) else 'NOT feasible'}")

    print(f"\n  What this does NOT do: choose between the feasible pairs. The")
    print(f"  turning radius spans a factor of {max(rad)/min(rad):.1f} across them, and every")
    print(f"  one of them is consistent with the published ranges and with")
    print(f"  every check this project can run. That factor is the uncertainty")
    print(f"  on heading authority, and it can only be closed by a turning")
    print(f"  trial or by PMM measurements on the real hull.")
    return good


if __name__ == "__main__":
    main()
