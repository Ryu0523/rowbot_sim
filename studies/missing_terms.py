#!/usr/bin/env python3
"""
What this model leaves out -- sorted by HOW WELL EACH CLAIM IS KNOWN.

The first version of this file ranked missing terms by magnitude and presented
every row the same way. That was wrong in a way worth spelling out, because it
is the standard failure of a simulation study.

Every magnitude in it was computed from the model's own trajectory, and divided
by the model's own resultant. So a term was called "24% of sway" using sway
velocities that the model produces WITHOUT that term, over a denominator that is
also missing wind, drift and rudder inflow. The measurement and the thing being
measured were the same object. It also asserted that a 2.32 L turning radius is
"in the middle of the realistic band" for a 10 m craft -- a band I supplied from
nothing. A fabricated reference used to validate a change is worse than no
reference, because it looks like evidence.

The same failure had already cost something real. `db.M`, the rigid-body mass
matrix under all of this, was low by 17-23% in roll, pitch and yaw. Displaced
volume was verified. Heave stiffness was verified. Roll stiffness was verified.
Twenty-seven physics audits passed. None of them touched the inertia, and it
took an EXTERNAL check -- exact hemisphere, exact box, closed-form Wigley -- to
find it. See `hydro/inertia.py`.

So this file is organised by evidence, not by size:

  TIER A   code facts. The term exists in real physics and is absent from the
           source. Verified by reading the source, and the reading is done by
           this script rather than asserted. Cannot be wrong.

  TIER B   magnitudes anchored OUTSIDE the model: from hull geometry plus
           published relations. Wrong only if the published relation is wrong.

  TIER C   magnitudes computed from the model's own motion. Circular. Reported
           with the span they cover as the placeholder coefficients are swept,
           because that span is the honest error bar and it is usually large.

WHAT NONE OF IT ESTABLISHES

No part of this project has ever been compared with a physical experiment. The
external anchors used so far are exact mathematics (analytic hydrostatics,
Hulme's sphere, the closed forms in `hydro/inertia.py`) and published empirical
relations. Those catch coding errors. They cannot tell you whether the model
describes a real 10 m USV in Sea State 5.

What would:

  Wigley hull seakeeping data. This is among the most-measured hulls in
  existence -- the ITTC cooperative experiments and Journee's Delft series give
  heave and pitch RAOs in regular waves at several Froude numbers, for exactly
  this hull form. Comparing this model's RAOs against them is the single
  highest-value test available, it needs no new equipment, and it has not been
  done.

  Free-decay tests, for the six viscous damping placeholders -- roll above all,
  where potential flow gives almost nothing and the placeholder therefore
  decides the entire resonant response.

  A turning trial, for Yv', Nr' and the rudder inflow angle together. They
  cannot be separated without one: see `consequence()` at the end of this file.

Run: python -m studies.missing_terms
"""
import os
import re

import numpy as np

from hydro import bem
from sim.vessel import NonlinearVessel, SIGN_PITCH
from sim.wavefield import SeaState

G = 9.81
RHO = 1025.0
RHO_AIR = 1.225
FREEBOARD = 0.55         # deck edge above the design waterline: the same
                         # number the viewer and studies/visualise.py draw with
GAMMA_FLOW = 0.45        # hull flow-straightening at the rudder (MMG, typical)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================ TIER A: the code
def source(*parts):
    with open(os.path.join(HERE, *parts), encoding="utf-8") as f:
        return f.read()


def tier_a():
    """Absences established by reading the source, not by running it.

    Each row is a predicate over the actual text of the module, so a row that
    stops firing means the term has been added. Rows that stopped firing are
    kept as CLOSED rather than deleted -- what a model used to be missing is
    part of how much to trust its history.

    One earlier finding here was WRONG, and it is kept too, under its own
    heading. This list said the rigid-body Coriolis vector had only 2 of its 6
    components, and that was acted on: the full body-fixed set was added, and
    it made a calm-water turn roll at 17 deg/s. A list of "missing terms" can
    be wrong in the direction of ADD THIS, and that failure is louder than an
    omission, so it stays visible (DEFECTS E11).

    Predicates are written to survive refactoring: they look for the thing
    itself, not for a particular spelling of the code around it.
    """
    vessel = source("sim", "vessel.py")
    act = source("sim", "actuators.py")
    sea = source("sim", "seakeeping.py")

    open_rows = [
        ("restoring matrix is constant",
         "self.C_full = enforce_symmetry(db.C)" in vessel
         and not re.search(r"C_full\s*\[", vessel),
         "GM cannot vary as a wave passes, so parametric roll is not merely "
         "small here -- it is unrepresentable. Judged out of reach anyway: "
         "w_e/(2 w_roll) is 0.10-0.18 across the mass-distribution range."),
        ("green water: a metric, never a force",
         "wet_rate" in sea and "wet_rate" not in vessel,
         "water on deck is counted and never weighs anything. Bounded at "
         "about 1% of the heave resultant, and its Tier C span says even "
         "that is not well determined."),
        ("second-order wave forces: excitation is linear in amplitude",
         bool(re.search(r"self\.sea\.a \* np\.exp", vessel)),
         "mean and slowly-varying drift forces are absent by construction. "
         "1 N at the spectral peak; 257 N from the short tail."),
        ("sinkage and trim: no speed-dependent draught",
         not re.search(r"sinkage|squat", vessel, re.I),
         "the vessel floats at its static waterline at every Froude number. "
         "0.054 m of squat at Fn 0.46, 7% of the draught."),
        ("forward-speed coupling of heave and pitch",
         "zero-speed hydrodynamic coefficients" in vessel,
         "U-dependent added mass and damping. This is the real physics that "
         "the body-frame Coriolis terms in w, p, q appeared to supply, and it "
         "belongs in speed-dependent hydrodynamic coefficients (strip theory "
         "or a forward-speed Green function), not in a Coriolis term."),
        ("hull cross-derivatives Y_r, and the lift part of N_v",
         "self.Yv_prime" in vessel and "Yr_prime" not in vessel,
         "N_v now arrives as the ideal-fluid Munk moment inside the "
         "added-mass Coriolis term, and it dominates the turn (0.66 L on its "
         "own). The lift-generated, stabilising part of N_v and all of Y_r "
         "are still missing; only PMM data can separate them."),
        ("propeller slipstream over the rudder",
         "def slipstream" not in act,
         "the rudder sees hull inflow only; the accelerated race behind the "
         "propeller is absent. It is what keeps a rudder working at low speed "
         "under thrust, it is the stated justification for the 38 deg stall "
         "angle, and the old hard-coded lift slope of 3.5 was very likely "
         "absorbing it (studies/external_methods.py)."),
    ]

    closed_rows = [
        ("Coriolis from the yaw rotation, rigid body + added mass",
         "coriolis_force(self.rad.A_inf, h)" in vessel, "sim/forces.py"),
        ("  ...bringing in the Munk moment (A22 - A11) u v",
         "coriolis_force(self.rad.A_inf, h)" in vessel, "sim/forces.py"),
        ("wind in surge, sway and yaw",
         bool(re.search(r"self\.wind\.force", vessel)), "sim/forces.py Wind"),
        ("rudder inflow angle", "def inflow_angle" in act,
         "sim/actuators.py"),
        ("rudder ventilation",
         bool(re.search(r"rudder\.ventilation_factor", vessel)),
         "sim/actuators.py"),
        ("propeller advance ratio K_T(J)", "def speed_correction" in act,
         "sim/actuators.py"),
        ("rudder lift slope from the blade geometry",
         "def whicker_fehlner" in act, "sim/actuators.py"),
        ("port/starboard symmetry on M, C, A_inf",
         "enforce_symmetry" in vessel, "hydro/symmetry.py"),
    ]

    retracted = [
        ("'rigid-body Coriolis: 2 of 6 components written'",
         "h[2] = h[3] = h[4] = 0.0" in vessel,
         "the other four are correctly ABSENT. kinematics() integrates heave, "
         "roll and pitch rates directly and rotates only the horizontal "
         "velocities, so the frame rotates in yaw alone and only yaw "
         "generates Coriolis terms. Adding the body-fixed set put 6 kN rms "
         "of spurious force into heave, made a calm-water turn roll at "
         "17 deg/s rms, and broke the MPC's model of bow acceleration. The "
         "Tier C shares it was ranked by (26-42%) were real magnitudes of a "
         "term that does not belong in this model."),
    ]

    print("TIER A -- what the source still does not contain\n")
    for name, holds, why in open_rows:
        print(f"  [{'confirmed' if holds else '  STALE  '}]  {name}")
        print(f"               {why}")
    print("\n  CLOSED since this file was first written:\n")
    for name, holds, where in closed_rows:
        print(f"  [{'  added  ' if holds else ' MISSING '}]  {name:<58}{where}")
    print("\n  RETRACTED -- listed here as missing, and wrongly:\n")
    for name, holds, why in retracted:
        print(f"  [{'retracted' if holds else ' BACK IN '}]  {name}")
        for line in _wrap(why, 66):
            print(f"               {line}")
    stale = [n for n, h, _ in open_rows if not h]
    gone = [n for n, h, _ in closed_rows if not h]
    back = [n for n, h, _ in retracted if not h]
    if stale:
        print(f"\n  Stopped firing: {stale}. Either the term was added -- move "
              f"the row to CLOSED -- or the predicate has rotted.")
    if gone:
        print(f"\n  REGRESSION: {gone} were added and are no longer detected.")
    if back:
        print(f"\n  REGRESSION: {back} -- a retracted term is back in the code.")
    print()
    return open_rows, closed_rows, retracted


def _wrap(s, n):
    out, cur = [], ""
    for w in s.split():
        if len(cur) + len(w) + 1 > n:
            out.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        out.append(cur)
    return out


# =================================================== TIER B: external anchors
def tier_b(L=10.0, B=2.5, T=0.8, u=4.5, thrust=7000.0, visc_surge=280.0):
    """Magnitudes from geometry plus published relations. No model trajectory.

    Each row carries its source. The point of the column is that you can check
    it without running anything in this repository.
    """
    rows = []

    # Pierson-Moskowitz relates fully developed Hs to wind speed: Hs = 0.0248 U^2
    U = np.sqrt(3.25 / 0.0248)
    # Blendermann / ITTC wind coefficients for a low-freeboard hull: C_Y ~ 0.9,
    # C_X ~ 0.7, and the yaw lever is a few percent of L for a symmetric hull.
    a_lat, a_front = L * FREEBOARD, B * FREEBOARD
    y_wind = 0.5 * RHO_AIR * a_lat * 0.9 * U ** 2
    rows.append(("wind side force, beam on", f"{y_wind:.0f} N",
                 f"Pierson-Moskowitz U = {U:.1f} m/s on {a_lat:.1f} m2, C_Y 0.9"))
    rows.append(("wind surge force, head on",
                 f"{0.5 * RHO_AIR * a_front * 0.7 * U ** 2:.0f} N",
                 f"{100*0.5*RHO_AIR*a_front*0.7*U**2/thrust:.1f}% of a 7 kN "
                 f"thrust -- small, and this one needs no model to say so"))
    rows.append(("wind yaw moment", f"{y_wind * 0.08 * L:.0f} N m",
                 "0.08 L lever, symmetric hull; needs standing helm to trim"))

    # Wageningen B-series: near J = 0.6, K_T ~ 0.25 and dK_T/dJ ~ -0.35.
    J, KT, dKT = 0.6, 0.25, -0.35
    nD = u / J
    dTdu = abs(thrust * (dKT / KT) / nD)
    d_visc = 2 * visc_surge * u
    rows.append(("propeller K_T(J) surge damping", f"{dTdu:.0f} N per m/s",
                 f"Wageningen B-series slope; the model has {d_visc:.0f}, so "
                 f"this is +{dTdu/d_visc:.0%}, and it acts with NO lag"))
    rows.append(("  -> effect on the surge time constant",
                 f"tau_u {6.30:.2f} s -> {6.30/(1+dTdu/d_visc):.2f} s",
                 f"changes 'thrust is 3.4x slower than T_e/4' to "
                 f"{6.30/(1+dTdu/d_visc)/1.87:.1f}x -- same conclusion, "
                 f"smaller margin"))

    # Mean drift force scales as (ka)^3 for a body small against the wave.
    for lam, tag in ((147.0, "spectral peak"), (25.0, "short tail")):
        ka = (2 * np.pi / lam) * (B / 2)
        rows.append((f"mean wave drift, {tag} (lambda {lam:.0f} m)",
                     f"{0.5 * RHO * G * (3.25/4) ** 2 * B * min(ka**3, 1.0):.0f} N",
                     f"(ka)^3 = {ka**3:.4f}; "
                     + ("the hull is nearly transparent at the peak"
                        if lam > 100 else
                        "the tail reflects, but carries little of the energy")))

    # Gyradius: the mass distribution assumption, against published ranges.
    rows.append(("mass distribution: uniform-density hull",
                 "k_roll .213B, k_pitch .225L",
                 "published small-craft values are 0.33-0.40 B and 0.24-0.26 L; "
                 "see hydro/inertia.py sensitivity()"))

    print("TIER B -- magnitudes anchored outside the model\n")
    print(f"  {'term':<44}{'magnitude':>24}   source / consequence")
    for n, val, src in rows:
        print(f"  {n:<44}{val:>24}   {src}")
    print()
    return rows


# ================================================= TIER C: model-conditioned
def collect(seed=0, t_end=180.0, dt=0.05, thrust=7000.0, rudder_deg=0.0,
            n_dir=5, visc_scale=1.0, hull_scale=1.0):
    db = bem.load("hydro_wigley_10m.npz")
    sea = SeaState(3.25, 9.7, n_freq=24, n_dir=n_dir, seed=seed)
    v = NonlinearVessel(db, sea, L=db.L, B=2.5, T=0.8, dt=dt)
    v.visc = v.visc * visc_scale
    v.Yv_prime *= hull_scale
    v.Nr_prime *= hull_scale
    s = v.initial_state(4.0)
    t = 0.0
    H, DS, D = [], [], []
    for _ in range(int(t_end / dt)):
        s = v.step(s, t, thrust, np.radians(rudder_deg), dt)
        t += dt
        ds, d = v.deriv(s, t)
        H.append(s.copy()); DS.append(ds[6:12]); D.append(d)
    return v, db, np.array(H), np.array(DS), np.array(D)


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def shares(visc_scale=1.0, hull_scale=1.0):
    """Every term of interest, as a fraction of its own dof's resultant.

    Split into terms the model still does not have and terms it now does. The
    second group is not padding: those are the sizes that justified adding
    them, and keeping them measured is how a regression would show up.

    The body-frame Coriolis terms in w, p and q that this function used to
    rank here (26-42% of their resultants) are gone from both groups. They are
    not "missing": this model's frame rotates in yaw only, and they do not
    belong in it (Tier A, RETRACTED; DEFECTS E11).
    """
    v, db, H, NUD, D = collect(visc_scale=visc_scale, hull_scale=hull_scale)
    eta, nu = H[:, :6], H[:, 6:12]
    u, sway, w, p, q, r = (nu[:, 0], nu[:, 1], nu[:, 2],
                           nu[:, 3], nu[:, 4], nu[:, 5])
    A = v.rad.A_inf
    L, B, T = v.L, v.B, v.T
    res = NUD @ v.Mtot.T
    R = dict(zip(("surge", "sway", "heave", "roll", "pitch", "yaw"),
                 (rms(res[:, i]) for i in range(6))))

    over = np.maximum(D[:, -1] - FREEBOARD, 0.0)
    gw = RHO * G * over * (0.15 * L * B * 0.6)
    hb = v.sec.y0 * (1.0 - np.clip(D, -T, 0.0) ** 2 / T ** 2)
    It = (2.0 / 3.0) * np.sum(hb ** 3, axis=1) * v.sec.dx
    It0 = (2.0 / 3.0) * np.sum(v.sec.y0 ** 3) * v.sec.dx

    missing = {
        "varying roll restoring (GM)": (float(np.std(It) / It0)
                                        * db.C[3, 3] * rms(eta[:, 3]), "roll"),
        "green water on the foredeck": (rms(gw), "heave"),
        "green water pitch moment": (rms(gw) * 0.42 * L, "pitch"),
    }
    present = {
        "added-mass Coriolis, Munk (A22-A11)uv": (
            rms((A[1, 1] - A[0, 0]) * u * sway), "yaw"),
        "added-mass Coriolis, A11 u r": (rms(A[0, 0] * u * r), "sway"),
        "wind, side force": (
            rms(np.array([v.wind.force(nu[k], eta[k, 5])[1]
                          for k in range(0, len(H), 20)])), "sway"),
    }
    out = {}
    for tag, group in (("missing", missing), ("present", present)):
        for k, (mag, dof) in group.items():
            out[k] = (mag / max(R[dof], 1e-12), dof, tag)
    return out


def tier_c():
    print("TIER C -- magnitudes computed from the model's own motion\n")
    print("  Circular by construction: a term is divided by a resultant the")
    print("  same model produced. The span is the placeholder sweep -- viscous")
    print("  damping x0.5 and x2, hull derivatives x0.5 and x2. Where the span")
    print("  is wide, the ranking is not information.\n")

    base = shares()
    runs = [shares(visc_scale=s) for s in (0.5, 2.0)]
    runs += [shares(hull_scale=s) for s in (0.5, 2.0)]

    for tag, title in (("missing", "STILL MISSING"),
                       ("present", "NOW IN THE MODEL (the sizes that "
                                   "justified adding them)")):
        keys = [k for k in base if base[k][2] == tag]
        print(f"  {title}")
        print(f"  {'term':<40}{'dof':>7}{'base':>8}"
              f"{'span over placeholders':>26}")
        for k in sorted(keys, key=lambda z: -base[z][0]):
            b, dof, _ = base[k]
            vals = [b] + [rn[k][0] for rn in runs]
            lo, hi = min(vals), max(vals)
            flag = "  <- unusable" if hi > 2.5 * max(lo, 1e-9) else ""
            print(f"  {k:<40}{dof:>7}{b:>7.0%}"
                  f"{f'{lo:.0%} .. {hi:.0%}':>26}{flag}")
        print()
    print("  Not in either list: the body-frame Coriolis terms in w, p, q this")
    print("  tier used to rank at 26-42%. They were real magnitudes of terms")
    print("  that do not belong in a yaw-rotating frame. See Tier A, RETRACTED.")
    print()
    return base


# ================================================================ consequence
def consequence():
    """What the steering channel looks like now that the terms are in.

    Heading is worth roughly 15x what thrust is worth on this vessel and every
    heading result in the project runs through it, so this is the one place
    the new physics has to be stated honestly rather than just listed.
    """
    import sim.test_manoeuvre as TM

    db = bem.load("hydro_wigley_10m.npz")
    print("CONSEQUENCE -- the steering channel as it now stands\n")
    print(f"  {'helm':>6}{'radius/L':>11}{'drift':>8}{'u':>8}")
    for d in (10.0, 25.0, 35.0):
        z = TM.steady_turn(db, d)
        print(f"  {d:>5.0f}d{z['radius']/z['L']:>11.2f}"
              f"{z['drift']:>7.1f}d{z['u']:>8.2f}")

    print("\n  How it got here: 35 degrees of helm, one term at a time, in the")
    print("  model's actual (yaw-rotating) frame:\n")
    for line in (
            "as it was                          1.63 L",
            "+ rudder inflow angle              2.32 L   the rudder loses "
            "about half its angle of attack in a steady turn",
            "+ resultant inflow speed           1.98 L   drift adds to the "
            "dynamic pressure as well as subtracting from the angle",
            "+ added-mass Coriolis, yaw frame   0.66 L   the Munk moment: "
            "destabilising, and by far the largest single effect",
            "+ K_T(J)                           0.66 L",
            "+ lift slope, Whicker-Fehlner      0.68 L   3.33 /rad from "
            "the blade, was a hard-coded 3.5",
            "+ Yv', Nr' re-identified           1.79 L   Nr' 0.0030 -> "
            "0.0060, inside the published range"):
        print(f"    {line}")

    print("\n  What that table actually says. The final turning circle is")
    print("  mostly a BALANCE: a potential-flow Munk moment that on its own")
    print("  would turn the vessel inside its own length, against a yaw")
    print("  damping derivative that had to double to hold it off. Both are")
    print("  uncertain. The Munk moment doubles again if the zero-frequency")
    print("  added mass is used instead of A_inf, and the lift-generated part")
    print("  of N_v that would oppose it is not modelled at all. N_r' is a")
    print("  placeholder chosen from a published range.")
    print("\n  So: 1.79 L is where the middle of the feasible region lands; the")
    print("  region spans 1.09-2.19 L, a factor of 2.0, every point consistent")
    print("  with published derivative ranges and with every check this project")
    print("  can run. That is the uncertainty on heading authority. A turning")
    print("  trial or PMM data closes it, and nothing internal can.")
    print("\n  The earlier claim that heading results were a clean OVER-estimate")
    print("  (inflow angle alone) is withdrawn: the Munk moment pushes the")
    print("  other way and harder. Direction and size are both open now.")


def main():
    tier_a()
    tier_b()
    tier_c()
    consequence()


if __name__ == "__main__":
    main()
