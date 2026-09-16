#!/usr/bin/env python3
"""
Rigid-body inertia, computed correctly, and checked against exact mathematics.

WHY THIS FILE EXISTS

`db.M` came from Capytaine's `compute_hydrostatics()["inertia_matrix"]`, and it
is wrong for this hull: roll -22.6%, pitch -16.7%, yaw -17.1%. Nothing in the
project caught it. Displaced volume was verified against the analytic Wigley
formula (+0.07%). Heave stiffness was verified (0.1%). Roll stiffness C44 is
right to +0.19%. Twenty-seven physics audits passed. The mass matrix underneath
all of it was still off by a fifth.

THE CAUSE, because it generalises

A volume moment can be written as a surface integral in three equivalent ways:

    int x^2 dV  =  oint n_x (x^3/3) dS  =  oint n_y (y x^2) dS  =  oint n_z (z x^2) dS

Capytaine evaluates all three on the panel mesh and AVERAGES them. On a sphere
that is a good idea -- all three are well conditioned, and averaging cancels
discretisation noise. On this hull it is not. A Wigley hull is nearly
wall-sided, so n_z is almost zero over the whole side; the entire n_z estimate
is carried by the thin sliver of near-horizontal panels around a knife-edge
keel where the beam has already gone to zero. Measured on the actual mesh:

    int x^2 dV     n_x form 44.431   n_y form 44.440   n_z form 22.216
    int y^2 dV     n_x form  2.175   n_y form  2.175   n_z form  0.589

Two forms agree to 0.02%. The third is off by a factor of two. Averaging the
three buries a 100% error as a 17% one -- large enough to matter, small enough
to look like discretisation, and invisible unless you look at the terms
separately.

> Averaging estimates that OUGHT to be identical destroys the only evidence you
> had that one of them is broken. Compare them, then average.

This module compares them, and raises when they disagree.

WHAT IS TRUTH HERE

Not another simulation. Three exact results, none of which involve this
codebase:

  solid hemisphere   I = (2/5) m R^2 about the centre of the sphere, all axes
  rectangular box    I_xx = m (b^2 + c^2) / 12
  Wigley hull        closed form, derived in `wigley_inertia` below

`verify()` checks the mesh routine against all three. That is what makes this
file worth more than the code it replaces.

AND ONE MORE THING THIS DOES NOT FIX

Getting the geometry right still leaves an ASSUMPTION: uniform density over the
immersed hull. A real 10 m USV is not a solid block of water. Its engine, fuel,
batteries, structure and payload sit where the designer put them, and published
gyradii for small craft are k_roll = 0.33-0.40 B and k_pitch = 0.24-0.26 L,
against the 0.213 B and 0.224 L this hull's own geometry gives. So the correct
geometric answer is still 2.4x low in roll inertia for a real vessel.

That is a MODELLING choice, not a bug, and it is exposed as `k_roll`/`k_pitch`/
`k_yaw` rather than buried. The default stays geometric -- swapping in a
"realistic" gyradius would replace a measurable error with an unmeasured guess
-- but `sensitivity()` reports what the published range does to every natural
period, so the assumption cannot be used without seeing its cost.

Run: python -m hydro.inertia
"""
import numpy as np

RHO = 1025.0
G = 9.81

# Published gyradius ranges for small craft, as fractions of B and L. These are
# the external anchors: they come from the naval-architecture literature, not
# from anything in this repository.
K_ROLL_REAL = (0.33, 0.40)      # x B
K_PITCH_REAL = (0.24, 0.26)     # x L
K_YAW_REAL = (0.25, 0.27)       # x L


# --------------------------------------------------------------- mesh version
def volume_moments(centers, normals, areas, rc=(0.0, 0.0, 0.0), rtol=0.02):
    """Volume moments of a closed (or waterline-open) surface, about `rc`.

    Returns dict with V, x, y, z, xx, yy, zz, xy, yz, zx -- each the integral of
    that monomial over the enclosed volume, in coordinates relative to `rc`.

    Only the n_x and n_y divergence forms are used, because on a wall-sided
    hull the n_z form is carried entirely by a degenerate strip of panels at the
    keel (see the module docstring). The two are cross-checked against each
    other and a disagreement beyond `rtol` raises rather than being averaged
    away: two well-conditioned estimates that disagree mean the MESH is bad, and
    that is a different problem which must not be silently smoothed.

    A hull mesh open at z = 0 is fine. Every integrand below carries a factor of
    x or y against n_x or n_y, and a horizontal lid has n_x = n_y = 0, so the
    missing lid contributes nothing to any of them.
    """
    c = np.asarray(centers, float) - np.asarray(rc, float)
    n = np.asarray(normals, float)
    a = np.asarray(areas, float)
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]

    def sx(f):                      # oint n_x f dS, with d f/dx = integrand
        return float(np.sum(nx * f * a))

    def sy(f):
        return float(np.sum(ny * f * a))

    def agree(name, u, v, scale):
        if abs(u - v) > rtol * max(abs(scale), 1e-30):
            raise ValueError(
                f"volume_moments: the n_x and n_y forms of {name} disagree by "
                f"{abs(u-v)/max(abs(scale),1e-30):.1%} ({u:.6g} vs {v:.6g}). "
                f"Both are well conditioned on a normal hull, so this is a mesh "
                f"problem -- do not average it away.")
        return 0.5 * (u + v)

    V = agree("V", sx(x), sy(y), sx(x))
    out = dict(V=V)
    # Each moment is judged against a scale of ITS OWN dimension. This used to
    # compare every moment's discrepancy with the volume V: dimensionally wrong,
    # invisible on the 10 m Wigley (where L^2 ~ 100), and on a 320 m tanker it
    # reported two estimates of int x^2 dV that agree to 1e-5 as "42% apart".
    S2 = max(abs(sx(x ** 3 / 3)), abs(sy(y ** 3 / 3)), abs(sx(x * z ** 2)),
             1e-300)                       # second-moment scale, m^5
    S1 = np.sqrt(S2 * abs(V))              # first-moment scale, m^4
    out["xx"] = agree("xx", sx(x ** 3 / 3), sy(y * x ** 2), S2)
    out["yy"] = agree("yy", sx(x * y ** 2), sy(y ** 3 / 3), S2)
    out["zz"] = agree("zz", sx(x * z ** 2), sy(y * z ** 2), S2)
    out["x"] = agree("x", sx(x ** 2 / 2), sy(y * x), S1)
    out["y"] = agree("y", sx(x * y), sy(y ** 2 / 2), S1)
    out["z"] = agree("z", sx(x * z), sy(y * z), S1)
    out["xy"] = agree("xy", sx(x ** 2 * y / 2), sy(x * y ** 2 / 2), S2)
    out["yz"] = agree("yz", sx(x * y * z), sy(y ** 2 * z / 2), S2)
    out["zx"] = agree("zx", sx(x ** 2 * z / 2), sy(x * y * z), S2)
    # the ill-conditioned third form, kept only so callers can SEE the problem
    out["xx_nz"] = float(np.sum(nz * z * x ** 2 * a))
    return out


def inertia_from_mesh(mesh, cog, mass=None, rho=RHO):
    """Full 6x6 rigid-body inertia about `cog`, uniform density."""
    mom = volume_moments(mesh.faces_centers, mesh.faces_normals,
                         mesh.faces_areas, rc=cog)
    V = mom["V"]
    m = rho * V if mass is None else float(mass)
    d = m / V                                   # density that gives that mass
    M = np.zeros((6, 6))
    M[0, 0] = M[1, 1] = M[2, 2] = m
    M[3, 3] = d * (mom["yy"] + mom["zz"])
    M[4, 4] = d * (mom["xx"] + mom["zz"])
    M[5, 5] = d * (mom["xx"] + mom["yy"])
    M[3, 4] = M[4, 3] = -d * mom["xy"]
    M[4, 5] = M[5, 4] = -d * mom["yz"]
    M[3, 5] = M[5, 3] = -d * mom["zx"]
    return M, mom


# ------------------------------------------------------------ analytic Wigley
def wigley_inertia(L, B, T, zg, rho=RHO):
    """Closed form for the Wigley hull  y = (B/2)(1 - (2x/L)^2)(1 - (z/T)^2).

    With xi = 2x/L in [-1, 1] and zeta = z/T in [-1, 0],
        dV = B (1 - xi^2)(1 - zeta^2) (L/2) T dxi dzeta
    and every moment reduces to a product of two elementary integrals:
        I0 = int_-1^1 (1-xi^2)     dxi   = 4/3      J0 = int_-1^0 (...) = 2/3
        I2 = int_-1^1 xi^2(1-xi^2) dxi   = 4/15     J1 = ...            = -1/4
        I3 = int_-1^1 (1-xi^2)^3   dxi   = 32/35    J2 = ...            = 2/15
                                                    J3 = int (1-z^2)^3  = 16/35
    The transverse moment uses int_{-y}^{y} s^2 ds = (2/3) y^3.
    """
    a, h = L / 2.0, B / 2.0
    I0, I2, I3 = 4 / 3, 4 / 15, 32 / 35
    J0, J1, J2, J3 = 2 / 3, -1 / 4, 2 / 15, 16 / 35
    V = B * a * T * I0 * J0
    xx = B * a ** 3 * T * I2 * J0
    zz0 = B * a * T ** 3 * I0 * J2                  # about z = 0
    z1 = B * a * T ** 2 * I0 * J1                   # int z dV, about z = 0
    yy = (2 / 3) * h ** 3 * a * T * I3 * J3
    zz = zz0 - 2 * zg * z1 + zg ** 2 * V            # parallel axis to z = zg
    m = rho * V
    M = np.zeros((6, 6))
    M[0, 0] = M[1, 1] = M[2, 2] = m
    M[3, 3] = rho * (yy + zz)
    M[4, 4] = rho * (xx + zz)
    M[5, 5] = rho * (xx + yy)
    return M, dict(V=V, xx=xx, yy=yy, zz=zz, z_buoy=z1 / V)


def with_gyradius(M, L, B, k_roll=None, k_pitch=None, k_yaw=None):
    """Override the rotational inertia with an explicit radius of gyration.

    This is where a real vessel's mass distribution goes -- machinery, fuel,
    structure, payload. It is a MODELLING assumption and is deliberately not the
    default, but it must be reachable, because the geometric value is the
    inertia of a hull-shaped block of water and no vessel is one.
    """
    M = M.copy()
    m = M[0, 0]
    for i, k, ref in ((3, k_roll, B), (4, k_pitch, L), (5, k_yaw, L)):
        if k is not None:
            M[i, i] = m * (k * ref) ** 2
    return M


# ------------------------------------------------------------- external truth
def verify(verbose=True):
    """Check the mesh routine against three exact results it cannot influence.

    Returns the worst relative error seen. This is the only kind of test in the
    project that can fail for a reason outside the project.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import capytaine as cpt

    worst = 0.0
    rows = []

    # 1. solid hemisphere: I = (2/5) m R^2 about the sphere centre, every axis
    for R in (1.0, 2.5):
        b = cpt.FloatingBody(mesh=cpt.mesh_sphere(radius=R, center=(0, 0, 0),
                                                  resolution=(60, 60)),
                             center_of_mass=(0, 0, 0))
        b.add_all_rigid_body_dofs()
        b = b.immersed_part()
        M, mom = inertia_from_mesh(b.mesh, (0, 0, 0))
        exact = 0.4 * M[0, 0] * R ** 2
        for i, nm in ((3, "roll"), (4, "pitch"), (5, "yaw")):
            e = (M[i, i] - exact) / exact
            worst = max(worst, abs(e))
            rows.append((f"hemisphere R={R} {nm}", M[i, i], exact, e))

    # 2. rectangular box, exactly half immersed: I_xx = m (b^2 + c^2) / 12
    bx, by, bz = 3.0, 2.0, 1.0                       # full box, immersed depth 1
    mesh = cpt.mesh_parallelepiped(size=(bx, by, 2 * bz), center=(0, 0, 0),
                                   resolution=(30, 20, 20))
    b = cpt.FloatingBody(mesh=mesh, center_of_mass=(0, 0, -bz / 2))
    b.add_all_rigid_body_dofs()
    b = b.immersed_part()
    M, mom = inertia_from_mesh(b.mesh, (0, 0, -bz / 2))
    m = M[0, 0]
    for i, nm, ex in ((3, "roll", m * (by ** 2 + bz ** 2) / 12),
                      (4, "pitch", m * (bx ** 2 + bz ** 2) / 12),
                      (5, "yaw", m * (bx ** 2 + by ** 2) / 12)):
        e = (M[i, i] - ex) / ex
        worst = max(worst, abs(e))
        rows.append((f"half-immersed box {nm}", M[i, i], ex, e))

    # 3. the Wigley hull itself, mesh against closed form
    import hydro.run_wigley as RW
    body = RW.build_body(nx=120, nz=40)
    zg = -RW.T / 3
    Mm, _ = inertia_from_mesh(body.mesh, (0, 0, zg))
    Ma, _ = wigley_inertia(RW.L, RW.B, RW.T, zg)
    for i, nm in ((3, "roll"), (4, "pitch"), (5, "yaw")):
        e = (Mm[i, i] - Ma[i, i]) / Ma[i, i]
        worst = max(worst, abs(e))
        rows.append((f"wigley mesh vs closed form {nm}", Mm[i, i], Ma[i, i], e))

    if verbose:
        print("  external checks -- exact mathematics, no simulation involved\n")
        print(f"  {'case':<36}{'computed':>12}{'exact':>12}{'error':>9}")
        for nm, got, ex, e in rows:
            print(f"  {nm:<36}{got:>12.4g}{ex:>12.4g}{e:>8.2%}"
                  + ("" if abs(e) < 0.01 else "   <-- FAIL"))
        print(f"\n  worst error {worst:.2%}")
    return worst


# ------------------------------------------------------------- what it changes
def sensitivity(db=None, path="hydro_wigley_10m.npz"):
    """Natural periods under: the stored M, the corrected M, and real gyradii.

    The point is not that one column is right. It is that the load-bearing
    claims of this project have to survive the whole span, and the reader gets
    to see the span instead of a single number.
    """
    from hydro import bem
    from sim.cummins import radiation_memory
    import hydro.run_wigley as RW

    db = bem.load(path) if db is None else db
    A = radiation_memory(db).A_inf
    L, B, T = RW.L, RW.B, RW.T
    Ma, _ = wigley_inertia(L, B, T, -T / 3)
    cases = [("stored (Capytaine)", db.M),
             ("corrected geometry", Ma)]
    for k, lab in ((0, "low"), (1, "high")):
        cases.append((f"real gyradius, {lab}",
                      with_gyradius(Ma, L, B, K_ROLL_REAL[k],
                                    K_PITCH_REAL[k], K_YAW_REAL[k])))
    T_e = 7.48
    print(f"\n  natural periods, and the encounter period they are judged "
          f"against (T_e = {T_e:.2f} s)\n")
    print(f"  {'mass matrix':<24}{'T_heave':>9}{'T_roll':>9}{'T_pitch':>9}"
          f"{'w_e/w_pitch':>13}{'w_e/(2w_roll)':>15}")
    for nm, M in cases:
        per = {}
        for i, k in ((2, "heave"), (3, "roll"), (4, "pitch")):
            per[k] = 2 * np.pi * np.sqrt((M[i, i] + A[i, i]) / db.C[i, i])
        print(f"  {nm:<24}{per['heave']:>8.2f}s{per['roll']:>8.2f}s"
              f"{per['pitch']:>8.2f}s{per['pitch']/T_e:>13.3f}"
              f"{per['roll']/(2*T_e):>15.3f}")
    print("\n  w_e/w_pitch << 1 means stiffness-controlled -- the vessel")
    print("  contours the wave instead of resonating with it. It holds across")
    print("  every column, which is what makes it a conclusion rather than a")
    print("  number. Parametric roll needs w_e/(2 w_roll) = 1; the largest")
    print("  value here is far below, so it is out of reach across the span too.")


def main():
    worst = verify()
    sensitivity()
    from hydro import bem
    import hydro.run_wigley as RW
    db = bem.load("hydro_wigley_10m.npz")
    Ma, _ = wigley_inertia(RW.L, RW.B, RW.T, -RW.T / 3)
    print(f"\n  stored vs corrected, the matrix everything else divides by:")
    for i, nm, ref, rl in ((3, "roll", RW.B, "B"), (4, "pitch", RW.L, "L"),
                           (5, "yaw", RW.L, "L")):
        m = db.M[0, 0]
        print(f"    {nm:<6}{db.M[i,i]:>9.0f} -> {Ma[i,i]:>9.0f} kg m2  "
              f"({100*(db.M[i,i]-Ma[i,i])/Ma[i,i]:+.1f}%)   "
              f"k {np.sqrt(db.M[i,i]/m)/ref:.3f}{rl} -> "
              f"{np.sqrt(Ma[i,i]/m)/ref:.3f}{rl}")
    return worst


if __name__ == "__main__":
    main()
