#!/usr/bin/env python3
"""
Experiment: predict the KVLCC2 roll decays from the lines and the loading
condition, then compare with the SSPA measurements.

NOTHING HERE IS FITTED TO THE DECAYS. Inputs:
  lines      SIMMAN IGES (data/external/kvlcc2_geometry), full scale,
             imported at 1:68 by hydro.cad_import
  loading    SSPA: volume 312653 m3, KG 18.6 m, kxx 23.2 m (0.40 B), kzz 80 m,
             LCG 11.27 m forward of midship, fresh water
             (data/external/kvlcc2_rolldecay/model_test_parameters.csv)
  bilge keels: none. SIMMAN lists none for any KVLCC2 model; the SSPA release
             leaves its BKL/BKB columns empty. Not confirmed for this model.
The project's own code does the rest:
  hydro.cad_import  mesh; waterline from the volume; GM from the waterplane
  Capytaine (BEM)   added mass and wave damping for sway, roll and yaw, about
                    G, zero speed
  hydro.ikeda       viscous damping: friction, eddy and (with speed) lift
The prediction is the coupled sway-roll-yaw system, with sway and yaw free (a
decay-test model is free to drift) and roll restored by rho g V GM. Sway and
yaw are eliminated exactly at each frequency, leaving a roll impedance
Z(w) = C44 - w^2 I_eff(w) + i w B_eff(w).

WHAT EACH COMPARISON TESTS
  natural period  C44 is exact given the lines and KG; I44 uses kxx, a target
                  value (0.40 B, presumably ballasted to); the added inertia
                  is the one modelled quantity.
  damping         BEM wave damping + Ikeda viscous parts, against the fitted
                  zeta_eq(phi_a) of studies/exp_kvlcc2_rolldecay.py, at 0 kn
                  (two repeats) and 15.5 kn.

Run: python -m studies.exp_kvlcc2_roll_prediction
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IGS = os.path.join(HERE, "data", "external", "kvlcc2_geometry", "kvlcc2.igs")
SCALE = 68.0
RHO, G = 1000.0, 9.81            # SSPA: rho 1000, g 9.81 (model_test_parameters)
NU = 1.14e-6                     # fresh water at 15 C; the tank temperature is not recorded
FULL = dict(Lpp=320.0, B=58.0, T=20.8, V=312653.0, KG=18.6, GM=5.73, kxx=23.2,
            kzz=80.0, lcg=11.2672, C_B=0.8098, C_M=0.998)
KNOTS = 0.514444
OMEGAS = np.array([1.6, 1.9, 2.1, 2.25, 2.35, 2.45, 2.55, 2.65, 2.8, 3.1])
DOFS = ["Sway", "Roll", "Yaw"]


def m(q, p=1):
    """full scale -> model scale, for a quantity of length dimension p"""
    return q / SCALE ** p


def hull(size_max=5.0, curvature=None):
    from hydro.cad_import import import_hull
    mesh, rep = import_hull(IGS, volume=m(FULL["V"], 3), scale=1.0 / SCALE,
                            size_max=size_max, curvature=curvature,
                            forward="-x", up="-z", symmetric=True,
                            verbose=False)
    return mesh, rep


def bem(mesh, x_G, z_G, omegas=OMEGAS):
    """A, B (n_omega, 3, 3) for sway, roll, yaw about G; [influenced, radiating]."""
    import capytaine as cpt
    import xarray as xr
    body = cpt.FloatingBody(
        mesh=mesh, center_of_mass=(x_G, 0.0, z_G),
        dofs=cpt.rigid_body_dofs(only=DOFS, rotation_center=(x_G, 0.0, z_G)))
    body = body.immersed_part()
    test = xr.Dataset({"omega": omegas, "radiating_dof": DOFS,
                       "rho": [RHO], "g": [G], "water_depth": [np.inf]})
    ds = cpt.BEMSolver().fill_dataset(test, body, _check_wavelength=False)
    A = ds["added_mass"].squeeze().transpose(
        "omega", "influenced_dof", "radiating_dof").sel(
        influenced_dof=DOFS, radiating_dof=DOFS).values
    B = ds["radiation_damping"].squeeze().transpose(
        "omega", "influenced_dof", "radiating_dof").sel(
        influenced_dof=DOFS, radiating_dof=DOFS).values
    return A, B


def roll_impedance(w, omegas, A, B, Mrb, C44, B_extra=0.0):
    """Z(w) with sway and yaw eliminated. A, B interpolated in w."""
    Aw = np.array([[np.interp(w, omegas, A[:, i, j]) for j in range(3)]
                   for i in range(3)])
    Bw = np.array([[np.interp(w, omegas, B[:, i, j]) for j in range(3)]
                   for i in range(3)])
    Z = -w * w * (Mrb + Aw) + 1j * w * Bw
    Z[1, 1] += C44 + 1j * w * B_extra
    s = [0, 2]
    return Z[1, 1] - Z[1, s] @ np.linalg.solve(Z[np.ix_(s, s)], Z[s, 1])


def natural(omegas, A, B, Mrb, C44, coupled=True):
    """Undamped natural frequency of the (condensed) roll equation and the
    effective inertia and wave damping there."""
    from scipy.optimize import brentq

    def parts(w):
        if coupled:
            Z = roll_impedance(w, omegas, A, B, Mrb, C44)
            I_eff = (C44 - Z.real) / (w * w)
            B_eff = Z.imag / w
        else:
            I_eff = Mrb[1, 1] + np.interp(w, omegas, A[:, 1, 1])
            B_eff = np.interp(w, omegas, B[:, 1, 1])
        return I_eff, B_eff

    wn = brentq(lambda w: C44 - w * w * parts(w)[0], omegas[0], omegas[-1])
    I_eff, B_eff = parts(wn)
    return wn, I_eff, B_eff


def viscous(w, phi_a, U, d):
    from hydro import ikeda
    r = ikeda.roll_damping(L=m(FULL["Lpp"]), B=m(FULL["B"]), d=d,
                           C_B=FULL["C_B"], C_M=FULL["C_M"], KG=m(FULL["KG"]),
                           w=w, phi_a=phi_a, U=U, V=m(FULL["V"], 3), rho=RHO,
                           nu=NU, g=G, B_W0=0.0)
    return r


def main(size_max=5.0, curvature=None):
    import warnings
    warnings.filterwarnings("ignore")
    from hydro import ikeda
    from studies import exp_kvlcc2_rolldecay as meas

    print("\nEXPERIMENT -- KVLCC2 roll decay predicted from the lines and the "
          "loading condition (1:68)\n")
    fits = {run: meas.fit(*meas.decay_window(*meas.load(run)))
            for run in meas.RUNS}

    mesh, rep = hull(size_max, curvature)
    d = rep["draft"]
    KG = m(FULL["KG"])
    GM = rep["KM"] - KG
    V = rep["volume"]
    x_G = m(160.0 + FULL["lcg"])           # vessel axes: AP at x = 0
    z_G = KG - d                           # waterline at z = 0
    mass = RHO * V
    I44 = mass * m(FULL["kxx"]) ** 2
    I66 = mass * m(FULL["kzz"]) ** 2
    C44 = RHO * G * V * GM
    print(f"  mesh: {rep['n_panels']} wetted panels (file size_max {size_max} m"
          f"{', curvature ' + str(curvature) if curvature else ''}), draft "
          f"{d*SCALE:.3f} m full scale (published {FULL['T']})")
    print(f"  GM from the waterplane {GM*SCALE:.3f} m full scale, SSPA "
          f"{FULL['GM']} m ({GM*SCALE/FULL['GM']-1:+.2%})")
    print(f"  dry roll period 2 pi kxx / sqrt(g GM) = "
          f"{2*np.pi*np.sqrt(I44/C44):.3f} s (no added inertia)")

    A, B = bem(mesh, x_G, z_G)
    Mrb = np.diag([mass, I44, I66])
    print(f"\n  BEM about G: A44 / I44 = {A[0,1,1]/I44:.3f} .. "
          f"{A[-1,1,1]/I44:.3f} over w = {OMEGAS[0]}..{OMEGAS[-1]} rad/s")

    meas_T = np.mean([fits[r]["T"] for r in ("21337", "21338")])
    rows = {}
    for lab, cpl in (("roll alone", False), ("sway-roll-yaw", True)):
        wn, I_eff, B_eff = natural(OMEGAS, A, B, Mrb, C44, coupled=cpl)
        rows[lab] = (wn, I_eff, B_eff)
        print(f"  {lab:<14} T_n {2*np.pi/wn:.3f} s  (measured {meas_T:.3f}, "
              f"{2*np.pi/wn/meas_T-1:+.1%})   I_eff/I44 {I_eff/I44:.3f}   "
              f"wave zeta {B_eff/(2*I_eff*wn):.5f}")

    wn, I_eff, B_wave = rows["sway-roll-yaw"]
    zcrit = 2.0 * I_eff * wn
    print("\n  damping, zeta_eq by component -- predicted, and measured "
          "(fits of the decays)\n")
    wh = wn * np.sqrt(m(FULL["B"]) / (2 * G))
    reg = ikeda.wave_hat(FULL["B"] / FULL["T"], FULL["C_B"], FULL["C_M"],
                         (FULL["T"] - FULL["KG"]) / FULL["T"], wh)
    B_reg = reg * RHO * V * m(FULL["B"]) ** 2 / np.sqrt(m(FULL["B"]) / (2 * G))
    print(f"  wave damping: BEM {B_wave/zcrit:.5f}, Ikeda regression "
          f"{B_reg/zcrit:.5f} (w_hat {wh:.3f}; regression out of range: "
          f"{ikeda.check_range(FULL['B']/FULL['T'], FULL['C_B'], FULL['C_M'], (FULL['T']-FULL['KG'])/FULL['T'], wh)})")
    print(f"\n  {'kn':>5}{'phi_a':>7}{'wave':>9}{'fric':>9}{'eddy':>9}{'lift':>9}"
          f"{'total':>9}   measured")
    out = []
    for kn, runs in ((0.0, ("21337", "21338")), (15.5, ("21340",))):
        U = m(kn * KNOTS, 0.5)
        fw = ikeda.wave_speed_factor(wn, d, U, G)
        for deg in (2, 5, 9):
            ph = np.radians(deg)
            r = viscous(wn, ph, U, d)
            z = dict(W=B_wave * fw / zcrit, F=r["F"] / zcrit, E=r["E"] / zcrit,
                     L=r["L"] / zcrit)
            tot = sum(z.values())
            mz = [meas.zeta_eq(fits[k], deg) for k in runs]
            out.append((kn, deg, z, tot, mz))
            print(f"  {kn:>5.1f}{deg:>6d}d{z['W']:>9.5f}{z['F']:>9.5f}"
                  f"{z['E']:>9.5f}{z['L']:>9.5f}{tot:>9.5f}   "
                  + " / ".join(f"{v:.5f}" for v in mz)
                  + f"   (pred/meas {tot/np.mean(mz):.2f})")
    print(f"\n  Ikeda range check for this hull: "
          f"{viscous(wn, np.radians(5), 0.0, d)['out_of_range']}")
    # C_M = 0.998 is outside the regression's 0.90-0.99, and the eddy part is
    # the one that is steep in C_M: how much of the 0 kn excess is that?
    print("\n  eddy part at 5 deg, 0 kn, against C_M (the only extrapolated "
          "input):")
    e_pub = viscous(wn, np.radians(5), 0.0, d)["E"]
    for cm in (0.98, 0.99, FULL["C_M"]):
        r = ikeda.roll_damping(L=m(FULL["Lpp"]), B=m(FULL["B"]), d=d,
                               C_B=FULL["C_B"], C_M=cm, KG=m(FULL["KG"]),
                               w=wn, phi_a=np.radians(5), U=0.0,
                               V=m(FULL["V"], 3), rho=RHO, nu=NU, g=G,
                               B_W0=0.0)
        tot = (B_wave + r["F"] + r["E"]) / zcrit
        mz = np.mean([meas.zeta_eq(fits[k], 5) for k in ("21337", "21338")])
        print(f"    C_M {cm:.3f}: eddy {r['E']/e_pub:.2f} x the value used, "
              f"total zeta {tot:.5f} (measured {mz:.5f}, {tot/mz:.2f}x)")
    return dict(rows=rows, table=out, rep=rep, A=A, B=B)


if __name__ == "__main__":
    main()
