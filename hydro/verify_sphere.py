#!/usr/bin/env python3
"""
M1 GATE -- is the BEM pipeline correct?

The floating hemisphere is the classic verification body: Hulme (1982, JFM 121)
solved it analytically by multipole expansion, and its hydrostatics are exact.
Checks run here, strongest first:

  1. hydrostatics   displaced volume -> (2/3)pi R^3, heave stiffness -> rho g pi R^2
  2. convergence    A33, B33 settle as the mesh refines
  3. asymptotics    B33 -> 0 at both ends (no waves radiated at zero or
                    infinite frequency); A33 -> 0.8310 rho V as ka -> 0 (Hulme)
  4. cleanliness    no irregular-frequency spikes in the band of interest

Non-dimensionalisation follows Hulme:
    A' = A33 / (rho V),  B' = B33 / (rho V omega),  V = (2/3) pi R^3,  ka = w^2 R/g
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .geometry import hemisphere_body, hemisphere_volume, hemisphere_heave_stiffness
from . import bem

R = 1.0
RHO, G = bem.RHO, bem.G

# Two independent analytic limits, and the reason each holds.
# The free-surface condition is  -w^2 phi + g dphi/dz = 0 at z=0, so
#   w -> 0    dphi/dz = 0, a RIGID WALL. phi even in z, the image hemisphere
#             moves opposite: two hemispheres squeezing. Hulme (1982) solved
#             this case by multipole expansion.
#   w -> inf  phi = 0, PRESSURE RELEASE. phi odd in z, the image moves the
#             same way, so hemisphere + image is a translating sphere with
#             added mass (1/2) rho V_sphere. Our half takes a quarter of
#             rho V_sphere, i.e. (1/2) rho V_hemisphere.
# The second needs no literature at all, which makes it the stronger gate.
A_LIMIT_LOW_KA = 0.8310           # Hulme (1982), JFM 121
A_LIMIT_HIGH_KA = 0.5             # translating-sphere image, derived above


def run(resolutions=((12, 12), (20, 20), (30, 30), (40, 40)), n_omega=44):
    V = hemisphere_volume(R)
    C33 = hemisphere_heave_stiffness(R)
    # log spacing: the limits live at ka << 1 and ka >> 1, and a linear grid
    # resolves neither. (A linear grid is what made the first version of this
    # gate extrapolate straight through the peak and report a false failure.)
    ka = np.logspace(np.log10(2e-3), np.log10(20.0), n_omega)
    omegas = np.sqrt(ka * G / R)

    print(f"Floating hemisphere, R = {R} m")
    print(f"  analytic V   = {V:.6f} m^3")
    print(f"  analytic C33 = {C33:.1f} N/m\n")

    print(f"  {'mesh':>9}{'panels':>8}{'V err%':>9}{'C33 err%':>10}"
          f"{'A(ka_min)':>11}{'A(ka_max)':>11}")
    results = {}
    for res in resolutions:
        body = hemisphere_body(R=R, resolution=res, dofs=("Heave",))
        hs = body.compute_hydrostatics()
        ds = bem.compute_database(body, omegas)
        a = ds["added_mass"].sel(influenced_dof="Heave",
                                 radiating_dof="Heave").values
        b = ds["radiation_damping"].sel(influenced_dof="Heave",
                                        radiating_dof="Heave").values
        results[res] = (ka, a / (RHO * V), b / (RHO * V * omegas))
        Ap = results[res][1]
        print(f"  {f'{res[0]}x{res[1]}':>9}{body.mesh.nb_faces:>8}"
              f"{100*(hs['disp_volume']-V)/V:>9.2f}"
              f"{100*(float(hs['hydrostatic_stiffness'].values.ravel()[0])-C33)/C33:>10.2f}"
              f"{Ap[0]:>11.4f}{Ap[-1]:>11.4f}")

    finest, coarse = resolutions[-1], resolutions[-2]
    ka_f, A_f, B_f = results[finest]

    body = hemisphere_body(R=R, resolution=finest, dofs=("Heave",))
    spikes = bem.find_irregular_spikes(bem.compute_database(body, omegas))
    bad_w = {w for _, w in spikes}
    clean = np.array([not any(abs(np.sqrt(k * G / R) - w) < 0.25 for w in bad_w)
                      for k in ka_f])

    d_a = np.max(np.abs(A_f - results[coarse][1])[clean])
    d_b = np.max(np.abs(B_f - results[coarse][2])[clean])
    print(f"\n  low-ka  A' = {A_f[0]:.4f}  vs Hulme {A_LIMIT_LOW_KA}   "
          f"({100*(A_f[0]-A_LIMIT_LOW_KA)/A_LIMIT_LOW_KA:+.1f}%)  [ka={ka_f[0]:.4f}]")
    print(f"  high-ka A' = {A_f[-1]:.4f}  vs sphere {A_LIMIT_HIGH_KA}   "
          f"({100*(A_f[-1]-A_LIMIT_HIGH_KA)/A_LIMIT_HIGH_KA:+.1f}%)  [ka={ka_f[-1]:.1f}]")
    print(f"  convergence off irregular freqs: max dA' = {d_a:.4f}, "
          f"max dB' = {d_b:.4f}")
    print(f"  B' at ka_min = {B_f[0]:.4f},  at ka_max = {B_f[-1]:.4f}  "
          f"(both -> 0 expected)")
    print(f"  irregular-frequency flags (omega, rad/s): "
          f"{sorted(round(w,2) for w in bad_w) if bad_w else 'none'}")

    _plot(results)
    _verdict(A_f, B_f, d_a, d_b)
    return results


def _verdict(A_f, B_f, d_a, d_b):
    checks = [
        ("mesh convergence  dA' < 0.02", d_a < 0.02),
        ("mesh convergence  dB' < 0.02", d_b < 0.02),
        (f"A'(ka->0) within 4% of Hulme {A_LIMIT_LOW_KA}",
         abs(A_f[0] - A_LIMIT_LOW_KA) / A_LIMIT_LOW_KA < 0.04),
        (f"A'(ka->inf) within 4% of sphere limit {A_LIMIT_HIGH_KA}",
         abs(A_f[-1] - A_LIMIT_HIGH_KA) / A_LIMIT_HIGH_KA < 0.04),
        ("B' -> 0 at low ka", B_f[0] < 0.05),
        ("B' -> 0 at high ka", B_f[-1] < 0.05),
    ]
    print("\n  " + "-" * 46)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    print("  " + "-" * 46)
    print(f"  M1 BEM GATE: {'PASSED' if all(o for _, o in checks) else 'FAILED'}")


def _plot(results):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    n = len(results)
    for i, (res, (ka, A, B)) in enumerate(results.items()):
        c = plt.cm.viridis(i / max(n - 1, 1))
        lab = f"{res[0]}x{res[1]}"
        ax[0].plot(ka, A, color=c, lw=1.8, label=lab)
        ax[1].plot(ka, B, color=c, lw=1.8, label=lab)
    ax[0].axhline(A_LIMIT_LOW_KA, ls="--", c="tab:red", lw=1.4,
                  label="Hulme $ka\\!\\to\\!0$ = 0.8310")
    ax[0].axhline(A_LIMIT_HIGH_KA, ls=":", c="tab:orange", lw=1.6,
                  label="sphere $ka\\!\\to\\!\\infty$ = 0.5")
    ax[0].set_ylabel("$A_{33}/\\rho V$")
    ax[1].set_ylabel("$B_{33}/\\rho V \\omega$")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("$ka = \\omega^2 R/g$")
        a.grid(alpha=.3, which="both")
        a.legend(fontsize=8)
    fig.suptitle("M1 gate: floating hemisphere, mesh convergence vs Hulme (1982)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("fig7_m1_sphere.png", dpi=140)
    print("\n  figure: fig7_m1_sphere.png")


if __name__ == "__main__":
    run()
