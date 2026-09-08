#!/usr/bin/env python3
"""
M1, second half -- build the hydrodynamic database for the 10 m Wigley hull.

Two jobs:

  scan()  wide, coarse frequency sweep with and without an interior lid, to
          locate irregular frequencies and to find where B(omega) has decayed
          far enough that the Ogilvie integral can be truncated.

  build() the production database on the frequency grid the scan justifies,
          saved to netCDF for the Cummins model to consume.

Irregular frequencies are a numerical artefact: a resonance of the *interior*
domain enclosed by the hull and the waterplane, which shows up as sharp spikes
in A and B that are not physics. A lid mesh over the interior waterplane
suppresses them. Left untreated they would be transformed straight into the
retardation function and pollute the whole time-domain model.

Wave-direction convention: beta is the direction the waves TRAVEL TOWARD, so
beta = pi is head seas for a vessel heading along +x.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import capytaine as cpt

from .geometry import (wigley_mesh, wigley_volume, wigley_waterplane_area,
                       wigley_heave_stiffness)
from . import bem

L, B, T = 10.0, 2.5, 0.8
NX, NZ = 60, 20
DIRECTIONS = np.linspace(0.0, np.pi, 13)      # 0..180 deg, 15 deg steps


def build_body(nx=NX, nz=NZ, lid=True):
    mesh = wigley_mesh(L, B, T, nx, nz)
    lid_mesh = mesh.generate_lid(z=0.0) if lid else None
    body = cpt.FloatingBody(mesh=mesh, lid_mesh=lid_mesh,
                            center_of_mass=(0.0, 0.0, -T / 3))
    body.add_all_rigid_body_dofs()
    return body.immersed_part()


def scan(omega_max=20.0, n=70):
    """Locate irregular frequencies and the decay of B(omega)."""
    omegas = np.linspace(0.1, omega_max, n)
    out = {}
    print(f"Wigley  L={L} B={B} T={T}   mesh {NX}x{NZ}")
    print(f"  analytic  V={wigley_volume(L,B,T):.4f} m^3  "
          f"A_wp={wigley_waterplane_area(L,B):.4f} m^2  "
          f"C33={wigley_heave_stiffness(L,B):.0f} N/m\n")

    for lid in (False, True):
        body = build_body(lid=lid)
        hs = body.compute_hydrostatics()
        ds = bem.compute_database(body, omegas, directions=[np.pi])
        out["lid" if lid else "nolid"] = ds
        spikes = sorted({round(w, 2) for _, w in bem.find_irregular_spikes(ds)})
        print(f"  lid={str(lid):5s} panels={body.mesh.nb_faces:5d}  "
              f"V err {100*(hs['disp_volume']-wigley_volume(L,B,T))/wigley_volume(L,B,T):+.2f}%  "
              f"irregular-freq flags: {spikes if spikes else 'none'}")

    ds = out["lid"]
    for dof in ("Heave", "Pitch"):
        b = ds["radiation_damping"].sel(influenced_dof=dof,
                                        radiating_dof=dof).values
        peak = omegas[np.argmax(b)]
        tail = b[-1] / max(b.max(), 1e-30)
        print(f"  {dof:6s}: B peaks at omega={peak:.2f} rad/s, "
              f"B(omega_max)/B_peak = {tail:.4f}")

    _plot_scan(omegas, out)
    return out


def build(omega_max=15.0, d_omega=0.1, nx=NX, nz=NZ,
          path="hydro_wigley_10m.npz"):
    """Production database on a uniform grid (uniform because the Ogilvie
    cosine transform downstream wants even spacing).

    d_omega sets how far K(t) can run before the transform wraps:
    t_max = pi/d_omega. The hull's memory is a few seconds, so 0.1 rad/s
    (t_max = 31 s) is ample and costs half the solves of 0.05.
    """
    omegas = np.arange(d_omega, omega_max + d_omega / 2, d_omega)
    body = build_body(nx=nx, nz=nz, lid=True)
    print(f"  {body.mesh.nb_faces} hull panels, {len(omegas)} frequencies, "
          f"{len(DIRECTIONS)} directions ...")
    ds = bem.compute_database(body, omegas, directions=DIRECTIONS,
                              progress=True)
    ds.attrs.update(hull="wigley", L=L, B=B, T=T, nx=nx, nz=nz)
    db = bem.save(ds, path)

    # reciprocity: A and B must be symmetric. Asymmetry is pure discretisation
    # error, so it is a free measure of whether the mesh is fine enough.
    for name, X in (("A", db.A), ("B", db.B)):
        asym = np.max(np.abs(X - X.transpose(0, 2, 1)))
        scale = max(np.max(np.abs(X)), 1e-30)
        print(f"  reciprocity {name}: max|X-X^T|/max|X| = {asym/scale:.2e}")
    print(f"  saved -> {path}")
    print(f"  t_max for K(t) = pi/d_omega = {np.pi/d_omega:.0f} s")
    return db


def _plot_scan(omegas, out):
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for j, dof in enumerate(("Heave", "Pitch")):
        for key, c, ls in (("nolid", "tab:red", "-"), ("lid", "tab:blue", "-")):
            ds = out[key]
            a = ds["added_mass"].sel(influenced_dof=dof, radiating_dof=dof)
            b = ds["radiation_damping"].sel(influenced_dof=dof,
                                            radiating_dof=dof)
            lab = "no lid" if key == "nolid" else "with lid"
            ax[0, j].plot(omegas, a, ls, c=c, lw=1.6, label=lab)
            ax[1, j].plot(omegas, b, ls, c=c, lw=1.6, label=lab)
        ax[0, j].set_title(f"{dof}: added mass $A$", fontsize=11)
        ax[1, j].set_title(f"{dof}: radiation damping $B$", fontsize=11)
        for i in (0, 1):
            ax[i, j].set_xlabel("$\\omega$ (rad/s)")
            ax[i, j].grid(alpha=.3)
            ax[i, j].legend(fontsize=8)
    ax[0, 0].axvspan(0.4, 2.0, color="tab:green", alpha=.10)
    ax[1, 0].axvspan(0.4, 2.0, color="tab:green", alpha=.10,
                     label="SS4-5 wave band")
    fig.suptitle(f"Wigley {L}x{B}x{T} m — irregular-frequency removal "
                 "(spikes without a lid are numerical, not physical)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("fig8_wigley_scan.png", dpi=140)
    print("\n  figure: fig8_wigley_scan.png")


if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        build()
    else:
        scan()
