#!/usr/bin/env python3
"""
Free-surface reconstruction: the wave the vessel actually sits in.

Everywhere else in this project the vessel's disturbance of the sea enters only
through FORCES -- diffraction force and radiation memory. That is complete for
the dynamics, but it leaves the SURFACE incomplete: `SeaState.eta()` returns the
undisturbed incident wave, so a simulated LiDAR would be looking at water that
does not exist near the hull.

This module closes that gap. The total elevation is

    eta = eta_I  +  eta_D  +  eta_R

  eta_I   incident, the JONSWAP sum, known analytically
  eta_D   diffracted -- the incident wave scattered by a stationary hull
  eta_R   radiated  -- the waves the hull makes by moving

Both disturbance terms come from the same boundary-element solution that already
produced the forces: once the source strengths are known, the potential can be
evaluated anywhere in the fluid, not just on the hull. The free-surface
elevation follows from the dynamic free-surface condition,

    eta = -(1/g) d(phi)/dt   at z = 0.

Because everything is linear, the two disturbance terms combine per frequency
into ONE complex transfer function per field point,

    eta_dist(omega) = eta_D(omega) + sum_j xi_j(omega) eta_R(omega, j)

with xi_j the complex RAO, so run time is a component sum exactly like the
force. The table is body-fixed: the pattern travels with the hull.

CARRIED LIMITATION: still zero forward speed, so there is no steady Kelvin wake
or bow wave -- only the unsteady scattering and radiation. A LiDAR looking a few
hull lengths ahead sees mostly incident wave anyway, which is what the preview
study depends on; the near field is where this matters.
"""
import numpy as np
import capytaine as cpt

from . import bem

G = 9.81
DOFS6 = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]


def field_grid(L, x_span=(-3.0, 3.0), y_span=(-2.0, 2.0), nx=61, ny=41):
    """Body-fixed grid of free-surface points, in hull lengths."""
    x = np.linspace(*x_span, nx) * L
    y = np.linspace(*y_span, ny) * L
    X, Y = np.meshgrid(x, y, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    return x, y, pts, X.shape


def disturbance_transfer(body, db, omegas, beta=np.pi, grid=None, L=10.0,
                         verbose=True):
    """Complex disturbance elevation per unit wave amplitude, per frequency.

    Returns (x, y, eta_dist) with eta_dist shaped (n_omega, nx, ny).
    """
    if grid is None:
        grid = field_grid(L)
    x, y, pts, shape = grid
    solver = cpt.BEMSolver()
    out = np.zeros((len(omegas),) + shape, complex)

    for i, w in enumerate(omegas):
        # diffraction: incident wave on a hull held still
        dif = solver.solve(cpt.DiffractionProblem(body=body, omega=w,
                                                  wave_direction=beta),
                           keep_details=True)
        eta = solver.compute_free_surface_elevation(pts, dif).reshape(shape)

        # radiation: the hull's own waves, weighted by how much it actually
        # moves at this frequency -- the RAO from the same database
        xi, _ = _rao(db, w, beta)
        # In head seas sway, roll and yaw have RAOs at the 1e-4 level by
        # symmetry. Solving their radiation problems costs as much as the ones
        # that matter and contributes nothing, so skip anything negligible
        # relative to the largest response at this frequency.
        cut = 1e-3 * np.max(np.abs(xi))
        for j, dof in enumerate(DOFS6):
            if abs(xi[j]) < max(cut, 1e-12):
                continue
            rad = solver.solve(cpt.RadiationProblem(body=body, omega=w,
                                                    radiating_dof=dof),
                               keep_details=True)
            eta = eta + xi[j] * solver.compute_free_surface_elevation(
                pts, rad).reshape(shape)
        out[i] = eta
        if verbose:
            inc = 1.0
            print(f"    omega={w:5.2f}  max|eta_dist| = "
                  f"{np.max(np.abs(eta))/inc:6.3f} x wave amplitude")
    return x, y, out


def _rao(db, omega, beta):
    k = int(np.argmin(np.abs(db.omega - omega)))
    j = int(np.argmin(np.abs(db.directions - beta)))
    w = db.omega[k]
    Z = -w ** 2 * (db.M + db.A[k]) + 1j * w * db.B[k] + db.C
    return np.linalg.solve(Z, db.F_exc[k, j, :]), w


class FreeSurface:
    """Runtime reconstruction of the total surface around the vessel."""

    def __init__(self, path="freesurface_wigley.npz"):
        z = np.load(path)
        self.x, self.y = z["x"], z["y"]
        self.omega = z["omega"]
        self.eta = z["re"] + 1j * z["im"]           # (n_omega, nx, ny)
        self.L = float(z["L"])

    def total(self, sea, x_v, y_v, t, incident_only=False):
        """Elevation on the body-fixed grid, in EARTH coordinates.

        Returns (Xe, Ye, eta_incident, eta_disturbance).
        """
        Xb, Yb = np.meshgrid(self.x, self.y, indexing="ij")
        Xe, Ye = Xb + x_v, Yb + y_v
        inc = sea.eta(Xe.ravel(), Ye.ravel(), t).reshape(Xb.shape)
        if incident_only:
            return Xe, Ye, inc, np.zeros_like(inc)

        # component sum, exactly as the forces are built
        psi = sea.phases(x_v, y_v, t)
        dist = np.zeros_like(inc)
        for a, wj, pj in zip(sea.a, sea.w, psi):
            k = int(np.argmin(np.abs(self.omega - wj)))
            if abs(self.omega[k] - wj) > 0.35:      # outside the tabulated band
                continue
            dist += a * np.real(self.eta[k] * np.exp(1j * pj))
        return Xe, Ye, inc, dist


def build(path="freesurface_wigley.npz", n_omega=16, band=(0.35, 1.9),
          nx=61, ny=41):
    """Precompute the disturbance table for the 10 m Wigley hull."""
    from .run_wigley import build_body, L
    db = bem.load("hydro_wigley_10m.npz")
    body = build_body(lid=True)
    omegas = np.linspace(*band, n_omega)
    print(f"  {body.mesh.nb_faces} panels, {n_omega} frequencies, "
          f"{nx*ny} field points")
    grid = field_grid(L, nx=nx, ny=ny)
    x, y, eta = disturbance_transfer(body, db, omegas, grid=grid, L=L)
    np.savez_compressed(path, x=x, y=y, omega=omegas,
                        re=eta.real, im=eta.imag, L=L)
    print(f"  saved -> {path}")
    return x, y, eta


if __name__ == "__main__":
    build()
