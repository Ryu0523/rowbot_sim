"""
Zero-speed radiation and diffraction with Rankine panels on the body and on a
truncated free surface with a numerical beach (design section 8.3 P-2/P-4 at
U = 0; step R2 checks at U = 0).

Equations (time factor e^{i omega t}; normals into the fluid, so on the free
surface n = -z and dphi/dn = -phi_z):

  kinematic   zeta_t = phi_z - nu(x) zeta
  dynamic     phi_t  = -g zeta - nu(x) phi
  =>          phi_z  = K_e phi,   K_e(x) = (omega - i nu(x))^2 / g          (B)

This is the linear free-surface pair with Newtonian damping in both
conditions (Kring 1994, ch. 4; the same form P-2 uses in the time domain), so
the frequency response here is exactly the transfer function of the
time-domain linear system with the same beach nu(x). nu = 0 recovers
-omega^2 phi + g phi_z = 0. With identity (E) of bie.py on body + free surface:

  sum_j [2 pi delta_ij - D_ij - S_ij K_e,j 1(j in FS)] phi_j = -sum_{j in B} S_ij (dphi/dn)_j

one complex dense system per frequency, unknown phi on body and free surface.

Radiation coefficients for unit-velocity potentials phi_k (dphi_k/dn = n_k):
  force on the body F_j = i omega rho int phi n_j dS (pressure -rho phi_t,
  force -int p n dS), so with phi = i omega xi_k phi_k
      A_jk - i B_jk / omega = -rho int_B phi_k n_j dS.
Excitation (incident wave potential phi_I, diffraction phi_D with
dphi_D/dn = -dphi_I/dn):  X_j = i omega rho int_B (phi_I + phi_D) n_j dS;
Haskind:                  X_j = i omega rho int_B (phi_I n_j - phi_j dphi_I/dn) dS,
valid here because phi_D and phi_j satisfy the same local condition (B), so
the free-surface part of Green's second identity cancels pointwise.
"""
import numpy as np

from .bie import ImageOperator, lu_solve


class ZeroSpeedFS:
    """Body panels + free-surface panels (tag 1 in `fs`) with images."""

    def __init__(self, body, fs, axes=(), parities=None, device=None):
        self.body, self.fs = body, fs
        self.pan = body + fs
        self.NB = len(body)
        self.axes = tuple(axes)
        self.device = device
        self.op = ImageOperator(self.pan, axes=self.axes, device=device,
                                parities=parities)

    def solve(self, omega, g, nu, phin_body, parity=None):
        """phi on all panels (N, m) for body normal data phin_body (NB, m);
        nu: beach damping per free-surface panel (array or scalar)."""
        NB = self.NB
        S, D = self.op.S(parity), self.op.D(parity)
        K = (omega - 1j * np.broadcast_to(np.asarray(nu, float), (len(self.fs),))) ** 2 / g
        A = 2 * np.pi * np.eye(len(S), dtype=complex) - D
        A[:, NB:] -= S[:, NB:] * K[None, :]
        rhs = -S[:, :NB] @ np.asarray(phin_body, complex).reshape(NB, -1)
        return lu_solve(A, rhs, self.device)

    def surface_elevation(self, phi, omega, g, nu):
        """zeta on the free-surface panels: zeta = -(i omega + nu) phi / g."""
        return -(1j * omega + np.asarray(nu)) * phi[self.NB:] / g


def radiation_coefficients(phi_k_body, n_j, area, omega, rho, copies=1):
    """(A_jk, B_jk) from the complex body potential of mode k."""
    I = copies * np.sum(phi_k_body * n_j * area)
    return -rho * I.real, omega * rho * I.imag


def beach(r, omega, r_start, length, strength=1.0):
    """Beach damping nu(r) = strength * omega * ((r - r_start)/length)^2 for
    r_start < r < r_start + length, constant beyond, 0 inside."""
    s = np.clip((np.asarray(r) - r_start) / length, 0.0, 1.0)
    return strength * omega * s ** 2


def incident_even_x(points, normals, omega, g, beta_amp=1.0):
    """x-even part of a deep-water incident wave travelling along +x,
    eta = A exp(i(omega t - k x)): phi_I = (i g A/omega) e^{kz} e^{-ikx};
    its x-even part (i g A/omega) e^{kz} cos(kx) carries all heave and pitch-
    free (x-even) excitation. Returns (phi, dphi/dn)."""
    k = omega ** 2 / g
    x, z = points[:, 0], points[:, 2]
    c0 = 1j * g * beta_amp / omega
    ph = c0 * np.exp(k * z) * np.cos(k * x)
    gx = -c0 * k * np.exp(k * z) * np.sin(k * x)
    gz = c0 * k * np.exp(k * z) * np.cos(k * x)
    return ph, gx * normals[:, 0] + gz * normals[:, 2]


def incident_odd_x(points, normals, omega, g, beta_amp=1.0):
    """x-odd part: (i g A/omega) e^{kz} (-i sin kx). Returns (phi, dphi/dn)."""
    k = omega ** 2 / g
    x, z = points[:, 0], points[:, 2]
    c0 = 1j * g * beta_amp / omega * (-1j)
    ph = c0 * np.exp(k * z) * np.sin(k * x)
    gx = c0 * k * np.exp(k * z) * np.cos(k * x)
    gz = c0 * k * np.exp(k * z) * np.sin(k * x)
    return ph, gx * normals[:, 0] + gz * normals[:, 2]


# ------------------------------------------------------------ case builders
DEFAULT_BEACH = dict(r_start_wl=1.0, length_wl=3.0, strength=1.0)


def hemisphere_case(ka, a=1.0, g=9.81, n_polar=24, n_az=16, n_wl=20,
                    growth=1.08, beach_kw=None, device=None, parities=((1, 1),)):
    """Floating hemisphere (radius a) + polar free-surface mesh sized by the
    wavelength: radial spacing grows from the body panel size by `growth` up
    to lambda/n_wl, beach from max(r_start_wl lambda, 2a) over length_wl
    lambda. Quarter domain, images in x and y. Returns (solver, omega, nu)."""
    from . import panels as P
    bk = dict(DEFAULT_BEACH, **(beach_kw or {}))
    k = ka / a
    lam = 2 * np.pi / k
    w = np.sqrt(g * k)
    body = P.hemisphere_quarter(a, n_polar, n_az)
    h0 = min(0.5 * np.pi * a / n_polar, lam / n_wl)
    r0 = max(bk["r_start_wl"] * lam, 2 * a)
    Lb = bk["length_wl"] * lam
    radii = P.graded_radii(a, h0, lam / n_wl, r0 + Lb, growth)
    fs = P.fs_polar_quarter(radii, n_az)
    sol = ZeroSpeedFS(body, fs, axes=(0, 1), parities=list(parities), device=device)
    r = np.linalg.norm(fs.c[:, :2], axis=1)
    nu = beach(r, w, r0, Lb, bk["strength"])
    return sol, w, nu


def barge_case(omega, L, B, T, g=9.81, nb=(12, 6, 4), n_wl=15, growth=1.1,
               beach_kw=None, device=None, parities=((1, 1), (-1, 1))):
    """Rectangular barge + free-surface mesh offset from the rectangular
    waterline (panels.fs_offset_quarter), quadrant x, y >= 0 with images.
    Returns (solver, nu)."""
    from . import panels as P
    bk = dict(DEFAULT_BEACH, **(beach_kw or {}))
    k = omega ** 2 / g
    lam = 2 * np.pi / k
    body = P.box_quarter(L, B, T, *nb)
    ys = np.linspace(0, B / 2, nb[1] + 1)
    xs = np.linspace(L / 2, 0, nb[0] + 1)
    wl = np.concatenate([np.stack([L / 2 + 0 * ys, ys], 1),
                         np.stack([xs[1:], B / 2 + 0 * xs[1:]], 1)])
    r0 = max(bk["r_start_wl"] * lam, L)
    Lb = bk["length_wl"] * lam
    h0 = min(0.5 * B / nb[1], 0.5 * L / nb[0], lam / n_wl)
    offs = P.graded_radii(0.0, h0, lam / n_wl, r0 + Lb, growth)
    fs = P.fs_offset_quarter(wl, offs)
    sol = ZeroSpeedFS(body, fs, axes=(0, 1), parities=list(parities), device=device)
    r = np.linalg.norm(fs.c[:, :2], axis=1)
    nu = beach(r, omega, r0, Lb, bk["strength"])
    return sol, nu
