"""
Far-field (wave-energy) quantities of the steady Neumann-Kelvin problem, as
independent checks of the Rankine solution (added after the 2026-10-06
review of stage 1).

1. Kochin function of a submerged body and the far-field wave resistance.
   Green's second identity in the fluid with the Kelvin Green function G_K
   (the potential of a unit source that satisfies the linear steady
   free-surface condition U^2 phi_xx + g phi_z = 0 and the radiation
   condition) and the Neumann-Kelvin potential phi gives, for a body that
   does not pierce the free surface (the free-surface integral vanishes
   because phi and G_K satisfy the same linear condition there, and there is
   no waterline term),
       phi(x) = int_B [-(dphi/dn) G_K(x, xi) + phi d/dn_xi G_K(x, xi)] dS_xi,
   with G_K normalised like -1/(4 pi r) and n out of the body (into the
   fluid). The first term is a source sheet of flux dphi/dn dS, the second a
   normal-dipole sheet of strength -phi dS (the derivative of a source with
   respect to its position). For a point source of flux Q at xi, Havelock's
   far field (havelock.py) is determined by Q E(xi),
       E(theta; xi) = exp(kappa xi_z - i kappa (xi_x cos theta + xi_y sin theta)),
       kappa = k0 sec^2 theta, k0 = g/U^2;
   by linearity the body's Kochin function is
       H(theta) = int_B [(dphi/dn) E - phi dE/dn_xi] dS                    (H)
   and its wave resistance, by the same energy-flux formula as havelock.py,
       R = (rho k0^2/pi) int_0^{pi/2} sec^3 theta |H(theta)|^2 d theta
   (y-symmetric body). Only body data enter, and only integrals of smooth
   functions against them (no surface gradient), so R_far is free of the
   pressure-integration error of the near field (steady.py docstring). It
   is blind to any field that is harmonic inside the body (Green's second
   identity makes its contribution to (H) vanish), so it complements, not
   replaces, the near field.
   Calibration (check S7a): for the body solve in an unbounded fluid the
   body data reproduce the exterior field of Havelock's line source of the
   spheroid, so H must equal Havelock's H and R_far = R_Havelock up to the
   body discretisation error.

2. Linear Kelvin wave elevation of submerged point sources (check S3 field
   comparison). For a source of flux Q at (x_k, y_k, -f_k) the linear
   steady potential is Havelock's form (havelock.py docstring); its pole
   (wave) part, doubled downstream and cancelled upstream by the far-field
   behaviour of the principal value, is
       phi_W = (Q/pi) int_{-pi/2}^{pi/2} H(-w) kappa e^{kappa (z - f)} sin(kappa w) d theta,
       w = (x - x_k) cos theta + (y - y_k) sin theta, H the Heaviside step,
   so at z = 0
       phi_W,x = (Q k0/pi) int H(-w) sec^2 theta kappa cos theta e^{-kappa f} cos(kappa w) d theta.
   The rest of the field at z = 0 is the double-body part 2 phi_s (source +
   rigid-lid image) plus a local non-wave remainder of relative size
   O(1/(k0 R)^2) at distance R, which is omitted: the comparison uses only
   points more than one transverse wavelength behind every source.
   With t = tan theta: d theta = dt/(1 + t^2), kappa w = k0 sqrt(1 + t^2)
   ((x - x_k) + (y - y_k) t), and the integrand becomes
       (Q k0^2/pi) sqrt(1 + t^2) e^{-k0 (1 + t^2) f} cos(kappa w) H(-w) dt.
"""
import numpy as np
import torch

from .havelock import resistance_from_kochin


def kochin_body(pan, phi, phin, k0, y_image=True):
    """H(theta) of (H) for body panels `pan` (n out of the body) carrying
    phi and dphi/dn; y_image adds the port/starboard mirror (pan = y >= 0
    half, phi even in y). Midpoint rule on each flat panel (O(h^2))."""
    c, n, a = pan.c, pan.n, pan.area
    if y_image:
        m = np.array([1.0, -1.0, 1.0])
        c = np.concatenate([c, c * m])
        n = np.concatenate([n, n * m])
        a = np.concatenate([a, a])
        phi = np.concatenate([phi, phi])
        phin = np.concatenate([phin, phin])

    def H(th):
        kap = k0 / np.cos(th) ** 2
        E = np.exp(kap * c[:, 2] - 1j * kap * (c[:, 0] * np.cos(th) + c[:, 1] * np.sin(th)))
        dEdn = kap * E * (n[:, 2] - 1j * (n[:, 0] * np.cos(th) + n[:, 1] * np.sin(th)))
        return np.sum((phin * E - phi * dEdn) * a)
    return H


def farfield_resistance(pan, phi, phin, U, g=9.81, rho=1.0, y_image=True):
    """Far-field wave resistance of a submerged body from its Kochin
    function (module docstring, item 1)."""
    k0 = g / U ** 2
    return resistance_from_kochin(kochin_body(pan, phi, phin, k0, y_image), k0, rho)


def kelvin_wave_x(Q, X, P, k0, n_t=36001, t_max=None, device=None):
    """phi_W,x at points P (M, 3) on z = 0 of point sources Q_k at X_k
    (depth -X_k[2] > 0), module docstring item 2. Trapezoidal rule in
    t = tan(theta) on [-t_max, t_max]; t_max so that e^{-k0 t^2 f_min} <
    1e-16 by default. GPU if available."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = np.asarray(X, float)
    f = -X[:, 2]
    if t_max is None:
        t_max = float(np.sqrt(37.0 / (k0 * f.min())))
    t = torch.linspace(-t_max, t_max, n_t, dtype=torch.float64, device=dev)
    dt = float(t[1] - t[0])
    s = torch.sqrt(1 + t * t)
    wq = torch.full_like(t, dt)
    wq[0] = wq[-1] = 0.5 * dt
    Pt = torch.as_tensor(np.asarray(P, float), device=dev)
    Xt = torch.as_tensor(X, device=dev)
    ft = torch.as_tensor(f, device=dev)
    Qt = torch.as_tensor(np.asarray(Q, float), device=dev)
    amp = Qt[:, None] * (k0 ** 2 / np.pi) * s[None] * torch.exp(-k0 * (1 + t * t)[None] * ft[:, None]) * wq[None]
    out = torch.zeros(len(Pt), dtype=torch.float64, device=dev)
    blk = max(1, int(3e7 // (len(X) * n_t)))
    for i0 in range(0, len(Pt), blk):
        p = Pt[i0:i0 + blk]
        lin = (p[:, None, 0, None] - Xt[None, :, 0, None]) + (p[:, None, 1, None] - Xt[None, :, 1, None]) * t
        ph = k0 * s * lin                                   # kappa w
        val = torch.where(lin < 0, torch.cos(ph), torch.zeros_like(ph))
        out[i0:i0 + blk] = torch.einsum("nkt,kt->n", val, amp)
    return out.cpu().numpy()
