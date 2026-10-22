"""
Michell's thin-ship wave resistance of a hull form (reference for the
thin-ship limit check of the surface-piercing Neumann-Kelvin solver, B4 check
H8; design WAVEBEM_PORT_DESIGN.md 8.4 S1 "thin-ship limit B/L -> 0 tends to
Michell").

Thin-ship linearisation (Michell 1898): a hull y = +-h(x, z) with h_x, h_z
small is replaced by a source sheet on its centre plane y = 0 with flux
density m = -2 U h_x per unit area (stream -U x_hat: dphi/dy(0+) = m/2 =
-U h_x), and the free-surface condition is the linear Kelvin one, so the wave
resistance follows from the Kochin function of the sheet with Havelock's
energy-flux formula (havelock.py):
    H(theta) = int int m(x, z) exp(kappa z - i kappa x cos theta) dx dz,
    kappa = k0 sec^2 theta,  R = (rho k0^2/pi) int_0^{pi/2} sec^3 |H|^2 d theta.
As B/L -> 0 at fixed T/L the Neumann-Kelvin solution with the exact hull
tends to this (the body condition transferred to the centre plane, the
waterline and sheet geometry collapse), so R_NK(B)/R_Michell(B) -> 1 with an
O(B/L) thickness correction; it is a limit, not a ceiling (no claim at
finite B).
"""
import numpy as np
from scipy.integrate import quad


def michell_resistance(hull, U, g=9.81, rho=1.0, nx=400, nz=200):
    """R_Michell of a HullForm (hull.py) at speed U: Gauss-Legendre in x and
    in z over [-D(x), 0] (h_x by a central difference of the form), theta
    integral by adaptive quadrature in t = tan(theta)."""
    k0 = g / U ** 2
    xg, wg = np.polynomial.legendre.leggauss(nx)
    x = hull.x_aft + 0.5 * hull.L * (xg + 1)
    wx = 0.5 * hull.L * wg
    zg, wzg = np.polynomial.legendre.leggauss(nz)
    D = hull.D(x)
    Z = -D[:, None] * 0.5 * (zg[None, :] + 1)
    WZ = 0.5 * D[:, None] * wzg[None, :]
    hs = 1e-6 * hull.L
    X = np.broadcast_to(x[:, None], Z.shape)
    hx = (hull.h(X + hs, Z) - hull.h(X - hs, Z)) / (2 * hs)
    m = -2.0 * U * hx * WZ * wx[:, None]

    def H(th):
        kap = k0 / np.cos(th) ** 2
        return np.sum(m * np.exp(kap * Z - 1j * kap * X * np.cos(th)))

    def f(t):
        th = np.arctan(t)
        return np.sqrt(1 + t * t) * abs(H(th)) ** 2
    val, _ = quad(f, 0.0, np.inf, limit=1000, epsabs=0.0, epsrel=1e-10)
    return rho * k0 ** 2 / np.pi * val


def wigley_kochin(theta, U, g, L, B, T):
    """Closed-form Kochin function of Michell's centre-plane sheet for the
    Wigley form y = (B/2)(1 - (2x/L)^2)(1 - (z/T)^2) (added after the B4
    review: replaces the 2-D quadrature of michell_resistance for the H8
    family, which dominated the tier time). With m = -2 U h_x =
    (8 U B/L^2) x (1 - z^2/T^2), kappa = k0 sec^2 theta, a = kappa cos theta:
        H = (8 U B/L^2) X Z,
        X = int_{-L/2}^{L/2} x e^{-i a x} dx
          = -2i [sin(a L/2)/a^2 - (L/2) cos(a L/2)/a],
        Z = int_{-T}^0 (1 - z^2/T^2) e^{kappa z} dz
          = 1/kappa - 2/(kappa^3 T^2) + e^{-kappa T} (2/(kappa^2 T) + 2/(kappa^3 T^2)).
    Returns complex H (array over theta)."""
    th = np.asarray(theta, float)
    k0 = g / U ** 2
    kap = k0 / np.cos(th) ** 2
    a = kap * np.cos(th)
    X = -2j * (np.sin(a * L / 2) / a ** 2 - (L / 2) * np.cos(a * L / 2) / a)
    e = np.exp(-kap * T)
    Z = 1 / kap - 2 / (kap ** 3 * T ** 2) + e * (2 / (kap ** 2 * T) + 2 / (kap ** 3 * T ** 2))
    return (8 * U * B / L ** 2) * X * Z


def michell_wigley(U, g=9.81, rho=1.0, L=1.0, B=0.1, T=0.0625):
    """R_Michell of the Wigley form from the closed-form Kochin function
    (wigley_kochin); theta integral by adaptive quadrature in t = tan(theta),
    R = (rho k0^2/pi) int_0^inf sqrt(1 + t^2) |H(atan t)|^2 dt. Agrees with
    michell_resistance(wigley(...)) to its quadrature error (check H8a
    compares the Kochin function with a direct quadrature to 1e-8)."""
    k0 = g / U ** 2

    def f(t):
        return np.sqrt(1 + t * t) * abs(wigley_kochin(np.arctan(t), U, g, L, B, T)) ** 2
    val, _ = quad(f, 0.0, np.inf, limit=2000, epsabs=0.0, epsrel=1e-12)
    return rho * k0 ** 2 / np.pi * val
