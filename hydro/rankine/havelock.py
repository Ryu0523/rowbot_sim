"""
Wave resistance of submerged singularity distributions under the linear
steady free surface (Havelock 1928, 1931), as independent references for the
Rankine steady solver.

Derivation (frame fixed to the body, stream -U x_hat, fluid z < 0, Rayleigh
friction mu -> 0+ for the radiation condition; full steps in
learn/meta/RANKINE_STAGE1_2026-10-06.md section 4):

  A source of flux Q at (x_s, y_s, z_s) has phi = -(Q/4 pi)/r. With the
  linear condition U^2 phi_xx - mu U phi_x + g phi_z = 0 on z = 0 and the
  2-D Fourier form of 1/r, the free-surface response is
      -(Q/4 pi)[-1/r' + (g/pi) int int e^{kappa (z + z_s)} e^{i(alpha (x - x_s) + beta (y - y_s))}
                                     / (g kappa - U^2 alpha^2 - i mu U alpha) d alpha d beta],
  kappa = sqrt(alpha^2 + beta^2), r' the distance to the image above z = 0.
  The force on a set of sources follows from Lagally's theorem F = -rho sum Q_k u_k
  (u_k = velocity at source k due to everything else). The principal-value
  part of the wave term is odd in alpha and integrates to zero; the pole
  part (1/(D - i mu U alpha) -> PV + i pi sgn(alpha) delta(D)) gives, with
  kappa_0(theta) = k0 sec^2 theta, k0 = g/U^2,

      R = (rho k0^2 / (4 pi)) int_0^{2 pi} |sec^3 theta| |H(theta)|^2 d theta,
      H(theta) = sum_k Q_k exp(kappa_0 z_k - i kappa_0 (x_k cos theta + y_k sin theta)),

  and for a distribution symmetric in y, R = (rho k0^2/pi) int_0^{pi/2} sec^3 |H|^2.
  For one source at depth f this is R = 16 pi rho k0^2 m^2 int sec^3 e^{-2 k0 f sec^2}
  with m = Q/(4 pi), Havelock's result (Lamb 1932, art. 249 form).

Prolate spheroid (semi-axes a > b, axis along x, centre depth f) moving
along its axis: the unbounded-fluid flow equals that of a line of sources
of density q(xi) = K xi on |xi| < c = a e (Lamb art. 104-105; Havelock 1931),
with K fixed by the dipole moment int q xi d xi = (1 + k1) V U:
K = 3 (1 + k1) V U / (2 c^3). Then
      H = K e^{-kappa_0 f} int xi e^{-i kappa_0 xi cos theta} d xi
        = -2 i K c^2 j1(kappa_0 c cos theta) e^{-kappa_0 f},
j1 the spherical Bessel function. These are Havelock's (1931) formulas; the
check S2 verifies numerically that the line distribution satisfies the
spheroid's boundary condition.
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn

from .analytic import spheroid_lamb


def resistance_from_kochin(H_of_theta, k0, rho=1.0):
    """R for a y-symmetric distribution: (rho k0^2/pi) int_0^{pi/2} sec^3 |H|^2,
    integrated in t = tan(theta) (sec^3 d theta = sqrt(1+t^2) dt)."""
    def f(t):
        th = np.arctan(t)
        return np.sqrt(1 + t * t) * abs(H_of_theta(th)) ** 2
    val, _ = quad(f, 0.0, np.inf, epsabs=0.0, epsrel=1e-12, limit=500)
    return rho * k0 ** 2 / np.pi * val


def kochin_points(Q, X, k0):
    """H(theta) for point sources Q_k at X_k (y-symmetric set listed in full)."""
    Q = np.asarray(Q, float)
    X = np.asarray(X, float)

    def H(th):
        kap = k0 / np.cos(th) ** 2
        return np.sum(Q * np.exp(kap * X[:, 2] - 1j * kap * (X[:, 0] * np.cos(th)
                                                             + X[:, 1] * np.sin(th))))
    return H


def spheroid_line_strength(a, b, U):
    """K, c of the equivalent line source q(xi) = K xi, |xi| < c."""
    L = spheroid_lamb(a, b)
    c = a * L["e"]
    return 3.0 * (1.0 + L["k1"]) * L["V"] * U / (2.0 * c ** 3), c


def spheroid_havelock(a, b, f, U, g=9.81, rho=1.0):
    """Havelock (1931) wave resistance of the submerged prolate spheroid."""
    K, c = spheroid_line_strength(a, b, U)
    k0 = g / U ** 2

    def H(th):
        kap = k0 / np.cos(th) ** 2
        return 2.0 * K * c ** 2 * spherical_jn(1, kap * c * np.cos(th)) * np.exp(-kap * f)
    return resistance_from_kochin(H, k0, rho)


def line_source_points(a, b, f, U, n=64):
    """Gauss-Legendre discretisation of the spheroid's line source (for the
    prescribed-singularity test S1): flux Q_k and positions X_k."""
    K, c = spheroid_line_strength(a, b, U)
    x, w = np.polynomial.legendre.leggauss(n)
    xi = c * x
    Q = K * xi * c * w
    X = np.stack([xi, 0 * xi, -f + 0 * xi], 1)
    return Q, X
