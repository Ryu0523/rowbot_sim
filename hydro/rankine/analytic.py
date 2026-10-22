"""
Closed-form and published reference values used by the verification.

Each function cites its source; nothing here is fitted.
"""
import numpy as np


def spheroid_lamb(a, b):
    """Prolate spheroid, semi-axes a (along x) > b: added-mass coefficients
    (Lamb 1932, Hydrodynamics, 6th ed., arts. 114-115 and 373; Newman 1977,
    Marine Hydrodynamics, table 4.2):
        e = sqrt(1 - b^2/a^2)
        alpha0 = 2 (1 - e^2)/e^3 (1/2 ln((1+e)/(1-e)) - e)
        beta0  = 1/e^2 - (1 - e^2)/(2 e^3) ln((1+e)/(1-e))
        k1 = alpha0/(2 - alpha0)      (axial translation, A11 = k1 rho V)
        k2 = beta0/(2 - beta0)        (lateral translation, A22 = A33 = k2 rho V)
        k' = e^4 (beta0 - alpha0) / ((2 - e^2) [2 e^2 - (2 - e^2)(beta0 - alpha0)])
             (rotation about a transverse axis, A55 = k' I, I = rho V (a^2+b^2)/5)
    Returns dict with k1, k2, kp, V, I_over_rho."""
    e = np.sqrt(1.0 - (b / a) ** 2)
    lg = np.log((1 + e) / (1 - e))
    alpha0 = 2 * (1 - e ** 2) / e ** 3 * (0.5 * lg - e)
    beta0 = 1 / e ** 2 - (1 - e ** 2) / (2 * e ** 3) * lg
    k1 = alpha0 / (2 - alpha0)
    k2 = beta0 / (2 - beta0)
    d = beta0 - alpha0
    kp = e ** 4 * d / ((2 - e ** 2) * (2 * e ** 2 - (2 - e ** 2) * d))
    V = 4.0 / 3.0 * np.pi * a * b * b
    return dict(k1=k1, k2=k2, kp=kp, V=V, I_over_rho=V * (a * a + b * b) / 5,
                e=e, alpha0=alpha0, beta0=beta0)


def spheroid_double_body(a, b, U, P):
    """EXACT potential flow past a fixed prolate spheroid (semi-axes a along x > b = c, centre at the origin) in
    the stream -U x_hat -- the rigid-lid double-body flow of the half-spheroid hull hull.half_spheroid(a, b) (the
    hull and its image in z = 0 form the full spheroid; added after the B6 review, stage note 12.2 item 2, for the
    operator-consistency rows DXa). Prolate spheroidal coordinates (Lamb 1932, art. 105; Newman 1977, sec. 4.13):
    focal half-distance k = sqrt(a^2 - b^2), r1, r2 the distances to the foci (+-k, 0, 0),
        zeta = (r1 + r2)/(2k) >= 1,  mu = (r2 - r1)/(2k) in [-1, 1]  (x = k mu zeta),  body zeta0 = a/k,
        Phi = -U x + B mu Q1(zeta),  Q1(zeta) = (zeta/2) ln((zeta + 1)/(zeta - 1)) - 1,
        B = U k / Q1'(zeta0)        (dPhi/dzeta = 0 on the body: -U k mu + B mu Q1'(zeta0) = 0),
        Q1' = (1/2) ln((zeta + 1)/(zeta - 1)) - zeta/(zeta^2 - 1),  Q1'' = 2/(zeta^2 - 1)^2.
    W = grad Phi from grad r_i = (x -+ k, y, z)/r_i. On z = 0: dmu/dz = dzeta/dz = 0 and d^2 r_i/dz^2 = 1/r_i, so
        Phi_zz = B [Q1 (1/r2 - 1/r1) + mu Q1' (1/r1 + 1/r2)] / (2k)   (z = 0 only).
    Far field: Phi + U x -> B k^2 x/(3 r^3) (dipole). P (n, 3) points outside the spheroid. Returns (W (n, 3),
    Phi_zz (n,) -- meaningful at z = 0 only, phi_pert (n,) = B mu Q1)."""
    P = np.asarray(P, float)
    k = np.sqrt(a * a - b * b)
    z0 = a / k

    def q1p(zz):
        return 0.5 * np.log((zz + 1) / (zz - 1)) - zz / (zz * zz - 1)
    Bc = U * k / q1p(z0)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    rho2 = y * y + z * z
    r1 = np.sqrt((x - k) ** 2 + rho2)
    r2 = np.sqrt((x + k) ** 2 + rho2)
    zeta = (r1 + r2) / (2 * k)
    mu = (r2 - r1) / (2 * k)
    Q1 = 0.5 * zeta * np.log((zeta + 1) / (zeta - 1)) - 1.0
    Q1p = q1p(zeta)
    g1 = np.stack([x - k, y, z], 1) / r1[:, None]
    g2 = np.stack([x + k, y, z], 1) / r2[:, None]
    gz = (g1 + g2) / (2 * k)
    gm = (g2 - g1) / (2 * k)
    W = Bc * (Q1[:, None] * gm + (mu * Q1p)[:, None] * gz)
    W[:, 0] -= U
    pzz = Bc * (Q1 * (1 / r2 - 1 / r1) + mu * Q1p * (1 / r1 + 1 / r2)) / (2 * k)
    return W, pzz, Bc * mu * Q1


# Hulme A. (1982) The wave forces acting on a floating hemisphere undergoing
# forced periodic oscillations. J. Fluid Mech. 121:443-463, heave table.
# Transcribed from the file comparison/HulmeResult.dat of the LHEEA repository
# github.com/LHEEA/Hulme_Heaving_Sphere (read 2026-10-06), which reproduces
# Hulme's table; columns ka, A33/(rho V), B33/(rho V omega), V = 2/3 pi a^3.
# The transcription is cross-checked against Capytaine in the verification
# (check Z2), so a typo would show up as a single-point outlier.
HULME_HEAVE = np.array([
    [0.0, 0.8310, 0.0000], [0.05, 0.8764, 0.1036], [0.1, 0.8627, 0.1816],
    [0.2, 0.7938, 0.2793], [0.3, 0.7157, 0.3254], [0.4, 0.6452, 0.3410],
    [0.5, 0.5861, 0.3391], [0.6, 0.5381, 0.3271], [0.7, 0.4999, 0.3098],
    [0.8, 0.4698, 0.2899], [0.9, 0.4464, 0.2691], [1.0, 0.4284, 0.2484],
    [1.2, 0.4047, 0.2096], [1.4, 0.3924, 0.1756], [1.6, 0.3871, 0.1469],
    [1.8, 0.3864, 0.1229], [2.0, 0.3884, 0.1031], [2.5, 0.3988, 0.0674],
    [3.0, 0.4111, 0.0452], [4.0, 0.4322, 0.0219], [5.0, 0.4471, 0.0116],
    [6.0, 0.4574, 0.0066], [7.0, 0.4647, 0.0040], [8.0, 0.4700, 0.0026],
    [9.0, 0.4740, 0.0017], [10.0, 0.4771, 0.0012]])
