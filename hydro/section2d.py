#!/usr/bin/env python3
"""
Two-dimensional section hydrodynamics of the planing strip plants -- step
P1b of the any-speed plant (sim/planing_anyspeed.py, switches fk_exact,
wagner_entry, lateral_phi0; learn/meta/ANYSPEED_PLAN_2026-10-05.md,
'P1b'). Two independent pieces, both pure functions of the section:

1. GRAVITY-FREE (phi = 0) ADDED MASSES OF A SECTION
The strip plants are 2D+t theories in the gravity-free limit: in each
cross-flow plane the free surface is a line on which the disturbance
potential vanishes (phi = 0, the high-frequency / high-speed condition).
v0 used two different bases for the section added mass: the vertical one
is Wagner's impact value (k_a pi/2 rho c^2 with the pile-up c = pi/2 d /
tan(beta)) also for a section oscillating about a fixed waterline, and the
lateral one is the rigid-lid plate (pi/2 rho d^2), a low-frequency value.
Here the exact phi = 0 problem is solved for the plant's own section -- a V
of deadrise beta from the keel to the chine half-beam yc, vertical walls
above it -- at any immersion and heel, by a boundary-element method:

  Green's third identity on the wetted contour, constant source and dipole
  strength per panel, the panel integrals of ln r and of its normal
  derivative in closed form, and the free surface z = 0 represented by the
  NEGATIVE image (phi odd in z, so phi = 0 on z = 0). Cosine-clustered
  panels on every edge (corners: keel, chines, waterline). This is the
  method of studies/principles_ledger_checks.py part B (validated there on
  circles and ellipses to 3e-5), written for three modes at once: sway,
  heave and roll about a reference point.

  added_mass_2d(Y, Z, ref)  the 3x3 matrix (sway, heave, roll about ref) of
                            a wetted contour in the earth frame (free
                            surface z = 0), per unit length and per unit
                            density.

Mode signs (right-handed x forward, y to port, z up; roll positive about +x,
lifting the +y side): the normal velocity of mode k is n . v_k with v_2 =
(1, 0), v_3 = (0, 1), v_4 = (-(z - z_r), y - y_r). The matrix is symmetric
and positive definite.

THE TABLE (phi0_table): one per deadrise. A section of chine height d_ch =
yc tan(beta) is a fixed shape in units of d_ch, so its coefficients depend
on the keel depth ratio r = d / d_ch (d the depth of the keel point below
the waterline, measured vertically -- the plant's d; negative when the boat
heels past the deadrise and only a chine is in) and on the heel phi, and
scale as d_ch^2 (translation), d_ch^3 (coupling with roll), d_ch^4 (roll).
Tabulated in the BODY frame -- body sway (along the heeled y axis), body
heave, roll about the keel -- because the plant's lateral equations are in
body axes: the section is heeled by phi, the problem solved in the earth
frame (free surface horizontal) and the translational modes rotated back.
Grid (xi_grid, phi_grid, wedge_phi_grid; constants below): in r the
kink-aligned coordinate xi -- at every heel the section's vertices (the
lowest one, the low chine or the keel, the high chine) reach the water at
xi = 0, 1, 2 (kinks), N_SEG points per segment, the third segment up to
r_hi (r_hi); the heel phi from 0 to PHI_MAX by DPHI, refined geometrically
towards the deadrise (PHI_NEAR, PHI_MIN, PHI_GEO; beyond PHI_MAX: clamped,
register); below the low chine at heels under the deadrise the bare wedge
by its exact scaling r^POW, its constants on their own fine heel grid;
deep sections (r > r_hi) on u = d_ch / d from U_LO to 1 / r_hi (values /
d^POW). Bilinear interpolation, derivatives of the interpolant (exact a.e.,
so the plant's Lagrangian forces are consistent with the interpolated
kinetic energy); in the first heel cell the even entries go in phi^2, so
that their heel derivative is zero at zero heel (Table._eval; S5a review). Built once per deadrise and kept on disk
(studies/_cache/anyspeed/sec2d/, keyed by beta, the grid and the sha1 of
this file); heel parity: a22, a33, a44 and a24 are even in phi, a23 and a34
odd (Table.lateral applies it, so the plant is exactly mirror symmetric).

The upright heave added mass for oscillation about a fixed waterline (the
vertical m_0 of switch wagner_entry) uses the exact wedge scaling below the
chine (m_0 = rho d^2 C33w, one solve) and a fine one-dimensional table above
it (Table.vertical): no interpolation error where a section is a pure wedge.

2. INCIDENT-WAVE (FROUDE-KRYLOV) INTEGRALS OF THE SECTION, IN CLOSED FORM
For a linear deep-water component, depth decay k, transverse wavenumber lam
(the body-y component of the wave vector), the incident pressure over the
upright section is rho g a e^{k z'} cos(psi + lam y), z' <= 0 the depth below
the local surface (Wheeler: the pressure is measured from the local surface,
so the total pressure vanishes there). All the section forces of switch
fk_exact (heave, sway, roll, the longitudinal force and its z lever) are
combinations of

    E     = int_A e^{k z'} cos(lam y) dA
    E_k   = dE/dk   = int_A z' e^{k z'} cos(lam y) dA
    E_lam = dE/dlam = -int_A y sin(lam y) e^{k z'} dA
    W     = int_{-c}^{c} cos(lam y) dy,  Wy = int_{-c}^{c} y sin(lam y) dy

(derivation in section_fk's docstring); section_fk gives them in closed
form, stable at k -> 0 and lam -> 0 (a double Taylor series below
|alpha h1| = SERIES_TOL), vectorised over stations x components.
"""
import hashlib
import os
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "studies", "_cache", "anyspeed", "sec2d")

# table grid and BEM resolution (part of the cache key): the kink-aligned
# xi grid (points per segment: the lowest vertex's wedge, up to the high
# chine, up to r_hi), the heel grid (DPHI_DEG to PHI_MAX_DEG, refined
# geometrically towards the deadrise: ratio PHI_GEO from PHI_NEAR down to
# PHI_MIN degrees away -- the coefficients of a nearly flat low side change
# fastest there; at phi = beta and zero immersion they jump, a flat side
# touching), and the bare-wedge constants' heel grid (ratio WEDGE_GEO down
# to WEDGE_MIN degrees below the deadrise)
N_SEG = (40, 60, 100)
PHI_MAX_DEG, DPHI_DEG = 45.0, 0.25
PHI_NEAR, PHI_MIN, PHI_GEO = 4.0, 0.02, 0.85
WEDGE_MIN, WEDGE_GEO, DPHI_WEDGE = 0.002, 0.99, 0.05
# panels per edge: N_BASE + N_LOG log(1 + length / d_ch), cosine-clustered
# (the error sits at the corners; 0.27 % of the largest coefficient against
# a 160-per-d_ch reference on the plant's sections, studies/
# anyspeed_p1b_check.py part sec2d)
N_BASE, N_LOG = 40, 20
WALL_TOP = 50.0         # wall height above the keel (in d_ch): above water
# the deep grid: u = d_ch / d from U_LO to 1 / R_HI (narrow, deep sections:
# the bow taper; u -> 0 is a plate), values / d^POW
U_LO, N_U = 0.01, 41
# the incident-wave integrals: below |alpha| h1 = SERIES_TOL the V part is
# its Taylor series to order SERIES_ORDER (error < 1e-12 there; the closed
# form's error above it, worst in E_k, eps / (|alpha| h1)^3 < 3e-11)
SERIES_TOL, SERIES_ORDER = 0.02, 4
# section_fk_grid (the plant's evaluation): second-order series below
# |alpha| h1 = GRID_TOL (its docstring)
GRID_TOL = 2e-3


# ===================================================================== BEM
def _angle(hp, X, Yl):
    """int_{-hp}^{hp} Yl / ((s - X)^2 + Yl^2) ds, continuous, 0 at Yl = 0
    inside the panel (principal value)."""
    small = np.abs(Yl) < 1e-14
    Ys = np.where(small, 1.0, Yl)
    v = np.arctan((hp - X) / Ys) - np.arctan((-hp - X) / Ys)
    return np.where(small, 0.0, v)


def _influence(m, cen, tt, nn, hh):
    """Single (S) and double (D) layer influence of constant panels (centre
    cen, unit tangent tt, unit normal nn, length hh; all (N, ...)) at the
    field points m (M, 2): S = (1/2 pi) int ln r ds, D = -(1/2 pi) int
    d(ln r)/dn_q ds (0 on the panel itself: principal value, the jump 1/2 is
    put in explicitly by the caller). In the panel's frame (X along it, Yl
    normal; ends at u1 = hp - X, u0 = -hp - X):

        int ln r ds = u1 ln r1 - u0 ln r0 - 2 hp + Yl theta,
        theta       = atan(u1/Yl) - atan(u0/Yl)    (the angle the panel
                      subtends, signed by Yl; arctan2 form, 0 where Yl = 0)
    """
    r = m[:, None, :] - cen[None, :, :]
    X = (r * tt[None]).sum(-1)
    Yl = (r * nn[None]).sum(-1)
    hp = 0.5 * hh[None, :]
    u1, u0 = hp - X, -hp - X
    Y2 = Yl * Yl
    r1, r0 = u1 * u1 + Y2, u0 * u0 + Y2
    with np.errstate(divide="ignore", invalid="ignore"):
        l1 = np.where(r1 > 0.0, 0.5 * np.log(np.where(r1 > 0, r1, 1.0)), 0.0)
        l0 = np.where(r0 > 0.0, 0.5 * np.log(np.where(r0 > 0, r0, 1.0)), 0.0)
    th = np.arctan2(Yl * (u1 - u0), Y2 + u1 * u0)
    th = np.where(np.abs(Yl) < 1e-14, 0.0, th)
    S = (u1 * l1 - u0 * l0 - 2.0 * hp + Yl * th) / (2.0 * np.pi)
    D = -th / (2.0 * np.pi)
    return S, D


def added_mass_2d(Y, Z, ref=(0.0, 0.0)):
    """phi = 0 added-mass matrix per unit length and density of the wetted
    contour (Y, Z) -- points in the earth frame, all Z <= 0, ordered from the
    +y waterline round the body to the -y waterline (the body to the left
    of the direction of travel... i.e. above the contour) -- for the modes
    (sway, heave, roll about ref = (y_r, z_r)). Returns the symmetric 3x3
    matrix (multiply by rho)."""
    P = np.stack([np.asarray(Y, float), np.asarray(Z, float)], -1)
    a, b = P[:-1], P[1:]
    h = np.linalg.norm(b - a, axis=1)
    keep = h > 0.0
    a, b, h = a[keep], b[keep], h[keep]
    t = (b - a) / h[:, None]
    n = np.stack([t[:, 1], -t[:, 0]], -1)        # into the body
    m = 0.5 * (a + b)
    N = h.size
    S0, D0 = _influence(m, m, t, n, h)
    R = np.array([1.0, -1.0])                    # image across z = 0
    Si, Di = _influence(m, m * R, t * R, n * R, h)
    S = S0 - Si                                  # phi = 0: negative image
    D = D0 - Di
    yr, zr = ref
    q = np.stack([n[:, 0], n[:, 1],
                  (m[:, 0] - yr) * n[:, 1] - (m[:, 1] - zr) * n[:, 0]], -1)
    A = 0.5 * np.eye(N) - D
    phi = np.linalg.solve(A, -S @ q)              # (N, 3)
    M = phi.T @ (q * h[:, None])
    return 0.5 * (M + M.T), M


def _cluster(n):
    """n + 1 points on [0, 1] clustered at both ends (cosine spacing)."""
    return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n + 1)))


def polyline(pts, n_per=None, scale=None, n_base=N_BASE, n_log=N_LOG):
    """Densify a polyline (list of (y, z)) with cosine-clustered panels on
    each edge: n_per per edge, or n_base + n_log log(1 + length / scale)."""
    Y, Z = [], []
    for (y0, z0), (y1, z1) in zip(pts[:-1], pts[1:]):
        ln = float(np.hypot(y1 - y0, z1 - z0))
        if ln <= 0.0:
            continue
        k = n_per if n_per else n_base + int(np.ceil(
            n_log * np.log1p(ln / scale)))
        s = _cluster(k)
        Y.append(y0 + (y1 - y0) * s[:-1])
        Z.append(z0 + (z1 - z0) * s[:-1])
    Y.append([pts[-1][0]])
    Z.append([pts[-1][1]])
    return np.concatenate(Y), np.concatenate(Z)


def section_body(yc, d_ch, top=None):
    """The plant's section in the body frame, keel at the origin, z up:
    the boundary from the top of the +y wall down to the +y chine, the keel,
    the -y chine and up the -y wall."""
    H = WALL_TOP * d_ch if top is None else top
    return [(yc, H), (yc, d_ch), (0.0, 0.0), (-yc, d_ch), (-yc, H)]


def wetted_contour(body_pts, d, phi):
    """The wetted part of a convex section boundary (body_pts, as
    section_body) heeled by phi (rotation about the keel point, positive
    lifting +y) with the keel d below the free surface, in the earth frame
    with the free surface at z = 0 (keel at (0, -d)). A list of vertices
    from the +y-side waterline round the bottom to the -y-side one, or None
    when dry. The section must be convex (the V-and-wall section is)."""
    c, s = np.cos(phi), np.sin(phi)
    P = [(y * c - z * s, y * s + z * c - d) for y, z in body_pts]
    out = []
    for (y0, z0), (y1, z1) in zip(P[:-1], P[1:]):
        if z0 <= 0.0 and z1 <= 0.0:
            if not out:
                out.append((y0, z0))
            out.append((y1, z1))
        elif z0 > 0.0 and z1 <= 0.0:
            t = z0 / (z0 - z1)
            out.append((y0 + t * (y1 - y0), 0.0))
            out.append((y1, z1))
        elif z0 <= 0.0 and z1 > 0.0:
            t = z0 / (z0 - z1)
            if not out:
                out.append((y0, z0))
            out.append((y0 + t * (y1 - y0), 0.0))
    if len(out) < 3:
        return None
    # drop repeated points (a vertex exactly on the surface)
    clean = [out[0]]
    for p in out[1:]:
        if abs(p[0] - clean[-1][0]) + abs(p[1] - clean[-1][1]) > 1e-15:
            clean.append(p)
    if len(clean) < 3 or abs(clean[0][1]) > 1e-12 or abs(clean[-1][1]) > 1e-12:
        return None
    return clean


def section_matrix(yc, d_ch, d, phi, n_base=N_BASE, n_log=N_LOG):
    """phi = 0 added masses per unit density of the plant's section (chine
    half-beam yc, chine height d_ch) with the keel d below the surface and
    heel phi, in the BODY frame: (a22, a23, a33, a24, a34, a44) for body
    sway, body heave and roll about the keel point. Zeros when dry."""
    pts = wetted_contour(section_body(yc, d_ch), d, phi)
    if pts is None:
        return np.zeros(6)
    Y, Z = polyline(pts, scale=d_ch, n_base=n_base, n_log=n_log)
    M, _ = added_mass_2d(Y, Z, ref=(0.0, -d))
    c, s = np.cos(phi), np.sin(phi)
    # body axes in the earth frame: y_b = (c, s), z_b = (-s, c)
    Rm = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    B = Rm @ M @ Rm.T
    return np.array([B[0, 0], B[0, 1], B[1, 1], B[0, 2], B[1, 2], B[2, 2]])


# ================================================================= table
NAMES = ("a22", "a23", "a33", "a24", "a34", "a44")
POW = np.array([2, 2, 2, 3, 3, 4])           # powers of the length scale
ODD = np.array([False, True, False, False, True, False])
_PAR = {1.0: np.ones(6), -1.0: np.where(ODD, -1.0, 1.0)}
_PARD = {1.0: np.ones(6), -1.0: np.where(ODD, 1.0, -1.0)}


def _powers(x, pw):
    """x[:, None] ** pw for the small integer powers pw (0..4)."""
    x2 = x * x
    tab = np.stack([np.ones_like(x), x, x2, x2 * x, x2 * x2], 1)
    return tab[:, pw]


_TABLE_CODE = ("_angle", "_influence", "added_mass_2d", "_cluster",
               "polyline", "section_body", "wetted_contour", "section_matrix",
               "kinks", "xi_grid", "phi_grid", "wedge_phi_grid", "r_hi",
               "_column", "_deep_column", "_wedge_column", "build_table")


_CODE_SHA1 = []


def _code_sha1():
    """sha1 of the source of the functions that compute the table (the
    table's cache key; editing the incident-wave integrals or the
    interpolation does not invalidate a table). Computed once per process,
    from the source as loaded (an edit of this file while a long run is
    under way must not change the key, nor make inspect read the edited
    file at the loaded functions' line numbers)."""
    if not _CODE_SHA1:
        import inspect
        g = globals()
        src = "".join(inspect.getsource(g[n]) for n in _TABLE_CODE)
        _CODE_SHA1.append(hashlib.sha1(src.encode()).hexdigest())
    return _CODE_SHA1[0]


def kinks(phi, tb):
    """The immersion ratios r = d / d_ch (keel depth over chine height) at
    which a vertex of the section heeled by phi >= 0 reaches the water, and
    their derivatives along phi: the lowest vertex k0 = min(0, z_l), the
    next k1 = max(0, z_l), the high chine k2 = z_h; z_l = cos(phi) -
    sin(phi) / tb and z_h = cos(phi) + sin(phi) / tb are the chines' heights
    above the keel in units of d_ch. The coefficients have kinks in r only
    there (below k1 the wetted part is a bare wedge)."""
    c, s = np.cos(phi), np.sin(phi)
    zl, zh = c - s / tb, c + s / tb
    dzl, dzh = -s - c / tb, -s + c / tb
    lowk = zl < 0.0
    k0 = np.where(lowk, zl, 0.0)
    k1 = np.where(lowk, 0.0, zl)
    dk0 = np.where(lowk, dzl, 0.0)
    dk1 = np.where(lowk, 0.0, dzl)
    return k0, k1, zh, dk0, dk1, dzh


def r_hi(tb):
    """The top of the shallow table: above the high chine at the largest
    heel by half a chine height, and at least 3."""
    return float(max(3.0, kinks(np.radians(PHI_MAX_DEG), tb)[2] + 0.5))


def xi_grid():
    """The table's grid in the kink-aligned coordinate xi: segment 0 (the
    lowest vertex's wedge, xi 0-1) clustered quadratically towards 0,
    segments 1 (xi 1-2, up to the high chine) and 2 (xi 2-3, to r_hi)
    uniform."""
    n0, n1, n2 = N_SEG
    s0 = (np.arange(n0 + 1) / n0) ** 2
    s1 = 1.0 + np.arange(1, n1 + 1) / n1
    s2 = 2.0 + np.arange(1, n2 + 1) / n2
    return np.concatenate([s0, s1, s2])


def phi_grid(beta_deg):
    """The heel grid (rad): 0 to PHI_MAX by DPHI, refined geometrically
    towards the deadrise on both sides (ratio PHI_GEO, from PHI_NEAR down to
    PHI_MIN degrees away), and the deadrise itself."""
    b = float(beta_deg)
    pts = set(np.round(np.arange(0.0, PHI_MAX_DEG + 1e-9, DPHI_DEG), 9))
    off = PHI_NEAR
    while off >= PHI_MIN:
        pts.add(round(b - off, 9))
        pts.add(round(b + off, 9))
        off *= PHI_GEO
    pts.add(round(b, 9))
    g = np.array(sorted(p for p in pts if 0.0 <= p <= PHI_MAX_DEG))
    return np.radians(g)


def wedge_phi_grid(beta_deg):
    """The heel grid of the bare-wedge constants (0 <= phi < beta): every
    DPHI_WEDGE, refined geometrically towards beta (ratio WEDGE_GEO down
    to WEDGE_MIN degrees)."""
    b = float(beta_deg)
    pts = set(np.round(np.arange(0.0, b - 1e-9, DPHI_WEDGE), 9))
    off = PHI_NEAR
    while off >= WEDGE_MIN:
        if b - off > 0.0:
            pts.add(round(b - off, 9))
        off *= WEDGE_GEO
    return np.radians(np.array(sorted(pts)))


def _column(args):
    """One heel column of the shallow table (a worker of build_table): the
    six coefficients (d_ch = 1) at the r of every xi of the grid (the kinks
    of this heel at xi = 1, 2). For a heel below the deadrise the first
    segment is the bare wedge, exactly self-similar about the keel: one
    solve, scaled as r^POW."""
    beta_deg, phi = args
    tb = np.tan(np.radians(beta_deg))
    yc = 1.0 / tb
    xi = xi_grid()
    k0, k1, k2 = (float(v) for v in kinks(phi, tb)[:3])
    rh = r_hi(tb)
    r = np.where(xi <= 1.0, k0 + xi * (k1 - k0),
                 np.where(xi <= 2.0, k1 + (xi - 1.0) * (k2 - k1),
                          k2 + (xi - 2.0) * (rh - k2)))
    out = np.zeros((xi.size, 6))
    wedge = k0 == 0.0 and k1 > 0.0
    ref = None
    if wedge:
        r_ref = 0.5 * k1
        ref = section_matrix(yc, 1.0, r_ref, phi)
    for i, ri in enumerate(r):
        if ri <= k0:
            continue
        if wedge and xi[i] <= 1.0:
            out[i] = ref * (ri / r_ref) ** POW
        else:
            out[i] = section_matrix(yc, 1.0, ri, phi)
    return out


def _deep_column(args):
    """One heel column of the deep table: keel depth d = 1, chine height
    d_ch = u (u on the deep grid); values / d^POW."""
    beta_deg, phi = args
    tb = np.tan(np.radians(beta_deg))
    rh = r_hi(tb)
    u = np.linspace(U_LO, 1.0 / rh, N_U)
    return np.array([section_matrix(ui / tb, ui, 1.0, phi) for ui in u])


def _wedge_column(args):
    """The bare wedge's constants a / r^POW (d_ch = 1) at heels below the
    deadrise."""
    beta_deg, phis = args
    tb = np.tan(np.radians(beta_deg))
    out = []
    for phi in phis:
        k1 = float(kinks(phi, tb)[1])
        r_ref = 0.5 * k1
        out.append(section_matrix(1.0 / tb, 1.0, r_ref, phi) / r_ref ** POW)
    return np.array(out)


def build_table(beta_deg, workers=1):
    """The coefficient arrays of one deadrise: T[ixi, iphi, 6] (shallow,
    d_ch = 1, on the kink-aligned xi grid), TD[iu, iphi, 6] (deep, d = 1),
    the bare-wedge constants CW[iphi_w, 6] (a / r^POW) and the upright
    wedge constants C33w, C22w (a / d^2 below the chine). Pure function of
    beta and the grids; workers > 1 spreads the heel columns over processes
    (the same floats)."""
    tb = np.tan(np.radians(beta_deg))
    ph = phi_grid(beta_deg)
    phw = wedge_phi_grid(beta_deg)
    jobs = [(float(beta_deg), float(p)) for p in ph]
    wchunks = [(float(beta_deg), c) for c in np.array_split(
        phw, max(1, min(workers, phw.size)))]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        # one BLAS thread per worker (spawned workers inherit the
        # environment; without it each worker's solver threads oversubscribe
        # the machine)
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(v, "1")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            cols = list(ex.map(_column, jobs))
            dcols = list(ex.map(_deep_column, jobs))
            wcols = list(ex.map(_wedge_column, wchunks))
    else:
        cols = [_column(j) for j in jobs]
        dcols = [_deep_column(j) for j in jobs]
        wcols = [_wedge_column(c) for c in wchunks]
    T = np.stack(cols, axis=1)
    TD = np.stack(dcols, axis=1)
    CW = np.concatenate(wcols, axis=0)
    T[:, 0, ODD] = 0.0                        # exact parity at phi = 0
    TD[:, 0, ODD] = 0.0
    CW[0, ODD] = 0.0
    wedge = section_matrix(1.0 / tb, 1.0, 0.5, 0.0)
    C33w = wedge[2] / 0.25
    C22w = wedge[0] / 0.25
    return dict(T=T, TD=TD, CW=CW, C33w=C33w, C22w=C22w, phi=ph, phiw=phw,
                xi=xi_grid(), beta_deg=float(beta_deg), r_hi=r_hi(tb))


def _key(beta_deg):
    spec = (f"{float(beta_deg):.12g}|{N_SEG}|{PHI_MAX_DEG}|{DPHI_DEG}|"
            f"{PHI_NEAR}|{PHI_MIN}|{PHI_GEO}|{WEDGE_MIN}|{WEDGE_GEO}|{DPHI_WEDGE}|"
            f"{U_LO}|{N_U}|{N_BASE}|{N_LOG}|{WALL_TOP}|" + _code_sha1())
    return hashlib.sha1(spec.encode()).hexdigest()[:16]


_TABLES = {}


def table_for(beta_deg, build=True, workers=1):
    """The Table of one deadrise: per process, then the disk cache, else
    built (single process: several minutes; python -m hydro.section2d
    --build BETA --workers N builds it in parallel) and saved atomically."""
    key = _key(beta_deg)
    tab = _TABLES.get(key)
    if tab is not None:
        return tab
    path = os.path.join(CACHE, f"phi0_b{float(beta_deg):.6f}_{key}.npz")
    data = None
    if os.path.exists(path):
        z = np.load(path)
        data = {k: z[k] for k in z.files}
        for k in ("beta_deg", "C33w", "C22w", "r_hi"):
            data[k] = float(data[k])
    elif build:
        data = build_table(beta_deg, workers=workers)
        os.makedirs(CACHE, exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp.npz"
        np.savez(tmp, **data)
        os.replace(tmp, path)
    else:
        raise FileNotFoundError(path)
    tab = Table(data)
    tab.path, tab.key = path, key
    _TABLES[key] = tab
    return tab


class Table:
    """Interpolation of the phi = 0 coefficients of one deadrise (module
    docstring, 'THE TABLE'). Shallow (r <= r_hi): bilinear in (xi, phi), xi
    the kink-aligned coordinate of r at the target heel (the vertices reach
    the water at xi = 0, 1, 2 for every heel, so no interpolation straddles
    a kink); below the low chine at a heel under the deadrise (the bare
    wedge) the exact scaling C_w(phi) r^POW with C_w on its own fine heel
    grid. Deep (r > r_hi): bilinear in (u = 1/r, phi), values / d^POW.
    Derivatives of the interpolant along d and phi (along phi at fixed r:
    the kinks' motion included), exact almost everywhere."""

    def __init__(self, data):
        self.T = np.asarray(data["T"], float)
        self.TD = np.asarray(data["TD"], float)
        self.CW = np.asarray(data["CW"], float)
        self.C33w = float(data["C33w"])
        self.C22w = float(data["C22w"])
        self.beta_deg = float(data["beta_deg"])
        self.tb = float(np.tan(np.radians(self.beta_deg)))
        self.beta = float(np.radians(self.beta_deg))
        self.r_hi = float(data["r_hi"])
        self.phi = np.asarray(data["phi"], float)
        self.phiw = np.asarray(data["phiw"], float)
        self.xi = np.asarray(data["xi"], float)
        self.u = np.linspace(U_LO, 1.0 / self.r_hi, N_U)
        self.du = float(self.u[1] - self.u[0])
        self.phi_max = float(self.phi[-1])
        n0 = N_SEG[0]
        self._n0 = n0

    @staticmethod
    def _bracket(grid, x):
        j = min(max(int(np.searchsorted(grid, x, side="right")) - 1, 0),
                grid.size - 2)
        h = grid[j + 1] - grid[j]
        return j, (x - grid[j]) / h, h

    def _xi_of(self, r, ap):
        """xi of r at heel ap (>= 0), dxi/dr and dxi/dphi at fixed r."""
        k0, k1, k2, dk0, dk1, dk2 = (float(v) for v in kinks(ap, self.tb))
        rh = self.r_hi
        L0, L1, L2 = k1 - k0, k2 - k1, rh - k2
        xi = np.empty_like(r)
        dr = np.empty_like(r)
        dp = np.empty_like(r)
        s0 = r <= k1
        s1 = (~s0) & (r <= k2)
        s2 = ~(s0 | s1)
        if L0 > 0.0:
            xi[s0] = (r[s0] - k0) / L0
            dr[s0] = 1.0 / L0
            dp[s0] = (-dk0 * L0 - (r[s0] - k0) * (dk1 - dk0)) / (L0 * L0)
        else:
            xi[s0], dr[s0], dp[s0] = 1.0, 0.0, 0.0
        if L1 > 0.0:
            xi[s1] = 1.0 + (r[s1] - k1) / L1
            dr[s1] = 1.0 / L1
            dp[s1] = (-dk1 * L1 - (r[s1] - k1) * (dk2 - dk1)) / (L1 * L1)
        else:
            xi[s1], dr[s1], dp[s1] = 2.0, 0.0, 0.0
        xi[s2] = 2.0 + (r[s2] - k2) / L2
        dr[s2] = 1.0 / L2
        dp[s2] = (-dk2 * L2 - (r[s2] - k2) * (-dk2)) / (L2 * L2)
        return xi, dr, dp, k0, k1

    def _shallow(self, r, ap, cols):
        """(F, dF/dr, dF/dphi) (n, k) of the shallow table's columns cols at
        r (n,) and heel ap: bilinear in (xi, phi), dry below the lowest
        vertex; the bare wedge by its exact scaling."""
        jp, fp, hph = self._bracket(self.phi, ap)
        xi, dxr, dxp, k0, k1 = self._xi_of(r, ap)
        g = self.xi
        x = np.minimum(np.maximum(xi, 0.0), g[-1])
        i = np.searchsorted(g, x, side="right") - 1
        i = np.minimum(np.maximum(i, 0), g.size - 2)
        hx = g[i + 1] - g[i]
        f = ((x - g[i]) / hx)[:, None]
        T = self.T
        a00, a10 = T[i, jp][:, cols], T[i + 1, jp][:, cols]
        a01, a11 = T[i, jp + 1][:, cols], T[i + 1, jp + 1][:, cols]
        b0 = a00 + f * (a10 - a00)
        b1 = a01 + f * (a11 - a01)
        F = b0 + fp * (b1 - b0)
        dFx = ((a10 - a00) + fp * ((a11 - a01) - (a10 - a00))) / hx[:, None]
        dFp = (b1 - b0) / hph
        dFr = dFx * dxr[:, None]
        dFphi = dFp + dFx * dxp[:, None]
        dry = r <= k0
        if dry.any():
            F[dry] = 0.0
            dFr[dry] = 0.0
            dFphi[dry] = 0.0
        if k0 == 0.0 and k1 > 0.0 and ap < self.beta:
            wd = (r > 0.0) & (r <= k1)
            if wd.any():
                # S5a (sim/planing_anyspeed, switch lateral_cross): the
                # kink row of the table (xi = 1, the low chine reaching the
                # water) interpolated in heel exactly as the region above
                # it, scaled by the bare wedge's exact self-similarity (r /
                # k1)^POW, so the interpolant is continuous at r = k1. The
                # P1b form scaled the wedge constants of their own finer
                # heel grid (CW), which match the kink row only on the
                # table's heel grid lines: between them it jumped at r = k1
                # by up to 2 % (a34 at 0.1 deg heel; 1e-4 at 3 deg), and a
                # station crossing r = k1 changed the lateral kinetic
                # energy discontinuously (check L4 of plan 'S5a'). At phi =
                # 0 the two forms give the same VALUES and r-derivatives
                # (the kink row there is the same solve as CW[0], scaled by
                # exact powers of 2), but not the same heel derivative: the
                # new one includes the kink's motion (on scarab195 the summed
                # dF/dphi at zero heel moved by 6.4 against a scale of 95.8,
                # on planing_b by 26.7 against 241.6), and the jump the P1b
                # form had between heel grid lines, up to 2.5 % (scarab195)
                # and 3.7 % (planing_b) of the coefficient scale over heels
                # 0.05-14 deg, is gone (S5a review, plan 'S5a：复查后的修正').
                # Since that review the even entries' heel derivative at
                # zero heel is exactly zero (_eval: phi^2 in the first heel
                # cell).
                n0 = self._n0
                Tk0, Tk1 = self.T[n0, jp][cols], self.T[n0, jp + 1][cols]
                Fk = Tk0 + fp * (Tk1 - Tk0)
                dFk = (Tk1 - Tk0) / hph
                dk1 = float(kinks(ap, self.tb)[4])
                rw = r[wd]
                pw = POW[cols]
                qw = rw / k1
                qp = _powers(qw, pw)
                F[wd] = Fk * qp
                dFr[wd] = Fk * pw * _powers(qw, np.maximum(pw - 1, 0)) / k1
                dFphi[wd] = dFk * qp - Fk * qp * pw * (dk1 / k1)
        return F, dFr, dFphi

    def _deep(self, d, d_ch, ap, cols):
        """(a, da/dd, da/dd_ch, da/dphi) / rho (n, k) of the deep table:
        a = d^POW G(u, phi), u = d_ch / d."""
        jp, fp, hph = self._bracket(self.phi, ap)
        uu = d_ch / d
        xu = np.minimum(np.maximum((uu - U_LO) / self.du, 0.0),
                        self.n_u - 1.000001)
        iu = xu.astype(np.intp)
        fu = (xu - iu)[:, None]
        T = self.TD
        a00, a10 = T[iu, jp][:, cols], T[iu + 1, jp][:, cols]
        a01, a11 = T[iu, jp + 1][:, cols], T[iu + 1, jp + 1][:, cols]
        b0 = a00 + fu * (a10 - a00)
        b1 = a01 + fu * (a11 - a01)
        Gd = b0 + fp * (b1 - b0)
        gu = ((a10 - a00) + fp * ((a11 - a01) - (a10 - a00))) / self.du
        gu[uu <= U_LO] = 0.0
        gp = (b1 - b0) / hph
        pw = POW[cols]
        sd = _powers(d, pw)
        a = sd * Gd
        dad = sd / d[:, None] * (pw * Gd - uu[:, None] * gu)
        dac = sd / d[:, None] * gu
        return a, dad, dac, sd * gp

    n_u = N_U

    def lateral(self, d, yc, phi):
        """(C, dC/dd, dC/dphi) with C (n, 6) = (a22, a23, a33, a24, a34,
        a44) / rho in the body frame (body sway, body heave, roll about the
        keel point), for keel depths d (n,), chine half-beams yc (n,), one
        heel phi (rad; |phi| clamped at PHI_MAX, its derivative 0 beyond).
        Parity in phi applied (a23, a34 odd)."""
        return self._eval(np.asarray(d, float), np.asarray(yc, float), phi,
                          np.arange(6))

    def _eval(self, d, yc, phi, cols):
        """(C, dC/dd, dC/dphi) at heel phi with the parity applied. The
        even entries (a22, a33, a24, a44) are even functions of the heel, so
        their heel derivative vanishes at phi = 0; the table is bilinear in
        |phi|, whose one-sided slope the parity flips at phi = 0 (a jump
        there: S5a review, issue E). Inside the first heel cell (|phi| <
        phi_1, the first heel grid value) the even entries are therefore
        interpolated in phi^2 between their values at 0 and at phi_1 (at the
        same keel depth): F = F(0) + (F(phi_1) - F(0)) (phi / phi_1)^2,
        dF/dphi = 2 (F(phi_1) - F(0)) phi / phi_1^2 -- zero at phi = 0,
        continuous in value at phi_1 (its slope has a kink there, as at every
        heel grid line of a bilinear table). The odd entries (a23, a34) are
        linear through 0 already (their parity keeps the slope). At phi = 0
        the values and the d-derivatives are the table's own, bit for bit."""
        sg = -1.0 if phi < 0.0 else 1.0
        clamped = abs(phi) > self.phi_max
        ap = min(abs(phi), self.phi_max)
        C, dCd, dCp = self._eval_at(d, yc, ap, cols)
        if clamped:
            dCp[:] = 0.0
        p1 = float(self.phi[1])
        if ap < p1:
            ev = ~ODD[cols]
            if ev.any():
                if ap == 0.0:
                    dCp[:, ev] = 0.0
                else:
                    C0, dCd0, _ = self._eval_at(d, yc, 0.0, cols)
                    C1, dCd1, _ = self._eval_at(d, yc, p1, cols)
                    w = (ap / p1) ** 2
                    C[:, ev] = C0[:, ev] + (C1[:, ev] - C0[:, ev]) * w
                    dCd[:, ev] = dCd0[:, ev] + (dCd1[:, ev] - dCd0[:, ev]) * w
                    dCp[:, ev] = 2.0 * (C1[:, ev] - C0[:, ev]) * (ap / (p1
                                                                      * p1))
        par = _PAR[sg][cols]
        return C * par, dCd * par, dCp * _PARD[sg][cols]

    def _eval_at(self, d, yc, ap, cols):
        """(C, dC/dd, dC/dphi) at the heel ap >= 0 (<= PHI_MAX), the table's
        bilinear interpolant, without the parity."""
        d_ch = yc * self.tb
        r = d / d_ch
        pw = POW[cols]
        sc = _powers(d_ch, pw)
        C = np.zeros((d.size, cols.size))
        dCd = np.zeros_like(C)
        dCp = np.zeros_like(C)
        sh = r <= self.r_hi
        if sh.any():
            j = np.nonzero(sh)[0] if not sh.all() else slice(None)
            F, dFr, dFp = self._shallow(r[j], ap, cols)
            scj = sc[j]
            C[j] = scj * F
            dCd[j] = scj / d_ch[j][:, None] * dFr
            dCp[j] = scj * dFp
        if not sh.all():
            j = np.nonzero(~sh)[0]
            a, dad, dac, dap = self._deep(d[j], d_ch[j], ap, cols)
            C[j], dCd[j], dCp[j] = a, dad, dap
        return C, dCd, dCp

    # ---------------------------------------------------- vertical (upright)
    def _vspline(self):
        """The upright heave column above the chine as cubic splines, C1
        throughout (the bilinear table made dm0/dd and dm0/dyc piecewise
        constant, jumping at every table cell: P1b review): F(r) = a33 /
        d_ch^2 in r = d / d_ch over [1, r_hi] through the shallow table's
        upright points, not-a-knot (the one physical kink, the chine at r =
        1, is its end; below it the exact wedge scaling), and the deep
        table's G(u) = a33 / d^2 in u = d_ch / d = 1/r over [U_LO, 1/r_hi]
        (its own uniform grid), clamped at u = 1/r_hi to the shallow
        spline's slope there (G' = 2 u F - F'), so value and slope are
        continuous at r_hi. Beyond r = 1/U_LO (a plate) G = G(U_LO), as the
        deep table. Knots and coefficients, once per table."""
        sp = self.__dict__.get("_vsp")
        if sp is None:
            from scipy.interpolate import CubicSpline
            ph0 = 0
            n0, n1 = N_SEG[0], N_SEG[1]
            # upright: k1 = k2 = 1 (both chines at the same height), so the
            # third segment (xi 2..3) spans r 1..r_hi
            xi = self.xi[n0 + n1:]
            r_sh = 1.0 + (xi - 2.0) * (self.r_hi - 1.0)
            F_sh = self.T[n0 + n1:, ph0, 2].copy()
            F_sh[0] = self.C33w                  # the wedge's value at r = 1
            cs = CubicSpline(r_sh, F_sh, bc_type="not-a-knot")
            # the deep part as F = G / u^2 (bounded: it tends to the narrow
            # section's constant as u -> 0) in u, its own uniform grid. The
            # table's narrowest columns hold zeros where the BEM did not
            # resolve the section (u 0.010 and 0.018 for 20 deg: d / d_ch >
            # 56, only at the stem tip); they are left out and F is
            # continued linearly below the first resolved u (register P14)
            uh = 1.0 / self.r_hi
            Fh, dFh = float(cs(self.r_hi)), float(cs(self.r_hi, 1))
            u = self.u
            G = self.TD[:, ph0, 2]
            ok = G > 0.0
            ok[-1] = True
            Fu = np.where(ok, G / (u * u), 0.0)
            Fu[-1] = Fh                           # the same section at r_hi
            # dF/du = dF/dr dr/du = -F'(r) / u^2 at u = 1/r_hi
            cd = CubicSpline(u[ok], Fu[ok], bc_type=("not-a-knot",
                                                     (1, -dFh / (uh * uh))))
            u0 = float(u[ok][0])
            sp = self._vsp = SimpleNamespace(
                x=cs.x, c=cs.c, ux=cd.x, uc=cd.c, u0=u0, F0=float(cd(u0)),
                dF0=float(cd(u0, 1)))
        return sp

    def vertical(self, d, yc):
        """(m0, dm0/dd, dm0/dyc) / rho of the upright section oscillating
        about a fixed waterline, keel depth d (array), chine half-beam yc:
        rho d^2 C33w below the chine (the bare wedge: exact scaling), the
        upright column's cubic spline above it (_vspline: C1 in d and yc
        everywhere but at the chine), 0 when dry."""
        d = np.asarray(d, float)
        d_ch = yc * self.tb
        dp = np.maximum(d, 0.0)
        m = self.C33w * dp * dp
        dm = 2.0 * self.C33w * dp
        dmy = np.zeros_like(dp)
        above = dp > d_ch
        if above.any():
            j = np.nonzero(above)[0]
            dj, cj = dp[j], d_ch[j]
            r = dj / cj
            sp = self._vspline()
            a = np.empty(j.size)
            dad = np.empty(j.size)
            dac = np.empty(j.size)
            sh = r <= self.r_hi
            if sh.any():
                rs, cs_ = r[sh], cj[sh]
                i = np.clip(np.searchsorted(sp.x, rs, side="right") - 1, 0,
                            sp.x.size - 2)
                t = rs - sp.x[i]
                c0, c1, c2, c3 = sp.c[:, i]
                F = ((c0 * t + c1) * t + c2) * t + c3
                dFr = (3.0 * c0 * t + 2.0 * c1) * t + c2
                a[sh] = cs_ * cs_ * F
                dad[sh] = cs_ * dFr
                # d/d(d_ch) at fixed d: d_ch^2 F(r) with r = d / d_ch
                dac[sh] = 2.0 * cs_ * F - rs * cs_ * dFr
            if not sh.all():
                k = ~sh
                dk, ck = dj[k], cj[k]
                u = ck / dk
                uu = np.maximum(u, sp.u0)
                i = np.clip(np.searchsorted(sp.ux, uu, side="right") - 1, 0,
                            sp.ux.size - 2)
                t = uu - sp.ux[i]
                c0, c1, c2, c3 = sp.uc[:, i]
                Fd = ((c0 * t + c1) * t + c2) * t + c3
                Fp = (3.0 * c0 * t + 2.0 * c1) * t + c2
                lo = u < sp.u0
                if lo.any():
                    Fd = np.where(lo, sp.F0 + sp.dF0 * (u - sp.u0), Fd)
                    Fp = np.where(lo, sp.dF0, Fp)
                # a = d_ch^2 F(u), u = d_ch / d
                a[k] = ck * ck * Fd
                dad[k] = -ck * u * u * Fp
                dac[k] = 2.0 * ck * Fd + ck * u * Fp
            m[j] = a
            dm[j] = dad
            dmy[j] = dac * self.tb
        return m, dm, dmy


# ================================================== incident-wave integrals
def _T(x):
    """(sin x / x - cos x) / x = (sin x - x cos x) / x^2, stable."""
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-2
    xs = np.where(small, 1.0, x)
    big = (np.sin(xs) - xs * np.cos(xs)) / (xs * xs)
    x2 = x * x
    ser = x * (1.0 / 3.0 - x2 / 30.0 + x2 * x2 / 840.0)
    return np.where(small, ser, big)


def section_fk(k, lam, d, yc, tb):
    """Closed-form incident-wave integrals of the upright V-and-wall section
    (keel depth d below the local surface, chine half-beam yc, tan(deadrise)
    tb) for depth decay k >= 0 and transverse wavenumber lam; arrays that
    broadcast (stations x components). Returns (E, E_k, E_lam, W, Wy, c):

        E     = int_A e^{k z'} cos(lam y) dA
        E_k   = int_A z' e^{k z'} cos(lam y) dA           (= dE/dk)
        E_lam = -int_A y sin(lam y) e^{k z'} dA            (= dE/dlam)
        W     = int_{-c}^{c} cos(lam y) dy
        Wy    = int_{-c}^{c} y sin(lam y) dy
        c     the waterline half-width (min(d/tb, yc), 0 when dry)

    DERIVATION. The V part (height s above the keel, 0 <= s <= h1 =
    min(d, d_ch), half-width s/tb, z' = s - d): with q = lam/tb,
        int_{-w}^{w} cos(lam y) dy = 2 sin(q s)/lam, and
        int_0^h1 e^{ks} sin(qs) ds = [e^{k h1}(k sin(q h1) - q cos(q h1)) + q]
                                     / (k^2 + q^2),
    so with x = q h1 = lam c_V, S0 = sin x / x, C0 = cos x, P = e^{-k Delta}
    (Delta = d - h1, the wall depth), K = e^{-k d}, D = k^2 + q^2:
        E_V = (2/tb) N / D,   N = P (k h1 S0 - C0) + K.
    The wall part (|y| <= yc from z' = -Delta to 0): E_W = 2 yc S0 g(k),
    g(k) = (1 - P)/k (= Delta at k = 0); on a wall station c_V = yc, so the
    same S0. E_k and E_lam are the derivatives of these expressions:
        dN/dk = -Delta P (k h1 S0 - C0) + P h1 S0 - d K,
        dN/dq = P h1 (x S0 - k h1 T(x)),   T(x) = (S0 - C0)/x,
        dE_V/dk = (2/tb)(dN/dk D - 2k N)/D^2,
        dE_V/dlam = (1/tb) (2/tb)(dN/dq D - 2q N)/D^2,
        dE_W/dk = 2 yc S0 g'(k),  g' = -Delta^2 phi2(k Delta),
                  phi2(u) = (1 - (1 + u) e^{-u}) / u^2,
        dE_W/dlam = -2 yc^2 T(x) g(k).
    W = 2 c S0 and Wy = 2 c^2 T(x) (x = lam c; c = c_V). Where |alpha| h1
    < SERIES_TOL (alpha = k + i q) the V part is its Taylor series,
        E_V = (2/tb) P [h1^2/2 - k h1^3/6 + (k^2 - q^2) h1^4/24 + ...],
    with its derivatives -- the double series in the code (relative error
    below 1e-12 there; the closed form's cancellation error, worst in E_k,
    eps / (|alpha| h1)^3, is below 3e-11 above it)."""
    k = np.asarray(k, float)
    lam = np.asarray(lam, float)
    d = np.asarray(d, float)
    yc = np.asarray(yc, float)
    dp = np.maximum(d, 0.0)
    d_ch = yc * tb
    h1 = np.minimum(dp, d_ch)
    Dl = dp - h1                                   # wall depth
    cV = h1 / tb
    q = lam / tb
    x = lam * cV
    xs = np.where(x == 0.0, 1.0, x)
    S0 = np.where(x == 0.0, 1.0, np.sin(xs) / xs)
    C0 = np.cos(x)
    Tx = _T(x)
    P = np.exp(-k * Dl)
    K = np.exp(-k * dp)
    D = k * k + q * q
    N = P * (k * h1 * S0 - C0) + K
    dNk = -Dl * P * (k * h1 * S0 - C0) + P * h1 * S0 - dp * K
    dNq = P * h1 * (x * S0 - k * h1 * Tx)
    with np.errstate(divide="ignore", invalid="ignore"):
        Dsafe = np.where(D > 0.0, D, 1.0)
        EV = (2.0 / tb) * N / Dsafe
        EVk = (2.0 / tb) * (dNk * Dsafe - 2.0 * k * N) / (Dsafe * Dsafe)
        EVq = (2.0 / tb) * (dNq * Dsafe - 2.0 * q * N) / (Dsafe * Dsafe)
    # series where |alpha| h1 is small (the closed form's cancellation
    # grows like eps / (|alpha| h1)^3 in its k-derivative): with kappa = k h1
    # and Q = q h1 = x,
    #   E_V = (2/tb) P h1^2 sum_{a,b>=0} (-kappa)^a (-Q^2)^b / (a + 2b + 2)!
    # (from int_0^1 t e^{-kappa(1-t)} sinc(Q t) dt, term by term with the
    # Beta integral), truncated at a + 2b <= SERIES_ORDER
    small = D * h1 * h1 < SERIES_TOL ** 2
    if np.any(small):
        shp = np.broadcast(k, lam, d, yc).shape
        iw = np.broadcast_to(small, shp)
        bk = lambda arr: np.broadcast_to(arr, shp)[iw]       # noqa: E731
        kap = bk(k * h1)
        xs_ = bk(x)
        Q2 = xs_ * xs_
        sE = np.zeros(kap.shape)
        sEk = np.zeros_like(sE)
        sEq = np.zeros_like(sE)
        from math import factorial
        for a in range(SERIES_ORDER + 1):
            for b in range((SERIES_ORDER - a) // 2 + 1):
                c = (-1.0) ** (a + b) / factorial(a + 2 * b + 2)
                ka = kap ** a
                qb = Q2 ** b
                sE = sE + c * ka * qb
                if a:
                    sEk = sEk + c * a * kap ** (a - 1) * qb
                if b:
                    sEq = sEq + c * ka * 2 * b * xs_ ** (2 * b - 1)
        h1w, Dlw = bk(h1), bk(Dl)
        pre = (2.0 / tb) * bk(P) * h1w * h1w
        EV = np.array(np.broadcast_to(EV, shp))
        EVk = np.array(np.broadcast_to(EVk, shp))
        EVq = np.array(np.broadcast_to(EVq, shp))
        EV[iw] = pre * sE
        EVk[iw] = -Dlw * pre * sE + pre * h1w * sEk
        EVq[iw] = pre * h1w * sEq
    # wall part
    kD = k * Dl
    with np.errstate(divide="ignore", invalid="ignore"):
        ksafe = np.where(k > 0.0, k, 1.0)
        g = np.where(k > 0.0, -np.expm1(-kD) / ksafe, Dl)
        usafe = np.where(kD > 1e-3, kD, 1.0)
        ph2 = np.where(kD > 1e-3,
                       (1.0 - (1.0 + usafe) * np.exp(-usafe)) / (usafe * usafe),
                       0.5 - kD / 3.0 + kD * kD / 8.0)
    gk = -Dl * Dl * ph2
    EW = 2.0 * yc * S0 * g
    EWk = 2.0 * yc * S0 * gk
    EWl = -2.0 * yc * yc * Tx * g
    wall = Dl > 0.0
    EW = np.where(wall, EW, 0.0)
    EWk = np.where(wall, EWk, 0.0)
    EWl = np.where(wall, EWl, 0.0)
    E = EV + EW
    Ek = EVk + EWk
    El = EVq / tb + EWl
    W = 2.0 * cV * S0
    Wy = 2.0 * cV * cV * Tx
    dry = d <= 0.0
    z = np.zeros_like(E)
    return (np.where(dry, z, E), np.where(dry, z, Ek), np.where(dry, z, El),
            np.where(dry, z, W), np.where(dry, z, Wy),
            np.where(dry, 0.0 * cV, cV))


def section_fk_grid(k, lam, d, yc, tb, ku=None, kinv=None):
    """section_fk for a grid of sections (rows: keel depths d > 0 and chine
    half-beams yc, (n,)) times wave components (columns: k > 0, lam, (M,)),
    arranged for speed (the plant's derivative evaluates it twice per
    step): the exponentials per distinct wavenumber (ku, kinv: k =
    ku[kinv]), lam = 0 per column, no where() over the grid. Returns (E,
    E_k, E_lam, W, Wy) (n, M) and c (n,).

    The same expressions as section_fk, with two simplifications whose
    error is absolute and negligible: S0 and T by their series on whole
    columns with |lam| c_V,max < 1e-3, by the closed form elsewhere (T's
    cancellation error there, eps/x on a row with tiny c_V, is absolute
    and below 1e-12 of the scale), and the V part's Taylor series (second
    order) only where |alpha| h1 < GRID_TOL
    = 2e-3 (the closed form's error above it, eps / (|alpha| h1)^3, is
    relative to E_k of a section of depth h1 < 2e-3 / |alpha|: absolute
    below 1e-12 of the scale). Agreement with section_fk: studies/
    anyspeed_p1b_check.py part fk2d."""
    if ku is None:
        ku, kinv = np.unique(k, return_inverse=True)
    d_ch = yc * tb
    h1 = np.minimum(d, d_ch)
    Dl = d - h1
    cV = h1 / tb
    q = lam / tb
    D = k * k + q * q
    iD = 1.0 / D
    # the columns with lam ~ 0 (|lam| c_V,max < 1e-3: head and following
    # waves, and near them) take S0 and T from their series
    zc = np.abs(lam) * float(cV.max(initial=0.0)) < 1e-3
    ilam = 1.0 / np.where(zc, 1.0, lam)
    ilam[zc] = 0.0
    x = cV[:, None] * lam
    xinv = (1.0 / cV)[:, None] * ilam
    sx, cx = np.sin(x), np.cos(x)
    S0 = sx * xinv
    T = (S0 - cx) * xinv
    if zc.any():
        xz = x[:, zc]
        x2 = xz * xz
        S0[:, zc] = 1.0 - x2 / 6.0 + x2 * x2 / 120.0
        T[:, zc] = xz * (1.0 / 3.0 - x2 / 30.0 + x2 * x2 / 840.0)
    K = np.exp(-d[:, None] * ku)[:, kinv]
    P = np.exp(-Dl[:, None] * ku)[:, kinv]
    kh1 = h1[:, None] * k
    A1 = kh1 * S0 - cx
    N = P * A1 + K
    Nk = P * (h1[:, None] * S0 - Dl[:, None] * A1) - d[:, None] * K
    Nq = (P * h1[:, None]) * (x * S0 - kh1 * T)
    c2 = 2.0 / tb
    NiD = N * iD
    E = c2 * NiD
    Ek = c2 * (Nk - (2.0 * k) * NiD) * iD
    El = (c2 / tb) * (Nq - (2.0 * q) * NiD) * iD
    if D.min() * float(np.min(h1 * h1)) < GRID_TOL ** 2:
        small = (h1 * h1)[:, None] * D < GRID_TOL ** 2
        if small.any():
            kap = kh1[small]
            xs_ = x[small]
            h1w = np.broadcast_to(h1[:, None], x.shape)[small]
            Dlw = np.broadcast_to(Dl[:, None], x.shape)[small]
            base = 0.5 - kap / 6.0 + (kap * kap - xs_ * xs_) / 24.0
            pre = c2 * P[small] * h1w * h1w
            E[small] = pre * base
            Ek[small] = pre * (-Dlw * base + h1w * (-1.0 / 6.0 + kap / 12.0))
            El[small] = (pre / tb) * h1w * (-xs_ / 12.0)
    if Dl.max() > 0.0:
        # the wall part: g = (1 - P)/k (0 where there is no wall: P = 1),
        # g' = (Delta P - g)/k
        ik = 1.0 / k
        g = (1.0 - P) * ik
        gk = (Dl[:, None] * P - g) * ik
        ys = (2.0 * yc)[:, None] * S0
        E += ys * g
        Ek += ys * gk
        El -= (2.0 * yc * yc)[:, None] * T * g
    W = (2.0 * cV)[:, None] * S0
    Wy = (2.0 * cV * cV)[:, None] * T
    return E, Ek, El, W, Wy, cV

if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser(description="build the phi = 0 section "
                                 "tables (cached on disk)")
    ap.add_argument("--build", default="20",
                    help="deadrise angles in degrees, comma separated")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    for b in a.build.split(","):
        t0 = time.time()
        tab = table_for(float(b), workers=a.workers)
        print(f"beta {b}: {tab.path} ({time.time() - t0:.1f} s)")
