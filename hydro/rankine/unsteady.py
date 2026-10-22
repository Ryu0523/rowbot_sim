"""
Forward-speed radiation and diffraction about the uniform stream (linear
Neumann-Kelvin, NK) as ONE linear time-invariant (LTI) system per speed and
attitude -- Track B step B5 (port design WAVEBEM_PORT_DESIGN.md P-2, P-4, P-5;
stage note RANKINE_STAGE1_2026-10-06.md section 10). Nothing is copied from
WaveBEM (licence rule L): WaveBEM has no linear forward-speed radiation solver;
the formulation follows Kring (1994), Nakos & Sclavounos (1990) and Ogilvie
(1964) as re-derived in the stage note.

1. Equations (body frame translating with the mean speed, x forward, y to port,
   z up, fluid z < 0, stream -U x_hat; normals out of the body / into the fluid).
   Perturbation potential phi, free-surface elevation zeta, beach nu(x, y) >= 0:
       zeta_t = U zeta_x + phi_z - nu zeta            on z = 0   (kinematic)
       phi_t  = U phi_x  - g zeta - nu phi            on z = 0   (dynamic)
       dphi/dn = sum_k [ xdot_k n_k + x_k m_k ]       on the mean hull S
   with the uniform-stream m-terms (Timman & Newman 1962; W = -U x_hat):
       (m_1, m_2, m_3) = 0,  (m_4, m_5, m_6) = -(n x W) = (0, U n_3, -U n_2).
   Linearised pressure p = -rho (phi_t - U phi_x) (hydrostatics separate).
   Modes: 1 surge, 3 heave, 5 pitch about x_ref (right-handed about +y, so a
   positive x_5 moves the bow DOWN; n_5 = (z - z_r) n_x - (x - x_r) n_z).

2. Representation (as shifted.py / piercing.py): Rankine sources and dipoles on
   the hull with the rigid-lid (z) and port/starboard (y) images, Rankine
   sources sigma = dphi/dn = -phi_z on the free-surface panels (their dipoles
   cancel with the lid image, the source doubles):
       (2 pi I - D_BB) phi_B + S_BF sigma = -S_BB phi_n
       4 pi phi_F = D_FB phi_B - S_FB phi_n - S_FF sigma
   Eliminating phi_B: phi_F = R_s sigma + R_n phi_n (Neumann-to-Dirichlet map).

3. LTI system. With N = diag(nu) - U D (D the discrete d/dx at fixed y, item 4):
       zeta' = -N zeta - sigma,   phi_F' = -N phi_F - g zeta,
       sigma = R_s^{-1} (phi_F - R_n phi_n(t)),
   i.e. y' = M y + b(t), y = (zeta, phi_F), one system for all frequencies. Its
   transfer function at e^{i omega t} is evaluated EXACTLY (no time
   discretisation) by eliminating zeta: with L = i omega + N,
       sigma = (1/g) L^2 phi_F   =>   [I - (1/g) L^2 R_s] sigma = (1/g) L^2 R_n phi_n,
   L^2 = N^2 - omega^2 + 2 i omega N (matrix products, so an x-dependent beach
   enters exactly as in the time domain). Precomputed real blocks R_s, N R_s,
   N^2 R_s; one complex dense LU per frequency (all modes and headings share it).
   omega -> infinity: phi_F = 0 (instantaneous problem, A_inf, B_inf, C_inf);
   omega = 0: the steady NK problem on the same grid (cross-check of B4).
   The time-domain form (RK4) is the same M (energy-balance check, cross-check
   of the transfer function).

4. Discrete d/dx (why upwind-biased). For a lattice mode the semi-discrete
   system gives lambda = -nu + U D^ +- i sqrt(g kappa_h) (kappa_h > 0 the
   discrete Dirichlet-to-Neumann symbol), so Re(lambda) = -nu + U Re(D^): any
   skew (central) D is neutral, but then the saw-tooth mode has D^ = 0, does not
   propagate and is never absorbed (an undamped interior mode); an
   upwind-biased D (upwind = against the stream) has U Re(D^) <= 0: every mode
   decays, the saw-tooth strongly. Third order (4 points): Re(D^) dx =
   -(1 - cos theta)^2/3, resolved waves damped by about (1/12)(k dx)^3 per
   radian of travel; fifth order (6 points, production, ORDER = 5):
   -(16/15) sin^6(theta/2) ~ -theta^6/60 (stage note 10.2 item 2; the third
   order damped the steady wake too much, exploration B5-E7). Curvilinear (band) grid: d/dx = i_x d/di + j_x d/dj
   with i_x = y_j/J, j_x = -y_i/J, each index derivative upwind-biased by the
   sign of its contravariant advection velocity (U i_x, U j_x); inflow ghosts
   zero (undisturbed stream), centre-plane ghosts mirrored, one-sided
   second-order formulas where the stencil leaves the grid elsewhere.
   Hull wall (stage note 10.8.2, replaces the B5 "wall rule"): where the fixed-y
   characteristics of the uniform stream leave the hull (afterbody for U > 0,
   forebody for U < 0) the wall is an INFLOW boundary of d/dx and the j-stencils
   of the first two band lines use a wall node on the waterline (index j = -1/2)
   carrying the trace of the field: D f = D0 f + C f_w. Traces: phi_w = the hull
   solution extrapolated to z = 0 (E phi_B); zeta_w = -(1/g)(phi_t,w - U phi_x,w)
   (dynamic condition at the contact line, nu = 0 there) with phi_x,w from the
   trace and the body condition, phi_x = (d phi_w/dx - w' Q phi_n)/(1 + w'^2),
   Q = |(-H_x, 1, -H_z)| (exact for a wall-sided waterline, H_z = 0; at a flared
   waterline the term w' H_z phi_z/(1 + w'^2) is omitted -- there the uniform-
   stream corner is singular, stage note 10.8.2, and flared-hull results are not
   verified). The old rule (cross term dropped at the wall line, central next to
   it) is kept as wall="rule" for comparison; it is first-order inconsistent
   (error w' f_y on the wall line).

5. Free-surface grid (build_grid): structured, half plane y >= 0; columns
   uniform over the waterline (M cells; the waterline ends are column edges),
   geometrically stretched upstream and downstream; lines: a body-fitted band of
   J lines bent around the waterline (B4 form Y_j = y_wl (1 - j/J) + j dy0),
   then lines stretched laterally. Beaches on ALL sides (upstream, downstream,
   lateral): nu = nu_max max(s_up^2, s_dn^2, s_lat^2), s the depth into the zone.

6. Forces. Direct pressure integration (primary, "D-form"):
       T_jk = -rho int_S (i omega phi_k - U d_x phi_k) n_j dS,   F_j = -T_jk x_k,
       T_jk = -omega^2 A_jk + i omega B_jk,
   d_x phi from the structured hull gradient (piercing.structured_gradient).
   The Ogilvie-Tuck m-term form T^OT_jk = -rho int_S phi_k (i omega n_j - m_j)
   is computed as information: the two differ by
   E_j(phi) = int_S (U n_j d_x phi - m_j phi) dS, which vanishes only when the
   basis flow is tangent to the hull (Tuck's theorem); for the uniform stream
   it is the NK inconsistency of order U n_x (stage note 10.2).
   Excitation (heading beta, incident phi_I = (i g/omega_0) e^{kz} e^{-ik(x cos
   beta + y sin beta)}, encounter omega_e = omega_0 - U k cos beta; y-even part):
       X_j = rho int_S (i omega_e - U d_x)(phi_I + phi_D) n_j dS,
   Froude-Krylov part exact (rho int i omega_0 phi_I n_j).
"""
import numpy as np
import scipy.sparse as sp
import torch

from .bie import ImageOperator
from .influence import influence
from .panels import Panels


# ===================================================================== grid
def stretched(h0, length, ratio, h_max=None):
    """Cell edges 0 = s_0 < s_1 < ... < s_n = length with first cell h0,
    growth `ratio` per cell (capped at h_max); the cells are rescaled
    uniformly so the last edge is exactly `length` (n chosen so the scale
    factor is in [0.9, 1.1])."""
    if length <= 0:
        return np.array([0.0])
    hs, tot, h = [], 0.0, h0
    while tot < length:
        hs.append(h)
        tot += h
        h = min(h * ratio, h_max) if h_max else h * ratio
    hs = np.array(hs)
    if len(hs) > 1 and (tot - length) > 0.5 * hs[-1]:
        hs = hs[:-1]
    hs *= length / hs.sum()
    return np.concatenate([[0.0], np.cumsum(hs)])


DEFAULT_GRID = dict(M=40, aspect=1.0, J_mult=2.0, up=1.5, down=4.0, lat=4.0, r_up=1.12, r_dn=1.06, r_lat=1.10,
                    h_max=0.5, b_up=0.75, b_dn=2.5, b_lat=2.5, nu_max=1.5, u_up=0.0, u_dn=0.0, u_lat=0.0)
# Opt-in grid options (S1 stage 2, 2026-10-09; derivation in studies/_cache/derive/s1b_progress_zh.md). With the
# defaults below build_grid runs the original code path and returns the original grid bit for bit.
#   band      "linear": band lines Y_j = y_wl (1 - j/J) + j dy0 (B4 form; the last band row is dy0 - y_wl/J high, up
#             to half the first lattice row -> spacing jump at the band/lattice junction, where the edge-flux pair's
#             dual operator has a bounded O(1) error, S1 stage 1 item 3);
#             "geom": per node column the J band rows grow geometrically from the waterline to the junction with the
#             LAST row exactly dy0 high (ratio r_i solved per column from sum_k dy0 r^-(J-1-k) = J dy0 - y_wl,i; r = 1
#             off the hull), so the row height is continuous across the junction and the adjacent-row ratio r - 1 =
#             O(1/J) = O(h): the dual defect ((h_{q-1} + h_{q+1})/2 - h_q)/h_q = (r - 1)^2/(2 r) is O(h^2).
#   end_width False: the band closes to a point at both waterline ends (node y = 0); True: the end node takes the
#             hull's actual half-breadth there (hull.waterline at the end; a blunt stem of half-width y_s). The first
#             cell outside that end then has a slanted inner edge (from (x_end, y_s) to the next node on y = 0), its
#             sliver of area dx y_s/2 ahead of the stem face is left out and that edge is treated as the wall / centre
#             plane (zero flux; exact as y_s -> 0, O(y_s/dy0) in one cell otherwise).
#   end_sub   (n_stern, n_stem) or one int for both, 0/1 = off: the end column at a waterline end is split into n
#             sub-columns with quadratic clustering toward the end, edges x_end -/+ dx (k/n)^2, so that a waterline
#             of the form y ~ c sqrt(distance) (a blunt elliptic nose, vertical tangent in plan view) is met by chords
#             with EQUAL y increments; the normal distance between the chord and the true waterline in the first
#             sub-column is ~0.21 dx/n^2 instead of ~0.21 dx (one straight chord over the whole nose cuts into the hull
#             by 2.5-4 cm on the scarab195 grids). Sub-columns are inserted in xe; i_b (and i_s) move accordingly;
#             G["dx"] stays the nominal (x_b - x_s)/M.
#   x_nodes   list of earth x where a column edge is forced (waterline knuckles, e.g. where the waterline reaches a
#             chine and its slope jumps): the nearest interior waterline edge is snapped to it if closer than
#             `snap` x the local spacing, else a new edge is inserted.
#   snap      see x_nodes (default 0.35).
DEFAULT_GRID.update(band="linear", end_width=False, end_sub=0, x_nodes=(), snap=0.35)


def _geom_ratio(total, dy0, J):
    """Ratio r >= 1 with dy0 sum_{k=0}^{J-1} r^-k = total (total <= J dy0); r = 1 when total = J dy0. Bisection in
    log r (monotone)."""
    if total >= J * dy0 * (1.0 - 1e-14):
        return 1.0
    lo, hi = 0.0, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        r = np.exp(mid)
        s = dy0 * (1.0 - r ** (-J)) / (1.0 - 1.0 / r)
        if s > total:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return float(np.exp(0.5 * (lo + hi)))


def _band_nodes_geom(ywl_e, dy0, J):
    """(nx+1, J+1) band node y per node column: row heights dy0 r^-(J-1-k), k = 0..J-1 (last row dy0), r per column."""
    n1 = len(ywl_e)
    Yb = np.zeros((n1, J + 1))
    for i in range(n1):
        tot = J * dy0 - ywl_e[i]
        r = _geom_ratio(tot, dy0, J)
        h = dy0 * r ** (-(J - 1 - np.arange(J)))
        if tot > 0:
            h *= tot / h.sum()                  # exact closure of the band at J dy0 (round-off of r)
        Yb[i, 0] = ywl_e[i]
        Yb[i, 1:] = ywl_e[i] + np.cumsum(h)
        Yb[i, J] = J * dy0
    return Yb


def _side(h0, length, ratio, h_max, uniform):
    """Edges over `length`: uniform cells h0 over the first `uniform`
    (rounded to whole cells), then geometric stretching."""
    nu_ = int(round(min(uniform, length) / h0)) if uniform > 0 else 0
    a = h0 * np.arange(nu_ + 1)
    rest = length - a[-1]
    if rest <= 1e-12 * max(1.0, length):
        return a
    b = stretched(h0 * ratio, rest, ratio, h_max)
    return np.concatenate([a, a[-1] + b[1:]])


def build_grid(hull, sink=0.0, trim=0.0, sgn=1.0, **kw):
    """Structured free-surface grid with beaches (module docstring 5).
    Lengths in units of hull.L (up/down are the UPSTREAM and DOWNSTREAM extents
    beyond the waterline ends for the stream direction sgn = sign(U); for U = 0
    use sgn = +1 and symmetric extents); nu_max in units of sqrt(g/L) (scaled by
    the caller with g). Returns dict(fs, shape (nx, ny), Xc, Yc (centroids,
    (nx, ny)), xe, ywl_e, i_s, i_b, J, nu (per panel, per sqrt(g/L)), params)."""
    p = dict(DEFAULT_GRID, **kw)
    L = hull.L
    xs, xb = hull.waterline_ends(sink, trim)
    M = int(p["M"])
    dx = (xb - xs) / M
    hmax = p["h_max"] * L if p["h_max"] else None
    up_side_len = p["up"] * L
    dn_side_len = p["down"] * L
    r_up, r_dn = p["r_up"], p["r_dn"]
    uu, ud = p["u_up"] * L, p["u_dn"] * L
    if sgn < 0:          # stream toward +x: upstream is the -x side
        right = _side(dx, dn_side_len, r_dn, hmax, ud)       # +x side (downstream)
        left = _side(dx, up_side_len, r_up, hmax, uu)        # -x side (upstream)
    else:
        right = _side(dx, up_side_len, r_up, hmax, uu)
        left = _side(dx, dn_side_len, r_dn, hmax, ud)
    xe = np.concatenate([xs - left[::-1], xs + dx * np.arange(1, M), xb + right])
    i_s = len(left) - 1
    i_b = i_s + M
    # ---- opt-in column edits (S1 stage 2): end sub-columns and forced knuckle edges inside the waterline range
    es = p["end_sub"]
    n_st, n_bw = (int(es), int(es)) if np.isscalar(es) else (int(es[0]), int(es[1]))
    xn_list = [float(v) for v in (p["x_nodes"] or ())]
    opt_cols = (n_st >= 2) or (n_bw >= 2) or bool(xn_list)
    if opt_cols:
        inner = list(xe[i_s + 1:i_b])                      # interior waterline edges (strictly between xs and xb)
        for xk in xn_list:
            if not (xs < xk < xb):
                continue
            if inner:
                k = int(np.argmin(np.abs(np.asarray(inner) - xk)))
                gaps = np.diff(np.concatenate([[xs], inner, [xb]]))
                loc = min(gaps[k], gaps[k + 1])
                if abs(inner[k] - xk) <= p["snap"] * loc:
                    inner[k] = xk
                    continue
            inner.append(xk)
            inner.sort()
        if n_bw >= 2:
            x_last = max([xs] + [v for v in inner if v < xb])
            w = xb - x_last
            inner += [xb - w * (k / n_bw) ** 2 for k in range(1, n_bw)]
        if n_st >= 2:
            x_first = min([xb] + [v for v in inner if v > xs])
            w = x_first - xs
            inner += [xs + w * (k / n_st) ** 2 for k in range(1, n_st)]
        inner = sorted(set(inner))
        xe = np.concatenate([xe[:i_s + 1], inner, xe[i_b:]])
        i_b = i_s + len(inner) + 1
    nx = len(xe) - 1
    ywl_e = np.zeros(nx + 1)
    ywl_e[i_s + 1:i_b] = hull.waterline(xe[i_s + 1:i_b], sink, trim)
    if p["end_width"]:
        ywl_e[[i_s, i_b]] = np.maximum(hull.waterline(xe[[i_s, i_b]], sink, trim), 0.0)
    ymax = float(ywl_e.max())
    dy0 = p["aspect"] * dx
    J = max(2, int(np.ceil(p["J_mult"] * ymax / dy0 - 1e-9))) if ymax > 0 else 0
    lat_edges = _side(dy0, p["lat"] * L - J * dy0, p["r_lat"], hmax, max(0.0, p["u_lat"] * L - J * dy0))
    ylines = np.concatenate([dy0 * np.arange(J + 1), J * dy0 + lat_edges[1:]])
    ny = len(ylines) - 1
    Yn = None
    if p["band"] == "geom" and J:
        Yn = np.repeat(np.asarray(ylines, float)[None, :], nx + 1, axis=0)
        Yn[:, :J + 1] = _band_nodes_geom(ywl_e, dy0, J)
    elif p["band"] != "linear":
        raise ValueError(f"band must be 'linear' or 'geom', not {p['band']!r}")

    def Ye(i, j):
        if Yn is not None:
            return Yn[i, j]
        if j <= J and J:
            return ywl_e[i] * (1.0 - j / J) + j * dy0
        return ylines[j]
    q = np.zeros((ny, nx, 4, 3))
    for j in range(ny):
        for i in range(nx):
            q[j, i] = [[xe[i], Ye(i, j), 0.0], [xe[i + 1], Ye(i + 1, j), 0.0],
                       [xe[i + 1], Ye(i + 1, j + 1), 0.0], [xe[i], Ye(i, j + 1), 0.0]]
    q = q.reshape(-1, 4, 3)
    fs = Panels(q, np.ones(len(q), int))
    if np.mean(fs.n[:, 2]) > 0:
        fs = Panels(q[:, ::-1], np.ones(len(q), int))
    assert np.all(fs.n[:, 2] < 0)
    Xc = fs.c[:, 0].reshape(ny, nx).T.copy()
    Yc = fs.c[:, 1].reshape(ny, nx).T.copy()
    # beaches (fraction s into each zone, from its inner edge to the outer edge)
    x0, x1 = xe[0], xe[-1]
    lup, ldn, llat = p["b_up"] * L, p["b_dn"] * L, p["b_lat"] * L
    if sgn < 0:
        s_up = np.clip((x0 + lup - Xc) / lup, 0, 1)
        s_dn = np.clip((Xc - (x1 - ldn)) / ldn, 0, 1)
    else:
        s_up = np.clip((Xc - (x1 - lup)) / lup, 0, 1)
        s_dn = np.clip((x0 + ldn - Xc) / ldn, 0, 1)
    y1 = ylines[-1]
    s_lat = np.clip((Yc - (y1 - llat)) / llat, 0, 1)
    nu = p["nu_max"] * np.maximum(np.maximum(s_up, s_dn), s_lat) ** 2
    out = dict(fs=fs, shape=(nx, ny), Xc=Xc, Yc=Yc, xe=xe, ylines=ylines, ywl_e=ywl_e, i_s=i_s, i_b=i_b, J=J,
               dx=dx, dy0=dy0, nu=nu.T.reshape(-1), params=p, sgn=sgn, L=L)
    if Yn is not None:
        out["Yn"] = Yn                     # node y (nx+1, ny+1) of the geometrically graded band (opt-in)
    return out


class NoWaterline:
    """Stand-in 'hull' for a SUBMERGED body (no waterline): build_grid then
    makes a plain structured grid with uniform columns over [x0, x1] and no
    band (check TN-d, classical Timman-Newman where it is a theorem)."""

    def __init__(self, x0, x1, L):
        self.x0, self.x1, self.L = float(x0), float(x1), float(L)
        self.name = f"submerged_{x0:g}_{x1:g}"

    def waterline_ends(self, sink=0.0, trim=0.0):
        return self.x0, self.x1

    def waterline(self, x, sink=0.0, trim=0.0):
        return np.zeros_like(np.asarray(x, float))


# ---------------------------------------------------------- d/dx operator
_UP_P = np.array([-2.0, -3.0, 6.0, -1.0]) / 6.0     # offsets -1, 0, +1, +2 (upwind side +), third order
_UP_M = np.array([1.0, -6.0, 3.0, 2.0]) / 6.0       # offsets -2, -1, 0, +1 (upwind side -)
_UP5_P = np.array([3.0, -30.0, -20.0, 60.0, -15.0, 2.0]) / 60.0   # offsets -2 .. +3 (upwind side +), fifth order
_UP5_M = np.array([-2.0, 15.0, -60.0, 20.0, 30.0, -3.0]) / 60.0   # offsets -3 .. +2 (upwind side -)
ORDER = 5            # interior order of the upwind-biased d/dx (3 or 5); stage note 10.3


def _index_derivative(n, side, ghost_lo, ghost_hi, order=None):
    """(n x n) sparse index-space derivative along one grid direction for the
    upwind side per point (side[k] = +1/-1/0; 0 -> central). End treatments
    ghost_lo/hi:
      "zero"   inflow: outside values 0 (undisturbed stream);
      "mirror" f_{-1-m} = f_m (centre plane, even field);
      "linear" outside values by linear extrapolation (coordinates only, so
               that linear functions are differentiated exactly);
      "wall"   hull side of the band: if the UPWIND points lie behind the wall
               the derivative is dropped at the wall point (the band line is
               then followed along the waterline) and taken central at the
               next point -- a downwind formula there makes the semi-discrete
               system unstable (B5 exploration, stage note 10.3);
      "onesided" outflow end: second-order fully upwind formula when the
               downwind point is missing.
    Downwind-biased formulas are never used. order: interior order (5: six-
    point upwind-biased formula, third order next to the ends; 3)."""
    order = ORDER if order is None else order
    rows, cols, vals = [], [], []

    def put(k, offs, w):
        cc, vv = [], []
        for o, ww in zip(offs, w):
            ii = k + o
            if 0 <= ii < n:
                cc.append(ii)
                vv.append(ww)
                continue
            mode = ghost_lo if ii < 0 else ghost_hi
            if mode == "zero":
                continue
            if mode == "mirror":
                cc.append(-1 - ii if ii < 0 else 2 * n - 1 - ii)
                vv.append(ww)
                continue
            if mode == "linear" and n >= 2:
                e, e2 = (0, 1) if ii < 0 else (n - 1, n - 2)
                d = abs(ii - e)
                cc += [e, e2]
                vv += [ww * (1 + d), -ww * d]
                continue
            return False, None, None
        return True, cc, vv

    for k in range(n):
        s = side[k]
        ok = False
        if s != 0 and order >= 5:
            ok, cc, vv = put(k, [-2, -1, 0, 1, 2, 3], _UP5_P) if s > 0 else put(k, [-3, -2, -1, 0, 1, 2], _UP5_M)
        if not ok:
            if s > 0:
                ok, cc, vv = put(k, [-1, 0, 1, 2], _UP_P)
            elif s < 0:
                ok, cc, vv = put(k, [-2, -1, 0, 1], _UP_M)
            else:
                ok, cc, vv = put(k, [-1, 1], [-0.5, 0.5])
        if not ok:
            lo_missing = k < 2
            up_side_missing = (s < 0 and lo_missing) or (s > 0 and not lo_missing)
            mode = ghost_lo if lo_missing else ghost_hi
            if up_side_missing and mode == "wall":
                if (lo_missing and k == 0) or (not lo_missing and k == n - 1):
                    cc, vv = [], []                         # dropped at the wall point
                else:
                    cc, vv = [k - 1, k + 1], [-0.5, 0.5]    # central next to it
            elif up_side_missing and mode == "downwind":
                cc, vv = ([k, k + 1, k + 2], [-1.5, 2.0, -0.5]) if lo_missing else ([k - 2, k - 1, k], [0.5, -2.0, 1.5])
            elif not up_side_missing and lo_missing and k + 2 < n:
                cc, vv = [k, k + 1, k + 2], [-1.5, 2.0, -0.5]   # fully upwind (upwind side +)
            elif not up_side_missing and not lo_missing and k - 2 >= 0:
                cc, vv = [k - 2, k - 1, k], [0.5, -2.0, 1.5]    # fully upwind (upwind side -)
            else:
                raise ValueError(f"no stable stencil at index {k} (side {s}, ends {ghost_lo}/{ghost_hi})")
        rows += [k] * len(cc)
        cols += cc
        vals += vv
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    return A


def _geom_derivative(F, axis, mirror_lo=False, odd=False):
    """Index derivative of a smooth geometric field (central, second-order
    one-sided at the ends; mirror at the low end if requested: F_{-1} = +-F_0)."""
    F = np.moveaxis(np.asarray(F, float), axis, 0)
    Dv = np.empty_like(F)
    Dv[1:-1] = 0.5 * (F[2:] - F[:-2])
    if mirror_lo:
        Dv[0] = 0.5 * (F[1] - (-F[0] if odd else F[0]))
    else:
        Dv[0] = 0.5 * (-3 * F[0] + 4 * F[1] - F[2])
    Dv[-1] = 0.5 * (3 * F[-1] - 4 * F[-2] + F[-3])
    return np.moveaxis(Dv, 0, axis)


def lagrange_deriv_weights(nodes, x0):
    """Weights of the derivative at x0 of the Lagrange interpolant through `nodes`."""
    nodes = np.asarray(nodes, float)
    n = len(nodes)
    w = np.zeros(n)
    for k in range(n):
        others = [m for m in range(n) if m != k]
        den = np.prod([nodes[k] - nodes[m] for m in others])
        w[k] = sum(np.prod([x0 - nodes[q] for q in others if q != m]) for m in others) / den
    return w


_W_WALL0 = lagrange_deriv_weights([-0.5, 0.0, 1.0], 0.0)          # band line 0: wall node, lines 0, 1
_W_WALL1 = lagrange_deriv_weights([-0.5, 0.0, 1.0, 2.0], 1.0)     # band line 1: wall node, lines 0, 1, 2


def _jstencil_trace(ny, side):
    """j-derivative of a hull column whose upwind side at the wall is the wall
    (side[0] < 0): rows 0 and 1 use the wall node at j = -1/2 (second- and
    third-order upwind-biased Lagrange formulas), the other rows the standard
    stencils. Returns (Dmain (ny x ny) csr, cw (ny,) wall-node weights)."""
    Dl = _index_derivative(ny, side, "wall", "zero").tolil()
    cw = np.zeros(ny)
    for k, Wk in ((0, _W_WALL0), (1, _W_WALL1)):
        if k >= ny or side[k] >= 0:
            continue
        Dl.rows[k], Dl.data[k] = [], []
        for c, v in enumerate(Wk[1:]):
            Dl[k, c] = v
        cw[k] = Wk[0]
    return Dl.tocsr(), cw


def dx_operator(G, sgn, wall="trace", hull=None, sink=0.0, trim=0.0):
    """Sparse D ~ d/dx at fixed y on the panel centroids of grid G (ordering
    k = j nx + i), upwind-biased for the advection term U d/dx with sign(U) =
    sgn (module docstring 4). The coordinates are differentiated with the SAME
    stencils as the field (linear extrapolation instead of the field's inflow
    ghosts; the exact wall-node coordinates where the wall node is used), so
    linear functions are differentiated exactly wherever the field stencil does
    not reach an inflow ghost. wall="trace" (production): hull columns whose
    j-upwind side is the wall use the wall node (D f = D0 f + C f_w; needs
    `hull`); wall="rule": the B5 rule (cross term dropped at the wall line).
    Returns (D0, info) with info["wall"] = dict(C (N x n_w csr), cols, xw, yw)
    or None."""
    nx, ny = G["shape"]
    X, Y = G["Xc"], G["Yc"]
    hull_j0 = np.zeros(nx, bool)
    if G["J"] > 0:
        hull_j0[G["i_s"]:G["i_b"]] = True          # line 0 touches the hull in these columns
    if wall == "trace" and G["J"] > 0 and hull is None:
        raise ValueError("dx_operator(wall='trace') needs the hull (waterline of the wall nodes)")
    s = 1.0 if sgn >= 0 else -1.0
    # 1) sides from central geometric derivatives
    Xi0, Yi0 = _geom_derivative(X, 0), _geom_derivative(Y, 0)
    Xj0, Yj0 = np.empty_like(X), np.empty_like(Y)
    for i in range(nx):
        mir = not hull_j0[i]
        Xj0[i] = _geom_derivative(X[i], 0, mirror_lo=mir, odd=False)
        Yj0[i] = _geom_derivative(Y[i], 0, mirror_lo=mir, odd=True)
    J0 = Xi0 * Yj0 - Xj0 * Yi0
    side_i = np.sign(s * Yj0 / J0)
    side_j = np.sign(-s * Yi0 / J0)
    # 2) stencils per line / column; geometry with linear ghosts, field with the physical ghosts
    gi_lo, gi_hi = ("zero", "onesided") if s < 0 else ("onesided", "zero")
    Di_f, Dj_f = [], {}
    Xi, Yi = np.empty_like(X), np.empty_like(Y)
    Xj, Yj = np.zeros_like(X), np.zeros_like(Y)
    for j in range(ny):
        Dg = _index_derivative(nx, side_i[:, j], "linear", "linear")
        Xi[:, j], Yi[:, j] = Dg @ X[:, j], Dg @ Y[:, j]
        Di_f.append(_index_derivative(nx, side_i[:, j], gi_lo, gi_hi))
    wall_cols = [i for i in range(nx) if wall == "trace" and hull_j0[i] and side_j[i, 0] < 0]
    xw = X[wall_cols, 0].copy() if wall_cols else np.zeros(0)
    yw = hull.waterline(xw, sink, trim) if wall_cols else np.zeros(0)
    cw_f = {}
    for i in range(nx):
        if i in wall_cols:
            Dm, cw = _jstencil_trace(ny, side_j[i])
            k = wall_cols.index(i)
            Dg = _index_derivative(ny, side_j[i], "linear", "linear")
            Xj[i], Yj[i] = Dg @ X[i], Dg @ Y[i]
            rw = cw != 0                                   # rows using the wall node: exact wall coordinates
            Xj[i, rw] = (Dm @ X[i] + cw * xw[k])[rw]
            Yj[i, rw] = (Dm @ Y[i] + cw * yw[k])[rw]
            Dj_f[i], cw_f[i] = Dm, cw
            continue
        lo = "linear" if hull_j0[i] else "mirror"
        Dg = _index_derivative(ny, side_j[i], lo, "linear")
        if lo == "mirror":
            # mirror for the even X, odd Y: build Y's derivative with an explicit odd ghost
            Xj[i] = Dg @ X[i]
            Yj[i] = _odd_mirror_derivative(Y[i], side_j[i], ny)
        else:
            Xj[i], Yj[i] = Dg @ X[i], Dg @ Y[i]
        Dj_f[i] = _index_derivative(ny, side_j[i], "wall" if hull_j0[i] else "mirror", "zero")
    Jac = Xi * Yj - Xj * Yi
    assert np.all(Jac > 0), "grid not right-handed"
    ix = Yj / Jac
    jx = -Yi / Jac
    rows, cols, vals = [], [], []
    for j in range(ny):
        Di = sp.coo_matrix(Di_f[j])
        rows += list(j * nx + Di.row)
        cols += list(j * nx + Di.col)
        vals += list(Di.data * ix[Di.row, j])
    scale = np.max(np.abs(ix))
    crow, ccol, cval = [], [], []
    for i in range(nx):
        if np.max(np.abs(jx[i])) < 1e-13 * scale and np.max(np.abs(Xj[i])) < 1e-13:
            continue
        Dj = sp.coo_matrix(Dj_f[i])
        rows += list(Dj.row * nx + i)
        cols += list(Dj.col * nx + i)
        vals += list(Dj.data * jx[i, Dj.row])
        if i in cw_f:
            k = wall_cols.index(i)
            for r in np.flatnonzero(cw_f[i]):
                crow.append(r * nx + i)
                ccol.append(k)
                cval.append(cw_f[i][r] * jx[i, r])
    N = nx * ny
    D = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    D.sum_duplicates()
    wl = None
    if wall_cols:
        C = sp.csr_matrix((cval, (crow, ccol)), shape=(N, len(wall_cols)))
        C.sum_duplicates()
        wl = dict(C=C, cols=wall_cols, xw=xw, yw=yw, sink=sink, trim=trim)
    return D, dict(ix=ix, jx=jx, Jac=Jac, side_i=side_i, side_j=side_j, wall=wl)


def wall_trace_data(wl, hull, body, hull_shape):
    """Trace operators of the wall closure (module docstring 4) for the wall
    nodes wl = dx_operator(...)[1]["wall"] and the structured hull mesh `body`
    (stations x rows, shape hull_shape):
      E  (n_w x NB): phi_w = E phi_B, linear extrapolation in z to z = 0 from the
         top two rows of each station, linear interpolation in x (top-row
         centroid x) to the wall nodes;
      En (n_w x NB): phi_n of the top row interpolated to the wall nodes;
      Tw (n_w x n_w), tn (n_w,): phi_x,w = Tw phi_w + tn (En phi_n) with Tw =
         (1/(1 + w'^2)) d/dx along each contiguous run of wall nodes (three-
         point Lagrange formulas) and tn = -w' Q/(1 + w'^2), Q = |(-H_x, 1, -H_z)|.
    Returns dict(C, E, En, Tw, tn, xw) or None."""
    if wl is None:
        return None
    xw = np.asarray(wl["xw"], float)
    sink, trim = wl["sink"], wl["trim"]
    nxh, nzh = hull_shape
    Cc = body.c.reshape(nxh, nzh, 3)
    z1, z2 = Cc[:, -1, 2], Cc[:, -2, 2]
    tz = (0.0 - z1) / (z1 - z2)
    xs = Cc[:, -1, 0]
    o = np.argsort(xs)
    xso = xs[o]
    er, ec, ev, nr, nc, nv = [], [], [], [], [], []
    for k, x in enumerate(xw):
        m = int(np.clip(np.searchsorted(xso, x) - 1, 0, nxh - 2))
        a = float(np.clip((x - xso[m]) / (xso[m + 1] - xso[m]), -0.5, 1.5))
        for ist, wg in ((o[m], 1.0 - a), (o[m + 1], a)):
            i1, i2 = ist * nzh + nzh - 1, ist * nzh + nzh - 2
            er += [k, k]
            ec += [i1, i2]
            ev += [wg * (1.0 + tz[ist]), -wg * tz[ist]]
            nr.append(k)
            nc.append(i1)
            nv.append(wg)
    NB = nxh * nzh
    E = sp.csr_matrix((ev, (er, ec)), shape=(len(xw), NB))
    E.sum_duplicates()
    En = sp.csr_matrix((nv, (nr, nc)), shape=(len(xw), NB))
    En.sum_duplicates()
    H, Hx, Hz, _, _ = hull.wall_derivatives(xw, sink, trim)
    Q = np.sqrt(1.0 + Hx ** 2 + Hz ** 2)
    n = len(xw)
    Dw = np.zeros((n, n))
    cols = list(wl["cols"])
    runs, cur = [], [0]
    for k in range(1, n):
        if cols[k] == cols[k - 1] + 1:
            cur.append(k)
        else:
            runs.append(cur)
            cur = [k]
    runs.append(cur)
    for run in runs:
        if len(run) == 1:
            continue                        # isolated wall node: d/dx along the wall unknown -> 0
        for p, k in enumerate(run):
            if len(run) == 2:
                q = run
            elif p == 0:
                q = run[:3]
            elif p == len(run) - 1:
                q = run[-3:]
            else:
                q = run[p - 1:p + 2]
            Dw[k, q] = lagrange_deriv_weights(xw[q], xw[k])
    Tw = sp.csr_matrix(Dw / (1.0 + Hx ** 2)[:, None])
    tn = -Hx * Q / (1.0 + Hx ** 2)
    return dict(C=sp.csr_matrix(wl["C"]), E=E, En=En, Tw=Tw, tn=tn, xw=xw)


def _odd_mirror_derivative(Y, side, n):
    """Index derivative of an odd field (Y at the centre plane: Y_{-1-m} =
    -Y_m) with the stencils of _index_derivative (sides as given)."""
    ext = np.concatenate([-Y[:2][::-1], Y, 2 * Y[-1:] - Y[-2:-1], 3 * Y[-1:] - 2 * Y[-2:-1]])
    out = np.empty(n)
    for k in range(n):
        e = k + 2
        if side[k] > 0:
            out[k] = np.dot(_UP_P, ext[e - 1:e + 3])
        elif side[k] < 0:
            out[k] = np.dot(_UP_M, ext[e - 2:e + 2])
        else:
            out[k] = 0.5 * (ext[e + 1] - ext[e - 1])
    return out


# ====================================================================== solver
def _ts(A, dev):
    A = sp.coo_matrix(A)
    idx = torch.as_tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=dev)
    return torch.sparse_coo_tensor(idx, torch.as_tensor(A.data, dtype=torch.float64, device=dev),
                                   A.shape).coalesce()


def mode_vectors(pan, ref=(0.0, 0.0), U=0.0, modes=(1, 3, 5)):
    """n_j and m_j (uniform stream W = -U x_hat) on the body panels for the
    symmetric modes; pitch about (x_ref, z_ref). Returns dict mode -> (n_j, m_j)."""
    n = pan.n
    c = pan.c
    xr, zr = ref
    out = {}
    for k in modes:
        if k == 1:
            out[k] = (n[:, 0].copy(), np.zeros(len(pan)))
        elif k == 3:
            out[k] = (n[:, 2].copy(), np.zeros(len(pan)))
        elif k == 5:
            out[k] = ((c[:, 2] - zr) * n[:, 0] - (c[:, 0] - xr) * n[:, 2], U * n[:, 2])
        else:
            raise ValueError("only the port/starboard symmetric modes 1, 3, 5")
    return out


class ForwardSpeedLTI:
    """Body panels (half hull y >= 0, normals into the fluid) + free-surface
    panels (y >= 0, z = 0, normals -z), images in y and the rigid lid z = 0;
    D: sparse d/dx on the free-surface panels for this U (dx_operator), nu:
    beach per free-surface panel (1/s). All dense blocks live on the GPU."""

    def __init__(self, body, fs, D, nu, U, g=9.81, rho=1.0, device=None, hull_shape=None, keep_blocks=False,
                 wall=None):
        """wall: wall_trace_data(...) of the wall closure (D is then D0), or None."""
        self.body, self.fs = body, fs
        self.NB, self.NF = len(body), len(fs)
        self.U, self.g, self.rho = float(U), float(g), float(rho)
        self.dev = dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hull_shape = hull_shape
        self.nu = np.asarray(nu, float)
        self.wall = wall if (wall is not None and wall["C"].shape[1] > 0) else None
        # body blocks with images (y even, lid z even)
        ob = ImageOperator(body, axes=(1, 2), device=dev, parities=[(1, 1)], on_device=True)
        S_BB, D_BB = ob.S((1, 1)), ob.D((1, 1))
        del ob
        MB = -D_BB
        MB.diagonal().add_(2 * np.pi)
        self._luB = torch.linalg.lu_factor(MB)
        del MB, D_BB
        self.S_BB = S_BB
        NF, NB = self.NF, self.NB
        D_FB = torch.empty((NF, NB), dtype=torch.float64, device=dev)
        S_FB = torch.empty((NF, NB), dtype=torch.float64, device=dev)
        for r0 in range(0, NF, 8192):
            r1 = min(NF, r0 + 8192)
            op = ImageOperator(body, axes=(1, 2), xf=fs.c[r0:r1], device=dev, parities=[(1, 1)], on_device=True)
            D_FB[r0:r1] = op.D((1, 1))
            S_FB[r0:r1] = op.S((1, 1))
            del op
        S_BF = torch.empty((NB, NF), dtype=torch.float64, device=dev)
        S1, _ = influence(body.c, fs, device=dev, as_numpy=False)
        S2, _ = influence(body.c, fs.reflect(1), device=dev, as_numpy=False)
        S_BF[:] = 2.0 * (S1 + S2)
        del S1, S2
        self.S_BF = S_BF
        # free surface on free surface: panel + lid image (coincident, x2) + y image (x2)
        S_FF = torch.empty((NF, NF), dtype=torch.float64, device=dev)
        si = np.arange(NF)
        for r0 in range(0, NF, 4096):
            r1 = min(NF, r0 + 4096)
            S1, _ = influence(fs.c[r0:r1], fs, device=dev, as_numpy=False)
            S2, _ = influence(fs.c[r0:r1], fs.reflect(1), device=dev, as_numpy=False)
            S_FF[r0:r1] = 2.0 * (S1 + S2)
            del S1, S2
        # Neumann-to-Dirichlet maps
        X = torch.linalg.lu_solve(*self._luB, S_BF)                       # Bi S_BF   (NB, NF)
        Rs = S_FF
        Rs.add_(D_FB @ X)
        Rs.mul_(-1.0 / (4 * np.pi))                                        # R_s = -(D_FB Bi S_BF + S_FF)/(4 pi)
        del X
        Xn = torch.linalg.lu_solve(*self._luB, S_BB)
        self.Rn = -(D_FB @ Xn + S_FB) / (4 * np.pi)                        # (NF, NB)
        del Xn
        if keep_blocks:
            self.D_FB, self.S_FB = D_FB, S_FB
        else:
            del D_FB, S_FB
        self.Rs = Rs
        self.set_operator(D, nu)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_operator(self, D, nu):
        """N0 = diag(nu) - U D0 and the products. Without a wall closure P1 =
        N R_s, P2 = N^2 R_s. With the wall closure (module docstring 3-4; stage
        note 10.8.2), the first application of N acts on phi_F with the hull
        trace (N phi = N0 phi - U C E phi_B, phi_B = P sigma + Q phi_n, P =
        -Bi S_BF, Q = -Bi S_BB) and the second on zeta with the contact-line
        trace zeta_w = -(1/g)(i w phi_w - U phi_x,w), which gives
            sigma = (1/g)[-w^2 phi + 2 i w N phi + N0 N phi + U^2 C phi_x,w],
        i.e. P1 = N0 R_s - U C (E P), P2 = N0 P1 + U^2 C (Tw E P)."""
        self.D = sp.csr_matrix(D)
        self.nu = np.asarray(nu, float)
        Nm = sp.diags(self.nu) - self.U * self.D
        self.Nsp = sp.csr_matrix(Nm)
        self._N = _ts(self.Nsp, self.dev)
        self.P1 = torch.sparse.mm(self._N, self.Rs)
        self._luRs = None
        if self.wall is None or self.U == 0.0:
            self._wall_on = False
            self.P2 = torch.sparse.mm(self._N, self.P1)
            return
        self._wall_on = True
        dev, U, w = self.dev, self.U, self.wall
        Ed = torch.as_tensor(w["E"].toarray(), device=dev)
        EBi = torch.linalg.lu_solve(*self._luB, Ed.T.contiguous(), adjoint=True).T       # E Bi
        self.EP = -(EBi @ self.S_BF)                                                     # (n_w, NF)
        self.EQ = -(EBi @ self.S_BB)                                                     # (n_w, NB)
        Twd = torch.as_tensor(w["Tw"].toarray(), device=dev)
        self.TEP = Twd @ self.EP
        self.TEQ = Twd @ self.EQ
        self.TnEn = torch.as_tensor((sp.diags(w["tn"]) @ w["En"]).toarray(), device=dev)  # (n_w, NB)
        self.Et = torch.as_tensor(w["E"].toarray(), device=dev)
        self.Twt = Twd
        self._C = _ts(w["C"], dev)
        self.P1 = self.P1 - U * torch.sparse.mm(self._C, self.EP)
        self.P2 = torch.sparse.mm(self._N, self.P1) + U * U * torch.sparse.mm(self._C, self.TEP)

    def Napply(self, v):
        """N0 v for v (NF, m) torch (real or complex)."""
        if torch.is_complex(v):
            return torch.complex(torch.sparse.mm(self._N, v.real.contiguous()),
                                 torch.sparse.mm(self._N, v.imag.contiguous()))
        return torch.sparse.mm(self._N, v)

    def _Capply(self, v):
        if torch.is_complex(v):
            return torch.complex(torch.sparse.mm(self._C, v.real.contiguous()),
                                 torch.sparse.mm(self._C, v.imag.contiguous()))
        return torch.sparse.mm(self._C, v)

    # ------------------------------------------------------------ frequency
    def system(self, omega):
        g = self.g
        A = torch.complex(-(self.P2 - omega * omega * self.Rs) / g, -(2.0 * omega / g) * self.P1)
        A.diagonal().add_(1.0)
        return A

    def solve_freq(self, omega, phin):
        """phin: (NB, m) complex normal velocities on the body panels (per unit
        amplitude). Returns dict(phi_B (NB, m), phi_F, sigma) numpy complex."""
        dev, g, U = self.dev, self.g, self.U
        pn = torch.as_tensor(np.asarray(phin, complex).reshape(self.NB, -1), dtype=torch.complex128, device=dev)

        def rmul(Mt, x):
            return torch.complex(Mt @ x.real.contiguous(), Mt @ x.imag.contiguous())
        v = rmul(self.Rn, pn)
        n1 = self.Napply(v)
        extra = 0.0
        if self._wall_on:
            n1 = n1 - U * self._Capply(rmul(self.EQ, pn))
            extra = U * U * self._Capply(rmul(self.TEQ, pn) + rmul(self.TnEn, pn))
        rhs = (-omega * omega * v + 2j * omega * n1 + self.Napply(n1) + extra) / g
        A = self.system(omega)
        sig = torch.linalg.solve(A, rhs)
        del A
        return self._finish(sig, pn, v)

    def state_matrix(self):
        """Dense matrix M of the free semi-discrete system y' = M y, y = (zeta,
        phi_F), phi_n = 0 (spectrum check SP): zeta' = -N zeta - sigma, phi_F' =
        -N phi_F - g zeta, sigma = R_s^-1 phi_F, with the wall traces when the
        closure is on (zeta_w uses phi_t,w = E P R_s^-1 phi_F')."""
        dev, g, U = self.dev, self.g, self.U
        NF = self.NF
        Ri = torch.linalg.inv(self.Rs)
        A = -torch.as_tensor(self.Nsp.toarray(), device=dev)
        Azp = -Ri
        if self._wall_on:
            Cd = torch.as_tensor(self.wall["C"].toarray(), device=dev)
            A = A + U * Cd @ (self.EP @ Ri)
            Azp = -Ri - (U / g) * Cd @ (self.EP @ Ri @ A) + (U * U / g) * Cd @ (self.TEP @ Ri)
        M = torch.zeros((2 * NF, 2 * NF), dtype=torch.float64, device=dev)
        M[:NF, :NF] = A
        M[:NF, NF:] = Azp
        M[NF:, :NF] = -g * torch.eye(NF, dtype=torch.float64, device=dev)
        M[NF:, NF:] = A
        out = M.cpu().numpy()
        del M, Ri, A, Azp
        return out

    def _finish(self, sig, pn, v):
        phiF = torch.complex(self.Rs @ sig.real.contiguous(), self.Rs @ sig.imag.contiguous()) + v
        rb = -torch.complex(self.S_BB @ pn.real.contiguous() + self.S_BF @ sig.real.contiguous(),
                            self.S_BB @ pn.imag.contiguous() + self.S_BF @ sig.imag.contiguous())
        phiB = torch.complex(torch.linalg.lu_solve(*self._luB, rb.real.contiguous()),
                             torch.linalg.lu_solve(*self._luB, rb.imag.contiguous()))
        return dict(phi_B=phiB.cpu().numpy(), phi_F=phiF.cpu().numpy(), sigma=sig.cpu().numpy())

    def solve_inf(self, phin, phiF=None):
        """Instantaneous (omega -> infinity) problem with prescribed free-surface
        potential phi_F (default 0): sigma = R_s^{-1} (phi_F - R_n phi_n) (real data)."""
        dev = self.dev
        pn = torch.as_tensor(np.asarray(phin, float).reshape(self.NB, -1), device=dev)
        if self._luRs is None:
            self._luRs = torch.linalg.lu_factor(self.Rs)
        v = self.Rn @ pn
        if phiF is not None:
            v = v - torch.as_tensor(np.asarray(phiF, float).reshape(self.NF, -1), device=dev)
        sig = -torch.linalg.lu_solve(*self._luRs, v)
        rb = -(self.S_BB @ pn + self.S_BF @ sig)
        phiB = torch.linalg.lu_solve(*self._luB, rb)
        pF = np.zeros((self.NF, pn.shape[1])) if phiF is None else np.asarray(phiF, float).reshape(self.NF, -1)
        return dict(phi_B=phiB.cpu().numpy(), phi_F=pF, sigma=sig.cpu().numpy())

    def solve_steady(self, phin):
        """omega = 0 (steady NK problem on this grid, e.g. phin = U n_x)."""
        return self.solve_freq(0.0, np.asarray(phin, float).astype(complex))

    # ---------------------------------------------------------- time domain
    def dtn_factor(self):
        if self._luRs is None:
            self._luRs = torch.linalg.lu_factor(self.Rs)

    def sigma_of(self, phiF, pn):
        """sigma = R_s^{-1} (phi_F - R_n phi_n) (torch, real, columns)."""
        self.dtn_factor()
        return torch.linalg.lu_solve(*self._luRs, (phiF - self.Rn @ pn))

    def phiB_of(self, sig, pn):
        return torch.linalg.lu_solve(*self._luB, -(self.S_BB @ pn + self.S_BF @ sig))

    def rhs_td(self, zeta, phiF, pn, pnd=None):
        """Right-hand side of the semi-discrete system for the body normal
        velocity pn (and its time derivative pnd, needed by the contact-line
        trace of the wall closure)."""
        sig = self.sigma_of(phiF, pn)
        dz = -torch.sparse.mm(self._N, zeta) - sig
        dp = -torch.sparse.mm(self._N, phiF) - self.g * zeta
        if self._wall_on:
            U, g = self.U, self.g
            pnd = torch.zeros_like(pn) if pnd is None else pnd
            phB = self.phiB_of(sig, pn)
            phw = self.Et @ phB
            dp = dp + U * torch.sparse.mm(self._C, phw)
            sigd = self.sigma_of(dp, pnd)
            phwt = self.Et @ self.phiB_of(sigd, pnd)
            phxw = self.Twt @ phw + self.TnEn @ pn
            zw = -(phwt - U * phxw) / g
            dz = dz + U * torch.sparse.mm(self._C, zw)
        return dz, dp, sig

    def wall_phi(self, sig, pn):
        """Hull trace phi_w at the wall nodes (None without the closure)."""
        if not self._wall_on:
            return None
        return self.Et @ self.phiB_of(sig, pn)

    def release(self):
        for a in ("S_BB", "S_BF", "Rn", "Rs", "P1", "P2", "_N", "_luB", "_luRs", "D_FB", "S_FB", "EP", "EQ", "TEP",
                  "TEQ", "TnEn", "Et", "Twt", "_C"):
            if hasattr(self, a):
                setattr(self, a, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ================================================================== physics
def incident(points, normals, omega0, beta, g, amp=1.0):
    """y-even part of the deep-water incident wave phi_I = (i g A/omega_0)
    e^{kz} e^{-ik(x cos b + y sin b)} (A = amp): returns (phi_I, dphi_I/dn,
    dphi_I/dx) at the points. For beta = 0, pi the wave is y-independent."""
    k = omega0 * omega0 / g
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    cb, sb = np.cos(beta), np.sin(beta)
    c0 = 1j * g * amp / omega0
    ex = np.exp(k * z - 1j * k * x * cb)
    cy = np.cos(k * y * sb)
    phi = c0 * ex * cy
    gx = -1j * k * cb * phi
    gy = -c0 * ex * k * sb * np.sin(k * y * sb)
    gz = k * phi
    return phi, gx * normals[:, 0] + gy * normals[:, 1] + gz * normals[:, 2], gx


def hull_dx(pan, phi, phin, shape):
    """d phi/dx on the structured hull mesh (complex allowed): structured
    gradient of the real and imaginary parts (piercing.structured_gradient)."""
    from .piercing import structured_gradient
    phi = np.asarray(phi)
    phin = np.asarray(phin)
    if np.iscomplexobj(phi) or np.iscomplexobj(phin):
        gr = structured_gradient(pan, phi.real, np.real(phin), shape)
        gi = structured_gradient(pan, phi.imag, np.imag(phin), shape)
        return gr[:, 0] + 1j * gi[:, 0]
    return structured_gradient(pan, phi, phin, shape)[:, 0]


def hull_waterline_values(pan, f, shape):
    """Values of f at the waterline (z = 0) of a structured hull mesh: linear
    extrapolation in z from the top two rows of each station; returns
    (x_c (station centroid x), f_wl) per station."""
    nx, nz = shape
    C = pan.c.reshape(nx, nz, 3)
    F = np.asarray(f).reshape(nx, nz)
    z1, z2 = C[:, -1, 2], C[:, -2, 2]
    t = (0.0 - z1) / (z1 - z2)
    return C[:, -1, 0], F[:, -1] + t * (F[:, -1] - F[:, -2])


class Coefficients:
    """Radiation and excitation evaluation on a ForwardSpeedLTI for a hull mesh
    (structured, shape (nx, nz)) with modes about ref."""

    def __init__(self, solver, ref=(0.0, 0.0), modes=(1, 3, 5)):
        self.s = solver
        self.modes = tuple(modes)
        self.mv = mode_vectors(solver.body, ref, solver.U, modes)
        self.ref = ref

    def radiation(self, omega, want_phi=False, d_form=True):
        s = self.s
        a = s.body.area
        ph = np.stack([1j * omega * self.mv[k][0] + self.mv[k][1] for k in self.modes], 1)
        sol = s.solve_freq(omega, ph)
        return self._coeffs(omega, sol, ph, want_phi, d_form)

    def _coeffs(self, omega, sol, ph, want_phi=False, d_form=True):
        s = self.s
        a = s.body.area
        m = len(self.modes)
        T = np.zeros((m, m), complex)
        Tot = np.zeros((m, m), complex)
        dx = []
        for c in range(m):
            phk = sol["phi_B"][:, c]
            dxk = hull_dx(s.body, phk, ph[:, c], s.hull_shape) if (d_form and s.hull_shape) else np.zeros(s.NB)
            dx.append(dxk)
            for r, j in enumerate(self.modes):
                nj, mj = self.mv[j]
                T[r, c] = -s.rho * 2 * np.sum((1j * omega * phk - s.U * dxk) * nj * a)
                Tot[r, c] = -s.rho * 2 * np.sum(phk * (1j * omega * nj - mj) * a)
        out = dict(omega=omega, T=T, T_ot=Tot, A=-(T.real) / omega ** 2 if omega else None,
                   B=T.imag / omega if omega else None)
        if want_phi:
            out.update(sol=sol, phin=ph, dx=np.stack(dx, 1))
        return out

    def instantaneous(self):
        """A_inf, B_inf, C_inf of the discrete system: phi_B -> i omega
        phi_B1 + phi_B0 as omega -> infinity with phi_B1 = phi_inf[n_k] (phi_F =
        0) and phi_B0 the response to m_k. Without the wall closure phi_B0 =
        phi_inf[m_k] (phi_F = 0). With the wall closure the O(1) free-surface
        potential does not vanish: the O(omega^2) balance of the closure system
        (ForwardSpeedLTI.set_operator) gives phi_F0 = 2 U C E phi_B1 (zero in the
        continuum, where the hull trace of the phi = 0 solution vanishes; a
        discretisation term otherwise), so phi_B0 solves the problem with that
        free-surface potential (stage note 10.8.2). A_inf is unchanged; B_inf
        (both forms) is the exact limit of the discrete system (needed for the
        Kramers-Kronig and Cummins consistency); C_inf (reference only, not used
        by the plant law) is evaluated with phi_B0 and omits the next-order term
        of the closure. D-form and OT-form."""
        s = self.s
        a = s.body.area
        m = len(self.modes)
        nmat = np.stack([self.mv[k][0] for k in self.modes], 1)
        mmat = np.stack([self.mv[k][1] for k in self.modes], 1)
        sn = s.solve_inf(nmat)["phi_B"]
        if s.U != 0 and getattr(s, "_wall_on", False):
            phiF0 = 2.0 * s.U * (s.wall["C"] @ (s.wall["E"] @ sn))
            sm = s.solve_inf(mmat, phiF=phiF0)["phi_B"]
        else:
            sm = s.solve_inf(mmat)["phi_B"] if s.U != 0 else np.zeros_like(sn)
        dn = np.stack([hull_dx(s.body, sn[:, c], nmat[:, c], s.hull_shape) for c in range(m)], 1) if s.hull_shape else 0 * sn
        dm = np.stack([hull_dx(s.body, sm[:, c], mmat[:, c], s.hull_shape) for c in range(m)], 1) if s.hull_shape else 0 * sm
        rho, U = s.rho, s.U
        A = np.zeros((m, m))
        B = np.zeros((m, m))
        C = np.zeros((m, m))
        Bot = np.zeros((m, m))
        Cot = np.zeros((m, m))
        for r in range(m):
            nj, mj = nmat[:, r], mmat[:, r]
            for c in range(m):
                A[r, c] = -rho * 2 * np.sum(sn[:, c] * nj * a)
                B[r, c] = -rho * 2 * np.sum((sm[:, c] - U * dn[:, c]) * nj * a)
                C[r, c] = rho * U * 2 * np.sum(dm[:, c] * nj * a)
                Bot[r, c] = -rho * 2 * np.sum((sm[:, c] * nj - sn[:, c] * mj) * a)
                Cot[r, c] = rho * 2 * np.sum(sm[:, c] * mj * a)
        return dict(A=A, B=B, C=C, B_ot=Bot, C_ot=Cot)

    def rad_exc(self, omega0, beta):
        """Radiation coefficients at the encounter frequency of the wave
        (omega0, beta) and the excitation, with ONE LU (four right-hand
        sides). Returns (radiation dict as radiation(), excitation dict as
        excitation())."""
        s = self.s
        g = s.g
        k = omega0 ** 2 / g
        we = omega0 - s.U * k * np.cos(beta)
        a = s.body.area
        ph = np.stack([1j * we * self.mv[kk][0] + self.mv[kk][1] for kk in self.modes], 1)
        phI, dnI, _ = incident(s.body.c, s.body.n, omega0, beta, g)
        sol = s.solve_freq(we, np.concatenate([ph, (-dnI)[:, None]], 1))
        m = len(self.modes)
        rsol = dict(phi_B=sol["phi_B"][:, :m], phi_F=sol["phi_F"][:, :m], sigma=sol["sigma"][:, :m])
        rad = self._coeffs(we, rsol, ph)
        phD = sol["phi_B"][:, m]
        dxD = hull_dx(s.body, phD, -dnI, s.hull_shape) if s.hull_shape else 0 * phD
        X, Xfk, Xd, Xdot = [], [], [], []
        for j in self.modes:
            nj, mj = self.mv[j]
            fk = s.rho * 2 * np.sum(1j * omega0 * phI * nj * a)
            xd = s.rho * 2 * np.sum((1j * we * phD - s.U * dxD) * nj * a)
            Xfk.append(fk)
            Xd.append(xd)
            Xdot.append(s.rho * 2 * np.sum(phD * (1j * we * nj - mj) * a))
            X.append(fk + xd)
        exc = dict(omega0=omega0, omega_e=we, k=k, beta=beta, X=np.array(X), X_fk=np.array(Xfk), X_d=np.array(Xd),
                   X_d_ot=np.array(Xdot))
        return rad, exc

    def excitation(self, omega0, beta, want_phi=False):
        """Excitation X_j (D-form; FK exact) for unit wave amplitude at
        absolute frequency omega0 and heading beta (pi = head seas); returns
        dict(omega_e, X, X_fk, X_d, X_d_ot, sol...)."""
        s = self.s
        g = s.g
        k = omega0 ** 2 / g
        we = omega0 - s.U * k * np.cos(beta)
        a = s.body.area
        phI, dnI, dxI = incident(s.body.c, s.body.n, omega0, beta, g)
        sol = s.solve_freq(we, (-dnI)[:, None])
        phD = sol["phi_B"][:, 0]
        dxD = hull_dx(s.body, phD, -dnI, s.hull_shape) if s.hull_shape else 0 * phD
        X, Xfk, Xd, Xdot = [], [], [], []
        for j in self.modes:
            nj, mj = self.mv[j]
            fk = s.rho * 2 * np.sum(1j * omega0 * phI * nj * a)
            xd = s.rho * 2 * np.sum((1j * we * phD - s.U * dxD) * nj * a)
            xdo = s.rho * 2 * np.sum(phD * (1j * we * nj - mj) * a)
            Xfk.append(fk)
            Xd.append(xd)
            Xdot.append(xdo)
            X.append(fk + xd)
        out = dict(omega0=omega0, omega_e=we, k=k, beta=beta, X=np.array(X), X_fk=np.array(Xfk), X_d=np.array(Xd),
                   X_d_ot=np.array(Xdot))
        if want_phi:
            out.update(sol=sol, phi_I=phI, dn_I=dnI, dx_D=dxD)
        return out


def green_terms(s, phi_a, phin_a, sig_a, phiF_a, phi_b, phin_b, sig_b, phiF_b):
    """Discrete Green's second identity between two solutions a, b (both
    sides of the hull via the y image): body sum int_S (phi_a dphi_b/dn -
    phi_b dphi_a/dn) dS and free-surface sum int_F (phi_a phi_b,z - phi_b
    phi_a,z) dA with phi_z = -sigma (stage note 10.2: equal for the exact
    solutions)."""
    aB, aF = s.body.area, s.fs.area
    body = 2 * np.sum((phi_a * phin_b - phi_b * phin_a) * aB)
    fsum = 2 * np.sum((-phiF_a * sig_b + phiF_b * sig_a) * aF)
    return body, fsum


def waterline_fs_term(s, phi_a, phi_b, dx_a, dx_b, U, omega, hull, sink=0.0, trim=0.0):
    """Continuous reduction of the free-surface Green integral to the
    waterline (stage note 10.2): int_F (phi_a psi_z - psi phi_a,z) dA =
    -(1/g) oint [2 i omega U phi_a psi + U^2 (phi_a psi_x - psi phi_a,x)] w'(x) dx
    for solution a at +U and psi = b at -U (both sides), with the waterline
    values extrapolated from the hull's top rows and w = y_wl(x)."""
    xa, fa = hull_waterline_values(s.body, phi_a, s.hull_shape)
    _, fb = hull_waterline_values(s.body, phi_b, s.hull_shape)
    _, ga = hull_waterline_values(s.body, dx_a, s.hull_shape)
    _, gb = hull_waterline_values(s.body, dx_b, s.hull_shape)
    xs, xb = hull.waterline_ends(sink, trim)
    # trapezoid over the station centroid x with w' from the hull form; the integrand at the two waterline
    # ends is extrapolated linearly from the two nearest stations (w' one-sided inside the waterline) instead
    # of being set to 0 (B5 review: w' is nonzero at pointed ends, e.g. +-0.24 for the Wigley I)
    Gv = 2j * omega * U * fa * fb + U * U * (fa * gb - fb * ga)
    order = np.argsort(xa)
    xo, Go = xa[order], Gv[order]
    G_s = Go[0] + (xs - xo[0]) * (Go[1] - Go[0]) / (xo[1] - xo[0])
    G_b = Go[-1] + (xb - xo[-1]) * (Go[-1] - Go[-2]) / (xo[-1] - xo[-2])
    xx = np.concatenate([[xs], xo, [xb]])
    wp = waterline_slope(hull, xx, sink, trim)
    yy = np.concatenate([[G_s], Go, [G_b]]) * wp
    return -(2.0 / s.g) * np.trapezoid(yy, xx)


def waterline_slope(hull, x, sink=0.0, trim=0.0):
    """w'(x) of the waterline half-breadth, central difference inside, one-sided
    (into the waterline) within 2e-6 L of its ends."""
    x = np.asarray(x, float)
    xs, xb = hull.waterline_ends(sink, trim)
    hb = 1e-6 * hull.L
    lo = np.clip(x - hb, xs, xb)
    hi = np.clip(x + hb, xs, xb)
    hi = np.where(hi - lo < 1.5 * hb, np.clip(lo + 2 * hb, xs, xb), hi)
    lo = np.where(hi - lo < 1.5 * hb, np.clip(hi - 2 * hb, xs, xb), lo)
    return (hull.waterline(hi, sink, trim) - hull.waterline(lo, sink, trim)) / (hi - lo)


# ================================================================ time domain
class TimeDomain:
    """Explicit RK4 integration of the SAME semi-discrete LTI system
    (module docstring 3) with prescribed rigid-body motion of the modes of a
    Coefficients object. Outputs per step: forces (D-form and OT-form), and
    the terms of the pseudo-energy balance of the linear NK problem (stage note
    10.2):
        H = E_kin + E_pot - U M,  E_kin = (rho/2)[int_F phi phi_z - int_S phi phi_n],
        E_pot = (rho g/2) int_F zeta^2,  M = rho int_F zeta phi_x,
        dH/dt = P_body + P_wl + P_beach,
        P_body = -rho int_S phi_t phi_n,
        P_wl   = rho U oint_C zeta phi_t n_x dl  (= -rho U int zeta phi_t w'(x) dx per side),
        P_beach = rho int nu phi sigma - rho g int nu zeta^2 + rho U int [nu zeta phi_x + zeta (nu phi)_x]
    (both sides via the y image; the upwind dissipation of D is not in the
    identity, so the residual measures it together with the BIE and RK4 errors)."""

    def __init__(self, co, hull=None, sink=0.0, trim=0.0):
        self.co, self.s = co, co.s
        self.hull, self.sink, self.trim = hull, sink, trim
        s = self.s
        s.dtn_factor()
        dev = s.dev
        self.n = torch.as_tensor(np.stack([co.mv[k][0] for k in co.modes], 1), device=dev)
        self.m = torch.as_tensor(np.stack([co.mv[k][1] for k in co.modes], 1), device=dev)
        self.aF = torch.as_tensor(s.fs.area, device=dev)
        self.aB = torch.as_tensor(s.body.area, device=dev)
        self.nuF = torch.as_tensor(s.nu, device=dev)
        self._D = _ts(s.D, dev)

    def _rhs(self, zeta, phiF, pn, pnd=None):
        return self.s.rhs_td(zeta, phiF, pn, pnd)

    def run(self, motion, dt, nsteps, zeta0=None, phi0=None, energy=True, every=1):
        """motion(t) -> (x, xd, xdd) arrays over the modes (or None: body at
        rest). Returns dict of time series (numpy)."""
        s, dev = self.s, self.s.dev
        NF = s.NF
        zeta = torch.zeros((NF, 1), dtype=torch.float64, device=dev) if zeta0 is None else \
            torch.as_tensor(np.asarray(zeta0, float).reshape(NF, 1), device=dev)
        phiF = torch.zeros_like(zeta) if phi0 is None else torch.as_tensor(np.asarray(phi0, float).reshape(NF, 1), device=dev)
        mm = len(self.co.modes)

        def pn_of(t, deriv=False):
            if motion is None:
                return torch.zeros((s.NB, 1), dtype=torch.float64, device=dev)
            x, xd, xdd = (torch.as_tensor(np.asarray(v, float), device=dev) for v in motion(t))
            if deriv:
                return (self.n @ xdd + self.m @ xd)[:, None]
            return (self.n @ xd + self.m @ x)[:, None]
        out = {k: [] for k in ("t", "F_D", "F_OT", "H", "Ek", "Ep", "Mx", "P_body", "P_wl", "P_beach", "x")}

        def record(t, zeta, phiF):
            pn = pn_of(t)
            pnd = pn_of(t, True)
            dz, dp, sig = self._rhs(zeta, phiF, pn, pnd)
            sigd = s.sigma_of(dp, pnd)
            phB = s.phiB_of(sig, pn)
            phBd = s.phiB_of(sigd, pnd)
            phB_n, phBd_n, pn_n = phB[:, 0].cpu().numpy(), phBd[:, 0].cpu().numpy(), pn[:, 0].cpu().numpy()
            dxB = hull_dx(s.body, phB_n, pn_n, s.hull_shape) if s.hull_shape else 0 * phB_n
            a = s.body.area
            FD, FO = [], []
            for kk, j in enumerate(self.co.modes):
                nj, mj = self.co.mv[j]
                FD.append(s.rho * 2 * np.sum((phBd_n - s.U * dxB) * nj * a))
                FO.append(s.rho * 2 * np.sum((phBd_n * nj - mj * phB_n) * a))
            out["t"].append(t)
            out["F_D"].append(FD)
            out["F_OT"].append(FO)
            out["x"].append(np.asarray(motion(t)[0], float) if motion is not None else np.zeros(mm))
            if energy:
                rho, g, U = s.rho, s.g, s.U
                Ek = 0.5 * rho * 2 * (-(self.aF * phiF[:, 0] * sig[:, 0]).sum() - (self.aB * phB[:, 0] * pn[:, 0]).sum())
                Ep = 0.5 * rho * g * 2 * (self.aF * zeta[:, 0] ** 2).sum()
                phx = torch.sparse.mm(self._D, phiF)[:, 0]
                if s._wall_on:
                    phx = phx + torch.sparse.mm(s._C, s.wall_phi(sig, pn))[:, 0]
                Mx = rho * 2 * (self.aF * zeta[:, 0] * phx).sum()
                Pb = -rho * 2 * (self.aB * phBd[:, 0] * pn[:, 0]).sum()
                nphx = torch.sparse.mm(self._D, (self.nuF * phiF[:, 0])[:, None])[:, 0]
                Pbe = rho * 2 * ((self.aF * self.nuF * phiF[:, 0] * sig[:, 0]).sum()
                                 - g * (self.aF * self.nuF * zeta[:, 0] ** 2).sum()
                                 + U * (self.aF * zeta[:, 0] * (self.nuF * phx + nphx)).sum())
                Pw = 0.0
                if self.hull is not None and s.hull_shape:
                    xa, ft = hull_waterline_values(s.body, phBd_n, s.hull_shape)
                    _, fx = hull_waterline_values(s.body, dxB, s.hull_shape)
                    zw = -(ft - U * fx) / g
                    hb = 1e-6 * self.hull.L
                    wp = (self.hull.waterline(xa + hb, self.sink, self.trim)
                          - self.hull.waterline(xa - hb, self.sink, self.trim)) / (2 * hb)
                    xs, xb = self.hull.waterline_ends(self.sink, self.trim)
                    o = np.argsort(xa)
                    Pw = -rho * U * 2 * np.trapezoid(np.concatenate([[0.0], (zw * ft * wp)[o], [0.0]]),
                                                     np.concatenate([[xs], xa[o], [xb]]))
                out["Ek"].append(float(Ek))
                out["Ep"].append(float(Ep))
                out["Mx"].append(float(Mx))
                out["H"].append(float(Ek + Ep - U * Mx))
                out["P_body"].append(float(Pb))
                out["P_wl"].append(float(Pw))
                out["P_beach"].append(float(Pbe))
        t = 0.0
        record(t, zeta, phiF)
        wall = s._wall_on
        for it in range(nsteps):
            p1 = pn_of(t)
            q1 = pn_of(t, True) if wall else None
            k1z, k1p, _ = self._rhs(zeta, phiF, p1, q1)
            p2 = pn_of(t + 0.5 * dt)
            q2 = pn_of(t + 0.5 * dt, True) if wall else None
            k2z, k2p, _ = self._rhs(zeta + 0.5 * dt * k1z, phiF + 0.5 * dt * k1p, p2, q2)
            k3z, k3p, _ = self._rhs(zeta + 0.5 * dt * k2z, phiF + 0.5 * dt * k2p, p2, q2)
            p4 = pn_of(t + dt)
            q4 = pn_of(t + dt, True) if wall else None
            k4z, k4p, _ = self._rhs(zeta + dt * k3z, phiF + dt * k3p, p4, q4)
            zeta = zeta + (dt / 6) * (k1z + 2 * k2z + 2 * k3z + k4z)
            phiF = phiF + (dt / 6) * (k1p + 2 * k2p + 2 * k3p + k4p)
            t += dt
            if (it + 1) % every == 0:
                record(t, zeta, phiF)
            if not torch.isfinite(zeta).all():
                raise FloatingPointError(f"time integration diverged at t = {t:g}")
        res = {k: np.array(v) for k, v in out.items() if len(v)}
        res["zeta_end"] = zeta[:, 0].cpu().numpy()
        res["phi_end"] = phiF[:, 0].cpu().numpy()
        return res


# ============================================================ steady (omega = 0)
def steady_forces(s, phiB, hull, sink=0.0, trim=0.0, cg=(0.0, 0.0)):
    """Steady NK forces from the omega = 0 solution of THIS system with the
    body condition dphi/dn = U n_x, by the B4 definitions (piercing.py,
    stage note 9.2): Bernoulli pressure rho (U phi_x - |grad phi|^2/2) from the
    structured hull gradient, d'Alembert-corrected with the rigid-lid double
    body on the same mesh, waterline term (rho g/2) oint zeta^2 n_x dl with the
    hull-side zeta = (U/g) phi_x at z = 0; R_near = R_corr - R_wl. Used for the
    cross-check of the omega -> 0 limit with the B4 steady solver."""
    from .piercing import structured_gradient
    from .bie import neumann_green
    U, g, rho = s.U, s.g, s.rho
    pan = s.body
    pn = U * pan.n[:, 0]
    phiB = np.real(np.asarray(phiB)).reshape(-1)
    op = ImageOperator(pan, axes=(1, 2), device=s.dev, parities=[(1, 1)])
    phidb = neumann_green(op, pn, (1, 1), device=s.dev)
    del op

    def pres(ph):
        gr = structured_gradient(pan, ph, pn, s.hull_shape)
        return rho * (U * gr[:, 0] - 0.5 * np.einsum("ij,ij->i", gr, gr)), gr
    p, gr = pres(phiB)
    pdb, _ = pres(phidb)
    cgE = hull.to_earth(np.array(cg[0]), np.array(0.0), np.array(cg[1]), sink, trim)
    r = pan.c - cgE
    F = -2 * np.sum((p * pan.area)[:, None] * pan.n, axis=0)
    Fdb = -2 * np.sum((pdb * pan.area)[:, None] * pan.n, axis=0)
    My = -2 * np.sum(p * pan.area * (r[:, 2] * pan.n[:, 0] - r[:, 0] * pan.n[:, 2]))
    xa, px = hull_waterline_values(pan, gr[:, 0], s.hull_shape)
    zeta = U / g * px
    hb = 1e-6 * hull.L
    yp = (hull.waterline(xa + hb, sink, trim) - hull.waterline(xa - hb, sink, trim)) / (2 * hb)
    o = np.argsort(xa)
    xs, xb = hull.waterline_ends(sink, trim)
    f = (zeta ** 2 * (-yp))[o]
    Rwl = float(rho * g * np.trapezoid(np.concatenate([[f[0]], f, [f[-1]]]), np.concatenate([[xs], xa[o], [xb]])))
    Rc = float(-(F[0] - Fdb[0]))
    return dict(Fx=float(F[0]), Fz=float(F[2]), My=float(My), R_corr=Rc, R_wl=Rwl, R_near=Rc - Rwl,
                dalembert=float(-Fdb[0]), Fz_db=float(Fdb[2]))
