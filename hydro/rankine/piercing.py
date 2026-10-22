"""
Steady Neumann-Kelvin flow about a SURFACE-PIERCING hull with the shifted
Numerov free-surface operator (Track B step B4; port design
WAVEBEM_PORT_DESIGN.md P-2/P-3, P-7, P-9, P-10; stage note
RANKINE_STAGE1_2026-10-06.md section 9). Linear free surface on z = 0 about
the uniform stream, exact hull below the calm waterplane at the given sinkage
and trim (hull.py), free sinkage and trim by a 2 x 2 Newton iteration with
exact hydrostatics, and a transom stern with a DRY (derived) or WET
(WaveBEM-style, registered empirical) branch.

1. Free-surface grid: lattice + body-fitted band (why). The B3 operator
   marches along streamwise grid lines: the radiation condition comes from
   two conditions at each line's upstream end (shifted.py). A lattice line at
   |y| < B/2 is cut by the hull: its part behind the hull would need two new
   "initial" conditions at the aft waterline, which the Neumann-Kelvin problem
   does not supply (the free waves next to the hull are carried along the hull
   side, not along a lattice line). The lines are therefore made continuous: in
   the hull's x-range the first J lines are bent around the waterline,
       Y_j(x) = y_wl(x) (1 - j/J) + j dy      (line edges, j = 0..J),
   so line 0 starts at the waterline and line J is the lattice line J dy (a
   band compressed by at most 1 - y_max/(J dy) >= 1/2 for J >= 2 y_max/dy).
   Columns stay uniform in x (dx = L_wl/M, the waterline ends are column
   edges), so every line has uniform x-spacing and the Numerov rows keep their
   form. Outside the band the grid is the B3 lattice (FFT products); band
   panels are ordinary flat quadrilaterals with dense influences.

2. Rows in the band. The Neumann-Kelvin condition k phi_xx + phi_z = 0 (k =
   U^2/g) needs phi_xx at FIXED y, not along the bent line. Row of the line
   point (c, j) (centre c = r - 1, shifted as in shifted.py):
       a0 sig(c+1; y_c) + a1 sig(c; y_c) + a2 sig(c-1; y_c)
         = k w_c [ (phi(c-1; y_c) - 2 phi(c; y_c) + phi(c+1; y_c))/dx^2
                   - nu_c (phi(c-1; y_c) - phi(c+1; y_c))/(2 dx) ]
   with f(c +- 1; y_c) the value at the neighbouring column at the SAME y as
   the centre, by quadratic Lagrange interpolation across the lines of that
   column, centred on the nearest line (PHI_MODE "interp", production). The
   y-image supplies a mirror point where the lowest line touches y = 0; where
   it touches the hull a VIRTUAL point behind the wall (_Columns, mode
   "wall"; wall conditions with the section-curvature and flare terms,
   corrected after the B4 review, stage note 9.7.4) keeps the stencil
   centred: a one-sided stencil there gave the row
   a complex symbol and short diverging waves grew along the afterbody
   (exploration B4-E3, stage note 9.2/9.3). Where the neighbouring line is at
   the same y (the whole lattice) the weights are exactly (1): the rows are
   the B3 rows (check H2). Interpolation error O(dy^3 f_yyy) per value.
   PHI_MODE "exact" (phi at the targets from the representation, extra
   evaluation points) was tried and rejected (unstable at Fn 0.4 on fine
   grids, B4-E7). Known limitation (check H9, open): in the long-wave limit
   (k0 L <~ 1.5) the interpolation deficit |I| < 1 splits the neutral double
   root of the march and the rows grow behind the stern (stage note 9.2).

3. Box correction: as in shifted.py along lines; across lines
   -(w_p^2/24) sig_yy with the panel width w_p and a non-uniform three-point
   sig_yy (reduces to the B3 form on the lattice; the even reflection at the
   hull wall and at y = 0); none on the first/last column of a line and on
   the outer line.

4. Hull (Green rows) with rigid-lid and port/starboard images, as shifted.py;
   body condition dphi/dn = U n_x on the hull (n out of the body, earth frame).
   Unknowns phi_B (hull, and transom face if any) and sig (all free-surface
   panels: the nx x ny logical lattice positions -- lattice or band panels --
   then the transom-wake lines). The free-surface potential
   4 pi phi_F = D_FB phi_B - S_FB dphi/dn_B - S_FF sig' with S_FF = FFT
   lattice (lattice panels at lattice points) + dense blocks (any pair with a
   band or transom panel).

5. Transom stern (P-9, P-10). Lines behind the transom (|y| < y_T, new lines
   j = -K..-1) START at the transom edge x_s: the two conditions of the
   marching problem are physical there.
   DRY branch (derived, G2): the free surface leaves the edge smoothly
   (Raven 1996; Vanden-Broeck 1980 for the 2-D stern wave): (D1) elevation
   pinned at the edge depth, zeta(x_s, y) = z_T(y) -- with the linear dynamic
   condition g zeta = U phi_x this is the same statement as atmospheric
   pressure at the edge (Bernoulli |grad Phi|^2 = U^2 - 2 g z_T linearised);
   (D2) tangential detachment, zeta_x(x_s, y) = s_b(y), the buttock slope of
   the hull at the edge (earth frame), i.e. sig = U zeta_x = U s_b (kinematic
   condition). Discretely (s = downstream distance from the edge, h = dx):
       D2 (row of sig_f):     (3 sig_f - sig_{f+1})/2 = U s_b      (linear extrapolation to s = 0)
       D1 (row of sig_{f+1}): (sig_f + sig_{f+1})/2 - k (phi_{f+1} - phi_f)/h^2 = k g z_T/(U h)
   (quadratic fit through phi_f (s = h/2), phi_{f+1} (3h/2) with
   dphi/ds(0) = -g z_T/U, condition at s = h). The transom face is closed in
   the linearised model (the Green identity needs a closed body) and carries
   the edge value of the same condition, phi_x = g z_T(y)/U (transferred
   hollow: in the linear transfer the face shrinks to the edge; the closure is
   a modelling choice, its sensitivity is a registered information row). The
   face is dry: no force on it.
   WET branch (WaveBEM W-33/W-34, re-implemented): the same rows plus a
   dead-water pressure patch on the free surface behind the transom,
   p_a = -(1 - eta) rho g |z_h(s, y)| (suction lifting the surface by
   (1 - eta)|z_h|), z_h the parabolic hollow from z_T with the buttock slope to
   0 at s = l_h |z_T| (WaveBEM's form, l_h = 5), eta the dry fraction of
   Doctors' regression (EMPIRICAL_REGISTER_RANKINE.md). Linear conditions:
   g zeta = U phi_x - p_a/rho, so k phi_xx + phi_z = (U/(rho g)) p_a,x and
   sig = k phi_xx + f, f = (1 - eta) U d|z_h|/dx; every row is written for
   sig - f (rhs + A f). The virtual face carries the hydrostatic pressure
   (1 - eta) rho g |z| (W-34), which enters the equilibrium (P-10; WaveBEM
   subtracts it only in post-processing). eta = 1 reproduces the dry branch
   exactly. The two branches are tabulated separately; nothing weights between
   them (G1).

6. Forces. Dynamic pressure p = rho (U phi_x - |grad phi|^2/2) on the hull
   panels; gradient from the STRUCTURED hull mesh (index-space differences,
   structured_gradient; the least-squares fit of steady.py is ill-conditioned
   on the thin top row, B4-E1). Wave resistance of the linear model
       R_near = -int_hull [p(phi) - p(phi_db)] n_x dS - (rho g/2) oint zeta^2 n_x dl
   (d'Alembert-corrected pressure term as for the spheroid; the waterline term
   is SUBTRACTED: the x-momentum balance of the linear model, with fluid
   crossing z = 0 and phi_z = -k phi_xx there, gives R_p = (rho g/2) oint
   zeta^2 n_x dl + R_waves, stage note 9.2; confirmed by the wave cut, B4-E5),
   zeta the hull-side elevation (U/g) phi_x at z = 0 (waterline_profile).
   Far field: transverse wave cut (cut_from_samples), calibrated on the exact
   Kelvin field (check H7). The transom closure lets fluid cross the face; the
   momentum identity of the linear model then has face and edge terms
   (PiercingNK.transom_identity, stage note 9.7.3, check T3).

7. Solver: GMRES (krylov.py) with a block preconditioner: hull block by
   dense LU, lattice by the B3 y-circulant operator, band and transom-line
   unknowns by an LU of their own block of the free-surface rows (added for
   the transom grid, B4-E9).
"""
import warnings

import numpy as np
import scipy.sparse as sp
import torch

from .bie import ImageOperator, neumann_green
from .influence import influence
from .krylov import gmres
from .panels import Panels
from .shifted import FSLattice, blend_weights, lattice_zmax, _torch_sparse
from .steady import surface_gradient

ENTRY_WL, BEACH_WL, BEACH_DECAY = 1.0, 2.0, 6.0
GHOST_MODE = "wall"      # hull-wall virtual point of the lateral interpolation of phi (outputs; rows with PHI_MODE "interp")
PHI_MODE = "interp"      # band rows: phi at the target points from the representation ("exact") or interpolated ("interp")


# ===================================================================== grid
def build_grid(hull, sink, trim, lam, n_per_lam, m_body=60, mult=(1.0, 1.0, 1.0, 1.0, 1.0),
               aspect=1.0, J_mult=2.0, up=None, down=None, lat=None, transom=None):
    """Free-surface grid for a surface-piercing hull (module docstring 1).

    Columns: dx = L_wl / M, M = max(ceil(L_wl/(lam/n_per_lam)), m_body), so
    dx <= lam/n and dx <= L_wl/m_body, and the waterline ends x_s, x_b are
    column edges. Registered domain (stage note 1.1, with f = 0): upstream
    1.5 L + entry lam ahead of the bow, downstream max(2.5 lam, ...) + beach
    2 lam behind the stern, lateral 0.8 L + 0.5 lam; mult scales (upstream,
    downstream, lateral, entry, beach). Lines: dy = aspect dx; band of J lines
    (J >= J_mult y_max/dy, >= 2). transom: None (pointed stern) or True:
    lines -K..-1 behind the transom (K = round(y_T/dy) >= 1)."""
    mu, md, ml, me, mb = mult
    L = hull.L
    xs, xb = hull.waterline_ends(sink, trim)
    Lwl = xb - xs
    M = max(int(np.ceil(Lwl / (lam / n_per_lam) - 1e-9)), int(m_body))
    dx = Lwl / M
    upx = (up if up is not None else 1.5 * L) * mu + me * ENTRY_WL * lam
    dnx = (down if down is not None else 2.5 * lam) * md + mb * BEACH_WL * lam
    latx = (lat if lat is not None else 0.8 * L + 0.5 * lam) * ml
    iu = int(np.ceil(upx / dx - 1e-9))
    idn = int(np.ceil(dnx / dx - 1e-9))
    nx = iu + M + idn
    xe = xb + iu * dx - dx * np.arange(nx + 1)
    xe[iu] = xb
    xe[iu + M] = xs
    i_bow, i_stern = iu, iu + M
    dy = aspect * dx
    ny = int(np.ceil(latx / dy - 1e-9))
    ywl_e = hull.waterline(xe, sink, trim)
    ywl_e[:i_bow] = 0.0
    ywl_e[i_stern + 1:] = 0.0
    is_tr = bool(transom) if transom is not None else bool(getattr(hull, "transom", False))
    yT = float(hull.waterline(np.array([xs]), sink, trim)[0]) if is_tr else 0.0
    if not is_tr:
        ywl_e[i_stern] = 0.0
        ywl_e[i_bow] = 0.0
    else:
        ywl_e[i_bow] = 0.0
        ywl_e[i_stern:] = yT
    ymax = float(ywl_e.max())
    J = max(2, int(np.ceil(J_mult * ymax / dy - 1e-9))) if ymax > 0 else 0
    if J >= ny:
        raise ValueError("lateral domain too narrow for the band")
    K = max(1, int(round(yT / dy))) if is_tr else 0

    def Ye(i, j):
        """Line-edge y at column edge i, line edge j (j = -K..ny)."""
        if j < 0:
            return yT * (j + K) / K
        if j <= J:
            return ywl_e[i] * (1.0 - j / J) + j * dy if J else j * dy
        return j * dy

    # logical lattice positions (j = 0..ny-1, all columns), then transom lines
    corners, ij, kind = [], [], []
    for j in range(ny):
        for i in range(nx):
            y00, y01 = Ye(i, j), Ye(i, j + 1)
            y10, y11 = Ye(i + 1, j), Ye(i + 1, j + 1)
            corners.append([[xe[i], y00, 0.0], [xe[i + 1], y10, 0.0], [xe[i + 1], y11, 0.0], [xe[i], y01, 0.0]])
            ij.append((i, j))
            band = j < J and ywl_e[i] + ywl_e[i + 1] > 0
            kind.append(1 if band else 0)
    for j in range(-K, 0):
        for i in range(i_stern, nx):
            y00, y01 = Ye(i, j), Ye(i, j + 1)
            y10, y11 = Ye(i + 1, j), Ye(i + 1, j + 1)
            corners.append([[xe[i], y00, 0.0], [xe[i + 1], y10, 0.0], [xe[i + 1], y11, 0.0], [xe[i], y01, 0.0]])
            ij.append((i, j))
            kind.append(2)
    q = np.array(corners, float)
    fs = Panels(q, np.ones(len(q), int))
    if np.mean(fs.n[:, 2]) > 0:
        fs = Panels(q[:, ::-1], np.ones(len(q), int))
    assert np.all(fs.n[:, 2] < 0)
    ij = np.array(ij, int)
    kind = np.array(kind, int)
    P = np.zeros((len(q), 3))
    P[:, 0] = 0.5 * (q[:, 0, 0] + q[:, 1, 0])
    P[:, 1] = q[:, :, 1].mean(1)
    width = 0.5 * (q[:, 3, 1] + q[:, 2, 1]) - 0.5 * (q[:, 0, 1] + q[:, 1, 1])
    # ramp and beach per column (functions of x only)
    xm = 0.5 * (xe[:-1] + xe[1:])
    ywl_mid = np.zeros(nx)
    ywlp_mid = np.zeros(nx)
    hz_mid, hzz_mid, hxz_mid = np.zeros(nx), np.zeros(nx), np.zeros(nx)
    hb = 1e-6 * L
    sl = slice(i_bow, i_stern)
    ywl_mid[sl] = hull.waterline(xm[sl], sink, trim)
    ywlp_mid[sl] = (hull.waterline(xm[sl] + hb, sink, trim) - hull.waterline(xm[sl] - hb, sink, trim)) / (2 * hb)
    # hull-wall derivatives at the waterline (earth frame, from below z = 0) for the wall conditions of the
    # virtual points (_Columns, stage note 9.7): H_z (flare), H_zz (section curvature), H_xz
    _, _, hz_mid[sl], hzz_mid[sl], hxz_mid[sl] = hull.wall_derivatives(xm[sl], sink, trim)
    x_up, x_dn = xe[0], xe[-1]
    entry = me * ENTRY_WL * lam
    s = np.clip((x_up - xm) / entry, 0.0, 1.0) if entry > 0 else np.ones(nx)
    w_col = s ** 3 * (10 - 15 * s + 6 * s * s) if entry > 0 else np.ones(nx)
    Lb = mb * BEACH_WL * lam
    sb = np.clip((x_dn + Lb - xm) / Lb, 0.0, 1.0) if Lb > 0 else np.zeros(nx)
    nu_col = 3.0 * BEACH_DECAY / Lb * sb * sb if Lb > 0 else np.zeros(nx)
    # index maps
    index = {(int(a), int(b)): k for k, (a, b) in enumerate(ij)}
    grid = dict(nx=nx, ny=ny, dx=dx, dy=dy, xe=xe, i_bow=i_bow, i_stern=i_stern, J=J, K=K, yT=yT,
                transom=is_tr, ywl_e=ywl_e, ywl_mid=ywl_mid, ywlp_mid=ywlp_mid, hz_mid=hz_mid, hzz_mid=hzz_mid,
                hxz_mid=hxz_mid, M=M, lam=lam, n_per_lam=n_per_lam, n_eff=lam / dx,
                w_col=w_col, nu_col=nu_col, entry=entry, beach=Lb, x_s=xs, x_b=xb)
    return dict(fs=fs, P=P, ij=ij, kind=kind, width=width, index=index, grid=grid)


# ================================================================ operators
class _Columns:
    """Per column: the free-surface points sorted by y, with one extra
    virtual point below the lowest line: the MIRROR of the lowest point in
    y = 0 where the lowest line touches the centre plane (port/starboard
    image, exact), or a GHOST behind the hull wall where it touches the
    hull (band columns). The ghost (exploration B4-E3/E4) lets the lateral
    interpolation of the hull-adjacent line stay centred: a one-sided
    stencil there gives the row a complex symbol (asymmetric offsets at the
    two neighbouring columns), which amplified short diverging waves along
    the afterbody and behind the stern.

    Ghost at y_g = y_w - d (y_w the waterline, d = y_0 - y_w, y_0 the line-0
    point), from the wall conditions at z = 0 on the hull side y = H(x, z)
    (earth frame at the attitude; outward normal N = (-H_x, 1, -H_z); stage
    note 9.7, corrected after the B4 review -- the B4 text claimed
    d(sig)/dn = 0 because "the wall normal does not depend on z", which is
    wrong for any hull whose sections curve below the waterline):
      body condition  (phi_x - U)(-H_x) + phi_y - H_z phi_z = 0, so with
      sig = -phi_z:   phi_y = -H_x (U - phi_x) - H_z sig;
      its derivative along the hull in z (tangent (0, H_z, 1)) at z = 0, for
      H_z = 0 there (Wigley, Wigley-transom):
                      sig_y = H_x sig_x + H_xz (U - phi_x) + H_zz sig
      (Wigley midship H_zz = -B/T^2 = -25.6/L). With H_z != 0 (flare) the
      same derivative also contains H_z (2 phi_yy + phi_xx) - H_x H_z phi_xy
      and a factor 1/(1 - H_z^2); these terms need lateral second
      derivatives at the wall and are NOT included (check H13 measures
      their effect on a flared hull; first-order consistent only for
      H_z = 0).
    First-order Taylor across 2 d with backward differences along line 0
    (columns i-1 = upstream, i: the marching structure is kept),
    a = 2 d H_x/dx, b = 2 d H_xz/dx:
      sig_g = (1 + a - 2 d H_zz) sig_0,i - a sig_0,i-1 + b (phi_0,i-1 - phi_0,i) - 2 d H_xz U
      phi_g = (1 + a) phi_0,i - a phi_0,i-1 + 2 d H_z sig_0,i + 2 d H_x U.
    So the two fields have SEPARATE ghost lists (mode "wall"); each virtual or
    real point is (own-field list of (index, coeff), other-field list, constant
    times U). Modes: "wall" (production after the B4 review), "wall_vertical"
    (the B4 ghost: sig_y = H_x sig_x, phi_y = -H_x (U - phi_x); comparison
    only), "wall_const", "even" (exploration B4-E8)."""

    def __init__(self, G, mode=None):
        g = G["grid"]
        self.dy = g["dy"]
        self.dx = g["dx"]
        mode = GHOST_MODE if mode is None else mode
        self.mode = mode
        self.cols = []
        self.ghost_curv = 0.0          # max 2 d |H_zz| over the hull-adjacent columns (diagnostic)
        ij, P = G["ij"], G["P"]
        index = G["index"]
        byc = [[] for _ in range(g["nx"])]
        for k, (i, j) in enumerate(ij):
            byc[i].append((P[k, 1], k, j))
        hz_mid = g.get("hz_mid", np.zeros(g["nx"]))
        hzz_mid = g.get("hzz_mid", np.zeros(g["nx"]))
        hxz_mid = g.get("hxz_mid", np.zeros(g["nx"]))
        for i in range(g["nx"]):
            e = sorted(byc[i])
            hull_adj = g["i_bow"] <= i < g["i_stern"]           # lowest line touches the hull
            ys = [y for y, _, _ in e]
            own = [([(k, 1.0)], [], 0.0) for _, k, _ in e]
            js = [j for _, _, j in e]
            k0 = e[0][1]
            if not hull_adj:
                ys = [-ys[0]] + ys
                gs = gp = ([(k0, 1.0)], [], 0.0)
            else:
                yw, hx = g["ywl_mid"][i], g["ywlp_mid"][i]
                hz, hzz, hxz = hz_mid[i], hzz_mid[i], hxz_mid[i]
                d = ys[0] - yw
                ys = [yw - d] + ys
                a = 2 * d * hx / self.dx
                km = index[(i - 1, 0)] if (i - 1, 0) in index else k0
                if mode == "wall":
                    b = 2 * d * hxz / self.dx
                    gs = ([(k0, 1.0 + a - 2 * d * hzz), (km, -a)], [(km, b), (k0, -b)], -2 * d * hxz)
                    gp = ([(k0, 1.0 + a), (km, -a)], [(k0, 2 * d * hz)], 2 * d * hx)
                    self.ghost_curv = max(self.ghost_curv, abs(2 * d * hzz))
                elif mode == "wall_vertical":
                    gs = ([(k0, 1.0 + a), (km, -a)], [], 0.0)
                    gp = ([(k0, 1.0 + a), (km, -a)], [], 2 * d * hx)
                elif mode == "wall_const":
                    gs = ([(k0, 1.0)], [], 0.0)
                    gp = ([(k0, 1.0)], [], 2 * d * hx)
                else:                                                # "even"
                    gs = gp = ([(k0, 1.0)], [], 0.0)
            js = [None] + js
            self.cols.append((np.array(ys), dict(sig=[gs] + own, phi=[gp] + own), js))
        self._cache = {}

    def weights(self, i, y, field="sig"):
        """Quadratic interpolation at y on column i of the field "sig" or
        "phi", centred on the nearest point (symmetric stencil), the virtual
        point included: value = sum ws f_own[ks] + sum wo f_other[ko] + cU U
        (other = phi for sig, sig for phi; only the hull-wall ghost couples
        the fields). Returns (ks, ws, ko, wo, cU). Exact identity when y is a
        line's own y. Cached per (column, y, field)."""
        key = (i, float(y), field)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        ys, pts_d, _ = self.cols[i]
        pts = pts_d[field]
        dd = np.abs(ys - y)
        m = int(np.argmin(dd))
        if dd[m] <= 1e-9 * self.dy and pts[m][2] == 0.0 and len(pts[m][0]) == 1 and not pts[m][1]:
            out = ([pts[m][0][0][0]], [1.0], [], [], 0.0)
            self._cache[key] = out
            return out
        n = len(ys)
        c0 = min(max(m, 1), n - 2)
        sel = [c0 - 1, c0, c0 + 1]
        yy = ys[sel]
        ks, ws, ko, wo, cU = [], [], [], [], 0.0
        for a in range(3):
            num, den = 1.0, 1.0
            for b in range(3):
                if b != a:
                    num *= (y - yy[b])
                    den *= (yy[a] - yy[b])
            w = num / den
            lst, oth, const = pts[sel[a]]
            for (kk, cc) in lst:
                ks.append(int(kk))
                ws.append(w * cc)
            for (kk, cc) in oth:
                ko.append(int(kk))
                wo.append(w * cc)
            cU += w * const
        out = (ks, ws, ko, wo, cU)
        self._cache[key] = out
        return out

    def neighbours(self, i, k):
        """Lateral neighbours (y_lo, k_lo), (y_hi, k_hi) of point k on column
        i (virtual point below the lowest line counts with its leading
        index, i.e. the even reflection); None where missing."""
        ys, pts_d, js = self.cols[i]
        pts = pts_d["sig"]
        pos = [p for p in range(len(pts)) if js[p] is not None and pts[p][0][0][0] == k][0]
        lo = (ys[pos - 1], int(pts[pos - 1][0][0][0])) if pos - 1 >= 0 else None
        hi = (ys[pos + 1], int(pts[pos + 1][0][0][0])) if pos + 1 < len(pts) else None
        return ys[pos], lo, hi


def build_rows(G, k, zgrid=None, guard=True, transom_data=None, ghost_mode=None):
    """Sparse A (mass), Bk (second difference with ramp/beach), C (box
    correction) and the extra right-hand side of the transom rows, for the
    free-surface unknowns of grid G (module docstring 2, 3, 5).
    transom_data: dict(zT (per transom-line point at the edge), sb) keyed by
    line j < 0, or None. ghost_mode: hull-wall virtual point (_Columns;
    None = GHOST_MODE). Rows: A sig - k Bk phi = k U rhs_cphi (+ transom
    terms); the hull-wall ghost couples the fields (sig ghost has phi terms,
    phi ghost has sig terms), which enter A and Bk accordingly."""
    g = G["grid"]
    nx, dx = g["nx"], g["dx"]
    ij, P = G["ij"], G["P"]
    NF = len(ij)
    index = G["index"]
    exact = PHI_MODE == "exact"
    cols_s = _Columns(G, "even" if exact else ghost_mode)    # sigma and (in "interp" mode) phi, separate ghost lists
    w_col, nu_col = g["w_col"], g["nu_col"]
    lam_c = blend_weights(w_col, zgrid if guard else None, True)
    xe = g["xe"]
    xmid = 0.5 * (xe[:-1] + xe[1:])
    ywm = g["ywl_mid"]
    i_b, i_s = g["i_bow"], g["i_stern"]

    def hull_adj(i):
        return i_b <= i < i_s

    extra, extra_idx = [], {}

    def phi_point(cc, yt, kown):
        """phi at (x_cc, y_t) as (phi list, sig list, constant x U): the
        column's own free-surface point if one lies at y_t (identity), else
        an extra evaluation point of the representation (exact) or the
        interpolation ("interp", with the phi ghost of _Columns)."""
        if kown is not None and abs(P[kown, 1] - yt) <= 1e-9 * g["dy"]:
            return [(kown, 1.0)], [], 0.0
        if not exact:
            ks_, ws_, ko_, wo_, cu_ = cols_s.weights(cc, yt, "phi")
            return list(zip(ks_, ws_)), list(zip(ko_, wo_)), cu_
        ks_, ws_, _, _, _ = cols_s.weights(cc, yt, "sig")
        if len(ks_) == 1 and ws_[0] == 1.0:
            return [(ks_[0], 1.0)], [], 0.0
        key = (int(cc), round(float(yt), 12))
        if key not in extra_idx:
            extra_idx[key] = len(extra)
            extra.append((xmid[cc], yt, 0.0))
        return [(NF + extra_idx[key], 1.0)], [], 0.0
    am = np.array([1.0, 10.0, 1.0]) / 12
    Ar, Ac, Av = [], [], []
    Br, Bc, Bv = [], [], []
    rhs_c = np.zeros(NF)

    def add_mass(kr, coef, cc, yt):
        """coef * sig(x_cc, y_t) on the left of row kr (sig ghost: own sig
        terms -> A; its phi terms -> Bk with -1/k; constant -> rhs)."""
        ks_, ws_, ko_, wo_, cu_ = cols_s.weights(cc, yt, "sig")
        for kk2, ww in zip(ks_, ws_):
            Ar.append(kr)
            Ac.append(kk2)
            Av.append(coef * ww)
        for kk2, ww in zip(ko_, wo_):
            Br.append(kr)
            Bc.append(kk2)
            Bv.append(-coef * ww / k)
        rhs_c[kr] -= coef * cu_ / k

    def add_phi(kr, coef, pt):
        """-k coef phi(point) on the left of row kr (phi list -> Bk; the
        ghost's sig terms -> A with -k; constant -> rhs)."""
        pl, sl, cu_ = pt
        for kk2, ww in pl:
            Br.append(kr)
            Bc.append(kk2)
            Bv.append(coef * ww)
        for kk2, ww in sl:
            Ar.append(kr)
            Ac.append(kk2)
            Av.append(-k * coef * ww)
        rhs_c[kr] += coef * cu_
    lines = {}
    for kk, (i, j) in enumerate(ij):
        lines.setdefault(int(j), []).append((int(i), kk))
    for j, pts in lines.items():
        pts.sort()
        first = pts[0][0]
        for (r, kr) in pts:
            if j < 0 and r == first:
                # D2: (3 sig_f - sig_{f+1})/2 = U s_b  (rhs filled by the solver: depends on U)
                kf1 = index[(r + 1, j)]
                Ar += [kr, kr]
                Ac += [kr, kf1]
                Av += [1.5, -0.5]
                continue
            if j < 0 and r == first + 1:
                kf = index[(r - 1, j)]
                Ar += [kr, kr]
                Ac += [kf, kr]
                Av += [0.5, 0.5]
                Br += [kr, kr]
                Bc += [kf, kr]
                Bv += [-1.0 / dx ** 2, 1.0 / dx ** 2]
                continue
            c = r - 1
            if c < 0:
                Ar.append(kr)
                Ac.append(kr)
                Av.append(1.0)
                continue
            kc = index[(c, j)]
            yc = P[kc, 1]
            lc = lam_c[c]
            a0 = (1 - lc) * am[0] + lc
            a1 = (1 - lc) * am[1]
            a2 = (1 - lc) * am[2]
            wc, nc = w_col[c], nu_col[c]
            yt = yc
            if exact and j == 0 and hull_adj(c):
                # hull-adjacent line: target y raised so that all three
                # target points lie outside the waterplane (module docstring 2)
                yt = max(ywm[cc] for cc in (c - 1, c, c + 1) if hull_adj(cc)) + (yc - ywm[c])
            # mass part: sigma at (x_{c+-1}, y_t), (x_c, y_t) (sig ghost at the hull wall)
            add_mass(kr, a0, r, yt)
            if yt == yc:
                Ar.append(kr)
                Ac.append(kc)
                Av.append(a1)
            else:
                add_mass(kr, a1, c, yt)
            if c == 0:
                # phi_{-1} = 2 phi_0 - phi_1 (B3 entry row): second difference 0
                if nc != 0.0:
                    add_phi(kr, -wc * nc / dx, phi_point(c, yt, kc))
                    add_phi(kr, wc * nc / dx, phi_point(1, yt, None))
                continue
            add_mass(kr, a2, c - 1, yt)
            vm = wc * (1 / dx ** 2 - nc / (2 * dx))
            vp = wc * (1 / dx ** 2 + nc / (2 * dx))
            for cc, v in ((c - 1, vm), (c, -2 * wc / dx ** 2), (c + 1, vp)):
                add_phi(kr, v, phi_point(cc, yt, kc if cc == c else None))
    A = sp.csr_matrix((Av, (Ar, Ac)), shape=(NF, NF))
    NE = len(extra)
    Bk = sp.csr_matrix((Bv, (Br, Bc)), shape=(NF, NF + NE))
    # box correction
    Cr, Cc, Cv = list(range(NF)), list(range(NF)), [1.0] * NF
    for j, pts in lines.items():
        for a in range(1, len(pts) - 1):
            kk = pts[a][1]
            for kk2, v in ((pts[a - 1][1], -1.0), (kk, 2.0), (pts[a + 1][1], -1.0)):
                Cr.append(kk)
                Cc.append(kk2)
                Cv.append(v / 24)
    ny = g["ny"]
    for kk, (i, j) in enumerate(ij):
        if j == ny - 1:
            continue
        yk, lo, hi = cols_s.neighbours(i, kk)
        if lo is None or hi is None:
            continue
        wdt = G["width"][kk]
        h1, h2 = yk - lo[0], hi[0] - yk
        # -(w^2/24) sig_yy, sig_yy = 2 [ (s_hi - s)/h2 - (s - s_lo)/h1 ] / (h1 + h2)
        f = -(wdt ** 2 / 24) * 2 / (h1 + h2)
        for kk2, v in ((hi[1], f / h2), (kk, -f / h2 - f / h1), (lo[1], f / h1)):
            Cr.append(kk)
            Cc.append(kk2)
            Cv.append(v)
    C = sp.csr_matrix((Cv, (Cr, Cc)), shape=(NF, NF))
    interior = w_col >= 1.0 - 1e-12
    guard_l = float(lam_c[interior].max()) if interior.any() else 0.0
    return dict(A=A, Bk=Bk, C=C, lam=lam_c, guard_l=guard_l, lines=lines, rhs_cphi=rhs_c,
                E=np.array(extra, float).reshape(-1, 3), cols=cols_s, ghost_curv=cols_s.ghost_curv)


# =================================================================== solver
class PiercingNK:
    """Steady linear (Neumann-Kelvin) flow about a surface-piercing hull at a
    given sinkage and trim (module docstring). branch: None (no transom),
    "dry" or "wet" (transom hulls); eta: dry fraction for "wet" (1 = dry).
    method: "fft" (GMRES, production) or "dense" (LU, verification)."""

    def __init__(self, hull, U, g=9.81, sink=0.0, trim=0.0, n_per_lam=30, m_body=60,
                 hull_mesh=(60, 12), face_mesh=(6, 8), mult=(1.0, 1.0, 1.0, 1.0, 1.0), aspect=1.0,
                 J_mult=2.0, branch=None, eta=1.0, l_h=5.0, rho=1.0, device=None, guard=True,
                 allow_guard=False, lat=None, down=None, up=None, ghost_mode=None):
        self.hull, self.U, self.g, self.rho = hull, float(U), float(g), float(rho)
        self.sink, self.trim = float(sink), float(trim)
        self.k = U * U / g
        self.lam = 2 * np.pi * self.k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.allow_guard = allow_guard
        self.transom = bool(getattr(hull, "transom", False))
        self.branch = branch if self.transom else None
        if self.transom and branch not in ("dry", "wet"):
            raise ValueError("transom hull needs branch 'dry' or 'wet'")
        self.eta = 1.0 if self.branch == "dry" else float(eta)
        self.l_h = float(l_h)
        # meshes
        nbx, nbz = hull_mesh
        self.hull_mesh_shape = (nbx, nbz)
        self.body = hull.mesh(nbx, nbz, sink, trim)
        assert len(self.body) == nbx * nbz, "hull mesh lost panels (structured gradient needs all)"
        self.face_shape = tuple(face_mesh)
        if self.transom:
            # face_mesh = (n across, n vertical)
            self.body = self.body + hull.face_mesh(face_mesh[0], face_mesh[1], sink, trim)
        self.is_face = self.body.tag == 2
        self.NB = len(self.body)
        self.G = build_grid(hull, sink, trim, self.lam, n_per_lam, m_body, mult, aspect, J_mult,
                            up=up, down=down, lat=lat, transom=self.transom)
        gr = self.G["grid"]
        self.fs, self.P = self.G["fs"], self.G["P"]
        self.NF = len(self.fs)
        self.NL = gr["nx"] * gr["ny"]
        kind = self.G["kind"]
        self.lat_mask = np.zeros(self.NF, bool)
        self.lat_mask[:self.NL] = kind[:self.NL] == 0
        self.dense_idx = np.nonzero(~self.lat_mask)[0]
        self.lat_idx = np.nonzero(self.lat_mask)[0]
        self.zgrid = lattice_zmax(gr["dx"], gr["dy"], self.k, True)
        self.ghost_mode = GHOST_MODE if ghost_mode is None else ghost_mode
        R = build_rows(self.G, self.k, self.zgrid, guard, ghost_mode=self.ghost_mode)
        self.A, self.Bk, self.C = R["A"], R["Bk"], R["C"]
        self._cols = R["cols"]
        self.ghost_curv = R["ghost_curv"]
        self.guard_l = R["guard_l"]
        self.lines = R["lines"]
        self.rhs_cphi = R["rhs_cphi"]
        self.E = R["E"]
        self.NE = len(self.E)
        dev = self.device
        self._A, self._Bk, self._C = (_torch_sparse(M, dev) for M in (self.A, self.Bk, self.C))
        # transom data
        self.phin = self._body_condition()
        self.rhs_extra, self.f_patch = self._transom_terms()
        self._blocks()

    # ----------------------------------------------------- data and blocks
    def _body_condition(self):
        """dphi/dn on the body: U n_x on the hull; on the transom face the
        edge value of the transom condition, phi_x = g z_T(y)/U (both
        branches), i.e. dphi/dn = n_x g z_T(y)/U."""
        pn = self.U * self.body.n[:, 0].copy()
        if self.transom:
            f = self.is_face
            zT, _ = self.hull.transom_edge(np.abs(self.body.c[f, 1]), self.sink, self.trim)
            pn[f] = self.body.n[f, 0] * self.g * zT / self.U
        return pn

    def hollow(self, s, y):
        """WaveBEM-form hollow behind the transom (module docstring 5):
        z_h(s) = a s^2 + s_b' s + z_T on 0 <= s <= l_h |z_T|, 0 beyond, with
        s_b' = dz/ds = -s_b (s downstream) and z_h(l_h |z_T|) = 0. Returns
        z_h and dz_h/ds."""
        zT, sb = self.hull.transom_edge(np.abs(y), self.sink, self.trim)
        sp_ = -sb
        ell = self.l_h * np.abs(zT)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(ell > 0, (-zT - sp_ * ell) / np.maximum(ell, 1e-300) ** 2, 0.0)
        inside = (s >= 0) & (s <= ell) & (np.abs(y) < self.G["grid"]["yT"]) & (ell > 0)
        zh = np.where(inside, a * s * s + sp_ * s + zT, 0.0)
        dzh = np.where(inside, 2 * a * s + sp_, 0.0)
        return zh, dzh

    def _transom_terms(self):
        """Extra right-hand side of the transom rows (D1, D2) and the patch
        forcing f = (1 - eta) U d|z_h|/dx (wet branch) at the free-surface
        points."""
        NF = self.NF
        rhs = np.zeros(NF)
        f = np.zeros(NF)
        if not self.transom:
            return rhs, f
        gr = self.G["grid"]
        dx, xs = gr["dx"], gr["x_s"]
        ij, P = self.G["ij"], self.P
        for j, pts in self.lines.items():
            if j >= 0:
                continue
            (i0, k0), (i1, k1) = pts[0], pts[1]
            y = P[k0, 1]
            zT, sb = self.hull.transom_edge(np.array([y]), self.sink, self.trim)
            rhs[k0] = self.U * float(sb[0])                         # D2
            rhs[k1] = self.k * self.g * float(zT[0]) / (self.U * dx)  # D1
        if self.branch == "wet" and self.eta < 1.0:
            s = xs - P[:, 0]
            behind = s > 0
            zh, dzh = self.hollow(np.where(behind, s, -1.0), P[:, 1])
            # |z_h| = -z_h; d|z_h|/dx = -d|z_h|/ds = dz_h/ds
            f = np.where(behind, (1.0 - self.eta) * self.U * dzh, 0.0)
        return rhs, f

    def _blocks(self, chunk=16384):
        dev, body, fs, P = self.device, self.body, self.fs, self.P
        NB, NF = self.NB, self.NF
        ob = ImageOperator(body, axes=(1, 2), device=dev, parities=[(1, 1)], on_device=True)
        self.S_BB, self.D_BB = ob.S((1, 1)), ob.D((1, 1))
        del ob
        pn = torch.as_tensor(self.phin, device=dev)
        self.D_FB = torch.empty((NF, NB), dtype=torch.float64, device=dev)
        self.SFBpn = torch.empty(NF, dtype=torch.float64, device=dev)
        for r0 in range(0, NF, chunk):
            r1 = min(NF, r0 + chunk)
            op = ImageOperator(body, axes=(1, 2), xf=P[r0:r1], device=dev, parities=[(1, 1)], on_device=True)
            self.D_FB[r0:r1] = op.D((1, 1))
            self.SFBpn[r0:r1] = op.S((1, 1)) @ pn
            del op
        self.S_BF = torch.empty((NB, NF), dtype=torch.float64, device=dev)
        for c0 in range(0, NF, chunk):
            c1 = min(NF, c0 + chunk)
            sub = fs.subset(np.arange(c0, c1))
            S1, _ = influence(body.c, sub, device=dev, as_numpy=False)
            S2, _ = influence(body.c, sub.reflect(1), device=dev, as_numpy=False)
            self.S_BF[:, c0:c1] = 2.0 * (S1 + S2)
            del S1, S2
        # free surface: FFT lattice + dense blocks for band / transom panels
        gr = self.G["grid"]
        self.lat = FSLattice(dict(nx=gr["nx"], ny=gr["ny"], dx=gr["dx"], dy=gr["dy"]), dev)
        D = self.dense_idx
        self.ND = len(D)
        fsD = fs.subset(D)
        self.S_xD = torch.empty((NF, self.ND), dtype=torch.float64, device=dev)   # all points x dense panels
        for r0 in range(0, NF, chunk):
            r1 = min(NF, r0 + chunk)
            S1, _ = influence(P[r0:r1], fsD, device=dev, as_numpy=False)
            S2, _ = influence(P[r0:r1], fsD.reflect(1), device=dev, as_numpy=False)
            self.S_xD[r0:r1] = 2.0 * (S1 + S2)
            del S1, S2
        Lp = self.lat_idx
        fsL = fs.subset(Lp)
        self.S_DL = torch.empty((self.ND, len(Lp)), dtype=torch.float64, device=dev)  # dense points x lattice panels
        for c0 in range(0, len(Lp), chunk):
            c1 = min(len(Lp), c0 + chunk)
            sub = fsL.subset(np.arange(c0, c1))
            S1, _ = influence(P[D], sub, device=dev, as_numpy=False)
            S2, _ = influence(P[D], sub.reflect(1), device=dev, as_numpy=False)
            self.S_DL[:, c0:c1] = 2.0 * (S1 + S2)
            del S1, S2
        # extra evaluation points of the band rows (phi from the representation)
        NE = self.NE
        self.D_EB = torch.empty((NE, NB), dtype=torch.float64, device=dev)
        self.SEBpn = torch.empty(NE, dtype=torch.float64, device=dev)
        self.S_EF = torch.empty((NE, NF), dtype=torch.float64, device=dev)
        if NE:
            op = ImageOperator(body, axes=(1, 2), xf=self.E, device=dev, parities=[(1, 1)], on_device=True)
            self.D_EB[:] = op.D((1, 1))
            self.SEBpn[:] = op.S((1, 1)) @ pn
            del op
            for c0 in range(0, NF, chunk):
                c1 = min(NF, c0 + chunk)
                sub = fs.subset(np.arange(c0, c1))
                S1, _ = influence(self.E, sub, device=dev, as_numpy=False)
                S2, _ = influence(self.E, sub.reflect(1), device=dev, as_numpy=False)
                self.S_EF[:, c0:c1] = 2.0 * (S1 + S2)
                del S1, S2
        self._D = torch.as_tensor(D, device=dev)
        self._L = torch.as_tensor(Lp, device=dev)
        lm = torch.zeros(self.NL, dtype=torch.float64, device=dev)
        lm[torch.as_tensor(Lp, device=dev)] = 1.0
        self._latmask = lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ products
    def _sp(self, M, v):
        return torch.sparse.mm(M, v[:, None])[:, 0]

    def SFF(self, sp_):
        """S_FF sig' at every free-surface point (FFT lattice + dense)."""
        NL = self.NL
        out = torch.empty(self.NF, dtype=torch.float64, device=self.device)
        conv = self.lat.apply(sp_[:NL] * self._latmask)
        out[:NL] = conv
        if self.NF > NL:
            out[NL:] = 0.0
        out[self._D] = 0.0
        out[self._D] += self.S_DL @ sp_[self._L]
        out += self.S_xD @ sp_[self._D]
        return out

    def matvec(self, x):
        NB = self.NB
        phB, sig = x[:NB], x[NB:]
        sp_ = self._sp(self._C, sig)
        phiF4 = -self.SFF(sp_) + self.D_FB @ phB
        if self.NE:
            phiF4 = torch.cat([phiF4, -(self.S_EF @ sp_) + self.D_EB @ phB])
        out = torch.empty_like(x)
        out[:NB] = 2 * np.pi * phB - self.D_BB @ phB + self.S_BF @ sp_
        out[NB:] = self._sp(self._A, sig) - (self.k / (4 * np.pi)) * self._sp(self._Bk, phiF4)
        return out

    def rhs(self):
        dev = self.device
        NB = self.NB
        b = torch.zeros(NB + self.NF, dtype=torch.float64, device=dev)
        pn = torch.as_tensor(self.phin, device=dev)
        b[:NB] = -(self.S_BB @ pn)
        known = -self.SFBpn if not self.NE else torch.cat([-self.SFBpn, -self.SEBpn])
        b[NB:] = (self.k / (4 * np.pi)) * self._sp(self._Bk, known)
        b[NB:] += torch.as_tensor(self.rhs_extra + self.A @ self.f_patch + self.k * self.U * self.rhs_cphi,
                                  device=dev)
        return b

    def dense_matrix(self):
        """Full system matrix by explicit assembly (verification, small N)."""
        NB, NF, dev = self.NB, self.NF, self.device
        M = torch.zeros((NB + NF, NB + NF), dtype=torch.float64, device=dev)
        SFF = torch.zeros((NF, NF), dtype=torch.float64, device=dev)
        SFF[:self.NL, :self.NL] = self.lat.dense() * self._latmask[None, :]
        SFF[self._D] = 0.0
        SFF[self._D[:, None], self._L[None, :]] = self.S_DL
        SFF[:, self._D] = self.S_xD
        if self.NE:
            SFF = torch.cat([SFF, self.S_EF], 0)
        SC = torch.sparse.mm(self._C.t().coalesce(), SFF.T.contiguous()).T
        del SFF
        M[NB:, NB:] = (self.k / (4 * np.pi)) * torch.sparse.mm(self._Bk, SC) + self._A.to_dense()
        M[:NB, :NB] = -self.D_BB
        M[:NB, :NB].diagonal().add_(2 * np.pi)
        M[:NB, NB:] = torch.sparse.mm(self._C.t().coalesce(), self.S_BF.T.contiguous()).T
        DXB = self.D_FB if not self.NE else torch.cat([self.D_FB, self.D_EB], 0)
        M[NB:, :NB] = -(self.k / (4 * np.pi)) * torch.sparse.mm(self._Bk, DXB)
        return M

    def preconditioner(self):
        """Block preconditioner: body block by dense LU; free surface by the
        B3 y-circulant operator of the LATTICE rows on the nx x ny logical
        grid (exact in x, circulant in y; the band rows differ only by the
        lateral interpolation); transom-line unknowns by the diagonal of A."""
        dev = self.device
        NB, NL = self.NB, self.NL
        MB = -self.D_BB.clone()
        MB.diagonal().add_(2 * np.pi)
        luB = torch.linalg.lu_factor(MB)
        del MB
        gr = self.G["grid"]
        nx, ny, dx = gr["nx"], gr["ny"], gr["dx"]
        lam_c = blend_weights(gr["w_col"], self.zgrid, True)
        am = np.array([1.0, 10.0, 1.0]) / 12
        lr = np.concatenate([[1.0], lam_c[:nx - 1]])
        Ax = (np.diag((1 - lr) * am[0] + lr) + np.diag(((1 - lr) * am[1])[1:], -1)
              + np.diag(((1 - lr) * am[2])[2:], -2))
        Bx = np.zeros((nx, nx))
        wc, nc = gr["w_col"], gr["nu_col"]
        for c in range(0, nx - 1):
            r = c + 1
            if c == 0:
                if nc[0] != 0:
                    Bx[r, 0] += -wc[0] * nc[0] / dx
                    Bx[r, 1] += wc[0] * nc[0] / dx
                continue
            Bx[r, c - 1] += wc[c] * (1 / dx ** 2 - nc[c] / (2 * dx))
            Bx[r, c] += -2 * wc[c] / dx ** 2
            Bx[r, c + 1] += wc[c] * (1 / dx ** 2 + nc[c] / (2 * dx))
        lat = self.lat
        n2 = 2 * ny
        nn = torch.arange(n2, device=dev)
        Kt = lat.K[:, torch.minimum(nn, n2 - nn)]
        Kq = torch.fft.fft(Kt, dim=1).real[:, :ny + 1]
        ii = torch.arange(nx, device=dev)
        T = Kq[(ii[:, None] - ii[None, :]).abs()].permute(2, 0, 1).contiguous()
        del Kq, Kt
        Cx = np.eye(nx)
        for i in range(1, nx - 1):
            Cx[i, i - 1:i + 2] += np.array([-1.0, 2.0, -1.0]) / 24
        cq = (2 - 2 * torch.cos(np.pi * torch.arange(ny + 1, device=dev, dtype=torch.float64) / ny)) / 24
        Cxq = (torch.as_tensor(Cx, device=dev)[None]
               + cq[:, None, None] * torch.eye(nx, device=dev, dtype=torch.float64)[None])
        O = T @ Cxq
        del T, Cxq
        O = (self.k / (4 * np.pi)) * (torch.as_tensor(Bx, device=dev)[None] @ O) + torch.as_tensor(Ax, device=dev)[None]
        LU, piv = torch.linalg.lu_factor(O)
        del O
        d = torch.arange(ny + 1, device=dev)
        dm = (n2 - d) % n2
        dA = torch.as_tensor(self.A.diagonal(), device=dev)
        NF = self.NF
        # dense set (band and transom-line unknowns): exact block of the
        # free-surface rows restricted to the dense unknowns, with S_FF C
        # approximated by S_FF[:, D] C[D, D] (the lattice neighbours' share
        # of the box correction left out), LU on the device (added in B4
        # after exploration B4-E7: the y-circulant block alone needed > 6000
        # iterations on the transom grid, where the band runs to the
        # domain end)
        luD, Didx = None, self._D
        if self.ND:
            Dn = self.dense_idx
            BkD = _torch_sparse(self.Bk[Dn], dev)                              # (ND, NF + NE), sparse
            CDD = _torch_sparse(self.C[Dn][:, Dn], dev)
            SxD = self.S_xD if not self.NE else torch.cat([self.S_xD, self.S_EF[:, self._D]], 0)
            BS = torch.sparse.mm(BkD, SxD)                                       # (ND, ND)
            MDD = torch.as_tensor(self.A[Dn][:, Dn].toarray(), device=dev) + (self.k / (4 * np.pi)) * torch.sparse.mm(
                CDD.t().coalesce(), BS.T.contiguous()).T
            del SxD, BS
            del BkD, CDD
            luD = torch.linalg.lu_factor(MDD)
            del MDD

        def fs_apply(v):
            a = v[:NL].reshape(ny, nx)
            ext = torch.cat([torch.flip(a, dims=[0]), a], dim=0)
            R = torch.fft.fft(ext, dim=0)
            rhs = torch.stack([R[d].real, R[d].imag, R[dm].real, R[dm].imag], dim=2)
            X = torch.linalg.lu_solve(LU, piv, rhs)
            Xc = torch.empty_like(R)
            Xc[dm] = torch.complex(X[..., 2], X[..., 3])
            Xc[d] = torch.complex(X[..., 0], X[..., 1])
            out = torch.empty_like(v)
            out[:NL] = torch.fft.ifft(Xc, dim=0).real[ny:].reshape(-1)
            if NF > NL:
                out[NL:] = v[NL:] / dA[NL:]
            if luD is not None:
                out[Didx] = torch.linalg.lu_solve(*luD, v[Didx][:, None])[:, 0]
            return out

        def apply(v):
            out = torch.empty_like(v)
            out[:NB] = torch.linalg.lu_solve(*luB, v[:NB, None])[:, 0]
            out[NB:] = fs_apply(v[NB:])
            return out
        return apply

    # -------------------------------------------------------------- solve
    def solve(self, method="fft", tol=1e-12, restart=300, maxiter=6000, floor=None, strict=True,
              verbose=False, fs_forcing=None):
        floor = 100.0 * tol if floor is None else float(floor)
        b = self.rhs()
        if fs_forcing is not None:
            b[self.NB:] += torch.as_tensor(np.asarray(fs_forcing, float), device=self.device)
        if method == "dense":
            M = self.dense_matrix()
            x = torch.linalg.solve(M, b)
            rel = float(torch.linalg.norm(b - M @ x) / torch.linalg.norm(b))
            del M
            info = dict(method="dense", relres=rel, iters=0, converged=rel <= tol, accepted=rel <= floor,
                        floor=floor)
        else:
            Pc = self.preconditioner()
            x, info = gmres(self.matvec, b, tol=tol, restart=restart, maxiter=maxiter, precond=Pc,
                            verbose=verbose, floor=floor)
            info.pop("history", None)
            info["method"] = "fft"
        info["guard_l"] = self.guard_l
        info["zgrid"] = self.zgrid
        info["ghost_curv"] = self.ghost_curv
        if self.guard_l > 0 and not self.allow_guard:
            warnings.warn(f"resolution guard active: l = {self.guard_l:.3f}", RuntimeWarning)
        if strict and not info["accepted"]:
            raise RuntimeError(f"piercing solve not accepted: relres {info['relres']:.2e} > floor {floor:.0e}")
        NB = self.NB
        sig = x[NB:]
        sp_ = self._sp(self._C, sig)
        phiF = (-self.SFF(sp_) + self.D_FB @ x[:NB] - self.SFBpn) / (4 * np.pi)
        self.sol = dict(phi_B=x[:NB].cpu().numpy(), sigma=sig.cpu().numpy(), sigma_eff=sp_.cpu().numpy(),
                        phi_F=phiF.cpu().numpy(), info=info)
        return self.sol

    def release(self):
        for a in ("S_BB", "D_BB", "D_FB", "S_BF", "S_xD", "S_DL", "lat", "_A", "_Bk", "_C", "SFBpn", "D_EB", "S_EF", "SEBpn"):
            if hasattr(self, a):
                setattr(self, a, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------ post-processing
    def phi_x_fs(self, phi=None, order=4, cU=None):
        """phi_x at fixed y at every free-surface point: central differences
        along the line (order 4, 2 near the line ends) with the lateral
        interpolation of the rows (module docstring 2). cU: constant of the
        hull-wall ghost (U for phi, the default when phi is None; 0 for a
        field with a homogeneous wall condition)."""
        own = phi is None
        if cU is None:
            cU = self.U if own else 0.0
        phi = self.sol["phi_F"] if own else phi
        Mp, Ms, uvec = self._dx_operator(order)
        out = Mp @ np.asarray(phi, float) + cU * uvec
        if own and Ms.nnz:
            out = out + Ms @ np.asarray(self.sol["sigma"], float)
        return out

    def _dx_operator(self, order=4):
        """Sparse form of phi_x_fs (built once per solver): phi_x = Mp phi +
        Ms sig + U u, with the lateral interpolation and the phi ghost of the
        rows (_Columns, field "phi"; the sig coupling is the H_z term of the
        ghost). Same stencils as the B4 loop version (results equal to
        round-off)."""
        cache = getattr(self, "_dxop", None)
        if cache is not None and order in cache:
            return cache[order]
        cols = self._cols
        dx = self.G["grid"]["dx"]
        NF = self.NF
        rp, cp_, vp = [], [], []
        rs, cs, vs = [], [], []
        uvec = np.zeros(NF)
        for j, pts in self.lines.items():
            ii = [p[0] for p in pts]
            i_lo, i_hi = ii[0], ii[-1]
            for (c, kc) in pts:
                yc = self.P[kc, 1]

                def add(i, coef):
                    ks, ws, ko, wo, cu = cols.weights(i, yc, "phi")
                    rp.extend([kc] * len(ks))
                    cp_.extend(ks)
                    vp.extend([coef * w for w in ws])
                    rs.extend([kc] * len(ko))
                    cs.extend(ko)
                    vs.extend([coef * w for w in wo])
                    uvec[kc] += coef * cu

                def addself(coef):
                    rp.append(kc)
                    cp_.append(kc)
                    vp.append(coef)
                if order == 4 and c - 2 >= i_lo and c + 2 <= i_hi:
                    for i, cf in ((c + 2, 1.0), (c + 1, -8.0), (c - 1, 8.0), (c - 2, -1.0)):
                        add(i, cf / (12 * dx))
                elif c - 1 >= i_lo and c + 1 <= i_hi:
                    add(c - 1, 1.0 / (2 * dx))
                    add(c + 1, -1.0 / (2 * dx))
                elif c + 1 <= i_hi:
                    addself(1.0 / dx)
                    add(c + 1, -1.0 / dx)
                else:
                    add(c - 1, 1.0 / dx)
                    addself(-1.0 / dx)
        Mp = sp.csr_matrix((vp, (rp, cp_)), shape=(NF, NF))
        Ms = sp.csr_matrix((vs, (rs, cs)), shape=(NF, NF))
        cache = cache or {}
        cache[order] = (Mp, Ms, uvec)
        self._dxop = cache
        return cache[order]

    def elevation(self):
        """zeta = (U/g) phi_x (+ (1 - eta)|z_h| on the wet patch)."""
        z = self.U / self.g * self.phi_x_fs(cU=self.U)
        if self.branch == "wet" and self.eta < 1.0:
            s = self.G["grid"]["x_s"] - self.P[:, 0]
            zh, _ = self.hollow(np.where(s > 0, s, -1.0), self.P[:, 1])
            z = z + (1 - self.eta) * np.abs(zh)
        self.zeta = z
        return z

    def waterline_profile(self, source="hull"):
        """Linear wave elevation along the hull waterline.
        source "hull" (default): zeta = (U/g) phi_x on the hull surface at
        z = 0, phi_x from the structured surface gradient of the top two
        panel rows extrapolated linearly to z = 0 (the quantity a wave probe
        on the hull measures). source "fs": quadratic extrapolation of the
        free-surface zeta from lines 0, 1, 2 to the waterline (noisier: it
        amplifies the lateral short-wave content next to the hull,
        exploration B4-E5). Returns x, y_wl, zeta (x increasing)."""
        if source == "fs":
            z = self.zeta if hasattr(self, "zeta") else self.elevation()
            gr = self.G["grid"]
            idx = self.G["index"]
            xs, ys, zs = [], [], []
            for i in range(gr["i_bow"], gr["i_stern"]):
                k0, k1, k2 = idx[(i, 0)], idx[(i, 1)], idx[(i, 2)]
                xc = self.P[k0, 0]
                yw = float(self.hull.waterline(np.array([xc]), self.sink, self.trim)[0])
                c = np.polyfit(self.P[[k0, k1, k2], 1] - yw, z[[k0, k1, k2]], 2)
                xs.append(xc)
                ys.append(yw)
                zs.append(np.polyval(c, 0.0))
            o = np.argsort(xs)
            return np.array(xs)[o], np.array(ys)[o], np.array(zs)[o]
        _, _, grad = self.hull_pressure(self.sol["phi_B"])
        nx_, nz_ = self.hull_mesh_shape
        G = grad.reshape(nx_, nz_, 3)
        C = self.body.subset(self.body.tag == 0).c.reshape(nx_, nz_, 3)
        z1, z2 = C[:, -1, 2], C[:, -2, 2]
        px = G[:, -1, 0] + (G[:, -1, 0] - G[:, -2, 0]) * (0.0 - z1) / (z1 - z2)
        x = C[:, -1, 0]
        o = np.argsort(x)
        x = x[o]
        return x, self.hull.waterline(x, self.sink, self.trim), (self.U / self.g * px)[o]

    def waterline_term(self, source="hull"):
        """(rho g/2) oint zeta^2 n_x dl over both sides of the waterline
        (n out of the body; n_x dl = -y_w'(x) dx on the port side),
        trapezoidal rule over the profile stations plus the ends."""
        x, _, z = self.waterline_profile(source)
        hs = 1e-6 * self.hull.L
        yp = (self.hull.waterline(x + hs, self.sink, self.trim)
              - self.hull.waterline(x - hs, self.sink, self.trim)) / (2 * hs)
        f = z ** 2 * (-yp)
        xs_, xb_ = self.hull.waterline_ends(self.sink, self.trim)
        xx = np.concatenate([[xs_], x, [xb_]])
        ff = np.concatenate([[f[0]], f, [f[-1]]])
        return float(self.rho * self.g * np.trapezoid(ff, xx))

    def double_body(self):
        """Rigid-lid double-body flow on the same body mesh (transom face as
        a wall: the closed doubled body for the d'Alembert identity)."""
        op = ImageOperator(self.body, axes=(1, 2), device=self.device, parities=[(1, 1)])
        phi = neumann_green(op, self.U * self.body.n[:, 0], (1, 1), device=self.device)
        del op
        return phi

    def hull_pressure(self, phi_B, sel=None, method="structured"):
        """Dynamic pressure rho (U phi_x - |grad phi|^2/2) on the hull panels
        (tag 0). Gradient (method "structured", default): the hull mesh is a
        structured (station, row) grid, so the tangential derivatives are
        central differences of phi and of the centroid positions along the
        two grid directions (second-order one-sided at the stem, stern,
        keel and waterline), and grad phi solves
        [r_xi; r_t; n] grad = [phi_xi, phi_t, dphi/dn]. The least-squares
        fit of steady.surface_gradient (method "lsq", used for the submerged
        spheroid) is ill-conditioned on the thin top row of a waterline-
        clustered mesh (its neighbours lie almost on one line), which made
        the near-field resistance diverge under hull-mesh refinement
        (exploration B4-E1). Returns (mask, p, grad)."""
        hull_mask = (self.body.tag == 0) if sel is None else sel
        pan = self.body.subset(hull_mask)
        ph = np.asarray(phi_B)[hull_mask]
        pn = self.phin[hull_mask]
        if method == "lsq":
            gs = surface_gradient(pan, ph, (1,), (1,))
            grad = gs + pn[:, None] * pan.n
        else:
            grad = structured_gradient(pan, ph, pn, self.hull_mesh_shape)
        p = self.rho * (self.U * grad[:, 0] - 0.5 * np.einsum("ij,ij->i", grad, grad))
        return hull_mask, p, grad

    def forces(self, cg=(0.0, 0.0), phi_db=None):
        """Dynamic force (x, z) and pitch moment about the CG (earth frame;
        CG at body (x_G, z_G)) on the hull panels (both sides), the waterline
        term, the near-field resistance (raw and d'Alembert-corrected) and
        the wet-branch virtual face force."""
        phB = self.sol["phi_B"]
        mask, p, _ = self.hull_pressure(phB)
        pan = self.body.subset(mask)
        cgE = self.hull.to_earth(np.array(cg[0]), np.array(0.0), np.array(cg[1]), self.sink, self.trim)
        r = pan.c - cgE
        F = -2 * np.sum((p * pan.area)[:, None] * pan.n, axis=0)
        My = -2 * np.sum(p * pan.area * (r[:, 2] * pan.n[:, 0] - r[:, 0] * pan.n[:, 2]))
        out = dict(Fx=float(F[0]), Fz=float(F[2]), My=float(My))
        # waterline term (rho g/2) oint zeta^2 n_x dl (both sides); the
        # linear model's momentum balance gives R_waves = R_p - this term
        # (stage note 9.2; verified by the wave cut, exploration B4-E5)
        out["R_wl"] = self.waterline_term("hull")
        out["R_raw"] = -out["Fx"]
        if phi_db is not None:
            _, pdb, _ = self.hull_pressure(phi_db)
            Fdb = -2 * np.sum((pdb * pan.area)[:, None] * pan.n, axis=0)
            out["dalembert"] = float(-Fdb[0])
            out["R_corr"] = float(-(F[0] - Fdb[0]))
            out["Fz_db"] = float(Fdb[2])
        out["R_near"] = (out.get("R_corr", out["R_raw"])) - out["R_wl"]
        if self.transom and self.branch == "wet":
            fm = self.is_face
            fp = self.body.subset(fm)
            pf = (1 - self.eta) * self.rho * self.g * np.abs(np.minimum(fp.c[:, 2], 0.0))
            rf = fp.c - cgE
            Ff = -2 * np.sum((pf * fp.area)[:, None] * fp.n, axis=0)
            out["Fx_face"] = float(Ff[0])
            out["Fz_face"] = float(Ff[2])
            out["My_face"] = float(-2 * np.sum(pf * fp.area * (rf[:, 2] * fp.n[:, 0] - rf[:, 0] * fp.n[:, 2])))
        return out


    def transom_identity(self, F=None):
        """Near-field side of the momentum identity of the linear model for a
        transom hull (stage note 9.7, derived after the B4 review): with the
        face closure (dphi/dn = n_x g z_T/U on the face, fluid crossing it)
            R_waves = R_dynH - W_hull + E_edge + K_face + P_patch,
        R_dynH the dynamic-pressure resistance of the hull panels (no face),
        W_hull = (rho g/2) int zeta^2 n_x dl along the hull waterline (both
        sides), E_edge = (rho g/2) int_edge zeta_e^2 dy (transom segment of
        the waterline contour, where n_x = -1), K_face = rho int_face
        [phi_x phi_n - |grad phi|^2 n_x/2] dS (momentum flux and dynamic
        pressure of the face closure; zero trim: (rho/2) int (phi_y^2 +
        phi_z^2 - phi_x^2) dS), P_patch = (1 - eta) rho g int_F zeta_0
        d|z_h|/dx dA (wet branch: the patch pressure's wave resistance),
        zeta_0 = (U/g) phi_x. The edge elevation zeta_e is estimated from
        the solution, independently of the D1 row: quadratic through phi at
        s = h/2, 3h/2, 5h/2 of each transom line, zeta_e = (U/g) phi_x(s = 0)
        = (U/g)(2 phi_1 - 3 phi_2 + phi_3)/h. Returns the terms, zeta_e and
        z_T per line, the D1 residual (rms(zeta_e - z_T)/rms(z_T), widths
        as weights) and R_id (with zeta_e; R_id_zT with zeta_e = z_T)."""
        if not self.transom:
            raise ValueError("no transom")
        F = F if F is not None else self.forces()
        rho, g, U = self.rho, self.g, self.U
        fm = self.is_face
        fp = self.body.subset(fm)
        gradf = structured_gradient(fp, self.sol["phi_B"][fm], self.phin[fm], self.face_shape)
        pnf = self.phin[fm]
        K = float(2 * rho * np.sum((gradf[:, 0] * pnf - 0.5 * np.einsum("ij,ij->i", gradf, gradf) * fp.n[:, 0])
                                   * fp.area))
        gr = self.G["grid"]
        idx = self.G["index"]
        h = gr["dx"]
        phi = self.sol["phi_F"]
        ze, zt, wd = [], [], []
        for j in range(-gr["K"], 0):
            k1, k2, k3 = (idx[(gr["i_stern"] + m, j)] for m in range(3))
            ze.append(U / g * (2 * phi[k1] - 3 * phi[k2] + phi[k3]) / h)
            zt.append(float(self.hull.transom_edge(np.array([self.P[k1, 1]]), self.sink, self.trim)[0][0]))
            wd.append(float(self.G["width"][k1]))
        ze, zt, wd = np.array(ze), np.array(zt), np.array(wd)
        E = float(2 * 0.5 * rho * g * np.sum(ze ** 2 * wd))
        E_zT = float(2 * 0.5 * rho * g * np.sum(zt ** 2 * wd))
        d1 = float(np.sqrt(np.sum((ze - zt) ** 2 * wd) / np.sum(zt ** 2 * wd)))
        Pp = 0.0
        if self.branch == "wet" and self.eta < 1.0:
            s_ = gr["x_s"] - self.P[:, 0]
            behind = s_ > 0
            _, dzh = self.hollow(np.where(behind, s_, -1.0), self.P[:, 1])
            z0 = U / g * self.phi_x_fs()
            Pp = float(2 * (1 - self.eta) * rho * g * np.sum(np.where(behind, z0 * dzh, 0.0) * self.fs.area))
        base = F["R_raw"] - F["R_wl"]
        return dict(R_dynH=F["R_raw"], W_hull=F["R_wl"], E_edge=E, E_edge_zT=E_zT, K_face=K, P_patch=Pp,
                    R_id=base + E + K + Pp, R_id_zT=base + E_zT + K + Pp, zeta_e=ze, z_T=zt, width=wd, D1_res=d1)


def _dgrid(F, axis):
    """Derivative along a grid axis in index units: central in the
    interior, second-order one-sided (-3 f0 + 4 f1 - f2)/2 at both ends."""
    F = np.moveaxis(np.asarray(F, float), axis, 0)
    D = np.empty_like(F)
    D[1:-1] = 0.5 * (F[2:] - F[:-2])
    D[0] = 0.5 * (-3 * F[0] + 4 * F[1] - F[2])
    D[-1] = 0.5 * (3 * F[-1] - 4 * F[-2] + F[-3])
    return np.moveaxis(D, 0, axis)


def structured_gradient(pan, phi, phin, shape):
    """Full gradient of phi on a structured panel grid (stations x rows,
    panel index = i * nz + j) from index-space differences of phi and of the
    centroids plus the normal derivative (PiercingNK.hull_pressure)."""
    nx, nz = shape
    C = pan.c.reshape(nx, nz, 3)
    Ph = np.asarray(phi, float).reshape(nx, nz)
    rx, rt = _dgrid(C, 0), _dgrid(C, 1)
    px, pt = _dgrid(Ph, 0), _dgrid(Ph, 1)
    Nn = pan.n.reshape(nx, nz, 3)
    Mtx = np.stack([rx, rt, Nn], axis=-2)                  # (nx, nz, 3, 3)
    rhs = np.stack([px, pt, np.asarray(phin, float).reshape(nx, nz)], -1)
    return np.linalg.solve(Mtx, rhs[..., None])[..., 0].reshape(-1, 3)


# ========================================================== wave-cut far field
CUT_BETA_FRAC = 0.5     # lateral wavenumbers kept: beta <= CUT_BETA_FRAC pi/dy (>= 4 points per lateral wavelength)


def cut_from_samples(yy, zz, zzx, Y, U, k, rho=1.0, beta_max=None, n_th=4000):
    """Kochin amplitude and wave resistance from a transverse cut (module
    docstring 6). yy (sorted, in (0, Y)), zz = zeta, zzx = zeta_x on the cut
    (port half; even extension); midpoint weights from the sample spacing.
    With beta = k0 sec^2(th) sin(th), kappa = k0 sec^2(th), k0 = 1/k:
      zh(beta) = 2 int_0^Y zeta cos(beta y) dy (and zh_x),
      |A(th)|^2 = (dbeta/dth / 2 pi)^2 (zh^2 + (zh_x/(kappa cos th))^2),
      R = pi rho U^2 int_0^th_max |A|^2 cos^3 th d th,
    th_max where beta = beta_max (the grid resolves no shorter lateral
    waves; the cut-off is part of the functional and is calibrated on the
    exact Kelvin field, check W-CUT). Derivation: zeta = Re int A
    e^{-i kappa (x cos th + y sin th)} d th downstream, A(-th) = A(th) for a
    y-symmetric field, the cosine transform in y inverts the th -> beta map;
    Havelock's energy-flux form R = (pi rho U^2/2) int_{-pi/2}^{pi/2}
    |A|^2 cos^3 equals (rho k0^2/pi) int_0^{pi/2} sec^3 |H|^2 of havelock.py
    for |A| = k0 sec^3(th) |H|/(pi U) (stage note 9.2)."""
    yy, zz, zzx = (np.asarray(a, float) for a in (yy, zz, zzx))
    k0 = 1.0 / k
    th_max = np.pi / 2 - 1e-9
    if beta_max is not None:
        # beta(th) = k0 sin/cos^2 is increasing: invert by bisection
        lo, hi = 0.0, np.pi / 2 - 1e-12
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if k0 * np.sin(mid) / np.cos(mid) ** 2 < beta_max:
                lo = mid
            else:
                hi = mid
        th_max = lo
    th = (np.arange(n_th) + 0.5) * th_max / n_th
    beta = k0 * np.sin(th) / np.cos(th) ** 2
    kap = k0 / np.cos(th) ** 2
    dbdt = k0 * (1 + np.sin(th) ** 2) / np.cos(th) ** 3
    edges = np.concatenate([[0.0], 0.5 * (yy[1:] + yy[:-1]), [Y]])
    wy = np.diff(edges)
    cosb = np.cos(np.outer(beta, yy))
    zh = 2 * cosb @ (zz * wy)
    zhx = 2 * cosb @ (zzx * wy)
    A2 = (dbdt / (2 * np.pi)) ** 2 * (zh ** 2 + (zhx / (kap * np.cos(th))) ** 2)
    R = np.pi * rho * U * U * np.sum(A2 * np.cos(th) ** 3) * th_max / n_th
    return float(R), dict(th=th, A2=A2, th_max=float(th_max))


def cut_resistance(solver, x_cut, rho=1.0, beta_frac=CUT_BETA_FRAC, n_th=4000):
    """Wave resistance from the transverse cut at the grid column nearest
    x_cut behind the hull (cut_from_samples), zeta and zeta_x (fixed y,
    fourth-order along the lines) at the column's points."""
    gr = solver.G["grid"]
    xe = gr["xe"]
    xm = 0.5 * (xe[:-1] + xe[1:])
    ic = int(np.argmin(np.abs(xm - x_cut)))
    z = solver.zeta if hasattr(solver, "zeta") else solver.elevation()
    zx = solver.phi_x_fs(phi=z, cU=0.0)
    ks = np.nonzero(solver.G["ij"][:, 0] == ic)[0]
    ks = ks[np.argsort(solver.P[ks, 1])]
    Y = float(gr["ny"] * gr["dy"])
    bmax = beta_frac * np.pi / gr["dy"] if beta_frac else None
    R, info = cut_from_samples(solver.P[ks, 1], z[ks], zx[ks], Y, solver.U, solver.k, rho, bmax, n_th)
    info.update(ic=ic, x_cut=float(xm[ic]))
    return R, info



# ============================================================ equilibrium
def total_load(solver, cg, F=None):
    """Vertical force and pitch moment about the CG on the hull at the
    solver's attitude (earth frame; CG at body (x_G, z_G)): exact closed-form
    buoyancy and its moment (hull.hydrostatic_load), minus eta times the
    exact hydrostatic force of the transom face (dry face: eta = 1; WaveBEM's
    virtual face carries (1 - eta) of it), plus the dynamic pressure force
    on the hull panels. Returns dict(Fz, My, parts)."""
    hull = solver.hull
    hl = hull.hydrostatic_load(solver.sink, solver.trim, solver.rho, solver.g, cg)
    fh = hull.face_hydrostatics(solver.sink, solver.trim, solver.rho, solver.g, cg)
    if F is None:
        F = solver.forces(cg)
    eta = solver.eta if solver.transom else 0.0
    Fz = hl["Fz"] - eta * fh["Fz"] + F["Fz"]
    My = hl["My"] - eta * fh["My"] + F["My"]
    Fx = -eta * fh["Fx"] + F["Fx"]
    return dict(Fz=Fz, My=My, Fx_static_face=-eta * fh["Fx"], hydro=hl, face=fh, dyn=F)


def free_equilibrium(hull, U, g=9.81, rho=1.0, cg=None, tol=1e-6, max_iter=12, verbose=False,
                     x0=(0.0, 0.0), **solver_kw):
    """Free sinkage and trim (design P-7): solve Fz_total(s, theta) = W and
    My_total(s, theta) = 0 about the CG, W = rho g V(0, 0), CG at
    (x_G, z_G) = (LCB at rest, 0) unless given (WaveBEM postulates z_G at
    the waterplane, W-38; registered as an assumption). Newton with the
    exact hydrostatic Jacobian (finite differences of the closed-form
    hydrostatics) as the initial matrix, then good-Broyden updates with the
    total (hydrostatic + hydrodynamic) residuals; every residual evaluation
    is a full solve at the new attitude (hull re-panelled below the calm
    waterplane, grid rebuilt). Stops when |Fz - W|/W and |My|/(W L) are
    both <= tol. Returns dict(sink, trim, solver, forces, history)."""
    hs0 = hull.hydrostatics(0.0, 0.0)
    W = rho * g * hs0["V"]
    if cg is None:
        cg = (float(hs0["centroid_body"][0]), 0.0)
    L = hull.L

    def resid(x):
        s, th = x
        sol = PiercingNK(hull, U, g, sink=s, trim=th, rho=rho, **solver_kw)
        sol.solve()
        F = sol.forces(cg, phi_db=sol.double_body())
        tl = total_load(sol, cg, F)
        r = np.array([(tl["Fz"] - W) / W, tl["My"] / (W * L)])
        return r, sol, F, tl

    def hs_res(x):
        s, th = x
        hl = hull.hydrostatic_load(s, th, rho, g, cg)
        fh = hull.face_hydrostatics(s, th, rho, g, cg)
        eta = 1.0 if (hull.transom and solver_kw.get("branch") == "dry") else (
            solver_kw.get("eta", 1.0) if hull.transom else 0.0)
        return np.array([(hl["Fz"] - eta * fh["Fz"] - W) / W, (hl["My"] - eta * fh["My"]) / (W * L)])

    x = np.array(x0, float)
    eps = np.array([1e-4 * hull.T if hasattr(hull, "T") else 1e-5 * L, 1e-4])
    Jm = np.zeros((2, 2))
    r0h = hs_res(x)
    for c in range(2):
        e = np.zeros(2)
        e[c] = eps[c]
        Jm[:, c] = (hs_res(x + e) - hs_res(x - e)) / (2 * eps[c])
    hist = []
    r, sol, F, tl = resid(x)
    hist.append(dict(sink=float(x[0]), trim=float(x[1]), res=r.tolist()))
    if verbose:
        print(f"    eq it 0: s {x[0]:.6g} th {x[1]:.6g} res {r}", flush=True)
    for it in range(1, max_iter + 1):
        if np.all(np.abs(r) <= tol):
            break
        dxv = -np.linalg.solve(Jm, r)
        sol.release()
        xn = x + dxv
        rn, soln, Fn_, tln = resid(xn)
        dr = rn - r
        Jm = Jm + np.outer(dr - Jm @ dxv, dxv) / (dxv @ dxv)
        x, r, sol, F, tl = xn, rn, soln, Fn_, tln
        hist.append(dict(sink=float(x[0]), trim=float(x[1]), res=r.tolist()))
        if verbose:
            print(f"    eq it {it}: s {x[0]:.6g} th {x[1]:.6g} res {r}", flush=True)
    return dict(sink=float(x[0]), trim=float(x[1]), solver=sol, forces=F, load=tl, history=hist,
                converged=bool(np.all(np.abs(r) <= tol)), W=W, cg=cg)
