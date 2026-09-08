#!/usr/bin/env python3
"""
Sectional hull geometry and nonlinear Froude-Krylov / restoring forces.

Why this exists: for a 10 m USV in SS5, Hs/draught is about 4. The hull leaves
the water and re-enters routinely, so the linear assumption -- pressure
integrated over a FIXED mean wetted surface -- is not a refinement to skip.

A "station" is a transverse slice of the hull at one longitudinal position. The
hull is a stack of them; forces are computed per station and integrated along
the length. Everything the vessel model needs from the hull FORM reduces to two
primitives per station:

    area(d)            immersed sectional area at immersion d
    centroid_depth(d)  depth of that area's centroid, for the Smith factor

`WigleySections` supplies them in closed form; `MeshSections` tabulates them
from an arbitrary mesh, offline. Nothing downstream -- the BEM database, the
Ogilvie transform, the state-space realisation, the Cummins assembly, the MPC
-- ever sees the hull form, so changing hulls touches only these two classes.

For the Wigley form the closed forms are

    y(x, z) = y0(x) (1 - (z/T)^2),  y0(x) = (B/2)(1 - (2x/L)^2),   -T <= z <= 0

extended wall-sided above the design waterline so a section can be immersed
deeper than its design draught:

    d <= -T          emerged, A = 0
    -T < d <= 0      A = 2 y0 [ d - d^3/(3T^2) + 2T/3 ]
    d > 0            A = 2 y0 [ 2T/3 + d ]

At d = 0 that integrates over the hull to exactly (4/9) L B T, the analytic
Wigley volume, which is the first thing the tests check.

Wave pressure carries the Smith effect: dynamic pressure decays as e^{kz} with
depth. It is applied at the sectional area centroid, exact to O((kT)^2) and
well inside that here -- at omega = 2 rad/s, kT = 0.33.
"""
import numpy as np

G = 9.81
RHO = 1025.0


class HullSections:
    """Strip representation of any hull. Subclasses supply area/centroid."""

    x = dx = y0 = None
    T = 0.0
    n = 0

    # ---------------------------------------------- hull-specific primitives
    def area(self, d):
        raise NotImplementedError

    def centroid_depth(self, d):
        raise NotImplementedError

    # ------------------------------------------------- generic from here on
    def volume(self, d=None):
        d = np.zeros(self.n) if d is None else d
        return float(np.sum(self.area(d)) * self.dx)

    def immersion(self, eta_eff, z, pitch, sign_pitch=-1.0):
        """Immersion of each station relative to its own design waterline.

        `sign_pitch` maps the pitch state onto geometry. It is DERIVED from the
        dof definition, not fitted: Capytaine's Pitch is a rotation about +y,
        so omega x r gives dz = -theta*x, hence sign_pitch = -1. C55 cannot
        settle it -- the sign appears in both the immersion and the moment arm
        and squares out.
        """
        return eta_eff - (z + sign_pitch * self.x * pitch)

    def fk_restoring(self, eta_eff, z, pitch, sign_pitch=-1.0, rho=RHO, g=G):
        """Nonlinear buoyancy + Froude-Krylov in heave and pitch.

        Returns (F_heave, M_pitch, immersion, area). This is the FULL buoyancy,
        so the caller must not also apply the linear C matrix in those dofs.
        """
        d = self.immersion(eta_eff, z, pitch, sign_pitch)
        a = self.area(d)
        f_sec = rho * g * a * self.dx
        return (float(np.sum(f_sec)),
                float(sign_pitch * np.sum(self.x * f_sec)), d, a)

    def fk_restoring_linear(self, eta_eff, z, pitch, sign_pitch=-1.0,
                            rho=RHO, g=G):
        """Linearisation of `fk_restoring` about the equilibrium.

        dA/dd -> 2 y0 at d = 0 for any hull with a finite waterline, so the
        linear force is the waterplane-weighted relative elevation. Subtracting
        it from the nonlinear force leaves exactly the nonlinear DEPARTURE,
        which rides on top of the exact linear BEM result -- so the model is
        exact in the linear limit and strip-theory error, which grows as the
        wavelength approaches the hull length, only ever acts on the correction.
        """
        d_lin = self.immersion(eta_eff, z, pitch, sign_pitch)
        f_sec = rho * g * 2 * self.y0 * d_lin * self.dx
        return float(np.sum(f_sec)), float(sign_pitch * np.sum(self.x * f_sec))

    def emerged_fraction(self, d):
        return float(np.mean(self.area(d) <= 0.0))

    def bow_immersion(self, d):
        return float(d[-1])


class SlamLoad:
    """Rate of change of sectional added mass with immersion, tabulated.

    This is the term a constant-added-mass model structurally cannot carry.
    The full added-mass force on a section is

        F = d/dt ( m_a v )  =  m_a dv/dt  +  (dm_a/dd)(dd/dt) v

    Cummins gives us the FIRST term (through A_inf) and the wave memory. The
    SECOND term exists only while the wetted width is changing, which is
    exactly water entry -- so it is missing precisely where slamming happens,
    and nowhere else. Adding it is complementary, not double counting: the
    buoyancy part of entry is already in the nonlinear Froude-Krylov force,
    and this is the inertial part.

    The sectional added mass uses the flat-plate value for a wetted half-beam b

        m_a(d) = (pi/2) rho b(d)^2

    and b comes straight from the section table rather than from any assumed
    shape, because the full beam at the waterline IS dA/dd:

        b(d) = (1/2) dA/dd

    so this works for `MeshSections` (a real hull) exactly as it does for the
    analytic Wigley form. Tabulated once at construction: it is pure geometry,
    and computing it per step cost four `area` calls per station.
    """

    # Wagner rather than von Karman. A wedge entering the water piles the free
    # surface up ahead of the geometric intersection, so the plate that is
    # really being accelerated is pi/2 wider than the hull section suggests,
    # and the added mass goes as the square of that. This factor is the one
    # modelling choice in the module; von Karman (1.0) is the lower bound and
    # is known to underpredict.
    PILE_UP = (np.pi / 2) ** 2

    def __init__(self, sec, n=241, rho=RHO, h_frac=0.02):
        # the band that matters: from just below the keel to just above the
        # design waterline. Outside it the term is zero anyway -- below the
        # keel because b = 0, above the waterline because a wall-sided
        # topside has db/dd = 0.
        self.d = np.linspace(-1.2 * sec.T, 0.4 * sec.T, n)
        self.dd = float(self.d[1] - self.d[0])
        h = h_frac * sec.T
        ones = np.ones(sec.n)
        b = np.array([0.25 * (sec.area(di * ones + h)
                              - sec.area(di * ones - h)) / h
                      for di in self.d])                    # (n, n_station)
        ma = 0.5 * np.pi * rho * np.maximum(b, 0.0) ** 2
        self.dma = np.gradient(ma, self.dd, axis=0)         # kg/m^2
        self.idx = np.arange(sec.n)

    def slope(self, d):
        """dm_a/dd at each station, linearly interpolated."""
        g = np.clip((np.asarray(d, float) - self.d[0]) / self.dd,
                    0.0, len(self.d) - 1.001)
        i = g.astype(int)
        f = g - i
        return ((1.0 - f) * self.dma[i, self.idx]
                + f * self.dma[i + 1, self.idx])


class WigleySections(HullSections):
    """Analytic Wigley sections."""

    def __init__(self, L=10.0, B=2.5, T=0.8, n_stations=41):
        self.L, self.B, self.T = L, B, T
        # midpoint rule: stations at strip centres, so sums are the integral
        edges = np.linspace(-L / 2, L / 2, n_stations + 1)
        self.x = 0.5 * (edges[:-1] + edges[1:])
        self.dx = edges[1] - edges[0]
        self.y0 = (B / 2) * (1 - (2 * self.x / L) ** 2)
        self.n = n_stations

    def area(self, d):
        T, y0 = self.T, self.y0
        d = np.asarray(d, float)
        below = np.clip(d, -T, 0.0)
        a = 2 * y0 * (below - below ** 3 / (3 * T ** 2) + 2 * T / 3)
        a = np.where(d <= -T, 0.0, a)
        return a + np.where(d > 0.0, 2 * y0 * d, 0.0)

    def centroid_depth(self, d):
        """Depth (negative down) of the immersed area centroid, per station."""
        T = self.T
        dc = np.clip(d, -T, 0.0)
        # first moment of 2 y0 (1-(z/T)^2) over [-T, dc] divided by the area;
        # the y0 factor cancels, so this is geometry only
        m = (dc ** 2 / 2 - dc ** 4 / (4 * T ** 2)) - (T ** 2 / 2 - T ** 2 / 4)
        a = dc - dc ** 3 / (3 * T ** 2) + 2 * T / 3
        zc = np.where(a > 1e-9, m / np.maximum(a, 1e-9), -T / 2)
        # area above the design waterline is wall-sided, centred at d/2
        extra = np.where(d > 0, d, 0.0)
        zc = (zc * a + (extra / 2) * extra) / np.maximum(a + extra, 1e-9)
        return np.clip(zc, -T, np.maximum(d, 0.0))


def _section_segments(tri, normals, x_plane, tol=1e-12):
    """Intersect a triangle soup with the plane x = x_plane.

    Returns (ya, za, yb, zb) for every crossing triangle, oriented so the
    section contour runs counter-clockwise in the (y, z) plane. Orientation is
    taken from the panel normal rather than guessed: for a CCW contour the
    outward 2-D normal is (t_z, -t_y), so the tangent is (-n_z, n_y). Getting
    this from the geometry means multi-loop sections -- a catamaran, a hull with
    a moonpool -- come out right without special-casing.
    """
    s = tri[:, :, 0] - x_plane                      # signed distance, (nt, 3)
    pos, neg = s > tol, s < -tol
    cross = pos.any(1) & neg.any(1)
    if not cross.any():
        return np.zeros((0, 4))
    tri, s, n = tri[cross], s[cross], normals[cross]

    out = np.empty((len(tri), 4))
    for i in range(len(tri)):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            sa, sb = s[i, a], s[i, b]
            if sa * sb < 0:
                f = sa / (sa - sb)
                p = tri[i, a] + f * (tri[i, b] - tri[i, a])
                pts.append((p[1], p[2]))
            elif abs(sa) <= tol:
                pts.append((tri[i, a, 1], tri[i, a, 2]))
        if len(pts) < 2:
            out[i] = np.nan
            continue
        (ya, za), (yb, zb) = pts[0], pts[1]
        # orient along t = (-n_z, n_y)
        if (yb - ya) * (-n[i, 2]) + (zb - za) * n[i, 1] < 0:
            ya, za, yb, zb = yb, zb, ya, za
        out[i] = (ya, za, yb, zb)
    return out[~np.isnan(out[:, 0])]


def _clip_below(seg, d):
    """Keep the part of each segment with z <= d, splitting where it crosses."""
    ya, za, yb, zb = seg.T
    keep_a, keep_b = za <= d, zb <= d
    both = keep_a & keep_b
    part = keep_a ^ keep_b
    out = [seg[both]]
    if part.any():
        p = seg[part].copy()
        pa, pb, pza, pzb = p[:, 0], p[:, 2], p[:, 1], p[:, 3]
        f = (d - pza) / np.where(np.abs(pzb - pza) < 1e-15, 1e-15, pzb - pza)
        ymid = pa + f * (pb - pa)
        below_a = pza <= d
        q = p.copy()
        q[below_a, 2] = ymid[below_a]; q[below_a, 3] = d
        q[~below_a, 0] = ymid[~below_a]; q[~below_a, 1] = d
        out.append(q)
    return np.vstack(out) if out else np.zeros((0, 4))


def _area_moment(seg, d):
    """Exact immersed area and first moment of a clipped section.

    Divergence theorem with F = (0, z-d) gives div F = 1, so

        A(d) = closed-contour integral of (z-d) n_z ds

    and (z-d) vanishes on the free surface, so the segments that close the
    contour along z = d contribute NOTHING. Only the hull's own segments are
    needed -- no contour assembly, no ordering. For a straight segment the
    integral is closed-form:

        A   = -sum (y_b - y_a) [ (z_a+z_b)/2 - d ]
        M   = -sum (y_b - y_a) [ (z_a^2 + z_a z_b + z_b^2)/6 - d^2/2 ]

    where M is the first moment about z = 0, giving the centroid as M/A.
    """
    if not len(seg):
        return 0.0, 0.0
    ya, za, yb, zb = seg.T
    dy = yb - ya
    A = -np.sum(dy * (0.5 * (za + zb) - d))
    M = -np.sum(dy * ((za ** 2 + za * zb + zb ** 2) / 6.0 - 0.5 * d ** 2))
    return float(A), float(M)


class MeshSections(HullSections):
    """Sections tabulated EXACTLY from an arbitrary hull mesh.

    Drop-in replacement for WigleySections when the real hull arrives. Each
    station plane is intersected with the panels to give the section contour,
    and the immersed area and centroid come from a closed-form contour integral
    -- exact for a piecewise-planar hull, not a discretised depth sum. The
    result is tabulated once, offline; at run time it is a lookup, which is
    cheaper than evaluating the analytic polynomial.

    Because the area comes from a contour integral rather than a maximum
    half-breadth, it handles sections a single offset cannot describe:
    catamarans, tunnels, flare, chines.
    """

    def __init__(self, mesh, L, T, n_stations=41, n_immersion=101,
                 freeboard=None):
        self.L, self.T, self.n = L, T, n_stations
        edges = np.linspace(-L / 2, L / 2, n_stations + 1)
        self.x = 0.5 * (edges[:-1] + edges[1:])
        self.dx = edges[1] - edges[0]

        tri, nrm = self._triangulate(mesh)
        fb = T if freeboard is None else freeboard
        self.d_grid = np.linspace(-T, fb, n_immersion)
        self._A = np.zeros((n_stations, n_immersion))
        self._zc = np.full((n_stations, n_immersion), -T / 2)
        self.y0 = np.zeros(n_stations)

        # A station plane that lands exactly on a mesh grid line finds NO
        # crossing triangles: every adjacent vertex sits at s = 0, so neither
        # the strictly-positive nor the strictly-negative test fires and the
        # whole section is skipped. With an odd station count and an even mesh
        # that happens at midship -- the widest section -- and quietly removed
        # 3.6% of the displacement while looking like a convergence problem.
        # Offsetting the plane by an incommensurable fraction of the length
        # avoids every such coincidence; the induced error is O(jitter dA/dx),
        # utterly negligible at 1e-7 L.
        jitter = L * 1e-7
        eps = T * 1e-3
        for i, xc in enumerate(self.x):
            seg = _section_segments(tri, nrm, xc + jitter)
            if not len(seg):
                continue
            a0, _ = _area_moment(_clip_below(seg, 0.0), 0.0)
            am, _ = _area_moment(_clip_below(seg, -eps), -eps)
            self.y0[i] = max((a0 - am) / eps, 0.0) / 2      # dA/dd = width
            for j, d in enumerate(self.d_grid):
                dd = min(d, 0.0)
                A, M = _area_moment(_clip_below(seg, dd), dd)
                if d > 0.0:                       # wall-sided above the WL
                    A += 2 * self.y0[i] * d
                    M += 2 * self.y0[i] * d * (d / 2)
                self._A[i, j] = max(A, 0.0)
                self._zc[i, j] = M / max(A, 1e-9) if A > 1e-9 else -T / 2

    @staticmethod
    def _triangulate(mesh):
        """Quads -> triangles, with outward normals from the face winding."""
        v = np.asarray(mesh.vertices, float)
        f = np.asarray(mesh.faces, int)
        tris = ([f[:, [0, 1, 2]], f[:, [0, 2, 3]]] if f.shape[1] == 4
                else [f])
        tri = np.concatenate([v[t] for t in tris], axis=0)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        return tri, n / np.maximum(ln, 1e-30)

    def area(self, d):
        d = np.atleast_1d(np.asarray(d, float))
        return np.array([np.interp(d[i], self.d_grid, self._A[i])
                         for i in range(self.n)])

    def centroid_depth(self, d):
        d = np.atleast_1d(np.asarray(d, float))
        return np.array([np.interp(d[i], self.d_grid, self._zc[i])
                         for i in range(self.n)])
