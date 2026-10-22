"""
Hull forms for the surface-piercing steady problem (Track B, step B4): offset
function, wetted-surface panelling below the instantaneous waterplane at any
sinkage and trim, the waterline in the earth frame, and EXACT hydrostatics of
the hull form cut by that plane (port design WAVEBEM_PORT_DESIGN.md P-7, 8.2
"exact polygon clipping + divergence theorem" -- here done directly on the
analytic form, which is exact to quadrature round-off; acceptance M3:
equilibrium at rest to 1e-9 of the weight).

Frames. Body frame (x_b forward, y to port, z_b up; design waterline z_b = 0,
origin at midship on the design waterline). Earth frame (x, y, z) with the
calm free surface z = 0, fluid z < 0, stream -U x_hat. Attitude: sinkage s
(> 0 = the origin moves DOWN) and trim theta (> 0 = bow UP):

    x = x_b cos(theta) - z_b sin(theta),   z = z_b cos(theta) + x_b sin(theta) - s.

So the bow (x_b > 0) rises for theta > 0. The calm waterplane z = 0 is, in the
body frame, the line z_b = zeta_w(x_b) = (s - x_b sin(theta)) / cos(theta).
Draught changes at the ends (positive = deeper): d_F' = zeta_w(x_F) - 0 ...
the definitions used for comparison with experiments are in `draught_changes`.

Form. y = h(x_b, z_b) >= 0 (half-breadth) for x_b in [x_aft, x_fwd] and z_b in
[-D(x_b), 0]; above the design waterline the sides are vertical,
h(x_b, z_b > 0) = h(x_b, 0) (an ASSUMPTION for the Wigley model, whose
published offsets stop at z = 0; its effect on any steady force is of second
order in sinkage and trim). A transom stern is the flat face x_b = x_aft where
h(x_aft, z) > 0.

Hydrostatics. The submerged volume is V = 2 int dx int_{-D}^{zeta_w} h dz with
the volume centroid and the waterplane area/moments computed by tensor
Gauss-Legendre rules in (x_b, z_b), split at z_b = 0 (the kink of the vertical
extension) and at the keel; for the Wigley form the integrands are polynomials
in both variables, so the rule is exact to round-off (checked against the closed
form V = 4 L B T / 9 by `wigley_volume_exact`).
"""
import numpy as np

from .panels import Panels, from_grid


class HullForm:
    """Port/starboard symmetric hull form in the body frame (module docstring).

    h(x, z): half-breadth for z in [-D(x), 0] (vectorised, numpy arrays);
    D(x): local keel depth below the design waterline (vectorised);
    transom: True if x_aft carries a flat transom face."""

    def __init__(self, h, D, x_aft, x_fwd, name, transom=False, z_deck=None, above="vertical"):
        self._h, self.D = h, D
        self.x_aft, self.x_fwd = float(x_aft), float(x_fwd)
        self.L = self.x_fwd - self.x_aft
        self.name = name
        self.transom = bool(transom)
        self.z_deck = z_deck
        if above not in ("vertical", "form"):
            raise ValueError(above)
        self.above = above

    # ----------------------------------------------------------- geometry
    def h(self, x, z):
        """Half-breadth. above="vertical" (default; forms whose offsets stop at the design waterline, e.g. the
        Wigley models): vertical-side extension above z = 0. above="form" (forms defined above the waterline,
        e.g. prismatic_v whose V bottom continues to the chine): the form's own h for z > 0 (added after the B6
        review: the vertical extension put a knuckle at z_b = 0 on the flared prism, so any sinkage moved its
        waterline onto a vertical wall)."""
        x = np.asarray(x, float)
        z = np.asarray(z, float)
        if self.above == "form":
            return self._h(x, z)
        return self._h(x, np.minimum(z, 0.0))

    def wet_range(self, sink=0.0, trim=0.0):
        """Body-frame x-range [x_lo, x_hi] of the WETTED stations at the attitude: where the keel lies below the
        calm waterplane, g(x_b) = zeta_w(x_b) + D(x_b) >= 0. A pointed end whose keel rises to the design waterline
        (D -> 0, e.g. prismatic_v) emerges for the smallest trim or rise of the right sign; the stations must then
        span the wetted range only -- the B4/B5 mesh spanned [x_aft, x_fwd] and produced inverted (dry) end stations
        with twisted panels (B6 review, finding B5: |W| 7e4 on the prism at trim -1e-3). Ends that are wet are kept
        EXACTLY (bit-identical meshes at the attitudes of the earlier stages); a dry end is moved to the root of g
        by bisection (g changes sign once between the end and the deepest station). Added after the B6 review
        (stage note 12.2 item 6)."""
        g = lambda xb: float(self.zeta_w(np.asarray(xb, float), sink, trim) + self.D(np.asarray(xb, float)))
        lo, hi = self.x_aft, self.x_fwd
        if g(lo) >= 0.0 and g(hi) >= 0.0:
            return lo, hi
        xs = np.linspace(lo, hi, 201)
        gs = np.array([g(x) for x in xs])
        if not np.any(gs > 0):
            raise ValueError("hull entirely above the calm waterplane at this attitude")
        k = int(np.argmax(gs))
        out = []
        for a, b in ((lo, xs[k]), (hi, xs[k])):
            if g(a) >= 0.0:
                out.append(a)
                continue
            dry, wet = a, b
            for _ in range(200):
                mid = 0.5 * (dry + wet)
                if g(mid) >= 0.0:
                    wet = mid
                else:
                    dry = mid
                if abs(wet - dry) <= 1e-15 * self.L:
                    break
            out.append(wet)
        return out[0], out[1]

    @staticmethod
    def zeta_w(xb, sink, trim):
        """Body-frame z of the calm waterplane at body x (module docstring)."""
        return (sink - np.asarray(xb, float) * np.sin(trim)) / np.cos(trim)

    @staticmethod
    def to_earth(xb, y, zb, sink, trim):
        c, s = np.cos(trim), np.sin(trim)
        return np.stack([xb * c - zb * s, y, zb * c + xb * s - sink], -1)

    @staticmethod
    def to_body(x, y, z, sink, trim):
        c, s = np.cos(trim), np.sin(trim)
        zz = z + sink
        return np.stack([x * c + zz * s, y, -x * s + zz * c], -1)

    def mesh(self, nx, nz, sink=0.0, trim=0.0, x_spacing="cosine", z_spacing="cosine", face_nz=None):
        """Wetted half hull (y >= 0) below the calm waterplane, in the earth
        frame, normals out of the body (into the fluid). Stations x_b in
        [x_aft, x_fwd] (cosine-clustered at both ends by default); in each
        station z_b from the keel -D(x_b) to the waterplane zeta_w(x_b)
        (cosine-clustered at keel and waterline), so the top edges lie on
        z = 0 to round-off and the hull with its rigid-lid image is closed.
        face_nz: if given and the hull has a transom, the wetted transom
        face (x_b = x_aft, from the keel up to the waterplane) is added with
        face_nz x nz2 panels and tag 2 (hull panels have tag 0).
        Returns Panels (tag 0 hull, 2 face)."""
        s = np.linspace(0.0, 1.0, nx + 1)
        if x_spacing == "cosine":
            s = 0.5 * (1 - np.cos(np.pi * s))
        x_lo, x_hi = self.wet_range(sink, trim)
        if x_lo == self.x_aft and x_hi == self.x_fwd:
            xb = self.x_aft + self.L * s                    # both ends wet: the B4-B6 stations, bit-identical
        else:
            xb = x_lo + (x_hi - x_lo) * s                   # stations over the wetted range only (B6 review)
        t = np.linspace(0.0, 1.0, nz + 1)
        if z_spacing == "cosine":
            t = 0.5 * (1 - np.cos(np.pi * t))
        Xb, Tt = np.meshgrid(xb, t, indexing="ij")
        zw = self.zeta_w(Xb, sink, trim)
        Zb = -self.D(Xb) + Tt * np.maximum(zw + self.D(Xb), 0.0)
        Y = self.h(Xb, Zb)
        P = self.to_earth(Xb, Y, Zb, sink, trim)
        P[:, -1, 2] = 0.0                                   # top edge exactly on z = 0
        pan = from_grid(P, tag=0)
        if np.mean(pan.n[:, 1]) < 0:
            pan = from_grid(P, flip=True, tag=0)
        if face_nz and self.transom:
            pan = pan + self.face_mesh(face_nz, nz, sink, trim)
        return pan

    def face_mesh(self, ny, nz, sink=0.0, trim=0.0):
        """Wetted transom face x_b = x_aft, y in [0, h(x_aft, z_b)], z_b from
        the keel to the waterplane; normal -x_b (out of the body, aft)."""
        xa = self.x_aft
        zw = float(self.zeta_w(xa, sink, trim))
        t = 0.5 * (1 - np.cos(np.pi * np.linspace(0.0, 1.0, nz + 1)))
        zb = -self.D(np.array(xa)) + t * (zw + self.D(np.array(xa)))
        u = np.linspace(0.0, 1.0, ny + 1)
        U_, Zb = np.meshgrid(u, zb, indexing="ij")
        Y = U_ * self.h(np.full_like(Zb, xa), Zb)
        P = self.to_earth(np.full_like(Zb, xa), Y, Zb, sink, trim)
        P[:, -1, 2] = 0.0
        pan = from_grid(P, tag=2)
        if np.mean(pan.n[:, 0]) > 0:
            pan = from_grid(P, flip=True, tag=2)
        return pan

    def waterline(self, x, sink=0.0, trim=0.0, tol=1e-15):
        """Half-breadth y_wl(x) of the calm-waterplane cut at earth abscissae
        x (array), 0 outside [x_stern, x_bow]. The waterline point at body x_b
        has earth x = x_b cos(theta) - zeta_w(x_b) sin(theta) = x_b / cos(theta)
        - sink tan(theta) (monotone), inverted exactly."""
        x = np.asarray(x, float)
        xb = (x + sink * np.tan(trim)) * np.cos(trim)
        x_lo, x_hi = self.wet_range(sink, trim)             # = (x_aft, x_fwd) unless an end is dry (B6 review)
        inside = (xb >= x_lo - tol) & (xb <= x_hi + tol)
        xbc = np.clip(xb, x_lo, x_hi)
        y = self.h(xbc, self.zeta_w(xbc, sink, trim))
        return np.where(inside, np.maximum(y, 0.0), 0.0)

    def half_breadth_earth(self, x, z, sink=0.0, trim=0.0):
        """Half-breadth H(x, z) of the hull surface in the EARTH frame at the
        attitude (the hull side is the graph y = H(x, z); the rotation acts in
        the x-z plane only). 0 outside the hull's x-range."""
        x = np.asarray(x, float)
        z = np.asarray(z, float)
        P = self.to_body(x, np.zeros_like(x), z, sink, trim)
        xb, zb = P[..., 0], P[..., 2]
        inside = (xb >= self.x_aft) & (xb <= self.x_fwd)
        return np.where(inside, self.h(np.clip(xb, self.x_aft, self.x_fwd), zb), 0.0)

    def wall_derivatives(self, x, sink=0.0, trim=0.0, dz=None, dx=None):
        """Derivatives of the earth-frame half-breadth H(x, z) at the calm
        waterplane z = 0 (from below: the linear model's hull ends at z = 0),
        for the hull-wall conditions of the free-surface band (piercing.py,
        stage note 9.7): returns (H, H_x, H_z, H_zz, H_xz) at earth abscissae x.
        One-sided differences in z over [-2 dz, 0] (H_z second order, H_zz exact
        for a quadratic in z), central in x; dz = 1e-4 T (or 1e-5 L)."""
        x = np.asarray(x, float)
        dz = dz if dz is not None else 1e-4 * getattr(self, "T", 0.1 * self.L)
        dx = dx if dx is not None else 1e-6 * self.L

        def Hz_parts(xx):
            h0 = self.half_breadth_earth(xx, 0.0 * xx, sink, trim)
            h1 = self.half_breadth_earth(xx, 0.0 * xx - dz, sink, trim)
            h2 = self.half_breadth_earth(xx, 0.0 * xx - 2 * dz, sink, trim)
            return h0, (3 * h0 - 4 * h1 + h2) / (2 * dz), (h0 - 2 * h1 + h2) / dz ** 2
        H, Hz, Hzz = Hz_parts(x)
        Hp, Hzp, _ = Hz_parts(x + dx)
        Hm, Hzm, _ = Hz_parts(x - dx)
        return H, (Hp - Hm) / (2 * dx), Hz, Hzz, (Hzp - Hzm) / (2 * dx)

    def waterline_ends(self, sink=0.0, trim=0.0):
        """Earth x of the stern and bow ends of the waterline (of the wetted range, wet_range: a dry pointed end
        is excluded since the B6 review)."""
        f = lambda xb: xb / np.cos(trim) - sink * np.tan(trim)
        x_lo, x_hi = self.wet_range(sink, trim)
        return f(x_lo), f(x_hi)

    def transom_edge(self, y, sink=0.0, trim=0.0):
        """Earth z of the transom's bottom edge at half-breadth y (the hull
        section at x_aft), 0 where y >= the waterline half-breadth, and the
        slope dz/dx of the hull surface along the buttock y = const at the
        edge in the earth frame (central difference in x_b of the buttock
        height; smooth-detachment data of the dry branch)."""
        y = np.asarray(y, float)
        xa = self.x_aft

        def zbut(xb):
            # buttock height z_b(xb; y): solve h(xb, z) = y for z in [-D, 0] (h increasing in z)
            lo = -self.D(np.full_like(y, xb)) * np.ones_like(y)
            hi = np.zeros_like(y)
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                up = self.h(np.full_like(y, xb), mid) < y
                lo = np.where(up, mid, lo)
                hi = np.where(up, hi, mid)
            return 0.5 * (lo + hi)
        zb = zbut(xa)
        dx = 1e-5 * self.L
        dzdx_b = (zbut(xa + dx) - zb) / dx                 # one-sided (inside the hull)
        zw = self.zeta_w(xa, sink, trim)
        zb = np.minimum(zb, zw)
        P = self.to_earth(np.full_like(y, xa), y, zb, sink, trim)
        # buttock direction (1, dzdx_b) in the body frame -> earth slope
        c, s = np.cos(trim), np.sin(trim)
        slope = (dzdx_b * c + s) / (c - dzdx_b * s)
        return P[..., 2], slope

    # ------------------------------------------------------- hydrostatics
    def hydrostatics(self, sink=0.0, trim=0.0, nq=48):
        """Exact (to quadrature round-off) hydrostatics of the form cut by
        the calm waterplane: V, volume centroid (earth frame), waterplane
        area A_w, its centroid x_f (earth) and second moment about x_f
        (I_L), all for the full (two-sided) hull. Tensor Gauss-Legendre in
        x_b over [x_aft, x_fwd] and z_b over [-D, min(zeta_w, 0)] and
        [0, zeta_w] (vertical sides) -- exact for polynomial forms."""
        gx, wx = np.polynomial.legendre.leggauss(nq)
        gz, wz = np.polynomial.legendre.leggauss(nq)
        xb = self.x_aft + 0.5 * self.L * (gx + 1)
        wxb = 0.5 * self.L * wx
        zw = self.zeta_w(xb, sink, trim)
        D = self.D(xb)
        V = Mx = Mz = 0.0
        for a, b in ((-D, np.minimum(zw, 0.0)), (np.zeros_like(zw), np.maximum(zw, 0.0))):
            ln = np.maximum(b - a, 0.0)
            Z = a[:, None] + 0.5 * ln[:, None] * (gz[None, :] + 1)
            Wz = 0.5 * ln[:, None] * wz[None, :]
            H = self.h(np.broadcast_to(xb[:, None], Z.shape), Z)
            area = 2 * np.sum(H * Wz, 1)                    # section area
            mz = 2 * np.sum(H * Z * Wz, 1)                  # first moment in z_b
            V += np.sum(wxb * area)
            Mx += np.sum(wxb * area * xb)
            Mz += np.sum(wxb * mz)
        xc_b, zc_b = Mx / V, Mz / V
        c = self.to_earth(np.array(xc_b), np.array(0.0), np.array(zc_b), sink, trim)
        # waterplane: in the earth frame the cut has half-breadth h(x_b, zeta_w) at x = x_b/cos - s tan
        yw = self.h(xb, zw)
        jac = 1.0 / np.cos(trim)
        xe = xb / np.cos(trim) - sink * np.tan(trim)
        Aw = np.sum(wxb * 2 * yw * jac)
        xf = np.sum(wxb * 2 * yw * jac * xe) / Aw
        IL = np.sum(wxb * 2 * yw * jac * (xe - xf) ** 2)
        IT = np.sum(wxb * (2.0 / 3.0) * yw ** 3 * jac)
        return dict(V=float(V), centroid=c.astype(float), Aw=float(Aw), xf=float(xf), IL=float(IL),
                    IT=float(IT), centroid_body=np.array([xc_b, 0.0, zc_b]))

    def face_hydrostatics(self, sink=0.0, trim=0.0, rho=1.0, g=9.81, cg=(0.0, 0.0), nq=48):
        """Hydrostatic force (x, z) and pitch moment about the CG (body
        (x_G, z_G)) that the calm water would exert on the wetted transom
        face (both sides), exact by Gauss-Legendre in z_b (and the face is
        flat, x_b = x_aft): F = -int p n dS, p = -rho g z (earth), n = the
        face normal out of the body, (-cos, 0, -sin)(trim) in the earth
        frame. Zero without a transom. Used for the dry branch (the face
        is dry: the closed-body buoyancy minus this force acts on the hull)
        and the wet branch (WaveBEM's virtual face, factor 1 - eta)."""
        if not self.transom:
            return dict(Fx=0.0, Fz=0.0, My=0.0, area=0.0)
        xa = self.x_aft
        zw = float(self.zeta_w(xa, sink, trim))
        Dd = float(self.D(np.array(xa)))
        gz, wz = np.polynomial.legendre.leggauss(nq)
        Fx = Fz = My = area = 0.0
        c, s = np.cos(trim), np.sin(trim)
        n = np.array([-c, -s])
        cgE = self.to_earth(np.array(cg[0]), np.array(0.0), np.array(cg[1]), sink, trim)
        for a, b in ((-Dd, min(zw, 0.0)), (0.0, max(zw, 0.0))):
            if b <= a:
                continue
            zb = a + 0.5 * (b - a) * (gz + 1)
            w = 0.5 * (b - a) * wz
            yh = self.h(np.full_like(zb, xa), zb)              # half-breadth: dS = 2 yh dz_b
            P = self.to_earth(np.full_like(zb, xa), np.zeros_like(zb), zb, sink, trim)
            p = -rho * g * P[:, 2]
            dA = 2 * yh * w
            Fx += np.sum(-p * n[0] * dA)
            Fz += np.sum(-p * n[1] * dA)
            rx, rz = P[:, 0] - cgE[0], P[:, 2] - cgE[2]
            My += np.sum(-p * dA * (rz * n[0] - rx * n[1]))
            area += np.sum(dA)
        return dict(Fx=float(Fx), Fz=float(Fz), My=float(My), area=float(area))

    def hydrostatic_load(self, sink=0.0, trim=0.0, rho=1.0, g=9.81, cg=(0.0, 0.0)):
        """Buoyancy (vertical) and its pitch moment about the CG of the
        closed form cut by the calm waterplane (exact): Fz = rho g V,
        My = -(x_c - x_G,earth) rho g V (moment of a vertical force about +y,
        same sign convention as PiercingNK.forces: M = r x F)."""
        hs = self.hydrostatics(sink, trim)
        cgE = self.to_earth(np.array(cg[0]), np.array(0.0), np.array(cg[1]), sink, trim)
        Fz = rho * g * hs["V"]
        return dict(Fx=0.0, Fz=Fz, My=float(-(hs["centroid"][0] - cgE[0]) * Fz), V=hs["V"])

    def draught_changes(self, sink, trim):
        """Draught changes (positive = deeper) at the forward and aft
        perpendiculars x_b = x_fwd, x_aft, measured vertically in the earth
        frame: d' = -(z of the design-waterline point at the perpendicular)
        = s - x_b sin(theta) (design waterline point z_b = 0)."""
        dF = sink - self.x_fwd * np.sin(trim)
        dA = sink - self.x_aft * np.sin(trim)
        return dF, dA


def wigley(L=1.0, B=0.1, T=0.0625):
    """Standard parabolic Wigley hull (Kajitani et al. 1983 cooperative
    model): y = (B/2)(1 - (2x/L)^2)(1 - (z/T)^2), L/B = 10, B/T = 1.6."""
    def h(x, z):
        return 0.5 * B * (1 - (2 * x / L) ** 2) * (1 - (z / T) ** 2)

    def D(x):
        return T * np.ones_like(np.asarray(x, float))
    hull = HullForm(h, D, -L / 2, L / 2, f"wigley_L{L:g}_B{B:g}_T{T:g}")
    hull.B, hull.T = B, T
    return hull


def wigley_transom(L=1.0, B=0.1, T=0.0625, x_t=-0.4, x_1=0.0, T_t=0.25):
    """Wigley-like hull with a shallow transom (Track B B4 test geometry; no
    published data exist for it): the standard Wigley waterlines
    (B/2)(1 - (2x/L)^2), parabolic sections (1 - (z/D(x))^2), and a keel
    rising from depth T (x >= x_1 L) to T_t T at the transom x = x_t L with
    a C1 smoothstep, D = T_t T + (T - T_t T) S((x - x_t L)/((x_1 - x_t) L)),
    S(u) = 3u^2 - 2u^3, so the buttocks leave the transom horizontally.
    Transom half-breadth at the waterline (B/2)(1 - 4 x_t^2) (0.018 L for
    the defaults), transom draught T_t T."""
    xt, x1 = x_t * L, x_1 * L
    Tt = T_t * T

    def D(x):
        u = np.clip((np.asarray(x, float) - xt) / (x1 - xt), 0.0, 1.0)
        return Tt + (T - Tt) * (3 * u * u - 2 * u ** 3)

    def h(x, z):
        d = D(x)
        return 0.5 * B * (1 - (2 * x / L) ** 2) * np.clip(1 - (z / d) ** 2, 0.0, None)
    hull = HullForm(h, D, xt, L / 2, f"wigleyT_L{L:g}_B{B:g}_T{T:g}_xt{x_t:g}_Tt{T_t:g}", transom=True)
    hull.B, hull.T, hull.T_t = B, T, Tt
    return hull


def wigley_flare(L=1.0, B=0.1, T=0.0625, f=0.3):
    """Wigley hull with FLARE at the design waterline (Track B check H13, a
    test geometry for the hull-wall conditions; no data exist): y = (B/2)
    (1 - (2x/L)^2)(1 - (z/T)^2)(1 + f z/T), so h_z(z = 0) = (B/2)(1 - (2x/L)^2)
    f/T > 0 (sides opening upwards; flare angle atan(B f/(2T)) = 13.5 deg at
    midship for the defaults) and h >= 0 on [-T, 0] for f < 1."""
    def h(x, z):
        return 0.5 * B * (1 - (2 * x / L) ** 2) * (1 - (z / T) ** 2) * (1 + f * z / T)

    def D(x):
        return T * np.ones_like(np.asarray(x, float))
    hull = HullForm(h, D, -L / 2, L / 2, f"wigleyF_L{L:g}_B{B:g}_T{T:g}_f{f:g}")
    hull.B, hull.T = B, T
    return hull


def wigley_volume_exact(L, B, T):
    """Closed form of the Wigley displacement at rest: V = 4 L B T / 9."""
    return 4.0 * L * B * T / 9.0


def wigley_journee(L=1.0, B=0.1, T=0.0625):
    """Modified Wigley hull of the Delft seakeeping experiments (Journee 1992,
    TU Delft report 909, 'Wigley I' for L/B = 10, L/T = 16; formula as quoted
    by Kara 2022, Ships Offshore Struct., eq. 10, and by J. Mar. Sci. Eng.
    9:87 (2021), eq. 41 -- the report itself is not retrievable here):
        2 y / B = (1 - zeta^2)(1 - xi^2)(1 + 0.2 xi^2) + zeta^2 (1 - zeta^8)(1 - xi^2)^4,
    xi = 2 x / L, zeta = z / T (z <= 0). Wall-sided at z = 0 (d h/d z = 0) and
    pointed at both ends; fuller sections than the parabolic Wigley (C_B =
    0.5606: V = 0.0946 m^3 for the 3 m model, as tabulated by those sources)."""
    def h(x, z):
        xi = 2 * np.asarray(x, float) / L
        ze = np.asarray(z, float) / T
        a = 1 - xi * xi
        return 0.5 * B * ((1 - ze * ze) * a * (1 + 0.2 * xi * xi) + ze * ze * (1 - ze ** 8) * a ** 4)

    def D(x):
        return T * np.ones_like(np.asarray(x, float))
    hull = HullForm(h, D, -L / 2, L / 2, f"wigleyJ_L{L:g}_B{B:g}_T{T:g}")
    hull.B, hull.T = B, T
    return hull


def half_spheroid(a=0.5, b=0.1):
    """Surface-piercing half prolate spheroid (semi-axes a along x, b across and down; centre on the calm
    waterplane): h = b sqrt(1 - (x/a)^2 - (z/b)^2), keel depth D = b sqrt(1 - (x/a)^2). Its rigid-lid double body
    is the full spheroid, whose flow is known in closed form (analytic.spheroid_double_body): the test geometry of
    the operator-consistency rows DXa (B6 review, stage note 12.2 item 2). Blunt waterline ends (elliptic
    waterline, vertical tangent at x = +-a); wall-sided at z = 0 (h_z = 0)."""
    def h(x, z):
        x = np.asarray(x, float)
        z = np.asarray(z, float)
        return b * np.sqrt(np.clip(1 - (x / a) ** 2 - (z / b) ** 2, 0.0, None))

    def D(x):
        return b * np.sqrt(np.clip(1 - (np.asarray(x, float) / a) ** 2, 0.0, None))
    hull = HullForm(h, D, -a, a, f"halfspheroid_a{a:g}_b{b:g}")
    hull.B, hull.T = 2 * b, b
    return hull


def prismatic_v(L=1.0, b0=None, beta_deg=20.0, D0=0.035, x_p=(-0.2, 0.05)):
    """Planing-type prismatic V body (Track B step B5 demo geometry; no data
    exist for it): constant deadrise beta, sections similar V's scaled by a
    plan-form factor p(x) so that the deadrise is the same at every station,
        keel depth D(x) = D0 p(x),  half-breadth h = (z + D(x)) / tan(beta)
        below the chine (chine half-breadth b(x) = b0 p(x), vertical sides
        above it),
    p = 1 on the prismatic part x_p[0] L <= x <= x_p[1] L and a parabolic taper
    p = s (2 - s) to a POINT at both ends (s = distance from the end / taper
    length; bow taper longer than the stern taper, so the body is fore-aft
    asymmetric). At the design waterline z = 0 the waterline half-breadth is
    D(x)/tan(beta) < b(x) (the chine is above the water at rest): the waterline
    cuts the inclined V bottom, i.e. the hull is strongly FLARED there (h_z =
    cot(beta)). No transom: the immersed body is pointed at the stern as well
    (the transom branches of B4 are implemented but not verified, stage note
    9.9 item 7, so the B5 demo avoids them). Default b0 = L/9 (L/B = 4.5),
    D0 = 0.035 L, beta = 20 deg: waterline half-breadth 0.0962 L. Above the design
    waterline the V continues to the chine (above="form", B6 review; the chine is
    p(x) (b0 tan(beta) - D0) = 0.0054 L above the design waterline amidships)."""
    b0 = L / 9.0 if b0 is None else b0
    tb = np.tan(np.radians(beta_deg))
    xa, xb = -0.5 * L, 0.5 * L
    x1, x2 = x_p[0] * L, x_p[1] * L
    D0 = D0 * L

    def p(x):
        x = np.asarray(x, float)
        s_a = np.clip((x - xa) / (x1 - xa), 0.0, 1.0)
        s_b = np.clip((xb - x) / (xb - x2), 0.0, 1.0)
        return np.minimum(s_a * (2 - s_a), s_b * (2 - s_b))

    def D(x):
        return D0 * p(x)

    def h(x, z):
        x = np.asarray(x, float)
        z = np.asarray(z, float)
        pp = p(x)
        return np.clip(np.minimum((z + D0 * pp) / tb, b0 * pp), 0.0, None)
    hull = HullForm(h, D, xa, xb, f"prismV_L{L:g}_b{b0:g}_beta{beta_deg:g}_D{D0:g}", above="form")
    hull.B, hull.T, hull.beta = 2 * b0, D0, np.radians(beta_deg)
    return hull
