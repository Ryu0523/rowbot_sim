#!/usr/bin/env python3
"""
Planing-hull geometry as data -- module M1 of the modular split
(learn/meta/ANYSPEED_PRINCIPLES_2026-10-05.md section 1; ACCEPTANCE G4).

WHAT THE STRIP PLANTS NEED
The planing plants (sim/planing_vessel.py 'v0', sim/planing_anyspeed.py
'as1') are 2D+t strip theories on V sections. Per strip, numbered from the
transom forward (strip centre xi, measured from the transom), they need

    yc    chine half-beam
    rise  keel height above the keel at the transom (the plant's keel in the
          body frame is zk = -vcg + rise)
    kp    d rise / dx            (the keel's own slope: planing lift u kp)
    kpp   d^2 rise / dx^2        (its curvature: the m_a u^2 kpp term)
          at the strip centre -- v0's discretisation, kept for the family
          (v0 builds it) -- and kpp_strip, the strip integral of the
          curvature divided by the strip length,
              kpp_strip_j = (kp(x_{j+1/2}) - kp(x_{j-1/2})) / dx,
          which needs only kp, so it converges at kp's order, and resolves
          a jump of kpp inside a strip (v0's parabolic keel at x0) exactly.
          An offsets table uses it directly (its PCHIP second derivative
          converges at first order only: 51 % of the largest curvature at
          13 offsets); a family hull uses it under the as1 switch
          kpp_strip (sim/planing_anyspeed.py)
    beta  deadrise (constant: see below)
    depth keel to sheer (constant along the keel; the GZ sections use it)

and, for the strip-exact location of chine wetting in the momentum flux of
as1 (sim/planing_anyspeed._StripAS._substrip, switch dDt_full), the chine as
a function of x -- not only at the strip centres -- together with the roots
of kc d(x) = yc(x) for a keel depth d linear along a strip, kc = pi/2 /
tan(beta) the Wagner wetted half-beam per unit depth. The keel profile at
any x (Stations.keel: rise, kp, kpp) is there too, so that a study needing
the hull between stations (a continuum reference, a mesh, a viewer) reads
the geometry here instead of re-deriving it from hull fields -- the fields
taper and stem_rise describe v0's family member only, and an offsets hull
carries taper 0 (studies/anyspeed_s1_check.continuum_D55, studies/
visual_export.py, studies/anyspeed_bem_probe.py, studies/
principles_ledger_checks.py read stations_of).

v0 wrote these as formulas inside its constructor (sim/planing_vessel.py
:91-102, from the hull's L, b, taper, stem_rise). Here they are the output
of a hull description, so a hull is changed by changing data:

  PlaningHullForm   the parametric family: principal dimensions, deadrise,
                    a chine plan (elliptic forebody taper, optional linear
                    transom run), a keel profile (stem rise s^p over its own
                    bow fraction: forefoot and bow rake), depth. With v0's
                    parameters (transom_run 0, rise_from = taper, rise_power
                    2, yc_floor 1e-4) its stations are v0's array for array:
                    the SAME floating-point expressions are evaluated
                    (stations() takes v0's branch), so a v0 hull and an as1
                    plant built from this module agree to the last bit
                    (studies/anyspeed_p0_check.py, part 'geom').
  OffsetsForm       an offsets table (x, chine half-beam, keel height), the
                    way real lines arrive: chine piecewise linear between the
                    offsets (exact for its own representation, root by root),
                    keel a monotone cubic (PCHIP) so that kp is continuous
                    and kpp bounded.

stations_of(hull) is the one entry the plant uses: a hull carrying `form`
(hydro/vessel_spec.ParametricPlaningHull) gets that form; a plain
hydro.planing.PlaningHull gets the family member of its own fields.

CONSTANT DEADRISE
A warped bottom (deadrise varying along x) makes the Wagner added mass vary
along x at fixed depth even below the chine, which adds a term
-u (dm_a/dx)|_d = -u k_a pi rho kc kc' d^2 to the complete material
derivative of S1 (design 2.5). That term is not derived and implemented
yet, so every form here has one deadrise, and the as plants check it.

CHINE CONCAVITY
The plant decides whether a strip has a chine-wetted point from its two ends
(kc d+ - yc is convex along a strip when the chine is concave). The family's
chine is concave by construction (rising linear transom run, flat middle,
elliptic taper). An offsets table need not be; its interior knots are then
tested too (Stations.knots, Stations.concave).

WHERE THE FAMILY APPLIES (ACCEPTANCE M1; RANGES, form.validity())
  deadrise 10-30 deg  the data behind Zarnick's 2D+t coefficients (Fridsma
                      1969's prismatic models; learn/meta/
                      EMPIRICAL_REGISTER.md E1); outside, a parametric hull
                      builds and warns (hydro/vessel_spec)
  L / b >= 3          slender-body strip theory needs a slender wetted body
                      (an assumption of the theory, no data bound); the
                      validation case (Fridsma 1969 models A, B) is L/b 5.
                      scarab195 (2.67) is below it -- recorded, not hidden
                      (design section 1: 'no 3D relief at L/b 2.67')
Known limitations of the family (register H5): the bow taper is a quarter
ellipse whose chine slope is infinite at the stem (a blunt, spoon-shaped
bow with no finite entrance angle; rest hydrostatics converge at order ~1.5
with station count when the bow is wet); deadrise is constant, so the chine
runs down to the keel at the stem; above the chine the hull is wall-sided
(no flare, no rising chine), so bow reserve buoyancy is low -- which matters
for bow burying and capsize (ACCEPTANCE R4). A bow with a finite entrance
angle, flare or a rising chine is described by an offsets table
(OffsetsForm: piecewise-linear chine, exact roots, any entrance angle) --
the planned route for real hull lines (register H3); a warped bottom waits
for its D/Dt term (above).

Lengths in m, angles in degrees, x from the transom forward.
"""
from dataclasses import dataclass, asdict
from types import SimpleNamespace

import numpy as np

# v0's values of the parameters the plant formulas did not expose
V0_YC_FLOOR = 1e-4          # sim/planing_vessel.py:98, '+ 1e-4'
# the reference hull's keel length, m (scarab195): its values are the
# defaults of hydro/vessel_spec, Froude-scaled to a new hull by L / L_REF;
# one definition, imported by hydro/vessel_spec, sim/vessel_profile and
# control/mpc
L_REF = 5.4


def _floor(yc_floor, L):
    """The chine floor: given, or v0's 0.1 mm scaled with L / L_REF (a
    numerical guard, not an absolute number)."""
    return V0_YC_FLOOR * L / L_REF if yc_floor is None else float(yc_floor)


# ------------------------------------------------------------ the result
class Stations(SimpleNamespace):
    """What a strip plant reads (see the module docstring). Attributes:
    n, L, edges (n+1), xi (n), dx, yc, rise, kp, kpp, kpp_strip, beta_deg,
    depth, eps (the chine floor), chine(x), dchine(x), keel(x) -> (rise, kp,
    kpp), roots(xc, d, m, kc), varying (strips whose chine is not
    constant), concave, knots, source."""

    def check(self):
        """Raise unless the arrays are finite, the chine positive and the
        chine function reproduces the station chine exactly."""
        for k in ("xi", "yc", "rise", "kp", "kpp", "kpp_strip"):
            a = getattr(self, k)
            if a.shape != (self.n,) or not np.all(np.isfinite(a)):
                raise ValueError(f"stations: {k} is not a finite ({self.n},) "
                                 f"array")
        if np.any(self.yc <= 0.0):
            raise ValueError("stations: chine half-beam must be positive")
        if not np.array_equal(self.chine(self.xi), self.yc):
            raise ValueError("stations: chine(xi) differs from yc")
        return self


def _strips(L, n):
    """Strip edges, centres and length: sim/planing_vessel.py:91-95."""
    edges = np.linspace(0.0, L, n + 1)
    xi = 0.5 * (edges[:-1] + edges[1:])
    return edges, xi, L / n


def _kpp_strip(kp_fn, edges, dx):
    """The strip integral of the keel curvature over each strip divided by
    its length: (kp(right edge) - kp(left edge)) / dx (module docstring)."""
    return (kp_fn(edges[1:]) - kp_fn(edges[:-1])) / dx


# validity ranges of the family (module docstring, 'WHERE THE FAMILY
# APPLIES'): (low, high, why)
RANGES = dict(
    beta_deg=(10.0, 30.0, "deadrise of the data behind Zarnick's 2D+t "
              "coefficients (Fridsma 1969 prismatic models; register E1)"),
    L_over_b=(3.0, None, "slender-body strip theory assumes a slender wetted "
              "body (theory, no data bound; the validation case is L/b 5)"))


def _validity(beta_deg, L, b):
    out = []
    for k, v in (("beta_deg", beta_deg), ("L_over_b", L / b)):
        lo, hi, why = RANGES[k]
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            out.append(f"{k} {v:.3g} outside [{lo}, {hi}]: {why}")
    return out


def _linear_root(xc, di, mi, kc, A, B):
    """Root of kc (di + mi (x - xc)) = A + B x (one chine piece linear in x;
    B = 0 for a flat one): x = xc + (A + B xc - kc di) / (kc mi - B)."""
    return xc + (A + B * xc - kc * di) / (kc * mi - B)


# ------------------------------------------------------- parametric family
@dataclass
class PlaningHullForm:
    """The parametric planing-hull family (module docstring). FIELDS gives
    each parameter's unit, Froude-scaling exponent of length and meaning."""

    L: float
    b: float
    beta_deg: float
    depth: float
    taper: float = 0.35
    stem_rise: float = 0.0
    rise_from: float = None
    rise_power: float = 2.0
    transom_beam: float = 1.0
    transom_run: float = 0.0
    yc_floor: float = None
    n_stations: int = 48

    FIELDS = dict(
        L=(1, "hull length along the keel, transom to stem, m"),
        b=(1, "maximum chine beam, m"),
        beta_deg=(0, "deadrise, constant along the hull, deg"),
        depth=(1, "keel to sheer, constant along the keel, m"),
        taper=(0, "bow fraction of L over which the chine curves in to the "
               "stem, as a quarter ellipse"),
        stem_rise=(1, "keel rise at the stem above the transom keel, m "
                   "(forefoot and bow rake)"),
        rise_from=(0, "bow fraction of L over which the keel rises; None: "
                   "the taper's"),
        rise_power=(0, "keel rise ~ s^p over that fraction (p >= 2; 2: v0's "
                    "parabola)"),
        transom_beam=(0, "chine beam at the transom over the maximum"),
        transom_run=(0, "fraction of L from the transom over which the chine "
                     "widens linearly to its maximum (0: full beam at the "
                     "transom, v0)"),
        yc_floor=(1, "half-beam added to the chine everywhere, m (keeps the "
                  "stem section non-degenerate; None: v0's 1e-4 x L / "
                  "L_REF)"),
        n_stations=(0, "number of strips"))

    def __post_init__(self):
        if not (self.L > 0 and self.b > 0 and self.depth > 0):
            raise ValueError("PlaningHullForm: L, b, depth must be positive")
        if not 0.0 < self.beta_deg < 60.0:
            raise ValueError("PlaningHullForm: deadrise in (0, 60) deg")
        if not 0.0 <= self.taper <= 1.0:
            raise ValueError("PlaningHullForm: taper in [0, 1]")
        rf = self.taper if self.rise_from is None else self.rise_from
        if not 0.0 <= rf <= 1.0:
            raise ValueError("PlaningHullForm: rise_from in [0, 1]")
        if self.rise_power < 2.0:
            raise ValueError("PlaningHullForm: rise_power >= 2 (a keel "
                             "profile with bounded curvature)")
        if not 0.0 < self.transom_beam <= 1.0:
            raise ValueError("PlaningHullForm: transom_beam in (0, 1]")
        if not 0.0 <= self.transom_run <= 1.0 - self.taper:
            raise ValueError("PlaningHullForm: transom_run in [0, 1 - taper]"
                             " (it ends where the bow taper starts)")
        self.yc_floor = _floor(self.yc_floor, self.L)
        if self.yc_floor < 0.0:
            raise ValueError("PlaningHullForm: yc_floor >= 0")
        self.n_stations = int(self.n_stations)
        if self.n_stations < 4:
            raise ValueError("PlaningHullForm: at least 4 strips")

    # ----------------------------------------------------- the v0 member
    @classmethod
    def of_planing_hull(cls, h):
        """The family member of a hydro.planing.PlaningHull's own fields:
        v0's geometry (sim/planing_vessel.py:91-102)."""
        return cls(L=h.L, b=h.b, beta_deg=h.beta_deg, depth=h.depth,
                   taper=h.taper, stem_rise=h.stem_rise,
                   yc_floor=V0_YC_FLOOR, n_stations=h.n_stations)

    KIND = "family"

    def as_dict(self):
        return {k: v for k, v in asdict(self).items()}

    def max_beam(self):
        """The largest chine beam, m (without the floor yc_floor)."""
        return self.b

    def validity(self):
        """Messages for the parameters outside RANGES (empty: inside)."""
        return _validity(self.beta_deg, self.L, self.b)

    def v0_member(self):
        """True if the frozen v0 plant builds exactly this geometry from the
        hull fields L, b, beta_deg, depth, taper, stem_rise, n_stations."""
        return (not self._has_transom_piece() and self._keel_is_v0()
                and self.yc_floor == V0_YC_FLOOR)

    def froude(self, lam):
        """The geometrically similar member, lengths x lam."""
        d = self.as_dict()
        for k, (e, _) in self.FIELDS.items():
            if e and d[k] is not None:
                d[k] = d[k] * lam ** e
        return type(self)(**d)

    # ------------------------------------------------------------ pieces
    def _has_transom_piece(self):
        return self.transom_run > 0.0 and self.transom_beam != 1.0

    def _keel_is_v0(self):
        return ((self.rise_from is None or self.rise_from == self.taper)
                and self.rise_power == 2.0)

    def stations(self):
        L, b = self.L, self.b
        n = self.n_stations
        edges, xi, dx = _strips(L, n)
        x0 = (1.0 - self.taper) * L
        lt = max(L - x0, 1e-9)
        eps = self.yc_floor
        R = 0.5 * b
        transom = self._has_transom_piece()
        xr = self.transom_run * L
        RT = self.transom_beam * R

        def chine(x):
            # sim/planing_vessel.py:97-98, the floor a parameter
            s = np.clip((x - x0) / lt, 0.0, 1.0)
            y = 0.5 * b * np.sqrt(np.maximum(1.0 - s ** 2, 0.0)) + eps
            if transom:
                y = np.where(x < xr, RT + (R - RT) * (x / xr) + eps, y)
            return y

        def dchine(x):
            raw = (x - x0) / lt
            s = np.clip(raw, 0.0, 1.0)
            inside = (raw > 0.0) & (raw < 1.0)
            root = np.sqrt(np.maximum(1.0 - s ** 2, 1e-300))
            dy = np.where(inside, -0.5 * b * s / (root * lt), 0.0)
            if transom:
                dy = np.where(x < xr, (R - RT) / xr, dy)
            return dy

        def roots(xc, di, mi, kc):
            """Candidate chine-wetting points of strips with centres xc and
            keel depth di + mi (x - xc): the roots of kc d = yc of every
            piece (a root outside its own piece is a harmless extra
            breakpoint; sim/planing_anyspeed._substrip classifies each
            sub-piece at its midpoint). The flat and elliptic roots are the
            S1 expressions, unchanged."""
            out = []
            if transom:
                out.append(_linear_root(xc, di, mi, kc, RT + eps,
                                        (R - RT) / xr))
            # the flat piece: kc d = R + eps
            out.append(xc + ((R + eps) / kc - di) / mi)
            # the elliptic taper: (kc d - eps)^2 = R^2 (1 - s^2), s linear
            # in x -- an ellipse cut by a line; the squaring adds spurious
            # roots, harmless as extra breakpoints
            e0, e1 = kc * di - eps, kc * mi
            s0 = (xc - x0) / lt
            qa = e1 * e1 + (R / lt) ** 2
            qb = e0 * e1 + R * R * s0 / lt
            qc = e0 * e0 + R * R * (s0 * s0 - 1.0)
            disc = qb * qb - qa * qc
            sq = np.sqrt(np.where(disc >= 0.0, disc, np.nan))
            out.append(xc + (-qb - sq) / qa)
            out.append(xc + (-qb + sq) / qa)
            return out

        # strips whose chine varies: the taper (fwd end beyond x0), and the
        # transom run (aft end before xr)
        varying = edges[1:] > x0
        if transom:
            varying = varying | (edges[:-1] < xr)
        rise, kp, kpp = self._keel(xi, x0)

        def keel(x):
            return self._keel(np.asarray(x, float), x0)
        kpp_s = _kpp_strip(lambda x: keel(x)[1], edges, dx)
        st = Stations(n=n, L=L, edges=edges, xi=xi, dx=dx, yc=chine(xi),
                      rise=rise, kp=kp, kpp=kpp, kpp_strip=kpp_s,
                      beta_deg=self.beta_deg, depth=self.depth, eps=eps,
                      chine=chine, dchine=dchine, keel=keel, roots=roots,
                      varying=varying, concave=True, knots=np.zeros(0),
                      source=dict(kind="family", **self.as_dict()))
        return st.check()

    def _keel(self, xi, x0):
        L, sr_ = self.L, self.stem_rise
        if self._keel_is_v0():
            # sim/planing_vessel.py:97, 99-102, verbatim
            s = np.clip((xi - x0) / max(L - x0, 1e-9), 0.0, 1.0)
            rise = sr_ * s ** 2
            kp = 2.0 * sr_ * s / max(L - x0, 1e-9)
            kpp = np.where(s > 0, 2.0 * sr_ / max(L - x0, 1e-9) ** 2, 0.0)
            return rise, kp, kpp
        p = float(self.rise_power)
        xr0 = (1.0 - self.rise_from) * L
        lr = max(L - xr0, 1e-9)
        s = np.clip((xi - xr0) / lr, 0.0, 1.0)
        rise = sr_ * s ** p
        kp = p * sr_ * s ** (p - 1.0) / lr
        kpp = np.where(s > 0, p * (p - 1.0) * sr_ * s ** (p - 2.0) / lr ** 2,
                       0.0)
        return rise, kp, kpp

    def offsets(self, n_off):
        """This form sampled as an offsets table at n_off points (for the
        OffsetsForm convergence check)."""
        xo = np.linspace(0.0, self.L, n_off)
        st = self.stations()
        x0 = (1.0 - self.taper) * self.L
        keel = self._keel(xo, x0)[0]
        return OffsetsForm(x=xo, yc=st.chine(xo) - self.yc_floor, keel=keel,
                           beta_deg=self.beta_deg, depth=self.depth,
                           yc_floor=self.yc_floor,
                           n_stations=self.n_stations)


# ----------------------------------------------------------- offsets table
@dataclass
class OffsetsForm:
    """An offsets table: chine half-beam yc and keel height (above the
    transom keel, or any datum: the transom value is subtracted) at stations
    x from the transom (x[0] = 0, x[-1] = L). Chine piecewise linear, keel
    PCHIP. One deadrise (module docstring)."""

    x: list
    yc: list
    keel: list
    beta_deg: float
    depth: float
    yc_floor: float = None
    n_stations: int = 48

    def __post_init__(self):
        self.x = np.asarray(self.x, float)
        self.yc = np.asarray(self.yc, float)
        self.keel = np.asarray(self.keel, float)
        if not (self.x.ndim == 1 and self.x.size >= 3
                and self.yc.shape == self.x.shape
                and self.keel.shape == self.x.shape):
            raise ValueError("OffsetsForm: x, yc, keel 1-D of equal length "
                             ">= 3")
        if self.x[0] != 0.0 or np.any(np.diff(self.x) <= 0.0):
            raise ValueError("OffsetsForm: x increasing from 0 (the transom)")
        if self.x.size:
            self.yc_floor = _floor(self.yc_floor, float(self.x[-1]))
        if np.any(self.yc < 0.0) or self.yc_floor < 0.0:
            raise ValueError("OffsetsForm: half-beams >= 0")
        if not (np.all(np.isfinite(self.yc)) and np.all(np.isfinite(
                self.keel)) and np.max(self.yc) > 0.0):
            raise ValueError("OffsetsForm: finite offsets, some beam")
        # PlaningHullForm's parameter checks
        if not self.depth > 0:
            raise ValueError("OffsetsForm: depth must be positive")
        if not 0.0 < self.beta_deg < 60.0:
            raise ValueError("OffsetsForm: deadrise in (0, 60) deg")
        self.n_stations = int(self.n_stations)
        if self.n_stations < 4:
            raise ValueError("OffsetsForm: at least 4 strips")

    KIND = "offsets"

    @property
    def L(self):
        return float(self.x[-1])

    @property
    def stem_rise(self):
        """Keel height at the stem above the transom keel, m."""
        return float(self.keel[-1] - self.keel[0])

    def max_beam(self):
        """The largest chine beam, m (2 x the largest offset half-beam)."""
        return 2.0 * float(np.max(self.yc))

    def validity(self):
        return _validity(self.beta_deg, self.L, self.max_beam())

    def as_dict(self):
        return dict(x=self.x.tolist(), yc=self.yc.tolist(),
                    keel=self.keel.tolist(), beta_deg=self.beta_deg,
                    depth=self.depth, yc_floor=self.yc_floor,
                    n_stations=self.n_stations)

    def froude(self, lam):
        return OffsetsForm(x=self.x * lam, yc=self.yc * lam,
                           keel=self.keel * lam, beta_deg=self.beta_deg,
                           depth=self.depth * lam,
                           yc_floor=self.yc_floor * lam,
                           n_stations=self.n_stations)

    def v0_member(self):
        return False

    def stations(self):
        from scipy.interpolate import PchipInterpolator
        xo, yo, eps = self.x, self.yc, self.yc_floor
        L, n = self.L, self.n_stations
        edges, xi, dx = _strips(L, n)
        B = np.diff(yo) / np.diff(xo)
        A = yo[:-1] - B * xo[:-1] + eps

        def chine(x):
            return np.interp(x, xo, yo) + eps

        def dchine(x):
            j = np.clip(np.searchsorted(xo, x, side="right") - 1, 0,
                        B.size - 1)
            return B[j]

        def roots(xc, di, mi, kc):
            return [_linear_root(xc, di, mi, kc, a, bb) for a, bb in zip(A, B)]

        sp = PchipInterpolator(xo, self.keel - self.keel[0])

        def keel(x):
            x = np.asarray(x, float)
            return sp(x), sp(x, 1), sp(x, 2)
        # the curvature as the strip integral of kpp (module docstring):
        # PCHIP's own second derivative converges at first order only
        rise, kp = sp(xi), sp(xi, 1)
        kpp = _kpp_strip(lambda x: sp(x, 1), edges, dx)
        # strips whose chine is not constant somewhere along them
        varying = np.zeros(n, bool)
        for j in np.nonzero(B != 0.0)[0]:
            varying |= (edges[1:] > xo[j]) & (edges[:-1] < xo[j + 1])
        concave = bool(np.all(np.diff(B) <= 0.0))
        st = Stations(n=n, L=L, edges=edges, xi=xi, dx=dx, yc=chine(xi),
                      rise=rise, kp=kp, kpp=kpp, kpp_strip=kpp,
                      beta_deg=self.beta_deg, depth=self.depth, eps=eps,
                      chine=chine, dchine=dchine, keel=keel, roots=roots,
                      varying=varying, concave=concave,
                      knots=xo[1:-1].copy(),
                      source=dict(kind="offsets", **self.as_dict()))
        return st.check()


# ----------------------------------------------------------------- entry
FORMS = dict(family=PlaningHullForm, offsets=OffsetsForm)


def form_of(hull):
    """The geometry description of a planing hull: its own `form` (a dict
    with 'kind', hydro/vessel_spec.ParametricPlaningHull), else the family
    member of a plain PlaningHull's fields (v0's geometry). The hull's own
    fields L, b, beta_deg, depth must agree with the form."""
    spec = getattr(hull, "form", None)
    if spec is None:
        f = PlaningHullForm.of_planing_hull(hull)
    else:
        spec = dict(spec)
        kind = spec.pop("kind", "family")
        if kind not in FORMS:
            raise ValueError(f"hull {getattr(hull, 'name', hull)!r}: form "
                             f"kind {kind!r}, one of {sorted(FORMS)}")
        f = FORMS[kind](**spec)
    for k in ("L", "beta_deg", "depth", "n_stations"):
        if getattr(f, k) != getattr(hull, k):
            raise ValueError(f"hull {getattr(hull, 'name', hull)!r}: field "
                             f"{k} {getattr(hull, k)!r} disagrees with its "
                             f"form ({getattr(f, k)!r})")
    if isinstance(f, PlaningHullForm):
        # the v0 plant builds from these fields: they must be the form's,
        # or v0 and as1 would build two boats under one hull
        for k in ("b", "taper", "stem_rise"):
            if getattr(f, k) != getattr(hull, k):
                raise ValueError(f"hull {getattr(hull, 'name', hull)!r}: "
                                 f"field {k} {getattr(hull, k)!r} disagrees "
                                 f"with its form ({getattr(f, k)!r})")
    return f


def stations_of(hull):
    """The Stations a strip plant builds for `hull` (module docstring)."""
    return form_of(hull).stations()


def v0_geometry(hull):
    """True if the frozen v0 plant computes `hull`'s geometry exactly: a
    plain PlaningHull (no `form`: v0's formulas ARE its geometry, nothing is
    validated or built here, so every hull v0 built before still builds),
    or a form that is the family member of its fields."""
    if getattr(hull, "form", None) is None:
        return True
    return form_of(hull).v0_member()
