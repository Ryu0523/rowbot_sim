#!/usr/bin/env python3
"""
The planing plant (sim/planing_vessel.py) with a large-angle righting arm,
so that it can capsize -- the high-fidelity world of the final closed-loop
evaluation (learn/meta/ROADMAP_2026-09-30.md, sections 9 and 10).

WHY A SUBCLASS
The parent's roll restoring is linear, -m g GM (phi - wbar), wbar the
transverse wave slope under the hull. A linear moment never vanishes, so the
boat could not capsize however far it heeled. Here the moment is

    K = -W GZ(phi - wbar)

with GZ the hull's own static righting arm. The damping is the parent's,
2 zeta wn Ix p (a placeholder), unchanged. The parent file is under the code
hashes of running jobs and is not edited: in the parent's equations roll
enters nothing but its own row (phi and p appear only in the roll moment),
so the subclass takes the parent's derivative and swaps the restoring term
in that one row -- exact, with no copy of the strip theory.

GZ(phi), ONCE AT CONSTRUCTION
Every station is the parent's own section closed into a polygon: keel, V
bottom to the chine (half-beam yc, height yc tan(beta)), vertical side to the
deck edge at the hull depth above the keel, flat watertight deck. Heeled by
phi about the body x axis at the rest trim (free trim not solved), the
waterplane is raised or lowered by one sinkage delta until the displaced
volume is the rest volume, and

    GZ = -y_B,      the horizontal offset of the centre of buoyancy from G.

The area and the lateral moment of each section's immersed part are boundary
integrals over the immersed part of its edges, A = closed-int Y dZ and
int Y dA = closed-int Y^2/2 dZ; the waterline chord has dZ = 0 and adds
nothing, so no polygon clipping is needed and the table is one vectorised
bisection over all angles. The same sweep gives the angle of vanishing
stability (GZ back through zero) and the deck-edge immersion angle (the low
deck edge of any station reaching the water).

The slope at zero is the geometric GM. The parent's GM takes the centroid of
every wetted section at 2/3 of its immersion, exact for a V below the chine;
at rest the aft stations are a little above it, so the two differ by a few
millimetres. To leave the plant unchanged near upright the curve is shifted
by (GM - GM_geom) sin(phi) -- the same as moving G by that much -- and the
slope then equals the parent's GM exactly (report() prints both).

CAPSIZE
Once |phi| passes the angle of vanishing stability after a step, the plant
records a capsize (capsized, capsize_t) and from then on returns the state of
that moment unchanged: nothing integrates past it, so nothing blows up. The
caller stops on `plant.capsized`; learn/repro/task.Mission does not know the
flag, so a loop reads it after every control step.

RECORDING (record=True)
Every step logs the state at its start with the measurements the parent takes
from that state -- CG and bow vertical acceleration, bow immersion, substep
count -- plus the water over the jet intake (computed by the parent at every
step but not kept) and whether the step ended finite or capsized.
trajectory() returns it as arrays at the plant step, the finest resolution
this plant has, for studies/safety_events.py.

WHAT IS NOT MODELLED
Roll stays one-way: it does not feed back into heave, pitch, sway or yaw, and
nothing but the transverse wave slope drives it -- no heel from turning, no
lift asymmetry of a heeled planing bottom, no chine walking or dynamic loss
of roll stability at speed. GZ is the static curve at rest, used at every
speed, as the parent used the static GM.
"""
import numpy as np

from .planing_vessel import PlaningVessel, SP, G, N_SUB_MAX

# heel angles of the table: 0 to 180 deg every 0.25 deg; GZ is odd in phi
PHI_STEP = np.radians(0.25)
PHI_TAB = np.arange(0.0, np.pi + 0.5 * PHI_STEP, PHI_STEP)
FAIL_MAX = np.radians(40.0)          # MSC.1/Circ.1627 3.2.1
_GZ = {}
_TOP = {}


def _immersed(Y, Z, H):
    """(area, int Y dA) of the parts of closed polygons below Z = H.

    Y, Z (..., k): vertices, counter-clockwise in (Y, Z); H broadcasts to
    (...). Edge P0 -> P1 at P0 + t (P1 - P0): its immersed part is t <= tc
    on a rising edge and t >= tc on a falling one, tc where it meets H."""
    Y1, Z1 = np.roll(Y, -1, axis=-1), np.roll(Z, -1, axis=-1)
    dY, dZ = Y1 - Y, Z1 - Z
    h = np.asarray(H)[..., None]
    flat = dZ == 0.0
    tc = np.clip((h - Z) / np.where(flat, 1.0, dZ), 0.0, 1.0)
    up = dZ > 0.0
    t0 = np.where(up, 0.0, tc)
    t1 = np.where(up, tc, 1.0)
    d1, d2, d3 = t1 - t0, t1 ** 2 - t0 ** 2, t1 ** 3 - t0 ** 3
    A = (dZ * (Y * d1 + 0.5 * dY * d2)).sum(-1)
    M = (0.5 * dZ * (Y * Y * d1 + Y * dY * d2 + dY * dY * d3 / 3.0)).sum(-1)
    return A, M


def _first_root(x, f):
    """First x > x[0] where f falls through zero (linear between grid
    points); None if it never does."""
    idx = np.nonzero((f[:-1] > 0.0) & (f[1:] <= 0.0))[0]
    if idx.size == 0:
        return None
    i = idx[0]
    return float(x[i] + (x[i + 1] - x[i]) * f[i] / (f[i] - f[i + 1]))


class PlaningVesselGZ(PlaningVessel):
    """PlaningVessel whose roll restoring is -W GZ(phi - wbar) and which
    stops at a capsize. Same constructor, plus `record`."""

    def __init__(self, hull, sea, dt=0.01, prop=None, rudder=None,
                 captive_u=None, record=False, **kw):
        super().__init__(hull, sea, dt=dt, prop=prop, rudder=rudder,
                         captive_u=captive_u, **kw)
        self._init_kw = dict(dt=dt, record=record)
        self.record = bool(record)
        self._righting_arm()
        self._w_last = None

    # ------------------------------------------------------------ the curve
    def _sections(self):
        """Station polygons in body (y, z) about the CG, counter-clockwise:
        keel, starboard chine, starboard deck edge, port deck edge, port
        chine; and the rest waterline height at each station."""
        h = self.hull
        yc, zk = self.yc, self.zk
        zc = zk + np.minimum(yc * self.tanb, h.depth)
        zd = zk + h.depth
        py = np.stack([np.zeros_like(yc), yc, yc, -yc, -yc], -1)
        pz = np.stack([zk, zc, zd, zd, zc], -1)
        zw = -(self.Z_rest + SP * self.x * self.Th_rest)
        return py, pz, zw

    def gz_geometric(self, phi):
        """(GZ, sinkage, lowest deck-edge height above the water) at heel
        angles phi >= 0 (rad), hydrostatic, constant displacement, rest
        trim. Positive phi lifts the +y side."""
        phi = np.atleast_1d(np.asarray(phi, float))
        py, pz, zw = self._sections()
        dx = self.dx
        v0 = _immersed(py, pz, zw)[0].sum() * dx
        c, s = np.cos(phi)[:, None, None], np.sin(phi)[:, None, None]
        Y, Z = py * c - pz * s, py * s + pz * c          # (P, n, 5)
        lo = np.full(phi.size, -3.0)
        hi = np.full(phi.size, 3.0)
        for _ in range(50):                  # 6 m / 2^50: far below a nm
            mid = 0.5 * (lo + hi)
            v = _immersed(Y, Z, zw + mid[:, None])[0].sum(-1) * dx
            over = v > v0
            hi, lo = np.where(over, mid, hi), np.where(over, lo, mid)
        delta = 0.5 * (lo + hi)
        H = zw + delta[:, None]
        A, M = _immersed(Y, Z, H)
        gz = -M.sum(-1) / A.sum(-1)
        # the low deck edge is the port one (vertex 3) for phi > 0
        margin = (Z[..., 3] - H).min(-1)
        return gz, delta, margin, v0

    def _righting_arm(self):
        h = self.hull
        key = (h.name, id(h), h.L, h.b, h.beta_deg, h.mass, h.lcg, h.vcg,
               h.depth, h.taper, h.stem_rise, h.n_stations, h.rho)
        hit = _GZ.get(key)
        if hit is None:
            gz, delta, margin, v0 = self.gz_geometric(PHI_TAB)
            eps = 1e-4
            gm_geom = float(self.gz_geometric([eps])[0][0] / eps)
            # match the parent's GM at zero heel: a shift of G by dgm
            dgm = self.GM - gm_geom
            gz = gz + dgm * np.sin(PHI_TAB)
            gz[0] = 0.0
            phi_v = _first_root(PHI_TAB[1:], gz[1:])
            phi_d = _first_root(PHI_TAB, margin)
            i_max = int(np.argmax(gz))
            hit = dict(gz=gz, delta=delta, margin=margin, v0=v0,
                       gm_geom=gm_geom, dgm=dgm,
                       phi_vanish=np.pi if phi_v is None else phi_v,
                       phi_deck=np.pi if phi_d is None else phi_d,
                       gz_max=float(gz[i_max]),
                       phi_gz_max=float(PHI_TAB[i_max]))
            _GZ[key] = hit
        self.gz_tab = hit["gz"]
        # GZ / phi, interpolated: its value at zero is the parent's GM, so
        # the slope near upright is GM exactly (a table of GZ itself would
        # give the secant over the first 0.25 deg, 0.02% low)
        self.r_tab = np.concatenate([[self.GM], self.gz_tab[1:]
                                     / PHI_TAB[1:]])
        self.gm_geom, self.dgm = hit["gm_geom"], hit["dgm"]
        self.phi_vanish, self.phi_deck = hit["phi_vanish"], hit["phi_deck"]
        # MSC.1/Circ.1627 3.2.1: the deck edge stands in for the openings
        # (an open cockpit floods once its gunwale is under)
        self.phi_fail = float(min(FAIL_MAX, self.phi_vanish, self.phi_deck))
        self._gz_info = hit

    def gz(self, phi):
        """Righting arm (m) at heel phi (rad), any angle: odd, 2 pi
        periodic; GZ / phi linear in the 0.25 deg table."""
        a = (phi + np.pi) % (2.0 * np.pi) - np.pi
        return float(a * np.interp(abs(a), PHI_TAB, self.r_tab))

    def report(self):
        """The curve's numbers, for printing."""
        i = self._gz_info
        wn = np.sqrt(self.m * G * self.GM / self.Mtot[3, 3])
        return dict(GM=float(self.GM), GM_geom=i["gm_geom"], dGM=i["dgm"],
                    v0=i["v0"], v_plant=float(self.m / self.rho),
                    phi_vanish_deg=float(np.degrees(self.phi_vanish)),
                    phi_deck_deg=float(np.degrees(self.phi_deck)),
                    phi_fail_deg=float(np.degrees(self.phi_fail)),
                    gz_max=i["gz_max"],
                    phi_gz_max_deg=float(np.degrees(i["phi_gz_max"])),
                    T_roll=float(2.0 * np.pi / wn))

    # ------------------------------------------------------------ dynamics
    def _waves(self, X, Y, t):
        out = super()._waves(X, Y, t)
        self._w_last = (np.size(X), out)
        return out

    def deriv(self, s, t):
        self._w_last = None
        ds, aux = super().deriv(s, t)
        if self._w_last is None or self._w_last[0] != self.x.size:
            raise RuntimeError("PlaningVesselGZ: the parent's derivative no "
                               "longer evaluates the waves at the stations "
                               "once; the roll swap needs updating")
        _, _, _, ex, ey = self._w_last[1]
        psi = s[5]
        s_tr = -ex * np.sin(psi) + ey * np.cos(psi)
        A = self._section(aux["d"])[0]
        # the parent's wbar, exactly as it computes it
        wbar = (A * s_tr).sum() / max(A.sum(), 1e-12)
        rel = s[3] - wbar
        ds[9] += self.m * G * (self.GM * rel - self.gz(rel)) \
            / self.Mtot[3, 3]
        return ds, aux

    def prop_submergence(self, eta, t):
        sub = super().prop_submergence(eta, t)
        self.last_submergence = sub
        return sub

    def _reset(self):
        super()._reset()
        self.capsized = False
        self.capsize_t = None
        self._capsize_state = None
        self.last_submergence = 0.0
        self._log = []

    def step(self, s, t, thrust_cmd, rudder_cmd, dt=None):
        dt = self.dt if dt is None else dt
        if self.capsized:
            # the run is over: hold the state of the capsize
            self.last_cg_acc = self.last_bow_acc = 0.0
            self.last_n_sub = 0
            return self._capsize_state.copy()
        s_new = super().step(s, t, thrust_cmd, rudder_cmd, dt)
        ok = bool(np.all(np.isfinite(s_new)))
        if ok and abs(s_new[3]) >= self.phi_vanish:
            self.capsized = True
            self.capsize_t = float(t + dt)
            self._capsize_state = s_new.copy()
        if self.record:
            self._log.append((float(t), np.array(s, float),
                              self.last_cg_acc, self.last_bow_acc,
                              self.last_rel_bow, self.last_n_sub,
                              self.last_submergence, float(thrust_cmd),
                              float(rudder_cmd), ok, self.capsized))
        return s_new

    def with_sea(self, sea, **overrides):
        kw = dict(self._init_kw)
        kw.update(overrides)
        kw.pop("wind", None)
        return type(self)(self.hull, sea, **kw)

    # ------------------------------------------------------------- outputs
    def top_speed(self):
        """Calm-water speed at full thrust, m/s: where the jet's net thrust
        at t_max equals the calm resistance at the running attitude.
        Cached per hull."""
        key = (self.hull.name, id(self.hull), float(self.prop.t_max))
        if key in _TOP:
            return _TOP[key]
        from scipy.optimize import brentq
        prime, self.prop.prime = self.prop.prime, 1.0
        t_max = self.prop.t_max

        def excess(u):
            return (self.prop.forces(t_max, 0.0, u)[0]
                    - self.calm_resistance(u)["R"])
        try:
            lo = float(self.hull.u_design)
            hi = 1.5 * lo
            while excess(hi) > 0.0 and hi < 6.0 * lo:
                lo, hi = hi, 1.3 * hi
            u = float(brentq(excess, lo, hi, xtol=0.01))
        finally:
            self.prop.prime = prime
        _TOP[key] = u
        return u

    def trajectory(self, u_max=None):
        """The recorded run as arrays at the plant step, with the constants
        studies/safety_events.py needs. Row k: the state at t[k] and what
        the plant measured from it; ok / capsized describe the end of that
        step."""
        if not self.record:
            raise RuntimeError("built with record=False")
        rows = self._log
        n = len(rows)
        cols = list(zip(*rows)) if n else [[]] * 11
        sea = self.sea
        meta = dict(
            dt=float(self.dt), L=float(self.L), mass=float(self.m),
            freeboard=float(self.freeboard), draft_bow=float(self.draft_bow),
            depth=float(self.hull.depth), bow_half_beam=float(self.yc[-1]),
            bow_keel_z=float(self.zk[-1]), th_rest=float(self.Th_rest),
            rud_max=float(self.rudder.max),
            x_nozzle=float(self.thrusters[0][0]), n_sub_max=int(N_SUB_MAX),
            GM=float(self.GM),
            phi_vanish=float(self.phi_vanish), phi_deck=float(self.phi_deck),
            phi_fail=float(self.phi_fail), capsized=bool(self.capsized),
            capsize_t=self.capsize_t,
            capsize_state=(None if self._capsize_state is None
                           else self._capsize_state.copy()),
            u_max=float(self.top_speed() if u_max is None else u_max),
            sea=dict(hs=getattr(sea, "hs", None), tp=getattr(sea, "tp", None),
                     theta0=getattr(sea, "theta0", None)))
        f = np.asarray
        return dict(
            t=f(cols[0], float), s=(np.stack(cols[1]) if n
                                    else np.zeros((0, 14))),
            a_cg=f(cols[2], float), a_bow=f(cols[3], float),
            rel_bow=f(cols[4], float), n_sub=f(cols[5], int),
            submergence=f(cols[6], float), thrust_cmd=f(cols[7], float),
            rudder_cmd=f(cols[8], float), ok=f(cols[9], bool),
            capsized=f(cols[10], bool), meta=meta)


# ------------------------------------------------------------- factories
def plant_for(db, sea, hull=None, dt=None, **kw):
    """sim.config.plant_for(db, sea, hull, dt, fidelity="high") for a
    planing hull, with the large-angle roll: the same path, through the
    hull's own time step (Hull.scales) when dt is None."""
    from sim import config
    h = config.resolve(db, hull)
    if getattr(h, "kind", None) != "planing":
        raise ValueError("the GZ plant is the planing plant's; "
                         f"{getattr(h, 'name', h)!r} is not a planing hull")
    return PlaningVesselGZ(h, sea, dt=h.scales()["dt"] if dt is None else dt,
                           **kw)


def from_plant(plant, **kw):
    """The GZ twin of a built PlaningVessel: same hull, sea, time step and
    options (the plant a learn/repro/task.Mission or sim.env.Episode built
    for fidelity="high"). The caller takes a fresh initial_state from it."""
    opts = dict(plant._init_kw)
    opts.update(kw)
    opts.pop("wind", None)
    if plant.captive_u is not None:
        opts.setdefault("captive_u", plant.captive_u)
    return PlaningVesselGZ(plant.hull, plant.sea, **opts)
