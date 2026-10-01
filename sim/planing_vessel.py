#!/usr/bin/env python3
"""
A planing hull in waves: 2D+t strip theory for heave and pitch, the waterjet
(sim/waterjet.py), and a slender-body model for sway, yaw and roll -- the
plant for boats that are carried by dynamic lift rather than buoyancy.

WHY A SECOND PLANT
sim/vessel.py rests on zero-speed BEM coefficients and on buoyancy about the
static waterline. A planing boat rises, trims by the bow and wets only part
of its bottom; above a length Froude number of ~1 the water it pushes down
carries most of its weight. Nothing in the displacement plant describes that.

THE VERTICAL PLANE (Zarnick 1978, the standard 2D+t model)
Watch one earth-fixed transverse plane as the hull passes through it: a V
section is being pushed into the water. Per unit length, with d the depth of
the keel below the local surface and V = dd/dt,

    V   = eta_t - (w + SP x q) + u (k'(x) + SP theta)
    c   = min(pi/2 d / tan(beta), y_chine(x))     wetted half-beam (Wagner)
    m_a = k_a pi/2 rho c^2                         2-D added mass
    f   = d/dt(m_a V) + 1/2 rho C_dc 2c V|V| + a_b rho g A(d)

d/dt(m_a V) = m_a dV/dt + (dm_a/dd) V^2, the second term only while the
section enters (on exit the flow separates). u (SP theta) is the planing
lift: the hull slides forward on a slope, so every plane it passes sees the
keel descend. m_a dV/dt contains the hull's own acceleration, so the added
mass sits on the left of the equations, recomputed every step. SP is
SIGN_PITCH (dz = SP x theta), as in the displacement plant: bow-up trim is
negative pitch. a_b, the buoyancy reduction at planing speed (Zarnick's 0.5),
blends to 1 at rest.

The rest
    surge   jet, the normal force tilted by the local bottom slope (pressure
            drag, Delta tan(tau) at equilibrium), ITTC-1957 friction on the
            wetted bottom, air drag, Froude-Krylov force of the wave slope
    sway,   slender-body lift of the wetted keel, Y = -U m22(transom) v_T
    yaw     (Munk), acting at the transom; crossflow drag along the hull;
            the jet's side force; Froude-Krylov force of the transverse slope
    roll    static GM about the wave slope, linear damping (placeholder)

STATE: the displacement plant's layout without radiation states,
[x, y, z, roll, pitch, yaw, u, v, w, p, q, r, thrust, nozzle]. z and pitch are
measured from the vessel floating at rest, so at planing speed z > 0 (it
rises) and pitch < 0 (bow up).
"""
from types import SimpleNamespace

import numpy as np
from scipy.optimize import fsolve

from .forces import point_load  # noqa: F401  (the displacement plant's)
from .vessel import SIGN_PITCH
from .waterjet import JetPump, Waterjet, Nozzle

G = 9.81
RHO_AIR = 1.225
NU = 1.19e-6
SP = SIGN_PITCH
_STEADY = {}
# Heun substeps (step): taken when the sway/yaw rate times dt exceeds
# STIFF_STEP -- just inside Heun's stability limit of 2, so that wherever
# the single step was stable nothing changes -- and sized so that the rate
# times the substep is at most STIFF_SUB
STIFF_STEP, STIFF_SUB, N_SUB_MAX = 1.8, 0.5, 64


def _load(fx, fy, fz, x, y, z):
    """[F, r x F] for a point load, written out: np.cross on 3-vectors
    was a quarter of this plant's run time."""
    return np.array([fx, fy, fz, y * fz - z * fy, z * fx - x * fz,
                     x * fy - y * fx])


class PlaningVessel:
    """Interface of sim/vessel.NonlinearVessel, for a PlaningHull."""

    kind = "planing"
    propulsor = "waterjet"

    def __init__(self, hull, sea, dt=0.01, prop=None, rudder=None,
                 captive_u=None, **_ignored):
        self._init_kw = dict(dt=dt)
        self.hull, self.sea, self.dt = hull, sea, dt
        h = hull
        self.L, self.B, self.rho = h.L, h.b, h.rho
        self.m = h.mass
        self.tanb = np.tan(np.radians(h.beta_deg))
        self.cosb = np.cos(np.radians(h.beta_deg))
        self.k_a, self.c_dc, self.cd_lat = h.k_a, h.c_dc, h.cd_lat
        # stations, from the transom forward; body x from the CG
        n = h.n_stations
        edges = np.linspace(0.0, h.L, n + 1)
        xi = 0.5 * (edges[:-1] + edges[1:])
        self.dx = h.L / n
        self.x = xi - h.lcg
        x0 = (1.0 - h.taper) * h.L
        s = np.clip((xi - x0) / max(h.L - x0, 1e-9), 0.0, 1.0)
        self.yc = 0.5 * h.b * np.sqrt(np.maximum(1.0 - s ** 2, 0.0)) + 1e-4
        self.zk = -h.vcg + h.stem_rise * s ** 2          # keel, body frame
        self.kp = 2.0 * h.stem_rise * s / max(h.L - x0, 1e-9)   # dz_k/dx
        self.kpp = np.where(s > 0, 2.0 * h.stem_rise
                            / max(h.L - x0, 1e-9) ** 2, 0.0)
        # rigid body
        m = self.m
        Ix = m * (h.k_roll * h.b) ** 2
        Iy = m * (h.k_pitch * h.L) ** 2
        Iz = m * (h.k_yaw * h.L) ** 2
        self.M = np.diag([m, m, m, Ix, Iy, Iz])
        self.Ix, self.Iy, self.Iz = Ix, Iy, Iz
        # at rest: where it floats, its draught and roll stability
        self.Z_rest, self.Th_rest = self._static()
        d0 = -(self.Z_rest + self.zk + SP * self.x * self.Th_rest)
        self.T = float(max(d0[0], 0.0))
        if h.T is None:
            h.T = self.T
        self.GM = self._gm(d0)
        # the waterjet at the transom, the intake on the keel ahead of it
        j = h.jet
        x_n = -h.lcg + j.get("x_nozzle", 0.0)
        z_n = -h.vcg + j.get("z_nozzle", 0.15)
        x_i = -h.lcg + j.get("x_intake", 0.8)
        z_i = -h.vcg + j.get("z_intake", 0.0)
        pump = JetPump(np.pi * j.get("d_nozzle", 0.085) ** 2 / 4,
                       j.get("p_max", 224e3), eta_pump=j.get("eta_pump", 0.775),
                       rho=self.rho)
        u_d = h.u_design
        t_max = pump.thrust(pump.p_max, u_d)
        self.prop = prop or Waterjet(
            pump, x_n, z_n, x_i, z_i, k_rev=j.get("k_rev", 0.6),
            tau_prime_loss=0.05, tau_prime_back=j.get("tau_prime_back", 0.5),
            t_max=t_max, t_min=-t_max / 3, tau=j.get("tau", 0.6),
            rate_max=t_max / 0.3, dead_band=t_max * 50 / 12000,
            delay=j.get("delay", 0.1), u_ref=u_d, dt=dt)
        self.rudder = rudder or Nozzle(
            max_deg=j.get("nozzle_max_deg", 25.0),
            rate_deg=j.get("nozzle_rate_deg", 40.0),
            tau=j.get("nozzle_tau", 0.1), delay=j.get("nozzle_delay", 0.1),
            dt=dt, x_nozzle=x_n, z_nozzle=z_n)
        self.thrusters = [(float(x_n), 0.0, float(z_n))]
        self.rudder_points = self.thrusters
        self.intakes = [(float(x_i), 0.0, float(z_i))]
        # what the harnesses read off a plant
        keel_rest = self.Z_rest + self.zk + SP * self.x * self.Th_rest
        self.sec = SimpleNamespace(
            x=self.x, dx=self.dx, n=n, x_bow=float(self.x[-1]),
            x_stern=float(self.x[0]), keel=keel_rest, y0=self.yc)
        self.slam = SimpleNamespace(
            wagner_slope=lambda d, kr=keel_rest: self._section(d - kr)[3])
        self.draft_bow = float(-keel_rest[-1])
        self.freeboard = float(h.depth + keel_rest[-1])
        self.v_slam = 0.093 * np.sqrt(G * h.L)
        self.captive_u = captive_u
        # sway/roll/yaw added mass and damping at the design attitude, for
        # identification and for the equations (placeholder grade)
        self.visc = np.zeros(6)
        self.Mtot = self.M.copy()
        self._design_lateral()
        self._reset()

    # ------------------------------------------------------------ geometry
    def _section(self, d):
        """(area, wetted half-beam, added mass, d m_a/dd, girth) per station
        at immersion d (keel depth below the local surface)."""
        tb, yc = self.tanb, self.yc
        dp = np.maximum(d, 0.0)
        dch = yc * tb
        below = np.minimum(dp, dch)
        A = below ** 2 / tb + 2.0 * yc * np.maximum(dp - dch, 0.0)
        cw = 0.5 * np.pi * dp / tb
        c = np.minimum(cw, yc)
        ma = self.k_a * 0.5 * np.pi * self.rho * c ** 2
        dma = np.where((cw < yc) & (dp > 0),
                       self.k_a * np.pi * self.rho * c * 0.5 * np.pi / tb, 0.0)
        girth = 2.0 * c / self.cosb + 2.0 * np.maximum(dp - dch, 0.0)
        return A, c, ma, dma, girth

    def _static(self):
        """Absolute CG height and pitch of the vessel floating at rest."""
        rg = self.rho * G

        def res(v):
            Z, Th = v
            d = -(Z + self.zk + SP * self.x * Th)
            A = self._section(d)[0]
            return [(rg * A).sum() * self.dx / (self.m * G) - 1.0,
                    (SP * self.x * rg * A).sum() * self.dx
                    / (self.m * G * self.L)]
        T0 = self.m / (self.rho * self.L * self.hull.b * 0.4)
        sol = fsolve(res, [self.hull.vcg - T0, 0.0], full_output=True)
        return float(sol[0][0]), float(sol[0][1])

    def _gm(self, d0):
        """Static metacentric height from the V sections at rest."""
        A, c, _, _, _ = self._section(d0)
        half = np.minimum(np.maximum(d0, 0) / self.tanb, self.yc)
        vol = A.sum() * self.dx
        It = (2.0 / 3.0 * half ** 3).sum() * self.dx
        # centroid height of each wetted section above its keel: 2/3 d for
        # a V section below the chine
        kb = ((2.0 / 3.0) * np.maximum(d0, 0) * A).sum() * self.dx / max(vol,
                                                                        1e-9)
        return float(kb + It / max(vol, 1e-9) - self.hull.vcg)

    # --------------------------------------------------------------- waves
    def _wave_arrays(self):
        """Per-component coefficients, computed once per sea: the surface,
        its rate and acceleration and both slopes come from one phase
        evaluation as five matrix products."""
        s = self.sea
        a = np.atleast_1d(np.asarray(s.a, float))
        if a.size == 0 or not np.any(a):
            self._wa = None
            return
        w, k, th = (np.atleast_1d(np.asarray(v, float))
                    for v in (s.w, s.k, s.th))
        self._wkx, self._wky = k * np.cos(th), k * np.sin(th)
        self._ww, self._wphi = w, np.atleast_1d(np.asarray(s.phi, float))
        self._wa = np.stack([a, a * w, -a * w ** 2,
                             -a * self._wkx, -a * self._wky])   # (5, M)

    def _waves(self, X, Y, t):
        if not hasattr(self, "_wa"):
            self._wave_arrays()
        if self._wa is None:
            z = np.zeros_like(X)
            return z, z, z, z, z
        ph = (np.outer(X, self._wkx) + np.outer(Y, self._wky)
              - self._ww * t + self._wphi)
        c, sn = np.cos(ph), np.sin(ph)
        A = self._wa
        return (c @ A[0], sn @ A[1], c @ A[2], sn @ A[3], sn @ A[4])

    def eta_points(self, X, Y, t):
        return self._waves(np.atleast_1d(X), np.atleast_1d(Y), t)[0]

    # ------------------------------------------------------------ dynamics
    def _ctr(self, u):
        if not getattr(self.hull, "transom_correction", True):
            return 1.0
        cv = abs(u) / np.sqrt(G * self.hull.b)
        a = 0.34 * self.hull.b * cv
        if a < 1e-6:
            return 1.0
        return np.tanh(2.5 * (self.x - self.x[0] + 0.5 * self.dx) / a)

    def _a_b(self, u):
        cv = abs(u) / np.sqrt(G * self.hull.b)
        return 1.0 - (1.0 - self.hull.a_b) * float(np.clip((cv - 0.5), 0, 1))

    def deriv(self, s, t):
        eta, nu = s[:6], s[6:12]
        thr, rud = s[12], s[13]
        x0, y0, z, phi, th, psi = eta
        u, v, w, p, q, r = nu
        x = self.x
        cps, sps = np.cos(psi), np.sin(psi)
        X, Y = x0 + x * cps, y0 + x * sps
        e, et, ett, ex, ey = self._waves(X, Y, t)
        s_lon = ex * cps + ey * sps
        s_tr = -ex * sps + ey * cps
        Th = self.Th_rest + th
        d = e - (self.Z_rest + z + self.zk + SP * x * Th)
        tau = self.kp + SP * Th                     # local bottom slope
        V = et - (w + SP * x * q) + u * tau
        A, c, ma, dma, girth = self._section(d)
        wet = d > 0.0
        dx, rho = self.dx, self.rho
        ab = self._a_b(u)
        # Transom correction (Garme 2005): the pressure falls to the
        # atmosphere at the transom, which strip theory does not know, so
        # every sectional load is tapered to zero there over a length that
        # grows with speed, a = 0.34 b C_V. Without it the lift sat too far
        # aft: Fridsma's model A (V/sqrt(L) 4) ran low and flat and then
        # porpoised by itself in calm water.
        ctr = self._ctr(u)
        ma, dma = ma * ctr, dma * ctr
        mom = np.where(V > 0.0, dma * V * V, 0.0)
        cf = 0.5 * rho * self.c_dc * 2.0 * c * V * np.abs(V) * ctr
        buoy = ab * rho * G * A * ctr
        rest = ma * (ett + 2.0 * SP * u * q - u * u * self.kpp) + mom + cf \
            + buoy

        # ---- everything that is not the strip force, as generalised forces
        tau_o = np.zeros(6)
        lw = wet.sum() * dx
        S = (girth * wet).sum() * dx
        uu = max(abs(u), 0.05)
        re = uu * max(lw, 0.05) / NU
        cfr = 0.075 / (np.log10(re) - 2.0) ** 2 + 0.0004
        Df = 0.5 * rho * u * abs(u) * cfr * S
        if S > 0:
            xw = (x * girth * wet).sum() * dx / S
            zw = float(np.mean(self.zk[wet])) if wet.any() else self.zk[0]
        else:
            xw, zw = 0.0, self.zk[0]
        tau_o += _load(-Df, 0.0, 0.0, xw, 0.0, zw)
        Da = 0.5 * RHO_AIR * self.hull.cda * u * abs(u)
        tau_o += _load(-Da, 0.0, 0.0, 0.0, 0.0, 0.5)
        # Froude-Krylov force of the wave slopes, surge and sway
        Xw = -(rho * G * ab * A * s_lon).sum() * dx
        Ysec = -rho * G * ab * A * s_tr
        tau_o[0] += Xw
        tau_o[1] += Ysec.sum() * dx
        tau_o[5] += (x * Ysec).sum() * dx
        # sway / yaw: slender-body (Munk) forces of the wetted hull, with
        # m22 = pi/2 rho d^2 per unit length (a plate of draught d and its
        # free-surface image). Per unit length f = U d(m22 v_l)/dx, which
        # integrates to a side force at the transom, Y = -U m22_T v_T, and a
        # moment N = x_T Y - U int m22 v_l dx. The integral is the Munk
        # moment; without it all the hull's side force sat at the transom,
        # on top of the jet, and cancelled the jet's turning moment -- the
        # boat turned at 2 deg/s at 34 kn with the nozzle hard over.
        m22 = 0.5 * np.pi * rho * np.maximum(d, 0.0) ** 2
        vl = v + x * r
        Ylift = -abs(u) * m22[0] * vl[0]
        tau_o[1] += Ylift
        tau_o[5] += x[0] * Ylift - abs(u) * (m22 * vl).sum() * dx
        Ycf = -0.5 * rho * self.cd_lat * np.maximum(d, 0.0) * vl * np.abs(vl)
        tau_o[1] += Ycf.sum() * dx
        tau_o[5] += (x * Ycf).sum() * dx
        # roll: static stability about the transverse wave slope
        wbar = (A * s_tr).sum() / max(A.sum(), 1e-12)
        wn_r = np.sqrt(max(self.m * G * self.GM, 1e-6) / self.Ix)
        tau_o[3] += (-self.m * G * self.GM * (phi - wbar)
                     - 2.0 * self.hull.roll_zeta * wn_r * self.Ix * p)
        # the jet, or in captive runs a force exactly holding the speed
        if self.captive_u is None:
            fx, fy = self.prop.forces(thr, rud, u)
            self._jet_f = (fx, fy)
        else:
            fx, fy = 0.0, 0.0
        xn, _, zn = self.thrusters[0]
        # rigid-body Coriolis of the yaw-rotating frame
        tau_o[0] += self.m * v * r
        tau_o[1] -= self.m * u * r

        # ---- heave and pitch, instantaneous added mass on the left
        A33 = ma.sum() * dx
        A35 = (ma * SP * x).sum() * dx
        A55 = (ma * x * x).sum() * dx
        Fz = rest.sum() * dx - self.m * G
        My = (SP * x * rest).sum() * dx
        if self.captive_u is not None:
            # the force that holds the speed: everything else, at the nozzle
            pres = -(rest * tau).sum() * dx          # first guess, no accel
            fx = Df + Da - Xw - pres
        jet = _load(fx, fy, 0.0, xn, 0.0, zn)
        tau_o += jet
        m11, m12, m22_ = self.m + A33, A35, self.Iy + A55
        r1, r2 = Fz + tau_o[2], My + tau_o[4]
        det = m11 * m22_ - m12 * m12
        wd = (r1 * m22_ - m12 * r2) / det
        qd = (m11 * r2 - m12 * r1) / det
        f = rest - ma * (wd + SP * x * qd)           # full strip force
        Xp = -(f * tau).sum() * dx                   # pressure drag
        # ---- surge, sway, roll, yaw
        ud = 0.0 if self.captive_u is not None else \
            (tau_o[0] + Xp) / self.Mtot[0, 0]
        vd = tau_o[1] / self.Mtot[1, 1]
        pd = tau_o[3] / self.Mtot[3, 3]
        rd = tau_o[5] / self.Mtot[5, 5]
        ds = np.zeros_like(s)
        ds[0] = u * cps - v * sps
        ds[1] = u * sps + v * cps
        ds[2], ds[3], ds[4], ds[5] = w, p, q, r
        ds[6:12] = ud, vd, wd, pd, qd, rd
        aux = dict(d=d, V=V, mom=mom, wd=wd, qd=qd, lw=lw, S=S, Df=Df,
                   Xp=Xp, Da=Da, fx=fx)
        return ds, aux

    def _design_lateral(self):
        """Sway/yaw/roll added mass and the quadratic sway coefficient, at
        the running attitude at the design speed (placeholder grade)."""
        z, th = self.running_attitude(self.hull.u_design)
        d = -(self.Z_rest + z + self.zk + SP * self.x * (self.Th_rest + th))
        dp = np.maximum(d, 0.0)
        m22 = 0.5 * np.pi * self.rho * dp ** 2
        self.Mtot = self.M.copy()
        self.Mtot[0, 0] += 0.05 * self.m
        self.Mtot[1, 1] += m22.sum() * self.dx
        self.Mtot[3, 3] += 0.2 * self.Ix
        self.Mtot[5, 5] += (m22 * self.x ** 2).sum() * self.dx
        self.visc[1] = 0.5 * self.rho * self.cd_lat * dp.sum() * self.dx

    # --------------------------------------------------------- steady state
    def running_attitude(self, u):
        """(z, pitch) at steady speed u in calm water, from rest by a
        captive run (surge held, a force at the nozzle holding it). Cached
        per hull and speed."""
        key = (self.hull.name, id(self.hull), round(float(u), 3),
               getattr(self.hull, "transom_correction", True),
               self.k_a, self.hull.a_b, self.c_dc)
        hit = _STEADY.get(key)
        if hit is not None:
            return hit
        if u <= 0.0:
            _STEADY[key] = (0.0, 0.0)
            return 0.0, 0.0
        from sim.test_vessel import Monochromatic
        v = PlaningVessel.__new__(PlaningVessel)
        v.__dict__.update(self.__dict__)
        v.sea = Monochromatic(1.0, 0.0)
        v._wave_arrays()                 # the calm copy's own (none)
        v.captive_u = float(u)
        s = np.zeros(14)
        s[6] = u
        dt = 0.005
        last = None
        for i in range(4000):
            k1, _ = v.deriv(s, 0.0)
            s2 = s + dt * k1
            k2, _ = v.deriv(s2, 0.0)
            s = s + 0.5 * dt * (k1 + k2)
            s[6] = u
            if i % 200 == 199:
                cur = (s[2], s[4])
                if last is not None and abs(cur[0] - last[0]) < 1e-5 \
                        and abs(cur[1] - last[1]) < 1e-6:
                    break
                last = cur
        _STEADY[key] = (float(s[2]), float(s[4]))
        return _STEADY[key]

    def calm_resistance(self, u):
        """Total calm-water resistance at steady speed u (N), with the
        breakdown, at the running attitude."""
        from sim.test_vessel import Monochromatic
        z, th = self.running_attitude(u)
        v = PlaningVessel.__new__(PlaningVessel)
        v.__dict__.update(self.__dict__)
        v.sea = Monochromatic(1.0, 0.0)
        v._wave_arrays()                 # the calm copy's own (none)
        v.captive_u = float(u)
        s = np.zeros(14)
        s[6], s[2], s[4] = u, z, th
        _, aux = v.deriv(s, 0.0)
        return dict(R=aux["fx"], pressure=-aux["Xp"], friction=aux["Df"],
                    air=aux["Da"], trim_deg=float(-np.degrees(
                        self.Th_rest + th)), rise=z, wetted_length=aux["lw"])

    # ------------------------------------------------------------- stepping
    def _reset(self):
        self.last_bow_acc = 0.0
        self.last_cg_acc = 0.0
        self.last_slam_force = 0.0
        self.peak_slam_force = 0.0
        self.last_rel_bow = 0.0
        self.last_jet_force = (0.0, 0.0)
        self.slam_count = 0
        self.last_n_sub = 1
        self._emerged = False
        self._jet_f = (0.0, 0.0)
        if hasattr(self, "prop"):
            self.prop.prime = 1.0

    def with_sea(self, sea, **overrides):
        import copy
        kw = dict(self._init_kw)
        kw.update(overrides)
        kw.pop("wind", None)
        return PlaningVessel(self.hull, sea, **kw)

    def unpack(self, s):
        return s[:6], s[6:12], s[12:12], s[12], s[13]

    def initial_state(self, u0=0.0):
        """At the running attitude for u0, with the thrust that holds u0 in
        calm water -- a planing boat started level at speed would first fall
        off the plane."""
        self._reset()
        s = np.zeros(14)
        s[6] = u0
        if u0 > 0:
            s[2], s[4] = self.running_attitude(u0)
            need = self.calm_resistance(u0)["R"]
            lo, hi = 0.0, 3.0 * self.prop.t_max
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if self.prop.forces(mid, 0.0, u0)[0] < need:
                    lo = mid
                else:
                    hi = mid
            s[12] = min(0.5 * (lo + hi), self.prop.t_max)
        return s

    def prop_submergence(self, eta, t):
        """Water over the jet intake (negative once it is out)."""
        xi, yi, zi = self.intakes[0]
        c, s_ = np.cos(eta[5]), np.sin(eta[5])
        X, Y = eta[0] + xi * c - yi * s_, eta[1] + xi * s_ + yi * c
        surf = self.eta_points(X, Y, t)[0]
        hull_z = (self.Z_rest + eta[2] + zi + SP * xi * (self.Th_rest
                                                         + eta[4])
                  + yi * eta[3])
        return float(surf - hull_z)

    def _lateral_rate(self, s, d):
        """Fastest rate (1/s) of the sway/yaw equations as deriv writes
        them: the eigenvalue of largest modulus of d(v', r')/d(v, r), from
        the transom lift, the Munk moment, the crossflow drag and the
        Coriolis term, at immersion d."""
        u, v, r = s[6], s[7], s[11]
        x, dx, rho = self.x, self.dx, self.rho
        dp = np.maximum(d, 0.0)
        m22 = 0.5 * np.pi * rho * dp ** 2
        au = abs(u)
        k = rho * self.cd_lat * dp * np.abs(v + x * r)   # dYcf/dv_l per m
        x0, mT = x[0], au * m22[0]
        Yv = -mT - k.sum() * dx
        Yr = -mT * x0 - (k * x).sum() * dx - self.m * u
        Nv = -mT * x0 - au * m22.sum() * dx - (k * x).sum() * dx
        Nr = (-mT * x0 * x0 - au * (m22 * x).sum() * dx
              - (k * x * x).sum() * dx)
        a, b = Yv / self.Mtot[1, 1], Yr / self.Mtot[1, 1]
        c, e = Nv / self.Mtot[5, 5], Nr / self.Mtot[5, 5]
        half, det = 0.5 * (a + e), a * e - b * c
        disc = half * half - det
        if disc >= 0.0:
            return float(abs(half) + np.sqrt(disc))
        return float(np.sqrt(det))

    def step(self, s, t, thrust_cmd, rudder_cmd, dt=None):
        dt = self.dt if dt is None else dt
        self.prop.update_prime(self.prop_submergence(s[:6], t), dt)
        k1, aux = self.deriv(s, t)
        # Substeps when the hull is buried. The sway/yaw forces grow with
        # the instantaneous immersion (m22 ~ d^2, crossflow ~ d) but the
        # sway/yaw inertia is the design attitude's (Mtot). When the boat
        # lands bow first after a jump (keel ~2 m under the surface against
        # 0.25 m running) the lateral rate reaches ~170 1/s; Heun is stable
        # only below 2/dt (100 1/s at 0.02 s), and each step then tripled
        # the error -- the yaw-rate blow-up at 26 kn (DEFECTS K). Below the
        # threshold nothing changes: wherever the single step was stable
        # the result is bit-identical.
        n = 1
        lam = self._lateral_rate(s, aux["d"]) * dt
        if lam > STIFF_STEP:
            n = min(int(np.ceil(lam / STIFF_SUB)), N_SUB_MAX)
        if n == 1:
            s2 = s + dt * k1
            k2, _ = self.deriv(s2, t + dt)
            s_new = s + 0.5 * dt * (k1 + k2)
        else:
            h = dt / n
            s_new, k = s, k1
            for i in range(n):
                if i:
                    k, _ = self.deriv(s_new, t + i * h)
                k2, _ = self.deriv(s_new + h * k, t + (i + 1) * h)
                s_new = s_new + 0.5 * h * (k + k2)
        self.last_n_sub = n
        s_new[12] = self.prop.advance(s[12], thrust_cmd, dt)
        s_new[13] = self.rudder.step(s[13], rudder_cmd, dt)
        # measurements, at the start of the step
        self.last_jet_force = self._jet_f
        self.last_cg_acc = float(aux["wd"])
        self.last_bow_acc = float(aux["wd"] + SP * self.x[-1] * aux["qd"])
        slam = float(aux["mom"].sum() * self.dx)
        self.last_slam_force = slam
        self.peak_slam_force = max(self.peak_slam_force, slam)
        d_b, V_b = float(aux["d"][-1]), float(aux["V"][-1])
        # immersion of the bow station measured the way the displacement
        # plant measures it (surface minus hull, from the rest waterline)
        self.last_rel_bow = d_b + float(self.sec.keel[-1])
        bow_out = d_b <= 0.0
        if self._emerged and not bow_out and V_b > self.v_slam:
            self.slam_count += 1
        self._emerged = bow_out
        return s_new
