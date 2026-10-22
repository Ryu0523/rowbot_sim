#!/usr/bin/env python3
"""
The v0 planing plant (sim/planing_vessel.py) with switches that reproduce
Zarnick's (1978) published 2D+t formulation term by term, and (step 2 of
learn/meta/SIM_REDESIGN_2026-10-08.md) a switch replacing Wagner's sectional
entry force by the modified Logvinovich model (MLM).

v0 is not edited: this is a subclass whose `deriv` is v0's with switch
branches. With every switch at its default (`ZOPT_V0`) the arithmetic is
the same as v0's to the last bit (checked by studies/exp_zarnick_mlm.py,
check Z0).

Sources for Zarnick's formulation (DTNSRDC 78/032 itself is not openly
retrievable; two independent restatements were read):
  * T. M. Sayeed, MEng thesis, Memorial University 2010, ch. 3 (eqs. 3.6-3.8,
    3.9-3.18, sec. 3.8: k_a 1.0, a_BF 0.5, a_BM = 0.5 a_BF, C_pu pi/2,
    C_D = 1.0 cos(beta); wave: only the vertical orbital velocity, eq. 3.4;
    thrust and friction with their own lever arms, eq. 3.2).
  * Jang, Yum & Kim, J. Ocean Eng. Technol. 29(3) 2015, sec. 2 (same
    coefficients: buoyancy correction 0.5, "the buoyancy moment arm is
    additionally multiplied by a correction factor 0.5"; for the towed
    model the thrust (tow) and friction act approximately at the CG and
    balance, so they drop out of the heave and pitch equations, eq. 11).

Switches (ZOPT, a dict; key: v0 value -> Zarnick value):
  bm      1.0 -> 0.5    factor on the BUOYANCY part of the pitch moment
                         (Zarnick: moment arm x 0.5, i.e. a_BM = 0.5 a_BF)
  tow     'nozzle' -> 'cg'   where the captive (towing) force acts
  fric    'keel' -> 'cg'     where the friction drag acts
  normal  False -> True  immersion measured along the hull normal
                         (d/cos(theta)), normal relative velocity, normal
                         force resolved to vertical (Zarnick eqs. 3.4, 3.16)
  mom     'entry' -> 'both'  momentum term dm_a/dd V^2 also on exit
                         (literal reading of D/Dt(m_a V); NOT asserted to
                         be Zarnick's -- tested because no source settles it)
  cf      'abs' -> 'sq'  crossflow drag C rho b V^2 (literal eq. 3.7) vs
                         V|V| (tested for the same reason)
  keel    None -> ('ellipse', rise_m)  Fridsma's (1969) elliptic keel
                         profile over the bow taper: z = R (1 - sqrt(1-s^2))
  mlm     False -> True  (step 2) modified Logvinovich sectional entry force
                         for chine-dry sections, see `_mlm_constants`
  chine   'frozen' -> 'extend'  (step 2b, a BOUND, not a model) chine-wetted
                         sections: Zarnick freezes the wetted half-beam at the
                         chine (dm_a/dd = 0); 'extend' treats the section as
                         an unbounded wedge (half-beam, m_a and the momentum
                         term keep growing; the crossflow drag keeps the real
                         half-beam) -- the upper bound of the post-wetting
                         force, registered in SIM_REDESIGN 进展 2 (d)

Opt-in hook (HULL_FORCE_AUDIT_REVIEW check V3, 2026-10-09): `ext_load`, an
instance attribute defaulting to None. When set to a 6-vector [X, Y, Z, K,
M, N] (body frame, about the CG) or a callable (s, t) -> 6-vector, it is
added to the generalised forces `tau_o` in `deriv`, so a captive settling
run (`running_attitude`) finds the equilibrium with that load. With
ext_load None nothing in deriv or in the steady-state cache key changes
(checked bit-identical, studies/derive_v3_3d_correction.py 'hook'). It is
not carried by `with_sea`.
"""
import math

import numpy as np

from .planing_vessel import (PlaningVessel, _load, _STEADY, G, RHO_AIR, NU,
                             SP)

ZOPT_V0 = dict(bm=1.0, tow="nozzle", fric="keel", normal=False,
               mom="entry", cf="abs", keel=None, mlm=False, chine="frozen")
ZOPT_ZARNICK = dict(ZOPT_V0, bm=0.5, tow="cg", fric="cg", normal=True)
# the configuration that reproduces Zarnick's published Fridsma column
# (studies/exp_zarnick_mlm.py step 1): towing force and friction through the
# CG (Zarnick's and Fridsma's set-up), buoyancy moment as v0 (a_BM = a_BF)
ZOPT_ZR = dict(ZOPT_V0, tow="cg", fric="cg")


def _mlm_constants(beta_deg):
    """Strip form of the modified Logvinovich model for a wedge, chine dry.

    hydro/rankine/planing_wedge2d.py (W1): the MLM pressure on a wedge with
    Wagner's half-width c = pi h / (2 tan b), integrated from the keel to the
    jet root (first zero of p), is (rho = 1)
        F = K_v(b) c V^2 + K_m(b) c^2 Vdot
    when the jet-root angle th is the constant-velocity (similarity) value
    -- the approximation made here (registered: the exact root moves with
    Vdot h / V^2; its effect is measured on the wedge drop, check W2).
    Wagner's strip force is the same with K_v = pi^2/(2 tan b), K_m = pi/2.
    Returns (K_v, K_m, th)."""
    from hydro.rankine.planing_wedge2d import _cutoff_theta, _force_terms
    T = math.tan(math.radians(beta_deg))
    b = math.atan(T)
    sb2, cb2 = math.sin(b) ** 2, math.cos(b) ** 2
    c, V = 1.0, 1.0
    h = 2.0 * T * c / math.pi
    cd = math.pi * V / (2.0 * T)
    th = _cutoff_theta(c, cd, V, 0.0, h, T, math.pi / 2 - 1e-15)
    Kv = _force_terms(c, cd, V, 0.0, h, T, sb2, cb2, th)
    Km = _force_terms(c, cd, 0.0, 1.0, h, T, sb2, cb2, th)
    return Kv, Km, th


class ZarnickVessel(PlaningVessel):
    """PlaningVessel with the switches above (`zopt`)."""

    def __init__(self, hull, sea, dt=0.01, zopt=None, **kw):
        self.zopt = dict(ZOPT_V0)
        self.zopt.update(zopt or {})
        self.ext_load = None        # opt-in external load, see module doc
        z = self.zopt
        if z["mlm"]:
            self._Kv, self._Km, self._mlm_th = _mlm_constants(hull.beta_deg)
        super().__init__(hull, sea, dt=dt, **kw)
        self._init_kw = dict(dt=dt, zopt=dict(self.zopt))
        if z["keel"] is not None or z["mlm"] or z["chine"] != "frozen":
            if z["keel"] is not None:
                kind, R = z["keel"]
                assert kind == "ellipse"
                h = hull
                n = h.n_stations
                edges = np.linspace(0.0, h.L, n + 1)
                xi = 0.5 * (edges[:-1] + edges[1:])
                x0 = (1.0 - h.taper) * h.L
                Lb = max(h.L - x0, 1e-9)
                s = np.clip((xi - x0) / Lb, 0.0, 1.0)
                r = np.sqrt(np.maximum(1.0 - s ** 2, 1e-12))
                self.zk = -h.vcg + R * (1.0 - r)
                self.kp = R * s / r / Lb
                self.kpp = np.where(s > 0, R / r ** 3 / Lb ** 2, 0.0)
            self._rebuild_static()

    def _rebuild_static(self):
        """v0's construction steps that depend on the keel or the section
        law, repeated after they changed (v0's own code, called again)."""
        from types import SimpleNamespace
        self.Z_rest, self.Th_rest = self._static()
        d0 = -(self.Z_rest + self.zk + SP * self.x * self.Th_rest)
        self.T = float(max(d0[0], 0.0))
        self.GM = self._gm(d0)
        keel_rest = self.Z_rest + self.zk + SP * self.x * self.Th_rest
        self.sec = SimpleNamespace(
            x=self.x, dx=self.dx, n=self.hull.n_stations,
            x_bow=float(self.x[-1]), x_stern=float(self.x[0]),
            keel=keel_rest, y0=self.yc)
        self.draft_bow = float(-keel_rest[-1])
        self.freeboard = float(self.hull.depth + keel_rest[-1])
        self.Mtot = self.M.copy()
        self._design_lateral()

    # ------------------------------------------------------------ geometry
    def _section(self, d):
        if self.zopt["chine"] == "extend":
            return self._section_extend(d)
        if not self.zopt["mlm"]:
            return PlaningVessel._section(self, d)
        # MLM: same wetted half-beam law (Wagner's condition, W2), MLM
        # added mass K_m rho c^2 and momentum coefficient K_v rho c (so the
        # momentum force dma V^2 = K_v rho c V^2); chine wetted: Zarnick's
        # rule (half-beam frozen at the chine, dma = 0) applied to the MLM
        # added mass -- continuous at chine wetting. NOT the Tassin-Korobkin
        # (W3) continuation, which needs a per-plane state (registered).
        tb, yc = self.tanb, self.yc
        dp = np.maximum(d, 0.0)
        dch = yc * tb
        below = np.minimum(dp, dch)
        A = below ** 2 / tb + 2.0 * yc * np.maximum(dp - dch, 0.0)
        cw = 0.5 * np.pi * dp / tb
        c = np.minimum(cw, yc)
        ma = self._Km * self.rho * c ** 2
        dma = np.where((cw < yc) & (dp > 0), self._Kv * self.rho * c, 0.0)
        girth = 2.0 * c / self.cosb + 2.0 * np.maximum(dp - dch, 0.0)
        return A, c, ma, dma, girth

    def _section_extend(self, d):
        """Bound (d): no chine. The wetted half-beam follows Wagner's law
        past the chine (cw = pi d / (2 tan b), not capped), the added mass
        and the momentum coefficient with it (Wagner's or MLM's constants);
        area and girth are the real section's, and the returned half-beam
        (used only by the crossflow drag) is capped at the chine."""
        tb, yc = self.tanb, self.yc
        dp = np.maximum(d, 0.0)
        dch = yc * tb
        below = np.minimum(dp, dch)
        A = below ** 2 / tb + 2.0 * yc * np.maximum(dp - dch, 0.0)
        cw = 0.5 * np.pi * dp / tb
        c = np.minimum(cw, yc)
        if self.zopt["mlm"]:
            ma = self._Km * self.rho * cw ** 2
            dma = np.where(dp > 0, self._Kv * self.rho * cw, 0.0)
        else:
            ma = self.k_a * 0.5 * np.pi * self.rho * cw ** 2
            dma = np.where(dp > 0, self.k_a * np.pi * self.rho * cw * 0.5
                           * np.pi / tb, 0.0)
        girth = 2.0 * c / self.cosb + 2.0 * np.maximum(dp - dch, 0.0)
        return A, c, ma, dma, girth

    # ------------------------------------------------------------ dynamics
    def deriv(self, s, t):
        zo = self.zopt
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
        dx, rho = self.dx, self.rho
        if zo["normal"]:
            # Zarnick: immersion along the hull normal, normal velocity,
            # normal force; vertical part = f_n cos(theta)
            cth = math.cos(Th)
            A, _, _, _, girth = self._section(d)
            _, c, ma, dma, _ = self._section(d / cth)
            Vn = V * cth
            Vq = np.abs(Vn) if zo["cf"] == "abs" else Vn
            cf = 0.5 * rho * self.c_dc * 2.0 * c * Vn * Vq * cth
            # m_a-dot Vn = dma (dd_n/dt) Vn = dma (V/cos)(V cos) = dma V^2
            # (Zarnick eq. 3.17); vertical part x cos
            if zo["mom"] == "entry":
                mom = np.where(V > 0.0, dma * V * V, 0.0) * cth
            else:
                mom = dma * V * V * cth
            ma = ma * cth * cth              # vertical part of m_a d(Vn)/dt
        else:
            A, c, ma, dma, girth = self._section(d)
        wet = d > 0.0
        ab = self._a_b(u)
        ctr = self._ctr(u)
        ma, dma = ma * ctr, dma * ctr
        if zo["normal"]:
            mom = mom * ctr
            cf = cf * ctr
        else:
            if zo["mom"] == "entry":
                mom = np.where(V > 0.0, dma * V * V, 0.0)
            else:
                mom = dma * V * V
            if zo["cf"] == "abs":
                cf = 0.5 * rho * self.c_dc * 2.0 * c * V * np.abs(V) * ctr
            else:
                cf = 0.5 * rho * self.c_dc * 2.0 * c * V * V * ctr
        buoy = ab * rho * G * A * ctr
        rest = ma * (ett + 2.0 * SP * u * q - u * u * self.kpp) + mom + cf \
            + buoy

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
        if zo["fric"] == "keel":
            tau_o += _load(-Df, 0.0, 0.0, xw, 0.0, zw)
        else:
            tau_o += _load(-Df, 0.0, 0.0, 0.0, 0.0, 0.0)
        Da = 0.5 * RHO_AIR * self.hull.cda * u * abs(u)
        tau_o += _load(-Da, 0.0, 0.0, 0.0, 0.0, 0.5)
        Xw = -(rho * G * ab * A * s_lon).sum() * dx
        Ysec = -rho * G * ab * A * s_tr
        tau_o[0] += Xw
        tau_o[1] += Ysec.sum() * dx
        tau_o[5] += (x * Ysec).sum() * dx
        m22 = 0.5 * np.pi * rho * np.maximum(d, 0.0) ** 2
        vl = v + x * r
        Ylift = -abs(u) * m22[0] * vl[0]
        tau_o[1] += Ylift
        tau_o[5] += x[0] * Ylift - abs(u) * (m22 * vl).sum() * dx
        Ycf = -0.5 * rho * self.cd_lat * np.maximum(d, 0.0) * vl * np.abs(vl)
        tau_o[1] += Ycf.sum() * dx
        tau_o[5] += (x * Ycf).sum() * dx
        wbar = (A * s_tr).sum() / max(A.sum(), 1e-12)
        wn_r = np.sqrt(max(self.m * G * self.GM, 1e-6) / self.Ix)
        tau_o[3] += (-self.m * G * self.GM * (phi - wbar)
                     - 2.0 * self.hull.roll_zeta * wn_r * self.Ix * p)
        if self.captive_u is None:
            fx, fy = self.prop.forces(thr, rud, u)
            self._jet_f = (fx, fy)
        else:
            fx, fy = 0.0, 0.0
        xn, _, zn = self.thrusters[0]
        tau_o[0] += self.m * v * r
        tau_o[1] -= self.m * u * r

        A33 = ma.sum() * dx
        A35 = (ma * SP * x).sum() * dx
        A55 = (ma * x * x).sum() * dx
        Fz = rest.sum() * dx - self.m * G
        My = (SP * x * rest).sum() * dx
        if zo["bm"] != 1.0:
            # Zarnick: the buoyancy moment arm x a_BM/a_BF (0.5)
            My -= (1.0 - zo["bm"]) * (SP * x * buoy).sum() * dx
        if self.captive_u is not None:
            pres = -(rest * tau).sum() * dx
            fx = Df + Da - Xw - pres
        if self.captive_u is not None and zo["tow"] == "cg":
            jet = _load(fx, fy, 0.0, 0.0, 0.0, 0.0)
        else:
            jet = _load(fx, fy, 0.0, xn, 0.0, zn)
        tau_o += jet
        if self.ext_load is not None:
            tau_o = tau_o + (self.ext_load(s, t) if callable(self.ext_load)
                             else self.ext_load)
        m11, m12, m22_ = self.m + A33, A35, self.Iy + A55
        r1, r2 = Fz + tau_o[2], My + tau_o[4]
        det = m11 * m22_ - m12 * m12
        wd = (r1 * m22_ - m12 * r2) / det
        qd = (m11 * r2 - m12 * r1) / det
        f = rest - ma * (wd + SP * x * qd)
        Xp = -(f * tau).sum() * dx
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

    # --------------------------------------------------------- steady state
    def _key(self, u):
        key = ("Z", self.hull.name, id(self.hull), round(float(u), 3),
               getattr(self.hull, "transom_correction", True),
               self.k_a, self.hull.a_b, self.c_dc,
               tuple(sorted((k, str(v)) for k, v in self.zopt.items())))
        if self.ext_load is not None:       # unchanged key when unset
            ext = self.ext_load
            key += (("ext", id(ext)) if callable(ext) else
                    ("ext",) + tuple(round(float(v), 9) for v in ext))
        return key

    def running_attitude(self, u):
        """v0's captive settling run, with this class's deriv."""
        key = self._key(u)
        hit = _STEADY.get(key)
        if hit is not None:
            return hit
        if u <= 0.0:
            _STEADY[key] = (0.0, 0.0)
            return 0.0, 0.0
        from sim.test_vessel import Monochromatic
        v = type(self).__new__(type(self))
        v.__dict__.update(self.__dict__)
        v.sea = Monochromatic(1.0, 0.0)
        v._wave_arrays()
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
        from sim.test_vessel import Monochromatic
        z, th = self.running_attitude(u)
        v = type(self).__new__(type(self))
        v.__dict__.update(self.__dict__)
        v.sea = Monochromatic(1.0, 0.0)
        v._wave_arrays()
        v.captive_u = float(u)
        s = np.zeros(14)
        s[6], s[2], s[4] = u, z, th
        _, aux = v.deriv(s, 0.0)
        return dict(R=aux["fx"], pressure=-aux["Xp"], friction=aux["Df"],
                    air=aux["Da"], trim_deg=float(-np.degrees(
                        self.Th_rest + th)), rise=z, wetted_length=aux["lw"])

    def with_sea(self, sea, **overrides):
        kw = dict(self._init_kw)
        kw.update(overrides)
        kw.pop("wind", None)
        return ZarnickVessel(self.hull, sea, **kw)
