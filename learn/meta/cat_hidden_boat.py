#!/usr/bin/env python3
"""
Catalogue items H1-H3 (hidden degrees of freedom) and B1-B3 (the boat's
own changes) of the D10 force-catalogue prior (learn/meta/PRIOR_D10_DRAFT.md
3.2 and 3.4, with sections 10 and 11; framework learn/meta/cat_base.py).

H1  hidden roll: a hidden oscillator phi, phidot driven by a heel target
    (load offset, turning, lateral wave slope, jet side force, impeller
    reaction torque with a random fixed sign, other roll moments) that
    feeds odd forces into sway / yaw and even ones into heave / surge /
    pitch; speed-dependent stiffness; w.p. 0.15 / 0.7 two stable heels
    (negative stiffness + cubic), w.p. 0.2 / 0.7 chine walking (negative
    damping above a speed, saturating in phi^2).
H2  fuel sloshing: 1-2 tanks, one spring-mass per tank and horizontal
    direction (Dodge's rectangular-tank first mode), wall at |s| = a / 4.
H3  bilge / green water: a water mass sliding along the hull between two
    end stops, grown by green-water events while the bow is under the
    surface (D9.6 bow height < 0) and by a leak, drained exponentially.
B1  loading: one hidden loading vector (eps_m, dx_g, dy_g) that changes
    the inertia, drag, running attitude, heave / pitch frequencies and the
    heel bias together, MERGED with the D9 inertia draw (draft open
    question 10): the layer-3 draw is the calibration-load boat, B1 scales
    and shifts it and the result is the shared inertia M_t of every lever
    arm (adjust_ctx).
B2  discrete load events: CG jumps (Poisson, heavy-tailed), one point-mass
    step, a towed drag switching on and off.
B3  fouling: a constant extra friction drag.

Numerics (draft section 6 item 6): the oscillators (H1, H2) use the exact
2 x 2 transition with the step's frozen stiffness and damping (H1: the
cubic linearised at the step start, Newton form, and the substep split into
up to 8 inner steps when |k| dt or c dt is large; see H1); first-
order states (H3 drainage, noise) exact exponentials; quadratic drags (H3
friction, B2 tow, B3) linearised implicit; stiffness-type changes (B1)
velocity kicks.

Cross-item interfaces (all optional; absent = no coupling):
  ctx.load        set by B1.adjust_ctx: dict(eps_m, dxg, dyg, dxg_L, dyg_B)
                  (read by H1 for the heel bias, H2 for the fuel level, W2
                  for its dolphin margin m_0, draft 3.4)
  q['K_roll']     roll moment (N m, positive raises the port side) that
                  stage-'env' items (A1 wind) put in q before H1 runs
  q['H1']         set by H1: dict(state, I_x); roll_kick(q, K, dt) lets a
                  later item (H2 lateral slosh, B2 lateral CG jump) apply a
                  roll moment to H1's state (felt from the next substep)
  q['phi'], q['phid'], q['phidd']   H1's roll at the substep start and its
                  mean roll acceleration over the substep
"""
import types

import numpy as np

from learn.meta import cat_base as CB
from learn.meta import operators_gen as G

GRAV, DEG, KN = CB.GRAV, CB.DEG, CB.KN
B_REF = 2.03            # beam (m) behind the draft's roll-frequency range
PHI_MAX = 0.5           # rad: hard stop of the hidden roll (guard)
PHI_LOAD_CAP = 10 * DEG  # soft cap of B1's heel bias through dy_g / GM


def logu(rng, lo, hi, size=None):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


def roll_kick(q, K, dt):
    """Apply a roll moment K (N m, (B,) or scalar, positive raises the
    port side) to H1's hidden roll over dt, if H1 ran this substep."""
    h = q.get("H1")
    if h is None:
        return False
    h["state"]["phid"] = h["state"]["phid"] + np.asarray(K, float) * dt \
        / h["I_x"]
    return True


def _state_rng(rng):
    return rng if rng is not None else np.random.default_rng(0)


def _point_acc(ctx, q, x, z):
    """Body-x and earth-vertical accelerations (B,) of hull point (x, 0, z)
    from the low-fidelity boat's own accelerations nu0 (the injected error
    is not known yet at this point of the substep)."""
    nu, sr = q["nu0"], q["sr"]
    thbdd = ctx.sp * nu[:, 4]                    # bow-up pitch acceleration
    ax = nu[:, 0] - sr[:, 8] * sr[:, 9] - z * thbdd
    az = nu[:, 3] + ctx.sp * x * nu[:, 4]
    return ax, az


# ============================================================== H1 roll
# D10.8 (orchestrator decision on D10.7 item 4): the moment form turns a
# heeling moment into the static heel phi_eq / (1 - kappa (U / U_max)^2),
# so kappa is also a heel amplification; its range (a scale) is narrowed
# from U[0, 0.8] to U[0, 0.5]: at most 2x over the task speed range
# (was up to 5x at U_max). The two-stable-heel draws keep their own kappa.
H1_KAP_MAX = 0.5
H1_AMP_MAX = 2.0


@CB.register
class H1(CB.CatItem):
    """phi'' = -k_lin phi + w0^2 phi_eq + c_K K_roll / I_x + noise - c phi',
    - c3 phi^3, k_lin = w0^2 (1 - kappa (U / U_max)^2) (the force is
    -k_lin phi), c = 2 zeta_eff w_d, w_d = w0 max(sqrt|k_lin| / w0, 0.3),
    zeta_eff = zeta - gamma sigma((U - U_cw) / w_U) (1 - phi^2 / phi_sat^2),
    phi_eq = phi_b + c_turn u r / g + c_w t_slope + c_d F_jet,y / W
             + phi_Q T / t_max.
    phi > 0 raises the port side (the framework's hull height + y phi).
    Forces at phi: Y = -W (a1 phi + a3 phi|phi|) (a1 = 1: the whole lift
    tilted), N = W (l1 phi + l3 phi|phi|) + N_p phi', F_z = -W c2 phi^2 at
    x_tf, X = -R c_x phi^2 (R the low-fidelity resistance), sway / yaw
    damping x (1 + e_phi phi^2)."""

    code, stage = "H1", "hidden"
    gated = True
    P_PERSIST, P_CW = 0.15 / 0.7, 0.2 / 0.7

    def draw_on(self, rng, ctx):
        u = rng.uniform
        L = ctx.L
        w0 = float(logu(rng, 1.5, 5.5)) * np.sqrt(B_REF / ctx.B)
        kx = float(u(0.35, 0.45)) * ctx.B
        persist = bool(rng.random() < self.P_PERSIST)
        if persist:
            Uc = float(u(ctx.u_lo, ctx.u_max))
            kap = (ctx.u_max / Uc) ** 2
            phi_l = float(u(2.0, 8.0)) * DEG
            c3 = w0 ** 2 * max(kap - 1.0, 0.25) / phi_l ** 2
        else:
            Uc, phi_l, c3 = 0.0, 0.0, 0.0
            kap = float(u(0.0, H1_KAP_MAX))
        zeta = float(logu(rng, 0.05, 0.4))
        GM = w0 ** 2 * kx ** 2 / GRAV
        phi_b0 = float(rng.normal(0.0, 2.0 * DEG))
        load = getattr(ctx, "load", None)
        phi_load = 0.0
        if load is not None:          # weight to port lowers the port side
            phi_load = -PHI_LOAD_CAP * float(np.tanh(
                load["dyg"] / GM / PHI_LOAD_CAP))
        turn = float(logu(rng, 0.05, 0.5))
        turn = -turn if rng.random() < 0.8 else turn    # inward: port down
        prm = dict(
            w0=w0, kx=kx, I_x=ctx.m * kx ** 2, GM=GM, persist=persist,
            U_c=Uc, kap=kap, phi_l=phi_l, c3=c3, zeta=zeta,
            phi_b=phi_b0 + phi_load, phi_load=phi_load, turn=turn,
            c_w=float(u(0.0, 1.0)), c_d=float(u(0.0, 0.5)),
            c_K=float(u(0.5, 1.5)),
            phi_Q=float(u(0.0, 3.0)) * DEG * (1.0 if rng.random() < 0.5
                                               else -1.0))
        a1 = float(u(-0.3, 1.0))
        l1 = float(u(-0.15, 0.15)) * L
        prm.update(a1=a1, a3=float(u(-1, 1)) * a1, l1=l1,
                   l3=float(u(-1, 1)) * l1, n_p=float(u(-0.05, 0.05)),
                   c2=float(u(0.0, 1.0)), cx=float(u(0.0, 2.0)),
                   x_tf=float(u(-0.15, 0.15)) * L, e_phi=float(u(0.0, 20.0)))
        cw = bool(rng.random() < self.P_CW)
        if cw:
            prm.update(cw=True, gam=float(u(0.0, 0.3)),
                       U_cw=float(u(25.0, 50.0)) * KN,
                       w_U=float(u(1.0, 5.0)) * KN,
                       phi_sat=float(logu(rng, 2.0, 10.0)) * DEG)
        else:
            prm.update(cw=False, gam=0.0, U_cw=0.0, w_U=1.0, phi_sat=1.0)
        prm.update(phi_n=float(logu(rng, 0.05, 0.5)) * DEG,
                   tau_n=float(logu(rng, 0.3, 3.0)))
        return prm

    def init_state(self, params, B=1, rng=None):
        rng = _state_rng(rng)
        return dict(phi=np.zeros(B), phid=np.zeros(B),
                    n=rng.normal(0.0, 1.0, B), rng=rng, init=False)

    @staticmethod
    def k_lin(prm, U, ctx):
        return prm["w0"] ** 2 * (1.0 - prm["kap"] * (U / ctx.u_max) ** 2)

    @staticmethod
    def forcing(prm, q, ctx, noise):
        sr = q["sr"]
        thr = q["thr"]
        Fy0 = ctx.k_js * np.maximum(thr, 0.0) * np.sin(
            np.clip(q["noz"], -ctx.rud_stall, ctx.rud_stall))
        phi_eq = (prm["phi_b"] + prm["turn"] * q["u_r"] * sr[:, 8] / GRAV
                  + prm["c_w"] * q["t_slope"] + prm["c_d"] * Fy0 / ctx.W
                  + prm["phi_Q"] * thr / ctx.t_max)
        K = np.asarray(q.get("K_roll", 0.0), float)
        return prm["w0"] ** 2 * phi_eq + prm["c_K"] * K / prm["I_x"] \
            + prm["w0"] ** 2 * prm["phi_n"] * noise

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        sr = q["sr"]
        B = q["B"]
        U = q["U_r"]
        kl = self.k_lin(prm, U, ctx)
        F = self.forcing(prm, q, ctx, state["n"])
        c3 = prm["c3"]
        if not state["init"]:
            # start at the static heel of the start state (no transient):
            # stable rows at F / k, bistable rows on a random side
            k0 = kl
            stab = k0 > 1e-6 * prm["w0"] ** 2
            side = np.where(state["rng"].random(B) < 0.5, -1.0, 1.0)
            bi = np.sqrt(np.maximum(-k0, 0.0) / max(c3, 1e-12)) \
                if c3 > 0 else np.zeros(B)
            state["phi"] = np.clip(np.where(stab, F / np.where(stab, k0, 1.0),
                                            side * bi), -PHI_MAX, PHI_MAX)
            state["phid"] = np.zeros(B)
            state["init"] = True
        phi0, phid0 = state["phi"].copy(), state["phid"].copy()
        wd = prm["w0"] * np.maximum(np.sqrt(np.abs(kl) / prm["w0"] ** 2),
                                    0.3)
        sg = 0.0
        if prm["cw"]:
            sg = 1.0 / (1.0 + np.exp(-(U - prm["U_cw"]) / prm["w_U"]))

        def lin(phi):
            # the cubic linearised at phi (tangent stiffness, Newton form),
            # the chine-walk damping at phi: frozen over the (inner) step
            kt = kl + 3.0 * c3 * phi ** 2
            Ft = F + 2.0 * c3 * phi ** 3
            zeff = prm["zeta"] - prm["gam"] * sg * (
                1.0 - phi ** 2 / prm["phi_sat"] ** 2)
            return kt, Ft, 2.0 * zeff * wd

        kt, _, c = lin(phi0)
        rate = float(np.max(np.maximum(np.sqrt(np.abs(kt)), 0.5 * np.abs(c))))
        n_in = int(np.clip(np.ceil(rate * dt / 0.3), 1, 8))
        h = dt / n_in
        phi1, phid1 = phi0, phid0
        for _ in range(n_in):
            kt, Ft, c = lin(phi1)
            big = np.abs(kt) > 1e-6 * prm["w0"] ** 2
            xeq = np.where(big, Ft / np.where(big, kt, 1.0), 0.0)
            phi1, phid1 = CB.osc_step(phi1, phid1, kt, c, h, x_eq=xeq,
                                      acc=np.where(big, 0.0, Ft))
            out = np.abs(phi1) > PHI_MAX
            phi1 = np.clip(phi1, -PHI_MAX, PHI_MAX)
            phid1 = np.where(out & (phi1 * phid1 > 0), 0.0, phid1)
        state["n"] = CB.ou_step(state["n"], prm["tau_n"], dt, state["rng"])
        state["phi"], state["phid"] = phi1, phid1
        q["phi"], q["phid"] = phi0, phid0
        q["phidd"] = (phid1 - phid0) / dt
        q["H1"] = dict(state=state, I_x=prm["I_x"])
        return CB.ItemOut(acc=self.damping(prm, ctx, q, phi0, dt),
                          Q=self.forces(prm, ctx, q, phi0, phid0),
                          obs=dict(roll=phi0, roll_rate=phid0), state=state)

    @staticmethod
    def forces(prm, ctx, q, phi, phid):
        W = ctx.W
        ur = q["u_r"]
        a = np.abs(phi)
        X = -ctx.k_drag * ur * np.abs(ur) * prm["cx"] * phi ** 2
        Fz = -W * prm["c2"] * phi ** 2
        Q = CB.lever_Q(ctx, (prm["x_tf"], 0.0, 0.0),
                       np.stack([X, np.zeros_like(X), Fz], 1), q["sr"][:, 5])
        Q[:, 1] += -W * (prm["a1"] * phi + prm["a3"] * phi * a)
        Q[:, 2] += W * (prm["l1"] * phi + prm["l3"] * phi * a) \
            + prm["n_p"] * ctx.L * W / prm["w0"] * phid
        return Q

    @staticmethod
    def damping(prm, ctx, q, phi, dt):
        p = ctx.p
        e = prm["e_phi"] * phi ** 2
        vr, r = q["v_r"], q["sr"][:, 8]
        acc = np.zeros((q["B"], 5))
        acc[:, 1] = CB.lin_damp_acc(
            e * (p["k_lin_sway"] + p["k_sway"] * np.abs(vr)) / p["m_sway"],
            vr, dt)
        acc[:, 2] = CB.lin_damp_acc(e / p["tau_r"], r, dt)
        return acc


# ============================================================ H2 sloshing
def dodge_mode(a, h, m_liq):
    """First sloshing mode of a rectangular tank of length a (along the
    motion) and liquid depth h (Dodge 2000 (1.19a), (3.10)): omega^2 and
    the sloshing mass."""
    th = np.tanh(np.pi * h / a)
    return np.pi * GRAV / a * th, m_liq * 8.0 * a * th / (np.pi ** 3 * h)


@CB.register
class H2(CB.CatItem):
    """Per tank i and direction d in (x, y): s'' + 2 zeta w s' + w^2 s =
    -a_t (a_t the tank point's acceleration incl. gravity along the pitch /
    roll: a_x = u' - r v + g sin(theta_b) - z_t theta_b'', a_y = v' + u r +
    r' x_t + g sin(phi) - z_t phi''), |s| <= a / 4 (wall, inelastic).
    Error on the hull: the low-fidelity boat carries the fuel as rigid
    mass, so only -m_1 s'' at the tank point (x_t, 0, z_t) plus the moment
    of the moved weight (pitch -m_1 g s_x; roll -m_1 g s_y + m_1 z_t s_y''
    to H1 through roll_kick). Calm steady running: exactly 0."""

    code, stage = "H2", "force"
    calm_zero = True

    def draw_on(self, rng, ctx):
        u = rng.uniform
        n = 1 if rng.random() < 0.6 else 2
        frac = float(logu(rng, 0.02, 0.15))
        w = rng.dirichlet(np.ones(n)) if n > 1 else np.ones(1)
        load = getattr(ctx, "load", None)
        lev = 1.0 + (load["eps_m"] if load is not None else 0.0)
        cols = {k: [] for k in ("ax", "ay", "h", "m_liq", "w2x", "w2y", "m1x",
                                "m1y", "zx", "zy", "x_t", "z_t")}
        for i in range(n):
            ax, ay = float(u(0.4, 1.5)), float(u(0.3, 1.2))
            h = float(np.clip(u(0.05, 0.8) * lev, 0.03, 0.9)) * ax
            ml = frac * float(w[i]) * ctx.m
            w2x, m1x = dodge_mode(ax, h, ml)
            w2y, m1y = dodge_mode(ay, h, ml)
            for k, v in (("ax", ax), ("ay", ay), ("h", h), ("m_liq", ml),
                         ("w2x", w2x), ("w2y", w2y), ("m1x", m1x),
                         ("m1y", m1y), ("zx", logu(rng, 0.005, 0.15)),
                         ("zy", logu(rng, 0.005, 0.15)),
                         ("x_t", u(-0.3, 0.2) * ctx.L), ("z_t", u(-0.2, 0.3))):
                cols[k].append(float(v))
        prm = {k: np.array(v) for k, v in cols.items()}
        prm["n"] = n
        return prm

    def init_state(self, params, B=1, rng=None):
        n = params["n"]
        z = lambda: np.zeros((B, n))                       # noqa: E731
        return dict(sx=z(), sxd=z(), sy=z(), syd=z())

    @staticmethod
    def _advance(s, sd, a_t, w2, zeta, a, dt):
        w = np.sqrt(w2)
        s1, sd1 = CB.osc_step(s, sd, w2[None], (2 * zeta * w)[None], dt,
                              x_eq=-a_t / w2[None])
        cap = 0.25 * a[None]
        out = np.abs(s1) > cap
        s1 = np.clip(s1, -cap, cap)
        sd1 = np.where(out & (s1 * sd1 > 0), 0.0, sd1)
        return s1, sd1

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        B = q["B"]
        sr, nu = q["sr"], q["nu0"]
        u, r, v = sr[:, 2], sr[:, 8], sr[:, 9]
        xt, zt = prm["x_t"][None], prm["z_t"][None]
        thb = q["trim_rel"][:, None]
        thbdd = (ctx.sp * nu[:, 4])[:, None]
        phi = np.asarray(q["phi"])[:, None]
        phidd = np.asarray(q.get("phidd", np.zeros(B)))[:, None]
        atx = (nu[:, 0] - r * v)[:, None] + GRAV * np.sin(thb) - zt * thbdd
        aty = (nu[:, 1] + u * r)[:, None] + xt * nu[:, 2][:, None] \
            + GRAV * np.sin(phi) - zt * phidd
        s0x, sd0x, s0y, sd0y = (state[k].copy() for k in ("sx", "sxd", "sy",
                                                        "syd"))
        s1x, sd1x = self._advance(s0x, sd0x, atx, prm["w2x"], prm["zx"],
                                  prm["ax"], dt)
        s1y, sd1y = self._advance(s0y, sd0y, aty, prm["w2y"], prm["zy"],
                                  prm["ay"], dt)
        state.update(sx=s1x, sxd=sd1x, sy=s1y, syd=sd1y)
        sddx, sddy = (sd1x - sd0x) / dt, (sd1y - sd0y) / dt
        smx, smy = 0.5 * (s0x + s1x), 0.5 * (s0y + s1y)
        Fx = -prm["m1x"][None] * sddx
        Fy = -prm["m1y"][None] * sddy
        Q = np.zeros((B, 5))
        for i in range(prm["n"]):
            Q += CB.lever_Q(ctx, (prm["x_t"][i], 0.0, prm["z_t"][i]),
                            np.stack([Fx[:, i], Fy[:, i], np.zeros(B)], 1),
                            sr[:, 5])
        Q[:, 4] += ctx.sp * (-(prm["m1x"][None] * GRAV * smx)).sum(1)
        K = (-prm["m1y"][None] * GRAV * smy
             + prm["m1y"][None] * zt * sddy).sum(1)
        roll_kick(q, K, dt)
        return CB.ItemOut(Q=Q, state=state)


# ======================================================= H3 bilge water
@CB.register
class H3(CB.CatItem):
    """Hidden water mass m_b at x_b (on the keel, + 5 cm):
    x_b'' = -a_x,local - g sin(theta_abs) - c x_b'|x_b'| (implicit),
    x_b in [x_aft, x_fwd] with inelastic end stops; theta_abs = the hidden
    true running trim tau0 + the pitch relative to the low-fidelity
    running attitude. m_b: green-water events at rate lambda while the D9.6
    bow height < 0 (each LogU[5, 100] kg), a leak, exact exponential
    drainage tau_d; capped at 0.1 m. Force on the hull = -m_b a_water +
    m_b g (the water is not in the low-fidelity boat at all): body-x
    -m_b (a_x,pt + x_b'') - m_b g sin(theta_abs) (= the friction and the
    end-stop force, the slide itself is free), vertical -m_b (g + a_z,pt)."""

    code, stage = "H3", "force"

    def draw_on(self, rng, ctx):
        u = rng.uniform
        x0 = float(ctx.x_st[0])
        x_aft = x0 + 0.03 * ctx.L
        x_fwd = x0 + float(u(0.3, 0.7)) * ctx.L
        area = (x_fwd - x_aft) * 0.5 * ctx.B
        leak = float(logu(rng, 0.001, 0.05)) / 3600.0         # m/s of depth
        return dict(m0=float(logu(rng, 0.002, 0.05)) * ctx.m,
                    rate_g=float(logu(rng, 0.2, 2.0)),
                    tau_d=float(logu(rng, 5.0, 300.0)),
                    leak_kg=CB.RHO_W * area * leak, c_f=float(logu(rng, 0.5,
                                                                   5.0)),
                    x_aft=x_aft, x_fwd=x_fwd, m_cap=0.1 * ctx.m)

    def init_state(self, params, B=1, rng=None):
        return dict(mb=np.full(B, params["m0"]),
                    xb=np.full(B, params["x_aft"]), xbd=np.zeros(B),
                    rng=_state_rng(rng))

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        B = q["B"]
        sr = q["sr"]
        rng = state["rng"]
        H = CB.bow_height(ctx, sr, q["eta"])
        hit = CB.poisson_hit(prm["rate_g"] * (H < 0.0), dt, rng, B)
        add = np.where(hit, logu(rng, 5.0, 100.0, B), 0.0)
        mb0, xb0, xd0 = state["mb"].copy(), state["xb"].copy(), \
            state["xbd"].copy()
        mb1 = CB.exp_update(mb0, prm["leak_kg"] * prm["tau_d"], prm["tau_d"],
                            dt) + add
        state["mb"] = np.minimum(mb1, prm["m_cap"])
        thb = ctx.tau0 + q["trim_rel"]
        zb = np.interp(xb0, ctx.x_st, ctx.keel5) + 0.05
        axp, azp = _point_acc(ctx, q, xb0, zb)
        xd1 = xd0 + (-axp - GRAV * np.sin(thb)) * dt
        xd1 = xd1 + dt * CB.implicit_quad_drag(xd1, prm["c_f"], 1.0, dt)
        x1 = xb0 + xd1 * dt
        lo, hi = x1 < prm["x_aft"], x1 > prm["x_fwd"]
        x1 = np.clip(x1, prm["x_aft"], prm["x_fwd"])
        xd1 = np.where((lo & (xd1 < 0)) | (hi & (xd1 > 0)), 0.0, xd1)
        state["xb"], state["xbd"] = x1, xd1
        xdd = (xd1 - xd0) / dt
        Fx = -mb0 * (axp + xdd) - mb0 * GRAV * np.sin(thb)
        Fz = -mb0 * (GRAV + azp)
        Q = CB.lever_Q(ctx, (xb0, 0.0, zb),
                       np.stack([Fx, np.zeros(B), Fz], 1), sr[:, 5])
        return CB.ItemOut(Q=Q, state=state)


# ============================================================ B1 loading
def _mrb(m, xg, yg, zg, Izz, Iyy):
    """Rigid-body inertia in the channels (u, v, r, w, q) with the CG-offset
    couplings, as operators_gen.OperatorGen._draw_inertia builds it."""
    M = np.diag([m, m, Izz, m, Iyy]).astype(float)
    for i, j, val in ((0, 2, -m * yg), (0, 4, m * zg), (1, 2, m * xg),
                      (3, 4, -m * xg)):
        M[i, j] = M[j, i] = val
    return M


INERTIA_KEYS = ("m", "xg", "yg", "zg", "Ixz", "Izz", "Iyy")


def base_inertia(ctx):
    """The calibration-load rigid body behind ctx.Mt: the layer-3 (D9) draw
    when present, else a centred body under the neutral diagonal."""
    inr = ctx.inertia
    Mb = np.array(ctx.Mt, float)
    if all(k in inr for k in INERTIA_KEYS):
        return Mb, {k: float(inr[k]) for k in INERTIA_KEYS}
    m = min(ctx.m, 0.97 * min(Mb[0, 0], Mb[1, 1]))
    I0 = m * (0.25 * ctx.L) ** 2
    return Mb, dict(m=m, xg=0.0, yg=0.0, zg=0.0, Ixz=0.0,
                    Izz=min(I0, 0.95 * Mb[2, 2]), Iyy=min(I0, 0.95 * Mb[4, 4]))


def b1_inertia(ctx, eps, dxg, dyg, aA):
    """The merged inertia draw (draft open question 10): the calibration
    boat's rigid part (mass, radii of gyration, I_xz) x (1 + eps) with the
    CG moved by (dxg, dyg); the added-mass part x (1 + aA eps). eps = dxg =
    dyg = 0 returns ctx.Mt exactly. Returns (M_base, M_new, base, new)."""
    Mb, b = base_inertia(ctx)
    m, xg, yg, zg = b["m"], b["xg"], b["yg"], b["zg"]
    MA = Mb - _mrb(m, xg, yg, zg, b["Izz"], b["Iyy"])
    rzz2 = b["Izz"] / m - xg ** 2 - yg ** 2
    ryy2 = b["Iyy"] / m - xg ** 2 - zg ** 2
    s = 1.0 + eps
    m1, xg1, yg1 = m * s, xg + dxg, yg + dyg
    Izz1 = m1 * (rzz2 + xg1 ** 2 + yg1 ** 2)
    Iyy1 = m1 * (ryy2 + xg1 ** 2 + zg ** 2)
    Ixz1 = (b["Ixz"] - m * xg * zg) * s + m1 * xg1 * zg
    Mn = _mrb(m1, xg1, yg1, zg, Izz1, Iyy1) + MA * (1.0 + aA * eps)
    new = dict(m=m1, xg=xg1, yg=yg1, zg=zg, Ixz=Ixz1, Izz=Izz1, Iyy=Iyy1,
               I_r=float(Mn[2, 2]), m_w=float(Mn[3, 3]), I_q=float(Mn[4, 4]))
    return Mb, Mn, b, new


RB_KEYS = INERTIA_KEYS + ("I_r", "m_w", "I_q")


def rigid_coriolis(rb, Minv, m0, sr):
    """operators_gen.OperatorGen.coriolis for a rigid body rb = (m, xg,
    yg, zg, Ixz, ...) and M_t."""
    ns = types.SimpleNamespace(m0=m0, Minv=Minv, **dict(zip(
        ("m", "xg", "yg", "zg", "Ixz"), (float(x) for x in rb[:5]))))
    return G.OperatorGen.coriolis(ns, sr)


@CB.register
class B1(CB.CatItem):
    """lambda = (eps_m, dx_g, dy_g) changes together (draft 3.4):
      inertia    merged with the D9 draw (b1_inertia); the shared M_t
      mass term  -(M_new - M_base) nu (D8.1 / D9.1 (e)) through M_new^-1,
                 nu = nu0 with heave / pitch replaced by the loaded boat's
                 (nu0 + the change below); the heave / pitch diagonal of
                 M_new - M_base is carried by the frequency change instead
      drag       -a_R eps k_drag u_r|u_r|
      heave      w_h' = w_h (1 - e_m eps) (zeta kept), equilibrium
                 dz0 = -d_m eps T
      pitch      w_p' = w_p (1 - e_m eps), running trim (bow up)
                 dth0 = -b_x dx_g / L + b_m eps tau0 (CG aft -> bow up)
      Coriolis   the D9.1 rigid-body difference of the loaded body minus
                 the one layer 3 already adds
      heel       dy_g -> H1's phi_b (ctx.load), dx_g -> W2's margin"""

    code, stage = "B1", "force"
    gated = True

    def draw_on(self, rng, ctx):
        u = rng.uniform
        eps = float(u(-0.1, 0.25))
        dxg = float(u(-0.03, 0.03)) * ctx.L
        dyg = float(u(-0.03, 0.03)) * ctx.B
        aA, aR = float(u(0.0, 1.0)), float(u(0.5, 1.5))
        bx = float(u(0.5, 1.5)) * DEG / 0.03
        bm, dm, em = (float(x) for x in u(0.0, 1.0, 3))
        return self.build(ctx, eps, dxg, dyg, aA, aR, bx, bm, dm, em)

    @staticmethod
    def build(ctx, eps, dxg, dyg, aA, aR, bx, bm, dm, em):
        Mb, Mn, b, new = b1_inertia(ctx, eps, dxg, dyg, aA)
        return dict(eps=eps, dxg=dxg, dyg=dyg, aA=aA, aR=aR, bx=bx, bm=bm,
                    dm=dm, em=em, dth0=-bx * dxg / ctx.L + bm * eps * ctx.tau0,
                    dz0=-dm * eps * ctx.T, M_base=Mb, M_new=Mn,
                    Minv_base=np.linalg.inv(Mb), Minv_new=np.linalg.inv(Mn),
                    rb_base=np.array([b[k] for k in INERTIA_KEYS]),
                    rb_new=np.array([new[k] for k in RB_KEYS]),
                    m0=float(ctx.p["m_coriolis"]))

    def adjust_ctx(self, params, ctx):
        ctx.set_inertia(params["M_new"], dict(
            zip(RB_KEYS, (float(x) for x in params["rb_new"])),
            b1_eps=float(params["eps"])))
        s = 1.0 + params["eps"]
        ctx.m *= s
        ctx.W *= s
        ctx.vol *= s
        ctx.load = dict(eps_m=params["eps"], dxg=params["dxg"],
                        dyg=params["dyg"], dxg_L=params["dxg"] / ctx.L,
                        dyg_B=params["dyg"] / ctx.B)

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        sr = q["sr"]
        dM = prm["M_new"] - prm["M_base"]
        dM[3, 3] = dM[4, 4] = 0.0
        # heave / pitch: the loaded boat's restoring (frequency, attitude)
        hp = np.zeros((q["B"], 2))
        f = 1.0 - prm["em"] * prm["eps"]
        for j, w, z, e, off, rate in (
                (0, ctx.wh, ctx.zh, q["e_z"], prm["dz0"], sr[:, 4]),
                (1, ctx.wp, ctx.zp, q["e_th"], ctx.sp * prm["dth0"],
                 sr[:, 6])):
            w1 = w * f
            hp[:, j] = (-(w1 ** 2 - w ** 2) * e + w1 ** 2 * off
                        + CB.lin_damp_acc(2 * z * (w1 - w), rate, dt))
        # inertia change on the loaded boat's accelerations (its heave /
        # pitch = the low-fidelity ones + hp; the diagonal of those two is
        # already in the frequency change)
        nu = np.array(q["nu0"], float)
        nu[:, 3:5] += hp
        Q = -(nu @ dM.T)
        ur = q["u_r"]
        Q[:, 0] += -prm["aR"] * prm["eps"] * ctx.k_drag * ur * np.abs(ur)
        acc = Q @ prm["Minv_new"].T
        acc[:, 3:5] += hp
        acc += rigid_coriolis(prm["rb_new"], prm["Minv_new"], prm["m0"], sr) \
            - rigid_coriolis(prm["rb_base"], prm["Minv_base"], prm["m0"],
                             sr)
        return CB.ItemOut(acc=acc, state=state)


# ===================================================== B2 load events
@CB.register
class B2(CB.CatItem):
    """(a) CG jumps (w.p. 0.05): Poisson rate LogU[1/3600, 1/60] 1/s,
    increments scale x t(df), the CG clipped to |dx| <= 0.05 L, |dy| <=
    0.1 B; pitch moment of the weight at dx, roll moment -W dy to H1.
    (b) one point-mass step (w.p. 0.03): dm ~ U[-0.1, 0.1] m at a Poisson
    time (rate LogU[1/300, 1/20]), at most once; its weight and inertia.
    (c) towed drag (w.p. 0.03): -c_t |V_p| V_p at the transom, direction
    offset psi_off, two-state on / off with random dwell times; c_t =
    f_t k_drag, f_t ~ LogU[0.02, 0.3] (share of the low-fidelity drag)."""

    code, stage = "B2", "force"
    P_PART = (0.05, 0.03, 0.03)

    def draw_on(self, rng, ctx):
        u = rng.uniform
        while True:
            on = [bool(rng.random() < p) for p in self.P_PART]
            if any(on):
                break
        prm = dict(cg=on[0], mass=on[1], tow=on[2])
        if on[0]:
            prm.update(rate_cg=float(logu(rng, 1 / 3600, 1 / 60)),
                       sx=float(logu(rng, 0.005, 0.03)) * ctx.L,
                       sy=float(logu(rng, 0.01, 0.05)) * ctx.B,
                       df=float(u(2.0, 5.0)), cap_x=0.05 * ctx.L,
                       cap_y=0.1 * ctx.B)
        if on[1]:
            prm.update(rate_m=float(logu(rng, 1 / 300, 1 / 20)),
                       dm=float(u(-0.1, 0.1)) * ctx.m,
                       x_m=float(u(-0.3, 0.3)) * ctx.L,
                       z_m=float(u(0.0, 0.5)))
        if on[2]:
            prm.update(c_t=float(logu(rng, 0.02, 0.3)) * ctx.k_drag,
                       dwell_on=float(logu(rng, 10.0, 300.0)),
                       dwell_off=float(logu(rng, 10.0, 300.0)),
                       psi_off=float(u(-10.0, 10.0)) * DEG,
                       x_tow=float(ctx.x_st[0]))
        return prm

    def init_state(self, params, B=1, rng=None):
        rng = _state_rng(rng)
        return dict(dx=np.zeros(B), dy=np.zeros(B), mdone=np.zeros(B, bool),
                    tow=rng.random(B) < 0.5, rng=rng)

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        B = q["B"]
        sr = q["sr"]
        rng = state["rng"]
        Q = np.zeros((B, 5))
        if prm["cg"]:
            hit = CB.poisson_hit(prm["rate_cg"], dt, rng, B)
            jx = prm["sx"] * rng.standard_t(prm["df"], B)
            jy = prm["sy"] * rng.standard_t(prm["df"], B)
            state["dx"] = np.clip(state["dx"] + np.where(hit, jx, 0.0),
                                  -prm["cap_x"], prm["cap_x"])
            state["dy"] = np.clip(state["dy"] + np.where(hit, jy, 0.0),
                                  -prm["cap_y"], prm["cap_y"])
            Q[:, 4] += ctx.sp * state["dx"] * (-ctx.W)
            roll_kick(q, -ctx.W * state["dy"], dt)
        if prm["mass"]:
            hit = CB.poisson_hit(prm["rate_m"], dt, rng, B)
            state["mdone"] = state["mdone"] | hit
            dm = np.where(state["mdone"], prm["dm"], 0.0)
            thb = ctx.tau0 + q["trim_rel"]
            axp, azp = _point_acc(ctx, q, prm["x_m"], prm["z_m"])
            F = np.stack([-dm * (axp + GRAV * np.sin(thb)), np.zeros(B),
                          -dm * (GRAV + azp)], 1)
            Q += CB.lever_Q(ctx, (prm["x_m"], 0.0, prm["z_m"]), F, sr[:, 5])
        if prm["tow"]:
            dwell = np.where(state["tow"], prm["dwell_on"], prm["dwell_off"])
            flip = rng.random(B) < -np.expm1(-dt / dwell)
            state["tow"] = state["tow"] ^ flip
            Vx = q["u_r"]
            Vy = q["v_r"] + sr[:, 8] * prm["x_tow"]
            V = np.hypot(Vx, Vy)
            ex, ey = Vx / np.maximum(V, 1e-9), Vy / np.maximum(V, 1e-9)
            c, s = np.cos(prm["psi_off"]), np.sin(prm["psi_off"])
            e = np.stack([c * ex - s * ey, s * ex + c * ey, np.zeros(B)], 1)
            g = CB.lever_Q(ctx, (prm["x_tow"], 0.0, 0.0), e, sr[:, 5])
            m_pt = 1.0 / np.einsum("bi,ij,bj->b", g, ctx.Minv, g)
            a = CB.implicit_quad_drag(V, prm["c_t"], m_pt, dt)
            Q += np.where(state["tow"], m_pt * a, 0.0)[:, None] * g
        return CB.ItemOut(Q=Q, state=state)


# ============================================================= B3 fouling
@CB.register
class B3(CB.CatItem):
    """Delta R = phi_F k_drag Delta_f u_r|u_r| against the motion, phi_F ~
    U[0.3, 0.6] (friction share), Delta_f ~ LogU[0.05, 0.8] (Schultz 2007);
    linearised implicit on the surge point mass."""

    code, stage = "B3", "force"

    def draw_on(self, rng, ctx):
        return dict(phi_F=float(rng.uniform(0.3, 0.6)),
                    D_f=float(logu(rng, 0.05, 0.8)))

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        m_pt = 1.0 / ctx.Minv[0, 0]
        c = prm["phi_F"] * ctx.k_drag * prm["D_f"]
        Q = np.zeros((q["B"], 5))
        Q[:, 0] = m_pt * CB.implicit_quad_drag(q["u_r"], c, m_pt, dt)
        return CB.ItemOut(Q=Q, state=state)


# ================================================================ checks
@CB.check("H1")
def heel_amplification_bounded(item, ctx):
    """D10.8: away from the two-stable-heel draws, the static heel
    amplification 1 / (1 - kappa (U / U_max)^2) is <= 2 over the task
    speeds (kappa <= 0.5)."""
    rng = np.random.default_rng(17)
    worst = 0.0
    for _ in range(400):
        prm = item.draw_on(rng, ctx)
        if prm["persist"]:
            continue
        k = item.k_lin(prm, ctx.u_max, ctx) / prm["w0"] ** 2
        worst = max(worst, 1.0 / k)
    return worst <= H1_AMP_MAX + 1e-12,         f"max static heel amplification at U_max {worst:.3f}"


def _rows(ctx, u):
    u = np.atleast_1d(np.asarray(u, float))
    sr = np.zeros((len(u), 10))
    sr[:, 2], sr[:, 3], sr[:, 5] = u, ctx.z0, ctx.th0
    return sr


def _q(ctx, sr, thr=0.0, nu0=None):
    sea = CB.CatSea(None, ctx).sample(sr, 0.0)
    return CB.substep_quantities(ctx, sr, thr, 0.0, sea, nu0=nu0)


def _h1_quiet(item, ctx, rng, **over):
    """An H1 draw with no heel target, no noise, no chine walk and no
    bistable part, then the overrides."""
    prm = item.draw_on(rng, ctx)
    prm.update(persist=False, kap=0.0, c3=0.0, cw=False, gam=0.0,
               phi_b=0.0, phi_Q=0.0, phi_n=0.0, c_w=0.0, c_d=0.0)
    prm.update(over)
    return prm


def _h1_run(item, prm, ctx, sr, n, thr=0.0, phi0=None, kick=None):
    B = len(sr)
    st = item.init_state(prm, B, rng=np.random.default_rng(0))
    if phi0 is not None:
        st["phi"], st["init"] = np.asarray(phi0, float).copy(), True
    hist, q = [], _q(ctx, sr, thr)
    for _ in range(n):
        pass
        item.step(prm, st, q, CB.DT_SUB)
        if kick is not None:
            roll_kick(q, kick, CB.DT_SUB)
        hist.append((st["phi"].copy(), st["phid"].copy()))
    return st, hist


@CB.check("H1")
def h1_free_decay(item, ctx):
    """Linear stable roll (kappa = 0) never gains energy: exact transition."""
    rng = np.random.default_rng(21)
    worst = -np.inf
    for _ in range(8):
        prm = _h1_quiet(item, ctx, rng)
        _, h = _h1_run(item, prm, ctx, _rows(ctx, [ctx.u_id]), 500,
                       phi0=[0.1])
        E = np.array([0.5 * (prm["w0"] ** 2 * p[0] ** 2 + v[0] ** 2)
                      for p, v in h])
        worst = max(worst, float(np.diff(E).max() / E[0]))
    return worst <= 1e-12, f"max relative energy step {worst:.1e}"


@CB.check("H1")
def h1_bistable(item, ctx):
    """Persistent heel: above the crossing speed the roll settles on
    +- sqrt(-k_lin / c3), on the side it started."""
    rng = np.random.default_rng(22)
    worst = 0.0
    for _ in range(5):
        Uc = float(rng.uniform(ctx.u_lo, 0.8 * ctx.u_max))
        kap = (ctx.u_max / Uc) ** 2
        prm = _h1_quiet(item, ctx, rng, persist=True, kap=kap, zeta=0.3)
        prm["c3"] = prm["w0"] ** 2 * max(kap - 1, 0.25) / (5 * DEG) ** 2
        sr = _rows(ctx, [ctx.u_max, ctx.u_max])
        st, _ = _h1_run(item, prm, ctx, sr, 750, phi0=[0.01, -0.01])
        want = np.sqrt(-item.k_lin(prm, ctx.u_max, ctx) / prm["c3"])
        worst = max(worst, float(np.abs(st["phi"] - [want, -want]).max()
                                 / want))
    return worst < 0.02, f"max relative error of the two heels {worst:.1e}"


@CB.check("H1")
def h1_chine_walk(item, ctx):
    """Chine walking: above U_cw with gamma > zeta a small roll grows into
    a bounded limit cycle."""
    rng = np.random.default_rng(23)
    amps = []
    for _ in range(5):
        prm = _h1_quiet(item, ctx, rng, cw=True, gam=0.3, zeta=0.05,
                        w_U=2 * KN, phi_sat=5 * DEG)
        prm["U_cw"] = float(rng.uniform(20, 25)) * KN
        sr = _rows(ctx, [prm["U_cw"] + 4 * prm["w_U"]])
        _, h = _h1_run(item, prm, ctx, sr, 1000, phi0=[0.2 * DEG])
        amps.append(float(np.abs([p[0] for p, _ in h[-250:]]).max()))
    lo, hi = min(amps), max(amps)
    ok = lo > 10 * 0.2 * DEG and hi <= PHI_MAX
    return ok, (f"limit-cycle amplitude {np.degrees(lo):.1f}-"
                f"{np.degrees(hi):.1f} deg from 0.2 deg")


@CB.check("H1")
def h1_torque_heel(item, ctx):
    """Impeller torque: the heel moves by phi_Q between zero and full
    thrust; the sign is random over draws, fixed within one."""
    rng = np.random.default_rng(24)
    worst, signs = 0.0, []
    for _ in range(20):
        prm = item.draw_on(rng, ctx)
        signs.append(float(np.sign(prm["phi_Q"])))
        prm2 = _h1_quiet(item, ctx, rng, phi_Q=prm["phi_Q"])
        sr = _rows(ctx, [ctx.u_id, ctx.u_id])
        st, _ = _h1_run(item, prm2, ctx, sr, 10,
                        thr=np.array([0.0, ctx.t_max]))
        worst = max(worst, abs(st["phi"][1] - st["phi"][0] - prm["phi_Q"]))
    both = min(signs.count(1.0), signs.count(-1.0)) > 0
    return worst < 1e-9 and both, (f"max error {worst:.1e} rad; both signs "
                                   f"drawn {both}")


@CB.check("H1")
def h1_parity(item, ctx):
    """Sway force and yaw moment odd in phi; heave, surge, pitch even."""
    rng = np.random.default_rng(25)
    q = _q(ctx, _rows(ctx, [ctx.u_id] * 3))
    phi = np.array([0.03, 0.08, 0.15])
    worst = 0.0
    for _ in range(20):
        prm = item.draw_on(rng, ctx)
        Qp = item.forces(prm, ctx, q, phi, np.zeros(3))
        Qm = item.forces(prm, ctx, q, -phi, np.zeros(3))
        worst = max(worst, float(np.abs(Qp[:, 1:3] + Qm[:, 1:3]).max()),
                    float(np.abs(Qp[:, [0, 3, 4]] - Qm[:, [0, 3, 4]]).max()))
    return worst < 1e-9, f"max parity defect {worst:.1e} N"


@CB.check("H1")
def h1_roll_moment(item, ctx):
    """A roll moment through roll_kick settles at K / (I_x w0^2)."""
    rng = np.random.default_rng(26)
    worst = 0.0
    for _ in range(5):
        prm = _h1_quiet(item, ctx, rng, zeta=0.3)
        K = 1000.0
        st, _ = _h1_run(item, prm, ctx, _rows(ctx, [ctx.u_id]), 750, kick=K)
        want = K / (prm["I_x"] * prm["w0"] ** 2)
        worst = max(worst, abs(st["phi"][0] / want - 1))
    return worst < 0.02, f"max relative error {worst:.1e}"


@CB.check("H2")
def h2_modes(item, ctx):
    """Dodge modes: frequency range over draws, sloshing mass < liquid."""
    rng = np.random.default_rng(31)
    w, ok = [], True
    for _ in range(500):
        p = item.draw_on(rng, ctx)
        w += list(np.sqrt(np.r_[p["w2x"], p["w2y"]]))
        ok &= bool(np.all(p["m1x"] < p["m_liq"])
                   and np.all(p["m1y"] < p["m_liq"]))
    return ok, (f"omega {min(w):.2f}-{max(w):.2f} rad/s (draft ~1.8-9); "
                f"m_1 < m_liq {ok}")


@CB.check("H2")
def h2_steady_accel(item, ctx):
    """Constant surge acceleration: the liquid settles at -a / w^2 (or on
    the wall), the dynamic reaction vanishes and only the moved weight's
    pitch moment stays; |s| <= a / 4 throughout."""
    rng = np.random.default_rng(32)
    worst, wall = 0.0, 0.0
    sr = _rows(ctx, [ctx.u_id])
    nu0 = np.array([[0.5, 0.0, 0.0, 0.0, 0.0]])
    for _ in range(8):
        p = item.draw_on(rng, ctx)
        p["zx"] = np.maximum(p["zx"], 0.2)
        st = item.init_state(p, 1)
        q = _q(ctx, sr, nu0=nu0)
        for _k in range(1000):
            o = item.step(p, st, q, CB.DT_SUB)
            wall = max(wall, float((np.abs(st["sx"])
                                    / (0.25 * p["ax"])).max()))
        want = np.clip(-0.5 / p["w2x"], -0.25 * p["ax"], 0.25 * p["ax"])
        worst = max(worst, float(np.abs(st["sx"][0] - want).max()))
        Mw = ctx.sp * (-(p["m1x"] * GRAV * st["sx"][0])).sum()
        worst = max(worst, abs(o.Q[0, 4] - Mw) / max(abs(Mw), 1.0),
                    abs(o.Q[0, 0]))
    return worst < 1e-3 and wall <= 1 + 1e-12, (
        f"max settle / reaction error {worst:.1e}; max |s| / (a/4) "
        f"{wall:.3f}")


@CB.check("H3")
def h3_slide(item, ctx):
    """The water stays between the stops, runs aft at the running trim and
    forward under hard braking; calm water: no green water, exact
    drainage; at rest on the stop the load is its weight."""
    rng = np.random.default_rng(41)
    bad = []
    sr = _rows(ctx, [ctx.u_id, ctx.u_id])
    nu0 = np.array([[0.0] * 5, [-4.0, 0.0, 0.0, 0.0, 0.0]])
    for _ in range(5):
        p = item.draw_on(rng, ctx)
        st = item.init_state(p, 2, rng=np.random.default_rng(1))
        st["xb"][:] = 0.5 * (p["x_aft"] + p["x_fwd"])
        n = 500
        q = _q(ctx, sr, nu0=nu0)
        for _k in range(n):
            mb = st["mb"][0]
            o = item.step(p, st, q, CB.DT_SUB)
            if (st["xb"] < p["x_aft"] - 1e-12).any() or \
                    (st["xb"] > p["x_fwd"] + 1e-12).any():
                bad.append("out of the stops")
        if not (st["xb"][0] == p["x_aft"] and st["xb"][1] == p["x_fwd"]):
            bad.append("end positions")
        inf = p["leak_kg"] * p["tau_d"]
        want = inf + (p["m0"] - inf) * np.exp(-n * CB.DT_SUB / p["tau_d"])
        if abs(st["mb"][0] / want - 1) > 1e-9:
            bad.append("drainage")
        Fz = o.Q[0, 3] - ctx.sp * ctx.th0 * o.Q[0, 0]
        if abs(o.Q[0, 0] + mb * GRAV * np.sin(ctx.tau0)) > 1e-9 * mb or \
                abs(Fz + mb * GRAV) > 1e-9 * mb * GRAV:
            bad.append("static load")
    return not bad, ("; ".join(sorted(set(bad))) or
                     "stops held, aft at trim / forward braking, exact "
                     "drainage, weight on the stop")


def _d9_ctx(ctx, seed):
    L3 = CB.OperatorGenL3(seed, "none", p=ctx.p, L=ctx.L)
    c = CB.CatCtx(ctx.p)
    c.set_inertia(L3.Mt, L3.inertia)
    return c


@CB.check("B1")
def b1_merged_inertia(item, ctx):
    """Zero loading = the D9 inertia exactly and no acceleration; every
    loaded M_t symmetric positive definite with rigid mass m_D9 (1 + eps);
    in a CatDraw the shared M_t is B1's and H1's heel bias reads dy_g."""
    rng = np.random.default_rng(51)
    bad, worst0 = [], 0.0
    for s in range(10):
        c = _d9_ctx(ctx, 1000 + s)
        p0 = item.build(c, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5)
        if np.abs(p0["M_new"] - c.Mt).max() != 0.0:
            bad.append("M_new != M_t at zero load")
        sr = _rows(c, rng.uniform(3, 25, 16))
        sr[:, 3] += rng.normal(0, 0.1, 16)
        sr[:, [4, 6, 8, 9]] = rng.normal(0, 0.5, (16, 4))
        o = item.step(p0, {}, _q(c, sr, nu0=rng.normal(0, 2, (16, 5))),
                      CB.DT_SUB)
        worst0 = max(worst0, float(np.abs(o.acc).max()))
        p = item.draw_on(rng, c)
        M = p["M_new"]
        if np.abs(M - M.T).max() > 1e-9 or np.linalg.eigvalsh(M).min() <= 0:
            bad.append("M_new not SPD")
        if abs(p["rb_new"][0] - c.inertia["m"] * (1 + p["eps"])) > 1e-9:
            bad.append("rigid mass")
    d = CB.CatDraw(7, tier="none", items=("B1", "H1"), layer1=False,
                   sparse=False, force_items={"B1": True, "H1": True})
    pb, ph = d.on["B1"], d.on["H1"]
    if np.abs(d.ctx.Mt - pb["M_new"]).max() != 0.0:
        bad.append("CatDraw M_t")
    want = -PHI_LOAD_CAP * np.tanh(pb["dyg"] / ph["GM"] / PHI_LOAD_CAP)
    if abs(ph["phi_load"] - want) > 1e-15 or not hasattr(d.ctx, "load"):
        bad.append("H1 heel bias not linked")
    msg = f"max |a| at zero load {worst0:.1e}"
    return not bad and worst0 == 0.0, ("; ".join(sorted(set(bad))) + "; "
                                       if bad else "") + msg + \
        "; SPD, rigid mass, CatDraw M_t and heel link checked"


@CB.check("B1")
def b1_attitude_and_mass(item, ctx):
    """The loaded boat rests at the shifted heave / trim (low-fidelity
    heave / pitch acceleration + B1's = 0 there, drag change off so that no
    surge force couples in through M_t); a heavier boat accelerates less under the same
    low-fidelity force."""
    rng = np.random.default_rng(52)
    worst, slower = 0.0, True
    for s in range(10):
        c = _d9_ctx(ctx, 2000 + s)
        p = item.draw_on(rng, c)
        sr = _rows(c, [c.u_id])
        sr[0, 3] += p["dz0"]
        sr[0, 5] += c.sp * p["dth0"]
        p = item.build(c, p["eps"], p["dxg"], p["dyg"], p["aA"], 0.0,
                       p["bx"], p["bm"], p["dm"], p["em"])
        q = _q(c, sr)
        o = item.step(p, {}, q, CB.DT_SUB)
        tot = o.acc[0, 3:5] + q["a_lofi_vert"][0]
        worst = max(worst, float(np.abs(tot).max()))
        p2 = item.build(c, 0.2, 0.0, 0.0, p["aA"], p["aR"], p["bx"], p["bm"],
                        p["dm"], p["em"])
        o2 = item.step(p2, {}, _q(c, _rows(c, [0.0]),
                                  nu0=np.array([[1.0, 0, 0, 0, 0]])),
                       CB.DT_SUB)
        slower &= bool(0.0 < 1.0 + o2.acc[0, 0] < 1.0)
    return worst < 1e-9 and slower, (
        f"max heave / pitch error at the shifted rest {worst:.1e}; "
        f"heavier boat slower {slower}")


@CB.check("B2")
def b2_parts_and_bounds(item, ctx):
    """Sub-parts on at p_i / P_ON; the CG stays within its caps; the mass
    step happens at most once; the tow force never adds power at the tow
    point (psi_off = 0)."""
    rng = np.random.default_rng(61)
    n, cnt = 20000, np.zeros(3)
    for _ in range(n):
        p = item.draw_on(rng, ctx)
        cnt += [p["cg"], p["mass"], p["tow"]]
    want = np.array(item.P_PART) / CB.P_ON["B2"]
    z = np.abs(cnt / n - want) / np.sqrt(want * (1 - want) / n)
    p = item.draw_on(np.random.default_rng(1), ctx)
    p.update(rate_cg=5.0, sx=0.05 * ctx.L, sy=0.1 * ctx.B, df=2.0,
             cap_x=0.05 * ctx.L, cap_y=0.1 * ctx.B, rate_m=2.0, dm=100.0,
             x_m=0.0, z_m=0.2, c_t=4.0, dwell_on=1.0, dwell_off=1.0,
             psi_off=0.0, x_tow=float(ctx.x_st[0]))
    B = 64
    st = item.init_state(p, B, rng=np.random.default_rng(2))
    sr = _rows(ctx, rng.uniform(2, 20, B))
    sr[:, 8], sr[:, 9] = rng.normal(0, 0.3, B), rng.normal(0, 1, B)
    ok, pw = True, -np.inf
    q = _q(ctx, sr)
    for _ in range(300):
        was = st["mdone"].copy()
        o = item.step(dict(p, cg=False, mass=False, tow=True), st, q,
                      CB.DT_SUB)
        vy = sr[:, 9] + sr[:, 8] * p["x_tow"]
        pw = max(pw, float((o.Q[:, 0] * sr[:, 2] + o.Q[:, 1] * vy).max()))
        item.step(dict(p, cg=True, mass=True, tow=False), st, q, CB.DT_SUB)
        ok &= bool(np.all(np.abs(st["dx"]) <= p["cap_x"] + 1e-12)
                   and np.all(np.abs(st["dy"]) <= p["cap_y"] + 1e-12)
                   and np.all(st["mdone"] >= was))
    return z.max() < 4 and ok and pw <= 1e-9, (
        f"sub-part rates {np.round(cnt / n, 3).tolist()} (expect "
        f"{np.round(want, 3).tolist()}); caps and one mass step {ok}; "
        f"max tow power {pw:.1e} W")


@CB.check("B3")
def b3_drag(item, ctx):
    """Zero at rest, against the motion, ~ u|u| (implicit correction
    small), odd in u."""
    rng = np.random.default_rng(71)
    u = np.array([0.0, 8.0, 16.0, -8.0])
    q = _q(ctx, _rows(ctx, u))
    for _ in range(20):
        p = item.draw_on(rng, ctx)
        Q = item.step(p, {}, q, CB.DT_SUB).Q[:, 0]
        if Q[0] != 0.0 or not (Q[1:] * u[1:] < 0).all() or \
                abs(Q[2] / Q[1] / 4 - 1) > 1e-2 or abs(Q[3] + Q[1]) > 1e-9:
            return False, f"form: {Q}"
    return True, "zero at rest, opposes u, quadratic, odd"
