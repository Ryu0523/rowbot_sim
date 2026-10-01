#!/usr/bin/env python3
"""
Catalogue group P (learn/meta/PRIOR_D10_DRAFT.md 3.5, with sections 10 and
11): propulsion and steering, P1-P14, on the framework learn/meta/cat_base.py.

How the items plug in. Every item here has stage 'prop'. CatDraw._prop runs
them in cat_base.PROP_ORDER on one shared chain dict pc (q['prop']):

    c -> P12 rules -> M10 actuator -> P7, P11 -> nozzle angle noz_eff
    T_pre = T_real(P3) x rho(P10) x p(P1, P13) x (nu / nu_ss)^2 (P2)
            x (1 + pulse)(P9)
    T_eff = P4 saturation of T_pre (cavitation, last; P9's cavitation part
            of the noise is applied inside P4, before the saturation)
    F_y   = P5 side force; P8 bucket; P14 residual; P6 vertical force
    error = M_t^-1 J(p_j)^T [F - the low-fidelity boat's own jet force]

An item's step(params, state, q, dt) modifies pc in place when q has
'prop' (inside CatDraw). Called alone (the harness, a per-item preview) it
runs the same CatDraw._prop chain with only itself switched on and returns
the resulting acceleration (PropItem._solo), so the harness sees exactly
the chain's arithmetic.

Chain keys written here beyond cat_base's pc (all optional, read with
defaults): load (pump torque factor for P2, from P1 / P13), kap_p, kap_rho
(side-force exponents of P1 / P10), p1_tau (P1's loss / recovery times,
used by P13), m_torque, lost / reimm (P1's latched suction loss and its
end, for P2's simple variant), nu (P2's speed, for P14), eps9 / sig_c9
(P9's noise for P4's cavitation part), cav (P4's cavitation degree).

Scales only from the low-fidelity boat itself (t_max, u_design = u_id,
u_max, rud_max, k_drag, k_jet_side, the calm-running intake immersion
h_run, rs) and general physics; the section-11 engine power 150-250 kW is
the VM18 specification the user gave. No range comes from Scarab target
statistics. All switch-on probabilities are the draft's [assumption].

Numerics (draft section 6 item 6): first-order states (P1 suction p, P2
engine speed nu, P10 recovery, P12 temperature, P13 air lag) by exact
exponential updates; the P9 coloured noise by the exact OU step; rate
limits and backlash (P7) are exact discrete maps; no quadratic drag or
impulse in this group.

Deviations from the draft (reported to the orchestrator; see also the
item docstrings):
  P1  the static loss S is a logistic normalised to 0 at the calm-running
      immersion (the low-fidelity thrust is right in calm running); the
      recovery threshold h_on + dh is capped below 0.9 x the calm-running
      immersion so the latch always clears in calm water.
  P2  q_e(c, nu) = nu_ss(c)^2 (1 - beta_e (nu / nu_ss(c) - 1)); the ODE is
      advanced by the exact exponential of its linearisation about the
      current equilibrium (the limiter clips it).
  P3  plus a power ceiling T <= eta_P P_eng / u_r from the section-11
      power (the item's only use of it; floored so it never binds at u_id).
      The draft's d2T/dc du <= 0 holds for the phi g part only; the idle
      ram-drag term makes it positive at low throttle (checked and stated).
  P6  the vertical force acts at the jet point (cat_base's chain has no
      moment slot, so the draft's point range [x_stern, x_stern + 0.15 L]
      collapses to its stern end); T_real read as the effective thrust
      T_eff; the section-11 nozzle pitch bias is a second sub-part.
  P7  jet momentum G / G_max read as the actuator throttle (momentum flux
      ~ rpm^2 ~ throttle, available before P3 in the chain order).
  P8  only the stuck-bucket fault (the controller never reverses); the
      gross thrust is read as T_eff.
  P9  inflow speed floored at u_lo / 2 (the pump's own suction).
  P13 the air pulse runs through the same asymmetric lag as P1's suction
      (P13 comes after P1 in the chain), strength over 3 v_slam, bounded
      by 1 - tanh.
"""
import math

import numpy as np

from learn.meta import cat_base as CB
from learn.meta.operators import _logu

DEG, KN = CB.DEG, CB.KN
T_EP = 90.0          # episode length (s), learn/meta/data2.T_EP (not imported)
P_ATM, P_VAP = 101325.0, 2300.0
U_A = math.sqrt(2.0 * (P_ATM - P_VAP) / CB.RHO_W)   # ~13.9 m/s (P4)
PULSE_MAX = 0.9      # |P9 pulse| < 0.9: thrust never reverses from noise


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _cn(pc, ctx):
    """Normalised throttle of the (true) actuator, [0, 1]."""
    return np.clip(pc["thr_act"] / ctx.t_max, 0.0, 1.0)


def _at_intake(ctx, a5):
    """(B, 5) station values -> (B,) at the intake x (linear)."""
    xs, x = ctx.x_st, float(ctx.intake[0])
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    f = float(np.clip((x - xs[i]) / (xs[i + 1] - xs[i]), 0.0, 1.0))
    return a5[:, i] * (1.0 - f) + a5[:, i + 1] * f


def _parts(rng, probs):
    """Sub-part switches given that at least one is on (conditional
    Bernoulli draws by rejection): the item-level P_ON is 1 - prod(1 - p)."""
    while True:
        on = [rng.random() < p for p in probs]
        if any(on):
            return [float(o) for o in on]


def new_pc(q):
    """A fresh chain dict as cat_base.CatDraw._prop starts it (for the
    construction checks that drive one item's chain step directly)."""
    B = q["B"]
    cmd = q["cmd"] if q["cmd"] is not None else np.stack(
        [q["thr"], q["noz"]], 1)
    return dict(c_thr=cmd[:, 0].copy(), c_noz=cmd[:, 1].copy(),
                cmd_changed=False, thr_act=q["thr"].copy(),
                noz_act=q["noz"].copy(), u_r=q["u_r"], T=None,
                rho=np.ones(B), p=np.ones(B), nu_fac=np.ones(B),
                pulse=np.zeros(B), T_pre=None, T_eff=None, Fx=None,
                Fy=None, Fz=np.zeros(B), dF=np.zeros((B, 2)),
                noz_eff=q["noz"].copy())


class _Solo:
    """Stand-in for CatDraw so CatDraw._prop runs with one item on."""

    _t_pre = staticmethod(CB.CatDraw._t_pre)
    _true_actuator = CB.CatDraw._true_actuator

    def __init__(self, code, prm, ctx):
        self.on, self.ctx, self.act = {code: prm}, ctx, None


class PropItem(CB.CatItem):
    """A propulsion-chain item: chain(params, state, pc, q, dt) -> obs."""

    stage = "prop"
    clock = False            # keeps an episode clock state['t']

    def init_state(self, params, B=1, rng=None):
        st = dict(rng=rng if rng is not None else np.random.default_rng(0))
        if self.clock:
            st["t"] = 0.0
        st.update(self.init_more(params, B, st["rng"]))
        return st

    def init_more(self, params, B, rng):
        return {}

    def step(self, params, state, q, dt):
        if "prop" in q:
            obs = self.chain(params, state, q["prop"], q, dt) or {}
            if self.clock:
                state["t"] += dt
            return CB.ItemOut(obs=obs, state=state)
        return self._solo(params, state, q, dt)

    def _solo(self, params, state, q, dt):
        solo = _Solo(self.code, params, q["ctx"])
        st = dict(items={self.code: state}, act=state.get("_solo_act"))
        acc, obs = CB.CatDraw._prop(solo, st, q, dt)
        state["_solo_act"] = st["act"]
        return CB.ItemOut(acc=acc, obs=obs, state=state)

    def chain(self, prm, st, pc, q, dt):
        raise NotImplementedError


# =================================================================== P1
@CB.register
class P1(PropItem):
    """Intake air suction / emergence: a hidden suction degree p driven by
    the intake immersion h (q['h_in'], incl. roll), with the intake point
    perturbed by dx (+-0.05 L) and dz (+-0.3 h_run).
      S(h)  = (sig((h_on - h) / w) - s0)_+ / (1 - s0),  s0 = the same at
              calm running (monotone, saturating, 0 at calm running)
      beta  = 1 - A S(h)
      latch: lost when h < h_on; cleared after h > h_on + dh for t_dwell;
             while lost p cannot rise (target min(beta, p))
      p' = (target - p) / tau, tau_loss falling, tau_rec rising (exact)
    Effects: thrust x p, side force x p^kappa (P5), pump torque x p^m (P2).
    Ranges (draft P1): h_on / h_run ~ U[-0.3, 0.6], w / h_run ~ LogU[0.05,
    1], A ~ U[0.6, 1], dh / h_run ~ U[0, 0.5] (capped, module docstring),
    t_dwell ~ U[0, 0.5] s, tau_loss ~ LogU[0.02, 0.3] s (duct 1-2 m over
    10-25 m/s, widened), tau_rec ~ LogU[0.2, 5] s (ITTC 2002, Kozlowska
    2020), kappa ~ U[0.8, 1.5], m ~ U[0.8, 0.85]."""

    code, calm_zero = "P1", True

    def draw_on(self, rng, ctx):
        u = rng.uniform
        hr = ctx.h_run
        dx = u(-0.05, 0.05) * ctx.L
        dz = u(-0.3, 0.3) * hr
        h_run_p = hr - ctx.sp * dx * ctx.th0 - dz
        h_on = u(-0.3, 0.6) * hr
        dh = u(0.0, 0.5) * hr
        dh = float(np.clip(dh, 0.0, max(0.9 * h_run_p - h_on, 0.0)))
        return dict(dx=dx, dz=dz, h_run_p=h_run_p, h_on=h_on,
                    w=float(_logu(rng, 0.05, 1.0)) * hr, A=u(0.6, 1.0),
                    dh=dh, t_dwell=u(0.0, 0.5),
                    tau_loss=float(_logu(rng, 0.02, 0.3)),
                    tau_rec=float(_logu(rng, 0.2, 5.0)),
                    kap=u(0.8, 1.5), m=u(0.8, 0.85))

    def init_more(self, prm, B, rng):
        return dict(p=np.ones(B), lost=np.zeros(B, bool), tim=np.zeros(B))

    @staticmethod
    def immersion(prm, ctx, q):
        return q["h_in"] - ctx.sp * prm["dx"] * q["sr"][:, 5] \
            + q["slope"] * prm["dx"] - prm["dz"]

    @staticmethod
    def beta(prm, h):
        w = prm["w"]
        s0 = _sig((prm["h_on"] - prm["h_run_p"]) / w)
        S = np.clip((_sig((prm["h_on"] - h) / w) - s0) / (1.0 - s0), 0.0,
                    1.0)
        return 1.0 - prm["A"] * S

    def chain(self, prm, st, pc, q, dt):
        h = self.immersion(prm, q["ctx"], q)
        lost = st["lost"]
        above = h > prm["h_on"] + prm["dh"]
        st["tim"] = np.where(lost & above, st["tim"] + dt, 0.0)
        reimm = lost & above & (st["tim"] >= prm["t_dwell"])
        lost = (lost | (h < prm["h_on"])) & ~reimm
        st["lost"] = lost
        beta = self.beta(prm, h)
        target = np.where(lost, np.minimum(beta, st["p"]), beta)
        st["p"] = CB.exp_update_asym(st["p"], target, prm["tau_rec"],
                                     prm["tau_loss"], dt)
        p = st["p"]
        pc["p"] = pc["p"] * p
        pc["load"] = pc.get("load", 1.0) * p ** prm["m"]
        pc["kap_p"], pc["m_torque"] = prm["kap"], prm["m"]
        pc["p1_tau"] = (prm["tau_loss"], prm["tau_rec"])
        pc["lost"], pc["reimm"] = lost, reimm
        pc["p1"] = p                  # P1's own suction (diagnostics)
        return {}


# =================================================================== P2
@CB.register
class P2(PropItem):
    """Engine-pump speed and the rev limiter (drawn only with P1). Full
    variant (w.p. 0.5): normalised speed nu (1 = full throttle, full load),
    nu_ss(c) = sqrt(c_idle + (1 - c_idle) c) (thrust ~ nu^2; idle speed
    0.15-0.3 of full), engine torque q_e = nu_ss^2 (1 - beta_e (nu / nu_ss
    - 1)) (non-increasing in nu), pump torque load x nu^2 (load = p^m from
    P1 / P13):  tau_n nu' = q_e - load nu^2, nu <= nu_lim = r_lim. Advanced
    by the exact exponential towards the current equilibrium nu_inf with
    tau_eff = tau_n (beta_e + 2) / (beta_e nu_ss + 2 load nu_inf) (= tau_n
    at full throttle and load, slower at low speed). Thrust x (nu /
    nu_ss)^2. Simple variant: at re-immersion (P1's latch clears) x =
    r_lim^2 - 1, x' = -x / tau_n, thrust x (1 + x).
    Ranges: tau_n ~ LogU[0.1, 1.5] s (draft), r_lim ~ U[1.02, 1.25] (draft:
    overshoot 0-0.56), beta_e ~ LogU[2, 10] (unloaded part-throttle
    overspeed <= 1.5 nu_ss) and c_idle ~ U[0.0225, 0.09] [assumption].
    Observed signal 'rpm' = nu (draft section 11: stored, not an input)."""

    code, calm_zero = "P2", True

    def draw_on(self, rng, ctx):
        u = rng.uniform
        return dict(simple=float(rng.random() < 0.5),
                    tau_n=float(_logu(rng, 0.1, 1.5)), r_lim=u(1.02, 1.25),
                    beta_e=float(_logu(rng, 2.0, 10.0)),
                    c_idle=u(0.0225, 0.09))

    def init_more(self, prm, B, rng):
        return dict(nu=np.full(B, np.nan), x=np.zeros(B))

    @staticmethod
    def nu_ss(prm, cn):
        return np.sqrt(prm["c_idle"] + (1.0 - prm["c_idle"]) * cn)

    @staticmethod
    def nu_inf(prm, nss, load):
        b = prm["beta_e"] * nss
        c0 = (1.0 + prm["beta_e"]) * nss * nss
        nu = 2.0 * c0 / (b + np.sqrt(b * b + 4.0 * load * c0))
        return np.minimum(nu, prm["r_lim"])

    def chain(self, prm, st, pc, q, dt):
        ctx = q["ctx"]
        nss = self.nu_ss(prm, _cn(pc, ctx))
        load = np.broadcast_to(np.asarray(pc.get("load", 1.0), float),
                               nss.shape)
        if prm["simple"]:
            reimm = pc.get("reimm")
            if reimm is not None:
                st["x"] = np.where(reimm, prm["r_lim"] ** 2 - 1.0, st["x"])
            fac = 1.0 + st["x"]
            nu = nss * np.sqrt(fac)
            st["x"] = st["x"] * math.exp(-dt / prm["tau_n"])
        else:
            ninf = self.nu_inf(prm, nss, load)
            st["nu"] = np.where(np.isnan(st["nu"]), ninf, st["nu"])
            nu = st["nu"].copy()
            fac = (nu / nss) ** 2
            tau = prm["tau_n"] * (prm["beta_e"] + 2.0) / (
                prm["beta_e"] * nss + 2.0 * load * ninf)
            st["nu"] = np.minimum(CB.exp_update(st["nu"], ninf, tau, dt),
                                  prm["r_lim"])
        pc["nu_fac"] = pc["nu_fac"] * fac
        pc["nu"] = nu
        return dict(rpm=nu)


# =================================================================== P3
@CB.register
class P3(PropItem):
    """Thrust-speed-throttle surface (always on; replaces T5's eta_T):
      T = t_max [g(c) min(phi(u), eta_P P_eng / (t_max u)) - r_idle(u)
                 (1 - g(c) / g_0)_+^q_i]
      g: w.p. 0.6 a Bernstein polynomial of order 3-5 with Dirichlet(2)
         increments (monotone, g(0) = 0, g(1) = 1), else g(c) = c
      log phi = log phi_id - k1 (s - 1) - k2 max(0, s - 1)^2, s = u_r / u_id
      r_idle = k_idle s^s_i (ram drag at idle; 0 at rest)
    The ram-drag term fades out at the throttle g_0 ~ U[0.05, 0.1]
    [assumption: the thrust fraction of idle-to-low rpm, idle / full rpm
    0.15-0.3]: above it the jet is faster than the inflow and the loss
    with speed is phi's alone. The draft's (1 - g)^q_i did not fade at
    cruise throttle (review 2026-09-30: at u_id and the low-fidelity
    steady throttle T_true / T_lofi had median 0.78, at 16 kn 0.62 with
    12% negative), so phi(u_id) = 1 +- 0.1 was not the identification
    point; now T(c, u_id) = t_max phi_id g(c) wherever g(c) >= g_0.
    Ranges (draft P3): phi_id ~ U[0.9, 1.1], k1 ~ U[0, 0.6], k2 ~ U[0, 0.3]
    (momentum theory at fixed power, jet velocity ratio 0.5-0.75, Bulten
    2006), k_idle ~ U[0, 0.4], s_i ~ U[1, 2], q_i ~ U[1, 3]. Section 11:
    P_eng ~ U[150, 250] kW; eta_P ~ U[0.55, 0.70] the overall propulsive
    efficiency of a jet at planing speed [assumption, unverified]; eta_P
    P_eng is floored at phi_id t_max u_id so the ceiling never binds at the
    identification speed. The power is also put on the context
    (ctx.P_eng, adjust_ctx) for other groups."""

    code = "P3"

    def draw_on(self, rng, ctx):
        u = rng.uniform
        if rng.random() < 0.6:
            n = int(rng.integers(3, 6))
            bw = np.concatenate([[0.0], np.cumsum(rng.dirichlet(
                2.0 * np.ones(n)))])
            bw[-1] = 1.0
        else:
            bw = np.array([0.0, 1.0])
        phi_id = u(0.9, 1.1)
        P_eng, eta = u(150e3, 250e3), u(0.55, 0.70)
        return dict(bw=bw, phi_id=phi_id, k1=u(0.0, 0.6), k2=u(0.0, 0.3),
                    k_idle=u(0.0, 0.4), s_i=u(1.0, 2.0), q_i=u(1.0, 3.0),
                    P_eng=P_eng, eta_P=eta,
                    etaP=max(eta * P_eng, phi_id * ctx.t_max * ctx.u_id),
                    g_0=u(0.05, 0.1))

    def adjust_ctx(self, prm, ctx):
        ctx.P_eng = float(prm["P_eng"])

    @staticmethod
    def g(prm, c):
        bw = np.asarray(prm["bw"], float)
        n = len(bw) - 1
        c = np.clip(np.asarray(c, float), 0.0, 1.0)
        return sum(bw[k] * math.comb(n, k) * c ** k * (1.0 - c) ** (n - k)
                   for k in range(n + 1))

    @staticmethod
    def thrust(prm, ctx, cn, u_r):
        s = np.maximum(np.asarray(u_r, float), 0.0) / ctx.u_id
        phi = prm["phi_id"] * np.exp(-prm["k1"] * (s - 1.0) - prm["k2"]
                                     * np.maximum(s - 1.0, 0.0) ** 2)
        cap = prm["etaP"] / (ctx.t_max * np.maximum(u_r, 1.0))
        g = P3.g(prm, cn)
        fade = np.maximum(1.0 - g / prm["g_0"], 0.0)
        return ctx.t_max * (g * np.minimum(phi, cap) - prm["k_idle"]
                            * s ** prm["s_i"] * fade ** prm["q_i"])

    def chain(self, prm, st, pc, q, dt):
        ctx = q["ctx"]
        pc["T"] = self.thrust(prm, ctx, _cn(pc, ctx), pc["u_r"])
        return {}


# =================================================================== P4
@CB.register
class P4(PropItem):
    """Cavitation thrust breakdown, the last thrust factor:
      T_lim(u) = T_lim0 rho [1 + eps (u_r / u_a)^2], u_a = sqrt(2 (p_atm -
      p_v) / rho_w) ~ 13.9 m/s (a constant); T_eff = T_lim S(T_pre / T_lim),
      S(x) = x / (1 + x^m)^(1/m) (positive thrust only). The cavitation
      degree cav = 1 - (1 + x^m)^(-1/m) scales P9's cavitation noise
      (applied here, before the saturation). rho = P10's loss factor.
    Ranges (draft P4): T_lim0 / t_max ~ LogU[0.6, 2], m ~ U[2, 8], eps ~
    U[0.5, 0.9] (Bulten 2006 example 0.62). Optional hysteresis omitted."""

    code = "P4"

    def draw_on(self, rng, ctx):
        return dict(Tl0=float(_logu(rng, 0.6, 2.0)) * ctx.t_max,
                    m=rng.uniform(2.0, 8.0), eps=rng.uniform(0.5, 0.9))

    @staticmethod
    def t_lim(prm, u_r, rho):
        return prm["Tl0"] * rho * (1.0 + prm["eps"] * (
            np.maximum(u_r, 0.0) / U_A) ** 2)

    def chain(self, prm, st, pc, q, dt):
        Tl = np.maximum(self.t_lim(prm, pc["u_r"], pc["rho"]), 1e-9)
        Tp = pc["T_pre"]
        m = prm["m"]

        def xm(T):
            return np.minimum(np.maximum(T, 0.0) / Tl, 1e6)
        x = xm(Tp)
        cav = 1.0 - np.exp(-np.log1p(x ** m) / m)
        if "eps9" in pc:
            n = pc["sig_c9"] * cav * pc["eps9"]
            Tp = Tp * (1.0 + PULSE_MAX * np.tanh(n / PULSE_MAX))
            x = xm(Tp)
        S = x * np.exp(-np.log1p(x ** m) / m)
        pc["T_eff"] = np.where(Tp > 0.0, Tl * S, Tp)
        pc["cav"] = cav
        return {}


# =================================================================== P5
@CB.register
class P5(PropItem):
    """Nozzle side force (always on; replaces T5's eta_S and c3):
      F_y = [k_js max(T_eff, 0) psi(u) (p rho)^(kappa - 1)
             + k_i t_max iota(c) (p rho)^kappa] h(delta)
      log psi = k_s (u_r / u_id - 1), clipped to [0.5, 2.5]
      iota = exp(-c / c_i)  (idle jet, 1 at zero throttle)
      h = sin(e_d d_s tanh(delta / d_s)) (1 + a sgn delta)
    (p, rho, kappa from P1 / P10: the same loss factors as the thrust, so
    F_y ~ p^kappa; the idle jet is the same jet, so it carries them too.)
    Ranges (draft P5): k_s ~ U[0, 1] (momentum theory: side force per net
    thrust x1.6-2.5 from 20 to 50 kn), k_i ~ U[0, 0.15], c_i ~ U[0.05, 0.3]
    (NTSB 1998: almost no steering off throttle), e_d ~ U[0.75, 1], d_s /
    rud_max ~ LogU[0.6, 3], a ~ U[-0.1, 0.1]."""

    code, calm_zero = "P5", True

    def draw_on(self, rng, ctx):
        u = rng.uniform
        return dict(k_s=u(0.0, 1.0), k_i=u(0.0, 0.15), c_i=u(0.05, 0.3),
                    e_d=u(0.75, 1.0),
                    d_s=float(_logu(rng, 0.6, 3.0)) * ctx.rud_max,
                    a=u(-0.1, 0.1))

    @staticmethod
    def psi(prm, ctx, u_r):
        s = np.maximum(u_r, 0.0) / ctx.u_id
        return np.clip(np.exp(prm["k_s"] * (s - 1.0)), 0.5, 2.5)

    @staticmethod
    def h(prm, d):
        return np.sin(prm["e_d"] * prm["d_s"] * np.tanh(d / prm["d_s"])) \
            * (1.0 + prm["a"] * np.sign(d))

    def chain(self, prm, st, pc, q, dt):
        ctx = q["ctx"]
        p = np.maximum(pc["p"], 1e-6)
        rho = np.maximum(pc["rho"], 1e-6)
        kp, kr = pc.get("kap_p", 1.0), pc.get("kap_rho", 1.0)
        lp, lr = np.log(p), np.log(rho)
        h = self.h(prm, pc["noz_eff"])
        jet = ctx.k_js * np.maximum(pc["T_eff"], 0.0) \
            * self.psi(prm, ctx, pc["u_r"]) \
            * np.exp((kp - 1.0) * lp + (kr - 1.0) * lr)
        idle = prm["k_i"] * ctx.t_max * np.exp(-_cn(pc, ctx) / prm["c_i"]) \
            * np.exp(kp * lp + kr * lr)
        pc["Fy"] = (jet + idle) * h
        return {}


# =================================================================== P6
@CB.register
class P6(PropItem):
    """Jet-hull interaction and the nozzle pitch bias. Sub-parts, given the
    item is on: (a) w.p. 0.5 (draft P6)
      F_z = k_z (T_eff - T_id) + k_th T_eff trim_rel,  dF_x = -t_d T_eff
      T_id = k_drag u_id^2 the low-fidelity steady thrust at the
      identification speed (only the difference: the calibrated attitude
      is not counted twice); trim_rel the pitch relative to the running
      attitude, bow up positive;
    (b) w.p. 0.6 (section 11) a fixed nozzle pitch bias d_tp: F_z += T_eff
      sin d_tp, dF_x += T_eff (cos d_tp - 1) (pitch moment ~ thrust).
    Both act at the jet point (module docstring). Ranges: k_z ~ U[-0.08,
    0.12], k_th ~ U[-1, 1], t_d ~ U[-0.05, 0.1] (draft); d_tp ~ U[-6, 6] deg
    [assumption, unverified PWC trim range; centred on the low-fidelity
    boat's own nozzle]."""

    code = "P6"

    def draw_on(self, rng, ctx):
        a, b = _parts(rng, (0.5, 0.6))
        u = rng.uniform
        return dict(a=a, b=b, k_z=u(-0.08, 0.12), k_th=u(-1.0, 1.0),
                    t_d=u(-0.05, 0.1), d_tp=u(-6.0, 6.0) * DEG)

    def chain(self, prm, st, pc, q, dt):
        ctx = q["ctx"]
        T = pc["T_eff"]
        Tid = ctx.k_drag * ctx.u_id ** 2
        Fz = prm["a"] * (prm["k_z"] * (T - Tid)
                         + prm["k_th"] * T * q["trim_rel"]) \
            + prm["b"] * T * math.sin(prm["d_tp"])
        dFx = -prm["a"] * prm["t_d"] * T \
            + prm["b"] * T * (math.cos(prm["d_tp"]) - 1.0)
        pc["Fz"] = pc["Fz"] + Fz
        pc["dF"] = pc["dF"] + np.stack([dFx, np.zeros_like(dFx)], 1)
        return {}


# =================================================================== P7
@CB.register
class P7(PropItem):
    """Steering-mechanism load effects beyond M10, with g = G / G_max the
    jet momentum flux read as the actuator throttle:
      rate (w.p. 0.3): an extra rate limit rate0 / (1 + lam_r g)
      backlash (w.p. 0.2): extra play of width db0 lam_b g
      compliance (w.p. 0.3): delta -= k_c g sin(delta)
    Ranges: lam_r, lam_b ~ U[0, 1], k_c ~ U[0, 0.1] rud_max (draft, general
    mechanical scales); rate0 ~ LogU[0.25, 2] / rs rad/s and db0 ~ U[0,
    0.03] rud_max are M10's nozzle rate and backlash ranges (lofi.ACT_
    FAMILY), since the servo's unloaded values are unknown here."""

    code, calm_zero = "P7", True

    def draw_on(self, rng, ctx):
        r, c, b = _parts(rng, (0.3, 0.3, 0.2))
        u = rng.uniform
        return dict(rate=r, comp=c, bl=b, lam_r=u(0.0, 1.0),
                    rate0=float(_logu(rng, 0.25, 2.0)) / ctx.rs,
                    k_c=u(0.0, 0.1) * ctx.rud_max, lam_b=u(0.0, 1.0),
                    db0=u(0.0, 0.03) * ctx.rud_max)

    def init_more(self, prm, B, rng):
        return dict(d_rl=np.full(B, np.nan), d_bl=np.full(B, np.nan))

    def chain(self, prm, st, pc, q, dt):
        g = _cn(pc, q["ctx"])
        d = pc["noz_eff"]
        if prm["rate"]:
            st["d_rl"] = np.where(np.isnan(st["d_rl"]), d, st["d_rl"])
            r = prm["rate0"] / (1.0 + prm["lam_r"] * g) * dt
            st["d_rl"] = st["d_rl"] + np.clip(d - st["d_rl"], -r, r)
            d = st["d_rl"].copy()
        if prm["bl"]:
            st["d_bl"] = np.where(np.isnan(st["d_bl"]), d, st["d_bl"])
            hw = 0.5 * prm["db0"] * prm["lam_b"] * g
            gap = d - st["d_bl"]
            st["d_bl"] = np.where(gap > hw, d - hw,
                                  np.where(gap < -hw, d + hw, st["d_bl"]))
            d = st["d_bl"].copy()
        if prm["comp"]:
            d = d - prm["k_c"] * g * np.sin(d)
        pc["noz_eff"] = d
        return {}


# =================================================================== P8
@CB.register
class P8(PropItem):
    """Reversing bucket, stuck-half-down fault only (the controller never
    reverses): from t_f the bucket stays at b; b' = (b - b0)_+ / (1 - b0);
    F_x x (1 - (1 + k_rev) b'), F_y x the same factor. Ranges (draft P8):
    k_rev ~ U[0.4, 0.8], b0 ~ U[0, 0.15], b ~ U[0.05, 0.3], t_f uniform in
    the episode."""

    code, clock = "P8", True

    def draw_on(self, rng, ctx):
        u = rng.uniform
        return dict(k_rev=u(0.4, 0.8), b0=u(0.0, 0.15), b=u(0.05, 0.3),
                    t_f=u(0.0, T_EP))

    @staticmethod
    def factor(prm, t):
        bp = max(prm["b"] - prm["b0"], 0.0) / (1.0 - prm["b0"])
        return 1.0 - (1.0 + prm["k_rev"]) * bp if t >= prm["t_f"] else 1.0

    def chain(self, prm, st, pc, q, dt):
        f = self.factor(prm, st["t"])
        if f != 1.0:
            ctx = q["ctx"]
            ax = 1.0 + ctx.kc * (np.cos(pc["noz_eff"]) - 1.0)
            Fx = pc["T_eff"] * ax if pc["Fx"] is None else pc["Fx"]
            pc["Fx"] = Fx * f
            pc["Fy"] = pc["Fy"] * f
        return {}


# =================================================================== P9
@CB.register
class P9(PropItem):
    """Thrust pulsation, a state-dependent multiplicative coloured noise:
      pulse = 0.9 tanh(sigma eps / 0.9), eps a unit OU process of bandwidth
      omega (exact step), sigma = s0 + s_v (1 - p) + s_r (1 - rho) + s_w
      |u_orb(x_i)| / V_in, V_in = max(u_r, u_lo / 2); the cavitation part
      s_c cav eps is applied by P4. The same factor reaches the side force
      (through T_eff).
    Ranges (draft P9): s0 ~ LogU[0.003, 0.03], s_v ~ U[0, 0.3], s_c ~ U[0,
    0.5], s_w ~ U[0, 3] (dT / T ~ -u_orb mu / ((1 - mu) V_in)), omega ~
    LogU[2, 60] rad/s; s_r ~ U[0, 0.3] the draft's 'sigma0 grows when
    clogged' (P10) [assumption, same range as s_v]."""

    code = "P9"

    def draw_on(self, rng, ctx):
        u = rng.uniform
        return dict(s0=float(_logu(rng, 0.003, 0.03)), s_v=u(0.0, 0.3),
                    s_c=u(0.0, 0.5), s_w=u(0.0, 3.0), s_r=u(0.0, 0.3),
                    om=float(_logu(rng, 2.0, 60.0)))

    def init_more(self, prm, B, rng):
        return dict(eps=rng.normal(0.0, 1.0, B))

    def sigma(self, prm, ctx, pc, q):
        uo = np.abs(_at_intake(ctx, q["uorb"][:, :, 1]))
        vin = np.maximum(pc["u_r"], 0.5 * ctx.u_lo)
        return prm["s0"] + prm["s_v"] * (1.0 - pc["p"]) \
            + prm["s_r"] * (1.0 - pc["rho"]) + prm["s_w"] * uo / vin

    def chain(self, prm, st, pc, q, dt):
        e = st["eps"].copy()
        st["eps"] = CB.ou_step(st["eps"], 1.0 / prm["om"], dt, st["rng"])
        n = self.sigma(prm, q["ctx"], pc, q) * e
        pulse = PULSE_MAX * np.tanh(n / PULSE_MAX)
        pc["pulse"] = (1.0 + pc["pulse"]) * (1.0 + pulse) - 1.0
        pc["eps9"], pc["sig_c9"] = e, prm["s_c"]
        return {}


# =================================================================== P10
@CB.register
class P10(PropItem):
    """Partial thrust loss (debris on the grate, impeller damage): rho = 1
    before t_f, then rho_f and w.p. 0.6 persistent, w.p. 0.2 recovering
    (exact exponential to 1 with tau_r), w.p. 0.2 intermittent (two-state
    Markov, dwell times d_on / d_off). Thrust x rho, side force x
    rho^kappa (P5), cavitation limit x rho (P4), noise up (P9).
    Ranges (draft P10): P_ON 0.075 (the draft's 0.05-0.1), rho_f ~ U[0.3,
    0.95], tau_r ~ LogU[2, 60] s, dwell ~ LogU[1, 30] s, t_f uniform in the
    episode; kappa ~ U[0.8, 1.5] as P1's."""

    code, clock = "P10", True

    def draw_on(self, rng, ctx):
        u = rng.uniform
        r = rng.random()
        mode = 0.0 if r < 0.6 else (1.0 if r < 0.8 else 2.0)
        return dict(mode=mode, rho_f=u(0.3, 0.95),
                    tau_r=float(_logu(rng, 2.0, 60.0)),
                    d_on=float(_logu(rng, 1.0, 30.0)),
                    d_off=float(_logu(rng, 1.0, 30.0)),
                    t_f=u(0.0, T_EP), kap=u(0.8, 1.5))

    def init_more(self, prm, B, rng):
        return dict(rho=np.ones(B), deg=np.zeros(B, bool), started=False)

    def chain(self, prm, st, pc, q, dt):
        B = q["B"]
        if st["t"] >= prm["t_f"] and not st["started"]:
            st["started"] = True
            st["rho"] = np.full(B, prm["rho_f"])
            st["deg"] = np.ones(B, bool)
        rho = st["rho"].copy()
        if st["started"]:
            if prm["mode"] == 1.0:
                st["rho"] = CB.exp_update(st["rho"], 1.0, prm["tau_r"], dt)
            elif prm["mode"] == 2.0:
                rate = np.where(st["deg"], 1.0 / prm["d_on"],
                                1.0 / prm["d_off"])
                flip = CB.poisson_hit(rate, dt, st["rng"], B)
                st["deg"] = st["deg"] ^ flip
                st["rho"] = np.where(st["deg"], prm["rho_f"], 1.0)
        pc["rho"] = pc["rho"] * rho
        pc["kap_rho"] = prm["kap"]
        return {}


# =================================================================== P11
@CB.register
class P11(PropItem):
    """Nozzle faults from t_f, one of: stuck at its angle (0.4), hard over
    +-rud_max (0.1), reduced authority rho_d delta + d_off (0.3), slow drift
    delta + a random walk (0.2); always within +-rud_max.
    Ranges (draft P11): rho_d ~ U[0.3, 0.9], drift LogU[0.1, 2] deg /
    sqrt(min), t_f uniform in the episode; d_off ~ U[-0.05, 0.05] rud_max
    and the hard-over side (+-1) [assumption]."""

    code, clock = "P11", True

    def draw_on(self, rng, ctx):
        u = rng.uniform
        r = rng.random()
        mode = 0.0 if r < 0.4 else (1.0 if r < 0.5 else (
            2.0 if r < 0.8 else 3.0))
        return dict(mode=mode, side=1.0 if rng.random() < 0.5 else -1.0,
                    rho_d=u(0.3, 0.9), d_off=u(-0.05, 0.05) * ctx.rud_max,
                    s_dr=float(_logu(rng, 0.1, 2.0)) * DEG / math.sqrt(60.0),
                    t_f=u(0.0, T_EP))

    def init_more(self, prm, B, rng):
        return dict(d_st=np.full(B, np.nan), walk=np.zeros(B))

    def chain(self, prm, st, pc, q, dt):
        if st["t"] < prm["t_f"]:
            return {}
        ctx = q["ctx"]
        d = pc["noz_eff"]
        md = prm["mode"]
        if md == 0.0:
            st["d_st"] = np.where(np.isnan(st["d_st"]), d, st["d_st"])
            d = st["d_st"].copy()
        elif md == 1.0:
            d = np.full_like(d, prm["side"] * ctx.rud_max)
        elif md == 2.0:
            d = prm["rho_d"] * d + prm["d_off"]
        else:
            d = d + st["walk"]
            st["walk"] = st["walk"] + prm["s_dr"] * math.sqrt(dt) \
                * st["rng"].normal(0.0, 1.0, len(d))
        pc["noz_eff"] = np.clip(d, -ctx.rud_max, ctx.rud_max)
        return {}


# =================================================================== P12
@CB.register
class P12(PropItem):
    """Factory engine-controller rules (section 11: a controller w.p. 0.8,
    then each rule w.p. its draft value; the item is on when at least one
    rule is): steering assist (0.2): c < c_th and |delta_cmd| > d_th for
    t_d -> c = max(c, c_a), held t_h; speed limit (0.05): c x clip((u_cap -
    u) / du, 0, 1); limp mode (0.03): from t_f c <= c_cap; overheat (0.02):
    c <= 1 - k_h clip((T - T_on) / (1 - T_on), 0, 1), T a first-order
    temperature driven by the throttle (exact). Caps apply last. A changed
    command goes through the true actuator copy (cat_base M10 step).
    Ranges (draft P12, all unverified): c_cap ~ U[0.3, 0.7], c_th ~ U[0.03,
    0.2], d_th / rud_max ~ U[0.5, 0.95], t_d ~ U[0, 0.5] s, c_a ~ U[0.1,
    0.3], t_h ~ U[0.5, 3] s, u_cap ~ U[0.6, 1] u_max, du ~ U[0.5, 2] m/s;
    overheat tau_h ~ LogU[60, 600] s, T_on ~ U[0.5, 0.9], k_h ~ U[0.3, 0.7],
    initial T ~ U[0.2, 1] [assumption]. The rev limiter and the airborne
    rev protection are P2's nu_lim."""

    code, clock = "P12", True
    RULES = (("assist", 0.2), ("slim", 0.05), ("limp", 0.03),
             ("heat", 0.02))

    def draw_on(self, rng, ctx):
        f = _parts(rng, [p for _, p in self.RULES])
        u = rng.uniform
        d = {nm: v for (nm, _), v in zip(self.RULES, f)}
        d.update(c_th=u(0.03, 0.2), d_th=u(0.5, 0.95) * ctx.rud_max,
                 t_d=u(0.0, 0.5), c_a=u(0.1, 0.3), t_h=u(0.5, 3.0),
                 u_cap=u(0.6, 1.0) * ctx.u_max, du=u(0.5, 2.0),
                 c_cap=u(0.3, 0.7), t_f=u(0.0, T_EP),
                 tau_h=float(_logu(rng, 60.0, 600.0)), T_on=u(0.5, 0.9),
                 k_h=u(0.3, 0.7))
        return d

    def init_more(self, prm, B, rng):
        return dict(tc=np.zeros(B), th=np.zeros(B),
                    temp=rng.uniform(0.2, 1.0, B), engaged=False)

    def chain(self, prm, st, pc, q, dt):
        ctx = q["ctx"]
        c0 = pc["c_thr"] / ctx.t_max
        c = c0.copy()
        if prm["assist"]:
            cond = (c < prm["c_th"]) & (np.abs(pc["c_noz"]) > prm["d_th"])
            st["tc"] = np.where(cond, st["tc"] + dt, 0.0)
            st["th"] = np.maximum(st["th"] - dt, 0.0)
            st["th"] = np.where(cond & (st["tc"] >= prm["t_d"]), prm["t_h"],
                                st["th"])
            c = np.where(st["th"] > 0.0, np.maximum(c, prm["c_a"]), c)
        if prm["slim"]:
            c = c * np.clip((prm["u_cap"] - q["sr"][:, 2]) / prm["du"], 0.0,
                            1.0)
        if prm["limp"] and st["t"] >= prm["t_f"]:
            c = np.minimum(c, prm["c_cap"])
        if prm["heat"]:
            st["temp"] = CB.exp_update(st["temp"], np.clip(c, 0.0, 1.0),
                                       prm["tau_h"], dt)
            x = np.clip((st["temp"] - prm["T_on"]) / (1.0 - prm["T_on"]),
                        0.0, 1.0)
            c = np.minimum(c, 1.0 - prm["k_h"] * x)
        if st["engaged"] or np.any(c != c0):
            st["engaged"] = True
            pc["c_thr"] = c * ctx.t_max
            pc["cmd_changed"] = True
        return {}


# =================================================================== P13
@CB.register
class P13(PropItem):
    """Slam-entrained air drawn into the pump (drawn only with P1): each
    low-fidelity slam k (q['slam'], entry speed v_k) starts a causal pulse
    alpha_k(t) = A_k g((t - t_k - D_k) / w_p), g(x) = x e^(1 - x) (x > 0,
    peak 1), A_k = cA (v_k / (3 v_slam))^gam, D_k = L_p / max(u_r, 1);
    thrust factor p13 follows 1 - tanh(sum alpha) through P1's asymmetric
    lag (tau_loss falling, tau_rec rising); x p (thrust, side force), x
    p13^m pump torque (P2). At most 8 pending pulses per row.
    Ranges (draft P13, all unverified): L_p ~ U[0.5, 1] (x_bow - x_i), w_p
    ~ LogU[0.05, 0.5] s, cA ~ U[0, 0.5], gam ~ U[0.5, 1.5]; the reference
    entry speed 3 v_slam (Froude-scaled like v_slam) [assumption]."""

    code, calm_zero = "P13", True
    K = 8

    def draw_on(self, rng, ctx):
        u = rng.uniform
        return dict(L_p=u(0.5, 1.0) * (ctx.x_st[-1] - ctx.intake[0]),
                    w_p=float(_logu(rng, 0.05, 0.5)), cA=u(0.0, 0.5),
                    gam=u(0.5, 1.5))

    def init_more(self, prm, B, rng):
        return dict(t=0.0, t0=np.full((B, self.K), -1e9),
                    amp=np.zeros((B, self.K)), p13=np.ones(B))

    def chain(self, prm, st, pc, q, dt):
        ctx = q["ctx"]
        slam = q.get("slam")
        if slam is not None and np.any(slam):
            rows = np.flatnonzero(slam)
            j = np.argmin(st["t0"][rows], 1)               # oldest slot
            v = np.asarray(q["slam_v"], float)[rows]
            D = prm["L_p"] / np.maximum(pc["u_r"][rows], 1.0)
            st["t0"][rows, j] = st["t"] + D
            st["amp"][rows, j] = prm["cA"] * (np.maximum(v, 0.0) / (
                3.0 * ctx.v_slam)) ** prm["gam"]
        x = (st["t"] - st["t0"]) / prm["w_p"]
        g = np.where(x > 0.0, np.maximum(x, 0.0) * np.exp(1.0 - np.maximum(
            x, 0.0)), 0.0)
        alpha = (st["amp"] * g).sum(1)
        tl, tr = pc.get("p1_tau", (0.1, 1.0))
        st["p13"] = CB.exp_update_asym(st["p13"], 1.0 - np.tanh(alpha), tr,
                                       tl, dt)
        f = st["p13"]
        pc["p13"] = f                 # P13's own air factor (diagnostics)
        pc["p"] = pc["p"] * f
        pc["load"] = pc.get("load", 1.0) * f ** pc.get("m_torque", 0.8)
        st["t"] += dt
        return {}


# =================================================================== P14
@CB.register
class P14(PropItem):
    """Small general residual of the propulsion channel, only through the
    jet point: dF = t_max eps c tanh(W phi(X)), phi(X) = sqrt(2 / J) cos(
    Omega X + b) random Fourier features (J = 24, length scale ell) of X =
    (c, u_r / u_id, delta / rud_max, h_in / h_run, nu); |dF| <= eps t_max c
    (zero at zero throttle). Ranges: eps ~ LogU[0.005, 0.1] (draft); ell ~
    LogU[0.3, 3] in the inputs' own units [assumption]."""

    code = "P14"
    J = 24

    def draw_on(self, rng, ctx):
        ell = float(_logu(rng, 0.3, 3.0))
        return dict(eps=float(_logu(rng, 0.005, 0.1)), ell=ell,
                    Om=rng.normal(0.0, 1.0, (self.J, 5)) / ell,
                    b=rng.uniform(0.0, 2 * np.pi, self.J),
                    W=rng.normal(0.0, 1.0, (2, self.J)))

    def residual(self, prm, ctx, pc, q):
        cn = _cn(pc, ctx)
        nu = pc.get("nu", np.sqrt(cn))
        X = np.stack([cn, pc["u_r"] / ctx.u_id, pc["noz_eff"] / ctx.rud_max,
                      q["h_in"] / ctx.h_run, nu], 1)
        f = math.sqrt(2.0 / self.J) * np.cos(X @ np.asarray(prm["Om"]).T
                                             + np.asarray(prm["b"]))
        return ctx.t_max * prm["eps"] * cn[:, None] * np.tanh(
            f @ np.asarray(prm["W"]).T)

    def chain(self, prm, st, pc, q, dt):
        pc["dF"] = pc["dF"] + self.residual(prm, q["ctx"], pc, q)
        return {}


# ================================================================ checks
# Construction checks, run by studies/test_cat_items.py (I5) through the
# cat_base.check registry. Short, deterministic, a few MB.
PCODES = tuple(f"P{i}" for i in range(1, 15))
DT = CB.DT_SUB


def _q(ctx, B=1, u=0.0, thr=0.0, noz=0.0, dz=0.0, sea=None):
    sr = np.zeros((B, 10))
    sr[:, 2], sr[:, 3], sr[:, 5] = u, ctx.z0 + dz, ctx.th0
    s = CB.CatSea(None, ctx).sample(sr, 0.0) if sea is None else sea
    thr = np.broadcast_to(np.asarray(thr, float), (B,)).copy()
    noz = np.broadcast_to(np.asarray(noz, float), (B,)).copy()
    return CB.substep_quantities(ctx, sr, thr, noz, s,
                                 cmd=np.stack([thr, noz], 1))


def zero_at_rest(item, ctx):
    """Draft 7.1: at rest (u = 0, zero throttle, nozzle centred, calm) every
    P item gives exactly zero, before and after its fault time (P6 with its
    part (a) off: k_z (T - T_id) is not zero at T = 0 by the draft's own
    form)."""
    rng = np.random.default_rng(21)
    worst = 0.0
    for _ in range(20):
        prm = item.draw_on(rng, ctx)
        if item.code == "P6":
            prm = dict(prm, a=0.0, b=1.0)
        for t0 in (0.0, T_EP + 1.0):
            st = item.init_state(prm, 4, rng=np.random.default_rng(1))
            if "t" in st:
                st["t"] = t0
            for _k in range(5):
                o = item.step(prm, st, _q(ctx, 4), DT)
                worst = max(worst, float(np.abs(o.acc).max()))
    return worst == 0.0, f"max |a| at rest {worst:.1e}"


for _c in PCODES:
    CB.check(_c)(zero_at_rest)


@CB.check("P1")
def suction_latch(item, ctx):
    rng = np.random.default_rng(22)
    bad = []
    for _ in range(20):
        prm = item.draw_on(rng, ctx)
        st = item.init_state(prm, 1)
        ps = []
        for k in range(200):
            q = _q(ctx, 1, u=ctx.u_id, dz=1.0 if 25 <= k < 35 else 0.0)
            item.chain(prm, st, new_pc(q), q, DT)
            ps.append(float(st["p"][0]))
        ps = np.array(ps)
        lo = 1.0 - prm["A"] - 1e-12
        if ps.min() < lo or ps.max() > 1.0 + 1e-12 or ps[24] != 1.0:
            bad.append("bounds / calm")
        if np.any(np.diff(ps[24:35]) > 1e-15):
            bad.append("rises while out of the water")
        kd = int(np.ceil(prm["t_dwell"] / DT - 1e-9))   # calm steps held
        held = ps[34:35 + max(kd - 1, 0)]
        if np.any(np.abs(np.diff(held)) > 1e-15):
            bad.append("recovers before the dwell")
        if np.any(np.diff(ps[35:]) < -1e-15) or st["lost"][0]:
            bad.append("recovery not monotone / latch not cleared")
    return not bad, (", ".join(sorted(set(bad))) or
                     "p in [1-A, 1], 1 in calm running, falls out of the "
                     "water, held for t_dwell, then rises")


@CB.check("P2")
def rpm_limits(item, ctx):
    rng = np.random.default_rng(23)
    bad = []
    for _ in range(10):
        prm = dict(item.draw_on(rng, ctx), simple=0.0)
        st = item.init_state(prm, 1)
        q = _q(ctx, 1, u=ctx.u_id, thr=ctx.t_max)

        def run(load, n):
            f, nu = [], []
            for _k in range(n):
                pc = new_pc(q)
                pc["load"] = np.array([load])
                item.chain(prm, st, pc, q, DT)
                f.append(float(pc["nu_fac"][0]))
                nu.append(float(pc["nu"][0]))
            return np.array(f), np.array(nu)
        f1, _ = run(1.0, 5)
        n0 = int(40 * prm["tau_n"] / DT) + 5
        _, nu0 = run(0.0, n0)
        want = min(prm["r_lim"], (1 + prm["beta_e"]) / prm["beta_e"])
        f2, _ = run(1.0, n0)
        if np.abs(f1 - 1.0).max() > 1e-12:
            bad.append("not steady at full load")
        if nu0.max() > prm["r_lim"] + 1e-12 or abs(nu0[-1] - want) > 1e-6:
            bad.append("unloaded speed")
        if not (1.0 < f2[0] <= prm["r_lim"] ** 2 + 1e-12) or \
                abs(f2[-1] - 1.0) > 1e-3:
            bad.append("re-load overshoot")
        prs = dict(prm, simple=1.0)
        st = item.init_state(prs, 1)
        pc = new_pc(q)
        pc["reimm"] = np.array([True])
        item.chain(prs, st, pc, q, DT)
        pc2 = new_pc(q)
        item.chain(prs, st, pc2, q, DT)
        x0 = prm["r_lim"] ** 2 - 1.0
        if abs(pc["nu_fac"][0] - 1 - x0) > 1e-12 or abs(
                pc2["nu_fac"][0] - 1 - x0 * math.exp(-DT / prm["tau_n"])) \
                > 1e-12:
            bad.append("simple variant")
    return not bad, (", ".join(sorted(set(bad))) or
                     "steady at full load, unloaded speed = min(nu_lim, "
                     "(1+beta)/beta), overshoot <= nu_lim^2 then decays; "
                     "simple variant exact")


@CB.check("P3")
def surface_monotone(item, ctx):
    rng = np.random.default_rng(24)
    c = np.linspace(0, 1, 21)[:, None]
    u = np.linspace(0, 30, 61)[None, :]
    bad, n_cross, ratios = [], 0, []
    for _ in range(60):
        prm = item.draw_on(rng, ctx)
        T = item.thrust(prm, ctx, c, u)
        if np.diff(T, axis=0).min() < -1e-9 or np.diff(T, axis=1).max() > \
                1e-9:
            bad.append("not monotone")
        if T[0].max() > 1e-12 or T[:, 0].min() < -1e-12 or T[0, 0] != 0.0:
            bad.append("signs at zero throttle / rest")
        g = item.g(prm, np.array([0.0, 1.0]))
        if abs(g[0]) > 1e-15 or abs(g[1] - 1) > 1e-12:
            bad.append("g ends")
        Tid = item.thrust(prm, ctx, 1.0, ctx.u_id)
        if abs(Tid - prm["phi_id"] * ctx.t_max) > 1e-9 * ctx.t_max:
            bad.append("power ceiling binds at u_id")
        cr = np.diff(np.diff(item.thrust(dict(prm, k_idle=0.0), ctx, c, u),
                             axis=0), axis=1)
        if cr.max() > 1e-9:
            bad.append("cross derivative > 0 without ram drag")
        n_cross += np.diff(np.diff(T, axis=0), axis=1).max() > 1e-9
        # centred identification point: no ram drag where g(c) >= g_0
        cc = np.linspace(0.0, 1.0, 101)
        gg = item.g(prm, cc)
        hi = gg >= prm["g_0"]
        Tc = item.thrust(prm, ctx, cc, ctx.u_id)
        err = np.abs(Tc[hi] - ctx.t_max * prm["phi_id"] * gg[hi])
        if err.size and err.max() > 1e-9 * ctx.t_max:
            bad.append("ram drag at cruise throttle")
        c_lo = ctx.k_drag * ctx.u_id ** 2 / ctx.t_max
        ratios.append(float(item.thrust(prm, ctx, c_lo, ctx.u_id))
                      / (c_lo * ctx.t_max))
    med = float(np.median(ratios))
    if not 0.9 <= med <= 1.1:
        bad.append(f"T_true / T_lofi at u_id median {med:.2f}")
    return not bad, (", ".join(sorted(set(bad))) or
                     "T non-decreasing in c, non-increasing in u, 0 at rest, "
                     "phi(u_id) t_max at full throttle; d2T/dcdu <= 0 "
                     f"without the ram term (with it > 0 in {n_cross}/60 "
                     "draws, as stated); at u_id and the low-fidelity steady "
                     f"throttle T_true / T_lofi median {med:.3f}, quantiles "
                     f"5/95% {np.quantile(ratios, 0.05):.2f}/"
                     f"{np.quantile(ratios, 0.95):.2f}")


@CB.check("P4")
def cavitation_cap(item, ctx):
    rng = np.random.default_rng(25)
    Tp = np.linspace(0, 3 * ctx.t_max, 61)[:, None] * np.ones((1, 31))
    u = np.ones((61, 1)) * np.linspace(0, 30, 31)[None, :]
    bad = []
    for _ in range(40):
        prm = item.draw_on(rng, ctx)
        q = _q(ctx, Tp.size)
        pc = new_pc(q)
        pc["u_r"], pc["T_pre"] = u.ravel(), Tp.ravel()
        item.chain(prm, {}, pc, q, DT)
        Te = pc["T_eff"].reshape(Tp.shape)
        Tl = item.t_lim(prm, u, 1.0)
        if (Te > Tl + 1e-9).any() or (Te > Tp + 1e-9).any() or \
                np.diff(Te, axis=0).min() < -1e-9 or \
                np.diff(Tl, axis=1).min() < 0:
            bad.append("cap")
    return not bad, (", ".join(set(bad)) or "T_eff <= min(T_pre, T_lim(u)), "
                     "monotone in T_pre, T_lim rises with speed")


@CB.check("P5")
def side_force_factors(item, ctx):
    rng = np.random.default_rng(26)
    bad = []
    for _ in range(20):
        prm = item.draw_on(rng, ctx)
        q = _q(ctx, 3, u=ctx.u_id, thr=0.3 * ctx.t_max,
               noz=np.array([0.0, 0.2, -0.3]))
        f = []
        for p, rho, s in ((1.0, 1.0, 1.0), (0.5, 0.8, 0.4)):
            pc = new_pc(q)
            pc["p"], pc["rho"] = np.full(3, p), np.full(3, rho)
            pc["kap_p"], pc["kap_rho"] = 1.2, 0.9
            pc["T_eff"] = np.full(3, 3000.0 * s)
            item.chain(prm, {}, pc, q, DT)
            f.append(pc["Fy"])
        want = 0.5 ** 1.2 * 0.8 ** 0.9
        if f[0][0] != 0.0 or np.abs(f[1][1:] - want * f[0][1:]).max() > \
                1e-9 * np.abs(f[0]).max():
            bad.append("loss factors")
        ps = item.psi(prm, ctx, np.linspace(0, 40, 81))
        if ps.min() < 0.5 or ps.max() > 2.5 or np.diff(ps).min() < 0:
            bad.append("psi")
    return not bad, (", ".join(set(bad)) or "F_y ~ p^kappa rho^kappa (thrust "
                     "and idle jet), 0 at delta = 0, psi in [0.5, 2.5] "
                     "non-decreasing")


@CB.check("P6")
def calibrated_zero(item, ctx):
    rng = np.random.default_rng(27)
    Tid = ctx.k_drag * ctx.u_id ** 2
    q = _q(ctx, 2, u=ctx.u_id, thr=Tid)
    worst, dev = 0.0, 0.0
    for _ in range(20):
        prm = item.draw_on(rng, ctx)
        pc = new_pc(q)
        pc["T_eff"] = np.full(2, Tid)
        item.chain(dict(prm, a=1.0, b=0.0), {}, pc, q, DT)
        worst = max(worst, float(np.abs(pc["Fz"]).max()))
        pc = new_pc(q)
        pc["T_eff"] = np.array([1000.0, 2000.0])
        item.chain(dict(prm, a=0.0, b=1.0), {}, pc, q, DT)
        dev = max(dev, float(np.abs(pc["Fz"] - pc["T_eff"] * math.sin(
            prm["d_tp"])).max()))
    ok = worst < 1e-9 and dev < 1e-9
    return ok, (f"(a) F_z at T_id and running trim {worst:.1e} N; (b) F_z - "
                f"T sin(d_tp) {dev:.1e} N")


@CB.check("P7")
def steering_bounds(item, ctx):
    rng = np.random.default_rng(28)
    bad = []
    for _ in range(20):
        prm = dict(item.draw_on(rng, ctx), rate=1.0, comp=1.0, bl=1.0)
        st = item.init_state(prm, 1)
        ds = []
        for k in range(60):
            cmd = 0.0 if k < 5 else (ctx.rud_max if k < 35 else -0.1)
            q = _q(ctx, 1, u=ctx.u_id, thr=ctx.t_max, noz=cmd)
            pc = new_pc(q)
            item.chain(dict(prm, comp=0.0, bl=0.0), st, pc, q, DT)
            ds.append(float(pc["noz_eff"][0]))
        r = prm["rate0"] / (1 + prm["lam_r"]) * DT
        if np.abs(np.diff(ds)).max() > r + 1e-12:
            bad.append("rate")
        st = item.init_state(prm, 1)
        hw = 0.5 * prm["db0"] * prm["lam_b"]
        for d in np.linspace(-ctx.rud_max, ctx.rud_max, 41):
            q = _q(ctx, 1, u=ctx.u_id, thr=ctx.t_max, noz=d)
            pc = new_pc(q)
            item.chain(dict(prm, rate=0.0, comp=0.0), st, pc, q, DT)
            if abs(pc["noz_eff"][0] - d) > hw + 1e-12:
                bad.append("backlash")
            pc = new_pc(q)
            item.chain(dict(prm, rate=0.0, bl=0.0), st, pc, q, DT)
            o = pc["noz_eff"][0]
            if abs(o) > abs(d) + 1e-15 or o * d < 0:
                bad.append("compliance")
    return not bad, (", ".join(sorted(set(bad))) or
                     "rate <= rate0 / (1 + lam_r g), play <= db0 lam_b g / 2,"
                     " compliance only reduces |delta|")


@CB.check("P8")
def stuck_after_tf(item, ctx):
    rng = np.random.default_rng(29)
    bad = []
    for _ in range(20):
        prm = item.draw_on(rng, ctx)
        q = _q(ctx, 1, u=ctx.u_id, thr=0.5 * ctx.t_max, noz=0.1)
        lo = 1.0 - (1.0 + prm["k_rev"]) * 0.3
        for t0 in (0.0, T_EP + 1.0):
            pc = new_pc(q)
            pc["T_eff"], pc["Fy"] = pc["thr_act"].copy(), np.array([500.0])
            item.chain(prm, dict(t=t0), pc, q, DT)
            f = pc["Fy"][0] / 500.0
            changes = t0 >= prm["t_f"] and item.factor(prm, t0) != 1.0
            if changes == (pc["Fx"] is None) or not (lo - 1e-12 <= f <= 1.0):
                bad.append("timing / factor")
    return not bad, (", ".join(set(bad)) or "no change before t_f, then F_x "
                     "and F_y x (1 - (1 + k_rev) b') within [1 - 1.8 x 0.3, "
                     "1]")


@CB.check("P9")
def pulse_bounds(item, ctx):
    rng = np.random.default_rng(30)
    prm = item.draw_on(rng, ctx)
    B = 64
    st = item.init_state(prm, B, rng=np.random.default_rng(3))
    q = _q(ctx, B, u=ctx.u_id, thr=0.5 * ctx.t_max)
    E, pm = [], 0.0
    for _ in range(2000):
        pc = new_pc(q)
        item.chain(prm, st, pc, q, DT)
        E.append(pc["eps9"])
        pm = max(pm, float(np.abs(pc["pulse"]).max()))
    v = float(np.var(np.array(E)))
    pc = new_pc(q)
    s1 = item.sigma(prm, ctx, pc, q)
    pc["p"] = np.full(B, 0.3)
    s2 = item.sigma(prm, ctx, pc, q)
    ok = pm < PULSE_MAX and abs(v - 1.0) < 0.1 and np.all(s2 >= s1)
    return ok, (f"max |pulse| {pm:.3f} (< 0.9), OU variance {v:.3f} (1), "
                "sigma grows as p falls")


@CB.check("P10")
def fault_timing(item, ctx):
    rng = np.random.default_rng(31)
    bad = []
    q = _q(ctx, 2, u=ctx.u_id, thr=0.5 * ctx.t_max)
    for mode in (0.0, 1.0, 2.0):
        prm = dict(item.draw_on(rng, ctx), mode=mode, t_f=1.0)
        st = item.init_state(prm, 2)
        R = []
        for _k in range(300):
            pc = new_pc(q)
            item.step(prm, st, dict(q, prop=pc), DT)
            R.append(pc["rho"].copy())
        R = np.array(R)
        pre, post = R[:25], R[26:]
        if np.any(pre != 1.0) or post.min() < prm["rho_f"] - 1e-12 or \
                post.max() > 1.0:
            bad.append(f"mode {mode:.0f} bounds")
        if mode == 0.0 and np.any(post != prm["rho_f"]):
            bad.append("persist")
        if mode == 1.0 and np.any(np.diff(post, axis=0) < 0):
            bad.append("recover")
        if mode == 2.0 and not np.all(np.isin(post, [prm["rho_f"], 1.0])):
            bad.append("intermittent")
    return not bad, (", ".join(bad) or "rho = 1 before t_f; persistent, "
                     "recovering and intermittent modes as drawn")


@CB.check("P11")
def nozzle_fault(item, ctx):
    rng = np.random.default_rng(32)
    bad = []
    for mode in (0.0, 1.0, 2.0, 3.0):
        prm = dict(item.draw_on(rng, ctx), mode=mode, t_f=1.0)
        st = item.init_state(prm, 1, rng=np.random.default_rng(4))
        D, C = [], []
        for k in range(200):
            cmd = 0.3 * math.sin(0.1 * k)
            q = _q(ctx, 1, u=ctx.u_id, thr=0.5 * ctx.t_max, noz=cmd)
            pc = new_pc(q)
            item.step(prm, st, dict(q, prop=pc), DT)
            D.append(pc["noz_eff"][0])
            C.append(cmd)
        D, C = np.array(D), np.array(C)
        if np.any(D[:25] != C[:25]) or np.abs(D).max() > ctx.rud_max:
            bad.append(f"mode {mode:.0f}")
        if mode == 0.0 and np.ptp(D[26:]) != 0.0:
            bad.append("stuck moves")
    return not bad, (", ".join(bad) or "pass-through before t_f, within "
                     "+-rud_max after, stuck holds")


@CB.check("P12")
def controller_rules(item, ctx):
    rng = np.random.default_rng(33)
    prm = dict(item.draw_on(rng, ctx), assist=1.0, slim=1.0, limp=1.0,
               heat=0.0, t_f=1.0)
    st = item.init_state(prm, 3)
    thr = np.array([0.0, ctx.t_max, ctx.t_max])
    noz = np.array([ctx.rud_max, 0.0, 0.0])
    sr_u = np.array([5.0, prm["u_cap"] + 0.01, 5.0])
    out, flag = [], []
    for _k in range(60):
        q = _q(ctx, 3, thr=thr, noz=noz)
        q["sr"][:, 2] = sr_u
        pc = new_pc(q)
        item.step(prm, st, dict(q, prop=pc), DT)
        out.append(pc["c_thr"] / ctx.t_max)
        flag.append(pc["cmd_changed"])
    out = np.array(out)
    ok = (out[-1, 0] >= min(prm["c_a"], prm["c_cap"]) - 1e-12
          and out[-1, 1] == 0.0 and out[-1, 2] <= prm["c_cap"] + 1e-12
          and out[0, 2] == 1.0 and all(flag))
    return ok, (f"assist raises idle to {out[-1, 0]:.2f} (c_a "
                f"{prm['c_a']:.2f}), speed limit 0 above u_cap, limp cap "
                f"{prm['c_cap']:.2f} only after t_f, command flagged")


@CB.check("P13")
def slam_pulse_causal(item, ctx):
    rng = np.random.default_rng(34)
    bad = []
    for _ in range(10):
        prm = dict(item.draw_on(rng, ctx), cA=0.5)
        st = item.init_state(prm, 1)
        F = []
        for k in range(400):
            q = _q(ctx, 1, u=ctx.u_id, thr=0.5 * ctx.t_max)
            q["slam"], q["slam_v"] = np.array([k == 0]), np.array([2.0])
            pc = new_pc(q)
            item.chain(prm, st, pc, q, DT)
            F.append(pc["p"][0])
        F = np.array(F)
        n0 = int(np.floor(prm["L_p"] / ctx.u_id / DT))
        if np.any(F[:n0 + 1] != 1.0) or F.min() <= 0 or F.min() >= 1 or \
                F[-1] < 1 - 1e-3:
            bad.append("pulse")
    return not bad, (", ".join(set(bad)) or "no loss before the transport "
                     "delay, 0 < factor < 1 after, back to 1")


@CB.check("P14")
def residual_bound(item, ctx):
    rng = np.random.default_rng(35)
    worst = 0.0
    B = 200
    for _ in range(20):
        prm = item.draw_on(rng, ctx)
        thr = rng.uniform(0, ctx.t_max, B)
        thr[:10] = 0.0
        q = _q(ctx, B, u=rng.uniform(0, 30), thr=thr,
               noz=rng.uniform(-ctx.rud_max, ctx.rud_max, B),
               dz=rng.uniform(-0.5, 0.5))
        dF = item.residual(prm, ctx, new_pc(q), q)
        lim = prm["eps"] * ctx.t_max * np.clip(thr / ctx.t_max, 0, 1)
        worst = max(worst, float((np.abs(dF).max(1) - lim).max()))
        if np.abs(dF[:10]).max() != 0.0:
            worst = max(worst, 1.0)
    return worst <= 1e-9, f"max(|dF| - eps t_max c) {worst:.1e} N; 0 at c = 0"


@CB.check("P3")
def whole_chain(item, ctx):
    """All 14 items forced on in one CatDraw (no layers 1 / 3 forces): the
    chain is exactly 0 at rest (P6 part (a) off), a 10 s closed loop in a
    sea with a steering sweep stays finite, rpm is reported."""
    from control.reduced import ReducedModel
    from learn.meta import operators_rb as R
    force = {c: True for c in PCODES}
    p6 = CB.REGISTRY["P6"].draw_on(np.random.default_rng(5), ctx)
    force["P6"] = dict(p6, a=0.0, b=1.0)
    d = CB.CatDraw(3, tier="none", layer1=False, items=PCODES,
                   force_items=force, sparse=False)
    B = 4
    st = d.new_state(B)
    q0 = _q(d.ctx, B)
    out = d.substep(st, q0["sr"], q0["thr"], q0["noz"],
                    CB.CatSea(None, d.ctx).sample(q0["sr"], 0.0),
                    cmd=np.zeros((B, 2)), parts=True)
    rest = float(np.abs(out["PARTS"]["prop"]).max())
    rng = np.random.default_rng(11)
    sea = R.sea_state(R.sea_dict(rng), 11)
    rs = R.RowSeas([sea] * B, d.ctx.x_st, d.ctx.y_off)
    cs = CB.CatSea(rs, d.ctx)
    red = ReducedModel(d.ctx.p)
    rv = list(R.RED_VEL)
    sr = np.zeros((B, 10))
    sr[:, 2] = [6.0, 10.0, 15.0, 20.0]
    sr[:, 3], sr[:, 5] = d.ctx.z0, d.ctx.th0
    st = d.new_state(B)
    pmax = 0.0
    for j in range(250):
        t = j * DT
        thr = np.minimum(d.ctx.k_drag * sr[:, 2] ** 2, d.ctx.t_max)
        noz = 0.3 * np.sin(0.5 * t) * np.ones(B)
        s = cs.sample(sr, t)
        ns = red.step(sr, thr, noz, s["eta"], d.ctx.x_st, DT)[0]
        nu0 = (ns[:, rv] - sr[:, rv]) / DT
        o = d.substep(st, sr, thr, noz, s, nu0=nu0,
                      cmd=np.stack([thr, noz], 1))
        rc, _ = CB.clip_push(o["acc"], np.zeros_like(o["acc"]))
        ns[:, rv] += rc * DT + o["imp"]
        pmax = max(pmax, float(np.abs(o["acc"]).max()))
        sr = ns
    fin = bool(np.isfinite(sr).all()) and "rpm" in o["obs"]
    return rest == 0.0 and fin, (f"at rest {rest:.1e}; 10 s loop finite "
                                 f"{fin}, max |a| {pmax:.2f}")
