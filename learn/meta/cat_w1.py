#!/usr/bin/env python3
"""
Catalogue item W1 (learn/meta/PRIOR_D10_DRAFT.md 3.1, with the corrections
of section 10): the nonlinearity of planing lift in the immersion and trim,
and its extrapolation in speed. The worked example of the item pattern
(learn/meta/cat_base.py): draw -> state -> step -> checks.

Form (only the part the low-fidelity boat lacks; its own linear restoring
is the zero point):
  eps   = mean over the 5 centre-line stations of max(h_i, 0) / the same
          at calm running - 1   (h_i keel immersion; dry stations count 0;
          eps = -1 when the whole keel is out of the water)
  Phi   = (tau_e / tau_eq(U))^q [(1 - w_h)(1 + eps)_+^p_d
                                     + w_h (1 + eps)_+^p_h]
          w_h = r / (1 + r), r = c_h / Cv^2 (Savitsky's static / dynamic
          lift ratio 0.458 lambda^2 / Cv^2); 0 out of the water,
          non-decreasing in eps. Phi = 1 at calm running at the true boat's
          speed-dependent running trim tau_eq(U) = soft floor of tau0
          s^-p_tau (the draft's tau_eq; = tau_run at U_id): in steady
          running the true lift equals the weight at any speed, so the lift
          excess and the drag below are 0 there (review 2026-09-30: referred
          to tau_run, the trim shift alone gave a steady one-sided drag of
          up to -4.5 m/s^2 at 16 kn)
  F_z   = W [Phi - 1 - Phi_eps eps - Phi_tau (tau_e - tau_eq)]  (tangent
          plane at the running point removed: curvature only)
  M     = s_p W Phi l_x eps / (1 + |eps|)  (the centre of pressure moves
          forward monotonically with immersion; generalised pitch force;
          kept whole as the draft writes it; its linear heave-to-pitch
          slope s_p W l_x / hbar_run is the only linear coupling while W1
          is on: layer-1 T4's k35 / k53 are switched off, D10.8)
  X     = -W (Phi - 1) tan(tau_e)          (pressure drag of the lift in
          excess of equilibrium, F_z,tot - W = W (Phi - 1))
  speed extrapolation, monotone power laws in s = U_r / U_id (U_id the
  low-fidelity identification speed u_design), not clipped at the task
  range's upper end (floored at its lower end u_lo / U_id, where W11
  takes over):
    running trim shift   a_theta += w_p^2 s_p tau0 (s^-p_tau - 1)
    heave stiffness      a_z += -w_h^2 (s^p_K - 1) (z - z_ref)
    heave damping        a_z += extra linear damping 2 z_h w_h (s^p_B - 1)
                         on the heave rate relative to the water reference
                         (exact exponential when positive)
Stiffness-type terms are velocity kicks (draft 6 item 6). Heave and pitch
are vertical hydrodynamic forces (gated by W4 when it lands). Overlaps:
T4's speed polynomial, bounded tanh stiffness and heave-pitch coupling off
(cat_base.OVERLAP).
"""
import numpy as np

from learn.meta import cat_base as CB


@CB.register
class W1(CB.CatItem):
    code, stage = "W1", "force"
    gated, calm_zero = True, True

    def draw_on(self, rng, ctx):
        u = rng.uniform
        return dict(p_d=u(0.3, 0.8), p_h=u(1.5, 3.0), q=u(0.8, 1.3),
                    c_h=u(0.46, 4.1), l_x=u(0.0, 0.15) * ctx.L,
                    p_tau=u(0.5, 1.8), p_K=u(0.0, 1.8), p_B=u(0.0, 1.0))

    def nominal(self, params, ctx):
        """Soft-saturation scale: at most half the clip per channel
        (cat_base.half_clip_nominal, D10.8)."""
        return CB.half_clip_nominal(ctx)

    def calm_params(self, params):
        d = dict(params)
        d["p_tau"] = 0.0
        return d

    @staticmethod
    def eps(ctx, q):
        hbar = np.maximum(q["hc"], 0.0).mean(1)
        return hbar / max(ctx.hbar_run, 1e-9) - 1.0

    @staticmethod
    def speed_ratio(ctx, q):
        """s = U_r / U_id, floored at the task range's lower end."""
        return np.maximum(q["U_r"] / ctx.u_id, ctx.u_lo / ctx.u_id)

    @classmethod
    def tau_eq(cls, prm, ctx, q):
        """The true running trim at this speed: the soft floor of tau0
        s^-p_tau (what tau_e settles to once the trim kick below has
        shifted the pitch equilibrium; = tau_run at U_id)."""
        s = cls.speed_ratio(ctx, q)
        return CB.softplus_floor(ctx.tau0 * s ** -prm["p_tau"], CB.TAU_MIN,
                                 CB.TAU_SOFT)

    @classmethod
    def phi(cls, prm, ctx, q, eps):
        """(Phi, Phi_eps, Phi_tau, tau_eq) at this substep."""
        one = np.maximum(1.0 + eps, 0.0)
        r = prm["c_h"] / np.maximum(q["Cv"] ** 2, 1e-6)
        wh = r / (1.0 + r)
        teq = cls.tau_eq(prm, ctx, q)
        tr = q["tau_e"] / teq
        Phi = tr ** prm["q"] * ((1.0 - wh) * one ** prm["p_d"]
                                + wh * one ** prm["p_h"])
        return Phi, (1.0 - wh) * prm["p_d"] + wh * prm["p_h"], \
            prm["q"] / teq, teq

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        B = q["B"]
        eps = self.eps(ctx, q)
        Phi, Pe, Pt, teq = self.phi(prm, ctx, q, eps)
        W = ctx.W
        Q = np.zeros((B, 5))
        Q[:, 0] = -W * (Phi - 1.0) * np.tan(q["tau_e"])
        Q[:, 3] = W * (Phi - 1.0 - Pe * eps - Pt * (q["tau_e"] - teq))
        Q[:, 4] = ctx.sp * W * Phi * prm["l_x"] * eps / (1.0 + np.abs(eps))
        s = self.speed_ratio(ctx, q)
        acc = np.zeros((B, 5))
        acc[:, 4] = ctx.wp ** 2 * ctx.sp * ctx.tau0 * (s ** -prm["p_tau"]
                                                       - 1.0)
        acc[:, 3] = -ctx.wh ** 2 * (s ** prm["p_K"] - 1.0) * q["e_z"] \
            + CB.lin_damp_acc(2 * ctx.zh * ctx.wh * (s ** prm["p_B"] - 1.0),
                              q["w_r"], dt)
        return CB.ItemOut(acc=acc, Q=Q, state=state)


# ------------------------------------------------------------ checks
def _states(ctx, dz):
    """Calm rows at the running attitude, heave offsets dz (bow and stern
    together), speed u_id."""
    n = len(dz)
    sr = np.zeros((n, 10))
    sr[:, 2] = ctx.u_id
    sr[:, 3] = ctx.z0 + np.asarray(dz, float)
    sr[:, 5] = ctx.th0
    return sr


def _q(ctx, sr):
    sea = CB.CatSea(None, ctx).sample(sr, 0.0)
    return CB.substep_quantities(ctx, sr, 0.0, 0.0, sea)


@CB.check("W1")
def lift_monotone_and_dry(item, ctx):
    """Total lift W Phi is non-decreasing as the hull sinks and exactly 0
    once the whole keel is out of the water."""
    rng = np.random.default_rng(3)
    dz = np.linspace(0.6, -0.4, 201)             # from dry to deep
    q = _q(ctx, _states(ctx, dz))
    worst = 0.0
    for _ in range(20):
        prm = item.draw_on(rng, ctx)
        Phi = item.phi(prm, ctx, q, item.eps(ctx, q))[0]
        worst = min(worst, float(np.diff(Phi).min()))
        dry = (q["hc"] <= 0).all(1)
        if not dry.any() or np.abs(Phi[dry]).max() > 0:
            return False, "lift not zero out of the water"
    return worst >= -1e-12, f"min step of Phi along sinkage {worst:.2e}"


@CB.check("W1")
def tangent_plane_removed(item, ctx):
    """F_z has no linear part in heave or trim at the running point: a
    small offset gives a second-order force."""
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(20):
        prm = item.draw_on(rng, ctx)
        for k in (3, 5):
            for d in (1e-4, -1e-4):
                sr = _states(ctx, [0.0])
                sr[0, k] += d
                q = _q(ctx, sr)
                F = item.step(prm, {}, q, CB.DT_SUB).Q[0, 3]
                worst = max(worst, abs(F) / (ctx.W * abs(d) / ctx.T))
    return worst < 1e-2, f"max |F_z| / (W |d| / T) = {worst:.2e}"


@CB.check("W1")
def no_steady_drag_off_design(item, ctx):
    """Calm running off U_id with the pitch at the shifted running trim
    (tau_e = tau_eq(U)) and the calm-running immersion: Phi = 1, so the
    drag X and the vertical force are zero (review 2026-09-30)."""
    rng = np.random.default_rng(6)
    worst = 0.0
    for U in (ctx.u_lo, 0.5 * (ctx.u_lo + ctx.u_id), ctx.u_hi):
        for _ in range(10):
            prm = item.draw_on(rng, ctx)
            sr = _states(ctx, [0.0])
            sr[:, 2] = U
            s = float(item.speed_ratio(ctx, _q(ctx, sr))[0])
            sr[:, 5] = ctx.th0 + ctx.sp * ctx.tau0 * (s ** -prm["p_tau"]
                                                      - 1.0)
            q = _q(ctx, sr)
            q["hc"] = np.broadcast_to(ctx.hc_run, q["hc"].shape).copy()
            o = item.step(prm, {}, q, CB.DT_SUB)
            worst = max(worst, abs(o.Q[0, 0]) / ctx.W,
                        abs(o.Q[0, 3]) / ctx.W)
    return worst < 1e-9, f"max |X|, |F_z| / W = {worst:.2e}"


@CB.check("W1")
def cp_moves_forward(item, ctx):
    """Sinking (eps > 0) moves the centre of pressure forward: a bow-up
    pitch acceleration (s_p theta_ddot > 0) from M alone."""
    rng = np.random.default_rng(5)
    q = _q(ctx, _states(ctx, [-0.05]))
    for _ in range(20):
        prm = item.draw_on(rng, ctx)
        prm["l_x"] = max(prm["l_x"], 0.01 * ctx.L)
        Qm = np.zeros((1, 5))
        Qm[0, 4] = item.step(prm, {}, q, CB.DT_SUB).Q[0, 4]
        a = CB.to_acc(ctx, Qm)[0, 4]
        if not ctx.sp * a > 0:
            return False, f"pitch acceleration {a:.3g} not bow-up"
    return True, "bow-up pitch acceleration for eps > 0 in 20 draws"
