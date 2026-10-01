#!/usr/bin/env python3
"""
Catalogue items W7-W12 (learn/meta/PRIOR_D10_DRAFT.md 3.1 with sections 10
and 11), group "water b", on the learn/meta/cat_base.py interface. Every
item adds only what the low-fidelity boat lacks; forces are mapped to the
five velocity channels with the draw's shared inertia M_t (lever_Q, to_acc)
unless written as accelerations. Sign conventions of cat_base: x forward,
y to port, z up; yaw positive bow to port; station index 0 = stern, lateral
index 0 = starboard, 2 = port; d = station immersion, h = keel immersion,
dd = entry velocity (> 0 into the water); pitch through s_p (bow-up trim
= s_p (theta - th0)).

W7  viscous crossflow drag, vertical and lateral (on 0.7 each; item-level
    1 - 0.3^2, the parts drawn conditionally).
      vertical, at each of the 5 x 3 station points (width b / 3, length
      dx_i trapezoidal): F_z = rho (b/3) dx_i C_Dc chi_ij |V| V (upward when
      entering, V = dd), chi_ij = sigma((h_ij - h_off) / w_d) the contact
      gate by keel immersion (W4's form and ranges), times r_exit when the
      point leaves the water (dd < 0; the W4 bullet "W1, W3, W7 speed parts
      x r_exit");
      lateral, per station: F_y = -1/2 rho C_Dl h_i+ dx_i |w| w, w = v_r +
      x_i r - c_orb v_orb,i (local dynamic draft = centre-line keel
      immersion, so the wetted length follows speed and waves).
      Numerics: ONE linearised implicit update of all stations together,
      (M_t + 2 dt D) dnu = dt F, D = sum c_k |V_k| g_k g_k^T (the vector form
      of draft W7's dv = -dt c|v|v / (m + 2 dt c|v|)): the velocity relative
      to the water contracts in the M_t norm by (1 + dt l) / (1 + 2 dt l) in
      every eigen-direction, so it never reverses and never gains energy.
      Ranges: C_Dc ~ U[0.5, 1.2] (cos beta, stronger separation),
      C_Dl ~ U[0.3, 1.5], h_off ~ U[-0.05, 0.05] m, w_d ~ U[0.02, 0.1] m,
      r_exit ~ U[0, 0.3], c_orb ~ U[0, 1] [assumption].
      Overlaps: T3's lateral kappa off with the lateral part, T3's
      heave / pitch kappa off with the vertical part (changes 5, 10).
W8  second-order wave forces (on 0.8 with waves). zeta = d - <d> at the
    5 x 3 points (<d> an exact first-order mean, tau_m ~ LogU[5, 20] s);
    zeta_s = mean over the three lateral points, zeta_a = (port - stbd) / 2.
      surge  X = -tanh(u_r / 0.5) W c1 s / (1 + c2 s) max(U_r / U_id,
             u_lo / U_id)^p_u, s = sqrt(LPF_tau2[zeta_s^T Q_x zeta_s]) / b,
             Q_x = 16 A A^T / tr(A A^T) (so isotropic zeta of rms sigma
             gives s = 4 sigma / b = Hs / b, Savitsky-Brown's H / b);
      sway   a_y = k_y 0.3 m/s^2 LPF[zeta_s^T Q_y zeta_a] / zeta_ref^2,
      yaw    a_r = k_n 0.1 rad/s^2 LPF[zeta_s^T Q_n zeta_a] / zeta_ref^2:
             symmetric x antisymmetric, so a mirror image flips their sign;
             zeta_ref = 0.25 m (the rms elevation of Hs = 1 m, the draft's
             scaling sea state), ||Q||_F = 1, k ~ U[0, 1] (the sign is in Q);
      heave / pitch the same symmetric form, k ~ U[-1, 1] x 0.03 a_ref.
      c1 ~ U[0.1, 0.6], c2 ~ U[1, 3], p_u ~ U[-1, 1], tau2 ~ LogU[2, 30] s.
      Overlap: T7's one-direction quadratic wave term off (change 4).
W9  Froude-Krylov surge force in following seas (on 0.7). X_w = -mu_x m g
    F(s_bar), s_bar the mean local longitudinal surface slope over the wet
    centre-line stations (h > 0; 0 when none is wet), F w.p. 0.5 a random
    first-order lead or lag (1 + tau_z s) / (1 + tau_p s) with unit static
    gain, tau_p ~ LogU[0.07, 1.4] rs s, tau_z / tau_p ~ LogU[1/3, 3]
    [assumption], mu_x ~ U[0.4, 1.3]. W.p. 0.5 also the first-order drag
    change of the orbital velocity X_o = +2 k_drag |u_r| mu_o u_orb (u_orb
    the wet-station mean of the forward orbital velocity, mu_o ~ U[0, 1]).
W10 destabilising bow force when the bow digs in (on 0.4). Y_b = -c U_r
    g_b (v_r + x_b r) at (x_b, 0, 0), N_b = x_b Y_b, g_b = ((h_bow - d_0)+ /
    T)^q_b, h_bow the mean keel immersion of the two foremost centre-line
    stations, c U_r = kap_b k_lin_sway U_r / (10 m/s): the draft's
    c_b U / (m_v x the low-fidelity sway damping k_lin_sway / m_sway) ~
    LogU[0.1, 2] at U = 10 m/s. x_b ~ U[0.25, 0.5] L, q_b ~ U[1, 2],
    d_0 ~ U[0, 0.5] T. Numerics: the bow-point lateral velocity decays by
    the exact exponential exp(-c dt / m_g) (m_g the point's generalised
    mass), never reverses. W.p. 0.5 also: the low-fidelity yaw damping and
    the yaw effect of its jet side force (control/reduced.py Nomoto law,
    (k_nomoto_f lift - r) / tau_r) times eps_r tanh(slope / s_ref),
    eps_r ~ U[-1, 1], s_ref = 0.05 rad.
W11 transition resistance and trim hump (on 0.6). B = exp(-x^2 / 2),
    x = (Fn_vol - F_c) / w (w.p. 0.5 the two sides' widths times U[0.6,
    1.6] each); dX = -W r_h B tanh(u_r / 0.5) (drag only), trim +theta_h B,
    sinkage -z_h B as shifts of the low-fidelity equilibrium (a = w^2 x
    shift, velocity kicks). W.p. 0.3 hysteresis: a hidden switch (flips up
    above F_h + dF / 2, down below F_h - dF / 2, smoothed by tau_s ~
    LogU[1, 10] s) moves the centre F_c from F_h + dF / 2 (accelerating) to
    F_h - dF / 2 (decelerating). F_h ~ U[0.9, 2], w ~ U[0.2, 0.6], r_h ~
    U[0, 0.15], theta_h ~ U[0, 5 deg], z_h ~ U[0, 0.3] T, dF ~ U[0, 0.3].
    Overlap: T3's surge shape gated to the task speeds (change 12).
W12 stepped hull (section 11; on 0.3; form written here after W5 / W10):
    a step at x_s = x_stern + f_s (x_bow - x_stern), f_s ~ U[0.25, 0.5];
    the wetted region behind it is centred at x_c = x_s - U[0, 0.15] L.
    Hidden states: a speed switch s_U (W5(b) form: flips up above U_v +
    h_U / 2, down below U_v - h_U / 2, smoothed by tau_s) and one
    ventilation state per side v_j in [0, 1] (1 = air behind the step),
    exact exponential towards
      v_j,inf = s_U (1 - S((h_eff,j - h_v) / w_v)),   S = smoothstep on
      [0, 1], h_eff,j = h_j + k_beta s_j (B / 2) beta_s,
    h_j the keel immersion at the step on side j (s_j = +1 port, -1
    starboard; roll from H1 enters through h_j), beta_s = (v_r + x_s r) /
    max(U_r, 2) the sideslip at the step, falling with tau_loss (fast) and
    rising with tau_rec. The step is ventilated at calm running by
    construction (h_v = h_run,s + dh_v > h_run,s), which is the zero point.
      collapse (per side, contact c_j = clip(h_j / 0.05 m, 0, 1), wet_j =
      (1 - v_j) c_j): at (x_c, s_j B / 4, keel) F = W / 2 wet_j (-r_c,
      -s_j c_y, l_c): drag, a lateral force and a lift change; unequal
      sides give the "hook": the drag difference yaws the bow towards the
      wetted side, the lateral force (either sign: deadrise pressure pushes
      away from the wetted side, suction or a digging chine pulls towards
      it; the step may lie either side of the centre of gravity) adds a
      sudden sway and yaw of random direction;
      ventilated-region lift loss (centre, section 11: "depends on the step
      immersion and trim"): F_z = -W l_v (G - 1) v_bar c_c, G = (h_c+ /
      h_run,s)^a_h (tau_e / tau_run)^a_tau, and its pressure drag -F_z tan
      tau_e, at (x_c, 0, keel).
    Ranges [assumption]: U_v ~ U[10, 25] kn, h_U ~ U[0, 4] kn, tau_s ~
    LogU[0.3, 3] s, dh_v ~ U[0.2, 2] h_run,s, w_v ~ U[0.1, 0.4] h_run,s,
    k_beta ~ U[-2, 2], tau_loss ~ LogU[0.02, 0.3] s (as P1's loss),
    tau_rec ~ LogU[0.1, 2] s, l_c ~ U[-0.15, 0.1], r_c ~ U[0, 0.05],
    c_y ~ U[-0.25, 0.25], l_v ~ U[0, 0.1], a_h ~ U[0.5, 2], a_tau ~ U[0, 1.5].

W7, W9, W10 and W12 gate themselves by keel immersion per station or per
side, so they are NOT marked `gated` (W4's mean gate would count the
contact twice); W8 and W11 vertical parts are gated like W1. W7, W8, W9,
W10 are exactly zero in calm steady straight running (calm_zero); W11 and
W12 are speed-dependent steady terms by design (like W5(b)).
Construction checks: @cat_base.check below (studies/test_cat_items.py I5).
"""
import numpy as np

from learn.meta import cat_base as CB
from learn.meta.operators import _logu

KN = CB.KN
ZETA_REF = 0.25                  # m: rms elevation of Hs = 1 m (W8 scaling)
W8_SWAY, W8_YAW, W8_VERT = 0.3, 0.1, 0.03
W10_U_REF = 10.0                 # m/s: the draft's reference speed of c_b
W10_S_REF = 0.05                 # rad: slope scale of the yaw modulation
W12_W_CONTACT = 0.05             # m: step contact ramp
SIGN_U = 0.5                     # m/s: smooth sign of the surge terms


def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, float)))


def _dx(ctx):
    """Trapezoidal station lengths (5,) over the station span."""
    d = (ctx.x_st[-1] - ctx.x_st[0]) / 4.0
    w = np.full(5, d)
    w[[0, -1]] *= 0.5
    return w


def _gv(ctx):
    """Generalised directions (5, 5) of a unit upward force at stations."""
    return np.stack([CB.unit_vertical(ctx, x) for x in ctx.x_st])


def _gl(ctx):
    """Generalised directions (5, 5) of a unit port force at stations."""
    g = np.zeros((5, 5))
    g[:, 1], g[:, 2] = 1.0, ctx.x_st
    return g


def _interp_w(xs, x):
    """Linear interpolation weights (len(xs),) of the point x."""
    w = np.zeros(len(xs))
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    f = float(np.clip((x - xs[i]) / (xs[i + 1] - xs[i]), 0.0, 1.0))
    w[i], w[i + 1] = 1.0 - f, f
    return w


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _both_parts(rng, p=0.7):
    """Two independent parts on w.p. p each, given at least one is on."""
    while True:
        a, b = rng.random() < p, rng.random() < p
        if a or b:
            return bool(a), bool(b)


# ------------------------------------------------------------------ W7
@CB.register
class W7(CB.CatItem):
    code, stage = "W7", "force"
    gated, calm_zero = False, True

    def draw_on(self, rng, ctx):
        vert, lat = _both_parts(rng)
        u = rng.uniform
        return dict(vert=vert, lat=lat, C_Dc=u(0.5, 1.2), h_off=u(-0.05, 0.05),
                    w_d=u(0.02, 0.1), r_exit=u(0.0, 0.3), C_Dl=u(0.3, 1.5),
                    c_orb=u(0.0, 1.0))

    def overlaps(self, params):
        s = set()
        if params["vert"]:
            s.add("T3_kap_vert_off")
        if params["lat"]:
            s.add("T3_kap_lat_off")
        return s

    def forces(self, prm, q):
        """(F (B, 5) generalised force, D (B, 5, 5) its linearisation)."""
        ctx = q["ctx"]
        B = q["B"]
        dx = _dx(ctx)
        F, D = np.zeros((B, 5)), np.zeros((B, 5, 5))
        if prm["vert"]:
            gv = _gv(ctx)
            V = q["dd"]                                   # (B, 5, 3), in > 0
            chi = _sig((q["h"] - prm["h_off"]) / prm["w_d"])
            c = CB.RHO_W * (ctx.b / 3.0) * dx[None, :, None] * prm["C_Dc"] \
                * chi * np.where(V < 0.0, prm["r_exit"], 1.0)
            # hull velocity relative to the water along g_i is -V
            F += np.einsum("bi,ik->bk", (c * np.abs(V) * V).sum(2), gv)
            D += np.einsum("bi,ik,il->bkl", (c * np.abs(V)).sum(2), gv, gv)
        if prm["lat"]:
            gl = _gl(ctx)
            r = q["sr"][:, 8]
            w = q["v_r"][:, None] + ctx.x_st[None] * r[:, None] \
                - prm["c_orb"] * q["vorb"][:, :, 1]
            c = 0.5 * CB.RHO_W * prm["C_Dl"] * np.maximum(q["hc"], 0.0) \
                * dx[None]
            F -= np.einsum("bi,ik->bk", c * np.abs(w) * w, gl)
            D += np.einsum("bi,ik,il->bkl", c * np.abs(w), gl, gl)
        return F, D

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        F, D = self.forces(prm, q)
        A = ctx.Mt[None] + 2.0 * dt * D
        acc = np.linalg.solve(A, F[..., None])[..., 0]
        return CB.ItemOut(acc=acc, state=state)


# ------------------------------------------------------------------ W8
def _unit_fro(A):
    return A / max(float(np.linalg.norm(A)), 1e-12)


@CB.register
class W8(CB.CatItem):
    code, stage = "W8", "force"
    gated, calm_zero = True, True

    def draw_on(self, rng, ctx):
        u = rng.uniform
        A = rng.normal(0.0, 1.0, (5, 5))
        AA = A @ A.T
        Qz = rng.normal(0.0, 1.0, (5, 5))
        Qt = rng.normal(0.0, 1.0, (5, 5))
        return dict(c1=u(0.1, 0.6), c2=u(1.0, 3.0), p_u=u(-1.0, 1.0),
                    tau2=float(_logu(rng, 2.0, 30.0)),
                    tau_m=float(_logu(rng, 5.0, 20.0)),
                    Qx=16.0 * AA / np.trace(AA),
                    Qy=_unit_fro(rng.normal(0.0, 1.0, (5, 5))),
                    Qn=_unit_fro(rng.normal(0.0, 1.0, (5, 5))),
                    Qz=_unit_fro(Qz + Qz.T), Qt=_unit_fro(Qt + Qt.T),
                    k_y=u(0.0, 1.0), k_n=u(0.0, 1.0), k_z=u(-1.0, 1.0),
                    k_t=u(-1.0, 1.0))

    def init_state(self, params, B=1, rng=None):
        return dict(dm=None, L=np.zeros((B, 5)))

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        d = q["d"]
        if state["dm"] is None:
            state["dm"] = d.copy()
        z = d - state["dm"]
        state["dm"] = CB.exp_update(state["dm"], d, prm["tau_m"], dt)
        zs = z.mean(2)
        za = 0.5 * (z[:, :, 2] - z[:, :, 0])
        f = np.stack([np.einsum("bi,ij,bj->b", zs, prm[k], zz)
                      for k, zz in (("Qx", zs), ("Qy", za), ("Qn", za),
                                    ("Qz", zs), ("Qt", zs))], 1)
        state["L"] = L = CB.exp_update(state["L"], f, prm["tau2"], dt)
        s = np.sqrt(np.maximum(L[:, 0], 0.0)) / ctx.b
        su = np.maximum(q["U_r"] / ctx.u_id, ctx.u_lo / ctx.u_id)
        B = q["B"]
        Q = np.zeros((B, 5))
        Q[:, 0] = -np.tanh(q["u_r"] / SIGN_U) * ctx.W * prm["c1"] * s \
            / (1.0 + prm["c2"] * s) * su ** prm["p_u"]
        acc = np.zeros((B, 5))
        z2 = ZETA_REF ** 2
        acc[:, 1] = prm["k_y"] * W8_SWAY * L[:, 1] / z2
        acc[:, 2] = prm["k_n"] * W8_YAW * L[:, 2] / z2
        acc[:, 3] = prm["k_z"] * W8_VERT * ctx.a_ref[3] * L[:, 3] / z2
        acc[:, 4] = prm["k_t"] * W8_VERT * ctx.a_ref[4] * L[:, 4] / z2
        return CB.ItemOut(acc=acc, Q=Q, state=state)


# ------------------------------------------------------------------ W9
def _wet_mean(x, wet):
    n = wet.sum(1)
    return np.where(n > 0, (x * wet).sum(1) / np.maximum(n, 1), 0.0)


@CB.register
class W9(CB.CatItem):
    code, stage = "W9", "force"
    gated, calm_zero = False, True

    def draw_on(self, rng, ctx):
        u = rng.uniform
        return dict(mu_x=u(0.4, 1.3), filt=bool(rng.random() < 0.5),
                    tau_p=float(_logu(rng, 0.07, 1.4)) * ctx.rs,
                    ratio=float(_logu(rng, 1.0 / 3.0, 3.0)),
                    orb=bool(rng.random() < 0.5), mu_o=u(0.0, 1.0))

    def init_state(self, params, B=1, rng=None):
        return dict(y=np.zeros(B))

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        wet = (q["hc"] > 0.0).astype(float)
        sl = np.gradient(q["eta"][:, :, 1], ctx.x_st, axis=1)
        s = _wet_mean(sl, wet)
        if prm["filt"]:
            state["y"] = CB.exp_update(state["y"], s, prm["tau_p"], dt)
            s = prm["ratio"] * s + (1.0 - prm["ratio"]) * state["y"]
        B = q["B"]
        Q = np.zeros((B, 5))
        Q[:, 0] = -prm["mu_x"] * ctx.m * CB.GRAV * s
        if prm["orb"]:
            uo = _wet_mean(q["uorb"][:, :, 1], wet)
            Q[:, 0] += 2.0 * ctx.k_drag * np.abs(q["u_r"]) * prm["mu_o"] * uo
        return CB.ItemOut(Q=Q, state=state)


# ------------------------------------------------------------------ W10
@CB.register
class W10(CB.CatItem):
    code, stage = "W10", "force"
    gated, calm_zero = False, True

    def draw_on(self, rng, ctx):
        u = rng.uniform
        return dict(x_b=u(0.25, 0.5) * ctx.L, q_b=u(1.0, 2.0),
                    d_0=u(0.0, 0.5) * ctx.T,
                    kap_b=float(_logu(rng, 0.1, 2.0)),
                    mod=bool(rng.random() < 0.5), eps_r=u(-1.0, 1.0))

    @staticmethod
    def g_dir(prm):
        return np.array([0.0, 1.0, prm["x_b"], 0.0, 0.0])

    def bow_coef(self, prm, q):
        """c (B,) N s/m of the bow lateral damping."""
        ctx = q["ctx"]
        hb = q["hc"][:, 3:5].mean(1)
        gb = (np.maximum(hb - prm["d_0"], 0.0) / ctx.T) ** prm["q_b"]
        return prm["kap_b"] * float(ctx.p["k_lin_sway"]) * q["U_r"] \
            / W10_U_REF * gb

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        g = self.g_dir(prm)
        mg = CB.point_mass(ctx, g)
        c = self.bow_coef(prm, q)
        w = q["v_r"] + prm["x_b"] * q["sr"][:, 8]
        F = mg * w * np.expm1(-c * dt / mg) / dt          # exact decay of w
        Q = F[:, None] * g[None]
        acc = None
        if prm["mod"]:
            acc = np.zeros((q["B"], 5))
            p = ctx.p
            r = q["sr"][:, 8]
            m = prm["eps_r"] * np.tanh(q["slope"] / W10_S_REF)
            if p.get("steer_jet", 0.0) > 0.5:
                lift = ctx.k_js * np.maximum(q["thr"], 0.0) * np.sin(
                    np.clip(q["noz"], -ctx.rud_stall, ctx.rud_stall))
                rt = p["k_nomoto_f"] * lift
            else:
                rt = p["k_nomoto"] * q["noz"]
            acc[:, 2] = m * rt / p["tau_r"] \
                + CB.lin_damp_acc(m / p["tau_r"], r, dt)
        return CB.ItemOut(acc=acc, Q=Q, state=state)


# ------------------------------------------------------------------ W11
@CB.register
class W11(CB.CatItem):
    code, stage = "W11", "force"
    gated, calm_zero = True, False

    def draw_on(self, rng, ctx):
        u = rng.uniform
        asym = bool(rng.random() < 0.5)
        f_lo, f_hi = u(0.6, 1.6), u(0.6, 1.6)
        hyst = bool(rng.random() < 0.3)
        return dict(F_h=u(0.9, 2.0), w_h=u(0.2, 0.6), r_h=u(0.0, 0.15),
                    th_h=u(0.0, 5.0) * CB.DEG, z_h=u(0.0, 0.3) * ctx.T,
                    asym=asym, f_lo=f_lo if asym else 1.0,
                    f_hi=f_hi if asym else 1.0, hyst=hyst,
                    dF=u(0.0, 0.3) if hyst else 0.0,
                    tau_s=float(_logu(rng, 1.0, 10.0)))

    def init_state(self, params, B=1, rng=None):
        return dict(h=None, s=None)

    def bump(self, prm, state, q, dt):
        Fn = q["Fn_vol"]
        c = np.full_like(Fn, prm["F_h"])
        if prm["hyst"]:
            if state["h"] is None:
                state["h"] = (Fn > prm["F_h"]).astype(float)
                state["s"] = state["h"].copy()
            s = CB.hyst_switch(state, Fn, prm["F_h"] + 0.5 * prm["dF"],
                               prm["F_h"] - 0.5 * prm["dF"], prm["tau_s"], dt)
            c = prm["F_h"] + 0.5 * prm["dF"] * (1.0 - 2.0 * s)
        x = Fn - c
        w = prm["w_h"] * np.where(x < 0.0, prm["f_lo"], prm["f_hi"])
        return np.exp(-0.5 * (x / w) ** 2)

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        Bm = self.bump(prm, state, q, dt)
        B = q["B"]
        Q = np.zeros((B, 5))
        Q[:, 0] = -ctx.W * prm["r_h"] * Bm * np.tanh(q["u_r"] / SIGN_U)
        acc = np.zeros((B, 5))
        acc[:, 3] = -ctx.wh ** 2 * prm["z_h"] * Bm
        acc[:, 4] = ctx.wp ** 2 * ctx.sp * prm["th_h"] * Bm
        return CB.ItemOut(acc=acc, Q=Q, state=state)


# ------------------------------------------------------------------ W12
@CB.register
class W12(CB.CatItem):
    code, stage = "W12", "force"
    gated, calm_zero = False, False

    def draw_on(self, rng, ctx):
        u = rng.uniform
        f_s = u(0.25, 0.5)
        x_s = ctx.x_st[0] + f_s * (ctx.x_st[-1] - ctx.x_st[0])
        h_run = float(_interp_w(ctx.x_st, x_s) @ ctx.hc_run)
        return dict(x_s=x_s, x_c=x_s - u(0.0, 0.15) * ctx.L,
                    z_k=float(np.interp(x_s, ctx.x_st, ctx.keel5)),
                    h_run=h_run,
                    U_v=u(10.0, 25.0) * KN, h_U=u(0.0, 4.0) * KN,
                    tau_s=float(_logu(rng, 0.3, 3.0)),
                    dh_v=u(0.2, 2.0) * h_run, w_v=u(0.1, 0.4) * h_run,
                    k_beta=u(-2.0, 2.0),
                    tau_loss=float(_logu(rng, 0.02, 0.3)),
                    tau_rec=float(_logu(rng, 0.1, 2.0)),
                    l_c=u(-0.15, 0.1), r_c=u(0.0, 0.05), c_y=u(-0.25, 0.25),
                    l_v=u(0.0, 0.1), a_h=u(0.5, 2.0), a_tau=u(0.0, 1.5))

    def init_state(self, params, B=1, rng=None):
        return dict(h=None, s=None, v=None)

    def vent(self, prm, state, q, dt):
        """Advance the hidden states; returns (v (B, 2) stbd / port,
        h_step (B, 3) keel immersion at the step)."""
        ctx = q["ctx"]
        hs = np.einsum("bij,i->bj", q["h"], _interp_w(ctx.x_st, prm["x_s"]))
        U = q["U_r"]
        up, dn = prm["U_v"] + 0.5 * prm["h_U"], prm["U_v"] - 0.5 * prm["h_U"]
        if state["h"] is None:
            state["h"] = (U > prm["U_v"]).astype(float)
            state["s"] = state["h"].copy()
        sU = CB.hyst_switch(state, U, up, dn, prm["tau_s"], dt)
        beta = (q["v_r"] + prm["x_s"] * q["sr"][:, 8]) / np.maximum(U, 2.0)
        sg = np.array([-1.0, 1.0])
        h2 = hs[:, [0, 2]]
        heff = h2 + prm["k_beta"] * sg[None] * 0.5 * ctx.B * beta[:, None]
        hv = prm["h_run"] + prm["dh_v"]
        vinf = sU[:, None] * (1.0 - _smoothstep((heff - hv) / prm["w_v"]))
        if state["v"] is None:
            state["v"] = vinf.copy()
        state["v"] = CB.exp_update_asym(state["v"], vinf, prm["tau_rec"],
                                        prm["tau_loss"], dt)
        return state["v"], hs

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        B = q["B"]
        v, hs = self.vent(prm, state, q, dt)
        th = q["sr"][:, 5]
        W = ctx.W
        Q = np.zeros((B, 5))
        for k, (j, sg) in enumerate(((0, -1.0), (2, 1.0))):
            cj = np.clip(hs[:, j] / W12_W_CONTACT, 0.0, 1.0)
            wet = (1.0 - v[:, k]) * cj
            F = 0.5 * W * wet[:, None] * np.array(
                [-prm["r_c"], -sg * prm["c_y"], prm["l_c"]])[None]
            Q += CB.lever_Q(ctx, (prm["x_c"], sg * 0.25 * ctx.B, prm["z_k"]),
                            F, th)
        cc = np.clip(hs[:, 1] / W12_W_CONTACT, 0.0, 1.0)
        G = (np.maximum(hs[:, 1], 0.0) / prm["h_run"]) ** prm["a_h"] \
            * (q["tau_e"] / ctx.tau_run) ** prm["a_tau"]
        Fz = -W * prm["l_v"] * (G - 1.0) * v.mean(1) * cc
        F = np.stack([-Fz * np.tan(q["tau_e"]), np.zeros(B), Fz], 1)
        Q += CB.lever_Q(ctx, (prm["x_c"], 0.0, prm["z_k"]), F, th)
        return CB.ItemOut(Q=Q, state=state)


# ================================================================ checks
def _run(ctx, u=None, dz=0.0, sr=None):
    """Calm rows at the running attitude (speeds u, heave offset dz)."""
    if sr is None:
        u = np.atleast_1d(np.asarray(u, float))
        sr = np.zeros((len(u), 10))
        sr[:, 2], sr[:, 3], sr[:, 5] = u, ctx.z0 + dz, ctx.th0
    return sr


def _q(ctx, sr, rowseas=None, t=0.0, thr=0.0, noz=0.0):
    sea = CB.CatSea(rowseas, ctx).sample(sr, t)
    return CB.substep_quantities(ctx, sr, thr, noz, sea)


def _seas(ctx, B, seed, hs=1.5):
    from learn.meta import operators_rb as R
    rng = np.random.default_rng(seed)
    d = R.sea_dict(rng)
    d["hs"] = hs
    return R.RowSeas([R.sea_state(d, seed)] * B, ctx.x_st, ctx.y_off)


def _nu(sr):
    return sr[:, [2, 9, 8, 4, 6]]


@CB.check("W7")
def w7_contracts_energy(item, ctx):
    """Calm water, random body velocities up to 5 m/s, largest ranges: the
    implicit update never raises nu^T M_t nu, never pushes along nu, and
    every part gives the right overlap switches."""
    rng = np.random.default_rng(21)
    B = 400
    worst, push = -np.inf, -np.inf
    for _ in range(10):
        prm = item.draw_on(rng, ctx)
        prm.update(vert=True, lat=True, C_Dc=1.2, C_Dl=1.5, r_exit=0.3)
        sr = _run(ctx, rng.uniform(1, 25, B), rng.uniform(-0.4, 0.1, B))
        sr[:, 4], sr[:, 6] = rng.uniform(-5, 5, B), rng.uniform(-2, 2, B)
        sr[:, 9], sr[:, 8] = rng.uniform(-3, 3, B), rng.uniform(-1, 1, B)
        a = item.step(prm, {}, _q(ctx, sr), CB.DT_SUB).acc
        nu = _nu(sr)
        nu[:, 0] = 0.0                  # surge: no W7 direction
        E0 = np.einsum("bi,ij,bj->b", nu, ctx.Mt, nu)
        n1 = nu + a * CB.DT_SUB
        E1 = np.einsum("bi,ij,bj->b", n1, ctx.Mt, n1)
        worst = max(worst, float(((E1 - E0) / np.maximum(E0, 1e-9)).max()))
        push = max(push, float(np.einsum("bi,ij,bj->b", nu, ctx.Mt, a).max()))
    sw = [item.overlaps(dict(vert=v, lat=l)) for v, l in
          ((True, False), (False, True), (True, True))]
    ok_sw = sw == [{"T3_kap_vert_off"}, {"T3_kap_lat_off"},
                   {"T3_kap_vert_off", "T3_kap_lat_off"}]
    ok = worst <= 1e-12 and push <= 1e-9 and ok_sw
    return ok, (f"max relative energy change {worst:.1e} (<= 0), max "
                f"nu^T M a {push:.1e}; overlap switches per part {ok_sw}")


@CB.check("W7")
def w7_dry_and_lateral_scale(item, ctx):
    """Airborne rows: the vertical part vanishes (below 1e-6 of a_ref);
    the draft's lateral example (v = 1 m/s, draft 0.3 m, 3 m long,
    C_Dl = 1) gives about 0.29 m/s^2 on the sway mass."""
    rng = np.random.default_rng(22)
    prm = item.draw_on(rng, ctx)
    prm.update(vert=True, lat=False)
    sr = _run(ctx, [12.0, 20.0], 1.5)
    sr[:, 4], sr[:, 6] = -4.0, 1.0
    a = item.step(prm, {}, _q(ctx, sr), CB.DT_SUB).acc
    dry = float(np.abs(a[:, 3:5] / ctx.a_ref[3:5]).max())
    F = 0.5 * CB.RHO_W * 1.0 * 0.3 * 3.0 * 1.0
    ex = F / ctx.m_v
    return dry < 1e-6, (f"airborne vertical part {dry:.1e} a_ref; lateral "
                        f"example {ex:.2f} m/s^2 (draft 0.29)")


@CB.check("W8")
def w8_resists_and_mirror(item, ctx):
    """In a sea (300 substeps along a straight path): the surge force always
    opposes the motion and stays below W c1 / c2 x the speed factor; the
    port / starboard mirror image flips sway and yaw and keeps surge,
    heave and pitch."""
    rng = np.random.default_rng(23)
    B = 16
    rs = _seas(ctx, B, 5, hs=2.0)
    worst_sign, worst_cap, worst_m = 0.0, 0.0, 0.0
    for _ in range(3):
        prm = item.draw_on(rng, ctx)
        s1, s2 = item.init_state(prm, B), item.init_state(prm, B)
        sr = _run(ctx, rng.uniform(5, 25, B))
        sr[:, 7] = rng.uniform(-np.pi, np.pi, B)
        for k in range(300):
            t = k * CB.DT_SUB
            sr2 = sr.copy()
            sr2[:, 0] += sr[:, 2] * np.cos(sr[:, 7]) * t
            sr2[:, 1] += sr[:, 2] * np.sin(sr[:, 7]) * t
            q = _q(ctx, sr2, rs, t)
            qm = dict(q)
            qm["d"] = q["d"][:, :, ::-1].copy()
            o1 = item.step(prm, s1, q, CB.DT_SUB)
            o2 = item.step(prm, s2, qm, CB.DT_SUB)
            X = o1.Q[:, 0]
            worst_sign = max(worst_sign, float((X * sr[:, 2]).max()))
            su = np.maximum(sr[:, 2] / ctx.u_id, ctx.u_lo / ctx.u_id)
            cap = ctx.W * prm["c1"] / prm["c2"] * su ** prm["p_u"]
            worst_cap = max(worst_cap, float((np.abs(X) / cap).max()))
            sgn = np.array([1.0, -1.0, -1.0, 1.0, 1.0])
            sc = 1.0 + np.abs(o1.acc).max() + np.abs(o1.Q).max()
            worst_m = max(worst_m, float(np.abs(o1.acc - sgn * o2.acc).max()
                                         / sc),
                          float(np.abs(o1.Q - o2.Q).max() / sc))
    ok = worst_sign <= 0 and worst_cap <= 1 and worst_m < 1e-9
    return ok, (f"max X u {worst_sign:.1e} (<= 0), max |X| / cap "
                f"{worst_cap:.2f} (<= 1), mirror mismatch {worst_m:.1e}")


def _slope_q(ctx, s, u=12.0, dz=0.0):
    sr = _run(ctx, [u], dz)
    q = _q(ctx, sr)
    q["eta_add"] = s * ctx.x_st[None, :, None] * np.ones((1, 5, 3))
    return CB.derive_quantities(ctx, q)


@CB.check("W9")
def w9_sign_gain_dry(item, ctx):
    """A surface rising towards the bow (slope 0.05, all stations wet)
    pushes the boat back with -mu_x m g s; the lead / lag has unit static
    gain (60 s held); a dry hull gets nothing; a forward orbital velocity
    gives a forward push."""
    rng = np.random.default_rng(24)
    bad = []
    for _ in range(20):
        prm = item.draw_on(rng, ctx)
        prm.update(orb=False)
        q = _slope_q(ctx, 0.05)
        if not (q["hc"] > 0).all():
            return False, "test surface dries a station"
        st = item.init_state(prm, 1)
        X = item.step(prm, st, q, 60.0).Q[0, 0]     # exact: 60 s held
        want = -prm["mu_x"] * ctx.m * CB.GRAV * 0.05
        if abs(X - want) > 1e-6 * abs(want):
            bad.append("static gain")
        qd = _slope_q(ctx, 0.05, dz=1.5)
        if item.step(prm, item.init_state(prm, 1), qd, CB.DT_SUB).Q[0, 0]:
            bad.append("dry not zero")
    prm.update(orb=True, filt=False, mu_o=1.0)
    q = _slope_q(ctx, 0.0)
    q["uorb"] = np.full((1, 5, 3), 0.5)
    Xo = item.step(prm, item.init_state(prm, 1), q, CB.DT_SUB).Q[0, 0]
    if not Xo > 0:
        bad.append("orbital sign")
    return not bad, ", ".join(sorted(set(bad))) or (
        f"static -mu m g s exact in 20 draws; orbital push {Xo:.0f} N")


@CB.check("W10")
def w10_gate_implicit_moment(item, ctx):
    """Bow out of the water: exactly zero; bow deep, fast, largest kap_b:
    the bow-point lateral velocity decays without reversing; the yaw part
    is x_b times the side force; the yaw modulation stays within |eps_r| of
    the low-fidelity yaw acceleration."""
    rng = np.random.default_rng(25)
    bad = []
    wmin = np.inf
    for _ in range(20):
        prm = item.draw_on(rng, ctx)
        prm.update(mod=False)
        sr = _run(ctx, [15.0, 25.0, 8.0], 0.0)
        sr[:, 9], sr[:, 8] = [1.5, -2.0, 0.7], [0.4, 0.3, -0.5]
        up = sr.copy()
        up[:, 3] += 1.0
        o = item.step(prm, {}, _q(ctx, up), CB.DT_SUB)
        if np.abs(o.Q).max() != 0.0:
            bad.append("bow out not zero")
        prm["kap_b"] = 2.0
        dn = sr.copy()
        dn[:, 5] = ctx.th0 - ctx.sp * 0.1         # bow down 0.1 rad
        q = _q(ctx, dn)
        Qb = item.step(prm, {}, q, CB.DT_SUB).Q
        g = item.g_dir(prm)
        w = q["v_r"] + prm["x_b"] * dn[:, 8]
        dw = CB.to_acc(ctx, Qb) @ g * CB.DT_SUB
        wmin = min(wmin, float(((w + dw) / w).min()))
        if np.abs(Qb[:, 2] - prm["x_b"] * Qb[:, 1]).max() > 1e-9 or \
                (Qb[:, 1] * w > 0).any():
            bad.append("moment or direction")
    prm = item.draw_on(rng, ctx)
    prm.update(mod=True, eps_r=-0.8, kap_b=0.1, d_0=10.0)
    sr = _run(ctx, [15.0, 15.0], 0.0)
    sr[:, 8] = 0.2
    q = _q(ctx, sr, thr=3000.0, noz=0.2)
    q["eta_add"] = 0.2 * ctx.x_st[None, :, None] * np.ones((2, 5, 3))
    CB.derive_quantities(ctx, q)
    a = item.step(prm, {}, q, CB.DT_SUB).acc[:, 2]
    p = ctx.p
    lift = ctx.k_js * 3000.0 * np.sin(0.2)
    alo = (p["k_nomoto_f"] * lift - 0.2) / p["tau_r"]
    ratio = float(np.abs(a / alo).max())
    if ratio > 0.8 + 1e-3:
        bad.append(f"modulation {ratio:.3f} > |eps_r|")
    ok = not bad and 0.0 <= wmin < 1.0
    return ok, ", ".join(sorted(set(bad))) or (
        f"bow out 0; min w+ / w {wmin:.3f} (in [0, 1)); N = x_b Y; "
        f"modulation ratio {ratio:.3f} <= 0.8")


def _sweep(item, prm, ctx, u_path, dt=0.2):
    st = item.init_state(prm, 1)
    X, Fn = [], []
    for u in u_path:
        q = _q(ctx, _run(ctx, [u]))
        X.append(item.step(prm, st, q, dt).Q[0, 0])
        Fn.append(q["Fn_vol"][0])
    return np.array(X), np.array(Fn)


@CB.check("W11")
def w11_drag_only_and_hysteresis(item, ctx):
    """Drag only, peak r_h W; bow-up and sinkage at the peak; with the
    hysteresis switch the resistance peak sits dF higher on a slow
    acceleration than on the deceleration."""
    rng = np.random.default_rng(26)
    bad = []
    for _ in range(20):
        prm = item.draw_on(rng, ctx)
        prm.update(hyst=False, dF=0.0)
        u = np.linspace(0.2, 14.0, 300)
        X = item.step(prm, item.init_state(prm, 300), _q(ctx, _run(ctx, u)),
                      CB.DT_SUB).Q[:, 0]
        if (X * u > 0).any() or np.abs(X).max() > prm["r_h"] * ctx.W + 1e-9:
            bad.append("drag sign or peak")
        prm["r_h"] = max(prm["r_h"], 0.01)
        q = _q(ctx, _run(ctx, [prm["F_h"] * np.sqrt(
            CB.GRAV * ctx.vol ** (1 / 3))]))
        a = item.step(prm, item.init_state(prm, 1), q, CB.DT_SUB).acc[0]
        if a[3] > 0 or ctx.sp * a[4] < 0:
            bad.append("trim / sinkage sign")
    prm = item.draw_on(rng, ctx)
    prm.update(hyst=True, dF=0.3, tau_s=1.0, asym=False, f_lo=1.0,
               f_hi=1.0, r_h=0.1)
    un = np.sqrt(CB.GRAV * ctx.vol ** (1 / 3))
    u = np.linspace(0.2, (prm["F_h"] + 4 * prm["w_h"]) * un, 1500)
    Xu, Fu = _sweep(item, prm, ctx, np.concatenate([u, u[::-1]]), dt=0.4)
    n = len(u)
    f_up, f_dn = Fu[np.argmin(Xu[:n])], Fu[n + np.argmin(Xu[n:])]
    d = f_up - f_dn
    if abs(d - prm["dF"]) > 0.03:
        bad.append(f"hysteresis offset {d:.3f}")
    return not bad, ", ".join(sorted(set(bad))) or (
        f"drag only, peak <= r_h W, bow-up + sinkage; peak Fn up "
        f"{f_up:.3f} / down {f_dn:.3f} (dF {prm['dF']:.2f})")


@CB.check("W12")
def w12_zero_symmetric_hook(item, ctx):
    """Calm running above the ventilation speed: exactly zero; below it
    (collapsed): straight running gives no sway / yaw and only drag in
    surge; the port side immersed deeper (roll) collapses that side alone;
    its drag difference (c_y = 0) yaws the bow to port, towards it (the
    hook); c_y > 0 pushes to starboard; a dry step: zero."""
    rng = np.random.default_rng(27)
    bad = []
    for _ in range(20):
        prm = item.draw_on(rng, ctx)
        prm.update(k_beta=0.0)
        uh = prm["U_v"] + prm["h_U"] + 1.0
        ul = max(prm["U_v"] - prm["h_U"] - 1.0, 1.0)
        for u, zero in ((uh, True), (ul, False)):
            st = item.init_state(prm, 1)
            q = _q(ctx, _run(ctx, [u]))
            for _k in range(10):
                Q = item.step(prm, st, q, CB.DT_SUB).Q[0]
            if zero and np.abs(Q).max() != 0.0:
                bad.append("not zero when ventilated")
            if not zero and (abs(Q[1]) + abs(Q[2]) > 1e-9 or Q[0] > 1e-9):
                bad.append("symmetric collapse")
        pr = dict(prm, dh_v=0.2 * prm["h_run"], w_v=0.2 * prm["h_run"],
                  c_y=0.0, r_c=0.03, tau_loss=0.05, tau_rec=1.0)
        st = item.init_state(pr, 1)
        q = _q(ctx, _run(ctx, [uh]))
        q["phi"] = np.array([-0.1])                  # port side down
        CB.derive_quantities(ctx, q)
        for _k in range(50):
            o = item.step(pr, st, q, CB.DT_SUB)
        a = CB.to_acc(ctx, o.Q)[0]
        if not (st["v"][0, 1] < 0.01 < 0.99 < st["v"][0, 0]):
            bad.append("one-sided collapse")
        if not a[2] > 0:
            bad.append(f"hook yaw {a[2]:.3g} not towards port")
        pr["c_y"] = 0.2
        Qy = item.step(pr, st, q, CB.DT_SUB).Q[0]
        if not Qy[1] < 0:
            bad.append("lateral force sign")
        qd = _q(ctx, _run(ctx, [ul], 1.5))
        if np.abs(item.step(pr, item.init_state(pr, 1), qd,
                            CB.DT_SUB).Q).max() != 0.0:
            bad.append("dry step")
    return not bad, ", ".join(sorted(set(bad))) or (
        "zero when ventilated, symmetric collapse drag only, one-sided "
        "collapse yaws towards the wetted side, dry step zero (20 draws)")

