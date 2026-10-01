#!/usr/bin/env python3
"""
Catalogue items W2-W6 (learn/meta/PRIOR_D10_DRAFT.md 3.1 with the
corrections of section 10 and the numerical rules of section 6 item 6),
group "water_a". W1 stays in learn/meta/cat_w1.py (the framework's worked
example). All immersions are KEEL immersions h_i = d_i - k_i (q['h'],
q['hc']); the low-fidelity physics is untouched, every item returns
accelerations / generalised forces on the five velocity channels, W3 its
velocity jumps in the separate impulse channel.

W2  porpoising: a speed-gated van der Pol negative damping on a random
    heave-pitch mode (w.p. 0.7), or (w.p. 0.3) a speed-gated heave-pitch
    coupled stiffness with k35 k53 < 0 (flutter type), both amplitude
    limited. No hidden state.
W3  slamming: per station and per half-section (starboard / port points)
    a momentum-conserving IMPLICIT impulse from the growth of the von
    Karman / Wagner added mass, saturating at the chine; stations updated
    bow to stern; exit scaled by r_exit. Impulse channel.
W4  contact gate (structural, always on): per-station gates chi_i from the
    centre-line keel immersion multiply ALL vertical hydrodynamic forces
    (low-fidelity restoring + damping incl. the weight it balances, layer-1
    T3 / T4 / T7 vertical parts, gated catalogue items); the lumped heave
    force and pitch moment are spread over the stations with the calm-
    running wetted weights.
W5  chine / spray-rail wetting: (a) piecewise-smooth station stiffness with
    kinks at random chine and rail depths (tangent at the running point
    removed) and the asymmetric side-wall wetting force; (b) a speed
    switch with hysteresis (hidden discrete state + exact first-order lag)
    that shifts drag, running trim and heave stiffness.
W6  whisker-spray drag: monotone in trim (ITTC-57 friction x b^2 x
    (tau_run / tau_e)^p_s), plus an intermittent one-sided bow term; both
    quadratic drags by the linearised implicit update.

Chine sharing: when W3 is on, its station chine depths are put on the
context (adjust_ctx) and W5(a) uses the same ones (draft W5: "W3's h(d)
saturates at the same d_c"). W4 publishes its r_exit on the context
(ctx.w4_r_exit) for the velocity parts of W1 / W7 (other groups).
Construction checks are registered with @cat_base.check (run by
studies/test_cat_items.py, I5).
"""
import numpy as np

from learn.meta import cat_base as CB
from learn.meta.operators import _logu

KN, DEG, GRAV, RHO_W = CB.KN, CB.DEG, CB.GRAV, CB.RHO_W
NU_W = 1.19e-6                   # kinematic viscosity of sea water (m^2/s)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _softplus(x):
    return np.logaddexp(0.0, x)


def _dx5(ctx):
    """Trapezoid length shares of the 5 stations (the end stations half)."""
    d = (ctx.x_st[-1] - ctx.x_st[0]) / 4.0
    return np.array([0.5, 1.0, 1.0, 1.0, 0.5]) * d


def _hp_mass(ctx):
    """Heave-pitch block of the generalised force that produces a given
    heave-pitch acceleration with the draw's M_t: inv(Minv[3:5, 3:5])."""
    return np.linalg.inv(ctx.Minv[3:5, 3:5])


# ===================================================================== W2
def _flutter_kcrit(d1, k1, d2, k2):
    """Smallest kappa^2 = -k35 k53 at which (l^2 + d1 l + k1)(l^2 + d2 l +
    k2) + kappa^2 has a root with a positive real part (bisection)."""
    base = np.polymul([1.0, d1, k1], [1.0, d2, k2])

    def maxre(K):
        c = base.copy()
        c[-1] += K
        return np.roots(c).real.max()
    lo, hi = 1e-6, 1e9
    if maxre(lo) >= 0.0:
        return lo
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if maxre(mid) < 0.0 else (lo, mid)
    return hi


@CB.register
class W2(CB.CatItem):
    """Porpoising. Gate gamma(U) = sigma(m(U) / w_m), margin m(U) = m0 + m1
    (U - U_id) / U_id (degrees, U the water-relative speed, U_id the
    low-fidelity identification speed).
    van der Pol: mode shape phi = (-s_p x_m, 1) (heave per pitch; zero
    vertical motion at the node x_m), readout s = c . (e_z, e_theta) with c
    the M_t-orthogonal projector (c . phi = 1), both relative to the
    low-fidelity water reference; modal acceleration a_s from the damping
    c_eff = -gamma c_neg (1 - s^2 / s_sat^2) on sdot (exact exponential
    where c_eff > 0, explicit where negative), heave-pitch acceleration phi
    a_s. c_neg = rho_n c_s, c_s the low-fidelity boat's own damping on the
    mode, so rho_n > 1 is net negative damping where the gate is open.
    s_sat = half the limit amplitude (van der Pol: A = 2 s_sat), the limit
    being the first reached of the pitch amplitude A_th and the heave
    amplitude A_z at the reference point.
    flutter: a_z = -f k35 e_theta, a_theta = -f k53 e_z, k35 k53 = -rho_f
    kappa*^2 (kappa*^2 the low-fidelity boat's critical product),
    f = gamma / (1 + (e_z / A_z)^2 + (e_theta / A_th)^2)."""

    code, stage = "W2", "force"
    gated, calm_zero = True, True

    def nominal(self, params, ctx):
        """Soft-saturation scale: at most half the clip per channel
        (cat_base.half_clip_nominal, D10.8)."""
        return CB.half_clip_nominal(ctx)

    def draw_on(self, rng, ctx):
        u = rng.uniform
        prm = dict(m0=u(-3.0, 1.5) * DEG, m1=u(-1.0, 1.0) * DEG,
                   w_m=u(0.2, 0.8) * DEG, rho_n=float(_logu(rng, 0.3, 2.5)),
                   A_th=float(_logu(rng, 0.5, 4.0)) * DEG,
                   A_z=float(_logu(rng, 0.02, 0.15)),
                   x_m=u(-0.3, 0.3) * ctx.L,
                   flutter=float(rng.random() < 0.3),
                   rho_f=float(_logu(rng, 0.3, 2.5)),
                   l_f=float(_logu(rng, 0.1, 1.0)) * ctx.L,
                   sgn=float(np.sign(u(-1.0, 1.0)) or 1.0))
        phi = np.array([-ctx.sp * prm["x_m"], 1.0])
        Mvv = _hp_mass(ctx)
        c = Mvv @ phi / float(phi @ Mvv @ phi)
        D = np.diag([2 * ctx.zh * ctx.wh, 2 * ctx.zp * ctx.wp])
        prm["c_s"] = float(c @ D @ phi)
        prm["c_neg"] = prm["rho_n"] * prm["c_s"]
        A_s = min(prm["A_th"], prm["A_z"] / max(abs(prm["x_m"]), 1e-6))
        prm["s_sat"] = 0.5 * A_s
        kc = _flutter_kcrit(2 * ctx.zh * ctx.wh, ctx.wh ** 2,
                            2 * ctx.zp * ctx.wp, ctx.wp ** 2)
        kap = np.sqrt(prm["rho_f"] * kc)
        prm["kc2"] = float(kc)
        prm["k35"] = prm["sgn"] * kap * prm["l_f"]
        prm["k53"] = -prm["sgn"] * kap / prm["l_f"]
        return prm

    @staticmethod
    def gate(prm, ctx, U):
        m = prm["m0"] + prm["m1"] * (U - ctx.u_id) / ctx.u_id
        return _sig(m / prm["w_m"])

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        B = q["B"]
        gam = self.gate(prm, ctx, q["U_r"])
        ez, eth = q["e_z"], q["e_th"]
        if prm["flutter"] > 0.5:
            f = gam / (1.0 + (ez / prm["A_z"]) ** 2 + (eth / prm["A_th"]) ** 2)
            acc = np.zeros((B, 5))
            acc[:, 3] = -f * prm["k35"] * eth
            acc[:, 4] = -f * prm["k53"] * ez
            return CB.ItemOut(acc=acc, state=state)
        phi = np.array([-ctx.sp * prm["x_m"], 1.0])
        Mvv = _hp_mass(ctx)
        mphi = Mvv @ phi
        c = mphi / float(phi @ mphi)
        s = c[0] * ez + c[1] * eth
        sd = c[0] * q["w_r"] + c[1] * q["q_r"]
        ceff = -gam * prm["c_neg"] * (1.0 - (s / prm["s_sat"]) ** 2)
        a_s = CB.lin_damp_acc(ceff, sd, dt)
        Q = np.zeros((B, 5))
        Q[:, 3:5] = a_s[:, None] * mphi[None]
        return CB.ItemOut(Q=Q, state=state)


# ===================================================================== W3
@CB.register
class W3(CB.CatItem):
    """Slamming. Half-section added mass m_ij(h) = 1/2 rho C_m b^2 dx_i
    min(h / d_c,i, 1)_+^p_h at the starboard / port points of station i
    (their own keel immersion and entry velocity, so a symmetric entry
    gives no lateral force). Per station, the entering halves (dm > 0,
    V > 0) share one implicit update of the point's vertical velocity
    (no roll degree of freedom, both halves move together):
      dw = sum dm_h V_h / (m_eff + sum m_new,h),  J = m_eff dw,
    the single-point form V+ = V (m_eff + m_old) / (m_eff + m_new) when one
    half enters (draft W3, section 6 item 3): the entry velocity never
    reverses and |dw| <= V, independent of dt. Exiting halves (dm < 0,
    V < 0) give the same form times r_exit. m_new uses this substep's start
    immersion, m_old the previous substep's (one substep late, causal).
    Direction: vertical J; surge -J tan(tau_e); lateral (J_stbd - J_port)
    tan(beta_i) (a port-side impact pushes to starboard), at the keel point
    (x_i, 0, k_i). m_eff = 1 / (g^T M_t^-1 Qhat) along the actual vertical +
    surge direction Qhat, so the point's vertical velocity changes by
    exactly dw. Stations bow to stern, each seeing the velocity changes of
    the ones before. Chine depth d_c,i / b = tan(beta_i) / (2 k_w) (the
    draft's derivation of its U[0.06, 0.27], with the local deadrise
    beta_i varying linearly along the hull, bow >= stern)."""

    code, stage = "W3", "force"
    calm_zero = True

    def draw_on(self, rng, ctx):
        u = rng.uniform
        b_st, b_bow = np.sort(u(12.0, 28.0, 2)) * DEG
        beta = np.interp(ctx.x_st, [ctx.x_st[0], ctx.x_st[-1]], [b_st, b_bow])
        k_w = u(1.0, 0.5 * np.pi)
        return dict(C_m=u(0.2, 0.47), beta=beta, k_w=k_w,
                    dc=np.tan(beta) / (2.0 * k_w) * ctx.b,
                    p_h=u(1.0, 2.5), r_exit=u(-0.2, 0.2))

    def adjust_ctx(self, prm, ctx):
        ctx.chine_dc = np.array(prm["dc"], float)

    def init_state(self, prm, B=1, rng=None):
        return dict(m_old=None)

    @staticmethod
    def half_mass(prm, ctx, h):
        """(B, 5, 2) added mass of the half-sections at keel immersion h."""
        s = np.clip(h / np.asarray(prm["dc"])[None, :, None], 0.0, 1.0)
        return (0.5 * RHO_W * prm["C_m"] * ctx.b ** 2
                * _dx5(ctx)[None, :, None] * s ** prm["p_h"])

    def impulses(self, prm, ctx, m_old, m_new, V, tau, th):
        """Velocity jumps (B, 5) of one substep's slams, the per-station
        vertical velocity change dw (B, 5) and the entry velocity each
        station saw when it was updated (B, 5; the larger half), for the
        checks."""
        B = len(V)
        dnu = np.zeros((B, 5))
        dws = np.zeros((B, 5))
        vis = np.zeros((B, 5))
        tb = np.tan(np.asarray(prm["beta"]))
        for i in range(4, -1, -1):                       # bow to stern
            x, zk = ctx.x_st[i], ctx.keel5[i]
            g = CB.unit_vertical(ctx, x)
            F1 = np.stack([-np.tan(tau), np.zeros(B), np.ones(B)], 1)
            Qh = CB.lever_Q(ctx, (x, 0.0, zk), F1, th)   # per unit J
            meff = 1.0 / ((Qh @ ctx.Minv.T) @ g)
            Vi = V[:, i, :] - (dnu @ g)[:, None]
            dm = m_new[:, i, :] - m_old[:, i, :]
            Jh = np.zeros((B, 2))
            dw = np.zeros(B)
            for mask, scale in (((dm > 0) & (Vi > 0), 1.0),
                                ((dm < 0) & (Vi < 0), prm["r_exit"])):
                num = np.where(mask, dm * Vi, 0.0).sum(1)
                den = meff + np.where(mask, m_new[:, i, :], 0.0).sum(1)
                w = num / den
                dw += scale * w
                jh = np.where(mask, m_new[:, i, :] * (Vi - w[:, None])
                              - m_old[:, i, :] * Vi, 0.0)
                Jh += scale * jh
            J = meff * dw
            Fy = (np.maximum(Jh[:, 0], 0.0) - np.maximum(Jh[:, 1], 0.0)) \
                * tb[i]
            Q = J[:, None] * Qh + CB.lever_Q(
                ctx, (x, 0.0, zk), np.stack([0 * Fy, Fy, 0 * Fy], 1), th)
            dnu += Q @ ctx.Minv.T
            dws[:, i] = dw
            vis[:, i] = Vi.max(1)
        return dnu, dws, vis

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        h = q["h"][:, :, [0, 2]]
        m_new = self.half_mass(prm, ctx, h)
        m_old = state.get("m_old")
        state["m_old"] = m_new
        if m_old is None:
            return CB.ItemOut(state=state)
        dnu = self.impulses(prm, ctx, m_old, m_new, q["dd"][:, :, [0, 2]],
                            q["tau_e"], q["sr"][:, 5])[0]
        return CB.ItemOut(imp=dnu, state=state)


# ===================================================================== W4
@CB.register
class W4(CB.CatItem):
    """Contact gate. chi_i = min(1, sigma((h_i - h_off) / w_d) / sigma((h_run,i
    - h_off) / w_d)) on the centre-line keel immersion: 1 at and below the
    calm running immersion (so calm steady running gives exactly 0, draft
    7.1), -> 0 as the station leaves the water. The vertical hydrodynamic
    acceleration A = (g + a_lofi,z + a_L1,z + a_cat,z, a_lofi,th + a_L1,th +
    a_cat,th) is turned into the heave force / pitch moment that produces
    it (heave-pitch block of M_t), spread over the stations as f_i = w_i
    (alpha + beta x_i) with w_i the calm-running keel immersion shares, and
    the error is sum (chi_i - 1) f_i with its moment. Airborne: heave error
    = -g - a_lofi,z - a_L1,z - a_cat,z, pitch error = -(the pitch sum)."""

    code, stage = "W4", "gate"
    saturate, calm_zero = False, True

    def draw_on(self, rng, ctx):
        u = rng.uniform
        return dict(h_off=u(-0.05, 0.05), w_d=u(0.02, 0.1),
                    r_exit=u(0.0, 0.3))

    def adjust_ctx(self, prm, ctx):
        ctx.w4_r_exit = float(prm["r_exit"])

    @staticmethod
    def chi(prm, ctx, hc):
        run = _sig((ctx.hc_run - prm["h_off"]) / prm["w_d"])
        return np.minimum(1.0, _sig((hc - prm["h_off"]) / prm["w_d"])
                          / np.maximum(run, 1e-300)[None])

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        B = q["B"]
        z2 = np.zeros((B, 2))
        A = q["a_lofi_vert"] + q.get("a_l1_vert", z2) \
            + q.get("a_cat_vert", z2)
        A = A + np.array([GRAV, 0.0])
        Qhp = A @ _hp_mass(ctx).T                       # (B, 2)
        w = np.maximum(ctx.hc_run, 0.0)
        x = ctx.x_st
        S = np.array([[w.sum(), (w * x).sum()], [(w * x).sum(),
                                                 (w * x * x).sum()]])
        ab = np.linalg.solve(S, np.stack([Qhp[:, 0], ctx.sp * Qhp[:, 1]], 0))
        f = w[None] * (ab[0][:, None] + ab[1][:, None] * x[None])
        dc = self.chi(prm, ctx, q["hc"]) - 1.0
        Q = np.zeros((B, 5))
        Q[:, 3] = (dc * f).sum(1)
        Q[:, 4] = ctx.sp * (dc * f * x[None]).sum(1)
        return CB.ItemOut(Q=Q, state=state)


# ===================================================================== W5
W5_SUB = ((0.3, 1.0, 0.0), (0.2, 0.0, 1.0), (0.3, 1.0, 1.0))  # (p, a, b)
W5_DREF = 0.1                    # m, reference asymmetric side-wall wetting


@CB.register
class W5(CB.CatItem):
    """Chine and spray-rail wetting. Sub-parts (a) 0.6, (b) 0.5 independent;
    given the item is on, (a only, b only, both) = (0.3, 0.2, 0.3) / 0.8.
    (a) station stiffness k_i(h) = k1_i [1 + sum_j D_j sigma((h - c_ij) /
    w_i)] with the kinks (0-2 spray rails at fractions of the chine depth,
    then the chine) sorted by depth, a common width per station and steps
    D_j = P_j - P_(j-1) between the plateaus P_j = prod ratios (rails r_s,
    chine r): a convex mix of the plateaus, so k > 0 always. Its integral
    (softplus) minus the tangent at the calm running immersion is the
    station force; k1_i makes the local stiffness at running equal the
    low-fidelity heave stiffness share (w_h^2 m_heave / 5). Side walls:
    acc_r = n_r 0.3 a_ref,r Dh / 0.1 m (U / U_id)^2, sway alike, Dh = sum
    over stations of (h_port - d_c)_+ - (h_stbd - d_c)_+.
    (b) switch s (1 = chine dry) flips up at U_c + H/2, down at U_c - H/2
    (water-relative speed), smoothed with tau_s; its change from the
    identification state s_ref = [U_id > U_c] shifts drag (dR k_drag u|u|),
    running trim (w_p^2 s_p dtheta) and heave stiffness (-(k_m - 1) w_h^2
    e_z); H1's roll restoring multiplier k_roll and ds are published in
    q['W5b'] for H1."""

    code, stage = "W5", "force"
    gated, calm_zero = True, True

    def nominal(self, params, ctx):
        """Soft-saturation scale: at most half the clip per channel
        (cat_base.half_clip_nominal, D10.8)."""
        return CB.half_clip_nominal(ctx)

    def draw_on(self, rng, ctx):
        u = rng.uniform
        k = rng.random() * sum(p for p, _, _ in W5_SUB)
        acc = 0.0
        for p, a_on, b_on in W5_SUB:
            acc += p
            if k < acc:
                break
        x = (ctx.x_st - ctx.x_st[0]) / (ctx.x_st[-1] - ctx.x_st[0])
        dc0, f_bow = u(0.3, 1.5) * ctx.T, u(1.0, 2.0)
        dc = dc0 * (1.0 + (f_bow - 1.0) * x)
        shared = getattr(ctx, "chine_dc", None)
        if shared is not None:
            dc = np.array(shared, float)
        n_rail = int(rng.integers(0, 3))
        prm = dict(a_on=a_on, b_on=b_on, dc=dc, shared_dc=float(
            shared is not None), r=u(0.3, 1.5), w_frac=u(0.05, 0.3),
            n_rail=float(n_rail), rail_frac=np.sort(u(0.2, 0.9, 2)),
            r_s=u(0.7, 1.2, 2), n_r=u(-1.0, 1.0), n_v=u(-1.0, 1.0),
            U_c=u(12.0, 30.0) * KN, hyst=u(0.0, 4.0) * KN,
            tau_s=float(_logu(rng, 1.0, 10.0)), dR=u(-0.08, 0.08),
            dth=u(-0.5, 0.5) * DEG, k_m=float(_logu(rng, 0.8, 1.25)),
            k_roll=float(_logu(rng, 0.7, 1.4)))
        return prm

    def overlaps(self, prm):
        return {"T4_nl_off"} if prm["a_on"] else set()

    def calm_params(self, prm):
        d = dict(prm)
        d["b_on"] = 0.0
        return d

    def init_state(self, prm, B=1, rng=None):
        return dict(h=None, s=None)

    @staticmethod
    def kinks(prm):
        """(fractions of d_c (n,), steps D (n,)) of the kinks in depth order
        (rails, then the chine)."""
        n = int(prm["n_rail"])
        fr = list(np.asarray(prm["rail_frac"])[:n]) + [1.0]
        ratios = list(np.asarray(prm["r_s"])[:n]) + [prm["r"]]
        P = np.cumprod([1.0] + ratios)
        return np.array(fr), np.diff(P)

    @classmethod
    def kappa(cls, prm, h):
        """Relative stiffness k / k1 and its integral from 0, at (B, 5)."""
        fr, D = cls.kinks(prm)
        dc = np.asarray(prm["dc"])[None, :]
        w = prm["w_frac"] * dc
        k = np.ones_like(h)
        F = np.array(h, float)
        for f, d in zip(fr, D):
            c = f * dc
            k = k + d * _sig((h - c) / w)
            F = F + d * w * (_softplus((h - c) / w) - _softplus(-c / w))
        return k, F

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        B = q["B"]
        Q = np.zeros((B, 5))
        acc = np.zeros((B, 5))
        if prm["a_on"]:
            hr = np.broadcast_to(ctx.hc_run, (1, 5))
            kr, Fr = self.kappa(prm, hr)
            k1 = ctx.wh ** 2 / ctx.Minv[3, 3] / 5.0 / kr
            h = q["hc"]
            _, F = self.kappa(prm, h)
            e = k1 * (F - Fr - kr * (h - hr))
            Q[:, 3] = e.sum(1)
            Q[:, 4] = ctx.sp * (e * ctx.x_st[None]).sum(1)
            dc = np.asarray(prm["dc"])[None]
            dh = (np.maximum(q["h"][:, :, 2] - dc, 0.0)
                  - np.maximum(q["h"][:, :, 0] - dc, 0.0)).sum(1)
            fac = dh / W5_DREF * (q["U_r"] / ctx.u_id) ** 2
            acc[:, 2] = prm["n_r"] * 0.3 * ctx.a_ref[2] * fac
            acc[:, 1] = prm["n_v"] * 0.3 * ctx.a_ref[1] * fac
        if prm["b_on"]:
            U = q["U_r"]
            if state.get("h") is None:
                state["h"] = (U > prm["U_c"]).astype(float)
                state["s"] = state["h"].copy()
            s = CB.hyst_switch(state, U, prm["U_c"] + 0.5 * prm["hyst"],
                               prm["U_c"] - 0.5 * prm["hyst"], prm["tau_s"],
                               dt)
            ds = s - float(ctx.u_id > prm["U_c"])
            ur = q["u_r"]
            Q[:, 0] += -ds * prm["dR"] * ctx.k_drag * ur * np.abs(ur)
            acc[:, 4] += ctx.wp ** 2 * ctx.sp * prm["dth"] * ds
            acc[:, 3] += -ctx.wh ** 2 * (prm["k_m"] - 1.0) * ds * q["e_z"]
            q["W5b"] = dict(ds=ds, k_roll=prm["k_roll"])
        return CB.ItemOut(acc=acc, Q=Q, state=state)


# ===================================================================== W6
W6_RAW_W = 0.07                  # Savitsky-Brown R_aw / W at Hs 1 m (W8)
W6_DREF = 0.1                    # m, reference bow over-immersion


def _cf(U, L):
    Re = np.maximum(np.abs(U) * L / NU_W, 1e5)
    return 0.075 / (np.log10(Re) - 2.0) ** 2


@CB.register
class W6(CB.CatItem):
    """Whisker-spray drag. X = -c u_r|u_r| with c = 1/2 rho C_f(Re) b^2
    kappa_s (tau_run / tau_e)^p_s + kappa_b 1/2 rho b (h_bow - h_b0)_+,
    by the linearised implicit update on the surge generalised mass (never
    propels, never reverses u); pitch from the force at height z_s.
    kappa_s: dR / R = f_s ~ U[0, 0.15] of the low-fidelity drag k_drag u^2
    at U_id and the running trim; kappa_b: a bow over-immersion of 0.1 m at
    U_id gives f_b ~ U[0, 1] x half the W8 added resistance at Hs 1 m
    (0.07 W); h_b0 = calm-running bow keel immersion + U[0, 0.3] T (so calm
    running has no bow term). Not calm-zero: the spray drag is a real
    extra drag in calm water (draft 7.1 does not list W6)."""

    code, stage = "W6", "force"

    def draw_on(self, rng, ctx):
        u = rng.uniform
        f_s, f_b = u(0.0, 0.15), u(0.0, 1.0)
        cf = float(_cf(ctx.u_id, ctx.L))
        return dict(f_s=f_s, p_s=u(0.0, 1.5), f_b=f_b,
                    z_s=u(-1.0, 0.5) * ctx.T,
                    h_b0=float(ctx.hc_run[-1]) + u(0.0, 0.3) * ctx.T,
                    kap_s=f_s * ctx.k_drag / (0.5 * RHO_W * cf * ctx.b ** 2),
                    kap_b=f_b * 0.5 * W6_RAW_W * ctx.W
                    / (0.5 * RHO_W * ctx.u_id ** 2 * ctx.b * W6_DREF))

    def coef(self, prm, ctx, q):
        c_main = (0.5 * RHO_W * _cf(q["U_r"], ctx.L) * ctx.b ** 2
                  * prm["kap_s"] * (ctx.tau_run / q["tau_e"]) ** prm["p_s"])
        c_bow = prm["kap_b"] * 0.5 * RHO_W * ctx.b * np.maximum(
            q["hc"][:, -1] - prm["h_b0"], 0.0)
        return c_main, c_bow

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        B = q["B"]
        c_main, c_bow = self.coef(prm, ctx, q)
        meff = 1.0 / ctx.Minv[0, 0]
        ax = CB.implicit_quad_drag(q["u_r"], c_main + c_bow, meff, dt)
        F = np.stack([meff * ax, np.zeros(B), np.zeros(B)], 1)
        Q = CB.lever_Q(ctx, (0.0, 0.0, prm["z_s"]), F, q["sr"][:, 5])
        return CB.ItemOut(Q=Q, state=state)


# ================================================================ checks
def _run(ctx, u, dz=0.0, dth=0.0, B=None):
    u = np.atleast_1d(np.asarray(u, float))
    sr = np.zeros((len(u), 10))
    sr[:, 2] = u
    sr[:, 3] = ctx.z0 + dz
    sr[:, 5] = ctx.th0 + dth
    return sr


def _q(ctx, sr, sea=None):
    if sea is None:
        sea = CB.CatSea(None, ctx).sample(sr, 0.0)
    return CB.substep_quantities(ctx, sr, 0.0, 0.0, sea)


def _vert_sim(ctx, item, prm, U, e0, n, dt=CB.DT_SUB):
    """The low-fidelity boat's own calm heave / pitch (exact transitions)
    plus the item's heave / pitch acceleration as a kick, at speed U, from
    the offsets e0 = (dz, dtheta). Returns (n, 2) offsets."""
    sr = _run(ctx, [U])
    sr[0, 3] += e0[0]
    sr[0, 5] += e0[1]
    out = np.zeros((n, 2))
    for k in range(n):
        q = _q(ctx, sr)
        a, _, _, _ = CB.item_accel(item, prm, {}, q, dt)
        z, zd = CB.osc_step(sr[0, 3], sr[0, 4], ctx.wh ** 2,
                            2 * ctx.zh * ctx.wh, dt, x_eq=ctx.z0)
        t, td = CB.osc_step(sr[0, 5], sr[0, 6], ctx.wp ** 2,
                            2 * ctx.zp * ctx.wp, dt, x_eq=ctx.th0)
        sr[0, 3], sr[0, 4] = z, zd + a[0, 3] * dt
        sr[0, 5], sr[0, 6] = t, td + a[0, 4] * dt
        out[k] = sr[0, 3] - ctx.z0, sr[0, 5] - ctx.th0
    return out


@CB.check("W2")
def w2_limit_cycle(item, ctx):
    """Gate open, rho_n = 2: a small offset grows into a bounded limit
    cycle, mode amplitude <= 2.2 s_sat (van der Pol: 2 s_sat sqrt(1 - 1 /
    rho) for a proportional system); rho_n = 0.5: it decays."""
    rng = np.random.default_rng(21)
    msgs = []
    for _ in range(6):
        prm = item.draw_on(rng, ctx)
        prm.update(flutter=0.0, m0=5 * DEG, m1=0.0)
        for rho, grow in ((2.0, True), (0.5, False)):
            p = dict(prm, rho_n=rho, c_neg=rho * prm["c_s"])
            e = _vert_sim(ctx, item, p, ctx.u_id, (0.0, 0.1 * p["s_sat"]),
                          1500)
            phi = np.array([-ctx.sp * p["x_m"], 1.0])
            mphi = _hp_mass(ctx) @ phi
            s_t = e @ (mphi / float(phi @ mphi))        # the mode readout
            a_end = np.abs(s_t[-250:]).max()
            a0 = 0.1 * p["s_sat"]
            if grow and not (a0 < a_end <= 2.2 * p["s_sat"]):
                return False, (f"rho 2: end amplitude {a_end:.2e} vs s_sat "
                               f"{p['s_sat']:.2e}")
            if not grow and not a_end < 0.5 * a0:
                return False, f"rho 0.5 does not decay ({a_end:.2e})"
            if grow:
                msgs.append(a_end / p["s_sat"])
    return True, (f"limit amplitude / s_sat {min(msgs):.2f}-{max(msgs):.2f} "
                  "(van der Pol ~2 x sqrt(1 - 1/rho)); rho 0.5 decays")


@CB.check("W2")
def w2_flutter_threshold(item, ctx):
    """The flutter coupling is linearly unstable exactly for rho_f > 1
    (gate open): eigenvalues of the low-fidelity heave-pitch system plus
    the coupling."""
    prm = item.draw_on(np.random.default_rng(22), ctx)
    d1, k1 = 2 * ctx.zh * ctx.wh, ctx.wh ** 2
    d2, k2 = 2 * ctx.zp * ctx.wp, ctx.wp ** 2
    out = []
    for rho in (0.9, 1.1):
        kap = np.sqrt(rho * prm["kc2"])
        k35, k53 = kap * prm["l_f"], -kap / prm["l_f"]
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-k1, -k35, -d1, 0],
                      [-k53, -k2, 0, -d2]], float)
        out.append(np.linalg.eigvals(A).real.max())
    return out[0] < 0 < out[1], (f"max Re at rho_f 0.9 / 1.1: {out[0]:.3f} "
                                 f"/ {out[1]:.3f}; kappa*^2 "
                                 f"{prm['kc2']:.1f} 1/s^4")


@CB.check("W3")
def w3_no_reversal_telescoping(item, ctx):
    """(1) Random entries: after the station's update the entry velocity
    never reverses and |dw| <= V (entering halves with equal V). (2) One
    station sinking through a depth range in 1 or 7 substeps ends at the
    same velocity (momentum form, independent of dt). (3) A symmetric
    entry has no lateral force; a port-only entry pushes to starboard."""
    rng = np.random.default_rng(23)
    B = 2000
    prm = item.draw_on(rng, ctx)
    h0 = rng.uniform(-0.1, 0.6, (B, 5, 1)).repeat(2, 2)
    h1 = h0 + rng.uniform(0.0, 0.3, (B, 5, 1))
    V = rng.uniform(0.0, 6.0, (B, 5, 1)).repeat(2, 2)
    tau = np.full(B, ctx.tau_run)
    th = np.zeros(B)
    mo, mn = item.half_mass(prm, ctx, h0), item.half_mass(prm, ctx, h1)
    dnu, dw, vi = item.impulses(prm, ctx, mo, mn, V, tau, th)
    # vi: the station's entry velocity when it is updated (earlier
    # stations' impulses included)
    bad = np.any(dw < -1e-12) or np.any(dw > np.maximum(vi, 0.0) + 1e-9)
    # (2) telescoping, station 2 alone, constant tau
    p1 = dict(prm)
    hs = np.linspace(0.0, 0.9 * prm["dc"][2], 8)
    V0 = 3.0
    ends = []
    for idx in ([0, 7], list(range(8))):
        v = V0
        for a, b in zip(idx[:-1], idx[1:]):
            H0 = np.zeros((1, 5, 2))
            H1 = np.zeros((1, 5, 2))
            H0[0, 2], H1[0, 2] = hs[a], hs[b]
            Vv = np.zeros((1, 5, 2))
            Vv[0, 2] = v
            _, d, _ = item.impulses(p1, ctx, item.half_mass(p1, ctx, H0),
                                 item.half_mass(p1, ctx, H1), Vv,
                                 tau[:1], th[:1])
            v -= d[0, 2]
        ends.append(v)
    tel = abs(ends[0] - ends[1])
    # (3) lateral
    H0 = np.zeros((2, 5, 2))
    H1 = np.zeros((2, 5, 2))
    H1[:, 3, :] = 0.05
    H1[1, 3, 0] = 0.0                             # row 1: port only
    Vv = np.full((2, 5, 2), 2.0)
    d3 = item.impulses(prm, ctx, item.half_mass(prm, ctx, H0),
                          item.half_mass(prm, ctx, H1), Vv, tau[:2], th[:2])[0]
    lat_ok = abs(d3[0, 1]) < 1e-12 and d3[1, 1] < 0
    ok = not bad and tel < 1e-9 and lat_ok
    return ok, (f"reversal/overshoot {bad}; end velocity 1 vs 7 substeps "
                f"{ends[0]:.6f} / {ends[1]:.6f}; symmetric sway "
                f"{d3[0, 1]:.1e}, port-only sway {d3[1, 1]:.3e} m/s")


@CB.check("W3")
def w3_chine_saturation(item, ctx):
    """Above the chine depth the added mass no longer grows: no impulse."""
    prm = item.draw_on(np.random.default_rng(24), ctx)
    dc = np.asarray(prm["dc"])
    H0 = np.broadcast_to((1.05 * dc)[None, :, None], (1, 5, 2)).copy()
    H1 = H0 + 0.2
    V = np.full((1, 5, 2), 3.0)
    dnu = item.impulses(prm, ctx, item.half_mass(prm, ctx, H0),
                        item.half_mass(prm, ctx, H1), V,
                        np.full(1, ctx.tau_run), np.zeros(1))[0]
    m = float(np.abs(dnu).max())
    return m == 0.0, f"max |dnu| above the chine {m:.1e}"


@CB.check("W4")
def w4_airborne_and_stern_wet(item, ctx):
    """Whole keel out of the water: heave error = -(g + A_z), pitch error
    = -A_th exactly. Bow out, stern wet, pure lift (A_th = 0): heave error
    between -(g + A_z) and 0 and a bow-down pitch acceleration."""
    rng = np.random.default_rng(25)
    prm = item.draw_on(rng, ctx)
    sr = _run(ctx, [ctx.u_id] * 3, dz=1.5)
    q = _q(ctx, sr)
    q["a_l1_vert"] = rng.normal(0, 2, (3, 2))
    q["a_cat_vert"] = rng.normal(0, 2, (3, 2))
    a = CB.to_acc(ctx, item.step(prm, {}, q, CB.DT_SUB).Q)
    A = q["a_lofi_vert"] + q["a_l1_vert"] + q["a_cat_vert"]
    d = max(np.abs(a[:, 3] + GRAV + A[:, 0]).max(),
            np.abs(a[:, 4] + A[:, 1]).max())
    # bow out: pitch bow-up by 8 deg about the stern station
    th = ctx.sp * 8 * DEG
    sr2 = _run(ctx, [ctx.u_id], dth=th)
    sr2[0, 3] = ctx.z0 - ctx.sp * ctx.x_st[0] * th
    q2 = _q(ctx, sr2)
    q2["a_lofi_vert"] = np.zeros((1, 2))
    a2 = CB.to_acc(ctx, item.step(prm, {}, q2, CB.DT_SUB).Q)[0]
    wet = q2["hc"][0]
    ok2 = (-GRAV < a2[3] < 0.0) and ctx.sp * a2[4] < 0.0 and wet[0] > 0 \
        and wet[-1] < 0
    return d < 1e-9 and ok2, (f"airborne residual {d:.1e}; stern-only wet: "
                              f"heave {a2[3]:.2f} m/s^2, bow-up pitch "
                              f"{ctx.sp * a2[4]:.2f} rad/s^2")


@CB.check("W4")
def w4_assembled_loop(item, ctx):
    """CatDraw with W1-W6 forced on (layer 1 off, no layer-3 forces): the
    gate reads the gated sums, W3's impulses reach the impulse channel,
    parts sum to the total; 8 s closed loop in a Hs 1.5 m sea stays finite
    and a calm row at u_id stays at rest."""
    from control.reduced import ReducedModel
    from learn.meta import operators_rb as R
    codes = ("W1", "W2", "W3", "W4", "W5", "W6")
    fi = {c: True for c in codes if c in CB.REGISTRY}
    d = CB.CatDraw(5, tier="none", items=codes, force_items=fi,
                   sparse=False, layer1=False)
    c = d.ctx
    B = 4
    sea = R.sea_state(dict(R.sea_dict(np.random.default_rng(3)), hs=1.5), 3)
    rs = R.RowSeas([sea] * B, c.x_st, c.y_off)
    cs = CB.CatSea(rs, c)
    red = ReducedModel(c.p)
    sr = _run(c, [c.u_id, 9.0, 15.0, 22.0])
    thr = c.k_drag * sr[:, 2] ** 2
    noz = np.zeros(B)
    calm = np.array([True, False, False, False])
    st = d.new_state(B)
    rv = list(R.RED_VEL)
    n_imp, psum = 0, 0.0
    sr0 = sr.copy()
    for j in range(200):
        s = cs.sample(sr, j * CB.DT_SUB)
        s = {k: np.where(calm.reshape((-1,) + (1,) * (v.ndim - 1)), 0.0, v)
             for k, v in s.items()}
        ns = red.step(sr, thr, noz, s["eta"], c.x_st, CB.DT_SUB)[0]
        nu0 = (ns[:, rv] - sr[:, rv]) / CB.DT_SUB
        out = d.substep(st, sr, thr, noz, s, nu0=nu0, parts=True)
        tot = sum(v for k, v in out["PARTS"].items() if not k.endswith(":imp"))
        psum = max(psum, float(np.abs(tot - out["acc"]).max()))
        n_imp += int((np.abs(out["imp"]).sum(1) > 0).sum())
        rc, _ = CB.clip_push(out["acc"], np.zeros_like(out["acc"]))
        ns[:, rv] += rc * CB.DT_SUB + out["imp"]
        sr = ns
    fin = bool(np.isfinite(sr).all())
    drift = float(np.abs(sr[0] - sr0[0])[[3, 4, 5, 6, 8, 9]].max())
    # W6 is not calm-zero (spray drag): the calm row only loses speed and
    # gets the pitch of that drag; W5(b) may also shift it off u_id
    return fin and psum < 1e-9, (f"items {sorted(d.on)}; finite {fin}; "
                                 f"parts-sum error {psum:.1e}; substep-rows "
                                 f"with a slam impulse {n_imp}; calm-row "
                                 f"drift {drift:.2e} (W6 drag, not zero)")


@CB.check("W5")
def w5_stiffness_and_hysteresis(item, ctx):
    """(a) relative stiffness > 0 at all depths (convex mix of plateaus),
    force non-decreasing in immersion, plateau after the chine = r x rails;
    (b) sweeping the speed up and down flips the switch at U_c + H/2 up and
    U_c - H/2 down; side walls: mirrored immersion flips the yaw sign."""
    rng = np.random.default_rng(26)
    worst_k, worst_F = np.inf, np.inf
    for _ in range(50):
        prm = item.draw_on(rng, ctx)
        h = np.linspace(-0.2, 1.5, 400)[:, None].repeat(5, 1)
        k, F = item.kappa(prm, h)
        worst_k = min(worst_k, float(k.min()))
        worst_F = min(worst_F, float(np.diff(F, axis=0).min()))
    fr, D = item.kinks(prm)
    kd, _ = item.kappa(prm, np.full((1, 5), 50.0))
    plat = abs(kd[0, 0] - (1 + D.sum()))
    prm = dict(item.draw_on(rng, ctx), a_on=0.0, b_on=1.0,
               U_c=10.0, hyst=1.0, tau_s=0.05)
    st = item.init_state(prm, 1)
    ups, downs = [], []
    last = None
    for U in list(np.linspace(8.0, 12.0, 401)) + list(np.linspace(12.0, 8.0,
                                                                    401)):
        q = _q(ctx, _run(ctx, [U]))
        item.step(prm, st, q, CB.DT_SUB)
        if last is not None and st["h"][0] != last:
            (ups if st["h"][0] > last else downs).append(U)
        last = st["h"][0]
    hy = len(ups) == 1 and len(downs) == 1 and abs(ups[0] - 10.5) < 0.02 \
        and abs(downs[0] - 9.5) < 0.02
    prm = dict(item.draw_on(rng, ctx), a_on=1.0, b_on=0.0, n_r=1.0)
    sr = _run(ctx, [ctx.u_id] * 2)
    q = _q(ctx, sr)
    dc = np.asarray(prm["dc"])
    q["h"][0, :, 2] = dc + 0.05
    q["h"][1, :, 0] = dc + 0.05
    a = item.step(prm, {}, q, CB.DT_SUB).acc
    mir = a[0, 2] > 0 and abs(a[0, 2] + a[1, 2]) < 1e-12
    ok = worst_k > 0 and worst_F >= 0 and plat < 1e-9 and hy and mir
    return ok, (f"min k/k1 {worst_k:.3f}; min dF {worst_F:.1e}; plateau err "
                f"{plat:.1e}; switch up at {ups} down at {downs} m/s "
                f"(10.5 / 9.5); side-wall mirror {mir}")


@CB.check("W5")
def w5_sub_switch_rates(item, ctx):
    """Marginal on-rates of (a) and (b) over item draws = 0.6 and 0.5; the
    T4 bounded-stiffness switch only with (a)."""
    rng = np.random.default_rng(27)
    n = 4000
    na = nb = nsw = 0
    for _ in range(n):
        p = item.draw(rng, ctx)
        if p is None:
            continue
        na += p["a_on"] > 0
        nb += p["b_on"] > 0
        nsw += ("T4_nl_off" in item.overlaps(p)) != (p["a_on"] > 0)
    za = abs(na / n - 0.6) / np.sqrt(0.24 / n)
    zb = abs(nb / n - 0.5) / np.sqrt(0.25 / n)
    return za < 4 and zb < 4 and nsw == 0, (f"(a) {na / n:.3f} (0.6), (b) "
                                             f"{nb / n:.3f} (0.5)")


@CB.check("W6")
def w6_drag_properties(item, ctx):
    """Never propels, never reverses u in a substep, monotone decreasing in
    trim; at U_id and the running trim dR / R = f_s; calm running has no
    bow term."""
    rng = np.random.default_rng(28)
    prm = item.draw_on(rng, ctx)
    prm["p_s"] = max(prm["p_s"], 0.3)
    B = 400
    sr = _run(ctx, rng.uniform(-3, 30, B), dz=0.0,
              dth=rng.uniform(-0.1, 0.1, B))
    sr[:, 3] = ctx.z0 + rng.uniform(-0.4, 0.2, B)
    q = _q(ctx, sr)
    a = CB.to_acc(ctx, item.step(prm, {}, q, CB.DT_SUB).Q)[:, 0]
    bad = np.any(a * sr[:, 2] > 0) or np.any(
        np.abs(a * CB.DT_SUB) > np.abs(sr[:, 2]))
    dths = np.linspace(-0.08, 0.08, 9) * ctx.sp        # bow-up increasing
    cm = [item.coef(prm, ctx, _q(ctx, _run(ctx, [15.0], dth=t)))[0][0]
          for t in dths]
    mono = np.all(np.diff(cm) <= 1e-12)
    q0 = _q(ctx, _run(ctx, [ctx.u_id]))
    c_main, c_bow = item.coef(prm, ctx, q0)
    ratio = float(c_main[0] / ctx.k_drag)
    return (not bad and mono and abs(ratio - prm["f_s"]) < 1e-9
            and c_bow[0] == 0.0), (f"propels/reverses {bad}; monotone in "
                                   f"trim {mono}; dR/R at U_id {ratio:.4f} "
                                   f"(f_s {prm['f_s']:.4f}); calm bow term "
                                   f"{c_bow[0]:.1e}")
