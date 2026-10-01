#!/usr/bin/env python3
"""
Catalogue items A1-A3 (air) and E1-E4 (environment) of the D10 force-
catalogue prior (learn/meta/PRIOR_D10_DRAFT.md 3.3 and 3.6, with sections
10 and 11; framework learn/meta/cat_base.py). The low-fidelity world has no
wind, no aerodynamic lift, no current, one constant water depth, no debris
and a stationary sea; every item below is a hidden process plus the
acceleration (or impulse) it adds to the five velocity channels.

A1  mean wind load (stage 'env'): hidden earth-fixed wind vector (U10 at
    10 m, log profile to the boat's 1-2 m, slow mean-reverting drift of
    speed and direction), relative wind = wind - the boat's velocity OVER
    GROUND; load q_rw [A_F C_X, A_L C_Y, A_L L C_N](gamma) with a
    Blendermann shape (w.p. 0.5) or a low-order Fourier shape; only
    "with wind minus without wind" is added (the boat's own air drag is in
    the identified k_drag). Heel moment -z_a F_y into H1 (q['K_roll']).
A2  gusts and squalls (stage 'env', only with A1): the three turbulence
    components of a frozen field crossed at the relative wind speed
    (Taylor), each a sum of unit-variance Ornstein-Uhlenbeck processes
    whose spectrum has the -5/3 tail of every common wind spectrum
    (w.p. 0.3 the one-pole Dryden form), corner frequency V_rel / L_u
    updated every substep; squalls = Poisson jumps of the mean speed and
    direction, ramped by an exact first-order lag. The load increment is
    A1's load at (mean + gust) minus at the mean (the exact version of the
    draft's "A1 linearised"). The vertical gust feeds A3.
A3  aerodynamic lift and pitch moment (stage 'force'): F_z = q_x S C_L(alpha)
    G(h_b), Polhamus C_L = K_p sin a cos^2 a + K_v cos a sin a |sin a|, ground
    effect G = 1 + c_g exp(-h_b / l_g) on the D9.6 bow height, moment arm
    x_ac = x_0 - x_1 sin a ahead of the reference point (bow-up moment grows
    with trim: negative pitch stiffness), induced drag F_z tan a. Not
    subtracted at the calibration state (draft section 10 item 18).
E1  current (stage 'env'): hidden earth-fixed 2-D current (wind-driven k U10
    near the wind direction, or tidal / coastal LogU[0.1, 2] m/s), slow
    drift, fronts (Poisson in distance, jump <= |V_c|, crossed over a front
    width). Sets q['nu_c'] so layer 1 and every item read water-relative
    velocities; adds the low-fidelity boat's own terms shifted to the
    relative velocity (surge drag, sway damping, rudder lift of a non-jet
    boat), the added-mass terms A11 du_c/dt, A22 dv_c/dt (incl. r v_c,
    -r u_c) and, at a front, half the water's vorticity as a yaw-rate
    offset of the low-fidelity yaw damping.
E2  shallow water and banks (stage 'force'): depth and bank distance are
    random fields along the track (log-OU in distance); drag multiplier
    1 + a_h Bump(F_h), equilibrium offsets through a smooth step in F_h - 1
    and a monotone lift / trim change in T / h, both applied through the
    low-fidelity boat's own stiffness; bank suction toward the bank and
    bow-out moment ~ u^2 exp(-d / l) (banks off by default, E2_BANK_P).
E3  debris impacts (stage 'force', IMPULSE channel): Poisson events, a
    point impulse m du along a mostly aft-and-up direction at a keel point,
    spread over its duration d, never through the substep clip (draft 6
    items 3 and 6); w.p. 0.3 an event flags q['debris_hit'] for P10.
E4  wave packets (stage 'env'): Poisson (and w.p. 0.5 own-wake after a turn
    of more than 150 deg within 60 s) wave trains A E(t) cos(k x - w t + p)
    added to the station elevations, their moving-station rate, the intake
    elevation and the orbital velocities every later term reads, plus the
    low-fidelity boat's own heave / pitch (and sway / yaw) response to the
    packet (it never sees it).

Numerical rules (draft section 6 item 6): every first-order hidden state
(wind drift, gust filters, squall ramp, current drift and front crossing,
depth / bank fields) uses the exact exponential / exact OU update, per
substep or per distance travelled; impulses (E3) go to the separate impulse
channel with a physical bound; no explicit stiff update anywhere.

Cross-item interfaces (all optional):
  q['wind']      set by A1 (updated by A2): dict(U10, zr, psi0 (B,), Vmean
                 (B, 2) earth, Vtot (B, 2) earth, w_g (B,) vertical gust,
                 load(V) -> (Fx, Fy, N), z_a); read by A2, A3, E1
  q['K_roll']    roll moment (N m, positive raises port) added by A1 / A2
                 for H1 (learn/meta/cat_hidden_boat.py)
  q['nu_c']      body-frame current (B, 2) set by E1 (cat_base contract)
  q['eta_add'], q['etad_add'], q['eta_in_add'], q['uorb'], q['vorb']
                 E4's packet added to the shared sea quantities
  q['debris_hit']  (B,) bool, E3 events that damage the intake (P10)
  A1 state['wave_dir'] (B,) the sea's mean propagation direction, set by
                 the episode stage with bind_wave_dir(); without it the
                 "within 30 deg of the waves" wind falls back to a uniform
                 direction
"""
import numpy as np

from learn.meta import cat_base as CB

GRAV, DEG, RHO_A, RHO_W = CB.GRAV, CB.DEG, CB.RHO_A, CB.RHO_W
TWO_PI = 2.0 * np.pi

# A1 [assumption where the draft is silent]
A1_SD_U, A1_SD_PSI = 0.10, 10.0 * DEG   # stationary sd of the slow drift
A1_CDL_BOW = (0.45, 0.95)               # head-wind C_X A_F / A_F split
# A2: OU sum with a -5/3 tail. tau_i = beta_i / omega_c; weights fitted
# (NNLS, relative error <= 12 % on 0.01-500 omega_c, tail slope -1.66) to
# the von Karman shape (1 + (w / w_c)^2)^(-5/6), normalised to unit variance
KAIMAL_BETA = np.array([1.0, 0.2, 0.04, 0.008])
KAIMAL_W = np.array([0.48087396, 0.14056077, 0.01683696, 0.01621446])
KAIMAL_W = KAIMAL_W / KAIMAL_W.sum()
A2_V_MIN = 0.5                          # m/s floor of the crossing speed
# E1
E1_SD_REL = 0.2                         # drift sd / |V_c| [assumption]
# E2
E2_BANK_P = 0.0                         # banks off (open-water tasks)
# E4
E4_RING_DT, E4_RING_N = 1.0, 61         # heading history: 1 s x 60 s
E4_TURN = 150.0 * DEG
E4_G3 = np.exp(-4.5)                    # envelope value at 3 sigma


def logu(rng, lo, hi, size=None):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


def e2b(V, psi):
    """Earth vector (B, 2) -> body components (u, v) at heading psi."""
    c, s = np.cos(psi), np.sin(psi)
    return V[:, 0] * c + V[:, 1] * s, -V[:, 0] * s + V[:, 1] * c


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def station_xy(ctx, sr):
    """Earth positions and velocities (B, 16) of the 5 x 3 stations and the
    intake (the CatSea.sample formulas)."""
    x, y, u, psi, r, v = sr[:, 0], sr[:, 1], sr[:, 2], sr[:, 7], sr[:, 8], \
        sr[:, 9]
    cp, sn = np.cos(psi)[:, None], np.sin(psi)[:, None]
    xs = np.concatenate([np.repeat(ctx.x_st, 3), [ctx.intake[0]]])[None]
    ys = np.concatenate([np.tile(ctx.y_off, 5), [ctx.intake[1]]])[None]
    X = x[:, None] + xs * cp - ys * sn
    Y = y[:, None] + xs * sn + ys * cp
    Xd = u[:, None] * cp - v[:, None] * sn - r[:, None] * (xs * sn + ys * cp)
    Yd = u[:, None] * sn + v[:, None] * cp + r[:, None] * (xs * cp - ys * sn)
    return X, Y, Xd, Yd


def rel_wind_obs(V, sr):
    """The boat's velocity relative to the air (what an anemometer reads,
    obs_defaults' convention): speed and angle atan2(v_rel, u_rel)."""
    wx, wy = e2b(V, sr[:, 7])
    ux, uy = sr[:, 2] - wx, sr[:, 9] - wy
    return dict(wind_speed_rel=np.hypot(ux, uy),
                wind_angle_rel=np.arctan2(uy, ux))


# =================================================================== A1
def wind_coeffs(prm, gam):
    """(C_X, C_Y, C_N) at the relative-wind angle gam (0 = air from ahead,
    positive = from port). F_x = q A_F C_X, F_y = -q A_L C_Y (towards
    starboard for air from port), N = -q A_L L C_N (= x_eff F_y)."""
    if prm["blend"]:                      # Blendermann 1994 (Fossen MSS)
        ag = np.abs(gam)
        CDlAF = np.where(ag <= 0.5 * np.pi, prm["CDl_bow"],
                         prm["CDl_bow"] * prm["stern"])
        CDl = CDlAF * prm["A_F"] / prm["A_L"]
        den = 1.0 - 0.5 * prm["delta"] * (1.0 - CDl / prm["CDt"]) \
            * np.sin(2.0 * ag) ** 2
        CX = -CDlAF * np.cos(ag) / den
        CY = np.sign(gam) * prm["CDt"] * np.sin(ag) / den
        CN = (prm["sL"] - 0.18 * (ag - 0.5 * np.pi)) * CY
        return CX, CY, CN
    a, b, c, e = (np.asarray(prm[k]) for k in ("a", "b", "c", "asym"))
    k = np.arange(1, 4)[:, None]
    g = np.asarray(gam)[None]
    CX = -(a[:, None] * np.cos(k * g)).sum(0) + 0.05 * a[0] * e[0] \
        * np.sin(gam)
    CY = (b[:, None] * np.sin(k * g)).sum(0) + 0.05 * b[0] * e[1] \
        * np.cos(gam)
    CN = (c[:, None] * np.sin(k * g)).sum(0) + 0.05 * 0.1 * b[0] * e[2] \
        * np.cos(gam)
    return CX, CY, CN


def wind_load(prm, ctx, V, sr):
    """Total aerodynamic load (Fx, Fy, N) of earth wind V (B, 2) on a boat
    with ground velocity (u, v) at heading psi."""
    wx, wy = e2b(V, sr[:, 7])
    wx, wy = wx - sr[:, 2], wy - sr[:, 9]          # air relative to boat
    qa = 0.5 * RHO_A * (wx * wx + wy * wy)
    gam = np.arctan2(-wy, -wx)
    CX, CY, CN = wind_coeffs(prm, gam)
    return (qa * prm["A_F"] * CX, -qa * prm["A_L"] * CY,
            -qa * prm["A_L"] * ctx.L * CN)


def bind_wave_dir(state_a1, rowseas):
    """Store the rows' mean wave propagation direction (energy weighted,
    from an operators_rb.RowSeas) in A1's state before the first substep."""
    a2 = np.asarray(rowseas.a) ** 2
    state_a1["wave_dir"] = np.arctan2((a2 * rowseas.s).sum(1),
                                      (a2 * rowseas.c).sum(1))


@CB.register
class A1(CB.CatItem):
    code, stage = "A1", "env"

    def draw_on(self, rng, ctx):
        u = rng.uniform
        z, z0 = u(1.0, 2.0), float(logu(rng, 1e-4, 1e-2))
        f_air = float(logu(rng, 0.03, 0.3))
        CA = 2.0 * f_air * ctx.k_drag / RHO_A      # C_X(0) A_F
        CDl = u(*A1_CDL_BOW)
        A_F = CA / CDl
        prm = dict(U10=u(0.0, 15.0), zr=np.log(z / z0) / np.log(10.0 / z0),
                   wave_rel=bool(rng.random() < 0.7), dpsi=u(-30, 30) * DEG,
                   psi_u=u(0.0, TWO_PI), f_air=f_air, CDl_bow=CDl, A_F=A_F,
                   A_L=A_F * u(1.5, 3.5), tau_d=float(logu(rng, 300, 3000)),
                   z_a=u(0.5, 1.5))
        if rng.random() < 0.5:
            prm.update(blend=True, CDt=u(0.7, 1.0), delta=u(0.1, 0.8),
                       stern=u(0.9, 1.2), sL=u(-0.15, 0.15))
        else:
            ar = np.array([1.0, u(-0.3, 0.3), u(-0.3, 0.3)])
            b1 = u(0.7, 1.0)
            prm.update(blend=False, a=CDl * ar / ar.sum(),
                       b=b1 * np.array([1.0, u(-0.3, 0.3), u(-0.3, 0.3)]),
                       c=b1 * np.array([u(-0.15, 0.15), u(-0.1, 0.3),
                                        u(-0.05, 0.05)]),
                       asym=(u(-1, 1, 3) if rng.random() < 0.2
                             else np.zeros(3)))
        return prm

    def init_state(self, params, B=1, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        return dict(x=rng.normal(0.0, 1.0, (B, 2)), rng=rng, psi0=None,
                    wave_dir=None)

    def mean_wind(self, prm, state, B):
        if state["psi0"] is None:
            wd = state.get("wave_dir")
            base = np.asarray(wd, float) if (prm["wave_rel"] and wd is not
                                              None) else np.full(B, prm[
                                                  "psi_u"])
            state["psi0"] = base + (prm["dpsi"] if prm["wave_rel"] else 0.0)
        x = state["x"]
        U = prm["U10"] * prm["zr"] * np.maximum(1.0 + A1_SD_U * x[:, 0], 0.0)
        psi = state["psi0"] + A1_SD_PSI * x[:, 1]
        return np.stack([U * np.cos(psi), U * np.sin(psi)], 1)

    def step(self, prm, state, q, dt):
        ctx, sr, B = q["ctx"], q["sr"], q["B"]
        V = self.mean_wind(prm, state, B)
        state["x"] = CB.ou_step(state["x"], prm["tau_d"], dt, state["rng"])
        fx, fy, n = wind_load(prm, ctx, V, sr)
        f0x, f0y, n0 = wind_load(prm, ctx, np.zeros((B, 2)), sr)
        Q = np.zeros((B, 5))
        Q[:, 0], Q[:, 1], Q[:, 2] = fx - f0x, fy - f0y, n - n0
        # the low-fidelity boat has no roll at all: the whole side force
        # heels the hidden roll (force at z_a above the roll axis)
        q["K_roll"] = np.asarray(q.get("K_roll", 0.0), float) \
            - prm["z_a"] * fy
        q["wind"] = dict(U10=prm["U10"], zr=prm["zr"], psi0=state["psi0"],
                         Vmean=V, Vtot=V.copy(), w_g=np.zeros(B),
                         z_a=prm["z_a"],
                         load=lambda W, s=sr: wind_load(prm, ctx, W, s))
        return CB.ItemOut(Q=Q, obs=rel_wind_obs(V, sr), state=state)


# =================================================================== A2
@CB.register
class A2(CB.CatItem):
    code, stage = "A2", "env"

    def draw_on(self, rng, ctx):
        u = rng.uniform
        return dict(I_u=u(0.08, 0.2), r_v=u(0.75, 0.8), r_w=0.5,
                    L_u=float(logu(rng, 3.0, 200.0)),
                    dryden=bool(rng.random() < 0.3),
                    sq_rate=float(logu(rng, 1 / 7200, 1 / 900)),
                    sq_tau=float(logu(rng, 5.0, 60.0)))

    def init_state(self, params, B=1, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        n = 1 if params["dryden"] else len(KAIMAL_BETA)
        z = np.zeros(B)
        return dict(x=rng.normal(0.0, 1.0, (B, 3, n)), rng=rng,
                    sq_on=np.zeros(B, bool), sq_left=z.copy(),
                    sq_dU=z.copy(), sq_dpsi=z.copy(), dU=z.copy(),
                    dpsi=z.copy())

    @staticmethod
    def betas_weights(prm):
        if prm["dryden"]:
            return np.ones(1), np.ones(1)
        return KAIMAL_BETA, KAIMAL_W

    def step(self, prm, state, q, dt):
        ctx, sr, B = q["ctx"], q["sr"], q["B"]
        rng = state["rng"]
        w = q.get("wind")
        if w is None:                     # A1 absent: still air, no load
            U10, zr, Vm = 0.0, 1.0, np.zeros((B, 2))
            psi0 = np.zeros(B)
        else:
            U10, zr, Vm, psi0 = w["U10"], w["zr"], w["Vmean"], w["psi0"]
        # squalls: Poisson onsets, U[0.3, 1] U10 and U[-90, 90] deg, lasting
        # U[600, 3600] s, ramped with the exact first-order lag sq_tau
        hit = CB.poisson_hit(prm["sq_rate"], dt, rng, B) & ~state["sq_on"]
        if hit.any():
            n = int(hit.sum())
            state["sq_dU"][hit] = rng.uniform(0.3, 1.0, n) * U10 * zr
            state["sq_dpsi"][hit] = rng.uniform(-90, 90, n) * DEG
            state["sq_left"][hit] = rng.uniform(600, 3600, n)
            state["sq_on"] |= hit
        state["sq_left"] = state["sq_left"] - dt
        end = state["sq_on"] & (state["sq_left"] <= 0)
        state["sq_on"] &= ~end
        state["sq_dU"][end], state["sq_dpsi"][end] = 0.0, 0.0
        state["dU"] = CB.exp_update(state["dU"], state["sq_dU"],
                                    prm["sq_tau"], dt)
        state["dpsi"] = CB.exp_update(state["dpsi"], state["sq_dpsi"],
                                      prm["sq_tau"], dt)
        spd = np.hypot(Vm[:, 0], Vm[:, 1])
        ang = np.where(spd > 0, np.arctan2(Vm[:, 1], Vm[:, 0]), psi0) \
            + state["dpsi"]
        spd = spd + state["dU"]
        e1 = np.stack([np.cos(ang), np.sin(ang)], 1)
        e2 = np.stack([-np.sin(ang), np.cos(ang)], 1)
        Vm2 = spd[:, None] * e1
        # turbulence: frozen field crossed at the relative wind speed
        Vr = rel_wind_obs(Vm2, sr)["wind_speed_rel"]
        wc = np.maximum(Vr, A2_V_MIN) / prm["L_u"]
        beta, wts = self.betas_weights(prm)
        tau = beta[None, None, :] / wc[:, None, None]
        state["x"] = CB.ou_step(state["x"], tau, dt, rng)
        g = (state["x"] * np.sqrt(wts)[None, None, :]).sum(2)   # (B, 3)
        su = prm["I_u"] * spd
        gu, gv, gw = su * g[:, 0], prm["r_v"] * su * g[:, 1], \
            prm["r_w"] * su * g[:, 2]
        Vt = Vm2 + gu[:, None] * e1 + gv[:, None] * e2
        Q = np.zeros((B, 5))
        if w is not None:
            f1 = w["load"](Vt)
            f0 = w["load"](Vm)
            Q[:, 0], Q[:, 1], Q[:, 2] = (a - b for a, b in zip(f1, f0))
            q["K_roll"] = np.asarray(q.get("K_roll", 0.0), float) \
                - w["z_a"] * Q[:, 1]
            w["Vtot"], w["w_g"], w["Vmean2"] = Vt, gw, Vm2
        return CB.ItemOut(Q=Q, obs=rel_wind_obs(Vt, sr), state=state)


# =================================================================== A3
def polhamus(a, r_v):
    """C_L / K_p = sin a cos^2 a + (K_v / K_p) cos a sin a |sin a| (odd)."""
    s, c = np.sin(a), np.cos(a)
    return s * c * c + r_v * c * s * np.abs(s)


@CB.register
class A3(CB.CatItem):
    code, stage = "A3", "force"
    MAX_TRIES = 200

    @staticmethod
    def ref_state(ctx):
        """Calm running at u_max, no wind: (q_ref, bow height)."""
        hb = ctx.F_b + ctx.z0 + ctx.sp * ctx.x_st[-1] * ctx.th0
        return 0.5 * RHO_A * ctx.u_max ** 2, max(hb, 0.0)

    def draw_on(self, rng, ctx):
        """kappa_theta (aerodynamic / low-fidelity pitch stiffness) and
        lambda_L (lift / weight), both at u_max without wind, fix S K_p and
        the arm x_0 together; a draw whose x_0 leaves the draft's [0.05,
        0.35] L is redrawn (the stated ranges hold jointly)."""
        u = rng.uniform
        qr, hb = self.ref_state(ctx)
        I = float(ctx.Mt[4, 4])
        best = None
        for k in range(self.MAX_TRIES):
            prm = dict(kap=float(logu(rng, 0.01, 0.3)),
                       lam=float(logu(rng, 0.003, 0.08)),
                       r_v=u(0.0, 3.0), x_1=u(0.0, 0.15) * ctx.L,
                       a_0=u(-0.03, 0.1), c_g=u(0.0, 1.0), l_g=u(0.2, 1.0))
            a = prm["a_0"] + ctx.tau0
            G = 1.0 + prm["c_g"] * np.exp(-hb / prm["l_g"])
            c = polhamus(a, prm["r_v"])
            SK = prm["lam"] * ctx.W / (qr * c * G)
            A = qr * SK * G
            h = 1e-6
            cd = (polhamus(a + h, prm["r_v"]) - polhamus(a - h, prm["r_v"])) \
                / (2 * h)
            x0 = (prm["kap"] * ctx.wp ** 2 * I / A
                  + prm["x_1"] * np.cos(a) * c) / cd + prm["x_1"] * np.sin(a)
            prm.update(SK=float(SK), x_0=float(x0), tries=k + 1)
            if 0.05 * ctx.L <= x0 <= 0.35 * ctx.L:
                return prm
            if best is None:
                best = prm
        best["x_0"] = float(np.clip(best["x_0"], 0.05 * ctx.L, 0.35 * ctx.L))
        return best

    def forces(self, prm, q):
        ctx, sr, B = q["ctx"], q["sr"], q["B"]
        w = q.get("wind")
        Vt = w["Vtot"] if w is not None else np.zeros((B, 2))
        wg = w["w_g"] if w is not None else np.zeros(B)
        wx, _ = e2b(Vt, sr[:, 7])
        Vx = np.maximum(sr[:, 2] - wx, 0.0)             # air from ahead
        qx = 0.5 * RHO_A * Vx * Vx
        th_up = ctx.tau0 + q["trim_rel"]
        wz = sr[:, 4] + ctx.sp * prm["x_0"] * sr[:, 6]  # plunge at x_0
        a = prm["a_0"] + th_up + (wg - wz) / np.maximum(Vx, 1.0)
        a = np.clip(a, -0.25 * np.pi, 0.25 * np.pi)     # Polhamus range
        hb = np.maximum(CB.bow_height(ctx, sr, q["eta"]), 0.0)
        G = 1.0 + prm["c_g"] * np.exp(-hb / prm["l_g"])
        Fz = qx * prm["SK"] * polhamus(a, prm["r_v"]) * G
        return Fz, -Fz * np.tan(a), prm["x_0"] - prm["x_1"] * np.sin(a)

    def step(self, prm, state, q, dt):
        ctx, B = q["ctx"], q["B"]
        Fz, Fx, xac = self.forces(prm, q)
        F = np.stack([Fx, np.zeros(B), Fz], 1)
        return CB.ItemOut(Q=CB.lever_Q(ctx, (xac, 0.0, 0.0), F,
                                       q["sr"][:, 5]), state=state)


# =================================================================== E1
@CB.register
class E1(CB.CatItem):
    code, stage = "E1", "env"

    def draw_on(self, rng, ctx):
        u = rng.uniform
        return dict(wind_mode=bool(rng.random() < 0.5), k_w=u(0.015, 0.03),
                    dpsi=u(-30, 30) * DEG, U10_fb=u(0.0, 15.0),
                    psi_fb=u(0.0, TWO_PI),
                    V_tid=float(logu(rng, 0.1, 2.0)), psi_t=u(0.0, TWO_PI),
                    tau_d=float(logu(rng, 300, 3000)),
                    front_s=float(logu(rng, 200, 5000)),
                    front_w=float(logu(rng, 5.0, 50.0)))

    def init_state(self, params, B=1, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        z = np.zeros((B, 2))
        return dict(rng=rng, Vbar=None, x=rng.normal(0.0, 1.0, (B, 2)),
                    tgt=z.copy(), cur=z.copy(), V=None)

    def vbar(self, prm, q, B):
        if not prm["wind_mode"]:
            ang = np.full(B, prm["psi_t"])
            mag = np.full(B, prm["V_tid"])
        else:
            w = q.get("wind")
            if w is not None:
                mag = np.full(B, prm["k_w"] * w["U10"])
                ang = np.asarray(w["psi0"], float) + prm["dpsi"]
            else:
                mag = np.full(B, prm["k_w"] * prm["U10_fb"])
                ang = np.full(B, prm["psi_fb"] + prm["dpsi"])
        return mag[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)

    @staticmethod
    def value(prm, state):
        mag = np.hypot(*state["Vbar"].T)[:, None]
        return state["Vbar"] + E1_SD_REL * mag * state["x"] + state["cur"]

    def step(self, prm, state, q, dt):
        ctx, sr, B = q["ctx"], q["sr"], q["B"]
        rng = state["rng"]
        if state["Vbar"] is None:
            state["Vbar"] = self.vbar(prm, q, B)
            state["V"] = self.value(prm, state)
        V = state["V"]
        u, v, psi, r = sr[:, 2], sr[:, 9], sr[:, 7], sr[:, 8]
        # advance the hidden current: drift (time), fronts (distance)
        Ug = q["U_g"]
        ds = Ug * dt
        state["x"] = CB.ou_step(state["x"], prm["tau_d"], dt, rng)
        hit = CB.poisson_hit(1.0 / prm["front_s"], ds, rng, B)
        mag = np.hypot(*state["Vbar"].T)
        if hit.any():
            n = int(hit.sum())
            ph = rng.uniform(0, TWO_PI, n)
            J = (mag[hit] * rng.uniform(0, 1, n))[:, None] * np.stack(
                [np.cos(ph), np.sin(ph)], 1)
            t = state["tgt"][hit] + J
            nt = np.hypot(*t.T)
            cap = np.minimum(1.0, mag[hit] / np.maximum(nt, 1e-12))
            state["tgt"][hit] = t * cap[:, None]
        cur0 = state["cur"]
        state["cur"] = CB.exp_update(cur0, state["tgt"],
                                     prm["front_w"] / np.maximum(Ug, 1e-3)
                                     [:, None], dt)
        Vn = self.value(prm, state)
        state["V"] = Vn
        # body-frame current, its rate (rotation with r + earth change)
        uc, vc = e2b(V, psi)
        dux, duy = e2b((Vn - V) / dt, psi)
        duc, dvc = r * vc + dux, -r * uc + duy
        q["nu_c"] = q["nu_c"] + np.stack([uc, vc], 1)
        ur, vr = u - uc, v - vc
        # vorticity of the front crossed along the course (normal = track)
        chi = psi + np.arctan2(v, u)
        dV = (state["cur"] - cur0) / np.maximum(ds, 1e-9)[:, None]
        omg = np.cos(chi) * dV[:, 1] - np.sin(chi) * dV[:, 0]
        p = ctx.p
        m = ctx.m
        acc = np.zeros((B, 5))
        acc[:, 0] = (-ctx.k_drag * (ur * np.abs(ur) - u * np.abs(u))
                     + max(ctx.m_u - m, 0.0) * duc) / ctx.m_u
        lift = 0.0
        if p.get("steer_jet", 0.0) <= 0.5:          # rudder boat
            rud = np.clip(q["noz"], -ctx.rud_stall, ctx.rud_stall)
            lift = p.get("k_lift", 0.0) * (ur * np.abs(ur) - u * np.abs(u)) \
                * rud
        acc[:, 1] = (lift - p["k_lin_sway"] * (vr - v)
                     - p["k_sway"] * (vr * np.abs(vr) - v * np.abs(v))
                     + max(ctx.m_v - m, 0.0) * dvc) / ctx.m_v
        acc[:, 2] = 0.5 * omg / p["tau_r"]
        return CB.ItemOut(acc=acc, obs=dict(stw_u=ur, stw_v=vr), state=state)


# =================================================================== E2
def smoothstep(x):
    """0 for x <= -1, 1 for x >= 1, C^1 cubic in between."""
    s = np.clip(0.5 * (np.asarray(x, float) + 1.0), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def bump(x):
    """(1 - x^2)^2 on |x| < 1, else 0: compact, peak 1 at x = 0."""
    return np.maximum(1.0 - np.asarray(x, float) ** 2, 0.0) ** 2


@CB.register
class E2(CB.CatItem):
    code, stage = "E2", "force"
    gated = True

    def draw_on(self, rng, ctx):
        u = rng.uniform
        prm = dict(h_bar=float(logu(rng, max(1.5, 4 * ctx.T), 60.0)),
                   s_h=u(0.1, 0.5), l_h=float(logu(rng, 100, 2000)),
                   a_h=u(-0.2, 0.3), w=u(0.1, 0.4),
                   dz_s=u(-0.3, 0.3) * ctx.T, dth_s=u(-0.35, 0.15) * ctx.tau0,
                   dz_T=u(0.0, 0.3) * ctx.T, dth_T=u(-0.35, 0.1) * ctx.tau0,
                   bank=bool(rng.random() < E2_BANK_P))
        prm.update(side=float(rng.choice([-1.0, 1.0])),
                   d_bar=float(logu(rng, 3.0, 50.0)), s_d=u(0.2, 0.6),
                   l_d=float(logu(rng, 50, 1000)),
                   c_y=float(logu(rng, 0.002, 0.05)))
        prm.update(c_n=u(0.2, 1.0) * prm["c_y"], l_b=u(1.0, 4.0) * ctx.B)
        return prm

    def init_state(self, params, B=1, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        return dict(rng=rng, xh=rng.normal(0.0, 1.0, B),
                    xd=rng.normal(0.0, 1.0, B))

    @staticmethod
    def fields(prm, ctx, state):
        h = np.maximum(prm["h_bar"] * np.exp(prm["s_h"] * state["xh"]),
                       2.0 * ctx.T)
        d = prm["d_bar"] * np.exp(prm["s_d"] * state["xd"])
        return h, d

    def terms(self, prm, ctx, q, h, d):
        B = q["B"]
        ur = q["u_r"]
        Fh = np.abs(ur) / np.sqrt(GRAV * h)
        x = (Fh - 1.0) / prm["w"]
        f = (2.0 * ctx.T / h) ** 2
        S = smoothstep(x)
        acc = np.zeros((B, 5))
        acc[:, 0] = -prm["a_h"] * bump(x) * ctx.k_drag * ur * np.abs(ur) \
            / ctx.m_u
        acc[:, 3] = ctx.wh ** 2 * (prm["dz_s"] * S + prm["dz_T"] * f)
        acc[:, 4] = ctx.wp ** 2 * ctx.sp * (prm["dth_s"] * S
                                            + prm["dth_T"] * f)
        Q = np.zeros((B, 5))
        if prm["bank"]:
            e = np.exp(-d / prm["l_b"]) * 0.5 * RHO_W * ur * np.abs(ur) \
                * ctx.L * ctx.T
            Q[:, 1] = prm["side"] * prm["c_y"] * e           # towards bank
            Q[:, 2] = -prm["side"] * prm["c_n"] * ctx.L * e  # bow away
        return acc, Q, Fh

    def step(self, prm, state, q, dt):
        ctx = q["ctx"]
        h, d = self.fields(prm, ctx, state)
        acc, Q, _ = self.terms(prm, ctx, q, h, d)
        ds = q["U_g"] * dt
        state["xh"] = CB.ou_step(state["xh"], prm["l_h"], ds, state["rng"])
        state["xd"] = CB.ou_step(state["xd"], prm["l_d"], ds, state["rng"])
        return CB.ItemOut(acc=acc, Q=Q, state=state)


# =================================================================== E3
@CB.register
class E3(CB.CatItem):
    code, stage = "E3", "force"

    def draw_on(self, rng, ctx):
        return dict(rate=float(logu(rng, 1e-4, 3e-3)), p_dmg=0.3)

    def init_state(self, params, B=1, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        return dict(rng=rng, rem=np.zeros((B, 5)), rem_t=np.zeros(B),
                    n=np.zeros(B, int))

    @staticmethod
    def event(ctx, rng, n, U_r, th):
        """n events: velocity jumps (n, 5) through M_t and their du."""
        du = logu(rng, 0.05, 1.5, n)
        du = np.minimum(du, 0.5 * np.maximum(U_r, 0.0))   # debris <= boat
        d = logu(rng, 0.02, 0.2, n)
        x = rng.uniform(0.0, ctx.x_st[-1], n)
        y = np.clip(rng.normal(0.0, 0.25 * ctx.B, n), -0.5 * ctx.B,
                    0.5 * ctx.B)
        z = np.interp(x, ctx.x_st, ctx.keel5)
        e = np.stack([-np.ones(n), rng.normal(0.0, 0.2, n),
                      rng.uniform(0.0, 1.0, n)], 1)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        J = (ctx.m * du)[:, None] * e
        return CB.to_acc(ctx, CB.lever_Q(ctx, (x, y, z), J, th)), du, d

    def step(self, prm, state, q, dt):
        ctx, B = q["ctx"], q["B"]
        rng = state["rng"]
        hit = CB.poisson_hit(prm["rate"], dt, rng, B)
        if hit.any():
            n = int(hit.sum())
            dnu, _, d = self.event(ctx, rng, n, q["U_r"][hit],
                                   q["sr"][hit, 5])
            state["rem"][hit] += dnu
            state["rem_t"][hit] = np.maximum(state["rem_t"][hit], d)
            state["n"][hit] += 1
            dmg = np.zeros(B, bool)
            dmg[hit] = rng.random(n) < prm["p_dmg"]
            q["debris_hit"] = np.asarray(q.get("debris_hit", np.zeros(
                B, bool))) | dmg
        rt = state["rem_t"]
        f = np.where(rt > 0, np.minimum(dt, rt) / np.maximum(rt, 1e-12), 0.0)
        imp = state["rem"] * f[:, None]
        state["rem"] = state["rem"] - imp
        state["rem_t"] = np.maximum(rt - dt, 0.0)
        state["rem"][state["rem_t"] <= 0] = 0.0
        return CB.ItemOut(imp=imp, state=state)


# =================================================================== E4
def packet_sea(pk, ctx, sr, t):
    """A packet's elevation, moving-station rate and orbital velocity at
    the 5 x 3 stations and the intake, for rows with an active packet dict
    pk of (B,) arrays A, w, k, beta, phi, tk, sig (A = 0: none)."""
    X, Y, Xd, Yd = station_xy(ctx, sr)
    t = np.broadcast_to(np.asarray(t, float), (len(sr),))
    cb, sb = np.cos(pk["beta"])[:, None], np.sin(pk["beta"])[:, None]
    xk, xkd = X * cb + Y * sb, Xd * cb + Yd * sb
    tau = ((t - pk["tk"]) / pk["sig"])[:, None]
    g = np.exp(-0.5 * tau * tau)
    live = (np.abs(tau) < 3.0) & (pk["A"][:, None] > 0)
    E = np.where(live, (g - E4_G3) / (1.0 - E4_G3), 0.0)
    Ed = np.where(live, -tau * g / (1.0 - E4_G3), 0.0) \
        / pk["sig"][:, None]
    k, w, A = pk["k"][:, None], pk["w"][:, None], pk["A"][:, None]
    ph = k * xk - w * (t - pk["tk"])[:, None] + pk["phi"][:, None]
    cph, sph = np.cos(ph), np.sin(ph)
    eta = A * E * cph
    etad = A * (Ed * cph + E * sph * (w - k * xkd))
    uo = A * w * E * cph
    rel = pk["beta"][:, None] - sr[:, 7][:, None]
    B = len(sr)
    f = lambda z: z[:, :15].reshape(B, 5, 3)                # noqa: E731
    return dict(eta=f(eta), etad=f(etad), eta_in=eta[:, 15],
                uorb=f(uo * np.cos(rel)), vorb=f(uo * np.sin(rel)))


def lofi_wave_response(ctx, eta0, eta_p):
    """The low-fidelity boat's own acceleration change (B, 5) when its sea
    gains eta_p: heave / pitch targets (reduced.py's rule, wh^2 and wp^2
    times the target shift) and its (zero-coefficient today) sway / yaw
    wave terms."""
    B = len(eta0)
    ec0, ecp = eta0[:, :, 1], eta_p[:, :, 1]
    s0 = (ec0 * ctx.xc).sum(1) / ctx.den
    sp_ = (ecp * ctx.xc).sum(1) / ctx.den
    acc = np.zeros((B, 5))
    acc[:, 3] = ctx.wh ** 2 * ctx.kwh * ecp.mean(1)
    acc[:, 4] = ctx.wp ** 2 * ctx.kwp * ctx.sp * (np.arctan(s0 + sp_)
                                                  - np.arctan(s0))
    tsl = (eta_p[:, :, 2] - eta_p[:, :, 0]) / ctx.B
    acc[:, 1] = ctx.p.get("c_wave_sway", 0.0) * GRAV * tsl.mean(1)
    acc[:, 2] = ctx.p.get("c_wave_yaw", 0.0) * GRAV \
        * (tsl * ctx.xc).sum(1) / ctx.den
    return acc


@CB.register
class E4(CB.CatItem):
    code, stage = "E4", "env"
    gated = True

    def draw_on(self, rng, ctx):
        return dict(rate=float(logu(rng, 1 / 1800, 1 / 120)),
                    own=bool(rng.random() < 0.5),
                    p_trig=rng.uniform(0.2, 0.8))

    def init_state(self, params, B=1, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        z = np.zeros(B)
        pk = dict(A=z.copy(), w=np.ones(B), k=np.ones(B), beta=z.copy(),
                  phi=z.copy(), tk=z.copy(), sig=np.ones(B))
        return dict(rng=rng, pk=pk, t=0.0, ring=None, chi=None, t_ring=0.0,
                    armed=np.ones(B, bool), n=np.zeros(B, int))

    @staticmethod
    def new_packets(pk, rows, rng, t, beta):
        n = len(rows)
        A = logu(rng, 0.05, 0.6, n)
        Tp = rng.uniform(1.5, 5.0, n)
        lam = GRAV * Tp * Tp / TWO_PI
        A = np.minimum(A, lam / 14.0)           # height <= lambda / 7
        w = TWO_PI / Tp
        sig = rng.integers(3, 11, n) * Tp / 4.0  # 3-10 waves in +-2 sigma
        pk["A"][rows], pk["w"][rows], pk["k"][rows] = A, w, w * w / GRAV
        pk["beta"][rows], pk["phi"][rows] = beta, rng.uniform(0, TWO_PI, n)
        pk["sig"][rows], pk["tk"][rows] = sig, t + 3.0 * sig

    def step(self, prm, state, q, dt):
        ctx, sr, B = q["ctx"], q["sr"], q["B"]
        rng, pk, t = state["rng"], state["pk"], state["t"]
        psi = sr[:, 7]
        chi = psi + np.arctan2(sr[:, 9], sr[:, 2])      # course over ground
        if state["ring"] is None:
            state["ring"] = np.repeat(psi[:, None], E4_RING_N, 1)
            state["chi"] = np.repeat(chi[:, None], E4_RING_N, 1)
            state["t_ring"] = t
        if t - state["t_ring"] >= E4_RING_DT - 1e-9:
            state["ring"] = np.concatenate([state["ring"][:, 1:],
                                            psi[:, None]], 1)
            state["chi"] = np.concatenate([state["chi"][:, 1:],
                                           chi[:, None]], 1)
            state["t_ring"] = t
        active = pk["A"] > 0
        new = CB.poisson_hit(prm["rate"], dt, rng, B) & ~active
        beta = rng.uniform(0, TWO_PI, B)
        if prm["own"]:
            cum = np.abs(wrap(np.diff(state["ring"], axis=1))).sum(1)
            over = cum > E4_TURN
            fire = over & state["armed"] & ~active & ~new \
                & (rng.random(B) < prm["p_trig"])
            state["armed"] = np.where(over, False, True)
            lag = rng.integers(30, E4_RING_N, B)       # 30-60 s back
            old = state["chi"][np.arange(B), E4_RING_N - 1 - lag]
            beta = np.where(fire, old, beta)
            new = new | fire
        if new.any():
            rows = np.nonzero(new)[0]
            self.new_packets(pk, rows, rng, t, beta[rows])
            state["n"][rows] += 1
        acc = np.zeros((B, 5))
        if (pk["A"] > 0).any():
            s = packet_sea(pk, ctx, sr, t)
            q["eta_add"] = q["eta_add"] + s["eta"]
            q["etad_add"] = q["etad_add"] + s["etad"]
            q["eta_in_add"] = q["eta_in_add"] + s["eta_in"]
            q["uorb"] = q["uorb"] + s["uorb"]
            q["vorb"] = q["vorb"] + s["vorb"]
            acc = lofi_wave_response(ctx, q["eta0"], s["eta"])
        done = (pk["A"] > 0) & (t >= pk["tk"] + 3.0 * pk["sig"])
        pk["A"][done] = 0.0
        state["t"] = t + dt
        return CB.ItemOut(acc=acc, state=state)


# ================================================================ checks
def _run(ctx, sr, sea=None, thr=0.0, noz=0.0):
    sea = CB.CatSea(None, ctx).sample(sr, 0.0) if sea is None else sea
    return CB.substep_quantities(ctx, sr, thr, noz, sea)


def _rows(ctx, u, psi=0.0, v=0.0, r=0.0):
    u = np.atleast_1d(np.asarray(u, float))
    sr = np.zeros((len(u), 10))
    sr[:, 2], sr[:, 3], sr[:, 5] = u, ctx.z0, ctx.th0
    sr[:, 7], sr[:, 8], sr[:, 9] = psi, r, v
    return sr


@CB.check("A1")
def a1_zero_signs_symmetry(item, ctx):
    """No wind -> exactly 0; head wind slows the boat; wind from port
    pushes to starboard and heels to starboard (K_roll > 0); mirrored
    wind angle flips sway and yaw, keeps surge (symmetric shapes)."""
    rng = np.random.default_rng(21)
    sr = _rows(ctx, [15.0, 15.0])
    worst, ay = 0.0, []
    for _ in range(40):
        prm = item.draw_on(rng, ctx)
        prm["asym"] = np.zeros(3)
        prm["U10"], prm["wave_rel"], prm["psi_u"] = 0.0, False, 0.0
        q = _run(ctx, sr)
        worst = max(worst, float(np.abs(item.step(
            prm, item.init_state(prm, 2), q, CB.DT_SUB).Q).max()))
        # head wind (air moving -x), beam wind from port (air moving -y)
        prm["U10"] = 10.0 / prm["zr"]
        for psi_u, chk in ((np.pi, "head"), (-0.5 * np.pi, "port")):
            prm["psi_u"] = psi_u
            st = item.init_state(prm, 2)
            st["x"][:] = 0.0
            q = _run(ctx, sr)
            Q = item.step(prm, st, q, CB.DT_SUB).Q[0]
            if chk == "head" and not Q[0] < 0:
                return False, f"head wind surge {Q[0]:.3g} not negative"
            if chk == "port":
                if not (Q[1] < 0 and q["K_roll"][0] > 0):
                    return False, "beam wind from port: wrong sway / heel"
                ay.append(CB.to_acc(ctx, Q[None])[0, 1])
        gam = np.linspace(-3.0, 3.0, 13)
        a, b = wind_coeffs(prm, gam), wind_coeffs(prm, -gam)
        if not (np.allclose(a[0], b[0]) and np.allclose(a[1], -b[1])
                and np.allclose(a[2], -b[2])):
            return False, f"blend={prm['blend']}: not mirror symmetric"
    ay = np.abs(ay)
    return worst == 0.0, (f"calm max |Q| {worst:.1e}; beam 10 m/s at 15 m/s"
                          f": sway {np.median(ay):.2f} m/s^2 median "
                          f"({ay.min():.2f}-{ay.max():.2f}; draft hand "
                          "value 0.38)")


def _wind_q(ctx, B):
    """A1 with 10 m/s of head wind at the boat's height, no drift."""
    a1 = CB.REGISTRY["A1"]
    prm = a1.draw_on(np.random.default_rng(5), ctx)
    prm.update(U10=10.0 / prm["zr"], wave_rel=False, psi_u=np.pi,
               tau_d=1e15)
    st = a1.init_state(prm, B)
    st["x"][:] = 0.0
    return a1, prm, st


@CB.check("A2")
def a2_gust_spectrum_and_squall(item, ctx):
    """Gust sd = I_u U within 15 %; spectrum tail slope -5/3 (OU sum) or
    -2 (Dryden) +- 0.25 on 8-240 corner frequencies; squalls keep the mean
    speed within [U, U + U10 zr] and turn <= 90 deg."""
    B, N, dt = 32, 4096, CB.DT_SUB
    sr = _rows(ctx, np.full(B, 15.0))
    msgs = []
    for dry in (False, True):
        prm = item.draw_on(np.random.default_rng(3), ctx)
        prm.update(dryden=dry, L_u=200.0, sq_rate=1e-12, I_u=0.15)
        a1, p1, s1 = _wind_q(ctx, B)
        st = item.init_state(prm, B, rng=np.random.default_rng(4))
        g = np.empty((N, B))
        for j in range(N):
            q = _run(ctx, sr)
            a1.step(p1, s1, q, dt)
            item.step(prm, st, q, dt)
            w = q["wind"]
            e1 = w["Vmean2"] / np.hypot(*w["Vmean2"].T)[:, None]
            g[j] = ((w["Vtot"] - w["Vmean2"]) * e1).sum(1)
        sd = g.std() / (prm["I_u"] * 10.0)
        F = np.fft.rfft(g - g.mean(0), axis=0)
        P = (np.abs(F) ** 2).mean(1)
        om = np.fft.rfftfreq(N, dt) * TWO_PI
        wc = 25.0 / prm["L_u"]
        i = (om > 8 * wc) & (om < 240 * wc)
        sl = np.polyfit(np.log(om[i]), np.log(P[i]), 1)[0]
        want = -2.0 if dry else -5.0 / 3.0
        if abs(sd - 1) > 0.15 or abs(sl - want) > 0.25:
            return False, f"dryden={dry}: sd ratio {sd:.2f}, slope {sl:.2f}"
        msgs.append(f"{'Dryden' if dry else 'OU-sum'} sd ratio {sd:.2f} "
                    f"slope {sl:.2f}")
    prm = item.draw_on(np.random.default_rng(6), ctx)
    prm.update(sq_rate=0.5, sq_tau=5.0)
    a1, p1, s1 = _wind_q(ctx, B)
    st = item.init_state(prm, B, rng=np.random.default_rng(7))
    lo, hi, turn = np.inf, 0.0, 0.0
    for j in range(500):
        q = _run(ctx, sr)
        a1.step(p1, s1, q, dt)
        item.step(prm, st, q, dt)
        V = q["wind"]["Vmean2"]
        s = np.hypot(*V.T)
        lo, hi = min(lo, s.min()), max(hi, s.max())
        turn = max(turn, float(np.abs(wrap(np.arctan2(V[:, 1], V[:, 0])
                                           - np.pi)).max()))
    ok = lo >= 10.0 - 1e-9 and hi <= 20.0 + 1e-9 and turn <= 0.5 * np.pi
    return ok, "; ".join(msgs) + (f"; squall mean speed {lo:.1f}-{hi:.1f} "
                                  f"m/s, turn {np.degrees(turn):.0f} deg")


@CB.check("A3")
def a3_reference_ratios(item, ctx):
    """At calm running at u_max without wind: lift / W = lambda_L and the
    aerodynamic bow-up stiffness / (w_p^2 I_q) = kappa_theta (accepted
    draws), destabilising (dM_up / dtheta > 0); lift scales with the
    apparent head-wind dynamic pressure; ground effect falls with bow
    height."""
    rng = np.random.default_rng(31)
    sr = _rows(ctx, [ctx.u_max])
    I = float(ctx.Mt[4, 4])
    tries, bad, worst = [], 0, 0.0
    for _ in range(60):
        prm = item.draw_on(rng, ctx)
        tries.append(prm["tries"])
        q = _run(ctx, sr)
        Fz, _, xac = item.forces(prm, q)
        e_l = abs(Fz[0] / ctx.W / prm["lam"] - 1)
        h = 1e-6
        M = []
        for d in (h, -h):
            qq = dict(q)
            qq["trim_rel"] = q["trim_rel"] + d
            f, _, x = item.forces(prm, qq)
            M.append(f[0] * x[0])
        k = (M[0] - M[1]) / (2 * h) / (ctx.wp ** 2 * I)
        if k <= 0:
            return False, f"aerodynamic pitch stiffness {k:.3g} restoring"
        if 0.05 * ctx.L < prm["x_0"] < 0.35 * ctx.L:
            worst = max(worst, e_l, abs(k / prm["kap"] - 1))
        else:
            bad += 1
    # dynamic pressure: 15 m/s + 10 m/s head wind = 25 m/s in still air
    prm = item.draw_on(rng, ctx)
    q1 = _run(ctx, _rows(ctx, [25.0]))
    q2 = _run(ctx, _rows(ctx, [15.0]))
    q2["wind"] = dict(Vtot=np.array([[-10.0, 0.0]]), w_g=np.zeros(1))
    r = item.forces(prm, q2)[0][0] / item.forces(prm, q1)[0][0]
    s_hi = _rows(ctx, [20.0])
    s_hi[:, 3] += 0.3
    prm["c_g"] = 0.5
    g_ok = bool(item.forces(prm, _run(ctx, s_hi))[0][0]
                < item.forces(prm, _run(ctx, _rows(ctx, [20.0])))[0][0])
    ok = worst < 1e-4 and abs(r - 1) < 1e-12 and g_ok
    return ok, (f"ratios within {worst:.1e}; mean tries {np.mean(tries):.1f}"
                f" (acceptance {1 / np.mean(tries):.2f}), {bad} clipped of "
                f"60; head wind / still-air lift {r:.6f}; ground effect "
                f"falls with height {g_ok}")


def _e1_state(item, prm, B, V):
    st = item.init_state(prm, B)
    st["Vbar"] = np.tile(np.asarray(V, float), (B, 1))
    st["V"] = st["Vbar"].copy()
    st["x"][:] = 0.0
    return st


@CB.check("E1")
def e1_relative_velocity(item, ctx):
    """No current -> exactly 0; a boat drifting with the water (nu = nu_c,
    r = 0) loses exactly the low-fidelity boat's own surge drag and sway
    damping; a turning boat in a steady current gets A11 r v_c; speed
    through water = nu - nu_c; fronts keep the current offset within
    |V_c|."""
    prm = item.draw_on(np.random.default_rng(41), ctx)
    prm.update(tau_d=1e15, front_s=1e12)
    p = ctx.p
    q = _run(ctx, _rows(ctx, [10.0, 5.0]))
    o = item.step(prm, _e1_state(item, prm, 2, [0.0, 0.0]), q, CB.DT_SUB)
    z0 = float(np.abs(o.acc).max())
    uc, vc = 1.2, -0.7                     # heading 0: body = earth
    q = _run(ctx, _rows(ctx, [uc], v=vc))
    o = item.step(prm, _e1_state(item, prm, 1, [uc, vc]), q, CB.DT_SUB)
    want = np.array([ctx.k_drag * uc * abs(uc) / ctx.m_u,
                     (p["k_lin_sway"] * vc + p["k_sway"] * vc * abs(vc))
                     / ctx.m_v])
    e1 = float(np.abs(o.acc[0, :2] - want).max())
    stw = max(abs(o.obs["stw_u"][0]), abs(o.obs["stw_v"][0]))
    r = 0.3
    q = _run(ctx, _rows(ctx, [12.0], v=0.2, r=r))
    o = item.step(prm, _e1_state(item, prm, 1, [uc, vc]), q, CB.DT_SUB)
    ur = 12.0 - uc
    drag = -ctx.k_drag * (ur * abs(ur) - 144.0) / ctx.m_u
    e2 = abs(o.acc[0, 0] - drag - (ctx.m_u - ctx.m) * r * vc / ctx.m_u)
    prm2 = dict(prm, front_s=50.0, front_w=10.0)
    B = 16
    st = _e1_state(item, prm2, B, [0.8, 0.3])
    sr = _rows(ctx, np.full(B, 15.0))
    mx, yaw = 0.0, 0.0
    for _ in range(1500):
        o = item.step(prm2, st, _run(ctx, sr), CB.DT_SUB)
        mx = max(mx, float((np.hypot(*st["cur"].T)
                            / np.hypot(0.8, 0.3)).max()))
        yaw = max(yaw, float(np.abs(o.acc[:, 2]).max()))
    ok = z0 == 0.0 and e1 < 1e-6 and e2 < 1e-6 and stw < 1e-12 \
        and mx <= 1 + 1e-9 and np.isfinite(yaw)
    return ok, (f"no current {z0:.1e}; drift-with-water error {e1:.1e}; "
                f"A11 r v_c error {e2:.1e}; stw {stw:.1e}; front offset "
                f"<= {mx:.2f} |V_c|; max front yaw {yaw:.3f} rad/s^2")


@CB.check("E2")
def e2_depth_terms(item, ctx):
    """Deep water: nothing (< 1e-6 a_ref); at F_h = 1 the drag change is
    a_h k_drag u^2; banks off by default; a bank pulls the boat towards
    it and turns the bow away."""
    rng = np.random.default_rng(51)
    prm = item.draw_on(rng, ctx)
    u = 15.0
    q = _run(ctx, _rows(ctx, [u]))
    acc, Q, _ = item.terms(prm, ctx, q, np.array([1e4]), np.array([1e9]))
    deep = float((np.abs(acc) / ctx.a_ref).max())
    acc, _, _ = item.terms(prm, ctx, q, np.array([u * u / GRAV]),
                           np.array([1e9]))
    e_b = abs(acc[0, 0] + prm["a_h"] * ctx.k_drag * u * u / ctx.m_u)
    n_bank = sum(item.draw_on(rng, ctx)["bank"] for _ in range(200))
    prm.update(bank=True, side=1.0)
    _, Q, _ = item.terms(prm, ctx, q, np.array([1e4]), np.array([5.0]))
    ok = deep < 1e-6 and e_b < 1e-9 and n_bank == 0 and Q[0, 1] > 0 \
        and Q[0, 2] < 0
    return ok, (f"deep {deep:.1e} a_ref; bump error {e_b:.1e}; bank draws "
                f"{n_bank}/200; bank on port: sway {Q[0, 1]:.0f} N, yaw "
                f"{Q[0, 2]:.0f} N m")


@CB.check("E3")
def e3_impulse(item, ctx):
    """One event per row: the whole impulse arrives over its duration
    through the impulse channel; the surge jump is backwards and never
    exceeds min(1.5, U_r / 2) m/s; the largest 0.24 s average stays under
    the surge clip 12 m/s^2."""
    B = 400
    prm = dict(item.draw_on(np.random.default_rng(61), ctx), rate=1e9)
    st = item.init_state(prm, B, rng=np.random.default_rng(62))
    U = np.random.default_rng(63).uniform(0.5, 30.0, B)
    sr = _rows(ctx, U)
    tot = np.zeros((B, 5))
    run = []
    for j in range(8):
        o = item.step(prm, st, _run(ctx, sr), CB.DT_SUB)
        if j == 0:
            first = st["rem"] + o.imp
            prm["rate"] = 0.0
        tot += o.imp
        run.append(o.imp[:, 0])
    run = np.array(run)
    peak = max(float(np.abs(run[j:j + 6].sum(0)).max()) / 0.24
               for j in range(len(run) - 5))
    e = float(np.abs(tot - first).max())
    cap = np.minimum(1.5, 0.5 * U)
    ok = e < 1e-12 and np.all(tot[:, 0] <= 0) \
        and np.all(-tot[:, 0] <= cap + 1e-12) and peak < 12.0
    return ok, (f"delivered - event {e:.1e}; max surge jump "
                f"{-tot[:, 0].min():.2f} m/s; max 0.24 s average "
                f"{peak:.2f} m/s^2")


@CB.check("E4")
def e4_packet(item, ctx):
    """The packet's moving-station rate equals the finite difference of its
    elevation along the moving stations; the low-fidelity response equals
    lofi_accel_vertical(eta0 + packet) - lofi_accel_vertical(eta0); height
    <= lambda / 7; an own-wake packet fires once after a 200 deg turn."""
    rng = np.random.default_rng(71)
    B = 8
    st = item.init_state(item.draw_on(rng, ctx), B, rng=rng)
    pk = st["pk"]
    item.new_packets(pk, np.arange(B), rng, 0.0, rng.uniform(0, TWO_PI, B))
    sr = _rows(ctx, rng.uniform(3, 25, B), psi=rng.uniform(-3, 3, B),
               v=rng.uniform(-1, 1, B), r=rng.uniform(-0.4, 0.4, B))
    sr[:, 0], sr[:, 1] = rng.uniform(-50, 50, (2, B))
    t = pk["tk"] - 0.3 * pk["sig"]
    s0 = packet_sea(pk, ctx, sr, t)
    h = 1e-6
    s1 = sr.copy()
    c, s = np.cos(sr[:, 7]), np.sin(sr[:, 7])
    s1[:, 0] += (sr[:, 2] * c - sr[:, 9] * s) * h
    s1[:, 1] += (sr[:, 2] * s + sr[:, 9] * c) * h
    s1[:, 7] += sr[:, 8] * h
    fd = (packet_sea(pk, ctx, s1, t + h)["eta"] - s0["eta"]) / h
    e_r = float(np.abs(fd - s0["etad"]).max() / np.abs(s0["etad"]).max())
    eta0 = rng.normal(0, 0.2, (B, 5, 3))
    a = lofi_wave_response(ctx, eta0, s0["eta"])
    d = CB.lofi_accel_vertical(ctx, sr, eta0 + s0["eta"]) \
        - CB.lofi_accel_vertical(ctx, sr, eta0)
    e_l = float(np.abs(a[:, 3:5] - d).max())
    steep = float((2 * pk["A"] / (TWO_PI / pk["k"])).max())
    # own wake: turn at 0.1 rad/s for 35 s, then straight for 45 s
    prm = dict(rate=0.0, own=True, p_trig=1.0)
    st = item.init_state(prm, 1, rng=np.random.default_rng(72))
    s_ = _rows(ctx, [10.0])
    fired = []
    for j in range(2000):
        s_[0, 8] = 0.1 if j < 875 else 0.0
        item.step(prm, st, _run(ctx, s_), CB.DT_SUB)
        if st["n"][0] > len(fired):
            fired.append((j, round(float(st["pk"]["beta"][0]), 3)))
        s_[0, 7] += s_[0, 8] * CB.DT_SUB
    ok = e_r < 1e-4 and e_l < 1e-9 and steep <= 1 / 7 + 1e-12 \
        and len(fired) == 1
    return ok, (f"rate vs finite difference {e_r:.1e}; low-fi response "
                f"error {e_l:.1e}; max H / lambda {steep:.3f}; own-wake "
                f"packets (substep, direction) {fired}")


@CB.check("A1")
def air_env_chain_assembled(item, ctx):
    """CatDraw with A1, A2, A3, E1, E4 forced on (no layer 1, no layer 3
    force): a 5 s closed loop in a sea stays finite; the stored signals
    are the items' values (speed through water, wind relative to air)."""
    from control.reduced import ReducedModel
    from learn.meta import operators_rb as R
    codes = ("A1", "A2", "A3", "E1", "E4")
    d = CB.CatDraw(3, tier="none", layer1=False, items=codes,
                   force_items={c: True for c in codes}, sparse=False)
    B = 4
    st = d.new_state(B)
    rs = R.RowSeas([R.sea_state(R.sea_dict(np.random.default_rng(8)), 8)]
                   * B, d.ctx.x_st, d.ctx.y_off)
    bind_wave_dir(st["items"]["A1"], rs)
    red = ReducedModel(d.ctx.p)
    cs = CB.CatSea(rs, d.ctx)
    sr = _rows(d.ctx, [8.0, 12.0, 16.0, 21.0])
    thr = d.ctx.k_drag * sr[:, 2] ** 2
    noz = np.full(B, 0.1)
    rv = list(R.RED_VEL)
    pmax = 0.0
    for j in range(125):
        sea = cs.sample(sr, j * CB.DT_SUB)
        ns = red.step(sr, thr, noz, sea["eta"], d.ctx.x_st, CB.DT_SUB)[0]
        nu0 = (ns[:, rv] - sr[:, rv]) / CB.DT_SUB
        o = d.substep(st, sr, thr, noz, sea, nu0=nu0, t=j * CB.DT_SUB)
        rc, _ = CB.clip_push(o["acc"], np.zeros_like(o["acc"]))
        pmax = max(pmax, float(np.abs(rc).max()))
        ns[:, rv] += rc * CB.DT_SUB + o["imp"]
        sr = ns
    q, ob = o["q"], o["obs"]
    e_stw = float(np.abs(ob["stw_u"] - q["u_r"]).max())
    d_w = float(np.abs(ob["wind_speed_rel"] - q["U_g"]).max())
    fin = bool(np.isfinite(sr).all())
    ok = fin and e_stw < 1e-12 and "wind" in q
    return ok, (f"finite {fin}; max |push| {pmax:.2f}; stw = u_r "
                f"({e_stw:.1e}); relative wind differs from still air by up "
                f"to {d_w:.2f} m/s; items {sorted(d.on)}")
