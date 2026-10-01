#!/usr/bin/env python3
"""
Observation model O1-O5 of the D10 force-catalogue prior
(learn/meta/PRIOR_D10_DRAFT.md section 4 with sections 10-11; framework
learn/meta/cat_base.py, slot ObsItem / OBS_REGISTRY).

What it is. The sensors and the logger of the real boat, applied to a
RECORDED closed-loop trajectory after the fact: it never acts on the
dynamics and never changes its input (run_obs works on copies; checked).
The history inputs and the e0 labels of a later stage are then computed on
the observed values (draft section 4; e0_observed). OFF in D10a data
(cat_base.OBS_ON_D10A = False: ObsModel returns the true values sampled at
the control steps); CatDraw(obs_on=True) must be used together with it so
that the old G12 observation noise is switched off (draft change 11).

Input: the recorded truth on the uniform substep grid (0.04 s), a dict
  t      (N,)      times
  sr     (N, 10)   reduced state (x, y, u, z, zd, th, thd, psi, r, v) in the
                   low-fidelity conventions (u, v level-frame body speeds, z
                   heave up, bow-up angle = s_p th, y and the roll phi to
                   port positive, as cat_base.derive_quantities)
  cmd    (N, 2)    physical commands (thrust N, nozzle rad), optional
  extra  {name: (N,)} the section-11 signals of CatDraw.substep()['obs']
                   (roll, roll_rate, rpm as a fraction of rated,
                   wind_speed_rel, wind_angle_rel, stw_u, stw_v, gnss_q),
                   optional; roll / roll_rate enter the kinematics
  acc    (N, 5)    true accelerations (du, dv, dr, zdd, thdd), optional
                   (else central differences of the velocities)
  sub or k_out     output every `sub` samples (default the 0.24 s control
                   step) or at the given indices
Output: dict(t, k, sr (M, 10) observed, cmd (M, 2) as logged, extra
{name: (M,)} observed, obs_on, items).

Full rigid-body kinematics (no small-angle shortcuts in the sensor part):
attitude R = Rz(psi) Ry(beta) Rx(phi), beta = -s_p th (right-handed about
the port axis, bow down positive), body rates from the ZYX Euler rates;
the cg acceleration in the level (heading) frame a = (du - r v, dv + r u,
zdd); a hull point r_p moves with V + R (w x r_p) and has the kinematic
acceleration a + R_l (w' x r_p + w x (w x r_p)); an accelerometer reads
the specific force R_l^T (a_p + g e_z) in body axes.

Pipeline (STAGE_ORDER; an absent item is ideal):
  O5 mount     antenna and IMU points: v_ant, p_ant, IMU specific force
  O2 imu       gyro and accelerometer errors, vibration, range clipping
  O3 attitude  roll / pitch / heading estimates (acceleration tilt error)
  O1 gnss      rate, latency, noise, dropouts (hold or INS bridge), RTK
  O3 heading   dual-antenna heading bridged by the gyro in dropouts
  O5 convert   the logger's lever-arm compensation (calibrated arms with
               error) and ground -> body rotation by the MEASURED heading
  O3 heave     heave from GNSS height or high-passed INS heave
  O4 output    estimator low-pass and delay per signal group, time-stamp
               jitter, command-state offset, the plain section-11 sensors
               (rpm, anemometer, speed log), resampling to the control steps

Numerics (draft section 6 item 6): first-order filters exact for a
piecewise-linear input (lpf_exact, first-order hold); the second-order
heave high-pass by the exact 2 x 2 transition (cat_base.osc_phi); bias
random walks and noises exact in discrete time (noise sd = density /
sqrt(dt)); the vibration hull-ringing envelope an exact exponential.
Every item is on whenever the observation model is on (P_ON_OBS = 1; the
draft gives no switch probability for section 4); their sub-options are
drawn per episode (see each item).
"""
import numpy as np
from scipy.signal import lfilter

from learn.meta import cat_base as CB
from learn.meta.operators_rb import RED_VEL

GRAV, DEG = CB.GRAV, CB.DEG
MG, UG = 1e-3 * CB.GRAV, 1e-6 * CB.GRAV
E_Z = np.array([0.0, 0.0, 1.0])
SID_OBS, SID_OBS_NOISE = 900, 950          # streams [seed, 71, attempt, id]
P_ON_OBS = dict(O1=1.0, O2=1.0, O3=1.0, O4=1.0, O5=1.0)
STAGE_ORDER = (("O5", "mount"), ("O2", "imu"), ("O3", "attitude"),
               ("O1", "gnss"), ("O3", "heading"), ("O5", "convert"),
               ("O3", "heave"), ("O4", "output"))
O4_GROUPS = ("gnss", "imu", "est", "plain")
VIB_ORDERS = (0.5, 1.0, 1.5, 2.0, 3.0)     # engine orders (4-stroke)
DT_CTRL = 0.24


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


# ----------------------------------------------------------- kinematics
def rot(axis, a):
    """Batched rotation matrices (..., 3, 3) about 'x', 'y' or 'z'."""
    a = np.asarray(a, float)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    i, j = dict(x=(1, 2), y=(2, 0), z=(0, 1))[axis]
    k = 3 - i - j
    R[..., k, k] = 1.0
    R[..., i, i], R[..., j, j] = c, c
    R[..., i, j], R[..., j, i] = -s, s
    return R


def attitude_R(psi, beta, phi):
    return rot("z", psi) @ rot("y", beta) @ rot("x", phi)


def level_R(beta, phi):
    """Body -> level (heading) frame."""
    return rot("y", beta) @ rot("x", phi)


def body_rates(phi, beta, phid, betad, psid):
    """Body angular velocity (..., 3) from the ZYX Euler angles and rates."""
    return np.stack([phid - psid * np.sin(beta),
                     betad * np.cos(phi) + psid * np.cos(beta) * np.sin(phi),
                     -betad * np.sin(phi) + psid * np.cos(beta) * np.cos(phi)],
                    -1)


def euler_rates(phi, beta, w):
    """Inverse of body_rates: (phid, betad, psid)."""
    p, q, r = w[..., 0], w[..., 1], w[..., 2]
    a = q * np.sin(phi) + r * np.cos(phi)
    return p + a * np.tan(beta), q * np.cos(phi) - r * np.sin(phi), \
        a / np.cos(beta)


def mv(R, v):
    return np.einsum("nij,nj->ni", R, np.broadcast_to(v, R.shape[:-1]))


def mtv(R, v):
    return np.einsum("nji,nj->ni", R, np.broadcast_to(v, R.shape[:-1]))


def axis_angle(axis, ang):
    """Rotation matrix (3, 3) of angle ang about a unit axis (Rodrigues)."""
    k = np.asarray(axis, float)
    k = k / max(np.linalg.norm(k), 1e-300)
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx


def cumtrapz0(a, dt):
    """Cumulative trapezoid along axis 0, starting at 0."""
    a = np.asarray(a, float)
    out = np.zeros_like(a)
    out[1:] = np.cumsum(0.5 * (a[1:] + a[:-1]) * dt, axis=0)
    return out


def interp_cols(tq, t, A):
    A = np.asarray(A, float)
    if A.ndim == 1:
        return np.interp(tq, t, A)
    return np.stack([np.interp(tq, t, A[:, c]) for c in range(A.shape[1])],
                    1)


def lpf_exact(s, tau, dt):
    """First-order low-pass tau y' = s - y along axis 0, exact for s linear
    between samples (first-order hold): y_i = s_i - tau m + (y_{i-1} -
    s_{i-1} + tau m) e^{-dt/tau}, m = (s_i - s_{i-1}) / dt; y_0 = s_0. A
    ramp comes out delayed by exactly tau; tau -> 0 is the identity."""
    s = np.asarray(s, float)
    if not tau > 1e-9 * dt or len(s) < 2:
        return s.copy()
    a = np.exp(-dt / tau)
    m = np.diff(s, axis=0) / dt
    x = s[1:] - tau * m - a * (s[:-1] - tau * m)
    y = np.empty_like(s)
    y[0] = s[0]
    zi = (a * s[0])[None] if s.ndim > 1 else np.array([a * s[0]])
    y[1:] = lfilter([1.0], [1.0, -a], x, axis=0, zi=zi)[0]
    return y


def hpf2(s, sd, f_c, dt, zeta=np.sqrt(0.5)):
    """Second-order high-pass s^2 / (s^2 + 2 zeta w s + w^2) (Butterworth
    for zeta = 1/sqrt 2) of s with rate sd, as s - x with x the low-pass
    x'' = -w^2 (x - s_e) - 2 zeta w x' of s_e = s + (2 zeta / w) sd (so
    that X / S = (2 zeta w s + w^2) / (...)), advanced by the exact 2 x 2
    transition with s_e held at the step's mean value (draft 6 item 6);
    returns (s - x, sd - x'). DC gain exactly 0."""
    w = 2 * np.pi * f_c
    P = CB.osc_phi(w * w, 2 * zeta * w, dt)
    N = len(s)
    s_e = s + (2 * zeta / w) * np.asarray(sd, float)
    x, xd = np.empty(N), np.empty(N)
    x[0], xd[0] = s_e[0], 0.0
    for i in range(1, N):
        se = 0.5 * (s_e[i - 1] + s_e[i])
        e = x[i - 1] - se
        x[i] = se + P[0, 0] * e + P[0, 1] * xd[i - 1]
        xd[i] = P[1, 0] * e + P[1, 1] * xd[i - 1]
    return s - x, sd - xd


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def truth_kinematics(ctx, traj):
    """The true motion of the recorded trajectory (module docstring)."""
    t = np.asarray(traj["t"], float)
    N = len(t)
    sr = np.asarray(traj["sr"], float)
    if sr.shape != (N, 10) or N < 3:
        raise ValueError(f"cat_obs: sr {sr.shape} for {N} times")
    dts = np.diff(t)
    dt = float(np.median(dts))
    if np.abs(dts - dt).max() > 1e-6 * dt:
        raise ValueError("cat_obs: the truth must be on a uniform grid")
    x, y, u, z, zd, th, thd, psi, r, v = (sr[:, i].copy() for i in range(10))
    ex = {k: np.asarray(a, float).copy()
          for k, a in (traj.get("extra") or {}).items()}
    phi = ex.get("roll", np.zeros(N))
    phid = ex.get("roll_rate", np.zeros(N))
    acc = traj.get("acc")
    if acc is None:
        du, dv, zdd = (np.gradient(a, t) for a in (u, v, zd))
    else:
        acc = np.asarray(acc, float)
        du, dv, zdd = acc[:, 0].copy(), acc[:, 1].copy(), acc[:, 3].copy()
    sp = ctx.sp
    beta, betad = -sp * th, -sp * thd
    w = body_rates(phi, beta, phid, betad, r)
    cmd = traj.get("cmd")
    cmd = None if cmd is None else np.asarray(cmd, float).reshape(N, 2).copy()
    thr_frac = np.zeros(N) if cmd is None else cmd[:, 0] / ctx.t_max
    rpm = ex.get("rpm", np.sqrt(np.maximum(thr_frac, 0.0)))
    c, s = np.cos(psi), np.sin(psi)
    return dict(t=t, dt=dt, N=N, sr=sr.copy(), x=x, y=y, u=u, v=v, r=r, z=z,
                zd=zd, th=th, thd=thd, psi=psi, phi=phi, phid=phid,
                beta=beta, betad=betad, R=attitude_R(psi, beta, phi),
                Rl=level_R(beta, phi), w=w, wd=np.gradient(w, t, axis=0),
                a_lvl=np.stack([du - r * v, dv + r * u, zdd], 1),
                V=np.stack([u * c - v * s, u * s + v * c, zd], 1),
                P=np.stack([x, y, z], 1), cmd=cmd, thr_frac=thr_frac,
                rpm=np.asarray(rpm, float), extra=ex)


def point_kin(K, r_p):
    """Earth velocity, earth position, level-frame kinematic acceleration
    and body w x r of the hull point r_p (body axes from the cg)."""
    r_p = np.asarray(r_p, float)
    wr = np.cross(K["w"], r_p)
    V = K["V"] + mv(K["R"], wr)
    P = K["P"] + mv(K["R"], r_p)
    a_rot = np.cross(K["wd"], r_p) + np.cross(K["w"], wr)
    return V, P, K["a_lvl"] + mv(K["Rl"], a_rot), wr


# ------------------------------------------------------------ the stages
def st_mount(prm, W, rng):
    """O5 part 1: antenna and IMU points (absent: both at the cg)."""
    K = W["K"]
    z3 = np.zeros(3)
    if prm is None:
        r_a = r_i = ra_hat = ri_hat = z3
        comp = True
    else:
        r_a, r_i = np.asarray(prm["r_a"]), np.asarray(prm["r_i"])
        ra_hat = r_a + np.asarray(prm["cal_a"])
        ri_hat = r_i + np.asarray(prm["cal_i"])
        comp = bool(prm["comp"] > 0.5)
    Va, Pa, aa, wra = point_kin(K, r_a)
    Vi, Pi, ai, _ = point_kin(K, r_i)
    W["mount"] = dict(r_a=r_a, r_i=r_i, ra_hat=ra_hat, ri_hat=ri_hat,
                      comp=comp, V_ant=Va, P_ant=Pa,
                      a_ant=np.linalg.norm(aa, axis=1), wr_a=wra,
                      a_imu=ai, P_imu=Pi, f_b=mtv(K["Rl"], ai + GRAV * E_Z))


def vibration(prm, K, f_true, rng):
    """a_vib (N, 3): engine-order sinusoids at rpm x f_rated (amplitude
    ~ rpm^2 at the rated value), each through the IMU's anti-alias low-pass
    |H| = 1 / sqrt(1 + (f / f_aa)^4) and sampled on the grid (aliasing
    included), plus hull ringing at f_h (draft 3.4) whose envelope is
    kicked by the jumps of the vertical specific force and decays exactly
    with exp(-2 pi f_h zeta_h dt)."""
    t, dt = K["t"], K["dt"]
    rpm = np.maximum(K["rpm"], 0.0)
    ph = 2 * np.pi * cumtrapz0(prm["f_rated"] * rpm, dt)
    H = lambda f: 1.0 / np.sqrt(1.0 + (f / prm["f_aa"]) ** 4)  # noqa: E731
    a = np.zeros((K["N"], 3))
    for o, amp, d, p0 in zip(prm["vib_orders"], prm["vib_amp"],
                             prm["vib_dirs"], prm["vib_ph"]):
        comp = amp * rpm ** 2 * H(o * prm["f_rated"] * rpm) \
            * np.sin(o * ph + p0)
        a += comp[:, None] * np.asarray(d)[None]
    if prm["k_h"] > 0:
        dfz = np.abs(np.diff(f_true[:, 2], prepend=f_true[0, 2]))
        ah = np.exp(-2 * np.pi * prm["f_h"] * prm["z_h"] * dt)
        env = lfilter([prm["k_h"]], [1.0, -ah], dfz)
        a[:, 2] += env * H(prm["f_h"]) * np.sin(2 * np.pi * prm["f_h"] * t
                                                + prm["ph_h"])
    return a


def st_imu(prm, W, rng):
    """O2: w_m = (1 + s_g) R_mis w + b_g(t) + G_s a + n_g,
    f_m = clip((1 + s_a) R_mis f + b_a(t) + n_a + a_vib, +- a_FS)."""
    K, M = W["K"], W["mount"]
    w, f = K["w"], M["f_b"]
    if prm is None:
        W["imu"] = dict(w_m=w.copy(), f_m=f.copy())
        return
    N, dt = K["N"], K["dt"]
    Rm = axis_angle(prm["mis_axis"], prm["mis"])
    a_vib = vibration(prm, K, f, rng)
    fv = f + a_vib
    sq = np.sqrt(dt)
    bg = np.asarray(prm["bg0"]) + np.cumsum(
        rng.normal(0.0, prm["bgw"] * sq, (N, 3)), 0)
    ba = np.asarray(prm["ba0"]) + np.cumsum(
        rng.normal(0.0, prm["baw"] * sq, (N, 3)), 0)
    G = prm["gs"] * np.asarray(prm["Gs"])
    w_m = (1 + np.asarray(prm["sg"])) * (w @ Rm.T) + bg + fv @ G.T \
        + rng.normal(0.0, prm["ng"] / sq, (N, 3))
    fs = prm["fs"]
    f_m = np.clip((1 + np.asarray(prm["sa"])) * (f @ Rm.T) + ba
                  + rng.normal(0.0, prm["na"] / sq, (N, 3)) + a_vib, -fs, fs)
    W["imu"] = dict(w_m=w_m, f_m=f_m, a_vib=a_vib)


def st_attitude(prm, W, rng):
    """O3 part 1: bow-up_m = bow-up + kappa_a LPF_tau_a(a_x / g) + b + n,
    roll the same with a_y (a the level-frame kinematic acceleration of the
    IMU point: a sustained acceleration tilts the specific force); heading
    magnetic psi + b0 + b1 c + n (c the throttle fraction) or dual-antenna
    GNSS psi + b0 + n."""
    K = W["K"]
    if prm is None:
        W["att"] = dict(phi=K["phi"].copy(), th=K["th"].copy(),
                        beta=K["beta"].copy(), psi=K["psi"].copy())
        return
    N, dt, sp = K["N"], K["dt"], W["ctx"].sp
    aI = W["mount"]["a_imu"]
    L = lpf_exact(aI[:, :2] / GRAV, prm["tau_a"], dt)
    bow = prm["kappa_a"] * L[:, 0] + prm["b_th"] \
        + rng.normal(0.0, prm["n_th"], N)
    th = K["th"] + sp * bow
    phi = K["phi"] + prm["kappa_a"] * L[:, 1] + prm["b_phi"] \
        + rng.normal(0.0, prm["n_phi"], N)
    psi = K["psi"] + prm["b_psi0"] + prm["b_psi1"] * K["thr_frac"] \
        + rng.normal(0.0, prm["n_psi"], N)
    W["att"] = dict(phi=phi, th=th, beta=-sp * th, psi=psi)


def st_gnss(prm, W, rng):
    """O1: fixes at epochs t_j (rate f, random phase) of the antenna's
    truth + position bias (RTK jumps) + noise; the fix of epoch j is
    available from t_j + tau_g and held (zero-order hold); Gilbert-Elliott
    good / bad states per epoch and loss of lock w.p. p_hd when the antenna
    acceleration exceeds 4 g; a lost epoch is bridged by holding the last
    good fix or by INS dead reckoning from it (measured specific force
    rotated by the estimated attitude, lever arm antenna - IMU with the
    calibrated arms). q = 1 while the latest available epoch is a fix."""
    K, M = W["K"], W["mount"]
    t, dt, N = K["t"], K["dt"], K["N"]
    if prm is None:
        W["gnss"] = dict(vel=M["V_ant"].copy(), pos=M["P_ant"].copy(),
                         q=np.ones(N))
        return
    f = float(prm["rate"])
    tj = t[0] + prm["t_off"] / f + np.arange(
        int(np.floor((t[-1] - t[0]) * f - prm["t_off"])) + 1) / f
    Mj = len(tj)
    if Mj == 0:                                # shorter than one epoch
        W["gnss"] = dict(vel=M["V_ant"].copy(), pos=M["P_ant"].copy(),
                         q=np.zeros(N))
        return
    Vj, Pj = interp_cols(tj, t, M["V_ant"]), interp_cols(tj, t, M["P_ant"])
    a_ep = np.interp(tj, t, M["a_ant"])
    # Gilbert-Elliott chain, loss of lock at high acceleration
    pg = -np.expm1(-1.0 / (f * prm["T_good"]))
    pb = -np.expm1(-1.0 / (f * prm["T_bad"]))
    s = rng.random() < prm["T_good"] / (prm["T_good"] + prm["T_bad"])
    uu = rng.random(Mj)
    good = np.empty(Mj, bool)
    for j in range(Mj):
        good[j] = s
        s = (uu[j] >= pg) if s else (uu[j] < pb)
    hd = (a_ep > 4.0 * GRAV) & (rng.random(Mj) < prm["p_hd"])
    ok = good & ~hd
    # RTK position jumps: a piecewise-constant bias; each event either
    # returns a wrong fix to 0 (w.p. 0.5, if biased) or adds a jump
    jumps = np.flatnonzero(rng.random(Mj) < -np.expm1(-prm["rtk_rate"] / f))
    bias = np.zeros((Mj, 3))
    cur, prev = np.zeros(3), 0
    for j in jumps:
        bias[prev:j] = cur
        if cur.any() and rng.random() < 0.5:
            cur = np.zeros(3)
        else:
            A = _logu(rng, 0.1, 2.0)
            ang = rng.uniform(0, 2 * np.pi)
            cur = cur + A * np.array([np.cos(ang), np.sin(ang),
                                      prm["vf"] * rng.uniform(-1, 1)])
        prev = j
    bias[prev:] = cur
    sv, sp_ = prm["sig_v"], prm["sig_p"]
    Vm = Vj + rng.normal(0.0, 1.0, (Mj, 3)) * np.array([sv, sv, sv * prm["vf"]])
    Pm = Pj + bias + rng.normal(0.0, 1.0, (Mj, 3)) * np.array(
        [sp_, sp_, prm["sig_pz"]])
    # the grid: latest available epoch, last good epoch
    jl = np.searchsorted(tj + prm["tau_g"], t + 1e-12, side="right") - 1
    jc = np.maximum(jl, 0)
    lastgood = np.maximum.accumulate(np.where(ok, np.arange(Mj), -1))
    fresh = (jl >= 0) & ok[jc]
    g = np.where(jl >= 0, lastgood[jc], -1)
    vel, pos = np.empty((N, 3)), np.empty((N, 3))
    vel[fresh], pos[fresh] = Vm[jc[fresh]], Pm[jc[fresh]]
    nog = ~fresh & (g < 0)                     # no fix yet (start-up)
    vel[nog], pos[nog] = Vm[0], Pm[0]
    br = ~fresh & (g >= 0)
    if br.any():
        gb = g[br]
        if prm["mode"] < 0.5:                  # hold the last good fix
            vel[br], pos[br] = Vm[gb], Pm[gb]
        else:                                  # INS dead reckoning
            A_ = W["att"]
            Rm = attitude_R(A_["psi"], A_["beta"], A_["phi"])
            wm, fm = W["imu"]["w_m"], W["imu"]["f_m"]
            ae = mv(Rm, fm) - GRAV * E_Z
            dr = (M["ra_hat"] - M["ri_hat"]) if M["comp"] else np.zeros(3)
            Lv = mv(Rm, np.cross(wm, dr))
            Lp = mv(Rm, dr)
            C = cumtrapz0(ae, dt)
            D = cumtrapz0(C, dt)
            tg, te = tj[gb], t[br] - prm["tau_g"]
            dT = (te - tg)[:, None]
            ic = lambda A, tq: interp_cols(tq, t, A)          # noqa: E731
            Cg, Ce, Lg = ic(C, tg), ic(C, te), ic(Lv, tg)
            vel[br] = Vm[gb] + Ce - Cg + ic(Lv, te) - Lg
            pos[br] = Pm[gb] + Vm[gb] * dT + ic(D, te) - ic(D, tg) \
                - Cg * dT + ic(Lp, te) - ic(Lp, tg) - Lg * dT
    W["gnss"] = dict(vel=vel, pos=pos, q=fresh.astype(float), tj=tj, ok=ok,
                     a_ep=a_ep, bias=bias)


def st_heading(prm, W, rng):
    """O3 part 2: a dual-antenna heading has no fix while GNSS is lost; the
    logger bridges it with the gyro (heading rate from the measured rates
    and attitude, integrated from the last good sample)."""
    if prm is None or prm["psi_src"] < 0.5:
        return
    K, A = W["K"], W["att"]
    q = W["gnss"]["q"]
    if q.all():
        return
    psid = euler_rates(A["phi"], A["beta"], W["imu"]["w_m"])[2]
    Psi = cumtrapz0(psid, K["dt"])
    lg = np.maximum.accumulate(np.where(q > 0.5, np.arange(K["N"]), -1))
    br = (q < 0.5) & (lg >= 0)
    psi = A["psi"].copy()
    psi[br] = psi[lg[br]] + Psi[br] - Psi[lg[br]]
    A["psi"] = psi


def st_convert(prm, W, rng):
    """O5 part 2 (the logger; runs also when O5 is absent, with arms 0):
    v_cg = v_ant - R_m (w_m x r_a_hat), p_cg = p_ant - R_m r_a_hat when the
    lever arm is compensated (calibrated arm = truth + error), else the
    antenna motion with only the nominal offset r_a_hat removed; body-level
    speeds by the MEASURED heading (a heading error b turns into a sway
    error ~ -u b); rates from the gyro with the measured attitude."""
    K, M, A, I, Gn = W["K"], W["mount"], W["att"], W["imu"], W["gnss"]
    Rm = attitude_R(A["psi"], A["beta"], A["phi"])
    if M["comp"]:
        V = Gn["vel"] - mv(Rm, np.cross(I["w_m"], M["ra_hat"]))
        P = Gn["pos"] - mv(Rm, M["ra_hat"])
    else:
        V, P = Gn["vel"].copy(), Gn["pos"] - M["ra_hat"]
    c, s = np.cos(A["psi"]), np.sin(A["psi"])
    phid, betad, psid = euler_rates(A["phi"], A["beta"], I["w_m"])
    W["conv"] = dict(x=P[:, 0], y=P[:, 1], zg=P[:, 2], zdg=V[:, 2],
                     u=c * V[:, 0] + s * V[:, 1], v=-s * V[:, 0] + c * V[:, 1],
                     r=psid, thd=-W["ctx"].sp * betad, roll_rate=phid)


def st_heave(prm, W, rng):
    """O3 part 3: heave and heave rate from the GNSS height of the cg, or
    the INS heave of the IMU point (moved to the cg with the calibrated
    arm when compensated) through the second-order high-pass f_hp, plus
    noise sigma_z (the running rise is lost with the DC)."""
    K = W["K"]
    if prm is None:
        W["heave"] = dict(z=K["z"].copy(), zd=K["zd"].copy(), src="truth")
        return
    if prm["heave_src"] < 0.5:
        C = W["conv"]
        W["heave"] = dict(z=C["zg"].copy(), zd=C["zdg"].copy(), src="gnss")
        return
    M, A = W["mount"], W["att"]
    h = M["P_imu"][:, 2].copy()
    if M["comp"]:
        Rm = attitude_R(A["psi"], A["beta"], A["phi"])
        h = h - mv(Rm, M["ri_hat"])[:, 2]
    hp, hpd = hpf2(h, np.gradient(h, K["t"]), prm["f_hp"], K["dt"])
    W["heave"] = dict(z=hp + rng.normal(0.0, prm["sig_z"], K["N"]), zd=hpd,
                      src="ins")


def st_output(prm, W, rng):
    """O4 and the resampling: per signal group (GNSS-derived, gyro-derived,
    estimates, plain sensors) a causal first-order low-pass of bandwidth
    f_F (lpf_exact) and a delay tau; one logger time-stamp jitter per
    output sample (U[-j, j]); commands logged with the offset tau_c (ZOH);
    the plain section-11 sensors y = (1 + s) x + n (+ bias for the wind
    angle), the anemometer at the antenna mast (its lever arm w x r_a)."""
    K, C, A, H = W["K"], W["conv"], W["att"], W["heave"]
    t, dt, N = K["t"], K["dt"], K["N"]
    k = W["k_out"]
    tk = t[k]
    on = prm is not None
    ex = K["extra"]
    grp = dict(gnss=dict(x=C["x"], y=C["y"], u=C["u"], v=C["v"]),
               imu=dict(r=C["r"], thd=C["thd"]),
               est=dict(th=A["th"], psi=np.unwrap(A["psi"])),
               plain={})
    grp["gnss" if H["src"] == "gnss" else "est"].update(z=H["z"], zd=H["zd"])
    if "roll" in ex:
        grp["est"]["roll"] = A["phi"]
    if "roll_rate" in ex:
        grp["imu"]["roll_rate"] = C["roll_rate"]
    nrm = (lambda n, sd: rng.normal(0.0, sd, n)) if on else \
        (lambda n, sd: np.zeros(n))
    if "rpm" in ex:
        grp["plain"]["rpm"] = (1 + (prm["rpm_s"] if on else 0.0)) * ex["rpm"] \
            + nrm(N, prm["rpm_n"] if on else 0.0)
    if "wind_speed_rel" in ex and "wind_angle_rel" in ex:
        vr = ex["wind_speed_rel"][:, None] * np.stack(
            [np.cos(ex["wind_angle_rel"]), np.sin(ex["wind_angle_rel"])], 1)
        vr = vr + W["mount"]["wr_a"][:, :2]
        ws = np.hypot(vr[:, 0], vr[:, 1])
        wa = np.unwrap(np.arctan2(vr[:, 1], vr[:, 0]))
        if on:
            ws = (1 + prm["ws_s"]) * ws + nrm(N, prm["ws_n"])
            wa = wa + prm["wa_b"] + nrm(N, prm["wa_n"])
        grp["plain"].update(wind_speed_rel=ws, wind_angle_rel=wa)
    for nm in ("stw_u", "stw_v"):
        if nm in ex:
            grp["plain"][nm] = (1 + (prm["stw_s"] if on else 0.0)) * ex[nm] \
                + nrm(N, prm["stw_n"] if on else 0.0)
    jit = rng.uniform(-prm["jit"], prm["jit"], len(k)) if on and \
        prm["jit"] > 0 else np.zeros(len(k))
    out = {}
    tau_g = {}
    for gi, g in enumerate(O4_GROUPS):
        if not grp[g]:
            continue
        names = list(grp[g])
        S = np.stack([np.asarray(grp[g][n], float) for n in names], 1)
        if on:
            S = lpf_exact(S, 1.0 / (2 * np.pi * prm["f_F"][gi]), dt)
            tau_g[g] = prm["tau"][gi]
        else:
            tau_g[g] = 0.0
        Y = interp_cols(tk + jit - tau_g[g], t, S) if on else S[k]
        out.update({n: Y[:, i] for i, n in enumerate(names)})
    # angles back into the truth's wrapping convention
    out["psi"] = K["psi"][k] + wrap(out["psi"] - K["psi"][k])
    if "wind_angle_rel" in out:
        out["wind_angle_rel"] = wrap(out["wind_angle_rel"])
    sr = np.stack([out[n] for n in ("x", "y", "u", "z", "zd", "th", "thd",
                                    "psi", "r", "v")], 1)
    extra = {n: out[n] for n in CB.OBS_EXTRA if n in out}
    if "gnss_q" in ex or "O1" in W["items"]:
        q = W["gnss"]["q"] if "O1" in W["items"] else ex["gnss_q"]
        tq = tk + jit - tau_g.get("gnss", 0.0)
        iq = np.clip(np.searchsorted(t, tq + 1e-9, side="right") - 1, 0, N - 1)
        extra["gnss_q"] = np.asarray(q, float)[iq]
    cmd = None
    if K["cmd"] is not None:
        tc = tk - (prm["tau_c"] if on else 0.0)
        ic = np.clip(np.searchsorted(t, tc + 1e-9, side="right") - 1, 0, N - 1)
        cmd = K["cmd"][ic]
    W["out"] = dict(t=tk.copy(), k=k.copy(), sr=sr, cmd=cmd, extra=extra)


STAGE_FN = {("O5", "mount"): st_mount, ("O2", "imu"): st_imu,
            ("O3", "attitude"): st_attitude, ("O1", "gnss"): st_gnss,
            ("O3", "heading"): st_heading, ("O5", "convert"): st_convert,
            ("O3", "heave"): st_heave, ("O4", "output"): st_output}


def out_indices(K, traj):
    if traj.get("k_out") is not None:
        return np.asarray(traj["k_out"], int)
    sub = int(traj.get("sub") or max(1, round(DT_CTRL / K["dt"])))
    return np.arange(0, K["N"], sub)


def run_obs(ctx, traj, params, rngs=None, debug=False):
    """The observation pipeline with the items in `params` {code: params}
    (absent codes are ideal), noise from rngs {code: Generator}. Returns
    the observed record (module docstring); debug=True also returns the
    working dict of every stage. The input is never modified."""
    rngs = dict(rngs or {})
    for code in params:                  # one stream per item, all stages
        if code not in rngs:
            rngs[code] = np.random.default_rng(
                [SID_OBS_NOISE, CB.OBS_CODES.index(code)])
    K = truth_kinematics(ctx, traj)
    W = dict(ctx=ctx, K=K, items=set(c for c, p in params.items()
                                     if p is not None))
    W["k_out"] = out_indices(K, traj)
    for code, stage in STAGE_ORDER:
        prm = params.get(code)
        rng = rngs.get(code) or np.random.default_rng(0)   # unused if absent
        STAGE_FN[(code, stage)](prm, W, rng)
    out = dict(W["out"], obs_on=True, items=sorted(W["items"]))
    return (out, W) if debug else out


def ideal_obs(ctx, traj):
    """The observation model OFF (D10a): the true values at the output
    steps, the commands and extras unchanged."""
    K = truth_kinematics(ctx, traj)
    k = out_indices(K, traj)
    return dict(t=K["t"][k].copy(), k=k, sr=K["sr"][k].copy(),
                cmd=None if K["cmd"] is None else K["cmd"][k].copy(),
                extra={n: a[k].copy() for n, a in K["extra"].items()},
                obs_on=False, items=[])


# ------------------------------------------------------------ the items
def register_obs(cls):
    """Class decorator: one instance per observation code in
    cat_base.OBS_REGISTRY (the catalogue REGISTRY holds forces only)."""
    if cls.code not in CB.OBS_CODES:
        raise ValueError(f"register_obs: unknown code {cls.code!r}")
    CB.OBS_REGISTRY[cls.code] = cls()
    return cls


class ObsCat(CB.ObsItem):
    """An observation item: draw(rng, ctx) -> params (always on when the
    model is on), apply(params, traj, rng, ctx) = run_obs with this item
    alone (every other item ideal)."""

    code = ""

    def p_on(self, ctx):
        return P_ON_OBS[self.code]

    def draw(self, rng, ctx):
        if rng.random() >= self.p_on(ctx):
            return None
        return self.draw_on(rng, ctx)

    def draw_on(self, rng, ctx):
        raise NotImplementedError

    def apply(self, params, traj, rng, ctx=None):
        return run_obs(ctx or CB.CatCtx(), traj, {self.code: params},
                       {self.code: rng})


@register_obs
class O1(ObsCat):
    """GNSS (draft table O1; u-blox ZED-F9P data sheet: velocity 0.05 m/s,
    7-25 Hz, 4 g dynamics). sig_v ~ LogU[0.02, 0.2] m/s, vertical x vf ~
    U[1.5, 3]; rate in {5, 10, 20} Hz; tau_g ~ U[0.02, 0.25] s; good / bad
    mean dwell LogU[60, 3000] s / LogU[0.2, 10] s; p_hd ~ U[0, 0.5]; RTK
    jump rate LogU[1e-4, 1e-2] 1/s, amplitude LogU[0.1, 2] m per event.
    Added [assumption]: dropout bridge hold / INS w.p. 0.5 each ("保持或由
    惯导外推"); horizontal position noise LogU[0.01, 0.5] m (RTK to
    standalone); vertical position noise = the draft's O3 sigma_z ~
    LogU[0.02, 0.5] m ("GNSS 高度 + 噪声"); the bias b_g is the RTK
    position-bias process, no velocity bias (Doppler)."""

    code = "O1"

    def draw_on(self, rng, ctx):
        u = rng.uniform
        return dict(sig_v=_logu(rng, 0.02, 0.2), vf=u(1.5, 3.0),
                    rate=float(rng.choice([5.0, 10.0, 20.0])),
                    t_off=u(0.0, 1.0), tau_g=u(0.02, 0.25),
                    T_good=_logu(rng, 60.0, 3000.0),
                    T_bad=_logu(rng, 0.2, 10.0), p_hd=u(0.0, 0.5),
                    rtk_rate=_logu(rng, 1e-4, 1e-2),
                    sig_p=_logu(rng, 0.01, 0.5), sig_pz=_logu(rng, 0.02, 0.5),
                    mode=float(rng.random() < 0.5))


@register_obs
class O2(ObsCat):
    """IMU (draft table O2; Bosch BMI088 data sheet; kalibr IMU noise model,
    x 10 for low-cost parts). Gyro noise LogU[0.003, 0.05] deg/s/sqrt(Hz);
    bias U[-1, 1] deg/s x U[0.01, 1]; g-sensitivity U[0, 0.1] deg/s/g
    (times a random matrix of entries U[-1, 1]); accelerometer noise
    LogU[60, 400] ug/sqrt(Hz); offset U[-20, 20] mg; range {6, 12, 24} g;
    scale U[-0.01, 0.01] per axis (gyro and accelerometer); misalignment
    U[0, 1] deg about a random axis (one mounting rotation for both);
    vibration LogU[0.01, 1] g at rated rpm. Added [assumption]: bias random
    walks gyro LogU[2e-5, 2e-3] rad/s/sqrt(s), accelerometer LogU[3e-4,
    3e-2] m/s^2/sqrt(s) (kalibr's example x 0.1-10); 1-3 engine orders of
    VIB_ORDERS sharing the amplitude (Dirichlet), mostly vertical
    directions; rated engine speed U[100, 135] rev/s (6000-8100 rpm);
    anti-alias corner LogU[5, 200] Hz; hull ringing f_h ~ U[20, 67] Hz
    (draft 3.4), zeta_h ~ LogU[0.01, 0.05], kick gain U[0, 1] per m/s^2 of
    specific-force jump."""

    code = "O2"

    def draw_on(self, rng, ctx):
        u = rng.uniform
        n = int(rng.integers(1, 4))
        orders = sorted(rng.choice(VIB_ORDERS, n, replace=False).tolist())
        A = _logu(rng, 0.01, 1.0) * GRAV
        dirs = rng.normal(0.0, 1.0, (n, 3)) * np.array([0.4, 0.6, 1.0])
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        ax = rng.normal(0.0, 1.0, 3)
        return dict(
            ng=_logu(rng, 0.003, 0.05) * DEG,
            bg0=(u(-1, 1, 3) * u(0.01, 1.0) * DEG).tolist(),
            bgw=_logu(rng, 2e-5, 2e-3),
            gs=u(0.0, 0.1) * DEG / GRAV, Gs=u(-1, 1, (3, 3)).tolist(),
            na=_logu(rng, 60.0, 400.0) * UG, ba0=(u(-20, 20, 3) * MG).tolist(),
            baw=_logu(rng, 3e-4, 3e-2),
            fs=float(rng.choice([6.0, 12.0, 24.0])) * GRAV,
            sg=u(-0.01, 0.01, 3).tolist(), sa=u(-0.01, 0.01, 3).tolist(),
            mis=u(0.0, 1.0) * DEG, mis_axis=(ax / np.linalg.norm(ax)).tolist(),
            vib_orders=orders, vib_amp=(A * rng.dirichlet(np.ones(n))).tolist(),
            vib_dirs=dirs.tolist(), vib_ph=u(0, 2 * np.pi, n).tolist(),
            f_rated=u(100.0, 135.0), f_aa=_logu(rng, 5.0, 200.0),
            f_h=u(20.0, 67.0), z_h=_logu(rng, 0.01, 0.05), k_h=u(0.0, 1.0),
            ph_h=u(0, 2 * np.pi))


@register_obs
class O3(ObsCat):
    """Attitude, heave, heading estimation (draft table O3; VectorNav AHRS
    theory page, F9P data sheet). kappa_a w.p. 0.3 U[0, 1] (no velocity
    aiding) else U[0, 0.1]; tau_a ~ LogU[1, 30] s; f_hp ~ LogU[0.02, 0.2]
    Hz; sigma_z ~ LogU[0.02, 0.5] m (INS option; the GNSS option's height
    noise is O1's sig_pz); b_psi1 ~ U[-3, 3] deg (magnetic). Added
    [assumption]: heave source GNSS / INS w.p. 0.5; heading magnetic /
    dual-antenna w.p. 0.5; pitch and roll bias U[-1, 1] deg, noise
    LogU[0.02, 0.3] deg; magnetic b_psi0 ~ U[-3, 3] deg, dual-antenna
    U[-0.5, 0.5] deg; heading noise LogU[0.1, 1] deg (F9P 0.4 deg)."""

    code = "O3"

    def draw_on(self, rng, ctx):
        u = rng.uniform
        k = u(0.0, 1.0) if rng.random() < 0.3 else u(0.0, 0.1)
        mag = rng.random() < 0.5
        return dict(kappa_a=k, tau_a=_logu(rng, 1.0, 30.0),
                    b_th=u(-1, 1) * DEG, n_th=_logu(rng, 0.02, 0.3) * DEG,
                    b_phi=u(-1, 1) * DEG, n_phi=_logu(rng, 0.02, 0.3) * DEG,
                    heave_src=float(rng.random() < 0.5),
                    f_hp=_logu(rng, 0.02, 0.2), sig_z=_logu(rng, 0.02, 0.5),
                    psi_src=float(not mag),
                    b_psi0=(u(-3, 3) if mag else u(-0.5, 0.5)) * DEG,
                    b_psi1=(u(-3, 3) if mag else 0.0) * DEG,
                    n_psi=_logu(rng, 0.1, 1.0) * DEG)


@register_obs
class O4(ObsCat):
    """Time alignment and estimator bandwidth (draft table O4, no external
    source): F bandwidth LogU[1, 20] Hz, delay U[0, 0.15] s - drawn per
    signal group O4_GROUPS [interpretation: channels of one logger need not
    share filter and latency]; time-stamp jitter j ~ U[0, 0.02] s (per
    sample U[-j, j]); command-state offset tau_c ~ U[-0.05, 0.1] s. Also
    the plain section-11 sensors the draft lists without a model
    [assumption]: rpm scale U[-0.02, 0.02], noise LogU[0.002, 0.02] of
    rated; anemometer scale U[-0.1, 0.1], noise LogU[0.1, 1] m/s, angle
    bias U[-5, 5] deg, noise LogU[1, 10] deg; speed log scale U[-0.1, 0.1],
    noise LogU[0.05, 0.5] m/s."""

    code = "O4"

    def draw_on(self, rng, ctx):
        u = rng.uniform
        ng = len(O4_GROUPS)
        return dict(f_F=np.exp(u(np.log(1.0), np.log(20.0), ng)).tolist(),
                    tau=u(0.0, 0.15, ng).tolist(), jit=u(0.0, 0.02),
                    tau_c=u(-0.05, 0.1),
                    rpm_s=u(-0.02, 0.02), rpm_n=_logu(rng, 0.002, 0.02),
                    ws_s=u(-0.1, 0.1), ws_n=_logu(rng, 0.1, 1.0),
                    wa_b=u(-5, 5) * DEG, wa_n=_logu(rng, 1.0, 10.0) * DEG,
                    stw_s=u(-0.1, 0.1), stw_n=_logu(rng, 0.05, 0.5))


@register_obs
class O5(ObsCat):
    """Mounting positions and frame conversion (draft table O5, rigid-body
    kinematics): antenna r_a x ~ U[-0.3, 0.3] L, y ~ U[-0.2, 0.2] B, z ~
    U[0.3, 2] m; IMU r_i x ~ U[-0.3, 0.3] L, z ~ U[-0.3, 0.8] m;
    calibration error N(0, s^2) per component, s ~ U[0.05, 0.2] m.
    Added [assumption]: IMU y ~ U[-0.2, 0.2] B (the draft gives none); the
    logger compensates the antenna lever arm w.p. 0.7 (otherwise it only
    removes the nominal offset; the draft's magnitudes 0.75 / 0.45 m/s are
    the uncompensated case)."""

    code = "O5"

    def draw_on(self, rng, ctx):
        u = rng.uniform
        s = u(0.05, 0.2)
        return dict(r_a=[u(-0.3, 0.3) * ctx.L, u(-0.2, 0.2) * ctx.B,
                         u(0.3, 2.0)],
                    r_i=[u(-0.3, 0.3) * ctx.L, u(-0.2, 0.2) * ctx.B,
                         u(-0.3, 0.8)],
                    s_cal=s, cal_a=rng.normal(0.0, s, 3).tolist(),
                    cal_i=rng.normal(0.0, s, 3).tolist(),
                    comp=float(rng.random() < 0.7))


# ------------------------------------------------------------ the model
class ObsModel:
    """The observation model of one draw: every O item drawn from its
    stream [seed, CAT_STREAM, attempt, SID_OBS + i]; observe() uses the
    noise streams [seed, CAT_STREAM, attempt, SID_OBS_NOISE + i, noise_seed].
    on=None means cat_base.OBS_ON_D10A (False in D10a data)."""

    def __init__(self, seed, ctx=None, attempt=0, on=None, force=None):
        self.seed, self.attempt = int(seed), int(attempt)
        self.ctx = ctx or CB.CatCtx()
        self.on = CB.OBS_ON_D10A if on is None else bool(on)
        force = force or {}
        self.params = {}
        for i, c in enumerate(CB.OBS_CODES):
            it = CB.OBS_REGISTRY.get(c)
            if it is None:
                continue
            rng = self._rng(SID_OBS + i)
            prm = dict(force[c]) if c in force else it.draw(rng, self.ctx)
            if prm is not None:
                self.params[c] = prm

    def _rng(self, sid, *more):
        return np.random.default_rng([self.seed, CB.CAT_STREAM, self.attempt,
                                      int(sid)] + [int(m) for m in more])

    def observe(self, traj, noise_seed=0, debug=False):
        if not self.on:
            o = ideal_obs(self.ctx, traj)
            return (o, None) if debug else o
        rngs = {c: self._rng(SID_OBS_NOISE + CB.OBS_CODES.index(c),
                             noise_seed) for c in self.params}
        return run_obs(self.ctx, traj, self.params, rngs, debug=debug)

    def style(self):
        return dict(obs_on=self.on, obs_params={c: {k: (np.asarray(v).tolist()
                                                        if not np.isscalar(v)
                                                        else float(v))
                                                    for k, v in p.items()}
                                                for c, p in self.params.items()})


def e0_observed(obs, model0, dtc=DT_CTRL):
    """e0_obs[k] = (y_{k+1} - model0(y_k, c_k, t_k)) / dtc on the observed
    record (draft O4): model0(sr_k (10,), cmd_k (2,) or None, t_k) returns
    the low-fidelity prediction of the five velocities (RED_VEL order).
    How model0 reads the waves on the real boat is open (draft question
    15); the caller decides."""
    sr, cmd, t = obs["sr"], obs["cmd"], obs["t"]
    out = np.zeros((max(len(sr) - 1, 0), 5))
    for k in range(len(out)):
        pred = np.asarray(model0(sr[k], None if cmd is None else cmd[k],
                                 t[k]), float)
        out[k] = (sr[k + 1, list(RED_VEL)] - pred) / dtc
    return out


# ------------------------------------------------------------ checks
def synthetic_truth(ctx, T=60.0, dt=0.04, **kw):
    """An analytic smooth manoeuvre (exact accelerations), for the
    construction checks: u = u0 + a_u t + A_u sin(w_u t), v, r, heave,
    pitch and roll sinusoids; keyword overrides."""
    P = dict(u0=15.0, a_u=0.0, A_u=1.0, w_u=0.3, A_v=0.4, w_v=0.5, r0=0.05,
             A_r=0.15, w_r=0.4, A_z=0.1, w_z=2.0, A_th=0.02, w_th=1.7,
             A_phi=0.05, w_phi=1.1, psi0=0.3, thr=0.5)
    P.update(kw)
    t = np.arange(int(round(T / dt)) + 1) * dt
    S, Cs = np.sin, np.cos
    u = P["u0"] + P["a_u"] * t + P["A_u"] * S(P["w_u"] * t)
    du = P["a_u"] + P["A_u"] * P["w_u"] * Cs(P["w_u"] * t)
    v, dv = P["A_v"] * S(P["w_v"] * t), P["A_v"] * P["w_v"] * Cs(P["w_v"] * t)
    r = P["r0"] + P["A_r"] * S(P["w_r"] * t)
    dr = P["A_r"] * P["w_r"] * Cs(P["w_r"] * t)
    psi = P["psi0"] + P["r0"] * t + P["A_r"] / P["w_r"] * (1 - Cs(P["w_r"]
                                                                  * t))
    z = ctx.z0 + P["A_z"] * S(P["w_z"] * t)
    zd = P["A_z"] * P["w_z"] * Cs(P["w_z"] * t)
    zdd = -P["A_z"] * P["w_z"] ** 2 * S(P["w_z"] * t)
    th = ctx.th0 + P["A_th"] * S(P["w_th"] * t)
    thd = P["A_th"] * P["w_th"] * Cs(P["w_th"] * t)
    thdd = -P["A_th"] * P["w_th"] ** 2 * S(P["w_th"] * t)
    phi = P["A_phi"] * S(P["w_phi"] * t)
    phid = P["A_phi"] * P["w_phi"] * Cs(P["w_phi"] * t)
    Vx, Vy = u * Cs(psi) - v * S(psi), u * S(psi) + v * Cs(psi)
    x, y = cumtrapz0(Vx, dt), cumtrapz0(Vy, dt)
    sr = np.stack([x, y, u, z, zd, th, thd, psi, r, v], 1)
    N = len(t)
    thr = np.full(N, P["thr"] * ctx.t_max)
    return dict(t=t, sr=sr, cmd=np.stack([thr, np.zeros(N)], 1),
                acc=np.stack([du, dv, dr, zdd, thdd], 1),
                extra=dict(roll=phi, roll_rate=phid,
                           rpm=np.full(N, np.sqrt(P["thr"])),
                           wind_speed_rel=np.hypot(u, v),
                           wind_angle_rel=np.arctan2(v, u), stw_u=u.copy(),
                           stw_v=v.copy(), gnss_q=np.ones(N)))


def quiet(code, ctx, **over):
    """The item's params with every random error switched off (the pure
    structure), then `over` applied."""
    z = dict(
        O1=dict(sig_v=0.0, vf=2.0, rate=10.0, t_off=0.0, tau_g=0.0,
                T_good=1e12, T_bad=1.0, p_hd=0.0, rtk_rate=0.0, sig_p=0.0,
                sig_pz=0.0, mode=0.0),
        O2=dict(ng=0.0, bg0=[0.0] * 3, bgw=0.0, gs=0.0, Gs=np.eye(3).tolist(),
                na=0.0, ba0=[0.0] * 3, baw=0.0, fs=24 * GRAV, sg=[0.0] * 3,
                sa=[0.0] * 3, mis=0.0, mis_axis=[0.0, 0.0, 1.0],
                vib_orders=[1.0], vib_amp=[0.0], vib_dirs=[[0.0, 0.0, 1.0]],
                vib_ph=[0.0], f_rated=120.0, f_aa=50.0, f_h=40.0, z_h=0.02,
                k_h=0.0, ph_h=0.0),
        O3=dict(kappa_a=0.0, tau_a=5.0, b_th=0.0, n_th=0.0, b_phi=0.0,
                n_phi=0.0, heave_src=0.0, f_hp=0.1, sig_z=0.0, psi_src=0.0,
                b_psi0=0.0, b_psi1=0.0, n_psi=0.0),
        O4=dict(f_F=[1e9] * 4, tau=[0.0] * 4, jit=0.0, tau_c=0.0, rpm_s=0.0,
                rpm_n=0.0, ws_s=0.0, ws_n=0.0, wa_b=0.0, wa_n=0.0, stw_s=0.0,
                stw_n=0.0),
        O5=dict(r_a=[0.0] * 3, r_i=[0.0] * 3, s_cal=0.0, cal_a=[0.0] * 3,
                cal_i=[0.0] * 3, comp=1.0))[code]
    z.update(over)
    return z


def _obs(ctx, traj, params, seed=0, debug=True):
    rngs = {c: np.random.default_rng([seed, i]) for i, c in
            enumerate(CB.OBS_CODES)}
    return run_obs(ctx, traj, params, rngs, debug=debug)


@CB.check("O1")
def o1_hold_and_latency(item, ctx):
    """With a linear surge ramp the held GNSS speed reveals its epoch: every
    grid value is the truth of an epoch in [t - tau_g - 1/f, t - tau_g];
    the number of distinct fixes = rate x duration."""
    a, u0 = 0.7, 10.0
    tr = synthetic_truth(ctx, T=30.0, u0=u0, a_u=a, A_u=0.0, A_v=0.0, r0=0.0,
                         A_r=0.0, psi0=0.0, A_phi=0.0, A_th=0.0, A_z=0.0)
    f, tg = 10.0, 0.1
    _, W = _obs(ctx, tr, dict(O1=quiet("O1", ctx, rate=f, tau_g=tg,
                                       t_off=0.37)))
    t = W["K"]["t"]
    te = (W["gnss"]["vel"][:, 0] - u0) / a
    m = t > t[0] + tg + 1.0 / f + 1e-9
    lo = te[m] - (t[m] - tg - 1.0 / f)
    hi = (t[m] - tg) - te[m]
    nd = len(np.unique(np.round(W["gnss"]["vel"][m, 0], 9)))
    want = (t[m][-1] - t[m][0]) * f
    ok = lo.min() > -1e-9 and hi.min() > -1e-9 and abs(nd - want) <= 2
    return ok, (f"epoch age in [{hi.min():.3f}, {1 / f - lo.min():.3f}] s "
                f"after tau_g; {nd} distinct fixes (rate x T = {want:.0f})")


@CB.check("O1")
def o1_noise_dropout_e0(item, ctx):
    """Velocity noise sd = sig_v (vertical x vf) at fresh fixes; the bad-
    state share = T_bad / (T_good + T_bad); loss of lock exactly at the
    epochs above 4 g with p_hd = 1; e0 noise sd = sqrt(2) sig_v / dt_ctrl
    (draft section 4 magnitude note) with 20 Hz fixes."""
    tr = synthetic_truth(ctx, T=4000.0, A_u=0.0, A_v=0.0, r0=0.0, A_r=0.0,
                         A_phi=0.0, A_th=0.0, A_z=0.0)
    p = quiet("O1", ctx, sig_v=0.1, vf=2.0, rate=20.0, T_good=10.0,
              T_bad=2.5)
    _, W = _obs(ctx, tr, dict(O1=p), seed=1)
    Gn = W["gnss"]
    sel = Gn["q"] > 0.5
    err = Gn["vel"][sel] - W["mount"]["V_ant"][sel]
    # one sample per epoch: the grid repeats held values
    _, first = np.unique(np.round(Gn["vel"][sel, 0], 12), return_index=True)
    sd = err[first].std(0)
    bad = 1.0 - Gn["ok"].mean()
    msgs = [f"sd {np.round(sd, 3).tolist()} (0.1, 0.1, 0.2)",
            f"bad share {bad:.3f} (0.200)"]
    ok = np.all(np.abs(sd / np.array([0.1, 0.1, 0.2]) - 1) < 0.05) and \
        abs(bad - 0.2) < 0.05
    tr2 = synthetic_truth(ctx, T=20.0, A_u=0.0, A_v=0.0, r0=0.0, A_r=0.0,
                          A_phi=0.0, A_th=0.0, A_z=0.5, w_z=10.0)
    _, W2 = _obs(ctx, tr2, dict(O1=quiet("O1", ctx, p_hd=1.0, rate=20.0)))
    hi = W2["gnss"]["a_ep"] > 4 * GRAV
    ok_hd = hi.any() and np.array_equal(~W2["gnss"]["ok"], hi)
    msgs.append(f"lock lost at {hi.sum()} epochs above 4 g: {ok_hd}")
    tr3 = synthetic_truth(ctx, T=400.0, A_u=0.0, A_v=0.0, r0=0.0, A_r=0.0,
                          A_phi=0.0, A_th=0.0, A_z=0.0)
    o3 = _obs(ctx, tr3, dict(O1=quiet("O1", ctx, sig_v=0.05, rate=20.0)),
              seed=2)[0]
    e = np.diff(o3["sr"][:, 2] - tr3["sr"][o3["k"], 2]) / DT_CTRL
    want = np.sqrt(2) * 0.05 / DT_CTRL
    ok_e = abs(e.std() / want - 1) < 0.08
    msgs.append(f"e0 noise sd {e.std():.3f} m/s^2 (sqrt2 sig/dt {want:.3f})")
    return ok and ok_hd and ok_e, "; ".join(msgs)


@CB.check("O1")
def o1_ins_bridge(item, ctx):
    """Dropouts bridged by INS dead reckoning (ideal IMU) follow the true
    speed; holding the last fix does not."""
    tr = synthetic_truth(ctx, T=240.0, A_u=3.0, w_u=0.4, A_r=0.25)
    e = {}
    for mode in (0.0, 1.0):
        p = quiet("O1", ctx, T_good=8.0, T_bad=3.0, mode=mode, rate=10.0)
        _, W = _obs(ctx, tr, dict(O1=p), seed=3)
        st = W["gnss"]["q"] < 0.5
        d = W["gnss"]["vel"][st] - W["mount"]["V_ant"][st]
        e[mode] = float(np.sqrt((d ** 2).sum(1).mean()))
    ok = e[1.0] < 0.1 * e[0.0] and st.any()
    return ok, (f"speed error in dropouts rms {e[1.0]:.4f} m/s with INS, "
                f"{e[0.0]:.3f} m/s holding")


@CB.check("O2")
def o2_level_rest(item, ctx):
    """At rest and level an error-free IMU reads (0, 0, g) and zero rates;
    the misalignment alone keeps |f| = g and tilts it by at most mis."""
    tr = synthetic_truth(ctx, T=4.0, u0=0.0, A_u=0.0, A_v=0.0, r0=0.0,
                         A_r=0.0, A_z=0.0, A_th=0.0, A_phi=0.0, psi0=0.0)
    tr["sr"][:, 5] = 0.0
    _, W = _obs(ctx, tr, dict(O2=quiet("O2", ctx)))
    d0 = max(np.abs(W["imu"]["f_m"] - GRAV * E_Z).max(),
             np.abs(W["imu"]["w_m"]).max())
    mis = 0.7 * DEG
    _, W = _obs(ctx, tr, dict(O2=quiet("O2", ctx, mis=mis,
                                       mis_axis=[0.3, -0.8, 0.2])))
    f = W["imu"]["f_m"]
    dn = np.abs(np.linalg.norm(f, axis=1) - GRAV).max()
    tilt = np.arccos(np.clip(f[:, 2] / GRAV, -1, 1)).max()
    ok = d0 < 1e-12 and dn < 1e-9 and tilt <= mis + 1e-12 and tilt > 0
    return ok, (f"ideal error {d0:.1e}; misaligned |f| - g {dn:.1e}, tilt "
                f"{np.degrees(tilt):.3f} deg <= 0.7")


@CB.check("O2")
def o2_noise_density(item, ctx):
    """White noise sd = density / sqrt(dt) for both sensors."""
    tr = synthetic_truth(ctx, T=800.0, u0=0.0, A_u=0.0, A_v=0.0, r0=0.0,
                         A_r=0.0, A_z=0.0, A_th=0.0, A_phi=0.0)
    na, ng = 200 * UG, 0.01 * DEG
    _, W = _obs(ctx, tr, dict(O2=quiet("O2", ctx, na=na, ng=ng)), seed=4)
    dt = W["K"]["dt"]
    ra = (W["imu"]["f_m"] - W["mount"]["f_b"]).std(0) / (na / np.sqrt(dt))
    rg = (W["imu"]["w_m"] - W["K"]["w"]).std(0) / (ng / np.sqrt(dt))
    ok = np.all(np.abs(ra - 1) < 0.03) and np.all(np.abs(rg - 1) < 0.03)
    return ok, (f"sd / (density / sqrt dt): accel {np.round(ra, 3).tolist()},"
                f" gyro {np.round(rg, 3).tolist()}")


@CB.check("O2")
def o2_range_and_vibration(item, ctx):
    """The accelerometer clips at its range (a 5.8 g heave motion into a 6 g
    part: vertical specific force up to 6.8 g); engine vibration is bounded by
    sum A_k |H(f_k)| rpm^2 and grows with rpm."""
    tr = synthetic_truth(ctx, T=10.0, A_z=0.7, w_z=9.0)
    _, W = _obs(ctx, tr, dict(O2=quiet("O2", ctx, fs=6 * GRAV)))
    fm = W["imu"]["f_m"]
    clip_ok = np.abs(fm).max() <= 6 * GRAV + 1e-9 and \
        np.isclose(np.abs(fm[:, 2]).max(), 6 * GRAV)
    p = quiet("O2", ctx, vib_orders=[1.0, 1.5], vib_amp=[3.0, 2.0],
              vib_dirs=[[0, 0, 1.0], [0, 1.0, 0]], vib_ph=[0.1, 0.2],
              f_aa=150.0)
    worst, amp = 0.0, []
    for rpm in (0.4, 1.0):
        tr = synthetic_truth(ctx, T=20.0, thr=rpm ** 2)
        _, W = _obs(ctx, tr, dict(O2=p))
        a = W["imu"]["a_vib"]
        H = 1 / np.sqrt(1 + (np.array([1.0, 1.5]) * 120 * rpm / 150.0) ** 4)
        bound = (np.array([3.0, 2.0]) * H).sum() * rpm ** 2
        worst = max(worst, float(np.linalg.norm(a, axis=1).max() / bound))
        amp.append(float(np.abs(a).max()))
    ok = clip_ok and worst <= 1 + 1e-9 and amp[1] > amp[0] > 0
    return ok, (f"clipped at 6 g: {clip_ok}; vibration / bound max "
                f"{worst:.3f}; peak {amp[0]:.2f} -> {amp[1]:.2f} m/s^2 "
                "at rpm 0.4 -> 1")


@CB.check("O3")
def o3_acceleration_tilt(item, ctx):
    """A sustained surge acceleration a makes the pitch estimate bow-up by
    kappa_a a / g after the filter settles; a steady turn (a_y = r u) makes
    the roll estimate kappa_a r u / g."""
    kap, tau = 0.6, 2.0
    p = quiet("O3", ctx, kappa_a=kap, tau_a=tau)
    tr = synthetic_truth(ctx, T=40.0, u0=8.0, a_u=1.5, A_u=0.0, A_v=0.0,
                         r0=0.0, A_r=0.0, A_z=0.0, A_th=0.0, A_phi=0.0)
    _, W = _obs(ctx, tr, dict(O3=p))
    m = W["K"]["t"] > 12 * tau
    e_bow = ctx.sp * (W["att"]["th"] - W["K"]["th"])[m]
    tr2 = synthetic_truth(ctx, T=40.0, u0=15.0, A_u=0.0, A_v=0.0, r0=0.2,
                          A_r=0.0, A_z=0.0, A_th=0.0, A_phi=0.0)
    _, W2 = _obs(ctx, tr2, dict(O3=p))
    e_phi = (W2["att"]["phi"] - W2["K"]["phi"])[m]
    w1, w2 = kap * 1.5 / GRAV, kap * 0.2 * 15.0 / GRAV
    ok = np.abs(e_bow / w1 - 1).max() < 1e-3 and \
        np.abs(e_phi / w2 - 1).max() < 1e-3
    return ok, (f"bow-up error {np.degrees(e_bow.mean()):.3f} deg (kappa a/g"
                f" {np.degrees(w1):.3f}); roll error "
                f"{np.degrees(e_phi.mean()):.3f} deg ({np.degrees(w2):.3f})")


@CB.check("O3")
def o3_ins_heave_highpass(item, ctx):
    """INS heave: the running rise (DC) is removed; the gain at f / f_hp =
    0.3, 1, 10 equals the Butterworth |H| = x^2 / sqrt(1 + x^4)."""
    f_hp = 0.2
    p = quiet("O3", ctx, heave_src=1.0, f_hp=f_hp)
    res = []
    for x in (0.3, 1.0, 10.0):
        f = x * f_hp
        tr = synthetic_truth(ctx, T=max(40.0 / f, 60.0), A_z=0.2,
                             w_z=2 * np.pi * f, A_th=0.0, A_phi=0.0)
        _, W = _obs(ctx, tr, dict(O3=p))
        t, z = W["K"]["t"], W["heave"]["z"]
        m = t > t[-1] / 2
        ph = 2 * np.pi * f * t[m]
        c = np.linalg.lstsq(np.stack([np.sin(ph), np.cos(ph),
                                      np.ones(m.sum())], 1), z[m],
                            rcond=None)[0]
        res.append((np.hypot(c[0], c[1]) / 0.2 / (x * x / np.sqrt(1 + x ** 4)),
                    abs(c[2])))
    ok = all(abs(g - 1) < 0.03 and dc < 2e-3 for g, dc in res)
    return ok, ("gain / Butterworth " + ", ".join(f"{g:.3f}" for g, _ in res)
                + f"; |mean| <= {max(d for _, d in res):.1e} m (rise "
                f"{ctx.z0:.2f} m removed)")


@CB.check("O3")
def o3_heading_sources(item, ctx):
    """Magnetic heading error = b0 + b1 c exactly (c throttle fraction);
    dual-antenna heading bridged through GNSS dropouts by the gyro follows
    the true heading."""
    tr = synthetic_truth(ctx, T=60.0, thr=0.36)
    b0, b1 = 2.0 * DEG, -1.5 * DEG
    _, W = _obs(ctx, tr, dict(O3=quiet("O3", ctx, b_psi0=b0, b_psi1=b1)))
    d = np.abs(W["att"]["psi"] - W["K"]["psi"] - (b0 + b1 * 0.36)).max()
    _, W = _obs(ctx, tr, dict(O3=quiet("O3", ctx, psi_src=1.0),
                              O1=quiet("O1", ctx, T_good=6.0, T_bad=2.0)),
                seed=5)
    lost = W["gnss"]["q"] < 0.5
    e = np.abs(wrap(W["att"]["psi"] - W["K"]["psi"])).max()
    ok = d < 1e-12 and lost.any() and e < 1e-3
    return ok, (f"magnetic error - (b0 + b1 c) {d:.1e}; dual-antenna max "
                f"error {np.degrees(e):.4f} deg over {lost.sum()} lost "
                "samples")


@CB.check("O4")
def o4_delay_filter_jitter(item, ctx):
    """On a surge ramp (slope a): a pure delay tau gives -a tau exactly; the
    first-order filter gives the continuous-time lag -a / (2 pi f_F) after
    settling; jitter j keeps |error| <= a j; a command step appears tau_c
    late."""
    a = 0.8
    tr = synthetic_truth(ctx, T=40.0, u0=8.0, a_u=a, A_u=0.0, A_v=0.0,
                         r0=0.0, A_r=0.0, A_z=0.0, A_th=0.0, A_phi=0.0,
                         psi0=0.0)
    tr["cmd"][250:, 0] = 0.9 * ctx.t_max           # step at t = 10 s
    sel = lambda o: (o["sr"][:, 2] - tr["sr"][o["k"], 2])[o["t"] > 5.0]  # noqa
    tau, fF, j, tc = 0.11, 2.0, 0.02, 0.07
    e1 = sel(_obs(ctx, tr, dict(O4=quiet("O4", ctx, tau=[tau] * 4)))[0])
    e2 = sel(_obs(ctx, tr, dict(O4=quiet("O4", ctx, f_F=[fF] * 4)))[0])
    e3 = sel(_obs(ctx, tr, dict(O4=quiet("O4", ctx, jit=j)))[0])
    o4 = _obs(ctx, tr, dict(O4=quiet("O4", ctx, tau_c=tc)))[0]
    t_step = o4["t"][np.argmax(o4["cmd"][:, 0] > 0.7 * ctx.t_max)]
    want_step = o4["t"][np.argmax(o4["t"] >= 10.0 + tc - 1e-9)]
    d1 = np.abs(e1 + a * tau).max()
    d2 = np.abs(e2 + a / (2 * np.pi * fF)).max()
    ok = d1 < 1e-9 and d2 < 1e-9 and np.abs(e3).max() <= a * j + 1e-12 \
        and np.abs(e3).max() > 0 and abs(t_step - want_step) < 1e-9
    return ok, (f"delay error {d1:.1e}, filter lag error {d2:.1e}, jitter "
                f"max {np.abs(e3).max():.4f} <= {a * j:.3f} m/s, command "
                f"step logged at {t_step:.2f} s")


@CB.check("O5")
def o5_lever_kinematics(item, ctx):
    """The antenna velocity v + R (w x r) and the IMU specific force
    R_l^T (a + g) + w' x r + w x (w x r) against numerical derivatives of
    the rotated lever R(t) r on a fine grid; the level-frame acceleration
    (du - r v, dv + r u) against the derivative of the earth velocity."""
    tr = synthetic_truth(ctx, T=6.0, dt=0.002, A_th=0.1, A_phi=0.2,
                         w_th=2.5, w_phi=2.0, A_r=0.4)
    K = truth_kinematics(ctx, tr)
    r = np.array([1.1, -0.4, 1.3])
    t = K["t"]
    Rr = mv(K["R"], r)
    num_v = np.gradient(Rr, t, axis=0)
    ana_v = mv(K["R"], np.cross(K["w"], r))
    num_a = mtv(K["R"], np.gradient(num_v, t, axis=0))
    ana_a = np.cross(K["wd"], r) + np.cross(K["w"], np.cross(K["w"], r))
    Vd = np.gradient(K["V"], t, axis=0)
    c, s = np.cos(K["psi"]), np.sin(K["psi"])
    num_l = np.stack([c * Vd[:, 0] + s * Vd[:, 1], -s * Vd[:, 0]
                      + c * Vd[:, 1], Vd[:, 2]], 1)
    m = slice(10, -10)
    ev = np.abs(num_v - ana_v)[m].max() / np.abs(ana_v).max()
    ea = np.abs(num_a - ana_a)[m].max() / np.abs(ana_a).max()
    el = np.abs(num_l - K["a_lvl"])[m].max() / np.abs(K["a_lvl"]).max()
    ok = max(ev, ea, el) < 1e-3
    return ok, (f"relative errors: lever velocity {ev:.1e}, lever "
                f"acceleration {ea:.1e}, level acceleration {el:.1e}")


@CB.check("O5")
def o5_compensation_and_heading(item, ctx):
    """Exact arms + ideal sensors: the logger recovers the cg speeds; no
    compensation: the error is the antenna's own w x r_a (draft: 0.5 rad/s
    x 1.5 m = 0.75 m/s); a heading bias b rotates the speeds by -b (sway
    error ~ -u b, 0.35 m/s for 1 deg at 20 m/s)."""
    tr = synthetic_truth(ctx, T=30.0, A_th=0.05, A_phi=0.1, w_th=2.0)
    ra = [0.8, 0.3, 1.5]
    o = _obs(ctx, tr, dict(O5=quiet("O5", ctx, r_a=ra, r_i=[0.5, 0, 0.2])))[0]
    tk = tr["sr"][o["k"]]
    e_c = max(np.abs(o["sr"][:, [0, 1, 2, 9, 8, 6]]
                     - tk[:, [0, 1, 2, 9, 8, 6]]).max(), 0.0)
    o, W = _obs(ctx, tr, dict(O5=quiet("O5", ctx, r_a=ra, comp=0.0)))
    K = W["K"]
    k = o["k"]
    dV = mv(K["R"], np.cross(K["w"], ra))[k]
    c, s = np.cos(K["psi"][k]), np.sin(K["psi"][k])
    want = np.stack([c * dV[:, 0] + s * dV[:, 1], -s * dV[:, 0]
                     + c * dV[:, 1]], 1)
    e_u = np.abs(o["sr"][:, [2, 9]] - tk[:, [2, 9]] - want).max()
    b = 1.0 * DEG
    tr2 = synthetic_truth(ctx, T=10.0, u0=20.0, A_u=0.0, A_v=0.0)
    o = _obs(ctx, tr2, dict(O3=quiet("O3", ctx, b_psi0=b)))[0]
    t2 = tr2["sr"][o["k"]]
    wv = -np.sin(b) * t2[:, 2] + (np.cos(b) - 1) * t2[:, 9]
    e_h = np.abs(o["sr"][:, 9] - t2[:, 9] - wv).max()
    ok = e_c < 1e-9 and e_u < 1e-9 and e_h < 1e-9
    return ok, (f"compensated error {e_c:.1e}; uncompensated = w x r_a to "
                f"{e_u:.1e}; heading bias 1 deg at 20 m/s -> sway "
                f"{np.abs(wv).mean():.3f} m/s (to {e_h:.1e}); "
                f"0.5 rad/s x 1.5 m = {0.5 * 1.5:.2f} m/s")
