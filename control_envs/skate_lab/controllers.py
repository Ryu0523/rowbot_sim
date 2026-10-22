"""Controllers for the skateboard toy, all behind one small interface.

    ctrl.reset(env)              called at the start of every episode
    u = ctrl.act(obs, env)       scalar push in [-1, 1]
    ctrl(obs, env)               same, as a (1,) float32 array for env.step

``env`` is passed so that *privileged* controllers (MPPI, which plans on the
true wave model) can read ``env.unwrapped.state`` and ``env.unwrapped.phase``;
the feedback controllers use only the observation.

Controllers
-----------
PISpeed        PI (+ optional slope feed-forward) speed hold          -> baseline
LQISpeed       LQR with an integrator state on the linearised surge  -> 第 13/14 讲 LQR/积分控制
MPPI           sampling MPC on the batched true model                -> 第 14 讲 抽样 MPC
PhaseLQR       LQR about a riding equilibrium in the wave frame      -> 平衡点线性化与稳定性
BangBangPitch  the teammate's rule from control.py (sign of pitch)
PhaseObserver  recursive-least-squares phase estimate from pitch, vy, pitch rate (POMDP variant)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from .envpath import SkateConfig
from .model import SkateBatch, reduced_surge_model, surge_step_model, wrap

# observation indices of the raw env
X, HEIGHT, PITCH, VX, VY, OM, SLOPE, SINP, COSP = range(9)


class Controller:
    name = "controller"

    def reset(self, env=None) -> None:
        pass

    def act(self, obs, env=None) -> float:
        raise NotImplementedError

    def __call__(self, obs, env=None):
        return np.array([float(np.clip(self.act(obs, env), -1.0, 1.0))], dtype=np.float32)


# ----------------------------------------------------------------------------
# Feed-forward of the slope-gravity term
# ----------------------------------------------------------------------------
def slope_feedforward(cfg: SkateConfig, slope: float, pitch: float, v_ref: float,
                      height: float | None = None, geometry_factor: bool = True) -> float:
    """Push (in units of max_force) that cancels, quasi-statically, the three
    forces acting along x on a board riding the surface at speed v_ref:

        gravity along the slope   m g hx_eff      (hx_eff = slope * cos(k L/2):
                                                   each wheel sees its own slope,
                                                   the board averages the two)
        rolling resistance        mu m g sign(v)
        linear drag               c_d v_ref

    divided by cos(pitch) because the push acts along the deck.  The slope and
    rolling terms act only through the wheel contact, so they are switched off
    when the measured height above the surface says the board is in the air
    (centre more than 5 cm above its resting height).
    """
    k = 2 * math.pi / cfg.wave_length
    f = math.cos(k * cfg.wheelbase / 2) if geometry_factor else 1.0
    grounded = True if height is None else         height < (cfg.truck_height + cfg.wheel_radius) / max(math.cos(pitch), 0.2) + 0.05
    F = cfg.linear_drag * v_ref
    if grounded:
        F += (cfg.mass * cfg.gravity * slope * f
              + cfg.rolling_resistance * cfg.mass * cfg.gravity * np.sign(v_ref))
    return float(F / (cfg.max_force * max(math.cos(pitch), 0.2)))


# ----------------------------------------------------------------------------
# PI (+ feed-forward)
# ----------------------------------------------------------------------------
class PISpeed(Controller):
    """u = u_ff + kp e + ki * integral(e),  e = v_ref - vx, with anti-windup
    (the integral stops growing while the push is saturated in its direction)."""

    def __init__(self, cfg: SkateConfig, v_ref: float, kp: float = 1.0, ki: float = 0.5,
                 feedforward: bool = False, i_max: float = 1.0, name: str | None = None):
        self.cfg, self.v_ref, self.kp, self.ki = cfg, v_ref, kp, ki
        self.feedforward, self.i_max = feedforward, i_max
        self.name = name or ("PI+ff" if feedforward else "PI")
        self.I = 0.0

    def reset(self, env=None):
        self.I = 0.0
        if env is not None:                       # pick up the wave actually in use
            self.cfg = env.unwrapped.cfg

    def act(self, obs, env=None):
        e = self.v_ref - float(obs[VX])
        u_ff = slope_feedforward(self.cfg, float(obs[SLOPE]), float(obs[PITCH]), self.v_ref,
                                 float(obs[HEIGHT])) if self.feedforward else 0.0
        u = u_ff + self.kp * e + self.ki * self.I
        if abs(u) < 1.0 or np.sign(u) != np.sign(e):
            self.I = float(np.clip(self.I + e * self.cfg.dt, -self.i_max, self.i_max))
        return u


# ----------------------------------------------------------------------------
# LQI: LQR with an integrator state on the linearised surge model
# ----------------------------------------------------------------------------
class LQISpeed(Controller):
    """Discrete-time LQR on the surge error with an integrator state.

    Plant (from the numerically linearised one-step map, ``surge_step_model``):
        vx+ = a vx + b u + d_t       (a ~ 1 - c_d dt/m, b ~ Fmax dt/m, d_t = wave force)
    States x = [e, I],  e = v_ref - vx,  I+ = I + dt e:
        x+ = A x + B u,  A = [[a, 0], [dt, 1]],  B = [[-b], [0]]
    Cost  sum x' Q x + r u^2,  Q = diag(q_e, q_i).  Riccati -> K,  u = u_ff - K x.
    The integrator absorbs the constant part of d_t (rolling resistance, drag
    model error); the feed-forward cancels the periodic part (slope gravity).
    """

    def __init__(self, cfg: SkateConfig, v_ref: float, q_e: float = 1.0, q_i: float = 0.2,
                 r: float = 0.2, feedforward: bool = True, i_max: float = 2.0,
                 name: str | None = None):
        self.cfg, self.v_ref, self.feedforward, self.i_max = cfg, v_ref, feedforward, i_max
        self.name = name or ("LQI+ff" if feedforward else "LQI")
        a, b, S0, u0, _, _ = surge_step_model(cfg, v_ref)
        dt = cfg.dt
        self.a, self.b = a, b
        self.A = np.array([[a, 0.0], [dt, 1.0]])
        self.B = np.array([[-b], [0.0]])
        self.Q = np.diag([q_e, q_i])
        self.R = np.array([[r]])
        P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.solve(self.R + self.B.T @ P @ self.B, self.B.T @ P @ self.A)  # (1,2)
        self.eig_cl = np.linalg.eigvals(self.A - self.B @ self.K)
        self.eig_cl_cont = np.log(self.eig_cl.astype(complex)) / dt
        self.gain_margin = self._gain_margin()
        self.I = 0.0

    def _gain_margin(self):
        """Range of plant-gain scalings g (true b = g * modelled b) for which
        the closed loop stays stable; the interval containing g = 1."""
        gs = np.logspace(-2, 2, 801)
        stable = np.array([np.max(np.abs(np.linalg.eigvals(self.A - g * self.B @ self.K))) < 1.0
                           for g in gs])
        i1 = int(np.argmin(np.abs(gs - 1.0)))
        if not stable[i1]:
            return (np.nan, np.nan)
        lo = i1
        while lo > 0 and stable[lo - 1]:
            lo -= 1
        hi = i1
        while hi < len(gs) - 1 and stable[hi + 1]:
            hi += 1
        return (float(gs[lo]) if lo > 0 else 0.0, float(gs[hi]) if hi < len(gs) - 1 else np.inf)

    def reset(self, env=None):
        self.I = 0.0
        if env is not None:                       # pick up the wave actually in use
            self.cfg = env.unwrapped.cfg

    def act(self, obs, env=None):
        e = self.v_ref - float(obs[VX])
        u_ff = slope_feedforward(self.cfg, float(obs[SLOPE]), float(obs[PITCH]), self.v_ref,
                                 float(obs[HEIGHT])) if self.feedforward else 0.0
        x = np.array([e, self.I])
        u = u_ff - float((self.K @ x)[0])
        if abs(u) < 1.0 or np.sign(u) != np.sign(e):
            self.I = float(np.clip(self.I + e * self.cfg.dt, -self.i_max, self.i_max))
        return u

    def describe(self) -> str:
        return (f"{self.name} v*={self.v_ref:.2f}: a={self.a:.6f} b={self.b:.5f} "
                f"K=[{self.K[0,0]:.3f}, {self.K[0,1]:.3f}] "
                f"eig_d={np.round(self.eig_cl, 4)} eig_c(1/s)={np.round(self.eig_cl_cont, 2)} "
                f"gain-margin g in [{self.gain_margin[0]:.2f}, {self.gain_margin[1]:.2f}]")


# ----------------------------------------------------------------------------
# MPPI (sampling MPC) on the batched true model
# ----------------------------------------------------------------------------
class MPPI(Controller):
    """Model Predictive Path Integral control.

    Every ``replan_every`` control steps: copy the true state into K batched
    boards, sample K action sequences around the current mean plan (Gaussian,
    per-step std sigma), roll them out H steps with the exact physics, score
    them with ``cost_fn``, and update the mean with Gibbs weights
    w_k = exp(-(J_k - J_min)/lambda).  Inner iterations re-sample around the
    new mean with the sampled covariance (covariance adaptation) and a shrinking
    temperature/noise (annealing).  The plan is warm-started by shifting.

    Temperature: with ``lam_relative=True`` (default) the temperature is
    lam * (mean(J) - min(J)), i.e. measured in units of the cost spread of the
    current sample set, so one value of ``lam`` works for costs of any scale
    (lam = 0.1 means a sample that is 10 % of the spread worse than the best
    gets weight e^-1).  ``lam_relative=False`` uses lam as an absolute temperature.

    cost_fn(roll, batch) -> J (K,) where roll is ``SkateBatch.rollout`` output.
    """

    def __init__(self, cfg: SkateConfig, cost_fn, K: int = 128, H: int = 36, lam: float = 0.1,
                 sigma0: float = 0.5, n_iter: int = 2, anneal: float = 0.6,
                 replan_every: int = 3, sigma_min: float = 0.1, sigma_max: float = 1.0,
                 u_init: float = 0.0, seed: int = 0, lam_relative: bool = True,
                 ess_min: float | None = None, name: str = "MPPI"):
        self.cfg, self.cost_fn = cfg, cost_fn
        self.lam_relative = lam_relative
        self.ess_min = ess_min if ess_min is not None else max(4.0, K / 16)
        self.K, self.H, self.lam, self.sigma0 = K, H, lam, sigma0
        self.n_iter, self.anneal, self.replan_every = n_iter, anneal, replan_every
        self.sigma_min, self.sigma_max, self.u_init = sigma_min, sigma_max, u_init
        self.rng = np.random.default_rng(seed)
        self.name = name
        self.reset()

    def reset(self, env=None):
        self.mu = np.full(self.H, self.u_init)
        self.sigma = np.full(self.H, self.sigma0)
        self.t = 0
        self.idx = 0
        self.n_plans = 0
        self.last_ess = np.nan
        self.last_J = np.nan

    def _shift(self, n):
        self.mu = np.concatenate([self.mu[n:], np.repeat(self.mu[-1:], n)])
        self.sigma = np.concatenate([self.sigma[n:], np.full(n, self.sigma0)])

    def plan(self, env):
        uw = env.unwrapped
        b = SkateBatch.from_env(uw, self.K)
        s0, p0, c0 = b.state.copy(), b.phase.copy(), b.n_contacts.copy()
        mu, sigma = self.mu.copy(), self.sigma.copy()
        for it in range(self.n_iter):
            shrink = self.anneal ** it
            eps = self.rng.standard_normal((self.K, self.H)) * (sigma * shrink)
            eps[0] = 0.0                                   # the mean itself is one sample
            U = np.clip(mu + eps, -1.0, 1.0)
            b.set(s0, p0)
            b.n_contacts = c0.copy()
            roll = b.rollout(U)
            J = np.asarray(self.cost_fn(roll, b), float)
            J = np.where(np.isfinite(J), J, 1e12)
            lam = self.lam * shrink
            if self.lam_relative:
                lam = lam * max(float(J.mean() - J.min()), 1e-9)
            w = np.exp(-(J - J.min()) / lam)
            w /= w.sum()
            # keep the effective sample size above a floor: a temperature so low
            # that one sample takes all the weight turns the update into noise
            while 1.0 / float(np.sum(w ** 2)) < self.ess_min and lam < 1e6:
                lam *= 1.5
                w = np.exp(-(J - J.min()) / lam)
                w /= w.sum()
            mu = w @ U
            var = w @ (U - mu) ** 2
            sigma = np.clip(np.sqrt(var), self.sigma_min, self.sigma_max)
            self.last_ess = 1.0 / float(np.sum(w ** 2))
            self.last_J = float(w @ J)
        self.mu, self.sigma = mu, sigma
        self.n_plans += 1

    def act(self, obs, env=None):
        if self.idx == 0:
            if self.t > 0:
                self._shift(self.replan_every)
            self.plan(env)
        u = float(self.mu[self.idx])
        self.idx = (self.idx + 1) % self.replan_every
        self.t += 1
        return u


# --- pluggable costs -----------------------------------------------------------
def speed_cost(v_ref: float, r_u: float = 0.05, crash_pen: float = 100.0):
    """sum (vx - v*)^2 + r_u sum u^2 + crash penalty."""
    def cost(roll, b):
        vx = roll["state"][:, :, 3]
        return ((vx - v_ref) ** 2).sum(1) + r_u * (roll["u"] ** 2).sum(1) \
            + crash_pen * roll["crashed"][:, -1]
    return cost


def envelope_cost(lam_air: float, lam_land: float, crash_pen: float = 50.0, r_u: float = 0.0,
                  w_progress: float = 1.0, v_anchor: float | None = None, lam_v: float = 0.0):
    """Objective-2 cost:  -w_p * progress + lam_air * airtime(s) + lam_land * sum impacts
    + crash penalty [+ lam_v * sum (vx - v_anchor)^2 dt, an optional soft speed
    anchor from the slow layer; the horizon alone cannot see the consequences
    of speed built up now].  Same scale as wrappers.EnvelopeReward."""
    def cost(roll, b):
        dt = b.cfg.dt
        J = (-w_progress * roll["dx"].sum(1)
             + lam_air * dt * roll["airborne"].sum(1)
             + lam_land * roll["impact"].sum(1)
             + crash_pen * roll["crashed"][:, -1]
             + r_u * (roll["u"] ** 2).sum(1) * dt)
        if v_anchor is not None and lam_v > 0:
            J = J + lam_v * dt * ((roll["state"][:, :, 3] - v_anchor) ** 2).sum(1)
        return J
    return cost


def phase_cost(xi_star: float, lam_w: float = 0.2, r_u: float = 0.02, crash_pen: float = 100.0):
    """Objective-3 cost: hold wave-frame position xi* while moving with the wave."""
    def cost(roll, b):
        xi = b.k * roll["state"][:, :, 0] + roll["phase"]
        w = roll["state"][:, :, 3] + b.cfg.wave_speed
        return (wrap(xi - xi_star) ** 2).sum(1) + lam_w * (w ** 2).sum(1) \
            + r_u * (roll["u"] ** 2).sum(1) + crash_pen * roll["crashed"][:, -1]
    return cost


# ----------------------------------------------------------------------------
# Phase-LQR for surf-riding (objective 3)
# ----------------------------------------------------------------------------
class PhaseLQR(Controller):
    """LQR about the riding equilibrium xi* of the reduced wave-frame surge model
    (``model.reduced_surge_model``):

        d/dt [dxi, w] = A2 [dxi, w] + B2 (u - u*),   dxi = xi - xi*,  w = vx + c

    discretised exactly over one control step, optional integrator on dxi,
    gains from the discrete Riccati equation.  u = u* - K [dxi, w, (I)].
    ``phase_source(obs) -> xi`` lets an observer replace the measured phase.
    """

    def __init__(self, cfg: SkateConfig, xi_star: float, q_xi: float = 1.0, q_w: float = 0.3,
                 r: float = 0.05, integrator: bool = False, q_i: float = 0.2,
                 phase_source=None, dither: tuple | None = None, name: str | None = None):
        self.cfg, self.xi_star, self.integrator = cfg, xi_star, integrator
        self.phase_source = phase_source
        self.dither = dither          # (amplitude rad, frequency Hz) wobble added to the
        self.t = 0                    # reference xi*: persistent excitation for an observer
        self.name = name or ("phase-LQI" if integrator else "phase-LQR")
        red = reduced_surge_model(cfg, xi_star)
        self.red = red
        self.u_eq = red["u_eq"]
        A2, B2 = red["A2"], red["B2"].reshape(2, 1)
        dt = cfg.dt
        M = expm(np.block([[A2, B2], [np.zeros((1, 3))]]) * dt)   # exact discretisation
        Ad, Bd = M[:2, :2], M[:2, 2:3]
        if integrator:
            Ad = np.block([[Ad, np.zeros((2, 1))], [np.array([[dt, 0.0, 1.0]])]])
            Bd = np.vstack([Bd, [[0.0]]])
            Q = np.diag([q_xi, q_w, q_i])
        else:
            Q = np.diag([q_xi, q_w])
        R = np.array([[r]])
        P = solve_discrete_are(Ad, Bd, Q, R)
        self.K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        self.Ad, self.Bd = Ad, Bd
        self.eig_ol = red["eig_slow"]
        self.eig_cl = np.log(np.linalg.eigvals(Ad - Bd @ self.K).astype(complex)) / dt
        self.I = 0.0
        self.last_xi = np.nan

    def reset(self, env=None):
        self.I = 0.0
        self.t = 0
        if hasattr(self.phase_source, "reset"):
            self.phase_source.reset()

    def act(self, obs, env=None):
        if self.phase_source is not None:
            xi = float(self.phase_source(obs))
        else:
            xi = math.atan2(float(obs[SINP]), float(obs[COSP]))
        self.last_xi = xi
        xi_ref = self.xi_star
        if self.dither is not None:   # reference dither: the loop tracks a small wobble,
            xi_ref += self.dither[0] * math.sin(2 * math.pi * self.dither[1] * self.t * self.cfg.dt)
        dxi = float(wrap(xi - xi_ref))   # so the board keeps moving relative to the wave
        w = float(obs[VX]) + self.cfg.wave_speed
        x = [dxi, w, self.I] if self.integrator else [dxi, w]
        u = self.u_eq - float((self.K @ np.array(x))[0])
        if self.integrator and abs(u) < 1.0:
            self.I = float(np.clip(self.I + dxi * self.cfg.dt, -3.0, 3.0))
        self.t += 1
        return u

    def describe(self) -> str:
        return (f"{self.name} xi*={self.xi_star:+.3f} u*={self.u_eq:+.3f} K={np.round(self.K[0], 3)} "
                f"open-loop eig(1/s)={np.round(self.eig_ol, 3)} closed-loop eig(1/s)={np.round(self.eig_cl, 3)}")


# ----------------------------------------------------------------------------
# Bang-bang on pitch (the teammate's control.py rule)
# ----------------------------------------------------------------------------
class BangBangPitch(Controller):
    """control.py: nose up (pitch > 0, climbing a face) -> full push forward,
    nose down -> full push backward.  ``amplitude`` scales the two levels;
    ``bias`` adds a constant push, or ``base`` (another controller, e.g. an LQI
    speed hold) supplies the slowly varying part, so the rule becomes a
    per-wave modulation on top of a controlled mean speed (the plain rule has
    no net speed)."""

    def __init__(self, amplitude: float = 1.0, bias: float = 0.0, base: Controller | None = None,
                 name: str | None = None):
        self.amplitude, self.bias, self.base = amplitude, bias, base
        if name is None:
            name = "bang-bang" if amplitude == 1.0 else f"bang-bang x{amplitude:g}"
            if bias:
                name += f" +bias {bias:g}"
            if base is not None:
                name = f"{base.name} + {name}"
        self.name = name

    def reset(self, env=None):
        if self.base is not None:
            self.base.reset(env)

    def act(self, obs, env=None):
        u0 = self.bias if self.base is None else self.base.act(obs, env)
        return u0 + (self.amplitude if float(obs[PITCH]) > 0 else -self.amplitude)


# ----------------------------------------------------------------------------
# Phase observer: recursive least squares on the unit vector (cos xi, sin xi)
# ----------------------------------------------------------------------------
class PhaseObserver:
    """Estimate the wave phase xi from the board's own motion, knowing the
    wave (A, k, c) but not measuring it.

    Quasi-static riding relations (board follows the surface, two wheels
    a distance L apart, d = k L / 2):
        tan(pitch)                 =  a1 cos xi          a1 = 2 A sin d / L
        vy                         =  a2 w cos xi        a2 = A k cos d,  w = vx + c
        (1 + tan^2 pitch) pitch'   = -a3 w sin xi        a3 = a1 k
    All three are linear in p = (cos xi, sin xi): a recursive least-squares fit
    with forgetting.  Between measurements p is rotated by k w dt (the known
    relative motion), so it is a phase-locked loop in disguise.  When w = 0 the
    second and third rows vanish: only |cos xi| remains observable (the sign of
    sin xi is remembered from earlier motion).
    """

    def __init__(self, cfg: SkateConfig, forget: float = 1.0, P0: float = 10.0,
                 meas_var=(5e-4, 5e-4, 5e-3), q: float = 1e-3):
        """meas_var: noise variances of the three rows (open-loop residuals on
        the toy are about (5e-5, 1e-5, 2e-4); these are inflated for bounces).
        q: process noise added to the covariance every step, so the filter never
        becomes over-confident and keeps listening to the sign-carrying rows."""
        self.cfg = cfg
        self.q = q
        k = 2 * math.pi / cfg.wave_length
        d = k * cfg.wheelbase / 2
        self.k = k
        self.a1 = 2 * cfg.wave_amplitude * math.sin(d) / cfg.wheelbase
        self.a2 = cfg.wave_amplitude * k * math.cos(d)
        self.a3 = self.a1 * k
        self.forget, self.P0, self.meas_var = forget, P0, np.asarray(meas_var, float)
        self.reset()

    def reset(self):
        self.p = np.array([1e-3, 0.0])
        self.P = np.eye(2) * self.P0
        self.xi_hat = 0.0

    def __call__(self, obs) -> float:
        pitch, vx, vy, om = float(obs[PITCH]), float(obs[VX]), float(obs[VY]), float(obs[OM])
        w = vx + self.cfg.wave_speed
        dt = self.cfg.dt
        # predict: rotate the unit vector by the known phase advance, forget a little
        dlt = self.k * w * dt
        R = np.array([[math.cos(dlt), -math.sin(dlt)], [math.sin(dlt), math.cos(dlt)]])
        self.p = R @ self.p
        self.P = R @ self.P @ R.T / self.forget + self.q * np.eye(2)
        # correct with the three linear measurements
        tp = math.tan(pitch)
        z = np.array([tp, vy, (1 + tp * tp) * om])
        Hm = np.array([[self.a1, 0.0], [self.a2 * w, 0.0], [0.0, -self.a3 * w]])
        for i in range(3):
            h = Hm[i]
            S = float(h @ self.P @ h) + self.meas_var[i]
            Kg = self.P @ h / S
            self.p = self.p + Kg * (z[i] - float(h @ self.p))
            self.P = self.P - np.outer(Kg, h @ self.P)
        n = np.linalg.norm(self.p)
        if n > 1e-9:
            self.p = self.p / n
        self.xi_hat = math.atan2(self.p[1], self.p[0])
        return self.xi_hat
