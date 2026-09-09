#!/usr/bin/env python3
"""
M5 -- sampling MPC with a wave-preview interface.

MPPI rather than a gradient solver, for two reasons that are specific to this
problem: the slamming term is a discrete event, so the cost is non-smooth, and
the preview enters as a time-varying disturbance forecast rather than as a
constraint. acados/SQP is the right choice for deployment on the vessel once
the formulation is frozen -- the cost structure here is written so it can be
ported without redesign.

Controls are parameterised by a few KNOTS across the horizon, not one value per
step. A 24-step horizon sampled directly is a 24-dimensional search that a few
hundred rollouts cannot explore; a 7-knot spline is a 7-dimensional one that
they can, and it produces the smooth coordinated slow-down/speed-up plans that
exploiting a wave group actually requires.

PREVIEW is deliberately a first-class, parameterised input:

    t_preview   how far ahead the wave field is known
    noise       measurement error on it

so the value of preview can be swept without touching the controller, which is
what M7 needs.

WEIGHTS are the object RL tunes in M6. They are named, bounded and few.
"""
import numpy as np

WEIGHT_NAMES = ["w_acc", "w_slam", "w_speed", "w_track", "w_heading",
                "w_dthrust", "w_drudder"]
WEIGHT_BOUNDS = np.array([[0.1, 20.0], [0.5, 60.0], [0.1, 20.0],
                          [0.0, 10.0], [0.0, 20.0],
                          [0.0, 5.0], [0.0, 5.0]])
DEFAULT_WEIGHTS = np.array([1.0, 12.0, 2.0, 1.0, 4.0, 0.3, 0.3])


class PreviewProvider:
    """Wave elevation ahead of the vessel, with a horizon and a noise model.

    Beyond `t_preview` the controller is given the statistical expectation of a
    zero-mean surface -- the honest stand-in for "not measured".

    Note what t_preview = 0 therefore means: the controller still sees the wave
    under the hull at the first horizon step, because a real vessel always knows
    its own current motion. It is a REACTIVE baseline, not a blind one, which is
    the harder and fairer thing for preview to beat.
    """

    def __init__(self, sea, t_preview=0.0, rel_noise=0.0, seed=0):
        self.sea = sea
        self.t_preview = t_preview
        self.rel_noise = rel_noise
        self.rng = np.random.default_rng(seed)

    def at(self, x_positions, t, t_ahead):
        if t_ahead > self.t_preview:
            return np.zeros_like(x_positions)
        e = self.sea.eta(x_positions.ravel(),
                         np.zeros(x_positions.size), t).reshape(
                             np.shape(x_positions))
        if self.rel_noise > 0:
            e = e + self.rng.normal(0.0, self.rel_noise * self.sea.hs / 4,
                                    e.shape)
        return e


class MPPIController:
    def __init__(self, model, preview, weights=None, horizon=24,
                 dt_ctrl=0.5, n_knots=7, n_samples=192, sigma=(0.25, 0.25),
                 lam=0.6, u_ref=4.0, seed=0, n_stations=5,
                 use_rudder=True):
        self.m = model
        self.pv = preview
        self.w = np.array(DEFAULT_WEIGHTS if weights is None else weights,
                          float)
        self.H, self.dt = horizon, dt_ctrl
        self.n_knots, self.K = n_knots, n_samples
        self.sigma = np.array(sigma, float)
        self.lam, self.u_ref = lam, u_ref
        self.rng = np.random.default_rng(seed)
        self.knot_t = np.linspace(0, horizon - 1, n_knots)
        self.nominal = np.zeros((2, n_knots))
        self.nominal[0] = 0.5                      # mid throttle
        self.x_st = np.linspace(-model.p["L"] / 2, model.p["L"] / 2,
                                n_stations)
        self.last_cmd = np.array([0.5, 0.0])
        # The rudder can be locked out for studies that are about thrust
        # scheduling alone. It is not cosmetic: rudder lift produces a SWAY
        # force as well as a yaw moment, and the reduced model has no sway
        # state, so the controller cannot see the drift it is causing and
        # cannot correct it. Leaving the channel free contaminates the very
        # metric a preview study is trying to measure.
        self.use_rudder = use_rudder

    def _expand(self, knots):
        grid = np.arange(self.H)
        out = np.empty((knots.shape[0], 2, self.H))
        for i in range(knots.shape[0]):
            for c in (0, 1):
                out[i, c] = np.interp(grid, self.knot_t, knots[i, c])
        return out

    def rollout_cost(self, s0, seq, t0, track_ref):
        K = seq.shape[0]
        s = np.repeat(s0[None, :], K, axis=0)
        cost = np.zeros(K)
        emerged = np.zeros(K, bool)
        p = self.m.p
        for h in range(self.H):
            t = t0 + h * self.dt
            xs = s[:, 0:1] + self.x_st[None, :]
            eta = self.pv.at(xs, t, h * self.dt)
            thrust = np.clip(seq[:, 0, h], 0.0, 1.0) * 12000.0
            rudder = np.clip(seq[:, 1, h], -1.0, 1.0) * np.radians(35.0)
            s, a_bow, rel = self.m.step(s, thrust, rudder, eta, self.x_st,
                                        self.dt)
            out = rel > p["draft"]
            slam = (emerged & ~out & (s[:, 4] < -p["v_slam"])).astype(float)
            emerged = out
            cost += (self.w[0] * (a_bow / 9.81) ** 2
                     + self.w[1] * slam
                     + self.w[2] * ((self.u_ref - s[:, 2]) / self.u_ref) ** 2
                     + self.w[3] * (s[:, 1] / max(self.m.p["L"], 1.0)) ** 2
                     + self.w[4] * (s[:, 7] - track_ref) ** 2) * self.dt
        cost += self.w[5] * np.sum(np.diff(seq[:, 0], axis=1) ** 2, axis=1)
        cost += self.w[6] * np.sum(np.diff(seq[:, 1], axis=1) ** 2, axis=1)
        return cost

    def __call__(self, s_reduced, t, track_ref=0.0):
        noise = self.rng.normal(0.0, 1.0, (self.K, 2, self.n_knots))
        noise *= self.sigma[None, :, None]
        if not self.use_rudder:
            noise[:, 1] = 0.0
        cand = self.nominal[None, :, :] + noise
        cand[:, 0] = np.clip(cand[:, 0], 0.0, 1.0)
        cand[:, 1] = np.clip(cand[:, 1], -1.0, 1.0)
        seq = self._expand(cand)
        c = self.rollout_cost(s_reduced, seq, t, track_ref)
        wts = np.exp(-(c - c.min()) / self.lam)
        self.nominal = (wts[:, None, None] * cand).sum(0) / wts.sum()
        if not self.use_rudder:
            self.nominal[1] = 0.0
        cmd = np.array([self.nominal[0, 0], self.nominal[1, 0]])
        self.nominal = self._shift(self.nominal)
        self.last_cmd = cmd
        return cmd

    def _shift(self, knots):
        """Warm start: advance the plan by ONE control step.

        The plan lives in knot space, so shifting the knot array by one index
        would advance it by (H-1)/(n_knots-1) steps -- here nearly four. The
        plan has to be expanded to the horizon, shifted there, and resampled
        back onto the knots.
        """
        grid = np.arange(self.H)
        out = np.empty_like(knots)
        for c in (0, 1):
            full = np.interp(grid, self.knot_t, knots[c])
            full = np.r_[full[1:], full[-1]]
            out[c] = np.interp(self.knot_t, grid, full)
        return out

    def to_actuator(self, cmd):
        return float(cmd[0]) * 12000.0, float(cmd[1]) * np.radians(35.0)
