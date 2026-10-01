#!/usr/bin/env python3
"""
The two ways a learned policy can act on the task of learn/repro/task.py.

MPCEnv     the policy sets the MPC's parameters at every control step:
           z in [-1, 1]^8 = the seven cost weights + the thrust floor, the
           same box as studies/sim2real_jet.theta, optionally followed by
           the MPC MODEL's mass and damping (Berg et al.). The MPC then
           plans thrust and the autopilot holds heading, exactly as every
           other study runs it. SG-RL, Berg.
DirectEnv  the policy commands thrust and nozzle itself (a single jet has
           no side thruster, so where Jiang et al. command Fx, Fy, Mz the
           action here is thrust and nozzle angle). RMA, Jiang, bi-level.

Both are Gymnasium environments for stable-baselines3 and both are built on
task.Mission, so the plant, the scoring and the evaluation are shared. The
reward is minus the operator cost (task.py) -- in DirectEnv plus a small
training-only penalty on action jumps, the smoothing term every direct-
control paper has (Jiang's p_dtau, RMA's smoothness terms).

Evaluation does not go through Gymnasium: `rollout` runs a policy function
on one fixed (world, seed, heading) and returns task.Mission.metrics, so
every method is scored on the same 16 target episodes as the fixed-weight
MPC.
"""
import numpy as np

from control.mpc import MPPIController, WEIGHT_BOUNDS
from learn.repro import task
from learn.repro.task import (A_LIM, K_A, K_Y, Y_SCALE, Mission, ctx,
                              draw_vessel)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:                                 # pragma: no cover
    gym, spaces = None, None

# Berg et al. move the NMPC model's M and D by +-25% of their original
# values (alpha = 0.25; alpha = 1 made their solver fail). The reduced
# model's counterparts: surge mass and quadratic drag, sway mass, linear and
# quadratic damping and the Coriolis coefficient, the yaw time constant and
# gain (a first-order yaw channel folds I_z and N_r into these two).
TWIN_KEYS = ("m_surge", "k_drag", "m_sway", "k_lin_sway", "k_sway",
             "m_coriolis", "tau_r", "k_nomoto_f")
TWIN_ALPHA = 0.25
N_THETA = 8
LO, HI = WEIGHT_BOUNDS[:, 0], WEIGHT_BOUNDS[:, 1]


def theta(z):
    from studies.sim2real_jet import theta as th
    return th(z)


# ------------------------------------------------------ sensitivity MPC
class MPPISens(MPPIController):
    """MPPI that also reports, at every solve, how its plan's predicted
    task loss changes with its cost weights -- SG-RL's solver gradient.

    SG-RL differentiates a smooth surrogate of the task loss through the
    NMPC's KKT system (implicit function theorem, acados sensitivities).
    MPPI has no KKT system, but its cost is linear in the weights,
    c_i = w . Phi_i, and its plan is a softmax average of the samples,
    pi_i ~ exp(-c_i / lam). The expected surrogate loss under that average,
    L = sum_i pi_i L_i, then has the exact gradient

        dL/dw_k = -(1 / lam) Cov_pi(L_i, Phi_ik)

    from the samples the solve already drew: no extra rollouts. L_i is the
    task's operator cost (task.py) on sample i's PREDICTED trajectory:
    speed lost along the track, CG acceleration above the limit, cross-
    track. The gradient is taken in the policy's action box and normalised
    to unit length, as in the paper (g / (|g| + eps)); only its direction
    is used."""

    def rollout_terms(self, s0, seq, t0, track_ref):
        K = seq.shape[0]
        s = np.repeat(s0[None, :], K, axis=0)
        phi = np.zeros((K, 7))
        lt = np.zeros(K)
        emerged = np.zeros(K, bool)
        p = self.m.p
        L = max(p["L"], 1.0)
        for h in range(self.H):
            t = t0 + h * self.dt
            psi_h = s[:, 7:8]
            ch, sh = np.cos(psi_h), np.sin(psi_h)
            xb = self.x_st[None, :, None]
            yb = self.y_off[None, None, :]
            xs = s[:, 0:1, None] + xb * ch[..., None] - yb * sh[..., None]
            ys = s[:, 1:2, None] + xb * sh[..., None] + yb * ch[..., None]
            eta = self.pv.at(xs, ys, t, h * self.dt)
            thrust = np.clip(seq[:, 0, h], self.thrust_floor, 1.0) * self.t_max
            rudder = np.clip(seq[:, 1, h], -1.0, 1.0) * self.rud_max
            zd0 = s[:, 4].copy()
            s, a_bow, rel = self.m.step(s, thrust, rudder, eta, self.x_st,
                                        self.dt)
            out = rel > p.get("draft_bow", p["draft"])
            slam = (emerged & ~out & (s[:, 4] < -p["v_slam"])).astype(float)
            emerged = out
            phi[:, 0] += (a_bow / 9.81) ** 2 * self.dt
            phi[:, 1] += slam * self.dt
            phi[:, 2] += ((self.u_ref - s[:, 2]) / self.u_ref) ** 2 * self.dt
            phi[:, 3] += (s[:, 1] / L) ** 2 * self.dt
            phi[:, 4] += (s[:, 7] - track_ref) ** 2 * self.dt
            a_cg = (s[:, 4] - zd0) / self.dt / 9.81
            ua = s[:, 2] * np.cos(s[:, 7]) - s[:, 9] * np.sin(s[:, 7])
            lt += (np.maximum(0.0, (self.u_ref - ua) / self.u_ref)
                   + K_A * np.maximum(0.0, np.abs(a_cg) / A_LIM - 1.0) ** 2
                   + K_Y * (s[:, 1] / (Y_SCALE * p["L"])) ** 2) * self.dt
        phi[:, 5] = np.sum(np.diff(seq[:, 0], axis=1) ** 2, axis=1)
        phi[:, 6] = np.sum(np.diff(seq[:, 1], axis=1) ** 2, axis=1)
        return phi, lt

    def rollout_cost(self, s0, seq, t0, track_ref):
        phi, lt = self.rollout_terms(s0, seq, t0, track_ref)
        self._phi, self._lt = phi, lt
        return phi @ self.w

    def __call__(self, s_reduced, t, track_ref=0.0):
        noise = self.rng.normal(0.0, 1.0, (self.K, 2, self.n_knots))
        noise *= self.sigma[None, :, None]
        if not self.use_rudder:
            noise[:, 1] = 0.0
        cand = self.nominal[None, :, :] + noise
        cand[:, 0] = np.clip(cand[:, 0], self.thrust_floor, 1.0)
        cand[:, 1] = np.clip(cand[:, 1], -1.0, 1.0)
        seq = self._expand(cand)
        c = self.rollout_cost(s_reduced, seq, t, track_ref)
        wts = np.exp(-(c - c.min()) / self.lam)
        pi = wts / wts.sum()
        # SG-RL's gradient, in weight space, then in the action box
        dphi = self._phi - pi @ self._phi
        dl = self._lt - pi @ self._lt
        g_w = -(pi * dl) @ dphi / self.lam
        g = np.zeros(N_THETA)
        g[:7] = g_w * 0.5 * (HI - LO)
        n = np.linalg.norm(g)
        self.last_grad = (g / (n + 1e-8)) if n > 1e-12 else np.zeros(N_THETA)
        self.last_loss = float(pi @ self._lt)
        self.nominal = (wts[:, None, None] * cand).sum(0) / wts.sum()
        if not self.use_rudder:
            self.nominal[1] = 0.0
        cmd = np.array([self.nominal[0, 0], self.nominal[1, 0]])
        self.nominal = self._shift(self.nominal)
        self.last_cmd = cmd
        return cmd


def sens_factory(red, pv, ep):
    """A task.Mission ctrl_factory: the Episode's MPC, with sensitivities."""
    c = ep.ctrl
    return MPPISens(red, pv, weights=c.w, dt_ctrl=c.dt, u_ref=c.u_ref,
                    seed=int(c.rng.integers(1 << 30)), n_samples=c.K,
                    use_rudder=c.use_rudder, thrust_floor=c.thrust_floor)


# -------------------------------------------------------------- harnesses
class MPCAgent:
    """A Mission whose MPC parameters are set by a policy each step.

    Observation (SG-RL's recipe, on this task): the vessel's own state
    (task.Mission.features) plus the last n_hist steps of the tracking
    errors -- speed, heading, cross-track and the worst CG acceleration.
    SG-RL also previews the reference (speed, curvature); here the
    reference is a constant speed on a straight line, so the preview is
    replaced by what does change ahead of the boat: the relative wave
    direction, already in the features."""

    def __init__(self, mission, twin=False, sens=False, n_hist=8):
        from control.reduced import ReducedModel
        self.m = mission
        self.twin = twin
        if twin:
            # the MPC's own copy of the model: the autopilot keeps the
            # identified one, as in Berg et al. only the NMPC's model moves
            self.m.ep.ctrl.m = ReducedModel(dict(self.m.red.p))
            self.nominal = dict(self.m.red.p)
        self.sens = sens
        self.n_hist = n_hist
        self.hist = np.zeros((n_hist, 4), np.float32)
        self.last_grad = np.zeros(self.n_act)

    @property
    def n_act(self):
        return N_THETA + (len(TWIN_KEYS) if self.twin else 0)

    @staticmethod
    def n_obs(n_hist=8):
        return task.N_FEATURES + 4 * n_hist

    def obs(self):
        h = np.clip(np.nan_to_num(self.hist.ravel(), nan=0.0), -20.0, 20.0)
        return np.concatenate([self.m.features(), h]).astype(np.float32)

    def act(self, a):
        a = np.clip(np.asarray(a, float), -1.0, 1.0)
        ctrl = self.m.ep.ctrl
        w, floor = theta(a[:N_THETA])
        ctrl.w = np.asarray(w, float)
        ctrl.thrust_floor = float(floor)
        if self.twin:
            p = ctrl.m.p
            for k, ak in zip(TWIN_KEYS, a[N_THETA:]):
                p[k] = self.nominal[k] * (1.0 + TWIN_ALPHA * ak)
        r = -self.m.advance(*self.m.mpc_command())
        if self.sens:
            g = np.zeros(self.n_act)
            g[:N_THETA] = ctrl.last_grad
            self.last_grad = g
        s = self.m.s
        ua = s[6] * np.cos(s[5]) - s[7] * np.sin(s[5])
        e = [ua / self.m.u_ref - 1.0,
             task.wrap(s[5]) / 0.2, s[1] / (Y_SCALE * self.m.L),
             self.m.last_amax / A_LIM]
        self.hist = np.roll(self.hist, -1, axis=0)
        self.hist[-1] = e
        return r


class DirectAgent:
    """A Mission driven by thrust and nozzle commands.

    Observation: the vessel's state (task.Mission.features) and the last
    action. The privileged vector `e` -- the source vessel's coefficients
    on the PRIOR box, zero for the nominal vessel and unknown (None) on the
    target -- is kept apart, for the teacher / base policy."""

    N_ACT = 2

    def __init__(self, mission, e=None):
        self.m = mission
        self.e = e
        self.a_prev = np.zeros(2)

    @staticmethod
    def n_obs():
        return task.N_FEATURES + 2

    def obs(self):
        return np.concatenate([self.m.features(),
                               self.a_prev]).astype(np.float32)

    def command(self, a):
        a = np.clip(np.asarray(a, float), -1.0, 1.0)
        return 0.5 * (a[0] + 1.0) * self.m.t_max, a[1] * self.m.rud_max

    def act(self, a, reward_w=None):
        """-> (reward, action jump). reward_w reprices the three cost parts
        (the bi-level method's simulator reward); None = the task's."""
        a = np.clip(np.asarray(a, float), -1.0, 1.0)
        r = -self.m.advance(*self.command(a))
        if reward_w is not None:
            r = -float(np.dot(reward_w, self.m.last_parts))
        jump = float(np.sum((a - self.a_prev) ** 2))
        self.a_prev = a
        return r, jump

    def lost(self):
        s = self.m.s
        return abs(s[1]) > 10.0 * self.m.L or abs(task.wrap(s[5])) > 0.5 * np.pi


# ------------------------------------------------------------ gym envs
if gym is not None:
    class _Base(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, world="low", randomize=True, prior_scale=1.0,
                     t_end=task.T_TRAIN, seed=0, vessel_fn=None):
            self.world, self.randomize = world, randomize
            self.prior_scale = prior_scale
            self.t_end = t_end
            self.rng = np.random.default_rng(seed)
            # vessel_fn(rng) -> (overrides, e): lets a caller fix or move
            # the source world (the bi-level method moves it)
            self.vessel_fn = vessel_fn
            self.params = {}

        def set_params(self, params):
            """Fixed overrides of the source world's coefficients (merged
            under any randomisation)."""
            self.params = dict(params or {})

        def _mission(self, disturb=False, **kw):
            c = ctx()
            head = int(self.rng.integers(len(task.HEADINGS)))
            seed = int(1000 + self.rng.integers(10 ** 6))
            pp = dict(self.params)
            e = task.to_unit(c["p"], pp, c["names"])
            if self.world == "low":
                if self.vessel_fn is not None:
                    over, e = self.vessel_fn(self.rng)
                    pp.update(over)
                elif self.randomize:
                    over, e = draw_vessel(self.rng, self.prior_scale)
                    pp.update(over)
            dist = (task.LateralOU(self.rng)
                    if disturb and self.world == "low" else None)
            m = Mission(self.world, seed, head, plant_params=pp or None,
                        t_end=self.t_end, disturb=dist, **kw)
            return m, e

    class MPCEnv(_Base):
        """Action: MPC parameters (8, or 16 with the twin). One step = one
        control step."""

        def __init__(self, twin=False, sens=False, n_hist=8, **kw):
            super().__init__(**kw)
            self.twin, self.sens, self.n_hist = twin, sens, n_hist
            n_act = N_THETA + (len(TWIN_KEYS) if twin else 0)
            self.action_space = spaces.Box(-1.0, 1.0, (n_act,), np.float32)
            self.observation_space = spaces.Box(
                -np.inf, np.inf, (MPCAgent.n_obs(n_hist),), np.float32)

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            m, self.e = self._mission(
                ctrl_factory=sens_factory if self.sens else None)
            self.agent = MPCAgent(m, twin=self.twin, sens=self.sens,
                                  n_hist=self.n_hist)
            return self.agent.obs(), {}

        def step(self, action):
            r = self.agent.act(action)
            m = self.agent.m
            info = {}
            if self.sens:
                info["sens"] = self.agent.last_grad.astype(np.float32)
            if not m.finite:
                return np.zeros(self.observation_space.shape, np.float32), \
                    -task.BAD_SCORE, True, False, info
            if m.done():
                info["metrics"] = m.metrics()
            return self.agent.obs(), float(r), False, m.done(), info

    class DirectEnv(_Base):
        """Action: thrust and nozzle in [-1, 1]. `priv` appends the
        privileged vector to the observation (teacher / base policy)."""

        def __init__(self, priv=False, lost_penalty=10.0, c_jump=0.02,
                     disturb=True, alive=1.0, **kw):
            super().__init__(**kw)
            self.priv, self.disturb = priv, disturb
            self.lost_penalty, self.c_jump = lost_penalty, c_jump
            # A reward that is minus a cost makes ending the episode early
            # attractive: getting lost costs lost_penalty once, finishing a
            # rough episode can cost 100+. The first RMA run learnt nothing
            # in 1M steps for exactly this reason. A bonus per step alive
            # (MuJoCo's "healthy reward") removes the incentive; it is a
            # constant per step, so among finished episodes it changes
            # nothing, and the task score (task.py) never sees it.
            self.alive = alive
            self.reward_w = None
            n_e = len(ctx()["names"]) if priv else 0
            self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
            self.observation_space = spaces.Box(
                -np.inf, np.inf, (DirectAgent.n_obs() + n_e,), np.float32)

        def _obs(self):
            o = self.agent.obs()
            if self.priv:
                o = np.concatenate([o, self.e.astype(np.float32)])
            return o

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            m, self.e = self._mission(disturb=self.disturb)
            self.agent = DirectAgent(m, self.e)
            return self._obs(), {"e": self.e}

        def set_reward_w(self, w):
            self.reward_w = None if w is None else np.asarray(w, float)

        def step(self, action):
            r, jump = self.agent.act(action, self.reward_w)
            m = self.agent.m
            info = {"e": self.e}
            if not m.finite:
                return np.zeros(self.observation_space.shape, np.float32), \
                    -task.BAD_SCORE, True, False, info
            if self.lost_penalty and self.agent.lost():
                info["metrics"] = dict(m.metrics(), lost=True)
                return self._obs(), -self.lost_penalty, True, False, info
            if m.done():
                info["metrics"] = m.metrics()
            return self._obs(), float(r - self.c_jump * jump + self.alive), \
                False, m.done(), info
else:                                                # pragma: no cover
    MPCEnv = DirectEnv = None


# ------------------------------------------------------------ evaluation
def rollout(kind, policy, world, seed, head, t_end=task.T_EVAL, **kw):
    """Run `policy` on one episode and return Mission.metrics.

    kind "mpc":    policy(obs) -> MPC parameters; kw: twin, n_hist
    kind "direct": policy(obs, agent) -> (thrust, nozzle) in [-1, 1]; the
                   agent is passed so a history policy can keep its state
    kind "fixed":  policy = z (constant MPC parameters)"""
    if kind == "fixed":
        from learn.repro.task import run_job
        return run_job(dict(z=policy, world=world, seed=seed, head=head,
                            t_end=t_end))
    m = Mission(world, seed, head, t_end=t_end)
    if kind == "mpc":
        ag = MPCAgent(m, twin=kw.get("twin", False),
                      n_hist=kw.get("n_hist", 8))
        while not m.done():
            ag.act(policy(ag.obs()))
    else:
        ag = DirectAgent(m)
        if hasattr(policy, "reset"):
            policy.reset()
        while not m.done():
            ag.act(policy(ag.obs(), ag))
    return m.metrics()
