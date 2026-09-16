#!/usr/bin/env python3
"""
Direct-control environment: the policy commands thrust and rudder itself.

The existing `USVEnv` in sim/env.py is not this. Its action is the MPC's seven
WEIGHTS and one of its steps is a whole episode -- a bandit over controller
tuning, useful for M6 and useless for learning control. Nothing in this project
has ever exposed thrust and rudder as an action, which is also why the rudder
channel has never been tested.

That matters now for a specific reason. Every negative preview result so far
came through the thrust channel, and thrust has roughly a fifteenth of the
authority over encounter frequency that HEADING has:

    encounter period, head to following seas :  7.59 -> 13.42 s  (1.77x)
    encounter period, thrust +-20%           :  7.49 ->  7.12 s  (1.05x)

So the open question is whether a policy that steers can do what a policy that
only throttles could not. This environment is built to answer that, and it is
built so the answer cannot be faked: the preview the policy receives is a
parameter, so the same policy can be trained with 0 s and with 30 s of
look-ahead and the difference measured.

DELIBERATE CHOICES

  * No gymnasium dependency. The API matches (reset/step/close, spaces as
    simple named tuples) but it runs on a bare install; a gymnasium subclass is
    exposed only if the package is present. The project's whole argument is
    that its methods must run inside a sea-trial budget, and that argument is
    weaker if the code needs a stack to start.

  * The reward is DENSE and per-step, built from the same quantities as the
    operator objective, because that objective is defined on run-level
    statistics (p99, rms) that a per-step reward cannot see. Anything episodic
    -- slam counts especially -- is deliberately absent: DEFECTS.md A22 is the
    record of what happens when a rare event carries a conclusion.

  * The observation carries the vessel's own state plus wave elevation sampled
    on a grid AHEAD along the projected track. What the policy gets to see is
    the experiment, so it is configurable rather than baked in.

Run `python -m sim.rl_env` to benchmark the step rate and print an honest
estimate of what a training run would cost.
"""
import time
import numpy as np

from hydro import bem
from sim import config
from sim.vessel import NonlinearVessel, SIGN_PITCH  # noqa: F401
from sim.wavefield import SeaState

G = 9.81

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAVE_GYM = True
except ImportError:
    gym, spaces = None, None
    HAVE_GYM = False


class Box:
    """Stand-in for gymnasium.spaces.Box so the env runs without the package."""

    def __init__(self, low, high, shape):
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)
        self.shape = shape

    def sample(self, rng=np.random):
        return rng.uniform(self.low, self.high).astype(np.float32)


class USVControlEnv:
    """Thrust and rudder, commanded directly, every `dt_ctrl` seconds.

    action[0] -> thrust, mapped from [-1, 1] to [0, T_max]
    action[1] -> rudder, mapped from [-1, 1] to [-max, +max]
    """

    def __init__(self, db=None, db_path=None, hull=None,
                 hs=3.25, tp=9.7, theta0=np.pi, n_freq=24, n_dir=1,
                 dt=None, dt_ctrl=None, t_end=None, u_ref=None,
                 t_preview=None, n_preview=6, n_station=3,
                 seeds=tuple(range(32)), heading_ref=0.0,
                 # w[3], the cross-track weight, is 1.5 rather than 0.3. The
                 # Huber form that made training possible also made drifting
                 # cheap, and the first trained policy took the offer: it beat
                 # the MPC on relative motion at matched speed (-8.1%,
                 # resolved) while letting cross-track go from 0.9 m to 8.3 m.
                 # That is not PPO being clever, it is the reward disagreeing
                 # with the operator objective, which prices cross-track at
                 # 0.4*(y/10)^2. Raising the weight puts the two back in line
                 # so the comparison measures control rather than bookkeeping.
                 weights=(1.0, 0.5, 1.0, 1.5, 2.0, 0.05)):
        # `hull`: a registered name or a Hull (sim/config.py); None = the
        # vessel `db` was computed for, or the 10 m USV if there is no db
        if db is None:
            if db_path is not None:
                db = bem.load(db_path)
            else:
                hull, db = config.load("wigley10" if hull is None else hull)
        self.db = db
        # resolved once: a Hull caches its mesh and station table, and a
        # plant is built every episode
        self.hull = config.resolve(db, hull)
        sc = config.scales_for(db, self.hull)
        self._lam, self._rs = sc["lam"], float(np.sqrt(sc["lam"]))
        # None = the vessel's own; for the 10 m USV the old 0.05 s, 0.5 s,
        # 200 s, 4.5 m/s and 8 s of preview
        dt = sc["dt"] if dt is None else dt
        dt_ctrl = sc["dt_ctrl"] if dt_ctrl is None else dt_ctrl
        t_end = 200.0 * self._rs if t_end is None else t_end
        u_ref = sc["u_design"] if u_ref is None else u_ref
        t_preview = 8.0 * self._rs if t_preview is None else t_preview
        self.hs, self.tp, self.theta0 = hs, tp, theta0
        self.n_freq, self.n_dir = n_freq, n_dir
        self.dt, self.dt_ctrl = dt, dt_ctrl
        self.sub = max(int(round(dt_ctrl / dt)), 1)
        self.t_end, self.u_ref = t_end, u_ref
        self.t_preview, self.n_preview = t_preview, n_preview
        self.n_station = n_station
        self.seeds = list(seeds)
        self.heading_ref = heading_ref
        self.w = np.asarray(weights, float)
        self._rng = np.random.default_rng(0)
        self._plant = None
        self._last_a = np.zeros(2)

        n_obs = 18 + n_station + n_preview * n_station
        self.observation_space = Box(-np.inf, np.inf, (n_obs,))
        self.action_space = Box(-1.0, 1.0, (2,))

    # ------------------------------------------------------------- helpers
    def _build(self, seed):
        sea = SeaState(self.hs, self.tp, theta0=self.theta0,
                       n_freq=self.n_freq, n_dir=self.n_dir, seed=seed)
        # A fresh vessel per reset costs a retardation-model lookup, which is
        # cached, so cycling a fixed seed set keeps resets cheap.
        self._plant = config.plant_for(self.db, sea, self.hull, dt=self.dt)
        self.sea = sea

    def _observe(self):
        """What the policy is allowed to know.

        Audited against the plant, and two entries here are corrections to
        outright defects rather than refinements: the reward penalised
        cross-track and action rate, and the observation contained NEITHER the
        cross-track error NOR the previous action. A policy cannot learn to
        reduce a quantity it is punished for and cannot see -- the gradient
        says "wrong" without saying "which way".

        The rest close gaps between what the plant simulates and what the
        controller was told:

          actual thrust / rudder  the propeller has a 1.5 s lag, a rate limit
                                  and a 0.15 s transport delay, so command and
                                  output differ substantially at a 0.5 s
                                  control step. Without these the policy is
                                  reasoning about a lagged plant as if it were
                                  instantaneous.
          roll and roll rate      an entire degree of freedom, reaching 4 deg
                                  in oblique seas, and rudder lift generates a
                                  heel moment directly.
          relative bow motion     the parent signal for both slamming and deck
                                  wetness, both of which are in the reward.
          propeller submergence   thrust collapses when it broaches; without
                                  this the policy cannot tell why its throttle
                                  stopped working.

        Everything is scaled to order one. Raw metres and radians alongside
        accelerations in g span three decades, and VecNormalize should be
        correcting a well-posed observation rather than rescuing a badly posed
        one.
        """
        s, p = self._s, self._plant
        x, y, z = s[0], s[1], s[2]
        roll, pitch, psi = s[3], s[4], s[5]
        u, v, w_, r = s[6], s[7], s[8], s[11]
        _, _, _, thr_a, rud_a = p.unpack(s)
        sub = p.prop_submergence(s[:6], self._t)
        # Froude-scaled to the 10 m USV (lam = L / 10): velocities over
        # sqrt(lam), lengths over lam, rates times sqrt(lam). The USV's own
        # numbers are unchanged, and a geometrically similar vessel at the
        # same Froude number sees the same ones -- in raw m/s and rad/s a
        # tanker model's yaw rate and a dinghy's are decades apart.
        rs, lam = self._rs, self._lam
        base = np.array([u / self.u_ref, v / rs, z / lam, w_ / rs, pitch,
                         s[10] * rs,
                         _wrap(psi - self.heading_ref), r * rs,
                         p.last_bow_acc / G,
                         y / p.L,                       # penalised, was unseen
                         self._last_a[0], self._last_a[1],   # ditto
                         thr_a / p.prop.t_max, rud_a / p.rudder.max,
                         roll, s[9] * rs,               # the missing dof
                         p.last_rel_bow / p.T,
                         sub / p.T], float)

        # wave elevation ahead, on a grid the vessel will actually cross:
        # `n_preview` times spread over the preview horizon, `n_station`
        # points across the hull at each. Beyond the horizon it is zero --
        # the honest stand-in for "not measured", same convention as the MPC.
        if self.t_preview <= 0 or self.n_preview == 0:
            pv = np.zeros(self.n_preview * self.n_station)
        else:
            dts = np.linspace(self.t_preview / self.n_preview,
                              self.t_preview, self.n_preview)
            xs, ys = [], []
            for dtp in dts:
                xa = x + u * np.cos(psi) * dtp
                ya = y + u * np.sin(psi) * dtp
                off = np.linspace(p.sec.x_stern, p.sec.x_bow, self.n_station)
                xs.append(xa + off * np.cos(psi))
                ys.append(ya + off * np.sin(psi))
            xs = np.concatenate(xs); ys = np.concatenate(ys)
            pv = self.sea.eta(xs, ys, self._t) / max(self.hs / 4.0, 1e-6)
        # The water the vessel is standing in RIGHT NOW. The preview grid
        # starts at t_preview/n_preview ahead -- 1.33 s with the defaults -- so
        # the policy could see where it was going and not where it was, while
        # every quantity it is judged on happens here.
        off = np.linspace(p.sec.x_stern, p.sec.x_bow, self.n_station)
        cp, sp = np.cos(psi), np.sin(psi)
        here = self.sea.eta(x + off * cp, y + off * sp,
                            self._t) / max(self.hs / 4.0, 1e-6)
        return np.concatenate([base, here, pv]).astype(np.float32)

    # ----------------------------------------------------------------- API
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._build(int(self._rng.choice(self.seeds)))
        self._s = self._plant.initial_state(self.u_ref * 0.8)
        self._t = 0.0
        self._last_a = np.zeros(2)
        return self._observe(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, float).ravel(), -1.0, 1.0)
        thrust = 0.5 * (a[0] + 1.0) * self._plant.prop.t_max
        rudder = a[1] * self._plant.rudder.max

        acc2 = 0.0
        slam = 0.0
        for _ in range(self.sub):
            self._s = self._plant.step(self._s, self._t, thrust, rudder,
                                       self.dt)
            self._t += self.dt
            acc2 += (self._plant.last_bow_acc / G) ** 2
            slam += max(self._plant.last_slam_force, 0.0) / 1e3
        acc2 /= self.sub
        slam /= self.sub

        s = self._s
        # Dense, per-step, and every term instantaneous. Nothing here counts a
        # rare event; the slamming term is the impact FORCE, sampled every step
        # and converging like a continuous statistic.
        #
        # EVERY TERM IS BOUNDED, and that is not tidiness. The first version had
        # an unbounded cross-track penalty, (y/L)^2 with y in metres: a policy
        # that wandered 100 m paid 30 per step where a good one paid 0.01, and
        # the slam term was a raw force in kN peaking near 42. Per-step cost
        # therefore spanned four orders of magnitude, the value targets with it,
        # and 400k steps of PPO moved nothing -- explained_variance 2e-6, action
        # std still 1.01 from its initial 1.0. The critic could not fit a target
        # that ranged over 10^4, so every advantage was noise.
        d = s[1] / self._plant.L
        # quadratic near the track, linear once far from it: keeps a useful
        # gradient at 100 m without letting one bad episode dominate the batch
        cross = d * d if abs(d) < 2.0 else 2.0 * (2.0 * abs(d) - 2.0)
        cost = (self.w[0] * acc2
                # 20 kN on the USV; x lam^3, as a force scales (Froude)
                + self.w[1] * min(slam / (20.0 * self._lam ** 3), 3.0)
                + self.w[2] * min(((self.u_ref - s[6]) / self.u_ref) ** 2, 4.0)
                + self.w[3] * cross
                + self.w[4] * (1.0 - np.cos(_wrap(s[5] - self.heading_ref)))
                + self.w[5] * float(np.sum((a - self._last_a) ** 2)))
        self._last_a = a

        finite = bool(np.all(np.isfinite(s)))
        terminated = not finite
        truncated = self._t >= self.t_end
        reward = float(-cost) if finite else -50.0
        return (self._observe(), reward, terminated, truncated,
                dict(t=self._t, u=float(s[6]), y=float(s[1]),
                     acc=float(self._plant.last_bow_acc / G)))

    def close(self):
        self._plant = None


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


if HAVE_GYM:
    class USVControlGym(gym.Env):
        """Thin gymnasium wrapper, present only if the package is installed."""
        metadata = {"render_modes": []}

        def __init__(self, **kw):
            self._e = USVControlEnv(**kw)
            n = self._e.observation_space.shape[0]
            self.observation_space = spaces.Box(-np.inf, np.inf, (n,),
                                                dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return self._e.reset(seed=seed, options=options)

        def step(self, action):
            return self._e.step(action)


def main(hull="wigley10"):
    print(f"direct-control environment ({hull}): action = (thrust, rudder)\n")
    env = USVControlEnv(hull=hull)
    obs, _ = env.reset(seed=0)
    print(f"  observation {obs.shape[0]} numbers "
          f"= 9 state + {env.n_preview}x{env.n_station} wave preview")
    print(f"  action      2  (thrust 0-{env._plant.prop.t_max:.0f} N, "
          f"rudder +-{np.degrees(env._plant.rudder.max):.0f} deg)")
    print(f"  control     every {env.dt_ctrl} s, "
          f"{int(env.t_end/env.dt_ctrl)} steps per episode")
    print(f"  gymnasium   {'available' if HAVE_GYM else 'NOT installed '
                          '(the bare class still runs)'}\n")

    rng = np.random.default_rng(0)
    n = 400
    t0 = time.perf_counter()
    for _ in range(n):
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 2))
        if term or trunc:
            env.reset()
    rate = n / (time.perf_counter() - t0)

    t0 = time.perf_counter()
    for _ in range(10):
        env.reset()
    reset_s = (time.perf_counter() - t0) / 10

    sim_speed = rate * env.dt_ctrl
    print(f"  throughput  {rate:.0f} control steps/s "
          f"= {sim_speed:.0f}x real time, single core")
    print(f"  reset cost  {reset_s*1000:.0f} ms\n")
    for budget in (1e5, 1e6, 1e7):
        h = budget / rate / 3600
        print(f"  {budget:>10.0e} steps -> {h:>6.1f} h on one core, "
              f"{h/8:>5.1f} h on eight")
    print("\n  A PPO/SAC run on continuous control of this size typically needs")
    print("  1e6 to 3e6 steps. That is hours, not weeks -- the plant is fast")
    print("  enough for RL, which was never the obstacle. What was missing is")
    print("  the environment, and it now exists.")
    return env


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2])            # python -m sim.rl_env [HULL]
