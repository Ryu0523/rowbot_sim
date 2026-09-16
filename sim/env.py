#!/usr/bin/env python3
"""
M5 -- closed-loop harness and Gymnasium environment.

`Episode` runs the verified nonlinear plant under the MPC and returns the
metrics the whole project is judged on. `USVEnv` wraps the same thing in the
Gymnasium API so an RL algorithm can drive it, but note what the action space
is: the MPC WEIGHTS, not the rudder.

That choice is the reason this project can finish on the water. Tuning seven
bounded weights takes 10^5-10^6 steps, which fits a few hundred episodes of sea
trials; learning a neural policy end to end takes 10^7-10^9, which does not.
The same interface therefore serves simulation and the real vessel.

Gymnasium is optional -- the harness runs without it, so nothing in the physics
pipeline depends on an RL package being installed.

Every vessel-dependent number comes from the vessel (`hull=`, sim/config.py):
time steps, speed, rudder limit, autopilot bandwidth, slamming and deck-wetness
thresholds, and the length the cross-track error is measured in. For the 10 m
USV each is exactly its old constant.
"""
import numpy as np

from . import config, seakeeping
from .wavefield import SeaState
from control.reduced import ReducedModel
from control.mpc import (MPPIController, PreviewProvider,
                         WEIGHT_BOUNDS, WEIGHT_NAMES)

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAVE_GYM = True
except Exception:                                   # pragma: no cover
    gym, spaces, HAVE_GYM = object, None, False

G = 9.81


class Episode:
    """One closed-loop run: nonlinear plant + reduced-model MPC.

    `hull` (a Hull or a registered name) is the vessel; None = the one the
    database was computed for. dt, dt_ctrl, u_ref None = the vessel's own
    (Hull.scales: 0.05 s, 0.5 s and 4.5 m/s for the USV, Froude-scaled for
    any other). u_ref used to default to 4.0 m/s for every vessel.
    """

    def __init__(self, db, reduced, hs=3.25, tp=9.7, seed=0, t_preview=0.0,
                 preview_noise=0.0, weights=None, u_ref=None, dt=None,
                 dt_ctrl=None, theta0=np.pi, n_freq=32, n_dir=6,
                 n_samples=192, use_rudder=True, autopilot=False,
                 heading_ref=0.0, hull=None):
        hull = config.resolve(db, hull)
        sc = config.scales_for(db, hull)
        dt = sc["dt"] if dt is None else dt
        dt_ctrl = sc["dt_ctrl"] if dt_ctrl is None else dt_ctrl
        u_ref = sc["u_design"] if u_ref is None else u_ref
        self.sea = SeaState(hs, tp, theta0=theta0, n_freq=n_freq,
                            n_dir=n_dir, seed=seed)
        self.plant = config.plant_for(db, self.sea, hull, dt=dt)
        self.pv = PreviewProvider(self.sea, t_preview, preview_noise, seed)
        self.ctrl = MPPIController(reduced, self.pv, weights=weights,
                                   dt_ctrl=dt_ctrl, u_ref=u_ref, seed=seed,
                                   n_samples=n_samples,
                                   use_rudder=use_rudder)
        self.reduced = reduced
        self.dt, self.dt_ctrl = dt, dt_ctrl
        self.sub = max(int(round(dt_ctrl / dt)), 1)
        self.u_ref = u_ref
        # Short-crested seas put a yaw moment on the hull that the MPC's
        # reduced model cannot see -- it has neither a sway state nor lateral
        # wave forcing -- so letting the MPC steer makes things worse: with the
        # rudder free the heading diverged outright. Splitting the job is both
        # the fix and the way real vessels are built: MPC schedules thrust, a
        # classical autopilot with integral action holds heading against a
        # disturbance it never has to model.
        self.autopilot = autopilot
        self.heading_ref = heading_ref
        self._psi_i = 0.0

    def _steer(self, s, wn=None, zeta=1.0, lim=None):
        """Heading hold, gains placed on the IDENTIFIED Nomoto model.

        For  r' = (K delta - r)/T  and  delta = -(a psi_err + b r),

            a = wn^2 T / K ,     b = (2 zeta wn T - 1) / K

        With K = -1.50 and T = 3.35 that gives |a| ~ 0.14. Hand-picked gains of
        2-5 were more than an order of magnitude too high: the rudder chased
        every wave, its drag bled off speed, lift falls with u^2, and the loop
        lost the authority it needed -- the heading diverged outright.

        wn is deliberately well below the encounter frequency (~0.9 rad/s) so
        the autopilot holds the mean heading instead of fighting each wave.
        It is the USV's 0.30 rad/s Froude-scaled -- a frequency goes as
        1/sqrt(L), so a 0.30 rad/s autopilot on a 300 m ship would fight
        every wave -- and the integral's time constants likewise. The limit
        is the plant's own rudder, not 35 deg.

        Integration is CONDITIONAL: winding up while the rudder is already hard
        over cannot help and guarantees an overshoot when authority returns.
        """
        rs = np.sqrt(self.plant.L / 10.0)
        wn = 0.30 / rs if wn is None else wn
        lim = self.plant.rudder.max if lim is None else lim
        p = self.reduced.p
        K = p.get("k_nomoto", -1.0)
        T = max(p.get("tau_r", 3.0), 0.5 * rs)
        a = wn ** 2 * T / K
        b = (2 * zeta * wn * T - 1.0) / K
        err = _wrap(s[5] - self.heading_ref)
        raw = -(a * err + b * s[11] + (0.12 / rs) * a * self._psi_i)
        if abs(raw) < lim:
            self._psi_i = float(np.clip(self._psi_i + err * self.dt_ctrl,
                                        -6.0 * rs, 6.0 * rs))
        return float(np.clip(raw, -lim, lim))

    def run(self, t_end=200.0, trace=False):
        s = self.plant.initial_state(self.u_ref * 0.8)
        n_ctrl = int(t_end / self.dt_ctrl)
        acc, spd, yaw, cross, tr = [], [], [], [], []
        # Slam severity as a CONTINUOUS signal, not an event count. The count
        # is a rare event -- at 0.5/min over a 200 s run it is one or two per
        # seed, so its Poisson noise is around 45% and comparing two of them
        # carries about 59%. No amount of careful interpretation rescues a
        # statistic that noisy; it has to be replaced. The impact force is the
        # same physics sampled every step instead of a few times per run.
        slamf, relb = [], []
        t = 0.0
        for _ in range(n_ctrl):
            sr = self.reduced.from_plant_state(s)
            cmd = self.ctrl(sr, t)
            thrust, rudder = self.ctrl.to_actuator(cmd)
            if self.autopilot:
                rudder = self._steer(s)
            for _ in range(self.sub):
                s = self.plant.step(s, t, thrust, rudder, self.dt)
                acc.append(self.plant.last_bow_acc)
                slamf.append(self.plant.last_slam_force)
                relb.append(self.plant.last_rel_bow)
                spd.append(s[6]); yaw.append(s[5])
                cross.append(s[1])
                if trace:
                    tr.append((t, s[0], s[6], self.plant.last_bow_acc / G, s[2], s[4],
                               thrust, rudder))
                t += self.dt
            if not np.all(np.isfinite(s)):
                break
        acc = np.asarray(acc)
        sf = np.asarray(slamf)
        # Analytic criteria from the continuous relative motion. The counted
        # slam rate is kept alongside as `slam_per_min` so the two can be
        # compared -- they must agree, and only one of them converges.
        # Thresholds are the BOW's: its own keel depth and freeboard (they
        # were the USV's 0.8 m draught and 0.55 m for every vessel).
        sk = seakeeping.summarise(relb, acc / G, self.dt,
                                  self.plant.draft_bow, self.plant.freeboard,
                                  self.plant.v_slam)
        out = dict(
            slam_p99=float(np.percentile(sf, 99) / 1e3) if sf.size else 0.0,
            slam_peak=float(sf.max() / 1e3) if sf.size else 0.0,
            # impulse per minute: the total momentum the water puts into the
            # hull through entry. Integrates the whole signal rather than its
            # tail, so it is the lowest-variance of the three.
            slam_impulse=float(sf.sum() * self.dt / (t / 60.0 + 1e-9) / 1e3)
            if sf.size else 0.0,
            # acc_rms / acc_p99 come from seakeeping.summarise
            slam_per_min=float(self.plant.slam_count / (t / 60.0 + 1e-9)),
            u_mean=float(np.mean(spd)),
            cross_rms=float(np.sqrt(np.mean(np.square(cross)))),
            heading_rms=float(np.sqrt(np.mean(np.square(yaw)))),
            finite=bool(np.all(np.isfinite(s))),
            t_end=float(t),
            L=float(self.plant.L),
            **sk)
        if trace:
            out["trace"] = np.asarray(tr)
        return out


# The 10 m USV's topside height above the design waterline, m. Kept for the
# Wigley studies that import it; a plant carries its own as plant.freeboard.
FREEBOARD = 0.55


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def score(m, u_ref=4.0, w=(1.0, 0.6, 1.0, 0.4)):
    """Scalar objective for weight tuning.

    Deliberately NOT the MPC's own cost: tuning a controller on its own
    objective only teaches it to agree with itself. This is the operator's
    objective -- keep the speed up, keep the motion and slamming down, stay on
    track -- and the weights are free to trade against it however they like.

    Cross-track is measured in hull lengths (m["L"], from Episode.run); it was
    divided by 10 m, the USV's length, whatever the vessel.
    """
    if not m["finite"]:
        return 1e6
    # slam_rate_ochi, NOT slam_per_min. The counted rate carries a 24% standard
    # error over 12 seeds against 5.8% for the analytic one (DEFECTS.md A22),
    # so a quarter of this objective used to be noise -- and CMA-ES will
    # happily spend generations chasing it.
    return (w[0] * (m["acc_p99"]) ** 2
            + w[1] * m["slam_rate_ochi"]
            + w[2] * ((u_ref - m["u_mean"]) / u_ref) ** 2
            + w[3] * (m["cross_rms"] / m.get("L", 10.0)) ** 2)


if HAVE_GYM:
    class USVEnv(gym.Env):
        """Action = MPC weights (normalised to [-1,1]); one step = one episode.

        Framed as a bandit over weights rather than a per-timestep control
        problem, because that is the formulation that transfers: the same
        seven numbers can be tuned on the real vessel in a few hundred runs.

        `hull`: a registered name or a Hull. t_end and u_ref None = the
        vessel's own (150 s Froude-scaled, the design speed).
        """

        metadata = {"render_modes": []}

        def __init__(self, hull="wigley10", t_end=None,
                     sea_states=((3.25, 9.7), (1.88, 8.0)), u_ref=None,
                     t_preview=0.0, n_seeds=2):
            self.hull, self.db = config.load(hull)
            sc = self.hull.scales()
            self.reduced = None
            self.t_end = 150.0 * np.sqrt(sc["lam"]) if t_end is None else t_end
            self.u_ref = sc["u_design"] if u_ref is None else u_ref
            self.sea_states, self.n_seeds = sea_states, n_seeds
            self.t_preview = t_preview
            self.action_space = spaces.Box(-1.0, 1.0, (len(WEIGHT_NAMES),),
                                           np.float32)
            self.observation_space = spaces.Box(-np.inf, np.inf, (3,),
                                                np.float32)
            self._rng = np.random.default_rng(0)

        def _ensure_model(self):
            if self.reduced is None:
                plant, _ = config.calm_plant(self.hull, db=self.db)
                self.reduced = ReducedModel.identify(plant)

        def _weights(self, action):
            a = np.clip(action, -1, 1)
            lo, hi = WEIGHT_BOUNDS[:, 0], WEIGHT_BOUNDS[:, 1]
            return lo + (a + 1) * 0.5 * (hi - lo)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._ensure_model()
            hs, tp = self.sea_states[0]
            return np.array([hs, tp, self.t_preview], np.float32), {}

        def step(self, action):
            self._ensure_model()
            w = self._weights(action)
            vals = []
            for hs, tp in self.sea_states:
                for sd in range(self.n_seeds):
                    ep = Episode(self.db, self.reduced, hs=hs, tp=tp, seed=sd,
                                 t_preview=self.t_preview, weights=w,
                                 u_ref=self.u_ref, hull=self.hull)
                    vals.append(score(ep.run(self.t_end), self.u_ref))
            r = -float(np.mean(vals))
            hs, tp = self.sea_states[0]
            obs = np.array([hs, tp, self.t_preview], np.float32)
            return obs, r, True, False, {"weights": w}
else:                                                # pragma: no cover
    USVEnv = None
