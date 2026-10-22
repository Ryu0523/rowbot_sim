"""Gymnasium wrappers and a thin env subclass for the skateboard toy.

SkateLabEnv       SkateWavesEnv + reset options (start phase / wave-frame position,
                  start speed, start resting on the surface) - no physics change.
DropObs           drop x (non-stationary) and optionally slope + phase
                  (the "no wave measurement" case -> a POMDP for the policy).
RandomWaves       reset-time domain randomisation of amplitude, wavelength, speed
                  (SkateConfig is per instance; its fields are read at every
                  step, so setting them before reset is enough).
EnvelopeReward    objective-2 reward: progress - lam_air*airtime - lam_land*impact - crash.
make_env          one factory used by the scripts and by PPO.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .envpath import SkateConfig, SkateWavesEnv
from .model import surface_rest_state


class SkateLabEnv(SkateWavesEnv):
    """Same physics; ``reset(options=...)`` understands
        phase: float      initial wave phase (default: random, as in the env)
        xi:    float      initial wave-frame position of the board (board at x = 0)
        vx:    float      initial horizontal speed
        rest:  bool       start resting on the surface instead of dropped from above
    """

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if options:
            if "phase" in options:
                self.phase = float(options["phase"]) % (2 * math.pi)
            if "xi" in options:
                self.phase = float(options["xi"]) % (2 * math.pi)   # x = 0 so xi = phase
            if options.get("rest", False):
                y, th = surface_rest_state(self.cfg, self.phase)
                self.state[1], self.state[2] = y, th
            if "vx" in options:
                self.state[3] = float(options["vx"])
            obs, info = self._get_obs(), self._get_info()
        return obs, info


class DropObs(gym.ObservationWrapper):
    """Remove observation entries. drop_x removes x (index 0); drop_wave also
    removes slope, sin(phase), cos(phase) (indices 6, 7, 8)."""

    def __init__(self, env, drop_x: bool = True, drop_wave: bool = False):
        super().__init__(env)
        keep = list(range(9))
        if drop_x:
            keep.remove(0)
        if drop_wave:
            for i in (6, 7, 8):
                keep.remove(i)
        self.keep = np.array(keep)
        high = np.full(len(keep), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def observation(self, obs):
        return np.asarray(obs, dtype=np.float32)[self.keep]


class RandomWaves(gym.Wrapper):
    """Draw (amplitude, wavelength, speed) uniformly from the given ranges at
    every reset and write them into the env's SkateConfig before resetting.
    ``options={"wave": (A, L, c)}`` forces a condition."""

    def __init__(self, env, amp=(0.05, 0.30), length=(2.0, 5.0), speed=(0.5, 2.5), seed=None):
        super().__init__(env)
        self.amp, self.length, self.speed = amp, length, speed
        self.rng = np.random.default_rng(seed)
        self.current_wave = None
        self.forced = None

    def force(self, wave):
        """Use this (A, L, c) at every reset until force(None)."""
        self.forced = wave

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed + 12345)
        if self.forced is not None:
            A, L, c = self.forced
        elif options and "wave" in options:
            A, L, c = options["wave"]
        else:
            A = float(self.rng.uniform(*self.amp))
            L = float(self.rng.uniform(*self.length))
            c = float(self.rng.uniform(*self.speed))
        cfg = self.env.unwrapped.cfg
        cfg.wave_amplitude, cfg.wave_length, cfg.wave_speed = A, L, c
        self.current_wave = (A, L, c)
        return self.env.reset(seed=seed, options=options)


class EnvelopeReward(gym.Wrapper):
    """r_t = dx_t - lam_air * dt * airborne_t - lam_land * impact_t - crash_pen * crashed_t.
    Components are added to info ('r_progress', 'r_air', 'r_land', 'impact')."""

    def __init__(self, env, lam_air: float = 1.0, lam_land: float = 1.0, crash_pen: float = 50.0):
        super().__init__(env)
        self.lam_air, self.lam_land, self.crash_pen = lam_air, lam_land, crash_pen

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        uw = self.env.unwrapped
        self._x = uw.state[0]
        self._vy = uw.state[4]
        self._ht = uw.surface(uw.state[0])[2]
        self._air = uw.n_contacts == 0
        return obs, info

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        uw = self.env.unwrapped
        st = uw.state
        air = info["wheels_in_contact"] == 0
        impact = max(0.0, self._ht - self._vy) if (self._air and not air) else 0.0
        dx = st[0] - self._x
        r = dx - self.lam_air * uw.cfg.dt * float(air) - self.lam_land * impact \
            - (self.crash_pen if term else 0.0)
        info.update(r_progress=dx, r_air=-self.lam_air * uw.cfg.dt * float(air),
                    r_land=-self.lam_land * impact, impact=impact)
        self._x, self._vy, self._air = st[0], st[4], air
        self._ht = uw.surface(st[0])[2]
        return obs, float(r), term, trunc, info


def make_env(cfg: SkateConfig | None = None, random_waves: bool = False, drop_x: bool = False,
             drop_wave: bool = False, envelope_reward: dict | None = None,
             time_limit: int | None = None, frame_stack: int | None = None,
             record_stats: bool = False, seed: int | None = None,
             wave_ranges: dict | None = None):
    """Build the toy env with the requested wrappers (render_mode=None always).

    Order (inside -> outside): SkateLabEnv, EnvelopeReward, RandomWaves,
    TimeLimit, DropObs, FrameStackObservation, RecordEpisodeStatistics.
    """
    env = SkateLabEnv(render_mode=None, config=cfg or SkateConfig())
    if envelope_reward is not None:
        env = EnvelopeReward(env, **envelope_reward)
    if random_waves:
        env = RandomWaves(env, seed=seed, **(wave_ranges or {}))
    if time_limit:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=time_limit)
    if drop_x or drop_wave:
        env = DropObs(env, drop_x=drop_x, drop_wave=drop_wave)
    if frame_stack:
        env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
    if record_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)
    return env
