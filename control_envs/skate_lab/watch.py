"""Watch any controller or trained policy drive the skateboard in the pygame window.

    python -m control_envs.skate_lab.watch --ctrl lqi --v 2.0
    python -m control_envs.skate_lab.watch --ctrl pi  --v 1.0 --ff
    python -m control_envs.skate_lab.watch --ctrl mppi --v 2.5
    python -m control_envs.skate_lab.watch --ctrl mppi-env --anchor 2.5     # objective-2 cost
    python -m control_envs.skate_lab.watch --ctrl phase --xi 0.6             # ride the wave face
    python -m control_envs.skate_lab.watch --ctrl phase-obs --xi 0.6         # same, phase estimated
    python -m control_envs.skate_lab.watch --ctrl ppo --variant privileged
    python -m control_envs.skate_lab.watch --ctrl ppo --variant onboard
    python -m control_envs.skate_lab.watch --ctrl bang
    add --random for randomised waves, --seconds 30 for a longer run,
    --headless to run without a window (prints the summary only).

At the end it prints mean speed, speed error, airborne fraction, impacts and crashes
for the run (metrics.py definitions).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "control_envs.skate_lab"

import gymnasium as gym
import numpy as np

from .envpath import SkateConfig
from .controllers import (LQISpeed, MPPI, PISpeed, PhaseLQR, PhaseObserver, BangBangPitch,
                          speed_cost, envelope_cost, phase_cost)
from .metrics import aggregate, run_episode
from .wrappers import DropObs, RandomWaves, SkateLabEnv


def build_env(cfg: SkateConfig, headless: bool, random_waves: bool, seed: int,
              drop_x: bool = False, drop_wave: bool = False, frame_stack: int | None = None):
    env = SkateLabEnv(render_mode=None if headless else "human", config=cfg)
    if random_waves:
        env = RandomWaves(env, seed=seed)
    if drop_x or drop_wave:
        env = DropObs(env, drop_x=drop_x, drop_wave=drop_wave)
    if frame_stack:
        env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
    return env


def make_controller(args, cfg: SkateConfig):
    c = args.ctrl
    if c == "pi":
        return PISpeed(cfg, args.v, kp=1.0, ki=0.5, feedforward=args.ff)
    if c == "lqi":
        ctrl = LQISpeed(cfg, args.v, feedforward=not args.no_ff)
        print(ctrl.describe())
        return ctrl
    if c == "mppi":
        return MPPI(cfg, speed_cost(args.v), K=128, H=36, lam=0.1, n_iter=2, replan_every=3,
                    name=f"MPPI speed v*={args.v}")
    if c == "mppi-env":
        cost = envelope_cost(2.0, 1.0, 50.0, v_anchor=args.anchor, lam_v=1.0 if args.anchor else 0.0)
        return MPPI(cfg, cost, K=128, H=36, lam=0.1, n_iter=2, replan_every=3, u_init=0.3,
                    name=f"MPPI envelope anchor={args.anchor}")
    if c in ("phase", "phase-obs"):
        obs_src = PhaseObserver(cfg) if c == "phase-obs" else None
        ctrl = PhaseLQR(cfg, args.xi, phase_source=obs_src)
        print(ctrl.describe())
        return ctrl
    if c == "mppi-phase":
        return MPPI(cfg, phase_cost(args.xi), K=128, H=36, lam=0.1, n_iter=2, replan_every=3,
                    name=f"MPPI phase xi*={args.xi}")
    if c == "bang":
        return BangBangPitch()
    if c == "ppo":
        from .rl_ppo import PPOPolicy, load
        model, venv = load(args.variant, cfg)
        return PPOPolicy(model, venv, f"PPO {args.variant}")
    raise SystemExit(f"unknown controller {c}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctrl", default="lqi",
                    choices=["pi", "lqi", "mppi", "mppi-env", "phase", "phase-obs", "mppi-phase", "bang", "ppo"])
    ap.add_argument("--v", type=float, default=2.0, help="speed set-point (m/s) for pi/lqi/mppi")
    ap.add_argument("--ff", action="store_true", help="PI: add the slope feed-forward")
    ap.add_argument("--no-ff", action="store_true", help="LQI: switch the feed-forward off")
    ap.add_argument("--anchor", type=float, default=None, help="mppi-env: speed reference from the slow layer")
    ap.add_argument("--xi", type=float, default=0.6, help="phase target (rad) for phase/phase-obs/mppi-phase")
    ap.add_argument("--variant", default="privileged", choices=["privileged", "onboard"])
    ap.add_argument("--random", action="store_true", help="randomised waves at reset")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    cfg = SkateConfig()
    ppo = args.ctrl == "ppo"
    env = build_env(cfg, args.headless, args.random, args.seed,
                    drop_x=ppo, drop_wave=(ppo and args.variant == "onboard"),
                    frame_stack=4 if (ppo and args.variant == "onboard") else None)
    ctrl = make_controller(args, cfg)
    T = int(round(args.seconds / cfg.dt))
    opts = None
    if args.ctrl.startswith("phase") or args.ctrl == "mppi-phase":
        # start resting on the surface, moving with the wave, 0.8 rad away from the target
        opts = {"xi": args.xi - 0.8, "rest": True, "vx": -cfg.wave_speed}
    v_ref = args.v if args.ctrl in ("pi", "lqi", "mppi") else None
    try:
        if opts is not None:
            # run_episode resets with the seed only; apply the options through a thin shim
            orig_reset = env.reset

            def reset_with_opts(*, seed=None, options=None):
                return orig_reset(seed=seed, options=opts)
            env.reset = reset_with_opts
        ep = run_episode(env, ctrl, T=T, seed=args.seed, record=True)
        if hasattr(ctrl, "last_xi") and ep.get("series") is not None:
            xi_err = np.abs(((np.array(ep["series"]["xi"]) - args.xi + math.pi) % (2 * math.pi)) - math.pi)
            print(f"phase error: mean {xi_err[-300:].mean():.3f} rad over the last 5 s")
        summary = aggregate([ep], v_ref=v_ref, name=getattr(ctrl, "name", args.ctrl))
        keys = ["mean_speed", "speed_rmse", "airborne_frac", "impact_mean", "impact_top10",
                "crashes_per_min", "effort_rms"]
        print(f"\n{summary['name']}  ({args.seconds:.0f} s, {'randomised' if args.random else 'nominal'} wave)")
        for k in keys:
            v = summary.get(k, np.nan)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                print(f"  {k:16s} {v:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
