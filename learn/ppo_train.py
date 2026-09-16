#!/usr/bin/env python3
"""
Train a PPO policy to drive the vessel, and hold seas back to test it on.

The policy commands thrust and rudder directly against the full 110-state
plant. There is no reduced model anywhere in this loop -- that is the point of
doing it this way. The MPC has to plan through a 10-state caricature because it
searches at runtime; a policy searches at training time and then just runs, so
it can be trained against the real dynamics including everything the caricature
drops: fluid memory, diffraction, roll, slamming, added resistance.

TWO THINGS ARE FIXED BEFORE TRAINING STARTS, because this project has been
burned by both.

  DISJOINT SEEDS. Training uses one set of wave realisations, evaluation uses
  another that the policy never sees. Nothing else distinguishes a policy that
  learned the physics from one that memorised a particular sea -- and the
  oracle experiment already showed how large a "gain" pure overfitting can
  manufacture (+14%, of which random search reproduced 78%).

  A MEMORY WINDOW. The plant carries 2.2-2.9 s of hydrodynamic memory that is
  NOT in the observation, because it is not measurable on a real vessel either.
  A feedforward policy seeing one frame is therefore acting on a partially
  observed system. Stacking six frames covers 3 s, which is that memory span --
  the number comes from the physics, not from a hyperparameter sweep.

Run: python -m learn.ppo_train [--steps 400000] [--hull NAME]
"""
import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

TRAIN_SEEDS = tuple(range(0, 24))
EVAL_SEEDS = tuple(range(100, 112))      # never seen during training
T_END = 120.0                            # s for the USV; x sqrt(L / 10 m)
# 6 x 0.5 s = 3 s ~ the memory span. Both the control step and the memory
# scale as sqrt(L) (Froude), so six frames cover it on any vessel.
N_STACK = 6
OUT = "ppo_usv"


def _time_scale(hull):
    """sqrt(L / 10 m). Episode lengths and the preview horizon were chosen
    for the 10 m USV and scale with the vessel's own time scale."""
    from sim.config import hull_of
    return float(np.sqrt(hull_of(hull).L / 10.0))


def make_env(seeds, t_preview=None, t_end=None, rank=0, hull="wigley10"):
    """t_preview None = the environment's own, 8 s Froude-scaled."""
    from sim.rl_env import USVControlGym
    if t_end is None:
        t_end = T_END * _time_scale(hull)

    def _f():
        from stable_baselines3.common.monitor import Monitor
        # 16x4 = 64 components, matching the viewer. The wave sum is the
        # per-step cost driver, and 24x5 = 120 made it 40% slower for no
        # change in the physics that matters here.
        e = USVControlGym(hull=hull, seeds=seeds, t_end=t_end,
                          t_preview=t_preview, n_freq=16, n_dir=4)
        e.reset(seed=1000 + rank)
        # Monitor, or SB3 logs no episode returns at all. Building a
        # SubprocVecEnv by hand skips the wrapper that `make_vec_env` adds for
        # you, and the first 400k-step run produced a training log with no
        # learning curve in it -- 32 minutes of compute that could not be
        # judged from its own output.
        return Monitor(e)
    return _f


def train(steps=400_000, n_envs=6, t_preview=None, out=None, seed=0,
          hull="wigley10"):
    # the vessel's own preview horizon (8 s for the USV) and file name
    t_preview = 8.0 * _time_scale(hull) if t_preview is None else t_preview
    out = out or (OUT if hull == "wigley10" else f"ppo_{hull}")
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import (SubprocVecEnv, VecNormalize,
                                                  VecFrameStack)

    # SubprocVecEnv, not DummyVecEnv, and the reason is measurable: importing
    # torch drops this environment from 169 to 62 steps/s in the same process,
    # because torch loads its own OpenMP runtime alongside the one numpy is
    # already using and the idle pool spins on the cores the physics needs.
    # No OMP setting recovered it (KMP_BLOCKTIME=0 got to 73). Separate
    # processes do: each worker pays the same 2.7x, but six of them in parallel
    # beat one fast one comfortably.
    torch.set_num_threads(1)     # the policy net is tiny; give the cores to physics
    venv = SubprocVecEnv([make_env(TRAIN_SEEDS, t_preview, rank=i, hull=hull)
                          for i in range(n_envs)], start_method="spawn")
    venv = VecFrameStack(venv, n_stack=N_STACK)
    # Observation scaling matters more than usual here: elevations are O(1),
    # accelerations O(0.1), rates O(0.01). Reward scaling is left alone so the
    # numbers stay comparable with the operator objective.
    # norm_reward=True. It was False, with a comment about keeping the numbers
    # comparable to the operator objective -- which was simply wrong: the
    # EVALUATION computes its own metrics, so the training reward never needed
    # to be readable. Combined with the unbounded cross-track term it left PPO
    # unable to fit a value function at all.
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, clip_reward=10.0, gamma=0.995)

    model = PPO("MlpPolicy", venv, seed=seed, verbose=1,
                n_steps=512, batch_size=512, gae_lambda=0.95, gamma=0.995,
                # ent_coef non-zero. In the run before this the action std
                # collapsed 1.01 -> 0.58 and then sat there, and the policy
                # scored 3.8x worse than the MPC on its OWN reward -- stuck,
                # not mis-priced. A small entropy bonus keeps it exploring.
                learning_rate=3e-4, ent_coef=0.003, clip_range=0.2,
                n_epochs=10, policy_kwargs=dict(net_arch=[128, 128]))
    print(f"  {n_envs} envs x {N_STACK} stacked frames, "
          f"{len(TRAIN_SEEDS)} training seas, preview {t_preview} s")
    print(f"  {steps:,} steps -- the plant runs at ~85x real time per core\n")
    model.learn(total_timesteps=steps, progress_bar=False)
    model.save(out)
    venv.save(out + "_vecnorm.pkl")
    print(f"\n  saved {out}.zip and {out}_vecnorm.pkl")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--preview", type=float, default=None,
                    help="s; default 8 s for the USV, Froude-scaled")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hull", default="wigley10",
                    help="a vessel from hydro/hulls.py")
    a = ap.parse_args()
    train(a.steps, a.envs, a.preview, a.out, hull=a.hull)


if __name__ == "__main__":
    main()
