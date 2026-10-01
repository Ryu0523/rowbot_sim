#!/usr/bin/env python3
"""
SG-RL (Zarrouki et al., "Accelerating Reinforcement Learning via MPC
Solver-Gradient Guidance for Weights-varying MPC", arXiv 2609.01061), on
this task.

As published: a PPO policy outputs the NMPC's seven cost weights at every
control step (a weights-varying MPC), from the local state, a preview of
the reference and the recent tracking errors; the weights map linearly
onto a box. The NMPC's sensitivity of a smooth tracking loss to the
weights, normalised, guides PPO (ppo_sg.py). Trained in simulation (the
plant deliberately differs from the NMPC's model), 12 envs, gSDE, 1M steps
(PPO baseline 2M); tested zero-shot on unseen tracks.

Here:
  * The MPC is this project's MPPI (control/mpc.py), which has exactly
    seven weights; the action adds the thrust floor, the eighth parameter
    of the fixed-weight studies (studies/sim2real_jet.theta), whose
    gradient is left at zero.
  * The solver gradient is MPPI's own, exact for its sampled plan
    (envs.MPPISens): no KKT system, no extra rollouts.
  * Trained in the source world with its coefficients drawn from
    lofi.PRIOR (plant != MPC model, as in the paper), 6 envs, gSDE, 300k
    steps for both SG-RL and the PPO baseline; tested zero-shot on the
    target -- the analogue of the unseen tracks.
  * The PPO baseline (sg_mode=None) is also Berg et al.'s first experiment
    (PPO setting the NMPC weights each step), so it appears in both.
"""
import json
import os

from learn.repro.ppo_sg import (MetricsCallback, SensCallback, SGPPO,
                                out_dir, vec_env)

PPO_KW = dict(learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=10,
              gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
              use_sde=True, sde_sample_freq=8,
              policy_kwargs=dict(net_arch=dict(pi=[128, 128],
                                               vf=[128, 128])))


def train(name, sg_mode=None, twin=False, world="low", steps=300_000,
          n_envs=6, seed=0):
    d = out_dir(name)
    env = vec_env("MPCEnv", dict(world=world, randomize=(world == "low"),
                                 sens=sg_mode is not None, twin=twin),
                  n_envs, seed)
    model = SGPPO("MlpPolicy", env, sg_mode=sg_mode, seed=seed,
                  device="cpu", verbose=0, **PPO_KW)
    cbs = [MetricsCallback(os.path.join(d, "curve.json"))]
    if sg_mode is not None:
        cbs.append(SensCallback())
    model.learn(steps, callback=cbs)
    cbs[0].save()
    model.save(os.path.join(d, "model.zip"))
    with open(os.path.join(d, "sg_log.json"), "w") as f:
        json.dump(model.sg_log, f)
    env.close()
    return model


def load(name):
    # SGPPO, not PPO: the SCA variant saved two optimiser parameter groups
    p = os.path.join(out_dir(name), "model.zip")
    return SGPPO.load(p, device="cpu") if os.path.exists(p) else None
