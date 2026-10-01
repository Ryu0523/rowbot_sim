#!/usr/bin/env python3
"""
Berg, Menges, Tengesdal, Rasheed, "Digital twin syncing for autonomous
surface vessels using reinforcement learning and nonlinear model predictive
control" (Sci. Rep. 2025), on this task.

As published: an ASV path-following NMPC (3-DoF Fossen model, CasADi,
N = 10 at 1 s). A PPO agent sets, at every step and absolutely (each step
maps again from fixed bounds),
  (a) the NMPC's weight matrices Q, R, W, or
  (b) the NMPC MODEL's mass and damping matrices, M and D each within
      +-25% of their original values (+-100% made the solver fail).
The reward is path-following only; there is no model-error term, so the
"syncing" of (b) is whatever model makes the controller perform. The agent
is trained on the vessel's own simulator (no randomisation), 10M steps.

Here:
  * (a) is PPO setting the MPC weights each step -- identical to SG-RL's
    PPO baseline, so it is trained once (sgrl.train(name="ppo_weights")).
  * (b) is trained with the weights (the paper says the two "form the
    action space together"): 8 MPC parameters + 8 model coefficients,
    +-25% about the identified values (envs.TWIN_KEYS: surge mass and
    drag, sway mass, damping and Coriolis term, yaw time constant and
    gain). Only the MPC's copy of the model moves; the heading autopilot
    keeps the identified one.
  * Where it is trained is the sim-to-real question. The paper trains on
    the vessel itself, which here is the target -- at 300k steps that is
    ~1250 target episodes, a budget this study treats as an oracle. The
    reproduction trains on the randomised source (as SG-RL, RMA and Jiang
    here) and is tested zero-shot on the target; `world="high"` gives the
    paper's own setting, if the time is spent.
"""
from learn.repro import sgrl


def train(world="low", steps=300_000, n_envs=6, seed=0):
    name = "berg_twin" if world == "low" else "berg_twin_target"
    return sgrl.train(name, sg_mode=None, twin=True, world=world,
                      steps=steps, n_envs=n_envs, seed=seed)
