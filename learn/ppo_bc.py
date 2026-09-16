#!/usr/bin/env python3
"""
Warm-start PPO by imitating the MPC, then fine-tune.

The measurement that motivates this is unambiguous. Scored on PPO's OWN dense
reward, over the same seas:

    PPO (400k steps)   -177.0
    MPC u_ref=4.5       -46.2      3.8x better

So PPO was not mis-priced, it was stuck. It had not found the optimum of the
objective it was given, and no amount of extra penalty weight fixes a policy
that already under-performs on the reward it has. What it needed was to be
shown where the good region is.

That is exactly what behaviour cloning does, and here the expert is free: the
MPC is a planner, it needs no training, and it already scores 3.8x better on
the policy's own reward. Clone it, then let PPO improve from there. If PPO can
beat the MPC starting from the MPC, that is a real result. If it cannot even
from there, the answer for this problem is that planning beats learning -- and
that conclusion is far harder than "our training run did not work".

Run: python -m learn.ppo_bc [--demos 40000] [--steps 300000] [--hull NAME]
"""
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from learn.ppo_train import TRAIN_SEEDS, N_STACK, T_END, make_env, _time_scale

OUT = "ppo_bc"


def stack_init(o, n):
    return np.tile(o, n)


def stack_push(buf, o):
    return np.concatenate([buf[len(o):], o])


def collect_demos(n_steps, t_preview=None, u_ref=None, seeds=TRAIN_SEEDS,
                  hull="wigley10"):
    """Drive the MPC and record what it saw and what it did. t_preview,
    u_ref None = the vessel's own (8 s Froude-scaled, the design speed)."""
    from sim import config
    from sim.rl_env import USVControlEnv
    from control.reduced import ReducedModel
    from control.mpc import MPPIController, PreviewProvider

    h, db = config.load(hull)
    red = ReducedModel.identify(config.calm_plant(h, db=db)[0])
    u_ref = red.p["u_design"] if u_ref is None else u_ref
    t_end = T_END * _time_scale(h)

    X, Y = [], []
    si = 0
    while len(X) < n_steps:
        seed = seeds[si % len(seeds)]
        si += 1
        env = USVControlEnv(db=db, hull=h, seeds=(seed,), t_end=t_end,
                            t_preview=t_preview, n_freq=16, n_dir=4)
        o, _ = env.reset(seed=0)
        ctrl = MPPIController(red, PreviewProvider(env.sea, env.t_preview,
                                                   0.0, 0),
                              dt_ctrl=env.dt_ctrl, u_ref=u_ref, seed=seed,
                              n_samples=192, use_rudder=True)
        buf = stack_init(o, N_STACK)
        t = 0.0
        while True:
            cmd = ctrl(red.from_plant_state(env._s), t)
            t += env.dt_ctrl
            thr, rud = ctrl.to_actuator(cmd)
            a = np.array([2 * (thr / env._plant.prop.t_max) - 1,
                          rud / env._plant.rudder.max])
            a = np.clip(a, -1, 1)
            X.append(buf.copy()); Y.append(a)
            o, _, term, trunc, _ = env.step(a)
            buf = stack_push(buf, o)
            if term or trunc:
                break
        print(f"    seed {seed}: {len(X)} / {n_steps} demo steps")
    return np.array(X[:n_steps], np.float32), np.array(Y[:n_steps], np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos", type=int, default=30000)
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--preview", type=float, default=None,
                    help="s; default 8 s for the USV, Froude-scaled")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hull", default="wigley10",
                    help="a vessel from hydro/hulls.py")
    # Capacity and exploration are the two live explanations for "matches but
    # does not exceed the MPC", and they need to be varied separately.
    ap.add_argument("--arch", default="128,128")
    ap.add_argument("--std", type=float, default=0.15)
    ap.add_argument("--ent", type=float, default=0.0)
    a = ap.parse_args()
    a.out = a.out or (OUT if a.hull == "wigley10" else f"{OUT}_{a.hull}")

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import (SubprocVecEnv, VecNormalize,
                                                  VecFrameStack)
    torch.set_num_threads(1)

    from learn.bc_only import get_demos
    X, Y = get_demos(a.demos, a.preview, hull=a.hull)
    mean, std = X.mean(0), X.std(0) + 1e-8
    Xn = np.clip((X - mean) / std, -10, 10).astype(np.float32)
    print(f"  demos {X.shape}, actions {Y.shape}\n")

    venv = SubprocVecEnv([make_env(TRAIN_SEEDS, a.preview, rank=i,
                                   hull=a.hull)
                          for i in range(a.envs)], start_method="spawn")
    venv = VecFrameStack(venv, n_stack=N_STACK)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, clip_reward=10.0, gamma=0.995)
    # Seed the running normaliser with the demonstration statistics so the
    # cloned policy sees the same scaling once PPO takes over. Without this the
    # clone is trained in one coordinate system and fine-tuned in another, and
    # the first PPO updates undo it.
    venv.obs_rms.mean = mean.astype(np.float64)
    venv.obs_rms.var = (std ** 2).astype(np.float64)
    venv.obs_rms.count = float(len(X))

    model = PPO("MlpPolicy", venv, seed=0, verbose=1,
                n_steps=512, batch_size=512, gae_lambda=0.95, gamma=0.995,
                clip_range=0.2, n_epochs=10,
                # ZERO, deliberately, and this reverses the previous run.
                # An entropy bonus is the right medicine for a policy stuck in
                # a bad basin, which is what plain PPO was. It is exactly the
                # wrong medicine when starting from a good clone: it actively
                # resists the noise shrinking, pushing the policy away from the
                # precise actions cloning just taught it. Same hyperparameter,
                # opposite effect, depending on the initialisation.
                ent_coef=a.ent,
                # And a gentler step, for the same reason: 3e-4 from a random
                # init is exploration; 3e-4 from a good policy is vandalism.
                learning_rate=1e-4,
                policy_kwargs=dict(
                    net_arch=[int(v) for v in a.arch.split(",")]))

    # ---- behaviour cloning -------------------------------------------------
    dev = model.policy.device
    Xt = torch.as_tensor(Xn, device=dev)
    Yt = torch.as_tensor(Y, device=dev)
    opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    n, bs = len(Xt), 256
    print(f"  cloning for {a.epochs} epochs ...")
    for ep in range(a.epochs):
        idx = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            dist = model.policy.get_distribution(Xt[j])
            loss = torch.nn.functional.mse_loss(dist.distribution.mean, Yt[j])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(j)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"    epoch {ep:>3}  action MSE {tot/n:.5f}")

    # Cloning trains the action MEAN and never touches log_std, so the policy
    # still samples with std ~ 1.0 on actions bounded in [-1, 1]. The first
    # PPO rollouts are then near-random no matter how good the mean is, and
    # they threw the clone away before it could be improved -- measured: the
    # deterministic clone matches the MPC (acc_p99 0.546 vs 0.566, cross 1.27 m
    # vs 0.83 m) while the fine-tuned result spun at 385-795 deg of heading
    # RMS. Pin the noise to something a good policy would actually use.
    with torch.no_grad():
        model.policy.log_std.fill_(float(np.log(a.std)))
    print(f"  net_arch {a.arch}, log_std pinned to std = {a.std}, "
          f"ent_coef {a.ent}")

    # ---- fine-tune ---------------------------------------------------------
    print(f"\n  fine-tuning with PPO for {a.steps:,} steps "
          f"(ent_coef {model.ent_coef})\n")
    model.learn(total_timesteps=a.steps, progress_bar=False,
                reset_num_timesteps=False)
    model.save(a.out)
    venv.save(a.out + "_vecnorm.pkl")
    print(f"\n  saved {a.out}.zip and {a.out}_vecnorm.pkl")


if __name__ == "__main__":
    main()
