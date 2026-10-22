"""PPO (stable-baselines3) on objective 2's reward, two observation variants.

reward_t = dx_t - lam_air * dt * airborne_t - lam_land * impact_t - crash_pen * crashed_t
(same weights as the MPPI envelope cost with lam_air = 2 in obj2_envelope.py).

Variants
  privileged : obs = [height, pitch, vx, vy, pitch rate, slope, sin(phase), cos(phase)]
               (x dropped; the wave is measured)
  onboard    : obs = 4 stacked frames of [height, pitch, vx, vy, pitch rate]
               (no wave measurement at all; the policy must read the wave from
               its own motion history -> POMDP with a short memory)
Both train on randomised waves (domain randomisation) with TimeLimit 20 s,
RecordEpisodeStatistics and VecNormalize.

Evaluation (metrics.py) on the nominal wave and on randomised waves, against
LQI at the set-point chosen in objective 2 and MPPI with the same cost.

Run:  python -m control_envs.skate_lab.rl_ppo [--steps 300000] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "control_envs.skate_lab"

import numpy as np

from .envpath import OUT_DIR, SkateConfig
from .controllers import LQISpeed, MPPI, envelope_cost
from .metrics import evaluate, evaluate_parallel, markdown_table, strip_series
from .wrappers import make_env

LAM_AIR, LAM_LAND, CRASH_PEN = 2.0, 1.0, 50.0
REWARD = dict(lam_air=LAM_AIR, lam_land=LAM_LAND, crash_pen=CRASH_PEN)
N_ENVS = 8
EPISODE_STEPS = 1200

COLS = [
    ("name", "policy", "{}"),
    ("mean_speed", "mean speed (m/s)", "{:.3f}"),
    ("airborne_frac", "airborne (%)", "{:.1%}"),
    ("lqi_air_same_speed", "LQI airborne at same speed (%)", "{:.1%}"),
    ("impact_top10", "impact top-10% (m/s)", "{:.2f}"),
    ("crashes", "crashes", "{}"),
    ("mean_return", "return / episode", "{:.2f}"),
    ("effort_rms", "effort RMS", "{:.3f}"),
    ("episodes", "episodes", "{}"),
    ("wall_s", "wall (s)", "{:.0f}"),
]


def env_kwargs(variant: str, random_waves: bool, with_reward: bool = True, time_limit=None):
    kw = dict(random_waves=random_waves, drop_x=True, drop_wave=(variant == "onboard"),
              frame_stack=4 if variant == "onboard" else None, time_limit=time_limit)
    if with_reward:
        kw["envelope_reward"] = REWARD
    return kw


class PPOPolicy:
    """metrics.py-compatible wrapper: normalises the observation like training did."""

    def __init__(self, model, vecnorm, name):
        self.model, self.vecnorm, self.name = model, vecnorm, name

    def reset(self, env=None):
        pass

    def __call__(self, obs, env=None):
        o = self.vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32))
        a, _ = self.model.predict(o, deterministic=True)
        return np.asarray(a, dtype=np.float32).reshape(1)


def train(variant: str, total_steps: int, cfg: SkateConfig, seed: int = 0):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    torch.set_num_threads(4)

    def thunk(i):
        return lambda: make_env(cfg, record_stats=True, seed=seed * 100 + i,
                                **env_kwargs(variant, random_waves=True, time_limit=EPISODE_STEPS))

    venv = VecNormalize(DummyVecEnv([thunk(i) for i in range(N_ENVS)]), norm_obs=True,
                        norm_reward=True, clip_obs=10.0, gamma=0.99)
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=256, n_epochs=10, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                policy_kwargs=dict(net_arch=[64, 64]), seed=seed, verbose=0, device="cpu")
    t0 = time.perf_counter()
    model.learn(total_timesteps=total_steps)
    wall = time.perf_counter() - t0
    model.save(OUT_DIR / f"ppo_{variant}")
    venv.save(str(OUT_DIR / f"vecnorm_{variant}.pkl"))
    # training-curve summary from the episode statistics buffer
    ep = list(model.ep_info_buffer)
    summary = {"variant": variant, "steps": total_steps, "wall_s": wall,
               "last_episodes_mean_return": float(np.mean([e["r"] for e in ep])) if ep else np.nan,
               "last_episodes_mean_length": float(np.mean([e["l"] for e in ep])) if ep else np.nan}
    venv.training = False
    venv.norm_reward = False
    return model, venv, summary


def load(variant: str, cfg: SkateConfig):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    model = PPO.load(OUT_DIR / f"ppo_{variant}", device="cpu")
    dummy = DummyVecEnv([lambda: make_env(cfg, **env_kwargs(variant, random_waves=True))])
    venv = VecNormalize.load(str(OUT_DIR / f"vecnorm_{variant}.pkl"), dummy)
    venv.training = False
    venv.norm_reward = False
    return model, venv


def interp_lqi(rows, x, key="airborne_frac"):
    if not rows:
        return np.nan
    xs = np.array([r["mean_speed"] for r in rows])
    ys = np.array([r[key] for r in rows])
    o = np.argsort(xs)
    return float(np.interp(x, xs[o], ys[o]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--load", action="store_true", help="skip training, load saved models")
    ap.add_argument("--train-only", action="store_true", help="train and save, skip the evaluation")
    ap.add_argument("--no-mppi", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()
    cfg = SkateConfig()
    steps = 30_000 if args.quick else args.steps
    T, warm = (600, 180) if args.quick else (1200, 180)
    N_nom, N_rnd, N_mppi = (2, 3, 2) if args.quick else (5, 8, 3)

    try:
        import stable_baselines3  # noqa: F401
    except ImportError:
        print("stable_baselines3 is not importable in this venv; nothing to train.")
        return

    obj2 = {}
    p2 = OUT_DIR / "obj2_results.json"
    if p2.exists():
        obj2 = json.loads(p2.read_text())
    v_nom = obj2.get("a_nominal", {}).get("v_env") or 2.5
    v_rnd = obj2.get("b_random", {}).get("chosen", {}).get("all waves pooled") or 2.0
    lqi_rows = obj2.get("a_nominal", {}).get("rows", [])
    print(f"LQI set-points from objective 2: nominal {v_nom} m/s, randomised {v_rnd} m/s")

    results = {"reward": REWARD, "training": {}}
    policies = {}
    for variant in ("privileged", "onboard"):
        if args.load:
            model, venv = load(variant, cfg)
            summary = {"variant": variant, "loaded": True}
        else:
            print(f"\n## training PPO [{variant}] for {steps} steps on randomised waves ...")
            model, venv, summary = train(variant, steps, cfg)
            print(f"   wall {summary['wall_s']/60:.1f} min; last-100-episode mean return "
                  f"{summary['last_episodes_mean_return']:.2f}, mean length {summary['last_episodes_mean_length']:.0f}")
        results["training"][variant] = summary
        policies[variant] = PPOPolicy(model, venv, f"PPO {variant}")

    if args.train_only:
        results["wall_s"] = time.perf_counter() - t_start
        (OUT_DIR / "rl_ppo_training.json").write_text(json.dumps(results, indent=1, default=str))
        print(f"\ntraining done, models saved under {OUT_DIR}; wall-clock {results['wall_s']/60:.1f} min")
        return
    if args.load and (OUT_DIR / "rl_ppo_training.json").exists():
        results["training"] = json.loads((OUT_DIR / "rl_ppo_training.json").read_text()).get("training", {})

    for setting, random_waves, N, v_lqi in (("nominal wave", False, N_nom, v_nom),
                                            ("randomised waves", True, N_rnd, v_rnd)):
        print(f"\n## evaluation on the {setting} ({N} episodes x {T/60:.0f} s)")
        rows = []
        seed0 = 0 if not random_waves else 100
        for variant, pol in policies.items():
            env = make_env(cfg, **env_kwargs(variant, random_waves=random_waves))
            rows.append(evaluate(env, pol, n_episodes=N, T=T, warmup=warm, seed0=seed0, record_first=False))
        env_raw = make_env(cfg, random_waves=random_waves, envelope_reward=REWARD)
        lqi = LQISpeed(cfg, v_lqi, q_e=1.0, q_i=1.0, r=0.2, feedforward=True, name=f"LQI+ff v*={v_lqi}")
        rows.append(evaluate(env_raw, lqi, n_episodes=N, T=T, warmup=warm, seed0=seed0, record_first=False))
        if not args.no_mppi:
            def mppi_factory():
                return MPPI(cfg, envelope_cost(LAM_AIR, LAM_LAND, CRASH_PEN), K=128, H=36, lam=0.1,
                            n_iter=2, replan_every=3, u_init=0.3, name=f"MPPI lam_air={LAM_AIR:g}")
            rows.append(evaluate_parallel(lambda: make_env(cfg, random_waves=random_waves, envelope_reward=REWARD),
                                          mppi_factory, n_episodes=N_mppi, T=T, procs=args.procs,
                                          name=f"MPPI lam_air={LAM_AIR:g}", warmup=warm, seed0=seed0,
                                          record_first=False))
        for r in rows:
            r["lqi_air_same_speed"] = interp_lqi(lqi_rows, r["mean_speed"]) if not random_waves else np.nan
        print(markdown_table(rows, COLS))
        results[setting] = {"rows": [strip_series(r) for r in rows], "table": markdown_table(rows, COLS)}

    results["wall_s"] = time.perf_counter() - t_start
    (OUT_DIR / "rl_ppo_results.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"\ntotal wall-clock: {results['wall_s']/60:.1f} min")


if __name__ == "__main__":
    main()
