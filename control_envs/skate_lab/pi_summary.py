"""Plain PI controller: run it across speed set-points on the nominal and on
randomised waves and summarise (the objective-1/2 picture for PI alone).

    python -m control_envs.skate_lab.pi_summary [--ff] [--episodes 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "control_envs.skate_lab"

from .envpath import SkateConfig
from .controllers import PISpeed
from .metrics import evaluate, markdown_table
from .wrappers import make_env

COLS = [
    ("name", "controller", "{}"),
    ("v_ref", "set-point (m/s)", "{:.2f}"),
    ("mean_speed", "mean speed (m/s)", "{:.3f}"),
    ("speed_rmse", "speed RMSE (m/s)", "{:.3f}"),
    ("airborne_frac", "airborne (%)", "{:.1%}"),
    ("impact_top10", "impact top-10% (m/s)", "{:.2f}"),
    ("crashes", "crashes", "{}"),
    ("effort_rms", "effort RMS", "{:.3f}"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff", action="store_true", help="add the slope feed-forward")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()
    cfg = SkateConfig()
    T = int(round(args.seconds / cfg.dt))
    for setting, rnd in (("nominal wave", False), ("randomised waves", True)):
        rows = []
        for v in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
            env = make_env(cfg, random_waves=rnd)
            ctrl = PISpeed(cfg, v, kp=1.0, ki=0.5, feedforward=args.ff)
            r = evaluate(env, ctrl, n_episodes=args.episodes, T=T, warmup=180,
                         seed0=0 if not rnd else 100, v_ref=v, record_first=False)
            r["v_ref"] = v
            rows.append(r)
        print(f"\n## {setting}")
        print(markdown_table(rows, COLS))


if __name__ == "__main__":
    main()
