"""Objective 1 - hold the board's speed at 1.0 and 2.0 m/s over the waves.

Compares PI, PI + slope feed-forward, LQI (+ff) and MPPI (speed cost) on the
nominal wave and on randomised waves; prints the tracking tables, the LQI
closed-loop eigenvalues / gain margin, and saves a plot of vx(t).

Run:  python -m control_envs.skate_lab.obj1_speed  [--quick]
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
from .controllers import LQISpeed, MPPI, PISpeed, speed_cost
from .metrics import evaluate, evaluate_parallel, markdown_table, strip_series
from .model import verify_against_env
from .wrappers import make_env

COLS = [
    ("name", "controller", "{}"),
    ("mean_speed", "mean speed (m/s)", "{:.3f}"),
    ("speed_rmse", "speed RMSE (m/s)", "{:.3f}"),
    ("effort_rms", "effort RMS", "{:.3f}"),
    ("airborne_frac", "airborne (%)", "{:.1%}"),
    ("crashes", "crashes", "{}"),
    ("episodes", "episodes", "{}"),
    ("wall_s", "wall (s)", "{:.0f}"),
]


def linear_controllers(cfg, v):
    return [
        PISpeed(cfg, v, kp=1.0, ki=0.5),
        PISpeed(cfg, v, kp=1.0, ki=0.5, feedforward=True),
        LQISpeed(cfg, v, q_e=1.0, q_i=1.0, r=0.2, feedforward=False),
        LQISpeed(cfg, v, q_e=1.0, q_i=1.0, r=0.2, feedforward=True),
    ]


def mppi_factory(cfg, v, seed=0):
    return lambda: MPPI(cfg, speed_cost(v, r_u=0.05), K=128, H=24, lam=0.1, n_iter=2,
                        anneal=0.6, replan_every=3, sigma0=0.5, seed=seed, name="MPPI")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer/shorter episodes")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--no-mppi", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()
    cfg = SkateConfig()
    T = 600 if args.quick else 1200           # 10 s / 20 s per episode
    warm = 180                                # first 3 s (drop + spin-up) not scored
    N_nom = 2 if args.quick else 5
    N_rnd = 3 if args.quick else 8
    N_mppi = 2 if args.quick else 4

    print("## model check")
    ver = verify_against_env(n_seq=4, T=300)
    print(f"batched model vs env: max |deviation| = {ver['max_abs_deviation']:.2e} "
          f"over {ver['steps']} steps")

    results = {"verify": ver, "speeds": {}}
    series = {}
    for v in (1.0, 2.0):
        print(f"\n## v* = {v} m/s")
        rows_nom, rows_rnd = [], []
        env_nom = make_env(cfg)
        env_rnd = make_env(cfg, random_waves=True)
        for ctrl in linear_controllers(cfg, v):
            if hasattr(ctrl, "describe"):
                print(ctrl.describe())
            r = evaluate(env_nom, ctrl, n_episodes=N_nom, T=T, v_ref=v, warmup=warm)
            rows_nom.append(r)
            series[(v, ctrl.name)] = r["series"]
            r2 = evaluate(env_rnd, ctrl, n_episodes=N_rnd, T=T, v_ref=v, warmup=warm, seed0=100)
            rows_rnd.append(r2)
        if not args.no_mppi:
            r = evaluate_parallel(lambda: make_env(cfg), mppi_factory(cfg, v), n_episodes=N_mppi,
                                  T=T, v_ref=v, procs=args.procs, name="MPPI", warmup=warm)
            rows_nom.append(r)
            series[(v, "MPPI")] = r["series"]
            r2 = evaluate_parallel(lambda: make_env(cfg, random_waves=True), mppi_factory(cfg, v),
                                   n_episodes=N_mppi, T=T, v_ref=v, procs=args.procs, name="MPPI",
                                   warmup=warm, seed0=100)
            rows_rnd.append(r2)
        print("\nnominal wave (A 0.15 m, wavelength 3 m, speed 1.5 m/s):")
        print(markdown_table(rows_nom, COLS))
        print("\nrandomised waves (A 0.05-0.3 m, wavelength 2-5 m, speed 0.5-2.5 m/s):")
        print(markdown_table(rows_rnd, COLS))
        lqi = [c for c in linear_controllers(cfg, v) if isinstance(c, LQISpeed)][-1]
        results["speeds"][str(v)] = {
            "nominal": [strip_series(r) for r in rows_nom],
            "random": [strip_series(r) for r in rows_rnd],
            "lqi": {"K": lqi.K[0].tolist(), "eig_discrete": [complex(e).real for e in lqi.eig_cl],
                    "eig_continuous_1_per_s": [complex(e).real for e in lqi.eig_cl_cont],
                    "gain_margin": list(lqi.gain_margin), "a": lqi.a, "b": lqi.b},
            "tables": {"nominal": markdown_table(rows_nom, COLS),
                       "random": markdown_table(rows_rnd, COLS)},
        }

    # plot vx(t)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
        for ax, v in zip(axes, (1.0, 2.0)):
            for (vv, name), s in series.items():
                if vv != v or s is None:
                    continue
                ax.plot(s["t"], s["vx"], label=name, lw=1)
            ax.axhline(v, color="k", ls="--", lw=0.8)
            ax.set_ylabel("vx (m/s)")
            ax.set_title(f"speed hold at {v} m/s, nominal wave (first episode)")
            ax.set_xlim(0, min(12, s["t"][-1]))
            ax.grid(alpha=0.3)
        axes[0].legend(ncol=5, fontsize=8)
        axes[1].set_xlabel("t (s)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "obj1_vx.png", dpi=120)
        print(f"\nplot saved: {OUT_DIR / 'obj1_vx.png'}")
    except Exception as e:  # pragma: no cover
        print("plot failed:", e)

    results["wall_s"] = time.perf_counter() - t_start
    (OUT_DIR / "obj1_results.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"\ntotal wall-clock: {results['wall_s']/60:.1f} min")


if __name__ == "__main__":
    main()
