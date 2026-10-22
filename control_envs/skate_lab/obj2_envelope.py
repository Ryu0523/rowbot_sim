"""Objective 2 - "as fast as possible under constraints".

(a) Envelope: sweep the LQI speed set-point 0.5..4.0 m/s on the nominal wave and
    tabulate mean speed vs airborne fraction, landing impact and crashes; read
    off the highest admissible speed under the stated budgets.
(b) Slow layer: the same sweep on randomised waves (3 x N_draw random
    (A, wavelength, speed) draws, the same draws for every set-point), grouped
    post hoc into three steepness classes (A k = maximum slope); the set-point
    is chosen from an UPPER quantile of the airborne fraction over the draws
    (uncertainty-aware), per class and pooled.
(c) Per-wave question at matched mean speed: constant-speed LQI vs bang-bang on
    pitch vs MPPI with cost = -progress + lam_air*airtime + lam_land*impact + crash.

Budgets used here (stated, not derived): airborne fraction <= 10 %, zero crashes
in the evaluation time (5 x 20 s nominal; 8 x 20 s per class randomised).

Run:  python -m control_envs.skate_lab.obj2_envelope [--quick]
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
from .controllers import BangBangPitch, LQISpeed, MPPI, envelope_cost
from .metrics import evaluate, evaluate_parallel, markdown_table, strip_series
from .wrappers import make_env

BUDGET_AIR = 0.10
QUANTILE = 0.8
SPEEDS = np.round(np.arange(0.5, 4.01, 0.25), 2)
LAM_AIRS = [0.5, 1.0, 2.0, 4.0, 8.0]
LAM_LAND = 1.0
CRASH_PEN = 50.0

ENV_COLS = [
    ("v_ref", "set-point (m/s)", "{:.2f}"),
    ("mean_speed", "mean speed (m/s)", "{:.3f}"),
    ("airborne_frac", "airborne (%)", "{:.1%}"),
    ("impact_mean", "impact mean (m/s)", "{:.2f}"),
    ("impact_top10", "impact top-10% (m/s)", "{:.2f}"),
    ("crashes", "crashes", "{}"),
    ("effort_rms", "effort RMS", "{:.3f}"),
]
CMP_COLS = [
    ("name", "controller", "{}"),
    ("mean_speed", "mean speed (m/s)", "{:.3f}"),
    ("airborne_frac", "airborne (%)", "{:.1%}"),
    ("lqi_air_same_speed", "LQI airborne at same speed (%)", "{:.1%}"),
    ("impact_top10", "impact top-10% (m/s)", "{:.2f}"),
    ("lqi_imp_same_speed", "LQI impact top-10% at same speed", "{:.2f}"),
    ("crashes", "crashes", "{}"),
    ("effort_rms", "effort RMS", "{:.3f}"),
    ("episodes", "episodes", "{}"),
    ("wall_s", "wall (s)", "{:.0f}"),
]


def lqi(cfg, v):
    return LQISpeed(cfg, v, q_e=1.0, q_i=1.0, r=0.2, feedforward=True)


def admissible_speed(rows, air_key="airborne_frac", budget=BUDGET_AIR):
    """Highest set-point such that it and every lower set-point satisfy the
    budget (airborne <= budget, no crash). None if even the lowest fails."""
    best = None
    for r in rows:
        if r[air_key] <= budget and r["crashes"] == 0:
            best = r["v_ref"]
        else:
            break
    return best


def mppi_env_factory(cfg, lam_air, v_anchor=None, lam_v=0.0, seed=0):
    def f():
        return MPPI(cfg, envelope_cost(lam_air, LAM_LAND, CRASH_PEN, v_anchor=v_anchor, lam_v=lam_v),
                    K=128, H=36, lam=0.1, n_iter=2, anneal=0.6, replan_every=3, sigma0=0.5,
                    u_init=0.3, seed=seed, name="MPPI")
    return f


def interp_curve(rows, x, key):
    xs = np.array([r["mean_speed"] for r in rows])
    ys = np.array([r[key] for r in rows])
    o = np.argsort(xs)
    return float(np.interp(x, xs[o], ys[o]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--no-mppi", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()
    cfg = SkateConfig()
    T = 600 if args.quick else 1200
    warm = 180
    N_nom = 2 if args.quick else 5
    N_draw = 3 if args.quick else 8
    N_mppi = 2 if args.quick else 3
    speeds = SPEEDS[::2] if args.quick else SPEEDS
    results = {"budget_air": BUDGET_AIR, "quantile": QUANTILE}

    # ---------------- (a) nominal envelope ----------------------------------
    print("## (a) envelope on the nominal wave, LQI+ff speed hold")
    env = make_env(cfg)
    rows_a = []
    t0 = time.perf_counter()
    for v in speeds:
        r = evaluate(env, lqi(cfg, v), n_episodes=N_nom, T=T, v_ref=v, warmup=warm, record_first=False)
        r["v_ref"] = float(v)
        rows_a.append(r)
    v_env = admissible_speed(rows_a)
    print(markdown_table(rows_a, ENV_COLS))
    print(f"budget: airborne <= {BUDGET_AIR:.0%}, crashes = 0  ->  highest admissible set-point: {v_env} m/s"
          f"   ({time.perf_counter()-t0:.0f} s)")
    results["a_nominal"] = {"rows": [strip_series(r) for r in rows_a], "v_env": v_env,
                           "table": markdown_table(rows_a, ENV_COLS)}

    # ---------------- (b) randomised waves, upper quantile -------------------
    print("\n## (b) slow layer: randomised waves, airborne fraction upper quantile per steepness class")
    t0 = time.perf_counter()
    rng = np.random.default_rng(2026)
    waves = [(float(rng.uniform(0.05, 0.30)), float(rng.uniform(2.0, 5.0)), float(rng.uniform(0.5, 2.5)))
             for _ in range(N_draw * 3)]
    steep = np.array([2 * np.pi * A / L for A, L, _ in waves])          # A k = max slope
    edges = np.quantile(steep, [1 / 3, 2 / 3])
    cls_of = np.digitize(steep, edges)                                     # 0 gentle, 1 medium, 2 steep
    class_names = [f"gentle (A k <= {edges[0]:.2f})", f"medium (A k {edges[0]:.2f}-{edges[1]:.2f})",
                   f"steep (A k > {edges[1]:.2f})"]
    env_r = make_env(cfg, random_waves=True)
    per_wave = np.zeros((len(speeds), len(waves)))        # airborne fraction
    crashed = np.zeros((len(speeds), len(waves)), dtype=bool)
    for wi, w in enumerate(waves):
        env_r.force(w)
        for vi, v in enumerate(speeds):
            r = evaluate(env_r, lqi(cfg, v), n_episodes=1, T=T, v_ref=v, warmup=warm, seed0=500 + wi,
                         record_first=False)
            per_wave[vi, wi] = r["airborne_frac"]
            crashed[vi, wi] = r["crashes"] > 0
    rows_b = {}
    chosen = {}
    for ci, cname in enumerate(class_names):
        idx = np.where(cls_of == ci)[0]
        rows = []
        for vi, v in enumerate(speeds):
            rows.append({"v_ref": float(v), "air_q": float(np.quantile(per_wave[vi, idx], QUANTILE)),
                         "air_mean": float(per_wave[vi, idx].mean()), "crashes": int(crashed[vi, idx].sum()),
                         "n": int(len(idx))})
        rows_b[cname] = rows
        chosen[cname] = admissible_speed(rows, air_key="air_q")
    pooled = []
    for vi, v in enumerate(speeds):
        pooled.append({"v_ref": float(v), "air_q": float(np.quantile(per_wave[vi], QUANTILE)),
                       "air_mean": float(per_wave[vi].mean()), "crashes": int(crashed[vi].sum()), "n": len(waves)})
    chosen["all waves pooled"] = admissible_speed(pooled, air_key="air_q")
    rows_b["all waves pooled"] = pooled
    head = "| set-point (m/s) | " + " | ".join(f"{c} (n={rows_b[c][0]['n']}): airborne q{int(QUANTILE*100)} / crashes"
                                             for c in rows_b) + " |"
    lines = [head, "|" + "---|" * (len(rows_b) + 1)]
    for vi, v in enumerate(speeds):
        cells = [f"{v:.2f}"] + [f"{rows_b[c][vi]['air_q']:.1%} / {rows_b[c][vi]['crashes']}" for c in rows_b]
        lines.append("| " + " | ".join(cells) + " |")
    table_b = "\n".join(lines)
    print(table_b)
    print("chosen set-point per class (airborne q%d <= %.0f%%, no crash):" % (QUANTILE * 100, BUDGET_AIR * 100))
    for c, v in chosen.items():
        print(f"  {c}: {v} m/s")
    print(f"  ({time.perf_counter()-t0:.0f} s)")
    results["b_random"] = {"chosen": chosen, "table": table_b, "rows": rows_b,
                           "waves": waves, "class_of_wave": cls_of.tolist(),
                           "per_wave_airborne": per_wave.tolist(), "per_wave_crashed": crashed.tolist()}

    # ---------------- (c) per-wave modulation at matched speed ---------------
    print("\n## (c) matched mean speed: constant-speed LQI vs bang-bang vs MPPI (nominal wave)")
    rows_c = []
    r = evaluate(env, BangBangPitch(1.0), n_episodes=N_nom, T=T, warmup=warm, record_first=False)
    rows_c.append(r)
    for v_bb in (2.5, 3.0, 3.5):
        for amp in (0.3, 0.6):
            r = evaluate(env, BangBangPitch(amp, base=lqi(cfg, v_bb)), n_episodes=N_nom, T=T,
                         warmup=warm, record_first=False)
            rows_c.append(r)
    if not args.no_mppi:
        for la in LAM_AIRS:
            r = evaluate_parallel(lambda: make_env(cfg), mppi_env_factory(cfg, la), n_episodes=N_mppi,
                                  T=T, procs=args.procs, name=f"MPPI lam_air={la:g}", warmup=warm,
                                  record_first=False)
            rows_c.append(r)
        anchors = [v_env] if v_env is not None else [2.0]
        if v_env is not None:
            anchors.append(v_env + 0.5)
        for va in anchors:
            r = evaluate_parallel(lambda: make_env(cfg), mppi_env_factory(cfg, 2.0, v_anchor=va, lam_v=2.0),
                                  n_episodes=N_mppi, T=T, procs=args.procs,
                                  name=f"MPPI lam_air=2 + anchor v={va:.2f}", warmup=warm, record_first=False)
            rows_c.append(r)
    for r in rows_c:
        r["lqi_air_same_speed"] = interp_curve(rows_a, r["mean_speed"], "airborne_frac")
        r["lqi_imp_same_speed"] = interp_curve(rows_a, r["mean_speed"], "impact_top10")
    print(markdown_table(rows_c, CMP_COLS))
    results["c_matched"] = {"rows": [strip_series(r) for r in rows_c], "table": markdown_table(rows_c, CMP_COLS)}

    # plot the envelope curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        xs = [r["mean_speed"] for r in rows_a]
        ax[0].plot(xs, [100 * r["airborne_frac"] for r in rows_a], "o-", label="LQI constant speed")
        ax[1].plot(xs, [r["impact_top10"] for r in rows_a], "o-", label="LQI constant speed")
        for r in rows_c:
            mk = "s" if r["name"].startswith("MPPI") else "^"
            ax[0].plot(r["mean_speed"], 100 * r["airborne_frac"], mk, label=r["name"])
            ax[1].plot(r["mean_speed"], r["impact_top10"], mk, label=r["name"])
        ax[0].axhline(100 * BUDGET_AIR, color="r", ls="--", lw=0.8)
        ax[0].set_xlabel("mean speed (m/s)"); ax[0].set_ylabel("airborne (%)")
        ax[1].set_xlabel("mean speed (m/s)"); ax[1].set_ylabel("impact top-10% (m/s)")
        for a in ax:
            a.grid(alpha=0.3)
        ax[1].legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "obj2_envelope.png", dpi=120)
        print(f"plot saved: {OUT_DIR / 'obj2_envelope.png'}")
    except Exception as e:  # pragma: no cover
        print("plot failed:", e)

    results["wall_s"] = time.perf_counter() - t_start
    (OUT_DIR / "obj2_results.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"\ntotal wall-clock: {results['wall_s']/60:.1f} min")


if __name__ == "__main__":
    main()
