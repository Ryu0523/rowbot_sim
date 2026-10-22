"""Objective 3 - ride WITH the wave (vx = -c) holding a chosen wave-frame position xi*.

1. Riding equilibria of the wave-frame surge dynamics (push u* that balances
   slope gravity, rolling resistance and drag at speed c) for xi* around the
   wave, with the open-loop eigenvalues of the reduced 2-state model
   [xi - xi*, w = vx + c] (from the full physics, fast contact/pitch modes
   residualised) next to the textbook one-point formula.
2. Open-loop check: hold u = u* from a small offset at an unstable and at a
   stable equilibrium.
3. Phase-LQR (optionally with an integrator) from several initial phases and
   speeds; convergence, settling time, effort.
4. MPPI with a phase-holding cost from the same starts.
5. Partially observed variant: phase and slope removed from the observation;
   a recursive-least-squares observer rebuilds the phase from pitch, vertical
   speed and pitch rate; the phase-LQR runs on the estimate.

Run:  python -m control_envs.skate_lab.obj3_surf [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "control_envs.skate_lab"

import numpy as np

from .envpath import OUT_DIR, SkateConfig
from .controllers import Controller, MPPI, PhaseLQR, PhaseObserver, phase_cost
from .metrics import markdown_table
from .model import quasi_static_surge, reduced_surge_model, wrap
from .wrappers import DropObs, make_env

XI_UNSTABLE = 0.6      # upper face: above the mean level, slope > 0 (gravity pulls the board with the wave)
XI_STABLE = -0.6       # lower face: below the mean level, same slope sign


class Const(Controller):
    def __init__(self, u, name="open loop u*"):
        self.u, self.name = u, name

    def act(self, obs, env=None):
        return self.u


def region(xi):
    face = "face" if math.cos(xi) > 0 else ("back" if math.cos(xi) < 0 else "crest/trough")
    half = "upper" if math.sin(xi) > 1e-9 else ("lower" if math.sin(xi) < -1e-9 else "mean level")
    return f"{half} {face}"


def classify(eig):
    if np.max(eig.real) > 1e-3:
        return "unstable (saddle)"
    if np.max(np.abs(eig.imag)) > 1e-3:
        return "oscillation, damping %.3f" % (-np.max(eig.real))
    return "stable"


def run_surf(env, ctrl, xi_star, xi0, vx0, T, observer=None):
    """Run one riding episode from wave-frame position xi0 at speed vx0 (resting
    on the surface) and return convergence metrics and the series."""
    obs, _ = env.reset(seed=0, options={"xi": xi0, "vx": vx0, "rest": True})
    ctrl.reset(env)
    uw = env.unwrapped
    dt = uw.cfg.dt
    xi_l, w_l, u_l, est_l = [], [], [], []
    crashed = False
    for t in range(T):
        a = ctrl(obs, env)
        obs, _, term, trunc, _ = env.step(a)
        xi_l.append(float(wrap(uw.k * uw.state[0] + uw.phase - xi_star)))
        w_l.append(uw.state[3] + uw.cfg.wave_speed)
        u_l.append(float(a[0]))
        if observer is not None:
            est_l.append(float(wrap(observer.xi_hat - (uw.k * uw.state[0] + uw.phase))))
        if term:
            crashed = True
            break
    dxi = np.array(xi_l)
    w = np.array(w_l)
    u = np.array(u_l)
    n = len(dxi)
    tol = 0.05
    inside = np.abs(dxi) < tol
    settle = np.nan
    if n > 0 and inside[-1]:
        last_out = np.where(~inside)[0]
        settle = (last_out[-1] + 1) * dt if last_out.size else 0.0
    n_tail = min(120, n)                                  # last 2 s
    out = {
        "xi0": xi0, "vx0": vx0, "crashed": crashed,
        "converged": bool(n == T and np.all(inside[-n_tail:])),
        "riding": bool(n == T and np.sqrt(np.mean(dxi[-n_tail:] ** 2)) < 0.15
                       and np.sqrt(np.mean(w[-n_tail:] ** 2)) < 0.1),
        "settle_s": settle,
        "final_dxi": float(dxi[-1]) if n else np.nan,
        "rms_dxi_tail": float(np.sqrt(np.mean(dxi[-n_tail:] ** 2))) if n else np.nan,
        "rms_w_tail": float(np.sqrt(np.mean(w[-n_tail:] ** 2))) if n else np.nan,
        "effort_rms": float(np.sqrt(np.mean(u ** 2))) if n else np.nan,
        "series": {"t": np.arange(1, n + 1) * dt, "dxi": dxi, "w": w, "u": u},
    }
    if observer is not None and est_l:
        est = np.array(est_l)
        out["est_err_rms_after_2s"] = float(np.sqrt(np.mean(est[min(120, n - 1):] ** 2)))
        out["est_err_max_after_2s"] = float(np.max(np.abs(est[min(120, n - 1):])))
        out["series"]["est_err"] = est
    return out


RUN_COLS = [
    ("name", "controller", "{}"),
    ("xi0", "start xi (rad)", "{:+.2f}"),
    ("vx0", "start vx (m/s)", "{:+.1f}"),
    ("converged", "converged (|dxi|<0.05 last 2 s)", "{}"),
    ("settle_s", "settling time (s)", "{:.2f}"),
    ("rms_dxi_tail", "tail RMS dxi (rad)", "{:.3f}"),
    ("rms_w_tail", "tail RMS w (m/s)", "{:.3f}"),
    ("effort_rms", "effort RMS", "{:.3f}"),
    ("crashed", "crashed", "{}"),
]
OBS_COLS = RUN_COLS[:4] + [("riding", "riding (tail RMS dxi<0.15, |w|<0.1)", "{}"),
                           ("est_err_rms_after_2s", "phase est. error RMS after 2 s (rad)", "{:.3f}"),
                           ("est_err_max_after_2s", "max after 2 s (rad)", "{:.2f}")] + RUN_COLS[4:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--no-mppi", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()
    cfg = SkateConfig()
    T = 360 if args.quick else 600                 # 6 s / 10 s
    results = {}

    # ---------------- 1. equilibria and eigenvalues --------------------------
    print("## 1. riding equilibria (vx = -c = %.1f m/s) and open-loop eigenvalues of the surge model" % -cfg.wave_speed)
    rows = []
    for xi in np.linspace(-math.pi, math.pi, 13)[:-1]:
        red = reduced_surge_model(cfg, xi)
        u1, eig1, _, _ = quasi_static_surge(cfg, xi)
        rows.append({"xi": xi, "region": region(xi), "u_full": red["u_eq"], "u_1pt": u1,
                     "eig": ", ".join(f"{e.real:+.2f}{e.imag:+.2f}j" for e in red["eig_slow"]),
                     "eig_1pt": ", ".join(f"{e.real:+.2f}{e.imag:+.2f}j" for e in eig1),
                     "cls": classify(red["eig_slow"])})
    cols = [("xi", "xi* (rad)", "{:+.2f}"), ("region", "where on the wave", "{}"),
            ("u_full", "u* full model", "{:+.3f}"), ("u_1pt", "u* one-point formula", "{:+.3f}"),
            ("eig", "eigenvalues full/reduced (1/s)", "{}"), ("eig_1pt", "eigenvalues one-point (1/s)", "{}"),
            ("cls", "open loop", "{}")]
    print(markdown_table(rows, cols))
    results["equilibria"] = {"rows": rows, "table": markdown_table(rows, cols)}

    # ---------------- 2. open-loop check -------------------------------------
    print("\n## 2. open loop: hold u = u* from xi* + 0.05")
    rows_ol = []
    env = make_env(cfg)
    for xi_s in (XI_UNSTABLE, XI_STABLE):
        red = reduced_surge_model(cfg, xi_s)
        r = run_surf(env, Const(red["u_eq"]), xi_s, xi_s + 0.05, -cfg.wave_speed, T)
        d = r["series"]["dxi"]
        out = np.where(np.abs(d) > 0.5)[0]
        rows_ol.append({"xi_star": xi_s, "region": region(xi_s), "u_eq": red["u_eq"],
                        "eig": ", ".join(f"{e.real:+.2f}{e.imag:+.2f}j" for e in red["eig_slow"]),
                        "left_s": (out[0] + 1) * cfg.dt if out.size else np.nan,
                        "max_dxi": float(np.max(np.abs(d))), "crashed": r["crashed"]})
    cols_ol = [("xi_star", "xi* (rad)", "{:+.2f}"), ("region", "where", "{}"), ("u_eq", "u*", "{:+.3f}"),
               ("eig", "eigenvalues (1/s)", "{}"), ("left_s", "time until |dxi| > 0.5 (s)", "{:.2f}"),
               ("max_dxi", "max |dxi| in run (rad)", "{:.2f}"), ("crashed", "crashed", "{}")]
    print(markdown_table(rows_ol, cols_ol))
    results["open_loop"] = {"rows": rows_ol, "table": markdown_table(rows_ol, cols_ol)}

    # ---------------- 3. phase-LQR from several starts ------------------------
    print("\n## 3. phase-LQR from several initial phases and speeds")
    deltas = [-2.0, -1.0, -0.3, 0.3, 1.0, 2.0, 3.0]
    if args.quick:
        deltas = [-1.0, 0.3, 3.0]
    series_lqr = {}
    rows_lqr = []
    for xi_s in (XI_UNSTABLE, XI_STABLE):
        ctrl = PhaseLQR(cfg, xi_s, q_xi=1.0, q_w=0.3, r=0.05)
        print(ctrl.describe())
        for vx0 in (-cfg.wave_speed, 0.0):
            for d in deltas:
                r = run_surf(env, ctrl, xi_s, xi_s + d, vx0, T)
                r["name"] = f"{ctrl.name} xi*={xi_s:+.1f}"
                rows_lqr.append(r)
                series_lqr[(xi_s, vx0, d)] = r["series"]
    print(markdown_table(rows_lqr, RUN_COLS))
    conv = sum(r["converged"] for r in rows_lqr)
    print(f"converged {conv}/{len(rows_lqr)} runs; mean settling time of converged runs "
          f"{np.nanmean([r['settle_s'] for r in rows_lqr if r['converged']]):.2f} s; "
          f"mean effort RMS {np.mean([r['effort_rms'] for r in rows_lqr]):.3f}")
    ctrl_i = PhaseLQR(cfg, XI_UNSTABLE, q_xi=1.0, q_w=0.3, r=0.05, integrator=True)
    print(ctrl_i.describe())
    rows_lqi = []
    for d in (deltas[0], deltas[-1]):
        r = run_surf(env, ctrl_i, XI_UNSTABLE, XI_UNSTABLE + d, 0.0, T)
        r["name"] = f"{ctrl_i.name} xi*={XI_UNSTABLE:+.1f}"
        rows_lqi.append(r)
    print(markdown_table(rows_lqi, RUN_COLS))
    results["phase_lqr"] = {
        "rows": [{k: v for k, v in r.items() if k != "series"} for r in rows_lqr + rows_lqi],
        "table": markdown_table(rows_lqr + rows_lqi, RUN_COLS),
        "converged": conv, "n": len(rows_lqr),
        "controller_unstable": PhaseLQR(cfg, XI_UNSTABLE, q_xi=1.0, q_w=0.3, r=0.05).describe(),
        "controller_stable": PhaseLQR(cfg, XI_STABLE, q_xi=1.0, q_w=0.3, r=0.05).describe(),
        "controller_integrator": ctrl_i.describe()}

    # ---------------- 4. MPPI with a phase-holding cost -----------------------
    rows_mppi = []
    if not args.no_mppi:
        print("\n## 4. MPPI with phase-holding cost (true model, K=128, H=24)")
        t0 = time.perf_counter()
        starts = [(-1.0, 0.0), (1.0, 0.0), (3.0, 0.0), (-1.0, -cfg.wave_speed)]
        if args.quick:
            starts = starts[:2]

        def one(start):
            d, vx0 = start
            env_m = make_env(cfg)
            m = MPPI(cfg, phase_cost(XI_UNSTABLE, lam_w=0.2, r_u=0.02), K=128, H=24, lam=0.1, n_iter=2,
                     replan_every=3, u_init=0.0, seed=0, name="MPPI phase")
            r = run_surf(env_m, m, XI_UNSTABLE, XI_UNSTABLE + d, vx0, T)
            r["name"] = f"MPPI phase xi*={XI_UNSTABLE:+.1f}"
            return r

        if args.procs > 1:
            import multiprocessing as mp
            import cloudpickle
            ctx = mp.get_context("spawn")
            with ctx.Pool(min(args.procs, len(starts))) as pool:
                rows_mppi = pool.map(_mppi_worker, [cloudpickle.dumps((one, s)) for s in starts])
        else:
            rows_mppi = [one(s) for s in starts]
        print(markdown_table(rows_mppi, RUN_COLS))
        print(f"({time.perf_counter()-t0:.0f} s wall for {len(starts)} runs)")
        results["mppi"] = {"rows": [{k: v for k, v in r.items() if k != "series"} for r in rows_mppi],
                           "table": markdown_table(rows_mppi, RUN_COLS)}

    # ---------------- 5. partially observed: observer + phase-LQR -------------
    print("\n## 5. no wave measurement: phase from an RLS observer on pitch, vy, pitch rate")
    env_po = DropObs(make_env(cfg), drop_x=False, drop_wave=True)   # obs = [x, height, pitch, vx, vy, om]
    rows_obs = []
    series_obs = {}
    for xi_s in (XI_UNSTABLE, XI_STABLE):
        for vx0 in (-cfg.wave_speed, 0.0):
            for d in ([-1.0, 0.3, 1.0, 3.0] if not args.quick else [0.3, 3.0]):
                observer = PhaseObserver(cfg)
                ctrl = PhaseLQR(cfg, xi_s, q_xi=1.0, q_w=0.3, r=0.05, phase_source=observer,
                                name="phase-LQR + observer")
                r = run_surf(env_po, ctrl, xi_s, xi_s + d, vx0, T, observer=observer)
                r["name"] = f"{ctrl.name} xi*={xi_s:+.1f}"
                rows_obs.append(r)
                series_obs[(xi_s, vx0, d)] = r["series"]
    print(markdown_table(rows_obs, OBS_COLS))
    conv_o = sum(r["converged"] for r in rows_obs)
    ride_o = sum(r["riding"] for r in rows_obs)
    print(f"riding {ride_o}/{len(rows_obs)} (strict |dxi|<0.05: {conv_o}/{len(rows_obs)}); "
          f"phase estimation error RMS after 2 s: "
          f"median {np.nanmedian([r['est_err_rms_after_2s'] for r in rows_obs]):.3f} rad, "
          f"worst {np.nanmax([r['est_err_rms_after_2s'] for r in rows_obs]):.3f} rad")
    results["observer"] = {"rows": [{k: v for k, v in r.items() if k != "series"} for r in rows_obs],
                           "table": markdown_table(rows_obs, OBS_COLS), "converged": conv_o,
                           "riding": ride_o, "n": len(rows_obs)}

    # ---------------- plot ------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for (xi_s, vx0, d), s in series_lqr.items():
            if xi_s != XI_UNSTABLE:
                continue
            axes[0].plot(s["t"], s["dxi"], lw=1, color="C0" if vx0 == 0.0 else "C1", alpha=0.8)
        axes[0].set_title(f"phase-LQR, xi* = {XI_UNSTABLE:+.1f} (unstable open loop)\nblue: start at rest, orange: start at wave speed")
        axes[0].set_ylabel("xi - xi* (rad)")
        for r in rows_mppi:
            s = r["series"]
            axes[1].plot(s["t"], s["dxi"], lw=1)
        axes[1].set_title("MPPI with phase cost")
        for (xi_s, vx0, d), s in series_obs.items():
            if xi_s != XI_UNSTABLE:
                continue
            axes[2].plot(s["t"], s["dxi"], lw=1, color="C0" if vx0 == 0.0 else "C1", alpha=0.8)
            if "est_err" in s:
                axes[2].plot(s["t"], s["est_err"], lw=0.6, color="C3", alpha=0.5)
        axes[2].set_title("phase-LQR on the observer estimate\n(red: estimation error)")
        for a in axes:
            a.set_xlabel("t (s)")
            a.axhline(0, color="k", lw=0.5)
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "obj3_phase.png", dpi=120)
        print(f"plot saved: {OUT_DIR / 'obj3_phase.png'}")
    except Exception as e:  # pragma: no cover
        print("plot failed:", e)

    results["wall_s"] = time.perf_counter() - t_start
    (OUT_DIR / "obj3_results.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"\ntotal wall-clock: {results['wall_s']/60:.1f} min")


def _mppi_worker(payload):
    import cloudpickle
    fn, start = cloudpickle.loads(payload)
    return fn(start)


if __name__ == "__main__":
    main()
