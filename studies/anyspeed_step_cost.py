#!/usr/bin/env python3
"""
Compute budget of the current high-fidelity plant (sim/planing_vessel_gz.py
on sim/planing_vessel.py), measured, so that an any-speed redesign knows
what one plant step and one control step cost today.

Measures (read-only: no tracked file is touched, nothing is cached outside
studies/_cache/anyspeed/):
  build    constructor of the GZ plant the final evaluation builds
           (Mission('high') + from_plant(record=True)), cold and warm
           (running_attitude and the GZ table are cached per process)
  step     PlaningVesselGZ.step at the evaluation's dt 0.02 s, constant
           commands, in the task sea (24 x 5 = 120 components), the jittered
           training-like sea (48 x 8 = 384), and calm water, at 0, 5, 12,
           20 (0.8 u_ref) and 25 kn starts
  parts    one deriv, the wave evaluation (_waves at 48 stations), the
           lateral-rate eigenvalue, prop_submergence + update_prime

    .venv/Scripts/python.exe -m studies.anyspeed_step_cost
"""
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np   # noqa: E402

OUT = os.path.join(HERE, "_cache", "anyspeed")
KN = 0.514444


def timeit(fn, n):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    os.makedirs(OUT, exist_ok=True)
    from learn.repro import task
    from sim.planing_vessel_gz import from_plant
    import sim.planing_vessel as PV
    import sim.planing_vessel_gz as PG
    res = {}
    t0 = time.perf_counter()
    m = task.Mission("high", 0, 0, t_end=1.0, track=task.LEGS[0])
    res["mission_build_cold_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pl = from_plant(m.plant, record=True)
    res["gz_from_plant_warm_s"] = time.perf_counter() - t0
    # cold GZ build: clear the per-process caches
    PG._GZ.clear()
    PV._STEADY.clear()
    t0 = time.perf_counter()
    pl = from_plant(m.plant, record=True)
    res["gz_from_plant_cold_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    s0 = pl.initial_state(0.8 * m.u_ref)
    res["initial_state_cold_s"] = time.perf_counter() - t0
    dt = m.dt
    res["dt"] = dt
    res["sub_per_control_step"] = m.sub
    res["n_components_task_sea"] = int(np.size(m.sea.a))

    m_j = task.Mission("high", 100, 0, t_end=1.0, track=task.LEGS[0],
                       sea=dict(hs=1.0, tp=5.0, theta0=float(np.pi),
                                n_freq=48, n_dir=8, jitter=True,
                                spread_s=8.0))
    pl_j = from_plant(m_j.plant, record=True)
    res["n_components_jitter_sea"] = int(np.size(m_j.sea.a))
    from sim.test_vessel import Monochromatic
    pl_c = pl.with_sea(Monochromatic(1.0, 0.0))

    # ---- parts, at the 20 kn start in the task sea
    s = pl.initial_state(0.8 * m.u_ref)
    s[5] = 0.0
    X = s[0] + pl.x
    Y = np.zeros_like(X)
    parts = {}
    for name, p in (("task", pl), ("jitter", pl_j), ("calm", pl_c)):
        parts[f"deriv_{name}_us"] = 1e6 * timeit(lambda: p.deriv(s, 1.0),
                                                 2000)
        parts[f"waves_{name}_us"] = 1e6 * timeit(
            lambda: p._waves(X, Y, 1.0), 2000)
    _, aux = pl.deriv(s, 1.0)
    parts["lateral_rate_us"] = 1e6 * timeit(
        lambda: pl._lateral_rate(s, aux["d"]), 5000)
    parts["prop_sub_prime_us"] = 1e6 * timeit(
        lambda: pl.prop.update_prime(pl.prop_submergence(s[:6], 1.0), dt),
        5000)
    parts["gz_interp_us"] = 1e6 * timeit(lambda: pl.gz(0.1), 20000)
    res["parts"] = parts

    # ---- full steps, constant commands, 1200 steps (24 s) per case
    cases = []
    for sea_name, p_ in (("task", pl), ("jitter", pl_j), ("calm", pl_c)):
        for kn in (0.0, 5.0, 12.0, 0.8 * m.u_ref / KN, 25.0):
            p = p_
            p._reset()
            s = p.initial_state(kn * KN)
            thr = float(s[12])
            N = 1200
            nsub = []
            t = 0.0
            t0 = time.perf_counter()
            ok = True
            for i in range(N):
                s = p.step(s, t, thr, 0.0, dt)
                nsub.append(p.last_n_sub)
                t += dt
                if not np.all(np.isfinite(s)):
                    ok = False
                    break
            el = time.perf_counter() - t0
            n_done = len(nsub)
            ns = np.asarray(nsub)
            cases.append(dict(
                sea=sea_name, kn0=float(kn), thrust=thr, steps=n_done,
                finite=ok, us_per_step=1e6 * el / max(n_done, 1),
                ms_per_control_step=1e3 * el / max(n_done, 1) * m.sub,
                realtime_factor=(n_done * dt) / el,
                n_sub_mean=float(ns.mean()), n_sub_max=int(ns.max()),
                share_sub=float(np.mean(ns > 1)),
                derivs_per_step=float(np.mean(np.where(ns == 1, 2,
                                                       2 * ns))),
                kn_end=float(s[6] / KN), log_rows=len(p._log)))
            p._log.clear()
    res["cases"] = cases
    with open(os.path.join(OUT, "step_cost.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "cases"},
                     indent=1, default=float))
    for c in cases:
        print(f"{c['sea']:<6} start {c['kn0']:5.1f} kn -> {c['kn_end']:5.1f}"
              f" kn: {c['us_per_step']:7.0f} us/step, "
              f"{c['ms_per_control_step']:5.1f} ms/control step, x"
              f"{c['realtime_factor']:5.1f} real time, n_sub mean "
              f"{c['n_sub_mean']:.2f} max {c['n_sub_max']}, "
              f"derivs/step {c['derivs_per_step']:.2f}, "
              f"finite {c['finite']}")


if __name__ == "__main__":
    main()
