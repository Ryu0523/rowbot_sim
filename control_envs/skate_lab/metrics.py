"""Evaluation over N seeded episodes and markdown tables.

Metrics (all from the raw env state, independent of observation wrappers):
  mean_speed       mean of vx over every simulated step (m/s)
  speed_rmse       root-mean-square of (vx - v_ref) when a v_ref is given
  airborne_frac    fraction of steps with no wheel in contact (env info)
  impact_mean      mean touchdown impact (m/s): closing speed between the
                   board and the surface just before the first wheel touches
                   (surface vertical speed c*slope minus board vy, >= 0)
  impact_top10     mean of the largest 10 % of impacts (0 if no touchdown)
  crashes_per_min  episodes ended by the env's crash test, per simulated minute
  effort_rms       root-mean-square push u (fraction of max force)
  effort_abs       mean |u|
"""
from __future__ import annotations

import time

import numpy as np


def run_episode(env, policy, T: int, seed: int, record: bool = False) -> dict:
    """One episode of at most T control steps. ``policy(obs, env) -> action``;
    ``policy.reset(env)`` is called first if it exists."""
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(env)
    uw = env.unwrapped
    c = uw.cfg
    vx_l, u_l, air_l = [], [], []
    impacts, impact_steps = [], []
    prev_air = info.get("wheels_in_contact", uw.n_contacts) == 0
    _, _, ht_prev = uw.surface(uw.state[0])
    vy_prev = uw.state[4]
    crashed = False
    steps = 0
    ret = 0.0
    series = {"t": [], "vx": [], "u": [], "air": [], "pitch": [], "xi": []} if record else None
    for t in range(T):
        a = policy(obs, env)
        obs, r, term, trunc, info = env.step(a)
        ret += float(r)
        u = float(np.asarray(a).reshape(-1)[0])
        st = uw.state
        air = info["wheels_in_contact"] == 0
        if prev_air and not air:
            impacts.append(max(0.0, ht_prev - vy_prev))
            impact_steps.append(t)
        vx_l.append(st[3])
        u_l.append(u)
        air_l.append(air)
        if record:
            series["t"].append((t + 1) * c.dt)
            series["vx"].append(st[3])
            series["u"].append(u)
            series["air"].append(air)
            series["pitch"].append(st[2])
            series["xi"].append(uw.k * st[0] + uw.phase)
        _, _, ht_prev = uw.surface(st[0])
        vy_prev = st[4]
        prev_air = air
        steps += 1
        if term:
            crashed = True
            break
        if trunc:
            break
    out = {"vx": np.array(vx_l), "u": np.array(u_l), "air": np.array(air_l, dtype=bool),
           "impacts": np.array(impacts), "impact_steps": np.array(impact_steps, dtype=int),
           "crashed": crashed, "steps": steps, "ret": ret,
           "sim_seconds": steps * c.dt}
    if record:
        out["series"] = {k: np.array(v) for k, v in series.items()}
    return out


def aggregate(eps: list[dict], v_ref=None, name="policy", wall_s=np.nan, warmup: int = 0) -> dict:
    """Aggregate a list of ``run_episode`` results into the metric dict.
    ``warmup`` steps at the start of every episode are left out of the speed,
    effort, airborne and impact statistics (the board is dropped from rest)."""
    def cut(e, key):      # drop the warm-up, unless the episode ended inside it (crash)
        a = e[key]
        return a[warmup:] if len(a) > warmup else a

    vx = np.concatenate([cut(e, "vx") for e in eps])
    u = np.concatenate([cut(e, "u") for e in eps])
    air = np.concatenate([cut(e, "air") for e in eps])
    has_imp = any(len(e["impacts"]) for e in eps)
    imp = np.concatenate([e["impacts"][e["impact_steps"] >= min(warmup, e["steps"] - 1)]
                          for e in eps]) if has_imp else np.zeros(0)
    sim_s = sum(e["sim_seconds"] for e in eps)
    crashes = sum(e["crashed"] for e in eps)
    top10 = 0.0
    if imp.size:
        srt = np.sort(imp)[::-1]
        top10 = float(srt[:max(1, int(np.ceil(0.1 * imp.size)))].mean())
    out = {
        "name": name,
        "mean_speed": float(vx.mean()),
        "speed_rmse": float(np.sqrt(np.mean((vx - v_ref) ** 2))) if v_ref is not None else np.nan,
        "airborne_frac": float(air.mean()),
        "impact_mean": float(imp.mean()) if imp.size else 0.0,
        "impact_top10": top10,
        "n_impacts": int(imp.size),
        "crashes": int(crashes),
        "crashes_per_min": float(crashes / (sim_s / 60.0)) if sim_s > 0 else np.nan,
        "effort_rms": float(np.sqrt(np.mean(u ** 2))),
        "effort_abs": float(np.mean(np.abs(u))),
        "episodes": len(eps),
        "mean_return": float(np.mean([e.get("ret", np.nan) for e in eps])),
        "sim_seconds": float(sim_s),
        "wall_s": wall_s,
        "episode_steps": [e["steps"] for e in eps],
        "per_episode_airborne": [float(cut(e, "air").mean()) for e in eps],
        "per_episode_speed": [float(cut(e, "vx").mean()) for e in eps],
        "warmup_steps": warmup,
        "per_episode_crashed": [bool(e["crashed"]) for e in eps],
    }
    for e in eps:
        if "series" in e:
            out["series"] = e["series"]
            break
    return out


def evaluate(env, policy, n_episodes: int = 5, T: int = 1200, seed0: int = 0,
             v_ref: float | None = None, record_first: bool = True, name: str | None = None,
             warmup: int = 0) -> dict:
    """Run n_episodes seeded episodes and aggregate the metrics into a dict."""
    t0 = time.perf_counter()
    eps = [run_episode(env, policy, T, seed0 + i, record=(record_first and i == 0))
           for i in range(n_episodes)]
    return aggregate(eps, v_ref, name or getattr(policy, "name", "policy"),
                     time.perf_counter() - t0, warmup)


# --- parallel evaluation (one process per episode; factories are cloudpickled) --
def _worker(payload):
    import cloudpickle
    env_factory, policy_factory, T, seed, record = cloudpickle.loads(payload)
    env = env_factory()
    policy = policy_factory()
    return run_episode(env, policy, T, seed, record=record)


def evaluate_parallel(env_factory, policy_factory, n_episodes: int = 5, T: int = 1200,
                      seed0: int = 0, v_ref: float | None = None, procs: int = 4,
                      name: str = "policy", record_first: bool = True, warmup: int = 0) -> dict:
    """Same as ``evaluate`` but each episode runs in its own worker process.
    ``env_factory()`` and ``policy_factory()`` build fresh objects in the worker
    (closures are fine: they are serialised with cloudpickle)."""
    import multiprocessing as mp
    import cloudpickle
    t0 = time.perf_counter()
    payloads = [cloudpickle.dumps((env_factory, policy_factory, T, seed0 + i,
                                   record_first and i == 0)) for i in range(n_episodes)]
    if procs <= 1:
        eps = [_worker(p) for p in payloads]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(procs, n_episodes)) as pool:
            eps = pool.map(_worker, payloads)
    return aggregate(eps, v_ref, name, time.perf_counter() - t0, warmup)


# default column set: (key, header, format)
DEFAULT_COLUMNS = [
    ("name", "controller", "{}"),
    ("mean_speed", "mean speed (m/s)", "{:.3f}"),
    ("speed_rmse", "speed RMSE (m/s)", "{:.3f}"),
    ("airborne_frac", "airborne (%)", "{:.1%}"),
    ("impact_mean", "impact mean (m/s)", "{:.2f}"),
    ("impact_top10", "impact top-10% (m/s)", "{:.2f}"),
    ("crashes", "crashes", "{}"),
    ("crashes_per_min", "crashes/min", "{:.2f}"),
    ("effort_rms", "effort RMS", "{:.3f}"),
    ("wall_s", "wall (s)", "{:.0f}"),
]


def markdown_table(rows: list[dict], columns=None) -> str:
    """rows: list of dicts (e.g. from ``evaluate``); columns: list of
    (key, header, format) triples. Missing keys print as '-'."""
    columns = columns or DEFAULT_COLUMNS
    head = "| " + " | ".join(h for _, h, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for k, _, f in columns:
            v = r.get(k, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("-")
            else:
                try:
                    cells.append(f.format(v))
                except (ValueError, TypeError):
                    cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def strip_series(d: dict) -> dict:
    """Copy of a result dict without the bulky time series (for JSON dumps)."""
    return {k: v for k, v in d.items() if k != "series"}
