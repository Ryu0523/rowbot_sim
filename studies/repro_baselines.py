#!/usr/bin/env python3
"""
Five published methods reproduced on this project's sim-to-real task
(learn/repro/): Scarab 195, source = low-fidelity world, target = the full
planing plant, sea state 3, head seas and 45 deg off the bow.

Every method is scored on the SAME 16 held-out target episodes (8 wave
seeds x 2 headings, 120 s) by the operator cost of learn/repro/task.py,
alongside three reference rows run on the same task:

  A0   hand     the MPC's default weights                     0 target eps
  A1   src      CMA-ES on the MPC parameters in the source    0
  B2   MF-GP    the few-shot GP of studies/sim2real_jet.py    32

  PPO-w  PPO sets the MPC weights every step (= Berg (a),     0
         SG-RL's baseline), randomised source
  SG-RL  the same, with MPPI's solver-gradient guidance       0
         (SG-SCA)
  Berg   PPO sets the MPC weights AND its model's mass and    0
         damping every step, randomised source
  RMA    direct control, 1-D CNN estimate of the vessel       0
  Jiang  direct control, GRU estimate of the vessel           0
  in-sim direct control trained in the nominal source         0
  Bi-lvl the in-sim policy after 8 outer steps moving the     32
         source's dynamics and reward

Phases (cached in studies/_cache/repro/): ref sgrl berg rma jiang bilevel
eval report. `--smoke` runs every phase with tiny budgets into a separate
cache, to check the pipeline end to end (~5 min).

Run:  python -m studies.repro_baselines [--phase all|<phase>] [--workers 6]
"""
import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                    # noqa: E402

from learn.repro import task                          # noqa: E402

SMOKE = False
# Direct-control budgets were halved after the first attempt ran at ~300
# steps/s (the main process, not the envs, is the bottleneck): 1M steps is
# five times where a plain direct policy's learning curve flattened.
BUDGET = dict(mpc_steps=300_000, direct_steps=1_000_000, adapt_iters=20,
              bl_pretrain=750_000, bl_outer=8, bl_inner=10, cma_gen=10)
SMOKE_BUDGET = dict(mpc_steps=6_144, direct_steps=12_288, adapt_iters=2,
                    bl_pretrain=12_288, bl_outer=1, bl_inner=1, cma_gen=1)


def B(k):
    return (SMOKE_BUDGET if SMOKE else BUDGET)[k]


def cache_path():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache",
                     "repro")
    os.makedirs(d, exist_ok=True)
    # REPRO_RESULTS lets two phases run side by side in separate files,
    # merged before the evaluation
    name = os.environ.get("REPRO_RESULTS", "results.json")
    return os.path.join(d, "results_smoke.json" if SMOKE else name)


def eval_seeds():
    return task.EVAL_SEEDS[:1] if SMOKE else task.EVAL_SEEDS


# ------------------------------------------------------------------ phases
def phase_ref(R, workers):
    """A1 and B2 through studies/sim2real_jet's own CMA-ES and GP search,
    run on this task's episodes (task.run_job)."""
    from studies import sim2real_jet as s2r
    run = task.Runner(workers)
    try:
        if "A1" not in R:
            print("  A1: CMA-ES in the source world", flush=True)
            R["A1"] = dict(zip(("z", "hist"), s2r.tune_cma(
                run, s2r.z_hand(), 0.35, "low", n_gen=B("cma_gen"), pop=8,
                seed0=11, per_seeds=2, tag="A1")))
            save(R)
        if "B2" not in R:
            print("  B2: multi-fidelity GP, 32 target episodes", flush=True)
            if SMOKE:
                s2r.TGT_BUDGET = 8
            R["B2"] = s2r.gp_search(run, R, use_source=True, tag="B2")
            save(R)
    finally:
        run.close()


def phase_sgrl(R, workers):
    from learn.repro import sgrl
    for name, mode in (("ppo_weights", None), ("sgrl_sca", "sca")):
        if name in R.get("trained", {}):
            continue
        print(f"  {name}: PPO on the MPC weights, {B('mpc_steps')} steps "
              f"(guidance: {mode})", flush=True)
        t0 = time.time()
        sgrl.train(name, sg_mode=mode, steps=B("mpc_steps"), n_envs=workers)
        R.setdefault("trained", {})[name] = dict(
            steps=B("mpc_steps"), minutes=(time.time() - t0) / 60)
        save(R)


def phase_berg(R, workers):
    from learn.repro import berg
    if "berg_twin" in R.get("trained", {}):
        return
    print(f"  berg_twin: PPO on MPC weights + model, {B('mpc_steps')} steps",
          flush=True)
    t0 = time.time()
    berg.train(steps=B("mpc_steps"), n_envs=workers)
    R.setdefault("trained", {})["berg_twin"] = dict(
        steps=B("mpc_steps"), minutes=(time.time() - t0) / 60)
    save(R)


def phase_latent(R, workers, variant):
    from learn.repro import latent
    tr = R.setdefault("trained", {})
    if variant not in tr:
        print(f"  {variant}: phase 1, PPO with the vessel's parameters, "
              f"{B('direct_steps')} steps", flush=True)
        t0 = time.time()
        model = latent.train_teacher(variant, steps=B("direct_steps"),
                                     n_envs=workers)
        print(f"  {variant}: phase 2, adaptation module", flush=True)
        latent.train_adapter(variant, model, iters=B("adapt_iters"),
                             n_envs=workers)
        tr[variant] = dict(steps=B("direct_steps"),
                           minutes=(time.time() - t0) / 60)
        save(R)


def phase_bilevel(R, workers):
    from learn.repro import bilevel
    if "bilevel" in R.get("trained", {}):
        return
    t0 = time.time()
    bilevel.run(pretrain=B("bl_pretrain"), outer=B("bl_outer"),
                inner_updates=B("bl_inner"), n_envs=workers,
                workers=workers)
    R.setdefault("trained", {})["bilevel"] = dict(
        steps=B("bl_pretrain"), minutes=(time.time() - t0) / 60)
    save(R)


# -------------------------------------------------------------- evaluation
def eval_job(job):
    from learn.repro.envs import rollout
    return rollout(job["kind"], job["policy"], job["world"], job["seed"],
                   job["head"], **job.get("kw", {}))


def _jobs(kind, policy, world, kw=None):
    return [dict(kind=kind, policy=policy, world=world, seed=int(s), head=h,
                 kw=kw or {}) for s in eval_seeds()
            for h in range(len(task.HEADINGS))]


def phase_eval(R, workers):
    from learn.repro import bilevel, latent, sgrl
    from learn.repro.ppo_sg import NumpyMLP
    from studies.sim2real_jet import z_hand
    ev = R.setdefault("eval", {})
    todo = [("A0 hand", "fixed", z_hand(), {})]
    for k, nm in (("A1", "A1 src"), ("B2", "B2 MF-GP")):
        if k in R:
            todo.append((nm, "fixed", R[k]["z"], {}))
    # where PPO-w, SG-RL and Berg start (zero output = the middle of the
    # parameter box): what their training added beyond it
    todo.append(("RL start", "fixed", [0.0] * 8, {}))
    for name, nm, kw in (("ppo_weights", "PPO-w", {}),
                         ("sgrl_sca", "SG-RL", {}),
                         ("berg_twin", "Berg", dict(twin=True))):
        m = sgrl.load(name)
        if m is not None:
            todo.append((nm, "mpc", NumpyMLP(m.policy), kw))
    bl = bilevel.load()
    for tag, nm in (("insim", "in-sim"), ("bilevel", "Bi-level")):
        if tag in bl and (tag != "bilevel" or "bilevel" in R.get("trained",
                                                                 {})):
            todo.append((nm, "direct", NumpyMLP(bl[tag].policy), {}))
    pool = task.Runner(workers, fn=eval_job)
    try:
        for nm, kind, pol, kw in todo:
            if nm in ev:
                continue
            print(f"  evaluating {nm}", flush=True)
            ev[nm] = dict(high=pool(_jobs(kind, pol, "high", kw)),
                          low=pool(_jobs(kind, pol, "low", kw)))
            save(R)
    finally:
        pool.close()
    # the latent policies carry a torch history module: run them here
    from learn.repro.envs import rollout
    for variant, nm in (("rma", "RMA"), ("jiang", "Jiang")):
        if variant not in R.get("trained", {}):
            continue
        model, net = latent.load(variant)
        # the ablation both papers report (Jiang: "student, mean z"): the
        # same policy with the latent of the middle of the parameter box
        # instead of the history module's estimate
        z_mid = latent._actor(model)[0](np.zeros(latent.n_priv(),
                                                 np.float32)).numpy()
        for tag, pol in ((nm, latent.LatentPolicy(variant, model, net)),
                         (nm + " fixed-z", latent.LatentPolicy(
                             variant, model, None, z_fixed=z_mid))):
            if tag in ev:
                continue
            print(f"  evaluating {tag}", flush=True)
            ev[tag] = {w: [rollout("direct", pol, w, int(s), h)
                           for s in eval_seeds()
                           for h in range(len(task.HEADINGS))]
                       for w in ("high", "low")}
            save(R)


# ------------------------------------------------------------------ report
BUDGET_TGT = {"A0 hand": 0, "A1 src": 0, "B2 MF-GP": 32, "RL start": 0, "PPO-w": 0,
              "SG-RL": 0, "Berg": 0, "RMA": 0, "Jiang": 0, "in-sim": 0,
              "RMA fixed-z": 0, "Jiang fixed-z": 0,
              "Bi-level": 32}


def report(R):
    ev = R.get("eval", {})
    if not ev:
        print("  nothing evaluated yet")
        return
    arr = lambda rs, k: np.array([r[k] for r in rs], float)   # noqa: E731
    base = arr(ev["A0 hand"]["high"], "score") if "A0 hand" in ev else None
    print("\n  TARGET (full plant), 16 paired held-out episodes; lower "
          "score is better")
    print(f"    {'method':<10}{'tgt eps':>8}{'score':>8}{'+-se':>7}"
          f"{'vs A0':>8}{'+-se':>6}{'source':>8}{'kn':>6}{'a_cg99':>8}"
          f"{'speed':>7}{'impact':>7}{'track':>7}")
    for nm, d in ev.items():
        s = arr(d["high"], "score")
        line = (f"    {nm:<10}{BUDGET_TGT.get(nm, 0):>8}{s.mean():>8.3f}"
                f"{s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0:>7.3f}")
        if base is not None and len(base) == len(s):
            dd = (s - base) / base.mean() * 100
            se = dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else 0
            line += f"{dd.mean():>+7.0f}%{se:>5.0f}%"
        else:
            line += " " * 14
        line += (f"{arr(d['low'], 'score').mean():>8.3f}"
                 f"{np.nanmean(arr(d['high'], 'u_along')) / 0.514444:>6.1f}"
                 f"{np.nanmean(arr(d['high'], 'acc_cg_p99')):>8.2f}"
                 f"{np.nanmean(arr(d['high'], 'c_speed')):>7.3f}"
                 f"{np.nanmean(arr(d['high'], 'c_impact')):>7.3f}"
                 f"{np.nanmean(arr(d['high'], 'c_track')):>7.3f}")
        print(line)
    names = list(ev)
    if len(names) > 2:
        from scipy.stats import spearmanr
        lo = [arr(ev[n]["low"], "score").mean() for n in names]
        hi = [arr(ev[n]["high"], "score").mean() for n in names]
        print(f"\n    source vs target ranking of these {len(names)} "
              f"controllers: Spearman {spearmanr(lo, hi).correlation:+.2f}")
    if "trained" in R:
        print("\n    training cost: " + ", ".join(
            f"{k} {v['minutes']:.0f} min" for k, v in R["trained"].items()))


# -------------------------------------------------------------------- main
def save(R):
    with open(cache_path(), "w") as f:
        json.dump(R, f)


def main(phase="all", workers=6):
    import torch
    torch.set_num_threads(1)
    p = cache_path()
    R = json.load(open(p)) if os.path.exists(p) else {}
    order = ("ref", "sgrl", "berg", "rma", "jiang", "bilevel", "eval")
    todo = order if phase == "all" else () if phase == "report" else (phase,)
    t0 = time.time()
    fns = {"ref": phase_ref, "sgrl": phase_sgrl, "berg": phase_berg,
           "rma": lambda R, w: phase_latent(R, w, "rma"),
           "jiang": lambda R, w: phase_latent(R, w, "jiang"),
           "bilevel": phase_bilevel, "eval": phase_eval}
    for ph in todo:
        print(f"\n  [{ph}]  {task.avail_gb():.1f} GB free", flush=True)
        fns[ph](R, workers)
        save(R)
        print(f"  [{ph}] done, {(time.time() - t0) / 60:.1f} min so far",
              flush=True)
    report(R)
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        SMOKE = True
        # the smoke run must not touch the real models: separate cache dirs
        import learn.repro.ppo_sg as _ps
        _orig = _ps.out_dir
        _ps.out_dir = lambda name: _orig("smoke_" + name)
        for _m in ("sgrl", "latent", "bilevel"):
            mod = __import__(f"learn.repro.{_m}", fromlist=["out_dir"])
            mod.out_dir = _ps.out_dir
    main(a.phase, a.workers)
    sys.stdout.flush()
