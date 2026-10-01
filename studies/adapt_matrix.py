#!/usr/bin/env python3
"""
Stage 1 of the adaptation study: every combination of

    features     hand-chosen | random Fourier
    confounder   without | with the motion-intensity feature
    speed input  'fast': speed at the start of each ~2 s impact sample |
                 'slow': mean speed over the 30 s before it
    probing      none | dither | explore-then-commit | optimism |
                 Thompson sampling | information bonus

for the corrected MPC (learn/adapt/cmpc.py), one learning seed each (a
second for the best combinations afterwards), on
the Scarab task of learn/repro/task.py in its turned-track form (waves
fixed, legs heading 0 and 45 deg: the heading tells the leg, the wave
direction is never given).

Protocol per combination: 12 learning episodes in the target (120 s,
legs alternating, the corrections carried across episodes, probing on),
then the 16 evaluation episodes of every other study here (seeds 0-7 x
both legs), probing off, corrections frozen. Reported: the evaluation
score, and the mean score paid WHILE learning (probing costs included).

Reference rows, same geometry and evaluation:
  A0 hand      the hand-tuned MPC (fixed weights)
  B2 MF-GP     the source-prior GP search over MPC weights, 32 target eps
  C0 prior     the corrected MPC with the source prior only (0 target eps)
  C-big        the corrected MPC fitted on 32 target episodes of rich
               excitation -- the approach's ceiling at this data size

Phases (cached in studies/_cache/adapt/): source big learn eval ref report.
`--smoke`: every phase, tiny, separate cache.

Run:  python -m studies.adapt_matrix [--workers 6] [--phase all|<phase>]
"""
import argparse
import os
import pickle
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                    # noqa: E402

from learn.repro import task                          # noqa: E402

SMOKE = False
KINDS = ("hand", "rff")
INTENS = (False, True)
REGS = ("fast", "slow")
N_LEARN, LEARN_SEEDS = 12, (1,)
N_SOURCE, N_BIG = 16, 32
EXCITE = (-4.0, -2.0, 0.0, 2.0)          # kn offsets for data collection


def P(key):
    small = dict(N_LEARN=2, LEARN_SEEDS=(1,), N_SOURCE=2, N_BIG=2, T=30.0,
                 EVAL_SEEDS=(0,))
    full = dict(N_LEARN=N_LEARN, LEARN_SEEDS=LEARN_SEEDS, N_SOURCE=N_SOURCE,
                N_BIG=N_BIG, T=task.T_EVAL, EVAL_SEEDS=task.EVAL_SEEDS)
    return (small if SMOKE else full)[key]


def cache_path():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache",
                     "adapt")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "smoke.pkl" if SMOKE else "results.pkl")


def feat_sets():
    return [(k, i, r) for k in KINDS for i in INTENS for r in REGS]


def configs():
    from learn.adapt.cmpc import STRATEGIES
    return [(k, i, r, s) for (k, i, r) in feat_sets() for s in STRATEGIES]


def flabel(k, i, r):
    return f"{k}{'+I' if i else ''}/{r}"


def label(k, i, r, s):
    return f"{flabel(k, i, r)}/{s}"


def schedule(i):
    """Excitation for data-collection runs: four 30 s blocks, offsets
    rotated per episode so every speed meets every block position."""
    # rotated by i // 2: the legs alternate with i, and each leg must meet
    # every rotation
    order = [EXCITE[(i // 2 + j) % len(EXCITE)] for j in range(4)]
    return [(30.0 * j, kn) for j, kn in enumerate(order)]


# ------------------------------------------------------------------ jobs
def job_collect(j):
    """A data-collection episode: zero corrections, excitation schedule."""
    from learn.adapt.cmpc import Corrections, run_episode
    corr = Corrections("hand", False)
    m, raw = run_episode(corr, "none", j["world"], j["seed"], j["leg"],
                         j["T"], learn=False, explore=False,
                         du_const=schedule(j["i"]), collect_raw=True,
                         rng_seed=j["seed"])
    return dict(metrics=m, raw=raw)


def job_learn(j):
    """One learning sequence: N episodes, corrections carried across."""
    from learn.adapt.cmpc import Corrections, run_episode
    k, i, r, s = j["cfg"]
    corr = Corrections.from_frozen(j["prior"])
    rows = []
    for e in range(j["n"]):
        seed = 20000 + 1000 * j["lseed"] + e
        m, _ = run_episode(corr, s, "high", seed, e % 2, j["T"], learn=True,
                           explore=True, ep_idx=e, n_learn=j["n"],
                           rng_seed=seed)
        rows.append(dict(ep=e, leg=e % 2, **{q: m[q] for q in (
            "score", "u_along", "acc_cg_p99", "c_speed", "c_impact",
            "c_track", "finite")}))
    return dict(cfg=j["cfg"], lseed=j["lseed"], rows=rows,
                frozen=corr.frozen(), n_imp=corr.imp.n, n_srg=corr.srg.n)


def job_eval(j):
    from learn.adapt.cmpc import Corrections, run_episode
    corr = Corrections.from_frozen(j["frozen"])
    m, _ = run_episode(corr, "none", "high", j["seed"], j["leg"], j["T"],
                       learn=False, explore=False, rng_seed=j["seed"])
    return m


def job_fixed(j):
    return task.run_job(dict(j, geometry="track"))


# ---------------------------------------------------------------- phases
def fit_priors(raw_s, raw_i, base=None, widen=True):
    """Posterior of each feature set on the raw rows, from `base` priors
    (None = broad); returned as target priors (widened) or as fitted
    posteriors (widen=False, for C-big)."""
    from learn.adapt.cmpc import Corrections, prior_from
    out = {}
    for (k, i, r) in feat_sets():
        c = Corrections(k, i, regressor=r,
                        prior=None if base is None else base[(k, i, r)])
        c.add_raw(raw_s, raw_i)
        if widen:
            out[(k, i, r)] = (
                prior_from(c.imp, Corrections.IMP_STD, c.f_imp,
                           Corrections.IMP_S2),
                prior_from(c.srg, Corrections.SRG_STD, c.f_srg,
                           Corrections.SRG_S2))
        else:
            out[(k, i, r)] = ((c.imp.m, c.imp.S, c.imp.sigma2),
                              (c.srg.m, c.srg.S, c.srg.sigma2))
    return out


def phase_source(R, run):
    jobs = [dict(world="low", seed=30000 + n, leg=n % 2, i=n, T=P("T"))
            for n in range(P("N_SOURCE"))]
    res = run(jobs)
    rs = [r for x in res for r in x["raw"][0]]
    ri = [r for x in res for r in x["raw"][1]]
    R["source"] = dict(n_surge=len(rs), n_imp=len(ri),
                       metrics=[x["metrics"] for x in res])
    R["prior"] = fit_priors(rs, ri)
    R["raw_source"] = (rs, ri)
    print(f"    source: {len(ri)} impact rows, {len(rs)} surge rows; mean "
          f"impact sample {np.mean([r[4] for r in ri]):.4f}; speed "
          f"{np.percentile([r[0] for r in ri], [5, 50, 95]) * 25 + 25} kn",
          flush=True)


def phase_big(R, run):
    jobs = [dict(world="high", seed=40000 + n, leg=n % 2, i=n, T=P("T"))
            for n in range(P("N_BIG"))]
    res = run(jobs)
    rs = [r for x in res for r in x["raw"][0]]
    ri = [r for x in res for r in x["raw"][1]]
    R["big"] = fit_priors(rs, ri, base=R["prior"], widen=False)
    R["raw_big"] = (rs, ri)
    R["big_info"] = dict(n_imp=len(ri), mean_imp=float(np.mean(
        [r[4] for r in ri])))
    print(f"    big: {len(ri)} impact rows; mean impact sample "
          f"{R['big_info']['mean_imp']:.4f}", flush=True)


def _frozen(k, i, r, pr):
    return dict(kind=k, intensity=i, regressor=r, imp=pr[0], srg=pr[1])


def _imap(run, jobs, R, store, every=6):
    """Results as they arrive, stored and checkpointed, so a late failure
    does not lose the phase."""
    if task.avail_gb() < 1.5:
        raise MemoryError(f"only {task.avail_gb():.1f} GB of RAM left")
    for n, (idx, res) in enumerate(run.pool.imap_unordered(
            _indexed, [(run.fn, i, j) for i, j in enumerate(jobs)])):
        store(idx, res)
        if (n + 1) % every == 0:
            save(R)


def _indexed(arg):
    fn, i, j = arg
    return i, fn(j)


def save(R):
    p = cache_path()
    pickle.dump(R, open(p + ".tmp", "wb"))
    os.replace(p + ".tmp", p)


def phase_learn(R, run):
    done = R.setdefault("learn", {})
    jobs = [dict(cfg=c, lseed=ls, n=P("N_LEARN"), T=P("T"),
                 prior=_frozen(c[0], c[1], c[2], R["prior"][c[:3]]))
            for c in configs() for ls in P("LEARN_SEEDS")
            if (c, ls) not in done]
    t0 = time.time()

    def store(idx, r):
        done[(r["cfg"], r["lseed"])] = r
    _imap(run, jobs, R, store)
    print(f"    {len(jobs)} learning sequences in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


def phase_eval(R, run):
    ev = R.setdefault("eval", {})
    targets = {}
    for (c, ls), r in R["learn"].items():
        targets[f"{label(*c)}#{ls}"] = r["frozen"]
    for (k, i, r) in feat_sets():
        targets[f"C0 prior {flabel(k, i, r)}"] = _frozen(
            k, i, r, R["prior"][(k, i, r)])
        if "big" in R:
            targets[f"C-big {flabel(k, i, r)}"] = _frozen(
                k, i, r, R["big"][(k, i, r)])
    todo = [n for n in targets if n not in ev]
    jobs, owner = [], []
    for n in todo:
        for sd in P("EVAL_SEEDS"):
            for leg in (0, 1):
                jobs.append(dict(frozen=targets[n], seed=sd, leg=leg,
                                 T=P("T")))
                owner.append(n)
    if jobs:
        t0 = time.time()
        got = {}

        def store(idx, m):
            got[idx] = m
            n = owner[idx]
            mine = [q for q, o in enumerate(owner) if o == n]
            if all(q in got for q in mine):
                ev[n] = [got[q] for q in mine]      # seeds x legs, in order
        _imap(run, jobs, R, store)
        print(f"    evaluated {len(todo)} controllers in "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)


def phase_ref(R, workers):
    from studies import sim2real_jet as s2r
    ev = R.setdefault("eval", {})
    run = task.Runner(workers, fn=job_fixed)
    try:
        if "A0 hand" not in ev:
            ev["A0 hand"] = run([dict(z=s2r.z_hand(), world="high", seed=sd,
                                      head=leg, t_end=P("T"))
                                 for sd in P("EVAL_SEEDS") for leg in (0, 1)])
        if "B2 MF-GP" not in ev:
            import json
            rp = os.path.join(os.path.dirname(cache_path()), "..", "repro",
                              "results.json")
            A1 = json.load(open(rp))["A1"]          # source history, reused
            if SMOKE:
                s2r.TGT_BUDGET = 8
            R["B2"] = s2r.gp_search(run, {"A1": A1}, use_source=True,
                                    tag="B2", seed=22)
            ev["B2 MF-GP"] = run([dict(z=R["B2"]["z"], world="high", seed=sd,
                                       head=leg, t_end=P("T"))
                                  for sd in P("EVAL_SEEDS") for leg in (0, 1)])
    finally:
        run.close()


# ---------------------------------------------------------------- report
def report(R):
    ev = R.get("eval", {})
    if "A0 hand" not in ev:
        print("  no reference yet")
        return
    a = lambda rs, k: np.array([r[k] for r in rs], float)   # noqa: E731
    base = a(ev["A0 hand"], "score")
    rows = []
    for n, rs in ev.items():
        s = a(rs, "score")
        if len(s) != len(base):
            continue
        d = (s - base) / base.mean() * 100
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        head, obl = s[0::2].mean(), s[1::2].mean()
        u = a(rs, "u_along")
        rows.append((s.mean(), n, d.mean(), se, head, obl,
                     np.nanmean(u[0::2]) / 0.514444,
                     np.nanmean(u[1::2]) / 0.514444))
    learn_cost = {}
    for (c, ls), r in R.get("learn", {}).items():
        learn_cost[f"{label(*c)}#{ls}"] = np.mean([q["score"]
                                                   for q in r["rows"]])
    print("\n  EVALUATION, 16 paired target episodes (lower is better); "
          "learning = mean score paid over the learning episodes")
    print(f"    {'controller':<24}{'score':>7}{'vs A0':>8}{'+-se':>6}"
          f"{'head':>7}{'45deg':>7}{'kn head':>8}{'kn 45':>7}{'learning':>9}")
    for sc, n, dm, se, hd, ob, uh, uo in sorted(rows):
        lc = learn_cost.get(n)
        print(f"    {n:<24}{sc:>7.3f}{dm:>+7.0f}%{se:>5.0f}%{hd:>7.3f}"
              f"{ob:>7.3f}{uh:>8.1f}{uo:>7.1f}"
              + (f"{lc:>9.3f}" if lc is not None else ""))
    # combination means over the two learning seeds
    if R.get("learn"):
        print("\n  per combination, mean over learning seeds (score vs A0):")
        from learn.adapt.cmpc import STRATEGIES
        hdr = "".join(f"{s:>9}" for s in STRATEGIES)
        print(f"    {'features':<14}{hdr}")
        for (k, i, r) in feat_sets():
            line = f"    {flabel(k, i, r):<14}"
            for s in STRATEGIES:
                v = [(a(ev[n], 'score') - base).mean() / base.mean() * 100
                     for n in ev if n.startswith(label(k, i, r, s) + "#")
                     and len(ev[n]) == len(base)]
                line += f"{np.mean(v):>+8.0f}%" if v else f"{'':>9}"
            print(line)


# ------------------------------------------------------------------ main
def main(phase="all", workers=6):
    p = cache_path()
    R = pickle.load(open(p, "rb")) if os.path.exists(p) else {}
    order = ("source", "big", "learn", "eval", "ref")
    todo = order if phase == "all" else () if phase == "report" else (phase,)
    t0 = time.time()
    for ph in todo:
        print(f"\n  [{ph}]  {task.avail_gb():.1f} GB free", flush=True)
        if ph == "source" and "prior" in R:
            print("    cached"); continue
        if ph == "big" and "big" in R:
            print("    cached"); continue
        if ph == "ref":
            phase_ref(R, workers)
        else:
            fn = {"source": job_collect, "big": job_collect,
                  "learn": job_learn, "eval": job_eval}[ph]
            run = task.Runner(workers, fn=fn)
            try:
                {"source": phase_source, "big": phase_big,
                 "learn": phase_learn, "eval": phase_eval}[ph](R, run)
            finally:
                run.close()
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
    SMOKE = a.smoke
    main(a.phase, a.workers)
