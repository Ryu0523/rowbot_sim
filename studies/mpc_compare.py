#!/usr/bin/env python3
"""
The learned one-step error model inside the MPC, compared in closed loop
on the target (design: the scratchpad's DESIGN_MPC.md, brief BRIEF_MPC.md;
controller learn/meta/mpc_learned.py). Task: learn/repro/task.py in the
turned-track geometry of DEFECTS J (waves from pi, legs 0 and 45 deg),
high-fidelity world, 120 s; the 16 evaluation missions (task.EVAL_SEEDS x
both legs) are the same for every variant (same sea, same knot-noise and
base-noise streams per row), so every difference is paired.

Variants (rows of the report):
  hand     control/mpc.py as the repro baselines run it (z_hand)
  c0       learn/adapt's corrected MPC with the source prior rff/slow,
           learning off (DEFECTS J: the right reference for learned rows)
  m0       the new controller with e = 0 (Model0T rollouts only)
  prior    the new controller with model3_cov.pt, zero-shot (in-context
           from the episode's own history), spread s = 1
  prior_s  the same net with the spread calibrated on the online pool
           (separates spread scaling from head adaptation)
  online   model3_cov.pt with its flow head adapted after every pair of
           learning episodes, spread calibrated each time; frozen for the
           evaluation
  oracle   model3_oracle.pt (every weight fine-tuned on 96 Cbh target
           episodes), spread calibrated on its 32 held-out Cbh episodes
hand and c0 see the measured wave under the hull at the first horizon
step (PreviewProvider t_preview = 0); the new controller's rows get no
wave input (D6).

Phases, in order (cached in studies/_cache/mpc_learned/, results.pkl;
--smoke: mpc_learned_smoke/, tiny):
  link         impact link (A, K) from 16 target episodes on seeds 100-103
               x legs: the hand MPC (~21 kn) and C0 with the planning-speed
               offsets -4/-2/0/+2 kn in 30 s blocks (adapt_matrix.schedule,
               ~19-27 kn), windows indexed by their START speed; a speed-
               dependent threshold if one threshold does not track the
               growth across speed bins (mpc_learned.fit_impact_link)
  link_src     the same fit in the source world (sensitivity only, never
               used by a controller; not in 'all')
  hand, c0     the reference rows (CPU, --workers processes)
  calib        the oracle's spread on its held-out Cbh episodes (and the
               prior's coverage there, for reference)
  hcheck_pre   rolled vs true impact exceedance / link cost / speed cost
               per horizon step on the Cb blocks for prior and oracle (m0
               reported): the impact horizon j_imp the learning episodes
               plan with (mpc_learned.impact_horizon); if a model already
               fails below the floor step the run STOPS here, before hours
               of GPU learning (--hcheck-continue: go on, rows marked not
               interpretable)
  learn        online learning: 6 pairs (seeds 21000 + e, leg e % 2, J's
               learning missions), pair 0 run by the prior (shared with
               prior_learn), head-only adaptation after every pair
  prior_learn  the prior on the remaining learning pairs, in the SAME pairs
               (the online - prior curve per episode)
  calib_prior  prior_s's spread on all online-pool episodes
  hcheck       the same check adding prior_s and online; j_imp re-decided
               on every available learned net (identical for every
               new-controller row, m0 included); the same stop rule
  m0, prior, prior_s, oracle, online   evaluations (GPU rows in lockstep
               groups of 8 = both legs of 4 seeds; m0 all 16 on the CPU)
  report

Resolved against the design (review of 2026-09-30), in short: the link is
fitted across the speed range (not on hand runs alone) and checked per
speed bin; every net gets its spread from the same rule (cost channels
surge + heave rate); a horizon check on unconfounded Cb blocks decides an
impact horizon before the learning episodes (prior, oracle) and again
before the evaluations (all learned nets), and stops the run when a cut
cannot help; the impact threshold reads the measured speed, never a rolled
one; adaptation after every PAIR of
episodes (not every episode, as the brief asked: one episode per call runs
at the launch-bound floor, 2.0-2.3 s per step against 2.9 s for a pair, so
per-episode adaptation would add ~1.5 h to the learning phase and ~2 h to
the prior's matched runs); learning seeds 21000 + e (J's); from 4 pool
episodes on, whole episodes (pair 0, one per leg) are held out for early
stopping and the spread, scored from position 40; base noise from one
generator per row seeded (seed, leg, 7); the prior's first learning pair is
online's first pair.

Data budget: every new-controller row (m0 and the zero-shot prior
included) uses the 16 target link episodes through the impact link (a
sensor-resolution conversion, part of the task definition) AND the Cb
target blocks of the horizon check through j_imp, which is decided with
the target-trained oracle and online nets as well; the online row adds its
12 learning episodes. 'prior (zero-shot)' is zero-shot in its error model
only.

Compute: every row's s/step times the controller call only (hand and c0:
mpc_command; the new controller: the lockstep call / rows), never the
plant.

    python -m studies.mpc_compare [--phases all|p1,p2] [--variants ...]
        [--missions 0-7] [--episodes 12] [--workers 3] [--smoke]
        [--hcheck-continue] [--redo variant,...]
"""
import argparse
import os
import pickle
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                    # noqa: E402

from learn.repro import task                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
META3 = os.path.join(HERE, "_cache", "meta3")
SMOKE = False
LEARN_SEED0 = 21000            # J's learning missions: 20000 + 1000 x lseed 1
LINK_SEEDS = (100, 101, 102, 103)
C0_KEY = ("rff", False, "slow")
PHASES = ("link", "hand", "c0", "calib", "hcheck_pre", "learn",
          "prior_learn", "calib_prior", "hcheck", "m0", "prior", "prior_s",
          "oracle", "online", "report")
VARIANTS = ("hand", "c0", "m0", "prior", "prior_s", "online", "oracle")
GPU_VARIANTS = ("prior", "prior_s", "online", "oracle")
# the nets whose horizon check decides j_imp (m0 is checked and reported:
# its e = 0 model is the baseline, not a candidate error model)
HC_DECIDE = GPU_VARIANTS
OPT = dict(missions=None, episodes=None, workers=3, gpu_frac=0.35,
           hcheck_continue=False)


def P(key):
    small = dict(T=20.0, EVAL_SEEDS=(0,), N_LEARN=2, LINK_SEEDS=(100,),
                 GROUP=2, ADAPT_STEPS=50, ADAPT_CHECK=25, HC_EPS=4,
                 CAL_EPS=4, N_MIN=3)
    full = dict(T=task.T_EVAL, EVAL_SEEDS=task.EVAL_SEEDS, N_LEARN=12,
                LINK_SEEDS=LINK_SEEDS, GROUP=8, ADAPT_STEPS=4000,
                ADAPT_CHECK=250, HC_EPS=None, CAL_EPS=None, N_MIN=15)
    v = (small if SMOKE else full)[key]
    if key == "EVAL_SEEDS" and OPT["missions"] is not None:
        v = OPT["missions"]
    if key == "N_LEARN" and OPT["episodes"] is not None:
        v = OPT["episodes"]
    return v


def cache():
    d = os.path.join(HERE, "_cache",
                     "mpc_learned_smoke" if SMOKE else "mpc_learned")
    os.makedirs(d, exist_ok=True)
    return d


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(cache(), "run.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_R():
    p = os.path.join(cache(), "results.pkl")
    return pickle.load(open(p, "rb")) if os.path.exists(p) else {}


def save(R):
    p = os.path.join(cache(), "results.pkl")
    pickle.dump(R, open(p + ".tmp", "wb"))
    os.replace(p + ".tmp", p)


# ------------------------------------------------------------ CPU jobs
def _row(j, m, lg, extra=None):
    from learn.meta.mpc_learned import KN
    met = m.metrics() if hasattr(m, "metrics") else m
    r = dict(seed=j["seed"], leg=j["leg"], **met, log=lg.arrays(),
             **(extra or {}))
    r["kn"] = r["u_along"] / KN if np.isfinite(r["u_along"]) else np.nan
    return r


def job_hand(j):
    """The hand MPC (task.run_job's episode, geometry 'track') with a
    per-step log; world j['world'] (default the target)."""
    from learn.meta.mpc_learned import StepLog
    from studies.sim2real_jet import theta, z_hand
    w, floor = theta(z_hand())
    m = task.Mission(j.get("world", "high"), j["seed"], j["leg"],
                     t_end=j["T"], weights=w, floor=floor,
                     track=task.LEGS[j["leg"]])
    lg = StepLog()
    lg(m)
    t_ctl = 0.0
    while not m.done():
        t0 = time.perf_counter()
        cmd = m.mpc_command()
        t_ctl += time.perf_counter() - t0
        m.advance(*cmd)
        if m.finite:
            lg(m)
    return _row(j, m, lg, dict(t_step_mean=t_ctl / max(m.k, 1)))


class _TimedLog:
    """run_episode's on_step for C0: the StepLog, and on its first call
    (the state at step 0) a timer around the mission's mpc_command, so C0's
    s/step is the controller alone as hand's and the new controller's are
    (not the plant's 12 substeps, the Mission set-up or the corrections'
    bookkeeping). The wrapper returns mpc_command's result unchanged."""

    def __init__(self, lg):
        self.lg, self.t, self.n = lg, 0.0, 0

    def __call__(self, m):
        if not self.lg.u:
            inner = m.mpc_command

            def timed():
                t0 = time.perf_counter()
                out = inner()
                self.t += time.perf_counter() - t0
                self.n += 1
                return out
            m.mpc_command = timed
        self.lg(m)


def job_c0(j):
    """C0: cmpc.run_episode with the frozen source prior, learning and
    probing off (adapt_matrix.job_eval), optionally a planning-speed
    schedule j['du'] (link data), with a per-step log; t_step_mean = the
    controller call's time per step (_TimedLog)."""
    from learn.adapt.cmpc import Corrections, run_episode
    from learn.meta.mpc_learned import StepLog
    corr = Corrections.from_frozen(j["frozen"])
    lg = StepLog()
    tl = _TimedLog(lg)
    met, _ = run_episode(corr, "none", j.get("world", "high"), j["seed"],
                         j["leg"], j["T"], learn=False, explore=False,
                         rng_seed=j["seed"], du_const=j.get("du"),
                         on_step=tl)
    return _row(j, met, lg, dict(t_step_mean=tl.t / max(tl.n, 1),
                                 du=j.get("du")))


def run_cpu(fn, jobs):
    """Jobs in this process (workers <= 1, smoke) or a task.Runner pool."""
    if OPT["workers"] <= 1 or len(jobs) == 1:
        from learn.repro.task import ctx
        ctx()
        return [fn(j) for j in jobs]
    run = task.Runner(min(OPT["workers"], len(jobs)), fn=fn)
    try:
        return run(jobs)
    finally:
        run.close()


def c0_frozen():
    from studies.adapt_matrix import _frozen
    A = pickle.load(open(os.path.join(HERE, "_cache", "adapt",
                                      "results.pkl"), "rb"))
    return _frozen(*C0_KEY, A["prior"][C0_KEY])


def eval_jobs():
    return [dict(seed=int(s), leg=leg) for s in P("EVAL_SEEDS")
            for leg in (0, 1)]


def learn_pairs():
    n = P("N_LEARN")
    assert n % 2 == 0, "learning episodes come in pairs (one per leg)"
    return [[dict(seed=LEARN_SEED0 + e, leg=e % 2) for e in (2 * i, 2 * i + 1)]
            for i in range(n // 2)]


# ------------------------------------------------------------ phases
def phase_link(R, world="high", key="link"):
    from learn.meta.mpc_learned import fit_impact_link
    from studies.adapt_matrix import schedule
    fz = c0_frozen()
    seeds = P("LINK_SEEDS")
    jobs_h = [dict(seed=s, leg=leg, T=P("T"), world=world) for s in seeds
              for leg in (0, 1)]
    jobs_c = [dict(seed=s, leg=leg, T=P("T"), world=world, frozen=fz,
                   du=schedule(i)) for i, (s, leg) in enumerate(
                       [(s, leg) for s in seeds for leg in (0, 1)])]
    t0 = time.time()
    rows = run_cpu(job_hand, jobs_h) + run_cpu(job_c0, jobs_c)
    recs = [dict(r["log"], leg=r["leg"]) for r in rows if r["finite"]]
    for r, who in zip(rows, ["hand"] * len(jobs_h) + ["c0"] * len(jobs_c)):
        log(f"    {key} {who} seed {r['seed']} leg {r['leg']}: score "
            f"{r['score']:.3f}, {r['kn']:.1f} kn, impact {r['c_impact']:.3f}"
            f", speed range {np.min(r['log']['u']) / 0.514444:.1f}-"
            f"{np.max(r['log']['u']) / 0.514444:.1f} kn")
    lk, info = fit_impact_link(recs, n_min=P("N_MIN"), log=log)
    R[key] = dict(link=lk.to_dict(), info=info, world=world,
                  n_eps=len(recs), rows=[{k: v for k, v in r.items()
                                          if k != "log"} for r in rows])
    import json
    json.dump(dict(link=lk.to_dict(), chosen=info["chosen"], world=world),
              open(os.path.join(cache(), f"{key}.json"), "w"), indent=1)
    log(f"  [{key}] {len(rows)} episodes in {(time.time() - t0) / 60:.1f} "
        "min")


def phase_ref(R, name):
    ev = R.setdefault("eval", {}).setdefault(name, {})
    jobs = [dict(j, T=P("T")) for j in eval_jobs()
            if (j["seed"], j["leg"]) not in ev]
    if not jobs:
        return
    if name == "c0":
        fz = c0_frozen()
        jobs = [dict(j, frozen=fz) for j in jobs]
    t0 = time.time()
    rows = run_cpu(job_hand if name == "hand" else job_c0, jobs)
    for r in rows:
        r.pop("du", None)
        ev[(r["seed"], r["leg"])] = r
        _log_row(name, r)
    log(f"  [{name}] {len(rows)} missions in {(time.time() - t0) / 60:.1f} "
        "min")


def _log_row(name, r):
    r.setdefault("kn", r["u_along"] / 0.514444)
    log(f"    {name} seed {r['seed']} leg {r['leg']}: score {r['score']:.3f}"
        f" (speed {r['c_speed']:.3f}, impact {r['c_impact']:.3f}, track "
        f"{r['c_track']:.3f}), {r['kn']:.1f} kn, p99 {r['acc_cg_p99']:.2f} g"
        + (f", {r['t_step_mean']:.2f} s per episode-step" if 't_step_mean'
           in r else "")
        + (f", non-finite samples {r['n_bad']} ({_nf(r):.2%}), capped "
           f"{r.get('n_clip', 0)}" if r.get("n_samp") else ""))


def _nf(r):
    """Share of the planner's rolled samples whose cost was non-finite."""
    return r["n_bad"] / max(r["n_samp"], 1)


# ------------------------------------------------------------ GPU context
_G = {}


def gpu():
    """Device, checkpoints and env (once per process)."""
    if not _G:
        import torch

        from learn.meta import model3 as M
        from learn.meta import mpc_learned as ML
        dev = M.device(OPT["gpu_frac"])
        ck = {n: torch.load(os.path.join(META3, f"model3{t}.pt"),
                            weights_only=False)
              for n, t in (("cov", "_cov"), ("oracle", "_oracle"))}
        _G.update(dev=dev, ck=ck, env=ML.env(), torch=torch, M=M, ML=ML)
    return _G


def link_of(R, j_imp):
    """The fitted impact link with impact horizon j_imp (explicit: the
    learning episodes and the evaluation may have different ones, and each
    row records its own)."""
    from learn.meta.mpc_learned import ImpactLink
    if "link" not in R:
        raise SystemExit("run the link phase first")
    d = dict(R["link"]["link"])
    d["j_imp"] = int(j_imp)
    return ImpactLink.from_dict(d)


def need_hcheck(R, what):
    if "j_imp" not in R.get("hcheck", {}):
        raise SystemExit(f"run the horizon check first (hcheck_pre / "
                         f"hcheck): its impact horizon is part of {what}")
    return R["hcheck"]["j_imp"]


def net_of(R, name):
    """(net, spread s, stats) of a GPU variant."""
    g = gpu()
    ML = g["ML"]
    if name in ("prior", "prior_s"):
        s = 1.0 if name == "prior" else R["calib"]["prior_s"]["s"]
        return ML.load_net(g["ck"]["cov"], g["dev"]), s, \
            g["ck"]["cov"]["stats"]
    if name == "oracle":
        return ML.load_net(g["ck"]["oracle"], g["dev"]), \
            R["calib"]["oracle"]["s"], g["ck"]["oracle"]["stats"]
    if name == "online":
        n = len(learn_pairs())
        hd = g["torch"].load(os.path.join(cache(), f"online_head_{n}.pt"),
                             weights_only=False)
        return ML.load_net(g["ck"]["cov"], g["dev"], head=hd["head"]), \
            hd["s"], g["ck"]["cov"]["stats"]
    raise KeyError(name)


def phase_calib(R):
    g = gpu()
    M, ML = g["M"], g["ML"]
    cal = R.setdefault("calib", {})
    if "oracle" in cal:
        return
    ck = g["ck"]["oracle"]
    D = M.Data3(META3, "Cbh", g["dev"], stats=ck["stats"])
    eps = list(ck["holdout"])
    if P("CAL_EPS"):
        eps = eps[:P("CAL_EPS")]
    t0 = time.time()
    log(f"  oracle spread on {len(eps)} held-out Cbh episodes:")
    s, tab = ML.calibrate_spread(ML.load_net(ck, g["dev"]), D, eps, 40,
                                 log=log)
    log("  prior (model3_cov) on the same episodes, for reference:")
    sp, tabp = ML.calibrate_spread(ML.load_net(g["ck"]["cov"], g["dev"]), D,
                                   eps, 40, log=log)
    cal["oracle"] = dict(s=s, table=tab, eps=eps, prior_s_cbh=sp,
                         prior_table_cbh=tabp)
    log(f"  [calib] {time.time() - t0:.0f} s")


def _pool(R, n_pairs):
    g = gpu()
    eps = [e for i in range(n_pairs)
           for e in R["learn"]["pairs"][i]["episodes"]]
    return g["ML"].Pool3(eps, g["ck"]["cov"]["stats"], g["dev"])


def phase_learn(R):
    g = gpu()
    ML, torch = g["ML"], g["torch"]
    j_now = need_hcheck(R, "the learning episodes' cost")
    L = R.setdefault("learn", dict(pairs={}, adapt={}))
    # fixed at the first pair (hcheck_pre's value), kept across resumes so
    # every learning episode plans with the same objective
    L.setdefault("j_imp", j_now)
    link = link_of(R, L["j_imp"])
    log(f"  learning episodes plan with impact horizon j_imp = {L['j_imp']}")
    ck = g["ck"]["cov"]
    pairs = learn_pairs()
    for i, jobs in enumerate(pairs):
        if i not in L["pairs"]:
            if i == 0:
                net, s, who = ML.load_net(ck, g["dev"]), 1.0, "prior"
            else:
                hd = torch.load(os.path.join(cache(), f"online_head_{i}.pt"),
                                weights_only=False)
                net, s = ML.load_net(ck, g["dev"], head=hd["head"]), hd["s"]
                who = f"online head {i}"
            t0 = time.time()
            rows = ML.run_group([dict(j) for j in jobs], g["env"], link,
                                net=net, stats=ck["stats"], spread=s,
                                T=P("T"), collect=True, log=log,
                                tag=f"learn pair {i}")
            eps = [r.pop("episode") for r in rows]
            L["pairs"][i] = dict(rows=rows, episodes=eps, who=who, s=s)
            save(R)
            for r in rows:
                _log_row(f"learn pair {i} ({who}, s {s})", r)
            log(f"  learn pair {i}: {(time.time() - t0) / 60:.1f} min")
            del net
            torch.cuda.empty_cache()
        if (i + 1) not in L["adapt"]:
            pool = _pool(R, i + 1)
            hold = None if pool.n < 4 else [0, 1]
            net_a, info = ML.adapt_head(ck, pool, holdout=hold,
                                        steps=P("ADAPT_STEPS"),
                                        check=P("ADAPT_CHECK"), log=log)
            head = info.pop("head")
            torch.save(dict(head=head, s=info["s"], info=info),
                       os.path.join(cache(), f"online_head_{i + 1}.pt"))
            L["adapt"][i + 1] = info
            save(R)
            del net_a, pool
            torch.cuda.empty_cache()
    pool = _pool(R, len(pairs))
    pool.save(os.path.join(cache(), "online_eps.npz"))


def phase_prior_learn(R):
    g = gpu()
    ML, torch = g["ML"], g["torch"]
    if 0 not in R.get("learn", {}).get("pairs", {}):
        raise SystemExit("run the learn phase first (pair 0 is shared)")
    PL = R.setdefault("prior_learn", {})
    PL[0] = R["learn"]["pairs"][0]["rows"]
    link = link_of(R, R["learn"]["j_imp"])           # the online side's
    ck = g["ck"]["cov"]
    for i, jobs in enumerate(learn_pairs()):
        if i in PL:
            continue
        net = ML.load_net(ck, g["dev"])
        rows = ML.run_group([dict(j) for j in jobs], g["env"], link, net=net,
                            stats=ck["stats"], spread=1.0, T=P("T"), log=log,
                            tag=f"prior pair {i}")
        PL[i] = rows
        save(R)
        for r in rows:
            _log_row(f"prior learning pair {i}", r)
        del net
        torch.cuda.empty_cache()


def phase_calib_prior(R):
    g = gpu()
    ML = g["ML"]
    cal = R.setdefault("calib", {})
    if "prior_s" in cal:
        return
    pool = _pool(R, len(learn_pairs()))
    log(f"  prior_s spread on the online pool ({pool.n} episodes, from "
        "position 40):")
    s, tab = ML.calibrate_spread(ML.load_net(g["ck"]["cov"], g["dev"]), pool,
                                 list(range(pool.n)), 40, log=log)
    cal["prior_s"] = dict(s=s, table=tab, n_eps=pool.n)


def _f32(v):
    """Float arrays as float32 for the cache, integer ones unchanged."""
    if isinstance(v, tuple):
        return tuple(_f32(x) for x in v)
    a = np.asarray(v)
    return a.astype(np.float32) if a.dtype.kind == "f" else a


def phase_hcheck(R, pre=False):
    """The horizon check on the Cb blocks and the one impact horizon j_imp.
    pre (hcheck_pre, before learning): m0, prior, oracle; else also
    prior_s and online. Per-model results are cached with what they were
    computed with (spread s, online head index, the link); a model that is
    new or changed is (re)computed and j_imp re-decided on ALL available
    deciding nets, so a later run with more variants never reuses a j_imp
    decided without them. A deciding net that fails below the floor step
    stops the run (SystemExit, results saved) unless --hcheck-continue."""
    g = gpu()
    M, ML = g["M"], g["ML"]
    link = link_of(R, ML.H)                 # the check prices every step
    hc = R.setdefault("hcheck", {})
    if hc.get("link") != R["link"]["link"]:
        hc.clear()
        hc.update(link=dict(R["link"]["link"]), sig={}, arrays={})
    names = ("m0", "prior", "oracle") if pre else ("m0",) + HC_DECIDE
    want = {}
    for name in names:
        if name == "m0":
            want[name] = (None, 1.0, "m0")
            continue
        try:
            net, s, _ = net_of(R, name)
        except (KeyError, FileNotFoundError):
            log(f"    horizon check: {name} not available, skipped")
            continue
        tag = len(learn_pairs()) if name == "online" else 0
        want[name] = (net, s, (float(s), tag))
    todo = {n: (w[0], w[1]) for n, w in want.items()
            if hc["sig"].get(n) != w[2]}
    deciding = [n for n in want if n in HC_DECIDE]
    if todo:
        D = M.Data3(META3, "Cb", g["dev"], stats=g["ck"]["cov"]["stats"])
        u_ref = float(task.ctx()["p"]["u_design"])
        out = ML.horizon_check(todo, D, g["env"], link, u_ref,
                               max_eps=P("HC_EPS"), log=log)
        hc["arrays"].update({k: _f32(v) for k, v in out.items()})
        hc["sig"].update({n: want[n][2] for n in todo})
        hc["n_blocks"] = int(len(out["ep"]))
        hc["n_eps"] = int(len(np.unique(out["ep"])))
        del D
    if todo or hc.get("decided_on") != sorted(deciding):
        arr = hc["arrays"]
        j_imp, tab, early = ML.impact_horizon(arr, list(want),
                                              decide=deciding)
        log(ML.format_horizon(arr, tab, list(want), link, g["env"]))
        hc.update(j_imp=j_imp, tab=tab, early=early,
                  decided_on=sorted(deciding),
                  models={n: w[1] for n, w in want.items()})
        save(R)
    log(f"  impact horizon for every new-controller row: j_imp = "
        f"{hc['j_imp']} (decided on {hc['decided_on']}, {hc['n_blocks']} "
        f"Cb blocks of {hc['n_eps']} episodes; first bad step per model: "
        + ", ".join(f"{n} {t['first_bad']}" for n, t in hc["tab"].items())
        + ")")
    if hc["early"]:
        msg = ("the impact term is off BELOW the floor step for "
               + ", ".join(f"{n} (first bad step {fb})"
                           for n, fb in hc["early"].items())
               + ": a shorter impact horizon keeps exactly the bad steps")
        if not OPT["hcheck_continue"]:
            raise SystemExit("  STOP: " + msg + ". Fix the link / spread, or "
                             "rerun with --hcheck-continue to evaluate anyway"
                             " (rows marked not interpretable).")
        log("  WARNING: " + msg + "; --hcheck-continue: every new-"
            "controller row is marked not interpretable")


def phase_eval(R, name):
    j_imp = need_hcheck(R, "every new-controller row")
    ev = R.setdefault("eval", {}).setdefault(name, {})
    stale = [k for k, r in ev.items() if r.get("j_imp") != j_imp]
    if stale:
        raise SystemExit(
            f"{name}: {len(stale)} cached rows were run with impact horizon "
            f"{ev[stale[0]].get('j_imp')}, the horizon check now gives "
            f"{j_imp} (it was re-decided on more models); rerun with --redo "
            f"{name}")
    todo = [j for j in eval_jobs() if (j["seed"], j["leg"]) not in ev]
    if not todo:
        return
    link = link_of(R, j_imp)
    early = bool(R["hcheck"].get("early"))
    if name == "m0":
        from learn.meta import mpc_learned as ML
        groups, net, s, stats = [todo], None, 1.0, None
        en = ML.env()
    else:
        g = gpu()
        ML, en = g["ML"], g["env"]
        net, s, stats = net_of(R, name)
        G = P("GROUP")
        groups = [todo[i:i + G] for i in range(0, len(todo), G)]
    for gi, jobs in enumerate(groups):
        t0 = time.time()
        rows = ML.run_group(jobs, en, link, net=net, stats=stats, spread=s,
                            T=P("T"), log=log, tag=f"{name} group {gi}")
        for r in rows:
            r["j_imp"] = link.j_imp
            r["not_interpretable"] = early      # --hcheck-continue past a stop
            if net is None:
                r["spread"] = None              # m0 draws no samples
            ev[(r["seed"], r["leg"])] = r
            _log_row(name, r)
        save(R)
        log(f"  {name} group {gi}: {len(jobs)} missions in "
            f"{(time.time() - t0) / 60:.1f} min, {rows[0]['t_step_mean']:.2f}"
            f" s per episode-step ({rows[0]['t_call_mean']:.2f} s per call, "
            f"p95 {rows[0]['t_call_p95']:.2f}), GPU peak "
            f"{rows[0]['gpu_peak_gb']:.2f} GB, snapshot "
            f"{rows[0]['gpu_snap']}")


# ------------------------------------------------------------ report
def _arr(rows, keys, q):
    return np.array([rows[k][q] if k in rows else np.nan for k in keys],
                    float)


def report(R):
    from learn.meta.mpc_learned import format_link
    ev = R.get("eval", {})
    keys = [(j["seed"], j["leg"]) for j in eval_jobs()]
    if "link" in R:
        log(format_link(R["link"]["info"]))
    if "link_src" in R:
        a, b = R["link"]["link"], R["link_src"]["link"]
        log(f"  link sensitivity: target A0 {a['A0']:.3f} A1 {a['A1']:+.3f} K "
            f"{a['K']:.3f} | source A0 {b['A0']:.3f} A1 {b['A1']:+.3f} K "
            f"{b['K']:.3f}")
    for nm, c in R.get("calib", {}).items():
        t = c["table"][c["s"]]
        log(f"  spread {nm}: s = {c['s']} (cov cost channels "
            f"{t['cov_cost']:.2f}, 5-ch {t['cov5']:.2f})")
    for i, a in sorted(R.get("learn", {}).get("adapt", {}).items()):
        t, t0 = a["cov"][a["s"]], a["prior_cov"][1.0]
        log(f"  adaptation {i}: {a['n_eps']} episodes, held out "
            f"{a['holdout']}, kept step {a['best_it']}, s {a['s']} (cov "
            f"{t['cov_cost']:.2f} / 5-ch {t['cov5']:.2f}); unadapted prior on "
            f"the same positions: cov {t0['cov_cost']:.2f} / 5-ch "
            f"{t0['cov5']:.2f} at s 1, its own s {a['prior_s']}")
    hc = R.get("hcheck", {})
    if "j_imp" in hc:
        from learn.meta.mpc_learned import env, format_horizon
        log(format_horizon(hc["arrays"], hc["tab"], list(hc["tab"]),
                           link_of(R, hc["j_imp"]), env()))
        log(f"  impact horizon j_imp = {hc['j_imp']} for every new-controller"
            f" row (decided on {hc['decided_on']}, the exceedance level and "
            f"growth; {hc['n_blocks']} Cb target blocks of {hc['n_eps']} "
            "episodes)")
        if hc.get("early"):
            log("  WARNING: NOT INTERPRETABLE -- the impact term is off below"
                " the floor step for " + ", ".join(
                    f"{n} (first bad step {fb})" for n, fb in
                    hc["early"].items()) + "; the rows ran past the stop "
                "with --hcheck-continue")
    # online vs prior per learning episode (paired: the same missions)
    PL, LN = R.get("prior_learn", {}), R.get("learn", {}).get("pairs", {})
    if LN:
        jl = R["learn"].get("j_imp", "?")
        log(f"  learning episodes (both columns plan with impact horizon "
            f"j_imp = {jl}; the evaluation rows use {hc.get('j_imp', '?')}):"
            " score online / prior / difference")
        for i in sorted(LN):
            for e, r in enumerate(LN[i]["rows"]):
                if i == 0:
                    log(f"    episode {e} (seed {r['seed']}, leg {r['leg']}, "
                        f"prior): {r['score']:.3f} -- shared, one run for "
                        "both columns")
                    continue
                p = PL.get(i, [None, None])[e]
                ps = p["score"] if p else float("nan")
                log(f"    episode {2 * i + e} (seed {r['seed']}, leg "
                    f"{r['leg']}, {LN[i]['who']}): {r['score']:.3f} / "
                    f"{ps:.3f} / {r['score'] - ps:+.3f}")
    rows = []
    for nm in VARIANTS:
        if nm not in ev:
            continue
        s = _arr(ev[nm], keys, "score")
        if np.isnan(s).any():
            log(f"  {nm}: {int(np.isnan(s).sum())} of {len(s)} missions "
                "missing")
            continue
        rows.append(nm)
    if not rows:
        return
    log("\n  EVALUATION, paired target missions (lower is better); "
        "mean +- SE over missions")
    hdr = (f"    {'variant':<9}{'score':>13}{'speed':>7}{'impact':>7}"
           f"{'track':>7}{'kn 0':>6}{'kn 45':>6}{'p99 g':>6}{'bad':>4}"
           f"{'nf%':>6}{'s':>6}{'s/step':>8}{'p95':>6}")
    log(hdr)
    for nm in rows:
        e = ev[nm]
        s = _arr(e, keys, "score")
        kn = _arr(e, keys, "kn")
        se = s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0
        bad = int(sum(not e[k]["finite"] for k in keys))
        sp = e[keys[0]].get("spread", "")
        sp = f"{sp:.2f}" if isinstance(sp, float) else "-"
        ts = _arr(e, keys, "t_step_mean")
        tp = _arr(e, keys, "t_step_p95") if "t_step_p95" in e[keys[0]] \
            else np.full(len(keys), np.nan)
        if "n_samp" in e[keys[0]]:
            n_b = sum(e[k]["n_bad"] for k in keys)
            n_s = sum(e[k]["n_samp"] for k in keys)
            nf = f"{100 * n_b / max(n_s, 1):.2f}"
        else:
            nf = "-"
        # * = ran past the horizon check's stop (--hcheck-continue)
        star = "*" if any(e[k].get("not_interpretable") for k in keys) \
            else ""
        log(f"    {nm + star:<9}{s.mean():>7.3f}+-{se:<4.3f}"
            f"{np.nanmean(_arr(e, keys, 'c_speed')):>7.3f}"
            f"{np.nanmean(_arr(e, keys, 'c_impact')):>7.3f}"
            f"{np.nanmean(_arr(e, keys, 'c_track')):>7.3f}"
            f"{np.nanmean(kn[0::2]):>6.1f}{np.nanmean(kn[1::2]):>6.1f}"
            f"{np.nanmean(_arr(e, keys, 'acc_cg_p99')):>6.2f}{bad:>4d}"
            f"{nf:>6}{sp:>6}{np.nanmean(ts):>8.2f}{np.nanmean(tp):>6.2f}")
    log("    notes: hand and c0 see the measured wave under the hull at the "
        "first horizon step, the new controller's rows (m0 ... oracle) see "
        "no wave. Target data in every new-controller row, m0 and the zero-"
        "shot prior included: the impact link fitted on "
        f"{R.get('link', {}).get('n_eps', '?')} target episodes, and the "
        f"impact horizon j_imp = {hc.get('j_imp', '?')} chosen on "
        f"{hc.get('n_blocks', '?')} Cb target blocks ({hc.get('n_eps', '?')} "
        "episodes) with the target-trained oracle and online nets; online "
        "adds its learning episodes. bad = non-finite missions; nf% = "
        "planner samples with a non-finite cost; s = the sampler's spread "
        "scale; s/step = controller wall time per episode-step, the "
        "controller call only in every row (hand, c0: mpc_command; the new "
        "controller: lockstep call time / rows)")
    for leg in (0, 1):
        log(f"  leg {leg}: " + ", ".join(
            f"{nm} {np.mean(_arr(ev[nm], keys, 'score')[leg::2]):.3f}"
            for nm in rows))
    log("\n  paired differences (variant - reference), mean +- SE, and as % "
        "of the reference mean")
    for ref in ("hand", "c0", "m0", "prior"):
        if ref not in rows:
            continue
        b = _arr(ev[ref], keys, "score")
        for nm in rows:
            if nm == ref:
                continue
            d = _arr(ev[nm], keys, "score") - b
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            log(f"    {nm:<8} - {ref:<6}: {d.mean():+.3f} +- {se:.3f} "
                f"({d.mean() / b.mean() * 100:+.0f}% +- "
                f"{se / b.mean() * 100:.0f}%)")


# ------------------------------------------------------------ main
def main(phases, variants, redo=()):
    R = load_R()
    for v in redo:                      # drop cached evaluation rows
        if R.get("eval", {}).pop(v, None) is not None:
            log(f"  --redo: cached {v} rows dropped")
    t0 = time.time()
    for ph in phases:
        if ph in VARIANTS and ph not in variants:
            continue
        log(f"\n  [{ph}]  {task.avail_gb():.1f} GB RAM free")
        if task.avail_gb() < 1.5:
            raise MemoryError(f"only {task.avail_gb():.1f} GB of RAM left")
        if ph == "link":
            if "link" not in R:
                phase_link(R)
        elif ph == "link_src":
            if "link_src" not in R:
                phase_link(R, world="low", key="link_src")
        elif ph in ("hand", "c0"):
            phase_ref(R, ph)
        elif ph == "calib":
            phase_calib(R)
        elif ph == "learn":
            phase_learn(R)
        elif ph == "prior_learn":
            phase_prior_learn(R)
        elif ph == "calib_prior":
            phase_calib_prior(R)
        elif ph == "hcheck_pre":
            phase_hcheck(R, pre=True)
        elif ph == "hcheck":
            phase_hcheck(R)
        elif ph in VARIANTS:
            phase_eval(R, ph)
        elif ph == "report":
            report(R)
            continue
        else:
            raise SystemExit(f"unknown phase {ph}")
        save(R)
        log(f"  [{ph}] done, {(time.time() - t0) / 60:.1f} min so far")
    return R


def _seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return tuple(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phases", default="all")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="evaluation rows to run (the phases they need "
                         "run as well)")
    ap.add_argument("--missions", default=None,
                    help="evaluation seeds, e.g. 0-7 (both legs each)")
    ap.add_argument("--episodes", type=int, default=None,
                    help="online learning episodes (even; default 12)")
    ap.add_argument("--workers", type=int, default=3,
                    help="processes for the CPU phases (link, hand, c0)")
    ap.add_argument("--gpu-frac", type=float, default=0.35)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hcheck-continue", action="store_true",
                    help="go on past a horizon-check stop (rows marked not "
                         "interpretable)")
    ap.add_argument("--redo", default="",
                    help="evaluation rows to drop from the cache and rerun")
    a = ap.parse_args()
    SMOKE = a.smoke
    OPT.update(workers=1 if a.smoke else a.workers, gpu_frac=a.gpu_frac,
               episodes=a.episodes, hcheck_continue=a.hcheck_continue,
               missions=_seeds(a.missions) if a.missions else None)
    var = tuple(v for v in a.variants.split(",") if v)
    for v in var:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}")
    phs = list(PHASES) if a.phases == "all" else a.phases.split(",")
    if a.phases == "all":
        learned = set(var) & set(("m0",) + GPU_VARIANTS)
        if not learned:
            phs = [p for p in phs if p in ("link", "hand", "c0", "report")]
        if "online" not in var and "prior_s" not in var:
            phs = [p for p in phs if p not in ("learn", "prior_learn",
                                               "calib_prior")]
        if "online" not in var:
            phs = [p for p in phs if p != "prior_learn"]
    main(phs, var, tuple(v for v in a.redo.split(",") if v))
