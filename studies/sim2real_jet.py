#!/usr/bin/env python3
"""
Sim-to-real, first pass: tune the MPC in the low-fidelity world, then adapt
it to the full plant with a small budget of target episodes.

The vessel is the 10 m test hull driven by a waterjet (wigley10_jet). The
SOURCE is the MPC's own reduced model run as a plant (sim/lofi.py): cheap,
and exactly what the controller believes. The TARGET is the full plant, which
has what the source lacks -- radiation memory, nonlinear buoyancy, slam loads,
wave forces in sway and yaw, the jet's momentum thrust, the pump losing its
prime when the intake breaks the surface, actuator lags. Here the target
stands in for the real boat; the budget that matters is target EPISODES.

The controller is the tuning setup of learn/tune.py: MPC on thrust, heading
held by the autopilot (gains scheduled on thrust, since a jet steers with its
flow). What is tuned is theta = the MPC's seven cost weights plus one jet
parameter, the least thrust it may plan ("thrust floor", 0-50% of the limit).

Methods (all recommendations evaluated on the SAME held-out target episodes,
so every comparison is paired):
  A0  hand      the default weights, no floor
  A1  src       CMA-ES in the source, nominal parameters             0 target
  A2  src-DR    CMA-ES in the source, parameters drawn from PRIOR --
                every candidate of a generation meets the SAME drawn
                vessels (A2r: repeated with another seed)             0 target
  B1  tgt-CMA   CMA-ES in the target, warm-started at A1             32 target
  B2  MF-GP     target score = source GP + residual GP, LCB search
                (B2r: another seed)                                  32 target
  B2t tgt-GP    the same search with a source that knows nothing,
                i.e. plain Bayesian optimisation (B2tr: another seed) 32 target
  B3  model     CMA-ES in the target on the MPC's MODEL (wave gains,
                heave/pitch damping), weights of A1                  32 target
  B3b model+    B3's model, weights retuned in the source rebuilt on
                that model (the model is also the source world)       0 more
  O   oracle    CMA-ES in the target with the source's budget        320 target
plus a sweep of the cross-track weight alone in both worlds (phase follow).

Mission: SS5 (Hs 3.25 m, Tp 9.7 s), head seas and 45 deg off the bow,
120 s episodes. Objective: the operator score of sim/env.py (lower is better).

What the first pass found (DEFECTS H5): the source ranks controllers no
better than chance (Spearman -0.33 over 13); with zero preview the MPC's wave
terms and wave model are inert, and the target's best controllers slow down
through the cross-track weight, which only acts where waves turn the hull --
not in the source. A2's gain did not repeat (-29% then -7%); B2 repeated
(-25%, -34%) where plain BO did not (-13%, 0%).

Run:  python -m studies.sim2real_jet [--workers 4] [--phase all|gap|src|
      fewshot|ablate|oracle|follow|eval|report]    (~50 min at 4-6 workers)
Results are cached per phase in studies/_cache/sim2real_<hull>.json.
"""
import argparse
import ctypes
import json
import multiprocessing as mp
import os
import time

import numpy as np

HULL = "wigley10_jet"
HS, TP = 3.25, 9.7
HEADINGS = (np.pi, np.pi - np.pi / 4)          # head, 45 deg off the bow
T_END = 120.0
EP_KW = dict(n_samples=128, n_freq=24, n_dir=5, use_rudder=False,
             autopilot=True)
FLOOR_MAX = 0.5
EVAL_SEEDS = tuple(range(8))                    # x 2 headings = 16 episodes
MODEL_KEYS = ("k_wave_heave", "k_wave_pitch", "z_heave", "z_pitch")
MODEL_RANGE = {"k_wave_heave": (0.5, 2.0), "k_wave_pitch": (0.5, 2.0),
               "z_heave": (0.5, 2.0), "z_pitch": (0.5, 2.0)}
TGT_BUDGET = 32


# ---------------------------------------------------------------- helpers
def avail_gb():
    """Available physical memory, GB (Windows; no psutil in this venv)."""
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 30
    except Exception:
        return float("nan")


def theta(z):
    """z in [-1, 1]^8 -> (weights, thrust floor)."""
    from control.mpc import WEIGHT_BOUNDS
    z = np.clip(np.asarray(z, float), -1.0, 1.0)
    lo, hi = WEIGHT_BOUNDS[:, 0], WEIGHT_BOUNDS[:, 1]
    w = lo + 0.5 * (z[:7] + 1.0) * (hi - lo)
    return w, 0.5 * (z[7] + 1.0) * FLOOR_MAX


def z_hand():
    from control.mpc import WEIGHT_BOUNDS, DEFAULT_WEIGHTS
    lo, hi = WEIGHT_BOUNDS[:, 0], WEIGHT_BOUNDS[:, 1]
    return list(2 * (DEFAULT_WEIGHTS - lo) / (hi - lo) - 1) + [-1.0]


def model_over(y, nominal):
    """y in [-1, 1]^4 -> overrides of the MPC's model, log-scaled."""
    out = {}
    for k, yi in zip(MODEL_KEYS, np.clip(np.asarray(y, float), -1, 1)):
        lo, hi = MODEL_RANGE[k]
        out[k] = float(nominal[k] * np.exp(np.log(lo) + 0.5 * (yi + 1)
                                           * np.log(hi / lo)))
    return out


# ------------------------------------------------------------------ worker
_W = {}


def _init():
    import warnings
    warnings.filterwarnings("ignore")
    import contextlib
    import io
    from sim import config, lofi
    with contextlib.redirect_stdout(io.StringIO()):
        h, db = config.load(HULL)
        geom, p = lofi.template(db, h)
    _W.update(h=h, db=db, p=dict(p))


def run_episode(job):
    """job = dict(z, world, seed, head, plant=None, model=None) -> metrics.

    `plant` overrides the source world's coefficients (randomisation, or an
    adapted model); `model` overrides the MPC's internal model. The source
    world is built from the MPC's model unless `plant` says otherwise."""
    import contextlib
    import io
    from control.reduced import ReducedModel
    from sim.env import Episode, score
    if not _W:
        _init()
    p = dict(_W["p"])
    if job.get("model"):
        p.update(job["model"])
    red = ReducedModel(p)
    w, floor = theta(job["z"])
    with contextlib.redirect_stdout(io.StringIO()):
        ep = Episode(_W["db"], red, hs=HS, tp=TP, seed=int(job["seed"]),
                     u_ref=p["u_design"], weights=w, hull=_W["h"],
                     theta0=HEADINGS[job["head"]], fidelity=job["world"],
                     plant_params=job.get("plant"), thrust_floor=floor,
                     **EP_KW)
        m = ep.run(T_END)
    m = {k: v for k, v in m.items() if np.isscalar(v)}
    m["score"] = float(score(m, p["u_design"]))
    return m


class Runner:
    def __init__(self, workers):
        self.workers = workers
        self.pool = mp.Pool(workers, initializer=_init)
        self.n = {"low": 0, "high": 0}

    def __call__(self, jobs):
        if avail_gb() < 1.5:
            raise MemoryError(f"only {avail_gb():.1f} GB of RAM left")
        for j in jobs:
            self.n[j["world"]] += 1
        return self.pool.map(run_episode, jobs, chunksize=1)

    def close(self):
        self.pool.close()
        self.pool.join()


# -------------------------------------------------------------- optimisers
class CMA:
    """(mu/mu_w, lambda)-CMA-ES in ask/tell form, the update of learn/tune.py
    so a whole generation can be evaluated in parallel."""

    def __init__(self, x0, sigma0, popsize, seed=0):
        n = len(x0)
        self.n, self.lam = n, popsize
        self.rng = np.random.default_rng(seed)
        mu = popsize // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.mu, self.w = mu, w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        me = self.mueff
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n)
        self.cs = (me + 2) / (n + me + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.damps = 1 + 2 * max(0, np.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self.mean, self.sigma = np.array(x0, float), float(sigma0)
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.C, self.B, self.D = np.eye(n), np.eye(n), np.ones(n)
        self.gen = 0

    def ask(self):
        self._y = self.rng.normal(size=(self.lam, self.n)) @ (self.B
                                                             * self.D).T
        return np.clip(self.mean + self.sigma * self._y, -1.0, 1.0)

    def tell(self, fs):
        order = np.argsort(fs)
        y = self._y
        old = self.mean
        self.mean = np.clip(old + self.sigma * (self.w @ y[order[:self.mu]]),
                            -1.0, 1.0)
        Cis = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * self.mueff) * (Cis @ (self.mean - old)
                                                     / self.sigma)
        self.gen += 1
        hsig = (np.linalg.norm(self.ps)
                / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chiN
                < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff) * (self.mean - old) \
            / self.sigma
        a = y[order[:self.mu]]
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc)
                               + (not hsig) * self.cc * (2 - self.cc)
                               * self.C)
                  + self.cmu * (a.T * self.w) @ a)
        self.sigma *= np.exp((self.cs / self.damps)
                             * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        d2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(d2, 1e-20))


class GP:
    """Isotropic RBF Gaussian process on standardised targets, lengthscale
    and noise chosen by marginal likelihood on a small grid. Enough for a
    few dozen points in eight dimensions; no sklearn in this venv."""

    def fit(self, X, y):
        X, y = np.atleast_2d(np.asarray(X, float)), np.asarray(y, float)
        self.X, self.m, self.s = X, float(y.mean()), float(y.std() + 1e-9)
        t = (y - self.m) / self.s
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        best = None
        for ls in (0.3, 0.5, 0.8, 1.2, 2.0):
            for sn in (0.05, 0.15, 0.3, 0.6):
                K = np.exp(-0.5 * d2 / ls ** 2) + (sn + 1e-8) * np.eye(len(t))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    continue
                a = np.linalg.solve(L.T, np.linalg.solve(L, t))
                nll = 0.5 * t @ a + np.log(np.diag(L)).sum()
                if best is None or nll < best[0]:
                    best = (nll, ls, sn, L, a)
        _, self.ls, self.sn, self.L, self.a = best
        return self

    def predict(self, Xs):
        Xs = np.atleast_2d(np.asarray(Xs, float))
        d2 = ((Xs[:, None, :] - self.X[None, :, :]) ** 2).sum(-1)
        k = np.exp(-0.5 * d2 / self.ls ** 2)
        mu = k @ self.a
        v = np.linalg.solve(self.L, k.T)
        var = np.maximum(1.0 - (v ** 2).sum(0), 1e-12)
        return self.m + self.s * mu, self.s * np.sqrt(var)


# ------------------------------------------------------------------ phases
def jobs_for(z, world, seeds, plant=None, model=None):
    return [dict(z=list(map(float, z)), world=world, seed=int(s), head=h,
                 plant=plant, model=model)
            for s in seeds for h in range(len(HEADINGS))]


def mean_score(results):
    return float(np.mean([r["score"] for r in results]))


def tune_cma(run, x0, sigma0, world, n_gen, pop, seed0, per_seeds,
             plant_fn=None, model=None, tag=""):
    """CMA-ES where each candidate is scored on `per_seeds` seeds x both
    headings; the seeds change every generation and are shared within it
    (common random numbers inside a generation)."""
    es = CMA(x0, sigma0, pop, seed=seed0)
    hist = []
    for g in range(n_gen):
        X = es.ask()
        seeds = [seed0 * 1000 + g * per_seeds + i for i in range(per_seeds)]
        jobs, owner = [], []
        for i, x in enumerate(X):
            for j in jobs_for(x, world, seeds, model=model):
                if plant_fn is not None:
                    j["plant"] = plant_fn(seed0, g, i, j["seed"], j["head"])
                jobs.append(j)
                owner.append(i)
        res = run(jobs)
        fs = np.zeros(pop)
        for i, r in zip(owner, res):
            fs[i] += r["score"]
        fs /= len(seeds) * len(HEADINGS)
        es.tell(fs)
        hist.append(dict(gen=g, X=X.tolist(), f=fs.tolist(),
                         mean=es.mean.tolist(), sigma=es.sigma))
        print(f"    {tag} gen {g:2d}: best {fs.min():.4f}  median "
              f"{np.median(fs):.4f}  sigma {es.sigma:.3f}  "
              f"[{avail_gb():.1f} GB free]", flush=True)
    return es.mean.tolist(), hist


def phase_gap(run, R):
    """The same controllers in both worlds, paired: hand weights at two
    floors, on the evaluation seeds."""
    out = {}
    for name, z in (("hand", z_hand()),
                    ("hand_floor30", z_hand()[:7] + [2 * 0.3 / FLOOR_MAX - 1])):
        for world in ("low", "high"):
            out[f"{name}/{world}"] = run(jobs_for(z, world, EVAL_SEEDS))
    R["gap"] = out


def phase_src(run, R):
    from sim import lofi
    nominal = None
    z0 = z_hand()
    if "A1" not in R:
        print("  A1: CMA-ES in the source world, nominal", flush=True)
        R["A1"] = dict(zip(("z", "hist"), tune_cma(
            run, z0, 0.35, "low", n_gen=10, pop=8, seed0=11, per_seeds=2,
            tag="A1")))

    def draw(seed0, g, i, s, h):
        # The randomised vessels depend on the generation and the episode,
        # NOT on the candidate: every candidate of a generation meets the
        # same vessels. The first run drew them per candidate, and the
        # ranking inside a generation then mostly said which candidate had
        # drawn the easier boats (median 2-5x the best).
        nonlocal nominal
        if nominal is None:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                nominal = lofi.nominal_params(hull=HULL)
        rng = np.random.default_rng([seed0, g, s, h])
        return lofi.perturb(nominal, rng)

    print("  A2: CMA-ES in the source world, randomised (PRIOR)", flush=True)
    R["A2"] = dict(zip(("z", "hist"), tune_cma(
        run, z0, 0.35, "low", n_gen=10, pop=8, seed0=12, per_seeds=2,
        plant_fn=draw, tag="A2")))


def phase_fewshot(run, R):
    z1 = R["A1"]["z"]
    evals = TGT_BUDGET // len(HEADINGS)          # one seed per evaluation

    # B1: direct CMA-ES in the target, warm-started at A1
    print("  B1: CMA-ES in the target", flush=True)
    R["B1"] = dict(zip(("z", "hist"), tune_cma(
        run, z1, 0.2, "high", n_gen=evals // 4, pop=4, seed0=21,
        per_seeds=1, tag="B1")))

    # B2: multi-fidelity GP. Source GP from A1's whole history (80
    # candidates, each 4 source episodes); residual GP on target - source.
    print("  B2: multi-fidelity GP", flush=True)
    R["B2"] = gp_search(run, R, use_source=True, tag="B2")

    # B3: adapt the MPC's MODEL in the target, weights fixed at A1
    print("  B3: model adaptation in the target", flush=True)
    nominal = dict(_nominal())
    es = CMA(np.zeros(len(MODEL_KEYS)), 0.4, 4, seed=23)
    hist = []
    for g in range(evals // 4):
        Y = es.ask()
        jobs = []
        for i, y in enumerate(Y):
            jobs += jobs_for(z1, "high", [23000 + g], model=model_over(
                y, nominal))
        res = run(jobs)
        fs = np.array([mean_score(res[2 * i:2 * i + 2]) for i in range(4)])
        es.tell(fs)
        hist.append(dict(gen=g, Y=Y.tolist(), f=fs.tolist(),
                         mean=es.mean.tolist()))
        print(f"    B3 gen {g}: best {fs.min():.4f}  sigma {es.sigma:.3f}",
              flush=True)
    R["B3"] = dict(z=z1, model=model_over(es.mean, nominal), hist=hist)

    # B3b: the adapted model is also the source world -> retune there
    print("  B3b: retune weights in the source rebuilt on B3's model",
          flush=True)
    m3 = R["B3"]["model"]
    z3b, h3b = tune_cma(run, z1, 0.25, "low", n_gen=6, pop=8, seed0=24,
                        per_seeds=2, model=m3,
                        plant_fn=lambda *a: m3, tag="B3b")
    R["B3b"] = dict(z=z3b, model=m3, hist=h3b)


class _Flat:
    """A source 'GP' that knows nothing: constant mean, fixed spread."""

    def __init__(self, m):
        self.m, self.ls, self.sn = float(m), None, None

    def predict(self, X):
        n = len(np.atleast_2d(X))
        return np.full(n, self.m), np.full(n, 0.1)


def gp_search(run, R, use_source=True, tag="B2", seed=22):
    """Batch GP search in the target: 4 rounds x 4 candidates x 2 episodes.

    Predicted target score = source GP + residual GP (target - source); the
    candidates minimise the lower confidence bound, spread at least 0.25
    apart, sampled around the incumbent and around A1. use_source=False is
    the ablation: the same search with a source that knows nothing, i.e.
    plain Bayesian optimisation on the target -- same seeds, same budget."""
    z1 = R["A1"]["z"]
    evals = TGT_BUDGET // len(HEADINGS)
    Xs = np.array([x for h in R["A1"]["hist"] for x in h["X"]])
    ys = np.array([f for h in R["A1"]["hist"] for f in h["f"]])
    gs = GP().fit(Xs, ys) if use_source else _Flat(ys.mean())
    rng = np.random.default_rng(seed)
    Xt, yt, hist = [], [], []
    inc = np.array(z1)
    for rnd in range(evals // 4):
        cand = np.clip(np.vstack([inc + 0.12 * rng.normal(size=(1500, 8)),
                                  np.array(z1) + 0.3
                                  * rng.normal(size=(1500, 8))]), -1, 1)
        mu_s, sd_s = gs.predict(cand)
        if Xt:
            gr = GP().fit(np.array(Xt), np.array(yt)
                          - gs.predict(np.array(Xt))[0])
            mu_r, sd_r = gr.predict(cand)
        else:
            mu_r, sd_r = np.zeros(len(cand)), np.full(len(cand), 0.1)
        lcb = mu_s + mu_r - 1.0 * np.sqrt(sd_s ** 2 + sd_r ** 2)
        pick = []
        for _ in range(4):
            if pick:
                d = np.min([np.linalg.norm(cand - cand[p], axis=1)
                            for p in pick], axis=0)
                score_ = np.where(d < 0.25, np.inf, lcb)
            else:
                score_ = lcb
            pick.append(int(np.argmin(score_)))
        seeds = [seed * 1000 + rnd * 4 + i for i in range(4)]
        jobs = []
        for i, p in enumerate(pick):
            jobs += jobs_for(cand[p], "high", [seeds[i]])
        res = run(jobs)
        for i, p in enumerate(pick):
            Xt.append(cand[p].tolist())
            yt.append(mean_score(res[2 * i:2 * i + 2]))
        # recommend the evaluated point with the best posterior mean
        gr = GP().fit(np.array(Xt), np.array(yt)
                      - gs.predict(np.array(Xt))[0])
        post = gs.predict(np.array(Xt))[0] + gr.predict(np.array(Xt))[0]
        inc = np.array(Xt[int(np.argmin(post))])
        hist.append(dict(round=rnd, X=[cand[p].tolist() for p in pick],
                         f=yt[-4:], post_best=float(post.min())))
        print(f"    {tag} round {rnd}: target {np.round(yt[-4:], 4).tolist()}"
              f"  posterior best {post.min():.4f}", flush=True)
    return dict(z=inc.tolist(), hist=hist,
                source_gp=dict(ls=gs.ls, sn=gs.sn))


def phase_ablate(run, R):
    print("  B2t: the same GP search without the source (plain BO)",
          flush=True)
    R["B2t"] = gp_search(run, R, use_source=False, tag="B2t")


def phase_follow(run, R):
    """Follow-ups to what the first evaluation showed.

    1. The speed the tuned controllers settle to in the TARGET falls in step
       with their cross-track weight, while in the source every one of them
       runs at full speed. Sweep that weight alone, A1's others fixed.
    2. A2's gain may be luck -- a weight inert in the source drifting high.
       Repeat it with another search seed.
    3. B2 beat B2t once; repeat both with another seed."""
    from control.mpc import WEIGHT_BOUNDS
    lo, hi = WEIGHT_BOUNDS[3]
    sw = R.setdefault("sweep_track", {})
    for wt in (0.0, 2.5, 5.0, 10.0):
        key = f"{wt:g}"
        if key in sw:
            continue
        z = list(R["A1"]["z"])
        z[3] = 2 * (wt - lo) / (hi - lo) - 1
        sw[key] = dict(high=run(jobs_for(z, "high", EVAL_SEEDS)),
                       low=run(jobs_for(z, "low", EVAL_SEEDS)))
        print(f"    w_track {wt:g}: target "
              f"{mean_score(sw[key]['high']):.4f}, source "
              f"{mean_score(sw[key]['low']):.4f}", flush=True)
    if "A2r" not in R:
        from sim import lofi
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            nominal = lofi.nominal_params(hull=HULL)

        def draw(seed0, g, i, s, h):
            return lofi.perturb(nominal, np.random.default_rng([seed0, g, s,
                                                                h]))
        print("  A2r: A2 repeated with another seed", flush=True)
        R["A2r"] = dict(zip(("z", "hist"), tune_cma(
            run, z_hand(), 0.35, "low", n_gen=10, pop=8, seed0=13,
            per_seeds=2, plant_fn=draw, tag="A2r")))
    if "B2r" not in R:
        R["B2r"] = gp_search(run, R, use_source=True, tag="B2r", seed=25)
    if "B2tr" not in R:
        R["B2tr"] = gp_search(run, R, use_source=False, tag="B2tr",
                              seed=25)


def phase_oracle(run, R):
    print("  O: CMA-ES in the target with the source budget", flush=True)
    R["O"] = dict(zip(("z", "hist"), tune_cma(
        run, R["A1"]["z"], 0.35, "high", n_gen=10, pop=8, seed0=31,
        per_seeds=2, tag="O")))


def _nominal():
    import contextlib
    import io
    from sim import lofi
    with contextlib.redirect_stdout(io.StringIO()):
        return lofi.nominal_params(hull=HULL)


def recommendations(R):
    recs = {"A0 hand": (z_hand(), None)}
    for k, nm in (("A1", "A1 src"), ("A2", "A2 src-DR"), ("A2r", "A2r src-DR"),
                  ("B1", "B1 tgt-CMA"), ("B2", "B2 MF-GP"),
                  ("B2r", "B2r MF-GP"), ("B2t", "B2t tgt-GP"),
                  ("B2tr", "B2tr tgt-GP"), ("B3", "B3 model"),
                  ("B3b", "B3b model+"), ("O", "O oracle")):
        if k in R:
            recs[nm] = (R[k]["z"], R[k].get("model"))
    return recs


def phase_eval(run, R):
    out = R.setdefault("eval", {})
    for nm, (z, model) in recommendations(R).items():
        if nm in out:
            continue
        print(f"  evaluating {nm}", flush=True)
        out[nm] = dict(high=run(jobs_for(z, "high", EVAL_SEEDS, model=model)),
                       low=run(jobs_for(z, "low", EVAL_SEEDS, model=model,
                                        plant=model)))


# ------------------------------------------------------------------ report
def report(R):
    def arr(rs, k="score"):
        return np.array([r[k] for r in rs])

    print("\n  THE GAP: the same controller in both worlds, 16 paired "
          "episodes (8 seeds x head / 45 deg)")
    keys = ("score", "u_mean", "acc_p99", "slam_rate_ochi", "rvm_rms",
            "cross_rms", "heading_rms")
    if "gap" in R:
        print("    " + " " * 20 + "".join(f"{k:>15}" for k in keys))
        for nm, rs in R["gap"].items():
            print(f"    {nm:<20}" + "".join(
                f"{np.mean(arr(rs, k)) * (57.2958 if k == 'heading_rms' else 1):>15.3f}"
                for k in keys))
    if "sweep_track" in R:
        print("\n  CROSS-TRACK WEIGHT alone (A1's others), 16 paired episodes"
              " per world")
        print(f"    {'w_track':>8}{'target score':>14}{'target u':>10}"
              f"{'heading rms':>13}{'source score':>14}{'source u':>10}")
        for wt, d in R["sweep_track"].items():
            print(f"    {wt:>8}{np.mean(arr(d['high'])):>14.4f}"
                  f"{np.mean(arr(d['high'], 'u_mean')):>10.2f}"
                  f"{np.degrees(np.mean(arr(d['high'], 'heading_rms'))):>9.1f} deg"
                  f"{np.mean(arr(d['low'])):>14.4f}"
                  f"{np.mean(arr(d['low'], 'u_mean')):>10.2f}")
    if "eval" not in R:
        return
    ev = R["eval"]
    base = arr(ev["A1 src"]["high"]) if "A1 src" in ev else None
    print("\n  EVALUATION in the target (full plant), 16 held-out paired "
          "episodes; lower score is better")
    print(f"    {'method':<13}{'target eps':>11}{'score':>9}{'+-se':>7}"
          f"{'vs A1':>9}{'+-se':>7}{'src says':>10}{'u':>7}{'acc_p99':>9}"
          f"{'slam':>7}{'floor':>7}")
    budget = {"A0 hand": 0, "A1 src": 0, "A2 src-DR": 0, "A2r src-DR": 0,
              "B1 tgt-CMA": 32, "B2 MF-GP": 32, "B2r MF-GP": 32,
              "B2t tgt-GP": 32, "B2tr tgt-GP": 32, "B3 model": 32,
              "B3b model+": 32,
              "O oracle": 320}
    recs = recommendations(R)
    for nm, d in ev.items():
        s = arr(d["high"])
        # an evaluated entry kept for the record (the first A2, whose
        # candidates each drew their own boats) has no current recommendation
        z = recs.get(nm, (R.get("A2_first_run_percandidate_draws", {})
                          .get("z"), None))[0]
        line = (f"    {nm:<13}{budget.get(nm, 0):>11}{s.mean():>9.4f}"
                f"{s.std(ddof=1) / np.sqrt(len(s)):>7.4f}")
        if base is not None:
            dd = (s - base) / base.mean() * 100
            line += f"{dd.mean():>+8.1f}%{dd.std(ddof=1) / np.sqrt(len(dd)):>6.1f}"
        else:
            line += " " * 16
        line += (f"{arr(d['low']).mean():>10.4f}"
                 f"{np.mean(arr(d['high'], 'u_mean')):>7.2f}"
                 f"{np.mean(arr(d['high'], 'acc_p99')):>9.3f}"
                 f"{np.mean(arr(d['high'], 'slam_rate_ochi')):>7.3f}"
                 f"{theta(z)[1] if z is not None else float('nan'):>7.2f}")
        print(line)
    # does the source rank controllers as the target does?
    names = list(ev)
    lo = np.array([arr(ev[n]["low"]).mean() for n in names])
    hi = np.array([arr(ev[n]["high"]).mean() for n in names])
    from scipy.stats import spearmanr
    rho = spearmanr(lo, hi).correlation
    print(f"\n    source vs target ranking of these {len(names)} controllers:"
          f" Spearman {rho:+.2f}")


# -------------------------------------------------------------------- main
def main(phase="all", workers=4):
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_cache", f"sim2real_{HULL}.json")
    R = json.load(open(cache)) if os.path.exists(cache) else {}
    order = ("gap", "src", "fewshot", "ablate", "oracle", "follow", "eval")
    todo = order if phase == "all" else (phase,) if phase != "report" else ()
    run = None
    t0 = time.time()
    try:
        for ph in todo:
            done = {"gap": "gap" in R, "src": "A2" in R,
                    "fewshot": "B3b" in R, "ablate": "B2t" in R,
                    "oracle": "O" in R,
                    "follow": "B2tr" in R and len(R.get("sweep_track",
                                                        {})) == 4,
                    "eval": False}[ph]
            if done:
                print(f"  [{ph}] cached", flush=True)
                continue
            if run is None:
                print(f"  {workers} workers, {avail_gb():.1f} GB free",
                      flush=True)
                run = Runner(workers)
            print(f"\n  [{ph}]", flush=True)
            {"gap": phase_gap, "src": phase_src, "fewshot": phase_fewshot,
             "ablate": phase_ablate, "oracle": phase_oracle,
             "follow": phase_follow, "eval": phase_eval}[ph](run, R)
            json.dump(R, open(cache, "w"))
            print(f"  [{ph}] done, {(time.time() - t0) / 60:.1f} min so far,"
                  f" episodes low {run.n['low']} high {run.n['high']}",
                  flush=True)
    finally:
        if run is not None:
            run.close()
    report(R)
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    main(a.phase, a.workers)
