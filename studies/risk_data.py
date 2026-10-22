#!/usr/bin/env python3
"""
Training data for the slow layer's risk functions (studies/risk_function.py):
constant-speed runs of the frozen plant scarab195@as1 over a space-filling
design of sea state, relative heading and speed, with the randomisation on
(hull fields, sea inputs, thrust gain; output scales off -- as the
envelope study's random phase), one wave seed per point.

Design (Latin hypercube, N points, seed DESIGN_SEED):
  Hs        0.5 - 1.75 m
  Tp        5.0 s x sqrt(Hs / 1 m) x [0.8, 1.25]  (steepness about that of
            the evaluation sea, Hs / lambda_p ~ 1/39, spread either way;
            the draw's own Tp factor 0.8-1.25 comes on top)
  heading   relative wave direction 90 - 180 deg (beam to head seas; the
            draw shifts it by up to +-15 deg)
  speed     8 - 30 kn, the thrust fold 12.5-14 kn excluded
Each run: 60 s settling + 600 s of statistics (studies/envelope_run.run_one),
the same outputs as the envelope runs, so studies/risk_function reads both
caches alike.

    python -m studies.risk_data run [N]      (default N = 160)
    python -m studies.risk_data jobs

Outputs: studies/_cache/riskdata/runs/<name>.json + .npz, progress.log.
Read-only use of sim/, hydro/, app/, studies/envelope_run.
"""
import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np   # noqa: E402

from studies import envelope_run as R            # noqa: E402
from studies import envelope_pass1 as P          # noqa: E402

OUT = os.path.join(ROOT, "studies", "_cache", "riskdata")
RUNS = os.path.join(OUT, "runs")
DESIGN_SEED = 20261019
N_DEFAULT = 160
FOLD = (12.5, 14.0)
MAX_PROCS = 8
GB_PER_JOB = 0.2


def lhs(n, d, rng):
    """Latin hypercube in [0, 1]^d."""
    u = np.empty((n, d))
    for j in range(d):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u


def design(n=N_DEFAULT, seed=DESIGN_SEED):
    rng = np.random.default_rng(seed)
    u = lhs(n, 4, rng)
    hs = 0.5 + 1.25 * u[:, 0]
    tp = 5.0 * np.sqrt(hs) * (0.8 + 0.45 * u[:, 1])
    rel = 90.0 + 90.0 * u[:, 2]
    # speed: 8-30 kn with the fold removed (map the unit interval onto the
    # two allowed segments, proportionally to their lengths)
    seg = [(8.0, FOLD[0]), (FOLD[1], 30.0)]
    L = np.array([b - a for a, b in seg])
    frac = L / L.sum()
    v = np.where(u[:, 3] < frac[0],
                 seg[0][0] + u[:, 3] / frac[0] * L[0],
                 seg[1][0] + (u[:, 3] - frac[0]) / frac[1] * L[1])
    return hs, tp, rel, v


def jobs(n=N_DEFAULT):
    hs, tp, rel, v = design(n)
    # every episode draws a new hull here, so a plant cache only costs
    # memory (pass 1's cache of 8 plants made each worker ~1 GB)
    rz = dict(P.rz_config(), plant_cache=1)
    out = []
    for i in range(n):
        out.append(dict(
            name=f"rd{i:03d}_hs{hs[i]:.2f}_tp{tp[i]:.1f}_rel{rel[i]:.0f}_"
                 f"{v[i]:.1f}kn", design_index=i, hs=float(hs[i]),
            tp=float(tp[i]), rel_deg=float(rel[i]),
            heading_rad=math.radians(180.0 - rel[i]), speed_kn=float(v[i]),
            seed=3000 + i, sea_seed=100 + i, randomize=True,
            randomization=rz, i_clip=P.I_CLIP))
    return out


def paths(name):
    return (os.path.join(RUNS, name + ".json"),
            os.path.join(RUNS, name + ".npz"))


def done(name):
    pj, pn = paths(name)
    return os.path.exists(pj) and os.path.exists(pn)


def job(arg):
    os.makedirs(RUNS, exist_ok=True)
    name = arg["name"]
    pj, pn = paths(name)
    t0 = time.time()
    if done(name):
        with open(pj) as f:
            return json.load(f)
    r = R.run_one(arg)
    st = r.pop("stats", None)
    series = r.pop("series", None)
    arrays = {}
    light = dict(r)
    light["arg"] = {k: v for k, v in arg.items() if k != "randomization"}
    if st is not None:
        st = dict(st)
        for g in ("cg", "bow"):
            pk = st[g].pop("peaks")
            arrays[f"peaks_{g}"] = np.asarray(pk, float)
        light["stats"] = st
    else:
        light["stats"] = None
    if series:
        arrays.update(series)
    np.savez_compressed(pn, **arrays)
    with open(pj, "w") as f:
        json.dump(light, f, indent=1, default=P._plain)
    with open(os.path.join(OUT, "progress.log"), "a") as f:
        s = light.get("stats") or {}
        f.write(f"{time.strftime('%H:%M:%S')} {name} done in "
                f"{time.time() - t0:.0f} s ({light['ms_per_step']:.2f} ms/"
                f"step) reason {light['reason']} u "
                f"{s.get('u_mean_kn', float('nan')):.2f} kn cg a110 "
                f"{(s.get('cg') or {}).get('a110', float('nan')):.2f} "
                f"imm {s.get('n_immersion', -1)}\n")
    return light


def cmd_run(n):
    from studies.parallel_runner import free_gb, n_workers, run_parallel
    os.makedirs(RUNS, exist_ok=True)
    todo = [a for a in jobs(n) if not done(a["name"])]
    nw = n_workers(GB_PER_JOB, 1.5, MAX_PROCS)
    msg = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} run: {len(todo)} runs to "
           f"do, free RAM {free_gb():.1f} GB, {nw} workers")
    print(msg, flush=True)
    with open(os.path.join(OUT, "progress.log"), "a") as f:
        f.write(msg + "\n")
    t0 = time.time()
    res = run_parallel(job, todo, gb_per_job=GB_PER_JOB, reserve_gb=1.5,
                       max_procs=MAX_PROCS)
    bad = [(a["name"], r) for a, r in zip(todo, res)
           if isinstance(r, Exception)]
    msg = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} run done: {len(todo)} "
           f"runs in {(time.time() - t0) / 60:.1f} min, {len(bad)} failed")
    print(msg, flush=True)
    with open(os.path.join(OUT, "progress.log"), "a") as f:
        f.write(msg + "\n")
        for nm, e in bad:
            f.write(f"  FAILED {nm}: {e!r}\n")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "jobs"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else N_DEFAULT
    if cmd == "run":
        cmd_run(n)
    elif cmd == "jobs":
        for a in jobs(n):
            print(f"{'done' if done(a['name']) else '    '}  {a['name']}")
    else:
        raise SystemExit(__doc__)
