#!/usr/bin/env python3
"""
M6 -- tune the MPC weights.

Seven bounded numbers, not a neural network. That is the whole point: this
budget (10^2-10^3 episodes) fits inside a sea-trial campaign, so the same
procedure that runs here can run on the water. A policy network needing 10^7
steps cannot, and would have to accept whatever sim-to-real gap remained.

CMA-ES rather than PPO because the search is 7-dimensional and the objective is
an episode return, not a per-step reward: there is no credit-assignment problem
to solve, so a policy-gradient method would only add variance. PPO becomes the
right tool if the weights are later made state-dependent (scheduled on sea
state), which is the natural next step and the reason `evaluate` already takes
a list of sea states.

The objective is the OPERATOR's, deliberately different from the MPC's own cost
-- tuning a controller against its own objective only teaches it to agree with
itself.
"""
import json
import numpy as np

from hydro import bem
from sim.env import Episode, score
from sim.vessel import NonlinearVessel
from sim.test_vessel import Monochromatic
from control.reduced import ReducedModel
from control.mpc import WEIGHT_BOUNDS, WEIGHT_NAMES, DEFAULT_WEIGHTS

DB = "hydro_wigley_10m.npz"
SEA_STATES = [(3.25, 9.7), (1.88, 8.0)]
U_REF = 4.5


def denorm(z):
    lo, hi = WEIGHT_BOUNDS[:, 0], WEIGHT_BOUNDS[:, 1]
    return lo + 0.5 * (np.clip(z, -1, 1) + 1) * (hi - lo)


def norm(w):
    lo, hi = WEIGHT_BOUNDS[:, 0], WEIGHT_BOUNDS[:, 1]
    return 2 * (np.asarray(w) - lo) / (hi - lo) - 1


def evaluate(db, red, w, seeds=(0, 1), t_end=120.0, t_preview=0.0,
             sea_states=SEA_STATES):
    vals = []
    for hs, tp in sea_states:
        for sd in seeds:
            # Short-crested seas with the MPC steering diverge -- the reduced
            # model has no sway state, so the controller cannot see the drift
            # it causes. Thrust from the MPC, heading from the autopilot, which
            # is the split the vessel would actually be built with.
            ep = Episode(db, red, hs=hs, tp=tp, seed=sd, weights=w,
                         t_preview=t_preview, u_ref=U_REF, n_samples=128,
                         n_freq=24, n_dir=5, use_rudder=False, autopilot=True)
            vals.append(score(ep.run(t_end), U_REF))
    return float(np.mean(vals))


def cma_es(f, x0, sigma0=0.45, n_gen=18, popsize=10, seed=0, verbose=True):
    """Compact (mu/mu_w, lambda)-CMA-ES. Written out rather than pulled in as a
    dependency: seven dimensions does not justify one, and having the update
    visible makes the search reproducible."""
    n = len(x0)
    rng = np.random.default_rng(seed)
    mu = popsize // 2
    wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    wts /= wts.sum()
    mueff = 1.0 / np.sum(wts ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    xmean, sigma = np.array(x0, float), sigma0
    pc, ps = np.zeros(n), np.zeros(n)
    C, B, D = np.eye(n), np.eye(n), np.ones(n)
    best, best_f, hist = xmean.copy(), np.inf, []

    for gen in range(n_gen):
        z = rng.normal(size=(popsize, n))
        y = z @ (B * D).T
        x = xmean + sigma * y
        fs = np.array([f(np.clip(xi, -1, 1)) for xi in x])
        order = np.argsort(fs)
        if fs[order[0]] < best_f:
            best_f, best = fs[order[0]], np.clip(x[order[0]], -1, 1).copy()
        hist.append((gen, float(fs[order[0]]), float(np.median(fs)), best_f))
        if verbose:
            print(f"  gen {gen:2d}  best {fs[order[0]]:.4f}  "
                  f"median {np.median(fs):.4f}  overall {best_f:.4f}  "
                  f"sigma {sigma:.3f}")

        xold = xmean
        xmean = xold + sigma * (wts @ y[order[:mu]])
        Cinv_sqrt = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (
            Cinv_sqrt @ (xmean - xold) / sigma)
        hsig = (np.linalg.norm(ps)
                / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN
                < 1.4 + 2 / (n + 1))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (
            xmean - xold) / sigma
        artmp = y[order[:mu]]
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (not hsig) * cc * (2 - cc) * C)
             + cmu * (artmp.T * wts) @ artmp)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        C = np.triu(C) + np.triu(C, 1).T
        D2, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D2, 1e-20))
    return best, best_f, hist


def main(n_gen=14, popsize=8, t_preview=0.0, out="tuned_weights.json"):
    db = bem.load(DB)
    plant = NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L)
    print("identifying reduced model ...")
    red = ReducedModel.identify(plant, db)

    f0 = evaluate(db, red, DEFAULT_WEIGHTS, t_preview=t_preview)
    print(f"\nbaseline (hand-set weights) score = {f0:.4f}")
    print(f"  " + "  ".join(f"{n}={v:g}" for n, v in
                            zip(WEIGHT_NAMES, DEFAULT_WEIGHTS)))
    print(f"\nCMA-ES, {n_gen} generations x {popsize} = "
          f"{n_gen*popsize} episodes-of-4:")

    def obj(z):
        return evaluate(db, red, denorm(z), t_preview=t_preview)

    z, fz, hist = cma_es(obj, norm(DEFAULT_WEIGHTS), n_gen=n_gen,
                         popsize=popsize)
    w = denorm(z)
    print(f"\ntuned score = {fz:.4f}  "
          f"({100*(f0-fz)/max(abs(f0),1e-9):+.1f}% vs hand-set)")
    for n_, v, d in zip(WEIGHT_NAMES, w, DEFAULT_WEIGHTS):
        print(f"  {n_:>10}  {v:8.3f}   (was {d:g})")

    # held-out check: unseen seeds
    ho0 = evaluate(db, red, DEFAULT_WEIGHTS, seeds=(7, 8), t_preview=t_preview)
    ho1 = evaluate(db, red, w, seeds=(7, 8), t_preview=t_preview)
    print(f"\nheld-out seeds: hand-set {ho0:.4f} -> tuned {ho1:.4f} "
          f"({100*(ho0-ho1)/max(abs(ho0),1e-9):+.1f}%)")
    generalises = ho1 < ho0

    json.dump(dict(weights=list(map(float, w)), names=WEIGHT_NAMES,
                   score=fz, baseline=f0, holdout_tuned=ho1,
                   holdout_baseline=ho0, history=hist), open(out, "w"),
              indent=1)
    print(f"\n  [{'PASS' if fz < f0 else 'FAIL'}]  tuning improves training score")
    print(f"  [{'PASS' if generalises else 'FAIL'}]  improvement holds on "
          f"unseen wave realisations")
    print(f"  M6 WEIGHT-TUNING GATE: "
          f"{'PASSED' if (fz < f0 and generalises) else 'FAILED'}")
    return w


if __name__ == "__main__":
    main()
