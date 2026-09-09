#!/usr/bin/env python3
"""
How good could ANY thrust policy be -- and is the search good enough to say?

Two probes have failed to beat constant thrust at equal speed: a 90 s-horizon
MPC and a hand-built envelope scheduler. Neither licenses "no policy can".

The natural next step is a non-causal oracle: hand it the entire episode's sea
and let it tune the whole throttle curve directly against the operator
objective, on the real 110-state plant, per seed. Structurally that dominates
any causal policy, because a causal policy only ever sees a prefix of what this
sees, and this is additionally allowed to overfit each individual sea.

BUT THE STRUCTURE IS NOT THE PROBLEM, THE SEARCH IS. Writing J* for the true
optimum and J_cma for what CMA-ES returns, J_cma >= J*. So a flat result bounds
the optimum from the WRONG SIDE: it is entirely consistent with J* << J_flat and
the optimiser simply failing. "How to use the information" is itself a policy,
and an optimiser that finds nothing has demonstrated nothing.

So this runs two controls alongside the measurement:

  POSITIVE CONTROL -- the same oracle, same budget, same parameterisation, on a
    sea where schedule shape is PREDICTED to matter. At 4.5 m/s this vessel
    resonates at wave periods of 2.9-3.7 s, where the response stops being
    quasi-static and changing speed genuinely detunes the encounter frequency.
    If the optimiser cannot find a gain there, it cannot be trusted to have
    found the absence of one at Tp = 9.7 s.

  SEARCH CONTROL -- pure random search with an identical evaluation budget. If
    CMA-ES does not clearly beat random sampling, it is not searching.

Only if the positive control lights up and CMA-ES beats random search does the
null at Tp = 9.7 s mean anything.

Run: python -m studies.oracle_ceiling
"""
import json
import numpy as np

from hydro import bem
from sim.env import FREEBOARD, score
from sim.vessel import NonlinearVessel
from sim.wavefield import SeaState
from sim import seakeeping
from learn.tune import cma_es

DB = "hydro_wigley_10m.npz"
G = 9.81
SEEDS = (0, 1, 2)
T_END = 200.0
DT = 0.05
N_KNOT = 10             # one free choice every 20 s -- finer than a wave group
U_PENALTY = 40.0
THRUST_LO, THRUST_HI = 1500.0, 12000.0
TRIM = 7000.0

# (label, Hs, Tp, expectation)
CONDITIONS = [
    ("SS5 swell   Tp 9.7 s", 3.25, 9.7,
     "quasi-static, we/wn = 0.27 -- expect nothing"),
    ("wind chop   Tp 3.5 s", 1.20, 3.5,
     "near resonance, we/wn ~ 1 -- expect a real gain"),
]


def simulate(db, seed, knots, hs, tp):
    sea = SeaState(hs, tp, n_freq=32, n_dir=1, seed=seed)
    v = NonlinearVessel(db, sea, L=db.L, B=float(db.attrs.get("B", 2.5)),
                        T=float(db.attrs.get("T", 0.8)), dt=DT)
    n = int(T_END / DT)
    sched = np.interp(np.arange(n), np.linspace(0, n - 1, len(knots)), knots)
    s = v.initial_state(2.0)
    acc = np.empty(n); spd = np.empty(n); rel = np.empty(n); sf = np.empty(n)
    t = 0.0
    for i in range(n):
        s = v.step(s, t, float(sched[i]), 0.0, DT)
        acc[i] = v.last_bow_acc; spd[i] = s[6]
        rel[i] = v.last_rel_bow; sf[i] = v.last_slam_force
        t += DT
    m = seakeeping.summarise(rel, acc / G, DT, v.T, FREEBOARD, v.v_slam)
    m.update(u_mean=float(np.mean(spd[n // 4:])), cross_rms=0.0, finite=True,
             slam_impulse=float(sf.sum() * DT / (t / 60.0) / 1e3),
             slam_p99=float(np.percentile(sf, 99) / 1e3),
             slam_per_min=float(v.slam_count / (t / 60.0)))
    return m


def denorm(z):
    return THRUST_LO + 0.5 * (np.clip(z, -1, 1) + 1) * (THRUST_HI - THRUST_LO)


def flat_z():
    return np.full(N_KNOT, 2 * (TRIM - THRUST_LO) / (THRUST_HI - THRUST_LO) - 1)


def make_obj(db, seed, hs, tp, u_target):
    def f(z):
        m = simulate(db, seed, denorm(z), hs, tp)
        # pin the speed: otherwise the oracle wins by going slower, every time,
        # and measures nothing but the speed/comfort trade already known
        return score(m, u_target) + U_PENALTY * (m["u_mean"] - u_target) ** 2
    return f


def random_search(f, n_eval, seed):
    """Same budget, no adaptation. If CMA-ES cannot beat this, it is not
    searching and its null result carries no information."""
    rng = np.random.default_rng(seed + 7717)
    best = np.inf
    x0 = flat_z()
    for _ in range(n_eval):
        z = np.clip(x0 + rng.normal(0.0, 0.35, N_KNOT), -1, 1)
        best = min(best, f(z))
    return float(best)


def run_condition(db, label, hs, tp, note, n_gen, popsize):
    print(f"\n  {label}   ({note})")
    n_eval = n_gen * popsize
    rows = []
    for sd in SEEDS:
        base = simulate(db, sd, denorm(flat_z()), hs, tp)
        u_t = base["u_mean"]
        f = make_obj(db, sd, hs, tp, u_t)
        j0 = f(flat_z())
        z, jz, hist = cma_es(f, flat_z(), sigma0=0.35, n_gen=n_gen,
                             popsize=popsize, seed=sd, verbose=False)
        jr = random_search(f, n_eval, sd)
        best = simulate(db, sd, denorm(z), hs, tp)
        rows.append(dict(seed=sd, hs=hs, tp=tp, j_flat=float(j0),
                         j_cma=float(jz), j_rand=float(jr),
                         u_flat=float(u_t), u_cma=float(best["u_mean"]),
                         thrust_std=float(np.std(denorm(z))),
                         gain=100 * (j0 - jz) / abs(j0),
                         gain_rand=100 * (j0 - jr) / abs(j0),
                         hist=[float(h[3]) for h in hist]))
        r = rows[-1]
        print(f"    seed {sd}:  flat {j0:.4f}  cma {jz:.4f} ({r['gain']:+.1f}%)"
              f"   random {jr:.4f} ({r['gain_rand']:+.1f}%)"
              f"   u {u_t:.2f}->{best['u_mean']:.2f}"
              f"   thrust sd {r['thrust_std']:.0f} N")
    g = np.array([r["gain"] for r in rows])
    gr = np.array([r["gain_rand"] for r in rows])
    print(f"    mean gain: CMA-ES {g.mean():+.1f}%   random {gr.mean():+.1f}%")
    return rows, float(g.mean()), float(gr.mean())


def main(n_gen=20, popsize=10):
    db = bem.load(DB)
    print(f"non-causal oracle, {len(SEEDS)} seeds x {T_END:.0f} s, "
          f"{N_KNOT} knots, {n_gen*popsize} evaluations per seed")
    out, gains = {}, {}
    for label, hs, tp, note in CONDITIONS:
        rows, g, gr = run_condition(db, label, hs, tp, note, n_gen, popsize)
        out[label] = rows
        gains[label] = (g, gr)

    swell = gains[CONDITIONS[0][0]]
    chop = gains[CONDITIONS[1][0]]
    print("\n  " + "-" * 66)
    print(f"  {'condition':<24}{'CMA-ES':>10}{'random':>10}{'verdict':>22}")
    for label, _, _, _ in CONDITIONS:
        g, gr = gains[label]
        v = "search works" if g > gr + 1.0 else "search no better than random"
        print(f"  {label:<24}{g:>+9.1f}%{gr:>+9.1f}%{v:>22}")

    print("\n  " + "-" * 66)
    # The decision rule as first written asked only whether CMA-ES beat random
    # by a percentage point. That is the wrong question. What matters is what
    # FRACTION of the measured gain random search already reproduces: if it
    # reproduces most of it, the gain is the best-of-N order statistic on one
    # fixed sea, not exploitable structure. Measured: random reaches 78% of
    # CMA-ES in both conditions, so almost all of it is luck.
    frac = [gains[c[0]][1] / max(gains[c[0]][0], 1e-9) for c in CONDITIONS]
    luck = max(frac)
    searched = chop[0] > chop[1] + 1.0 and chop[0] > 3.0
    print(f"  random search reproduces {100*luck:.0f}% of the optimiser's gain")
    if luck > 0.5:
        print()
        print("  THE MEASUREMENT IS VOID. Most of the 'gain' is the best of N")
        print("  random draws on one fixed sea -- an order statistic, not")
        print("  structure. A non-causal per-realisation optimum overfits")
        print("  without bound BY CONSTRUCTION, so it cannot bound what a")
        print("  causal policy can achieve, and this whole framing was wrong.")
        print("  The only valid test is a CAUSAL policy scored on HELD-OUT")
        print("  seas, which is what sim/rl_env.py exists for.")
    elif not searched:
        print("  POSITIVE CONTROL FAILED. The optimiser found no gain even where")
        print("  the physics says one exists, so its null on the swell case is")
        print("  a statement about the optimiser, not about the vessel. Neither")
        print("  number below can be used.")
    elif swell[0] < 3.0:
        print(f"  Positive control lights up ({chop[0]:+.1f}% in chop), and the same")
        print(f"  optimiser with the same budget finds {swell[0]:+.1f}% in swell.")
        print("  The search is competent, so the swell null is about the vessel:")
        print("  at Tp = 9.7 s there is nothing for a thrust policy to exploit,")
        print("  and that now rests on a demonstrated ability to find signal")
        print("  rather than on a failure to find it.")
    else:
        print(f"  Both conditions show a gain ({swell[0]:+.1f}% swell, "
              f"{chop[0]:+.1f}% chop).")
        print("  The earlier probes missed exploitable structure; RL is worth it.")
    json.dump(dict(gains={k: list(v) for k, v in gains.items()}, runs=out),
              open("oracle_ceiling.json", "w"), indent=1, default=float)
    return gains


if __name__ == "__main__":
    main()
