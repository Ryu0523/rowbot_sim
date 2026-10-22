#!/usr/bin/env python3
"""
How close does the catalogue family come to the gz target's error law, and
how much prior mass lies near it (learn/meta/ROADMAP_2026-09-30.md section
11.13; the comparison's computation (F) 8)?

For a law theta of the family (operators_cat.OperatorCat, the D10 prior the
pretrained net was trained on), its one-step error on the gz states is
computed TEACHER-FORCED: from every recorded gz state s_k, one control step
of the family's source world (operators_cat.simulate_cat's substep loop:
the reduced model with the gz sea at its 5 x 3 stations, the law's
control-step part held and its substep push added after every 0.04-s
substep, the impulse channel with its per-step cap), with the gz actuator
POSITIONS as the applied actuators (linear in time between the recorded
values at the step's ends: an M10 draw that reproduces the gz actuator, the
comparison's matching condition C.4), noise off. Then

    e0_theta,k = (P s^src_{k+1} - M(P s_k, U_k)) / 0.24       (data3.e0_from)

on the five velocity channels, against gz's e0_k from the same s_k. The
law's hidden states run along the gz trajectory (op.step with the recorded
inputs every control step), so every law sees the true states, sea and
commands: e0_theta is the best one-step prediction the law itself could
make, with full knowledge of the state (an oracle for its own hidden
states up to their start). Scores per channel, pooled over steps:

    D_c(theta) = sum_k (e0_gz - e0_theta)^2 / sum_k e0_gz^2

(the network's skill convention: 1 = no better than predicting 0). Also
the baseline with no push (law = 0: the reduced model's own wave response
and the gz actuator positions) and, per law, the relaxed refit: one free
non-negative scale per part of the law (operators_cat PARTS, step means),
least squares on e0_gz - e0_base (the propagation within the step taken
as linear; a relaxation of the family's own forms, not a predictor).

Reported: the distribution of D over N prior draws (how much prior mass
lies near gz), the best draw, and the refit of every draw. Every law's
e0_theta is saved (closure.pkl) for studies/family_hull.py, the best
non-negative mixture of the laws (the ideal predictor's posterior mean lies
in their convex hull). Compare with
the network on the same missions: frozen skill, and the copy trained on 6
gz missions (an upper bound on gz's information floor).

    python -m studies.family_closure --n-laws 200 --procs 4
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
import numpy as np   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_cache", "family_closure")
C7 = os.path.join(HERE, "_cache", "meta7")
CH = ("surge", "sway", "yaw", "heave", "pitch")
SEED0 = 8_000_000                # no training law
_W = {}


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def _init(keys):
    from studies.theory_checks import gz_trajs
    from learn.meta import data3
    from learn.meta.data2 import plant_to_reduced
    Lb = np.load(os.path.join(C7, "lib.npz"))
    xs, U, t0, seas, env = gz_trajs(keys)
    B, n = U.shape[:2]
    a = plant_to_reduced(xs[:, :-1].reshape(-1, 14))
    m0 = data3.model0_step(env, a, U.reshape(-1, 2)).reshape(B, n, 10)
    e0 = np.stack([data3.e0_from(env, xs[b, :-1], xs[b, 1:], U[b])[:, :5]
                   for b in range(B)])
    _W.update(lib={k: Lb[k] for k in ("mu", "sd", "S")}, xs=xs, U=U, t0=t0,
              seas=seas, env=env, m0=m0[:, :, list(data3.E7[:5])], e0=e0)


def one_step(op, seed, parts=False):
    """Teacher-forced one-step e0 of law op on every gz state: (B, n, 5),
    and the per-part step-mean pushes {part: (B, n, 5)} if parts."""
    from learn.meta import cat_base as CB
    from learn.meta import data3
    from learn.meta import operators_cat as OC
    from learn.meta import operators_rb as R
    from learn.meta.operators import clip_push
    xs, U, t0, seas, env = (_W[k] for k in ("xs", "U", "t0", "seas", "env"))
    red, dt, sub = env["red"], env["dt"], env["sub"]
    dtc = dt * sub
    B, n = U.shape[:2]
    rv = list(OC.RED_VEL)
    cap = OC.LIM * dtc
    st = None
    if op is not None:
        st = op.new_state(B, rng=np.random.default_rng([seed, 1]))
        op.bind_sea(st, seas)
        cs = CB.CatSea(seas, op.ctx)
    pred = np.zeros((B, n, 5))
    PT = {}
    e7 = list(data3.E7[:5])
    for k in range(n):
        t = t0 + k * dtc
        xs_now = xs[:, k]
        sr = R.to_reduced(xs_now)
        Uk = U[:, k]
        held = np.zeros((B, 5))
        if op is not None:
            w15 = R.mid_waves(seas, sr, t, 0.5 * dtc)
            rr, _ = op.step(st, R.s28_of(env, xs_now, w15, Uk), noise=False)
            held, _ = clip_push(rr, np.zeros_like(rr))
        cmd2 = np.stack([Uk[:, 0] * env["t_max"], Uk[:, 1] * env["rud_max"]],
                        1)
        a0, a1 = xs[:, k, 12:14], xs[:, k + 1, 12:14]
        isum = np.zeros((B, 5))
        for kk in range(sub):
            tt = t + kk * dt
            app = a0 + (kk + 0.5) / sub * (a1 - a0)
            thr, noz = app[:, 0], app[:, 1]
            eta, _ = seas.stations(sr[:, 0], sr[:, 1], sr[:, 7], tt)
            ns = red.step(sr, thr, noz, eta, env["x_st"], dt)[0]
            if op is not None:
                nu0 = (ns[:, rv] - sr[:, rv]) / dt
                o = op.cat_accel(st, sr, thr, noz, nu0, cs.sample(sr, tt), dt,
                                 cmd=cmd2, t=tt, parts=parts)
                rc, _ = clip_push(o["acc"] + held, np.zeros((B, 5)))
                im = np.asarray(o["imp"], float)
                capd = np.clip(isum + im, -cap, cap)
                ns[:, rv] += rc * dt + (capd - isum)
                isum = capd
                if parts:
                    for key, v in o["PARTS"].items():
                        acc_ = PT.setdefault(key, np.zeros((B, n, 5)))
                        acc_[:, k] += (v / dt if key.endswith(":imp")
                                       else v) / sub
                    hp = PT.setdefault("held", np.zeros((B, n, 5)))
                    hp[:, k] += held / sub
            sr = ns
        pred[:, k] = sr[:, e7]
    e0t = (pred - _W["m0"]) / dtc
    return e0t, PT


def scores(e0t):
    e0 = _W["e0"]
    num = ((e0 - e0t) ** 2).sum((0, 1))
    return num / (e0 ** 2).sum((0, 1))


def refit(e0b, PT):
    """Non-negative scale per part, least squares on e0 - e0_base over all
    channels (each channel weighted by 1 / sum e0^2 so the score is the
    mean of D_c): returns D per channel after the refit."""
    from scipy.optimize import nnls
    e0 = _W["e0"]
    w = 1.0 / np.sqrt((e0 ** 2).sum((0, 1)))
    keys = [k for k in PT if np.abs(PT[k]).max() > 0]
    if not keys:
        return scores(e0b), {}
    A = np.stack([(PT[k] * w).reshape(-1) for k in keys], 1)
    y = ((e0 - e0b) * w).reshape(-1)
    beta, _ = nnls(A, y)
    fit = e0b + sum(b * PT[k] for b, k in zip(beta, keys))
    return scores(fit), dict(zip(keys, beta))


def law_job(i):
    from learn.meta import operators_cat as OC
    seed = SEED0 + i
    t = time.time()
    try:
        op = OC.OperatorCat(seed, _W["lib"], dt=0.24, L=5.4, accept=False)
        e0t, PT = one_step(op, seed, parts=True)
        d = scores(e0t)
        dr, beta = refit(_W["e0b"], PT)
        items = sorted(op.on)
        return dict(i=i, seed=seed, D=d, D_refit=dr, items=items,
                    tier=op.cat.tier, beta=beta, e0t=e0t.astype(np.float32),
                    finite=bool(np.isfinite(e0t).all()), s=time.time() - t)
    except Exception as ex:                        # keep the batch going
        import traceback
        return dict(i=i, seed=seed, error=repr(ex) + " | "
                    + traceback.format_exc()[-400:])


def _pool_init(keys, e0b):
    import warnings
    warnings.filterwarnings("ignore")
    _init(keys)
    _W["e0b"] = e0b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", default="4-7")
    ap.add_argument("--n-laws", type=int, default=200)
    ap.add_argument("--law0", type=int, default=0,
                    help="first law index (seed SEED0 + index)")
    ap.add_argument("--out", default="closure.pkl")
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    from studies import onestep_gz as G
    keys = G.parse_missions(args.missions)
    os.makedirs(OUT, exist_ok=True)
    _init(keys)
    e0b, _ = one_step(None, 0)
    db = scores(e0b)
    _W["e0b"] = e0b
    log(f"{len(keys)} gz missions, {_W['U'].shape[1]} steps each; e0 rms "
        f"{np.round(np.sqrt((_W['e0'] ** 2).mean((0, 1))), 3)}; baseline (no "
        f"push: reduced model + its wave response + the gz actuator "
        f"positions) D {np.round(db, 3)}")
    from concurrent.futures import ProcessPoolExecutor
    res = []
    t0 = time.time()
    with ProcessPoolExecutor(args.procs, initializer=_pool_init,
                             initargs=(keys, e0b)) as ex:
        for r in ex.map(law_job, range(args.law0,
                                       args.law0 + args.n_laws)):
            res.append(r)
            if len(res) % 20 == 0:
                log(f"  {len(res)}/{args.n_laws} laws ({time.time() - t0:.0f}"
                    " s)")
    pickle.dump(dict(keys=keys, base=db, laws=res, e0=_W["e0"], e0b=e0b),
                open(os.path.join(OUT, args.out), "wb"))
    ok = [r for r in res if "error" not in r and r["finite"]]
    bad = [r for r in res if "error" in r]
    log(f"{len(ok)} laws scored, {len(bad)} errors"
        + (f" (first: {bad[0]['error'][:300]})" if bad else ""))
    D = np.array([r["D"] for r in ok])
    Dr = np.array([r["D_refit"] for r in ok])
    for lab, A in (("as drawn", D), ("relaxed refit", Dr)):
        log(f"D per channel [surge sway yaw heave pitch], {lab}: quantiles "
            "10/50/90 % " + " | ".join(
                f"{CH[c]} {np.round(np.quantile(A[:, c], [.1, .5, .9]), 3)}"
                for c in range(5)))
        log(f"  best law per channel {np.round(A.min(0), 3)}; share of laws "
            f"better than the baseline per channel "
            f"{np.round((A < db[None]).mean(0), 3)}")
        j = int(np.argmin(A.mean(1)))
        log(f"  best law on the mean over channels: seed {ok[j]['seed']} "
            f"D {np.round(A[j], 3)} tier {ok[j]['tier']} items "
            f"{ok[j]['items']}")


if __name__ == "__main__":
    main()
