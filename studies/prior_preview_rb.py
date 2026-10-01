#!/usr/bin/env python3
"""
Prior-predictive preview of the rigid-body operator family (M16;
learn/meta/operators_rb.py, PRIOR_DERIVATION.md D8.12). STANDALONE: no data
pipeline, no training data are written. Descriptive only (no statement
about information or ceilings).

Two parts.

REPLAY (default; critique item 19: a SIZE check, biased on the M15
statistics). Recorded low-fidelity episodes of the meta3 train split are
walked step by step. At every control step, from the RECORDED state, the
reduced model is stepped over the 6 substeps twice: with the episode's
recorded push (RR + RN, the old family) and with the new operator's push
(the rigid-body part at every substep from the substep's state and the
episode's own sea, rebuilt from its meta; the residual / noise part held).
The new family's one-step error is

    e_rb = recorded e0 + (state after the step with the new push
                          - state after the step with the recorded push) / step

so it has the low-fidelity boat's own wave response and actuators exactly
as recorded, and the new push's one-step effect in place of the old one.
THIS IGNORES THE CLOSED-LOOP EFFECT OF THE PUSH ON THE TRAJECTORY: every
step restarts from the recorded state, which the new push never moved (a
boat with more damping is paired with a trajectory of one without), so
correlations of e with the velocities are shifted further than they would
be in closed loop. Substep actuator positions and poses are interpolated
between the recorded control steps; the operator's wave inputs are the
recorded step-start elevations (meta3 has no mid-step ones).

CLOSED (--closed N). N fresh episodes simulated in the reduced world with
the new operator in the loop (operators_rb.simulate_rb: data2's scenario,
dithers, speed PI and heading autopilot, training seas), with the M10
actuator family drawn per episode as data2's 'm10' does (--act ideal for
ideal actuators). e0 is data3.e0_from on the simulated states, so it
carries the actuator share as the recorded e0 does. With --components
every component is also run alone (critique item 11b).

The operator's rigid-body part reads the elevation rate along the MOVING
stations (review of M16 item 1), in both parts.

For each variant the per-episode statistics of studies/prior_check_m15.py
are computed on e0 channels 0-4, and the table gives the prior's 5 / 50 / 95%
next to the old family's (studies/_cache/meta3/prior_check_m15.log, 2279
episodes; and the same recorded episodes replayed here, 'old*'), the target
median (C + Cb) and the target episodes' median percentile in the new prior.

    python studies/prior_preview_rb.py [--n-replay 150] [--closed 96]
                                       [--components] [--n-comp 40]
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from learn.meta import operators_rb as R
from learn.meta.operators import clip_push

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="meta3")
ap.add_argument("--n-replay", type=int, default=150)
ap.add_argument("--closed", type=int, default=96)
ap.add_argument("--components", action="store_true")
ap.add_argument("--n-comp", type=int, default=40)
ap.add_argument("--T", type=int, default=375)
ap.add_argument("--out", default=None)
ap.add_argument("--act", default="m10", choices=("m10", "ideal"),
                help="closed loop: the M10 actuator family (data2's 'm10', "
                     "lofi.draw_act per episode) or ideal actuators")
args = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "studies", "_cache", args.cache)
CH = R.CHANNELS
VEL = (6, 7, 11, 8, 10)
VARIANTS = ("all",) + R.COMPS + ("T8", "T9")


# ------------------------------------------------ prior_check_m15's stats
def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-30))


def slope(y, x):
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / (x @ x + 1e-30) * (x.std() + 1e-30)
                 / (y.std() + 1e-30))


def stats_one(e, xs):
    """prior_check_m15.stats_of for one episode: e (n, 5) from step 40,
    xs (n, 14) the matching plant states."""
    if len(e) < 60 or not np.isfinite(e).all():
        return None
    ok = e.std(0) > 1e-9          # a channel a component never touches
    r = {}
    for c, nm in enumerate(CH):
        r[f"rms_{nm}"] = float(np.sqrt((e[:, c] ** 2).mean()))
        r[f"ac1_{nm}"] = corr(e[1:, c], e[:-1, c])
        r[f"damp_{nm}"] = corr(e[:, c], xs[:, VEL[c]])
    r["sync_sy"] = corr(e[:, 1], e[:, 2])
    r["sync_hp"] = corr(e[:, 3], e[:, 4])
    r["slope_th"] = slope(e[:, 4], xs[:, 4])
    r["slope_u"] = slope(e[:, 4], xs[:, 6])
    # statistics of an all-zero channel are undefined (NaN, left out of
    # the quantiles); its rms stays 0
    for c, nm in enumerate(CH):
        if not ok[c]:
            r[f"ac1_{nm}"] = r[f"damp_{nm}"] = np.nan
    if not (ok[1] and ok[2]):
        r["sync_sy"] = np.nan
    if not (ok[3] and ok[4]):
        r["sync_hp"] = np.nan
    if not ok[4]:
        r["slope_th"] = r["slope_u"] = np.nan
    return r


def split_stats(split, n_max):
    d = np.load(os.path.join(C, f"{split}.npz"))
    e0 = np.load(os.path.join(C, f"{split}_e0.npz"))["E0"]
    L, XS = d["len"], d["XS"]
    n_e = L - 1 if XS.shape[1] == d["U"].shape[1] else L
    rows = []
    for i in range(min(n_max, len(n_e))):
        n = int(n_e[i])
        if n < 100:
            continue
        r = stats_one(e0[i, 40:n, :5].astype(float),
                      XS[i, 40:n].astype(float))
        if r is not None:
            rows.append(r)
    return rows


def old_log():
    """The old family's quantiles from prior_check_m15.log."""
    out = {}
    p = os.path.join(C, "prior_check_m15.log")
    if not os.path.exists(p):
        return out
    for line in open(p):
        f = line.split()
        if len(f) >= 5 and (f[0].startswith(("rms_", "ac1_", "damp_",
                                             "sync_", "slope_"))):
            out[f[0]] = tuple(float(x) for x in f[1:4])
    return out


# --------------------------------------------------------------- replay
def replay(lib, env, n_eps):
    d = np.load(os.path.join(C, "train.npz"))
    Lr = d["len"]
    meta = pickle.load(open(os.path.join(C, "train_meta.pkl"), "rb"))
    pick = [i for i in range(len(Lr)) if Lr[i] >= 200
            and meta[i].get("finite", True)][:n_eps]
    XS = d["XS"][pick].astype(np.float64)
    S = d["S"][pick].astype(np.float64)
    U = d["U"][pick].astype(np.float64)
    RP = (d["RR"][pick] + d["RN"][pick]).astype(np.float64)
    del d
    E0 = np.load(os.path.join(C, "train_e0.npz"))["E0"][pick, :, :5] \
        .astype(np.float64)
    red, dt, sub = env["red"], env["dt"], env["sub"]
    dtc = dt * sub
    V = len(VARIANTS)
    mask = np.zeros((V, len(R.COMPS) + 2))
    names = list(R.COMPS) + ["T8", "T9"]
    for v, nm in enumerate(VARIANTS):
        if nm == "all":
            mask[v] = 1.0
        else:
            mask[v, names.index(nm)] = 1.0
    res = {nm: [] for nm in ("old*",) + VARIANTS}
    clipc = np.zeros(5)
    nsub = 0
    rms_push = {nm: [] for nm in VARIANTS}
    t0 = time.time()
    wave_err = 0.0
    for b, i in enumerate(pick):
        n = int(Lr[i]) - 1
        m = meta[i]
        op = R.OperatorRB(m["op_seed"], lib, dt=dtc, L=env["L"],
                          relay=m["relay"])
        sea = R.sea_state(m["sea"], int(m["seed"]) + 50000)
        seas = R.RowSeas([sea], env["x_st"], env["y_off"])
        X28 = np.concatenate([S[b], U[b]], 1)
        nrng = np.random.default_rng([m["op_seed"], 1])
        st = op.new_state(1, rng=nrng, init_s=X28[:1], rb_rows=V)
        E = np.zeros((V, n, 5))
        pushes = np.zeros((V, n, 5))
        for k in range(n):
            a14, b14 = XS[b, k], XS[b, k + 1]
            rr, nz = op.step(st, X28[k][None], noise=True, rng=nrng)
            rr, nz = clip_push(rr[0], nz[0])
            hr = mask[:, -2:-1] * rr[None]
            hn = mask[:, -1:] * nz[None]
            s0 = R.to_reduced(a14[None])
            sr = np.repeat(s0, V + 1, 0)
            for j in range(sub):
                f = (j + 0.5) / sub
                thr = a14[12] + (b14[12] - a14[12]) * f
                noz = a14[13] + (b14[13] - a14[13]) * f
                g = j / sub
                dpsi = (b14[5] - a14[5] + np.pi) % (2 * np.pi) - np.pi
                pose = (np.array([a14[0] + (b14[0] - a14[0]) * g]),
                        np.array([a14[1] + (b14[1] - a14[1]) * g]),
                        np.array([a14[5] + dpsi * g]))
                # station rates along the moving hull (the recorded
                # velocities interpolated like the pose)
                vel = tuple(np.array([a14[c] + (b14[c] - a14[c]) * g])
                            for c in (6, 7, 11))
                eta, etad = seas.stations(*pose, np.array([k * dtc + j * dt]),
                                          vel=vel)
                if j == 0 and k < 50:
                    wave_err = max(wave_err, float(np.abs(
                        eta.reshape(15) - S[b, k, 11:26]).max()))
                parts = op.rb_accel(st, sr[1:], thr, noz,
                                    np.repeat(eta, V, 0),
                                    np.repeat(etad, V, 0), dt, parts=True)
                rb = np.stack([parts[c] for c in R.COMPS], 1)  # (V, 6, 5)
                rb = (rb * mask[:, :len(R.COMPS), None]).sum(1)
                rc, nc = clip_push(rb + hr, hn)
                push = rc + nc
                clipc += (np.abs(rb[0] + hr[0] + hn[0]) > 3 * R.A_REF)
                nsub += 1
                sr = red.step(sr, np.full(V + 1, thr), np.full(V + 1, noz),
                              np.repeat(eta, V + 1, 0), env["x_st"], dt)[0]
                sr[0, list(R.RED_VEL)] += RP[b, k] * dt
                sr[1:, list(R.RED_VEL)] += push * dt
                pushes[:, k] += push / sub
            E[:, k] = E0[b, k] + (sr[1:, list(R.RED_VEL)]
                                  - sr[0, list(R.RED_VEL)]) / dtc
        xs = XS[b, 40:n]
        r = stats_one(E0[b, 40:n], xs)
        if r is not None:
            res["old*"].append(r)
        for v, nm in enumerate(VARIANTS):
            r = stats_one(E[v, 40:], xs)
            if r is not None:
                res[nm].append(r)
            rms_push[nm].append(np.sqrt((pushes[v, 40:] ** 2).mean(0)))
        if b % 25 == 24:
            print(f"    replay {b + 1}/{len(pick)} ({time.time() - t0:.0f} s)",
                  flush=True)
    print(f"  replay: {len(pick)} episodes, {time.time() - t0:.0f} s; rebuilt"
          f" sea vs recorded step-start elevations: max |diff| "
          f"{wave_err:.1e} m")
    print("  push size (per-episode rms, median) by variant:")
    for nm in VARIANTS:
        q = np.median(np.array(rms_push[nm]), 0)
        print(f"    {nm:<6} " + "  ".join(f"{c} {x:.2f}" for c, x in
                                          zip(CH, q)))
    print(f"  clip rate of the full push (substeps): "
          + ", ".join(f"{c} {x:.1e}" for c, x in zip(CH, clipc / max(nsub,
                                                                    1))))
    return res


# --------------------------------------------------------------- closed
def closed(lib, env, n_eps, variants, seed0=777):
    from learn.meta.data3 import e0_from
    res = {}
    G = 32
    for nm in variants:
        rows_all = []
        t0 = time.time()
        for g0 in range(0, n_eps, G):
            B = min(G, n_eps - g0)
            rng = np.random.default_rng([seed0, g0])
            ops = [R.OperatorRB(int(seed0 * 1000 + g0 + i), lib,
                                dt=env["dt"] * env["sub"], L=env["L"])
                   for i in range(B)]
            for o in ops:
                o.enabled = set(R.ALL) if nm == "all" else (
                    set() if nm == "none" else {nm})
            groups = [dict(op=o, st=o.new_state(1, rng=np.random.default_rng(
                [o.seed, 1])), rows=np.array([i]),
                rng=np.random.default_rng([o.seed, 1]))
                for i, o in enumerate(ops)]
            seas = R.RowSeas([R.sea_state(R.sea_dict(np.random.default_rng(
                [seed0, g0, i])), 50000 + g0 + i) for i in range(B)],
                env["x_st"], env["y_off"])
            xs = R.start_states(env, B, rng)
            act = None
            if args.act == "m10":
                from sim import lofi
                ds = [lofi.draw_act(np.random.default_rng([o.seed, 11]),
                                    env["red"].p, env["dt"]) for o in ops]
                act = {k: np.array([d[k] for d in ds])
                       for k in lofi.ACT_KEYS}
            out = R.simulate_rb(env, xs, R.Autopilot(env, B, rng),
                                np.zeros(B), seas, groups, args.T, act=act)
            XS, UU = out["XS"], out["U"]
            for i in range(B):
                if not np.isfinite(XS[i]).all():
                    continue
                e0 = e0_from(env, XS[i, :-1], XS[i, 1:], UU[i])[:, :5]
                r = stats_one(e0[40:], XS[i, 40:-1])
                if r is not None:
                    rows_all.append(r)
        res[nm] = rows_all
        print(f"  closed '{nm}': {len(rows_all)} episodes, "
              f"{time.time() - t0:.0f} s", flush=True)
    return res


# ---------------------------------------------------------------- table
def table(title, prior_sets, target, old):
    keys = list(target[0])
    print(f"\n{title}")
    hdr = f"  {'statistic':<11} {'old (log) 5/50/95':>23}"
    for nm in prior_sets:
        hdr += f" | {nm + ' 5/50/95':>23}"
    hdr += " | target med | target pct in: " + " ".join(prior_sets)
    print(hdr)
    for k in keys:
        t = np.array([r[k] for r in target])
        line = f"  {k:<11} "
        o = old.get(k)
        line += (f"{o[0]:+7.2f} {o[1]:+7.2f} {o[2]:+7.2f}" if o else
                 " " * 23)
        pcts = []
        for nm, rows in prior_sets.items():
            p = np.array([r[k] for r in rows])
            p = p[np.isfinite(p)]
            if len(p) < 3:
                line += " | " + " " * 23
                pcts.append("  -")
                continue
            q = np.quantile(p, [0.05, 0.5, 0.95])
            line += f" | {q[0]:+7.2f} {q[1]:+7.2f} {q[2]:+7.2f}"
            pct = np.array([(p < v).mean() for v in t]) * 100
            pcts.append(f"{np.median(pct):5.1f}% ({(pct < 5).mean():.2f}/"
                        f"{(pct > 95).mean():.2f})")
        line += f" | {np.median(t):+9.3f} | " + "  ".join(pcts)
        print(line)


def main():
    from learn.meta import relabel
    Lb = np.load(os.path.join(C, "lib.npz"))
    lib = {k: Lb[k] for k in Lb.files}
    env = relabel._env()
    target = split_stats("C", 64) + split_stats("Cb", 64)
    old = old_log()
    print(f"target: {len(target)} episodes (C + Cb)")
    if args.n_replay > 0:
        rp = replay(lib, env, args.n_replay)
        table("REPLAY on recorded meta3 train episodes (size check; ignores "
              "the push's closed-loop effect on the trajectory)",
              {"old*": rp["old*"], "rb all": rp["all"]}, target, old)
        comp = {nm: rp[nm] for nm in R.COMPS + ("T8", "T9")}
        table("REPLAY, each component alone (+ the recorded low-fidelity "
              "response)", comp, target, old)
    if args.closed > 0:
        cl = closed(lib, env, args.closed, ("all",))
        table(f"CLOSED LOOP in the reduced world ({args.act} actuators)",
              {"rb all": cl["all"]}, target, old)
    if args.components:
        cc = closed(lib, env, args.n_comp, ("none",) + R.COMPS
                    + ("T8", "T9"))
        table(f"CLOSED LOOP, each component alone ({args.n_comp} episodes "
              "each; 'none' = low-fidelity boat alone)", cc, target, old)


if __name__ == "__main__":
    main()
