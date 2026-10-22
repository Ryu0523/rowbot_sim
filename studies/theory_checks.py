#!/usr/bin/env python3
"""
Checks of three assumptions of the theory in learn/meta/ROADMAP_2026-09-30.md
section 11.11 (each check measures an assumption, not an outcome):

  info      how many directions of plant variation one 256-step window can
            identify, and how much of the window's information comes right
            after turns. The directions are the catalogue family's own: the
            contribution of each active part of a law (catalogue items, the
            propulsion chain, layer-1 terms, the generic layer 3) taken as
            'this part stronger or weaker', evaluated open loop along
            recorded trajectories (eval_parts). For a window W, with Phi_t
            (5 x K) the parts' one-step velocity errors at step t, noise
            variances s2_c per channel and a prior relative spread s0 per
            part: J = s0^2 sum_t Phi_t' diag(1/s2) Phi_t, mu = eig(J), and
            the effective number of identified directions gamma = sum
            mu / (1 + mu) (MacKay 1992). Trajectories: the 16 recorded gz
            missions (hand controller) and each law's own closed-loop
            episode in the low-fidelity world (the data autopilot).
  density   how many of the laws the net was trained on (meta7 train
            generation records) lie near the gz structure: the six
            catalogue items with gz counterparts (W1 nonlinear lift, W3
            slam, W7 lateral crossflow, W9 wave slope surge, W10 bow
            damping, P1 jet suction) on, and strong (gain in the upper part
            of the item's own on-distribution).
  additive  whether catalogue items add: on a FIXED trajectory, the
            accelerations with item subsets switched on, all(S) - none
            against the sum over k of single(k) - none, and pairwise
            interactions among the six gz-like items.

Parts are evaluated with the trajectory fixed: per control step the
recorded state, then the low-fidelity model's own substeps (6 x 0.04 s)
with the recorded commands, each substep's catalogue substep (parts=True),
averaged over the control step; the next control step restarts from the
recorded state.

With --pred, info reports instead the identification term of the one-step
error (window_pred: the predictive variance the window's posterior over
the part strengths adds, over the noise variance); with --loo, additive
also reports each item's interaction with all the others.

    python -m studies.theory_checks --check density
    python -m studies.theory_checks --check info --n-laws 8
    python -m studies.theory_checks --check info --pred --n-laws 2
    python -m studies.theory_checks --check additive --n-laws 3 --loo
"""
import argparse
import os
import pickle
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
import numpy as np   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C7 = os.path.join(HERE, "_cache", "meta7")
CH = ("surge", "sway", "yaw", "heave", "pitch")
GZ_LIKE = ("W1", "W3", "W7", "W9", "W10", "P1")
GAIN = dict(W1="p_h", W3="C_m", W7="C_Dl", W9="mu_x", W10="kap_b", P1="A")
# one-step skill on gz of the copy fully fine-tuned on 6 gz missions
# (studies/_cache/onestep_gz/heads.log; its residual: a LOW noise estimate)
# and of the frozen pretrained net (a HIGH one)
FT_SKILL = np.array([0.022, 0.069, 0.098, 0.010, 0.125])
FROZEN_SKILL = np.array([0.111, 0.384, 0.487, 0.051, 0.354])


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


# ------------------------------------------------------------ evaluation
def make_op(seed, lib, **kw):
    from learn.meta import operators_cat as OC
    return OC.OperatorCat(int(seed), lib, dt=0.24, L=5.4, accept=False, **kw)


def eval_parts(op, xs, U, t0, seas, env, seed=0):
    """Open-loop parts along fixed trajectories: xs (B, n+1, 14) plant
    states, U (B, n, 2) command fractions, t0 (B,), seas an operators_rb
    RowSeas of the B rows. Returns ({part: (B, n, 5)} control-step mean
    accelerations, acc (B, n, 5))."""
    from learn.meta import cat_base as CB
    from learn.meta import operators_cat as OC
    from learn.meta import operators_rb as R
    red, dt, sub = env["red"], env["dt"], env["sub"]
    dtc = dt * sub
    B, n = U.shape[:2]
    rv = list(OC.RED_VEL)
    st = op.new_state(B, rng=np.random.default_rng([seed, 1]))
    op.bind_sea(st, seas)
    cs = CB.CatSea(seas, op.ctx)
    parts, acc = {}, np.zeros((B, n, 5))
    for k in range(n):
        t = t0 + k * dtc
        xs_now = xs[:, k]
        sr = R.to_reduced(xs_now)
        Uk = U[:, k]
        w15 = R.mid_waves(seas, sr, t, 0.5 * dtc)
        op.step(st, R.s28_of(env, xs_now, w15, Uk), noise=False)
        thr = Uk[:, 0] * env["t_max"]
        noz = Uk[:, 1] * env["rud_max"]
        cmd = np.stack([thr, noz], 1)
        for kk in range(sub):
            tt = t + kk * dt
            eta, _ = seas.stations(sr[:, 0], sr[:, 1], sr[:, 7], tt)
            ns = red.step(sr, thr, noz, eta, env["x_st"], dt)[0]
            nu0 = (ns[:, rv] - sr[:, rv]) / dt
            o = op.cat_accel(st, sr, thr, noz, nu0, cs.sample(sr, tt), dt,
                             cmd=cmd, t=tt, parts=True)
            acc[:, k] += o["acc"] / sub
            for key, v in o["PARTS"].items():
                a = parts.setdefault(key, np.zeros((B, n, 5)))
                a[:, k] += (v / dt if key.endswith(":imp") else v) / sub
            sr = ns
    return parts, acc


def gz_trajs(keys):
    from learn.meta import operators_rb as R
    from learn.meta import relabel
    env = relabel._env()
    S = pickle.load(open(os.path.join(HERE, "_cache", "onestep_gz",
                                      "streams_T120.pkl"), "rb"))
    n = min(int(S[k]["n"]) for k in keys)
    xs = np.stack([S[k]["xs"][:n + 1] for k in keys])
    U = np.stack([S[k]["u"][:n] for k in keys])
    t0 = np.array([float(S[k]["t"][0]) for k in keys])
    seas = R.RowSeas([SimpleNamespace(**{a: np.asarray(S[k]["sea"][a])
                                         for a in ("a", "k", "th", "w",
                                                   "phi")})
                      for k in keys], env["x_st"], env["y_off"])
    return xs, U, t0, seas, env


def own_traj(op, env, seed, n):
    """The law's own closed-loop episode in the low-fidelity world (the
    data autopilot, a training sea), as in studies/prior_preview_cat."""
    from learn.meta import operators_cat as OC
    from learn.meta import operators_rb as R
    rng = np.random.default_rng([seed, 7])
    groups = [dict(op=op, st=op.new_state(1, rng=np.random.default_rng(
        [seed, 1])), rows=np.array([0]), rng=np.random.default_rng([seed, 1]))]
    seas = R.RowSeas([R.sea_state(R.sea_dict(np.random.default_rng(
        [seed, 3])), 50000 + seed)], env["x_st"], env["y_off"])
    xs0 = R.start_states(env, 1, rng)
    res = OC.simulate_cat(env, xs0, R.Autopilot(env, 1, rng), np.zeros(1),
                          seas, groups, n, obs=False)
    m = n if res["div"][0] < 0 else int(res["div"][0])
    return res["XS"][:, :m + 1], res["U"][:, :m], np.zeros(1), seas, m


def mean_e2(env):
    """Per-channel mean squared one-step velocity error e0 (raw units) on
    the held-out gz missions; noise variances are skill x this."""
    import torch
    from learn.meta import model3 as M
    from studies import onestep_gz as G
    ck = torch.load(os.path.join(C7, "model_p.pt"), map_location="cpu",
                    weights_only=False)
    keys = G.parse_missions("4-7")
    S = G.load_streams(keys, 120.0)
    e2 = []
    for k in keys:
        live = G.full_live(S[k], ck["stats"], torch.device("cpu"))
        ii = torch.zeros(1, dtype=torch.long)
        _, tgt, ok = M.window_tokens(live, ii, torch.zeros_like(ii),
                                     int(S[k]["n"]))
        y = tgt[0][ok[0]][:, :5].numpy()
        e2.append((y[np.isfinite(y).all(1)] ** 2).mean(0))
    e_sd = np.asarray(ck["stats"]["e_sd"])[:5]
    return np.mean(e2, 0) * e_sd ** 2


def window_info(parts, U, s2, s0, dtc, L=256, stride=64, k0=0):
    """Per row and window: gamma, the top eigenvalues, the share of trace(J)
    from steps within 8 after a top-10 % nozzle change."""
    keys = [k for k in parts if not k.endswith(":imp") or True]
    keys = [k for k in keys if np.abs(parts[k]).max() > 0]
    Phi = np.stack([parts[k] * dtc for k in keys], -1)     # (B, n, 5, K)
    B, n = Phi.shape[:2]
    dn = np.abs(np.diff(U[:, :, 1], axis=1, prepend=U[:, :1, 1]))
    out = []
    for b in range(B):
        thr = np.quantile(dn[b], 0.9)
        ev = np.where(dn[b] >= thr)[0]
        post = np.zeros(n, bool)
        for e in ev:
            post[e:e + 9] = True
        for a in range(k0, n - L + 1, stride):
            P = Phi[b, a:a + L]                            # (L, 5, K)
            Jt = np.einsum("tck,c,tcj->tkj", P, 1.0 / s2, P) * s0 ** 2
            J = Jt.sum(0)
            mu = np.clip(np.linalg.eigvalsh(J), 0, None)[::-1]
            tr = np.trace(J)
            share = np.trace(Jt[post[a:a + L]].sum(0)) / max(tr, 1e-30)
            out.append(dict(gamma=float((mu / (1 + mu)).sum()),
                            mu=mu[:6], share_post=float(share),
                            frac_post=float(post[a:a + L].mean()),
                            K=len(keys)))
    return out, keys


def window_pred(parts, s2, s0, dtc, L=256, stride=64, ahead=16):
    """The identification term of the one-step error ledger in the same
    linear-Gaussian surrogate: per window, the posterior covariance of the
    relative part strengths s0^2 (I + J)^-1 and, over the next `ahead`
    steps, the predictive variance it adds per channel, (Phi_t Sigma
    Phi_t')_cc, divided by the noise variance s2_c; also the same with the
    prior covariance s0^2 I (no window). Returns (prior ratios, posterior
    ratios), each (n_windows, 5)."""
    keys = [k for k in parts if np.abs(parts[k]).max() > 0]
    Phi = np.stack([parts[k] * dtc for k in keys], -1)     # (B, n, 5, K)
    B, n, _, K = Phi.shape
    pri, pos = [], []
    for b in range(B):
        for a in range(0, n - L - ahead + 1, stride):
            P = Phi[b, a:a + L]
            J = np.einsum("tck,c,tcj->kj", P, 1.0 / s2, P) * s0 ** 2
            S = s0 ** 2 * np.linalg.inv(np.eye(K) + J)
            F = Phi[b, a + L:a + L + ahead]                # (ahead, 5, K)
            vpos = np.einsum("tck,kj,tcj->tc", F, S, F).mean(0)
            vpri = s0 ** 2 * np.einsum("tck,tck->tc", F, F).mean(0)
            pri.append(vpri / s2)
            pos.append(vpos / s2)
    return np.array(pri), np.array(pos)


# ------------------------------------------------------------ checks
def check_density(args):
    rows = pickle.load(open(os.path.join(C7, "train_meta.pkl"), "rb"))
    styles = [r["style"] for r in rows]
    n = len(styles)
    on = {c: np.array([c in s["items"] and (c != "W7" or bool(
        s["item_params"][c].get("lat", 1))) for s in styles]) for c in GZ_LIKE}
    gain = {c: np.array([float(s["item_params"][c][GAIN[c]])
                         if on[c][i] else np.nan
                         for i, s in enumerate(styles)]) for c in GZ_LIKE}
    log(f"meta7 train: {n} laws (the pretrained net's training set)")
    for c in GZ_LIKE:
        g = gain[c][on[c]]
        log(f"  {c} on {on[c].mean():.3f}; gain {GAIN[c]} quantiles 50/67/90 "
            f"% {np.round(np.quantile(g, [0.5, 0.67, 0.9]), 3)}")
    all_on = np.all([on[c] for c in GZ_LIKE], 0)
    log(f"  all six on: {all_on.sum()} laws ({all_on.mean():.4f}); "
        f"product of single rates {np.prod([on[c].mean() for c in GZ_LIKE]):.4f}")
    for q, lab in ((0.5, "upper half"), (2 / 3, "upper third")):
        strong = {c: on[c] & (gain[c] >= np.nanquantile(gain[c], q))
                  for c in GZ_LIKE}
        cnt = np.sum([strong[c] for c in GZ_LIKE], 0)
        prod = np.prod([strong[c].mean() for c in GZ_LIKE])
        log(f"  strong = {lab} of the item's on-distribution: laws with k of "
            f"6 strong: " + ", ".join(f"{k}: {(cnt == k).sum()}"
                                      for k in range(7))
            + f" | all 6 expected under independence {prod * n:.2f}")


def law_list(args, lib):
    """(label, op) laws: drawn from the prior (seeds of no training law) and
    'gz-like' ones with the six items forced on (their own random
    parameters within the prior)."""
    laws = []
    for i in range(args.n_laws):
        s = 7_000_000 + i
        laws.append((f"prior {s}", s, make_op(s, lib)))
    for i in range(args.n_laws):
        s = 7_100_000 + i
        laws.append((f"gz-like {s}", s, make_op(
            s, lib, force_items={c: True for c in GZ_LIKE}, sparse=False)))
    return laws


def check_info(args):
    from learn.meta import relabel
    Lb = np.load(os.path.join(C7, "lib.npz"))
    lib = {k: Lb[k] for k in ("mu", "sd", "S")}
    env = relabel._env()
    dtc = env["dt"] * env["sub"]
    E2 = mean_e2(env)
    cfgs = [("low noise, s0 0.6", FT_SKILL * E2, 0.6),
            ("high noise, s0 0.6", FROZEN_SKILL * E2, 0.6),
            ("low noise, s0 0.3", FT_SKILL * E2, 0.3)]
    s2, s0 = cfgs[0][1], cfgs[0][2]
    log("noise variances per channel (raw one-step velocity error): low "
        f"{np.array2string(cfgs[0][1], precision=5)}, high "
        f"{np.array2string(cfgs[1][1], precision=5)}; prior relative spread "
        "per part s0 0.6 or 0.3")
    from studies import onestep_gz as G
    keys = G.parse_missions("0-7")
    xs, U, t0, seas, _ = gz_trajs(keys)
    rows = []
    for lab, seed, op in law_list(args, lib):
        tb = time.time()
        parts, _ = eval_parts(op, xs, U, t0, seas, env, seed)
        if args.pred:
            for name, s2_, s0_ in cfgs:
                pri, pos = window_pred(parts, s2_, s0_, dtc)
                log(f"  {lab} ({name}): added one-step variance / noise "
                    f"variance per channel, median over gz windows: no "
                    f"window {np.round(np.median(pri, 0), 3)}, after one "
                    f"window {np.round(np.median(pos, 0), 4)} (90 % "
                    f"{np.round(np.quantile(pos, 0.9, 0), 4)})")
            continue
        wi, pk = window_info(parts, U, s2, s0, dtc)
        sens = {c[0]: np.median([w["gamma"] for w in window_info(
            parts, U, c[1], c[2], dtc)[0]]) for c in cfgs}
        try:
            ox, oU, ot0, oseas, m = own_traj(make_op(seed, lib) if
                                             lab.startswith("prior") else
                                             make_op(seed, lib, force_items={
                                                 c: True for c in GZ_LIKE},
                                                 sparse=False), env, seed,
                                             args.T)
            op2 = make_op(seed, lib) if lab.startswith("prior") else make_op(
                seed, lib, force_items={c: True for c in GZ_LIKE},
                sparse=False)
            p2, _ = eval_parts(op2, ox, oU, ot0, oseas, env, seed)
            wo, _ = window_info(p2, oU, s2, s0, dtc)
        except Exception as ex:                     # noqa: BLE001
            log(f"  {lab}: own trajectory failed ({ex!r})")
            wo = []
        rows.append(dict(law=lab, gz=wi, own=wo, parts=pk, sens=sens))
        f = lambda w, k: np.array([x[k] for x in w])          # noqa: E731
        msg = (f"  {lab}: {len(pk)} parts | gz windows gamma median "
               f"{np.median(f(wi, 'gamma')):.2f} (range "
               f"{f(wi, 'gamma').min():.2f}-{f(wi, 'gamma').max():.2f}), "
               f"post-turn share of information {np.median(f(wi, 'share_post')):.2f}"
               f" on {np.median(f(wi, 'frac_post')):.2f} of the steps")
        if wo:
            msg += (f" | own episode gamma median "
                    f"{np.median(f(wo, 'gamma')):.2f}, post-turn share "
                    f"{np.median(f(wo, 'share_post')):.2f} on "
                    f"{np.median(f(wo, 'frac_post')):.2f}")
        log(msg + f" | gz gamma median by assumption: " + ", ".join(
            f"{k} {v:.2f}" for k, v in sens.items())
            + f" | top eigenvalues (one gz window) "
            f"{np.array2string(wi[0]['mu'], precision=1)}"
            + f" ({time.time() - tb:.0f} s)")
    if args.pred:
        return
    pickle.dump(rows, open(os.path.join(HERE, "_cache", "theory_info.pkl"),
                           "wb"))
    for grp in ("prior", "gz-like"):
        g = [w for r in rows if r["law"].startswith(grp) for w in r["gz"]]
        o = [w for r in rows if r["law"].startswith(grp) for w in r["own"]]
        if g:
            gg = np.array([w["gamma"] for w in g])
            log(f"{grp} laws, gz windows: gamma quantiles 10/50/90 % "
                f"{np.round(np.quantile(gg, [0.1, 0.5, 0.9]), 2)}; post-turn "
                f"share {np.median([w['share_post'] for w in g]):.2f}")
        if o:
            oo = np.array([w["gamma"] for w in o])
            log(f"{grp} laws, own episodes: gamma quantiles 10/50/90 % "
                f"{np.round(np.quantile(oo, [0.1, 0.5, 0.9]), 2)}; post-turn "
                f"share {np.median([w['share_post'] for w in o]):.2f}")


def check_additive(args):
    from learn.meta import relabel
    from studies import onestep_gz as G
    Lb = np.load(os.path.join(C7, "lib.npz"))
    lib = {k: Lb[k] for k in ("mu", "sd", "S")}
    env = relabel._env()
    keys = G.parse_missions(args.gz_missions)
    xs, U, t0, seas, _ = gz_trajs(keys)
    if args.n_steps:
        xs, U = xs[:, :args.n_steps + 1], U[:, :args.n_steps]
    for i in range(args.law0, args.law0 + args.n_laws):
        seed = 7_100_000 + i
        base = make_op(seed, lib, force_items={c: True for c in GZ_LIKE},
                       sparse=False)
        on = dict(base.on)
        cat = [c for c in on if c != "W4"]
        tier = base.cat.tier
        keep = {"W4": on["W4"]} if "W4" in on else {}

        def run(subset):
            force = dict(keep, **{c: on[c] for c in subset})
            op = make_op(seed, lib, items=tuple(force), force_items=force,
                         sparse=False, tier=tier)
            return eval_parts(op, xs, U, t0, seas, env, seed)[1]

        tb = time.time()
        a0 = run(())
        aS = run(cat)
        single = {c: run((c,)) - a0 for c in cat}
        dS = aS - a0
        resid = dS - sum(single.values())
        num = (resid ** 2).sum((0, 1))
        den = (dS ** 2).sum((0, 1))
        log(f"law gz-like {seed}: {len(cat)} catalogue items {cat}; "
            f"non-additive share of all-items minus none, per channel "
            f"{np.round(num / np.maximum(den, 1e-30), 3)} "
            f"({time.time() - tb:.0f} s)")
        sing = sum((v ** 2).sum((0, 1)) for v in single.values())
        log(f"  energies per channel: all-minus-none {np.round(den, 2)}, sum "
            f"of single items {np.round(sing, 2)}, non-additive rest "
            f"{np.round(num, 2)}")
        if args.loo:
            # item k's interaction with all the others:
            # (all - all-but-k) - (k alone - none)
            msgs = []
            for c in cat:
                ik = aS - run([d for d in cat if d != c]) - single[c]
                e = (ik ** 2).sum((0, 1))
                msgs.append((e.max() / max(num.max(), 1e-30), c, e))
            msgs.sort(reverse=True)
            log("  each item's interaction with all the others, energy per "
                "channel (largest first): " + "; ".join(
                    f"{c} {np.round(e, 2)}" for _, c, e in msgs[:8]))
        six = [c for c in GZ_LIKE if c in cat]
        msgs = []
        for a in range(len(six)):
            for b in range(a + 1, len(six)):
                pa, pb = six[a], six[b]
                ab = run((pa, pb)) - a0
                inter = ab - single[pa] - single[pb]
                sz = (ab ** 2).sum((0, 1))
                r = (inter ** 2).sum((0, 1)) / np.maximum(sz, 1e-30)
                msgs.append(f"{pa}+{pb} {np.round(r.max(), 3)}")
        log("  pairwise interaction share (largest over channels): "
            + ", ".join(msgs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default="density",
                    choices=("density", "info", "additive"))
    ap.add_argument("--n-laws", type=int, default=6)
    ap.add_argument("--s0", type=float, default=0.6)
    ap.add_argument("--T", type=int, default=480)
    ap.add_argument("--gz-missions", default="4-5")
    ap.add_argument("--n-steps", type=int, default=0,
                    help="additive: use only the first steps of each mission")
    ap.add_argument("--pred", action="store_true",
                    help="info: only the identification term of the "
                         "one-step error (window_pred), gz windows")
    ap.add_argument("--loo", action="store_true",
                    help="additive: each item's interaction with the rest")
    ap.add_argument("--law0", type=int, default=0,
                    help="additive: first gz-like law index")
    args = ap.parse_args()
    dict(density=check_density, info=check_info,
         additive=check_additive)[args.check](args)


if __name__ == "__main__":
    main()
