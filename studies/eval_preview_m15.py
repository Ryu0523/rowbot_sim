#!/usr/bin/env python3
"""
M15 evaluation (DEFECTS M16, whose code keeps the name m15; brief
BRIEF_PREVIEW.md item 5): does the
preview become useful when the training world's future waves carry
information? Three models trained on the same M15 data (studies/
meta_step5.py): (a) model3.pt, the D6 inputs; (w) model_w.pt, the measured
elevations at t; (p) model_p.pt, measured elevations + preview W_MID
(learn/meta/model_preview.py for the convention). One scoring code for all
(model_preview.eval_one_step_p / eval_multi_step_w; (a) through
rollout_core_w without hook = model3.rollout_core).

  one step    A, At (in family), C, Cb (target; W_MID from the rebuilt sea,
              meta_step5 phase wmid): every step with its recorded history,
              all tokens measured; (p) at forecast levels --levels (0 =
              perfect preview at lead dtc / 2)
  multi step  A_branches, At_branches (M15 operator, meta_step5 phase
              branches) and Cb blocks (every 24-step block from step 48;
              plans fixed in advance): the MPC's rollout convention -- the
              measurement only at the moment k, then (w) nothing, (p) the
              forecast at the rolled pose within the preview horizon
              (--horizons steps) at levels --levels
  wave use    in family (A), one-step skill at t in [240, 288) with the
              last 16 / 64 / 232 steps as history, episodes grouped by
              their operator's same-step wave dependence (style
              wave_same_step: an undelayed wave filter with feedthrough, a
              direct wave input or an own wave event with feedthrough),
              and 'reads no waves at all'

Surge / sway / yaw (whose wave relation comes only from the operators) are
reported apart from heave / pitch (where e0 also holds the low-fidelity
boat's own wave response, the same in every episode).

Every context is capped at the trained context ctx the three checkpoints
record (model_preview.train_ctx; they must agree): one-step windows of ctx
tokens, rollout history ctx - HB, wave-use histories <= ctx - 1. With
90-s training episodes ctx = W_CTX = 256 and nothing is capped; with 45-s
ones ctx = 179 (positions and attention spans beyond are never trained).

    python studies/eval_preview_m15.py [--cache-name meta5] [--smoke]
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
import numpy as np   # noqa: E402
import torch   # noqa: E402

from learn.meta import model3 as M   # noqa: E402
from learn.meta import model_preview as MP   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LH = (16, 64, 232)
T0, NT = 240, 48
CACHE = None


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(CACHE, "eval_m15.log"), "a",
              encoding="utf-8") as fh:
        fh.write(line + "\n")


def fmt1(r, label):
    s, c = r["skill"], r["cov"]
    return (f"  {label:<22} skill [surge sway yaw] " + " ".join(
        f"{x:.2f}" for x in s[:3]) + " | [heave pitch] " + " ".join(
        f"{x:.2f}" for x in s[3:5]) + " | cov90 " + " ".join(
        f"{x:.2f}" for x in c[:5]))


def fmtm(r, label):
    out = [f"  {label}:"]
    for a, b in ((0, 1), (1, 5), (5, M.HB)):
        s = r["skill"][a:b, :5].mean(0)
        c = r["cov"][a:b, :5].mean(0)
        st = r["states"]["skill"][a:b].mean(0)
        out.append(f"    steps {a:2d}-{b - 1:2d}: e skill [surge sway yaw] "
                   + " ".join(f"{x:.2f}" for x in s[:3]) + " | [heave pitch] "
                   + " ".join(f"{x:.2f}" for x in s[3:]) + " | cov90 "
                   + " ".join(f"{x:.2f}" for x in c) + " | states [u v r z "
                   "th psi] " + " ".join(f"{x:.2f}" for x in st))
    return "\n".join(out)


def cb_blocks(D):
    E0all = np.load(os.path.join(CACHE, "Cb_e0.npz"))["E0"]
    ep, kk, Us, XS0, E0 = [], [], [], [], []
    for i in range(D.n):
        for k in range(48, int(D.len[i]) - M.HB + 1, M.HB):
            ep.append(i)
            kk.append(k)
            Us.append(D.Uraw[i, k:k + M.HB])
            XS0.append(D.XS[i, k:k + M.HB + 1])
            E0.append(E0all[i, k:k + M.HB])
    return dict(ep=np.array(ep), k=np.array(kk), U=np.stack(Us)[:, None],
                XS=np.stack(XS0)[:, None], E0=np.stack(E0)[:, None])


def trained_ctx(cks):
    """The trained context of the checkpoints {variant: ck}: all must
    record the same ctx (meta_step5 since 2026-09-30); refuses otherwise."""
    got = {v: ck.get("ctx") for v, ck in cks.items()}
    if any(c is None for c in got.values()):
        raise SystemExit(f"checkpoints without a trained context {got}: "
                         "retrain with studies/meta_step5.py")
    if len(set(got.values())) != 1:
        raise SystemExit(f"checkpoints trained on different contexts {got}")
    ctx = int(next(iter(got.values())))
    if not M.HB < ctx <= M.W_CTX:
        raise SystemExit(f"trained context {ctx} outside ({M.HB}, "
                         f"{M.W_CTX}]")
    return ctx


def history_lengths(ctx):
    """LH capped at the trained context (a window of lh + 1 tokens)."""
    return tuple(sorted({min(lh, ctx - 1) for lh in LH}))


def wave_skills(net, D, eps, lh, variant, gen, gw):
    """check_wave_use_m14.skills for any variant."""
    se, s0 = np.zeros(5), np.zeros(5)
    with torch.no_grad():
        for t in range(T0, T0 + NT, 4):
            for b0 in range(0, len(eps), 64):
                ii = eps[b0:b0 + 64]
                a = torch.full_like(ii, t - lh)
                if variant == "a":
                    tok, tgt, ok = M.window_tokens(D, ii, a, lh + 1)
                else:
                    cfg = MP.fixed_cfg(D, a + lh + 1, 0.0,
                                       1 if variant == "p" else 0, 0.01, gw)
                    tok, tgt, ok = MP.tokens_p(D, ii, a, lh + 1, cfg, gw,
                                               obs=False)
                h = net.encode(tok)[:, -1]
                y, v = tgt[:, -1], ok[:, -1]
                if not v.any():
                    continue
                s = net.sample(h[v], 32, gen=gen)
                m = s.mean(1)
                se += ((m - y[v]) ** 2 - s.var(1) / 32).sum(0)[:5].cpu(
                ).numpy()
                s0 += (y[v] ** 2).sum(0)[:5].cpu().numpy()
    return se / np.maximum(s0, 1e-12)


def main():
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-name", default="meta5")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--levels", default="0,0.1,0.5")
    ap.add_argument("--horizons", default="1,8,24")
    ap.add_argument("--n-samp", type=int, default=16)
    ap.add_argument("--one-samp", type=int, default=64)
    ap.add_argument("--msd", type=float, default=0.01,
                    help="sensor noise sd of the measured elevations "
                         "(library units)")
    ap.add_argument("--skip", default="",
                    help="comma list of parts to skip: one,multi,use")
    args = ap.parse_args()
    CACHE = os.path.join(HERE, "_cache", args.cache_name
                         + ("_smoke" if args.smoke else ""))
    levels = [float(x) for x in args.levels.split(",")]
    horizons = [int(x) for x in args.horizons.split(",")]
    skip = set(args.skip.split(","))
    from learn.meta import relabel
    env = relabel._env()
    dev = M.device()
    cks = {v: torch.load(os.path.join(CACHE, f), weights_only=False)
           for v, f in (("a", "model3.pt"), ("w", "model_w.pt"),
                        ("p", "model_p.pt"))}
    nets = {v: MP.load_variant(ck, dev) for v, ck in cks.items()}
    ctx = trained_ctx(cks)
    nh = ctx - M.HB
    lhs = history_lengths(ctx)
    log(f"trained context {ctx} tokens: one-step windows {ctx}, rollout "
        f"history {nh}, wave-use histories {lhs}")
    stats = dict(cks["p"]["stats"])
    for v in ("a", "w"):
        if not np.allclose(cks[v]["stats"]["e_sd"], stats["e_sd"]):
            raise SystemExit(f"model {v} has another error scale; retrain "
                             "w / p after a (meta_step5 uses model3.pt's)")
    Dtr = M.Data3(CACHE, "train", dev, stats=stats)
    q = torch.quantile(Dtr.E[Dtr.valid].abs(), 0.99, dim=0)
    del Dtr
    res = {}
    settings1 = [("a", {}), ("w", {})] + [("p", dict(lam=x)) for x in levels]
    for split in ("A", "At", "C", "Cb"):
        if not os.path.exists(os.path.join(CACHE, f"{split}.npz")):
            continue
        D = MP.DataP(CACHE, split, dev, stats=stats)
        if "one" not in skip:
            lines = []
            for v, kw in settings1:
                r = MP.eval_one_step_p(nets[v], D, q, variant=v,
                                       msd=args.msd, n_samp=args.one_samp,
                                       L=ctx, **kw)
                lab = v + (f" level {kw['lam']:g}" if kw else "")
                res[(split, "one", lab)] = r
                lines.append(fmt1(r, lab))
            log(f"one step, {split} ({D.n} episodes):\n" + "\n".join(lines))
        br = None
        if split in ("A", "At"):
            f = os.path.join(CACHE, f"{split}_branches.npz")
            if os.path.exists(f):
                b = np.load(f)
                br = {k_: b[k_] for k_ in b.files}
        elif split == "Cb":
            br = cb_blocks(D)
        if br is not None and "multi" not in skip:
            gw_seed = 101
            runs = [("a", None), ("w", dict(lam=0.0, hp=0))]
            runs += [("p", dict(lam=x, hp=max(horizons))) for x in levels]
            runs += [("p", dict(lam=levels[min(1, len(levels) - 1)], hp=h))
                     for h in horizons if h != max(horizons)]
            lines = []
            for v, kw in runs:
                if kw is None:
                    wf, lab = None, "a"
                else:
                    def wf(D_, ep, k, kw=kw):
                        g = torch.Generator().manual_seed(
                            gw_seed + int(np.asarray(k)[0]))
                        return MP.WaveRoll(D_, ep, k, kw["lam"], kw["hp"],
                                           args.msd, gen=g)
                    lab = v if v == "w" else (f"p level {kw['lam']:g} "
                                              f"horizon {kw['hp']}")
                r = MP.eval_multi_step_w(nets[v], D, br, env, q,
                                         n_samp=args.n_samp, wave_fn=wf,
                                         nh=nh)
                res[(split, "multi", lab)] = r
                lines.append(fmtm(r, lab))
            log(f"multi step, {split} ({br['U'].shape[0]} moments x "
                f"{br['U'].shape[1]} plans; measurement at the moment only, "
                "then the forecast at the rolled pose):\n" + "\n".join(lines))
        if split == "A" and "use" not in skip:
            ok = (D.len > T0 + NT + 1).cpu().numpy()
            st = [m["style"] for m in D.meta]
            same = np.array([bool(s.get("wave_same_step")) for s in st])
            reads = np.array([bool(s.get("reads_waves")) and not s["null"]
                              for s in st])
            groups = {"same-step wave path": ok & same,
                      "lagged wave paths only": ok & reads & ~same,
                      "reads no waves": ok & ~reads}
            gen = torch.Generator(device=dev).manual_seed(0)
            lines = ["one-step skill at t in [240, 288) by history length, "
                     "a / w / p (perfect preview):"]
            for g, msk in groups.items():
                idx = np.flatnonzero(msk)
                lines.append(f"  {g} ({len(idx)} episodes):")
                if len(idx) == 0:
                    continue
                eps = torch.as_tensor(idx, device=dev)
                for lh in lhs:
                    sk = {v: wave_skills(nets[v], D, eps, lh, v, gen,
                                         torch.Generator().manual_seed(7))
                          for v in ("a", "w", "p")}
                    res[("use", g, lh)] = sk
                    lines.append(f"    history {lh:3d}: " + "  ".join(
                        f"{n} " + "/".join(f"{sk[v][c]:.2f}" for v in
                                           ("a", "w", "p"))
                        for c, n in enumerate(("surge", "sway", "yaw",
                                               "heave", "pitch"))))
            log("wave use in family (A):\n" + "\n".join(lines))
        del D
        torch.cuda.empty_cache()
    out = os.path.join(CACHE, "eval_m15.pkl")
    with open(out + ".tmp", "wb") as fh:
        pickle.dump(res, fh)
    os.replace(out + ".tmp", out)
    log(f"saved eval_m15.pkl ({len(res)} results)")


if __name__ == "__main__":
    main()
