#!/usr/bin/env python3
"""
Tests of the M9 pieces (DEFECTS M9: rollout-robust training of the one-step
error model), on the smoke cache, in one process, in a few minutes:

  1  Model0T (torch float64) vs data3.model0_step:                < 1e-9
  2  KV history pass + incremental tokens vs encode() on the concatenated
     sequence, several Lh in one batch, train() + grad and eval() + no grad:
                                                                   < 1e-4
  3  rollout_core (no grad) vs the old rollout() with identical base noise:
     step 0 < 1e-4; the 24-step differences are reported (chaotic
     amplification allowed)
  4  leakage: changing the recorded tokens at t >= k (k < 232) leaves
     rollout_core's outputs bit-identical
  5  rollout_core grad=True and grad=False agree for the same base noise
  6  train() unchanged: val_episodes = the old inline draw; a copy of the old
     train loop and the new one give identical logs and weights (40 steps;
     OneCycleLR with pct_start 0.05 cannot run 20)
  7  wide_plans leaves make_plans' draws unchanged (plans and the stream's
     state); branches_one still reproduces the stored A_branches
  8  smoke pipeline end to end in a temporary copy of the smoke cache:
     tbranches (a few jobs), train_cov and train_es (20 steps), eval
  9  review fixes: LF line endings of the M9 files; main() refuses an
     --out-tag that would name several outputs; Branches and
     build_tbranches drop moments with non-finite XS / E0

    python -m studies.test_m9 [1 2 ...]
"""
import argparse
import copy
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn

from learn.meta import data3
from learn.meta import model3 as M
from learn.meta import relabel

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.path.join(HERE, "_cache", "meta2_smoke")
RES = {}


def _setup():
    if "net" not in RES:
        dev = M.device(0.5)
        ck = torch.load(os.path.join(SMOKE, "model3.pt"), weights_only=False)
        net = M.Net().to(dev)
        net.load_state_dict(ck["net"])
        net.eval()
        RES.update(dev=dev, ck=ck, net=net, env=relabel._env(),
                   D=M.Data3(SMOKE, "A", dev, stats=ck["stats"]))
    return RES["dev"], RES["net"], RES["env"], RES["D"]


def report(name, ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
    RES.setdefault("fails", [])
    if not ok:
        RES["fails"].append(name)


# ------------------------------------------------------------------ 1
def test_1():
    dev, net, env, D = _setup()
    d = np.load(os.path.join(SMOKE, "train.npz"))
    rng = np.random.default_rng(0)
    n = 4000
    i = rng.integers(0, d["XS"].shape[0], n)
    t = rng.integers(0, 359, n)
    from learn.meta.data2 import plant_to_reduced
    sr = plant_to_reduced(d["XS"][i, t].astype(float))
    U = d["U"][i, t].astype(float)
    # half recorded commands, half anywhere in range (full reversals)
    U[n // 2:] = np.column_stack([rng.uniform(0, 1, n // 2),
                                  rng.uniform(-1, 1, n // 2)])
    ref = data3.model0_step(env, sr, U)
    m0 = M.Model0T(env)
    out = m0(torch.tensor(sr, dtype=torch.float64, device=dev),
             torch.tensor(U, dtype=torch.float64, device=dev)).cpu().numpy()
    err = np.abs(out - ref).max()
    # also 24 chained steps (the rollout's use)
    s1, s2 = sr[:500].copy(), torch.tensor(sr[:500], dtype=torch.float64,
                                           device=dev)
    for j in range(24):
        s1 = data3.model0_step(env, s1, U[:500])
        s2 = m0(s2, torch.tensor(U[:500], dtype=torch.float64, device=dev))
    err24 = np.abs(s2.cpu().numpy() - s1).max()
    report("1 Model0T vs model0_step", err < 1e-9,
           f"max |diff| one step {err:.2e}, 24 chained steps {err24:.2e}")


# ------------------------------------------------------------------ 2
def test_2():
    dev, net, env, D = _setup()
    NH = M.W_CTX - M.HB
    ks = torch.tensor([0, 3, 100, 231, 232, 300], device=dev)
    B = len(ks)
    ii = torch.arange(B, device=dev) % D.n
    a = (ks - NH).clamp(min=0)
    Lh = ks - a
    tok_h, _, _ = M.window_tokens(D, ii, a, NH)
    P, S, G = 2, 2, 6
    g = torch.Generator(device="cpu").manual_seed(3)
    gtok = torch.randn(B, P, S, G, M.D_TOK, generator=g).to(dev)
    hide = torch.arange(NH, device=dev)[None] >= Lh[:, None]
    for mode in ("train+grad", "eval+no_grad"):
        net.train(mode.startswith("train"))
        with torch.set_grad_enabled(mode.startswith("train")):
            hist, hout = M.kv_history(net, tok_h)
            cache = [([], []) for _ in net.tf.layers]
            outs = []
            for j in range(G):
                outs.append(M.kv_step(net, gtok[:, :, :, j],
                                      net.pos[Lh + j].view(B, 1, 1, -1),
                                      hist, hide, cache))
            outs = torch.stack(outs, 3)                  # (B, P, S, G, d)
            e_g, e_h = 0.0, 0.0
            for b in range(B):
                L = int(Lh[b])
                if L:
                    ref_h = net.encode(tok_h[b:b + 1, :L])[0]
                    e_h = max(e_h, (ref_h - hout[b, :L]).abs().max().item())
                for p in range(P):
                    for s in range(S):
                        seq = torch.cat([tok_h[b, :L], gtok[b, p, s]], 0)
                        ref = net.encode(seq[None])[0, L:]
                        e_g = max(e_g, (ref - outs[b, p, s]).abs().max()
                                  .item())
            if mode.startswith("train"):
                outs.sum().backward()          # the graph is usable
                net.zero_grad(set_to_none=True)
        report(f"2 KV vs encode ({mode})", e_g < 1e-4 and e_h < 1e-4,
               f"Lh {Lh.tolist()}: generated tokens max |diff| {e_g:.2e}, "
               f"history outputs {e_h:.2e}")
    net.eval()


# ------------------------------------------------------------------ 3
def test_3():
    dev, net, env, D = _setup()
    b = np.load(os.path.join(SMOKE, "A_branches.npz"))
    S = 8
    for kv in (120, 240):
        sel = np.flatnonzero(b["k"] == kv)[:4]
        B = len(sel)
        plans = b["U"][sel, 1].astype(float)
        xs0 = b["XS"][sel, 1, 0].astype(float)
        g1 = torch.Generator(device=dev).manual_seed(5)
        g2 = torch.Generator(device=dev).manual_seed(5)
        t0 = time.time()
        e_old, s_old = M.rollout(net, D, b["ep"][sel], b["k"][sel], plans,
                                 xs0, env, n_samp=S, gen=g1)
        t_old = time.time() - t0
        # the old sampler draws (B * S, 10) per step, rows sample-minor
        draws = torch.stack([torch.randn(B * S, M.C7, device=dev,
                                         generator=g2) for _ in range(24)])
        base = draws.permute(1, 0, 2).reshape(B, S, 24, M.C7)[:, None]
        t0 = time.time()
        e_new, s_new, _ = M.rollout_core(net, D, b["ep"][sel], b["k"][sel],
                                         plans[:, None], xs0, env, S,
                                         base=base)
        t_new = time.time() - t0
        e_new, s_new = e_new[:, 0], s_new[:, 0].cpu().numpy()
        d0 = (e_new[:, :, 0] - e_old[:, :, 0]).abs().max().item()
        dall = (e_new - e_old).abs().max(-1)[0].amax((0, 1)).cpu().numpy()
        ds = np.abs(s_new - s_old).max((0, 1))            # (H, 10)
        sc = np.abs(s_old).max((0, 1, 2)) + 1e-12
        report(f"3 rollout_core vs rollout (k = {kv})", d0 < 1e-4,
               f"step-0 samples max |diff| {d0:.2e}; per step "
               f"{np.array2string(dall[[1, 4, 9, 23]], precision=2)} at "
               f"steps 1/4/9/23 (normalised e); states max |diff| / max "
               f"|state| at step 23 "
               f"{np.array2string((ds[23] / sc)[[2, 9, 8, 3, 5, 7]], precision=2)}"
               f" (u v r z th psi); time old {t_old:.2f} s, new "
               f"{t_new:.2f} s")


def _mixed_rows(D):
    b = np.load(os.path.join(SMOKE, "A_branches.npz"))
    ks = np.array([0, 50, 120, 200, 240, 300])
    ep = np.arange(len(ks)) % D.n
    XS = D.XS[ep, ks].astype(float)
    plans = np.stack([b["U"][r % len(b["U"]), :2] for r in range(len(ks))])
    return ep, ks, plans.astype(float), XS


# ------------------------------------------------------------------ 4
def test_4():
    dev, net, env, D = _setup()
    ep, ks, plans, XS = _mixed_rows(D)
    sel = ks < M.W_CTX - M.HB
    ep, ks, plans, XS = ep[sel], ks[sel], plans[sel], XS[sel]
    g = torch.Generator(device="cpu").manual_seed(1)
    base = torch.randn(len(ks), 2, 4, 24, M.C7, generator=g)
    out1 = M.rollout_core(net, D, ep, ks, plans, XS, env, 4, base=base)
    D2 = copy.copy(D)
    D2.X, D2.U, D2.E = D.X.clone(), D.U.clone(), D.E.clone()
    for i, k in zip(ep, ks):
        n = D.T - k
        D2.X[i, k:] += torch.randn(n, D.X.shape[-1], generator=g).to(dev)
        D2.U[i, k:] += torch.randn(n, 2, generator=g).to(dev)
        # E[k - 1] is token 0's recorded e_prev (before k): keep it
        D2.E[i, k:] += torch.randn(n, M.C7, generator=g).to(dev)
    out2 = M.rollout_core(net, D2, ep, ks, plans, XS, env, 4, base=base)
    same = all(torch.equal(x, y) for x, y in zip(out1, out2))
    diff = max((x - y).abs().max().item() for x, y in zip(out1, out2))
    report("4 no leakage of the recorded future", same,
           f"k {ks.tolist()}: outputs bit-identical {same} (max |diff| "
           f"{diff:.1e})")


# ------------------------------------------------------------------ 5
def test_5():
    dev, net, env, D = _setup()
    ep, ks, plans, XS = _mixed_rows(D)
    g = torch.Generator(device="cpu").manual_seed(2)
    base = torch.randn(len(ks), 2, 4, 24, M.C7, generator=g)
    o1 = M.rollout_core(net, D, ep, ks, plans, XS, env, 4, base=base)
    o2 = M.rollout_core(net, D, ep, ks, plans, XS, env, 4, base=base,
                        grad=True)
    diff = max((x - y.detach()).abs().max().item() for x, y in zip(o1, o2))
    report("5 grad=True vs grad=False", diff < 1e-6 and o2[0].requires_grad,
           f"k {ks.tolist()}: max |diff| {diff:.1e}, grad path requires_grad "
           f"{o2[0].requires_grad}")


# ------------------------------------------------------------------ 6
def _window_tokens_old(D, ii, a, L, obs_sd=None, gen=None):
    ar = torch.arange(L, device=D.dev)[None]
    t = a[:, None] + ar
    tc = t.clamp(max=D.T - 1)
    X, U = D.X[ii[:, None], tc], D.U[ii[:, None], tc]
    tp = (t - 1).clamp(min=0)
    Ep = D.E[ii[:, None], tp] * (t >= 1)[..., None]
    if obs_sd is not None:
        Ep = Ep + obs_sd[:, None, :] * torch.randn(Ep.shape,
                                                   generator=gen).to(D.dev)
    first = (ar == 0).float().expand_as(t)
    tgt = D.E[ii[:, None], tc]
    ok = D.valid[ii[:, None], tc] & (t < D.T)
    return M.make_tokens(X, U, Ep, first), tgt, ok


def _obs_sd_old(B, dev, gen=None):
    u = torch.rand(B, 1, generator=gen).to(dev)
    base = torch.exp(math.log(0.01) + u * (math.log(0.1) - math.log(0.01)))
    return base * torch.exp(0.3 * torch.randn(B, M.C7, generator=gen).to(dev))


def _train_old(D, steps=20000, batch=48, L=M.W_CTX, lr=3e-4, log=print,
               seed=0, patience=3):
    """model3.train as it was before M9 (verbatim, helpers inlined)."""
    dev = D.dev
    torch.manual_seed(seed)
    net = M.Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=steps,
                                                pct_start=0.05)
    g = torch.Generator(device="cpu").manual_seed(seed)
    gv = torch.Generator(device="cpu").manual_seed(seed + 5)
    ep_val = torch.randperm(D.n, generator=gv)[:max(2, D.n // 33)]
    is_val = torch.zeros(D.n, dtype=torch.bool)
    is_val[ep_val] = True
    tr = torch.nonzero(~is_val)[:, 0]
    va = torch.nonzero(is_val)[:, 0]
    Lmax = D.len.cpu()

    def draw(pool, b, gen):
        ii = pool[torch.randint(0, len(pool), (b,), generator=gen)]
        span = (Lmax[ii] - 1 - L).clamp(min=0)
        a = (torch.rand(b, generator=gen) * (span + 1).float()).long()
        a = torch.where(torch.rand(b, generator=gen) < 0.25,
                        torch.zeros_like(a), a)
        return ii.to(dev), a.to(dev)

    def loss_of(ii, a, gen, noise=True):
        tok, tgt, ok = _window_tokens_old(D, ii, a, L, _obs_sd_old(
            len(ii), dev, gen) if noise else None, gen)
        h = net.encode(tok)
        y1 = torch.asinh(tgt / M.KAPPA)
        y0 = torch.randn(y1.shape, generator=gen).to(dev)
        tau = torch.rand(y1.shape[:2], generator=gen).to(dev)
        yt = (1 - tau[..., None]) * y0 + tau[..., None] * y1
        v = net.velocity(yt, tau, h)
        se = ((v - (y1 - y0)) ** 2).mean(-1)
        return (se * ok).sum() / ok.sum().clamp(min=1)

    @torch.no_grad()
    def validate():
        gen = torch.Generator(device="cpu").manual_seed(4321)
        ls = []
        for _ in range(8):
            ii, a = draw(va, 32, gen)
            ls.append(loss_of(ii, a, gen).item())
        return float(np.mean(ls))

    best = dict(v=float("inf"), it=-1, bad=0)
    t0 = time.time()
    for it in range(steps):
        ii, a = draw(tr, batch, g)
        loss = loss_of(ii, a, g)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == steps - 1:
            v = validate()
            log(f"    one-step model {it:6d}: train {loss.item():.4f}  "
                f"episodes kept out of training {v:.4f}  "
                f"({time.time() - t0:.0f} s)")
            if it > steps // 5:
                if v < best["v"]:
                    best.update(v=v, it=it, bad=0, state={
                        k_: x.detach().clone() for k_, x in
                        net.state_dict().items()})
                else:
                    best["bad"] += 1
                    if best["bad"] >= patience:
                        log(f"    not better for {patience} checks, "
                            f"stopping at step {it}")
                        break
    if best["it"] >= 0 and best["it"] != it:
        net.load_state_dict(best["state"])
        log(f"    restored step {best['it']} ({best['v']:.4f})")
    return net


def test_6():
    dev, _, _, _ = _setup()
    ok_v = True
    for n in (2, 40, 100, 5000):
        gv = torch.Generator(device="cpu").manual_seed(5)
        old = torch.randperm(n, generator=gv)[:max(2, n // 33)]
        ok_v &= torch.equal(old, M.val_episodes(n))
    # on the CPU: two runs of the OLD loop already differ in the last bits on
    # the GPU (even with deterministic algorithms), so bitwise needs the CPU
    D = M.Data3(SMOKE, "train", torch.device("cpu"))
    runs = []
    for fn in (_train_old, M.train, _train_old):
        logs = []
        net = fn(D, steps=40, log=lambda s: logs.append(
            s.rsplit("(", 1)[0] if "one-step" in s else s))
        runs.append((logs, {k_: v.clone() for k_, v in
                            net.state_dict().items()}))
    same = lambda r1, r2: r1[0] == r2[0] and all(  # noqa: E731
        torch.equal(r1[1][k_], r2[1][k_]) for k_ in r1[1])
    det = same(runs[0], runs[2])
    eq = same(runs[0], runs[1])
    wd = max((runs[0][1][k_] - runs[1][1][k_]).abs().max().item()
             for k_ in runs[0][1])
    report("6 train() unchanged", ok_v and eq,
           f"val_episodes equal {ok_v}; old vs new 40 steps: logs and "
           f"weights identical {eq} (max weight diff {wd:.1e}; old vs old "
           f"identical {det}); logs {runs[1][0]}")


# ------------------------------------------------------------------ 7
def test_7():
    ok, n_wide, n_pl = True, 0, 0
    for s in range(300):
        u = np.random.default_rng([s, 1]).uniform([0, -1], [1, 1])
        p1, p2 = np.random.default_rng(s), np.random.default_rng(s)
        m = data3.make_plans(p1, u, 8)
        w, wide = data3.wide_plans(p2, u, 8, rng2=np.random.default_rng(
            [s, 7]))
        ok &= p1.bit_generator.state == p2.bit_generator.state
        ok &= np.array_equal(m[~wide], w[~wide]) and not wide[0]
        # a flagged plan equals make_plans' up to its first step change
        for p in np.flatnonzero(wide):
            dj = np.flatnonzero(np.any(m[p] != w[p], 1))
            ok &= len(dj) == 0 or np.array_equal(m[p, :dj[0]], w[p, :dj[0]])
        n_wide += wide.sum()
        n_pl += 7
    # the refactored branches_one reproduces the stored A_branches, and the
    # wide version keeps its unflagged plans
    lib = dict(np.load(os.path.join(SMOKE, "lib.npz")))
    relabel._init(os.path.join(SMOKE, "A.npz"), lib)
    b = np.load(os.path.join(SMOKE, "A_branches.npz"))
    ok_b, n_same = True, []
    for r in (0, 3):
        job = (int(b["ep"][r]), int(b["k"][r]), 8, 0)
        _, _, U, XS, E0 = data3.branches_one(job)
        ok_b &= (np.array_equal(U, b["U"][r]) and np.array_equal(XS, b["XS"][r])
                 and np.array_equal(E0, b["E0"][r]))
        _, _, Uw, XSw, E0w, wide = data3.tbranches_one(job)
        ok_b &= np.array_equal(Uw[~wide], b["U"][r][~wide])
        # their TRUTHS need not match: the P branches share the operator's
        # noise stream, and state-dependent draws couple them (info only)
        n_same.append(f"{sum(np.array_equal(XSw[p], b['XS'][r][p]) for p in np.flatnonzero(~wide))}/{(~wide).sum()}")
    report("7 wide_plans / branches unchanged", ok and ok_b,
           f"300 seeds: make_plans' stream state and unflagged plans equal "
           f"{ok}; {n_wide / n_pl:.2f} of plans >= 1 flagged; branches_one = "
           f"stored A_branches and tbranches_one's unflagged plans = them "
           f"{ok_b} (unflagged truths identical in {n_same} plans: the "
           f"branches share one operator noise stream)")


# ------------------------------------------------------------------ 8
def test_8():
    from studies import meta_step3 as S3
    tmp = tempfile.mkdtemp(prefix="m9_smoke_")
    try:
        for f in os.listdir(SMOKE):
            if f.split("_")[0].split(".")[0] in ("train", "A", "B", "C", "Cb",
                                                 "lib") and f.endswith(
                    (".npz", ".pkl")) or f == "model3.pt":
                if "targets" not in f:
                    shutil.copy(os.path.join(SMOKE, f), tmp)
        S3.CACHE = tmp
        args = argparse.Namespace(smoke=True, force=True, procs=1, max_jobs=6,
                                  steps=20, tag="", out_tag=None)
        t0 = time.time()
        S3.phase_tbranches(args)
        t_b = time.time() - t0
        tb = np.load(os.path.join(tmp, "train_tbranches.npz"))
        t0 = time.time()
        S3.phase_train_cov(args)
        S3.phase_train_es(args)
        t_t = time.time() - t0
        ck = torch.load(os.path.join(tmp, "model3_es.pt"), weights_only=False)
        info = ck["info"]
        t0 = time.time()
        for tag, out in (("", "_kv"), ("_cov", None), ("_es", None)):
            S3.phase_eval(argparse.Namespace(**{**vars(args), "tag": tag,
                                                "out_tag": out}))
        t_e = time.time() - t0
        got = sorted(f for f in os.listdir(tmp) if f.startswith(("eval3",
                                                                 "model3")))
        ok = all(os.path.exists(os.path.join(tmp, f)) for f in (
            "eval3_kv.pkl", "eval3_cov.pkl", "eval3_es.pkl",
            "A_wbranches.npz", "train_vbranches.npz"))
        refused = False
        try:
            S3._train_rollout(argparse.Namespace(**{**vars(args),
                                                    "out_tag": ""}), 1.0, "")
        except SystemExit:
            refused = True
        report("8 smoke pipeline", ok and refused,
               f"tbranches {t_b:.0f} s (train_tbranches U {tb['U'].shape}), "
               f"train_cov + train_es 20 steps each {t_t:.0f} s (train_es "
               f"{info['s_per_step']:.2f} s/step, GPU peak "
               f"{info['gpu_peak_gb']:.2f} GB), 3 evals {t_e:.0f} s; files "
               f"{got}; refuses model3.pt {refused}")
    finally:
        S3.CACHE = os.path.join(HERE, "_cache", "meta2")
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 9
def test_9():
    from studies import meta_step3 as S3
    # (a) the M9 files keep their LF line endings
    root = os.path.dirname(HERE)
    files = ["learn/meta/model3.py", "learn/meta/data3.py",
             "studies/meta_step3.py", "studies/test_m9.py",
             "studies/diag_yaw_m8_blocks.py"]
    crlf = [f for f in files
            if b"\r" in open(os.path.join(root, f), "rb").read()]
    ok_lf = not crlf

    # (b) --out-tag names one output: main() refuses to combine the phases
    # that write model3<out-tag>.pt / eval3<out-tag>.pkl (stub phases)
    calls = []
    saved = (dict(S3.PHASES), dict(S3.EXTRA), S3.CACHE, S3.PROCS, sys.argv)
    for d in (S3.PHASES, S3.EXTRA):
        for nm in d:
            d[nm] = (lambda nm_: lambda a: calls.append(nm_))(nm)
    cases = [("train_cov,train_es", "_v2", True),
             ("train_es,eval", "_v2", True),
             ("train_cov,eval", "_v2", True),
             ("all", "_v2", False), ("eval", "_v2", False),
             ("train_es", "_v2", False), ("train_cov,train_es", None, False)]
    ok_tag, got = True, []
    try:
        for ph, ot, want_refuse in cases:
            sys.argv = ["meta_step3", "--smoke", "--phase", ph] + (
                ["--out-tag", ot] if ot is not None else [])
            calls.clear()
            try:
                S3.main()
                refused = False
            except SystemExit:
                refused = True
            ok_tag &= refused == want_refuse and (refused == (not calls))
            got.append(f"{ph}/{ot}:{'refused' if refused else len(calls)}")
    finally:
        S3.PHASES.clear()
        S3.PHASES.update(saved[0])
        S3.EXTRA.clear()
        S3.EXTRA.update(saved[1])
        S3.CACHE, S3.PROCS, sys.argv = saved[2], saved[3], saved[4]

    # (c) Branches drops a moment with a non-finite truth (files written
    # before build_tbranches filtered); the rest equals the clean build
    dev, net, env, _ = _setup()
    ck = RES["ck"]
    Dtr = M.Data3(SMOKE, "train", dev, stats=ck["stats"])
    br = dict(np.load(os.path.join(SMOKE, "train_vbranches.npz")))
    bad = {k_: v.copy() for k_, v in br.items()}
    bad["XS"][1, 1, 3, 2] = np.nan
    bad["E0"][2, 0, 5, 0] = np.inf
    keep = np.ones(len(br["ep"]), bool)
    keep[[1, 2]] = False
    B0 = M.Branches({k_: v[keep] for k_, v in br.items()}, Dtr, env)
    B1 = M.Branches(bad, Dtr, env)
    ok_br = (B1.N == B0.N == keep.sum() and B1.n_dropped == 2
             and torch.equal(B1.dev_true, B0.dev_true)
             and torch.equal(B1.E, B0.E) and torch.equal(B1.ep, B0.ep)
             and bool(torch.isfinite(B1.dev_true.std((0, 1))).all()))
    del Dtr, B0, B1

    # (d) build_tbranches drops such moments before saving (ep / k / U / XS
    # / E0 / wide aligned) and refuses an all-bad file
    b = np.load(os.path.join(SMOKE, "A_branches.npz"))
    jobs = [(int(b["ep"][r]), int(b["k"][r]), 4, 0) for r in (0, 1, 2)]
    lib = dict(np.load(os.path.join(SMOKE, "lib.npz")))
    orig = data3.tbranches_one
    poison = {jobs[1][:2]}

    def one(job):
        r = list(orig(job))
        if job[:2] in poison:
            r[3] = r[3].copy()
            r[3][0, 2, 0] = np.inf
        return tuple(r)

    tmp = tempfile.mkdtemp(prefix="m9_nf_")
    try:
        data3.tbranches_one = one
        out = os.path.join(tmp, "x_tbranches.npz")
        data3.build_tbranches(os.path.join(SMOKE, "A.npz"), lib, jobs, out,
                              procs=1)
        f = np.load(out)
        ok_bt = (f["U"].shape[0] == 2
                 and [(int(e), int(k)) for e, k in zip(f["ep"], f["k"])]
                 == sorted(j[:2] for j in jobs if j[:2] not in poison)
                 and all(f[n].shape[0] == 2 for n in ("XS", "E0", "wide"))
                 and np.isfinite(f["XS"]).all())
        # the kept rows are the unpoisoned results themselves
        r0 = orig(jobs[0])
        ok_bt &= np.array_equal(f["XS"][0], r0[3]) and np.array_equal(
            f["wide"][0], r0[5])
        f.close()
        poison = {j[:2] for j in jobs}
        try:
            data3.build_tbranches(os.path.join(SMOKE, "A.npz"), lib,
                                  jobs[:1], out, procs=1)
            raised = False
        except RuntimeError:
            raised = True
        ok_bt &= raised
    finally:
        data3.tbranches_one = orig
        shutil.rmtree(tmp, ignore_errors=True)
    report("9 review fixes", ok_lf and ok_tag and ok_br and ok_bt,
           f"LF endings {ok_lf} {crlf}; --out-tag guard {ok_tag} ({got}); "
           f"Branches drops non-finite moments {ok_br}; build_tbranches "
           f"drops them / refuses all-bad {ok_bt}")


def main():
    which = sys.argv[1:] or [str(i) for i in range(1, 10)]
    for w in which:
        t0 = time.time()
        globals()[f"test_{w}"]()
        print(f"      ({time.time() - t0:.1f} s)", flush=True)
    fails = RES.get("fails", [])
    print("ALL PASSED" if not fails else f"FAILED: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
