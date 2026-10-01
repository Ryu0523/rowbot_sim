#!/usr/bin/env python3
"""
DEFECTS M14 / M16 note: the meta4d models (model3.pt without waves,
model3_waves_m11.pt with the 15 elevations) were trained on 45-s episodes,
so token positions >= 179 were never trained, while their evaluations in
run_m14_diversity.sh used windows up to the full 256-token context. This
re-runs the same evaluations with EVERY context cut to at most CTX = 179
tokens (positions 0..178), on meta4d and on the baseline meta3 (M10 family,
90-s training episodes; its models see the same cut windows, so the pair is
like for like even though meta3 itself was trained on all 256 positions).
The test splits A, B, At, C, Cb and the branch files are byte-identical in
both caches, and every evaluation uses the same episodes, steps and seeds,
so meta3 - meta4d differences are paired.

  1  one step on A, B, At, C, Cb (skill, cov90, CRPS skill per channel):
     model3.pt in both caches, the wave model where it exists (meta4d only:
     meta3 has no model3_waves_m11.pt). Every step t >= 40 is scored once:
     window [0, 179) scores t < 179 (identical to the original for those
     steps, the network being causal), then windows [c + 45 - 179, c + 45)
     score t in [c, c + 45), so each scored step has 134..178 steps of
     history (the original's second window gave 153..255).
  2  multi step (physics rollouts) on A, B, At (branch files) and Cb (block
     starts), model3.pt, history window NH = 155 + the 24 generated steps
     (rollout_core reimplemented here with NH as a parameter; the original
     hard-codes NH = W_CTX - HB = 232). Moments k <= 155 are unchanged.
  3  check_wave_use_m14 (its skills() imported): split A, t in [240, 288),
     operator reads waves / does not, history 16, 64, 178.
  4  diag_context_m11's curves (same computation as skills()) on A, B, C,
     Cb, history 1, 4, 16, 64, 178.

Tail ("large errors") rates are not computed (q = None), so the training
split is never loaded. Results -> studies/_cache/meta4d/eval_ctx179.pkl
(written atomically after every section).

    python studies/eval_ctx_m14.py [--check]
"""
import argparse
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np   # noqa: E402
import torch   # noqa: E402

from learn.meta import model3 as M   # noqa: E402
import check_wave_use_m14 as CW   # noqa: E402  (read-only: skills())
import info_waves_m11 as IW   # noqa: E402  (read-only: DataW, NetW)

CTX = 179                       # trained positions of the 45-s episodes
NH = CTX - M.HB                 # 155 history tokens before a rollout
STRIDE = 45
LH_WAVE = (16, 64, CTX - 1)
LH_DIAG = (1, 4, 16, 64, CTX - 1)
CACHES = ("meta3", "meta4d")
NAMES = ("surge", "sway", "yaw", "heave", "pitch")
OUT = os.path.join(HERE, "_cache", "meta4d", "eval_ctx179.pkl")


def cpath(c):
    return os.path.join(HERE, "_cache", c)


def load(cache, wave=False):
    dev = M.device()
    f = "model3_waves_m11.pt" if wave else "model3.pt"
    p = os.path.join(cpath(cache), f)
    if not os.path.exists(p):
        return None, None
    ck = torch.load(p, weights_only=False)
    net = (IW.NetW() if wave else M.Net()).to(dev)
    net.load_state_dict(ck["net"])
    net.eval()
    return net, ck["stats"]


def data(cache, split, stats, wave=False):
    cls = IW.DataW if wave else M.Data3
    return cls(cpath(cache), split, M.device(), stats=stats)


def save(res):
    tmp = OUT + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(res, fh)
    os.replace(tmp, OUT)


@torch.no_grad()
def eval_one_step_cut(net, D, n_samp=64, start=40, batch=16, gen_seed=0,
                      L=CTX, stride=STRIDE):
    """M.eval_one_step with windows of at most L tokens (see the header)."""
    torch.manual_seed(gen_seed)                  # Tally's randomised PIT
    dev = D.dev
    gen = torch.Generator(device=dev).manual_seed(gen_seed)
    tal = M.Tally((M.C7,))
    for s0 in range(0, D.n, batch):
        ii = torch.arange(s0, min(D.n, s0 + batch), device=dev)
        Tn = int(D.len[ii].max())
        wins = [(0, start, L)] + [(c + stride - L, c, c + stride)
                                  for c in range(L, Tn, stride)]
        for a0, lo, hi in wins:
            hi = min(hi, Tn)
            if hi <= lo:
                continue
            a = torch.full_like(ii, a0)
            tok, tgt, ok = M.window_tokens(D, ii, a, L)
            h = net.encode(tok)
            pos = torch.arange(L, device=dev) + a0
            sel = ok & (pos[None] >= lo) & (pos[None] < hi)
            hs, ys = h[sel], tgt[sel]
            for c0 in range(0, len(hs), 2048):
                s = net.sample(hs[c0:c0 + 2048], n_samp, gen=gen)
                tal.add(s, ys[c0:c0 + 2048], None)
    return tal.summary()


@torch.no_grad()
def rollout_core_cut(net, D, ii, k, plans, xs_k, env, S, gen, nh=NH):
    """M._rollout_core (no grad, no obs noise, no steer) with the history
    window length nh as a parameter instead of W_CTX - HB."""
    from learn.meta.data3 import E7
    dev, st = D.dev, D.stats
    iit = torch.as_tensor(ii, device=dev).long()
    kt = torch.as_tensor(k, device=dev).long()
    pl = torch.as_tensor(plans, dtype=torch.float64, device=dev)
    B, P, Hh, _ = pl.shape
    assert Hh <= M.HB and nh + Hh <= M.W_CTX
    a = (kt - nh).clamp(min=0)
    Lh = kt - a
    tok_h, _, _ = M.window_tokens(D, iit, a, nh)
    hist, _ = M.kv_history(net, tok_h)
    hide = torch.arange(nh, device=dev)[None] >= Lh[:, None]
    e_prev = D.E[iit, (kt - 1).clamp(min=0)] * (kt >= 1)[:, None]
    e_prev = e_prev[:, None, None].expand(B, P, S, M.C7)
    xs = torch.as_tensor(xs_k, dtype=torch.float64, device=dev)
    amax = torch.tensor([env["t_max"], env["rud_max"]], dtype=torch.float64,
                        device=dev)
    act = (xs[:, 12:14] / amax)[:, None, None].expand(B, P, S, 2)
    sr = M.plant_to_reduced_t(xs)[:, None, None].expand(B, P, S, 10)
    gdev = gen.device
    base = torch.randn((B, P, S, Hh, M.C7), generator=gen,
                       device=gdev).to(dev)
    u_mu = torch.as_tensor(st["u_mu"], dtype=torch.float64, device=dev)
    u_sd = torch.as_tensor(st["u_sd"], dtype=torch.float64, device=dev)
    Un = ((pl - u_mu) / u_sd).float()
    e_sd = torch.as_tensor(st["e_sd"], device=dev)
    sel = torch.zeros(len(E7), 10, dtype=torch.float64, device=dev)
    sel[torch.arange(len(E7)), torch.tensor(E7)] = 1.0
    lo = torch.tensor([0.0, -1.0], dtype=torch.float64, device=dev)
    hi = torch.tensor([1.0, 1.0], dtype=torch.float64, device=dev)
    dtc = env["dt"] * env["sub"]
    m0 = M.model0t(env)
    n_st = len(E7)
    cache = [([], []) for _ in net.tf.layers]
    outs, states = [], []
    for j in range(Hh):
        X = M.state_tokens_t(sr, act, st, env).float()
        first = ((Lh + j) == 0).float().view(B, 1, 1).expand(B, P, S)
        Uj = pl[:, :, None, j]
        Un_j = Un[:, :, None, j].expand(B, P, S, 2)
        tok = M.make_tokens(X, Un_j, e_prev, first)
        h = M.kv_step(net, tok, net.pos[Lh + j].view(B, 1, 1, -1), hist,
                      hide, cache)
        e = net.sample_grad(h, base[:, :, :, j], 24)
        e_raw = (e * e_sd).double()
        sr = m0(sr, Uj) + (e_raw[..., :n_st] * dtc) @ sel
        act = torch.maximum(torch.minimum(
            Uj + e_raw[..., n_st:n_st + 2] * dtc, hi), lo)
        outs.append(e)
        states.append(sr)
        e_prev = e
    return torch.stack(outs, 3), torch.stack(states, 3)


@torch.no_grad()
def eval_multi_step_cut(net, D, br, env, n_samp=16, batch=48, gen_seed=0,
                        nh=NH):
    """M.eval_multi_step (no mask, q = None) through rollout_core_cut."""
    from learn.meta.data2 import plant_to_reduced
    from learn.meta.data3 import model0_step
    torch.manual_seed(gen_seed)
    dev = D.dev
    gen = torch.Generator(device=dev).manual_seed(gen_seed)
    NB, P, Hh, _ = br["U"].shape
    C7, SC = M.C7, M.SC_ES
    tal = M.Tally((Hh, C7))
    tal_s = M.Tally((Hh, len(SC)))
    U = br["U"].reshape(NB * P, Hh, 2)
    XS0 = br["XS"][:, :, 0].reshape(NB * P, 14)
    XT = plant_to_reduced(br["XS"][:, :, 1:].reshape(-1, 14)).reshape(
        NB * P, Hh, 10)
    E0 = br["E0"].reshape(NB * P, Hh, C7) / D.stats["e_sd"]
    ref_all = np.zeros_like(XT)
    for s0 in range(0, NB * P, 4096):
        sr = plant_to_reduced(XS0[s0:s0 + 4096].astype(float))
        for j in range(Hh):
            sr = model0_step(env, sr, U[s0:s0 + 4096, j].astype(float))
            ref_all[s0:s0 + 4096, j] = sr
    ssc = (XT - ref_all)[..., list(SC)].reshape(-1, len(SC)).std(0) + 1e-9
    ssc_t = torch.tensor(ssc, device=dev)
    f = lambda a: torch.as_tensor(  # noqa: E731
        a, device=dev)[..., list(SC)].div(ssc_t).float()
    per = max(1, batch // P)
    for s0 in range(0, NB, per):
        mb = np.arange(s0, min(NB, s0 + per))
        sl = (mb[:, None] * P + np.arange(P)[None]).ravel()
        e, sts = rollout_core_cut(net, D, br["ep"][mb], br["k"][mb],
                                  br["U"][mb].astype(float),
                                  br["XS"][mb, 0, 0].astype(float), env,
                                  n_samp, gen, nh)
        s = e.reshape(len(sl), n_samp, Hh, C7)
        tal.add(s, torch.tensor(E0[sl], dtype=torch.float32, device=dev))
        tal_s.add(f(sts.reshape(len(sl), n_samp, Hh, 10)), f(XT[sl]),
                  ref=f(ref_all[sl]))
    out = tal.summary()
    out["states"] = tal_s.summary(chans=len(SC))
    return out


def branches(cache, split, D):
    """meta_step3.phase_eval's multi-step truth: the branch file for A, B,
    At; block starts (k = 48, 72, ...) for Cb."""
    if split != "Cb":
        b = np.load(os.path.join(cpath(cache), f"{split}_branches.npz"))
        return {k_: b[k_] for k_ in b.files}
    ep, kk, Us, XS0, E0 = [], [], [], [], []
    E0all = np.load(os.path.join(cpath(cache), "Cb_e0.npz"))["E0"]
    for i in range(D.n):
        for k in range(48, int(D.len[i]) - M.HB + 1, M.HB):
            ep.append(i)
            kk.append(k)
            Us.append(D.Uraw[i, k:k + M.HB])
            XS0.append(D.XS[i, k:k + M.HB + 1])
            E0.append(E0all[i, k:k + M.HB])
    return dict(ep=np.array(ep), k=np.array(kk), U=np.stack(Us)[:, None],
                XS=np.stack(XS0)[:, None], E0=np.stack(E0)[:, None])


def f5(x):
    return " ".join(f"{v:.2f}" for v in x[:5])


def free():
    torch.cuda.empty_cache()


def check():
    """rollout_core_cut(nh=232) == M.rollout_core on a few A moments, and
    the cut one-step on 16 A episodes runs."""
    from learn.meta import relabel
    net, st = load("meta4d")
    D = data("meta4d", "A", st)
    br = branches("meta4d", "A", D)
    env = relabel._env()
    mb = np.array([0, 1, 60, 61])                # k = 120 and 240
    args = (br["ep"][mb], br["k"][mb], br["U"][mb, :2].astype(float),
            br["XS"][mb, 0, 0].astype(float), env, 4)
    g1 = torch.Generator(device=D.dev).manual_seed(0)
    g2 = torch.Generator(device=D.dev).manual_seed(0)
    e1, _, _ = M.rollout_core(net, D, *args, gen=g1)
    e2, _ = rollout_core_cut(net, D, *args, gen=g2, nh=M.W_CTX - M.HB)
    print("k", br["k"][mb], "max |orig - copy| (nh 232):",
          float((e1 - e2).abs().max()))
    D.n = 16
    r = eval_one_step_cut(net, D)
    print("one step, 16 A episodes:", f5(r["skill"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
        return
    from learn.meta import relabel
    env = relabel._env()
    t0 = time.time()
    res = dict(ctx=CTX, nh=NH, stride=STRIDE, one={}, multi={}, wave={},
               diag={})
    # 1 one step
    for cache in CACHES:
        for wave in (False, True):
            net, st = load(cache, wave)
            if net is None:
                print(f"[{cache}] no {'wave' if wave else 'no-wave'} model,"
                      " skipped", flush=True)
                continue
            for split in ("A", "B", "At", "C", "Cb"):
                D = data(cache, split, st, wave)
                r = eval_one_step_cut(net, D)
                res["one"][(cache, "waves" if wave else "model3", split)] = r
                print(f"[{time.time() - t0:5.0f} s] one step {cache} "
                      f"{'waves ' if wave else 'model3'} {split}: skill "
                      f"{f5(r['skill'])} | cov90 {f5(r['cov'])} | CRPS "
                      f"{f5(r['crps'])}", flush=True)
                del D
                free()
            del net
            free()
    save(res)
    # 2 multi step
    for cache in CACHES:
        net, st = load(cache)
        for split in ("A", "B", "At", "Cb"):
            D = data(cache, split, st)
            r = eval_multi_step_cut(net, D, branches(cache, split, D), env)
            res["multi"][(cache, split)] = r
            print(f"[{time.time() - t0:5.0f} s] multi step {cache} {split}:"
                  f"\n" + M.fmt_multi(r, "rollout"), flush=True)
            del D
            free()
    save(res)
    # 3 wave-use check (split A)
    import pickle as pk
    meta = pk.load(open(os.path.join(cpath("meta4d"), "A_meta.pkl"), "rb"))
    wv = np.array([any(m["style"].get("wave_in", [])) and
                   not m["style"].get("null", False) for m in meta])
    for cache in CACHES:
        for wave in (False, True):
            net, st = load(cache, wave)
            if net is None:
                continue
            D = data(cache, "A", st, wave)
            okk = (D.len > CW.T0 + CW.NT + 1).cpu().numpy()
            for gname, gm in (("reads", okk & wv), ("not", okk & ~wv)):
                eps = torch.as_tensor(np.flatnonzero(gm), device=D.dev)
                for lh in LH_WAVE:
                    gen = torch.Generator(device=D.dev).manual_seed(0)
                    res["wave"][(cache, "waves" if wave else "model3",
                                 gname, lh)] = CW.skills(net, D, eps, lh, gen)
                res["wave"][("n", gname)] = int(gm.sum())
            del D, net
            free()
    save(res)
    # 4 history-length curves (model3.pt)
    for cache in CACHES:
        net, st = load(cache)
        for split in ("A", "B", "C", "Cb"):
            D = data(cache, split, st)
            eps = torch.nonzero(D.len > CW.T0 + CW.NT + 1)[:, 0]
            for lh in LH_DIAG:
                gen = torch.Generator(device=D.dev).manual_seed(0)
                res["diag"][(cache, split, lh)] = CW.skills(net, D, eps, lh,
                                                            gen)
            del D
            free()
    save(res)
    report(res)
    print(f"done in {time.time() - t0:.0f} s -> {OUT}")


def report(res):
    print("\n=== 1. one step, context <= 179 (skill | cov90 | CRPS skill; "
          "surge sway yaw heave pitch)")
    for split in ("A", "B", "At", "C", "Cb"):
        print(f"  {split}:")
        for key in (("meta3", "model3"), ("meta4d", "model3"),
                    ("meta4d", "waves")):
            r = res["one"].get((*key, split))
            if r is not None:
                print(f"    {key[0]:6s} {key[1]:6s}  {f5(r['skill'])} | "
                      f"{f5(r['cov'])} | {f5(r['crps'])}")
    print("\n=== 2. multi step, history 155 + 24 (model3.pt; error skill "
          "and rolled-state skill [u v r z theta psi], meta3 / meta4d)")
    for split in ("A", "B", "At", "Cb"):
        print(f"  {split}:")
        for a, b in ((0, 1), (1, 5), (5, M.HB)):
            r3, r4 = res["multi"][("meta3", split)], res["multi"][
                ("meta4d", split)]
            print(f"    steps {a}-{b - 1}: err " + "  ".join(
                f"{x:.2f}/{y:.2f}" for x, y in zip(
                    r3["skill"][a:b, :5].mean(0), r4["skill"][a:b, :5].mean(
                        0))) + " | states " + "  ".join(
                f"{x:.2f}/{y:.2f}" for x, y in zip(
                    r3["states"]["skill"][a:b].mean(0),
                    r4["states"]["skill"][a:b].mean(0))))
    print("\n=== 3. wave use, A, t in [240, 288): skill no-wave / wave "
          "(meta4d), [meta3 no-wave]")
    for g, lab in (("reads", "operator reads waves"),
                   ("not", "operator does not")):
        print(f"  {lab} ({res['wave'][('n', g)]} episodes):")
        for lh in LH_WAVE:
            a = res["wave"][("meta4d", "model3", g, lh)]
            b = res["wave"].get(("meta4d", "waves", g, lh))
            c = res["wave"][("meta3", "model3", g, lh)]
            print(f"    history {lh:3d}: " + "  ".join(
                f"{n} {x:.2f}/{y:.2f} [{z:.2f}]" for n, x, y, z in zip(
                    NAMES, a, b if b is not None else [np.nan] * 5, c)))
    print("\n=== 4. history-length curves, t in [240, 288), model3.pt: "
          "meta3 / meta4d")
    for split in ("A", "B", "C", "Cb"):
        print(f"  {split}:")
        for lh in LH_DIAG:
            a, b = res["diag"][("meta3", split, lh)], res["diag"][
                ("meta4d", split, lh)]
            print(f"    history {lh:3d}: " + "  ".join(
                f"{n} {x:.2f}/{y:.2f}" for n, x, y in zip(NAMES, a, b)))


if __name__ == "__main__":
    main()
