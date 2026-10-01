#!/usr/bin/env python3
"""
Tests of the cheaper learned-error controllers (learn/meta/mpc_variants.py)
in one process, minutes; data: studies/_cache/meta3/model3_cov.pt, read
only.

   1  with a zero error model (sample_grad patched to return zeros) hold,
      hold constant, nominal and sens reduce to m0 (LearnedMPPI, net None):
      two captured calls, states and per-sample costs bit-identical to
      m0's (repeated over S), commands equal; a 1.5 s closed loop of each
      gives m0's step log and score
   2  sens with zero nudge sensitivity = nominal: an error model that
      returns its base noise (plan independent, so every finite difference
      is exactly 0): D = 0, the candidates' errors and costs bit-identical
      to nominal's in a captured call, and the 1.5 s closed loops agree
   3  nominal vs the full controller (real net, live history after one
      step): a candidate equal to the warm-start plan, rolled by the full
      rollout_core with the same base noise among 127 others, gets the
      errors and states nominal gives every candidate (the one-plan call
      reproduces nominal's errors bit-identically; the 128-plan batch only
      differs by float32 batch rounding)
   4  sens_combine on an exactly linear synthetic error model: recovers
      the candidates' errors inside the trust region (< 1e-12), holds the
      boundary value outside it, caps the correction at c_max e_sd; the
      nudge sign flips at the upper bound
   5  hold: its errors are e0 x w_j (w_0 = 1, exp(-j dtc / tau); constant
      for tau = inf) with e0 = nominal's step-0 errors for the same row,
      base and warm start (bit-identical); sequences per call S (hold,
      nominal) and 8 S (sens)
   6  LF line endings of the new files
   7  sens on a synthetic error map exactly linear in each sample's plan
      thrust (_learned patched): candidates inside the trust region get
      the map evaluated on their own clipped knots (< 1e-12), a candidate
      = warm + delta on knot i gets nudged plan 1 + i's errors, the nudged
      knots and signs are the ones applied; through __call__ (trust off)
      the captured candidates' errors equal the map on their knots
   8  the main results file read through a temporary hard link
      (_open_linked): the writer's os.replace onto it succeeds while it is
      open, the reader gets the old bytes, the link is removed (a plain
      open() blocks the replace on Windows, even with FILE_SHARE_DELETE)
   (3's same1 / shared and 5's prod / const restate the implementation;
   the checks that can fail on a wrong design are 3's dK and dS, 5's
   step0, 1, 2, 4 and 7)

    python -m studies.test_mpc_variants [1 2 ...]
"""
import argparse
import math
import os
import time

import numpy as np
import torch

from learn.meta import model3 as M
from learn.meta import mpc_learned as ML
from learn.meta import mpc_variants as MV

HERE = os.path.dirname(os.path.abspath(__file__))
META3 = os.path.join(HERE, "_cache", "meta3")
RES = {}
KINDS = (("hold", dict(tau=1.0)), ("hold", dict(tau=math.inf)),
         ("nominal", {}), ("sens", {}))


def _setup():
    if "net" not in RES:
        dev = M.device(0.2)
        ck = torch.load(os.path.join(META3, "model3_cov.pt"),
                        weights_only=False)
        RES.update(dev=dev, ck=ck, net=ML.load_net(ck, dev), env=ML.env())
    return RES["dev"], RES["net"], RES["env"]


def report(name, ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
    RES.setdefault("fails", [])
    if not ok:
        RES["fails"].append(name)


def _net(kind):
    """The real net with sample_grad replaced: 'zero' returns 0, 'base'
    returns its base noise (independent of history and plan)."""
    net = ML.load_net(RES["ck"], RES["dev"])
    if kind == "zero":
        net.sample_grad = lambda h, y0, steps=24: torch.zeros_like(y0)
    else:
        net.sample_grad = lambda h, y0, steps=24: y0.clone()
    return net


JOBS = [dict(seed=0, leg=0), dict(seed=1, leg=1)]


def _group(jobs=JOBS, T=2.5):
    from learn.repro.task import LEGS, Mission
    ms = [Mission("high", j["seed"], j["leg"], t_end=T, track=LEGS[j["leg"]])
          for j in jobs]
    live = ML.LiveData(len(jobs), ms[0].n_ctrl + 1, RES["ck"]["stats"],
                       RES["dev"], ML.mission_consts(ms[0]))
    for b, m in enumerate(ms):
        live.start(b, m.s)
    return ms, live


def _advance(ms, live, cmds):
    for b, m in enumerate(ms):
        thr = float(cmds[b]) * m.t_max
        rud = m.ep._steer(m.s, thr)
        m.advance(thr, rud)
        live.push(b, (thr / m.t_max, rud / m.rud_max), m.s)


def _calls(make, n=2):
    """n captured calls of a controller on its own fresh group, advancing
    the plants with its commands in between."""
    ms, live = _group()
    ctrl = make(ms, live)
    ctrl.capture = True
    out = []
    for k in range(n):
        cmds = ctrl([0, 1], k)
        out.append(dict(ctrl.last))
        _advance(ms, live, cmds)
    return out, ctrl


LINK = ML.ImpactLink(0.4, 1.0, A1=0.02, j_imp=12)


def _close(a, b):
    """Two closed-loop rows: bit-identical step logs and score."""
    return all(np.array_equal(a["log"][q], b["log"][q])
               for q in ("u", "zd", "thr")) and a["score"] == b["score"]


# ------------------------------------------------------------------ 1
def test_1():
    dev, net, env = _setup()
    net0 = _net("zero")
    ref, _ = _calls(lambda ms, live: ML.LearnedMPPI(JOBS, ms, env, LINK))
    r0 = ML.run_group(JOBS, env, LINK, T=1.5, log=lambda *a: None)
    ok, msg = True, []
    for var, kw in KINDS:
        got, ctrl = _calls(lambda ms, live: MV.VariantMPPI(
            JOBS, ms, env, LINK, net=net0, live=live, variant=var, **kw))
        s_eq = c_eq = True
        dc = 0.0
        for g, r in zip(got, ref):
            s_eq &= torch.equal(g["states"], r["states"].expand_as(
                g["states"]))
            c_eq &= torch.equal(g["cost"], r["cost"].expand_as(g["cost"]))
            dc = max(dc, float(np.abs(g["cmds"] - r["cmds"]).max()))
        rows = MV.run_group_v(JOBS, env, LINK, var, net0, RES["ck"]["stats"],
                              T=1.5, log=lambda *a: None, **kw)
        loop = all(_close(a, b) for a, b in zip(rows, r0))
        ok &= s_eq and c_eq and dc == 0.0 and loop
        msg.append(f"{var}{kw.get('tau', '')}: states {s_eq}, costs {c_eq},"
                   f" commands diff {dc:.1e}, closed loop {loop}")
    report("1 zero error model = m0", ok, "; ".join(msg))


# ------------------------------------------------------------------ 2
def test_2():
    dev, net, env = _setup()
    netb = _net("base")
    mk = lambda var: (lambda ms, live: MV.VariantMPPI(   # noqa: E731
        JOBS, ms, env, LINK, net=netb, live=live, variant=var))
    nom, _ = _calls(mk("nominal"))
    sen, _ = _calls(mk("sens"))
    d0 = all(bool((s["D"] == 0).all()) for s in sen)
    e_eq = all(torch.equal(s["e_raw"], n["e_raw"].expand_as(s["e_raw"]))
               for s, n in zip(sen, nom))
    c_eq = all(torch.equal(s["cost"], n["cost"]) for s, n in zip(sen, nom))
    rn = MV.run_group_v(JOBS, env, LINK, "nominal", netb, RES["ck"]["stats"],
                        T=1.5, log=lambda *a: None)
    rs = MV.run_group_v(JOBS, env, LINK, "sens", netb, RES["ck"]["stats"],
                        T=1.5, log=lambda *a: None)
    loop = all(_close(a, b) for a, b in zip(rs, rn))
    fin = all(np.isfinite(r["score"]) for r in rs)
    report("2 sens with zero sensitivity = nominal",
           d0 and e_eq and c_eq and loop and fin,
           f"D == 0 {d0}, candidate errors bit-identical {e_eq}, costs {c_eq}"
           f", closed loops identical {loop} (finite {fin}, scores "
           f"{[round(r['score'], 4) for r in rs]})")


# ------------------------------------------------------------------ 3
def test_3():
    dev, net, env = _setup()
    got, ctrl = _calls(lambda ms, live: MV.VariantMPPI(
        JOBS, ms, env, LINK, net=net, live=live, variant="nominal",
        spread=1.25), n=2)
    cap = got[1]                                # k = 1: one step of history
    ms, live = ctrl.ms, ctrl.live
    e_sd = torch.as_tensor(RES["ck"]["stats"]["e_sd"], device=dev)
    warm = cap["warm"] @ ctrl.E.T                          # (2, H)
    full = cap["plans"].copy()
    full[:, 0, :, 0] = warm
    base = ctrl.base_tensor(list(cap["base_rows"]))
    steer = lambda: ML.SteerT(ms[0].ep, cap["phi"], cap["psi_i"],  # noqa
                              env)
    e1, _, _ = M.rollout_core(net, live, [0, 1], [1, 1],
                              torch.tensor(full[:, :1]), cap["xs"], env,
                              ctrl.S, base=base[:, :1], steer=steer())
    eK, sK, _ = M.rollout_core(net, live, [0, 1], [1, 1], torch.tensor(full),
                               cap["xs"], env, ctrl.S, base=base,
                               steer=steer())
    e1r = (e1 * e_sd).double().cpu()
    eKr = (eK[:, :1] * e_sd).double().cpu()
    same1 = torch.equal(e1r, cap["e_raw"])
    dK = float((eKr - cap["e_raw"]).abs().max())
    scale = float(cap["e_raw"].abs().max())
    s_nom, _ = MV.rollout_with_e(torch.tensor(full[:, :1]), cap["xs"], env,
                                 cap["e_raw"], steer())
    s_nomK, _ = MV.rollout_with_e(torch.tensor(full[:, :1]), cap["xs"], env,
                                  eKr, steer())
    dS = float((s_nomK[:, 0] - sK[:, 0].cpu()).abs().max())
    dS_nom = float((s_nom[:, 0] - sK[:, 0].cpu()).abs().max())
    shared = cap["e_raw"].shape[1] == 1        # one set for every candidate
    ok = same1 and dK <= 1e-5 * max(scale, 1.0) and dS < 1e-9 and shared
    report("3 nominal = full controller on the warm-start candidate", ok,
           f"one-plan rollout_core errors bit-identical {same1}; in the 128-"
           f"plan batch max |de| {dK:.1e} (errors up to {scale:.1f}); states"
           f" of rollout_with_e vs rollout_core with the same errors "
           f"{dS:.1e} (with nominal's errors {dS_nom:.1e}); nominal's errors "
           f"shared by all candidates {shared}")


# ------------------------------------------------------------------ 4
def test_4():
    rng = np.random.default_rng(0)
    B, K, S, H, C = 2, 64, 3, 5, 10
    e0 = torch.tensor(rng.normal(size=(B, 1, S, H, C)))
    A = torch.tensor(rng.normal(size=(B, 7, S, H, C)))
    warm = rng.uniform(0.2, 1.0, (B, 7))
    warm[0, 3] = 0.99                          # +delta would leave [0, 1]
    dsign = np.where(warm + 0.05 <= 1.0, 0.05, -0.05)
    e8 = torch.cat([e0, e0 + A * torch.tensor(dsign)[:, :, None, None,
                                                     None]], 1)
    dk = rng.uniform(-0.2, 0.2, (B, K, 7))
    big = np.full(C, 1e9)
    e, D = MV.sens_combine(e8, torch.tensor(dsign), dk, 0.25, big)
    ref = e0 + torch.einsum("bki,bishc->bkshc", torch.tensor(dk), A)
    d_in = float((e - ref).abs().max())
    dD = float((D - A).abs().max())
    dk2 = dk.copy()
    dk2[..., 2] = 0.9                          # outside the trust region
    e2, _ = MV.sens_combine(e8, torch.tensor(dsign), dk2, 0.25, big)
    dk3 = dk.copy()
    dk3[..., 2] = 0.25
    e3, _ = MV.sens_combine(e8, torch.tensor(dsign), dk3, 0.25, big)
    d_tr = float((e2 - e3).abs().max())
    cap = np.full(C, 0.1)
    e4, _ = MV.sens_combine(e8, torch.tensor(dsign), dk, 0.25, cap)
    d_cap = float((e4 - e0).abs().max())
    flip = dsign[0, 3] < 0
    ok = d_in < 1e-12 and dD < 1e-12 and d_tr == 0.0 and \
        d_cap <= 0.1 + 1e-15 and flip
    report("4 sens_combine on a linear model", ok,
           f"inside the trust region {d_in:.1e}, D {dD:.1e}; beyond it held "
           f"at the boundary {d_tr:.1e}; capped |corr| max {d_cap:.3f} "
           f"(cap 0.1); nudge sign flips at the bound {flip}")


# ------------------------------------------------------------------ 5
def test_5():
    dev, net, env = _setup()
    out = {}
    for var, kw in KINDS:
        got, ctrl = _calls(lambda ms, live: MV.VariantMPPI(
            JOBS, ms, env, LINK, net=net, live=live, variant=var, **kw), n=1)
        out[(var, kw.get("tau"))] = (got[0], ctrl)
    (h, ch), (hc, _) = out[("hold", 1.0)], out[("hold", math.inf)]
    (n, _), (s, cs) = out[("nominal", None)], out[("sens", None)]
    w = MV.decay_weights(1.0, env)
    dtc = env["dt"] * env["sub"]
    w_ok = w[0].item() == 1.0 and abs(w[4].item() - math.exp(-4 * dtc)) \
        < 1e-15
    prod = torch.equal(h["e_raw"], h["e0"] * w.view(1, 1, 1, ML.H, 1))
    const = torch.equal(hc["e_raw"], hc["e0"].expand_as(hc["e_raw"]))
    step0 = torch.equal(h["e0"][:, :, :, 0], n["e_raw"][:, :, :, 0])
    nseq = (ch.n_seq, out[("nominal", None)][1].n_seq, cs.n_seq)
    ok = w_ok and prod and const and step0 and nseq == (4, 4, 32) and \
        (ch.depth, cs.depth) == (1, ML.H)
    report("5 hold = e0 x decay", ok,
           f"weights {w_ok} (w_4 = {w[4]:.3f}); e = e0 w {prod}; constant "
           f"hold {const}; e0 = nominal's step-0 errors {step0}; sequences "
           f"per call (hold, nominal, sens) {nseq}, depth hold {ch.depth} "
           f"sens {cs.depth}")


# ------------------------------------------------------------------ 7
def test_7():
    dev, net, env = _setup()
    rng = np.random.default_rng(1)
    nb, H, C = len(JOBS), ML.H, M.C7

    def make():
        ms, live = _group()
        c = MV.VariantMPPI(JOBS, ms, env, LINK, net=net, live=live,
                           variant="sens")
        c.cap = torch.full((C,), 1e9, dtype=torch.float64)     # no cap
        c._learned = fake
        return c

    S = ML.S_SAMP
    a = torch.tensor(rng.normal(0.0, 0.1, (nb, S, H, C)))
    G = torch.tensor(rng.normal(0.0, 0.1, (nb, S, H, C, H)))
    E = ML.knot_matrices()[0]
    calls = []

    def emap(knots):
        """The synthetic error map on knots (nb, P, 7): a + G thr per
        sample, exactly linear in the plan's thrust."""
        thr = torch.tensor(np.asarray(knots) @ E.T)            # (nb, P, H)
        return a[:, None] + torch.einsum("bshcj,bpj->bpshc", G, thr)

    def fake(rows, k, knots, base, xs, steer, Hh):
        calls.append(np.array(knots))
        return emap(knots)[..., :Hh, :]

    # errors() on candidates inside the trust region + one per nudged knot
    ctrl = make()
    warm = rng.uniform(0.3, 0.9, (nb, 7))
    warm[0, 3] = 0.98                          # +delta would leave [0, 1]
    ctrl.nominal[:] = warm
    dsign = np.where(warm + ctrl.delta <= 1.0, ctrl.delta, -ctrl.delta)
    off = rng.uniform(-0.2, 0.2, (nb, 64, 7))
    cand = np.clip(warm[:, None] + off, ctrl.floor, 1.0)
    one = np.repeat(warm[:, None], 7, 1)
    one[:, np.arange(7), np.arange(7)] += dsign
    cand = np.concatenate([cand, one], 1)                   # (nb, 71, 7)
    bases = [ctrl.draws(b)[1] for b in range(nb)]
    e, info = ctrl.errors([0, 1], 0, cand, bases, None, None)
    d_lin = float((e - emap(cand)).abs().max())
    d_one = float((e[:, 64:] - info["e8"][:, 1:]).abs().max())
    d_ds = float(np.abs(info["dsign"] - dsign).max())
    d_kn = float(np.abs(calls[-1][:, 1:] - one).max())
    flip = info["dsign"][0, 3] < 0
    # the whole call (trust region off): the captured candidates' errors
    ctrl = make()
    ctrl.trust = 1e9
    ctrl.capture = True
    ctrl.nominal[:] = warm
    ctrl([0, 1], 0)
    cp = ctrl.last
    d_call = float((cp["e_raw"] - emap(cp["cand"])).abs().max())
    scale = float(emap(cand).abs().max())
    ok = d_lin < 1e-12 and d_one < 1e-12 and d_ds == 0.0 and d_kn < 1e-15 \
        and flip and d_call < 1e-12
    report("7 sens on an error map linear in the plan", ok,
           f"candidates inside the trust region vs the map on their knots "
           f"{d_lin:.1e} (errors up to {scale:.2f}); warm + delta on one knot"
           f" vs that nudged plan's errors {d_one:.1e}; nudged knots "
           f"{d_kn:.1e}, dsign {d_ds:.1e}, sign flips at the bound {flip}; "
           f"through __call__ (captured candidates, no trust clip) {d_call:.1e}")


# ------------------------------------------------------------------ 8
def test_8():
    import tempfile
    from studies import mpc_compare_variants as MCV
    d = tempfile.mkdtemp()
    p = os.path.join(d, "results.pkl")

    def put(b):
        with open(p + ".tmp", "wb") as f:
            f.write(b)

    put(b"old")
    os.replace(p + ".tmp", p)
    with MCV._open_linked(p, d) as f:
        linked = f.name != p
        put(b"new")
        try:
            os.replace(p + ".tmp", p)          # the main run's save
            rep = True
        except PermissionError:
            rep = False
        old = f.read()
    gone = os.listdir(d) == ["results.pkl"]
    with open(p, "rb") as g:
        now = g.read()
    g = open(p, "rb")                          # plain open, for contrast
    put(b"newer")
    try:
        os.replace(p + ".tmp", p)
        plain = "replace ok"
    except PermissionError:
        plain = "replace blocked"
    g.close()
    for q in (p, p + ".tmp"):
        if os.path.exists(q):
            os.remove(q)
    os.rmdir(d)
    ok = linked and rep and old == b"old" and now == b"new" and gone
    report("8 main results read without blocking its save", ok,
           f"read through a hard link {linked}; os.replace onto the file "
           f"while it is open {rep}, reader got {old!r}, file now {now!r}, "
           f"link removed {gone}; with a plain open(): {plain}")


# ------------------------------------------------------------------ 6
def test_6():
    root = os.path.dirname(HERE)
    files = ("learn/meta/mpc_variants.py", "studies/mpc_compare_variants.py",
             "studies/test_mpc_variants.py")
    bad = [f for f in files if b"\r\n" in open(os.path.join(root, f),
                                               "rb").read()]
    report("6 LF line endings", not bad, f"CRLF in {bad}" if bad else
           f"{len(files)} files LF")


TESTS = {1: test_1, 2: test_2, 3: test_3, 4: test_4, 5: test_5, 6: test_6,
         7: test_7, 8: test_8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", type=int)
    a = ap.parse_args()
    t0 = time.time()
    for i in a.which or sorted(TESTS):
        t1 = time.time()
        try:
            TESTS[i]()
        except Exception as ex:           # a crash is a failure, go on
            import traceback
            traceback.print_exc()
            report(f"{i} (crashed)", False, repr(ex))
        print(f"    ({time.time() - t1:.0f} s)", flush=True)
    fails = RES.get("fails", [])
    print(f"\n{'ALL PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}"
          f" ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
