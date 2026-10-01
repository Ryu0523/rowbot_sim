#!/usr/bin/env python3
"""
Tests of the learned-model MPC (learn/meta/mpc_learned.py, the steer hook
in model3.rollout_core, studies/mpc_compare.py), in one process, minutes;
data: studies/_cache/meta3 (Cb, model3_cov.pt), read only.

   1  LiveData fed a recorded Cb episode's XS and U reproduces Data3's X,
      U, E (atol 1e-5) and window_tokens at k in {0, 1, 100, 300}; the
      rollout from the buffer (full and filled only up to k) vs from Data3
      with the same base: step-0 samples within 1e-3
   2  rollout_zero = a numpy loop of data3.model0_step (< 1e-9), and with
      steer = the loop driven by deep copies of the live Episode's _steer
   3  rollout_zero = rollout_core with a zero-error net (sample_grad
      patched to return zeros), with and without steer (< 1e-9)
   4  rollout_core(steer=None) bit-identical to the backed-up
      _rollout_core (scratchpad/backup_pre_mpc/model3.py)
   5  SteerT = Episode._steer incl. the conditional integral and the
      heading wrap, random states and thrusts over 24 steps (< 1e-12)
   6  CRN: duplicated candidates get identical costs (exact), the base is
      one draw per row expanded over the candidates (stride 0); a row's
      base / knot draws and its m0 closed loop do not depend on the group
      it runs in (exact); base_tensor = s x the row draws for a scalar s
      and for one s per horizon step (exact)
   7  the controller's call = a direct rollout_core call with the captured
      arguments (bit-identical states and costs)
   8  task_cost on hand-built states (speed, impact link incl. the speed-
      dependent threshold and j_imp, huber track, smoothness); the
      threshold reads the measured speed only (rolled speeds leave the
      impact part bit-identical); knot expansion and shift =
      MPPIController._expand / _shift (< 1e-12)
   9  adapt_head changes only head parameters (every other tensor bitwise
      equal) and trains on W_CTX windows (time split: 2 episodes, 50
      steps; whole-episode holdout: 4 episodes, 20 steps); on a pool whose
      errors are shifted by 2 e_sd (the head must learn it), with a
      12-step episode among the training ones: the head changes, a step
      > 0 is kept, the validation FM loss falls, windows stay W_CTX long
  10  short closed loops (2.5 s): hand, c0, m0, prior in lockstep of 2;
      finite, metrics present, per-step time
  11  LF line endings of the new / edited files
  12  fit_impact_link on synthetic logs: recovers a known single (A, K),
      and switches to a speed-dependent threshold when the true one moves
      with speed
  13  impact_horizon on a synthetic check: a model equal to the truth and
      one with a heavy link-cost tail but the right exceedance pass, one
      3x too high from step 10 sets j_imp 8-10, a non-deciding m0 does not
      move it; a deciding model bad from step 0 is reported as early; a
      model without the growth with speed fails at step 0
  14  phase_hcheck (stubbed nets, synthetic check): hcheck_pre computes m0,
      prior, oracle; adding prior_s and online computes only those and
      re-decides j_imp; a rerun is cached; a changed spread recomputes that
      model; phase_eval refuses rows run with another j_imp; a model bad
      from step 0 stops the run, --hcheck-continue goes on

    python -m studies.test_mpc_learned [1 2 ...]
"""
import argparse
import copy
import importlib.util
import os
import time

import numpy as np
import torch

from learn.meta import data3
from learn.meta import model3 as M
from learn.meta import mpc_learned as ML

HERE = os.path.dirname(os.path.abspath(__file__))
META3 = os.path.join(HERE, "_cache", "meta3")
BACKUP = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Temp", "claude",
                      "C--Users-Administrator-Documents",
                      "97e7fd1d-c515-4a09-869b-41596c911be0", "scratchpad",
                      "backup_pre_mpc", "model3.py")
RES = {}


def _setup():
    if "net" not in RES:
        from learn.repro.task import Mission
        dev = M.device(0.35)
        ck = torch.load(os.path.join(META3, "model3_cov.pt"),
                        weights_only=False)
        net = ML.load_net(ck, dev)
        m = Mission("high", 0, 0, t_end=20.0, track=0.0)
        RES.update(dev=dev, ck=ck, net=net, env=ML.env(), m=m,
                   D=M.Data3(META3, "Cb", dev, stats=ck["stats"]))
    return RES["dev"], RES["net"], RES["env"], RES["D"]


def report(name, ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
    RES.setdefault("fails", [])
    if not ok:
        RES["fails"].append(name)


def _feed(D, i, n, stats, dev):
    """A LiveData row fed Cb episode i's recorded XS and U for n steps."""
    live = ML.LiveData(1, D.T + 1, stats, dev, ML.mission_consts(RES["m"]))
    live.start(0, D.XS[i, 0].astype(float))
    for k in range(n):
        live.push(0, D.Uraw[i, k].astype(float), D.XS[i, k + 1].astype(float))
    return live


def _base(B, P, S, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((B, 1, S, M.HB, M.C7), generator=g).expand(
        B, P, S, M.HB, M.C7)


# ------------------------------------------------------------------ 1
def test_1():
    dev, net, env, D = _setup()
    st = RES["ck"]["stats"]
    L = D.len.cpu().numpy()
    NH = M.W_CTX - M.HB
    worst = dict(X=0.0, U=0.0, E=0.0, tok=0.0, roll=0.0, roll_part=0.0)
    for i in (0, 5):
        n = int(L[i])
        live = _feed(D, i, n, st, dev)
        worst["X"] = max(worst["X"], (live.X[0, :n] - D.X[i, :n]).abs()
                         .max().item())
        worst["U"] = max(worst["U"], (live.U[0, :n] - D.U[i, :n]).abs()
                         .max().item())
        worst["E"] = max(worst["E"], (live.E[0, :n] - D.E[i, :n]).abs()
                         .max().item())
        ok_v = bool((live.valid[0, :n] == D.valid[i, :n]).all())
        for k in (0, 1, 100, 300):
            if k + M.HB > n:
                continue
            a = torch.tensor([max(0, k - NH)], device=dev)
            t1, _, _ = M.window_tokens(live, torch.tensor([0], device=dev),
                                       a, NH)
            t2, _, _ = M.window_tokens(D, torch.tensor([i], device=dev), a,
                                       NH)
            m_ = int(min(n, int(a) + NH) - int(a))
            worst["tok"] = max(worst["tok"], (t1[0, :m_] - t2[0, :m_]).abs()
                               .max().item())
            pl = torch.tensor(D.Uraw[i, k:k + M.HB][None, None],
                              dtype=torch.float64)
            xs = D.XS[i, k][None].astype(float)
            base = _base(1, 1, 4, seed=k)
            e1, _, _ = M.rollout_core(net, live, [0], [k], pl, xs, env, 4,
                                      base=base)
            e2, _, _ = M.rollout_core(net, D, [i], [k], pl, xs, env, 4,
                                      base=base)
            part = _feed(D, i, k, st, dev)       # filled only up to k
            e3, _, _ = M.rollout_core(net, part, [0], [k], pl, xs, env, 4,
                                      base=base)
            worst["roll"] = max(worst["roll"], (e1[..., 0, :] - e2[..., 0, :])
                                .abs().max().item())
            worst["roll_part"] = max(worst["roll_part"], (
                e3[..., 0, :] - e2[..., 0, :]).abs().max().item())
    ok = ok_v and all(worst[k] < 1e-5 for k in ("X", "U", "E", "tok")) and \
        worst["roll"] < 1e-3 and worst["roll_part"] < 1e-3
    report("1 LiveData vs Data3", ok,
           ", ".join(f"{k} {v:.1e}" for k, v in worst.items())
           + f"; valid equal {ok_v}")


# ------------------------------------------------------------------ 2
def _moments(D, n, seed):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, D.n, n)
    k = rng.integers(0, 300, n)
    return i, k


def _plans(rng, B, P):
    pl = np.zeros((B, P, M.HB, 2))
    pl[..., 0] = rng.uniform(0, 1, (B, P, M.HB))
    pl[..., 1] = rng.uniform(-1, 1, (B, P, M.HB))
    return pl


def _ref_loop(env, xs, pl, eps=None):
    """numpy model0_step loop over plans (B, P, H, 2) from xs (B, 14); eps:
    per (b, p) Episode copies whose _steer sets the nozzle."""
    from learn.meta.data2 import plant_to_reduced
    B, P = pl.shape[:2]
    sr = np.repeat(plant_to_reduced(xs), P, 0)
    U = pl.reshape(B * P, M.HB, 2).copy()
    out = np.zeros((B * P, M.HB, 10))
    for j in range(M.HB):
        if eps is not None:
            for r in range(B * P):
                s14 = np.zeros(14)
                s14[5], s14[11] = sr[r, 7], sr[r, 8]
                U[r, j, 1] = eps[r]._steer(s14, U[r, j, 0] * env["t_max"]) \
                    / env["rud_max"]
        sr = data3.model0_step(env, sr, U[:, j])
        out[:, j] = sr
    return out.reshape(B, P, M.HB, 10), U.reshape(B, P, M.HB, 2)


def _eps_for(B, P, phi, psi_i):
    ep = RES["m"].ep
    out = []
    for b in range(B):
        for _ in range(P):
            e = copy.deepcopy(ep)
            e.heading_ref, e._psi_i = float(phi[b]), float(psi_i[b])
            out.append(e)
    return out


def test_2():
    dev, net, env, D = _setup()
    rng = np.random.default_rng(1)
    B, P = 3, 4
    i, k = _moments(D, B, 2)
    xs = D.XS[i, k].astype(float)
    pl = _plans(rng, B, P)
    _, st, act = ML.rollout_zero(torch.tensor(pl), xs, env)
    ref, _ = _ref_loop(env, xs, pl)
    e1 = np.abs(st[:, :, 0].numpy() - ref).max()
    phi = rng.uniform(-np.pi, np.pi, B)
    psi_i = rng.uniform(-0.3, 0.3, B)
    steer = ML.SteerT(RES["m"].ep, phi, psi_i, env)
    _, st2, act2 = ML.rollout_zero(torch.tensor(pl), xs, env, steer)
    ref2, U2 = _ref_loop(env, xs, pl, _eps_for(B, P, phi, psi_i))
    e2 = np.abs(st2[:, :, 0].numpy() - ref2).max()
    ea = np.abs(act2[:, :, 0].numpy() - np.clip(U2, [0, -1], [1, 1])).max()
    report("2 rollout_zero vs model0_step loop", e1 < 1e-9 and e2 < 1e-9
           and ea < 1e-9, f"without steer {e1:.1e}, with steer {e2:.1e} "
           f"(actuator column {ea:.1e})")


# ------------------------------------------------------------------ 3
def test_3():
    dev, net, env, D = _setup()
    rng = np.random.default_rng(3)
    B, P = 3, 4
    i, k = _moments(D, B, 4)
    xs = D.XS[i, k].astype(float)
    pl = _plans(rng, B, P)
    net0 = ML.load_net(RES["ck"], dev)
    net0.sample_grad = lambda h, y0, steps=24: torch.zeros_like(y0)
    phi = rng.uniform(-np.pi, np.pi, B)
    psi_i = rng.uniform(-0.3, 0.3, B)
    msg, ok = [], True
    for use in (False, True):
        s1 = ML.SteerT(RES["m"].ep, phi, psi_i, env) if use else None
        s2 = ML.SteerT(RES["m"].ep, phi, psi_i, env) if use else None
        _, a_st, a_act = ML.rollout_zero(torch.tensor(pl), xs, env, s1)
        _, b_st, b_act = M.rollout_core(net0, D, i, k, torch.tensor(pl), xs,
                                        env, 1, base=_base(B, P, 1),
                                        steer=s2)
        d = max((a_st - b_st.cpu()).abs().max().item(),
                (a_act - b_act.cpu()).abs().max().item())
        ok &= d < 1e-9
        msg.append(f"{'with' if use else 'without'} steer {d:.1e}")
    report("3 rollout_zero vs rollout_core(e = 0)", ok, ", ".join(msg))


# ------------------------------------------------------------------ 4
def test_4():
    dev, net, env, D = _setup()
    spec = importlib.util.spec_from_file_location("model3_bak", BACKUP)
    bak = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bak)
    rng = np.random.default_rng(5)
    B, P, S = 3, 5, 3
    i, k = _moments(D, B, 6)
    k[0] = 0
    xs = D.XS[i, k].astype(float)
    pl = torch.tensor(_plans(rng, B, P))
    base = torch.randn((B, P, S, M.HB, M.C7),
                       generator=torch.Generator().manual_seed(9))
    new = M.rollout_core(net, D, i, k, pl, xs, env, S, base=base)
    old = bak.rollout_core(net, D, i, k, pl, xs, env, S, base=base)
    same = all(torch.equal(a, b) for a, b in zip(new, old))
    d = max((a - b).abs().max().item() for a, b in zip(new, old))
    report("4 rollout_core(steer=None) vs backup", same,
           f"bit-identical {same} (max |diff| {d:.1e})")


# ------------------------------------------------------------------ 5
def test_5():
    dev, net, env, D = _setup()
    rng = np.random.default_rng(7)
    N = 12
    phi = rng.uniform(-np.pi, np.pi, N)
    psi_i = rng.uniform(-1.0, 1.0, N)
    eps = _eps_for(N, 1, phi, psi_i)
    steer = ML.SteerT(RES["m"].ep, phi, psi_i, env)
    worst, worst_i = 0.0, 0.0
    for j in range(24):
        sr = np.zeros((N, 1, 1, 10))
        sr[..., 7] = rng.uniform(-2 * np.pi, 2 * np.pi, (N, 1, 1))
        sr[..., 8] = rng.normal(0, 0.3, (N, 1, 1))
        U = np.zeros((N, 1, 1, 2))
        U[..., 0] = rng.uniform(0, 1, (N, 1, 1))
        U[:3, ..., 0] = rng.uniform(0, 0.1, (3, 1, 1))     # below the floor
        out = steer(j, torch.tensor(sr), torch.tensor(U)).numpy()
        for r in range(N):
            s14 = np.zeros(14)
            s14[5], s14[11] = sr[r, 0, 0, 7], sr[r, 0, 0, 8]
            ref = eps[r]._steer(s14, U[r, 0, 0, 0] * env["t_max"]) \
                / env["rud_max"]
            worst = max(worst, abs(out[r, 0, 0, 1] - ref))
            worst_i = max(worst_i, abs(float(steer.psi_i.view(-1)[r])
                                       - eps[r]._psi_i))
    report("5 SteerT vs Episode._steer", worst < 1e-12 and worst_i < 1e-12,
           f"nozzle {worst:.1e}, integral {worst_i:.1e}")


# ------------------------------------------------------------------ 6
def _group(jobs, T=2.5):
    from learn.repro.task import LEGS, Mission
    ms = [Mission("high", j["seed"], j["leg"], t_end=T, track=LEGS[j["leg"]])
          for j in jobs]
    live = ML.LiveData(len(jobs), ms[0].n_ctrl + 1, RES["ck"]["stats"],
                       RES["dev"], ML.mission_consts(ms[0]))
    for b, m in enumerate(ms):
        live.start(b, m.s)
    return ms, live


def test_6():
    dev, net, env, D = _setup()
    link = ML.ImpactLink(0.4, 1.0)
    jobs = [dict(seed=0, leg=0), dict(seed=1, leg=1)]
    ms, live = _group(jobs)
    ctrl = ML.LearnedMPPI(jobs, ms, env, link, net=net, live=live, K=8)
    noise, _ = ctrl.draws(0)
    bases = [ctrl.draws(b)[1] for b in range(2)]
    base = ctrl.base_tensor(bases)
    stride0 = base.stride(1) == 0
    pl = np.zeros((2, 8, M.HB, 2))
    pl[..., 0] = np.random.default_rng(0).uniform(0, 1, (2, 8, M.HB))
    pl[:, 5] = pl[:, 0]
    xs = np.stack([m.s for m in ms])
    phi = np.array([m.phi for m in ms])
    steer = ML.SteerT(ms[0].ep, phi, [m.ep._psi_i for m in ms], env)
    _, st, _ = M.rollout_core(net, live, [0, 1], [0, 0], torch.tensor(pl),
                              xs, env, 4, base=base, steer=steer)
    c, _ = ML.task_cost(st, pl[..., 0], xs, phi, link, env, ms[0].u_ref,
                        ms[0].L)
    dup = torch.equal(c[:, 0], c[:, 5])
    diff = not torch.equal(c[:, 0], c[:, 1])
    # a row's draws do not depend on its group
    ca = ML.LearnedMPPI(jobs, ms, env, link, net=net, live=live, K=8)
    cb = ML.LearnedMPPI(jobs[1:], ms[1:], env, link, net=net, live=live,
                        K=8)
    same_draws = True
    for _ in range(3):
        ca.draws(0)
        na, ba = ca.draws(1)
        nb_, bb = cb.draws(0)
        same_draws &= np.array_equal(na, nb_) and torch.equal(ba, bb)
    # m0 closed loop of the leg-1 row alone and in a group of two
    r2 = ML.run_group(jobs, env, link, T=2.5, log=lambda *a: None)
    r1 = ML.run_group(jobs[1:], env, link, T=2.5, log=lambda *a: None)
    same_m0 = all(np.array_equal(r2[1]["log"][q], r1[0]["log"][q])
                  for q in ("u", "zd", "thr")) and \
        r2[1]["score"] == r1[0]["score"]
    # the spread really scales the controller's base noise: base_tensor =
    # s x the row draws, for a scalar s and for one s per horizon step
    sp_ok = []
    for sp in (1.25, np.linspace(1.0, 3.0, M.HB)):
        cs = ML.LearnedMPPI(jobs, ms, env, link, net=net, live=live, K=8,
                            spread=sp)
        bb = [cs.draws(b)[1] for b in range(2)]
        got = cs.base_tensor(bb).cpu()
        ref = torch.stack(bb) * torch.as_tensor(
            np.broadcast_to(sp, (M.HB,)), dtype=torch.float32).view(
                1, 1, M.HB, 1)
        sp_ok.append(all(torch.equal(got[:, q], ref) for q in range(8)))
    report("6 CRN and group independence",
           dup and diff and stride0 and same_draws and same_m0
           and all(sp_ok),
           f"duplicate candidates equal {dup} (others differ {diff}), base "
           f"expanded (stride 0) {stride0}; draws group-independent "
           f"{same_draws}; m0 loop group-independent {same_m0}; base = s x "
           f"draws (scalar, per step) {sp_ok}")


# ------------------------------------------------------------------ 7
def test_7():
    dev, net, env, D = _setup()
    link = ML.ImpactLink(0.4, 1.0, A1=0.02)
    jobs = [dict(seed=2, leg=0), dict(seed=3, leg=1)]
    ms, live = _group(jobs)
    ctrl = ML.LearnedMPPI(jobs, ms, env, link, net=net, live=live,
                          spread=1.25)
    ctrl.capture = True
    # two steps, so the history is not empty
    for k in range(2):
        cmds = ctrl([0, 1], k)
        if k == 1:
            break
        for b, m in enumerate(ms):
            thr = float(cmds[b]) * m.t_max
            rud = m.ep._steer(m.s, thr)
            m.advance(thr, rud)
            live.push(b, (thr / m.t_max, rud / m.rud_max), m.s)
    cap = ctrl.last
    steer = ML.SteerT(ms[0].ep, cap["phi"], cap["psi_i"], env)
    _, st, _ = M.rollout_core(net, live, cap["rows"], [cap["k"]] * 2,
                              torch.tensor(cap["plans"]), cap["xs"], env,
                              ctrl.S, base=ctrl.base_tensor(
                                  list(cap["base_rows"])), steer=steer)
    c, _ = ML.task_cost(st, cap["plans"][..., 0], cap["xs"], cap["phi"],
                        link, env, ms[0].u_ref, ms[0].L)
    same_s = torch.equal(st.cpu(), cap["states"])
    same_c = torch.equal(c.cpu(), cap["cost"])
    report("7 controller call vs direct rollout_core", same_s and same_c,
           f"states bit-identical {same_s}, costs {same_c}; commands "
           f"{np.round(cap['cmds'], 3)}")


# ------------------------------------------------------------------ 8
def test_8():
    from learn.repro.task import K_Y
    dev, net, env, D = _setup()
    m = RES["m"]
    dtc = env["dt"] * env["sub"]
    Hh = M.HB
    st = torch.zeros(1, 1, 1, Hh, 10, dtype=torch.float64)
    u_ref, L = m.u_ref, m.L
    st[..., 2] = u_ref / 2                         # speed loss 0.5
    st[..., 1] = 3 * 5 * L                         # cross-track d = 3
    xs = np.zeros((1, 14))
    lk = ML.ImpactLink(0.4, 2.0)
    xs[0, 8] = 2 * 0.4 * dtc * ML.G                # step 0: abar = 2 A
    xs[0, 6] = u_ref / 2
    thr = np.linspace(0, 1, Hh)[None, None]
    c, parts = ML.task_cost(st, thr, xs, [0.0], lk, env, u_ref, L)
    exp = [0.5 * Hh * dtc, 2.0 * dtc, K_Y * 8.0 * Hh * dtc]
    sm = ML.W_JUMP * (Hh - 1) * (1 / (Hh - 1)) ** 2
    e_parts = max(abs(float(p) - e) for p, e in zip(parts, exp))
    e_tot = abs(float(c) - sum(exp) - sm)
    lk.j_imp = 0
    _, p0 = ML.task_cost(st, thr, xs, [0.0], lk, env, u_ref, L)
    cut_ok = float(p0[1]) == 0.0
    ls = ML.ImpactLink(0.4, 2.0, A1=0.05, pivot=20.0)
    a_ok = abs(ls.A(np.array(22 * ML.KN)) - 0.5) < 1e-12 and \
        abs(float(ls.A(torch.tensor(10 * ML.KN, dtype=torch.float64)))
                - ls.a_min) < 1e-12
    # the speed-dependent threshold reads the MEASURED speed only: rolled
    # speeds (which the plan controls) leave the impact part unchanged,
    # the measured one sets it exactly
    _, pa = ML.task_cost(st, thr, xs, [0.0], ls, env, u_ref, L)
    st_fast = st.clone()
    st_fast[..., 2] = 1.5 * u_ref
    _, pb = ML.task_cost(st_fast, thr, xs, [0.0], ls, env, u_ref, L)
    xs_f = xs.copy()
    xs_f[0, 6] = 1.2 * u_ref
    _, pc = ML.task_cost(st, thr, xs_f, [0.0], ls, env, u_ref, L)
    exp_c = [2.0 * max(0.0, 0.8 / float(ls.A(q)) - 1.0) ** 2 * dtc
             for q in (xs[0, 6], xs_f[0, 6])]
    meas_ok = float(pa[1]) == float(pb[1]) and \
        abs(float(pa[1]) - exp_c[0]) < 1e-12 and \
        abs(float(pc[1]) - exp_c[1]) < 1e-12 and exp_c[0] != exp_c[1]
    a_ok = a_ok and meas_ok
    # knots vs MPPIController
    ctrl = m.ep.ctrl
    E, Sh = ML.knot_matrices()
    kn = np.random.default_rng(0).uniform(0, 1, (16, 2, ML.N_KNOTS))
    ex = np.abs(ctrl._expand(kn) - np.einsum("kcn,hn->kch", kn, E)).max()
    sh = max(np.abs(ctrl._shift(kn[q]) - kn[q] @ Sh.T).max()
             for q in range(16))
    ok = e_parts < 1e-12 and e_tot < 1e-12 and cut_ok and a_ok and \
        ex < 1e-12 and sh < 1e-12
    report("8 task_cost and knots", ok,
           f"parts {e_parts:.1e}, total {e_tot:.1e}, j_imp cut {cut_ok}, "
           f"speed-dependent A {a_ok} (measured speed only {meas_ok}: "
           f"{float(pa[1]):.4f} = rolled fast {float(pb[1]):.4f}, measured "
           f"fast {float(pc[1]):.4f}); expand {ex:.1e}, shift {sh:.1e}")


# ------------------------------------------------------------------ 9
def _biased(eps, stats, shift=2.0, ch=(0, 3)):
    """Copies of pool episodes whose recorded errors are moved by `shift`
    e_sd on channels ch: a target the unadapted head cannot know and the
    head must learn (a pool where adaptation has to help)."""
    out = []
    for e in eps:
        e = dict(e, e0=e["e0"].copy())
        for c in ch:
            e["e0"][:, c] += shift * float(stats["e_sd"][c])
        out.append(e)
    return out


def test_9():
    dev, net, env, D = _setup()
    ck = RES["ck"]
    msg, ok = [], True
    # real Cb episodes (time split, whole-episode holdout), then biased
    # ones with a short episode among the training ones: there the head
    # MUST change and the validation FM loss MUST fall, and the training
    # windows stay W_CTX long despite the short episode
    cases = ((2, 50, None, False), (4, 20, [0, 1], False),
             (5, 60, [0, 1], True))
    for n_eps, n_steps, hold, biased in cases:
        eps = [_feed(D, i, 80, ck["stats"], dev).episode(0)
               for i in range(n_eps)]
        if biased:
            eps[-1] = _feed(D, n_eps, 12, ck["stats"], dev).episode(0)
            eps = _biased(eps, ck["stats"])
        pool = ML.Pool3(eps, ck["stats"], dev)
        t0 = time.time()
        net_a, info = ML.adapt_head(ck, pool, holdout=hold, steps=n_steps,
                                    check=10 if not biased else 20,
                                    log=lambda *a: None)
        ref = ML.load_net(ck, dev).state_dict()
        new = net_a.state_dict()
        other = all(torch.equal(new[k_], ref[k_]) for k_ in ref
                    if not k_.startswith("head."))
        head_changed = any(not torch.equal(new[k_], ref[k_]) for k_ in ref
                           if k_.startswith("head."))
        ok &= other and info["win"] == M.W_CTX
        if biased:
            ok &= head_changed and info["best_it"] > 0 and \
                info["val_fm"] < info["val_fm0"] and info["n_short"] >= 1
        msg.append(f"{n_eps} eps ({'holdout' if hold else 'time split'}"
                   f"{', biased + short' if biased else ''}): others bitwise"
                   f" equal {other}, head changed {head_changed} (kept step "
                   f"{info['best_it']}, val FM {info['val_fm0']:.4f} -> "
                   f"{info['val_fm']:.4f}), window {info['win']}, s "
                   f"{info['s']}, {time.time() - t0:.0f} s")
    report("9 adapt_head", ok, "; ".join(msg))


# ------------------------------------------------------------------ 10
def test_10():
    dev, net, env, D = _setup()
    from studies import mpc_compare as MC
    link = ML.ImpactLink(0.4, 1.0)
    jobs = [dict(seed=0, leg=0), dict(seed=0, leg=1)]
    out = []
    t0 = time.time()
    r = MC.job_hand(dict(jobs[0], T=2.5))
    out.append(("hand", r, time.time() - t0))
    t0 = time.time()
    r = MC.job_c0(dict(jobs[1], T=2.5, frozen=MC.c0_frozen()))
    out.append(("c0", r, time.time() - t0))
    for name, nt in (("m0", None), ("prior", net)):
        t0 = time.time()
        rows = ML.run_group(jobs, env, link, net=nt, stats=RES["ck"]["stats"],
                            T=2.5, log=lambda *a: None)
        out.append((name, rows[0], time.time() - t0))
    ok = all(r["finite"] and np.isfinite(r["score"]) for _, r, _ in out)
    report("10 short closed loops", ok, "; ".join(
        f"{n} score {r['score']:.3f} ({r.get('t_step_mean', 0):.2f} s/step, "
        f"{t:.0f} s)" for n, r, t in out))


# ------------------------------------------------------------------ 11
def test_11():
    root = os.path.dirname(HERE)
    files = ("learn/meta/mpc_learned.py", "learn/meta/model3.py",
             "learn/adapt/cmpc.py", "studies/mpc_compare.py",
             "studies/test_mpc_learned.py")
    bad = [f for f in files if b"\r\n" in open(os.path.join(root, f),
                                               "rb").read()]
    report("11 LF line endings", not bad, f"CRLF in {bad}" if bad else
           f"{len(files)} files LF")


# ------------------------------------------------------------------ 12
def _synth(link, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for leg in (0, 1):
        for rep in range(4):
            n = 480
            u = np.linspace(18, 29, n + 1) * ML.KN
            if rep % 2:
                u = u[::-1].copy()
            ab = rng.lognormal(np.log(0.15 + 0.03 * (u[:-1] / ML.KN - 18)
                                      * (1.3 if leg == 0 else 1.0)), 0.5)
            recs.append(dict(u=u, abar=ab, c_imp=link.cost(ab, u[:-1]),
                             leg=leg))
    return recs


def test_12():
    msg, ok = [], True
    true1 = ML.ImpactLink(0.5, 3.0)
    lk, info = ML.fit_impact_link(_synth(true1), log=lambda *a: None)
    ok1 = info["chosen"] == "single" and abs(lk.A0 - 0.5) < 0.03 and \
        abs(lk.K / 3.0 - 1) < 0.15
    msg.append(f"single: chosen {info['chosen']}, A0 {lk.A0:.3f}, K "
               f"{lk.K:.2f}")
    true2 = ML.ImpactLink(0.55, 3.0, A1=-0.03)
    lk2, info2 = ML.fit_impact_link(_synth(true2, 1), log=lambda *a: None)
    ok2 = info2["chosen"] == "speed" and lk2.A1 < -0.01
    msg.append(f"speed-dependent: chosen {info2['chosen']}, A0 "
               f"{lk2.A0:.3f}, A1 {lk2.A1:+.3f}, K {lk2.K:.2f} (single "
               f"growth ratio {info2['single']['growth']:.2f})")
    ok = ok1 and ok2
    report("12 fit_impact_link", ok, "; ".join(msg))


# ------------------------------------------------------------------ 13
def _hc_synth(N=2000, seed=0):
    """A synthetic horizon check: true exceedance drawn with a probability
    that rises with the block start speed, and rolled models given as
    exact expected exceedances. Returns (hc truth dict, p (N, H))."""
    rng = np.random.default_rng(seed)
    H = M.HB
    u0 = rng.uniform(16.0, 30.0, N) * ML.KN
    p = np.clip(0.01 + 0.02 * (u0 / ML.KN - 16.0), 0, 1)[:, None] \
        * np.ones(H)
    te = (rng.uniform(size=(N, H)) < p).astype(float)
    hc = dict(true_imp=te * rng.exponential(0.5, (N, H)),
              true_spd=np.zeros((N, H)), true_exc=te, u0=u0,
              ep=np.arange(N) // 8, k=np.zeros(N, int))
    return hc, p


def _hc_model(p, fac=1.0, j0=0, heavy=False, flat=False):
    """Rolled (imp, spd, exc) of a model: exceedance p x fac from step j0
    on (flat: the speed-mean p for every block, no growth with speed);
    heavy: the same exceedance with a few huge link costs (a heavy tail)."""
    exc = np.full_like(p, p.mean()) if flat else p.copy()
    exc[:, j0:] *= fac
    imp = exc * 0.5
    if heavy:
        imp = imp.copy()
        imp[::97] *= 400.0
    return imp, np.zeros_like(p), exc


def test_13():
    hc, p = _hc_synth()
    hc.update(same=_hc_model(p), late=_hc_model(p, 3.0, 10),
              early=_hc_model(p, 3.0, 0), heavy=_hc_model(p, heavy=True),
              flat=_hc_model(p, flat=True), m0=_hc_model(p, 0.2, 0))
    names = ["same", "late", "heavy", "m0"]
    j1, t1, e1 = ML.impact_horizon(hc, names, decide=["same", "late",
                                                      "heavy"])
    fb = {n: t1[n]["first_bad"] for n in names}
    ok1 = fb["same"] == 24 and fb["heavy"] == 24 and 8 <= fb["late"] <= 10 \
        and j1 == fb["late"] and not e1 and fb["m0"] == 0 and \
        max(t1["heavy"]["mean_ratio"]) > 1.5
    j2, t2, e2 = ML.impact_horizon(hc, ["same", "early"])
    ok2 = e2 == {"early": 0} and j2 == 4
    _, t3, e3 = ML.impact_horizon(hc, ["flat"])
    ok3 = t3["flat"]["first_bad"] == 0 and e3 == {"flat": 0}
    report("13 impact_horizon", ok1 and ok2 and ok3,
           f"first bad {fb}, j_imp {j1} (m0 reported, not deciding; heavy "
           f"tail max mean ratio {max(t1['heavy']['mean_ratio']):.1f} but "
           f"not bad); bad from step 0 -> early {e2}, j_imp {j2}; no growth"
           f" with speed -> first bad {t3['flat']['first_bad']}")


# ------------------------------------------------------------------ 14
def test_14():
    """phase_hcheck's cache and stop rule, phase_eval's stale-row guard,
    with stubbed nets and a synthetic horizon_check (no GPU work)."""
    import types
    from studies import mpc_compare as MC
    dev, net, env, D = _setup()
    hc0, p = _hc_synth()
    syn = dict(m0=_hc_model(p, 0.2, 0), prior=_hc_model(p),
               oracle=_hc_model(p), prior_s=_hc_model(p),
               online=_hc_model(p, 3.0, 10))
    calls = []

    def fake_check(models, D_, en, link, u_ref, max_eps=None, log=None):
        calls.append(sorted(models))
        return dict(hc0, **{n: syn[n] for n in models})

    avail = dict(prior=1.0, oracle=2.0)

    def fake_net_of(R, name):
        if name not in avail:
            raise KeyError(name)
        return object(), avail[name], None

    fake_ml = types.SimpleNamespace(
        H=M.HB, horizon_check=fake_check, impact_horizon=ML.impact_horizon,
        format_horizon=ML.format_horizon)
    g = dict(M=types.SimpleNamespace(Data3=lambda *a, **k: None),
             ML=fake_ml, env=env, dev=dev,
             ck=dict(cov=dict(stats=RES["ck"]["stats"])))
    saved = {k: getattr(MC, k) for k in ("gpu", "net_of", "save", "log")}
    opt = dict(MC.OPT)
    MC.gpu, MC.net_of = (lambda: g), fake_net_of
    MC.save, MC.log = (lambda R: None), (lambda m: None)
    msg, ok = [], True
    try:
        R = dict(link=dict(link=ML.ImpactLink(0.5, 3.0).to_dict()))
        MC.phase_hcheck(R, pre=True)
        ok &= calls == [["m0", "oracle", "prior"]] and \
            R["hcheck"]["j_imp"] == 24
        msg.append(f"pre: computed {calls[-1]}, j_imp "
                   f"{R['hcheck']['j_imp']}")
        avail.update(prior_s=1.25, online=1.0)
        MC.phase_hcheck(R)
        j_full = R["hcheck"]["j_imp"]
        ok &= calls[-1] == ["online", "prior_s"] and j_full < 24 and \
            R["hcheck"]["decided_on"] == sorted(MC.HC_DECIDE)
        msg.append(f"more models: computed only {calls[-1]}, j_imp "
                   f"re-decided {j_full}")
        n = len(calls)
        MC.phase_hcheck(R)
        ok &= len(calls) == n
        msg.append(f"rerun: cached {len(calls) == n}")
        R["eval"] = {"prior": {(0, 0): dict(j_imp=24)}}
        try:
            MC.phase_eval(R, "prior")
            stale = False
        except SystemExit:
            stale = True
        ok &= stale
        msg.append(f"row run with another j_imp refused {stale}")
        avail["online"] = 1.5
        MC.phase_hcheck(R)
        ok &= calls[-1] == ["online"]
        msg.append(f"online spread changed: recomputed {calls[-1]}")
        syn["online"] = _hc_model(p, 3.0, 0)
        avail["online"] = 2.0
        try:
            MC.phase_hcheck(R)
            stopped = False
        except SystemExit:
            stopped = True
        MC.OPT["hcheck_continue"] = True
        MC.phase_hcheck(R)
        ok &= stopped and R["hcheck"]["early"] == {"online": 0}
        msg.append(f"bad from step 0: stopped {stopped}, --hcheck-continue "
                   f"goes on with early {R['hcheck']['early']}")
    finally:
        for k, v in saved.items():
            setattr(MC, k, v)
        MC.OPT.clear()
        MC.OPT.update(opt)
    report("14 hcheck cache and stop", ok, "; ".join(msg))


TESTS = {i: globals()[f"test_{i}"] for i in range(1, 15)}


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
