#!/usr/bin/env python3
"""
Tests of M10 (DEFECTS M10: a random actuator family on the source world's
actuator channels, sim/lofi.py ACT_FAMILY / act_step), one process, smoke
caches only:

  1  old path bit-identical: ReducedPlant with old keys only (none, lag,
     rate, both) against the pre-M10 sim/lofi.py (scratchpad backup), and
     data2.episode with the old draw against the pre-M10 data2.py + lofi.py
     (sys.modules['sim.lofi'], sim.lofi and task._CTX swapped): every record
     equal bit for bit; an m10 episode keeps the old one's sea, scenario and
     operator seed
  2  new family, exact replay: float64 episodes (job raw64) with forced
     delay / dead zone / backlash / asymmetric lag / gain / second order,
     replayed by relabel.simulate with the recorded pushes from step 0 and
     from k = 1 (inside the padded history), 7, 60, 120 with u_hist / av0
     from relabel.hist_for: states and actuator positions <= 1e-9 relative
     (E reported only: simulate's e is not the recorded E by construction)
  3  branches: data3._branch_sim with the recorded commands as the plan
     (P = 1) and the recorded pushes: <= 1e-9 relative; with the operator's
     own noise continuation instead: reported
  4  sampling: 10000 draws of lofi.draw_act against ACT_FAMILY's
     probabilities (4 sigma) and ranges (bounds, KS on the uniform / log-
     uniform laws); the target-like set AS lofi.act_target_like RETURNS IT
     lies inside every range and equals the target plant's own Waterjet /
     Nozzle values, except the delay: the smallest whole number of source
     steps not shorter than the target's (3 x 0.04 s vs 0.1 s)
  5  old caches still replay: relabel.check_replay on meta2_smoke A and B
     (float32 records: tolerance relabel.REPLAY_TOL = 1e-5 relative), and
     branches_one reproduces the stored meta2_smoke A_branches bit for bit
  6  smoke pipeline on studies/_cache/meta3_smoke: meta_step2 copy / pilot
     (cached from the copy) / data / pack (train, A, B, At; --procs 1),
     refusals (meta2 + m10, meta3 + old), copies identical to meta2_smoke,
     seed-by-seed pairing with meta2_smoke; check_replay on A and At;
     meta_step3 e0, branches, tbranches, train (smoke), train_cov (20
     steps), eval ("" and _cov); stamps = simulation code + pack; cbhold
     (split Cbh, held-nozzle blocks), train_oracle (20 steps), eval
     _oracle; diag_yaw_m8_blocks (Cb, At, Cbh --holdout with the oracle)
     and diag_m9_infamily (A, At) on meta3_smoke; non-finite and heading-
     oscillation episodes against the actuator draw
  7  the review fixes, on synthetic inputs with known answers: block kinds
     (a nozzle step inside the window is not steady), the shared stall
     measure (known open fraction; a rollout equal to the truth -> gone, a
     frozen one -> there, in every bin), step-3 stamps follow the pack and
     the job list, step-2 legacy codes only on old-family splits, copy /
     pilot guards, targets stamped with the pack's code

    python -m studies.test_m10 [1 2 ...]
"""
import importlib.util
import json
import os
import pickle
import runpy
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SMOKE2 = os.path.join(HERE, "_cache", "meta2_smoke")
SMOKE3 = os.path.join(HERE, "_cache", "meta3_smoke")
BACKUP = os.path.join(
    os.environ.get("M10_BACKUP", r"C:\Users\Administrator\AppData\Local\Temp"
                   r"\claude\C--Users-Administrator-Documents"
                   r"\97e7fd1d-c515-4a09-869b-41596c911be0\scratchpad"),
    "backup_pre_m10")
TOL = 1e-9
RES = {}


def report(name, ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
    RES.setdefault("fails", [])
    if not ok:
        RES["fails"].append(name)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lib():
    d = np.load(os.path.join(SMOKE2, "lib.npz"))
    return dict(mu=d["mu"], sd=d["sd"], S=d["S"])


def _rel(a, b, floor=1e-6):
    """max over columns of max_t |a - b| / max(max_t |b|, floor)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    sc = np.maximum(np.abs(b).max(0), floor)
    return float((np.abs(a - b).max(0) / sc).max())


class _OldLofi:
    """sim.lofi swapped for the pre-M10 copy while the block runs."""

    def __enter__(self):
        import sim
        from learn.repro import task
        self.bk = RES.get("lofi_bk") or _load("lofi_bk", os.path.join(
            BACKUP, "lofi.py"))
        RES["lofi_bk"] = self.bk
        self.saved = (sys.modules["sim.lofi"], sim.lofi, dict(task._CTX))
        sys.modules["sim.lofi"] = self.bk
        sim.lofi = self.bk
        task._CTX.clear()
        return self.bk

    def __exit__(self, *a):
        import sim
        from learn.repro import task
        sys.modules["sim.lofi"], sim.lofi = self.saved[0], self.saved[1]
        task._CTX.clear()
        task._CTX.update(self.saved[2])


# ------------------------------------------------------------------ 1
def test_1():
    from learn.meta import data2
    from learn.repro.task import ctx
    from sim import lofi
    from sim.wavefield import SeaState
    c = ctx()
    bk = RES.get("lofi_bk") or _load("lofi_bk", os.path.join(BACKUP,
                                                             "lofi.py"))
    RES["lofi_bk"] = bk
    p = c["p"]
    rs = np.sqrt(p["L"] / 10.0)
    sd = data2.sea_for(np.random.default_rng(5))
    sea = SeaState(sd.pop("hs"), sd.pop("tp"), theta0=sd.pop("theta0"),
                   n_freq=sd.pop("n_freq"), n_dir=sd.pop("n_dir"), seed=5,
                   **sd)
    rng = np.random.default_rng(0)
    cmds = np.stack([rng.uniform(0, p["t_max"], 300),
                     rng.uniform(-1, 1, 300) * p["rud_max"]], 1)
    ok_p, n_cmp = True, 0
    for prm in ({}, dict(tau_thrust=0.8 * rs), dict(rud_rate=0.6 / rs),
                dict(tau_thrust=1.3 * rs, rud_rate=0.4 / rs)):
        out = []
        for mod in (lofi, bk):
            pl = mod.plant_for(c["db"], sea, c["h"], dt=0.04, params=prm)
            s = pl.initial_state(0.8 * p["u_design"])
            traj = []
            for i in range(300 * 6):
                s = pl.step(s, i * 0.04, *cmds[i // 6], 0.04)
                traj.append(np.concatenate([s, [pl.last_thr_app,
                                                pl.last_cg_acc,
                                                pl.last_bow_acc,
                                                pl.slam_count]]))
            out.append(np.array(traj))
        ok_p &= np.array_equal(out[0], out[1])
        n_cmp += out[0].size
    # data2.episode: the new code with the old draw vs the pre-M10 code
    d2bk = _load("data2_bk", os.path.join(BACKUP, "data2.py"))
    lib = _lib()
    # seeds 1-3 draw an actuator; add the first seed that draws none
    s_none = 5
    while True:
        r_ = np.random.default_rng(s_none)
        data2.sea_for(r_)
        if r_.random() >= 0.75:
            break
        s_none += 1
    jobs = [dict(seed=s, world="low", lib=lib, T=30.0)
            for s in (1, 2, 3, s_none)]
    jobs.append(dict(seed=300001, world="high", T=8.0))
    ok_e, n_act, info = True, 0, []
    for job in jobs:
        new = data2.episode(dict(job))
        with _OldLofi():
            old = d2bk.episode(dict(job))
        same = all(np.array_equal(new[k], old[k]) for k in
                   ("S", "XS", "U", "E", "RR", "RN", "ACG", "D"))
        same &= new["act"] == old["act"] and new["informative"] == \
            old["informative"] and np.array_equal(new["ck"], old["ck"])
        ok_e &= same
        n_act += old["act"] is not None
        info.append(f"{job['seed']}:{'=' if same else 'DIFF'}"
                    f"{'' if old['act'] is None else '(act)'}")
    # pairing: the m10 episode of a seed has the old one's sea, scenario
    # switch and operator seed, only other actuators
    pair = True
    for s in (1, 2, 3):
        a = data2.episode(dict(seed=s, world="low", lib=lib, T=5.0))
        b = data2.episode(dict(seed=s, world="low", lib=lib, T=5.0,
                               act_family="m10"))
        pair &= (a["sea"] == b["sea"] and a["informative"]
                 == b["informative"] and a["op_seed"] == b["op_seed"]
                 and b["act"]["act_family"] == 1.0)
    report("1 old path bit-identical", ok_p and ok_e and pair,
           f"ReducedPlant old keys (none / lag / rate / both), 300 control "
           f"steps x 6: identical to backup lofi {ok_p} ({n_cmp} numbers); "
           f"data2.episode old draw vs pre-M10 data2 + lofi (4 source "
           f"episodes, {n_act} with a drawn actuator, 1 target episode): "
           f"{' '.join(info)}; m10 keeps sea / scenario / operator seed "
           f"{pair}")


# ------------------------------------------------------------------ 2
def _act_sets(p):
    from sim import lofi
    tm, rm = p["t_max"], p["rud_max"]
    base = {k: lofi.EXTRA[k] for k in lofi.ACT_KEYS}
    a = dict(base, act_family=1.0,
             thr_delay=3.0, thr_dz=0.02 * tm, thr_gain=1.05,
             thr_off=0.01 * tm, thr_db=0.01 * tm, thr_tau=0.4,
             thr_ratio=0.6, thr_rate=tm / 0.6,
             noz_delay=6.0, noz_dz=0.01 * rm, noz_gain=0.95,
             noz_off=-0.01 * rm, noz_db=0.015 * rm, noz_tau=0.06,
             noz_ratio=1.8, noz_rate=0.5)
    b = dict(base, act_family=1.0,
             thr_delay=1.0, thr_tau=0.3, thr_zeta=0.4, thr_rate=tm / 0.5,
             noz_delay=4.0, noz_tau=0.08, noz_zeta=0.35, noz_ratio=1.5,
             noz_rate=0.6, noz_db=0.01 * rm)
    from learn.meta import data2
    c_ = lofi.draw_act(np.random.default_rng(12), p, data2.low_dt())
    t_ = lofi.act_target_like(p, data2.low_dt())
    return [("first order, all parts", a), ("second order + delay", b),
            ("a draw", c_), ("target-like", t_)]


def _episodes():
    """Float64 new-family episodes, one per forced parameter set."""
    if "eps64" not in RES:
        from learn.meta import data2
        from learn.repro.task import ctx
        p = ctx()["p"]
        lib = _lib()
        eps = []
        for q, (nm, act) in enumerate(_act_sets(p)):
            e = data2.episode(dict(seed=11 + q, world="low", lib=lib, T=40.0,
                                   act_params=act, raw64=True))
            e["name"] = nm
            eps.append(e)
        RES["eps64"] = eps
    return RES["eps64"]


def _W_of(eps, lib):
    """relabel's worker dict for float64 episodes (equal lengths)."""
    n = min(len(e["U"]) for e in eps)
    W = dict(S=np.stack([e["S"][:n] for e in eps]),
             XS=np.stack([e["XS"][:n] for e in eps]),
             U=np.stack([e["U"][:n] for e in eps]),
             AV=np.stack([e["AV"][:n] for e in eps]),
             len=np.full(len(eps), n), lib=lib,
             meta=[dict(seed=e["seed"], op_seed=e["op_seed"],
                        relay=e["relay"], sea=e["sea"], act=e["act"],
                        world=e["world"], style=e["style"]) for e in eps])
    return W


def test_2():
    from learn.meta import relabel as R
    eps = _episodes()
    env = R._env()
    dtc = env["dt"] * env["sub"]
    Hh = 20
    W = _W_of(eps, _lib())
    worst, worst_e, lines, ok = 0.0, 0.0, [], True
    for i, e in enumerate(eps):
        m = W["meta"][i]
        seas = R.Seas([R._sea(m)], env["x_st"], env["y_off"])
        errs = []
        for k in (0, 1, 7, 60, 120):
            uh, av = R.hist_for(env, W, [(i, k)])
            E, XS = R.simulate(env, W["XS"][i, k][None],
                               W["U"][i, k:k + Hh][None],
                               np.array([k * dtc]), seas,
                               R.act_rows([R._act(m)]),
                               pushes=e["R"][k:k + Hh][None], u_hist=uh,
                               av0=av)
            err = _rel(XS[0], W["XS"][i, k:k + Hh + 1])
            errs.append(err)
            worst = max(worst, err)
            worst_e = max(worst_e, _rel(E[0], e["E"][k:k + Hh], 1e-3))
        # the actuators really acted: position trails command, servo moved
        gap = np.abs(W["XS"][i, 1:, 12:14] / [env["t_max"], env["rud_max"]]
                     - W["U"][i, :-1]).max(0)
        lines.append(f"{e['name']}: max rel {max(errs):.1e} (gap thr "
                     f"{gap[0]:.2f} noz {gap[1]:.2f}, |AV| max "
                     f"{np.abs(e['AV']).max():.2g})")
        ok &= max(errs) <= TOL and e["finite"]
    report("2 new family exact replay", ok,
           f"{len(eps)} float64 episodes x starts 0, 1, 7, 60, 120 x 20 "
           f"steps, recorded pushes: states + actuator positions max "
           f"relative error {worst:.2e} (<= {TOL:g}); " + "; ".join(lines)
           + f". e vs recorded E (not exact by construction): max relative "
           f"{worst_e:.2f}")


# ------------------------------------------------------------------ 3
def test_3():
    from learn.meta import data3
    from learn.meta import relabel as R
    eps = _episodes()
    W = _W_of(eps, _lib())
    saved = dict(R._W)
    R._W.clear()
    R._W.update(W, env=saved.get("env") or R._env())
    Hh = 20
    det, full, ok = 0.0, [], True
    try:
        for i, e in enumerate(eps):
            for k in (60, 120):
                plans = W["U"][i, k:k + Hh][None]
                rec = W["XS"][i, k:k + Hh + 1]
                XS, E0 = data3._branch_sim(i, k, plans,
                                           pushes=e["R"][k:k + Hh][None])
                det = max(det, _rel(XS[0], rec))
                XSn, _ = data3._branch_sim(i, k, plans)
                full.append(_rel(XSn[0], rec))
                e0r = data3.e0_from(R._env(), rec[:-1], rec[1:], plans[0])
                ok &= np.abs(E0[0] - e0r).max() <= 1e-6 * max(
                    np.abs(e0r).max(), 1.0)
    finally:
        R._W.clear()
        R._W.update(saved)
    ok &= det <= TOL
    report("3 branches", ok,
           f"_branch_sim, recorded commands as the plan (P = 1), recorded "
           f"pushes: max relative state error {det:.2e} (<= {TOL:g}), E0 "
           f"consistent; with the replayed operator's own noise "
           f"continuation instead of the recorded pushes: max relative "
           f"{max(full):.2e} (median {np.median(full):.1e}; the operator "
           f"reads inputs recomputed from the simulated state and "
           f"elevations summed in another order, so this part is not "
           f"required to be exact)")


# ------------------------------------------------------------------ 4
def _ks_uniform(u):
    u = np.sort(np.asarray(u))
    n = len(u)
    if n == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float(max((i / n - u).max(), (u - (i - 1) / n).max()))


def test_4():
    from learn.meta import data2
    from learn.repro.task import ctx
    from sim import lofi
    p = ctx()["p"]
    dt = data2.low_dt()
    rs = np.sqrt(p["L"] / 10.0)
    N = 10000
    rng = np.random.default_rng(2026)
    D = [lofi.draw_act(rng, p, dt) for _ in range(N)]
    F = lofi.ACT_FAMILY
    ok, notes = True, []

    def prob(name, got, n, want):
        nonlocal ok
        sig = np.sqrt(want * (1 - want) / max(n, 1))
        good = abs(got - want) <= 4 * sig + 1e-12
        ok &= bool(good)
        notes.append(f"{name} {got:.3f}/{want:.3f}")

    for c, scale in (("thr", p["t_max"]), ("noz", p["rud_max"])):
        a = {k: np.array([d[f"{c}_{k}"] for d in D]) for k in lofi.ACT_NEUTRAL}
        ideal = np.ones(N, bool)
        for k, v in lofi.ACT_NEUTRAL.items():
            ideal &= a[k] == v
        # an all-off non-ideal draw also looks ideal: P = 0.8 prod(1 - p_i)
        # (delay 'on' but rounded to 0 counts as off)
        a_max = F["delay"][3] * rs / dt
        p_d = F["delay"][0] * (1 - 0.5 / a_max)
        p_off = (1 - p_d) * (1 - F[f"tau_{c}"][0]) * (1 - F[f"rate_{c}"][0]) \
            * (1 - F["dz"][0]) * (1 - F["db"][0]) * (1 - F["gain"][0])
        prob(f"{c} ideal", ideal.mean(), N, lofi.ACT_P_IDEAL
             + (1 - lofi.ACT_P_IDEAL) * p_off)
        for nm, key, pw in (("delay>0", "delay", p_d),
                            ("tau", "tau", F[f"tau_{c}"][0]),
                            ("rate", "rate", F[f"rate_{c}"][0]),
                            ("dz", "dz", F["dz"][0]), ("db", "db", F["db"][0]),
                            ("gain", "gain", F["gain"][0])):
            ne = a[key] != lofi.ACT_NEUTRAL[key]
            prob(f"{c} {nm}", ne.mean(), N, (1 - lofi.ACT_P_IDEAL) * pw)
        lag = a["tau"] > 0
        prob(f"{c} ratio|tau", (a["ratio"][lag] != 1).mean(), lag.sum(),
             F["ratio"][0])
        prob(f"{c} zeta|tau", (a["zeta"][lag] > 0).mean(), lag.sum(),
             F["zeta"][0])
        # ranges and laws (u = the value mapped onto [0, 1] by its law)
        rng_ok = True
        ks = {}
        for nm, v, law, lo, hi in (
                ("tau", a["tau"][lag] / rs, "logU", F[f"tau_{c}"][2],
                 F[f"tau_{c}"][3]),
                ("ratio", a["ratio"][a["ratio"] != 1], "logU",
                 F["ratio"][2], F["ratio"][3]),
                ("zeta", a["zeta"][a["zeta"] > 0], "logU", F["zeta"][2],
                 F["zeta"][3]),
                ("rate", (scale / (a["rate"][a["rate"] > 0] * rs)
                          if c == "thr" else a["rate"][a["rate"] > 0] * rs),
                 "logU", F[f"rate_{c}"][2], F[f"rate_{c}"][3]),
                ("dz", a["dz"][a["dz"] > 0] / scale, "U", F["dz"][2],
                 F["dz"][3]),
                ("db", a["db"][a["db"] > 0] / scale, "U", F["db"][2],
                 F["db"][3]),
                ("gain", a["gain"][a["gain"] != 1], "logU", F["gain"][2],
                 F["gain"][3]),
                ("off", a["off"][a["gain"] != 1] / scale, "U", F["off"][2],
                 F["off"][3])):
            rng_ok &= bool(len(v) and v.min() >= lo * (1 - 1e-12)
                           and v.max() <= hi * (1 + 1e-12))
            u = (np.log(v / lo) / np.log(hi / lo) if law == "logU"
                 else (v - lo) / (hi - lo))
            ks[nm] = _ks_uniform(u)
            rng_ok &= ks[nm] <= 1.63 / np.sqrt(len(v))       # 1% level
        dl = a["delay"]
        rng_ok &= bool(np.all(dl == np.round(dl)) and dl.min() >= 0
                       and dl.max() <= round(a_max))
        ok &= rng_ok
        notes.append(f"{c} ranges + laws {rng_ok} (KS max "
                     f"{max(ks.values()):.3f}, delay steps 0..{int(dl.max())})")
    # the target-like set AS IMPLEMENTED (lofi.act_target_like), each value's
    # position u in its range
    t = lofi.act_target_like(p, dt)
    tgt = [("noz delay", t["noz_delay"] * dt / rs, "U", F["delay"]),
           ("noz tau", t["noz_tau"] / rs, "logU", F["tau_noz"]),
           ("noz rate", t["noz_rate"] * rs, "logU", F["rate_noz"]),
           ("thr delay", t["thr_delay"] * dt / rs, "U", F["delay"]),
           ("thr tau", t["thr_tau"] / rs, "logU", F["tau_thr"]),
           ("thr full range", p["t_max"] / t["thr_rate"] / rs, "logU",
            F["rate_thr"]),
           ("thr dead zone", t["thr_dz"] / p["t_max"], "U", F["dz"])]
    ins, pos = True, []
    for nm, v, law, (_, _, lo, hi, _) in tgt:
        u = (np.log(v / lo) / np.log(hi / lo) if law == "logU"
             else (v - lo) / (hi - lo))
        ins &= 0.0 < u < 1.0
        pos.append(f"{nm} u={u:.2f}")
    # ... and against the target plant's own Waterjet / Nozzle (delay in
    # steps x the plant's dt): equal, except the delay, which is the
    # smallest whole number of source steps not shorter than the target's
    tp_ = _target_actuators()
    d_t = tp_["noz_delay_s"]
    same = dict(
        noz_tau=abs(t["noz_tau"] - tp_["noz_tau"]) < 1e-12,
        noz_rate=abs(t["noz_rate"] - tp_["noz_rate"]) < 1e-12,
        thr_tau=abs(t["thr_tau"] - tp_["thr_tau"]) < 1e-12,
        thr_full=abs(p["t_max"] / t["thr_rate"] - tp_["thr_full"]) < 1e-12,
        thr_dz=abs(t["thr_dz"] / p["t_max"] - tp_["thr_dz"]) < 1e-12,
        delay=(t["noz_delay"] == t["thr_delay"]
               and t["noz_delay"] * dt >= d_t - 1e-12
               and (t["noz_delay"] - 1) * dt < d_t - 1e-12
               and abs(tp_["thr_delay_s"] - d_t) < 1e-12))
    tsame = all(same.values())
    report("4 sampling", ok and ins and tsame,
           f"{N} draws: " + ", ".join(notes) + f"; target-like set (from "
           f"lofi.act_target_like) inside every range {ins}: "
           + ", ".join(pos) + f"; vs the target plant's actuators "
           f"{ {k: bool(v) for k, v in same.items()} } (delay "
           f"{t['noz_delay']:.0f} x {dt:g} s = {t['noz_delay'] * dt:.2f} s "
           f"vs the target's {d_t:.2f} s)")


def _target_actuators():
    """The target plant's (sim/planing_vessel.py) Waterjet / Nozzle values,
    read from a built plant: delays as steps x its own dt."""
    from learn.repro.task import Mission
    from learn.meta.data2 import sea_for
    m = Mission("high", 1, 0, t_end=1.0, track=0.0,
                sea=sea_for(np.random.default_rng(0), target=True))
    jet, noz = m.plant.prop, m.plant.rudder
    dtp = m.dt
    return dict(noz_delay_s=noz.delay.n * dtp, thr_delay_s=jet.delay.n * dtp,
                noz_tau=noz.tau, noz_rate=noz.rate, thr_tau=jet.tau,
                thr_full=jet.t_max / jet.rate_max,
                thr_dz=jet.dead_band / jet.t_max)


# ------------------------------------------------------------------ 5
def test_5():
    from learn.meta import data3
    from learn.meta import relabel as R
    lib = _lib()
    out = []
    ok = True
    for sp in ("A", "B"):
        r = R.check_replay(os.path.join(SMOKE2, f"{sp}.npz"), lib)
        ok &= r["ok"]
        out.append(f"{sp}: max rel XS {r['max_xs']:.1e} over "
                   f"{len(r['xs'])} episodes, e {max(r['e']):.1e}")
    R._init(os.path.join(SMOKE2, "A.npz"), lib)
    R._W.pop("seas", None)
    b = np.load(os.path.join(SMOKE2, "A_branches.npz"))
    ok_b = True
    for r_ in (0, 3):
        _, _, U, XS, E0 = data3.branches_one((int(b["ep"][r_]),
                                              int(b["k"][r_]), 8, 0))
        ok_b &= (np.array_equal(U, b["U"][r_]) and np.array_equal(
            XS, b["XS"][r_]) and np.array_equal(E0, b["E0"][r_]))
    report("5 old caches replay", ok and ok_b,
           f"check_replay meta2_smoke (tolerance {R.REPLAY_TOL:g}): "
           + "; ".join(out) + f"; branches_one = stored A_branches bit for "
           f"bit {ok_b}")


# ------------------------------------------------------------------ 6
def _argv_run(mod, argv):
    saved = sys.argv
    sys.argv = [mod.__name__] + argv
    try:
        mod.main()
    finally:
        sys.argv = saved


def _same_file(a, b):
    return open(a, "rb").read() == open(b, "rb").read()


def test_6():
    from learn.meta import relabel as R
    from studies import meta_step2 as S2
    from studies import meta_step3 as S3
    t0 = time.time()
    notes, ok = [], True
    # refusals first: they must write nothing
    ref = []
    for argv in (["--cache-name", "meta2", "--act-family", "m10",
                  "--phase", "data"],
                 ["--cache-name", "meta3", "--smoke", "--phase", "data"]):
        try:
            _argv_run(S2, argv)
            ref.append(False)
        except SystemExit:
            ref.append(True)
    ok &= all(ref)
    notes.append(f"refusals (meta2 + m10, meta3 + old) {ref}")
    _argv_run(S2, ["--cache-name", "meta3", "--smoke", "--act-family", "m10",
                   "--copy-from", "meta2", "--phase", "copy,pilot,data,pack",
                   "--splits", "train,A,B,At", "--procs", "1"])
    t2 = time.time() - t0
    same = all(_same_file(os.path.join(SMOKE2, f), os.path.join(SMOKE3, f))
               for f in S2.COPY_FILES)
    ok &= same
    m2 = pickle.load(open(os.path.join(SMOKE2, "A_meta.pkl"), "rb"))
    m3 = pickle.load(open(os.path.join(SMOKE3, "A_meta.pkl"), "rb"))
    mt = pickle.load(open(os.path.join(SMOKE3, "At_meta.pkl"), "rb"))
    pair = all(a["seed"] == b["seed"] and a["sea"] == b["sea"]
               and a["informative"] == b["informative"]
               and a["op_seed"] == b["op_seed"] for a, b in zip(m2, m3))
    fam = (all(m["act_family"] == "m10" and m["act"]["act_family"] == 1.0
               for m in m3) and all(m["act_family"] == "at" for m in mt))
    ok &= pair and fam and len(m2) == len(m3)
    notes.append(f"step 2 {t2:.0f} s; copies identical to meta2_smoke "
                 f"{same}; A pairs seed by seed with meta2_smoke A (sea, "
                 f"scenario, operator seed) {pair}; families m10 / at {fam}")
    # replay of the packed new-family splits (float32 tolerance)
    lib = dict(np.load(os.path.join(SMOKE3, "lib.npz")))
    for sp in ("A", "At", "train"):
        r = R.check_replay(os.path.join(SMOKE3, f"{sp}.npz"), lib)
        ok &= r["ok"]
        notes.append(f"check_replay {sp} {r['max_xs']:.1e}")
    # step 3
    t1 = time.time()
    base = ["--cache-name", "meta3", "--smoke", "--procs", "1"]
    _argv_run(S3, base + ["--phase", "e0,branches,tbranches,train"])
    _argv_run(S3, base + ["--phase", "train_cov", "--steps", "20"])
    _argv_run(S3, base + ["--phase", "eval"])
    _argv_run(S3, base + ["--phase", "eval", "--tag", "_cov"])
    want = ["train_e0.npz", "A_e0.npz", "B_e0.npz", "At_e0.npz",
            "A_branches.npz", "B_branches.npz", "At_branches.npz",
            "train_tbranches.npz", "train_vbranches.npz", "A_wbranches.npz",
            "At_wbranches.npz", "model3.pt", "model3_cov.pt", "eval3.pkl",
            "eval3_cov.pkl"]
    miss = [f for f in want if not os.path.exists(os.path.join(SMOKE3, f))]
    # stamps: simulation code + the pack (+ n_eps / the job list)
    rd = lambda f: open(os.path.join(SMOKE3, f + ".code")).read()  # noqa
    stamps = (rd("A_e0.npz") == S3._stamp_text("A")
              and rd("At_branches.npz") == S3._stamp_text("At", 8)
              and rd("At_wbranches.npz").startswith(S3.sim_code() + "|")
              and "learn/meta/operators.py" in S3._sim_sources())
    ev = pickle.load(open(os.path.join(SMOKE3, "eval3.pkl"), "rb"))
    ok &= not miss and stamps and ("At", "multi") in ev and ("Cb", "multi") \
        in ev
    notes.append(f"step 3 {time.time() - t1:.0f} s; missing {miss}; stamps "
                 f"{stamps}; eval keys {sorted(set(k[0] for k in ev))}")
    # the steady-block data and the oracle-data test (D7)
    t3 = time.time()
    _argv_run(S3, base + ["--phase", "cbhold"])
    _argv_run(S3, base + ["--phase", "train_oracle", "--steps", "20"])
    _argv_run(S3, base + ["--phase", "eval", "--tag", "_oracle"])
    import torch
    ko = torch.load(os.path.join(SMOKE3, "model3_oracle.pt"),
                    weights_only=False)
    mh = pickle.load(open(os.path.join(SMOKE3, "Cbh_meta.pkl"), "rb"))
    ev_o = pickle.load(open(os.path.join(SMOKE3, "eval3_oracle.pkl"), "rb"))
    d = np.load(os.path.join(SMOKE3, "Cbh.npz"))
    # held blocks: the nozzle column constant over the whole block
    held = [float(np.ptp(d["U"][i, k:k + 24, 1]) == 0.0)
            for i in range(len(d["len"]))
            for k in range(0, int(d["len"][i]) - 24 + 1, 24)]
    ok_h = (ko["holdout"] == S3.oracle_holdout(len(mh)).tolist()
            and all(m["p_hold"] == S3.P_HOLD for m in mh)
            and ("Cbh", "multi") in ev_o and np.mean(held) > 0.4)
    ok &= ok_h
    notes.append(f"cbhold + train_oracle + eval _oracle {ok_h} ({len(mh)} "
                 f"Cbh episodes, held-nozzle blocks {np.mean(held):.2f}, "
                 f"{time.time() - t3:.0f} s)")
    # the diagnostics on meta3_smoke
    for script, argv in (("diag_yaw_m8_blocks.py",
                          ["--cache-name", "meta3_smoke"]),
                         ("diag_yaw_m8_blocks.py",
                          ["--cache-name", "meta3_smoke", "--split", "At"]),
                         ("diag_yaw_m8_blocks.py",
                          ["--cache-name", "meta3_smoke", "--split", "Cbh",
                           "--holdout", "--model", "model3_oracle.pt"]),
                         ("diag_m9_infamily.py",
                          ["--cache-name", "meta3_smoke"]),
                         ("diag_m9_infamily.py",
                          ["--cache-name", "meta3_smoke", "--split", "At"])):
        saved, cwd = sys.argv, os.getcwd()
        sys.argv = [script] + argv
        try:
            runpy.run_path(os.path.join(HERE, script), run_name="__main__")
            notes.append(f"{script} {' '.join(argv[2:])} ran")
        except Exception as ex:
            ok = False
            notes.append(f"{script} FAILED {ex!r}")
        finally:
            sys.argv = saved
            os.chdir(cwd)
    # non-finite / heading-oscillation episodes against the actuator draw
    notes.append(_stability())
    report("6 smoke pipeline", ok, "; ".join(notes)
           + f" ({time.time() - t0:.0f} s)")


def _stability():
    """Per source episode of meta3_smoke: finite?, full length?, share of
    steps with the nozzle command saturated (|U| > 0.98) and yaw-rate rms,
    against the nozzle's slowness (delay + tau + full travel / rate)."""
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(SMOKE3, "raw", "*.pkl"))):
        ch = pickle.load(open(f, "rb"))
        for e in ch["eps"]:
            a = e["act"]
            if not a or a.get("act_family", 0) < 0.5:
                continue
            slow = (a["noz_delay"] * 0.04 + a["noz_tau"] * max(1.0, a[
                "noz_ratio"]) + (0.87 / a["noz_rate"] if a["noz_rate"] > 0
                                  else 0.0))
            sat = float((np.abs(e["U"][:, 1]) > 0.98).mean())
            rows.append((slow, e["finite"], len(e["U"]), sat,
                         float(np.sqrt((e["XS"][:, 11] ** 2).mean())),
                         a["noz_zeta"] > 0))
    if not rows:
        return "no episodes"
    R_ = np.array(rows, float)
    n = len(R_)
    nf = int((R_[:, 1] < 0.5).sum())
    short = int((R_[:, 2] < R_[:, 2].max()).sum())
    slow = R_[:, 0] >= np.quantile(R_[:, 0], 0.8)
    return (f"source episodes {n}: non-finite {nf}, shorter than full "
            f"{short}; nozzle saturated share slowest 20% "
            f"{R_[slow, 3].mean():.3f} vs rest {R_[~slow, 3].mean():.3f}, "
            f"yaw-rate rms {R_[slow, 4].mean():.3f} vs "
            f"{R_[~slow, 4].mean():.3f} rad/s; max slowness "
            f"{R_[:, 0].max():.2f} s (saturated share there "
            f"{R_[np.argmax(R_[:, 0]), 3]:.2f}); second-order nozzles "
            f"{int(R_[:, 5].sum())}")


# ------------------------------------------------------------------ 7
def _first_order(U, a0, tau, dtc, u_prev=None):
    """Actual nozzle a (n, H + 1) of an exact first-order lag on plans U
    (n, H, 2) from a0, and its gap channel (a[j + 1] - U[j]) / dtc."""
    n, H = U.shape[:2]
    a = np.zeros((n, H + 1))
    a[:, 0] = a0
    f = np.exp(-dtc / tau)
    for j in range(H):
        a[:, j + 1] = U[:, j, 1] + (a[:, j] - U[:, j, 1]) * f
    return a, (a[:, 1:] - U[:, :, 1]) / dtc


def test_7():
    """Units of the M10 review fixes: the diagnostics' block kinds and
    stall measure on synthetic plans with known answers, meta_step3's
    stamps, meta_step2's legacy codes, copy / pilot guards and the targets'
    code."""
    import tempfile
    import types

    from studies import diag_stall as DS
    from studies import meta_step2 as S2
    from studies import meta_step3 as S3
    notes, ok = [], True
    dtc, H = 0.24, 24
    # -- block kinds: a step inside the window is NOT steady (the start-only
    # rule called it steady), a held command with the actual on it is
    U = np.zeros((4, H, 2))
    U[:, :, 1] = 0.3
    U[1, 5:, 1] = 0.6                            # step at j = 5
    U[2, :, 1] = 0.3 + 0.05 * np.sin(np.arange(H))   # wandering command
    a = np.full((4, H + 1), 0.3)
    a[1, 6:] = 0.6
    a[3, :4] = [0.1, 0.2, 0.25, 0.29]            # catching up at the start
    k = DS.block_kinds(U, a)
    want = [True, False, False, False]
    ok_k = k["steady"].tolist() == want and bool((k["g0"] < 0.05)[:3].all())
    ok &= ok_k
    notes.append(f"block kinds steady {k['steady'].tolist()} (want {want}; "
                 f"the start-only rule would say "
                 f"{(k['g0'] < 0.05).tolist()}) {ok_k}")
    # -- stall measure: a truth that closes as a first-order lag; a rollout
    # equal to it -> gone; a rollout that never closes -> THERE
    U = np.zeros((3, H, 2))
    U[:, :, 1] = -0.2
    U[0, 4:, 1] = 0.2                                   # 0.4 at j0 = 4
    U[1, 2:, 1] = 0.5                                   # 0.7 at j0 = 2
    U[2, 0:, 1] = 0.8                                   # 1.0 at j0 = 0
    a, gap = _first_order(U, np.full(3, -0.2), 0.5, dtc)
    ev = DS.jump_events(U, a, np.full(3, -0.2))
    ok_e = [(r, j0) for r, j0, _ in ev] == [(0, 4), (1, 2), (2, 0)]
    quiet = []
    same = DS.stall_table(ev, gap, np.repeat(gap[:, None], 4, 1), dtc, "x",
                          show=quiet.append)
    stuck = np.repeat(gap[:, None], 4, 1).copy()
    for r, j0, jm in ev:
        stuck[r, :, j0:] = -jm / dtc                    # never closes
    bad = DS.stall_table(ev, gap, stuck, dtc, "x", show=quiet.append)
    o5 = np.exp(-6 * dtc / 0.5)                         # open after step 5
    ok_s = (ok_e and all(v["gone"] for v in same.values())
            and not any(v["gone"] for v in bad.values()) and len(same) == 3
            and all(abs(v["open5_truth"] - o5) < 1e-9 for v in same.values()))
    ok &= ok_s
    notes.append(f"stall measure: events {[(r, j0) for r, j0, _ in ev]}, "
                 f"open fraction at step 5 {o5:.3f} = truth, same rollout "
                 f"gone / frozen rollout THERE in all 3 bins {ok_s}")
    tmp = tempfile.mkdtemp(prefix="m10_t7_")
    saved = (S2.CACHE, S3.CACHE)
    try:
        # -- step 3 stamps follow the pack
        S3.CACHE = tmp
        pickle.dump([dict(seed=1, code="x")],
                    open(os.path.join(tmp, "A_meta.pkl"), "wb"))
        np.savez(os.path.join(tmp, "A.npz"), U=np.zeros(3))
        f = os.path.join(tmp, "A_e0.npz")
        np.savez(f, E0=np.zeros(3))
        a3 = types.SimpleNamespace(force=False)
        un = S3._fresh(f, a3, "A")                       # unstamped: kept
        S3._stamp(f, "A")
        fr = S3._fresh(f, a3, "A")
        pickle.dump([dict(seed=1, code="y")],
                    open(os.path.join(tmp, "A_meta.pkl"), "wb"))
        st_new = S3._fresh(f, a3, "A")                   # new pack: rebuilt
        S3._stamp(f, "A", [(1, 2, 8, 0)])
        jobs_ch = S3._fresh(f, a3, "A", [(1, 3, 8, 0)])
        ok_st = un and fr and not st_new and not jobs_ch
        ok &= ok_st
        notes.append(f"step-3 stamps: unstamped kept {un}, same pack kept "
                     f"{fr}, new pack rebuilt {not st_new}, other jobs "
                     f"rebuilt {not jobs_ch}")
        # -- step 2 legacy code
        args = types.SimpleNamespace(code="newhash", act_family="old")
        am = types.SimpleNamespace(code="newhash", act_family="m10")
        ok_l = (S2.code_ok("dd91c9b16c09", "A", args)
                and S2.code_ok("newhash", "A", args)
                and not S2.code_ok("dd91c9b16c09", "A", am)
                and not S2.code_ok("dd91c9b16c09", "At", args)
                and S2.code_ok("newhash-m10", "A", am)
                and S2.code_ok("dd91c9b16c09", "C", am))
        ok &= ok_l
        notes.append(f"legacy code accepted only on old-family splits {ok_l}")
        # -- copy / pilot guards
        src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
        os.makedirs(src)
        os.makedirs(dst)
        for fn in S2.COPY_FILES:
            open(os.path.join(src, fn), "wb").write(fn.encode())
        S2.CACHE = dst
        ac = types.SimpleNamespace(copy_from="src", smoke=False, force=False,
                                   act_family="m10", code="h")
        refused = []
        try:
            S2.phase_pilot(ac)                  # m10, no copied library
            refused.append(False)
        except SystemExit:
            refused.append(True)
        open(os.path.join(dst, "lib.npz"), "wb").write(b"made here")
        try:
            S2.phase_copy(ac)                   # differs from the source
            refused.append(False)
        except SystemExit:
            refused.append(True)
        ac.force = True
        S2.phase_copy(ac)                       # copies over it
        ac.force = False
        S2.phase_copy(ac)                       # identical: fine
        cj = json.load(open(os.path.join(dst, "copied.json")))
        S2.phase_pilot(ac)                      # copied: kept, returns
        open(os.path.join(dst, "G.npy"), "wb").write(b"changed")
        try:
            S2.phase_pilot(ac)                  # copied file changed
            refused.append(False)
        except SystemExit:
            refused.append(True)
        ok_c = all(refused) and len(refused) == 3 and set(cj["sha1"]) == \
            set(S2.COPY_FILES) and open(os.path.join(dst, "lib.npz"),
                                        "rb").read() == b"lib.npz"
        ok &= ok_c
        notes.append(f"copy / pilot guards (pilot without copied lib, copy "
                     f"over a differing file, pilot on a changed copy) "
                     f"refused {refused}, sha1 recorded {ok_c}")
        # -- targets carry the pack's code
        from learn.meta import relabel as R
        np.savez(os.path.join(dst, "lib.npz"), mu=0, sd=1, S=0)
        np.save(os.path.join(dst, "G.npy"), np.zeros(1))
        pickle.dump([dict(code="abc-m10")],
                    open(os.path.join(dst, "A_meta.pkl"), "wb"))
        got = {}
        bt = R.build_targets
        R.build_targets = lambda path, lib, procs=1, code="": got.update(
            code=code)
        try:
            S2.phase_targets(types.SimpleNamespace(
                split_list=["A"], force=False, code="h", act_family="m10"))
        finally:
            R.build_targets = bt
        ok_t = got.get("code") == "abc-m10"
        ok &= ok_t
        notes.append(f"targets stamped with the pack's code {got} {ok_t}")
    finally:
        S2.CACHE, S3.CACHE = saved
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    report("7 review fixes", ok, "; ".join(notes))


def main():
    which = sys.argv[1:] or [str(i) for i in range(1, 8)]
    for w in which:
        t0 = time.time()
        globals()[f"test_{w}"]()
        print(f"      ({time.time() - t0:.1f} s)", flush=True)
    fails = RES.get("fails", [])
    print("ALL PASSED" if not fails else f"FAILED: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
