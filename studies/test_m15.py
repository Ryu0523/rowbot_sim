#!/usr/bin/env python3
"""
Tests of M15 (DEFECTS M16, whose code keeps the name m15; brief
BRIEF_PREVIEW.md): the M15 operator family (learn/meta/ops_m15.py,
episode5.py, data5.py), the wave / preview network
(learn/meta/model_preview.py), the driver studies/meta_step5.py and the
evaluation studies/eval_preview_m15.py. One process, smoke caches only
(writes only studies/_cache/meta5_smoke; reads meta2_smoke and meta2's
lib.npz).

  1  nothing the running jobs use changed: the shared files are
     byte-identical to the pre-M15 snapshot (scratchpad backup_pre_m15),
     except model3.py, which differs from it by exactly the 2026-09-30
     upper clamp in window_tokens (MODEL3_FIX),
     meta_step2.code_hash() is still meta3 / meta4d's 8144d9fa6e23 and
     meta_step3.sim_code() meta3's aa6f9637e815; data5.episode with the
     default family is data2.episode bit for bit (the pickled dict, keys
     included) for old (with / without an actuator draw), m10 with and
     without relays, at and a target episode; op_inputs5(v0) = op_inputs;
     data5.branch_sim5 on a v0 episode = data3._branch_sim bit for bit
  2  the M15 operator reads mid-step elevations (float64 episodes): the
     recorded W_MID equals the rebuilt sea evaluated directly at the
     dead-reckoned mid-step pose; the operator replayed on
     op_inputs5(S, U, W_MID) reproduces the recorded rule and noise bit for
     bit, and on the step-start elevations it does not; a P = 1 branch
     with the recorded commands reproduces the recorded states (recorded
     pushes, and the replayed operator's own continuation on the
     simulated W_MID); the v0 rebuild path raises on an M15 meta
  3  target W_MID (data5.target_wmid on meta2_smoke C / Cb, in memory)
     against the twin's own surface at the dead-reckoned pose; the rebuilt
     sea reproduces the recorded elevations at t
  4  the projection patterns span slopes: the library readings of the five
     patterns are unit-std and the four non-mean ones uncorrelated with
     the mean; over many operator draws on meta2's library, the share of
     each wave reading's library variance explained by the mean pattern
     (M15 vs v0); M15 operators pair with v0 seed by seed (every
     non-wave style key equal); direct wave inputs are pattern readings
  5  smoke pipeline on meta5_smoke: refusals (cache name, meta_step2 on
     the M15 cache); meta_step5 copy / data / pack / wmid / e0 / branches
     (procs 1, tiny sizes); copies identical to meta2_smoke; packs carry
     W_MID, op_family m15 and op_seed_m15 only; the old replay raises on
     them; the packed float32 inputs replay the packed rule; stamps;
     train_a / train_w / train_p (smoke steps); eval_preview_m15 end to end
  6  preview conventions (model_preview on meta5_smoke A): measured
     elevations on up to the moment (with sensor noise of the requested
     sd), off after it with the since feature; preview on exactly within
     the horizon, off for variant w; lam = 0 gives W_MID exactly; the
     forecast error grows with lead and is spatially coherent (the
     transverse difference keeps its information); the rollout hook's
     token 0 equals the training token at the moment, and at the true
     states its preview is W_MID
  7  rollout_core_w(wave=None) = model3.rollout_core bit for bit (smoke
     model3.pt); with a hook the preview network rolls out
  8  the review fixes of 2026-09-30 (each check fails on the code before
     them):
     a  training moments anywhere in the window: the moment's position
        covers [0, W_CTX - HB] about uniformly, the loss mask stops J0_MAX
        steps after it, and scored rollout tokens (measured off) occur at
        every position 1 .. W_CTX - 1 (they were only in the last 33);
     b  trained context: the smoke checkpoints record ctx = train_ctx of
        the train split; with net.pos poisoned (NaN) from ctx on, the
        capped one-step evaluation and the rollout with nh = ctx - HB stay
        finite (so they never touch an untrained position), the uncapped
        rollout does not; one_step_windows = model3's two windows for 90-s
        splits and covers every step once for a short context;
     c  model3.window_tokens on an UNPADDED short split (T < W_CTX; it
        indexed past T before) = the padded split on every valid token,
        the encoder's outputs there too;
     d  no torch in the numpy workers: an M15 episode, a branch
        (branch_sim5 on W_MID) and target_wmid never execute an 'import
        torch' (mid_pose imported it on every call); episode5 / data5 /
        ops_m15 import no torch at module level;
     e  the data chunks' code hash (meta_step5.code5) covers exactly the
        episode code: every function an episode runs lives in a file of
        meta_step5.NEW, data5.py is only in the simulation hash, episode5
        does not import data5;
     f  a chunk with failed episodes gets exactly those rerun on the next
        data run (bit-identical to the original episode); checkpoints,
        family files and stamps are written through a temporary file

    python -m studies.test_m15 [1 2 ...]
"""
import hashlib
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SMOKE2 = os.path.join(HERE, "_cache", "meta2_smoke")
SMOKE5 = os.path.join(HERE, "_cache", "meta5_smoke")
META2_LIB = os.path.join(HERE, "_cache", "meta2", "lib.npz")
BACKUP = os.path.join(
    os.environ.get("M15_BACKUP", r"C:\Users\Administrator\AppData\Local\Temp"
                   r"\claude\C--Users-Administrator-Documents"
                   r"\97e7fd1d-c515-4a09-869b-41596c911be0\scratchpad"),
    "backup_pre_m15")
SHARED = ("learn/meta/data2.py", "learn/meta/operators.py",
          "learn/meta/relabel.py", "learn/meta/data3.py",
          "learn/meta/model3.py", "learn/meta/mpc_learned.py",
          "studies/meta_step2.py", "studies/meta_step3.py",
          "studies/info_waves_m11.py", "studies/check_wave_use_m14.py")
# model3.py's only change since the pre-M15 snapshot (2026-09-30, DEFECTS
# M16 review: window_tokens indexed E past T on splits shorter than W_CTX)
MODEL3_FIX = (b"    tp = (t - 1).clamp(min=0)\n",
              b"    tp = (t - 1).clamp(min=0, max=D.T - 1)\n")
TOL = 1e-9
RES = {}


def report(name, ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
    RES.setdefault("fails", [])
    if not ok:
        RES["fails"].append(name)


def _lib(path=None):
    d = np.load(path or os.path.join(SMOKE2, "lib.npz"))
    return dict(mu=d["mu"], sd=d["sd"], S=d["S"])


def _rel(a, b, floor=1e-6):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sc = np.maximum(np.abs(b).max(0), floor)
    return float((np.abs(a - b).max(0) / sc).max())


def _sha(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


def _same_as_snapshot(f):
    """A shared file equals its pre-M15 snapshot (model3.py: up to exactly
    MODEL3_FIX, applied once)."""
    now = open(os.path.join(ROOT, f), "rb").read()
    old = open(os.path.join(BACKUP, os.path.basename(f)), "rb").read()
    if f.endswith("model3.py"):
        if old.count(MODEL3_FIX[0]) != 1:
            return False
        old = old.replace(*MODEL3_FIX)
    return now == old


def _argv_run(mod, argv):
    saved = sys.argv
    sys.argv = [mod.__name__] + argv
    try:
        mod.main()
    finally:
        sys.argv = saved


def _meta_of(e):
    m = dict(seed=e["seed"], relay=e["relay"], sea=e["sea"], act=e["act"],
             world=e["world"], style=e["style"])
    if e.get("op_family") == "m15":
        m.update(op_seed_m15=e["op_seed"], op_family="m15")
    else:
        m["op_seed"] = e["op_seed"]
    return m


def _W_of(eps, lib):
    n = min(len(e["U"]) for e in eps)
    W = dict(S=np.stack([e["S"][:n] for e in eps]),
             XS=np.stack([e["XS"][:n] for e in eps]),
             U=np.stack([e["U"][:n] for e in eps]),
             AV=np.stack([e["AV"][:n] for e in eps]),
             len=np.full(len(eps), n), lib=lib,
             meta=[_meta_of(e) for e in eps])
    W["W_MID"] = (np.stack([e["W_MID"][:n] for e in eps])
                  if all("W_MID" in e for e in eps) else None)
    return W


class _WSwap:
    """relabel._W replaced by a test dict while the block runs."""

    def __init__(self, W):
        self.W = W

    def __enter__(self):
        from learn.meta import relabel as R
        self.saved = dict(R._W)
        R._W.clear()
        R._W.update(self.W, env=self.saved.get("env") or R._env())
        return R

    def __exit__(self, *a):
        from learn.meta import relabel as R
        R._W.clear()
        R._W.update(self.saved)


# ------------------------------------------------------------------ 1
def test_1():
    from learn.meta import data2, data3, data5
    from studies import meta_step2 as S2
    from studies import meta_step3 as S3
    same_files = {f: _same_as_snapshot(f) for f in SHARED}
    ch, sc = S2.code_hash(), S3.sim_code()
    ok = all(same_files.values()) and ch == "8144d9fa6e23" \
        and sc == "aa6f9637e815"
    lib = _lib()
    jobs = [("old", dict(seed=1, world="low", lib=lib, T=12.0)),
            ("old", dict(seed=2, world="low", lib=lib, T=12.0)),
            ("m10", dict(seed=3, world="low", lib=lib, T=12.0,
                         act_family="m10")),
            ("m10 relay", dict(seed=4, world="low", lib=lib, T=12.0,
                               act_family="m10", relay=True)),
            ("at", dict(seed=5, world="low", lib=lib, T=12.0,
                        act_family="at")),
            ("target", dict(seed=300001, world="high", T=6.0))]
    lines = []
    for nm, j in jobs:
        a = pickle.dumps(data2.episode(j))
        b = pickle.dumps(data5.episode(j))
        c = pickle.dumps(data5.episode(dict(j, op_family="v0")))
        eq = a == b == c
        ok &= eq
        lines.append(f"{nm} (seed {j['seed']}) {eq}")
    S = np.random.default_rng(0).normal(size=(3, 7, 26))
    U = np.random.default_rng(1).normal(size=(3, 7, 2))
    oi = np.array_equal(data5.op_inputs5(S, U), data2.op_inputs(S, U))
    ok &= oi
    # branches of a v0 episode: the M15 code path = data3's
    e = data2.episode(dict(seed=6, world="low", lib=lib, T=40.0,
                           act_family="m10", raw64=True))
    W = _W_of([e], lib)
    with _WSwap(W):
        plans = W["U"][0, 60:80][None]
        a, a0 = data3._branch_sim(0, 60, plans)
        b, b0 = data5.branch_sim5(0, 60, plans)
    br = np.array_equal(a, b) and np.array_equal(a0, b0)
    ok &= br
    report("1 v0 paths unchanged", ok,
           f"shared files identical to the pre-M15 snapshot (model3.py "
           f"up to the window_tokens clamp) "
           f"{all(same_files.values())} ({sum(same_files.values())}/"
           f"{len(SHARED)}); meta_step2.code_hash {ch} (meta4d's "
           f"8144d9fa6e23), meta_step3.sim_code {sc} (meta3's stamps "
           f"aa6f9637e815); data5.episode (default and op_family v0) vs "
           f"data2.episode, pickled dicts identical: " + ", ".join(lines)
           + f"; op_inputs5(v0) = op_inputs {oi}; branch_sim5 = "
           f"data3._branch_sim on a v0 episode {br}")


# ------------------------------------------------------------------ 2
def _eps15():
    if "eps15" not in RES:
        from learn.meta import data5
        lib = _lib()
        RES["eps15"] = [data5.episode(dict(
            seed=s, world="low", lib=lib, T=40.0, act_family="m10",
            relay=rl, op_family="m15", raw64=True))
            for s, rl in ((21, False), (22, True), (23, False))]
    return RES["eps15"]


def test_2():
    from learn.meta import data3, data5
    from learn.meta import relabel as R
    from learn.meta.operators import clip_push
    env = R._env()
    dtc = env["dt"] * env["sub"]
    h = 0.5 * dtc
    lib = _lib()
    eps = _eps15()
    ok, lines = True, []
    w_err, d_rule, rep = 0.0, [], True
    for e in eps:
        meta = _meta_of(e)
        sea = R._sea(meta)
        n = len(e["U"])
        xm, ym, pm = data5.mid_pose(e["XS"], "plant14", h)
        X, Y = data5.station_xy(xm, ym, pm, env["x_st"], env["y_off"])
        Wd = data5.sea_eval(sea, X, Y, (np.arange(n) * dtc + h)[:, None,
                                                              None])
        w_err = max(w_err, _rel(Wd.reshape(n, 15), e["W_MID"]))
        diff = np.sqrt(((e["W_MID"] - e["S"][:, 11:26]) ** 2).mean()
                       / (e["S"][:, 11:26] ** 2).mean())
        lines.append(f"seed {e['seed']}: rms(W_MID - W(t)) / rms W(t) "
                     f"{diff:.2f}")
        # replay: the recorded rule and noise, bit for bit
        for fam, X_ in (("m15", data5.op_inputs5(e["S"], e["U"], e["W_MID"],
                                                 "m15")),
                        ("start", data5.op_inputs5(e["S"], e["U"]))):
            op = data5.make_operator(meta, lib, env)
            rng = np.random.default_rng([meta["op_seed_m15"], 1])
            st = op.new_state(1, rng=rng, init_s=X_[0][None])
            RR, RN = np.zeros((n, 5)), np.zeros((n, 5))
            for t in range(n):
                rr, rn = op.step(st, X_[t][None], noise=True, rng=rng)
                RR[t], RN[t] = clip_push(rr[0], rn[0])
            if fam == "m15":
                rep &= np.array_equal(RR, e["RR"]) and np.array_equal(
                    RN, e["RN"])
            else:
                d_rule.append(float(np.sqrt(((RR - e["RR"]) ** 2).sum() / max(
                    (e["RR"] ** 2).sum(), 1e-30))))
    ok &= w_err <= 1e-9 and rep
    # branches with the recorded commands (P = 1)
    W = _W_of(eps, lib)
    det, full = 0.0, []
    with _WSwap(W):
        for i, e in enumerate(eps):
            for k in (60, 120):
                plans = W["U"][i, k:k + 20][None]
                rec = W["XS"][i, k:k + 21]
                XS, _ = data5.branch_sim5(i, k, plans,
                                          pushes=e["R"][k:k + 20][None])
                det = max(det, _rel(XS[0], rec))
                XSn, _ = data5.branch_sim5(i, k, plans)
                full.append(_rel(XSn[0], rec))
        try:
            data3._replay_operator(W["meta"][0], lib,
                                   data5.op_inputs5(W["S"][0], W["U"][0]),
                                   5, R._env())
            refused = False
        except KeyError:
            refused = True
    ok &= det <= TOL and max(full) <= 1e-8 and refused
    styles = [e["style"] for e in eps]
    report("2 operator reads W_MID", ok,
           f"{len(eps)} float64 M15 episodes (m10, one with relays): "
           f"recorded W_MID vs the rebuilt sea at the dead-reckoned mid "
           f"pose, max relative {w_err:.1e} (<= 1e-9); " + "; ".join(lines)
           + f"; operator replayed on op_inputs5(S, U, W_MID) reproduces "
           f"RR and RN bit for bit {rep}; on the step-start elevations the "
           f"rule differs by {', '.join(f'{x:.2f}' for x in d_rule)} "
           f"(relative rms; operators reading waves: "
           f"{[s['reads_waves'] for s in styles]}, same-step "
           f"{[s['wave_same_step'] for s in styles]}); P = 1 branch, "
           f"recorded commands, recorded pushes: max relative state error "
           f"{det:.1e} (<= {TOL:g}); replayed operator's own continuation "
           f"on the simulated W_MID: {max(full):.1e} (<= 1e-8, median "
           f"{np.median(full):.1e}); data3._replay_operator on the M15 meta "
           f"raises KeyError {refused}")


# ------------------------------------------------------------------ 3
def test_3():
    from learn.meta import data5
    from learn.meta import relabel as R
    from learn.repro.task import ctx
    from sim import lofi
    env = R._env()
    dtc = env["dt"] * env["sub"]
    h = 0.5 * dtc
    c = ctx()
    worst, worst_s, n_ep = 0.0, 0.0, 0
    for split in ("C", "Cb"):
        path = os.path.join(SMOKE2, f"{split}.npz")
        W = data5.target_wmid(path, env)
        d = np.load(path)
        meta = pickle.load(open(path.replace(".npz", "_meta.pkl"), "rb"))
        for i in range(len(meta)):
            L = int(d["len"][i])
            sea = R._sea(meta[i])
            twin = lofi.plant_for(c["db"], sea, c["h"], dt=env["dt"])
            XS = d["XS"][i, :L].astype(float)
            for t in range(0, L, 7):
                ref = data5.mid_stations_twin(twin, XS[t], t * dtc, h)
                worst = max(worst, float(np.abs(W[i, t] - ref).max()
                                         / (np.abs(ref).max() + 1e-9)))
                now = data2_stations(twin, XS[t], t * dtc)
                rec = d["S"][i, t, 11:26]
                worst_s = max(worst_s, float(np.abs(now - rec).max()
                                             / (np.abs(rec).max() + 1e-9)))
            n_ep += 1
    ok = worst <= 1e-6 and worst_s <= 1e-4
    report("3 target W_MID", ok,
           f"{n_ep} target episodes (meta2_smoke C, Cb): data5.target_wmid "
           f"(stored float32, as the packs' W_MID) vs the twin's own surface "
           f"at the dead-reckoned mid pose, max relative {worst:.1e} (<= "
           f"1e-6); the rebuilt sea at the recorded (float32) poses vs the "
           f"recorded elevations at t {worst_s:.1e} (<= 1e-4: float32 "
           f"positions of ~1 km carry ~1e-4 m, k dx ~ 1e-5 rad)")


def data2_stations(twin, s, t):
    from learn.meta.data2 import _stations
    return _stations(twin, np.asarray(s, float), t)


# ------------------------------------------------------------------ 4
def test_4():
    from learn.meta import ops_m15 as O
    from learn.meta.operators import G_WAVE, Operator
    lib = _lib(META2_LIB)
    S = np.asarray(lib["S"], float)
    Sf = S.reshape(-1, S.shape[-1])
    Q = O.pattern_projections(Sf, lib["sd"])
    R_ = Sf @ Q.T
    R_ = R_ - R_.mean(0)
    stds = R_.std(0)
    corr0 = [abs(float(R_[:, m] @ R_[:, 0]) / np.sqrt(
        (R_[:, m] @ R_[:, m]) * (R_[:, 0] @ R_[:, 0]))) for m in range(1, 5)]
    B = O.pattern_basis()
    raw = np.zeros((5, 28))
    raw[:, list(G_WAVE)] = B * lib["sd"][list(G_WAVE)] / np.sqrt(
        (B ** 2).sum(1, keepdims=True))
    raw_sd = (Sf @ raw.T).std(0)
    ok = np.allclose(stds, 1.0, atol=1e-9) and max(corr0) < 1e-9
    sh15, sh0, pair, dirs_ok, n_dw = [], [], True, True, 0
    keys = ("K", "types", "taus", "wns", "zetas", "scheduled", "pre",
            "n_dir", "n_events", "event_b", "bursts", "relays", "env",
            "gain_drift", "bias_drift", "regime", "null", "A", "pi_m", "nu",
            "beta", "wave_in")
    N = 120
    for s in range(N):
        seed = 7 * (500 + s) + 1
        o15 = O.OperatorM15(seed, lib, relay=s % 3 == 0)
        o0 = Operator(seed, lib, relay=s % 3 == 0)
        pair &= all(o15.style[k] == o0.style[k] for k in keys)
        sh15 += o15.style["mean_share"]
        for p in ([f["p"] for f in o0.filters]
                  + [ev["src"][1] for ev in o0.events
                     if ev["src"][0] == "own"]
                  + [x["p"] for x in (o0.env, o0.bursts) if x is not None]):
            if np.abs(np.asarray(p)[list(G_WAVE)]).sum() > 0:
                sh0.append(O.mean_share(p, Q, Sf))
        for d in o0.dirs:
            if int(d) in G_WAVE:
                p = np.zeros(28)
                p[int(d)] = 1.0
                sh0.append(O.mean_share(p, Q, Sf))
        wd = [int(d) for d in o15.dirs if int(d) >= 28]
        n_dw += len(wd)
        dirs_ok &= len(wd) == o15.n_x == o15.Wdir.shape[1] and not any(
            int(d) in G_WAVE for d in o15.dirs)
    sh15, sh0 = np.array(sh15), np.array(sh0)
    ok &= pair and dirs_ok and sh15.mean() < 0.35 and \
        (sh15 > 0.5).mean() < 0.25 and sh0.mean() > sh15.mean()
    report("4 projection patterns", ok,
           f"meta2 library: pattern readings unit std {np.round(stds, 6)}, "
           f"|corr| of the non-mean ones with the mean {max(corr0):.1e}; "
           f"raw unit-norm pattern reading std [mean long trans curv "
           f"twist] {np.round(raw_sd, 3)}; {N} operator draws: mean "
           f"pattern's share of a wave reading's library variance, M15 "
           f"mean {sh15.mean():.2f} median {np.median(sh15):.2f}, > 0.5 in "
           f"{(sh15 > 0.5).mean():.2f} of {len(sh15)} readings; v0 mean "
           f"{sh0.mean():.2f} median {np.median(sh0):.2f}, > 0.5 in "
           f"{(sh0 > 0.5).mean():.2f} of {len(sh0)}; every non-wave style "
           f"key paired with v0 seed by seed {pair}; direct wave inputs are "
           f"pattern columns {dirs_ok} ({n_dw} in all)")


# ------------------------------------------------------------------ 5
def test_5():
    from learn.meta import data3, data5
    from learn.meta import relabel as R
    from studies import eval_preview_m15 as EV
    from studies import meta_step2 as S2
    from studies import meta_step5 as S5
    t0 = time.time()
    notes, ok = [], True
    ref = []
    for mod, argv in ((S5, ["--cache-name", "meta4d", "--phase", "data"]),):
        try:
            _argv_run(mod, argv)
            ref.append(False)
        except SystemExit:
            ref.append(True)
    base = ["--smoke", "--procs", "1", "--t-train", "20", "--size",
            "train=12", "--size", "A=3", "--size", "B=2", "--size", "At=3"]
    _argv_run(S5, base + ["--phase", "copy,data,pack,wmid,e0,branches"])
    t_data = time.time() - t0
    # meta_step2 must refuse the M15 cache (its families are old / m10)
    try:
        _argv_run(S2, ["--cache-name", "meta5", "--smoke", "--act-family",
                       "m10", "--phase", "data"])
        ref.append(False)
    except SystemExit:
        ref.append(True)
    ok &= all(ref)
    notes.append(f"refusals (meta_step5 on meta4d, meta_step2 on "
                 f"meta5_smoke) {ref}")
    same = all(_sha(os.path.join(SMOKE2, f)) == _sha(os.path.join(SMOKE5, f))
               for f in S2.COPY_FILES)
    ok &= same
    good = True
    for sp in ("train", "A", "B", "At"):
        d = np.load(os.path.join(SMOKE5, f"{sp}.npz"))
        meta = pickle.load(open(os.path.join(SMOKE5, f"{sp}_meta.pkl"),
                                "rb"))
        good &= "W_MID" in d.files and d["W_MID"].shape == d["S"].shape[
            :2] + (15,)
        good &= all(m["op_family"] == "m15" and "op_seed" not in m
                    and "op_seed_m15" in m and "wave_same_step" in m["style"]
                    for m in meta)
        good &= all(m["act_family"] == ("at" if sp == "At" else "m10")
                    for m in meta)
    ok &= good
    notes.append(f"step 2 + wmid / e0 / branches {t_data:.0f} s; copies "
                 f"identical to meta2_smoke {same}; packs carry W_MID, "
                 f"op_family m15, op_seed_m15 only, families m10 / at {good}")
    # the old replay path raises on the M15 pack; the packed float32
    # inputs replay the packed rule
    lib = _lib(os.path.join(SMOKE5, "lib.npz"))
    path = os.path.join(SMOKE5, "A.npz")
    d = np.load(path)
    meta = pickle.load(open(path.replace(".npz", "_meta.pkl"), "rb"))
    env = R._env()
    try:
        data3._replay_operator(meta[0], lib, data5.op_inputs5(d["S"][0],
                                                              d["U"][0]),
                               3, env)
        refused = False
    except KeyError:
        refused = True
    worst = 0.0
    for i in range(len(meta)):
        n = int(d["len"][i])
        X = data5.op_inputs5(d["S"][i, :n], d["U"][i, :n], d["W_MID"][i, :n],
                             "m15")
        op = data5.make_operator(meta[i], lib, env)
        rng = np.random.default_rng([meta[i]["op_seed_m15"], 1])
        st = op.new_state(1, rng=rng, init_s=X[0][None])
        RR = np.zeros((n, 5))
        for t in range(n):
            rr, _ = op.step(st, X[t][None], noise=True, rng=rng)
            RR[t] = rr[0]
        lim = 3.0 * np.array([4.0, 4.0, 2.0, 12.0, 3.0])
        RR = np.clip(RR, -lim, lim)
        sc = np.abs(d["RR"][i, :n]).max() + 1e-9
        worst = max(worst, float(np.abs(RR[:60] - d["RR"][i, :60]).max()
                                 / sc))
    ok &= refused and worst <= 1e-4
    notes.append(f"data3._replay_operator raises on the M15 meta {refused}; "
                 f"replay on the packed float32 inputs reproduces the packed "
                 f"rule over the first 60 steps to {worst:.1e} of its max "
                 f"(<= 1e-4; float32 inputs)")
    stamps = all(open(os.path.join(SMOKE5, f + ".code")).read().startswith(
        S5.sim5()) for f in ("A_branches.npz", "At_branches.npz",
                             "C_wmid.npz", "Cb_wmid.npz", "A_e0.npz"))
    ok &= stamps
    notes.append(f"stamps = M15 simulation code + pack {stamps}")
    t1 = time.time()
    _argv_run(S5, base + ["--phase", "train_a,train_w,train_p"])
    t_train = time.time() - t1
    t1 = time.time()
    _argv_run(EV, ["--smoke", "--n-samp", "4", "--one-samp", "16"])
    t_eval = time.time() - t1
    res = pickle.load(open(os.path.join(SMOKE5, "eval_m15.pkl"), "rb"))
    kinds = sorted({k[1] for k in res if k[0] != "use"})
    n_use = sum(1 for k in res if k[0] == "use")
    fin = all(np.isfinite(r["skill"]).all() for k, r in res.items()
              if k[0] != "use")
    ok &= fin and kinds == ["multi", "one"] and n_use > 0
    notes.append(f"train a / w / p {t_train:.0f} s; eval {t_eval:.0f} s, "
                 f"{len(res)} results ({kinds}, {n_use} wave-use rows), all "
                 f"finite {fin}")
    report("5 smoke pipeline", ok, "; ".join(notes))


# ------------------------------------------------------------------ 6
def test_6():
    import torch

    from learn.meta import model3 as M
    from learn.meta import model_preview as MP
    dev = M.device()
    ck = torch.load(os.path.join(SMOKE5, "model3.pt"), weights_only=False)
    D = MP.DataP(SMOKE5, "A", dev, stats=ck["stats"])
    notes, ok = [], True
    gen = torch.Generator().manual_seed(3)
    ii = torch.arange(D.n, device=dev).repeat(6)
    B, L = len(ii), M.W_CTX
    a = torch.zeros(B, dtype=torch.long, device=dev)
    okm = MP.window_ok(D, ii, a, L)
    cfg = MP.draw_cfg(D, a, okm, gen, "p")
    t = a[:, None] + torch.arange(L, device=dev)[None]
    wx = MP.wave_cols(D, ii, t, cfg, gen)
    rel = t - cfg["m"][:, None]
    meas = wx[..., MP.C_MON] > 0.5
    c1 = bool((meas == (rel <= 0)).all())
    c2 = bool((wx[..., MP.C_WM][~meas] == 0).all()) and bool(torch.allclose(
        wx[..., MP.C_SINCE], rel.clamp(min=0).float() / M.HB))
    nz = (wx[..., MP.C_WM] - D.Wm[ii[:, None], t.clamp(max=D.T - 1)])
    live = meas & (t < D.len[ii][:, None])
    est = (nz[live].pow(2).mean(-1).sqrt()).mean().item()
    want = cfg["msd"][:, None].expand(B, L)[live].mean().item()
    c3 = abs(est / want - 1) < 0.1
    hp = cfg["hp"][:, None]
    pon = wx[..., MP.C_PON] > 0.5
    c4 = bool((pon == ((rel < hp) & (hp > 0))).all()) and bool(
        (wx[..., MP.C_WP][~pon] == 0).all())
    z = (cfg["lam"] == 0) & (cfg["hp"] > 0)
    tl = (t < D.len[ii][:, None])
    c5 = bool((wx[z][..., MP.C_WP][pon[z] & tl[z]] == D.Wmid[ii[z][:, None],
                                                              t[z].clamp(
        max=D.T - 1)][pon[z] & tl[z]]).all()) if z.any() else None
    cw = MP.draw_cfg(D, a, okm, torch.Generator().manual_seed(3), "w")
    c6 = bool((cw["hp"] == 0).all())
    ok &= c1 and c2 and c3 and c4 and c5 is not False and c6
    notes.append(f"measured flag = (t <= m) {c1}; off after m with since = "
                 f"(t - m) / HB {c2}; sensor noise rms {est:.4f} vs the "
                 f"drawn sd {want:.4f} {c3}; preview on exactly for 0 <= "
                 f"t - m < hp (history included), 0 elsewhere {c4}; lam = 0 "
                 f"rows: preview = W_MID exactly {c5}; variant w never has "
                 f"a preview {c6}")
    # error growth with lead, and coherence
    n = min(D.n, 3)
    iiq = torch.arange(n, device=dev)
    tq = torch.arange(40, 300, device=dev)[None].expand(n, -1)
    rows = []
    for lead_steps in (0, 4, 12, 24):
        # the same perturbation draw at every lead (fc_eval takes the
        # lead explicitly)
        cf = MP.fixed_cfg(D, torch.zeros(n, device=dev), 0.1, 100, 0.0,
                          torch.Generator().manual_seed(11))
        tt = tq
        pm = D.PM[iiq[:, None], tt]
        X, Y = MP.stations_t(pm[..., 0], pm[..., 1], pm[..., 2], D.x_st,
                             D.y_off)
        lead = torch.full(tt.shape, lead_steps * D.dtc + D.h, device=dev)
        err, _ = MP.fc_eval(D.SEA[iiq], X, Y, tt.float() * D.dtc + D.h, lead,
                            cf)
        e = err / D.w_sd
        tr = (D.Wmid[iiq[:, None], tt].view(n, -1, 5, 3)[..., 2]
              - D.Wmid[iiq[:, None], tt].view(n, -1, 5, 3)[..., 0])
        etr = e.view(n, -1, 5, 3)[..., 2] - e.view(n, -1, 5, 3)[..., 0]
        rows.append((lead_steps, e.pow(2).mean().sqrt().item(),
                     (etr.pow(2).mean() / tr.pow(2).mean()).sqrt().item()))
    grows = all(rows[q + 1][1] > rows[q][1] for q in range(len(rows) - 1))
    # iid noise of the same rms per station would give a transverse error
    # of sqrt(2) rms / rms(transverse difference)
    tr_all = (D.Wmid.view(D.n, D.T, 5, 3)[..., 2]
              - D.Wmid.view(D.n, D.T, 5, 3)[..., 0])
    tr_rms = tr_all[D.valid].pow(2).mean().sqrt().item()
    iid = [np.sqrt(2) * r[1] / tr_rms for r in rows]
    coh = all(r[2] < q for r, q in zip(rows, iid))
    ok &= grows and coh
    notes.append("forecast error at level 0.1, lead steps / rms (library "
                 "units) / transverse-difference error rms over its signal "
                 "rms (iid of the same size): " + ", ".join(
                     f"{r[0]}: {r[1]:.3f} / {r[2]:.2f} ({q:.2f})"
                     for r, q in zip(rows, iid))
                 + f"; grows with lead {grows}; coherent {coh}")
    # the rollout hook: token 0 = the training token at the moment; at the
    # true states the preview is W_MID
    k = torch.tensor([150] * n, device=dev)
    g1, g2 = (torch.Generator().manual_seed(5) for _ in range(2))
    hook = MP.WaveRoll(D, iiq, k, 0.0, 30, 0.0, gen=g1)
    cfg_t = MP.fixed_cfg(D, k, 0.0, 30, 0.0, g2)
    tok_m = MP.wave_cols(D, iiq, k[:, None], cfg_t, g2)[:, 0]
    xs = torch.as_tensor(D.XS[:n, 150:160].astype(float), device=dev)
    sr0 = M.plant_to_reduced_t(xs[:, 0])[:, None, None]
    h0 = hook(0, sr0)[:, 0, 0]
    c7 = bool(torch.equal(h0, tok_m))
    errs = []
    for j in (1, 5, 9):
        srj = M.plant_to_reduced_t(xs[:, j])[:, None, None]
        hj = hook(j, srj)[:, 0, 0]
        errs.append((hj[:, MP.C_WP] - D.Wmid[iiq, 150 + j]).abs().max().item()
                    / D.Wmid[iiq, 150 + j].abs().max().item())
        c7 &= bool((hj[:, MP.C_WM] == 0).all()) and bool(
            (hj[:, MP.C_MON] == 0).all())
    hb = MP.WaveRoll(D, iiq, k, 0.1, 4, 0.0, gen=g1)(6, sr0)
    c8 = bool((hb[..., MP.C_WP] == 0).all()) and bool(
        (hb[..., MP.C_PON] == 0).all())
    ok &= c7 and max(errs) < 1e-4 and c8
    notes.append(f"hook token 0 = training token at the moment {c7}; hook "
                 f"preview at the true states vs W_MID max relative "
                 f"{max(errs):.1e} (float32, < 1e-4); beyond the horizon "
                 f"off {c8}")
    report("6 preview conventions", ok, "; ".join(notes))


# ------------------------------------------------------------------ 7
def test_7():
    import torch

    from learn.meta import model3 as M
    from learn.meta import model_preview as MP
    from learn.meta import relabel as R
    dev = M.device()
    env = R._env()
    ck = torch.load(os.path.join(SMOKE5, "model3.pt"), weights_only=False)
    net = MP.load_variant(ck, dev)
    D = MP.DataP(SMOKE5, "A", dev, stats=ck["stats"])
    rng = np.random.default_rng(5)
    B, P, S = D.n, 3, 3
    ii = np.arange(B)
    k = np.array([0] + [int(x) for x in rng.integers(60, 300, B - 1)])
    xs = D.XS[ii, k].astype(float)
    pl = torch.tensor(np.clip(rng.uniform(-1, 1, (B, P, M.HB, 2)), [0, -1],
                              [1, 1]))
    base = torch.randn((B, P, S, M.HB, M.C7),
                       generator=torch.Generator().manual_seed(9))
    new = MP.rollout_core_w(net, D, ii, k, pl, xs, env, S, base=base)
    old = M.rollout_core(net, D, ii, k, pl, xs, env, S, base=base)
    same = all(torch.equal(x, y) for x, y in zip(new, old))
    ckp = torch.load(os.path.join(SMOKE5, "model_p.pt"), weights_only=False)
    netp = MP.load_variant(ckp, dev)
    hook = MP.WaveRoll(D, ii, k, 0.1, 8, 0.01,
                       gen=torch.Generator().manual_seed(1))
    e, s, _ = MP.rollout_core_w(netp, D, ii, k, pl, xs, env, S, base=base,
                                wave=hook)
    fin = bool(torch.isfinite(e).all() and torch.isfinite(s).all())
    report("7 rollout hook", same and fin,
           f"rollout_core_w(wave=None) vs model3.rollout_core, {B} moments "
           f"(k = 0 included) x {P} plans x {S} samples: bit-identical "
           f"{same}; the preview network with a WaveRoll hook (level 0.1, "
           f"horizon 8): e {tuple(e.shape)}, finite {fin}")


# ------------------------------------------------------------------ 8
def _torch_imports(fn):
    """Run fn() and return every 'import torch...' executed meanwhile (an
    import statement calls builtins.__import__ even when torch is already
    loaded)."""
    import builtins
    seen, orig = [], builtins.__import__

    def hook(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            seen.append(name)
        return orig(name, *a, **k)
    builtins.__import__ = hook
    try:
        fn()
    finally:
        builtins.__import__ = orig
    return seen


def _module_imports(path):
    import ast
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add(node.module or "")
    return out


def test_8():
    import inspect

    import torch

    from learn.meta import data5, episode5
    from learn.meta import model3 as M
    from learn.meta import model_preview as MP
    from learn.meta import ops_m15
    from learn.meta import relabel as R
    from studies import eval_preview_m15 as EV
    from studies import meta_step5 as S5
    dev = M.device()
    env = R._env()
    notes, ok = [], True
    ck = torch.load(os.path.join(SMOKE5, "model3.pt"), weights_only=False)
    DA = MP.DataP(SMOKE5, "A", dev, stats=ck["stats"])
    # a: training moments anywhere in the window
    gen = torch.Generator().manual_seed(8)
    pool = torch.arange(DA.n)
    Lmax = DA.len.cpu()
    L = M.W_CTX
    mr, cover = [], torch.zeros(L, dtype=torch.long)
    for _ in range(16):
        ii, a = M.fm_draw(pool, 256, gen, Lmax, L, dev)
        okw = MP.window_ok(DA, ii, a, L)
        cfg = MP.draw_cfg(DA, a, okw, gen, "p")
        rel_m = (cfg["m"] - a).cpu()
        mr.append(rel_m[rel_m < L])
        t = a[:, None] + torch.arange(L, device=dev)[None]
        rel = t - cfg["m"][:, None]
        roll = okw & (rel >= 1) & (rel <= MP.J0_MAX)
        cover += roll.sum(0).cpu()
    mr = torch.cat(mr).float()
    q4 = [float(((mr >= L * i / 4) & (mr < L * (i + 1) / 4)).float().mean())
          for i in range(4)]
    ca = int(mr.min()) == 0 and int(mr.max()) >= L - M.HB and all(
        0.18 < x < 0.32 for x in q4) and bool((cover[1:] > 0).all())
    ii, a = M.fm_draw(pool, 32, gen, Lmax, L, dev)
    tok, tgt, okl, cfg = MP.train_tokens_p(DA, ii, a, L, gen, "p")
    t = a[:, None] + torch.arange(L, device=dev)[None]
    want = MP.window_ok(DA, ii, a, L) & (t - cfg["m"][:, None] <= MP.J0_MAX)
    ca &= bool(torch.equal(okl, want))
    ok &= ca
    notes.append(f"a: moment position over {len(mr)} rolled windows min "
                 f"{int(mr.min())} max {int(mr.max())} (>= {L - M.HB}), "
                 f"quarter shares {', '.join(f'{x:.2f}' for x in q4)}; scored "
                 f"rollout tokens at every position 1..{L - 1} "
                 f"{bool((cover[1:] > 0).all())} (fewest "
                 f"{int(cover[1:].min())}); loss mask = valid & t - m <= J0_MAX "
                 f"{bool(torch.equal(okl, want))} -> {ca}")
    # b: trained context
    Dtr = M.Data3(SMOKE5, "train", dev, stats=ck["stats"])
    ctx = MP.train_ctx(Dtr)
    cks = {v: torch.load(os.path.join(SMOKE5, f), weights_only=False)
           for v, f in (("a", "model3.pt"), ("w", "model_w.pt"),
                        ("p", "model_p.pt"))}
    rec = {v: c.get("ctx") for v, c in cks.items()}
    cb = all(c == ctx for c in rec.values()) and EV.trained_ctx(cks) == ctx \
        and ctx < M.W_CTX
    net = MP.load_variant(ck, dev)
    with torch.no_grad():
        net.pos[ctx:] = float("nan")
    q = torch.quantile(DA.E[DA.valid].abs(), 0.99, dim=0)
    r1 = MP.eval_one_step_p(net, DA, q, variant="a", n_samp=4, L=ctx)
    fin1 = bool(np.isfinite(r1["skill"]).all())
    rng = np.random.default_rng(8)
    iiA = np.arange(DA.n)
    kA = np.full(DA.n, int(DA.len.min()) - M.HB - 1)
    xs = DA.XS[iiA, kA].astype(float)
    pl = torch.tensor(np.clip(rng.uniform(-1, 1, (DA.n, 2, M.HB, 2)),
                              [0, -1], [1, 1]))
    e_c, _, _ = MP.rollout_core_w(net, DA, iiA, kA, pl, xs, env, 2,
                                  nh=ctx - M.HB)
    e_u, _, _ = MP.rollout_core_w(net, DA, iiA, kA, pl, xs, env, 2)
    fin2, nan_u = bool(torch.isfinite(e_c).all()), bool(
        torch.isnan(e_u).any())
    w90 = MP.one_step_windows(360, M.W_CTX, 40)
    ws = MP.one_step_windows(360, ctx, 40)
    steps = sorted(s for _, lo, hi in ws for s in range(lo, hi))
    cov_ok = steps == list(range(40, 360)) and all(
        a0 >= 0 and hi - a0 <= ctx and lo - a0 >= 40 for a0, lo, hi in ws)
    cb &= fin1 and fin2 and nan_u and w90 == [(0, 40, 256), (104, 256, 360)] \
        and cov_ok
    ok &= cb
    notes.append(f"b: train split's context {ctx}, recorded in the "
                 f"checkpoints {rec}; net.pos NaN from {ctx} on: capped "
                 f"one-step finite {fin1}, rollout with nh = {ctx - M.HB} at "
                 f"k = {int(kA[0])} finite {fin2}, uncapped rollout NaN "
                 f"{nan_u}; one_step_windows for 90 s = model3's {w90}, "
                 f"context {ctx}: {len(ws)} windows covering [40, 360) once "
                 f"with >= 40 steps of history {cov_ok} -> {cb}")
    # c: model3.window_tokens on an unpadded short split
    D0 = M.Data3(SMOKE5, "train", dev, stats=ck["stats"])
    D1 = MP.Data3P(SMOKE5, "train", dev, stats=ck["stats"])
    iiT = torch.arange(D0.n, device=dev)
    aT = torch.zeros_like(iiT)
    try:
        t0_, g0, o0 = M.window_tokens(D0, iiT, aT, M.W_CTX)
        t1_, g1, o1 = M.window_tokens(D1, iiT, aT, M.W_CTX)
        with torch.no_grad():
            net0 = MP.load_variant(ck, dev)
            h0, h1 = net0.encode(t0_), net0.encode(t1_)
        cc = bool(torch.equal(o0, o1)) and bool(torch.equal(
            t0_[o0], t1_[o1])) and bool(torch.equal(g0[o0], g1[o1]))
        dh = float((h0[o0] - h1[o1]).abs().max())
        cc &= dh < 1e-4
        msg = (f"T = {D0.T} < {M.W_CTX}: tokens, targets and mask equal to "
               f"the padded split on {int(o0.sum())} valid tokens "
               f"{cc and dh < 1e-4}, encoder outputs there max diff "
               f"{dh:.1e}")
    except (IndexError, RuntimeError) as ex:
        cc, msg = False, f"raises {type(ex).__name__}: {ex}"
    ok &= cc
    notes.append(f"c: model3.window_tokens unpadded, {msg} -> {cc}")
    # d: no torch in the numpy workers
    lib = _lib()

    def work():
        e = data5.episode(dict(seed=31, world="low", lib=lib, T=8.0,
                               act_family="m10", op_family="m15"))
        assert len(e["W_MID"]) > 20
        eps = _eps15()
        W = _W_of(eps, lib)
        with _WSwap(W):
            data5.branch_sim5(0, 60, W["U"][0, 60:66][None])
        data5.target_wmid(os.path.join(SMOKE2, "C.npz"), env)
        data5.mid_pose(np.zeros((4, 14)), "plant14", 0.12)
    seen = _torch_imports(work)
    top = {f: sorted(m for m in _module_imports(os.path.join(ROOT, f))
                     if m.split(".")[0] == "torch")
           for f in ("learn/meta/episode5.py", "learn/meta/data5.py",
                     "learn/meta/ops_m15.py")}
    xs_t = torch.zeros(3, 10, dtype=torch.float64)
    tt = data5.mid_pose(xs_t, "red10", 0.12)
    cd = not seen and not any(top.values()) and torch.is_tensor(tt[0])
    ok &= cd
    notes.append(f"d: 'import torch' executed by an M15 episode, a branch "
                 f"and target_wmid: {seen or 'none'}; module-level torch "
                 f"imports {top}; mid_pose still returns tensors for a "
                 f"tensor {torch.is_tensor(tt[0])} -> {cd}")
    # e: the chunks' code hash covers exactly the episode code
    fns = (data5.episode, data5.run_job, data5.pack5, data5.mid_pose,
           data5.station_xy, data5.mid_stations_twin, ops_m15.OperatorM15,
           ops_m15.pattern_projections, ops_m15.draw_pattern)
    files = {os.path.relpath(inspect.getsourcefile(f), ROOT).replace(
        "\\", "/") for f in fns}
    ce = files <= set(S5.NEW) and "learn/meta/data5.py" not in S5.NEW \
        and "learn/meta/data5.py" in S5.SIM_NEW and not any(
            "data5" in m for m in _module_imports(os.path.join(
                ROOT, "learn/meta/episode5.py"))) \
        and S5.code5() == S5._hash(tuple(__import__(
            "studies.meta_step2", fromlist=["x"]).SOURCES) + S5.NEW) \
        and episode5.episode is data5.episode
    ok &= ce
    notes.append(f"e: episode code in {sorted(files)} (code5 files "
                 f"{list(S5.NEW)}), data5.py only in sim5 -> {ce}")
    # f: failed episodes rerun; atomic writes
    f = os.path.join(SMOKE5, "raw", "B_000.pkl")
    ch = pickle.load(open(f, "rb"))
    # per episode: a list pickle also records objects shared ACROSS
    # episodes, which differ between a loaded chunk and a fresh run
    orig = [pickle.dumps(e) for e in ch["eps"]]
    lost = ch["eps"][-1]["seed"]
    bad = dict(ch, eps=ch["eps"][:-1], n_errors=1)
    with open(f + ".tmp", "wb") as fh:
        pickle.dump(bad, fh)
    os.replace(f + ".tmp", f)
    _argv_run(S5, ["--smoke", "--procs", "1", "--splits", "B", "--size",
                   f"B={len(ch['seeds'])}", "--phase", "data"])
    ch2 = pickle.load(open(f, "rb"))
    cf = [pickle.dumps(e) for e in ch2["eps"]] == orig and         ch2["n_errors"] == 0
    src = open(os.path.join(ROOT, "studies/meta_step5.py"),
               encoding="utf-8").read()
    probe = os.path.join(SMOKE5, "atomic_probe.pt")
    S5._save_ckpt(dict(x=1), probe)
    at = torch.load(probe, weights_only=False) == dict(x=1) and \
        not os.path.exists(probe + ".tmp") and src.count("torch.save(") == 1 \
        and "json.dump(want, open" not in src
    os.remove(probe)
    ok &= cf and at
    notes.append(f"f: chunk B_000 with episode {lost} dropped and n_errors 1: "
                 f"the data run reran it, chunk identical to the original "
                 f"{cf}; checkpoints / json / stamps through a temporary "
                 f"file {at}")
    report("8 review fixes", ok, "; ".join(notes))


def main():
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8"]
    for w in which:
        t0 = time.time()
        globals()[f"test_{w}"]()
        print(f"      ({time.time() - t0:.1f} s)", flush=True)
    fails = RES.get("fails", [])
    print("ALL PASSED" if not fails else f"FAILED: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
