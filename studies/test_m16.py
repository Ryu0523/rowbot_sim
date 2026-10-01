#!/usr/bin/env python3
"""
Tests of the meta6 pipeline (learn/meta/episode6.py, data6.py,
studies/meta_step6.py). One process, numpy only, a few minutes.

  1  an episode6 episode (Mission path: RBPlant + the family's injector,
     M10 actuators, M15 wave inputs) equals the family's batched closed-
     loop rollout of the same draw (operators_gen.simulate_gen /
     operators_rb.simulate_rb with the same operator seed, noise stream,
     sea, actuator draw and the recorded commands), both families; for
     'gen' also the safety quantities APK / AMIN / HMIN
  2  a branch from a snapshot (data6.continue_world, operator rebuilt from
     the meta) under the recorded commands equals the episode's own
     continuation, both families; another plan runs and stays finite
  3  no torch in the workers' code: after episodes, packing and branches
     (and importing studies.meta_step6) torch is not loaded
  4  the running pipelines are untouched: every .code stamp in
     studies/_cache/meta4d and meta5, and meta_step2.code_hash /
     meta_step5.code5 / sim5, are the same before and after this test
  5  the packed arrays have meta5's keys, shapes and dtypes (pack6 of an
     episode6 episode vs data5.pack5 of the episode5 M15 episode of the
     same seed; and meta5_smoke/train.npz's keys and dtypes when present),
     the extra arrays APK / AMIN / HMIN / PUSH are float32 (n, T[, 5]),
     the meta has op_family 'm15', prior_family and no op_seed / op_seed_m15

    python studies/test_m16.py
"""
import glob
import hashlib
import os
import pickle
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

FAIL = []
TOL = 1e-9
T_EP = 20.0
K_SNAP = 30
SEED = 100007                      # an A-split seed


def check(name, ok, msg=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {msg}", flush=True)
    if not ok:
        FAIL.append(name)


def stamps():
    out = {}
    for d in ("meta4d", "meta5"):
        for f in sorted(glob.glob(os.path.join(ROOT, "studies", "_cache", d,
                                               "**", "*.code"),
                                  recursive=True)):
            out[os.path.relpath(f, ROOT)] = hashlib.sha1(
                open(f, "rb").read()).hexdigest()
    from studies import meta_step2 as S2
    from studies import meta_step5 as S5
    out["code_hash(meta_step2)"] = S2.code_hash()
    out["code5"], out["sim5"] = S5.code5(), S5.sim5()
    return out


def lib():
    d = np.load(os.path.join(ROOT, "studies", "_cache", "meta2", "lib.npz"))
    return dict(mu=d["mu"], sd=d["sd"], S=d["S"])


def job(fam, **kw):
    j = dict(seed=SEED, world="low", relay=False, act_family="m10",
             family=fam, lib=lib(), T=T_EP, raw64=True, snaps=(K_SNAP,))
    j.update(kw)
    return j


# ------------------------------------------------------------------ 1
def test_batched(fam, ep):
    from learn.meta import episode6 as E6
    from learn.meta import operators_gen as G
    from learn.meta import operators_rb as R
    from learn.meta import relabel
    from sim import lofi
    env = relabel._env()
    n = len(ep["E"])
    dtc = env["dt"] * env["sub"]
    op = E6.make_op(fam, ep["op_seed"], lib(), dtc, env["L"])
    g = np.random.default_rng([ep["op_seed"], 1])
    s28_0 = np.concatenate([ep["S"][0, :11], ep["W_MID"][0], ep["U"][0]])
    groups = [dict(op=op, st=op.new_state(1, rng=g, init_s=s28_0[None]),
                   rows=np.array([0]), rng=g)]
    seas = R.RowSeas([relabel._sea(dict(seed=ep["seed"], sea=ep["sea"]))],
                     env["x_st"], env["y_off"])
    act = {k: np.array([ep["act"][k]], float) for k in lofi.ACT_KEYS}
    sim = G.simulate_gen if fam == "gen" else R.simulate_rb
    out = sim(env, ep["XS"][0][None], ep["U"][None], np.zeros(1), seas,
              groups, n, act=act)
    XS = out["XS"][0]
    err = float(np.abs(XS[:n] - ep["XS"]).max())
    scale = float(np.abs(ep["XS"]).max())
    check(f"{fam}: episode = batched rollout ({n} steps, M10 actuators)",
          err <= TOL * max(scale, 1.0),
          f"(max |diff| {err:.1e}, state scale {scale:.0f}, div "
          f"{ep['div']})")
    if fam == "gen":
        e = max(float(np.abs(out[k][0] - ep[k]).max())
                for k in ("APK", "AMIN", "HMIN"))
        check("gen: safety quantities = simulate_gen's", e <= TOL,
              f"(max |diff| {e:.1e})")


# ------------------------------------------------------------------ 2
def test_branch(fam, ep):
    from learn.meta import data6
    from learn.meta import episode6 as E6
    from learn.meta import relabel
    env = relabel._env()
    arrs, meta, snaps = E6.pack6([ep])
    m0 = meta[0]
    op = data6.rebuild_op(m0, lib(), env)
    snap = snaps[0][K_SNAP]
    H = 24
    plan = np.asarray(ep["U"][K_SNAP:K_SNAP + H], float)[None]
    XS, E0, div = data6.continue_world(op, m0, snap, plan, env)
    ref = ep["XS"][K_SNAP:K_SNAP + H + 1]
    err = float(np.abs(XS[0] - ref).max())
    check(f"{fam}: branch from the snapshot at step {K_SNAP} = the "
          "episode's continuation", err <= TOL * max(np.abs(ref).max(), 1),
          f"(max |diff| {err:.1e})")
    other = plan.copy()
    other[0, :, 0] = np.clip(other[0, :, 0] + 0.2, 0, 1)
    other[0, 8:, 1] = np.clip(other[0, 8:, 1] - 0.4, -1, 1)
    XS2, E02, div2 = data6.continue_world(op, m0, snap, other, env)
    check(f"{fam}: another plan runs, finite, differs", bool(
        np.isfinite(XS2).all() and np.isfinite(E02).all()
        and np.abs(XS2 - XS).max() > 1e-6), f"(div {div2[0]})")
    snaps_ok = len(pickle.dumps(snaps)) > 0
    check(f"{fam}: snapshots pickle", snaps_ok)


# ------------------------------------------------------------------ 5
def test_formats(ep):
    from learn.meta import data5
    from learn.meta import episode6 as E6
    j5 = dict(seed=SEED, world="low", relay=False, act_family="m10",
              op_family="m15", lib=lib(), T=T_EP)
    e5 = data5.episode(j5)
    a5, m5 = data5.pack5([e5])
    ep32 = dict(ep)
    for k in list(ep32):
        if isinstance(ep32[k], np.ndarray) and ep32[k].dtype == np.float64:
            ep32[k] = ep32[k].astype(np.float32)
    a6, m6, _ = E6.pack6([ep32])
    bad = [k for k in a5 if k not in a6 or a6[k].dtype != a5[k].dtype
           or a6[k].shape != a5[k].shape]
    check("pack6 has every pack5 array with its shape and dtype", not bad,
          f"(mismatch {bad}; keys {sorted(a5)})")
    same_draw = (m6[0]["sea"] == m5[0]["sea"]
                 and m6[0]["act"] == m5[0]["act"]
                 and m6[0]["informative"] == m5[0]["informative"])
    check("episode6 seed s has meta5's sea, actuators and scenario draw",
          same_draw)
    n, T = a6["E"].shape[:2]
    ext = all(a6[k].dtype == np.float32 and a6[k].shape == (n, T)
              for k in ("APK", "AMIN", "HMIN")) \
        and a6["PUSH"].shape == (n, T, 5)
    check("extra arrays APK / AMIN / HMIN (n, T), PUSH (n, T, 5) float32",
          ext)
    mm = m6[0]
    check("meta: op_family 'm15', prior_family, op_seed_<family>, no "
          "op_seed / op_seed_m15",
          mm["op_family"] == "m15" and mm["prior_family"] == ep["family"]
          and f"op_seed_{ep['family']}" in mm and "op_seed" not in mm
          and "op_seed_m15" not in mm and "null" in mm["style"]
          and "reads_waves" in mm["style"]
          and "wave_same_step" in mm["style"])
    if ep["family"] == "gen":
        check("gen style: null False (the Coriolis difference is always "
              "on), no_forces present",
              mm["style"]["null"] is False and "no_forces" in mm["style"])
    f = os.path.join(ROOT, "studies", "_cache", "meta5_smoke", "train.npz")
    if os.path.exists(f):
        d = np.load(f)
        bad = [k for k in d.files if k not in a6 or a6[k].dtype != d[k].dtype
               or a6[k].ndim != d[k].ndim or a6[k].shape[2:] != d[k].shape[2:]]
        check("keys, dtypes and trailing shapes of meta5_smoke/train.npz",
              not bad, f"(mismatch {bad})")


def main():
    t0 = time.time()
    before = stamps()
    from studies import meta_step6  # noqa: F401  (its import chain)
    from learn.meta import episode6 as E6
    eps = {}
    for fam in ("gen", "rb"):
        print(f"1/2. family {fam}", flush=True)
        t1 = time.time()
        eps[fam] = E6.episode(job(fam))
        print(f"  episode of {T_EP:.0f} s: {time.time() - t1:.1f} s "
              f"({len(eps[fam]['E'])} steps)", flush=True)
        test_batched(fam, eps[fam])
        test_branch(fam, eps[fam])
    print("5. formats", flush=True)
    test_formats(eps["gen"])
    print("3. no torch", flush=True)
    check("torch not loaded by episodes, packing, branches, meta_step6",
          "torch" not in sys.modules)
    print("4. running pipelines untouched", flush=True)
    after = stamps()
    diff = [k for k in set(before) | set(after)
            if before.get(k) != after.get(k)]
    check(f"{len(before) - 3} stamps and 3 code hashes unchanged", not diff,
          f"(changed {diff})")
    print(f"{'ALL OK' if not FAIL else 'FAILED: ' + ', '.join(FAIL)} "
          f"({time.time() - t0:.0f} s)")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
