#!/usr/bin/env python3
"""
Tests of the meta7 pipeline (learn/meta/episode7.py, data7.py,
studies/meta_step7.py; the D10a catalogue family). One process, numpy
only.

  1  an episode7 episode (Mission path: CatPlant + CatInjector, M10
     actuators, M15 wave inputs) equals the batched closed-loop rollout of
     the same draw (operators_cat.simulate_cat with the same operator seed,
     noise stream, sea, actuator draw and the recorded commands): states,
     the safety quantities APK / AMIN / HMIN, the impulse channel IMP, the
     slams and the extra observed signals OBS
  2  a branch from a snapshot (data7.continue_world, operator rebuilt from
     the meta) under the recorded commands equals the episode's own
     continuation; another plan runs and stays finite
  3  no torch in the workers' code: after episodes, packing and branches
     (and importing studies.meta_step7) torch is not loaded
  4  the running pipelines are untouched, against what the RUNNING jobs
     carry (read-only): (a) meta_step5.code5 / sim5 and meta_step6.code6 /
     sim6 now equal the codes in the latest header line of run5.log /
     run6.log; meta_step3.sim_code equals every meta4d .code stamp and
     sim5 every meta5 .code stamp; (b) no file those jobs import (the hard
     list: learn/meta data2, operators, relabel*, model3, episode5, data5,
     ops_m15, model_preview, operators_gen, operators_rb, episode6, data6;
     studies meta_step2/3/5/6, eval_preview_m15; sim/*, control/*) was
     modified after the start of the latest meta5 / meta6 run (the header
     time); (c) the stamps and hashes are also the same before and after
     this test (this part alone cannot fail, the test edits nothing)
  5  formats: pack7 of an episode7 episode has every pack6 array of the
     episode6 ('gen') episode of the same seed with its shape and dtype
     (and meta6_smoke/train.npz's keys, dtypes and trailing shapes when
     present); episode7 seed s has meta6's sea, actuators and scenario
     draw; the extra arrays IMP (n, T, 5), SLAM (n, T), OBS (n, T, 8) are
     float32; the meta has op_family 'm15', prior_family 'cat',
     op_seed_cat, obs_names and no op_seed / op_seed_m15 / op_seed_gen

    python studies/test_m17.py
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
SEED = 100007                      # an A-split seed (test_m16's)


def check(name, ok, msg=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {msg}", flush=True)
    if not ok:
        FAIL.append(name)


def stamps():
    out = {}
    for d in ("meta4d", "meta5", "meta6"):
        for f in sorted(glob.glob(os.path.join(ROOT, "studies", "_cache", d,
                                               "**", "*.code"),
                                  recursive=True)):
            out[os.path.relpath(f, ROOT)] = hashlib.sha1(
                open(f, "rb").read()).hexdigest()
    from studies import meta_step2 as S2
    from studies import meta_step5 as S5
    from studies import meta_step6 as S6
    out["code_hash(meta_step2)"] = S2.code_hash()
    out["code5"], out["sim5"] = S5.code5(), S5.sim5()
    out["code6"], out["sim6"] = S6.code6(), S6.sim6()
    return out


PROTECTED = (["learn/meta/" + f + ".py" for f in (
    "data2", "operators", "model3", "episode5", "data5", "ops_m15",
    "model_preview", "operators_gen", "operators_rb", "episode6", "data6")]
    + ["learn/meta/relabel*.py", "sim/**/*.py", "control/**/*.py"]
    + ["studies/" + f + ".py" for f in (
        "meta_step2", "meta_step3", "meta_step5", "meta_step6",
        "eval_preview_m15")])


def run_header(log):
    """(code, sim code, start time as a timestamp) of the latest header
    line 'HH:MM:SS ... code X, simulation code Y' of a run log; the date is
    the log's modification date (a day earlier if that gives a time after
    the modification)."""
    import datetime
    import re
    pat = re.compile(r"^(\d\d):(\d\d):(\d\d)(?: .*?)? code ([0-9a-f]{12}), "
                     r"simulation code ([0-9a-f]{12})")
    last = None
    for line in open(log, encoding="utf-8", errors="replace"):
        m = pat.match(line)
        if m:
            last = m
    if last is None:
        return None
    mt = datetime.datetime.fromtimestamp(os.path.getmtime(log))
    t = mt.replace(hour=int(last.group(1)), minute=int(last.group(2)),
                   second=int(last.group(3)), microsecond=0)
    if t > mt:
        t -= datetime.timedelta(days=1)
    return last.group(4), last.group(5), t.timestamp()


def running_jobs_check():
    """Check 4 (a), (b): list of problems (empty = fine) and a summary."""
    from studies import meta_step3 as S3
    from studies import meta_step5 as S5
    from studies import meta_step6 as S6
    bad, starts, info = [], [], []
    cache = os.path.join(ROOT, "studies", "_cache")
    for d, log, now in (("meta5", "run5.log", (S5.code5(), S5.sim5())),
                        ("meta6", "run6.log", (S6.code6(), S6.sim6()))):
        h = run_header(os.path.join(cache, d, log))
        if h is None:
            bad.append(f"{d}: no header in {log}")
            continue
        starts.append(h[2])
        info.append(f"{d} {h[0]}/{h[1]}")
        if (h[0], h[1]) != now:
            bad.append(f"{d}: running {h[0]}/{h[1]}, now {now}")
    for d, ref in (("meta4d", S3.sim_code()), ("meta5", S5.sim5())):
        for f in glob.glob(os.path.join(cache, d, "**", "*.code"),
                           recursive=True):
            if open(f).read().split("|")[0].strip() != ref:
                bad.append(f"{os.path.relpath(f, ROOT)} != {ref}")
    t0 = min(starts) if starts else 0.0
    n = 0
    for pat in PROTECTED:
        for f in glob.glob(os.path.join(ROOT, pat), recursive=True):
            n += 1
            if os.path.getmtime(f) > t0:
                bad.append(f"{os.path.relpath(f, ROOT)} modified after "
                           "the running jobs started")
    info.append(f"{n} imported files older than the start "
                f"{time.strftime('%H:%M:%S', time.localtime(t0))}")
    return bad, "; ".join(info)


def lib():
    d = np.load(os.path.join(ROOT, "studies", "_cache", "meta2", "lib.npz"))
    return dict(mu=d["mu"], sd=d["sd"], S=d["S"])


def job(**kw):
    j = dict(seed=SEED, world="low", relay=False, act_family="m10",
             family="cat", lib=lib(), T=T_EP, raw64=True, snaps=(K_SNAP,))
    j.update(kw)
    return j


# ------------------------------------------------------------------ 1
def test_batched(ep):
    from learn.meta import episode7 as E7
    from learn.meta import operators_cat as OC
    from learn.meta import operators_rb as R
    from learn.meta import relabel
    from sim import lofi
    env = relabel._env()
    n = len(ep["E"])
    dtc = env["dt"] * env["sub"]
    op = E7.make_op("cat", ep["op_seed"], lib(), dtc, env["L"],
                    act=ep["act"])
    g = np.random.default_rng([ep["op_seed"], 1])
    s28_0 = np.concatenate([ep["S"][0, :11], ep["W_MID"][0], ep["U"][0]])
    groups = [dict(op=op, st=op.new_state(1, rng=g, init_s=s28_0[None]),
                   rows=np.array([0]), rng=g)]
    seas = R.RowSeas([relabel._sea(dict(seed=ep["seed"], sea=ep["sea"]))],
                     env["x_st"], env["y_off"])
    act = {k: np.array([ep["act"][k]], float) for k in lofi.ACT_KEYS}
    out = OC.simulate_cat(env, ep["XS"][0][None], ep["U"][None],
                          np.zeros(1), seas, groups, n, act=act)
    XS = out["XS"][0]
    err = float(np.abs(XS[:n] - ep["XS"]).max())
    scale = float(np.abs(ep["XS"]).max())
    check(f"episode = batched rollout ({n} steps, M10 actuators, "
          f"{len(op.on)} items)", err <= TOL * max(scale, 1.0),
          f"(max |diff| {err:.1e}, state scale {scale:.0f}, div "
          f"{ep['div']})")
    e = max(float(np.abs(out[k][0] - ep[k]).max())
            for k in ("APK", "AMIN", "HMIN"))
    check("safety quantities = simulate_cat's (impulses included)",
          e <= TOL, f"(max |diff| {e:.1e})")
    e2 = max(float(np.abs(out["IMP"][0] - ep["IMP"]).max()),
             float(np.abs(out["OBS"][0] - ep["OBS"]).max()),
             float(np.abs(out["SLAM"][0] - ep["SLAM"]).max()),
             float(np.abs(out["P"][0] - ep["PUSH"]).max()))
    check("IMP, OBS, SLAM, PUSH = simulate_cat's", e2 <= TOL,
          f"(max |diff| {e2:.1e}; steps with impulses "
          f"{int((np.abs(ep['IMP']).sum(-1) > 0).sum())}, slams "
          f"{int(ep['SLAM'].sum())})")


# ------------------------------------------------------------------ 2
def test_branch(ep):
    from learn.meta import data7
    from learn.meta import episode7 as E7
    from learn.meta import relabel
    env = relabel._env()
    arrs, meta, snaps = E7.pack7([ep])
    m0 = meta[0]
    op = data7.rebuild_op(m0, lib(), env)
    snap = snaps[0][K_SNAP]
    H = 24
    plan = np.asarray(ep["U"][K_SNAP:K_SNAP + H], float)[None]
    XS, E0, div = data7.continue_world(op, m0, snap, plan, env)
    ref = ep["XS"][K_SNAP:K_SNAP + H + 1]
    err = float(np.abs(XS[0] - ref).max())
    check(f"branch from the snapshot at step {K_SNAP} = the episode's "
          "continuation", err <= TOL * max(np.abs(ref).max(), 1),
          f"(max |diff| {err:.1e})")
    other = plan.copy()
    other[0, :, 0] = np.clip(other[0, :, 0] + 0.2, 0, 1)
    other[0, 8:, 1] = np.clip(other[0, 8:, 1] - 0.4, -1, 1)
    XS2, E02, div2 = data7.continue_world(op, m0, snap, other, env)
    check("another plan runs, finite, differs", bool(
        np.isfinite(XS2).all() and np.isfinite(E02).all()
        and np.abs(XS2 - XS).max() > 1e-6), f"(div {div2[0]})")
    check("snapshots pickle", len(pickle.dumps(snaps)) > 0,
          f"({len(pickle.dumps(snaps)) / 1e3:.0f} kB)")


# ------------------------------------------------------------------ 5
def test_formats(ep):
    from learn.meta import episode6 as E6
    from learn.meta import episode7 as E7
    j6 = dict(seed=SEED, world="low", relay=False, act_family="m10",
              family="gen", lib=lib(), T=T_EP)
    e6 = E6.episode(j6)
    a6, m6, _ = E6.pack6([e6])
    ep32 = dict(ep)
    for k in list(ep32):
        if isinstance(ep32[k], np.ndarray) and ep32[k].dtype == np.float64:
            ep32[k] = ep32[k].astype(np.float32)
    a7, m7, _ = E7.pack7([ep32])
    bad = [k for k in a6 if k not in a7 or a7[k].dtype != a6[k].dtype
           or a7[k].shape != a6[k].shape]
    check("pack7 has every pack6 array with its shape and dtype", not bad,
          f"(mismatch {bad}; keys {sorted(a6)})")
    same_draw = (m7[0]["sea"] == m6[0]["sea"]
                 and m7[0]["act"] == m6[0]["act"]
                 and m7[0]["informative"] == m6[0]["informative"])
    check("episode7 seed s has meta6's sea, actuators and scenario draw",
          same_draw)
    n, T = a7["E"].shape[:2]
    no = len(E7.obs_names())
    ext = all(a7[k].dtype == np.float32 for k in ("IMP", "SLAM", "OBS")) \
        and a7["IMP"].shape == (n, T, 5) and a7["SLAM"].shape == (n, T) \
        and a7["OBS"].shape == (n, T, no)
    check(f"extra arrays IMP (n, T, 5), SLAM (n, T), OBS (n, T, {no}) "
          "float32", ext)
    mm = m7[0]
    check("meta: op_family 'm15', prior_family 'cat', op_seed_cat, "
          "obs_names; no op_seed / op_seed_m15 / op_seed_gen",
          mm["op_family"] == "m15" and mm["prior_family"] == "cat"
          and "op_seed_cat" in mm and "op_seed" not in mm
          and "op_seed_m15" not in mm and "op_seed_gen" not in mm
          and list(mm["obs_names"]) == list(E7.obs_names())
          and mm["style"]["null"] is False
          and "reads_waves" in mm["style"]
          and "wave_same_step" in mm["style"])
    for d in ("meta6_smoke", "meta5_smoke"):
        f = os.path.join(ROOT, "studies", "_cache", d, "train.npz")
        if os.path.exists(f):
            z = np.load(f)
            bad = [k for k in z.files if k not in a7
                   or a7[k].dtype != z[k].dtype or a7[k].ndim != z[k].ndim
                   or a7[k].shape[2:] != z[k].shape[2:]]
            check(f"keys, dtypes and trailing shapes of {d}/train.npz",
                  not bad, f"(mismatch {bad})")


def main():
    t0 = time.time()
    try:
        import psutil
        tw = time.time()
        while psutil.virtual_memory().available / 1e9 < 3.0:
            if time.time() - tw > 1800:
                print("SKIPPED: free RAM below 3 GB for 30 min")
                sys.exit(2)
            time.sleep(60)
    except ImportError:
        pass
    before = stamps()
    from studies import meta_step7  # noqa: F401  (its import chain)
    from learn.meta import episode7 as E7
    print("1/2. family cat", flush=True)
    t1 = time.time()
    ep = E7.episode(job())
    print(f"  episode of {T_EP:.0f} s: {time.time() - t1:.1f} s "
          f"({len(ep['E'])} steps, items {ep['style']['items']}, "
          f"rejections {ep['style']['n_reject']})", flush=True)
    test_batched(ep)
    test_branch(ep)
    print("5. formats", flush=True)
    test_formats(ep)
    print("3. no torch", flush=True)
    check("torch not loaded by episodes, packing, branches, meta_step7",
          "torch" not in sys.modules)
    print("4. running pipelines untouched", flush=True)
    bad, msg = running_jobs_check()
    check("current code hashes = the running jobs' codes, stamps = sim "
          "codes, no imported file modified after the jobs started",
          not bad, f"({msg}{'; ' + '; '.join(bad) if bad else ''})")
    after = stamps()
    diff = [k for k in set(before) | set(after)
            if before.get(k) != after.get(k)]
    check(f"{len(before) - 5} stamps and 5 code hashes unchanged", not diff,
          f"(changed {diff})")
    print(f"{'ALL OK' if not FAIL else 'FAILED: ' + ', '.join(FAIL)} "
          f"({time.time() - t0:.0f} s)")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
