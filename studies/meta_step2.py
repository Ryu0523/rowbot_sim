#!/usr/bin/env python3
"""
Step 2 of the general adaptation method's inference network: the v0
operator family (learn/meta/operators.py), the jittered random seas, and
the latent-decoder model (learn/meta/model2.py). Design and decisions:
learn/meta/PRIOR_DERIVATION.md (D1-D4); results: DEFECTS.md section M.

Phases (each cached under studies/_cache/meta2/, rerun with --force):
  copy     (meta3 only, with --copy-from) see below
  pilot    40 operator-free source episodes -> input normalisation, the
           operator library (lib.npz) and the push gain G (G.npy)
  data     train (source, v0 operators), A (source, v0, unseen draws),
           B (source, v0 + relays: memory that does not fade, never in
           training), C (target: the full planing plant, the task's sea
           state with a random direction, no operator). Chunks of 250
           episodes, resumable.
  pack     padded arrays per split (<split>.npz, <split>_meta.pkl)
  targets  relabelled rule targets and horizon inputs (data2.build_targets)
  checks   descriptive prior-predictive statistics, target vs training draws
  train    model2 stages A (decoder + codes), B+C (encoder, c-flow, nu-flow)
  eval     A, B, C

    python -m studies.meta_step2 --phase all [--n-train 5000] [--smoke]

The source actuators (DEFECTS M10; learn/meta/data2.py): --act-family old
(default, the meta2 cache) or m10 (the random actuator family, the meta3
cache). The family goes into every job (pilot and data) and into the code
string every chunk and pack carries, so files of the other family are
regenerated or refused, never mixed; a cache remembers its family
(act_family.json) and refuses the other one; meta2 refuses m10 and meta3
refuses old. Split At (test only, never trained on): A's seeds with fixed
target-shaped actuators (lofi.act_target_like), for M10's reading of a
remaining failure (PRIOR_DERIVATION.md D7).

The meta3 cache: the target splits and the operator library do not depend
on the source actuators, so they are COPIED from meta2 (phase copy, a
marker copied.json; data and pack skip the copied splits), not remade:
lib.npz (the operators are calibrated on it: a new pilot would change the
operator family too), G.npy, C / Cb with their meta and e0. Then

    python -m studies.meta_step2 --cache-name meta3 --act-family m10 \\
        --copy-from meta2 --phase copy,pilot,data,pack --splits train,A,B,At
    python -m studies.meta_step3 --cache-name meta3 --phase e0,branches,...

(targets / checks / train / eval of step 2 are not needed by step 3.)
With --smoke both cache names get _smoke. --procs 1 runs the episodes in
this process; the default is 4 worker processes (the 16 GB laptop's cap).
A copy never replaces a differing file without --force, and pilot never
rebuilds a copied lib.npz / G.npy (copied.json records their sha1) and
refuses to build a library from m10 episodes. A pack rewrite removes the
files computed from the old pack (targets, step 3's e0 / branches /
tbranches). Old-family chunks and packs made by the pre-M10 code
(LEGACY_OLD) count as current, so rerunning step 2 on meta2 changes
nothing there.
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache",
                     "meta2")
SPLITS = dict(train=dict(seed0=1, world="low", relay=False),
              A=dict(seed0=100000, world="low", relay=False),
              B=dict(seed0=200000, world="low", relay=True),
              C=dict(seed0=300000, world="high", relay=False),
              # A's episodes with fixed target-shaped actuators (M10 test)
              At=dict(seed0=100000, world="low", relay=False, act="at"))
SIZES = dict(train=5000, A=300, B=300, C=64, At=300)
FAMILIES = ("old", "m10")
# files the meta3 cache copies from meta2 (phase copy)
COPY_FILES = ("lib.npz", "G.npy", "C.npz", "C_meta.pkl", "C_e0.npz",
              "Cb.npz", "Cb_meta.pkl", "Cb_e0.npz")
COPY_SPLITS = ("C", "Cb")
# the code that makes the data: every chunk, pack and target file carries
# its hash, and files made by other code are regenerated or refused
SOURCES = ("learn/meta/operators.py", "learn/meta/data2.py",
           "learn/meta/residuals.py", "learn/repro/task.py", "sim/lofi.py",
           "sim/env.py", "sim/planing_vessel.py", "step0_preview_spec.py")
# SOURCES' hash before M10. M10 edited data2.py and lofi.py, so the hash
# changed, but their old-family path is bit-identical (studies/test_m10.py
# test 1). Old-family files carrying this code are accepted as current
# (code_ok), so a step-2 rerun on meta2 neither regenerates its ~5.6k
# episodes nor refuses its packs (nor deletes the targets model2 needs).
LEGACY_OLD = ("dd91c9b16c09",)


def code_hash():
    import hashlib
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha1()
    for f in SOURCES:
        h.update(open(os.path.join(root, f), "rb").read())
    return h.hexdigest()[:12]
CHUNK = 250
PROCS = 4                # worker processes: at most 4 on the 16 GB laptop


def log(msg, f="run.log"):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(CACHE, f), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


_LIB = {}


def _init(lib_path):
    if lib_path and os.path.exists(lib_path):
        d = np.load(lib_path)
        _LIB.update(mu=d["mu"], sd=d["sd"], S=d["S"])


def _run(job):
    from learn.meta import data2
    j = dict(job)
    if j.pop("use_lib", False):
        j["lib"] = dict(_LIB)
    try:
        return data2.episode(j)
    except Exception as ex:                       # keep the chunk going
        import traceback
        return dict(error=repr(ex) + " | " + traceback.format_exc()[-600:],
                    seed=job["seed"])


def _pool(lib_path, procs=None):
    from multiprocessing import Pool
    return Pool(procs or PROCS, initializer=_init, initargs=(lib_path,))


STALL_S = 600           # no episode back for this long: the pool is dead


def run_jobs(jobs, lib_path, label=""):
    """Run episode jobs in worker processes and return the results.

    Survives dead workers. A multiprocessing.Pool whose worker is killed
    (the memory watchdog, the OS) waits forever for the lost result: that
    hung the 2026-09-29 run for four hours. Here a broken pool, or one that
    returns nothing for STALL_S seconds, is torn down, and the jobs not yet
    returned go to a fresh pool (at most 5 restarts)."""
    from concurrent.futures import (FIRST_COMPLETED, ProcessPoolExecutor,
                                    wait)
    from concurrent.futures.process import BrokenProcessPool
    todo = {j["seed"]: j for j in jobs}
    out, restarts, t0 = [], 0, time.time()
    if PROCS <= 1:                  # in this process (one-process machines)
        _init(lib_path)
        for j in jobs:
            out.append(_run(j))
            if len(out) % 10 == 0:
                log(f"    {label}: {len(out)}/{len(jobs)} episodes "
                    f"({time.time() - t0:.0f} s)")
        return out
    while todo and restarts <= 5:
        ex = ProcessPoolExecutor(PROCS, initializer=_init,
                                 initargs=(lib_path,))
        futs = {ex.submit(_run, j): s for s, j in todo.items()}
        broken = False
        try:
            pending = set(futs)
            while pending:
                done, pending = wait(pending, timeout=STALL_S,
                                     return_when=FIRST_COMPLETED)
                if not done:
                    raise TimeoutError("no episode returned in "
                                       f"{STALL_S} s")
                for fu in done:
                    r = fu.result()
                    out.append(r)
                    todo.pop(futs[fu], None)
                    if len(out) % 50 == 0:
                        log(f"    {label}: {len(out)}/{len(jobs)} episodes "
                            f"({time.time() - t0:.0f} s)")
        except (BrokenProcessPool, TimeoutError) as e:
            broken = True
            restarts += 1
            log(f"    {label}: pool lost ({type(e).__name__}: {e}); "
                f"{len(todo)} jobs left, restart {restarts}")
        finally:
            if broken:
                for pr in list(getattr(ex, "_processes", {}).values()):
                    try:
                        pr.kill()
                    except Exception:
                        pass
                ex.shutdown(wait=False, cancel_futures=True)
            else:
                ex.shutdown(wait=True)
    for s in todo:
        out.append(dict(error="never returned (pool lost 6 times)", seed=s))
    return out


def phase_pilot(args):
    """Builds lib.npz and G.npy, except where they were copied (copied.json:
    never rebuilt, even under --force, and checked against the recorded
    sha1). A non-old family without a copied library refuses: a library
    from m10 episodes would change the operator family (D7)."""
    from learn.meta import data2
    lib_path = os.path.join(CACHE, "lib.npz")
    cp = _check_copied(("lib.npz", "G.npy"))
    if cp:
        log(f"pilot: {', '.join(cp)} copied from {_copied()['source']}, "
            "kept")
        if len(cp) == 2:
            return
        raise SystemExit("pilot: copied.json lists only " + ", ".join(cp)
                         + "; rerun the copy phase")
    if args.act_family != "old":
        raise SystemExit(f"pilot: family {args.act_family} needs the library "
                         "copied from the old-family cache (phase copy with "
                         "--copy-from meta2) before pilot")
    if os.path.exists(lib_path) and not args.force:
        log("pilot: cached")
        return
    n = 8 if args.smoke else 40
    jobs = [dict(seed=700000 + i, world="low", informative=True,
                 act_family=args.act_family) for i in range(n)]
    t0 = time.time()
    if PROCS <= 1:
        eps = [e for e in map(_run, jobs) if "error" not in e]
    else:
        with _pool(None) as pool:
            eps = [e for e in pool.map(_run, jobs) if "error" not in e]
    lib = data2.build_library(eps, n_seq=min(16, len(eps)))
    np.savez(lib_path, **lib)
    G, Gs = data2.push_gain(n_states=6 if args.smoke else 24)
    np.save(os.path.join(CACHE, "G.npy"), G)
    log(f"pilot: {len(eps)} episodes in {time.time() - t0:.0f} s; "
        f"library {lib['S'].shape}")
    log("push gain G0 (rows: e channel, cols: push channel), G1 (slope "
        "in u / u_ref - 1), rms misfit:\n"
        + np.array2string(G, precision=3, suppress_small=True) + "\n"
        + np.array2string(Gs, precision=3, suppress_small=True))


def split_family(split, args):
    """The actuator family of a split's jobs: At's own, the option's for
    the other source splits, 'old' (no actuator draw) for the target."""
    cfg = SPLITS[split]
    if cfg["world"] != "low":
        return "old"
    return cfg.get("act", args.act_family)


def split_code(split, args):
    """The code string a split's chunks and pack carry: the code hash, plus
    the actuator family unless it is the old one (so meta2's strings keep
    their form)."""
    fam = split_family(split, args)
    return args.code if fam == "old" else f"{args.code}-{fam}"


def code_ok(stored, split, args):
    """A chunk's / pack's stored code string is current for this split: the
    split's code, or (old family only) a pre-M10 code (LEGACY_OLD)."""
    return stored == split_code(split, args) or (
        split_family(split, args) == "old" and stored in LEGACY_OLD)


def _copied():
    f = os.path.join(CACHE, "copied.json")
    return json.load(open(f)) if os.path.exists(f) else {}


def _sha1(path):
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def phase_copy(args):
    """Copy the family-independent files (COPY_FILES: the operator library,
    G, the target splits C and Cb with meta and e0) from the cache
    --copy-from into this one and record them, with a sha1 each, in
    copied.json; data and pack then skip C (and step 3's cblocks finds Cb).
    A file already here must be byte-identical to the source, else the copy
    refuses (--force copies it again; the source is only read), so
    copied.json never claims a file came from the source when it did not."""
    import shutil
    if not args.copy_from:
        log("copy: no --copy-from, nothing to do")
        return
    src = os.path.join(os.path.dirname(CACHE), args.copy_from
                       + ("_smoke" if args.smoke else ""))
    if os.path.abspath(src) == os.path.abspath(CACHE):
        raise SystemExit("copy: --copy-from is this cache")
    done, same, sums = [], [], {}
    for f in COPY_FILES:
        a, b = os.path.join(src, f), os.path.join(CACHE, f)
        if not os.path.exists(a):
            raise SystemExit(f"copy: {a} missing")
        ha = _sha1(a)
        if os.path.exists(b) and _sha1(b) == ha:
            same.append(f)
        elif os.path.exists(b) and not args.force:
            raise SystemExit(f"copy: {b} differs from {a} (made here, e.g. "
                             "by a pilot run before the copy?); --force "
                             "copies the source over it")
        else:
            shutil.copy2(a, b + ".tmp")
            os.replace(b + ".tmp", b)
            done.append(f)
        sums[f] = ha
    json.dump(dict(source=src, splits=list(COPY_SPLITS),
                   files=list(COPY_FILES), sha1=sums),
              open(os.path.join(CACHE, "copied.json"), "w"), indent=1)
    log(f"copy from {src}: {len(done)} files copied ({', '.join(done)}), "
        f"{len(same)} already here and identical")


def _check_copied(names):
    """For each of names listed in copied.json: still the recorded file
    (sha1). Returns the names that are copied; refuses on a mismatch."""
    cp = _copied()
    got = []
    for f in names:
        if f in cp.get("files", []):
            want = cp.get("sha1", {}).get(f)
            path = os.path.join(CACHE, f)
            if not os.path.exists(path) or (want and _sha1(path) != want):
                raise SystemExit(f"{path} is not the file copied from "
                                 f"{cp['source']} (copied.json); rerun the "
                                 "copy phase with --force")
            got.append(f)
    return got


def phase_data(args):
    lib_path = os.path.join(CACHE, "lib.npz")
    raw = os.path.join(CACHE, "raw")
    os.makedirs(raw, exist_ok=True)
    copied = _copied().get("splits", [])
    for split, cfg in SPLITS.items():
        if split not in args.split_list:
            continue
        if split in copied:
            log(f"data {split}: copied from {_copied()['source']}, skipped")
            continue
        n = args.sizes[split]
        code = split_code(split, args)
        n_chunks = int(np.ceil(n / CHUNK))
        for ci in range(n_chunks):
            f = os.path.join(raw, f"{split}_{ci:03d}.pkl")
            lo, hi = ci * CHUNK, min(n, (ci + 1) * CHUNK)
            want = [cfg["seed0"] + i for i in range(lo, hi)]
            if os.path.exists(f) and not args.force:
                try:
                    old = pickle.load(open(f, "rb"))
                    if old["seeds"] == want and code_ok(old.get("code"),
                                                        split, args):
                        continue
                except Exception:
                    pass                  # stale, other code, or truncated
            jobs = [dict(seed=cfg["seed0"] + i, world=cfg["world"],
                         relay=cfg["relay"], use_lib=cfg["world"] == "low",
                         act_family=split_family(split, args),
                         **(dict(T=args.t_train) if split == "train" and
                            args.t_train else {}))
                    for i in range(lo, hi)]
            t0 = time.time()
            eps = run_jobs(jobs, lib_path, label=f"{split} chunk {ci + 1}")
            bad = [e for e in eps if "error" in e]
            eps = [e for e in eps if "error" not in e]
            eps.sort(key=lambda e: e["seed"])
            pickle.dump(dict(seeds=want, code=code, eps=eps),
                        open(f + ".tmp", "wb"))
            os.replace(f + ".tmp", f)
            n_nf = sum(not e["finite"] for e in eps)
            log(f"data {split} chunk {ci + 1}/{n_chunks}: {len(eps)} episodes"
                f" ({len(bad)} errors, {n_nf} non-finite) in "
                f"{time.time() - t0:.0f} s"
                + (f"; first error (seed {bad[0]['seed']}) "
                   f"{bad[0]['error']}" if bad else ""))


def phase_pack(args):
    from learn.meta import data2
    copied = _copied().get("splits", [])
    for split in args.split_list:
        if split in copied:
            continue
        out = os.path.join(CACHE, f"{split}.npz")
        n = args.sizes[split]
        code = split_code(split, args)
        want = [SPLITS[split]["seed0"] + i for i in range(n)]
        mp = out.replace(".npz", "_meta.pkl")
        if os.path.exists(out) and os.path.exists(mp) and not args.force:
            old = pickle.load(open(mp, "rb"))
            if old and code_ok(old[0].get("code"), split, args) \
                    and old[0].get("req") == want[:1] + want[-1:]:
                continue
        n_chunks = int(np.ceil(n / CHUNK))
        eps = []
        for ci in range(n_chunks):
            f = os.path.join(CACHE, "raw", f"{split}_{ci:03d}.pkl")
            ch = pickle.load(open(f, "rb"))
            lo, hi = ci * CHUNK, min(n, (ci + 1) * CHUNK)
            if ch["seeds"] != want[lo:hi] or not code_ok(ch.get("code"),
                                                         split, args):
                raise RuntimeError(f"{f}: other seeds or other code; rerun "
                                   "the data phase")
            eps += ch["eps"]
        arrs, meta = data2.pack(eps)
        for m_ in meta:
            m_["code"] = code
            m_["req"] = want[:1] + want[-1:]
        np.savez(out + ".tmp.npz", **arrs)
        os.replace(out + ".tmp.npz", out)
        pickle.dump(meta, open(out.replace(".npz", "_meta.pkl"), "wb"))
        # files computed from the old pack follow it: step 2's targets and
        # step 3's e0 / branches (and, for train, the tbranches)
        dep = ["_targets.npz", "_e0.npz", "_branches.npz", "_wbranches.npz"]
        dep = [out.replace(".npz", x) for x in dep]
        if split == "train":
            dep += [os.path.join(CACHE, f"train_{x}branches.npz")
                    for x in ("t", "v")]
        gone = []
        for f in dep:
            for g in (f, f + ".code"):
                if os.path.exists(g):
                    os.remove(g)
                    gone.append(os.path.basename(g))
        if gone:
            log(f"pack {split}: removed the files made from the old pack: "
                + ", ".join(gone))
        log(f"pack {split}: {len(meta)} episodes, arrays "
            f"{arrs['S'].shape}")


def phase_targets(args):
    from learn.meta import data2
    d = np.load(os.path.join(CACHE, "lib.npz"))
    lib = dict(mu=d["mu"], sd=d["sd"], S=d["S"])
    G = np.load(os.path.join(CACHE, "G.npy"))
    for split in args.split_list:
        path = os.path.join(CACHE, f"{split}.npz")
        out = path.replace(".npz", "_targets.npz")
        # the targets carry the code of the pack they describe (a source
        # split's '<hash>-m10', a copied C's meta2 hash), which model2.Data
        # compares with the pack's
        meta = pickle.load(open(path.replace(".npz", "_meta.pkl"), "rb"))
        code = meta[0].get("code", "") if meta else split_code(split, args)
        if os.path.exists(out) and not args.force:
            try:
                if str(np.load(out)["code"]) == code:
                    continue
            except Exception:
                pass
            log(f"targets {split}: made for another pack, rebuilt")
        t0 = time.time()
        # closed-loop relabelling (learn/meta/relabel.py, DEFECTS M3); the
        # open-loop data2.build_targets is kept only for comparison
        from learn.meta import relabel
        relabel.build_targets(path, lib, procs=PROCS, code=code)
        log(f"targets {split} (closed loop): {time.time() - t0:.0f} s")


def phase_checks(args):
    from learn.meta import checks
    rep = checks.report(CACHE)
    log("checks:\n" + rep, f="checks.log")


def phase_train(args):
    from learn.meta import model2
    model2.train_all(CACHE, args, log=log)


def phase_eval(args):
    from learn.meta import model2
    model2.eval_all(CACHE, args, log=log)


PHASES = dict(copy=phase_copy, pilot=phase_pilot, data=phase_data,
              pack=phase_pack,
              targets=phase_targets, checks=phase_checks, train=phase_train,
              eval=phase_eval)


def main():
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--procs", type=int, default=PROCS)
    ap.add_argument("--waves", default="off", choices=("off", "on"),
                    help="off (main): the model sees only the boat's states, "
                    "the commands and the error history; on: also the 15 "
                    "wave elevations (comparison)")
    ap.add_argument("--kind", default="film", choices=("film", "linear"))
    ap.add_argument("--steps-a", type=int, default=10000,
                    help="stage A (decoder + codes) training steps")
    ap.add_argument("--steps-b", type=int, default=15000,
                    help="stage B (encoder + flows) training steps")
    ap.add_argument("--cache-name", default="meta2",
                    help="cache folder under studies/_cache (a separate one "
                    "for a preliminary run on part of the data)")
    ap.add_argument("--splits", default="train,A,B,C",
                    help="splits the data / pack / targets phases work on")
    ap.add_argument("--t-train", type=float, default=None,
                    help="episode length of the train split, s (default "
                         "data2.T_EP; more, shorter episodes = more distinct "
                         "operators for the same steps, DEFECTS M14)")
    ap.add_argument("--size", action="append", default=[],
                    help="override a split size, e.g. --size A=100")
    ap.add_argument("--act-family", default="old", choices=FAMILIES,
                    help="source actuators (DEFECTS M10): old = lofi.PRIOR's "
                    "lag / rate w.p. 0.75 (meta2); m10 = the random actuator "
                    "family lofi.ACT_FAMILY (meta3)")
    ap.add_argument("--copy-from", default=None,
                    help="phase copy: the cache name to copy the target "
                    "splits and the library from (e.g. meta2)")
    args = ap.parse_args()
    globals()["PROCS"] = args.procs
    CACHE = os.path.join(os.path.dirname(CACHE), args.cache_name)
    if args.smoke:
        CACHE = CACHE + "_smoke"
    base = args.cache_name
    if base == "meta2" and args.act_family != "old":
        raise SystemExit("meta2 holds the old actuator family; use another "
                         "--cache-name (meta3) for --act-family "
                         f"{args.act_family}")
    if base.startswith("meta3") and args.act_family == "old":
        raise SystemExit("meta3 holds the M10 actuator family: pass "
                         "--act-family m10")
    os.makedirs(CACHE, exist_ok=True)
    fam_f = os.path.join(CACHE, "act_family.json")
    if os.path.exists(fam_f):
        had = json.load(open(fam_f))["act_family"]
        if had != args.act_family:
            raise SystemExit(f"{CACHE} holds actuator family {had!r}, not "
                             f"{args.act_family!r}")
    elif args.act_family != "old":
        json.dump(dict(act_family=args.act_family), open(fam_f, "w"))
    args.sizes = dict(SIZES)
    if args.smoke:
        args.sizes = dict(train=40, A=10, B=10, C=4, At=10)
    if args.n_train:
        args.sizes["train"] = args.n_train
    for s in args.size:
        k, v = s.split("=")
        args.sizes[k] = int(v)
    args.split_list = args.splits.split(",")
    args.cache = CACHE
    args.code = code_hash()
    log(f"code {args.code}, actuator family {args.act_family}")
    names = list(PHASES) if args.phase == "all" else args.phase.split(",")
    for nm in names:
        log(f"=== phase {nm}")
        PHASES[nm](args)


if __name__ == "__main__":
    main()
