#!/usr/bin/env python3
"""
DEFECTS M16 (code, family tag and cache keep the name m15 / meta5; brief
BRIEF_PREVIEW.md): make the future waves informative in the training
world, and train the one-step models with and without wave / preview
inputs on it. The operator family is M15 (learn/meta/ops_m15.py,
episode5.py, data5.py: the operator reads the elevations the hull meets at
mid-step, W_MID, through random spatial-pattern readings);
the actuators are the M10 family (At: target-like), so the only difference
from meta4d is the operator's waves. Cache: studies/_cache/meta5 (--smoke:
meta5_smoke).

Phases (each cached, rerun with --force):
  copy      lib.npz, G.npy and the target splits C / Cb with meta and e0
            from --copy-from (meta2; meta_step2.phase_copy: sha1 in
            copied.json, never rebuilt): the operators are calibrated on
            meta2's library, which also normalises W_MID (same
            distribution as the elevations at t)
  data      train (--size train=N, --t-train s), A, B (relays), At (A's
            seeds, target-like actuators), op family m15; chunks of 250
            episodes, resumable, each stamped <code5>-<act>-m15 (code5: the
            hash of meta_step2.SOURCES + episode5.py + ops_m15.py, the code
            the episodes run and nothing else); a chunk with failed
            episodes gets those seeds rerun on the next data run
  pack      data5.pack5: W_MID kept, meta op_family 'm15' and op_seed_m15
            (so the v0 rebuild paths of meta_step2 / meta_step3 raise on
            this data instead of rebuilding a v0 operator)
  wmid      C_wmid.npz, Cb_wmid.npz: the target splits' W_MID from their
            rebuilt seas at the dead-reckoned mid-step poses (data5.
            target_wmid; separate files, the copied C / Cb stay as copied)
  e0        the error against the MPC's model for the source splits
            (data3.e0_split; C / Cb's e0 are copied)
  branches  multi-step truths for A, B, At with the M15 operator replayed
            on W_MID and run on the simulated states' W_MID
            (data5.build_branches5; data3.build_branches' episodes,
            moments and plans)
  train_a   model3.pt: the D6 network (model3.train, unchanged)
  train_w   model_w.pt: waves at t (model_preview, variant w)
  train_p   model_p.pt: waves + preview (model_preview, variant p)
            every checkpoint records ctx, the context length its training
            split trains (model_preview.train_ctx: 256 for 90-s episodes,
            179 for 45-s ones); the evaluation caps every context at it.
            Checkpoints and the family files are written atomically.
Evaluation: studies/eval_preview_m15.py.

Files wmid / e0 / branches write carry a stamp <file>.code = the M15
simulation code (meta_step3's simulation sources + episode5.py + data5.py
+ ops_m15.py) and the pack they were made from; rebuilt when either
changed.

Guards: the cache name must start with meta5 (meta2 / meta3 / meta4d hold
the v0 family); the cache records act_family.json = 'm10-m15' (so
meta_step2, whose families are old / m10, refuses it) and op_family.json.
None of meta_step2's or meta_step3's files are edited; this driver
imports their helpers (copy, run_jobs, the pack signature) and hands
meta_step2.run_jobs the M15 episode (data5.run_job).

    python -m studies.meta_step5 --phase copy,data,pack,wmid,e0,branches \\
        --size train=10000 --procs 8
    python -m studies.meta_step5 --phase train_a,train_w,train_p \\
        --steps 40000
    python studies/eval_preview_m15.py --cache-name meta5
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from studies import meta_step2 as S2   # noqa: E402
from studies import meta_step3 as S3   # noqa: E402

CACHE = os.path.join(HERE, "_cache", "meta5")
ACT = "m10"                     # actuators of train / A / B (At: 'at')
# the code the episodes run (code5) and the rest of the M15 simulation
# code (sim5): data5.py (rebuilds, branches, target W_MID) is NOT in code5,
# so editing it never makes the data chunks stale
NEW = ("learn/meta/episode5.py", "learn/meta/ops_m15.py")
SIM_NEW = NEW + ("learn/meta/data5.py",)
SPLITS5 = ("train", "A", "B", "At")
PROCS = 4


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(CACHE, "run5.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _hash(files):
    h = hashlib.sha1()
    for f in files:
        h.update(open(os.path.join(ROOT, f), "rb").read())
    return h.hexdigest()[:12]


def code5():
    """The data's code: meta_step2.SOURCES + episode5.py + ops_m15.py."""
    return _hash(tuple(S2.SOURCES) + NEW)


def sim5():
    """The simulation code of wmid / e0 / branches."""
    return _hash(tuple(S3._sim_sources()) + SIM_NEW)


def _write_atomic(path, write, mode="w"):
    """write(fh) into path + '.tmp', then os.replace: a killed process never
    leaves a truncated file behind."""
    with open(path + ".tmp", mode) as fh:
        write(fh)
    os.replace(path + ".tmp", path)


def _save_ckpt(obj, path):
    import torch
    _write_atomic(path, lambda fh: torch.save(obj, fh), "wb")


def split_act(split):
    return S2.SPLITS[split].get("act", ACT)


def split_code(split, args):
    return f"{args.code}-{split_act(split)}-m15"


def _stamp_text(split, extra=None):
    return sim5() + "|" + S3._input_sig(split, extra)


def _fresh(path, args, split, extra=None):
    if not os.path.exists(path) or args.force:
        return False
    st = path + ".code"
    if os.path.exists(st) and open(st).read().strip() == _stamp_text(
            split, extra):
        return True
    log(f"{os.path.basename(path)}: made by other code or from another "
        "pack, rebuilt")
    return False


def _stamp(path, split, extra=None):
    _write_atomic(path + ".code", lambda f: f.write(_stamp_text(split,
                                                                 extra)))


def _lib():
    d = np.load(os.path.join(CACHE, "lib.npz"))
    return dict(mu=d["mu"], sd=d["sd"], S=d["S"])


def _has(split):
    return os.path.exists(os.path.join(CACHE, f"{split}.npz"))


# ------------------------------------------------------------ phases
def phase_copy(args):
    S2.phase_copy(args)


def phase_data(args):
    from learn.meta import data5
    lib_path = os.path.join(CACHE, "lib.npz")
    if not S2._check_copied(("lib.npz",)):
        raise SystemExit("data: lib.npz must be copied from meta2 first "
                         "(phase copy)")
    raw = os.path.join(CACHE, "raw")
    os.makedirs(raw, exist_ok=True)
    S2._run = data5.run_job          # the episode meta_step2.run_jobs runs
    for split in args.split_list:
        cfg = S2.SPLITS[split]
        if cfg["world"] != "low":
            raise SystemExit(f"data: {split} is a target split (copied)")
        n = args.sizes[split]
        code = split_code(split, args)
        n_chunks = int(np.ceil(n / S2.CHUNK))
        for ci in range(n_chunks):
            f = os.path.join(raw, f"{split}_{ci:03d}.pkl")
            lo, hi = ci * S2.CHUNK, min(n, (ci + 1) * S2.CHUNK)
            want = [cfg["seed0"] + i for i in range(lo, hi)]
            kept = []
            if os.path.exists(f) and not args.force:
                try:
                    old = pickle.load(open(f, "rb"))
                    if old["seeds"] == want and old.get("code") == code:
                        if not old.get("n_errors", 0):
                            continue
                        # rerun only the episodes that failed (a killed
                        # worker, a transient OS error)
                        kept = old["eps"]
                except Exception:
                    pass                  # stale, other code, or truncated
            have = {e["seed"] for e in kept}
            jobs = [dict(seed=cfg["seed0"] + i, world="low",
                         relay=cfg["relay"], use_lib=True, lib_path=lib_path,
                         act_family=split_act(split), op_family="m15",
                         **(dict(T=args.t_train) if split == "train" and
                            args.t_train else {}))
                    for i in range(lo, hi) if cfg["seed0"] + i not in have]
            t0 = time.time()
            eps = S2.run_jobs(jobs, lib_path, label=f"{split} chunk {ci + 1}"
                              + (f" (rerun of {len(jobs)} failed)" if kept
                                 else ""))
            bad = [e for e in eps if "error" in e]
            eps = kept + [e for e in eps if "error" not in e]
            eps.sort(key=lambda e: e["seed"])
            pickle.dump(dict(seeds=want, code=code, eps=eps,
                             n_errors=len(bad)), open(f + ".tmp", "wb"))
            os.replace(f + ".tmp", f)
            n_nf = sum(not e["finite"] for e in eps)
            log(f"data {split} chunk {ci + 1}/{n_chunks}: {len(eps)} episodes"
                f" ({len(bad)} errors, {n_nf} non-finite) in "
                f"{time.time() - t0:.0f} s"
                + (f"; first error (seed {bad[0]['seed']}) "
                   f"{bad[0]['error']}" if bad else ""))


def phase_pack(args):
    from learn.meta import data5
    for split in args.split_list:
        out = os.path.join(CACHE, f"{split}.npz")
        mp = out.replace(".npz", "_meta.pkl")
        n = args.sizes[split]
        code = split_code(split, args)
        want = [S2.SPLITS[split]["seed0"] + i for i in range(n)]
        if os.path.exists(out) and os.path.exists(mp) and not args.force:
            old = pickle.load(open(mp, "rb"))
            if old and old[0].get("code") == code \
                    and old[0].get("req") == want[:1] + want[-1:]:
                continue
        eps, n_err = [], 0
        for ci in range(int(np.ceil(n / S2.CHUNK))):
            f = os.path.join(CACHE, "raw", f"{split}_{ci:03d}.pkl")
            ch = pickle.load(open(f, "rb"))
            lo, hi = ci * S2.CHUNK, min(n, (ci + 1) * S2.CHUNK)
            if ch["seeds"] != want[lo:hi] or ch.get("code") != code:
                raise RuntimeError(f"{f}: other seeds or other code; rerun "
                                   "the data phase")
            eps += ch["eps"]
            n_err += ch.get("n_errors", 0)
        arrs, meta = data5.pack5(eps)
        for m_ in meta:
            m_["code"] = code
            m_["req"] = want[:1] + want[-1:]
        np.savez(out + ".tmp.npz", **arrs)
        os.replace(out + ".tmp.npz", out)
        pickle.dump(meta, open(mp + ".tmp", "wb"))
        os.replace(mp + ".tmp", mp)
        gone = []
        for x in ("_e0.npz", "_branches.npz"):
            for g in (out.replace(".npz", x), out.replace(".npz", x)
                      + ".code"):
                if os.path.exists(g):
                    os.remove(g)
                    gone.append(os.path.basename(g))
        log(f"pack {split}: {len(meta)} episodes of {n} requested "
            f"({n_err} errors in the chunks), arrays {arrs['S'].shape}, "
            f"W_MID {arrs['W_MID'].shape}"
            + (f"; removed {', '.join(gone)}" if gone else ""))


def phase_wmid(args):
    from learn.meta import data5, relabel
    env = relabel._env()
    for split in ("C", "Cb"):
        path = os.path.join(CACHE, f"{split}.npz")
        out = os.path.join(CACHE, f"{split}_wmid.npz")
        if _fresh(out, args, split):
            continue
        t0 = time.time()
        W = data5.target_wmid(path, env)
        np.savez(out + ".tmp.npz", W_MID=W)
        os.replace(out + ".tmp.npz", out)
        _stamp(out, split)
        log(f"wmid {split}: {W.shape} in {time.time() - t0:.0f} s")


def phase_e0(args):
    from learn.meta import data3
    for split in SPLITS5:
        if not _has(split):
            continue
        out = os.path.join(CACHE, f"{split}_e0.npz")
        if _fresh(out, args, split):
            continue
        t0 = time.time()
        data3.e0_split(os.path.join(CACHE, f"{split}.npz"))
        _stamp(out, split)
        log(f"e0 {split}: {time.time() - t0:.0f} s")


def phase_branches(args):
    from learn.meta import data5
    for split in ("A", "B", "At"):
        if not _has(split):
            continue
        out = os.path.join(CACHE, f"{split}_branches.npz")
        n_eps = 8 if args.smoke else 60
        if _fresh(out, args, split, n_eps):
            continue
        t0 = time.time()
        data5.build_branches5(os.path.join(CACHE, f"{split}.npz"), _lib(),
                              n_eps=n_eps, procs=PROCS)
        _stamp(out, split, n_eps)
        log(f"branches {split} (M15 operator): {time.time() - t0:.0f} s")


def _steps(args):
    return args.steps or (300 if args.smoke else 40000)


def _have_ckpt(args, name):
    """True (and the phase is skipped) when the checkpoint exists and
    --force is not given: rerunning the train phases after a stop trains
    only the networks that are missing; an unfinished one continues from
    its resume_<name> file (learn/meta/train_resume.py)."""
    if os.path.exists(os.path.join(CACHE, name)) and not args.force:
        log(f"{name} exists: skipped (--force retrains it)")
        return True
    return False


def phase_train_a(args):
    if _have_ckpt(args, "model3.pt"):
        return
    import torch

    from learn.meta import model3 as M
    from learn.meta import model_preview as MP
    dev = M.device()
    # time axis padded to W_CTX: model3.window_tokens fails on splits
    # shorter than its window (45-s episodes; model_preview.pad_time)
    D = MP.Data3P(CACHE, "train", dev)
    log(f"train_a (D6, model3.train): {D.n} episodes; e0 sd "
        f"{np.round(D.stats['e_sd'], 3)}")
    ctx = MP.train_ctx(D)
    log(f"  trained context {ctx} tokens (model_preview.train_ctx)")
    net = M.train(D, steps=_steps(args), log=log,
                  resume=os.path.join(CACHE, "resume_model3.pt"))
    _save_ckpt(dict(net=net.state_dict(), stats=D.stats, variant="a",
                    ctx=ctx), os.path.join(CACHE, "model3.pt"))
    log("saved model3.pt")


def _train_var(args, variant):
    if _have_ckpt(args, f"model_{variant}.pt"):
        return
    import torch

    from learn.meta import model3 as M
    from learn.meta import model_preview as MP
    dev = M.device()
    f3 = os.path.join(CACHE, "model3.pt")
    # the same error scale as the D6 model when it exists
    stats = torch.load(f3, weights_only=False)["stats"] \
        if os.path.exists(f3) else None
    D = MP.DataP(CACHE, "train", dev, stats=stats)
    ctx = MP.train_ctx(D)
    log(f"train_{variant}: {D.n} episodes, {MP.N_WX} wave columns, "
        f"stats from {'model3.pt' if stats is not None else 'the data'}, "
        f"trained context {ctx} tokens")
    net = MP.train_p(D, variant, steps=_steps(args), log=log,
                     resume=os.path.join(CACHE,
                                         f"resume_model_{variant}.pt"))
    conv = dict(P_NOROLL=MP.P_NOROLL, P_NOPREV=MP.P_NOPREV,
                HP_MAX=MP.HP_MAX, J0_MAX=MP.J0_MAX, P_LAM0=MP.P_LAM0,
                LAM_RANGE=MP.LAM_RANGE, MSD_RANGE=MP.MSD_RANGE,
                moment="uniform over the window, loss to m + J0_MAX")
    _save_ckpt(dict(net=net.state_dict(), stats=D.stats, variant=variant,
                    n_wx=MP.N_WX, fc=dict(MP.FC), conv=conv, ctx=ctx),
               os.path.join(CACHE, f"model_{variant}.pt"))
    log(f"saved model_{variant}.pt")


def phase_train_w(args):
    _train_var(args, "w")


def phase_train_p(args):
    _train_var(args, "p")


PHASES = dict(copy=phase_copy, data=phase_data, pack=phase_pack,
              wmid=phase_wmid, e0=phase_e0, branches=phase_branches,
              train_a=phase_train_a, train_w=phase_train_w,
              train_p=phase_train_p)


def main():
    global CACHE, PROCS
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="copy,data,pack,wmid,e0,branches")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--procs", type=int, default=PROCS)
    ap.add_argument("--cache-name", default="meta5")
    ap.add_argument("--copy-from", default="meta2")
    ap.add_argument("--splits", default=",".join(SPLITS5))
    ap.add_argument("--t-train", type=float, default=None,
                    help="episode length of the train split, s (default "
                         "data2.T_EP = 90)")
    ap.add_argument("--size", action="append", default=[],
                    help="override a split size, e.g. --size train=20000")
    ap.add_argument("--steps", type=int, default=None,
                    help="training steps (default 40000; smoke 300)")
    args = ap.parse_args()
    if not args.cache_name.startswith("meta5"):
        raise SystemExit("the M15 family lives in caches named meta5*; "
                         "meta2 / meta3 / meta4d hold the v0 family")
    PROCS = args.procs
    CACHE = os.path.join(HERE, "_cache", args.cache_name
                         + ("_smoke" if args.smoke else ""))
    os.makedirs(CACHE, exist_ok=True)
    # meta_step2 / meta_step3 helpers work on this cache
    S2.CACHE = S3.CACHE = CACHE
    S2.PROCS = S3.PROCS = PROCS
    fam_f = os.path.join(CACHE, "act_family.json")
    op_f = os.path.join(CACHE, "op_family.json")
    tag = f"{ACT}-m15"
    for f, want in ((fam_f, dict(act_family=tag)),
                    (op_f, dict(op_family="m15", act_family=ACT))):
        if os.path.exists(f):
            if json.load(open(f)) != want:
                raise SystemExit(f"{f} holds {json.load(open(f))}, not "
                                 f"{want}")
        else:
            _write_atomic(f, lambda fh, w=want: json.dump(w, fh))
    args.act_family = tag
    args.sizes = dict(S2.SIZES)
    if args.smoke:
        args.sizes = dict(train=40, A=10, B=10, C=4, At=10)
    for s in args.size:
        k, v = s.split("=")
        args.sizes[k] = int(v)
    args.split_list = [s for s in args.splits.split(",") if s]
    for s in args.split_list:
        if s not in SPLITS5:
            raise SystemExit(f"split {s}: one of {SPLITS5}")
    args.cache = CACHE
    args.code = code5()
    log(f"code {args.code}, simulation code {sim5()}, actuators {ACT} "
        f"(At: at), operator family m15, cache {CACHE}")
    for nm in args.phase.split(","):
        log(f"=== phase {nm}")
        PHASES[nm](args)


if __name__ == "__main__":
    main()
