#!/usr/bin/env python3
"""
M16 pipeline (cache meta6): the one-step error models trained on the
rigid-body error families instead of the M15 operators. Two families by
--family (learn/meta/episode6.py):
  gen  the general family, a rigid body acted on by ARBITRARY causal forces
       (learn/meta/operators_gen.py, PRIOR_DERIVATION.md D9.8): the main
       line; cache meta6 (--smoke: meta6_smoke)
  rb   the D8 rigid-body family (learn/meta/operators_rb.py, D8.12): the
       comparison arm; cache meta6_rb (--smoke: meta6_rb_smoke)
Everything else is meta_step5's M15 pipeline: the M10 actuators (At:
target-like), the M15 wave inputs (W_MID, the elevations met at mid-step;
S keeps the elevations at t; the preview = W_MID), the same scenario,
splits, seeds and training. Evaluation: studies/eval_preview_m15.py
--cache-name <cache> (unchanged).

Phases (each cached, rerun with --force):
  copy      lib.npz, G.npy and the target splits C / Cb with meta and e0
            from --copy-from (meta2; meta_step2.phase_copy, sha1 in
            copied.json)
  data      train (--size train=N, --t-train s), A, B (relays), At
            (target-like actuators); chunks of 250 episodes, resumable,
            each stamped <code6>-<act>-<family> (code6: the hash of
            meta_step2.SOURCES + EP_SOURCES, the code the episodes run);
            a chunk with failed episodes gets those seeds rerun on the next
            data run. A / B / At episodes keep snapshots at the branch
            moments (SNAPS)
  pack      episode6.pack6: W_MID, APK / AMIN / HMIN, PUSH; meta op_family
            'm15' (the wave-input format), prior_family, op_seed_<family>
            (so every rebuild path of meta_step2 / 3 / 5 raises on this
            data); <split>_snaps.pkl. The pack carries the size and mtime
            of every chunk it read ('chunks'), so a chunk rewritten by a
            later data run (failed seeds rerun) makes the pack stale. Logs
            the acceptance-test rejections per reason ('gen')
  wmid      C_wmid.npz, Cb_wmid.npz (data5.target_wmid, unchanged)
  e0        the error against the MPC's model (data3.e0_split)
  branches  multi-step truths for A, B, At: closed-loop continuations
            from the episodes' snapshots (data6.build_branches6; same
            episodes, moments and plans as meta5's)
  train_a   model3.pt (model3.train, unchanged)
  train_w   model_w.pt (model_preview, variant w)
  train_p   model_p.pt (model_preview, variant p)
Stamps <file>.code on wmid / e0 / branches: sim6 (meta_step3's simulation
sources + the episode sources + data6.py + data5.py) and the pack. All
files are written atomically (temp file + os.replace).

Memory: at most 8 worker processes (--procs; use 2 while other data jobs run); the data
and branch workers import no torch. Training uses the GPU only when
nvidia-smi reports at least 2 GB free (else CUDA is hidden: CPU).

    python -m studies.meta_step6 --family gen \\
        --phase copy,data,pack,wmid,e0,branches --size train=10000 --procs 2
    python -m studies.meta_step6 --family gen \\
        --phase train_a,train_w,train_p --steps 40000
    python studies/eval_preview_m15.py --cache-name meta6
"""
import argparse
import collections
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from studies import meta_step2 as S2   # noqa: E402
from studies import meta_step3 as S3   # noqa: E402

CACHE = os.path.join(HERE, "_cache", "meta6")
ACT = "m10"
FAMILIES = ("gen", "rb")
DEFAULT_CACHE = dict(gen="meta6", rb="meta6_rb")
# the code the episodes run (code6) and the rest of the simulation code
# (the plant model control/reduced.py, the sea sim/wavefield.py and
# sim/config.py are run by the episodes and by operators_gen's input library
# and acceptance test, so they belong to the episodes' code too)
EP_SOURCES = ("learn/meta/episode6.py", "learn/meta/episode5.py",
              "learn/meta/ops_m15.py", "learn/meta/operators_rb.py",
              "learn/meta/operators_gen.py", "control/reduced.py",
              "sim/wavefield.py", "sim/config.py")
SIM_SOURCES = EP_SOURCES + ("learn/meta/data6.py", "learn/meta/data5.py")
SPLITS6 = ("train", "A", "B", "At")
SNAPS = (120, 240)              # branch moments (data6 / data5 defaults)
SNAP_SPLITS = ("A", "B", "At")
PROCS = 2
MAX_PROCS = 8  # 2 while other data pipelines run; up to 8 alone (meta5 ran 8)


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(CACHE, "run6.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _hash(files):
    h = hashlib.sha1()
    for f in files:
        h.update(open(os.path.join(ROOT, f), "rb").read())
    return h.hexdigest()[:12]


def code6():
    """The data's code: meta_step2.SOURCES + the episode sources."""
    return _hash(tuple(S2.SOURCES) + EP_SOURCES)


def sim6():
    """The simulation code of wmid / e0 / branches."""
    return _hash(tuple(S3._sim_sources()) + SIM_SOURCES)


def _chunks_sig(split, n):
    """(size, mtime_ns) of every raw chunk of a split: changes whenever a
    data run rewrites a chunk (e.g. failed seeds rerun)."""
    out = []
    for ci in range(int(np.ceil(n / S2.CHUNK))):
        f = os.path.join(CACHE, "raw", f"{split}_{ci:03d}.pkl")
        st = os.stat(f) if os.path.exists(f) else None
        out.append((st.st_size, st.st_mtime_ns) if st else None)
    return tuple(out)


def _write_atomic(path, write, mode="w"):
    with open(path + ".tmp", mode) as fh:
        write(fh)
    os.replace(path + ".tmp", path)


def _save_ckpt(obj, path):
    import torch
    _write_atomic(path, lambda fh: torch.save(obj, fh), "wb")


def split_act(split):
    return S2.SPLITS[split].get("act", ACT)


def split_code(split, args):
    return f"{args.code}-{split_act(split)}-{args.family}"


def _stamp_text(split, extra=None):
    return sim6() + "|" + S3._input_sig(split, extra)


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


def gpu_ok(min_mib=2048):
    """nvidia-smi reports at least min_mib free on the first GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0]) >= min_mib
    except Exception:
        return False


# ------------------------------------------------------------ phases
def phase_copy(args):
    S2.phase_copy(args)


def phase_data(args):
    from learn.meta import episode6
    lib_path = os.path.join(CACHE, "lib.npz")
    if not S2._check_copied(("lib.npz",)):
        raise SystemExit("data: lib.npz must be copied from meta2 first "
                         "(phase copy)")
    raw = os.path.join(CACHE, "raw")
    os.makedirs(raw, exist_ok=True)
    S2._run = episode6.run_job          # the episode meta_step2.run_jobs runs
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
                        kept = old["eps"]     # rerun only the failed ones
                except Exception:
                    pass                  # stale, other code, or truncated
            have = {e["seed"] for e in kept}
            jobs = [dict(seed=cfg["seed0"] + i, world="low",
                         relay=cfg["relay"], use_lib=True, lib_path=lib_path,
                         act_family=split_act(split), family=args.family,
                         snaps=SNAPS if split in SNAP_SPLITS else (),
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
            _write_atomic(f, lambda fh: pickle.dump(
                dict(seeds=want, code=code, eps=eps, n_errors=len(bad)), fh),
                "wb")
            n_nf = sum(not e["finite"] for e in eps)
            n_div = sum(e["div"] >= 0 for e in eps)
            dt_ep = (time.time() - t0) / max(len(jobs), 1)
            log(f"data {split} chunk {ci + 1}/{n_chunks}: {len(eps)} episodes"
                f" ({len(bad)} errors, {n_nf} non-finite, {n_div} diverged) "
                f"in {time.time() - t0:.0f} s ({dt_ep:.1f} s per episode "
                f"with {PROCS} procs)"
                + (f"; first error (seed {bad[0]['seed']}) "
                   f"{bad[0]['error']}" if bad else ""))


def phase_pack(args):
    from learn.meta import episode6
    for split in args.split_list:
        out = os.path.join(CACHE, f"{split}.npz")
        mp = out.replace(".npz", "_meta.pkl")
        sp = out.replace(".npz", "_snaps.pkl")
        n = args.sizes[split]
        code = split_code(split, args)
        want = [S2.SPLITS[split]["seed0"] + i for i in range(n)]
        csig = _chunks_sig(split, n)
        if all(os.path.exists(x) for x in (out, mp, sp)) and not args.force:
            old = pickle.load(open(mp, "rb"))
            if old and old[0].get("code") == code \
                    and old[0].get("req") == want[:1] + want[-1:] \
                    and old[0].get("chunks") == csig:
                continue
            if old and old[0].get("chunks") != csig:
                log(f"pack {split}: chunks changed since the last pack, "
                    "rebuilt")
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
        arrs, meta, snaps = episode6.pack6(eps)
        del eps
        for m_ in meta:
            m_["code"] = code
            m_["req"] = want[:1] + want[-1:]
            m_["chunks"] = csig
        np.savez(out + ".tmp.npz", **arrs)
        os.replace(out + ".tmp.npz", out)
        _write_atomic(mp, lambda fh: pickle.dump(meta, fh), "wb")
        _write_atomic(sp, lambda fh: pickle.dump(snaps, fh), "wb")
        gone = []
        for x in ("_e0.npz", "_branches.npz"):
            for g in (out.replace(".npz", x), out.replace(".npz", x)
                      + ".code"):
                if os.path.exists(g):
                    os.remove(g)
                    gone.append(os.path.basename(g))
        n_div = sum(m_["div"] >= 0 for m_ in meta)
        log(f"pack {split}: {len(meta)} episodes of {n} requested "
            f"({n_err} errors in the chunks, {n_div} truncated by a "
            f"divergence), arrays {arrs['S'].shape}, W_MID "
            f"{arrs['W_MID'].shape}, snapshots of {len(snaps)} episodes"
            + (f"; removed {', '.join(gone)}" if gone else ""))
        if args.family == "gen":
            st = [m_["style"] for m_ in meta]
            why = collections.Counter(r for s_ in st
                                      for r in s_.get("reject_reasons", ()))
            n_rej = sum(s_.get("n_reject", 0) > 0 for s_ in st)
            n_off = sum(bool(s_.get("forces_off")) for s_ in st)
            n_nof = sum(bool(s_.get("no_forces")) for s_ in st)
            log(f"  acceptance ({split}): {n_rej} of {len(st)} draws "
                f"rejected at least once; rejections per draw by reason: "
                + (", ".join(f"{k} {v / max(len(st), 1):.3f}"
                             for k, v in why.items()) or "none")
                + f"; forces_off {n_off}, no_forces {n_nof}")


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
    for split in SPLITS6:
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
    from learn.meta import data6
    for split in SNAP_SPLITS:
        if not _has(split):
            continue
        out = os.path.join(CACHE, f"{split}_branches.npz")
        n_eps = 8 if args.smoke else 60
        if _fresh(out, args, split, n_eps):
            continue
        t0 = time.time()
        data6.build_branches6(os.path.join(CACHE, f"{split}.npz"), _lib(),
                              n_eps=n_eps, moments=SNAPS, procs=PROCS)
        b = np.load(out)
        _stamp(out, split, n_eps)
        log(f"branches {split} ({args.family}, from snapshots): "
            f"{b['U'].shape[0]} moments x {b['U'].shape[1]} plans kept, "
            f"{int(b['n_dropped'])} moments dropped (a plan left the "
            f"envelope), {time.time() - t0:.0f} s")


def _steps(args):
    return args.steps or (300 if args.smoke else 40000)


def phase_train_a(args):
    from learn.meta import model3 as M
    from learn.meta import model_preview as MP
    dev = M.device()
    D = MP.Data3P(CACHE, "train", dev)
    log(f"train_a (D6, model3.train) on {dev}: {D.n} episodes; e0 sd "
        f"{np.round(D.stats['e_sd'], 3)}")
    ctx = MP.train_ctx(D)
    log(f"  trained context {ctx} tokens (model_preview.train_ctx)")
    net = M.train(D, steps=_steps(args), log=log)
    _save_ckpt(dict(net=net.state_dict(), stats=D.stats, variant="a",
                    ctx=ctx, prior_family=args.family),
               os.path.join(CACHE, "model3.pt"))
    log("saved model3.pt")


def _train_var(args, variant):
    import torch

    from learn.meta import model3 as M
    from learn.meta import model_preview as MP
    dev = M.device()
    f3 = os.path.join(CACHE, "model3.pt")
    stats = torch.load(f3, weights_only=False)["stats"] \
        if os.path.exists(f3) else None
    D = MP.DataP(CACHE, "train", dev, stats=stats)
    ctx = MP.train_ctx(D)
    log(f"train_{variant} on {dev}: {D.n} episodes, {MP.N_WX} wave columns, "
        f"stats from {'model3.pt' if stats is not None else 'the data'}, "
        f"trained context {ctx} tokens")
    net = MP.train_p(D, variant, steps=_steps(args), log=log)
    conv = dict(P_NOROLL=MP.P_NOROLL, P_NOPREV=MP.P_NOPREV,
                HP_MAX=MP.HP_MAX, J0_MAX=MP.J0_MAX, P_LAM0=MP.P_LAM0,
                LAM_RANGE=MP.LAM_RANGE, MSD_RANGE=MP.MSD_RANGE,
                moment="uniform over the window, loss to m + J0_MAX")
    _save_ckpt(dict(net=net.state_dict(), stats=D.stats, variant=variant,
                    n_wx=MP.N_WX, fc=dict(MP.FC), conv=conv, ctx=ctx,
                    prior_family=args.family),
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
TRAIN_PHASES = ("train_a", "train_w", "train_p")


def main():
    global CACHE, PROCS
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--phase", default="copy,data,pack,wmid,e0,branches")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--procs", type=int, default=PROCS)
    ap.add_argument("--cache-name", default=None,
                    help="default meta6 (gen) / meta6_rb (rb)")
    ap.add_argument("--copy-from", default="meta2")
    ap.add_argument("--splits", default=",".join(SPLITS6))
    ap.add_argument("--t-train", type=float, default=None,
                    help="episode length of the train split, s (default "
                         "data2.T_EP = 90)")
    ap.add_argument("--size", action="append", default=[],
                    help="override a split size, e.g. --size train=20000")
    ap.add_argument("--steps", type=int, default=None,
                    help="training steps (default 40000; smoke 300)")
    args = ap.parse_args()
    name = args.cache_name or DEFAULT_CACHE[args.family]
    if not name.startswith("meta6"):
        raise SystemExit("the rigid-body families live in caches named "
                         "meta6*")
    if not 1 <= args.procs <= MAX_PROCS:
        raise SystemExit(f"--procs {args.procs}: 1 .. {MAX_PROCS} (16 GB "
                         "laptop shared with the running pipelines)")
    PROCS = args.procs
    CACHE = os.path.join(HERE, "_cache", name
                         + ("_smoke" if args.smoke else ""))
    os.makedirs(CACHE, exist_ok=True)
    S2.CACHE = S3.CACHE = CACHE
    S2.PROCS = S3.PROCS = PROCS
    tag = f"{ACT}-{args.family}"
    for f, want in ((os.path.join(CACHE, "act_family.json"),
                     dict(act_family=tag)),
                    (os.path.join(CACHE, "op_family.json"),
                     dict(op_family="m15-format", prior_family=args.family,
                          act_family=ACT))):
        if os.path.exists(f):
            if json.load(open(f)) != want:
                raise SystemExit(f"{f} holds {json.load(open(f))}, not "
                                 f"{want}")
        else:
            _write_atomic(f, lambda fh, w=want: json.dump(w, fh))
    args.act_family = tag
    args.sizes = dict(S2.SIZES)
    if args.smoke:
        args.sizes = dict(train=40, A=4, B=2, C=4, At=4)
    for s in args.size:
        k, v = s.split("=")
        args.sizes[k] = int(v)
    args.split_list = [s for s in args.splits.split(",") if s]
    for s in args.split_list:
        if s not in SPLITS6:
            raise SystemExit(f"split {s}: one of {SPLITS6}")
    phases = [p for p in args.phase.split(",") if p]
    if any(p in TRAIN_PHASES for p in phases) and not gpu_ok():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""     # before torch loads
    args.cache = CACHE
    args.code = code6()
    log(f"family {args.family}, code {args.code}, simulation code "
        f"{sim6()}, actuators {ACT} (At: at), cache {CACHE}, procs {PROCS}")
    for nm in phases:
        log(f"=== phase {nm}")
        PHASES[nm](args)


if __name__ == "__main__":
    main()
