#!/usr/bin/env python3
"""
M17 pipeline (cache meta7): the one-step error models trained on the D10a
force-catalogue error family (learn/meta/operators_cat.py, the draft
learn/meta/PRIOR_D10_DRAFT.md, PRIOR_DERIVATION.md D10): layer 1 the D8.12
rigid body with the overlap switches, layer 2 the catalogue items
learn/meta/cat_*.py, layer 3 the D9 residual at the declared shares. One
family, --family cat (learn/meta/episode7.py); cache meta7 (--smoke:
meta7_smoke). A copy of studies/meta_step6.py (not edited) for this family:
the M10 actuators (At: target-like), the M15 wave-input FORMAT (W_MID; the
preview = W_MID; D10a keeps meta5's network inputs), the same scenario,
splits, seeds and training, so the three priors (meta5 old projection,
meta6 general D9, meta7 catalogue) compare cleanly. Evaluation:
studies/eval_preview_m15.py --cache-name meta7 (unchanged).

Phases (each cached, rerun with --force):
  copy      lib.npz, G.npy and the target splits C / Cb with meta and e0
            from --copy-from (meta2; meta_step2.phase_copy)
  data      train (--size train=N, --t-train s), A, B, At; chunks of 250
            episodes, resumable, each stamped <code7>-<act>-cat (code7: the
            hash of meta_step2.SOURCES + EP_SOURCES: every file the
            episodes run, incl. all catalogue item modules and the meta6
            modules imported read-only); a chunk with failed episodes gets
            those seeds rerun on the next data run. A / B / At episodes
            keep snapshots at the branch moments (SNAPS)
  pack      episode7.pack7: W_MID, APK / AMIN / HMIN, PUSH, IMP, SLAM and
            OBS (the extra observed signals of draft section 11, stored,
            not network inputs); meta op_family 'm15', prior_family 'cat',
            op_seed_cat; <split>_snaps.pkl; the chunk signature ('chunks').
            Logs the acceptance-test rejections per catalogue item
  wmid      C_wmid.npz, Cb_wmid.npz (data5.target_wmid, unchanged)
  e0        the error against the MPC's model (data3.e0_split)
  branches  closed-loop continuations from the episodes' snapshots
            (data7.build_branches7; meta5's episodes, moments and plans)
  train_a / train_w / train_p   model3.pt, model_w.pt, model_p.pt
            (model3.train / model_preview, unchanged)
Stamps <file>.code on wmid / e0 / branches: sim7 (meta_step3's simulation
sources + the episode sources + data7 / data6 / data5) and the pack. All
files are written atomically (temp file + os.replace).

Memory: at most 8 worker processes (--procs; 2 while other data jobs
run); the data and branch workers import no torch. Training uses the GPU
only when nvidia-smi reports at least 2 GB free (else CPU).

    python -m studies.meta_step7 --family cat \\
        --phase copy,data,pack,wmid,e0,branches --size train=10000 --procs 2
    python -m studies.meta_step7 --family cat \\
        --phase train_a,train_w,train_p --steps 40000
    python studies/eval_preview_m15.py --cache-name meta7
"""
import argparse
import collections
import glob
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

CACHE = os.path.join(HERE, "_cache", "meta7")
ACT = "m10"
FAMILIES = ("cat",)
DEFAULT_CACHE = dict(cat="meta7")


def _cat_modules():
    """Every catalogue module (cat_base.load_items imports all of them)."""
    return tuple(sorted(
        os.path.relpath(f, ROOT).replace(os.sep, "/")
        for f in glob.glob(os.path.join(ROOT, "learn", "meta", "cat_*.py"))))


# the code the episodes run (code7): the new modules, every catalogue
# module, and the meta6 modules / simulation code imported read-only
EP_SOURCES = (("learn/meta/episode7.py", "learn/meta/operators_cat.py")
              + _cat_modules()
              + ("learn/meta/episode6.py", "learn/meta/episode5.py",
                 "learn/meta/ops_m15.py", "learn/meta/operators_rb.py",
                 "learn/meta/operators_gen.py", "control/reduced.py",
                 "sim/wavefield.py", "sim/config.py"))
SIM_SOURCES = EP_SOURCES + ("learn/meta/data7.py", "learn/meta/data6.py",
                            "learn/meta/data5.py")
SPLITS7 = ("train", "A", "B", "At")
SNAPS = (120, 240)              # branch moments (data6 / data5 defaults)
SNAP_SPLITS = ("A", "B", "At")
PROCS = 2
MAX_PROCS = 8  # 2 while other data pipelines run; up to 8 alone


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(CACHE, "run7.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _hash(files):
    h = hashlib.sha1()
    for f in files:
        h.update(open(os.path.join(ROOT, f), "rb").read())
    return h.hexdigest()[:12]


def code7():
    """The data's code: meta_step2.SOURCES + the episode sources."""
    return _hash(tuple(S2.SOURCES) + EP_SOURCES)


def sim7():
    """The simulation code of wmid / e0 / branches."""
    return _hash(tuple(S3._sim_sources()) + SIM_SOURCES)


def _chunks_sig(split, n):
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
    return sim7() + "|" + S3._input_sig(split, extra)


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
    from learn.meta import episode7
    lib_path = os.path.join(CACHE, "lib.npz")
    if not S2._check_copied(("lib.npz",)):
        raise SystemExit("data: lib.npz must be copied from meta2 first "
                         "(phase copy)")
    raw = os.path.join(CACHE, "raw")
    os.makedirs(raw, exist_ok=True)
    S2._run = episode7.run_job          # the episode meta_step2.run_jobs runs
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


def _log_rejections(split, meta):
    """Acceptance-test rejections per catalogue item: rejected attempts
    with the item on / all attempts with the item on."""
    st = [m_["style"] for m_ in meta]
    n_att, n_rej = collections.Counter(), collections.Counter()
    for s_ in st:
        for items in s_.get("reject_items", ()):
            for c in items:
                n_att[c] += 1
                n_rej[c] += 1
        for c in s_.get("items", ()):
            n_att[c] += 1
    n_r = sum(s_.get("n_reject", 0) > 0 for s_ in st)
    n_off = sum(bool(s_.get("forces_off")) for s_ in st)
    worst = sorted(((n_rej[c] / max(n_att[c], 1), c) for c in n_att
                    if n_rej[c]), reverse=True)[:8]
    tiers = collections.Counter(s_.get("tier") for s_ in st)
    log(f"  acceptance ({split}): {n_r} of {len(st)} draws rejected at "
        f"least once, forces_off {n_off}; rejected share of attempts per "
        "item (highest): " + (", ".join(f"{c} {r:.3f}" for r, c in worst)
                              or "none")
        + f"; layer-3 tiers {dict(tiers)}; sparse "
        f"{sum(bool(s_.get('sparse')) for s_ in st)}")


def phase_pack(args):
    from learn.meta import episode7
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
        arrs, meta, snaps = episode7.pack7(eps)
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
            f"{arrs['W_MID'].shape}, OBS {arrs['OBS'].shape}, snapshots of "
            f"{len(snaps)} episodes"
            + (f"; removed {', '.join(gone)}" if gone else ""))
        _log_rejections(split, meta)


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
    for split in SPLITS7:
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
    from learn.meta import data7
    for split in SNAP_SPLITS:
        if not _has(split):
            continue
        out = os.path.join(CACHE, f"{split}_branches.npz")
        n_eps = 8 if args.smoke else 60
        if _fresh(out, args, split, n_eps):
            continue
        t0 = time.time()
        data7.build_branches7(os.path.join(CACHE, f"{split}.npz"), _lib(),
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
    ap.add_argument("--family", default="cat", choices=FAMILIES)
    ap.add_argument("--phase", default="copy,data,pack,wmid,e0,branches")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--procs", type=int, default=PROCS)
    ap.add_argument("--cache-name", default=None, help="default meta7")
    ap.add_argument("--copy-from", default="meta2")
    ap.add_argument("--splits", default=",".join(SPLITS7))
    ap.add_argument("--t-train", type=float, default=None,
                    help="episode length of the train split, s (default "
                         "data2.T_EP = 90)")
    ap.add_argument("--size", action="append", default=[],
                    help="override a split size, e.g. --size train=20000")
    ap.add_argument("--steps", type=int, default=None,
                    help="training steps (default 40000; smoke 300)")
    args = ap.parse_args()
    name = args.cache_name or DEFAULT_CACHE[args.family]
    if not name.startswith("meta7"):
        raise SystemExit("the catalogue family lives in caches named meta7*")
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
        if s not in SPLITS7:
            raise SystemExit(f"split {s}: one of {SPLITS7}")
    phases = [p for p in args.phase.split(",") if p]
    if any(p in TRAIN_PHASES for p in phases) and not gpu_ok():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""     # before torch loads
    args.cache = CACHE
    args.code = code7()
    log(f"family {args.family}, code {args.code}, simulation code "
        f"{sim7()}, actuators {ACT} (At: at), cache {CACHE}, procs {PROCS}")
    for nm in phases:
        log(f"=== phase {nm}")
        PHASES[nm](args)


if __name__ == "__main__":
    main()
