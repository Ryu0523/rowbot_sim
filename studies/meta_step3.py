#!/usr/bin/env python3
"""
Step 3 of the inference network: the ONE-STEP error model with physics
rollouts (learn/meta/model3.py, data3.py; DEFECTS M7). It reuses the step-2
episodes (studies/_cache/meta2: train, A, B, C) and adds, in the same cache:

  e0        the error against the MPC's rollout model (no waves, ideal
            actuators), 7 channels: <split>_e0.npz for train, A, B, C
  branches  multi-step truth for A and B: from moments 120 and 240 of 60
            episodes, 8 command sequences fixed in advance, each simulated in
            closed loop with the full operator (its internal state recovered
            by replay): <split>_branches.npz
  cblocks   64 target-world episodes whose commands are fixed in advance in
            24-step blocks (split 'Cb'): an unconfounded multi-step truth
  train     the causal transformer + flow on the training split (model3.pt)
  eval      one-step on A, B, C, Cb; multi-step (physics rollouts) on A, B
            (branches), A_wbranches (if built) and Cb (block starts);
            eval3<out-tag>.pkl

    python -m studies.meta_step3 --phase all [--smoke]

Rollout-robust fine-tuning (DEFECTS M9), NOT part of --phase all; init is
always model3.pt, which these phases never overwrite:

  tbranches   wide-plan (full-range command steps) branches of the training
              split: train_tbranches (training episodes), train_vbranches
              (model3.train's held-out episodes), and A_wbranches (test)
  train_cov   V0: FM loss on recorded + branch windows -> model3_cov.pt
  train_es    V1: plus the energy score of 24-step rollouts -> model3_es.pt

    python -m studies.meta_step3 --phase tbranches
    python -m studies.meta_step3 --phase train_cov   (then train_es)
    python -m studies.meta_step3 --phase eval --tag "" --out-tag _kv
    python -m studies.meta_step3 --phase eval --tag _cov   (and _es)

--cache-name picks the cache (default meta2; meta3 = the M10 actuator family,
made by studies/meta_step2.py --act-family m10, with C / Cb copied from
meta2; --smoke appends _smoke). A split At (A's episodes with fixed
target-shaped actuators, a test split) is used wherever it exists: e0,
branches, tbranches (At_wbranches) and eval. The files e0, branches and
tbranches write carry a stamp <file>.code: the hash of the simulation code
(step 2's SOURCES, which include operators.py, plus relabel, data3, the
wavefield and the reduced model) and of the pack they were computed from
(its _meta.pkl and size; for tbranches also the job list). They are
rebuilt when either changed; files without a stamp (made before M10, e.g.
meta2's) are kept as they are. Default 4 worker processes.

The steady-block yaw reading and the oracle-data test (PRIOR_DERIVATION.md
D7), NOT part of --phase all:

  cbhold        split Cbh: target block episodes whose nozzle is held over
                whole blocks (w.p. P_HOLD), with e0; evaluated like Cb
  train_oracle  model3.pt fine-tuned on Cbh's episodes except every 4th
                -> model3_oracle.pt; scored by diag_yaw_m8_blocks --split
                Cbh --holdout --model model3_oracle.pt
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache", "meta2")
PROCS = 4                # worker processes: at most 4 on the 16 GB laptop


def log(msg, f="run3.log"):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(CACHE, f), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _sim_sources():
    """Step 2's SOURCES (the data's code: operators, residuals, data2, lofi,
    task, ...) plus what step 3 itself simulates with: relabel, data3, the
    wavefield and the reduced model (Mission.red)."""
    from studies.meta_step2 import SOURCES
    return tuple(SOURCES) + ("learn/meta/relabel.py", "learn/meta/data3.py",
                             "sim/wavefield.py", "control/reduced.py")


def sim_code():
    """Hash of the code that simulates e0 / branches / tbranches."""
    import hashlib
    h = hashlib.sha1()
    for f in _sim_sources():
        h.update(open(os.path.join(os.path.dirname(HERE), f), "rb").read())
    return h.hexdigest()[:12]


def _input_sig(split, extra=None):
    """What a derived file was computed from: the sha1 of the source pack's
    <split>_meta.pkl (seeds, code, request, actuator draws of every episode)
    and the size of <split>.npz, plus `extra` (e.g. a job list)."""
    import hashlib
    h = hashlib.sha1()
    mp = os.path.join(CACHE, f"{split}_meta.pkl")
    if os.path.exists(mp):
        h.update(open(mp, "rb").read())
    pk = os.path.join(CACHE, f"{split}.npz")
    h.update(str(os.path.getsize(pk) if os.path.exists(pk) else -1).encode())
    if extra is not None:
        h.update(repr(extra).encode())
    return h.hexdigest()[:12]


def _stamp_text(split, extra=None):
    return sim_code() + "|" + _input_sig(split, extra)


def _fresh(path, args, split, extra=None):
    """path exists and may be reused: not --force, and its stamp (if any)
    matches the current simulation code AND the pack it was computed from
    (and `extra`, e.g. the tbranches job list). Unstamped files are pre-M10
    (meta2's) and are kept."""
    if not os.path.exists(path) or args.force:
        return False
    st = path + ".code"
    if not os.path.exists(st):
        return True
    if open(st).read().strip() == _stamp_text(split, extra):
        return True
    log(f"{os.path.basename(path)}: made by other simulation code or from "
        "another pack, rebuilt")
    return False


def _stamp(path, split, extra=None):
    with open(path + ".code", "w") as f:
        f.write(_stamp_text(split, extra))


def _has(split):
    return os.path.exists(os.path.join(CACHE, f"{split}.npz"))


def _lib():
    d = np.load(os.path.join(CACHE, "lib.npz"))
    return dict(mu=d["mu"], sd=d["sd"], S=d["S"])


def _cb_e0(name="Cb"):
    """e for a block split from its stored states (T + 1 rows)."""
    from learn.meta import data3, relabel
    d = np.load(os.path.join(CACHE, f"{name}.npz"))
    XS, U, L = d["XS"], d["U"], d["len"]
    env = relabel._env()
    E0 = np.zeros((len(L), U.shape[1], data3.N_E), np.float32)
    for i in range(len(L)):
        n_ = int(L[i])
        E0[i, :n_] = data3.e0_from(env, XS[i, :n_], XS[i, 1:n_ + 1],
                                   U[i, :n_])
    np.savez(os.path.join(CACHE, f"{name}_e0.npz"), E0=E0)


def phase_e0(args):
    from learn.meta import data3
    for split in ("train", "A", "B", "C", "At"):
        if split == "At" and not _has(split):
            continue
        out = os.path.join(CACHE, f"{split}_e0.npz")
        if _fresh(out, args, split):
            continue
        t0 = time.time()
        data3.e0_split(os.path.join(CACHE, f"{split}.npz"))
        _stamp(out, split)
        log(f"e0 {split}: {time.time() - t0:.0f} s")
    for name in ("Cb", "Cbh"):
        if os.path.exists(os.path.join(CACHE, f"{name}.npz")) and (
                args.force or not os.path.exists(os.path.join(
                    CACHE, f"{name}_e0.npz"))):
            _cb_e0(name)
            log(f"e0 {name}")


def phase_branches(args):
    from learn.meta import data3
    for split in ("A", "B", "At"):
        if split == "At" and not _has(split):
            continue
        out = os.path.join(CACHE, f"{split}_branches.npz")
        n_eps = 8 if args.smoke else 60
        if _fresh(out, args, split, n_eps):
            continue
        t0 = time.time()
        data3.build_branches(os.path.join(CACHE, f"{split}.npz"), _lib(),
                             n_eps=n_eps, procs=PROCS)
        _stamp(out, split, n_eps)
        log(f"branches {split}: {time.time() - t0:.0f} s")


def _block_job(job):
    from learn.meta import data3
    try:
        return data3.episode_blocks(job)
    except Exception as ex:
        import traceback
        return dict(error=repr(ex) + traceback.format_exc()[-400:],
                    seed=job["seed"])


def _blocks_split(name, jobs):
    """Run block-planned target episodes (data3.episode_blocks; in this
    process when PROCS <= 1) -> <name>.npz, <name>_meta.pkl, <name>_e0.npz.
    Returns (episodes kept, errors)."""
    from concurrent.futures import ProcessPoolExecutor
    out = os.path.join(CACHE, f"{name}.npz")
    if PROCS <= 1:
        eps = [_block_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(PROCS) as ex:
            eps = list(ex.map(_block_job, jobs))
    bad = [e for e in eps if "error" in e]
    eps = [e for e in eps if "error" not in e and len(e["U"]) > 60]
    T = max(len(e["U"]) for e in eps)
    nE = len(eps)
    S = np.zeros((nE, T, 26), np.float32)
    U = np.zeros((nE, T, 2), np.float32)
    E = np.zeros((nE, T, 5), np.float32)
    XS = np.zeros((nE, T + 1, 14), np.float32)
    L = np.zeros(nE, np.int32)
    for i, e in enumerate(eps):
        n_ = len(e["U"])
        S[i, :n_], U[i, :n_], E[i, :n_] = e["S"], e["U"], e["E"]
        XS[i, :n_ + 1] = e["XS"]
        L[i] = n_
    np.savez(out, S=S, U=U, E=E, XS=XS, len=L)
    p_hold = {j["seed"]: j.get("p_hold", 0.0) for j in jobs}
    pickle.dump([dict(seed=e["seed"], sea=e["sea"], world="high",
                      finite=e["finite"], p_hold=p_hold[e["seed"]])
                 for e in eps],
                open(out.replace(".npz", "_meta.pkl"), "wb"))
    _cb_e0(name)
    return nE, bad


def phase_cblocks(args):
    out = os.path.join(CACHE, "Cb.npz")
    if os.path.exists(out) and not args.force:
        return
    n = 6 if args.smoke else 64
    t0 = time.time()
    nE, bad = _blocks_split("Cb", [dict(seed=400000 + i) for i in range(n)])
    log(f"cblocks: {nE} episodes ({len(bad)} errors) in "
        f"{time.time() - t0:.0f} s" + (f"; first error {bad[0]['error']}"
                                       if bad else ""))


CBH_SEED0 = 410000      # Cb uses 400000 + i
P_HOLD = 0.6            # share of Cbh blocks whose nozzle is held


def phase_cbhold(args):
    """Target-world block episodes with the nozzle HELD over whole blocks,
    split 'Cbh' (D7): as cblocks, but w.p. P_HOLD a block keeps the previous
    block's last nozzle command for all 24 steps (data3.episode_blocks
    p_hold). Cb has too few blocks with the nozzle command constant and the
    actual nozzle at rest on it for the steady-block yaw reading; Cbh also
    feeds the oracle-data test (train_oracle). The target world does not
    depend on the source actuators, so Cbh is the same in every cache."""
    out = os.path.join(CACHE, "Cbh.npz")
    if os.path.exists(out) and not args.force:
        return
    n = 6 if args.smoke else 128
    t0 = time.time()
    nE, bad = _blocks_split("Cbh", [dict(seed=CBH_SEED0 + i, p_hold=P_HOLD)
                                    for i in range(n)])
    log(f"cbhold: {nE} episodes ({len(bad)} errors) in "
        f"{time.time() - t0:.0f} s" + (f"; first error {bad[0]['error']}"
                                       if bad else ""))


def oracle_holdout(n):
    """Cbh episodes held out of the oracle fine-tuning (every 4th): the
    oracle-data test scores steady-block yaw on these only."""
    return np.arange(0, n, 4)


def phase_train_oracle(args):
    """The oracle-data test (D7): model3.pt fine-tuned on the TARGET world
    itself, Cbh's episodes except oracle_holdout (the same teacher-forced
    FM loss as train, lr 1e-4) -> model3_oracle.pt; never overwrites
    model3.pt. Then diag_yaw_m8_blocks --split Cbh --holdout --model
    model3_oracle.pt: if the oracle learns steady-block yaw on the held-out
    target episodes, the information is in the inputs the model already
    sees (a prior gap: broaden the operator family); if even the oracle
    cannot, the inputs lack it, and only then is it an observation
    question."""
    import torch

    from learn.meta import model3 as M
    out = os.path.join(CACHE, "model3_oracle.pt")
    dev = M.device()
    ck = torch.load(os.path.join(CACHE, "model3.pt"), weights_only=False)
    net = M.Net().to(dev)
    net.load_state_dict(ck["net"])
    D = M.Data3(CACHE, "Cbh", dev, stats=ck["stats"])
    hold = set(oracle_holdout(D.n).tolist())
    rest = [i for i in range(D.n) if i not in hold]
    va = rest[::8] if len(rest) >= 16 else rest[:1]
    tr = [i for i in rest if i not in set(va)]
    log(f"train_oracle: Cbh {D.n} episodes, {len(tr)} training, {len(va)} "
        f"validation, {len(hold)} held out for the test")
    net = M.train(D, steps=args.steps or (20 if args.smoke else 5000),
                  lr=1e-4, log=log, net=net,
                  pools=(torch.tensor(tr), torch.tensor(va)))
    torch.save(dict(net=net.state_dict(), stats=ck["stats"],
                    init="model3.pt", holdout=sorted(hold)), out)
    log("saved model3_oracle.pt")


def _tjobs(L, eps, seed, P=4, n_mom=2, k_min=8):
    """Branch jobs: n_mom DISTINCT moments k ~ U{k_min .. len - 1} per
    episode (a branch needs only XS[k] and U[k]), P plans each."""
    rng = np.random.default_rng(seed)
    jobs = []
    for i in eps:
        if int(L[i]) - k_min < n_mom:
            continue
        ks = rng.choice(np.arange(k_min, int(L[i])), n_mom, replace=False)
        jobs += [(int(i), int(k), P, seed) for k in sorted(ks)]
    return jobs


def phase_tbranches(args):
    """Wide-plan branches (full-range command steps; DEFECTS M9):
    train_tbranches / train_vbranches for model3.train_rollout (the training
    episodes, INCLUDING null-style operators, and exactly model3.train's
    held-out episodes; 2 moments x 4 plans each) and the in-family test
    A_wbranches (A_branches' episodes and moments, 8 wide plans; never used
    for training or stopping)."""
    from learn.meta import data3
    from learn.meta.model3 import val_episodes
    procs = args.procs or 4
    path = os.path.join(CACHE, "train.npz")
    L = np.load(path)["len"]
    va = sorted(val_episodes(len(L)).tolist())
    tr = [i for i in range(len(L)) if i not in set(va)]
    jt, jv = _tjobs(L, tr, seed=1), _tjobs(L, va, seed=2)
    assert not ({j[0] for j in jt} & {j[0] for j in jv})
    if args.max_jobs:
        jt, jv = jt[:args.max_jobs], jv[:args.max_jobs]
    # A: the episode choice of build_branches (same seed and filter) and
    # its plan seed, so the plans are A_branches' plus the full-range steps;
    # At (A's episodes, target-shaped actuators) the same, where it exists
    todo = [(path, jt, "train_tbranches", "train"),
            (path, jv, "train_vbranches", "train")]
    for sp in ("A", "At"):
        if sp == "At" and not _has(sp):
            continue
        pa = os.path.join(CACHE, f"{sp}.npz")
        La = np.load(pa)["len"]
        meta = pickle.load(open(pa.replace(".npz", "_meta.pkl"), "rb"))
        moments = (120, 240)
        ok = [i for i in range(len(La))
              if La[i] >= max(moments) + data3.HB + 1
              and not meta[i]["style"]["null"]]
        n_eps = 8 if args.smoke else 60
        eps = np.random.default_rng(0).choice(ok, size=min(n_eps, len(ok)),
                                              replace=False)
        ja = [(int(i), int(k), 8, 0) for i in eps for k in moments]
        if args.max_jobs:
            ja = ja[:args.max_jobs]
        todo.append((pa, ja, f"{sp}_wbranches", sp))
    for src, jobs, name, sp in todo:
        out = os.path.join(CACHE, f"{name}.npz")
        if _fresh(out, args, sp, jobs):
            continue
        t0 = time.time()
        data3.build_tbranches(src, _lib(), jobs, out, procs=procs)
        _stamp(out, sp, jobs)
        w = np.load(out)["wide"]
        log(f"tbranches {name}: {len(jobs)} moments x {w.shape[1]} plans, "
            f"{w.mean():.2f} of the plans with a full-range step, "
            f"{time.time() - t0:.0f} s")


def _train_rollout(args, es_weight, default_tag):
    """V0 / V1 fine-tuning from model3.pt (DEFECTS M9) -> model3<tag>.pt;
    never writes model3.pt itself."""
    import torch

    from learn.meta import model3 as M
    from learn.meta import relabel
    tag = args.out_tag if args.out_tag is not None else default_tag
    out = os.path.join(CACHE, f"model3{tag}.pt")
    if not tag or os.path.basename(out) == "model3.pt":
        raise SystemExit("refusing to overwrite model3.pt; pass --out-tag")
    dev = M.device(0.5)
    ck = torch.load(os.path.join(CACHE, "model3.pt"), weights_only=False)
    net = M.Net().to(dev)
    net.load_state_dict(ck["net"])
    D = M.Data3(CACHE, "train", dev, stats=ck["stats"])
    ld = lambda n: dict(np.load(os.path.join(CACHE, n)))  # noqa: E731
    tb, vb = ld("train_tbranches.npz"), ld("train_vbranches.npz")
    log(f"train_rollout (ES weight {es_weight}): {D.n} episodes, "
        f"{tb['U'].shape[0]} training / {vb['U'].shape[0]} validation "
        f"branch moments x {tb['U'].shape[1]} plans")
    steps = args.steps if args.steps is not None else 3000
    net, info = M.train_rollout(net, D, tb, vb, relabel._env(),
                                es_weight=es_weight, steps=steps, log=log)
    torch.save(dict(net=net.state_dict(), stats=ck["stats"],
                    es_scale=info["scale"], info=info, init="model3.pt"),
               out)
    log(f"saved {os.path.basename(out)} ({info['s_per_step']:.2f} s/step, "
        f"GPU peak {info['gpu_peak_gb']:.2f} GB)")


def phase_train_cov(args):
    _train_rollout(args, 0.0, "_cov")


def phase_train_es(args):
    _train_rollout(args, 1.0, "_es")


def phase_train(args):
    import torch

    from learn.meta import model3 as M
    dev = M.device()
    D = M.Data3(CACHE, "train", dev)
    log(f"train: {D.n} episodes; e0 sd {np.round(D.stats['e_sd'], 3)}")
    net = M.train(D, steps=300 if args.smoke else (args.steps or 20000),
                  log=log)
    torch.save(dict(net=net.state_dict(), stats=D.stats),
               os.path.join(CACHE, f"model3{args.tag}.pt"))
    log("saved model3")


def phase_eval(args):
    import torch

    from learn.meta import model3 as M
    from learn.meta import relabel
    dev = M.device()
    ck = torch.load(os.path.join(CACHE, f"model3{args.tag}.pt"),
                    weights_only=False)
    net = M.Net().to(dev)
    net.load_state_dict(ck["net"])
    net.eval()
    stats = ck["stats"]
    env = relabel._env()
    Dtr = M.Data3(CACHE, "train", dev, stats=stats)
    q = torch.quantile(Dtr.E[Dtr.valid].abs(), 0.99, dim=0)
    del Dtr
    res = {}
    for split in ("A", "B", "At", "C", "Cb", "Cbh"):
        if split in ("At", "Cbh") and not _has(split):
            continue
        D = M.Data3(CACHE, split, dev, stats=stats)
        r1 = M.eval_one_step(net, D, q)
        log(f"eval {split} ({D.n} episodes), one step:\n"
            + M.fmt_one(r1, "one step"))
        res[(split, "one")] = r1
        br = None
        if split in ("A", "B", "At"):
            f = os.path.join(CACHE, f"{split}_branches.npz")
            if os.path.exists(f):
                b = np.load(f)
                br = {k_: b[k_] for k_ in b.files}
        elif split in ("Cb", "Cbh"):
            ep, kk, Us, XS0, E0 = [], [], [], [], []
            E0all = np.load(os.path.join(CACHE, f"{split}_e0.npz"))["E0"]
            for i in range(D.n):
                for k in range(48, int(D.len[i]) - M.HB + 1, M.HB):
                    ep.append(i)
                    kk.append(k)
                    Us.append(D.Uraw[i, k:k + M.HB])
                    XS0.append(D.XS[i, k:k + M.HB + 1])
                    E0.append(E0all[i, k:k + M.HB])
            br = dict(ep=np.array(ep), k=np.array(kk),
                      U=np.stack(Us)[:, None], XS=np.stack(XS0)[:, None],
                      E0=np.stack(E0)[:, None])
        n_samp = 8 if args.smoke else 16
        if br is not None:
            rm = M.eval_multi_step(net, D, br, env, q, n_samp=n_samp)
            log(f"eval {split}, multi step (physics rollouts, plans fixed in "
                f"advance, {br['U'].shape[0] * br['U'].shape[1]} cases):\n"
                + M.fmt_multi(rm, "rollout") + "\n" + _fmt_act(rm))
            res[(split, "multi")] = rm
        fw = os.path.join(CACHE, f"{split}_wbranches.npz")
        if split in ("A", "At") and os.path.exists(fw):
            # in-family full reversals (never used for training or stopping):
            # all wide plans, and only those with a full-range step
            b = np.load(fw)
            bw = {k_: b[k_] for k_ in b.files}
            for key, mask, what in (("multi_wide_all", None, "all plans"),
                                    ("multi_wide", bw["wide"],
                                     "plans with a full-range step")):
                rm = M.eval_multi_step(net, D, bw, env, q, n_samp=n_samp,
                                       mask=mask)
                n_c = int(bw["wide"].sum()) if mask is not None else \
                    bw["wide"].size
                log(f"eval {split}_wbranches, multi step ({what}, {n_c} "
                    f"cases):\n" + M.fmt_multi(rm, "rollout"))
                res[(split, key)] = rm
        del D
        torch.cuda.empty_cache()
    out_tag = args.out_tag if args.out_tag is not None else args.tag
    pickle.dump(res, open(os.path.join(CACHE, f"eval3{out_tag}.pkl"), "wb"))


def _fmt_act(res):
    """The actuator channels' (e 8-9: actual - command, thrust and nozzle)
    multi-step skill, which fmt_multi leaves out: the like-for-like stall
    check across At and Cb next to the diagnostics' open fractions."""
    sk = res["skill"]
    return "  actuator channels [thrust nozzle], skill: " + ", ".join(
        f"steps {a}-{b - 1} " + " ".join(f"{x:.2f}" for x in
                                         sk[a:b, 8:10].mean(0))
        for a, b in ((0, 1), (1, 5), (5, sk.shape[0])))


PHASES = dict(e0=phase_e0, branches=phase_branches, cblocks=phase_cblocks,
              train=phase_train, eval=phase_eval)
# DEFECTS M9, run explicitly (not part of --phase all)
EXTRA = dict(tbranches=phase_tbranches, train_cov=phase_train_cov,
             train_es=phase_train_es, cbhold=phase_cbhold,
             train_oracle=phase_train_oracle)


def main():
    global CACHE, PROCS
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--procs", type=int, default=None,
                    help="worker processes (default 4)")
    ap.add_argument("--steps", type=int, default=None,
                    help="training steps (train 20000, train_cov / "
                         "train_es 3000)")
    ap.add_argument("--tag", default="", help="model3<tag>.pt to evaluate")
    ap.add_argument("--out-tag", default=None,
                    help="names ONE output, so only with a single one of "
                         "train_cov / train_es / eval per call: eval -> "
                         "eval3<out-tag>.pkl (default --tag, the model it "
                         "loads); train_cov / train_es -> model3<out-tag>.pt "
                         "(default _cov / _es)")
    ap.add_argument("--max-jobs", type=int, default=0,
                    help="tbranches: at most this many moments per file "
                         "(tests)")
    ap.add_argument("--cache-name", default="meta2",
                    help="cache folder under studies/_cache (meta3: the M10 "
                         "actuator family); --smoke appends _smoke")
    args = ap.parse_args()
    PROCS = args.procs or PROCS
    CACHE = os.path.join(HERE, "_cache", args.cache_name
                         + ("_smoke" if args.smoke else ""))
    if not os.path.isdir(CACHE):
        raise SystemExit(f"{CACHE}: no such cache (make it with "
                         "studies/meta_step2.py)")
    phases = list(PHASES) if args.phase == "all" else args.phase.split(",")
    if args.out_tag is not None and \
            len({"train_cov", "train_es", "eval"} & set(phases)) > 1:
        raise SystemExit("--out-tag names a single output; run train_cov, "
                         "train_es and eval in separate calls")
    for nm in phases:
        log(f"=== phase {nm}")
        {**PHASES, **EXTRA}[nm](args)


if __name__ == "__main__":
    main()
