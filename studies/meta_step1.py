#!/usr/bin/env python3
"""
Step 1 of the general adaptation method: can a network infer the model's
error from a short history -- including errors of a kind it never saw --
and know what it does NOT know?

Data (learn/meta/data.py): the Scarab in sea state 3, randomly excited
(speed and heading targets; a third of the episodes steady and
uninformative), 86 s each (360 control steps).
  train   source world + a random residual function; the HYSTERESIS style
          held out of the family
  A       the same family, new functions (in-family generalisation)
  B       every function has a relay (hysteresis): a style never seen in
          training; scored on all of B and on the episodes whose relay
          really switched ('memory')
  C       the full planing plant, no injected residual: the real gap
          between it and the model's equations (never seen in any form)

Model (learn/meta/model.py): a learned basis, a transformer history
encoder, a flow-matching posterior over the basis coefficients, trained on
the function's truth over the whole input box.

Two scores, both as SKILL = error / error of 'no correction', per channel
then averaged (1.0 = no better than no correction, 0 = perfect):
  box     the whole function (128 probe states: half near visited states,
          half anywhere in the input box) after a history of H steps --
          with the 90% interval's coverage (0.90 = calibrated; 256
          samples) and width; steady and informative episodes apart. A
          steady boat has learnt nothing about speeds it never ran: its
          box posterior must be WIDE there and still cover the truth.
  traj    the observed error over the next 40 steps (~10 s) after a prefix
          of L steps: the only score split C allows.
Baselines: zero; persistence (last 4 / 16 steps, prefix mean); ridge
regression on the history with the same learned basis (whole history, or
its last 40 steps); a Gaussian Bayesian regression on the same basis with a
prior fitted to the training functions (what a non-Gaussian flow must
beat); and ceilings: the function's own fit (box) or the future window's
own fit (traj).

Run:  python -m studies.meta_step1 [--phase all|data|train|eval|report]
      [--workers 5] [--smoke]
"""
import argparse
import hashlib
import os
import pickle
import time

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                    # noqa: E402

from learn.repro import task                          # noqa: E402

SMOKE = False
N_TRAIN = None          # --n-train: train on the first N episodes (learning curve)
AUG = False             # --aug: noisy-history augmentation (model.noisy_history)
SIZES = dict(train=2000, A=200, B=200, C=48)
SMOKE_SIZES = dict(train=24, A=6, B=6, C=6)
SOURCES = ("learn/meta/residuals.py", "learn/meta/data.py",
           "learn/repro/task.py", "sim/lofi.py", "sim/planing_vessel.py",
           "control/reduced.py")
CH = ("surge", "sway", "yaw", "heave", "pitch")


def code_hash():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha1()
    for f in SOURCES:
        h.update(open(os.path.join(root, f), "rb").read())
    return h.hexdigest()[:12]


def cache_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache",
                     "meta" + ("_smoke" if SMOKE else ""))
    os.makedirs(d, exist_ok=True)
    return d


def jobs(split, n):
    base = dict(train=1, A=100000, B=200000, C=300000)[split]
    out = []
    for i in range(n):
        j = dict(seed=base + i)
        if split in ("train", "A"):
            j.update(world="low", exclude=("hysteresis",))
        elif split == "B":
            j.update(world="low", force=("hysteresis",))
        else:
            j.update(world="high", informative=(i % 3 != 0))
        if SMOKE:
            j["T"] = 40.0
        out.append(j)
    return out


def phase_data(workers):
    from learn.meta.data import episode
    sizes = SMOKE_SIZES if SMOKE else SIZES
    code = code_hash()
    for split, n in sizes.items():
        p = os.path.join(cache_dir(), f"{split}.pkl")
        if os.path.exists(p):
            if pickle.load(open(p, "rb")).get("code") == code:
                print(f"    {split}: cached")
                continue
            print(f"    {split}: code changed since it was built; rebuilding")
        t0 = time.time()
        run = task.Runner(workers, fn=episode)
        try:
            eps = run(jobs(split, n))
        finally:
            run.close()
        pickle.dump(dict(code=code, eps=eps), open(p, "wb"))
        bad = sum(not e["finite"] for e in eps)
        print(f"    {split}: {len(eps)} episodes in {(time.time() - t0) / 60:.1f}"
              f" min ({bad} non-finite)", flush=True)


def load(split):
    d = pickle.load(open(os.path.join(cache_dir(), f"{split}.pkl"), "rb"))
    if d.get("code") != code_hash():
        print(f"    WARNING: {split} was built by other code")
    return d["eps"]


def _tag():
    return ("" if N_TRAIN is None else f"_n{N_TRAIN}") + ("_aug" if AUG else "")


def phase_train():
    import torch
    from learn.meta import model as M
    dev = M.device()
    eps = load("train")
    if N_TRAIN is not None:
        eps = eps[:N_TRAIN]
    D = M.Episodes(eps, dev)
    print(f"    {D.n} training episodes ({D.n_dropped} dropped), "
          f"{int(D.len.sum())} steps, device {dev}; y std "
          f"{np.round(D.y_std, 3).tolist()}", flush=True)
    t0 = time.time()
    basis = M.train_basis(D, steps=200 if SMOKE else 3000)
    Zk = M.probe_targets(basis, D)
    enc, flow, zst = M.train_flow(D, Zk, steps=200 if SMOKE else 6000,
                                  augment=AUG)
    noise = M.traj_noise(basis, D)
    torch.save(dict(basis=basis.state_dict(), enc=enc.state_dict(),
                    flow=flow.state_dict(), z_mu=zst[0].cpu(),
                    z_sd=zst[1].cpu(), y_std=D.y_std, noise=noise.cpu(),
                    Zk=Zk.cpu(), HP=D.HP.cpu()),
               os.path.join(cache_dir(), f"model{_tag()}.pt"))
    print(f"    trained in {(time.time() - t0) / 60:.1f} min; blr noise "
          f"{np.round(noise.cpu().numpy(), 3).tolist()}", flush=True)


def phase_eval():
    import torch
    from learn.meta import model as M
    dev = M.device()
    ck = torch.load(os.path.join(cache_dir(), f"model{_tag()}.pt"),
                    weights_only=False)
    basis, enc, flow = M.Basis().to(dev), M.Encoder().to(dev), M.Flow().to(dev)
    basis.load_state_dict(ck["basis"])
    enc.load_state_dict(ck["enc"])
    flow.load_state_dict(ck["flow"])
    for m in (basis, enc, flow):
        m.eval()
    zst = (ck["z_mu"].to(dev), ck["z_sd"].to(dev))
    blr = M.GaussBLR(ck["Zk"].to(dev), ck["HP"].to(dev), ck["noise"].to(dev))
    res = dict(y_std_train=ck["y_std"])
    Hs = (0, 8, 32) if SMOKE else (0, 8, 32, 64, 120)
    Ls = (0, 8, 32) if SMOKE else (0, 8, 32, 128, 240)
    for split in ("A", "B", "C"):
        D = M.Episodes(load(split), dev, y_std=ck["y_std"])
        r = dict(n=D.n, n_dropped=D.n_dropped, y_rms=D.y_rms_raw,
                 traj=M.evaluate_traj(basis, enc, flow, zst, blr, D, Ls=Ls,
                                      horizon=10 if SMOKE else 40))
        if bool(D.HP.any()):
            r["box"] = M.evaluate_probes(basis, enc, flow, zst, blr, D, Hs=Hs)
        res[split] = r
    pickle.dump(res, open(os.path.join(cache_dir(), f"eval{_tag()}.pkl"),
                          "wb"))
    return res


# ---------------------------------------------------------------- report
def _skill(rows, key, methods, base="zero"):
    """Per method: mean over channels of sum(se)/sum(se of zero)."""
    if not rows:
        return None
    z = np.sum([r[f"se_{base}{key}"] for r in rows], 0) + 1e-12
    return {m: float(np.mean(np.sum([r[f"se_{m}{key}"] for r in rows], 0)
                             / z)) for m in methods}


def report():
    p = os.path.join(cache_dir(), f"eval{_tag()}.pkl")
    if not os.path.exists(p):
        print("  no evaluation yet")
        return
    res = pickle.load(open(p, "rb"))
    ys = res["y_std_train"]
    print("\n  error size per channel (rms, the channels' own units) vs the "
          "training std")
    print("    " + " " * 10 + "".join(f"{c:>8}" for c in CH))
    print("    " + f"{'train std':<10}" + "".join(f"{v:>8.3f}" for v in ys))
    for s in ("A", "B", "C"):
        print("    " + f"{s + ' rms':<10}"
              + "".join(f"{v:>8.3f}" for v in res[s]["y_rms"])
              + f"   ({res[s]['n']} episodes, {res[s]['n_dropped']} dropped)")
    bm = ("ridge", "blr", "flow", "oracle")
    print("\n  BOX: the whole function after a history window of H steps "
          "(skill: 1 = no correction; cov90 0.90 = calibrated)")
    for s in ("A", "B"):
        rows = res[s].get("box", [])
        subsets = [("all", rows), ("steady", [r for r in rows if not r["info"]]),
                   ("moving", [r for r in rows if r["info"]])]
        if s == "B":
            subsets.append(("memory", [r for r in rows if r["mem"]]))
        for sub, rr in subsets:
            if not rr:
                continue
            n_ep = len({r["ep"] for r in rr})
            print(f"\n  {s} / {sub} ({n_ep} episodes)")
            print(f"    {'H':>4} {'region':>6}" + "".join(f"{m:>8}" for m in bm)
                  + f"{'cov flow':>9}{'cov blr':>8}{'wid flow':>9}{'wid blr':>8}")
            for H in sorted({r["H"] for r in rr}):
                for reg in ("near", "box"):
                    q = [r for r in rr if r["H"] == H]
                    sk = _skill(q, f"_{reg}", bm)
                    cf = np.mean([r[f"cov_flow_{reg}"] for r in q])
                    cb = np.mean([r[f"cov_blr_{reg}"] for r in q])
                    wf = np.mean([r[f"wid_flow_{reg}"] for r in q])
                    wb = np.mean([r[f"wid_blr_{reg}"] for r in q])
                    print(f"    {H:>4} {reg:>6}"
                          + "".join(f"{sk[m]:>8.3f}" for m in bm)
                          + f"{cf:>9.2f}{cb:>8.2f}{wf:>9.2f}{wb:>8.2f}")
    tm = ("last4", "last16", "mean", "ridge", "ridge40", "blr", "flow",
          "window")
    print("\n  TRAJ: the observed error over the next 40 steps after a "
          "prefix of L steps (skill: 1 = no correction)")
    for s in ("A", "B", "C"):
        rows = res[s]["traj"]
        subsets = [("all", rows)]
        if s == "B":
            subsets.append(("memory", [r for r in rows if r["mem"]]))
        if s == "C":
            subsets += [("steady", [r for r in rows if not r["info"]]),
                        ("moving", [r for r in rows if r["info"]])]
        for sub, rr in subsets:
            if not rr:
                continue
            print(f"\n  {s} / {sub}")
            print(f"    {'L':>4}" + "".join(f"{m:>8}" for m in tm))
            for L in sorted({r["L"] for r in rr}):
                q = [r for r in rr if r["L"] == L]
                sk = _skill(q, "", tm)
                print(f"    {L:>4}" + "".join(f"{sk[m]:>8.3f}" for m in tm))
    rows = res["C"]["traj"]
    if rows:
        print("\n  C per channel, flow and ridge skill by prefix:")
        for L in sorted({r["L"] for r in rows}):
            q = [r for r in rows if r["L"] == L]
            z = np.sum([r["se_zero"] for r in q], 0)
            f = np.sum([r["se_flow"] for r in q], 0) / z
            g = np.sum([r["se_ridge"] for r in q], 0) / z
            print(f"    L {L:>4}  flow " + " ".join(f"{c}={v:.2f}" for c, v
                                                   in zip(CH, f))
                  + "  | ridge " + " ".join(f"{v:.2f}" for v in g))


def main(phase="all", workers=5):
    t0 = time.time()
    order = ("data", "train", "eval")
    todo = order if phase == "all" else () if phase == "report" else (phase,)
    for ph in todo:
        print(f"\n  [{ph}]  {task.avail_gb():.1f} GB free", flush=True)
        {"data": lambda: phase_data(workers), "train": phase_train,
         "eval": phase_eval}[ph]()
        print(f"  [{ph}] done, {(time.time() - t0) / 60:.1f} min", flush=True)
    report()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--aug", action="store_true")
    a = ap.parse_args()
    SMOKE = a.smoke
    N_TRAIN = a.n_train
    AUG = a.aug
    main(a.phase, a.workers)
