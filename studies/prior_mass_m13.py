#!/usr/bin/env python3
"""
Does the synthetic prior put any mass near the TARGET's error function
(DEFECTS M13)? The in-context model predicts what is probable under the
prior, so if no probable member of the operator family behaves like the
target, it cannot be expected to predict the target from one episode's
history, whatever its capacity.

N operators are drawn from the training prior (learn/meta/operators.py,
relay off, null operators excluded) and each is run, without noise, on the
target's own recorded inputs (the 26 measured signals, wave elevations
included, and the commands of C and Cb). Its rule output is compared with
the target's one-step errors of the 5 velocity channels (e0 channels 0-4,
the same units as a push), with one least-squares scale per channel (the
family draws amplitudes anyway): skill = remaining MSE / MSE of the error,
over all valid target steps from step 40 on, ONE operator for all target
episodes (the target is one boat; the waves are among the inputs).

Reported: the distribution of skill over the draws per channel, how many
draws come within reach of what the same network reached after training on
target data (DEFECTS M11: surge ~0.1, sway ~0.1, yaw ~0.3, heave ~0.08,
pitch ~0.47 on held-out target episodes), and the best draws.

This is a statement about the prior (where its mass is), not an estimate of
what the learned model can do.

    python studies/prior_mass_m13.py [--n 20000] [--procs 8]
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache", "meta3")
NAMES = ("surge", "sway", "yaw", "heave", "pitch")
REF = np.array([0.1, 0.1, 0.3, 0.08, 0.47])      # DEFECTS M11, target data
_W = {}


def _init():
    from learn.meta import relabel
    from learn.meta.data2 import op_inputs
    lib = dict(np.load(os.path.join(C, "lib.npz")))
    Xs, Ys, Ms = [], [], []
    for split in ("C", "Cb"):
        d = np.load(os.path.join(C, f"{split}.npz"))
        e0 = np.load(os.path.join(C, f"{split}_e0.npz"))["E0"]
        T = d["U"].shape[1]
        L = d["len"]
        n_e = L - 1 if d["XS"].shape[1] == T else L
        Xs.append(op_inputs(d["S"], d["U"]).astype(float))
        Ys.append(e0[..., :5].astype(float))
        Ms.append((np.arange(T)[None] >= 40) & (np.arange(T)[None]
                                               < n_e[:, None]))
    T = min(x.shape[1] for x in Xs)
    _W.update(lib=lib, X=np.concatenate([x[:, :T] for x in Xs]),
              Y=np.concatenate([y[:, :T] for y in Ys]),
              M=np.concatenate([m[:, :T] for m in Ms]),
              L=float(relabel._env()["L"]),
              dt=float(relabel._env()["dt"] * relabel._env()["sub"]))


def _one(seed):
    from learn.meta.operators import Operator
    if not _W:
        _init()
    op = Operator(seed, _W["lib"], dt=_W["dt"], L=_W["L"], relay=False,
                  null=False)
    R = op.run(_W["X"])                                # (B, T, 5)
    m = _W["M"]
    r, y = R[m], _W["Y"][m]                            # (n, 5)
    rr = (r * r).sum(0)
    a = np.where(rr > 1e-12, (r * y).sum(0) / np.maximum(rr, 1e-12), 0.0)
    res = ((y - a * r) ** 2).sum(0)
    return seed, res / (y * y).sum(0), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    from multiprocessing import Pool
    t0 = time.time()
    seeds = list(range(9_000_000, 9_000_000 + args.n))
    out = []
    with Pool(args.procs, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, seeds, chunksize=16)):
            out.append(r)
            if (i + 1) % 2000 == 0:
                print(f"{i + 1} operators in {time.time() - t0:.0f} s",
                      flush=True)
    sk = np.array([o[1] for o in out])
    sd = np.array([o[0] for o in out])
    pickle.dump(dict(seed=sd, skill=sk, scale=np.array([o[2] for o in out])),
                open(os.path.join(C, "prior_mass_m13.pkl"), "wb"))
    print(f"\n{len(out)} operators from the training prior, run on the "
          f"target's inputs ({time.time() - t0:.0f} s)")
    print("skill (remaining MSE / MSE of the target's error, one scale per "
          "channel), quantiles over the draws:")
    for c, nm in enumerate(NAMES):
        q = np.quantile(sk[:, c], [0.0, 0.001, 0.01, 0.1, 0.5])
        print(f"  {nm:<6} best {q[0]:.3f}  0.1% {q[1]:.3f}  1% {q[2]:.3f}  "
              f"10% {q[3]:.3f}  median {q[4]:.3f}   draws below the "
              f"target-data reference {REF[c]:.2f}: "
              f"{(sk[:, c] <= REF[c]).sum()}, below 0.9: "
              f"{(sk[:, c] <= 0.9).sum()}")
    tot = sk.mean(1)
    best = np.argsort(tot)[:5]
    print("best draws by the mean over channels:")
    for b in best:
        print(f"  seed {sd[b]}: " + " ".join(f"{nm} {x:.3f}" for nm, x in
                                             zip(NAMES, sk[b])))


if __name__ == "__main__":
    main()
