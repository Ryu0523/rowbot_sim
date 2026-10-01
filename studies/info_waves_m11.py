#!/usr/bin/env python3
"""
Information test (DEFECTS M11): would measuring the waves make the
target's yaw and pitch errors predictable? The SAME one-step model as
model3.pt (same network, training, early stopping), trained on the same
episodes (studies/_cache/meta2 train split), with one change: its state
input also carries the 15 wave elevations at the MPC's stations (S[11:26],
present in the source and the target records alike), normalised by the
library like the other inputs. Compared with model3.pt on the one-step
evaluation of A, B, C and Cb.

This is not a ceiling estimate with a weaker model: it is the real model
with one extra observation, and the difference answers whether the missing
information is in the waves. (The source's errors depend on the waves
through the reduced model's heave / pitch forcing and through the operators
that read the elevations, so the network can learn to use them.)

    python studies/info_waves_m11.py [--steps 20000]
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch import nn

from learn.meta import model3 as M

C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache",
                 os.environ.get("M11_CACHE", "meta2"))
DXW = 26                          # state inputs: S's 11 + the 15 elevations


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(C, "info_waves_m11.log"), "a",
              encoding="utf-8") as fh:
        fh.write(line + "\n")


class DataW(M.Data3):
    """Data3 with the 15 elevations in the state input."""

    def __init__(self, cache, split, dev, stats=None):
        super().__init__(cache, split, dev, stats=stats)
        d = np.load(os.path.join(cache, f"{split}.npz"))
        lib = np.load(os.path.join(cache, "lib.npz"))
        mu, sd = lib["mu"].astype(np.float32), lib["sd"].astype(np.float32)
        T, L = d["U"].shape[1], d["len"]
        X = (d["S"][..., :DXW] - mu[:DXW]) / sd[:DXW]
        self.X = torch.tensor(X * (np.arange(T)[None] < L[:, None])[..., None],
                              dtype=torch.float32, device=dev)


class NetW(M.Net):
    def __init__(self):
        super().__init__()
        self.inp = M.mlp(DXW + M.D_U + M.C7 + 1, (self.d,), self.d)


def train(D, steps, seed=0, L=M.W_CTX, lr=3e-4, patience=3):
    """model3.train with NetW (same schedule, pools and early stopping)."""
    dev = D.dev
    torch.manual_seed(seed)
    net = NetW().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=steps,
                                                pct_start=0.05)
    g = torch.Generator(device="cpu").manual_seed(seed)
    tr, va = M.split_pools(D.n, seed)
    Lmax = D.len.cpu()
    best = dict(v=float("inf"), it=-1, bad=0)
    t0 = time.time()
    for it in range(steps):
        ii, a = M.fm_draw(tr, 48, g, Lmax, L, dev)
        loss = M.fm_loss(net, D, ii, a, L, g)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == steps - 1:
            v = M.fm_validate(net, D, va, L, Lmax)
            log(f"    wave-input model {it:6d}: train {loss.item():.4f}  "
                f"validation {v:.4f}  ({time.time() - t0:.0f} s)")
            if it > steps // 5:
                if v < best["v"]:
                    best.update(v=v, it=it, bad=0, state={
                        k: x.detach().clone()
                        for k, x in net.state_dict().items()})
                else:
                    best["bad"] += 1
                    if best["bad"] >= patience:
                        log(f"    stopping at step {it}")
                        break
    if best["it"] >= 0:
        net.load_state_dict(best["state"])
        log(f"    restored step {best['it']} ({best['v']:.4f})")
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    dev = M.device()
    ck = torch.load(os.path.join(C, "model3.pt"), weights_only=False)
    stats = ck["stats"]                   # same e scale as model3.pt
    path = os.path.join(C, "model3_waves_m11.pt")
    if not args.eval_only:
        D = DataW(C, "train", dev, stats=stats)
        log(f"train: {D.n} episodes, state input {DXW} (with elevations)")
        net = train(D, args.steps)
        torch.save(dict(net=net.state_dict(), stats=stats), path)
        q = torch.quantile(D.E[D.valid].abs(), 0.99, dim=0)
        del D
        torch.cuda.empty_cache()
    else:
        net = NetW().to(dev)
        net.load_state_dict(torch.load(path, weights_only=False)["net"])
        Dt = M.Data3(C, "train", dev, stats=stats)
        q = torch.quantile(Dt.E[Dt.valid].abs(), 0.99, dim=0)
        del Dt
    net.eval()
    base = pickle.load(open(os.path.join(C, os.environ.get(
        "M11_BASE", "eval3_kv.pkl")), "rb"))
    res = {}
    for split in ("A", "B", "C", "Cb"):
        D = DataW(C, split, dev, stats=stats)
        r = M.eval_one_step(net, D, q)
        res[split] = r
        log(f"eval {split}, one step, WITH the elevations:\n"
            + M.fmt_one(r, "waves ") + "\n"
            + M.fmt_one(base[(split, "one")], "model3"))
        del D
        torch.cuda.empty_cache()
    pickle.dump(res, open(os.path.join(C, "eval_waves_m11.pkl"), "wb"))


if __name__ == "__main__":
    main()
