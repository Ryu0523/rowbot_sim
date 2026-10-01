#!/usr/bin/env python3
"""
What does the synthetic prior buy over training on target data alone
(DEFECTS M13)? The same network trained on the same target episodes three
ways, scored on the same held-out target episodes:

  scratch   random initialisation, every weight (lr 3e-4, as model3.train)
  full      from model3.pt, every weight (lr 1e-4)
  head      from model3.pt, the flow head only (lr 1e-4)

for n = 4, 16, 84 training episodes, and two training pools:

  mixed     C and Cb episodes alternating (as info_target_size_m11.py)
  C only    feedback-controlled episodes only, scored on Cb as well: plans
            fixed in advance with full nozzle reversals, commands unlike the
            training ones (the shift an MPC's candidate plans bring)

Early stopping on validation episodes of the same pool; held-out skill
(remaining MSE / MSE of the error) and 90% coverage per channel, raw and
with the spread correction of adapt_calib_m11.py (base-noise scale chosen
on the validation episodes).

    python studies/scratch_vs_prior_m13.py
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np   # noqa: E402
import torch   # noqa: E402
from torch import nn   # noqa: E402

import adapt_calib_m11 as AC   # noqa: E402
import info_target_m11 as I   # noqa: E402
from learn.meta import model3 as M   # noqa: E402
from learn.meta import relabel   # noqa: E402

SIZES = (4, 16, 84)
MODES = tuple(os.environ.get("M13_MODES", "scratch,full,head").split(","))
POOLS = ("mixed", "C only")


def pools(D, pool, n):
    idx = np.arange(D.n)
    within = np.concatenate([np.arange((D.split == s).sum()) for s in (0, 1)])
    hold = idx[within % 4 == 3]
    rest = idx[within % 4 != 3]
    val = rest[np.arange(len(rest)) % 8 == 1]
    tr = rest[np.arange(len(rest)) % 8 != 1]
    if pool == "C only":
        val = val[D.split[val] == 0]
        tr = tr[D.split[tr] == 0][:n]
    else:
        trC, trB = tr[D.split[tr] == 0], tr[D.split[tr] == 1]
        tr = np.array([x for p in zip(trC, trB) for x in p][:n], int)
    return tr, val, hold


def train(D, mode, tr, val, ck, dev, L=M.W_CTX, steps=8000):
    torch.manual_seed(0)
    net = M.Net().to(dev)
    if mode != "scratch":
        net.load_state_dict(ck["net"])
    train_p = list(net.head.parameters()) if mode == "head" else \
        list(net.parameters())
    ids = {id(p) for p in train_p}
    for p in net.parameters():
        p.requires_grad_(id(p) in ids)
    lr = 3e-4 if mode == "scratch" else 1e-4
    opt = torch.optim.AdamW(train_p, lr, weight_decay=1e-4)
    warm = 200 if mode == "scratch" else 1
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / warm))
    g = torch.Generator().manual_seed(0)
    Lmax = D.len.cpu()
    trt, vat = torch.as_tensor(tr), torch.as_tensor(val)

    @torch.no_grad()
    def validate():
        gv = torch.Generator().manual_seed(4321)
        ls = []
        for _ in range(8):
            ii, a = M.fm_draw(vat, 16, gv, Lmax, L, dev)
            tok, tgt, ok = I.tokens(D, ii, a, L, gv)
            ls.append(M.fm_loss_h(net, net.encode(tok), tgt, ok, gv).item())
        return float(np.mean(ls))

    best = dict(v=validate(), it=0, state={k_: x.clone() for k_, x in
                                            net.state_dict().items()}, bad=0)
    patience = 8 if mode == "scratch" else 4
    for it in range(1, steps + 1):
        ii, a = M.fm_draw(trt, 24, g, Lmax, L, dev)
        tok, tgt, ok = I.tokens(D, ii, a, L, g)
        loss = M.fm_loss_h(net, net.encode(tok), tgt, ok, g)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(train_p, 1.0)
        opt.step()
        sched.step()
        if it % 250 == 0:
            v = validate()
            if v < best["v"]:
                best.update(v=v, it=it, bad=0, state={
                    k_: x.clone() for k_, x in net.state_dict().items()})
            else:
                best["bad"] += 1
            if best["bad"] >= patience:
                break
    net.load_state_dict(best["state"])
    net.eval()
    return net, best["it"]


def main():
    dev = M.device()
    ck = torch.load(os.path.join(I.C, os.environ.get("M13_MODEL", "model3.pt")),
                    weights_only=False)
    env = relabel._env()
    D = I.Target(dev, ck["stats"], env, [])
    res = {}
    for pool in POOLS:
        for n in SIZES:
            for mode in MODES:
                tr, val, hold = pools(D, pool, n)
                if len(tr) < n:
                    continue
                net, it = train(D, mode, tr, val, ck, dev)
                covs = {s: AC.score(net, D, val, s, dev)["cov"][:5].mean()
                        for s in AC.SCALES}
                sb = min(AC.SCALES, key=lambda s: abs(covs[s] - 0.9))
                out = {}
                for s in sorted({1.0, sb}):
                    out[s] = {nm: AC.score(net, D, hold[D.split[hold] == k],
                                           s, dev)
                              for k, nm in ((0, "C"), (1, "Cb"))}
                res[(pool, n, mode)] = (out, sb, it)
                I.log(f"  {pool} n={n} {mode}: kept step {it}, spread "
                      f"scale {sb}")
                pickle.dump(res, open(os.path.join(
                    I.C, "scratch_vs_prior_m13.pkl"), "wb"))
                del net
                torch.cuda.empty_cache()
    ch = "surge sway yaw heave pitch"
    for split in ("C", "Cb"):
        I.log(f"held-out {split}: skill [{ch}] | cov90 (raw)   | with the "
              f"spread correction")
        for (pool, n, mode), (out, sb, it) in res.items():
            a, b = out[1.0][split], out[sb][split]
            I.log(f"  {pool:<6} n={n:2d} {mode:<7} " + " ".join(
                f"{x:.2f}" for x in a["skill"][:5]) + " | " + " ".join(
                f"{x:.2f}" for x in a["cov"][:5]) + f" | s={sb}: " + " ".join(
                f"{x:.2f}" for x in b["skill"][:5]) + " / " + " ".join(
                f"{x:.2f}" for x in b["cov"][:5]))


if __name__ == "__main__":
    main()
