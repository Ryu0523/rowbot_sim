#!/usr/bin/env python3
"""
How much target data does the fine-tuning of info_target_m11.py need
(DEFECTS M11)? The base variant (no extra input) and the wave variant
(the 15 elevations) fine-tuned on the first n of the training target
episodes (n = 0 means model3.pt itself), scored on the same held-out
episodes. Each episode is about 86 s of driving.

    python studies/info_target_size_m11.py
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np   # noqa: E402
import torch   # noqa: E402

import info_target_m11 as I   # noqa: E402
from learn.meta import model3 as M   # noqa: E402
from learn.meta import relabel   # noqa: E402

SIZES = (0, 4, 8, 16, 32, 84)


def main():
    dev = M.device()
    ck = torch.load(os.path.join(I.C, "model3.pt"), weights_only=False)
    env = relabel._env()
    res = {}
    for name, names in (("base", []), ("wave", ["wave"])):
        for n in SIZES:
            if n == 0 and name != "base":
                continue
            res[(name, n)] = _run(name, names, dev, ck, env, n)
            pickle.dump(res, open(os.path.join(I.C, "info_target_size_m11.pkl"),
                                  "wb"))
    ch = "surge sway yaw heave pitch"
    for split in ("C", "Cb"):
        I.log(f"held-out {split}, one-step skill [{ch}] by training episodes:")
        for (name, n), r in res.items():
            s = r[split]
            I.log(f"  {name:<5} n={n:3d} " + " ".join(
                f"{x:.2f}" for x in s["skill"][:5]) + " | cov90 " + " ".join(
                f"{x:.2f}" for x in s["cov"][:5]))


def _run(name, names, dev, ck, env, n, lr=1e-4, L=M.W_CTX, steps=4000):
    """info_target_m11.run_variant with the training pool cut to the first
    n episodes (alternating C and Cb); n = 0 scores model3.pt as it is."""
    import torch.nn as nn
    D = I.Target(dev, ck["stats"], env, names)
    idx = np.arange(D.n)
    within = np.concatenate([np.arange((D.split == s).sum()) for s in (0, 1)])
    hold = idx[within % 4 == 3]
    rest = idx[within % 4 != 3]
    val = rest[np.arange(len(rest)) % 8 == 1]
    tr = rest[np.arange(len(rest)) % 8 != 1]
    # the first n training episodes, alternating C and Cb
    trC, trB = tr[D.split[tr] == 0], tr[D.split[tr] == 1]
    inter = [x for pair in zip(trC, trB) for x in pair]
    tr = np.array(inter[:n], int)
    torch.manual_seed(0)
    net = M.Net().to(dev)
    net.load_state_dict(ck["net"])
    net = I.widen(net, D.k)
    if n > 0:
        opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
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
                ls.append(M.fm_loss_h(net, net.encode(tok), tgt, ok,
                                      gv).item())
            return float(np.mean(ls))

        best = dict(v=validate(), it=0, state={k_: x.clone() for k_, x in
                                                net.state_dict().items()},
                    bad=0)
        for it in range(1, steps + 1):
            ii, a = M.fm_draw(trt, 24, g, Lmax, L, dev)
            tok, tgt, ok = I.tokens(D, ii, a, L, g)
            loss = M.fm_loss_h(net, net.encode(tok), tgt, ok, g)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if it % 250 == 0:
                v = validate()
                if v < best["v"]:
                    best.update(v=v, it=it, bad=0, state={
                        k_: x.clone() for k_, x in net.state_dict().items()})
                else:
                    best["bad"] += 1
                if best["bad"] >= 4:
                    break
        net.load_state_dict(best["state"])
        I.log(f"  {name} n={n}: kept step {best['it']} (validation "
              f"{best['v']:.4f})")
    net.eval()
    gen = torch.Generator(device=dev).manual_seed(0)
    out = {}
    for s, nm in ((0, "C"), (1, "Cb")):
        tal = M.Tally((M.C7,))
        hs = torch.as_tensor(hold[D.split[hold] == s], device=dev)
        with torch.no_grad():
            for b0 in range(0, len(hs), 8):
                ii = hs[b0:b0 + 8]
                Tn = int(D.len[ii].max())
                for a0, lo, hi in ((0, 40, L), (max(0, Tn - L), L, Tn)):
                    if hi <= lo:
                        continue
                    a = torch.full_like(ii, a0)
                    tok, tgt, ok = I.tokens(D, ii, a, L, noise=False)
                    h = net.encode(tok)
                    pos = torch.arange(L, device=dev) + a0
                    sel = ok & (pos[None] >= lo) & (pos[None] < hi)
                    smp = net.sample(h[sel], 64, gen=gen)
                    tal.add(smp, tgt[sel])
        out[nm] = tal.summary()
    del D
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    main()
