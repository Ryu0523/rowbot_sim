#!/usr/bin/env python3
"""
Adapting to the target with little data without losing calibration
(DEFECTS M11). Full fine-tuning on 4 target episodes (info_target_size_m11)
cut yaw skill from ~0.9 to ~0.6 but left 90% intervals covering only
30-40%. Compared here on n = 4, 8, 16 training episodes, the same
held-out episodes and the same early stopping:

  full      every weight (the earlier run)
  prior     every weight, plus an L2 pull toward model3.pt's weights
            (strength 1e-2 x number of parameters normalised)
  head      only the flow head (the transformer frozen)
  last      the last transformer layer, its final LayerNorm and the head

and, for each, a spread correction chosen on the validation episodes
only: the base noise of the sampler scaled by s in {1, 1.25, 1.5, 2, 3}
(the s whose validation coverage is closest to 0.90), then scored on the
held-out episodes.

    python studies/adapt_calib_m11.py
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np   # noqa: E402
import torch   # noqa: E402
from torch import nn   # noqa: E402

import info_target_m11 as I   # noqa: E402
from learn.meta import model3 as M   # noqa: E402
from learn.meta import relabel   # noqa: E402

SIZES = (4, 8, 16)
MODES = ("full", "prior", "head", "last")
SCALES = (1.0, 1.25, 1.5, 2.0, 3.0)


@torch.no_grad()
def sample_s(net, h, n, s, gen, steps=24):
    """net.sample with the base noise scaled by s."""
    B = h.shape[0]
    hh = h.repeat_interleave(n, 0)
    y = s * torch.randn(B * n, M.C7, device=h.device, generator=gen)
    for i in range(steps):
        tau = torch.full((B * n,), i / steps, device=h.device)
        y = y + net.velocity(y, tau, hh) / steps
    return (M.KAPPA * torch.sinh(y)).view(B, n, M.C7)


@torch.no_grad()
def score(net, D, eps, s, dev, L=M.W_CTX, seed=0):
    gen = torch.Generator(device=dev).manual_seed(seed)
    tal = M.Tally((M.C7,))
    hs = torch.as_tensor(eps, device=dev)
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
            tal.add(sample_s(net, h[sel], 64, s, gen), tgt[sel])
    return tal.summary()


def run(D, mode, n, ck, dev, lr=1e-4, L=M.W_CTX, steps=4000):
    idx = np.arange(D.n)
    within = np.concatenate([np.arange((D.split == s).sum()) for s in (0, 1)])
    hold = idx[within % 4 == 3]
    rest = idx[within % 4 != 3]
    val = rest[np.arange(len(rest)) % 8 == 1]
    tr = rest[np.arange(len(rest)) % 8 != 1]
    trC, trB = tr[D.split[tr] == 0], tr[D.split[tr] == 1]
    tr = np.array([x for p in zip(trC, trB) for x in p][:n], int)
    torch.manual_seed(0)
    net = M.Net().to(dev)
    net.load_state_dict(ck["net"])
    theta0 = [p.detach().clone() for p in net.parameters()]
    if mode == "head":
        train_p = list(net.head.parameters())
    elif mode == "last":
        train_p = (list(net.tf.layers[-1].parameters())
                   + list(net.out.parameters()) + list(net.head.parameters()))
    else:
        train_p = list(net.parameters())
    ids = {id(p) for p in train_p}
    for p in net.parameters():
        p.requires_grad_(id(p) in ids)
    opt = torch.optim.AdamW(train_p, lr, weight_decay=0.0)
    n_par = sum(p.numel() for p in net.parameters())
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
    for it in range(1, steps + 1):
        ii, a = M.fm_draw(trt, 24, g, Lmax, L, dev)
        tok, tgt, ok = I.tokens(D, ii, a, L, g)
        loss = M.fm_loss_h(net, net.encode(tok), tgt, ok, g)
        if mode == "prior":
            loss = loss + 1e-2 * sum(((p - p0) ** 2).sum() for p, p0 in
                                     zip(net.parameters(), theta0)) \
                * (1e4 / n_par)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(train_p, 1.0)
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
    net.eval()
    # spread correction chosen on the validation episodes only
    covs = {s: score(net, D, val, s, dev)["cov"][:5].mean() for s in SCALES}
    s_best = min(SCALES, key=lambda s: abs(covs[s] - 0.9))
    out = {}
    for s in sorted({1.0, s_best}):
        out[s] = {nm: score(net, D, hold[D.split[hold] == k], s, dev)
                  for k, nm in ((0, "C"), (1, "Cb"))}
    I.log(f"  {mode} n={n}: kept step {best['it']}; validation coverage by "
          f"scale " + ", ".join(f"{s}: {c:.2f}" for s, c in covs.items())
          + f" -> {s_best}")
    return out, s_best


def main():
    dev = M.device()
    ck = torch.load(os.path.join(I.C, "model3.pt"), weights_only=False)
    env = relabel._env()
    D = I.Target(dev, ck["stats"], env, [])
    res = {}
    for n in SIZES:
        for mode in MODES:
            res[(mode, n)] = run(D, mode, n, ck, dev)
            pickle.dump(res, open(os.path.join(I.C, "adapt_calib_m11.pkl"),
                                  "wb"))
    ch = "surge sway yaw heave pitch"
    for split in ("C", "Cb"):
        I.log(f"held-out {split}: skill [{ch}] | cov90, without and with the "
              f"spread correction")
        for (mode, n), (out, sb) in res.items():
            a, b = out[1.0][split], out[sb][split]
            I.log(f"  {mode:<5} n={n:2d} " + " ".join(
                f"{x:.2f}" for x in a["skill"][:5]) + " | " + " ".join(
                f"{x:.2f}" for x in a["cov"][:5]) + f"   s={sb}: " + " ".join(
                f"{x:.2f}" for x in b["skill"][:5]) + " | " + " ".join(
                f"{x:.2f}" for x in b["cov"][:5]))


if __name__ == "__main__":
    main()
