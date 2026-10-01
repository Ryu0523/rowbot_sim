#!/usr/bin/env python3
"""
Which extra observation carries the most information about the TARGET's
one-step errors (DEFECTS M11)? An information test run where the relation
exists: the target world's own episodes (C: feedback commands, Cb: plans
fixed in 24-step blocks). The source world has no roll, so this cannot be
a training recipe; it only ranks the candidate signals.

Every variant starts from model3.pt (studies/_cache/meta2) and is
fine-tuned identically on the same target episodes; the extra inputs are
appended to each step's token with their input weights initialised to 0,
so every variant starts as model3.pt:

  base   no extra input (what target data alone adds)
  roll   roll angle and roll rate at step t
  wave   the 15 elevations at the MPC's stations at step t
  wmid   the 15 elevations half a step later at the hull's dead-reckoned
         mid-step pose (from the CURRENT state only, so the realised
         motion does not leak): a perfect short preview
  all    roll + wave + wmid

Episodes: every 4th of each split held out for scoring; of the rest, every
8th for early stopping; the others train. One-step skill (MSE of the
sample mean / MSE of 0, sample-variance corrected) and 90% coverage on the
held-out episodes, per channel.

    python studies/info_target_m11.py [--steps 4000]
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
from learn.meta import relabel

C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache",
                 os.environ.get("M11_CACHE", "meta2"))
LOG = os.path.join(C, "info_target_m11.log")


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def elev(sea, X, Y, t):
    ph = sea.k * (X[..., None] * np.cos(sea.th) + Y[..., None] * np.sin(sea.th)) \
        - sea.w * np.asarray(t)[..., None] + sea.phi
    return (sea.a * np.cos(ph)).sum(-1)


def extras_of(split, env, lib):
    """Per step raw extra features of a target split: dict name -> (n, T, k)."""
    d = np.load(os.path.join(C, f"{split}.npz"))
    meta = pickle.load(open(os.path.join(C, f"{split}_meta.pkl"), "rb"))
    XS, S = d["XS"].astype(float), d["S"]
    n, T = d["U"].shape[:2]
    dtc = env["dt"] * env["sub"]
    x_st, y_off = np.asarray(env["x_st"]), np.asarray(env["y_off"])
    roll = np.stack([XS[:, :T, 3], XS[:, :T, 9]], -1)
    wave = (S[..., 11:26] - lib["mu"][11:26]) / lib["sd"][11:26]
    wmid = np.zeros((n, T, 15))
    for i in range(n):
        sea = relabel._sea(meta[i])
        L = int(d["len"][i])
        s = XS[i, :L]
        x, y, psi, u, v, r = s[:, 0], s[:, 1], s[:, 5], s[:, 6], s[:, 7], \
            s[:, 11]
        h = 0.5 * dtc
        xm = x + (u * np.cos(psi) - v * np.sin(psi)) * h
        ym = y + (u * np.sin(psi) + v * np.cos(psi)) * h
        pm = psi + r * h
        c, s_ = np.cos(pm)[:, None, None], np.sin(pm)[:, None, None]
        X = xm[:, None, None] + x_st[None, :, None] * c - y_off[None, None, :] * s_
        Y = ym[:, None, None] + x_st[None, :, None] * s_ + y_off[None, None, :] * c
        tt = (np.arange(L) * dtc + h)[:, None, None]
        wmid[i, :L] = elev(sea, X, Y, tt).reshape(L, 15)
    wmid = (wmid - lib["mu"][11:26]) / lib["sd"][11:26]
    return dict(roll=roll, wave=wave, wmid=wmid)


class Target:
    """C and Cb together as one Data3-like set, with extra token columns."""

    def __init__(self, dev, stats, env, names):
        lib = np.load(os.path.join(C, "lib.npz"))
        parts = [M.Data3(C, s, dev, stats=stats) for s in ("C", "Cb")]
        T = max(p.T for p in parts)

        def pad(x):
            return torch.nn.functional.pad(x, (0, 0, 0, T - x.shape[1])) \
                if x.dim() == 3 else torch.nn.functional.pad(
                    x, (0, T - x.shape[1]))
        self.X = torch.cat([pad(p.X) for p in parts])
        self.U = torch.cat([pad(p.U) for p in parts])
        self.E = torch.cat([pad(p.E) for p in parts])
        self.valid = torch.cat([pad(p.valid.to(torch.uint8)).bool()
                                for p in parts])
        self.len = torch.cat([p.len for p in parts])
        self.split = np.array([0] * parts[0].n + [1] * parts[1].n)
        self.n, self.T, self.dev, self.stats = len(self.len), T, dev, stats
        ex = {k: [] for k in ("roll", "wave", "wmid")}
        for s in ("C", "Cb"):
            e = extras_of(s, env, lib)
            for k in ex:
                v = e[k]
                ex[k].append(np.pad(v, ((0, 0), (0, T - v.shape[1]), (0, 0))))
        ex = {k: np.concatenate(v) for k, v in ex.items()}
        sd_roll = ex["roll"][self.valid.cpu().numpy()].std(0) + 1e-9
        ex["roll"] = ex["roll"] / sd_roll
        cols = [ex[k] for k in names]
        self.XE = torch.tensor(np.concatenate(cols, -1) if cols else
                               np.zeros((self.n, T, 0)), dtype=torch.float32,
                               device=dev) * (self.valid[..., None] |
                                              (torch.arange(T, device=dev)[
                                                  None, :, None]
                                               < self.len[:, None, None]))
        self.k = self.XE.shape[-1]


def tokens(D, ii, a, L, gen=None, noise=True):
    tok, tgt, ok = M.window_tokens(D, ii, a, L, M.obs_sd(len(ii), D.dev, gen)
                                   if noise else None, gen)
    t = (a[:, None] + torch.arange(L, device=D.dev)[None]).clamp(max=D.T - 1)
    return torch.cat([tok, D.XE[ii[:, None], t]], -1), tgt, ok


def widen(net, k):
    """model3's Net with k extra token inputs, their weights 0."""
    if k == 0:
        return net
    old = net.inp[0]
    new = nn.Linear(old.in_features + k, old.out_features).to(old.weight.device)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_features] = old.weight
        new.bias.copy_(old.bias)
    net.inp[0] = new
    return net


def run_variant(name, names, dev, ck, env, steps, lr=1e-4, L=M.W_CTX):
    D = Target(dev, ck["stats"], env, names)
    idx = np.arange(D.n)
    within = np.concatenate([np.arange((D.split == s).sum()) for s in (0, 1)])
    hold = idx[within % 4 == 3]
    rest = idx[within % 4 != 3]
    val = rest[np.arange(len(rest)) % 8 == 1]
    tr = rest[np.arange(len(rest)) % 8 != 1]
    torch.manual_seed(0)
    net = M.Net().to(dev)
    net.load_state_dict(ck["net"])
    net = widen(net, D.k)
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
            tok, tgt, ok = tokens(D, ii, a, L, gv)
            ls.append(M.fm_loss_h(net, net.encode(tok), tgt, ok, gv).item())
        return float(np.mean(ls))

    best = dict(v=validate(), it=0, state={k_: x.clone() for k_, x in
                                            net.state_dict().items()}, bad=0)
    log(f"  {name}: {D.k} extra inputs; {len(tr)} training / {len(val)} "
        f"validation / {len(hold)} held-out episodes; step 0 validation "
        f"{best['v']:.4f}")
    for it in range(1, steps + 1):
        ii, a = M.fm_draw(trt, 24, g, Lmax, L, dev)
        tok, tgt, ok = tokens(D, ii, a, L, g)
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
    net.eval()
    log(f"  {name}: kept step {best['it']} (validation {best['v']:.4f})")
    # one-step on the held-out episodes, full history as eval_one_step
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
                    tok, tgt, ok = tokens(D, ii, a, L, noise=False)
                    h = net.encode(tok)
                    pos = torch.arange(L, device=dev) + a0
                    sel = ok & (pos[None] >= lo) & (pos[None] < hi)
                    smp = net.sample(h[sel], 64, gen=gen)
                    tal.add(smp, tgt[sel])
        out[nm] = tal.summary()
    del D
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()
    dev = M.device()
    ck = torch.load(os.path.join(C, "model3.pt"), weights_only=False)
    env = relabel._env()
    VARIANTS = dict(base=[], roll=["roll"], wave=["wave"], wmid=["wmid"],
                    all=["roll", "wave", "wmid"])
    res = {}
    for name, names in VARIANTS.items():
        res[name] = run_variant(name, names, dev, ck, env, args.steps)
        pickle.dump(res, open(os.path.join(C, "info_target_m11.pkl"), "wb"))
    ch = "surge sway yaw heave pitch"
    for split in ("C", "Cb"):
        log(f"held-out {split}, one-step skill [{ch}] | cov90:")
        for name, r in res.items():
            s = r[split]
            log(f"  {name:<5} " + " ".join(f"{x:.2f}" for x in s["skill"][:5])
                + " | " + " ".join(f"{x:.2f}" for x in s["cov"][:5]))


if __name__ == "__main__":
    main()
