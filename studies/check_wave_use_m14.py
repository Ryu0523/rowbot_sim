#!/usr/bin/env python3
"""
Does the source-trained wave-input model (info_waves_m11) use the waves IN
FAMILY, where the relation exists (ROADMAP_2026-09-30 step 1, check a)?
On split A, the same steps t in [240, 288) are predicted with the last Lh
recorded steps as history (Lh = 16, 64, 232), by model3.pt (no waves) and
by model3_waves_m11.pt (with the 15 elevations), and the episodes are split
by their operator: one whose filters read the wave group (style wave_in)
versus none. If the network learned to recognise a wave relation from the
history, the wave model's gain should be larger on wave-reading operators
and grow with Lh; if the gain is the same in both groups (only the
low-fidelity boat's own heave / pitch wave response, present in every
episode), it did not.

    python studies/check_wave_use_m14.py
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np   # noqa: E402
import torch   # noqa: E402

import info_waves_m11 as IW   # noqa: E402
from learn.meta import model3 as M   # noqa: E402

C = IW.C
LH = (16, 64, 232)
T0, NT = 240, 48
NAMES = ("surge", "sway", "yaw", "heave", "pitch")


def skills(net, D, eps, lh, gen):
    se, s0 = np.zeros(5), np.zeros(5)
    with torch.no_grad():
        for t in range(T0, T0 + NT, 4):
            for b0 in range(0, len(eps), 64):
                ii = eps[b0:b0 + 64]
                a = torch.full_like(ii, t - lh)
                tok, tgt, ok = M.window_tokens(D, ii, a, lh + 1)
                h = net.encode(tok)[:, -1]
                y, v = tgt[:, -1], ok[:, -1]
                s = net.sample(h[v], 32, gen=gen)
                m = s.mean(1)
                se += ((m - y[v]) ** 2 - s.var(1) / 32).sum(0)[:5].cpu().numpy()
                s0 += (y[v] ** 2).sum(0)[:5].cpu().numpy()
    return se / np.maximum(s0, 1e-12)


def main():
    dev = M.device()
    ck = torch.load(os.path.join(C, "model3.pt"), weights_only=False)
    base = M.Net().to(dev)
    base.load_state_dict(ck["net"])
    base.eval()
    wv = IW.NetW().to(dev)
    wv.load_state_dict(torch.load(os.path.join(C, "model3_waves_m11.pt"),
                                  weights_only=False)["net"])
    wv.eval()
    Db = M.Data3(C, "A", dev, stats=ck["stats"])
    Dw = IW.DataW(C, "A", dev, stats=ck["stats"])
    meta = pickle.load(open(os.path.join(C, "A_meta.pkl"), "rb"))
    ok = (Db.len > T0 + NT + 1).cpu().numpy()
    wave = np.array([any(m["style"].get("wave_in", [])) and
                     not m["style"].get("null", False) for m in meta])
    groups = {"operator reads waves": np.flatnonzero(ok & wave),
              "operator does not": np.flatnonzero(ok & ~wave)}
    gen = torch.Generator(device=dev).manual_seed(0)
    print("one-step skill at t in [240, 288) by history length (lower is "
          "better): no-wave model / wave model")
    for g, idx in groups.items():
        eps = torch.as_tensor(idx, device=dev)
        print(f"  {g} ({len(idx)} episodes):")
        for lh in LH:
            sb = skills(base, Db, eps, lh, gen)
            sw = skills(wv, Dw, eps, lh, gen)
            print(f"    history {lh:3d}: " + "  ".join(
                f"{n} {a:.2f}/{b:.2f}" for n, a, b in zip(NAMES, sb, sw)))


if __name__ == "__main__":
    main()
