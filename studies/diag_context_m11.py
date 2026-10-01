#!/usr/bin/env python3
"""
Does the one-step model learn the rules of the system it is used on from
the history (in-context), and does it on the target? The SAME steps
t (every step in [T0, T0 + NT) of every episode) are predicted by
model3.pt with only the last Lh recorded steps as history (window
[t - Lh, t], its first token flagged as a window start, as in training),
Lh = 1, 4, 16, 64, 232. Same targets, only the history length changes, so
the time in the episode cannot confound it. If the network identifies the
system from the history, skill improves with Lh; flat means it predicts
from the current state and the last few errors only.

    python studies/diag_context_m11.py [--model model3.pt] [--cache-name meta2]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from learn.meta import model3 as M

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="model3.pt")
ap.add_argument("--cache-name", default="meta2")
args = ap.parse_args()
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache",
                 args.cache_name)
dev = M.device()
ck = torch.load(os.path.join(C, args.model), weights_only=False)
net = M.Net().to(dev)
net.load_state_dict(ck["net"])
net.eval()
LH = (1, 4, 16, 64, 232)
T0, NT = 240, 48
NAMES = ("surge", "sway", "yaw", "heave", "pitch")
gen = torch.Generator(device=dev).manual_seed(0)
print(f"one-step skill at the same steps t in [{T0}, {T0 + NT}) by the "
      f"history length given (MSE of the sample mean / MSE of 0), "
      f"{args.model}")
for split in ("A", "B", "C", "Cb"):
    D = M.Data3(C, split, dev, stats=ck["stats"])
    ok_ep = torch.nonzero(D.len > T0 + NT + 1)[:, 0]
    se = np.zeros((len(LH), 5))
    s0_ = np.zeros((len(LH), 5))
    with torch.no_grad():
        for li, lh in enumerate(LH):
            for t in range(T0, T0 + NT, 4):
                for b0 in range(0, len(ok_ep), 64):
                    ii = ok_ep[b0:b0 + 64]
                    a = torch.full_like(ii, t - lh)
                    tok, tgt, ok = M.window_tokens(D, ii, a, lh + 1)
                    h = net.encode(tok)[:, -1]
                    y = tgt[:, -1]
                    v = ok[:, -1]
                    s = net.sample(h[v], 32, gen=gen)
                    m = s.mean(1)
                    se[li] += ((m - y[v]) ** 2 - s.var(1) / 32).sum(0)[
                        :5].cpu().numpy()
                    s0_[li] += (y[v] ** 2).sum(0)[:5].cpu().numpy()
    sk = se / np.maximum(s0_, 1e-12)
    print(f"  {split} ({len(ok_ep)} episodes):")
    for li, lh in enumerate(LH):
        print(f"    history {lh:3d} steps: " + " ".join(
            f"{n} {x:.2f}" for n, x in zip(NAMES, sk[li])))
    del D
    torch.cuda.empty_cache()
