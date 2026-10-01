#!/usr/bin/env python3
"""
Why is pitch not learned on the target (C / Cb one-step skill ~0.91 against
~0.66 in family)? Descriptive, like diag_yaw_m8.py:

  1  the character of the pitch-rate error (e 4) and the pitch-angle error
     (e 6): size, step-to-step correlation, where its power sits in
     frequency, the share of the largest 1%, for A (source) and C / Cb
     (target)
  2  what it goes with (correlation; the other quantity leading by 0 / 1 /
     3 steps): the wave slope along the hull (bow minus stern centre
     elevation at the MPC's stations), the mean elevation, heave error,
     states (u, w, q, z, theta, roll, roll rate), actuators
  3  model3.pt's one-step prediction on Cb against the truth, and the
     same on A (the network is what is evaluated; no reference predictor)

    python studies/diag_pitch_m11.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache", "meta2")


def load(split):
    d = np.load(os.path.join(C, f"{split}.npz"))
    e = np.load(os.path.join(C, f"{split}_e0.npz"))["E0"]
    L = d["len"]
    T = d["U"].shape[1]
    n_e = L - 1 if d["XS"].shape[1] == T else L
    return d, e, n_e


def rows_of(split, n_max=300):
    d, e, n_e = load(split)
    out = []
    for i in range(min(n_max, len(n_e))):
        n = int(n_e[i])
        if n < 80:
            continue
        s, xs, ee = d["S"][i, 40:n], d["XS"][i, 40:n], e[i, 40:n]
        cen = s[:, 11 + 1::3]                      # centre line, stern..bow
        out.append(dict(
            thd=ee[:, 4], th=ee[:, 6], zd_e=ee[:, 3], z_e=ee[:, 5],
            slope=cen[:, -1] - cen[:, 0], elev=cen.mean(1),
            u=xs[:, 6], w=xs[:, 8], q=xs[:, 10], z=xs[:, 2], theta=xs[:, 4],
            roll=xs[:, 3], roll_rate=xs[:, 9], thr=xs[:, 12], noz=xs[:, 13]))
    return out


def corr_lag(rows, a, b, lag):
    xs, ys = [], []
    for r in rows:
        x, y = r[a], r[b]
        if lag:
            x, y = x[:-lag], y[lag:]
        xs.append(x - x.mean())
        ys.append(y - y.mean())
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float(x @ y / np.sqrt((x @ x) * (y @ y) + 1e-30))


def ac(x, lag):
    x = x - x.mean()
    return float(x[lag:] @ x[:-lag] / max(x @ x, 1e-30))


DT = 0.24
print("1  character of the pitch errors")
for split in ("A", "C", "Cb"):
    R = rows_of(split)
    for ch in ("thd", "th"):
        y = np.concatenate([r[ch] for r in R])
        top = np.sort(np.abs(y))[::-1]
        share = (top[:max(1, len(top) // 100)] ** 2).sum() / (top ** 2).sum()
        P = np.mean([np.abs(np.fft.rfft(r[ch][:200] - r[ch][:200].mean()))
                     ** 2 for r in R if len(r[ch]) >= 200], 0)
        f = np.fft.rfftfreq(200, DT)
        cum = np.cumsum(P) / P.sum()
        f50 = f[np.searchsorted(cum, 0.5)]
        print(f"   {split:<3} {ch:<4} rms {np.sqrt((y ** 2).mean()):.3f}, "
              f"step-to-step correlation lag 1 / 2 / 4: "
              f"{np.mean([ac(r[ch], 1) for r in R]):+.2f} / "
              f"{np.mean([ac(r[ch], 2) for r in R]):+.2f} / "
              f"{np.mean([ac(r[ch], 4) for r in R]):+.2f}; half the power "
              f"below {f50:.2f} Hz (peak {f[np.argmax(P[1:]) + 1]:.2f} Hz); "
              f"largest 1% share {share:.2f}")

print("\n2  correlation of the pitch-rate error with ... (leading by 0 / 1 / 3"
      " steps)")
names = ["slope", "elev", "zd_e", "u", "w", "q", "z", "theta", "roll",
         "roll_rate", "thr", "noz"]
for split in ("A", "Cb"):
    R = rows_of(split)
    print(f"   {split}:")
    for nm in names:
        v = [corr_lag(R, nm, "thd", lg) for lg in (0, 1, 3)]
        print(f"     {nm:<10} " + " ".join(f"{x:+.2f}" for x in v))

print("\n3  model3.pt one-step prediction of the pitch errors")
import torch   # noqa: E402

from learn.meta import model3 as M   # noqa: E402
dev = M.device()
ck = torch.load(os.path.join(C, "model3.pt"), weights_only=False)
net = M.Net().to(dev)
net.load_state_dict(ck["net"])
net.eval()
gen = torch.Generator(device=dev).manual_seed(0)
for split in ("A", "Cb"):
    D = M.Data3(C, split, dev, stats=ck["stats"])
    P, Y = [], []
    with torch.no_grad():
        for s0 in range(0, min(D.n, 64), 16):
            ii = torch.arange(s0, min(D.n, s0 + 16), device=dev)
            a = torch.full_like(ii, 0)
            tok, tgt, ok = M.window_tokens(D, ii, a, M.W_CTX)
            h = net.encode(tok)
            sel = ok & (torch.arange(M.W_CTX, device=dev)[None] >= 40)
            s = net.sample(h[sel], 32, gen=gen).mean(1)
            P.append(s.cpu().numpy())
            Y.append(tgt[sel].cpu().numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    for c, nm in ((4, "pitch rate"), (6, "pitch angle"), (2, "yaw rate"),
                  (3, "heave rate")):
        print(f"   {split:<3} {nm:<11} corr(predicted mean, truth) "
              f"{np.corrcoef(P[:, c], Y[:, c])[0, 1]:+.2f}; rms of the "
              f"predicted mean / truth {np.sqrt((P[:, c] ** 2).mean()):.2f} / "
              f"{np.sqrt((Y[:, c] ** 2).mean()):.2f} (training sd units)")
