#!/usr/bin/env python3
"""
Prior-predictive coverage checks (learn/meta/PRIOR_DERIVATION.md, D1):
does the target's observed error (split C, the full planing plant) look
like a typical draw of the training distribution? Only DESCRIPTIVE
statistics of the data are compared; no reference predictor is fitted.

Per episode, per channel (surge, sway, yaw, heave, pitch):
  log_rms       log RMS of e
  band_lo/mid/hi  share of e's spectrum below 0.7, within 0.7-1.4 and above
                1.4 x the encounter frequency (the centroid of the wave
                points' spectrum, as the boat meets them)
  coh           coherence of e with the first two principal modes of the 15
                wave points (fixed, from the library), mean over the
                encounter band 0.7-1.4 x, the larger of the two
  top1          share of sum(e^2) in the largest 1% of |e|
  q99_q90       ratio of the 99th to the 90th percentile of |e|
  lag1          lag-1 autocorrelation
and per episode: cv_blocks (variation of 10-s block variances), eig1 (top
eigenvalue share of the 5 x 5 correlation).

The target's median over its episodes is ranked among the training
episodes' values: flagged if outside the 1-99% range. A joint rank uses
a robust (median / MAD standardised) Mahalanobis distance.
"""
import os

import numpy as np
from scipy.signal import csd, welch

CH = ("surge", "sway", "yaw", "heave", "pitch")
DT = 0.24


def _modes(lib_path):
    d = np.load(lib_path)
    W = d["S"][..., 11:26].reshape(-1, 15)
    W = W - W.mean(0)
    _, _, Vt = np.linalg.svd(W, full_matrices=False)
    return Vt[:2], d["mu"][11:26], d["sd"][11:26]


def episode_stats(E, S, modes):
    """E (T, 5) observed error, S (T, 26) raw inputs -> dict of stats."""
    Vt, mu, sd = modes
    fs = 1.0 / DT
    W = (S[:, 11:] - mu) / sd
    f, Pw = welch(W, fs=fs, nperseg=min(128, len(W)), axis=0)
    Pm = Pw.mean(1)
    w = 2 * np.pi * f
    we = (w[1:] * Pm[1:]).sum() / max(Pm[1:].sum(), 1e-12)
    f, Pe = welch(E - E.mean(0), fs=fs, nperseg=min(128, len(E)), axis=0)
    w = 2 * np.pi * f
    tot = Pe[1:].sum(0) + 1e-20
    lo = Pe[1:][w[1:] < 0.7 * we].sum(0) / tot
    mid = Pe[1:][(w[1:] >= 0.7 * we) & (w[1:] < 1.4 * we)].sum(0) / tot
    hi = Pe[1:][w[1:] >= 1.4 * we].sum(0) / tot
    band = (w >= 0.7 * we) & (w < 1.4 * we)
    K = W @ Vt.T
    coh = np.zeros(5)
    for c in range(5):
        best = 0.0
        for m in range(2):
            _, Pxy = csd(E[:, c], K[:, m], fs=fs, nperseg=min(128, len(E)))
            _, Pxx = welch(E[:, c], fs=fs, nperseg=min(128, len(E)))
            _, Pyy = welch(K[:, m], fs=fs, nperseg=min(128, len(E)))
            ch = np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-20)
            if band.any():
                best = max(best, float(ch[band].mean()))
        coh[c] = best
    a = np.abs(E)
    top1 = np.zeros(5)
    q = np.zeros(5)
    lag1 = np.zeros(5)
    for c in range(5):
        s = np.sort(a[:, c] ** 2)[::-1]
        n1 = max(1, int(0.01 * len(s)))
        top1[c] = s[:n1].sum() / max(s.sum(), 1e-20)
        q[c] = np.quantile(a[:, c], 0.99) / max(np.quantile(a[:, c], 0.9),
                                                1e-12)
        x = E[:, c] - E[:, c].mean()
        lag1[c] = (x[1:] @ x[:-1]) / max(x @ x, 1e-20)
    nb = int(round(10.0 / DT))
    blocks = [E[i:i + nb].var(0).sum() for i in range(0, len(E) - nb + 1,
                                                       nb)]
    cv = float(np.std(blocks) / max(np.mean(blocks), 1e-20))
    Cr = np.corrcoef((E - E.mean(0)).T + 1e-12 * np.random.default_rng(
        0).normal(size=E.T.shape))
    eig = np.linalg.eigvalsh(np.nan_to_num(Cr))
    out = dict(cv_blocks=cv, eig1=float(eig[-1] / 5.0),
               omega_e=float(we))
    for c, nm in enumerate(CH):
        out.update({f"log_rms_{nm}": float(np.log(np.sqrt(
            (E[:, c] ** 2).mean()) + 1e-12)),
            f"band_lo_{nm}": float(lo[c]), f"band_mid_{nm}": float(mid[c]),
            f"band_hi_{nm}": float(hi[c]), f"coh_{nm}": coh[c],
            f"top1_{nm}": top1[c], f"q99_q90_{nm}": q[c],
            f"lag1_{nm}": lag1[c]})
    return out


def split_stats(cache, split, modes, n_max=None,
                skip=int(round(10.0 / DT))):
    d = np.load(os.path.join(cache, f"{split}.npz"))
    n = len(d["len"]) if n_max is None else min(n_max, len(d["len"]))
    rows = []
    for i in range(n):
        L = int(d["len"][i])
        if L < 100:
            continue
        e = d["E"][i, skip:L].astype(np.float64)
        if np.sqrt((e ** 2).mean(0)).max() < 1e-9:
            continue                    # a null operator: no error at all
        rows.append(episode_stats(d["E"][i, skip:L], d["S"][i, skip:L],
                                  modes))
    return rows


def report(cache, n_train=1000):
    modes = _modes(os.path.join(cache, "lib.npz"))
    tr = split_stats(cache, "train", modes, n_max=n_train)
    tg = split_stats(cache, "C", modes)
    keys = [k for k in tr[0] if k != "omega_e"]
    T = np.array([[r[k] for k in keys] for r in tr])
    G = np.array([[r[k] for k in keys] for r in tg])
    med = np.median(G, 0)
    lines = [f"training episodes {len(T)}, target episodes {len(G)}; "
             f"encounter frequency (median) train "
             f"{np.median([r['omega_e'] for r in tr]):.2f}, target "
             f"{np.median([r['omega_e'] for r in tg]):.2f} rad/s",
             f"{'statistic':<22}{'target':>9}{'train 5%':>10}"
             f"{'50%':>8}{'95%':>8}{'rank':>7}"]
    flags = []
    for j, k in enumerate(keys):
        rank = float((T[:, j] < med[j]).mean())
        p5, p50, p95 = np.quantile(T[:, j], [0.05, 0.5, 0.95])
        flag = " <--" if (rank < 0.01 or rank > 0.99) else ""
        if flag:
            flags.append(k)
        lines.append(f"{k:<22}{med[j]:>9.3f}{p5:>10.3f}{p50:>8.3f}"
                     f"{p95:>8.3f}{rank:>7.2f}{flag}")
    m0 = np.median(T, 0)
    mad = np.median(np.abs(T - m0), 0) * 1.4826 + 1e-9
    Z = (T - m0) / mad
    Cv = np.cov(np.clip(Z, -10, 10).T) + 1e-3 * np.eye(len(keys))
    Pi = np.linalg.inv(Cv)
    dt = np.einsum("ij,jk,ik->i", np.clip(Z, -10, 10), Pi,
                   np.clip(Z, -10, 10))
    # like with like: every target EPISODE on the training episodes' scale
    Zg = np.clip((G - m0) / mad, -10, 10)
    dgi = np.einsum("ij,jk,ik->i", Zg, Pi, Zg)
    lines.append(
        "joint (robust Mahalanobis, per episode): share of training episodes"
        f" farther than the median target episode "
        f"{float((dt > np.median(dgi)).mean()):.2f} (0.5 = typical); share "
        f"of target episodes beyond the training 99% distance "
        f"{float((dgi > np.quantile(dt, 0.99)).mean()):.2f} (0.01 = typical)")
    lines.append("flagged: " + (", ".join(flags) if flags else "none"))
    return "\n".join(lines)
