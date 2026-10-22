#!/usr/bin/env python3
"""
The best MIXTURE of catalogue laws on the gz target (learn/meta/
ROADMAP_2026-09-30.md section 11.13): the ideal predictor under the family
prior predicts the posterior mean, a convex combination of the laws'
predictions, so its distance to gz is bounded below by the distance from
gz's error to the convex hull of the laws' predictions, not by the best
single law (studies/family_closure.py).

From studies/_cache/family_closure/closure.pkl (every law's teacher-forced
one-step e0 on the gz states, with the true state, sea and the law's own
hidden states): weights w_j >= 0, sum_j w_j = 1, SHARED by the five
channels (a posterior weight belongs to a law, not to a channel), chosen to
minimise sum_c D_c, D_c = sum (e0_gz - sum_j w_j e0_j)^2 / sum e0_gz^2:

  boat     one weight vector for all missions (the posterior over the
           boat's single law), fitted on one half of the missions and
           scored on the other half (both ways), and fitted on all
  mission  one weight vector per mission (in-sample: the most a posterior
           that settles per mission could reach with these draws)

with and without the no-push baseline as an extra candidate. A finite
sample of N laws gives an UPPER bound on the hull distance of the whole
family (more laws can only get closer); the mixture weights are free, which
gives the ideal predictor more than it has (its posterior weights follow
the data and the prior), so 'mission' is an optimistic bound for it.

    python -m studies.family_hull
"""
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
import numpy as np   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(HERE, "_cache", "family_closure", "closure.pkl")
CH = ("surge", "sway", "yaw", "heave", "pitch")


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def simplex_ls(A, y, iters=4000):
    """min ||A w - y||^2 over the probability simplex (projected gradient
    with the exact simplex projection; A (m, J), y (m,))."""
    J = A.shape[1]
    w = np.full(J, 1.0 / J)
    L = np.linalg.norm(A, 2) ** 2 + 1e-12
    for _ in range(iters):
        g = A.T @ (A @ w - y)
        v = w - g / L
        u = np.sort(v)[::-1]
        css = np.cumsum(u) - 1
        ind = np.arange(1, J + 1)
        rho = np.nonzero(u - css / ind > 0)[0][-1]
        w = np.maximum(v - css[rho] / (rho + 1), 0)
    return w


def design(P, e0, rows):
    """Channel-normalised stacked design over the given mission rows."""
    s = 1.0 / np.sqrt((e0[rows] ** 2).sum((0, 1)))
    A = np.stack([(p[rows] * s).reshape(-1) for p in P], 1)
    y = (e0[rows] * s).reshape(-1)
    return A, y


def D_of(P, w, e0, rows):
    pred = sum(wj * p for wj, p in zip(w, P))
    num = ((e0[rows] - pred[rows]) ** 2).sum((0, 1))
    return num / (e0[rows] ** 2).sum((0, 1))


def main():
    import glob
    d = pickle.load(open(PKL, "rb"))
    e0 = np.asarray(d["e0"], float)
    laws = [r for r in d["laws"] if "e0t" in r and r["finite"]]
    for f in sorted(glob.glob(PKL.replace("closure.pkl", "closure_*.pkl"))):
        d2 = pickle.load(open(f, "rb"))
        assert list(d2["keys"]) == list(d["keys"])
        laws += [r for r in d2["laws"] if "e0t" in r and r["finite"]]
    P = [np.asarray(r["e0t"], float) for r in laws]
    B = e0.shape[0]
    log(f"{len(P)} laws, {B} missions x {e0.shape[1]} steps; baseline D "
        f"{np.round(d['base'], 3)}")
    halves = (np.arange(B) < B // 2, np.arange(B) >= B // 2)
    for lab, cands in (("laws", P), ("laws + baseline", P + [d["e0b"]])):
        allr = np.ones(B, bool)
        A, y = design(cands, e0, allr)
        w = simplex_ls(A, y)
        Din = D_of(cands, w, e0, allr)
        top = np.argsort(w)[::-1][:5]
        log(f"[{lab}] boat mixture fitted on all missions: D "
            f"{np.round(Din, 3)}; largest weights "
            + ", ".join(f"{w[j]:.2f}" for j in top))
        cross = []
        for a, b in ((0, 1), (1, 0)):
            A, y = design(cands, e0, halves[a])
            wa = simplex_ls(A, y)
            cross.append(D_of(cands, wa, e0, halves[b]))
        log(f"[{lab}] boat mixture, fitted on one half of the missions, "
            f"scored on the other (mean of both ways): D "
            f"{np.round(np.mean(cross, 0), 3)}")
        per = []
        for b in range(B):
            r = np.zeros(B, bool)
            r[b] = True
            A, y = design(cands, e0, r)
            wb = simplex_ls(A, y)
            per.append(((e0[r] - sum(wj * p for wj, p in zip(wb, cands))[r])
                        ** 2).sum((0, 1)))
        Dm = np.sum(per, 0) / (e0 ** 2).sum((0, 1))
        log(f"[{lab}] one mixture per mission (in-sample): D "
            f"{np.round(Dm, 3)}")


if __name__ == "__main__":
    main()
