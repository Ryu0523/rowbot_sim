#!/usr/bin/env python3
"""Which nondimensional group orders the residual of the repo's steady
planing computations (stage B7 linear 3-D, stage B8 composites) against
Savitsky? Diagnostic only (rank correlations; no fitted model).
Input: studies/_cache/rankine/planing_b8_posttier.json (24 cases, beta 20).
Output: studies/_cache/derive/b8_residual.txt
"""
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "studies", "_cache", "rankine",
                   "planing_b8_posttier.json")
OUT = os.path.join(ROOT, "studies", "_cache", "derive", "b8_residual.txt")
TANB = math.tan(math.radians(20.0))


def rank(a):
    o = np.argsort(a)
    r = np.empty(len(a))
    r[o] = np.arange(len(a))
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main():
    d = json.load(open(SRC))
    rows = []
    for v in d["V"]:
        cv, tau, lam = v["Cv"], v["tau"], v["lam"]
        t = math.tan(math.radians(tau))
        rows.append(dict(
            Cv=cv, tau=tau, lam=lam, conv=v["conv"],
            lin=v["lin"] / v["sav"], mlm=v["mlm_only"] / v["sav"],
            comp=v["comp"] / v["sav"],
            Fn_c=math.sqrt(2.0) * cv * t,                # V/sqrt(g b/2)
            stat=0.458 * lam ** 2 / cv ** 2,             # Savitsky term ratio
            dry=min(TANB / (math.pi * t * lam), 1.0),    # chine-dry share
            eps=1.0 / lam))
    keys = ["Fn_c", "stat", "Cv", "tau", "lam", "dry"]
    lines = ["Cv tau lam conv | lin/S mlm/S comp/S | Fn_c stat/dyn dry"]
    for r in rows:
        lines.append(f"{r['Cv']:.0f} {r['tau']:.0f} {r['lam']:.1f} "
                     f"{int(r['conv'])} | {r['lin']:.3f} {r['mlm']:.3f} "
                     f"{r['comp']:.3f} | {r['Fn_c']:.3f} {r['stat']:.3f} "
                     f"{r['dry']:.2f}")
    for m in ["lin", "mlm", "comp"]:
        y = np.log([r[m] for r in rows])
        s = ", ".join(f"{k} {spearman(y, np.array([r[k] for r in rows])):+.2f}"
                      for k in keys)
        lines.append(f"Spearman(log {m}/Sav, .): {s}")
    # how much of the mlm spread is ordered by Fn_c alone: residual spread
    # within Fn_c bins vs total
    y = np.log([r["mlm"] for r in rows])
    f = np.array([r["Fn_c"] for r in rows])
    o = np.argsort(f)
    ys = y[o]
    within = np.std(np.diff(ys)) / math.sqrt(2)
    lines.append(f"mlm: sd(log ratio) total {np.std(y):.3f}; "
                 f"neighbour-difference sd after ordering by Fn_c "
                 f"{within:.3f}")
    for k in ["stat", "lam", "tau"]:
        g = np.array([r[k] for r in rows])
        ys = y[np.argsort(g)]
        lines.append(f"   ordering by {k}: {np.std(np.diff(ys))/math.sqrt(2):.3f}")
    txt = "\n".join(lines)
    open(OUT, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
