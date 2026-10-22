#!/usr/bin/env python3
"""Checks prompted by the independent audit of HULL_FORCE_DERIVATION_2026-10-09.

1. Brard parameter with the encounter frequency, tau_B = U |omega_e| / g,
   instead of U omega / g (derive_regime.py saved the latter). Also the
   following-sea speeds at which omega_e = 0 and the speed band in which
   tau_B < 1/4 (radiated waves run ahead of the hull).
2. The B8 "mlm_only" residual statistics redone with average ranks for ties
   (scipy) and with the non-converged cases removed; plus a main-effects
   split of log(mlm/Sav) over the grid factors C_V, tau, lambda, to see
   whether the data can separate cross-plane gravity (Fn_c ~ C_V tan tau)
   from 3-D gravity (C_V) at all.
Diagnostic only. Output: studies/_cache/derive/audit_checks.txt / .json
"""
import json
import math
import os

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "studies", "_cache", "rankine",
                   "planing_b8_posttier.json")
OUT = os.path.join(ROOT, "studies", "_cache", "derive", "audit_checks")
G, KN = 9.81, 0.514444
TANB = math.tan(math.radians(20.0))


def brard_rows():
    rows = []
    for kn in [5, 10, 15, 20, 25, 30, 40]:
        U = kn * KN
        for Tp in [3.0, 5.0, 8.0, 10.0]:
            w = 2 * math.pi / Tp
            k = w * w / G
            for head, mu in [("head", 0.0), ("beam", 90.0), ("foll", 180.0)]:
                we = w + k * U * math.cos(math.radians(mu))
                rows.append(dict(kn=kn, Tp=Tp, heading=head, omega_e=we,
                                 tau_old=U * w / G, tau_B=U * abs(we) / G,
                                 ahead=(U * abs(we) / G) < 0.25))
    return rows


def following_band(Tp):
    """Following seas: omega_e = w - k U. tau_B = U |w - w^2 U/g| / g < 1/4.
    Returns the zero-encounter speed and the speed band (kn) with tau_B<1/4."""
    w = 2 * math.pi / Tp
    k = w * w / G
    U0 = w / k
    us = np.linspace(0.0, 25.0, 25001)
    tb = us * np.abs(w - k * us) / G
    inside = us[tb < 0.25]
    return dict(Tp=Tp, U0_kn=U0 / KN, band_kn=(inside.min() / KN,
                                               inside.max() / KN))


def b8_stats():
    d = json.load(open(SRC))
    rows = []
    for v in d["V"]:
        cv, tau, lam = v["Cv"], v["tau"], v["lam"]
        t = math.tan(math.radians(tau))
        rows.append(dict(Cv=cv, tau=tau, lam=lam, conv=int(v["conv"]),
                         y=math.log(v["mlm_only"] / v["sav"]),
                         Fn_c=math.sqrt(2.0) * cv * t,
                         stat=0.458 * lam ** 2 / cv ** 2,
                         dry=min(TANB / (math.pi * t * lam), 1.0)))
    out = {}
    for label, sel in [("all24", rows),
                       ("converged", [r for r in rows if r["conv"]])]:
        y = np.array([r["y"] for r in sel])
        res = {}
        for k in ["Fn_c", "stat", "Cv", "tau", "lam", "dry"]:
            x = np.array([r[k] for r in sel])
            rho, p = spearmanr(x, y)
            res[k] = dict(rho=float(rho), p=float(p),
                          distinct=int(len(set(x.tolist()))))
        # main effects (additive) of the three grid factors on log ratio
        yb = y.mean()
        eff, ss = {}, {}
        for k in ["Cv", "tau", "lam"]:
            x = np.array([r[k] for r in sel])
            eff[k] = {float(v): float(y[x == v].mean() - yb)
                      for v in sorted(set(x.tolist()))}
            ss[k] = float(sum(((y[x == v].mean() - yb) ** 2) * (x == v).sum()
                              for v in set(x.tolist())))
        fit = yb + sum(np.array([eff[k][r[k]] for r in sel])
                       for k in ["Cv", "tau", "lam"])
        res["main_effects"] = eff
        res["ss_share"] = {k: ss[k] / float(((y - yb) ** 2).sum())
                           for k in ss}
        res["additive_resid_sd"] = float(np.std(y - fit))
        res["total_sd"] = float(np.std(y))
        res["n"] = len(sel)
        out[label] = res
    return out


def main():
    br = brard_rows()
    bands = [following_band(T) for T in (3.0, 5.0, 8.0, 10.0)]
    st = b8_stats()
    lines = ["Brard parameter with encounter frequency: kn Tp heading "
             "omega_e tau_old(Uw/g) tau_B(U|we|/g) waves-ahead(<1/4)"]
    for r in br:
        lines.append(f"{r['kn']:3d} {r['Tp']:4.1f} {r['heading']:4s} "
                     f"{r['omega_e']:+6.2f} {r['tau_old']:5.2f} "
                     f"{r['tau_B']:5.2f} {'YES' if r['ahead'] else '-'}")
    lines.append("")
    lines.append("following seas: Tp  U(omega_e=0) kn   band with tau_B<1/4 (kn)")
    for b in bands:
        lines.append(f"  {b['Tp']:4.1f}  {b['U0_kn']:6.2f}   "
                     f"{b['band_kn'][0]:5.2f} .. {b['band_kn'][1]:5.2f}")
    lines.append("")
    for label, res in st.items():
        lines.append(f"B8 mlm_only/Sav, {label} (n={res['n']}): Spearman "
                     "(average ranks) rho [p] (distinct values)")
        for k in ["Fn_c", "stat", "Cv", "tau", "lam", "dry"]:
            r = res[k]
            lines.append(f"   {k:5s} {r['rho']:+.2f} [{r['p']:.3f}] "
                         f"({r['distinct']})")
        lines.append("   main effects on log ratio: " + "; ".join(
            f"{k}: " + ", ".join(f"{v:g}:{e:+.3f}" for v, e in
                                 res['main_effects'][k].items())
            for k in ["Cv", "tau", "lam"]))
        lines.append("   share of sum of squares: " + ", ".join(
            f"{k} {s:.2f}" for k, s in res["ss_share"].items())
            + f"; additive-fit residual sd {res['additive_resid_sd']:.3f}"
              f" of total {res['total_sd']:.3f}")
    txt = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT + ".txt", "w").write(txt + "\n")
    json.dump(dict(brard=br, following=bands, b8=st),
              open(OUT + ".json", "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
