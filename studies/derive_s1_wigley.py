#!/usr/bin/env python3
"""
S1 step 2b: the Wigley I / Journee (1992) validation rows (ledger B6 "Vd", the quick-tier configuration COARSE: GRID_BASE
M 40, hull 60 x 12, Fn 0.2 and 0.3) and the Timman-Newman / Haskind / spectrum implementation rows, with the production
kinematic operator ("cons") and the opt-in edge-flux pair ("flux"), side by side. Reuses the row functions of
studies/rankine_db_verify.py unchanged, by redirecting its solver builder `_db` to the requested operator.

Output: studies/_cache/derive/s1_wigley.txt, s1_wigley.json, s1_wigley.log
"""
import datetime
import functools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "_cache", "derive")
os.makedirs(OUT, exist_ok=True)

import studies.rankine_db_verify as dbv  # noqa: E402
from studies.rankine_unsteady_verify import COARSE, CHECK_GRID, SMALL_GRID  # noqa: E402

LOGF = open(os.path.join(OUT, "s1_wigley.log"), "a", encoding="utf-8")


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOGF.write(line + "\n")
    LOGF.flush()


def mem():
    import psutil
    import torch
    return psutil.Process().memory_info().rss / 1e9, (torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0)


ORIG_DB = dbv._db


def with_kinematic(kin):
    """Force the operator: _pm passes kinematic="cons" explicitly, so a partial would be overridden (first run of
    S1: the TN/HK rows of 'flux' were in fact cons). The TNn row (kinematic 'nat') then also uses `kin`."""
    if kin == "cons":
        return ORIG_DB

    def forced(*a, **k):
        k = dict(k)
        k["kinematic"] = kin
        return ORIG_DB(*a, **k)
    return forced


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tn_only = "--tn-only" in sys.argv
    suffix = "_tn" if tn_only else ""
    kinds = args or ["cons", "flux"]
    res = {}
    t_all = time.time()
    for kin in kinds:
        dbv._db = with_kinematic(kin)
        out = {}
        t0 = time.time()
        rows = [] if tn_only else dbv.validation((COARSE,))
        out["V"] = rows
        log(f"{kin}: validation rows {len(rows)} in {time.time() - t0:.0f} s; pass {sum(r['ok'] for r in rows)}/{len(rows)} "
            f"(binding {sum(1 for r in rows if r['tol'] != 'info' and 'info' not in str(r['tol']))})")
        t0 = time.time()
        rows = dbv.tn_wigley((0.3, (1.0,), (2.212,), dict(CHECK_GRID, M=30), (60, 12)))
        out["TN"] = rows
        for r in rows:
            log(f"  {r['check']:4s} {r['value']:.3e} tol {r['tol']} {'PASS' if r['ok'] else 'FAIL'} | {r['quantity'][:90]}")
        log(f"{kin}: TN/FSR/TK/HK rows in {time.time() - t0:.0f} s")
        t0 = time.time()
        rows = dbv.spectrum(((("wigleyJ", 0.3, True), ("prismV", 0.4, True)), dict(SMALL_GRID, M=16), (16, 4)))
        out["SP"] = rows
        for r in rows:
            log(f"  {r['check']:4s} {r['value']:+.3e} {'PASS' if r['ok'] else 'FAIL'} | {r['quantity'][:90]} | {r.get('note', '')}")
        log(f"{kin}: SP rows in {time.time() - t0:.0f} s")
        rss, gpu = mem()
        log(f"{kin}: RSS {rss:.2f} GB, GPU peak {gpu:.2f} GB")
        res[kin] = out
    dbv._db = ORIG_DB
    # side-by-side table of the validation rows
    lines = []
    if len(kinds) >= 2 and all(k in res for k in kinds):
        k0 = kinds[0]
        lines.append(f"{'row':70s} " + " ".join(f"{k:>12s}" for k in kinds) + f" {'exp':>12s} {'tol':>22s}")
        for i, r in enumerate(res[k0]["V"]):
            vals = [res[k]["V"][i]["value"] for k in kinds]
            oks = ["P" if res[k]["V"][i]["ok"] else "F" for k in kinds]
            lines.append(f"{r['quantity'][:70]:70s} " + " ".join(f"{v:12.4f}" for v in vals) + f" {r['ref']:12.4f} {str(r['tol'])[:22]:>22s} " + "/".join(oks))
        for fam in ("TN", "SP"):
            for i, r in enumerate(res[k0][fam]):
                vals = [res[k][fam][i]["value"] for k in kinds]
                oks = ["P" if res[k][fam][i]["ok"] else "F" for k in kinds]
                lines.append(f"{(r['check'] + ': ' + r['quantity'])[:70]:70s} " + " ".join(f"{v:12.3e}" for v in vals) + f" {'':12s} {str(r['tol'])[:22]:>22s} " + "/".join(oks))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, f"s1_wigley{suffix}.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    with open(os.path.join(OUT, f"s1_wigley{suffix}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(res=res, t_total=time.time() - t_all), fh, indent=1, default=str)
    log(f"total {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
