#!/usr/bin/env python3
"""
S1 stage 2 (S1b) tables, assembled from the json records of derive_s1_scarab.py (stage 1: flux operator on the stage-1
grid), derive_s1b_scarab.py (stage 2: flux + grid fixes [+ near-wall sampling rule]) and derive_s1b_nk.py (Neumann-Kelvin).

  1. Convergence tables coarse / base / fine for the stage-2 main sequence next to the stage-1 flux sequence, with
     r = (base - coarse)/(fine - base) and the relative base -> fine change d = |fine - base|/|fine|, and a verdict per
     quantity: converged (d <= 5 % and |r| >= 2, or d <= 2 %), improved (d less than 0.8 x the stage-1 d), unchanged
     (within 0.8-1.25 x), worse (> 1.25 x).
  2. Base-grid sensitivity to the options (grid variant, kappa, forcing flag).
  3. Overlap check NK vs double-body at 5 kn (base grid): A_inf, B_inf, static stiffness (NK C0 vs DB C0 + C_s),
     radiation and excitation at the four wave cases, passivity, steady Fz / My.

Output: studies/_cache/derive/s1b_tables.txt (+ s1b_tables.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "_cache", "derive")
sys.path.insert(0, HERE)
from derive_s1_tables import quantities, cx, ratio, GR, CASES  # noqa: E402

LINES = []


def log(s=""):
    print(s)
    LINES.append(s)


def load(name):
    p = os.path.join(OUT, name + ".json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def verdict(v1, v2):
    """v1, v2: (coarse, base, fine) of stage 1 and stage 2 -> (d1, d2, r2, text)."""
    def d(v):
        if v is None or any(x is None for x in v):
            return None
        return abs(v[2] - v[1]) / max(abs(v[2]), 1e-300)
    d1, d2 = d(v1), d(v2)
    r2 = ratio(v2) if v2 is not None and all(x is not None for x in v2) else None
    if d2 is None:
        return d1, d2, r2, "---"
    if d2 <= 0.02 or (d2 <= 0.05 and r2 is not None and abs(r2) >= 2):
        t = "converged"
    elif d1 is None:
        t = "not conv."
    elif d2 < 0.8 * d1:
        t = "improved"
    elif d2 <= 1.25 * d1:
        t = "unchanged"
    else:
        t = "worse"
    return d1, d2, r2, t


def fmtv(v, fmt):
    return " ".join(fmt.format(x) if x is not None else " " * 12 for x in v)


def seq_table(title, rec1, rec2, names=None):
    Q1 = {g: quantities(rec1[g]) for g in GR}
    Q2 = {g: quantities(rec2[g]) for g in GR}
    names = names or list(Q2["base"].keys())
    log(title)
    log(f"{'quantity':44s} |{'stage-1 flux coarse/base/fine':>40s}  r      d   |{'stage 2 coarse/base/fine':>40s}  r      d    verdict")
    out = {}
    for nm in names:
        v1 = [Q1[g].get(nm) for g in GR]
        v2 = [Q2[g].get(nm) for g in GR]
        fmt = "{:12.4f}" if ("B_sym" in nm or "|W|/U" in nm) else "{:12.0f}"
        d1, d2, r2, t = verdict(v1, v2)
        r1 = ratio(v1) if all(x is not None for x in v1) else None

        def rs(r):
            return f"{r:+6.1f}" if r is not None and np.isfinite(r) else "   ---"

        def ds(d):
            return f"{100 * d:5.1f}%" if d is not None else "   ---"
        log(f"{nm:44s} |{fmtv(v1, fmt)} {rs(r1)} {ds(d1)} |{fmtv(v2, fmt)} {rs(r2)} {ds(d2)}  {t}")
        out[nm] = dict(stage1=v1, stage2=v2, d1=d1, d2=d2, r2=r2, verdict=t)
    return out


def main():
    res = {}
    s1 = {g: load(f"s1_scarab_flux_{g}") for g in GR}
    main_lab = "esg_nw0.5f"
    s2 = {g: load(f"s1b_scarab_{main_lab}_{g}") for g in GR}
    s2n = {g: load(f"s1b_scarab_esg_{g}") for g in GR}
    for lab, rr in (("stage 2 main (esg + near-wall 0.5 + forcing)", s2), ("stage 2 grid fixes only (esg)", s2n)):
        for g in GR:
            r = rr[g]
            if r:
                nw = r.get("near_wall", {})
                log(f"  {lab} {g:6s}: NB {r['NB']}, NF {r['NF']}, grid {r['shape']}, J {r['J']}, build {r['t_build']:.0f} s, total {r['t_total']:.0f} s, "
                    f"RSS {r['peak_rss_gb']:.2f} GB, GPU {r['peak_gpu_gb']:.2f} GB; |W|/U hull {r['W_hull_max_over_U']:.2f} (x_frac {r['W_hull_max_at']['x_frac']:.4f}), "
                    f"fs centroids {r['W_fs']['max']:.2f} (x_frac {r['W_fs']['x_frac']:.3f}), Gauss {r['W_fs']['max_gauss']:.2f} (x_frac {r['W_fs']['gauss_at']['x_frac']:.4f}, "
                    f"y {r['W_fs']['gauss_at']['y']:.4f})" + (f"; near-wall: Gauss moved {nw.get('gauss_moved')}/{nw.get('gauss_total')}, max shift {100 * nw.get('gauss_shift_max', 0):.1f} cm, "
                                                            f"centroids {nw.get('centroids')}" if nw else ""))
    for g in GR:
        r = s1[g]
        if r:
            log(f"  stage-1 flux {g:6s}: NB {r['NB']}, NF {r['NF']}, |W|/U hull {r['W_hull_max_over_U']:.2f}, fs centroids {r['W_fs']['max']:.2f}, Gauss {r['W_fs'].get('max_gauss', float('nan')):.2f}")
    log()
    if all(s2.values()):
        res["main"] = seq_table("1a. scarab195 5 kn, coarse (M 32) / base (M 48) / fine (M 64): stage-1 flux (old grid) | stage 2 = flux + grid fixes (end width, "
                                "end sub-columns 2/4, stern knuckle node, graded band) + near-wall sampling kappa 0.5 incl. forcing", s1, s2)
        log()
    if all(s2n.values()):
        res["grid_only"] = seq_table("1b. the same with the grid fixes only (no near-wall sampling rule)", s1, s2n)
        log()
    # steady
    log("1c. steady double-body solution (Fz, My about the CG; N, N m) per grid:")
    for lab, rr in (("stage 2 main", s2), ("grid fixes only", s2n)):
        for g in GR:
            r = rr[g]
            if r and "steady" in r:
                st = r["steady"]
                log(f"  {lab:16s} {g:6s}: Fz {st['Fz']:9.1f} (base-pressure part {st['Fz_db']:9.1f}), My {st['My']:9.1f} ({st['My_db']:9.1f}), R_lin {st['R_lin']:8.1f}")
    log()
    # Haskind
    for lab, rr in (("stage 2 main", s2), ("stage-1 flux", s1)):
        r = rr["base"]
        if r and "haskind" in r:
            h = r["haskind"]
            log(f"Haskind / Timman-Newman with the -U system on the same grid ({lab} base; same panels {h['same_panels']}):")
            for key, c in h["cases"].items():
                log(f"  {key}: |X_D^OT - Haskind|/|X| per mode (1,3,5) {np.round(c['haskind_rel'], 4).tolist()}; classical-only "
                    f"{np.round(c['haskind_classical_rel'], 3).tolist()}; TN pairs (35,13,15,53) {np.round(c['tn_rel'], 4).tolist()}")
    log()
    # 2. base-grid sensitivity
    log("2. base grid: sensitivity to the options (quantities per record)")
    variants = [("stage-1 flux (old grid)", s1["base"]), ("esg", s2n["base"]), ("esg nw0.5", load("s1b_scarab_esg_nw0.5_base")),
                ("esg nw0.5f (main)", s2["base"]), ("esg nw0.25f", load("s1b_scarab_esg_nw0.25f_base")), ("es nw0.5f (no graded band)", load("s1b_scarab_es_nw0.5f_base")),
                ("g nw0.5f (graded band only, no end fixes)", load("s1b_scarab_g_nw0.5f_base")),
                ("kn nw0.5f (knuckle node only + rule)", load("s1b_scarab_kn_nw0.5f_base"))]
    variants = [(l, r) for l, r in variants if r]
    keys = ["C0 + C_s heave", "C0 + C_s pitch", "A33 head_Tp3 (we 3.24)", "B33 head_Tp3 (we 3.24)", "A55 head_Tp3 (we 3.24)", "B55 head_Tp3 (we 3.24)",
            "|X3| total head_Tp3 (we 3.24)", "|X5|/k total head_Tp3 (we 3.24)", "|X3| total head_Tp5 (we 1.67)", "|X5|/k total head_Tp5 (we 1.67)",
            "|X3| total follow_Tp3 (we 0.94)", "|X5|/k total follow_Tp3 (we 0.94)", "|X3| total follow_Tp5 (we 0.84)", "|X5|/k total follow_Tp5 (we 0.84)",
            "B33 follow_Tp3 (we 0.94)", "B55 follow_Tp5 (we 0.84)", "B_sym worst (all frequencies)", "|W|/U max free surface (centroids)"]
    Qv = [(l, quantities(r)) for l, r in variants]
    hdr = f"{'quantity':40s}" + "".join(f"|{l[:22]:>22s}" for l, _ in Qv)
    log(hdr)
    for k in keys:
        fmt = "{:22.4f}" if ("B_sym" in k or "|W|/U" in k) else "{:22.0f}"
        log(f"{k:40s}" + "".join("|" + (fmt.format(q.get(k)) if q.get(k) is not None else " " * 22) for _, q in Qv))
    res["variants_base"] = {l: {k: q.get(k) for k in keys} for l, q in Qv}
    log()
    # 3. NK overlap
    log("3. overlap check at 5 kn (Fn_L 0.35), base grid: Neumann-Kelvin (uniform stream, wall-trace closure) vs double body (stage 2 main and grid-only)")
    nk = {"NK trace esg": load("s1b_nk_esg_base"), "NK rule esg": load("s1b_nk_esg_rule_base"), "NK trace kn": load("s1b_nk_kn_base")}
    nk = {k: v for k, v in nk.items() if v and "failed" not in v}
    for k, v in nk.items():
        log(f"  {k}: NB {v['NB']}, NF {v['NF']}, wall nodes {v['n_wall']}, build {v['t_build']:.0f} s, total {v['t_total']:.0f} s, GPU {v['peak_gpu_gb']:.2f} GB")
    cols = [("DB stage 2 main", s2["base"]), ("DB grid only", s2n["base"]), ("DB stage-1 flux", s1["base"])] + list(nk.items())
    cols = [(l, r) for l, r in cols if r]

    def q_of(r):
        if "W_fs" not in r:                      # NK records carry no base flow
            r = dict(r, W_fs=dict(max=float("nan")), W_hull_max_over_U=float("nan"))
        q = quantities(r)
        C0, Cs = np.array(r["C0"]), np.array(r["C_s"])
        q["static heave total (C0 + C_s | NK C0)"] = C0[1, 1] + Cs[1, 1]
        q["static pitch total (C0 + C_s | NK C0)"] = C0[2, 2] + Cs[2, 2]
        q["static (3,5) total"] = C0[1, 2] + Cs[1, 2]
        q["static (5,3) total"] = C0[2, 1] + Cs[2, 1]
        if "steady" in r:
            q["steady Fz (N)"] = r["steady"]["Fz"]
            q["steady My (N m)"] = r["steady"]["My"]
        if r.get("lin") == "NK":
            q["C0 pitch OT form"] = np.array(r["C0_ot"])[2, 2]
        return q
    Qo = [(l, q_of(r)) for l, r in cols]
    keys3 = ["A_inf heave (kg)", "A_inf pitch (kg m2)", "B_inf (3,5)", "B_inf (5,3)", "static heave total (C0 + C_s | NK C0)", "static pitch total (C0 + C_s | NK C0)",
             "static (3,5) total", "static (5,3) total", "steady Fz (N)", "steady My (N m)"]
    for key in CASES:
        we = None
        for _, r in cols:
            if r and key in r["cases"]:
                we = r["cases"][key]["omega_e"]
                break
        tagc = f"{key} (we {we:.2f})"
        keys3 += [f"A33 {tagc}", f"B33 {tagc}", f"A55 {tagc}", f"B55 {tagc}", f"|X3| total {tagc}", f"|X3| diffraction {tagc}", f"|X5|/k total {tagc}",
                  f"|X5|/k diffraction {tagc}", f"B_sym min/max {tagc}"]
    keys3 += ["B_sym worst (all frequencies)"]
    log(f"{'quantity':44s}" + "".join(f"|{l[:18]:>18s}" for l, _ in Qo) + "|  NK/DB(main)")
    res["overlap"] = {}
    for k in keys3:
        fmt = "{:18.4f}" if "B_sym" in k else "{:18.0f}"
        vals = [q.get(k) for _, q in Qo]
        rat = ""
        if len(vals) >= 2 and vals[0] not in (None, 0) and nk:
            qn = Qo[len(cols) - len(nk)][1]
            v_nk = qn.get(k)
            if v_nk is not None and abs(vals[0]) > 1e-300:
                rat = f"  {v_nk / vals[0]:+8.3f}"
        log(f"{k:44s}" + "".join("|" + (fmt.format(v) if v is not None else " " * 18) for v in vals) + rat)
        res["overlap"][k] = {l: q.get(k) for l, q in Qo}
    # NK convergence (coarse / base / fine) if present
    nkg = {g: load(f"s1b_nk_esg_{g}") for g in GR}
    if all(v and "failed" not in v for v in nkg.values()):
        log("\n3b. NK (trace, esg grids) coarse / base / fine:")
        Qn = {g: q_of(nkg[g]) for g in GR}
        for k in keys3:
            v = [Qn[g].get(k) for g in GR]
            if all(x is not None for x in v):
                fmt = "{:12.4f}" if "B_sym" in k else "{:12.0f}"
                r_ = ratio(v)
                log(f"  {k:44s} {fmtv(v, fmt)}  r {r_:+7.2f}" if r_ is not None and np.isfinite(r_) else f"  {k:44s} {fmtv(v, fmt)}")
    with open(os.path.join(OUT, "s1b_tables.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(os.path.join(OUT, "s1b_tables.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
