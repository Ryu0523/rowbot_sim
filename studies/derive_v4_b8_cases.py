#!/usr/bin/env python3
"""V4 of HULL_FORCE_AUDIT_REVIEW_2026-10-09.md section 4 (check C3 of the derivation, section 8):
extra steady planing cases that break the confounding of the cross-plane Froude number
Fn_c = sqrt(2) C_V tan(tau) with the speed coefficient C_V on the B8 3x3x3 grid.

Solver and discretisation: identical to the post-tier B8 run (studies/_cache/rankine/planing_b8_posttier.json):
PlaningSprayRootFast(Cv, beta 20, h = 1/32, ns = 32 lateral intervals, cluster 'uni', lateral 'hat0', lam_max 6,
Richardson damping extrapolation), solve_lambda(tau, lam) with the default max_iter 200, composite of
hydro/rankine/planing_nl.py with the registered variants of studies/rankine_planing_b8.py (mlm_only, sep_only,
anchor_wen, v_slender), Savitsky (1964) lift at the same (tau, lam, C_V, beta).

    .venv/Scripts/python.exe studies/derive_v4_b8_cases.py run       # compute the new cases (incremental JSON)
    .venv/Scripts/python.exe studies/derive_v4_b8_cases.py analyse   # statistics on old (post-tier) + new cases

Outputs: studies/_cache/derive/v4_cases.json (new cases, with timing and memory), v4_stats.txt / v4_stats.json.
Diagnostic only; nothing under sim/ or hydro/ is changed.
"""
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUTDIR = os.path.join(ROOT, "studies", "_cache", "derive")
CASES_JSON = os.path.join(OUTDIR, "v4_cases.json")
OLD_JSON = os.path.join(ROOT, "studies", "_cache", "rankine", "planing_b8_posttier.json")
STATS = os.path.join(OUTDIR, "v4_stats")
BETA = 20.0
LAM = 2.5
U_NUM = 0.024          # B8 PC1u: lateral bias of the production 3-D linear part (PL1B), relative

# (C_V, tau deg): three equal-Fn_c groups plus wide-tau points at fixed C_V
NEW_CASES = [
    (8.0, 1.5),   # Fn_c 0.296: with (6, 2) and (2, 6) of the grid and (3, 4) below
    (3.0, 4.0),   # Fn_c 0.297
    (8.0, 2.0),   # Fn_c 0.395: with (4, 4) of the grid and (2, 8) below
    (2.0, 8.0),   # Fn_c 0.397
    (8.0, 3.0),   # Fn_c 0.593: with (4, 6) of the grid and (2.5, 9.5) below
    (2.5, 9.5),   # Fn_c 0.592
    (6.0, 1.5),   # Fn_c 0.222: C_V 6 at low trim (tau 1 is fully chine-dry at lam 2.5: not representable, skipped)
    (2.0, 10.0),  # Fn_c 0.499: C_V 2 at high trim
]


def fnc(cv, tau):
    return math.sqrt(2.0) * cv * math.tan(math.radians(tau))


def _rss_gb(proc):
    mi = proc.memory_info()
    peak = getattr(mi, "peak_wset", None)
    return mi.rss / 1e9, (peak / 1e9 if peak else None)


def run():
    import psutil
    import studies.rankine_planing_b8 as g
    from hydro.rankine import planing_nl as nl
    from hydro.rankine.planing_steady import savitsky
    proc = psutil.Process()
    os.makedirs(OUTDIR, exist_ok=True)
    res = dict(settings=dict(PROD=g.PROD, lam_max=6.0, beta=BETA, lam=LAM, max_iter=200,
                             variants=[v for v, _ in g.VARIANTS]),
               V=[])
    if os.path.exists(CASES_JSON):
        res = json.load(open(CASES_JSON))
    done = {(r["Cv"], r["tau"]) for r in res["V"]}
    t_all = time.time()
    for cv, tau in NEW_CASES:
        if (cv, tau) in done:
            continue
        rss0, _ = _rss_gb(proc)
        free0 = psutil.virtual_memory().available / 1e9
        t0 = time.time()
        S = g.solver(cv, BETA)
        t_kernel = time.time() - t0
        t1 = time.time()
        st = S.solve_lambda(math.radians(tau), LAM)
        t_solve = time.time() - t1
        sv = savitsky(tau, LAM, cv, BETA)
        rec = dict(Cv=cv, tau=tau, lam=LAM, beta=BETA, Fn_c=fnc(cv, tau), conv=bool(st["converged"]),
                   iters=int(st.get("iters", -1)), le_res_cells=float(st.get("le_res_cells", float("nan"))),
                   LK=float(st["LK"]), LC=float(st["LC"]), sav=float(sv["CL"]), lin=float(st["CL"]),
                   comp=float(nl.composite(S, st)["CL"]))
        for name, kw in g.VARIANTS:
            rec[name] = float(nl.composite(S, st, **kw)["CL"])
        rss1, peak = _rss_gb(proc)
        rec.update(t_kernel_s=t_kernel, t_solve_s=t_solve, rss_before_gb=rss0, rss_after_gb=rss1,
                   peak_rss_gb=peak, free_before_gb=free0, free_after_gb=psutil.virtual_memory().available / 1e9)
        try:
            import torch
            rec["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        except Exception:
            pass
        res["V"].append(rec)
        print("V", json.dumps(rec), flush=True)
        json.dump(res, open(CASES_JSON, "w"), indent=1)
    res["t_total_s"] = time.time() - t_all
    json.dump(res, open(CASES_JSON, "w"), indent=1)
    print(f"done, {res['t_total_s']:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------- statistics
def load_rows():
    rows = []
    for v in json.load(open(OLD_JSON))["V"]:
        rows.append(dict(src="old", Cv=v["Cv"], tau=v["tau"], lam=v["lam"], conv=bool(v["conv"]),
                         sav=v["sav"], lin=v["lin"], comp=v["comp"], mlm_only=v["mlm_only"],
                         sep_only=v["sep_only"], Fn_c=fnc(v["Cv"], v["tau"])))
    if os.path.exists(CASES_JSON):
        for v in json.load(open(CASES_JSON))["V"]:
            rows.append(dict(src="new", Cv=v["Cv"], tau=v["tau"], lam=v["lam"], conv=bool(v["conv"]),
                             sav=v["sav"], lin=v["lin"], comp=v["comp"], mlm_only=v["mlm_only"],
                             sep_only=v["sep_only"], Fn_c=v["Fn_c"], le_res=v.get("le_res_cells"),
                             iters=v.get("iters")))
    for r in rows:
        for k in ("lin", "comp", "mlm_only", "sep_only"):
            r["r_" + k] = r[k] / r["sav"]
        r["stat"] = 0.458 * r["lam"] ** 2 / r["Cv"] ** 2
    return rows


def spearman_block(rows, key):
    from scipy.stats import spearmanr
    y = np.log([r["r_" + key] for r in rows])
    out = {}
    for k in ("Fn_c", "Cv", "tau", "lam", "stat"):
        x = np.array([r[k] for r in rows])
        rho, p = spearmanr(x, y)
        out[k] = dict(rho=float(rho), p=float(p), distinct=len(set(x.tolist())))
    return out


def main_effects(rows, key):
    """One-way (additive) main effects of log ratio on the factors C_V, tau, lam. On the enlarged set the design
    is not balanced, so the one-way sums of squares need not add to the total; the shares are reported as they
    come, together with the residual sd of the additive fit (alternating means, 50 sweeps)."""
    y = np.log([r["r_" + key] for r in rows])
    yb = y.mean()
    facs = ("Cv", "tau", "lam")
    X = {k: np.array([r[k] for r in rows]) for k in facs}
    eff = {k: {float(v): 0.0 for v in sorted(set(X[k].tolist()))} for k in facs}
    for _ in range(50):
        for k in facs:
            other = sum(np.array([eff[j][x] for x in X[j]]) for j in facs if j != k)
            resid = y - yb - other
            for v in eff[k]:
                eff[k][v] = float(resid[X[k] == v].mean())
        # centre
        for k in facs:
            w = {v: int((X[k] == v).sum()) for v in eff[k]}
            m = sum(eff[k][v] * w[v] for v in eff[k]) / len(y)
            for v in eff[k]:
                eff[k][v] -= m
            yb += m
    fit = yb + sum(np.array([eff[k][x] for x in X[k]]) for k in facs)
    sst = float(((y - y.mean()) ** 2).sum())
    ss_oneway = {k: float(sum(((y[X[k] == v].mean() - y.mean()) ** 2) * (X[k] == v).sum()
                             for v in eff[k])) / sst for k in facs}
    # drop-one: loss of explained SS when one factor is removed from the additive fit (unbalanced-safe)
    ss_fit = float(((fit - y.mean()) ** 2).sum())
    drop = {}
    for k in facs:
        sub = [j for j in facs if j != k]
        e2 = {j: {float(v): 0.0 for v in sorted(set(X[j].tolist()))} for j in sub}
        yb2 = y.mean()
        for _ in range(50):
            for j in sub:
                other = sum(np.array([e2[i][x] for x in X[i]]) for i in sub if i != j)
                resid = y - yb2 - other
                for v in e2[j]:
                    e2[j][v] = float(resid[X[j] == v].mean())
        fit2 = yb2 + sum(np.array([e2[j][x] for x in X[j]]) for j in sub)
        drop[k] = (ss_fit - float(((fit2 - y.mean()) ** 2).sum())) / sst
    return dict(effects=eff, ss_share_oneway=ss_oneway, ss_share_dropone=drop,
                additive_resid_sd=float(np.std(y - fit)), total_sd=float(np.std(y)), n=len(y),
                explained=ss_fit / sst)


def groups_equal_fnc(rows, tol=0.02):
    """Cases whose Fn_c agree within tol (relative), with at least two distinct C_V."""
    rows = sorted(rows, key=lambda r: r["Fn_c"])
    used, groups = set(), []
    for i, r in enumerate(rows):
        if i in used:
            continue
        grp = [j for j in range(len(rows)) if j not in used and abs(rows[j]["Fn_c"] / r["Fn_c"] - 1) <= tol]
        if len({rows[j]["Cv"] for j in grp}) >= 2:
            groups.append([rows[j] for j in grp])
            used.update(grp)
    return groups


def fmt_case(r):
    conv = "" if r["conv"] else " NOT CONV"
    return (f"C_V {r['Cv']:>4g} tau {r['tau']:>4g} lam {r['lam']:g} Fn_c {r['Fn_c']:.3f} [{r['src']}]"
            f"  mlm/Sav {r['r_mlm_only']:.3f}  lin/Sav {r['r_lin']:.3f}  comp/Sav {r['r_comp']:.3f}{conv}")


def analyse():
    rows = load_rows()
    conv = [r for r in rows if r["conv"]]
    lines = []
    L = lines.append
    L("V4: B8 steady planing cases, old post-tier grid (24) + new cases; beta 20 deg; ratios lift / Savitsky")
    L(f"numerical uncertainty of the 3-D linear part (B8 PL1B/PC1u): {U_NUM:.1%} relative, one sign per C_V")
    L("")
    L("All cases, ordered by Fn_c:")
    for r in sorted(rows, key=lambda r: (r["Fn_c"], r["Cv"])):
        extra = ""
        if r["src"] == "new":
            extra = f"  (iters {r.get('iters')}, spray-root residual {r.get('le_res')} cells)"
        L("  " + fmt_case(r) + extra)
    L("")
    out = dict(n_all=len(rows), n_conv=len(conv), U_num=U_NUM)

    # --- equal Fn_c, different C_V (lam 2.5 only, so that lam does not move)
    L("TEST A/B 1: equal Fn_c, different C_V (lam 2.5, converged only). Explanation A (cross-plane gravity)")
    L("predicts equal ratios inside a group; explanation B (3-D gravity) predicts ratios falling with C_V.")
    grp_out = []
    for grp in groups_equal_fnc([r for r in conv if r["lam"] == LAM]):
        grp = sorted(grp, key=lambda r: r["Cv"])
        L(f"  Fn_c ~ {np.mean([r['Fn_c'] for r in grp]):.3f}:")
        for r in grp:
            L("     " + fmt_case(r))
        for key in ("mlm_only", "lin", "comp"):
            v = [r["r_" + key] for r in grp]
            L(f"     {key:9s}: spread max/min {max(v) / min(v):.3f}; "
              f"ratio vs C_V: " + ", ".join(f"{r['Cv']:g}->{r['r_' + key]:.3f}" for r in grp))
        grp_out.append(dict(Fn_c=float(np.mean([r['Fn_c'] for r in grp])),
                            cases=[dict(Cv=r["Cv"], tau=r["tau"], mlm=r["r_mlm_only"], lin=r["r_lin"],
                                        comp=r["r_comp"]) for r in grp]))
    out["equal_fnc_groups"] = grp_out
    L("")

    # --- equal C_V, different tau (lam 2.5)
    L("TEST A/B 2: equal C_V, different tau (lam 2.5, converged only). A predicts the ratio falls with tau")
    L("(Fn_c rises); B predicts a flat ratio in tau.")
    cv_out = {}
    for cv in sorted({r["Cv"] for r in conv}):
        sel = sorted([r for r in conv if r["Cv"] == cv and r["lam"] == LAM], key=lambda r: r["tau"])
        if len(sel) < 2:
            continue
        L(f"  C_V {cv:g}:")
        for r in sel:
            L("     " + fmt_case(r))
        for key in ("mlm_only", "lin", "comp"):
            v = [r["r_" + key] for r in sel]
            L(f"     {key:9s}: spread max/min {max(v) / min(v):.3f}; ratio vs tau: "
              + ", ".join(f"{r['tau']:g}->{r['r_' + key]:.3f}" for r in sel))
        cv_out[cv] = [dict(tau=r["tau"], Fn_c=r["Fn_c"], mlm=r["r_mlm_only"], lin=r["r_lin"], comp=r["r_comp"])
                      for r in sel]
    out["equal_cv"] = cv_out
    L("")

    # --- lam 2.5 slice: ratio against Fn_c, and against C_V, as two orderings
    sel = [r for r in conv if r["lam"] == LAM]
    L(f"lam 2.5 slice, converged (n={len(sel)}): neighbour-difference spread of log(mlm/Sav) when ordered by")
    for k in ("Fn_c", "Cv", "tau"):
        s = sorted(sel, key=lambda r: (r[k], r["Cv"]))
        d = np.diff(np.log([r["r_mlm_only"] for r in s]))
        L(f"   {k:5s}: rms neighbour difference {math.sqrt(np.mean(d * d)):.3f}")
    L("")

    # --- Spearman and main effects on the enlarged converged set
    for label, sub in (("old grid, converged", [r for r in conv if r["src"] == "old"]),
                       ("old + new, converged", conv),
                       ("old + new, converged, lam 2.5 only", sel)):
        L(f"{label} (n={len(sub)}):")
        blk = {}
        for key in ("mlm_only", "lin", "comp"):
            sp = spearman_block(sub, key)
            L(f"   Spearman (average ranks) of log({key}/Sav): " + "; ".join(
                f"{k} {v['rho']:+.2f} [p {v['p']:.3f}]" for k, v in sp.items()))
            me = main_effects(sub, key)
            L(f"   {key} main effects: " + "; ".join(
                f"{k}: " + ", ".join(f"{v:g}:{e:+.3f}" for v, e in me['effects'][k].items())
                for k in ("Cv", "tau", "lam")))
            L(f"   {key} SS share one-way: " + ", ".join(f"{k} {s:.2f}" for k, s in me['ss_share_oneway'].items())
              + "; drop-one: " + ", ".join(f"{k} {s:.2f}" for k, s in me['ss_share_dropone'].items())
              + f"; additive fit explains {me['explained']:.2f}, residual sd {me['additive_resid_sd']:.3f}"
                f" of total {me['total_sd']:.3f}")
            blk[key] = dict(spearman=sp, main_effects=me)
        out[label] = blk
        L("")

    # --- does lin show the same ordering as mlm_only?
    from scipy.stats import spearmanr
    a = np.log([r["r_mlm_only"] for r in conv])
    b = np.log([r["r_lin"] for r in conv])
    rho, p = spearmanr(a, b)
    L(f"ordering of lin/Sav against mlm_only/Sav over the converged set: Spearman {rho:+.2f} [p {p:.3f}];")
    ratio = np.exp(a - b)
    L(f"mlm_only/lin (the section nonlinearity alone): mean {ratio.mean():.3f}, sd {ratio.std():.3f}, "
      f"range {ratio.min():.3f}..{ratio.max():.3f}")
    for k in ("Fn_c", "Cv", "tau", "lam"):
        x = np.array([r[k] for r in conv])
        rho2, p2 = spearmanr(x, np.log(ratio))
        L(f"   Spearman of log(mlm_only/lin) with {k}: {rho2:+.2f} [p {p2:.3f}]")
    out["mlm_over_lin"] = dict(mean=float(ratio.mean()), sd=float(ratio.std()))
    # --- is the section correction a fixed amount? dC_L = C_L(mlm_only) - C_L(lin) = 2 int (F_MLM - F_W) dx over the
    # chine-dry part; with V/U ~ tan(tau) and int F_W dx = pi V U / 8 this is -(1 - 0.733) pi tan(tau)/4 = -0.210 tan(tau)
    L("")
    L("section correction dC_L = C_L(mlm_only) - C_L(lin), divided by tan(tau) (closed form -0.210 for V = U tan tau),")
    L("and relative to C_L(lin): Cv tau lam  dC_L  dC_L/tan(tau)  dC_L/C_L(lin)")
    dil = []
    for r in sorted(conv, key=lambda r: (r["tau"], r["Cv"], r["lam"])):
        d = 2.0 * 0.5 * (r["mlm_only"] - r["lin"])
        t = math.tan(math.radians(r["tau"]))
        dil.append(dict(Cv=r["Cv"], tau=r["tau"], lam=r["lam"], dCL=d, dCL_over_tan=d / t, dCL_rel=d / r["lin"]))
        L(f"   {r['Cv']:4g} {r['tau']:4g} {r['lam']:3g}  {d:+.4f}  {d / t:+.4f}  {d / r['lin']:+.3f}")
    v = np.array([x["dCL_over_tan"] for x in dil])
    L(f"   dC_L/tan(tau): mean {v.mean():+.4f}, range {v.min():+.4f}..{v.max():+.4f} (spread {v.max() / v.min():.3f}); "
      f"dC_L/C_L(lin): range {min(x['dCL_rel'] for x in dil):+.3f}..{max(x['dCL_rel'] for x in dil):+.3f}")
    out["dilution"] = dil
    txt = "\n".join(lines)
    open(STATS + ".txt", "w").write(txt + "\n")
    json.dump(out, open(STATS + ".json", "w"), indent=1, default=float)
    print(txt)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    if mode == "run":
        run()
        analyse()
    else:
        analyse()
