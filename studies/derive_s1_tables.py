#!/usr/bin/env python3
"""
S1 step 3 tables: assemble studies/_cache/derive/s1_scarab_<kin>_<grid>*.json into the convergence tables (old operator
"cons" vs new "flux"): C0 and C_s, A/B at the four encounter frequencies, head/following-sea excitation (total,
Froude-Krylov, diffraction), max |W|/U on the free surface, passivity, Haskind; convergence ratios
(base - coarse)/(fine - base) per quantity; cross-check of the cons rows against the V1b tables (same code path).

Output: studies/_cache/derive/s1_tables.txt (+ s1_tables.json)
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "_cache", "derive")
GR = ("coarse", "base", "fine")
CASES = ("head_Tp3", "head_Tp5", "follow_Tp3", "follow_Tp5")
LINES = []


def log(s=""):
    print(s)
    LINES.append(s)


def load(kin, grid, suffix=""):
    p = os.path.join(OUT, f"s1_scarab_{kin}_{grid}{suffix}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def ratio(v):
    """(base - coarse)/(fine - base); None if a level is missing."""
    if any(x is None for x in v):
        return None
    c, b, f = v
    d = f - b
    if abs(d) < 1e-300:
        return float("inf")
    return (b - c) / d


def row(name, vals, fmt="{:12.0f}", unit=""):
    out = f"{name:44s}"
    for kin in ("cons", "flux"):
        v = vals[kin]
        out += " |" + " ".join(fmt.format(x) if x is not None else " " * 12 for x in v)
        r = ratio(v)
        out += f"  r {r:+7.2f}" if r is not None and np.isfinite(r) else "  r     ---"
    log(out + ("  " + unit if unit else ""))


def cx(lst):
    """complex values serialised by json.dump(default=str) -> array."""
    return np.array([complex(x) if isinstance(x, str) else complex(x) for x in lst])


def quantities(rec):
    """Flatten one record into name -> value."""
    q = {}
    if rec is None:
        return q
    C0, Cs = np.array(rec["C0"]), np.array(rec["C_s"])
    q["C0 heave (N/m)"] = C0[1, 1]
    q["C0 pitch (N m/rad)"] = C0[2, 2]
    q["C_s heave"] = Cs[1, 1]
    q["C_s pitch"] = Cs[2, 2]
    q["C0 + C_s heave"] = C0[1, 1] + Cs[1, 1]
    q["C0 + C_s pitch"] = C0[2, 2] + Cs[2, 2]
    q["C0 heave-pitch (3,5)"] = C0[1, 2]
    q["C0 pitch-heave (5,3)"] = C0[2, 1]
    A, B = np.array(rec["A_inf"]), np.array(rec["B_inf"])
    q["A_inf heave (kg)"] = A[1, 1]
    q["A_inf pitch (kg m2)"] = A[2, 2]
    q["B_inf (3,5)"] = B[1, 2]
    q["B_inf (5,3)"] = B[2, 1]
    for key in CASES:
        c = rec["cases"][key]
        Ae, Be = np.array(c["A_e"]), np.array(c["B_e"])
        X, Xfk, Xd = cx(c["X"]), cx(c["X_fk"]), cx(c["X_d"])
        we = c["omega_e"]
        k = c["k"]
        tagc = f"{key} (we {we:.2f})"
        q[f"A33 {tagc}"] = Ae[1, 1]
        q[f"B33 {tagc}"] = Be[1, 1]
        q[f"A55 {tagc}"] = Ae[2, 2]
        q[f"B55 {tagc}"] = Be[2, 2]
        q[f"|X3| total {tagc}"] = abs(X[1])
        q[f"|X3| diffraction {tagc}"] = abs(Xd[1])
        q[f"|X3| FK {tagc}"] = abs(Xfk[1])
        q[f"|X5|/k total {tagc}"] = abs(X[2]) / k
        q[f"|X5|/k diffraction {tagc}"] = abs(Xd[2]) / k
        q[f"|X5|/k FK {tagc}"] = abs(Xfk[2]) / k
        q[f"B_sym min/max {tagc}"] = c["bsym_rel"]
    q["|W|/U max free surface (centroids)"] = rec["W_fs"]["max"]
    q["|W|/U max hull"] = rec["W_hull_max_over_U"]
    if "bsym_worst" in rec:
        q["B_sym worst (all frequencies)"] = rec["bsym_worst"]
    return q


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else ""
    recs = {kin: {g: load(kin, g, suffix) for g in GR} for kin in ("cons", "flux")}
    Q = {kin: {g: quantities(recs[kin][g]) for g in GR} for kin in ("cons", "flux")}
    log(f"scarab195 5 kn, V1b mesh sequence coarse (M 32) / base (M 48) / fine (M 64){' ' + suffix if suffix else ''}: "
        "old operator (cons) | new operator (flux); r = (base - coarse)/(fine - base) (|r| >> 1: converging; ~1: not)")
    for kin in ("cons", "flux"):
        for g in GR:
            r = recs[kin][g]
            if r:
                log(f"  {kin:5s} {g:6s}: NB {r['NB']}, NF {r['NF']}, build {r['t_build']:.0f} s, total {r['t_total']:.0f} s, "
                    f"RSS {r['peak_rss_gb']:.2f} GB, GPU {r['peak_gpu_gb']:.2f} GB; |W|/U fs max {r['W_fs']['max']:.2f} at x_frac "
                    f"{r['W_fs']['x_frac']:.3f}, y {r['W_fs']['y']:.3f} m" + (f", Gauss {r['W_fs'].get('max_gauss', float('nan')):.2f}" if kin == "flux" else ""))
    log()
    names = list(Q["cons"]["base"].keys()) or list(Q["flux"]["base"].keys())
    hdr = f"{'quantity':44s}" + " |" + " ".join(f"{'cons ' + g:>12s}" for g in GR) + " " * 11 + " |" + " ".join(f"{'flux ' + g:>12s}" for g in GR)
    log(hdr)
    for nm in names:
        vals = {kin: [Q[kin][g].get(nm) for g in GR] for kin in ("cons", "flux")}
        fmt = "{:12.4f}" if ("B_sym" in nm or "|W|/U" in nm) else "{:12.0f}"
        row(nm, vals, fmt)
    # Haskind
    for kin in ("cons", "flux"):
        for g in GR:
            r = recs[kin][g]
            if r and "haskind" in r:
                h = r["haskind"]
                log(f"\nHaskind / Timman-Newman with the -U system on the same grid ({kin} {g}; same panels {h['same_panels']}):")
                for key, c in h["cases"].items():
                    log(f"  {key}: |X_D^OT - Haskind|/|X| per mode (1,3,5) {np.round(c['haskind_rel'], 4).tolist()}; "
                        f"classical-only {np.round(c['haskind_classical_rel'], 3).tolist()}; TN pairs (35,13,15,53) {np.round(c['tn_rel'], 4).tolist()}")
    # cross-check of cons against the V1b tables (same code path: must reproduce)
    try:
        from studies.derive_v1b_build import load as v1b_load
        lab = {"coarse": "smoke", "base": "base_r0.3", "fine": "fine_r0.3_n4"}
        log("\nreproduction check: cons rows vs the V1b tables (v1b_tab_*.npz; same code path, must agree to round-off):")
        for g in GR:
            r = recs["cons"][g]
            if r is None or not os.path.exists(os.path.join(OUT, f"v1b_tab_{lab[g]}.npz")):
                continue
            v = v1b_load(lab[g])
            d = []
            d.append(("C0 heave", r["C0"][1][1], float(np.asarray(v["C0"])[1, 1])))
            d.append(("C0 pitch", r["C0"][2][2], float(np.asarray(v["C0"])[2, 2])))
            d.append(("C_s heave", r["C_s"][1][1], float(np.asarray(v["C_s"])[1, 1])))
            d.append(("A_inf heave", r["A_inf"][1][1], float(np.asarray(v["A_inf"])[1, 1])))
            if "cases" in v:
                for key in CASES:
                    if key in v["cases"]:
                        d.append((f"|X3| {key}", abs(cx(r["cases"][key]["X"])[1]), float(abs(np.asarray(v["cases"][key]["X"])[1]))))
                        d.append((f"B33 {key}", r["cases"][key]["B_e"][1][1], float(np.asarray(v["cases"][key]["B_e"])[1, 1])))
            log(f"  {g} vs v1b_tab_{lab[g]}: " + "; ".join(f"{n} {a:.6g} / {b:.6g} ({(a - b) / b if b else 0:+.1e})" for n, a, b in d))
    except Exception as e:  # noqa: BLE001
        log(f"  (cross-check skipped: {e!r})")
    with open(os.path.join(OUT, f"s1_tables{suffix}.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(os.path.join(OUT, f"s1_tables{suffix}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(Q=Q), fh, indent=1, default=str)


if __name__ == "__main__":
    main()
