#!/usr/bin/env python3
"""
S1 stage 2 (S1b) item 2c: scarab195 at 5 kn on the V1b mesh sequence (coarse / base / fine) with the edge-flux operator
pair ("flux", stage 1) PLUS the opt-in grid fixes of this stage (derive_s1b_common.grid_opts: stem end width, end
sub-columns, stern knuckle node, graded band) and the opt-in near-hull sampling rule (doublebody.NearWallSampler,
flux_kw["near_wall"]). Same quantities and log lines as studies/derive_s1_scarab.py (whose helpers are reused): C0,
C_s, A_inf / B_inf, radiation and excitation at the four wave cases, passivity at extra frequencies, max |W|/U at the
free-surface centroids and at the Gauss points, Haskind / Timman-Newman with the -U system on the same grid (option).

Output: studies/_cache/derive/s1b_scarab_<variant>[_nw<kappa>[f]]_<grid>.json and s1b_scarab_<label>.log

    .venv/Scripts/python.exe studies/derive_s1b_scarab.py --variant esg --near-wall 0.5 --forcing --grids coarse,base,fine --haskind base
"""
import argparse
import datetime
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

from studies.derive_v1b_build import (G, RHO, TP_LIST, HEADINGS, GRIDS, plant_numbers, scarab_form, mem_now,  # noqa: E402
                                      ramp_panels, fk_on)
from studies.derive_s1_scarab import EXTRA_OMEGAS, fs_max_W, haskind  # noqa: E402
from studies.derive_s1b_common import grid_opts  # noqa: E402


def label_of(variant, kappa, forcing):
    return variant + (f"_nw{kappa:g}" + ("f" if forcing else "") if kappa else "")


def run(variant, kappa, forcing, gname, ramp, yc, extra, do_haskind, log, label_suffix=""):
    import torch
    from hydro.rankine.doublebody import build_db_solver, frozen_pressure_stiffness
    kin = "flux"
    t_all = time.time()
    p = plant_numbers(False)
    hull = scarab_form(p, ramp, yc_floor=yc)
    grid_kw0, hm = GRIDS[gname]
    gopts = grid_opts(variant, hull, p["sink"], p["trim"])
    grid_kw = dict(grid_kw0, **gopts)
    flux_kw = dict(near_wall=dict(kappa=kappa, forcing=forcing)) if kappa else None
    rec = dict(kin=kin, variant=variant, grid=gname, grid_kw=grid_kw, grid_opts=gopts, hull_mesh=list(hm), ramp=ramp, yc_floor=yc,
               flux_kw=flux_kw, sink=p["sink"], trim=p["trim"], u=p["u"], tau_crit=G / (4 * p["u"]))
    t0 = time.time()
    s, co, Gg, db = build_db_solver(hull, p["u"], p["sink"], p["trim"], grid_kw, hm, g=G, rho=RHO, ref=(0.0, 0.0),
                                    kinematic=kin, flux_kw=flux_kw)
    t_build = time.time() - t0
    rss, gpu = mem_now()
    log(f"[{variant} nw{kappa:g}{'f' if forcing else ''} {gname}] built: NB {s.NB}, NF {s.NF}, dx {Gg['dx']:.4f} m, grid {Gg['shape']}, J {Gg['J']}; "
        f"{t_build:.0f} s; RSS {rss:.2f} GB, GPU peak {gpu:.2f} GB")
    rec.update(NB=int(s.NB), NF=int(s.NF), dx=float(Gg["dx"]), shape=list(Gg["shape"]), J=int(Gg["J"]), t_build=t_build)
    W = db.W
    rec["W_hull_max_over_U"] = float(np.linalg.norm(W, axis=1).max() / p["u"])
    kmax = int(np.argmax(np.linalg.norm(W, axis=1)))
    xs_, xb_ = hull.waterline_ends(p["sink"], p["trim"])
    ch = s.body.c[kmax]
    rec["W_hull_max_at"] = dict(x=float(ch[0]), y=float(ch[1]), z=float(ch[2]), x_frac=float((ch[0] - xs_) / (xb_ - xs_)))
    rec["W_fs"] = fs_max_W(s, co, p, hull)
    info = getattr(s, "_opinfo", {})
    rec["W_fs"]["max_gauss"] = float(info["Wmax"] / p["u"])
    rec["divW_max"] = float(np.abs(info["divW"]).max())
    at = info["Wmax_at"]
    rec["W_fs"]["gauss_at"] = dict(x=float(at[0]), y=float(at[1]), x_frac=float((at[0] - xs_) / (xb_ - xs_)))
    mg = info["Wgauss"] / p["u"]
    rec["W_fs"]["gauss_n_above_2"] = int(np.sum(mg > 2.0))
    rec["W_fs"]["gauss_n_above_1_5"] = int(np.sum(mg > 1.5))
    if "near_wall" in info:
        nwi = info["near_wall"]
        gs = nwi["gauss"]
        rec["near_wall"] = dict(kappa=nwi["kappa"], gauss_moved=int(sum(g["moved"] for g in gs)), gauss_inside=int(sum(g["inside"] for g in gs)),
                                gauss_total=int(sum(g["n"] for g in gs)), gauss_shift_max=float(max(g["shift_max"] for g in gs)),
                                centroids=nwi["centroids"])
        log(f"  near-wall rule kappa {nwi['kappa']}: Gauss points moved {rec['near_wall']['gauss_moved']} of {rec['near_wall']['gauss_total']} "
            f"({rec['near_wall']['gauss_inside']} were inside the hull), max shift {rec['near_wall']['gauss_shift_max'] * 100:.2f} cm; centroids {nwi['centroids']}")
    log(f"  |W|/U max: hull {rec['W_hull_max_over_U']:.2f} at x_frac {rec['W_hull_max_at']['x_frac']:.4f} (y {ch[1]:.3f}, z {ch[2]:.3f}); "
        f"free surface centroids {rec['W_fs']['max']:.2f} at x_frac {rec['W_fs']['x_frac']:.3f}, y {rec['W_fs']['y']:.3f} m; "
        f"Gauss points {rec['W_fs']['max_gauss']:.2f} at x_frac {rec['W_fs']['gauss_at']['x_frac']:.4f}, y {at[1]:.4f} m; "
        f"cells > 1.5: {rec['W_fs']['n_above_1_5']}, Gauss > 1.5: {rec['W_fs']['gauss_n_above_1_5']}, > 2: {rec['W_fs']['gauss_n_above_2']}")
    t0 = time.time()
    inst = co.instantaneous()
    C0, C0_ot = co.static_stiffness()
    Cs = frozen_pressure_stiffness(co.db, co.ref, RHO)
    rec.update(A_inf=inst["A"].tolist(), B_inf=inst["B"].tolist(), C0=C0.tolist(), C0_ot=C0_ot.tolist(), C_s=Cs.tolist())
    log(f"  A_inf diag {np.diag(inst['A']).round(1).tolist()}; B_inf {inst['B'].round(0).tolist()}; "
        f"C0 heave/pitch {C0[1, 1]:.0f} / {C0[2, 2]:.0f} (OT {C0_ot[1, 1]:.0f} / {C0_ot[2, 2]:.0f}); C_s {Cs[1, 1]:.0f} / {Cs[2, 2]:.0f}; "
        f"sum {C0[1, 1] + Cs[1, 1]:.0f} / {C0[2, 2] + Cs[2, 2]:.0f} [{time.time() - t0:.0f} s]")
    # steady double-body wave problem at the attitude (information; the NK overlap check compares its Fz / My)
    try:
        from hydro.rankine.doublebody import steady_db
        t0 = time.time()
        st = steady_db(s, co, hull, p["sink"], p["trim"], cg=(0.0, 0.0))
        rec["steady"] = {k: float(v) for k, v in st.items() if np.isscalar(v)}
        log("  steady double-body solution: " + ", ".join(f"{k} {v:.1f}" for k, v in rec["steady"].items()) + f" [{time.time() - t0:.0f} s]")
    except Exception as e:  # noqa: BLE001
        log(f"  steady_db failed: {e!r}")
    cases = {}
    mask = ramp_panels(hull, s, p)
    t0 = time.time()
    Bs_list = []
    for hd, beta in HEADINGS.items():
        for Tp in TP_LIST:
            w0 = 2 * np.pi / Tp
            rad, exc = co.rad_exc(w0, beta)
            we = exc["omega_e"]
            fkr = fk_on(s, co, mask, w0, beta)
            key = f"{hd}_Tp{Tp:g}"
            Bsym = 0.5 * (rad["B"] + rad["B"].T)
            ev = np.linalg.eigvalsh(Bsym)
            cases[key] = dict(heading=hd, beta=beta, Tp=Tp, omega0=w0, k=float(exc["k"]), omega_e=float(we),
                              tau_B=p["u"] * abs(we) / G, X=exc["X"].tolist(), X_ot=exc["X_ot"].tolist(),
                              X_fk=exc["X_fk"].tolist(), X_d=exc["X_d"].tolist(), X_fk_ramp=fkr.tolist(),
                              A_e=rad["A"].tolist(), B_e=rad["B"].tolist(),
                              bsym_rel=float(ev[0] / max(np.abs(ev).max(), 1e-300)))
            Bs_list.append((we, cases[key]["bsym_rel"]))
            kk = exc["k"]
            log(f"  {key}: omega_e {we:+.4f}, tau_B {p['u'] * abs(we) / G:.3f}; |X3| {abs(exc['X'][1]):.0f} N/m (FK {abs(exc['X_fk'][1]):.0f}, "
                f"diffraction {abs(exc['X_d'][1]):.0f}); |X5|/k {abs(exc['X'][2]) / kk:.0f} N m/m (FK {abs(exc['X_fk'][2]) / kk:.0f}, "
                f"diffr {abs(exc['X_d'][2]) / kk:.0f}); A33 {rad['A'][1, 1]:.0f} B33 {rad['B'][1, 1]:.0f} A55 {rad['A'][2, 2]:.0f} "
                f"B55 {rad['B'][2, 2]:.0f} A35 {rad['A'][1, 2]:.0f} A53 {rad['A'][2, 1]:.0f}; B_sym min/max {cases[key]['bsym_rel']:+.4f}")
    rec["cases"] = cases
    log(f"  wave cases: {time.time() - t0:.0f} s")
    if extra:
        t0 = time.time()
        pas = {}
        for w in extra:
            r = co.radiation(w)
            Bm = r["T"].imag / w
            ev = np.linalg.eigvalsh(0.5 * (Bm + Bm.T))
            pas[f"{w:g}"] = dict(bsym_rel=float(ev[0] / max(np.abs(ev).max(), 1e-300)), A=(-r["T"].real / w ** 2).tolist(), B=Bm.tolist())
            Bs_list.append((w, pas[f"{w:g}"]["bsym_rel"]))
        rec["extra"] = pas
        log(f"  passivity (min eig of B_sym / max |eig|): " + ", ".join(f"{w:.2f}: {v:+.4f}" for w, v in sorted(Bs_list)) + f" [{time.time() - t0:.0f} s]")
        rec["bsym_worst"] = min(v for _, v in Bs_list)
    if do_haskind:
        # the -U system on the same grid: build_db_solver with the same grid_kw (flux_kw passed through)
        from hydro.rankine import doublebody as _dbm
        orig = _dbm.build_db_solver

        def patched(*a, **k):
            k = dict(k)
            k["flux_kw"] = flux_kw
            return orig(*a, **k)
        _dbm.build_db_solver = patched
        try:
            rec["haskind"] = haskind(hull, p, grid_kw, hm, kin, co, s, cases, log)
        finally:
            _dbm.build_db_solver = orig
    rss, gpu = mem_now()
    s.release()
    del s, co, db
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rec.update(t_total=time.time() - t_all, peak_rss_gb=rss, peak_gpu_gb=gpu, created=datetime.datetime.now().isoformat(timespec="seconds"))
    log(f"[{variant} {gname}] total {rec['t_total']:.0f} s; RSS {rss:.2f} GB, GPU peak {gpu:.2f} GB")
    name = f"s1b_scarab_{label_of(variant, kappa, forcing)}_{gname}" + (f"_r{ramp:g}" if ramp != 0.3 else "") + (f"_yc{yc:g}" if yc != 2e-3 else "") + label_suffix
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="esg")
    ap.add_argument("--near-wall", type=float, default=0.0, help="kappa of the near-hull sampling rule (0 = off)")
    ap.add_argument("--forcing", action="store_true", help="apply the rule to the centroid values of the diffraction forcing too")
    ap.add_argument("--grids", default="coarse,base,fine")
    ap.add_argument("--ramp", type=float, default=0.3)
    ap.add_argument("--ycfloor", type=float, default=2e-3)
    ap.add_argument("--no-extra", action="store_true")
    ap.add_argument("--haskind", default="")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    lab = label_of(a.variant, a.near_wall, a.forcing) + (f"_r{a.ramp:g}" if a.ramp != 0.3 else "") + (f"_yc{a.ycfloor:g}" if a.ycfloor != 2e-3 else "") + a.label
    logf = open(os.path.join(OUT, f"s1b_scarab_{lab}.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    try:
        from studies.parallel_runner import free_gb
        log(f"free RAM at start {free_gb():.2f} GB")
    except Exception:
        pass
    hk = [g for g in a.haskind.split(",") if g]
    for gname in a.grids.split(","):
        run(a.variant, a.near_wall, a.forcing, gname, a.ramp, a.ycfloor, () if a.no_extra else EXTRA_OMEGAS, gname in hk, log, a.label)
    logf.close()


if __name__ == "__main__":
    main()
