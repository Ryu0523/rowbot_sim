#!/usr/bin/env python3
"""
S1 step 3-4: scarab195 at 5 kn on the V1b mesh sequence (coarse / base / fine of studies/derive_v1b_build.py, same hull
form, attitude, ramp and stem floor), with the production kinematic operator ("cons") or the opt-in edge-flux pair
("flux"): static stiffnesses C0 and C_s, instantaneous limits, radiation coefficients and the head/following-sea
excitation at Tp 3 and 5 s (encounter frequencies 3.24 / 1.67 / 0.94 / 0.84 rad/s), passivity of B(omega_e) at the
four encounter frequencies plus a few extra frequencies, the base-flow maximum on the free surface and where it
occurs, and (option) the Haskind / Timman-Newman check with a reversed-flow system on the same grid (the flux pair's
end closures do not depend on the flow direction, so the -U system is built on the +U grid).

Output: studies/_cache/derive/s1_scarab_<kin>_<grid>[_r<ramp>][_yc<floor>].json and s1_scarab_<label>.log.

    .venv/Scripts/python.exe studies/derive_s1_scarab.py --kin flux --grids coarse,base,fine
"""
import argparse
import datetime
import json
import math
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

EXTRA_OMEGAS = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0)


def tag(ramp, yc):
    return (f"_r{ramp:g}" if ramp != 0.3 else "") + (f"_yc{yc:g}" if yc != 2e-3 else "")


def fs_max_W(s, co, p, hull):
    """max |W|/U at the free-surface centroids and its location (x along the waterline in fractions from the stern
    end, y in hull half-breadths)."""
    Wf, _ = co.fs_base()
    m = np.linalg.norm(Wf, axis=1) / p["u"]
    k = int(np.argmax(m))
    c = s.fs.c[k]
    xs, xb = hull.waterline_ends(p["sink"], p["trim"])
    return dict(max=float(m[k]), x=float(c[0]), y=float(c[1]), x_frac=float((c[0] - xs) / (xb - xs)),
                n_above_2=int(np.sum(m > 2.0)), n_above_1_5=int(np.sum(m > 1.5)))


def haskind(hull, p, grid_kw, hm, kin, co_plus, s_plus, cases, log):
    """Timman-Newman and Haskind with the -U system on the SAME grid (sgn=+1 grid, base flow reversed)."""
    from hydro.rankine.doublebody import build_db_solver
    t0 = time.time()
    sm, cm, Gm, dbm = build_db_solver(hull, -p["u"], p["sink"], p["trim"], grid_kw, hm, g=G, rho=RHO, ref=(0.0, 0.0),
                                      kinematic=kin, sgn=1.0)
    same = bool(np.allclose(s_plus.fs.c, sm.fs.c) and np.allclose(s_plus.body.c, sm.body.c))
    out = dict(same_panels=same, t_build_minus=time.time() - t0, cases={})
    a = s_plus.body.area
    af = s_plus.fs.area
    for key, cs in cases.items():
        we = cs["omega_e"]
        w0, beta = cs["omega0"], cs["beta"]
        ex = co_plus.excitation(w0, beta, want_phi=True, forcing=True)
        rm = cm.radiation(we, want_phi=True)
        rp = co_plus.radiation(we)
        hk, hkc, tn = [], [], []
        for jj, j in enumerate(co_plus.modes):
            psi = rm["sol"]["phi_B"][:, jj]
            cl = -RHO * 2 * np.sum(psi * ex["dn_I"] * a)
            zpsi = sm.zeta_of(we, rm["sol"]["phi_F"][:, jj])
            GF = 2 * np.sum((zpsi * ex["fd"] + rm["sol"]["phi_F"][:, jj] * ex["fk"]) * af)
            Xd = ex["X_d_ot"][jj]
            hk.append(abs(Xd - (cl + RHO * GF)) / abs(ex["X"][jj]))
            hkc.append(abs(Xd - cl) / abs(ex["X"][jj]))
        for (j, k) in ((1, 2), (0, 1), (0, 2), (2, 1)):
            dT = rp["T_ot"][j, k] - rm["T_ot"][k, j]
            scl = max(abs(rp["T_ot"][j, k]), abs(rm["T_ot"][k, j]))
            tn.append(abs(dT) / max(scl, 1e-300))
        out["cases"][key] = dict(haskind_rel=[float(v) for v in hk], haskind_classical_rel=[float(v) for v in hkc],
                                 tn_rel=[float(v) for v in tn], omega_e=float(we))
        log(f"  Haskind {key} (omega_e {we:.3f}): |X_D^OT - (classical + fs term)|/|X| per mode "
            f"{np.round(hk, 4).tolist()}; classical only {np.round(hkc, 3).tolist()}; TN (35,13,15,53) {np.round(tn, 4).tolist()}")
    sm.release()
    del sm, cm, dbm
    return out


def run(kin, gname, ramp, yc, extra, do_haskind, log, flux_kw=None, label_suffix=""):
    import torch
    from hydro.rankine.doublebody import build_db_solver, frozen_pressure_stiffness
    t_all = time.time()
    p = plant_numbers(False)
    hull = scarab_form(p, ramp, yc_floor=yc)
    grid_kw, hm = GRIDS[gname]
    rec = dict(kin=kin, grid=gname, grid_kw=grid_kw, hull_mesh=list(hm), ramp=ramp, yc_floor=yc, flux_kw=flux_kw,
               sink=p["sink"], trim=p["trim"], u=p["u"], tau_crit=G / (4 * p["u"]))
    t0 = time.time()
    s, co, Gg, db = build_db_solver(hull, p["u"], p["sink"], p["trim"], grid_kw, hm, g=G, rho=RHO, ref=(0.0, 0.0),
                                    kinematic=kin, flux_kw=flux_kw)
    t_build = time.time() - t0
    rss, gpu = mem_now()
    log(f"[{kin} {gname}] built: NB {s.NB}, NF {s.NF}, dx {Gg['dx']:.4f} m, grid {Gg['shape']}, J {Gg['J']}; {t_build:.0f} s; "
        f"RSS {rss:.2f} GB, GPU peak {gpu:.2f} GB")
    rec.update(NB=int(s.NB), NF=int(s.NF), dx=float(Gg["dx"]), shape=list(Gg["shape"]), J=int(Gg["J"]), t_build=t_build)
    W = db.W
    rec["W_hull_max_over_U"] = float(np.linalg.norm(W, axis=1).max() / p["u"])
    rec["W_fs"] = fs_max_W(s, co, p, hull)
    info = getattr(s, "_opinfo", {})
    if kin == "flux":
        rec["W_fs"]["max_gauss"] = float(info["Wmax"] / p["u"])
        rec["divW_max"] = float(np.abs(info["divW"]).max())
        xs_, xb_ = hull.waterline_ends(p["sink"], p["trim"])
        at = info["Wmax_at"]
        rec["W_fs"]["gauss_at"] = dict(x=float(at[0]), y=float(at[1]), x_frac=float((at[0] - xs_) / (xb_ - xs_)))
        mg = info["Wgauss"] / p["u"]
        rec["W_fs"]["gauss_n_above_2"] = int(np.sum(mg > 2.0))
        rec["W_fs"]["gauss_n_above_1_5"] = int(np.sum(mg > 1.5))
        log(f"  Gauss points: |W|/U max {rec['W_fs']['max_gauss']:.2f} at x_frac {rec['W_fs']['gauss_at']['x_frac']:.4f}, "
            f"y {at[1]:.4f} m; points > 1.5: {rec['W_fs']['gauss_n_above_1_5']}, > 2: {rec['W_fs']['gauss_n_above_2']}")
    log(f"  |W|/U max: hull {rec['W_hull_max_over_U']:.2f}; free surface centroids {rec['W_fs']['max']:.2f} at x {rec['W_fs']['x']:+.3f} m "
        f"(fraction {rec['W_fs']['x_frac']:.3f} of the waterline from the stern end), y {rec['W_fs']['y']:.3f} m; "
        f"cells with |W|/U > 1.5: {rec['W_fs']['n_above_1_5']}, > 2: {rec['W_fs']['n_above_2']}"
        + (f"; Gauss points {rec['W_fs']['max_gauss']:.2f}" if kin == "flux" else ""))
    t0 = time.time()
    inst = co.instantaneous()
    C0, C0_ot = co.static_stiffness()
    Cs = frozen_pressure_stiffness(co.db, co.ref, RHO)
    rec.update(A_inf=inst["A"].tolist(), B_inf=inst["B"].tolist(), C0=C0.tolist(), C0_ot=C0_ot.tolist(), C_s=Cs.tolist())
    log(f"  A_inf diag {np.diag(inst['A']).round(1).tolist()}; B_inf {inst['B'].round(0).tolist()}; "
        f"C0 heave/pitch {C0[1, 1]:.0f} / {C0[2, 2]:.0f} (OT {C0_ot[1, 1]:.0f} / {C0_ot[2, 2]:.0f}); C_s {Cs[1, 1]:.0f} / {Cs[2, 2]:.0f}; "
        f"sum {C0[1, 1] + Cs[1, 1]:.0f} / {C0[2, 2] + Cs[2, 2]:.0f} [{time.time() - t0:.0f} s]")
    # wave cases
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
    # passivity at extra frequencies
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
        rec["haskind"] = haskind(hull, p, grid_kw, hm, kin, co, s, cases, log)
    rss, gpu = mem_now()
    s.release()
    del s, co, db
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rec.update(t_total=time.time() - t_all, peak_rss_gb=rss, peak_gpu_gb=gpu, created=datetime.datetime.now().isoformat(timespec="seconds"))
    log(f"[{kin} {gname}] total {rec['t_total']:.0f} s; RSS {rss:.2f} GB, GPU peak {gpu:.2f} GB")
    name = f"s1_scarab_{kin}_{gname}{tag(ramp, yc)}" + (f"_{flux_kw['recon']}" if flux_kw and flux_kw.get("recon") else "") + label_suffix
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kin", default="flux", choices=("cons", "flux"))
    ap.add_argument("--grids", default="coarse,base,fine")
    ap.add_argument("--ramp", type=float, default=0.3)
    ap.add_argument("--ycfloor", type=float, default=2e-3)
    ap.add_argument("--no-extra", action="store_true")
    ap.add_argument("--haskind", default="", help="comma list of grids for which the -U system / Haskind check is run")
    ap.add_argument("--recon", default=None, help="flux reconstruction variant (fv6t default, fv6, lin)")
    ap.add_argument("--label", default="", help="suffix of the output names (e.g. _hk for a Haskind-only run)")
    a = ap.parse_args()
    flux_kw = dict(recon=a.recon) if a.recon else None
    label = f"{a.kin}{tag(a.ramp, a.ycfloor)}" + (f"_{a.recon}" if a.recon else "") + a.label
    logf = open(os.path.join(OUT, f"s1_scarab_{label}.log"), "a", encoding="utf-8")

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
        run(a.kin, gname, a.ramp, a.ycfloor, () if a.no_extra else EXTRA_OMEGAS, gname in hk, log, flux_kw, a.label)
    logf.close()


if __name__ == "__main__":
    main()
