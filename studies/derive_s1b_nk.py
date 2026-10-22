#!/usr/bin/env python3
"""
S1 stage 2 (S1b) item 3: the Neumann-Kelvin (uniform-stream) linearisation of hydro/rankine/unsteady.py on the
scarab195 hull at 5 kn (tables.build_solver(..., linearisation="NK"): production wall-trace closure, D-form forces),
on the V1b grids with the opt-in grid fixes of this stage (derive_s1b_common.grid_opts). In V1b the build failed in
dx_operator with "grid not right-handed": derive_s1b_diag.py locates the negative Jacobian in the SECOND free-surface
column from the stern end, where the band's straight wall chord cuts the knuckle of the waterline (the chine corner,
slope ~7 -> 0 within one column) and the trace closure's wall node (the true waterline at the column centre) lies
OUTSIDE the cell -- the stern knuckle node option (x_nodes) removes it; nothing in dx_operator is changed.

Quantities (same definitions as the double-body runs, for the overlap check): A_inf, B_inf, C0 (tables.static_stiffness;
the uniform stream has no base pressure, so C_s = 0 and C0 plays the role of C0 + C_s of the double-body model),
radiation and excitation at the four wave cases (NK excitation: no free-surface forcing, the incident wave satisfies
the uniform-stream conditions exactly), passivity at the extra frequencies, and the steady NK force at omega = 0
(unsteady.steady_forces: Fz, My about the CG, resistance).

Output: studies/_cache/derive/s1b_nk_<variant>_<grid>.json, s1b_nk_<label>.log
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
from studies.derive_s1_scarab import EXTRA_OMEGAS  # noqa: E402
from studies.derive_s1b_common import grid_opts  # noqa: E402


def build_nk(hull, U, sink, trim, grid_kw, hull_mesh, wall="trace", rho=RHO, g=G, ref=(0.0, 0.0)):
    """tables.build_solver(linearisation="NK") with the wall closure selectable: "trace" (production) or "rule" (the
    B5 rule: cross term dropped on the wall line, no hull traces; information)."""
    from hydro.rankine.tables import build_solver
    if wall == "trace":
        return build_solver(hull, U, sink, trim, grid_kw, hull_mesh, rho=rho, g=g, ref=ref, linearisation="NK")
    from hydro.rankine.unsteady import build_grid, dx_operator, wall_trace_data, ForwardSpeedLTI, Coefficients
    sgn = 1.0 if U >= 0 else -1.0
    body = hull.mesh(*hull_mesh, sink=sink, trim=trim)
    Gg = build_grid(hull, sink, trim, sgn, **dict(grid_kw))
    D, info = dx_operator(Gg, sgn, wall=wall, hull=hull, sink=sink, trim=trim)
    wt = wall_trace_data(info["wall"], hull, body, hull_mesh)
    s = ForwardSpeedLTI(body, Gg["fs"], D, Gg["nu"] * np.sqrt(g / hull.L), U, g, rho, hull_shape=hull_mesh, wall=wt)
    refE = hull.to_earth(np.array(ref[0]), np.array(0.0), np.array(ref[1]), sink, trim)
    co = Coefficients(s, ref=(float(refE[0]), float(refE[2])))
    return s, co, Gg


def run(variant, gname, ramp, yc, extra, log, label_suffix="", wall="trace"):
    import torch
    from hydro.rankine.tables import static_stiffness
    from hydro.rankine.unsteady import steady_forces
    t_all = time.time()
    p = plant_numbers(False)
    hull = scarab_form(p, ramp, yc_floor=yc)
    grid_kw0, hm = GRIDS[gname]
    gopts = grid_opts(variant, hull, p["sink"], p["trim"])
    grid_kw = dict(grid_kw0, **gopts)
    rec = dict(lin="NK", wall=wall, variant=variant, grid=gname, grid_kw=grid_kw, grid_opts=gopts, hull_mesh=list(hm), ramp=ramp, yc_floor=yc,
               sink=p["sink"], trim=p["trim"], u=p["u"], tau_crit=G / (4 * p["u"]))
    t0 = time.time()
    try:
        s, co, Gg = build_nk(hull, p["u"], p["sink"], p["trim"], grid_kw, hm, wall=wall)
    except AssertionError as e:
        log(f"[NK {variant} {gname}] build FAILED: {e!r}")
        rec["failed"] = repr(e)
        with open(os.path.join(OUT, f"s1b_nk_{variant}{'' if wall == 'trace' else '_' + wall}_{gname}{label_suffix}.json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1, default=str)
        return rec
    t_build = time.time() - t0
    rss, gpu = mem_now()
    wl = s.wall
    nwall = 0 if wl is None else int(wl["C"].shape[1])
    log(f"[NK {variant} wall={wall} {gname}] built: NB {s.NB}, NF {s.NF}, dx {Gg['dx']:.4f} m, grid {Gg['shape']}, J {Gg['J']}; wall-trace nodes {nwall}; "
        f"{t_build:.0f} s; RSS {rss:.2f} GB, GPU peak {gpu:.2f} GB")
    rec.update(NB=int(s.NB), NF=int(s.NF), dx=float(Gg["dx"]), shape=list(Gg["shape"]), J=int(Gg["J"]), t_build=t_build, n_wall=nwall)
    t0 = time.time()
    inst = co.instantaneous()
    C0, C0_ot = static_stiffness(co)
    rec.update(A_inf=inst["A"].tolist(), B_inf=inst["B"].tolist(), B_inf_ot=inst["B_ot"].tolist(), C0=C0.tolist(), C0_ot=C0_ot.tolist(),
               C_s=np.zeros((3, 3)).tolist())
    log(f"  A_inf diag {np.diag(inst['A']).round(1).tolist()}; B_inf {inst['B'].round(0).tolist()}; "
        f"C0 heave/pitch {C0[1, 1]:.0f} / {C0[2, 2]:.0f} (OT {C0_ot[1, 1]:.0f} / {C0_ot[2, 2]:.0f}) [{time.time() - t0:.0f} s]")
    # steady NK problem (omega = 0, phin = U n_x)
    try:
        t0 = time.time()
        sol = s.solve_steady(p["u"] * s.body.n[:, 0])
        st = steady_forces(s, sol["phi_B"][:, 0], hull, p["sink"], p["trim"], cg=(0.0, 0.0))
        rec["steady"] = {k: float(v) for k, v in st.items()}
        log("  steady NK solution: " + ", ".join(f"{k} {v:.1f}" for k, v in rec["steady"].items()) + f" [{time.time() - t0:.0f} s]")
    except Exception as e:  # noqa: BLE001
        log(f"  steady NK failed: {e!r}")
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
                              tau_B=p["u"] * abs(we) / G, X=exc["X"].tolist(), X_fk=exc["X_fk"].tolist(), X_d=exc["X_d"].tolist(),
                              X_d_ot=exc["X_d_ot"].tolist(), X_fk_ramp=fkr.tolist(), A_e=rad["A"].tolist(), B_e=rad["B"].tolist(),
                              A_e_ot=(-rad["T_ot"].real / we ** 2).tolist(), B_e_ot=(rad["T_ot"].imag / we).tolist(),
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
    rss, gpu = mem_now()
    s.release()
    del s, co
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rec.update(t_total=time.time() - t_all, peak_rss_gb=rss, peak_gpu_gb=gpu, created=datetime.datetime.now().isoformat(timespec="seconds"))
    log(f"[NK {variant} {gname}] total {rec['t_total']:.0f} s; RSS {rss:.2f} GB, GPU peak {gpu:.2f} GB")
    with open(os.path.join(OUT, f"s1b_nk_{variant}{'' if wall == 'trace' else '_' + wall}_{gname}{label_suffix}.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="esg")
    ap.add_argument("--grids", default="base")
    ap.add_argument("--ramp", type=float, default=0.3)
    ap.add_argument("--ycfloor", type=float, default=2e-3)
    ap.add_argument("--no-extra", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--wall", default="trace", choices=("trace", "rule"))
    a = ap.parse_args()
    logf = open(os.path.join(OUT, f"s1b_nk_{a.variant}{'' if a.wall == 'trace' else '_' + a.wall}{a.label}.log"), "a", encoding="utf-8")

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
    for gname in a.grids.split(","):
        run(a.variant, gname, a.ramp, a.ycfloor, () if a.no_extra else EXTRA_OMEGAS, log, a.label, wall=a.wall)
    logf.close()


if __name__ == "__main__":
    main()
