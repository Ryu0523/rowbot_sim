#!/usr/bin/env python3
"""
V1 (forward-speed half), step 1-2 of learn/meta/HULL_FORCE_AUDIT_REVIEW_2026-10-09.md
section 4 row V1: build the 3-D linear frequency-domain quantities of scarab195 at
U = 5 kn (Fn_L 0.35) with the forward-speed solver of hydro/rankine -- added mass
A(omega_e), damping B(omega_e), instantaneous limits A_inf / B_inf, static
hydrodynamic stiffness C0 and frozen base-pressure stiffness C_s, and the wave
excitation (Froude-Krylov + diffraction) for head and following seas at Tp 3 and
5 s. Nothing under sim/ or hydro/ is edited; everything here is assembled from the
public functions of hydro/rankine (read-only).

Choices (all recorded in the npz/json):
  * linearisation: double body (hydro/rankine/doublebody.py, Track B step B6) --
    the one the repo built for flared / planing-type hulls (the Neumann-Kelvin
    wall closure is flagged unverified on a flared waterline, stage note 10.8).
    The repo's own record says flared-hull coefficients are NOT verified with
    either linearisation until the edge-flux operator exists (stage note 12.6
    item 9); this build is therefore a consistency basis for V1, not a validated
    prediction, and two grids and two stern closures are run to measure the
    sensitivity.
  * hull: the plant's own geometry (sim/planing_vessel.py stations: straight keel
    0.55 m below the CG, deadrise 20 deg, chine half-beam 1.0126 m with the
    elliptic taper over the forward 35 %, vertical sides above the chine), as a
    hydro/rankine/hull.HullForm in the plant's body frame (origin at the CG).
    The wet transom (draught 0.385 m, transom Froude number U/sqrt(g T) = 1.3,
    not ventilated) cannot be panelled by the unsteady solver (structured
    stations x rows mesh only), so it is CLOSED by a short smooth ramp of the
    keel to the waterline (C1 smoothstep, length `ramp`); the ramp's share of
    displacement and of the Froude-Krylov force is reported and two lengths are
    compared.
  * attitude: the plant's calm running attitude at 5 kn from ZarnickVessel
    .running_attitude (ZR configuration; rise +6.5 mm, trim 2.01 deg bow up),
    converted to hull.py's (sink, trim) convention.
  * moments about the CG; modes surge, heave, pitch (pitch positive bow down).

Outputs: studies/_cache/derive/v1b_tab_<label>.npz + .json (+ progress log
v1b_build_<label>.log when run detached).

Run (quick smoke):  .venv/Scripts/python.exe -m studies.derive_v1b_build --grid coarse --ramp 0.3 --nfreq 6
Full:               .venv/Scripts/python.exe -m studies.derive_v1b_build --grid base --ramp 0.3
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

G = 9.81
RHO = 1025.0
KNOT = 0.514444
U_KN = 5.0
TP_LIST = (3.0, 5.0)
HEADINGS = dict(head=np.pi, follow=0.0)
# free-surface grids (lengths in units of L = 5.4 m; the B5/B6 GRID_BASE of studies/rankine_unsteady_verify.py)
GRID_BASE = dict(up=1.5, down=4.0, lat=3.0, r_up=1.12, r_dn=1.06, r_lat=1.08, b_up=0.75, b_dn=2.0, b_lat=1.5,
                 u_dn=0.5, u_lat=0.3, nu_max=1.5)
GRIDS = {
    "coarse": (dict(GRID_BASE, M=32), (32, 7)),
    "base": (dict(GRID_BASE, M=48), (48, 10)),
    "fine": (dict(GRID_BASE, M=64), (64, 13)),
    # long upstream / lateral domain and beaches for the following-sea encounter frequencies (tau_B -> 1/4:
    # waves of 14-21 m run AHEAD of the hull; the transverse wavelength 2 pi g/omega_e^2 is 69 m at 0.94 rad/s)
    "long": (dict(GRID_BASE, M=48, up=5.0, b_up=4.0, lat=8.0, b_lat=6.0, down=6.0, b_dn=4.0), (48, 10)),
}
OMEGA_UNIFORM = np.round(np.arange(0.2, 8.0 + 1e-9, 0.1), 6)          # encounter frequencies for the memory fit
OMEGA_CRIT = np.array([0.93, 0.955, 0.97])                              # around g/(4U) = 0.954 rad/s


# ------------------------------------------------------------------ plant geometry and attitude
def plant_numbers(mlm=False):
    """The plant's hull parameters and its calm running attitude at 5 kn (ZR configuration), from the sim
    (read-only). Returns dict."""
    from studies.derive_v1_linear_consistency import make_vessel
    u = U_KN * KNOT
    v = make_vessel(mlm, u)
    z0, th0 = v.running_attitude(u)
    h = v.hull
    return dict(L=float(h.L), b=float(h.b), beta_deg=float(h.beta_deg), lcg=float(h.lcg), vcg=float(h.vcg),
                taper=float(h.taper), stem_rise=float(h.stem_rise), depth=float(h.depth), mass=float(h.mass),
                Iy=float(v.Iy), rho=float(v.rho), Z_rest=float(v.Z_rest), Th_rest=float(v.Th_rest),
                rise=float(z0), dth=float(th0), u=u, kp_zero=bool(np.all(v.kp == 0)),
                sink=-(float(v.Z_rest) + float(z0)), trim=-(float(v.Th_rest) + float(th0)),
                sink_rest=-float(v.Z_rest), trim_rest=-float(v.Th_rest))


def scarab_form(p, ramp=0.3, yc_floor=2e-3):
    """hull.HullForm of the plant's scarab195 geometry in the plant's body frame (x forward from the CG, z up
    from the CG; hull.py's 'design waterline' z_b = 0 is the CG height, so sink = -(CG height above water)).
    Stern closed by a smooth ramp of the keel from depth vcg at the transom (x_b = -lcg) to 0 at x_b = -lcg -
    ramp (C1 smoothstep as hull.wigley_transom). Half-breadth h = min((z + D)/tan(beta), y_chine(x)) (vertical
    sides above the chine), y_chine = 0.5 b sqrt(1 - s^2) + floor over the forward taper."""
    from hydro.rankine.hull import HullForm
    L, b, lcg, vcg, taper = p["L"], p["b"], p["lcg"], p["vcg"], p["taper"]
    tb = math.tan(math.radians(p["beta_deg"]))
    x0 = (1.0 - taper) * L                       # start of the chine taper, from the transom
    Lt = max(L - x0, 1e-9)

    def yc(xb):
        xft = np.asarray(xb, float) + lcg          # from the transom
        s = np.clip((xft - x0) / Lt, 0.0, 1.0)
        return 0.5 * b * np.sqrt(np.maximum(1.0 - s * s, 0.0)) + yc_floor

    def D(xb):
        xft = np.asarray(xb, float) + lcg
        u = np.clip((xft + ramp) / max(ramp, 1e-12), 0.0, 1.0)
        S = 3 * u * u - 2 * u ** 3
        return vcg * S - p["stem_rise"] * np.clip((xft - x0) / Lt, 0.0, 1.0) ** 2

    def h(xb, zb):
        xb = np.asarray(xb, float)
        zb = np.asarray(zb, float)
        return np.clip(np.minimum((zb + D(xb)) / tb, yc(xb)), 0.0, None)
    hull = HullForm(h, D, -lcg - ramp, L - lcg, f"scarab195_ramp{ramp:g}", above="form")
    hull.B, hull.T, hull.beta = b, vcg, math.radians(p["beta_deg"])
    hull.ramp = ramp
    hull.x_transom = -lcg
    return hull


# ------------------------------------------------------------------ build and sweep
def build(hull, p, grid_kw, hull_mesh, device=None, lin="DB"):
    """LTI system at the running attitude; returns (s, co, G, db). lin "DB": double body (primary); "NK":
    Neumann-Kelvin with the production wall-trace closure (tables.build_solver; db = None, C_s = 0) -- the second
    linearisation of the repo, run only to measure the model spread on this hull."""
    if lin == "NK":
        from hydro.rankine.tables import build_solver
        s, co, Gg = build_solver(hull, p["u"], p["sink"], p["trim"], grid_kw, hull_mesh, rho=RHO, g=G, ref=(0.0, 0.0),
                                 device=device, linearisation="NK")
        return s, co, Gg, None
    from hydro.rankine.doublebody import build_db_solver
    return build_db_solver(hull, p["u"], p["sink"], p["trim"], grid_kw, hull_mesh, g=G, rho=RHO, ref=(0.0, 0.0),
                           device=device)


def ramp_panels(hull, s, p):
    """Boolean mask of the body panels that lie on the stern ramp (body x < x_transom)."""
    c = s.body.c
    xb = hull.to_body(c[:, 0], c[:, 1], c[:, 2], p["sink"], p["trim"])[:, 0]
    return xb < hull.x_transom


def fk_on(s, co, mask, omega0, beta):
    """Classical Froude-Krylov force (rho int i omega0 phi_I n_j, both sides) restricted to the panels in mask."""
    from hydro.rankine.unsteady import incident
    phI, _, _ = incident(s.body.c, s.body.n, omega0, beta, G)
    a = s.body.area * mask
    return np.array([s.rho * 2 * np.sum(1j * omega0 * phI * co.mv[j][0] * a) for j in co.modes])


def mem_now():
    import psutil
    import torch
    rss = psutil.Process().memory_info().rss / 1e9
    gpu = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return rss, gpu


def run(label, grid_name, ramp, nfreq=None, mlm=False, wave_cases=True, log=print, lin="DB", yc_floor=2e-3):
    import torch
    from hydro.rankine.tables import validity_band, kernel_window_start
    from hydro.rankine.doublebody import frozen_pressure_stiffness, steady_db
    t_all = time.time()
    try:
        from studies.parallel_runner import free_gb
        log(f"free RAM at start {free_gb():.2f} GB")
    except Exception:
        pass
    p = plant_numbers(mlm)
    log(f"plant ({'MLM' if mlm else 'ZR'}): rest CG height {p['Z_rest']:+.4f} m, rest trim {-math.degrees(p['Th_rest']):+.3f} "
        f"deg bow up; 5 kn rise {p['rise']:+.4f} m, trim change {math.degrees(p['dth']):+.3f} deg -> hull.py sink "
        f"{p['sink']:+.4f} m, trim {math.degrees(p['trim']):+.3f} deg (bow up +); straight keel {p['kp_zero']}")
    hull = scarab_form(p, ramp, yc_floor=yc_floor)
    grid_kw, hm = GRIDS[grid_name]
    # hydrostatics of the form (geometry check): at the REST attitude the displaced volume must be m/rho up to the ramp
    hs_rest = hull.hydrostatics(p["sink_rest"], p["trim_rest"])
    hs_run = hull.hydrostatics(p["sink"], p["trim"])
    hull0 = scarab_form(p, ramp=1e-3)
    hs0 = hull0.hydrostatics(p["sink_rest"], p["trim_rest"])
    log(f"hull {hull.name}: L {hull.L:.3f} m incl. ramp {ramp} m; displaced volume at the rest attitude {hs_rest['V']:.4f} m3 "
        f"(plant m/rho {p['mass'] / RHO:.4f}; without ramp {hs0['V']:.4f} -> ramp share {100 * (hs_rest['V'] / hs0['V'] - 1):+.2f} %); "
        f"at the running attitude {hs_run['V']:.4f} m3, waterplane {hs_run['Aw']:.3f} m2, x_f {hs_run['xf']:+.3f} m")
    xs, xb = hull.waterline_ends(p["sink"], p["trim"])
    log(f"waterline at 5 kn: earth x from {xs:+.3f} to {xb:+.3f} m (length {xb - xs:.3f} m); transom draught "
        f"{float(hull.zeta_w(hull.x_transom, p['sink'], p['trim']) + hull.D(np.array(hull.x_transom))):.3f} m, chine height above keel "
        f"{(0.5 * p['b'] + 2e-3) * math.tan(math.radians(p['beta_deg'])):.3f} m")
    rec = dict(label=label, grid=grid_name, grid_kw=grid_kw, hull_mesh=list(hm), ramp=ramp, yc_floor=yc_floor, plant=p, hull_name=hull.name,
               linearisation=("double body (doublebody.build_db_solver, kinematic cons, m-terms sbp, gop fv)" if lin == "DB"
                              else "Neumann-Kelvin uniform stream, wall-trace closure, D-form (tables.build_solver NK)"),
               conventions="x forward, z up, pitch about +y positive bow DOWN, moments about the CG; time factor "
                           "e^{i omega_e t}; incident elevation zeta_I = Re e^{i(omega_e t - k x cos beta)} for unit amplitude; "
                           "F_rad = -T x with T = -omega^2 A + i omega B; equation of motion (-omega_e^2 M + T + C_hs + C_s) xi = X",
               V_rest=hs_rest["V"], V_rest_noramp=hs0["V"], V_run=hs_run["V"], Aw_run=hs_run["Aw"], xf_run=hs_run["xf"],
               IL_run=hs_run["IL"], zb_run=float(hs_run["centroid"][2]))
    t0 = time.time()
    s, co, Gg, db = build(hull, p, grid_kw, hm, lin=lin)
    t_build = time.time() - t0
    rss, gpu = mem_now()
    log(f"built: NB {s.NB} hull panels ({hm[0]}x{hm[1]}), NF {s.NF} free-surface panels (dx {Gg['dx']:.4f} m, grid {Gg['shape']}), "
        f"{t_build:.0f} s; RSS {rss:.2f} GB, GPU peak {gpu:.2f} GB")
    rec.update(NB=int(s.NB), NF=int(s.NF), dx=float(Gg["dx"]), t_build=t_build)
    if db is not None:
        W = db.W
        log(f"base flow on the hull: |W|/U max {np.linalg.norm(W, axis=1).max() / p['u']:.2f}, "
            f"at the free surface max {np.linalg.norm(co.fs_base()[0], axis=1).max() / p['u']:.2f}")
        rec["W_hull_max_over_U"] = float(np.linalg.norm(W, axis=1).max() / p["u"])
    wlo, whi = validity_band(Gg, p["u"], G, hull.L)
    wt = kernel_window_start(Gg, p["u"], G)
    log(f"validity band of this grid (tables.validity_band, 8 cells per shortest along-track wave; every beach >= 1 wavelength): "
        f"omega_e in [{wlo:.3f}, {whi:.3f}] rad/s; kernel window start (zero-speed wavelength = 4 cells) {wt:.2f} rad/s")
    rec.update(omega_lo=wlo, omega_hi=whi, omega_t=wt, tau_crit=G / (4 * p["u"]))
    # limits
    t0 = time.time()
    inst = co.instantaneous()
    if db is not None:
        C0, C0_ot = co.static_stiffness()
        Cs = frozen_pressure_stiffness(co.db, co.ref, RHO)
    else:
        from hydro.rankine.tables import static_stiffness
        C0, C0_ot = static_stiffness(co)
        Cs = np.zeros_like(C0)
    log(f"A_inf diag {np.diag(inst['A']).round(1).tolist()}; B_inf diag {np.diag(inst['B']).round(1).tolist()}; "
        f"C0 (D) heave/pitch {C0[1, 1]:.0f} / {C0[2, 2]:.0f}, C0 (OT) {C0_ot[1, 1]:.0f} / {C0_ot[2, 2]:.0f}; "
        f"C_s heave/pitch {Cs[1, 1]:.0f} / {Cs[2, 2]:.0f} [{time.time() - t0:.0f} s]")
    rec.update(A_inf=inst["A"], B_inf=inst["B"], C_inf=inst["C"], B_inf_ot=inst["B_ot"], C0=C0, C0_ot=C0_ot, C_s=Cs)
    # steady force at the attitude (information)
    try:
        if db is None:
            raise RuntimeError("NK: steady force not evaluated here")
        t0 = time.time()
        st = steady_db(s, co, hull, p["sink"], p["trim"], cg=(0.0, 0.0))
        rec["steady"] = {k: (float(v) if np.isscalar(v) else None) for k, v in st.items() if np.isscalar(v)}
        log(f"steady double-body solution at the attitude: " + ", ".join(f"{k} {v:.1f}" for k, v in rec["steady"].items()
                                                                           if v is not None) + f" [{time.time() - t0:.0f} s]")
    except Exception as e:      # noqa: BLE001
        log(f"steady_db failed: {e!r}")
    # frequency sweep
    omegas = OMEGA_UNIFORM if nfreq is None else np.linspace(0.3, 8.0, nfreq)
    omegas = np.unique(np.concatenate([omegas, OMEGA_CRIT]))
    m = len(co.modes)
    A = np.zeros((len(omegas), m, m))
    B = np.zeros_like(A)
    Aot, Bot = np.zeros_like(A), np.zeros_like(A)
    t0 = time.time()
    peak_rss = rss
    for i, w in enumerate(omegas):
        r = co.radiation(w)
        A[i], B[i] = -r["T"].real / w ** 2, r["T"].imag / w
        Aot[i], Bot[i] = -r["T_ot"].real / w ** 2, r["T_ot"].imag / w
        if i % 10 == 0 or i == len(omegas) - 1:
            rss, gpu = mem_now()
            peak_rss = max(peak_rss, rss)
            log(f"  omega_e {w:5.2f}: A33 {A[i, 1, 1]:8.1f} B33 {B[i, 1, 1]:8.1f} A55 {A[i, 2, 2]:8.1f} B55 {B[i, 2, 2]:8.1f} "
                f"A35 {A[i, 1, 2]:8.1f} A53 {A[i, 2, 1]:8.1f}  [{time.time() - t0:.0f} s, RSS {rss:.2f} GB, GPU {gpu:.2f} GB]")
    t_sweep = time.time() - t0
    log(f"sweep of {len(omegas)} frequencies: {t_sweep:.0f} s ({t_sweep / len(omegas):.2f} s each)")
    rec.update(omega=omegas, A=A, B=B, A_ot=Aot, B_ot=Bot, t_sweep=t_sweep)
    # passivity
    Bs = 0.5 * (B + np.swapaxes(B, 1, 2))
    ev = np.linalg.eigvalsh(Bs)
    rel = ev[:, 0] / np.maximum(np.abs(ev).max(axis=1), 1e-300)
    bad = omegas[rel < -1e-9]
    log(f"passivity of B(omega_e) (min eigenvalue of the symmetric part / max |eig|): worst {rel.min():+.3f} at "
        f"{omegas[np.argmin(rel)]:.2f} rad/s; non-passive frequencies: {bad.round(2).tolist() if len(bad) else 'none'}")
    rec.update(bsym_rel=rel)
    # wave cases
    if wave_cases:
        cases = {}
        mask = ramp_panels(hull, s, p)
        t0 = time.time()
        for hd, beta in HEADINGS.items():
            for Tp in TP_LIST:
                w0 = 2 * np.pi / Tp
                rad, exc = co.rad_exc(w0, beta)
                we = exc["omega_e"]
                fkr = fk_on(s, co, mask, w0, beta)
                key = f"{hd}_Tp{Tp:g}"
                cases[key] = dict(heading=hd, beta=beta, Tp=Tp, omega0=w0, k=exc["k"], omega_e=we, tau_B=p["u"] * abs(we) / G,
                                  X=exc["X"], X_ot=exc["X_ot"], X_fk=exc["X_fk"], X_d=exc["X_d"], X_fk_ramp=fkr,
                                  A_e=rad["A"], B_e=rad["B"])
                kk = exc["k"]
                log(f"  {key}: omega0 {w0:.4f}, k {kk:.4f} 1/m, omega_e {we:+.4f}, tau_B {p['u'] * abs(we) / G:.3f}; "
                    f"|X3| {abs(exc['X'][1]):.0f} N/m (FK {abs(exc['X_fk'][1]):.0f}, ramp FK share {abs(fkr[1]) / max(abs(exc['X_fk'][1]), 1e-9):.3f}), "
                    f"|X5|/k {abs(exc['X'][2]) / kk:.0f} N m/m (FK {abs(exc['X_fk'][2]) / kk:.0f}); "
                    f"A33(we) {rad['A'][1, 1]:.0f} B33(we) {rad['B'][1, 1]:.0f} A55 {rad['A'][2, 2]:.0f} B55 {rad['B'][2, 2]:.0f}")
        log(f"wave cases: {time.time() - t0:.0f} s")
        rec["cases"] = cases
    rss, gpu = mem_now()
    peak_rss = max(peak_rss, rss)
    s.release()
    del s, co, db
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rec.update(t_total=time.time() - t_all, peak_rss_gb=peak_rss, peak_gpu_gb=gpu,
               created=datetime.datetime.now().isoformat(timespec="seconds"))
    log(f"total {rec['t_total']:.0f} s ({rec['t_total'] / 60:.1f} min); peak RSS {peak_rss:.2f} GB, peak GPU {gpu:.2f} GB")
    save(rec, label)
    return rec


def save(rec, label):
    path = os.path.join(OUT, f"v1b_tab_{label}.npz")
    flat = {}
    meta = {}

    def put(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                put(f"{prefix}{k}.", v)
            elif isinstance(v, np.ndarray):
                flat[prefix + k] = v
            elif isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], (int, float, np.floating, np.integer)):
                flat[prefix + k] = np.asarray(v)
            else:
                meta[prefix + k] = v
    put("", rec)
    np.savez(path, **flat)
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=str)
    return path


def load(label):
    """Re-assemble the nested record from the npz + json."""
    path = os.path.join(OUT, f"v1b_tab_{label}.npz")
    z = np.load(path)
    meta = json.load(open(os.path.splitext(path)[0] + ".json", encoding="utf-8"))
    rec = {}
    for src in (dict(z.items()), meta):
        for k, v in src.items():
            parts = k.split(".")
            d = rec
            for pp in parts[:-1]:
                d = d.setdefault(pp, {})
            d[parts[-1]] = v
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="base", choices=sorted(GRIDS))
    ap.add_argument("--ramp", type=float, default=0.3)
    ap.add_argument("--nfreq", type=int, default=None, help="quick: this many uniform frequencies instead of the 79")
    ap.add_argument("--label", default=None)
    ap.add_argument("--mlm", action="store_true")
    ap.add_argument("--no-cases", action="store_true")
    ap.add_argument("--lin", default="DB", choices=("DB", "NK"))
    ap.add_argument("--ycfloor", type=float, default=2e-3, help="floor of the chine half-breadth at the stem (m); the plant uses 1e-4")
    a = ap.parse_args()
    label = a.label or f"{a.grid}_r{a.ramp:g}" + ("_mlm" if a.mlm else "") + (f"_n{a.nfreq}" if a.nfreq else "") + ("_NK" if a.lin == "NK" else "") + (f"_yc{a.ycfloor:g}" if a.ycfloor != 2e-3 else "")
    logf = open(os.path.join(OUT, f"v1b_build_{label}.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    run(label, a.grid, a.ramp, nfreq=a.nfreq, mlm=a.mlm, wave_cases=not a.no_cases, log=log, lin=a.lin, yc_floor=a.ycfloor)
    logf.close()


if __name__ == "__main__":
    main()
