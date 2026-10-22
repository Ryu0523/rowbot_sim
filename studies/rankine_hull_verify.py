#!/usr/bin/env python3
"""
Track B step B4: surface-piercing hull (Wigley) and transom stern --
verification and validation tiers (acceptance learn/meta/ACCEPTANCE_2026-10-06.md
T1: quick <= 10 min, full <= 1 h). Solver: hydro/rankine/piercing.py (linear
Neumann-Kelvin free surface with the shifted Numerov operator of B3, body-fitted
band next to the hull, exact hull and hydrostatics at the attitude, free
sinkage and trim, dry/wet transom branches). Registration: stage note
learn/meta/RANKINE_STAGE1_2026-10-06.md section 9.1 (written, and hashed with
--register, BEFORE the tiers ran); derivations section 9.2; exploration runs
9.3; results 9.4.

    .venv/Scripts/python.exe studies/rankine_hull_verify.py --tier quick [--only H0,V1]
    .venv/Scripts/python.exe studies/rankine_hull_verify.py --register "note"

Rows: H* verification (exact or semi-analytic references, identities, limits,
convergence), V* validation against measurements (Kajitani et al. 1983 data as
transcribed in the files of data/external/wigley/, SOURCES.md), T* transom
branches (no measured data exist for the test geometry: verification and
information only). Binding rows carry a registered tolerance; information rows
("info") pass by construction and are counted separately.

B4 review fixes (stage note 9.7, registered before the re-run): V1 kept as
registered but marked "invalid reference" (the WaveBEM C_W files do not
reproduce the scan's markers at Fn < 0.3); amended rows added after a run
(computed values known): V1a hull force R_p vs the U. Tokyo C_W curve, V1b
R_near (wave energy flux) vs the U. Tokyo C_WP markers, V5a free-model R_p vs
the free C_W curve, tolerances max(20 %, absolute floor from the measured lab
scatter); transom momentum identity T3 binding, D1 residual T6, COARSE/PROD
pair T7, wet free equilibria T5; hull-wall ghost with the section-curvature
and flare terms and the flared-hull check H13; closed-form Michell (H8a) and the
extrapolation uncertainty of H8; separate free-surface and hull grid series
with observed orders and the waterline-estimator convergence (H11b-e); H6 at
Fn 0.5; sha256 of the solver and check code in the registration entries and in
every tier JSON (acceptance N2).
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from studies.rankine_verify import _row, _richardson, _fmt_rich   # noqa: E402

OUT = os.path.join(ROOT, "studies", "_cache", "rankine")
DATA = os.path.join(ROOT, "data", "external", "wigley")
G0 = 9.81
RHO = 1.0
S_WIGLEY = None       # wetted surface at rest (computed once, wigley_wetted_surface)

# production settings (registered, stage note 9.1)
PROD = dict(n_per_lam=30, m_body=90, hull_mesh=(90, 18), J_mult=2.0)
COARSE = dict(n_per_lam=30, m_body=60, hull_mesh=(60, 12), J_mult=2.0)
FN_FIXED = (0.2, 0.25, 0.267, 0.289, 0.316, 0.354, 0.408)
FN_FREE = (0.25, 0.267, 0.289, 0.316, 0.354, 0.374, 0.408)
CUT_D = 1.0           # cut 1 lambda0 behind the stern
# B4 review (stage note 9.7): amended validation rows and their tolerances
FN_H6_EXTRA = (0.5,)                 # near/far only (no data); validity statement up to 0.5
TOL_REL = 0.20                       # relative part (design proposal, unchanged)
FLOOR_CW = 0.20e-3                   # C_W (C_T-based) absolute floor: S.R.I./U.T. markers vs U.T. curve, Fn 0.19-0.21
FLOOR_CWP = 0.10e-3                  # C_WP absolute floor: S.R.I. vs U.T. at Fn 0.35/0.354 + 2 x reading error
D1_TOL = 0.10                        # transom edge elevation residual at PROD (T6)


def _gpu():
    import torch
    return torch.cuda.is_available()


def _free():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def wigley_wetted_surface(n=400):
    """Wetted surface of the Wigley form at rest (both sides), L = 1:
    S = 2 int int sqrt(1 + h_x^2 + h_z^2) dz dx (Gauss-Legendre)."""
    from hydro.rankine.hull import wigley
    h = wigley()
    xg, wx = np.polynomial.legendre.leggauss(n)
    zg, wz = np.polynomial.legendre.leggauss(n)
    x = 0.5 * xg
    z = -h.T * 0.5 * (zg + 1)
    X, Z = np.meshgrid(x, z, indexing="ij")
    hx = -0.5 * h.B * 8 * X * (1 - Z ** 2 / h.T ** 2)
    hz = 0.5 * h.B * (1 - 4 * X ** 2) * (-2 * Z / h.T ** 2)
    return float(2 * np.sum(np.sqrt(1 + hx ** 2 + hz ** 2) * (0.5 * wx)[:, None] * (0.5 * h.T * wz)[None, :]))


def _q(U):
    global S_WIGLEY
    if S_WIGLEY is None:
        S_WIGLEY = wigley_wetted_surface()
    return 0.5 * RHO * U * U * S_WIGLEY


def _res(infos):
    infos = [i for i in infos if i]
    if not infos:
        return "no solve"
    rr = [i["relres"] for i in infos]
    it = [i.get("iters", 0) for i in infos]
    gl = max(i.get("guard_l", 0.0) for i in infos)
    return (f"{len(infos)} solve(s): true residual max {max(rr):.1e} (floor 1e-10), iterations "
            f"{min(it)}-{max(it)}, interior guard blend max {gl:.3f}")


_SOLVES = {}


def _solve_fixed(Fn, cfg, hull=None, sink=0.0, trim=0.0, **kw):
    """One captive solve: forces, d'Alembert-corrected near field with the
    waterline term (hull-side zeta; the free-surface estimate R_wl_fs as
    information), wave-cut far field, waterline profile. Memoised per
    (Fn, hull, attitude, settings) within a tier (the same solve feeds V, H6,
    H8 and H11 members)."""
    from hydro.rankine.hull import wigley
    from hydro.rankine.piercing import PiercingNK, cut_resistance
    hull = hull or wigley()
    key = (Fn, hull.name, sink, trim, tuple(sorted((k, str(v)) for k, v in {**cfg, **kw}.items())))
    if key in _SOLVES:
        return _SOLVES[key]
    U = Fn * np.sqrt(G0 * hull.L)
    t = time.time()
    s = PiercingNK(hull, U, G0, sink=sink, trim=trim, rho=RHO, **cfg, **kw)
    sol = s.solve()
    F = s.forces(phi_db=s.double_body() if not s.transom else None)
    Rc, ci = cut_resistance(s, s.G["grid"]["x_s"] - CUT_D * s.lam, RHO)
    xw, yw, zw = s.waterline_profile("hull")
    R_wl_fs = s.waterline_term("fs")
    out = dict(Fn=Fn, U=U, q=_q(U), F=F, R_cut=Rc, x_cut=ci["x_cut"], th_max=ci["th_max"], xw=xw, zw=zw,
               R_wl_fs=R_wl_fs, info=sol["info"], NF=s.NF, NB=s.NB, ND=s.ND, n_eff=s.G["grid"]["n_eff"],
               seconds=time.time() - t, grid=dict(dx=s.G["grid"]["dx"], J=s.G["grid"]["J"], K=s.G["grid"]["K"]))
    s.release()
    del s
    _free()
    _SOLVES[key] = out
    return out


# =====================================================================
# data
def _exp_cw_fixed():
    """The four C_W series (fixed model) digitised by the WaveBEM authors
    from Hendrix & Noblesse's figure (labels SRI, UT1, UT2, UT3; Y.N.U. is not
    digitised; at Fn < 0.3 the files follow the theory lines, not the
    markers: invalid reference, stage note 9.7.1)."""
    out = {}
    for k, name in ((1, "SRI"), (2, "UT1"), (3, "UT2"), (4, "UT3")):
        a = np.loadtxt(os.path.join(DATA, f"wavebem_wigleyDragExp{k}.csv"), delimiter=",", comments="%")
        out[name] = a[np.argsort(a[:, 0])]
    return out


def _exp_mean_at(series, Fn):
    vals = [float(np.interp(Fn, a[:, 0], a[:, 1])) for a in series.values() if a[0, 0] <= Fn <= a[-1, 0]]
    return (float(np.mean(vals)), float(min(vals)), float(max(vals)), len(vals)) if vals else (np.nan,) * 3 + (0,)


def _yang(name):
    a = np.loadtxt(os.path.join(DATA, f"yang2000_{name}.csv"), delimiter=",", comments="#")
    a = a[np.argsort(a[:, 0], kind="stable")]
    return a


def _yang_at(a, Fn):
    # vertices lie at the measured Fn; average duplicates
    m = np.abs(a[:, 0] - Fn) < 2e-3
    if m.any():
        return float(a[m, 1].mean())
    return float(np.interp(Fn, a[:, 0], a[:, 1]))


def _cwp_ut():
    """U. Tokyo C_WP markers of Hendrix & Noblesse (1988) p. 61, read by
    studies/rankine_wigley_scan.py (labs "UT" and "SRI+UT" = overlapping S.R.I.
    and U.T. markers): array (Fn, C_WP, reading error of C_WP)."""
    rows = []
    for ln in open(os.path.join(DATA, "hn1988_p61_cwp_markers.csv"), encoding="utf-8"):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.strip().split(",")
        if "UT" in f[2]:
            rows.append((float(f[0]), float(f[1]) * 1e-3, float(f[4]) * 1e-3))
    return np.array(rows)


def _cwp_at(Fn, tol_fn=0.003):
    a = _cwp_ut()
    j = int(np.argmin(abs(a[:, 0] - Fn)))
    return (float(a[j, 1]), float(a[j, 0]), float(a[j, 2])) if abs(a[j, 0] - Fn) <= tol_fn else (None, None, None)


def _tol(ref, floor):
    """Registered tolerance of the amended rows (stage note 9.7): max(20 % of
    the reference, absolute floor from the measured lab scatter)."""
    return max(TOL_REL * abs(ref), floor)


def _profile_exp(Fn):
    """U. Tokyo wave profile (fixed model) at Fn: x/L in OUR frame (x
    forward; the file's x is WaveBEM's, bow at -0.5, so x_ours = -x_file)
    and 2 g zeta/U^2 (Curve1 = U. Tokyo)."""
    tag = {0.25: "025", 0.267: "0267", 0.289: "0289", 0.316: "0316", 0.354: "0354", 0.408: "0408"}[Fn]
    a = np.loadtxt(os.path.join(DATA, f"wavebem_tokyoFr{tag}.csv"), delimiter=",", comments="%")
    return -a[:, 0], a[:, 1]


# =====================================================================
# H: verification
def h0_hydrostatics(args):
    """H0 (M2, M3): exact hydrostatics of the Wigley form vs the closed form
    V = 4 L B T/9, z_c = -3 T/8, A_w = 2 L B/3 (rel. 1e-12); the free-
    equilibrium residual at U = 0 at the rest attitude (exactly 0) and the
    hydrostatic Newton from a perturbed attitude back to rest (|F - W| and
    |M|/L <= 1e-9 W; acceptance M3); panel-integrated displacement order
    (info)."""
    from hydro.rankine.hull import wigley
    h = wigley()
    hs = h.hydrostatics()
    eV = abs(hs["V"] / (4 * h.L * h.B * h.T / 9) - 1)
    ez = abs(hs["centroid"][2] / (-3 * h.T / 8) - 1)
    eA = abs(hs["Aw"] / (2 * h.L * h.B / 3) - 1)
    W = RHO * G0 * hs["V"]
    cg = (float(hs["centroid_body"][0]), 0.0)
    x = np.array([0.004, -0.01])
    hist = []
    for it in range(30):
        hl = h.hydrostatic_load(x[0], x[1], RHO, G0, cg)
        r = np.array([(hl["Fz"] - W) / W, hl["My"] / (W * h.L)])
        hist.append(float(np.abs(r).max()))
        if np.abs(r).max() < 1e-14:
            break
        J = np.zeros((2, 2))
        for c in range(2):
            e = np.zeros(2)
            e[c] = 1e-6
            hp = h.hydrostatic_load(*(x + e), RHO, G0, cg)
            hm = h.hydrostatic_load(*(x - e), RHO, G0, cg)
            J[:, c] = (np.array([hp["Fz"], hp["My"] / h.L]) - np.array([hm["Fz"], hm["My"] / h.L])) / (2e-6 * W)
        x = x - np.linalg.solve(J, r)
    Vp = []
    for n in (20, 40, 80):
        pan = h.mesh(2 * n, n // 2)
        Vp.append(float(2 * np.sum(pan.c[:, 2] * pan.n[:, 2] * pan.area)))
    ep = [abs(v / hs["V"] - 1) for v in Vp]
    return [_row("H0", "[B4] exact hydrostatics vs closed form: |V/(4LBT/9) - 1|, |z_c/(-3T/8) - 1|, |A_w/(2LB/3) - 1|",
                 [eV, ez, eA], 0.0, "closed form (Wigley polynomial)", 1e-12, max(eV, ez, eA) <= 1e-12),
            _row("H0", "[B4] U = 0: hydrostatic Newton from (s, theta) = (0.004, -0.01) back to rest: final |F - W|/W, "
                 "|M|/(W L); attitude reached", [hist[-1], float(x[0]), float(x[1])], 0.0,
                 "acceptance M3 (equilibrium at rest to 1e-9 W)", 1e-9,
                 hist[-1] <= 1e-9 and abs(x[0]) < 1e-9 and abs(x[1]) < 1e-9),
            _row("H0", "[B4] panelled displacement (divergence theorem on the hull panels) relative error at 40x10, "
                 "80x20, 160x40 (not used for forces)", ep, 0.0, "exact V", "info: order "
                 f"{np.log(ep[0] / ep[1]) / np.log(2):.2f}, {np.log(ep[1] / ep[2]) / np.log(2):.2f}", True)]


def h1_fft_dense(args):
    """H1 (M2): FFT-GMRES vs dense LU of the same discrete system on a small
    Wigley grid (Fn 0.316, n 12/lambda0, m_body 30) and a small dry-transom
    grid (Fn 0.45): relative difference of phi_B, sigma and R_near <= 1e-8."""
    from hydro.rankine.hull import wigley, wigley_transom
    from hydro.rankine.piercing import PiercingNK
    rows = []
    for name, hull, Fn, kw in (("Wigley", wigley(), 0.316, dict(n_per_lam=12, m_body=30, hull_mesh=(30, 6))),
                               ("Wigley-transom dry", wigley_transom(), 0.45,
                                dict(n_per_lam=12, m_body=30, hull_mesh=(30, 6), face_mesh=(4, 4), branch="dry"))):
        U = Fn * np.sqrt(G0 * hull.L)
        s = PiercingNK(hull, U, G0, mult=(0.5, 0.4, 0.5, 1, 1), **kw)
        a = s.solve(method="fft")
        pa, sa = a["phi_B"].copy(), a["sigma"].copy()
        Fa = s.forces()
        b = s.solve(method="dense")
        Fb = s.forces()
        e1 = float(np.abs(pa - b["phi_B"]).max() / np.abs(b["phi_B"]).max())
        e2 = float(np.abs(sa - b["sigma"]).max() / np.abs(b["sigma"]).max())
        e3 = abs(Fa["R_raw"] / Fb["R_raw"] - 1)
        rows.append(_row("H1", f"[B4] {name}: FFT-GMRES vs dense LU (N = {s.NB + s.NF}): rel. diff phi_B, sigma, R",
                         [e1, e2, e3], 0.0, "dense LU, same matrix", 1e-8, max(e1, e2, e3) <= 1e-8,
                         _res([a["info"], b["info"]])))
        s.release()
        _free()
    return rows


def h2_lattice_rows(args):
    """H2 (M2): the B4 row builder reproduces the B3 rows (shifted.py) on
    every lattice row of a Wigley grid that does not touch the band (max abs
    difference of the A, Bk, C entries, 1e-14 of the largest entry)."""
    from hydro.rankine.hull import wigley
    from hydro.rankine.piercing import build_grid, build_rows
    from hydro.rankine.shifted import mass_operator, shifted_operators, blend_weights, lattice_zmax
    h = wigley()
    Fn = 0.316
    U = Fn * np.sqrt(G0)
    k = U * U / G0
    G = build_grid(h, 0.0, 0.0, 2 * np.pi * k, 20, 40, (0.5, 0.4, 0.5, 1, 1), 1.0, 2.0)
    g = G["grid"]
    zg = lattice_zmax(g["dx"], g["dy"], k, True)
    R = build_rows(G, k, zg, True)
    nx, ny = g["nx"], g["ny"]
    w = np.tile(g["w_col"], ny)
    nu = np.tile(g["nu_col"], ny)
    Bk3, C3 = shifted_operators(nx, ny, g["dx"], g["dy"], w, nu, True)
    lam = blend_weights(g["w_col"], zg, True)
    A3 = mass_operator(nx, ny, lam)
    NL = nx * ny
    # rows whose stencils stay on the lattice: lines j >= J + 2 (all columns)
    sel = np.arange(NL)[(G["ij"][:NL, 1] >= g["J"] + 2) & (G["ij"][:NL, 1] < ny - 1)]
    dA = abs(R["A"][sel][:, :NL] - A3[sel]).max()
    dB = abs(R["Bk"][sel][:, :NL] - Bk3[sel]).max()
    dC = abs(R["C"][sel][:, :NL] - C3[sel]).max()
    sc = max(abs(A3).max(), abs(Bk3).max(), abs(C3).max())
    e = float(max(dA, dB, dC) / sc)
    return [_row("H2", f"[B4] lattice rows (lines >= J + 2, {len(sel)} rows): B4 builder vs B3 shifted.py, "
                 "max |difference| of A, Bk, C / max entry", e, 0.0, "B3 operator (identity)", 1e-14, e <= 1e-14)]


def h4_split(args):
    """H4 (M4, analogue of S4b): O(Fn^2) coefficient c = (Fz/(U^2 Fz_db) - 1)/Fn^2
    of the implemented solve vs the split phi = phi_db + psi with the exact
    double-body forcing at the row targets (stage note 9.2), Fn 0.25, coarse
    production grid, guard inactive: |c_impl/c_split - 1| <= 5 %."""
    import torch
    from hydro.rankine.hull import wigley
    from hydro.rankine.piercing import PiercingNK, _Columns
    from hydro.rankine.steady import rigid_lid_field
    Fn = args[0]
    h = wigley()
    U = Fn * np.sqrt(G0)
    s = PiercingNK(h, U, G0, rho=RHO, **COARSE)
    a = s.solve()
    phidb = s.double_body()
    Fi = s.forces(phi_db=phidb)
    # exact double-body second derivative at the row targets
    g = s.G["grid"]
    P = s.P
    xe = g["xe"]
    xm = 0.5 * (xe[:-1] + xe[1:])
    ij = s.G["ij"]
    hh = 0.25 * g["dx"]
    tgt = {}
    rowsinfo = []
    lines = s.lines
    from hydro.rankine.shifted import blend_weights
    lam_c = blend_weights(g["w_col"], s.zgrid, True)
    for j, pts in lines.items():
        for (r, kr) in pts:
            c = r - 1
            if c < 1:
                continue
            kc = s.G["index"][(c, j)]
            yc = P[kc, 1]
            rowsinfo.append((kr, c, yc))
            for cc in (c - 1, c, c + 1):
                tgt[(cc, round(yc, 12))] = None
    keys = list(tgt)
    pts3 = np.array([[xm[cc], y, 0.0] for cc, y in keys])
    pn = s.U * s.body.n[:, 0]
    vals = {}
    for m in (-2, -1, 0, 1, 2):
        vals[m] = rigid_lid_field(s.body, phidb, pn, pts3 + [m * hh, 0.0, 0.0], device="cuda" if _gpu() else "cpu")
    fx = (-vals[2] + 8 * vals[1] - 8 * vals[-1] + vals[-2]) / (12 * hh)
    fxx = (-vals[2] + 16 * vals[1] - 30 * vals[0] + 16 * vals[-1] - vals[-2]) / (12 * hh * hh)
    d2 = {kk: v for kk, v in zip(keys, fxx)}
    d1 = {kk: v for kk, v in zip(keys, fx)}
    am = np.array([1.0, 10.0, 1.0]) / 12
    forcing = np.zeros(s.NF)
    for kr, c, yc in rowsinfo:
        l = lam_c[c]
        key = lambda cc: (cc, round(yc, 12))
        num = (1 - l) * (am[0] * d2[key(c + 1)] + am[1] * d2[key(c)] + am[2] * d2[key(c - 1)]) + l * d2[key(c)]
        forcing[kr] = s.k * g["w_col"][c] * (num - g["nu_col"][c] * d1[key(c)])
    # psi: body condition 0; free-surface rows with the exact base forcing
    s.phin = np.zeros(s.NB)
    s.SFBpn.zero_()
    s.rhs_cphi = np.zeros(s.NF)
    b = s.solve(fs_forcing=forcing)
    phi_tot = phidb + b["phi_B"]
    s.phin = s.U * s.body.n[:, 0]
    s.sol = dict(phi_B=phi_tot, sigma=b["sigma"], info=b["info"])
    Fs = s.forces(phi_db=phidb)
    fdb = Fi["Fz_db"]
    ci = (Fi["Fz"] / fdb - 1) / Fn ** 2
    cs = (Fs["Fz"] / fdb - 1) / Fn ** 2
    s.release()
    _free()
    e = abs(ci / cs - 1)
    return [_row("H4", f"[B4] Fn {Fn}: c = (Fz/Fz_db - 1)/Fn^2 implemented vs split with exact double-body forcing "
                 f"(coarse production grid, guard l = {a['info']['guard_l']:.3f})", [ci, cs, e], None,
                 "split formulation (exact base flow at the row targets)", 0.05,
                 e <= 0.05 and a["info"]["guard_l"] == 0.0, _res([a["info"], b["info"]]))]


def h5_similarity(args):
    """H5 (M3): Froude similarity, Wigley L = 1 and L = 1.5 at Fn 0.316 on
    the coarse grid: max relative difference of R_near/(rho g L^3), R_cut,
    Fz, My/L <= 1e-4."""
    from hydro.rankine.hull import wigley
    from hydro.rankine.piercing import PiercingNK, cut_resistance
    out = []
    for L in (1.0, 1.5):
        h = wigley(L=L, B=0.1 * L, T=0.0625 * L)
        U = 0.316 * np.sqrt(G0 * L)
        s = PiercingNK(h, U, G0, n_per_lam=20, m_body=40, hull_mesh=(40, 8), mult=(0.5, 0.4, 0.5, 1, 1))
        s.solve()
        F = s.forces(phi_db=s.double_body())
        Rc, _ = cut_resistance(s, s.G["grid"]["x_s"] - s.lam)
        sc = RHO * G0 * L ** 3
        out.append(np.array([F["R_near"] / sc, Rc / sc, F["Fz"] / sc, F["My"] / (sc * L)]))
        s.release()
        _free()
    e = float(np.max(np.abs(out[1] / out[0] - 1)))
    return [_row("H5", "[B4] Froude similarity L = 1 vs 1.5 at Fn 0.316: max rel. diff of R_near, R_cut, Fz, My (/ rho g L^3)",
                 e, 0.0, "exact", 1e-4, e <= 1e-4)]


def h7_cut_calibration(args):
    """H7 (M2): the wave-cut functional (registered: cut 1 lambda0 behind the
    body, lateral wavenumbers <= 0.5 pi/dy) applied to the EXACT linear
    Kelvin elevation of Havelock's spheroid line source (havelock.py,
    farfield.kelvin_wave_x) vs Havelock's R: |R_cut/R_H - 1| <= 1 % for
    (Fn, f) = (0.4, 4), (0.5, 2.5), (0.3, 1.5) m, L = 10 m, n = 20 and 30
    panels per lambda0 (CPU)."""
    from hydro.rankine.havelock import spheroid_havelock, line_source_points
    from hydro.rankine.farfield import kelvin_wave_x
    from hydro.rankine.piercing import cut_from_samples, CUT_BETA_FRAC
    a, b, L = 5.0, 1.0, 10.0
    errs = []
    for Fn, f in ((0.4, 4.0), (0.5, 2.5), (0.3, 1.5)):
        U = Fn * np.sqrt(G0 * L)
        k = U * U / G0
        lam = 2 * np.pi * k
        Rh = spheroid_havelock(a, b, f, U, G0)
        Q, X = line_source_points(a, b, f, U, 64)
        for nl in (20, 30):
            dy = lam / nl
            Y = 0.8 * L + 0.5 * lam
            ny = int(np.ceil(Y / dy))
            Y = ny * dy
            yy = (np.arange(ny) + 0.5) * dy
            x0 = -a - CUT_D * lam
            hstep = dy / 20
            Pm = np.stack([np.full(ny, x0), yy, np.zeros(ny)], 1)
            zf = lambda d: (U / G0) * kelvin_wave_x(Q, X, Pm + [d, 0, 0], 1 / k, device="cpu")
            z0 = zf(0.0)
            zx = (zf(hstep) - zf(-hstep)) / (2 * hstep)
            R, _ = cut_from_samples(yy, z0, zx, Y, U, k, 1.0, CUT_BETA_FRAC * np.pi / dy)
            errs.append(R / Rh - 1)
    m = float(np.max(np.abs(errs)))
    return [_row("H7", "[B4; values known from exploration B4-E4] wave-cut functional on the exact Kelvin field "
                 "(3 cases x n 20, 30): R_cut/R_H - 1", [float(e) for e in errs], 0.0,
                 "Havelock 1931 (exact for these equations)", 0.01, m <= 0.01)]


def h8_thin_ship(args):
    """H8 (design S1 limit check, M4-type): R_near(B)/R_Michell(B) of the
    Wigley family B/L = 0.1, 0.05, 0.025 (T/L 0.0625) at Fn 0.316 on the
    production grid; extrapolated to B -> 0 by a quadratic in B through the
    three: |limit - 1| <= 3 % (it is a limit, not a ceiling: the finite-B
    values are information). B4 review: Michell's integral in closed form in
    x and z (thinship.michell_wigley; H8a checks its Kochin function against a
    direct quadrature, 1e-8); the extrapolation weights (1/3, -2, 8/3) amplify
    per-point errors, so the uncertainty of the limit is propagated from the
    per-point COARSE -> PROD change (information); R_p + strip and the cut
    are information (they also tend to Michell: H8 cannot test the sign of the
    waterline term, only H6 does)."""
    from hydro.rankine.hull import wigley
    from hydro.rankine.thinship import michell_wigley
    Bs = (0.1, 0.05, 0.025)
    w = np.array([1.0 / 3.0, -2.0, 8.0 / 3.0])
    r, rc, rs, rcoarse, infos = [], [], [], [], []
    for B in Bs:
        h = wigley(B=B)
        o = _solve_fixed(0.316, PROD, hull=h)
        oc = _solve_fixed(0.316, COARSE, hull=h)
        Rm = michell_wigley(o["U"], G0, RHO, 1.0, B, 0.0625)
        r.append(o["F"]["R_near"] / Rm)
        rc.append(o["R_cut"] / Rm)
        rs.append((o["F"]["R_corr"] + o["F"]["R_wl"]) / Rm)
        rcoarse.append(oc["F"]["R_near"] / Rm)
        infos += [o["info"], oc["info"]]
    cn = np.polyfit(Bs, r, 2)
    cc = np.polyfit(Bs, rc, 2)
    lim, limc = float(cn[-1]), float(cc[-1])
    sig = float(np.sqrt(np.sum((w * (np.array(r) - np.array(rcoarse))) ** 2)))
    lims = float(w @ np.array(rs))
    limco = float(w @ np.array(rcoarse))
    return [_row("H8", "[B4; values partly known (exploration B4-E5, older waterline estimate)] thin-ship limit at Fn 0.316: "
                 f"R_near/R_Michell at B/L = {list(Bs)}; quadratic extrapolation to B = 0",
                 [float(x) for x in r] + [lim], 1.0, "Michell 1898 thin-ship integral (closed form, thinship.michell_wigley)",
                 0.03, abs(lim - 1) <= 0.03,
                 f"B4 review: uncertainty of the limit from the per-point COARSE -> PROD change propagated through the "
                 f"weights (1/3, -2, 8/3): +-{sig:.3f} (COARSE ratios {np.round(rcoarse, 4).tolist()}, COARSE limit "
                 f"{limco:.4f}); {_res(infos)}"),
            _row("H8", "[B4] same with the wave-cut far field: R_cut/R_Michell; extrapolated", [float(x) for x in rc] + [limc],
                 1.0, "Michell", "info", True),
            _row("H8", "[B4 review, info] R_p + strip (the B4 sign error) / R_Michell; extrapolated: also tends to Michell, "
                 "so H8 does not test the waterline-term sign (H6 does)", [float(x) for x in rs] + [lims], 1.0, "Michell",
                 "info", True)]


def h8a_michell_closed_form(args):
    """H8a (M2, added in the B4 review): the closed-form Kochin function of the
    Wigley centre-plane sheet (thinship.wigley_kochin) vs a direct adaptive
    quadrature of the x and z integrals at theta = 0.1, 0.5, 1.0, 1.3, 1.5 and
    Fn 0.2, 0.316, 0.5: max relative difference <= 1e-8."""
    import warnings as _w
    from scipy.integrate import quad
    from hydro.rankine.thinship import wigley_kochin
    L, B, T = 1.0, 0.1, 0.0625
    es = []
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        for Fn in (0.2, 0.316, 0.5):
            U = Fn * np.sqrt(G0)
            k0 = G0 / U ** 2
            for th in (0.1, 0.5, 1.0, 1.3, 1.5):
                kap = k0 / np.cos(th) ** 2
                a = kap * np.cos(th)
                xr = quad(lambda x: x * np.cos(a * x), -L / 2, L / 2, limit=500, epsabs=0, epsrel=1e-13)[0]
                xi = quad(lambda x: -x * np.sin(a * x), -L / 2, L / 2, limit=500, epsabs=0, epsrel=1e-13)[0]
                z = quad(lambda zz: (1 - zz * zz / T ** 2) * np.exp(kap * zz), -T, 0, limit=500, epsabs=0,
                         epsrel=1e-13)[0]
                hq = 8 * U * B / L ** 2 * (xr + 1j * xi) * z
                hc = wigley_kochin(th, U, G0, L, B, T)
                es.append(float(abs(hc - hq) / abs(hq)))
    m = max(es)
    return [_row("H8a", "[B4 review, new] closed-form Wigley Kochin function vs direct quadrature (15 cases): max rel. diff",
                 m, 0.0, "adaptive quadrature (scipy quad, epsrel 1e-13)", 1e-8, m <= 1e-8)]


def h9_gravity_limit(args):
    """H9 (M4, g -> 0): Wigley at Fn 2, 4, 8 on a body-scale grid (dx = L/60;
    upstream 1.5 L + entry L, downstream 2 L + beach L, lateral 1.5 L) vs the
    phi = 0 image solve (odd rigid-lid image) on the same hull mesh, in the
    vertical-force scale of the double body: e = |Fz/U^2 - Fz0/U^2| /
    |Fz_db/U^2| <= 5 % at Fn 8 (acceptance M4). Expected to FAIL from the
    exploration (B4-E10: the band rows are unstable for k0 L <~ 1.5);
    registered as binding anyway."""
    from hydro.rankine.hull import wigley
    from hydro.rankine.piercing import PiercingNK, structured_gradient
    from hydro.rankine.bie import ImageOperator, neumann_green
    h = wigley()
    L = 1.0
    body = h.mesh(60, 12)
    dev = "cuda" if _gpu() else "cpu"
    out = []
    for par in ((1, -1), (1, 1)):
        op = ImageOperator(body, axes=(1, 2), device=dev, parities=[par])
        ph = neumann_green(op, body.n[:, 0], par, device=dev)
        del op
        gr_ = structured_gradient(body, ph, body.n[:, 0], (60, 12))
        p = gr_[:, 0] - 0.5 * np.einsum("ij,ij->i", gr_, gr_)
        out.append(float(-2 * np.sum(p * body.area * body.n[:, 2])))
    Fz0, Fzdb = out
    es, infos = [], []
    for Fn in (2.0, 4.0, 8.0):
        U = Fn * np.sqrt(G0 * L)
        lam = 2 * np.pi * U * U / G0
        try:
            s = PiercingNK(h, U, G0, n_per_lam=lam / (L / 60), m_body=60, hull_mesh=(60, 12), up=1.5 * L,
                           down=2.0 * L, lat=1.5 * L, mult=(1.0, 1.0, 1.0, L / lam, 0.5 * L / lam))
            a = s.solve()
            infos.append(a["info"])
            F = s.forces()
            es.append(abs(F["Fz"] / U ** 2 - Fz0) / abs(Fzdb))
            s.release()
        except RuntimeError as e:
            es.append(float("inf"))
        _free()
    return [_row("H9", "[B4; expected FAIL from exploration B4-E10] g -> 0: |Fz/U^2 - Fz(phi = 0)/U^2| / |Fz_db/U^2| at "
                 "Fn 2, 4, 8; binding at Fn 8", es, 0.0, "odd rigid-lid image solve (exact g -> 0 limit)", 0.05,
                 es[-1] <= 0.05, _res(infos) + f"; Fz(phi=0)/U^2 = {Fz0:.3e}, Fz_db/U^2 = {Fzdb:.3e}")]


# =====================================================================
# fixed model sweep: validation and verification rows
_CACHE = {}


def _fixed_sweep(fns, cfg):
    key = (tuple(fns), tuple(sorted((k, str(v)) for k, v in cfg.items())))
    if key not in _CACHE:
        _CACHE[key] = [_solve_fixed(Fn, cfg) for Fn in fns]
    return _CACHE[key]


def v_fixed(args):
    """V1 (C_W fixed vs experiment, registered B4; reference invalid, see
    below), V1a/V1b (amended, B4 review), V2 (wave profile vs U. Tokyo), H6
    (near/far), on the given Fn list and grid; H6 only on the extra Fn
    (module docstring)."""
    fns, cfgname = args[0], args[1]
    extra = args[2] if len(args) > 2 else ()
    cfg = PROD if cfgname == "prod" else COARSE
    res = _fixed_sweep(fns, cfg)
    ser = _exp_cw_fixed()
    ut = _yang("fig5a_cw_fixed")
    rows = []
    for o in res:
        Fn = o["Fn"]
        cw = o["F"]["R_near"] / o["q"]
        cp = o["F"]["R_corr"] / o["q"]
        cstrip = (o["F"]["R_corr"] + o["F"]["R_wl"]) / o["q"]
        ccut = o["R_cut"] / o["q"]
        mean, lo, hi, nser = _exp_mean_at(ser, Fn)
        rows.append(_row("V1", "[B4 validation; INVALID REFERENCE (B4 review): the WaveBEM C_W files do not reproduce the "
                         "scan's markers at Fn < 0.3 (transcription defect), so this row is not a model verdict] "
                         f"Fn {Fn}: C_W (fixed model, R_near = R_p - waterline term) vs mean of "
                         f"{nser} digitised series (S.R.I., U.T. x2, UT3; range {lo*1e3:.3f}-{hi*1e3:.3f} e-3)"
                         + (" [value at Fn 0.316 known from exploration]" if Fn == 0.316 else ""),
                         cw, mean, "Kajitani et al. 1983 via Hendrix & Noblesse fig. (WaveBEM files)", 0.20,
                         abs(cw / mean - 1) <= 0.20,
                         f"ratio {cw/mean:.3f}; U.T. curve (Yang et al. 2000 fig 5a) {_yang_at(ut, Fn)*1e3:.3f}e-3; "
                         f"R_corr {o['F']['R_corr']/o['q']*1e3:.3f}e-3, waterline term {o['F']['R_wl']/o['q']*1e3:.3f}e-3; "
                         f"N_F {o['NF']}, n_eff {o['n_eff']:.1f}; {_res([o['info']])}"))
        if ut[0, 0] <= Fn <= ut[-1, 0]:
            ref = _yang_at(ut, Fn)
            tol = _tol(ref, FLOOR_CW)
            rows.append(_row("V1a", f"[B4 review, amended V1, added after a run, computed value known] Fn {Fn}: hull force "
                             "C_p = R_p/q of the linear model (d'Alembert-corrected pressure resistance) vs U. Tokyo C_W "
                             "(C_T-based) curve", cp, ref, "U.T., Kajitani et al. 1983, via Yang et al. 2000 fig 5a (vector)",
                             f"max(20 %, {FLOOR_CW:.2e}) = {tol:.3e}", abs(cp - ref) <= tol,
                             f"ratio {cp/ref:.3f}; R_p + strip (info) {cstrip*1e3:.3f}e-3 (ratio {cstrip/ref:.3f}); "
                             f"R_near {cw*1e3:.3f}e-3 (ratio {cw/ref:.3f})"))
        wp, fwp, ewp = _cwp_at(Fn)
        if wp is not None:
            tol = _tol(wp, FLOOR_CWP)
            rows.append(_row("V1b", f"[B4 review, new, added after a run, computed value known] Fn {Fn}: wave energy flux "
                             f"C_near = R_near/q vs U. Tokyo wave-pattern resistance C_WP (marker at Fn {fwp:.4f}, reading "
                             f"error {ewp*1e3:.3f}e-3)", cw, wp,
                             "U.T. C_WP, Hendrix & Noblesse 1988 p. 61 (McCarthy 1985), read by rankine_wigley_scan.py",
                             f"max(20 %, {FLOOR_CWP:.2e}) = {tol:.3e}", abs(cw - wp) <= tol,
                             f"ratio {cw/wp:.3f}; wave cut C_cut {ccut*1e3:.3f}e-3 (ratio {ccut/wp:.3f})"))
        e = o["F"]["R_near"] / o["R_cut"] - 1
        rows.append(_row("H6", f"[B4] Fn {Fn}: R_near/R_cut - 1 (same solution; cut {CUT_D} lambda0 behind the stern, "
                         f"th_max {np.degrees(o['th_max']):.1f} deg)" + (" [Fn 0.316 known]" if Fn == 0.316 else ""),
                         e, 0.0, "wave-cut energy flux (H7 calibrated)", 0.05, abs(e) <= 0.05,
                         f"C_cut {ccut*1e3:.4f}e-3"))
        if Fn in (0.25, 0.267, 0.289, 0.316, 0.354, 0.408):
            xe_, ze_ = _profile_exp(Fn)
            zc = np.interp(xe_, o["xw"], 2 * G0 * o["zw"] / o["U"] ** 2)
            rms = float(np.sqrt(np.mean((zc - ze_) ** 2)) / np.abs(ze_).max())
            ib = int(np.argmax(ze_ * (xe_ > 0.25)))
            crest_c = float((2 * G0 * o["zw"] / o["U"] ** 2)[o["xw"] > 0.25].max())
            rows.append(_row("V2", f"[B4 validation] Fn {Fn}: wave profile along the hull (fixed), RMS(calc - exp)/max|exp| "
                             f"over {len(xe_)} U.T. points" + (" [Fn 0.316 partly known]" if Fn == 0.316 else ""),
                             rms, 0.0, "U. Tokyo, Kajitani et al. 1983 (WaveBEM tokyoFr*.csv)", 0.20, rms <= 0.20,
                             f"bow crest 2g zeta/U^2 calc {crest_c:.3f} vs exp {ze_[ib]:.3f} at x/L {xe_[ib]:.3f}; grid "
                             "sensitivity at Fn 0.316: row H11e"))
    for Fn in extra:
        o = _solve_fixed(Fn, cfg)
        e = o["F"]["R_near"] / o["R_cut"] - 1
        rows.append(_row("H6", f"[B4 review, new point, added after a run] Fn {Fn}: R_near/R_cut - 1 (validity statement "
                         "up to this Fn)", e, 0.0, "wave-cut energy flux (H7 calibrated)", 0.05, abs(e) <= 0.05,
                         f"C_near {o['F']['R_near']/o['q']*1e3:.4f}e-3, C_cut {o['R_cut']/o['q']*1e3:.4f}e-3; "
                         f"{_res([o['info']])}"))
    return rows


def _order(vals, r=1.5):
    """Successive differences, contraction and observed order from their
    magnitudes (oscillatory convergence allowed: signs reported)."""
    v = np.asarray(vals, float)
    d1, d2 = v[1] - v[0], v[2] - v[1]
    contr = abs(d2) < abs(d1)
    p = float(np.log(abs(d1) / abs(d2)) / np.log(r)) if contr and d2 != 0 else None
    return dict(d=[float(d1), float(d2)], contraction=bool(contr), order=p, monotone=bool(d1 * d2 > 0),
                amplitude=float(v.max() - v.min()))


def _ofmt(o, scale=1.0, unit=""):
    p = "n/a" if o["order"] is None else f"{o['order']:.2f}"
    return (f"diffs {np.round(np.array(o['d']) * scale, 6).tolist()}{unit}, order {p} "
            f"({'monotone' if o['monotone'] else 'oscillatory'}), amplitude {o['amplitude'] * scale:.5f}{unit}")


def h11_convergence(args):
    """H11 (M5): Fn 0.316 grid series m_body = 60, 90, 120 (hull mesh m/5
    rows, n 30/lambda0): R_near, R_cut, Fz, My with observed order and
    Richardson value (ratio 1.5, 1.33: orders reported); information, plus
    the binding change between the two finest of R_near <= 5 % and of Fz
    <= 1 %. B4 review (stage note 9.7): separate free-surface series
    (m_body 60/90/135 at hull 90 x 18) and hull series (hull 60x12/90x18/
    135x27 at m_body 90), ratio 1.5: observed orders of C_near and Fz (info)
    and binding contraction |d2| < |d1| (M5 convergence); the two waterline
    elevation estimators must converge to each other on the joint series
    (binding: |C_wl,fs/C_wl,hull - 1| decreases monotonically); V2 at Fn 0.316 on
    the joint members (info)."""
    joint = [(60, (60, 12)), (90, (90, 18)), (120, (120, 24))]
    fs_ser = [(60, (90, 18)), (90, (90, 18)), (135, (90, 18))]
    hull_ser = [(90, (60, 12)), (90, (90, 18)), (90, (135, 27))]

    def run(series):
        out = []
        for mb, hm in series:
            cfg = dict(n_per_lam=30, m_body=mb, hull_mesh=hm, J_mult=2.0)
            out.append(_solve_fixed(0.316, cfg))
        return out
    oj, of, oh = run(joint), run(fs_ser), run(hull_ser)
    q = oj[0]["q"]
    Rn = [o["F"]["R_near"] / q for o in oj]
    Rc = [o["R_cut"] / q for o in oj]
    Fz = [o["F"]["Fz"] / q for o in oj]
    My = [o["F"]["My"] / q for o in oj]
    dR = abs(Rn[2] / Rn[1] - 1)
    dF = abs(Fz[2] / Fz[1] - 1)
    rows = [_row("H11", "[B4] Fn 0.316, m_body 60/90/120: C_near, C_cut (x1e3), Fz/q, My/(qL)",
                 dict(C_near=[round(x * 1e3, 5) for x in Rn], C_cut=[round(x * 1e3, 5) for x in Rc],
                      Fz=[round(x, 6) for x in Fz], My=[round(x, 7) for x in My]), None, "own", "info", True,
                 f"Richardson (ratio ~1.4) C_near: {_fmt_rich(_richardson(Rn, 1.4))}; Fz: {_fmt_rich(_richardson(Fz, 1.4))}"),
            _row("H11", "[B4] change between the two finest: |dC_near|/C_near and |dFz|/Fz", [dR, dF], 0.0, "own",
                 "5 %; 1 %", dR <= 0.05 and dF <= 0.01, _res([o["info"] for o in oj]))]
    for tag, name, ser in (("H11b", "free-surface series m_body 60/90/135 (hull 90x18)", of),
                           ("H11c", "hull series 60x12/90x18/135x27 (m_body 90)", oh)):
        cn = [o["F"]["R_near"] / q for o in ser]
        fz = [o["F"]["Fz"] / q for o in ser]
        cc = [o["R_cut"] / q for o in ser]
        on, of_, oc = _order(cn), _order(fz), _order(cc)
        rows.append(_row(tag, f"[B4 review, new] Fn 0.316 {name}: C_near, Fz/q, C_cut; observed orders (ratio 1.5)",
                         dict(C_near=[round(x * 1e3, 5) for x in cn], Fz=[round(x, 6) for x in fz],
                              C_cut=[round(x * 1e3, 5) for x in cc]), None, "own", "info", True,
                         f"C_near: {_ofmt(on, 1e3, 'e-3')}; Fz: {_ofmt(of_)}; C_cut: {_ofmt(oc, 1e3, 'e-3')}"))
        rows.append(_row(tag, f"[B4 review, new] {name}: contraction |d2| < |d1| of C_near and of Fz (M5 convergence)",
                         [on["contraction"], of_["contraction"]], True, "own", "both True",
                         on["contraction"] and of_["contraction"], _res([o["info"] for o in ser])))
    d = [abs(o["R_wl_fs"] / o["F"]["R_wl"] - 1) for o in oj]
    rows.append(_row("H11d", "[B4 review, new] waterline-term estimators on the joint series m 60/90/120: "
                     "|C_wl(free-surface zeta)/C_wl(hull-side zeta) - 1| must decrease monotonically",
                     [float(x) for x in d], None, "own (the two must converge to each other)", "monotone decrease",
                     d[0] > d[1] > d[2],
                     f"C_wl hull {[round(o['F']['R_wl'] / q * 1e3, 4) for o in oj]}e-3, fs "
                     f"{[round(o['R_wl_fs'] / q * 1e3, 4) for o in oj]}e-3"))
    xe_, ze_ = _profile_exp(0.316)
    v2 = []
    for o in oj:
        zc = np.interp(xe_, o["xw"], 2 * G0 * o["zw"] / o["U"] ** 2)
        v2.append(float(np.sqrt(np.mean((zc - ze_) ** 2)) / np.abs(ze_).max()))
    rows.append(_row("H11e", "[B4 review, info] V2 measure at Fn 0.316 on the joint series m 60/90/120 (grid sensitivity "
                     "of the V2 verdict)", v2, None, "U. Tokyo profile", "info", True))
    return rows


# =====================================================================
# free model
def _sinkage_reading():
    """Resolution rule of the sinkage definition (registered, stage note
    9.1): double-body (Fn -> 0) free sinkage coefficient c_db = (h/L)/Fn^2
    from the exact rigid-lid Fz, My and the exact hydrostatic stiffness,
    against the U.T. curve's s/Fn^2 averaged over its vertices with
    0.13 <= Fn <= 0.20: reading "mean" (s = h/L, Kajitani's (dF'+dA')/(2L))
    if |log(r/c_db)| < |log(r/(2 c_db))|, else "twice" (s = 2h/L as printed by
    Yang et al.)."""
    from hydro.rankine.hull import wigley
    from hydro.rankine.piercing import structured_gradient
    from hydro.rankine.bie import ImageOperator, neumann_green
    h = wigley()
    body = h.mesh(90, 18)
    dev = "cuda" if _gpu() else "cpu"
    op = ImageOperator(body, axes=(1, 2), device=dev, parities=[(1, 1)])
    ph = neumann_green(op, body.n[:, 0], (1, 1), device=dev)
    del op
    gr_ = structured_gradient(body, ph, body.n[:, 0], (90, 18))
    p = gr_[:, 0] - 0.5 * np.einsum("ij,ij->i", gr_, gr_)              # U = 1, rho = 1
    hs = h.hydrostatics()
    cg = (float(hs["centroid_body"][0]), 0.0)
    cgE = np.array([cg[0], 0.0, cg[1]])
    Fz = float(-2 * np.sum(p * body.area * body.n[:, 2]))
    r = body.c - cgE
    My = float(-2 * np.sum(p * body.area * (r[:, 2] * body.n[:, 0] - r[:, 0] * body.n[:, 2])))
    W = G0 * hs["V"]
    J = np.zeros((2, 2))
    for c in range(2):
        e = np.zeros(2)
        e[c] = 1e-6
        hp = h.hydrostatic_load(*e, 1.0, G0, cg)
        hm = h.hydrostatic_load(*(-e), 1.0, G0, cg)
        J[:, c] = (np.array([hp["Fz"], hp["My"]]) - np.array([hm["Fz"], hm["My"]])) / 2e-6
    # Fn^2 = U^2/(g L): forces scale with U^2 = Fn^2 g
    x = np.linalg.solve(J, -np.array([Fz, My]) * G0)
    c_db = float(x[0])                                       # (h/L)/Fn^2 with L = 1
    a = _yang("fig4a_sinkage_s")
    m = (a[:, 0] >= 0.13) & (a[:, 0] <= 0.20)
    rr = float(np.mean(a[m, 1] / a[m, 0] ** 2))
    reading = "mean" if abs(np.log(rr / c_db)) < abs(np.log(rr / (2 * c_db))) else "twice"
    return dict(c_db=c_db, trim_db=float(x[1]), r_exp=rr, reading=reading, n=int(m.sum()))


def v_free(args):
    """V3 (sinkage), V4 (trim), V5 (C_W free) vs the U.T. curves of Yang et
    al. (2000) fig 4a / 5a; H12 (equilibrium convergence) on every Fn."""
    fns, cfgname = args
    from hydro.rankine.hull import wigley
    from hydro.rankine.piercing import free_equilibrium, cut_resistance
    cfg = PROD if cfgname == "prod" else COARSE
    rd = _sinkage_reading()
    sk, tr, cwf = _yang("fig4a_sinkage_s"), _yang("fig4a_trim_t"), _yang("fig5a_cw_free")
    rows = [_row("V3", "[B4; registered rule, definition resolution] sinkage reading of the U.T. curve: double-body "
                 "coefficient c_db = (h/L)/Fn^2 vs curve s/Fn^2 (0.13 <= Fn <= 0.20)",
                 dict(c_db=rd["c_db"], r_exp=rd["r_exp"], reading=rd["reading"], n=rd["n"]), None,
                 "rigid-lid double body + exact hydrostatics", "info", True,
                 "reading 'mean': s_exp = h/L; 'twice': s_exp = 2 h/L")]
    h = wigley()
    infos = []
    for Fn in fns:
        U = Fn * np.sqrt(G0)
        t = time.time()
        r = free_equilibrium(h, U, G0, RHO, tol=1e-6, max_iter=12, **cfg)
        s = r["solver"]
        F = r["forces"]
        infos.append(s.sol["info"])
        q = _q(U)
        sink_calc = r["sink"] / h.L * (1.0 if rd["reading"] == "mean" else 2.0)
        t_calc = float(np.sin(r["trim"]))
        s_exp, t_exp, cw_exp = _yang_at(sk, Fn), _yang_at(tr, Fn), _yang_at(cwf, Fn)
        cw = F["R_near"] / q
        Rc, _ = cut_resistance(s, s.G["grid"]["x_s"] - CUT_D * s.lam, RHO)
        s.release()
        _free()
        res_last = max(abs(v) for v in r["history"][-1]["res"])
        rows.append(_row("H12", f"[B4] Fn {Fn}: free-equilibrium residual max(|Fz - W|/W, |My|/(W L)) after "
                         f"{len(r['history']) - 1} Newton-Broyden steps", res_last, 0.0, "design P-7", 1e-6,
                         r["converged"], f"{time.time() - t:.0f} s"))
        rows.append(_row("V3", f"[B4 validation] Fn {Fn}: free-model sinkage ({rd['reading']} reading)"
                         + (" [Fn 0.354 value known from exploration B4-E6]" if Fn == 0.354 else ""),
                         sink_calc, s_exp, "U.T., Kajitani et al. 1983, via Yang et al. 2000 fig 4a (vector curve)", 0.15,
                         abs(sink_calc / s_exp - 1) <= 0.15, f"ratio {sink_calc/s_exp:.3f}; sinkage/L {r['sink']:.5f}"))
        tol_t = np.radians(0.05)
        # known model omissions next to V4 (B4 review, stage note 9.7): friction acting about T/2 below the CG,
        # M_f ~ C_F q S T/2 -> d(theta) = M_f / (rho g I_L-type stiffness); unknown tow point and KG
        rows.append(_row("V4", f"[B4 validation] Fn {Fn}: free-model trim t = (dA - dF)/L = sin(theta) (> 0 bow up)",
                         t_calc, t_exp, "U.T. via Yang et al. 2000 fig 4a", "+-0.05 deg (8.7e-4)",
                         abs(t_calc - t_exp) <= tol_t, f"difference {np.degrees(t_calc - t_exp):+.3f} deg; known "
                         "omissions (not modelled): friction moment about T/2 below the CG ~0.014 deg at Fn 0.316 "
                         "(C_F ~ 3.5e-3), tow point and KG ~0.02 deg"))
        rows.append(_row("V5", f"[B4 validation; definition mismatch found in the B4 review: R_near is the wave energy "
                         f"flux, the U.T. C_W is C_T-based (amended row V5a)] Fn {Fn}: C_W of the free model (R_near at "
                         "the equilibrium attitude)",
                         cw, cw_exp, "U.T. via Yang et al. 2000 fig 5a (free)", 0.20, abs(cw / cw_exp - 1) <= 0.20,
                         f"ratio {cw/cw_exp:.3f}; C_cut {Rc/q*1e3:.3f}e-3; near/cut - 1 {F['R_near']/Rc - 1:+.3f}"))
        cpf = F["R_corr"] / q
        tol5 = _tol(cw_exp, FLOOR_CW)
        rows.append(_row("V5a", f"[B4 review, amended V5, added after a run] Fn {Fn}: hull force C_p = R_p/q of the free "
                         "model at its equilibrium attitude vs U.T. C_W (free, C_T-based) curve", cpf, cw_exp,
                         "U.T. via Yang et al. 2000 fig 5a (free)", f"max(20 %, {FLOOR_CW:.2e}) = {tol5:.3e}",
                         abs(cpf - cw_exp) <= tol5,
                         f"ratio {cpf/cw_exp:.3f}; R_p + strip (info) {(F['R_corr'] + F['R_wl'])/q*1e3:.3f}e-3"))
    rows.append(_row("H12", "[B4] residuals of the final solves of the free sweep", _res(infos), None, "", "info", True))
    return rows


# =====================================================================
# transom
def _eta_of(h, U, spec):
    """eta spec: 'D' Doctors' regression, 'lo' its registered lower scatter
    bound (Fn_T x 0.8), or a number."""
    if spec == "D":
        return doctors_eta(h, U)
    if spec == "lo":
        return doctors_eta(h, U, 0.8)
    return float(spec)


def _transom_solve(h, Fn, cfg, branch, eta, sink=0.0, trim=0.0, cg=(0.0, 0.0)):
    """One transom solve at a fixed attitude: hull force C_total, wave cut,
    the momentum identity of stage note 9.7 (PiercingNK.transom_identity),
    the D1 edge residual, the D2 row residual."""
    from hydro.rankine.piercing import PiercingNK, cut_resistance, total_load
    U = Fn * np.sqrt(G0)
    q = _q(U)
    t = time.time()
    s = PiercingNK(h, U, G0, sink=sink, trim=trim, rho=RHO, branch=branch, eta=eta, **cfg)
    a = s.solve()
    F = s.forces(cg)
    tl = total_load(s, cg, F)
    Rc, _ = cut_resistance(s, s.G["grid"]["x_s"] - CUT_D * s.lam, RHO)
    ti = s.transom_identity(F)
    g = s.G["grid"]
    idx = s.G["index"]
    sig = a["sigma"]
    d2 = []
    for j in range(-g["K"], 0):
        k0, k1 = idx[(g["i_stern"], j)], idx[(g["i_stern"] + 1, j)]
        d2.append(abs((1.5 * (sig[k0] - s.f_patch[k0]) - 0.5 * (sig[k1] - s.f_patch[k1])) - s.rhs_extra[k0]))
    out = dict(Fn=Fn, branch=branch, eta=eta, q=q, C_total=(F["R_raw"] - F["R_wl"] - tl["Fx_static_face"]) / q,
               C_cut=Rc / q, id=ti, C_id=ti["R_id"] / q, d2=max(d2), info=a["info"], K=g["K"], NF=s.NF, ND=s.ND,
               Fz=tl["Fz"] / q, seconds=time.time() - t)
    s.release()
    del s
    _free()
    return out


def t_transom(args):
    """Transom rows (stage note 9.1; B4 review 9.7). Wigley-transom test hull
    (hull.wigley_transom, x_t = -0.4 L, T_t = 0.25 T; no measured data exist):
    DRY and WET (WaveBEM-style patch, eta from Doctors' regression as quoted in
    WaveBEM, l_h = 5) branches. args = dict(fixed=[(Fn, grid, [(branch, eta
    spec)])], t1=bool, free=[(Fn, branch, eta spec)]). Rows: T1 (identity, small
    grid), T2 (D2 rows), T3 (momentum identity, binding 5 %), T6 (D1 edge
    residual: binding at PROD), T4 (branch table, info), T7 (COARSE/PROD pair:
    binding D1 decrease, info values), T5 (free equilibrium)."""
    from hydro.rankine.hull import wigley_transom
    from hydro.rankine.piercing import PiercingNK, free_equilibrium
    h = wigley_transom()
    rows = []
    if args.get("t1"):
        # T1 on a small grid: wet branch at eta = 1 reproduces the dry branch (construction check)
        U = 0.45 * np.sqrt(G0)
        kw = dict(n_per_lam=12, m_body=30, hull_mesh=(30, 6), face_mesh=(4, 4), mult=(0.5, 0.4, 0.5, 1, 1))
        R = []
        for br in ("dry", "wet"):
            s = PiercingNK(h, U, G0, rho=RHO, branch=br, eta=1.0, **kw)
            s.solve()
            F = s.forces()
            R.append(F["R_raw"] - F["R_wl"])
            s.release()
            _free()
        d = abs(R[0] - R[1]) / abs(R[0])
        rows.append(_row("T1", "[B4; moved to a small grid in the B4 review (identity by construction)] Fn 0.45: wet "
                         "branch at eta = 1 reproduces the dry branch: |dR|/R", d, 0.0, "construction", 1e-12, d <= 1e-12))
    res = {}
    d2max = 0.0
    for Fn, grid, cases in args.get("fixed", []):
        cfg = dict(COARSE if grid == "coarse" else PROD)
        cfg["face_mesh"] = (6, 8)
        U = Fn * np.sqrt(G0)
        for br, es in cases:
            eta = 1.0 if br == "dry" else _eta_of(h, U, es)
            o = _transom_solve(h, Fn, cfg, br, eta)
            res[(Fn, grid, br, es)] = o
            d2max = max(d2max, o["d2"])
            ti = o["id"]
            q = o["q"]
            e = o["C_id"] / o["C_cut"] - 1
            name = f"{br}" + ("" if br == "dry" else f"(eta {eta:.3f}{', lower scatter bound' if es == 'lo' else ''})")
            rows.append(_row("T3", f"[B4 review, binding identity, derived in 9.7] Fn {Fn} {grid.upper()} {name}: "
                             "R_id/R_cut - 1, R_id = R_dynH - W_hull + E_edge + K_face + P_patch", e, 0.0,
                             "wave cut (momentum identity of the linear model with the face closure)", 0.05,
                             abs(e) <= 0.05,
                             f"C x1e3: dynH {ti['R_dynH']/q*1e3:.4f}, W {ti['W_hull']/q*1e3:.4f}, E {ti['E_edge']/q*1e3:.4f} "
                             f"(zeta_e; with z_T {ti['E_edge_zT']/q*1e3:.4f}), K {ti['K_face']/q*1e3:.4f}, P "
                             f"{ti['P_patch']/q*1e3:.4f}; C_id {o['C_id']*1e3:.4f}, C_cut {o['C_cut']*1e3:.4f}; with "
                             f"zeta_e = z_T: {ti['R_id_zT']/(o['C_cut']*q) - 1:+.4f}; hull force C_total "
                             f"{o['C_total']*1e3:.4f} (C_total/C_cut - 1 {o['C_total']/o['C_cut'] - 1:+.4f}); K lines "
                             f"{o['K']}; {_res([o['info']])}"))
            rows.append(_row("T6", f"[B4 review, new; COARSE and PROD values at Fn 0.35 known from exploration] Fn {Fn} "
                             f"{grid.upper()} {name}: D1 edge residual rms(zeta_e - z_T)/rms(z_T), zeta_e from phi of "
                             "the first three points of each transom line (independent of the D1 row)",
                             ti["D1_res"], 0.0, "D1: zeta = z_T at the edge",
                             D1_TOL if grid == "prod" else "info", (ti["D1_res"] <= D1_TOL) if grid == "prod" else True,
                             f"zeta_e {np.round(ti['zeta_e'], 5).tolist()}, z_T {np.round(ti['z_T'], 5).tolist()}"))
    if res:
        rows.append(_row("T2", "[B4; implementation check (re-evaluates the imposed rows)] discrete D2 rows (smooth "
                         "detachment), max abs residual over all fixed transom solves", d2max, 0.0, "M2 identity", 1e-10,
                         d2max <= 1e-10))
        for Fn in sorted(set(k[0] for k in res)):
            U = Fn * np.sqrt(G0)
            etaD = doctors_eta(h, U)
            for grid in ("coarse", "prod"):
                sel = {k: v for k, v in res.items() if k[0] == Fn and k[1] == grid}
                if not sel:
                    continue
                tab = "; ".join(f"{k[2]}{'' if k[2] == 'dry' else '(' + format(v['eta'], '.3f') + ')'} "
                                f"{v['C_total']*1e3:.4f}" for k, v in sel.items())
                rows.append(_row("T4", f"[B4] Fn {Fn} {grid.upper()}: branch table, hull force C_total x1e3 (never "
                                 f"weighted); Doctors eta_D = {etaD:.3f}" + (" (wet(eta_D) identical to dry, not solved)"
                                                                          if etaD >= 1 else ""), tab, None,
                                 "Doctors' regression as quoted in WaveBEM (EMPIRICAL_REGISTER_RANKINE.md)", "info", True,
                                 f"Fn_T = {U/np.sqrt(G0*h.T_t):.2f}; eta at the registered scatter bounds "
                                 f"{doctors_eta(h, U, 0.8):.3f} / {doctors_eta(h, U, 1.2):.3f}; C_cut: "
                                 + "; ".join(f"{k[2]} {v['C_cut']*1e3:.4f}" for k, v in sel.items())))
        for (Fn, grid, br, es), op in res.items():
            if grid != "prod" or (Fn, "coarse", br, es) not in res:
                continue
            oc = res[(Fn, "coarse", br, es)]
            rows.append(_row("T7", f"[B4 review, new] Fn {Fn} {br}: COARSE -> PROD pair: D1 edge residual must decrease; "
                             "C_total, C_cut, C_id, edge zeta (info)", [oc["id"]["D1_res"], op["id"]["D1_res"]], None,
                             "own (grid pair)", "decrease", op["id"]["D1_res"] < oc["id"]["D1_res"],
                             f"C_total {oc['C_total']*1e3:.4f} -> {op['C_total']*1e3:.4f}; C_cut {oc['C_cut']*1e3:.4f} -> "
                             f"{op['C_cut']*1e3:.4f}; C_id {oc['C_id']*1e3:.4f} -> {op['C_id']*1e3:.4f}; T3 "
                             f"{oc['C_id']/oc['C_cut'] - 1:+.4f} -> {op['C_id']/op['C_cut'] - 1:+.4f}; zeta_e "
                             f"{np.round(oc['id']['zeta_e'], 5).tolist()} -> {np.round(op['id']['zeta_e'], 5).tolist()}"))
    if args.get("dalembert"):
        # T8 (info): d'Alembert defect of the hull + face mesh in the double-body flow (face as a wall; the total
        # x-force on the closed doubled body is exactly 0): a measure of the pressure-integration error at the
        # transom corner (exploration B4R-E5)
        from hydro.rankine.bie import ImageOperator, neumann_green
        from hydro.rankine.piercing import structured_gradient
        U = 0.45 * np.sqrt(G0)
        q = _q(U)
        out = []
        for (nbx, nbz) in ((60, 12), (90, 18)):
            for (sk, tr) in ((0.0, 0.0), (0.0066, 0.0255)):
                body = h.mesh(nbx, nbz, sk, tr) + h.face_mesh(6, 8, sk, tr)
                fm = body.tag == 2
                op = ImageOperator(body, axes=(1, 2), device="cuda" if _gpu() else "cpu", parities=[(1, 1)])
                pn = U * body.n[:, 0]
                ph = neumann_green(op, pn, (1, 1), device="cuda" if _gpu() else "cpu")
                del op
                fx = []
                for m, shp in ((~fm, (nbx, nbz)), (fm, (6, 8))):
                    pan = body.subset(m)
                    gr_ = structured_gradient(pan, ph[m], pn[m], shp)
                    pp = U * gr_[:, 0] - 0.5 * np.einsum("ij,ij->i", gr_, gr_)
                    fx.append(float(-2 * RHO * np.sum(pp * pan.area * pan.n[:, 0])))
                out.append((f"hull {nbx}x{nbz} (s, theta) = ({sk}, {tr})", round(-(fx[0] + fx[1]) / q * 1e3, 4),
                            round(-fx[0] / q * 1e3, 4)))
                _free()
        rows.append(_row("T8", "[B4 review, new info; values known from exploration B4R-E5] d'Alembert defect of the transom "
                         "hull + face mesh (face 6x8) in the double-body flow at Fn 0.45 scale: total x-force / q x1e3 "
                         "(exactly 0 analytically); hull-only part", [list(o) for o in out], 0.0,
                         "d'Alembert (closed doubled body)", "info", True))
    for Fn, br, es in args.get("free", []):
        U = Fn * np.sqrt(G0)
        eta = 1.0 if br == "dry" else _eta_of(h, U, es)
        cfg = dict(COARSE)
        cfg["face_mesh"] = (6, 8)
        t = time.time()
        r = free_equilibrium(h, U, G0, RHO, tol=1e-6, max_iter=12, branch=br, eta=eta, **cfg)
        q = _q(U)
        F = r["forces"]
        tl = r["load"]
        s = r["solver"]
        Rtot = (F["R_raw"] - F["R_wl"] - tl["Fx_static_face"]) / q
        ti = s.transom_identity(F)
        from hydro.rankine.piercing import cut_resistance
        Rc, _ = cut_resistance(s, s.G["grid"]["x_s"] - CUT_D * s.lam, RHO)
        res_last = max(abs(v) for v in r["history"][-1]["res"])
        newrow = not (Fn == 0.45 and br == "dry")
        rows.append(_row("T5", (f"[B4 review, new: wet patch and (1 - eta) face force in the equilibrium] " if newrow else
                                "[B4] ") + f"Fn {Fn} {br} (eta {eta:.3f}): free equilibrium residual; sinkage/L, trim "
                         "(rad), C_total", [res_last, r["sink"], r["trim"], Rtot], None, "design P-7", 1e-6,
                         r["converged"], f"{len(r['history']) - 1} Newton-Broyden steps, {time.time() - t:.0f} s; "
                         f"T3 identity at the equilibrium (info): {ti['R_id']/Rc - 1:+.4f}; "
                         f"{_res([s.sol['info']])}"))
        s.release()
        _free()
    return rows


def h13_flare(args):
    """H13 (B4 review, new; G4 check before any non-Wigley use): Wigley hull
    with flare at the waterline (hull.wigley_flare, f = 0.3: H_z = 0.24 at
    midship, 13.5 deg) at Fn 0.316 on the production grid with the production
    hull-wall ghost (H_x, H_z, H_zz, H_xz terms; the H_z terms of the sig
    ghost are omitted, stage note 9.7): near/far |R_near/R_cut - 1| <= 5 %
    (binding); the B4 ghost without the H_z/H_zz terms ("wall_vertical") on
    the same grid: change of C_near and of the bow crest (info)."""
    from hydro.rankine.hull import wigley_flare
    h = wigley_flare()
    o = _solve_fixed(0.316, PROD, hull=h)
    ov = _solve_fixed(0.316, PROD, hull=h, ghost_mode="wall_vertical")
    e = o["F"]["R_near"] / o["R_cut"] - 1
    q = o["q"]
    cr = float((2 * G0 * o["zw"] / o["U"] ** 2)[o["xw"] > 0.25].max())
    crv = float((2 * G0 * ov["zw"] / ov["U"] ** 2)[ov["xw"] > 0.25].max())
    return [_row("H13", "[B4 review, new] flared Wigley (f 0.3) Fn 0.316 PROD: R_near/R_cut - 1 with the production ghost",
                 e, 0.0, "wave-cut energy flux", 0.05, abs(e) <= 0.05,
                 f"C_near {o['F']['R_near']/q*1e3:.4f}e-3, C_cut {o['R_cut']/q*1e3:.4f}e-3; {_res([o['info']])}"),
            _row("H13", "[B4 review, info] same with the B4 ghost (no H_z, H_zz terms): C_near, near/cut - 1, bow crest "
                 "2g zeta/U^2 (production vs B4 ghost)",
                 [ov["F"]["R_near"] / q * 1e3, ov["F"]["R_near"] / ov["R_cut"] - 1, crv, cr], None, "own", "info", True,
                 f"production ghost: C_near {o['F']['R_near']/q*1e3:.4f}e-3, crest {cr:.4f}")]


def doctors_eta(hull, U, fn_scale=1.0, L_model=2.5, nu=1.307e-6, g=G0):
    """Dry fraction eta = min(1, 0.05 Fn_T^2.834 (B_T/T_T)^0.1352
    Re_T^0.01338) (WaveBEM free_surface.cc, quoted as Doctors' regression;
    EMPIRICAL_REGISTER_RANKINE.md B4-1), at model scale L_model (the Re_T
    term needs a dimensional draught; Kajitani-size model, 2.5 m), with
    Fn_T = U/sqrt(g T_T) scaled by fn_scale (the registered scatter is a
    shift of the ventilation curve by +-20 % in Fn_T)."""
    Tt = hull.T_t
    BT = 2 * float(hull.waterline(np.array([hull.x_aft]))[0])
    FnT = fn_scale * U / np.sqrt(g * Tt)
    ReT = np.sqrt(g * (Tt * L_model) ** 3) / nu
    return float(min(1.0, 0.05 * FnT ** 2.834 * (BT / Tt) ** 0.1352 * ReT ** 0.01338))


# =====================================================================
T_QUICK = dict(fixed=[(0.45, "coarse", [("dry", None), ("wet", 0.0)])], t1=True, free=[])
T_FULL = dict(fixed=[(0.35, "coarse", [("dry", None), ("wet", "lo"), ("wet", 0.0)]),
                     (0.45, "coarse", [("dry", None), ("wet", 0.0)]),
                     (0.35, "prod", [("dry", None), ("wet", 0.0)])],
              t1=True, dalembert=True, free=[(0.45, "dry", None), (0.35, "wet", 0.0), (0.35, "wet", "lo")])

TIERS = {
    "quick": [
        ("H0", h0_hydrostatics, (), "gpu"),
        ("H1", h1_fft_dense, (), "gpu"),
        ("H2", h2_lattice_rows, (), "gpu"),
        ("H5", h5_similarity, (), "gpu"),
        ("H7", h7_cut_calibration, (), "gpu"),
        ("H8a", h8a_michell_closed_form, (), "cpu"),
        ("V", v_fixed, ((0.316,), "coarse"), "gpu"),
        ("VF", v_free, ((0.316,), "coarse"), "gpu"),
        ("T", t_transom, T_QUICK, "gpu"),
    ],
    "full": [
        ("H0", h0_hydrostatics, (), "gpu"),
        ("H1", h1_fft_dense, (), "gpu"),
        ("H2", h2_lattice_rows, (), "gpu"),
        ("H4", h4_split, (0.25,), "gpu"),
        ("H5", h5_similarity, (), "gpu"),
        ("H7", h7_cut_calibration, (), "gpu"),
        ("H8a", h8a_michell_closed_form, (), "cpu"),
        ("V", v_fixed, (FN_FIXED, "prod", FN_H6_EXTRA), "gpu"),
        ("H11", h11_convergence, (), "gpu"),
        ("H8", h8_thin_ship, (), "gpu"),
        ("H13", h13_flare, (), "gpu"),
        ("H9", h9_gravity_limit, (), "gpu"),
        ("VF", v_free, (FN_FREE, "coarse"), "gpu"),
        ("T", t_transom, T_FULL, "gpu"),
    ],
}

STAGE_NOTE = os.path.join(ROOT, "learn", "meta", "RANKINE_STAGE1_2026-10-06.md")
REG_LOG = os.path.join(OUT, "registration_log.json")
REG_SECTIONS = ("### 9.1 ", "### 9.7 ")
CODE_FILES = ("hydro/rankine/*.py", "studies/rankine_hull_verify.py", "studies/rankine_wigley_scan.py",
              "studies/rankine_verify.py")


def _code_hashes():
    """sha256 of the solver and check code (acceptance N2; added in the B4
    review: the B4 registration hashed the thresholds and jobs but not the
    code that produced the tier JSON)."""
    import glob
    out = {}
    for pat in CODE_FILES:
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            rel = os.path.relpath(f, ROOT).replace("\\", "/")
            with open(f, "rb") as fh:
                out[rel] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _note_sections():
    lines = open(STAGE_NOTE, encoding="utf-8").read().replace("\r\n", "\n").split("\n")
    out = {}
    for head in REG_SECTIONS:
        hits = [k for k, ln in enumerate(lines) if ln.startswith(head)]
        if not hits:
            out["b4:" + head.strip()] = dict(title=None, sha256=None)
            continue
        i = hits[0]
        j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ") or lines[k].startswith("### ")),
                 len(lines))
        sec = "\n".join(lines[i:j]).strip() + "\n"
        out["b4:" + head.strip()] = dict(title=lines[i].strip(), sha256=hashlib.sha256(sec.encode("utf-8")).hexdigest())
    return out


def _code_thresholds():
    import ast
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    tree = ast.parse(src)
    items = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_row":
            a = node.args
            items.append(" | ".join([ast.unparse(a[0]) if a else "", "tol " + (ast.unparse(a[5]) if len(a) > 5 else ""),
                                     "pass if " + (ast.unparse(a[6]) if len(a) > 6 else "")]))
    items.sort()
    consts = f"PROD={PROD!r} COARSE={COARSE!r} FN_FIXED={FN_FIXED!r} FN_FREE={FN_FREE!r} CUT_D={CUT_D!r}"
    jobs = [f"{t} {n} {fn.__name__} {a!r}" for t in ("quick", "full") for n, fn, a, _ in TIERS[t]]
    h1 = hashlib.sha256(("\n".join(items) + "\n" + consts).encode("utf-8")).hexdigest()
    h2 = hashlib.sha256("\n".join(jobs).encode("utf-8")).hexdigest()
    return h1, h2, items


def _registration_state():
    secs = _note_sections()
    h_thr, h_jobs, _ = _code_thresholds()
    try:
        log = json.load(open(REG_LOG, encoding="utf-8"))
    except (OSError, ValueError):
        log = []
    cur = {k: v["sha256"] for k, v in secs.items()}
    cur["b4:code_thresholds"] = h_thr
    cur["b4:tier_jobs"] = h_jobs
    reg = {}
    for k, h in cur.items():
        hit = [e["utc"] for e in log if h is not None and e["hashes"].get(k) == h]
        reg[k] = dict(sha256=h, registered_utc=hit[-1] if hit else None)
    return reg


def register(note):
    secs = _note_sections()
    h_thr, h_jobs, items = _code_thresholds()
    try:
        log = json.load(open(REG_LOG, encoding="utf-8"))
    except (OSError, ValueError):
        log = []
    hashes = {k: v["sha256"] for k, v in secs.items()}
    hashes.update({"b4:code_thresholds": h_thr, "b4:tier_jobs": h_jobs})
    entry = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), note=note,
                 titles={k: v["title"] for k, v in secs.items()}, hashes=hashes, threshold_table=items,
                 code_sha256=_code_hashes())
    log.append(entry)
    os.makedirs(OUT, exist_ok=True)
    with open(REG_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    print(f"registered at {entry['utc']}: " + ", ".join(f"{k} {v[:12]}" for k, v in hashes.items()))


def run(tier, only=None):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    reg = _registration_state()
    print(f"=== rankine_hull_verify tier={tier} start {start_utc}; registration record:", flush=True)
    for k, v in reg.items():
        print(f"    {k}: sha256 {v['sha256'][:16] if v['sha256'] else None} registered "
              f"{v['registered_utc'] or 'NOT IN registration_log.json'}", flush=True)
    code0 = _code_hashes()
    rows = []
    for name, fn, a, _ in TIERS[tier]:
        if only and name not in only:
            continue
        t = time.time()
        try:
            rr = fn(a)
        except Exception as e:      # noqa: BLE001
            import traceback
            traceback.print_exc()
            rr = [_row(name, "EXCEPTION", repr(e), None, "", "", False)]
        for r in rr:
            r["seconds"] = round(time.time() - t, 1)
        rows += rr
        print(f"  [{time.time() - t0:6.0f} s] {name} done ({time.time() - t:.0f} s)", flush=True)
    wall = time.time() - t0
    path = os.path.join(OUT, f"hull_verify_{tier}{'_' + '_'.join(only) if only else ''}.json")
    info_rows = [r for r in rows if isinstance(r["tol"], str) and r["tol"].startswith("info")]
    bind_rows = [r for r in rows if not (isinstance(r["tol"], str) and r["tol"].startswith("info"))]
    summary = dict(binding_pass=sum(r["ok"] for r in bind_rows), binding=len(bind_rows), info=len(info_rows),
                   failing=[f"{r['check']}: {r['quantity'][:100]}" for r in bind_rows if not r["ok"]])
    code1 = _code_hashes()
    changed = sorted(k for k in set(code0) | set(code1) if code0.get(k) != code1.get(k))
    try:
        log = json.load(open(REG_LOG, encoding="utf-8"))
    except (OSError, ValueError):
        log = []
    code_reg = [e["utc"] for e in log if e.get("code_sha256") == code0]
    with open(path, "w") as f:
        json.dump(dict(tier=tier, start_utc=start_utc, wall_s=wall, registration=reg, summary=summary,
                       code_sha256_start=code0, code_changed_during_run=changed,
                       code_registered_utc=code_reg[-1] if code_reg else None, rows=rows),
                  f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(f"=== code sha256 at start registered: {code_reg[-1] if code_reg else 'NO'}; files changed during the run: "
          f"{changed or 'none'}")
    print(f"\n=== rankine_hull_verify tier={tier}  wall {wall:.0f} s  -> {path}")
    for r in rows:
        v = r["value"]
        if isinstance(v, float):
            v = f"{v:.4g}"
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['check']:4s} {r['quantity']}\n"
              f"        value={v}  ref={r['ref']}  tol={r['tol']}  ({r['source']})"
              f"{'  ' + r['note'] if r['note'] else ''}  [{r.get('seconds', '')} s]")
    print(f"=== binding rows: {summary['binding_pass']}/{summary['binding']} passed; information rows: {summary['info']}")
    for fl in summary["failing"]:
        print(f"    FAIL {fl}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="quick", choices=["quick", "full"])
    ap.add_argument("--only", default="")
    ap.add_argument("--register", default=None)
    a = ap.parse_args()
    if a.register is not None:
        register(a.register)
        sys.exit(0)
    run(a.tier, [s for s in a.only.split(",") if s] or None)
