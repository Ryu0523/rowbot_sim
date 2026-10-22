#!/usr/bin/env python3
"""
Track B step B5: forward-speed radiation and diffraction as one linear
time-invariant system (hydro/rankine/unsteady.py) and the radiation-table
interface (hydro/rankine/tables.py) -- verification and validation tiers
(acceptance learn/meta/ACCEPTANCE_2026-10-06.md, T1: quick <= 10 min, full <= 1 h).
Registration: stage note learn/meta/RANKINE_STAGE1_2026-10-06.md section 10.1
(written, and hashed with --register, BEFORE the tiers ran); derivations 10.2;
exploration runs 10.3; results 10.4. B5 review: changed and new rows registered
in section 10.8 before the re-run (results 10.9).

    .venv/Scripts/python.exe studies/rankine_unsteady_verify.py --tier quick [--only U0,KK]
    .venv/Scripts/python.exe studies/rankine_unsteady_verify.py --register "note"

Row families: U0* zero-speed limit (M4), GI discrete Green identity of the BIE
representation (stage-1 type; B5 called these Haskind/Timman-Newman "(a)"), HK
Haskind and TN/TNp/TNw Timman-Newman with the waterline reduction (M3), TNs
classical Timman-Newman for a submerged body (M3), WE submerged sphere at
forward speed vs Wu & Eatock Taylor 1988 (independent forward-speed reference),
KKu/KK Kramers-Kronig (evaluation unit test; M3), INF/G0 high-frequency and
zero-gravity limits (M4), SP spectrum of the LTI matrix (M3/E2), DXM
manufactured-solution test of the discrete d/dx (M2), TD/CU time domain =
frequency domain and Cummins reconstruction through the documented plant law
(M2), EN energy balance (M3/E2), FS Froude similarity (M3), GS/GSo/DOM grid and
domain convergence (M5), ST0/C0/C0B steady limit and static stiffness, V*
validation against Journee (1992) for both force forms (V*o: Ogilvie-Tuck;
secondary transcriptions, data/external/wigley_journee/), STF strip-theory
consistency (information), CD free decay of the plant law, PV/PVm demo table
(interface). Binding rows carry a registered tolerance; information rows ("info") are counted separately.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from studies.rankine_verify import _row     # noqa: E402

OUT = os.path.join(ROOT, "studies", "_cache", "rankine")
DATA = os.path.join(ROOT, "data", "external", "wigley_journee")
REG_LOG = os.path.join(OUT, "registration_log.json")
STAGE_NOTE = os.path.join(ROOT, "learn", "meta", "RANKINE_STAGE1_2026-10-06.md")
REG_SECTIONS = ("### 10.1", "### 10.8")
CODE_FILES = ("hydro/rankine/*.py", "studies/rankine_unsteady_verify.py", "studies/rankine_journee_data.py")
G0 = 9.81
RHO = 1.0
L = 1.0
SC = np.sqrt(G0 / L)

# ----------------------------------------------------------- registered settings (stage note 10.1)
WJ = dict(L=1.0, B=0.1, T=0.0625)                 # Journee's Wigley I, L/B 10, L/T 16 (3 m model scaled to L = 1)
KG_M, L_M = 0.170, 3.0                             # assumed KG of the 3 m model (not in the available sources)
ZG = (KG_M - 0.1875) / L_M                         # z_G / L relative to the waterline (= -0.00583)
KYY = 0.25                                         # k_yy / L (J. Mar. Sci. Eng. 9:87, Table 2)
W_J = np.array([3, 4, 5, 6, 7, 8, 10]) * np.sqrt(L_M / G0)   # Journee's test frequencies 3..10 rad/s -> omega'
GRID_BASE = dict(up=1.5, down=4.0, lat=3.0, r_up=1.12, r_dn=1.06, r_lat=1.08, b_up=0.75, b_dn=2.0, b_lat=1.5,
                 u_dn=0.5, u_lat=0.3, nu_max=1.5)
PROD = dict(M=60, hull=(60, 12))
COARSE = dict(M=40, hull=(60, 12))
FINE = dict(M=80, hull=(60, 12))
HULL_SERIES = ((40, 8), (60, 12), (90, 18))
CHECK_GRID = dict(M=40, up=3.0, down=3.0, lat=3.0, r_up=1.06, r_dn=1.06, r_lat=1.06, b_up=1.5, b_dn=1.5, b_lat=1.5,
                  u_dn=0.0, u_lat=0.3, nu_max=1.5)       # symmetric domain: identical panels for +U and -U
SMALL_GRID = dict(M=20, up=1.5, down=3.0, lat=2.5, r_up=1.15, r_dn=1.08, r_lat=1.10, b_up=0.75, b_dn=1.5, b_lat=1.2,
                  nu_max=1.5)
SMALL_HULL = (20, 5)
TOL_ID = 0.03          # Haskind / Timman-Newman identities (acceptance M3: Haskind 3 %; design TN proposal 3 %)
TOL_KK = 0.02          # design proposal
TOL_U0 = 0.05          # acceptance M4
TOL_HULME = 0.03       # design R2 (Hulme)
TOL_G0 = 0.05          # acceptance M4
TOL_INF = 0.01         # design R2 (A -> A_inf)
TOL_TD = 0.01
TOL_EN = 0.01
TOL_FS = 1e-4
TOL_GS = 0.02          # grid series: change between the two finest members
TOL_DOM = 0.02         # domain / beach changes
TOL_V_REL, V_FLOOR = 0.20, 0.10   # validation: max(20 % of the value, 10 % of max over the series)


def _free():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _wj():
    from hydro.rankine.hull import wigley_journee
    return wigley_journee(**WJ)


_VOL = {}


def _V(h):
    if h.name not in _VOL:
        _VOL[h.name] = h.hydrostatics()["V"]
    return _VOL[h.name]


def _solver(hull, U, grid, hm, sink=0.0, trim=0.0, sgn=None, g=G0, ref=None, wall="trace"):
    """LTI system with the production wall closure (wall="trace", stage note 10.8.2); wall="rule" = the B5
    rule (comparison rows only)."""
    from hydro.rankine.unsteady import build_grid, dx_operator, wall_trace_data, ForwardSpeedLTI, Coefficients
    sgn = (1.0 if U >= 0 else -1.0) if sgn is None else sgn
    body = hull.mesh(*hm, sink=sink, trim=trim)
    G = build_grid(hull, sink, trim, sgn, **grid)
    D, info = dx_operator(G, sgn, wall=wall, hull=hull, sink=sink, trim=trim)
    wt = wall_trace_data(info["wall"], hull, body, hm) if wall == "trace" else None
    s = ForwardSpeedLTI(body, G["fs"], D, G["nu"] * np.sqrt(g / hull.L), U, g, RHO, hull_shape=hm, wall=wt)
    ref = (0.0, ZG) if ref is None else ref
    return s, Coefficients(s, ref=ref), G


def _nd(T, w, V, Lr=1.0, g=G0):
    """Non-dimensional A', B' of a (3 x 3) T for modes (1, 3, 5) (Journee's
    normalisation: A/(rho V L^p), B/(rho V L^p) sqrt(L/g), p = 0, 1, 2 for
    translation-translation, translation-rotation, rotation-rotation)."""
    A, B = -T.real / w ** 2, T.imag / w
    p = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 2]])
    sc = RHO * V * Lr ** p
    return A / sc, B / sc * np.sqrt(Lr / g)


# =====================================================================
# zero-speed limit (M4)
def u0_wigley(args):
    """U0 (M4, binding 5 %): the forward-speed system at U = 0 vs the stage-2
    zero-speed solver (radiation.ZeroSpeedFS, Green formulation with phi on the
    free-surface panels) on the SAME body and free-surface panels and beach.
    Measure per coefficient: |new - stage2| / max(|stage2|, 0.1 max_omega |stage2|)."""
    wps, grid = args
    from hydro.rankine.radiation import ZeroSpeedFS, radiation_coefficients
    h = _wj()
    V = _V(h)
    g = dict(grid, down=grid["up"], b_dn=grid["b_up"], u_dn=0.0)
    s, co, G = _solver(h, 0.0, g, COARSE["hull"])
    zs = ZeroSpeedFS(s.body, s.fs, axes=(1,))
    res = {}
    for wp in wps:
        w = wp * SC
        r = co.radiation(w)
        ph = np.stack([co.mv[k][0] for k in co.modes], 1)
        phi = zs.solve(w, G0, s.nu, ph)
        T0 = np.zeros((3, 3), complex)
        for c in range(3):
            for rr, j in enumerate(co.modes):
                a, b = radiation_coefficients(phi[:s.NB, c], co.mv[j][0], s.body.area, w, RHO, copies=2)
                T0[rr, c] = -w ** 2 * a + 1j * w * b
        res[wp] = (_nd(r["T"], w, V), _nd(T0, w, V))
    rows = []
    for nm, (i, j) in (("A33", (1, 1)), ("B33", (1, 1)), ("A55", (2, 2)), ("B55", (2, 2)), ("A11", (0, 0)),
                       ("B11", (0, 0)), ("A15", (0, 2)), ("B15", (0, 2))):
        q = 0 if nm[0] == "A" else 1
        new = np.array([res[wp][0][q][i, j] for wp in wps])
        old = np.array([res[wp][1][q][i, j] for wp in wps])
        den = np.maximum(np.abs(old), 0.1 * np.abs(old).max())
        e = float(np.max(np.abs(new - old) / den))
        rows.append(_row("U0", f"U = 0: {nm}' new formulation vs stage-2 zero-speed solver, Wigley I, omega' {list(wps)}",
                         e, 0.0, "radiation.ZeroSpeedFS (stage 2, verified vs Hulme/Capytaine)", TOL_U0, e <= TOL_U0,
                         note="new " + ", ".join(f"{v:.4f}" for v in new) + " | stage2 " + ", ".join(f"{v:.4f}" for v in old)))
    s.release()
    _free()
    return rows


def u0_hemisphere(args):
    """U0h (M4): floating hemisphere at U = 0 with the polar free-surface mesh
    of stage 2 (half domain): new formulation vs Hulme 1982 (3 %, design R2)
    and vs the stage-2 solver (5 %)."""
    kas, mesh = args
    import scipy.sparse as sp
    from hydro.rankine.radiation import hemisphere_case, radiation_coefficients
    from hydro.rankine.unsteady import ForwardSpeedLTI, Coefficients
    from hydro.rankine.analytic import HULME_HEAVE
    HV = 2 / 3 * np.pi
    eh, es, notes = [], [], []
    for ka in kas:
        sol, w, nu = hemisphere_case(ka, n_polar=mesh[0], n_az=mesh[1], n_wl=mesh[2])
        body = sol.body + sol.body.reflect(0)
        fs = sol.fs + sol.fs.reflect(0)
        s = ForwardSpeedLTI(body, fs, sp.csr_matrix((len(fs), len(fs))), np.concatenate([nu, nu]), 0.0, G0, RHO)
        r = Coefficients(s, modes=(3,)).radiation(w, d_form=False)
        A, B = r["A"][0, 0] / HV, r["B"][0, 0] / (HV * w)
        phi = sol.solve(w, G0, nu, sol.body.n[:, 2:3], (1, 1))
        Az, Bz = radiation_coefficients(phi[:len(sol.body), 0], sol.body.n[:, 2], sol.body.area, w, 1.0, copies=4)
        Az, Bz = Az / HV, Bz / (HV * w)
        hA = np.interp(ka, HULME_HEAVE[:, 0], HULME_HEAVE[:, 1])
        hB = np.interp(ka, HULME_HEAVE[:, 0], HULME_HEAVE[:, 2])
        bfl = 0.1 * HULME_HEAVE[:, 2].max()
        eh.append(max(abs(A - hA) / hA, abs(B - hB) / max(hB, bfl)))
        es.append(max(abs(A - Az) / Az, abs(B - Bz) / max(Bz, bfl)))
        notes.append(f"ka {ka}: new ({A:.4f}, {B:.4f}) stage2 ({Az:.4f}, {Bz:.4f}) Hulme ({hA:.4f}, {hB:.4f})")
        s.release()
    _free()
    return [_row("U0h", f"U = 0 hemisphere heave A', B' vs Hulme 1982, ka {list(kas)}, mesh {mesh}", float(max(eh)), 0.0,
                 "Hulme 1982 (B' relative to max(B', 0.1 peak))", TOL_HULME, max(eh) <= TOL_HULME, note="; ".join(notes)),
            _row("U0h", "U = 0 hemisphere: new formulation vs stage-2 solver, same panels", float(max(es)), 0.0,
                 "radiation.ZeroSpeedFS", TOL_U0, max(es) <= TOL_U0)]


def u0_continuity(args):
    """U0c (M4, binding 5 %): the forward-speed coefficients tend to the U = 0
    values as U -> 0 on one grid: diagonal A', B' (33, 55) at Fn 0.05 within
    5 % of U = 0 (same panels; measure relative to max(|value|, 0.1 max));
    Fn 0.025 and the ratio of the two changes (expected ~2 for an O(U) approach)
    as information."""
    wps, grid = args[:2]
    binding = args[2] if len(args) > 2 else True
    h = _wj()
    V = _V(h)
    g = dict(grid, down=grid["up"], b_dn=grid["b_up"], u_dn=0.0)
    vals = {}
    nf = None
    for Fn in (0.0, 0.025, 0.05):
        U = Fn * np.sqrt(G0 * L)
        s, co, _ = _solver(h, U, g, COARSE["hull"], sgn=1.0)
        nf = s.NF
        vals[Fn] = np.array([np.concatenate([_nd(co.radiation(wp * SC)["T"], wp * SC, V)[q][[1, 2], [1, 2]]
                                             for q in (0, 1)]) for wp in wps])
        s.release()
    _free()
    den = np.maximum(np.abs(vals[0.0]), 0.1 * np.abs(vals[0.0]).max(axis=0))
    d05 = np.abs(vals[0.05] - vals[0.0]) / den
    d025 = np.abs(vals[0.025] - vals[0.0]) / den
    ratio = float(np.median(d05 / np.maximum(d025, 1e-12)))
    vnote = ("; values (A33 A55 B33 B55) per omega' at Fn 0/0.025/0.05: " + " | ".join(
        "/".join(" ".join(f"{v:.4f}" for v in vals[F][k]) for F in (0.0, 0.025, 0.05)) for k in range(len(wps))))
    return [_row("U0c", f"U -> 0 continuity: diagonal A', B' (33, 55) at Fn 0.05 vs U = 0, omega' {list(wps)}, M "
                 f"{grid['M']} (NF {nf})" + ("" if binding else " -- refinement (information, B5 review)"),
                 float(d05.max()), 0.0, "U = 0 solution, same grid", TOL_U0 if binding else "info",
                 (d05.max() <= TOL_U0) if binding else True,
                 note=f"Fn 0.025 max change {d025.max():.3e}; median ratio of changes Fn 0.05/0.025 = {ratio:.2f} "
                      f"(2 for an O(U) approach, 4 for O(U^2))" + vnote)]


# =====================================================================
# Haskind and Timman-Newman (M3)
def _pm_solvers(hull, Fn, grid, hm, ref):
    U = Fn * np.sqrt(G0 * hull.L)
    S = {}
    for sgn in (1, -1):
        S[sgn] = _solver(hull, sgn * U, grid, hm, sgn=sgn, ref=ref)
    same = np.allclose(S[1][2]["fs"].c, S[-1][2]["fs"].c) and np.allclose(S[1][2]["nu"], S[-1][2]["nu"])
    return S, U, same


def hk_tn_wigley(args):
    """HK, TN (M3) on the Wigley I at forward speed (symmetric domain, so the
    +U and -U problems share the free-surface panels). Identities (stage note
    10.2, derived): with psi_j the radiation potential at -U,
      Haskind:  X^OT_D,j(U) = -rho int_S psi_j dphi_I/dn dS + rho G_F(phi_D, psi_j),
      Green/TN: T^OT_jk(U) - T^OT_kj(-U) = -rho G_F(phi_k, psi_j),
    G_F the free-surface Green term int_F (phi psi_z - psi phi_z) dA, evaluated
    (a) as the discrete free-surface sum and (b) reduced to the waterline
    integral -(1/g) oint [2 i w U phi psi + U^2 (phi psi_x - psi phi_x)] w' dx.
    Binding 3 % of |X_j| (total) / max(|T_jk(U)|, |T_kj(-U)|) for (a) and (b);
    the classical forms (G_F omitted) and the direct-pressure (D-form) defects
    are information. Haskind at U = 0 (classical, binding 3 %)."""
    Fn, lams, wps, grid = args
    from hydro.rankine.unsteady import green_terms, waterline_fs_term
    h = _wj()
    S, U, same = _pm_solvers(h, Fn, grid, COARSE["hull"], (0.0, ZG))
    sp_, cp, _ = S[1]
    sm, cm, _ = S[-1]
    rows = []
    hk = {"a": [], "b": [], "c": [], "bd": []}
    notes = []
    for lam in lams:
        k = 2 * np.pi / (lam * L)
        w0 = np.sqrt(G0 * k)
        ex = cp.excitation(w0, np.pi, want_phi=True)
        we = ex["omega_e"]
        rm = cm.radiation(we, want_phi=True)
        a, b = ex["sol"], rm["sol"]
        for jj, j in enumerate(cp.modes):
            body_s, fs_s = green_terms(sp_, a["phi_B"][:, 0], -ex["dn_I"], a["sigma"][:, 0], a["phi_F"][:, 0],
                                       b["phi_B"][:, jj], rm["phin"][:, jj], b["sigma"][:, jj], b["phi_F"][:, jj])
            wl = waterline_fs_term(sp_, a["phi_B"][:, 0], b["phi_B"][:, jj], ex["dx_D"], rm["dx"][:, jj], U, we, h)
            cl = -RHO * 2 * np.sum(b["phi_B"][:, jj] * ex["dn_I"] * sp_.body.area)
            Xd = ex["X_d_ot"][jj]
            scl = abs(ex["X"][jj])
            hk["a"].append(abs(Xd - (cl + RHO * fs_s)) / scl)
            hk["b"].append(abs(Xd - (cl + RHO * wl)) / scl)
            hk["c"].append(abs(Xd - cl) / scl)
            hk["bd"].append(abs(Xd - (cl + RHO * wl)) / abs(Xd))
            notes.append(f"lam/L {lam} X{j}: OT diffraction {Xd:.4e}, Haskind+FS {cl + RHO * fs_s:.4e}, +WL "
                         f"{cl + RHO * wl:.4e}, classical {cl:.4e}, D-form diffraction {ex['X_d'][jj]:.4e}, |X| {scl:.3e}")
    rows.append(_row("GI", f"discrete Green identity of the BIE representation (stage-1 type; not an M3 Haskind check): "
                     f"diffraction(+U) vs radiation(-U) with the FS panel sum, Wigley I Fn {Fn}, lambda/L {list(lams)}",
                     float(max(hk["a"])), 0.0, "Green's second identity (stage note 10.2, 10.8.1)",
                     TOL_ID, max(hk["a"]) <= TOL_ID and same, note=("same free-surface panels for +-U" if same else
                                                                     "PANELS DIFFER (row invalid)")))
    rows.append(_row("HK", "Haskind at forward speed with the waterline-reduced free-surface term (b), relative to |X| "
                     "(total incl. Froude-Krylov, as registered)",
                     float(max(hk["b"])), 0.0, "identity (stage note 10.2)", TOL_ID, max(hk["b"]) <= TOL_ID))
    rows.append(_row("HK", "Haskind (b) relative to |X_diffraction| (information, B5 review)", float(max(hk["bd"])), 0.0,
                     "identity", "info", True, note="per entry: " + ", ".join(f"{v:.3f}" for v in hk["bd"])))
    rows.append(_row("HK", "classical Haskind (free-surface term omitted): size of the NK waterline effect",
                     float(max(hk["c"])), 0.0, "information", "info", True, note="; ".join(notes)))
    tn = {"a": [], "b": [], "ot": [], "d": []}
    notes = []
    for wp in wps:
        w = wp * SC
        rp = cp.radiation(w, want_phi=True)
        rm = cm.radiation(w, want_phi=True)
        for (j, kk) in ((1, 2), (0, 1), (0, 2)):
            A_, B_ = rp["sol"], rm["sol"]
            body_s, fs_s = green_terms(sp_, A_["phi_B"][:, kk], rp["phin"][:, kk], A_["sigma"][:, kk],
                                       A_["phi_F"][:, kk], B_["phi_B"][:, j], rm["phin"][:, j], B_["sigma"][:, j],
                                       B_["phi_F"][:, j])
            wl = waterline_fs_term(sp_, A_["phi_B"][:, kk], B_["phi_B"][:, j], rp["dx"][:, kk], rm["dx"][:, j], U, w, h)
            dT = rp["T_ot"][j, kk] - rm["T_ot"][kk, j]
            scl = max(abs(rp["T_ot"][j, kk]), abs(rm["T_ot"][kk, j]))
            tn["a"].append(abs(dT + RHO * fs_s) / scl)
            tn["b"].append(abs(dT + RHO * wl) / scl)
            tn["ot"].append(abs(dT) / scl)
            sd = max(abs(rp["T"][j, kk]), abs(rm["T"][kk, j]))
            tn["d"].append(abs(rp["T"][j, kk] - rm["T"][kk, j]) / sd)
            notes.append(f"w' {wp} ({cp.modes[j]}{cp.modes[kk]}): T^OT_jk(U) {rp['T_ot'][j, kk]:.4e}, T^OT_kj(-U) "
                         f"{rm['T_ot'][kk, j]:.4e}, -rho G_F sum {-RHO * fs_s:.3e}, WL {-RHO * wl:.3e}; D-form "
                         f"{rp['T'][j, kk]:.4e} / {rm['T'][kk, j]:.4e}")
    rows.append(_row("GI", f"discrete Green identity of the BIE representation (stage-1 type; not an M3 Timman-Newman "
                     f"check): radiation +U vs -U with the FS panel sum, Wigley I Fn {Fn}, omega' {list(wps)}, pairs (35), "
                     f"(13), (15)", float(max(tn["a"])), 0.0, "Green's second identity (stage note 10.2, 10.8.1)",
                     TOL_ID, max(tn["a"]) <= TOL_ID and same))
    rows.append(_row("TN", "Timman-Newman identity with the waterline-reduced free-surface term (b)", float(max(tn["b"])),
                     0.0, "identity (stage note 10.2)", TOL_ID, max(tn["b"]) <= TOL_ID))
    rows.append(_row("TN", "classical Timman-Newman T^OT_jk(U) = T^OT_kj(-U) (Ogilvie-Tuck form): defect = the NK "
                     "waterline term", float(max(tn["ot"])), 0.0, "information", "info", True))
    rows.append(_row("TN", "classical Timman-Newman for the direct-pressure (D-form) coefficients: defect = waterline "
                     "term + NK inconsistency E_j", float(max(tn["d"])), 0.0, "information", "info", True,
                     note="; ".join(notes)))
    # Haskind at U = 0 (classical)
    for sgn in (1, -1):
        S[sgn][0].release()
    s0, c0, _ = _solver(h, 0.0, grid, COARSE["hull"], sgn=1.0, ref=(0.0, ZG))
    e0 = []
    for lam in lams:
        k = 2 * np.pi / (lam * L)
        w0 = np.sqrt(G0 * k)
        ex = c0.excitation(w0, np.pi, want_phi=True)
        r0 = c0.radiation(w0, want_phi=True)
        for jj in range(3):
            cl = -RHO * 2 * np.sum(r0["sol"]["phi_B"][:, jj] * ex["dn_I"] * s0.body.area)
            e0.append(abs(ex["X_d"][jj] - cl) / abs(ex["X"][jj]))
    s0.release()
    _free()
    rows.append(_row("HK", "Haskind at U = 0 (classical), Wigley I", float(max(e0)), 0.0, "identity", TOL_ID,
                     max(e0) <= TOL_ID))
    return rows


def tn_submerged(args):
    """TN-d (M3, binding 3 %): classical Timman-Newman T^OT_jk(U) = T^OT_kj(-U)
    for a SUBMERGED prolate spheroid (a/b = 5, centre depth 2 b), where it is a
    theorem (no waterline); Fn 0.3."""
    Fn, wps, grid = args
    from hydro.rankine import panels as P
    from hydro.rankine.panels import Panels
    from hydro.rankine.unsteady import (build_grid, dx_operator, ForwardSpeedLTI, Coefficients, NoWaterline)
    sph = P.ellipsoid(0.5, 0.1, 0.1, 40, 12, phi_range=(-np.pi / 2, np.pi / 2), spacing="cosine")
    raw = sph.raw.copy()
    raw[:, :, 2] -= 0.2
    body = Panels(raw)
    nw = NoWaterline(-0.5, 0.5, 1.0)
    U = Fn * SC
    S = {}
    for sgn in (1, -1):
        G = build_grid(nw, sgn=sgn, **grid)
        D, _ = dx_operator(G, sgn)
        s = ForwardSpeedLTI(body, G["fs"], D, G["nu"] * SC, sgn * U, G0, RHO)
        S[sgn] = (s, Coefficients(s, ref=(0.0, -0.2)), G)
    errs, notes = [], []
    for wp in wps:
        rp = S[1][1].radiation(wp * SC, d_form=False)
        rm = S[-1][1].radiation(wp * SC, d_form=False)
        for (j, k) in ((1, 2), (0, 1), (0, 2)):
            a, b = rp["T_ot"][j, k], rm["T_ot"][k, j]
            errs.append(abs(a - b) / max(abs(a), abs(b)))
            notes.append(f"w' {wp} ({(1, 3, 5)[j]}{(1, 3, 5)[k]}): {a:.4e} / {b:.4e}")
    for sgn in (1, -1):
        S[sgn][0].release()
    _free()
    return [_row("TN", f"classical Timman-Newman, submerged spheroid a/b 5 depth 2b, Fn {Fn}, omega' {list(wps)}",
                 float(max(errs)), 0.0, "Timman & Newman 1962 (theorem without a waterline)", TOL_ID,
                 max(errs) <= TOL_ID, note="; ".join(notes))]


def tn_prismatic(args):
    """TNp (M3): Timman-Newman identities for the fore-aft ASYMMETRIC
    planing-type prismatic V body, +U and -U solved explicitly on shared
    panels: (a) discrete Green identity (GI, stage-1 type), (b) waterline
    reduction (binding 3 %: restored in the B5 review -- the B5 reason for
    making it information, a re-entrant 200 deg corner, was geometrically
    wrong: the fluid corner of the flared waterline is CONVEX, 180 - 20 = 160
    deg; stage note 10.8.1), classical defect (information). args: Fn, omega',
    grid, hull mesh, hull keywords (D0 = 0.06: waterline on the vertical side
    above the chine, a wall-sided planing-type variant), tag, binding (False:
    information rows for the refinement series)."""
    Fn, wps, grid, hm = args[:4]
    hkw = args[4] if len(args) > 4 else {}
    tag = args[5] if len(args) > 5 else "TNp"
    binding = args[6] if len(args) > 6 else True
    from hydro.rankine.hull import prismatic_v
    from hydro.rankine.unsteady import green_terms, waterline_fs_term
    h = prismatic_v(**hkw)
    S, U, same = _pm_solvers(h, Fn, grid, hm, (0.0, 0.0))
    sp_, cp, _ = S[1]
    sm, cm, _ = S[-1]
    ea, eb, eot, notes = [], [], [], []
    for wp in wps:
        w = wp * SC
        rp = cp.radiation(w, want_phi=True)
        rm = cm.radiation(w, want_phi=True)
        for (j, kk) in ((1, 2), (2, 1), (0, 1), (0, 2)):
            A_, B_ = rp["sol"], rm["sol"]
            _, fs_s = green_terms(sp_, A_["phi_B"][:, kk], rp["phin"][:, kk], A_["sigma"][:, kk], A_["phi_F"][:, kk],
                                  B_["phi_B"][:, j], rm["phin"][:, j], B_["sigma"][:, j], B_["phi_F"][:, j])
            wl = waterline_fs_term(sp_, A_["phi_B"][:, kk], B_["phi_B"][:, j], rp["dx"][:, kk], rm["dx"][:, j], U, w, h)
            dT = rp["T_ot"][j, kk] - rm["T_ot"][kk, j]
            scl = max(abs(rp["T_ot"][j, kk]), abs(rm["T_ot"][kk, j]))
            ea.append(abs(dT + RHO * fs_s) / scl)
            eb.append(abs(dT + RHO * wl) / scl)
            eot.append(abs(dT) / scl)
            notes.append(f"w' {wp} ({cp.modes[j]}{cp.modes[kk]}): dT {dT:.3e}, -rho G_F {-RHO * fs_s:.3e}, WL {-RHO * wl:.3e}, |T| {scl:.3e}")
    for sgn in (1, -1):
        S[sgn][0].release()
    _free()
    desc = (f"prismatic V body {hkw or '(flared waterline, deadrise 20 deg)'}, Fn {Fn}, M {grid['M']}, hull {hm}")
    tol = TOL_ID if binding else "info"
    return [_row("GI" if binding else tag, f"{desc}: discrete Green identity (a) of the BIE representation (stage-1 "
                 "type, not an M3 check)", float(max(ea)), 0.0, "Green's second identity", tol,
                 (max(ea) <= TOL_ID and same) if binding else True),
            _row(tag, f"{desc}: Timman-Newman identity (b), waterline-reduced FS term", float(max(eb)), 0.0,
                 "identity (stage note 10.2 item 4)", tol, (max(eb) <= TOL_ID) if binding else True),
            _row(tag, f"{desc}: classical Timman-Newman defect (OT form)", float(max(eot)), 0.0,
                 "information", "info", True, note="; ".join(notes))]


# =====================================================================
# independent forward-speed reference: submerged sphere (Wu & Eatock Taylor 1988; B5 review)
WE_DATA = os.path.join(ROOT, "data", "external", "wu_eatock_taylor_sphere")
WE_R = 0.1                     # sphere radius (units of the grid length scale L = 1); centre depth 2 r
WE_GRID = dict(M=8, up=1.0, down=6.0, lat=4.0, r_up=1.08, r_dn=1.05, r_lat=1.06, b_up=0.5, b_dn=3.0, b_lat=2.0,
               u_dn=1.5, u_lat=0.5, nu_max=1.5)
WE_MESH = (24, 12)             # polar x azimuth panels on the half sphere y >= 0


def sphere_db_mterms(pan, c, r, U):
    """Ogilvie-Tuck m-terms of the TRANSLATIONS for the double-body flow past a
    sphere (radius r, centre c, stream -U x_hat): Phi = -U x'(1 + r^3/(2 R^3)),
    W = grad Phi (W.n = 0 on the sphere), m_j = -(n.grad) W_j (derivation:
    linearised exact body condition on the displaced sphere, stage note
    10.8.4). Returns (m1, m3) on the panel centroids."""
    Xr = pan.c - np.asarray(c, float)[None]
    R = np.linalg.norm(Xr, axis=1)
    a = r ** 3 / 2
    x = Xr[:, 0]
    d = np.eye(3)
    H = np.zeros((len(R), 3, 3))
    for i in range(3):
        for j in range(3):
            H[:, i, j] = -U * a * (-3 * d[i, 0] * Xr[:, j] / R ** 5 - 3 * d[j, 0] * Xr[:, i] / R ** 5
                                   - 3 * x * d[i, j] / R ** 5 + 15 * x * Xr[:, i] * Xr[:, j] / R ** 7)
    m = -np.einsum("nj,nij->ni", pan.n, H)
    return m[:, 0], m[:, 2]


def we_sphere(args):
    """WE (forward-speed reference, B5 review): submerged sphere, radius r,
    centre depth h = 2 r, Fr = U/sqrt(g r) = 0.4, deep water, against the
    exact-Green-function linear solution of Wu & Eatock Taylor (1988) as
    tabulated by Mei et al. (2020, J. Mar. Sci. Eng. 8:106, tables 3, 5, 6;
    data/external/wu_eatock_taylor_sphere/). No waterline: the check exercises
    the forward-speed free-surface operator (upwind d/dx, beaches, the LTI
    system) independently of the waterline identities. Reference model (stage
    note 10.8.4): Neumann-Kelvin free-surface condition with the m-terms of the
    STEADY (double-body) flow past the sphere in the body condition and the
    m-term (Ogilvie-Tuck) force -- identified from the tables (A_jj - A_inf ~
    0.35/(k_e r), i.e. a C_inf/omega^2 term, and B13 -> const at high
    frequency, both absent with uniform-stream m-terms, which vanish for
    translations); the same model is run here (sphere_db_mterms; the steady
    flow's effect on the free-surface condition is neglected, as in the
    reference). Radiation at the 13 tabulated k_e r (encounter, 0.6-5.0, all
    above the critical 0.3906), binding: A11, A33 within 3 % of the value;
    B11, B33 within max(5 % of the value, 2 % of the series maximum); A13,
    A31, B13, B31 within max(5 % of the value, 5 % of the series maximum of
    |.|). Information: the uniform-stream (NK m-term) variant, the classical
    Timman-Newman defect, the exciting forces (both variants) against the
    reference and against Mei et al.'s NK column (tables 1-2)."""
    grid, mesh, rad_only = args
    from hydro.rankine import panels as P
    from hydro.rankine.panels import Panels
    from hydro.rankine.unsteady import build_grid, dx_operator, ForwardSpeedLTI, Coefficients, NoWaterline
    r = WE_R
    sph = P.ellipsoid(r, r, r, mesh[0], mesh[1], phi_range=(-np.pi / 2, np.pi / 2), spacing="cosine")
    raw = sph.raw.copy()
    raw[:, :, 2] -= 2 * r
    body = Panels(raw)
    nw = NoWaterline(-r, r, 1.0)
    U = 0.4 * np.sqrt(G0 * r)
    G = build_grid(nw, sgn=1.0, **grid)
    D, _ = dx_operator(G, 1.0)
    s = ForwardSpeedLTI(body, G["fs"], D, G["nu"] * SC, U, G0, RHO)
    co_nk = Coefficients(s, ref=(0.0, -2 * r), modes=(1, 3))
    co_db = Coefficients(s, ref=(0.0, -2 * r), modes=(1, 3))
    m1, m3 = sphere_db_mterms(body, (0.0, 0.0, -2 * r), r, U)
    co_db.mv = {1: (body.n[:, 0].copy(), m1), 3: (body.n[:, 2].copy(), m3)}
    ref = _load_we("we1988_sphere_fr040_h2r_radiation.csv")
    calc = {"DB": [], "NK": []}
    for ker in ref[:, 0]:
        w = np.sqrt(G0 * ker / r)
        for key, co in (("DB", co_db), ("NK", co_nk)):
            T = co.radiation(w, d_form=False)["T_ot"]
            A, B = -T.real / w ** 2, T.imag / w
            a_, b_ = A / (RHO * np.pi * r ** 3), B / (RHO * np.pi * w * r ** 3)
            calc[key].append([a_[0, 0], b_[0, 0], a_[1, 1], b_[1, 1], a_[0, 1], a_[1, 0], b_[0, 1], b_[1, 0]])
    rows = []
    names = ["A11", "B11", "A33", "B33", "A13", "A31", "B13", "B31"]
    src = "Wu & Eatock Taylor 1988 via Mei et al. 2020 (JMSE 8:106) tables 3, 5, 6"
    for key in ("DB", "NK"):
        c = np.array(calc[key])
        worst, fails, notes = 0.0, 0, []
        for q, nm in enumerate(names):
            e = ref[:, q + 1]
            if nm in ("A11", "A33"):
                tol = 0.03 * np.abs(e)
            elif nm in ("B11", "B33"):
                tol = np.maximum(0.05 * np.abs(e), 0.02 * np.abs(e).max())
            else:
                tol = np.maximum(0.05 * np.abs(e), 0.05 * np.abs(e).max())
            ex = np.abs(c[:, q] - e) / tol
            worst = max(worst, float(ex.max()))
            fails += int(np.sum(ex > 1))
            notes.append(f"{nm}: calc " + " ".join(f"{v:.4f}" for v in c[:, q]) + f" (max err/tol {ex.max():.2f})")
        tn = np.abs(c[:, 4] + c[:, 5]) / np.maximum(np.abs(c[:, 4]), np.abs(c[:, 5]))
        binding = key == "DB"
        rows.append(_row("WE" if binding else "WEnk", f"submerged sphere Fr 0.4, h = 2r, radiation with "
                         f"{'double-body (reference-model)' if binding else 'uniform-stream (NK)'} m-terms, m-term force "
                         f"(A, B of 11, 33, 13, 31 at {len(ref)} k_e r): max error / tolerance (NF {s.NF})",
                         worst, 1.0, src, "<= 1 (per-coefficient rules)" if binding else "info",
                         worst <= 1.0 if binding else True,
                         note=f"{fails} of {8 * len(ref)} values outside; classical TN defect |A13 + A31| / max max "
                              f"{tn.max():.3f}; " + "; ".join(notes)))
    if not rad_only:
        refx = _load_we("we1988_sphere_fr040_h2r_excitation.csv")
        out = {"DB": ([], []), "NK": ([], [])}
        for kr in refx[:, 0]:
            k = kr / r
            w0 = np.sqrt(G0 * k)
            ex = co_nk.excitation(w0, np.pi, want_phi=True)
            we = ex["omega_e"]
            phD = ex["sol"]["phi_B"][:, 0]
            phI = ex["phi_I"]
            sc = RHO * G0 * np.pi * r ** 3 * (we ** 2 / G0)
            # NK: direct pressure (Froude-Krylov exact, -rho (phi_t - U phi_x)); DB: m-term form with the
            # double-body m_j, equal to the direct pressure -rho (phi_t + W.grad phi) by Tuck's theorem (W.n = 0)
            out["NK"][0].append(ex["X"][0] / sc)
            out["NK"][1].append(ex["X"][1] / sc)
            F = []
            for j in (1, 3):
                nj, mj = co_db.mv[j]
                F.append(RHO * 2 * np.sum((phI + phD) * (1j * we * nj - mj) * body.area) / sc)
            out["DB"][0].append(F[0])
            out["DB"][1].append(F[1])
        notes = []
        worst = {}
        for key in ("DB", "NK"):
            f1, f3 = np.array(out[key][0]), np.array(out[key][1])
            ws = 0.0
            for nm, c, re_, im_, nre, nim in (("f1", f1, refx[:, 1], refx[:, 2], refx[:, 5], refx[:, 6]),
                                              ("f3", f3, refx[:, 3], refx[:, 4], refx[:, 7], refx[:, 8])):
                e = np.hypot(re_, im_)
                ws = max(ws, float(np.max(np.abs(np.abs(c) - e) / np.maximum(0.03 * e, 0.01 * e.max()))))
                notes.append(f"{key} |{nm}| calc " + " ".join(f"{v:.4f}" for v in np.abs(c)) + " WE " +
                             " ".join(f"{v:.4f}" for v in e) + " Mei-NK " + " ".join(f"{v:.4f}" for v in np.hypot(nre, nim)))
            worst[key] = ws
        rows.append(_row("WEx", f"submerged sphere: exciting-force amplitudes |f1|, |f3| (head seas, {len(refx)} kr) vs the "
                         f"reference, max error/max(3 %, 1 % of max) (information): double-body m-terms {worst['DB']:.2f}, "
                         f"uniform-stream {worst['NK']:.2f}", worst["DB"], 1.0, "Wu & Eatock Taylor 1988 via Mei et al. 2020 "
                         "tables 1-2", "info", True, note="; ".join(notes)))
    s.release()
    _free()
    return rows


def _load_we(name):
    lines = [ln for ln in open(os.path.join(WE_DATA, name), encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    return np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]])


# =====================================================================
# Kramers-Kronig, limits, spectrum (M3, M4)
def _kk_errors(w, T, Ainf, Binf, C0, wtest, wband):
    from scipy.interpolate import PchipInterpolator
    A = -T.real / w[:, None, None] ** 2
    B = T.imag / w[:, None, None]
    out = {}
    for j in range(3):
        for k in range(3):
            Aref = np.abs(A[(w >= wband[0]) & (w <= wband[1]), j, k]).max()
            if Aref < 1e-3 * np.abs(A[:, 1, 1]).max():
                continue
            f = B[:, j, k] - Binf[j, k]
            fi = PchipInterpolator(np.log(w), f)
            wq = np.exp(np.linspace(np.log(w[0]), np.log(w[-1]), 200001))
            fq = fi(np.log(wq))
            errs = []
            for wt in wtest:
                ft = float(fi(np.log(wt)))
                den = wq ** 2 - wt ** 2
                with np.errstate(divide="ignore", invalid="ignore"):
                    ig = np.where(np.abs(den) > 1e-12, (fq - ft) / den, 0.0)
                I1 = np.trapezoid(ig * wq, np.log(wq))
                I2 = ft * (np.log(abs((w[-1] - wt) / (w[-1] + wt))) - np.log(abs((w[0] - wt) / (w[0] + wt)))) / (2 * wt)
                # [0, w0]: B - B_inf = f[0] (constant extension); I1 and I2 already cover f - ft and ft on
                # [w0, w_end]. (B5 had (f[0] - ft) here, i.e. -ft int_0^w0 dw/(w^2 - wt^2) counted twice: an
                # evaluation bug of (2/pi)(B(wt) - B_inf) w0/wt^2, corrected after the B5 run; stage note 10.8.3.)
                I0 = float(f[0]) * np.log(abs((w[0] - wt) / (w[0] + wt))) / (2 * wt)
                Akk = Ainf[j, k] - C0[j, k] / wt ** 2 + 2 / np.pi * (I1 + I2 + I0)
                Ad = float(PchipInterpolator(np.log(w), A[:, j, k])(np.log(wt)))
                errs.append((Akk - Ad) / Aref)
            out[(j, k)] = np.array(errs)
    return out


def kk_unit(args):
    """KKu (M2, binding 1e-4; added in the B5 review): _kk_errors on an
    analytic causal system T = -w^2 A_inf + i w B_inf + C0 + i w s K^(w), K(t) =
    e^{-t/tc} cos(b t) (K^ = (1/2)[1/(1/tc + i(w - b)) + 1/(1/tc + i(w + b))],
    exact A and B), on the KK frequency grid (log omega' 0.01-200, npts points
    + a band of npts/4), for (tc', b') = (0.6, 2.5) and (0.3, 4.0) with
    nonzero B_inf and C0; max error over the test frequencies and entries."""
    npts, = args
    errs, notes = [], []
    for tcp, bp in ((0.6, 2.5), (0.3, 4.0)):
        tc, b = tcp / SC, bp * SC
        a = 1.0 / tc
        Ainf = np.diag([0.5, 1.0, 0.05])
        Binf = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.05], [0.0, -0.05, 0.02]])
        C0 = np.diag([0.0, -0.02, -0.004])
        wp = np.unique(np.concatenate([np.exp(np.linspace(np.log(0.01), np.log(200.0), npts)),
                                       np.linspace(0.7 * 2.6, 1.3 * 2.6, npts // 4)]))
        w = wp * SC
        Kh = 0.5 * (1 / (a + 1j * (w - b)) + 1 / (a + 1j * (w + b)))
        T = np.zeros((len(w), 3, 3), complex)
        for j in range(3):
            for k in range(3):
                T[:, j, k] = -w ** 2 * Ainf[j, k] + 1j * w * Binf[j, k] + C0[j, k]
            T[:, j, j] += 1j * w * (1.0, 2.0, 0.3)[j] * Kh
        wtest = np.array([1.0, 1.659, 2.212, 3.318, 4.424, 5.53, 8.0]) * SC
        e = _kk_errors(w, T, Ainf, Binf, C0, wtest, (1.0 * SC, 8.0 * SC))
        m = max(float(np.abs(v).max()) for v in e.values())
        errs.append(m)
        notes.append(f"(tc', b') = ({tcp}, {bp}): {m:.2e}")
    return [_row("KKu", f"Kramers-Kronig evaluation on an analytic causal kernel ({npts} log points + band)",
                 float(max(errs)), 0.0, "exact A, B of K(t) = exp(-t/tc) cos(b t)", 1e-4, max(errs) <= 1e-4,
                 note="; ".join(notes))]


def kk_check(args):
    """KK (M3, binding 2 %): A(omega) from B(omega) by the forward-speed
    Kramers-Kronig relation (stage note 10.2)
        A(w) = A_inf - C0/w^2 + (2/pi) PV int_0^inf (B(w') - B_inf)/(w'^2 - w^2) dw',
    A_inf, B_inf from the instantaneous (phi_F = 0) problem of the same
    discrete system, C0 from its omega = 0 solve; B on a log grid omega' in
    [0.01, 200] (+ a dense band around tau = 1/4). Error relative to
    max_{omega' in [1, 8]} |A_jk| at omega' in {1, 1.659, 2.212, 3.318, 4.424,
    5.53, 8}; entries with max |A_jk| < 1e-3 max |A33| skipped. Both force forms."""
    Fn, npts, grid, hm = args
    from hydro.rankine.tables import static_stiffness
    h = _wj()
    U = Fn * np.sqrt(G0 * L)
    s, co, _ = _solver(h, U, grid, hm)
    inst = co.instantaneous()
    C0, C0ot = static_stiffness(co)
    wc = G0 / (4 * U) / SC if U else 1.0
    wp = np.unique(np.concatenate([np.exp(np.linspace(np.log(0.01), np.log(200.0), npts)),
                                   np.linspace(0.7 * wc, 1.3 * wc, npts // 4)]))
    w = wp * SC
    TD, TO = [], []
    for x in w:
        r = co.radiation(x)
        TD.append(r["T"])
        TO.append(r["T_ot"])
    TD, TO = np.array(TD), np.array(TO)
    wtest = np.array([1.0, 1.659, 2.212, 3.318, 4.424, 5.53, 8.0]) * SC
    rows = []
    for form, T, Bi, Ci in (("D", TD, inst["B"], C0), ("OT", TO, inst["B_ot"], C0ot)):
        errs = _kk_errors(w, T, inst["A"], Bi, Ci, wtest, (1.0 * SC, 8.0 * SC))
        emax = max(float(np.abs(e).max()) for e in errs.values())
        rows.append(_row("KK", f"Kramers-Kronig, Wigley I Fn {Fn}, {form}-form, {len(w)} frequencies, entries "
                         f"{sorted(errs)}", emax, 0.0, "Kramers-Kronig with A_inf, B_inf, C0 (stage note 10.2)", TOL_KK,
                         emax <= TOL_KK, note="; ".join(f"{(1, 3, 5)[j]}{(1, 3, 5)[k]}: {np.abs(e).max():.2e}"
                                                        for (j, k), e in errs.items())
                         + f"; C0 diag {np.diag(Ci)}, C_inf diag {np.diag(inst['C'] if form == 'D' else inst['C_ot'])}"))
    # high-frequency limit (INF): A(omega' = 200) vs A_inf
    Ahi = -TD[-1].real / w[-1] ** 2
    ehi = float(np.abs(Ahi - inst["A"]).max() / np.abs(inst["A"]).max())
    rows.append(_row("INF", f"high-frequency limit: A(omega' = {wp[-1]:.0f}) vs A_inf of the discrete system", ehi, 0.0,
                     "limit (M4)", TOL_INF, ehi <= TOL_INF))
    s.release()
    _free()
    return rows


def inf_g0(args):
    """INF (M4, binding 1 %): A_inf of the discrete system (phi_F = 0 on the
    truncated free surface) vs the exact phi = 0 solve (odd lid image) on the
    same hull mesh. G0 (M4, binding 5 %): g -> 0 at fixed U, omega (omega L/U
    = 2 pi): max |T - T_inf| / max |T_inf| at Fn 8 (Fn 1, 2, 4 information;
    expected ~1/Fn^2)."""
    grid, hm = args
    from hydro.rankine.bie import ImageOperator, neumann_green
    h = _wj()
    Uq = 1.0
    errs = {}
    inst = None
    for Fn in (1.0, 2.0, 4.0, 8.0):
        g = (Uq / Fn) ** 2 / L
        s, co, _ = _solver(h, Uq, grid, hm, g=g)
        inst = co.instantaneous()
        w = 2 * np.pi * Uq / L
        r = co.radiation(w)
        Tinf = -w ** 2 * inst["A"] + 1j * w * inst["B"] + inst["C"]
        errs[Fn] = float(np.abs(r["T"] - Tinf).max() / np.abs(Tinf).max())
        body = s.body
        mv = co.mv
        s.release()
    op = ImageOperator(body, axes=(1, 2), parities=[(1, -1)])
    nmat = np.stack([mv[k][0] for k in (1, 3, 5)], 1)
    phi = np.stack([neumann_green(op, nmat[:, c], (1, -1)) for c in range(3)], 1)
    Aex = np.array([[-RHO * 2 * np.sum(phi[:, c] * nmat[:, r] * body.area) for c in range(3)] for r in range(3)])
    ea = float(np.abs(inst["A"] - Aex).max() / np.abs(Aex).max())
    ed = np.abs(np.diag(inst["A"]) - np.diag(Aex)) / np.abs(np.diag(Aex))
    _free()
    return [_row("INF", "A_inf (phi_F = 0 on the truncated free surface) vs exact phi = 0 image solve, Wigley I "
                 "(relative to the largest entry, as registered)", ea, 0.0,
                 "odd lid image", TOL_INF, ea <= TOL_INF,
                 note=f"diag discrete {np.diag(inst['A'])}, exact {np.diag(Aex)}"),
            _row("INF", "A_inf vs exact phi = 0 image solve per diagonal entry (11, 33, 55; information, B5 review)",
                 float(ed.max()), 0.0, "odd lid image", "info", True,
                 note="relative errors " + ", ".join(f"{v:.3e}" for v in ed)),
            _row("G0", "g -> 0: max |T - T_inf| / max |T_inf| at Fn 8 (omega L/U = 2 pi)", errs[8.0], 0.0,
                 "phi = 0 (high-speed) theory, same discrete system", TOL_G0, errs[8.0] <= TOL_G0,
                 note="Fn 1, 2, 4, 8: " + ", ".join(f"{errs[f]:.3e}" for f in (1.0, 2.0, 4.0, 8.0)))]


def spectrum(args):
    """SP (M3/E2): eigenvalues of the LTI matrix M = [[-N, -R_s^-1], [-g I, -N]]
    on a small grid. Forward speed (beach on): max Re(lambda) < 0 (binding).
    U = 0 without any beach (nu = 0, no dissipation): max Re(lambda) <= 1e-5
    sqrt(g/L) (binding; acceptance M3 "energy drift <= 0.1 % per 100 s" at the
    USV scale L = 5.4 m is 7.4e-6 sqrt(g/L); expected to FAIL: the collocation
    Dirichlet-to-Neumann map is not exactly symmetric, exploration B5-E11).
    U = 0 with the beach: max Re(lambda) <= 1e-5 sqrt(g/L) as well."""
    cases, grid, hm = args
    from hydro.rankine.hull import prismatic_v
    rows = []
    for hname, Fn, beach in cases:
        h = _wj() if hname == "wigleyJ" else prismatic_v()
        U = Fn * np.sqrt(G0 * L)
        gk = dict(grid) if beach else dict(grid, nu_max=0.0)
        s, co, _ = _solver(h, U, gk, hm, sgn=1.0)
        NF = s.NF
        Mc = s.state_matrix()           # includes the wall traces of the closure (stage note 10.8.2)
        s.release()
        _free()
        ev = np.linalg.eigvals(Mc)
        del Mc
        mr = float(ev.real.max() / SC)
        if beach and U != 0:
            rows.append(_row("SP", f"spectrum, {hname} Fn {Fn}, beach on (NF {NF}): max Re(lambda) [sqrt(g/L)]", mr, 0.0,
                             "< 0 (every mode decays)", "< 0", mr < 0.0,
                             note=f"max |lambda| {np.abs(ev).max() / SC:.1f}, eigenvalues with Re > 0: {int(np.sum(ev.real > 0))}"))
        else:
            rows.append(_row("SP", f"spectrum, {hname} U = 0, {'beach on' if beach else 'no beach (nu = 0)'} (NF {NF}): "
                             f"max Re(lambda) [sqrt(g/L)]", mr, 0.0,
                             "acceptance M3 energy drift 0.1 %/100 s at L = 5.4 m (7.4e-6 sqrt(g/L))", 1e-5, mr <= 1e-5,
                             note=f"eigenvalues with Re > 0: {int(np.sum(ev.real > 0))}; max |Re| "
                                  f"{np.abs(ev.real).max() / SC:.2e}"))
    return rows


# =====================================================================
# manufactured-solution test of the discrete d/dx (M2; B5 review)
def _mms_field(X, Y, a, b, Lx=0.9, Ly=0.5):
    """f = cos(a x) cos(b y) wx(x) wy(y) (super-Gaussian windows, even in y) and its exact d/dx."""
    wx = np.exp(-(X / Lx) ** 8)
    wy = np.exp(-(Y / Ly) ** 8)
    dwx = wx * (-8 * X ** 7 / Lx ** 8)
    f = np.cos(a * X) * np.cos(b * Y) * wx * wy
    fx = (-a * np.sin(a * X) * wx + np.cos(a * X) * dwx) * np.cos(b * Y) * wy
    return f, fx


def dxm_check(args):
    """DXM (M2, B5 review): the discrete d/dx at fixed y (production wall
    closure, wall nodes carrying the exact trace) applied to a manufactured
    field f = cos(a x) cos(b y) w(x, y), a = b = 2 pi/0.25 L, on the CHECK
    band grids (Wigley I and the prismatic V body), stream +x and -x, M = 20,
    40, 80. Error max |D f - f_x|/a over three regions inside the window: the
    wall line (band line 0, hull columns), the rest of the band, the interior.
    Binding: observed order between M = 40 and 80 >= 1 on the wall line and in
    the band, >= 1.5 in the interior (min over both stream directions); linear
    functions differentiated exactly (<= 1e-10) in the band at M = 80. The B5
    wall rule as information (its wall-line error does not decrease)."""
    Ms, = args
    from hydro.rankine.hull import prismatic_v
    from hydro.rankine.unsteady import build_grid, dx_operator
    rows = []
    lin = 0.0
    for hname in ("wigleyJ", "prismV"):
        h = _wj() if hname == "wigleyJ" else prismatic_v()
        res = {}
        for wall in ("trace", "rule"):
            for sgn in (1.0, -1.0):
                e = {}
                for M in Ms:
                    G = build_grid(h, 0.0, 0.0, sgn, **dict(CHECK_GRID, M=M))
                    D, info = dx_operator(G, sgn, wall=wall, hull=h)
                    wl = info["wall"]
                    nx, ny = G["shape"]
                    X, Y = G["Xc"], G["Yc"]
                    a = b = 2 * np.pi / 0.25
                    f, fx = _mms_field(X, Y, a, b)
                    Df = D @ f.T.reshape(-1)
                    fl = X + 2 * Y
                    Dl = D @ fl.T.reshape(-1)
                    if wl is not None:
                        fw, _ = _mms_field(wl["xw"], wl["yw"], a, b)
                        Df = Df + wl["C"] @ fw
                        Dl = Dl + wl["C"] @ (wl["xw"] + 2 * wl["yw"])
                    err = np.abs(Df.reshape(ny, nx).T - fx) / a
                    i_s, i_b, J = G["i_s"], G["i_b"], G["J"]
                    reg = np.zeros((nx, ny), int)
                    reg[i_s:i_b, 1:J] = 1
                    reg[i_s:i_b, 0] = 2
                    inwin = (np.abs(X) < 0.8) & (Y < 0.45)
                    e[M] = [float(err[(reg == r) & inwin].max()) for r in (2, 1, 0)]
                    if wall == "trace" and M == Ms[-1]:
                        lin = max(lin, float(np.abs(Dl.reshape(ny, nx).T[i_s:i_b, :J] - 1).max()))
                res[(wall, sgn)] = e
        for wall in ("trace", "rule"):
            orders = np.array([[np.log(res[(wall, sg)][Ms[-2]][k] / res[(wall, sg)][Ms[-1]][k]) /
                                np.log(Ms[-1] / Ms[-2]) for k in range(3)] for sg in (1.0, -1.0)])
            omin = orders.min(axis=0)
            note = "; ".join(f"stream {'+' if sg > 0 else '-'}: " + ", ".join(
                f"M{M} {res[(wall, sg)][M][0]:.2e}/{res[(wall, sg)][M][1]:.2e}/{res[(wall, sg)][M][2]:.2e}"
                for M in Ms) for sg in (1.0, -1.0)) + " (wall line / band / interior)"
            if wall == "trace":
                for k, (rn, thr) in enumerate((("wall line", 1.0), ("band", 1.0), ("interior", 1.5))):
                    rows.append(_row("DXM", f"manufactured d/dx, {hname}, {rn}: observed order M {Ms[-2]} -> {Ms[-1]} "
                                     "(min over both stream directions)", float(omin[k]), 0.0,
                                     "exact derivative of the manufactured field", f">= {thr}", omin[k] >= thr,
                                     note=note if k == 0 else ""))
            else:
                rows.append(_row("DXM", f"manufactured d/dx, {hname}, B5 wall rule (information): wall-line order "
                                 f"{omin[0]:.2f}", float(omin[0]), 0.0, "information", "info", True, note=note))
    rows.append(_row("DXM", f"manufactured d/dx: linear functions in the band and on the wall line at M {Ms[-1]}",
                     lin, 0.0, "exact", 1e-10, lin <= 1e-10))
    return rows


# =====================================================================
# time domain (M2, M3, E2)
def _pulse(mode, a0, t0, tau):
    def motion(t):
        e = np.exp(-((t - t0) / tau) ** 2)
        x, xd, xdd = np.zeros(3), np.zeros(3), np.zeros(3)
        x[mode] = a0 * e
        xd[mode] = a0 * e * (-2 * (t - t0) / tau ** 2)
        xdd[mode] = a0 * e * (4 * (t - t0) ** 2 / tau ** 4 - 2 / tau ** 2)
        return x, xd, xdd
    return motion


def td_checks(args):
    """TD (M2, binding 1 %): time-domain RK4 integration of the LTI system
    (heave pulse) vs the frequency-domain transfer function T33 at omega' 2, 3,
    4.5 (FFT of the transient, record t' = 60, dt' = 0.05). CU (M2, binding
    1 %): Cummins reconstruction through the TABLE pipeline (K from B(omega) by
    tables.memory_kernel, A_inf, B_inf, C0, tables.RadiationForce) vs the
    time-domain force, heave and pitch pulses: rms error / rms of the diagonal
    force. EN (M3/E2): pseudo-energy balance H - H0 = int (P_body + P_wl +
    P_beach) dt (stage note 10.2): (a) U = 0 and (b) Fn 0.3, band-limited heave
    pulse (tau' = 1, t0' = 5): at U = 0 |residual| / int |P_body| at the end
    of the forcing t' = 8 <= 1 % (binding); at forward speed the residual is
    the numerical absorption of the upwind d/dx (magnitude information) and
    must be <= 0 at t' = 8 and over the whole record t' = 40 (no spurious
    energy creation, binding sign); (c) U = 0, nu = 0, free evolution of a
    surface hump: RK4 drift |E_dt - E_dt/2| / E0 at t' = 10 <= 1e-4 (binding)."""
    Fn, grid, hm = args
    from hydro.rankine.unsteady import TimeDomain
    from hydro.rankine.tables import memory_kernel, static_stiffness, RadiationForce
    h = _wj()
    U = Fn * np.sqrt(G0 * L)
    rows = []
    s, co, _ = _solver(h, U, grid, hm)
    td = TimeDomain(co, hull=h)
    dt = 0.05 / SC
    # TD: transfer function
    mot = _pulse(1, 0.01, 3.0 / SC, 0.5 / SC)
    r = td.run(mot, dt, int(60 / 0.05), energy=False)
    F, x = r["F_D"][:, 1], r["x"][:, 1]
    nfft = 1 << int(np.ceil(np.log2(len(F) * 4)))
    Xf, Ff = np.fft.rfft(x, nfft), np.fft.rfft(F, nfft)
    wv = 2 * np.pi * np.fft.rfftfreq(nfft, dt)
    errs, notes = [], []
    for wp in (2.0, 3.0, 4.5):
        kk = int(np.argmin(np.abs(wv - wp * SC)))
        Ttd = -Ff[kk] / Xf[kk]
        Tfd = co.radiation(wv[kk])["T"][1, 1]
        errs.append(abs(Ttd - Tfd) / abs(Tfd))
        notes.append(f"w' {wv[kk] / SC:.3f}: TD {Ttd:.4e} FD {Tfd:.4e}")
    rows.append(_row("TD", f"time domain (RK4) = frequency domain, T33, Wigley I Fn {Fn} (small grid NF {s.NF})",
                     float(max(errs)), 0.0, "same LTI system", TOL_TD, max(errs) <= TOL_TD, note="; ".join(notes)))
    # CU: Cummins reconstruction through the table pipeline and the DOCUMENTED plant law (stage note 10.8.5):
    # F = -A_inf a - B_inf v - int K v - C0 x (RadiationForce(stiffness=True)), D-form and OT-form, each with its
    # own K, B_inf, C0 (exact kernels: no window, so the check is the pipeline itself)
    inst = co.instantaneous()
    C0, C0ot = static_stiffness(co)
    wp = np.concatenate([np.exp(np.linspace(np.log(0.01), np.log(0.6), 30)), np.linspace(0.6, 1.2, 60)[1:],
                         np.exp(np.linspace(np.log(1.2), np.log(150), 120))[1:]])
    w = wp * SC
    BD, BO = [], []
    for x_ in w:
        r_ = co.radiation(x_)
        BD.append(r_["B"])
        BO.append(r_["T_ot"].imag / x_)
    BD, BO = np.array(BD), np.array(BO)
    tk = np.arange(0, 40 + 1e-9, 0.05) / SC
    K, tail = memory_kernel(w, BD, inst["B"], tk)
    Kot, tail_ot = memory_kernel(w, BO, inst["B_ot"], tk)
    ecu = {"D": [], "OT": []}
    for mode in (1, 2):
        mot = _pulse(mode, 0.01, 5.0 / SC, 1.0 / SC)
        rr = td.run(mot, dt, int(40 / 0.05), energy=False)
        for form, Kx, Bi, Cx, Fkey in (("D", K, inst["B"], C0, "F_D"), ("OT", Kot, inst["B_ot"], C0ot, "F_OT")):
            rf = RadiationForce(dict(A_inf=inst["A"], B_inf=Bi, C0=Cx, K=Kx, t=tk), dt, stiffness=True)
            Fc = np.array([rf(mot(t)[2], mot(t)[1], mot(t)[0]) for t in rr["t"]])
            ecu[form].append(float(np.sqrt(np.mean((Fc - rr[Fkey]) ** 2, axis=0)).max()
                                   / np.sqrt(np.mean(rr[Fkey][:, mode] ** 2))))
    for form in ("D", "OT"):
        rows.append(_row("CU", f"Cummins reconstruction through the documented plant law (RadiationForce with C0; K from "
                         f"B(omega)), {form}-form, vs the time-domain {form} force, heave and pitch pulses",
                         float(max(ecu[form])), 0.0, "same LTI system", TOL_TD, max(ecu[form]) <= TOL_TD,
                         note=f"kernel tail {(tail if form == 'D' else tail_ot):.1e}; per mode {ecu[form]}"))
    s.release()
    _free()
    # EN
    for Fen in (0.0, Fn):
        Ue = Fen * np.sqrt(G0 * L)
        g2 = dict(grid, down=grid["up"], b_dn=grid["b_up"]) if Fen == 0 else grid
        s, co, _ = _solver(h, Ue, g2, hm, sgn=1.0)
        td = TimeDomain(co, hull=h)
        rr = td.run(_pulse(1, 0.01, 5.0 / SC, 1.0 / SC), dt, int(40 / 0.05))
        tt = rr["t"]
        P = rr["P_body"] + rr["P_wl"] + rr["P_beach"]
        I = np.concatenate([[0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(tt))])
        Wb = np.concatenate([[0], np.cumsum(0.5 * (np.abs(rr["P_body"][1:]) + np.abs(rr["P_body"][:-1])) * np.diff(tt))])
        res = (rr["H"] - rr["H"][0] - I) / Wb[-1]
        k8 = int(np.argmin(np.abs(tt * SC - 8.0)))
        parts = (f"whole record t' = 40: {res[-1]:+.3e}; parts body {np.trapezoid(rr['P_body'], tt):.3e}, waterline "
                 f"{np.trapezoid(rr['P_wl'], tt):.3e}, beach {np.trapezoid(rr['P_beach'], tt):.3e}")
        if Fen == 0:
            rows.append(_row("EN", "energy balance, U = 0, heave pulse: |residual| / int|P_body| at t' = 8 (end of forcing)",
                             float(abs(res[k8])), 0.0, "energy identity (stage note 10.2)", TOL_EN,
                             abs(res[k8]) <= TOL_EN, note=parts))
        else:
            rows.append(_row("EN", f"energy balance Fn {Fen}: residual / int|P_body| at t' = 8 and t' = 40 (magnitude = "
                             "numerical absorption by the upwind d/dx; information)", float(res[k8]), 0.0,
                             "pseudo-energy identity (stage note 10.2)", "info", True, note=parts))
            rows.append(_row("EN", f"energy balance Fn {Fen}: residual <= 0 at t' = 8 and over the whole record (no "
                             "spurious energy creation)", float(max(res[k8], res[-1])), 0.0, "sign", "<= 0",
                             max(res[k8], res[-1]) <= 0.0))
        s.release()
    _free()
    # EN-c: RK4 drift without dissipation
    s, co, G = _solver(h, 0.0, dict(grid, nu_max=0.0, down=grid["up"], b_dn=grid["b_up"]), hm, sgn=1.0)
    td = TimeDomain(co, hull=h)
    X, Y = G["fs"].c[:, 0], G["fs"].c[:, 1]
    z0 = 0.01 * np.exp(-((X - 0.9) ** 2 + Y ** 2) / 0.04)
    E = []
    for dtp in (0.05, 0.025):
        rr = td.run(None, dtp / SC, int(round(10 / dtp)), zeta0=z0)
        E.append(rr["H"])
    drift = float(abs(E[0][-1] - E[1][-1]) / E[0][0])
    s.release()
    _free()
    rows.append(_row("EN", "RK4 drift without dissipation (U = 0, nu = 0, free evolution): |E_dt - E_dt/2| / E0 at t' = 10",
                     drift, 0.0, "integrator error", 1e-4, drift <= 1e-4,
                     note=f"E(t' = 10)/E0 - 1 = {E[0][-1] / E[0][0] - 1:+.3e} (the panel-quadrature energy is not the "
                          f"exact invariant of the collocation system)"))
    return rows


def froude_similarity(args):
    """FS (M3, binding 1e-4): the same Wigley I at L = 1 and L = 1.5 (grid and
    meshes scaled), Fn 0.3, omega' 2.212: non-dimensional T (all 9 entries) and
    the excitation at lambda/L = 1.2."""
    grid, hm = args
    from hydro.rankine.hull import wigley_journee
    out = []
    for Ls in (1.0, 1.5):
        h = wigley_journee(L=Ls, B=0.1 * Ls, T=0.0625 * Ls)
        V = h.hydrostatics()["V"]
        U = 0.3 * np.sqrt(G0 * Ls)
        s, co, _ = _solver(h, U, grid, hm, ref=(0.0, ZG * Ls))
        w = 2.212 * np.sqrt(G0 / Ls)
        T = co.radiation(w)["T"]
        p = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 2]])
        Tn = T / (RHO * V * Ls ** p * w ** 2)
        k = 2 * np.pi / (1.2 * Ls)
        X = co.excitation(np.sqrt(G0 * k), np.pi)["X"] / (RHO * G0 * V / Ls ** np.array([1, 1, 0]))
        out.append(np.concatenate([Tn.ravel(), X]))
        s.release()
    _free()
    e = float(np.abs(out[0] - out[1]).max() / np.abs(out[0]).max())
    return [_row("FS", "Froude similarity L = 1 vs 1.5 (Fn 0.3, omega' 2.212; excitation lambda/L 1.2)", e, 0.0, "exact",
                 TOL_FS, e <= TOL_FS)]


# =====================================================================
# convergence, domain (M5)
def _coef_series(h, Fn, grid, hm, wps):
    s, co, _ = _solver(h, Fn * np.sqrt(G0 * L), grid, hm)
    V = _V(h)
    out = np.array([np.concatenate([_nd(co.radiation(wp * SC)["T"], wp * SC, V)[q][[1, 2, 1, 2], [1, 2, 2, 1]]
                                    for q in (0, 1)]) for wp in wps])
    NF = s.NF
    s.release()
    _free()
    return out, NF


def observed_order(v1, v2, v3, h1, h2, h3):
    """Observed order p from three members with ARBITRARY refinement ratios
    (B5 review): solve (h1^p - h2^p)/(h2^p - h3^p) = (v2 - v1)/(v3 - v2) by
    bisection on p in [0.05, 12] (NaN if the changes are not monotone or no
    root); Richardson value v_inf = v3 + (v3 - v2)/((h2/h3)^p - 1)."""
    d1, d2 = v2 - v1, v3 - v2
    if d2 == 0 or d1 / d2 <= 0:
        return float("nan"), float("nan")
    r = d1 / d2
    f = lambda p: (h1 ** p - h2 ** p) / (h2 ** p - h3 ** p) - r
    lo, hi = 0.05, 12.0
    if f(lo) * f(hi) > 0:
        return float("nan"), float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    p = 0.5 * (lo + hi)
    return p, v3 + d2 / ((h2 / h3) ** p - 1)


def grid_series(args):
    """GS (M5, binding 2 %): free-surface series M = 40/60/80 (hull 60 x 12) and
    hull series 40x8 / 60x12 / 90x18 (M = 60) at Fn 0.2, Journee's seven
    frequencies: max change of A33', B33', A55', B55' between the two finest
    members relative to max(|value|, 0.1 max over omega); A35', A53', B35',
    B53' as information. B5 review: observed orders from the ACTUAL refinement
    ratios (FS series 1.5 and 4/3; B5 used log 1.5 for both) and the
    Richardson-extrapolated error of the PROD member (M = 60 / hull 60 x 12),
    information rows."""
    Fn, = args
    h = _wj()
    rows = []
    for nm, members, hs in (("free-surface M 40/60/80", [(dict(GRID_BASE, M=M), (60, 12)) for M in (40, 60, 80)],
                             (1 / 40, 1 / 60, 1 / 80)),
                            ("hull 40x8/60x12/90x18", [(dict(GRID_BASE, M=60), hm) for hm in HULL_SERIES],
                             (1 / 40, 1 / 60, 1 / 90))):
        vals, nfs = [], []
        for grid, hm in members:
            v, nf = _coef_series(h, Fn, grid, hm, W_J)
            vals.append(v)
            nfs.append(nf)
        diag = [0, 1, 4, 5]          # A33 A55 B33 B55
        den = np.maximum(np.abs(vals[2]), 0.1 * np.abs(vals[2]).max(axis=0))
        d1 = np.abs(vals[1] - vals[0]) / den
        d2 = np.abs(vals[2] - vals[1]) / den
        e = float(d2[:, diag].max())
        P = np.full(vals[0].shape, np.nan)
        R = np.full(vals[0].shape, np.nan)
        for a_ in range(vals[0].shape[0]):
            for b_ in range(vals[0].shape[1]):
                p, vinf = observed_order(vals[0][a_, b_], vals[1][a_, b_], vals[2][a_, b_], *hs)
                P[a_, b_] = p
                if np.isfinite(vinf):
                    R[a_, b_] = abs(vals[1][a_, b_] - vinf) / den[a_, b_]
        rows.append(_row("GS", f"grid series ({nm}), Fn {Fn}: change of A33', A55', B33', B55' between the two finest "
                         f"(NF {nfs})", e, 0.0, "own", TOL_GS, e <= TOL_GS,
                         note=f"first change {d1[:, diag].max():.2e}; couplings (35, 53) finest change "
                              f"{d2[:, [2, 3, 6, 7]].max():.2e}"))
        rows.append(_row("GSo", f"grid series ({nm}): observed orders (actual refinement ratios) and Richardson error "
                         f"of the PROD member, diagonal A', B' (information)",
                         float(np.nanmax(R[:, diag])) if np.isfinite(R[:, diag]).any() else float("nan"), 0.0,
                         "Richardson extrapolation with the observed order", "info", True,
                         note=f"median order {np.nanmedian(P[:, diag]):.2f} (finite {int(np.isfinite(P[:, diag]).sum())}"
                              f"/{P[:, diag].size}); Richardson error at PROD median {np.nanmedian(R[:, diag]):.2e}, "
                              f"max {np.nanmax(R[:, diag]) if np.isfinite(R[:, diag]).any() else float('nan'):.2e}; "
                              f"couplings max {np.nanmax(R[:, [2, 3, 6, 7]]) if np.isfinite(R[:, [2, 3, 6, 7]]).any() else float('nan'):.2e}"))
    return rows


def domain_sensitivity(args):
    """DOM (M5, binding 2 %): beach strength x2 and x0.5, lateral extent x1.5,
    downstream x1.5, upstream x1.5, beach lengths x1.5 (COARSE, Fn 0.2, omega'
    1.659, 2.765, 4.424): max relative change of A33', B33', A55', B55'."""
    Fn, = args
    h = _wj()
    base = dict(GRID_BASE, M=COARSE["M"])
    wps = (1.659, 2.765, 4.424)
    ref, _ = _coef_series(h, Fn, base, COARSE["hull"], wps)
    rows = []
    for name, d in (("nu_max x2", dict(nu_max=3.0)), ("nu_max x0.5", dict(nu_max=0.75)),
                    ("lateral x1.5", dict(lat=4.5, b_lat=2.25)), ("downstream x1.5", dict(down=6.0, b_dn=3.0)),
                    ("upstream x1.5", dict(up=2.25, b_up=1.125)), ("beach lengths x1.5", dict(b_up=1.125, b_dn=3.0, b_lat=2.25))):
        v, nf = _coef_series(h, Fn, dict(base, **d), COARSE["hull"], wps)
        e = float(np.max(np.abs(v[:, [0, 1, 4, 5]] / ref[:, [0, 1, 4, 5]] - 1)))
        rows.append(_row("DOM", f"domain/beach: {name} (NF {nf}), Fn {Fn}", e, 0.0, "own", TOL_DOM, e <= TOL_DOM))
    return rows


# =====================================================================
# steady limit and static stiffness (information)
def steady_limit(args):
    """ST0 (information): the omega = 0 solution of the B5 system (parabolic
    Wigley, Fn 0.316, M = 40 and 60) vs the B4 steady solver (PiercingNK,
    COARSE) -- C_near, C_p, Fz, My; and the B5 near field vs its own wave cut.
    C0 (information): static stiffness of the radiation problem (omega -> 0)
    vs the finite-difference attitude derivative of the B5 steady force
    (heave dz = 1e-3 L, pitch 1e-3 rad), Wigley I Fn 0.3."""
    from hydro.rankine.hull import wigley
    from hydro.rankine.unsteady import steady_forces
    from hydro.rankine.piercing import PiercingNK, cut_from_samples
    from studies.rankine_hull_verify import wigley_wetted_surface
    from hydro.rankine.tables import static_stiffness
    h = wigley()
    S = wigley_wetted_surface()
    Fn = 0.316
    U = Fn * np.sqrt(G0 * L)
    q = 0.5 * RHO * U * U * S
    rows = []
    b4 = PiercingNK(h, U, G0, rho=RHO, n_per_lam=30, m_body=60, hull_mesh=(60, 12))
    b4.solve()
    F4 = b4.forces(phi_db=b4.double_body())
    b4.release()
    notes = [f"B4 COARSE: C_near {F4['R_near'] / q * 1e3:.4f}e-3, C_p {F4['R_corr'] / q * 1e3:.4f}e-3, Fz/q "
             f"{F4['Fz'] / q:.5f}, My/qL {F4['My'] / q:.3e}"]
    lam0 = 2 * np.pi * U * U / G0
    for M in (40, 60):
        grid = dict(GRID_BASE, M=M, u_dn=1.0, u_lat=0.5)
        s, co, G = _solver(h, U, grid, (60, 12), ref=(0.0, 0.0))
        sol = s.solve_steady((U * s.body.n[:, 0])[:, None])
        Fr = steady_forces(s, sol["phi_B"][:, 0], h)
        phiF = np.real(sol["phi_F"][:, 0])
        nx, ny = G["shape"]
        zeta = U / G0 * (s.D @ phiF)
        zx = s.D @ zeta
        ic = int(np.argmin(np.abs(G["Xc"][:, 0] - (h.waterline_ends()[0] - lam0))))
        ks = np.arange(ny) * nx + ic
        Rc, _ = cut_from_samples(G["fs"].c[ks, 1], zeta[ks], zx[ks], G["ylines"][-1], U, U * U / G0, RHO,
                                 0.5 * np.pi / G["dy0"])
        notes.append(f"B5 M {M} (NF {s.NF}): C_near {Fr['R_near'] / q * 1e3:.4f}e-3, C_p {Fr['R_corr'] / q * 1e3:.4f}e-3, "
                     f"Fz/q {Fr['Fz'] / q:.5f}, My/qL {Fr['My'] / q:.3e}, C_cut(1 lambda0) {Rc / q * 1e3:.4f}e-3 "
                     f"(near/cut {Fr['R_near'] / Rc - 1:+.1%})")
        s.release()
    _free()
    rows.append(_row("ST0", "omega = 0 limit of the B5 system vs the B4 steady solver (parabolic Wigley Fn 0.316)",
                     float(Fr["R_near"] / F4["R_near"] - 1), 0.0, "B4 PiercingNK (COARSE)", "info", True,
                     note="; ".join(notes)))
    # C0 vs steady derivative (Wigley I, Fn 0.3)
    hj = _wj()
    U = 0.3 * np.sqrt(G0 * L)
    grid = dict(GRID_BASE, M=40)
    s, co, _ = _solver(hj, U, grid, COARSE["hull"])
    C0, C0ot = static_stiffness(co)
    s.release()
    F = {}
    for key, (sk, tr) in (("0", (0.0, 0.0)), ("dz", (-1e-3, 0.0)), ("dth", (0.0, -1e-3))):
        s, co, _ = _solver(hj, U, grid, COARSE["hull"], sink=sk, trim=tr, ref=(0.0, ZG))
        sol = s.solve_steady((U * s.body.n[:, 0])[:, None])
        F[key] = steady_forces(s, sol["phi_B"][:, 0], hj, sink=sk, trim=tr, cg=(0.0, ZG))
        s.release()
    _free()
    # heave up by dz: sink = -dz; pitch x5 = +dth (bow down) <-> hull.py trim = -dth
    dF3_dz = -(F["dz"]["Fz"] - F["0"]["Fz"]) / 1e-3
    dM5_dz = -(F["dz"]["My"] - F["0"]["My"]) / 1e-3
    dF3_dth = -(F["dth"]["Fz"] - F["0"]["Fz"]) / 1e-3
    dM5_dth = -(F["dth"]["My"] - F["0"]["My"]) / 1e-3
    Ks = np.array([[dF3_dz, dF3_dth], [dM5_dz, dM5_dth]])       # -dF_steady/dq (heave, pitch)
    rows.append(_row("C0", "static stiffness C0 (radiation, omega -> 0) vs -dF_steady/dq of the B5 steady force, "
                     "Wigley I Fn 0.3 (values)", float(C0[2, 2] / dM5_dth - 1) if dM5_dth else float("nan"), 0.0,
                     "information", "info", True, note=f"C0 (33, 35, 53, 55) D-form {C0[1, 1]:.3e} {C0[1, 2]:.3e} "
                     f"{C0[2, 1]:.3e} {C0[2, 2]:.3e}; OT {C0ot[1, 1]:.3e} {C0ot[1, 2]:.3e} {C0ot[2, 1]:.3e} "
                     f"{C0ot[2, 2]:.3e}; -dF/dq: (33) {dF3_dz:.3e} (35) {dF3_dth:.3e} (53) {dM5_dz:.3e} (55) "
                     f"{dM5_dth:.3e}"))
    # C0B (binding, B5 review): consistency of each form's C0 with -dF_steady/dq, relative to the hydrostatic
    # stiffness about the same point (diagonal: C_jj; cross: sqrt(C33 C55))
    hs = hj.hydrostatics()
    zb = hs["centroid"][2]
    C33 = RHO * G0 * hs["Aw"]
    C55 = RHO * G0 * (hs["IL"] + hs["Aw"] * hs["xf"] ** 2 + hs["V"] * (zb - ZG))
    scl = np.array([[C33, np.sqrt(C33 * C55)], [np.sqrt(C33 * C55), C55]])
    for form, Cx in (("D", C0), ("OT", C0ot)):
        dev = np.abs(Cx[np.ix_([1, 2], [1, 2])] - Ks) / scl
        rows.append(_row("C0B", f"consistency of the {form}-form static stiffness C0 with -dF_steady/dq (Wigley I Fn 0.3, "
                         "heave-pitch), relative to the hydrostatic stiffness", float(dev.max()), 0.0,
                         "plant law of stage note 10.8.5 (the slope of F_steady is NOT used dynamically)", 0.05,
                         dev.max() <= 0.05, note=f"(33) {dev[0, 0]:.3f} (35) {dev[0, 1]:.3f} (53) {dev[1, 0]:.3f} "
                         f"(55) {dev[1, 1]:.3f}; C33 {C33:.4f}, C55 {C55:.5f}"))
    return rows


# =====================================================================
# validation (Journee 1992, secondary transcriptions)
def _load(name):
    """Journee data CSV (comment lines '#', one header line, then numbers)."""
    lines = [ln for ln in open(os.path.join(DATA, name), encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    return np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]])


def _vrows(tag, qname, xs, calc, exp, src, unit_note="", binding=True):
    floor = V_FLOOR * np.abs(exp).max()
    tol = np.maximum(TOL_V_REL * np.abs(exp), floor)
    ok = np.abs(calc - exp) <= tol
    rows = []
    for x, c, e, t, o in zip(xs, calc, exp, tol, ok):
        rows.append(_row(tag, f"{qname} at {x:.3f}{unit_note}", float(c), float(e), src,
                         f"max(20 %, {floor:.4g})" if binding else "info (selection data)", bool(o) or not binding,
                         note=f"ratio {c / e if e else float('nan'):.3f}, diff {c - e:+.4f} (tol {t:.4f})"
                              + ("" if binding else f"; would {'pass' if o else 'FAIL'} the rule")))
    return rows


def _rao(co, s, V, hull, w0, beta=np.pi, form="D"):
    """Heave-pitch RAO about G (2 DOF; surge restrained, as in a towed model):
    (-w_e^2 M + T(w_e) + C) xi = X."""
    key = (round(w0, 12), beta)
    cache = co.__dict__.setdefault("_rao_cache", {})
    if key not in cache:
        cache[key] = co.rad_exc(w0, beta)
    r, ex = cache[key]
    we = ex["omega_e"]
    T = r["T"] if form == "D" else r["T_ot"]
    hs = hull.hydrostatics()
    xg, zg = co.ref
    Aw, xf, IL = hs["Aw"], hs["xf"], hs["IL"]
    zb = hs["centroid"][2]
    C33 = RHO * G0 * Aw
    C35 = -RHO * G0 * Aw * (xf - xg)
    C55 = RHO * G0 * (IL + Aw * (xf - xg) ** 2 + V * (zb - zg))
    m = RHO * V
    Mm = np.array([[m, 0.0], [0.0, m * (KYY * hull.L) ** 2]])
    Cm = np.array([[C33, C35], [C35, C55]])
    T2 = T[np.ix_([1, 2], [1, 2])]
    X = ex["X"][[1, 2]] if form == "D" else (ex["X_fk"] + ex["X_d_ot"])[[1, 2]]
    xi = np.linalg.solve(-we ** 2 * Mm + T2 + Cm, X)
    return np.abs(xi[0]), np.abs(xi[1]) * hull.L / (2 * np.pi), ex


def validation(args):
    """V (validation; registered tolerances max(20 %, 10 % of the series
    maximum) per point): Wigley I against Journee (1992) as transcribed from
    the vector figures of J. Mar. Sci. Eng. 9:87 (Fn 0.2: A33', B33', A55',
    B55', heave/pitch excitation, heave/pitch RAOs) and Kara (2022) (Fn 0.3:
    A33', B33'); computed at the transcribed abscissae (PROD grid, direct-
    pressure force = primary); the Ogilvie-Tuck variant and the RAO sensitivity
    to the assumed KG (+-0.05 m on the 3 m model) as information; STF-type
    strip-theory consistency (3D zero-speed coefficients + Salvesen-Tuck-
    Faltinsen speed terms) as information."""
    cfg = args[0] if args else PROD
    grid = dict(GRID_BASE, M=cfg["M"])
    hm = cfg["hull"]
    h = _wj()
    V = _V(h)
    rows = []
    src2 = "Journee 1992 via J. Mar. Sci. Eng. 9:87 (2021) figs 12-16 (secondary transcription)"
    src3 = "Journee 1992 via Kara (2022) fig 9 (secondary transcription)"
    for Fn in (0.2, 0.3):
        s, co, _ = _solver(h, Fn * np.sqrt(G0 * L), grid, hm)
        memo = {}

        def rad(w):
            kk = round(w, 12)
            if kk not in memo:
                memo[kk] = co.radiation(w)
            return memo[kk]
        tag = "V1" if Fn == 0.2 else "V2"
        fsuf = "020" if Fn == 0.2 else "030"
        names = ("A33", "B33", "A55", "B55") if Fn == 0.2 else ("A33", "B33")
        for nm in names:
            d = _load(f"journee1992_wigley1_fn{fsuf}_{nm}.csv")
            i, j = {"33": (1, 1), "55": (2, 2)}[nm[1:]]
            q = 0 if nm[0] == "A" else 1
            calc, calc_ot = [], []
            for wp in d[:, 0]:
                r = rad(wp * SC)
                calc.append(_nd(r["T"], wp * SC, V)[q][i, j])
                calc_ot.append(_nd(r["T_ot"], wp * SC, V)[q][i, j])
            # B5 review: the Fn 0.3 A33' series was used to choose the primary force form (exploration B5-E2):
            # selection data, not validation (both forms, information); every other series binding for BOTH forms
            sel = Fn == 0.3 and nm == "A33"
            for form, cv, tg in (("D", calc, tag), ("OT", calc_ot, tag + "o")):
                rows += _vrows(tg, f"Fn {Fn} {nm}' ({form}-form) vs exp, omega'" + (" [used for force-form selection, "
                               "not validation]" if sel else ""), d[:, 0], np.array(cv), d[:, 1],
                               src2 if Fn == 0.2 else src3, binding=not sel)
        if Fn == 0.2:
            for nm, jj, scale in (("F3", 1, lambda X: abs(X) * L / (RHO * G0 * V)),
                                  ("F5", 2, lambda X: abs(X) / (RHO * G0 * V))):
                d = _load(f"journee1992_wigley1_fn020_{nm}.csv")
                calc, calc_ot = [], []
                for nu in d[:, 0]:
                    k = nu / L
                    key = (round(np.sqrt(G0 * k), 12), np.pi)
                    cache = co.__dict__.setdefault("_rao_cache", {})
                    if key not in cache:
                        cache[key] = co.rad_exc(np.sqrt(G0 * k), np.pi)
                    ex_ = cache[key][1]
                    calc.append(scale(ex_["X"][jj]))
                    calc_ot.append(scale(ex_["X_fk"][jj] + ex_["X_d_ot"][jj]))
                rows += _vrows("V3", f"Fn 0.2 {nm}' excitation (D-form) (head seas) vs exp, nu' = kL", d[:, 0],
                               np.array(calc), d[:, 1], src2)
                rows += _vrows("V3o", f"Fn 0.2 {nm}' excitation (OT-form) (head seas) vs exp, nu' = kL", d[:, 0],
                               np.array(calc_ot), d[:, 1], src2)
            for nm, jj in (("RAO3", 0), ("RAO5", 1)):
                d = _load(f"journee1992_wigley1_fn020_{nm}.csv")
                calc, calc_ot = [], []
                for lam in d[:, 0]:
                    k = 2 * np.pi / (lam * L)
                    w0 = np.sqrt(G0 * k)
                    calc.append(_rao(co, s, V, h, w0)[jj])
                    calc_ot.append(_rao(co, s, V, h, w0, form="OT")[jj])
                rows += _vrows("V4", f"Fn 0.2 {nm} (D-form; KG 0.170 m assumed) vs exp, lambda/L", d[:, 0],
                               np.array(calc), d[:, 1], src2)
                rows += _vrows("V4o", f"Fn 0.2 {nm} (OT-form; KG 0.170 m assumed) vs exp, lambda/L", d[:, 0],
                               np.array(calc_ot), d[:, 1], src2)
            # KG sensitivity (information): rebuild with z_G shifted
            for dKG in (-0.05, 0.05):
                zg2 = ZG + dKG / L_M
                s2, co2, _ = _solver(h, 0.2 * np.sqrt(G0 * L), grid, hm, ref=(0.0, zg2))
                d = _load("journee1992_wigley1_fn020_RAO5.csv")
                v = [_rao(co2, s2, V, h, np.sqrt(G0 * 2 * np.pi / (lam * L)))[1] for lam in d[:, 0]]
                rows.append(_row("V4", f"RAO5 sensitivity to KG {KG_M + dKG:.3f} m (information)", 0.0, 0.0,
                                 "information", "info", True, note=", ".join(f"{x:.4f}" for x in v)))
                s2.release()
        # STF consistency (information): zero-speed 3D coefficients + STF speed terms (fore-aft symmetric hull)
        s0, c0, _ = _solver(h, 0.0, dict(grid, down=grid["up"], b_dn=grid["b_up"], u_dn=0.0), hm, sgn=1.0)
        U = Fn * np.sqrt(G0 * L)
        notes = []
        for wp in (2.212, 3.318, 4.424):
            w = wp * SC
            A0, B0 = (lambda T: (-T.real / w ** 2, T.imag / w))(c0.radiation(w)["T"])
            A, B = (lambda T: (-T.real / w ** 2, T.imag / w))(rad(w)["T"])
            Ao, Bo = (lambda T: (-T.real / w ** 2, T.imag / w))(rad(w)["T_ot"])
            stf = dict(A35=-U * B0[1, 1] / w ** 2, A53=U * B0[1, 1] / w ** 2, B35=U * A0[1, 1], B53=-U * A0[1, 1],
                       A55=A0[2, 2] + U * U / w ** 2 * A0[1, 1], B55=B0[2, 2] + U * U / w ** 2 * B0[1, 1])
            got = dict(A35=(A[1, 2], Ao[1, 2]), A53=(A[2, 1], Ao[2, 1]), B35=(B[1, 2], Bo[1, 2]), B53=(B[2, 1], Bo[2, 1]),
                       A55=(A[2, 2], Ao[2, 2]), B55=(B[2, 2], Bo[2, 2]))
            notes.append(f"w' {wp}: " + ", ".join(f"{k} STF {stf[k] / V:.4f} D {got[k][0] / V:.4f} OT {got[k][1] / V:.4f}"
                                                   for k in stf))
        s0.release()
        rows.append(_row("STF", f"Fn {Fn}: STF speed terms on the 3D zero-speed coefficients vs 3D forward speed "
                         "(units rho V L^p; information)", 0.0, 0.0, "Salvesen, Tuck & Faltinsen 1970 (consistency)",
                         "info", True, note="; ".join(notes)))
        s.release()
        _free()
    return rows


# =====================================================================
# demo table (interface)
def _heave_pitch_statics(hull, ref, kyy=KYY, zg=None, xg=None):
    """Mass and hydrostatic matrices (heave, pitch) about the table reference
    point ref = (x_r, z_r) for the free-decay check CD: mass rho V with the CG at
    (x_G, z_G) (default: x_G = LCB, z_G = z_r), k_yy about the CG; hydrostatic
    C33 = rho g A_w, C35 = C53 = -rho g A_w (x_f - x_r), C55 = rho g [I_L +
    A_w (x_f - x_r)^2 + V (z_B - z_G)] (pitch bow down positive)."""
    hs = hull.hydrostatics()
    V = hs["V"]
    xr, zr = ref
    xg = hs["centroid"][0] if xg is None else xg
    zg = zr if zg is None else zg
    m = RHO * V
    dx = xg - xr
    Mm = np.array([[m, -m * dx], [-m * dx, m * ((kyy * hull.L) ** 2 + dx ** 2 + (zg - zr) ** 2)]])
    C33 = RHO * G0 * hs["Aw"]
    C35 = -RHO * G0 * hs["Aw"] * (hs["xf"] - xr)
    C55 = RHO * G0 * (hs["IL"] + hs["Aw"] * (hs["xf"] - xr) ** 2 + V * (hs["centroid"][2] - zg))
    return Mm, np.array([[C33, C35], [C35, C55]])


def _decay_rows(tab, hull, label, statics_kw=None):
    """CD (binding, B5 review): free heave-pitch decay of the DOCUMENTED plant law
    (tables.free_decay: (M + A_inf) a = -B_inf v - int K v ds - (C_hs + C0) x,
    the table's windowed kernels) at every node of `tab`, both force forms,
    x0 = (0.005 L, 0.02 rad), dt' = 0.02, horizon t' = 200 (about 150 s at the
    5.4 m plant hull): bounded energy, max E(t) <= 1.2 E(0), and no late growth,
    max E over the last quarter <= max(max E over the third quarter, 1e-8 E(0));
    amendment 1 (stage note 10.8.11): and a positive-definite symmetric part of
    C_hs + C0 (static stability; otherwise E is not a norm)."""
    from hydro.rankine.tables import interp_slice, free_decay
    Mm, Ch = _heave_pitch_statics(hull, tab.ref, **(statics_kw or {}))
    rows = []
    for form in ("D", "OT"):
        res, ok_all = [], True
        for iu, u in enumerate(tab.U):
            for i_s, sk in enumerate(tab.sink):
                for i_t, tr in enumerate(tab.trim):
                    sl = interp_slice(tab, u, sk, tr, form=form)
                    Cs = Ch + np.asarray(sl["C0"])[1:, 1:]
                    lam = float(np.linalg.eigvalsh(0.5 * (Cs + Cs.T)).min())
                    with np.errstate(over="ignore", invalid="ignore"):
                        out = free_decay(sl, Mm, Ch, np.array([0.005 * hull.L, 0.02]), 0.02 / SC, 200.0 / SC)
                    E = out["E"]
                    n = len(E)
                    e0 = E[0]
                    q3 = np.nanmax(E[n // 2: 3 * n // 4]) if n > 4 else np.nan
                    q4 = np.nanmax(E[3 * n // 4:]) if n > 4 else np.nan
                    # amendment 1 (stage note 10.8.11): static stability required -- with an indefinite stiffness the
                    # "energy" is not a norm and a diverging trajectory could satisfy the energy rules
                    ok = bool(lam > 0 and np.all(np.isfinite(E)) and E.max() <= 1.2 * e0 and q4 <= max(q3, 1e-8 * e0)
                              and len(E) == int(round(200.0 / 0.02)) + 1)
                    ok_all &= ok
                    res.append(f"Fn {u / SC:.2f} s {sk:.4f} t {tr:+.3f}: min eig sym(C_hs + C0) {lam:+.3e}, maxE/E0 "
                               f"{np.nanmax(E) / e0:.3f}, E(end)/E0 {E[-1] / e0:.2e}, q4/q3 {q4 / max(q3, 1e-300):.2f}"
                               f"{'' if ok else ' FAIL'}")
        rows.append(_row("CD", f"{label}: free heave-pitch decay of the plant law, {form}-form, all {len(res)} nodes, horizon "
                         "t' 200: bounded energy and no late growth", float(not ok_all), 0.0, "plant law (stage note 10.8.5)",
                         "all nodes pass", ok_all, note="; ".join(res)))
    return rows


def demo_table(args):
    """PV (interface): radiation table for the planing-type prismatic V body
    over (U, sinkage, trim) = (Fn 0.3, 0.5) x (0, 0.004 L) x (0, -0.01 rad), 40
    frequencies omega' in [0.05, 40] + kernels to t' = 30 -- INTERFACE DEMO
    ONLY: the free-surface operator is not verified for this flared geometry
    (stage note 10.8.1). Checks: save/load round trip bit-exact and sha256
    recorded (binding, exact), interpolation at the nodes reproduces the node
    values (binding 1e-12), mid-box slice computed directly vs interpolated
    (PVm, binding: A_inf 1 % of max |A_inf|, kernels rms 10 % of the direct
    kernel's rms, both forms), free decay of the plant law at every node (CD),
    kernel tail / passivity / validity band / kernel-window KK distortion
    (information)."""
    grid, hm, label = args
    from hydro.rankine.hull import prismatic_v
    from hydro.rankine.tables import (build_radiation_table, save_table, load_table, interp_slice, table_path,
                                      node_slice)
    h = prismatic_v()
    U = np.array([0.3, 0.5]) * np.sqrt(G0 * L)
    sinks = np.array([0.0, 0.004])
    trims = np.array([-0.01, 0.0])
    wp = np.concatenate([np.exp(np.linspace(np.log(0.05), np.log(1.5), 12)),
                         np.exp(np.linspace(np.log(1.5), np.log(40.0), 29))[1:]])
    t = np.arange(0, 30 + 1e-9, 0.1) / SC
    t0 = time.time()
    tab = build_radiation_table(h, U, sinks, trims, wp * SC, t, grid, hm, rho=RHO, g=G0, ref=(0.0, 0.0))
    tb = time.time() - t0
    code = hashlib.sha256(open(os.path.join(ROOT, "hydro", "rankine", "unsteady.py"), "rb").read()).hexdigest()[:12]
    path = table_path(OUT, h.name, code, label + datetime.datetime.now().strftime("_%Y%m%dT%H%M%S"))
    sha = save_table(tab, path)
    tab2 = load_table(path)
    same = all(np.array_equal(getattr(tab, k), getattr(tab2, k), equal_nan=np.issubdtype(np.asarray(getattr(tab, k)).dtype,
                                                                                   np.floating))
               for k in tab.ARRAYS if getattr(tab, k) is not None)
    ei = 0.0
    for iu, u in enumerate(U):
        for i_s, sk in enumerate(sinks):
            for i_t, tr in enumerate(trims):
                for form, kk in (("D", "K"), ("OT", "K_ot")):
                    sl = interp_slice(tab2, u, sk, tr, form=form)
                    ei = max(ei, float(np.abs(sl["A_inf"] - tab.A_inf[iu, i_s, i_t]).max()),
                             float(np.abs(sl["K"] - getattr(tab, kk)[iu, i_s, i_t]).max()))
    # PVm: mid-box node computed directly
    um, sm, tm = U.mean(), sinks.mean(), trims.mean()
    nd = node_slice(h, um, sm, tm, wp * SC, t, grid, hm, rho=RHO, g=G0, ref=(0.0, 0.0))
    pvm, pvm_notes = {}, []
    for form, kk in (("D", "K"), ("OT", "K_ot")):
        sl = interp_slice(tab2, um, sm, tm, form=form)
        ea = float(np.abs(sl["A_inf"] - nd["A_inf"]).max() / np.abs(nd["A_inf"]).max())
        ek = float(np.sqrt(np.mean((sl["K"] - nd[kk]) ** 2)) / np.sqrt(np.mean(nd[kk] ** 2)))
        pvm[form] = (ea, ek)
        pvm_notes.append(f"{form}: A_inf {ea:.2e}, K rms {ek:.3f}")
    rows = [_row("PV", f"demo table {os.path.relpath(path, ROOT)} ({tb:.0f} s; INTERFACE DEMO ONLY, free-surface operator "
                 "unverified for this flared geometry): save/load bit-exact", float(not same), 0.0, "exact", 0.0, same,
                 note=f"sha256 {sha}"),
            _row("PV", "demo table: interpolation reproduces the node values (both forms)", ei, 0.0, "exact", 1e-12,
                 ei <= 1e-12)]
    for form in ("D", "OT"):
        ea, ek = pvm[form]
        rows.append(_row("PVm", f"demo table: mid-box slice (Fn {um / SC:.2f}, sink {sm:.4f}, trim {tm:+.4f}) interpolated vs "
                         f"computed directly, {form}-form: A_inf (1 % of max) and kernel rms (10 %)", max(ea / 0.01, ek / 0.10),
                         1.0, "direct node_slice", "<= 1 (A_inf 1 %, K 10 %)", ea <= 0.01 and ek <= 0.10,
                         note="; ".join(pvm_notes)))
    rows.append(_row("PV", "demo table: kernel tail, passivity, validity band, kernel window (information)",
                     float(tab.meta["kernel_tail_max"]), 0.0, "information", "info", True,
                     note=f"passive D {tab.passive.ravel().tolist()}, OT {tab.passive_ot.ravel().tolist()}; min "
                          f"lambda_min(B_sym)/max abs lambda: D {np.round(tab.bsym_min.ravel(), 3).tolist()}, OT "
                          f"{np.round(tab.bsym_min_ot.ravel(), 3).tolist()}; validity band omega' per node "
                          f"{[(round(a / SC, 2), round(b / SC, 2)) for a, b in zip(tab.omega_lo.ravel(), tab.omega_hi.ravel())]}; "
                          f"kernel window start omega' {np.round(tab.omega_t.ravel() / SC, 2).tolist()}; KK distortion of the "
                          f"window in the band D {np.round(tab.taper_dA.ravel(), 4).tolist()}, OT "
                          f"{np.round(tab.taper_dA_ot.ravel(), 4).tolist()}"))
    rows += _decay_rows(tab, h, "demo table (prismatic V, interface demo)")
    return rows


def decay_wigley(args):
    """CD on the verified hull (B5 review; basis of the D-vs-OT recommendation):
    radiation table of the Wigley I at Fn 0.2 and 0.3 (zero sinkage and trim),
    same frequencies and kernel times as the demo table, grid as given; free
    heave-pitch decay of the plant law at both nodes and both forms (rules of
    _decay_rows; CG at (0, z_G), KG assumed as in V4)."""
    grid, hm = args
    from hydro.rankine.tables import build_radiation_table
    h = _wj()
    U = np.array([0.2, 0.3]) * np.sqrt(G0 * L)
    wp = np.concatenate([np.exp(np.linspace(np.log(0.05), np.log(1.5), 12)),
                         np.exp(np.linspace(np.log(1.5), np.log(40.0), 29))[1:]])
    t = np.arange(0, 30 + 1e-9, 0.1) / SC
    tab = build_radiation_table(h, U, [0.0], [0.0], wp * SC, t, grid, hm, rho=RHO, g=G0, ref=(0.0, ZG))
    rows = _decay_rows(tab, h, "Wigley I table (Fn 0.2, 0.3)", statics_kw=dict(xg=0.0, zg=ZG))
    rows.append(_row("CD", "Wigley I table: validity band, kernel window, KK distortion, passivity (information)", 0.0, 0.0,
                     "information", "info", True,
                     note=f"band omega' {[(round(a / SC, 2), round(b / SC, 2)) for a, b in zip(tab.omega_lo.ravel(), tab.omega_hi.ravel())]}; "
                          f"window start {np.round(tab.omega_t.ravel() / SC, 2).tolist()}; KK distortion D "
                          f"{np.round(tab.taper_dA.ravel(), 4).tolist()}, OT {np.round(tab.taper_dA_ot.ravel(), 4).tolist()}; "
                          f"bsym_min D {np.round(tab.bsym_min.ravel(), 3).tolist()}, OT {np.round(tab.bsym_min_ot.ravel(), 3).tolist()}"))
    return rows


# =====================================================================
def as_info(fn, tag_suffix="r"):
    """Run a check function as a refinement / information job: every row becomes information (tol 'info', ok True,
    check id + suffix); the original pass/fail is kept in the note."""
    def wrapped(args):
        out = []
        for r in fn(args):
            r = dict(r)
            r["note"] = (f"[would be {'PASS' if r['ok'] else 'FAIL'} vs tol {r['tol']}] " + (r["note"] or ""))
            r["tol"], r["ok"], r["check"] = "info", True, r["check"] + tag_suffix
            out.append(r)
        return out
    wrapped.__name__ = fn.__name__ + "_info"
    return wrapped


hk_tn_wigley_info = as_info(hk_tn_wigley)
tn_prismatic_info = as_info(tn_prismatic)


TIERS = {
    "quick": [
        ("KKu", kk_unit, (160,), None),
        ("DXM", dxm_check, ((20, 40, 80),), None),
        ("U0", u0_wigley, ((2.212, 3.318), dict(GRID_BASE, M=30)), None),
        ("U0h", u0_hemisphere, ((0.3, 1.0), (24, 16, 20)), None),
        ("HK", hk_tn_wigley, (0.3, (1.0,), (2.212,), dict(CHECK_GRID, M=30)), None),
        ("KK", kk_check, (0.3, 160, SMALL_GRID, SMALL_HULL), None),
        ("G0", inf_g0, (dict(GRID_BASE, M=30), (40, 8)), None),
        ("SP", spectrum, ((("wigleyJ", 0.3, True), ("wigleyJ", 0.0, False)), dict(SMALL_GRID, M=16), (16, 4)), None),
        ("TD", td_checks, (0.3, SMALL_GRID, SMALL_HULL), None),
        ("FS", froude_similarity, (SMALL_GRID, SMALL_HULL), None),
        ("V", validation, (COARSE,), None),
        ("PV", demo_table, (dict(GRID_BASE, M=20, u_dn=0.3, u_lat=0.2), (24, 6), "quick"), None),
    ],
    "full": [
        ("KKu", kk_unit, (400,), None),
        ("DXM", dxm_check, ((20, 40, 80),), None),
        ("U0", u0_wigley, ((1.659, 2.212, 3.318, 4.424), dict(GRID_BASE, M=40)), None),
        ("U0h", u0_hemisphere, ((0.3, 1.0, 2.0), (24, 16, 20)), None),
        ("U0c", u0_continuity, ((2.212, 3.318), dict(GRID_BASE, M=40)), None),
        ("U0cr", u0_continuity, ((2.212, 3.318), dict(GRID_BASE, M=60), False), None),
        ("U0cr2", u0_continuity, ((2.212, 3.318), dict(GRID_BASE, M=80), False), None),
        ("HK", hk_tn_wigley, (0.3, (1.0, 1.5), (2.212, 4.424), CHECK_GRID), None),
        ("HKr", hk_tn_wigley_info, (0.3, (1.0, 1.5), (2.212, 4.424), dict(CHECK_GRID, M=60)), None),
        ("TNs", tn_submerged, (0.3, (2.212, 4.424), dict(CHECK_GRID, u_dn=0.0, u_lat=0.0)), None),
        ("TNp", tn_prismatic, (0.4, (2.212, 4.424), CHECK_GRID, (40, 8)), None),
        ("TNpr", tn_prismatic_info, (0.4, (2.212, 4.424), dict(CHECK_GRID, M=60), (40, 8)), None),
        ("TNpr2", tn_prismatic_info, (0.4, (2.212, 4.424), dict(CHECK_GRID, M=60), (60, 12)), None),
        ("TNw", tn_prismatic, (0.4, (2.212, 4.424), CHECK_GRID, (40, 8), dict(D0=0.06), "TNw"), None),
        ("WE", we_sphere, (WE_GRID, WE_MESH, False), None),
        ("KK", kk_check, (0.3, 400, SMALL_GRID, SMALL_HULL), None),
        ("KK2", kk_check, (0.2, 400, SMALL_GRID, SMALL_HULL), None),
        ("G0", inf_g0, (dict(GRID_BASE, M=40), (40, 8)), None),
        ("SP", spectrum, ((("wigleyJ", 0.2, True), ("wigleyJ", 0.3, True), ("wigleyJ", 1.0, True),
                           ("prismV", 0.4, True), ("wigleyJ", 0.0, False)), SMALL_GRID, SMALL_HULL), None),
        ("TD", td_checks, (0.3, SMALL_GRID, SMALL_HULL), None),
        ("FS", froude_similarity, (SMALL_GRID, SMALL_HULL), None),
        ("GS", grid_series, (0.2,), None),
        ("DOM", domain_sensitivity, (0.2,), None),
        ("ST0", steady_limit, (), None),
        ("V", validation, (PROD,), None),
        ("CDw", decay_wigley, (dict(GRID_BASE, M=30, u_dn=0.3, u_lat=0.2), (40, 8)), None),
        ("PV", demo_table, (dict(GRID_BASE, M=30, u_dn=0.3, u_lat=0.2), (30, 6), "full"), None),
    ],
}


# ------------------------------------------------------------ registration (as rankine_hull_verify, keys b5:*)
def _code_hashes():
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
            out["b5:" + head.strip()] = dict(title=None, sha256=None)
            continue
        i = hits[0]
        j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ") or lines[k].startswith("### ")),
                 len(lines))
        sec = "\n".join(lines[i:j]).strip() + "\n"
        out["b5:" + head.strip()] = dict(title=lines[i].strip(), sha256=hashlib.sha256(sec.encode("utf-8")).hexdigest())
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
    consts = (f"GRID_BASE={GRID_BASE!r} PROD={PROD!r} COARSE={COARSE!r} FINE={FINE!r} CHECK={CHECK_GRID!r} "
              f"SMALL={SMALL_GRID!r} KG={KG_M} KYY={KYY} TOLS={(TOL_ID, TOL_KK, TOL_U0, TOL_HULME, TOL_G0, TOL_INF, TOL_TD, TOL_EN, TOL_FS, TOL_GS, TOL_DOM, TOL_V_REL, V_FLOOR)}")
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
    cur["b5:code_thresholds"] = h_thr
    cur["b5:tier_jobs"] = h_jobs
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
    hashes.update({"b5:code_thresholds": h_thr, "b5:tier_jobs": h_jobs})
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
    print(f"=== rankine_unsteady_verify tier={tier} start {start_utc}; registration record:", flush=True)
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
    rows.append(_row("T", f"tier wall time ({tier})", wall, 0.0, "acceptance T1", 600.0 if tier == "quick" else 3600.0,
                     wall <= (600.0 if tier == "quick" else 3600.0)))
    path = os.path.join(OUT, f"unsteady_verify_{tier}{'_' + '_'.join(only) if only else ''}.json")
    info_rows = [r for r in rows if isinstance(r["tol"], str) and r["tol"].startswith("info")]
    bind_rows = [r for r in rows if not (isinstance(r["tol"], str) and r["tol"].startswith("info"))]
    summary = dict(binding_pass=sum(r["ok"] for r in bind_rows), binding=len(bind_rows), info=len(info_rows),
                   failing=[f"{r['check']}: {r['quantity'][:110]}" for r in bind_rows if not r["ok"]])
    for form, tags in (("D", ("V1", "V2", "V3", "V4")), ("OT", ("V1o", "V2o", "V3o", "V4o"))):
        vr = [r for r in bind_rows if r["check"] in tags]
        summary[f"validation_{form}"] = f"{sum(r['ok'] for r in vr)}/{len(vr)}"
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
                  f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"=== code sha256 at start registered: {code_reg[-1] if code_reg else 'NO'}; files changed during the run: "
          f"{changed or 'none'}")
    print(f"\n=== rankine_unsteady_verify tier={tier}  wall {wall:.0f} s  -> {path}")
    for r in rows:
        v = r["value"]
        if isinstance(v, float):
            v = f"{v:.4g}"
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['check']:4s} {r['quantity']}\n"
              f"        value={v}  ref={r['ref']}  tol={r['tol']}  ({r['source']})"
              f"{'  ' + r['note'] if r['note'] else ''}  [{r.get('seconds', '')} s]")
    print(f"=== binding rows: {summary['binding_pass']}/{summary['binding']} passed; information rows: {summary['info']}; "
          f"validation points passed: D-form {summary['validation_D']}, OT-form {summary['validation_OT']}")
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
    else:
        run(a.tier, [x for x in a.only.split(",") if x] or None)
