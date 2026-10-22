#!/usr/bin/env python3
"""
Track B step B6: the DOUBLE-BODY (steady base flow) linearisation of the forward-speed problems
(hydro/rankine/doublebody.py) -- verification and validation tiers (acceptance learn/meta/ACCEPTANCE_2026-10-06.md,
T1: quick <= 10 min, full <= 1 h). Registration: stage note learn/meta/RANKINE_STAGE1_2026-10-06.md section 11.1
(written, and hashed with --register, BEFORE the tiers ran); derivations 11.2; exploration runs 11.3; results 11.4.

    .venv/Scripts/python.exe studies/rankine_db_verify.py --tier quick [--only TN,WE]
    .venv/Scripts/python.exe studies/rankine_db_verify.py --register "note"

Row families (stage note 11.1): MT/SBP/TK m-terms and the discrete Tuck identity (M2), DXD manufactured test of the
double-body free-surface operators (M2), FSR exact discrete reciprocity of the free-surface operator (M2/M3), TN/TNp/
TNw classical Timman-Newman (= check (b) of B5: the waterline term vanishes in this linearisation; M3), HK Haskind with
the incident-wave/base-flow free-surface term (M3), TNs submerged body, WE submerged sphere vs Wu & Eatock Taylor
(1988), KK/KKu, INF/G0, SP spectrum, TD/CU/EN time domain, FS similarity, GS/GSp/DOM convergence (M5), U0/U0c low-speed
limit (M4), C0B static stiffness vs steady slope, SF steady Wigley (fixed and free) vs Kajitani et al. 1983 (validation,
B4 rules), V Journee (1992) validation (B5 rules; D and OT forms coincide), CD free decay of the plant law, PV first
planing-type table (interface; speed interpolation). Binding rows carry a registered tolerance; information rows
("info") are counted separately.

B6 review (2026-10-07, stage note section 12; registration 12.1 BEFORE the re-run): the Green-identity rows SBP, TK, FSR,
TN, TNp, TNw, TNs and HK are relabelled IMPLEMENTATION rows ("impl <tol>"): with the adjoint constructions they hold for
any hull operator / base flow (they measure the panel representation's Green-identity error), so they are reported
with their tolerances but counted separately, not as M3 passes. New binding rows: DXa (A, K, K 1 against the EXACT
double-body flow of a half spheroid), DXu (A, K at the grid non-smoothness lines with the exact uniform stream), CS
(frozen base-pressure stiffness C_s two ways), C0B (C0 + C_s vs the steady slope, relative to the dynamic stiffness),
CD/CDw (corrected static criterion: eigenvalues of M^-1 (C_hs + C0 + C_s), positive-norm trajectory decay), CDs (static
verdict on a grid series of the planing nodes), SK (double-body steady vs the exact Kelvin solve, deep spheroid,
wavelength-scaled domain), AT (attitude grid: no dry stations, bounded base flow, continuity of the base-pressure
force); information: DXb, HFi, KCN, SKd, SPr.
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
from studies.rankine_unsteady_verify import (G0, RHO, L, SC, WJ, ZG, KYY, W_J, GRID_BASE, PROD, COARSE,   # noqa: E402
                                             CHECK_GRID, SMALL_GRID, SMALL_HULL, HULL_SERIES, TOL_ID, TOL_KK, TOL_U0,
                                             TOL_G0, TOL_INF, TOL_TD, TOL_EN, TOL_FS, TOL_GS, TOL_DOM, TOL_V_REL,
                                             V_FLOOR, _nd, _wj, _V, _load, _vrows, _load_we, _pulse, _mms_field,
                                             observed_order, sphere_db_mterms, kk_unit, _heave_pitch_statics,
                                             WE_GRID, WE_R)
from studies.rankine_db_kk import kk_errors_band  # noqa: E402

OUT = os.path.join(ROOT, "studies", "_cache", "rankine")
REG_LOG = os.path.join(OUT, "registration_log.json")
STAGE_NOTE = os.path.join(ROOT, "learn", "meta", "RANKINE_STAGE1_2026-10-06.md")
REG_SECTIONS = ("### 11.1", "### 12.1")
CODE_FILES = ("hydro/rankine/*.py", "studies/rankine_db_verify.py", "studies/rankine_db_kk.py",
              "studies/rankine_unsteady_verify.py", "studies/rankine_hull_verify.py")
TOL_SBP = 1e-12        # exact constructions (round-off)
TOL_FSR = 1e-10
TOL_C0B = 0.05         # C0B tolerance (value of B5 10.8.5), since the B6 review relative to the dynamic stiffness
TOL_AT_W = 1.5         # attitude grid: base-flow speed bound |W|/U (B6 review, 12.1)
TOL_AT_LIN = 0.10      # attitude grid: deviation of the base-pressure force change from its linear prediction
TOL_SK = 0.03          # double-body steady vs exact Kelvin (deep spheroid; 12.1)


def _impl(tol):
    """Tolerance label of an IMPLEMENTATION / Green-identity row (B6 review, finding B1; stage note 12.1)."""
    return f"impl {tol}"
FN_FIX = (0.25, 0.267, 0.289, 0.316, 0.354, 0.408)
FN_FREE = (0.289, 0.316, 0.354)


def _free():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _db(hull, U, grid, hm, sink=0.0, trim=0.0, sgn=None, g=G0, ref=None, kinematic="cons", mterms="sbp"):
    from hydro.rankine.doublebody import build_db_solver
    ref = (0.0, ZG) if ref is None else ref
    s, co, G, db = build_db_solver(hull, U, sink, trim, grid, hm, g=g, rho=RHO, ref=ref, kinematic=kinematic,
                                   mterms=mterms, sgn=sgn)
    return s, co, G


def _prism(kind):
    from hydro.rankine.hull import prismatic_v
    return prismatic_v() if kind == "flared" else prismatic_v(D0=0.06)


# =====================================================================
# m-terms, Tuck, D = OT (M2)
def mterm_checks(args):
    """MT (M2, binding): pointwise m-terms (formula of 11.2 item 6 with surface_gradient2) on a sphere in unbounded
    flow (no lid) vs the analytic sphere flow at mesh 64 x 32: max error <= 1 % of max |m_1|, observed order (32x16 ->
    64x32) >= 1.5. MT-sbp (information): the production (finite-volume adjoint) m-terms vs the analytic ones.
    SBP (M2, binding 1e-12): discrete Tuck identity sum_p a_p (n_j G f + m_j f)_p = 0 with the production m-terms for
    test functions on the Wigley I, the flared and the wall-sided prismatic bodies (60 x 12) and the sphere."""
    from hydro.rankine import panels as P
    from hydro.rankine.panels import Panels
    from hydro.rankine.doublebody import DoubleBodyFlow
    r, U = 0.1, 1.0
    errs, notes, sbp_notes = {}, [], []
    for mesh in ((32, 16), (64, 32)):
        sph = P.ellipsoid(r, r, r, mesh[0], mesh[1], phi_range=(-np.pi / 2, np.pi / 2), spacing="cosine")
        raw = sph.raw.copy()
        raw[:, :, 2] -= 0.2
        body = Panels(raw)
        db = DoubleBodyFlow(body, U, mesh, lid=False)
        m1a, m3a = sphere_db_mterms(body, (0.0, 0.0, -0.2), r, U)
        sc = np.abs(m1a).max()
        mp = db.mterms(ref=(0.0, -0.2), modes=(1, 3), method="pointwise")
        ms = db.mterms(ref=(0.0, -0.2), modes=(1, 3), method="sbp")
        errs[mesh] = max(np.abs(mp[1][1] - m1a).max(), np.abs(mp[3][1] - m3a).max()) / sc
        es = np.abs(ms[1][1] - m1a) / sc
        sbp_notes.append(f"{mesh}: max {es.max():.3f}, rms {np.sqrt(np.mean(es ** 2)):.3f}")
    order = float(np.log(errs[(32, 16)] / errs[(64, 32)]) / np.log(2.0))
    rows = [_row("MT", "pointwise double-body m-terms (11.2 item 6) on a sphere in unbounded flow vs the analytic flow, "
                 "mesh 64 x 32: max error / max |m_1|", float(errs[(64, 32)]), 0.0, "analytic sphere flow", 0.01,
                 errs[(64, 32)] <= 0.01, note=f"32x16 {errs[(32, 16)]:.2e}"),
            _row("MT", "pointwise m-terms: observed order 32x16 -> 64x32", order, 0.0, "analytic sphere flow", ">= 1.5",
                 order >= 1.5),
            _row("MT", "production (finite-volume adjoint) m-terms vs the analytic flow (pointwise values; information: "
                 "weakly consistent, grid-scale errors where the spacing changes, 11.2 item 6)",
                 float(es.max()), 0.0, "analytic", "info", True, note="; ".join(sbp_notes))]
    from hydro.rankine.hull import wigley_journee, prismatic_v
    worst, notes = 0.0, []
    for name, h in (("wigleyJ", wigley_journee()), ("prismV", prismatic_v()), ("prismV_D0.06", prismatic_v(D0=0.06))):
        body = h.mesh(60, 12)
        db = DoubleBodyFlow(body, 1.0, (60, 12))
        mv = db.mterms(method="sbp")
        c, a = body.c, body.area
        w_case = 0.0
        for f in (c[:, 0], c[:, 0] * c[:, 2], np.cos(6 * c[:, 0]) * np.exp(8 * c[:, 2]), c[:, 2] ** 2 - c[:, 0]):
            Gf = db.wgrad(f)
            for j in (1, 3, 5):
                nj, mj = mv[j]
                w_case = max(w_case, abs(np.sum((nj * Gf + mj * f) * a)) / max(np.sum(np.abs(nj * Gf) * a), 1e-300))
        worst = max(worst, w_case)
        notes.append(f"{name} {w_case:.1e}")
    rows.append(_row("SBP", "discrete Tuck identity with the production m-terms, |sum a (n_j G f + m_j f)| / sum |a n_j G f| "
                     "(Wigley I, flared and wall-sided prismatic, 60 x 12; four test functions; modes 1, 3, 5)", worst, 0.0,
                     "exact by construction (11.2 item 6)", _impl(TOL_SBP), worst <= TOL_SBP, note="; ".join(notes)))
    return rows


# =====================================================================
# manufactured test of the double-body operators (M2)
def dxd_check(args):
    """DXD (M2): the advective operator A and the conservative K = Wa^-1 A_{-U}^T Wa applied to the manufactured
    field f of the B5 DXM check (a = b = 2 pi/0.25 L) on CHECK band grids M = 20, 40, 80 (Wigley I and the flared
    prismatic body, stream +x), with the base velocity W and Phi_zz of the double-body flow at the centroids (from the
    representation, independent of the operators): error of A f vs W.grad f and of K f vs W.grad f - Phi_zz f
    (= div_h(W f)), max |.| / (a |W|_max) over three regions inside the window (wall line = band line 0 over the hull,
    rest of the band, interior). Binding: observed order M 40 -> 80 >= 1 on the wall line and in the band, >= 1.5 in
    the interior, for both operators."""
    Ms, = args
    from hydro.rankine.hull import prismatic_v
    from hydro.rankine.unsteady import build_grid
    from hydro.rankine.doublebody import DoubleBodyFlow, fs_operators
    rows = []
    for hname in ("wigleyJ", "prismV"):
        h = _wj() if hname == "wigleyJ" else prismatic_v()
        body = h.mesh(40, 8)
        db = DoubleBodyFlow(body, 1.0, (40, 8))
        res = {"A": {}, "K": {}}
        for M in Ms:
            G = build_grid(h, 0.0, 0.0, 1.0, **dict(CHECK_GRID, M=M))
            P = G["fs"].c
            Wf = db.field_velocity(P)
            pz = db.field_phizz(P)
            A, K, _ = fs_operators(G, Wf, 1.0, "cons")
            nx, ny = G["shape"]
            X, Y = G["Xc"], G["Yc"]
            a = b = 2 * np.pi / 0.25
            wx = np.exp(-(X / 0.9) ** 8)
            wy = np.exp(-(Y / 0.5) ** 8)
            f = np.cos(a * X) * np.cos(b * Y) * wx * wy
            fx = (-a * np.sin(a * X) * wx + np.cos(a * X) * wx * (-8 * X ** 7 / 0.9 ** 8)) * np.cos(b * Y) * wy
            fy = np.cos(a * X) * wx * (-b * np.sin(b * Y) * wy + np.cos(b * Y) * wy * (-8 * Y ** 7 / 0.5 ** 8))
            Wx2, Wy2 = Wf[:, 0].reshape(ny, nx).T, Wf[:, 1].reshape(ny, nx).T
            pz2 = pz.reshape(ny, nx).T
            ex_A = Wx2 * fx + Wy2 * fy
            ex_K = ex_A - pz2 * f
            Af = (A @ f.T.reshape(-1)).reshape(ny, nx).T
            Kf = (K @ f.T.reshape(-1)).reshape(ny, nx).T
            sc = a * np.abs(Wf).max()
            i_s, i_b, J = G["i_s"], G["i_b"], G["J"]
            reg = np.zeros((nx, ny), int)
            reg[i_s:i_b, 1:J] = 1
            reg[i_s:i_b, 0] = 2
            inwin = (np.abs(X) < 0.8) & (Y < 0.45)
            for key, num, ex in (("A", Af, ex_A), ("K", Kf, ex_K)):
                err = np.abs(num - ex) / sc
                res[key][M] = [float(err[(reg == r) & inwin].max()) for r in (2, 1, 0)]
        for key in ("A", "K"):
            e1, e2 = res[key][Ms[-2]], res[key][Ms[-1]]
            orders = [float(np.log(e1[k] / e2[k]) / np.log(Ms[-1] / Ms[-2])) for k in range(3)]
            note = "; ".join(f"M{M} " + "/".join(f"{v:.2e}" for v in res[key][M]) for M in Ms) + " (wall line/band/interior)"
            for k, (rn, thr) in enumerate((("wall line", 1.0), ("band", 1.0), ("interior", 1.5))):
                rows.append(_row("DXD", f"manufactured {'advective A' if key == 'A' else 'conservative K'} vs exact, "
                                 f"{hname}, {rn}: observed order M {Ms[-2]} -> {Ms[-1]}", orders[k], 0.0,
                                 "exact derivatives with the representation's W and Phi_zz", f">= {thr}",
                                 orders[k] >= thr, note=note if k == 0 else ""))
    return rows


# =====================================================================
# Timman-Newman, Haskind (M3)
def _pm(hull, Fn, grid, hm, ref, kinematic="cons", mterms="sbp"):
    U = Fn * np.sqrt(G0 * hull.L)
    S = {sg: _db(hull, sg * U, grid, hm, sgn=sg, ref=ref, kinematic=kinematic, mterms=mterms) for sg in (1, -1)}
    same = np.allclose(S[1][2]["fs"].c, S[-1][2]["fs"].c) and np.allclose(S[1][2]["nu"], S[-1][2]["nu"])
    return S, U, same


def _tn_pairs(S, wps, pairs, zero_m=False):
    from hydro.rankine.unsteady import green_terms
    if zero_m:
        for sg in (1, -1):
            S[sg][1].mv = {k: (v[0], 0.0 * v[1]) for k, v in S[sg][1].mv.items()}
    out = []
    for wp in wps:
        w = wp * SC
        rp = S[1][1].radiation(w, want_phi=True)
        rm = S[-1][1].radiation(w, want_phi=True)
        for (j, kk) in pairs:
            A_, B_ = rp["sol"], rm["sol"]
            _, fs_s = green_terms(S[1][0], A_["phi_B"][:, kk], rp["phin"][:, kk], A_["sigma"][:, kk], A_["phi_F"][:, kk],
                                  B_["phi_B"][:, j], rm["phin"][:, j], B_["sigma"][:, j], B_["phi_F"][:, j])
            dT = rp["T_ot"][j, kk] - rm["T_ot"][kk, j]
            scl = max(abs(rp["T_ot"][j, kk]), abs(rm["T_ot"][kk, j]))
            dD = abs(rp["T"][j, kk] - rp["T_ot"][j, kk]) / scl
            out.append(dict(wp=wp, pair=f"{(1, 3, 5)[j]}{(1, 3, 5)[kk]}", tn=abs(dT) / scl, fs=abs(RHO * fs_s) / scl,
                            dD=dD, Tp=rp["T_ot"][j, kk], Tm=rm["T_ot"][kk, j]))
    return out


def tn_wigley(args):
    """TN (M3, binding 3 %): classical Timman-Newman T^OT_jk(U) = T^OT_kj(-U) on the Wigley I at Fn 0.3 (CHECK grid,
    +U and -U on shared panels), omega' 2.212 and 4.424, pairs (35), (13), (15) -- the double-body form of B5 check
    (b) (the waterline term vanishes, 11.2 item 4). FSR (binding 1e-10): the discrete free-surface Green sum of the
    same pairs relative to |T| (exactly zero with the conservative operator). TK (binding 1e-10): |T^D - T^OT| / |T|.
    HK (M3, binding 3 % of |X|): Haskind with the incident-wave/base-flow free-surface term, X^OT_D,j = -rho int psi_j
    dphi_I/dn + rho int_F (zeta_psi f_d + psi f_k), lambda/L 1.0, 1.5; classical Haskind and the effect of the forcing
    on |X| (information). TNn (information): the same TN pairs with the natural kinematic operator K = A - Phi_zz
    (consistency of the pair). Haskind at U = 0 (binding 3 %)."""
    Fn, lams, wps, grid, hm = args[:5]
    light = len(args) > 5 and args[5] == "light"
    h = _wj()
    S, U, same = _pm(h, Fn, grid, hm, (0.0, ZG))
    res = _tn_pairs(S, wps, ((1, 2), (0, 1), (0, 2)))
    rows = [_row("TN", f"classical Timman-Newman (double-body form of check (b)), Wigley I Fn {Fn}, CHECK M {grid['M']}, "
                 f"hull {hm}, omega' {list(wps)}, pairs (35), (13), (15)", float(max(r["tn"] for r in res)), 0.0,
                 "identity (11.2 item 4)", _impl(TOL_ID), max(r["tn"] for r in res) <= TOL_ID and same,
                 note=("same panels" if same else "PANELS DIFFER (row invalid)") + "; " + "; ".join(
                     f"{r['pair']}@{r['wp']}: {r['Tp']:.4e} / {r['Tm']:.4e}" for r in res)),
            _row("FSR", "discrete free-surface Green sum between +U and -U solutions / |T| (Wigley I)",
                 float(max(r["fs"] for r in res)), 0.0, "exact (11.2 item 4)", _impl(TOL_FSR),
                 max(r["fs"] for r in res) <= TOL_FSR),
            _row("TK", "|T^D - T^OT| / |T| (Wigley I radiation; D and OT forms coincide with the adjoint m-terms)",
                 float(max(r["dD"] for r in res)), 0.0, "exact (11.2 items 6-7)", _impl(TOL_FSR),
                 max(r["dD"] for r in res) <= TOL_FSR)]
    if light:
        for sg in (1, -1):
            S[sg][0].release()
        _free()
        return rows
    cp, cm = S[1][1], S[-1][1]
    hk, hkc, dX, notes = [], [], [], []
    for lam in lams:
        k = 2 * np.pi / (lam * L)
        w0 = np.sqrt(G0 * k)
        ex = cp.excitation(w0, np.pi, want_phi=True, forcing=True)
        ex0 = cp.excitation(w0, np.pi, forcing=False)
        we = ex["omega_e"]
        rm = cm.radiation(we, want_phi=True)
        for jj, j in enumerate(cp.modes):
            psi = rm["sol"]["phi_B"][:, jj]
            cl = -RHO * 2 * np.sum(psi * ex["dn_I"] * S[1][0].body.area)
            zpsi = S[-1][0].zeta_of(we, rm["sol"]["phi_F"][:, jj])
            GF = 2 * np.sum((zpsi * ex["fd"] + rm["sol"]["phi_F"][:, jj] * ex["fk"]) * S[1][0].fs.area)
            Xd = ex["X_d_ot"][jj]
            hk.append(abs(Xd - (cl + RHO * GF)) / abs(ex["X"][jj]))
            hkc.append(abs(Xd - cl) / abs(ex["X"][jj]))
            dX.append(abs(ex["X"][jj]) / abs(ex0["X"][jj]) - 1)
            notes.append(f"lam/L {lam} X{j}: |X| {abs(ex['X'][jj]):.4e} (no forcing {abs(ex0['X'][jj]):.4e}); X_D^OT "
                         f"{Xd:.4e}, Haskind {cl + RHO * GF:.4e}, classical {cl:.4e}")
    rows += [_row("HK", f"Haskind with the incident-wave/base-flow free-surface term (11.2 item 5), Wigley I Fn {Fn}, "
                  f"lambda/L {list(lams)}, relative to |X|", float(max(hk)), 0.0, "identity (11.2 item 5)", _impl(TOL_ID),
                  max(hk) <= TOL_ID),
             _row("HK", "classical Haskind (free-surface term omitted): size of the interaction term (information)",
                  float(max(hkc)), 0.0, "information", "info", True, note="; ".join(notes)),
             _row("HK", "effect of the incident-wave/base-flow forcing on |X| (forced / simplified - 1; information)",
                  float(max(np.abs(dX))), 0.0, "information", "info", True,
                  note=", ".join(f"{v:+.3f}" for v in dX))]
    for sg in (1, -1):
        S[sg][0].release()
    _free()
    Sn, _, _ = _pm(h, Fn, grid, hm, (0.0, ZG), kinematic="nat")
    resn = _tn_pairs(Sn, wps, ((1, 2), (0, 1), (0, 2)))
    rows.append(_row("TNn", "TN with the natural kinematic operator K = A - diag(Phi_zz) (consistency of the pair; "
                     "information)", float(max(r["tn"] for r in resn)), 0.0, "information", "info", True))
    for sg in (1, -1):
        Sn[sg][0].release()
    s0, c0, _ = _db(h, 0.0, grid, hm, sgn=1.0)
    e0 = []
    for lam in lams:
        w0 = np.sqrt(G0 * 2 * np.pi / (lam * L))
        ex = c0.excitation(w0, np.pi, want_phi=True)
        r0 = c0.radiation(w0, want_phi=True)
        for jj in range(3):
            cl = -RHO * 2 * np.sum(r0["sol"]["phi_B"][:, jj] * ex["dn_I"] * s0.body.area)
            e0.append(abs(ex["X_d"][jj] - cl) / abs(ex["X"][jj]))
    s0.release()
    _free()
    rows.append(_row("HK", "Haskind at U = 0 (classical), Wigley I", float(max(e0)), 0.0, "identity", _impl(TOL_ID),
                     max(e0) <= TOL_ID))
    return rows


def tn_planing(args):
    """TNp / TNw (M3): classical Timman-Newman on the planing-type prismatic bodies, Fn 0.4, CHECK grid, +U and -U on
    shared panels, omega' 2.212, 4.424, pairs (35), (53), (13), (15). Binding 3 % at the registered member (M 40, hull
    40 x 8, as B5); refinement members (hull 60 x 12, 90 x 18 at M 40; joint M 60 / 60 x 12) information, with the
    observed order of the hull series; the m = 0 diagnostic (the identity holds for any m-terms; it isolates the
    representation's own reciprocity) information."""
    kind, tag, members = args
    h = _prism(kind)
    rows, ser = [], []
    pairs = ((1, 2), (2, 1), (0, 1), (0, 2))
    for i, (M, hm, binding) in enumerate(members):
        S, U, same = _pm(h, 0.4, dict(CHECK_GRID, M=M), hm, (0.0, 0.0))
        res = _tn_pairs(S, (2.212, 4.424), pairs)
        e = float(max(r["tn"] for r in res))
        fsr = float(max(r["fs"] for r in res))
        ser.append((M, hm, e))
        rows.append(_row(tag if binding else tag + "r", f"classical Timman-Newman, prismatic V {kind} body Fn 0.4, CHECK "
                         f"M {M}, hull {hm}" + ("" if binding else " (refinement member; information)"), e, 0.0,
                         "identity (11.2 item 4)", _impl(TOL_ID) if binding else "info",
                         (e <= TOL_ID and same) if binding else True,
                         note=f"FS Green sum {fsr:.1e}; " + "; ".join(f"{r['pair']}@{r['wp']}: {r['tn']:.3f}" for r in res)))
        if i == 0:
            res0 = _tn_pairs(S, (2.212, 4.424), pairs, zero_m=True)
            rows.append(_row(tag + "m0", f"TN with the m-terms set to zero (diagnostic, {kind} body, M {M}, hull {hm}; "
                             "information)", float(max(r["tn"] for r in res0)), 0.0, "information", "info", True))
        for sg in (1, -1):
            S[sg][0].release()
        _free()
    hs = [(m, hm, e) for (m, hm, e) in ser if m == members[0][0]]
    if len(hs) >= 3:
        p = np.log(hs[-2][2] / hs[-1][2]) / np.log(hs[-1][1][0] / hs[-2][1][0])
        rows.append(_row(tag + "o", f"TN hull series at M {hs[0][0]} ({', '.join(str(x[1]) for x in hs)}): defects "
                         f"{', '.join(f'{x[2]:.4f}' for x in hs)}, observed order of the last step (information)", float(p),
                         0.0, "information", "info", True))
    return rows


def tn_submerged(args):
    """TNs (M3, binding 3 %): classical Timman-Newman for a submerged spheroid (a/b 5, depth 2b) with the double-body
    base flow (lid image), Fn 0.3."""
    Fn, wps, grid = args
    from hydro.rankine import panels as P
    from hydro.rankine.panels import Panels
    from hydro.rankine.unsteady import build_grid, NoWaterline
    from hydro.rankine.doublebody import DoubleBodyFlow, fs_operators, DBForwardSpeedLTI, DBCoefficients
    sph = P.ellipsoid(0.5, 0.1, 0.1, 40, 12, phi_range=(-np.pi / 2, np.pi / 2), spacing="cosine")
    raw = sph.raw.copy()
    raw[:, :, 2] -= 0.2
    body = Panels(raw)
    nw = NoWaterline(-0.5, 0.5, 1.0)
    U = Fn * SC
    S = {}
    for sg in (1, -1):
        G = build_grid(nw, sgn=sg, **grid)
        db = DoubleBodyFlow(body, sg * U, (40, 12))
        Wf = db.field_velocity(G["fs"].c)
        A, K, _ = fs_operators(G, Wf, sg, "cons")
        s = DBForwardSpeedLTI(body, G["fs"], A, K, G["nu"] * SC, sg * U, G0, RHO, hull_shape=(40, 12))
        S[sg] = (s, DBCoefficients(s, db, ref=(0.0, -0.2)), G)
    errs, notes = [], []
    for wp in wps:
        rp = S[1][1].radiation(wp * SC)
        rm = S[-1][1].radiation(wp * SC)
        for (j, k) in ((1, 2), (0, 1), (0, 2)):
            a, b = rp["T_ot"][j, k], rm["T_ot"][k, j]
            errs.append(abs(a - b) / max(abs(a), abs(b)))
            notes.append(f"w' {wp} ({(1, 3, 5)[j]}{(1, 3, 5)[k]}): {a:.4e} / {b:.4e}")
    for sg in (1, -1):
        S[sg][0].release()
    _free()
    return [_row("TNs", f"classical Timman-Newman, submerged spheroid a/b 5 depth 2b, double-body basis, Fn {Fn}",
                 float(max(errs)), 0.0, "theorem (11.2 item 4)", _impl(TOL_ID), max(errs) <= TOL_ID,
                 note="; ".join(notes))]


# =====================================================================
# submerged sphere vs Wu & Eatock Taylor (1988)
def _we_tol(nm, e):
    if nm in ("A11", "A33"):
        return 0.03 * np.abs(e)
    if nm in ("B11", "B33"):
        return np.maximum(0.05 * np.abs(e), 0.02 * np.abs(e).max())
    return np.maximum(0.05 * np.abs(e), 0.05 * np.abs(e).max())


def we_sphere(args):
    """WE (binding, rules of B5 10.8.4): submerged sphere r, depth 2r, Fr = U/sqrt(g r) = 0.4, against Wu & Eatock
    Taylor (1988) via Mei et al. (2020): reference model = uniform-stream free surface + double-body m-terms + m-term
    force; here with the PRODUCTION double-body m-terms (finite-volume adjoint, rigid-lid base flow) computed from
    the panel solution. Members (mesh, binding): the B5 mesh 24 x 12 and 48 x 24. WEdb (information): the full
    double-body model (double-body free surface as well) -- a different model from the reference. WEx (information):
    exciting forces."""
    members, = args
    from hydro.rankine import panels as P
    from hydro.rankine.panels import Panels
    from hydro.rankine.unsteady import build_grid, dx_operator, ForwardSpeedLTI, Coefficients, NoWaterline
    from hydro.rankine.doublebody import DoubleBodyFlow, fs_operators, DBForwardSpeedLTI, DBCoefficients
    r = WE_R
    U = 0.4 * np.sqrt(G0 * r)
    ref = _load_we("we1988_sphere_fr040_h2r_radiation.csv")
    names = ["A11", "B11", "A33", "B33", "A13", "A31", "B13", "B31"]
    src = "Wu & Eatock Taylor 1988 via Mei et al. 2020 (JMSE 8:106) tables 3, 5, 6"
    rows = []
    for mesh, binding, full_db in members:
        sph = P.ellipsoid(r, r, r, mesh[0], mesh[1], phi_range=(-np.pi / 2, np.pi / 2), spacing="cosine")
        raw = sph.raw.copy()
        raw[:, :, 2] -= 2 * r
        body = Panels(raw)
        nw = NoWaterline(-r, r, 1.0)
        G = build_grid(nw, sgn=1.0, **WE_GRID)
        db = DoubleBodyFlow(body, U, mesh, lid=True)
        variants = []
        D, _ = dx_operator(G, 1.0)
        s = ForwardSpeedLTI(body, G["fs"], D, G["nu"] * SC, U, G0, RHO)
        co = Coefficients(s, ref=(0.0, -2 * r), modes=(1, 3))
        co.mv = db.mterms(ref=(0.0, -2 * r), modes=(1, 3), method="sbp")
        variants.append(("NK free surface + double-body m-terms (reference model)", co, s, binding, "WE"))
        if full_db:
            Wf = db.field_velocity(G["fs"].c)
            A, K, _ = fs_operators(G, Wf, 1.0, "cons")
            s2 = DBForwardSpeedLTI(body, G["fs"], A, K, G["nu"] * SC, U, G0, RHO, hull_shape=mesh)
            variants.append(("full double-body model (double-body free surface; not the reference model)",
                             DBCoefficients(s2, db, ref=(0.0, -2 * r), modes=(1, 3)), s2, False, "WEdb"))
        for label, cx, sx, bind, tag in variants:
            calc = []
            for ker in ref[:, 0]:
                w = np.sqrt(G0 * ker / r)
                T = cx.radiation(w, d_form=False)["T_ot"]
                A_, B_ = -T.real / w ** 2, T.imag / w
                a_, b_ = A_ / (RHO * np.pi * r ** 3), B_ / (RHO * np.pi * w * r ** 3)
                calc.append([a_[0, 0], b_[0, 0], a_[1, 1], b_[1, 1], a_[0, 1], a_[1, 0], b_[0, 1], b_[1, 0]])
            c = np.array(calc)
            worst, fails, notes = 0.0, 0, []
            for q, nm in enumerate(names):
                e = ref[:, q + 1]
                ex = np.abs(c[:, q] - e) / _we_tol(nm, e)
                worst = max(worst, float(ex.max()))
                fails += int(np.sum(ex > 1))
                notes.append(f"{nm}: calc " + " ".join(f"{v:.4f}" for v in c[:, q]) + f" (max err/tol {ex.max():.2f})")
            rows.append(_row(tag if bind else (tag if tag == "WEdb" else tag + "r"),
                             f"submerged sphere Fr 0.4, h = 2r, mesh {mesh}: {label}; max error / tolerance (NF {sx.NF})",
                             worst, 1.0, src, "<= 1 (per-coefficient rules)" if bind else "info", worst <= 1.0 if bind else True,
                             note=f"{fails} of {8 * len(ref)} values outside; " + "; ".join(notes)))
            sx.release()
        _free()
    return rows


# =====================================================================
# Kramers-Kronig, limits, spectrum
def kk_check(args):
    """KK (M3, binding 2 %): forward-speed Kramers-Kronig (stage note 10.2 item 5) for the double-body system,
    Wigley I, SMALL grid, log omega' 0.01-200 + band around tau = 1/4; entries skipped below 1e-3 x the band maximum of
    |A33| (rankine_db_kk: C0_33 != 0 here). D = OT, one row. INF (binding 1 %): A(omega' 200) vs A_inf."""
    Fn, npts, grid, hm = args
    from hydro.rankine.tables import static_stiffness
    h = _wj()
    U = Fn * np.sqrt(G0 * L)
    s, co, _ = _db(h, U, grid, hm)
    inst = co.instantaneous()
    C0, _ = static_stiffness(co)
    wc = G0 / (4 * U) / SC
    wp = np.unique(np.concatenate([np.exp(np.linspace(np.log(0.01), np.log(200.0), npts)),
                                   np.linspace(0.7 * wc, 1.3 * wc, npts // 4)]))
    w = wp * SC
    TD = np.array([co.radiation(x)["T"] for x in w])
    wtest = np.array([1.0, 1.659, 2.212, 3.318, 4.424, 5.53, 8.0]) * SC
    errs = kk_errors_band(w, TD, inst["A"], inst["B"], C0, wtest, (1.0 * SC, 8.0 * SC))
    emax = max(float(np.abs(e).max()) for e in errs.values())
    Ahi = -TD[-1].real / w[-1] ** 2
    ehi = float(np.abs(Ahi - inst["A"]).max() / np.abs(inst["A"]).max())
    s.release()
    _free()
    return [_row("KK", f"Kramers-Kronig, double-body system, Wigley I Fn {Fn}, {len(w)} frequencies, entries {sorted(errs)}",
                 emax, 0.0, "Kramers-Kronig with A_inf, B_inf, C0 (10.2 item 5)", TOL_KK, emax <= TOL_KK,
                 note=f"C0 diag {np.diag(C0)}"),
            _row("INF", f"A(omega' = {wp[-1]:.0f}) vs A_inf of the double-body system", ehi, 0.0, "limit (M4)", TOL_INF,
                 ehi <= TOL_INF)]


def g0_limit(args):
    """G0 (M4, binding 5 %): g -> 0 at fixed U and omega L/U = 2 pi: max |T - T_inf| / max |T_inf| at Fn 8 (Fn 1, 2,
    4 information), double-body system."""
    grid, hm = args
    h = _wj()
    Uq = 1.0
    errs = {}
    for Fn in (1.0, 2.0, 4.0, 8.0):
        g = (Uq / Fn) ** 2 / L
        s, co, _ = _db(h, Uq, grid, hm, g=g)
        inst = co.instantaneous()
        w = 2 * np.pi * Uq / L
        T = co.radiation(w)["T"]
        Tinf = -w ** 2 * inst["A"] + 1j * w * inst["B"] + inst["C"]
        errs[Fn] = float(np.abs(T - Tinf).max() / np.abs(Tinf).max())
        s.release()
    _free()
    return [_row("G0", "g -> 0 (double-body system): max |T - T_inf| / max |T_inf| at Fn 8 (omega L/U = 2 pi)", errs[8.0],
                 0.0, "phi = 0 theory, same discrete system", TOL_G0, errs[8.0] <= TOL_G0,
                 note="Fn 1, 2, 4, 8: " + ", ".join(f"{errs[f]:.3e}" for f in (1.0, 2.0, 4.0, 8.0)))]


def spectrum(args):
    """SP (M3/E2): eigenvalues of the double-body LTI matrix on a small grid; forward speed with beaches: max
    Re(lambda) < 0 (binding); U = 0 without beach: <= 1e-5 sqrt(g/L) (binding; expected FAIL as in B5: at U = 0 the
    system is the B5 one)."""
    cases, grid, hm = args
    rows = []
    for hname, Fn, beach in cases:
        h = _wj() if hname == "wigleyJ" else _prism("flared" if hname == "prismV" else "wall")
        U = Fn * np.sqrt(G0 * L)
        gk = dict(grid) if beach else dict(grid, nu_max=0.0)
        if U == 0:
            gk = dict(gk, down=gk.get("up", 1.5), b_dn=gk.get("b_up", 0.75))
        s, co, _ = _db(h, U, gk, hm, sgn=1.0, ref=(0.0, 0.0))
        NF = s.NF
        Mc = s.state_matrix()
        nrs = None
        if U == 0 and not beach:          # B6 review (B7): complex eigenvalues of the discrete NtD map R_s
            er = np.linalg.eigvals(s.Rs.cpu().numpy())
            nrs = (int(np.sum(np.abs(er.imag) > 1e-9 * np.abs(er).max())), float(er.real.max()))
        s.release()
        _free()
        ev = np.linalg.eigvals(Mc)
        del Mc
        mr = float(ev.real.max() / SC)
        if beach and U != 0:
            rows.append(_row("SP", f"spectrum, double-body system, {hname} Fn {Fn}, beach on (NF {NF}): max Re(lambda) "
                             "[sqrt(g/L)]", mr, 0.0, "< 0 (every mode decays)", "< 0", mr < 0.0,
                             note=f"eigenvalues with Re > 0: {int(np.sum(ev.real > 0))}"))
        else:
            rows.append(_row("SP", f"spectrum, {hname} U = 0, {'beach on' if beach else 'no beach'} (NF {NF}): max "
                             "Re(lambda) [sqrt(g/L)]", mr, 0.0, "acceptance M3 drift 0.1 %/100 s at L = 5.4 m", 1e-5,
                             mr <= 1e-5, note=f"eigenvalues with Re > 0: {int(np.sum(ev.real > 0))}"))
            if nrs is not None:
                rows.append(_row("SPr", f"U = 0 without a beach (NF {NF}): non-real eigenvalues of the discrete "
                                 "Neumann-to-Dirichlet map R_s (lambda^2 = g eig(R_s^-1): a complex pair of R_s gives a "
                                 "growing mode; information, B6 review B7)", float(nrs[0]), 0.0, "information", "info", True,
                                 note=f"max Re eig(R_s) {nrs[1]:+.3e}"))
    return rows


# =====================================================================
# time domain
def td_checks(args):
    """TD (M2, binding 1 %): RK4 integration of the double-body system (heave pulse, record t' 100) vs its frequency
    response T33 at omega' 2, 3, 4.5. CU (M2, binding 1 %): Cummins reconstruction through the table pipeline and the
    documented plant law (D = OT). EN: pseudo-energy H = E + M_W (11.2 item 8): U = 0 residual at t' 8 <= 1 % (binding);
    Fn 0.3 residual <= 0 at t' 8 and over the record (binding sign; magnitude information)."""
    Fn, grid, hm = args
    from hydro.rankine.doublebody import DBTimeDomain
    from hydro.rankine.tables import memory_kernel, static_stiffness, RadiationForce
    h = _wj()
    U = Fn * np.sqrt(G0 * L)
    rows = []
    s, co, _ = _db(h, U, grid, hm)
    td = DBTimeDomain(co)
    dt = 0.05 / SC
    mot = _pulse(1, 0.01, 3.0 / SC, 0.5 / SC)
    r = td.run(mot, dt, int(100 / 0.05), energy=False)
    F, x = r["F_D"][:, 1], r["x"][:, 1]
    nfft = 1 << int(np.ceil(np.log2(len(F) * 4)))
    Xf, Ff = np.fft.rfft(x, nfft), np.fft.rfft(F, nfft)
    wv = 2 * np.pi * np.fft.rfftfreq(nfft, dt)
    errs, notes = [], []
    for wp in (2.0, 3.0, 4.5):
        kk = int(np.argmin(np.abs(wv - wp * SC)))
        Tfd = co.radiation(wv[kk])["T"][1, 1]
        errs.append(abs(-Ff[kk] / Xf[kk] - Tfd) / abs(Tfd))
        notes.append(f"w' {wv[kk] / SC:.3f}: {errs[-1]:.2e}")
    rows.append(_row("TD", f"time domain (RK4, record t' 100) = frequency domain, T33, double-body Wigley I Fn {Fn} (NF {s.NF})",
                     float(max(errs)), 0.0, "same LTI system", TOL_TD, max(errs) <= TOL_TD, note="; ".join(notes)))
    inst = co.instantaneous()
    C0, _ = static_stiffness(co)
    wq = np.concatenate([np.exp(np.linspace(np.log(0.01), np.log(0.6), 30)), np.linspace(0.6, 1.2, 60)[1:],
                         np.exp(np.linspace(np.log(1.2), np.log(150), 120))[1:]]) * SC
    BD = np.array([co.radiation(x_)["B"] for x_ in wq])
    tk = np.arange(0, 40 + 1e-9, 0.05) / SC
    K, tail = memory_kernel(wq, BD, inst["B"], tk)
    ecu = []
    for mode in (1, 2):
        mt = _pulse(mode, 0.01, 5.0 / SC, 1.0 / SC)
        rr = td.run(mt, dt, int(40 / 0.05), energy=False)
        rf = RadiationForce(dict(A_inf=inst["A"], B_inf=inst["B"], C0=C0, K=K, t=tk), dt, stiffness=True)
        Fc = np.array([rf(mt(tt)[2], mt(tt)[1], mt(tt)[0]) for tt in rr["t"]])
        ecu.append(float(np.sqrt(np.mean((Fc - rr["F_D"]) ** 2, axis=0)).max() / np.sqrt(np.mean(rr["F_D"][:, mode] ** 2))))
    rows.append(_row("CU", "Cummins reconstruction through the documented plant law (double-body; D = OT), heave and "
                     "pitch pulses", float(max(ecu)), 0.0, "same LTI system", TOL_TD, max(ecu) <= TOL_TD,
                     note=f"kernel tail {tail:.1e}; per mode {ecu}"))
    s.release()
    _free()
    for Fen in (0.0, Fn):
        Ue = Fen * np.sqrt(G0 * L)
        g2 = dict(grid, down=grid["up"], b_dn=grid["b_up"]) if Fen == 0 else grid
        s, co, _ = _db(h, Ue, g2, hm, sgn=1.0)
        td = DBTimeDomain(co)
        rr = td.run(_pulse(1, 0.01, 5.0 / SC, 1.0 / SC), dt, int(40 / 0.05))
        tt = rr["t"]
        P = rr["P_body"] + rr["P_beach"]
        I = np.concatenate([[0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(tt))])
        Wb = np.concatenate([[0], np.cumsum(0.5 * (np.abs(rr["P_body"][1:]) + np.abs(rr["P_body"][:-1])) * np.diff(tt))])
        res = (rr["H"] - rr["H"][0] - I) / Wb[-1]
        k8 = int(np.argmin(np.abs(tt * SC - 8.0)))
        if Fen == 0:
            rows.append(_row("EN", "energy balance, U = 0, heave pulse: |residual| / int|P_body| at t' = 8",
                             float(abs(res[k8])), 0.0, "energy identity", TOL_EN, abs(res[k8]) <= TOL_EN))
        else:
            rows.append(_row("EN", f"pseudo-energy E + M_W balance Fn {Fen} (no waterline power): residual at t' 8 "
                             "(magnitude = upwind dissipation; information)", float(res[k8]), 0.0, "11.2 item 8", "info",
                             True, note=f"whole record {res[-1]:+.3e}"))
            rows.append(_row("EN", f"pseudo-energy balance Fn {Fen}: residual <= 0 at t' 8 and over the record", float(
                max(res[k8], res[-1])), 0.0, "sign", "<= 0", max(res[k8], res[-1]) <= 0.0))
        s.release()
    _free()
    return rows


def froude_similarity(args):
    """FS (M3, binding 1e-4): double-body Wigley I at L = 1 and 1.5 (scaled grid and meshes), Fn 0.3: non-dimensional
    T at omega' 2.212 and the excitation at lambda/L 1.2."""
    grid, hm = args
    from hydro.rankine.hull import wigley_journee
    out = []
    for Ls in (1.0, 1.5):
        h = wigley_journee(L=Ls, B=0.1 * Ls, T=0.0625 * Ls)
        V = h.hydrostatics()["V"]
        U = 0.3 * np.sqrt(G0 * Ls)
        s, co, _ = _db(h, U, grid, hm, ref=(0.0, ZG * Ls))
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
    return [_row("FS", "Froude similarity L = 1 vs 1.5, double-body system (Fn 0.3)", e, 0.0, "exact", TOL_FS, e <= TOL_FS)]


# =====================================================================
# convergence
def _coef(h, Fn, grid, hm, wps, ref=None):
    s, co, _ = _db(h, Fn * np.sqrt(G0 * L), grid, hm, ref=ref)
    V = h.hydrostatics()["V"]
    out = np.array([np.concatenate([_nd(co.radiation(wp * SC)["T"], wp * SC, V)[q][[1, 2, 1, 2], [1, 2, 2, 1]]
                                    for q in (0, 1)]) for wp in wps])
    NF = s.NF
    s.release()
    _free()
    return out, NF


def grid_series(args):
    """GS (M5, binding 2 %): Wigley I Fn 0.2, free-surface series M 40/60/80 (hull 60 x 12) at Journee's seven
    frequencies and hull series 40x8/60x12/90x18 (M 60) at omega' 1.659, 2.765, 4.424: change of A33', A55', B33',
    B55' between the two finest members (relative to max(|v|, 0.1 max)). GSo: observed orders with the actual ratios and
    the Richardson error of the PROD member (information)."""
    Fn, = args
    h = _wj()
    rows = []
    for nm, members, hs, wps in (("free-surface M 40/60/80", [(dict(GRID_BASE, M=M), (60, 12)) for M in (40, 60, 80)],
                                  (1 / 40, 1 / 60, 1 / 80), W_J),
                                 ("hull 40x8/60x12/90x18", [(dict(GRID_BASE, M=60), hm) for hm in HULL_SERIES],
                                  (1 / 40, 1 / 60, 1 / 90), (1.659, 2.765, 4.424))):
        vals, nfs = [], []
        for grid, hm in members:
            v, nf = _coef(h, Fn, grid, hm, wps)
            vals.append(v)
            nfs.append(nf)
        diag = [0, 1, 4, 5]
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
        rows.append(_row("GS", f"double-body grid series ({nm}), Wigley I Fn {Fn}: change of A33', A55', B33', B55' between "
                         f"the two finest (NF {nfs})", e, 0.0, "own", TOL_GS, e <= TOL_GS,
                         note=f"first change {d1[:, diag].max():.2e}; couplings finest {d2[:, [2, 3, 6, 7]].max():.2e}"))
        rows.append(_row("GSo", f"double-body grid series ({nm}): observed orders and Richardson error of the PROD member "
                         "(information)", float(np.nanmax(R[:, diag])) if np.isfinite(R[:, diag]).any() else float("nan"),
                         0.0, "Richardson", "info", True,
                         note=f"median order {np.nanmedian(P[:, diag]):.2f} ({int(np.isfinite(P[:, diag]).sum())}/"
                              f"{P[:, diag].size} finite); Richardson error median {np.nanmedian(R[:, diag]):.2e}"))
    return rows


def planing_series(args):
    """GSp (M5): joint series M 30/40/60/80 with hull 30x6/40x8/60x12/80x16 (GRID_BASE domain), Fn 0.4, omega' 2.212,
    3.318, 4.424: flared prismatic body binding 2 % (change of A33', A55', B33', B55' between the two finest members,
    relative to max(|v|, 0.1 max)); the wall-sided variant (submerged chine: non-integrable base-flow singularity,
    11.2 item 6) information."""
    kinds = args[0] if args else ("flared", "wall")
    rows = []
    for kind, binding in (("flared", True), ("wall", False)):
        if kind not in kinds:
            continue
        h = _prism(kind)
        vals, nfs = [], []
        for M, hm in ((30, (30, 6)), (40, (40, 8)), (60, (60, 12)), (80, (80, 16))):
            v, nf = _coef(h, 0.4, dict(GRID_BASE, M=M), hm, (2.212, 3.318, 4.424), ref=(0.0, 0.0))
            vals.append(v)
            nfs.append(nf)
        diag = [0, 1, 4, 5]
        den = np.maximum(np.abs(vals[-1]), 0.1 * np.abs(vals[-1]).max(axis=0))
        ch = [float((np.abs(vals[i + 1] - vals[i]) / den)[:, diag].max()) for i in range(3)]
        e = ch[-1]
        rows.append(_row("GSp" if binding else "GSpw", f"planing-type prismatic V ({kind}) Fn 0.4, joint series M 30/40/60/80: "
                         f"change of the diagonal A', B' between the two finest (NF {nfs})", e, 0.0, "own",
                         TOL_GS if binding else "info", (e <= TOL_GS) if binding else True,
                         note="successive changes " + ", ".join(f"{c:.3e}" for c in ch) + "; A33' " +
                              "/".join(f"{v[0, 0]:.4f}" for v in vals) + ", B55' (w' 4.424) " +
                              "/".join(f"{v[2, 5]:.5f}" for v in vals)))
    return rows


def domain_sensitivity(args):
    """DOM (M5, binding 2 %): beach x2 / x0.5, lateral, downstream, upstream x1.5, beach lengths x1.5 (COARSE, Fn 0.2,
    omega' 1.659, 2.765, 4.424), double-body system."""
    Fn, = args
    h = _wj()
    base = dict(GRID_BASE, M=COARSE["M"])
    wps = (1.659, 2.765, 4.424)
    ref, _ = _coef(h, Fn, base, COARSE["hull"], wps)
    rows = []
    for name, d in (("nu_max x2", dict(nu_max=3.0)), ("nu_max x0.5", dict(nu_max=0.75)),
                    ("lateral x1.5", dict(lat=4.5, b_lat=2.25)), ("downstream x1.5", dict(down=6.0, b_dn=3.0)),
                    ("upstream x1.5", dict(up=2.25, b_up=1.125)), ("beach lengths x1.5", dict(b_up=1.125, b_dn=3.0, b_lat=2.25))):
        v, nf = _coef(h, Fn, dict(base, **d), COARSE["hull"], wps)
        e = float(np.max(np.abs(v[:, [0, 1, 4, 5]] / ref[:, [0, 1, 4, 5]] - 1)))
        rows.append(_row("DOM", f"double-body domain/beach: {name} (NF {nf}), Fn {Fn}", e, 0.0, "own", TOL_DOM, e <= TOL_DOM))
    return rows


# =====================================================================
# low-speed limit
def u0_checks(args):
    """U0 (M4, binding 5 %): the double-body system at U = 0 vs the B5 system at U = 0 on the same panels (both reduce
    to the zero-speed problem; regression). U0c (M4, binding 5 %): diagonal A', B' (33, 55) at Fn 0.05 vs U = 0 on one
    grid (COARSE hull, M 40, symmetric domain), omega' 2.212, 3.318; Fn 0.0125, 0.025, 0.1 and the ratio of the
    changes Fn 0.05/0.025 (4 for O(U^2), implied by the exact evenness of 11.2 item 10) information."""
    wps, grid = args
    from studies.rankine_unsteady_verify import _solver
    h = _wj()
    V = _V(h)
    g = dict(grid, down=grid["up"], b_dn=grid["b_up"], u_dn=0.0)
    vals = {}
    for Fn in (0.0, 0.0125, 0.025, 0.05, 0.1):
        U = Fn * SC
        s, co, _ = _db(h, U, g, COARSE["hull"], sgn=1.0)
        vals[Fn] = np.array([np.concatenate([_nd(co.radiation(wp * SC)["T"], wp * SC, V)[q][[1, 2], [1, 2]]
                                             for q in (0, 1)]) for wp in wps])
        s.release()
    s5, c5, _ = _solver(h, 0.0, g, COARSE["hull"], sgn=1.0)
    v5 = np.array([np.concatenate([_nd(c5.radiation(wp * SC)["T"], wp * SC, V)[q][[1, 2], [1, 2]] for q in (0, 1)])
                   for wp in wps])
    s5.release()
    _free()
    e0 = float(np.max(np.abs(vals[0.0] - v5) / np.maximum(np.abs(v5), 0.1 * np.abs(v5).max(axis=0))))
    den = np.maximum(np.abs(vals[0.0]), 0.1 * np.abs(vals[0.0]).max(axis=0))
    d = {Fn: np.abs(vals[Fn] - vals[0.0]) / den for Fn in (0.0125, 0.025, 0.05, 0.1)}
    ratio = float(np.median(d[0.05] / np.maximum(d[0.025], 1e-12)))
    return [_row("U0", "double-body system at U = 0 vs the B5 system at U = 0 (same panels), A', B' (33, 55)", e0, 0.0,
                 "B5 zero-speed limit (verified vs stage 2)", TOL_U0, e0 <= TOL_U0),
            _row("U0c", f"U -> 0 continuity (double-body): diagonal A', B' at Fn 0.05 vs U = 0, omega' {list(wps)}, M "
                 f"{grid['M']}", float(d[0.05].max()), 0.0, "U = 0, same grid", TOL_U0, d[0.05].max() <= TOL_U0,
                 note=f"max changes Fn 0.0125/0.025/0.05/0.1: " + "/".join(f"{d[F].max():.4f}" for F in d) +
                      f"; median ratio Fn 0.05/0.025 {ratio:.2f} (4 for O(U^2)); B55' changes at w' {wps[0]}: " +
                      "/".join(f"{d[F][0, 3]:+.4f}" for F in d))]


# =====================================================================
# steady Wigley (validation; B4 rules)
def steady_wigley(args):
    """SF (validation, rules of B4 9.7.2 / 9.1; parabolic Wigley, double-body steady problem on lambda0-sized grids,
    hull 60 x 12): fixed model -- SF1a hull force C_lin = R_lin/q vs U. Tokyo C_W (C_T-based, Yang et al. 2000 fig 5a),
    max(20 %, 0.20e-3); SF1b wave cut C_cut (1 lambda0 behind the stern) vs U. Tokyo C_WP, max(20 %, 0.10e-3); SF2 wave
    profile along the hull vs U. Tokyo, RMS/max <= 20 %; free model (CG at LCB, z_G = 0 as B4) -- SF3 sinkage 15 % (mean
    reading of B4), SF4 trim +-0.05 deg, SF5a hull force vs the free C_W curve max(20 %, 0.20e-3). B4 values at the
    same Fn (stage note 9.8) and the quadratic-pressure variant are information. Fn 0.2 is outside the dense solver's
    memory at 30 cells per wavelength (not run)."""
    fns_fix, fns_free = args[:2]
    dom_fn = args[2] if len(args) > 2 else None
    from hydro.rankine.hull import wigley
    from hydro.rankine.doublebody import build_db_solver, steady_db, steady_grid, steady_cut, db_free_equilibrium
    from studies.rankine_hull_verify import (wigley_wetted_surface, _yang, _yang_at, _cwp_at, _profile_exp, _tol,
                                             FLOOR_CW, FLOOR_CWP)
    h = wigley()
    S = wigley_wetted_surface()
    ut, sk, tr, cwf = _yang("fig5a_cw_fixed"), _yang("fig4a_sinkage_s"), _yang("fig4a_trim_t"), _yang("fig5a_cw_free")
    b4 = {0.25: (0.946, 0.687), 0.267: (0.891, 0.633), 0.289: (1.348, 1.049), 0.316: (1.431, 1.083),
          0.354: (1.246, 0.915), 0.408: (2.145, 1.900)}
    b6 = {0.25: (0.998, 1.039), 0.267: (0.907, 0.941), 0.289: (1.435, 1.435), 0.316: (1.547, 1.580),     # B6 (11.4),
          0.354: (1.332, 1.356), 0.408: (2.263, 2.159)}                                                 # unscaled domain
    rows = []
    for Fn in fns_fix:
        U = Fn * np.sqrt(G0)
        q = 0.5 * RHO * U * U * S
        kw = steady_grid(h, U, G0)
        t = time.time()
        s, co, G, db = build_db_solver(h, U, 0.0, 0.0, kw, (60, 12), g=G0, rho=RHO)
        out = steady_db(s, co, h)
        Rc, ci = steady_cut(s, G, out["zeta"], h, U, G0, RHO)
        NF = s.NF
        s.release()
        _free()
        cl, cq, cc = out["R_lin"] / q, out["R_quad"] / q, Rc / q
        bnote = (f"B4 (PROD): C_p {b4[Fn][0]:.3f}e-3, C_cut {b4[Fn][1]:.3f}e-3; B6 (unscaled domain): C_lin {b6[Fn][0]:.3f}e-3, "
                 f"C_cut {b6[Fn][1]:.3f}e-3; domain up {kw['up']:.3f} down {kw['down']:.3f} lat {kw['lat']:.3f}"
                 if Fn in b4 else "")
        ref = _yang_at(ut, Fn)
        tol = _tol(ref, FLOOR_CW)
        rows.append(_row("SF1a", f"Fn {Fn}: double-body hull force C_lin vs U. Tokyo C_W (fixed; M {kw['M']}, NF {NF})", cl, ref,
                         "U.T., Kajitani et al. 1983, via Yang et al. 2000 fig 5a", f"max(20 %, {FLOOR_CW:.2e}) = {tol:.3e}",
                         abs(cl - ref) <= tol, note=f"ratio {cl / ref:.3f}; quadratic-pressure variant {cq * 1e3:.3f}e-3; "
                         f"{bnote}; {time.time() - t:.0f} s"))
        wp, fwp, ewp = _cwp_at(Fn)
        if wp is not None:
            tol = _tol(wp, FLOOR_CWP)
            rows.append(_row("SF1b", f"Fn {Fn}: double-body wave cut C_cut vs U. Tokyo C_WP (marker at Fn {fwp:.4f})", cc, wp,
                             "U.T. C_WP, Hendrix & Noblesse 1988 p. 61, read by rankine_wigley_scan.py",
                             f"max(20 %, {FLOOR_CWP:.2e}) = {tol:.3e}", abs(cc - wp) <= tol,
                             note=f"ratio {cc / wp:.3f}; near (C_lin)/cut {cl / cc - 1:+.3f}; {bnote}"))
        if Fn in (0.25, 0.267, 0.289, 0.316, 0.354, 0.408):
            xe_, ze_ = _profile_exp(Fn)
            zc = np.interp(xe_, out["x_wl"][np.argsort(out["x_wl"])], (2 * G0 * out["zeta_wl"] / U ** 2)[np.argsort(out["x_wl"])])
            rms = float(np.sqrt(np.mean((zc - ze_) ** 2)) / np.abs(ze_).max())
            crest = float((2 * G0 * out["zeta_wl"] / U ** 2)[out["x_wl"] > 0.25].max())
            ib = int(np.argmax(ze_ * (xe_ > 0.25)))
            rows.append(_row("SF2", f"Fn {Fn}: double-body wave profile along the hull, RMS(calc - exp)/max|exp|", rms, 0.0,
                             "U. Tokyo (WaveBEM tokyoFr*.csv)", 0.20, rms <= 0.20,
                             note=f"bow crest 2 g zeta/U^2 calc {crest:.3f} vs exp {ze_[ib]:.3f}"))
    if dom_fn is not None:
        U = dom_fn * np.sqrt(G0)
        q = 0.5 * RHO * U * U * S
        out_d = {}
        for lab, fac in (("x1", 1.0), ("x1.5", 1.5)):
            kw = steady_grid(h, U, G0)
            kw = dict(kw, **{k: fac * kw[k] for k in ("up", "down", "lat", "b_up", "b_dn", "b_lat")})
            s, co, G, db = build_db_solver(h, U, 0.0, 0.0, kw, (60, 12), g=G0, rho=RHO)
            out = steady_db(s, co, h)
            Rc, _ = steady_cut(s, G, out["zeta"], h, U, G0, RHO)
            out_d[lab] = (out["R_lin"] / q, Rc / q, s.NF)
            s.release()
            _free()
        rows.append(_row("SFd", f"steady Wigley Fn {dom_fn}: domain and beaches x 1.5 (information: is the low-Fn excess of "
                         "the wave cut a domain effect?)", float(abs(out_d["x1.5"][1] / out_d["x1"][1] - 1)), 0.0,
                         "information", "info", True,
                         note=f"C_lin x1 {out_d['x1'][0] * 1e3:.3f}e-3 (NF {out_d['x1'][2]}), x1.5 {out_d['x1.5'][0] * 1e3:.3f}e-3 "
                              f"(NF {out_d['x1.5'][2]}); C_cut x1 {out_d['x1'][1] * 1e3:.3f}e-3, x1.5 {out_d['x1.5'][1] * 1e3:.3f}e-3"))
    hs0 = h.hydrostatics()
    for Fn in fns_free:
        U = Fn * np.sqrt(G0)
        q = 0.5 * RHO * U * U * S
        kw = steady_grid(h, U, G0)
        t = time.time()
        r = db_free_equilibrium(h, U, kw, (60, 12), G0, RHO)
        s, co, G, db, out = r["pack"]
        s.release()
        _free()
        res_last = max(abs(v) for v in r["history"][-1]["res"])
        s_exp, t_exp, cw_exp = _yang_at(sk, Fn), _yang_at(tr, Fn), _yang_at(cwf, Fn)
        t_calc = float(np.sin(r["trim"]))
        cpf = out["R_lin"] / q
        rows.append(_row("SF0", f"Fn {Fn}: double-body free equilibrium residual after {len(r['history']) - 1} steps", res_last,
                         0.0, "design P-7", 1e-6, r["converged"], note=f"{time.time() - t:.0f} s"))
        rows.append(_row("SF3", f"Fn {Fn}: double-body free sinkage (mean reading)", r["sink"] / h.L, s_exp,
                         "U.T. via Yang et al. 2000 fig 4a", 0.15, abs(r["sink"] / h.L / s_exp - 1) <= 0.15,
                         note=f"ratio {r['sink'] / h.L / s_exp:.3f}"))
        rows.append(_row("SF4", f"Fn {Fn}: double-body free trim sin(theta) (> 0 bow up)", t_calc, t_exp,
                         "U.T. via Yang et al. 2000 fig 4a", "+-0.05 deg (8.7e-4)", abs(t_calc - t_exp) <= np.radians(0.05),
                         note=f"difference {np.degrees(t_calc - t_exp):+.3f} deg"))
        tol5 = _tol(cw_exp, FLOOR_CW)
        rows.append(_row("SF5a", f"Fn {Fn}: double-body free-model hull force C_lin vs U.T. C_W (free)", cpf, cw_exp,
                         "U.T. via Yang et al. 2000 fig 5a (free)", f"max(20 %, {FLOOR_CW:.2e}) = {tol5:.3e}",
                         abs(cpf - cw_exp) <= tol5, note=f"ratio {cpf / cw_exp:.3f}"))
    return rows


# =====================================================================
# validation (Journee 1992)
def _rao(co, V, hull, w0, beta=np.pi, forcing=True):
    key = (round(w0, 12), beta, forcing)
    cache = co.__dict__.setdefault("_rao_cache", {})
    if key not in cache:
        cache[key] = co.rad_exc(w0, beta, forcing=forcing)
    r, ex = cache[key]
    we = ex["omega_e"]
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
    T2 = r["T"][np.ix_([1, 2], [1, 2])]
    xi = np.linalg.solve(-we ** 2 * Mm + T2 + Cm, ex["X"][[1, 2]])
    return np.abs(xi[0]), np.abs(xi[1]) * hull.L / (2 * np.pi), ex


def validation(args):
    """V (validation; per point max(20 % of the value, 10 % of the series maximum), the B5 rules): Wigley I against
    Journee (1992) via J. Mar. Sci. Eng. 9:87 (Fn 0.2: A33', B33', A55', B55', heave/pitch excitation, RAOs) and Kara
    (2022) (Fn 0.3: B33'; A33' was selection data in B5 and stays information), double-body model (D and OT forms
    coincide). Excitation and RAOs with the incident-wave/base-flow forcing (consistent model); the simplified
    diffraction (no forcing) as information."""
    cfg = args[0] if args else PROD
    grid = dict(GRID_BASE, M=cfg["M"])
    hm = cfg["hull"]
    h = _wj()
    V = _V(h)
    rows = []
    src2 = "Journee 1992 via J. Mar. Sci. Eng. 9:87 (2021) figs 12-16 (secondary transcription)"
    src3 = "Journee 1992 via Kara (2022) fig 9 (secondary transcription)"
    for Fn in (0.2, 0.3):
        s, co, _ = _db(h, Fn * np.sqrt(G0 * L), grid, hm)
        names = ("A33", "B33", "A55", "B55") if Fn == 0.2 else ("A33", "B33")
        fsuf = "020" if Fn == 0.2 else "030"
        for nm in names:
            d = _load(f"journee1992_wigley1_fn{fsuf}_{nm}.csv")
            i, j = {"33": (1, 1), "55": (2, 2)}[nm[1:]]
            q = 0 if nm[0] == "A" else 1
            calc = [_nd(co.radiation(wp * SC)["T"], wp * SC, V)[q][i, j] for wp in d[:, 0]]
            sel = Fn == 0.3 and nm == "A33"
            rows += _vrows("Vd", f"Fn {Fn} {nm}' (double body) vs exp, omega'" + (" [B5 selection data, information]" if sel
                                                                               else ""), d[:, 0], np.array(calc), d[:, 1],
                           src2 if Fn == 0.2 else src3, binding=not sel)
        if Fn == 0.2:
            for nm, jj, scale in (("F3", 1, lambda X: abs(X) * L / (RHO * G0 * V)), ("F5", 2, lambda X: abs(X) / (RHO * G0 * V))):
                d = _load(f"journee1992_wigley1_fn020_{nm}.csv")
                calc, calc0 = [], []
                for nu in d[:, 0]:
                    w0 = np.sqrt(G0 * nu / L)
                    calc.append(scale(_rao(co, V, h, w0)[2]["X"][jj]))
                    calc0.append(scale(_rao(co, V, h, w0, forcing=False)[2]["X"][jj]))
                rows += _vrows("Vd", f"Fn 0.2 {nm}' excitation (double body, with the interaction forcing) vs exp, nu' = kL",
                               d[:, 0], np.array(calc), d[:, 1], src2)
                rows.append(_row("Vdi", f"Fn 0.2 {nm}' excitation without the interaction forcing (simplified diffraction; "
                                 "information)", 0.0, 0.0, "information", "info", True,
                                 note="ratios " + ", ".join(f"{c / e:.3f}" for c, e in zip(calc0, d[:, 1]))))
            for nm, jj in (("RAO3", 0), ("RAO5", 1)):
                d = _load(f"journee1992_wigley1_fn020_{nm}.csv")
                calc = [_rao(co, V, h, np.sqrt(G0 * 2 * np.pi / (lam * L)))[jj] for lam in d[:, 0]]
                rows += _vrows("Vd", f"Fn 0.2 {nm} (double body; KG 0.170 m assumed) vs exp, lambda/L", d[:, 0], np.array(calc),
                               d[:, 1], src2)
        s.release()
        _free()
    return rows


# =====================================================================
# free decay of the plant law and the first planing-type table
def _decay(tab, hull, label, statics_kw=None):
    """CD / CDw (binding; criterion corrected after the B6 review, finding B4, stage note 12.1): at every node of the
    table, (i) STATIC: every eigenvalue of (M + A_inf)^-1 (C_hs + C0 + C_s) has a positive real part (no divergence;
    doublebody.static_stability; a complex pair -- circulatory coupling of the undamped system -- is reported and left
    to the trajectory; the B6 row used the minimum eigenvalue of the SYMMETRIC part of C_hs + C0, sufficient but not
    necessary for a non-symmetric stiffness, and omitted C_s); (ii) DYNAMIC: the free decay of the documented plant
    law (tables.free_decay, now with C0 + C_s) stays finite over t' 200 and decays in the positive-definite norm Ep =
    1/2 v.(M + A_inf)_sym v + 1/2 x.C_hs x: Ep(end) <= Ep(0) and max Ep over the last quarter <= max(max over the third
    quarter, 1e-8 Ep(0)) (the B6 rule used E with the indefinite symmetric part of the total stiffness). The largest
    transient Ep/Ep(0), the B6 criterion and the passivity of B(omega) are information in the note."""
    from hydro.rankine.tables import interp_slice, free_decay
    from hydro.rankine.doublebody import static_stability
    Mm, Ch = _heave_pitch_statics(hull, tab.ref, **(statics_kw or {}))
    res, ok_all = [], True
    for iu, u in enumerate(tab.U):
        for i_s, sk in enumerate(tab.sink):
            for i_t, tr in enumerate(tab.trim):
                sl = interp_slice(tab, u, sk, tr, form="D")
                C0 = np.asarray(sl["C0"])[1:, 1:]
                Cs = np.asarray(sl["C_s"])[1:, 1:]
                Ct = Ch + C0 + Cs
                st = static_stability(Mm + np.asarray(sl["A_inf"])[1:, 1:], Ct)
                lam_b6 = float(np.linalg.eigvalsh(0.5 * ((Ch + C0) + (Ch + C0).T)).min())
                with np.errstate(over="ignore", invalid="ignore"):
                    out = free_decay(sl, Mm, Ch, np.array([0.005 * hull.L, 0.02]), 0.02 / SC, 200.0 / SC)
                E = out["Ep"]
                n = len(E)
                q3 = np.nanmax(E[n // 2: 3 * n // 4]) if n > 4 else np.nan
                q4 = np.nanmax(E[3 * n // 4:]) if n > 4 else np.nan
                dyn = bool(np.all(np.isfinite(E)) and len(E) == int(round(200.0 / 0.02)) + 1 and E[-1] <= E[0]
                           and q4 <= max(q3, 1e-8 * E[0]))
                ok = bool(st["stable"] and dyn)
                ok_all &= ok
                pas = (f", bsym_min {float(tab.bsym_min[iu, i_s, i_t]):+.3f}" if tab.bsym_min is not None else "")
                res.append(f"Fn {u / SC:.2f}: eig((M+A_inf)^-1 C_tot) {np.round(st['eig'], 5).tolist()} "
                           f"({'static OK' if st['stable'] else 'static FAIL'}{', circulatory' if st['circulatory'] else ''}), "
                           f"det {st['det']:.3e}; Ep max/Ep0 "
                           f"{np.nanmax(E) / E[0]:.3f}, Ep(end)/Ep0 {E[-1] / E[0]:.1e} ({'decays' if dyn else 'NO decay'}); "
                           f"B6 criterion (C0 only, sym part) {lam_b6:+.2e}; C_s diag {np.round(np.diag(Cs), 5).tolist()}"
                           f"{pas}{'' if ok else ' FAIL'}")
    return _row("CD", f"{label}: static stability (eigenvalues of (M + A_inf)^-1 (C_hs + C0 + C_s)) and free decay of the "
                "plant law (double body; D = OT), all " + f"{len(res)} nodes", float(not ok_all), 0.0,
                "plant law (10.8.5 with C_s, 12.2 items 4-5)", "all nodes pass", ok_all, note="; ".join(res))


def decay_wigley(args):
    """CDw (binding): radiation table of the Wigley I (Fn 0.2, 0.3, zero attitude, double body) and the free decay of
    the documented plant law at both nodes (rules of B5 10.8.6 and its amendment)."""
    grid, hm = args
    from hydro.rankine.tables import build_radiation_table
    h = _wj()
    wp = np.concatenate([np.exp(np.linspace(np.log(0.05), np.log(1.5), 12)), np.exp(np.linspace(np.log(1.5), np.log(40.0), 29))[1:]])
    t = np.arange(0, 30 + 1e-9, 0.1) / SC
    tab = build_radiation_table(h, np.array([0.2, 0.3]) * SC, [0.0], [0.0], wp * SC, t, grid, hm, rho=RHO, g=G0,
                                ref=(0.0, ZG), linearisation="DB")
    return [_decay(tab, h, "Wigley I double-body table (Fn 0.2, 0.3)", statics_kw=dict(xg=0.0, zg=ZG))]


def _passivity_note(tab):
    """omega' ranges where the symmetric part of B(omega) (D-form, surge-heave-pitch) has a negative eigenvalue, per
    node; the most negative relative eigenvalue and whether it lies in the validity band."""
    out = []
    w = tab.omega / SC
    for iu, u in enumerate(tab.U):
        B = tab.B[iu, 0, 0]
        Bs = 0.5 * (B + np.swapaxes(B, 1, 2))
        ev = np.linalg.eigvalsh(Bs)
        rel = ev[:, 0] / np.maximum(np.abs(ev).max(axis=1), 1e-300)
        neg = rel < -1e-9
        lo, hi = tab.omega_lo[iu, 0, 0] / SC, tab.omega_hi[iu, 0, 0] / SC
        runs, k = [], 0
        while k < len(w):
            if neg[k]:
                k0 = k
                while k + 1 < len(w) and neg[k + 1]:
                    k += 1
                runs.append(f"{w[k0]:.2f}-{w[k]:.2f}")
            k += 1
        kmin = int(np.argmin(rel))
        out.append(f"Fn {u / SC:.2f}: negative at omega' {', '.join(runs) or 'none'}; min {rel[kmin]:+.3f} at {w[kmin]:.2f} "
                   f"(band {lo:.2f}-{hi:.2f}: {'inside' if lo <= w[kmin] <= hi else 'outside'}; in-band min "
                   f"{(rel[(w >= lo) & (w <= hi)].min() if np.any((w >= lo) & (w <= hi)) else float('nan')):+.3f})")
    return "; ".join(out)


def planing_table(args):
    """PV (interface): first double-body radiation table of the planing-type prismatic V body (flared), zero attitude,
    speeds Fn 0.4, 0.6, 0.8, 1.0, 40 frequencies omega' 0.05-40, kernels to t' 30: save/load bit-exact (binding),
    node interpolation exact (binding 1e-12); PVs (binding, B5 PVm rule): mid-node slices Fn 0.5, 0.7, 0.9 computed
    directly vs interpolated, A_inf within 1 % of max |A_inf| and kernels within 10 % rms; PVd (information): the same
    interpolation error with half the spacing (Fn 0.4-0.5 at 0.45) -- the speed-node density estimate; CD free decay at
    every node (binding)."""
    grid, hm, label = args
    from hydro.rankine.tables import build_radiation_table, save_table, load_table, interp_slice, table_path, node_slice
    h = _prism("flared")
    U = np.array([0.4, 0.6, 0.8, 1.0]) * SC
    wp = np.concatenate([np.exp(np.linspace(np.log(0.05), np.log(1.5), 12)), np.exp(np.linspace(np.log(1.5), np.log(40.0), 29))[1:]])
    t = np.arange(0, 30 + 1e-9, 0.1) / SC
    t0 = time.time()
    tab = build_radiation_table(h, U, [0.0], [0.0], wp * SC, t, grid, hm, rho=RHO, g=G0, ref=(0.0, 0.0), linearisation="DB")
    tb = time.time() - t0
    code = hashlib.sha256(open(os.path.join(ROOT, "hydro", "rankine", "doublebody.py"), "rb").read()).hexdigest()[:12]
    path = table_path(OUT, h.name + "_DB", code, label + datetime.datetime.now().strftime("_%Y%m%dT%H%M%S"))
    sha = save_table(tab, path)
    tab2 = load_table(path)
    same = all(np.array_equal(getattr(tab, k), getattr(tab2, k), equal_nan=np.issubdtype(np.asarray(getattr(tab, k)).dtype,
                                                                                   np.floating))
               for k in tab.ARRAYS if getattr(tab, k) is not None)
    ei = 0.0
    for iu, u in enumerate(U):
        sl = interp_slice(tab2, u, 0.0, 0.0, form="D")
        ei = max(ei, float(np.abs(sl["A_inf"] - tab.A_inf[iu, 0, 0]).max()), float(np.abs(sl["K"] - tab.K[iu, 0, 0]).max()))
    rows = [_row("PV", f"double-body planing table {os.path.relpath(path, ROOT)} ({tb:.0f} s): save/load bit-exact",
                 float(not same), 0.0, "exact", 0.0, same, note=f"sha256 {sha}"),
            _row("PV", "planing table: interpolation reproduces the node values", ei, 0.0, "exact", 1e-12, ei <= 1e-12)]
    notes, worst = [], 0.0
    for fm in (0.5, 0.7, 0.9):
        nd = node_slice(h, fm * SC, 0.0, 0.0, wp * SC, t, grid, hm, rho=RHO, g=G0, ref=(0.0, 0.0), linearisation="DB")
        sl = interp_slice(tab2, fm * SC, 0.0, 0.0, form="D")
        ea = float(np.abs(sl["A_inf"] - nd["A_inf"]).max() / np.abs(nd["A_inf"]).max())
        ek = float(np.sqrt(np.mean((sl["K"] - nd["K"]) ** 2)) / np.sqrt(np.mean(nd["K"] ** 2)))
        worst = max(worst, ea / 0.01, ek / 0.10)
        notes.append(f"Fn {fm}: A_inf {ea:.2e}, K rms {ek:.3f}")
    rows.append(_row("PVs", "planing table: mid-node slices (Fn 0.5, 0.7, 0.9) interpolated vs computed directly: A_inf (1 % "
                     "of max) and kernel rms (10 %)", worst, 1.0, "direct node_slice", "<= 1", worst <= 1.0,
                     note="; ".join(notes)))
    nd45 = node_slice(h, 0.45 * SC, 0.0, 0.0, wp * SC, t, grid, hm, rho=RHO, g=G0, ref=(0.0, 0.0), linearisation="DB")
    nd50 = node_slice(h, 0.5 * SC, 0.0, 0.0, wp * SC, t, grid, hm, rho=RHO, g=G0, ref=(0.0, 0.0), linearisation="DB")
    K45 = 0.5 * (tab.K[0, 0, 0] + nd50["K"])
    A45 = 0.5 * (tab.A_inf[0, 0, 0] + nd50["A_inf"])
    ea = float(np.abs(A45 - nd45["A_inf"]).max() / np.abs(nd45["A_inf"]).max())
    ek = float(np.sqrt(np.mean((K45 - nd45["K"]) ** 2)) / np.sqrt(np.mean(nd45["K"] ** 2)))
    rows.append(_row("PVd", "speed-node density: interpolation error at Fn 0.45 with nodes 0.4/0.5 (half spacing; information)",
                     ek, 0.0, "direct node_slice", "info", True, note=f"A_inf {ea:.2e}, K rms {ek:.3f}"))
    rows.append(_row("PVp", "planing table: where B(omega) is not passive (information, B6 review): omega' ranges with "
                     "lambda_min(B_sym) < 0 per node, and the share inside the validity band", 0.0, 0.0, "information", "info",
                     True, note=_passivity_note(tab)))
    rows.append(_row("PV", "planing table: validity band, kernel window, passivity (information)", 0.0, 0.0, "information",
                     "info", True, note=f"band omega' {[(round(a / SC, 2), round(b / SC, 2)) for a, b in zip(tab.omega_lo.ravel(), tab.omega_hi.ravel())]}; "
                     f"bsym_min {np.round(tab.bsym_min.ravel(), 3).tolist()}; passive {tab.passive.ravel().tolist()}; "
                     f"A_inf diag per node {[np.round(np.diag(tab.A_inf[i, 0, 0]), 6).tolist() for i in range(len(U))]}"))
    rows.append(_decay(tab, h, "planing table (prismatic V flared, double body)"))
    return rows


# =====================================================================
# B6 review (stage note 12): operator consistency with exact base flows
def _mms(X, Y):
    """Manufactured field of DXD (a = b = 2 pi/0.25 L) and its exact gradient."""
    a = b = 2 * np.pi / 0.25
    wx = np.exp(-(X / 0.9) ** 8)
    wy = np.exp(-(Y / 0.5) ** 8)
    f = np.cos(a * X) * np.cos(b * Y) * wx * wy
    fx = (-a * np.sin(a * X) * wx + np.cos(a * X) * wx * (-8 * X ** 7 / 0.9 ** 8)) * np.cos(b * Y) * wy
    fy = np.cos(a * X) * wx * (-b * np.sin(b * Y) * wy + np.cos(b * Y) * wy * (-8 * Y ** 7 / 0.5 ** 8))
    return f, fx, fy, a


def _orders(res, Ms, keys):
    """Observed orders between the two finest members (errors below 1e-12 count as exact: order +inf)."""
    out = {}
    for k in keys:
        e1, e2 = res[Ms[-2]][k], res[Ms[-1]][k]
        if e2 <= 1e-12:
            out[k] = float("inf")
        elif e1 <= 1e-12:
            out[k] = float("-inf")
        else:
            out[k] = float(np.log(e1 / e2) / np.log(Ms[-1] / Ms[-2]))
    return out


def dxa_check(args):
    """DXa (M2, binding; B6 review, stage note 12.1): the production operators A (advective) and K = Wa^-1 A_{-U}^T
    Wa (conservative) with the EXACT double-body flow of the half prolate spheroid a 0.5, b 0.1 (its double body is
    the full spheroid: analytic.spheroid_double_body, W and Phi_zz in closed form, tangent to the waterline) on the
    CHECK band grids M 20/40/80 (quick 20/40) of hull.half_spheroid: errors of A f vs W.grad f and K f vs
    div_h(W f) = W.grad f - Phi_zz f (manufactured f of DXD, relative to a max|W|) and of K 1 vs div_h W = -Phi_zz
    (relative to max|Phi_zz| in the window), in three regions: near wall (band lines 0-2 over |x| <= 0.6 a), band
    (lines 3 .. J-1 over |x| <= 0.6 a; the blunt waterline ends are excluded, their elliptic waterline has a vertical
    tangent), lattice (lines >= J, |x| < 0.8, y < 0.45: band/lattice junction, lateral stretching start, column
    stretching at the waterline ends). Binding: observed order between the two finest members >= 1, >= 1, >= 1.5
    (the DXD thresholds). No panel representation is involved: the rows test the operators alone."""
    Ms, = args
    from hydro.rankine.hull import half_spheroid
    from hydro.rankine.analytic import spheroid_double_body
    from hydro.rankine.unsteady import build_grid
    from hydro.rankine.doublebody import fs_operators
    a_, b_, U = 0.5, 0.1, 1.0
    h = half_spheroid(a_, b_)
    regs = ("near wall (lines 0-2)", "band (lines 3..J-1)", "lattice (lines >= J)")
    res = {}
    for M in Ms:
        G = build_grid(h, 0.0, 0.0, 1.0, **dict(CHECK_GRID, M=M))
        Wa, pz, _ = spheroid_double_body(a_, b_, U, G["fs"].c)
        A, K, _ = fs_operators(G, Wa[:, :2], 1.0, "cons")
        nx, ny = G["shape"]
        X, Y = G["Xc"], G["Yc"]
        mat = lambda v: v.reshape(ny, nx).T
        vec = lambda F: F.T.reshape(-1)
        f, fx, fy, kk = _mms(X, Y)
        Wx, Wy, P2 = mat(Wa[:, 0]), mat(Wa[:, 1]), mat(pz)
        exA = Wx * fx + Wy * fy
        exK = exA - P2 * f
        sc = kk * np.abs(Wa[:, :2]).max()
        i_s, i_b, J = G["i_s"], G["i_b"], G["J"]
        jj = np.arange(ny)[None, :] * np.ones((nx, 1), int)
        hull = np.zeros((nx, ny), bool)
        hull[i_s:i_b, :] = True
        win = np.abs(X) <= 0.6 * a_
        masks = {regs[0]: hull & win & (jj <= 2), regs[1]: hull & win & (jj >= 3) & (jj < J),
                 regs[2]: (jj >= J) & (np.abs(X) < 0.8) & (Y < 0.45)}
        sc1 = np.abs(P2[(np.abs(X) < 0.8) & (Y < 0.45)]).max()
        EA = np.abs(mat(A @ vec(f)) - exA) / sc
        EK = np.abs(mat(K @ vec(f)) - exK) / sc
        E1 = np.abs(mat(K @ np.ones(nx * ny)) + P2) / sc1
        res[M] = {(q, r): float(E[m].max()) for q, E in (("A", EA), ("K", EK), ("K1", E1)) for r, m in masks.items()}
    keys = [(q, r) for q in ("A", "K", "K1") for r in regs]
    od = _orders(res, Ms, keys)
    rows = []
    thr = {regs[0]: 1.0, regs[1]: 1.0, regs[2]: 1.5}
    for (q, r) in keys:
        lab = {"A": "advective A f", "K": "conservative K f", "K1": "conservative K 1 (= div_h W)"}[q]
        rows.append(_row("DXa", f"exact half-spheroid double-body flow, {lab} vs exact, {r}: observed order M {Ms[-2]} -> "
                         f"{Ms[-1]}", od[(q, r)], 0.0, "closed-form flow (analytic.spheroid_double_body)",
                         f">= {thr[r]}", od[(q, r)] >= thr[r],
                         note="errors " + " / ".join(f"M{M} {res[M][(q, r)]:.2e}" for M in Ms)))
    return rows


def dxb_check(args):
    """DXb (information; B6 review): the panel representation's base flow (DoubleBodyFlow of the half spheroid,
    hull meshes 40 x 8, 60 x 12, 90 x 18) against the exact flow at the centroids of the CHECK grid M 40 in the regions
    of DXa: max |W_rep - W| / U and max |Phi_zz,rep - Phi_zz| / max|Phi_zz| -- the base-flow error that the B6 DXD
    rows mixed with the operator errors."""
    from hydro.rankine.hull import half_spheroid
    from hydro.rankine.analytic import spheroid_double_body
    from hydro.rankine.unsteady import build_grid
    from hydro.rankine.doublebody import DoubleBodyFlow
    a_, b_, U = 0.5, 0.1, 1.0
    h = half_spheroid(a_, b_)
    G = build_grid(h, 0.0, 0.0, 1.0, **dict(CHECK_GRID, M=40))
    nx, ny = G["shape"]
    X, Y = G["Xc"], G["Yc"]
    i_s, i_b, J = G["i_s"], G["i_b"], G["J"]
    jj = np.arange(ny)[None, :] * np.ones((nx, 1), int)
    hull = np.zeros((nx, ny), bool)
    hull[i_s:i_b, :] = True
    win = np.abs(X) <= 0.6 * a_
    masks = {"line 0": hull & win & (jj == 0), "lines 1-2": hull & win & (jj >= 1) & (jj <= 2),
             "band 3..J-1": hull & win & (jj >= 3) & (jj < J), "lattice": (jj >= J) & (np.abs(X) < 0.8) & (Y < 0.45)}
    sel = np.zeros((nx, ny), bool)
    for m in masks.values():
        sel |= m
    k_of = (np.arange(ny)[None, :] * nx + np.arange(nx)[:, None])
    ks = k_of[sel]
    P = G["fs"].c[ks]
    We, pze, _ = spheroid_double_body(a_, b_, U, P)
    notes, worst = [], 0.0
    for hm in ((40, 8), (60, 12), (90, 18)):
        db = DoubleBodyFlow(h.mesh(*hm), U, hm)
        Wr = db.field_velocity(P)
        pr = db.field_phizz(P)
        del db
        _free()
        eW = np.full((nx, ny), np.nan)
        eP = np.full((nx, ny), np.nan)
        eW[sel] = np.linalg.norm(Wr - We[:, :2], axis=1) / U
        eP[sel] = np.abs(pr - pze) / np.abs(pze).max()
        parts = [f"{r}: W {np.nanmax(eW[m]):.2e}, Phi_zz {np.nanmax(eP[m]):.2e}" for r, m in masks.items()]
        worst = max(worst, float(np.nanmax(eW[masks['line 0']])))
        notes.append(f"hull {hm}: " + "; ".join(parts))
    return [_row("DXb", "panel-representation base flow vs the exact half-spheroid flow on the CHECK grid M 40 "
                 "(information): max |W_rep - W|/U on band line 0", worst, 0.0, "closed-form flow", "info", True,
                 note=" | ".join(notes))]


def dxu_check(args):
    """DXu (M2, binding; B6 review finding B2, stage note 12.1): A and K with the EXACT uniform stream W = (-1, 0)
    (div_h W = 0, Phi_zz = 0; exact A f = K f = -f_x, K 1 = 0) on the CHECK band grids M 20/40/80 (quick 20/40) of
    the Wigley I and the flared prismatic body, away from the hull wall (the uniform stream crosses the curved wall;
    band lines 0-2 over the hull, whose stencils reach the mirror ghosts, are excluded): region "non-smooth lines" =
    the cells within 4 of a grid non-smoothness edge (the stencil reach of the six-point formulas is 3): the
    waterline-end kinks of the band lines and the start of the column stretching (edges i_s, i_b, all lines), the
    band/lattice junction (edge J over the hull columns) and the start of the lateral stretching, inside the window
    |x| < 0.8, y < 0.45; region "smooth" = the rest of the window. Binding: observed order between the two finest members >= 1 for A f, K f and K 1 in both
    regions (errors below 1e-12 count as exact)."""
    Ms, = args
    from hydro.rankine.hull import prismatic_v
    from hydro.rankine.unsteady import build_grid
    from hydro.rankine.doublebody import fs_operators
    rows = []
    regs = ("non-smooth lines", "smooth")
    for hname in ("wigleyJ", "prismV"):
        h = _wj() if hname == "wigleyJ" else prismatic_v()
        res = {}
        for M in Ms:
            G = build_grid(h, 0.0, 0.0, 1.0, **dict(CHECK_GRID, M=M))
            nx, ny = G["shape"]
            X, Y = G["Xc"], G["Yc"]
            A, K, _ = fs_operators(G, np.tile([-1.0, 0.0], (nx * ny, 1)), 1.0, "cons")
            mat = lambda v: v.reshape(ny, nx).T
            vec = lambda F: F.T.reshape(-1)
            f, fx, fy, kk = _mms(X, Y)
            i_s, i_b, J = G["i_s"], G["i_b"], G["J"]
            jj = np.arange(ny)[None, :] * np.ones((nx, 1), int)
            ii = np.arange(nx)[:, None] * np.ones((1, ny), int)
            hull = (ii >= i_s) & (ii < i_b)
            win = (np.abs(X) < 0.8) & (Y < 0.45)
            wall = hull & (jj <= 2)
            kink = (np.abs(ii - (i_s - 0.5)) <= 3.5) | (np.abs(ii - (i_b - 0.5)) <= 3.5)
            junc = hull & (np.abs(jj - (J - 0.5)) <= 3.5)
            yl = G["ylines"]
            dyl = np.diff(yl)
            jt = int(np.argmax(np.abs(dyl[1:] / dyl[:-1] - 1) > 1e-6) + 1) if len(dyl) > 2 else 0   # first stretched line
            lat = np.abs(jj - (jt - 0.5)) <= 3.5
            nons = (kink | junc | lat) & win & ~wall
            smooth = win & ~wall & ~nons
            EA = np.abs(mat(A @ vec(f)) + fx) / kk
            EK = np.abs(mat(K @ vec(f)) + fx) / kk
            E1 = np.abs(mat(K @ np.ones(nx * ny)))
            res[M] = {(q, r): float(E[m].max()) for q, E in (("A", EA), ("K", EK), ("K1", E1))
                      for r, m in ((regs[0], nons), (regs[1], smooth))}
        keys = [(q, r) for q in ("A", "K", "K1") for r in regs]
        od = _orders(res, Ms, keys)
        for (q, r) in keys:
            lab = {"A": "advective A f", "K": "conservative K f", "K1": "conservative K 1 (exact 0)"}[q]
            rows.append(_row("DXu", f"exact uniform stream, {hname} CHECK grid, {lab}, {r}: observed order M {Ms[-2]} -> "
                             f"{Ms[-1]}", od[(q, r)], 0.0, "exact (uniform stream)", ">= 1", od[(q, r)] >= 1.0,
                             note="errors " + " / ".join(f"M{M} {res[M][(q, r)]:.2e}" for M in Ms)))
    return rows


# =====================================================================
# B6 review: frozen base-pressure stiffness, C0B, static stability
def cs_c0b_check(args):
    """CS (M2, binding 2 %; B6 review finding B3, stage note 12.1): the rigid-lid base-flow stiffness two ways, K_db
    = -dF_db/dq by central differences (delta 1e-3) of the re-panelled base-pressure force (doublebody.
    base_pressure_stiffness) against C0_lid + C_s with C_s from the frozen-field formula (doublebody.
    frozen_pressure_stiffness, production), (33) and (55) relative to K_db, hull 60 x 12, Fn 0.3: the wall-sided Wigley
    I (about (0, z_G)) and the FLARED Wigley (hull.wigley_flare with its own form continued above the waterline,
    above="form", about (0, 0)), where the waterline-strip term of the formula is active. C_s itself (FD vs formula,
    relative to C_s) is information. C0B (binding, re-registered after the review):
    the plant-law static stiffness C0 + C_s (D-form) vs the steady-force slope K_full = -dF_steady/dq (steady_db,
    central differences +-1e-3 L / rad, hull re-panelled, grid and base flow rebuilt), each heave-pitch entry relative
    to the DYNAMIC stiffness (|K_full_jj| on the diagonal, sqrt(|K_full_33 K_full_55|) off it), tolerance 5 %
    (the B5 10.8.5 value; B6 compared C0 alone relative to the hydrostatic stiffness). C0B-B6 (information): the B6
    measure for the record."""
    from hydro.rankine.doublebody import (steady_db, frozen_pressure_stiffness, base_pressure_stiffness,
                                          attitude_of_modes)
    from hydro.rankine.tables import static_stiffness
    hj = _wj()
    U = 0.3 * np.sqrt(G0 * L)
    grid = dict(GRID_BASE, M=40)
    hm = COARSE["hull"]
    ref = (0.0, ZG)
    s, co, _ = _db(hj, U, grid, hm)
    C0, _ = static_stiffness(co)
    Csf = frozen_pressure_stiffness(co.db, co.ref, RHO)
    s.release()
    _free()
    bp = base_pressure_stiffness(hj, U, 0.0, 0.0, hm, ref=ref, rho=RHO, delta=1e-3)
    Csd = bp["C_s"]
    e_cs = [abs(Csf[k, k] - Csd[k, k]) / abs(Csd[k, k]) for k in (1, 2)]
    e_kd = [abs((bp["C0_lid"] + Csf)[k, k] - bp["K_db"][k, k]) / abs(bp["K_db"][k, k]) for k in (1, 2)]
    from hydro.rankine.hull import wigley_flare
    from hydro.rankine.doublebody import DoubleBodyFlow
    hf = wigley_flare()
    hf.above = "form"
    bpf = base_pressure_stiffness(hf, U, 0.0, 0.0, hm, ref=(0.0, 0.0), rho=RHO, delta=1e-3)
    dbf = DoubleBodyFlow(hf.mesh(*hm), U, hm)
    Csff = frozen_pressure_stiffness(dbf, (0.0, 0.0), RHO)
    del dbf
    _free()
    e_kdf = [abs((bpf["C0_lid"] + Csff)[k, k] - bpf["K_db"][k, k]) / abs(bpf["K_db"][k, k]) for k in (1, 2)]
    e_csf = [abs(Csff[k, k] - bpf["C_s"][k, k]) / abs(bpf["C_s"][k, k]) for k in (1, 2)]
    F = {}
    d = 1e-3
    for key, (x3, x5) in (("z+", (d, 0.0)), ("z-", (-d, 0.0)), ("t+", (0.0, d)), ("t-", (0.0, -d)), ("0", (0.0, 0.0))):
        sk, tr = attitude_of_modes(hj, 0.0, 0.0, x3, x5, ref)
        s2, co2, _ = _db(hj, U, grid, hm, sink=sk, trim=tr)
        F[key] = steady_db(s2, co2, hj, sk, tr, cg=ref)
        s2.release()
        _free()
    Kf = np.array([[-(F["z+"]["Fz"] - F["z-"]["Fz"]) / (2 * d), -(F["t+"]["Fz"] - F["t-"]["Fz"]) / (2 * d)],
                   [-(F["z+"]["My"] - F["z-"]["My"]) / (2 * d), -(F["t+"]["My"] - F["t-"]["My"]) / (2 * d)]])
    Cd = (C0 + Csf)[1:, 1:]
    scl = np.array([[abs(Kf[0, 0]), np.sqrt(abs(Kf[0, 0] * Kf[1, 1]))], [np.sqrt(abs(Kf[0, 0] * Kf[1, 1])), abs(Kf[1, 1])]])
    dev = np.abs(Cd - Kf) / scl
    hs = hj.hydrostatics()
    zb = hs["centroid"][2]
    C33 = RHO * G0 * hs["Aw"]
    C55 = RHO * G0 * (hs["IL"] + hs["Aw"] * hs["xf"] ** 2 + hs["V"] * (zb - ZG))
    sch = np.array([[C33, np.sqrt(C33 * C55)], [np.sqrt(C33 * C55), C55]])
    Kone = np.array([[-(F["z+"]["Fz"] - F["0"]["Fz"]) / d, -(F["t+"]["Fz"] - F["0"]["Fz"]) / d],
                     [-(F["z+"]["My"] - F["0"]["My"]) / d, -(F["t+"]["My"] - F["0"]["My"]) / d]])
    dev_b6 = np.abs(C0[1:, 1:] - Kone) / sch
    return [
        _row("CS", "base-flow stiffness two ways, Fn 0.3, hull 60 x 12: re-panelled central differences K_db vs C0_lid + "
             "C_s (frozen-field formula), max of (33), (55) relative to K_db, Wigley I and flared Wigley",
             float(max(e_kd + e_kdf)), 0.0, "two computations of the same continuum quantity (12.2 item 4)", TOL_GS,
             max(e_kd + e_kdf) <= TOL_GS,
             note=f"Wigley I (33, 55) {e_kd[0]:.2e} {e_kd[1]:.2e}, flared {e_kdf[0]:.2e} {e_kdf[1]:.2e}; C_s itself "
                  f"(formula vs FD, relative to C_s): Wigley I {e_cs[0]:.3f} {e_cs[1]:.3f}, flared {e_csf[0]:.3f} "
                  f"{e_csf[1]:.3f}; Wigley I C_s frozen (33, 35, 53, 55) {Csf[1, 1]:.5e} {Csf[1, 2]:.3e} {Csf[2, 1]:.3e} "
                  f"{Csf[2, 2]:.5e}, FD {Csd[1, 1]:.5e} {Csd[1, 2]:.3e} {Csd[2, 1]:.3e} {Csd[2, 2]:.5e}; C_s/C_hs (33, 55) "
                  f"{Csf[1, 1] / C33:.4f} {Csf[2, 2] / C55:.4f}; flared C_s frozen (33, 55) {Csff[1, 1]:.4e} {Csff[2, 2]:.4e}"),
        _row("C0B", "plant-law static stiffness C0 + C_s vs the steady-force slope -dF_steady/dq (central differences), "
             "Wigley I Fn 0.3, relative to the dynamic stiffness (re-registered after the B6 review)", float(dev.max()), 0.0,
             "steady_db slope (12.2 item 4)", TOL_C0B, dev.max() <= TOL_C0B,
             note=f"(33) {dev[0, 0]:.3f} (35) {dev[0, 1]:.3f} (53) {dev[1, 0]:.3f} (55) {dev[1, 1]:.3f}; C0 + C_s "
                  f"{Cd[0, 0]:.4e} {Cd[0, 1]:.3e} {Cd[1, 0]:.3e} {Cd[1, 1]:.4e}; -dF/dq {Kf[0, 0]:.4e} {Kf[0, 1]:.3e} "
                  f"{Kf[1, 0]:.3e} {Kf[1, 1]:.4e}; C0 {C0[1, 1]:.4e} {C0[1, 2]:.3e} {C0[2, 1]:.3e} {C0[2, 2]:.4e}; "
                  f"relative to C_hs: {np.abs(Cd - Kf)[0, 0] / C33:.4f} (33), {np.abs(Cd - Kf)[1, 1] / C55:.4f} (55)"),
        _row("C0B", "B6 measure for the record: C0 alone vs the one-sided steady slope, relative to the hydrostatic "
             "stiffness (information)", float(dev_b6.max()), 0.0, "information", "info", True,
             note=f"(33) {dev_b6[0, 0]:.3f} (35) {dev_b6[0, 1]:.3f} (53) {dev_b6[1, 0]:.3f} (55) {dev_b6[1, 1]:.3f}")]


def _static_rows(tag, label, Mm, Ct):
    from hydro.rankine.doublebody import static_stability
    st = static_stability(Mm, Ct)
    return st, (f"eig(M^-1 C) {np.round(st['eig'], 5).tolist()}, eig(C) {np.round(st['eig_C'], 5).tolist()}, det "
                f"{st['det']:.3e}, sym-part min {st['sym_min']:+.3e}")


def planing_stiffness_series(args):
    """CDs (binding; B6 review finding B4, stage note 12.1): static stiffness of the plant law at the planing-table
    nodes (flared prismatic body, zero attitude, Fn 0.4, 0.6, 0.8, 1.0; moments about (0, 0)) on the grid series
    (GRID_BASE with u_dn 0.3, u_lat 0.2) M 30 / hull 30x6, M 45 / 45x9, M 60 / 60x12: C_tot = C_hs + C0 + C_s
    (heave-pitch), eigenvalues of (M + A_inf)^-1 C_tot (static_stability: positive real parts = statically stable,
    no divergence; complex pairs reported as circulatory coupling).
    Binding: at every node the static verdict is the same on the three grids (the conclusion is grid-independent).
    Information per node: the eigenvalues per grid, C0 + C_s, the verdict of the B6 criterion (C0 alone,
    symmetric part)."""
    from hydro.rankine.doublebody import frozen_pressure_stiffness, static_stability
    from hydro.rankine.tables import static_stiffness
    fns, members = args
    h = _prism("flared")
    Mm, Ch = _heave_pitch_statics(h, (0.0, 0.0))
    base = dict(GRID_BASE, u_dn=0.3, u_lat=0.2)
    rows, all_ok = [], True
    for Fn in fns:
        verdicts, notes = [], []
        for M, hm in members:
            s, co, _ = _db(h, Fn * SC, dict(base, M=M), hm, ref=(0.0, 0.0))
            C0, _ = static_stiffness(co)
            Cs = frozen_pressure_stiffness(co.db, co.ref, RHO)
            Ainf = co.instantaneous()["A"]
            NF = s.NF
            s.release()
            _free()
            Ct = Ch + (C0 + Cs)[1:, 1:]
            st = static_stability(Mm + Ainf[1:, 1:], Ct)
            old = float(np.linalg.eigvalsh(0.5 * ((Ch + C0[1:, 1:]) + (Ch + C0[1:, 1:]).T)).min())
            verdicts.append(st["stable"])
            notes.append(f"M {M} (NF {NF}): eig(M^-1 C_tot) {np.round(st['eig'].real, 5).tolist()}"
                         f"{'' if np.all(np.abs(st['eig'].imag) < 1e-12) else ' (complex ' + str(np.round(st['eig'].imag, 5).tolist()) + ')'}, "
                         f"det {st['det']:.3e}, C0+C_s diag {np.round(np.diag((C0 + Cs)[1:, 1:]), 4).tolist()}, C_s diag "
                         f"{np.round(np.diag(Cs[1:, 1:]), 4).tolist()}; B6 criterion (C0 only, sym part) {old:+.3e}")
        same = len(set(verdicts)) == 1
        all_ok &= same
        rows.append(_row("CDs", f"planing node Fn {Fn}: static verdict of C_hs + C0 + C_s on M {'/'.join(str(m) for m, _ in members)} "
                         f"({'stable' if verdicts[-1] else 'NOT stable'} on the finest)", float(not same), 0.0,
                         "grid series (12.1)", "same verdict on all grids", same, note="; ".join(notes)))
    return rows


def high_fn_image(args):
    """HFi (information; B6 review experiment 3', stage note 12.1): the total static base-flow stiffness of the
    double body without waves, K = -dF_db/dq (= C0 + C_s of that body problem), by re-panelled central differences
    (delta 1e-3) of the base-pressure force, with the rigid-lid image (Fn -> 0 limit of the steady free surface) and
    with the pressure-release image (phi = 0 on z = 0, z-odd; Fn -> infinity limit); flared prismatic body Fn 0.4-1.0
    (hulls 30 x 6 and 60 x 12; about (0, 0)) and Wigley I Fn 0.3. Does a diagonal sign flip? The split C0 + C_s with
    the production operators (edge-flux G, adjoint m-terms, frozen-field C_s) is printed for the rigid lid; for the
    odd image the base flow crosses z = 0, where the edge-flux operator (zero waterline flux) is not consistent, so
    only the finite-difference total is used."""
    from hydro.rankine.doublebody import base_pressure_stiffness, frozen_pressure_stiffness, DoubleBodyFlow
    rows = []
    cases = [("wigleyJ", _wj(), 0.3, (60, 12), (0.0, ZG))] + [("prismV", _prism("flared"), Fn, hm, (0.0, 0.0))
                                                            for hm in ((30, 6), (60, 12)) for Fn in (0.4, 0.6, 0.8, 1.0)]
    for name, h, Fn, hm, ref in cases:
        U = Fn * SC
        Mm, Ch = _heave_pitch_statics(h, ref)
        K = {}
        for zp in (1, -1):
            K[zp] = base_pressure_stiffness(h, U, 0.0, 0.0, hm, ref=ref, rho=RHO, delta=1e-3, zpar=zp,
                                            gop="fv" if zp == 1 else "lagr", mterms="sbp" if zp == 1 else "pointwise")
        db = DoubleBodyFlow(h.mesh(*hm), U, hm)
        Ksplit = (K[1]["C0_lid"] + frozen_pressure_stiffness(db, ref, RHO))[1:, 1:]
        del db
        _free()
        Kl, Ko = K[1]["K_db"][1:, 1:], K[-1]["K_db"][1:, 1:]
        flip = bool(np.any(np.sign(np.diag(Kl)) != np.sign(np.diag(Ko))))
        el, eo = np.sort(np.linalg.eigvals(Ch + Kl).real), np.sort(np.linalg.eigvals(Ch + Ko).real)
        rows.append(_row("HFi", f"{name} Fn {Fn} hull {hm}: base-flow stiffness -dF_db/dq without waves, rigid lid vs "
                         f"pressure-release image (information): a diagonal sign flips: {flip}", float(flip), 0.0,
                         "information", "info", True,
                         note=f"lid diag {np.round(np.diag(Kl), 5).tolist()} (C0_lid + C_s split {np.round(np.diag(Ksplit), 5).tolist()}), "
                              f"couplings {Kl[0, 1]:+.4f} {Kl[1, 0]:+.4f}, eig(C_hs + K) {np.round(el, 4).tolist()}; odd image diag "
                              f"{np.round(np.diag(Ko), 5).tolist()}, couplings {Ko[0, 1]:+.4f} {Ko[1, 0]:+.4f}, eig(C_hs + K) "
                              f"{np.round(eo, 4).tolist()}; ratio odd/lid (33, 55) {Ko[0, 0] / Kl[0, 0]:.3f} {Ko[1, 1] / Kl[1, 1]:.3f}; "
                              f"C_hs diag {np.round(np.diag(Ch), 4).tolist()}"))
    return rows


def kcn_check(args):
    """KCN (information; B6 review finding B2): effect of the inconsistency of the conservative kinematic operator on
    the coefficients: cons (production) vs nat (A - diag(Phi_zz), consistent but not adjoint), max and median relative
    difference of A33', B33', A55', B55' at omega' 1.659, 2.212, 3.318, 4.424 and of the C0 diagonal; Wigley I Fn 0.3
    (GRID_BASE M 40, hull 60 x 12) and flared prismatic Fn 0.4 (M 40, hull 40 x 8)."""
    from hydro.rankine.tables import static_stiffness
    rows = []
    for name, h, Fn, M, hm, ref in (("wigleyJ", _wj(), 0.3, 40, (60, 12), (0.0, ZG)),
                                    ("prismV", _prism("flared"), 0.4, 40, (40, 8), (0.0, 0.0))):
        V = _V(h)
        vals = {}
        for kin in ("cons", "nat"):
            s, co, _ = _db(h, Fn * SC, dict(GRID_BASE, M=M), hm, ref=ref, kinematic=kin)
            row = []
            for wp in (1.659, 2.212, 3.318, 4.424):
                A_, B_ = _nd(co.radiation(wp * SC)["T"], wp * SC, V)
                row += [A_[1, 1], B_[1, 1], A_[2, 2], B_[2, 2]]
            C0, _ = static_stiffness(co)
            vals[kin] = (np.array(row), np.diag(C0)[1:])
            s.release()
            _free()
        d = np.abs(vals["cons"][0] - vals["nat"][0]) / np.maximum(np.abs(vals["cons"][0]), 1e-12)
        dc = np.abs(vals["cons"][1] - vals["nat"][1]) / np.abs(vals["cons"][1])
        rows.append(_row("KCN", f"{name} Fn {Fn}: coefficients with the production (cons) vs the natural kinematic operator "
                         "(information: size of the effect of K's inconsistency)", float(d.max()), 0.0, "information", "info",
                         True, note=f"median {np.median(d):.4f}; C0 diag (33, 55) cons {np.round(vals['cons'][1], 5).tolist()} "
                                    f"nat {np.round(vals['nat'][1], 5).tolist()} ({np.round(dc, 3).tolist()})"))
    return rows


# =====================================================================
# B6 review: steady double-body problem vs the exact Kelvin solve (deep spheroid)
def sk_check(args):
    """SK (binding 3 %; B6 review finding B6, stage note 12.1): the double-body steady problem (omega = 0 response
    with f_d = (U^2 - |W|^2)/2, near-field force with the quadratic pressure, R_quad) on the deep prolate spheroid
    a/b 5 (a 0.5 L), centre depth 6 b, Fn 0.5, body 60 x 16, against the exact-Kelvin body-only BIE (kelvin.solve_body,
    full Bernoulli on the same panels; B3 check S2K) on the wavelength-scaled steady domain (steady_grid since the
    review). The two linearisations coincide as the body's disturbance at the free surface vanishes (here max|W +
    U x_hat|/U on z = 0 = 1 %), so the row tests the steady pipeline (domain, beaches, operators) against an exact
    solution. SKd (information): the B6 domain (STEADY_GRID unscaled) and the scaled domain x 1.5."""
    full = bool(args and args[0] == "full")
    from hydro.rankine import panels as P
    from hydro.rankine.panels import Panels
    from hydro.rankine import kelvin as KV
    from hydro.rankine.unsteady import build_grid, NoWaterline
    from hydro.rankine.doublebody import (DoubleBodyFlow, fs_operators, DBForwardSpeedLTI, surface_gradient2, STEADY_GRID,
                                          steady_grid)
    a_, b_, f, Fn, hm = 0.5, 0.1, 0.6, 0.5, (60, 16)
    U = Fn * np.sqrt(G0)
    sph = P.ellipsoid(a_, b_, b_, hm[0], hm[1], phi_range=(-np.pi / 2, np.pi / 2), spacing="cosine")
    raw = sph.raw.copy()
    raw[:, :, 2] -= f
    body = Panels(raw)
    phn = U * body.n[:, 0]
    phi = KV.solve_body(body, U, G0, nth=64)
    g = surface_gradient2(body, phi, phn, hm)
    Rk = 2 * np.sum(RHO * (U * g[:, 0] - 0.5 * np.einsum("ij,ij->i", g, g)) * body.area * body.n[:, 0])
    nw = NoWaterline(-a_, a_, 1.0)
    db = DoubleBodyFlow(body, U, hm, lid=True)

    def solve(grid):
        G = build_grid(nw, sgn=1.0, **grid)
        Wf = db.field_velocity(G["fs"].c)
        A, Kk, _ = fs_operators(G, Wf, 1.0, "cons")
        s = DBForwardSpeedLTI(body, G["fs"], A, Kk, G["nu"] * np.sqrt(G0), U, G0, RHO, hull_shape=hm)
        fd = 0.5 * (U * U - np.einsum("ij,ij->i", Wf, Wf))
        sol = s.solve_freq(0.0, np.zeros((s.NB, 1)), fd=fd[:, None])
        phB = np.real(sol["phi_B"][:, 0])
        gph = surface_gradient2(body, phB, np.zeros(len(body)), hm)
        p = -RHO * np.einsum("ij,ij->i", db.W, gph) - 0.5 * RHO * np.einsum("ij,ij->i", gph, gph)
        R = 2 * np.sum(p * body.area * body.n[:, 0])
        eps = float(np.abs(Wf + np.array([U, 0.0])).max() / U)
        NF = s.NF
        s.release()
        _free()
        return R, NF, eps
    kw = steady_grid(nw, U, G0)
    R, NF, eps = solve(kw)
    e = R / Rk - 1
    rows = [_row("SK", f"double-body steady R (near field, quadratic pressure) vs the exact-Kelvin body solve, deep spheroid "
                 f"a/b 5 depth 6b Fn {Fn}, body {hm}, scaled steady domain (M {kw['M']}, NF {NF})", float(abs(e)), 0.0,
                 "kelvin.solve_body (exact Green function, B3 S2K)", TOL_SK, abs(e) <= TOL_SK,
                 note=f"R/R_Kelvin {R / Rk:.4f}; base-flow disturbance on z = 0 {eps:.4f}; domain up {kw['up']:.3f} down "
                      f"{kw['down']:.3f} lat {kw['lat']:.3f}, beaches {kw['b_up']:.3f}/{kw['b_dn']:.3f}/{kw['b_lat']:.3f}")]
    if full:
        R0, NF0, _ = solve(dict(STEADY_GRID, M=kw["M"]))
        kw15 = dict(kw, **{k: 1.5 * kw[k] for k in ("up", "down", "lat", "b_up", "b_dn", "b_lat")})
        R15, NF15, _ = solve(kw15)
        rows.append(_row("SKd", "SK with the B6 (unscaled) steady domain and with the scaled domain x 1.5 (information)",
                         float(abs(R15 / R - 1)), 0.0, "information", "info", True,
                         note=f"B6 domain (NF {NF0}): R/R_Kelvin {R0 / Rk:.4f}; scaled x 1.5 (NF {NF15}): {R15 / Rk:.4f}"))
    del db
    _free()
    return rows


# =====================================================================
# B6 review: attitude grid (sliver panels at the pointed ends)
def attitude_check(args):
    """AT (binding; B6 review finding B5, stage note 12.1): rigid-lid double-body flow and base-pressure force on an
    attitude grid -- sink in +-{1e-4, 3e-4, 1e-3, 2e-3} L at zero trim and trim in +-{1e-4, 3e-4, 1e-3, 2e-3} rad at
    zero sink -- for the Wigley I (Fn 0.3, hull 60 x 12, about (0, z_G)) and the flared prismatic body (Fn 0.6, hulls 30
    x 6 and 60 x 12, about (0, 0)). AT1: no dry or inverted station in any mesh (min over stations of zeta_w + D >= 0;
    the B6 failure: inverted end stations). AT2: Wigley max |W|/U <= 1.5 at every attitude. AT3: prism area-weighted
    rms |W|/U <= 1.5 at every attitude (max reported: singular tip corners and the submerged chine near the tips,
    12.2 item 6). AT4/AT5: continuity and smoothness: max over the grid of |dF - (-K_lin dq)| / |K_lin dq| for the
    heave and pitch base-pressure forces, K_lin = C0_lid + C_s (frozen formula) at zero attitude, <= 0.10 (Wigley AT4,
    prism AT5 per mesh)."""
    from hydro.rankine.doublebody import base_pressure_force, base_pressure_stiffness, frozen_pressure_stiffness, \
        DoubleBodyFlow
    rows = []
    offs = (-2e-3, -1e-3, -3e-4, -1e-4, 1e-4, 3e-4, 1e-3, 2e-3)
    dry_total = 0
    for name, h, Fn, hm, ref in (("wigleyJ", _wj(), 0.3, (60, 12), (0.0, ZG)), ("prismV", _prism("flared"), 0.6, (30, 6), (0.0, 0.0)),
                                 ("prismV", _prism("flared"), 0.6, (60, 12), (0.0, 0.0))):
        U = Fn * SC
        bp = base_pressure_stiffness(h, U, 0.0, 0.0, hm, ref=ref, rho=RHO, delta=1e-3)
        db0 = DoubleBodyFlow(h.mesh(*hm), U, hm)
        Klin = (bp["C0_lid"] + frozen_pressure_stiffness(db0, ref, RHO))[1:, 1:]
        del db0
        F0 = bp["F0"][1:]
        wmax, wrms, dev, dry, notes = 0.0, 0.0, 0.0, 0, []
        for kind in ("sink", "trim"):
            for o in offs:
                sk, tr = (o, 0.0) if kind == "sink" else (0.0, o)
                xlo, xhi = h.wet_range(sk, tr)
                xb = xlo + (xhi - xlo) * 0.5 * (1 - np.cos(np.pi * np.linspace(0.0, 1.0, hm[0] + 1)))   # mesh stations
                g = h.zeta_w(xb, sk, tr) + h.D(xb)
                dry += int(np.sum(g < -1e-12))
                F, db = base_pressure_force(h, U, sk, tr, hm, ref, RHO)
                dry += int(np.sum(db.body.n[:, 1] < -1e-3))                       # inverted (twisted) panels
                Wn = np.linalg.norm(db.W, axis=1) / U
                a = db.body.area
                wmax = max(wmax, float(Wn.max()))
                wrms = max(wrms, float(np.sqrt(np.sum(a * Wn ** 2) / np.sum(a))))
                del db
                # q = (x3 heave up, x5 pitch bow down) of this attitude about ref
                dq = np.array([ref[1] * (np.cos(tr) - 1) + ref[0] * np.sin(tr) - sk, -tr])   # (x3 of ref, x5)
                pred = -Klin @ dq
                d_ = (F[1:] - F0) - pred
                rel = float(np.max(np.abs(d_)) / max(np.max(np.abs(pred)), 1e-300))
                dev = max(dev, rel)
                notes.append(f"{kind} {o:+.0e}: |W|max {Wn.max():.2f} rel {rel:.3f}")
        _free()
        dry_total += dry
        lab = f"{name} hull {hm} Fn {Fn}"
        if name == "wigleyJ":
            rows.append(_row("AT2", f"{lab}: max |W|/U over the attitude grid", wmax, 0.0, "bounded (12.1)", TOL_AT_W,
                             wmax <= TOL_AT_W))
            rows.append(_row("AT4", f"{lab}: continuity/smoothness, max |dF - (-K_lin dq)|/|K_lin dq| (heave, pitch)", dev,
                             0.0, "linear prediction C0_lid + C_s", TOL_AT_LIN, dev <= TOL_AT_LIN, note="; ".join(notes)))
        else:
            rows.append(_row("AT3", f"{lab}: max over the attitude grid of the area-weighted rms |W|/U", wrms, 0.0,
                             "bounded (12.1)", TOL_AT_W, wrms <= TOL_AT_W, note=f"max |W|/U {wmax:.2f} (singular tip corners)"))
            rows.append(_row("AT5", f"{lab}: continuity/smoothness, max |dF - (-K_lin dq)|/|K_lin dq| (heave, pitch)", dev,
                             0.0, "linear prediction C0_lid + C_s", TOL_AT_LIN, dev <= TOL_AT_LIN, note="; ".join(notes)))
    rows.insert(0, _row("AT1", "attitude grid (Wigley I, prism 30x6 and 60x12): mesh stations outside the wetted range "
                        "(dry) plus inverted panels (n_y < -1e-3)", float(dry_total), 0.0, "exact (wet_range)", 0.0,
                        dry_total == 0))
    return rows


# =====================================================================
def as_info(fn, tag_suffix="r"):
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


tn_wigley_info = as_info(tn_wigley)

TIERS = {
    "quick": [
        ("KKu", kk_unit, (160,), None),
        ("MT", mterm_checks, (), None),
        ("DXa", dxa_check, ((20, 40),), None),
        ("DXu", dxu_check, ((20, 40),), None),
        ("DXD", dxd_check, ((20, 40),), None),
        ("TN", tn_wigley, (0.3, (1.0,), (2.212,), dict(CHECK_GRID, M=30), (60, 12)), None),
        ("TNp", tn_planing, ("flared", "TNp", ((40, (40, 8), True),)), None),
        ("TNw", tn_planing, ("wall", "TNw", ((40, (40, 8), True),)), None),
        ("WE", we_sphere, ((((24, 12), True, False),),), None),
        ("KK", kk_check, (0.3, 160, SMALL_GRID, SMALL_HULL), None),
        ("SP", spectrum, ((("wigleyJ", 0.3, True), ("prismV", 0.4, True), ("wigleyJ", 0.0, False)), dict(SMALL_GRID, M=16),
                          (16, 4)), None),
        ("TD", td_checks, (0.3, SMALL_GRID, SMALL_HULL), None),
        ("FS", froude_similarity, (SMALL_GRID, SMALL_HULL), None),
        ("AT", attitude_check, (), None),
        ("SK", sk_check, ("quick",), None),
        ("SF", steady_wigley, ((0.316,), ()), None),
        ("V", validation, (COARSE,), None),
        ("PV", planing_table, (dict(GRID_BASE, M=20, u_dn=0.3, u_lat=0.2), (24, 6), "quick"), None),
    ],
    "full": [
        ("KKu", kk_unit, (400,), None),
        ("MT", mterm_checks, (), None),
        ("DXa", dxa_check, ((20, 40, 80),), None),
        ("DXu", dxu_check, ((20, 40, 80),), None),
        ("DXb", dxb_check, (), None),
        ("DXD", dxd_check, ((20, 40, 80),), None),
        ("TN", tn_wigley, (0.3, (1.0, 1.5), (2.212, 4.424), CHECK_GRID, (60, 12)), None),
        ("TNp", tn_planing, ("flared", "TNp", ((40, (40, 8), True),)), None),
        ("TNw", tn_planing, ("wall", "TNw", ((40, (40, 8), True),)), None),
        ("TNs", tn_submerged, (0.3, (2.212, 4.424), dict(CHECK_GRID, u_dn=0.0, u_lat=0.0)), None),
        ("WE", we_sphere, ((((24, 12), True, True), ((48, 24), True, False)),), None),
        ("KK", kk_check, (0.3, 400, SMALL_GRID, SMALL_HULL), None),
        ("KK2", kk_check, (0.2, 400, SMALL_GRID, SMALL_HULL), None),
        ("G0", g0_limit, (dict(GRID_BASE, M=40), (40, 8)), None),
        ("SP", spectrum, ((("wigleyJ", 0.2, True), ("wigleyJ", 0.3, True), ("wigleyJ", 1.0, True), ("prismV", 0.4, True),
                           ("prismVw", 0.4, True), ("wigleyJ", 0.0, False)), SMALL_GRID, SMALL_HULL), None),
        ("TD", td_checks, (0.3, SMALL_GRID, SMALL_HULL), None),
        ("FS", froude_similarity, (SMALL_GRID, SMALL_HULL), None),
        ("GS", grid_series, (0.2,), None),
        ("GSp", planing_series, (("flared",),), None),
        ("DOM", domain_sensitivity, (0.2,), None),
        ("U0", u0_checks, ((2.212, 3.318), dict(GRID_BASE, M=40)), None),
        ("CS", cs_c0b_check, (), None),
        ("AT", attitude_check, (), None),
        ("SK", sk_check, ("full",), None),
        ("SF", steady_wigley, (FN_FIX, FN_FREE, 0.289), None),
        ("V", validation, (PROD,), None),
        ("CDw", decay_wigley, (dict(GRID_BASE, M=30, u_dn=0.3, u_lat=0.2), (40, 8)), None),
        ("PV", planing_table, (dict(GRID_BASE, M=30, u_dn=0.3, u_lat=0.2), (30, 6), "full"), None),
        ("CDs", planing_stiffness_series, ((0.6, 0.8, 1.0), ((30, (30, 6)), (45, (45, 9)), (60, (60, 12)))), None),
        ("HFi", high_fn_image, (), None),
        ("KCN", kcn_check, (), None),
    ],
}


# ------------------------------------------------------------ registration (keys b6:*)
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
            out["b6:" + head.strip()] = dict(title=None, sha256=None)
            continue
        i = hits[0]
        j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ") or lines[k].startswith("### ")), len(lines))
        sec = "\n".join(lines[i:j]).strip() + "\n"
        out["b6:" + head.strip()] = dict(title=lines[i].strip(), sha256=hashlib.sha256(sec.encode("utf-8")).hexdigest())
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
    consts = (f"GRID_BASE={GRID_BASE!r} PROD={PROD!r} COARSE={COARSE!r} CHECK={CHECK_GRID!r} SMALL={SMALL_GRID!r} "
              f"TOLS={(TOL_ID, TOL_KK, TOL_U0, TOL_G0, TOL_INF, TOL_TD, TOL_EN, TOL_FS, TOL_GS, TOL_DOM, TOL_V_REL, V_FLOOR, TOL_SBP, TOL_FSR)} "
              f"FN_FIX={FN_FIX} FN_FREE={FN_FREE} TOL_C0B={TOL_C0B} TOL_AT_W={TOL_AT_W} TOL_AT_LIN={TOL_AT_LIN} "
              f"TOL_SK={TOL_SK}")
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
    cur["b6:code_thresholds"] = h_thr
    cur["b6:tier_jobs"] = h_jobs
    reg = {}
    for k, hh in cur.items():
        hit = [e["utc"] for e in log if hh is not None and e["hashes"].get(k) == hh]
        reg[k] = dict(sha256=hh, registered_utc=hit[-1] if hit else None)
    return reg


def register(note):
    secs = _note_sections()
    h_thr, h_jobs, items = _code_thresholds()
    try:
        log = json.load(open(REG_LOG, encoding="utf-8"))
    except (OSError, ValueError):
        log = []
    hashes = {k: v["sha256"] for k, v in secs.items()}
    hashes.update({"b6:code_thresholds": h_thr, "b6:tier_jobs": h_jobs})
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
    print(f"=== rankine_db_verify tier={tier} start {start_utc}; registration record:", flush=True)
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
    path = os.path.join(OUT, f"db_verify_{tier}{'_' + '_'.join(only) if only else ''}.json")
    info_rows = [r for r in rows if isinstance(r["tol"], str) and r["tol"].startswith("info")]
    impl_rows = [r for r in rows if isinstance(r["tol"], str) and r["tol"].startswith("impl")]
    bind_rows = [r for r in rows if not (isinstance(r["tol"], str) and (r["tol"].startswith("info")
                                                                       or r["tol"].startswith("impl")))]
    summary = dict(binding_pass=sum(r["ok"] for r in bind_rows), binding=len(bind_rows), info=len(info_rows),
                   impl_pass=sum(r["ok"] for r in impl_rows), impl=len(impl_rows),
                   failing=[f"{r['check']}: {r['quantity'][:110]}" for r in bind_rows if not r["ok"]],
                   impl_failing=[f"{r['check']}: {r['quantity'][:110]}" for r in impl_rows if not r["ok"]])
    vr = [r for r in bind_rows if r["check"] == "Vd"]
    summary["validation_DB"] = f"{sum(r['ok'] for r in vr)}/{len(vr)}"
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
    print(f"\n=== rankine_db_verify tier={tier}  wall {wall:.0f} s  -> {path}")
    for r in rows:
        v = r["value"]
        if isinstance(v, float):
            v = f"{v:.4g}"
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['check']:5s} {r['quantity']}\n"
              f"        value={v}  ref={r['ref']}  tol={r['tol']}  ({r['source']})"
              f"{'  ' + r['note'] if r['note'] else ''}  [{r.get('seconds', '')} s]")
    print(f"=== binding rows: {summary['binding_pass']}/{summary['binding']} passed; implementation (Green-identity) rows: "
          f"{summary['impl_pass']}/{summary['impl']} passed (not M3 passes, B6 review); information rows: {summary['info']}; "
          f"validation points passed: {summary['validation_DB']}")
    for fl in summary["failing"]:
        print(f"    FAIL {fl}")
    for fl in summary["impl_failing"]:
        print(f"    FAIL (implementation row) {fl}")
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
