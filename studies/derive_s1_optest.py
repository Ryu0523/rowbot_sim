#!/usr/bin/env python3
"""
S1 step 2 (learn/meta/HULL_FORCE_AUDIT_REVIEW_2026-10-09.md, "下一步" item 1): unit tests of the opt-in edge-flux
free-surface operator pair (hydro/rankine/doublebody.fs_operators_flux, kinematic="flux") against the production
pair (kinematic="cons"). CPU only, no panel representation (exact base flows), so it runs beside the GPU jobs.

  A. nodes of fs_nodes reproduce the panel corners of build_grid (geometry check);
  B. flat uniform grid, uniform stream: the flux pair equals the production pair to round-off in the interior
     (sixth-order central + (1/60) delta6) -- the reduction claim of the derivation;
  C. exact discrete reciprocity Wa K_U = (Wa A_{-U})^T, skewness of S, PSD of Wa V, K 1 = div W (discrete Gauss);
  D. DXa (exact half-spheroid double-body flow) and DXu (exact uniform stream, Wigley I and flared prism) on the
     CHECK band grids M 20/40/80: errors and observed orders of A f, K f, K 1 in the regions of the B6 review rows,
     max norm and area-weighted rms, for cons / flux fv6 / flux lin;
  E. the same uniform-stream test on the real scarab195 5 kn grids (coarse / base / fine of derive_v1b_build).

Output: studies/_cache/derive/s1_optest.txt and s1_optest.json.
"""
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

from studies.rankine_unsteady_verify import CHECK_GRID, _wj  # noqa: E402
from studies.rankine_db_verify import _mms  # noqa: E402
from hydro.rankine.unsteady import build_grid, NoWaterline  # noqa: E402
from hydro.rankine.doublebody import fs_operators, fs_operators_flux, fs_nodes  # noqa: E402

LINES = []


def log(msg=""):
    print(msg, flush=True)
    LINES.append(msg)


def ops(G, Wc, Wfun, sgn, kind):
    if kind == "cons":
        A, K, info = fs_operators(G, Wc, sgn, "cons")
        return A, K, info
    recon = kind.split("-")[1]
    return fs_operators_flux(G, Wfun, sgn, recon=recon)


def regions_dxa(G, a_):
    nx, ny = G["shape"]
    X, Y = G["Xc"], G["Yc"]
    i_s, i_b, J = G["i_s"], G["i_b"], G["J"]
    jj = np.arange(ny)[None, :] * np.ones((nx, 1), int)
    hull = np.zeros((nx, ny), bool)
    hull[i_s:i_b, :] = True
    win = np.abs(X) <= 0.6 * a_
    return {"near wall (lines 0-2)": hull & win & (jj <= 2), "band (lines 3..J-1)": hull & win & (jj >= 3) & (jj < J),
            "lattice (lines >= J)": (jj >= J) & (np.abs(X) < 0.8) & (Y < 0.45)}


def regions_dxu(G):
    nx, ny = G["shape"]
    X, Y = G["Xc"], G["Yc"]
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
    jt = int(np.argmax(np.abs(dyl[1:] / dyl[:-1] - 1) > 1e-6) + 1) if len(dyl) > 2 else 0
    lat = np.abs(jj - (jt - 0.5)) <= 3.5
    nons = (kink | junc | lat) & win & ~wall
    smooth = win & ~wall & ~nons
    return {"non-smooth lines": nons, "smooth": smooth, "wall lines 0-2": wall & win}


def errors(G, A, K, exA, exK, exK1, sc, sc1, masks, f):
    nx, ny = G["shape"]
    mat = lambda v: v.reshape(ny, nx).T
    vec = lambda F: F.T.reshape(-1)
    a = mat(G["fs"].area)
    EA = np.abs(mat(A @ vec(f)) - exA) / sc
    EK = np.abs(mat(K @ vec(f)) - exK) / sc
    E1 = np.abs(mat(K @ np.ones(nx * ny)) - exK1) / sc1
    out = {}
    for r, m in masks.items():
        for q, E in (("A", EA), ("K", EK), ("K1", E1)):
            if not np.any(m):
                out[(q, r, "max")] = out[(q, r, "rms")] = float("nan")
                continue
            out[(q, r, "max")] = float(E[m].max())
            out[(q, r, "rms")] = float(np.sqrt(np.sum(a[m] * E[m] ** 2) / np.sum(a[m])))
    return out


def order(e1, e2, M1, M2):
    if e2 <= 1e-13:
        return float("inf")
    if e1 <= 1e-13:
        return float("-inf")
    return float(np.log(e1 / e2) / np.log(M2 / M1))


def structural_checks(G, kind, Wc, Wfun, sgn, label):
    """C: reciprocity, skewness, PSD, discrete Gauss."""
    import scipy.sparse as sp
    a = G["fs"].area
    Ap, Kp, ip = ops(G, Wc, Wfun, sgn, kind)
    Am, Km, im = ops(G, -Wc, (lambda P: -np.asarray(Wfun(P))), -sgn, kind)
    Wa = sp.diags(a)
    rec = abs((Wa @ Kp) - (Wa @ Am).T).max() / max(abs(Wa @ Kp).max(), 1e-300)
    rec2 = abs((Wa @ Ap) - (Wa @ Km).T).max() / max(abs(Wa @ Ap).max(), 1e-300)
    out = dict(reciprocity=float(rec), reciprocity_rev=float(rec2))
    if kind != "cons":
        S = ip["S"]
        out["skew"] = float(abs(S + S.T).max() / max(abs(S).max(), 1e-300))
        WaV = ip["WaV"]
        out["WaV_sym"] = float(abs(WaV - WaV.T).max() / max(abs(WaV).max(), 1e-300))
        # PSD: smallest eigenvalue of the (small) dense matrix, or a random-vector test for large ones
        n = WaV.shape[0]
        if n <= 2500:
            ev = np.linalg.eigvalsh(WaV.toarray())
            out["WaV_min_eig_rel"] = float(ev.min() / max(ev.max(), 1e-300))
        else:
            rng = np.random.default_rng(0)
            Z = rng.standard_normal((n, 64))
            q = np.einsum("ij,ij->j", Z, WaV @ Z) / np.einsum("ij,ij->j", Z, Z)
            out["WaV_rayleigh_min"] = float(q.min())
        # K 1 = div W (discrete Gauss) relative to max |div W|
        k1 = Kp @ np.ones(n)
        out["K1_minus_divW_rel"] = float(np.abs(k1 - ip["divW"]).max() / max(np.abs(ip["divW"]).max(), 1e-300))
        # odd/even in U
        out["F_odd_in_U"] = float(np.abs(ip["F"] + im["F"]).max() / max(np.abs(ip["F"]).max(), 1e-300))
    log(f"  [{label}] {kind}: " + ", ".join(f"{k} {v:.2e}" for k, v in out.items()))
    return out


def main():
    t_all = time.time()
    res = {}
    # ---------------------------------------------------------------- A. nodes
    log("A. node coordinates of fs_nodes vs the panel corners of build_grid")
    for h, kw in ((_wj(), dict(CHECK_GRID, M=20)), (NoWaterline(-0.5, 0.5, 1.0), dict(CHECK_GRID, M=20))):
        G = build_grid(h, 0.0, 0.0, 1.0, **kw)
        Xn, Yn = fs_nodes(G)
        nx, ny = G["shape"]
        raw = G["fs"].raw.reshape(ny, nx, 4, 3)
        corners = np.stack([Xn, Yn], -1)                                 # (nx+1, ny+1, 2)
        err = 0.0
        for j in range(ny):
            for i in range(nx):
                quad = raw[j, i][:, :2]
                want = np.array([corners[i, j], corners[i + 1, j], corners[i + 1, j + 1], corners[i, j + 1]])
                # the panel may be stored reversed; compare as sets
                d = min(np.abs(quad - want).max(), np.abs(quad[::-1] - want).max(),
                        np.abs(np.roll(quad[::-1], 1, 0) - want).max())
                err = max(err, d)
        log(f"  {h.name}: max corner mismatch {err:.2e}")
        res[f"A_nodes_{h.name}"] = float(err)
    # ---------------------------------------------------------------- B. uniform-grid reduction
    log("B. flat uniform grid (NoWaterline, no stretching inside the window), uniform stream W = (-1, 0): flux vs cons")
    hN = NoWaterline(-0.5, 0.5, 1.0)
    kwN = dict(M=40, up=1.0, down=1.0, lat=1.0, r_up=1.0, r_dn=1.0, r_lat=1.0, b_up=0.5, b_dn=0.5, b_lat=0.5, h_max=None,
               u_up=1.0, u_dn=1.0, u_lat=1.0, aspect=1.0, J_mult=2.0)
    G = build_grid(hN, 0.0, 0.0, 1.0, **kwN)
    nx, ny = G["shape"]
    Wc = np.tile([-1.0, 0.0], (nx * ny, 1))
    Wfun = lambda P: np.tile([-1.0, 0.0], (len(P), 1))
    Ac, Kc, _ = fs_operators(G, Wc, 1.0, "cons")
    for kind in ("flux-fv6t", "flux-fv6", "flux-lin"):
        Af, Kf, inf = fs_operators_flux(G, Wfun, 1.0, recon=kind.split("-")[1])
        X, Y = G["Xc"], G["Yc"]
        dx = G["dx"]
        # interior: at least 4 cells from every edge (stencil reach 3 + one)
        ii = np.arange(nx)[:, None] * np.ones((1, ny), int)
        jj = np.arange(ny)[None, :] * np.ones((nx, 1), int)
        interior = (ii >= 6) & (ii <= nx - 7) & (jj >= 6) & (jj <= ny - 7)   # the transposed one-sided end stencils of cons reach 5 cells
        ks = (jj * nx + ii)[interior]
        dA = abs(Af[ks] - Ac[ks]).max() / abs(Ac[ks]).max()
        dK = abs(Kf[ks] - Kc[ks]).max() / abs(Kc[ks]).max()
        # near the centre plane (mirror) and the inflow end (zero ghosts): rows differ by the end treatments only
        edge = ~interior
        kse = (jj * nx + ii)[edge]
        dAe = abs(Af[kse] - Ac[kse]).max() / abs(Ac[ks]).max()
        log(f"  {kind}: interior rows |A_flux - A_cons| / max|A| {dA:.2e}, |K_flux - K_cons| / max|K| {dK:.2e}; "
            f"non-interior rows (end treatments differ by design) {dAe:.2e}; dx {dx:.4f}, grid {G['shape']}")
        res[f"B_uniform_{kind}"] = dict(dA_interior=float(dA), dK_interior=float(dK), dA_edge=float(dAe))
    # ---------------------------------------------------------------- C/D. exact flows on CHECK grids
    from hydro.rankine.hull import half_spheroid, prismatic_v
    from hydro.rankine.analytic import spheroid_double_body
    kinds = ("cons", "flux-fv6t", "flux-fv6", "flux-lin")
    Ms = (20, 40, 80)
    log("C/D. DXa: exact half-spheroid double-body flow (a 0.5, b 0.1), CHECK grids")
    a_, b_, U = 0.5, 0.1, 1.0
    hs = half_spheroid(a_, b_)
    dxa = {k: {} for k in kinds}
    for M in Ms:
        G = build_grid(hs, 0.0, 0.0, 1.0, **dict(CHECK_GRID, M=M))
        nx, ny = G["shape"]
        X, Y = G["Xc"], G["Yc"]
        mat = lambda v: v.reshape(ny, nx).T
        Wa_, pz, _ = spheroid_double_body(a_, b_, U, G["fs"].c)
        Wfun = lambda P: spheroid_double_body(a_, b_, U, P)[0][:, :2]
        f, fx, fy, kk = _mms(X, Y)
        Wx, Wy, P2 = mat(Wa_[:, 0]), mat(Wa_[:, 1]), mat(pz)
        exA = Wx * fx + Wy * fy
        exK = exA - P2 * f
        sc = kk * np.abs(Wa_[:, :2]).max()
        sc1 = np.abs(P2[(np.abs(X) < 0.8) & (Y < 0.45)]).max()
        masks = regions_dxa(G, a_)
        for kind in kinds:
            t0 = time.time()
            A, K, info = ops(G, Wa_[:, :2], Wfun, 1.0, kind)
            dxa[kind][M] = errors(G, A, K, exA, exK, -P2, sc, sc1, masks, f)
            dxa[kind][M]["t_build"] = time.time() - t0
            if M == 40:
                res[f"C_struct_spheroid_M40_{kind}"] = structural_checks(G, kind, Wa_[:, :2], Wfun, 1.0, "spheroid M40")
    res["D_dxa"] = {k: {str(M): {"|".join(kk): v for kk, v in d.items() if kk != "t_build"} | {"t_build": d["t_build"]}
                        for M, d in v.items()} for k, v in dxa.items()}
    log("  errors (max | rms) and observed orders M 40 -> 80:")
    for reg in regions_dxa(build_grid(hs, 0.0, 0.0, 1.0, **dict(CHECK_GRID, M=20)), a_):
        for q in ("A", "K", "K1"):
            parts = []
            for kind in kinds:
                e = [dxa[kind][M][(q, reg, "max")] for M in Ms]
                er = [dxa[kind][M][(q, reg, "rms")] for M in Ms]
                parts.append(f"{kind}: " + " ".join(f"{v:.1e}" for v in e) + f" (ord {order(e[1], e[2], 40, 80):+.2f}) | rms ord "
                             f"{order(er[1], er[2], 40, 80):+.2f}")
            log(f"  {q:2s} {reg:24s} " + " || ".join(parts))
    log("D. DXu: exact uniform stream W = (-1, 0), CHECK grids of the Wigley I and the flared prism")
    dxu = {}
    for hname in ("wigleyJ", "prismV"):
        h = _wj() if hname == "wigleyJ" else prismatic_v()
        dxu[hname] = {k: {} for k in kinds}
        for M in Ms:
            G = build_grid(h, 0.0, 0.0, 1.0, **dict(CHECK_GRID, M=M))
            nx, ny = G["shape"]
            X, Y = G["Xc"], G["Yc"]
            Wc = np.tile([-1.0, 0.0], (nx * ny, 1))
            Wfun = lambda P: np.tile([-1.0, 0.0], (len(P), 1))
            f, fx, fy, kk = _mms(X, Y)
            masks = regions_dxu(G)
            for kind in kinds:
                A, K, info = ops(G, Wc, Wfun, 1.0, kind)
                dxu[hname][kind][M] = errors(G, A, K, -fx, -fx, np.zeros_like(fx), kk, 1.0, masks, f)
                if M == 40 and hname == "prismV":
                    res[f"C_struct_prismV_M40_{kind}"] = structural_checks(G, kind, Wc, Wfun, 1.0, "prismV M40 uniform")
        res[f"D_dxu_{hname}"] = {k: {str(M): {"|".join(kk): v for kk, v in d.items()} for M, d in v.items()} for k, v in dxu[hname].items()}
        log(f"  {hname}: errors (max) M 20/40/80 and observed orders 40 -> 80 (max | rms):")
        for reg in regions_dxu(build_grid(h, 0.0, 0.0, 1.0, **dict(CHECK_GRID, M=20))):
            for q in ("A", "K", "K1"):
                parts = []
                for kind in kinds:
                    e = [dxu[hname][kind][M][(q, reg, "max")] for M in Ms]
                    er = [dxu[hname][kind][M][(q, reg, "rms")] for M in Ms]
                    parts.append(f"{kind}: " + " ".join(f"{v:.1e}" for v in e) + f" (ord {order(e[1], e[2], 40, 80):+.2f} | "
                                 f"{order(er[1], er[2], 40, 80):+.2f})")
                log(f"  {q:2s} {reg:18s} " + " || ".join(parts))
    # ---------------------------------------------------------------- E. scarab grids, uniform stream
    log("E. uniform stream on the scarab195 5 kn grids of derive_v1b_build (coarse / base / fine, ramp 0.3, stem floor 2 mm)")
    from studies.derive_v1b_build import plant_numbers, scarab_form, GRIDS
    p = plant_numbers(False)
    hull = scarab_form(p, 0.3)
    sc_res = {k: {} for k in kinds}
    Mseq = []
    for gname in ("coarse", "base", "fine"):
        gk, hm = GRIDS[gname]
        G = build_grid(hull, p["sink"], p["trim"], 1.0, **gk)
        nx, ny = G["shape"]
        M = gk["M"]
        Mseq.append(M)
        X, Y = G["Xc"], G["Yc"]
        Wc = np.tile([-1.0, 0.0], (nx * ny, 1))
        Wfun = lambda P: np.tile([-1.0, 0.0], (len(P), 1))
        # manufactured field scaled to the hull length (the DXD field is for L = 1)
        L = hull.L
        xs, xb = hull.waterline_ends(p["sink"], p["trim"])
        xm = 0.5 * (xs + xb)
        f, fx, fy, kk = _mms((X - xm) / L, Y / L)
        fx, fy, kk = fx / L, fy / L, kk / L
        i_s, i_b, J = G["i_s"], G["i_b"], G["J"]
        ii = np.arange(nx)[:, None] * np.ones((1, ny), int)
        jj = np.arange(ny)[None, :] * np.ones((nx, 1), int)
        hullc = (ii >= i_s) & (ii < i_b)
        win = (np.abs(X - xm) < 0.8 * L) & (Y < 0.45 * L)
        # regions by PHYSICAL distance (a cell-count window would sample ever steeper band lines near the knife-edge
        # stem as the grid is refined); the wall lines 0-2 over the hull are excluded from the kink regions: the
        # uniform test stream crosses the hull wall there, so the imposed zero wall flux makes K 1 != 0 for every
        # operator (the DXu row of the ledger excludes them for the same reason) -- reported separately, information
        wall = hullc & (jj <= 2) & win
        stem = (np.abs(X - xb) <= 0.1 * L) & win & ~wall
        stern = (np.abs(X - xs) <= 0.1 * L) & win & ~wall
        band = hullc & (jj > 2) & (jj < J) & win & ~stem & ~stern
        rest = win & ~stem & ~stern & ~wall & ~band
        masks = {"stem kink": stem, "stern kink": stern, "wall lines 0-2": wall, "band": band, "rest of window": rest}
        for kind in kinds:
            t0 = time.time()
            A, K, info = ops(G, Wc, Wfun, 1.0, kind)
            e = errors(G, A, K, -fx, -fx, np.zeros_like(fx), kk, 1.0, masks, f)
            e["t_build"] = time.time() - t0
            sc_res[kind][M] = e
        log(f"  {gname} M {M}: grid {G['shape']}, NF {nx * ny}, J {J}; build times " +
            ", ".join(f"{k} {sc_res[k][M]['t_build']:.1f} s" for k in kinds))
    res["E_scarab"] = {k: {str(M): {"|".join(kk): v for kk, v in d.items() if kk != "t_build"} | {"t_build": d["t_build"]}
                           for M, d in v.items()} for k, v in sc_res.items()}
    log("  errors (max) coarse/base/fine and observed order base -> fine (max | rms); relative to kk max|W|; stem/stern = within 0.1 L of the waterline end, wall lines excluded (information: the uniform stream crosses the wall there):")
    for reg in ("stem kink", "stern kink", "wall lines 0-2", "band", "rest of window"):
        for q in ("A", "K", "K1"):
            parts = []
            for kind in kinds:
                e = [sc_res[kind][M][(q, reg, "max")] for M in Mseq]
                er = [sc_res[kind][M][(q, reg, "rms")] for M in Mseq]
                parts.append(f"{kind}: " + " ".join(f"{v:.1e}" for v in e) + f" (ord {order(e[1], e[2], Mseq[1], Mseq[2]):+.2f} | "
                             f"{order(er[1], er[2], Mseq[1], Mseq[2]):+.2f})")
            log(f"  {q:2s} {reg:16s} " + " || ".join(parts))
    res["t_total"] = time.time() - t_all
    log(f"total {res['t_total']:.0f} s")
    with open(os.path.join(OUT, "s1_optest.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(os.path.join(OUT, "s1_optest.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
