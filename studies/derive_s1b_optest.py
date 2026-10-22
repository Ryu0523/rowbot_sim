#!/usr/bin/env python3
"""
S1 stage 2 (S1b) operator tests, CPU only (exact base flows, no panel representation):

  J. band/lattice junction (stage-1 item 3): the DXa test (exact half-spheroid double-body flow) and the DXu test
     (exact uniform stream, Wigley I and flared prism) on the CHECK band grids M 20/40/80 with the stage-1 band
     (band="linear": the last band row is up to half the first lattice row) and the graded band (band="geom": row
     height continuous across the junction). Regions: the band interior (lines 3..J-1), the junction rows J-3..J+2, the
     lattice. Operators: production "cons" and the edge-flux pair "flux-fv6t". Observed orders M 40 -> 80.
  S. stem/stern end columns (stage-1 item 1): the uniform-stream test on the real scarab195 5 kn grids (coarse / base
     / fine) with the grid variants of derive_s1b_common (stage-1 grid, end_width, end sub-columns + stern knuckle,
     + graded band): errors of A f, K f in the stem and stern regions and in the band; K 1 (exact by construction).
     Also the geometric measures per variant: normal distance between the band's wall chord and the true waterline,
     the Neumann-Kelvin Jacobian sign, the distance of the end-column Gauss points to the hull's top edges.

Output: studies/_cache/derive/s1b_optest.txt, s1b_optest.json
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
from hydro.rankine.unsteady import build_grid  # noqa: E402
from hydro.rankine.doublebody import fs_operators, fs_operators_flux, fs_nodes  # noqa: E402
from studies.derive_s1_optest import errors, order  # noqa: E402
from studies.derive_s1b_common import grid_opts  # noqa: E402
from studies.derive_s1b_diag import nk_jacobian  # noqa: E402

LINES = []


def log(msg=""):
    print(msg, flush=True)
    LINES.append(msg)


def ops(G, Wc, Wfun, sgn, kind):
    if kind == "cons":
        return fs_operators(G, Wc, sgn, "cons")
    return fs_operators_flux(G, Wfun, sgn, recon=kind.split("-")[1])


def regions_j(G, a_):
    nx, ny = G["shape"]
    X, Y = G["Xc"], G["Yc"]
    i_s, i_b, J = G["i_s"], G["i_b"], G["J"]
    jj = np.arange(ny)[None, :] * np.ones((nx, 1), int)
    hull = np.zeros((nx, ny), bool)
    hull[i_s:i_b, :] = True
    win = np.abs(X) <= 0.6 * a_
    winl = (np.abs(X) < 0.8) & (Y < 0.45)
    return {"wall (lines 0-2)": hull & win & (jj <= 2),
            "band (lines 3..J-1)": hull & win & (jj >= 3) & (jj < J),
            "junction (J-3..J+2)": hull & win & (jj >= J - 3) & (jj <= J + 2),
            "lattice (>= J+3)": (jj >= J + 3) & winl}


def regions_u(G):
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
    return {"junction rows": junc & win & ~wall & ~kink, "end kinks": kink & win & ~wall,
            "smooth": win & ~wall & ~kink & ~junc}


def main():
    t_all = time.time()
    res = {}
    from hydro.rankine.hull import half_spheroid, prismatic_v
    from hydro.rankine.analytic import spheroid_double_body
    kinds = ("cons", "flux-fv6t")
    Ms = (20, 40, 80)
    bands = ("linear", "geom")
    # ------------------------------------------------------------------ J1. DXa spheroid
    log("J1. DXa: exact half-spheroid double-body flow (a 0.5, b 0.1), CHECK grids M 20/40/80; band 'linear' (stage 1) vs 'geom'")
    a_, b_, U = 0.5, 0.1, 1.0
    hs = half_spheroid(a_, b_)
    dxa = {b: {k: {} for k in kinds} for b in bands}
    for band in bands:
        for M in Ms:
            G = build_grid(hs, 0.0, 0.0, 1.0, **dict(CHECK_GRID, M=M, band=band))
            nx, ny = G["shape"]
            X, Y = G["Xc"], G["Yc"]
            mat = lambda v: v.reshape(ny, nx).T  # noqa: E731
            Wa_, pz, _ = spheroid_double_body(a_, b_, U, G["fs"].c)
            Wfun = lambda P: spheroid_double_body(a_, b_, U, P)[0][:, :2]  # noqa: E731
            f, fx, fy, kk = _mms(X, Y)
            Wx, Wy, P2 = mat(Wa_[:, 0]), mat(Wa_[:, 1]), mat(pz)
            exA = Wx * fx + Wy * fy
            exK = exA - P2 * f
            sc = kk * np.abs(Wa_[:, :2]).max()
            sc1 = np.abs(P2[(np.abs(X) < 0.8) & (Y < 0.45)]).max()
            masks = regions_j(G, a_)
            # junction row-height ratio over the hull
            Xn, Yn = fs_nodes(G)
            J = G["J"]
            hb = Yn[G["i_s"]:G["i_b"] + 1, J] - Yn[G["i_s"]:G["i_b"] + 1, J - 1]
            hl = Yn[0, J + 1] - Yn[0, J]
            if M == 40:
                log(f"  band {band} M {M}: J {J}, last band row over the hull {hb.min():.4f}..{hb.max():.4f}, first lattice row {hl:.4f} "
                    f"(ratio up to {hl / hb.min():.2f}); first band row min {np.diff(Yn[G['i_s']:G['i_b'] + 1, :2], axis=1).min():.4f} (dy0 {G['dy0']:.4f})")
            for kind in kinds:
                A, K, info = ops(G, Wa_[:, :2], Wfun, 1.0, kind)
                dxa[band][kind][M] = errors(G, A, K, exA, exK, -P2, sc, sc1, masks, f)
    res["J1_dxa"] = {b: {k: {str(M): {"|".join(q): v for q, v in d.items()} for M, d in v.items()} for k, v in dd.items()} for b, dd in dxa.items()}
    log("  errors max (rms) M 20/40/80 and observed orders 40 -> 80 (max | rms):")
    for reg in regions_j(build_grid(hs, 0.0, 0.0, 1.0, **dict(CHECK_GRID, M=20)), a_):
        for q in ("A", "K", "K1"):
            parts = []
            for band in bands:
                for kind in kinds:
                    e = [dxa[band][kind][M][(q, reg, "max")] for M in Ms]
                    er = [dxa[band][kind][M][(q, reg, "rms")] for M in Ms]
                    parts.append(f"{band[:3]}/{kind}: " + " ".join(f"{v:.1e}" for v in e) + f" ({order(e[1], e[2], 40, 80):+.2f} | {order(er[1], er[2], 40, 80):+.2f})")
            log(f"  {q:2s} {reg:22s} " + " || ".join(parts))
    # ------------------------------------------------------------------ J2. DXu wigley / prism
    log("J2. DXu: exact uniform stream, CHECK grids of the Wigley I and the flared prism; band linear vs geom")
    dxu = {}
    for hname in ("wigleyJ", "prismV"):
        h = _wj() if hname == "wigleyJ" else prismatic_v()
        dxu[hname] = {b: {k: {} for k in kinds} for b in bands}
        for band in bands:
            for M in Ms:
                G = build_grid(h, 0.0, 0.0, 1.0, **dict(CHECK_GRID, M=M, band=band))
                nx, ny = G["shape"]
                X, Y = G["Xc"], G["Yc"]
                Wc = np.tile([-1.0, 0.0], (nx * ny, 1))
                Wfun = lambda P: np.tile([-1.0, 0.0], (len(P), 1))  # noqa: E731
                f, fx, fy, kk = _mms(X, Y)
                masks = regions_u(G)
                for kind in kinds:
                    A, K, info = ops(G, Wc, Wfun, 1.0, kind)
                    dxu[hname][band][kind][M] = errors(G, A, K, -fx, -fx, np.zeros_like(fx), kk, 1.0, masks, f)
        res[f"J2_dxu_{hname}"] = {b: {k: {str(M): {"|".join(q): v for q, v in d.items()} for M, d in v.items()} for k, v in dd.items()} for b, dd in dxu[hname].items()}
        log(f"  {hname}: errors max M 20/40/80, orders 40 -> 80 (max | rms):")
        for reg in regions_u(build_grid(h, 0.0, 0.0, 1.0, **dict(CHECK_GRID, M=20))):
            for q in ("A", "K"):
                parts = []
                for band in bands:
                    for kind in kinds:
                        e = [dxu[hname][band][kind][M][(q, reg, "max")] for M in Ms]
                        er = [dxu[hname][band][kind][M][(q, reg, "rms")] for M in Ms]
                        parts.append(f"{band[:3]}/{kind}: " + " ".join(f"{v:.1e}" for v in e) + f" ({order(e[1], e[2], 40, 80):+.2f} | {order(er[1], er[2], 40, 80):+.2f})")
                log(f"  {q:2s} {reg:14s} " + " || ".join(parts))
    # ------------------------------------------------------------------ S. scarab grids, variants
    log("S. uniform stream on the scarab195 5 kn grids (coarse / base / fine), grid variants; flux-fv6t (and cons on s1 for reference)")
    from studies.derive_v1b_build import plant_numbers, scarab_form, GRIDS
    p = plant_numbers(False)
    hull = scarab_form(p, 0.3)
    sink, trim = p["sink"], p["trim"]
    xs, xb = hull.waterline_ends(sink, trim)
    L = hull.L
    variants = ("s1", "ew", "es", "esg")
    sc_res = {v: {} for v in variants}
    geo = {v: {} for v in variants}
    gp = (0.5 * (1 - 1 / np.sqrt(3.0)), 0.5 * (1 + 1 / np.sqrt(3.0)))
    for gname in ("coarse", "base", "fine"):
        gk, hm = GRIDS[gname]
        body = hull.mesh(*hm, sink=sink, trim=trim)
        raw = body.raw.reshape(hm[0], hm[1], 4, 3)[:, -1]
        segs = []
        for ist in range(hm[0]):
            pts = raw[ist][np.abs(raw[ist][:, 2]) < 1e-9]
            if len(pts) >= 2:
                u = np.unique(np.round(pts[:, :2], 12), axis=0)
                if len(u) >= 2:
                    segs.append((u[0], u[-1]))
        segs = np.array(segs)
        Aseg, Bseg = segs[:, 0], segs[:, 1]
        Eseg = Bseg - Aseg
        EE = np.maximum(np.einsum("ij,ij->i", Eseg, Eseg), 1e-300)

        def dseg(P):
            t = np.clip(np.einsum("ij,ij->i", P[None, :] - Aseg, Eseg) / EE, 0, 1)
            Q = Aseg + t[:, None] * Eseg
            d = np.linalg.norm(P[None, :] - Q, axis=1)
            k = int(np.argmin(d))
            return d[k], np.sqrt(EE[k])
        for v in variants:
            kw = dict(gk, **grid_opts(v, hull, sink, trim))
            G = build_grid(hull, sink, trim, 1.0, **kw)
            nx, ny = G["shape"]
            X, Y = G["Xc"], G["Yc"]
            xe, ywl = G["xe"], G["ywl_e"]
            i_s, i_b, J = G["i_s"], G["i_b"], G["J"]
            # geometry
            sag = 0.0
            for i in range(i_s, i_b):
                t = np.linspace(0, 1, 11)[1:-1]
                xq = xe[i] + t * (xe[i + 1] - xe[i])
                chord = ywl[i] + t * (ywl[i + 1] - ywl[i])
                true = hull.waterline(xq, sink, trim)
                slope = abs(ywl[i + 1] - ywl[i]) / (xe[i + 1] - xe[i])
                sag = max(sag, (true - chord).max() / np.sqrt(1 + slope ** 2))
            Jac = nk_jacobian(G, hull, sink, trim)[0]
            Xn, Yn = fs_nodes(G)
            dmin = []
            for ic in (i_b, i_b - 1, i_s, i_s + 1):
                for j in (0, 1):
                    p0, p1 = np.array([Xn[ic, j], Yn[ic, j]]), np.array([Xn[ic, j + 1], Yn[ic, j + 1]])
                    for q in gp:
                        d, a = dseg(p0 + q * (p1 - p0))
                        dmin.append(d / a)
            geo[v][gname] = dict(NF=nx * ny, shape=list(G["shape"]), J=J, sag_normal=float(sag), nk_bad=int(np.sum(Jac <= 0)),
                                 gauss_min_d_over_a=float(min(dmin)), min_area=float(G["fs"].area.min()))
            Wc = np.tile([-1.0, 0.0], (nx * ny, 1))
            Wfun = lambda P: np.tile([-1.0, 0.0], (len(P), 1))  # noqa: E731
            xm = 0.5 * (xs + xb)
            f, fx, fy, kk = _mms((X - xm) / L, Y / L)
            fx, fy, kk = fx / L, fy / L, kk / L
            ii = np.arange(nx)[:, None] * np.ones((1, ny), int)
            jj = np.arange(ny)[None, :] * np.ones((nx, 1), int)
            hullc = (ii >= i_s) & (ii < i_b)
            win = (np.abs(X - xm) < 0.8 * L) & (Y < 0.45 * L)
            wall = hullc & (jj <= 2) & win
            stem = (np.abs(X - xb) <= 0.1 * L) & win & ~wall
            stern = (np.abs(X - xs) <= 0.1 * L) & win & ~wall
            band = hullc & (jj > 2) & (jj < J) & win & ~stem & ~stern
            junc = hullc & (np.abs(jj - (J - 0.5)) <= 3.5) & win & ~stem & ~stern
            rest = win & ~stem & ~stern & ~wall & ~band
            masks = {"stem": stem, "stern": stern, "band": band, "junction": junc, "rest": rest}
            kk_ = ("cons", "flux-fv6t") if v == "s1" else ("flux-fv6t",)
            for kind in kk_:
                A, K, info = ops(G, Wc, Wfun, 1.0, kind)
                sc_res[v].setdefault(kind, {})[gname] = errors(G, A, K, -fx, -fx, np.zeros_like(fx), kk, 1.0, masks, f)
            log(f"  {gname} {v:4s}: grid {G['shape']} NF {nx * ny} J {J}; chord/waterline normal gap {sag * 100:.2f} cm; NK Jac<=0 cells {geo[v][gname]['nk_bad']}; "
                f"end-column Gauss points: min distance/segment length {min(dmin):.3f}; min cell area {G['fs'].area.min():.2e} m2")
    res["S_geo"] = geo
    res["S_err"] = {v: {k: {g: {"|".join(q): val for q, val in d.items()} for g, d in dd.items()} for k, dd in vv.items()} for v, vv in sc_res.items()}
    log("  errors max coarse/base/fine and order base -> fine (max | rms), relative to k max|W|; A f = K f for the uniform stream (flux), K 1 exact:")
    for reg in ("stem", "stern", "band", "junction", "rest"):
        for q in ("A", "K"):
            parts = []
            for v in variants:
                for kind in sc_res[v]:
                    if q == "K" and kind != "cons" and v != "s1":
                        continue
                    e = [sc_res[v][kind][g][(q, reg, "max")] for g in ("coarse", "base", "fine")]
                    er = [sc_res[v][kind][g][(q, reg, "rms")] for g in ("coarse", "base", "fine")]
                    parts.append(f"{v}/{kind}: " + " ".join(f"{x:.1e}" for x in e) + f" ({order(e[1], e[2], 48, 64):+.2f} | {order(er[1], er[2], 48, 64):+.2f})")
            if parts:
                log(f"  {q:2s} {reg:9s} " + " || ".join(parts))
    res["t_total"] = time.time() - t_all
    log(f"total {res['t_total']:.0f} s")
    with open(os.path.join(OUT, "s1b_optest.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(os.path.join(OUT, "s1b_optest.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
