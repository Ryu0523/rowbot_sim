#!/usr/bin/env python3
"""
S1b step 0 (diagnostics, CPU only): where exactly the free-surface band grid of unsteady.build_grid is inconsistent with
the scarab195 hull at 5 kn, on the V1b mesh sequence (coarse / base / fine).

  1. Chord penetration: the band's j = 0 edge is a straight chord between the waterline nodes of neighbouring
     columns; where the waterline is convex (the whole forebody taper, and the blunt elliptic nose above all) the chord
     lies INSIDE the hull. Reports the maximum sag (true half-breadth minus chord) per column near both ends.
  2. Neumann-Kelvin d/dx metric (unsteady.dx_operator, wall="trace"): the sign of the Jacobian Xi Yj - Xj Yi per
     cell, i.e. which columns trigger "grid not right-handed", with the wall node of the trace closure
     (x = column centre, y = the TRUE waterline there) versus the cell's own corners.
  3. Positions of the two Gauss points of every i-face of the stem end column, their distance to the hull's top
     edges (the top row of the hull panels, on z = 0), and the hull panel size there.
  4. Band row heights at the band/lattice junction (last band row vs first lattice row) over the hull.

Output: studies/_cache/derive/s1b_diag.txt
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "_cache", "derive")
os.makedirs(OUT, exist_ok=True)

from hydro.rankine.unsteady import build_grid, _geom_derivative, _index_derivative, _jstencil_trace, _odd_mirror_derivative  # noqa: E402
from studies.derive_v1b_build import plant_numbers, scarab_form, GRIDS  # noqa: E402

LINES = []


def log(s=""):
    print(s, flush=True)
    LINES.append(s)


def nk_jacobian(G, hull, sink, trim, sgn=1.0):
    """The metric part of unsteady.dx_operator(wall='trace') without the assertion; returns Jac (nx, ny), J0."""
    nx, ny = G["shape"]
    X, Y = G["Xc"], G["Yc"]
    hull_j0 = np.zeros(nx, bool)
    if G["J"] > 0:
        hull_j0[G["i_s"]:G["i_b"]] = True
    s = 1.0 if sgn >= 0 else -1.0
    Xi0, Yi0 = _geom_derivative(X, 0), _geom_derivative(Y, 0)
    Xj0, Yj0 = np.empty_like(X), np.empty_like(Y)
    for i in range(nx):
        mir = not hull_j0[i]
        Xj0[i] = _geom_derivative(X[i], 0, mirror_lo=mir, odd=False)
        Yj0[i] = _geom_derivative(Y[i], 0, mirror_lo=mir, odd=True)
    J0 = Xi0 * Yj0 - Xj0 * Yi0
    side_i = np.sign(s * Yj0 / J0)
    side_j = np.sign(-s * Yi0 / J0)
    Xi, Yi = np.empty_like(X), np.empty_like(Y)
    Xj, Yj = np.zeros_like(X), np.zeros_like(Y)
    for j in range(ny):
        Dg = _index_derivative(nx, side_i[:, j], "linear", "linear")
        Xi[:, j], Yi[:, j] = Dg @ X[:, j], Dg @ Y[:, j]
    wall_cols = [i for i in range(nx) if hull_j0[i] and side_j[i, 0] < 0]
    xw = X[wall_cols, 0].copy() if wall_cols else np.zeros(0)
    yw = hull.waterline(xw, sink, trim) if wall_cols else np.zeros(0)
    for i in range(nx):
        if i in wall_cols:
            Dm, cw = _jstencil_trace(ny, side_j[i])
            k = wall_cols.index(i)
            Dg = _index_derivative(ny, side_j[i], "linear", "linear")
            Xj[i], Yj[i] = Dg @ X[i], Dg @ Y[i]
            rw = cw != 0
            Xj[i, rw] = (Dm @ X[i] + cw * xw[k])[rw]
            Yj[i, rw] = (Dm @ Y[i] + cw * yw[k])[rw]
            continue
        lo = "linear" if hull_j0[i] else "mirror"
        Dg = _index_derivative(ny, side_j[i], lo, "linear")
        if lo == "mirror":
            Xj[i] = Dg @ X[i]
            Yj[i] = _odd_mirror_derivative(Y[i], side_j[i], ny)
        else:
            Xj[i], Yj[i] = Dg @ X[i], Dg @ Y[i]
    Jac = Xi * Yj - Xj * Yi
    return Jac, J0, wall_cols, xw, yw, side_j


def main():
    p = plant_numbers(False)
    hull = scarab_form(p, 0.3)
    sink, trim = p["sink"], p["trim"]
    xs, xb = hull.waterline_ends(sink, trim)
    log(f"scarab195 at 5 kn: waterline x from {xs:+.4f} to {xb:+.4f} m (length {xb - xs:.4f}); hull x_aft {hull.x_aft:+.4f} "
        f"(stern end of the waterline is on the ramp, {xs - hull.x_aft / np.cos(trim) + sink * np.tan(trim):+.4f} m ahead of the ramp end)")
    # true waterline near both ends
    for lab, x0, sgn_ in (("stem", xb, -1.0), ("stern", xs, +1.0)):
        ds = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
        yy = hull.waterline(x0 + sgn_ * ds, sink, trim)
        log(f"  waterline half-breadth {lab}: at the end {hull.waterline(np.array([x0]), sink, trim)[0] * 1e3:.2f} mm; "
            + ", ".join(f"{d * 100:g} cm: {y * 100:.1f} cm" for d, y in zip(ds, yy)))
        c = yy[2] / np.sqrt(0.01)
        log(f"    fit y = c sqrt(d): c = {c:.3f} m^0.5 from d = 1 cm; predicted at 10 cm {c * np.sqrt(0.1) * 100:.1f} cm (true {yy[5] * 100:.1f})")
    for gname in ("coarse", "base", "fine"):
        gk, hm = GRIDS[gname]
        G = build_grid(hull, sink, trim, 1.0, **gk)
        nx, ny = G["shape"]
        xe, yw_e, J, dy0 = G["xe"], G["ywl_e"], G["J"], G["dy0"]
        i_s, i_b = G["i_s"], G["i_b"]
        log(f"\n[{gname}] M {gk['M']}, dx {G['dx']:.4f} m, dy0 {dy0:.4f} m, J {J}, grid {G['shape']}, NF {nx * ny}, hull mesh {hm}")
        # 1. chord penetration per column (sag at 9 interior points)
        sag = np.zeros(nx)
        where = np.zeros(nx)
        for i in range(i_s, i_b):
            t = np.linspace(0, 1, 11)[1:-1]
            xq = xe[i] + t * (xe[i + 1] - xe[i])
            chord = yw_e[i] + t * (yw_e[i + 1] - yw_e[i])
            true = hull.waterline(xq, sink, trim)
            k = int(np.argmax(true - chord))
            sag[i], where[i] = (true - chord)[k], xq[k]
        for i in list(range(i_s, i_s + 3)) + list(range(i_b - 3, i_b)):
            log(f"  column {i:3d} x [{xe[i]:+.3f}, {xe[i + 1]:+.3f}]: nodes y {yw_e[i] * 100:6.2f} / {yw_e[i + 1] * 100:6.2f} cm, "
                f"max sag (true waterline outside the chord) {sag[i] * 100:6.2f} cm at x {where[i]:+.3f}; chord slope {abs(yw_e[i + 1] - yw_e[i]) / (xe[i + 1] - xe[i]):.2f}")
        log(f"  max sag over all hull columns {sag.max() * 100:.2f} cm (column {int(np.argmax(sag))}); mid-hull columns max "
            f"{sag[i_s + 3:i_b - 3].max() * 100:.2f} cm; first band row height dy0 - y_wl/J: min {np.min(dy0 - yw_e / J) * 100:.2f} cm")
        # 4. junction
        band_last = (yw_e * (1 - (J - 1) / J) + (J - 1) * dy0)
        h_last = J * dy0 - band_last                  # height of the last band row at each node column
        yl = G["ylines"]
        h_lat = yl[J + 1] - yl[J]
        log(f"  junction: last band row height over the hull min / max {h_last[i_s:i_b + 1].min() * 100:.2f} / {h_last[i_s:i_b + 1].max() * 100:.2f} cm, "
            f"first lattice row {h_lat * 100:.2f} cm -> ratio up to {h_lat / h_last[i_s:i_b + 1].min():.2f}")
        # 2. NK Jacobian
        Jac, J0, wall_cols, xw, yw_w, side_j = nk_jacobian(G, hull, sink, trim)
        bad = np.argwhere(Jac <= 0)
        log(f"  NK dx_operator metric: wall-node columns (j-upwind side = wall) {wall_cols}; cells with Jac <= 0: {len(bad)}"
            + (f"; columns {sorted(set(bad[:, 0].tolist()))}, rows {sorted(set(bad[:, 1].tolist()))}" if len(bad) else ""))
        for i in sorted(set(bad[:, 0].tolist())):
            k = wall_cols.index(i) if i in wall_cols else None
            cy = G["Yc"][i, 0]
            log(f"    column {i} (x_c {G['Xc'][i, 0]:+.3f}): wall node y {yw_w[k] * 100 if k is not None else float('nan'):.2f} cm vs cell-0 centroid y "
                f"{cy * 100:.2f} cm, chord nodes {yw_e[i] * 100:.2f} / {yw_e[i + 1] * 100:.2f} cm; Jac row 0 {Jac[i, 0]:+.3e}, row 1 {Jac[i, 1]:+.3e}, J0 row 0 {J0[i, 0]:+.3e}")
        # 3. Gauss points of the stem end column i-face (node column i_b) and of the previous one
        body = hull.mesh(*hm, sink=sink, trim=trim)
        raw = body.raw.reshape(hm[0], hm[1], 4, 3)
        top = raw[:, -1]                                   # top-row panels
        segs = []
        for ist in range(hm[0]):
            pts = top[ist][np.abs(top[ist][:, 2]) < 1e-9]
            if len(pts) >= 2:
                u = np.unique(np.round(pts[:, :2], 12), axis=0)
                if len(u) >= 2:
                    segs.append((u[0], u[-1]))
        segs = np.array(segs)                              # (n, 2, 2)

        def dist_to_segs(P):
            A, B = segs[:, 0], segs[:, 1]
            E = B - A
            EE = np.maximum(np.einsum("ij,ij->i", E, E), 1e-300)
            t = np.clip(np.einsum("ij,ij->i", P[None, :] - A, E) / EE, 0, 1)
            Q = A + t[:, None] * E
            d = np.linalg.norm(P[None, :] - Q, axis=1)
            k = int(np.argmin(d))
            return d[k], k, np.sqrt(EE[k])
        from hydro.rankine.doublebody import fs_nodes
        Xn, Yn = fs_nodes(G)
        gp = (0.5 * (1 - 1 / np.sqrt(3.0)), 0.5 * (1 + 1 / np.sqrt(3.0)))
        for ic in (i_b, i_b - 1, i_s, i_s + 1):
            for j in (0, 1):
                p0 = np.array([Xn[ic, j], Yn[ic, j]])
                p1 = np.array([Xn[ic, j + 1], Yn[ic, j + 1]])
                for q in gp:
                    P = p0 + q * (p1 - p0)
                    d, k, a = dist_to_segs(P)
                    log(f"    i-face node column {ic} row {j}, Gauss point x {P[0]:+.4f} y {P[1] * 100:6.2f} cm: distance to the nearest hull top edge "
                        f"{d * 1e3:7.2f} mm (segment {k}, length {a * 100:.2f} cm, d/a {d / a:.2f})")
        # hull station spacing at the ends and mid-hull
        L_seg = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
        log(f"  hull top-edge segment lengths: stern end {L_seg[0] * 100:.2f}, {L_seg[1] * 100:.2f} cm; mid {np.median(L_seg) * 100:.2f} cm; "
            f"stem end {L_seg[-2] * 100:.2f}, {L_seg[-1] * 100:.2f} cm; stem segment direction (dx, dy) {(segs[-1, 1] - segs[-1, 0]) * 100} cm")
    with open(os.path.join(OUT, "s1b_diag.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
