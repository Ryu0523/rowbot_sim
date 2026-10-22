#!/usr/bin/env python3
"""
Forward-speed Wigley seakeeping data (Journee 1992, TU Delft report 909,
'Wigley I') for Track B step B5 (validation, acceptance V1-V3), transcribed
from the VECTOR figures of two open-access papers that plot Journee's
experimental values as markers. The report itself is not retrievable here
(repository.tudelft.nl answers HTTP 403 to scripted access and lists the file
under an indefinite embargo), so these are SECONDARY transcriptions: the marker
positions are read from the PDF drawing commands (precision 0.01 pt), but how
accurately the papers' authors placed Journee's values is not known.

Sources (registered in data/external/SOURCES.md, section wigley_journee/):
  * J. Mar. Sci. Eng. 2021, 9, 87 ("Hydrodynamic analysis and motions of ship
    with forward speed via a three-dimensional time-domain panel method",
    MDPI, CC BY 4.0), pages 17-19 (PDF pages 16-18): Fig. 12 (A33', B33'),
    Fig. 13 (A55', B55'), Fig. 14 (heave and pitch wave exciting force
    amplitudes), Fig. 16 (heave and pitch RAOs), all at Fn = 0.2, head seas;
    markers "Experiment" (blue down-triangles) = Journee (1992), their ref. 29.
    Definitions (their section 4.2): A33' = A33/(rho V), A55' = A55/(rho V L^2),
    B33' = B33/(rho V) sqrt(L/g), B55' = B55/(rho V L^2) sqrt(L/g),
    omega' = omega sqrt(L/g); F3' = F3 L/(rho g eta0 V), F5' = F5/(rho g eta0 V)
    (checked: F3' -> A_w L/V = 19.8 as k -> 0, the hydrostatic limit, as plotted),
    abscissa nu' = k L = omega_0^2 L/g (wave number); RAOs zeta3/eta0 and
    zeta5 L/(2 pi eta0) vs lambda/L. Hull: Table 2, L 3.0 m, B 0.3 m, draft
    0.1875 m, k_yy 0.25 L, V 0.0946 m^3; formula their eq. (41).
  * Kara F. (2022), "Time domain potential and source methods and their
    application to twin-hull high-speed crafts", Ships and Offshore Structures
    (open access, Sheffield Hallam repository SHURA 29602, VoR), page 8 (PDF page
    9), Fig. 9 (a) A33/(rho V), (b) B33/(rho V sqrt(g/L)) vs omega sqrt(L/g) at
    Fn = 0.3; markers "Experiment - Journee (1992)" (orange filled circles).
    Hull formula as their eq. (10) (modified Wigley, L/B = 10, L/T = 16).

Calibration: per panel an affine map PDF -> data from the axis tick marks
(MDPI figures, MATLAB export: tick marks are short grey rectangles; the tick
values are read from the rendered figure and written below) or from the
tick-label glyph centres and the axis line (Kara, Excel export); the residual of
the map at all ticks is reported. Marker position = centre of the marker's
bounding box; reading uncertainty = 1/6 of the marker height (bounding-box
centre vs centroid of a triangle) plus the calibration residual.

Run: python studies/rankine_journee_data.py  (writes the CSVs and prints a summary).
"""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "external", "wigley_journee")
PDF_JMSE = os.path.join(DATA, "jmse-09-00087.pdf")
PDF_KARA = os.path.join(DATA, "kara2022_shura29602.pdf")

# JMSE 9:87 panels: page, figure frame (white rect origin), tick values (x, y) in the order of increasing
# PDF coordinate (x left->right, y top->bottom), quantity names
JMSE_PANELS = [
    dict(name="A33", page=16, origin=(63.1, 91.3), xt=[0, 2, 4, 6, 8], yt=[2.0, 1.5, 1.0, 0.5], xq="omega'"),
    dict(name="B33", page=16, origin=(324.7, 91.3), xt=[0, 2, 4, 6, 8], yt=[1.5, 1.0, 0.5, 0.0], xq="omega'"),
    dict(name="A55", page=16, origin=(63.1, 292.7), xt=[0, 2, 4, 6, 8], yt=[0.25, 0.2, 0.15, 0.1, 0.05, 0.0],
         xq="omega'"),
    dict(name="B55", page=16, origin=(324.7, 292.7), xt=[0, 2, 4, 6, 8], yt=[0.1, 0.08, 0.06, 0.04, 0.02, 0.0],
         xq="omega'"),
    dict(name="F3", page=17, origin=(63.1, 538.1), xt=[0, 5, 10, 15], yt=[20, 15, 10, 5, 0], xq="nu'=kL"),
    dict(name="F5", page=17, origin=(324.7, 538.1), xt=[0, 5, 10, 15], yt=[2.5, 2.0, 1.5, 1.0, 0.5, 0.0],
         xq="nu'=kL"),
    dict(name="RAO3", page=18, origin=(63.1, 571.6), xt=[1.2, 1.7, 2.2], yt=[1.2, 1.0, 0.8, 0.6], xq="lambda/L"),
    dict(name="RAO5", page=18, origin=(324.7, 571.6), xt=[1.2, 1.7, 2.2], yt=[2.0, 1.6, 1.2, 0.8, 0.4],
         xq="lambda/L"),
]
GREY = 0.149


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _fig_rect(page, origin):
    import pymupdf
    for g in page.get_drawings():
        r = g["rect"]
        if abs(r.x0 - origin[0]) < 1 and abs(r.y0 - origin[1]) < 1 and abs(r.width - 207.6) < 1 \
                and g.get("fill") == (1.0, 1.0, 1.0):
            return pymupdf.Rect(r)
    raise RuntimeError(f"figure frame at {origin} not found")


def _ticks(page, box):
    vx, hy = set(), set()
    for g in page.get_drawings():
        r = g["rect"]
        if not (box.x0 <= r.x0 <= box.x1 and box.y0 <= r.y0 <= box.y1):
            continue
        c = g.get("fill") or g.get("color")
        if c is None or abs(c[0] - GREY) > 0.01:
            continue
        if r.width < 0.3 and 1.0 < r.height < 2.5:
            vx.add(round(r.x0 + r.width / 2, 3))
        elif r.height < 0.3 and 1.0 < r.width < 2.5:
            hy.add(round(r.y0 + r.height / 2, 3))
    return np.array(sorted(vx)), np.array(sorted(hy))


def _fit(pos, val):
    a, b = np.polyfit(pos, val, 1)
    res = np.abs(a * pos + b - np.asarray(val)).max()
    return a, b, res


def jmse_panel(doc, spec):
    p = doc[spec["page"]]
    box = _fig_rect(p, spec["origin"])
    vx, hy = _ticks(p, box)
    # ticks may be drawn on two opposite axes (boxed plots): keep unique positions
    if len(vx) != len(spec["xt"]) or len(hy) != len(spec["yt"]):
        raise RuntimeError(f"{spec['name']}: ticks {len(vx)}/{len(hy)} vs values {len(spec['xt'])}/{len(spec['yt'])}")
    ax, bx, rx = _fit(vx, spec["xt"])
    ay, by, ry = _fit(hy, spec["yt"])
    legend = None
    for g in p.get_drawings():
        r = g["rect"]
        # legend box: a black-stroked rectangle inside the frame, much smaller than the axes
        if box.contains(r) and g.get("color") == (0.0, 0.0, 0.0) and g["type"] == "s" and 30 < r.width < 120                 and 15 < r.height < 60 and len(g["items"]) <= 4:
            legend = r
    pts = []
    for g in p.get_drawings():
        r = g["rect"]
        if g.get("fill") != (0.0, 0.0, 1.0) or len(g["items"]) != 3 or not box.contains(r):
            continue
        if legend is not None and legend.contains(r):
            continue
        cx, cy = r.x0 + r.width / 2, r.y0 + r.height / 2
        pts.append((ax * cx + bx, ay * cy + by, abs(ay) * r.height / 6 + ry, abs(ax) * r.width / 6 + rx))
    pts.sort()
    return np.array(pts), dict(cal_res_x=rx, cal_res_y=ry, legend=legend is not None)


def kara_panels(doc):
    """Fig. 9 of Kara (2022): A33 and B33 at Fn 0.3. Calibration from the
    tick-label glyph centres (grey) along each axis."""
    p = doc[9]
    ORANGE = (1.0, 0.494, 0.133)
    GLYPH = (0.349, 0.349, 0.349)
    dr = p.get_drawings()

    def col(c):
        return None if c is None else tuple(round(x, 3) for x in c)
    out = {}
    for name, xlim, yt, xt in (("A33", (60.0, 310.0), [2.5, 2.0, 1.5, 1.0, 0.5, 0.0], [0, 1, 2, 3, 4, 5, 6]),
                               ("B33", (315.0, 560.0), [1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
                                [0, 1, 2, 3, 4, 5, 6])):
        gall = [g["rect"] for g in dr if col(g.get("fill")) == GLYPH and xlim[0] <= g["rect"].x0 <= xlim[1]
                and g["rect"].y0 > 580 and g["rect"].height < 8]
        legend = [r for r in gall if r.width > 20]
        gl = [r for r in gall if r.width <= 20 and r.height < 5]
        # y labels: left column (smallest x), x labels: bottom row (largest y)
        xmin = min(r.x0 for r in gl)
        ylab = sorted([r for r in gl if abs(r.x0 - xmin) < 1.5], key=lambda r: r.y0)
        ymax = max(r.y0 for r in gl)
        xlab = sorted([r for r in gl if abs(r.y0 - ymax) < 1.0], key=lambda r: r.x0)

        if len(ylab) != len(yt) or len(xlab) != len(xt):
            raise RuntimeError(f"Kara {name}: {len(ylab)} y labels, {len(xlab)} x labels")
        ay, by, ry = _fit(np.array([(r.y0 + r.y1) / 2 for r in ylab]), yt)
        ax, bx, rx = _fit(np.array([(r.x0 + r.x1) / 2 for r in xlab]), xt)
        pts = []
        for g in dr:
            r = g["rect"]
            if col(g.get("fill")) != ORANGE or g["items"][0][0] != "c" or not (xlim[0] <= r.x0 <= xlim[1]):
                continue
            cy = (r.y0 + r.y1) / 2
            if any(lg.y0 - 3 <= cy <= lg.y1 + 3 and r.x1 < lg.x0 + 2 for lg in legend):
                continue
            cx = (r.x0 + r.x1) / 2
            pts.append((ax * cx + bx, ay * cy + by, abs(ay) * r.height / 6 + ry, abs(ax) * r.width / 6 + rx))
        pts.sort()
        out[name] = (np.array(pts), dict(cal_res_x=rx, cal_res_y=ry))
    return out


def main():
    import pymupdf
    os.makedirs(DATA, exist_ok=True)
    rows = {}
    d = pymupdf.open(PDF_JMSE)
    for spec in JMSE_PANELS:
        pts, info = jmse_panel(d, spec)
        rows[spec["name"]] = (pts, info, spec["xq"])
    k = kara_panels(pymupdf.open(PDF_KARA))
    hdr = ("# Journee (1992) Wigley I experiments, secondary transcription from vector figure markers "
           "(studies/rankine_journee_data.py; sources in data/external/SOURCES.md, wigley_journee/)\n")

    def write(fname, cols, data, note):
        path = os.path.join(DATA, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(hdr)
            f.write("# " + note + "\n")
            f.write(",".join(cols) + "\n")
            for r in data:
                f.write(",".join(f"{v:.5f}" for v in r) + "\n")
        return path
    out = {}
    # Fn 0.2 coefficients: merge A33, B33, A55, B55 on omega'
    for nm in ("A33", "B33", "A55", "B55"):
        pts = rows[nm][0]
        out[nm] = write(f"journee1992_wigley1_fn020_{nm}.csv", ["omega_prime", nm + "_prime", "u_value", "u_omega"],
                        pts, f"Fn 0.2, J. Mar. Sci. Eng. 9:87 (2021) Fig. 12/13, markers 'Experiment'; calibration "
                        f"residual x {rows[nm][1]['cal_res_x']:.2e}, y {rows[nm][1]['cal_res_y']:.2e}")
    for nm, qn in (("F3", "F3_prime"), ("F5", "F5_prime")):
        out[nm] = write(f"journee1992_wigley1_fn020_{nm}.csv", ["nu_prime_kL", qn, "u_value", "u_nu"], rows[nm][0],
                        f"Fn 0.2 head seas, J. Mar. Sci. Eng. 9:87 Fig. 14; F3' = F3 L/(rho g eta0 V), "
                        f"F5' = F5/(rho g eta0 V), nu' = k L")
    for nm in ("RAO3", "RAO5"):
        out[nm] = write(f"journee1992_wigley1_fn020_{nm}.csv", ["lambda_over_L", nm, "u_value", "u_lambda"],
                        rows[nm][0], "Fn 0.2 head seas, J. Mar. Sci. Eng. 9:87 Fig. 16; RAO3 = zeta3/eta0, "
                        "RAO5 = zeta5 L/(2 pi eta0)")
    for nm in ("A33", "B33"):
        out["fn03_" + nm] = write(f"journee1992_wigley1_fn030_{nm}.csv",
                                  ["omega_prime", nm + "_prime", "u_value", "u_omega"], k[nm][0],
                                  f"Fn 0.3, Kara (2022) Fig. 9, markers 'Experiment - Journee (1992)'; calibration "
                                  f"residual x {k[nm][1]['cal_res_x']:.2e}, y {k[nm][1]['cal_res_y']:.2e}")
    summary = {os.path.basename(v): _sha(v) for v in out.values()}
    summary.update({os.path.basename(PDF_JMSE): _sha(PDF_JMSE), os.path.basename(PDF_KARA): _sha(PDF_KARA)})
    for nm, (pts, info, xq) in rows.items():
        print(f"JMSE {nm} ({xq}): " + "; ".join(f"({p[0]:.3f}, {p[1]:.4f})" for p in pts)
              + f"  [cal res {info['cal_res_x']:.1e}/{info['cal_res_y']:.1e}]")
    for nm in ("A33", "B33"):
        print(f"Kara Fn 0.3 {nm}: " + "; ".join(f"({p[0]:.3f}, {p[1]:.4f})" for p in k[nm][0]))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
