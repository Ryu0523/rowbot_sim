"""
Experimental Wigley-hull data for Track B step B4 (validation, acceptance V1-V3):
extraction of the vector curves of Yang, Lohner, Noblesse & Huang (2000) and the
WaveBEM-shipped digitisations, written to data/external/wigley/*.csv.

Sources (registered in data/external/SOURCES.md, section wigley/):
  * Yang C., Lohner R., Noblesse F., Huang T.T. (2000). Calculation of ship sinkage
    and trim using unstructured grids. ECCOMAS 2000, Barcelona, paper 367
    (congress2.cimne.com/eccomas/proceedings/eccomas2000/pdf/367.pdf).
    Figure 4a (sinkage s and trim t of the Wigley model free to sink and trim) and
    Figure 5a (wave drag coefficient, model fixed / free): the curve labelled
    "Experiment Results" is the University of Tokyo measurement of the cooperative
    Wigley experiment (Kajitani et al. 1983; the paper's text, section 5). The
    figures are vector graphics (gnuplot): the polylines are read from the PDF
    drawing commands, so the transcription error is the PDF coordinate precision
    (0.01 pt, about 1e-6 of the axis range), not a digitisation of a picture. The
    curve vertices are the values the authors gave to gnuplot (a faired
    experimental curve, about one vertex per 0.007 to 0.015 in Fn).
    Definitions (paper, section 5): s = (dF' + dA')/Lpp (dF', dA' the draught
    CHANGES at the forward and aft perpendiculars, positive = deeper; s > 0
    downward), t = (dA - dF)/Lpp (> 0 bow up). NOTE: s as printed is TWICE the
    mean sinkage over Lpp; whether the plotted experimental curve follows this
    definition or Kajitani's original one could not be confirmed from the original
    report (DTIC ADP003037 not retrievable here); see the stage note, B4 section,
    for how the ambiguity is handled (registered before any comparison).
  * WaveBEM experimental-results/wigley (Mola, Heltai & DeSimone; LGPL-2.1
    repository, commit b44b0fb, 2017-03-07): tokyoFr*.csv = wave profiles
    2 g zeta/U^2 along the hull (x/L from -0.5 at the stern... see note below),
    digitised from Hendrix & Noblesse's figure (copy shipped as tokyoWigley.pdf,
    page 62: "U. of Tokyo (*) and S.R.I. (x) experimental data for the Wigley hull
    in fixed position"); wigleyDragExp1-4.csv = C_W of the FIXED model from the
    same document's page 61, labelled SRI, UT1, UT2, UT3 (Y.N.U. is not
    digitised), digitised by the WaveBEM authors from a scanned figure.
    B4 review (stage note 9.7.1): at Fn < 0.3 these four files follow the
    slender-ship theory lines of that figure, not its measured markers -- an
    invalid C_W reference there; the markers were re-read by
    studies/rankine_wigley_scan.py.

Run: python studies/rankine_wigley_data.py  (writes the CSVs and prints a summary).
"""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "external", "wigley")
PDF = os.path.join(DATA, "yang_lohner_noblesse_huang_2000_eccomas367.pdf")


def _num(s):
    try:
        return float(s.strip())
    except ValueError:
        return None


def extract_axes_curves(page, box):
    """Polylines (>= 8 segments) inside `box` (x0, y0, x1, y1 in PDF points),
    calibrated by the numeric tick labels left of and below the box snapped to
    the tick lines. Returns (curves in data units, squares (markers) in data
    units, calibration dict)."""
    import pymupdf  # noqa: F401
    x0, y0, x1, y1 = box
    dr = page.get_drawings()
    # tick lines: horizontal (y ticks) and vertical (x ticks) 2-item lines at the box edge
    hticks, vticks = [], []
    for g in dr:
        r = g["rect"]
        if len(g["items"]) == 2 and abs(r.y1 - r.y0) < 1e-6 and abs(r.x0 - x0) < 1.0:
            hticks.append(r.y0)
        if len(g["items"]) == 2 and abs(r.x1 - r.x0) < 1e-6 and abs(r.y1 - y1) < 1.0:
            vticks.append(r.x0)
    hticks, vticks = np.array(sorted(set(hticks))), np.array(sorted(set(vticks)))
    ylab, xlab = [], []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for sp in ln["spans"]:
                v = _num(sp["text"])
                if v is None:
                    continue
                bx = sp["bbox"]
                cy, cx = 0.5 * (bx[1] + bx[3]), 0.5 * (bx[0] + bx[2])
                if bx[2] <= x0 + 0.5 and y0 - 5 <= cy <= y1 + 5 and bx[0] > x0 - 40:
                    ylab.append((hticks[np.argmin(abs(hticks - cy))], v))
                elif bx[1] >= y1 - 0.5 and bx[1] <= y1 + 8 and x0 - 8 <= cx <= x1 + 8:
                    xlab.append((vticks[np.argmin(abs(vticks - cx))], v))
    ylab, xlab = np.array(ylab), np.array(xlab)
    by = np.polyfit(ylab[:, 0], ylab[:, 1], 1)
    bxp = np.polyfit(xlab[:, 0], xlab[:, 1], 1)
    resy = float(np.max(abs(np.polyval(by, ylab[:, 0]) - ylab[:, 1])))
    resx = float(np.max(abs(np.polyval(bxp, xlab[:, 0]) - xlab[:, 1])))
    curves, squares = [], []
    for g in dr:
        r = g["rect"]
        if r.x0 < x0 - 1 or r.x1 > x1 + 1 or r.y0 < y0 - 1 or r.y1 > y1 + 1:
            continue
        its = g["items"]
        if len(its) == 1 and its[0][0] == "re":
            q = its[0][1]
            squares.append((np.polyval(bxp, 0.5 * (q.x0 + q.x1)), np.polyval(by, 0.5 * (q.y0 + q.y1))))
            continue
        segs = [it for it in its if it[0] == "l"]
        if len(segs) < 8:
            continue
        # the first segment is the legend sample (horizontal, not connected)
        pts = []
        for k, it in enumerate(segs):
            a, b = it[1], it[2]
            if k == 0 and abs(a.y - b.y) < 1e-6 and (len(segs) < 2 or (abs(segs[1][1].x - b.x) > 1e-3)):
                continue
            if not pts:
                pts.append((a.x, a.y))
            pts.append((b.x, b.y))
        P = np.array(pts)
        curves.append(np.stack([np.polyval(bxp, P[:, 0]), np.polyval(by, P[:, 1])], 1))
    cal = dict(y_fit=by.tolist(), x_fit=bxp.tolist(), y_label_resid=resy, x_label_resid=resx,
               n_ylab=len(ylab), n_xlab=len(xlab))
    return curves, squares, cal


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    import pymupdf
    doc = pymupdf.open(PDF)
    out = {}
    # Figure 4a (page index 10): sinkage (left), trim (right)
    p = doc[10]
    specs = [("yang2000_fig4a_sinkage_s", p, (117.16, 555.41, 285.61, 625.63)),
             ("yang2000_fig4a_trim_t", p, (339.57, 555.41, 504.89, 625.63))]
    # Figure 5a (page index 11): Cw fixed (left), free (right); boxes from the tick lines
    p2 = doc[11]
    specs += [("yang2000_fig5a_cw_fixed", p2, None), ("yang2000_fig5a_cw_free", p2, None)]
    # find the fig 5a boxes: frames ('qu' items) whose rect spans the Cw axes (y ~ 250-322)
    frames = []
    for g in p2.get_drawings():
        r = g["rect"]
        if len(g["items"]) == 1 and g["items"][0][0] == "qu" and 240 < r.y0 < 260 and r.width > 100:
            frames.append((r.x0, r.y0, r.x1, r.y1))
    frames = sorted(set(tuple(round(v, 2) for v in f) for f in frames))
    assert len(frames) == 2, frames
    specs[2] = (specs[2][0], p2, frames[0])
    specs[3] = (specs[3][0], p2, frames[1])
    for name, page, box in specs:
        curves, squares, cal = extract_axes_curves(page, box)
        assert len(curves) == 1, (name, len(curves))
        C = curves[0]
        fn = os.path.join(DATA, name + ".csv")
        hdr = {"yang2000_fig4a_sinkage_s": "Fn,s  (s = (dF'+dA')/Lpp as printed in the paper; UT experiment curve)",
               "yang2000_fig4a_trim_t": "Fn,t  (t = (dA-dF)/Lpp, >0 bow up; UT experiment curve)",
               "yang2000_fig5a_cw_fixed": "Fn,Cw  (wave drag coefficient, model fixed; UT experiment curve)",
               "yang2000_fig5a_cw_free": "Fn,Cw  (wave drag coefficient, model free to sink and trim; UT experiment curve)"}[name]
        np.savetxt(fn, C, delimiter=",", header=hdr, fmt="%.6g")
        out[name] = dict(n=len(C), Fn_range=[float(C[:, 0].min()), float(C[:, 0].max())],
                         markers_present_results=[[round(a, 4), round(b, 6)] for a, b in squares], cal=cal,
                         sha256=sha256(fn))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
