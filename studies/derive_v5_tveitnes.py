"""
Check V5 (learn/meta/HULL_FORCE_AUDIT_REVIEW_2026-10-09.md, sections 2.1 and 4): the post-chine-wetting
dynamic force coefficient of a wedge section at cross-plane Froude numbers 0.14-0.69, from the constant-
velocity entry tests of Tveitnes, Fairlie-Clarke & Varyani (2008, Ocean Eng. 35:1463-1478), against
  * 0                 (the "frozen beam" rule of the plant),
  * Wen et al. 2022   (gravity-free decay from the chine-wetting force to the steady free-streamline value
                       Cs_inf; hydro/rankine/planing_nl.py, anchored on the MLM force at chine wetting),
  * the measurements.

Part A digitises the vector charts of the PDF directly from the drawing paths (no pixel reading):
  Fig. 4   total fluid force (model weight removed, hydrostatic force INCLUDED) vs z/d for 12 (deadrise,
           velocity) runs: experiment markers (x) and the authors' "Gravity force" curve (their hydrostatic
           plus fitted residual-gravity force);
  Fig. 13  the authors' fitted flow-momentum drag coefficient Cfm per deadrise (experiment and CFD);
  Fig. 16  the authors' de-noised hydrodynamic force coefficient Cfb*tan(beta) vs z/d at 0.94/1.19 m/s.
Part B converts to C(h*) = (F - rho g L A_sub(z)) / (rho L c V^2), bins in z/d, and compares.

Definitions used throughout (Tveitnes' symbols): z = depth of the keel below the undisturbed surface,
d = c tan(beta) = chine height above the keel, c = B/2 = 0.3 m half-beam, L = 0.3 m section length,
V = entry velocity. Our coefficient C = f/(rho c V^2) with f the dynamic force per unit length equals
Tveitnes' Cfm (F_fm = Cfm (rho/2) L (2 y_w) v^2 with y_w = B/2 after chine wetting) and equals his Cfb
(= Force/((rho/2) L B v^2)); Wen's Cs (F = rho V^2 l Cs, l = half-beam) is the same normalisation.
Wen's h* = ((h - h_cw)/h_cw) cot(beta)^1.3075 with h_cw = (2/pi) c tan(beta), i.e. z/d = 2/pi at Wagner's
chine wetting, so h/h_cw = (pi/2) z/d.

Run from the repo root:  .venv/Scripts/python.exe -I studies/derive_v5_tveitnes.py
Outputs: data/external/tveitnes_2008/digitised_*.csv (+ README.md), studies/_cache/derive/v5_*.{txt,json,csv,png}
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pymupdf  # noqa: E402

from hydro.rankine import planing_nl as pnl  # noqa: E402
from hydro.rankine import planing_wedge2d as w2  # noqa: E402

PDF = os.path.join(ROOT, "data", "external", "tveitnes_2008", "tveitnes_2008_constant_velocity_entry.pdf")
DATA_DIR = os.path.join(ROOT, "data", "external", "tveitnes_2008")
OUT_DIR = os.path.join(ROOT, "studies", "_cache", "derive")

RHO, G, L, C_HALF = 1000.0, 9.81, 0.3, 0.3        # fresh-water tank, section length, half-beam (paper sec. 2.2.1)
PANELS = {  # Fig. 4 caption: (panel letter) -> (deadrise deg, velocity m/s); page and layout order a..f, g..l
    "a": (5, 0.94), "b": (5, 0.48), "c": (5, 0.24), "d": (10, 0.94), "e": (10, 0.72), "f": (10, 0.48),
    "g": (15, 0.94), "h": (15, 0.72), "i": (15, 0.48), "j": (30, 0.94), "k": (30, 0.72), "l": (45, 1.19),
}
PANEL_PAGES = {5: "abcdef", 6: "ghijkl"}
FIG16_SERIES = [(5, 0.94), (10, 0.94), (15, 0.94), (30, 0.94), (45, 1.19)]   # legend order = drawing order


# ----------------------------------------------------------------------------------------------------- Part A
def _numeric_labels(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for ln in b["lines"]:
            for s in ln["spans"]:
                t = s["text"].strip()
                if t and t.replace("-", "").replace(".", "").isdigit() and s["size"] > 7.5:
                    try:
                        v = float(t)
                    except ValueError:
                        continue
                    x0, y0, x1, y1 = s["bbox"]
                    out.append(dict(v=v, cx=(x0 + x1) / 2, cy=(y0 + y1) / 2, x1=x1, y0=y0, y1=y1, t=t))
    return out


def _text_boxes(page, words):
    boxes = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for ln in b["lines"]:
            for s in ln["spans"]:
                if s["text"].strip() in words:
                    boxes.append(pymupdf.Rect(s["bbox"]))
    return boxes


def _find_panels(page):
    """Panels of Fig. 4: x-label rows (decimal labels, same baseline, >= 4 members) and the y-label column
    right-aligned just left of the row's first label. Returns a list ordered by (row y, x)."""
    labs = _numeric_labels(page)
    rows = defaultdict(list)
    for lb in labs:
        if "." in lb["t"]:
            rows[round(lb["cy"])].append(lb)
    panels = []
    for cy, members in rows.items():
        members.sort(key=lambda d: d["cx"])
        # split a row where the label value resets (two panels side by side: ... 8.0 | 0.0 ...)
        groups, cur = [], [members[0]]
        for m in members[1:]:
            if m["v"] <= cur[-1]["v"]:
                groups.append(cur)
                cur = [m]
            else:
                cur.append(m)
        groups.append(cur)
        for grp in groups:
            if len(grp) < 4:
                continue
            xs = [m["cx"] for m in grp]
            x_first = xs[0]
            ycol = [lb for lb in labs if "." not in lb["t"] and x_first - 16 < lb["x1"] < x_first + 3
                    and cy - 200 < lb["cy"] < cy - 5]
            if len(ycol) < 4:
                continue
            ycol.sort(key=lambda d: d["cy"])
            panels.append(dict(row_y=cy, xlabels=grp, ylabels=ycol,
                               box=pymupdf.Rect(x_first - 1, ycol[0]["cy"] - 3, xs[-1] + 1, cy - 8)))
    panels.sort(key=lambda p: (round(p["row_y"] / 50), p["box"].x0))
    return panels


def _axis_maps(page, panel, drawings):
    """Linear maps pdf -> data. y from horizontal gridlines matched to the y labels (fallback label
    centres), x from axis tick marks matched to the x labels (fallback label centres)."""
    box = panel["box"]
    grid_y = []
    for d in drawings:
        if d.get("fill") or not box.intersects(d["rect"]):
            continue
        for it in d["items"]:
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.y - p2.y) < 0.05 and abs(p1.x - p2.x) > 0.8 * box.width:
                    grid_y.append(p1.y)
    yv, yp = [], []
    for lb in panel["ylabels"]:
        near = [g for g in grid_y if abs(g - lb["cy"]) < 3.0]
        yv.append(lb["v"])
        yp.append(float(np.mean(near)) if near else lb["cy"])
    tick_x = []
    for d in drawings:
        if d.get("fill"):
            continue
        for it in d["items"]:
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.x - p2.x) < 0.05 and 0.3 < abs(p1.y - p2.y) < 1.5 and box.x0 - 2 < p1.x < box.x1 + 2 \
                        and box.y1 - 2 < max(p1.y, p2.y) < box.y1 + 3:
                    tick_x.append(p1.x)
    xv, xp = [], []
    for lb in panel["xlabels"]:
        near = [t for t in tick_x if abs(t - lb["cx"]) < 3.0]
        xv.append(lb["v"])
        xp.append(float(np.mean(near)) if near else lb["cx"])
    ay, by = np.polyfit(yp, yv, 1)
    ax, bx = np.polyfit(xp, xv, 1)
    resid_y = np.max(np.abs(np.polyval([ay, by], yp) - np.array(yv)))
    resid_x = np.max(np.abs(np.polyval([ax, bx], xp) - np.array(xv)))
    return (lambda y: ay * y + by), (lambda x: ax * x + bx), dict(
        y_per_pt=abs(ay), x_per_pt=abs(ax), y_fit_resid=float(resid_y), x_fit_resid=float(resid_x),
        n_grid=len(set(np.round(grid_y, 1))), n_tick=len(tick_x))


ARM = 1.74  # pt, bbox side of one arm of an 'x' marker; the centre sits 0.29 pt (half stroke) inside a corner


def _cross_centres(pieces):
    """pieces: list of (idx, rect). Each 'x' marker is 4 filled arm capsules of ~1.74 x 1.69 pt whose
    bboxes have the marker centre 0.29 pt inside one corner. Vote over the 4 corner hypotheses."""
    votes = defaultdict(set)
    for idx, r in pieces:
        # two marker sizes in Fig. 4: arms 1.74 x 1.69 pt (panels i-l) and 2.25 x 2.35 pt (panels a-h)
        if not (1.5 <= r.width <= 2.6 and 1.5 <= r.height <= 2.6 and abs(r.width - r.height) < 0.25):
            continue
        for cx, cy, arm in ((r.x1 - 0.29, r.y1 - 0.29, "NW"), (r.x0 + 0.29, r.y0 + 0.29, "SE"),
                            (r.x0 + 0.29, r.y1 - 0.29, "NE"), (r.x1 - 0.29, r.y0 + 0.29, "SW")):
            votes[(round(cx * 4) / 4, round(cy * 4) / 4)].add((arm, idx))
    # merge neighbouring keys (rounding splits), keep centres with >= 3 distinct arms
    keys = sorted(votes)
    used, centres, members = set(), [], []
    for k in keys:
        if k in used:
            continue
        cluster = {k}
        for k2 in keys:
            if k2 not in used and abs(k2[0] - k[0]) <= 0.3 and abs(k2[1] - k[1]) <= 0.3:
                cluster.add(k2)
        arms = {}
        for kk in cluster:
            for arm, idx in votes[kk]:
                arms.setdefault(arm, idx)
        if len(arms) >= 3:
            used |= cluster
            xs = [kk[0] for kk in cluster]
            ys = [kk[1] for kk in cluster]
            centres.append((float(np.mean(xs)), float(np.mean(ys))))
            members.append(set(arms.values()))
    return centres, members


def _chains(pieces, max_gap=3.0, min_len=15):
    """Consecutive-index runs of pieces whose centres move < max_gap pt: the continuous curves."""
    runs, cur = [], []
    for idx, r in pieces:
        c = ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
        if cur and idx == cur[-1][0] + 1 and abs(c[0] - cur[-1][1][0]) < max_gap and abs(c[1] - cur[-1][1][1]) < max_gap:
            cur.append((idx, c))
        else:
            if len(cur) >= min_len:
                runs.append(cur)
            cur = [(idx, c)]
    if len(cur) >= min_len:
        runs.append(cur)
    return runs


def extract_fig4(doc):
    rows, meta = [], {}
    for pno, letters in PANEL_PAGES.items():
        page = doc[pno - 1]
        drawings = page.get_drawings()
        panels = _find_panels(page)
        assert len(panels) == 6, (pno, len(panels))
        legend_words = _text_boxes(page, {"Experiment", "Simulation", "Gravity force"})
        for letter, panel in zip(letters, panels):
            beta, V = PANELS[letter]
            box = panel["box"]
            fy, fx, info = _axis_maps(page, panel, drawings)
            leg = [b for b in legend_words if box.contains(b.tl) or box.contains(b.br)]
            legend_box = None
            if leg:
                legend_box = pymupdf.Rect(min(b.x0 for b in leg) - 40, min(b.y0 for b in leg) - 2,
                                          max(b.x1 for b in leg) + 2, max(b.y1 for b in leg) + 2)
            pieces = []
            for i, d in enumerate(drawings):
                if not d.get("fill"):
                    continue
                r = d["rect"]
                if not (box.x0 - 2 <= (r.x0 + r.x1) / 2 <= box.x1 + 2 and box.y0 - 2 <= (r.y0 + r.y1) / 2 <= box.y1 + 2):
                    continue
                if legend_box is not None and legend_box.contains(pymupdf.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)):
                    continue
                n_l = sum(1 for it in d["items"] if it[0] == "l")
                if n_l >= 8:
                    pieces.append((i, r))
            centres, members = _cross_centres(pieces)
            cross_idx = set().union(*members) if members else set()
            rest = [(i, r) for i, r in pieces if i not in cross_idx]
            curves = _chains(rest)
            for cx, cy in centres:
                rows.append(dict(panel=letter, beta_deg=beta, V_mps=V, series="experiment", z_over_d=fx(cx), force_N=fy(cy)))
            n_curve = 0
            for ch in curves:
                for _, (cx, cy) in ch:
                    rows.append(dict(panel=letter, beta_deg=beta, V_mps=V, series="gravity_force", z_over_d=fx(cx), force_N=fy(cy)))
                    n_curve += 1
            meta[letter] = dict(beta_deg=beta, V_mps=V, page=pno, n_experiment=len(centres), n_gravity_pts=n_curve,
                                n_curve_chains=len(curves), **info)
    return rows, meta


def extract_fig13(doc):
    page = doc[10]
    drawings = page.get_drawings()
    labs = _numeric_labels(page)
    # Fig. 13 axes: y labels 0..3 right-aligned at x ~70, x labels 0,15,30,45 at y ~615
    ylab = sorted([lb for lb in labs if 55 < lb["x1"] < 72 and 470 < lb["cy"] < 615], key=lambda d: d["cy"])
    xlab = sorted([lb for lb in labs if 605 < lb["cy"] < 625 and 70 < lb["cx"] < 280], key=lambda d: d["cx"])
    assert [lb["v"] for lb in xlab] == [0, 15, 30, 45], [lb["t"] for lb in xlab]
    box = pymupdf.Rect(xlab[0]["cx"] - 1, ylab[0]["cy"] - 3, xlab[-1]["cx"] + 25, xlab[0]["cy"] - 8)
    fy, fx, info = _axis_maps(page, dict(box=box, xlabels=xlab, ylabels=ylab), drawings)
    diamonds, crosses = [], []
    small = []
    for i, d in enumerate(drawings):
        if not d.get("fill"):
            continue
        r = d["rect"]
        if not box.contains(pymupdf.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)):
            continue
        if 5.5 < r.width < 6.0 and 5.5 < r.height < 6.0:
            diamonds.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
        elif 3.0 < r.width < 3.4 and 3.0 < r.height < 3.4:
            small.append((i, r))
    # CFD 'x' markers: 4 arms of 3.2 pt each; group consecutive indices
    grp = []
    for i, r in small:
        if grp and i == grp[-1][0] + 1 and abs(r.x0 - grp[-1][1].x0) < 4 and abs(r.y0 - grp[-1][1].y0) < 4:
            grp.append((i, r))
        else:
            if len(grp) == 4:
                crosses.append(grp)
            grp = [(i, r)]
    if len(grp) == 4:
        crosses.append(grp)
    out = []
    for cx, cy in diamonds:
        b = round(fx(cx))
        if b in (5, 10, 15, 30, 45):          # the legend sample sits at about 33 deg
            out.append(dict(series="experiment", beta_deg=b, Cfm=fy(cy)))
    for g in crosses:
        xs = [(r.x0 + r.x1) / 2 for _, r in g]
        ys = [(r.y0 + r.y1) / 2 for _, r in g]
        b = round(fx(float(np.mean(xs))))
        if b in (5, 10, 15, 30, 45):
            out.append(dict(series="CFD_FairlieClarke_Tveitnes_2008", beta_deg=b, Cfm=fy(float(np.mean(ys)))))
    return out, info


def extract_fig16(doc):
    page = doc[11]
    drawings = page.get_drawings()
    labs = _numeric_labels(page)
    ylab = sorted([lb for lb in labs if 140 < lb["x1"] < 156 and 540 < lb["cy"] < 722], key=lambda d: d["cy"])
    xlab = sorted([lb for lb in labs if 718 < lb["cy"] < 732 and 150 < lb["cx"] < 460], key=lambda d: d["cx"])
    box = pymupdf.Rect(xlab[0]["cx"] - 1, ylab[0]["cy"] - 3, xlab[-1]["cx"] + 1, xlab[0]["cy"] - 8)
    fy, fx, info = _axis_maps(page, dict(box=box, xlabels=xlab, ylabels=ylab), drawings)
    legend = _text_boxes(page, {"5 deg. v = 0.94 m/s", "10 deg. v = 0.94 m/s", "15 deg. v = 0.94 m/s",
                                "30 deg. v = 0.94 m/s", "45 deg. v = 1.19 m/s"})
    legend_box = pymupdf.Rect(min(b.x0 for b in legend) - 45, min(b.y0 for b in legend) - 3,
                              max(b.x1 for b in legend) + 2, max(b.y1 for b in legend) + 3) if legend else None
    pts = []
    for i, d in enumerate(drawings):
        r = d["rect"]
        if r.width > 40 or r.height > 40:
            continue                       # frame, gridlines
        c = pymupdf.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
        if not box.contains(c) or (legend_box is not None and legend_box.contains(c)):
            continue
        if d.get("fill") or (d.get("width") or 0) > 0:
            pts.append((i, c.x, c.y))
    pts.sort()
    series, cur = [], []
    for i, x, y in pts:
        if cur and x < cur[-1][1] - 25:
            series.append(cur)
            cur = []
        cur.append((i, x, y))
    series.append(cur)
    series = [s for s in series if len(s) >= 15]
    assert len(series) == 5, [len(s) for s in series]
    out = []
    for (beta, V), s in zip(FIG16_SERIES, series):
        for _, x, y in s:
            out.append(dict(beta_deg=beta, V_mps=V, z_over_d=fx(x), cfb_tanb=fy(y)))
    return out, info


def write_csv(path, rows, header_lines):
    with open(path, "w", newline="", encoding="utf-8") as f:
        for h in header_lines:
            f.write("# " + h + "\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


# ----------------------------------------------------------------------------------------------------- Part B
def a_sub(z, beta_deg):
    """Submerged section area below the undisturbed level: wedge (half-beam c, chine height d) + box of beam 2c."""
    T = math.tan(math.radians(beta_deg))
    d = C_HALF * T
    z = np.asarray(z, float)
    return np.where(z <= d, z * z / T, C_HALF * d + 2 * C_HALF * (z - d))


def wen_curve(beta_deg, zd):
    """Wen et al. 2022 decay anchored on the MLM chine-wetting force (as in planing_nl.section_forces)."""
    b = math.radians(beta_deg)
    T = math.tan(b)
    kM = pnl.kappa_mlm(beta_deg)
    csm = kM * (2.0 / math.pi) * T          # C at Wagner chine wetting = kappa * h_cw / c
    ci = pnl.cs_inf(b)
    hs = ((math.pi / 2) * np.asarray(zd, float) - 1.0) / T ** pnl.WEN_LAMBDA
    cs = ci + pnl.wen_cstar(np.maximum(hs, 0.0)) * (csm - ci)
    return np.where(hs >= 0, cs, np.nan), hs, csm, ci


def analyse(fig4, fig13, fig16):
    cfm_exp = {r["beta_deg"]: r["Cfm"] for r in fig13 if r["series"] == "experiment"}
    cfm_cfd = {r["beta_deg"]: r["Cfm"] for r in fig13 if r["series"] != "experiment"}
    by_panel = defaultdict(lambda: defaultdict(list))
    for r in fig4:
        by_panel[r["panel"]][r["series"]].append((r["z_over_d"], r["force_N"]))
    f16 = defaultdict(list)
    for r in fig16:
        f16[r["beta_deg"]].append((r["z_over_d"], r["cfb_tanb"]))

    summary, bins_rows, lines = {}, [], []
    EDGES = [0.65, 0.9, 1.15, 1.4, 1.65, 1.9, 2.2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]
    for letter in sorted(by_panel):
        beta, V = PANELS[letter]
        T = math.tan(math.radians(beta))
        d = C_HALF * T
        fn = V / math.sqrt(G * C_HALF)
        q = RHO * L * C_HALF * V * V                   # dynamic normaliser rho L c V^2 (N)
        exp = np.array(sorted(by_panel[letter]["experiment"]))
        zd, F = exp[:, 0], exp[:, 1]
        z = zd * d
        Fhs = RHO * G * L * a_sub(z, beta)
        Cdyn = (F - Fhs) / q
        # authors' gravity curve vs our hydrostatic
        gcurve = np.array(sorted(by_panel[letter]["gravity_force"])) if by_panel[letter]["gravity_force"] else None
        resid_g = None
        if gcurve is not None and len(gcurve) > 10:
            gz, gF = gcurve[:, 0], gcurve[:, 1]
            gH = RHO * G * L * a_sub(gz * d, beta)
            resid_g = dict(zd=gz, excess_N=gF - gH, excess_over_C=(gF - gH) / q)
        # bins
        wen_all, hs_all, csm, ci = wen_curve(beta, zd)
        pbins = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (zd >= lo) & (zd < hi)
            if m.sum() < 3:
                continue
            zc = float(np.mean(zd[m]))
            cw, hs, _, _ = wen_curve(beta, zc)
            sem = float(np.std(Cdyn[m], ddof=1) / math.sqrt(m.sum()))
            gex = None
            if resid_g is not None:
                mm = (resid_g["zd"] >= lo) & (resid_g["zd"] < hi)
                gex = float(np.mean(resid_g["excess_over_C"][mm])) if mm.any() else None
            row = dict(panel=letter, beta_deg=beta, V_mps=V, Fn_c=round(fn, 3), zd_lo=lo, zd_hi=hi, zd_mean=zc,
                       h_star=float(hs), n=int(m.sum()), C_tveitnes=float(np.mean(Cdyn[m])), C_sem=sem,
                       C_wen=float(cw), Cs_inf=ci, ratio_tv_wen=float(np.mean(Cdyn[m]) / cw),
                       authors_residual_gravity_as_C=gex)
            pbins.append(row)
            bins_rows.append(row)
        # asymptote: z/d >= 2 if available (>= 1.5 for 45 deg), else the last window
        zmax = float(zd.max())
        z_asym = 2.0 if zmax >= 2.6 else (1.5 if zmax >= 2.0 else max(1.2, zmax - 0.5))
        m = zd >= z_asym
        C_asym = float(np.mean(Cdyn[m]))
        C_asym_sem = float(np.std(Cdyn[m], ddof=1) / math.sqrt(m.sum()))
        # coherent-vibration check: means over the two halves and the four quarters of the asymptotic window
        halves = [float(np.mean(Cdyn[(zd >= z_asym) & (zd < (z_asym + zmax) / 2)])), float(np.mean(Cdyn[zd >= (z_asym + zmax) / 2]))]
        qe = np.linspace(z_asym, zmax + 1e-9, 5)
        quarters = [float(np.mean(Cdyn[(zd >= qe[k]) & (zd < qe[k + 1])])) for k in range(4) if ((zd >= qe[k]) & (zd < qe[k + 1])).sum() >= 2]
        cw_asym = float(np.nanmean(wen_all[m]))
        cw_end = float(wen_curve(beta, zmax)[0])
        # peak (chine wetting) force: the maximum of the dynamic coefficient in the chine-wetting region
        ip = int(np.argmax(np.where(zd < 1.3, Cdyn, -np.inf)))
        kM = pnl.kappa_mlm(beta)
        kW = math.pi ** 3 / (4 * T * T)
        kZ = pnl.kappa_wen(math.radians(beta))        # similarity-solution fit of Wen (k2 cot b)
        C_cw = dict(MLM=kM * (2 / math.pi) * T, Wagner=kW * (2 / math.pi) * T, ZF_similarity_fit=kZ * (2 / math.pi) * T)
        f16_peak = f16_asym = None
        if beta in f16 and abs(V - dict(FIG16_SERIES)[beta]) < 1e-9:
            arr = np.array(sorted(f16[beta]))
            f16_peak = float(arr[:, 1].max() / T)
            mm = arr[:, 0] >= min(2.0, arr[:, 0].max() - 0.4)
            f16_asym = float(np.mean(arr[mm, 1]) / T)
        # residual gravity of the authors at the asymptotic window
        g_asym = None
        if resid_g is not None:
            mm = resid_g["zd"] >= z_asym
            if mm.any():
                g_asym = float(np.mean(resid_g["excess_over_C"][mm]))
        summary[letter] = dict(
            beta_deg=beta, V_mps=V, Fn_c=fn, d_m=d, q_N=q, n_points=int(len(zd)), zd_max=zmax,
            noise_N=float(np.std(F[m] - np.polyval(np.polyfit(zd[m], F[m], 1), zd[m]), ddof=1)) if m.sum() > 3 else None,
            asymptote=dict(zd_from=z_asym, n=int(m.sum()), C_tveitnes=C_asym, C_sem=C_asym_sem, C_halves=halves, C_quarters=quarters,
                           C_wen_window_mean=cw_asym, C_wen_at_zd_max=cw_end, Cs_inf=ci, C_frozen=0.0,
                           ratio_tv_over_wen=C_asym / cw_asym, ratio_tv_over_Cs_inf=C_asym / ci,
                           authors_Cfm_fig13=cfm_exp.get(beta), authors_Cfm_CFD_fig13=cfm_cfd.get(beta),
                           authors_fig16_asymptote_C=f16_asym,
                           authors_residual_gravity_as_C=g_asym,
                           C_tveitnes_minus_authors_residual_gravity=(C_asym - g_asym) if g_asym is not None else None),
            chine_wetting=dict(zd_peak=float(zd[ip]), F_peak_N=float(F[ip]), C_peak_raw=float((F[ip] - Fhs[ip]) / q),
                               C_peak_fig16_authors=f16_peak, C_cw_theory=C_cw, C_wen_anchor_MLM=csm),
        )
        a = summary[letter]["asymptote"]
        lines.append(f"{letter}: beta={beta:2d} V={V:.2f} Fn_c={fn:.3f} n={len(zd):3d} zd_max={zmax:.2f} | asym z/d>={z_asym:.1f}: "
                     f"C_Tv={C_asym:+.3f}+-{C_asym_sem:.3f} (quarters {min(quarters):+.2f}..{max(quarters):+.2f})  C_Wen={cw_asym:.3f} (end {cw_end:.3f})  "
                     f"Cs_inf={ci:.3f}  Tv/Wen={C_asym / cw_asym:.2f}  authors Cfm={cfm_exp.get(beta):.3f}  "
                     f"resid.grav(as C)={'n/a' if g_asym is None else f'{g_asym:+.3f}'} | peak: z/d={zd[ip]:.2f} C_raw={(F[ip] - Fhs[ip]) / q:.2f} "
                     f"MLM {C_cw['MLM']:.2f} ZFfit {C_cw['ZF_similarity_fit']:.2f} Wagner {C_cw['Wagner']:.2f}"
                     + (f" Fig16 {f16_peak:.2f}" if f16_peak else ""))
    return summary, bins_rows, lines


def plots(fig4, bins_rows, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    letters = sorted(summary)
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    by = defaultdict(lambda: defaultdict(list))
    for r in fig4:
        by[r["panel"]][r["series"]].append((r["z_over_d"], r["force_N"]))
    for ax, letter in zip(axes.flat, letters):
        beta, V = PANELS[letter]
        d = C_HALF * math.tan(math.radians(beta))
        e = np.array(sorted(by[letter]["experiment"]))
        ax.plot(e[:, 0], e[:, 1], "kx", ms=4, label="experiment (digitised)")
        if by[letter]["gravity_force"]:
            g = np.array(sorted(by[letter]["gravity_force"]))
            ax.plot(g[:, 0], g[:, 1], "r-", lw=1, label="authors' gravity force")
        zz = np.linspace(0, e[:, 0].max(), 200)
        ax.plot(zz, RHO * G * L * a_sub(zz * d, beta), "b--", lw=1, label="hydrostatic rho g L A(z)")
        q = summary[letter]["q_N"]
        wz = zz[zz > 2 / math.pi]
        ax.plot(wz, RHO * G * L * a_sub(wz * d, beta) + q * wen_curve(beta, wz)[0], "g-", lw=1, label="hydrostatic + Wen")
        ax.set_title(f"({letter}) beta={beta} deg, V={V} m/s, Fn_c={summary[letter]['Fn_c']:.2f}")
        ax.set_xlabel("z/d")
        ax.set_ylabel("force N")
        ax.grid(alpha=0.3)
        if letter == "a":
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    # coefficient plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, beta in zip(axes.flat, [5, 10, 15, 30, 45]):
        for r in [b for b in bins_rows if b["beta_deg"] == beta]:
            pass
        for letter in letters:
            if PANELS[letter][0] != beta:
                continue
            rows = [b for b in bins_rows if b["panel"] == letter]
            ax.errorbar([b["zd_mean"] for b in rows], [b["C_tveitnes"] for b in rows], yerr=[b["C_sem"] for b in rows],
                        fmt="o-", ms=4, capsize=2, label=f"Tveitnes V={PANELS[letter][1]} Fn_c={summary[letter]['Fn_c']:.2f}")
        zz = np.linspace(0.65, 8, 300)
        ax.plot(zz, wen_curve(beta, zz)[0], "g-", label="Wen (MLM anchor)")
        ax.axhline(pnl.cs_inf(math.radians(beta)), color="g", ls=":", label="Cs_inf")
        ax.axhline(0, color="k", ls="--", label="frozen (0)")
        ax.set_title(f"beta={beta} deg")
        ax.set_xlabel("z/d")
        ax.set_ylabel("C = (F - F_hydrostatic)/(rho L c V^2)")
        ax.set_ylim(-1, 4)
        ax.set_xlim(0.5, 8.2)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(path.replace(".png", "_coef.png"), dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    doc = pymupdf.open(PDF)
    fig4, meta4 = extract_fig4(doc)
    fig13, info13 = extract_fig13(doc)
    fig16, info16 = extract_fig16(doc)
    src = "Tveitnes, Fairlie-Clarke & Varyani (2008) Ocean Eng. 35:1463-1478, author PDF (Strathclyde), vector charts read from the PDF drawing paths"
    write_csv(os.path.join(DATA_DIR, "digitised_fig4_entry_force.csv"), fig4, [
        src, "Fig. 4 (pp. 1467-1468): constant-velocity water entry. force_N = sum of the four load cells averaged over 4 runs, MODEL WEIGHT REMOVED, HYDROSTATIC FORCE INCLUDED.",
        "series 'experiment' = the x markers (centre of the 4 arm capsules); series 'gravity_force' = the authors' solid curve (their hydrostatic + fitted residual gravity).",
        "z = keel depth below the undisturbed surface, d = chine height = 0.3 m * tan(beta). Section length 0.3 m, beam 0.6 m.",
        "Reading accuracy: axis calibration from gridlines/tick marks and label positions (fit residual < 0.3 pt); marker centre +-0.15 pt; 1 pt = see README (0.5-5 N, 0.004-0.03 in z/d)."])
    write_csv(os.path.join(DATA_DIR, "digitised_fig13_cfm.csv"), fig13, [
        src, "Fig. 13 (p. 1473): flow-momentum drag coefficient Cfm, F_fm = Cfm (rho/2) L (2 y_w) v^2, y_w = B/2 after chine wetting (so Cfm = f/(rho c V^2)).",
        "series 'experiment' = diamonds (one value per deadrise, fitted by the authors to the asymptote of the entry force; no velocity dependence resolved), 'CFD_...' = crosses.",
        "Reading accuracy: +-0.01 in Cfm."])
    write_csv(os.path.join(DATA_DIR, "digitised_fig16_cfb_tanb.csv"), fig16, [
        src, "Fig. 16 (p. 1474): the authors' ESTIMATED (de-noised through their rig simulation) hydrodynamic entry force coefficient Cfb*tan(beta) vs z/d,",
        "Cfb = Force/((rho/2) L B v^2) = f/(rho c V^2) with the hydrostatic and residual gravity forces removed by the authors. Series assigned in legend order (= drawing order).",
        "Points are the centres of the dash/dot pieces of each curve. Reading accuracy: +-0.02 in Cfb*tan(beta), +-0.01 in z/d."])
    summary, bins_rows, lines = analyse(fig4, fig13, fig16)
    with open(os.path.join(OUT_DIR, "v5_bins.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bins_rows[0].keys()))
        w.writeheader()
        for r in bins_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    consts = {b: dict(Cs_inf=pnl.cs_inf(math.radians(b)), kappa_MLM=pnl.kappa_mlm(b), kappa_ZF_fit_Wen=pnl.kappa_wen(math.radians(b)),
                      kappa_Wagner=math.pi ** 3 / (4 * math.tan(math.radians(b)) ** 2),
                      C_cw_MLM=pnl.kappa_mlm(b) * 2 / math.pi * math.tan(math.radians(b)),
                      zd_chine_wetting_Wagner=2 / math.pi, zd_chine_wetting_authors_WF=None) for b in (5, 10, 15, 30, 45)}
    out = dict(constants=consts, extraction_fig4=meta4, extraction_fig13=info13, extraction_fig16=info16,
               fig13=fig13, runs=summary, rho=RHO, g=G, L=L, c=C_HALF)
    with open(os.path.join(OUT_DIR, "v5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    with open(os.path.join(OUT_DIR, "v5_summary.txt"), "w", encoding="utf-8") as f:
        f.write("V5: Tveitnes et al. 2008 post-chine dynamic coefficient C = (F - rho g L A_sub)/(rho L c V^2) vs frozen (0) and Wen (MLM anchor)\n")
        f.write(f"rho={RHO} g={G} L={L} c={C_HALF}; Fn_c = V/sqrt(g c)\n\n")
        f.write("constants per deadrise: " + json.dumps(consts, default=lambda x: round(x, 4) if isinstance(x, float) else x) + "\n\n")
        f.write("Fig.13 authors' Cfm: " + ", ".join(f"{r['beta_deg']}deg {r['series'][:3]} {r['Cfm']:.3f}" for r in fig13) + "\n\n")
        f.write("\n".join(lines) + "\n\n")
        f.write("bins (panel, z/d window, h*, n, C_Tv +- sem, C_Wen, ratio, authors' residual gravity as C):\n")
        for r in bins_rows:
            f.write(f"{r['panel']} b={r['beta_deg']:2d} V={r['V_mps']:.2f} zd[{r['zd_lo']:.2f},{r['zd_hi']:.2f}) h*={r['h_star']:6.2f} n={r['n']:3d} "
                    f"C={r['C_tveitnes']:+.3f}+-{r['C_sem']:.3f}  Wen={r['C_wen']:.3f}  ratio={r['ratio_tv_wen']:+.2f}  "
                    f"rg={'n/a' if r['authors_residual_gravity_as_C'] is None else f'{r['authors_residual_gravity_as_C']:+.3f}'}\n")
        f.write("\nextraction meta: " + json.dumps(meta4, default=float) + "\n")
    plots(fig4, bins_rows, summary, os.path.join(OUT_DIR, "v5_fig4_check.png"))
    print("\n".join(lines))
    print("extraction:", {k: (v["n_experiment"], v["n_gravity_pts"], round(v["y_fit_resid"], 2), round(v["x_fit_resid"], 3)) for k, v in meta4.items()})
    print("fig13:", [(r["series"][:3], r["beta_deg"], round(r["Cfm"], 3)) for r in fig13])
    print("fig16 n per series:", {b: sum(1 for r in fig16 if r["beta_deg"] == b) for b in (5, 10, 15, 30, 45)})


if __name__ == "__main__":
    main()
