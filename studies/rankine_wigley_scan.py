"""
Re-reading of the scanned Wigley resistance figure (Hendrix & Noblesse 1988,
copy shipped with WaveBEM as tokyoWigley.pdf, page "- 61 -", "Wave-resistance
coefficients C_W & C_WP for the Wigley hull held in fixed position"; their data
reference: McCarthy 1985, DTNSRDC-85/011) for the B4 review fixes (stage note
learn/meta/RANKINE_STAGE1_2026-10-06.md section 9.7; review issue "V1 reference
data faulty").

What it does (no data are typed in by hand except the approximate marker
positions listed below, which the script refines):
  1. extracts the page's embedded 300-dpi bilevel scan at native resolution;
  2. calibrates both panels (top: 10^3 C_W, bottom: 10^3 C_WP) from the tick
     marks it detects on the frame lines (affine map pixel -> (Fn, C), fitted
     separately per panel to the x ticks on the lower and upper frame and the
     y ticks on the left and right frame, so a small scan rotation/skew is
     absorbed); prints the fit residuals;
  3. refines the marker centres from approximate positions read on zoomed
     crops (centroid of the dark pixels of the marker blob after a 3x3
     morphological opening, within +-7 px; markers merged with a curve or
     with another marker are flagged and keep a larger reading error);
  4. writes the C_WP markers of S.R.I. and U. Tokyo (all of them) and the C_W
     markers in the window Fn 0.185-0.215 (lab scatter at low Fn, used for the
     absolute tolerance floor of the amended rows) to data/external/wigley/;
  5. draws the overlay figure studies/_cache/rankine/b4_cw_overlay.png: the four
     WaveBEM C_W series (wavebem_wigleyDragExp1-4.csv) and the U. Tokyo vector
     curve of Yang et al. (2000) fig 5a (fixed) mapped onto the scan.

Reading error: +-3 px for isolated markers (+-0.0012 in Fn, +-0.025e-3 in C_W,
+-0.027e-3 in C_WP), +-5 px for flagged (merged) markers. Symbol identification
in the C_W panel at Fn ~ 0.2 (S.R.I. open circle vs U. Tokyo open square, U.
Tokyo filled square vs Y.N.U. cross) is uncertain for some markers at 300 dpi;
the assignment is given per marker with a "?" where uncertain.

Run: .venv/Scripts/python.exe studies/rankine_wigley_scan.py
"""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "external", "wigley")
OUT = os.path.join(ROOT, "studies", "_cache", "rankine")
PDF = os.path.join(DATA, "wavebem_tokyoWigley.pdf")
CROP = (1400, 500, 2450, 1800)          # figure region of the page image (native px); coordinates below are in this crop

# Panels (crop px): frame lines found as long dark rows/columns (see main(): printed)
PANELS = {
    "CW": dict(top=43.5, bottom=612.5, zero=494.5, left=86.5, right=931.5, yvals=(3.0, 2.0, 1.0), px_c=118.8),
    "CWP": dict(top=649.0, bottom=1196.0, zero=1087.5, left=86.5, right=931.5, yvals=(3.0, 2.0, 1.0), px_c=109.8),
}
FN_TICKS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)

# Approximate marker centres (crop px) read on 3x-14x zoomed crops of the C_WP panel; lab, flag
CWP_MARKERS = [
    ((409.3, 1046.0), "SRI", ""), ((459.3, 1039.0), "SRI", ""), ((508.6, 1009.2), "SRI+UT", "merged"),
    ((550.9, 1016.1), "UT", ""), ((604.0, 979.3), "SRI+UT", "merged"), ((632.3, 955.7), "SRI", ""),
    ((670.0, 923.0), "SRI+UT", "merged with the curves"), ((706.7, 940.3), "SRI", ""), ((756.0, 956.7), "SRI", ""),
    ((766.7, 964.0), "UT", ""), ((831.7, 934.3), "SRI", ""), ((881.7, 904.3), "SRI", ""), ((900.7, 860.4), "UT", ""),
]
# C_W markers in the window Fn ~0.185-0.215 (crop px), read on a 10x crop; symbol -> lab
CW_MARKERS_020 = [
    ((359.0, 453.0), "UT? (filled square or YNU cross; merged pair)", "merged"),
    ((371.0, 453.0), "UT (diamond; merged pair)", "merged"),
    ((384.5, 453.0), "UT (filled square)", ""),
    ((384.5, 467.5), "SRI (open circle)", ""),
    ((359.0, 471.5), "SRI (open circle)", ""),
    ((354.5, 483.0), "YNU? (cross)", ""),
    ((380.0, 489.0), "YNU (cross)", ""),
    ((396.5, 494.5), "YNU (cross, on the zero line)", "merged"),
    ((403.0, 437.0), "UT? (open square; merged with a solid marker)", "merged"),
    ((408.5, 452.5), "SRI? (open circle or UT open square)", ""),
    ((415.0, 470.0), "YNU (cross)", ""),
]


def _runs(v):
    idx = np.nonzero(v)[0]
    if len(idx) == 0:
        return []
    grp = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    return [(int(g[0]), int(g[-1])) for g in grp]


def load_scan():
    import pymupdf
    from PIL import Image
    d = pymupdf.open(PDF)
    p = d[0]
    xref = p.get_images()[0][0]
    pix = pymupdf.Pixmap(d, xref)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    a = np.asarray(img) < 128
    x0, y0, x1, y1 = CROP
    return a[y0:y1, x0:x1], (pix.width, pix.height)


def _ticks_on_hline(R, yline, direction, lo=95, hi=925, minh=3):
    """x positions of tick bumps adjacent to a horizontal frame line at row
    yline (direction 'up' = bumps above the line, 'down' = below)."""
    y = int(round(yline))
    h = np.zeros(R.shape[1], int)
    for c in range(lo, hi):
        n = 0
        rng = range(y - 2, y - 16, -1) if direction == "up" else range(y + 2, y + 16)
        for r in rng:
            if R[r, c]:
                n += 1
            else:
                break
        h[c] = n
    return [0.5 * (a + b) for a, b in _runs(h >= minh)]


def _ticks_on_vline(R, xline, direction, lo, hi, minw=3):
    x = int(round(xline))
    w = np.zeros(R.shape[0], int)
    for r in range(lo, hi):
        n = 0
        rng = range(x + 2, x + 16) if direction == "right" else range(x - 2, x - 16, -1)
        for c in rng:
            if R[r, c]:
                n += 1
            else:
                break
        w[r] = n
    return [0.5 * (a + b) for a, b in _runs(w >= minw)]


def _line_centre(R, y_guess, x_lo, x_hi):
    """Centre row of a long horizontal line near y_guess over columns x_lo..x_hi."""
    rows = np.arange(int(y_guess) - 6, int(y_guess) + 7)
    frac = R[rows][:, x_lo:x_hi].mean(1)
    sel = rows[frac > 0.5]
    return float(sel.mean())


def calibrate(R, name):
    """Affine pixel -> (Fn, C) map of a panel from its detected ticks."""
    P = PANELS[name]
    # x ticks on the lower and upper frame lines
    xb = _ticks_on_hline(R, P["bottom"], "up")
    xt = _ticks_on_hline(R, P["top"], "down")

    def pick(xs):
        # the 7 ticks: nearest detections to the nominal spacing from the first one
        xs = np.array(xs)
        out = []
        for k in range(7):
            guess = 136.5 + 124.25 * k
            j = int(np.argmin(abs(xs - guess)))
            if abs(xs[j] - guess) > 6:
                raise RuntimeError(f"{name}: x tick {k} not found near {guess}")
            out.append(float(xs[j]))
        return out
    xb, xt = pick(xb), pick(xt)
    # y ticks on the left and right frames (3, 2, 1), plus the zero line at both ends
    yl = _ticks_on_vline(R, P["left"], "right", int(P["top"]) + 8, int(P["zero"]) - 8)
    yr = _ticks_on_vline(R, P["right"], "left", int(P["top"]) + 8, int(P["zero"]) - 8)
    span = P["px_c"]          # nominal px per unit, only to locate the ticks

    def picky(ys):
        ys = np.array(ys)
        out = []
        for v in P["yvals"]:
            guess = P["zero"] - v * span
            j = int(np.argmin(abs(ys - guess)))
            if abs(ys[j] - guess) > 10:
                raise RuntimeError(f"{name}: y tick {v} not found near {guess}")
            out.append(float(ys[j]))
        return out
    yl, yr = picky(yl), picky(yr)
    # zero line: centre rows over two column bands free of markers and curves (Fn ~0.29-0.33 and ~0.39-0.42)
    zl = _line_centre(R, P["zero"], 600, 700)
    zr = _line_centre(R, P["zero"], 850, 920)
    # Fn = a0 + a1 px + a2 py
    Ax, bx = [], []
    for xs, yrow in ((xb, P["bottom"]), (xt, P["top"])):
        for fn, xp in zip(FN_TICKS, xs):
            Ax.append([1.0, xp, yrow])
            bx.append(fn)
    a, *_ = np.linalg.lstsq(np.array(Ax), np.array(bx), rcond=None)
    Ay, by = [], []
    for ys, xcol in ((yl, P["left"]), (yr, P["right"])):
        for v, yp in zip(P["yvals"], ys):
            Ay.append([1.0, xcol, yp])
            by.append(v)
    for z, xcol in ((zl, 650.0), (zr, 885.0)):
        Ay.append([1.0, xcol, z])
        by.append(0.0)
    b, *_ = np.linalg.lstsq(np.array(Ay), np.array(by), rcond=None)
    rx = np.array(Ax) @ a - np.array(bx)
    ry = np.array(Ay) @ b - np.array(by)
    return dict(a=a.tolist(), b=b.tolist(), x_ticks_bottom=xb, x_ticks_top=xt, y_ticks_left=yl, y_ticks_right=yr,
                zero_left=zl, zero_right=zr, resid_Fn_max=float(abs(rx).max()), resid_C_max=float(abs(ry).max()),
                px_per_Fn=float(1.0 / a[1]), px_per_C=float(-1.0 / b[2]))


def to_data(cal, xp, yp):
    a, b = cal["a"], cal["b"]
    return a[0] + a[1] * xp + a[2] * yp, b[0] + b[1] * xp + b[2] * yp


def to_px(cal, fn, c):
    a, b = np.array(cal["a"]), np.array(cal["b"])
    M = np.array([[a[1], a[2]], [b[1], b[2]]])
    rhs = np.stack([np.asarray(fn) - a[0], np.asarray(c) - b[0]])
    return np.linalg.solve(M, rhs)


def refine(R, xy, win=7):
    from scipy import ndimage as nd
    x, y = xy
    x0, y0 = int(round(x)) - win, int(round(y)) - win
    sub = R[y0:y0 + 2 * win + 1, x0:x0 + 2 * win + 1]
    op = nd.binary_opening(sub, np.ones((3, 3)))
    if op.sum() < 5:
        return xy, 0
    yy, xx = np.nonzero(op)
    return (float(x0 + xx.mean()), float(y0 + yy.mean())), int(op.sum())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    R, shape = load_scan()
    out = dict(scan_px=shape, crop=CROP)
    cals = {n: calibrate(R, n) for n in PANELS}
    out["calibration"] = cals
    # C_WP markers
    cwp = []
    for (xy, lab, flag) in CWP_MARKERS:
        (xr, yr), n = refine(R, xy)
        shift = float(np.hypot(xr - xy[0], yr - xy[1]))
        use = (xr, yr) if (shift <= 3.0 and not flag) else xy
        fn, c = to_data(cals["CWP"], *use)
        err_px = 5.0 if flag else 3.0
        cwp.append(dict(Fn=round(float(fn), 4), CWP_1e3=round(float(c), 3), lab=lab, flag=flag, px=[round(use[0], 1),
                        round(use[1], 1)], refine_shift_px=round(shift, 1),
                        err_Fn=round(err_px / cals["CWP"]["px_per_Fn"], 4),
                        err_C_1e3=round(err_px / cals["CWP"]["px_per_C"], 3)))
    cw = []
    for (xy, lab, flag) in CW_MARKERS_020:
        (xr, yr), n = refine(R, xy)
        shift = float(np.hypot(xr - xy[0], yr - xy[1]))
        use = (xr, yr) if (shift <= 3.0 and not flag) else xy
        fn, c = to_data(cals["CW"], *use)
        err_px = 5.0 if flag else 3.0
        cw.append(dict(Fn=round(float(fn), 4), CW_1e3=round(float(c), 3), lab=lab, flag=flag,
                       px=[round(use[0], 1), round(use[1], 1)], refine_shift_px=round(shift, 1),
                       err_C_1e3=round(err_px / cals["CW"]["px_per_C"], 3)))
    out["CWP_markers"] = cwp
    out["CW_markers_Fn020"] = cw
    # U. Tokyo vector curve (Yang et al. 2000 fig 5a, fixed) at the C_W markers: lab scatter about it
    yang = np.loadtxt(os.path.join(DATA, "yang2000_fig5a_cw_fixed.csv"), delimiter=",", comments="#")
    yang = yang[np.argsort(yang[:, 0], kind="stable")]
    for m in cw:
        m["UT_curve_1e3"] = round(float(np.interp(m["Fn"], yang[:, 0], yang[:, 1]) * 1e3), 3)
        m["minus_curve_1e3"] = round(m["CW_1e3"] - m["UT_curve_1e3"], 3)
    # files
    f1 = os.path.join(DATA, "hn1988_p61_cwp_markers.csv")
    with open(f1, "w", encoding="utf-8") as f:
        f.write("# Fn, 10^3 C_WP, lab (SRI, UT, SRI+UT = overlapping markers read as one), reading error Fn, "
                "reading error 10^3 C_WP; source: Hendrix & Noblesse (1988) p. 61 lower panel (scan in "
                "wavebem_tokyoWigley.pdf), data McCarthy 1985 DTNSRDC-85/011; read by studies/rankine_wigley_scan.py\n")
        for m in cwp:
            f.write(f"{m['Fn']:.4f},{m['CWP_1e3']:.3f},{m['lab']},{m['err_Fn']:.4f},{m['err_C_1e3']:.3f}\n")
    f2 = os.path.join(DATA, "hn1988_p61_cw_markers_fn020.csv")
    with open(f2, "w", encoding="utf-8") as f:
        f.write("# Fn, 10^3 C_W, lab/symbol (? = uncertain identification), reading error 10^3 C_W; markers of the "
                "upper panel of Hendrix & Noblesse (1988) p. 61 in the window Fn 0.185-0.215 only (lab scatter at "
                "low Fn); read by studies/rankine_wigley_scan.py\n")
        for m in cw:
            f.write(f"{m['Fn']:.4f},{m['CW_1e3']:.3f},{m['lab']},{m['err_C_1e3']:.3f}\n")
    out["files"] = {os.path.relpath(p, ROOT).replace("\\", "/"): sha256(p) for p in (f1, f2)}
    # overlay figure
    from PIL import Image, ImageDraw
    img = Image.fromarray(np.where(R, 0, 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(img)
    colours = {1: (0, 160, 0), 2: (0, 0, 255), 3: (255, 0, 0), 4: (255, 140, 0)}
    for k in (1, 2, 3, 4):
        s = np.loadtxt(os.path.join(DATA, f"wavebem_wigleyDragExp{k}.csv"), delimiter=",", comments="%")
        px = to_px(cals["CW"], s[:, 0], s[:, 1] * 1e3)
        for xp, yp in zip(*px):
            d.ellipse((xp - 7, yp - 7, xp + 7, yp + 7), outline=colours[k], width=2)
    px = to_px(cals["CW"], yang[:, 0], yang[:, 1] * 1e3)
    d.line(list(zip(px[0], px[1])), fill=(200, 0, 200), width=3)
    for m in cwp:
        xp, yp = m["px"]
        d.rectangle((xp - 9, yp - 9, xp + 9, yp + 9), outline=(0, 170, 170), width=2)
    for m in cw:
        xp, yp = m["px"]
        d.rectangle((xp - 7, yp - 7, xp + 7, yp + 7), outline=(0, 170, 170), width=1)
    d.text((100, 615), "circles: WaveBEM wigleyDragExp1 (green, SRI), 2 (blue, UT1), 3 (red, UT2), 4 (orange, UT3);"
           " magenta: U. Tokyo curve, Yang et al. 2000 fig 5a; cyan boxes: markers read here", fill=(0, 0, 0))
    os.makedirs(OUT, exist_ok=True)
    fo = os.path.join(OUT, "b4_cw_overlay.png")
    img.save(fo)
    out["overlay"] = os.path.relpath(fo, ROOT).replace("\\", "/")
    # WaveBEM series minus the U. Tokyo curve (Fn < 0.3), the transcription defect in numbers
    dev = {}
    for k in (1, 2, 3, 4):
        s = np.loadtxt(os.path.join(DATA, f"wavebem_wigleyDragExp{k}.csv"), delimiter=",", comments="%")
        m = s[:, 0] < 0.3
        dev[f"Exp{k}"] = dict(Fn_range=[float(s[:, 0].min()), float(s[:, 0].max())],
                             mean_minus_UTcurve_Fn_lt_0p3=round(float(np.mean(
                                 s[m, 1] * 1e3 - np.interp(s[m, 0], yang[:, 0], yang[:, 1] * 1e3))), 3))
    out["wavebem_series_minus_UT_curve_1e3"] = dev
    js = os.path.join(OUT, "b4_scan_reading.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
