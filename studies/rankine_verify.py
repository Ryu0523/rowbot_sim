#!/usr/bin/env python3
"""
Verification of hydro/rankine (WaveBEM port, stage 1), tiers per
learn/meta/ACCEPTANCE_2026-10-06.md T1:

    quick  <= 10 min   coarse members of each mesh series
    full   <= 1 h      mesh series to the finest member, sweeps

    .venv/Scripts/python.exe studies/rankine_verify.py --tier quick
        [--stages stage1,stage2,stage3] [--only C3,Z1]
    (env RANKINE_GB_PER_JOB = memory estimate per CPU worker, default 0.6;
     use 1.5 for the full tier, whose Capytaine reference needs ~1 GB)

Stage 1 = BIE core (C*), stage 2 = zero-speed free surface (Z*), stage 3 =
steady linear free surface (S*). Thresholds are those registered in
learn/meta/WAVEBEM_PORT_DESIGN.md section 8.4 BEFORE these runs (V1); where
the design gave none, the value written in the check's docstring was
registered here before its first run (see learn/meta/RANKINE_STAGE1_2026-10-06.md,
section 1). Every number is reported with its reference value and source; a
failing check is reported as FAIL, never re-tuned; rows added after a first
run are labelled "[added after the first run]"; rows added or re-registered
after the 2026-10-06 review are labelled "[added after review]" /
"[re-registered after review]" (registration: stage note section 1.1).
Measured 2026-10-06 after the review: quick 234 s (45/48 pass), full 1798 s
(70/74 pass); a rerun of the quick stage-3 rows in a new process is bitwise
identical. Before the review: quick 204 s (36/38), full 2055 s (50/55).

Stage B3 (2026-10-06, stage note section 8): stage-3 rows use the production
free-surface operator of hydro/rankine/shifted.py (shifted Numerov rows, box
correction, FFT-GMRES); new rows D1a-e (coupled symbol analysis,
studies/rankine_d1.py), F1-F3 (FFT lattice and FFT-GMRES vs dense), S3b at
45 and 60 panels per wavelength, S3d (amplitude trend), S9 (g -> 0 limit);
S1/S2 series extended to 101.25 panels per wavelength (N ~ 1e5). Registered
before the run in stage note section 8.1; rows marked "[B3]". Partial runs
(--stages or --only) write their own json name (verify_<tier>_<stages>...),
so they cannot overwrite a complete tier record. Measured after B3: quick
174 s (56/58 pass), full 1562 s (85/88; fails: S2 Fn 0.4 f/b 4, S4 and S8,
all as registered); stage 1-2 rows identical to the post-review record,
stage-3 rows bitwise identical between two processes.

B3 review (2026-10-06, stage note section 8.8, registered before the
re-run): S2K (Rankine NK vs an independent exact-Kelvin body-only BIE,
hydro/rankine/kelvin.py) and S2K-G (checks of the Kelvin function); S2m
body-mesh series; S2 relabelled information (interaction size, registered
5 % kept), S1 verification (M2/M5); S3 wavelength reference = the same
estimator on the exact elevation; S4b split forcing consistent with the
blend and S4b members on grids per lambda0 (production rows); S9r
(Richardson in 1/Fn); GMRES residual floor 1e-10 (solve raises above it)
and the residuals and guard state printed in every stage-3 row; binding
and information rows counted separately; registration hashes printed at
the start of every tier and stored in its json (--register writes them).
Measured after the B3 review fixes (registered 2026-10-06T15:48:50 UTC):
quick 328 s (binding 58/60, 6 information rows), full 3496 s (binding
83/86, 18 information rows; fails S2 Fn 0.4 f/b 4, S4, S8, all as
registered), with another track's job sharing the machine.

Independent CPU checks run in parallel through studies/parallel_runner.py
(reserve_gb=3.0, plus 2 GB for this process when GPU jobs run), in a thread,
while the checks that need the GPU (large dense solves) run one at a time in
the main process. Results: studies/_cache/rankine/verify_<tier>.json.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "studies", "_cache", "rankine")


def _cpu_only():
    import torch
    torch.set_num_threads(1)
    return "cpu"


def _order(h, e):
    """Observed convergence order from successive (h, error) pairs, for
    errors against an EXACT reference. NaN where the error changes sign
    (the ratio of magnitudes is then meaningless; review 2026-10-06). Series
    without an exact reference use _richardson (successive differences)."""
    h, e = np.asarray(h, float), np.asarray(e, float)
    out = []
    for k in range(len(e) - 1):
        if e[k] * e[k + 1] <= 0:
            out.append(float("nan"))
        else:
            out.append(float(np.log(abs(e[k] / e[k + 1])) / np.log(h[k] / h[k + 1])))
    return out


def _row(check, quantity, value, ref, source, tol, ok, note=""):
    return dict(check=check, quantity=quantity, value=value, ref=ref,
                source=source, tol=tol, ok=bool(ok), note=note)


# =====================================================================
# Stage 1 -- BIE core in unbounded fluid and with images (design step R1)
# =====================================================================
def c1_panel_integrals(args):
    """C1 (M2): closed-form S, D, grad S of a flat panel vs independent
    adaptive quadrature, near (1e-4 panel sizes) to far (100 sizes), in plane
    inside/outside, near a corner. Threshold: relative 1e-8 (design R1).
    Added after review: far points at 1e3 and 1e4 sizes, relative 1e-10.
    The in-plane D row is non-binding since the review: the closed form
    sets D = 0 there by its |z| <= 1e-13 s branch, so the row only measured
    the reference's round-off (replaced by the near-plane rows of C1b)."""
    seed, = args
    dev = _cpu_only()
    from hydro.rankine.panels import Panels
    from hydro.rankine.influence import influence, quad_reference
    import warnings
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(seed)
    base = [np.array([[0, 0, 0], [1.0, 0.1, 0], [1.2, 0.9, 0], [0.1, 0.8, 0]]),
            np.array([[0, 0, 0], [1.0, 0, 0], [0.4, 0.7, 0], [0.4, 0.7, 0]]),
            np.array([[0, 0, 0], [3.0, 0, 0], [3.0, 0.3, 0], [0, 0.3, 0]])]
    q = base[seed % len(base)].astype(float)
    Rm = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(Rm) < 0:
        Rm[:, 0] *= -1
    q = q @ Rm.T + rng.normal(size=3)
    pan = Panels(q[None])
    n, c = pan.n[0], pan.c[0]
    size = pan.size[0]
    pts, kinds = [], []
    for dist in (1e-4, 1e-3, 1e-2, 0.1, 0.5, 2.0, 10.0, 100.0):
        for _ in range(2):
            u = rng.normal(size=3)
            pts.append(c + dist * size * u / np.linalg.norm(u))
            kinds.append("off")
    far_from = len(pts)
    for dist in (1e3, 1e4):                        # added after review
        for _ in range(2):
            u = rng.normal(size=3)
            pts.append(c + dist * size * u / np.linalg.norm(u))
            kinds.append("off")
    pts.append(c + 1e-4 * size * n + 0.3 * (q[1] - q[0])); kinds.append("off")
    pts.append(q[2] + 1e-3 * size * n); kinds.append("off")
    pts.append(c + 0.6 * (q[1] - c)); kinds.append("in-plane inside")
    pts.append(c + 1.7 * (q[1] - c)); kinds.append("in-plane outside")
    pts = np.array(pts)
    S, D, G = influence(pts, pan, want_grad=True, device=dev)
    worst_rel, worst_far, worst_abs_inplane = 0.0, 0.0, 0.0
    for i, x in enumerate(pts):
        sr = quad_reference(x, pan.verts[0], n, "S")
        dr = quad_reference(x, pan.verts[0], n, "D")
        e = abs(S[i, 0] - sr) / abs(sr)
        if kinds[i] == "off":
            gr = quad_reference(x, pan.verts[0], n, "G")
            e = max(e, abs(D[i, 0] - dr) / abs(dr), np.linalg.norm(G[i, 0] - gr) / np.linalg.norm(gr))
        else:
            worst_abs_inplane = max(worst_abs_inplane, abs(D[i, 0] - dr))
        if i >= far_from and kinds[i] == "off":
            worst_far = max(worst_far, e)
        else:
            worst_rel = max(worst_rel, e)
    return [_row("C1", f"panel {seed} ({len(pts)} points to 100 sizes): max rel. error S, D, gradS",
                 worst_rel, 0.0, "adaptive quadrature (influence.quad_reference)",
                 1e-8, worst_rel <= 1e-8),
            _row("C1", f"[added after review] panel {seed}, 4 points at 1e3 and 1e4 sizes: max rel. error S, D, gradS",
                 worst_far, 0.0, "adaptive quadrature (influence.quad_reference)", 1e-10, worst_far <= 1e-10),
            _row("C1", f"panel {seed}: in-plane D abs. error [non-binding since review: D = 0 by construction]",
                 worst_abs_inplane, 0.0, "exact (solid angle 0)", "info", True)]


def c1b_rectangle(args):
    """C1b (M2, added after review): an axis-aligned rectangle with binary-
    fraction corners in z = 0, so that the closed form and the reference see
    the same inputs exactly; reference: the rectangle antiderivative
    F(u, v, z) = u ln(v + R) + v ln(u + R) - z atan(uv/(zR)) for S and
    sum atan(uv/(zR)) for D (Hess & Smith 1964) in 50-digit mpmath, grad S by
    mpmath differentiation of S. Points: far, 10 to 1e5 panel sizes (binding
    1e-10 up to 1e4, information beyond); near-plane, z = +-1e-9 h above the
    interior, 1e-3 h and 1e-6 h from an edge, 1e-4 h from a vertex, inside
    and outside (binding 1e-8)."""
    _cpu_only()
    import mpmath as mp
    from hydro.rankine.panels import Panels
    from hydro.rankine.influence import influence
    mp.mp.dps = 50
    x1, x2, y1, y2 = -0.375, 0.875, -0.25, 0.1875
    pan = Panels(np.array([[[x1, y1, 0], [x2, y1, 0], [x2, y2, 0], [x1, y2, 0]]], float))
    assert pan.n[0, 2] == 1.0
    h = float(pan.size[0])

    def F(u, v, z):
        R = mp.sqrt(u * u + v * v + z * z)
        t = u * mp.log(v + R) + v * mp.log(u + R)
        return t - z * mp.atan(u * v / (z * R)) if z != 0 else t

    def S_ref(X, Y, Z):
        return (F(x2 - X, y2 - Y, Z) - F(x1 - X, y2 - Y, Z) - F(x2 - X, y1 - Y, Z)
                + F(x1 - X, y1 - Y, Z))

    def D_ref(X, Y, Z):
        def A(u, v):
            return mp.atan(u * v / (Z * mp.sqrt(u * u + v * v + Z * Z)))
        return A(x2 - X, y2 - Y) - A(x1 - X, y2 - Y) - A(x2 - X, y1 - Y) + A(x1 - X, y1 - Y)
    rng = np.random.default_rng(11)
    centre = np.array([0.25, -0.03125, 0.0])
    pts, kind = [], []
    for dist in (10.0, 100.0, 1e3, 1e4, 3e4, 1e5):
        for _ in range(3):
            u = rng.normal(size=3)
            pts.append(centre + dist * h * u / np.linalg.norm(u))
            kind.append("far" if dist <= 1e4 else "far-info")
    zt = 1e-9 * h
    for sz in (1.0, -1.0):
        for xy in ((0.1, 0.0), (x2 - 1e-3 * h, 0.0), (x2 + 1e-3 * h, 0.0), (0.1, y2 - 1e-6 * h),
                   (0.1, y2 + 1e-6 * h), (x2 - 1e-4 * h, y2 - 1e-4 * h), (x2 + 1e-4 * h, y2 + 1e-4 * h)):
            pts.append(np.array([xy[0], xy[1], sz * zt]))
            kind.append("near")
    pts = np.array(pts)
    S, D, G = influence(pts, pan, want_grad=True, device="cpu")
    worst = {"far": 0.0, "far-info": 0.0, "near": 0.0}
    for i, p in enumerate(pts):
        X, Y, Z = (mp.mpf(float(c)) for c in p)
        sr, dr = S_ref(X, Y, Z), D_ref(X, Y, Z)
        gr = [mp.diff(lambda q: S_ref(q, Y, Z), X), mp.diff(lambda q: S_ref(X, q, Z), Y),
              mp.diff(lambda q: S_ref(X, Y, q), Z)]
        es = abs(float((mp.mpf(float(S[i, 0])) - sr) / sr))
        ed = abs(float((mp.mpf(float(D[i, 0])) - dr) / dr))
        g = np.array([float(c) for c in gr])
        eg = float(np.linalg.norm(G[i, 0] - g) / np.linalg.norm(g))
        worst[kind[i]] = max(worst[kind[i]], es, ed, eg)
    return [_row("C1b", "[added after review] rectangle, far points 10..1e4 panel sizes: max rel. error S, D, gradS",
                 worst["far"], 0.0, "50-digit closed form (mpmath)", 1e-10, worst["far"] <= 1e-10,
                 note=f"3e4..1e5 sizes (info): {worst['far-info']:.2e}"),
            _row("C1b", "[added after review] rectangle, near-plane points z = +-1e-9 h (interior, edges, vertex): "
                 "max rel. error S, D, gradS", worst["near"], 0.0, "50-digit closed form (mpmath)", 1e-8,
                 worst["near"] <= 1e-8)]


def c2_green_identity(args):
    """C2 (M3, M5): Green's identity on a closed sphere (full mesh, no images,
    so the raw operator is tested). Interior phi = 1 (identity (I) is exact
    for a closed flat-panel polyhedron: residual at round-off, threshold
    1e-10), interior phi = x, x^2 - y^2 and exterior 1/|x - x0|: residual
    -> 0 with observed order reported (threshold: order >= 1.5)."""
    nts, = args
    dev = _cpu_only()
    from hydro.rankine import panels as P
    from hydro.rankine.influence import influence
    funcs = {
        "interior 1": (lambda X: np.ones(len(X)), lambda X: np.zeros_like(X), +1),
        "interior x": (lambda X: X[:, 0], lambda X: np.tile([1.0, 0, 0], (len(X), 1)), +1),
        "interior x2-y2": (lambda X: X[:, 0] ** 2 - X[:, 1] ** 2,
                           lambda X: np.stack([2 * X[:, 0], -2 * X[:, 1], 0 * X[:, 0]], 1), +1),
    }
    x0 = np.array([0.3, 0.2, -0.1])

    def ext_phi(X):
        return 1.0 / np.linalg.norm(X - x0, axis=1)

    def ext_grad(X):
        d = X - x0
        return -d / np.linalg.norm(d, axis=1)[:, None] ** 3
    funcs["exterior 1/|x-x0|"] = (ext_phi, ext_grad, -1)
    res = {k: [] for k in funcs}
    hs = []
    for nt in nts:
        pan = P.sphere(1.0, nt, 2 * nt)
        S, D = influence(pan.c, pan, self_idx=np.arange(len(pan)), device=dev)
        hs.append(np.pi / nt)
        for k, (f, g, side) in funcs.items():
            phi = f(pan.c)
            phin = np.einsum("ij,ij->i", g(pan.c), pan.n)
            if side > 0:
                r = 2 * np.pi * phi + D @ phi - S @ phin
            else:
                r = 2 * np.pi * phi - D @ phi + S @ phin
            res[k].append(float(np.abs(r).max() / (2 * np.pi * np.abs(phi).max())))
    rows = []
    for k, e in res.items():
        if k == "interior 1":
            rows.append(_row("C2", f"Green identity {k}, residual (nt={nts})", e, 0.0,
                             "exact for a closed polyhedron", 1e-10, max(e) <= 1e-10))
        else:
            o = _order(hs, e)
            rows.append(_row("C2", f"Green identity {k}, residual (nt={nts}); order",
                             dict(residual=e, order=o), 0.0, "exact identity", ">= 1.5",
                             o[-1] >= 1.5))
    return rows


def c3_sphere(args):
    """C3 (M5): translating sphere in unbounded fluid, octant + 3 images,
    Green formulation (E). A11 = rho V / 2 (exact). Threshold (design R1):
    < 0.5 % at the finest mesh, observed order >= 1.8. The source formulation
    is reported alongside (not a pass criterion)."""
    nts, device = args
    dev = _cpu_only() if device == "cpu" else device
    from hydro.rankine import panels as P
    from hydro.rankine.bie import ImageOperator, neumann_green, neumann_source, added_mass
    errs, errs_src, Ns = [], [], []
    par = P.MODE_PARITY[1]
    for nt in nts:
        oc = P.sphere(1.0, nt, nt, octant=True)
        want_src = len(oc) <= 1100
        op = ImageOperator(oc, axes=(0, 1, 2), want_grad=want_src, device=dev,
                           parities=[par])
        phi = neumann_green(op, oc.n[:, 0], par, device=dev)
        A = 8 * added_mass(phi, oc.n[:, 0], oc.area)
        errs.append(A / (2 * np.pi / 3) - 1)
        if want_src:
            _, phs = neumann_source(op, oc.n[:, 0], par, device=dev)
            errs_src.append(8 * added_mass(phs, oc.n[:, 0], oc.area) / (2 * np.pi / 3) - 1)
        Ns.append(8 * len(oc))
    o = _order([1.0 / n for n in nts], errs)
    return [_row("C3", f"sphere A11/(rho V/2) - 1, N_full={Ns}", errs, 0.0,
                 "exact, rho V/2 (Lamb art. 92)", "<0.5% finest, order>=1.8",
                 abs(errs[-1]) < 5e-3 and o[-1] >= 1.8, note=f"order {o}"),
            _row("C3b", f"sphere, source formulation, A11 rel. error (N={Ns[:len(errs_src)]})",
                 errs_src, 0.0, "exact", "info", True,
                 note=f"order {_order([1.0 / n for n in nts[:len(errs_src)]], errs_src)}")]


def c4_spheroid(args):
    """C4: prolate spheroid a/b = 5, octant + images, surge A11, heave A33,
    pitch A55 vs Lamb closed forms. Threshold (design R1): < 1 % at the
    finest mesh; observed order reported."""
    series, device = args
    dev = _cpu_only() if device == "cpu" else device
    from hydro.rankine import panels as P
    from hydro.rankine.bie import ImageOperator, neumann_green, added_mass
    from hydro.rankine.analytic import spheroid_lamb
    a, b = 5.0, 1.0
    ref = spheroid_lamb(a, b)
    refs = {1: ref["k1"] * ref["V"], 3: ref["k2"] * ref["V"], 5: ref["kp"] * ref["I_over_rho"]}
    out = {1: [], 3: [], 5: []}
    hs = []
    for nt, nphi in series:
        oc = P.ellipsoid(a, b, b, nt, nphi, (0.0, np.pi / 2), (0.0, np.pi / 2),
                         spacing="uniform")
        nm = P.mode_normals(oc)
        pars = [P.MODE_PARITY[m] for m in (1, 3, 5)]
        op = ImageOperator(oc, axes=(0, 1, 2), device=dev, parities=pars)
        for m in (1, 3, 5):
            phi = neumann_green(op, nm[:, m - 1], P.MODE_PARITY[m], device=dev)
            out[m].append(8 * added_mass(phi, nm[:, m - 1], oc.area) / refs[m] - 1)
        hs.append(1.0 / nt)
    rows = []
    names = {1: "surge A11", 3: "heave A33", 5: "pitch A55"}
    for m in (1, 3, 5):
        o = _order(hs, out[m])
        rows.append(_row("C4", f"spheroid a/b=5 {names[m]} rel. error, meshes {series}",
                         out[m], refs[m], "Lamb closed form (analytic.spheroid_lamb)",
                         "<1% finest", abs(out[m][-1]) < 0.01, note=f"order {o}"))
    return rows


def c5c6_hemisphere_limits(args):
    """C5, C6: floating hemisphere heave in the two zero-gravity-type limits
    by z-images: phi = 0 on z = 0 (odd image, omega -> inf): A33 = rho V/2
    exactly (translating-sphere image); rigid lid (even image, omega -> 0):
    A33 = 0.8310 rho V (Hulme 1982). Threshold (design R1): < 1 % finest."""
    series, = args
    dev = _cpu_only()
    from hydro.rankine import panels as P
    from hydro.rankine.bie import ImageOperator, neumann_green, added_mass
    V = 2.0 / 3.0 * np.pi
    res = {-1: [], +1: []}
    for npol, naz in series:
        hq = P.hemisphere_quarter(1.0, npol, naz)
        pars = [(+1, +1, -1), (+1, +1, +1)]
        op = ImageOperator(hq, axes=(0, 1, 2), device=dev, parities=pars)
        for zp in (-1, +1):
            phi = neumann_green(op, hq.n[:, 2], (+1, +1, zp), device=dev)
            res[zp].append(4 * added_mass(phi, hq.n[:, 2], hq.area) / V)
    e5 = [x / 0.5 - 1 for x in res[-1]]
    e6 = [x / 0.8310 - 1 for x in res[+1]]
    h = [1.0 / s[0] for s in series]
    return [_row("C5", f"hemisphere A33/(rho V), phi=0 on z=0, meshes {series}", res[-1], 0.5,
                 "exact (translating sphere image)", "<1% finest", abs(e5[-1]) < 0.01,
                 note=f"rel. errors {e5}, order {_order(h, e5)}"),
            _row("C6", f"hemisphere A33/(rho V), rigid lid, meshes {series}", res[+1], 0.8310,
                 "Hulme 1982 (ka -> 0)", "<1% finest", abs(e6[-1]) < 0.01,
                 note=f"rel. errors {e6}")]


def c7_symmetry(args):
    """C7 (M3): port/starboard symmetry. (a) octant image solve vs the same
    sphere meshed in full without images: same A11 (threshold 1e-12 rel.).
    (b) a body without symmetry (offset, rotated ellipsoid) meshed explicitly
    and its mirror image in y: sway A22 and mirrored phi must agree (threshold
    1e-12 rel.; bitwise identity is not expected because the mirrored mesh
    lists corners in reversed order, which changes summation order)."""
    dev = _cpu_only()
    from hydro.rankine import panels as P
    from hydro.rankine.bie import ImageOperator, neumann_green, added_mass
    oc = P.sphere(1.0, 8, 8, octant=True)
    op = ImageOperator(oc, axes=(0, 1, 2), device=dev, parities=[P.MODE_PARITY[1]])
    A_oct = 8 * added_mass(neumann_green(op, oc.n[:, 0], P.MODE_PARITY[1], device=dev),
                           oc.n[:, 0], oc.area)
    full = P.sphere(1.0, 16, 32)
    A_full = added_mass(neumann_green(ImageOperator(full, device=dev), full.n[:, 0],
                                      device=dev), full.n[:, 0], full.area)
    da = abs(A_oct - A_full) / abs(A_full)
    # (b)
    body = P.ellipsoid(1.3, 0.7, 0.5, 14, 20)
    rng = np.random.default_rng(3)
    Rm = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(Rm) < 0:
        Rm[:, 0] *= -1
    body = P.Panels(body.raw @ Rm.T + np.array([0.2, 0.4, -0.1]))
    mirr = body.reflect(1)
    phi1 = neumann_green(ImageOperator(body, device=dev), body.n[:, 1], device=dev)
    phi2 = neumann_green(ImageOperator(mirr, device=dev), mirr.n[:, 1], device=dev)
    A1 = added_mass(phi1, body.n[:, 1], body.area)
    A2 = added_mass(phi2, mirr.n[:, 1], mirr.area)
    dphi = np.abs(phi1 + phi2).max() / np.abs(phi1).max()   # sway phi is odd in y
    return [_row("C7", "octant-image vs full-mesh sphere A11, rel. diff.", da, 0.0,
                 "exact by construction", 1e-12, da <= 1e-12),
            _row("C7", "mirrored asymmetric body: |A22 - A22_mirror|/A22 ; max|phi + phi_mirror|/max|phi|",
                 [abs(A1 - A2) / abs(A1), dphi], 0.0, "exact (symmetry)", 1e-12,
                 max(abs(A1 - A2) / abs(A1), dphi) <= 1e-12)]


def c8_similarity(args):
    """C8 (M3): geometric similarity. The spheroid mesh scaled by 1.5 must give
    the same A/(rho V) and A55/(rho V L^2) (threshold 1e-4, design R1)."""
    dev = _cpu_only()
    from hydro.rankine import panels as P
    from hydro.rankine.bie import ImageOperator, neumann_green, added_mass
    vals = []
    for s in (1.0, 1.5):
        oc = P.ellipsoid(5.0 * s, s, s, 16, 6, (0, np.pi / 2), (0, np.pi / 2))
        nm = P.mode_normals(oc)
        op = ImageOperator(oc, axes=(0, 1, 2), device=dev,
                           parities=[P.MODE_PARITY[1], P.MODE_PARITY[5]])
        a1 = added_mass(neumann_green(op, nm[:, 0], P.MODE_PARITY[1], device=dev), nm[:, 0], oc.area)
        a5 = added_mass(neumann_green(op, nm[:, 4], P.MODE_PARITY[5], device=dev), nm[:, 4], oc.area)
        vals.append((a1 / s ** 3, a5 / s ** 5))
    d = max(abs(vals[0][k] / vals[1][k] - 1) for k in range(2))
    return [_row("C8", "spheroid scaled x1.5: max rel. diff. of A11/L^3, A55/L^5", d, 0.0,
                 "exact (similarity)", 1e-4, d <= 1e-4)]


# =====================================================================
# Stage 2 -- free surface with gravity at zero speed (design step R2, U = 0)
# =====================================================================
G0 = 9.81
HEMI_V = 2.0 / 3.0 * np.pi           # a = 1
B_FLOOR = 0.1 * 0.3410               # 0.1 x peak of Hulme's B33/(rho V omega)


def _hemi(ka, mesh=(24, 16, 20), beach_kw=None, diffraction=False, device="cuda"):
    """Hemisphere heave at ka: A', B' (Hulme normalisation), and with
    diffraction=True also X3 direct, X3 Haskind, B from the energy relation."""
    from hydro.rankine.radiation import (hemisphere_case, radiation_coefficients,
                                         incident_even_x)
    sol, w, nu = hemisphere_case(ka, n_polar=mesh[0], n_az=mesh[1], n_wl=mesh[2],
                                 beach_kw=beach_kw, device=device)
    bp = sol.body
    NB = len(bp)
    cols = [bp.n[:, 2]]
    if diffraction:
        phI, dphI = incident_even_x(bp.c, bp.n, w, G0)
        cols.append(-dphI)
    phi = sol.solve(w, G0, nu, np.stack(cols, 1), (1, 1))
    A, B = radiation_coefficients(phi[:NB, 0], bp.n[:, 2], bp.area, w, 1.0, copies=4)
    out = dict(ka=ka, A=A / HEMI_V, B=B / (HEMI_V * w), N=len(sol.pan))
    if diffraction:
        Xd = 1j * w * 4 * np.sum((phI + phi[:NB, 1]) * bp.n[:, 2] * bp.area)
        Xh = 1j * w * 4 * np.sum((phI * bp.n[:, 2] - phi[:NB, 0] * dphI) * bp.area)
        k = ka
        out.update(Xd=Xd, Xh=Xh,
                   B_energy=w * k * abs(Xd) ** 2 / (2 * G0 ** 2) / (HEMI_V * w))
    return out


def _hulme(ka):
    from hydro.rankine.analytic import HULME_HEAVE as H
    i = np.where(np.isclose(H[:, 0], ka))[0]
    if len(i):
        return H[i[0], 1], H[i[0], 2]
    return np.interp(ka, H[:, 0], H[:, 1]), np.interp(ka, H[:, 0], H[:, 2])


def z1_hulme(args):
    """Z1: hemisphere heave A33/(rho V), B33/(rho V omega) vs Hulme 1982.
    Threshold (design R2): 3 %; A' relative to Hulme's value; B' relative to
    max(B'_Hulme, 0.1 x its peak 0.341) (registered before the run: B' -> 0
    at both ends, where a pure relative error is meaningless). With two
    meshes, the pass decision is on the finer one."""
    kas, meshes, device = args
    rows = []
    for mesh in meshes:
        errA, errB, vals = [], [], []
        for ka in kas:
            r = _hemi(ka, mesh, device=device)
            Ah, Bh = _hulme(ka)
            errA.append((r["A"] - Ah) / Ah)
            errB.append((r["B"] - Bh) / max(Bh, B_FLOOR))
            vals.append((ka, round(r["A"], 4), round(r["B"], 4), Ah, Bh, r["N"]))
        eA, eB = float(np.max(np.abs(errA))), float(np.max(np.abs(errB)))
        rows.append(_row("Z1", f"hemisphere vs Hulme, mesh (n_polar, n_az, panels/wavelength)={mesh}: "
                         "max |err A'|, max |err B'|", [eA, eB], "Hulme 1982 table",
                         "Hulme 1982 JFM 121 (analytic.HULME_HEAVE)", 0.03,
                         eA <= 0.03 and eB <= 0.03,
                         note="(ka, A', B', A'_H, B'_H, N): " + "; ".join(str(v) for v in vals)))
    if len(meshes) > 1:
        rows[0]["ok"] = True
        rows[0]["note"] = "[coarse member, info] " + rows[0]["note"]
    return rows


def z2_capytaine_hemisphere(args):
    """Z2 (M4, binding 5 %): our zero-speed result vs the zero-speed BEM
    hydro/bem.py (Capytaine 3.0.0, exact free-surface Green function) on the
    hemisphere; also cross-checks the Hulme transcription (information).
    Reference mesh: Capytaine converges slowly on this body (2026-10-06 run:
    at (30,30) = 450 panels B' at ka = 2 is 7.7 % below Hulme, at (90,90) =
    4050 panels 2.6 % below), so (60,60) / (90,90) are used; the first quick
    run with (30,30) failed this check for that reason (recorded in
    learn/meta/RANKINE_STAGE1_2026-10-06.md)."""
    kas, res = args
    import logging
    logging.disable(logging.WARNING)
    import capytaine as cpt
    from hydro import bem
    # lid just below the waterplane: removes Capytaine's irregular
    # frequencies (the first full run without it had a spike at ka = 2.5,
    # B' 29 % below Hulme, while its neighbours were within 1-5 %)
    mesh = cpt.mesh_sphere(radius=1.0, center=(0, 0, 0), resolution=res).immersed_part()
    body = cpt.FloatingBody(mesh=mesh, lid_mesh=mesh.generate_lid(z=-0.01), center_of_mass=(0, 0, 0))
    body.add_translation_dof(name="Heave")
    w = np.sqrt(np.asarray(kas) * G0)
    ds = bem.compute_database(body, w, rho=1.0, g=G0)
    A = ds["added_mass"].sel(influenced_dof="Heave", radiating_dof="Heave").values / HEMI_V
    B = ds["radiation_damping"].sel(influenced_dof="Heave", radiating_dof="Heave").values / (HEMI_V * w)
    return [dict(kind="capy", kas=list(kas), A=A.tolist(), B=B.tolist(), res=res)]


def z2_compare(capy, ours):
    eA = [(a - o["A"]) / a for a, o in zip(capy["A"], ours)]
    eB = [(b - o["B"]) / max(b, B_FLOOR) for b, o in zip(capy["B"], ours)]
    hA = [(a - _hulme(k)[0]) / _hulme(k)[0] for a, k in zip(capy["A"], capy["kas"])]
    hB = [(b - _hulme(k)[1]) / max(_hulme(k)[1], B_FLOOR) for b, k in zip(capy["B"], capy["kas"])]
    mA, mB = float(np.max(np.abs(eA))), float(np.max(np.abs(eB)))
    return [_row("Z2", f"hemisphere: ours (24,16,20) vs Capytaine (mesh {capy['res']}), "
                 f"max |dA'|/A', max |dB'|/max(B',floor), ka={capy['kas']}",
                 [mA, mB], "Capytaine 3.0.0 via hydro/bem.py", "zero-speed BEM (M4)", 0.05,
                 mA <= 0.05 and mB <= 0.05),
            _row("Z2b", "Capytaine vs Hulme table (transcription cross-check, info): errors A', B'",
                 dict(A=np.round(hA, 4).tolist(), B=np.round(hB, 4).tolist()), "Hulme 1982",
                 "information", "info", True)]


def z3_barge(args):
    """Z3 (M4, binding 5 %): rectangular barge L=4, B=2, T=0.5 m, heave and
    pitch (about the waterplane centre) A, B vs Capytaine at kL = 2, 4, 8.
    Non-axisymmetric body, x-odd pitch mode (image parity tested)."""
    kLs, nb, n_wl, device = args
    from hydro.rankine import panels as P
    from hydro.rankine.radiation import barge_case, radiation_coefficients
    L, B, T = 4.0, 2.0, 0.5
    out = []
    for kL in kLs:
        w = np.sqrt(G0 * kL / L)
        sol, nu = barge_case(w, L, B, T, nb=nb, n_wl=n_wl, device=device)
        bp = sol.body
        nm = P.mode_normals(bp)
        ph3 = sol.solve(w, G0, nu, nm[:, 2:3], (1, 1))[:len(bp), 0]
        ph5 = sol.solve(w, G0, nu, nm[:, 4:5], (-1, 1))[:len(bp), 0]
        a3, b3 = radiation_coefficients(ph3, nm[:, 2], bp.area, w, 1.0, 4)
        a5, b5 = radiation_coefficients(ph5, nm[:, 4], bp.area, w, 1.0, 4)
        out.append((a3, b3, a5, b5, len(sol.pan)))
    return [dict(kind="barge_ours", kLs=list(kLs), vals=out, nb=nb, n_wl=n_wl)]


def z3_capy(args):
    kLs, res = args
    import logging
    logging.disable(logging.WARNING)
    import capytaine as cpt
    from hydro import bem
    L, B, T = 4.0, 2.0, 0.5
    mesh = cpt.mesh_parallelepiped(size=(L, B, T), center=(0, 0, -T / 2), resolution=res,
                                   missing_sides={"top"}, name="barge")
    body = cpt.FloatingBody(mesh=mesh, center_of_mass=(0, 0, 0))
    body.add_translation_dof(name="Heave")
    body.add_rotation_dof(rotation_center=(0, 0, 0), direction=(0, 1, 0), name="Pitch")
    body = body.immersed_part()
    w = np.sqrt(G0 * np.asarray(kLs) / L)
    ds = bem.compute_database(body, w, rho=1.0, g=G0)

    def sel(v, d):
        return ds[v].sel(influenced_dof=d, radiating_dof=d).values
    vals = np.stack([sel("added_mass", "Heave"), sel("radiation_damping", "Heave"),
                     sel("added_mass", "Pitch"), sel("radiation_damping", "Pitch")], 1)
    return [dict(kind="barge_capy", vals=vals.tolist(), res=res)]


def z3_compare(ours, capy):
    o = np.array([v[:4] for v in ours["vals"]])
    c = np.array(capy["vals"])
    rel = np.abs(o - c) / np.abs(c)
    return [_row("Z3", f"barge heave/pitch vs Capytaine (ours nb={ours['nb']}, {ours['n_wl']}/wavelength, "
                 f"N={[v[4] for v in ours['vals']]}; Capytaine {capy['res']}), kL={ours['kLs']}: "
                 "max rel. diff of A33, B33, A55, B55",
                 np.round(rel.max(axis=0), 4).tolist(), "Capytaine 3.0.0 via hydro/bem.py",
                 "zero-speed BEM (M4)", 0.05, rel.max() <= 0.05,
                 note=f"ours {np.round(o, 1).tolist()} capy {np.round(c, 1).tolist()}")]


def z45_haskind_energy(args):
    """Z4 (M3, binding 3 %): Haskind excitation vs direct diffraction; Z5
    (energy identity, registered 3 %): B33 from the radiated energy
    B33 = omega k |X3|^2 / (2 rho g^2) (Newman 1962, deep water, axisymmetric
    body) vs B33 from the body pressure, relative to max(B', floor)."""
    kas, mesh, device = args
    eh, ee, notes = [], [], []
    for ka in kas:
        r = _hemi(ka, mesh, diffraction=True, device=device)
        eh.append(abs(r["Xh"] - r["Xd"]) / abs(r["Xd"]))
        ee.append(abs(r["B_energy"] - r["B"]) / max(r["B"], B_FLOOR))
        notes.append((ka, round(abs(r["Xd"]), 4), round(abs(r["Xh"]), 4), round(r["B"], 4),
                      round(r["B_energy"], 4)))
    return [_row("Z4", f"Haskind vs direct |X3| (hemisphere, mesh {mesh}, ka={kas}): max rel. diff",
                 float(max(eh)), 0.0, "Haskind relation (identity)", 0.03, max(eh) <= 0.03,
                 note="(ka,|Xdir|,|Xhask|,B',B'_energy): " + "; ".join(map(str, notes))),
            _row("Z5", "energy relation B33 vs omega k |X3|^2/(2 rho g^2), max rel. diff", float(max(ee)),
                 0.0, "Newman 1962 (identity)", 0.03, max(ee) <= 0.03)]


def z6_kramers_kronig(args):
    """Z6 (M3, design proposal 2 %): Kramers-Kronig. A(omega) predicted from
    B over a dense log grid ka in [ka_min, ka_max] and A_inf (phi = 0 image
    solve on the same body mesh):
        A(w) = A_inf + (2/pi) PV int_0^inf B(w')/(w'^2 - w^2) dw',
    vs the directly computed A at test frequencies. B outside the grid is
    taken as 0 (B' ~ ka below the grid, B' at ka_max reported)."""
    ka_grid, ka_test, mesh, device = args
    from hydro.rankine import panels as P
    from hydro.rankine.bie import ImageOperator, neumann_green, added_mass
    from scipy.interpolate import PchipInterpolator
    res = [_hemi(ka, mesh, device=device) for ka in ka_grid]
    w = np.sqrt(G0 * np.asarray(ka_grid))
    Bdim = np.array([r["B"] for r in res]) * HEMI_V * w            # rho = 1
    hq = P.hemisphere_quarter(1.0, mesh[0], mesh[1])
    op = ImageOperator(hq, axes=(0, 1, 2), device=device, parities=[(1, 1, -1)])
    A_inf = 4 * added_mass(neumann_green(op, hq.n[:, 2], (1, 1, -1), device=device),
                           hq.n[:, 2], hq.area)
    Bi = PchipInterpolator(np.log(w), Bdim)
    errs, notes = [], []
    wq = np.exp(np.linspace(np.log(w[0]), np.log(w[-1]), 400001))
    Bq = Bi(np.log(wq))
    for kt in ka_test:
        wt = np.sqrt(G0 * kt)
        Bt = float(Bi(np.log(wt)))
        den = wq ** 2 - wt ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = np.where(np.abs(den) > 1e-12, (Bq - Bt) / den, 0.0)
        I1 = np.trapezoid(integrand * wq, np.log(wq))
        I2 = Bt * (np.log(abs((w[-1] - wt) / (w[-1] + wt)))
                   - np.log(abs((w[0] - wt) / (w[0] + wt)))) / (2 * wt)
        A_kk = A_inf + 2 / np.pi * (I1 + I2)
        A_dir = _hemi(kt, mesh, device=device)["A"] * HEMI_V
        errs.append(abs(A_kk - A_dir) / A_dir)
        notes.append((kt, round(A_kk / HEMI_V, 4), round(A_dir / HEMI_V, 4)))
    return [_row("Z6", f"Kramers-Kronig A from B (hemisphere, {len(ka_grid)} ka in "
                 f"[{ka_grid[0]}, {ka_grid[-1]}], mesh {mesh}): max rel. diff",
                 float(max(errs)), 0.0, "Kramers-Kronig (Ogilvie 1964)", 0.02, max(errs) <= 0.02,
                 note=f"A_inf/(rho V)={A_inf / HEMI_V:.4f}; (ka, A'_KK, A'_direct): "
                 + "; ".join(map(str, notes)) + f"; B'(ka_max)={res[-1]['B']:.2e}, "
                 f"B'(ka_min)={res[0]['B']:.2e}")]


def z7_limits(args):
    """Z7 (M4): the free-surface solver at its two limits on a fixed
    free-surface mesh (radius R_out): K = omega^2/g -> inf (phi -> 0 on the
    free surface; high-frequency / zero-gravity limit) vs the phi = 0 image
    solve, and K = 0 (rigid lid, omega -> 0) vs the even-image solve.
    Threshold 1 % at the largest R_out (design R2 'A -> A_inf: 1 %')."""
    R_outs, device = args
    from hydro.rankine import panels as P
    from hydro.rankine.radiation import ZeroSpeedFS, radiation_coefficients
    from hydro.rankine.bie import ImageOperator, neumann_green, added_mass
    hq = P.hemisphere_quarter(1.0, 24, 16)
    op = ImageOperator(hq, axes=(0, 1, 2), device=device, parities=[(1, 1, -1), (1, 1, 1)])
    A_odd = 4 * added_mass(neumann_green(op, hq.n[:, 2], (1, 1, -1), device=device),
                           hq.n[:, 2], hq.area) / HEMI_V
    A_even = 4 * added_mass(neumann_green(op, hq.n[:, 2], (1, 1, 1), device=device),
                            hq.n[:, 2], hq.area) / HEMI_V
    hi, lo = [], []
    for R in R_outs:
        radii = P.graded_radii(1.0, 0.5 * np.pi / 24, 2.0, R, 1.08)
        sol = ZeroSpeedFS(hq, P.fs_polar_quarter(radii, 16), axes=(0, 1), parities=[(1, 1)],
                          device=device)
        for Kw, store in ((1e5, hi), (1e-9, lo)):
            w = np.sqrt(Kw * G0)
            ph = sol.solve(w, G0, 0.0, hq.n[:, 2:3], (1, 1))[:len(hq), 0]
            A, _ = radiation_coefficients(ph, hq.n[:, 2], hq.area, w, 1.0, 4)
            store.append(A / HEMI_V)
    e_hi = [abs(a / A_odd - 1) for a in hi]
    e_lo = [abs(a / A_even - 1) for a in lo]
    return [_row("Z7", f"K->inf (phi=0) limit of the FS solver vs odd-image solve, R_out={R_outs}",
                 e_hi, A_odd, "phi = 0 image solve (exact limit)", 0.01, e_hi[-1] <= 0.01,
                 note=f"A'={np.round(hi, 4).tolist()}"),
            _row("Z7", f"K->0 (rigid lid) limit of the FS solver vs even-image solve, R_out={R_outs}",
                 e_lo, A_even, "rigid-lid image solve (exact limit)", 0.01, e_lo[-1] <= 0.01,
                 note=f"A'={np.round(lo, 4).tolist()}")]


def z8_beach(args):
    """Z8: beach sensitivity, start x2 and length x2 (design S1 threshold
    1 %, applied to the radiation coefficients, relative to max(value, floor))."""
    kas, mesh, device = args
    worst, notes = 0.0, []
    for ka in kas:
        base = _hemi(ka, mesh, device=device)
        for bk in (dict(r_start_wl=2.0), dict(length_wl=6.0)):
            r = _hemi(ka, mesh, beach_kw=bk, device=device)
            dA = abs(r["A"] - base["A"]) / base["A"]
            dB = abs(r["B"] - base["B"]) / max(base["B"], B_FLOOR)
            worst = max(worst, dA, dB)
            notes.append((ka, bk, round(dA, 5), round(dB, 5)))
    return [_row("Z8", f"beach sensitivity (start x2, length x2), hemisphere ka={kas}: max rel. change",
                 worst, 0.0, "own", 0.01, worst <= 0.01, note=str(notes))]


STAGE2_KA_ALL = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6,
                 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
STAGE2 = {
    "quick": [
        ("Z1", z1_hulme, [((0.1, 0.5, 1.0, 2.0, 5.0), [(24, 16, 20)], "gpu")], "gpu"),
        ("Z2", z2_capytaine_hemisphere, [((0.1, 0.5, 1.0, 2.0, 5.0), (60, 60))], "cpu"),
        ("Z3", z3_barge, [((4.0,), (24, 12, 8), 20, "gpu")], "gpu"),
        ("Z3c", z3_capy, [((4.0,), (64, 32, 8))], "cpu"),
        ("Z4", z45_haskind_energy, [((0.25, 0.5, 1.0, 2.0), (24, 16, 20), "gpu")], "gpu"),
        ("Z6", z6_kramers_kronig, [(tuple(np.round(np.geomspace(0.01, 20, 36), 5)),
                                    (0.25, 0.5, 1.0, 2.0), (16, 16, 15), "gpu")], "gpu"),
        ("Z7", z7_limits, [((20.0, 40.0), "gpu")], "gpu"),
        ("Z8", z8_beach, [((0.5,), (16, 16, 15), "gpu")], "gpu"),
    ],
    "full": [
        ("Z1", z1_hulme, [(tuple(STAGE2_KA_ALL), [(24, 16, 20), (48, 16, 30)], "gpu")], "gpu"),
        ("Z2", z2_capytaine_hemisphere, [(tuple(STAGE2_KA_ALL), (90, 90))], "cpu"),
        ("Z3", z3_barge, [((2.0, 4.0, 8.0), (32, 16, 8), 25, "gpu")], "gpu"),
        ("Z3c", z3_capy, [((2.0, 4.0, 8.0), (64, 32, 8))], "cpu"),
        ("Z4", z45_haskind_energy, [((0.1, 0.25, 0.5, 1.0, 2.0, 3.0), (32, 24, 25), "gpu")], "gpu"),
        ("Z6", z6_kramers_kronig, [(tuple(np.round(np.geomspace(0.01, 20, 64), 5)),
                                    (0.1, 0.25, 0.5, 1.0, 2.0, 3.0), (24, 16, 20), "gpu")], "gpu"),
        ("Z7", z7_limits, [((20.0, 40.0, 80.0), "gpu")], "gpu"),
        ("Z8", z8_beach, [((0.2, 0.5, 2.0), (24, 16, 20), "gpu")], "gpu"),
    ],
}


def _gpu():
    import torch
    return torch.cuda.is_available()


def _postprocess(rows):
    """Pair raw stage-2 outputs (Capytaine and ours) into comparison rows."""
    out, capy_h, capy_b, ours_b = [], None, None, None
    for r in rows:
        k = r.get("kind")
        if k == "capy":
            capy_h = r
        elif k == "barge_capy":
            capy_b = r
        elif k == "barge_ours":
            ours_b = r
        else:
            out.append(r)
    if capy_h is not None:
        dev = "cuda" if _gpu() else "cpu"
        ours = [_hemi(ka, (24, 16, 20), device=dev) for ka in capy_h["kas"]]
        out += z2_compare(capy_h, ours)
    if capy_b is not None and ours_b is not None:
        out += z3_compare(ours_b, capy_b)
    return out


# =====================================================================
# Stage 3 -- steady forward speed, linear free surface (design step S1)
# Re-registered after the 2026-10-06 review (stage note section 1.1) and
# for stage B3 (stage note section 8.1): production operator = shifted
# Numerov rows with box correction (hydro/rankine/shifted.py, decision D1),
# solved by FFT-GMRES; the stage-1 operator (steady.py, upwind5) is used only
# where a row says so (S1b).
# =====================================================================
SPH_A, SPH_B = 5.0, 1.0              # prolate spheroid a/b = 5, L = 10 m
SPH_L = 2 * SPH_A
ENTRY_WL, BEACH_WL, BEACH_DECAY = 1.0, 2.0, 6.0
DOM_BASE = (1.0, 1.0, 1.0, 1.0, 1.0)  # multipliers: upstream, downstream, lateral, entry, beach


def _grid(f, lam, nl, L=SPH_L, a=SPH_A, mult=DOM_BASE, wake=None, lat=None):
    """Registered free-surface domain (section 1.1): upstream 1.5 max(L, 3f)
    ahead of the bow plus an entry zone of lambda0; downstream max(2.5
    lambda0, 3f) behind the stern (or `wake`) plus a beach of 2 lambda0;
    lateral max(0.8 L + 0.5 lambda0, 3f) (or `lat`). dx = lambda0/nl exactly,
    x = 0 a cell edge. mult scales (upstream, downstream, lateral, entry,
    beach) for the domain series. Returns fs, grid, solver kwargs."""
    from hydro.rankine.steady import fs_rect
    mu, md, ml, me, mb = mult
    up = mu * 1.5 * max(L, 3 * f) + me * ENTRY_WL * lam
    dn = md * (wake if wake is not None else max(2.5 * lam, 3 * f)) + mb * BEACH_WL * lam
    yl = ml * (lat if lat is not None else max(0.8 * L + 0.5 * lam, 3 * f))
    dx = lam / nl
    iu = int(np.ceil((a + up) / dx - 1e-9))
    idn = int(np.ceil((a + dn) / dx - 1e-9))
    jy = int(np.ceil(yl / dx - 1e-9))
    fs, grid = fs_rect(iu * dx, -idn * dx, jy * dx, iu + idn, jy)
    kw = dict(entry=me * ENTRY_WL * lam,
              beach=dict(length=mb * BEACH_WL * lam, decay=BEACH_DECAY))
    return fs, grid, kw


def _richardson(vals, r=1.5):
    """Observed order from successive differences of values on grids refined
    by r (Roache 1998): p = ln(d1/d2)/ln r, Richardson value v + d2/(r^p - 1),
    uncertainty 1.25 |v_R - v| (GCI). If the differences change sign or do
    not shrink, the series is not asymptotic: value = finest, uncertainty =
    max |difference|."""
    v = np.asarray(vals, float)
    if len(v) < 3:
        return dict(diffs=np.diff(v).tolist(), order=None, value=float(v[-1]),
                    unc=float(abs(v[-1] - v[0])) if len(v) > 1 else None, asymptotic=None)
    d1, d2 = v[-2] - v[-3], v[-1] - v[-2]
    if d1 * d2 > 0 and abs(d2) < abs(d1):
        p = float(np.log(d1 / d2) / np.log(r))
        ext = float(v[-1] + d2 / (r ** p - 1))
        return dict(diffs=[float(d1), float(d2)], order=p, value=ext,
                    unc=1.25 * abs(ext - v[-1]), asymptotic=True)
    return dict(diffs=[float(d1), float(d2)], order=None, value=float(v[-1]),
                unc=float(max(abs(d1), abs(d2))), asymptotic=False)


def _fmt_rich(rc):
    if rc["order"] is None:
        return (f"successive differences {np.round(rc['diffs'], 5).tolist()}: not asymptotic; "
                f"finest {rc['value']:.4f} +- {rc['unc']:.4f}")
    return (f"successive differences {np.round(rc['diffs'], 5).tolist()}, observed order "
            f"{rc['order']:.2f}, Richardson {rc['value']:.4f} +- {rc['unc']:.4f} (GCI)")


def _energy_wavelength(f, U, frac=0.9):
    """lambda_E = lambda0 cos^2(theta_E), theta_E the heading below which
    `frac` of Havelock's integral (line source of the spheroid) is generated:
    the shortest wavelength that carries the resistance."""
    from scipy.special import spherical_jn
    from hydro.rankine.havelock import spheroid_line_strength
    K, c = spheroid_line_strength(SPH_A, SPH_B, U)
    k0 = G0 / U ** 2

    def dens(th):
        kap = k0 / np.cos(th) ** 2
        return (spherical_jn(1, kap * c * np.cos(th)) * np.exp(-kap * f)) ** 2 / np.cos(th) ** 3
    th = np.linspace(0, np.pi / 2 - 1e-3, 4000)
    d = np.array([dens(t) for t in th])
    cum = np.cumsum(d) / d.sum()
    thE = th[np.searchsorted(cum, frac)]
    return 2 * np.pi * U ** 2 / G0 * np.cos(thE) ** 2, float(np.degrees(thE))


def _free(device):
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _s1_case(Fn, f, nl, scheme="numerov_shift", device="cuda", mult=DOM_BASE, method="fft"):
    """Body-free S1 case: the spheroid's line source (64 Gauss points) under
    the linear free surface; R by Lagally vs Havelock. scheme: production
    "numerov_shift" (shifted.py) or a stage-1 operator ("upwind5", "dawson4",
    steady.py)."""
    from hydro.rankine.steady import SteadyLinearFS, lagally_resistance, point_source_fields
    from hydro.rankine.shifted import SteadyShiftedFS, lagally_resistance as lagally_shift
    from hydro.rankine.havelock import spheroid_havelock, line_source_points
    U = Fn * np.sqrt(G0 * SPH_L)
    lam = 2 * np.pi * U ** 2 / G0
    Rh = spheroid_havelock(SPH_A, SPH_B, f, U, G0)
    Q, X = line_source_points(SPH_A, SPH_B, f, U, 64)
    fs, grid, kw = _grid(f, lam, nl, mult=mult)
    t = time.time()
    phs, gs, xx = point_source_fields(Q, X, fs.c)
    info = {}
    if scheme == "numerov_shift":
        sol = SteadyShiftedFS(None, fs, grid, device=device, **kw)
        r = sol.solve(U, G0, phi_ext=2 * phs, method=method)
        R = lagally_shift(fs, r["sigma_eff"], Q, X, device=device)
        info = r["info"]
        sol.release()
    else:
        sol = SteadyLinearFS(None, fs, grid, device=device, scheme=scheme, **kw)
        _, phinF = sol.solve(U, G0, None, sol.forcing(U, G0, 2 * xx, 2 * gs[:, 0]), consume=True)
        R = lagally_resistance(sol, phinF, Q, X)
        sol.release()
    _free(device)
    lamE, thE = _energy_wavelength(f, U)
    return dict(ratio=R / Rh, Rh=Rh, N=len(fs), per_lamE=lamE / (lam / nl), thE=thE,
                k0f=f * G0 / U ** 2, iters=info.get("iters"), relres=info.get("relres"),
                guard_l=info.get("guard_l", 0.0), info=info, seconds=time.time() - t)


def _res(infos):
    """Residual summary of a list of solve infos (B3 review: every stage-3
    row prints the true relative residual of every solve; a solve above the
    registered floor 1e-10 raises in SteadyShiftedFS.solve, so the row
    becomes an EXCEPTION/FAIL row)."""
    infos = [i for i in infos if i]
    if not infos:
        return "no shifted-operator solve"
    rr = [i["relres"] for i in infos]
    it = [i.get("iters", 0) for i in infos]
    gl = max(i.get("guard_l", 0.0) for i in infos)
    return (f"{len(infos)} solve(s): true residual max {max(rr):.1e} (floor 1e-10), "
            f"iterations {min(it)}-{max(it)}, interior guard blend max {gl:.3f}")


def s1_prescribed(args):
    """S1: prescribed singularity under the linear free surface (tests the
    free-surface operator alone): the spheroid's equivalent line source
    (a/b = 5, L = 10 m) at depth f, wave resistance by Lagally's theorem vs
    Havelock's integral (havelock.py; same equations, exact). Threshold
    (design S1): 5 % at the finest grid (binding for every case; the
    registered resolution criterion >= 20 panels per lambda_E is reported
    and a case below it is flagged, not exempted). Convergence: successive
    differences and Richardson value (section 1.1). Production operator
    (B3); series extended to 67.5 and 101.25 per lambda0 (section 8.1)."""
    cases, nls, device = args
    rows = []
    for Fn, f in cases:
        res = [_s1_case(Fn, f, nl, device=device) for nl in nls]
        vals = [r["ratio"] for r in res]
        e = vals[-1] - 1
        note = "; ".join(f"n/lambda0={nl:g}: R/R_H={r['ratio']:.5f} (N={r['N']}, {r['per_lamE']:.0f} per lambda_E, "
                         f"{r['iters']} GMRES its, residual {r['relres']:.1e}, guard l {r['guard_l']:.3f}, "
                         f"{r['seconds']:.1f} s)" for nl, r in zip(nls, res))
        if len(nls) > 2:
            note += "; " + _fmt_rich(_richardson(vals))
        if res[-1]["per_lamE"] < 20:
            note += "; FLAG: finest grid below the registered 20 panels per lambda_E"
        rows.append(_row("S1", f"[verification M2/M5] line source Fn={Fn} f={f} m (k0 f={res[0]['k0f']:.2f}, theta_E="
                         f"{res[0]['thE']:.0f} deg): R/R_Havelock - 1 at finest grid", e, 0.0,
                         "Havelock 1931 (havelock.spheroid_havelock)", 0.05, abs(e) <= 0.05, note=note))
    return rows


def s1_operator_compare(args):
    """S1b (information): the stage-1 operators (Dawson's four points,
    upwind5) and the production operator on the same S1 cases at one grid."""
    cases, nl, device = args
    rows = []
    for Fn, f in cases:
        v = [round(_s1_case(Fn, f, nl, scheme=s, device=device)["ratio"], 5)
             for s in ("dawson4", "upwind5", "numerov_shift")]
        rows.append(_row("S1b", f"operator comparison Fn={Fn} f={f} n/lambda0={nl}: R/R_H (dawson4, upwind5, "
                         "shifted Numerov)", v, 1.0, "Havelock 1931", "info", True,
                         note="dawson4/upwind5: stage-1 dense LU (steady.py); shifted Numerov: FFT-GMRES with "
                              "the residual floor of SteadyShiftedFS.solve"))
    return rows


def _spheroid_half(nt, nphi, f, gamma=None):
    """Half spheroid (y >= 0 is NOT assumed: phi in (-pi/2, pi/2) covers the
    y >= 0 half), centre depth f. gamma: t-grid s^gamma, a fore-aft
    asymmetric mesh (fine at the bow, coarse at the stern) for the
    d'Alembert checks."""
    from hydro.rankine import panels as P
    if gamma is None:
        pan = P.ellipsoid(SPH_A, SPH_B, SPH_B, nt, nphi, (0, np.pi), (-np.pi / 2, np.pi / 2))
    else:
        t = np.pi * np.linspace(0, 1, nt + 1) ** gamma
        p = np.linspace(-np.pi / 2, np.pi / 2, nphi + 1)
        T, Pp = np.meshgrid(t, p, indexing="ij")
        Xg = np.stack([SPH_A * np.cos(T), SPH_B * np.sin(T) * np.cos(Pp),
                       SPH_B * np.sin(T) * np.sin(Pp)], -1)
        pan = P.from_grid(Xg)
        if np.mean(np.einsum("ij,ij->i", pan.c, pan.n)) < 0:
            pan = P.from_grid(Xg, flip=True)
    v = pan.raw.copy()
    v[:, :, 2] -= f
    return P.Panels(v)


_S2_CACHE = {}


def _s2_case(Fn, f, nl=30, nb=(80, 32), device="cuda", L_scale=1.0, mult=DOM_BASE, gamma=None,
             method="fft"):
    """Rankine NK solve of the submerged spheroid (production operator).
    Results are cached per process (B3 review: S2K and the body-mesh rows
    reuse the S2 solves; the solves are deterministic, bitwise
    reproducible between processes)."""
    key = (Fn, f, nl, tuple(nb), L_scale, tuple(mult), gamma, method, device)
    if key not in _S2_CACHE:
        _S2_CACHE[key] = _s2_case_run(Fn, f, nl, nb, device, L_scale, mult, gamma, method)
    return _S2_CACHE[key]


def _s2_case_run(Fn, f, nl, nb, device, L_scale, mult, gamma, method):
    from hydro.rankine import panels as P
    from hydro.rankine.steady import double_body, near_field_resistance
    from hydro.rankine.shifted import SteadyShiftedFS
    from hydro.rankine.havelock import spheroid_havelock
    from hydro.rankine.farfield import farfield_resistance
    s = L_scale
    U = Fn * np.sqrt(G0 * SPH_L * s)
    lam = 2 * np.pi * U ** 2 / G0
    body = P.Panels(_spheroid_half(*nb, f, gamma).raw * s)
    fs, grid, kw = _grid(f * s, lam, nl, SPH_L * s, SPH_A * s, mult=mult)
    t = time.time()
    sol = SteadyShiftedFS(body, fs, grid, device=device, **kw)
    N = len(body) + len(fs)
    r = sol.solve(U, G0, U * body.n[:, 0], method=method)
    sol.release()
    del sol
    _free(device)
    phB = r["phi_B"]
    nf = near_field_resistance(body, phB, U, double_body(body, U, device))
    Rh = spheroid_havelock(SPH_A * s, SPH_B * s, f * s, U, G0)
    far = farfield_resistance(body, phB, U * body.n[:, 0], U, G0)
    return dict(R=nf["corrected"], R_raw=nf["raw"], dal=nf["dalembert"], far=far, Fz=nf["Fz"],
                Rh=Rh, N=N, U=U, iters=r["info"].get("iters"), relres=r["info"].get("relres"),
                guard_l=r["info"].get("guard_l", 0.0), info=r["info"], seconds=time.time() - t)


def _mesh_extrap(nts, vals):
    """Body-mesh extrapolation (B3 review) from meshes with nt = 80, 120,
    160 (h = 1/nt, ratios 1.5 and 1.33): order-2 value from the two finest
    (the BIE's measured order, C3/C4/S7a), and the observed order from all
    three when the differences have one sign and shrink."""
    from scipy.optimize import brentq
    h = 1.0 / np.asarray(nts, float)
    v = np.asarray(vals, float)
    e2 = float(v[-1] + (v[-1] - v[-2]) * h[-1] ** 2 / (h[-2] ** 2 - h[-1] ** 2))
    p, ep = None, None
    if len(v) >= 3:
        d1, d2 = v[-2] - v[-3], v[-1] - v[-2]
        if d1 * d2 > 0 and abs(d2) < abs(d1):
            def fp(q):
                return (h[-3] ** q - h[-2] ** q) / (h[-2] ** q - h[-1] ** q) - d1 / d2
            try:
                p = float(brentq(fp, 0.2, 8.0))
                ep = float(v[-1] + (v[-1] - v[-2]) * h[-1] ** p / (h[-2] ** p - h[-1] ** p))
            except ValueError:
                p = None
    return dict(order2=e2, order=p, observed=ep)


def s2_body(args):
    """S2: submerged prolate spheroid (body panels, Green formulation, full
    Bernoulli pressure, d'Alembert-corrected near field) vs Havelock 1931.
    Havelock neglects the body/free-surface interaction, so since the B3
    review this row is labelled INFORMATION (the size of the interaction
    missing from Havelock's approximation), neither validation nor
    verification; the registered threshold 5 % for f/b >= 4 (design S1) is
    kept and its FAIL stays in the record; f/b < 4 information. The
    verification of the same solutions is S2K (exact-Kelvin body BIE).
    S7 (added after review): near vs far-field (Kochin) resistance of the
    same solution, 3 % of R, every case. Production operator (B3). Body-mesh
    series (B3 review, full tier): binding cases also at 120x48 and 160x64
    with n/lambda0 = 30, order-2 extrapolation reported (row S2m)."""
    cases, nls_bind, nls_info, device = args[:4]
    nb_series = args[4] if len(args) > 4 else ()
    rows = []
    for Fn, f in cases:
        binding = f / SPH_B >= 4
        nls = nls_bind if binding else nls_info
        res = [_s2_case(Fn, f, nl, device=device) for nl in nls]
        r = res[-1]
        ratio, far = r["R"] / r["Rh"], r["far"] / r["Rh"]
        note = "; ".join(f"n/lambda0={nl:g}: near {x['R'] / x['Rh']:.5f} (raw {x['R_raw'] / x['Rh']:.5f}), "
                         f"far {x['far'] / x['Rh']:.5f}, N={x['N']}, {x['iters']} its, residual {x['relres']:.1e}, "
                         f"guard l {x['guard_l']:.3f}, {x['seconds']:.0f} s"
                         for nl, x in zip(nls, res))
        if len(nls) > 2:
            note += "; near: " + _fmt_rich(_richardson([x["R"] / x["Rh"] for x in res]))
            note += "; far: " + _fmt_rich(_richardson([x["far"] / x["Rh"] for x in res]))
        mrow = None
        if binding and nb_series:
            meshes = [(80, 32)] + list(nb_series)
            rm = [_s2_case(Fn, f, 30, nb=nb, device=device) for nb in meshes]
            vn = [x["R"] / x["Rh"] for x in rm]
            vf = [x["far"] / x["Rh"] for x in rm]
            en, ef = _mesh_extrap([m[0] for m in meshes], vn), _mesh_extrap([m[0] for m in meshes], vf)
            # combined estimate: finest free-surface value (80x32) + body-mesh correction at n = 30
            cn = ratio + en["order2"] - vn[0]
            cf = far + ef["order2"] - vf[0]
            note += (f"; body-mesh-extrapolated (order 2, row S2m): near {cn:.5f}, far {cf:.5f} "
                     f"(80x32 at the finest grid: near {ratio:.5f}, far {far:.5f})")
            mrow = _row("S2m", f"[added after the B3 review] Fn={Fn} f/b={f}: body-mesh series at n/lambda0=30, "
                        f"meshes {meshes}: R/R_H near; far; order-2 extrapolation (near, far)",
                        [vn, vf, [en["order2"], ef["order2"]]], None, "own series (body mesh)", "info", True,
                        note=f"observed order near {en['order']}, far {ef['order']} (values {en['observed']}, "
                             f"{ef['observed']}); combined estimate (finest free-surface grid + body-mesh "
                             f"correction) near {cn:.5f}, far {cf:.5f}; " + _res([x["info"] for x in rm]))
        rows.append(_row("S2", f"[information: size of the interaction missing from Havelock's approximation; "
                         f"registered 5 % kept] spheroid a/b=5 Fn={Fn} f/b={f}: R_near/R_Havelock at finest grid "
                         f"(body 80x32)", ratio, 1.0,
                         "Havelock 1931 (approximation without the body/free-surface interaction)",
                         0.05 if binding else "info (interaction)",
                         (abs(ratio - 1) <= 0.05) if binding else True, note=note))
        if mrow is not None:
            rows.append(mrow)
        d = ratio / far - 1
        rows.append(_row("S7", f"[added after review] Fn={Fn} f/b={f}: R_near/R_far - 1 (same solution, finest grid)",
                         d, 0.0, "far field: Kochin function of the body data (farfield.py)", 0.03,
                         abs(d) <= 0.03, note=f"R_far/R_H = {far:.5f}"))
    return rows


def _kelvin_green_rows(device):
    """S2K-G [added after the B3 review]: verification (M2) of the exact
    Kelvin Green function of hydro/rankine/kelvin.py (values of
    exploration run B3R-E1 known before registration)."""
    from scipy.integrate import quad
    from hydro.rankine import kelvin as K
    from hydro.rankine.havelock import spheroid_havelock, line_source_points
    from hydro.rankine.steady import double_body, near_field_resistance
    from hydro.rankine.farfield import farfield_resistance
    rows = []
    e1, npts, zw = K.e1_check(n=3000, seed=0)
    rows.append(_row("S2K-G", "[added after the B3 review] e^z (E1(z) + i pi) of kelvin.A_upper vs mpmath (40 digits), "
                     f"{npts} points, Re z in [-200, -0.5], Im z in [0, 400] and on the cut: max rel. error", e1, 0.0,
                     "mpmath.e1", 1e-12, e1 <= 1e-12, note=f"worst at z = {zw:.4g}"))
    e_src, e_line = [], []
    for Fn, f in ((0.4, 4.0), (0.5, 4.0), (0.4, 2.5)):
        U = Fn * np.sqrt(G0 * SPH_L)
        k0 = G0 / U ** 2
        gx = K.wave_part(np.array([0.0]), np.array([0.0]), np.array([-2 * f]), k0, 128, ("X",))["X"][0]
        ana = k0 ** 2 / np.pi * quad(lambda t: np.cos(t) ** -3 * np.exp(-2 * k0 * f / np.cos(t) ** 2),
                                     0, np.pi / 2, epsabs=0, epsrel=1e-13, limit=200)[0]
        e_src.append(float(gx / ana - 1))
        Q, Xs = line_source_points(SPH_A, SPH_B, f, U, 64)
        DX = Xs[:, 0][:, None] - Xs[:, 0][None, :]
        GX = K.wave_part(DX, np.zeros_like(DX), np.full(DX.shape, -2 * f), k0, 128, ("X",))["X"]
        e_line.append(float(Q @ GX @ Q / spheroid_havelock(SPH_A, SPH_B, f, U, G0) - 1))
    rows.append(_row("S2K-G", "[added after the B3 review] Lagally: single source R/(rho Q^2) = G_W,x(0, 0, -2f) vs "
                     "(k0^2/pi) int sec^3 e^(-2 k0 f sec^2); spheroid line source (64 points) vs "
                     "havelock.spheroid_havelock; (Fn, f) = (0.4, 4), (0.5, 4), (0.4, 2.5)",
                     [e_src, e_line], 0.0, "Havelock 1931 (closed integrals)", 1e-10,
                     max(np.abs(e_src).max(), np.abs(e_line).max()) <= 1e-10))
    U = 0.4 * np.sqrt(G0 * SPH_L)
    k0, f = G0 / U ** 2, 4.0
    Pt = np.array([(3.0, 0.5), (-5.0, 1.0), (-12.0, 3.0), (0.0, 0.0), (8.0, -2.0), (-30.0, 7.0), (25.0, 1.0)])
    r = K.wave_part(Pt[:, 0], Pt[:, 1], np.full(len(Pt), -f), k0, 256, ("XX", "Z"))
    R0 = np.sqrt(Pt[:, 0] ** 2 + Pt[:, 1] ** 2 + f ** 2)
    rxx = 3 * Pt[:, 0] ** 2 / R0 ** 5 - 1 / R0 ** 3
    res = r["XX"] - rxx / (2 * np.pi) + k0 * r["Z"]
    rel = float(np.max(np.abs(res) / (np.abs(r["XX"]) + np.abs(rxx) / (2 * np.pi) + np.abs(k0 * r["Z"]))))
    lim = K.wave_part(np.array([2.0, 2.0]), np.array([0.7, 0.7]), np.array([-7.0, -7.0]), 1e-5, 512, ("G",))["G"]
    rows.append(_row("S2K-G", "[added after the B3 review] free-surface condition U^2 G_xx + g G_z = 0 on z = 0 "
                     "(analytic G_W,XX), 7 points upstream and downstream, Fn 0.4 f 4: max residual / size of terms",
                     rel, 0.0, "the linear condition (exact)", 1e-10, rel <= 1e-10,
                     note=f"k0 -> 0 limit (k0 = 1e-5): G_W = {lim[0]:.8f} vs 1/(2 pi r') = "
                          f"{1 / (2 * np.pi * np.sqrt(4 + 0.49 + 49)):.8f} (phi = 0 on z = 0)"))
    body = _spheroid_half(40, 16, f)
    pdb = double_body(body, U, device)
    vals = []
    for nth in (64, 128):
        phi = K.solve_body(body, U, G0, nth=nth, device=device)
        vals.append((near_field_resistance(body, phi, U, pdb)["corrected"],
                     farfield_resistance(body, phi, U * body.n[:, 0], U, G0)))
    d = max(abs(vals[0][0] / vals[1][0] - 1), abs(vals[0][1] / vals[1][1] - 1))
    rows.append(_row("S2K-G", "[added after the B3 review] theta quadrature: body BIE 40x16, Fn 0.4 f/b 4, R near and "
                     "far with 64 vs 128 Gauss nodes: max rel. change", d, 0.0, "own (convergence)", 1e-8, d <= 1e-8))
    return rows


def s2k_kelvin(args):
    """S2K [added after the B3 review; reviewer values known]: verification
    (M2) of the Rankine Neumann-Kelvin solution of the submerged spheroid
    against an independent body-only BIE with the exact Kelvin Green
    function (hydro/rankine/kelvin.py; no free-surface panels, entry zone,
    beach or domain) on the SAME body mesh: R_near and R_far of the Rankine
    solve at the given free-surface grids over those of the Kelvin solve.
    Tolerances (registered in stage note 8.8 after the reviewer's own
    comparison was known): near 0.5 %, far 0.1 %. Preceded by the S2K-G
    checks of the Kelvin function. Uses the cached S2 solves."""
    cases, device, gchecks = args
    from hydro.rankine import kelvin as K
    from hydro.rankine.steady import double_body, near_field_resistance
    from hydro.rankine.farfield import farfield_resistance
    rows = _kelvin_green_rows(device) if gchecks else []
    for Fn, f, nb, nls in cases:
        U = Fn * np.sqrt(G0 * SPH_L)
        body = _spheroid_half(*nb, f)
        t = time.time()
        phi = K.solve_body(body, U, G0, nth=64, device=device)
        tk = time.time() - t
        nk = near_field_resistance(body, phi, U, double_body(body, U, device))["corrected"]
        fk = farfield_resistance(body, phi, U * body.n[:, 0], U, G0)
        for nl in nls:
            r = _s2_case(Fn, f, nl, nb=nb, device=device)
            dn, df = r["R"] / nk - 1, r["far"] / fk - 1
            rows.append(_row("S2K", f"[added after the B3 review; reviewer values known] Fn={Fn} f/b={f} body "
                             f"{nb[0]}x{nb[1]}: Rankine NK (n/lambda0={nl:g}) / exact-Kelvin body BIE - 1: near; far",
                             [float(dn), float(df)], 0.0,
                             "body-only BIE with the exact Kelvin Green function, same body mesh (kelvin.py)",
                             "near 0.5 %, far 0.1 %", abs(dn) <= 0.005 and abs(df) <= 0.001,
                             note=f"Kelvin BIE R/R_H near {nk / r['Rh']:.5f}, far {fk / r['Rh']:.5f} (64 theta "
                                  f"nodes, {tk:.0f} s); Rankine near {r['R'] / r['Rh']:.5f}, far "
                                  f"{r['far'] / r['Rh']:.5f}; " + _res([r["info"]])))
    return rows


def s3_pattern(args):
    """S3: Kelvin pattern of a line source (Fn 0.5, f = 1.5 m, 4 lambda0 of
    undamped wake + beach), production operator. At n/lambda0 = 30 (as
    registered; the first member of nls, both tiers): transverse wavelength
    by the centre-row zero-crossing estimator vs the same estimator on the
    exact linear Kelvin elevation on the same window (+-2 %; reference
    re-registered after the B3 review, it was 2 pi U^2/g; the estimator
    alone moves by ~1 % with the window, so the binding wavelength checks
    are D1a and S3b); no upstream waves (no sign change from 0.25 L ahead, |zeta| <=
    1 % of max from 1 L ahead); Kelvin half-angle by the outer-envelope
    estimator (|zeta| > 10 % of the column max, fit y = A x + B x^(1/3)) vs
    the same estimator on the exact linear Kelvin elevation on the same grid
    (+-1 deg); S3b normalised L2 error of zeta vs the exact elevation in the
    wake from 1 lambda0 behind the stern to the beach (10 %; [B3] also at
    45 and 60); S3d [B3] amplitude trend (least-squares slope of the log of
    the solver/exact ratio per 0.75 lambda0 window, 1 % per lambda0); S3c
    spectral band errors (information). Exact elevation: zeta = (U/g)(2
    phi_s,x + phi_W,x) (farfield.kelvin_wave_x; the local non-wave remainder,
    O(1/(k0 R)^2), is omitted, hence the 1 lambda0 distance). Computed
    elevation: (U/g) d/dx of the total potential on z = 0 (fourth-order
    central differences of phi_F, shifted.phi_x4, plus the exact 2 phi_s,x)."""
    nls, device = args
    from hydro.rankine.steady import point_source_fields
    from hydro.rankine.shifted import SteadyShiftedFS, phi_x4
    from hydro.rankine.havelock import line_source_points
    from hydro.rankine.farfield import kelvin_wave_x
    a, L = SPH_A, SPH_L
    Fn, f = 0.5, 1.5
    U = Fn * np.sqrt(G0 * L)
    lam = 2 * np.pi * U ** 2 / G0
    k0 = 2 * np.pi / lam
    Q, X = line_source_points(SPH_A, SPH_B, f, U, 64)
    rows = []
    for nl in nls:
        fs, grid, kw = _grid(f, lam, nl, wake=max(4 * lam, 3 * f), lat=max(0.8 * L + 1.5 * lam, 3 * f))
        nx, ny = grid["nx"], grid["ny"]
        phs, gs, _ = point_source_fields(Q, X, fs.c)
        sol = SteadyShiftedFS(None, fs, grid, device=device, **kw)
        r = sol.solve(U, G0, phi_ext=2 * phs)
        sol.release()
        _free(device)
        zeta = (U / G0 * (phi_x4(r["phi_F"], nx, ny, grid["dx"]) + 2 * gs[:, 0])).reshape(ny, nx)
        xc = fs.c[:, 0].reshape(ny, nx)[0]
        yc = fs.c[:, 1].reshape(ny, nx)[:, 0]
        x_beach = xc.min() - 0.5 * grid["dx"] + BEACH_WL * lam
        cols = np.nonzero((xc < -a - 1.0 * lam) & (xc > x_beach))[0]
        Pw = np.stack(np.meshgrid(xc[cols], yc, indexing="xy"), -1).reshape(-1, 2)
        Pw = np.concatenate([Pw, np.zeros((len(Pw), 1))], 1)
        _, gsw, _ = point_source_fields(Q, X, Pw)
        zex = (U / G0 * (2 * gsw[:, 0] + kelvin_wave_x(Q, X, Pw, k0, device=device))).reshape(ny, len(cols))
        zsv = zeta[:, cols]
        l2 = float(np.linalg.norm(zsv - zex) / np.linalg.norm(zex))
        win, xm = [], []
        xw = xc[cols]
        for kq in range(8):
            mw = (xw <= -a - lam * (1 + 0.75 * kq)) & (xw > -a - lam * (1 + 0.75 * (kq + 1)))
            if mw.sum():
                win.append(float(np.linalg.norm(zsv[:, mw]) / np.linalg.norm(zex[:, mw])))
                xm.append(float(np.mean(-xw[mw])))
        trend = float(np.polyfit(np.array(xm) / lam, np.log(win), 1)[0]) if len(win) > 1 else 0.0
        tag = f"n/lambda0={nl}, N={len(fs)}, {r['info'].get('iters')} its"
        rnote = _res([r["info"]])
        if nl != nls[0]:
            rows.append(_row("S3b", f"[B3; members n = 45, 60 fixed after rankine_d1 and B3-E7 had run (values "
                             f"known)] wake zeta vs exact linear Kelvin elevation, normalised L2 error ({tag})",
                             l2, 0.0, "exact (farfield.kelvin_wave_x + double body)", 0.10, l2 <= 0.10,
                             note=f"amplitude ratio solver/exact per 0.75 lambda0 window: {np.round(win, 4).tolist()}; "
                                  f"trend {100 * (np.exp(trend) - 1):+.3f} % per lambda0; " + rnote))
            continue
        z0 = zeta[0]
        m = (xc < -a - 0.5 * lam) & (xc > x_beach + 0.5 * lam)
        zz = z0[m] - z0[m].mean()
        xx_ = xc[m]

        def wavelength(zrow):
            zr = zrow - zrow.mean()
            sg = np.sign(zr)
            idx = np.nonzero(sg[1:] != sg[:-1])[0]
            xz = xx_[idx] - zr[idx] * (xx_[idx + 1] - xx_[idx]) / (zr[idx + 1] - zr[idx])
            return 2 * np.mean(-np.diff(xz)), len(xz)
        lam_num, ncross = wavelength(z0[m])
        e_lam = lam_num / lam - 1
        # B3 review: the same estimator on the exact linear Kelvin elevation of
        # the same sources on the same centre-row window is the reference
        Pm = np.stack([xx_, np.full(len(xx_), yc[0]), np.zeros(len(xx_))], 1)
        _, gsm, _ = point_source_fields(Q, X, Pm)
        zex_row = U / G0 * (2 * gsm[:, 0] + kelvin_wave_x(Q, X, Pm, k0, device=device))
        lam_ex, ncross_ex = wavelength(zex_row)
        e_lam_ex = lam_ex / lam - 1
        d_lam = lam_num / lam_ex - 1
        zu = z0[xc > a + 0.25 * L]
        nchg = int(np.sum(np.sign(zu[1:]) != np.sign(zu[:-1])))
        up1 = float(np.abs(zeta[:, xc > a + L]).max() / np.abs(zeta).max())
        up05 = float(np.abs(zeta[:, xc > a + 0.5 * L]).max() / np.abs(zeta).max())

        def angle(Z, xcols):
            sel = np.nonzero(xcols > x_beach + 0.3 * lam)[0]
            xs, ys = [], []
            for i in sel:
                col = np.abs(Z[:, i])
                jj = np.nonzero(col > 0.1 * col.max())[0]
                xs.append(a - xcols[i])
                ys.append(yc[jj.max()])
            xs, ys = np.array(xs), np.array(ys)
            A = np.stack([xs, xs ** (1 / 3)], 1)
            return float(np.degrees(np.arctan(np.linalg.lstsq(A, ys, rcond=None)[0][0])))
        ang, ang_ex = angle(zsv, xc[cols]), angle(zex, xc[cols])

        def spec(Z):
            Zf = np.concatenate([Z[::-1], Z], 0)
            w = np.hanning(Zf.shape[1])[None]
            return np.abs(np.fft.fft2(Zf * w))
        Sn, Se = spec(zsv), spec(zex)
        dxg = grid["dx"]
        kx = 2 * np.pi * np.fft.fftfreq(Sn.shape[1], dxg)
        ky = 2 * np.pi * np.fft.fftfreq(Sn.shape[0], dxg)
        KK = np.sqrt(kx[None] ** 2 + ky[:, None] ** 2) / k0
        bands = {}
        for lo, hi in ((0.0, 1.25), (1.25, 1.75), (1.75, 2.5)):
            bm = (KK >= lo) & (KK < hi)
            bands[f"{lo}-{hi}"] = round(float(np.linalg.norm((Sn - Se)[bm]) / max(np.linalg.norm(Se[bm]), 1e-300)), 4)
        rows += [_row("S3", f"[reference re-registered after the B3 review] transverse wavelength, centre-row "
                      f"zero-crossing estimator: solver / same estimator on the exact elevation - 1 ({tag})",
                      d_lam, round(lam_ex, 5), "same estimator on the exact linear Kelvin elevation, same window "
                      "(farfield.kelvin_wave_x + double body)", 0.02, abs(d_lam) <= 0.02,
                      note=f"solver vs 2 pi U^2/g {100 * e_lam:+.3f} % ({ncross} crossings); exact field vs "
                           f"2 pi U^2/g {100 * e_lam_ex:+.3f} % ({ncross_ex} crossings): the estimator alone moves "
                           f"by ~1 % with the window, so this row cannot resolve errors below ~1 %; the binding "
                           f"wavelength checks are D1a and S3b; " + rnote),
                 _row("S3", "upstream: sign changes from 0.25 L ahead; max|zeta|/max from 1 L (and 0.5 L) ahead",
                      [nchg, up1, up05], 0, "radiation condition", "0 changes, <=1 % at 1 L",
                      nchg == 0 and up1 <= 0.01, note=rnote),
                 _row("S3", "[reference re-registered after review] Kelvin half-angle, outer-envelope estimator, deg",
                      ang, round(ang_ex, 3), "same estimator on the exact linear Kelvin elevation, same grid "
                      "(19.47 deg asymptotically)", "+-1 deg", abs(ang - ang_ex) <= 1.0, note=rnote),
                 _row("S3b", f"[added after review] wake zeta vs exact linear Kelvin elevation: normalised L2 error "
                      f"({tag})", l2, 0.0, "exact (farfield.kelvin_wave_x + double body)", 0.10, l2 <= 0.10,
                      note=f"{zsv.size} points, x from {xc[cols].max():.1f} to {xc[cols].min():.1f} m; "
                           f"amplitude ratio solver/exact per 0.75 lambda0 window downstream: "
                           f"{np.round(win, 4).tolist()} (stage-1 upwind5: 1.09, 1.14, 1.20, 1.26); " + rnote),
                 _row("S3d", "[B3] amplitude trend downstream: slope of log(window ratio) per lambda0",
                      float(np.exp(trend) - 1), 0.0, "exact linear Kelvin field (neutral propagation)",
                      "abs 1 % per lambda0", abs(np.exp(trend) - 1) <= 0.01,
                      note=f"{len(win)} windows of 0.75 lambda0 from 1 lambda0 behind the stern"),
                 _row("S3c", "[added after review] spectral band errors |abs FFT(zeta) - abs FFT(zeta_exact)|/|.| "
                      "by abs(k)/k0 band", bands, 0.0, "exact", "info", True,
                      note="stage-1 upwind5 at 30: 0.13, 0.16, 0.30")]
    return rows


def _split_forcing(sol, U, bx, bxx):
    """Exact base-flow forcing of the split phi = phi_db + psi, consistent
    with the blended rows (B3 review): row r, centre c = r - 1,
        k w_c [(1 - l_c) (phi''_r + 10 phi''_{r-1} + phi''_{r-2})/12 + l_c phi''_c - nu_c phi'_c],
    i.e. the Numerov average for the Numerov part of the row and phi''(c)
    for the part whose second difference is centred at c (before the
    review the blended mass operator put phi'' at r, an O(l dx phi''')
    inconsistency). Column c = 0 (whose second difference is replaced by
    the linear extrapolation, w ~ 1e-4 there) uses phi''(0)."""
    from hydro.rankine.shifted import mass_operator
    nx, ny = sol.grid["nx"], sol.grid["ny"]
    k = U * U / G0
    AN = mass_operator(nx, ny, np.zeros(nx), "numerov")
    num = (AN @ (U * bxx)).reshape(ny, nx)
    pxx = (U * bxx).reshape(ny, nx)
    zero = np.zeros((ny, 1))
    pxx_c = np.concatenate([zero, pxx[:, :-1]], 1)                       # phi''(c = r - 1) on row r
    lsh = np.concatenate([[1.0], np.asarray(sol.lam, float)[:nx - 1]])   # l_c per row r
    lsh[1] = 1.0
    fxx = (1 - lsh)[None, :] * num + lsh[None, :] * pxx_c
    wsh = np.concatenate([[0.0], sol.w[:nx][:-1]])
    nush = np.concatenate([[0.0], sol.nu[:nx][:-1]])
    fx_c = np.concatenate([zero, (U * bx).reshape(ny, nx)[:, :-1]], 1)
    return (k * wsh[None, :] * (fxx - nush[None, :] * fx_c)).reshape(-1)


def _s4b_production(body, phi1, Fdb, f, Fn, nl, device):
    """S4b member on a grid chosen for the member's own lambda0 (n/lambda0 =
    nl, registered domain), so the resolution guard is inactive and the
    production Numerov rows are tested (B3 review)."""
    from hydro.rankine.steady import body_force, base_flow_derivatives
    from hydro.rankine.shifted import SteadyShiftedFS
    U = Fn * np.sqrt(G0 * SPH_L)
    lam = 2 * np.pi * U ** 2 / G0
    fs, grid, kw = _grid(f, lam, nl)
    t = time.time()
    sol = SteadyShiftedFS(body, fs, grid, device=device, **kw)
    bx, bxx = base_flow_derivatives(body, phi1, body.n[:, 0], fs.c, device=device)
    r = sol.solve(U, G0, U * body.n[:, 0])
    F, _, _ = body_force(body, r["phi_B"], U)
    c_impl = (F[2] / U ** 2 / Fdb[2] - 1) / Fn ** 2
    rs = sol.solve(U, G0, np.zeros(len(body)), fs_forcing=_split_forcing(sol, U, bx, bxx))
    Fs, _, _ = body_force(body, U * phi1 + rs["phi_B"], U)
    c_split = (Fs[2] / U ** 2 / Fdb[2] - 1) / Fn ** 2
    gl = sol.guard_l
    sol.release()
    _free(device)
    d = c_impl / c_split - 1
    return _row("S4b", f"[added after the B3 review] production rows (guard inactive): c = (Fz/(U^2 Fz_db) - 1)/Fn^2 "
                f"at Fn={Fn} on its own grid n/lambda0={nl} (N={len(body) + len(fs)}): as implemented / split - 1",
                d, 0.0, "split formulation, base-flow forcing consistent with the rows", 0.05,
                abs(d) <= 0.05 and gl == 0.0,
                note=f"c as implemented {c_impl:.5f}, split {c_split:.5f}; interior guard blend {gl:.3f} (pass "
                     f"requires 0); {time.time() - t:.0f} s; " + _res([r["info"], rs["info"]]))


def s4_low_fn(args):
    """S4 (M4): Fn -> 0 tends to the double-body (rigid-lid) solution: vertical
    force on the submerged spheroid (f/b = 2.5) at Fn = 0.2, 0.1, 0.05, 0.025
    on ONE fixed free-surface grid (that of Fn 0.5, 20 per lambda0; the
    limit concerns the equations) vs the double-body image solve on the same
    body mesh. Registered rule: 1 % at Fn 0.05 (mis-specified: tests the
    physical O(Fn^2) correction; kept as registered). S4b (re-registered
    after review): the O(Fn^2) coefficient c = (Fz/(U^2 Fz_db) - 1)/Fn^2 as
    implemented vs the split phi = phi_db + psi with the exact double-body
    forcing, Fn 0.05: 5 %. On this fixed grid the low-Fn wavelengths are
    far below 3.4 panels per wavelength, so the resolution guard of
    shifted.blend_weights is ACTIVE (interior blend l 0.02 at Fn 0.2 to
    0.39 at Fn 0.025): S4 and this S4b row test the guarded operator, not
    the production Numerov rows (B3 review). The split forcing is the one
    consistent with the blended rows (_split_forcing, changed after the B3
    review). Added after the B3 review: S4b members on grids chosen for the
    member's own lambda0 (guard inactive, production rows), args[1]."""
    device = args[0]
    members = args[1] if len(args) > 1 else ()
    from hydro.rankine.steady import body_force, double_body, base_flow_derivatives
    from hydro.rankine.shifted import SteadyShiftedFS
    f = 2.5
    body = _spheroid_half(48, 20, f)
    phi1 = double_body(body, 1.0, device)                 # U = 1
    Fdb, _, _ = body_force(body, phi1, 1.0)
    lam = 2 * np.pi * 0.25 * SPH_L
    fs, grid, kw = _grid(f, lam, 20)
    nx = grid["nx"]
    sol = SteadyShiftedFS(body, fs, grid, device=device, allow_guard=True, **kw)
    bx, bxx = base_flow_derivatives(body, phi1, body.n[:, 0], fs.c, device=device)
    out, c_impl, c_split, lam_int, infos = [], {}, {}, {}, []
    fns = (0.2, 0.1, 0.05, 0.025)
    for Fn in fns:
        U = Fn * np.sqrt(G0 * SPH_L)
        r = sol.solve(U, G0, U * body.n[:, 0])
        infos.append(r["info"])
        F, _, _ = body_force(body, r["phi_B"], U)
        out.append(F[2] / U ** 2 / Fdb[2])
        c_impl[Fn] = (out[-1] - 1) / Fn ** 2
        lam_int[Fn] = float(sol.lam[nx // 2])
        if Fn in (0.1, 0.05):
            # split: psi with body condition 0 and the exact base-flow forcing
            rs = sol.solve(U, G0, np.zeros(len(body)), fs_forcing=_split_forcing(sol, U, bx, bxx))
            infos.append(rs["info"])
            Fs, _, _ = body_force(body, U * phi1 + rs["phi_B"], U)
            c_split[Fn] = (Fs[2] / U ** 2 / Fdb[2] - 1) / Fn ** 2
    sol.release()
    _free(device)
    e = [abs(x - 1) for x in out]
    d = c_impl[0.05] / c_split[0.05] - 1
    rows = [_row("S4", "Fz/U^2 relative to the double-body value at Fn = 0.2, 0.1, 0.05 (fixed grid; guarded operator); "
                 "pass rule as registered: 1 % at Fn 0.05",
                 out[:3], 1.0, "rigid-lid image solve (exact limit)", 0.01, e[2] <= 0.01,
                 note=f"double-body Fz/U^2 = {Fdb[2]:.4f}, N={len(body) + len(fs)}; deviation at Fn 0.025: "
                      f"{e[3]:.5f}; order {_order(fns, e)}; guard blend l in the interior "
                      f"{ {k_: round(v_, 3) for k_, v_ in lam_int.items()} }; " + _res(infos)),
            _row("S4b", "[re-registered after review] O(Fn^2) coefficient c = (Fz/(U^2 Fz_db) - 1)/Fn^2 at Fn 0.05 "
                 "(fixed grid, GUARDED operator; split forcing consistent with the blend since the B3 review): "
                 "as implemented / split - 1",
                 d, 0.0, "split formulation (steady.base_flow_derivatives)", 0.05, abs(d) <= 0.05,
                 note=f"c as implemented {c_impl[0.1]:.4f} (Fn 0.1), {c_impl[0.05]:.4f} (0.05), "
                      f"{c_impl[0.025]:.4f} (0.025); split {c_split[0.1]:.4f}, {c_split[0.05]:.4f}")]
    for Fn_m, nl_m in members:
        rows.append(_s4b_production(body, phi1, Fdb, f, Fn_m, nl_m, device))
    return rows


def s5_similarity(args):
    """S5 (M3): Froude similarity: the spheroid case scaled x1.5 at the same
    Fn gives the same R/(rho g L^3) (threshold 1e-4; with the deterministic
    neighbour tie-break expected at round-off). Production operator; the
    GMRES tolerance 1e-12 bounds the difference."""
    device, = args
    r1 = _s2_case(0.5, 4.0, nl=16, nb=(40, 16), device=device, L_scale=1.0)
    r2 = _s2_case(0.5, 4.0, nl=16, nb=(40, 16), device=device, L_scale=1.5)
    d = max(abs((r2[k] / 1.5 ** 3) / r1[k] - 1) for k in ("R", "R_raw", "far"))
    return [_row("S5", "spheroid x1.5 at the same Fn: max rel. diff of R/(rho g L^3) (near corrected, raw, far)",
                 d, 0.0, "exact (similarity)", 1e-4, d <= 1e-4, note=_res([r1["info"], r2["info"]]))]


def s6_domain(args):
    """S6 (re-registered after review): domain series per extent at the
    lowest Fn and deepest f (Fn 0.4, f = 4 m), body-free (S1) and body (S2,
    near and far): upstream, downstream, lateral extents x1.5, entry and
    beach lengths x2, each alone. Threshold: each change < 1 % (design S1
    rule). Production operator."""
    nl, device = args
    names = ["upstream x1.5", "downstream x1.5", "lateral x1.5", "entry x2", "beach x2"]
    mults = [(1.5, 1, 1, 1, 1), (1, 1.5, 1, 1, 1), (1, 1, 1.5, 1, 1), (1, 1, 1, 2, 1), (1, 1, 1, 1, 2)]
    r1 = _s1_case(0.4, 4.0, nl, device=device)
    b1 = r1["ratio"]
    b2 = _s2_case(0.4, 4.0, nl, device=device)
    infos = [r1["info"], b2["info"]]
    ch1, ch2, chf = [], [], []
    for mu in mults:
        q = _s1_case(0.4, 4.0, nl, device=device, mult=mu)
        ch1.append(q["ratio"] / b1 - 1)
        r = _s2_case(0.4, 4.0, nl, device=device, mult=mu)
        ch2.append(r["R"] / b2["R"] - 1)
        chf.append(r["far"] / b2["far"] - 1)
        infos += [q["info"], r["info"]]
    w = max(np.abs(ch1).max(), np.abs(ch2).max(), np.abs(chf).max())
    return [_row("S6", f"[re-registered after review] domain series per extent, Fn 0.4 f 4 m, n/lambda0={nl}: "
                 "max rel. change of R (S1 line source; S2 near; S2 far)", float(w), 0.0, "own", 0.01, w <= 0.01,
                 note="; ".join(f"{n}: S1 {a:+.5f}, S2 near {b:+.5f}, far {c:+.6f}"
                                for n, a, b, c in zip(names, ch1, ch2, chf))
                 + f"; base S1 {b1:.5f}, S2 near {b2['R'] / b2['Rh']:.5f}, far {b2['far'] / b2['Rh']:.5f}; "
                 + _res(infos))]


def s7_farfield(args):
    """S7a (M2, added after review): Kochin calibration -- body solve in an
    unbounded fluid (y image only): its body data reproduce the exterior
    field of Havelock's line source, so R_far = R_Havelock up to the body
    discretisation (1 % at 80x32; order reported). S7b: near (corrected) vs
    far on a fore-aft asymmetric body mesh (gamma 1.6), Fn 0.4, f/b = 4,
    n/lambda0 = 30: 3 % (production operator since B3). S8 (added after
    review): raw double-body x-force (exactly 0) on the asymmetric meshes,
    relative to R_Havelock(Fn 0.4, f/b = 4), the smallest binding R: 1 % at
    the production mesh 80x32."""
    meshes, device = args
    from hydro.rankine.bie import ImageOperator, neumann_green
    from hydro.rankine.steady import double_body, near_field_resistance
    from hydro.rankine.havelock import spheroid_havelock
    from hydro.rankine.farfield import farfield_resistance
    Fn, f = 0.4, 4.0
    U = Fn * np.sqrt(G0 * SPH_L)
    Rh = spheroid_havelock(SPH_A, SPH_B, f, U, G0)
    cal, dal = [], []
    for nb in meshes:
        body = _spheroid_half(*nb, f)
        op = ImageOperator(body, axes=(1,), device=device, parities=[(1,)])
        ph = neumann_green(op, U * body.n[:, 0], (1,), device=device)
        del op
        cal.append(farfield_resistance(body, ph, U * body.n[:, 0], U, G0) / Rh - 1)
        ba = _spheroid_half(*nb, f, gamma=1.6)
        dal.append(near_field_resistance(ba, double_body(ba, U, device), U)["raw"] / Rh)
    i80 = list(meshes).index((80, 32))
    r = _s2_case(Fn, f, 30, nb=(80, 32), device=device, gamma=1.6)
    d = r["R"] / r["far"] - 1
    return [_row("S7a", f"[added after review] Kochin calibration (unbounded body solve) R_far/R_Havelock - 1, "
                 f"meshes {list(meshes)}", [float(x) for x in cal], 0.0, "Havelock 1931 (same line source)",
                 "1 % at 80x32", abs(cal[i80]) <= 0.01, note=f"order {_order([1 / m[0] for m in meshes], cal)}"),
            _row("S7b", "[added after review] asymmetric body mesh 80x32 (gamma 1.6), Fn 0.4 f/b 4: R_near/R_far - 1",
                 d, 0.0, "far field (Kochin)", 0.03, abs(d) <= 0.03,
                 note=f"near corrected {r['R'] / r['Rh']:.5f}, raw {r['R_raw'] / r['Rh']:.5f}, far "
                      f"{r['far'] / r['Rh']:.5f} (R_Havelock units); " + _res([r["info"]])),
            _row("S8", f"[added after review] d'Alembert: raw double-body x-force / R_Havelock(Fn 0.4, f/b 4), "
                 f"asymmetric meshes {list(meshes)}", [float(x) for x in dal], 0.0, "exact 0",
                 "1 % at 80x32", abs(dal[i80]) <= 0.01,
                 note="the near field is d'Alembert-corrected (steady.near_field_resistance), so this error "
                      "does not enter the S2/S7 near values; it bounds the raw pressure integral")]


def s9_gravity_limit(args):
    """S9 [B3] (M4, g -> 0): steady spheroid (f/b = 2.5, mesh 48x20) at
    high Fn vs the phi = 0 image solve (odd rigid-lid image, the g -> 0 limit
    of U^2 phi_xx + g phi_z = 0 with no waves upstream). Grid from the body
    scale (dx = L/20; upstream 1.5 max(L, 3f) + entry L, downstream 2 L +
    beach L, lateral 1.5 L), registered in section 8.1. Threshold 5 % at
    Fn 8 (acceptance M4); the approach order in 1/Fn is reported."""
    fns, device = args
    from hydro.rankine.steady import fs_rect, body_force
    from hydro.rankine.shifted import SteadyShiftedFS
    from hydro.rankine.bie import ImageOperator, neumann_green
    f = 2.5
    L = SPH_L
    body = _spheroid_half(48, 20, f)
    op = ImageOperator(body, axes=(1, 2), device=device, parities=[(1, -1)])
    ph0 = neumann_green(op, body.n[:, 0], (1, -1), device=device)
    del op
    F0, _, _ = body_force(body, ph0, 1.0)
    dx = L / 20
    up, dn, yl = 1.5 * max(L, 3 * f) + L, 2 * L + L, 1.5 * L
    iu = int(np.ceil((SPH_A + up) / dx - 1e-9))
    idn = int(np.ceil((SPH_A + dn) / dx - 1e-9))
    jy = int(np.ceil(yl / dx - 1e-9))
    fs, grid = fs_rect(iu * dx, -idn * dx, jy * dx, iu + idn, jy)
    sol = SteadyShiftedFS(body, fs, grid, device=device, entry=L, beach=dict(length=L, decay=BEACH_DECAY))
    ratios, fxs, infos = [], [], []
    for Fn in fns:
        U = Fn * np.sqrt(G0 * L)
        r = sol.solve(U, G0, U * body.n[:, 0])
        infos.append(r["info"])
        F, _, _ = body_force(body, r["phi_B"], U)
        ratios.append(float(F[2] / U ** 2 / F0[2]))
        fxs.append(float(F[0] / U ** 2))
    sol.release()
    _free(device)
    e = [x - 1 for x in ratios]
    i8 = list(fns).index(8.0)
    rows = [_row("S9", f"[B3; evaluation point Fn 8 fixed after exploration B3-E10 (values known)] g -> 0: Fz/U^2 "
                 f"relative to the phi = 0 image solve at Fn {list(fns)}; binding at Fn 8",
                 ratios, 1.0, "odd rigid-lid image solve (exact g -> 0 limit of the linear problem)", 0.05,
                 abs(e[i8]) <= 0.05,
                 note=f"phi = 0 limit Fz/U^2 = {F0[2]:.5f}; observed order in 1/Fn {_order([1 / x for x in fns], e)}; "
                      f"Fx/U^2 {np.round(fxs, 7).tolist()} (exactly 0 in the limit); N = {len(body) + len(fs)}; "
                      + _res(infos))]
    if all(x in fns for x in (4.0, 8.0, 16.0)):
        v4, v8, v16 = (ratios[list(fns).index(x)] for x in (4.0, 8.0, 16.0))
        d1, d2 = v8 - v4, v16 - v8
        if d1 * d2 > 0 and abs(d2) < abs(d1):
            pr = float(np.log2(d1 / d2))
            lim = float(v16 + d2 / (2 ** pr - 1))
        else:
            pr, lim = float("nan"), float(v16)
        rows.append(_row("S9r", "[added after the B3 review; reviewer values known] g -> 0, Fn-independent criterion: "
                         "Richardson extrapolation in 1/Fn over Fn 4, 8, 16 (ratio 2): limit of Fz/U^2 ratio; "
                         "observed order", [lim, pr], [1.0, 2.0], "odd rigid-lid image solve (exact limit); "
                         "the correction is O(1/Fn^2)", "limit within 0.5 % of 1; order 2 +- 0.3",
                         abs(lim - 1) <= 0.005 and abs(pr - 2) <= 0.3,
                         note=f"differences {d1:.5f}, {d2:.5f}"))
    return rows


# ----------------------------------------------------------- B3 rows D1, F
def d1_symbol(args):
    """D1a-e [B3]: coupled discrete dispersion, damping and row stability
    (studies/rankine_d1.py; stage note sections 8.1-8.2)."""
    ns, full = args
    from studies import rankine_d1 as D
    rows = []
    t30 = D.dispersion_table(30, "numerov_shift", True)
    e30 = max(abs(e) for p, e, g in t30 if p <= 55.0)
    note = "; ".join(f"n={n}: " + ", ".join(f"{p:.0f} deg {100 * e:+.3f} %" for p, e, g in
                                           (t30 if n == 30 else D.dispersion_table(n, "numerov_shift", True)))
                     for n in ns)
    rows.append(_row("D1a", "[B3; 55 deg cut-off fixed after rankine_d1 had run (values known)] coupled wavenumber "
                     "error of the production operator, n/lambda0 = 30, psi <= 55 deg",
                     e30, 0.0, "exact dispersion k alpha^2 = |kappa| (Kelvin)", 0.005, e30 <= 0.005, note=note))
    gr = {n: D.max_growth_on_grid(n, "numerov_shift", True) for n in ns}
    damp = {n: min(g for p, e, g in D.dispersion_table(n, "numerov_shift", True)) for n in ns}
    ok = all(gr[n] <= 1e-10 for n in ns) and all(damp[n] >= -0.01 for n in ns)
    rows.append(_row("D1b", f"[B3] max amplitude change per lambda0 of the physical root over every lateral mode, n = {list(ns)}",
                     [gr[n] for n in ns], 0.0, "neutral (real symbol)", "growth <= 1e-10, damping <= 1 %", ok,
                     note=f"stage-1 upwind5 on the same grids: {[round(D.max_growth_on_grid(n, 'upwind5', False), 3) for n in ns]} "
                          "(growth per lambda0, short diverging waves)"))
    zm = {n: D.z_max(n, True) for n in (12, 20, 30, 45)}
    zg = zm[20]
    bl = D.blend_max_root(zg, nl=101 if not full else 201)
    be = D.beach_max_root(20, zg)
    # guard rows: l >= 0.4 for all z up to 1e8; l < 0.4 with z <= 12/(1.25 (2 - 5 l))
    a_n, b_n = D.row_weights("numerov_shift")
    gw = 0.0
    for lam in np.linspace(0.0, 0.4, 41):
        a = (1 - lam) * a_n + lam * np.array([1.0, 0.0, 0.0])
        zmax_l = 1e8 if lam >= 0.4 - 1e-12 else 12.0 / (1.25 * (2 - 5 * lam))
        for z in np.concatenate([np.linspace(1e-6, min(zmax_l, 50.0), 300), np.logspace(0, np.log10(zmax_l), 30)]):
            if z <= zmax_l:
                gw = max(gw, float(np.max(np.abs(D.row_roots(a, b_n, z)))))
    ok = all(v < 6 for v in zm.values()) and max(bl, be, gw) <= 1 + 1e-12
    rows.append(_row("D1c", "[B3] row stability: grid z_max(n) for n = 12, 20, 30, 45 (< 6); max |root| of blended entry, "
                     "beach and guard rows (<= 1 + 1e-12)", [round(v, 4) for v in zm.values()] + [bl, be, gw], 6.0,
                     "Numerov interval of periodicity (0, 6); unconditional for l >= 0.4 (proof in shifted.py)",
                     "z_max < 6; roots <= 1 + 1e-12", ok))
    z30 = D.z_max(30, False)
    old = {s: (D.stability_interval(s), round(D.max_root_at(s, z30), 4)) for s in ("dawson4", "upwind5", "upwind6", "upwind7")}
    rows.append(_row("D1d", "[B3] stage-1 operators: stability interval in z and max |root| at the grid's z_max(30) "
                     "(upwind rows, no box)", {k_: [float(v[0]), v[1]] for k_, v in old.items()}, None,
                     "Dahlquist 1978: unconditional stability only for order <= 2", "info", True,
                     note="n=30 growth per lambda0 along the dispersion curve (0..65 deg): " + "; ".join(
                         f"{s}: " + ", ".join(f"{100 * g:+.2f}" for p, e, g in D.dispersion_table(30, s, False))
                         for s in ("upwind5", "upwind6", "upwind7"))))
    if full:
        rows.append(_row("D1e", "[B3] raised panels + shift (alternative): wavenumber error at n 30 for h/dx = 1, 1.5, 2 "
                         "(psi 0, 35, 55 deg)", {h: [round(D.raised_error(30, h, p), 6) for p in (0, 35.26, 55)]
                                                 for h in (1.0, 1.5, 2.0)}, None, "exact dispersion", "info", True))
    return rows


def f_fft(args):
    """F1-F4 [B3]: FFT lattice and FFT-GMRES vs dense (same discrete
    system), stage note section 8.1."""
    device, = args
    import torch
    from hydro.rankine.bie import ImageOperator
    from hydro.rankine.shifted import SteadyShiftedFS, FSLattice, lagally_resistance
    from hydro.rankine.steady import point_source_fields, double_body, near_field_resistance
    from hydro.rankine.havelock import line_source_points, spheroid_havelock
    rows = []
    # F1: kernel vs image operator, small grid
    U = 0.5 * np.sqrt(G0 * SPH_L)
    fs, grid, kw = _grid(2.5, 2 * np.pi * U ** 2 / G0, 12)
    lat = FSLattice(grid, device)
    Sd = lat.dense()
    op = ImageOperator(fs, axes=(1, 2), device=device, parities=[(1, 1)], on_device=True)
    S = op.S((1, 1))
    e1 = float((Sd - S).abs().max() / S.abs().max())
    del op, S, Sd, lat
    rows.append(_row("F1", f"[B3] lattice kernel S_FF vs image operator (bie.py), N={len(fs)}: max abs diff / max abs S",
                     e1, 0.0, "bie.ImageOperator (all four images)", 1e-12, e1 <= 1e-12))
    # F2, F3 body-free: S1 Fn 0.4 f 4, n 30
    Fn, f = 0.4, 4.0
    U = Fn * np.sqrt(G0 * SPH_L)
    lam = 2 * np.pi * U ** 2 / G0
    fs, grid, kw = _grid(f, lam, 30)
    lat = FSLattice(grid, device)
    gen = torch.Generator(device="cpu").manual_seed(1)
    v = torch.rand(len(fs), generator=gen, dtype=torch.float64).to(device)
    Sd = lat.dense()
    ref = Sd @ v
    e2 = float(torch.linalg.norm(lat.apply(v) - ref) / torch.linalg.norm(ref))
    del Sd, lat
    _free(device)
    rows.append(_row("F2", f"[B3] FFT product vs dense product (same kernel), N={len(fs)}", e2, 0.0,
                     "dense matrix-vector product", 1e-12, e2 <= 1e-12))
    Q, X = line_source_points(SPH_A, SPH_B, f, U, 64)
    phs, _, _ = point_source_fields(Q, X, fs.c)
    sol = SteadyShiftedFS(None, fs, grid, device=device, **kw)
    t = time.time()
    rd = sol.solve(U, G0, phi_ext=2 * phs, method="dense")
    td = time.time() - t
    t = time.time()
    rf = sol.solve(U, G0, phi_ext=2 * phs, method="fft")
    tf = time.time() - t
    sol.release()
    ds = float(np.linalg.norm(rf["sigma"] - rd["sigma"]) / np.linalg.norm(rd["sigma"]))
    Rd = lagally_resistance(fs, rd["sigma_eff"], Q, X, device=device)
    Rf = lagally_resistance(fs, rf["sigma_eff"], Q, X, device=device)
    dR = abs(Rf / Rd - 1)
    rows.append(_row("F3", f"[B3] FFT-GMRES vs dense LU, body-free S1 Fn 0.4 f 4 n 30 (N={len(fs)}): rel. diff of sigma; of R",
                     [ds, dR], 0.0, "dense LU of the same matrix", 1e-8, max(ds, dR) <= 1e-8,
                     note=f"GMRES {rf['info']['iters']} its, true residual {rf['info']['relres']:.1e}, {tf:.1f} s; "
                          f"dense {td:.1f} s; R/R_H {Rd / spheroid_havelock(SPH_A, SPH_B, f, U, G0):.6f}"))
    _free(device)
    # F3 body: S2 Fn 0.4 f/b 4 n 30
    body = _spheroid_half(80, 32, f)
    sol = SteadyShiftedFS(body, fs, grid, device=device, **kw)
    t = time.time()
    rd = sol.solve(U, G0, U * body.n[:, 0], method="dense")
    td = time.time() - t
    t = time.time()
    rf = sol.solve(U, G0, U * body.n[:, 0], method="fft")
    tf = time.time() - t
    sol.release()
    del sol
    _free(device)
    dp = float(np.linalg.norm(rf["phi_B"] - rd["phi_B"]) / np.linalg.norm(rd["phi_B"]))
    ds = float(np.linalg.norm(rf["sigma"] - rd["sigma"]) / np.linalg.norm(rd["sigma"]))
    pdb = double_body(body, U, device)
    Rd = near_field_resistance(body, rd["phi_B"], U, pdb)["corrected"]
    Rf = near_field_resistance(body, rf["phi_B"], U, pdb)["corrected"]
    dR = abs(Rf / Rd - 1)
    rows.append(_row("F3", f"[B3] FFT-GMRES vs dense LU, body S2 Fn 0.4 f/b 4 n 30 (N={len(fs) + len(body)}): "
                     "rel. diff of phi_B; sigma; R_near", [dp, ds, dR], 0.0, "dense LU of the same matrix", 1e-8,
                     max(dp, ds, dR) <= 1e-8,
                     note=f"GMRES {rf['info']['iters']} its, true residual {rf['info']['relres']:.1e}, {tf:.1f} s; "
                          f"dense {td:.1f} s"))
    return rows


STAGE3 = {
    "quick": [
        ("D1", d1_symbol, [((30,), False)], "gpu"),
        ("F", f_fft, [("gpu",)], "gpu"),
        ("S1", s1_prescribed, [(((0.4, 2.5), (0.5, 2.5), (0.5, 4.0)), (30,), "gpu")], "gpu"),
        ("S2", s2_body, [(((0.5, 4.0),), (30,), (30,), "gpu")], "gpu"),
        ("S2K", s2k_kelvin, [(((0.4, 4.0, (80, 32), (30,)), (0.5, 4.0, (80, 32), (30,))), "gpu", True)], "gpu"),
        ("S4", s4_low_fn, [("gpu", ((0.2, 20),))], "gpu"),
        ("S5", s5_similarity, [("gpu",)], "gpu"),
        ("S7", s7_farfield, [(((40, 16), (80, 32)), "gpu")], "gpu"),
        ("S3", s3_pattern, [((30,), "gpu")], "gpu"),
        ("S9", s9_gravity_limit, [((4.0, 8.0, 16.0), "gpu")], "gpu"),
    ],
    "full": [
        ("D1", d1_symbol, [((20, 30, 45), True)], "gpu"),
        ("F", f_fft, [("gpu",)], "gpu"),
        ("S1", s1_prescribed, [(((0.4, 2.5), (0.5, 2.5), (0.5, 4.0), (0.6, 4.0), (0.7, 4.0), (0.4, 4.0)),
                                (20, 30, 45, 67.5, 101.25), "gpu")], "gpu"),
        ("S1b", s1_operator_compare, [(((0.5, 2.5), (0.4, 4.0)), 30, "gpu")], "gpu"),
        ("S2", s2_body, [(((0.5, 2.5), (0.5, 4.0), (0.5, 6.0), (0.4, 2.5), (0.4, 4.0)), (20, 30, 45, 67.5, 101.25),
                          (30,), "gpu", ((120, 48), (160, 64)))], "gpu"),
        ("S2K", s2k_kelvin, [(((0.4, 4.0, (80, 32), (30, 101.25)), (0.5, 4.0, (80, 32), (30, 101.25)),
                               (0.4, 2.5, (80, 32), (30,)), (0.4, 4.0, (120, 48), (30,))), "gpu", True)], "gpu"),
        ("S3", s3_pattern, [((30, 45, 60), "gpu")], "gpu"),
        ("S4", s4_low_fn, [("gpu", ((0.2, 30), (0.1, 12)))], "gpu"),
        ("S5", s5_similarity, [("gpu",)], "gpu"),
        ("S6", s6_domain, [(30, "gpu")], "gpu"),
        ("S7", s7_farfield, [(((40, 16), (60, 24), (80, 32), (120, 48)), "gpu")], "gpu"),
        ("S9", s9_gravity_limit, [((2.0, 4.0, 8.0, 16.0), "gpu")], "gpu"),
    ],
}


# =====================================================================
STAGE1 = {
    "quick": [
        ("C1", c1_panel_integrals, [(0,), (1,)], "cpu"),
        ("C1b", c1b_rectangle, [()], "cpu"),
        ("C2", c2_green_identity, [((6, 12, 24),)], "cpu"),
        ("C3", c3_sphere, [((8, 16, 32), "cpu")], "cpu"),
        ("C4", c4_spheroid, [(((16, 4), (32, 8), (64, 16)), "cpu")], "cpu"),
        ("C5", c5c6_hemisphere_limits, [(((8, 8), (16, 16), (32, 32)),)], "cpu"),
        ("C7", c7_symmetry, [()], "cpu"),
        ("C8", c8_similarity, [()], "cpu"),
    ],
    "full": [
        ("C1", c1_panel_integrals, [(0,), (1,), (2,), (3,), (4,), (5,)], "cpu"),
        ("C1b", c1b_rectangle, [()], "cpu"),
        ("C2", c2_green_identity, [((6, 12, 24, 48),)], "cpu"),
        ("C3", c3_sphere, [((8, 16, 32, 64), "gpu")], "gpu"),
        ("C4", c4_spheroid, [(((16, 4), (32, 8), (64, 16), (128, 32)), "gpu")], "gpu"),
        ("C5", c5c6_hemisphere_limits, [(((8, 8), (16, 16), (32, 32), (48, 48)),)], "cpu"),
        ("C7", c7_symmetry, [()], "cpu"),
        ("C8", c8_similarity, [()], "cpu"),
    ],
}

REGISTRY = {"stage1": STAGE1, "stage2": STAGE2, "stage3": STAGE3}


# ------------------------------------------------- registration record (B3 review)
STAGE_NOTE = os.path.join(ROOT, "learn", "meta", "RANKINE_STAGE1_2026-10-06.md")
REG_LOG = os.path.join(OUT, "registration_log.json")
REG_SECTIONS = ("### 1.1 ", "### 8.1 ", "### 8.8 ")


def _note_sections():
    """sha256 of each registration section of the stage note (heading line
    to the next heading of level 2 or 3; line endings normalised)."""
    lines = open(STAGE_NOTE, encoding="utf-8").read().replace("\r\n", "\n").split("\n")
    out = {}
    for head in REG_SECTIONS:
        hits = [k for k, ln in enumerate(lines) if ln.startswith(head)]
        if not hits:
            out[head.strip()] = dict(title=None, sha256=None)
            continue
        i = hits[0]
        j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ") or lines[k].startswith("### ")),
                 len(lines))
        sec = "\n".join(lines[i:j]).strip() + "\n"
        out[head.strip()] = dict(title=lines[i].strip(), sha256=hashlib.sha256(sec.encode("utf-8")).hexdigest())
    return out


def _code_thresholds():
    """Threshold table built from this file's code: for every _row(...)
    call, the check id, the tolerance argument and the pass expression
    (source text, via ast), sorted; plus the tier job table (functions and
    their arguments). Returns (sha256 of the table, sha256 of the jobs,
    table lines)."""
    import ast
    tree = ast.parse(open(os.path.abspath(__file__), encoding="utf-8").read())
    items = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_row":
            a = node.args
            items.append(" | ".join([ast.unparse(a[0]) if a else "",
                                     "tol " + (ast.unparse(a[5]) if len(a) > 5 else ""),
                                     "pass if " + (ast.unparse(a[6]) if len(a) > 6 else "")]))
    items.sort()
    jobs = []
    for st, reg in (("stage1", STAGE1), ("stage2", STAGE2), ("stage3", STAGE3)):
        for tier in ("quick", "full"):
            for name, fn, arglist, kind in reg[tier]:
                jobs.append(f"{st} {tier} {name} {fn.__name__} {kind} {arglist!r}")
    h1 = hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()
    h2 = hashlib.sha256("\n".join(jobs).encode("utf-8")).hexdigest()
    return h1, h2, items


def _registration_state():
    """Current hashes compared with registration_log.json (written by
    --register when a registration section is created or changed, BEFORE
    a tier runs). For every item: the UTC time of the latest log entry with
    the same hash, or None (not registered in that state)."""
    secs = _note_sections()
    h_thr, h_jobs, _ = _code_thresholds()
    try:
        log = json.load(open(REG_LOG, encoding="utf-8"))
    except (OSError, ValueError):
        log = []
    cur = {k: v["sha256"] for k, v in secs.items()}
    cur["code_thresholds"] = h_thr
    cur["tier_jobs"] = h_jobs
    reg = {}
    for k, h in cur.items():
        hit = [e["utc"] for e in log if e["hashes"].get(k) == h]
        reg[k] = dict(sha256=h, registered_utc=hit[-1] if hit else None)
    return reg


def register(note):
    """Append the current registration hashes to registration_log.json."""
    secs = _note_sections()
    h_thr, h_jobs, items = _code_thresholds()
    try:
        log = json.load(open(REG_LOG, encoding="utf-8"))
    except (OSError, ValueError):
        log = []
    hashes = {k: v["sha256"] for k, v in secs.items()}
    hashes.update(code_thresholds=h_thr, tier_jobs=h_jobs)
    entry = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), note=note,
                 titles={k: v["title"] for k, v in secs.items()}, hashes=hashes, threshold_table=items)
    log.append(entry)
    os.makedirs(OUT, exist_ok=True)
    with open(REG_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    print(f"registered at {entry['utc']}: " + ", ".join(f"{k} {v[:12]}" for k, v in hashes.items()))
    return entry


def _job(packed):
    name, fn, a = packed
    t = time.time()
    rows = fn(a)
    for r in rows:
        r["seconds"] = round(time.time() - t, 1)
    return rows


def run(tier, only=None, stages=("stage1",)):
    from studies.parallel_runner import run_parallel
    os.makedirs(OUT, exist_ok=True)
    cpu_jobs, gpu_jobs, order = [], [], []
    for st in stages:
        for name, fn, arglist, kind in REGISTRY[st][tier]:
            if only and name not in only:
                continue
            for a in arglist:
                if kind == "gpu":
                    a = tuple("cuda" if x == "gpu" else x for x in a)
                    gpu_jobs.append((name, fn, a))
                    order.append(("gpu", len(gpu_jobs) - 1))
                else:
                    cpu_jobs.append((name, fn, a))
                    order.append(("cpu", len(cpu_jobs) - 1))
    t0 = time.time()
    start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    reg = _registration_state()
    print(f"=== rankine_verify tier={tier} start {start_utc}; registration record (stage note sections, code "
          f"threshold table, tier jobs):", flush=True)
    for k, v in reg.items():
        print(f"    {k}: sha256 {v['sha256'][:16] if v['sha256'] else None} registered "
              f"{v['registered_utc'] or 'NOT IN registration_log.json'}", flush=True)
    results = []
    gb = float(os.environ.get("RANKINE_GB_PER_JOB", "0.6"))
    # The CPU pool runs in a thread while the GPU jobs run in this process
    # (added after review: wall = max of the two, not the sum). The pool's
    # memory reserve includes ~2 GB for this process's GPU-side host use.
    cpu_out = {}

    def _cpu_pool():
        cpu_out["res"] = run_parallel(_job, cpu_jobs, gb_per_job=gb,
                                      reserve_gb=3.0 + (2.0 if gpu_jobs else 0.0))
    th = None
    if cpu_jobs:
        th = threading.Thread(target=_cpu_pool)
        th.start()
    gpu_res = []
    for job in gpu_jobs:
        try:
            gpu_res.append(_job(job))
        except Exception as e:      # noqa: BLE001
            gpu_res.append([_row(job[0], "EXCEPTION", repr(e), None, "", "", False)])
        print(f"  [{time.time() - t0:6.0f} s] {job[0]} done", flush=True)
    cpu_res = []
    if th is not None:
        th.join()
        for job, r in zip(cpu_jobs, cpu_out["res"]):
            if isinstance(r, Exception):
                cpu_res.append([_row(job[0], "EXCEPTION", repr(r), None, "", "", False)])
            else:
                cpu_res.append(r)
    results = [(gpu_res if k == "gpu" else cpu_res)[i] for k, i in order]
    rows = _postprocess([r for rr in results for r in rr])
    wall = time.time() - t0
    # partial runs (some stages or rows) get their own file name, so they never
    # overwrite the record of a complete tier (B3: the post-review quick json
    # was overwritten by a stage-3-only run before this rule)
    part = "" if tuple(stages) == ("stage1", "stage2", "stage3") else "_" + "_".join(stages)
    path = os.path.join(OUT, f"verify_{tier}{part}{'_' + '_'.join(only) if only else ''}.json")
    info_rows = [r for r in rows if isinstance(r["tol"], str) and r["tol"].startswith("info")]
    bind_rows = [r for r in rows if not (isinstance(r["tol"], str) and r["tol"].startswith("info"))]
    summary = dict(binding_pass=sum(r["ok"] for r in bind_rows), binding=len(bind_rows),
                   info=len(info_rows), all_pass=sum(r["ok"] for r in rows), all=len(rows),
                   failing=[f"{r['check']}: {r['quantity'][:90]}" for r in bind_rows if not r["ok"]])
    with open(path, "w") as f:
        json.dump(dict(tier=tier, start_utc=start_utc, wall_s=wall, registration=reg, summary=summary, rows=rows),
                  f, indent=1, default=float)
    print(f"\n=== rankine_verify tier={tier}  wall {wall:.0f} s  -> {path}")
    for r in rows:
        v = r["value"]
        if isinstance(v, float):
            v = f"{v:.4g}"
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['check']:4s} {r['quantity']}\n"
              f"        value={v}  ref={r['ref']}  tol={r['tol']}  ({r['source']})"
              f"{'  ' + r['note'] if r['note'] else ''}  [{r.get('seconds', '')} s]")
    print(f"=== binding rows (registered thresholds): {summary['binding_pass']}/{summary['binding']} passed; "
          f"information rows: {summary['info']} (pass by construction, not counted); all rows "
          f"{summary['all_pass']}/{summary['all']}")
    for fl in summary["failing"]:
        print(f"    FAIL {fl}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="quick", choices=["quick", "full"])
    ap.add_argument("--only", default="")
    ap.add_argument("--stages", default="stage1,stage2,stage3")
    ap.add_argument("--register", default=None,
                    help="append the current registration hashes (stage note sections 1.1, 8.1, 8.8, code "
                         "threshold table, tier jobs) to registration_log.json with this note, and exit")
    a = ap.parse_args()
    if a.register is not None:
        register(a.register)
        sys.exit(0)
    run(a.tier, [s for s in a.only.split(",") if s] or None,
        tuple(a.stages.split(",")))
