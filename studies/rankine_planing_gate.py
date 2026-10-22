#!/usr/bin/env python3
"""
Stage B7 -- the planing FEASIBILITY GATE of Track B (stage note learn/meta/RANKINE_STAGE1_2026-10-06.md, section 13;
acceptance rules learn/meta/ACCEPTANCE_2026-10-06.md): steady planing of prismatic hard-chine surfaces in the linear
potential theory WITH gravity in the free-surface condition (hydro/rankine/planing_kernel.py, planing_steady.py),
wetted area from the solution (spray-root condition), Kutta condition at the transom; verification (kernel, limits,
convergence) and validation against Savitsky (1964); the GO/NO-GO rule of 13.1.

    .venv/Scripts/python.exe studies/rankine_planing_gate.py --register "note"   # hash 13.1 + code BEFORE running
    .venv/Scripts/python.exe studies/rankine_planing_gate.py --tier quick          # <= 10 min
    .venv/Scripts/python.exe studies/rankine_planing_gate.py --tier full           # <= 1 h, run ONCE at the end

Rows: binding rows carry a registered two-sided tolerance and pass/fail; "info" rows are counted separately.
Outputs: studies/_cache/rankine/planing_gate_<tier>.json/.log; registration log planing_registration_log.json.
"""
import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hydro.rankine import planing_kernel as pk                     # noqa: E402
from hydro.rankine.planing_steady import PlaningSprayRootFast, savitsky   # noqa: E402
from hydro.rankine.vlm_ref import planing_lift_hf                 # noqa: E402

OUT = os.path.join(ROOT, "studies", "_cache", "rankine")
REG_LOG = os.path.join(OUT, "planing_registration_log.json")
STAGE_NOTE = os.path.join(ROOT, "learn", "meta", "RANKINE_STAGE1_2026-10-06.md")
REG_SECTIONS = ("### 13.1",)
CODE_FILES = ("hydro/rankine/planing_kernel.py", "hydro/rankine/planing_steady.py", "hydro/rankine/vlm_ref.py",
              "studies/rankine_planing_gate.py")

# ------------------------------------------------------------------ registered settings and thresholds (13.1)
PROD = dict(h=1.0 / 32, ns=16, cluster="uni", lateral="hat0")     # production discretisation (scheme H-Y0)
GRID_SERIES = ((1.0 / 16, 8), (1.0 / 32, 16), (1.0 / 64, 32))     # (h, ns) refined together
REF = dict(Cv=4.0, beta=20.0, tau=4.0, lam=3.0)                   # reference condition (user's deadrise)
TOL_PK1 = 5e-3      # kernel vs brute-force 2-D FFT, max abs diff / max abs entry
TOL_PK2 = 1e-6      # lateral Filon weights vs accurate direct quadrature (relative)
TOL_PK3 = 2e-2      # hydrostatic limit C_V = 0.01: max |K - K_hyd| / C_V^2
TOL_PK4 = 5e-3      # quadrature / box / folding / damping sensitivity of C_L at REF (relative)
TOL_PC1 = 2e-2      # C_L change between the two finest grids (relative)
TOL_UNUM = 5e-2     # numerical uncertainty U_num (finest vs production C_L at REF) allowed for a NO-GO verdict
TOL_PC1_CP = 1e-2   # l_p/(lam b) change between the two finest grids (absolute)
TOL_PL1 = 3e-2      # high-speed limit vs half the thin-wing lift (relative, extrapolated C_V -> inf)
TOL_PL2 = 0.10      # slender (Wagner-Tulin) keel-chine difference at C_V = 10 (relative)
TOL_VL = 0.10       # C_L vs Savitsky (relative, two-sided)
TOL_VCP = 0.05      # l_p/(lam b) vs Savitsky (absolute)
TOL_VW = 0.15       # (L_K - L_C)/b vs Savitsky's tan(beta)/(pi tan(tau)) (relative)
TOL_VEQ_TAU = 1.0   # equilibrium trim vs Savitsky (deg, absolute)
TOL_VEQ_LAM = 0.15  # equilibrium mean wetted length vs Savitsky (relative)
FRAC_PASS = 0.80    # fraction of validation conditions that must be inside the tolerance
V_BETA = (10.0, 20.0, 30.0)
V_CV = (2.0, 4.0, 6.0)
V_TAU = (2.0, 4.0, 6.0)
V_LAM = (1.5, 2.5, 3.5)
LC_MIN = 0.25       # condition used only if Savitsky's chine wetted length lam - dKC/2 >= LC_MIN (chines wetted)
EQ_CASES = tuple((cv, cd, lcg) for cv in (2.0, 3.0, 4.0, 6.0) for cd in (0.3, 0.6) for lcg in (1.5, 2.0))
QUICK_V = ((20.0, 4.0, 4.0, 2.5), (20.0, 2.0, 4.0, 2.5), (10.0, 4.0, 4.0, 2.5), (30.0, 4.0, 6.0, 2.5))
QUICK_EQ = ((4.0, 0.3, 1.5),)


def _row(rid, what, value, ref, tol, ok, note=""):
    r = dict(id=rid, what=what, value=value, ref=ref, tol=tol, ok=ok, note=note)
    flag = "info" if ok is None else ("PASS" if ok else "FAIL")
    print(f"[{flag}] {rid}: {what} = {value} (ref {ref}, tol {tol}) {note}", flush=True)
    return r


_SOLVERS = {}
_STATE = {}


def solver(Cv, beta, h=None, ns=None, **kw):
    h = PROD["h"] if h is None else h
    ns = PROD["ns"] if ns is None else ns
    key = (Cv, beta, h, ns, tuple(sorted(kw.items())))
    if key not in _SOLVERS:
        # tables do not depend on beta: reuse them across deadrise
        k2 = (Cv, h, ns, tuple(sorted(kw.items())))
        base = next((S for kk, S in _SOLVERS.items() if (kk[0], kk[2], kk[3], kk[4]) == k2), None)
        S = PlaningSprayRootFast(Cv, beta, h=h, ns=ns, cluster=PROD["cluster"], lam_max=6.0,
                                 lateral=PROD["lateral"], **kw) if base is None else _rebeta(base, beta)
        _SOLVERS[key] = S
    return _SOLVERS[key]


def _rebeta(S, beta):
    import copy
    T = copy.copy(S)
    T.beta_deg = float(beta)
    T.tb = math.tan(math.radians(beta))
    return T


# ------------------------------------------------------------------ verification rows
def pk1_bruteforce(quick):
    """YHatTables (production lateral hat machinery) vs a brute-force periodic 2-D FFT with folding in both
    directions, same transfer function and Rayleigh eps; interior and keel elements, specs z_hat and s_hat."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Cv, h, ns, l_mu = 2.0, 1.0 / 16, 4, 20.0
    nodes = np.linspace(0.0, 0.5, ns + 1)
    m_max = 16
    T = pk.YHatTables(Cv, h, nodes, m_max, l_mu=l_mu, box_mult=8.0, n_fold=8, chine="zero")
    targets = T.targets
    ry = 8
    dyf = (nodes[1] - nodes[0]) / ry
    # targets: lateral nodes y_0 .. y_{n-1} (on the fine lattice)
    Nx = int(math.ceil(8 * l_mu / h)); Nx += Nx % 2
    Ny = int(round(40.0 / dyf)); Ny += Ny % 2                      # amendment 2: 40 b in both tiers
    kx = 2 * math.pi * torch.fft.fftfreq(Nx, d=h, dtype=torch.float64, device=dev)
    ky = 2 * math.pi * torch.fft.fftfreq(Ny, d=dyf, dtype=torch.float64, device=dev)
    eps = 1.0 / l_mu
    # amendment 2 (stage note 13.1c): 128 lateral images (the brute-force reference converges slowly in them:
    # 12 % at 8, 5.7 % at 32, 1.2 % at 64, 0.4 % at 128 images for the slope spec); quick tier one combination
    nfx, nfy = 6, 128
    worst = 0.0
    out = []
    combos = ((("s_hat", 0.5, 1), 0),) if quick else tuple((sp, el) for sp in (("z_hat", 0.0, 0), ("s_hat", 0.5, 1))
                                                          for el in (0, 2))
    for (spec_name, shift, deriv), s in combos:
        if True:
            a = nodes[1] - nodes[0]
            ys = nodes[s]
            F = torch.zeros((Ny, Nx), dtype=torch.complex128, device=dev)
            for lx in range(-nfx, nfx + 1):
                kxl = kx + 2 * math.pi * lx / h
                Sx = pk.shape("hat", kxl, h) * torch.exp(1j * kxl * shift * h)
                if deriv:
                    Sx = Sx * (1j * kxl)
                for ly in range(-nfy, nfy + 1):
                    kyl = ky + 2 * math.pi * ly / dyf
                    u = kyl * a / 2
                    Sy = a * torch.where(torch.abs(u) < 1e-12, torch.ones_like(u),
                                         (torch.sin(u) / torch.where(torch.abs(u) < 1e-12, torch.ones_like(u), u)) ** 2)
                    Sy = Sy * (2.0 * torch.cos(kyl * ys) if s > 0 else 1.0)   # element + mirror (keel: own mirror)
                    M = pk.transfer(kxl[None, :].to(torch.complex128), kyl[:, None].to(torch.complex128), Cv, eps)
                    F += M * Sx[None, :] * Sy[:, None].to(torch.complex128)
            G = (torch.fft.ifft2(F) / (h * dyf)).real.cpu().numpy()
            for t in range(len(targets)):
                iy = int(round(targets[t] / dyf))
                assert abs(iy * dyf - targets[t]) < 1e-9
                bf = G[iy % Ny, np.arange(-m_max, m_max + 1) % Nx]
                tab = T.K[spec_name][t, s]
                d = np.abs(bf - tab).max() / max(np.abs(tab).max(), 1e-30)
                worst = max(worst, d)
            out.append((spec_name, s))
    return [_row("PK1", "kernel tables vs brute-force 2-D FFT (C_V 2, h 1/16, 4 lateral intervals; z_hat, s_hat; "
                 "keel and interior elements [quick: s_hat, keel]; all targets, |m| <= 16; 128 lateral images), "
                 "max |diff| / max |entry|",
                 float(worst), 0.0, TOL_PK1, worst <= TOL_PK1)]


def pk2_filon():
    nodes = np.linspace(0.0, 0.5, 9)
    els, _, _ = pk.y_elements(nodes)
    targets = pk.lateral_targets(nodes)
    kys = pk.ky_nodes(4.0, kmax=2e4)
    R = 1.0 / (1.0 + kys ** 2) + 0.3 * kys / (1 + kys) ** 2
    gx, gw = np.polynomial.legendre.leggauss(64)
    worst = 0.0
    for t in (0, 3, 8):
        for s in (0, 4, 8):
            v = pk.filon_hat_weights(kys, targets[t], els[s], pk._pieces_of(nodes, s))
            # accurate reference: per k interval 64-point Gauss in k, W by 400-point Gauss per linear piece
            # (amendment 1, 13.1b: all intervals at once, closed-form piece integrals of cos -- same quantity)
            a, b = kys[:-1, None], kys[1:, None]
            kk = 0.5 * (a + b) + 0.5 * (b - a) * gx[None, :]
            Rk = R[:-1, None] + (R[1:, None] - R[:-1, None]) * (kk - a) / (b - a)
            Wk = _w_accurate(kk.ravel(), targets[t], pk._pieces_of(nodes, s)).reshape(kk.shape)
            ref = float(np.sum(0.5 * (b - a) * gw[None, :] * Wk * Rk))
            worst = max(worst, abs(v @ R - ref) / max(abs(ref), 1e-12))
    return [_row("PK2", "lateral hat Filon weights vs accurate direct quadrature (smooth test R, 3 targets x 3 elements)",
                 float(worst), 0.0, TOL_PK2, worst <= TOL_PK2)]


def _w_accurate(k, yt, pieces):
    """(1/pi) int g(y') cos(k (y_t - y')) dy' exactly for the piecewise-linear g (closed-form integral of
    (alpha + beta y') cos(k (y_t - y')) over each piece; series for small k)."""
    out = np.zeros_like(k)
    ks = np.where(np.abs(k) < 1e-8, 1e-8, k)
    for (ya, yb, ga, gb) in pieces:
        sl = (gb - ga) / (yb - ya)
        # int_ya^yb (ga + sl (y - ya)) cos(k (yt - y)) dy ; u = yt - y
        ua, ub = yt - ya, yt - yb
        # g(y) = ga + sl (yt - ua_) ... write g = c0 + c1 u with u = yt - y: y - ya = ua - u
        c0, c1 = ga + sl * ua, -sl
        # int_{ub}^{ua} (c0 + c1 u) cos(k u) du
        F = lambda u: c0 * np.sin(ks * u) / ks + c1 * (np.cos(ks * u) / ks ** 2 + u * np.sin(ks * u) / ks)
        out += F(ua) - F(ub)
    return out / math.pi


def pk3_hydrostatic():
    Cv, h = 0.01, 1.0 / 32
    nodes = np.linspace(0.0, 0.5, 9)
    T = pk.YHatTables(Cv, h, nodes, 8, l_mu=20.0, chine="zero")
    val = pk.element_values(nodes, T.targets, "zero")
    ref = np.zeros_like(T.K["z_hat"])
    ref[:, :, 8] = -Cv ** 2 * val
    err = np.abs(T.K["z_hat"] - ref).max() / Cv ** 2
    return [_row("PK3", "hydrostatic limit C_V = 0.01 (h 1/32): max |K_z_hat - (-C_V^2 g_s(y_t) delta_m0)| / C_V^2",
                 float(err), 0.0, TOL_PK3, err <= TOL_PK3)]


def _ref_solve(S):
    return S.solve_lambda(math.radians(REF["tau"]), REF["lam"])


def pk4_sensitivity(quick):
    rows = []
    S0 = solver(REF["Cv"], REF["beta"])
    st0 = _ref_solve(S0)
    cl0 = st0["CL"]
    variants = [("kmax x10", dict(ky_kw=dict(kmax=10 * max(2e6, 20 * REF["Cv"] ** 2 * (math.pi / PROD["h"]) ** 2)))),
                ("k_y ratio 1.0025", dict(ky_kw=dict(ratio=1.0025, kmax=max(2e6, 20 * REF["Cv"] ** 2 * (math.pi / PROD["h"]) ** 2)))),
                ("Richardson pair (2 l_mu, 4 l_mu)", dict(l_mu=2 * S0.l_mu))]
    if not quick:
        variants += [("box_mult 12", dict(box_mult=12.0)), ("n_fold 16", dict(n_fold=16))]
    worst = 0.0
    for name, kw in variants:
        S = PlaningSprayRootFast(REF["Cv"], REF["beta"], h=PROD["h"], ns=PROD["ns"], cluster=PROD["cluster"],
                                 lam_max=6.0, lateral=PROD["lateral"], **kw)
        st = _ref_solve(S)
        d = abs(st["CL"] / cl0 - 1)
        worst = max(worst, d)
        rows.append(_row("PK4." + name, "C_L change at REF", float(d), 0.0, TOL_PK4, d <= TOL_PK4))
    S = PlaningSprayRootFast(REF["Cv"], REF["beta"], h=PROD["h"], ns=PROD["ns"], cluster=PROD["cluster"],
                             lam_max=6.0, lateral=PROD["lateral"], richardson_mu=False)
    st = _ref_solve(S)
    rows.append(_row("PK4i", "C_L change without the Richardson extrapolation in eps (damped kernel, l_mu only)",
                     float(abs(st["CL"] / cl0 - 1)), 0.0, None, None))
    return rows


def pc1_grid(quick):
    rows = []
    series = GRID_SERIES[:2] if quick else GRID_SERIES
    res = []
    for h, ns in series:
        t0 = time.time()
        S = PlaningSprayRootFast(REF["Cv"], REF["beta"], h=h, ns=ns, cluster=PROD["cluster"], lam_max=6.0,
                                 lateral=PROD["lateral"])
        st = _ref_solve(S)
        res.append((h, ns, st["CL"], st["lp"] / st["lam"], st["LK"] - st["LC"], st["d"], st["converged"]))
        rows.append(_row(f"PC1.h{round(1 / h)}ns{ns}", "REF: C_L, l_p/(lam b), L_K-L_C, d, converged",
                         [round(st["CL"], 6), round(st["lp"] / st["lam"], 5), round(st["LK"] - st["LC"], 4),
                          round(st["d"], 5), st["converged"]], None, None, None, f"{time.time() - t0:.0f}s"))
    a, b = res[-2], res[-1]
    dcl = abs(b[2] / a[2] - 1)
    prod = next((r for r in res if r[0] == PROD["h"] and r[1] == PROD["ns"]), a)
    _STATE["U_num"] = abs(res[-1][2] / prod[2] - 1)
    rows.append(_row("PC1u", "U_num = |C_L(finest) / C_L(production) - 1| at REF (used by the verdict rule)",
                     float(_STATE["U_num"]), 0.0, TOL_UNUM, _STATE["U_num"] <= TOL_UNUM))
    dcp = abs(b[3] - a[3])
    rows.append(_row("PC1", "C_L change between the two finest grids (h, ns refined together)", float(dcl), 0.0,
                     TOL_PC1, (dcl <= TOL_PC1) and a[6] and b[6]))
    rows.append(_row("PC1cp", "l_p/(lam b) change between the two finest grids", float(dcp), 0.0, TOL_PC1_CP,
                     (dcp <= TOL_PC1_CP) and a[6] and b[6]))
    if len(res) == 3:
        e1, e2 = res[1][2] - res[0][2], res[2][2] - res[1][2]
        p = math.log(abs(e1 / e2)) / math.log(2.0) if e2 != 0 and e1 * e2 > 0 else float("nan")
        rows.append(_row("PC1p", "observed order of C_L (refinement ratio 2)", p, None, None, None,
                         "monotone" if e1 * e2 > 0 else "non-monotone"))
    return rows


def pl1_highspeed(quick):
    """High-speed limit: rectangular flat planform (lam 2), planform mode (given wetted area, lifting surface with
    gravity) at C_V 10 and 20 with l_mu = 50 (Richardson), extrapolated in 1/C_V^2; vs half the thin-wing VLM lift."""
    rows = []
    tau = math.radians(4.0)
    lam = 2.0
    cls = []
    cvs = (10.0, 20.0)
    for cv in cvs:
        S = PlaningSprayRootFast(cv, 0.0, h=PROD["h"], ns=PROD["ns"], cluster=PROD["cluster"], lam_max=3.0,
                                 lateral=PROD["lateral"], l_mu=50.0)
        n = np.full(S.ns, int(round(lam / S.h)))
        P = S.planform(n)
        cl = 4.0 * np.sum(S.w * P["Ls"]) * math.tan(tau)
        lp = np.sum(S.w * P["Ms"]) / np.sum(S.w * P["Ls"])
        cls.append((cl, lp))
        rows.append(_row(f"PL1.Cv{cv:g}", "rectangular planform lam 2, tau 4 deg: C_L, l_p", [round(cl, 6), round(lp, 4)],
                         None, None, None))
    # extrapolate in 1/C_V^2
    x1, x2 = 1 / cvs[0] ** 2, 1 / cvs[1] ** 2
    cl_inf = cls[1][0] + (cls[1][0] - cls[0][0]) * (0 - x2) / (x2 - x1)
    lp_inf = cls[1][1] + (cls[1][1] - cls[0][1]) * (0 - x2) / (x2 - x1)
    # VLM: Richardson in resolution (ns, nc) -> (2 ns, 2 nc)
    v1 = planing_lift_hf(np.linspace(0, 0.5, 13), np.full(12, lam), math.tan(tau), nc=30, spacing="cos")
    v2 = planing_lift_hf(np.linspace(0, 0.5, 25), np.full(24, lam), math.tan(tau), nc=60, spacing="cos")
    v3 = None if quick else planing_lift_hf(np.linspace(0, 0.5, 37), np.full(36, lam), math.tan(tau), nc=80,
                                            spacing="cos")
    vl = v2["CL"] if v3 is None else v3["CL"]
    vlp = v2["lp"] if v3 is None else v3["lp"]
    rows.append(_row("PL1v", "VLM half-wing C_L at increasing resolution", [round(v1["CL"], 6), round(v2["CL"], 6),
                     None if v3 is None else round(v3["CL"], 6)], None, None, None))
    d = abs(cl_inf / vl - 1)
    rows.append(_row("PL1", "high-speed limit: C_L (C_V -> inf, extrapolated) vs half the thin-wing VLM lift",
                     float(cl_inf), float(vl), TOL_PL1, d <= TOL_PL1, f"rel diff {d:.4f}"))
    rows.append(_row("PL1lp", "high-speed limit: centre of pressure from the transom / b", float(lp_inf), float(vlp),
                     None, None))
    return rows


def pl2_wagner(quick):
    rows = []
    S = solver(10.0, REF["beta"], l_mu=50.0)            # near-field limit: l_mu 50 b with Richardson (as PL1)
    for lam in ((3.0,) if quick else (3.0, 4.0)):
        st = S.solve_lambda(math.radians(REF["tau"]), lam)
        dk = math.tan(math.radians(REF["beta"])) / (math.pi * math.tan(math.radians(REF["tau"])))
        d = abs((st["LK"] - st["LC"]) / dk - 1)
        rows.append(_row(f"PL2.lam{lam:g}", "slender limit: (L_K - L_C)/b at C_V 10, beta 20, tau 4 vs tan(beta)/(pi tan(tau))",
                         float(st["LK"] - st["LC"]), float(dk), TOL_PL2, (d <= TOL_PL2) and st["converged"]))
    return rows


# ------------------------------------------------------------------ validation rows
def v_conditions(quick):
    if quick:
        return list(QUICK_V)
    out = []
    for b in V_BETA:
        for cv in V_CV:
            for tau in V_TAU:
                for lam in V_LAM:
                    dkc = math.tan(math.radians(b)) / (math.pi * math.tan(math.radians(tau)))
                    if lam - 0.5 * dkc >= LC_MIN:
                        out.append((b, cv, tau, lam))
    return out


def v_savitsky(quick):
    rows = []
    recs = []
    for (b, cv, tau, lam) in v_conditions(quick):
        S = solver(cv, b)
        t0 = time.time()
        st = S.solve_lambda(math.radians(tau), lam)
        sv = savitsky(tau, lam, cv, b)
        rec = dict(beta=b, Cv=cv, tau=tau, lam=lam, CL=st["CL"], CL_sav=sv["CL"], cp=st["lp"] / st["lam"],
                   cp_sav=sv["cp"], dKC=st["LK"] - st["LC"], dKC_sav=sv["LKmLC"], d=st["d"],
                   converged=st["converged"], chine=S.chine_ratio(st), t=time.time() - t0)
        recs.append(rec)
        print(f"   V b{b:g} Cv{cv:g} tau{tau:g} lam{lam:g}: CL {rec['CL']:.5f} (Sav {rec['CL_sav']:.5f}, "
              f"{rec['CL'] / rec['CL_sav']:.3f}) cp {rec['cp']:.3f} ({rec['cp_sav']:.3f}) dKC {rec['dKC']:.3f} "
              f"({rec['dKC_sav']:.3f}) conv {rec['converged']} chine {rec['chine']:.2f} {rec['t']:.1f}s", flush=True)

    def frac(sel, test):
        s = [r for r in recs if sel(r)]
        return (sum(1 for r in s if test(r)) / len(s), len(s)) if s else (float("nan"), 0)
    okL = lambda r: r["converged"] and abs(r["CL"] / r["CL_sav"] - 1) <= TOL_VL
    okC = lambda r: r["converged"] and abs(r["cp"] - r["cp_sav"]) <= TOL_VCP
    okW = lambda r: r["converged"] and abs(r["dKC"] / r["dKC_sav"] - 1) <= TOL_VW
    un = _STATE.get("U_num", 0.0)
    okLr = lambda r: r["converged"] and abs(r["CL"] / r["CL_sav"] - 1) <= TOL_VL + un
    for tag, sel in (("all", lambda r: True), ("beta20", lambda r: r["beta"] == 20.0)):
        f, n = frac(sel, okL)
        rows.append(_row(f"VL.{tag}", f"fraction of conditions with C_L within +-{TOL_VL:.0%} of Savitsky", f, FRAC_PASS,
                         TOL_VL, f >= FRAC_PASS, f"n={n}"))
        fr, n = frac(sel, okLr)
        rows.append(_row(f"VLR.{tag}", f"robust fraction: C_L within +-({TOL_VL:.0%} + U_num) of Savitsky", fr,
                         FRAC_PASS, round(TOL_VL + un, 4), fr >= FRAC_PASS, f"n={n}, U_num={un:.4f}"))
    f, n = frac(lambda r: True, okC)
    rows.append(_row("VCP", f"fraction with l_p/(lam b) within +-{TOL_VCP} of Savitsky", f, FRAC_PASS, TOL_VCP,
                     f >= FRAC_PASS, f"n={n}"))
    f, n = frac(lambda r: True, okW)
    rows.append(_row("VW", f"fraction with (L_K-L_C)/b within +-{TOL_VW:.0%} of Savitsky", f, FRAC_PASS, TOL_VW,
                     f >= FRAC_PASS, f"n={n}"))
    for b in sorted(set(r["beta"] for r in recs)):
        rr = [r["CL"] / r["CL_sav"] for r in recs if r["beta"] == b]
        rows.append(_row(f"VLr.beta{b:g}", "C_L / C_L,Savitsky: min, median, max", [round(min(rr), 3),
                         round(float(np.median(rr)), 3), round(max(rr), 3)], None, None, None))
    rows.append(_row("VCH", "chine pressure ratio (strip-mean pressure extrapolated to the chine / mean pressure): "
                     "min, median, max (0 = smooth detachment)",
                     [round(min(r["chine"] for r in recs), 3), round(float(np.median([r["chine"] for r in recs])), 3),
                      round(max(r["chine"] for r in recs), 3)], None, None, None))
    return rows, recs


def savitsky_equilibrium(Cv, CD, lcg, beta):
    """Savitsky's equations at equilibrium with the pressure force only: c.p. at the CG
    (lam * cp(lam, Cv) = lcg) and C_L,beta(tau, lam) = C_Delta / (0.5 C_V^2)."""
    from scipy.optimize import brentq
    lam = brentq(lambda l: l * savitsky(4.0, l, Cv, beta)["cp"] - lcg, 0.2, 10.0)
    CLb = CD / (0.5 * Cv ** 2)
    tau = brentq(lambda t: savitsky(t, lam, Cv, beta)["CL"] - CLb, 0.2, 40.0)
    return tau, lam


def v_equilibrium(quick):
    rows = []
    recs = []
    beta = REF["beta"]
    for (cv, cd, lcg) in (QUICK_EQ if quick else EQ_CASES):
        tau_s, lam_s = savitsky_equilibrium(cv, cd, lcg, beta)
        inside = (2.0 <= tau_s <= 15.0) and (lam_s <= 4.0)
        S = solver(cv, beta)
        lift = 0.5 * cd / (0.5 * cv ** 2)            # lift / (rho U^2 b^2) = C_L / 2
        t0 = time.time()
        try:
            st = S.equilibrium(lift, lcg, tau0=math.radians(max(tau_s, 2.0)), lam0=lam_s)
            tau_m, lam_m, ok_solve = math.degrees(st["tau"]), st["lam"], bool(st.get("eq_ok")) and st["converged"]
        except Exception as e:                       # noqa: BLE001
            tau_m, lam_m, ok_solve = float("nan"), float("nan"), False
        rec = dict(Cv=cv, CD=cd, lcg=lcg, tau=tau_m, lam=lam_m, tau_sav=tau_s, lam_sav=lam_s, inside=inside,
                   solved=ok_solve, t=time.time() - t0)
        recs.append(rec)
        print(f"   EQ Cv{cv:g} CD{cd:g} lcg{lcg:g}: tau {tau_m:.3f} (Sav {tau_s:.3f}) lam {lam_m:.3f} (Sav {lam_s:.3f}) "
              f"inside {inside} solved {ok_solve} {rec['t']:.0f}s", flush=True)
    use = [r for r in recs if r["inside"]]
    ok = [r for r in use if r["solved"] and abs(r["tau"] - r["tau_sav"]) <= TOL_VEQ_TAU]
    okl = [r for r in use if r["solved"] and abs(r["lam"] / r["lam_sav"] - 1) <= TOL_VEQ_LAM]
    f = len(ok) / len(use) if use else float("nan")
    fl = len(okl) / len(use) if use else float("nan")
    rows.append(_row("VEQ", f"equilibrium trim within +-{TOL_VEQ_TAU} deg of Savitsky (beta 20, pressure moment only)", f,
                     FRAC_PASS, TOL_VEQ_TAU, f >= FRAC_PASS, f"n={len(use)}"))
    rows.append(_row("VEQl", f"equilibrium mean wetted length within +-{TOL_VEQ_LAM:.0%} of Savitsky", fl, FRAC_PASS,
                     TOL_VEQ_LAM, fl >= FRAC_PASS, f"n={len(use)}"))
    return rows, recs


def v_flat(quick):
    if quick:
        return [], []
    rows, recs = [], []
    for cv in (4.0,):
        S = solver(cv, 0.0)
        for tau, lam in ((4.0, 2.5), (6.0, 2.5), (4.0, 1.5)):
            st = S.solve_lambda(math.radians(tau), lam)
            sv = savitsky(tau, lam, cv, 0.0)
            recs.append(dict(Cv=cv, tau=tau, lam=lam, CL=st["CL"], CL_sav=sv["CL"], converged=st["converged"]))
            rows.append(_row(f"VF.Cv{cv:g}tau{tau:g}lam{lam:g}", "flat plate (beta 0): C_L / C_L,Savitsky",
                             round(st["CL"] / sv["CL"], 4), 1.0, None, None, f"converged {st['converged']}"))
    return rows, recs


# ------------------------------------------------------------------ registration
def _code_hashes():
    out = {}
    for pat in CODE_FILES:
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            rel = os.path.relpath(f, ROOT).replace("\\", "/")
            out[rel] = hashlib.sha256(open(f, "rb").read()).hexdigest()
    return out


def _note_sections():
    lines = open(STAGE_NOTE, encoding="utf-8").read().replace("\r\n", "\n").split("\n")
    out = {}
    for head in REG_SECTIONS:
        hits = [k for k, ln in enumerate(lines) if ln.startswith(head)]
        if not hits:
            out["b7:" + head.strip()] = None
            continue
        i = hits[0]
        j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ") or lines[k].startswith("### ")),
                 len(lines))
        out["b7:" + head.strip()] = hashlib.sha256(("\n".join(lines[i:j]).strip() + "\n").encode("utf-8")).hexdigest()
    return out


def _consts():
    names = [k for k in globals() if k.isupper() and k not in ("ROOT", "OUT", "REG_LOG", "STAGE_NOTE")]
    s = "\n".join(f"{k}={globals()[k]!r}" for k in sorted(names))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def register(note):
    os.makedirs(OUT, exist_ok=True)
    try:
        log = json.load(open(REG_LOG, encoding="utf-8"))
    except (OSError, ValueError):
        log = []
    hashes = _note_sections()
    hashes["b7:constants"] = _consts()
    entry = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), note=note,
                 hashes=hashes, code_sha256=_code_hashes())
    log.append(entry)
    json.dump(log, open(REG_LOG, "w", encoding="utf-8"), indent=1)
    print(f"registered at {entry['utc']}: " + ", ".join(f"{k} {(v or 'None')[:12]}" for k, v in hashes.items()))
    for k, v in entry["code_sha256"].items():
        print(f"    {k} {v[:12]}")


def run(tier):
    quick = tier == "quick"
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    start = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    try:
        log = json.load(open(REG_LOG, encoding="utf-8"))
    except (OSError, ValueError):
        log = []
    code0 = _code_hashes()
    secs = _note_sections()
    consts = _consts()
    reg_code = [e["utc"] for e in log if e["code_sha256"] == code0]
    reg_sec = [e["utc"] for e in log if all(e["hashes"].get(k) == v for k, v in secs.items())
               and e["hashes"].get("b7:constants") == consts]
    print(f"=== rankine_planing_gate tier={tier} start {start}; code registered: {reg_code[-1] if reg_code else 'NO'}; "
          f"13.1 + constants registered: {reg_sec[-1] if reg_sec else 'NO'}", flush=True)
    rows, extra = [], {}
    jobs = [("PK2", lambda: pk2_filon()), ("PK3", lambda: pk3_hydrostatic()), ("PK1", lambda: pk1_bruteforce(quick)),
            ("PK4", lambda: pk4_sensitivity(quick)), ("PC1", lambda: pc1_grid(quick)),
            ("PL1", lambda: pl1_highspeed(quick)), ("PL2", lambda: pl2_wagner(quick))]
    for name, fn in jobs:
        t = time.time()
        try:
            rows += fn()
        except Exception as e:                       # noqa: BLE001
            rows.append(_row(name, "job raised", repr(e)[:300], None, None, False))
        print(f"--- {name} {time.time() - t:.0f}s", flush=True)
    for name, fn in (("V", v_savitsky), ("VEQ", v_equilibrium), ("VF", v_flat)):
        t = time.time()
        try:
            r, recs = fn(quick)
            rows += r
            extra[name] = recs
        except Exception as e:                       # noqa: BLE001
            rows.append(_row(name, "job raised", repr(e)[:300], None, None, False))
        print(f"--- {name} {time.time() - t:.0f}s", flush=True)
    wall = time.time() - t0
    lim = 600 if quick else 3600
    rows.append(_row("T", "tier wall time [s]", round(wall, 1), lim, None, wall <= lim))
    binding = [r for r in rows if r["ok"] is not None]
    npass = sum(1 for r in binding if r["ok"])
    ver_ok = all(r["ok"] for r in binding if r["id"].startswith(("PK", "PC", "PL")))
    num_adequate = all(r["ok"] for r in binding if r["id"].startswith(("PK", "PL", "PC1u")))
    val_rows = [r for r in binding if r["id"] in ("VL.all", "VL.beta20", "VCP", "VEQ")]
    val_ok = bool(val_rows) and all(r["ok"] for r in val_rows)
    robust_fail = any((not r["ok"]) for r in binding if r["id"] in ("VLR.all", "VLR.beta20"))
    if ver_ok and val_ok:
        verdict = "GO"
    elif num_adequate and robust_fail:
        verdict = "NO-GO"
    else:
        verdict = "INCONCLUSIVE"
    if quick:
        verdict = "quick tier (no verdict): " + verdict
    end = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    code1 = _code_hashes()
    print(f"=== {npass}/{len(binding)} binding rows pass; verification ok {ver_ok}; validation ok {val_ok}; "
          f"VERDICT {verdict}; wall {wall:.0f}s; code changed during run: {code1 != code0}", flush=True)
    json.dump(dict(tier=tier, start=start, end=end, wall=wall, rows=rows, extra=extra, verdict=verdict,
                   code_sha256_start=code0, code_registered=reg_code[-1] if reg_code else None,
                   reg_13_1=reg_sec[-1] if reg_sec else None, code_changed_during_run=code1 != code0),
              open(os.path.join(OUT, f"planing_gate_{tier}.json"), "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("quick", "full"))
    ap.add_argument("--register")
    a = ap.parse_args()
    if a.register:
        register(a.register)
    elif a.tier:
        run(a.tier)
