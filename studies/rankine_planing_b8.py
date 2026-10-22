#!/usr/bin/env python3
"""
Stage B8 -- route 1 of the planing gate (stage note learn/meta/RANKINE_STAGE1_2026-10-06.md, section 14; acceptance
rules learn/meta/ACCEPTANCE_2026-10-06.md): the 3-D linear free surface with gravity of B7 with the flat-plate cross-
section replaced by a nonlinear wedge-entry section (modified Logvinovich model) and flow separation at wetted chines
(hydro/rankine/planing_wedge2d.py, planing_nl.py); 2-D checks against Zhao & Faltinsen (1993), the gravity-free limit
(PL1 corrected), convergence, validation against Savitsky (1964); the GO/NO-GO rule of 14.3.

    .venv/Scripts/python.exe studies/rankine_planing_b8.py --register "note"   # hash 14.3 + constants + code BEFORE running
    .venv/Scripts/python.exe studies/rankine_planing_b8.py --tier quick          # <= 10 min
    .venv/Scripts/python.exe studies/rankine_planing_b8.py --tier full           # <= 1 h, run ONCE

Outputs: studies/_cache/rankine/planing_b8_<tier>.json/.log; registration log planing_registration_log.json (shared
with B7, keys prefixed b8:).
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

from hydro.rankine import planing_kernel as pk                        # noqa: E402
from hydro.rankine import planing_wedge2d as w2                       # noqa: E402
from hydro.rankine import planing_nl as nl                            # noqa: E402
from hydro.rankine.planing_steady import PlaningSprayRootFast, savitsky   # noqa: E402
from hydro.rankine.vlm_ref import planing_lift_hf                     # noqa: E402

OUT = os.path.join(ROOT, "studies", "_cache", "rankine")
REG_LOG = os.path.join(OUT, "planing_registration_log.json")
STAGE_NOTE = os.path.join(ROOT, "learn", "meta", "RANKINE_STAGE1_2026-10-06.md")
REG_SECTIONS = ("### 14.2", "### 14.3")
CODE_FILES = ("hydro/rankine/planing_kernel.py", "hydro/rankine/planing_steady.py", "hydro/rankine/vlm_ref.py",
              "hydro/rankine/planing_wedge2d.py", "hydro/rankine/planing_nl.py", "studies/rankine_planing_b8.py")

# ------------------------------------------------------------------ registered settings and thresholds (14.3)
PROD = dict(h=1.0 / 32, ns=32, cluster="uni", lateral="hat0")     # B7 scheme, lateral intervals doubled (14.1)
GRID_SERIES = ((1.0 / 16, 16), (1.0 / 32, 32), (1.0 / 64, 64))
REF = dict(Cv=4.0, beta=20.0, tau=4.0, lam=3.0)
MLM_PUBLISHED = {10.0: 40.89, 20.0: 10.49, 30.0: 4.81}            # MLM p_max/(rho V^2), Korobkin model as tabulated
ZF_CPMAX = {10.0: 77.85, 20.0: 17.77, 30.0: 6.927}               # Zhao & Faltinsen 1993 similarity, p_max/(rho V^2/2)
ZF_ZMAX = {10.0: 0.5556, 20.0: 0.5087, 30.0: 0.4243}             # height of the pressure peak / (V t)
TOL_W2A = 1.5e-3    # MLM code vs published MLM values (rounding of the 2-decimal table)
TOL_W2B = 1e-8      # closed-form force integral vs direct quadrature of the MLM pressure (relative)
TOL_W2C = 0.10      # MLM force vs similarity-solution force (acceptance V row, Zhao & Faltinsen: +-10 %)
TOL_W2P = 0.10      # MLM peak pressure vs Zhao & Faltinsen similarity (acceptance V row: +-10 %)
TOL_W2S = 1e-8      # steady free-streamline force at beta 0 vs Kirchhoff 2 pi/(pi + 4)
TOL_C1 = 1e-6       # quadrature of the composite increments: int F_W dx vs closed form pi V U / 8
TOL_PL1M = 0.03     # gravity-free limit (C_V = inf, off-kink collocation) vs half-wing VLM
TOL_PC1 = 2e-2      # composite C_L change between the two finest grids
TOL_UNUM = 5e-2     # numerical uncertainty allowed for a NO-GO verdict
TOL_PL2 = 0.10      # slender keel-chine difference (B7 PL2, re-run: the inner velocity (K) depends on it)
TOL_VL = 0.10       # composite C_L vs Savitsky (relative, two-sided)
TOL_VEQ_TAU = 1.0   # equilibrium trim vs Savitsky (deg)
TOL_VEQ_LAM = 0.15
FRAC_PASS = 0.80
V_BETA = (10.0, 20.0, 30.0)
V_CV = (2.0, 4.0, 6.0)
V_TAU = (2.0, 4.0, 6.0)
V_LAM = (1.5, 2.5, 3.5)
LC_MIN = 0.25
EQ_CASES = tuple((cv, cd, lcg) for cv in (2.0, 3.0, 4.0, 6.0) for cd in (0.3, 0.6) for lcg in (1.5, 2.0))
QUICK_V = ((20.0, 4.0, 4.0, 2.5), (20.0, 2.0, 4.0, 2.5), (20.0, 6.0, 4.0, 2.5), (10.0, 4.0, 4.0, 2.5),
           (30.0, 4.0, 6.0, 2.5))
QUICK_EQ = ((4.0, 0.3, 1.5),)
PL1M_SERIES = ((1.0 / 16, 8), (1.0 / 32, 16), (1.0 / 64, 32))
PL1B_REF_NS = 64    # reference lateral resolution of the kink-bias row (h 1/32; mean of node and off-kink collocation)
VARIANTS = (("mlm_only", dict(part="mlm_only")), ("sep_only", dict(part="sep_only")),
            ("anchor_wen", dict(anchor="wen")), ("v_slender", dict(vmode="slender")))


def _row(rid, what, value, ref, tol, ok, note=""):
    r = dict(id=rid, what=what, value=value, ref=ref, tol=tol, ok=ok, note=note)
    flag = "info" if ok is None else ("PASS" if ok else "FAIL")
    print(f"[{flag}] {rid}: {what} = {value} (ref {ref}, tol {tol}) {note}", flush=True)
    return r


_SOLVERS = {}
_STATE = {}


def solver(Cv, beta, h=None, ns=None, **kw):
    import copy
    h = PROD["h"] if h is None else h
    ns = PROD["ns"] if ns is None else ns
    key = (Cv, beta, h, ns, tuple(sorted(kw.items())))
    if key not in _SOLVERS:
        k2 = (Cv, h, ns, tuple(sorted(kw.items())))
        base = next((S for kk, S in _SOLVERS.items() if (kk[0], kk[2], kk[3], kk[4]) == k2), None)
        if base is None:
            S = PlaningSprayRootFast(Cv, beta, h=h, ns=ns, cluster=PROD["cluster"], lam_max=6.0,
                                     lateral=PROD["lateral"], **kw)
        else:
            S = copy.copy(base)
            S.beta_deg = float(beta)
            S.tb = math.tan(math.radians(beta))
        _SOLVERS[key] = S
    return _SOLVERS[key]


# ------------------------------------------------------------------ 2-D section checks
def w2_checks():
    rows = []
    worst_a, worst_b = 0.0, 0.0
    for b, ref in MLM_PUBLISHED.items():
        r = w2.similarity_coefficients(b)
        worst_a = max(worst_a, abs(0.5 * r["Cp_max"] / ref - 1))
        # W2B: closed form vs direct quadrature (constant V, chine-dry, h = 1)
        from scipy.integrate import quad
        T = math.tan(math.radians(b))
        c, cdot = math.pi / (2 * T), math.pi / (2 * T)
        th = w2._cutoff_theta(c, cdot, 1.0, 0.0, 1.0, T, math.pi / 2 - 1e-15)
        num = 2 * quad(lambda t: float(w2.mlm_pressure_theta(t, c, cdot, 1.0, 0.0, 1.0, T)) * c * math.cos(t), 0, th,
                       limit=400, epsabs=1e-14, epsrel=1e-13)[0]
        worst_b = max(worst_b, abs(num / r["F"] - 1))
        # also with acceleration (Vdot = 0.3), separation-free
        Vd = 0.3
        th2 = w2._cutoff_theta(c, cdot, 1.0, Vd, 1.0, T, math.pi / 2 - 1e-15)
        bb = math.atan(T)
        cf = w2._force_terms(c, cdot, 1.0, Vd, 1.0, T, math.sin(bb) ** 2, math.cos(bb) ** 2, th2)
        num2 = 2 * quad(lambda t: float(w2.mlm_pressure_theta(t, c, cdot, 1.0, Vd, 1.0, T)) * c * math.cos(t), 0, th2,
                        limit=400, epsabs=1e-14, epsrel=1e-13)[0]
        worst_b = max(worst_b, abs(num2 / cf - 1))
        bf = math.radians(b)
        dC = r["F"] / nl.kappa_wen(bf) - 1
        rows.append(_row(f"W2C.beta{b:g}", "MLM force F/(rho V^3 t) vs similarity solution (Wen et al. 2022 k2 cot b)",
                         round(r["F"], 4), round(nl.kappa_wen(bf), 4), TOL_W2C, abs(dC) <= TOL_W2C,
                         f"rel diff {dC:+.4f}; Wagner {r['F_wagner']:.3f}, MLM/Wagner {r['F'] / r['F_wagner']:.4f}"))
        dP = r["Cp_max"] / ZF_CPMAX[b] - 1
        rows.append(_row(f"W2P.beta{b:g}", "MLM peak pressure p_max/(rho V^2/2) vs Zhao & Faltinsen 1993",
                         round(r["Cp_max"], 3), ZF_CPMAX[b], TOL_W2P, abs(dP) <= TOL_W2P, f"rel diff {dP:+.4f}"))
        rows.append(_row(f"W2Z.beta{b:g}", "height of the pressure peak / (V t): MLM vs Zhao & Faltinsen 1993",
                         round(r["z_max"], 4), ZF_ZMAX[b], None, None))
    rows.insert(0, _row("W2A", "MLM code: p_max/(rho V^2) vs published MLM values (10/20/30 deg)", float(worst_a), 0.0,
                        TOL_W2A, worst_a <= TOL_W2A))
    rows.insert(1, _row("W2B", "closed-form MLM force vs direct quadrature (V const and Vdot = 0.3)", float(worst_b), 0.0,
                        TOL_W2B, worst_b <= TOL_W2B))
    cs0 = nl.cs_inf(0.0)
    e = abs(cs0 / (2 * math.pi / (math.pi + 4)) - 1)
    rows.append(_row("W2S", "steady free-streamline force at beta 0 vs Kirchhoff 2 pi/(pi + 4)", float(e), 0.0, TOL_W2S,
                     e <= TOL_W2S))
    for b in V_BETA:
        bf = math.radians(b)
        rows.append(_row(f"W2I.beta{b:g}", "steady supercavitating force Cs_inf; MLM vs Wen Cs at chine wetting",
                         [round(nl.cs_inf(bf), 4), round(nl.kappa_mlm(b) * 2 / math.pi * math.tan(bf), 4),
                          round(nl.kappa_wen(bf) * 2 / math.pi * math.tan(bf), 4)], None, None, None))
        # the rejected zero-pressure (Logvinovich-Tassin) separation model: transition impulse vs Wen over [2/pi, 2]
        T = math.tan(bf)
        r = w2.constant_speed(T, 1.0, 2.0 * T, n=4000)
        s = r["h"] / T
        m = (s >= 2 / math.pi) & np.isfinite(r["F"])
        hcw = 2 / math.pi * T
        csm = nl.kappa_wen(bf) * 2 / math.pi * T
        cs_w = nl.cs_inf(bf) + nl.wen_cstar((r["h"][m] - hcw) / hcw / T ** nl.WEN_LAMBDA) * (csm - nl.cs_inf(bf))
        rows.append(_row(f"W2R.beta{b:g}", "rejected zero-pressure separation model: transition impulse / Wen et al.",
                         round(float(np.trapezoid(r["F"][m], s[m]) / np.trapezoid(cs_w, s[m])), 4), 1.0, None, None))
    # C1: quadrature of the increments
    x, FN, FW = nl.section_forces(20.0, math.tan(math.radians(4.0)), 3.5, 1.84)
    ex = math.pi * math.tan(math.radians(4.0)) / 8
    e = abs(np.trapezoid(FW, x) / ex - 1)
    rows.append(_row("C1", "int F_W dx (Wagner sections, chine-dry) vs closed form pi V U/8", float(e), 0.0, TOL_C1,
                     e <= TOL_C1))
    return rows


# ------------------------------------------------------------------ gravity-free limit (PL1 corrected)
def _pl1_tables(Cv, h, nodes, targets, m_max, l_mu, kmax):
    specs = (("s_hat", 0.5, 1, "hat"), ("s_half", 0.5, 1, "half"))
    kys = pk.ky_nodes(Cv, kmax=kmax, fine=16)
    I, _ = pk.x_transform_specs(Cv, h, kys, m_max, l_mu, specs, 8.0, 8, 0.25)
    W, val, wy = pk.hat_weight_matrix(kys, nodes, targets, "zero")
    m = np.arange(-m_max, m_max + 1)
    K = {}
    for name, shift, deriv, el in specs:
        dv = np.zeros(len(m))
        if el == "hat":
            dv[m == -1] = 1 / h
            dv[m == 0] = -1 / h
        else:
            dv[m == 0] = -1 / h
        xinf = np.zeros(len(m)) if math.isinf(Cv) else -Cv ** 2 * dv
        K[name] = np.einsum("tsk,km->tsm", W, I[name] - xinf[None, :]) + val[:, :, None] * xinf[None, None, :]
    return K, wy


def planform_lift_rect(Cv, h, ns, lam=2.0, tau=4.0, mode="mid"):
    """Rectangular planform (all lines wetted over lam), lifting surface with gravity, production x discretisation and
    lateral hats (pressure zero at the chine); collocation on the lateral nodes (production, 'node' = on the hat kinks)
    or at the interval midpoints ('mid', off the kinks). Richardson in the damping as in B7."""
    nodes = np.linspace(0, 0.5, ns + 1)
    targets = nodes[:-1] if mode == "node" else 0.5 * (nodes[:-1] + nodes[1:])
    Nc = int(round(lam / h))
    m_max = Nc + 4
    kmax = 2e6 if math.isinf(Cv) else max(2e6, 20 * Cv * Cv * (math.pi / h) ** 2)
    l_mu = 400.0 if math.isinf(Cv) else float(np.clip(0.5 * 2 * math.pi * Cv * Cv, 20, 400))
    K1, wy = _pl1_tables(Cv, h, nodes, targets, m_max, l_mu, kmax)
    K2, _ = _pl1_tables(Cv, h, nodes, targets, m_max, 2 * l_mu, kmax)
    K = {k: 2 * K2[k] - K1[k] for k in K1}
    cs = np.repeat(np.arange(ns), Nc)
    cj = np.tile(np.arange(1, Nc + 1), ns)
    ce = (cj == Nc).astype(int)
    rt = np.repeat(np.arange(ns), Nc)
    ri = np.tile(np.arange(Nc), ns)
    A = np.zeros((len(rt), len(cs)))
    for code, nm in ((0, "s_hat"), (1, "s_half")):
        cm = np.nonzero(ce == code)[0]
        A[:, cm] = K[nm][rt[:, None], cs[cm][None, :], cj[cm][None, :] - ri[:, None] - 1 + m_max]
    x = np.linalg.solve(A, -np.ones(len(rt)))
    wx = np.where(ce == 0, h, 0.5 * h)
    Ls = np.bincount(cs, weights=wx * x, minlength=ns)
    return 4.0 * np.sum(wy * Ls) * math.tan(math.radians(tau))


def pl1m(quick):
    rows = []
    series = PL1M_SERIES[:2] if quick else PL1M_SERIES
    cls = []
    for h, ns in series:
        t0 = time.time()
        cl = planform_lift_rect(float("inf"), h, ns, mode="mid")
        cls.append(cl)
        rows.append(_row(f"PL1M.h{round(1 / h)}ns{ns}", "C_V = inf, rectangular lam 2, tau 4, off-kink collocation: C_L",
                         round(cl, 6), None, None, None, f"{time.time() - t0:.0f}s"))
    # VLM half-wing at 24 x 60 and 36 x 80 panels, extrapolated linearly in 1/n_lateral
    v24 = planing_lift_hf(np.linspace(0, 0.5, 25), np.full(24, 2.0), math.tan(math.radians(4.0)), nc=60,
                          spacing="cos")["CL"]
    v36 = planing_lift_hf(np.linspace(0, 0.5, 37), np.full(36, 2.0), math.tan(math.radians(4.0)), nc=80,
                          spacing="cos")["CL"]
    vl = v36 - (v24 - v36) / (1 / 24 - 1 / 36) / 36
    if len(cls) == 3:
        e1, e2 = cls[1] - cls[0], cls[2] - cls[1]
        p = math.log(abs(e1 / e2)) / math.log(2) if e1 * e2 > 0 else float("nan")
        if not (0.5 <= p <= 3.0):
            p = 1.0
    else:
        e2, p = cls[1] - cls[0], 1.0
    ext = cls[-1] + e2 / (2 ** p - 1)
    d = abs(ext / vl - 1)
    rows.append(_row("PL1M", "gravity-free limit: extrapolated off-kink C_L (observed order, first order if outside "
                     "[0.5, 3] or two levels) vs half-wing VLM extrapolated in 1/n (24 x 60, 36 x 80)",
                     [round(ext, 6), round(p, 3)], round(vl, 6), TOL_PL1M, d <= TOL_PL1M,
                     f"rel diff {ext / vl - 1:+.4f}; VLM {v24:.6f}, {v36:.6f}; finest off-kink {cls[-1]:.6f}"))
    # production-collocation bias at C_V 2-6: production (node collocation) vs the mean of node and off-kink
    # collocation at PL1B_REF_NS lateral intervals (same h)
    worst = 0.0
    for cv in ((4.0,) if quick else (2.0, 4.0, 6.0)):
        a = planform_lift_rect(cv, PROD["h"], PROD["ns"], mode="node")
        r1 = planform_lift_rect(cv, PROD["h"], PL1B_REF_NS, mode="node")
        r2 = planform_lift_rect(cv, PROD["h"], PL1B_REF_NS, mode="mid")
        ref = 0.5 * (r1 + r2)
        worst = max(worst, abs(a / ref - 1))
        rows.append(_row(f"PL1B.Cv{cv:g}", "rectangular planform C_L: production vs reference (node, mid at 64 intervals)",
                         [round(a, 6), round(r1, 6), round(r2, 6)], round(ref, 6), None, None,
                         f"rel diff {a / ref - 1:+.4f}"))
    _STATE["kink_bias"] = worst
    return rows


# ------------------------------------------------------------------ convergence, slender limit
def pc1(quick):
    rows, res = [], []
    series = GRID_SERIES[:2] if quick else GRID_SERIES
    for h, ns in series:
        t0 = time.time()
        S = PlaningSprayRootFast(REF["Cv"], REF["beta"], h=h, ns=ns, cluster=PROD["cluster"], lam_max=6.0,
                                 lateral=PROD["lateral"])
        o = nl.solve_lambda(S, math.radians(REF["tau"]), REF["lam"])
        res.append((h, ns, o["CL"], o["lp"] / o["lam"], o["CL_lin"], o["VU"], o["converged"]))
        rows.append(_row(f"PC1.h{round(1 / h)}ns{ns}", "REF composite: C_L, l_p/(lam b), C_L linear, V/U, converged",
                         [round(o["CL"], 6), round(o["lp"] / o["lam"], 5), round(o["CL_lin"], 6), round(o["VU"], 5),
                          o["converged"]], None, None, None, f"{time.time() - t0:.0f}s"))
    a, b = res[-2], res[-1]
    dcl = abs(b[2] / a[2] - 1)
    prod = next((r for r in res if r[0] == PROD["h"] and r[1] == PROD["ns"]), a)
    un = max(abs(res[-1][2] / prod[2] - 1), _STATE.get("kink_bias", 0.0))
    _STATE["U_num"] = un
    rows.append(_row("PC1", "composite C_L change between the two finest grids", float(dcl), 0.0, TOL_PC1,
                     dcl <= TOL_PC1 and a[6] and b[6]))
    rows.append(_row("PC1u", "U_num = max(|C_L(finest)/C_L(production) - 1|, kink bias PL1B)", float(un), 0.0, TOL_UNUM,
                     un <= TOL_UNUM))
    return rows


def pl2(quick):
    rows = []
    S = solver(10.0, REF["beta"], l_mu=50.0)
    for lam in ((3.0,) if quick else (3.0, 4.0)):
        st = S.solve_lambda(math.radians(REF["tau"]), lam)
        dk = math.tan(math.radians(REF["beta"])) / (math.pi * math.tan(math.radians(REF["tau"])))
        d = abs((st["LK"] - st["LC"]) / dk - 1)
        rows.append(_row(f"PL2.lam{lam:g}", "slender limit (L_K - L_C)/b at C_V 10 vs tan(beta)/(pi tan(tau)); V/U vs tan(tau)",
                         [round(st["LK"] - st["LC"], 4), round(nl.inner_velocity(REF["beta"], st["LK"], st["LC"]) /
                                                                math.tan(math.radians(REF["tau"])), 4)],
                         round(dk, 4), TOL_PL2, (d <= TOL_PL2) and st["converged"]))
    return rows


# ------------------------------------------------------------------ validation
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
    rows, recs = [], []
    for (b, cv, tau, lam) in v_conditions(quick):
        S = solver(cv, b)
        st = S.solve_lambda(math.radians(tau), lam)
        sv = savitsky(tau, lam, cv, b)
        o = nl.composite(S, st)
        rec = dict(beta=b, Cv=cv, tau=tau, lam=lam, CL=o["CL"], CL_lin=st["CL"], CL_sav=sv["CL"], CL0_sav=sv["CL0"],
                   cp=o["lp"] / st["lam"], cp_sav=sv["cp"], VU=o["VU"], dCL=2 * o["dlift"], CL_NL2D=2 * o["L_NL2D"],
                   CL_W2D=2 * o["L_W2D"], converged=st["converged"])
        for name, kw in VARIANTS:
            rec["CL_" + name] = nl.composite(S, st, **kw)["CL"]
        recs.append(rec)
        print(f"   V b{b:g} Cv{cv:g} tau{tau:g} lam{lam:g}: CL {rec['CL']:.5f} (lin {rec['CL_lin']:.5f}, Sav "
              f"{rec['CL_sav']:.5f}; ratio {rec['CL'] / rec['CL_sav']:.3f}, lin {rec['CL_lin'] / rec['CL_sav']:.3f}) "
              f"2D NL {rec['CL_NL2D']:.4f} W {rec['CL_W2D']:.4f} V/U/tan {rec['VU'] / math.tan(math.radians(tau)):.3f} "
              f"cp {rec['cp']:.3f} ({rec['cp_sav']:.3f}) conv {rec['converged']}", flush=True)
    un = _STATE.get("U_num", 0.0)

    def frac(sel, tol, key="CL"):
        s = [r for r in recs if sel(r)]
        ok = [r for r in s if r["converged"] and abs(r[key] / r["CL_sav"] - 1) <= tol]
        return (len(ok) / len(s), len(s)) if s else (float("nan"), 0)
    f, n = frac(lambda r: r["beta"] == 20.0, TOL_VL)
    rows.append(_row("VL.beta20", f"fraction of beta-20 conditions with composite C_L within +-{TOL_VL:.0%} of Savitsky",
                     f, FRAC_PASS, TOL_VL, f >= FRAC_PASS, f"n={n}"))
    f, n = frac(lambda r: r["beta"] == 20.0, TOL_VL + un)
    rows.append(_row("VLR.beta20", "robust fraction: within +-(10 % + U_num)", f, FRAC_PASS, round(TOL_VL + un, 4),
                     f >= FRAC_PASS, f"n={n}, U_num={un:.4f}"))
    for b in V_BETA:
        if b == 20.0:
            continue
        f, n = frac(lambda r, b=b: r["beta"] == b, TOL_VL)
        rows.append(_row(f"VL.beta{b:g}", "fraction within +-10 % (information)", f, FRAC_PASS, TOL_VL, None, f"n={n}"))
    for b in sorted(set(r["beta"] for r in recs)):
        sel = [r for r in recs if r["beta"] == b]
        for key, lab in (("CL", "composite"), ("CL_lin", "B7 linear")) + tuple(("CL_" + v, v) for v, _ in VARIANTS):
            rr = [r[key] / r["CL_sav"] for r in sel]
            rows.append(_row(f"VLr.beta{b:g}.{lab}", "C_L / C_L,Savitsky: min, median, max",
                             [round(min(rr), 3), round(float(np.median(rr)), 3), round(max(rr), 3)], None, None, None))
        for cv in V_CV:
            rr = [r["CL"] / r["CL_sav"] for r in sel if r["Cv"] == cv]
            if rr:
                rows.append(_row(f"VLr.beta{b:g}.Cv{cv:g}", "composite C_L / Savitsky: median", round(float(np.median(rr)), 3),
                                 None, None, None))
    cps = [r for r in recs if r["converged"]]
    fc = sum(1 for r in cps if abs(r["cp"] - r["cp_sav"]) <= 0.05) / max(len(recs), 1)
    rows.append(_row("VCP", "fraction with l_p/(lam b) within 0.05 of Savitsky (information)", fc, FRAC_PASS, 0.05, None))
    return rows, recs


def savitsky_equilibrium(Cv, CD, lcg, beta):
    from scipy.optimize import brentq
    lam = brentq(lambda l: l * savitsky(4.0, l, Cv, beta)["cp"] - lcg, 0.2, 10.0)
    CLb = CD / (0.5 * Cv ** 2)
    tau = brentq(lambda t: savitsky(t, lam, Cv, beta)["CL"] - CLb, 0.2, 40.0)
    return tau, lam


def v_equilibrium(quick):
    rows, recs = [], []
    beta = REF["beta"]
    for (cv, cd, lcg) in (QUICK_EQ if quick else EQ_CASES):
        tau_s, lam_s = savitsky_equilibrium(cv, cd, lcg, beta)
        inside = (2.0 <= tau_s <= 15.0) and (lam_s <= 4.0)
        S = solver(cv, beta)
        lift = 0.5 * cd / (0.5 * cv ** 2)
        t0 = time.time()
        try:
            o = nl.equilibrium(S, lift, lcg, tau0=math.radians(max(tau_s, 2.0)), lam0=lam_s)
            tau_m, lam_m, ok = math.degrees(o["tau"]), o["lam"], bool(o.get("eq_ok")) and o["converged"]
        except Exception as e:                       # noqa: BLE001
            print("   EQ failed", repr(e)[:200])
            tau_m, lam_m, ok = float("nan"), float("nan"), False
        rec = dict(Cv=cv, CD=cd, lcg=lcg, tau=tau_m, lam=lam_m, tau_sav=tau_s, lam_sav=lam_s, inside=inside, solved=ok,
                   t=time.time() - t0)
        recs.append(rec)
        print(f"   EQ Cv{cv:g} CD{cd:g} lcg{lcg:g}: tau {tau_m:.3f} (Sav {tau_s:.3f}) lam {lam_m:.3f} (Sav {lam_s:.3f}) "
              f"inside {inside} solved {ok} {rec['t']:.0f}s", flush=True)
    use = [r for r in recs if r["inside"]]
    f = sum(1 for r in use if r["solved"] and abs(r["tau"] - r["tau_sav"]) <= TOL_VEQ_TAU) / max(len(use), 1)
    fl = sum(1 for r in use if r["solved"] and abs(r["lam"] / r["lam_sav"] - 1) <= TOL_VEQ_LAM) / max(len(use), 1)
    rows.append(_row("VEQ", "equilibrium trim within +-1 deg of Savitsky (beta 20, pressure moment only)", f, FRAC_PASS,
                     TOL_VEQ_TAU, f >= FRAC_PASS, f"n={len(use)}"))
    rows.append(_row("VEQl", "equilibrium mean wetted length within 15 % (information)", fl, FRAC_PASS, TOL_VEQ_LAM, None,
                     f"n={len(use)}"))
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
            out["b8:" + head.strip()] = None
            continue
        i = hits[0]
        j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ") or lines[k].startswith("### ")),
                 len(lines))
        out["b8:" + head.strip()] = hashlib.sha256(("\n".join(lines[i:j]).strip() + "\n").encode("utf-8")).hexdigest()
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
    hashes["b8:constants"] = _consts()
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
               and e["hashes"].get("b8:constants") == consts]
    print(f"=== rankine_planing_b8 tier={tier} start {start}; code registered: {reg_code[-1] if reg_code else 'NO'}; "
          f"14.2/14.3 + constants registered: {reg_sec[-1] if reg_sec else 'NO'}", flush=True)
    rows, extra = [], {}
    for name, fn in (("W2", w2_checks), ("PL1M", lambda: pl1m(quick)), ("PC1", lambda: pc1(quick)),
                     ("PL2", lambda: pl2(quick))):
        t = time.time()
        try:
            rows += fn()
        except Exception as e:                       # noqa: BLE001
            rows.append(_row(name, "job raised", repr(e)[:300], None, None, False))
        print(f"--- {name} {time.time() - t:.0f}s", flush=True)
    for name, fn in (("V", v_savitsky), ("VEQ", v_equilibrium)):
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
    adequate_ids = ("W2A", "W2B", "W2S", "C1", "PL1M", "PC1", "PC1u")
    num_rows = [r for r in binding if r["id"] in adequate_ids or r["id"].startswith("PL2.")]
    adequate = bool(num_rows) and all(r["ok"] for r in num_rows)
    by = {r["id"]: r for r in binding}
    go = adequate and by.get("VL.beta20", {}).get("ok") and by.get("VEQ", {}).get("ok")
    nogo = adequate and ((not by.get("VLR.beta20", {}).get("ok", True)) or (not by.get("VEQ", {}).get("ok", True)))
    verdict = "GO" if go else ("NO-GO" if nogo else "INCONCLUSIVE")
    if quick:
        verdict = "quick tier (no verdict): " + verdict
    end = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    code1 = _code_hashes()
    print(f"=== {npass}/{len(binding)} binding rows pass; numerics adequate {adequate}; VERDICT {verdict}; wall {wall:.0f}s;"
          f" code changed during run: {code1 != code0}", flush=True)
    json.dump(dict(tier=tier, start=start, end=end, wall=wall, rows=rows, extra=extra, verdict=verdict,
                   numerics_adequate=adequate, code_sha256_start=code0, code_registered=reg_code[-1] if reg_code else None,
                   reg_14=reg_sec[-1] if reg_sec else None, code_changed_during_run=code1 != code0),
              open(os.path.join(OUT, f"planing_b8_{tier}.json"), "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("quick", "full"))
    ap.add_argument("--register")
    a = ap.parse_args()
    if a.register:
        register(a.register)
    elif a.tier:
        run(a.tier)
