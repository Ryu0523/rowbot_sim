#!/usr/bin/env python3
"""
Step P1c, track W, round 4 (learn/meta/ANYSPEED_PLAN_2026-10-05.md, '###
wagner 低速：复查后的修正'): the checks registered there before they ran,
after the review of round 3. Helpers, stamps and report format of
studies/anyspeed_p1b_check.py; the round-3 runs of studies/anyspeed_p1c_w3.

  w19p      W19' -- the Jacobi-form power budget of S1 (b)'s run (captive,
            calm, root + 5 cm, dt 0.02) at 1.0 / 1.5 kn, windows 2-30 s and
            30-90 s (2-10, 10-20, 20-30 s recorded), the budget built on the
            plant's own mass field (kb.ma x the transom taper ctr, the
            hydrostatic potential and cross-flow with ctr -- ctr is 1 on
            both as1 hulls --, the boundary jumps from the station masses
            interpolated linearly to the boundary, as round 3; the station
            values of the strip that holds it, strip_b=True, count the whole
            jump at a waterline crossing and closed worse):
              dH/dt = P_T + sum_pairs P_b[aft|fwd] + sum (f - f_att) g dx
                      + P_cf + P_hold + P_fric + P_air + P_Mh,
            P_hold the captive holding force's pitch moment (external input,
            not tested), the state boundaries split by the pair of states
            (aft|fwd). A group is an input when its mean exceeds both 1 % of
            the gross and twice the |residual|. (a) memory off: no input but
            the gate-closed widening planes; (b) the plant: no input but the
            X|X boundaries (the registered residual of assumption (b)), and
            those within the registered bound (part rbound)
  rbound    the registered bound on the X|X residual: (i) the internal
            budget (all groups but P_hold) negative in every window of the
            S1 (b) runs and of the W18 runs (0.5-4 kn, 600 s); (ii) X|X at
            most 25 % of the removed input (memory off, gate-closed widening
            planes) at matched amplitude, S1 (b) 1.0 / 1.5 kn (2-10, 2-30 s)
            and W4-type head seas (6, 12, 28 kn, last 40 s)
  w1d       the X|X term of a turning point moving at a known speed through
            a widening taper (synthetic field): the discrete -u sum 1/2 [m]
            V^2 against the derived 1/2 int c_turn k V^2 dx, k = xt' / (xt' +
            u) (positive when the turning point moves forward)
  bracket   assumption (b) against the attached alternative (a) (memory off
            and the E/X flux V Dm/Dt whenever Dm/Dt > 0): S1 (b) at 1.0 /
            1.5 kn and W4-type head seas at 6, 12, 28 kn, on P1a and on the
            current as1 preset, beside the plant without wagner_entry
  w6b       planing_b (keel rise: an X plane's mass enters the static
            captive force): S1 (g)'s census and winding scan, memory on vs
            off, at W6's 38 speeds
  s1fb      S1 (f) on planing_b (basin, following seas Hs x1 / x1.5),
            memory on vs off
  calmid    the current as1 preset + wagner_entry against P1a +
            wagner_entry in calm water (fk_exact takes P1a's lines there,
            fast_path is implementation only): S1 (b)'s runs at 1.0 / 1.5
            kn and the census at 3 speeds, bit for bit
  w4as1     W4 (registered criterion unchanged) on the current as1 preset
            + wagner_entry: 12 and 28 kn, dt 0.02 / 0.01 / 0.005
  w5y       W5 (c) and W5cap (c) replayed from the first recorded row with
            a finite turn half-beam Y, bit for bit; the record without 'y'
            (an older record) recorded as a stated difference
  e1        E1 on the current preset: as1, as1 + wagner_entry (P0's
            interleaved method)

    .venv/Scripts/python.exe -m studies.anyspeed_p1c_w4 --parts w19p,w1d
    .venv/Scripts/python.exe -m studies.anyspeed_p1c_w4 --jobs w19p,...
"""
import argparse
import copy
import hashlib
import math
import os
import subprocess
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np                      # noqa: E402

from studies import anyspeed_p1b_check as C     # noqa: E402
from studies import anyspeed_p1b_fix as FX      # noqa: E402

OUT = os.path.join(HERE, "_cache", "anyspeed", "p1c", "w4")
KN = C.KN
G = C.G
P1A = FX.P1A
W = FX.W
out, check, dump, flush = C.out, C.check, C.dump, C.flush
PAIRS = ("E|X", "X|E", "X|F", "F|X", "E|F", "F|E", "X|X", "E|E", "F|F")
_NM = {0: "E", 1: "X", 2: "F"}


def as1_preset():
    from sim import planing_anyspeed as PA
    return tuple(PA.PRESETS["as1"])


def as1w():
    """The current as1 preset with wagner_entry (resolved order)."""
    from sim import planing_anyspeed as PA
    return PA.resolve("as1", as1_preset() + ("wagner_entry",))


# ================================================================= budget
def _ctr_arr(p, u, n):
    c = p._ctr(u)
    return np.broadcast_to(np.asarray(c, float), (n,)).copy()


def terms4(p, s, ds, aux, u, ctr_on=True, strip_b=False):
    """The budget quantities of one derivative (a step's first) of a captive
    calm-water W plant, as powers (W; positive = into H): the groups of the
    round-3 identity on the plant's own mass field. ctr_on: the masses, the
    hydrostatic potential and the cross-flow with the transom taper ctr (as
    the plant applies them); strip_b: a boundary's mass jump from the
    station values of the strip that holds it (the plant's strip fractions)
    instead of the linear interpolation of round 3."""
    from sim import planing_anyspeed as PA
    from sim.planing_vessel import SP
    kb = aux["kb"]
    x, dx, rho = p.x, p.dx, p.rho
    n = x.size
    d, V, Vg, ur = aux["d"], kb.V, kb.Vg, kb.ur
    w, q = s[8], s[10]
    tau = p.kp + SP * (p.Th_rest + s[4])
    A, c, maZ, dma, _ = p._section(d)
    ctr = _ctr_arr(p, u, n) if ctr_on else np.ones(n)
    wet = d > 0.0
    f = np.asarray(kb.f, float)
    fX = np.zeros(n) if kb.fX is None else np.asarray(kb.fX, float)
    mem = kb.esy is not None
    M = kb.ma * ctr
    km = p.k_a * 0.5 * np.pi * rho
    kc = 0.5 * np.pi / p.tanb
    m0 = np.zeros(n)
    dm0 = np.zeros(n)
    dm0y = np.zeros(n)
    j = np.nonzero(wet)[0]
    if j.size:
        a0, a1, a2 = p._tab().vertical(d[j], p.yc[j])
        m0[j], dm0[j], dm0y[j] = rho * a0, rho * a1, rho * a2
    # the E/X rates exactly as _flux_b computes them (calm: slope -tau; the
    # Zarnick rate of the strips without a located chine carries ctr, as the
    # plant passes dma x ctr; the located strips' rates do not)
    r_d = dma * ctr * Vg
    r_c = np.zeros(n)
    act, ycap, iunc = p._substrip(d, -tau)
    if act.size:
        r_d[act] = Vg[act] * (2.0 * km * kc * kc) * iunc / dx
        r_c[act] = ur[act] * km * ycap / dx
    rate = r_d + r_c
    gEX = ((V > 0.0) & (rate > 0.0)).astype(float)
    conv0 = -ur * dm0y * p._dyc_dx()
    gF = ((V > 0.0) & (conv0 > 0.0)).astype(float)
    g = w + SP * x * q
    wE = np.maximum(1.0 - f - fX, 0.0)
    # X planes with the memory: m = km min(kc d, yc, Y)^2; their attached
    # force has 1/2 dm/dd V^2 where kc d binds (none where Y binds); the
    # plant gives them no flux of their own (sX), but the strip's E/X flux
    # when the strip-centre gate is open (Xfl)
    dX = np.zeros(n)
    if mem and fX.any():
        b, st = kb.es
        y = kb.esy
        e = p._es_geom().edges
        for kk in np.nonzero(st[1:-1] == PA.ES_X)[0]:
            a_, b_, Y = b[kk], b[kk + 1], y[kk + 1]
            for jj in range(max(int(np.searchsorted(e, a_, "right")) - 1, 0),
                            min(int(np.searchsorted(e, b_, "left")), n)):
                ln = max(min(b_, e[jj + 1]) - max(a_, e[jj]), 0.0)
                if ln > 0.0 and kc * max(d[jj], 0.0) <= min(p.yc[jj], Y):
                    dX[jj] += ln / (e[jj + 1] - e[jj])
    sX = -km * kc * kc * np.maximum(d, 0.0) * V * V * dX * ctr
    Xfl = (fX * gEX * V * rate) if mem else np.zeros(n)
    wEX = wE + (0.0 if mem else 1.0) * fX
    dEo = wEX * gEX * 0.5 * V * r_d
    dEc = wEX * (1.0 - gEX) * (-0.5) * V * r_d
    cEX = wEX * (gEX - 1.0) * V * r_c
    # F planes: the plant's (Vg - V/2) V dm0 + gate V conv0, unscaled by
    # ctr, against the attached force of their mass ctr m0
    cF = f * (gF - 1.0) * V * conv0
    ctrF = f * (1.0 - ctr) * (0.5 * V * V * dm0 + V * conv0)
    cf = 0.5 * rho * p.c_dc * 2.0 * c * V * np.abs(V) * ctr
    ab = p._a_b(u)
    qsec = PA._vsec_q(d, p.yc, p.tanb)[0]
    zw = float(np.mean(p.zk[wet])) if wet.any() else float(p.zk[0])
    _, _, zn = p.thrusters[0]
    # state boundaries, split by the pair (aft|fwd)
    b, st = kb.es
    y = kb.esy if mem else None
    e = p._es_geom().edges
    pb = {k: 0.0 for k in PAIRS}

    def m_strip(state, jj, Y):
        if state == PA.ES_F:
            return m0[jj] * ctr[jj]
        if state == PA.ES_X and y is not None:
            cb = min(kc * max(d[jj], 0.0), p.yc[jj], Y)
            return km * cb * cb * ctr[jj]
        return maZ[jj] * ctr[jj]

    def m_interp(state, xb, Y):
        mz = float(np.interp(xb, x, maZ * ctr))
        if state == PA.ES_F:
            return float(np.interp(xb, x, m0 * ctr))
        if state == PA.ES_X and y is not None:
            cb = min(kc * max(float(np.interp(xb, x, d)), 0.0),
                     float(np.interp(xb, x, p.yc)), Y)
            return km * cb * cb * float(np.interp(xb, x, ctr))
        return mz
    nb = 0
    for i, bb in enumerate(b):
        sa, sf = int(st[i]), int(st[i + 1])
        ya = np.inf if y is None else y[i]
        yf = np.inf if y is None else y[i + 1]
        if sa == sf and ya == yf:
            continue
        if strip_b:
            jj = int(np.searchsorted(e, bb, side="right")) - 1
            if jj < 0 or jj >= n or d[jj] <= 0.0:
                continue
            de = 0.5 * (m_strip(sf, jj, yf) - m_strip(sa, jj, ya)) * V[jj] ** 2
        else:
            if bb <= x[0] or bb >= x[-1] or np.interp(bb, x, d) <= 0.0:
                continue
            Vb = float(np.interp(bb, x, V))
            de = 0.5 * (m_interp(sf, bb, yf) - m_interp(sa, bb, ya)) * Vb * Vb
        pb[f"{_NM[sa]}|{_NM[sf]}"] += -u * de
        nb += 1

    def S(z):
        return float(np.sum(z) * dx)
    # the bow region's fluctuating relative normal velocity (matched
    # amplitude): stations in the forward quarter of the hull
    fwd = x >= x[-1] - 0.25 * (x[-1] - x[0])
    r = dict(
        z_=float(s[2]), th_=float(s[4]), w_=float(w), q_=float(q),
        KE=0.5 * p.m * w * w + 0.5 * p.Iy * q * q,
        U=p.m * G * float(s[2]) + ab * rho * G * S(qsec * ctr),
        Tf=0.5 * S(M * V * V), Gm=S(M * V * tau),
        P_dEo=S(dEo * g), P_dEc=S(dEc * g), P_cEX=S(cEX * g),
        P_Xfl=S(Xfl * g), P_cF=S(cF * g), P_ctrF=S(ctrF * g),
        P_X=S(sX * g), P_T=u * M[0] * V[0] * g[0],
        P_cf=S(cf * g), P_hold=float(q * zn * aux["fx"]),
        P_fric=float(-q * zw * aux["Df"]),
        P_air=float(-q * p.num.z_air * aux["Da"]),
        P_Mh=float(q * aux["Mh"]),
        Vf=V[fwd].copy(), nb=nb,
        Fsh=float(f[wet].mean()) if wet.any() else 0.0,
        Xsh=float(fX[wet].mean()) if wet.any() else 0.0)
    for k in PAIRS:
        r["Pb_" + k] = pb[k]
    return r


GROUPS = (("P_dEo", "P_dEc", "P_cEX", "P_Xfl", "P_cF", "P_ctrF", "P_X",
           "P_T") + tuple("Pb_" + k for k in PAIRS)
          + ("P_cf", "P_hold", "P_fric", "P_air", "P_Mh"))
EXTERNAL = ("P_hold",)
NAMES = dict(
    P_dEo="E planes entering, Wagner d-part (+1/2 dm/dd V^2)",
    P_dEc="E/X d-part, gate closed (-1/2 dm/dd V^2)",
    P_cEX="E/X widening, gate closed (-V c)",
    P_Xfl="X planes given the strip's open-gate E/X flux",
    P_cF="F widening, convective gate closed (-V c0)",
    P_ctrF="F Lagrangian terms unscaled by the transom taper",
    P_X="X with the memory: d-part (-km kc^2 d V^2)",
    P_T="transom shedding u M_T V_T g_T",
    P_cf="cross-flow drag",
    P_hold="captive holding force's pitch moment (EXTERNAL)",
    P_fric="friction's pitch moment", P_air="air drag's pitch moment",
    P_Mh="longitudinal hydrostatic moment")
for _k in PAIRS:
    NAMES["Pb_" + _k] = f"state boundaries {_k} (aft|fwd), -u 1/2 [m] V^2"


def attached(p):
    """The attached alternative (a) on plant p (never in the plant): the X
    memory off and the E/X planes' momentum flux V Dm/Dt counted whenever
    Dm/Dt > 0, whatever the sign of V (round 2's diagnostic Xsuction,
    studies/anyspeed_p1c_w._xsuction; the F planes as in the plant). Apply
    after any steady-state solve: an instance attribute is copied into the
    steady-state copies."""
    from studies import anyspeed_p1c_w as PW
    p.es_xmem = False
    PW._xsuction(p)


def run_b4(kn, T1, dz=0.05, keep=True, xmem=True, variant=None, sw=None,
           dt=0.02, opts=None, hull="scarab195@as1"):
    """S1 (b)'s run (captive, calm, Newton root + dz) of a W plant; the
    budget terms of every step's first derivative (keep). variant
    'attached': (a). Returns the steady terms, the rows, the 10-s
    envelopes."""
    from hydro import hulls
    from sim import planing_anyspeed as PA
    from studies import anyspeed_s1_check as S1
    opts = opts or {}
    u = kn * KN
    sw = W if sw is None else sw
    p = PA.PlaningVesselAS(hulls.get(hull), C.calm(), dt=dt, captive_u=u,
                           test_switches=sw)
    if not xmem:
        p.es_xmem = False
    sol, res = S1.newton(p, u, np.array(p.running_attitude(u)))
    z0, th0 = sol
    if variant == "attached":
        attached(p)
    cls = type(p)
    sroot = S1.state(u, z0, th0)
    steady = None
    if keep:
        ds_r, aux_r = cls.deriv(p, sroot, 0.0)
        steady = terms4(p, sroot, ds_r, aux_r, u, **opts)
    cap = {}

    def spy(s_, t_):
        r_ = cls.deriv(p, s_, t_)
        if "aux" not in cap:
            cap["aux"], cap["ds"], cap["s"] = (r_[1], r_[0].copy(),
                                               s_.copy())
        return r_
    p.deriv = spy
    s = S1.state(u, z0 + dz, th0)
    p.initial_state(0.0)
    n = int(round(T1 / dt))
    rows, zz = [], [dz]
    try:
        for i in range(n):
            cap.clear()
            s = p.step(s, i * dt, 0.0, 0.0)
            s[6] = u
            zz.append(s[2] - z0)
            if keep:
                rows.append(terms4(p, cap["s"], cap["ds"], cap["aux"], u,
                                   **opts))
    finally:
        del p.deriv
    k = int(round(10 / dt))
    env = [float(np.abs(np.array(zz[j_ * k:(j_ + 1) * k])).max())
           for j_ in range(len(zz) // k)]
    return dict(steady=steady, rows=rows, env=env, resid=float(res),
                z0=float(z0), th0=float(th0), dt=dt)


def window_budget(r, kn, T0, T1):
    """Window means minus the steady state's; dH/dt from H at the window's
    ends; the matched-amplitude scale A2 (mean square of the bow region's
    relative normal velocity about its steady value)."""
    u, dt = kn * KN, r["dt"]
    rows, st = r["rows"], r["steady"]
    n0, n1 = int(round(T0 / dt)), int(round(T1 / dt))

    def H(i):
        x_ = rows[i]
        return x_["KE"] + x_["U"] + x_["Tf"] - u * x_["Gm"]
    means = {k: float(np.mean([rows[i][k] for i in range(n0, n1)]) - st[k])
             for k in GROUPS}
    dH = (H(n1) - H(n0)) / (T1 - T0)
    tot = sum(means.values())
    gross = sum(abs(v) for v in means.values())
    Vf = np.array([rows[i]["Vf"] for i in range(n0, n1)])
    A2 = float(np.mean((Vf - Vf.mean(axis=0)) ** 2))
    return dict(T0=T0, T1=T1, means=means, dH=float(dH), tot=float(tot),
                gross=float(gross), resid=float(dH - tot), A2=A2,
                Fsh=float(np.mean([rows[i]["Fsh"] for i in range(n0, n1)])),
                Xsh=float(np.mean([rows[i]["Xsh"] for i in range(n0, n1)])))


def inputs_of(wb, allowed=()):
    """The groups counted as inputs by W19' (mean above 1 % of the gross
    and above twice the |residual|), external groups excluded."""
    thr = max(0.01 * wb["gross"], 2.0 * abs(wb["resid"]))
    return [k for k in GROUPS if k not in EXTERNAL and k not in allowed
            and wb["means"][k] > thr]


def added_masses(p, aux):
    """(A33, A55) of a derivative: the plant's section added mass (the flow
    states' mixture with wagner_entry, Zarnick's without), about the CG."""
    if "kb" in aux:
        ma = aux["kb"].ma
    else:
        ma = p._section(aux["d"])[2]
    return float(ma.sum() * p.dx), float((ma * p.x * p.x).sum() * p.dx)


def terms_free(p, s, aux, slope, u):
    """Work rates on the hull (heave and pitch) of a free run in waves (not
    a closed wave budget): the gate-closed widening E/X planes (P_cEX;
    with the memory, E only), the X planes given the open-gate E/X flux
    (P_Xfl), the F convective gate (P_cF), the state boundaries by pair
    (interpolated station masses, the relative normal velocity with the
    waves), the bow region's relative normal velocity (matched amplitude),
    the added masses and the state shares."""
    from sim import planing_anyspeed as PA
    from sim.planing_vessel import SP
    kb = aux["kb"]
    x, dx, rho = p.x, p.dx, p.rho
    n = x.size
    d, V, Vg, ur = aux["d"], kb.V, kb.Vg, kb.ur
    g = s[8] + SP * x * s[10]
    A, c, maZ, dma, _ = p._section(d)
    km = p.k_a * 0.5 * np.pi * rho
    kc = 0.5 * np.pi / p.tanb
    r_d = dma * Vg
    r_c = np.zeros(n)
    act, ycap, iunc = p._substrip(d, slope)
    if act.size:
        r_d[act] = Vg[act] * (2.0 * km * kc * kc) * iunc / dx
        r_c[act] = ur[act] * km * ycap / dx
    rate = r_d + r_c
    gEX = ((V > 0.0) & (rate > 0.0)).astype(float)
    f = np.asarray(kb.f, float)
    fX = np.zeros(n) if kb.fX is None else np.asarray(kb.fX, float)
    mem = kb.esy is not None
    wEX = np.maximum(1.0 - f - fX, 0.0) + (0.0 if mem else 1.0) * fX
    conv0 = -ur * kb.dm0y * p._dyc_dx()
    gF = ((V > 0.0) & (conv0 > 0.0)).astype(float)
    wet = d > 0.0
    m0 = np.zeros(n)
    j = np.nonzero(wet)[0]
    if j.size:
        m0[j] = rho * p._tab().vertical(d[j], p.yc[j])[0]
    b, st = kb.es
    y = kb.esy if mem else None
    pb = {k: 0.0 for k in PAIRS}

    def m_of(state, xb, Y):
        if state == PA.ES_F:
            return float(np.interp(xb, x, m0))
        if state == PA.ES_X and y is not None:
            cb = min(kc * max(float(np.interp(xb, x, d)), 0.0),
                     float(np.interp(xb, x, p.yc)), Y)
            return km * cb * cb
        return float(np.interp(xb, x, maZ))
    for i, bb in enumerate(b):
        sa, sf = int(st[i]), int(st[i + 1])
        ya = np.inf if y is None else y[i]
        yf = np.inf if y is None else y[i + 1]
        if sa == sf and ya == yf:
            continue
        if bb <= x[0] or bb >= x[-1] or np.interp(bb, x, d) <= 0.0:
            continue
        Vb = float(np.interp(bb, x, V))
        pb[f"{_NM[sa]}|{_NM[sf]}"] += -u * 0.5 * (m_of(sf, bb, yf)
                                                 - m_of(sa, bb, ya)) * Vb * Vb
    fwd = x >= x[-1] - 0.25 * (x[-1] - x[0])
    A33, A55 = added_masses(p, aux)
    r = dict(P_cEX=float(np.sum(wEX * (gEX - 1.0) * V * r_c * g) * dx),
             P_Xfl=(float(np.sum(fX * gEX * V * rate * g) * dx) if mem
                    else 0.0),
             P_cF=float(np.sum(f * (gF - 1.0) * V * conv0 * g) * dx),
             Vf=V[fwd].copy(), A33=A33, A55=A55,
             Fsh=float(f[wet].mean()) if wet.any() else 0.0,
             Xsh=float(fX[wet].mean()) if wet.any() else 0.0)
    for k in PAIRS:
        r["Pb_" + k] = pb[k]
    return r


def head_run(arg):
    """W4-type free run (regular head waves lam 2 L, a 0.12 m; constant
    thrust from initial_state; dt 0.02; 60 s, statistics over the last 40
    s) of scarab195 with switch set sw and variant ('b' the plant, 'off'
    the memory off, 'a' the attached alternative, 'noW' no wagner_entry);
    terms: the work-rate groups of every step's first derivative."""
    sw, variant, kn, terms = arg
    warnings.filterwarnings("ignore")
    from hydro import hulls
    from sim import planing_anyspeed as PA
    dt = 0.02
    p = PA.PlaningVesselAS(hulls.get(C.AS), FX._head_sea(hulls.get(C.AS)),
                           dt=dt, test_switches=sw)
    if variant in ("off", "a"):
        p.es_xmem = False
    s = p.initial_state(kn * KN)
    thr = s[12]
    if variant == "a":
        attached(p)
    cls = type(p)
    cap = {}
    orig_flux = p._flux_b

    def flux_spy(d_, kb_, dm_, slope_):
        if "slope" not in cap:
            cap["slope"] = np.array(slope_, float)
        return orig_flux(d_, kb_, dm_, slope_)

    def spy(s_, t_):
        r_ = cls.deriv(p, s_, t_)
        if "aux" not in cap:
            cap["aux"], cap["s"] = r_[1], s_.copy()
        return r_
    p.deriv = spy
    p._flux_b = flux_spy
    n = int(round(60.0 / dt))
    S, rows, am = [], [], []
    t = 0.0
    try:
        for i in range(n):
            cap.clear()
            s = p.step(s, t, thr, 0.0)
            t += dt
            S.append(s.copy())
            am.append(added_masses(p, cap["aux"]))
            if terms and "kb" in cap["aux"]:
                rows.append(terms_free(p, cap["s"], cap["aux"],
                                       cap["slope"], float(cap["s"][6])))
    finally:
        del p.deriv
        if variant == "a":
            # the attached variant's own instance _flux_b stays
            p._flux_b = orig_flux
        else:
            del p._flux_b
    S = np.array(S)
    k0 = len(S) // 3
    h2 = S[k0:]
    am = np.array(am)[k0:]
    res = dict(variant=variant, sw="as1" if "fk_exact" in sw else "P1a",
               kn=kn, u=float(h2[:, 6].mean() / KN),
               z=float(h2[:, 2].std()), th=float(np.degrees(h2[:, 4].std())),
               A33=float(am[:, 0].mean()), A55=float(am[:, 1].mean()),
               resets=int(getattr(p, "es_resets", 0)))
    if rows:
        R = rows[k0:]
        keys = [k for k in R[0] if k != "Vf"]
        res["means"] = {k: float(np.mean([r_[k] for r_ in R])) for k in keys}
        Vf = np.array([r_["Vf"] for r_ in R])
        res["A2"] = float(np.mean((Vf - Vf.mean(axis=0)) ** 2))
    return res


# ==================================================================== w19p
W19P_WINS = ((2.0, 10.0), (10.0, 20.0), (20.0, 30.0), (10.0, 30.0),
             (2.0, 30.0), (30.0, 90.0))
GATE_WINS = ("2-30", "30-90")
NET_WINS = ("2-10", "10-30", "30-90")
RATIO_WINS = ("2-10", "2-30")
R_BOUND = 0.25
NET_FLOOR = 1e-9


def w19p_job(arg):
    kn, xmem = arg
    warnings.filterwarnings("ignore")
    r = run_b4(kn, 90.1, xmem=xmem)
    return dict(kn=kn, xmem=xmem, env=r["env"],
                wins={f"{a:g}-{b:g}": window_budget(r, kn, a, b)
                      for a, b in W19P_WINS})


def _print_budget(wb, tag):
    net = wb["dH"]
    out(f"    [{tag}] dH/dt {wb['dH']:+.4f} W; sum of groups {wb['tot']:+.4f};"
        f" residual {wb['resid']:+.4f} = {100 * abs(wb['resid']) / max(wb['gross'], 1e-30):.2f} % "
        f"of the gross {wb['gross']:.4f}, "
        f"{100 * abs(wb['resid']) / max(abs(net), 1e-30):.0f} % of the net; "
        f"F share {wb['Fsh']:.3f}, X share {wb['Xsh']:.3f}")
    for k in GROUPS:
        v = wb["means"][k]
        if abs(v) > 5e-5 or not k.startswith("Pb_"):
            out(f"        {k:9s} {v:+9.4f}  {NAMES[k]}")


def part_w19p():
    from studies.parallel_runner import run_parallel
    out("W19': the Jacobi-form power budget of S1 (b)'s run (captive, calm, "
        "root + 5 cm, dt 0.02); window means minus the steady state's (W; "
        "positive = into H). Input = mean above max(1 % of the gross, 2 x "
        "|residual|); P_hold external, not tested")
    res = run_parallel(w19p_job, [(kn, xm) for xm in (False, True)
                                  for kn in (1.0, 1.5)],
                       gb_per_job=0.5, reserve_gb=3.0)
    by = {}
    ok_a = ok_b = True
    for r in res:
        if isinstance(r, Exception):
            out(f"    job failed: {r!r}")
            ok_a = ok_b = False
            continue
        by[(r["kn"], r["xmem"])] = r
        tag = "(b) the plant, X memory on" if r["xmem"] else \
            "(a) X memory off (round 2)"
        out(f"  {r['kn']:.1f} kn {tag}: 10-s envelope every 30 s "
            + " ".join(f"{a:.1e}" for a in r["env"][::3]))
        for wn, wb in r["wins"].items():
            _print_budget(wb, f"{wn} s")
        for wn in GATE_WINS:
            wb = r["wins"][wn]
            close = abs(wb["resid"]) <= 0.05 * wb["gross"]
            allowed = (("Pb_X|X", "P_cF") if r["xmem"]
                       else ("P_cEX", "P_cF"))
            ins = inputs_of(wb, allowed)
            out(f"    gate window {wn} s: closes (5 %) {close}; inputs other "
                f"than {allowed}: {ins}; threshold "
                f"{max(0.01 * wb['gross'], 2 * abs(wb['resid'])):.4f} W")
            if r["xmem"]:
                ok_b &= close and not ins
            else:
                ok_a &= close and not ins
    check("w19p", "W19' (a): memory off -- closes within 5 % and no input "
          "but the gate-closed widening planes (2-30, 30-90 s; 1.0, 1.5 kn)",
          ok_a)
    check("w19p", "W19' (b): the plant -- closes within 5 % and no input but"
          " the X|X boundaries (registered residual) and the F convective "
          "gate (registered residual) (2-30, 30-90 s; 1.0, 1.5 kn)", ok_b)
    # the registered bound, S1 (b) part: (i) the internal net (dH/dt less
    # the holding force's pitch work) not positive; (ii) the X|X residual
    # at most 25 % of the removed input at matched amplitude
    ok_i = ok_ii = True
    rows = []
    for kn in (1.0, 1.5):
        on, off = by.get((kn, True)), by.get((kn, False))
        if not on or not off:
            ok_i = ok_ii = False
            continue
        for wn in NET_WINS:
            wb = on["wins"][wn]
            net = wb["dH"] - wb["means"]["P_hold"]
            good = net < NET_FLOOR
            ok_i &= good
            out(f"    bound (i) {kn:.1f} kn {wn} s: internal net "
                f"{net:+.3e} W {'ok' if good else 'POSITIVE'}")
        for wn in RATIO_WINS:
            a_, b_ = on["wins"][wn], off["wins"][wn]
            xx, rem = a_["means"]["Pb_X|X"], b_["means"]["P_cEX"]
            R = ((xx / a_["A2"]) / (rem / b_["A2"])
                 if rem > 0.0 and a_["A2"] > 0 else float("nan"))
            good = R <= R_BOUND
            ok_ii &= bool(good)
            rows.append(dict(kn=kn, win=wn, XX=xx, removed=rem,
                             A2_on=a_["A2"], A2_off=b_["A2"], R=R))
            out(f"    bound (ii) {kn:.1f} kn {wn} s: X|X {xx:+.4f} W (A2 "
                f"{a_['A2']:.3e}), removed input {rem:+.4f} W (A2 "
                f"{b_['A2']:.3e}): ratio at matched amplitude {R:.3f} "
                f"(unscaled {xx / rem if rem else float('nan'):.3f}) "
                f"{'ok' if good else 'ABOVE 0.25'}")
    check("w19p", "bound (i), S1 (b): the internal net budget of the plant "
          "not positive in 2-10, 10-30, 30-90 s (1.0, 1.5 kn)", ok_i)
    check("w19p", "bound (ii), S1 (b): X|X at most 25 % of the removed "
          "input at matched amplitude (2-10, 2-30 s; 1.0, 1.5 kn)", ok_ii)
    dump("w19p.json", dict(runs=[r for r in res
                                 if not isinstance(r, Exception)],
                           ratios=rows))
    flush("w19p.txt")


# ================================================================== rbound
def w18b_job(arg):
    kn = arg
    warnings.filterwarnings("ignore")
    r = run_b4(kn, 600.1, xmem=True)
    wins = {f"{a:g}-{b:g}": window_budget(r, kn, a, b)
            for a, b in ((2.0, 30.0), (30.0, 90.0), (90.0, 600.0))}
    return dict(kn=kn, env=r["env"], wins=wins)


def part_rbound():
    from studies.parallel_runner import run_parallel
    out("Registered bound on the X|X residual of assumption (b): (i) W18's "
        "runs (0.5-4 kn, + 5 cm, 600 s), internal net (dH/dt less the "
        "holding force's pitch work) not positive in 2-30, 30-90, 90-600 s;"
        " (ii) W4-type head seas (P1a + wagner_entry, 6 / 12 / 28 kn, last "
        "40 s): X|X at most 25 % of the removed input (memory off) at "
        "matched amplitude")
    kns = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
    args_h = [(W, v, kn, True) for kn in (6.0, 12.0, 28.0)
              for v in ("b", "off")]
    res = run_parallel(_rbound_job, [("w18", kn) for kn in kns]
                       + [("head", a) for a in args_h],
                       gb_per_job=0.5, reserve_gb=3.0)
    ok_i = ok_ii = True
    w18 = [r for r in res[:len(kns)]]
    for r in w18:
        if isinstance(r, Exception):
            out(f"    job failed: {r!r}")
            ok_i = False
            continue
        for wn, wb in r["wins"].items():
            net = wb["dH"] - wb["means"]["P_hold"]
            good = net < NET_FLOOR
            ok_i &= good
            xx = wb["means"]["Pb_X|X"]
            out(f"    W18 {r['kn']:.1f} kn {wn:7s} s: internal net {net:+.3e}"
                f" W {'ok' if good else 'POSITIVE'}; X|X {xx:+.3e}, residual "
                f"{wb['resid']:+.2e} ({100 * abs(wb['resid']) / max(wb['gross'], 1e-30):.1f} % of "
                f"the gross)")
    heads = {}
    for r in res[len(kns):]:
        if isinstance(r, Exception):
            out(f"    job failed: {r!r}")
            ok_ii = False
            continue
        heads[(r["kn"], r["variant"])] = r
    ratios = []
    for kn in (6.0, 12.0, 28.0):
        on, off = heads.get((kn, "b")), heads.get((kn, "off"))
        if not on or not off:
            ok_ii = False
            continue
        for v, r in (("on", on), ("off", off)):
            m = r["means"]
            out(f"    head seas {kn:4.1f} kn memory {v:3s}: speed "
                f"{r['u']:.3f} kn, heave rms {1000 * r['z']:.1f} mm, pitch "
                f"rms {r['th']:.3f} deg; P_cEX {m['P_cEX']:+.3f}, P_Xfl "
                f"{m['P_Xfl']:+.3f}, P_cF {m['P_cF']:+.3f}; boundaries "
                + ", ".join(f"{k} {m['Pb_' + k]:+.2f}" for k in PAIRS
                            if abs(m['Pb_' + k]) > 5e-3)
                + f"; A2 {r['A2']:.3e}")
        xx, rem = on["means"]["Pb_X|X"], off["means"]["P_cEX"]
        R = ((xx / on["A2"]) / (rem / off["A2"]) if rem > 0.0
             else float("nan"))
        good = R <= R_BOUND or (xx <= 0.0)
        ok_ii &= bool(good)
        ratios.append(dict(kn=kn, XX=xx, removed=rem, A2_on=on["A2"],
                           A2_off=off["A2"], R=R))
        out(f"    head seas {kn:4.1f} kn: X|X {xx:+.3f} W, removed input "
            f"{rem:+.3f} W; ratio at matched amplitude {R:.3f} (unscaled "
            f"{xx / rem if rem else float('nan'):.3f}) "
            f"{'ok' if good else 'ABOVE 0.25'}")
    check("rbound", "bound (i), W18: the internal net budget not positive "
          "in 2-30, 30-90, 90-600 s (0.5-4 kn)", ok_i)
    check("rbound", "bound (ii), head seas: X|X at most 25 % of the removed "
          "input at matched amplitude (6, 12, 28 kn)", ok_ii)
    dump("rbound.json", dict(w18=[r for r in w18
                                  if not isinstance(r, Exception)],
                             heads=list(heads.values()), ratios=ratios))
    flush("rbound.txt")


def _rbound_job(arg):
    kind, a = arg
    if kind == "w18":
        return w18b_job(a)
    return head_run(a)


# ===================================================================== w1d
def part_w1d():
    from sim import planing_anyspeed as PA
    out("W1d: the X|X boundary term of a turning point moving at xt' through"
        " a widening taper (synthetic: chine wetted everywhere, yc = 0.2 + "
        "0.1 (x_s - x), Vg = x - xt(t), constant u_r = u = 1 m/s), discrete "
        "-u sum 1/2 km (Y_f^2 - Y_a^2) V_b^2 (V = 1 + 0.3 x) against the "
        "derived 1/2 int_X c_turn k V^2 dx, c_turn = -u km d(yc^2)/dx at the"
        " turn, k = xt' / (xt' + u)")
    n, dx = 96, 0.05
    xs = (np.arange(n) + 0.5) * dx - 2.4
    edges = np.concatenate([[xs[0] - 0.5 * dx], xs + 0.5 * dx])
    lo, hi = float(edges[0]), float(edges[-1])
    u = 1.0
    km = 1.0
    xs_ = hi

    def ycf(xb):
        return 0.2 + 0.1 * (xs_ - np.asarray(xb, float))

    def Vf(xb):
        return 1.0 + 0.3 * np.asarray(xb, float)
    d = np.full(n, 10.0)            # chine wetted: cz = yc everywhere
    rows, ok = [], True
    for rate in (0.5, 2.0, -0.5, -0.8):
        xt0 = -1.6 if rate > 0 else 1.6
        dt = 0.002
        b = np.array([], float)
        st = np.array([PA.ES_E], np.int8)
        y = np.array([np.inf])
        T = 1.0
        for k in range(int(round(T / dt))):
            t1 = (k + 1) * dt
            xt = xt0 + rate * t1
            vg = xs - xt
            U = np.full(n, u)
            b, st, y = PA._es_step1(b, st, xs, d, vg, U, dt, edges, U, True,
                                    y, ycf, 0.0)
        # discrete X|X term on the final field, over the planes that turned
        # during the run (the block that turned at the first step, aft of
        # x_first, excluded)
        x_first = xt0 - u * T
        tol = 2.0 * max(u, abs(rate)) * dt
        disc = 0.0
        for i, bb in enumerate(b):
            if (st[i] == PA.ES_X and st[i + 1] == PA.ES_X and lo < bb < hi
                    and bb > x_first + tol):
                disc += -u * 0.5 * km * (min(ycf(bb), y[i + 1]) ** 2
                                         - min(ycf(bb), y[i]) ** 2) \
                    * float(Vf(bb)) ** 2
        # derived: over the X region of planes that turned during the run
        xt_end = xt0 + rate * T
        kap = rate / (rate + u)
        # X planes: from the turning point back to where the first turned
        # plane has travelled (it turned at t = 0 at xt0)
        a_, b2 = max(min(x_first, xt_end), lo), min(max(x_first, xt_end), hi)
        xq = np.linspace(a_, b2, 4001)
        # the plane at xq turned at x_turn = xt0 + rate t_turn, t_turn = (xq
        # - xt0 + u T) / (rate + u)
        tt = (xq - xt0 + u * T) / (rate + u)
        xturn = xt0 + rate * tt
        yt = ycf(xturn)
        c_turn = -u * km * 2.0 * yt * (-0.1)       # d(yc^2)/dx = 2 yc yc'
        integrand = 0.5 * c_turn * kap * Vf(xq) ** 2
        der = float(np.sum(0.5 * (integrand[1:] + integrand[:-1])
                           * np.diff(xq)))
        rel = abs(disc - der) / max(abs(der), 1e-30)
        good = rel <= 0.05 and np.sign(disc) == np.sign(rate)
        ok &= bool(good)
        rows.append(dict(rate=rate, kappa=kap, discrete=disc, derived=der,
                         rel=rel, nseg=int(st.size)))
        out(f"    xt' = {rate:+.1f} u: k = {kap:+.3f}; discrete {disc:+.5e},"
            f" derived {der:+.5e}, relative difference {rel:.2e}; sign "
            f"{'+' if disc > 0 else '-'} ({st.size} segments) "
            f"{'ok' if good else 'FAIL'}")
    check("w1d", "W1d: the discrete X|X term within 5 % of the derived 1/2 "
          "int c_turn k V^2, positive when the turning point moves forward, "
          "negative when it moves aft (xt' = +0.5, +2, -0.5, -0.8 u)", ok)
    dump("w1d.json", dict(rows=rows))
    flush("w1d.txt")


# ================================================================= bracket
def s1b_variant_job(arg):
    """S1 (b)'s run (captive, calm, root + 5 cm, dt 0.02, 120 s) of variant
    'noW' (P1a), 'b' (P1a + wagner_entry), 'off' (memory off), 'a'
    (attached): the 10-s envelopes, S1 (b)'s verdict, heave and pitch rms
    over 2-30 s, mean A33 / A55 over 2-30 s."""
    variant, kn = arg
    warnings.filterwarnings("ignore")
    from hydro import hulls
    from sim import planing_anyspeed as PA
    from studies import anyspeed_s1_check as S1
    from studies.anyspeed_p1c_w3 import s1b_verdict
    u = kn * KN
    dt = 0.02
    sw = P1A if variant == "noW" else W
    p = PA.PlaningVesselAS(hulls.get(C.AS), C.calm(), dt=dt, captive_u=u,
                           test_switches=sw)
    if variant in ("off", "a"):
        p.es_xmem = False
    sol, res = S1.newton(p, u, np.array(p.running_attitude(u)))
    z0, th0 = sol
    if variant == "a":
        attached(p)
    cls = type(p)
    cap = {}

    def spy(s_, t_):
        r_ = cls.deriv(p, s_, t_)
        if "aux" not in cap:
            cap["aux"] = r_[1]
        return r_
    p.deriv = spy
    s = S1.state(u, z0 + 0.05, th0)
    p.initial_state(0.0)
    n = int(round(120.0 / dt))
    zz, tt, am = [0.05], [0.0], []
    try:
        for i in range(n):
            cap.clear()
            s = p.step(s, i * dt, 0.0, 0.0)
            s[6] = u
            zz.append(s[2] - z0)
            tt.append(s[4] - th0)
            am.append(added_masses(p, cap["aux"]))
    finally:
        del p.deriv
    k = int(round(10 / dt))
    env = [float(np.abs(np.array(zz[j_ * k:(j_ + 1) * k])).max())
           for j_ in range(len(zz) // k)]
    a0, a1 = int(round(2 / dt)), int(round(30 / dt))
    am = np.array(am)
    return dict(variant=variant, kn=kn, env=env, verdict=s1b_verdict(env),
                z_rms=float(np.std(zz[a0:a1])),
                th_rms=float(np.degrees(np.std(tt[a0:a1]))),
                A33=float(am[a0:a1, 0].mean()),
                A55=float(am[a0:a1, 1].mean()))


def _bracket_job(arg):
    kind, a = arg
    if kind == "s1b":
        return s1b_variant_job(a)
    return head_run(a)


def part_bracket():
    from studies.parallel_runner import run_parallel
    out("Sensitivity bracket of assumption (b) (an exiting plane draws no "
        "new fluid) against the attached alternative (a) (memory off, E/X "
        "flux V Dm/Dt whenever Dm/Dt > 0); 'off' the memory off (round 2), "
        "'noW' the plant without wagner_entry")
    VS = ("noW", "b", "off", "a")
    a_s1 = [("s1b", (v, kn)) for kn in (1.0, 1.5) for v in VS]
    sets = (("P1a", P1A), ("as1", as1_preset()))
    a_h = []
    for name, base in sets:
        for kn in (6.0, 12.0, 28.0):
            for v in VS:
                sw = base if v == "noW" else \
                    tuple(PA_resolve(base + ("wagner_entry",)))
                a_h.append(("head", (sw, v, kn, False)))
    res = run_parallel(_bracket_job, a_s1 + a_h, gb_per_job=0.4,
                       reserve_gb=3.0)
    rows = [r for r in res if not isinstance(r, Exception)]
    for r in res:
        if isinstance(r, Exception):
            out(f"    job failed: {r!r}")
    out("  S1 (b) runs (calm, captive): envelope 0-10 / 50-60 / 110-120 s; "
        "verdict; heave and pitch rms 2-30 s; mean A33, A55 2-30 s")
    s1 = {(r["variant"], r["kn"]): r for r in rows if "verdict" in r}
    for kn in (1.0, 1.5):
        for v in VS:
            r = s1.get((v, kn))
            if r:
                out(f"    {kn:.1f} kn {v:4s}: {r['env'][0]:.2e} "
                    f"{r['env'][5]:.2e} {r['env'][-1]:.2e} {r['verdict']:20s}"
                    f" heave rms {1000 * r['z_rms']:.2f} mm, pitch rms "
                    f"{r['th_rms']:.3f} deg; A33 {r['A33']:.0f} kg, A55 "
                    f"{r['A55']:.0f} kg m2")
        a_, b_ = s1.get(("a", kn)), s1.get(("b", kn))
        if a_ and b_:
            out(f"      spread (a) vs (b): heave rms "
                f"{100 * (a_['z_rms'] / b_['z_rms'] - 1):+.1f} %, pitch rms "
                f"{100 * (a_['th_rms'] / b_['th_rms'] - 1):+.1f} %, A33 "
                f"{100 * (a_['A33'] / b_['A33'] - 1):+.1f} %, A55 "
                f"{100 * (a_['A55'] / b_['A55'] - 1):+.1f} %")
    out("  W4-type head seas (free, constant thrust, last 40 s): mean speed,"
        " heave rms, pitch rms, mean A33, A55")
    hd = {(r["sw"], r["variant"], r["kn"]): r for r in rows
          if "verdict" not in r}
    for name, _ in sets:
        for kn in (6.0, 12.0, 28.0):
            for v in VS:
                r = hd.get((name, v, kn))
                if r:
                    out(f"    [{name}] {kn:4.1f} kn {v:4s}: speed "
                        f"{r['u']:.3f} kn, heave rms {1000 * r['z']:.1f} mm,"
                        f" pitch rms {r['th']:.3f} deg; A33 {r['A33']:.0f}, "
                        f"A55 {r['A55']:.0f}")
            a_, b_ = hd.get((name, "a", kn)), hd.get((name, "b", kn))
            if a_ and b_:
                out(f"      spread (a) vs (b): speed "
                    f"{100 * (a_['u'] / b_['u'] - 1):+.2f} %, heave rms "
                    f"{100 * (a_['z'] / b_['z'] - 1):+.1f} %, pitch rms "
                    f"{100 * (a_['th'] / b_['th'] - 1):+.1f} %, A33 "
                    f"{100 * (a_['A33'] / b_['A33'] - 1):+.1f} %, A55 "
                    f"{100 * (a_['A55'] / b_['A55'] - 1):+.1f} %")
    dump("bracket.json", dict(rows=rows))
    flush("bracket.txt")


def PA_resolve(sw):
    from sim import planing_anyspeed as PA
    return PA.resolve("as1", tuple(sw))


# ===================================================================== w6b
W6_KNS = ([float(k) for k in np.arange(1.0, 9.51, 0.5)]
          + [10.0, 12.0, 16.0, 20.0, 25.0] + [float(k) for k in range(26, 41)])


def w6b_job(arg):
    kn, mem = arg
    warnings.filterwarnings("ignore")
    from hydro import hulls
    from sim import planing_anyspeed as PA
    from studies import anyspeed_s1_check as S1
    u = kn * KN
    p = PA.PlaningVesselAS(hulls.get("planing_b@as1"), C.calm(), dt=0.02,
                           captive_u=u, test_switches=W)
    p.es_xmem = mem
    res, log, ne = S1.census(p, u)
    cells = S1.winding(p, u)
    roots = [(float(r["z"]), float(r["th"]), r["label"]) for r in res]
    free = []
    for (zc, tc, deg) in cells:
        if not any(abs(r["z"] - zc) <= 0.0151 and abs(r["trim"] - tc) <= 0.376
                   for r in res):
            free.append((round(zc, 4), round(tc, 3), deg))
    return dict(kn=kn, mem=mem, roots=roots, ncells=len(cells),
                cells=[tuple(c_) for c_ in cells], free=free)


def part_w6b():
    from studies.parallel_runner import run_parallel
    out("W6 on planing_b (12 stations with keel curvature: an X plane's "
        "mass enters the static captive force): S1 (g)'s census and winding"
        " scan, P1a + wagner_entry, X memory on vs off, at W6's 38 speeds")
    res = run_parallel(w6b_job, [(kn, m) for kn in W6_KNS
                                 for m in (False, True)],
                       gb_per_job=0.3, reserve_gb=3.0)
    by = {}
    for r in res:
        if isinstance(r, Exception):
            out(f"    job failed: {r!r}")
            continue
        by[(r["kn"], r["mem"])] = r
    diffs = []
    for kn in W6_KNS:
        a_, b_ = by.get((kn, False)), by.get((kn, True))
        if not a_ or not b_:
            diffs.append(kn)
            continue
        same = (len(a_["roots"]) == len(b_["roots"]) and all(
            x[2] == y[2] and abs(x[0] - y[0]) <= 1e-9
            and abs(x[1] - y[1]) <= 1e-9
            for x, y in zip(a_["roots"], b_["roots"]))
            and a_["cells"] == b_["cells"] and a_["free"] == b_["free"])
        if not same:
            diffs.append(kn)
        out(f"   {kn:4.1f} kn: roots "
            f"{[(round(x[0], 4), round(x[1], 5), x[2]) for x in b_['roots']]}"
            f", winding cells {b_['ncells']}, zeros away from roots "
            f"{len(b_['free'])}; memory on == off {same}")
    check("w6b", "W6 on planing_b: census roots (position, type), winding "
          "cells and zeros away from roots identical with the X memory on "
          "and off at every speed 1-40 kn", not diffs,
          f"{len(diffs)} speeds differ {diffs[:8]}")
    dump("w6b.json", dict(runs=list(by.values()), diffs=diffs))
    flush("w6b.txt")


# ==================================================================== s1fb
def _fol_sea(scale):
    from learn.repro import task
    m = task.Mission("high", 0, 0, t_end=1.0, track=task.LEGS[0])
    sea = copy.deepcopy(m.plant.sea)
    sea.th = np.asarray(sea.th, float) + np.pi
    sea.a = np.asarray(sea.a, float) * scale
    return sea, m.plant.dt


def s1f_job(arg):
    """One job of S1 (f) on (hull, switch set, memory): ('basin', kn) the
    8 captive calm runs from root + (dz, dtrim); ('fol', scale, kn) the
    following-sea open loop, 240 s."""
    hull, sw, mem, kind, a1, a2 = arg
    warnings.filterwarnings("ignore")
    from hydro import hulls
    from sim import planing_anyspeed as PA
    from studies import anyspeed_s1_check as S1
    h = hulls.get(hull)
    if kind == "basin":
        kn = a1
        u = kn * KN
        p = PA.PlaningVesselAS(h, C.calm(), dt=S1.DT, captive_u=u,
                               test_switches=sw)
        p.es_xmem = mem
        g = np.array(p.running_attitude(u))
        root, r = S1.newton(p, u, g)
        if r > 1e-10:
            z, th, _, _ = S1.settle(p, u, T=60.0)
            root = np.array([z, th])
        rows = S1.basin(p, u, root)
        return dict(hull=hull, mem=mem, kind=kind, kn=kn,
                    root=[float(root[0]), float(root[1])], resid=float(r),
                    runs=rows)
    scale, kn = a1, a2
    sea, dt = _fol_sea(scale)
    p = PA.PlaningVesselAS(h, sea, dt=dt, test_switches=sw)
    p.es_xmem = mem
    r = S1.open_loop(p, kn)
    return dict(hull=hull, mem=mem, kind=kind, scale=scale, kn=kn, **r)


def part_s1fb():
    from studies.parallel_runner import run_parallel
    out("S1 (f) robustness on planing_b (P1a + wagner_entry, X memory on vs"
        " off) and on scarab195 with the current as1 preset + wagner_entry:"
        " basin (16-35 kn, root + {-0.2, 0, 0.2} m x {-5, 0, 5} deg, 40 s)"
        " and following seas (the task sea turned 180 deg, Hs x1 / x1.5, "
        "12-30 kn, 240 s open loop)")
    combos = [("planing_b@as1", W, False), ("planing_b@as1", W, True),
              ("scarab195@as1", tuple(as1w()), True)]
    args = []
    for hull, sw, mem in combos:
        for kn in (16, 20, 25, 30, 35):
            args.append((hull, sw, mem, "basin", kn, None))
        for scale in (1.0, 1.5):
            for kn in (12, 20, 25, 30):
                args.append((hull, sw, mem, "fol", scale, kn))
    res = run_parallel(s1f_job, args, gb_per_job=0.4, reserve_gb=3.0)
    ok = {}
    for r in res:
        if isinstance(r, Exception):
            out(f"    job failed: {r!r}")
            continue
        key = (r["hull"], r["mem"])
        tag = f"{r['hull']} memory {'on' if r['mem'] else 'off'}"
        if r["kind"] == "basin":
            rows = r["runs"]
            nd = [x for x in rows if x["diverged_t"] is not None]
            far = [x for x in rows if x["diverged_t"] is None
                   and x["end_dist"] > 1e-3]
            good = not nd and not far
            out(f"   [{tag}] basin {r['kn']} kn root z {r['root'][0]:+.4f} "
                f"(r {r['resid']:.0e}): diverged {len(nd)}, not back within "
                f"1 mm / 0.01 deg {len(far)}; min trim "
                f"{min(x['trmin'] for x in rows):+.2f} deg")
        else:
            good = r["diverged_t"] is None and not r["capsized"]
            out(f"   [{tag}] following Hs x{r['scale']} {r['kn']} kn: "
                + (f"DIVERGED at {r['diverged_t']:.2f} s"
                   if r["diverged_t"] is not None else "ok")
                + f"; min trim {r['min_trim']:+.2f} deg, min z "
                f"{r['min_z']:+.3f} m, end {r['end_kn']:.2f} kn, capsized "
                f"{r['capsized']}")
        ok[key] = ok.get(key, True) and good
    for hull, sw, mem in combos:
        check("s1fb", f"S1 (f) on {hull} ({'as1 + wagner_entry' if 'fk_exact' in sw else 'P1a + wagner_entry'}, "
              f"memory {'on' if mem else 'off'}): every basin start back, "
              "no divergence or capsize in following seas",
              ok.get((hull, mem), False))
    dump("s1fb.json", dict(runs=[r for r in res
                                 if not isinstance(r, Exception)]))
    flush("s1fb.txt")


# ================================================================== calmid
def calmid_job(arg):
    kind, sw, kn = arg
    warnings.filterwarnings("ignore")
    from hydro import hulls
    from sim import planing_anyspeed as PA
    from studies import anyspeed_s1_check as S1
    u = kn * KN
    p = PA.PlaningVesselAS(hulls.get(C.AS), C.calm(), dt=0.02, captive_u=u,
                           test_switches=sw)
    if kind == "census":
        res, log, ne = S1.census(p, u)
        return [(float(r["z"]), float(r["th"]), r["label"]) for r in res]
    sol, res = S1.newton(p, u, np.array(p.running_attitude(u)))
    zz, tt = S1.decay_run(p, u, sol[0], sol[1], 0.05)
    return np.stack([zz, tt])


def part_calmid():
    from studies.parallel_runner import run_parallel
    a1 = tuple(as1w())
    out(f"calmid: the current as1 preset + wagner_entry {a1} against P1a + "
        "wagner_entry in calm water: S1 (b)'s runs (1.0, 1.5 kn, 120 s) and"
        " the census (1, 9, 30 kn), bit for bit")
    args = [(k, sw, kn) for k, kns in (("b", (1.0, 1.5)),
                                       ("census", (1.0, 9.0, 30.0)))
            for kn in kns for sw in (W, a1)]
    res = run_parallel(calmid_job, args, gb_per_job=0.3, reserve_gb=3.0)
    ok = True
    for i in range(0, len(args), 2):
        ra, rb = res[i], res[i + 1]
        if isinstance(ra, Exception) or isinstance(rb, Exception):
            out(f"    job failed: {ra!r} {rb!r}")
            ok = False
            continue
        if args[i][0] == "b":
            same = bool(np.array_equal(ra, rb))
        else:
            same = ra == rb
        ok &= same
        out(f"    {args[i][0]:6s} {args[i][2]:4.1f} kn: identical {same}")
    check("calmid", "the current as1 preset + wagner_entry equals P1a + "
          "wagner_entry bit for bit in calm water (S1 b runs, census)", ok)
    flush("calmid.txt")


# ================================================================== w4as1
def part_w4as1():
    from studies.parallel_runner import run_parallel
    a1 = tuple(as1w())
    out(f"W4 on the current as1 preset + wagner_entry {a1}: regular head "
        "waves (lam 2 L, a 0.12 m), constant thrust, 60 s (last 40 s): F "
        "share, mean speed, heave and pitch rms by step; criterion "
        "unchanged (1 % over dt 0.02 / 0.01 / 0.005 at 12 and 28 kn)")
    dts = (0.02, 0.01, 0.005, 0.0025)
    base = as1_preset()
    args = [(sw, kn, dt) for kn in (12.0, 28.0) for dt in dts
            for sw in (a1, base)]
    res = run_parallel(FX.w4_job, args, gb_per_job=0.3, reserve_gb=3.0)
    by = {(r["sw"], r["kn"], r["dt"]): r for r in res
          if not isinstance(r, Exception)}
    for r in res:
        if isinstance(r, Exception):
            out(f"    job failed: {r!r}")
    ok = True
    for kn in (12.0, 28.0):
        for sw in ("W", "P1a"):
            for dt in dts:
                r = by.get((sw, kn, dt))
                if r:
                    out(f"    [{'as1+W' if sw == 'W' else 'as1'}] {kn:4.1f} "
                        f"kn dt {dt:<6}: F share {r['F']:.4f}, speed "
                        f"{r['u']:.4f} kn, heave rms {1000 * r['z']:.3f} mm,"
                        f" pitch rms {r['th']:.4f} deg")
        rs = [by.get(("W", kn, dt)) for dt in dts[:3]]
        if any(r is None for r in rs):
            ok = False
            continue
        for q in ("F", "u", "z", "th"):
            v = [r[q] for r in rs]
            spread = max(abs(a / b - 1) for a in v for b in v if b != 0.0) \
                if all(x != 0.0 for x in v) else 0.0
            ok &= spread <= 0.01
            v4 = [by[("W", kn, dt)][q] for dt in dts]
            d1, d2, d3 = (abs(v4[0] - v4[1]), abs(v4[1] - v4[2]),
                          abs(v4[2] - v4[3]))
            od = [math.log2(a / b) if a > 0 and b > 0 else float("nan")
                  for a, b in ((d1, d2), (d2, d3))]
            out(f"      {kn:4.1f} kn {q:2s}: largest relative difference "
                f"over dt 0.02/0.01/0.005 {100 * spread:.3f} % "
                f"{'ok' if spread <= 0.01 else 'FAIL'}; measured order "
                f"{od[0]:.2f}, {od[1]:.2f}")
    check("w4as1", "W4 on as1 + wagner_entry: F share, mean speed, heave "
          "and pitch rms within 1 % between dt 0.02, 0.01, 0.005 at 12 and "
          "28 kn", ok)
    dump("w4as1.json", dict(rows=list(by.values())))
    flush("w4as1.txt")


# ===================================================================== w5y
def _first_finite_y(hid, start=1):
    for i in range(start, len(hid)):
        y = hid[i].get("y") if isinstance(hid[i], dict) else None
        if y is not None and np.any(np.isfinite(np.asarray(y, float))):
            return i
    return None


def part_w5y():
    from hydro import hulls
    from studies import anyspeed_s1_check as S1
    out("W5 (c) / W5cap (c) replayed from the first recorded row whose "
        "hidden state has a finite turn half-beam Y (P1a + wagner_entry), "
        "bit for bit to the end of the record; the same row without its "
        "'y' (an older record) as a stated difference after 50 steps")
    rows = []
    ok = True
    # free: 12 kn regular head waves (W5's run), 1000 steps
    sea = FX._head_sea(hulls.get(C.AS))
    p = FX._plant(W, sea=sea, record=True)
    s = p.initial_state(12 * KN)
    thr = s[12]
    S = [s.copy()]
    snaps = {}
    t = 0.0
    for i in range(1000):
        snaps[i] = (copy.deepcopy(p.prop), copy.deepcopy(p.rudder), t)
        s = p.step(s, t, thr, 0.0)
        t += 0.02
        S.append(s.copy())
    tr = p.trajectory()
    hid = tr["hidden"]
    for label, k in (("free 12 kn", _first_finite_y(hid)),):
        if k is None:
            ok = False
            out(f"    {label}: no row with a finite Y")
            continue
        yk = [round(float(v), 4) for v in hid[k]["y"] if np.isfinite(v)]

        def replay(h, n_steps):
            q = FX._plant(W, sea=sea, record=False)
            q.initial_state(12 * KN)
            q.prop, q.rudder = (copy.deepcopy(snaps[k][0]),
                                copy.deepcopy(snaps[k][1]))
            q.set_hidden_state(h)
            s2 = np.array(tr["s"][k], float)
            t2 = snaps[k][2]
            same = True
            for i in range(k, min(k + n_steps, 1000)):
                s2 = q.step(s2, t2, thr, 0.0)
                t2 += 0.02
                same &= bool(np.array_equal(s2, S[i + 1]))
            return same, s2
        same, _ = replay(hid[k], 1000)
        h2 = dict(hid[k])
        h2.pop("y", None)
        _, s3 = replay(h2, 50)
        dev = float(np.abs(s3 - S[k + 50]).max())
        ok &= same
        rows.append(dict(run=label, row=k, Y=yk, replay=same,
                         without_y_dev=dev))
        out(f"    {label}: first row with a finite Y {k} (Y {yk}); replay "
            f"rows {k}-1000 bit for bit {same}; without 'y' the state after "
            f"50 steps differs by {dev:.2e} (stated difference)")
    # captive: S1 (b)'s run, 1.5 kn, root + 5 cm, 1500 steps (W5cap's run)
    u = 1.5 * KN
    p = FX._plant(W, captive_u=u, dt=0.02, record=True)
    sol, res = S1.newton(p, u, np.array(p.running_attitude(u)))
    s = S1.state(u, sol[0] + 0.05, sol[1])
    p.initial_state(0.0)
    S = [s.copy()]
    for i in range(1500):
        s = p.step(s, i * 0.02, 0.0, 0.0)
        s[6] = u
        S.append(s.copy())
    hid = p.trajectory()["hidden"]
    k = _first_finite_y(hid)
    if k is None:
        ok = False
        out("    captive 1.5 kn: no row with a finite Y")
    else:
        yk = [round(float(v), 4) for v in hid[k]["y"] if np.isfinite(v)]

        def replay_c(h, n_steps):
            q_ = FX._plant(W, captive_u=u, dt=0.02)
            q_.initial_state(0.0)
            q_.set_hidden_state(h)
            s2 = S[k].copy()
            same = True
            for i in range(k, min(k + n_steps, 1500)):
                s2 = q_.step(s2, i * 0.02, 0.0, 0.0)
                s2[6] = u
                same &= bool(np.array_equal(s2, S[i + 1]))
            return same, s2
        same, _ = replay_c(hid[k], 1500)
        h2 = dict(hid[k])
        h2.pop("y", None)
        _, s3 = replay_c(h2, 50)
        dev = float(np.abs(s3 - S[k + 50]).max())
        ok &= same
        rows.append(dict(run="captive 1.5 kn", row=k, Y=yk, replay=same,
                         without_y_dev=dev))
        out(f"    captive 1.5 kn: first row with a finite Y {k} (Y {yk}); "
            f"replay rows {k}-1500 bit for bit {same}; without 'y' the state"
            f" after 50 steps differs by {dev:.2e} (stated difference)")
    check("w5y", "W5 (c) and W5cap (c) from the first row with a finite Y: "
          "replay bit for bit", ok)
    dump("w5y.json", dict(rows=rows))
    flush("w5y.txt")


# ===================================================================== e1
def part_e1():
    from hydro import hulls
    from learn.repro import task
    from sim import physics
    from sim import planing_anyspeed as PA
    from studies import anyspeed_p0_check as P0
    out("E1 on the current preset: ms per 0.02 s step of the GZ plant in the"
        " task sea (P0's run and interleaved method, ratio to v0 x v0's "
        "idle cost)")
    m = task.Mission("high", 0, 0, t_end=1.0, track=task.LEGS[0])
    sea = m.plant.sea
    sets = {"P1a": P1A, "P1a+W": W, "as1": as1_preset(),
            "as1+W": tuple(as1w())}
    plants = {"v0": physics.gz_twin(hulls.get("scarab195").plant(sea,
                                                                 dt=0.02))}
    for k, sw in sets.items():
        plants[k] = PA.PlaningVesselAS(hulls.get(C.AS), sea, dt=0.02,
                                       test_switches=sw)
    keys = list(plants)
    N, rounds = 300, 9
    per = {k: [] for k in keys}
    for r in range(rounds):
        order_ = keys[r % len(keys):] + keys[:r % len(keys)]
        for k in order_:
            p = plants[k]
            s_ = p.initial_state(0.9 * float(p.hull.u_design))
            tm, rm = p.prop.t_max, p.rudder.max
            ta = time.perf_counter()
            for i in range(N):
                s_ = p.step(s_, i * 0.02, tm * (0.6 + 0.3 * math.sin(
                    0.05 * i)), rm * 0.5 * math.sin(0.03 * i), 0.02)
            per[k].append(1e3 * (time.perf_counter() - ta) / N)
    rat = {k: float(np.median(np.array(per[k]) / np.array(per["v0"])))
           for k in keys[1:]}
    est = {k: rat[k] * P0.V0_IDLE_MS for k in rat}
    for k in keys[1:]:
        out(f"    {k:<7} / v0 {rat[k]:.3f} -> idle estimate {est[k]:.2f} ms")
    out(f"    wagner_entry adds {est['as1+W'] - est['as1']:.2f} ms on the "
        f"current preset ({est['P1a+W'] - est['P1a']:.2f} ms on P1a)")
    check("e1", f"E1: as1 + wagner_entry at most {P0.E1_MS} ms per step "
          "(recorded shortfall under the 2026-10-06 revision if not)",
          est["as1+W"] <= P0.E1_MS, f"{est['as1+W']:.2f} ms")
    dump("e1.json", dict(per_round_ms=per, ratio_to_v0=rat,
                         idle_estimate_ms=est))
    flush("e1.txt")


PARTS = ("w19p", "rbound", "w1d", "bracket", "w6b", "s1fb", "calmid",
         "w4as1", "w5y", "e1")


def _sub(arg):
    name, out_dir = arg
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"p1cw4_{name}_stdout.log"), "w") as log:
        rc = subprocess.run([sys.executable, "-m", "studies.anyspeed_p1c_w4",
                             "--parts", name, "--out", out_dir], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT).returncode
    return name, rc, time.time() - t0


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="")
    ap.add_argument("--jobs", default="", help="parts, one process each")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    OUT = os.path.abspath(a.out)
    C.OUT = OUT
    os.makedirs(OUT, exist_ok=True)
    if a.jobs:
        from studies.parallel_runner import run_parallel
        t0 = time.time()
        res = run_parallel(_sub, [(x, OUT) for x in a.jobs.split(",") if x],
                           gb_per_job=0.5, reserve_gb=3.0, max_procs=3)
        for r in res:
            print(f"  {r}", flush=True)
        print(f"all done in {time.time() - t0:.0f} s wall", flush=True)
        return
    C.CAND = W
    C.STAMP = C.stamp_all()
    with open(os.path.abspath(__file__), "rb") as f:
        C.STAMP["p1c_w4_sha1"] = hashlib.sha1(f.read()).hexdigest()
    t0 = time.time()
    for part in [x for x in a.parts.split(",") if x]:
        if part not in PARTS:
            raise SystemExit(f"unknown part {part}")
        globals()[f"part_{part}"]()
    tag = "_".join(x for x in a.parts.split(",") if x)
    dump(f"checks_{tag}.json", C.CHECKS)
    nf = sum(not c["ok"] for c in C.CHECKS)
    print(f"{len(C.CHECKS)} checks, {nf} failed, {time.time() - t0:.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
