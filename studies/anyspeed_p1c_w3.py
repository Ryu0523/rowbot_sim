#!/usr/bin/env python3
"""
Step P1c, track W, round 3 (learn/meta/ANYSPEED_PLAN_2026-10-05.md, '###
wagner 低速'): the checks registered there before they ran. The re-runs of
earlier W checks use their own scripts (studies/anyspeed_p1b_fix.py,
studies/anyspeed_p1b_check.py --cand w, studies/anyspeed_p1c_w.py,
studies/anyspeed_p0_s1rerun.py); this module adds:

  w19       the Jacobi-form power budget of S1 (b)'s run (captive, calm,
            root + 5 cm, dt 0.02) at 1.0 / 1.5 kn over 30-90 s:
                dH/dt = u m_T V_T g_T - u sum_b [e]_b + sum (f - f_att) g dx
                        + cross-flow + holding/friction/air moments + q Mh,
            H = KE + U + sum 1/2 m V^2 dx - u sum m V tau dx, f_att the
            attached (Lagrangian) force of an earth-fixed plane whose section
            changes as the hull passes; (a) the plant with the X memory off
            (P1c round 2), (b) the plant as it is. Gates: closure within 5 %
            of the gross; (a) no group but the gate-closed widening planes
            above +1 % of the gross; (b) no group above +1 %
  w1c       transport of the X planes' turn half-beams y (synthetic field,
            Courant 0.3 / 1 / 3, 200 steps): boundaries within 1e-9 m of
            exact, every segment's y unchanged bit for bit
  w18iv     S1 (b)'s run from + 5 cm at 0.5, 1, 1.5, 2, 3, 4 kn, 600 s: S1
            (b)'s verdict at 600 s (round 2's improved W18 (iv))
  ytol      (recorded) S1 (b) at 1.0 / 1.5 kn with ES_YTOL_REL / 10
  radiation (recorded) the zero-speed 3D radiation power (Capytaine, the
            P1b review's exact-geometry mesh) on the 30-90 s trajectory of
            the plant with the memory off, against w19 (a)'s input
  goldenw3  N2: the golden runs of P1a + wagner_entry in two processes, bit
            for bit, largest calm heel <= 15 deg, written to
            golden_as_...+wagner_entry_p1c3.npz (older files kept)

    .venv/Scripts/python.exe -m studies.anyspeed_p1c_w3 --parts w1c,w19
    .venv/Scripts/python.exe -m studies.anyspeed_p1c_w3 --jobs w19,w18iv,...
"""
import argparse
import hashlib
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

OUT = os.path.join(HERE, "_cache", "anyspeed", "p1c", "w3")
KN = C.KN
G = C.G
W = FX.W
out, check, dump, flush = C.out, C.check, C.dump, C.flush


# ===================================================================== w19
def _terms(p, s, ds, aux, u):
    """The budget quantities of one derivative (the step's first) of a
    captive calm-water W plant: the groups of the identity, as powers."""
    from sim import planing_anyspeed as PA
    from sim.planing_vessel import SP
    kb = aux["kb"]
    x, dx, rho = p.x, p.dx, p.rho
    d, V, Vg, ur = aux["d"], kb.V, kb.Vg, kb.ur
    w, q, wd, qd = s[8], s[10], ds[8], ds[10]
    tau = p.kp + SP * (p.Th_rest + s[4])
    A, c, maZ, dma, _ = p._section(d)
    wet = d > 0.0
    f = np.asarray(kb.f, float)
    fX = np.zeros_like(d) if kb.fX is None else np.asarray(kb.fX, float)
    M = kb.ma
    km = p.k_a * 0.5 * np.pi * rho
    kc = 0.5 * np.pi / p.tanb
    m0 = np.zeros_like(d)
    dm0y = np.zeros_like(d)
    j = np.nonzero(wet)[0]
    a0, _, a2 = p._tab().vertical(d[j], p.yc[j])
    m0[j], dm0y[j] = rho * a0, rho * a2
    a = 2.0 * SP * u * q - u * u * p.kpp - (wd + SP * x * qd)
    # E/X rates as _flux_b (calm: slope -tau), split d-part / convective
    r_d = dma * Vg
    r_c = np.zeros_like(d)
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
    # the X planes with the memory: m = km min(kc d, yc, Y)^2; attached
    # force 1/2 dm/dd V^2 where kc d binds (no growth where Y binds), the
    # plant's none: delta = -km kc^2 d V^2 on that share
    dX = np.zeros_like(d)
    if kb.esy is not None and fX.any():
        b, st = kb.es
        y = kb.esy
        eg = p._es_geom()
        cz = np.minimum(kc * np.maximum(d, 0.0), p.yc)
        k = np.nonzero(st[1:-1] == PA.ES_X)[0]
        e = eg.edges
        for kk in k:
            a_, b_, Y = b[kk], b[kk + 1], y[kk + 1]
            for jj in range(max(int(np.searchsorted(e, a_, "right")) - 1, 0),
                            min(int(np.searchsorted(e, b_, "left")),
                                e.size - 1)):
                ln = max(min(b_, e[jj + 1]) - max(a_, e[jj]), 0.0)
                if ln > 0.0 and kc * max(d[jj], 0.0) <= min(p.yc[jj], Y):
                    dX[jj] += ln / (e[jj + 1] - e[jj])
        sX = -km * kc * kc * np.maximum(d, 0.0) * V * V * dX
        wXEX = 0.0
    else:
        sX = np.zeros_like(d)
        wXEX = 1.0
    # E (and, without the memory, X) planes: Zarnick's rule against the
    # attached force 1/2 V r_d + V r_c
    wEX = wE + wXEX * fX
    dEo = wEX * gEX * 0.5 * V * r_d
    dEc = wEX * (1.0 - gEX) * (-0.5) * V * r_d
    cEX = wEX * (gEX - 1.0) * V * r_c
    cF = f * (gF - 1.0) * V * conv0
    cf = 0.5 * rho * p.c_dc * 2.0 * c * V * np.abs(V)
    ab = p._a_b(u)
    qsec = PA._vsec_q(d, p.yc, p.tanb)[0]
    zw = float(np.mean(p.zk[wet])) if wet.any() else float(p.zk[0])
    _, _, zn = p.thrusters[0]
    Mload = zn * aux["fx"] - zw * aux["Df"] - p.num.z_air * aux["Da"]
    # state boundaries: m on either side (E Zarnick, X the memory's, F m0)
    b, st = kb.es
    y = kb.esy

    def m_of(state, xb, Y):
        mz = float(np.interp(xb, x, maZ))
        if state == PA.ES_F:
            return float(np.interp(xb, x, m0))
        if state == PA.ES_X and y is not None:
            cb = min(kc * max(float(np.interp(xb, x, d)), 0.0),
                     float(np.interp(xb, x, p.yc)), Y)
            return km * cb * cb
        return mz
    eb = 0.0
    for i, bb in enumerate(b):
        sa, sf = st[i], st[i + 1]
        if sa == sf and (y is None or y[i] == y[i + 1]):
            continue
        if bb <= x[0] or bb >= x[-1] or np.interp(bb, x, d) <= 0.0:
            continue
        Vb = float(np.interp(bb, x, V))
        ya = np.inf if y is None else y[i]
        yf = np.inf if y is None else y[i + 1]
        eb += 0.5 * (m_of(sf, bb, yf) - m_of(sa, bb, ya)) * Vb * Vb

    def S(z):
        return float(np.sum(z) * dx)
    return dict(
        z_=float(s[2]), th_=float(s[4]), w_=float(w), q_=float(q),
        KE=0.5 * p.m * w * w + 0.5 * p.Iy * q * q,
        U=p.m * G * float(s[2]) + ab * rho * G * S(qsec),
        Tf=0.5 * S(M * V * V), Gm=S(M * V * tau),
        P_dEo=S(dEo * g), P_dEc=S(dEc * g), P_cEX=S(cEX * g),
        P_cF=S(cF * g), P_X=S(sX * g),
        P_T=u * M[0] * V[0] * g[0], P_b=-u * eb,
        P_cf=S(cf * g), P_ld=q * Mload, P_Mh=q * aux["Mh"],
        Fsh=float(f[wet].mean()) if wet.any() else 0.0,
        Xsh=float(fX[wet].mean()) if wet.any() else 0.0)


W19_GROUPS = ("P_dEo", "P_dEc", "P_cEX", "P_cF", "P_X", "P_T", "P_b",
              "P_cf", "P_ld", "P_Mh")
W19_NAMES = dict(
    P_dEo="E planes entering, Wagner d-part (+1/2 dm/dd V^2)",
    P_dEc="E/X d-part, gate closed (-1/2 dm/dd V^2)",
    P_cEX="E/X widening, gate closed (-V c): the input candidate",
    P_cF="F widening, convective gate closed (-V c0)",
    P_X="X with the memory: d-part (-km kc^2 d V^2)",
    P_T="transom shedding u m_T V_T g_T", P_b="state boundaries -u [e]",
    P_cf="cross-flow drag", P_ld="holding force, friction, air: pitch",
    P_Mh="longitudinal hydrostatic moment")


def _run_b(kn, T1, dz=0.05, keep=True, xmem=True, ytol_rel=None):
    """S1 (b)'s run (captive, calm, Newton root + dz, dt 0.02) of the W
    plant; the budget terms of every step's first derivative (keep)."""
    from sim import planing_anyspeed as PA
    from studies import anyspeed_s1_check as S1
    u = kn * KN
    dt = 0.02
    keep_mem, keep_tol = PA.ES_XMEM, PA.ES_YTOL_REL
    PA.ES_XMEM = xmem
    if ytol_rel is not None:
        PA.ES_YTOL_REL = ytol_rel
    try:
        p = FX._plant(W, captive_u=u, dt=dt)
        sol, res = S1.newton(p, u, np.array(p.running_attitude(u)))
        z0, th0 = sol
        sroot = S1.state(u, z0, th0)
        ds_r, aux_r = type(p).deriv(p, sroot, 0.0)
        steady = _terms(p, sroot, ds_r, aux_r, u) if keep else None
        cls = type(p)
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
        for i in range(n):
            cap.clear()
            s = p.step(s, i * dt, 0.0, 0.0)
            s[6] = u
            zz.append(s[2] - z0)
            if keep:
                rows.append(_terms(p, cap["s"], cap["ds"], cap["aux"], u))
    finally:
        PA.ES_XMEM, PA.ES_YTOL_REL = keep_mem, keep_tol
    k = int(round(10 / dt))
    env = [float(np.abs(np.array(zz[j * k:(j + 1) * k])).max())
           for j in range(len(zz) // k)]
    return dict(steady=steady, rows=rows, env=env, resid=float(res))


def w19_job(arg):
    kn, xmem = arg
    warnings.filterwarnings("ignore")
    T0, T1 = 30.0, 90.0
    r = _run_b(kn, T1 + 0.1, xmem=xmem)
    u, dt = kn * KN, 0.02
    rows, st = r["rows"], r["steady"]
    Z = {k: np.array([x_[k] for x_ in rows]) for k in rows[0]}
    H = Z["KE"] + Z["U"] + Z["Tf"] - u * Z["Gm"]
    n0, n1 = int(round(T0 / dt)), int(round(T1 / dt))
    idx = np.arange(n0, n1)
    means = {k: float(Z[k][idx].mean() - st[k]) for k in W19_GROUPS}
    return dict(kn=kn, xmem=xmem, env=r["env"], means=means,
                dH=float((H[n1] - H[n0]) / (T1 - T0)),
                Fsh=float(Z["Fsh"][idx].mean()),
                Xsh=float(Z["Xsh"][idx].mean()))


def part_w19():
    from studies.parallel_runner import run_parallel
    out("W19: the Jacobi-form budget of S1 (b)'s run (captive, calm, root + "
        "5 cm, dt 0.02), 30-90 s, window means minus the steady state's (W;"
        " positive = into H)")
    args = [(kn, xm) for xm in (False, True) for kn in (1.0, 1.5)]
    res = run_parallel(w19_job, args, gb_per_job=0.5, reserve_gb=3.0)
    rows = []
    ok_a = ok_b = True
    for r in res:
        if isinstance(r, Exception):
            out(f"    job failed: {r!r}")
            ok_a = ok_b = False
            continue
        rows.append(r)
        tot = sum(r["means"].values())
        gross = sum(abs(v) for v in r["means"].values())
        res_ = r["dH"] - tot
        tag = "(b) X memory on" if r["xmem"] else "(a) X memory off"
        out(f"  {r['kn']:.1f} kn {tag}: envelope every 30 s "
            + " ".join(f"{a:.1e}" for a in r["env"][::3]))
        out(f"    measured dH/dt {r['dH']:+.4f} W; sum of groups {tot:+.4f};"
            f" residual {res_:+.4f} = {100 * abs(res_) / gross:.2f} % of the "
            f"gross {gross:.3f}; F share {r['Fsh']:.3f}, X share "
            f"{r['Xsh']:.3f}")
        for k in W19_GROUPS:
            out(f"      {k:6s} {r['means'][k]:+9.4f}  {W19_NAMES[k]}")
        close = abs(res_) <= 0.05 * gross
        others = [k for k in W19_GROUPS if r["means"][k] > 0.01 * gross
                  and (r["xmem"] or k not in ("P_cEX", "P_cF"))]
        out(f"    closes (5 %): {close}; groups above +1 % of the gross "
            f"{'(other than the gate-closed widening planes) ' if not r['xmem'] else ''}"
            f"{others}")
        if r["xmem"]:
            ok_b &= close and not others
        else:
            ok_a &= close and not others
    check("w19", "W19 (a): memory off -- the budget closes within 5 % and no "
          "group but the gate-closed widening planes is above +1 % of the "
          "gross (1.0, 1.5 kn)", ok_a)
    check("w19", "W19 (b): the plant -- the budget closes within 5 % and no "
          "group is above +1 % of the gross (1.0, 1.5 kn)", ok_b)
    dump("w19.json", dict(rows=rows))
    flush("w19.txt")


# ===================================================================== w1c
def part_w1c():
    from sim import planing_anyspeed as PA
    out("W1c: transport of the X planes' turn half-beams (synthetic: every "
        "plane wet and not entering, X blocks of distinct Y, one F block, "
        "two E blocks that turn to X at the first step, constant u_r)")
    n, dx = 48, 0.1125
    xs = (np.arange(n) + 0.5) * dx - 2.0
    edges = np.concatenate([[xs[0] - 0.5 * dx], xs + 0.5 * dx])
    lo, hi = float(edges[0]), float(edges[-1])
    d = np.full(n, 0.3)
    vg = np.full(n, -0.5)        # not entering: X stays X, F stays F
    ur = 2.0

    def ycf(xb):
        return 0.5 + 0.0 * np.asarray(xb, float)
    ytol = 2e-3
    worst_b, same_y, ok, first_ok = 0.0, True, True, True
    for cour in (0.3, 1.0, 3.0):
        dt = cour * dx / ur
        b0 = np.array([0.30, 0.55, 0.80, 1.20, 1.45, 1.70])
        st0 = np.array([PA.ES_E, PA.ES_X, PA.ES_X, PA.ES_E, PA.ES_X,
                        PA.ES_F, PA.ES_E], np.int8)
        y0 = np.array([np.inf, 0.40, 0.45, np.inf, 0.52, np.inf, np.inf])
        b, st, y = b0.copy(), st0.copy(), y0.copy()
        U = np.full(n, ur)
        # after the first step: the E blocks inside turned (Y = ycf = 0.5;
        # the arriving planes join the last one), the rest unchanged
        y_right = np.array([0.40, 0.45, 0.50, 0.52, np.inf, 0.50])
        for k in range(200):
            b, st, y = PA._es_step1(b, st, xs, d, vg, U, dt, edges, U, True,
                                    y, ycf, ytol)
            ex = b0 - ur * dt * (k + 1)
            m = ex > lo
            inner = b[(b > lo) & (b < hi)]
            if inner.size != int(m.sum()):
                ok = False
                break
            worst_b = max(worst_b, float(np.abs(inner - ex[m]).max())
                          if inner.size else 0.0)
            yi = y[1:][(b > lo) & (b < hi)]
            same_y &= bool(np.array_equal(yi, y_right[m]))
            if k == 0:
                i_lo = int(np.searchsorted(b, lo, side="right"))
                first_ok &= bool(y[i_lo] == 0.5 and st[i_lo] == PA.ES_X)
        out(f"    Courant {cour}: worst boundary error {worst_b:.1e} m; Y "
            f"moved with its segment bit for bit {same_y}; the turned blocks"
            f" took the half-beam {first_ok}; segment count consistent {ok}")
    same_y &= first_ok
    check("w1c", "W1c: boundaries within 1e-9 m of exact and every Y "
          "unchanged (Courant 0.3, 1, 3; 200 steps)",
          ok and worst_b <= 1e-9 and same_y, f"{worst_b:.1e} m")
    flush("w1c.txt")


# =================================================================== w18iv
def s1b_verdict(env, k=None):
    e = env if k is None else env[:k]
    falling = e[-1] < 0.9 * e[-2] or e[-1] < 1e-12
    if e[-1] < 0.01 * e[0] and falling:
        return "decays"
    if e[-1] > e[0]:
        return "GROWS"
    return "no decay (bounded)"


def w18_job(arg):
    kn, T = arg
    warnings.filterwarnings("ignore")
    r = _run_b(kn, T, keep=False)
    return dict(kn=kn, T=T, env=r["env"])


def part_w18iv():
    from studies.parallel_runner import run_parallel
    out("W18 (iv), improved form: S1 (b)'s run from + 5 cm, 600 s; S1 (b)'s "
        "verdict at 600 s (last 10-s envelope below 1 % of the first and "
        "still falling, or below 1e-12 m)")
    res = run_parallel(w18_job, [(kn, 600.0) for kn in
                                 (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)],
                       gb_per_job=0.4, reserve_gb=3.0)
    ok = True
    rows = []
    for r in res:
        if isinstance(r, Exception):
            out(f"    job failed: {r!r}")
            ok = False
            continue
        v = s1b_verdict(r["env"])
        v120 = s1b_verdict(r["env"], 12)
        rows.append(dict(r, verdict=v, verdict_120=v120))
        out(f"    {r['kn']:3.1f} kn: verdict at 600 s {v:18s} (at 120 s "
            f"{v120}); envelope every 60 s "
            + " ".join(f"{a:.1e}" for a in r["env"][::6]))
        ok &= v == "decays"
    check("w18iv", "W18 (iv): 0.5-4 kn from + 5 cm decay by S1 (b)'s verdict"
          " at 600 s", ok)
    dump("w18iv.json", dict(rows=rows))
    flush("w18iv.txt")


# ==================================================================== ytol
def ytol_job(arg):
    kn, rel = arg
    warnings.filterwarnings("ignore")
    r = _run_b(kn, 120.0, keep=False, ytol_rel=rel)
    return dict(kn=kn, rel=rel, env=r["env"])


def part_ytol():
    from studies.parallel_runner import run_parallel
    from sim import planing_anyspeed as PA
    out("ytol (recorded): S1 (b) at 1.0 / 1.5 kn with the plant's "
        "ES_YTOL_REL (0: X segments merged only at equal Y) and the binned "
        "2e-3 first registered")
    args = [(kn, rel) for kn in (1.0, 1.5)
            for rel in (PA.ES_YTOL_REL, 2e-3)]
    res = run_parallel(ytol_job, args, gb_per_job=0.4, reserve_gb=3.0)
    rows = [r for r in res if not isinstance(r, Exception)]
    for r in rows:
        out(f"    {r['kn']:.1f} kn, ES_YTOL_REL {r['rel']:.0e}: verdict "
            f"{s1b_verdict(r['env'])}; envelope every 30 s "
            + " ".join(f"{a:.2e}" for a in r["env"][::3]))
    for kn in (1.0, 1.5):
        e = [np.array(r["env"]) for r in rows if r["kn"] == kn]
        if len(e) == 2:
            out(f"    {kn:.1f} kn: largest relative envelope difference "
                f"{float(np.max(np.abs(e[0] - e[1]) / e[0])):.2e}")
    dump("ytol.json", dict(rows=rows))
    flush("ytol.txt")


# =============================================================== radiation
def part_radiation():
    import capytaine as cpt
    import logging
    cpt.set_logging(logging.WARNING)
    out("radiation (recorded): the zero-speed 3D radiation power of the "
        "heave-pitch motion of S1 (b)'s 30-90 s trajectory, X memory off "
        "(P1c round 2), with Capytaine's B(omega) at the trajectory's main "
        "frequency (exact-geometry mesh of the P1b review, h 0.06 m, lid)")
    p = FX._plant(W, captive_u=0.0)
    body = C.capytaine_body_exact(p, 0.06, lid=True)
    solver = cpt.BEMSolver()
    rows = []
    for kn in (1.0, 1.5):
        r = _run_b(kn, 90.1, xmem=False)
        Z = r["rows"]
        dt, n0, n1 = 0.02, 1500, 4500
        z = np.array([x_["z_"] for x_ in Z])[n0:n1]
        W_ = np.array([x_["w_"] for x_ in Z])[n0:n1]
        Q_ = np.array([x_["q_"] for x_ in Z])[n0:n1]
        zc = z - z.mean()
        F = np.fft.rfft(zc * np.hanning(z.size))
        fr = np.fft.rfftfreq(z.size, dt) * 2 * np.pi
        w0 = float(fr[int(np.argmax(np.abs(F[1:]))) + 1])
        B = np.zeros((2, 2))
        for i, dof in enumerate(("Heave", "Pitch")):
            rr = solver.solve(cpt.RadiationProblem(
                body=body, omega=w0, radiating_dof=dof, rho=p.rho, g=G))
            for jj, d2 in enumerate(("Heave", "Pitch")):
                B[jj, i] = float(rr.radiation_dampings[d2])
        # Capytaine's pitch is the plant's th (z_e = Z + z + SP x th, SP =
        # -1; a rotation about +y moves (x, 0, 0) to z = -x theta)
        P = float(np.mean(B[0, 0] * W_ ** 2 + (B[0, 1] + B[1, 0]) * W_ * Q_
                          + B[1, 1] * Q_ ** 2))
        rows.append(dict(kn=kn, w0=w0, B=B.tolist(), P=P))
        out(f"    {kn:.1f} kn: main frequency {w0:.3f} rad/s; B33 "
            f"{B[0, 0]:.0f} N s/m, B35 {B[0, 1]:.0f}, B53 {B[1, 0]:.0f}, B55 "
            f"{B[1, 1]:.0f} N m s; radiation power on the trajectory "
            f"{P:.2f} W")
    dump("radiation.json", dict(rows=rows))
    flush("radiation.txt")


# ================================================================= goldenw3
def part_goldenw3():
    from studies.parallel_runner import run_parallel
    out("N2: the golden runs of P1a + wagner_entry (round 3: the X planes' "
        "memory), two processes")
    res = run_parallel(FX.golden_job, [(W, True), (W, True)],
                       gb_per_job=0.4, reserve_gb=3.0)
    a_, b_ = res
    if isinstance(a_, Exception) or isinstance(b_, Exception):
        check("goldenw3", "N2 wagner_entry: runs", False, f"{a_!r} {b_!r}")
        flush("goldenw3.txt")
        return
    same = set(a_) == set(b_) and all(
        np.array_equal(np.asarray(a_[k]), np.asarray(b_[k])) for k in a_)
    heel = max(float(np.degrees(np.abs(np.asarray(v)[:, 3]).max()))
               for k, v in a_.items()
               if k.startswith("calm_") and k.endswith("_s"))
    path = FX.golden_path(W, True).replace(".npz", "_p1c3.npz")
    old = FX.golden_path(W, True).replace(".npz", "_p1c2.npz")
    if os.path.exists(old):
        with np.load(old) as z:
            prev = {k: z[k] for k in z.files}
        diff = sorted(k for k in a_ if not np.array_equal(
            prev.get(k), np.asarray(a_[k])))
        out(f"    against _p1c2.npz: {len(diff)}/{len(a_)} arrays differ "
            f"{diff}")
    if os.path.exists(path):
        with np.load(path) as z:
            prev = {k: z[k] for k in z.files}
        wrote = "exists; " + ("identical" if set(prev) == set(a_) and all(
            np.array_equal(prev[k], np.asarray(a_[k])) for k in a_)
            else "DIFFERS (not overwritten)")
    else:
        np.savez(path, **{k: np.asarray(v) for k, v in a_.items()})
        wrote = "written"
    with open(path, "rb") as f:
        sha = hashlib.sha1(f.read()).hexdigest()
    out(f"    two processes identical {same}; largest calm heel {heel:.1f} "
        f"deg; {os.path.basename(path)} {wrote}, sha1 {sha}")
    check("goldenw3", "N2 wagner_entry: identical in two processes", same)
    check("goldenw3", "N2 wagner_entry: largest calm heel <= 15 deg",
          heel <= 15.0, f"{heel:.1f} deg")
    dump("goldenw3.json", dict(same=same, heel=heel, path=path, sha1=sha,
                               file=wrote))
    flush("goldenw3.txt")


PARTS = ("w19", "w1c", "w18iv", "ytol", "radiation", "goldenw3")


def _sub(arg):
    name, out_dir = arg
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"p1cw3_{name}_stdout.log"), "w") as log:
        rc = subprocess.run([sys.executable, "-m", "studies.anyspeed_p1c_w3",
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
        C.STAMP["p1c_w3_sha1"] = hashlib.sha1(f.read()).hexdigest()
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
