#!/usr/bin/env python3
"""
The checks registered after the P1b review (learn/meta/
ANYSPEED_PLAN_2026-10-05.md, 'P1b 复查后的修正', 运行前登记的检查),
before any of them ran. Helpers, stamps and report format of studies/
anyspeed_p1b_check.py (this module sets its OUT, STAMP and CAND); outputs
in studies/_cache/anyspeed/p1b/review_fix/ (or --out).

Parts (the register's numbers):
  w1      W1  exact transport of a two-valued state field (synthetic), CFL
              0.3-3
  w2      W2  a fixed-waterline block leaves the hull within its travel time
              +- one step (plant, calm, 4 / 12 / 25 kn, dt 0.02 / 0.01 /
              0.005)
  w3      W3  free run from rest at the 25-kn thrust, three steps, vs P1a
  w4      W4  regular head waves (lam 2 L, a 0.12 m) at 12 and 28 kn, dt
              0.02 / 0.01 / 0.005 (+ 0.0025 for the order): F share, mean
              speed, heave and pitch rms
  w5      W5  hidden state on a non-uniform field: replay from a recorded
              row, the derivative between steps, the reset of a step that
              does not continue the last one
  w6      W6  the census (S1 part g run on P1a + wagner_entry, its output
              directory --census) against P1a's, root by root
  w8      W8  energy at rest, dissipation off, with the consistent measure
  w9      W9  M2: the fixed-waterline planes' horizontal force against
              -d/dt sum m V tau, no table-cell exception
  cfree   W11 free runs from rest at five thrust levels (recorded)
  fkref   F1-F4 the 48-station plant against the continuum (1536 stations,
              same attitude), the continuum against an independent integral
              and against Capytaine on the exact mesh; the station-count
              order
  calmid  F6  calm-water bit-for-bit identity of fk_exact and fk_exact +
              orbital with P1a
  l1      L1  the lateral_phi0 Munk / Kirchhoff pair's power, ahead and
              astern; the shedding's power
  e1      E1  per switch set on a physical run (P0's interleaved method),
              with the largest heel and substep count

    .venv/Scripts/python.exe -m studies.anyspeed_p1b_fix --parts w1,w5
    .venv/Scripts/python.exe -m studies.anyspeed_p1b_fix --jobs w1,w2,...
"""
import argparse
import copy
import json
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

OUT = os.path.join(HERE, "_cache", "anyspeed", "p1b", "review_fix")
KN = C.KN
G = C.G
AS = C.AS
P1A = ("dDt_full", "chine_exact_all", "kpp_strip", "transom", "normal_rate",
       "verlet")
W = P1A + ("wagner_entry",)
FK = P1A + ("fk_exact",)
FO = P1A + ("fk_exact", "orbital")
LP = P1A + ("lateral_phi0",)
out, check, dump, flush = C.out, C.check, C.dump, C.flush


def _head_sea(h, a=0.12, lam_L=2.0):
    from sim.test_vessel import Monochromatic
    k = 2 * math.pi / (lam_L * h.L)
    return Monochromatic(math.sqrt(G * k), a, theta0=math.pi)


def _plant(sw, sea=None, dt=0.02, captive_u=None, record=False):
    from hydro import hulls
    from sim.planing_anyspeed import PlaningVesselAS
    return PlaningVesselAS(hulls.get(AS), C.calm() if sea is None else sea,
                           dt=dt, captive_u=captive_u, record=record,
                           test_switches=sw)


def _fshare(p, s, t):
    """The F share (mean F fraction over the wet stations) the plant's next
    step would start from at state s."""
    _, aux = p.deriv(s, t)
    w = aux["d"] > 0.0
    return float(aux["kb"].f[w].sum() / max(int(w.sum()), 1))


# ======================================================================= w1
def part_w1():
    from sim import planing_anyspeed as PA
    out("W1: exact transport of a two-valued state field (synthetic: every "
        "plane wet and entering, one block in state F, constant u_r)")
    n, dx = 48, 0.1125
    xs = (np.arange(n) + 0.5) * dx - 2.0
    edges = np.concatenate([[xs[0] - 0.5 * dx], xs + 0.5 * dx])
    lo, hi = float(edges[0]), float(edges[-1])
    d = np.full(n, 0.3)
    vg = np.full(n, 1.0)
    worst_b = worst_f = 0.0
    rows = []
    for cfl in (0.3, 0.67, 1.0, 1.5, 3.0):
        u0 = 5.0
        dt = cfl * dx / u0
        ur = np.full(n, u0)
        xa, xb = xs[30] - 0.013, xs[40] + 0.021
        b = np.array([xa, xb])
        st = np.array([PA.ES_E, PA.ES_F, PA.ES_E], np.int8)
        eb = ef = 0.0
        for i in range(200):
            b, st = PA._es_advance(b, st, xs, d, vg, ur, dt, edges)
            t = (i + 1) * dt
            a_c, b_c = max(xa - u0 * t, lo), min(xb - u0 * t, hi)
            segF = [(b[k - 1], b[k]) for k in range(1, b.size)
                    if st[k] == PA.ES_F]
            if b_c > a_c:
                if len(segF) != 1:
                    eb = np.inf
                else:
                    eb = max(eb, abs(segF[0][0] - a_c), abs(segF[0][1] - b_c))
                fx = np.clip(np.minimum(edges[1:], b_c)
                             - np.maximum(edges[:-1], a_c), 0.0, None) / dx
            else:
                if segF:
                    eb = np.inf
                fx = np.zeros(n)
            f = PA._es_frac(b, st, edges, PA.ES_F)
            ef = max(ef, float(np.abs(f - fx).max()))
        worst_b, worst_f = max(worst_b, eb), max(worst_f, ef)
        rows.append(dict(cfl=cfl, boundary=eb, fraction=ef))
        out(f"    CFL {cfl:4.2f}: largest boundary error {eb:.1e} m, "
            f"largest strip-fraction error {ef:.1e}")
    check("w1", "W1: two-valued field, CFL 0.3-3, 200 steps: state "
          "boundaries within 1e-9 m of x0 - u_r t, strip fractions within "
          "1e-9", worst_b <= 1e-9 and worst_f <= 1e-9,
          f"boundary {worst_b:.1e} m, fraction {worst_f:.1e}")
    dump("w1.json", dict(rows=rows))
    flush("w1.txt")


# ======================================================================= w2
def w2_job(arg):
    from sim import planing_anyspeed as PA
    kn, dt = arg
    warnings.filterwarnings("ignore")
    p = _plant(W, dt=dt)
    s = p.initial_state(kn * KN)
    thr = s[12]
    t = 0.0
    for i in range(int(round(2.0 / dt))):
        s = p.step(s, t, thr, 0.0)
        t += dt
    _, aux = p.deriv(s, t)
    dd = aux["d"]
    eg = p._es_geom()
    wet = np.nonzero(dd > 0.0)[0]
    # a run of wet strips from the third wet strip aft
    j0 = int(wet[2])
    j1 = j0
    while j1 + 1 < dd.size and dd[j1 + 1] > 0.0 and j1 - j0 < 12:
        j1 += 1
    xa, xb = float(eg.edges[j0]), float(eg.edges[j1 + 1])
    b = np.array([xa, xb])
    st = np.array([PA.ES_E, PA.ES_F, PA.ES_E], np.int8)
    # the injected field is the one the last step used: at t - dt
    p.set_hidden_state(dict(entry_state=(b, st), s=s))
    us, n_clear = [], None
    t_run = 1.5 * (xb - eg.lo) / max(float(s[6]), 0.1) + 1.0
    for k in range(int(round(t_run / dt))):
        us.append(float(s[6]))
        s = p.step(s, t, thr, 0.0)
        t += dt
        prev = p._es_prev
        fF = PA._es_frac(prev[0], prev[1], eg.edges, PA.ES_F)
        if n_clear is None and not np.any(fF > 0.0):
            n_clear = k
    # exact: the block's front plane passes the transom edge when the
    # distance travelled from t0 - dt reaches xb - lo (trapezoid of the
    # surge speed at the step starts; the speed one step before t0 taken
    # as the first)
    us = np.array([us[0]] + us)
    trav = np.concatenate([[0.0], np.cumsum(0.5 * (us[1:] + us[:-1]) * dt)])
    k_exact = int(np.argmax(trav >= xb - eg.lo)) - 1
    return dict(kn=kn, dt=dt, strips=(j0, j1), n_clear=n_clear,
                k_exact=k_exact, travel_time=(xb - eg.lo) / float(us.mean()))


def part_w2():
    from studies.parallel_runner import run_parallel
    out("W2: a block of fixed-waterline planes injected into steady calm "
        "planing at constant thrust leaves the hull within its travel time "
        "+- one step")
    args = [(kn, dt) for kn in (4.0, 12.0, 25.0) for dt in (0.02, 0.01,
                                                             0.005)]
    res = run_parallel(w2_job, args, gb_per_job=0.3, reserve_gb=3.0)
    ok = True
    for r in res:
        if isinstance(r, Exception):
            out(f"    job failed: {r!r}")
            ok = False
            continue
        good = r["n_clear"] is not None and abs(r["n_clear"]
                                                - r["k_exact"]) <= 1
        ok &= good
        out(f"    {r['kn']:4.1f} kn, dt {r['dt']}: strips {r['strips']}, "
            f"cleared at step {r['n_clear']}, exact {r['k_exact']} "
            f"(travel {r['travel_time']:.3f} s) {'ok' if good else 'FAIL'}")
    check("w2", "W2: the injected F block leaves the hull within its travel "
          "time +- one step at 4, 12, 25 kn and dt 0.02, 0.01, 0.005", ok)
    dump("w2.json", dict(rows=[r for r in res
                               if not isinstance(r, Exception)]))
    flush("w2.txt")


# ======================================================================= w3
def w3_job(arg):
    sw, dt, frac = arg
    warnings.filterwarnings("ignore")
    p = _plant(sw, dt=dt)
    thr = (p.initial_state(25.0 * KN)[12] if frac is None
           else frac * p.prop.t_max)
    s = p.initial_state(0.0)
    T = 60.0 if frac is None else 120.0
    n = int(round(T / dt))
    us = []
    t = 0.0
    for i in range(n):
        s = p.step(s, t, thr, 0.0)
        t += dt
        us.append(float(s[6]))
    last = int(round(10.0 / dt)) if frac is None else int(round(20.0 / dt))
    fs = _fshare(p, s, t) if "wagner_entry" in sw else 0.0
    return dict(sw="W" if "wagner_entry" in sw else "P1a", dt=dt,
                frac=frac, u_end=float(np.mean(us[-last:])) / KN,
                F_end=fs, resets=getattr(p, "es_resets", 0))


def part_w3():
    from studies.parallel_runner import run_parallel
    out("W3: free run from rest at the 25-kn thrust (60 s), P1a + "
        "wagner_entry vs P1a, dt 0.02 / 0.01 / 0.005")
    args = [(sw, dt, None) for dt in (0.02, 0.01, 0.005) for sw in (P1A, W)]
    res = run_parallel(w3_job, args, gb_per_job=0.3, reserve_gb=3.0)
    by = {(r["sw"], r["dt"]): r for r in res if not isinstance(r, Exception)}
    ok = len(by) == 6
    for dt in (0.02, 0.01, 0.005):
        a, b = by.get(("P1a", dt)), by.get(("W", dt))
        if a is None or b is None:
            continue
        rel = abs(b["u_end"] / a["u_end"] - 1)
        good = rel <= 5e-4 and b["F_end"] == 0.0
        ok &= good
        out(f"    dt {dt}: end speed P1a {a['u_end']:.5f} kn, + wagner_entry "
            f"{b['u_end']:.5f} kn (rel {rel:.1e}), end F share {b['F_end']}"
            f", resets {b['resets']}")
    check("w3", "W3: from rest at the 25-kn thrust, end speed within 0.05 % "
          "of P1a's at each dt, end F share 0", ok)
    dump("w3.json", dict(rows=list(by.values())))
    flush("w3.txt")


def part_cfree():
    from studies.parallel_runner import run_parallel
    out("W11 (S1 c, recorded): free runs from rest, 120 s, thrust 0.3 / 0.45 "
        "/ 0.55 / 0.7 / 1.0 x t_max, P1a + wagner_entry vs P1a (dt 0.02)")
    fr = (0.3, 0.45, 0.55, 0.7, 1.0)
    args = [(sw, 0.02, f) for f in fr for sw in (P1A, W)]
    res = run_parallel(w3_job, args, gb_per_job=0.3, reserve_gb=3.0)
    by = {(r["sw"], r["frac"]): r for r in res
          if not isinstance(r, Exception)}
    for f in fr:
        a, b = by.get(("P1a", f)), by.get(("W", f))
        if a and b:
            out(f"    thrust {f:4.2f} t_max: end speed P1a {a['u_end']:.4f} "
                f"kn, + wagner_entry {b['u_end']:.4f} kn (rel "
                f"{b['u_end'] / a['u_end'] - 1:+.1e}), end F share "
                f"{b['F_end']:.3f}")
    dump("cfree.json", dict(rows=list(by.values())))
    flush("cfree.txt")


# ======================================================================= w4
def w4_job(arg):
    sw, kn, dt = arg
    from hydro import hulls
    warnings.filterwarnings("ignore")
    p = _plant(sw, sea=_head_sea(hulls.get(AS)), dt=dt)
    s = p.initial_state(kn * KN)
    thr = s[12]
    n = int(round(60.0 / dt))
    every = int(round(0.02 / dt))
    S, F = [], []
    t = 0.0
    for i in range(n):
        s = p.step(s, t, thr, 0.0)
        t += dt
        if (i + 1) % every == 0:
            S.append(s.copy())
            if "wagner_entry" in sw:
                F.append(_fshare(p, s, t))
    S = np.array(S)
    k0 = len(S) // 3
    h2 = S[k0:]
    return dict(sw="W" if "wagner_entry" in sw else "P1a", kn=kn, dt=dt,
                F=float(np.mean(F[k0:])) if F else 0.0,
                u=float(h2[:, 6].mean() / KN), z=float(h2[:, 2].std()),
                th=float(np.degrees(h2[:, 4].std())),
                resets=getattr(p, "es_resets", 0))


def part_w4():
    from studies.parallel_runner import run_parallel
    out("W4: regular head waves (lam 2 L, a 0.12 m), constant thrust, 60 s "
        "(statistics over the last 40 s): F share, mean speed, heave and "
        "pitch rms by step")
    dts = (0.02, 0.01, 0.005, 0.0025)
    args = [(sw, kn, dt) for kn in (12.0, 28.0) for dt in dts
            for sw in (W, P1A)]
    res = run_parallel(w4_job, args, gb_per_job=0.3, reserve_gb=3.0)
    by = {(r["sw"], r["kn"], r["dt"]): r for r in res
          if not isinstance(r, Exception)}
    ok = True
    orders = {}
    for kn in (12.0, 28.0):
        for sw in ("W", "P1a"):
            for dt in dts:
                r = by.get((sw, kn, dt))
                if r:
                    out(f"    [{sw:3s}] {kn:4.1f} kn dt {dt:<6}: F share "
                        f"{r['F']:.4f}, speed {r['u']:.4f} kn, heave rms "
                        f"{1000 * r['z']:.3f} mm, pitch rms {r['th']:.4f} deg")
        rs = [by[("W", kn, dt)] for dt in dts[:3]]
        for q in ("F", "u", "z", "th"):
            v = [r[q] for r in rs]
            spread = max(abs(a / b - 1) for a in v for b in v if b != 0.0) \
                if all(x != 0.0 for x in v) else 0.0
            good = spread <= 0.01
            ok &= good
            v4 = [by[("W", kn, dt)][q] for dt in dts]
            d1, d2 = abs(v4[0] - v4[1]), abs(v4[1] - v4[2])
            d3 = abs(v4[2] - v4[3])
            od = [math.log2(a / b) if a > 0 and b > 0 else float("nan")
                  for a, b in ((d1, d2), (d2, d3))]
            orders[f"{kn:g}_{q}"] = od
            out(f"      {kn:4.1f} kn {q:2s}: largest relative difference over"
                f" dt 0.02/0.01/0.005 {100 * spread:.2f} % "
                f"{'ok' if good else 'FAIL'}; measured order "
                f"{od[0]:.2f}, {od[1]:.2f}")
    check("w4", "W4: F share, mean speed, heave and pitch rms within 1 % "
          "between dt 0.02, 0.01, 0.005 at 12 and 28 kn", ok)
    dump("w4.json", dict(rows=list(by.values()), orders=orders))
    flush("w4.txt")


# ======================================================================= w5
def part_w5():
    from hydro import hulls
    from sim import planing_anyspeed as PA
    out("W5 (c, d, e): the hidden state on a non-uniform field (12 kn, "
        "regular head waves)")
    sea = _head_sea(hulls.get(AS))
    p = _plant(W, sea=sea, record=True)
    s = p.initial_state(12 * KN)
    thr = s[12]
    rec_d, snap = [], {}
    calls = []
    orig = p.deriv

    def spy(s_, t_):
        r_ = orig(s_, t_)
        calls.append((s_.copy(), t_, r_[0].copy()))
        return r_
    p.deriv = spy
    t = 0.0
    S = [s.copy()]
    for i in range(1000):
        if i == 500:
            snap = dict(prop=copy.deepcopy(p.prop),
                        rudder=copy.deepcopy(p.rudder), t=t)
        if i == 700:
            # (d) the derivative between steps vs the next step's first
            calls.clear()
            k_between = orig(s, t)[0].copy()
            n0 = len(calls)
        s = p.step(s, t, thr, 0.0)
        if i == 700:
            k_first = calls[n0][2]
            same_d = bool(np.array_equal(k_between, k_first))
        t += 0.02
        S.append(s.copy())
    # back to the class's method (P1c round 3, track W): an instance
    # attribute 'deriv' (even the bound original) is copied with __dict__
    # into running_attitude's steady-state copy, whose pseudo-transient then
    # ran this free plant's derivative (trajectory() -> top_speed())
    del p.deriv
    tr = p.trajectory()
    hid = tr["hidden"]
    # (c) replay from row 500 with the recorded hidden state and the
    # actuators as they were then
    q = _plant(W, sea=sea, record=False)
    q.initial_state(12 * KN)
    q.prop, q.rudder = snap["prop"], snap["rudder"]
    q.set_hidden_state(hid[500])
    s2 = np.array(tr["s"][500], float)
    t2 = snap["t"]
    same_c = True
    for i in range(500, 1000):
        s2 = q.step(s2, t2, thr, 0.0)
        t2 += 0.02
        same_c &= bool(np.array_equal(s2, S[i + 1]))
    # how non-uniform the field was: F share and number of segments at row
    # 500
    b5, st5 = hid[500]["entry_state"]
    nseg = int(st5.size)
    _, aux = q.deriv(S[1000], t2)
    fsh = float(aux["kb"].f[aux["d"] > 0].mean())
    # (e) a step from a state that is not the continuation
    r0 = q.es_resets
    s3 = S[1000].copy()
    s3[2] += 1e-9
    calls2 = []
    o2 = q.deriv

    def spy2(s_, t_):
        r_ = o2(s_, t_)
        calls2.append(r_[1])
        return r_
    q.deriv = spy2
    q.step(s3, t2, thr, 0.0)
    q.deriv = o2
    aux1 = calls2[0]
    kb = aux1["kb"]
    eg = q._es_geom()
    b_sc, st_sc = PA._es_scan(q.x, aux1["d"], kb.Vg, float(s3[6]),
                              eg.edges)
    same_e = (q.es_resets == r0 + 1 and np.array_equal(kb.es[0], b_sc)
              and np.array_equal(kb.es[1], st_sc))
    out(f"    field at row 500: {nseg} segments, F share at the end "
        f"{fsh:.3f}; replay rows 500-1000 bit for bit: {same_c}; derivative"
        f" between steps == next step's first: {same_d}; non-continuation "
        f"step: resets +{q.es_resets - r0}, steady field used: {same_e}")
    check("w5", "W5 (c) replay from a recorded row (state, hidden state, "
          "actuators) reproduces the run bit for bit", same_c)
    check("w5", "W5 (d) the derivative between steps equals the next step's"
          " first derivative bit for bit", same_d)
    check("w5", "W5 (e) a step from a state that is not the last step's "
          "result counts a reset and starts from the steady field", same_e)
    dump("w5.json", dict(segments=nseg, F_end=fsh, replay=same_c,
                         between=same_d, reset=bool(same_e)))
    flush("w5.txt")


# ======================================================================= w6
CENSUS = os.path.join(OUT, "s1_w")
P1A_CENSUS = os.path.join(HERE, "_cache", "anyspeed", "p1a", "review_fix",
                          "s1_rerun", "g_census.json")


def part_w6():
    """The census of S1 part g (the same functions: census, winding, the
    same speeds) on P1a + wagner_entry only -- S1's part g re-runs v0 too,
    half its cost -- against P1a's g_census.json (the P1a review's S1
    re-run), root by root; and the captive derivative at every winding
    grid point of every speed equal to P1a's bit for bit."""
    from sim import planing_anyspeed as PA
    from studies import anyspeed_s1_check as S1_
    out("W6: the census (S1 part g's census and winding scan) on P1a + "
        f"wagner_entry against P1a's ({P1A_CENSUS}), root by root")
    with open(P1A_CENSUS) as f:
        b_ = json.load(f)
    PA.PRESETS["as1"] = PA.resolve("as1", W)
    p = S1_.build("as1")
    PA.PRESETS["as1"] = PA.resolve("as1", P1A)
    q = S1_.build("as1")
    kns = ([float(k) for k in np.arange(1.0, 9.51, 0.5)]
           + [10.0, 12.0, 16.0, 20.0, 25.0]
           + [float(k) for k in range(26, 41)])
    diffs, n_bit, n_pts, cen = [], 0, 0, {}
    for kn in kns:
        u = kn * KN
        res, log, ne = S1_.census(p, u)
        cells = S1_.winding(p, u)
        free_cells = []
        for (zc, tc, deg) in cells:
            near = any(abs(r["z"] - zc) <= 0.0151
                       and abs(r["trim"] - tc) <= 0.376 for r in res)
            if not near:
                free_cells.append((round(zc, 4), round(tc, 3), deg))
        cen[f"as1_{kn:g}"] = dict(roots=res, winding=cells,
                                  nonroot_zeros=free_cells)
        rb = b_[f"as1_{kn:g}"]
        A = [(r["z"], r["th"], r["label"]) for r in res]
        B = [(r["z"], r["th"], r["label"]) for r in rb["roots"]]
        same = len(A) == len(B) and all(
            x[2] == y[2] and abs(x[0] - y[0]) <= 1e-9
            and abs(x[1] - y[1]) <= 1e-9 for x, y in zip(A, B))
        zb = [tuple(z) for z in rb["nonroot_zeros"]]
        za = [tuple(z) for z in free_cells]
        if not same or za != zb or len(cells) != len(rb["winding"]):
            diffs.append((kn, dict(roots=(A, B), zeros=(za, zb))))
        # the captive equations themselves on the winding grid
        for z in np.linspace(-0.6, 0.3, 61)[::4]:
            for t in np.linspace(-15.0, 12.0, 73)[::4]:
                v = np.array([z, S1_.th_of(p, t)])
                n_pts += 1
                n_bit += bool(np.array_equal(S1_.F(p, u, v),
                                             S1_.F(q, u, v)))
        out(f"   {kn:4.1f} kn: {len(res)} roots "
            f"{[(round(r['z'], 4), round(r['trim'], 3), r['label']) for r in res]}"
            f", {len(free_cells)} zero(s) away from roots; same as P1a "
            f"{same and za == zb}")
    out(f"    speeds: {len(kns)}; differing from P1a: {len(diffs)} "
        f"{[d[0] for d in diffs][:8]}; captive residuals equal to P1a's "
        f"bit for bit at {n_bit} of {n_pts} grid points")
    check("w6", "W6: census roots (position, type), winding cells and zeros"
          " away from roots identical to P1a's at every speed 1-40 kn",
          not diffs, f"{len(diffs)} speeds differ")
    dump("w6.json", dict(diffs=diffs, census=cen, bitwise=[n_bit, n_pts]))
    flush("w6.txt")


# ======================================================================= w8
def energy_consistent(sw, dt, T, start):
    """Energy at rest, dissipation off, captive surge 0: rigid and fluid
    kinetic energy of heave and pitch on the sections' actual added mass
    (kb.ma: the flow states' mixture), roll on the plant's roll inertia
    Mtot[3, 3] (v0's lateral; the heel moves no sway or yaw), the
    hydrostatic potential of heave and pitch and the GZ potential."""
    from sim.planing_anyspeed import _vsec_q
    from sim.planing_vessel import SP
    from studies import anyspeed_p1a_check as P1A_
    p = _plant(sw, dt=dt, captive_u=0.0)
    p.c_dc = 0.0
    p.hull = copy.deepcopy(p.hull)
    p.hull.roll_zeta = 0.0
    v, _ = P1A_.eq_rest(p)
    rg = p.rho * G

    def U(z, th):
        d = -(p.Z_rest + z + p.zk + SP * p.x * (p.Th_rest + th))
        return float((rg * _vsec_q(d, p.yc, p.tanb)[0]).sum() * p.dx
                     + p.m * G * z)
    U0 = U(v[0], v[1])
    gzpot = P1A_._gz_pot_table(p)
    s = np.zeros(14)
    s[2], s[4], s[3] = v[0] + start[0], v[1] + start[1], start[2]
    n = int(round(T / dt))
    every = max(1, int(round(0.02 / dt)))
    E, tt = [], []
    for i in range(n + 1):
        if i % every == 0:
            _, aux = p.deriv(s, i * dt)
            ma = aux["kb"].ma
            A33 = ma.sum() * p.dx
            A35 = (ma * SP * p.x).sum() * p.dx
            A55 = (ma * p.x * p.x).sum() * p.dx
            Ev = 0.5 * ((p.m + A33) * s[8] ** 2 + 2 * A35 * s[8] * s[10]
                        + (p.Iy + A55) * s[10] ** 2)
            e = (Ev + 0.5 * p.Mtot[3, 3] * s[9] ** 2 + U(s[2], s[4]) - U0
                 + gzpot(s[3]))
            E.append(e)
            tt.append(i * dt)
        if i < n:
            s = p.step(s, i * dt, 0.0, 0.0)
            s[6] = 0.0
    E, tt = np.asarray(E), np.asarray(tt)
    w10 = tt <= 10.0
    l10 = tt >= tt[-1] - 10.0
    drift = (E[l10].mean() - E[w10].mean()) / E[0] / (T / 100.0)
    return dict(dt=dt, T=T, E0=float(E[0]), drift_per_100s=float(drift),
                max_dev=float(np.abs(E - E[0]).max() / E[0]),
                resets=p.es_resets)


def part_w8():
    out("W8: energy at rest, dissipation off, verlet, P1a's start (heave "
        "+0.01 m, trim +0.003, heel 0.05), the consistent measure")
    r = energy_consistent(W, 0.002, 100.0, (0.01, 0.003, 0.05))
    out(f"    dt 0.002: drift {100 * r['drift_per_100s']:+.5f} % per 100 s, "
        f"max deviation {100 * r['max_dev']:.4f} %, resets {r['resets']}")
    check("w8", "W8: drift <= 0.1 % per 100 s at dt 0.002",
          abs(r["drift_per_100s"]) <= 1e-3,
          f"{100 * r['drift_per_100s']:+.5f} % per 100 s")
    dump("w8.json", r)
    flush("w8.txt")


# ======================================================================= w9
def part_w9():
    from sim.planing_vessel import SP
    from studies import anyspeed_p1a_check as P1A_
    out("W9: at rest, surge held, dt 0.5 ms, 2 s from heave +5 mm / trim "
        "+0.1 deg: dynamic horizontal force (Xp - Xh) vs -d/dt sum kb.ma V "
        "tau dx, every sample (no table-cell exception)")
    q = _plant(W, dt=0.0005, captive_u=0.0)
    q.c_dc = 0.0
    v, _ = P1A_.eq_rest(_plant(W, captive_u=0.0))
    tab = q._tab()
    s = np.zeros(14)
    s[2], s[4] = v[0] + 0.005, v[1] + np.radians(0.1) / SP
    Gs, Xd, rr, fx = [], [], [], []
    for i in range(4001):
        _, aux = q.deriv(s, i * q.dt)
        tau = q.kp + SP * (q.Th_rest + s[4])
        Gs.append((aux["kb"].ma * aux["V"] * tau).sum() * q.dx)
        Xd.append(aux["Xp"] - aux["Xh"])
        rr.append(aux["d"] / (q.yc * tab.tb))
        fx.append(aux["kb"].f.copy())
        s = q.step(s, i * q.dt, 0.0, 0.0)
        s[6] = 0.0
    Gs, Xd = np.asarray(Gs), np.asarray(Xd)
    rr, fx = np.asarray(rr), np.asarray(fx)
    dG = (Gs[2:] - Gs[:-2]) / (2 * q.dt)
    rs = np.abs(Xd[1:-1] + dG) / np.abs(Xd).max()
    # samples whose difference spans a chine crossing (r = 1) of a strip
    # with fixed-waterline planes
    side = rr > 1.0
    ch = np.array([np.any((side[k - 1] != side[k + 1])
                          & ((fx[k - 1] > 0) | (fx[k + 1] > 0)))
                   for k in range(1, len(Gs) - 1)])
    other = rs[~ch]
    out(f"    {len(rs)} samples: largest {rs.max():.1e}; away from chine "
        f"crossings {other.max():.1e}; {int(ch.sum())} samples span a chine "
        f"crossing (largest there {rs[ch].max() if ch.any() else 0.0:.1e})")
    check("w9", "W9: dynamic horizontal force = -d/dt sum m V tau within "
          "1e-5 of its largest value at every sample not spanning a chine "
          "crossing (table cells included)", other.max() <= 1e-5,
          f"{other.max():.1e}; chine-crossing samples {int(ch.sum())}")
    dump("w9.json", dict(largest=float(rs.max()), elsewhere=float(
        other.max()), n_chine=int(ch.sum()),
        at_chine=float(rs[ch].max()) if ch.any() else 0.0))
    flush("w9.txt")


# ==================================================================== fkref
NS = (48, 96, 192, 384, 1536)


def _variant(n):
    from hydro import hulls
    from sim import physics
    if n == 48:
        return hulls.get(AS)
    h = copy.deepcopy(hulls.get("scarab195"))
    h.n_stations = n
    h.name = f"scarab195_n{n}"
    return physics.variant(h, "as1", name=h.name)


def part_fkref():
    import capytaine as cpt         # noqa: F401
    from hydro.hull_geometry import stations_of
    from hydro.section2d import section_fk
    from sim.planing_anyspeed import _vsec, PlaningVesselASLin
    from sim.planing_vessel import SP
    out("F1-F4 (3'): the zero-speed linear Froude-Krylov force of the "
        "48-station plant against the continuum (the same code at 1536 "
        "stations, same absolute attitude), the continuum against an "
        "independent integral and against Capytaine on the exact mesh")
    pcs = {n: PlaningVesselASLin(_variant(n), C.calm(), dt=0.02,
                                 captive_u=0.0, test_switches=FK)
           for n in NS}
    p48 = pcs[48]
    Z0, T0 = p48.Z_rest, p48.Th_rest
    L, dx = p48.L, p48.dx
    lam_list = [8 * dx, 12 * dx, 16 * dx, 0.25 * L, 0.5 * L, L, 2 * L,
                4 * L, 8 * L]
    heads = [0, 30, 60, 90, 120, 150, 180]

    def st(n):
        s = np.zeros(14)
        s[2], s[4] = Z0 - pcs[n].Z_rest, T0 - pcs[n].Th_rest
        return s
    Fp = {n: [] for n in NS}
    t0 = time.time()
    for lw in lam_list:
        k = 2 * math.pi / lw
        om = math.sqrt(G * k)
        for n in NS:
            Fp[n].append(np.array([C.plant_lin_fk(pcs[n], st(n), om,
                                                  math.radians(h), a=1e-6)
                                   for h in heads]))
    out(f"    plant forces: {time.time() - t0:.0f} s")
    rows, worst, zero_bad = [], 0.0, 0.0
    errs = {n: [] for n in NS[:-1]}
    for il, lw in enumerate(lam_list):
        ref = Fp[1536][il]
        sc = np.maximum(np.abs(ref).max(axis=0), 1e-300)
        row = dict(lam=lw, lamL=lw / L)
        for n in NS[:-1]:
            e_ = (np.abs(Fp[n][il] - ref) / sc).max(axis=0)
            errs[n].append(e_)
            row[f"err_{n}"] = e_
        worst = max(worst, float(row["err_48"].max()))
        for ih, hd in enumerate(heads):
            if hd in (0, 180):
                zero_bad = max(zero_bad, float(np.max(
                    np.abs(Fp[48][il][ih, [1, 3, 5]]) / sc[[1, 3, 5]])))
        rows.append(row)
        out(f"    lam {lw:6.2f} m (lam/L {lw / L:5.2f}): 48 stations vs "
            f"continuum, X Y Z K M N: "
            + " ".join(f"{100 * v:5.2f}%" for v in row["err_48"])
            + "; 96 / 192 / 384 largest "
            + " / ".join(f"{100 * row[f'err_{n}'].max():.3f}%"
                         for n in (96, 192, 384)))
    order = []
    for il in range(len(lam_list)):
        e1 = np.array([errs[n][il].max() for n in (48, 96, 192, 384)])
        order.append([math.log2(e1[i] / e1[i + 1]) if e1[i + 1] > 0 else
                      float("nan") for i in range(3)])
    out("    measured order by station count (largest DOF error; 48->96, "
        "96->192, 192->384): " + "; ".join(
            f"lam/L {lam_list[i] / L:.2f}: "
            + ", ".join(f"{o:.2f}" for o in order[i])
            for i in range(len(lam_list))))
    check("fkref", "F1 (3'): 48-station plant vs the continuum, 6 DOF, lam 8"
          " station spacings to 8 L, headings 0-180: within 3 % of each "
          "DOF's largest amplitude over the headings", worst <= 0.03,
          f"largest {100 * worst:.2f} %")
    check("fkref", "F5 (3, symmetric zeros): sway, roll, yaw in head and "
          "following waves <= 1e-12 of the scale", zero_bad <= 1e-12,
          f"{zero_bad:.1e}")
    # F2: the continuum's X against the independent integral
    p1 = pcs[1536]
    h = p48.hull
    g = stations_of(h)
    rg = p48.rho * G
    N = 400000
    xs = (np.arange(N) + 0.5) * h.L / N
    xb = xs - h.lcg
    dxc = h.L / N
    dc = -(Z0 + (-h.vcg + g.keel(xs)[0]) + SP * xb * T0)
    ycc = g.chine(xs)
    f2 = []
    for lamL, thd in ((8 * dx / L, 0), (8 * dx / L, 60), (0.5, 0)):
        k = 2 * math.pi / (lamL * L)
        om = math.sqrt(G * k)
        th = math.radians(thd)
        kap, lam = k * math.cos(th), k * math.sin(th)
        E = np.array(section_fk(k, lam, dc, ycc, p48.tanb)[0])
        Xc = -(rg * (1j * kap) * E * np.exp(1j * kap * xb)).sum() * dxc
        X1 = C.plant_lin_fk(p1, st(1536), om, th, a=1e-6)[0]
        X48 = C.plant_lin_fk(p48, st(48), om, th, a=1e-6)[0]
        f2.append(dict(lamL=lamL, head=thd, rel_1536=float(abs(X1 - Xc)
                                                           / abs(Xc)),
                       rel_48=float(abs(X48 - Xc) / abs(Xc))))
        out(f"    F2 lam/L {lamL:.3f} heading {thd:2d}: X at 1536 stations "
            f"vs the independent continuum integral "
            f"{100 * f2[-1]['rel_1536']:.4f} %, 48 stations "
            f"{100 * f2[-1]['rel_48']:.3f} % (before the review: 5.59, 3.12,"
            f" 3.67 %)")
    check("fkref", "F2: the continuum (1536 stations) X vs the independent "
          "continuum integral within 0.1 %",
          max(r["rel_1536"] for r in f2) <= 1e-3,
          f"largest {100 * max(r['rel_1536'] for r in f2):.4f} %")
    # F3: the exact mesh
    Ac = _vsec(dc, ycc, p48.tanb)[0]
    Bc = np.where(dc > 0, 2 * np.minimum(dc / p48.tanb, ycc), 0.0)
    cont = (Ac.sum() * dxc, (Ac * xb).sum() / Ac.sum(), Bc.sum() * dxc,
            (Bc * xb).sum() / Bc.sum())
    bodies = {}
    geo_ok = True
    for hm in (0.03, 0.02):
        b = C.capytaine_body_exact(p48, hm)
        bodies[hm] = b
        mm = (float(b.volume), float(b.center_of_buoyancy[0]),
              float(b.waterplane_area), float(b.waterplane_center[0]))
        okg = (abs(mm[0] / cont[0] - 1) <= 1e-3
               and abs(mm[1] - cont[1]) <= 5e-4
               and abs(mm[3] - cont[3]) <= 5e-4)
        geo_ok &= okg
        out(f"    F3 mesh h {hm}: {b.mesh.nb_faces} panels; volume "
            f"{mm[0]:.5f} (continuum {cont[0]:.5f}) m3, centre of buoyancy "
            f"{1000 * mm[1]:+.2f} ({1000 * cont[1]:+.2f}) mm, waterplane "
            f"{mm[2]:.5f} ({cont[2]:.5f}) m2, its centre {1000 * mm[3]:+.2f} "
            f"({1000 * cont[3]:+.2f}) mm")
    check("fkref", "F3: the exact mesh's volume within 0.1 %, centre of "
          "buoyancy and waterplane centre within 0.5 mm of the continuum's",
          geo_ok)
    # F4: continuum vs Capytaine
    cog = (0.0, 0.0, Z0)
    w4, wref = 0.0, 0.0
    f4 = []
    for il, lw in enumerate(lam_list):
        k = 2 * math.pi / lw
        om = math.sqrt(G * k)
        Fc = np.array([C.cap_fk_q(bodies[0.02], om, math.radians(h_),
                                  p48.rho, cog) for h_ in heads])
        Fc2 = np.array([C.cap_fk_q(bodies[0.03], om, math.radians(h_),
                                   p48.rho, cog) for h_ in heads])
        ref = Fp[1536][il]
        sc = np.maximum(np.abs(Fc).max(axis=0), 1e-300)
        e_ = (np.abs(ref - Fc) / sc).max(axis=0)
        er = (np.abs(Fc2 - Fc) / sc).max(axis=0)
        e48 = (np.abs(Fp[48][il] - Fc) / sc).max(axis=0)
        w4, wref = max(w4, float(e_.max())), max(wref, float(er.max()))
        f4.append(dict(lam=lw, err=e_, ref_conv=er, err_48=e48))
        out(f"    F4 lam {lw:6.2f} m: continuum vs Capytaine, X Y Z K M N: "
            + " ".join(f"{100 * v:5.2f}%" for v in e_)
            + f" (meshes differ {100 * er.max():.3f} %; 48-station plant vs"
            f" Capytaine largest {100 * e48.max():.2f} %)")
    check("fkref", "F4: the continuum vs Capytaine on the exact mesh within "
          "3 % of each DOF's largest amplitude over the headings",
          w4 <= 0.03, f"largest {100 * w4:.2f} %")
    check("fkref", "F4: Capytaine's two meshes within 0.3 %", wref <= 3e-3,
          f"{100 * wref:.3f} %")
    dump("fkref.json", dict(rows=rows, order=order, f2=f2,
                            geometry=dict(continuum=cont), f4=f4))
    flush("fkref.txt")


# =================================================================== calmid
def part_calmid():
    from hydro import hulls
    from sim import planing_anyspeed as PA
    out("F6: calm water, fk_exact and fk_exact + orbital vs P1a, bit for "
        "bit (derivatives at 50 random states, 200-step runs; GZ and "
        "linear-roll plants)")
    ok = True
    rows = []
    for sw in (FK, FO):
        for gz in (True, False):
            cls = PA.PlaningVesselAS if gz else PA.PlaningVesselASLin
            h = hulls.get(AS)
            a = cls(h, C.calm(), dt=0.02, test_switches=P1A)
            b = cls(h, C.calm(), dt=0.02, test_switches=sw)
            rng = np.random.default_rng(0)
            okd = True
            for i in range(50):
                s = np.zeros(14)
                s[2] = rng.uniform(-0.2, 0.2)
                s[3] = rng.uniform(-0.3, 0.3)
                s[4] = rng.uniform(-0.05, 0.05)
                s[6] = rng.uniform(-2.0, 18.0)
                s[7:12] = rng.uniform(-0.5, 0.5, 5)
                s[12], s[13] = 3000.0, 0.1
                okd &= bool(np.array_equal(a.deriv(s, 0.3)[0],
                                           b.deriv(s, 0.3)[0]))
            sa = a.initial_state(20 * KN)
            sb = b.initial_state(20 * KN)
            for i in range(200):
                c_ = 0.2 * math.sin(0.05 * i)
                sa = a.step(sa, i * 0.02, sa[12] * 1.1, c_)
                sb = b.step(sb, i * 0.02, sb[12] * 1.1, c_)
            okr = bool(np.array_equal(sa, sb))
            ok &= okd and okr
            rows.append(dict(sw=sw[-1], gz=gz, deriv=okd, run=okr))
            out(f"    + {'+'.join(sw[6:]):17s} {'GZ' if gz else 'linear'}: "
                f"derivatives {okd}, run {okr}")
    check("calmid", "F6: calm-water derivatives and runs of P1a + fk_exact "
          "(+ orbital) equal P1a's bit for bit", ok)
    dump("calmid.json", dict(rows=rows))
    flush("calmid.txt")


# ======================================================================= l1
def part_l1():
    out("L1: lateral_phi0's convective terms -- the Munk moment and "
        "Kirchhoff's surge partner do no work together, ahead and astern; "
        "the transom shedding (ahead only) does none or dissipates")
    p = _plant(LP, captive_u=0.0)
    rng = np.random.default_rng(3)
    worst, shed_max, rows = 0.0, -np.inf, []
    for u in (1.5, -1.5):
        for i in range(20):
            s = np.zeros(14)
            s[2] = rng.uniform(-0.05, 0.05)
            s[3] = rng.uniform(-0.2, 0.2)
            s[4] = rng.uniform(-0.01, 0.01)
            s[6] = u
            s[7], s[9], s[11] = rng.uniform(-0.3, 0.3, 3)
            _, aux = p.deriv(s, 0.0)
            kb = aux["kb"]
            lat1 = p._lateral_b(s, aux["d"], kb)
            kb0 = copy.copy(kb)
            kb0.ur = np.zeros_like(kb.ur)
            kb0.urT = 0.0
            lat0 = p._lateral_b(s, aux["d"], kb0)
            dY = lat1.tau[1] - lat0.tau[1]
            dK = lat1.tau[3] - lat0.tau[3]
            dN = lat1.tau[5] - lat0.tau[5]
            x0 = p.x[0]
            nm = dN - x0 * dY                       # the Munk moment
            Xk = lat1.tau[0]
            pw = nm * s[11] + Xk * u
            scale = abs(nm * s[11]) + abs(Xk * u) + 1e-300
            worst = max(worst, abs(pw) / scale)
            psh = dY * s[7] + x0 * dY * s[11] + dK * s[9]
            shed_max = max(shed_max, psh)
            rows.append(dict(u=u, pair=pw / scale, shed=psh,
                             astern_shed=(dY, dK) if u < 0 else None))
    astern_zero = all(r["astern_shed"] == (0.0, 0.0) for r in rows
                      if r["u"] < 0)
    out(f"    pair power / scale largest {worst:.1e}; shedding power "
        f"largest {shed_max:.3e} W; astern shedding exactly zero "
        f"{astern_zero}")
    check("l1", "L1: Munk + Kirchhoff power zero (1e-12 of scale) ahead and "
          "astern; shedding power <= 0 and none astern",
          worst <= 1e-12 and shed_max <= 1e-9 and astern_zero,
          f"{worst:.1e}, {shed_max:.2e} W")
    dump("l1.json", dict(rows=rows))
    flush("l1.txt")


# ======================================================================= e1
def part_e1():
    from hydro import hulls
    from learn.repro import task
    from sim import physics
    from sim import planing_anyspeed as PA
    from studies import anyspeed_p0_check as P0
    out("E1 by switch set: ms per 0.02 s step of the GZ plant in the task sea"
        " (mission s0 leg 0), 0.9 u_design, thrust 0.6 +- 0.3 t_max, nozzle "
        "0.5 x max sinusoid (P0's run); P0's interleaved rounds, ratio to v0 "
        "x v0's idle cost; the largest heel and substep count of the runs")
    m = task.Mission("high", 0, 0, t_end=1.0, track=task.LEGS[0])
    sea = m.plant.sea
    sets = dict(P1a=P1A, fk_exact=FK, fk_orbital=FO, wagner_entry=W)
    plants = {"v0": physics.gz_twin(hulls.get("scarab195").plant(sea,
                                                                 dt=0.02))}
    for k, sw in sets.items():
        plants[k] = PA.PlaningVesselAS(hulls.get(AS), sea, dt=0.02,
                                       test_switches=sw)
    keys = list(plants)
    N, rounds = 300, 9
    per = {k: [] for k in keys}
    heel = {k: 0.0 for k in keys}
    nsub = {k: 0 for k in keys}
    for r in range(rounds):
        order_ = keys[r % len(keys):] + keys[:r % len(keys)]
        for k in order_:
            p = plants[k]
            s_ = p.initial_state(0.9 * float(p.hull.u_design))
            tm, rm = p.prop.t_max, p.rudder.max
            ta = time.perf_counter()
            hh, ns = 0.0, 0
            for i in range(N):
                s_ = p.step(s_, i * 0.02, tm * (0.6 + 0.3 * math.sin(
                    0.05 * i)), rm * 0.5 * math.sin(0.03 * i), 0.02)
                hh = max(hh, abs(s_[3]))
                ns = max(ns, p.last_n_sub)
            per[k].append(1e3 * (time.perf_counter() - ta) / N)
            heel[k] = max(heel[k], math.degrees(hh))
            nsub[k] = max(nsub[k], ns)
    med = {k: float(np.median(v)) for k, v in per.items()}
    rat = {k: float(np.median(np.array(per[k]) / np.array(per["v0"])))
           for k in keys[1:]}
    est = {k: rat[k] * P0.V0_IDLE_MS for k in rat}
    for k in keys:
        out(f"    {k:<13} median {med[k]:.3f} ms" + (
            f", / v0 {rat[k]:.3f} -> idle estimate {est[k]:.2f} ms"
            if k in rat else "") + f"; largest heel {heel[k]:.1f} deg, "
            f"substeps {nsub[k]}")
    for k in ("fk_exact", "fk_orbital", "wagner_entry"):
        check("e1", f"E1: P1a + {k} at most {P0.E1_MS} ms per step",
              est[k] <= P0.E1_MS, f"{est[k]:.2f} ms (P1a {est['P1a']:.2f})")
    dump("e1.json", dict(per_round_ms=per, median_ms=med, ratio_to_v0=rat,
                         idle_estimate_ms=est, heel_deg=heel, n_sub=nsub))
    flush("e1.txt")



# ======================================================================= f7
def f7_job(arg):
    """One run of the wave parts of S1's standing checks for switch set sw:
    ('fol', scale, kn) S1 f's following-sea open loop, or ('p6', case, lam)
    S1 d's Fridsma P6 towed regular-wave run."""
    kind, sw, a1, a2 = arg
    warnings.filterwarnings("ignore")
    from hydro import hulls
    from sim import physics
    from sim.planing_anyspeed import PlaningVesselAS
    from studies import anyspeed_s1_check as S1_
    from studies import exp_planing as EP
    if kind == "fol":
        from learn.repro import task
        m = task.Mission("high", 0, 0, t_end=1.0, track=task.LEGS[0])
        sea = copy.deepcopy(m.plant.sea)
        sea.th = np.asarray(sea.th, float) + np.pi
        sea.a = np.asarray(sea.a, float) * a1
        p = PlaningVesselAS(hulls.get(AS), sea, dt=m.plant.dt,
                            test_switches=sw)
        r = S1_.open_loop(p, a2)
        return dict(kind=kind, sw=sw, scale=a1, kn=a2, **r)
    h, V = EP.fridsma_hull(a1, tow="nozzle")
    hv = physics.variant(h, "as1")

    def make(hull, sea, dt, V_):
        return PlaningVesselAS(hv, sea, dt=dt, captive_u=V_,
                               test_switches=sw)
    r1 = S1_.regular_wave_on(make, h, V, a2, 0.111 * h.b, T=16.0, dt=0.004)
    return dict(kind=kind, sw=sw, case=a1, lam=a2, res=r1,
                measured=EP.FRIDSMA[a1][a2])


def part_f7():
    from studies.parallel_runner import run_parallel
    from studies import exp_planing as EP
    out("F7: the wave parts of the standing checks for P1a + fk_exact and P1a"
        " + fk_exact + orbital, beside P1a: S1 f's following seas (task sea "
        "turned 180 deg, Hs x1 and x1.5, 12/20/25/30 kn, 240 s open loop; "
        "registered: no divergence, no capsize) and S1 d's Fridsma P6 "
        "(recorded)")
    sets = (("P1a", P1A), ("fk_exact", FK), ("fk_orbital", FO))
    args = [("fol", sw, sc, kn) for _, sw in sets for sc in (1.0, 1.5)
            for kn in (12, 20, 25, 30)]
    args += [("p6", sw, case, lam) for _, sw in sets for case in ("A", "B")
             for lam in EP.FRIDSMA[case]]
    res = run_parallel(f7_job, args, gb_per_job=0.3, reserve_gb=3.0)
    name = {tuple(sw): n for n, sw in sets}
    ok = True
    for r in res:
        if isinstance(r, Exception):
            out(f"    job failed {r!r}")
            ok = False
            continue
        r["set"] = name[tuple(r["sw"])]
    fol = [r for r in res if isinstance(r, dict) and r["kind"] == "fol"]
    for r in fol:
        bad = r["diverged_t"] is not None or r["capsized"]
        if r["set"] != "P1a":
            ok &= not bad
        out(f"    following Hs x{r['scale']} {r['kn']:2d} kn {r['set']:10s}: "
            + ("DIVERGED" if r["diverged_t"] is not None else "ok")
            + f", capsized {r['capsized']}, min trim {r['min_trim']:+.2f} "
            f"deg, min z {r['min_z']:+.3f} m, end {r['end_kn']:.2f} kn")
    p6 = [r for r in res if isinstance(r, dict) and r["kind"] == "p6"]
    ref = {(r["case"], r["lam"]): r["res"] for r in p6 if r["set"] == "P1a"}
    for r in p6:
        if r["set"] == "P1a":
            continue
        a_ = ref[(r["case"], r["lam"])]
        out(f"    P6 case {r['case']} lam/L {r['lam']}: {r['set']:10s} heave "
            f"h/H {r['res'][0]:.4f} (P1a {a_[0]:.4f}), pitch {r['res'][1]:.4f}"
            f" (P1a {a_[1]:.4f}), CG g {r['res'][2]:.3f} ({a_[2]:.3f}), bow g "
            f"{r['res'][3]:.3f} ({a_[3]:.3f}); measured {r['measured'][:2]}")
    check("f7", "F7: following seas Hs x1 and x1.5 at 12-30 kn: no "
          "divergence, no capsize (P1a + fk_exact, + orbital)", ok)
    dump("f7.json", dict(rows=[r for r in res if isinstance(r, dict)]))
    flush("f7.txt")


# ==================================================================== quick
def part_quick():
    """The quick tier of the P1b switches (T1, about 8 minutes with
    studies/anyspeed_regress_as.py --freeze --golden beside it): W1, L1, F6,
    W5 (c, d, e), W2 at 25 kn and dt 0.02, R3 from heave +5 cm on P1a +
    wagner_entry (600 s)."""
    from studies import anyspeed_p1a_check as P1A_
    part_w1()
    part_l1()
    part_calmid()
    part_w5()
    out("QUICK: W2 at 25 kn, dt 0.02; R3 from heave +5 cm (P1a + "
        "wagner_entry)")
    r = w2_job((25.0, 0.02))
    good = r["n_clear"] is not None and abs(r["n_clear"] - r["k_exact"]) <= 1
    out(f"    W2 25 kn dt 0.02: cleared at step {r['n_clear']}, exact "
        f"{r['k_exact']}")
    check("quick", "W2 (25 kn, dt 0.02): the F block leaves within its "
          "travel time +- one step", good)
    p0 = _plant(W)
    v, _ = P1A_.eq_rest(_plant(W, captive_u=0.0))
    s0 = np.zeros(14)
    s0[2] = v[0] + 0.05
    rr = P1A_.zero_thrust_run(p0, s0)
    out(f"    R3 heave +5 cm: u(600 s) {rr['u_end']:+.2e} m/s, distance "
        f"{rr['x_end']:+.3f} m")
    check("quick", "R3: |u| < 5 mm/s after 600 s at zero thrust from heave "
          "+5 cm", abs(rr["u_end"]) < 5e-3, f"{rr['u_end']:+.2e} m/s")
    flush("quick.txt")


# =================================================================== lorder
def part_lorder():
    """Minor review item (item 9, lateral energy): the measured order of the
    lateral energy error of P1a + lateral_phi0 at rest, dissipation off,
    over four step sizes (0.004, 0.002, 0.001, 0.0005 s), 20 s from heel 5
    deg, sway 0.2 m/s, yaw 0.05 rad/s (studies/anyspeed_p1b_check.
    lateral_energy_run's measure); recorded (lateral_phi0 is not a candidate
    before S4 / S5)."""
    C.CAND = LP
    out("lateral energy error by step (P1a + lateral_phi0, recorded): max "
        "deviation from E0 and end-minus-start drift over 20 s")
    rows = []
    for dt in (0.004, 0.002, 0.001, 0.0005):
        t0 = time.time()
        r = C.lateral_energy_run(dt, 20.0, (math.radians(5.0), 0.2, 0.05))
        r["seconds"] = time.time() - t0
        rows.append(r)
        out(f"    dt {dt}: max deviation {100 * r['max_dev']:.5f} %, drift "
            f"{100 * r['drift_per_100s']:+.5f} % per 100 s "
            f"({r['seconds']:.0f} s)")
    md = [r["max_dev"] for r in rows]
    od = [math.log2(md[i] / md[i + 1]) for i in range(3)]
    out(f"    measured order of the max deviation: "
        + ", ".join(f"{o:.2f}" for o in od))
    dump("lorder.json", dict(rows=rows, order=od))
    flush("lorder.txt")


# ================================================================== goldens
GOLDENS = (("fk_exact", FK, True), ("fk_orbital", FO, True),
           ("wagner_entry", W, True), ("lateral_phi0", LP, False))
HEEL_MAX_DEG = 15.0


def golden_job(arg):
    """anyspeed_regress.golden_runs on PlaningVesselAS with switches sw (the
    runs anyspeed_regress_as's GOLDEN_SETS re-runs), nozzle held at zero if
    not nozzle."""
    sw, nozzle = arg
    from hydro import hulls
    from sim.planing_anyspeed import PlaningVesselAS
    from studies import anyspeed_regress as AR
    from studies.anyspeed_regress_as import ON_EXTRA
    warnings.filterwarnings("ignore")
    h = hulls.get(AS)
    return AR.golden_runs(lambda p: PlaningVesselAS(
        h, p.sea, dt=p.dt, record=True, test_switches=sw), extra=ON_EXTRA,
        nozzle=nozzle)


def golden_path(sw, nozzle):
    return os.path.join(HERE, "_cache", "anyspeed", "golden_as_"
                        + "+".join(sw) + ("" if nozzle else "_nozzle0")
                        + ".npz")


def part_goldens():
    """N2: each switch's 'on' golden (P1b review: one per switch; the
    candidate's golden fixed a chaotic capsizing run): two processes, bit
    for bit; the largest heel of the calm runs <= HEEL_MAX_DEG; written to
    studies/_cache/anyspeed/ (never overwritten), sha1 printed for
    studies/anyspeed_regress_as.GOLDEN_SETS."""
    import hashlib
    from studies.parallel_runner import run_parallel
    out("N2: per-switch goldens (anyspeed_regress.golden_runs on "
        "PlaningVesselAS), two processes each")
    args = [(sw, nz) for _, sw, nz in GOLDENS for _ in range(2)]
    res = run_parallel(golden_job, args, gb_per_job=0.4, reserve_gb=3.0)
    rows = []
    for i, (name, sw, nz) in enumerate(GOLDENS):
        a_, b_ = res[2 * i], res[2 * i + 1]
        if isinstance(a_, Exception) or isinstance(b_, Exception):
            check("goldens", f"{name}: runs", False, f"{a_!r} {b_!r}")
            continue
        same = set(a_) == set(b_) and all(
            np.array_equal(np.asarray(a_[k]), np.asarray(b_[k])) for k in a_)
        heel = max(float(np.degrees(np.abs(np.asarray(v)[:, 3]).max()))
                   for k, v in a_.items()
                   if k.startswith("calm_") and k.endswith("_s"))
        heel_t = max(float(np.degrees(np.abs(np.asarray(v)[:, 3]).max()))
                     for k, v in a_.items()
                     if k.startswith("task_") and k.endswith("_s"))
        path = golden_path(sw, nz)
        if os.path.exists(path):
            with np.load(path) as z:
                old = {k: z[k] for k in z.files}
            wrote = "exists; " + ("identical" if set(old) == set(a_) and all(
                np.array_equal(old[k], np.asarray(a_[k])) for k in a_)
                else "DIFFERS (not overwritten)")
        else:
            np.savez(path, **{k: np.asarray(v) for k, v in a_.items()})
            wrote = "written"
        with open(path, "rb") as f:
            sha = hashlib.sha1(f.read()).hexdigest()
        rows.append(dict(name=name, sw=sw, nozzle=nz, same=same,
                         heel_calm=heel, heel_task=heel_t, path=path,
                         sha1=sha, file=wrote))
        out(f"    {name:13s} nozzle {'on' if nz else 'zero'}: two processes "
            f"identical {same}; largest heel calm {heel:.1f} deg, task sea "
            f"{heel_t:.1f} deg; {os.path.basename(path)} {wrote}, sha1 {sha}")
        check("goldens", f"N2 {name}: identical in two processes", same)
        check("goldens", f"N2 {name}: largest heel of the calm runs <= "
              f"{HEEL_MAX_DEG} deg", heel <= HEEL_MAX_DEG, f"{heel:.1f} deg")
    dump("goldens.json", dict(rows=rows))
    flush("goldens.txt")


PARTS = ("w1", "w2", "w3", "w4", "w5", "w6", "w8", "w9", "cfree", "fkref",
         "calmid", "l1", "e1", "goldens", "f7", "quick", "lorder")


def _sub(arg):
    name, out_dir = arg
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"fix_{name}_stdout.log"), "w") as log:
        rc = subprocess.run([sys.executable, "-m", "studies.anyspeed_p1b_fix",
                             "--parts", name, "--out", out_dir], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT).returncode
    return name, rc, time.time() - t0


def main():
    global OUT, CENSUS
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="")
    ap.add_argument("--jobs", default="", help="parts, one process each")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--census", default=None,
                    help="the S1 output directory of the census on P1a + "
                    "wagner_entry (part w6)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    OUT = os.path.abspath(a.out)
    if a.census:
        CENSUS = os.path.abspath(a.census)
    C.OUT = OUT
    if a.jobs:
        from studies.parallel_runner import run_parallel
        t0 = time.time()
        res = run_parallel(_sub, [(x, OUT) for x in a.jobs.split(",") if x],
                           gb_per_job=0.5, reserve_gb=3.0, max_procs=4)
        for r in res:
            print(f"  {r}", flush=True)
        print(f"all done in {time.time() - t0:.0f} s wall", flush=True)
        return
    C.CAND = W
    C.STAMP = C.stamp_all()
    C.STAMP["review_fix_sha1"] = __import__("hashlib").sha1(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()
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
