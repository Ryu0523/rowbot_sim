#!/usr/bin/env python3
"""
Step P1c, track W (switch wagner_entry): the checks registered in
learn/meta/ANYSPEED_PLAN_2026-10-05.md, '### P1c', track W ('运行前登记的
检查'), before any of them ran on the final code. The P1b review's W checks
(W1-W13) are re-run with their own scripts (studies/anyspeed_p1b_fix.py,
studies/anyspeed_p1b_check.py --cand w, studies/anyspeed_p0_s1rerun.py);
this module adds the new ones. Helpers, stamps and report format of
studies/anyspeed_p1b_check.py; outputs in studies/_cache/anyspeed/p1c/w/ (or
--out).

Parts:
  replay   W4' the field's own time discretisation, separated from the
               trajectory's: the W plant's kinematics (d, Vg, u_r at the
               stations) recorded at every step of a dt = 0.0025 s run in
               W4's sea (regular head waves, lam 2 L, a 0.12 m, constant
               thrust, 60 s) at 12 and 28 kn; the planes' state field
               replayed on them at Delta = 0.02 / 0.01 / 0.005 / 0.0025 with
               the transport of P1c (Heun, the inflow at both ends of the
               step) and with the end-of-step inflow (P1b), F share as W4
               (mean F fraction over the wet stations, 0.02 s samples, last
               40 s)
  lowspeed W18 the low-speed decay of S1 part b with wagner_entry: the run
               of S1 (b) (captive, speed held, calm, from the Newton root +
               5 cm heave, dt 0.02) with the plant as it is and with three
               diagnostic variants of the field / flux rules (never part of
               the plant): Xsuction (the E/X planes' momentum flux V Dm/Dt
               counted whenever Dm/Dt > 0, whatever the sign of V), noX
               (E -> F where X would be), arriveF (planes arriving over the
               leading edge in F); and the plant from + 5 cm for 600 s at
               0.5-4 kn (no growth, R2)
  goldenw  N2 the golden runs of P1a + wagner_entry (anyspeed_regress
               golden_runs, as studies/anyspeed_p1b_fix part goldens) in two
               processes, bit for bit, largest calm heel <= 15 deg, written to
               a new file (golden_as_...+wagner_entry_p1c.npz; the P1b one is
               kept, never overwritten)

Round 2 of the P1c review (plan '复查后的修正（W 线第二轮）'; the field in
ES_NSUB sub-steps on interpolated kinematics):
  w1b      W1b synthetic transport (uniform inflow linear / sinusoidal in
               time): single-step Heun, the plant's sub-stepped form, the
               end-of-step rule
  replay2  W4' / W4'' the field-only F share replayed on dt_f = 0.00125
               kinematics (d, Vg, u_r, ddot), Delta 0.00125 * 2^k; the
               12-kn multi-step fit F0 + a h + b h^2 + c h^3
  budget   B1 the power budget of S1 (b)'s run at 1.0 / 1.5 kn, 30-90 s,
               by force group (window means minus the steady state's),
               closed against the measured d(E_total)/dt
  control  B2 Xsuction's added dissipation on the plant's own trajectory as
               an equal-power dashpot aft (and, recorded, at the bow)
  w5cap    W5 (c, d) on a captive plant of its own
  goldenw  now writes golden_as_...+wagner_entry_p1c2.npz

    .venv/Scripts/python.exe -m studies.anyspeed_p1c_w --parts replay
    .venv/Scripts/python.exe -m studies.anyspeed_p1c_w --jobs replay,lowspeed,goldenw
"""
import argparse
import hashlib
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
from studies import anyspeed_p1b_fix as FX      # noqa: E402

OUT = os.path.join(HERE, "_cache", "anyspeed", "p1c", "w")
KN = C.KN
G = C.G
W, P1A = FX.W, FX.P1A
out, check, dump, flush = C.out, C.check, C.dump, C.flush


# ===================================================================== replay
def record_job(arg):
    """W plant, W4's sea and thrust, dt_f: (d, Vg, u_r) at every step's
    start (the kinematics the step's first derivative used)."""
    kn, dtf, T = arg
    from hydro import hulls
    warnings.filterwarnings("ignore")
    p = FX._plant(W, sea=FX._head_sea(hulls.get(C.AS)), dt=dtf)
    s = p.initial_state(kn * KN)
    thr = s[12]
    n = int(round(T / dtf))
    D, VG, UR = [], [], []
    cap = {}
    cls = type(p)

    def spy(s_, t_):
        r_ = cls.deriv(p, s_, t_)
        if "k" not in cap:
            cap["k"] = r_[1]["kb"]
            cap["d"] = r_[1]["d"]
        return r_
    p.deriv = spy
    t = 0.0
    for i in range(n):
        cap.clear()
        s = p.step(s, t, thr, 0.0)
        t += dtf
        D.append(cap["d"].copy())
        VG.append(cap["k"].Vg.copy())
        UR.append(cap["k"].ur.copy())
    # the kinematics at the end of the run (the next step's start)
    _, aux = cls.deriv(p, s, t)
    D.append(aux["d"].copy())
    VG.append(aux["kb"].Vg.copy())
    UR.append(aux["kb"].ur.copy())
    return kn, np.array(D), np.array(VG), np.array(UR)


def replay_share(p, D, VG, UR, dtf, Delta, heun):
    """The F share series of the field replayed at Delta on the recorded
    kinematics (rows every dtf), the W4 statistic (last 40 s of 60)."""
    from sim import planing_anyspeed as PA
    x = p.x
    eg = p._es_geom()
    every = int(round(Delta / dtf))
    k02 = int(round(0.02 / dtf))
    b, st = PA._es_scan(x, D[0], VG[0], float(UR[0][0]), eg.edges)
    sh = []
    n = D.shape[0] - 1
    for i in range(every, n + 1, every):
        b, st = PA._es_advance(b, st, x, D[i], VG[i], UR[i], Delta,
                               eg.edges, ur0=UR[i - every] if heun else None)
        if i % k02 == 0:
            w = D[i] > 0.0
            f = PA._es_frac(b, st, eg.edges, PA.ES_F)
            sh.append(float(f[w].sum() / max(int(w.sum()), 1)))
    sh = np.array(sh)
    return float(sh[len(sh) // 3:].mean())


def part_replay():
    from hydro import hulls
    from studies.parallel_runner import run_parallel
    out("W4': the field's own time discretisation: kinematics recorded at "
        "dt_f = 0.0025 s (W plant, W4's sea and thrust, 60 s), the field "
        "replayed at Delta 0.02 / 0.01 / 0.005 / 0.0025; F share as W4")
    dtf = 0.0025
    res = run_parallel(record_job, [(12.0, dtf, 60.0), (28.0, dtf, 60.0)],
                       gb_per_job=0.6, reserve_gb=3.0)
    p = FX._plant(W, sea=FX._head_sea(hulls.get(C.AS)), dt=0.02)
    Ds = (0.02, 0.01, 0.005, 0.0025)
    rows = []
    ok = True
    for r in res:
        if isinstance(r, Exception):
            out(f"    job failed: {r!r}")
            ok = False
            continue
        kn, D, VG, UR = r
        for heun in (True, False):
            v = [replay_share(p, D, VG, UR, dtf, De, heun) for De in Ds]
            dd = [v[0] - v[1], v[1] - v[2], v[2] - v[3]]
            od = [math.log2(abs(dd[0] / dd[1])) if dd[1] else float("nan"),
                  math.log2(abs(dd[1] / dd[2])) if dd[2] else float("nan")]
            rows.append(dict(kn=kn, heun=heun, F=v, diffs=dd, orders=od))
            out(f"    {kn:4.1f} kn {'Heun' if heun else 'end '}: F share "
                + ", ".join(f"{a:.6f}" for a in v) + f"; differences "
                + ", ".join(f"{a:+.2e}" for a in dd)
                + f"; orders {od[0]:.2f}, {od[1]:.2f}")
            if heun:
                good = od[0] >= 1.8
                ok &= good
        a = [x for x in rows if x["kn"] == kn]
        off = a[0]["F"][3] - a[1]["F"][3]
        out(f"      {kn:4.1f} kn: Heun - end-of-step at Delta 0.0025 "
            f"{off:+.2e} (the end-of-step rule's first-order offset there)")
    check("replay", "W4': with the transport of P1c the field-only F share "
          "converges at order >= 1.8 between Delta 0.02/0.01 and 0.01/0.005 "
          "at 12 and 28 kn", ok)
    dump("replay.json", dict(rows=rows))
    flush("replay.txt")


# =================================================================== lowspeed
def _xsuction(p):
    """Diagnostic variant (never in the plant): the E/X planes' momentum
    flux V Dm/Dt counted whenever Dm/Dt > 0, whatever the sign of V (S1's
    gate keeps V > 0 only); the F planes as in the plant."""
    cls = type(p)

    def fb(d, kb, dmaZ, slope):
        Vr, Vg, ur = kb.V, kb.Vg, kb.ur
        rate = dmaZ * Vg
        act, ycap, iunc = p._substrip(d, slope)
        km = p.k_a * 0.5 * np.pi * p.rho
        kc = 0.5 * np.pi / p.tanb
        if act.size:
            rate[act] = (Vg[act] * (2.0 * km * kc * kc) * iunc
                         + ur[act] * km * ycap) / p.dx
        gated = np.where((Vr > 0.0) & (rate > 0.0), Vr * rate, 0.0)
        anyv = np.where(rate > 0.0, Vr * rate, 0.0)
        f0, hx = cls._flux_b(p, d, kb, dmaZ, slope)
        if kb.fixed.any():
            out_ = np.where(kb.fixed, f0 + (1.0 - kb.f) * (anyv - gated),
                            anyv)
        else:
            out_ = anyv
        return out_, hx
    p._flux_b = fb


def _adv_arrive_F(b, st, xs, d, vg, ur, dt, edges, ur0=None):
    """sim/planing_anyspeed._es_advance with the planes arriving over the
    leading edge in F instead of E (diagnostic variant arriveF only)."""
    from sim import planing_anyspeed as PA
    lo, hi = float(edges[0]), float(edges[-1])
    m = (b > lo) & (b < hi)
    idx = np.nonzero(m)[0]
    s_in = st[np.searchsorted(b, lo, side="right")]
    b2 = np.concatenate([[lo], b[idx], [hi]])
    st2 = np.concatenate([[PA.ES_E, s_in], st[idx + 1],
                          [PA.ES_F]]).astype(np.int8)
    if ur0 is None:
        b3 = np.maximum.accumulate(b2 - np.interp(b2, xs, ur) * dt)
    else:
        v0 = np.interp(b2, xs, ur0)
        v1 = np.interp(b2 - v0 * dt, xs, ur)
        b3 = np.maximum.accumulate(b2 - 0.5 * (v0 + v1) * dt)
    inside = b3[(b3 > lo) & (b3 < hi)]
    P = np.unique(np.concatenate([inside, PA._es_points(xs, d, vg, edges)]))
    e = np.concatenate([[lo], P, [hi]])
    mid = 0.5 * (e[:-1] + e[1:])
    old = st2[np.searchsorted(b3, mid, side="right")]
    cls = PA._es_class(xs, d, vg, edges, mid)
    return PA._es_build(lo, hi, P, PA._ES_MAP[old, cls])


def _step1_arrive_F(b, st, xs, d, vg, ur, dt, edges, ur0=None, uni=False):
    """sim/planing_anyspeed._es_step1 with the planes arriving over the
    leading edge in F instead of E (diagnostic variant arriveF only; P1c
    round 2: the field advances in sub-steps of _es_step1, so the variant
    patches the sub-step, not _es_advance)."""
    from sim import planing_anyspeed as PA
    lo, hi = float(edges[0]), float(edges[-1])
    i0 = int(np.searchsorted(b, lo, side="right"))
    i1 = int(np.searchsorted(b, hi, side="left"))
    b2 = np.concatenate(([lo], b[i0:i1], [hi]))
    st2 = np.concatenate(([PA.ES_E, st[i0]], st[i0 + 1:i1 + 1],
                          [PA.ES_F])).astype(np.int8)
    if ur0 is None:
        b3 = np.maximum.accumulate(b2 - np.interp(b2, xs, ur) * dt)
    else:
        v0 = np.interp(b2, xs, ur0)
        v1 = np.interp(b2 - v0 * dt, xs, ur)
        b3 = np.maximum.accumulate(b2 - 0.5 * (v0 + v1) * dt)
    inside = b3[(b3 > lo) & (b3 < hi)]
    P = np.unique(np.concatenate([inside, PA._es_points(xs, d, vg, edges)]))
    e = np.concatenate([[lo], P, [hi]])
    mid = 0.5 * (e[:-1] + e[1:])
    old = st2[np.searchsorted(b3, mid, side="right")]
    cls = PA._es_class(xs, d, vg, edges, mid)
    return PA._es_build(lo, hi, P, PA._ES_MAP[old, cls])


def low_job(arg):
    from sim import planing_anyspeed as PA
    from studies import anyspeed_s1_check as S1
    variant, kn, dz, T = arg
    warnings.filterwarnings("ignore")
    E, X, F = PA.ES_E, PA.ES_X, PA.ES_F
    keep = (PA._ES_MAP, PA._es_advance, PA._es_step1)
    try:
        if variant == "noX":
            PA._ES_MAP = np.array([[E, F, E], [E, F, F], [E, F, F]],
                                  np.int8)
        if variant == "arriveF":
            PA._es_step1 = _step1_arrive_F
        u = kn * KN
        dt = 0.02
        p = FX._plant(W, captive_u=u, dt=dt)
        sol, res = S1.newton(p, u, np.array(p.running_attitude(u)))
        z0, th0 = sol
        if variant == "Xsuction":
            _xsuction(p)
        s = S1.state(u, z0 + dz, th0)
        p.initial_state(0.0)
        n = int(round(T / dt))
        zz = np.empty(n + 1)
        zz[0] = dz
        for i in range(n):
            s = p.step(s, i * dt, 0.0, 0.0)
            s[6] = u
            zz[i + 1] = s[2] - z0
    finally:
        # the variants never leak into another job of this worker
        PA._ES_MAP, PA._es_advance, PA._es_step1 = keep
    k = int(round(10 / dt))
    env = [float(np.abs(zz[j * k:(j + 1) * k]).max())
           for j in range(len(zz) // k)]
    return dict(variant=variant, kn=kn, dz=dz, T=T, env=env, resid=res)


def s1b_verdict(env):
    """S1 (b)'s verdict on the first 120 s (studies/anyspeed_s1_check
    part_b)."""
    e = env[:12]
    falling = e[-1] < 0.9 * e[-2] or e[-1] < 1e-12
    if e[-1] < 0.01 * e[0] and falling:
        return "decays"
    if e[-1] > e[0]:
        return "GROWS"
    return "no decay (bounded)"


def part_lowspeed():
    from studies.parallel_runner import run_parallel
    out("W18: S1 (b)'s run (captive, speed held, calm, from the root + 5 cm "
        "heave, dt 0.02) with wagner_entry, and diagnostic variants of the "
        "rules (never in the plant); 10-s envelopes max |z - z_root|")
    args = [(v, kn, 0.05, 240.0) for v in ("W", "Xsuction", "noX", "arriveF")
            for kn in (1.0, 1.5)]
    args += [("W", kn, 0.05, 600.0) for kn in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)]
    res = run_parallel(low_job, args, gb_per_job=0.4, reserve_gb=3.0)
    rows = [r for r in res if not isinstance(r, Exception)]
    for r in res:
        if isinstance(r, Exception):
            out(f"    job failed: {r!r}")
    ok_x, ok_g = True, True
    for r in rows:
        e = r["env"]
        v = s1b_verdict(e)
        r["verdict_120s"] = v
        out(f"    {r['variant']:8s} {r['kn']:3.1f} kn {r['T']:4.0f} s: "
            f"verdict (S1 b, 120 s) {v:18s}; envelope every 30 s "
            + " ".join(f"{a:.1e}" for a in e[::3]))
        if r["variant"] == "Xsuction":
            ok_x &= v == "decays"
        if r["variant"] == "W" and r["T"] >= 600.0:
            grow = max(e) > e[0] or max(e[-6:]) > max(e[-12:-6])
            r["grows"] = bool(grow)
            ok_g &= not grow
    check("lowspeed", "W18 (mechanism): with the E/X planes' V Dm/Dt counted "
          "whenever Dm/Dt > 0 (diagnostic), S1 (b)'s 1.0 and 1.5 kn runs "
          "from + 5 cm decay by S1 (b)'s verdict", ok_x)
    check("lowspeed", "W18 (R2): wagner_entry from + 5 cm, 0.5-4 kn, 600 s: "
          "no 10-s envelope above the first, the last 60 s not above the 60 "
          "s before", ok_g)
    dump("lowspeed.json", dict(rows=rows))
    flush("lowspeed.txt")


# ==================================================================== goldenw
def part_goldenw():
    from studies.parallel_runner import run_parallel
    out("N2: the golden runs of P1a + wagner_entry (anyspeed_regress."
        "golden_runs on PlaningVesselAS), two processes")
    res = run_parallel(FX.golden_job, [(W, True), (W, True)],
                       gb_per_job=0.4, reserve_gb=3.0)
    a_, b_ = res
    if isinstance(a_, Exception) or isinstance(b_, Exception):
        check("goldenw", "N2 wagner_entry: runs", False, f"{a_!r} {b_!r}")
        flush("goldenw.txt")
        return
    same = set(a_) == set(b_) and all(
        np.array_equal(np.asarray(a_[k]), np.asarray(b_[k])) for k in a_)
    heel = max(float(np.degrees(np.abs(np.asarray(v)[:, 3]).max()))
               for k, v in a_.items()
               if k.startswith("calm_") and k.endswith("_s"))
    # round 2 of the P1c review (sub-steps of the field): a new file; the
    # round-1 one (_p1c.npz) and P1b's stay on disk, never overwritten
    path = FX.golden_path(W, True).replace(".npz", "_p1c2.npz")
    old = FX.golden_path(W, True).replace(".npz", "_p1c.npz")
    if os.path.exists(old):
        with np.load(old) as z:
            prev = {k: z[k] for k in z.files}
        same_old = set(prev) == set(a_) and all(
            np.array_equal(prev[k], np.asarray(a_[k])) for k in a_)
        dmax = max((float(np.abs(prev[k] - np.asarray(a_[k])).max())
                    for k in a_ if k in prev
                    and np.shape(prev[k]) == np.shape(a_[k])
                    and np.asarray(a_[k]).dtype.kind == "f"), default=0.0)
        out(f"    against the previous golden ({os.path.basename(old)}): "
            f"identical {same_old}, largest difference {dmax:.2e}")
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
    check("goldenw", "N2 wagner_entry: identical in two processes", same)
    check("goldenw", "N2 wagner_entry: largest calm heel <= 15 deg",
          heel <= 15.0, f"{heel:.1f} deg")
    dump("goldenw.json", dict(same=same, heel=heel, path=path, sha1=sha,
                              file=wrote))
    flush("goldenw.txt")



# ======================================================== P1c review (round 2)
# The checks registered in the plan's '### P1c（W 线）', '复查后的修正（第二轮）'
# before they ran on the final code.

# ------------------------------------------------------------------- w1b
def part_w1b():
    """W1b: the transport alone on a synthetic field (every plane wet and
    entering, one F block, inflow uniform along the hull): (a) inflow linear
    in time: Heun (one step) and the plant's sub-stepped form exact; (b)
    sinusoidal: measured orders; the end-of-step rule for comparison."""
    from sim import planing_anyspeed as PA
    out("W1b: synthetic transport, inflow uniform in x; (a) linear in time, "
        "(b) 1 + 0.5 sin 3t; boundary error of the F block after 2 s")
    edges = np.linspace(-3.0, 3.0, 49)
    xs = 0.5 * (edges[:-1] + edges[1:])
    d = np.full(xs.size, 0.2)
    vg = np.full(xs.size, 1.0)
    dd = np.zeros(xs.size)

    def run(urf, T, dt, form):
        b, st = PA._es_build(-3.0, 3.0, np.array([1.0, 2.0]),
                             np.array([PA.ES_E, PA.ES_F, PA.ES_E]))
        n = int(round(T / dt))
        km = None
        for i in range(n):
            t0, t1 = i * dt, (i + 1) * dt
            u0 = np.full(xs.size, urf(t0))
            u1 = np.full(xs.size, urf(t1))
            if form == "end":
                b, st = PA._es_advance(b, st, xs, d, vg, u1, dt, edges)
            elif form == "heun":
                b, st = PA._es_advance(b, st, xs, d, vg, u1, dt, edges,
                                       ur0=u0)
            else:
                k0 = dict(t=t0, d=d, vg=vg, ur=u0, dd=dd)
                b, st = PA._es_advance(b, st, xs, d, vg, u1, dt, edges,
                                       ur0=u0, k0=k0, km=km, dd=dd,
                                       nsub=PA.ES_NSUB)
                km = k0
        i = np.nonzero(st == PA.ES_F)[0]
        return b[i[0] - 1], b[i[0]]

    cases = [("linear", lambda t: 0.5 + 0.3 * t,
              lambda T: 0.5 * T + 0.15 * T * T),
             ("sine", lambda t: 1.0 + 0.5 * math.sin(3.0 * t),
              lambda T: T + 0.5 / 3.0 * (1 - math.cos(3.0 * T)))]
    dts = (0.04, 0.02, 0.01, 0.005)
    rows = []
    ok = True
    for name, urf, X in cases:
        for form in ("heun", "plant", "end"):
            errs = []
            for dt in dts:
                lo, hi = run(urf, 2.0, dt, form)
                errs.append(max(abs(lo - (1.0 - X(2.0))),
                                abs(hi - (2.0 - X(2.0)))))
            od = [math.log2(errs[k] / errs[k + 1]) if errs[k + 1] > 0
                  else float("inf") for k in range(3)]
            rows.append(dict(case=name, form=form, errs=errs, orders=od))
            out(f"    {name:6s} {form:5s}: errors "
                + " ".join(f"{e:.2e}" for e in errs) + "; orders "
                + " ".join(f"{o:.2f}" for o in od))
            if name == "linear" and form in ("heun", "plant"):
                ok &= max(errs) <= 1e-12
            if name == "sine" and form in ("heun", "plant"):
                ok &= all(abs(o - 2.0) <= 0.1 for o in od)
            if name == "sine" and form == "end":
                ok &= all(abs(o - 1.0) <= 0.1 for o in od)
    check("w1b", "W1b: inflow linear in time: Heun and the sub-stepped form "
          "exact (<= 1e-12 m); sinusoidal: their orders 2 +- 0.1 and the "
          "end-of-step rule's 1 +- 0.1 (dt 0.04-0.005)", ok)
    dump("w1b.json", dict(rows=rows))
    flush("w1b.txt")


# ---------------------------------------------------------------- replay2
def record2_job(arg):
    """W plant (final code), W4's sea and thrust, dt_f: (d, Vg, u_r, ddot)
    at every step's start."""
    kn, dtf, T = arg
    from hydro import hulls
    warnings.filterwarnings("ignore")
    p = FX._plant(W, sea=FX._head_sea(hulls.get(C.AS)), dt=dtf)
    s = p.initial_state(kn * KN)
    thr = s[12]
    n = int(round(T / dtf))
    D, VG, UR, DD = [], [], [], []
    cap = {}
    cls = type(p)

    def spy(s_, t_):
        r_ = cls.deriv(p, s_, t_)
        if "k" not in cap:
            cap["k"] = r_[1]["kb"]
            cap["d"] = r_[1]["d"]
        return r_
    p.deriv = spy
    t = 0.0
    for i in range(n):
        cap.clear()
        s = p.step(s, t, thr, 0.0)
        t += dtf
        D.append(cap["d"].copy())
        VG.append(cap["k"].Vg.copy())
        UR.append(cap["k"].ur.copy())
        DD.append(cap["k"].ddot.copy())
    _, aux = cls.deriv(p, s, t)
    D.append(aux["d"].copy())
    VG.append(aux["kb"].Vg.copy())
    UR.append(aux["kb"].ur.copy())
    DD.append(aux["kb"].ddot.copy())
    return kn, dtf, np.array(D), np.array(VG), np.array(UR), np.array(DD)


def replay2_share(p, K, Delta, form, every_s):
    """The F share of the field replayed at Delta on recorded kinematics
    (rows every dtf), sampled every every_s seconds over the last 40 s of
    60: form 'plant' (ES_NSUB sub-steps on interpolated kinematics, the
    plant's), 'heun' (one Heun step), 'end' (end-of-step inflow)."""
    from sim import planing_anyspeed as PA
    kn, dtf, D, VG, UR, DD = K
    x = p.x
    eg = p._es_geom()
    every = int(round(Delta / dtf))
    ks = int(round(every_s / dtf))
    b, st = PA._es_scan(x, D[0], VG[0], float(UR[0][0]), eg.edges)
    sh = []
    n = D.shape[0] - 1
    km = None
    for i in range(every, n + 1, every):
        if form == "plant":
            k0 = dict(t=(i - every) * dtf, d=D[i - every], vg=VG[i - every],
                      ur=UR[i - every], dd=DD[i - every])
            b, st = PA._es_advance(b, st, x, D[i], VG[i], UR[i], Delta,
                                   eg.edges, ur0=UR[i - every], k0=k0,
                                   km=km, dd=DD[i], nsub=PA.ES_NSUB)
            km = k0
        elif form == "heun":
            b, st = PA._es_advance(b, st, x, D[i], VG[i], UR[i], Delta,
                                   eg.edges, ur0=UR[i - every])
        else:
            b, st = PA._es_advance(b, st, x, D[i], VG[i], UR[i], Delta,
                                   eg.edges)
        if i % ks == 0:
            w = D[i] > 0.0
            f = PA._es_frac(b, st, eg.edges, PA.ES_F)
            sh.append(float(f[w].sum() / max(int(w.sum()), 1)))
    sh = np.array(sh)
    return float(sh[len(sh) // 3:].mean())


def part_replay2():
    from hydro import hulls
    from studies.parallel_runner import run_parallel
    out("W4' / W4'': the field's own time discretisation on the final code: "
        "kinematics recorded at dt_f = 0.00125 s (W plant, W4's sea and "
        "thrust, 60 s), the field replayed at Delta = 0.00125 * 2^k")
    dtf = 0.00125
    res = run_parallel(record2_job, [(12.0, dtf, 60.0), (28.0, dtf, 60.0)],
                       gb_per_job=0.6, reserve_gb=3.0)
    p = FX._plant(W, sea=FX._head_sea(hulls.get(C.AS)), dt=0.02)
    rows = []
    ok_fit = True
    for r in res:
        if isinstance(r, Exception):
            out(f"    job failed: {r!r}")
            ok_fit = False
            continue
        kn = r[0]
        # W4' as registered in round 1 (samples every 0.02 s, Delta 0.02 /
        # 0.01 / 0.005 / 0.0025), now on the plant's form, plus the others
        for form in ("plant", "heun", "end"):
            v = [replay2_share(p, r, De, form, 0.02)
                 for De in (0.02, 0.01, 0.005, 0.0025)]
            dd_ = [v[0] - v[1], v[1] - v[2], v[2] - v[3]]
            od = [math.log2(abs(dd_[0] / dd_[1])) if dd_[1] else float("nan"),
                  math.log2(abs(dd_[1] / dd_[2])) if dd_[2] else float("nan")]
            out(f"    {kn:4.1f} kn {form:5s} (0.02-s samples): F share "
                + ", ".join(f"{a:.6f}" for a in v) + "; differences "
                + ", ".join(f"{a:+.2e}" for a in dd_)
                + f"; orders {od[0]:.2f}, {od[1]:.2f}")
            rows.append(dict(kn=kn, form=form, kind="W4'", F=v, diffs=dd_,
                             orders=od))
        # W4'': the multi-step fit (samples every 0.04 s, Delta 0.00125 *
        # 2^k up to 0.04): F0 + a h + b h^2 + c h^3, least squares
        hs = np.array([0.00125, 0.0025, 0.005, 0.01, 0.02, 0.04])
        for form in ("plant", "heun", "end"):
            v = np.array([replay2_share(p, r, h, form, 0.04) for h in hs])
            Amat = np.stack([np.ones_like(hs), hs, hs ** 2, hs ** 3], 1)
            cf, *_ = np.linalg.lstsq(Amat, v, rcond=None)
            resid = float(np.sqrt(np.mean((Amat @ cf - v) ** 2)))
            e02 = abs(float(v[4] - cf[0]))
            lin = abs(float(cf[1])) * 0.02
            good = lin <= 0.1 * e02
            out(f"    {kn:4.1f} kn {form:5s} fit (0.04-s samples): F0 "
                f"{cf[0]:.6f}, a {cf[1]:+.3e}, b {cf[2]:+.3e}, c {cf[3]:+.3e}"
                f", residual {resid:.1e}; |a| 0.02 = {lin:.2e} vs error at "
                f"0.02 {e02:.2e} ({'<=' if good else '>'} 10 %); F(h) "
                + ", ".join(f"{a:.6f}" for a in v))
            rows.append(dict(kn=kn, form=form, kind="fit", F=v.tolist(),
                             coef=cf.tolist(), resid=resid, e02=e02,
                             lin=lin, good=bool(good)))
            if kn == 12.0 and form == "plant":
                ok_fit &= bool(good)
    check("replay2", "W4'' (gate): at 12 kn the plant's field-only F share "
          "has |a| * 0.02 <= 10 % of its error at Delta 0.02 in the fit F0 + "
          "a h + b h^2 + c h^3 (Delta 0.00125-0.04)", ok_fit)
    dump("replay2.json", dict(rows=rows))
    flush("replay2.txt")


# ------------------------------------------------------------------ budget
def _bterms(p, s, ds, aux, u):
    """The power-budget quantities of one derivative of a captive calm-water
    W plant (the step's first): station arrays and scalars."""
    from sim import planing_anyspeed as PA
    from sim.planing_vessel import SP
    kb = aux["kb"]
    x, dx, rho = p.x, p.dx, p.rho
    d, V, Vg, ur = aux["d"], kb.V, kb.Vg, kb.ur
    w, q, wd, qd = s[8], s[10], ds[8], ds[10]
    Th = p.Th_rest + s[4]
    tau = p.kp + SP * Th
    ctr = p._ctr(u)
    A, c, _, dma, _ = p._section(d)
    dma = dma * ctr
    M = kb.ma * ctr
    mZ = kb.maZ * ctr
    f = np.asarray(kb.f, float)
    m0, dm0, dm0y = kb.m0 * ctr, kb.dm0 * ctr, kb.dm0y * ctr
    a_c = SP * u * q - u * u * p.kpp
    a = 2.0 * SP * u * q - u * u * p.kpp - (wd + SP * x * qd)
    # the E/X planes' rate of added mass, as _flux_b (calm: slope -tau)
    rate = dma * Vg
    act, ycap, iunc = p._substrip(d, -tau)
    km_ = p.k_a * 0.5 * np.pi * p.rho
    kc = 0.5 * np.pi / p.tanb
    if act.size:
        uu = ur[act] if p.sw.transom else np.maximum(ur[act], 0.0)
        rate[act] = (Vg[act] * (2.0 * km_ * kc * kc) * iunc
                     + uu * km_ * ycap) / dx
    fEX = np.where((V > 0.0) & (rate > 0.0), V * rate, 0.0)
    conv0 = -ur * dm0y * p._dyc_dx()
    gate = (V > 0.0) & (conv0 > 0.0)
    f0 = (Vg - 0.5 * V) * V * dm0 + np.where(gate, V * conv0, 0.0)
    flux = aux["mom_full"]
    on = f > 0.0
    flux_re = np.where(on, (1.0 - f) * fEX + f * f0, fEX)
    cf = 0.5 * rho * p.c_dc * 2.0 * c * V * np.abs(V) * ctr
    ab = p._a_b(u)
    buoy = ab * rho * G * A * ctr
    qsec = PA._vsec_q(d, p.yc, p.tanb)[0]
    g = w + SP * x * q
    fs = M * a + flux + cf + buoy
    wet = d > 0.0
    zw = float(np.mean(p.zk[wet])) if wet.any() else float(p.zk[0])
    xn, _, zn = p.thrusters[0]
    Mload = zn * aux["fx"] - zw * aux["Df"] - p.num.z_air * aux["Da"]
    P_kin = p.m * w * wd + p.Iy * q * qd
    P_eomres = P_kin - (float((fs * g).sum() * dx) - p.m * G * w
                        + q * (Mload + aux["Mh"]))
    sEX = 0.5 * V * V * rate - fEX * V
    wEX = 1.0 - f
    q1 = (V > 0.0) & (rate > 0.0)
    q2 = (V <= 0.0) & (rate > 0.0)
    q3 = (V > 0.0) & (rate <= 0.0)
    q4 = (V <= 0.0) & (rate <= 0.0)
    sFl = 0.5 * V * V * dm0 * (V - Vg) * f
    sFc = (0.5 - gate) * V * V * conv0 * f
    Me = 1.5 * M[0] - 0.5 * M[1]
    Ve = 1.5 * V[0] - 0.5 * V[1]

    def S(a_):
        return float(np.sum(a_) * dx)
    return dict(
        t=None, z=float(s[2]), th=float(s[4]), w=float(w), q=float(q),
        KE=0.5 * p.m * w * w + 0.5 * p.Iy * q * q,
        U=p.m * G * float(s[2]) + ab * rho * G * ctr * S(qsec),
        Tf=0.5 * S(M * V * V),
        P_A_I=u * S(M * a * tau), P_A_EX=u * S(wEX * fEX * tau),
        P_A_F=u * S(np.where(on, f * f0, 0.0) * tau),
        W_fl=S((M * a + flux) * V),
        src_E=S(wEX * sEX * q1), src_X=S(wEX * sEX * q2),
        src_q3=S(wEX * sEX * q3), src_q4=S(wEX * sEX * q4),
        src_Fl=S(sFl), src_Fc=S(sFc),
        P_cf=S(cf * g), P_hs=S(buoy * g) - p.m * G * w,
        P_ld=q * Mload, P_Mh=q * aux["Mh"], P_eomres=P_eomres,
        P_conv=-S(M * V * a_c),
        P_adv=u * S(0.5 * V * V * np.gradient(M, dx)),
        P_T0=-u * 0.5 * M[0] * V[0] ** 2, P_Te=-u * 0.5 * Me * Ve * Ve,
        flux_err=float(np.abs(flux_re - flux).max()),
        # station arrays (region report, state changes, the control)
        f=f, M=M, mZ=mZ, m0=m0, V=V, d=d, g=g,
        pX=wEX * sEX * q2, pE=wEX * sEX * q1,
        addX=wEX * np.where(q2, rate * V, 0.0) * g,
        act=np.isin(np.arange(x.size), act).astype(float),
        kin=dict(d=np.array(d), vg=np.array(Vg), ur=np.array(ur),
                 dd=np.array(kb.ddot)), es=kb.es)


def _brun(kn, T1, dz=0.05, variant=None, cd=0.0, region=None, keep=True):
    """S1 (b)'s run (captive, calm, root + dz) of the W plant, the budget
    quantities of every step's first derivative (keep), optionally with a
    variant (Xsuction; 'dashpot': -cd g on the stations of region)."""
    from studies import anyspeed_s1_check as S1
    from sim.planing_vessel import SP
    u = kn * KN
    dt = 0.02
    p = FX._plant(W, captive_u=u, dt=dt)
    sol, res = S1.newton(p, u, np.array(p.running_attitude(u)))
    z0, th0 = sol
    sroot = S1.state(u, z0, th0)
    ds_r, aux_r = type(p).deriv(p, sroot, 0.0)
    steady = _bterms(p, sroot, ds_r, aux_r, u)
    cls = type(p)
    cap = {}
    if variant == "Xsuction":
        _xsuction(p)
    if variant == "dashpot":
        base_fb = p._flux_b

        def fbd(d_, kb_, dmaZ_, slope_):
            fl, hx_ = base_fb(d_, kb_, dmaZ_, slope_)
            s_ = cap["s_now"]
            g_ = s_[8] + SP * p.x * s_[10]
            return fl - cd * g_ * region, hx_
        p._flux_b = fbd

    def spy(s_, t_):
        cap["s_now"] = s_
        r_ = cls.deriv(p, s_, t_)
        if "aux" not in cap:
            cap["aux"], cap["ds"], cap["s"] = r_[1], r_[0].copy(), s_.copy()
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
            r_ = _bterms(p, cap["s"], cap["ds"], cap["aux"], u)
            r_["t"] = i * dt
            rows.append(r_)
    k = int(round(10 / dt))
    env = [float(np.abs(np.array(zz[j * k:(j + 1) * k])).max())
           for j in range(len(zz) // k)]
    return p, steady, rows, env


def _state_change(p, rows, n):
    """The energy of the state changes inside the strips over step n -> n+1
    (rows n-1, n, n+1): the field re-advanced from row n's with the plant's
    arguments (checked equal to row n+1's), and with every plane keeping its
    state (transport only); E_st = sum 1/2 V^2 (f1 - f_tr) (m0 - mZ) dx,
    V^2 the mean of the two ends'."""
    from sim import planing_anyspeed as PA
    eg = p._es_geom()
    r0, r1 = rows[n], rows[n + 1]
    km = dict(rows[n - 1]["kin"]) if n >= 1 else None
    k0 = dict(r0["kin"])
    k0["t"] = r0["t"]
    if km is not None:
        km["t"] = rows[n - 1]["t"]
    k1 = r1["kin"]
    dt = r1["t"] - r0["t"]
    b, st = r0["es"]
    args = (b, st, p.x, k1["d"], k1["vg"], k1["ur"], dt, eg.edges)
    kw = dict(ur0=k0["ur"], k0=k0, km=km, dd=k1["dd"], nsub=PA.ES_NSUB)
    ba, sa = PA._es_advance(*args, **kw)
    same = (np.array_equal(ba, r1["es"][0])
            and np.array_equal(sa, r1["es"][1]))
    keep = PA._ES_MAP
    try:
        PA._ES_MAP = np.array([[PA.ES_E] * 3, [PA.ES_X] * 3, [PA.ES_F] * 3],
                              np.int8)
        bt, stt = PA._es_advance(*args, **kw)
    finally:
        PA._ES_MAP = keep
    wet = k1["d"] > 0.0
    ftr = np.where(wet, PA._es_frac(bt, stt, eg.edges, PA.ES_F), 0.0)
    dM = (r1["f"] - ftr) * (r1["m0"] - r1["mZ"])
    v2 = 0.5 * (r0["V"] ** 2 + r1["V"] ** 2)
    return float(np.sum(0.5 * v2 * dM) * p.dx), same


GROUPS = ("P_A_I", "P_A_EX", "P_A_F", "src_E", "src_X", "src_q3", "src_q4",
          "src_Fl", "src_Fc", "P_T", "P_st", "P_cf", "P_ld", "P_Mh")
_SRCS = ("src_E", "src_X", "src_q3", "src_q4", "src_Fl", "src_Fc")
_BKEYS = ("KE", "U", "Tf", "P_A_I", "P_A_EX", "P_A_F", "W_fl", "src_E",
          "src_X", "src_q3", "src_q4", "src_Fl", "src_Fc", "P_cf", "P_hs",
          "P_ld", "P_Mh", "P_eomres", "P_conv", "P_adv", "P_T0", "P_Te")


def budget_job(arg):
    kn, T0, T1 = arg
    warnings.filterwarnings("ignore")
    p, steady, rows, env = _brun(kn, T1 + 0.1)
    dt = 0.02
    n0, n1 = int(round(T0 / dt)), int(round(T1 / dt))
    Est, same_all = np.zeros(len(rows)), True
    for n in range(n0 - 2, n1 + 2):
        e_, same = _state_change(p, rows, n)
        Est[n] = e_
        same_all &= same
    Z = {k: np.array([r_[k] for r_ in rows]) for k in _BKEYS}
    E = Z["KE"] + Z["U"] + Z["Tf"]
    idx = np.arange(n0, n1 + 1)

    def cd(y):
        return (y[idx + 1] - y[idx - 1]) / (2 * dt)
    dE, dTf, dKU = cd(E), cd(Z["Tf"]), cd(Z["KE"] + Z["U"])
    P = {k: Z[k][idx] for k in _BKEYS}
    P["P_T"] = P["P_T0"]
    P["P_st"] = 0.5 * (Est[idx - 1] + Est[idx]) / dt
    st_ = dict(steady)
    st_["P_T"], st_["P_st"] = st_["P_T0"], 0.0
    # closures: (1) mechanical, (2) fluid, (3) total
    mech = P["P_A_I"] + P["P_A_EX"] + P["P_A_F"] - P["W_fl"] + P["P_cf"] \
        + P["P_ld"] + P["P_Mh"]
    res1 = dKU - mech
    fluid = P["W_fl"] + sum(P[k] for k in _SRCS) + P["P_T"] + P["P_st"]
    D = dTf - fluid
    D_st = -(st_["W_fl"] + sum(st_[k] for k in _SRCS) + st_["P_T"])
    tot = sum(P[k] for k in GROUPS)
    res3 = dE - tot
    means = {k: float(np.mean(P[k]) - st_[k]) for k in GROUPS}
    gross = sum(abs(v) for v in means.values())
    dEm = float((E[n1] - E[n0]) / (T1 - T0))
    region = {}
    for k in ("f", "pX", "pE", "addX", "act"):
        region[k] = np.mean([rows[n][k] for n in idx], axis=0)
    region["g2"] = np.mean([rows[n]["g"] ** 2 for n in idx], axis=0)
    return dict(
        kn=kn, env=env, means=means, gross=gross, dE_mean=dEm,
        dE_mean_cd=float(np.mean(dE)),
        res3_mean=float(np.mean(res3) - D_st),
        res1_mean=float(np.mean(res1)),
        D_mean=float(np.mean(D) - D_st), D_steady=float(D_st),
        res1_rms=float(np.sqrt(np.mean(res1 ** 2))),
        res3_rms=float(np.sqrt(np.mean((res3 - D_st) ** 2))),
        dE_rms=float(np.sqrt(np.mean(dE ** 2))),
        hs_cons=float(np.sqrt(np.mean((P["P_hs"] + cd(Z["U"])) ** 2))),
        hs_rms=float(np.sqrt(np.mean(P["P_hs"] ** 2))),
        eomres=float(np.abs(P["P_eomres"]).max()),
        flux_err=max(r_["flux_err"] for r_ in rows),
        conv=float(np.mean(P["P_conv"]) - steady["P_conv"]),
        adv=float(np.mean(P["P_adv"]) - steady["P_adv"]),
        Te=float(np.mean(P["P_Te"]) - steady["P_Te"]),
        steady={k: float(steady[k]) for k in _BKEYS},
        readvance_same=bool(same_all),
        region={k: np.asarray(v).tolist() for k, v in region.items()},
        x=p.x.tolist())


def part_budget():
    from studies.parallel_runner import run_parallel
    out("B1 (T9): the power budget of S1 (b)'s run (captive, calm, root + 5 "
        "cm, dt 0.02) at 1.0 and 1.5 kn over 30-90 s, by force group; window "
        "means minus the steady state's (W, positive = into the energy)")
    res = run_parallel(budget_job, [(1.0, 30.0, 90.0), (1.5, 30.0, 90.0)],
                       gb_per_job=0.5, reserve_gb=3.0)
    rows = []
    ok = True
    names = dict(P_A_I="surge transfer, inertia force (u sum M a tau)",
                 P_A_EX="surge transfer, E/X flux force",
                 P_A_F="surge transfer, F flux force",
                 src_E="E/X planes V>0, Dm/Dt>0 (entry loss -1/2 c V^2)",
                 src_X="E/X planes V<=0, Dm/Dt>0 (gated: +1/2 c V^2)",
                 src_q3="E/X planes V>0, Dm/Dt<=0",
                 src_q4="E/X planes V<=0, Dm/Dt<=0",
                 src_Fl="F Lagrangian 1/2 V^2 dm0/dd (V - Vg)",
                 src_Fc="F convective rate (1/2 - gate) V^2 conv0",
                 P_T="transom outflow -u 1/2 M_T V_T^2 (station 0)",
                 P_st="state changes in the strips 1/2 dM V^2",
                 P_cf="cross-flow drag", P_ld="moments of the holding "
                 "force, friction, air drag", P_Mh="longitudinal hydrostatic"
                 " moment q Mh")
    for r in res:
        if isinstance(r, Exception):
            out(f"    job failed: {r!r}")
            ok = False
            continue
        rows.append(r)
        out(f"  {r['kn']:.1f} kn: envelope every 30 s "
            + " ".join(f"{a:.1e}" for a in r["env"][::3]))
        for k in GROUPS:
            sv = r["steady"].get(k if k != "P_T" else "P_T0", 0.0)
            out(f"    {k:7s} {r['means'][k]:+9.4f} W   {names[k]}"
                f"   (steady {sv:+.4f})")
        out(f"    sum of groups {sum(r['means'].values()):+9.4f} W; measured "
            f"d(E_total)/dt {r['dE_mean']:+9.4f} W (central differences "
            f"{r['dE_mean_cd']:+.4f}); residual {r['res3_mean']:+.4f} W = "
            f"{100 * abs(r['res3_mean']) / r['gross']:.2f} % of the gross "
            f"{r['gross']:.3f} W")
        out(f"      of which mechanical (EOM + hydrostatic potential) "
            f"{r['res1_mean']:+.2e} W (rms {r['res1_rms']:.2e}; hydrostatic "
            f"power vs -dU/dt rms {r['hs_cons']:.2e} of {r['hs_rms']:.2e}; "
            f"EOM split max {r['eomres']:.1e}); fluid (unexplained "
            f"transport D, steady-subtracted) {r['D_mean']:+.4f} W (steady "
            f"D {r['D_steady']:+.4f}); flux re-computed max diff "
            f"{r['flux_err']:.1e}; field re-advance identical "
            f"{r['readvance_same']}")
        out(f"      transport pieces: convective inertia -sum M V a_c "
            f"{r['conv']:+.4f}, advection u sum 1/2 V^2 dM/dx {r['adv']:+.4f}"
            f", their sum {r['conv'] + r['adv']:+.4f} vs transom outflow "
            f"(station 0) {r['means']['P_T']:+.4f} / (edge-extrapolated) "
            f"{r['Te']:+.4f}")
        good = abs(r["res3_mean"]) <= 0.05 * r["gross"]
        ok &= good
    check("budget", "B1: the budget closes: |mean of d(E_total)/dt - sum of "
          "the groups| <= 5 % of the gross of the group means, 1.0 and 1.5 "
          "kn", ok)
    dump("budget.json", dict(rows=rows))
    flush("budget.txt")


# ----------------------------------------------------------------- control
def control_job(arg):
    kn = arg
    warnings.filterwarnings("ignore")
    p, steady, rows, env = _brun(kn, 90.0)
    dt = 0.02
    idx = np.arange(int(round(30 / dt)), int(round(90 / dt)))
    fbar = np.mean([rows[n]["f"] for n in idx], axis=0)
    act = np.mean([rows[n]["act"] for n in idx], axis=0)
    # the region (amended before the control was evaluated: 'never
    # chine-located' selected no station -- at 1-1.5 kn every wet strip is
    # chine-located): aft of the CG with a mean F fraction >= 0.5; the bow
    # stations (x > 2 m, wet) as a second, recorded dashpot
    region = ((fbar >= 0.5) & (p.x < 0.0)).astype(float)
    bow = ((fbar > 0.0) & (p.x > 2.0)).astype(float)
    Px = float(np.mean([np.sum(rows[n]["addX"]) * p.dx for n in idx]))

    def cal(reg):
        G2 = float(np.mean([np.sum(rows[n]["g"] ** 2 * reg) * p.dx
                            for n in idx]))
        return -Px / G2 if G2 > 0 else 0.0
    cd, cdb = cal(region), cal(bow)
    out_ = dict(kn=kn, region=region.tolist(), bow=bow.tolist(), Px=Px,
                cd=cd, cd_bow=cdb, x=p.x.tolist(), fbar=fbar.tolist(),
                act=act.tolist())
    for variant, kw in (("dashpot", dict(cd=cd, region=region)),
                        ("dashpot_bow", dict(cd=cdb, region=bow)),
                        ("Xsuction", {})):
        _, _, _, e2 = _brun(kn, 240.0, keep=False,
                            variant=("dashpot" if variant.startswith(
                                "dashpot") else variant), **kw)
        out_[variant] = dict(env=e2, verdict=s1b_verdict(e2))
    return out_


def part_control():
    from studies.parallel_runner import run_parallel
    out("B2 (T9 control): the dissipation the Xsuction variant adds, on the "
        "plant's own 30-90 s trajectory, put instead as a dashpot -c_d g on "
        "the aft fixed-waterline stations (mean F fraction >= 0.5, x < 0; "
        "region amended, plan); S1 (b)'s verdict at 120 s")
    res = run_parallel(control_job, [1.0, 1.5], gb_per_job=0.5,
                       reserve_gb=3.0)
    rows = []
    for r in res:
        if isinstance(r, Exception):
            out(f"    job failed: {r!r}")
            continue
        rows.append(r)
        xr = [round(a, 2) for a, m in zip(r["x"], r["region"]) if m]
        xb = [round(a, 2) for a, m in zip(r["x"], r["bow"]) if m]
        out(f"  {r['kn']:.1f} kn: Xsuction's added power on the plant's "
            f"trajectory {r['Px']:+.4f} W; dashpot c_d {r['cd']:.1f} N s/m2 "
            f"on {len(xr)} stations x = {xr[0] if xr else None} .. "
            f"{xr[-1] if xr else None}; bow dashpot (recorded) c_d "
            f"{r['cd_bow']:.1f} on x = {xb[0] if xb else None} .. "
            f"{xb[-1] if xb else None}")
        for v in ("dashpot", "dashpot_bow", "Xsuction"):
            out(f"    {v:8s}: verdict (S1 b, 120 s) {r[v]['verdict']:18s}; "
                f"envelope every 30 s "
                + " ".join(f"{a:.1e}" for a in r[v]["env"][::3]))
    out("    (recorded; the reading is registered: if the dashpot also "
        "decays within 120 s, T9's location-specific claim is not "
        "supported)")
    dump("control.json", dict(rows=rows))
    flush("control.txt")


# ------------------------------------------------------------------ w5cap
def part_w5cap():
    """W5 (c, d) on a captive plant of its own (S1 b's): the derivative
    between steps equals the next step's first; replay from a recorded row
    bit for bit."""
    from studies import anyspeed_s1_check as S1
    out("W5 (c, d) on a captive plant (S1 b's run, 1.5 kn, root + 5 cm)")
    u = 1.5 * KN
    p = FX._plant(W, captive_u=u, dt=0.02, record=True)
    sol, res = S1.newton(p, u, np.array(p.running_attitude(u)))
    z0, th0 = sol
    s = S1.state(u, z0 + 0.05, th0)
    p.initial_state(0.0)
    calls = []
    orig = p.deriv

    def spy(s_, t_):
        r_ = orig(s_, t_)
        calls.append(r_[0].copy())
        return r_
    p.deriv = spy
    S = [s.copy()]
    same_d = None
    fsh = None
    for i in range(1500):
        if i == 1200:
            calls.clear()
            k_between, aux_b = orig(s, i * 0.02)
            k_between = k_between.copy()
            fsh = float(aux_b["kb"].f[aux_b["d"] > 0].mean())
            n0 = len(calls)
        s = p.step(s, i * 0.02, 0.0, 0.0)
        s[6] = u
        if i == 1200:
            same_d = bool(np.array_equal(k_between, calls[n0]))
        S.append(s.copy())
    # the class's method again (P1c round 3: an instance attribute 'deriv'
    # is copied into running_attitude's steady-state copy by trajectory())
    del p.deriv
    hid = p.trajectory()["hidden"]
    q_ = FX._plant(W, captive_u=u, dt=0.02)
    q_.initial_state(0.0)
    q_.set_hidden_state(hid[1000])
    s2 = S[1000].copy()
    same_c = True
    for i in range(1000, 1500):
        s2 = q_.step(s2, i * 0.02, 0.0, 0.0)
        s2[6] = u
        same_c &= bool(np.array_equal(s2, S[i + 1]))
    out(f"    F share at row 1200 (between steps) {fsh:.3f}; derivative "
        f"between steps == next step's first: {same_d}; replay rows "
        f"1000-1500 bit for bit: {same_c}; resets {p.es_resets}")
    check("w5cap", "W5 (d) on a captive plant: the derivative between steps "
          "equals the next step's first bit for bit", bool(same_d))
    check("w5cap", "W5 (c) on a captive plant: replay from a recorded row "
          "bit for bit", bool(same_c))
    dump("w5cap.json", dict(between=same_d, replay=same_c, F=fsh))
    flush("w5cap.txt")


PARTS = ("replay", "lowspeed", "goldenw", "w1b", "replay2", "budget",
         "control", "w5cap")


def _sub(arg):
    name, out_dir = arg
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"p1cw_{name}_stdout.log"), "w") as log:
        rc = subprocess.run([sys.executable, "-m", "studies.anyspeed_p1c_w",
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
        C.STAMP["p1c_w_sha1"] = hashlib.sha1(f.read()).hexdigest()
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
