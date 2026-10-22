#!/usr/bin/env python3
"""
The checks of step P1c, P line (learn/meta/ANYSPEED_PLAN_2026-10-05.md,
'P1c（P 线…）', 运行前登记的检查): the row-stable Froude-Krylov kernel
(sim/as_kernels.py), the implementation switch fast_path, the batch mode
(sim/planing_batch.BatchAS), the new goldens, E1 and the batch throughput.
Helpers, stamps and report format of studies/anyspeed_p1b_check.py; outputs
in studies/_cache/anyspeed/p1c/ (or --out).

Parts (the register's numbers):
  repin     C2  the fk_exact sets' golden runs against P1b's goldens:
                calm identical, task sea within 1e-10 (round-off)
  fastpath  C3  golden runs with and without fast_path, bit for bit
  batch     C4  N plants against BatchAS, every step, bit for bit
  kernel    C5  the kernel's section integrals against section_fk_grid;
                row-stability (several boats at once = each alone)
  goldens   C7  the new goldens, two processes, written once (never
                overwritten); sha1 printed for anyspeed_regress_as
  e1        C8  E1 by P0's interleaved method (--before <file>: P1b's
                module for the 'before' numbers)
  thru      C9  batch throughput, N = 1, 16, 128 (as1 preset)

    .venv/Scripts/python.exe -m studies.anyspeed_p1c_check --parts kernel
    .venv/Scripts/python.exe -m studies.anyspeed_p1c_check --jobs \\
        repin,fastpath,batch,kernel,goldens
"""
import argparse
import hashlib
import importlib.util
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

OUT = os.path.join(HERE, "_cache", "anyspeed", "p1c")
KN = C.KN
AS = C.AS
P1A = ("dDt_full", "chine_exact_all", "kpp_strip", "transom", "normal_rate",
       "verlet")
FK = P1A + ("fk_exact",)
FO = P1A + ("fk_exact", "orbital")
AS1 = P1A + ("fk_exact", "fast_path")
out, check, dump, flush = C.out, C.check, C.dump, C.flush
GOLD = os.path.join(HERE, "_cache", "anyspeed")
BEFORE = None


def _golden_runs(sw, selector=False, nozzle=True):
    """anyspeed_regress.golden_runs on PlaningVesselAS with switches sw (or
    the as1 hull's preset through the selector)."""
    from hydro import hulls
    from sim import physics
    from sim.planing_anyspeed import PlaningVesselAS
    from studies import anyspeed_regress as AR
    from studies.anyspeed_regress_as import ON_EXTRA
    warnings.filterwarnings("ignore")
    h = hulls.get(AS)
    if selector:
        tw = (lambda p: physics.gz_twin(h.plant(p.sea, dt=p.dt),
                                        record=True))
    else:
        tw = (lambda p: PlaningVesselAS(h, p.sea, dt=p.dt, record=True,
                                        test_switches=sw))
    return AR.golden_runs(tw, extra=ON_EXTRA, nozzle=nozzle)


def _cmp(a, b):
    """(identical keys, {key: largest difference / column max |b|})."""
    same, rel = [], {}
    for k in sorted(b):
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape == y.shape and np.array_equal(x, y):
            same.append(k)
            continue
        if x.shape != y.shape:
            rel[k] = float("inf")
            continue
        sc = np.maximum(np.abs(y).max(axis=0), 1e-300)
        rel[k] = float((np.abs(x - y) / sc).max())
    return same, rel


# ===================================================================== repin
def part_repin():
    out("C2 (repin): the fk_exact sets on the P1c kernel against P1b's "
        "goldens (calm identical, task sea within 1e-10 of each column's "
        "largest value)")
    rows = []
    for name, sw, f in (("P1a + fk_exact", FK, "golden_as_" + "+".join(FK)
                         + ".npz"),
                        ("P1a + fk_exact + orbital", FO,
                         "golden_as_" + "+".join(FO) + ".npz")):
        t0 = time.time()
        g = _golden_runs(sw)
        with np.load(os.path.join(GOLD, f)) as z:
            ref = {k: z[k] for k in z.files}
        same, rel = _cmp(g, ref)
        calm_ok = all(k in same for k in ref if k.startswith("calm_"))
        task = {k: v for k, v in rel.items() if k.startswith("task_")}
        worst = max(task.values(), default=0.0)
        rows.append(dict(name=name, file=f, identical=same, rel=rel,
                         seconds=time.time() - t0))
        out(f"    {name}: identical {len(same)}/{len(ref)} arrays; task-sea "
            f"largest relative difference {worst:.2e} "
            + " ".join(f"{k}:{v:.1e}" for k, v in sorted(task.items())))
        check("repin", f"C2 {name}: calm runs identical to {f}", calm_ok)
        check("repin", f"C2 {name}: task-sea runs within 1e-10 of {f}",
              worst <= 1e-10 and not (set(rel) - set(task)),
              f"{worst:.2e}")
    dump("repin.json", dict(rows=rows))
    flush("repin.txt")


# ================================================================== fastpath
def part_fastpath():
    out("C3 (fastpath): golden runs with and without fast_path, bit for bit")
    rows = []
    for name, a, b in (("P1a vs P1a + fast_path", (P1A, False),
                        (P1A + ("fast_path",), False)),
                       ("P1a + fk_exact vs the as1 preset (selector)",
                        (FK, False), (None, True))):
        ga = _golden_runs(a[0], selector=a[1])
        gb = _golden_runs(b[0], selector=b[1])
        same, rel = _cmp(gb, ga)
        ok = not rel and len(same) == len(ga)
        rows.append(dict(name=name, identical=len(same), of=len(ga),
                         differ=rel))
        out(f"    {name}: {len(same)}/{len(ga)} arrays identical"
            + (f"; differ {rel}" if rel else ""))
        check("fastpath", f"C3 {name}: every array identical", ok)
    from hydro import hulls
    from sim import physics
    from sim.planing_anyspeed import AS1_P1C
    p = physics.gz_twin(hulls.get(AS).plant(C.calm(), dt=0.02))
    check("fastpath", "the selector builds the as1 preset P1a + fk_exact + "
          "fast_path", tuple(p.switches) == tuple(AS1_P1C), p.switches)
    dump("fastpath.json", dict(rows=rows))
    flush("fastpath.txt")


# ===================================================================== batch
SEAS = ((1.0, 5.0, math.pi, 0), (1.5, 6.0, 0.75 * math.pi, 1),
        (0.6, 4.0, 0.0, 2), None, (1.2, 8.0, 0.5 * math.pi, 3),
        (1.0, 5.0, 1.25 * math.pi, 4), (0.8, 4.5, 0.9 * math.pi, 5),
        (1.0, 5.0, math.pi, 6), (1.3, 7.0, 0.3 * math.pi, 7))


def _sea(spec):
    if spec is None:
        return C.calm()
    from sim.wavefield import SeaState
    hs, tp, th, seed = spec
    return SeaState(hs, tp, theta0=th, n_freq=24, n_dir=5, seed=seed)


def _start(p, kind, u0):
    """Initial states: 'u' from initial_state(u0); 'astern' backing at
    -2 m/s; 'deep' buried (heave -1.5 m, bow-up 0.1, 20 m/s, sway, yaw:
    substeps); 'capsize' heeled 106 deg (the GZ boat capsizes at once)."""
    s = p.initial_state(max(u0, 0.0))
    if kind == "astern":
        s[6] = -2.0
        s[12] = -0.2 * p.prop.t_max
    elif kind == "deep":
        s[2] += -1.5
        s[4] += 0.1
        s[6], s[7], s[11] = 20.0, 3.0, 1.0
    elif kind == "capsize":
        s[3] = math.radians(106.0)
    return s


def _cmds(i, k, tm, rm, kind):
    thr = tm * (0.55 + 0.4 * math.sin(0.05 * k + 0.3 * i))
    if kind == "astern":
        thr = -0.3 * tm
    rud = rm * 0.8 * math.sin(0.03 * k + 0.5 * i)
    return thr, rud


MEAS = ("last_cg_acc", "last_bow_acc", "last_rel_bow", "last_slam_force",
        "peak_slam_force", "last_slam_force_full", "peak_slam_force_full",
        "last_n_sub", "slam_count")


def batch_case(sw, gz, boats, nstep, chunk):
    """N plants stepped one by one against a twin set stepped by BatchAS:
    (number of mismatches, first mismatches, the largest substep count,
    capsized boats)."""
    from hydro import hulls
    from sim import planing_anyspeed as PA
    from sim.planing_batch import BatchAS
    warnings.filterwarnings("ignore")
    h = hulls.get(AS)
    cls = PA.PlaningVesselAS if gz else PA.PlaningVesselASLin
    kw = dict(record=True) if gz else {}
    P = [cls(h, _sea(SEAS[b[0] % len(SEAS)]), dt=0.02, test_switches=sw,
             **kw) for b in boats]
    Q = [cls(h, _sea(SEAS[b[0] % len(SEAS)]), dt=0.02, test_switches=sw,
             **kw) for b in boats]
    S1 = [_start(P[i], b[1], b[2]) for i, b in enumerate(boats)]
    S2 = [_start(Q[i], b[1], b[2]) for i, b in enumerate(boats)]
    B = BatchAS(Q, chunk=chunk)
    S = np.array(S2)
    tm, rm = P[0].prop.t_max, P[0].rudder.max
    bad, first, nsub = 0, [], 0
    t = 0.0
    meas = MEAS + (("last_submergence",) if gz else ())
    for k in range(nstep):
        cm = [_cmds(i, k, tm, rm, b[1]) for i, b in enumerate(boats)]
        for i in range(len(P)):
            S1[i] = P[i].step(S1[i], t, cm[i][0], cm[i][1], 0.02)
        S = B.step(S, t, np.array([c[0] for c in cm]),
                   np.array([c[1] for c in cm]), 0.02)
        t += 0.02
        for i in range(len(P)):
            msg = []
            if not np.array_equal(S1[i], S[i]):
                msg.append("state")
            for nm in meas:
                a_, b_ = getattr(P[i], nm), getattr(Q[i], nm)
                if not (a_ == b_ or (a_ != a_ and b_ != b_)):
                    msg.append(nm)
            if tuple(P[i].last_jet_force) != tuple(Q[i].last_jet_force):
                msg.append("jet force")
            if P[i].prop.prime != Q[i].prop.prime:
                msg.append("prime")
            if P[i].prop.delay.buf != Q[i].prop.delay.buf or \
                    P[i].rudder.delay.buf != Q[i].rudder.delay.buf:
                msg.append("delay lines")
            if gz and (P[i].capsized != Q[i].capsized):
                msg.append("capsized")
            if msg:
                bad += 1
                if len(first) < 5:
                    first.append(f"step {k} boat {i} ({boats[i]}): "
                                 + ", ".join(msg))
            nsub = max(nsub, int(P[i].last_n_sub))
    if gz:
        for i in range(len(P)):
            ta, tb = P[i].trajectory(), Q[i].trajectory()
            for key in ta:
                if key == "meta":
                    continue
                va, vb = ta[key], tb[key]
                eq = (np.array_equal(np.asarray(va), np.asarray(vb))
                      if not isinstance(va, list) else va == vb)
                if not eq:
                    bad += 1
                    if len(first) < 5:
                        first.append(f"boat {i}: trajectory {key}")
    cap = [i for i in range(len(P)) if gz and P[i].capsized]
    return bad, first, nsub, cap


def _batch_job(arg):
    name, sw, gz, boats, nstep, chunk = arg
    t0 = time.time()
    bad, first, nsub, cap = batch_case(sw, gz, boats, nstep, chunk)
    return dict(name=name, mismatches=bad, first=first, n_sub_max=nsub,
                capsized=cap, boats=len(boats), steps=nstep, chunk=chunk,
                seconds=time.time() - t0)


BOATS8 = [(0, "u", 20.0 * KN), (1, "u", 12.0 * KN), (2, "u", 25.0 * KN),
          (3, "u", 5.0 * KN), (4, "u", 0.0), (5, "u", 30.0 * KN),
          (0, "astern", 0.0), (1, "deep", 10.0)]


def part_batch():
    from studies.parallel_runner import run_parallel
    out("C4 (batch): N plants against BatchAS, every step: states, "
        "measurements, actuators, the GZ boats' logs -- bit for bit")
    gz9 = BOATS8 + [(2, "capsize", 5.0 * KN)]
    jobs = [("as1 preset, GZ, chunk 8", P1A + ("fk_exact", "fast_path"),
             True, gz9, 300, 8),
            ("as1 preset, GZ, chunk 3", P1A + ("fk_exact", "fast_path"),
             True, gz9, 300, 3),
            ("as1 preset, GZ, chunk 1", P1A + ("fk_exact", "fast_path"),
             True, gz9, 120, 1),
            ("P1a, GZ, chunk 8", P1A, True, gz9, 300, 8),
            ("as1 preset, linear roll", P1A + ("fk_exact", "fast_path"),
             False, BOATS8, 300, 8),
            ("P1a, linear roll", P1A, False, BOATS8, 300, 8),
            ("P1a + fk_exact (no fast_path), GZ", FK, True, gz9, 200, 8),
            ("as1 preset, 16 boats", P1A + ("fk_exact", "fast_path"), True,
             [(i % len(SEAS), "u", (5.0 + 2.0 * i) * KN)
              for i in range(16)], 100, 8)]
    res = run_parallel(_batch_job, jobs, gb_per_job=0.4, reserve_gb=3.0,
                       max_procs=4)
    rows = []
    for j, r in zip(jobs, res):
        if isinstance(r, Exception):
            check("batch", f"C4 {j[0]}: ran", False, repr(r))
            continue
        rows.append(r)
        out(f"    {r['name']}: {r['boats']} boats x {r['steps']} steps, "
            f"mismatches {r['mismatches']}; largest substep count "
            f"{r['n_sub_max']}, capsized {r['capsized']} "
            f"({r['seconds']:.0f} s) {r['first']}")
        check("batch", f"C4 {r['name']}: bit for bit", r["mismatches"] == 0,
              f"{r['mismatches']} mismatches")
    subs = max((r["n_sub_max"] for r in rows), default=0)
    caps = any(r["capsized"] for r in rows)
    check("batch", "C4 the cases exercised substeps (> 1) and a capsize",
          subs > 1 and caps, f"substeps {subs}, capsize {caps}")
    _batch_refusals()
    dump("batch.json", dict(rows=rows))
    flush("batch.txt")


def _batch_refusals():
    """P1c review: BatchAS refuses what no batch case covers -- stations not
    equally spaced, a chine with interior knots in the strips -- and still
    accepts the registered hull. (A plant's cached geometry is altered after
    its build, which is all BatchAS reads for these two tests.)"""
    from hydro import hulls
    from sim import planing_anyspeed as PA
    from sim.planing_batch import BatchAS
    h = hulls.get(AS)

    def plant():
        p = PA.PlaningVesselAS(h, C.calm(), dt=0.02, test_switches=AS1)
        p._transom_geom()
        p._taper()
        return p

    def refused(p, what):
        try:
            BatchAS([p])
        except NotImplementedError as e:
            return what in str(e), str(e)
        return False, "accepted"
    ok0 = True
    try:
        BatchAS([plant()])
    except Exception as e:                       # noqa: BLE001
        ok0 = False
        msg0 = repr(e)
    check("batch", "C4 refusals: the registered hull is accepted", ok0,
          "accepted" if ok0 else msg0)
    p = plant()
    p.x = p.x.copy()
    p.x[2] += 0.01 * p.dx
    ok, msg = refused(p, "equally spaced")
    check("batch", "C4 refusals: stations not equally spaced", ok, msg)
    p = plant()
    tg = p._taper()
    tg.knots = np.full((tg.idx.size, 1), np.nan)
    ok, msg = refused(p, "knots")
    check("batch", "C4 refusals: interior chine knots", ok, msg)


# ==================================================================== kernel
def part_kernel():
    from hydro.section2d import section_fk_grid
    from sim import as_kernels as AK
    out("C5 (kernel): the row kernel's section integrals against "
        "hydro/section2d.section_fk_grid; row-stability")
    rng = np.random.default_rng(11)
    worst = 0.0
    nid = 0
    for trial in range(200):
        tb = math.tan(math.radians(float(rng.choice([10, 16, 20, 30]))))
        n, M = int(rng.integers(1, 40)), 120
        d = rng.uniform(1e-5, 0.8, n) * rng.choice([1.0, 1e-3], n)
        yc = rng.uniform(0.01, 1.1, n)
        k = np.repeat(rng.uniform(0.04, 8, 24), 5)
        lam = k * np.sin(rng.uniform(-np.pi, np.pi, M))
        lam[:5] = 0.0
        lam[5:10] *= 1e-8
        ku, kinv = np.unique(k, return_inverse=True)
        r1 = section_fk_grid(k, lam, d, yc, tb, ku, kinv)
        cvm = float((np.minimum(d, yc * tb) / tb).max(initial=0.0))
        r2 = AK._fk_sections(d, yc, tb, k, lam, ku, kinv, cvm)
        A = float((np.minimum(d, yc * tb) ** 2 / tb + 2 * yc * np.maximum(
            d - yc * tb, 0)).max())
        sc = [A, A * d.max(), A * yc.max(), 2 * yc.max(), 2 * yc.max() ** 2,
              yc.max()]
        e_ = max(float(np.max(np.abs(a_ - b_))) / s_
                 for a_, b_, s_ in zip(r1, r2, sc))
        worst = max(worst, e_)
        nid += all(np.array_equal(a_, b_) for a_, b_ in zip(r1, r2))
    out(f"    200 random row sets: largest difference {worst:.1e} of the "
        f"scale; identical {nid}/200")
    check("kernel", "C5 section integrals vs section_fk_grid within 1e-14 "
          "of the scale", worst <= 1e-14, f"{worst:.1e}")
    # row-stability: several boats at once == each alone
    from hydro import hulls
    from sim.planing_anyspeed import PlaningVesselAS
    h = hulls.get(AS)
    seas = [_sea(SEAS[i]) for i in (0, 1, 2, 4, 5, 6)]
    plants = [PlaningVesselAS(h, s_, dt=0.02, test_switches=AS1)
              for s_ in seas]
    bad = 0
    for trial in range(20):
        nb = int(rng.integers(1, 7))
        sel = rng.choice(len(plants), nb, replace=False)
        args = []
        for i in sel:
            p = plants[i]
            s = p.initial_state(float(rng.uniform(0, 15)))
            s[2] += rng.uniform(-0.15, 0.1)
            s[4] += rng.uniform(-0.05, 0.05)
            s[5] = rng.uniform(-np.pi, np.pi)
            t = float(rng.uniform(0, 100))
            p.deriv(s, t)
            wb, tr = p._wb, p._trs
            st = wb.st
            d = p.deriv(s, t)[1]["d"]
            A = p._section(d)[0]
            args.append(dict(
                d=d, A=A, Th=p.Th_rest + s[4], ab=float(p._a_b(s[6])),
                dT=tr["dT"], hw=tr["hw"], head=tr["head"], z0T=tr["z0T"],
                a=st.a, k=st.k, ku=st.ku, kinv=st.kinv, kap=wb.kap,
                lam=wb.lam, Z=wb.Z, CT=wb.CT, ST=wb.ST, e=wb.e))
        g = plants[0]._fk_geo()
        nu = max(a["ku"].size for a in args)

        def stack(key, pad=None):
            if pad is None:
                return np.stack([a[key] for a in args])
            o = np.ones((len(args), pad))
            for i, a in enumerate(args):
                o[i, :a[key].size] = a[key]
            return o
        allr = AK.fk_rows(
            g, stack("d"), stack("A"), stack("Th"), stack("ab"),
            stack("dT"), stack("hw"), stack("head"), stack("z0T"),
            AK.FKWaves(a=stack("a"), k=stack("k"), ku=stack("ku", nu),
                       kinv=stack("kinv"), kap=stack("kap"),
                       lam=stack("lam"), Z=stack("Z"), CT=stack("CT"),
                       ST=stack("ST"), e=stack("e")))
        for i, a in enumerate(args):
            one = AK.fk_rows(
                g, a["d"][None], a["A"][None], np.array([a["Th"]]),
                np.array([a["ab"]]), np.array([a["dT"]]),
                np.array([a["hw"]]), np.array([a["head"]]),
                np.array([a["z0T"]]),
                AK.FKWaves(a=a["a"][None], k=a["k"][None],
                           ku=a["ku"][None], kinv=a["kinv"][None],
                           kap=a["kap"][None], lam=a["lam"][None],
                           Z=a["Z"][None], CT=a["CT"][None],
                           ST=a["ST"][None], e=a["e"][None]))
            for f in ("fz", "fy", "x", "m", "my_sig", "n_sig", "K"):
                if not np.array_equal(getattr(allr, f)[i],
                                      getattr(one, f)[0]):
                    bad += 1
    out(f"    20 random groups of 1-6 boats: outputs differing from the "
        f"boat alone: {bad}")
    check("kernel", "C5 row-stable: several boats at once give each boat's "
          "own outputs, bit for bit", bad == 0, f"{bad}")
    flush("kernel.txt")


# =================================================================== goldens
GOLDENS = (("as1 preset (selector)", None, True, ""),
           ("P1a + fk_exact", FK, False, "_p1c"),
           ("P1a + fk_exact + orbital", FO, False, "_p1c"))


def golden_path(sw, suffix):
    from sim.planing_anyspeed import PRESETS
    sw = PRESETS["as1"] if sw is None else sw
    return os.path.join(GOLD, "golden_as_" + "+".join(sw) + suffix + ".npz")


def _golden_job(arg):
    sw, selector = arg
    return _golden_runs(sw, selector=selector)


def part_goldens():
    from studies.parallel_runner import run_parallel
    out("C7 (goldens): the new goldens, two processes each, written once")
    args = [(sw, sel) for _, sw, sel, _ in GOLDENS for _ in range(2)]
    res = run_parallel(_golden_job, args, gb_per_job=0.4, reserve_gb=3.0,
                       max_procs=4)
    rows = []
    for i, (name, sw, sel, suf) in enumerate(GOLDENS):
        a_, b_ = res[2 * i], res[2 * i + 1]
        if isinstance(a_, Exception) or isinstance(b_, Exception):
            check("goldens", f"C7 {name}: runs", False, f"{a_!r} {b_!r}")
            continue
        same = set(a_) == set(b_) and all(
            np.array_equal(np.asarray(a_[k]), np.asarray(b_[k])) for k in a_)
        heel = max(float(np.degrees(np.abs(np.asarray(v)[:, 3]).max()))
                   for k, v in a_.items()
                   if k.startswith("calm_") and k.endswith("_s"))
        path = golden_path(sw, suf)
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
        rows.append(dict(name=name, path=path, sha1=sha, same=same,
                         heel_calm=heel, file=wrote))
        out(f"    {name}: two processes identical {same}; largest heel of "
            f"the calm runs {heel:.1f} deg; {os.path.basename(path)} {wrote}"
            f", sha1 {sha}")
        check("goldens", f"C7 {name}: identical in two processes", same)
        check("goldens", f"C7 {name}: calm runs' largest heel <= 15 deg",
              heel <= 15.0, f"{heel:.1f}")
        check("goldens", f"C7 {name}: file new or identical",
              not wrote.startswith("exists; DIFFERS"), wrote)
    dump("goldens.json", dict(rows=rows))
    flush("goldens.txt")


# ======================================================================= e1
def _load_before(path):
    spec = importlib.util.spec_from_file_location("sim._p1c_before", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sim._p1c_before"] = mod
    spec.loader.exec_module(mod)
    return mod


def part_e1():
    from hydro import hulls
    from learn.repro import task
    from sim import physics
    from sim import planing_anyspeed as PA
    from studies import anyspeed_p0_check as P0
    out("C8 (e1): ms per 0.02 s step of the GZ plant in the task sea "
        "(mission s0 leg 0), 0.9 u_design, thrust 0.6 +- 0.3 t_max, nozzle "
        "0.5 x max sinusoid (P1b's E1 run); P0's interleaved rounds, ratio "
        "to v0 x v0's idle cost " + f"{P0.V0_IDLE_MS} ms")
    m = task.Mission("high", 0, 0, t_end=1.0, track=task.LEGS[0])
    sea = m.plant.sea
    h = hulls.get(AS)
    plants = {"v0": physics.gz_twin(hulls.get("scarab195").plant(sea,
                                                                 dt=0.02))}
    if BEFORE:
        old = _load_before(BEFORE)
        plants["P1a (P1b code)"] = old.PlaningVesselAS(h, sea, dt=0.02,
                                                        test_switches=P1A)
        plants["P1a+fk (P1b code)"] = old.PlaningVesselAS(
            h, sea, dt=0.02, test_switches=FK)
    for k, sw in (("P1a", P1A), ("P1a+fast_path", P1A + ("fast_path",)),
                  ("P1a+fk_exact", FK), ("as1 preset", AS1)):
        plants[k] = PA.PlaningVesselAS(h, sea, dt=0.02, test_switches=sw)
    keys = list(plants)
    N, rounds = 300, 9
    per = {k: [] for k in keys}
    heel = {k: 0.0 for k in keys}
    for r in range(rounds):
        order = keys[r % len(keys):] + keys[:r % len(keys)]
        for k in order:
            p = plants[k]
            s_ = p.initial_state(0.9 * float(p.hull.u_design))
            tm, rm = p.prop.t_max, p.rudder.max
            ta = time.perf_counter()
            for i in range(N):
                s_ = p.step(s_, i * 0.02, tm * (0.6 + 0.3 * math.sin(
                    0.05 * i)), rm * 0.5 * math.sin(0.03 * i), 0.02)
                heel[k] = max(heel[k], abs(float(s_[3])))
            per[k].append(1e3 * (time.perf_counter() - ta) / N)
    med = {k: float(np.median(v)) for k, v in per.items()}
    rat = {k: float(np.median(np.array(per[k]) / np.array(per["v0"])))
           for k in keys[1:]}
    est = {k: rat[k] * P0.V0_IDLE_MS for k in rat}
    for k in keys:
        out(f"    {k:22s} median {med[k]:.3f} ms" + (
            f", / v0 {rat[k]:.3f} -> idle estimate {est[k]:.2f} ms"
            if k in rat else "") + f"; largest heel "
            f"{math.degrees(heel[k]):.1f} deg")
    out("    (absolute idle estimates: about +-10 % under load -- P1c review; "
        "differences between switch sets: the paired blocks below)")
    for k in ("P1a", "P1a+fast_path", "as1 preset"):
        check("e1", f"C8 E1: {k} at most {P0.E1_MS} ms per step",
              est[k] <= P0.E1_MS, f"{est[k]:.2f} ms")
    paired = _e1_paired(plants)
    dump("e1.json", dict(per_round_ms=per, median_ms=med, ratio_to_v0=rat,
                         idle_estimate_ms=est, before=BEFORE, paired=paired))
    flush("e1.txt")


# the pairs of switch sets compared by paired alternating blocks (P1c review:
# the differences between sets are a few per cent, the run-to-run noise
# under load up to ~25 %): (reference, candidate)
E1_PAIRS = (("P1a", "P1a+fast_path"), ("P1a+fk_exact", "as1 preset"),
            ("P1a", "P1a+fk_exact"))


def _e1_paired(plants, rounds=12, nb=150):
    """Paired alternating blocks: for each pair (a, b), `rounds` rounds of
    one block of nb steps of each (order a b, then b a, alternating), every
    block from the same start (initial_state(0.9 u_design)) with the same
    commands -- the same trajectory for a bit-identical pair; the ratio of
    the two blocks of a round, its median and quartiles over the rounds."""
    out(f"    paired alternating blocks ({rounds} rounds x {nb} steps, each "
        "block from the same start): time ratio candidate / reference")
    res = {}

    def block(p):
        s_ = p.initial_state(0.9 * float(p.hull.u_design))
        tm, rm = p.prop.t_max, p.rudder.max
        ta = time.perf_counter()
        for i in range(nb):
            s_ = p.step(s_, i * 0.02, tm * (0.6 + 0.3 * math.sin(0.05 * i)),
                        rm * 0.5 * math.sin(0.03 * i), 0.02)
        return time.perf_counter() - ta
    for a, b in E1_PAIRS:
        if a not in plants or b not in plants:
            continue
        block(plants[a])
        block(plants[b])
        r_ = []
        for r in range(rounds):
            if r % 2 == 0:
                ta_, tb_ = block(plants[a]), block(plants[b])
            else:
                tb_, ta_ = block(plants[b]), block(plants[a])
            r_.append(tb_ / ta_)
        q1, q2, q3 = (float(v) for v in np.percentile(r_, [25, 50, 75]))
        res[f"{b} / {a}"] = dict(ratios=r_, median=q2, q1=q1, q3=q3,
                                 lo=float(min(r_)), hi=float(max(r_)))
        out(f"      {b:16s} / {a:14s}: median {q2:.3f} (quartiles {q1:.3f}"
            f"-{q3:.3f}, range {min(r_):.3f}-{max(r_):.3f})")
    return res


# ===================================================================== thru
def part_thru():
    from hydro import hulls
    from sim import physics
    from sim import planing_anyspeed as PA
    from sim.planing_batch import BatchAS
    from studies import anyspeed_p0_check as P0
    out("C9 (thru): the batch mode's throughput (as1 preset, GZ plants, "
        "task-like seas: Hs 1 m, Tp 5 s, 24 x 5 components, heading and "
        "seed per boat; start 0.9 u_design; thrust and nozzle per boat); "
        "boat-steps per second, against loops of single plants (the same "
        "boats) and v0 in the same process. Every configuration over the "
        f"same window (P1c review): {THRU_WARM} warm-up steps from t = 0, then {THRU_STEPS} "
        f"timed steps, advanced in blocks of {THRU_BLOCK} with the "
        "configurations alternating (rotated order) so that load changes "
        "fall on all of them; gains from the paired blocks")
    h = hulls.get(AS)
    tm = None

    def seas(N):
        from sim.wavefield import SeaState
        return [SeaState(1.0, 5.0, theta0=math.pi + 0.05 * i, n_freq=24,
                         n_dir=5, seed=100 + i) for i in range(N)]

    def cmds(N, k):
        i = np.arange(N)
        return (tm * (0.6 + 0.3 * np.sin(0.05 * k + 0.4 * i)),
                rm * 0.5 * np.sin(0.03 * k + 0.7 * i))
    v0 = physics.gz_twin(hulls.get("scarab195").plant(seas(1)[0], dt=0.02))
    tm, rm = v0.prop.t_max, v0.rudder.max
    # the configurations, each with its own boats, states and step counter
    # k (the commands' argument; t = 0.02 k), all from t = 0
    cfg = {}
    for N in (1, 16, 128):
        P = [PA.PlaningVesselAS(h, s_, dt=0.02, test_switches=AS1)
             for s_ in seas(N)]
        S = np.array([p.initial_state(0.9 * float(h.u_design)) for p in P])
        cfg[f"batch N = {N}"] = dict(kind="batch", N=N, B=BatchAS(P), S=S,
                                      k=0)
    for N in (1, 16, 128):
        # the same boats as the batch of N, one plant step at a time
        P = [PA.PlaningVesselAS(h, s_, dt=0.02, test_switches=AS1)
             for s_ in seas(N)]
        cfg[f"single x {N}"] = dict(
            kind="single", N=N, P=P, k=0,
            S=[p.initial_state(0.9 * float(h.u_design)) for p in P])
    cfg["v0 (1 plant)"] = dict(kind="single", N=1, P=[v0], k=0,
                               S=[v0.initial_state(0.9 * float(
                                   v0.hull.u_design))])

    def advance(c, n):
        ta = time.perf_counter()
        for _ in range(n):
            k = c["k"]
            th, ru = cmds(c["N"], k)
            if c["kind"] == "batch":
                c["S"] = c["B"].step(c["S"], k * 0.02, th, ru)
            else:
                for i, p in enumerate(c["P"]):
                    c["S"][i] = p.step(c["S"][i], k * 0.02, th[i], ru[i],
                                       0.02)
            c["k"] = k + 1
        return time.perf_counter() - ta
    names = list(cfg)
    for nm in names:
        advance(cfg[nm], THRU_WARM)
    per = {nm: [] for nm in names}         # ms per boat-step, each block
    nr = THRU_STEPS // THRU_BLOCK
    for r in range(nr):
        order = names[r % len(names):] + names[:r % len(names)]
        for nm in order:
            c = cfg[nm]
            per[nm].append(1e3 * advance(c, THRU_BLOCK)
                           / (THRU_BLOCK * c["N"]))
    for nm in names:
        assert cfg[nm]["k"] == THRU_WARM + nr * THRU_BLOCK
    v0b = np.array(per["v0 (1 plant)"])
    rows = []
    for nm in names[:-1]:
        a = np.array(per[nm])
        tot = float(a.mean())                       # the same window each
        idle = tot / float(v0b.mean()) * P0.V0_IDLE_MS
        rows.append(dict(name=nm, N=cfg[nm]["N"], warm=THRU_WARM,
                         steps=nr * THRU_BLOCK, block_ms=per[nm],
                         ms_per_boat_step=tot, boat_steps_per_s=1e3 / tot,
                         idle_ms_per_boat_step=idle))
        out(f"    {nm:17s}: {1e3 / tot:8.1f} boat-steps/s, {tot:6.2f} ms "
            f"per boat-step over the window (idle estimate {idle:.2f} ms, "
            f"/ v0 over the same blocks x {P0.V0_IDLE_MS} ms)")
    gains = {}
    for N in (1, 16, 128):
        nm = f"batch N = {N}"
        g_ = np.array(per[f"single x {N}"]) / np.array(per[nm])
        q1, q2, q3 = (float(v) for v in np.percentile(g_, [25, 50, 75]))
        gains[nm] = dict(per_block=g_.tolist(), median=q2, q1=q1, q3=q3)
        out(f"    {nm:17s} against the same {N} boats as single plants "
            f"(paired blocks): throughput x {q2:.3f} (quartiles {q1:.3f}-"
            f"{q3:.3f})")
    dump("thru.json", dict(rows=rows, gains=gains,
                           v0_block_ms=per["v0 (1 plant)"]))
    flush("thru.txt")


# part thru's window (P1c review: every configuration over the same steps)
THRU_WARM, THRU_STEPS, THRU_BLOCK = 3, 80, 10

PARTS = ("repin", "fastpath", "batch", "kernel", "goldens", "e1", "thru")


def _sub(arg):
    name, out_dir, extra = arg
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"p1c_{name}_stdout.log"), "w") as log:
        rc = subprocess.run([sys.executable, "-m",
                             "studies.anyspeed_p1c_check", "--parts", name,
                             "--out", out_dir] + extra, cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    return name, rc, time.time() - t0


def main():
    global OUT, BEFORE
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="")
    ap.add_argument("--jobs", default="", help="parts, one process each")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--before", default=None,
                    help="part e1: P1b's sim/planing_anyspeed.py (a copy) "
                    "for the 'before' numbers")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    OUT = os.path.abspath(a.out)
    BEFORE = a.before
    C.OUT = OUT
    if a.jobs:
        from studies.parallel_runner import run_parallel
        t0 = time.time()
        extra = ["--before", a.before] if a.before else []
        res = run_parallel(_sub, [(x, OUT, extra) for x in a.jobs.split(",")
                                  if x], gb_per_job=0.5, reserve_gb=3.0,
                           max_procs=3)
        for r in res:
            print(f"  {r}", flush=True)
        print(f"all done in {time.time() - t0:.0f} s wall", flush=True)
        return
    C.CAND = AS1
    C.STAMP = C.stamp_all()
    with open(os.path.abspath(__file__), "rb") as f:
        C.STAMP["p1c_check_sha1"] = hashlib.sha1(f.read()).hexdigest()
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
