#!/usr/bin/env python3
"""
Verification of step P0 of the any-speed plan -- the modular split
(learn/meta/ANYSPEED_PRINCIPLES_2026-10-05.md sections 1 and 7; ACCEPTANCE
G4, N3, M2, M3, M5, T): the hull as data (hydro/hull_geometry.py,
hydro/vessel_spec.py), the as1 plant reading it (sim/planing_anyspeed.py),
caches keyed by physics + hull sha1 (sim/physics.py, sim/vessel_profile.py,
studies/final_eval.turn_path), the time-scale-free MPPI option
(control/mpc.py) and the strip-exact chine location for every strip
(switch chine_exact_all, orchestrator decision 2 on S1).

Parts (outputs in studies/_cache/anyspeed/p0/; quick tier: geom, cap,
cache, mppi -- a few minutes; full tier: twin, hullb, regress):
  geom    geometry exactness. For scarab195 and random members of v0's
          family (L, b, deadrise, taper, stem rise, depth, station count),
          the as plant with every switch off, its geometry from the module,
          against v0's own lines: every station array, the rest attitude,
          GM, lateral masses, the jet constants and the GZ table, EXACTLY.
          The parametric representation of scarab195
          (vessel_spec.parametric_of) runs bit for bit as the registered
          scarab195@as1. The family's chine: roots of kc d = yc per piece
          (residual), concavity. An offsets table sampled from planing_b at
          13, 27, 53, 107 offsets (counts that put no grid of 48 or 50
          strips on the knots: the review found 97 offsets aligned with
          every centre of 48 strips, a false 'order 8'): chine, keel, keel
          slope and the keel curvature (the strip integral; PCHIP's centre
          value beside it) and the plant's steady trim converge to the
          family's, with the measured order (M5)
  cap     the strip-exact chine location (sim/planing_anyspeed._substrip,
          now driven by the geometry): against an independent reference
          (dense sampling + brentq roots of kc d+ = yc, analytic integrals),
          on v0's chine, planing_b's (transom run, cubic keel) and a
          non-concave offsets chine, taper strips and all strips; M2
          criterion 1e-8 relative. Switch chine_exact_all: continuity of
          the momentum-flux force across prismatic chine capping (jump vs
          dz), and station-count convergence of the planing equilibria
          (48 ... 768 strips, Newton roots of the captive equations),
          dDt_full alone vs dDt_full + chine_exact_all, measured order (M5);
          the discrete momentum: strip-averaged vs centre m_a by station
          count (the flux differentiates the first, the inertia uses the
          second)
  cache   N3: physics.cache_tag (legacy name for scarab195 v0 only),
          final_eval.turn_path per hull, profile directories per physics
          sha1 and their meta check (protocol included: a changed protocol
          constant or Froude-number list recomputes), the steady-state cache
          under id reuse and keyed by the exact speed, actuators carried by
          with_sea / gz_twin of an as plant and dropped by v0's route,
          legacy cache files unchanged and no lofi_*.json created
  mppi    M7: control/mpc.MPPIController's time_ref option. On a Froude
          twin's reduced model (the base's identified coefficients scaled by
          their exponents) in a Froude-scaled sea, the same candidate plans
          cost the same with time_ref = froude_time_ref(L) and differ
          without; time_ref = froude_time_ref(5.4) on scarab195 equals the
          default, bit for bit
  steptime ACCEPTANCE E1 (3 ms per 0.02 s step, single core, task sea):
          scarab195 v0, scarab195@as1 and planing_b@as1 GZ plants timed in
          interleaved rounds; the as1 / v0 ratio of each round cancels the
          machine's load, times v0's idle cost (1.67 ms, the review's idle
          measurement) is the estimate checked against 3 ms
  twin    the Froude twin (principles T1; ACCEPTANCE M3, 1e-4): scarab195
          at lam 1 (its parametric representation), 1.5 and 0.6, every
          input scaled (vessel_spec.froude_twin), on as1: build and limits,
          steady attitude and resistance at six Froude numbers, the
          identified reduced model (coefficient / lam^exponent), the
          Froude-exact turning table, and a 600-step open-loop run in a
          Froude-scaled short-crested sea -- all nondimensional, against
          scarab195@as1; one process per hull (studies/parallel_runner).
          Beside the M3 limit, an internal bound of 1e-10 on all of them:
          with every input scaled the twin runs the same arithmetic, so
          anything above round-off is a defect (the review found the
          steady-state cache's 1 mm/s rounding this way)
  hullb   the second parametric hull end to end: planing_b refused by v0,
          planing_b@as1 built; its profile (limits, steady curve with
          captive residuals, reduced model, turning table) computed and
          cached by physics sha1; an open-loop run in a sea; Savitsky
          (prismatic) for information; the inputs it took from defaults
  hullb_s1 S1's standing checks on planing_b@as1 (the hull with the
          transom-run chine roots, the cubic keel and stem rise), through
          S1's own functions: root census and winding scan at 10 speeds
          (exactly one stable root, no zero away from the roots), one
          stepped sweep up and down (no limit cycle: trim p-p <= 0.01 deg,
          z p-p <= 1 mm; up-down trim difference <= 0.05 deg), decay from 1
          mm and 5 cm at 3, 6, 9, 12 kn (S1's verdict 'decays'), and D55
          against S1's continuum reference (now read from hydro/
          hull_geometry) at 2, 4, 6 kn (within 5 %); three processes
  regress v0 bit-identical and as1 exact: studies/anyspeed_regress_as.py
          --all and studies/anyspeed_regress.py (golden, lofi, turn,
          streams, the hand and m0 missions) as subprocesses

    .venv/Scripts/python.exe -m studies.anyspeed_p0_check --geom --cap \\
        --cache --mppi --steptime
    .venv/Scripts/python.exe -m studies.anyspeed_p0_check --twin --hullb \\
        --hullb_s1
    .venv/Scripts/python.exe -m studies.anyspeed_p0_check --regress
"""
import argparse
import copy
import gc
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

OUT = os.path.join(HERE, "_cache", "anyspeed", "p0")
G = 9.81
KN = 0.514444
LINES = []
RESULTS = {}


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def check(part, name, ok, detail=""):
    ok = bool(ok)
    RESULTS.setdefault(part, []).append(dict(name=name, ok=ok,
                                             detail=str(detail)))
    out(f"  [{'ok' if ok else 'FAIL'}] {part}: {name}"
        + (f"  -- {detail}" if detail != "" else ""))
    return ok


# a stamp line a caller may set (studies/anyspeed_p1a_check.py: the physics
# and code the outputs come from); empty by default, so this script's own
# outputs are unchanged
STAMP_LINE = ""


def flush(name):
    os.makedirs(OUT, exist_ok=True)
    head = [STAMP_LINE] if STAMP_LINE else []
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write("\n".join(head + LINES) + "\n")
    LINES.clear()


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    if STAMP_LINE and isinstance(obj, dict):
        obj = dict(obj, _stamp=STAMP_LINE)
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda v: (
            v.tolist() if isinstance(v, np.ndarray) else
            v.item() if isinstance(v, np.generic) else str(v)))


def calm():
    from sim.test_vessel import Monochromatic
    return Monochromatic(1.0, 0.0)


def sha1_file(path):
    with open(os.path.join(ROOT, path), "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def order(errs, ratio=2.0):
    """Observed orders log(e_k / e_{k+1}) / log(ratio)."""
    e = np.asarray(errs, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.log(e[:-1] / e[1:]) / np.log(ratio)).tolist()


# =================================================================== geom
PLANT_ARRAYS = ("x", "dx", "yc", "zk", "kp", "kpp", "Z_rest", "Th_rest", "T",
                "GM", "M", "Mtot", "visc", "draft_bow", "freeboard",
                "v_slam")
JET_CONSTS = ("t_max", "t_min", "tau", "rate_max", "dead_band",
              "tau_prime_loss", "tau_prime_back", "prime_band", "k_rev",
              "u_ref")


def random_v0_hulls(n, seed=7):
    from hydro.planing import PlaningHull
    rng = np.random.default_rng(seed)
    out_ = []
    for i in range(n):
        L = float(rng.uniform(4.0, 9.0))
        b = float(rng.uniform(0.3, 0.42) * L)
        beta = float(rng.uniform(10.0, 25.0))
        mass = float(1400.0 * (L / 5.4) ** 3 * rng.uniform(0.8, 1.2))
        out_.append(PlaningHull(
            name=f"p0_rand{i}", L=L, b=b, beta_deg=beta, mass=mass,
            lcg=float(rng.uniform(0.36, 0.42) * L),
            vcg=float(rng.uniform(0.09, 0.12) * L),
            depth=float(rng.uniform(0.17, 0.25) * L),
            taper=float(rng.uniform(0.2, 0.5)),
            stem_rise=float(rng.choice([0.0, rng.uniform(0.05, 0.4)])),
            n_stations=int(rng.choice([24, 48, 61])),
            jet=dict(d_nozzle=0.085 * L / 5.4, p_max=224e3 * (L / 5.4) ** 3.5)))
    return out_


def part_geom():
    from hydro import hulls, vessel_spec as VS, hull_geometry as HG
    from sim import physics
    from sim.planing_vessel import PlaningVessel
    from sim.planing_vessel_gz import PlaningVesselGZ
    from sim.planing_anyspeed import PlaningVesselAS, PlaningVesselASLin
    t0 = time.time()
    out("GEOM: the geometry module against v0's own lines")
    rows = []
    for h in [hulls.get("scarab195")] + random_v0_hulls(5):
        ha = (hulls.get("scarab195@as1") if h.name == "scarab195"
              else physics.variant(h, "as1"))
        p0 = PlaningVessel(h, calm(), dt=0.02)
        pa = PlaningVesselASLin(ha, calm(), dt=0.02, test_switches="v0")
        neq = [k for k in PLANT_ARRAYS if not np.array_equal(getattr(p0, k),
                                                             getattr(pa, k))]
        neq += [f"prop.{k}" for k in JET_CONSTS
                if getattr(p0.prop, k) != getattr(pa.prop, k)]
        st = HG.stations_of(h)
        neq += [f"stations.{k}" for k, a in (("yc", p0.yc), ("kp", p0.kp),
                                            ("kpp", p0.kpp))
                if not np.array_equal(getattr(st, k), a)]
        if not np.array_equal(-h.vcg + st.rise, p0.zk):
            neq.append("stations.rise")
        g0 = PlaningVesselGZ(h, calm(), dt=0.02)
        ga = PlaningVesselAS(ha, calm(), dt=0.02, test_switches="v0")
        if not np.array_equal(g0.gz_tab, ga.gz_tab):
            neq.append("gz_tab")
        s0, sa = p0.initial_state(0.9 * h.u_design * h.L / 5.4), \
            pa.initial_state(0.9 * h.u_design * h.L / 5.4)
        for i in range(40):
            s0 = p0.step(s0, i * 0.02, 0.5 * p0.prop.t_max, 0.1)
            sa = pa.step(sa, i * 0.02, 0.5 * pa.prop.t_max, 0.1)
        if not np.array_equal(s0, sa):
            neq.append("40-step run")
        rows.append(dict(hull=h.name, L=h.L, n=h.n_stations,
                         stem_rise=h.stem_rise, differ=neq,
                         v0_member=HG.form_of(h).v0_member(),
                         v0_ok=VS.v0_representable(h)[0]))
        check("geom", f"{h.name} (L {h.L:.2f}, n {h.n_stations}, stem rise "
              f"{h.stem_rise:.2f}): as plant all off == v0, arrays, jet, GZ "
              f"table, 40 steps, exactly; v0's family member",
              not neq and rows[-1]["v0_member"] and rows[-1]["v0_ok"],
              f"differ {neq}" if neq else "identical")
    # the parametric representation of scarab195 runs as the registered hull
    h1 = hulls.get("scarab195@as1")
    hp = physics.variant(VS.parametric_of(hulls.get("scarab195")), "as1")
    pa, pp = h1.plant(calm()), hp.plant(calm())
    sa, sp = pa.initial_state(12.0), pp.initial_state(12.0)
    for i in range(300):
        c = (0.55 + 0.35 * math.sin(0.05 * i), 0.3 * math.sin(0.03 * i))
        sa = pa.step(sa, i * 0.02, c[0] * pa.prop.t_max, c[1])
        sp = pp.step(sp, i * 0.02, c[0] * pp.prop.t_max, c[1])
    check("geom", "vessel_spec.parametric_of(scarab195) on as1 (its own "
          "manifest) runs 300 steps bit for bit as scarab195@as1",
          np.array_equal(sa, sp) and physics.hull_sha1(hp)
          != physics.hull_sha1(h1) and pa.switches == pp.switches,
          f"max |diff| {np.abs(sa - sp).max():.1e}")
    # the family's chine pieces: roots and concavity, planing_b
    hb = hulls.get("planing_b")
    f = HG.form_of(hb)
    st = f.stations()
    xs = np.linspace(0.0, f.L, 20001)
    y = st.chine(xs)
    d2 = y[:-2] - 2 * y[1:-1] + y[2:]
    ends = np.array([f.transom_run * f.L, (1.0 - f.taper) * f.L])
    jump = float(np.abs(st.chine(ends * (1 + 1e-13))
                        - st.chine(ends * (1 - 1e-13))).max())
    rng = np.random.default_rng(3)
    kc = 0.5 * np.pi / math.tan(math.radians(f.beta_deg))
    from scipy.optimize import brentq
    n_true, missed, worst = 0, 0, 0.0
    for _ in range(300):
        xc = float(rng.uniform(0.0, f.L))
        di = float(rng.uniform(0.0, 0.4))
        mi = float(rng.uniform(-0.3, 0.3))
        cand = np.array([float(r[0]) for r in st.roots(
            np.array([xc]), np.array([di]), np.array([mi]), kc)])
        cand = cand[np.isfinite(cand)]

        def fx(x):
            return kc * (di + mi * (x - xc)) - float(st.chine(x))
        grid = np.linspace(0.0, f.L, 2001)
        fg = np.array([fx(x) for x in grid])
        for i in np.nonzero(fg[:-1] * fg[1:] < 0.0)[0]:
            r = brentq(fx, grid[i], grid[i + 1], xtol=1e-15, rtol=1e-15)
            n_true += 1
            dmin = float(np.abs(cand - r).min()) if cand.size else np.inf
            worst = max(worst, dmin)
            missed += dmin > 1e-9
    check("geom", "planing_b's chine (transom run, flat, elliptic taper): "
          "concave and continuous; every true root of kc d = yc (dense "
          "sampling + brentq) is among the geometry's candidate breakpoints",
          d2.max() < 1e-12 and jump < 1e-9 and missed == 0 and n_true > 50,
          f"max 2nd difference {d2.max():.1e}, jump at the piece ends "
          f"{jump:.1e} m, {n_true} true roots, missed {missed}, farthest "
          f"candidate {worst:.1e} m")
    # an offsets table sampled from planing_b converges to the family:
    # chine on a fine grid, keel, slope and curvature at the centres of 50
    # strips, the plant's steady trim (48 strips). The offset counts put no
    # grid's every centre on a knot (n_off - 1 = 12, 26, 52, 106 against
    # 2 x 48 and 2 x 50 half-strips): at 97 offsets each centre of 48
    # strips is a knot and the station chine equals the family to round-off
    # -- the false 'order 8' of the first report
    import dataclasses
    from scipy.interpolate import PchipInterpolator
    f50 = dataclasses.replace(f, n_stations=50)
    s50 = f50.stations()
    kmax = float(np.abs(s50.kpp_strip).max())
    xf = np.linspace(0.0, f.L, 4001)
    conv = []
    offs = (13, 27, 53, 107)
    aligned = {}
    for n_off in offs:
        of = f.offsets(n_off)
        of50 = dataclasses.replace(of, n_stations=50)
        so = of50.stations()
        sp_ = PchipInterpolator(of.x, of.keel - of.keel[0])
        for ns in (48, 50):
            xc = (np.arange(ns) + 0.5) * f.L / ns
            aligned[(n_off, ns)] = int(sum(
                np.abs(of.x - x_).min() < 1e-9 * f.L for x_ in xc))
        conv.append(dict(
            n_off=n_off,
            yc_max=float(np.abs(of.stations().chine(xf) - st.chine(xf)).max()),
            yc_l1=float(np.abs(of.stations().chine(xf)
                               - st.chine(xf)).mean()),
            rise=float(np.abs(so.rise - s50.rise).max()),
            kp=float(np.abs(so.kp - s50.kp).max()),
            # the keel curvature the offsets plant uses (strip integral)
            # against the family's exact strip integral, over the largest
            # curvature; and PCHIP's own centre value (P0's first version)
            # against the family's centre value, for comparison
            kpp_strip=float(np.abs(so.kpp - s50.kpp_strip).max()) / kmax,
            kpp_centre=float(np.abs(sp_(so.xi, 2) - s50.kpp).max()) / kmax))
    out(f"    strip centres on an offset knot (48 / 50 strips): "
        + ", ".join(f"{n}: {aligned[(n, 48)]} / {aligned[(n, 50)]}"
                    for n in offs))
    trims = {}
    pb = hulls.get("planing_b@as1").plant(calm())
    fam = {}
    for Fn in (0.8, 1.4):
        fam[Fn] = pb.calm_resistance(Fn * math.sqrt(G * f.L))["trim_deg"]
    for n_off in offs:
        ho = VS.ParametricPlaningHull(
            f"planing_b_off{n_off}", f.offsets(n_off),
            dict(mass=hb.mass, lcg=hb.lcg, vcg=hb.vcg),
            jet=hb.jet, closures=dict(cda=hb.cda), u_design=hb.u_design)
        po = physics.variant(ho, "as1").plant(calm())
        for Fn in (0.8, 1.4):
            trims.setdefault(Fn, []).append(abs(po.calm_resistance(
                Fn * math.sqrt(G * f.L))["trim_deg"] - fam[Fn]))
    ratios = [(offs[i + 1] - 1) / (offs[i] - 1) for i in range(len(offs) - 1)]

    def orders(e):
        e = np.asarray(e, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return [float(np.log(e[i] / e[i + 1]) / np.log(ratios[i]))
                    for i in range(len(e) - 1)]
    for k in ("yc_max", "yc_l1", "rise", "kp", "kpp_strip", "kpp_centre"):
        e = [c[k] for c in conv]
        out(f"    offsets -> family, {k} error"
            + (" / max |kpp|" if k.startswith("kpp") else "") + ": "
            + ", ".join(f"{c['n_off']}: {v:.2e}" for c, v in zip(conv, e))
            + f"; orders {['%.2f' % o for o in orders(e)]}")
    for Fn, et in trims.items():
        out(f"    plant steady trim at Fn {Fn}, |offsets - family| "
            f"(family {fam[Fn]:.4f} deg): " + ", ".join(
                f"{n}: {v:.2e}" for n, v in zip(offs, et))
            + f"; orders {['%.2f' % o for o in orders(et)]}")
    # recorded beside the registered check (which is not changed): the
    # order fitted over all four counts (least squares of log error on log
    # (n_off - 1)) with its 1-sigma interval -- single-step orders scatter
    # with where the knots fall (the largest curvature error sits in the
    # stem strip, PCHIP's one-sided end derivative); and how many of the
    # 48-strip plant's sample points (strip centres and edges) fall in the
    # offsets interval holding the transom-run kink: the steady trim at Fn
    # 0.8 and 1.4 (bow dry, keel rise dry) sees the offsets only there, so
    # with none inside (107 offsets) the trim equals the family's to
    # round-off -- an alignment artefact, not convergence
    hh = np.log(np.array([n - 1.0 for n in offs]))
    A_ = np.vstack([hh, np.ones_like(hh)]).T
    fits = {}
    for k in ("kp", "kpp_strip", "kpp_centre"):
        y_ = np.log([c[k] for c in conv])
        c_, *_ = np.linalg.lstsq(A_, y_, rcond=None)
        r_ = y_ - A_ @ c_
        cov = (r_ @ r_) / max(len(y_) - 2, 1) * np.linalg.inv(A_.T @ A_)
        fits[k] = (float(-c_[0]), float(np.sqrt(cov[0, 0])))
    out("    fitted orders over all four counts (1 sigma): " + ", ".join(
        f"{k} {o:.2f} +- {e:.2f}" for k, (o, e) in fits.items()))
    xr_ = f.transom_run * f.L
    pts = np.r_[np.linspace(0.0, f.L, 49), (np.arange(48) + 0.5) * f.L / 48]
    kink = {}
    for n_off in offs:
        xo = np.linspace(0.0, f.L, n_off)
        i_ = int(np.searchsorted(xo, xr_) - 1)
        kink[n_off] = int(((pts > xo[i_]) & (pts < xo[i_ + 1])).sum())
    out("    plant sample points in the offsets interval holding the "
        "transom-run kink: " + ", ".join(f"{n}: {v}" for n, v in
                                         kink.items())
        + " (0: the trim cannot see the offsets -- its last value is an "
        "artefact)")
    keys = ("yc_max", "yc_l1", "rise", "kp", "kpp_strip")
    mono = all(conv[i + 1][k] < conv[i][k] for i in range(len(conv) - 1)
               for k in keys)
    o_kpp = orders([c["kpp_strip"] for c in conv])
    no_align = all(v == 0 for (n_, ns), v in aligned.items() if ns == 48)
    check("geom", "offsets tables sampled from planing_b converge to the "
          "family (chine on a fine grid; keel, slope and the strip-integral "
          "curvature at 50 strip centres, every refinement; the "
          "curvature's last order >= 1.5; no centre of the 48-strip plant "
          "on a knot; the plant's steady trim at Fn 0.8 and 1.4 falls, "
          "last < 0.05 deg)",
          mono and o_kpp[-1] >= 1.5 and no_align
          and all(et[-1] < et[0] and et[-1] < 0.05 for et in trims.values()),
          f"chine max {conv[0]['yc_max']:.1e} -> {conv[-1]['yc_max']:.1e}, "
          f"L1 {conv[0]['yc_l1']:.1e} -> {conv[-1]['yc_l1']:.1e}; kpp "
          f"(strip) {conv[0]['kpp_strip']:.1e} -> {conv[-1]['kpp_strip']:.1e}"
          f" (orders {['%.2f' % o for o in o_kpp]}; PCHIP centre "
          f"{conv[0]['kpp_centre']:.1e} -> {conv[-1]['kpp_centre']:.1e}); "
          f"trim " + ", ".join(f"Fn {Fn}: {et[0]:.1e} -> {et[-1]:.1e} deg"
                               for Fn, et in trims.items()))
    # the family's own curvature under switch kpp_strip: a v0-family hull
    # with stem rise (v0's parabola: kpp jumps at x0) -- the strip integral
    # is exact where the centre value is off by the jump in one strip
    from hydro.planing import PlaningHull
    hk = PlaningHull(name="p0_kpp", L=5.4, b=2.0, beta_deg=20.0, mass=1400.0,
                     lcg=2.18, vcg=0.55, depth=1.1, taper=0.35,
                     stem_rise=0.3, n_stations=48)
    gk = HG.stations_of(hk)
    # the exact strip mean of v0's parabolic keel curvature, kpp = 2 sr /
    # lt^2 ahead of x0 and 0 aft of it: c x (length of the strip ahead of
    # x0) / dx -- analytic, independent of kp
    x0k = (1.0 - hk.taper) * hk.L
    ck = 2.0 * hk.stem_rise / (hk.L - x0k) ** 2
    exact = ck * np.clip(gk.edges[1:] - np.maximum(gk.edges[:-1], x0k),
                         0.0, gk.dx) / gk.dx
    e_strip = float(np.abs(gk.kpp_strip - exact).max())
    e_centre = float(np.abs(gk.kpp - exact).max())
    pk0 = PlaningVesselASLin(physics.variant(hk, "as1"), calm(), dt=0.02,
                             test_switches=("dDt_full", "chine_exact_all"))
    pk1 = PlaningVesselASLin(physics.variant(hk, "as1"), calm(), dt=0.02,
                             test_switches=("dDt_full", "chine_exact_all",
                                            "kpp_strip"))
    check("geom", "switch kpp_strip on a v0-family hull with stem rise: the "
          "plant's curvature is the strip mean of kpp (analytic, 1e-12 "
          "relative) where v0's centre value is off at the strip holding "
          "the jump; off, the plant keeps v0's array",
          e_strip < 1e-12 * np.abs(exact).max() and e_centre > 0.1
          * np.abs(exact).max() and np.array_equal(pk1.kpp, gk.kpp_strip)
          and np.array_equal(pk0.kpp, gk.kpp),
          f"max |error| / max kpp: strip {e_strip / np.abs(exact).max():.1e}"
          f", centre {e_centre / np.abs(exact).max():.2f}")
    # the GZ plant's heeled-sinkage bracket scaled to the hull
    # (PlaningVesselAS.gz_geometric): scarab195 x 8 (L 43 m) has sections
    # reaching beyond the frozen +-3 m, so the scaled bisection runs; its
    # GZ table / lam must equal scarab195's (Froude similarity), where the
    # frozen method's +-3 m bracket gives a wrong curve
    from sim.planing_vessel_gz import PHI_TAB
    tb_ = physics.variant(VS.froude_twin(hulls.get("scarab195"), 8.0,
                                         name="p0_gzbig8"), "as1")
    gb_ = PlaningVesselAS(tb_, calm(), dt=tb_.scales()["dt"])
    ga_ = PlaningVesselAS(hulls.get("scarab195@as1"), calm(), dt=0.02)
    py_, pz_, zw_ = gb_._sections()
    S_ = float(np.sqrt(py_ ** 2 + pz_ ** 2).max() + np.abs(zw_).max())
    ref_ = np.abs(ga_.gz_tab).max()
    rel_ = float(np.abs(gb_.gz_tab / 8.0 - ga_.gz_tab).max() / ref_)
    fz_ = PlaningVesselGZ.gz_geometric(gb_, PHI_TAB)[0] / 8.0
    ok_ = ga_.gz_geometric(PHI_TAB)[0]
    rel_frozen = float(np.abs(fz_ - ok_).max() / ref_)
    check("geom", "GZ sinkage bracket scaled to the hull: scarab195 x 8 "
          "(sections beyond the frozen +-3 m) gives scarab195's GZ table / "
          "lam within 1e-9, where the frozen bracket does not; scarab195 "
          "itself runs the frozen method (bit-identity: the goldens)",
          S_ > 3.0 and rel_ < 1e-9 and rel_frozen > 1e-3,
          f"bound r + max|zw| {S_:.1f} m; scaled {rel_:.1e}, frozen "
          f"{rel_frozen:.1e} (relative to max GZ)")
    dump("geom.json", dict(rows=rows, offsets=conv, fitted_orders=fits,
                           kink_interval_points=kink, aligned={
        f"{k[0]}_{k[1]}": v for k, v in aligned.items()}, trims={
        str(k): v for k, v in trims.items()}, trim_family={
        str(k): v for k, v in fam.items()}, seconds=time.time() - t0))
    out(f"  ({time.time() - t0:.0f} s)")
    flush("geom.txt")


# ==================================================================== cap
def ref_substrip(g, kc, xa, xb, xc, di, mi, n_samp=400):
    """Independent reference of _substrip for ONE strip: roots of
    f(x) = kc d+(x) - yc(x) and of d(x) by dense sampling + brentq, then
    capped pieces (d > 0 and kc d >= yc at the midpoint) give
    yc(a)^2 - yc(b)^2, the others the exact integral of d+."""
    from scipy.optimize import brentq

    def d(x):
        return di + mi * (x - xc)

    def f(x):
        return kc * max(d(x), 0.0) - float(g.chine(x))
    xs = np.linspace(xa, xb, n_samp + 1)
    fs = np.array([f(x) for x in xs])
    br = [xa, xb]
    for i in range(n_samp):
        if fs[i] == 0.0:
            br.append(xs[i])
        elif fs[i] * fs[i + 1] < 0.0:
            br.append(brentq(f, xs[i], xs[i + 1], xtol=1e-15, rtol=1e-15))
    if mi != 0.0:
        xz = xc - di / mi
        if xa < xz < xb:
            br.append(xz)
    br = np.unique(np.clip(br, xa, xb))
    ycap = iunc = 0.0
    any_cap = False
    for a, b in zip(br[:-1], br[1:]):
        if b <= a:
            continue
        m = 0.5 * (a + b)
        if d(m) > 0.0 and kc * d(m) >= float(g.chine(m)):
            ycap += float(g.chine(a)) ** 2 - float(g.chine(b)) ** 2
            any_cap = True
        else:
            da, db = max(d(a), 0.0), max(d(b), 0.0)
            iunc += 0.5 * (da + db) * (b - a)       # d+ linear on a piece
    return any_cap, ycap, iunc


def bump_offsets_hull():
    """An offsets hull with a NON-concave chine (a waist and a bulge)."""
    from hydro import vessel_spec as VS
    from hydro.hull_geometry import OffsetsForm
    L = 5.4
    x = np.linspace(0.0, L, 13)
    yc = 0.9 * np.sqrt(np.clip(1.0 - np.clip((x - 0.6 * L) / (0.4 * L), 0,
                                              1) ** 2, 0, 1))
    yc = yc * (1.0 - 0.08 * np.exp(-((x - 0.35 * L) / 0.5) ** 2))
    keel = 0.25 * np.clip((x - 0.7 * L) / (0.3 * L), 0, 1) ** 2
    form = OffsetsForm(x=x, yc=yc, keel=keel, beta_deg=20.0, depth=1.1)
    return VS.ParametricPlaningHull(
        "p0_bump", form, dict(mass=1400.0, lcg=0.38 * 5.74, vcg=0.55),
        jet=dict(d_nozzle=0.085, p_max=300 * 745.7), u_design=25 * KN)


def flux_force(p, u, z, th):
    s = np.zeros(14)
    s[6], s[2], s[4] = u, z, th
    _, aux = p.deriv(s, 0.0)
    return float(aux["mom_full"].sum() * p.dx)


def captive_root(p, u, guess, tol=1e-13):
    """Newton root (z, th) of the captive equations w' = q' = 0 (fsolve
    then Newton on a central-difference Jacobian); (z, th, residual)."""
    from scipy.optimize import fsolve

    def F(v):
        s = np.zeros(14)
        s[6], s[2], s[4] = u, v[0], v[1]
        ds, _ = p.deriv(s, 0.0)
        return np.array([ds[8] / G, ds[10] * p.L / G])
    v = np.array(fsolve(F, guess, xtol=1e-13, full_output=True)[0])
    for _ in range(20):
        r = F(v)
        if np.abs(r).max() < tol:
            break
        J = np.zeros((2, 2))
        for j, e in enumerate((1e-7, 1e-8)):
            dv = np.zeros(2)
            dv[j] = e
            J[:, j] = (F(v + dv) - F(v - dv)) / (2 * e)
        try:
            v = v - np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            break
    return float(v[0]), float(v[1]), float(np.abs(F(v)).max())


def part_cap():
    from hydro import hulls
    from sim import physics
    from sim.planing_anyspeed import PlaningVesselASLin
    from sim.planing_vessel import SP
    t0 = time.time()
    out("CAP: strip-exact chine location from the geometry")
    rng = np.random.default_rng(11)
    sw_sets = (("dDt_full",), ("dDt_full", "chine_exact_all"))
    hulls_ = (("scarab195", hulls.get("scarab195@as1")),
              ("planing_b", hulls.get("planing_b@as1")),
              ("bump (offsets, non-concave)",
               physics.variant(bump_offsets_hull(), "as1")))
    summary = []
    for lab, h in hulls_:
        for sw in sw_sets:
            p = PlaningVesselASLin(h, calm(), dt=0.02, test_switches=sw)
            tg = p._taper()
            kc = tg.kc
            worst, n_strips, n_cap, bad_set = 0.0, 0, 0, 0
            for trial in range(60):
                # a realistic keel-depth line: rest draught, a random rise,
                # trim and wave slope
                z = rng.uniform(-0.25, 0.15)
                th = math.radians(rng.uniform(-6.0, 2.0))
                d = -(p.Z_rest + z + p.zk + SP * p.x * (p.Th_rest + th))
                slope = rng.uniform(-0.15, 0.15) - (p.kp
                                                    + SP * (p.Th_rest + th))
                act, ycap, iunc = p._substrip(d, slope)
                got = {int(i): (yc_, iu) for i, yc_, iu in zip(act, ycap,
                                                               iunc)}
                for j, i in enumerate(tg.idx):
                    cap_ref, yr, ir = ref_substrip(
                        p._geom, kc, tg.xa[j], tg.xb[j], tg.xc[j], d[i],
                        slope[i])
                    n_strips += 1
                    if cap_ref != (int(i) in got):
                        bad_set += 1
                        continue
                    if cap_ref:
                        n_cap += 1
                        yg, ig = got[int(i)]
                        scale = max(abs(yr), float(p._geom.yc.max()) ** 2
                                    * 1e-3)
                        worst = max(worst, abs(yg - yr) / scale,
                                    abs(ig - ir) / max(abs(ir), 1e-3
                                                       * p.dx * 0.3))
            summary.append(dict(hull=lab, switches=sw, strips=n_strips,
                                capped=n_cap, set_mismatch=bad_set,
                                worst_rel=worst, active=int(tg.idx.size)))
            check("cap", f"{lab}, {'+'.join(sw)}: _substrip == independent "
                  f"reference ({tg.idx.size} active strips)",
                  bad_set == 0 and worst < 1e-8 and n_cap > 0,
                  f"{n_strips} strip-states, {n_cap} chine-wetted, set "
                  f"mismatches {bad_set}, worst relative {worst:.1e}")
    # continuity across prismatic chine capping (scarab195, 12 kn trim)
    h1 = hulls.get("scarab195@as1")
    cont = {}
    u = 12.0 * KN
    for sw in sw_sets:
        p = PlaningVesselASLin(h1, calm(), dt=0.02, test_switches=sw)
        z0, th0 = p.running_attitude(u)
        row = {}
        for dz in (1e-4, 1e-5, 1e-6):
            zs = z0 + np.arange(-0.03, 0.03, dz)
            F = np.array([flux_force(p, u, z, th0) for z in zs])
            row[dz] = float(np.abs(np.diff(F)).max())
        cont["+".join(sw)] = row
        out(f"    {'+'.join(sw):<26} max |dF| between heave samples over "
            f"+-3 cm (12 kn trim): " + ", ".join(
                f"dz {k:g}: {v:.3g} N" for k, v in row.items()))
    a = cont["dDt_full+chine_exact_all"]
    b = cont["dDt_full"]
    check("cap", "chine_exact_all: the flux force is continuous across "
          "prismatic chine capping (max jump falls ~ dz); dDt_full alone "
          "keeps a finite jump",
          a[1e-6] < 0.2 * a[1e-5] and a[1e-5] < 0.2 * a[1e-4]
          and b[1e-6] > 10.0 * a[1e-6],
          f"with: {a[1e-4]:.3g} / {a[1e-5]:.3g} / {a[1e-6]:.3g} N; "
          f"without: {b[1e-4]:.3g} / {b[1e-5]:.3g} / {b[1e-6]:.3g} N")
    # station-count convergence of the planing equilibria
    conv = {}
    ns = (48, 96, 192, 384, 768)
    kns = (9.0, 12.0, 16.0, 20.0)
    for sw in sw_sets:
        key = "+".join(sw)
        conv[key] = {}
        guess = {kn: None for kn in kns}
        for n in ns:
            hn = copy.deepcopy(hulls.get("scarab195"))
            hn.n_stations = n
            hn.name = f"scarab195_n{n}"
            ha = physics.variant(hn, "as1", name=hn.name)
            p = PlaningVesselASLin(ha, calm(), dt=0.02, test_switches=sw)
            for kn in kns:
                u = kn * KN
                g0 = guess[kn] or p.running_attitude(u)
                z, th, r = captive_root(p, u, g0)
                guess[kn] = (z, th)
                conv[key].setdefault(kn, []).append(dict(
                    n=n, z=z, trim=float(-math.degrees(p.Th_rest + th)),
                    res=r))
    rows = []
    for key, d in conv.items():
        for kn, lst in d.items():
            tr = [x["trim"] for x in lst]
            zz = [x["z"] for x in lst]
            et = np.abs(np.diff(tr))
            ez = np.abs(np.diff(zz))
            rows.append(dict(sw=key, kn=kn, trim=tr, z=zz,
                             d_trim=et.tolist(), d_z=ez.tolist(),
                             order_trim=order(et), order_z=order(ez),
                             res=max(x["res"] for x in lst)))
            out(f"    {key:<26} {kn:4.0f} kn  trim "
                + " ".join(f"{t:.5f}" for t in tr)
                + "  |d trim| " + " ".join(f"{e:.1e}" for e in et)
                + f"  orders {['%.2f' % o for o in order(et)]}"
                f"  max residual {rows[-1]['res']:.0e}")
    good = [r for r in rows if r["sw"] == "dDt_full+chine_exact_all"]
    base = {r["kn"]: r for r in rows if r["sw"] == "dDt_full"}
    mx_good = max(max(r["d_trim"]) for r in good)
    mx_base = max(max(r["d_trim"]) for r in base.values())
    check("cap", "chine_exact_all: Newton roots of the captive equations at "
          "every station count (residual < 1e-9), and the trim's change "
          "with the station count at least 10x below dDt_full alone (whose "
          "trim jumps where a prismatic strip centre caps)",
          all(r["res"] < 1e-9 for r in good) and mx_base >= 10.0 * mx_good,
          f"largest change 48 ... 768 strips: with {mx_good:.1e} deg, "
          f"without {mx_base:.1e} deg")
    # the strict form, recorded as it comes out: every doubling of the
    # station count changes the trim less than the one before, at every
    # speed (M5 'converges with a measured order'). HISTORY, recorded as the
    # ACCEPTANCE preamble requires: this check's threshold was revised twice
    # AFTER running in the first P0 pass (a 10x-ratio form, then a 'largest
    # change below 1e-3 deg' form, before this strict form was restored);
    # the strict form is kept and its FAIL recorded, not loosened again. It
    # cannot pass at P0, by design of the test: the friction Reynolds
    # length (wetted stations x dx) and the friction lever (mean keel height
    # of the wetted stations) jump with the wetted-station count, an O(dx)
    # error that is bounded but not monotone in n; and dm_a/dd ~ d+ has a
    # kink at the moving wetted-length end, where the midpoint rule's local
    # error O(dx^2) changes sign with where the kink falls in its strip.
    # P1 fixes the first two behind a switch and is judged by the measure
    # pre-registered in learn/meta/ANYSPEED_PLAN_2026-10-05.md (reference
    # n = 3072, error bounds, fitted order with its interval), not by this
    # monotonicity
    strict = {r["kn"]: all(r["d_trim"][i + 1] < r["d_trim"][i]
                           for i in range(len(r["d_trim"]) - 1))
              for r in good}
    out("    (strict check: threshold revised twice after running in the "
        "first P0 pass; kept strict, FAIL recorded; P1's pre-registered "
        "measure is in learn/meta/ANYSPEED_PLAN_2026-10-05.md)")
    check("cap", "chine_exact_all, strict: the trim's change falls with "
          "every doubling of the station count at every speed (measured "
          "orders in cap.txt)", all(strict.values()),
          ", ".join(f"{r['kn']:g} kn: changes " + " ".join(
              f"{e:.1e}" for e in r["d_trim"]) + f" (monotone "
              f"{strict[r['kn']]})" for r in good))
    # what remains: small jumps of the captive residual where the count of
    # wetted stations changes -- the friction Reynolds length lw =
    # (wetted stations) x dx and the friction force's lever zw = mean keel
    # height of the wetted stations (sim/planing_anyspeed.py deriv; both v0
    # discretisations; the second jumps on any hull with keel rise) --
    # located here; P1 replaces both behind a switch
    p = PlaningVesselASLin(h1, calm(), dt=0.02, test_switches=sw_sets[1])
    u = 20.0 * KN
    z0, th0 = p.running_attitude(u)
    zs = z0 + np.arange(-0.02, 0.02, 2e-6)
    Q, NW = [], []
    for z in zs:
        s_ = np.zeros(14)
        s_[6], s_[2], s_[4] = u, z, th0
        ds, aux = p.deriv(s_, 0.0)
        Q.append(ds[10])
        NW.append(int((aux["d"] > 0).sum()))
    dQ = np.abs(np.diff(Q))
    chg = np.nonzero(np.diff(NW))[0]
    rest = np.delete(dQ, chg)
    out(f"    20 kn, heave +-2 cm in 2 um steps: |dq'| median "
        f"{np.median(dQ):.1e}, at the {chg.size} changes of the wetted "
        f"station count {dQ[chg].max():.1e}, elsewhere max {rest.max():.1e} "
        f"rad/s^2 -- the remaining non-smoothness is the wetted-station "
        f"count in the friction Reynolds length and the friction lever (v0 "
        f"discretisations, P1), not the chine")
    # the discrete momentum (switch chine_exact_all's derivation): the flux
    # of a located strip is the rate of change of the STRIP-AVERAGED m_a V
    # (d linear over the strip, exact chine), the inertia terms use m_a at
    # the strip centre. Their difference, sum (m_bar - m_c) dx / sum m_c dx,
    # by station count at rest and at the 12 and 20 kn attitudes (m_bar by
    # 512-point midpoint quadrature per strip on the exact chine)
    mism = {}
    for lab, kn in (("rest", 0.0), ("12 kn", 12.0), ("20 kn", 20.0)):
        mism[lab] = []
        for n in (48, 96, 192, 384):
            hn = copy.deepcopy(hulls.get("scarab195"))
            hn.n_stations = n
            hn.name = f"scarab195_n{n}"
            pn = PlaningVesselASLin(physics.variant(hn, "as1", name=hn.name),
                                    calm(), dt=0.02,
                                    test_switches=sw_sets[1])
            z_, th_ = (0.0, 0.0) if kn == 0.0 else pn.running_attitude(
                kn * KN)
            Th = pn.Th_rest + th_
            d = -(pn.Z_rest + z_ + pn.zk + SP * pn.x * Th)
            slope = -(pn.kp + SP * Th)
            g = pn._geom
            kc = 0.5 * np.pi / pn.tanb
            km = pn.k_a * 0.5 * np.pi * pn.rho
            t = (np.arange(512) + 0.5) / 512
            X = g.edges[:-1, None] + t[None, :] * g.dx
            D = d[:, None] + slope[:, None] * (X - g.xi[:, None])
            mbar = km * (np.minimum(kc * np.maximum(D, 0.0), g.chine(X))
                         ** 2).mean(axis=1)
            mc = pn._section(d)[2]
            mism[lab].append(float(np.abs((mbar - mc).sum())
                                   / mc.sum()))
        out(f"    discrete momentum, |sum (m_bar - m_c)| / sum m_c at "
            f"{lab}: " + ", ".join(f"{n}: {v:.2e}" for n, v in zip(
                (48, 96, 192, 384), mism[lab]))
            + f"; orders {['%.2f' % o for o in order(mism[lab])]}")
    check("cap", "the discrete momentum: the strip-averaged m_a (what the "
          "located flux differentiates) and the centre m_a (what the "
          "inertia uses) differ by a relative amount that falls with "
          "every doubling of the station count (rest, 12, 20 kn)",
          all(all(v[i + 1] < v[i] for i in range(len(v) - 1))
              for v in mism.values()),
          {k: f"{v[0]:.1e} -> {v[-1]:.1e}" for k, v in mism.items()})
    dump("cap.json", dict(substrip=summary, continuity={
        k: {str(a_): b_ for a_, b_ in v.items()} for k, v in cont.items()},
        convergence=rows, momentum_mismatch=mism,
        seconds=time.time() - t0))
    out(f"  ({time.time() - t0:.0f} s)")
    flush("cap.txt")


# ================================================================== cache
LEGACY_FILES = ("studies/_cache/lofi_scarab195.json",
                "studies/_cache/final_eval/turn_gz.json",
                "studies/_cache/anyspeed/golden_v0.npz",
                "studies/_cache/anyspeed/golden_as_dDt_full.npz",
                "studies/_cache/anyspeed/"
                "golden_as_dDt_full+chine_exact_all.npz",
                "studies/_cache/final_eval/meta7/results.pkl",
                "studies/_cache/onestep_gz/streams_T120.pkl")


def _lofi_files():
    d = os.path.join(ROOT, "studies", "_cache")
    return sorted(f for f in os.listdir(d) if f.startswith("lofi_"))


def part_cache():
    from hydro import hulls, vessel_spec as VS
    from sim import physics
    from sim import planing_anyspeed as PA
    from sim.vessel_profile import VesselProfile
    from studies import final_eval as FE
    t0 = time.time()
    out("CACHE: keys by physics + hull sha1 (N3)")
    before = {f: sha1_file(f) for f in LEGACY_FILES}
    lofi0 = _lofi_files()
    h0 = hulls.get("scarab195")
    h1 = hulls.get("scarab195@as1")
    hb = hulls.get("planing_b@as1")
    tw = physics.variant(VS.froude_twin(h0, 1.5), "as1")
    hm = copy.deepcopy(h0)
    hm.n_stations = 96
    hm.name = "scarab195_n96"
    tags = {h.name: physics.cache_tag(h) for h in (h0, h1, hb, tw, hm)}
    out(f"    cache tags: {tags}")
    check("cache", "physics.cache_tag: None (legacy names) for scarab195 on "
          "v0 only; distinct keyed tags for its as1 key, planing_b@as1, a "
          "Froude twin and a modified copy",
          tags["scarab195"] is None and len({v for k, v in tags.items()
                                             if k != "scarab195"}) == 4
          and all(v for k, v in tags.items() if k != "scarab195"), tags)
    paths = {}
    for plant in ("gz", "default"):
        FE.OPT["plant"] = plant
        for h in (h0, h1, hb, tw, hm):
            paths[(plant, h.name)] = os.path.relpath(FE.turn_path(h), ROOT)
    FE.OPT["plant"] = "gz"
    legacy = paths[("gz", "scarab195")].replace("\\", "/")
    check("cache", "final_eval.turn_path: scarab195 v0 keeps "
          "studies/_cache/final_eval/turn_gz.json; every other hull and "
          "plant kind gets its own file",
          legacy == "studies/_cache/final_eval/turn_gz.json"
          and len(set(paths.values())) == len(paths),
          {f"{k[0]}:{k[1]}": v for k, v in paths.items()
           if k[0] == "gz"})
    # profiles: one directory per physics sha1, meta checked
    profs = {h.name: VesselProfile(h) for h in (h0, h1, hb, tw)}
    dirs = {k: os.path.relpath(v.dir, ROOT) for k, v in profs.items()}
    check("cache", "VesselProfile directories: one per hull and physics "
          "sha1, under studies/_cache/profiles",
          len(set(dirs.values())) == len(dirs) and all(
              d.replace("\\", "/").startswith("studies/_cache/profiles/")
              for d in dirs.values()), dirs)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pr = VesselProfile(hb, root=tmp)
        calls = []
        a = pr._cached("probe", lambda: calls.append(1) or dict(v=1.0))
        b = pr._cached("probe", lambda: calls.append(1) or dict(v=2.0))
        rec = json.load(open(pr.path("probe")))
        rec["meta"]["physics_sha1"] = "0" * 40
        json.dump(rec, open(pr.path("probe"), "w"))
        c = pr._cached("probe", lambda: calls.append(1) or dict(v=3.0))
        # the protocol is in the meta: a changed protocol constant recomputes
        import sim.vessel_profile as VP
        t_old = dict(VP.TURN_T)
        try:
            VP.TURN_T["end"] = t_old["end"] + 1.0
            d_ = pr._cached("probe", lambda: calls.append(1) or dict(v=4.0))
        finally:
            VP.TURN_T.clear()
            VP.TURN_T.update(t_old)
        e_ = pr._cached("probe", lambda: calls.append(1) or dict(v=5.0))
        # the call's arguments are in the part's name: steady() at another
        # Froude-number list is its own part, the default one is not served
        # it (the review: a later default call returned the custom row)
        hs = hulls.get("scarab195@as1")
        ps = VesselProfile(hs, root=tmp)
        one = ps.steady(fn_list=(1.1,))
        full = ps.steady()
        one2 = ps.steady(fn_list=(1.1,))
        names = sorted(os.listdir(ps.dir))
    check("cache", "a profile part is read back only under its own meta "
          "(physics sha1, code, hull, switches, protocol: code of the "
          "profile and the identification, IDENT_VERSION, the protocol "
          "constants); a foreign meta or a changed protocol constant "
          "recomputes; steady() at another Froude-number list is a part of "
          "its own",
          a == {"v": 1.0} and b == {"v": 1.0} and c == {"v": 3.0}
          and d_ == {"v": 4.0} and e_ == {"v": 5.0} and len(calls) == 4
          and len(one) == 1 and len(full) == len(VP.FN_STEADY)
          and one2 == one and full[2] == one[0]
          and [r["Fn"] for r in full] == list(VP.FN_STEADY),
          f"{a} {b} {c} {d_} {e_}, computed {len(calls)}x; steady parts "
          f"{names}, rows {len(one)} / {len(full)}")
    # steady-state cache: two hulls of the same name, different geometry,
    # the first freed before the second is built (an id may be reused)
    from sim.planing_anyspeed import PlaningVesselASLin
    u = 9.0 * KN
    hA = physics.variant(VS.parametric_of(h0, name="p0_alias"), "as1")
    rA = PlaningVesselASLin(hA, calm(), dt=0.02).running_attitude(u)
    idA = id(hA)
    del hA
    gc.collect()
    f, mp, js, cl, nm = VS.parametric_of(h0, name="p0_alias").specs()
    f.transom_beam, f.transom_run = 0.85, 0.2
    hB = physics.variant(VS.ParametricPlaningHull(
        "p0_alias", f, mp, jet=js, closures=cl, numerics=nm,
        u_design=h0.u_design), "as1")
    rB = PlaningVesselASLin(hB, calm(), dt=0.02).running_attitude(u)
    PA._STEADY.clear()
    rB2 = PlaningVesselASLin(hB, calm(), dt=0.02).running_attitude(u)
    check("cache", "the as steady-state cache (keyed by the physics stamp) "
          "does not alias two hulls of one name, the first freed (the id "
          "kept: _PINNED)", rB == rB2 and rB != rA and idA in PA._PINNED,
          f"A {rA}, B {rB}, B fresh {rB2}; id(A) pinned "
          f"{idA in PA._PINNED}, id(B) == id(A): {id(hB) == idA}")
    # ... and keyed by the exact speed: two speeds 0.3 mm/s apart are two
    # entries, each equal to its own fresh captive run (v0 rounds to 1 mm/s
    # and serves the first one's attitude to both)
    pq = PlaningVesselASLin(h1, calm(), dt=0.02)
    ra, rb = pq.running_attitude(9.0004), pq.running_attitude(9.0001)
    PA._STEADY.clear()
    rb_fresh = PlaningVesselASLin(h1, calm(), dt=0.02).running_attitude(
        9.0001)
    check("cache", "the as steady-state cache is keyed by the exact speed: "
          "9.0001 m/s after 9.0004 m/s gets its own attitude, equal to a "
          "fresh run", ra != rb and rb == rb_fresh,
          f"9.0004: {ra}, 9.0001: {rb}, fresh {rb_fresh}")
    # actuators passed in are carried by with_sea and gz_twin
    from sim.waterjet import Waterjet, Nozzle
    ref = h1.plant(calm())
    pump = copy.deepcopy(ref.prop.pump)
    jet = Waterjet(pump, ref.prop.x_nozzle, ref.prop.z_nozzle,
                   ref.prop.x_intake, ref.prop.z_intake, t_max=ref.prop.t_max,
                   t_min=ref.prop.t_min, tau=0.9, rate_max=ref.prop.rate_max,
                   dead_band=ref.prop.dead_band, delay=0.2,
                   u_ref=ref.prop.u_ref, dt=0.02)
    noz = Nozzle(max_deg=20.0, dt=0.02, x_nozzle=ref.prop.x_nozzle,
                 z_nozzle=ref.prop.z_nozzle)
    pc = h1.plant(calm(), prop=jet, rudder=noz)
    ws, gt = pc.with_sea(calm()), physics.gz_twin(pc)
    ok = all(x.prop.tau == 0.9 and x.rudder.max == math.radians(20.0)
             and x.prop is not jet and x.rudder is not noz
             for x in (ws, gt))
    try:
        pc.with_sea(calm(), dt=0.01)
        refused = False
    except ValueError:
        refused = True
    p0c = h0.plant(calm(), prop=copy.deepcopy(jet),
                   rudder=copy.deepcopy(noz))
    g0c = physics.gz_twin(p0c)
    check("cache", "actuators given to an as plant are carried (fresh "
          "copies) by with_sea and physics.gz_twin, a rebuild at another "
          "step refused; for a v0 plant physics.gz_twin is the frozen "
          "from_plant and drops them, as v0 always did",
          ok and refused and g0c.prop.tau == 0.6
          and g0c.rudder.max == math.radians(25.0),
          f"as with_sea/gz_twin {ok}, dt refusal {refused}, v0 gz_twin "
          f"tau {g0c.prop.tau}")
    after = {f: sha1_file(f) for f in LEGACY_FILES}
    check("cache", "legacy cache files unchanged; no lofi_*.json created",
          after == before and _lofi_files() == lofi0,
          {k.split("_cache/")[-1]: v[:10] for k, v in after.items()})
    dump("cache.json", dict(tags=tags, turn_paths={
        f"{k[0]}:{k[1]}": v for k, v in paths.items()}, profiles=dirs,
        legacy=after, seconds=time.time() - t0))
    out(f"  ({time.time() - t0:.0f} s)")
    flush("cache.txt")


# =================================================================== mppi
class ScaledSea:
    """A sea Froude-scaled by lam: eta(x, y, t) = lam eta0(x/lam, y/lam,
    t/sqrt(lam))."""

    def __init__(self, base, lam):
        self.base, self.lam, self.s = base, lam, math.sqrt(lam)
        self.hs = base.hs * lam

    def eta(self, x, y, t):
        return self.lam * self.base.eta(np.asarray(x) / self.lam,
                                        np.asarray(y) / self.lam,
                                        t / self.s)


def part_mppi():
    from control.mpc import MPPIController, PreviewProvider, froude_time_ref
    from control.reduced import ReducedModel
    from sim.wavefield import SeaState
    from studies.principles_hull_swap import EXP
    t0 = time.time()
    out("MPPI: the time-scale-free objective option (M7)")
    pb = json.load(open(os.path.join(ROOT, "studies", "_cache",
                                     "lofi_scarab195.json")))["params"]
    lam = 1.5
    s = math.sqrt(lam)
    missing = sorted(k for k in pb if k not in EXP)
    pt = {k: (v * lam ** EXP[k] if k in EXP else v) for k, v in pb.items()}
    sea0 = SeaState(1.0, 5.0, n_freq=16, n_dir=3, seed=2)
    seas = (sea0, ScaledSea(sea0, lam))
    rng = np.random.default_rng(5)
    # [x, y, u, z, zdot, th, thdot, psi, r, v] (control/reduced.step)
    s0 = np.array([3.0, 1.0, 9.0, 0.02, 0.1, -0.02, 0.01, 0.05, 0.02, 0.1])
    sc = np.array([lam, lam, s, lam, s, 1.0, 1.0 / s, 1.0, 1.0 / s, s])
    res = {}
    for lab, tref in (("default", (None, None)),
                      ("time_ref", (froude_time_ref(pb["L"]),
                                    froude_time_ref(pt["L"])))):
        costs = []
        for k, (p, sea, st, tr) in enumerate(((pb, seas[0], s0, tref[0]),
                                              (pt, seas[1], s0 * sc,
                                               tref[1]))):
            c = MPPIController(ReducedModel(dict(p)), PreviewProvider(
                sea, t_preview=6.0 * (s if k else 1.0)), seed=0,
                time_ref=tr)
            r = np.random.default_rng(5)
            noise = r.normal(0.0, 1.0, (64, 2, c.n_knots)) * 0.25
            cand = np.clip(c.nominal[None] + noise, -1.0, 1.0)
            cand[:, 0] = np.clip(cand[:, 0], 0.0, 1.0)
            seq = c._expand(cand)
            costs.append(c.rollout_cost(st, seq, 10.0 * (s if k else 1.0),
                                        0.05))
        rel = float(np.max(np.abs(costs[1] - costs[0])
                           / np.maximum(np.abs(costs[0]), 1e-12)))
        res[lab] = dict(max_rel=rel, mean_ratio=float(np.mean(
            costs[1] / costs[0])))
        out(f"    {lab:<9} twin / base cost: max relative difference "
            f"{rel:.2e}, mean ratio {res[lab]['mean_ratio']:.6f}")
    check("mppi", "with time_ref = froude_time_ref(L) a Froude twin's model "
          "in a Froude-scaled sea costs every plan as the base does (1e-9); "
          "the default (seconds) does not",
          res["time_ref"]["max_rel"] < 1e-9
          and res["default"]["max_rel"] > 1e-3 and not missing,
          f"{res}; coefficients without exponent {missing}")
    # the reference hull: froude_time_ref(5.4) == 1.0 and the cost is v0's
    c1 = MPPIController(ReducedModel(dict(pb)), PreviewProvider(sea0, 6.0),
                        seed=0)
    c2 = MPPIController(ReducedModel(dict(pb)), PreviewProvider(sea0, 6.0),
                        seed=0, time_ref=froude_time_ref(pb["L"]))
    seq = c1._expand(np.clip(c1.nominal[None] + rng.normal(
        0, 0.25, (64, 2, c1.n_knots)), 0.0, 1.0))
    a, b = c1.rollout_cost(s0, seq, 10.0, 0.05), c2.rollout_cost(
        s0, seq, 10.0, 0.05)
    cmd1 = [c1(s0, 10.0 + 0.25 * i) for i in range(3)]
    cmd2 = [c2(s0, 10.0 + 0.25 * i) for i in range(3)]
    check("mppi", "on scarab195 (L 5.4 m) froude_time_ref == 1.0 and the "
          "option's cost and commands equal the default's bit for bit",
          froude_time_ref(5.4) == 1.0 and np.array_equal(a, b)
          and all(np.array_equal(x, y) for x, y in zip(cmd1, cmd2)),
          f"froude_time_ref(5.4) = {froude_time_ref(5.4)!r}")
    dump("mppi.json", dict(res=res, seconds=time.time() - t0))
    out(f"  ({time.time() - t0:.0f} s)")
    flush("mppi.txt")


# =============================================================== steptime
V0_IDLE_MS = 1.67      # v0's step on an idle machine (the P0 review)
E1_MS = 3.0            # ACCEPTANCE E1


def part_steptime():
    """E1: cost of one 0.02 s step of the GZ plant in the task sea (mission
    s0 leg 0), single process, threads 1. Rounds of N steps per plant,
    interleaved (the order rotates each round); each round's as1 / v0 ratio
    cancels the machine's load at that moment; the median ratio times v0's
    idle cost is the estimate checked against 3 ms (registered before the
    run; raw medians reported beside it)."""
    from hydro import hulls
    from learn.repro import task
    from sim import physics
    t0 = time.time()
    out("STEPTIME: E1, 3 ms per 0.02 s step, single core, task sea")
    m = task.Mission("high", 0, 0, t_end=1.0, track=task.LEGS[0])
    sea = m.plant.sea
    keys = ("scarab195", "scarab195@as1", "planing_b@as1")
    plants = {k: physics.gz_twin(hulls.get(k).plant(sea, dt=0.02))
              for k in keys}
    N, rounds = 300, 9
    per = {k: [] for k in keys}
    for r in range(rounds):
        order_ = keys[r % 3:] + keys[:r % 3]
        for k in order_:
            p = plants[k]
            s_ = p.initial_state(0.9 * float(p.hull.u_design))
            tm, rm = p.prop.t_max, p.rudder.max
            ta = time.perf_counter()
            for i in range(N):
                s_ = p.step(s_, i * 0.02, tm * (0.6 + 0.3 * math.sin(
                    0.05 * i)), rm * 0.5 * math.sin(0.03 * i), 0.02)
            per[k].append(1e3 * (time.perf_counter() - ta) / N)
    med = {k: float(np.median(v)) for k, v in per.items()}
    rat = {k: float(np.median(np.array(per[k]) / np.array(per["scarab195"])))
           for k in keys[1:]}
    est = {k: rat[k] * V0_IDLE_MS for k in rat}
    for k in keys:
        out(f"    {k:<15} ms per step: median {med[k]:.3f}, min "
            f"{min(per[k]):.3f}, max {max(per[k]):.3f}"
            + (f"; / v0 {rat[k]:.3f} -> idle estimate {est[k]:.2f} ms"
               if k in rat else ""))
    check("steptime", f"E1: as1 GZ plants in the task sea at most {E1_MS} ms "
          f"per 0.02 s step (load-cancelled: median as1 / v0 ratio of "
          f"interleaved rounds x v0's idle {V0_IDLE_MS} ms)",
          all(v <= E1_MS for v in est.values()),
          ", ".join(f"{k} {v:.2f} ms" for k, v in est.items())
          + f"; raw medians {', '.join(f'{k} {v:.2f}' for k, v in med.items())}")
    dump("steptime.json", dict(per_round_ms=per, median_ms=med,
                               ratio_to_v0=rat, idle_estimate_ms=est,
                               seconds=time.time() - t0))
    out(f"  ({time.time() - t0:.0f} s)")
    flush("steptime.txt")


# =================================================================== twin
FN_RUN = 1.2
TWIN_EXACT = 1e-10     # internal bound: the twin runs the same arithmetic


def scaled_sea_arrays(sea, lam):
    from types import SimpleNamespace
    s = math.sqrt(lam)
    return SimpleNamespace(a=np.asarray(sea.a) * lam, w=np.asarray(sea.w) / s,
                           k=np.asarray(sea.k) / lam, th=np.asarray(sea.th),
                           phi=np.asarray(sea.phi), hs=sea.hs * lam,
                           tp=sea.tp * s, theta0=getattr(sea, "theta0", None))


def open_loop_nd(h, lam):
    """600 steps of the GZ plant in a Froude-scaled sea, commands in Froude
    time; the nondimensional state and measurements."""
    from sim import physics
    from sim.wavefield import SeaState
    base_sea = SeaState(0.6, 4.0, n_freq=20, n_dir=5, seed=4)
    sea = scaled_sea_arrays(base_sea, lam)
    p = physics.gz_twin(h.plant(sea), record=False)
    L, m = float(p.L), float(p.m)
    U, T = math.sqrt(G * L), math.sqrt(L / G)
    s = p.initial_state(FN_RUN * U)
    rows, meas = [], []
    for i in range(600):
        tau = i * p.dt / T
        thr = p.prop.t_max * (0.55 + 0.35 * math.sin(0.3 * tau))
        rud = p.rudder.max * 0.6 * math.sin(0.2 * tau)
        s = p.step(s, i * p.dt, thr, rud, p.dt)
        rows.append(np.r_[s[:3] / L, s[3:6], s[6:9] / U, s[9:12] * T,
                          s[12] / (m * G), s[13]])
        meas.append([p.last_cg_acc / G, p.last_bow_acc / G,
                     p.last_rel_bow / L, p.last_slam_force / (m * G),
                     p.last_slam_force_full / (m * G), p.prop.prime,
                     p.last_n_sub])
    return dict(state=np.array(rows), meas=np.array(meas))


def twin_job(arg):
    """One hull of the twin test (in its own process)."""
    warnings.filterwarnings("ignore")
    from hydro import hulls, vessel_spec as VS
    from sim import physics
    from sim.vessel_profile import VesselProfile
    label, lam = arg
    if label == "base":
        h = hulls.get("scarab195@as1")
    else:
        h = physics.variant(VS.froude_twin(hulls.get("scarab195"), lam,
                                           name=f"scarab195_p0twin{lam:g}"),
                            "as1")
    t0 = time.time()
    prof = VesselProfile(h)
    res = dict(label=label, lam=lam, hull=h.name,
               profile=prof.build(), profile_dir=prof.dir)
    res["open_loop"] = {k: v.tolist() for k, v in
                        open_loop_nd(h, lam).items()}
    res["seconds"] = time.time() - t0
    return res


def nd_limits(lim):
    L, m = lim["L"], lim["mass"]
    U, T = math.sqrt(G * L), math.sqrt(L / G)
    return dict(T=lim["T"] / L, GM=lim["GM"] / L, freeboard=lim["freeboard"]
                / L, draft_bow=lim["draft_bow"] / L, t_max=lim["t_max"]
                / (m * G), t_min=lim["t_min"] / (m * G),
                rud_max=lim["rud_max"], u_design=lim["u_design"] / U,
                top_speed=lim["top_speed"] / U, gz_max=lim["gz_max"] / L,
                phi_vanish=lim["phi_vanish_deg"], phi_fail=lim["phi_fail_deg"],
                T_roll=lim["T_roll"] / T, dt=lim["dt"] / T,
                B=lim["B"] / L)


def part_twin():
    from studies.parallel_runner import run_parallel
    from studies.principles_hull_swap import EXP
    t0 = time.time()
    out("TWIN: Froude similarity on as1 (M3, 1e-4)")
    jobs = [("base", 1.0), ("lam1", 1.0), ("lam1.5", 1.5), ("lam0.6", 0.6)]
    res = run_parallel(twin_job, jobs, gb_per_job=0.4, reserve_gb=3.0)
    bad = [r for r in res if isinstance(r, Exception)]
    if bad:
        for r in bad:
            out(f"  job failed: {r!r}")
        check("twin", "every twin job ran", False, bad)
        flush("twin.txt")
        return
    R = {r["label"]: r for r in res}
    base = R["base"]
    lb = nd_limits(base["profile"]["limits"])
    summary = {}
    for lab in ("lam1", "lam1.5", "lam0.6"):
        r = R[lab]
        lam = r["lam"]
        worst = {}
        lt = nd_limits(r["profile"]["limits"])
        worst["limits"] = max(abs(lt[k] - lb[k]) / max(abs(lb[k]), 1e-12)
                              for k in lb)
        sb, st = base["profile"]["steady"], r["profile"]["steady"]
        ws = 0.0
        for a, b in zip(sb, st):
            La, Lb = base["profile"]["limits"]["L"], r["profile"]["limits"][
                "L"]
            ma, mb = base["profile"]["limits"]["mass"], r["profile"][
                "limits"]["mass"]
            ws = max(ws, abs(b["trim_deg"] - a["trim_deg"]) / max(
                abs(a["trim_deg"]), 1.0),
                abs(b["z"] / Lb - a["z"] / La) / max(abs(a["z"] / La), 1e-3),
                abs(b["R"] / (mb * G) - a["R"] / (ma * G))
                / (a["R"] / (ma * G)),
                abs(b["lw"] / Lb - a["lw"] / La) / max(a["lw"] / La, 1e-3))
        worst["steady"] = ws
        pa, pt = base["profile"]["reduced_model"]["p"], r["profile"][
            "reduced_model"]["p"]
        rel = {k: pt[k] / (pa[k] * lam ** EXP[k]) - 1.0 for k in pa
               if k in EXP and pa[k] != 0.0}
        worst["ident"] = max(abs(v) for v in rel.values())
        worst["ident_key"] = max(rel, key=lambda k: abs(rel[k]))
        ta, tt = base["profile"]["turn_table"], r["profile"]["turn_table"]
        La, Lt = base["profile"]["limits"]["L"], r["profile"]["limits"]["L"]
        wt = 0.0
        for ua, ra, ut, rt in zip(ta["u"], ta["r"], tt["u"], tt["r"]):
            wt = max(wt, abs(rt * math.sqrt(Lt / G) / (ra * math.sqrt(La / G))
                             - 1.0), abs(ut / math.sqrt(G * Lt)
                                         / (ua / math.sqrt(G * La)) - 1.0))
        worst["turn"] = wt if len(ta["u"]) == len(tt["u"]) else math.inf
        Sa, St = np.array(base["open_loop"]["state"]), np.array(
            r["open_loop"]["state"])
        Ma, Mt = np.array(base["open_loop"]["meas"]), np.array(
            r["open_loop"]["meas"])
        worst["open_loop_state"] = float(np.abs(St - Sa).max())
        worst["open_loop_meas"] = float(np.abs(Mt - Ma).max())
        worst["identical"] = bool(np.array_equal(Sa, St)
                                  and r["profile"]["steady"]
                                  == base["profile"]["steady"])
        summary[lab] = worst
        out(f"    {lab:<7} worst nondimensional differences: " + ", ".join(
            f"{k} {v:.2e}" if isinstance(v, float) else f"{k} {v}"
            for k, v in worst.items()))
    w1 = summary["lam1"]
    check("twin", "lam 1 (parametric representation of scarab195): every "
          "output identical to scarab195@as1",
          w1["identical"] and w1["limits"] == 0.0 and w1["ident"] == 0.0
          and w1["turn"] == 0.0, w1)
    for lab in ("lam1.5", "lam0.6"):
        w = summary[lab]
        worst_ = max(w[k] for k in ("limits", "steady", "ident", "turn",
                                    "open_loop_state", "open_loop_meas"))
        check("twin", f"{lab}: internal bound {TWIN_EXACT:g} on every "
              f"nondimensional output (limits, steady curve, reduced model, "
              f"turning table, open-loop run): the scaled twin runs the same "
              f"arithmetic, so only round-off may remain",
              worst_ < TWIN_EXACT, f"worst {worst_:.2e}")
    for lab in ("lam1.5", "lam0.6"):
        w = summary[lab]
        check("twin", f"{lab}: limits, steady curve, Froude-exact turning "
              f"table and the 600-step open-loop run in a scaled sea agree "
              f"nondimensionally within 1e-4",
              max(w["limits"], w["steady"], w["turn"],
                  w["open_loop_state"], w["open_loop_meas"]) < 1e-4, w)
        check("twin", f"{lab}: identified reduced model, every coefficient "
              f"/ lam^exponent within 1e-4", w["ident"] < 1e-4,
              f"worst {w['ident_key']}: {w['ident']:.2e}")
    dump("twin.json", dict(summary=summary, runs={
        k: {kk: vv for kk, vv in v.items() if kk != "open_loop"}
        for k, v in R.items()}, seconds=time.time() - t0))
    out(f"  ({time.time() - t0:.0f} s)")
    flush("twin.txt")


# ================================================================== hullb
def _scaled_defaults_ok(h):
    """Every input a parametric hull took from defaults equals the
    reference value times (L / L_REF)^exponent (KEEP: as is), and z_air is
    the geometry's."""
    from hydro import vessel_spec as VS
    lam = h.L / VS.L_REF
    ok = "defaulted" in vars(h)
    for cls, have, grp in ((VS.JetSpec, h.jet, "jet"),
                           (VS.PlantNumerics, h.numerics, "numerics")):
        d0 = cls.defaults()
        for k in h.defaulted.get(grp, []):
            e = 0 if k in cls.KEEP else cls.EXP.get(k, 0)
            want = d0[k] * lam ** e if d0[k] is not None else None
            ok &= have[k] == want
    f, mp, *_ = h.specs()
    ok &= h.z_air == VS.air_point_of_form(f, mp.vcg)
    return bool(ok)


def part_hullb():
    from hydro import hulls
    from hydro.planing import savitsky
    from sim import physics
    from sim.vessel_profile import VesselProfile
    from sim.wavefield import SeaState
    t0 = time.time()
    out("HULLB: the second parametric hull end to end (planing_b@as1)")
    hb0 = hulls.get("planing_b")
    try:
        hb0.plant(calm())
        refused = False
    except ValueError as e:
        refused = str(e)
    check("hullb", "planing_b refused by the v0 dispatch (geometry outside "
          "v0's family)", refused, refused)
    h = hulls.get("planing_b@as1")
    out("    inputs taken from defaults (reference values Froude-scaled to "
        f"L {h.L} m, z_air from the geometry: {h.z_air:.4f} m): "
        + json.dumps(h.defaulted))
    check("hullb", "planing_b's defaulted inputs are recorded in its "
          "manifest and none is a reference absolute: jet time constants "
          "and positions, plant tolerances and floors scale with "
          "(L / L_REF)^exponent; nu, rho_air, dt, dt_ctrl kept",
          _scaled_defaults_ok(h), h.defaulted)
    prof = VesselProfile(h)
    tb = time.time()
    P = prof.build()
    out(f"    profile built / read in {time.time() - tb:.0f} s: "
        f"{os.path.relpath(prof.dir, ROOT)}")
    lim = {k: v for k, v in P["limits"].items() if k != "outside_validity"}
    out("    limits: " + ", ".join(f"{k} {v:.4g}" for k, v in lim.items()))
    out(f"    outside the family's range of validity: "
        f"{P['limits'].get('outside_validity')}")
    out("    steady (Fn, trim deg, rise m, R N, captive residuals; "
        "Savitsky trim / R where valid):")
    st_ok = True
    for r in P["steady"]:
        sv = savitsky(r["u"], h.mass, h.lcg, h.b, h.beta_deg) \
            if r["Fn"] > 1.0 else None
        out(f"      Fn {r['Fn']:.3f} ({r['u'] / KN:5.1f} kn): trim "
            f"{r['trim_deg']:6.3f}, rise {r['z']:+.4f}, R {r['R']:7.1f}, "
            f"res {r['res_w']:.1e} {r['res_q']:.1e}"
            + (f"; Savitsky {sv['tau_deg']:.3f} deg, {sv['R']:.0f} N"
               f"{'' if sv['valid'] else ' (outside its range)'}"
               if sv else ""))
        st_ok &= r["res_w"] < 1e-4 and r["res_q"] < 1e-4
    pm = P["reduced_model"]["p"]
    tt = P["turn_table"]
    out(f"    reduced model: {len(pm)} coefficients, identified in "
        f"{P['reduced_model']['seconds']:.0f} s; tau_u {pm['tau_u']:.3f}, "
        f"k_drag {pm['k_drag']:.2f}, wn_heave {pm['wn_heave']:.3f}, "
        f"wn_pitch {pm['wn_pitch']:.3f}, k_nomoto {pm['k_nomoto']:.4f}, "
        f"u_ss {pm['u_ss']:.2f}")
    out("    turning table: " + ", ".join(
        f"{u / KN:.1f} kn -> {math.degrees(r):.1f} deg/s (std "
        f"{math.degrees(sd):.2f})" for u, r, sd in zip(tt["u"], tt["r"],
                                                       tt["r_std"])))
    fin = all(np.isfinite(v) for v in lim.values()) and all(
        np.isfinite(v) for v in pm.values())
    check("hullb", "planing_b@as1: limits and the identified reduced model "
          "finite, a top speed above the design speed, GZ vanishing beyond "
          "the 40 deg failure angle",
          fin and lim["top_speed"] > lim["u_design"]
          and lim["phi_vanish_deg"] > 40.0, f"top {lim['top_speed']:.2f} "
          f"m/s, phi_vanish {lim['phi_vanish_deg']:.1f} deg")
    check("hullb", "steady curve at six Froude numbers: the captive runs end "
          "on an equilibrium (residuals < 1e-4)", st_ok,
          [(r["Fn"], r["res_w"], r["res_q"]) for r in P["steady"]])
    check("hullb", "turning table: all six throttles finite, yaw rate "
          "positive", len(tt["u"]) == 6 and all(r > 0 for r in tt["r"]),
          f"{len(tt['u'])} rows")
    # cached under its sha1: a second profile object reads, does not compute
    t1 = time.time()
    P2 = VesselProfile("planing_b@as1").build()
    check("hullb", "the profile is read back from its sha1 directory "
          "(identical, no recomputation)", P2 == P and time.time() - t1 < 10,
          f"{time.time() - t1:.1f} s")
    # an open-loop run in a sea
    sea = SeaState(0.8, 5.0, n_freq=20, n_dir=5, seed=1)
    p = physics.gz_twin(h.plant(sea), record=True)
    s = p.initial_state(12.0)
    for i in range(3000):
        s = p.step(s, i * p.dt, 0.7 * p.prop.t_max,
                   0.5 * p.rudder.max * math.sin(0.01 * i), p.dt)
    tr = p.trajectory()
    check("hullb", "60 s open loop in Hs 0.8 m: finite, no capsize, stamped "
          "with its physics", np.all(np.isfinite(s)) and not p.capsized
          and tr["meta"]["hull"] == "planing_b@as1",
          f"final speed {s[6]:.2f} m/s, max roll "
          f"{math.degrees(np.abs(tr['s'][:, 3]).max()):.1f} deg")
    dump("hullb.json", dict(profile=P, profile_dir=prof.dir,
                            defaulted=h.defaulted, z_air=h.z_air,
                            seconds=time.time() - t0))
    out(f"  ({time.time() - t0:.0f} s)")
    flush("hullb.txt")


# =============================================================== hullb_s1
HB_CENSUS_KN = (2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0, 26.0, 32.0, 40.0)
HB_DECAY_KN = (3.0, 6.0, 9.0, 12.0)
HB_D55_KN = (2.0, 4.0, 6.0)
HB_SWEEP_UP = tuple([float(k) for k in range(1, 21)]
                    + [float(k) for k in range(22, 41, 2)])


def hullb_plant(captive_u=0.0, dt=0.02):
    from hydro import hulls
    from sim.planing_anyspeed import PlaningVesselAS
    return PlaningVesselAS(hulls.get("planing_b@as1"), calm(), dt=dt,
                           captive_u=captive_u)


def hullb_s1_job(which):
    """One group of S1's standing checks on planing_b@as1 (own process)."""
    warnings.filterwarnings("ignore")
    from studies import anyspeed_s1_check as S1
    t0 = time.time()
    p = hullb_plant()
    res = dict(which=which)
    if which == "census":
        rows = []
        for kn in HB_CENSUS_KN:
            u = kn * KN
            roots, log, ne = S1.census(p, u)
            cells = S1.winding(p, u)
            free = [(zc, tc, dg) for zc, tc, dg in cells if not any(
                abs(r["z"] - zc) <= 0.0151 and abs(r["trim"] - tc) <= 0.376
                for r in roots)]
            rows.append(dict(kn=kn, roots=[{k: v for k, v in r.items()
                                            if k != "lam"} for r in roots],
                             n_stable=sum(r["label"] == "stable"
                                          for r in roots),
                             free_zeros=free, passes=log))
        res["rows"] = rows
    elif which == "sweep":
        up, s_end = S1.step_sweep(p, HB_SWEEP_UP)
        dn, _ = S1.step_sweep(p, HB_SWEEP_UP[::-1], start=s_end)
        res["up"], res["down"] = up, dn
    elif which == "decay":
        rows = []
        g = np.zeros(2)
        for kn in HB_DECAY_KN:
            u = kn * KN
            sol, r = S1.newton(p, u, g)
            if r <= 1e-10:
                g = sol
            for dz in (0.001, 0.05):
                zz, tt = S1.decay_run(p, u, sol[0], sol[1], dz)
                k = int(round(10 / p.dt))
                env = [float(np.abs(zz[j * k:(j + 1) * k]).max())
                       for j in range(len(zz) // k)]
                falling = env[-1] < 0.9 * env[-2] or env[-1] < 1e-12
                rows.append(dict(kn=kn, dz=dz, resid=r, z=float(sol[0]),
                                 trim=S1.trim_deg(p, sol[1]), env=env,
                                 decays=bool(env[-1] < 0.01 * env[0]
                                             and falling and r <= 1e-10)))
        res["rows"] = rows
        d55 = []
        g = np.zeros(2)
        for kn in HB_D55_KN:
            u = kn * KN
            sol, r = S1.newton(p, u, g)
            g = sol
            D = S1.free_D(p, u, sol[0], sol[1])
            th = S1.continuum_D55(p, u, sol[0], sol[1])
            d55.append(dict(kn=kn, resid=r, D55_strip=float(D[1, 1]),
                            D55_continuum=th["total"],
                            rel=float(D[1, 1] / th["total"] - 1.0)))
        res["d55"] = d55
    res["seconds"] = time.time() - t0
    return res


def part_hullb_s1():
    from studies.parallel_runner import run_parallel
    t0 = time.time()
    out("HULLB_S1: S1's standing checks on planing_b@as1 (preset "
        "switches), through S1's functions")
    res = run_parallel(hullb_s1_job, ["census", "sweep", "decay"],
                       gb_per_job=0.3, reserve_gb=3.0)
    bad = [r for r in res if isinstance(r, Exception)]
    if bad:
        check("hullb_s1", "every job ran", False, [repr(r) for r in bad])
        flush("hullb_s1.txt")
        return
    R = {r["which"]: r for r in res}
    # census
    cen = R["census"]["rows"]
    for r in cen:
        st = [x for x in r["roots"] if x["label"] == "stable"]
        out(f"    census {r['kn']:4.0f} kn: {len(r['roots'])} root(s), "
            f"{r['n_stable']} stable "
            + ", ".join(f"(z {x['z']:+.4f}, trim {x['trim']:.3f})"
                        for x in st)
            + f"; zeros away from roots {len(r['free_zeros'])}")
    check("hullb_s1", "census: exactly one stable root at each of "
          f"{len(HB_CENSUS_KN)} speeds 2-40 kn and no zero of the residual "
          "away from the roots (winding scan)",
          all(r["n_stable"] == 1 and not r["free_zeros"] for r in cen),
          [(r["kn"], r["n_stable"], len(r["free_zeros"])) for r in cen])
    # sweep
    up, dn = R["sweep"]["up"], R["sweep"]["down"]
    dmap = {r["kn"]: r for r in dn}
    hyst = max(abs(a["trim"] - dmap[a["kn"]]["trim"]) for a in up)
    hk = max(up, key=lambda a: abs(a["trim"] - dmap[a["kn"]]["trim"]))["kn"]
    lc = sorted({r["kn"] for r in up + dn if r["trim_pp"] > 0.01
                 or r["z_pp"] > 1e-3})
    jumps = max(abs(up[i + 1]["trim"] - up[i]["trim"])
                for i in range(len(up) - 1))
    for a in up:
        b = dmap[a["kn"]]
        out(f"    sweep {a['kn']:4.0f} kn: up z {a['z']:+.4f} trim "
            f"{a['trim']:6.3f} (p-p {a['trim_pp']:.1e}) | down z "
            f"{b['z']:+.4f} trim {b['trim']:6.3f} (p-p {b['trim_pp']:.1e})"
            f" | up-down {a['trim'] - b['trim']:+.4f}")
    check("hullb_s1", "stepped sweep 1-40 kn up and down: no limit cycle "
          "(trim p-p <= 0.01 deg, z p-p <= 1 mm) and up-down trim "
          "difference <= 0.05 deg at every speed",
          not lc and hyst <= 0.05,
          f"largest up-down {hyst:.4f} deg at {hk:g} kn; limit cycles at "
          f"{lc}; largest trim change between speeds {jumps:.3f} deg")
    # decay
    dec = R["decay"]["rows"]
    for r in dec:
        out(f"    decay {r['kn']:4.0f} kn dz {r['dz']:.3f}: root resid "
            f"{r['resid']:.0e}, envelope {r['env'][0]:.2e} -> "
            f"{r['env'][-1]:.2e}: {'decays' if r['decays'] else 'NO'}")
    check("hullb_s1", "decay from 1 mm and 5 cm at 3, 6, 9, 12 kn: every "
          "run decays (S1's verdict: last 10 s envelope < 1 % of the first "
          "and still falling, from a Newton root)",
          all(r["decays"] for r in dec),
          f"{sum(r['decays'] for r in dec)} of {len(dec)}")
    d55 = R["decay"]["d55"]
    for r in d55:
        out(f"    D55 {r['kn']:3.0f} kn: strip {r['D55_strip']:.1f}, "
            f"continuum {r['D55_continuum']:.1f} N m s ({100 * r['rel']:+.2f}"
            f" %)")
    check("hullb_s1", "D55 of the plant against S1's continuum reference "
          "(exact geometry from hydro/hull_geometry) within 5 % at 2, 4, "
          "6 kn", all(abs(r["rel"]) <= 0.05 for r in d55),
          [f"{r['kn']:g} kn {100 * r['rel']:+.2f} %" for r in d55])
    dump("hullb_s1.json", dict(results=R, seconds=time.time() - t0))
    out(f"  ({time.time() - t0:.0f} s; jobs "
        + ", ".join(f"{k} {v['seconds']:.0f} s" for k, v in R.items()) + ")")
    flush("hullb_s1.txt")


# ================================================================ regress
def part_regress():
    t0 = time.time()
    out("REGRESS: v0 bit-identical, as1 exact")
    py = sys.executable
    cmds = (["-m", "studies.anyspeed_regress_as", "--all", "--report",
             "p0_regress_as.json"],
            ["-m", "studies.anyspeed_regress", "--golden", "check", "--lofi",
             "--turn", "--streams", "--rows", "hand,m0", "--missions",
             "s0_l0,s0_l1", "--report", "p0_regress_v0.json"])
    procs = []
    os.makedirs(OUT, exist_ok=True)
    for i, c in enumerate(cmds):
        log = open(os.path.join(OUT, f"regress_{i}_stdout.log"), "w")
        procs.append((c, subprocess.Popen([py] + c, cwd=ROOT, stdout=log,
                                          stderr=subprocess.STDOUT), log))
    for c, pr, log in procs:
        rc = pr.wait()
        log.close()
        out(f"    $ {' '.join(c)}: exit {rc}")
    as_rep = json.load(open(os.path.join(ROOT, "studies", "_cache",
                                         "anyspeed", "p0_regress_as.json")))
    check("regress", "studies/anyspeed_regress_as.py --all: every check "
          "passes", as_rep["n_failed"] == 0,
          f"{as_rep['n_checks']} checks, {as_rep['n_failed']} failed")
    v0 = json.load(open(os.path.join(ROOT, "studies", "_cache", "anyspeed",
                                     "p0_regress_v0.json")))
    ok = (v0["golden"]["identical"] and v0["lofi"]["exact"]
          and v0["turn"]["identical"] and v0["streams"]["identical"]
          and all(m["identical"] for m in v0["missions"])
          and len(v0["missions"]) == 4)
    check("regress", "studies/anyspeed_regress.py: golden, lofi "
          "fingerprint, turning table, onestep stream and the hand / m0 "
          "missions bit-identical to the v0 references", ok,
          {"golden": v0["golden"]["identical"],
           "lofi": v0["lofi"]["exact"], "turn": v0["turn"]["identical"],
           "streams": v0["streams"]["identical"],
           "missions": [m["identical"] for m in v0["missions"]]})
    out(f"  ({time.time() - t0:.0f} s)")
    flush("regress.txt")


def main():
    ap = argparse.ArgumentParser()
    parts = ("geom", "cap", "cache", "mppi", "steptime", "twin", "hullb",
             "hullb_s1", "regress")
    for k in parts:
        ap.add_argument(f"--{k}", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    t0 = time.time()
    for k in parts:
        if getattr(a, k):
            globals()[f"part_{k}"]()
    n = sum(len(v) for v in RESULTS.values())
    bad = sum(not r["ok"] for v in RESULTS.values() for r in v)
    tag = "_".join(k for k in parts if getattr(a, k))
    dump(f"results_{tag}.json", RESULTS)
    print(f"\n{n} checks, {bad} failed, {time.time() - t0:.0f} s", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
