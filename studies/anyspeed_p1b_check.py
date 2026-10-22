#!/usr/bin/env python3
"""
Verification of step P1b of the any-speed plant (sim/planing_anyspeed.py,
switches fk_exact, orbital, wagner_entry, lateral_phi0; hydro/section2d.py;
learn/meta/ACCEPTANCE_2026-10-06.md). The checks and their bounds were
registered before any of them ran, in learn/meta/ANYSPEED_PLAN_2026-10-05.md,
section '进展 / P1b' (the numbers below are those items).

The candidate is sim/planing_anyspeed.P1B_CANDIDATE (the as1 preset of P1a
plus the four switches). Parts that build 'NAME@as1' plants through the
selector put it into PRESETS['as1'] in their own process (use_candidate);
the P1a preset is the constant AS1_P1A. Every .txt output starts with a
stamp line (switches, physics sha1s, the as1 code sha1, this script's
sha1), every .json carries '_stamp'.

Parts (outputs in studies/_cache/anyspeed/p1b/, or --out):
  sec2d    1  the 2D phi = 0 BEM (hydro/section2d) against analytic values
              (semicircle, half-ellipses, plate), the ledger's numbers (20
              deg wedge, scarab195 at rest), the table against direct BEM at
              random points, heel parity
  fk2d     2  the section integrals' closed forms against quadrature; the
              plant's linear wave force (6 DOF) against an independent
              quadrature of the same integrals (rest and 25 kn attitudes)
  fk3d     3  the plant's zero-speed linear Froude-Krylov force (6 DOF)
              against Capytaine's on a fine mesh of the same geometry
  orb      4  orbital kinematics against independent Airy formulas;
              Galilean invariance in a very long following wave
  latexc   5  zero-speed lateral excitation (FK + diffraction) of the
              candidate against Capytaine, long waves
  lat3d    6  lateral (and the vertical m_0) added masses at rest against
              Capytaine at infinite frequency (recorded); table sums against
              direct BEM sums (gate)
  sym      7  port/starboard mirror symmetry of the derivative
  wag      8  wagner_entry: rest added mass, calm branch unchanged, rest
              oscillation frequencies, hidden state, material derivative
  energy   9  P1a's energy part on the candidate; the lateral energy at rest
  twin    10  Froude twins (P0's machinery) on the candidate
  stand   11  P1a's standing parts (rest, astern, e2speed, nr, hullb) on the
              candidate, and the low-speed longitudinal wave force against
              a_b x the exact linear value (fk11)
  v1      12  Savitsky, Fridsma 1969/1971 (P1a's part v1) on the candidate
  golden  13  the candidate's golden runs in two processes (anyspeed_regress
              golden_runs), identical
  steptime 14 E1 (P0's interleaved method) on the candidate
  s1          S1's standing checks (studies/anyspeed_p0_s1rerun.py) on the
              candidate, one process per part
  diag        diagnoses written after the final run (recorded, no gates):
              the beam-sea lateral excitation split into Froude-Krylov and
              diffraction (5), P1a's energy start with the candidate's own
              energy (9), P0's twin open loop by switch set (10: the heel),
              the horizontal force of the fixed-waterline planes against
              -d/dt sum m V tau and the 600 s rest runs by switch set (R3)
  fkconv      3 (diagnosis, recorded): the zero-speed longitudinal wave force
              against its continuum integral as the station count grows; the
              centre of buoyancy of the strips against Capytaine's mesh

    .venv/Scripts/python.exe -m studies.anyspeed_p1b_check --parts sec2d
    .venv/Scripts/python.exe -m studies.anyspeed_p1b_check --jobs \\
        sec2d,fk2d,fk3d,orb,latexc,lat3d,sym,wag --out <dir>

Since the P1b review: --cand w / f / fo / l checks one switch on the P1a
preset (P1a + wagner_entry / fk_exact / fk_exact + orbital / lateral_phi0)
instead of the four-switch candidate (the default, p1b); --stand <part>
runs one of the stand parts (rest, astern, e2speed, nr, hullb), and in
--jobs 'stand:<part>' gives each its own process. Part fk2d's reference
(ref_lin_fk) integrates fk_exact's second-order model along the strips;
part latexc uses the mesh of the exact geometry placed as the plant places
its hull (hull_mesh_exact); part golden builds the as plant (it compared
v0's runs before: golden_job). The review's own checks are in studies/
anyspeed_p1b_fix.py.
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

OUT = os.path.join(HERE, "_cache", "anyspeed", "p1b")
KN = 0.514444
G = 9.81
LINES = []
CHECKS = []
AS = "scarab195@as1"
STAMP = None
# the switch set the parts check (P1b review: one switch at a time); None:
# the P1b candidate (sim/planing_anyspeed.P1B_CANDIDATE). --cand sets it
CAND = None
CANDS = dict(
    p1b=None,
    w=("dDt_full", "chine_exact_all", "kpp_strip", "transom", "normal_rate",
       "verlet", "wagner_entry"),
    f=("dDt_full", "chine_exact_all", "kpp_strip", "transom", "normal_rate",
       "verlet", "fk_exact"),
    fo=("dDt_full", "chine_exact_all", "kpp_strip", "transom", "normal_rate",
        "verlet", "fk_exact", "orbital"),
    l=("dDt_full", "chine_exact_all", "kpp_strip", "transom", "normal_rate",
       "verlet", "lateral_phi0"),
    # P1c: the as1 preset that adopted fk_exact (with the implementation
    # switch fast_path), for running fk_exact's standing parts on it
    as1=("dDt_full", "chine_exact_all", "kpp_strip", "transom",
         "normal_rate", "verlet", "fk_exact", "fast_path"))


def cand():
    """The switch set under check (CAND, else the P1b candidate)."""
    if CAND is not None:
        return tuple(CAND)
    from sim.planing_anyspeed import P1B_CANDIDATE
    return tuple(P1B_CANDIDATE)


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def stamp_all():
    import hashlib
    from hydro import hulls
    from sim import physics
    P1B_CANDIDATE = cand()
    st = dict(switches=list(P1B_CANDIDATE),
              code_sha1_as1=physics.code_sha1("as1"))
    for key in (AS, "planing_b@as1"):
        st[key] = physics.stamp_for(hulls.get(key), "as1",
                                    P1B_CANDIDATE)["physics_sha1"]
    with open(os.path.abspath(__file__), "rb") as f:
        st["check_sha1"] = hashlib.sha1(f.read()).hexdigest()
    st["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return st


def stamp_line():
    return "# stamp " + json.dumps(STAMP, sort_keys=True) if STAMP else ""


def flush(name):
    os.makedirs(OUT, exist_ok=True)
    head = [stamp_line()] if STAMP else []
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write("\n".join(head + LINES) + "\n")
    LINES.clear()


def _js(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, complex):
        return [v.real, v.imag]
    return str(v)


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    if STAMP:
        obj = (dict(obj, _stamp=STAMP) if isinstance(obj, dict)
               else dict(_stamp=STAMP, rows=obj))
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=_js)


def check(part, name, ok, detail=""):
    ok = bool(ok)
    CHECKS.append(dict(part=part, name=name, ok=ok, detail=str(detail)))
    out(f"  [{'ok' if ok else 'FAIL'}] {part}: {name}"
        + (f"  -- {detail}" if detail != "" else ""))
    return ok


def use_candidate():
    """The candidate as the as1 preset of this process (parts that build
    'NAME@as1' through the selector, P1a's and S1's helpers)."""
    from sim import planing_anyspeed as PA
    PA.PRESETS["as1"] = PA.resolve("as1", cand())


def calm():
    from sim.test_vessel import Monochromatic
    return Monochromatic(1.0, 0.0)


def plant(hull=AS, sea=None, sw=None, captive_u=None, dt=0.02, gz=True):
    from hydro import hulls
    from sim.planing_anyspeed import PlaningVesselAS, PlaningVesselASLin
    h = hulls.get(hull) if isinstance(hull, str) else hull
    sea = calm() if sea is None else sea
    cls = PlaningVesselAS if gz else PlaningVesselASLin
    return cls(h, sea, dt=dt, captive_u=captive_u,
               test_switches=cand() if sw is None else sw)


def mono(om, a, th):
    from sim.test_vessel import Monochromatic
    return Monochromatic(om, a, theta0=th)


def S1():
    from studies import anyspeed_s1_check as s1
    return s1


# ================================================================== sec2d
def part_sec2d():
    from hydro import section2d as S
    from hydro import hulls
    out("P1b (sec2d)  1: the gravity-free (phi = 0) 2D section added masses "
        "(hydro/section2d)")
    th = np.pi * S._cluster(240)
    rows = []
    # (a) analytic bodies, 240 cosine-clustered points on the contour
    a = 1.0
    Yc, Zc = a * np.cos(th), -a * np.sin(th)
    M, Mr = S.added_mass_2d(Yc, Zc, ref=(0.0, 0.0))
    e = [M[1, 1] / (np.pi / 2) - 1, M[0, 0] / (2 / np.pi) - 1]
    rows.append(dict(body="semicircle", heave=M[1, 1], sway=M[0, 0],
                     roll=M[2, 2], sway_roll=M[0, 2], asym=float(np.abs(
                         Mr - Mr.T).max())))
    out(f"    semicircle a = 1: heave {M[1, 1]:.6f} (pi/2 {np.pi / 2:.6f}, "
        f"{e[0]:+.2e}), sway {M[0, 0]:.6f} (2/pi {2 / np.pi:.6f}, "
        f"{e[1]:+.2e}), roll about the centre {M[2, 2]:.2e}, sway-roll "
        f"{M[0, 2]:.2e}")
    errs = list(np.abs(e))
    zero_ok = abs(M[2, 2]) <= 1e-6 and abs(M[0, 2]) <= 1e-6
    asym = [float(np.abs(Mr - Mr.T).max() / np.abs(Mr).max())]
    for ae, be in ((0.4, 1.0), (3.0, 1.0)):
        Ye, Ze = ae * np.cos(th), -be * np.sin(th)
        M, Mr = S.added_mass_2d(Ye, Ze)
        ex = [np.pi / 2 * ae ** 2, 2 / np.pi * be ** 2,
              0.5 * np.pi / 8 * (ae ** 2 - be ** 2) ** 2]
        got = [M[1, 1], M[0, 0], M[2, 2]]
        ee = [g / x - 1 for g, x in zip(got, ex)]
        errs += [abs(v) for v in ee]
        asym.append(float(np.abs(Mr - Mr.T).max() / np.abs(Mr).max()))
        rows.append(dict(body=f"half-ellipse a {ae} b {be}", got=got,
                         exact=ex, rel=ee))
        out(f"    half-ellipse a {ae} b {be}: heave {ee[0]:+.2e}, sway "
            f"{ee[1]:+.2e}, roll about the waterline centre {ee[2]:+.2e} "
            f"(relative to pi/2 rho a^2, 2/pi rho b^2, (pi/16) rho (a^2 - "
            f"b^2)^2)")
    check("sec2d", "1a: BEM vs analytic for smooth bodies (semicircle, "
          "half-ellipses a/b 0.4 and 3: heave, sway, roll): relative error "
          "<= 1e-3; semicircle roll and sway-roll about the centre 0 "
          "(<= 1e-6)", max(errs) <= 1e-3 and zero_ok,
          f"largest {max(errs):.2e}; roll {rows[0]['roll']:.1e}, sway-roll "
          f"{rows[0]['sway_roll']:.1e}")
    # Chebyshev series of the plate (independent of the BEM)
    thc = np.linspace(0, np.pi, 200001)
    v = np.sign(np.cos(thc))
    tot = 0.0
    for nn in range(1, 4001):
        cn = 2 / np.pi * np.trapezoid(v * np.sin(thc) * np.sin(nn * thc), thc)
        tot += cn * cn / nn
    out(f"    vertical plate, Chebyshev series (4000 terms): phi0 / rigid "
        f"lid = {tot:.8f}; 4/pi^2 = {4 / np.pi ** 2:.8f} (so phi0 sway = "
        f"(4/pi^2)(pi/2) rho d^2 = 2/pi rho d^2)")
    check("sec2d", "1a: the plate's Chebyshev series equals 4/pi^2 (2/pi rho "
          "d^2), within 1e-6", abs(tot - 4 / np.pi ** 2) <= 1e-6,
          f"{tot - 4 / np.pi ** 2:+.2e}")
    Ys, Zs = S.polyline([(0.005, 0.0), (0.005, -1.0), (-0.005, -1.0),
                         (-0.005, 0.0)], n_per=160)
    Ms, _ = S.added_mass_2d(Ys, Zs)
    es = Ms[0, 0] / (2 / np.pi) - 1
    check("sec2d", "1a: a 1%-thick vertical slab, sway vs 2/pi rho d^2: "
          "within 1 %", abs(es) <= 0.01, f"{100 * es:+.2f} % (the slab's "
          f"own thickness raises it: rigid lid on the same slab "
          f"{S.added_mass_2d(Ys, Zs)[0][0, 0]:.4f} -- see the record)")
    check("sec2d", "1a: added-mass matrices symmetric (|a_ij - a_ji| <= 1e-6 "
          "of the scale)", max(asym) <= 1e-6, f"largest {max(asym):.1e}")
    # re-registered after the first run (plan doc, 'P1b 补充登记')
    rat_slab = Ms[0, 0] / _lid_sway(Ys, Zs)
    check("sec2d", "1a' (re-registered): the 1%-thick slab's phi = 0 / rigid "
          "lid ratio vs 4/pi^2: within 1 %", abs(rat_slab / (4 / np.pi ** 2)
                                                - 1) <= 0.01,
          f"{rat_slab:.5f} vs {4 / np.pi ** 2:.5f} "
          f"({100 * (rat_slab / (4 / np.pi ** 2) - 1):+.2f} %)")
    check("sec2d", "1a'' (re-registered): |a_ij - a_ji| <= 1e-4 of the scale "
          "(the collocation method's discretisation level)", max(asym) <= 1e-4,
          f"largest {max(asym):.1e}")
    # (b) the ledger's numbers
    tab = S.table_for(20.0)
    tb = tab.tb
    ratio = (np.pi / 2) ** 3 / (tab.C33w * tb * tb)
    p = plant(AS, sw=None)
    from sim.planing_vessel import SP
    d0 = -(p.Z_rest + p.zk + SP * p.x * p.Th_rest)
    h33 = h22 = 0.0
    for j in range(p.x.size):
        if d0[j] <= 0:
            continue
        m = S.section_matrix(p.yc[j], p.yc[j] * tb, d0[j], 0.0)
        h33 += m[2]
        h22 += m[0]
    h33 *= p.rho * p.dx
    h22 *= p.rho * p.dx
    out(f"    20 deg wedge (no chine): Wagner (pi/2)^3 / exact phi0 = "
        f"{ratio:.4f} (ledger 2.81); scarab195 at rest, 48 stations, "
        f"upright: heave {h33:.1f} kg (ledger 5247), sway {h22:.1f} kg "
        f"(ledger 340)")
    check("sec2d", "1b: the 20 deg wedge's Wagner / exact ratio 2.81 within "
          "0.5 %", abs(ratio / 2.81 - 1) <= 0.005, f"{ratio:.4f}")
    check("sec2d", "1b: scarab195 rest heave sum 5247 kg within 0.5 %, sway "
          "340 kg within 1 %", abs(h33 / 5247 - 1) <= 0.005
          and abs(h22 / 340 - 1) <= 0.01, f"{h33:.1f} kg ({100 * (h33 / 5247 - 1):+.2f} %), "
          f"{h22:.1f} kg ({100 * (h22 / 340 - 1):+.2f} %)")
    # (c) the table against direct BEM, and heel parity
    rng = np.random.default_rng(20261006)
    bad, worst, rec = 0, 0.0, []
    worst_small = 0.0
    for i in range(300):
        beta = 20.0 if i % 2 == 0 else 16.0
        t_ = S.table_for(beta)
        yc = float(rng.uniform(0.05, 1.2))
        dch = yc * t_.tb
        r = float(np.exp(rng.uniform(np.log(0.02), np.log(8.0))))
        ph = float(np.radians(rng.uniform(-45.0, 45.0)))
        d = r * dch
        C = t_.lateral(np.array([d]), np.array([yc]), ph)[0][0]
        ref = S.section_matrix(yc, dch, d, ph)
        if r >= 0.2:
            sc = np.sqrt(np.abs(ref[[0, 0, 2, 0, 2, 5]]
                                * ref[[0, 2, 2, 5, 5, 5]]))
            e_ = float(np.max(np.abs(C - ref) / np.maximum(sc, 1e-300)))
            worst = max(worst, e_)
            if e_ > 0.01:
                bad += 1
                rec.append(dict(beta=beta, r=r, phi_deg=math.degrees(ph),
                                err=e_))
        else:
            ref1 = S.section_matrix(yc, dch, dch, ph)
            sc1 = np.sqrt(np.abs(ref1[[0, 0, 2, 0, 2, 5]]
                                 * ref1[[0, 2, 2, 5, 5, 5]]))
            e_ = float(np.max(np.abs(C - ref) / np.maximum(sc1, 1e-300)))
            worst_small = max(worst_small, e_)
            if e_ > 1e-3:
                bad += 1
                rec.append(dict(beta=beta, r=r, phi_deg=math.degrees(ph),
                                err_small=e_))
    out(f"    table vs direct BEM, 300 random points (beta 20 and 16, r = "
        f"d/d_ch log-uniform 0.02-8, heel -45..45 deg): worst r >= 0.2 "
        f"{100 * worst:.3f} % of the value; worst r < 0.2 {worst_small:.2e} "
        f"of the r = 1 value; outside the bounds: {rec}")
    check("sec2d", "1c: table vs direct BEM at 300 random points: r >= 0.2 "
          "within 1 % of each value (couplings: of sqrt(a_ii a_jj)); r < 0.2 "
          "within 1e-3 of the r = 1 value", bad == 0,
          f"{bad} outside; worst {100 * worst:.3f} %, small {worst_small:.1e}")
    par = 0.0
    for i in range(20):
        yc = float(rng.uniform(0.1, 1.1))
        dch = yc * tb
        d = float(rng.uniform(0.3, 2.5)) * dch
        ph = float(np.radians(rng.uniform(1.0, 40.0)))
        a_ = S.section_matrix(yc, dch, d, ph)
        b_ = S.section_matrix(yc, dch, d, -ph)
        sgn = np.where(S.ODD, -1.0, 1.0)
        par = max(par, float(np.max(np.abs(a_ - sgn * b_)) / np.abs(a_).max()))
    check("sec2d", "1c: heel parity of the BEM (a22, a33, a44, a24 even; a23,"
          " a34 odd): within 1e-10", par <= 1e-10, f"{par:.1e}")
    # panel convergence of the table's resolution (recorded)
    conv = []
    for (r, ph) in ((0.5, 0.0), (1.0, 0.0), (1.3, 0.0), (2.5, 0.0),
                    (0.8, 0.2), (1.5, 0.3), (2.9, 0.78)):
        a1 = S.section_matrix(1 / tb, 1.0, r, ph)
        a2 = S.section_matrix(1 / tb, 1.0, r, ph, n_base=160, n_log=80)
        sc = np.sqrt(np.abs(a2[[0, 0, 2, 0, 2, 5]] * a2[[0, 2, 2, 5, 5, 5]]))
        conv.append(float(np.max(np.abs(a1 - a2) / sc)))
    out(f"    the table's panel rule vs 4x panels (recorded): largest "
        f"relative difference {100 * max(conv):.3f} %")
    dump("sec2d.json", dict(rows=rows, ratio=ratio, h33=h33, h22=h22,
                            table_worst=worst, table_worst_small=worst_small,
                            table_bad=rec, parity=par, conv=conv, plate=tot,
                            slab=float(es)))
    flush("sec2d.txt")


def _lid_sway(Y, Z):
    """Sway added mass (per unit density) of the contour with a rigid lid
    (positive image), the same constant-panel method."""
    from hydro import section2d as S
    P = np.stack([Y, Z], -1)
    a, b = P[:-1], P[1:]
    h = np.linalg.norm(b - a, axis=1)
    t = (b - a) / h[:, None]
    n = np.stack([t[:, 1], -t[:, 0]], -1)
    m = 0.5 * (a + b)
    S0, D0 = S._influence(m, m, t, n, h)
    R = np.array([1.0, -1.0])
    Si, Di = S._influence(m, m * R, t * R, n * R, h)
    A = 0.5 * np.eye(h.size) - (D0 + Di)
    phi = np.linalg.solve(A, -(S0 + Si) @ n[:, 0])
    return float((phi * n[:, 0] * h).sum())


# ================================================================== fk2d
def _quad_section(k, lam, d, yc, tb, n=60):
    """(E, E_k, E_lam, W, Wy) of section_fk by Gauss-Legendre quadrature
    (independent)."""
    from numpy.polynomial.legendre import leggauss
    xg, wg = leggauss(n)
    dch = yc * tb
    dp = max(d, 0.0)
    h1 = min(dp, dch)
    E = Ek = El = 0.0
    for lo, hi in ((-dp, -dp + h1), (-dp + h1, 0.0)):
        if hi <= lo:
            continue
        z = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        wz = 0.5 * (hi - lo) * wg
        s_ = z + dp
        w = np.minimum(s_ / tb, yc)
        y = w[:, None] * xg[None, :]
        wy = w[:, None] * wg[None, :]
        f = np.exp(k * z)[:, None]
        E += np.sum(wz[:, None] * wy * f * np.cos(lam * y))
        Ek += np.sum(wz[:, None] * wy * z[:, None] * f * np.cos(lam * y))
        El += np.sum(wz[:, None] * wy * (-y * np.sin(lam * y)) * f)
    c = h1 / tb
    y = c * xg
    wy = c * wg
    return np.array([E, Ek, El, np.sum(wy * np.cos(lam * y)),
                     np.sum(wy * y * np.sin(lam * y))])


def plant_lin_fk(p, s, om, th, a=1e-4, extra_fk=True):
    """The plant's linear wave force (complex amplitude per unit wave
    amplitude, e^{-i w t} convention) at state s (x0 = y0 = 0, the plant
    held), 6 DOF in the moments' convention of the plant (about the CG;
    pitch about +y, roll about +x, yaw about +z): central difference in the
    amplitude, phases 0 and T/4. Components from the derivative's own
    outputs: surge Xh, sway / roll / yaw the lateral right-hand sides,
    heave the pressure part of the strip force, pitch its moment plus Mh
    and the strip-exact sigma' term."""
    T = 2 * math.pi / om
    res = np.zeros((2, 6))
    for ip, t in enumerate((0.0, T / 4)):
        F = []
        for sgn in (1.0, -1.0):
            q = p.with_sea(mono(om, sgn * a, th))
            ds, aux = q.deriv(s, t)
            fk = aux.get("fk")
            my_sig = fk.my_sig if fk is not None else 0.0
            lat = aux["lat_rhs"]
            F.append(np.array([aux["Xh"], lat[0], aux["Fz_p"], lat[1],
                               aux["My_p"] + aux["Mh"] + my_sig, lat[2]]))
        res[ip] = (F[0] - F[1]) / (2 * a)
    return res[0] + 1j * res[1]


def _strip_nodes(g, j, n):
    """Gauss nodes and weights (body x from the geometry's origin) of strip
    j of the geometry g; on the stem strip in t with x = x_stem - t^2 (the
    reference's own quadrature of the chine's square root there)."""
    from numpy.polynomial.legendre import leggauss
    t, w = leggauss(n)
    a_, b_ = float(g.edges[j]), float(g.edges[j + 1])
    if j == g.n - 1:
        S_ = math.sqrt(b_ - a_)
        s_ = 0.5 * S_ * (t + 1.0)
        return b_ - s_ * s_, 0.5 * S_ * w * 2.0 * s_
    return 0.5 * (b_ - a_) * t + 0.5 * (a_ + b_), 0.5 * (b_ - a_) * w


def ref_lin_fk(p, s, om, th, ab, nx=24, nq=40):
    """Independent quadrature of the plant's linear incident-wave force
    (since the P1b review: its second-order model along x). Per wet station
    the section pieces of the exact linear pressure rho g e^{k z'} e^{i k_y
    y} by Gauss-Legendre over the section's wetted contour (keel, V bottom,
    walls, both sides: the vertical, lateral and roll forces) and over its
    area (the longitudinal force -int dp/dx dA, its lever, the yaw of the
    x-facing surfaces); the model along each strip: piece(x) = yc(x) (H_j +
    H'_j (x - x_j)), H = piece / yc_j at the station and H' its difference
    with the wet neighbours (central where both are wet, else one-sided),
    yc(x) the geometry's chine, integrated pointwise against the exact
    phase e^{i k_x x} (nx Gauss nodes per strip, in t = sqrt(x_stem - x)
    on the stem strip); then the transom section's and the wetted face's
    pressures as the plant writes them; all in the plant's moment
    convention; a_b on the vertical and longitudinal parts."""
    from numpy.polynomial.legendre import leggauss
    from sim.planing_vessel import SP
    from hydro.hull_geometry import stations_of
    k = om * om / G
    kx, ky = k * math.cos(th), k * math.sin(th)
    rg = p.rho * G
    qg, qw = leggauss(nq)
    z, Th = s[2], p.Th_rest + s[4]
    d = -(p.Z_rest + z + p.zk + SP * p.x * Th)
    tb = p.tanb
    g = stations_of(p.hull)
    lcg = p.hull.lcg
    F = np.zeros(6, complex)
    wet = np.nonzero(d > 0.0)[0]
    # pieces: 0 f_z, 1 f_y, 2 k_x (contour), 3 Ax, 4 Azx, 5 Ayx (area)
    Q = np.zeros((wet.size, 6), complex)
    for iw, j in enumerate(wet):
        yc = p.yc[j]
        dch = yc * tb
        dj = d[j]
        z0 = p.zk[j] + SP * p.x[j] * Th         # keel height rel. CG
        if dj > dch:
            pts = [(yc, dj), (yc, dch), (0.0, 0.0), (-yc, dch), (-yc, dj)]
        else:
            c = dj / tb
            pts = [(c, dj), (0.0, 0.0), (-c, dj)]
        for (y0, h0), (y1, h1_) in zip(pts[:-1], pts[1:]):
            L = math.hypot(y1 - y0, h1_ - h0)
            ty, tz = (y1 - y0) / L, (h1_ - h0) / L
            ny, nz = -tz, ty
            ss = 0.5 * L * (qg + 1.0)
            ws = 0.5 * L * qw
            yy = y0 + ty * ss
            hh = h0 + tz * ss
            pr = np.exp(k * (hh - dj)) * np.exp(1j * ky * yy)
            zc = z0 + hh
            lev = yy * nz - zc * ny
            Q[iw, 0] += -(pr * nz * ws).sum()
            Q[iw, 1] += -(pr * ny * ws).sum()
            Q[iw, 2] += -(pr * lev * ws).sum()
        for (yy, hh, ww) in _area_pts(yc, dch, dj, tb, qg, qw):
            pr = np.exp(k * (hh - dj)) * np.exp(1j * ky * yy)
            Q[iw, 3] += (pr * ww).sum()
            Q[iw, 4] += (pr * (z0 + hh) * ww).sum()
            Q[iw, 5] += (pr * yy * ww).sum()
    H = Q / p.yc[wet][:, None]
    Hp = np.zeros_like(H)
    for iw, j in enumerate(wet):
        lft = iw > 0 and wet[iw - 1] == j - 1
        rgt = iw < wet.size - 1 and wet[iw + 1] == j + 1
        if lft and rgt:
            Hp[iw] = (H[iw + 1] - H[iw - 1]) / (2.0 * p.dx)
        elif lft:
            Hp[iw] = (H[iw] - H[iw - 1]) / p.dx
        elif rgt:
            Hp[iw] = (H[iw + 1] - H[iw]) / p.dx
    for iw, j in enumerate(wet):
        xg, wx = _strip_nodes(g, j, nx)
        xb = xg - lcg                            # body x from the CG
        model = g.chine(xg)[:, None] * (H[iw][None, :] + Hp[iw][None, :]
                                         * (xb - p.x[j])[:, None])
        ex = (np.exp(1j * kx * xb) * wx)[:, None]
        I0 = (model * ex).sum(axis=0)
        I1 = (model * ex * xb[:, None]).sum(axis=0)
        F[2] += ab * rg * I0[0]
        F[4] += ab * rg * (-I1[0])
        F[1] += rg * I0[1]
        F[5] += rg * I1[1]
        F[3] += rg * I0[2]
        F[0] += ab * rg * (-1j * kx) * I0[3]
        F[4] += ab * rg * (-1j * kx) * I0[4]
        F[5] += rg * (1j * kx) * I0[5]
    # the transom section (hull side, at its own keel depth) and the face
    tr = p._transom_kin(0.0, 0.0, 0.0, z, Th, s[6], 0.0)
    tg = p._transom_geom()
    dT, hw, head = tr["dT"], tr["hw"], tr["head"]
    if dT > 0.0:
        exT = np.exp(1j * kx * tg.xT)
        dchT = tg.ycT * tb
        for (yy, hh, ww) in _area_pts(tg.ycT, dchT, dT, tb, qg, qw):
            pr = np.exp(k * (hh - dT)) * np.exp(1j * ky * yy)
            F[0] -= ab * rg * (pr * ww).sum() * exT
            F[4] -= ab * rg * (pr * (tr["z0T"] + hh) * ww).sum() * exT
            F[5] += rg * (pr * yy * ww).sum() * exT
        if hw > 0.0 and np.isfinite(head):
            for (yy, hh, ww) in _area_pts(tg.ycT, dchT, hw, tb, qg, qw):
                # the face below the dead-water level: depth below the
                # surface = head + (hw - hh)
                pr = np.exp(-k * (head + hw - hh)) * np.exp(1j * ky * yy)
                F[0] += ab * rg * (pr * ww).sum() * exT
                F[4] += ab * rg * (pr * (tr["z0T"] + hh) * ww).sum() * exT
                F[5] -= rg * (pr * yy * ww).sum() * exT
    return F


def _area_pts(yc, dch, d, tb, qg, qw):
    """Quadrature points (y, height above keel, weight) of the immersed
    section area, as a list of arrays (V triangle; wall rectangle)."""
    out_ = []
    h1 = min(d, dch)
    if h1 > 0.0:
        # V: height s in [0, h1], |y| <= s / tb
        s_ = 0.5 * h1 * (qg + 1.0)
        ws = 0.5 * h1 * qw
        w = s_ / tb
        yy = w[:, None] * qg[None, :]
        wy = w[:, None] * qw[None, :]
        out_.append((yy.ravel(), np.repeat(s_, qg.size),
                     (ws[:, None] * wy).ravel()))
    if d > dch:
        s_ = dch + 0.5 * (d - dch) * (qg + 1.0)
        ws = 0.5 * (d - dch) * qw
        yy = (yc * qg)[None, :] * np.ones_like(s_)[:, None]
        wy = (yc * qw)[None, :] * np.ones_like(s_)[:, None]
        out_.append((yy.ravel(), np.repeat(s_, qg.size),
                     (ws[:, None] * wy).ravel()))
    return out_


def part_fk2d():
    from hydro.section2d import section_fk, section_fk_grid
    from sim.planing_anyspeed import AS1_P1A
    out("P1b (fk2d)  2: the incident-wave integrals (M2)")
    rng = np.random.default_rng(1)
    cases = [(0.0, 0.0, 0.3, 1.0, math.tan(math.radians(20))),
             (1e-5, 0.0, 0.3, 1.0, math.tan(math.radians(20))),
             (0.5, 0.0, 0.3, 1.0, math.tan(math.radians(20))),
             (0.0, 2.0, 0.3, 1.0, math.tan(math.radians(20))),
             (3.0, 5.0, 0.364, 1.0, math.tan(math.radians(20))),
             (3.0, 5.0, 0.6, 1.0, math.tan(math.radians(20))),
             (1e-4, 1e-4, 0.5, 1.0, math.tan(math.radians(20))),
             (2.0, 1.0, math.tan(math.radians(20)), 1.0,
              math.tan(math.radians(20)))]
    for i in range(300):
        tb = math.tan(math.radians(float(rng.choice([10, 16, 20, 30]))))
        yc = float(rng.uniform(0.05, 1.5))
        cases.append((float(rng.uniform(0, 30) * rng.choice([1, 1e-2, 1e-3])),
                      float(rng.uniform(-30, 30) * rng.choice([1, 0, 1e-2,
                                                               1e-3])),
                      float(rng.uniform(0, 3) * yc * tb), yc, tb))
    worst = 0.0
    for kk, ll, dd, yy, tb in cases:
        q = _quad_section(kk, ll, dd, yy, tb)
        c = np.array(section_fk(kk, ll, dd, yy, tb)[:5], float)
        area = min(dd, yy * tb) ** 2 / tb + 2 * yy * max(dd - yy * tb, 0)
        sc = np.array([area, area * max(dd, 1e-9), area * yy, 2 * yy,
                       2 * yy * yy]) + 1e-300
        worst = max(worst, float(np.max(np.abs(c - q) / sc)))
    check("fk2d", "2a: section closed forms (E, E_k, E_lam, W, Wy) vs Gauss "
          "quadrature at 308 cases (random and limits): within 1e-8 of the "
          "scale (area, area x depth, area x half-beam, waterline)",
          worst <= 1e-8, f"largest {worst:.2e}")
    # the plant's grid evaluation against section_fk (code path, recorded)
    g_worst = 0.0
    for trial in range(6):
        n, M = 40, 120
        tb = math.tan(math.radians(20))
        d = rng.uniform(1e-4, 0.8, n)
        yc = rng.uniform(0.01, 1.1, n)
        k = np.repeat(rng.uniform(0.04, 8, 24), 5)
        lam = k * np.sin(rng.uniform(-np.pi, np.pi, M))
        lam[:5] = 0.0
        r1 = section_fk(k[None, :], lam[None, :], d[:, None], yc[:, None], tb)
        r2 = section_fk_grid(k, lam, d, yc, tb)
        A = float((np.minimum(d, yc * tb) ** 2 / tb + 2 * yc * np.maximum(
            d - yc * tb, 0)).max())
        sc = [A, A * d.max(), A * yc.max(), 2 * yc.max(), 2 * yc.max() ** 2]
        g_worst = max(g_worst, max(float(np.max(np.abs(a_ - b_))) / s_
                                   for a_, b_, s_ in zip(r1[:5], r2[:5], sc)))
    out(f"    the plant's grid evaluation (section_fk_grid) vs section_fk: "
        f"largest difference {g_worst:.1e} of the hull-level scale")
    # (b) the plant's linear force vs independent quadrature
    sw = AS1_P1A + ("fk_exact",)
    rows = []
    worst_b = 0.0
    for att, ukn in (("rest", 0.0), ("25 kn", 25.0)):
        u = ukn * KN
        pc = plant(AS, sw=sw, captive_u=u)
        if u > 0:
            z_, th_ = pc.running_attitude(u)
        else:
            z_, th_ = 0.0, 0.0
        s = np.zeros(14)
        s[2], s[4], s[6] = z_, th_, u
        ab = float(pc._a_b(u))
        for lamL in (8 * pc.dx / pc.L, 0.3, 0.5, 1.0, 2.0, 6.0):
            k = 2 * math.pi / (lamL * pc.L)
            om = math.sqrt(G * k)
            Fp, Fr = [], []
            for thd in (0, 45, 90, 135, 180):
                th = math.radians(thd)
                fp = plant_lin_fk(pc, s, om, th)
                fr = ref_lin_fk(pc, s, om, th, ab)
                Fp.append(fp)
                Fr.append(fr)
            Fp, Fr = np.array(Fp), np.array(Fr)
            sc = np.maximum(np.abs(Fr).max(axis=0), 1e-300)
            e_ = np.abs(Fp - Fr) / sc
            worst_b = max(worst_b, float(e_.max()))
            rows.append(dict(att=att, lamL=lamL, err=e_.max(axis=0),
                             scale=sc))
            out(f"    {att:6s} lam/L {lamL:5.3f}: max |plant - quadrature| / "
                f"max over headings, per DOF (X Y Z K M N): "
                + " ".join(f"{v:.1e}" for v in e_.max(axis=0)))
    rows2 = []
    worst_c = 0.0
    for att, ukn in (("rest", 0.0), ("25 kn", 25.0)):
        u = ukn * KN
        pc = plant(AS, sw=sw, captive_u=u, gz=False)
        if u > 0:
            z_, th_ = pc.running_attitude(u)
        else:
            z_, th_ = 0.0, 0.0
        s = np.zeros(14)
        s[2], s[4], s[6] = z_, th_, u
        ab = float(pc._a_b(u))
        for lamL in (8 * pc.dx / pc.L, 0.3, 0.5, 1.0, 2.0, 6.0):
            k = 2 * math.pi / (lamL * pc.L)
            om = math.sqrt(G * k)
            Fp, Fr = [], []
            for thd in (0, 45, 90, 135, 180):
                th = math.radians(thd)
                Fp.append(plant_lin_fk(pc, s, om, th, a=1e-6))
                Fr.append(ref_lin_fk(pc, s, om, th, ab))
            Fp, Fr = np.array(Fp), np.array(Fr)
            sc = np.maximum(np.abs(Fr).max(axis=0), 1e-300)
            e_ = np.abs(Fp - Fr) / sc
            worst_c = max(worst_c, float(e_.max()))
            rows2.append(dict(att=att, lamL=lamL, err=e_.max(axis=0)))
            out(f"    (2b') linear-roll plant, a = 1e-6: {att:6s} lam/L "
                f"{lamL:5.3f}: " + " ".join(f"{v:.1e}" for v in e_.max(axis=0)))
    check("fk2d", "2b' (re-registered): the same with the linear-roll plant "
          "(PlaningVesselASLin) and amplitude 1e-6 m: within 1e-6",
          worst_c <= 1e-6, f"largest {worst_c:.2e}")
    check("fk2d", "2b: the plant's linear wave force (6 DOF, rest and 25 kn "
          "attitudes, 5 headings, lam/L 0.167-6) vs independent quadrature of "
          "the same integrals: within 1e-6 of each DOF's largest amplitude "
          "over the headings (a_b at 25 kn on vertical and longitudinal "
          "parts)", worst_b <= 1e-6, f"largest {worst_b:.2e}")
    dump("fk2d.json", dict(closed_worst=worst, grid_worst=g_worst,
                           rows=rows, rows_linear=rows2))
    flush("fk2d.txt")


# ================================================================== fk3d
def capytaine_body(p, nx, nb, ns, lid=False, nt=10):
    """Capytaine body of the plant's hull (studies/anyspeed_bem_probe
    .hull_mesh, the geometry of hydro/hull_geometry) at the plant's rest
    attitude, rigid-body dofs about the CG."""
    import capytaine as cpt
    import logging
    cpt.set_logging(logging.WARNING)
    from studies.anyspeed_bem_probe import hull_mesh
    mesh = hull_mesh(p, nx=nx, nb=nb, ns=ns)
    mesh = mesh.rotated_y(p.Th_rest).translated_z(p.Z_rest)
    cog = (0.0, 0.0, p.Z_rest)
    body = cpt.FloatingBody(
        mesh=mesh, dofs=cpt.rigid_body_dofs(rotation_center=cog),
        center_of_mass=cog,
        lid_mesh=(mesh.generate_lid(z=-1e-3) if lid else None))
    return body.immersed_part()


def hull_mesh_exact(p, h_max, n_taper=None, nt=10):
    """Capytaine mesh of the plant's hull from hydro/hull_geometry itself
    (P1b review): the exact chine (with the geometry's own floor) and keel,
    nodes along x every h_max over the constant-chine part and at x0 + (L -
    x0) sin(theta), theta equally spaced, over the elliptic taper (dense at
    the stem, where the chine goes like sqrt(x_stem - x)); girth panels
    <= h_max (bottom keel -> chine, side chine -> deck); the transom a grid
    across the beam. Placed at the plant's rest attitude by the plant's own
    small-angle map -- sections stay vertical at body x, z_e = Z_rest + z_b
    + SP x_b Th_rest -- not by a rotation, which moves every point by z_b
    sin(Th) along x (8 mm at scarab195's rest trim, the previous mesh's
    offset of the centre of buoyancy). Returns the capytaine Mesh in the
    earth frame."""
    import capytaine as cpt
    from hydro.hull_geometry import stations_of
    from sim.planing_vessel import SP
    h = p.hull
    g = stations_of(h)
    L = float(g.L)
    src = g.source
    x0 = ((1.0 - float(src["taper"])) * L if src.get("kind") == "family"
          else 0.0)
    na = max(1, int(math.ceil(x0 / h_max)))
    xa = np.linspace(0.0, x0, na + 1)
    if n_taper is None:
        n_taper = max(8, int(math.ceil(0.5 * math.pi * (L - x0) / h_max)))
    th_ = np.linspace(0.0, 0.5 * math.pi, n_taper + 1)[1:]
    xs = np.concatenate([xa, x0 + (L - x0) * np.sin(th_)])
    if src.get("transom_run"):
        xr = float(src["transom_run"]) * L
        xs = np.unique(np.concatenate([xs, [xr]]))
    nx = xs.size - 1
    yc = g.chine(xs)
    zk = -h.vcg + g.keel(xs)[0]
    tb = p.tanb
    ycm = float(yc.max())
    nb = max(4, int(math.ceil(ycm / math.cos(math.atan(tb)) / h_max)))
    ns = max(4, int(math.ceil(0.8 / h_max)))
    Y = np.zeros((nx + 1, nb + ns + 1))
    Z = np.zeros_like(Y)
    for i in range(nx + 1):
        yb = np.linspace(0.0, yc[i], nb + 1)
        zb = zk[i] + yb * tb
        zc = zk[i] + yc[i] * tb
        zs = np.linspace(zc, zk[i] + h.depth, ns + 1)[1:]
        Y[i] = np.r_[yb, np.full(ns, yc[i])]
        Z[i] = np.r_[zb, zs]
    Xb = np.repeat((xs - h.lcg)[:, None], nb + ns + 1, axis=1)
    # the plant's small-angle placement at rest (sections vertical)
    Ze = p.Z_rest + Z + SP * Xb * p.Th_rest
    verts, groups = [], []

    def quads(idx):
        return np.column_stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(),
                                idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()])
    for sign in (+1, -1):
        base = sum(v.shape[0] for v in verts)
        pts = np.stack([Xb.ravel(), sign * Y.ravel(), Ze.ravel()], 1)
        verts.append(pts)
        idx = base + np.arange(pts.shape[0]).reshape(Xb.shape)
        groups.append(("side", sign, quads(idx)))
    u = np.linspace(-1.0, 1.0, nt + 1)
    base = sum(v.shape[0] for v in verts)
    Yt = u[:, None] * Y[0][None, :]
    Zt = np.repeat(Ze[0][None, :], nt + 1, axis=0)
    pts = np.stack([np.full(Yt.size, Xb[0, 0]), Yt.ravel(), Zt.ravel()], 1)
    verts.append(pts)
    idx = base + np.arange(pts.shape[0]).reshape(Yt.shape)
    groups.append(("transom", 0, quads(idx)))
    V = np.vstack(verts)
    F = []
    for kind, sign, q in groups:
        p0, p1, p2, p3 = V[q[:, 0]], V[q[:, 1]], V[q[:, 2]], V[q[:, 3]]
        n_ = np.cross(p2 - p0, p3 - p1)
        keep = np.linalg.norm(n_, axis=1) > 1e-14
        q, n_ = q[keep], n_[keep]
        if kind == "transom":
            want = -np.sign(n_[:, 0].mean())
        else:
            want = np.sign((sign * n_[:, 1]).mean())
        F.append(q if want > 0 else q[:, ::-1])
    return cpt.Mesh(vertices=V, faces=np.vstack(F), name="hull_exact")


def capytaine_body_exact(p, h_max, lid=False):
    """Capytaine body of the plant's hull on hull_mesh_exact (P1b review:
    the geometry of hydro/hull_geometry itself, placed by the plant's own
    small-angle map at its rest attitude), rigid-body dofs about the CG."""
    import capytaine as cpt
    import logging
    cpt.set_logging(logging.WARNING)
    mesh = hull_mesh_exact(p, h_max)
    cog = (0.0, 0.0, p.Z_rest)
    body = cpt.FloatingBody(
        mesh=mesh, dofs=cpt.rigid_body_dofs(rotation_center=cog),
        center_of_mass=cog,
        lid_mesh=(mesh.generate_lid(z=-1e-3) if lid else None))
    return body.immersed_part()


def cap_fk(body, om, th, rho):
    import capytaine as cpt
    from capytaine.bem.airy_waves import froude_krylov_force
    pb = cpt.DiffractionProblem(body=body, omega=om, wave_direction=th,
                                rho=rho, g=G)
    f = froude_krylov_force(pb)
    return np.array([complex(f[k]) for k in ("Surge", "Sway", "Heave",
                                             "Roll", "Pitch", "Yaw")])


_QUAD = {}


def cap_fk_q(body, om, th, rho, cog):
    """The incident-wave force on Capytaine's mesh of the body with 2 x 2
    Gauss points per panel (Capytaine's froude_krylov_force uses the panel
    centres): -int p n dS and -int p (r - cog) x n dS, p = rho g e^{k z}
    e^{i k (x cos th + y sin th)}, Capytaine's normals and conventions."""
    q = _QUAD.get(id(body))
    if q is None:
        m = body.mesh.with_quadrature("Gauss-Legendre 2")
        pts, w = m.quadrature_points
        q = _QUAD[id(body)] = (body, np.asarray(pts), np.asarray(w),
                               np.asarray(m.faces_normals)[:, None, :])
    _, pts, w, nrm = q
    k = om * om / G
    pr = rho * G * np.exp(k * pts[..., 2]) * np.exp(
        1j * k * (pts[..., 0] * math.cos(th) + pts[..., 1] * math.sin(th)))
    F = -(pr[..., None] * nrm * w[..., None]).sum(axis=(0, 1))
    r = pts - np.asarray(cog)[None, None, :]
    Mm = -(pr[..., None] * np.cross(r, np.broadcast_to(nrm, r.shape))
           * w[..., None]).sum(axis=(0, 1))
    return np.concatenate([F, Mm])


def part_fk3d():
    from sim.planing_anyspeed import AS1_P1A
    out("P1b (fk3d)  3: zero-speed Froude-Krylov force, 6 DOF, plant vs "
        "Capytaine (same geometry, rest attitude)")
    res = {}
    for hull in (AS, "planing_b@as1"):
        pc = plant(hull, sw=AS1_P1A + ("fk_exact",), captive_u=0.0)
        s = np.zeros(14)
        L, dx = pc.L, pc.dx
        lam_list = [8 * dx, 12 * dx, 16 * dx, 0.25 * L, 0.5 * L, L, 2 * L,
                    4 * L, 8 * L]
        heads = [0, 30, 60, 90, 120, 150, 180]
        h_min = lam_list[0] / 40.0
        nx = int(math.ceil(L / h_min))
        nb = int(math.ceil(1.1 * float(pc.yc.max()) / math.cos(math.atan(
            pc.tanb)) / h_min))
        ns = int(math.ceil(0.8 / h_min))
        t0 = time.time()
        b1 = capytaine_body(pc, nx, nb, ns)
        b2 = capytaine_body(pc, int(nx / 1.5), int(nb / 1.5), int(ns / 1.5))
        out(f"    {hull}: mesh {nx} x ({nb} + {ns}) per side, "
            f"{b1.mesh.nb_faces} immersed panels; check mesh "
            f"{b2.mesh.nb_faces} panels ({time.time() - t0:.0f} s)")
        rows = []
        worst, worst_ref, zero_bad = 0.0, 0.0, 0.0
        for lw in lam_list:
            k = 2 * math.pi / lw
            om = math.sqrt(G * k)
            Fp = np.array([plant_lin_fk(pc, s, om, math.radians(h))
                           for h in heads])
            cog = (0.0, 0.0, pc.Z_rest)
            Fc = np.array([cap_fk_q(b1, om, math.radians(h), pc.rho, cog)
                           for h in heads])
            Fc2 = np.array([cap_fk_q(b2, om, math.radians(h), pc.rho, cog)
                            for h in heads])
            Fc1p = np.array([cap_fk(b1, om, math.radians(h), pc.rho)
                             for h in heads])
            sc = np.maximum(np.abs(Fc).max(axis=0), 1e-300)
            e_ = np.abs(Fp - Fc) / sc
            er = np.abs(Fc2 - Fc) / sc
            for ih, hd in enumerate(heads):
                if hd in (0, 180):
                    zero_bad = max(zero_bad, float(np.max(
                        np.abs(Fp[ih, [1, 3, 5]]) / sc[[1, 3, 5]])))
            e1p = np.abs(Fc1p - Fc) / sc
            rows.append(dict(lam=lw, lamL=lw / L, err=e_, ref_conv=er,
                             plant=Fp, cap=Fc, scale=sc, cap_1point=Fc1p,
                             cap_1point_vs_gauss=e1p))
            worst = max(worst, float(e_.max()))
            worst_ref = max(worst_ref, float(er.max()))
            out(f"    lam {lw:6.2f} m (lam/L {lw / L:5.2f}): max over headings "
                f"of |plant - Cap| / max|Cap|, X Y Z K M N: "
                + " ".join(f"{100 * v:5.2f}%" for v in e_.max(axis=0))
                + f"   (reference meshes differ {100 * er.max():.2f} %; "
                f"Capytaine's 1-point FK vs 2x2 Gauss {100 * e1p.max():.2f} %)")
        res[hull] = dict(rows=rows, worst=worst, worst_ref=worst_ref,
                         zero=zero_bad)
        if hull == AS:
            check("fk3d", "3: scarab195 zero-speed FK, 6 DOF, lam 8 station "
                  "spacings to 8 L, headings 0-180 deg every 30: |plant - "
                  "Capytaine| <= 3 % of each DOF's largest amplitude over the "
                  "headings at that wavelength", worst <= 0.03,
                  f"largest {100 * worst:.2f} %")
            check("fk3d", "3: Capytaine's own convergence (two meshes) within "
                  "0.3 % of the same scale", worst_ref <= 0.003,
                  f"{100 * worst_ref:.3f} %")
            check("fk3d", "3: symmetric zeros (sway, roll, yaw in head and "
                  "following waves) in the plant <= 1e-12 of the scale",
                  zero_bad <= 1e-12, f"{zero_bad:.1e}")
        else:
            out(f"    planing_b (recorded, no gate): largest "
                f"{100 * worst:.2f} %, reference meshes "
                f"{100 * worst_ref:.2f} %, symmetric zeros {zero_bad:.1e}")
    dump("fk3d.json", res)
    flush("fk3d.txt")


# =================================================================== orb
def part_orb():
    from sim.planing_anyspeed import _vsec
    from sim.planing_vessel import SP
    from sim.wavefield import SeaState
    out("P1b (orb)  4: orbital kinematics (M2) and Galilean invariance")
    rng = np.random.default_rng(4)
    worst = 0.0
    for sea in (SeaState(1.0, 5.0, n_freq=24, n_dir=5, seed=3),
                mono(1.8, 0.3, 2.2)):
        p = plant(AS, sea=sea)
        for trial in range(10):
            s = np.zeros(14)
            s[0], s[1] = rng.uniform(-50, 50), rng.uniform(-50, 50)
            s[2], s[4] = rng.uniform(-0.1, 0.1), rng.uniform(-0.03, 0.03)
            s[5] = rng.uniform(-math.pi, math.pi)
            s[6], s[7] = rng.uniform(0, 15), rng.uniform(-1, 1)
            s[8], s[10], s[11] = rng.normal(0, 0.2, 3)
            t = float(rng.uniform(0, 100))
            p.captive_u = None
            _, aux = p.deriv(s, t)
            kb = aux["kb"]
            d = aux["d"]
            # independent: the centroid depth and the Airy kinematics
            A, m1 = _vsec(d, p.yc, p.tanb)[:2]
            zc = np.where(A > 0, -(np.maximum(d, 0) - m1 / np.where(
                A > 0, A, 1)), 0.0)
            psi = s[5]
            X = s[0] + p.x * math.cos(psi)
            Y = s[1] + p.x * math.sin(psi)
            a = np.atleast_1d(sea.a)
            w = np.atleast_1d(sea.w)
            k = np.atleast_1d(sea.k)
            th = np.atleast_1d(sea.th)
            ph0 = np.atleast_1d(sea.phi)
            uo = np.zeros_like(X)
            vo = np.zeros_like(X)
            wo = np.zeros_like(X)
            ax = np.zeros_like(X)
            ay = np.zeros_like(X)
            az = np.zeros_like(X)
            for m in range(a.size):
                ph = k[m] * (X * math.cos(th[m]) + Y * math.sin(th[m])) \
                    - w[m] * t + ph0[m]
                dec = np.exp(k[m] * zc)
                uh = a[m] * w[m] * dec * np.cos(ph)
                ah = a[m] * w[m] ** 2 * dec * np.sin(ph)
                cb = math.cos(th[m] - psi)
                sb = math.sin(th[m] - psi)
                uo += uh * cb
                vo += uh * sb
                wo += a[m] * w[m] * dec * np.sin(ph)
                ax += ah * cb
                ay += ah * sb
                az += -a[m] * w[m] ** 2 * dec * np.cos(ph)
            for got, ref in ((kb.uo, uo), (kb.vo, vo), (kb.wo, wo),
                             (kb.aox, ax), (kb.aoy, ay), (kb.aoz, az)):
                sc = max(float(np.sqrt(np.mean(ref ** 2))), 1e-30)
                worst = max(worst, float(np.max(np.abs(got - ref))) / sc)
    check("orb", "4a: orbital velocities and accelerations at the stations' "
          "centroid depths vs independent Airy sums (task sea and a single "
          "wave, random states): within 1e-12 of the RMS scale",
          worst <= 1e-12, f"largest {worst:.1e}")
    # Galilean: a very long following wave, the hull at the crest
    lw = 1e5
    k = 2 * math.pi / lw
    om = math.sqrt(G * k)
    a = 1.0 / om
    u = 8.0
    rows = []
    worst_g = 0.0
    pw = plant(AS, sea=mono(om, a, 0.0), captive_u=u)
    pcalm = plant(AS, captive_u=u)
    z_, th_ = pcalm.running_attitude(u - 1.0)
    s = np.zeros(14)
    s[2], s[4], s[6] = z_ + a, th_, u
    _, aw = pw.deriv(s, 0.0)
    kb = aw["kb"]
    uo = float(np.mean(kb.uo[aw["d"] > 0]))
    pcalm2 = plant(AS, captive_u=u - uo)
    s2 = np.zeros(14)
    s2[2], s2[4], s2[6] = z_, th_, u - uo
    _, ac = pcalm2.deriv(s2, 0.0)
    wet = (aw["d"] > 0) & (ac["d"] > 0)
    for lab, gw, gc in (("V", kb.V[wet], ac["kb"].V[wet]),
                        ("u_r", kb.ur[wet], ac["kb"].ur[wet]),
                        ("flux", aw["mom_full"][wet], ac["mom_full"][wet]),
                        ("Df", np.array([aw["Df"]]), np.array([ac["Df"]]))):
        sc = max(float(np.max(np.abs(gc))), 1e-30)
        e_ = float(np.max(np.abs(gw - gc))) / sc
        rows.append(dict(q=lab, rel=e_))
        worst_g = max(worst_g, e_)
        out(f"    Galilean, {lab}: max |wave - calm at u - u_o| / max|calm| "
            f"= {e_:.2e}")
    fz_w = aw["Fz_p"]
    out(f"    (recorded) u_o at the crest {uo:.4f} m/s; d (wave) - d (calm) "
        f"max {float(np.max(np.abs(aw['d'] - ac['d']))):.2e} m; the "
        f"vertical pressure force differs by {fz_w - ac['Fz_p']:+.2f} N")
    check("orb", "4b: Galilean -- in a 1e5 m following wave at its crest, V, "
          "u_r, the momentum flux and the friction equal the calm plant's at "
          "u - u_o with the surface raised by the amplitude: within 2e-3",
          worst_g <= 2e-3, f"largest {worst_g:.1e}")
    dump("orb.json", dict(kin_worst=worst, galilean=rows))
    flush("orb.txt")


# ================================================================ latexc
def part_latexc():
    out("P1b (latexc)  5: zero-speed lateral excitation (FK + diffraction), "
        "candidate vs Capytaine, long waves")
    import capytaine as cpt
    pc = plant(AS, captive_u=0.0)
    s = np.zeros(14)
    L = pc.L
    # since the P1b review the mesh of the exact geometry, placed as the
    # plant places its hull (hull_mesh_exact)
    body = capytaine_body_exact(pc, 0.06, lid=True)
    out(f"    mesh of the exact geometry, {body.mesh.nb_faces} immersed panels, "
        f"set {'+'.join(cand()[6:])} on P1a")
    solver = cpt.BEMSolver()
    heads = (60, 90, 120)
    rows = []
    worst_sr, worst_y = 0.0, 0.0
    for lamL in (3.0, 6.0, 12.0, 45.0, 0.5, 1.0, 2.0):
        k = 2 * math.pi / (lamL * L)
        om = math.sqrt(G * k)
        Fp, Fc = [], []
        for hd in heads:
            th = math.radians(hd)
            T = 2 * math.pi / om
            res = np.zeros((2, 3))
            for ip, t in enumerate((0.0, T / 4)):
                F = []
                for sg in (1.0, -1.0):
                    q = pc.with_sea(mono(om, sg * 1e-3, th))
                    _, aux = q.deriv(s, t)
                    F.append(np.array(aux["lat_rhs"]))
                res[ip] = (F[0] - F[1]) / 2e-3
            Fp.append(res[0] + 1j * res[1])
            pb = cpt.DiffractionProblem(body=body, omega=om, wave_direction=th,
                                        rho=pc.rho, g=G)
            r_ = solver.solve(pb)
            from capytaine.bem.airy_waves import froude_krylov_force
            fk = froude_krylov_force(pb)
            Fc.append(np.array([complex(r_.forces[d_]) + complex(fk[d_])
                                for d_ in ("Sway", "Roll", "Yaw")]))
        Fp, Fc = np.array(Fp), np.array(Fc)
        e_sr = np.abs(Fp[:, :2] - Fc[:, :2]) / np.abs(Fc[:, :2])
        sy = np.abs(Fc[:, 2]).max()
        e_y = np.abs(Fp[:, 2] - Fc[:, 2]) / sy
        rows.append(dict(lamL=lamL, heads=heads, plant=Fp, cap=Fc,
                         err_sway_roll=e_sr, err_yaw=e_y))
        out(f"    lam/L {lamL:5.1f}: sway {'/'.join(f'{100 * v:+.1f}' for v in (np.abs(Fp[:, 0]) / np.abs(Fc[:, 0]) - 1))} % (|plant|/|Cap| - 1 at 60/90/120 deg), "
            f"error {'/'.join(f'{100 * v:.1f}' for v in e_sr[:, 0])} %; roll "
            f"error {'/'.join(f'{100 * v:.1f}' for v in e_sr[:, 1])} %; yaw "
            f"error {'/'.join(f'{100 * v:.1f}' for v in e_y)} % of max")
        if lamL >= 3.0:
            worst_sr = max(worst_sr, float(e_sr.max()))
            worst_y = max(worst_y, float(e_y.max()))
    check("latexc", "5: lateral excitation in long waves (lam/L 3, 6, 12, 45;"
          " 60, 90, 120 deg), candidate vs Capytaine FK + diffraction: sway "
          "and roll within 10 % of |Capytaine|, yaw within 10 % of the "
          "largest |Capytaine| over the headings", worst_sr <= 0.10
          and worst_y <= 0.10, f"sway/roll largest {100 * worst_sr:.1f} %, "
          f"yaw {100 * worst_y:.1f} %")
    dump("latexc.json", dict(rows=rows))
    flush("latexc.txt")


# ================================================================= lat3d
def part_lat3d():
    import capytaine as cpt
    from hydro import section2d as S
    from sim.planing_vessel import SP
    out("P1b (lat3d)  6: added masses at rest, plant (2D phi = 0 strips) vs "
        "Capytaine at infinite frequency (consistency, not validation)")
    pc = plant(AS, captive_u=0.0)
    s = np.zeros(14)
    _, aux = pc.deriv(s, 0.0)
    d = aux["d"]
    lat = pc._lateral_b(s, d, aux["kb"])
    A22, A24, A26, A44, A46, A66 = lat.A
    ma = aux["kb"].ma
    A33 = float(ma.sum() * pc.dx)
    A35 = float((ma * SP * pc.x).sum() * pc.dx)
    A55 = float((ma * pc.x * pc.x).sum() * pc.dx)
    # direct BEM strip sums (no table)
    tb = pc.tanb
    hG = -pc.zk
    dir_ = np.zeros(6)
    for j in range(pc.x.size):
        if d[j] <= 0:
            continue
        m = S.section_matrix(pc.yc[j], pc.yc[j] * tb, d[j], 0.0)
        a22, a24 = m[0], m[3] + hG[j] * m[0]
        a44 = m[5] + 2 * hG[j] * m[3] + hG[j] ** 2 * m[0]
        x = pc.x[j]
        dir_ += np.array([a22, a24, a22 * x, a44, a24 * x, a22 * x * x])
    dir_ *= pc.rho * pc.dx
    tabv = np.array([A22, A24, A26, A44, A46, A66])
    rel = np.abs(tabv - dir_) / np.maximum(np.abs(dir_), 1e-9)
    names = ("A22", "A24", "A26", "A44", "A46", "A66")
    out("    table strip sums vs direct BEM strip sums: " + ", ".join(
        f"{n} {t:.1f} / {r:.1f} ({100 * e:.3f} %)" for n, t, r, e in
        zip(names, tabv, dir_, rel)))
    # A26 and A46 are differences of signed terms: judged on the scale
    big = np.array([dir_[0], np.sqrt(abs(dir_[0] * dir_[3])),
                    np.sqrt(abs(dir_[0] * dir_[5])), dir_[3],
                    np.sqrt(abs(dir_[3] * dir_[5])), dir_[5]])
    relb = np.abs(tabv - dir_) / big
    check("lat3d", "6 (code): the plant's lateral added masses from the "
          "table equal the direct-BEM strip sums within 0.5 % (couplings of "
          "sqrt(A_ii A_jj))", relb.max() <= 0.005,
          f"largest {100 * relb.max():.3f} %")
    body = capytaine_body(pc, 96, 16, 10)
    res = {}
    for dof in ("Sway", "Roll", "Yaw", "Heave", "Pitch"):
        pb = cpt.RadiationProblem(body=body, omega=np.inf, radiating_dof=dof,
                                  rho=pc.rho, g=G)
        r_ = cpt.BEMSolver().solve(pb)
        res[dof] = {k: float(v) for k, v in r_.added_masses.items()}
    cap = dict(A22=res["Sway"]["Sway"], A24=res["Sway"]["Roll"],
               A26=res["Sway"]["Yaw"], A44=res["Roll"]["Roll"],
               A46=res["Roll"]["Yaw"], A66=res["Yaw"]["Yaw"],
               A33=res["Heave"]["Heave"], A35=res["Heave"]["Pitch"],
               A55=res["Pitch"]["Pitch"])
    plantv = dict(zip(names, tabv), A33=A33, A35=A35, A55=A55)
    out("    plant (2D strips) vs Capytaine (3D, omega = inf), rest attitude, "
        "about the CG (Capytaine's sign convention for the couplings):")
    for kname in ("A22", "A24", "A26", "A44", "A46", "A66", "A33", "A35",
                  "A55"):
        out(f"      {kname}: plant {plantv[kname]:9.1f}  Capytaine "
            f"{cap[kname]:9.1f}  ratio {plantv[kname] / cap[kname]:.3f}")
    out(f"    (v0 for comparison: sway {pc.Mtot[1, 1] - pc.m:.0f} kg at the "
        f"25 kn attitude, roll 0.2 Ix = {0.2 * pc.Ix:.0f} kg m2, heave "
        f"Zarnick {float((aux['kb'].maZ).sum() * pc.dx):.0f} kg)")
    dump("lat3d.json", dict(plant=plantv, cap=cap, direct=dict(zip(names,
                                                                   dir_)),
                            rel=relb))
    flush("lat3d.txt")


# =================================================================== sym
def _mirror_sea(sea):
    from types import SimpleNamespace
    return SimpleNamespace(a=np.atleast_1d(sea.a).copy(),
                           w=np.atleast_1d(sea.w).copy(),
                           k=np.atleast_1d(sea.k).copy(),
                           th=-np.atleast_1d(sea.th).copy(),
                           phi=np.atleast_1d(sea.phi).copy(),
                           hs=getattr(sea, "hs", None),
                           tp=getattr(sea, "tp", None), theta0=None)


def part_sym():
    from sim.wavefield import SeaState
    out("P1b (sym)  7: port/starboard mirror symmetry of the derivative "
        "(candidate, GZ plants)")
    rng = np.random.default_rng(7)
    MIR = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], float)
    worst = 0.0
    for hull in (AS, "planing_b@as1"):
        sea = SeaState(1.0, 5.0, n_freq=12, n_dir=5, seed=11, theta0=2.5)
        p = plant(hull, sea=sea)
        q = plant(hull, sea=_mirror_sea(sea))
        for i in range(20):
            s = np.zeros(14)
            s[0], s[1] = rng.uniform(-20, 20), rng.uniform(-20, 20)
            s[2], s[4] = rng.uniform(-0.05, 0.1), rng.uniform(-0.04, 0.02)
            s[3] = math.radians(rng.uniform(-20, 20))
            s[5] = rng.uniform(-math.pi, math.pi)
            s[6] = rng.uniform(0, 35 * KN)
            s[7], s[8], s[9] = rng.normal(0, [0.3, 0.2, 0.2])
            s[10], s[11] = rng.normal(0, [0.1, 0.1])
            s[12] = rng.uniform(0, 1) * p.prop.t_max
            s[13] = rng.uniform(-0.3, 0.3)
            t = float(rng.uniform(0, 60))
            d1, _ = p.deriv(s, t)
            d2, _ = q.deriv(MIR * s, t)
            sc = np.maximum(np.abs(d1), 1e-12)
            e_ = float(np.max(np.abs(MIR * d2 - d1) / np.maximum(
                np.abs(d1).max(), 1e-30)))
            worst = max(worst, e_)
    check("sym", "7: mirrored state and sea give the mirrored derivative "
          "(20 random states, 0-35 kn, heel +-20 deg, short-crested sea; "
          "scarab195 and planing_b): within 1e-12 relative", worst <= 1e-12,
          f"largest {worst:.1e}")
    dump("sym.json", dict(worst=worst))
    flush("sym.txt")


# =================================================================== wag
def _lin_freqs(p, z, th):
    """Heave-pitch undamped frequencies (rad/s) of the plant's captive
    equations at rest about (z, th), velocities zero: eigenvalues of the
    numerical Jacobian."""
    s1 = S1()
    J = s1.jac4(p, 0.0, z, th)
    lam = np.linalg.eigvals(J)
    return np.sort(np.abs(lam.imag[lam.imag > 1e-9]))


def part_wag():
    from hydro import section2d as S
    P1B_CANDIDATE = cand()
    from sim.planing_vessel import SP
    out("P1b (wag)  8: Wagner's pile-up only in an entry from dry "
        f"(switch wagner_entry; set {'+'.join(P1B_CANDIDATE[6:])} on P1a)")
    sw_off = tuple(x for x in P1B_CANDIDATE if x != "wagner_entry")
    # (a) rest
    pc = plant(AS, captive_u=0.0)
    s1 = S1()
    v, r = s1.polish(pc, 0.0, np.zeros(2), tol=1e-14, it=60)
    s = np.zeros(14)
    s[2], s[4] = v
    _, aux = pc.deriv(s, 0.0)
    kb = aux["kb"]
    A33 = float(kb.ma.sum() * pc.dx)
    A55 = float((kb.ma * pc.x ** 2).sum() * pc.dx)
    d = aux["d"]
    tb = pc.tanb
    d33 = d55 = 0.0
    for j in range(pc.x.size):
        if d[j] <= 0:
            continue
        m = S.section_matrix(pc.yc[j], pc.yc[j] * tb, d[j], 0.0)[2]
        d33 += m
        d55 += m * pc.x[j] ** 2
    d33 *= pc.rho * pc.dx
    d55 *= pc.rho * pc.dx
    out(f"    rest: A33 {A33:.1f} kg (direct BEM {d33:.1f}), A55 {A55:.1f} "
        f"kg m2 (direct {d55:.1f}); Zarnick's {float(kb.maZ.sum() * pc.dx):.0f}"
        f" kg; all wet planes fixed-waterline: {bool(kb.fixed[d > 0].all())}")
    check("wag", "8a: at rest A33, A55 equal the direct-BEM 2D phi = 0 sums "
          "within 0.2 %; A33 = 5247 kg within 0.5 %",
          abs(A33 / d33 - 1) <= 0.002 and abs(A55 / d55 - 1) <= 0.002
          and abs(A33 / 5247 - 1) <= 0.005,
          f"A33 {100 * (A33 / d33 - 1):+.3f} %, A55 {100 * (A55 / d55 - 1):+.3f}"
          f" %, vs 5247: {100 * (A33 / 5247 - 1):+.3f} %")
    # (b) calm branch unchanged
    worst = 0.0
    rows = []
    for kn in (1, 2, 4, 6, 9, 12, 16, 20, 25, 30, 35, 40):
        u = kn * KN
        res = []
        for sw in (P1B_CANDIDATE, sw_off):
            p = plant(AS, sw=sw, captive_u=u)
            ra = p.running_attitude(u)
            vv, _ = s1.polish(p, u, np.array(ra), tol=1e-13, it=60)
            R = p.calm_resistance(u)["R"]
            res.append((vv[0], vv[1], R))
        e_ = max(abs(res[0][0] - res[1][0]) / max(abs(res[1][0]), 1e-6),
                 abs(res[0][1] - res[1][1]) / max(abs(res[1][1]), 1e-6),
                 abs(res[0][2] - res[1][2]) / max(abs(res[1][2]), 1e-6))
        worst = max(worst, e_)
        rows.append(dict(kn=kn, on=res[0], off=res[1], rel=e_))
    out("    calm captive equilibria and resistance, wagner_entry on vs off: "
        + ", ".join(f"{r_['kn']} kn {r_['rel']:.1e}" for r_ in rows))
    check("wag", "8b: calm equilibria (Newton-polished from the captive "
          "run) and calm resistance at 1-40 kn identical with and without "
          "wagner_entry (relative 1e-12)", worst <= 1e-12,
          f"largest {worst:.1e}")
    worst2 = 0.0
    for r_ in rows:
        u = r_["kn"] * KN
        Rs = []
        for sw, eq in ((P1B_CANDIDATE, r_["on"]), (sw_off, r_["off"])):
            p = plant(AS, sw=sw, captive_u=u)
            Rs.append(p.captive_deriv(u, eq[0], eq[1])[1]["fx"])
        e2 = max(abs(r_["on"][0] - r_["off"][0]) / max(abs(r_["off"][0]),
                                                       1e-6),
                 abs(r_["on"][1] - r_["off"][1]) / max(abs(r_["off"][1]),
                                                       1e-6),
                 abs(Rs[0] - Rs[1]) / abs(Rs[1]))
        r_["polished_R"] = Rs
        r_["rel_polished"] = e2
        worst2 = max(worst2, e2)
    check("wag", "8b' (re-registered): the polished equilibria and the "
          "resistance at them (the captive equations) identical with and "
          "without wagner_entry, 1-40 kn (relative 1e-12)", worst2 <= 1e-12,
          f"largest {worst2:.1e}")
    # (c) small oscillation at rest
    om_lin = _lin_freqs(pc, v[0], v[1])
    p_off = plant(AS, sw=sw_off, captive_u=0.0)
    om_w = _lin_freqs(p_off, v[0], v[1])
    p = plant(AS, captive_u=0.0, dt=0.02)
    st = np.zeros(14)
    st[2], st[4] = v[0] + 1e-3, v[1]
    T, dt = 400.0, 0.02
    zs, ths = [], []
    for i in range(int(T / dt)):
        zs.append(st[2] - v[0])
        ths.append(st[4] - v[1])
        st = p.step(st, i * dt, 0.0, 0.0)
        st[6] = 0.0
    zs, ths = np.array(zs), np.array(ths)
    nfft = 2 ** 22
    fr = 2 * np.pi * np.fft.rfftfreq(nfft, dt)
    pk = []
    for sig in (zs, ths):
        sp = np.abs(np.fft.rfft((sig - sig.mean()) * np.hanning(sig.size),
                                nfft))
        # the two largest separated peaks
        i1 = int(np.argmax(sp))
        sp2 = sp.copy()
        band = int(0.3 / (fr[1] - fr[0]))
        sp2[max(0, i1 - band):i1 + band] = 0
        i2 = int(np.argmax(sp2))
        pk.append(sorted([fr[i1], fr[i2]]))
    om_sim = np.array(pk[0])
    out(f"    rest oscillation from heave + 1 mm (400 s): spectral peaks "
        f"heave {pk[0][0]:.4f}, {pk[0][1]:.4f} rad/s; pitch {pk[1][0]:.4f}, "
        f"{pk[1][1]:.4f}; linearised with m_0: {om_lin}; with Wagner's "
        f"(switch off): {om_w}")
    rel_sim = max(abs(om_sim[i] / om_lin[i] - 1) for i in range(2))
    rel_w = min(abs(om_w[i] / om_lin[i] - 1) for i in range(2))
    check("wag", "8c: rest oscillation frequencies (spectral peaks) equal the "
          "linearised ones with m_0 within 0.5 %, and differ from Wagner's by"
          " more than 5 %", rel_sim <= 0.005 and rel_w > 0.05,
          f"sim vs m_0 {100 * rel_sim:.3f} %, Wagner vs m_0 {100 * rel_w:.1f}"
          " %")
    # (d) the hidden state (since the P1b review: the planes' flow-state
    # field; F fraction 0 = every plane in an entry from dry or the exit
    # after it, 1 = all oscillating about a fixed waterline)
    p = plant(AS, dt=0.02)
    st = p.initial_state(25 * KN)
    for i in range(1000):
        st = p.step(st, i * 0.02, st[12], 0.0)
    _, aux = p.deriv(st, 20.0)
    dd = aux["d"]
    f25 = aux["kb"].f[dd > 0]
    ok_run = bool(np.all(f25 == 0.0))
    p = plant(AS, dt=0.02)
    st = p.initial_state(0.0)
    for i in range(30000):
        st = p.step(st, i * 0.02, 0.0, 0.0)
    _, aux = p.deriv(st, 600.0)
    dd = aux["d"]
    fr = aux["kb"].f[dd > 0]
    ok_rest = bool(np.all(fr == 1.0))
    check("wag", "8d (W5 a, b): hidden state -- calm 25 kn at constant "
          "thrust, 20 s: F fraction 0 at every wet station; at rest after 600"
          " s: 1 at every wet station (exact)", ok_run and ok_rest,
          f"25 kn max F fraction {float(f25.max(initial=0.0)):.1e}; rest min "
          f"{float(fr.min(initial=1.0)):.6f}")
    # (e) the two ends, bitwise
    pc2 = plant(AS, captive_u=10 * KN)
    ra = pc2.running_attitude(10 * KN)
    s = np.zeros(14)
    s[2], s[4], s[6] = ra[0], ra[1], 10 * KN
    _, aux = pc2.deriv(s, 0.0)
    kb = aux["kb"]
    wet = aux["d"] > 0
    e1 = bool(np.all(kb.ma[wet & ~kb.fixed] == kb.maZ[wet & ~kb.fixed]))
    _, aux0 = pc.deriv(np.zeros(14), 0.0)
    kb0 = aux0["kb"]
    w0 = aux0["d"] > 0
    e2 = bool(np.all(kb0.ma[w0] == kb0.m0[w0]))
    check("wag", "8e: the two ends: F fraction 0 gives Zarnick's m_a and 1 "
          "gives m_0, bit for bit", e1 and e2, f"entry {e1}, fixed {e2}")
    # (f) the material derivative of m_0 along a plane's path (M2)
    tab = pc._tab()
    yc_fun = pc._geom.chine
    dyc = pc._geom.dchine
    worst_f = 0.0
    for (x0, ur, d0, vg) in ((2.0, 3.0, 0.15, 0.4), (2.4, 1.0, 0.2, -0.3),
                             (1.0, 5.0, 0.4, 0.2), (2.55, 2.0, 0.05, 0.6)):
        def m_of(tt):
            xb = x0 - ur * tt
            dd_ = d0 + vg * tt
            return float(tab.vertical(np.array([dd_]),
                                      np.array([yc_fun(np.array([xb]))[0]]))[0][0])
        h = 1e-5
        fd = (m_of(h) - m_of(-h)) / (2 * h)
        m0, dm0, dm0y = tab.vertical(np.array([d0]),
                                     np.array([yc_fun(np.array([x0]))[0]]))
        formula = float(dm0[0] * vg - ur * dm0y[0] * dyc(np.array([x0]))[0])
        e_ = abs(formula - fd) / max(abs(fd), 1e-12)
        worst_f = max(worst_f, e_)
    check("wag", "8f: the material derivative of m_0 along a plane moving "
          "aft along the taper (dm_0/dd Vg - u_r dm_0/dyc yc'): formula vs "
          "central difference within 1e-6", worst_f <= 1e-6,
          f"largest {worst_f:.1e}")
    dump("wag.json", dict(A33=A33, A55=A55, d33=d33, d55=d55, calm=rows,
                          om_lin=om_lin, om_wagner=om_w, peaks=pk,
                          state=[ok_run, ok_rest], ends=[e1, e2],
                          mat_der=worst_f))
    flush("wag.txt")


# ================================================================ energy
def lateral_energy_run(dt, T, start):
    """The candidate GZ plant at rest, dissipation off (lateral cross-flow
    drag, roll damping, vertical cross-flow drag zero), captive surge u =
    0, from a heel, sway and yaw rate; the total energy: rigid-body kinetic
    energy, the lateral fluid kinetic energy 1/2 nu^T A(t) nu (sway, roll,
    yaw), the vertical one 1/2 (A33 w^2 + 2 A35 w q + A55 q^2) of the
    plant's own added mass, the hydrostatic potential of heave and pitch
    and the roll potential W int GZ."""
    from sim.planing_anyspeed import _vsec_q
    from sim.planing_vessel import SP
    from studies.anyspeed_p1a_check import _gz_pot_table
    p = plant(AS, captive_u=0.0, dt=dt)
    p.c_dc = 0.0
    p.cd_lat = 0.0
    p.hull = copy.deepcopy(p.hull)
    p.hull.roll_zeta = 0.0
    s1 = S1()
    v, _ = s1.polish(p, 0.0, np.zeros(2), tol=1e-14, it=60)
    rg = p.rho * G

    def U(z, th):
        d = -(p.Z_rest + z + p.zk + SP * p.x * (p.Th_rest + th))
        return float((rg * _vsec_q(d, p.yc, p.tanb)[0]).sum() * p.dx
                     + p.m * G * z)
    U0 = U(v[0], v[1])
    gzpot = _gz_pot_table(p, phi_max=0.4)
    s = np.zeros(14)
    s[2], s[4] = v
    s[3], s[7], s[11] = start
    n = int(round(T / dt))
    every = max(1, int(round(0.02 / dt)))
    E, tt = [], []
    for i in range(n + 1):
        if i % every == 0:
            _, aux = p.deriv(s, i * dt)
            kb = aux["kb"]
            lat = p._lateral_b(s, aux["d"], kb)
            nu = np.array([s[7], s[9], s[11]])
            ma = kb.ma
            A33 = ma.sum() * p.dx
            A35 = (ma * SP * p.x).sum() * p.dx
            A55 = (ma * p.x * p.x).sum() * p.dx
            Ev = 0.5 * ((p.m + A33) * s[8] ** 2 + 2 * A35 * s[8] * s[10]
                        + (p.Iy + A55) * s[10] ** 2)
            El = 0.5 * nu @ lat.M3 @ nu
            e = Ev + El + U(s[2], s[4]) - U0 + gzpot(s[3])
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
                max_dev=float(np.abs(E - E[0]).max() / E[0]))


def part_energy():
    from studies import anyspeed_p1a_check as P1A
    from sim import planing_anyspeed as PA
    out("P1b (energy)  9: energy with dissipation off at rest")
    # P1a's part on the candidate (its sw_v / sw_h are the candidate with
    # and without verlet in this process)
    PA.P1A_CANDIDATE = cand()
    P1A.OUT = OUT
    P1A.STAMP = STAMP
    sw_v = tuple(cand())
    start = (0.01, 0.003, 0.05)
    runs = []
    for dt, T in ((0.002, 100.0), (0.02, 100.0)):
        t0 = time.time()
        r = P1A.energy_run(sw_v, dt, T, start)
        r["seconds"] = time.time() - t0
        runs.append(r)
        out(f"    P1a's energy run, candidate, dt {dt}: drift "
            f"{100 * r['drift_per_100s']:+.4f} % per 100 s, max dev "
            f"{100 * r['max_dev']:.4f} %, mass variation {r['M_var']:.1e}, "
            f"max flux {r['flux_max']:.1e} N/m ({r['seconds']:.0f} s)")
    r0 = runs[0]
    check("energy", "9: P1a's energy check on the candidate (rest, "
          "dissipation off, verlet, dt 0.002): drift <= 0.1 % per 100 s",
          abs(r0["drift_per_100s"]) <= 1e-3,
          f"{100 * r0['drift_per_100s']:+.5f} % per 100 s")
    out("    (P1a's energy uses Mtot[3,3] for the roll kinetic energy and "
        "its own heave-pitch mass: with lateral_phi0 and wagner_entry the "
        "lateral run below is the consistent one)")
    lruns = []
    for dt, T in ((0.002, 100.0), (0.02, 100.0)):
        t0 = time.time()
        r = lateral_energy_run(dt, T, (math.radians(5.0), 0.2, 0.05))
        r["seconds"] = time.time() - t0
        lruns.append(r)
        out(f"    lateral energy run, dt {dt}: E0 {r['E0']:.3f} J, drift "
            f"{100 * r['drift_per_100s']:+.4f} % per 100 s, max dev "
            f"{100 * r['max_dev']:.4f} % ({r['seconds']:.0f} s)")
    check("energy", "9: lateral energy (rest, dissipation off, heel 5 deg, "
          "sway 0.2 m/s, yaw 0.05 rad/s; rigid + fluid kinetic + potential): "
          "drift <= 0.1 % per 100 s at dt 0.002",
          abs(lruns[0]["drift_per_100s"]) <= 1e-3,
          f"{100 * lruns[0]['drift_per_100s']:+.5f} % per 100 s")
    dump("energy.json", dict(p1a_runs=runs, lateral_runs=lruns))
    flush("energy.txt")


# ================================================================= twin
def part_twin():
    from studies import anyspeed_p1a_check as P1A
    use_candidate()
    P1A.OUT = OUT
    P1A.STAMP = STAMP
    LINES_ = len(P1A.LINES)
    P1A.part_twin()
    for c in P1A.CHECKS:
        CHECKS.append(dict(c, part="twin"))


# ================================================================ stand
def part_stand():
    """P1a's standing parts on the candidate (each writes its own file) and
    the low-speed longitudinal wave force against a_b x the exact linear
    value (fk11)."""
    from studies import anyspeed_p1a_check as P1A
    from sim import planing_anyspeed as PA
    use_candidate()
    PA.P1A_CANDIDATE = cand()
    P1A.OUT = OUT
    P1A.STAMP = STAMP
    for part in (STAND_ONLY or STAND_PARTS):
        getattr(P1A, f"part_{part}")()
    for c in P1A.CHECKS:
        CHECKS.append(dict(c, part=f"stand/{c['part']}"))


STAND_PARTS = ("rest", "astern", "e2speed", "nr", "hullb")
# one of them only (P1b review: the stand job split into one process per
# part; --stand)
STAND_ONLY = None


def part_fk11():
    """11 (low-speed longitudinal wave force): transom forced wet, the
    candidate's longitudinal wave force vs a_b x the exact linear FK with
    depth decay (P1a's reference 'exact'), 0.5-6 kn x lam/L 0.5-6."""
    from hydro import hulls
    from hydro import vessel_spec as VS
    from studies import anyspeed_p1a_check as P1A
    use_candidate()
    out("P1b (fk11)  11: low-speed longitudinal wave force, transom wet, vs "
        "a_b x the exact linear value")
    hw = VS.with_transom_closure(hulls.get("scarab195"), "scarab195_wet",
                                 F_T_crit=float("inf"))
    a = 0.005
    rows = []
    worst, worst_lw = 0.0, 0.0
    for kn in (0.5, 1.0, 2.0, 4.0, 6.0):
        u = kn * KN
        pc = plant(AS, captive_u=u)
        eqv, r = S1().newton(pc, u, np.zeros(2))
        for lamL in (0.5, 1.0, 2.0, 4.0, 6.0):
            k = 2 * np.pi / (lamL * pc.L)
            om = math.sqrt(G * k)
            ome = om + k * u
            q = plant(hw, sea=mono(om, a, np.pi), captive_u=u)
            Te = 2 * np.pi / ome
            ts = np.linspace(0.0, 2 * Te, 128, endpoint=False)
            X = []
            for t in ts:
                s = np.zeros(14)
                s[0], s[2], s[4], s[6] = u * t, eqv[0], eqv[1], u
                _, aux = q.deriv(s, t)
                X.append(aux["Xh"])
            Xa = P1A.first_harmonic(X, ts, ome) / a
            tb_, ex_ = P1A.fk_refs(pc, eqv[0], eqv[1], k)
            ab = float(pc._a_b(u))
            e_ = Xa / (ab * ex_) - 1
            e_lw = Xa / (ab * tb_) - 1
            worst = max(worst, abs(e_))
            worst_lw = max(worst_lw, abs(e_lw))
            rows.append(dict(kn=kn, lamL=lamL, plant=Xa, exact=ex_,
                             textbook=tb_, a_b=ab, e_exact=e_, e_lw=e_lw))
            out(f"    {kn:4.1f} kn lam/L {lamL:3.1f}: plant {Xa:9.1f} N/m, a_b x"
                f" exact {ab * ex_:9.1f} ({100 * e_:+.2f} %), a_b x long-wave "
                f"{ab * tb_:9.1f} ({100 * e_lw:+.2f} %)")
    check("fk11", "11: transom wet, longitudinal wave force vs a_b x the "
          "exact linear FK with depth decay, 0.5-6 kn x lam/L 0.5-6: within "
          "1 %", worst <= 0.01, f"largest {100 * worst:.2f} %")
    out(f"    (P1a's FK1' against the long-wave value no longer applies with "
        f"fk_exact: largest difference {100 * worst_lw:.1f} %, recorded)")
    dump("fk11.json", dict(rows=rows))
    flush("fk11.txt")


# ================================================================== v1
def part_v1():
    from studies import anyspeed_p1a_check as P1A
    from sim import planing_anyspeed as PA
    use_candidate()
    PA.P1A_CANDIDATE = cand()
    P1A.OUT = OUT
    P1A.STAMP = STAMP
    P1A.part_v1()


# ================================================================== diag
# Diagnoses written after the final run (recorded, no gates): why 5
# (yaw), 9 (P1a's energy measure) and 10 (the twin's internal bound) fail.
def _lat_rhs_harm(p, om, th):
    T = 2 * math.pi / om
    res = np.zeros((2, 3))
    for ip, t in enumerate((0.0, T / 4)):
        F = []
        for sg in (1.0, -1.0):
            q = p.with_sea(mono(om, sg * 1e-3, th))
            _, aux = q.deriv(np.zeros(14), t)
            F.append(np.array(aux["lat_rhs"]))
        res[ip] = (F[0] - F[1]) / 2e-3
    return res[0] + 1j * res[1]


def _diag_yaw():
    """5: beam-sea (90 deg) lateral excitation split into Froude-Krylov and
    diffraction, plant and Capytaine, and the yaw / sway lever of each,
    against the added-mass levers A26 / A22 (3D at omega = inf and 0;
    strips phi = 0 and rigid lid)."""
    import capytaine as cpt
    from capytaine.bem.airy_waves import froude_krylov_force
    from hydro import section2d as S
    from sim.planing_anyspeed import P1B_CANDIDATE
    out("  (diag 5) beam-sea lateral excitation, Froude-Krylov and "
        "diffraction separately; levers N / Y (m, + forward of the CG)")
    pc = plant(AS, captive_u=0.0)
    pf = plant(AS, sw=tuple(x for x in P1B_CANDIDATE
                            if x != "lateral_phi0"), captive_u=0.0)
    s = np.zeros(14)
    _, aux = pc.deriv(s, 0.0)
    d = aux["d"]
    A22, _, A26 = pc._lateral_b(s, d, aux["kb"]).A[:3]
    lid = np.zeros_like(d)
    for j in range(d.size):
        if d[j] > 0:
            yc = pc.yc[j]
            Y, Z = S.polyline(S.wetted_contour(S.section_body(
                yc, yc * pc.tanb), d[j], 0.0), scale=yc * pc.tanb)
            lid[j] = _lid_sway(Y, Z)
    L22 = pc.rho * lid.sum() * pc.dx
    L26 = pc.rho * (lid * pc.x).sum() * pc.dx
    body0 = capytaine_body(pc, 64, 10, 6)
    body = capytaine_body(pc, 64, 10, 6, lid=True)
    solver = cpt.BEMSolver()
    lev = {}
    for omv in (np.inf, 0.0):
        am = {}
        for dof in ("Sway", "Yaw"):
            r_ = solver.solve(cpt.RadiationProblem(
                body=body0, omega=omv, radiating_dof=dof, rho=pc.rho, g=G))
            am[dof] = {k: float(v) for k, v in r_.added_masses.items()}
        lev[str(omv)] = (am["Sway"]["Sway"], am["Sway"]["Yaw"])
    out(f"    added-mass levers A26/A22: strips phi = 0 {A26 / A22:+.4f} "
        f"({A22:.0f} kg), strips rigid lid {L26 / L22:+.4f} ({L22:.0f} "
        f"kg); Capytaine omega = inf {lev['inf'][1] / lev['inf'][0]:+.4f} "
        f"({lev['inf'][0]:.0f} kg), omega = 0 "
        f"{lev['0.0'][1] / lev['0.0'][0]:+.4f} ({lev['0.0'][0]:.0f} kg)")
    rows = []
    th = math.radians(90.0)
    for lamL in (3.0, 6.0, 12.0, 45.0):
        k = 2 * math.pi / (lamL * pc.L)
        om = math.sqrt(G * k)
        Fp = _lat_rhs_harm(pc, om, th)
        Ff = _lat_rhs_harm(pf, om, th)
        Fd = Fp - Ff
        pb = cpt.DiffractionProblem(body=body, omega=om, wave_direction=th,
                                    rho=pc.rho, g=G)
        r_ = solver.solve(pb)
        fk = froude_krylov_force(pb)
        cfk = np.array([complex(fk[x]) for x in ("Sway", "Roll", "Yaw")])
        cdf = np.array([complex(r_.forces[x]) for x in ("Sway", "Roll",
                                                        "Yaw")])
        row = dict(lamL=lamL, fk_plant=Ff, fk_cap=cfk, dif_plant=Fd,
                   dif_cap=cdf)
        rows.append(row)
        out(f"    lam/L {lamL:4.1f}: FK sway plant {abs(Ff[0]):7.1f} Cap "
            f"{abs(cfk[0]):7.1f}, lever plant {(Ff[2] / Ff[0]).real:+.4f} "
            f"Cap {(cfk[2] / cfk[0]).real:+.4f} | diffraction sway plant "
            f"{abs(Fd[0]):7.1f} Cap {abs(cdf[0]):7.1f}, lever plant "
            f"{(Fd[2] / Fd[0]).real:+.4f} Cap {(cdf[2] / cdf[0]).real:+.4f}")
    return dict(levers=dict(strip_phi0=A26 / A22, strip_lid=L26 / L22,
                            cap_inf=lev["inf"][1] / lev["inf"][0],
                            cap_0=lev["0.0"][1] / lev["0.0"][0]), rows=rows)


def _energy_consistent(dt, T, start):
    """9: P1a's start (heave, trim offsets, heel) with the candidate's own
    energy: lateral_energy_run's measure (the flow-state added mass kb.ma,
    the lateral matrix lat.M3, the hydrostatic and GZ potentials)."""
    import inspect
    src = inspect.getsource(lateral_energy_run).replace(
        "def lateral_energy_run(dt, T, start):",
        "def _run(dt, T, start):").replace(
        "    s[3], s[7], s[11] = start\n",
        "    s[2] += start[0]\n    s[4] += start[1]\n    s[3] = start[2]\n")
    ns = dict(globals())
    exec(src, ns)
    return ns["_run"](dt, T, start)


def _steer_run(sw, sea=None, T_end=6.0, dt=0.02, patch=None):
    """P0's twin open loop (Fn_L 1.2 start, thrust and nozzle sinusoids in
    Froude time) on scarab195@as1 with switches sw: max |heel|, max |yaw
    rate|, max |sway|, the time reached (stops past 80 deg heel)."""
    from hydro import hulls
    from sim import physics
    from sim import planing_anyspeed as PA
    PA.PRESETS["as1"] = PA.resolve("as1", sw)
    p = physics.gz_twin(hulls.get(AS).plant(calm() if sea is None else sea,
                                            dt=dt), record=False)
    if patch:
        patch(p)
    L = float(p.L)
    U, T = math.sqrt(G * L), math.sqrt(L / G)
    s = p.initial_state(1.2 * U)
    hm, rm, vm, t = 0.0, 0.0, 0.0, 0.0
    for i in range(int(round(T_end / dt))):
        t = i * dt
        tau = t / T
        s = p.step(s, t, p.prop.t_max * (0.55 + 0.35 * math.sin(0.3 * tau)),
                   p.rudder.max * 0.6 * math.sin(0.2 * tau), dt)
        hm = max(hm, abs(math.degrees(s[3])))
        rm = max(rm, abs(math.degrees(s[11])))
        vm = max(vm, abs(s[7]))
        if hm > 80.0:
            break
    return dict(heel=hm, r=rm, v=vm, t=t + dt, kn=s[6] / KN)


def _zero_coeffs(a24=False, heel=False):
    def patch(p):
        orig = p._lat_coeffs

        def f(d, phi):
            (a22, b24, a44), (d22, d24, d44), (p22, p24, p44) = orig(d, phi)
            z = np.zeros_like(b24)
            if a24:
                b24, d24, p24 = z, z, z
            if heel:
                p22, p24, p44 = z, z, z
            return (a22, b24, a44), (d22, d24, d44), (p22, p24, p44)
        p._lat_coeffs = f
    return patch


def part_diag():
    from sim.planing_anyspeed import AS1_P1A, P1B_CANDIDATE
    from sim.wavefield import SeaState
    from studies import anyspeed_p0_check as P0
    out("P1b (diag): diagnoses of the failures of 5, 9, 10 (recorded, "
        "written after the final run; no gates)")
    res = dict(yaw=_diag_yaw())
    out("  (diag 9) P1a's energy start (z +0.01 m, trim +0.003 rad, heel "
        "0.05 rad) on the candidate, the candidate's own energy measure:")
    en = []
    for dt in (0.002, 0.02):
        r = _energy_consistent(dt, 100.0, (0.01, 0.003, 0.05))
        en.append(r)
        out(f"    dt {dt}: drift {100 * r['drift_per_100s']:+.4f} % per "
            f"100 s, max dev {100 * r['max_dev']:.4f} %")
    res["energy"] = en
    out("  (diag 10) P0's twin open loop (Fn_L 1.2, nozzle sinusoid "
        "0.6 x max) on scarab195@as1, first 6 s: max heel / yaw rate / "
        "sway, by switch set")
    wave = P0.scaled_sea_arrays(SeaState(0.6, 4.0, n_freq=20, n_dir=5,
                                         seed=4), 1.0)
    st = []
    for lab, sw, sea, patch in (
            ("P1a preset, task-like sea", AS1_P1A, wave, None),
            ("candidate, task-like sea", P1B_CANDIDATE, wave, None),
            ("P1a preset, calm", AS1_P1A, None, None),
            ("+lateral_phi0, calm", AS1_P1A + ("lateral_phi0",), None, None),
            ("+lateral_phi0, calm, dt 0.005", AS1_P1A + ("lateral_phi0",),
             None, "dt"),
            ("+lateral_phi0, calm, a24 = 0", AS1_P1A + ("lateral_phi0",),
             None, _zero_coeffs(a24=True)),
            ("+lateral_phi0, calm, no heel derivatives",
             AS1_P1A + ("lateral_phi0",), None, _zero_coeffs(heel=True)),
            ("+lateral_phi0, calm, a24 = 0 and no heel derivatives",
             AS1_P1A + ("lateral_phi0",), None,
             _zero_coeffs(a24=True, heel=True)),
            ("+fk_exact+orbital+wagner_entry (no lateral_phi0), calm",
             AS1_P1A + ("fk_exact", "orbital", "wagner_entry"), None,
             None)):
        if patch == "dt":
            r = _steer_run(sw, sea, dt=0.005)
        else:
            r = _steer_run(sw, sea, patch=patch)
        r["label"] = lab
        st.append(r)
        out(f"    {lab:55s}: max heel {r['heel']:5.1f} deg, max yaw rate "
            f"{r['r']:5.1f} deg/s, max sway {r['v']:4.2f} m/s, "
            f"{r['t']:.2f} s, {r['kn']:.1f} kn")
    res["steer"] = st
    out("  (diag 11, P1a's 13 on the candidate's own added mass) at rest, "
        "surge held, dt 0.5 ms, 2 s from +5 mm / +0.1 deg: dynamic "
        "horizontal force (Xp - Xh) vs -d/dt sum kb.ma V tau dx")
    from sim.planing_vessel import SP
    from studies import anyspeed_p1a_check as P1A_
    swc = AS1_P1A + ("wagner_entry",)
    q = P1A_.plant(AS, sw=swc, dt=0.0005, captive_u=0.0)
    q.c_dc = 0.0
    v, _ = P1A_.eq_rest(P1A_.plant(AS, sw=swc, captive_u=0.0))
    tab = q._tab()
    s = np.zeros(14)
    s[2], s[4] = v[0] + 0.005, v[1] + np.radians(0.1) / SP
    Gs, Xd, cells = [], [], []
    for i in range(4001):
        _, aux = q.deriv(s, i * q.dt)
        tau = q.kp + SP * (q.Th_rest + s[4])
        Gs.append((aux["kb"].ma * aux["V"] * tau).sum() * q.dx)
        Xd.append(aux["Xp"] - aux["Xh"])
        r_ = aux["d"] / (q.yc * tab.tb)
        cells.append(np.where(r_ > 1.0, np.searchsorted(
            tab.xi, tab._xi_of(r_, 0.0)[0], side="right"), -1))
        s = q.step(s, i * q.dt, 0.0, 0.0)
        s[6] = 0.0
    Gs, Xd, cells = np.asarray(Gs), np.asarray(Xd), np.asarray(cells)
    dG = (Gs[2:] - Gs[:-2]) / (2 * q.dt)
    rs = np.abs(Xd[1:-1] + dG) / np.abs(Xd).max()
    cr = np.array([np.any(cells[k - 1] != cells[k + 1])
                   for k in range(1, len(Gs) - 1)])
    out(f"    largest {rs.max():.1e} (at the {int(cr.sum())} samples whose "
        f"central difference spans a cell boundary of the m_0 table, where "
        f"dm_0/dd jumps: {rs[cr].max():.1e}; elsewhere {rs[~cr].max():.1e})")
    res["hx"] = dict(largest=float(rs.max()), at_cells=float(rs[cr].max()),
                     elsewhere=float(rs[~cr].max()))
    out("  (diag 11, R3) 600 s at zero thrust from heave +5 cm (P1a's "
        "zero_thrust_run), by switch set: speed at 600 s, largest speed, "
        "distance")
    from studies import anyspeed_p1a_check as P1A
    from sim import planing_anyspeed as PA
    rr = []
    for lab, extra in (("P1a preset", ()), ("+wagner_entry",
                                            ("wagner_entry",)),
                       ("+lateral_phi0", ("lateral_phi0",)),
                       ("+fk_exact+orbital", ("fk_exact", "orbital")),
                       ("candidate", ("fk_exact", "orbital", "wagner_entry",
                                      "lateral_phi0"))):
        sw = AS1_P1A + extra
        PA.PRESETS["as1"] = PA.resolve("as1", sw)
        p0 = P1A.plant(AS, sw=sw)
        v, _ = P1A.eq_rest(P1A.plant(AS, sw=sw, captive_u=0.0))
        s = np.zeros(14)
        s[2] = v[0] + 0.05
        r = P1A.zero_thrust_run(p0, s)
        rr.append(dict(label=lab, u_end=r["u_end"], u_max=r["u_max"],
                       x_end=r["x_end"]))
        out(f"    {lab:20s}: u(600 s) {r['u_end']:+.2e} m/s, max "
            f"{r['u_max']:.2e} m/s, distance {r['x_end']:+.3f} m")
    res["rest"] = rr
    dump("diag.json", res)
    flush("diag.txt")


# ================================================================ fkconv
def part_fkconv():
    """3 (diagnosis, recorded, no gate; written after the final run): the
    plant's zero-speed longitudinal wave force in short head waves against
    the continuum integral of the same section integrals over the hull
    (200 000 points), as the station count grows (scarab195's as1 hull with
    n_stations 48 ... 768); and the centre of buoyancy of the plant's strips
    against Capytaine's mesh of the same geometry (the long-wave yaw)."""
    from hydro import hulls
    from hydro.hull_geometry import stations_of
    from hydro.section2d import section_fk
    from sim import physics
    from sim.planing_anyspeed import AS1_P1A, _vsec
    out("P1b (fkconv)  3: station-count convergence of the zero-speed "
        "longitudinal wave force; centre of buoyancy, strips vs mesh")
    sw = AS1_P1A + ("fk_exact",)
    pcs = {}
    for n in (48, 96, 192, 384, 768):
        h = hulls.get("scarab195")
        if n != 48:
            h = copy.deepcopy(h)
            h.n_stations = n
            h.name = f"scarab195_n{n}"
        ha = (hulls.get(AS) if n == 48
              else physics.variant(h, "as1", name=h.name))
        pcs[n] = plant(ha, sw=sw, captive_u=0.0)
    p48 = pcs[48]
    h = p48.hull
    g = stations_of(h)
    rg = p48.rho * G
    rows = []
    for lamL, thd in ((8 * p48.dx / p48.L, 0), (8 * p48.dx / p48.L, 60),
                      (0.5, 0)):
        k = 2 * math.pi / (lamL * p48.L)
        om = math.sqrt(G * k)
        th = math.radians(thd)
        kap, lam = k * math.cos(th), k * math.sin(th)
        N = 200000
        xs = (np.arange(N) + 0.5) * h.L / N
        xb = xs - h.lcg
        d = -(p48.Z_rest + (-h.vcg + g.keel(xs)[0]))
        E = np.array(section_fk(k, lam, d, g.chine(xs), p48.tanb)[0])
        Xc = -(rg * (1j * kap) * E * np.exp(1j * kap * xb)).sum() * h.L / N
        errs = {}
        for n, pc in pcs.items():
            s = np.zeros(14)
            s[4] = -pc.Th_rest
            fp = plant_lin_fk(pc, s, om, th, a=1e-6)
            errs[n] = float(abs(fp[0] - Xc) / abs(Xc))
        rows.append(dict(lamL=lamL, head=thd, continuum=abs(Xc), err=errs))
        out(f"    lam/L {lamL:.3f}, heading {thd:3d} deg: |X| continuum "
            f"{abs(Xc):8.1f} N/m; plant vs continuum by station count: "
            + ", ".join(f"{n} {100 * e:.2f} %" for n, e in errs.items()))
    # check 3's comparison itself (rest attitude, the same mesh density) at
    # 48 and 192 stations, on the wavelengths where it failed
    lam0 = 8 * p48.dx
    h_min = lam0 / 40.0
    nx = int(math.ceil(p48.L / h_min))
    nb = int(math.ceil(1.1 * float(p48.yc.max()) / math.cos(math.atan(
        p48.tanb)) / h_min))
    ns = int(math.ceil(0.8 / h_min))
    heads = [0, 30, 60, 90, 120, 150, 180]
    f3 = []
    for n in (48, 192):
        pc = pcs[n]
        b = capytaine_body(pc, nx, nb, ns)
        cog = (0.0, 0.0, pc.Z_rest)
        s = np.zeros(14)
        for lw in (lam0, 1.5 * lam0, 0.5 * p48.L):
            k = 2 * math.pi / lw
            om = math.sqrt(G * k)
            Fp = np.array([plant_lin_fk(pc, s, om, math.radians(h))
                           for h in heads])
            Fc = np.array([cap_fk_q(b, om, math.radians(h), pc.rho, cog)
                           for h in heads])
            sc = np.maximum(np.abs(Fc).max(axis=0), 1e-300)
            e_ = (np.abs(Fp - Fc) / sc).max(axis=0)
            f3.append(dict(n=n, lam=lw, err=e_))
            out(f"    check 3 at {n:3d} stations, lam {lw:5.2f} m: max over "
                f"headings |plant - Cap| / max|Cap|, X Y Z K M N: "
                + " ".join(f"{100 * v:5.2f}%" for v in e_))
    # where the remaining difference sits: head seas at the rest attitude,
    # the plant (48, 192 stations), the continuum integral of the same
    # section integrals over the exact geometry (hydro/hull_geometry) and
    # Capytaine's mesh -- magnitude, and the phase as a longitudinal shift
    # of the force (phase / k); and the volume and waterplane centroids of
    # the four descriptions of the same hull
    from sim.planing_vessel import SP
    th = p48.Th_rest
    b48 = capytaine_body(p48, nx, nb, ns)
    cog = (0.0, 0.0, p48.Z_rest)
    N = 400000
    xs = (np.arange(N) + 0.5) * h.L / N
    xb = xs - h.lcg
    dxc = h.L / N
    dc = -(p48.Z_rest + (-h.vcg + g.keel(xs)[0]) + SP * xb * th)
    ycc = g.chine(xs)
    ph = []
    for lw in (lam0, 1.5 * lam0, 0.5 * p48.L):
        k = 2 * math.pi / lw
        om = math.sqrt(G * k)
        Fc = cap_fk_q(b48, om, 0.0, p48.rho, cog)[0]
        E = np.array(section_fk(k, 0.0, dc, ycc, p48.tanb)[0])
        Xc = -(rg * (1j * k) * E * np.exp(1j * k * xb)).sum() * dxc
        Xp = {n: plant_lin_fk(pcs[n], np.zeros(14), om, 0.0, a=1e-6)[0]
              for n in (48, 192)}
        row = dict(lam=lw, cap=abs(Fc), cont=abs(Xc),
                   shift_cont=float(np.angle(Xc / Fc) / k),
                   shift_48=float(np.angle(Xp[48] / Fc) / k),
                   shift_192=float(np.angle(Xp[192] / Fc) / k),
                   shift_48_cont=float(np.angle(Xp[48] / Xc) / k))
        ph.append(row)
        out(f"    head seas, rest attitude, lam {lw:5.2f} m: |X| continuum / "
            f"Capytaine - 1 {100 * (abs(Xc) / abs(Fc) - 1):+.2f} %, plant 48 "
            f"{100 * (abs(Xp[48]) / abs(Fc) - 1):+.2f} %; phase as a shift "
            f"against Capytaine: continuum {1000 * row['shift_cont']:+.1f} "
            f"mm, plant 48 {1000 * row['shift_48']:+.1f} mm, plant 192 "
            f"{1000 * row['shift_192']:+.1f} mm; plant 48 against the "
            f"continuum {1000 * row['shift_48_cont']:+.1f} mm")
    Ac = _vsec(dc, ycc, p48.tanb)[0]
    Bc = np.where(dc > 0, 2 * np.minimum(dc / p48.tanb, ycc), 0.0)
    geo = dict(continuum=(float(Ac.sum() * dxc), float((Ac * xb).sum()
                                                       / Ac.sum()),
                          float(Bc.sum() * dxc), float((Bc * xb).sum()
                                                       / Bc.sum())))
    for n in (48, 192):
        pc = pcs[n]
        s = np.zeros(14)
        _, aux = pc.deriv(s, 0.0)
        d_ = aux["d"]
        A_ = _vsec(d_, pc.yc, pc.tanb)[0]
        B_ = np.where(d_ > 0, 2 * np.minimum(d_ / pc.tanb, pc.yc), 0.0)
        geo[f"strips {n}"] = (float(A_.sum() * pc.dx), float(
            (A_ * pc.x).sum() / A_.sum()), float(B_.sum() * pc.dx),
            float((B_ * pc.x).sum() / B_.sum()))
    geo["mesh"] = (float(b48.volume), float(b48.center_of_buoyancy[0]),
                   float(b48.waterplane_area),
                   float(b48.waterplane_center[0]))
    for kk, (V_, xB_, Aw_, xF_) in geo.items():
        out(f"    {kk:10s}: volume {V_:.5f} m3, centre of buoyancy "
            f"{1000 * xB_:+.2f} mm, waterplane {Aw_:.5f} m2, its centre "
            f"{1000 * xF_:+.2f} mm (from the CG; rest attitude)")
    # centre of buoyancy, strips vs Capytaine's mesh
    cob = []
    for key, nb, ns in ((AS, 53, 36), ("planing_b@as1", 42, 30)):
        pc = plant(key, captive_u=0.0)
        _, aux = pc.deriv(np.zeros(14), 0.0)
        A = _vsec(aux["d"], pc.yc, pc.tanb)[0]
        V = float(A.sum() * pc.dx)
        xB = float((A * pc.x).sum() * pc.dx / V)
        b = capytaine_body(pc, 241, nb, ns)
        vm = float(b.volume)
        xm = float(b.center_of_buoyancy[0])
        cob.append(dict(hull=key, V_strips=V, xB_strips=xB, V_mesh=vm,
                        xB_mesh=xm))
        out(f"    {key}: strips volume {V:.5f} m3, centre of buoyancy "
            f"{1000 * xB:+.2f} mm from the CG; mesh {vm:.5f} m3, "
            f"{1000 * xm:+.2f} mm")
    dump("fkconv.json", dict(rows=rows, cob=cob, check3=f3, phase=ph,
                             geometry=geo))
    flush("fkconv.txt")


# ================================================================ golden
def golden_job(arg):
    """The golden runs on PlaningVesselAS with the switch set arg[0] (P1b
    review: the part ran AR.golden_runs() with its default twin -- the
    Mission's v0 plant, learn/repro/task.HULL -- so it compared v0's runs;
    the candidate's file in GOLDEN_SETS was re-run by anyspeed_regress_as,
    which does build the as plant)."""
    from hydro import hulls
    from sim.planing_anyspeed import PlaningVesselAS
    from studies import anyspeed_regress as AR
    from studies.anyspeed_regress_as import ON_EXTRA
    sw = tuple(arg[0])
    h = hulls.get(AS)
    return AR.golden_runs(lambda p: PlaningVesselAS(
        h, p.sea, dt=p.dt, record=True, test_switches=sw), extra=ON_EXTRA)


def part_golden():
    from studies.parallel_runner import run_parallel
    out("P1b (golden)  13: the candidate's golden runs in two processes")
    res = run_parallel(golden_job, [(cand(), 0), (cand(), 1)],
                       gb_per_job=0.4, reserve_gb=3.0)
    a_, b_ = res
    same = set(a_) == set(b_) and all(np.array_equal(np.asarray(a_[k]),
                                                     np.asarray(b_[k]))
                                      for k in a_)
    check("golden", "13: the candidate's golden runs identical in two "
          "processes", same, f"{len(a_)} arrays")
    if WRITE_GOLDEN:
        P1B_CANDIDATE = cand()
        tag = "+".join(P1B_CANDIDATE)
        path = os.path.join(OUT, f"golden_as_{tag}.npz")
        if os.path.exists(path):
            raise SystemExit(f"{path} exists; not overwritten")
        np.savez(path, **{k: np.asarray(v) for k, v in a_.items()})
        out(f"    written {path}")
    flush("golden.txt")


WRITE_GOLDEN = False


# ============================================================== steptime
def part_steptime():
    from studies import anyspeed_p0_check as P0
    use_candidate()
    P0.OUT = OUT
    P0.STAMP_LINE = stamp_line()
    P0.part_steptime()
    for r in P0.RESULTS.get("steptime", []):
        CHECKS.append(dict(r, part="steptime"))


# ===================================================================== s1
def s1_job(arg):
    part, out_dir, sw = arg
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"part_{part}_stdout.log"), "w") as log:
        rc = subprocess.run(
            [sys.executable, "-m", "studies.anyspeed_p0_s1rerun", "--part",
             part, "--switches", ",".join(sw), "--out", out_dir],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    return part, rc, time.time() - t0


def _sub(arg):
    name, out_dir, cnd = arg
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    extra = []
    if name.startswith("stand:"):
        name, st_ = name.split(":")
        extra = ["--stand", st_]
        tag = f"{name}_{st_}"
    else:
        tag = name
    with open(os.path.join(out_dir, f"p1b_{tag}_stdout.log"), "w") as log:
        rc = subprocess.run([sys.executable, "-m",
                             "studies.anyspeed_p1b_check", "--parts", name,
                             "--out", out_dir, "--cand", cnd] + extra,
                            cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    return name, rc, time.time() - t0


PARTS = ("sec2d", "fk2d", "fk3d", "orb", "latexc", "lat3d", "sym", "wag",
         "energy", "twin", "stand", "fk11", "v1", "golden", "steptime",
         "diag", "fkconv")


def main():
    global OUT, STAMP, WRITE_GOLDEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="")
    ap.add_argument("--jobs", default="", help="parts, one process each")
    ap.add_argument("--s1", default="", help="S1 parts, one process each")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--write-golden", action="store_true")
    ap.add_argument("--cand", default="p1b", choices=sorted(CANDS),
                    help="switch set: p1b (the candidate), w / f / fo / l "
                    "(P1a + wagner_entry / fk_exact / fk_exact + orbital / "
                    "lateral_phi0)")
    ap.add_argument("--stand", default="",
                    help="part stand: only this one of rest, astern, "
                    "e2speed, nr, hullb")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    global CAND, STAND_ONLY
    CAND = CANDS[a.cand]
    STAND_ONLY = (a.stand,) if a.stand else None
    OUT = os.path.abspath(a.out)
    WRITE_GOLDEN = a.write_golden
    if a.jobs or a.s1:
        from studies.parallel_runner import run_parallel
        from hydro import section2d as S
        for b in (20.0, 16.0):          # the tables before the workers
            S.table_for(b, workers=4)
        jobs = [(x, OUT, a.cand) for x in a.jobs.split(",") if x]
        t0 = time.time()
        res = []
        if jobs:
            res += run_parallel(_sub, jobs, gb_per_job=0.5, reserve_gb=3.0)
        s1j = [(x, os.path.join(OUT, "s1_rerun"), cand())
               for x in a.s1.split(",") if x]
        if s1j:
            res += run_parallel(s1_job, s1j, gb_per_job=0.4, reserve_gb=3.0)
        for r in res:
            print(f"  {r}", flush=True)
        print(f"all done in {time.time() - t0:.0f} s wall", flush=True)
        return
    STAMP = stamp_all()
    t0 = time.time()
    for part in [x for x in a.parts.split(",") if x]:
        if part not in PARTS:
            raise SystemExit(f"unknown part {part}")
        globals()[f"part_{part}"]()
    tag = "_".join(x for x in a.parts.split(",") if x) + (
        f"_{a.stand}" if a.stand else "")
    dump(f"checks_{tag}.json", CHECKS)
    nf = sum(not c["ok"] for c in CHECKS)
    print(f"{len(CHECKS)} checks, {nf} failed, {time.time() - t0:.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
