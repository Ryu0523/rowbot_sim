#!/usr/bin/env python3
"""
Verification of step P1a of the any-speed plant (sim/planing_anyspeed.py,
switches 'transom' and 'verlet'; learn/meta/ACCEPTANCE_2026-10-06.md; the
checks and their bounds were registered before any run in
learn/meta/ANYSPEED_PLAN_2026-10-05.md, section '进展 / P1a').

The candidate sim/planing_anyspeed.P1A_CANDIDATE (transom, normal_rate and
verlet on the pre-P1a preset) was verified here before it was adopted; since
the P1a review it IS the as1 preset (transom and normal_rate as a recorded
exception to two gates: sim/planing_anyspeed.PRESETS). use_candidate() puts
it into PRESETS['as1'] in this process (a no-op now) and the 'before'
plants run on the explicit constants AS1_PRE_P1A / AS1_P1A_VERLET.

Every .txt output starts with a stamp line and every .json carries '_stamp':
the as1 switches, the physics sha1 of scarab195@as1 and planing_b@as1, the
as1 code sha1 and this script's sha1 (P1a review: outputs traceable to the
code that made them).

The review round (2026-10-06) added parts, registered in the plan doc
before they ran: nr (normal_rate on a curved keel: the M2 identity, the
resistance change, the moments of the transom switch), r2 (the
thrust-parametrised calm equilibrium: root count of T - R and a slow
free-surge thrust ramp), e2speed (the integrator on the plant's linearised
heave/pitch at speed), hullb (S1's standing checks on planing_b with P1's
sweep criterion), p6attr (Fridsma P6 by switch); and extended m2 (the wave
moment and station convergence), fk (FK1 re-registered against a_b(u) x
textbook), astern (the coast-down re-registered by period-averaged |u|),
sens (the wave force, the implied base drag, the static-draught form) and
golden (compared with the stored reference; written only with --write).

Parts (outputs in studies/_cache/anyspeed/p1a/, or --out):
  m2      code = derivation (ACCEPTANCE M1, M2): the section closed forms
          (_vsec, _vsec_q) against Gauss-Legendre quadrature; dq/dd = A and
          ds2/dd = m1; the slice identity dQ/dx = rho g [A (eta_x - tau) +
          (d - d_ch)+^2 yc'] along the hull (the taper term v0 missed); the
          TELESCOPED hull force and moment against an independent pressure
          integration over a faceted hull surface (triangles clipped at the
          waterplane, edge-midpoint rule: exact for the linear calm-water
          pressure and the quadratic moment integrand), and the closed
          faceted body (hull + wetted transom face) giving zero force and
          moment; in waves, the faceted integral as reference for the
          plant's station sum (discretisation reported)
  rest    R3 / M3: forces at the Newton-polished hydrostatic equilibrium;
          the new force and moment at random calm attitudes with the face
          wet (scarab195, planing_b); 600 s zero-thrust runs from four
          starts (GZ plant, free surge), candidate vs the current as1
          preset and v0
  fk      the longitudinal wave force at 0.5-6 kn, head waves, lambda/L
          0.5-6 (hull held at its calm captive equilibrium, towed): first
          harmonic of the plant's force against (1) the textbook closed-body
          long-wave Froude-Krylov force on a 40 000-point grid of the same
          geometry, transom forced wet; (2) the same with the default
          transom state; (3) the exact linear FK with depth decay (closed
          form on the same sections), checked against Capytaine's FK of the
          probe mesh (studies/_cache/anyspeed/bem_probe_lid_24_64.npz)
  energy  M3 / E2: dissipation off at rest (cross-flow drag and roll
          damping zero; friction and momentum flux are zero there), the
          plant's exact energy over 100 s (and 600 s) at dt 0.002 and
          0.02, Heun (verlet off) vs the new scheme; the one-step
          amplification of the scheme's own code on a linear oscillator
  astern  -3..0 kn: captive resistance and its sign, zero-thrust coast-down
          from -3 kn, reverse thrust, astern in the task sea
  conv    M5 (recorded): calm captive equilibria at 6-20 kn with 48-768
          stations against 3072, P1's registered bounds
  sens    G3: the ventilation threshold at the ends and middle of its
          scatter (2.0, 3.25, 4.5) and the 'step' form: calm branch by
          Newton continuation up and down 1-25 kn; randomisation demo
  v1      V1 (recorded): Savitsky 20-40 kn; Fridsma 1969 calm and regular
          waves (P6), Fridsma 1971 irregular (P7); candidate and v0
  twin    M3 / G4: Froude twins (lam 1.5, 0.6) of the candidate: calm roots
          and 600 open-loop steps, nondimensional, within 1e-10
  golden  N2: anyspeed_regress golden runs on the candidate in two
          processes, identical to each other AND to the stored reference
          studies/_cache/anyspeed/p1a/golden_as_<candidate>.npz (any
          difference fails); the reference is written only with --write
  steptime  E1
  nr      M2 for normal_rate on a curved keel: in steady calm flow the
          strips' horizontal momentum force (v0's -sum dyn tau dx without
          the cross-flow drag, plus Xn) == the transom end term
          -u m_aT V_T tau_T, ungated, by station count (planing_b and
          scarab195); the gating's share; planing_b's calm resistance
          with and without the switch; scarab195's Mh and the missing
          dynamic z-lever moment (P1b)
  r2      R2 with thrust as the parameter: the captive R(u) on a 0.25 kn
          grid against the jet's T(u; c), roots of T - R counted on a
          c grid of 0.002 (a_b(C_V) as is, and frozen at 1 and 0.5 as a
          diagnostic); one slow free-surge thrust ramp up and down (open
          loop)
  e2speed E2 standing check: the one-step map of the verlet scheme (and
          Heun) on the plant's linearised heave/pitch at 0-40 kn against
          the exact exp(lambda h)
  hullb   S1's standing checks on planing_b on the preset (census, sweep,
          decay, D55: studies/anyspeed_p0_check.hullb_s1_job), judged by
          P0's criteria and by P1's registered sweep criterion
  p6attr  V1 attribution: Fridsma 1969 regular waves (P6) for v0, the
          preset before P1a, +transom, +transom+normal_rate, +verlet only
          and the candidate
  s1      S1's standing checks re-run on the candidate (parts a b c g f d of
          studies/anyspeed_s1_check.py, part c's sweeps to 40 kn), one
          process per part (studies/parallel_runner, 3 GB kept free)

    .venv/Scripts/python.exe -m studies.anyspeed_p1a_check --parts m2,rest
    .venv/Scripts/python.exe -m studies.anyspeed_p1a_check --jobs \\
        rest,fk,energy,astern,conv,sens,v1,steptime --out <dir>
    .venv/Scripts/python.exe -m studies.anyspeed_p1a_check --s1 a,b,c,g,f,d
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

OUT = os.path.join(HERE, "_cache", "anyspeed", "p1a")
KN = 0.514444
G = 9.81
LINES = []
CHECKS = []
AS = "scarab195@as1"


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


STAMP = None            # set in main (stamp_all)


def stamp_all():
    """What every output carries: the as1 preset's switches, the physics
    sha1 of the plants these checks build on it (code as imported, hull
    manifest, switches), the as1 code sha1 and this script's sha1."""
    import hashlib
    from hydro import hulls
    from sim import physics
    from sim.planing_anyspeed import PRESETS
    st = dict(switches=list(PRESETS["as1"]),
              code_sha1_as1=physics.code_sha1("as1"))
    for key in (AS, "planing_b@as1"):
        st[key] = physics.stamp_for(hulls.get(key), "as1",
                                    PRESETS["as1"])["physics_sha1"]
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


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    if STAMP:
        obj = (dict(obj, _stamp=STAMP) if isinstance(obj, dict)
               else dict(_stamp=STAMP, rows=obj))
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda v: (
            v.tolist() if isinstance(v, np.ndarray) else
            v.item() if isinstance(v, np.generic) else str(v)))


def check(part, name, ok, detail=""):
    ok = bool(ok)
    CHECKS.append(dict(part=part, name=name, ok=ok, detail=str(detail)))
    out(f"  [{'ok' if ok else 'FAIL'}] {part}: {name}"
        + (f"  -- {detail}" if detail != "" else ""))
    return ok


def use_candidate():
    """The candidate preset for 'NAME@as1' plants built in this process (a
    no-op since its adoption). Returns the as1 preset before P1a."""
    from sim import planing_anyspeed as PA
    PA.PRESETS["as1"] = PA.resolve("as1", PA.P1A_CANDIDATE)
    return tuple(PA.AS1_PRE_P1A)


# the as1 presets before P1a's adoption (sim/planing_anyspeed): before P1a,
# and it plus verlet (published between verlet's and transom's adoption)
PRE_P1A = ("dDt_full", "chine_exact_all", "kpp_strip")
PRE_VERLET = ("dDt_full", "chine_exact_all", "kpp_strip", "verlet")


def calm():
    from sim.test_vessel import Monochromatic
    return Monochromatic(1.0, 0.0)


def _without(sw, name):
    """The switch set sw without switch name and without every switch that
    then lacks a switch it needs (P1b review: wagner_entry needs
    normal_rate, so a set 'without normal_rate' drops it too)."""
    from sim.planing_anyspeed import SWITCHES
    out_ = [x for x in sw if x != name]
    while True:
        bad = [x for x in out_ if set(SWITCHES[x].get("needs", ()))
               - set(out_)]
        if not bad:
            return tuple(out_)
        out_ = [x for x in out_ if x not in bad]


def plant(hull=AS, sea=None, sw=None, captive_u=None, dt=0.02, gz=True):
    """An as plant for hull (key or object) on the candidate preset, or on
    switches sw (tests); v0 for a plain key."""
    from hydro import hulls
    from sim import physics
    from sim.planing_anyspeed import PlaningVesselAS, PlaningVesselASLin
    h = hulls.get(hull) if isinstance(hull, str) else hull
    sea = calm() if sea is None else sea
    if physics.version(h) == "v0":
        from sim.planing_vessel_gz import PlaningVesselGZ
        return PlaningVesselGZ(h, sea, dt=dt, captive_u=captive_u)
    cls = PlaningVesselAS if gz else PlaningVesselASLin
    return cls(h, sea, dt=dt, captive_u=captive_u, test_switches=sw)


def S1():
    from studies import anyspeed_s1_check as s1
    return s1


def eq_rest(p):
    """The hydrostatic equilibrium (z, th) at u = 0, Newton-polished on the
    plant's own captive equations, and its residual."""
    s1 = S1()
    v, r = s1.polish(p, 0.0, np.zeros(2), tol=1e-15, it=60)
    return v, r


# ============================================================== part m2
def _quad_section(d, yc, tb, n=40):
    """(A, m1, m2, q, s2) of the V-and-wall section by Gauss-Legendre in h,
    split at the chine height (independent of _vsec)."""
    from numpy.polynomial.legendre import leggauss
    xg, wg = leggauss(n)
    dp = max(d, 0.0)
    dch = yc * tb
    res = np.zeros(5)
    for lo, hi, wfun in ((0.0, min(dp, dch), lambda h: 2.0 * h / tb),
                         (dch, dp, lambda h: 2.0 * yc + 0.0 * h)):
        if hi <= lo:
            continue
        h = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wg * wfun(h)
        res += np.array([w.sum(), (w * h).sum(), (w * h * h).sum(),
                         (w * (dp - h)).sum(), (w * h * (dp - h)).sum()])
    return res


def facet_integrals(p, z, th, N=400, sea_eta=None, t=0.0, inc_face=False):
    """Pressure p = rho g (eta(X) - Z) integrated over a faceted surface of
    the plant's hull (the plant's idealised geometry: vertical V-and-wall
    sections at body x from hydro/hull_geometry, N + 1 stations, keel line
    zk(x) + SP x Th, the CG at height Z_rest + z): triangles clipped where
    p = 0 (linear interpolation of p along edges -- exact in calm water),
    each integrated with the edge-midpoint rule (exact for quadratics).
    Returns dict: X_hull (bottom, sides, taper, stem closure; no transom
    face), M_hull = sum z_v F_x over the same, Z_hull, and with
    inc_face the transom face's X_face, M_face. eta(X) = sea_eta(X) or 0."""
    from hydro.hull_geometry import stations_of
    from sim.planing_vessel import SP
    h = p.hull
    g = stations_of(h)
    xs = np.linspace(0.0, h.L, N + 1)
    xb = xs - h.lcg
    yc = g.chine(xs)
    zk = -h.vcg + g.keel(xs)[0]
    tb = p.tanb
    Th = p.Th_rest + th
    Zcg = p.Z_rest + z
    zkey = zk + SP * xb * Th           # keel height above the CG (vertical)
    rg = p.rho * G

    def P(i, yy, hh):
        return np.array([xb[i], yy, zkey[i] + hh])   # (X, Y, z_v)

    def pres(pt):
        e = 0.0 if sea_eta is None else float(sea_eta(pt[0], t))
        return rg * (e - (pt[2] + Zcg))

    def clip(poly):
        vals = [pres(q) for q in poly]
        outp = []
        n = len(poly)
        for k in range(n):
            a, b = poly[k], poly[(k + 1) % n]
            pa, pb = vals[k], vals[(k + 1) % n]
            if pa >= 0.0:
                outp.append(a)
            if (pa >= 0.0) != (pb >= 0.0):
                s = pa / (pa - pb)
                outp.append(a + s * (b - a))
        return outp

    acc = dict(X_hull=0.0, M_hull=0.0, Z_hull=0.0, X_face=0.0, M_face=0.0)

    def add(poly, key_x, key_m, sign_ref, axis):
        """Integrate over polygon poly (list of points), outward normal
        oriented so that n[axis] * sign_ref > 0."""
        if len(poly) < 3:
            return
        cp = clip(poly)
        if len(cp) < 3:
            return
        p0 = cp[0]
        for k in range(1, len(cp) - 1):
            a, b = cp[k], cp[k + 1]
            nA = 0.5 * np.cross(a - p0, b - p0)
            if nA[axis] * sign_ref < 0.0:
                nA = -nA
            mids = (0.5 * (p0 + a), 0.5 * (a + b), 0.5 * (b + p0))
            pv = [pres(m) for m in mids]
            ip = sum(pv) / 3.0                       # int p dS / |A|
            izp = sum(pp * m[2] for pp, m in zip(pv, mids)) / 3.0
            acc[key_x] += -ip * nA[0]
            acc[key_m] += -izp * nA[0]
            if key_x == "X_hull":
                acc["Z_hull"] += -ip * nA[2]
    for i in range(N):
        for sgn in (1.0, -1.0):
            K0, K1 = P(i, 0.0, 0.0), P(i + 1, 0.0, 0.0)
            C0 = P(i, sgn * yc[i], yc[i] * tb)
            C1 = P(i + 1, sgn * yc[i + 1], yc[i + 1] * tb)
            D0 = P(i, sgn * yc[i], h.depth)
            D1 = P(i + 1, sgn * yc[i + 1], h.depth)
            for tri in ((K0, K1, C1), (K0, C1, C0), (C0, C1, D1),
                        (C0, D1, D0)):
                add(list(tri), "X_hull", "M_hull", sgn, 1)
    # the stem closure (a face of half-width yc(L), the chine floor)
    i = N
    stem = [P(i, 0.0, 0.0), P(i, yc[i], yc[i] * tb), P(i, yc[i], h.depth),
            P(i, -yc[i], h.depth), P(i, -yc[i], yc[i] * tb)]
    add(stem, "X_hull", "M_hull", 1.0, 0)
    if inc_face:
        face = [P(0, 0.0, 0.0), P(0, yc[0], yc[0] * tb), P(0, yc[0], h.depth),
                P(0, -yc[0], h.depth), P(0, -yc[0], yc[0] * tb)]
        add(face, "X_face", "M_face", -1.0, 0)
    return acc


def part_m2():
    from hydro import hulls
    from sim import planing_anyspeed as PA
    from sim.planing_vessel import SP
    from sim.test_vessel import Monochromatic
    out("P1a (m2)  CODE = DERIVATION: closed forms, identities, faceted-hull "
        "pressure integration")
    tb = math.tan(math.radians(20.0))
    # (a) closed forms vs quadrature
    worst = 0.0
    rows = []
    for yc in (1.0126, 0.3, 1e-3):
        dch = yc * tb
        for d in (-0.1, 0.05, 0.2, dch, 0.45, 1.2):
            A, m1, m2 = PA._vsec(d, yc, tb)
            q, s2, _, _ = PA._vsec_q(d, yc, tb)
            ref = _quad_section(d, yc, tb)
            got = np.array([A, m1, m2, q, s2], float)
            sc = np.maximum(np.abs(ref), 1e-300)
            err = float(np.max(np.where(np.abs(ref) > 0, np.abs(got - ref)
                                        / sc, np.abs(got))))
            worst = max(worst, err)
            rows.append(dict(yc=yc, d=d, err=err))
    check("m2", "section closed forms (A, m1, m2, q, s2) == Gauss-Legendre "
          "quadrature, relative error <= 1e-8", worst <= 1e-8,
          f"worst {worst:.1e} over {len(rows)} (yc, d) incl. d = chine "
          f"height and d < 0")
    # (b) derivative identities (Richardson central differences)
    worst_q = worst_s = 0.0
    for yc in (1.0126, 0.3):
        dch = yc * tb
        for d in (0.07, 0.2, 0.31, 0.45, 0.9):
            if abs(d - dch) < 1e-3:
                continue

            def dd(f, x0, h=1e-4):
                d1 = (f(x0 + h) - f(x0 - h)) / (2 * h)
                d2 = (f(x0 + h / 2) - f(x0 - h / 2)) / h
                return (4 * d2 - d1) / 3
            A, m1, _ = PA._vsec(d, yc, tb)
            dq = dd(lambda x: PA._vsec_q(x, yc, tb)[0], d)
            ds = dd(lambda x: PA._vsec_q(x, yc, tb)[1], d)
            worst_q = max(worst_q, abs(dq - A) / A)
            worst_s = max(worst_s, abs(ds - m1) / m1)
    check("m2", "dq/dd = A and ds2/dd = m1 (Richardson differences), "
          "relative error <= 1e-8", max(worst_q, worst_s) <= 1e-8,
          f"dq/dd {worst_q:.1e}, ds2/dd {worst_s:.1e}")
    # (c) slice identity along the hull, calm and in a wave
    from hydro.hull_geometry import stations_of
    res_c = []
    for key in (AS, "planing_b@as1"):
        p = plant(key)
        h = p.hull
        g = stations_of(h)
        rg = p.rho * G
        for (zz, trd, amp) in ((0.0, None, 0.0), (0.04, 3.0, 0.0),
                               (-0.05, 1.0, 0.06)):
            th = 0.0 if trd is None else np.radians(trd) / SP - p.Th_rest
            Th = p.Th_rest + th
            k = 2 * np.pi / (1.3 * h.L)

            def eta(xx):
                return amp * np.cos(k * xx + 0.4)

            def eta_x(xx):
                return -amp * k * np.sin(k * xx + 0.4)

            def Q(xs):
                xb = xs - h.lcg
                d = eta(xb) - (p.Z_rest + zz + (-h.vcg + g.keel(xs)[0])
                               + SP * xb * Th)
                return rg * PA._vsec_q(d, g.chine(xs), p.tanb)[0]
            xs = np.linspace(0.01, h.L - 0.01, 997)
            xb = xs - h.lcg
            rise, kp, _ = g.keel(xs)
            d = eta(xb) - (p.Z_rest + zz + (-h.vcg + rise) + SP * xb * Th)
            yc = g.chine(xs)
            dch = yc * p.tanb
            A = PA._vsec(d, yc, p.tanb)[0]
            tau = kp + SP * Th
            form = rg * (A * (eta_x(xb) - tau)
                         + np.maximum(d - dch, 0.0) ** 2 * g.dchine(xs))
            hh = 1e-4
            num = (4 * (Q(xs + hh / 2) - Q(xs - hh / 2)) / hh
                   - (Q(xs + hh) - Q(xs - hh)) / (2 * hh)) / 3
            # away from kinks: d = 0, d = d_ch, the taper start, the keel
            # rise start, the transom run's end
            x0t = (1.0 - getattr(h, "taper", 0.35)) * h.L
            far = (np.abs(d) > 3e-3) & (np.abs(d - dch) > 3e-3)
            for xk in (x0t, h.L):
                far &= np.abs(xs - xk) > 3e-3
            if hasattr(h, "form") and h.form.get("transom_run"):
                far &= np.abs(xs - h.form["transom_run"] * h.L) > 3e-3
            if hasattr(h, "form") and h.form.get("rise_from"):
                far &= np.abs(xs - (1 - h.form["rise_from"]) * h.L) > 3e-3
            sc = np.abs(form[far]).max()
            err = float(np.abs(num[far] - form[far]).max() / sc)
            # the taper term's share (what v0's sum A tau dx misses)
            taper = rg * np.maximum(d - dch, 0.0) ** 2 * g.dchine(xs)
            res_c.append(dict(hull=key, z=zz, trim=trd, amp=amp, err=err,
                              n=int(far.sum()),
                              taper_N=float(taper.sum() * (xs[1] - xs[0]))))
    worst = max(r["err"] for r in res_c)
    check("m2", "slice identity dQ/dx = rho g [A (eta_x - tau) + (d - d_ch)+^2"
          " yc'] along the hull (scarab195, planing_b; calm trimmed and in a "
          "wave), relative to the largest |dQ/dx|, <= 1e-8", worst <= 1e-8,
          "; ".join(f"{r['hull'].split('@')[0]} z{r['z']:+.2f} a{r['amp']}: "
                    f"{r['err']:.1e} ({r['n']} pts, taper side force "
                    f"{r['taper_N']:+.1f} N)" for r in res_c))
    # (d) faceted hull: telescoped force and moment, closed body
    res_d = []
    for key in (AS, "planing_b@as1"):
        p = plant(key)
        tg = p._transom_geom()
        eqv, _ = eq_rest(p)
        for (dz, dtr) in ((0.0, 0.0), (0.05, 2.0), (-0.08, -1.5),
                          (0.25, 6.0)):
            zz = eqv[0] + dz
            th = eqv[1] + np.radians(dtr) / SP
            Th = p.Th_rest + th
            fc = facet_integrals(p, zz, th, N=300, inc_face=True)
            tr = p._transom_kin(0.0, 0.0, 0.0, zz, Th, 0.0, 0.0)
            ab = p._a_b(0.0)
            d = -(p.Z_rest + zz + p.zk + SP * p.x * Th)
            A = p._section(d)[0]
            tr_dry = dict(tr, hw=0.0)
            Xd, Md = p._x_hydro(d, A, np.zeros_like(d), ab, 0.0, Th, tr_dry)
            Xw, Mw = p._x_hydro(d, A, np.zeros_like(d), ab, 0.0, Th, dict(tr))
            W = p.m * G
            res_d.append(dict(
                hull=key, dz=dz, dtrim=dtr, dT=tr["dT"],
                X_facet=fc["X_hull"], X_tel=Xd / ab, M_facet=fc["M_hull"],
                M_tel=Md / ab, closed_X=(fc["X_hull"] + fc["X_face"]) / W,
                closed_M=(fc["M_hull"] + fc["M_face"]) / (W * p.L),
                face_X=fc["X_face"], plant_wet_X=Xw / W,
                plant_wet_M=Mw / (W * p.L),
                eX=abs(fc["X_hull"] - Xd / ab) / max(abs(fc["X_hull"]), 1e-9),
                eM=abs(fc["M_hull"] - Md / ab) / max(abs(fc["M_hull"]), 1e-9)))
    eX = max(r["eX"] for r in res_d if abs(r["X_facet"]) > 1e-6)
    eM = max(r["eM"] for r in res_d if abs(r["M_facet"]) > 1e-6)
    cl = max(max(abs(r["closed_X"]), abs(r["closed_M"])) for r in res_d)
    pw = max(max(abs(r["plant_wet_X"]), abs(r["plant_wet_M"]))
             for r in res_d)
    for r in res_d:
        out(f"    {r['hull']:15s} dz {r['dz']:+.2f} dtrim {r['dtrim']:+.1f}: "
            f"d_T {r['dT']:+.4f}  hull X facet {r['X_facet']:+10.4f} "
            f"telescoped {r['X_tel']:+10.4f} N | M facet {r['M_facet']:+10.4f}"
            f" telescoped {r['M_tel']:+10.4f} N m | closed body X/W "
            f"{r['closed_X']:+.1e} M/(WL) {r['closed_M']:+.1e}")
    check("m2", "faceted hull (300 stations, no transom face): x force and "
          "sum z_v F_x == the telescoped -rho g q_T and -R_T of the plant "
          "(_x_hydro, face dry), relative <= 1e-8 (exact for any faceting: "
          "divergence theorem with the transom section exact)",
          max(eX, eM) <= 1e-8, f"force {eX:.1e}, moment {eM:.1e}")
    check("m2", "closed faceted body (hull + wetted transom face): force and"
          " moment <= 1e-9 W (W L)", cl <= 1e-9, f"{cl:.1e}")
    check("m2", "CODE PATH (an identity by construction, not M2 evidence: "
          "Q_w is the same function as Q_T at h_w = d_T, so they cancel "
          "exactly): plant's new longitudinal force and moment with the "
          "face wet (calm, these attitudes, both hulls) <= 1e-9 W (W L); "
          "the M2 evidence for the transom terms is the faceted-hull "
          "integration and the slice identity above", pw <= 1e-9,
          f"{pw:.1e}")
    # (e) in a wave: faceted integral (fine) vs the plant's station sum,
    # force AND moment (sum z_v F_x); station convergence at a fixed
    # ABSOLUTE attitude (the n-station plant at s[2] = Z48 + z - Zn_rest,
    # s[4] = Th48 + th - Thn_rest)
    from hydro import hulls as _hulls
    from sim import physics as _physics
    from sim.planing_anyspeed import PlaningVesselAS
    p = plant(AS)
    eqv, _ = eq_rest(p)
    sw_t = ("dDt_full", "chine_exact_all", "kpp_strip", "transom")
    rows_w = []
    NREF = 3072

    def eta_of(q):
        def eta(X, tt):
            from sim.planing_vessel import PlaningVessel
            return PlaningVessel._waves(q, np.array([X]), np.array([0.0]),
                                        tt)[0][0]
        return eta
    hn = {n: (_hulls.get(AS) if n == 48 else _physics.variant(
        S1().hull_n(n), "as1", name=f"scarab195_n{n}")) for n in
        (48, 192, 768)}
    for lamL in (0.5, 1.0, 3.0):
        k = 2 * np.pi / (lamL * p.L)
        om = math.sqrt(G * k)
        a = 0.05
        sea = Monochromatic(om, a, theta0=np.pi)
        q = plant(AS, sea=sea, sw=sw_t)
        # two instants a quarter period apart (u = 0: the force is harmonic
        # in time to first order), so the amplitude is sqrt(X0^2 + X1^2)
        quarter = 0.5 * np.pi / om
        rws = []
        for t in (0.0, quarter):
            fr = facet_integrals(q, eqv[0], eqv[1], N=NREF, sea_eta=eta_of(q),
                                 t=t, inc_face=True)
            Xr = fr["X_hull"] + fr["X_face"]
            Mr = fr["M_hull"] + fr["M_face"]
            row = dict(lamL=lamL, t=t, X_ref=Xr, M_ref=Mr, n={})
            for n, hh in hn.items():
                qn = PlaningVesselAS(hh, sea, dt=0.02, test_switches=sw_t)
                s = np.zeros(14)
                s[2] = q.Z_rest + eqv[0] - qn.Z_rest
                s[4] = q.Th_rest + eqv[1] - qn.Th_rest
                _, aux = qn.deriv(s, t)
                row["n"][n] = dict(X=aux["Xh"], M=aux["Mh"])
            rws.append(row)
        AX = math.hypot(rws[0]["X_ref"], rws[1]["X_ref"])
        AM = math.hypot(rws[0]["M_ref"], rws[1]["M_ref"])
        for row in rws:
            row.update(AX=AX, AM=AM)
            for n, r in row["n"].items():
                r["eX"] = abs(r["X"] - row["X_ref"]) / AX
                r["eM"] = abs(r["M"] - row["M_ref"]) / AM
                r["eX_inst"] = abs(r["X"] - row["X_ref"]) / max(
                    abs(row["X_ref"]), 1e-9)
            rows_w.append(row)
            out(f"    wave lam/L {lamL} t {row['t']:.3f}: faceted closed body"
                f" ({NREF} stations) X {row['X_ref']:+9.3f} N, M "
                f"{row['M_ref']:+9.3f} N m (amplitudes {AX:.2f} N, {AM:.2f} N"
                f" m); plant, error / amplitude: " + "; ".join(
                    f"{n}: X {r['X']:+9.3f} ({r['eX']:.1e}) M {r['M']:+9.3f}"
                    f" ({r['eM']:.1e})" for n, r in row["n"].items()))
    reg = [r for r in rows_w if r["lamL"] in (1.0, 3.0)]
    e48 = max(max(r["n"][48]["eX"], r["n"][48]["eM"]) for r in reg)
    e192 = max(max(r["n"][192]["eX"], r["n"][192]["eM"]) for r in reg)
    check("m2", "(form 4', re-registered after this round's first run, "
          "review_fix/m2.txt, whose error relative to the INSTANTANEOUS "
          "value failed at a zero crossing) in a regular wave (H 0.1 m) the "
          "plant's longitudinal force AND its moment sum z_v F_x vs the "
          "faceted closed body (3072 stations), fixed absolute attitude, "
          "error relative to the amplitude: lam/L 1 and 3 within 5e-3 at 48"
          " stations and 5e-4 at 192 (lam/L 0.5 recorded)",
          e48 <= 5e-3 and e192 <= 5e-4,
          f"48: {e48:.1e}, 192: {e192:.1e}; lam/L 0.5 at 48/192/768: "
          + ", ".join(f"{max(r['n'][n]['eX'], r['n'][n]['eM']):.1e}"
                      for r in rows_w if r["lamL"] == 0.5 and r["t"] == 0.0
                      for n in (48, 192, 768)))
    dump("m2.json", dict(closed_forms=rows, slice=res_c, facet=res_d,
                         wave=rows_w))
    flush("m2.txt")


# ============================================================== part rest
def zero_thrust_run(p, s0, T=600.0):
    s = s0.copy()
    p.captive_u = None
    p._reset()
    n = int(round(T / p.dt))
    m60 = int(round(60.0 / p.dt))
    umax, zs, ts, bad = 0.0, [], [], None
    for i in range(n):
        s = p.step(s, i * p.dt, 0.0, 0.0)
        if not np.all(np.isfinite(s)):
            bad = i * p.dt
            break
        umax = max(umax, abs(s[6]))
        if i >= n - m60:
            zs.append(s[2])
            ts.append(s[4])
    return dict(u_end=float(s[6]), u_max=umax, x_end=float(s[0]),
                z_mean=float(np.mean(zs)) if zs else np.nan,
                th_mean=float(np.mean(ts)) if ts else np.nan,
                z_amp=float(0.5 * np.ptp(zs)) if zs else np.nan,
                th_amp=float(0.5 * np.ptp(ts)) if ts else np.nan,
                diverged=bad)


def part_rest():
    from sim.planing_vessel import SP
    out("P1a (rest)  R3 / M3: the vessel at rest")
    rows = []
    for key in (AS, "planing_b@as1"):
        p = plant(key, captive_u=0.0)
        v, r = eq_rest(p)
        W = p.m * G
        s = np.zeros(14)
        s[2], s[4] = v
        q = plant(key)                         # free surge, zero thrust
        ds, aux = q.deriv(s, 0.0)
        Fx = ds[6] * q.Mtot[0, 0]
        # Fz, My from the accelerations through the plant's mass matrix
        M2 = S1().mass2(q, 0.0, v[0], v[1])
        Fzm = M2 @ np.array([ds[8], ds[10]])
        rows.append(dict(hull=key, z=v[0], th=v[1], resid=r, Fx_W=Fx / W,
                         Fz_W=Fzm[0] / W, My_WL=Fzm[1] / (W * q.L),
                         Xh=aux["Xh"], Mh=aux["Mh"]))
        check("rest", f"{key}: at the Newton-polished hydrostatic equilibrium"
              f" |Fx|, |Fz| <= 1e-9 W, |My| <= 1e-9 W L",
              max(abs(Fx / W), abs(Fzm[0] / W)) <= 1e-9
              and abs(Fzm[1] / (W * q.L)) <= 1e-9,
              f"Fx/W {Fx / W:+.1e}, Fz/W {Fzm[0] / W:+.1e}, My/(WL) "
              f"{Fzm[1] / (W * q.L):+.1e} (residual {r:.1e}; new force "
              f"{aux['Xh']:+.1e} N, moment {aux['Mh']:+.1e} N m)")
        # random calm attitudes, face wet (u = 0)
        rng = np.random.default_rng(7)
        worst = 0.0
        for _ in range(200):
            zz = v[0] + rng.uniform(-0.15, 0.15)
            trd = rng.uniform(-3.0, 9.0)
            th = np.radians(trd) / SP - q.Th_rest
            s2 = np.zeros(14)
            s2[2], s2[4] = zz, th
            _, a2 = q.deriv(s2, 0.0)
            worst = max(worst, abs(a2["Xh"]) / W, abs(a2["Mh"]) / (W * q.L))
        check("rest", f"{key}: CODE PATH (identity by construction: Q_w is "
              f"Q_T's function at h_w = d_T; the exact zero at the "
              f"equilibrium above follows the same way; M2 evidence: part "
              f"m2's faceted hull and slice identity): new longitudinal "
              f"force and moment at 200 random calm attitudes (z +-15 cm, "
              f"trim -3..9 deg), face wet: <= 1e-9 W (W L)", worst <= 1e-9,
              f"{worst:.1e}")
    dump("rest_forces.json", rows)
    # check 13 (normal_rate): at u = 0 the dynamic horizontal force is
    # -d/dt sum m_a V tau dx while m_a is constant (every strip
    # chine-capped, as at rest) -- finite differences along a free
    # trajectory, cross-flow drag off. First run (registered): start +3 cm
    # heave / +0.5 deg, which UNCAPS strips; then the plant's entry-only
    # flux drops the exiting momentum and the identity does not hold (kept
    # as recorded). Second: +5 mm / +0.1 deg, every strip capped (counted).
    # Surge HELD at u = 0 (zero speed): with it free, V carries u tau and
    # -dG/dt picks up -sum m_a u' tau^2, a surge-heave added-mass coupling
    # v0's vertical DV/Dt does not have (first attempt: 6e-3 from it)
    q = plant(AS, dt=0.0005, captive_u=0.0)
    q.c_dc = 0.0
    v, _ = eq_rest(plant(AS, captive_u=0.0))
    kc = 0.5 * np.pi / q.tanb
    for lab, dz, dtr, gate in (("3 cm / 0.5 deg (first run)", 0.03, 0.5,
                                False),
                               ("5 mm / 0.1 deg, all capped", 0.005, 0.1,
                                True)):
        s = np.zeros(14)
        s[2], s[4] = v[0] + dz, v[1] + np.radians(dtr) / SP
        Gs, Xd, nunc, exitm = [], [], 0, []
        for i in range(4001):
            _, aux = q.deriv(s, i * q.dt)
            Th = q.Th_rest + s[4]
            tau = q.kp + SP * Th
            d = aux["d"]
            A_, c_, ma, dma, _ = q._section(d)
            V = aux["V"]
            Gs.append((ma * V * tau).sum() * q.dx)
            Xd.append(aux["Xp"] - aux["Xh"])
            unc = (kc * np.maximum(d, 0.0) < q.yc) & (d > 0)
            nunc = max(nunc, int(unc.sum()))
            exitm.append((np.where(V <= 0.0, dma * V * V, 0.0) * tau).sum()
                         * q.dx)
            s = q.step(s, i * q.dt, 0.0, 0.0)
            s[6] = 0.0
        Gs, Xd, exitm = np.asarray(Gs), np.asarray(Xd), np.asarray(exitm)
        dG = (Gs[2:] - Gs[:-2]) / (2 * q.dt)
        res = Xd[1:-1] + dG
        e13 = float(np.abs(res).max() / np.abs(Xd).max())
        ex = float(np.abs(exitm).max() / np.abs(Xd).max())
        txt = (f"{e13:.1e} (largest |force| {np.abs(Xd).max():.2f} N; most "
               f"uncapped strips {nunc}; exiting momentum the entry-only flux"
               f" drops, tilted: up to {ex:.1e} of it)")
        if gate:
            check("rest", "13 normal_rate: at zero speed (surge held), every"
                  " strip capped, the dynamic horizontal force == -d/dt sum "
                  "m_a V tau dx along a 2 s trajectory free in heave and "
                  "pitch (central differences, dt 0.5 ms), relative <= 1e-6",
                  e13 <= 1e-6 and nunc == 0, txt)
        else:
            out(f"    check 13 first run ({lab}; registered, FAILED: strips "
                f"uncap): {txt}")
    # 600 s zero-thrust runs
    p0 = plant(AS)
    v, _ = eq_rest(plant(AS, captive_u=0.0))
    starts = {"equilibrium": (0.0, 0.0, 0.0), "heave +5 cm": (0.05, 0.0, 0.0),
              "trim +1 deg": (0.0, 1.0, 0.0),
              "5 cm, -1 deg, roll 5 deg": (0.05, -1.0, 5.0)}
    runs = {}
    for lab, (dz, dtr, roll) in starts.items():
        s = np.zeros(14)
        s[2] = v[0] + dz
        s[4] = v[1] + np.radians(dtr) / SP
        s[3] = np.radians(roll)
        t0 = time.time()
        r = zero_thrust_run(p0, s)
        r["seconds"] = time.time() - t0
        runs[lab] = r
        ok = (r["diverged"] is None and abs(r["u_end"]) < 5e-3
              and abs(r["z_mean"] - v[0]) <= 1e-3
              and abs(np.degrees(r["th_mean"] - v[1])) <= 0.01)
        check("rest", f"600 s zero thrust from {lab}: |u(600 s)| < 5 mm/s, "
              f"mean attitude over the last 60 s within 1 mm / 0.01 deg of "
              f"the hydrostatic equilibrium", ok,
              f"u end {r['u_end']:+.2e} m/s (max {r['u_max']:.2e}), drift "
              f"{r['x_end']:+.3f} m, mean dz {r['z_mean'] - v[0]:+.1e} m, "
              f"mean dtrim {np.degrees(SP * (r['th_mean'] - v[1])):+.1e} deg,"
              f" heave amp {r['z_amp']:.1e} m, pitch amp "
              f"{np.degrees(r['th_amp']):.1e} deg")
    # without normal_rate (the first P1a run's candidate), heave +5 cm
    from sim.planing_anyspeed import P1A_CANDIDATE
    sw_nn = _without(P1A_CANDIDATE, "normal_rate")
    q = plant(AS, sw=sw_nn)
    s = np.zeros(14)
    s[2], s[4] = v[0] + 0.05, v[1]
    r = zero_thrust_run(q, s)
    runs["heave +5 cm, without normal_rate"] = r
    out(f"    without normal_rate, heave +5 cm: u(600 s) {r['u_end']:+.2e} m/s"
        f" (max {r['u_max']:.2e}), drift {r['x_end']:+.2f} m")
    # the plants before P1a's adoption, from the equilibrium, 60 s (they
    # run astern: the spurious sternward hydrostatic force, R3); the
    # published preset between verlet's and transom's adoption included
    from hydro import hulls
    for lab, q in (("v0", plant("scarab195")),
                   ("as1 before P1a", plant(AS, sw=PRE_P1A)),
                   ("as1 before adoption (+verlet)", plant(AS,
                                                         sw=PRE_VERLET))):
        s = q.initial_state(0.0)
        q.captive_u = None
        q._reset()
        marks = {}
        for i in range(int(round(60.0 / q.dt))):
            s = q.step(s, i * q.dt, 0.0, 0.0)
            tt = round((i + 1) * q.dt, 6)
            if tt in (5.0, 10.0, 30.0, 60.0):
                _, aux = q.deriv(s, tt)
                marks[tt] = (float(s[6]), float(aux["Xp"]))
        r = dict(u_end=float(s[6]), x_end=float(s[0]), marks=marks)
        runs[lab] = r
        out(f"    {lab:30s} zero thrust from rest: u "
            + ", ".join(f"{t:g} s {m[0]:+.3f} m/s" for t, m in marks.items())
            + f" ({r['u_end'] / KN:+.2f} kn at 60 s), drift "
            f"{r['x_end']:+.1f} m; pressure force Xp "
            + ", ".join(f"{t:g} s {m[1]:+.0f} N" for t, m in marks.items()))
    dump("rest_runs.json", runs)
    flush("rest.txt")


# ================================================================ part fk
def first_harmonic(sig, t, om):
    return 2.0 * abs(np.mean(np.asarray(sig) * np.exp(-1j * om * np.asarray(
        t))))


def fk_refs(p, z, th, k, N=40000):
    """(textbook long-wave, exact linear with depth decay) surge FK
    amplitudes per unit wave amplitude, head seas, on the plant's geometry
    (hydro/hull_geometry) at attitude (z, th): rho g k |int A e^{ikx} dx| and
    rho g k |int A~ e^{ikx} dx|, A~ = int_A e^{-k depth} dA."""
    from hydro.hull_geometry import stations_of
    from sim.planing_vessel import SP
    h = p.hull
    g = stations_of(h)
    dxg = h.L / N
    xs = (np.arange(N) + 0.5) * dxg
    xb = xs - h.lcg
    Th = p.Th_rest + th
    d = -(p.Z_rest + z + (-h.vcg + g.keel(xs)[0]) + SP * xb * Th)
    yc = g.chine(xs)
    tb = p.tanb
    from sim.planing_anyspeed import _vsec
    A = _vsec(d, yc, tb)[0]
    dp = np.maximum(d, 0.0)
    dch = yc * tb
    b = np.minimum(dp, dch)
    ek = np.exp(-k * dp)
    Vp = (2.0 / tb) * ek * (np.exp(k * b) * (b / k - 1.0 / k ** 2)
                            + 1.0 / k ** 2)
    Wp = np.where(dp > dch, (2.0 * yc / k) * (1.0 - np.exp(-k * (dp - dch))),
                  0.0)
    At = Vp + Wp
    ph = np.exp(1j * k * xb)
    rg = p.rho * G
    return (rg * k * abs((A * ph).sum() * dxg),
            rg * k * abs((At * ph).sum() * dxg))


def fk_harmonic(hh, u, eqv, lamL, a=0.005):
    """First harmonic per unit wave amplitude of the plant's longitudinal
    hydrostatic + FK force Xh (head waves, hull `hh` held at eqv and towed
    at u, as part fk), and the mean h_w / d_T."""
    from sim.test_vessel import Monochromatic
    L = hh.L
    k = 2 * np.pi / (lamL * L)
    om = math.sqrt(G * k)
    ome = om + k * u
    q = plant(hh, sea=Monochromatic(om, a, theta0=np.pi), captive_u=u)
    ts = np.linspace(0.0, 2 * (2 * np.pi / ome), 128, endpoint=False)
    X, hwv = [], []
    for t in ts:
        s = np.zeros(14)
        s[0], s[2], s[4], s[6] = u * t, eqv[0], eqv[1], u
        _, aux = q.deriv(s, t)
        X.append(aux["Xh"])
        hwv.append(aux["hw"] / max(aux["dT"], 1e-12))
    return first_harmonic(X, ts, ome) / a, float(np.mean(hwv))


def part_fk():
    from hydro import hulls
    from hydro import vessel_spec as VS
    from sim.test_vessel import Monochromatic
    out("P1a (fk)  LOW-SPEED LONGITUDINAL WAVE FORCE, head waves, hull held "
        "at its calm captive equilibrium and towed (x0 = u t)")
    hw = VS.with_transom_closure(hulls.get("scarab195"), "scarab195_wet",
                                 F_T_crit=float("inf"))
    a = 0.005
    rows = []
    for kn in (0.5, 1.0, 2.0, 4.0, 6.0):
        u = kn * KN
        pc = plant(AS, captive_u=u)
        eqv, r = S1().newton(pc, u, np.zeros(2))
        for lamL in (0.5, 1.0, 2.0, 4.0, 6.0):
            k = 2 * np.pi / (lamL * pc.L)
            om = math.sqrt(G * k)
            ome = om + k * u
            sea = Monochromatic(om, a, theta0=np.pi)
            res = {}
            for lab, hh in (("wet", hw), ("nominal", hulls.get(AS))):
                q = plant(hh, sea=sea, captive_u=u)
                Te = 2 * np.pi / ome
                ts = np.linspace(0.0, 2 * Te, 128, endpoint=False)
                X, Xfk, hwv = [], [], []
                for t in ts:
                    s = np.zeros(14)
                    s[0], s[2], s[4], s[6] = u * t, eqv[0], eqv[1], u
                    _, aux = q.deriv(s, t)
                    X.append(aux["Xh"])
                    Xfk.append(aux["Xfk"])
                    hwv.append(aux["hw"] / max(aux["dT"], 1e-12))
                res[lab] = dict(X=first_harmonic(X, ts, ome) / a,
                                Xfk=first_harmonic(Xfk, ts, ome) / a,
                                hw_over_dT=float(np.mean(hwv)))
            tb_, ex_ = fk_refs(pc, eqv[0], eqv[1], k)
            ab = float(pc._a_b(u))
            row = dict(kn=kn, lamL=lamL, textbook=tb_, exact=ex_, a_b=ab,
                       wet=res["wet"]["X"], nominal=res["nominal"]["X"],
                       hw_over_dT=res["nominal"]["hw_over_dT"],
                       e_wet=res["wet"]["X"] / tb_ - 1,
                       e_wet_ab=res["wet"]["X"] / (ab * tb_) - 1,
                       e_nom=res["nominal"]["X"] / tb_ - 1,
                       e_exact=res["nominal"]["X"] / ex_ - 1,
                       e_wet_exact=res["wet"]["X"] / ex_ - 1)
            rows.append(row)
    out("    kn  lam/L | textbook N/m  exact N/m | wet: plant  /tb-1  | "
        "nominal: plant  /tb-1   h_w/d_T | /exact-1 (wet, nominal)")
    for r in rows:
        out(f"   {r['kn']:4.1f} {r['lamL']:4.1f} | {r['textbook']:10.1f} "
            f"{r['exact']:10.1f} | {r['wet']:10.1f} {100 * r['e_wet']:+6.2f}% |"
            f" {r['nominal']:10.1f} {100 * r['e_nom']:+7.2f}%  "
            f"{r['hw_over_dT']:.3f} | {100 * r['e_wet_exact']:+6.1f}% "
            f"{100 * r['e_exact']:+6.1f}%")
    w1 = max(abs(r["e_wet"]) for r in rows)
    check("fk", "FK1 (gate 3a as registered; kept, its failure on record): "
          "transom wet -- plant's longitudinal wave force vs the "
          "textbook closed-body long-wave FK (same geometry, 40 000 points),"
          " 0.5-6 kn x lam/L 0.5-6: within 5 %", w1 <= 0.05,
          f"largest |plant/textbook - 1| {100 * w1:.2f} %")
    w1b = max(abs(r["e_wet_ab"]) for r in rows)
    check("fk", "FK1' (re-registered after the P1a review, for later steps):"
          " transom wet -- plant's longitudinal wave force vs a_b(u) x the "
          "textbook value (the plant's pressure field carries v0's inherited"
          " a_b(C_V), register E2), 0.5-6 kn x lam/L 0.5-6: within 1 %",
          w1b <= 0.01, f"largest |plant/(a_b textbook) - 1| {100 * w1b:.2f} "
          f"%; a_b at 0.5-6 kn: " + ", ".join(
              f"{r['kn']:g} kn {r['a_b']:.3f}" for r in rows
              if r["lamL"] == 1.0))
    bad2 = [(r["kn"], r["lamL"], round(100 * r["e_nom"], 1)) for r in rows
            if abs(r["e_nom"]) > 0.05]
    check("fk", "FK2: default transom state (progressive, F_T_crit 3.25) vs "
          "the same textbook value: within 5 % (expected to fail where the "
          "face is partly dry)", not bad2,
          f"{len(rows) - len(bad2)} of {len(rows)} within; outside "
          f"(kn, lam/L, %): {bad2}")
    bad3 = [(r["kn"], r["lamL"], round(100 * r["e_exact"], 1)) for r in rows
            if abs(r["e_exact"]) > 0.05]
    check("fk", "FK3: default plant vs the exact linear FK with depth decay "
          "(Capytaine-equivalent): within 5 % (expected to fail at short "
          "waves: long-wave pressure field, step S3)", not bad3,
          f"{len(rows) - len(bad3)} of {len(rows)} within; outside: {bad3}")
    # Capytaine's FK of the probe mesh (zero speed, rest attitude)
    cap = []
    path = os.path.join(HERE, "_cache", "anyspeed",
                        "bem_probe_lid_24_64.npz")
    if os.path.exists(path):
        z = np.load(path)
        j = int(np.argmin(np.abs(z["directions"] - np.pi)))
        pr = plant(AS, captive_u=0.0)
        v, _ = eq_rest(pr)
        for i, om in enumerate(z["omega"][:7]):
            k = om * om / G
            fkc = abs(z["FK_re"][i, j, 0] + 1j * z["FK_im"][i, j, 0])
            tb_, ex_ = fk_refs(pr, v[0], v[1], k)
            cap.append(dict(omega=float(om), lamL=float(2 * np.pi / k
                                                         / pr.L),
                            capytaine=float(fkc), exact=ex_, textbook=tb_))
            out(f"    Capytaine probe w {om:5.2f} (lam/L {cap[-1]['lamL']:5.2f}"
                f"): |FK surge| {fkc:9.1f} N/m, exact-linear on the sections "
                f"{ex_:9.1f} ({100 * (ex_ / fkc - 1):+.1f} %), textbook "
                f"long-wave {tb_:9.1f} ({100 * (tb_ / fkc - 1):+.1f} %)")
    dump("fk.json", dict(rows=rows, capytaine=cap))
    flush("fk.txt")


# ============================================================ part energy
def _gz_pot_table(p, phi_max=0.3, n=30001):
    """W int_0^phi GZ: the roll potential of the plant's own gz(), as a
    cumulative trapezoid table (interpolated)."""
    ph = np.linspace(0.0, phi_max, n)
    gz = np.array([p.gz(x) for x in ph])
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (gz[1:] + gz[:-1])
                                           * np.diff(ph))]) * p.m * G
    return lambda phi: float(np.interp(abs(phi), ph, cum))


def energy_run(sw, dt, T, start):
    """The plant at rest, dissipation off (c_dc 0, roll damping 0; flux and
    friction vanish there), captive surge u = 0; exact energy samples."""
    from sim.planing_anyspeed import _vsec_q
    p = plant(AS, sw=sw, captive_u=0.0, dt=dt)
    p.c_dc = 0.0
    p.hull = copy.deepcopy(p.hull)
    p.hull.roll_zeta = 0.0
    v, _ = eq_rest(p)
    rg = p.rho * G
    s1 = S1()
    M2 = s1.mass2(p, 0.0, v[0], v[1])
    I44 = p.Mtot[3, 3]
    from sim.planing_vessel import SP

    def U(z, th):
        d = -(p.Z_rest + z + p.zk + SP * p.x * (p.Th_rest + th))
        return float((rg * _vsec_q(d, p.yc, p.tanb)[0]).sum() * p.dx
                     + p.m * G * z)
    U0 = U(v[0], v[1])
    gzpot = _gz_pot_table(p)
    s = np.zeros(14)
    s[2], s[4], s[3] = v[0] + start[0], v[1] + start[1], start[2]
    n = int(round(T / dt))
    every = max(1, int(round(0.02 / dt)))
    E, tt, Mvar, fmax = [], [], 0.0, 0.0
    for i in range(n + 1):
        if i % every == 0:
            Mi = s1.mass2(p, 0.0, s[2], s[4])
            Mvar = max(Mvar, float(np.abs(Mi - M2).max() / np.abs(M2).max()))
            vv = np.array([s[8], s[10]])
            e = (0.5 * vv @ Mi @ vv + U(s[2], s[4]) - U0
                 + 0.5 * I44 * s[9] ** 2 + gzpot(s[3]))
            E.append(e)
            tt.append(i * dt)
            if i % (every * 50) == 0:
                _, aux = p.deriv(s, i * dt)
                fmax = max(fmax, float(np.abs(aux["mom_full"]).max()))
        if i < n:
            s = p.step(s, i * dt, 0.0, 0.0)
            s[6] = 0.0
    E, tt = np.asarray(E), np.asarray(tt)
    w10 = tt <= 10.0
    l10 = tt >= tt[-1] - 10.0
    drift = (E[l10].mean() - E[w10].mean()) / E[0] / (T / 100.0)
    return dict(sw="verlet" if "verlet" in sw else "heun", dt=dt, T=T,
                E0=float(E[0]), drift_per_100s=float(drift),
                max_dev=float(np.abs(E - E[0]).max() / E[0]),
                M_var=Mvar, flux_max=fmax,
                E_first=float(E[w10].mean()), E_last=float(E[l10].mean()))


def osc_radius(scheme_verlet, wh):
    """Spectral radius of the one-step map of the plant's step code on the
    undamped oscillator z'' = -w^2 z (h = 1, w = wh)."""
    from sim.planing_anyspeed import _StripAS

    class Osc:
        def deriv(self, s, t):
            ds = np.zeros(14)
            ds[2] = s[8]
            ds[8] = -(wh ** 2) * s[2]
            return ds, {}
        _kin = staticmethod(_StripAS._kin)
    o = Osc()
    Mmap = np.zeros((2, 2))
    for j, (z0, w0) in enumerate(((1.0, 0.0), (0.0, 1.0))):
        s = np.zeros(14)
        s[2], s[8] = z0, w0
        k1, _ = o.deriv(s, 0.0)
        if scheme_verlet:
            sn = _StripAS._sv(o, s, k1, 1.0, 1.0)
        else:
            k2, _ = o.deriv(s + k1, 1.0)
            sn = s + 0.5 * (k1 + k2)
        Mmap[:, j] = sn[2], sn[8]
    return float(np.abs(np.linalg.eigvals(Mmap)).max())


def part_energy():
    from sim.planing_anyspeed import P1A_CANDIDATE
    out("P1a (energy)  M3 / E2: energy with dissipation off at rest; the "
        "scheme's one-step amplification")
    whs = [0.01, 0.02, 0.05, 0.08, 0.1, 0.16, 0.2, 0.3, 1.0, 1.9]
    rv = [osc_radius(True, x) for x in whs]
    rh = [osc_radius(False, x) for x in whs]
    for x, a, b in zip(whs, rv, rh):
        out(f"    w h {x:5.2f}: spectral radius verlet {a:.15f}  heun "
            f"{b:.15f} (analytic heun sqrt(1 + (w h)^4/4) {math.sqrt(1 + x ** 4 / 4):.15f})")
    check("energy", "one-step map of the plant's verlet code on an undamped "
          "oscillator: spectral radius 1 (|.-1| <= 1e-12) for w h <= 0.3 "
          "(heun > 1)", max(abs(r - 1) for x, r in zip(whs, rv) if x <= 0.3)
          <= 1e-12 and all(r > 1 for x, r in zip(whs, rh) if x >= 0.05),
          f"verlet max |r - 1| {max(abs(r - 1) for r in rv[:8]):.1e}; heun "
          f"r - 1 at w h 0.08: {rh[3] - 1:.2e}")
    sw_v = tuple(P1A_CANDIDATE)
    sw_h = tuple(s for s in P1A_CANDIDATE if s != "verlet")
    start = (0.01, 0.003, 0.05)
    runs = []
    for sw in (sw_h, sw_v):
        for dt, T in ((0.002, 100.0), (0.02, 100.0), (0.02, 600.0)):
            t0 = time.time()
            r = energy_run(sw, dt, T, start)
            r["seconds"] = time.time() - t0
            runs.append(r)
            out(f"    {r['sw']:6s} dt {dt:.3f} T {T:5.0f} s: E0 {r['E0']:.4e} J,"
                f" drift {100 * r['drift_per_100s']:+.4f} % per 100 s, max "
                f"|E - E0|/E0 {100 * r['max_dev']:.4f} %; mass matrix "
                f"variation {r['M_var']:.1e}, max flux {r['flux_max']:.1e} N/m"
                f" ({r['seconds']:.0f} s)")
    # Heun's analytic growth per mode (linearised heave-pitch and roll)
    p = plant(AS, sw=sw_h, captive_u=0.0)
    v, _ = eq_rest(p)
    s1 = S1()
    J = s1.jac4(p, 0.0, v[0], v[1])
    om = sorted(abs(np.linalg.eigvals(J).imag))[::2]
    om_r = math.sqrt(p.m * G * p.GM / p.Mtot[3, 3])
    for dt in (0.002, 0.02):
        g_ = [((1 + (w * dt) ** 4 / 4) ** (100.0 / dt) - 1) for w in
              list(om) + [om_r]]
        out(f"    heun analytic energy growth per 100 s at dt {dt}: "
            + ", ".join(f"w {w:.2f}: {100 * x:.4f} %" for w, x in
                        zip(list(om) + [om_r], g_)))
    rv2 = [r for r in runs if r["sw"] == "verlet" and r["dt"] == 0.002][0]
    check("energy", "M3: new scheme, dissipation off, dt 0.002: energy drift"
          " <= 0.1 % per 100 s", abs(rv2["drift_per_100s"]) <= 1e-3,
          f"{100 * rv2['drift_per_100s']:+.5f} % per 100 s (max deviation "
          f"{100 * rv2['max_dev']:.5f} %)")
    rv3 = [r for r in runs if r["sw"] == "verlet" and r["dt"] == 0.02]
    out("    production step 0.02 s (recorded): " + "; ".join(
        f"T {r['T']:.0f} s drift {100 * r['drift_per_100s']:+.4f} %/100 s, "
        f"max dev {100 * r['max_dev']:.3f} %" for r in rv3))
    dump("energy.json", dict(radius=dict(wh=whs, verlet=rv, heun=rh),
                             runs=runs, modes=list(om), roll=om_r))
    flush("energy.txt")


# ============================================================ part astern
def part_astern():
    from learn.repro import task
    from sim.planing_vessel import SP
    out("P1a (astern)  -3..0 kn")
    s1 = S1()
    rows = []
    plants = (("candidate", lambda **k: plant(AS, **k)),
              ("as1 before", lambda **k: plant(AS, sw=PRE_P1A, **k)),
              ("v0", lambda **k: plant("scarab195", **k)))
    for lab, mk in plants:
        for kn in (-0.5, -1.0, -2.0, -3.0):
            u = kn * KN
            p = mk(captive_u=u)
            z, th, ppz, ppt = s1.settle(p, u, T=60.0)
            ok = np.isfinite(z) and np.isfinite(th)
            _, aux = p.deriv(s1.state(u, z, th), 0.0)
            rows.append(dict(plant=lab, kn=kn, z=z, trim=s1.trim_deg(p, th),
                             fx=float(aux["fx"]), Df=float(aux["Df"]),
                             Xp=float(aux["Xp"]),
                             Dface=float(aux.get("Dface", np.nan)),
                             finite=bool(ok), pp_z=ppz))
            out(f"    {lab:11s} {kn:+5.1f} kn: holding force {aux['fx']:+9.2f}"
                f" N (friction {aux['Df']:+8.2f}, pressure {aux['Xp']:+8.2f}, "
                f"face drag {aux.get('Dface', float('nan')):+8.2f}), z "
                f"{z:+.4f} trim {s1.trim_deg(p, th):+.3f}")
    cand = [r for r in rows if r["plant"] == "candidate"]
    signed = all(r["fx"] < 0 for r in cand)
    mono = all(abs(cand[i + 1]["fx"]) > abs(cand[i]["fx"])
               for i in range(len(cand) - 1))
    check("astern", "captive astern -0.5..-3 kn: finite, the holding force "
          "pulls astern (resistance opposes the motion) and grows with |u|",
          signed and mono and all(r["finite"] for r in cand),
          ", ".join(f"{r['kn']:+.1f} kn {r['fx']:+.1f} N" for r in cand))
    # coast-down from -3 kn
    cd = {}
    for lab, mk in plants:
        p = mk()
        s = p.initial_state(0.0)
        s[6] = -3.0 * KN
        p.captive_u = None
        us = [s[6]]
        bad = None
        for i in range(int(300 / p.dt)):
            s = p.step(s, i * p.dt, 0.0, 0.0)
            if not np.all(np.isfinite(s)):
                bad = i * p.dt
                break
            us.append(s[6])
        us = np.asarray(us)
        mono = bool(np.all(np.diff(np.abs(us)) <= 1e-12))
        inc = np.diff(np.abs(us))
        cd[lab] = dict(u_end=float(us[-1]), max_u=float(us.max()),
                       monotone=mono, diverged=bad,
                       u_30=float(us[int(30 / p.dt)]),
                       sign_changes=int(np.sum(np.diff(np.sign(us)) != 0)),
                       n_rising=int((inc > 0).sum()),
                       max_step_rise=float(inc.max()),
                       max_above_running_min=float(
                           (np.abs(us) - np.minimum.accumulate(
                               np.abs(us))).max()),
                       pp_290_300=float(np.ptp(us[int(290 / p.dt):])))
        if lab == "candidate":
            cand_us = us
        out(f"    coast-down {lab:11s}: u 30 s {cd[lab]['u_30'] / KN:+.3f} kn,"
            f" 300 s {us[-1] / KN:+.4f} kn, max u {us.max() / KN:+.4f} kn, "
            f"|u| monotone {mono}, diverged {bad}; |u| rises in "
            f"{cd[lab]['n_rising']} steps (largest single step "
            f"{cd[lab]['max_step_rise']:.1e} m/s, largest excursion above "
            f"its running minimum {cd[lab]['max_above_running_min']:.1e} "
            f"m/s; u p-p over 290-300 s {cd[lab]['pp_290_300']:.1e} m/s)")
    c = cd["candidate"]
    check("astern", "zero-thrust coast-down from -3 kn, 300 s: finite, |u| "
          "monotone decreasing, never ahead (gate 5 as registered; kept, "
          "its failure on record)", c["diverged"] is None and
          c["monotone"] and c["max_u"] <= 0.0,
          f"u(300 s) {c['u_end'] / KN:+.4f} kn")
    # re-registered after the P1a review (for later steps): a trimmed hull
    # that heaves exchanges horizontal fluid impulse with the surge at first
    # order, so |u| is judged averaged over the heave/pitch periods at rest
    # (no radiation damping there until S4): |u| passed through two
    # consecutive moving averages, one over each natural period of the
    # linearised heave/pitch at rest (each boxcar nulls its own mode), must
    # be non-increasing, and u must never change sign
    pr = plant(AS, captive_u=0.0)
    vr, _ = eq_rest(pr)
    lam = np.linalg.eigvals(S1().jac4(pr, 0.0, vr[0], vr[1]))
    oms = sorted({round(abs(x.imag), 9) for x in lam if abs(x.imag) > 1e-9})
    Ts = [2 * np.pi / w for w in oms]
    dtp = plant(AS).dt
    filt = np.abs(cand_us)
    for Tm in Ts:
        nw = max(1, int(round(Tm / dtp)))
        filt = np.convolve(filt, np.ones(nw) / nw, mode="valid")
    rise = float(np.diff(filt).max())
    cd["candidate"].update(rest_periods=Ts, filtered_max_rise=rise)
    check("astern", "coast-down, RE-REGISTERED (P1a review; first run of "
          "this form): u never changes sign and |u| averaged over the "
          "heave/pitch periods at rest (two moving averages, "
          + ", ".join(f"{T:.3f} s" for T in Ts) + ") non-increasing",
          c["sign_changes"] == 0 and c["max_u"] <= 0.0 and rise <= 0.0,
          f"sign changes {c['sign_changes']}, largest rise of the averaged "
          f"|u| {rise:+.2e} m/s per step")
    # reverse thrust, calm and task sea
    m = task.Mission("high", 0, 0, t_end=1.0, track=task.LEGS[0])
    rev = {}
    for sea_lab, sea in (("calm", None), ("task sea", m.plant.sea)):
        for frac in (0.25, 0.5, 1.0):
            p = plant(AS, sea=sea, dt=m.plant.dt)
            s = p.initial_state(0.0)
            cmd = frac * p.prop.t_min
            bad, us, trs = None, [], []
            for i in range(int(120 / p.dt)):
                s = p.step(s, i * p.dt, cmd, 0.0)
                if not np.all(np.isfinite(s)):
                    bad = i * p.dt
                    break
                if i * p.dt >= 90:
                    us.append(s[6])
                    trs.append(s1.trim_deg(p, s[4]))
            rev[f"{sea_lab}_{frac}"] = dict(
                diverged=bad, kn=float(np.mean(us) / KN) if us else np.nan,
                trim=float(np.mean(trs)) if trs else np.nan,
                capsized=bool(p.capsized))
            r = rev[f"{sea_lab}_{frac}"]
            out(f"    reverse thrust {frac:.2f} t_min, {sea_lab:8s}: speed "
                f"{r['kn']:+.2f} kn (mean of 90-120 s), trim {r['trim']:+.2f},"
                f" diverged {bad}, capsized {r['capsized']}")
    check("astern", "reverse thrust 0.25 / 0.5 / 1 x t_min, calm and the task"
          " sea, 120 s: finite, astern", all(r["diverged"] is None and
                                              r["kn"] < 0 for r in
                                              rev.values()),
          ", ".join(f"{k} {v['kn']:+.2f} kn" for k, v in rev.items()))
    dump("astern.json", dict(captive=rows, coast=cd, reverse=rev))
    flush("astern.txt")


# ============================================================== part conv
def part_conv():
    from hydro import hulls
    from sim import physics
    from sim.planing_anyspeed import PlaningVesselASLin
    s1 = S1()
    out("P1a (conv)  M5 (recorded): calm captive equilibria vs station count,"
        " the 3072-station Newton root as reference; P1's registered bounds "
        "(768: <= 0.001 deg, 48: <= 0.05 deg, fitted order (96-768) >= 1.0)")
    ns = (48, 96, 192, 384, 768, 3072)
    rows = []
    for kn in (6.0, 9.0, 12.0, 16.0, 20.0):
        u = kn * KN
        ref_g = np.array(s1.build("v0").running_attitude(u))
        res = {}
        for n in ns:
            hn = s1.hull_n(n)
            ha = (hulls.get(AS) if n == 48 else
                  physics.variant(hn, "as1", name=hn.name))
            p = PlaningVesselASLin(ha, calm(), captive_u=u)
            v, r = s1.newton(p, u, ref_g if n == 48 else
                             np.array(res[48][:2]))
            res[n] = (v[0], v[1], s1.trim_deg(p, v[1]), r)
        tr_ref, z_ref = res[3072][2], res[3072][0]
        e = {n: abs(res[n][2] - tr_ref) for n in ns[:-1]}
        ez = {n: abs(res[n][0] - z_ref) for n in ns[:-1]}
        xs = np.log([96, 192, 384, 768])
        ys = np.log([max(e[n], 1e-15) for n in (96, 192, 384, 768)])
        A_ = np.vstack([xs, np.ones_like(xs)]).T
        coef, resid, *_ = np.linalg.lstsq(A_, ys, rcond=None)
        dof = max(len(xs) - 2, 1)
        s2 = float(resid[0]) / dof if resid.size else 0.0
        cov = s2 * np.linalg.inv(A_.T @ A_)
        order, sig = -coef[0], math.sqrt(max(cov[0, 0], 0.0))
        ok = e[768] <= 1e-3 and e[48] <= 0.05 and order >= 1.0
        rows.append(dict(kn=kn, ref_trim=tr_ref, ref_z=z_ref, err=e,
                         err_z=ez, order=order, sigma=sig, ok=ok,
                         resid={n: res[n][3] for n in ns}))
        out(f"   {kn:4.1f} kn: ref trim {tr_ref:.6f} deg z {z_ref:+.6f} m; "
            "trim error " + ", ".join(f"{n}: {e[n]:.1e}" for n in ns[:-1])
            + f"; order {order:.2f} +- {sig:.2f}; z error 768: {ez[768]:.1e}"
            f" m; within P1's bounds: {ok}")
    check("conv", "M5 recorded against P1's registered bounds (they target "
          "P1b's friction items; not a P1a adoption gate)", True,
          ", ".join(f"{r['kn']:g} kn {'within' if r['ok'] else 'OUTSIDE'}"
                    for r in rows))
    dump("conv.json", rows)
    flush("conv.txt")


# ============================================================== part sens
def branch(p, kns, start):
    s1 = S1()
    g = np.array(start, float)
    rows = []
    for kn in kns:
        u = kn * KN
        p.captive_u = u
        v, r = s1.newton(p, u, g)
        J2 = s1.jac2(p, u, v)
        _, mre, _ = s1.modes(s1.jac4(p, u, v[0], v[1]))
        _, aux = p.deriv(s1.state(u, v[0], v[1]), 0.0)
        rows.append(dict(kn=kn, z=float(v[0]), th=float(v[1]),
                         trim=s1.trim_deg(p, v[1]), resid=r,
                         det=float(np.linalg.det(J2)), maxre=mre,
                         R=float(aux["fx"]), hw=float(aux["hw"]),
                         dT=float(aux["dT"]), face=float(aux["Qw"]),
                         QT=float(aux["QT"])))
        g = v
    return rows


def part_sens():
    from hydro import hulls
    from hydro import vessel_spec as VS
    out("P1a (sens)  G3: ventilation threshold over its registered scatter "
        f"{VS.TRANSOM_SCATTER}, the 'progressive_static' and 'step' forms "
        "(sensitivity only); calm captive branch by Newton continuation "
        "1 -> 25 -> 1 kn; the wave force FK2 at 4 and 6 kn; the implied "
        "base-drag coefficient; the static-to-instantaneous draught mapping")
    base = hulls.get("scarab195")
    variants = {
        "F_T_crit 2.0": dict(F_T_crit=2.0),
        "F_T_crit 3.25 (default)": dict(F_T_crit=3.25),
        "F_T_crit 4.5": dict(F_T_crit=4.5),
        "progressive_static 3.25": dict(F_T_crit=3.25,
                                        mode="progressive_static",
                                        sensitivity=True),
        "step 3.25": dict(F_T_crit=3.25, mode="step", sensitivity=True),
        "never ventilates": dict(F_T_crit=float("inf")),
    }
    kns = [float(k) for k in np.arange(1.0, 25.01, 0.5)]
    res = {}
    for i, (lab, cl) in enumerate(variants.items()):
        hv = VS.with_transom_closure(base, f"scarab195_vt{i}", **cl)
        p = plant(hv, captive_u=kns[0] * KN)
        v, _ = eq_rest(plant(hv, captive_u=0.0))
        up = branch(p, kns, v)
        dn = branch(p, kns[::-1], [up[-1]["z"], up[-1]["th"]])[::-1]
        dmax = max(abs(a["trim"] - b["trim"]) for a, b in zip(up, dn))
        jumps = [abs(up[j + 1]["trim"] - up[j]["trim"]) for j in
                 range(len(up) - 1)]
        jR = [abs(up[j + 1]["R"] - up[j]["R"]) / max(up[j]["R"], 1.0)
              for j in range(len(up) - 1)]
        signs = {np.sign(r["det"]) for r in up + dn}
        res[lab] = dict(closure=cl, up=up, down=dn, updown=dmax,
                        max_trim_step=max(jumps), max_R_step=max(jR),
                        det_signs=sorted(float(s) for s in signs),
                        unstable=[r["kn"] for r in up if r["maxre"] >= 0])
        out(f"  [{lab}] up-down {dmax:.1e} deg; largest trim change per 0.5 kn"
            f" {max(jumps):.3f} deg, largest R change {100 * max(jR):.1f} %; "
            f"det signs {res[lab]['det_signs']}; unstable at "
            f"{res[lab]['unstable']}")
        out("     kn: " + "  ".join(f"{r['kn']:g}:{r['trim']:.2f}/"
                                    f"{r['R']:.0f}/{r['hw'] / max(r['dT'], 1e-9):.2f}"
                                    for r in up[::2]))
    d = res["F_T_crit 3.25 (default)"]
    check("sens", "default closure: continuation up and down gives the same "
          "branch (<= 1e-6 deg), no fold (det sign constant), every root "
          "stable", d["updown"] <= 1e-6 and len(d["det_signs"]) == 1
          and not d["unstable"],
          f"up-down {d['updown']:.1e}, det signs {d['det_signs']}")
    # the static-to-instantaneous draught mapping: where each variant's
    # calm branch clears the transom (first speed with h_w = 0), the
    # transom draught there, and the threshold that speed means on the
    # static draught (the sources' variable) and on the instantaneous one
    hv0 = VS.with_transom_closure(base, "scarab195_vtmap", F_T_crit=3.25)
    TT0 = plant(hv0)._transom_geom().TT0
    mapping = {}
    out(f"    static transom draught T_T0 {TT0:.4f} m")
    for lab, r in res.items():
        up = r["up"]
        j = next((i for i, x in enumerate(up) if x["hw"] <= 0.0), None)
        if j is None or j == 0:
            mapping[lab] = None
            out(f"    [{lab}] the transom does not clear on 1-25 kn")
            continue
        # linear interpolation of h_w between the last wet and first dry
        a_, b_ = up[j - 1], up[j]
        f = a_["hw"] / (a_["hw"] - b_["hw"]) if a_["hw"] != b_["hw"] else 1.0
        kc_ = a_["kn"] + f * (b_["kn"] - a_["kn"])
        dTc = a_["dT"] + f * (b_["dT"] - a_["dT"])
        Uc = kc_ * KN
        mapping[lab] = dict(kn=kc_, dT=dTc, F_inst=Uc / math.sqrt(G * dTc),
                            F_static=Uc / math.sqrt(G * TT0))
        m = mapping[lab]
        out(f"    [{lab}] clears at ~{m['kn']:.2f} kn, d_T there "
            f"{m['dT']:.3f} m: F on the instantaneous draught "
            f"{m['F_inst']:.2f}, on the static draught {m['F_static']:.2f}")
    d10 = next(x for x in d["up"] if abs(x["kn"] - 10.0) < 1e-9)
    out(f"    default at 10 kn: d_T {d10['dT']:.3f} m against T_T0 "
        f"{TT0:.3f} m (squat of the stern {d10['dT'] - TT0:+.3f} m)")
    # the implied base-drag coefficient before clearance: a_b (Q(d_T) -
    # Q(h_w)) / (1/2 rho U^2 A_T) -> 2 / F_T_crit^2 as U -> 0 (a_b = 1
    # there); literature: Holtrop 1984 (ISP 31(363)), transom pressure
    # resistance 1/2 rho V^2 A_T c6, c6 = 0.2 (1 - 0.2 F_nT) for F_nT < 5
    # (second-hand, not checked against the original; register E15)
    cdb = {}
    from sim.planing_anyspeed import _vsec_qs
    for lab in ("F_T_crit 2.0", "F_T_crit 3.25 (default)", "F_T_crit 4.5"):
        F = variants[lab]["F_T_crit"]
        hv = VS.with_transom_closure(base, f"scarab195_cdb{F:g}",
                                     F_T_crit=F)
        q = plant(hv, captive_u=0.0)
        v, _ = eq_rest(plant(hv, captive_u=0.0))
        tg = q._transom_geom()
        rows_c = []
        for u in (0.25, 0.5, 1.0, 2.0, 3.0):
            q.captive_u = u
            _, aux = q.deriv(S1().state(u, v[0], v[1]), 0.0)
            AT = _vsec_qs(aux["dT"], tg.ycT, q.tanb)[2]
            Db = aux["QT"] - aux["Qw"]
            rows_c.append(dict(u=u, kn=u / KN, A_T=AT, D_base=Db,
                               a_b=float(q._a_b(u)),
                               C_D=Db / (0.5 * q.rho * u * u * AT)))
        cdb[lab] = dict(F=F, theory=2.0 / F ** 2, rows=rows_c)
        out(f"    [{lab}] implied C_D,base (on the immersed transom area) "
            + ", ".join(f"{r['kn']:.2f} kn {r['C_D']:.4f}" for r in rows_c)
            + f"; 2 / F_T_crit^2 = {2.0 / F ** 2:.4f} (Holtrop 1984: 0.2 "
            "at low transom Froude number)")
    check("sens", "registered consequence of the closure (E15): at low speed "
          "(0.25 m/s) the implied base-drag coefficient a_b (Q_T - Q_w) / "
          "(1/2 rho U^2 A_T) equals 2 / F_T_crit^2 within 2 % (recorded "
          "against Holtrop 1984's 0.2)",
          all(abs(c["rows"][0]["C_D"] / c["theory"] - 1) <= 0.02
              for c in cdb.values()),
          ", ".join(f"F {c['F']:g}: {c['rows'][0]['C_D']:.4f} vs "
                    f"{c['theory']:.4f}" for c in cdb.values()))
    # the wave force FK2 at 4 and 6 kn across the closure variants (the
    # closure's largest effect: the base-drag modulation rho g [A(d_T) -
    # A(h_w)] eta_T), against the closed body (never ventilates)
    fk2 = {}
    for kn in (4.0, 6.0):
        u = kn * KN
        for i, (lab, cl) in enumerate(variants.items()):
            hv = VS.with_transom_closure(base, f"scarab195_vt{i}", **cl)
            pc = plant(hv, captive_u=u)
            eqv, _ = S1().newton(pc, u, np.zeros(2))
            for lamL in (0.5, 1.0, 2.0, 4.0, 6.0):
                X, hwr = fk_harmonic(hv, u, eqv, lamL)
                fk2[(kn, lamL, lab)] = dict(X=X, hw_over_dT=hwr)
    out("    FK2 |X| per unit amplitude (N/m), its difference from the "
        "closed body (never ventilates), h_w/d_T")
    for kn in (4.0, 6.0):
        for lamL in (0.5, 1.0, 2.0, 4.0, 6.0):
            ref = fk2[(kn, lamL, "never ventilates")]["X"]
            out(f"     {kn:3.0f} kn lam/L {lamL:3.1f}: closed {ref:7.0f} | "
                + "  ".join(
                    f"{lab.split(' (')[0]}: {fk2[(kn, lamL, lab)]['X']:7.0f} "
                    f"({100 * (fk2[(kn, lamL, lab)]['X'] / ref - 1):+5.1f} %"
                    f", {fk2[(kn, lamL, lab)]['hw_over_dT']:.2f})"
                    for lab in variants if lab != "never ventilates"))

    def rng_(kn, lamL):
        ref = fk2[(kn, lamL, "never ventilates")]["X"]
        r = [fk2[(kn, lamL, lab)]["X"] / ref - 1 for lab in variants
             if lab != "never ventilates"]
        return f"{kn:g} kn lam/L {lamL:g}: {100 * min(r):+.0f}..{100 * max(r):+.0f} %"
    check("sens", "G3 sensitivity of the wave force (recorded): FK2 at 4 "
          "and 6 kn across the closure variants against the closed body",
          True, "; ".join(rng_(kn, lamL) for kn in (4.0, 6.0)
                          for lamL in (1.0, 4.0)))
    # randomisation over the scatter
    rng = np.random.default_rng(2026)
    draws = []
    for i in range(5):
        F = float(rng.uniform(*VS.TRANSOM_SCATTER))
        hv = VS.with_transom_closure(base, f"scarab195_rand{i}", F_T_crit=F)
        q = plant(hv)
        c = q.calm_resistance(10.0 * KN)
        draws.append(dict(F_T_crit=F, R=c["R"], trim=c["trim_deg"]))
        out(f"    random draw {i}: F_T_crit {F:.3f} -> 10 kn R {c['R']:.1f} N,"
            f" trim {c['trim_deg']:.3f} deg")
    dump("sens.json", dict(variants=res, draws=draws, mapping=mapping,
                           TT0=TT0, cd_base=cdb,
                           fk2=[dict(kn=k[0], lamL=k[1], variant=k[2], **v)
                                for k, v in fk2.items()]))
    flush("sens.txt")


# ================================================================ part v1
def part_v1():
    from hydro import hulls
    from hydro.planing import PlaningHull, savitsky
    from sim import physics
    from sim.planing_anyspeed import PlaningVesselAS
    from sim.planing_vessel import PlaningVessel
    from sim.wavefield import SeaState
    from studies import exp_planing as EP
    s1 = S1()
    out("P1a (v1)  V1 (recorded, not tuned): ACCEPTANCE tolerances -- "
        "Savitsky R +-10 %, trim +-1 deg; Fridsma amplitudes +-20 %, "
        "accelerations +-30 %")
    res = dict(savitsky=[], fridsma_calm=[], p6=[], p7=[])
    h0 = hulls.get("scarab195")
    pv = plant("scarab195")
    pa = plant(AS)
    for kn in (20, 25, 30, 35, 40):
        sv = savitsky(kn * KN, h0.mass, h0.lcg, h0.b, h0.beta_deg)
        row = dict(kn=kn, sav_R=sv["R"], sav_trim=sv["tau_deg"])
        for lab, p in (("v0", pv), ("as1", pa)):
            r = p.calm_resistance(kn * KN)
            R = r["R"] - r["air"]
            row[lab] = dict(R=R, trim=r["trim_deg"], eR=R / sv["R"] - 1,
                            et=r["trim_deg"] - sv["tau_deg"])
        res["savitsky"].append(row)
        out(f"    Savitsky {kn} kn: R {sv['R']:.0f} N trim {sv['tau_deg']:.2f}"
            f" | v0 R {row['v0']['R']:.0f} ({100 * row['v0']['eR']:+.1f} %) "
            f"trim {row['v0']['trim']:.2f} ({row['v0']['et']:+.2f}) | as1 R "
            f"{row['as1']['R']:.0f} ({100 * row['as1']['eR']:+.1f} %) trim "
            f"{row['as1']['trim']:.2f} ({row['as1']['et']:+.2f})")
    okS = [r["kn"] for r in res["savitsky"] if abs(r["as1"]["eR"]) <= 0.10
           and abs(r["as1"]["et"]) <= 1.0]
    check("v1", "Savitsky 20-40 kn (R +-10 %, trim +-1 deg), candidate as1",
          len(okS) == 5, f"within at {okS} kn of [20, 25, 30, 35, 40]")

    def mk(hv):
        def make(hull, sea, dt, V):
            return PlaningVesselAS(hv, sea, dt=dt, captive_u=V)
        return make
    meas_calm = {"A": (4.0, 0.05, 0.158), "B": (4.0, 0.13, 0.206)}
    n_in = n_tot = 0
    for case in ("A", "B"):
        h, V = EP.fridsma_hull(case, tow="nozzle")
        hv = physics.variant(h, "as1")
        c0 = PlaningVessel(h, calm(), dt=0.004).calm_resistance(V)
        c1 = PlaningVesselAS(hv, calm(), dt=0.004).calm_resistance(V)
        mt, mr, mR = meas_calm[case]
        res["fridsma_calm"].append(dict(case=case, v0=c0, as1=c1,
                                        measured=meas_calm[case]))
        out(f"    Fridsma {case} calm: trim v0 {c0['trim_deg']:.2f} as1 "
            f"{c1['trim_deg']:.2f} (meas {mt}); rise/b v0 "
            f"{c0['rise'] / h.b:.3f} as1 {c1['rise'] / h.b:.3f} (meas {mr}); "
            f"R/Delta v0 {c0['R'] / (h.mass * G):.3f} as1 "
            f"{c1['R'] / (h.mass * G):.3f} (meas {mR})")
        for lam, meas in EP.FRIDSMA[case].items():
            hb = 0.111 * h.b
            r0 = EP.regular_wave(h, V, lam, hb, T=16.0, dt=0.004)
            r1 = s1.regular_wave_on(mk(hv), h, V, lam, hb, T=16.0, dt=0.004)
            tol = (0.2, 0.2, 0.3, 0.3)
            ins = [abs(r1[j] / meas[j] - 1) <= tol[j] for j in range(4)]
            n_in += sum(ins)
            n_tot += 4
            res["p6"].append(dict(case=case, lam=lam, v0=r0, as1=r1,
                                  measured=meas, within=ins))
            out(f"     P6 {case} lam/L {lam}: heave v0 {r0[0]:.2f} as1 "
                f"{r1[0]:.2f} meas {meas[0]:.2f} | pitch v0 {r0[1]:.2f} as1 "
                f"{r1[1]:.2f} meas {meas[1]:.2f} | CG g v0 {r0[2]:.2f} as1 "
                f"{r1[2]:.2f} meas {meas[2]:.2f} | bow g v0 {r0[3]:.2f} as1 "
                f"{r1[3]:.2f} meas {meas[3]:.2f}  within {ins}")
    check("v1", "Fridsma 1969 regular waves (P6): quantities within the "
          "ACCEPTANCE tolerances, candidate as1 (recorded)", True,
          f"{n_in} of {n_tot} within")
    # P7
    b = 9 * 0.0254
    L = 5 * b
    n7 = t7 = 0
    for cond, (vl, lcg_aft, hb, m_cg, m_bow) in EP.FRIDSMA71.items():
        V = vl * 0.514444 * np.sqrt(L / 0.3048)
        h = PlaningHull(name=f"fridsma71_{cond}", L=L, b=b, beta_deg=20.0,
                        mass=0.600 * 62.4 * (b / 0.3048) ** 3 * 0.45359237,
                        lcg=(1 - lcg_aft / 100) * L, vcg=0.294 * b,
                        depth=0.6 * b, k_pitch=0.25, taper=0.2, cda=0.0,
                        u_design=V, rho=1000.0,
                        jet=dict(d_nozzle=0.02, p_max=500.0))
        hv = physics.variant(h, "as1")
        hs = hb * b
        tp = 15.63 * np.sqrt(hs / 9.81)
        sea = SeaState(hs, tp, gamma=1.0, theta0=np.pi, n_freq=40, n_dir=1,
                       seed=cond)
        row = dict(cond=cond, meas=(m_cg, m_bow))
        for lab, p in (("v0", PlaningVessel(h, sea, dt=0.004, captive_u=V)),
                       ("as1", PlaningVesselAS(hv, sea, dt=0.004,
                                               captive_u=V))):
            s = np.zeros(14)
            s[6] = V
            s[2], s[4] = p.running_attitude(V)
            xa = h.L - h.lcg - 0.1 * h.L
            n = int(60.0 / 0.004)
            a = np.empty((n, 2))
            for i in range(n):
                s = p.step(s, i * 0.004, 0.0, 0.0)
                s[6] = V
                ac, ab = p.last_cg_acc, p.last_bow_acc
                a[i] = ac / 9.81, (ac + (ab - ac) * xa / p.x[-1]) / 9.81
            a = a[n // 10:]
            row[lab] = (float(EP.excursion_peaks(a[:, 0]).mean()),
                        float(EP.excursion_peaks(a[:, 1]).mean()))
        ins = [abs(row["as1"][j] / row["meas"][j] - 1) <= 0.3 for j in (0, 1)]
        n7 += sum(ins)
        t7 += 2
        row["within"] = ins
        res["p7"].append(row)
        out(f"     P7 cond {cond}: CG v0 {row['v0'][0]:.2f} as1 "
            f"{row['as1'][0]:.2f} meas {m_cg:.2f} | bow v0 {row['v0'][1]:.2f}"
            f" as1 {row['as1'][1]:.2f} meas {m_bow:.2f}  within {ins}")
    check("v1", "Fridsma 1971 irregular (P7, nominal Tp): mean acceleration "
          "peaks within +-30 %, candidate as1 (recorded)", True,
          f"{n7} of {t7} within")
    dump("v1.json", res)
    flush("v1.txt")


# ============================================================== part twin
def part_twin():
    """M3 Froude similarity of the candidate (transom state, face drag,
    turning normal, integrator included): scarab195@as1 against its Froude
    twins at lam 1.5 and 0.6 (hydro/vessel_spec.froude_twin, every input
    scaled): calm captive roots at Fn_L 0.25, 0.5, 1.0, 1.6 (trim, z/L,
    R/(m g), h_w/d_T, Xh/(m g)) and 600 steps open loop in a Froude-scaled
    sea (studies/anyspeed_p0_check.open_loop_nd). Bounds: 1e-4
    (ACCEPTANCE M3) and the internal 1e-10 (the same arithmetic)."""
    from hydro import hulls
    from hydro import vessel_spec as VS
    from sim import physics
    from studies import anyspeed_p0_check as P0
    s1 = S1()
    out("P1a (twin)  M3 Froude similarity of the candidate")
    base = hulls.get(AS)
    twins = {lam: physics.variant(VS.froude_twin(
        hulls.get("scarab195"), lam, name=f"scarab195_p1atwin{lam:g}"), "as1")
        for lam in (1.5, 0.6)}

    resid = []

    def roots(h):
        # each speed from its own captive run's attitude (a first attempt
        # continued from the previous root and stopped at non-roots at Fn 1
        # and 1.6: residual unchecked)
        rows = []
        for fn in (0.25, 0.5, 1.0, 1.6):
            u = fn * math.sqrt(G * h.L)
            p = plant(h, captive_u=u)
            g = np.array(p.running_attitude(u))
            v, r = s1.newton(p, u, g)
            resid.append(r * h.L / G)      # nondimensional (w' / g, q' L / g)
            _, aux = p.deriv(s1.state(u, v[0], v[1]), 0.0)
            W = p.m * G
            rows.append(np.array([s1.trim_deg(p, v[1]), v[0] / h.L,
                                  aux["fx"] / W, aux["hw"] / max(aux["dT"],
                                                                 1e-12),
                                  aux["Xh"] / W]))
        return np.array(rows)
    rb = roots(base)
    ob = P0.open_loop_nd(base, 1.0)
    worst = {}
    for lam, h in twins.items():
        rt = roots(h)
        ot = P0.open_loop_nd(h, lam)
        worst[lam] = dict(
            roots=float(np.abs(rt - rb).max()),
            open_state=float(np.abs(ot["state"] - ob["state"]).max()),
            open_meas=float(np.abs(ot["meas"] - ob["meas"]).max()))
        out(f"    lam {lam}: roots max nondimensional diff "
            f"{worst[lam]['roots']:.1e}; open loop state "
            f"{worst[lam]['open_state']:.1e}, measurements "
            f"{worst[lam]['open_meas']:.1e}")
    w = max(max(v.values()) for v in worst.values())
    out(f"    largest nondimensional Newton residual {max(resid):.1e}")
    check("twin", "Froude twins lam 1.5 / 0.6 on the candidate: every "
          "nondimensional output within 1e-4 (M3) and within the internal "
          "1e-10", w <= 1e-10, f"worst {w:.1e}")
    dump("twin.json", dict(base_roots=rb, worst=worst))
    flush("twin.txt")


# ================================================================ part nr
def hullb_n(n):
    """planing_b on as1 with n stations (the same specs, its own name)."""
    import dataclasses
    from hydro import hulls
    from hydro.vessel_spec import ParametricPlaningHull
    from sim import physics
    if n == 48:
        return hulls.get("planing_b@as1")
    hb = hulls.get("planing_b")
    f, mp, js, cl, nm = hb.specs()
    h = ParametricPlaningHull(f"planing_b_n{n}",
                              dataclasses.replace(f, n_stations=n), mp,
                              jet=js, closures=cl, numerics=nm,
                              u_design=hb.u_design, rho=hb.rho)
    return physics.variant(h, "as1")


def momentum_identity(p, u, v):
    """At the steady calm captive root v = (z, th) of plant p at speed u:
    the strips' horizontal momentum force as the plant sums it (v0's
    -sum dyn tau dx without the cross-flow drag, plus Xn), the same with
    the flux UNGATED (V Dm_a/Dt at every station, exits included), and the
    transom end term -u m_aT V_T tau_T it must telescope to (module
    docstring 'P1a normal_rate'). Steady calm: ett = q = 0, V = u tau."""
    from sim.planing_vessel import SP
    s1 = S1()
    p.captive_u = u
    _, aux = p.deriv(s1.state(u, v[0], v[1]), 0.0)
    Th = p.Th_rest + v[1]
    tau = p.kp + SP * Th
    d, V = aux["d"], aux["V"]
    A, c, ma, dma, _ = p._section(d)
    ctr = p._ctr(u)
    ma, dma = ma * ctr, dma * ctr
    dx = p.dx
    curv = ma * (-u * u * p.kpp)
    flux = aux["mom_full"]
    Xn = float(aux.get("Xn", 0.0))
    lhs = float(-((curv + flux) * tau).sum() * dx + Xn)
    # the ungated flux: _flux_dDt without its gate (u >= 0 here)
    flu = dma * V * V
    act, ycap, iunc = p._substrip(d, -tau)
    if act.size:
        km = p.k_a * 0.5 * np.pi * p.rho
        kc = 0.5 * np.pi / p.tanb
        Dma = (V[act] * (2.0 * km * kc * kc) * iunc + u * km * ycap) / dx
        flu[act] = V[act] * Dma
    lhs_u = float(-((curv + flu) * tau).sum() * dx + Xn)
    cf = 0.5 * p.rho * p.c_dc * 2.0 * c * V * np.abs(V) * ctr
    # the transom end term, at the transom edge
    g = p._geom
    tg = p._transom_geom()
    kpT = float(g.keel(np.array([g.edges[0]]))[1][0])
    tauT = kpT + SP * Th
    kc = 0.5 * np.pi / p.tanb
    cT = min(kc * max(aux["dT"], 0.0), tg.ycT)
    maT = p.k_a * 0.5 * np.pi * p.rho * cT * cT
    rhs = float(-u * maT * (u * tauT) * tauT)
    # the bow end of the same telescoped sum, +u m_aS V_S tau_S: zero where
    # the keel leaves the water before the stem; where the stem is wet the
    # chine floor (register P10) leaves the stem section a small m_a
    xe = np.array([g.edges[-1]])
    riseS, kpS, _ = g.keel(xe)
    tauS = float(kpS[0] + SP * Th)
    dS = float(-(p.Z_rest + v[0] + (-p.hull.vcg + float(riseS[0]))
                 + SP * float(g.edges[-1] - p.hull.lcg) * Th))
    cS = min(kc * max(dS, 0.0), float(g.chine(xe)[0]))
    stem = float(u * p.k_a * 0.5 * np.pi * p.rho * cS * cS * (u * tauS)
                 * tauS)
    # what normal_rate adds at this state (whether or not it is on)
    Xn_would = float((ma * V * u * p.kpp).sum() * dx)
    return dict(lhs=lhs, lhs_ungated=lhs_u, rhs=rhs, stem=stem,
                rhs_full=rhs + stem, d_stem=dS, Xn=Xn, Xn_would=Xn_would,
                gate=lhs - lhs_u, cf_x=float(-(cf * tau).sum() * dx),
                tau_max_deg=float(np.degrees(tau[d > 0].max()))
                if (d > 0).any() else 0.0,
                R=float(aux["fx"]), trim=s1.trim_deg(p, v[1]))


def part_nr():
    """normal_rate on a curved keel (P1a review, major 2): the registered
    M2 identity by station count, planing_b's resistance with and without
    the switch, and scarab195's transom moments (minor 8)."""
    from sim.planing_anyspeed import P1A_CANDIDATE, PlaningVesselASLin
    s1 = S1()
    out("P1a (nr)  M2: in steady calm flow the strips' horizontal momentum "
        "force (-sum dyn tau dx without the cross-flow drag, + Xn), its flux "
        "UNGATED, == the telescoped ends -u m_aT V_T tau_T + u m_aS V_S "
        "tau_S (transom; stem, where the keel is wet there); by station "
        "count. Form 1' (re-registered after this round's first run, "
        "review_fix/nr.txt; plan doc)")
    sw_nn = _without(P1A_CANDIDATE, "normal_rate")
    rows = []
    cases = (
        ("planing_b", hullb_n, {4.0: 1536, 6.0: 1536, 8.0: 1536,
                                10.0: 384, 14.0: 384, 20.0: 384}),
        ("scarab195", lambda n: (hulls_get(AS) if n == 48 else
                                 physics_variant(S1().hull_n(n))),
         {4.0: 384, 6.0: 384, 10.0: 384, 14.0: 384, 20.0: 384}))
    for hull_lab, mk, kmax in cases:
        ns_all = [n for n in (48, 96, 192, 384, 768, 1536)
                  if n <= max(kmax.values())]
        for lab, sw in (("with normal_rate", None),
                        ("without normal_rate", sw_nn)):
            guess = {}
            for n in ns_all:
                p = PlaningVesselASLin(mk(n), calm(), dt=0.02,
                                       test_switches=sw)
                for kn, nmax in kmax.items():
                    if n > nmax:
                        continue
                    u = kn * KN
                    g0 = guess.get(kn)
                    if g0 is None:
                        g0 = np.array(p.running_attitude(u))
                    v, r = s1.newton(p, u, g0)
                    guess[kn] = v
                    mi = momentum_identity(p, u, v)
                    scale = max(abs(mi["rhs_full"]), abs(mi["Xn_would"]))
                    mi.update(hull=hull_lab, sw=lab, n=n, kn=kn, resid=r,
                              res=mi["lhs_ungated"] - mi["rhs_full"],
                              scale=scale,
                              e_ungated=abs(mi["lhs_ungated"]
                                            - mi["rhs_full"]) / scale)
                    rows.append(mi)
    out("    hull       switch               kn     n | R N     trim  | "
        "lhs (gated)  lhs ungated  ends (transom + stem)   residual   "
        "rel(scale)  gate share  Xn(would)  max tau")
    for r in rows:
        out(f"    {r['hull']:10s} {r['sw']:20s} {r['kn']:4.0f} {r['n']:5d} | "
            f"{r['R']:7.1f} {r['trim']:5.2f} | {r['lhs']:+10.3f} "
            f"{r['lhs_ungated']:+11.4f} {r['rhs_full']:+11.4f} "
            f"({r['stem']:+.1e}) {r['res']:+10.2e}  {r['e_ungated']:9.2e}  "
            f"{r['gate']:+9.2f}  {r['Xn_would']:+8.2f}  "
            f"{r['tau_max_deg']:5.1f}")
    fits = {}
    ok_with = True
    for hull_lab in ("planing_b", "scarab195"):
        for kn in sorted({r["kn"] for r in rows if r["hull"] == hull_lab}):
            sel = [r for r in rows if r["hull"] == hull_lab and r["kn"] == kn
                   and r["sw"] == "with normal_rate"]
            e = {r["n"]: r["e_ungated"] for r in sel}
            nf = max(e)
            # the exact case: a straight keel (no curvature term at all)
            # wet at the stem at every station count
            straight_wet = all(r["d_stem"] > 0.0 and r["Xn_would"] == 0.0
                               for r in sel)
            xs = np.log(list(e))
            ys = np.log([max(x, 1e-300) for x in e.values()])
            order = float(-np.polyfit(xs, ys, 1)[0])
            if straight_wet:
                ok = max(e.values()) <= 1e-12
            else:
                ok = e[nf] <= 1e-3 and (order >= 1.5 or e[nf] <= 1e-6)
            fits[(hull_lab, kn)] = dict(e48=e[48], e_finest=e[nf],
                                        n_finest=nf, order=order,
                                        exact_case=straight_wet, ok=ok)
            ok_with &= ok
    out("    with normal_rate, ungated (relative to max(|ends|, |Xn|)): "
        + "; ".join(f"{h} {kn:g} kn: 48 {f['e48']:.1e}, {f['n_finest']} "
                    f"{f['e_finest']:.1e}, order {f['order']:.2f}"
                    + (" (straight keel, stem wet: exact case)"
                       if f["exact_case"] else "")
                    + ("" if f["ok"] else " OUTSIDE")
                    for (h, kn), f in fits.items()))
    check("nr", "M2 (form 1', re-registered): with normal_rate the ungated "
          "momentum force == the telescoped ends; straight keel with the "
          "stem wet: <= 1e-12 relative (exact discrete telescoping); "
          "otherwise <= 1e-3 of max(|ends|, |Xn|) at the finest station "
          "count (1536 for planing_b 4-8 kn, else 384) with fitted order "
          ">= 1.5 (or <= 1e-6 already)", ok_with,
          "outside: " + (", ".join(f"{h} {kn:g} kn" for (h, kn), f in
                                   fits.items() if not f["ok"]) or "none"))
    wo = [r for r in rows if r["sw"] == "without normal_rate"
          and r["hull"] == "planing_b"]
    out("    without normal_rate (recorded): residual against the term the "
        "switch would add at the same state, planing_b: " + "; ".join(
            f"{r['kn']:g} kn n {r['n']}: residual {r['res']:+.2f} N, "
            f"-Xn(would) {-r['Xn_would']:+.2f} N" for r in wo
            if r["n"] in (48, 384)))
    # planing_b's calm resistance with and without the switch, 48 stations
    out("    planing_b calm resistance (Newton roots, 48 stations): with / "
        "without normal_rate")
    res_b = []
    for lab, sw in (("with", None), ("without", sw_nn)):
        p = PlaningVesselASLin(hullb_n(48), calm(), dt=0.02,
                               test_switches=sw)
        g = None
        for kn in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 17.0, 20.0, 25.0,
                   30.0, 35.0, 40.0):
            u = kn * KN
            g0 = np.array(p.running_attitude(u)) if g is None else g
            v, r = s1.newton(p, u, g0)
            g = v
            mi = momentum_identity(p, u, v)
            res_b.append(dict(sw=lab, kn=kn, R=mi["R"], trim=mi["trim"],
                              Xn=mi["Xn"], resid=r))
    for kn in sorted({r["kn"] for r in res_b}):
        a_ = next(r for r in res_b if r["sw"] == "with" and r["kn"] == kn)
        b_ = next(r for r in res_b if r["sw"] == "without" and r["kn"] == kn)
        out(f"     {kn:4.0f} kn: R {a_['R']:7.1f} / {b_['R']:7.1f} N "
            f"({100 * (a_['R'] / b_['R'] - 1):+6.1f} %), trim "
            f"{a_['trim']:5.2f} / {b_['trim']:5.2f} deg, Xn {a_['Xn']:+7.1f} N")
    # scarab195: the transom switch's moment Mh and the missing z lever of
    # the tilted dynamic force (P1b), and the trim effect of Mh alone
    out("    scarab195 (preset): Mh, the x force of the dynamic strip force at"
        " keel height (the missing z lever, an estimate), trim effect of Mh")
    p = PlaningVesselASLin(hulls_get(AS), calm(), dt=0.02)
    q = PlaningVesselASLin(hulls_get(AS), calm(), dt=0.02)
    orig = q._x_hydro

    def no_m(*a, **k):
        X, M = orig(*a, **k)
        return X, 0.0
    q._x_hydro = no_m
    mom = []
    from sim.planing_vessel import SP
    g = gq = None
    for kn in (6.0, 10.0, 13.0, 16.0, 20.0, 25.0, 30.0):
        u = kn * KN
        v, _ = s1.newton(p, u, np.array(p.running_attitude(u)) if g is None
                         else g)
        g = v
        vq, _ = s1.newton(q, u, v if gq is None else gq)
        gq = vq
        p.captive_u = u
        _, aux = p.deriv(s1.state(u, v[0], v[1]), 0.0)
        Th = p.Th_rest + v[1]
        tau = p.kp + SP * Th
        A, c, ma, dma, _ = p._section(aux["d"])
        V = aux["V"]
        dyn = (ma * (-u * u * p.kpp) + aux["mom_full"]
               + 0.5 * p.rho * p.c_dc * 2.0 * c * V * np.abs(V))
        z0 = p.zk + SP * p.x * Th
        Mdz = float((z0 * (-dyn * tau)).sum() * p.dx)
        mom.append(dict(kn=kn, Mh=float(aux["Mh"]), Mdz=Mdz,
                        trim=s1.trim_deg(p, v[1]),
                        trim_noMh=s1.trim_deg(q, vq[1])))
        out(f"     {kn:4.0f} kn: Mh {aux['Mh']:+7.1f} N m, missing dynamic "
            f"z-lever ~{Mdz:+7.1f} N m; trim {mom[-1]['trim']:.3f} deg, "
            f"without Mh {mom[-1]['trim_noMh']:.3f} deg "
            f"({mom[-1]['trim'] - mom[-1]['trim_noMh']:+.3f})")
    dump("nr.json", dict(rows=rows, fits={f"{h} {kn:g}": f for (h, kn), f in
                                          fits.items()},
                         resistance_b=res_b, moments=mom))
    flush("nr.txt")


def hulls_get(key):
    from hydro import hulls
    return hulls.get(key)


def physics_variant(h):
    from sim import physics
    return physics.variant(h, "as1", name=h.name)


# ================================================================ part r2
def captive_R(p, kns, start):
    """Captive calm resistance (the force holding the speed, air included)
    at the Newton roots, continued along kns."""
    s1 = S1()
    g = np.array(start, float)
    R = []
    for kn in kns:
        u = kn * KN
        v, r = s1.newton(p, u, g)
        g = v
        p.captive_u = u
        _, aux = p.deriv(s1.state(u, v[0], v[1]), 0.0)
        R.append(float(aux["fx"]))
    return np.array(R)


def count_roots(T, R, kns):
    """Sign changes of T - R along kns: (count, the speeds, stable flags:
    d(T - R)/du < 0 at a stable root)."""
    f = T - R
    idx = np.nonzero(np.sign(f[1:]) != np.sign(f[:-1]))[0]
    roots, stab = [], []
    for i in idx:
        w = f[i] / (f[i] - f[i + 1])
        roots.append(kns[i] + w * (kns[i + 1] - kns[i]))
        stab.append(bool(f[i + 1] < f[i]))
    return len(idx), roots, stab


def part_r2():
    """R2 with thrust as the parameter (P1a review, majors 4 and 13): the
    speed-held checks (census, continuation, stepped sweeps) cannot see a
    fold of the thrust-to-speed map."""
    out("P1a (r2)  R2 with the thrust as the parameter: roots of T(u; c) - "
        "R(u) (captive R on a 0.25 kn grid, the jet's net thrust at command"
        " c t_max) on a c grid of 0.002; a_b(C_V) frozen as a diagnostic; "
        "a slow free-surge thrust ramp up and down (open loop)")
    kns = np.arange(0.25, 45.0001, 0.25)
    p = plant(AS, captive_u=kns[0] * KN)
    v0, _ = eq_rest(plant(AS, captive_u=0.0))
    jet = p.prop
    prime = jet.prime
    jet.prime = 1.0
    cs = np.round(np.arange(0.002, 1.0 + 1e-9, 0.002), 6)
    Tm = np.array([[jet.forces(c * jet.t_max, 0.0, kn * KN)[0] for kn in kns]
                   for c in cs])
    jet.prime = prime
    res = {}
    for lab, ab in (("a_b(C_V) as is", None), ("a_b frozen at 1",
                                                1.0),
                    ("a_b frozen at 0.5", 0.5)):
        q = plant(AS, captive_u=kns[0] * KN)
        if ab is not None:
            q._a_b = (lambda u, ab=ab: ab)
        R = captive_R(q, kns, v0)
        counts = []
        for i, c in enumerate(cs):
            n_, roots, stab = count_roots(Tm[i], R, kns)
            counts.append(dict(c=float(c), n=n_, roots=roots, stable=stab))
        multi = [x for x in counts if x["n"] > 1]
        dRdu = np.diff(R) / np.diff(kns * KN)
        falling = [(float(kns[i]), float(kns[i + 1])) for i in
                   range(len(dRdu)) if dRdu[i] < 0]
        res[lab] = dict(R=R, counts=counts, multi=multi, falling=falling)
        band = ((multi[0]["c"], multi[-1]["c"]) if multi else None)
        out(f"  [{lab}] R(u) falls on {len(falling)} grid intervals"
            + (f" ({falling[0][0]:.2f}..{falling[-1][1]:.2f} kn)"
               if falling else "")
            + f"; c with more than one root: {len(multi)}"
            + (f", c {band[0]:.3f}..{band[1]:.3f}: roots "
               + "; ".join(f"c {x['c']:.3f}: " + ", ".join(
                   f"{rr:.2f} kn ({'s' if st else 'u'})" for rr, st in
                   zip(x["roots"], x["stable"])) for x in multi[::max(
                       1, len(multi) // 4)]) if multi else ""))
        out("     R(u): " + "  ".join(f"{k:g}:{r:.0f}" for k, r in
                                      zip(kns[::8], R[::8])))
    nom = res["a_b(C_V) as is"]
    check("r2", "R2 (thrust map, registered): exactly one root of T(u; c) - "
          "R(u) on 0.25-45 kn for every c on the 0.002 grid that has one",
          not nom["multi"],
          f"{len(nom['multi'])} c values with 3 roots"
          + (f" (c {nom['multi'][0]['c']:.3f}..{nom['multi'][-1]['c']:.3f})"
             if nom["multi"] else ""))
    check("r2", "diagnostic (recorded): with a_b frozen at 1 the thrust map "
          "has one root at every c", True,
          f"a_b = 1: {len(res['a_b frozen at 1']['multi'])} c values with "
          f"several roots, R falls on {len(res['a_b frozen at 1']['falling'])}"
          f" intervals; a_b = 0.5: {len(res['a_b frozen at 0.5']['multi'])}"
          f" c values, R falls on "
          f"{len(res['a_b frozen at 0.5']['falling'])} intervals")
    # the slow free-surge ramp: c 0.15 -> 0.35 -> 0.15 at 0.0002 per s, 60 s
    # holds; up and down speeds compared at the same c
    pf = plant(AS)
    pf.captive_u = None
    s = pf.initial_state(0.0)
    dt = pf.dt
    c0, c1, rate, hold = 0.15, 0.35, 2e-4, 60.0
    t_r = (c1 - c0) / rate
    T = hold + t_r + hold + t_r + hold

    def cmd(t):
        if t < hold:
            return c0, 0
        t -= hold
        if t < t_r:
            return c0 + rate * t, 1
        t -= t_r
        if t < hold:
            return c1, 0
        t -= hold
        if t < t_r:
            return c1 - rate * t, -1
        return c0, 0
    rec = []
    n = int(round(T / dt))
    t = 0.0
    for i in range(n):
        c, dr = cmd(t)
        s = pf.step(s, t, c * pf.prop.t_max, 0.0, dt)
        t += dt
        if i % 5 == 0:
            rec.append((t, c, dr, s[6] / KN, s1_trim(pf, s[4])))
        if not np.all(np.isfinite(s)):
            break
    rec = np.array(rec)
    up, dn = rec[rec[:, 2] == 1], rec[rec[:, 2] == -1]
    grid = np.round(np.arange(c0 + 0.005, c1 - 0.005 + 1e-9, 0.005), 4)
    ku = np.interp(grid, up[:, 1], up[:, 3])
    kd = np.interp(grid, dn[::-1, 1], dn[::-1, 3])
    diff = ku - kd
    jumps_u = np.abs(np.diff(ku)).max()
    out("     slow ramp (c, kn up, kn down, up - down):")
    for c, a_, b_ in zip(grid[::2], ku[::2], kd[::2]):
        out(f"      c {c:.3f}: {a_:6.2f} {b_:6.2f} {a_ - b_:+6.2f}")
    j = int(np.argmax(np.abs(diff)))
    check("r2", "R2 (thrust map, registered): slow free-surge ramp c 0.15 -> "
          "0.35 -> 0.15 at 2e-4 per s: up and down speeds at the same c "
          "within 0.5 kn (a fold shows as a jump of several knots)",
          np.abs(diff).max() <= 0.5,
          f"largest up - down {diff[j]:+.2f} kn at c {grid[j]:.3f}; largest "
          f"speed change between c samples 0.005 apart (up) {jumps_u:.2f} kn")
    dump("r2.json", dict(kns=kns, cs=cs, variants={
        k: dict(R=v["R"], multi=v["multi"], falling=v["falling"])
        for k, v in res.items()}, ramp=rec, ramp_grid=grid, up=ku, down=kd))
    flush("r2.txt")


def s1_trim(p, th):
    return S1().trim_deg(p, th)


# =========================================================== part e2speed
def part_e2speed():
    """E2 standing check (P1a review, minor 7): the one-step map of the
    verlet scheme and of Heun on the plant's linearised heave/pitch (the
    preset, calm, Newton root) at h = 0.02 s, against exp(lambda h)."""
    s1 = S1()
    h = 0.02
    out("P1a (e2speed)  E2: verlet and Heun on the linearised heave/pitch at"
        " 0-40 kn, h 0.02 s: modal growth rate ln|mu|/h against Re lambda")
    rows = []
    for kn in (0.0, 3.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0):
        u = kn * KN
        p = plant(AS, captive_u=u)
        if kn == 0.0:
            v, _ = eq_rest(p)
        else:
            v, _ = s1.newton(p, u, np.array(p.running_attitude(u)))
        J = s1.jac4(p, u, v[0], v[1])
        K, D = J[2:, :2], J[2:, 2:]

        def verlet(x):
            qq, w = x[:2], x[2:]
            a1 = K @ qq + D @ w
            wE = w + h * a1
            pos = qq + h * w + 0.5 * h * h * a1
            a2 = K @ pos + D @ wE
            return np.concatenate([pos, w + 0.5 * h * (a1 + a2)])

        def heun(x):
            xe = x + h * (J @ x)
            return x + 0.5 * h * (J @ x + J @ xe)
        lam = np.linalg.eigvals(J)
        ex = np.exp(lam * h)
        row = dict(kn=kn, lam=lam)
        for lab, fn in (("verlet", verlet), ("heun", heun)):
            M = np.column_stack([fn(e) for e in np.eye(4)])
            mu = np.linalg.eigvals(M)
            used, gr = set(), []
            for e, l_ in zip(ex, lam):
                jj = min((k for k in range(4) if k not in used),
                         key=lambda k: abs(mu[k] - e))
                used.add(jj)
                gr.append((math.log(abs(mu[jj])) / h, float(l_.real),
                           float(abs(mu[jj]))))
            row[lab] = gr
        row["oscillatory"] = [bool(abs(l_.imag) > 1e-6) for l_ in lam]
        rows.append(row)
        out(f"    {kn:4.1f} kn  lambda " + ", ".join(
            f"{l_.real:+.3f}{l_.imag:+.3f}i" for l_ in lam))
        for lab in ("verlet", "heun"):
            out(f"        {lab:6s} ln|mu|/h (Re lambda): " + ", ".join(
                f"{a:+.4f} ({b:+.4f})" for a, b, _ in row[lab])
                + f"   max |mu| {max(m for _, _, m in row[lab]):.6f}")
    stable = all(m < 1.0 for r in rows for a, b, m in r["verlet"] if b < 0)
    worst_osc = max(a - b for r in rows for (a, b, _), o in
                    zip(r["verlet"], r["oscillatory"]) if o)
    check("e2speed", "E2 (form 3', re-registered after this round's first "
          "run, review_fix/e2speed.txt, whose form -- every mode -- failed on"
          " the overdamped real modes): verlet on the linearised heave/pitch "
          "at 0-40 kn, h 0.02 s: |mu| < 1 for every mode with Re lambda < 0;"
          " the OSCILLATORY modes (E2's wave band) show no spurious growth: "
          "ln|mu|/h - Re lambda <= 0.01 1/s", stable and worst_osc <= 0.01,
          f"oscillatory: largest ln|mu|/h - Re lambda {worst_osc:+.2e} 1/s; "
          f"all modes |mu| < 1: {stable}")
    # recorded (not a gate): the same measure split by mode kind -- the
    # oscillatory modes (E2's wave band) and the overdamped real modes
    for kind, sel in (("oscillatory", True), ("real (overdamped)", False)):
        vals = [(a - b, r["kn"]) for r in rows for (a, b, _), o in
                zip(r["verlet"], r["oscillatory"]) if o == sel]
        hv = [(a - b) for r in rows for (a, b, _), o in
              zip(r["heun"], r["oscillatory"]) if o == sel]
        if vals:
            w_, k_ = max(vals)
            out(f"    {kind} modes: largest ln|mu|/h - Re lambda verlet "
                f"{w_:+.2e} 1/s (at {k_:g} kn), heun {max(hv):+.2e} 1/s")
    dump("e2speed.json", rows)
    flush("e2speed.txt")


# ============================================================ part r2fine
def hold_job(arg):
    """Free surge: settle at command c_from (from its static root) for
    `pre` s, then hold c_to for `hold` s. Returns the speed history."""
    warnings.filterwarnings("ignore")
    c_from, u_from, z_from, th_from, c_to, pre, hold = arg
    p = plant(AS)
    p.captive_u = None
    s = p.initial_state(0.0)
    s[2], s[4], s[6] = z_from, th_from, u_from
    s[12] = c_from * p.prop.t_max
    dt = p.dt
    t = 0.0
    us = []
    for i in range(int(round((pre + hold) / dt))):
        c = c_from if t < pre else c_to
        s = p.step(s, t, c * p.prop.t_max, 0.0, dt)
        t += dt
        if i % 5 == 0:
            us.append((t, s[6] / KN))
    return np.array(us)


def part_r2fine():
    """R2 with thrust as the parameter, resolved (registered after this
    round's first r2 run; plan doc): the near-neutral stretch of T - R
    (the captive R on a 0.05 kn grid by Newton, 12-15 kn; c on a 1e-5
    grid, 0.230-0.240): the fold's width in c and u, the relaxation time
    M / |d(T - R)/du| along the branch; and the decisive time-domain test:
    from the free-surge equilibria at c 0.22 (below) and 0.26 (above), hold
    c = 0.24 for 1500 s -- one equilibrium there means both end at the same
    speed (registered: within 0.1 kn of each other and of the static root);
    the same at the fold's c (recorded)."""
    from studies.parallel_runner import run_parallel
    s1 = S1()
    out("P1a (r2fine)  the thrust map resolved near its steep part, and "
        "holds at fixed thrust from below and above")
    kf = np.round(np.arange(12.0, 15.0001, 0.05), 4)
    p = plant(AS, captive_u=kf[0] * KN)
    g = np.array(p.running_attitude(kf[0] * KN))
    R, roots_zt = [], []
    for kn in kf:
        u = kn * KN
        v, r = s1.newton(p, u, g)
        g = v
        p.captive_u = u
        _, aux = p.deriv(s1.state(u, v[0], v[1]), 0.0)
        R.append(float(aux["fx"]))
        roots_zt.append(v)
    R = np.array(R)
    jet = p.prop
    prime, jet.prime = jet.prime, 1.0
    cs = np.round(np.arange(0.230, 0.240 + 1e-12, 1e-5), 6)
    uf = kf * KN
    counts = []
    for c in cs:
        T = np.array([jet.forces(c * jet.t_max, 0.0, x)[0] for x in uf])
        n_, rts, st = count_roots(T, R, kf)
        counts.append((float(c), n_, rts, st))
    Tm = np.array([jet.forces(0.235 * jet.t_max, 0.0, x)[0] for x in uf])
    jet.prime = prime
    slope = np.diff(Tm - R) / np.diff(uf)
    pos = [(float(kf[i]), float(kf[i + 1]), float(slope[i]))
           for i in range(len(slope)) if slope[i] > 0]
    multi = [x for x in counts if x[1] > 1]
    out(f"    R(u) 12-15 kn (0.05 kn, Newton): " + " ".join(
        f"{k:.2f}:{r:.1f}" for k, r in zip(kf[::4], R[::4])))
    out(f"    d(T - R)/du > 0 at c 0.235 on {len(pos)} intervals: "
        + ", ".join(f"{a:.2f}-{b:.2f} kn {s_:+.2f} N s/m"
                    for a, b, s_ in pos))
    if multi:
        out(f"    c with three roots: {len(multi)} of {len(cs)} on the 1e-5 "
            f"grid, c {multi[0][0]:.5f}..{multi[-1][0]:.5f}; e.g. c "
            f"{multi[len(multi) // 2][0]:.5f}: roots " + ", ".join(
                f"{r_:.3f} kn ({'stable' if st else 'unstable'})" for r_, st in
                zip(multi[len(multi) // 2][2], multi[len(multi) // 2][3])))
    else:
        out("    c with three roots on the 1e-5 grid: none")
    check("r2fine", "R2 thrust map resolved (recorded): the c window with "
          "three equilibria and the stretch where d(T - R)/du >= 0", True,
          (f"{len(multi)} c values (window {multi[0][0]:.5f}..{multi[-1][0]:.5f}"
           f", width {multi[-1][0] - multi[0][0] + 1e-5:.5f})" if multi else
           "no c with three roots") + f"; positive slope on {len(pos)} "
          "intervals" + (f", {pos[0][0]:.2f}-{pos[-1][1]:.2f} kn, at most "
                         f"{max(x[2] for x in pos):+.2f} N s/m" if pos
                         else ""))
    # static roots (the captive R on part r2's 0.25 kn grid) and the holds
    kns_ = np.arange(0.25, 45.0001, 0.25)
    Rc = captive_R(plant(AS, captive_u=0.25 * KN), kns_,
                   eq_rest(plant(AS, captive_u=0.0))[0])

    def roots_at(c):
        jet.prime = 1.0
        T = np.array([jet.forces(c * jet.t_max, 0.0, x * KN)[0]
                      for x in kns_])
        jet.prime = prime
        return count_roots(T, Rc, kns_)[1]

    from_lo, from_hi = 0.22, 0.26
    starts = {}
    for c in (from_lo, from_hi):
        rts = roots_at(c)
        ur = rts[0] * KN
        pc = plant(AS, captive_u=ur)
        v, _ = s1.newton(pc, ur, np.array(pc.running_attitude(ur)))
        starts[c] = (ur, v[0], v[1])
        out(f"    start at c {c:.3f}: static root {rts[0]:.3f} kn")
    c_fold = (0.5 * (multi[0][0] + multi[-1][0]) if multi else 0.235)
    jobs = [(c, *starts[c], c_to, 200.0, 1500.0) for c_to in (0.24, c_fold)
            for c in (from_lo, from_hi)]
    res = run_parallel(hold_job, jobs, gb_per_job=0.3, reserve_gb=3.0)
    fin = {}
    for job, r in zip(jobs, res):
        if isinstance(r, Exception):
            check("r2fine", "hold run ran", False, repr(r))
            continue
        last = r[r[:, 0] >= r[-1, 0] - 60.0]
        prev = r[(r[:, 0] >= r[-1, 0] - 360.0) & (r[:, 0] < r[-1, 0] - 300.0)]
        fin[(job[0], job[4])] = dict(kn=float(last[:, 1].mean()),
                                     drift_kn_per_300s=float(
                                         last[:, 1].mean()
                                         - prev[:, 1].mean()))
        f_ = fin[(job[0], job[4])]
        out(f"    hold c {job[4]:.5f} from c {job[0]:.2f}: speed after 1500 s"
            f" {f_['kn']:.3f} kn (change over the last 300 s "
            f"{f_['drift_kn_per_300s']:+.4f} kn)")
    r24 = roots_at(0.24)
    a_ = fin.get((from_lo, 0.24))
    b_ = fin.get((from_hi, 0.24))
    ok = (a_ is not None and b_ is not None and abs(a_["kn"] - b_["kn"])
          <= 0.1 and len(r24) == 1 and abs(a_["kn"] - r24[0]) <= 0.1)
    check("r2fine", "R2 (registered, decisive): free surge held at c 0.24 "
          "for 1500 s from the equilibria at c 0.22 and 0.26 ends at the "
          "same speed within 0.1 kn, and at the static root within 0.1 kn "
          "(no hysteresis there)", ok,
          f"from below {a_['kn'] if a_ else float('nan'):.3f} kn, from above"
          f" {b_['kn'] if b_ else float('nan'):.3f} kn, static root "
          f"{r24}")
    dump("r2fine.json", dict(kns=kf, R=R, counts=[(c, n_) for c, n_, _, _
                                                  in counts],
                             multi=multi, positive_slope=pos,
                             holds={f"{k[0]}->{k[1]}": v for k, v in
                                    fin.items()}, c_fold=c_fold))
    flush("r2fine.txt")


# ============================================================= part hullb
def part_hullb():
    """S1's standing checks on planing_b on the preset (P1a review, majors
    2 and 18b): studies/anyspeed_p0_check.part_hullb_s1 (census, sweep,
    decay, D55; P0's criteria), filed under <out>/hullb_s1/, then the sweep
    judged by P1's registered criterion (learn/meta/ANYSPEED_PLAN, P0
    review: a speed whose trim p-p exceeds 0.01 deg is still decaying if
    the last 10 s window's p-p is below half the previous window's, or if
    the root's eigenvalues all have negative real parts)."""
    from studies import anyspeed_p0_check as P0
    s1 = S1()
    P0.OUT = os.path.join(OUT, "hullb_s1")
    P0.STAMP_LINE = stamp_line()
    out("P1a (hullb)  S1's standing checks on planing_b@as1 (preset) -> "
        f"{P0.OUT}")
    P0.part_hullb_s1()
    for r in P0.RESULTS.get("hullb_s1", []):
        CHECKS.append(dict(r, part="hullb"))
        out(f"  [{'ok' if r['ok'] else 'FAIL'}] hullb (P0's criteria): "
            f"{r['name']}  -- {r['detail']}")
    with open(os.path.join(P0.OUT, "hullb_s1.json")) as f:
        R = json.load(f)["results"]
    up, dn = R["sweep"]["up"], R["sweep"]["down"]
    flagged = sorted({r["kn"] for r in up + dn if r["trim_pp"] > 0.01
                      or r["z_pp"] > 1e-3})
    verdict = []
    for kn in flagged:
        rr = [r for r in up + dn if r["kn"] == kn]
        ratios = [r["trim_pp"] / max(r.get("trim_pp_prev", np.nan), 1e-300)
                  for r in rr]
        u = kn * KN
        p = P0.hullb_plant(captive_u=u)
        v, res_ = s1.newton(p, u, np.array(p.running_attitude(u)))
        _, mre, _ = s1.modes(s1.jac4(p, u, v[0], v[1]))
        decaying = all(x < 0.5 for x in ratios) or mre < 0.0
        verdict.append(dict(kn=kn, ratios=ratios, max_re=mre,
                            decaying=bool(decaying)))
        out(f"    flagged {kn:g} kn: last/previous window trim p-p "
            + ", ".join(f"{x:.3f}" for x in ratios)
            + f"; root's largest Re lambda {mre:+.4f} 1/s -> "
            + ("decaying transient" if decaying else "NOT decaying"))
    check("hullb", "P1's registered sweep criterion: every speed flagged by "
          "the fixed 0.01 deg threshold is a decaying transient (window "
          "ratio < 0.5, or all eigenvalues of the root in the left half "
          "plane)", all(x["decaying"] for x in verdict),
          f"flagged {flagged}" if flagged else "nothing flagged")
    dump("hullb_p1.json", dict(flagged=flagged, verdict=verdict))
    flush("hullb.txt")


# ============================================================ part p6attr
P6_SETS = {
    "v0": None,
    "pre": ("dDt_full", "chine_exact_all", "kpp_strip"),
    "+transom": ("dDt_full", "chine_exact_all", "kpp_strip", "transom"),
    "+transom+normal_rate": ("dDt_full", "chine_exact_all", "kpp_strip",
                             "transom", "normal_rate"),
    "+verlet only": ("dDt_full", "chine_exact_all", "kpp_strip", "verlet"),
    "candidate": ("dDt_full", "chine_exact_all", "kpp_strip", "transom",
                  "normal_rate", "verlet"),
}


def p6_job(lab):
    """Fridsma 1969 regular waves (P6) on one switch set: per (case, lam)
    the four quantities and which are within the ACCEPTANCE tolerances."""
    warnings.filterwarnings("ignore")
    from sim import physics
    from sim.planing_anyspeed import PlaningVesselAS
    from studies import exp_planing as EP
    s1 = S1()
    sw = P6_SETS[lab]
    out_ = []
    for case in ("A", "B"):
        h, V = EP.fridsma_hull(case, tow="nozzle")
        hv = physics.variant(h, "as1")
        for lam, meas in EP.FRIDSMA[case].items():
            hb = 0.111 * h.b
            if sw is None:
                r = EP.regular_wave(h, V, lam, hb, T=16.0, dt=0.004)
            else:
                def make(hull, sea, dt, V_, hv=hv, sw=sw):
                    return PlaningVesselAS(hv, sea, dt=dt, captive_u=V_,
                                           test_switches=sw)
                r = s1.regular_wave_on(make, h, V, lam, hb, T=16.0, dt=0.004)
            tol = (0.2, 0.2, 0.3, 0.3)
            ins = [bool(abs(r[j] / meas[j] - 1) <= tol[j]) for j in range(4)]
            out_.append(dict(case=case, lam=lam, r=[float(x) for x in r],
                             meas=list(meas), within=ins))
    return lab, out_


def part_p6attr():
    """V1 attribution of the Fridsma P6 count (P1a review, minor 10), as
    p7_attribution.txt did for P7: v0, the preset before P1a, +transom,
    +transom+normal_rate, +verlet only, the candidate (one process each)."""
    from studies.parallel_runner import run_parallel
    out("P1a (p6attr)  Fridsma 1969 regular waves (P6), quantities within "
        "the ACCEPTANCE tolerances (heave, pitch +-20 %; CG, bow "
        "acceleration +-30 %), by switch set")
    res = run_parallel(p6_job, list(P6_SETS), gb_per_job=0.3, reserve_gb=3.0)
    tab = {}
    for r in res:
        if isinstance(r, Exception):
            check("p6attr", "every job ran", False, repr(r))
            continue
        lab, rows = r
        tab[lab] = rows
        n_in = sum(sum(x["within"]) for x in rows)
        out(f"    {lab:22s}: {n_in} of {4 * len(rows)} within; by quantity "
            f"(heave, pitch, CG, bow): " + ", ".join(
                str(sum(x["within"][j] for x in rows)) for j in range(4)))
    labs = [l_ for l_ in P6_SETS if l_ in tab]
    out("    per case (case lam: heave pitch CG bow, each set):")
    for i, row in enumerate(tab[labs[0]]):
        out(f"     {row['case']} {row['lam']}: meas "
            + " ".join(f"{m:.2f}" for m in row["meas"]) + " | "
            + " | ".join(f"{l_}: " + " ".join(
                f"{x:.2f}{'*' if w else ''}" for x, w in
                zip(tab[l_][i]["r"], tab[l_][i]["within"])) for l_ in labs))
    check("p6attr", "P6 attribution by switch set (recorded)", bool(tab),
          ", ".join(f"{l_} {sum(sum(x['within']) for x in tab[l_])}/44"
                    for l_ in labs))
    dump("p6attr.json", tab)
    flush("p6attr.txt")


# ============================================================ part golden
def golden_job(arg):
    """anyspeed_regress.golden_runs on the candidate preset, in a fresh
    process (spawned): N2 reproducibility. Returns the path written."""
    warnings.filterwarnings("ignore")
    idx, path = arg
    use_candidate()
    from hydro import hulls
    from sim import physics
    from studies import anyspeed_regress as AR
    from studies.anyspeed_regress_as import ON_EXTRA
    sel = (lambda p: physics.gz_twin(hulls.get(AS).plant(p.sea, dt=p.dt),
                                     record=True))
    g = AR.golden_runs(sel, extra=ON_EXTRA)
    np.savez(path, **g)
    return path


REF_DIR = OUT            # the candidate's stored reference (not --out)
WRITE_REF = False        # --write


def part_golden():
    """N2: the anyspeed_regress golden runs (task sea and calm, 0-25 kn
    starts, 300 steps, moving thrust and nozzle) on the candidate preset in
    two separate processes: identical to the last bit, AND identical to the
    stored reference golden_as_<candidate>.npz (studies/_cache/anyspeed/
    p1a/; the record the P1a review reproduced): any difference fails. The
    reference is written only with --write (and refused if it exists)."""
    import hashlib
    import shutil
    from sim.planing_anyspeed import P1A_CANDIDATE
    from studies.parallel_runner import run_parallel
    out("P1a (golden)  N2: candidate preset, two processes, against the "
        "stored reference")
    tag = "+".join(P1A_CANDIDATE)
    os.makedirs(OUT, exist_ok=True)
    paths = [os.path.join(OUT, f"golden_as_{tag}_proc{i}.npz")
             for i in (1, 2)]
    res = run_parallel(golden_job, list(enumerate(paths)), gb_per_job=0.4,
                       reserve_gb=3.0)
    bad = [r for r in res if isinstance(r, Exception)]
    if bad:
        check("golden", "both golden processes ran", False, bad)
        flush("golden.txt")
        return
    with np.load(paths[0]) as fa, np.load(paths[1]) as fb:
        a = {k: fa[k] for k in fa.files}
        b = {k: fb[k] for k in fb.files}
    same = set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)
    finite = all(np.all(np.isfinite(a[k])) for k in a)
    check("golden", "candidate golden runs identical in two processes, all "
          "finite", same and finite, f"{len(a)} arrays")
    ref = os.path.join(REF_DIR, f"golden_as_{tag}.npz")
    if WRITE_REF:
        if os.path.exists(ref):
            raise SystemExit(f"{ref} exists; delete it first")
        shutil.copyfile(paths[0], ref)
        out(f"    reference written: {ref}")
    if not os.path.exists(ref):
        check("golden", "the stored reference exists", False,
              f"{ref}: write it with --write")
    else:
        with np.load(ref) as fr:
            R = {k: fr[k] for k in fr.files}
        diff = sorted(k for k in set(a) | set(R) if k not in a or k not in R
                      or not np.array_equal(a[k], R[k]))
        with open(ref, "rb") as f:
            h = hashlib.sha1(f.read()).hexdigest()
        check("golden", f"candidate golden runs == the stored reference "
              f"{os.path.basename(ref)} (sha1 {h[:8]}), every array",
              not diff, f"{len(diff)} arrays differ {diff[:3]}")
    for pth in paths:
        os.remove(pth)
    flush("golden.txt")


# ========================================================== part steptime
def part_steptime():
    from studies import anyspeed_p0_check as P0
    P0.OUT = OUT
    P0.STAMP_LINE = stamp_line()
    P0.part_steptime()
    for r in P0.RESULTS.get("steptime", []):
        CHECKS.append(dict(r, part="steptime"))


# ================================================================ s1 parts
def run_s1_part(part):
    s1 = S1()
    s1.OUT = os.path.join(OUT, "s1_rerun")
    s1.TOP_KN = 40.0
    s1.STAMP_LINE = stamp_line()
    t0 = time.time()
    print(f"S1 part {part} on the P1a candidate preset -> {s1.OUT}",
          flush=True)
    getattr(s1, f"part_{part}")()
    print(f"done in {time.time() - t0:.0f} s", flush=True)


def _sub(arg):
    kind, name, out_dir = arg
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    flag = "--s1part" if kind == "s1" else "--parts"
    with open(os.path.join(out_dir, f"{kind}_{name}_stdout.log"), "w") as log:
        rc = subprocess.run([sys.executable, "-m",
                             "studies.anyspeed_p1a_check", flag, name,
                             "--out", out_dir],
                            cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    return kind, name, rc, time.time() - t0


PARTS = ("m2", "rest", "fk", "energy", "astern", "conv", "sens", "v1",
         "twin", "golden", "steptime", "nr", "r2", "r2fine", "e2speed",
         "hullb", "p6attr")


def main():
    global PRE_P1A, OUT, STAMP, WRITE_REF
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="")
    ap.add_argument("--jobs", default="", help="parts, one process each")
    ap.add_argument("--s1", default="", help="S1 parts, one process each")
    ap.add_argument("--s1part", default="")
    ap.add_argument("--out", default=OUT, help="output directory")
    ap.add_argument("--write", action="store_true",
                    help="part golden: write the stored reference (refused "
                    "if it exists)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    OUT = os.path.abspath(a.out)
    WRITE_REF = a.write
    if a.jobs or a.s1:
        from studies.parallel_runner import run_parallel
        jobs = ([("p1a", x, OUT) for x in a.jobs.split(",") if x]
                + [("s1", x, OUT) for x in a.s1.split(",") if x])
        t0 = time.time()
        res = run_parallel(_sub, jobs, gb_per_job=0.4, reserve_gb=3.0)
        for r in res:
            print(f"  {r}", flush=True)
        print(f"all done in {time.time() - t0:.0f} s wall", flush=True)
        return
    PRE_P1A = use_candidate()
    STAMP = stamp_all()
    if a.s1part:
        run_s1_part(a.s1part)
        return
    t0 = time.time()
    for part in [x for x in a.parts.split(",") if x]:
        if part not in PARTS:
            raise SystemExit(f"unknown part {part}")
        globals()[f"part_{part}"]()
    tag = "_".join(x for x in a.parts.split(",") if x)
    dump(f"checks_{tag}.json", CHECKS)
    nf = sum(not c["ok"] for c in CHECKS)
    print(f"{len(CHECKS)} checks, {nf} failed, {time.time() - t0:.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
