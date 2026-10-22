#!/usr/bin/env python3
"""
Verification of step S5a of the any-speed plant (sim/planing_anyspeed.py,
switches heel_hydro and lateral_cross; learn/meta/ACCEPTANCE_2026-10-06.md).
The checks and their bounds were registered before any of them ran, in
learn/meta/ANYSPEED_PLAN_2026-10-05.md, section '### S5a' (the H / L
numbers below are those items).

Candidates: hh = the as1 preset (P1c) + heel_hydro; hx = hh + lateral_phi0
+ lateral_cross. Every .txt output starts with a stamp line (switches,
physics sha1s, the as1 code sha1, this script's sha1).

Parts (outputs in studies/_cache/anyspeed/s5a/, or --out):
  h1     H1   polygon moments against the closed forms and an independent
              clipping (Sutherland-Hodgman + vertex formulas)
  h2     H2   the plant's own righting arm at rest against the frozen GZ
              table (scarab195, planing_b)
  h3     H3   upright: polygons forced vs P1c's closed forms; continuity at
              phi -> 0; the upright branch bit for bit P1c's
  h4     H4   R3 (P1a's rest part on the candidate, plus a 10 deg heel start)
  h5     H5   energy at rest, dissipation off (hh)
  sym    H6   mirror symmetry
  twin   H7   Froude twins (P1a's part twin)
  h8     H8   roll stability of the calm branch (static stiffness, lateral
              eigenvalues against P1c's)
  h9     H9   large-angle capsize: signs at rest, recovery at 25 kn
  h10    H10  steady turns: heel direction and size
  e1     H11  step time (P0's interleaved method)
  l1     L1   2D: impulse (Lagrange) formula vs direct pressure integration
  l1p    L1'  the plant's cross-plane strip forces vs finite differences
  l4     L4   energy at rest, dissipation off (hx), integrator-aware bound
  l8     L8   roll natural frequency at rest vs Capytaine (consistency);
              the BEM results cached by mesh hash, 4 frequencies (review)
  goldens H12 the goldens of the sets S5a changed (review: calm runs with
              the nozzle at 5 deg), two processes, written once

The S5a review's round (plan 'S5a：复查后的修正'; outputs in
studies/_cache/anyspeed/s5a/r2/, the first round's left in s5a/):
  l1pp   L1'+ the cross-plane strip forces vs the Lagrangian of ONE kinetic
              energy (the roll row with the heave equation's m33), n0 by
              differencing nu at fixed body x
  stf    STF  the lateral diffraction at speed vs the Salvesen-Tuck-
              Faltinsen strip formula (encounter frequency + transom)
  hb1    HB1  zero heel: hh's first-order wave forces vs P1c's
  e0     E0   the 2D table's even entries smooth at zero heel
  ga4    G-A4 the low-speed sway-yaw mode with the sway added mass scaled
              to rigid-lid / 3D zero-frequency sums (A4's attribution)
  H3(b) checks the roll row, H8 takes the registered 1e-4 step and the
  variants (hh_ab, hx_e33, hx_none), H10 the variants.

    .venv/Scripts/python.exe -m studies.anyspeed_s5a_check --parts h1,h3
    .venv/Scripts/python.exe -m studies.anyspeed_s5a_check --cand hx --parts sym
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np                      # noqa: E402

OUT = os.path.join(HERE, "_cache", "anyspeed", "s5a", "r2")
S5A_DIR = os.path.join(HERE, "_cache", "anyspeed", "s5a")
KN = 0.514444
G = 9.81
LINES = []
CHECKS = []
AS = "scarab195@as1"
HB = "planing_b@as1"
STAMP = None
CAND = "hh"


def sets():
    from sim.planing_anyspeed import PRESETS
    p1c = tuple(PRESETS["as1"])
    hh = p1c + ("heel_hydro",)
    hx = hh + ("lateral_phi0", "lateral_cross")
    return dict(p1c=p1c, hh=hh, hx=hx)


def cand(name=None):
    return sets()[name or CAND]


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def stamp_all():
    from hydro import hulls
    from sim import physics
    st = dict(cand=CAND, switches=list(cand()),
              code_sha1_as1=physics.code_sha1("as1"))
    for key in (AS, HB):
        st[key] = physics.stamp_for(hulls.get(key), "as1",
                                    cand())["physics_sha1"]
    with open(os.path.abspath(__file__), "rb") as f:
        st["check_sha1"] = hashlib.sha1(f.read()).hexdigest()
    st["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return st


def flush(name):
    os.makedirs(OUT, exist_ok=True)
    head = ["# stamp " + json.dumps(STAMP, sort_keys=True)] if STAMP else []
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write("\n".join(head + LINES) + "\n")
    LINES.clear()


def _js(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return str(v)


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    if STAMP:
        obj = dict(obj, _stamp=STAMP)
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=_js)


def check(part, name, ok, detail=""):
    ok = bool(ok)
    CHECKS.append(dict(part=part, name=name, ok=ok, detail=str(detail)))
    out(f"  [{'ok' if ok else 'FAIL'}] {part}: {name}"
        + (f"  -- {detail}" if detail != "" else ""))
    return ok


def calm():
    from sim.test_vessel import Monochromatic
    return Monochromatic(1.0, 0.0)


def plant(hull=AS, sea=None, sw=None, captive_u=None, dt=0.02, gz=True,
          record=False):
    from hydro import hulls
    from sim.planing_anyspeed import PlaningVesselAS, PlaningVesselASLin
    h = hulls.get(hull) if isinstance(hull, str) else hull
    sea = calm() if sea is None else sea
    kw = dict(record=record) if gz else {}
    cls = PlaningVesselAS if gz else PlaningVesselASLin
    return cls(h, sea, dt=dt, captive_u=captive_u,
               test_switches=cand() if sw is None else sw, **kw)


def use_candidate():
    from sim import planing_anyspeed as PA
    PA.PRESETS["as1"] = PA.resolve("as1", cand())


# ===================================================================== H1
def _sh_clip(P, a, b, c):
    """Sutherland-Hodgman: the part of polygon P (list of (y, z)) where a y
    + b z <= c."""
    outp = []
    n = len(P)
    for i in range(n):
        y0, z0 = P[i]
        y1, z1 = P[(i + 1) % n]
        f0 = a * y0 + b * z0 - c
        f1 = a * y1 + b * z1 - c
        if f0 <= 0.0:
            outp.append((y0, z0))
        if (f0 < 0.0 < f1) or (f1 < 0.0 < f0):
            t = f0 / (f0 - f1)
            outp.append((y0 + t * (y1 - y0), z0 + t * (z1 - z0)))
    return outp


def _vertex_moments(P):
    """(A, Sy, Sz, Syy, Syz, Szz) of a simple polygon from its vertices
    (counter-clockwise)."""
    if len(P) < 3:
        return np.zeros(6)
    Y = np.array([p[0] for p in P])
    Z = np.array([p[1] for p in P])
    Y1, Z1 = np.roll(Y, -1), np.roll(Z, -1)
    cr = Y * Z1 - Y1 * Z
    A = 0.5 * cr.sum()
    Sy = ((Y + Y1) * cr).sum() / 6.0
    Sz = ((Z + Z1) * cr).sum() / 6.0
    Syy = ((Y * Y + Y * Y1 + Y1 * Y1) * cr).sum() / 12.0
    Szz = ((Z * Z + Z * Z1 + Z1 * Z1) * cr).sum() / 12.0
    Syz = ((Y * Z1 + 2 * Y * Z + 2 * Y1 * Z1 + Y1 * Z) * cr).sum() / 24.0
    return np.array([A, Sy, Sz, Syy, Syz, Szz])


def part_h1():
    from sim.planing_anyspeed import _clip_moments, _vsec, _StripAS
    out("S5a (h1)  H1: polygon moments")
    rng = np.random.default_rng(1)
    worst_a = 0.0
    for _ in range(400):
        yc = rng.uniform(0.05, 1.5)
        tb = math.tan(math.radians(rng.choice([10, 16, 20, 30])))
        depth = rng.uniform(0.6, 1.2) + yc * tb
        zk = rng.uniform(-0.8, -0.2)
        d = rng.uniform(-0.1, 1.3 * depth)
        zc = zk + min(yc * tb, depth)
        py = np.array([0.0, yc, yc, -yc, -yc])
        pz = np.array([zk, zc, zk + depth, zk + depth, zc])
        A, Sy, Sz = _clip_moments(py, pz, np.array(zk + d))
        Av, m1, _ = _vsec(np.array([min(d, depth)]), np.array([yc]), tb)
        Av, m1 = float(Av[0]), float(m1[0])
        full = yc * yc * tb + 2 * yc * (depth - yc * tb)
        e = max(abs(float(A) - Av) / full,
                abs(float(Sz) - (zk * Av + m1)) / (full * depth),
                abs(float(Sy)) / (full * yc))
        worst_a = max(worst_a, e)
    check("h1", "H1(a): phi = 0, s = 0: A and int z dA equal _vsec's A and "
          "zk A + m1 (d up to 1.3 x depth, the deck caps it): <= 1e-12 of "
          "the scale", worst_a <= 1e-12, f"largest {worst_a:.1e}")
    worst_b = 0.0
    for _ in range(600):
        yc = rng.uniform(0.05, 1.5)
        tb = math.tan(math.radians(rng.choice([10, 16, 20, 30])))
        depth = rng.uniform(0.6, 1.2) + yc * tb
        zk = rng.uniform(-0.8, -0.2)
        zc = zk + min(yc * tb, depth)
        py = np.array([0.0, yc, yc, -yc, -yc])
        pz = np.array([zk, zc, zk + depth, zk + depth, zc])
        phi = math.radians(rng.uniform(-60, 60))
        s = rng.uniform(-0.3, 0.3)
        H = rng.uniform(zk - 1.0, zk + depth + 1.0)
        # the plant's route (_StripAS._hh_poly: aligned frame, rotation)
        al = math.atan(s)
        ca, sa = math.cos(al), math.sin(al)
        got = np.array(_StripAS._hh_poly(py, pz, phi, ca, sa, H * ca))
        # independent: the level-frame polygon clipped by z <= H + s y
        c, sn = math.cos(phi), math.sin(phi)
        P = [(y * c - z * sn, y * sn + z * c) for y, z in zip(py, pz)]
        ref = _vertex_moments(_sh_clip(P, -s, 1.0, H))
        full = yc * yc * tb + 2 * yc * (depth - yc * tb)
        R = max(yc, depth)
        sc = np.array([full, full * R, full * R, full * R * R, full * R * R,
                       full * R * R])
        worst_b = max(worst_b, float(np.max(np.abs(got - ref) / sc)))
    check("h1", "H1(b): random heel -60..60 deg, slope +-0.3, dry to "
          "submerged: A, Sy, Sz, Syy, Syz, Szz equal an independent "
          "Sutherland-Hodgman clip + vertex formulas: <= 1e-12 of the scale",
          worst_b <= 1e-12, f"largest {worst_b:.1e}")
    flush("h1.txt")


# ===================================================================== H3
def _calm_attitude(p, u):
    z, th = p.running_attitude(u)
    s = np.zeros(14)
    s[6], s[2], s[4] = u, z, th
    return s


def part_h3():
    from sim.test_vessel import Monochromatic
    out("S5a (h3)  H3: upright (phi = 0, no transverse slope)")
    sw = cand("hh")
    worst = worst2 = 0.0
    rows = []
    for hull in (AS, HB):
        for kn in (0.0, 5.0, 15.0, 25.0, 35.0):
            pf = plant(hull, sw=sw, captive_u=kn * KN)
            s = _calm_attitude(pf, kn * KN)
            _, a0 = pf.deriv(s, 0.0)
            pf._hh_force = True
            _, a1 = pf.deriv(s, 0.0)
            pf._hh_force = False
            ok_route = (not a0["heeled"]) and a1["heeled"]
            for k in ("Fz_p", "My_p", "Xh", "Mh"):
                sc = max(abs(a0[k]), 1e-9 * pf.m * G)
                e = abs(a1[k] - a0[k]) / sc
                worst = max(worst, e)
                # H3(a') (supplementary registration after the first run:
                # at an equilibrium the pitch moment is a residual near
                # zero, so the original 'relative to the quantity' is ill
                # posed there): relative to W (forces) or W L (moments)
                sc2 = pf.m * G * (pf.L if k in ("My_p", "Mh") else 1.0)
                worst2 = max(worst2, abs(a1[k] - a0[k]) / sc2)
                rows.append((hull, kn, k, a0[k], a1[k], e))
            if not ok_route:
                worst = np.inf
    for r in rows:
        out(f"    {r[0]:14s} {r[1]:4.0f} kn {r[2]:5s} closed {r[3]: .9e} "
            f"polygon {r[4]: .9e} rel {r[5]:.1e}")
    check("h3", "H3(a): polygons forced at phi = 0, s = 0 vs P1c's closed "
          "forms (vertical force, pitch moment, X, M_h at rest and 5-35 kn "
          "captive equilibria, scarab195 and planing_b): <= 1e-12 relative",
          worst <= 1e-12, f"largest {worst:.1e}")
    check("h3", "H3(a'): the same relative to W (forces) and W L (moments)"
          ": <= 1e-12", worst2 <= 1e-12, f"largest {worst2:.1e}")
    # (b) continuity at phi -> 0
    wb = wroll = 0.0
    for hull in (AS, HB):
        p = plant(hull, sw=sw)
        for kn in (0.0, 12.0, 30.0):
            s = p.initial_state(kn * KN)
            d0, a0 = p.deriv(s, 0.0)
            s1 = s.copy()
            s1[3] = 1e-9
            d1, a1 = p.deriv(s1, 0.0)
            assert a1["heeled"] and not a0["heeled"]
            dsc = np.maximum(np.abs(d0), 1.0)[6:12]
            # the roll row: the stiffness times 1e-9 rad is expected
            e = np.abs(d1 - d0)[6:12] / dsc
            e[3] = 0.0
            wb = max(wb, float(e.max()))
            # S5a review (P): the roll row against -K_phi 1e-9 / I, K_phi /
            # I the polygons' slope of the roll acceleration (central
            # difference over +-1e-5 rad, both on the polygon branch)
            sp_, sm_ = s.copy(), s.copy()
            sp_[3], sm_[3] = 1e-5, -1e-5
            slope = (p.deriv(sp_, 0.0)[0][9] - p.deriv(sm_, 0.0)[0][9]) / 2e-5
            want = slope * 1e-9
            wroll = max(wroll, abs((d1[9] - d0[9]) - want)
                        / max(abs(want), 1e-300))
    check("h3", "H3(b): phi = 1e-9 rad (polygons) vs phi = 0 (P1c's lines): "
          "accelerations (roll aside: its stiffness x 1e-9) within 1e-7 of "
          "the scale", wb <= 1e-7, f"largest {wb:.1e}")
    check("h3", "H3(b) roll row (S5a review): the roll acceleration's change "
          "equals -K_phi 1e-9 / I (the polygons' own slope) within 1e-6 "
          "relative", wroll <= 1e-6, f"largest {wroll:.1e}")
    # (c) bit for bit on the upright branch
    rng = np.random.default_rng(3)
    bad = []
    for hull in (AS, HB):
        for gz in (True, False):
            for sea in (calm(), Monochromatic(1.6, 0.25, theta0=0.0)):
                a = plant(hull, sea=sea, sw=cand("p1c"), gz=gz)
                b = plant(hull, sea=sea, sw=sw, gz=gz)
                for i in range(50):
                    s = np.zeros(14)
                    s[0] = rng.uniform(-30, 30)
                    s[2], s[4] = rng.uniform(-0.05, 0.15), \
                        rng.uniform(-0.05, 0.02)
                    s[6] = rng.uniform(-2, 35 * KN)
                    s[8], s[10] = rng.normal(0, [0.3, 0.1])
                    s[12] = rng.uniform(0, 1) * a.prop.t_max
                    t = float(rng.uniform(0, 60))
                    da, _ = a.deriv(s, t)
                    db, _ = b.deriv(s, t)
                    if not np.array_equal(da, db):
                        bad.append((hull, gz, "deriv", i))
                for kn in (0.0, 8.0, 25.0):
                    a = plant(hull, sea=sea, sw=cand("p1c"), gz=gz)
                    b = plant(hull, sea=sea, sw=sw, gz=gz)
                    sa = a.initial_state(kn * KN)
                    sb = b.initial_state(kn * KN)
                    sa[2] += 0.03
                    sb[2] += 0.03
                    for k in range(200):
                        sa = a.step(sa, k * a.dt, sa[12], 0.0)
                        sb = b.step(sb, k * b.dt, sb[12], 0.0)
                    if not np.array_equal(sa, sb):
                        bad.append((hull, gz, "run", kn))
    check("h3", "H3(c): the upright branch is P1c bit for bit (calm and "
          "long-crested head waves; 50 random symmetric states' derivatives "
          "and 200-step runs at 0, 8, 25 kn; scarab195, planing_b; GZ and "
          "linear-roll plants)", not bad, f"{len(bad)} differ: {bad[:5]}")
    flush("h3.txt")


# ===================================================================== H6
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
    out(f"S5a (sym)  H6: mirror symmetry ({CAND})")
    rng = np.random.default_rng(7)
    MIR = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], float)
    worst = 0.0
    for hull in (AS, HB):
        for seaname in ("calm", "short-crested"):
            sea = (calm() if seaname == "calm" else
                   SeaState(1.0, 5.0, n_freq=12, n_dir=5, seed=11,
                            theta0=2.5))
            p = plant(hull, sea=sea)
            q = plant(hull, sea=sea if seaname == "calm"
                      else _mirror_sea(sea))
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
                e_ = float(np.max(np.abs(MIR * d2 - d1)
                                  / max(np.abs(d1).max(), 1e-30)))
                worst = max(worst, e_)
    check("sym", f"H6: mirrored state (and sea) give the mirrored derivative "
          f"({CAND}; 20 random states x calm / short-crested, heel +-20 deg; "
          f"scarab195, planing_b): <= 1e-12 relative", worst <= 1e-12,
          f"largest {worst:.1e}")
    flush(f"sym_{CAND}.txt")


# ==================================================================== L1'
def _Mt(p, d, phi):
    """The cross-plane matrix of every strip, independent assembly: the
    table's body matrix about the keel, the shift to the CG and the
    rotation as 3x3 matrix products (R^T B R), heave-heave zeroed."""
    C, _, _ = p._tab().lateral(d, p.yc, phi)
    hG = -p.zk
    c, s = math.cos(phi), math.sin(phi)
    R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    out_ = np.zeros((d.size, 3, 3))
    for j in range(d.size):
        b22, b23, b33, b24, b34, b44 = C[j]
        B = np.array([[b22, b23, b24], [b23, b33, b34], [b24, b34, b44]])
        # roll about G = roll about the keel + h_G x body sway (the CG is
        # h_G up the body z axis from the keel)
        T = np.array([[1, 0, 0], [0, 1, 0], [hG[j], 0.0, 1.0]])
        BG = T @ B @ T.T
        M = p.rho * (R.T @ BG @ R)
        M[1, 1] = 0.0
        out_[j] = M
    return out_


def _d_of(p, s, t):
    """The plant's keel depths at state s, time t (heel_hydro's formula)."""
    _, aux = p.deriv(s, t)
    return aux["d"]


def part_l1p():
    from sim.wavefield import SeaState
    from sim.planing_vessel import SP
    out("S5a (l1p)  L1': the plant's cross-plane strip forces vs finite "
        "differences of the same table's M~(d, phi)")
    sw = cand("hx")
    rng = np.random.default_rng(5)
    HS = (1e-5, 1e-6, 1e-7)
    worst = {h: 0.0 for h in HS}
    best_state = 0.0
    for hull in (AS, HB):
        for seaname in ("calm", "sea"):
            sea = (calm() if seaname == "calm" else
                   SeaState(0.6, 4.0, n_freq=8, n_dir=3, seed=4,
                            theta0=1.2))
            p = plant(hull, sea=sea, sw=sw)
            for i in range(4):
                s = p.initial_state(rng.uniform(0, 30) * KN)
                s[3] = math.radians(rng.uniform(-25, 25))
                s[7], s[8], s[9] = rng.normal(0, [0.3, 0.2, 0.3])
                s[10], s[11] = rng.normal(0, [0.05, 0.1])
                t = float(rng.uniform(0, 30))
                ds, aux = p.deriv(s, t)
                lat, kb = aux["lat"], aux["kb"]
                d = aux["d"]
                nu = np.stack([lat.vr, lat.Wr, np.full(d.size, s[9])], 1)
                errs = []
                for h in HS:
                    # reactions: dT/dz (each strip's d moves by -dz) and
                    # dT/dphi at a fixed CG (d from the plant's formula)
                    Mp = _Mt(p, d - h, s[3])
                    Mm = _Mt(p, d + h, s[3])
                    Tz = 0.5 * np.einsum("ji,jik,jk->j", nu, (Mp - Mm)
                                         / (2 * h), nu)
                    sp_, sm_ = s.copy(), s.copy()
                    sp_[3] += h
                    sm_[3] -= h
                    dp_, dm_ = _d_of(p, sp_, t), _d_of(p, sm_, t)
                    Tp = 0.5 * np.einsum("ji,jik,jk->j", nu,
                                         (_Mt(p, dp_, s[3] + h)
                                          - _Mt(p, dm_, s[3] - h)) / (2 * h),
                                         nu)
                    # the rate at fixed body x along the motion: positions
                    # advanced by their rates, waves at t +- h
                    kin = np.zeros(14)
                    kin[:6] = ds[:6]
                    s1, s0 = s + h * kin, s - h * kin
                    Md = (_Mt(p, _d_of(p, s1, t + h), s1[3])
                          - _Mt(p, _d_of(p, s0, t - h), s0[3])) / (2 * h)
                    M0 = _Mt(p, d, s[3])
                    if p.sw.orbital and kb.wv:
                        aW0 = kb.aoz - (p.kp + SP * (p.Th_rest + s[4])) \
                            * kb.aox + kb.ur * SP * s[10]
                    else:
                        ett = p._waves(s[0] + p.x * math.cos(s[5]),
                                       s[1] + p.x * math.sin(s[5]), t)[2]
                        aW0 = ett + kb.ur * SP * s[10]
                    n0 = np.stack([-kb.aoy, -aW0, np.zeros(d.size)], 1)
                    F = (-np.einsum("jik,jk->ji", M0, n0)
                         - np.einsum("jik,jk->ji", Md, nu))
                    F[:, 1] += Tz
                    F[:, 2] += Tp
                    got = np.stack([lat.fy, lat.fz, lat.fL], 1)
                    sc = max(np.abs(got).max(), 1e-9)
                    e_ = float(np.abs(got - F).max() / sc)
                    errs.append(e_)
                    worst[h] = max(worst[h], e_)
                # the table is bilinear (its derivative jumps across a cell
                # line): a central difference whose step straddles one is
                # ambiguous, so each state takes its best step
                best_state = max(best_state, min(errs))
                out(f"    {hull} {'calm' if seaname == 'calm' else 'sea'} "
                    f"heel {math.degrees(s[3]):6.2f} deg, u "
                    f"{s[6] / KN:4.1f} kn: " + ", ".join(
                        f"h {h:.0e} {e_:.1e}" for h, e_ in zip(HS, errs)))
    out("    largest over the states, per step: " + ", ".join(
        f"h {h:.0e} {worst[h]:.2e}" for h in HS))
    best = best_state
    check("l1p", "L1': strip forces (fy, fz, fL) of lateral_cross vs the "
          "Lagrangian assembled by central differences of the table's M~ "
          "(independent rotation; 16 random heeled states, calm and sea, two "
          "hulls): <= 1e-6 relative", best <= 1e-6, f"{best:.2e}")
    flush("l1p.txt")



# ===================================================================== L1
class Sec2D:
    """A 2D section for L1: its boundary as a fixed list of body-frame
    nodes (y, z about the CG, counter-clockwise seen with y right, z up),
    each straight edge split into panels (cosine-clustered towards the
    body corners where asked, uniform elsewhere), so that a body point
    keeps its panel and its arclength s as the configuration changes;
    groups: the smooth parts of the boundary (between corners); the free
    surface z = 0 with phi = 0 (negative image: the method of
    hydro/section2d)."""

    def __init__(self, verts, n_per, cluster, groups=None):
        Y, Z, E = [], [], []
        for k in range(len(verts) - 1):
            (y0, z0), (y1, z1) = verts[k], verts[k + 1]
            n = n_per[k]
            if cluster[k] == "both":
                s = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n + 1)))
            elif cluster[k] == "end":
                s = 1 - np.cos(np.linspace(0, 0.5 * np.pi, n + 1))
            elif cluster[k] == "start":
                s = np.sin(np.linspace(0, 0.5 * np.pi, n + 1))
            else:
                s = np.linspace(0, 1, n + 1)
            Y.append(y0 + (y1 - y0) * s[:-1])
            Z.append(z0 + (z1 - z0) * s[:-1])
            E.append(np.full(n, k if groups is None else groups[k]))
        Y.append([verts[-1][0]])
        Z.append([verts[-1][1]])
        self.nodes = np.stack([np.concatenate(Y), np.concatenate(Z)], 1)
        self.group = np.concatenate(E)
        seg = np.linalg.norm(np.diff(self.nodes, axis=0), axis=1)
        self.s_node = np.concatenate([[0.0], np.cumsum(seg)])
        self.seg = seg

    def wetted(self, yG, zG, phi):
        """Panels of the wetted part at CG (yG, zG), heel phi: earth
        endpoints a, b (the two waterline panels cut at z = 0), the body
        panel index of each and the body arclength of their ends."""
        c, s = math.cos(phi), math.sin(phi)
        B = self.nodes
        P = np.stack([yG + B[:, 0] * c - B[:, 1] * s,
                      zG + B[:, 0] * s + B[:, 1] * c], 1)
        a_l, b_l, idx, sa, sb = [], [], [], [], []
        for i in range(len(P) - 1):
            p0, p1 = P[i], P[i + 1]
            w0, w1 = p0[1] <= 0.0, p1[1] <= 0.0
            if not (w0 or w1):
                continue
            t0_, t1_ = 0.0, 1.0
            if w0 != w1:
                tc = p0[1] / (p0[1] - p1[1])
                if w0:
                    t1_ = tc
                else:
                    t0_ = tc
            if t1_ - t0_ <= 0.0:
                continue
            pa, pb = p0 + t0_ * (p1 - p0), p0 + t1_ * (p1 - p0)
            if t1_ < 1.0:
                pb[1] = 0.0
            if t0_ > 0.0:
                pa[1] = 0.0
            a_l.append(pa)
            b_l.append(pb)
            idx.append(i)
            sa.append(self.s_node[i] + t0_ * self.seg[i])
            sb.append(self.s_node[i] + t1_ * self.seg[i])
        return (np.array(a_l), np.array(b_l), np.array(idx), np.array(sa),
                np.array(sb))

    @staticmethod
    def solve(a, b, yG, zG):
        """Unit potentials (N, 3) of earth sway, earth heave, roll about
        (yG, zG) on the panels a -> b (the method and conventions of
        hydro/section2d.added_mass_2d: n into the body), with the panel
        data."""
        from hydro.section2d import _influence
        h = np.linalg.norm(b - a, axis=1)
        t = (b - a) / h[:, None]
        n = np.stack([t[:, 1], -t[:, 0]], -1)
        m = 0.5 * (a + b)
        S0, D0 = _influence(m, m, t, n, h)
        R = np.array([1.0, -1.0])
        Si, Di = _influence(m, m * R, t * R, n * R, h)
        S, D = S0 - Si, D0 - Di
        q = np.stack([n[:, 0], n[:, 1],
                      (m[:, 0] - yG) * n[:, 1] - (m[:, 1] - zG) * n[:, 0]],
                     -1)
        A = 0.5 * np.eye(h.size) - D
        ph = np.linalg.solve(A, -S @ q)
        M = ph.T @ (q * h[:, None])
        return ph, dict(h=h, t=t, n=n, m=m, q=q, M=0.5 * (M + M.T))


def _l1_config(t, mot):
    """CG position, heel, their rates and accelerations of the prescribed
    motion at time t: each of y, z, phi = q0 + A sin(w t + f)."""
    q = np.array([m[0] + m[1] * math.sin(m[2] * t + m[3]) for m in mot])
    v = np.array([m[1] * m[2] * math.cos(m[2] * t + m[3]) for m in mot])
    a = np.array([-m[1] * m[2] ** 2 * math.sin(m[2] * t + m[3])
                  for m in mot])
    return q, v, a


def _l1_fit(x0, xs, vs):
    """(value, derivative) at x0 of the quadratic through the 3 points of
    (xs, vs) nearest x0 (a line through 2)."""
    xs, vs = np.asarray(xs, float) - x0, np.asarray(vs, float)
    o = np.argsort(np.abs(xs))[:3]
    xs, vs = xs[o], vs[o]
    if len(xs) >= 3:
        V = np.stack([np.ones(3), xs, xs * xs], 1)
        c_ = np.linalg.solve(V, vs)
        return c_[0], c_[1]
    sl = (vs[1] - vs[0]) / (xs[1] - xs[0])
    return vs[0] - xs[0] * sl, sl


def _l1_points(sec, idx, sa, sb, val, grp):
    """The (s, phi) samples of group grp: the panel centres and phi = 0 at
    the waterline points (the cut ends of the first and last panels)."""
    xs, vs = [], []
    N = len(idx)
    for k in range(N):
        if sec.group[idx[k]] == grp:
            xs.append(0.5 * (sa[k] + sb[k]))
            vs.append(val[k])
    if sec.group[idx[0]] == grp and sa[0] > sec.s_node[idx[0]]:
        xs.append(sa[0])
        vs.append(0.0)
    if sec.group[idx[-1]] == grp and sb[-1] < sec.s_node[idx[-1] + 1]:
        xs.append(sb[-1])
        vs.append(0.0)
    return xs, vs


def l1_forces(sec, t0, mot, dt=1e-4, eps=1e-6):
    """(F_pressure, F_lagrange, F_impulse_only) (3,): sway force, heave
    force, roll moment about the CG on the section at time t0 of the
    prescribed motion; rho = 1."""
    (yG, zG, phi), nu, nud = _l1_config(t0, mot)
    a, b, idx, sa, sb = sec.wetted(yG, zG, phi)
    ph0, geo = Sec2D.solve(a, b, yG, zG)
    m, n, h, tt = geo["m"], geo["n"], geo["h"], geo["t"]
    N = len(idx)
    sc_ = 0.5 * (sa + sb)

    def phi_at(tq):
        """The potential at the body points of the t0 panel centres (body
        arclength sc_) at time tq: a full panel of both times keeps its
        value; the others by a quadratic in s along their group, with phi =
        0 at the waterline points."""
        (yq, zq, pq), nq, _ = _l1_config(tq, mot)
        aq, bq, iq, saq, sbq = sec.wetted(yq, zq, pq)
        phq, _ = Sec2D.solve(aq, bq, yq, zq)
        val = phq @ nq
        scq = 0.5 * (saq + sbq)
        pos = {int(i): kk for kk, i in enumerate(iq)}
        res = np.zeros(N)
        for k in range(N):
            kk = pos.get(int(idx[k]))
            if kk is not None and abs(scq[kk] - sc_[k]) < 1e-13:
                res[k] = val[kk]
                continue
            xs, vs = _l1_points(sec, iq, saq, sbq, val, sec.group[idx[k]])
            res[k] = _l1_fit(sc_[k], xs, vs)[0]
        return res
    Dphi = (phi_at(t0 + dt) - phi_at(t0 - dt)) / (2 * dt)
    vb = np.stack([nu[0] - nu[2] * (m[:, 1] - zG),
                   nu[1] + nu[2] * (m[:, 0] - yG)], 1)
    phn = (vb * n).sum(1)
    val0 = ph0 @ nu
    phs = np.zeros(N)
    for k in range(N):
        xs, vs = _l1_points(sec, idx, sa, sb, val0, sec.group[idx[k]])
        phs[k] = _l1_fit(sc_[k], xs, vs)[1]
    grad = phn[:, None] * n + phs[:, None] * tt
    pt = Dphi - (vb * grad).sum(1)
    p = -(pt + 0.5 * (grad * grad).sum(1))
    dF = p[:, None] * n * h[:, None]
    Fp = np.array([dF[:, 0].sum(), dF[:, 1].sum(),
                   ((m[:, 0] - yG) * dF[:, 1]
                    - (m[:, 1] - zG) * dF[:, 0]).sum()])

    def Mof(z_, p_):
        a_, b_ = sec.wetted(yG, z_, p_)[:2]
        return Sec2D.solve(a_, b_, yG, z_)[1]["M"]
    M0 = geo["M"]
    Mz = (Mof(zG + eps, phi) - Mof(zG - eps, phi)) / (2 * eps)
    Mph = (Mof(zG, phi + eps) - Mof(zG, phi - eps)) / (2 * eps)
    Mdot = Mz * nu[1] + Mph * nu[2]
    imp = -(M0 @ nud + Mdot @ nu)
    react = np.array([0.0, 0.5 * nu @ Mz @ nu, 0.5 * nu @ Mph @ nu])
    return Fp, imp + react, imp


def _l1_sections(N):
    """The two L1 sections at resolution N: a convex ellipse (CG at its
    centre) and the plant's V + wall section (yc 0.9 m, 20 deg deadrise,
    CG 0.55 m above the keel), body nodes counter-clockwise."""
    th = np.linspace(-0.5 * np.pi, 1.5 * np.pi, 8 * N + 1)
    ell = Sec2D([(0.8 * math.cos(x), 0.35 * math.sin(x)) for x in th],
                [1] * (8 * N), ["none"] * (8 * N), groups=[0] * (8 * N))
    yc, tb, hG = 0.9, math.tan(math.radians(20.0)), 0.55
    dch = yc * tb
    top = 1.2
    V = [(yc, top - hG), (yc, dch - hG), (0.0, -hG), (-yc, dch - hG),
         (-yc, top - hG)]
    vw = Sec2D(V, [2 * N, 2 * N, 2 * N, 2 * N],
               ["end", "both", "both", "start"])
    return dict(ellipse=ell, vwall=vw)


def part_l1():
    out("S5a (l1)  L1: 2D phi = 0 section, sway + heave + roll together "
        "with heel: the Lagrangian (impulse + dT/dq) formula vs direct "
        "pressure integration, p = -(dphi/dt + |grad phi|^2 / 2) (rho = 1)")
    cases = []
    for name, phi0 in (("ellipse", 0.0), ("ellipse", 10.0),
                       ("ellipse", 25.0), ("vwall", 0.0), ("vwall", 10.0),
                       ("vwall", 25.0)):
        if name == "ellipse":
            # CG 0.05 m below the surface: well over half immersed
            mot = [(0.0, 0.10, 1.3, 0.4), (-0.05, 0.04, 1.7, 1.1),
                   (math.radians(phi0), math.radians(4.0), 2.1, 0.7)]
        else:
            # keel 0.75 m deep: the waterline on the walls at every heel
            mot = [(0.0, 0.10, 1.3, 0.4), (0.55 - 0.75, 0.04, 1.7, 1.1),
                   (math.radians(phi0), math.radians(4.0), 2.1, 0.7)]
        cases.append((name, phi0, mot))
    rows = []
    worst = 0.0
    for name, phi0, mot in cases:
        res = []
        for N in NS_L1:
            sec = _l1_sections(N)[name]
            Fp, FL, Fi = l1_forces(sec, 0.37, mot)
            res.append((N, Fp, FL, Fi))
        D = np.array([r[1] - r[2] for r in res])
        scale = max(np.abs(res[-1][2][:2]).max(), 1e-12)
        mscale = max(abs(res[-1][2][2]), 1e-12)
        sc = np.array([scale, scale, mscale])
        e = np.abs(D) / sc
        rel_imp = np.abs(res[-1][2] - res[-1][3]) / sc
        out(f"  {name} heel {phi0:4.0f} deg: F_L = {res[-1][2]}")
        for (N, Fp, FL, Fi), ee in zip(res, e):
            out(f"     N {N:4d}: F_p {Fp}  |F_p - F_L| / scale "
                f"{ee[0]:.1e} {ee[1]:.1e} {ee[2]:.1e}")
        dN, dN2, dN4 = D[-1], D[-2], D[-3]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs((dN4 - dN2) / (dN2 - dN))
        pord = np.clip(np.log2(np.where(np.isfinite(ratio) & (ratio > 1),
                                        ratio, 2.0)), 0.5, 4.0)
        Dinf = dN - (dN2 - dN) / (2 ** pord - 1)
        einf = np.abs(Dinf) / sc
        out(f"     extrapolated |F_p - F_L| / scale {einf}, observed orders "
            f"{pord}; the reaction dT/dq (F_L without it) "
            f"{rel_imp}")
        worst = max(worst, float(einf.max()))
        rows.append(dict(case=name, heel=phi0,
                         res=[(r[0], r[1], r[2], r[3]) for r in res],
                         einf=einf, imp_only=rel_imp))
    check("l1", "L1: the Lagrangian formula F = -d/dt(M nu) + dT/dq (M about "
          "the CG, earth frame) equals the direct pressure integral within "
          "1e-3 of the force / moment scale after refinement (ellipse and "
          "V + wall sections, heel 0, 10, 25 deg, sway + heave + roll)",
          worst <= 1e-3, f"largest extrapolated {worst:.1e}")
    dump("l1.json", dict(rows=rows))
    flush("l1.txt")


NS_L1 = (20, 40, 80, 160)



# ===================================================================== H2
def gz_plant(p, phis, free_trim=False):
    """The plant's own righting arm (heel_hydro's polygons and the
    functions deriv calls) at heels phis: calm water, displacement held by
    bisection on z (trim at rest, or free trim: pitch moment zero too, by
    Newton on (z, th)); GZ = -K / W, K the hydrostatic roll moment (a_b = 1
    at rest)."""
    from sim.planing_vessel import SP
    n = p.x.size
    W = p.m * G
    rg = p.rho * G
    e0 = np.zeros(n)
    out_ = []
    p._hh_force = True

    def FK(z, th, hh):
        p._hh_sections(hh, e0, z, p.Th_rest + th)
        F = rg * hh.A.sum() * p.dx - W
        My = (SP * p.x * rg * hh.A).sum() * p.dx
        K = rg * hh.Sy.sum() * p.dx
        return F, My, K
    for phi in phis:
        hh = p._hh_state(float(phi), e0)
        if not free_trim:
            lo, hi = -1.5, 1.5
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if FK(mid, 0.0, hh)[0] > 0.0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-15:
                    break
            z, th = 0.5 * (lo + hi), 0.0
        else:
            z, th = 0.0, 0.0
            for _ in range(60):
                F, My, _ = FK(z, th, hh)
                h_ = 1e-7
                Fz1, Mz1, _ = FK(z + h_, th, hh)
                Ft1, Mt1, _ = FK(z, th + h_, hh)
                J = np.array([[Fz1 - F, Ft1 - F], [Mz1 - My, Mt1 - My]]) / h_
                dz, dth = np.linalg.solve(J, [-F, -My])
                z, th = z + dz, th + dth
                if abs(dz) < 1e-13 and abs(dth) < 1e-13:
                    break
        out_.append((-FK(z, th, hh)[2] / W, z, th))
    p._hh_force = False
    return np.array(out_)


def part_h2():
    out("S5a (h2)  H2: the plant's own righting arm at rest (heel_hydro) vs "
        "the frozen GZ table")
    rows = {}
    for hull in (AS, HB):
        p = plant(hull, sw=cand("hh"))
        phv = p.phi_vanish
        phis = np.radians(np.arange(0.0, math.degrees(phv) + 1e-9, 1.0))
        gzp = gz_plant(p, phis)
        geo = p.gz_geometric(phis)[0]
        tab = np.array([p.gz(f) for f in phis])
        gmax = float(np.max(tab))
        e_geo = float(np.abs(gzp[:, 0] - geo).max())
        e_tab = float(np.abs(gzp[:, 0] - tab).max())
        # the plant's own vanishing angle: its zero crossing past the max,
        # on a fine grid near the table's
        fine = np.linspace(phv - math.radians(1.5), phv + math.radians(1.5),
                           61)
        gzf = gz_plant(p, fine)[:, 0]
        k = np.nonzero((gzf[:-1] > 0) & (gzf[1:] <= 0))[0]
        if k.size:
            k = k[0]
            phv_p = fine[k] + (fine[k + 1] - fine[k]) * gzf[k] / (gzf[k]
                                                                - gzf[k + 1])
        else:
            phv_p = np.nan
        rel_v = abs(phv_p - phv) / phv
        free = gz_plant(p, phis[::5], free_trim=True)
        out(f"  {hull}: GZ_max {gmax:.4f} m, vanishing angle table "
            f"{math.degrees(phv):.3f} deg, plant {math.degrees(phv_p):.3f} "
            f"deg; dGM shift {p.dgm * 1e3:.3f} mm")
        out("    phi    plant GZ    geometric   table(shifted)  free trim")
        fr = dict(zip(np.round(np.degrees(phis[::5]), 6), free[:, 0]))
        for f, g1, g2, g3 in zip(np.degrees(phis), gzp[:, 0], geo, tab):
            ft = fr.get(round(f, 6))
            out(f"    {f:5.1f} {g1: .6f} {g2: .6f} {g3: .6f}"
                + (f"   {ft: .6f}" if ft is not None else ""))
        check("h2", f"H2(a) {hull}: the plant's GZ equals the frozen "
              f"geometric curve (same polygons) within 1e-9 m, 0 deg to the "
              f"vanishing angle", e_geo <= 1e-9, f"{e_geo:.1e} m")
        check("h2", f"H2(b) {hull}: within 1 % of GZ_max of the frozen table "
              f"(shifted by dGM sin phi) and the vanishing angle within 1 %",
              e_tab <= 0.01 * gmax and rel_v <= 0.01,
              f"largest {e_tab * 1e3:.3f} mm ({100 * e_tab / gmax:.3f} % of "
              f"GZ_max), vanishing angle {100 * rel_v:.3f} %")
        rows[hull] = dict(phi=phis, plant=gzp, geo=geo, table=tab,
                          free=free, phv=phv, phv_plant=phv_p)
    dump("h2.json", rows)
    flush("h2.txt")


# ===================================================================== H4
def rest_run(p, s0, T=600.0):
    s = s0.copy()
    p.captive_u = None
    p._reset()
    n = int(round(T / p.dt))
    m60 = int(round(60.0 / p.dt))
    umax, zs, ts, ph, bad = 0.0, [], [], [], None
    for i in range(n):
        s = p.step(s, i * p.dt, 0.0, 0.0)
        if not np.all(np.isfinite(s)) or getattr(p, "capsized", False):
            bad = i * p.dt
            break
        umax = max(umax, abs(s[6]))
        if i >= n - m60:
            zs.append(s[2])
            ts.append(s[4])
            ph.append(abs(s[3]))
    return dict(u_end=float(s[6]), u_max=umax, x_end=float(s[0]),
                z_mean=float(np.mean(zs)) if zs else np.nan,
                th_mean=float(np.mean(ts)) if ts else np.nan,
                heel_mean=float(np.degrees(np.mean(ph))) if ph else np.nan,
                diverged=bad)


def _r3_job(arg):
    hull, lab, dz, dtr, roll, sw = arg
    from studies import anyspeed_p1a_check as P1A
    from sim.planing_vessel import SP
    pc = plant(hull, sw=sw, captive_u=0.0)
    v, _ = P1A.eq_rest(pc)
    p = plant(hull, sw=sw)
    s = np.zeros(14)
    s[2] = v[0] + dz
    s[4] = v[1] + np.radians(dtr) / SP
    s[3] = np.radians(roll)
    t0 = time.time()
    r = rest_run(p, s)
    r.update(hull=hull, lab=lab, z0=v[0], th0=v[1], seconds=time.time() - t0)
    return r


def part_h4():
    from studies.parallel_runner import run_parallel
    out(f"S5a (h4)  H4: R3 at rest, zero thrust, 600 s ({CAND})")
    starts = [("equilibrium", 0.0, 0.0, 0.0), ("heave +5 cm", 0.05, 0.0, 0.0),
              ("trim +1 deg", 0.0, 1.0, 0.0),
              ("5 cm, -1 deg, roll 5 deg", 0.05, -1.0, 5.0),
              ("roll 10 deg", 0.0, 0.0, 10.0)]
    jobs = [(AS, lab, dz, dtr, roll, cand()) for lab, dz, dtr, roll in starts]
    res = run_parallel(_r3_job, jobs, gb_per_job=0.25, reserve_gb=3.0)
    for r in res:
        ok = r["diverged"] is None and abs(r["u_end"]) < 5e-3
        if r["lab"] == "roll 10 deg":
            ok = ok and r["heel_mean"] < 0.1
        check("h4", f"H4 {r['lab']}: |u(600 s)| < 5 mm/s"
              + (", mean |heel| over the last 60 s < 0.1 deg"
                 if r["lab"] == "roll 10 deg" else ""), ok,
              f"u end {r['u_end']:+.2e} m/s (max {r['u_max']:.2e}), drift "
              f"{r['x_end']:+.3f} m, mean |heel| {r['heel_mean']:.2e} deg, "
              f"mean dz {r['z_mean'] - r['z0']:+.1e} m ({r['seconds']:.0f} s)")
    dump(f"h4_{CAND}.json", dict(runs=res))
    flush(f"h4_{CAND}.txt")


# ===================================================================== H5
def energy_heel(sw, dt, T, start, lateral_full=False):
    """Rest, dissipation off (roll damping, both cross-flow drags zero),
    surge held at 0: E = rigid + fluid kinetic energy + hydrostatic
    potential of the heeled polygons, U = W z - rho g sum_j int z_abs dA dx
    (calm, a_b = 1). Kinetic: lateral_full -> 1/2 qd^T M5 qd (lateral_cross
    mass matrix of v, w, p, q, r), else the vertical model's heave-pitch
    block and v0's diagonal lateral inertia (Mtot)."""
    from studies import anyspeed_p1a_check as P1A
    from sim.planing_vessel import SP
    p = plant(AS, sw=sw, captive_u=0.0, dt=dt)
    p.c_dc = 0.0
    p.cd_lat = 0.0
    import copy as _c
    p.hull = _c.deepcopy(p.hull)
    p.hull.roll_zeta = 0.0
    v, _ = P1A.eq_rest(plant(AS, sw=sw, captive_u=0.0))
    rg = p.rho * G
    n = p.x.size
    e0 = np.zeros(n)

    def U(s):
        p._hh_force = True
        hh = p._hh_state(float(s[3]), e0)
        p._hh_sections(hh, e0, s[2], p.Th_rest + s[4])
        p._hh_force = False
        za = p.Z_rest + s[2] + SP * p.x * (p.Th_rest + s[4])
        return float(p.m * G * s[2] - rg * ((za * hh.A + hh.Sz).sum()
                                            * p.dx))
    s = np.zeros(14)
    s[2], s[4] = v[0] + start[0], v[1] + start[1]
    s[3] = start[2]
    if len(start) > 3:
        s[7], s[11] = start[3], start[4]
    nst = int(round(T / dt))
    every = max(1, int(round(0.02 / dt)))
    E, tt = [], []
    for i in range(nst + 1):
        if i % every == 0:
            _, aux = p.deriv(s, i * dt)
            qd = np.array([s[7], s[8], s[9], s[10], s[11]])
            if lateral_full:
                Ek = 0.5 * qd @ aux["lat"].M5 @ qd
            else:
                ma = aux["kb"].ma
                A33 = ma.sum() * p.dx
                A35 = (ma * SP * p.x).sum() * p.dx
                A55 = (ma * p.x * p.x).sum() * p.dx
                Ek = 0.5 * ((p.m + A33) * s[8] ** 2 + 2 * A35 * s[8] * s[10]
                            + (p.Iy + A55) * s[10] ** 2
                            + p.Mtot[1, 1] * s[7] ** 2
                            + p.Mtot[3, 3] * s[9] ** 2
                            + p.Mtot[5, 5] * s[11] ** 2)
            E.append(Ek + U(s))
            tt.append(i * dt)
        if i < nst:
            s = p.step(s, i * dt, 0.0, 0.0)
            s[6] = 0.0
    E, tt = np.asarray(E), np.asarray(tt)
    E = E - U(np.r_[0, 0, v[0], 0, v[1], np.zeros(9)])
    w10 = tt <= min(10.0, 0.1 * T)
    l10 = tt >= tt[-1] - min(10.0, 0.1 * T)
    drift = (E[l10].mean() - E[w10].mean()) / E[0] / (T / 100.0)
    return dict(dt=dt, T=T, E0=float(E[0]), drift_per_100s=float(drift),
                max_dev=float(np.abs(E - E[0]).max() / E[0]))


def part_h5():
    out("S5a (h5)  H5: energy at rest, dissipation off (hh)")
    start = (0.01, 0.003, math.radians(10.0))
    rs = []
    for dt, T in ((0.002, 100.0), (0.02, 100.0)):
        t0 = time.time()
        r = energy_heel(cand("hh"), dt, T, start)
        r["seconds"] = time.time() - t0
        rs.append(r)
        out(f"    dt {dt}: E0 {r['E0']:.3f} J, drift "
            f"{100 * r['drift_per_100s']:+.5f} % per 100 s, max dev "
            f"{100 * r['max_dev']:.4f} % ({r['seconds']:.0f} s)")
    check("h5", "H5: rest, dissipation off, heel 10 deg + heave 1 cm + pitch "
          "0.003: energy drift <= 0.1 % per 100 s at dt 0.002 (verlet)",
          abs(rs[0]["drift_per_100s"]) <= 1e-3,
          f"{100 * rs[0]['drift_per_100s']:+.5f} % per 100 s")
    dump("h5.json", dict(runs=rs))
    flush("h5.txt")


def part_l4():
    out("S5a (l4)  L4: energy at rest, dissipation off (hx), integrator-"
        "aware bound")
    start = (0.01, 0.0, math.radians(5.0), 0.2, 0.05)
    rs = []
    for dt in (0.004, 0.002, 0.001):
        t0 = time.time()
        r = energy_heel(cand("hx"), dt, 20.0, start, lateral_full=True)
        r["seconds"] = time.time() - t0
        rs.append(r)
        out(f"    dt {dt}: E0 {r['E0']:.4f} J, max dev "
            f"{100 * r['max_dev']:.5f} %, drift "
            f"{100 * r['drift_per_100s']:+.5f} % per 100 s "
            f"({r['seconds']:.0f} s)")
    e4, e2, e1 = (r["max_dev"] for r in rs)
    ok = e1 <= 5e-4 and (e2 / max(e1, 1e-300) >= 2.5 or e2 <= 1e-6)
    check("l4", "L4: max energy deviation over 20 s E(0.001) <= 0.05 % and "
          "E(0.002) / E(0.001) >= 2.5 (or E(0.002) <= 1e-6)", ok,
          f"E(0.004) {100 * e4:.5f} %, E(0.002) {100 * e2:.5f} %, E(0.001) "
          f"{100 * e1:.5f} %, ratio {e2 / max(e1, 1e-300):.2f}")
    dump("l4.json", dict(runs=rs))
    flush("l4.txt")


# ===================================================================== H8
def lateral_jacobian(p, s, h=1e-6, hk=1e-4):
    """d(v', p', r', phi')/d(v, p, r, phi) at state s (central differences
    of the plant's derivative, step h) and the static roll stiffness -dK/dphi
    (K the roll right-hand side before the mass solve, aux['lat_rhs'][1], at
    zero lateral velocities; central difference with the registered step hk
    = 1e-4 rad -- the first round used h, a deviation the S5a review
    found)."""
    idx = (7, 9, 11, 3)
    J = np.zeros((4, 4))
    for jj, k in enumerate(idx):
        sp_, sm_ = s.copy(), s.copy()
        sp_[k] += h
        sm_[k] -= h
        dp_, _ = p.deriv(sp_, 0.0)
        dm_, _ = p.deriv(sm_, 0.0)
        J[:, jj] = [(dp_[i] - dm_[i]) / (2 * h) for i in (7, 9, 11, 3)]
    sp_, sm_ = s.copy(), s.copy()
    sp_[3] += hk
    sm_[3] -= hk
    Kp = p.deriv(sp_, 0.0)[1]["lat_rhs"][1]
    Km = p.deriv(sm_, 0.0)[1]["lat_rhs"][1]
    return J, -(Kp - Km) / (2 * hk)


def _var_plant(name, hull=AS, **kw):
    """A plant of candidate set name, or of a diagnostic variant (plant
    attribute _s5a_var, sim/planing_anyspeed._s5a_v; plan 'S5a：复查后的修正',
    decision 10): 'hh_ab' hh with the previous round's a_b (and ctr) on the
    lateral, roll and yaw parts; 'hx_e33' / 'hx_none' hx with the table's
    e33 / no heave-heave entry in the roll reaction (the previous round)."""
    base, _, var = name.partition("_")
    p = plant(hull, sw=cand(base), **kw)
    if var == "ab":
        p._s5a_var = {"ab_lat": True}
    elif var in ("e33", "none"):
        p._s5a_var = {"m33": var}
    return p


H8_SETS = ("p1c", "hh", "hx", "hh_ab", "hx_e33", "hx_none")


def _h8_job(arg):
    name, kn = arg
    warnings.filterwarnings("ignore")
    u = kn * KN
    p = _var_plant(name, captive_u=u)
    z, th = p.running_attitude(u)
    s = np.zeros(14)
    s[2], s[4], s[6] = z, th, u
    J, Kphi = lateral_jacobian(p, s)
    ev = np.linalg.eigvals(J)
    return dict(name=name, kn=kn, Kphi=Kphi, ratio=Kphi / (p.m * G * p.GM),
                ab=float(p._a_b(u)), max_re=float(ev.real.max()), ev=ev)


def part_h8():
    from studies.parallel_runner import run_parallel
    out("S5a (h8)  H8: roll stability of the calm branch (captive surge); "
        "stiffness by the registered 1e-4 rad central difference; the "
        "variants hh_ab, hx_e33, hx_none recorded (not gates)")
    kns = (1, 3, 6, 10, 15, 20, 25, 30, 35, 40)
    jobs = [(name, kn) for name in H8_SETS for kn in kns]
    allr = run_parallel(_h8_job, jobs, gb_per_job=0.3, reserve_gb=3.0)
    res = {name: [r for r in allr if r["name"] == name] for name in H8_SETS}
    base = {r["kn"]: r["max_re"] for r in res["p1c"]}
    for name in H8_SETS:
        out(f"  {name}:  kn   K_phi/(W GM)   max Re(eig) (1/s)"
            + ("   K_phi/(W GM a_b)" if name == "hh_ab" else ""))
        for r in res[name]:
            out(f"       {r['kn']:4.0f}   {r['ratio']: .4f}      "
                f"{r['max_re']: .4f}"
                + (f"      {r['ratio'] / r['ab']: .4f}" if name == "hh_ab"
                   else ""))
    for name in ("hh", "hx"):
        stiff = all(r["Kphi"] > 0 for r in res[name])
        newin = max(r["max_re"] - max(base[r["kn"]], 0.0) for r in
                    res[name])
        check("h8", f"H8(a) {name}: static roll stiffness > 0 at every speed "
              f"of the calm branch (1-40 kn)", stiff,
              "K_phi/(W GM) " + ", ".join(f"{r['kn']}:{r['ratio']:.3f}"
                                          for r in res[name]))
        check("h8", f"H8(b) {name}: largest real part of the lateral "
              f"eigenvalues <= max(P1c's, 0) + 1e-3 at every speed",
              newin <= 1e-3, f"largest excess {newin:+.4f} 1/s")
    dump("h8.json", res)
    flush("h8.txt")


# ===================================================================== H9
def part_h9():
    out(f"S5a (h9)  H9: large-angle capsize, signs ({CAND})")
    from studies import anyspeed_p1a_check as P1A
    p0 = plant(AS)
    phv = p0.phi_vanish
    rows = []
    for dphi in (-2.0, 2.0):
        pc = plant(AS, captive_u=0.0)
        v, _ = P1A.eq_rest(pc)
        p = plant(AS)
        s = np.zeros(14)
        s[2], s[4] = v
        s[3] = phv + math.radians(dphi)
        mx = 0.0
        for i in range(int(round(30.0 / p.dt))):
            s = p.step(s, i * p.dt, 0.0, 0.0)
            mx = max(mx, abs(s[3]))
            if p.capsized:
                break
        rows.append((dphi, p.capsized, math.degrees(mx)))
        want = dphi > 0
        check("h9", f"H9 rest: from the vanishing angle {dphi:+.0f} deg "
              f"({math.degrees(phv):.2f} {dphi:+.0f}) the plant "
              f"{'capsizes' if want else 'does not capsize'} in 30 s",
              p.capsized == want, f"capsized {p.capsized}, max heel "
              f"{math.degrees(mx):.1f} deg")
    p = plant(AS)
    s = p.initial_state(25 * KN)
    s[3] = math.radians(15.0)
    mx, back = 0.0, None
    for i in range(int(round(30.0 / p.dt))):
        s = p.step(s, i * p.dt, s[12], 0.0)
        mx = max(mx, abs(s[3]))
        if back is None and abs(s[3]) < math.radians(2.0):
            back = (i + 1) * p.dt
        if p.capsized:
            break
    check("h9", "H9 25 kn, nozzle 0, from heel 15 deg: no capsize in 30 s",
          not p.capsized, f"max heel {math.degrees(mx):.1f} deg, back "
          f"within 2 deg at {back} s, heel at 30 s "
          f"{math.degrees(s[3]):.2f} deg")
    flush(f"h9_{CAND}.txt")


# ==================================================================== H10
def _turn_job(arg):
    name, kn, noz = arg
    warnings.filterwarnings("ignore")
    p = _var_plant(name)
    s = p.initial_state(kn * KN)
    thr = s[12]
    ph, rr, uu = [], [], []
    mx = 0.0
    n = int(round(40.0 / p.dt))
    for i in range(n):
        s = p.step(s, i * p.dt, thr, math.radians(noz))
        mx = max(mx, abs(s[3]))
        if p.capsized or not np.all(np.isfinite(s)):
            break
        if i >= n - int(round(10.0 / p.dt)):
            ph.append(s[3])
            rr.append(s[11])
            uu.append(s[6])
    out_ = dict(name=name, kn=kn, noz=noz, capsized=bool(p.capsized),
                max_heel=math.degrees(mx))
    if ph:
        phm, rm, um = float(np.mean(ph)), float(np.mean(rr)), \
            float(np.mean(uu))
        out_.update(heel=math.degrees(phm), r=math.degrees(rm), u=um / KN,
                    coord=math.degrees(math.atan(um * rm / G)),
                    inward=bool(phm * rm < 0.0))
    return out_


def part_h10():
    from studies.parallel_runner import run_parallel
    out("S5a (h10)  H10: steady turns, heel direction (calm, 40 s, last 10 s "
        "averaged)")
    names = ("p1c", "hh", "hx", "hh_ab", "hx_e33", "hx_none")
    jobs = [(name, kn, noz) for name in names
            for kn in (15.0, 25.0, 35.0) for noz in (10.0, 20.0, -10.0,
                                                     -20.0)]
    res = run_parallel(_turn_job, jobs, gb_per_job=0.3, reserve_gb=3.0)
    for name in names:
        out(f"  {name}:  kn  nozzle  heel (deg)  yaw rate (deg/s)  speed (kn)"
            f"  coordinated atan(u r/g)  inward  max |heel|")
        rs = [r for r in res if r["name"] == name]
        for r in rs:
            if "heel" in r:
                out(f"      {r['kn']:4.0f} {r['noz']:+6.0f}   {r['heel']:+7.2f}"
                    f"     {r['r']:+7.2f}         {r['u']:5.1f}       "
                    f"{r['coord']:+6.1f}            {r['inward']}   "
                    f"{r['max_heel']:.1f}")
            else:
                out(f"      {r['kn']:4.0f} {r['noz']:+6.0f}   capsized "
                    f"(max heel {r['max_heel']:.1f})")
        inward = all(r.get("inward", False) for r in rs)
        safe = all((not r["capsized"]) and r["max_heel"] <= 40.0 for r in rs)
        if name not in ("hh", "hx"):
            # P1c and the diagnostic variants: recorded, not gates
            out(f"    ({name}, recorded: inward in every case {inward}; no "
                f"capsize and |heel| <= 40 deg {safe})")
            continue
        check("h10", f"H10 {name}: steady-turn heel inward in every case",
              inward, "")
        check("h10", f"H10 {name}: no capsize and |heel| <= 40 deg "
              f"(phi_fail) in every case", safe,
              f"largest |heel| {max(r['max_heel'] for r in rs):.1f} deg")
    dump("h10.json", dict(runs=res))
    flush("h10.txt")


# ==================================================================== H11
def part_e1():
    from studies import anyspeed_p1c_check as P1C
    out("S5a (e1)  H11: step time (P0's interleaved method, idle "
        "estimate = ratio to v0 x 1.67 ms)")
    try:
        r = P1C.steptime_sets({"P1c": cand("p1c"), "hh": cand("hh"),
                               "hx": cand("hx")})
        for k, v in r.items():
            out(f"    {k}: {v}")
    except Exception as exc:          # recorded, not a gate
        out(f"    (P1c's timing helper unavailable: {exc!r}; own "
            f"interleaved timing)")
        from sim.wavefield import SeaState
        from hydro import hulls
        from sim.planing_vessel_gz import PlaningVesselGZ
        sea = SeaState(1.0, 5.0, n_freq=12, n_dir=5, seed=11, theta0=2.5)
        ps = {"v0": PlaningVesselGZ(hulls.get("scarab195"), sea, dt=0.02)}
        for k in ("p1c", "hh", "hx"):
            ps[k] = plant(AS, sea=sea, sw=cand(k))
        st = {k: q.initial_state(0.9 * q.hull.u_design) for k, q in
              ps.items()}
        tim = {k: [] for k in ps}
        for rnd in range(6):
            for k, q in ps.items():
                s = st[k]
                t0 = time.perf_counter()
                for i in range(100):
                    s = q.step(s, (rnd * 100 + i) * q.dt, s[12],
                               q.rudder.max * 0.5 * math.sin(0.3 * i * q.dt))
                tim[k].append((time.perf_counter() - t0) / 100)
                st[k] = s
        v0m = np.median(tim["v0"])
        for k in ("p1c", "hh", "hx"):
            m_ = np.median(tim[k])
            out(f"    {k}: {1e3 * m_:.2f} ms/step under load, ratio to v0 "
                f"{m_ / v0m:.2f}, idle estimate {1.67 * m_ / v0m:.2f} ms")
    flush("e1.txt")


# ===================================================================== L8
def _l8_bem(p, h_max, oms):
    """Capytaine's sway / roll added masses (A22, A24, A44) at the
    frequencies oms and its roll stiffness C44 for the plant's hull mesh
    (studies/anyspeed_p1b_check.capytaine_body_exact, edge h_max), cached in
    s5a/l8_bem_<sha1 of the mesh's vertices and faces>.json (S5a review:
    the BEM part depends on the mesh only; the first round re-solved 26
    problems, 70 min, on every run). Missing entries of the h = 0.04 m mesh
    are seeded from the first round's s5a/l8.json when its frequencies
    include them (the mesh code, capytaine_body_exact, unchanged since that
    run: file dates), else solved. Returns (A {omega: (A22, A24, A44)},
    C44, how)."""
    import capytaine as cpt
    from studies import anyspeed_p1b_check as P1B
    body = P1B.capytaine_body_exact(p, h_max)
    m = body.mesh
    hsh = hashlib.sha1(np.ascontiguousarray(m.vertices, float).tobytes()
                       + np.ascontiguousarray(m.faces).tobytes()).hexdigest()
    path = os.path.join(S5A_DIR, f"l8_bem_{hsh[:16]}.json")
    cache = dict(h_max=h_max, mesh_sha1=hsh, n_faces=int(m.nb_faces), A={},
                 C44=None, sources={})
    if os.path.exists(path):
        with open(path) as f:
            cache = json.load(f)
    how = []
    old = os.path.join(S5A_DIR, "l8.json")
    seed = None
    if abs(h_max - 0.04) < 1e-12 and os.path.exists(old):
        with open(old) as f:
            j = json.load(f)
        seed = {round(float(o), 6): a for o, a in zip(j["oms"], j["A"])}
        seed_c44 = j.get("C44_cap")
    for om in oms:
        key = f"{float(om):.6f}"
        if key in cache["A"]:
            continue
        if seed is not None and round(float(om), 6) in seed:
            cache["A"][key] = [float(v) for v in seed[round(float(om), 6)]]
            cache["sources"][key] = "seeded from the first round's l8.json"
            continue
        r_ = {}
        for dof in ("Sway", "Roll"):
            pb = cpt.RadiationProblem(body=body, omega=float(om),
                                      radiating_dof=dof, rho=p.rho, g=G)
            sol = cpt.BEMSolver().solve(pb)
            r_[dof] = {k: float(v) for k, v in sol.added_masses.items()}
        cache["A"][key] = [r_["Sway"]["Sway"], r_["Sway"]["Roll"],
                           r_["Roll"]["Roll"]]
        cache["sources"][key] = "solved " + time.strftime("%Y-%m-%d %H:%M")
    if cache.get("C44") is None:
        if seed is not None and seed_c44 is not None:
            cache["C44"] = float(seed_c44)
            cache["sources"]["C44"] = "seeded from the first round's l8.json"
        else:
            hs = body.compute_hydrostatic_stiffness(rho=p.rho, g=G)
            try:
                cache["C44"] = float(hs.sel(influenced_dof="Roll",
                                            radiating_dof="Roll"))
            except Exception:
                cache["C44"] = float("nan")
            cache["sources"]["C44"] = "computed"
    with open(path + ".tmp", "w") as f:
        json.dump(cache, f, indent=1)
    os.replace(path + ".tmp", path)
    for om in oms:
        how.append(cache["sources"].get(f"{float(om):.6f}", "cached"))
    A = {om: tuple(cache["A"][f"{float(om):.6f}"]) for om in oms}
    return A, float(cache["C44"]), (f"mesh sha1 {hsh[:12]}, {m.nb_faces} "
                                    f"faces, cache {os.path.basename(path)}"
                                    f": " + "; ".join(sorted(set(how))))


def part_l8():
    import capytaine as cpt
    from studies import anyspeed_p1b_check as P1B
    from studies import anyspeed_p1a_check as P1A
    out("S5a (l8)  L8: roll natural frequency at rest, plant vs Capytaine "
        "(consistency, not validation)")
    res = {}
    for name in ("hh", "hx"):
        pc = plant(AS, sw=cand(name), captive_u=0.0)
        pc.hull = __import__("copy").deepcopy(pc.hull)
        pc.hull.roll_zeta = 0.0
        pc.cd_lat = 0.0
        v, _ = P1A.eq_rest(plant(AS, sw=cand(name), captive_u=0.0))
        s = np.zeros(14)
        s[2], s[4] = v
        J, Kphi = lateral_jacobian(pc, s, h=1e-6)
        ev = np.linalg.eigvals(J)
        osc = ev[np.abs(ev.imag) > 1e-6]
        w = float(np.abs(osc).max()) if osc.size else float("nan")
        res[name] = dict(omega=w, Kphi=Kphi, ev=ev)
        out(f"    plant {name}: roll stiffness {Kphi:.1f} N m/rad (W GM "
            f"{pc.m * G * pc.GM:.1f}), eigenvalues {np.round(ev, 4)}, roll "
            f"frequency {w:.4f} rad/s")
    p = plant(AS, sw=cand("hh"), captive_u=0.0)
    C44_plant = res["hh"]["Kphi"]
    # S5a review (test efficiency): the BEM part does not depend on the
    # plant's code, only on the mesh -- cached by the mesh's hash, 4
    # frequencies around the root (3-4.5 rad/s; the root is ~3.9)
    oms = np.array([3.0, 3.5, 4.0, 4.5])
    A, C44_cap, how = _l8_bem(p, 0.04, oms)
    A22 = np.interp
    a22 = np.array([A[o][0] for o in oms])
    a24 = np.array([A[o][1] for o in oms])
    a44 = np.array([A[o][2] for o in oms])

    def root(C44, a22=a22, a24=a24, a44=a44):
        w = 4.0
        for _ in range(100):
            m22 = p.m + A22(w, oms, a22)
            I = p.Ix + A22(w, oms, a44) - A22(w, oms, a24) ** 2 / m22
            wn = math.sqrt(C44 / I)
            if abs(wn - w) < 1e-10:
                break
            w = wn
        return wn
    w_cap = root(C44_plant)
    w_cap2 = root(C44_cap) if np.isfinite(C44_cap) else float("nan")
    out(f"    Capytaine (same geometry mesh, h 0.04 m; {how}): A22, A24, A44 "
        f"at 3-4.5 rad/s:")
    for o in oms:
        out(f"      {o:.2f}: {A[o][0]:8.1f} {A[o][1]:8.1f} {A[o][2]:8.1f}")
    out(f"    sway-roll root with the plant's C44 {C44_plant:.1f}: "
        f"{w_cap:.4f} rad/s; with Capytaine's C44 {C44_cap:.1f}: "
        f"{w_cap2:.4f} rad/s")
    # the one-time mesh-sensitivity note: the same 4 frequencies on a
    # coarser mesh (h 0.06 m), cached the same way
    Ac, _, howc = _l8_bem(p, 0.06, oms)
    w_c = root(C44_plant, np.array([Ac[o][0] for o in oms]),
               np.array([Ac[o][1] for o in oms]),
               np.array([Ac[o][2] for o in oms]))
    out(f"    mesh sensitivity (h 0.06 m, {howc}): root {w_c:.4f} rad/s, "
        f"{100 * (w_c - w_cap) / w_cap:+.2f} % against h 0.04 m")
    for name in ("hx", "hh"):
        e = abs(res[name]["omega"] - w_cap) / w_cap
        if name == "hx":
            check("l8", "L8 hx: roll natural frequency at rest within 10 % "
                  "of the Capytaine sway-roll root (consistency)", e <= 0.10,
                  f"plant {res[name]['omega']:.4f}, BEM {w_cap:.4f} rad/s, "
                  f"{100 * e:+.2f} %")
        else:
            out(f"    hh (v0 lateral inertia, recorded): plant "
                f"{res[name]['omega']:.4f} rad/s, {100 * e:+.2f} %")
    dump("l8.json", dict(plant=res, oms=oms, A=[A[o] for o in oms],
                         C44_plant=C44_plant, C44_cap=C44_cap, w_cap=w_cap,
                         w_cap2=w_cap2))
    flush("l8.txt")


# =================================================================== L1''
# Supplementary registration (after the first run of L1, before this
# form ran; plan 'S5a'): the registered L1 used a body-fixed panelling
# whose waterline panels are cut wherever the surface falls, so the
# sequence in N is noisy, and at a contact angle above 90 deg (one side of
# every heeled case) the gradient is singular there (phi ~ r^lam, lam =
# pi / (2 alpha) < 1), which uniform panels resolve at order 2 lam - 1.
# L1'' panels each smooth piece of the WETTED contour between its fixed
# corners and the waterline points (so the mesh deforms smoothly with the
# configuration and has a node at each waterline point), graded towards
# both ends of every piece (x = s^g / (s^g + (1 - s)^g), g = 3: the first
# panel ~ N^-3), and takes d phi / dt at fixed body points by the chain
# rule over the moving panel centres (no interpolation): D phi/Dt|_body =
# d phi_i / dt - phi_s ds_i/dt.
class Sec2Dg:
    """A section as body-frame pieces between its corners: for each piece
    (y, z) vertices of a straight line or an ellipse arc, parametrised by
    a body coordinate; the wetted contour of a configuration is the part of
    each piece below z = 0 (convex body: one wetted interval), panelled
    with N panels per piece, graded towards both ends."""

    def __init__(self, kind, N, g=3.0):
        self.kind, self.N, self.g = kind, N, g
        if kind == "ellipse":
            self.a_, self.b_ = 0.8, 0.35
        else:
            yc, tb, hG = 0.9, math.tan(math.radians(20.0)), 0.55
            dch = yc * tb
            top = 1.2
            self.V = [(yc, top - hG), (yc, dch - hG), (0.0, -hG),
                      (-yc, dch - hG), (-yc, top - hG)]

    def _grade(self, n):
        s = np.linspace(0.0, 1.0, n + 1)
        g = self.g
        return s ** g / (s ** g + (1.0 - s) ** g)

    def _pt(self, u):
        """Body point at body coordinate u (ellipse: the angle; V + wall:
        the polyline arclength), and the arclength itself."""
        if self.kind == "ellipse":
            return np.stack([self.a_ * np.cos(u), self.b_ * np.sin(u)], -1)
        V = np.array(self.V)
        seg = np.linalg.norm(np.diff(V, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        k = np.clip(np.searchsorted(cum, u, side="right") - 1, 0,
                    len(seg) - 1)
        f = (u - cum[k]) / seg[k]
        return V[k] + f[..., None] * (V[k + 1] - V[k])

    def wetted(self, yG, zG, phi):
        """(a, b, u_mid, piece) of the graded wetted panels: the waterline
        points by bisection on the body coordinate, each wetted piece
        panelled; u_mid the body coordinate of each panel's centre (as the
        point (a + b) / 2 maps back), piece the index of its piece."""
        c, s = math.cos(phi), math.sin(phi)

        def zeta(u):
            P = self._pt(np.atleast_1d(u))
            return zG + P[..., 0] * s + P[..., 1] * c

        if self.kind == "ellipse":
            # the lowest point, then the two crossings
            us = np.linspace(-math.pi, math.pi, 2001)
            zz = zeta(us)
            u0 = us[np.argmin(zz)]
            cuts = []
            for lo, hi in ((u0, u0 + math.pi), (u0 - math.pi, u0)):
                for _ in range(200):
                    mid = 0.5 * (lo + hi)
                    if (zeta(mid)[0] > 0.0) == (zeta(hi)[0] > 0.0):
                        hi = mid
                    else:
                        lo = mid
                cuts.append(0.5 * (lo + hi))
            # counter-clockwise: from the + side crossing down to the -
            u_end, u_start = cuts[0], cuts[1]
            if u_start > u_end:
                u_start -= 2 * math.pi
            # from the crossing at larger y... the order along the CCW
            # parametrisation is increasing u; the wetted arc is the one
            # containing u0
            if not (u_start < u0 < u_end or u_start < u0 + 2 * math.pi
                    < u_end):
                u_start, u_end = u_end, u_start + 2 * math.pi
            # traversed clockwise (u decreasing), hydro/section2d's
            # convention: from the +y waterline round the bottom
            pieces = [(u_end, u_start)]
        else:
            V = np.array(self.V)
            seg = np.linalg.norm(np.diff(V, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            zc = np.array([zeta(x)[0] for x in cum])
            pieces = []
            for k in range(len(seg)):
                lo, hi = cum[k], cum[k + 1]
                w0, w1 = zc[k] <= 0.0, zc[k + 1] <= 0.0
                if not (w0 or w1):
                    continue
                if w0 != w1:
                    a_, b_ = lo, hi
                    for _ in range(200):
                        mid = 0.5 * (a_ + b_)
                        if (zeta(mid)[0] <= 0.0) == w0:
                            a_ = mid
                        else:
                            b_ = mid
                    x = 0.5 * (a_ + b_)
                    lo, hi = (lo, x) if w0 else (x, hi)
                pieces.append((lo, hi))
        A_, B_, U_, K_ = [], [], [], []
        for k, (lo, hi) in enumerate(pieces):
            us = lo + (hi - lo) * self._grade(self.N)
            P = self._pt(us)
            E = np.stack([yG + P[:, 0] * c - P[:, 1] * s,
                          zG + P[:, 0] * s + P[:, 1] * c], 1)
            A_.append(E[:-1])
            B_.append(E[1:])
            U_.append(0.5 * (us[:-1] + us[1:]))
            K_.append(np.full(self.N, k))
        a = np.concatenate(A_)
        b = np.concatenate(B_)
        # the waterline points exactly on z = 0
        a[0, 1] = 0.0
        b[-1, 1] = 0.0
        return a, b, np.concatenate(U_), np.concatenate(K_)


def _l1_body(m, yG, zG, phi):
    """Earth points m -> body frame about the CG."""
    c, s = math.cos(phi), math.sin(phi)
    r = m - np.array([yG, zG])
    return np.stack([r[..., 0] * c + r[..., 1] * s,
                     -r[..., 0] * s + r[..., 1] * c], -1)


def l1g_forces(sec, t0, mot, dt=1e-5, eps=1e-6):
    """L1'': (F_pressure, F_lagrange, F_impulse_only) on the graded,
    deforming panelling (Sec2Dg); rho = 1."""
    (yG, zG, phi), nu, nud = _l1_config(t0, mot)
    a, b, um, pc = sec.wetted(yG, zG, phi)
    ph0, geo = Sec2D.solve(a, b, yG, zG)
    m, n, h, tt = geo["m"], geo["n"], geo["h"], geo["t"]
    N = len(um)
    # arclength along each piece, for the tangential fit
    sarc = np.zeros(N)
    for k in np.unique(pc):
        j = np.nonzero(pc == k)[0]
        hh_ = h[j]
        sarc[j] = np.cumsum(hh_) - 0.5 * hh_
    wl_first = (0, -0.5 * h[0])           # waterline at the start of piece 0
    wl_last = (pc[-1], sarc[-1] + 0.5 * h[-1])

    def phis(val):
        """d phi / ds at the centres: a quadratic through the 3 nearest
        samples of the same piece (with phi = 0 at a waterline end)."""
        out_ = np.zeros(N)
        for i in range(N):
            j = np.nonzero(pc == pc[i])[0]
            xs = list(sarc[j])
            vs = list(val[j])
            if pc[i] == 0:
                xs.append(wl_first[1])
                vs.append(0.0)
            if pc[i] == wl_last[0]:
                xs.append(wl_last[1])
                vs.append(0.0)
            out_[i] = _l1_fit(sarc[i], xs, vs)[1]
        return out_

    def state(tq):
        (yq, zq, pq), nq, _ = _l1_config(tq, mot)
        aq, bq, uq, _ = sec.wetted(yq, zq, pq)
        phq, gq = Sec2D.solve(aq, bq, yq, zq)
        return phq @ nq, _l1_body(gq["m"], yq, zq, pq)
    vp, mbp = state(t0 + dt)
    vm, mbm = state(t0 - dt)
    val0 = ph0 @ nu
    phs = phis(val0)
    # the collocation points move along the body as the mesh deforms: the
    # rate of each along its panel (body frame), the chain rule's term
    cph, sph = math.cos(phi), math.sin(phi)
    tb = np.stack([tt[:, 0] * cph + tt[:, 1] * sph,
                   -tt[:, 0] * sph + tt[:, 1] * cph], 1)
    sdot = (((mbp - mbm) / (2 * dt)) * tb).sum(1)
    Dphi = (vp - vm) / (2 * dt) - phs * sdot
    vb = np.stack([nu[0] - nu[2] * (m[:, 1] - zG),
                   nu[1] + nu[2] * (m[:, 0] - yG)], 1)
    phn = (vb * n).sum(1)
    grad = phn[:, None] * n + phs[:, None] * tt
    pt = Dphi - (vb * grad).sum(1)
    p = -(pt + 0.5 * (grad * grad).sum(1))
    dF = p[:, None] * n * h[:, None]
    Fp = np.array([dF[:, 0].sum(), dF[:, 1].sum(),
                   ((m[:, 0] - yG) * dF[:, 1]
                    - (m[:, 1] - zG) * dF[:, 0]).sum()])

    def Mof(z_, p_):
        a_, b_ = sec.wetted(yG, z_, p_)[:2]
        return Sec2D.solve(a_, b_, yG, z_)[1]["M"]
    M0 = geo["M"]
    Mz = (Mof(zG + eps, phi) - Mof(zG - eps, phi)) / (2 * eps)
    Mph = (Mof(zG, phi + eps) - Mof(zG, phi - eps)) / (2 * eps)
    Mdot = Mz * nu[1] + Mph * nu[2]
    imp = -(M0 @ nud + Mdot @ nu)
    react = np.array([0.0, 0.5 * nu @ Mz @ nu, 0.5 * nu @ Mph @ nu])
    return Fp, imp + react, imp


def part_l1g():
    out("S5a (l1g)  L1'' (supplementary): the same comparison on graded "
        "panels that deform with the configuration (a node at each "
        "waterline point; d phi/dt at fixed body points by the chain rule)")
    rows = []
    worst = 0.0
    for name, phi0 in (("ellipse", 0.0), ("ellipse", 10.0),
                       ("ellipse", 25.0), ("vwall", 0.0), ("vwall", 10.0),
                       ("vwall", 25.0)):
        zc = -0.05 if name == "ellipse" else 0.55 - 0.75
        mot = [(0.0, 0.10, 1.3, 0.4), (zc, 0.04, 1.7, 1.1),
               (math.radians(phi0), math.radians(4.0), 2.1, 0.7)]
        res = []
        for N in NS_L1G:
            Fp, FL, Fi = l1g_forces(Sec2Dg(name, N), 0.37, mot)
            res.append((N, Fp, FL, Fi))
        scale = max(np.abs(res[-1][2][:2]).max(), 1e-12)
        sc = np.array([scale, scale, max(abs(res[-1][2][2]), 1e-12)])
        D = np.array([r[1] - r[2] for r in res])
        e = np.abs(D) / sc
        out(f"  {name} heel {phi0:4.0f} deg: F_L = {res[-1][2]}")
        for (N, Fp, FL, Fi), ee in zip(res, e):
            out(f"     N {N:4d}: F_p {Fp}  |F_p - F_L| / scale "
                f"{ee[0]:.1e} {ee[1]:.1e} {ee[2]:.1e}")
        dN, dN2, dN4 = D[-1], D[-2], D[-3]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs((dN4 - dN2) / (dN2 - dN))
        pord = np.clip(np.log2(np.where(np.isfinite(ratio) & (ratio > 1),
                                        ratio, 2.0)), 0.5, 4.0)
        Dinf = dN - (dN2 - dN) / (2 ** pord - 1)
        einf = np.abs(Dinf) / sc
        out(f"     extrapolated {einf}, orders {pord}; finest {e[-1]}; "
            f"without dT/dq {np.abs(res[-1][2] - res[-1][3]) / sc}")
        worst = max(worst, float(np.minimum(einf, e[-1]).max()))
        rows.append(dict(case=name, heel=phi0, res=res, einf=einf))
    check("l1g", "L1'': Lagrangian formula vs direct pressure on graded, "
          "deforming panels: within 1e-3 of the scale (finest or "
          "extrapolated), all six cases", worst <= 1e-3,
          f"largest {worst:.1e}")
    dump("l1g.json", dict(rows=rows))
    flush("l1g.txt")


NS_L1G = (32, 64, 128, 256, 512)


# ===================================================================== H7
def part_twin():
    """H7: P1a's part twin (Froude twins at lam 1.5 and 0.6: captive roots
    and a 600-step open loop in a Froude-scaled sea) on the candidate."""
    from studies import anyspeed_p1a_check as P1A
    use_candidate()
    P1A.OUT = OUT
    P1A.STAMP = STAMP
    P1A.part_twin()
    for c in P1A.CHECKS:
        CHECKS.append(dict(c, part=f"twin_{CAND}"))
    # P1a's part writes twin.txt; keep one file per candidate
    src = os.path.join(OUT, "twin.txt")
    if os.path.exists(src):
        os.replace(src, os.path.join(OUT, f"twin_{CAND}.txt"))


# ==================================================================== H12
def _golden_job(arg):
    sw, opts = arg
    from hydro import hulls
    from sim.planing_anyspeed import PlaningVesselAS
    from studies import anyspeed_regress as AR
    from studies.anyspeed_regress_as import ON_EXTRA
    warnings.filterwarnings("ignore")
    h = hulls.get(AS)
    return AR.golden_runs(lambda p: PlaningVesselAS(
        h, p.sea, dt=p.dt, record=True, test_switches=sw), extra=ON_EXTRA,
        **opts)


def golden_sets_s5a2():
    """The goldens this round re-pins (plan 'S5a：复查后的修正', H12): name,
    switches, golden_runs options, file name, the base set (same options)
    whose runs it must differ from, and whether its calm runs can differ
    (fk_exact and orbital add nothing in calm water by design)."""
    from sim.planing_anyspeed import AS1_P1A
    p1c, p1a = cand("p1c"), tuple(AS1_P1A)
    n0c5 = dict(nozzle=False, calm_nozzle_deg=5.0)
    out_ = []
    for name, sw, base, opts, calm_differs in (
            ("hh", cand("hh"), p1c, n0c5, True),
            ("hx", cand("hx"), p1c, n0c5, True),
            ("p1a_phi0", p1a + ("lateral_phi0",), p1a, n0c5, True),
            ("p1a_fk_orb", p1a + ("fk_exact", "orbital"), p1a + ("fk_exact",),
             {}, False)):
        tag = "_n0c5" if opts else ""
        out_.append(dict(name=name, sw=sw, base=base, opts=opts,
                         calm_differs=calm_differs,
                         file="golden_as_" + "+".join(sw) + tag
                         + "_s5a2.npz"))
    return out_


def part_goldens():
    """H12 (N2), S5a review round: the goldens of the sets this round
    changed (hh, hx; lateral_phi0 and orbital through issues C and E), the
    S5a ones with the calm runs' nozzle held at 5 deg (issue K: the first
    round's nozzle-at-zero calm runs stayed upright and pinned nothing the
    switches change) and the task runs' at zero: two processes bit for bit,
    the calm runs' largest heel <= 15 deg (a set over it falls back to the
    nozzle at zero, recorded), at least one run of each sea different from
    its base set's (calm runs only where the switch acts in calm water);
    written once to studies/_cache/anyspeed/ (never overwritten), stamped
    record goldens_s5a.txt / .json (issue M)."""
    from studies.parallel_runner import run_parallel
    out("S5a review (goldens)  H12: the goldens of the sets this round "
        "changed, two processes each, against their base sets")
    S = golden_sets_s5a2()
    args = []
    for g in S:
        args += [(g["sw"], g["opts"]), (g["sw"], g["opts"]),
                 (g["base"], g["opts"])]
    res = run_parallel(_golden_job, args, gb_per_job=0.4, reserve_gb=3.0)
    rows = []

    def maxheel(runs, sea):
        return max(float(np.degrees(np.abs(np.asarray(v)[:, 3]).max()))
                   for k, v in runs.items()
                   if k.startswith(sea + "_") and k.endswith("_s"))
    for i, g in enumerate(S):
        a_, b_, base = res[3 * i], res[3 * i + 1], res[3 * i + 2]
        opts = g["opts"]
        heel = maxheel(a_, "calm")
        fallback = False
        if heel > 15.0 and opts.get("calm_nozzle_deg") is not None:
            # registered: over 15 deg with the nozzle at 5 deg, keep the
            # nozzle at zero (recorded)
            fallback = True
            opts = dict(nozzle=False)
            a_, b_, base = run_parallel(
                _golden_job, [(g["sw"], opts), (g["sw"], opts),
                              (g["base"], opts)], gb_per_job=0.4,
                reserve_gb=3.0)
            g = dict(g, opts=opts, file=g["file"].replace("_n0c5", "_n0"))
        same = set(a_) == set(b_) and all(
            np.array_equal(np.asarray(a_[k]), np.asarray(b_[k])) for k in a_)
        heel_c, heel_t = maxheel(a_, "calm"), maxheel(a_, "task")
        dif = {sea: [k for k in a_ if k.startswith(sea + "_")
                     and k.endswith("_s")
                     and not np.array_equal(np.asarray(a_[k]),
                                            np.asarray(base[k]))]
               for sea in ("calm", "task")}
        path = os.path.join(HERE, "_cache", "anyspeed", g["file"])
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
        rows.append(dict(name=g["name"], sw=g["sw"], opts=opts,
                         fallback=fallback, same=same, heel_calm=heel_c,
                         heel_task=heel_t, differ=dif, path=path, sha1=sha,
                         file=wrote))
        out(f"    {g['name']}: options {opts}{' (fallback)' if fallback else ''}"
            f"; two processes identical {same}; largest heel calm "
            f"{heel_c:.2f} deg, task sea {heel_t:.2f} deg; runs differing "
            f"from the base set: calm {len(dif['calm'])}, task "
            f"{len(dif['task'])}; {g['file']} {wrote}, sha1 {sha}")
        check("goldens", f"H12 {g['name']}: golden runs identical in two "
              f"processes", same)
        check("goldens", f"H12 {g['name']}: largest heel of the calm golden "
              f"runs <= 15 deg", heel_c <= 15.0, f"{heel_c:.1f} deg")
        need = (["calm"] if g["calm_differs"] else []) + ["task"]
        check("goldens", f"H12 {g['name']}: at least one run of each sea "
              f"({', '.join(need)}) differs from its base set's",
              all(dif[s_] for s_ in need),
              f"calm {len(dif['calm'])}, task {len(dif['task'])}")
    dump("goldens_s5a.json", dict(rows=rows))
    flush("goldens_s5a.txt")


# ============================================================ S5a review
# The checks registered in plan 'S5a：复查后的修正' (before they ran)
def _m33_of(p, d, u):
    """The heave equation's added mass of plant p (no wagner_entry) as an
    independent function of the keel depth: Zarnick's k_a pi/2 rho min(kc
    d+, yc)^2 (sim/planing_vessel._section's), times the transom correction
    ctr(u)."""
    kc = 0.5 * math.pi / p.tanb
    c = np.minimum(kc * np.maximum(d, 0.0), p.yc)
    return p.k_a * 0.5 * math.pi * p.rho * c * c * p._ctr(u)


def _Mfull(p, d, phi, u):
    """_Mt (M~, independent assembly) with the heave-heave entry the heave
    equation's m33: the plant's ONE kinetic energy (issue A)."""
    M = _Mt(p, d, phi)
    M[:, 1, 1] = _m33_of(p, d, u)
    return M


def part_l1pp():
    from sim.wavefield import SeaState
    out("S5a review (l1pp)  L1'+: the plant's cross-plane strip forces vs "
        "the Lagrangian of one kinetic energy T = 1/2 m33 W_r^2 + 1/2 nu^T M~ "
        "nu (sway and heave rows on M~, the roll row on the full M with the "
        "heave equation's m33), assembled independently by central "
        "differences of the table; n0 by differencing nu at fixed body x "
        "along the motion (positions advanced by their rates, time +- h, "
        "velocities fixed), not copied from the code")
    rng = np.random.default_rng(15)
    HS = (1e-5, 1e-6, 1e-7)
    worst = {h: 0.0 for h in HS}
    best_state = 0.0
    nstate = 0
    for setname in ("hx", "hx+orbital"):
        sw = cand("hx") + (("orbital",) if setname == "hx+orbital" else ())
        for hull in (AS, HB):
            for seaname in ("calm", "sea"):
                sea = (calm() if seaname == "calm" else
                       SeaState(0.6, 4.0, n_freq=8, n_dir=3, seed=4,
                                theta0=1.2))
                p = plant(hull, sea=sea, sw=sw)
                for i in range(4):
                    s = p.initial_state(rng.uniform(0, 30) * KN)
                    s[3] = math.radians(rng.uniform(-25, 25))
                    s[5] = rng.uniform(-math.pi, math.pi)
                    s[7], s[8], s[9] = rng.normal(0, [0.3, 0.2, 0.3])
                    s[10], s[11] = rng.normal(0, [0.05, 0.1])
                    t = float(rng.uniform(0, 30))
                    ds, aux = p.deriv(s, t)
                    lat = aux["lat"]
                    d = aux["d"].copy()
                    u = float(s[6])
                    vr0, Wr0 = lat.vr.copy(), lat.Wr.copy()
                    got = np.stack([lat.fy, lat.fz, lat.fL], 1).copy()
                    nu = np.stack([vr0, Wr0, np.full(d.size, s[9])], 1)
                    M0 = _Mt(p, d, s[3])
                    errs = []
                    for h in HS:
                        # dT~/dz: each strip's d moves by -dz
                        Tz = 0.5 * np.einsum("ji,jik,jk->j", nu,
                                             (_Mt(p, d - h, s[3])
                                              - _Mt(p, d + h, s[3]))
                                             / (2 * h), nu)
                        # dT/dphi at a fixed CG on the full M (d from the
                        # plant's formula, m33 the heave equation's)
                        sp_, sm_ = s.copy(), s.copy()
                        sp_[3] += h
                        sm_[3] -= h
                        dp_, dm_ = _d_of(p, sp_, t), _d_of(p, sm_, t)
                        Tp = 0.5 * np.einsum(
                            "ji,jik,jk->j", nu,
                            (_Mfull(p, dp_, s[3] + h, u)
                             - _Mfull(p, dm_, s[3] - h, u)) / (2 * h), nu)
                        # along the motion: positions advanced by their
                        # rates, waves at t +- h, velocities held
                        kin = np.zeros(14)
                        kin[:6] = ds[:6]
                        s1, s0 = s + h * kin, s - h * kin
                        _, a1 = p.deriv(s1, t + h)
                        d1, vr1, Wr1 = (a1["d"].copy(), a1["lat"].vr.copy(),
                                        a1["lat"].Wr.copy())
                        _, a0 = p.deriv(s0, t - h)
                        d0, vr0_, Wr0_ = (a0["d"].copy(), a0["lat"].vr.copy(),
                                          a0["lat"].Wr.copy())
                        Md = (_Mt(p, d1, s1[3]) - _Mt(p, d0, s0[3])) / (2 * h)
                        nud = np.stack([(vr1 - vr0_) / (2 * h),
                                        (Wr1 - Wr0_) / (2 * h),
                                        np.zeros(d.size)], 1)
                        F = (-np.einsum("jik,jk->ji", M0, nud)
                             - np.einsum("jik,jk->ji", Md, nu))
                        F[:, 1] += Tz
                        F[:, 2] += Tp
                        sc = max(np.abs(got).max(), 1e-9)
                        e_ = float(np.abs(got - F).max() / sc)
                        errs.append(e_)
                        worst[h] = max(worst[h], e_)
                    best_state = max(best_state, min(errs))
                    nstate += 1
                    out(f"    {setname:10s} {hull} {seaname:4s} heel "
                        f"{math.degrees(s[3]):6.2f} deg, u "
                        f"{s[6] / KN:4.1f} kn: " + ", ".join(
                            f"h {h:.0e} {e_:.1e}" for h, e_ in zip(HS, errs)))
    out("    largest over the states, per step: " + ", ".join(
        f"h {h:.0e} {worst[h]:.2e}" for h in HS))
    check("l1pp", f"L1'+: strip forces (fy, fz, fL) of lateral_cross vs the "
          f"one-T Lagrangian (roll row on the full M with m33), n0 by "
          f"differencing ({nstate} random heeled states: hx and hx + orbital, "
          f"calm and sea, two hulls; best of steps 1e-5..1e-7 per state): "
          f"<= 1e-6 relative", best_state <= 1e-6, f"{best_state:.2e}")
    flush("l1pp.txt")


def _wave_one(p, s, t, d=None):
    """(vo, dvo/dt at fixed body x, dvo/dt at a fixed earth point) of the
    plant's one regular component at the stations' centroid depths, from
    the wave's own definition (not the plant's code): v_o = a w e^{k z_c}
    (lam / k) cos psi, psi = kx X + ky Y - w t + phi, the station moving
    with (u, v + x r) horizontally; for a captive hull (v = r = 0, z_c's
    rate second order) d/dt|_body = (w - kappa u) a w e^{k z_c} (lam / k)
    sin psi -- the encounter frequency -- and d/dt|_earth = w a w e^{k
    z_c} (lam / k) sin psi. d: the keel depths for z_c (the calm ones for
    the linear part: the wave's own elevation moves z_c at first order,
    which is second order in v_o)."""
    sea = p.sea
    a = float(np.atleast_1d(sea.a)[0])
    w = float(np.atleast_1d(sea.w)[0])
    k = float(np.atleast_1d(sea.k)[0])
    th = float(np.atleast_1d(sea.th)[0])
    ph0 = float(np.atleast_1d(sea.phi)[0])
    psi_h = float(s[5])
    kx, ky = k * math.cos(th), k * math.sin(th)
    kap = kx * math.cos(psi_h) + ky * math.sin(psi_h)
    lam = -kx * math.sin(psi_h) + ky * math.cos(psi_h)
    X = s[0] + p.x * math.cos(psi_h)
    Y = s[1] + p.x * math.sin(psi_h)
    if d is None:
        d = p.deriv(s, t)[1]["d"]
    from sim.planing_anyspeed import _vsec
    A_, m1, _ = _vsec(d, p.yc, p.tanb)
    dp = np.maximum(d, 0.0)
    zc = np.where(A_ > 0.0, -(dp - m1 / np.where(A_ > 0.0, A_, 1.0)), 0.0)
    amp = a * w * np.exp(k * zc) * (lam / k)
    psi = kx * X + ky * Y - w * t + ph0
    u = float(s[6])
    return (amp * np.cos(psi), (w - kap * u) * amp * np.sin(psi),
            w * amp * np.sin(psi), d)


def part_stf():
    from sim.test_vessel import Monochromatic
    out("S5a review (stf)  STF: lateral diffraction at speed (25 kn, captive "
        "attitude, regular wave lambda 12 m, linear part by +-1e-6 m) vs the "
        "Salvesen-Tuck-Faltinsen strip formula -- each strip's added mass "
        "times the wave velocity's ENCOUNTER-frequency rate, plus the transom "
        "term and the Munk sum -- with the plant's own coefficients; the "
        "diffraction isolated as (set + orbital) - (set)")
    u = 25.0 * KN
    eps = 1e-6
    lam_w = 12.0
    worst = {}
    absdev = {}
    for label, sw_on, sw_off, phi0 in (
            ("P1c + orbital (orbital's diffraction: strip a22, transom "
             "plate m22)", cand("p1c") + ("orbital",), cand("p1c"), 0.0),
            ("P1c + orbital + lateral_phi0", cand("p1c")
             + ("lateral_phi0", "orbital"), cand("p1c") + ("lateral_phi0",),
             0.0),
            ("hx + orbital (heel 0)", cand("hx") + ("orbital",), cand("hx"),
             0.0)):
        e_max = dev_max = 0.0
        for mu in (45.0, 90.0, 135.0):
            w_reg = math.sqrt(G * 2.0 * math.pi / lam_w)

            def mk(sw, a):
                # Monochromatic(omega, amplitude, theta0): lambda 12 m
                return plant(AS, sea=Monochromatic(w_reg, a,
                                                   theta0=math.radians(mu)),
                             sw=sw, captive_u=u, gz=False)
            pc = mk(sw_off, 0.0)
            pcon = mk(sw_on, 0.0)
            z, th = pc.running_attitude(u)
            k = 2.0 * math.pi / lam_w
            w = math.sqrt(G * k)
            we = abs(w - k * math.cos(math.radians(mu)) * u)
            Te = 2.0 * math.pi / we
            # the linear part by Richardson over +-eps and +-2 eps: the
            # cross-flow drag v_r |v_r| is odd and quadratic in the
            # amplitude, so a plain central difference keeps a part ~ eps
            # (2e-6 of the amplitude at 1e-6; implementation note, plan)
            P = {key: mk(sw, a) for key, sw, a in (
                ("on+", sw_on, eps), ("on-", sw_on, -eps),
                ("off+", sw_off, eps), ("off-", sw_off, -eps),
                ("on++", sw_on, 2 * eps), ("on--", sw_on, -2 * eps),
                ("off++", sw_off, 2 * eps), ("off--", sw_off, -2 * eps))}
            L, R, Rabs = [], [], []
            for kk in range(16):
                t = kk * Te / 16.0
                s = np.zeros(14)
                s[0], s[2], s[3], s[4], s[6] = u * t, z, math.radians(phi0), \
                    th, u
                rr = {key: np.array(q.deriv(s, t)[1]["lat_rhs"])
                      for key, q in P.items()}
                L1 = (((rr["on+"] - rr["on-"]) - (rr["off+"] - rr["off-"]))
                      / (2 * eps))
                L2 = (((rr["on++"] - rr["on--"])
                       - (rr["off++"] - rr["off--"])) / (4 * eps))
                L.append(2.0 * L1 - L2)
                # the reference, per unit amplitude: from the wave's
                # definition at amplitude eps, divided
                # the coefficients and the centroid depths at the calm
                # keel depths (amplitude 0): the linear part
                d = pcon.deriv(s, t)[1]["d"]
                vo, vob, voe, _ = _wave_one(P["on+"], s, t, d=d)
                vo, vob, voe = vo / eps, vob / eps, voe / eps
                pp = P["on+"]
                x, dx = pp.x, pp.dx
                if "lateral_phi0" in sw_on and "lateral_cross" not in sw_on:
                    (a22, a24, _), _, _ = pp._lat_coeffs(d, 0.0)
                    m22T = a22[0]
                    a22M = a22
                    extraK = 0.0
                elif "lateral_cross" in sw_on:
                    M = _Mt(pp, d, 0.0)
                    a22, a24 = M[:, 0, 0], M[:, 0, 2]
                    m22T = a22[0]
                    a22M = a22
                    # the 2D Munk roll moment of the sway wave velocity
                    # with the steady planing flow (W_r = -u tau at the
                    # captive attitude): g23 v_r W_r, g23 = dM~_23/dphi at a
                    # fixed CG (at zero heel dd/dphi = 0), v_r = -v_o
                    # (Richardson over the steps h, 2h: below the low chine
                    # the odd entries go like h (r / k1(h))^POW, a one-sided
                    # curvature that makes the central difference's error
                    # linear in h, POW h / tan(beta) relative)
                    hph = 1e-7

                    def cd23(h_):
                        return (_Mt(pp, d, h_)[:, 0, 1]
                                - _Mt(pp, d, -h_)[:, 0, 1]) / (2 * h_)
                    g23 = 2.0 * cd23(hph) - cd23(2.0 * hph)
                    from sim.planing_vessel import SP as SP_
                    tau_ = pp.kp + SP_ * (pp.Th_rest + th)
                    extraK = (g23 * vo * (u * tau_)).sum() * dx
                else:
                    (a22, a24, _), _, _ = pp._lat_coeffs(d, 0.0)
                    m22 = 0.5 * math.pi * pp.rho * np.maximum(d, 0.0) ** 2
                    m22T = m22[0]
                    a22M = m22
                    extraK = 0.0

                def form(rate):
                    Yf = (a22 * rate).sum() * dx + u * m22T * vo[0]
                    Kf = (a24 * rate).sum() * dx + extraK
                    if "lateral_phi0" in sw_on:
                        Kf = Kf + u * a24[0] * vo[0]
                    Nf = ((x * a22 * rate).sum() * dx + x[0] * u * m22T
                          * vo[0] + u * (a22M * vo).sum() * dx)
                    return np.array([Yf, Kf, Nf])
                R.append(form(vob))
                Rabs.append(form(voe))
            L, R, Rabs = np.array(L), np.array(R), np.array(Rabs)
            sc = np.maximum(np.abs(R).max(axis=0), 1e-12)
            e_ = float((np.abs(L - R).max(axis=0) / sc).max())
            dv = (np.abs(Rabs - R).max(axis=0) / sc)
            e_max = max(e_max, e_)
            dev_max = max(dev_max, float(dv.max()))
            out(f"    {label}, heading {mu:.0f} deg (encounter {we:.3f} "
                f"against {w:.3f} rad/s): amplitudes sway {sc[0]:.2f} N, "
                f"roll {sc[1]:.2f} N m, yaw {sc[2]:.2f} N m per m; |plant - "
                f"STF| / amplitude {e_:.1e}; the absolute-frequency form "
                f"(the first round's code) differs by {np.round(dv, 3)}")
        worst[label] = e_max
        absdev[label] = dev_max
    big = max(worst.values())
    check("stf", "STF: the plant's lateral diffraction (sway, roll, yaw; "
          "three paths, headings 45/90/135 deg, 16 instants of an encounter "
          "period) equals the strip formula with the encounter-frequency "
          "rate and the transom term within 1e-6 of the amplitude",
          big <= 1e-6, f"largest {big:.1e}; the absolute-frequency form "
          f"would differ by up to {max(absdev.values()):.2f}")
    flush("stf.txt")


def part_hb1():
    from sim.test_vessel import Monochromatic
    out("S5a review (hb1)  HB1: zero heel, beam and 135 deg seas, 0 and 25 kn "
        "captive attitude, linear-roll plants: hh's first-order lateral force"
        ", roll and yaw moment (amplitude +-1e-6 m) vs P1c's")
    eps = 1e-6
    worst = 0.0
    for kn in (0.0, 25.0):
        u = kn * KN
        for mu in (90.0, 135.0):
            for t in (0.3, 1.1):
                lin = {}
                for name in ("p1c", "hh", "hh_ab"):
                    r_ = []
                    for a in (eps, -eps):
                        # Monochromatic(omega, amplitude, theta0): a
                        # wave of length 12 m
                        q = _var_plant(name, sea=Monochromatic(
                            math.sqrt(G * 2.0 * math.pi / 12.0), a,
                            theta0=math.radians(mu)), captive_u=u, gz=False)
                        z, th = q.running_attitude(u)
                        s = np.zeros(14)
                        s[2], s[4], s[6] = z, th, u
                        r_.append(np.array(q.deriv(s, t)[1]["lat_rhs"]))
                    lin[name] = (r_[0] - r_[1]) / (2 * eps)
                P, H, Hab = lin["p1c"], lin["hh"], lin["hh_ab"]
                e_ = np.abs(H - P) / (1e-5 * np.abs(P) + 1e-3)
                worst = max(worst, float(e_.max()))
                out(f"    {kn:4.0f} kn, {mu:5.0f} deg, t {t}: P1c (sway, roll"
                    f", yaw per m) {np.round(P, 3)}; hh - P1c "
                    f"{np.round(H - P, 6)}; the first round's form (a_b on "
                    f"them) - P1c {np.round(Hab - P, 3)}")
    check("hb1", "HB1: at zero heel hh's first-order lateral, roll and yaw "
          "wave forces equal P1c's within 1e-5 of P1c's (+ 1e-3 per m)",
          worst <= 1.0, f"largest ratio to the bound {worst:.2e}")
    flush("hb1.txt")


def part_e0():
    out("S5a review (e0)  E0: the 2D table's even entries smooth at zero "
        "heel (phi^2 in the first heel cell)")
    p = plant(AS, sw=cand("hx"), captive_u=25 * KN)
    z, th = p.running_attitude(25 * KN)
    s = np.zeros(14)
    s[2], s[4], s[6] = z, th, 25 * KN
    d = p.deriv(s, 0.0)[1]["d"]
    wet = d > 0.0
    tab = p._tab()
    cols = np.arange(6)
    C0, Cd0, Cp0 = tab.lateral(d[wet], p.yc[wet], 0.0)
    R0, Rd0, Rp0 = tab._eval_at(d[wet], p.yc[wet], 0.0, cols)
    same = np.array_equal(C0, R0) and np.array_equal(Cd0, Rd0)
    ev = np.array([0, 2, 3, 5])
    zero = bool(np.all(Cp0[:, ev] == 0.0))
    old = np.abs(tab._eval_at(d[wet], p.yc[wet], 1e-6, cols)[2][:, ev]
                 .sum(0))
    rat = []
    for ph in (1e-6, -1e-6):
        new = np.abs(tab.lateral(d[wet], p.yc[wet], ph)[2][:, ev].sum(0))
        rat.append(new / np.maximum(old, 1e-300))
    rat = np.max(np.array(rat), axis=0)
    p1 = float(tab.phi[1])
    Cm = tab.lateral(d[wet], p.yc[wet], p1 * (1 - 1e-12))[0]
    Cq = tab.lateral(d[wet], p.yc[wet], p1)[0]
    cont = float(np.abs(Cm - Cq).max() / np.abs(Cq).max())
    out(f"    25 kn wet stations ({int(wet.sum())}): at phi = 0 values and "
        f"d-derivatives bit for bit the table's own {same}; even entries' "
        f"heel derivative exactly 0: {zero}")
    out(f"    summed |dC/dphi| at +-1e-6 rad over the bilinear one-sided slope "
        f"(a22, a33, a24, a44): {np.round(rat, 6)}; the previous slopes "
        f"(per unit density) {np.round(old, 4)}")
    out(f"    continuity at phi_1 = {math.degrees(p1):.3f} deg: {cont:.1e}")
    check("e0", "E0: at zero heel the values bit for bit unchanged and the "
          "even entries' heel derivative exactly 0; at +-1e-6 rad <= 1e-3 of "
          "the previous one-sided slope; continuous at phi_1 (1e-12)",
          same and zero and bool(np.all(rat <= 1e-3)) and cont <= 1e-12,
          f"ratios {np.round(rat, 6)}, continuity {cont:.1e}")
    flush("e0.txt")


class _ScaledTable:
    """The plant's 2D table with the sway added mass a22 (column 0, and its
    derivatives) scaled -- G-A4's diagnostic of the low-speed mode."""

    def __init__(self, tab, f):
        self._t, self._f = tab, f

    def lateral(self, d, yc, phi):
        C, Cd, Cp = self._t.lateral(d, yc, phi)
        C, Cd, Cp = C.copy(), Cd.copy(), Cp.copy()
        for M in (C, Cd, Cp):
            M[:, 0] *= self._f
        return C, Cd, Cp

    def __getattr__(self, k):
        return getattr(self._t, k)


def part_ga4():
    out("S5a review (ga4)  G-A4: hx's lateral eigenvalues at 1, 3, 6 kn with "
        "the sway added mass a22 scaled to the strip rigid-lid sum (x1.55, "
        "524/339) and the 3D zero-frequency one (x1.38, 467/339); reading "
        "registered: if both scalings remove the unstable mode (max Re <= "
        "max(P1c, 0) + 1e-3), A4's attribution is supported")
    rows = []
    ok = True
    for kn in (1.0, 3.0, 6.0):
        u = kn * KN
        r_ = {}
        for name, f in (("p1c", None), ("hx", 1.0), ("hx x1.55", 1.55),
                        ("hx x1.38", 1.38)):
            p = plant(AS, sw=cand("p1c" if f is None else "hx"),
                      captive_u=u)
            if f is not None and f != 1.0:
                p._sec2d = _ScaledTable(p._tab(), f)
            z, th = p.running_attitude(u)
            s = np.zeros(14)
            s[2], s[4], s[6] = z, th, u
            J, _ = lateral_jacobian(p, s)
            r_[name] = float(np.linalg.eigvals(J).real.max())
        lim = max(r_["p1c"], 0.0) + 1e-3
        ok = ok and r_["hx x1.55"] <= lim and r_["hx x1.38"] <= lim
        rows.append(dict(kn=kn, **r_))
        out(f"    {kn:3.0f} kn: max Re (1/s) P1c {r_['p1c']:+.4f}, hx "
            f"{r_['hx']:+.4f}, a22 x1.55 {r_['hx x1.55']:+.4f}, a22 x1.38 "
            f"{r_['hx x1.38']:+.4f}")
    out(f"    reading: A4's attribution {'SUPPORTED' if ok else 'NOT supported (a hypothesis)'}")
    dump("ga4.json", dict(rows=rows, supported=ok))
    flush("ga4.txt")


PARTS = dict(twin=part_twin, goldens=part_goldens, l1g=part_l1g, h1=part_h1, h2=part_h2, h3=part_h3, h4=part_h4, h5=part_h5,
             sym=part_sym, h8=part_h8, h9=part_h9, h10=part_h10, e1=part_e1,
             l1=part_l1, l1p=part_l1p, l4=part_l4, l8=part_l8,
             l1pp=part_l1pp, stf=part_stf, hb1=part_hb1, e0=part_e0,
             ga4=part_ga4)


def _sub(arg):
    part, cand_, out_dir = arg
    import subprocess
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"run_{part}_{cand_}.log"), "w") as log:
        rc = subprocess.run([sys.executable, "-m",
                             "studies.anyspeed_s5a_check", "--parts", part,
                             "--cand", cand_, "--out", out_dir], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT).returncode
    return part, cand_, rc, time.time() - t0


def main():
    global OUT, STAMP, CAND
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="h1,h3")
    ap.add_argument("--cand", default="hh", choices=("hh", "hx"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", default=None,
                    help="part[:cand],... each in its own process")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    CAND = a.cand
    if a.out:
        OUT = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    if a.jobs:
        from studies.parallel_runner import run_parallel
        jobs = []
        for j in a.jobs.split(","):
            part, _, c_ = j.partition(":")
            jobs.append((part, c_ or a.cand, OUT))
        t0 = time.time()
        for r in run_parallel(_sub, jobs, gb_per_job=0.4, reserve_gb=3.0):
            print(f"  {r[0]}:{r[1]} rc {r[2]} {r[3]:.0f} s", flush=True)
        print(f"jobs done in {time.time() - t0:.0f} s")
        return
    STAMP = stamp_all()
    t0 = time.time()
    for part in a.parts.split(","):
        t1 = time.time()
        PARTS[part]()
        print(f"  ({part}: {time.time() - t1:.0f} s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    tag = "_".join(a.parts.split(",")) + f"_{CAND}"
    with open(os.path.join(OUT, f"checks_{tag}.json"), "w") as f:
        json.dump(dict(checks=CHECKS, _stamp=STAMP,
                       seconds=time.time() - t0), f, indent=1)
    n_bad = sum(not c["ok"] for c in CHECKS)
    print(f"{len(CHECKS)} checks, {n_bad} failed, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
