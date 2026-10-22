#!/usr/bin/env python3
"""
Verification of design step S1 of the any-speed plant (switch dDt_full in
sim/planing_anyspeed.py; learn/meta/anyspeed_design_en.md section 2.5): the
complete material derivative of the sectional added mass,

    Dm_a/Dt = (dm_a/dd) Dd/Dt - u (dm_a/dx)|_d,

over the forebody taper with the chine-wetting point located inside each
strip (d linear over the strip, chine analytic), the momentum flux
V Dm_a/Dt counted on entry only (V > 0 and Dm_a/Dt > 0).

Plants compared (scarab195, calm water unless stated, captive surge = speed
held by a force at the nozzle, free in heave and pitch, as the plant's own
captive mode):
  v0       sim/planing_vessel_gz.PlaningVesselGZ on hydro.hulls.get("scarab195")
  as1      sim/planing_anyspeed.PlaningVesselAS on "scarab195@as1" (its
           preset: dDt_full on; the v0 a_b blend, no transom face -- S2)
  ungated  as1 with the as-built gate of the first S1 (flux counted while
           Dm_a/Dt > 0 whatever the sign of V: a suction where a chine-wetted
           section rises out) -- for comparison only, parts f
Station-count studies use deep copies of scarab195 named scarab195_n<N>
(physics.variant refuses a modified hull under the registered name).
Equilibria are Newton roots of the captive equations w' = q' = 0 of the
plant's own deriv (fsolve, then Newton polishing on a central-difference
Jacobian), residual <= 1e-10 unless stated. Linear modes: eigenvalues of the
4x4 Jacobian d(z', th', w', q')/d(z, th, w, q), central differences.

Parts (each optional; outputs in studies/_cache/anyspeed/s1/):
  a  linear pitch damping D55 at 1-7 kn and at three partially chine-wetted
     speeds (8, 8.5, 9 kn), free-running (surge free, jet force fixed: no
     captive towing-force share), strip + cross-flow drag, against an
     independent continuum linearisation of the same 2D+t model on a fine
     grid (40 000 points; exact geometry, no stations) and, where the whole
     taper is chine-wetted, the closed form U x_T^2 m_aT + int rho C_dc 2c
     |V0| x^2 dx
  b  calm captive runs every 0.5 kn over 1-10 kn from 1 mm and 5 cm heave
     offsets, 120 s: decay or limit cycle, v0 and as1; a time-step study
     where as1 has no root or does not decay
  c  time-domain calm speed sweeps 0-35 kn up and down (captive surge): ramps
     at 0.1 and 0.05 kn/s (the up-down difference of a ramp is lag if it
     halves with the rate), a stepped sweep (30 s per speed, last 10 s), and
     a free-surge sweep (the jet's thrust stepped up and down, the speed
     free, 60 s per level)
  g  root census of the captive equations at 1-9.5 kn every 0.5 kn, 10, 12,
     16, 20, 25 and 26-40 kn every 1 kn (grid of 10 x 10 starts over
     z -0.6..0.3 m, trim -15..12 deg, deflated Newton until no new root;
     every root classified by its eigenvalues), plus a winding-number scan
     of the residual on a 60 x 72 grid of the same box, which finds zeros
     at a discontinuity too; the stable state at every speed
  f  robustness: basin test (captive, +-20 cm / +-5 deg from the planing
     root, 16-35 kn, 40 s), as1 vs ungated vs v0; open-loop runs in the
     task sea of mission s0 leg 0 turned into a following sea at Hs x1 and
     x1.5 (12-30 kn, 240 s, constant thrust), as1 vs v0
  w  regular head waves where the bow rises out of the water (8-14 kn,
     captive), as1 vs ungated vs v0
  d  planing range: Newton equilibria at 14-40 kn with 48 / 96 / 192
     stations, as1 vs v0; running_attitude, calm resistance, top speed;
     Fridsma P6 (studies/exp_planing.py) heave and pitch; the validation
     review's non-regression bounds (|dtrim| <= 0.05 deg, |dR| <= 1 %, top
     speed +-0.5 %)
  h  high speed (26-40 kn) station count and time step of the captive runs
     from rest; v0's 26-27 kn calm limit cycle beside its fixed point, by
     time step and station count, from rest and from the 25 kn state
  e  regression: v0 bit-identical and as1 all-off exact (runs the harnesses
     studies/anyspeed_regress_as.py and studies/anyspeed_regress.py)

    .venv/Scripts/python.exe -m studies.anyspeed_s1_check --a --b --c --g
    .venv/Scripts/python.exe -m studies.anyspeed_s1_check --f --w --d --h --e
"""
import argparse
import copy
import json
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
from scipy.optimize import fsolve       # noqa: E402

OUT = os.path.join(HERE, "_cache", "anyspeed", "s1")
KN = 0.514444
G = 9.81
DT = 0.02
# top speed of part c's sweeps (S1 ran 35 kn; studies/anyspeed_p1a_check.py
# sets 40 for ACCEPTANCE R2's 0-40 kn)
TOP_KN = 35.0
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


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


# ------------------------------------------------------------------ plants
def calm():
    from sim.test_vessel import Monochromatic
    return Monochromatic(1.0, 0.0)


_UNGATED = []


def ungated_cls():
    """PlaningVesselAS with the as-built gate of the first S1: the flux
    V Dm_a/Dt counted while Dm_a/Dt > 0 whatever the sign of V (the same
    sub-strip discretisation). Where a chine-wetted taper section rises out
    (V < 0) that is a suction. For comparison only."""
    if _UNGATED:
        return _UNGATED[0]
    from sim.planing_anyspeed import PlaningVesselAS

    class PlaningVesselASUngated(PlaningVesselAS):
        def _flux_dDt(self, d, V, u, mom, slope):
            flux = mom.copy()
            act, ycap, iunc = self._substrip(d, slope)
            self._taper_cap = act
            if act.size:
                Vj = V[act]
                km = self.k_a * 0.5 * np.pi * self.rho
                kc = 0.5 * np.pi / self.tanb
                Dma = (Vj * (2.0 * km * kc * kc) * iunc
                       + max(u, 0.0) * km * ycap) / self.dx
                flux[act] = np.where(Dma > 0.0, Vj * Dma, 0.0)
            return flux
    _UNGATED.append(PlaningVesselASUngated)
    return PlaningVesselASUngated


def hull_n(n):
    """scarab195 with n stations, under a name of its own."""
    from hydro import hulls
    h = copy.deepcopy(hulls.get("scarab195"))
    if n != h.n_stations:
        h.n_stations = n
        h.name = f"scarab195_n{n}"
    return h


def build(ver, captive_u=0.0, n=48, sea=None, dt=DT):
    from hydro import hulls
    from sim import physics
    sea = calm() if sea is None else sea
    h = hull_n(n)
    if ver == "v0":
        from sim.planing_vessel_gz import PlaningVesselGZ
        return PlaningVesselGZ(h, sea, dt=dt, captive_u=captive_u)
    ha = (hulls.get("scarab195@as1") if n == 48
          else physics.variant(h, "as1", name=h.name))
    if ver == "ungated":
        return ungated_cls()(ha, sea, dt=dt, captive_u=captive_u)
    from sim.planing_anyspeed import PlaningVesselAS
    sw = {"as1": None, "off": "v0"}[ver]
    return PlaningVesselAS(ha, sea, dt=dt, captive_u=captive_u,
                           test_switches=sw)


def state(u, z, th, w=0.0, q=0.0):
    s = np.zeros(14)
    s[2], s[4], s[6], s[8], s[10] = z, th, u, w, q
    return s


def acc(p, u, z, th, w=0.0, q=0.0):
    p.captive_u = u
    ds, aux = p.deriv(state(u, z, th, w, q), 0.0)
    return ds[8], ds[10], aux


def F(p, u, v):
    a, b, _ = acc(p, u, v[0], v[1])
    return np.array([a, b])


def trim_deg(p, th):
    from sim.planing_vessel import SP
    return float(np.degrees(SP * (p.Th_rest + th)))


def th_of(p, trim):
    from sim.planing_vessel import SP
    return np.radians(trim) / SP - p.Th_rest


def jac2(p, u, v, eps=(1e-7, 1e-8)):
    J = np.zeros((2, 2))
    for i in range(2):
        h = np.zeros(2)
        h[i] = eps[i]
        J[:, i] = (F(p, u, v + h) - F(p, u, v - h)) / (2 * eps[i])
    return J


def polish(p, u, v, tol=1e-12, it=30):
    """Newton on the captive equations from v: (root, max |residual|)."""
    v = np.asarray(v, float)
    r = np.abs(F(p, u, v)).max()
    for _ in range(it):
        if r <= tol:
            break
        try:
            step = np.linalg.solve(jac2(p, u, v), -F(p, u, v))
        except np.linalg.LinAlgError:
            break
        lam = 1.0
        while lam > 1e-4:
            vn = v + lam * step
            rn = np.abs(F(p, u, vn)).max()
            if rn < r:
                v, r = vn, rn
                break
            lam *= 0.5
        else:
            break
    return v, float(r)


def newton(p, u, guess):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol, _, ier, _ = fsolve(lambda v: F(p, u, v), guess,
                                full_output=True, xtol=1e-13)
    return polish(p, u, sol)


def mass2(p, u, z, th):
    """The heave-pitch mass matrix the plant puts on the left (m + A33 ...),
    at (z, th)."""
    from sim.planing_vessel import SP
    d = -(p.Z_rest + z + p.zk + SP * p.x * (p.Th_rest + th))
    ma = p._section(d)[2] * p._ctr(u)
    dx = p.dx
    return np.array([[p.m + ma.sum() * dx, (ma * SP * p.x).sum() * dx],
                     [(ma * SP * p.x).sum() * dx,
                      p.Iy + (ma * p.x * p.x).sum() * dx]])


def jac4(p, u, z, th, eps=(1e-6, 1e-7, 1e-6, 1e-7)):
    def f4(v):
        a, b, _ = acc(p, u, v[0], v[1], v[2], v[3])
        return np.array([v[2], v[3], a, b])
    x0 = np.array([z, th, 0.0, 0.0])
    J = np.zeros((4, 4))
    for i in range(4):
        h = np.zeros(4)
        h[i] = eps[i]
        J[:, i] = (f4(x0 + h) - f4(x0 - h)) / (2 * eps[i])
    return J


def modes(J):
    lam = np.linalg.eigvals(J)
    pairs = sorted((abs(l), -l.real / abs(l), 2 * np.pi / l.imag)
                   for l in lam if l.imag > 1e-9)
    return pairs, float(lam.real.max()), lam


def mstr(pairs):
    return "  ".join(f"{T:5.2f}/{z:+.3f}" for _, z, T in pairs[:2]) or "real"


def n_taper_capped(p, u, z, th):
    """(taper strips with a chine-wetted part, taper strips) at (z, th)."""
    _, _, aux = acc(p, u, z, th)
    if not getattr(p, "sw", None) or not p.sw.dDt_full:
        return None, None
    return int(p._taper_cap.size), int(p._taper().idx.size)


# ------------------------------------------------------------------ part a
def free_D(p, u, z, th, eps=(1e-6, 1e-7)):
    """D = -M d(w',q')/d(w,q) at (z, th) with the surge free: the jet's
    force does not depend on w or q, so the captive towing force's moment
    share (a test-rig artefact) is absent; w' and q' do not involve u'."""
    cu = p.captive_u
    p.captive_u = None
    try:
        def f2(w, q):
            ds, _ = p.deriv(state(u, z, th, w, q), 0.0)
            return np.array([ds[8], ds[10]])
        J = np.zeros((2, 2))
        J[:, 0] = (f2(eps[0], 0.0) - f2(-eps[0], 0.0)) / (2 * eps[0])
        J[:, 1] = (f2(0.0, eps[1]) - f2(0.0, -eps[1])) / (2 * eps[1])
    finally:
        p.captive_u = cu
    return -mass2(p, u, z, th) @ J


def continuum_D55(p, u, z, th, N=40000):
    """The linear pitch damping of the same 2D+t strip model, evaluated as a
    continuum on a fine grid (exact chine and keel, no stations, no strips;
    calm water, steady V0 = u tau): from deriv term by term,
      T1 = -2u int x m_a dx                 (the 2 SP u q m_a term)
      T2 = -u int_capped x^2 dm_a/dx|_d dx  (the S1 term, entering)
      T3 = 2 int_uncapped x^2 dm_a/dd V0 dx (v0's entry flux)
      T4 = int rho C_dc 2c |V0| x^2 dx      (cross-flow drag)
    and, where the whole taper is chine-wetted, the closed form by parts,
    T1 + T2 = u x_T^2 m_aT - u x_bow^2 m_a(bow). Returns a dict.

    The geometry between stations comes from hydro/hull_geometry (the
    plant's own source: chine, its slope, keel and keel slope at any x), not
    from the hull fields taper / stem_rise, which describe v0's family
    member only -- so this reference is right for planing_b's transom run
    and cubic keel and for an offsets hull too. 'taper' below is where the
    chine varies along x (the bow taper and a transom run)."""
    from hydro.hull_geometry import stations_of
    from sim.planing_vessel import SP
    h = p.hull
    g = stations_of(h)
    dxg = h.L / N
    xi = (np.arange(N) + 0.5) * dxg
    x = xi - h.lcg
    yc = g.chine(xi)
    ycp = g.dchine(xi)
    rise, kp, _ = g.keel(xi)
    zk = -h.vcg + rise
    Th = p.Th_rest + th
    d = -(p.Z_rest + z + zk + SP * x * Th)
    tau = kp + SP * Th
    V0 = u * tau
    kc = 0.5 * np.pi / p.tanb
    cw = kc * np.maximum(d, 0.0)
    c = np.minimum(cw, yc)
    wet = d > 0.0
    cap = (cw >= yc) & wet
    ma = p.k_a * 0.5 * np.pi * p.rho * c ** 2
    dma = np.where(~cap & wet, p.k_a * np.pi * p.rho * c * kc, 0.0)
    dmadx = np.where(cap, p.k_a * np.pi * p.rho * yc * ycp, 0.0)
    ent = V0 > 0.0
    T1 = -2.0 * u * (x * ma).sum() * dxg
    # the plant counts the flux only while Dm_a/Dt > 0: at a capped
    # section in steady state Dm_a/Dt = -u dm_a/dx, so only where the chine
    # narrows forward (the bow taper; a transom run, which widens, is
    # gated off and does not enter the linearisation). scarab195 has no
    # widening piece: unchanged there
    T2 = -u * (x ** 2 * np.minimum(dmadx, 0.0) * ent).sum() * dxg
    T3 = 2.0 * (x ** 2 * dma * V0 * ent).sum() * dxg
    T4 = (p.rho * p.c_dc * 2.0 * c * np.abs(V0) * x ** 2).sum() * dxg
    xT = -h.lcg
    dT = -(p.Z_rest + z + zk[0] + SP * xT * Th)
    maT = p.k_a * 0.5 * np.pi * p.rho * min(kc * max(dT, 0.0), yc[0]) ** 2
    taper = ycp != 0.0
    all_cap = (bool(cap[taper & wet].all()) and bool(wet[taper].all())
               and not bool((cap & (ycp > 0.0)).any()))
    return dict(T1=T1, T2=T2, T3=T3, T4=T4, strip=T1 + T2 + T3,
                total=T1 + T2 + T3 + T4, closed=u * xT ** 2 * maT + T4,
                transom=u * xT ** 2 * maT, all_capped=all_cap,
                frac_taper_capped=float(cap[taper].mean()))


def part_a():
    out("S1 (a)  LINEAR PITCH DAMPING D55, FREE-RUNNING, ON THE as1 "
        "EQUILIBRIUM")
    out("  D = -M d(w',q')/d(w,q), central differences on the plant's deriv "
        "with the surge free (the jet's force does not depend on w, q, so "
        "the captive towing force's share is absent); M is independent of "
        "w, q. D55 in N m s.")
    out("  FD = the plant (48 stations, sub-strip chine wetting); strip = FD "
        "with C_dc = 0 (momentum terms only); cont = the same model's "
        "continuum linearisation on a 40 000-point grid (T1 2u q m_a term, "
        "T2 the S1 term, T3 v0's entry flux, T4 cross-flow drag; see "
        "continuum_D55) -- independent of stations and strips;")
    out("  closed = U x_T^2 m_aT + int rho C_dc 2c |V0| x^2 dx (valid where "
        "the whole taper is chine-wetted). Partially chine-wetted speeds: "
        "8, 8.5, 9 kn.")
    out("   kn   z_eq      trim   resid  taper capped  modes T_n/zeta (as1)"
        "         maxRe     FD   strip     cf |   cont  T1+T2   T3     T4  "
        "closed | FD/cont  strip/(T1+T2+T3)  FD/closed | v0: trim  maxRe    "
        "FD")
    rows = []
    g1 = g0 = np.zeros(2)
    a1, v0 = build("as1"), build("v0")
    for kn in (1, 2, 3, 4, 5, 6, 7, 8, 8.5, 9):
        u = kn * KN
        r = {}
        for name, p, g in (("as1", a1, g1), ("v0", v0, g0)):
            sol, res = newton(p, u, g)
            z, th = sol
            J = jac4(p, u, z, th)
            pairs, mre, _ = modes(J)
            D = free_D(p, u, z, th)
            cdc = p.c_dc
            p.c_dc = 0.0
            Dn = free_D(p, u, z, th)
            p.c_dc = cdc
            r[name] = dict(z=z, th=th, trim=trim_deg(p, th), resid=res,
                           modes=pairs, maxre=mre, D=D, D_strip=Dn)
            if name == "as1":
                g1 = sol
            else:
                g0 = sol
        R = r["as1"]
        cont = continuum_D55(a1, u, R["z"], R["th"])
        nc, nt = n_taper_capped(a1, u, R["z"], R["th"])
        D55, D55s = R["D"][1, 1], R["D_strip"][1, 1]
        R.update(cont=cont, D55=D55, D55_strip=D55s, D55_cf=D55 - D55s,
                 taper_capped=(nc, nt))
        q0 = r["v0"]
        clo = cont["closed"] if cont["all_capped"] else np.nan
        out(f"  {kn:4.1f} {R['z']:+.5f} {R['trim']:7.3f}  {R['resid']:.0e}"
            f"    {nc:2d}/{nt:2d}     {mstr(R['modes']):28s} "
            f"{R['maxre']:+.3f}  {D55:7.0f} {D55s:7.0f} {D55 - D55s:6.0f} |"
            f" {cont['total']:7.0f} {cont['T1'] + cont['T2']:6.0f} "
            f"{cont['T3']:5.0f} {cont['T4']:6.0f} {clo:7.0f} |  "
            f"{D55 / cont['total']:.4f}    {D55s / cont['strip']:.4f}       "
            f"      {D55 / clo:.4f}  | {q0['trim']:6.3f} {q0['maxre']:+.3f} "
            f"{q0['D'][1, 1]:7.0f}")
        rows.append(dict(kn=kn, **r))
    out()
    w_all = max(abs(x["as1"]["D55"] / x["as1"]["cont"]["total"] - 1)
                for x in rows)
    w_cl = max(abs(x["as1"]["D55"] / x["as1"]["cont"]["closed"] - 1)
               for x in rows if x["as1"]["cont"]["all_capped"])
    out(f"  largest |FD/cont - 1| over all speeds: {100 * w_all:.2f} % "
        f"(discretisation: 48 stations, strip-end chine); largest |FD/closed"
        f" - 1| where the whole taper is chine-wetted: {100 * w_cl:.2f} %")
    out(f"  all as1 modes damped: {all(x['as1']['maxre'] < 0 for x in rows)}"
        f"; v0 max Re(lambda) > 0 at kn "
        f"{[x['kn'] for x in rows if x['v0']['maxre'] > 0]}")
    out("  PROVISIONAL (review): the 1-9 kn stabilisation is the transom flux "
        "u m_aT V_T with m_aT the Wagner-capped 2D value; S2 (transom "
        "wetness eps) and S4 (3D factor J(x)) rescale it -- this part, (b) "
        "and the basin test of (f) are standing checks to re-run then")
    flush("a_d55.txt")
    dump("a_d55.json", rows)


# ------------------------------------------------------------------ part b
def decay_run(p, u, z0, th0, dz, T=120.0):
    s = state(u, z0 + dz, th0)
    p.captive_u = u
    p.initial_state(0.0)        # resets the measurements only
    n = int(round(T / p.dt))
    zz, tt = np.empty(n + 1), np.empty(n + 1)
    zz[0], tt[0] = s[2] - z0, s[4] - th0
    for i in range(n):
        s = p.step(s, i * p.dt, 0.0, 0.0)
        s[6] = u
        zz[i + 1], tt[i + 1] = s[2] - z0, s[4] - th0
    return zz, tt


def settle(p, u, T=150.0, last=10.0, start=None):
    """Captive calm run at speed u from the rest attitude (or from
    start=(z, th)): (z, th) mean over the last `last` s, and their
    peak-to-peak there."""
    s = state(u, 0.0, 0.0) if start is None else state(u, *start)
    p.captive_u = u
    p.initial_state(0.0)
    n, m = int(round(T / p.dt)), int(round(last / p.dt))
    zs, ts = np.empty(m), np.empty(m)
    for i in range(n):
        s = p.step(s, i * p.dt, 0.0, 0.0)
        s[6] = u
        if i >= n - m:
            zs[i - n + m], ts[i - n + m] = s[2], s[4]
    return zs.mean(), ts.mean(), float(np.ptp(zs)), float(np.ptp(ts))


def part_b():
    from sim.planing_vessel import SP
    out("S1 (b)  CALM CAPTIVE RUNS FROM HEAVE OFFSETS, 120 s at the plant "
        f"step {DT} s (plant.step, speed held), every 0.5 kn over 1-10 kn")
    out("  reference: the Newton root (z_eq, th_eq) when its residual is <= "
        "1e-10, else the mean of the last 10 s of a 150 s captive run from "
        "rest ('td')")
    out("  start: reference + dz; columns: max |z - z_ref| over 0-10 s, "
        "50-60 s, 110-120 s; z and trim peak-to-peak over the last 10 s")
    out("  verdict: 'decays' if the last 10-s envelope is below 1 % of the "
        "first AND still falling (below 0.9 x the one before, or at round-"
        "off, < 1e-12 m); 'GROWS' if it ends above the first; else 'no "
        "decay (bounded)'")
    out("   plant   kn   ref  resid    dz m    z_ref     trim_ref  env 0-10 "
        "  env 50-60   env 110-120  z p-p last  trim p-p last deg  verdict")
    rows = []
    kns = [float(k) for k in np.arange(1.0, 10.01, 0.5)]
    for name in ("v0", "as1"):
        p = build(name)
        g = np.zeros(2)
        for kn in kns:
            u = kn * KN
            sol, res = newton(p, u, g)
            if res <= 1e-10:
                g, ref, how = sol, sol, "root"
            else:
                z, th, _, _ = settle(p, u)
                ref, how = np.array([z, th]), "td"
            for dz in (0.001, 0.05):
                zz, tt = decay_run(p, u, ref[0], ref[1], dz)
                k = int(round(10 / p.dt))
                env = [np.abs(zz[j * k:(j + 1) * k]).max()
                       for j in range(len(zz) // k)]
                last = zz[-k:]
                pp_z = float(last.max() - last.min())
                pp_t = float(np.degrees(tt[-k:].max() - tt[-k:].min()))
                falling = env[-1] < 0.9 * env[-2] or env[-1] < 1e-12
                if env[-1] < 0.01 * env[0] and falling:
                    verdict = "decays"
                elif env[-1] > env[0]:
                    verdict = "GROWS"
                elif pp_z < 1e-9 and pp_t < 1e-7:
                    verdict = (f"settles at another steady state "
                               f"{zz[-1]:+.2e} m away")
                else:
                    verdict = "no decay (bounded)"
                rows.append(dict(plant=name, kn=kn, dz=dz, z_ref=ref[0],
                                 th_ref=ref[1], trim_ref=trim_deg(p, ref[1]),
                                 ref=how, resid=res, env=env, pp_z=pp_z,
                                 pp_trim_deg=pp_t, verdict=verdict))
                out(f"   {name:5s} {kn:4.1f}  {how:4s} {res:.0e}  {dz:.3f}  "
                    f"{ref[0]:+.5f}  {trim_deg(p, ref[1]):7.3f}   "
                    f"{env[0]:.3e}   {env[5]:.3e}   {env[-1]:.3e}   "
                    f"{pp_z:.3e}   {pp_t:.3e}          {verdict}")
    out()
    for name in ("v0", "as1"):
        rr = [r for r in rows if r["plant"] == name]
        out(f"  {name}: {sum(r['verdict'] == 'decays' for r in rr)} of "
            f"{len(rr)} runs decay; roots at "
            f"{sum(r['ref'] == 'root' for r in rr) // 2} of {len(kns)} "
            f"speeds; not decaying: "
            f"{[(r['kn'], r['dz'], r['verdict']) for r in rr if r['verdict'] != 'decays']}")
    odd = sorted({r["kn"] for r in rows if r["plant"] == "as1"
                  and (r["ref"] == "td" or r["verdict"] != "decays")})
    study = []
    if odd:
        out()
        out("  TIME-STEP STUDY (as1, captive run from rest, 100 s, last 10 s)"
            f" at {odd} kn (no root or not decaying)")
        out("    kn    dt s     z mean      z p-p      trim mean   trim p-p "
            "deg   residual")
        from hydro import hulls
        from sim.planing_anyspeed import PlaningVesselAS
        h1 = hulls.get("scarab195@as1")
        for kn in odd:
            u = kn * KN
            for dt in (0.02, 0.01, 0.005, 0.0025):
                p = PlaningVesselAS(h1, calm(), dt=dt, captive_u=u)
                z, th, ppz, ppt = settle(p, u, T=100.0)
                res = float(np.abs(F(p, u, np.array([z, th]))).max())
                study.append(dict(kn=kn, dt=dt, z=z, trim=trim_deg(p, th),
                                  pp_z=ppz, pp_trim_deg=float(np.degrees(ppt)),
                                  resid=res))
                out(f"   {kn:4.1f}  {dt:.4f}  {z:+.6f}  {ppz:.2e}   "
                    f"{trim_deg(p, th):9.5f}   {np.degrees(ppt):.2e}    "
                    f"{res:.1e}")
    else:
        out("  as1 has a root and every run decays at every speed: no "
            "time-step study needed")
    flush("b_decay.txt")
    dump("b_decay.json", dict(runs=rows, dt_study=study))


# ------------------------------------------------------------------ part c
def ramp_sweep(p, rate_kn_s=0.1, top_kn=None, hold=20.0, every=10):
    """Captive calm run with the speed prescribed: 0 -> top -> 0 at rate,
    holding `hold` s at each end. Rows (t, u, z, trim, R, dir)."""
    dt = p.dt
    top_kn = TOP_KN if top_kn is None else top_kn
    t_up = top_kn / rate_kn_s
    T = hold + t_up + hold + t_up + hold
    n = int(round(T / dt))
    s = np.zeros(14)
    p.initial_state(0.0)
    rec = []

    def speed(t):
        if t < hold:
            return 0.0, 0
        t -= hold
        if t < t_up:
            return t * rate_kn_s, 1
        t -= t_up
        if t < hold:
            return top_kn, 0
        t -= hold
        if t < t_up:
            return top_kn - t * rate_kn_s, -1
        return 0.0, 0
    for i in range(n):
        kn, dr = speed(i * dt)
        u = kn * KN
        p.captive_u = u
        s[6] = u
        if i % every == 0:
            _, aux = p.deriv(s, i * dt)
            rec.append((i * dt, kn, s[2], trim_deg(p, s[4]), aux["fx"], dr))
        s = p.step(s, i * dt, 0.0, 0.0)
        s[6] = u
    return np.asarray(rec)


def step_sweep(p, kns, hold=30.0, last=10.0, start=None):
    """Captive calm speed steps through kns, `hold` s each: per speed the
    mean z, trim, R over the last `last` s and their peak-to-peak; also
    the trim and z peak-to-peak over the window of the same length before
    it (trim_pp_prev, z_pp_prev: P1's registered sweep criterion compares
    the two -- a ratio below 0.5 is a decaying transient)."""
    s = np.zeros(14) if start is None else start.copy()
    p.initial_state(0.0)
    rows = []
    dt = p.dt
    n, m = int(round(hold / dt)), int(round(last / dt))
    m0 = max(n - 2 * m, 0)
    for kn in kns:
        u = kn * KN
        p.captive_u = u
        zs, ts, rs, zp, tp = [], [], [], [], []
        for i in range(n):
            s[6] = u
            if i >= n - m:
                zs.append(s[2])
                ts.append(trim_deg(p, s[4]))
                if (i - (n - m)) % 10 == 0:
                    rs.append(p.deriv(s, 0.0)[1]["fx"])
            elif i >= m0:
                zp.append(s[2])
                tp.append(trim_deg(p, s[4]))
            s = p.step(s, i * dt, 0.0, 0.0)
            s[6] = u
        zs, ts = np.asarray(zs), np.asarray(ts)
        rows.append(dict(kn=kn, z=float(zs.mean()), z_pp=float(np.ptp(zs)),
                         trim=float(ts.mean()), trim_pp=float(np.ptp(ts)),
                         R=float(np.mean(rs)),
                         z_pp_prev=float(np.ptp(zp)) if zp else float("nan"),
                         trim_pp_prev=float(np.ptp(tp)) if tp
                         else float("nan")))
    return rows, s


def thrust_sweep(p, levels, hold=60.0, last=10.0):
    """Free surge: the jet's thrust command stepped through `levels`
    (fractions of t_max), the speed free; per level the mean speed, trim,
    heave and their peak-to-peak over the last `last` s."""
    p.captive_u = None
    s = p.initial_state(0.0)
    rows = []
    dt = p.dt
    n, m = int(round(hold / dt)), int(round(last / dt))
    t = 0.0
    for lv in levels:
        cmd = lv * p.prop.t_max
        us, zs, ts = [], [], []
        for i in range(n):
            s = p.step(s, t, cmd, 0.0, dt)
            t += dt
            if i >= n - m:
                us.append(s[6])
                zs.append(s[2])
                ts.append(trim_deg(p, s[4]))
        us, zs, ts = (np.asarray(v) for v in (us, zs, ts))
        rows.append(dict(level=lv, kn=float(us.mean() / KN),
                         kn_pp=float(np.ptp(us) / KN), z=float(zs.mean()),
                         z_pp=float(np.ptp(zs)), trim=float(ts.mean()),
                         trim_pp=float(np.ptp(ts))))
    return rows


def part_c():
    top = int(TOP_KN)
    out(f"S1 (c)  TIME-DOMAIN CALM SPEED SWEEPS 0-{top} kn")
    out("  captive surge -- the speed is PRESCRIBED (s[6] = u(t) each step, "
        "the plant's captive force holds it), free heave and pitch, "
        f"plant.step (Heun) at {DT} s; R = the captive force (= calm "
        "resistance; calm_resistance's definition)")
    big = {}
    lag = {}
    for rate in (0.1, 0.05):
        for name in ("v0", "as1"):
            p = build(name)
            t0 = time.time()
            rec = ramp_sweep(p, rate_kn_s=rate)
            big[f"ramp_{name}_{rate:g}"] = rec
            up = rec[rec[:, 5] == 1]
            dn = rec[rec[:, 5] == -1]
            per = int(round(0.5 / (rate * 0.2)))     # samples per 0.5 kn
            out()
            out(f"  [{name}] slow ramp 0 -> {top} -> 0 kn at {rate:g} kn/s (20 s"
                f" holds), {time.time() - t0:.0f} s wall; read at whole knots")
            out("     kn   up: z m    trim    R N     | down: z m   trim    R"
                " N     | trim up-down")
            diffs = {}
            for kn in range(0, top + 1):
                iu = np.argmin(np.abs(up[:, 1] - kn))
                idn = np.argmin(np.abs(dn[:, 1] - kn))
                a, b = up[iu], dn[idn]
                diffs[kn] = a[3] - b[3]
                out(f"    {kn:3d}   {a[2]:+.4f} {a[3]:6.2f} {a[4]:8.0f}   |  "
                    f"{b[2]:+.4f} {b[3]:6.2f} {b[4]:8.0f}   | {a[3] - b[3]:+.3f}")
            lag[(name, rate)] = diffs
            for lab, rr in (("up", up), ("down", dn)):
                dtr = np.abs(rr[per:, 3] - rr[:-per, 3])
                j = int(np.argmax(dtr))
                out(f"     ramp {lab}: largest trim change over 0.5 kn "
                    f"{dtr[j]:.3f} deg between {rr[j, 1]:.2f} and "
                    f"{rr[j + per, 1]:.2f} kn")
    out()
    out("  LAG vs HYSTERESIS: trim up-down difference of the ramp at 0.1 and "
        "0.05 kn/s (lag halves with the rate; hysteresis does not)")
    out("     kn   v0: 0.1     0.05   ratio | as1: 0.1     0.05   ratio")
    for kn in range(1, top + 1):
        a, b = lag[("v0", 0.1)][kn], lag[("v0", 0.05)][kn]
        c, d = lag[("as1", 0.1)][kn], lag[("as1", 0.05)][kn]
        out(f"    {kn:3d}   {a:+.3f}  {b:+.3f}  {b / a if abs(a) > 1e-3 else np.nan:5.2f} |"
            f"  {c:+.3f}  {d:+.3f}  {d / c if abs(c) > 1e-3 else np.nan:5.2f}")
    out()
    kns_up = list(np.arange(0.5, 20.01, 0.5)) + list(np.arange(21, TOP_KN + 0.01, 1))
    kns_dn = kns_up[::-1]
    steps = {}
    for name in ("v0", "as1"):
        p = build(name)
        t0 = time.time()
        rows_up, s_end = step_sweep(p, kns_up)
        rows_dn, _ = step_sweep(p, kns_dn, start=s_end)
        steps[name] = dict(up=rows_up, down=rows_dn)
        out(f"  [{name}] stepped sweep, 30 s per speed, mean and peak-to-peak"
            f" over the last 10 s ({time.time() - t0:.0f} s wall)")
        out("     kn   up: z m    z p-p    trim   trim p-p  R N    | down: "
            "z m    z p-p    trim   trim p-p  R N    | trim up-down")
        dn_map = {r["kn"]: r for r in rows_dn}
        for a in rows_up:
            b = dn_map[a["kn"]]
            out(f"    {a['kn']:4.1f}  {a['z']:+.4f} {a['z_pp']:.1e} "
                f"{a['trim']:6.2f}  {a['trim_pp']:.1e} {a['R']:7.0f}  |  "
                f"{b['z']:+.4f} {b['z_pp']:.1e} {b['trim']:6.2f}  "
                f"{b['trim_pp']:.1e} {b['R']:7.0f}  | "
                f"{a['trim'] - b['trim']:+.3f}")
        half = [r for r in rows_up if r["kn"] <= 20.0]
        jumps = [(abs(half[i + 1]["trim"] - half[i]["trim"]), half[i]["kn"],
                  half[i + 1]["kn"]) for i in range(len(half) - 1)]
        jz = [(abs(half[i + 1]["z"] - half[i]["z"]), half[i]["kn"])
              for i in range(len(half) - 1)]
        mj = max(jumps)
        hyst = max(abs(a["trim"] - dn_map[a["kn"]]["trim"]) for a in rows_up)
        hk = max(rows_up, key=lambda a: abs(a["trim"]
                                            - dn_map[a["kn"]]["trim"]))["kn"]
        lc = [r["kn"] for r in rows_up + rows_dn if r["trim_pp"] > 0.01
              or r["z_pp"] > 1e-3]
        out(f"     up sweep 0.5-20 kn: largest trim change per 0.5 kn "
            f"{mj[0]:.3f} deg ({mj[1]:.1f} -> {mj[2]:.1f} kn); largest heave "
            f"change {max(jz)[0] * 100:.2f} cm ({max(jz)[1]:.1f} kn ->); "
            f"largest up-down trim difference {hyst:.3f} deg at {hk:.1f} kn;"
            f" speeds with a limit cycle (trim p-p > 0.01 deg or z p-p > "
            f"1 mm): {sorted(set(lc))}")
        out()
    big["steps"] = steps
    # ---- free surge with the jet
    levels = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
              0.22, 0.25, 0.28, 0.32, 0.36, 0.40, 0.45, 0.50, 0.60, 0.70,
              0.80, 0.90, 1.00]
    free = {}
    out("  FREE-SURGE SWEEP: the jet's thrust command stepped up and down "
        "(fraction of t_max), the speed free, 60 s per level, means and "
        "peak-to-peak over the last 10 s")
    for name in ("v0", "as1"):
        p = build(name, captive_u=None)
        t0 = time.time()
        # up, then down from the top without a reset
        rows_all = thrust_sweep(p, levels + levels[::-1][1:])
        k = len(levels)
        up, dn = rows_all[:k], rows_all[k - 1:][::-1]
        free[name] = dict(up=up, down=dn)
        out(f"  [{name}] ({time.time() - t0:.0f} s wall)")
        out("    thrust  up: kn     kn p-p   z m     trim  trim p-p | down: "
            "kn     kn p-p   z m     trim  trim p-p")
        for a, b in zip(up, dn):
            out(f"     {a['level']:.2f}   {a['kn']:6.2f}  {a['kn_pp']:.1e} "
                f"{a['z']:+.4f} {a['trim']:6.2f} {a['trim_pp']:.1e} |  "
                f"{b['kn']:6.2f}  {b['kn_pp']:.1e} {b['z']:+.4f} "
                f"{b['trim']:6.2f} {b['trim_pp']:.1e}")
        gaps = [(up[i + 1]["kn"] - up[i]["kn"], up[i]["kn"], up[i + 1]["kn"])
                for i in range(len(up) - 1)]
        out(f"     largest speed jump between thrust levels (up) "
            f"{max(gaps)[0]:.2f} kn ({max(gaps)[1]:.2f} -> "
            f"{max(gaps)[2]:.2f}); largest up-down speed difference "
            f"{max(abs(a['kn'] - b['kn']) for a, b in zip(up, dn)):.2f} kn")
    big["free"] = free
    flush("c_sweeps.txt")
    np.savez(os.path.join(OUT, "c_ramps.npz"),
             **{k: v for k, v in big.items() if k.startswith("ramp")})
    dump("c_sweeps.json", dict(steps=steps, free=free))


# ------------------------------------------------------------------ part g
def census(p, u, nz=10, nt=10, passes=4):
    """Roots of the captive equations in the box z -0.6..0.3 m, trim
    -15..12 deg: Newton from a grid of starts, then deflated Newton from the
    same starts (Farrell et al. 2015: G = F prod(1/|v - r_i|^2 + 1), scaled
    distance) until a pass finds nothing new."""
    zg = np.linspace(-0.6, 0.3, nz)
    tg = np.linspace(-15.0, 12.0, nt)
    starts = [np.array([z, th_of(p, t)]) for z in zg for t in tg]
    sc = np.array([0.05, np.radians(2.0)])
    roots = []

    def inside(v):
        t = trim_deg(p, v[1])
        return -0.6 - 1e-9 <= v[0] <= 0.3 + 1e-9 and \
            -15.0 - 1e-9 <= t <= 12.0 + 1e-9

    def add(v, r):
        if r > 1e-10 or not inside(v):
            return False
        for q in roots:
            if np.all(np.abs((v - q) / sc) < 1e-5):
                return False
        roots.append(v)
        return True
    n_eval = [0]

    def Fd(v):
        n_eval[0] += 1
        f = F(p, u, v)
        for q in roots:
            f = f * (1.0 / (((v - q) / sc) ** 2).sum() + 1.0)
        return f
    log = []
    for k in range(passes):
        new = 0
        for st in starts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol, _, ier, _ = fsolve(Fd, st, full_output=True, xtol=1e-12)
            if not np.all(np.isfinite(sol)):
                continue
            if np.abs(F(p, u, sol)).max() > 1e-4:
                continue
            v, r = polish(p, u, sol)
            new += add(v, r)
        log.append(new)
        if new == 0:
            break
    res = []
    for v in roots:
        pairs, mre, lam = modes(jac4(p, u, v[0], v[1]))
        t = trim_deg(p, v[1])
        lab = ("stable" if mre < 0 else
               "unstable, bow-down (trim < 0)" if t < 0 else
               "unstable, bow-up")
        res.append(dict(z=float(v[0]), th=float(v[1]), trim=t,
                        resid=float(np.abs(F(p, u, v)).max()), maxre=mre,
                        label=lab, lam=lam))
    res.sort(key=lambda r: -r["z"])
    return res, log, n_eval[0]


def winding(p, u, nz=60, nt=72):
    """Cells of a (z, trim) grid over the census box around which the
    residual (w', q') winds (nonzero degree): a zero inside the cell, smooth
    or at a discontinuity. Returns the cell centres with their degree."""
    zg = np.linspace(-0.6, 0.3, nz + 1)
    tg = np.linspace(-15.0, 12.0, nt + 1)
    A = np.empty((nz + 1, nt + 1))
    for i, z in enumerate(zg):
        for j, t in enumerate(tg):
            f = F(p, u, np.array([z, th_of(p, t)]))
            A[i, j] = np.arctan2(f[1] / 1.0, f[0] / 1.0)

    def dang(a, b):
        return (b - a + np.pi) % (2 * np.pi) - np.pi
    cells = []
    for i in range(nz):
        for j in range(nt):
            loop = [A[i, j], A[i + 1, j], A[i + 1, j + 1], A[i, j + 1],
                    A[i, j]]
            w = sum(dang(loop[k], loop[k + 1]) for k in range(4))
            deg = int(round(w / (2 * np.pi)))
            if deg:
                cells.append((0.5 * (zg[i] + zg[i + 1]),
                              0.5 * (tg[j] + tg[j + 1]), deg))
    return cells


def part_g():
    out("S1 (g)  ROOT CENSUS OF THE CAPTIVE EQUATIONS w' = q' = 0, and a "
        "winding-number scan")
    out("  census: grid 10 x 10 of starts over z -0.6..0.3 m, trim -15..12 "
        "deg, deflated Newton until a pass finds nothing new; roots with "
        "residual <= 1e-10, classified by the eigenvalues of the 4x4 "
        "Jacobian")
    out("  winding: degree of the residual vector around each cell of a 60 x"
        " 72 grid of the same box (0.015 m x 0.375 deg); a nonzero degree "
        "marks a zero inside the cell -- also one at a discontinuity of the "
        "residual, which Newton cannot converge to; cells within one cell of"
        " a census root are that root")
    kns = ([float(k) for k in np.arange(1.0, 9.51, 0.5)]
           + [10.0, 12.0, 16.0, 20.0, 25.0]
           + [float(k) for k in range(26, 41)])
    cen = {}
    summary = []
    for name in ("v0", "as1"):
        p = build(name)
        for kn in kns:
            u = kn * KN
            t0 = time.time()
            res, log, ne = census(p, u)
            cells = winding(p, u)
            free_cells = []
            for (zc, tc, deg) in cells:
                near = any(abs(r["z"] - zc) <= 0.0151 and
                           abs(r["trim"] - tc) <= 0.376 for r in res)
                if not near:
                    free_cells.append((round(zc, 4), round(tc, 3), deg))
            cen[f"{name}_{kn:g}"] = dict(roots=res, winding=cells,
                                         nonroot_zeros=free_cells)
            out(f"   [{name}] {kn:4.1f} kn: {len(res)} roots (passes {log}, "
                f"{ne} evaluations), {len(cells)} winding cells, "
                f"{len(free_cells)} zero(s) away from every root "
                f"{free_cells[:4]} ({time.time() - t0:.0f} s)")
            for r in res:
                out(f"        z {r['z']:+.5f}  trim {r['trim']:7.3f}  resid "
                    f"{r['resid']:.0e}  max Re(lambda) {r['maxre']:+.3f}  "
                    f"{r['label']}")
            st = [r for r in res if r["label"] == "stable"]
            summary.append((name, kn, st, res, free_cells))
    out()
    out("  STABLE STATE AT EVERY SPEED (census roots with all eigenvalues in "
        "the left half-plane; other roots listed by type; 'zeros' = winding "
        "cells away from every root)")
    out("    kn  | v0: stable (z, trim)          others            zeros | "
        "as1: stable (z, trim)         others            zeros")
    by = {(n, k): (s, r, f) for n, k, s, r, f in summary}
    for kn in kns:
        cols = []
        for name in ("v0", "as1"):
            s, r, f = by[(name, kn)]
            sts = ", ".join(f"({x['z']:+.4f}, {x['trim']:.3f})" for x in s) \
                or "NONE"
            oth = ", ".join(("bd" if x["trim"] < 0 else "bu")
                            + f"({x['z']:+.3f},{x['trim']:.2f})"
                            for x in r if x["label"] != "stable") or "-"
            cols.append(f"{sts:30s} {oth:30s} {len(f):2d}")
        out(f"   {kn:4.1f} | " + " | ".join(cols))
    flush("g_census.txt")
    dump("g_census.json", cen)


# ------------------------------------------------------------------ part f
def basin(p, u, root, dzs=(-0.2, 0.0, 0.2), dtrs=(-5.0, 0.0, 5.0), T=40.0):
    """Captive calm runs from root + (dz, dtrim): end distance from the
    root, min z, min trim, diverged (non-finite or |trim| > 30 deg)."""
    rows = []
    for dz in dzs:
        for dtr in dtrs:
            if dz == 0.0 and dtr == 0.0:
                continue
            s = state(u, root[0] + dz, th_of(p, trim_deg(p, root[1]) + dtr))
            p.captive_u = u
            p.initial_state(0.0)
            zmin, trmin, bad = 9.0, 99.0, None
            for i in range(int(round(T / p.dt))):
                s = p.step(s, i * p.dt, 0.0, 0.0)
                s[6] = u
                tr = trim_deg(p, s[4])
                if not np.all(np.isfinite(s)) or abs(tr) > 30.0:
                    bad = i * p.dt
                    break
                zmin, trmin = min(zmin, s[2]), min(trmin, tr)
            end = (np.nan if bad is not None else
                   max(abs(s[2] - root[0]),
                       abs(trim_deg(p, s[4]) - trim_deg(p, root[1])) / 10.0))
            rows.append(dict(dz=dz, dtrim=dtr, diverged_t=bad, end_dist=end,
                             zmin=zmin, trmin=trmin))
    return rows


def open_loop(p, kn, T=240.0):
    s = p.initial_state(kn * KN)
    thr = s[12]
    bad, mn_tr, mn_z = None, 99.0, 9.0
    for i in range(int(round(T / p.dt))):
        s = p.step(s, i * p.dt, thr, 0.0, p.dt)
        tr = trim_deg(p, s[4])
        mn_tr, mn_z = min(mn_tr, tr), min(mn_z, s[2])
        if not np.all(np.isfinite(s)) or abs(tr) > 30 or s[2] < -3:
            bad = i * p.dt
            break
    return dict(diverged_t=bad, min_trim=mn_tr, min_z=mn_z,
                end_kn=float(s[6] / KN), capsized=bool(p.capsized))


def bow_out(p, kn, lam_L, H, T=60.0, dt=0.01):
    """Captive (speed held) in regular head waves of length lam_L x L and
    height H, from the running attitude: min / max trim, heave amplitude,
    share of the time the bow station is out of the water, the most
    negative station flux sum (suction), diverged."""
    from sim.test_vessel import Monochromatic
    u = kn * KN
    L = p.hull.L
    k = 2 * np.pi / (lam_L * L)
    sea = Monochromatic(np.sqrt(G * k), 0.5 * H, theta0=np.pi)
    q = p.with_sea(sea, captive_u=u, dt=dt)
    s = state(u, *q.running_attitude(u))
    q.initial_state(0.0)
    q.captive_u = u
    n = int(round(T / dt))
    trs, zs, out_n, suc = [], [], 0, 0.0
    bad = None
    for i in range(n):
        if i % 5 == 0:
            _, aux = q.deriv(s, i * dt)
            out_n += int(aux["d"][-1] <= 0.0)
            mf = aux.get("mom_full", aux["mom"])
            suc = min(suc, float(np.minimum(mf, 0.0).sum() * q.dx))
        s = q.step(s, i * dt, 0.0, 0.0, dt)
        s[6] = u
        tr = trim_deg(q, s[4])
        if not np.all(np.isfinite(s)) or abs(tr) > 30.0:
            bad = i * dt
            break
        if i >= n // 3:
            trs.append(tr)
            zs.append(s[2])
    trs, zs = np.asarray(trs), np.asarray(zs)
    return dict(diverged_t=bad,
                trim_min=float(trs.min()) if trs.size else np.nan,
                trim_max=float(trs.max()) if trs.size else np.nan,
                heave_amp=float(0.5 * np.ptp(zs)) if zs.size else np.nan,
                bow_out_frac=out_n / max(1, (n + 4) // 5), min_flux=suc)


def part_f():
    out("S1 (f)  ROBUSTNESS: basin, following seas, bow rising out in "
        "regular waves")
    # ---- basin
    out("  BASIN: captive calm runs, 40 s, from the planing root + (dz, "
        "dtrim) in {-0.2, 0, +0.2} m x {-5, 0, +5} deg; end = max(|z - z_eq| "
        "m, |trim - trim_eq| / 10 deg); DIVERGED = non-finite or |trim| > 30"
        " deg")
    bas = {}
    for name in ("v0", "as1", "ungated"):
        p = build(name)
        ref = build("v0")
        for kn in (16, 20, 25, 30, 35):
            u = kn * KN
            g = np.array(ref.running_attitude(u))
            root, r = newton(p, u, g)
            if r > 1e-10:
                z, th, _, _ = settle(p, u, T=60.0)
                root = np.array([z, th])
            rows = basin(p, u, root)
            bas[f"{name}_{kn}"] = dict(root=root, resid=r, runs=rows)
            nd = [x for x in rows if x["diverged_t"] is not None]
            far = [x for x in rows if x["diverged_t"] is None
                   and x["end_dist"] > 1e-3]
            out(f"   [{name:7s}] {kn} kn root z {root[0]:+.4f} trim "
                f"{trim_deg(p, root[1]):.3f} (r {r:.0e}): {len(rows)} starts,"
                f" diverged {len(nd)} {[(x['dz'], x['dtrim'], x['diverged_t']) for x in nd]},"
                f" not back within 1 mm / 0.01 deg {len(far)} "
                f"{[(x['dz'], x['dtrim'], round(x['end_dist'], 4)) for x in far]};"
                f" min trim {min(x['trmin'] for x in rows):+.2f}")
    # ---- following seas
    from learn.repro import task
    m = task.Mission("high", 0, 0, t_end=1.0, track=task.LEGS[0])
    base = m.plant.sea
    out()
    out(f"  FOLLOWING SEAS: the task sea of mission s0 leg 0 (Hs "
        f"{getattr(base, 'hs', None)}, Tp {getattr(base, 'tp', None)}) "
        "turned by 180 deg, at Hs x1 and x1.5; open loop, constant thrust "
        "from initial_state, nozzle 0, 240 s, dt "
        f"{m.plant.dt}; DIVERGED = non-finite, |trim| > 30 deg or z < -3 m")
    fol = {}
    for scale in (1.0, 1.5):
        sea = copy.deepcopy(base)
        sea.th = np.asarray(sea.th, float) + np.pi
        sea.a = np.asarray(sea.a, float) * scale
        for name in ("v0", "as1"):
            for kn in (12, 20, 25, 30):
                p = build(name, captive_u=None, sea=sea, dt=m.plant.dt)
                r = open_loop(p, kn)
                fol[f"{name}_{scale}_{kn}"] = r
                out(f"   Hs x{scale} {name:4s} {kn:2d} kn: "
                    + (f"DIVERGED at t {r['diverged_t']:.2f} s"
                       if r["diverged_t"] is not None else "ok")
                    + f"; min trim {r['min_trim']:+.2f} deg, min z "
                    f"{r['min_z']:+.3f} m, end {r['end_kn']:.2f} kn, capsized"
                    f" {r['capsized']}")
    flush("f_robust.txt")
    dump("f_robust.json", dict(basin=bas, following=fol))


def part_w():
    out("S1 (w)  BOW RISING OUT OF THE WATER IN REGULAR HEAD WAVES")
    out("  BOW RISING OUT: captive (speed held) in regular head waves, 60 s at"
        " dt 0.01 from the running attitude, statistics over the last 40 s; "
        "bow out = share of samples with the bow station dry; suction = the "
        "most negative sum of station fluxes (N)")
    bo = {}
    for kn in (8, 10, 12, 14):
        for lam_L, H in ((2.0, 0.5), (3.0, 0.8)):
            for name in ("v0", "as1", "ungated"):
                p = build(name)
                r = bow_out(p, kn, lam_L, H)
                bo[f"{name}_{kn}_{lam_L}_{H}"] = r
                out(f"   {kn:2d} kn lam/L {lam_L} H {H} m {name:7s}: "
                    + (f"DIVERGED at t {r['diverged_t']:.2f} s; "
                       if r["diverged_t"] is not None else "")
                    + f"trim {r['trim_min']:+6.2f} .. {r['trim_max']:+6.2f} "
                    f"deg, heave amp {r['heave_amp']:.3f} m, bow out "
                    f"{100 * r['bow_out_frac']:4.1f} %, suction "
                    f"{r['min_flux']:9.1f} N")
    flush("w_bow_out.txt")
    dump("w_bow_out.json", bo)


# ------------------------------------------------------------------ part d
def regular_wave_on(make, hull, V, lam_over_L, H, T=40.0, dt=0.005,
                    x_acc=None):
    """studies/exp_planing.regular_wave, line for line, with the plant
    built by make(hull, sea, dt, V) instead of PlaningVessel."""
    from sim.test_vessel import Monochromatic
    L = hull.L
    k = 2 * np.pi / (lam_over_L * L)
    a = 0.5 * H
    om = np.sqrt(9.81 * k)
    sea = Monochromatic(om, a, theta0=np.pi)
    p = make(hull, sea, dt, V)
    x_acc = (hull.L - hull.lcg - 0.1 * hull.L) if x_acc is None else x_acc
    s = np.zeros(14)
    s[6] = V
    s[2], s[4] = p.running_attitude(V)
    n = int(T / dt)
    rec = np.empty((n, 4))
    xb = p.x[-1]
    for i in range(n):
        s = p.step(s, i * dt, 0.0, 0.0)
        s[6] = V
        a_cg, a_b = p.last_cg_acc, p.last_bow_acc
        rec[i] = s[2], s[4], a_cg / 9.81, (a_cg + (a_b - a_cg) * x_acc
                                             / xb) / 9.81
    q = rec[n // 2:]
    amp = 0.5 * (q[:, :2].max(0) - q[:, :2].min(0))
    te = 2 * np.pi / (om + k * V)
    m = max(int(round(te / dt)), 1)
    cyc = [q[j:j + m] for j in range(0, len(q) - m + 1, m)]
    pk = np.mean([[c[:, 2].max(), c[:, 3].max()] for c in cyc], axis=0)
    return amp[0] / a, amp[1] / (k * a), float(pk[0]), float(pk[1])


def part_d():
    from sim import physics
    from sim.planing_anyspeed import PlaningVesselAS
    from studies import exp_planing as EP
    out("S1 (d)  PLANING RANGE: as1 vs v0")
    out("  Newton equilibria of the captive equations (as1 started from v0's"
        " root) with 48 / 96 / 192 stations; dz, dtrim = as1 - v0 at the same"
        " station count")
    out("    kn   n    v0: z m     trim deg   | as1: z m     trim deg   | "
        "dz m        dtrim deg   resid v0/as1  taper strips chine-wetted")
    rows = []
    for kn in (14, 16, 18, 20, 22, 24, 25, 26, 28, 30, 32, 35, 38, 40):
        u = kn * KN
        for n in (48, 96, 192):
            v0, a1 = build("v0", n=n), build("as1", n=n)
            g = np.array(v0.running_attitude(u))
            s0, r0 = newton(v0, u, g)
            s1, r1 = newton(a1, u, s0)
            dzm = s1[0] - s0[0]
            dtr = trim_deg(a1, s1[1]) - trim_deg(v0, s0[1])
            nc, nt = n_taper_capped(a1, u, s1[0], s1[1])
            rows.append(dict(kn=kn, n=n, v0=s0, as1=s1, dz=dzm, dtrim=dtr,
                             r0=r0, r1=r1, trim0=trim_deg(v0, s0[1]),
                             trim1=trim_deg(a1, s1[1]), capped=nc))
            out(f"    {kn:3d} {n:4d}  {s0[0]:+.6f} {trim_deg(v0, s0[1]):.6f} "
                f"| {s1[0]:+.6f} {trim_deg(a1, s1[1]):.6f} | {dzm:+.2e}  "
                f"{dtr:+.2e}   {r0:.0e}/{r1:.0e}     {nc}/{nt}")
    out()
    out("  station-count spread (max - min over 48/96/192) of the trim and z "
        "roots, and the converged S1 departure (192 stations)")
    out("    kn   v0 trim spread  as1 trim spread  v0 z spread  as1 z spread"
        " |  dz(192) m   dtrim(192) deg")
    for kn in sorted({r["kn"] for r in rows}):
        rr = [r for r in rows if r["kn"] == kn]
        t0s = [r["trim0"] for r in rr]
        t1s = [r["trim1"] for r in rr]
        z0s = [r["v0"][0] for r in rr]
        z1s = [r["as1"][0] for r in rr]
        r192 = [r for r in rr if r["n"] == 192][0]
        out(f"    {kn:3d}     {np.ptp(t0s):.4f}          {np.ptp(t1s):.4f}"
            f"          {np.ptp(z0s) * 1e3:6.2f} mm    {np.ptp(z1s) * 1e3:6.2f}"
            f" mm | {r192['dz']:+.4f}    {r192['dtrim']:+.4f}")
    out()
    out("  running_attitude (4000-step captive run from rest, each plant's "
        "own) and calm_resistance, 48 stations")
    out("    kn   v0: z        trim     R N        | as1: z       trim     R"
        " N        | dz m       dtrim deg   dR %")
    a1, v0 = build("as1"), build("v0")
    ra = []
    for kn in (14, 16, 18, 20, 22, 24, 25, 26, 28, 30, 32, 35, 38, 40):
        u = kn * KN
        c0, c1 = v0.calm_resistance(u), a1.calm_resistance(u)
        ra.append(dict(kn=kn, v0=c0, as1=c1))
        out(f"    {kn:3d}  {c0['rise']:+.6f} {c0['trim_deg']:8.5f} "
            f"{c0['R']:9.3f} | {c1['rise']:+.6f} {c1['trim_deg']:8.5f} "
            f"{c1['R']:9.3f} | {c1['rise'] - c0['rise']:+.1e}  "
            f"{c1['trim_deg'] - c0['trim_deg']:+.1e}   "
            f"{100 * (c1['R'] / c0['R'] - 1):+.2e}")
    tops = {}
    for n in (48, 96, 192):
        tv, ta = build("v0", n=n).top_speed(), build("as1", n=n).top_speed()
        tops[n] = (tv, ta)
        out(f"  top speed (full thrust, calm), {n} stations: v0 {tv / KN:.4f} "
            f"kn, as1 {ta / KN:.4f} kn ({100 * (ta / tv - 1):+.3f} %)")
    out()
    out("  Fridsma P6 (studies/exp_planing.p6: towed, regular head waves, "
        "H/b 0.111, T 16 s, dt 0.004): v0 = exp_planing.regular_wave "
        "(PlaningVessel) itself;")
    out("  off = PlaningVesselAS all off on the hull's @as1 variant (the GZ "
        "class: isolates linear vs GZ roll, expected equal), as1 = "
        "PlaningVesselAS on the as1 preset")

    def mk(hv, sw):
        def make(hull, sea, dt, V):
            return PlaningVesselAS(hv, sea, dt=dt, captive_u=V,
                                   test_switches=sw)
        return make
    p6 = []
    worst = 0.0
    for case in ("A", "B"):
        h, V = EP.fridsma_hull(case, tow="nozzle")
        hv = physics.variant(h, "as1")
        from sim.planing_vessel import PlaningVessel
        c0 = PlaningVessel(h, calm(), dt=0.004).calm_resistance(V)
        c1 = PlaningVesselAS(hv, calm(), dt=0.004).calm_resistance(V)
        out(f"   case {case}: V {V:.3f} m/s; calm trim v0 "
            f"{c0['trim_deg']:.5f} as1 {c1['trim_deg']:.5f} deg, rise/b v0 "
            f"{c0['rise'] / h.b:.5f} as1 {c1['rise'] / h.b:.5f}, R/Delta v0 "
            f"{c0['R'] / (h.mass * G):.5f} as1 {c1['R'] / (h.mass * G):.5f}")
        out("     lam/L   heave h/H: v0     off      as1   (as1/v0-1 %)   "
            "pitch: v0     off      as1   (as1/v0-1 %)   CG g v0/as1   bow g"
            " v0/as1   measured h/H, pitch")
        for lam in EP.FRIDSMA[case]:
            hb = 0.111 * h.b
            r0 = EP.regular_wave(h, V, lam, hb, T=16.0, dt=0.004)
            rf = regular_wave_on(mk(hv, "v0"), h, V, lam, hb, T=16.0,
                                 dt=0.004)
            r1 = regular_wave_on(mk(hv, None), h, V, lam, hb, T=16.0,
                                 dt=0.004)
            eh = 100 * (r1[0] / r0[0] - 1)
            ep = 100 * (r1[1] / r0[1] - 1)
            worst = max(worst, abs(eh), abs(ep))
            meas = EP.FRIDSMA[case][lam]
            p6.append(dict(case=case, lam=lam, v0=r0, off=rf, as1=r1,
                           heave_pct=eh, pitch_pct=ep, measured=meas))
            out(f"     {lam:4.1f}       {r0[0]:.4f}  {rf[0]:.4f}  {r1[0]:.4f}"
                f"   ({eh:+.3f})        {r0[1]:.4f}  {rf[1]:.4f}  "
                f"{r1[1]:.4f}   ({ep:+.3f})       {r0[2]:.3f}/{r1[2]:.3f}"
                f"   {r0[3]:.3f}/{r1[3]:.3f}    {meas[0]:.2f}, {meas[1]:.2f}")
    out(f"  P6 heave and pitch: largest |as1/v0 - 1| {worst:.3f} %")
    out()
    out("  NON-REGRESSION GATE (validation review): |dtrim| <= 0.05 deg, |dR| "
        "<= 1 % over the planing range, top speed within +-0.5 %; dtrim from"
        " the converged (192-station) Newton roots, dR from calm_resistance "
        "(48 stations)")
    gate = []
    for kn in sorted({r["kn"] for r in rows}):
        r192 = [r for r in rows if r["kn"] == kn and r["n"] == 192][0]
        c = [x for x in ra if x["kn"] == kn]
        dR = 100 * (c[0]["as1"]["R"] / c[0]["v0"]["R"] - 1) if c else np.nan
        ok = abs(r192["dtrim"]) <= 0.05 and (np.isnan(dR) or abs(dR) <= 1.0)
        gate.append(dict(kn=kn, dtrim=r192["dtrim"], dz=r192["dz"], dR=dR,
                         ok=ok))
        out(f"    {kn:3d} kn  dtrim {r192['dtrim']:+.4f} deg  dz "
            f"{r192['dz'] * 100:+.2f} cm  dR {dR:+.2f} %   "
            f"{'pass' if ok else 'FAIL'}")
    tv, ta = tops[192]
    out(f"    top speed (192 stations) {100 * (ta / tv - 1):+.3f} %   "
        f"{'pass' if abs(ta / tv - 1) <= 0.005 else 'FAIL'}")
    flush("d_planing.txt")
    dump("d_planing.json", dict(eq=rows, calm=ra, top=tops, p6=p6,
                                gate=gate))


# ------------------------------------------------------------------ part h
def part_h():
    """High speed: station count and time step of the captive runs from
    rest, and v0's 26-27 kn limit cycle beside its fixed point."""
    from sim.planing_vessel import SP
    out("S1 (h)  HIGH SPEED: station count and time step; v0's 26-27 kn "
        "calm limit cycle")
    out("  captive calm run from the rest attitude at each speed, 100 s, mean"
        " and peak-to-peak over the last 10 s; 'wet taper' = taper strips "
        "with a chine-wetted part at the mean state; S1 = the extra strip "
        "force and moment of the complete D/Dt at that state")
    rows = []
    out("    n    plant  kn   z mean     z p-p     trim mean  trim p-p deg "
        " wet taper strips   S1 Fz N   S1 My N m")
    for n in (48, 96, 192):
        plants = dict(v0=build("v0", n=n), as1=build("as1", n=n))
        for kn in (26, 28, 30, 32, 35, 40):
            u = kn * KN
            for name, p in plants.items():
                z, th, ppz, ppt = settle(p, u, T=100.0)
                _, _, aux1 = acc(plants["as1"], u, z, th)
                cap = plants["as1"]._taper_cap
                ex = aux1["mom_full"] - aux1["mom"]
                fz = float(ex.sum() * p.dx)
                my = float((SP * p.x * ex).sum() * p.dx)
                rows.append(dict(n=n, plant=name, kn=kn, z=z,
                                 trim=trim_deg(p, th), pp_z=ppz,
                                 pp_trim_deg=float(np.degrees(ppt)),
                                 capped=cap.tolist(), s1_fz=fz, s1_my=my))
                out(f"   {n:4d}  {name:5s} {kn:3d}  {z:+.5f}  {ppz:.2e}  "
                    f"{trim_deg(p, th):8.4f}   {np.degrees(ppt):.2e}    "
                    f"{len(cap):2d} {str(cap.tolist())[:14]:14s}  {fz:8.1f}"
                    f"  {my:9.1f}")
    out()
    out("  time step (n 48): mean and peak-to-peak over the last 10 s of a "
        "100 s captive run from rest")
    out("    plant  kn   dt s     z mean     z p-p     trim mean  trim p-p "
        "deg")
    for name in ("v0", "as1"):
        for kn in (28, 30, 35):
            u = kn * KN
            for dt in (0.02, 0.01, 0.005):
                p = build(name, captive_u=u, dt=dt)
                z, th, ppz, ppt = settle(p, u, T=100.0)
                rows.append(dict(n=48, plant=name, kn=kn, dt=dt, z=z,
                                 trim=trim_deg(p, th), pp_z=ppz,
                                 pp_trim_deg=float(np.degrees(ppt))))
                out(f"    {name:5s} {kn:3d}  {dt:.3f}   {z:+.5f}  {ppz:.2e}  "
                    f"{trim_deg(p, th):8.4f}   {np.degrees(ppt):.2e}")
    out()
    out("  26-27 kn: Newton root and its eigenvalues, and the captive run (100"
        " s, last 10 s) from rest and from the 25 kn root (as the stepped "
        "sweep arrives), by station count and time step; prismatic "
        "strips near their chine-wetting point at the root (v0's centre-"
        "decided cap, both plants)")
    out("    plant  kn   n    dt     root z     trim    maxRe  | from rest: "
        "z p-p     trim p-p | from 25 kn: z p-p    trim p-p  mean trim")
    cyc = []
    for name in ("v0", "as1"):
        for kn in (26, 27):
            u = kn * KN
            for n, dt in ((48, 0.02), (48, 0.01), (48, 0.005), (96, 0.02),
                          (192, 0.02)):
                p = build(name, captive_u=u, n=n, dt=dt)
                g = np.array(build("v0", n=n).running_attitude(u))
                root, r = newton(p, u, g)
                mre = modes(jac4(p, u, root[0], root[1]))[1] \
                    if r <= 1e-10 else np.nan
                r25, _ = newton(p, 25 * KN, root)
                _, _, ppz0, ppt0 = settle(p, u, T=100.0)
                zm, thm, ppz1, ppt1 = settle(p, u, T=100.0,
                                             start=(r25[0], r25[1]))
                d = -(p.Z_rest + root[0] + p.zk + SP * p.x
                      * (p.Th_rest + root[1]))
                gap = 0.5 * np.pi * np.maximum(d, 0) / p.tanb - p.yc
                j = int(np.argmin(np.abs(gap)))
                cyc.append(dict(plant=name, kn=kn, n=n, dt=dt, root=root,
                                resid=r, maxre=mre, pp_rest=(ppz0, ppt0),
                                pp_25=(ppz1, ppt1), mean_trim_25=trim_deg(
                                    p, thm), nearest_cap=(j, gap[j])))
                out(f"    {name:5s} {kn:3d} {n:4d}  {dt:.3f}  {root[0]:+.5f} "
                    f"{trim_deg(p, root[1]):7.4f}  {mre:+.2f}  |  "
                    f"{ppz0:.2e}  {np.degrees(ppt0):.2e} |  {ppz1:.2e}  "
                    f"{np.degrees(ppt1):.2e}  {trim_deg(p, thm):.4f}   "
                    f"nearest cap: station {j} gap {1e3 * gap[j]:+.2f} mm")
    flush("h_highspeed.txt")
    dump("h_highspeed.json", dict(rows=rows, cycle=cyc))


# ------------------------------------------------------------------ part e
def part_e():
    out("S1 (e)  REGRESSION (subprocesses, one at a time)")
    py = sys.executable
    cmds = [
        [py, "-m", "studies.anyspeed_regress_as", "--freeze", "--select",
         "--golden", "--lofi", "--report", "regress_as_s1.json"],
        [py, "-m", "studies.anyspeed_regress_as", "--missions",
         "hand:s0_l0,m0:s0_l1", "--report", "regress_as_s1_missions.json"],
        [py, "-m", "studies.anyspeed_regress", "--golden", "check", "--lofi",
         "--rows", "", "--missions", "", "--report", "regress_v0_s1.json"],
    ]
    for c in cmds:
        t0 = time.time()
        r = subprocess.run(c, cwd=ROOT, capture_output=True, text=True)
        out(f"  $ {' '.join(c[1:])}")
        out(f"    exit {r.returncode}, {time.time() - t0:.0f} s")
        for line in (r.stdout + r.stderr).strip().splitlines():
            out("    " + line)
    flush("e_regress.txt")


def main():
    ap = argparse.ArgumentParser()
    for k in "abcgfwdhe":
        ap.add_argument(f"--{k}", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    t0 = time.time()
    for k in "abcgfwdhe":
        if getattr(a, k):
            globals()[f"part_{k}"]()
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
