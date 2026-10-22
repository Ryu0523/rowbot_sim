#!/usr/bin/env python3
"""
Term-by-term audit of the planing plant (sim/planing_vessel.py and
sim/planing_vessel_gz.py), scarab195, across speed. Read-only probes of the
plant's own equations; nothing in the plant is changed.

A  statics at rest: draught, which stations have wetted chines (geometric and
   Wagner), the plant's heave added mass against the geometric one.
B  the surge force at rest: the plant's pressure drag -sum f tau dx against
   W tan(trim), against the strip sum rho g sum A tau dx and against the
   hydrostatic force a wetted transom would carry, rho g Q_T.
C  calm-water equilibrium (fsolve on the captive equations, not the end of a
   time run) and the linearised heave-pitch eigenvalues per speed, with each
   strip term's share of the stiffness and damping matrices.
E  the longitudinal wave force in a long head wave at 3 kn and 25 kn: the
   pressure drag split into its buoyancy-tilt part and the rest, against the
   separate Froude-Krylov term Xw and the reference W x slope.
G  roll: the hydrostatic roll stiffness of the sections actually immersed at
   speed (times a_b) against the static W GM the plant uses at every speed.
H  sway/yaw added mass the plant uses (design attitude) against the same
   slender-body formula at the rest attitude.

Output: studies/_cache/anyspeed/audit.txt (+ audit.json)
Run:    .venv/Scripts/python.exe -m studies.anyspeed_audit
"""
import contextlib
import io
import json
import os
import sys
import warnings

import numpy as np
from scipy.optimize import fsolve

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "studies", "_cache", "anyspeed")
KN = 0.514444
G = 9.81
LINES = []


def out(s=""):
    print(s)
    LINES.append(s)


def build(sea=None, captive_u=None):
    from sim import config
    from sim.planing_vessel_gz import PlaningVesselGZ
    from sim.test_vessel import Monochromatic
    with contextlib.redirect_stdout(io.StringIO()):
        h, _ = config.load("scarab195")
    sea = Monochromatic(1.0, 0.0) if sea is None else sea
    p = PlaningVesselGZ(h, sea, dt=h.scales()["dt"], captive_u=captive_u)
    return h, p


# ------------------------------------------------------------------ helpers
def q_first_moment(p, d, yc):
    """rho-free first moment about the surface, int depth dA = int_0^d A(s)
    ds, of a V-then-vertical-side section at keel depth d."""
    tb = p.tanb
    dch = yc * tb
    if d <= dch:
        return d ** 3 / (3.0 * tb)
    # below chine part to dch then rectangle growth
    q1 = dch ** 3 / (3.0 * tb)
    s = d - dch
    a_ch = dch ** 2 / tb
    return q1 + a_ch * s + yc * s * s     # int_dch^d (a_ch + 2 yc (s'-dch))


def terms(p, z, th, w, q, u, e=None, et=None, ett=None):
    """The plant's strip terms (calm unless e/et/ett given), exactly as
    deriv lines 261-281 form them. Returns dict of per-station arrays."""
    SP = p_SP(p)
    x = p.x
    n = x.size
    e = np.zeros(n) if e is None else e
    et = np.zeros(n) if et is None else et
    ett = np.zeros(n) if ett is None else ett
    Th = p.Th_rest + th
    d = e - (p.Z_rest + z + p.zk + SP * x * Th)
    tau = p.kp + SP * Th
    V = et - (w + SP * x * q) + u * tau
    A, c, ma, dma, girth = p._section(d)
    ab = p._a_b(u)
    ctr = p._ctr(u)
    ma, dma = ma * ctr, dma * ctr
    mom = np.where(V > 0.0, dma * V * V, 0.0)
    cf = 0.5 * p.rho * p.c_dc * 2.0 * c * V * np.abs(V) * ctr
    buoy = ab * p.rho * G * A * ctr
    uq = ma * 2.0 * SP * u * q
    kpp = -ma * u * u * p.kpp
    mett = ma * ett
    # NOT in the plant: the x-derivative of the sectional added mass at fixed
    # immersion, which D/Dt(m_a V) also contains. Nonzero only where the
    # Wagner half-beam is clipped at the chine (m_a = k_a pi/2 rho yc(x)^2)
    # and the chine beam varies along the hull (the forebody taper).
    ycp = np.gradient(p.yc, x)
    sat = (0.5 * np.pi * np.maximum(d, 0.0) / p.tanb) >= p.yc
    dmadx = np.where(sat & (d > 0), p.k_a * np.pi * p.rho * p.yc * ycp,
                     0.0) * ctr
    dmx = -u * V * dmadx
    return dict(d=d, tau=tau, V=V, A=A, c=c, ma=ma, dma=dma, girth=girth,
                mom=mom, cf=cf, buoy=buoy, uq=uq, kpp=kpp, mett=mett, ab=ab,
                dmx=dmx)


def p_SP(p):
    from sim.planing_vessel import SP
    return SP


def captive_rhs(p, z, th, w, q, u, drop=(), extra=()):
    """(wd, qd, F, M) of the captive calm plant, rebuilt from the terms.
    `drop`: names of strip terms left out of the generalised force."""
    SP = p_SP(p)
    T = terms(p, z, th, w, q, u)
    x, dx = p.x, p.dx
    rest = np.zeros_like(x)
    for k in ("mom", "cf", "buoy", "uq", "kpp", "mett") + tuple(extra):
        if k not in drop:
            rest = rest + T[k]
    wet = T["d"] > 0
    S = (T["girth"] * wet).sum() * dx
    lw = wet.sum() * dx
    uu = max(abs(u), 0.05)
    re = uu * max(lw, 0.05) / 1.19e-6
    cfr = 0.075 / (np.log10(re) - 2.0) ** 2 + 0.0004
    Df = 0.5 * p.rho * u * abs(u) * cfr * S
    zw = float(np.mean(p.zk[wet])) if wet.any() else p.zk[0]
    Da = 0.5 * 1.225 * p.hull.cda * u * abs(u)
    xn, _, zn = p.thrusters[0]
    pres = -(rest * T["tau"]).sum() * dx
    fx = Df + Da - pres
    My_o = -zw * Df - 0.5 * Da + zn * fx
    ma = T["ma"]
    A33 = ma.sum() * dx
    A35 = (ma * SP * x).sum() * dx
    A55 = (ma * x * x).sum() * dx
    Fz = rest.sum() * dx - p.m * G
    My = (SP * x * rest).sum() * dx + My_o
    m11, m12, m22 = p.m + A33, A35, p.Iy + A55
    det = m11 * m22 - m12 * m12
    wd = (Fz * m22 - m12 * My) / det
    qd = (m11 * My - m12 * Fz) / det
    return wd, qd, Fz, My, np.array([[m11, m12], [m12, m22]]), T


def term_force(p, xs, u, name):
    """Generalised (Fz, My) of ONE strip term at state xs, including its
    share of the captive jet's pitch moment (zn x sum term tau dx)."""
    SP = p_SP(p)
    z, th, w, q = xs
    T = terms(p, z, th, w, q, u)
    f = T[name]
    xn, _, zn = p.thrusters[0]
    Fz = f.sum() * p.dx
    My = (SP * p.x * f).sum() * p.dx + zn * (f * T["tau"]).sum() * p.dx
    return np.array([Fz, My])


def equilibrium(p, u, guess=(0.0, 0.0), extra=(), only=False):
    def res(v):
        wd, qd, *_ = captive_rhs(p, v[0], v[1], 0.0, 0.0, u, extra=extra)
        return [wd, qd]
    best = None
    gs = (guess,) if only else (guess, (0.0, 0.0), (0.05, -0.05),
                                (0.1, -0.08), (-0.05, 0.0), (0.15, -0.03))
    for g in gs:
        sol, info, ier, _ = fsolve(res, g, full_output=True, xtol=1e-12)
        r = np.abs(res(sol)).max()
        if ier == 1 and r < 1e-8:
            return sol, r
        if best is None or r < best[1]:
            best = (sol, r)
    return best


def jac(fun, x0, eps):
    x0 = np.asarray(x0, float)
    f0 = np.asarray(fun(x0))
    J = np.zeros((f0.size, x0.size))
    for i in range(x0.size):
        h = np.zeros_like(x0)
        h[i] = eps[i]
        J[:, i] = (np.asarray(fun(x0 + h)) - np.asarray(fun(x0 - h))) \
            / (2 * eps[i])
    return J


def modes(J):
    lam = np.linalg.eigvals(J)
    res = []
    for l in lam:
        if l.imag > 1e-9:
            wn = abs(l)
            res.append((wn, -l.real / wn, 2 * np.pi / l.imag))
    if not res:
        res = [(abs(l), np.nan, np.nan) for l in lam if l.real != 0]
    return sorted(res), lam


# ------------------------------------------------------------------ A, B
def part_ab(h, p):
    SP = p_SP(p)
    out("A  STATICS AT REST (scarab195, PlaningVesselGZ)")
    th0 = p.Th_rest
    d0 = -(p.Z_rest + p.zk + SP * p.x * th0)
    tau0 = p.kp + SP * th0
    dch = p.yc * p.tanb
    dwag = 2.0 / np.pi * dch
    d_T = d0[0] + tau0[0] * 0.5 * p.dx
    out(f"   Z_rest {p.Z_rest:+.4f} m (CG above still water), Th_rest "
        f"{np.degrees(th0):+.3f} deg (SP={SP:+.0f}: bow-up trim "
        f"{np.degrees(SP * th0):.3f} deg), keel slope tau0 {tau0[0]:.4f}")
    out(f"   draught at transom {d_T:.3f} m, at first station {d0[0]:.3f} m,"
        f" at bow station {d0[-1]:.3f} m; chine height aft "
        f"{dch[0]:.3f} m; Wagner chine-wetting depth (2/pi) yc tan(b) "
        f"{dwag[0]:.3f} m")
    nw = int((d0 > dwag).sum())
    ng = int((d0 > dch).sum())
    out(f"   stations (of {p.x.size}) with d > Wagner chine depth (dma = 0,"
        f" no momentum term): {nw}; with d > geometric chine: {ng}")
    A, c, ma, dma, girth = p._section(d0)
    cgeo = np.minimum(np.maximum(d0, 0) / p.tanb, p.yc)
    ma_geo = p.k_a * 0.5 * np.pi * p.rho * cgeo ** 2
    A33 = ma.sum() * p.dx
    A33g = ma_geo.sum() * p.dx
    bwl = np.where(d0 < dch, 2 * np.maximum(d0, 0) / p.tanb, 2 * p.yc)
    C33 = p.rho * G * bwl.sum() * p.dx
    Tn = 2 * np.pi * np.sqrt((p.m + A33) / C33)
    out(f"   heave added mass plant (Wagner c = pi/2 d/tan b, saturating at "
        f"yc) {A33:.0f} kg = {A33 / p.m:.2f} m; with the geometric "
        f"half-beam {A33g:.0f} kg; mass {p.m:.0f}")
    out(f"   heave stiffness rho g A_wp {C33:.0f} N/m; uncoupled heave period"
        f" {Tn:.2f} s")
    out(f"   GM {p.GM:.3f} m, Ix {p.Ix:.0f}, roll period "
        f"{2 * np.pi / np.sqrt(p.m * G * p.GM / p.Mtot[3, 3]):.2f} s")
    out()
    out("B  SURGE FORCE AT REST, calm, zero thrust")
    s = np.zeros(14)
    ds, aux = p.deriv(s, 0.0)
    W = p.m * G
    sumAt = (p.rho * G * A * tau0).sum() * p.dx
    QT = p.rho * G * q_first_moment(p, d_T, p.yc[0])
    QT1 = p.rho * G * q_first_moment(p, d0[0], p.yc[0])
    out(f"   plant: u' {ds[6]:+.4f} m/s2, Xp (pressure drag) {aux['Xp']:+.1f}"
        f" N, friction {aux['Df']:.2f} N")
    out(f"   W tan(trim) {W * np.tan(tau0[0]):.1f} N;  rho g sum A tau dx "
        f"{sumAt:.1f} N")
    out(f"   hydrostatic force on a wetted transom rho g Q_T (V section, d_T "
        f"{d_T:.3f} m): {QT:.1f} N  (at the first station's d: {QT1:.1f} N)")
    out("   -> the strip sum is the aft tilt of the bottom pressure; on a "
        "closed hull it is balanced by the transom's hydrostatic thrust, "
        "which the plant does not have")
    out()
    return dict(d_T=d_T, A33=A33, A33geo=A33g, C33=C33, Tn=Tn,
                Xp_rest=aux["Xp"], ud_rest=ds[6], QT=QT, sumAt=sumAt,
                n_wagner_wet=nw, n_geo_wet=ng)


# ------------------------------------------------------------------ C
def part_c(h, p):
    out("C  CALM EQUILIBRIUM (fsolve, captive surge) AND LINEAR HEAVE-PITCH "
        "MODES")
    out("   check: rebuilt captive rhs against plant.deriv")
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(5):
        u = rng.uniform(0.5, 15.0)
        z, th = rng.uniform(-0.05, 0.1), rng.uniform(-0.05, 0.02)
        w, q = rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2)
        s = np.zeros(14)
        s[2], s[4], s[6], s[8], s[10] = z, th, u, w, q
        p.captive_u = u
        ds, _ = p.deriv(s, 0.0)
        p.captive_u = None
        wd, qd, *_ = captive_rhs(p, z, th, w, q, u)
        errs.append(max(abs(wd - ds[8]), abs(qd - ds[10])))
    out(f"   max |difference| in (w', q') over 5 random states: "
        f"{max(errs):.2e}")
    out("   kn   z_eq    trim   snapshot(z,trim)   resid   modes: T_n s / "
        "zeta (two pairs)      det(K_sym)  K35-K53 skew   D_sym eig")
    rows = []
    names = ("buoy", "mom", "cf", "uq")
    guess = (0.0, 0.0)
    SP = p_SP(p)
    for kn in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18,
               20, 25, 30, 35):
        u = kn * KN
        sol, r = equilibrium(p, u, guess)
        guess = tuple(sol)
        z, th = sol
        snap = p.running_attitude(u) if u > 0 else (0.0, 0.0)

        def f4(v):
            wd, qd, *_ = captive_rhs(p, v[0], v[1], v[2], v[3], u)
            return [v[2], v[3], wd, qd]
        x0 = [z, th, 0.0, 0.0]
        eps = [1e-6, 1e-7, 1e-6, 1e-7]
        J = jac(f4, x0, eps)
        md, lam = modes(J)
        # generalised force derivatives per term: K = -dF/d(z,th),
        # D = -dF/d(w,q)
        _, _, _, _, Mm, T = captive_rhs(p, z, th, 0, 0, u)
        Ks, Ds = {}, {}
        for nm in names:
            Jt = jac(lambda v: term_force(p, v, u, nm), x0, eps)
            Ks[nm], Ds[nm] = -Jt[:, :2], -Jt[:, 2:]
        Ktot = sum(Ks.values())
        Dtot = sum(Ds.values())
        Ksym = 0.5 * (Ktot + Ktot.T)
        skew = Ktot[0, 1] - Ktot[1, 0]
        Dsym_eig = np.linalg.eigvalsh(0.5 * (Dtot + Dtot.T))
        # eigenvalues with one term's perturbation removed
        Minv = np.linalg.inv(Mm)
        drop = {}
        for nm in names:
            Jd = J.copy()
            Jd[2:, :2] += Minv @ Ks[nm]
            Jd[2:, 2:] += Minv @ Ds[nm]
            drop[nm] = modes(Jd)[0]
        # wagner chine-wet stations at equilibrium
        dch = p.yc * p.tanb
        nwag = int((T["d"] > 2 / np.pi * dch).sum())
        mstr = "  ".join(f"{m[2]:5.2f}/{m[1]:+.3f}" for m in md[:2])
        out(f"   {kn:3d} {z:+.4f} {np.degrees(SP * (p.Th_rest + th)):6.2f}"
            f"  ({snap[0]:+.3f},{np.degrees(SP * (p.Th_rest + snap[1])):5.2f})"
            f"  {r:.0e}  {mstr:30s} {np.linalg.det(Ksym):10.3e} "
            f"{skew:+10.0f}  [{Dsym_eig[0]:+.0f},{Dsym_eig[1]:+.0f}]")
        rows.append(dict(kn=kn, z=z, th=th, trim_deg=float(np.degrees(
            SP * (p.Th_rest + th))), resid=float(r), modes=md,
            snap=snap, K=Ktot.tolist(), D=Dtot.tolist(),
            Kterms={k: v.tolist() for k, v in Ks.items()},
            Dterms={k: v.tolist() for k, v in Ds.items()},
            drop={k: v for k, v in drop.items()}, n_wagner_wet=nwag,
            M=Mm.tolist(), ab=T["ab"],
            buoy_share=float(T["buoy"].sum() * p.dx / (p.m * G))))
    out()
    out("   damping ratio of the least-damped pair with ONE term's "
        "perturbation removed (equilibrium kept)")
    out("   kn   full     -buoy    -mom     -cf      -uq(2 SP u q m_a)   "
        "D33  D35  D53  D55 (N s/m, N s, N s, N m s)   chine-wet(Wagner)")
    for rw in rows:
        def least(md):
            v = [m[1] for m in md if np.isfinite(m[1])]
            return min(v) if v else np.nan
        D = np.array(rw["D"])
        out(f"   {rw['kn']:3d}  {least(rw['modes']):+.3f}   "
            + "  ".join(f"{least(rw['drop'][k]):+.3f} " for k in names)
            + f"           {D[0, 0]:7.0f} {D[0, 1]:7.0f} {D[1, 0]:7.0f} "
            f"{D[1, 1]:7.0f}      {rw['n_wagner_wet']}")
    out()
    out("   per-term damping matrices D (= -dF/d(w,q)) at 0, 4, 6, 8, 25 kn")
    for rw in rows:
        if rw["kn"] not in (0, 4, 6, 8, 25):
            continue
        out(f"   {rw['kn']} kn: " + "; ".join(
            f"{k} [[{v[0][0]:.0f},{v[0][1]:.0f}],[{v[1][0]:.0f},"
            f"{v[1][1]:.0f}]]" for k, v in rw["Dterms"].items()))
        out(f"        K: " + "; ".join(
            f"{k} [[{v[0][0]:.0f},{v[0][1]:.0f}],[{v[1][0]:.0f},"
            f"{v[1][1]:.0f}]]" for k, v in rw["Kterms"].items()))
    out()
    return rows


def part_c2(h, p):
    SP = p_SP(p)
    out("C2 BRANCHES AND THE MISSING d m_a/dx TERM")
    out("   (a) equilibria started from the captive run's end state "
        "(running_attitude) -- the planing branch above ~10 kn")
    out("   kn   z_eq    trim   resid  modes T_n/zeta          buoy_share "
        "chine-wet(Wagner)  wetted_len")
    rows = []
    for kn in (8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 25, 30, 35):
        u = kn * KN
        snap = p.running_attitude(u)
        sol, r = equilibrium(p, u, snap, only=True)
        z, th = sol

        def f4(v):
            wd, qd, *_ = captive_rhs(p, v[0], v[1], v[2], v[3], u)
            return [v[2], v[3], wd, qd]
        x0 = [z, th, 0.0, 0.0]
        J = jac(f4, x0, [1e-6, 1e-7, 1e-6, 1e-7])
        md, _ = modes(J)
        *_, T = captive_rhs(p, z, th, 0, 0, u)
        dch = p.yc * p.tanb
        nwag = int((T["d"] > 2 / np.pi * dch).sum())
        lw = (T["d"] > 0).sum() * p.dx
        bs = T["buoy"].sum() * p.dx / (p.m * G)
        mstr = "  ".join(f"{m[2]:5.2f}/{m[1]:+.3f}" for m in md[:2])
        out(f"   {kn:3d} {z:+.4f} {np.degrees(SP * (p.Th_rest + th)):6.2f}"
            f"  {r:.0e}  {mstr:28s} {bs:5.2f}   {nwag:3d}   {lw:5.2f}")
        rows.append(dict(kn=kn, z=z, th=th, ab=T["ab"], modes=md))
    out()
    out("   (b) linear heave-pitch modes with the plant's terms, and with "
        "-u V dm_a/dx|_d added (equilibrium re-solved), displacement branch")
    out("   kn   plant: T_n/zeta (pairs)          +dm_a/dx: T_n/zeta "
        "(pairs)         pitch damping D55 plant -> with term; transom "
        "estimate U x_T^2 m_aT")
    guess = (0.0, 0.0)
    for kn in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        u = kn * KN
        res = []
        for extra in ((), ("dmx",)):
            sol, r = equilibrium(p, u, guess, extra=extra)
            z, th = sol

            def f4(v, extra=extra):
                wd, qd, *_ = captive_rhs(p, v[0], v[1], v[2], v[3], u,
                                         extra=extra)
                return [v[2], v[3], wd, qd]
            x0 = [z, th, 0.0, 0.0]
            eps = [1e-6, 1e-7, 1e-6, 1e-7]
            J = jac(f4, x0, eps)
            md, _ = modes(J)
            names = ("buoy", "mom", "cf", "uq") + extra
            D = sum(-jac(lambda v, nm=nm: term_force(p, v, u, nm), x0,
                         eps)[:, 2:] for nm in names)
            res.append((md, D, sol))
        *_, T = captive_rhs(p, res[0][2][0], res[0][2][1], 0, 0, u)
        xT = p.x[0] - 0.5 * p.dx
        est = u * xT ** 2 * T["ma"][0]
        a = "  ".join(f"{m[2]:5.2f}/{m[1]:+.3f}" for m in res[0][0][:2])
        b = "  ".join(f"{m[2]:5.2f}/{m[1]:+.3f}" for m in res[1][0][:2])
        out(f"   {kn:3d}  {a:30s}  {b:30s}  {res[0][1][1, 1]:+8.0f} -> "
            f"{res[1][1][1, 1]:+8.0f}   {est:8.0f}")
    out()
    return rows


# ------------------------------------------------------------------ E
def part_e(h):
    from sim.test_vessel import Monochromatic
    out("E  LONGITUDINAL WAVE FORCE SPLIT, captive surge, regular head wave "
        "lambda 40 m, amplitude 0.25 m")
    lam = 40.0
    k = 2 * np.pi / lam
    w0 = np.sqrt(G * k)
    a = 0.25
    res = {}
    for kn in (3.0, 25.0):
        u = kn * KN
        hh, p = build(Monochromatic(w0, a, theta0=np.pi), captive_u=u)
        SP = p_SP(p)
        s = p.initial_state(0.0)
        if u > 0:
            z0, th0 = p.running_attitude(u)
            s[2], s[4] = z0, th0
        s[6] = u
        dt = p.dt
        rec = []
        n = int(60.0 / dt)
        for i in range(n):
            t = i * dt
            ds, aux = p.deriv(s, t)
            x = p.x
            X = s[0] + x * np.cos(s[5])
            Y = s[1] + x * np.sin(s[5])
            e, et, ett, ex, ey = p._waves(X, Y, t)
            T = terms(p, s[2], s[4], s[8], s[10], u, e, et, ett)
            f = (T["buoy"] + T["mom"] + T["cf"] + T["uq"] + T["kpp"]
                 + T["mett"] - T["ma"] * (aux["wd"] + SP * x * aux["qd"]))
            Xp = -(f * T["tau"]).sum() * p.dx
            Xp_b = -(T["buoy"] * T["tau"]).sum() * p.dx
            s_lon = ex * np.cos(s[5]) + ey * np.sin(s[5])
            Xw = -(p.rho * G * T["ab"] * T["A"] * s_lon).sum() * p.dx
            # what a closed hull (wetted transom) has: FK only, full rho g
            Xfk1 = -(p.rho * G * T["A"] * s_lon).sum() * p.dx
            slope_cg = float(np.interp(0.0, x, s_lon))
            rec.append((t, aux["Xp"], Xp, Xp_b, Xw, Xfk1, slope_cg, s[4]))
            s = p.step(s, t, 0.0, 0.0)
            s[6] = u
        R = np.array(rec[int(20 / dt):])
        t = R[:, 0]
        we = w0 + k * u          # head seas encounter
        E = np.exp(-1j * we * t)

        def amp(y):
            return 2 * np.mean((y - y.mean()) * E)
        W = p.m * G
        ref = -W * amp(R[:, 6])
        cols = dict(Xp_plant=R[:, 1], Xp_rebuilt=R[:, 2], Xp_buoy=R[:, 3],
                    Xp_rest=R[:, 2] - R[:, 3], Xw=R[:, 4],
                    FK_full=R[:, 5], plant_total=R[:, 4] + R[:, 1])
        out(f"   {kn:4.1f} kn  |ref = -W slope_cg| {abs(ref):.0f} N; "
            f"pitch amp / slope amp {abs(amp(R[:, 7])) / abs(amp(R[:, 6])):.2f}")
        rr = {}
        for nm, y in cols.items():
            c = amp(y) / ref
            rr[nm] = (float(abs(c)), float(np.degrees(np.angle(c))),
                      float(y.mean()))
            out(f"        {nm:12s} {abs(c):5.2f} @ {np.degrees(np.angle(c)):+5.0f}"
                f" deg   mean {y.mean():+8.1f} N")
        res[kn] = rr
    out("   -> at low speed the oscillating buoyancy-tilt part of Xp is a "
        "second copy of the Froude-Krylov force Xw")
    out()
    return res


# ------------------------------------------------------------------ G, H
def part_gh(h, p, rows):
    SP = p_SP(p)
    out("G  ROLL STIFFNESS: hydrostatic stiffness of the sections immersed "
        "at the calm equilibrium (x a_b) against the static W GM used at "
        "every speed")
    W = p.m * G
    for rw in rows:
        if rw["kn"] not in (0, 5, 10, 15, 20, 25, 35) and rw["kn"] != 16:
            continue
        d = -(p.Z_rest + rw["z"] + p.zk + SP * p.x * (p.Th_rest + rw["th"]))
        A = p._section(d)[0]
        half = np.minimum(np.maximum(d, 0) / p.tanb, p.yc)
        It = (2.0 / 3.0 * half ** 3).sum() * p.dx
        vol = A.sum() * p.dx
        kb = ((2.0 / 3.0) * np.maximum(d, 0) * A).sum() * p.dx / max(vol,
                                                                    1e-9)
        # moment per radian about the CG of the hydrostatic pressure on the
        # immersed sections (keel as reference, CG vcg above keel; the
        # running keel is at the same body height)
        k_h = rw["ab"] * p.rho * G * (It + vol * (kb - p.hull.vcg))
        out(f"   {rw['kn']:3d} kn  a_b {rw['ab']:.2f}  immersed vol "
            f"{vol:.3f} m3  hydrostatic roll stiffness {k_h:8.0f} N m/rad  "
            f"plant W GM {W * p.GM:8.0f}  ratio {k_h / (W * p.GM):5.2f}")
    out()
    out("H  SWAY / YAW ADDED MASS (slender-body m22 = pi/2 rho d^2) at rest "
        "and at the design attitude the plant freezes")
    for rw in rows:
        if rw["kn"] not in (0, 5, 10, 25):
            continue
        d = -(p.Z_rest + rw["z"] + p.zk + SP * p.x * (p.Th_rest + rw["th"]))
        m22 = 0.5 * np.pi * p.rho * np.maximum(d, 0) ** 2
        a22 = m22.sum() * p.dx
        a26 = (m22 * p.x).sum() * p.dx
        a66 = (m22 * p.x ** 2).sum() * p.dx
        out(f"   {rw['kn']:3d} kn  A22 {a22:6.0f} kg  A26 {a26:6.0f} kg m  "
            f"A66 {a66:6.0f} kg m2   (plant: A22 {p.Mtot[1, 1] - p.m:.0f}, "
            f"A26 0, A66 {p.Mtot[5, 5] - p.Iz:.0f}; m {p.m:.0f}, "
            f"Iz {p.Iz:.0f})")
    out()


def part_heun(p):
    out("J  INTEGRATOR (Heun, dt 0.02 s): growth per cycle of an undamped "
        "oscillator, as an equivalent damping ratio")
    dt = p.dt
    for nm, Tn in (("heave at rest", 2.06), ("roll", 1.41),
                   ("heave 25 kn", 1.47)):
        w = 2 * np.pi / Tn
        g = np.sqrt(1 + (w * dt) ** 4 / 4)
        nstep = Tn / dt
        zeta = -np.log(g) * nstep / (2 * np.pi)
        out(f"   {nm:14s} T {Tn:.2f} s: |G| per step {g:.8f}, equivalent "
            f"zeta {zeta:+.1e}")
    out()


def main():
    os.makedirs(OUT, exist_ok=True)
    h, p = build()
    ab = part_ab(h, p)
    if "--c2" in sys.argv:
        rows2 = part_c2(h, p)
        rows = [dict(kn=0, z=0.0, th=0.0, ab=1.0)] + rows2
        part_gh(h, p, rows)
        with open(os.path.join(OUT, "audit_c2.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(LINES) + "\n")
        return
    rows = part_c(h, p)
    rows2 = part_c2(h, p)
    by = {r["kn"]: r for r in rows}
    for r in rows2:
        if r["kn"] >= 10:
            by[r["kn"]] = r
    part_gh(h, p, [by[k] for k in sorted(by)])
    part_heun(p)
    e = part_e(h)
    with open(os.path.join(OUT, "audit.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(ab=ab, rows=rows, e={str(k): v for k, v in e.items()}),
                  fh, default=lambda o: o.tolist() if hasattr(o, "tolist")
                  else str(o), indent=1)


if __name__ == "__main__":
    main()
