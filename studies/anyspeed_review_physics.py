#!/usr/bin/env python3
"""
Adversarial review probes of design "as1" (read-only; nothing tracked is
changed). Reuses the rebuilt captive equations of studies/anyspeed_audit.py.

  1. The fold/branch claim under the as1 closures rather than the v0 C_V blend:
     complete D/Dt ("dmx") with
       (a) a_b = 1 (wet transom; what as1 has below ventilation),
       (b) a_b = a_dry + (1 - a_dry) eps, eps = E(Fn_T) on the INSTANTANEOUS
           transom draught (smoothstep, two threshold bands),
       (c) as (b) plus the wet-transom face force and its pitch moment,
       (d) as (c) plus the 3D factor J (uniform 0.5 proxy) weighted by eps
           on m_a (2.5 of the design).
     Continuation in speed from 1 kn and a multi-start enumeration of all
     calm equilibria per speed.
  2. The longitudinal-force identity term by term on the scarab geometry
     (stem rise 0: the bow closes only in plan).

Output: studies/_cache/anyspeed/review_physics.txt
"""
import os
import sys

import numpy as np
from scipy.optimize import fsolve

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "studies"))

import anyspeed_audit as AU  # noqa: E402

OUT = os.path.join(HERE, "studies", "_cache", "anyspeed",
                   "review_physics.txt")
KN, G = AU.KN, AU.G
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def smooth(x, a, b):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def section_first_moment(p, h, yc):
    return AU.q_first_moment(p, h, yc)


def face_cp_height(p, h, yc, n=200):
    """height above keel of the centre of pressure of a hydrostatic face of
    a V-then-vertical section wetted to height h above the keel."""
    if h <= 0:
        return 0.0
    z = np.linspace(0, h, n)
    dch = yc * p.tanb
    w = np.where(z < dch, 2 * z / p.tanb, 2 * yc)
    pr = h - z
    return float(np.trapezoid(pr * w * z, z) / max(np.trapezoid(pr * w, z),
                                                    1e-12))


class Cfg:
    def __init__(self, ab="plant", band=(3.0, 3.5), face=False, J=1.0,
                 dmx=True):
        self.ab, self.band, self.face, self.J, self.dmx = ab, band, face, \
            J, dmx


def eps_of(p, d, u, cfg):
    dT = d[0] + (d[0] - d[1]) * 0.5
    if dT <= 1e-4:
        return 0.0, dT, np.inf
    FnT = abs(u) / np.sqrt(G * dT)
    return float(1.0 - smooth(FnT, *cfg.band)), dT, FnT


def terms(p, z, th, w, q, u, cfg):
    SP = AU.p_SP(p)
    x = p.x
    Th = p.Th_rest + th
    d = -(p.Z_rest + z + p.zk + SP * x * Th)
    tau = p.kp + SP * Th
    V = -(w + SP * x * q) + u * tau
    A, c, ma, dma, girth = p._section(d)
    eps, dT, FnT = eps_of(p, d, u, cfg)
    if cfg.ab == "plant":
        ab = p._a_b(u)
    elif cfg.ab == "one":
        ab = 1.0
    elif cfg.ab == "eps":
        ab = 0.5 + 0.5 * eps
    else:
        ab = float(cfg.ab)
    Jf = 1.0 + eps * (cfg.J - 1.0)
    ma, dma = ma * Jf, dma * Jf
    mom = np.where(V > 0.0, dma * V * V, 0.0)
    cf = 0.5 * p.rho * p.c_dc * 2.0 * c * V * np.abs(V)
    buoy = ab * p.rho * G * A
    uq = ma * 2.0 * SP * u * q
    kpp = -ma * u * u * p.kpp
    ycp = np.gradient(p.yc, x)
    sat = (0.5 * np.pi * np.maximum(d, 0.0) / p.tanb) >= p.yc
    dmadx = np.where(sat & (d > 0), p.k_a * np.pi * p.rho * p.yc * ycp,
                     0.0) * Jf
    dmx = -u * V * dmadx if cfg.dmx else np.zeros_like(x)
    # face force (wet part of transom, hydrostatic about its own lowered
    # surface): horizontal, forward
    FT, zT = 0.0, 0.0
    if cfg.face and dT > 0:
        hw = eps * dT
        FT = p.rho * G * AU.q_first_moment(p, hw, p.yc[0])
        zT = p.zk[0] + face_cp_height(p, hw, p.yc[0])
    return dict(d=d, tau=tau, V=V, A=A, c=c, ma=ma, girth=girth, mom=mom,
                cf=cf, buoy=buoy, uq=uq, kpp=kpp, dmx=dmx, ab=ab, eps=eps,
                dT=dT, FnT=FnT, FT=FT, zT=zT)


def rhs(p, z, th, w, q, u, cfg):
    SP = AU.p_SP(p)
    T = terms(p, z, th, w, q, u, cfg)
    x, dx = p.x, p.dx
    rest = T["mom"] + T["cf"] + T["buoy"] + T["uq"] + T["kpp"] + T["dmx"]
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
    fx = Df + Da - pres - T["FT"]
    My_o = -zw * Df - 0.5 * Da + zn * fx + T["zT"] * T["FT"]
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
    return wd, qd, fx, T


def maxre(p, u, sol, cfg):
    def f4(v):
        wd, qd, *_ = rhs(p, v[0], v[1], v[2], v[3], u, cfg)
        return [v[2], v[3], wd, qd]
    J = AU.jac(f4, [sol[0], sol[1], 0.0, 0.0], [1e-6, 1e-7, 1e-6, 1e-7])
    lam = np.linalg.eigvals(J)
    return float(np.max(lam.real))


def solve(p, u, g, cfg):
    def res(v):
        wd, qd, *_ = rhs(p, v[0], v[1], 0.0, 0.0, u, cfg)
        return [wd, qd]
    sol, info, ier, _ = fsolve(res, g, full_output=True, xtol=1e-12)
    return sol, float(np.abs(res(sol)).max())


def trim_deg(p, th):
    return float(np.degrees(AU.p_SP(p) * (p.Th_rest + th)))


def th_of_trim(p, tr):
    return np.radians(tr) / AU.p_SP(p) - p.Th_rest


def enumerate_eq(p, u, cfg):
    sols = []
    for z0 in (-0.3, -0.15, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2):
        for tr in (-6.0, -2.0, 0.0, 2.0, 3.5, 5.0, 7.0, 10.0):
            sol, r = solve(p, u, (z0, th_of_trim(p, tr)), cfg)
            if r > 1e-9 or not np.all(np.isfinite(sol)):
                continue
            if abs(sol[0]) > 1.0 or abs(trim_deg(p, sol[1])) > 25:
                continue
            if any(abs(sol[0] - s[0]) < 1e-4 and abs(sol[1] - s[1]) < 1e-5
                   for s in sols):
                continue
            sols.append(sol)
    rows = []
    for s in sols:
        _, _, fx, T = rhs(p, s[0], s[1], 0, 0, u, cfg)
        rows.append((trim_deg(p, s[1]), s[0], maxre(p, u, s, cfg), T["eps"],
                     T["ab"], fx))
    return sorted(rows)


def continuation(p, cfg, kns):
    guess = (0.0, 0.0)
    rows = []
    for kn in kns:
        u = kn * KN
        sol, r = solve(p, u, guess, cfg)
        if r > 1e-8:
            rows.append((kn, None))
            out(f"     {kn:5.2f}  branch lost (resid {r:.1e})")
            break
        guess = tuple(sol)
        _, _, fx, T = rhs(p, sol[0], sol[1], 0, 0, u, cfg)
        mr = maxre(p, u, sol, cfg)
        rows.append((kn, trim_deg(p, sol[1]), sol[0], mr, T["eps"],
                     T["FnT"], fx))
    return rows


def part1(p):
    out("1  BRANCHES UNDER THE as1 CLOSURES (calm, captive surge, complete "
        "D/Dt on unless stated)")
    cfgs = [
        ("v0 C_V blend + dmx (design part 2)", Cfg(ab="plant")),
        ("a_b = 1 (wet transom) + dmx", Cfg(ab="one")),
        ("a_b(eps), E band Fn_T 3.0-3.5 (Maki)", Cfg(ab="eps",
                                                    band=(3.0, 3.5))),
        ("a_b(eps), E band Fn_T 4.0-5.0 (Saunders)", Cfg(ab="eps",
                                                        band=(4.0, 5.0))),
        ("a_b(eps) Maki + face force", Cfg(ab="eps", band=(3.0, 3.5),
                                           face=True)),
        ("a_b(eps) Maki + face + J 0.5 x eps on m_a", Cfg(
            ab="eps", band=(3.0, 3.5), face=True, J=0.5)),
        ("a_b(eps) Saunders + face + J 0.5 x eps", Cfg(
            ab="eps", band=(4.0, 5.0), face=True, J=0.5)),
    ]
    kns = list(np.arange(1.0, 26.0, 0.5))
    for name, cfg in cfgs:
        out(f"   == {name}")
        out("     kn     trim   z_eq    maxRe   eps   Fn_T   R_captive(N)")
        rows = continuation(p, cfg, kns)
        prev = None
        for rw in rows:
            if rw[1] is None:
                break
            kn, tr, z, mr, e, F, fx = rw
            jump = "" if prev is None else (
                "  <-- JUMP" if abs(tr - prev) > 0.5 else "")
            if (kn % 1.0 == 0.0) or jump:
                out(f"     {kn:5.2f} {tr:6.2f} {z:+.4f} {mr:+7.3f}  {e:4.2f}"
                    f"  {F:5.2f}  {fx:8.1f}{jump}")
            prev = tr
        out("     all equilibria (multi-start) at selected speeds: "
            "trim / z / maxRe / eps")
        for kn in (6, 9, 10, 11, 12, 13, 14, 16, 20):
            eqs = enumerate_eq(p, kn * KN, cfg)
            s = "; ".join(f"{a:+6.2f}/{b:+.3f}/{c:+.2f}/{e:.2f}"
                          for a, b, c, e, _, _ in eqs)
            out(f"       {kn:3d} kn: {s}")
        out()


def part2(p):
    """Longitudinal identity at rest and at a trimmed attitude, term by
    term, discretised as the plant: bottom tilt, taper (yc'), last-strip
    forward face, transom face."""
    out("2  LONGITUDINAL HYDROSTATIC IDENTITY ON THE scarab GEOMETRY "
        "(stem_rise = 0, taper sqrt(1-s^2); calm, static)")
    SP = AU.p_SP(p)
    rg = p.rho * G
    for tr_add in (0.0, 2.0, 4.0):
        th = th_of_trim(p, trim_deg(p, 0.0) + tr_add)
        cfg = Cfg(ab="one", dmx=False)
        T = terms(p, 0.0, th, 0, 0, 0.0, cfg)
        d, A, tau = T["d"], T["A"], T["tau"]
        x, dx = p.x, p.dx
        bottom = -(rg * A * tau).sum() * dx
        # telescoped: -rho g Q_T,edge + rho g Q_bow,edge (exact for the
        # strip geometry if the edges are the hull ends)
        dT = d[0] + (d[0] - d[1]) * 0.5
        dB = d[-1] + (d[-1] - d[-2]) * 0.5
        QT = AU.q_first_moment(p, dT, p.yc[0])
        yb = 1e-4
        QB_strip = AU.q_first_moment(p, d[-1], p.yc[-1])
        out(f"   trim {trim_deg(p, th):5.2f} deg: bottom -rho g sum A tau dx "
            f"{bottom:+7.1f} N; face +rho g Q_T {rg * QT:+7.1f} N; last strip "
            f"open forward face -rho g Q(d_bow, yc_last) "
            f"{-rg * QB_strip:+7.1f} N (yc_last {p.yc[-1]:.3f} m)")
    out()


def main():
    h, p = AU.build(captive_u=None)
    part2(p)
    part1(p)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__" and "--time" not in sys.argv:
    main()


# ---------------------------------------------------------------- part 3
def sweep_time(p, cfg, kns, T=40.0, dt=0.005, start=(0.0, 0.0)):
    """Captive calm time runs (surge held), speed stepped through kns, each
    run starting from the previous end state: steady state or limit cycle.
    Returns rows (kn, trim_mean, trim_pp, z_mean, z_pp)."""
    st = np.array([start[0], start[1], 0.0, 0.0])
    rows = []

    def f(s, u):
        wd, qd, *_ = rhs(p, s[0], s[1], s[2], s[3], u, cfg)
        return np.array([s[2], s[3], wd, qd])
    for kn in kns:
        u = kn * KN
        n = int(T / dt)
        rec = []
        for i in range(n):
            k1 = f(st, u)
            k2 = f(st + 0.5 * dt * k1, u)
            k3 = f(st + 0.5 * dt * k2, u)
            k4 = f(st + dt * k3, u)
            st = st + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.all(np.isfinite(st)):
                break
            if i * dt > T - 10.0:
                rec.append((trim_deg(p, st[1]), st[0]))
        R = np.array(rec) if rec else np.full((1, 2), np.nan)
        rows.append((kn, R[:, 0].mean(), np.ptp(R[:, 0]), R[:, 1].mean(),
                     np.ptp(R[:, 1])))
    return rows


def part3(p):
    out("3  CAPTIVE CALM TIME RUNS (surge held, RK4 dt 5 ms, 40 s per speed,"
        " speed stepped up then down; last 10 s): steady or limit cycle?")
    cfgs = [
        ("v0 C_V blend + dmx", Cfg(ab="plant")),
        ("a_b = 1 + dmx", Cfg(ab="one")),
        ("a_b(eps) Maki + face", Cfg(ab="eps", band=(3.0, 3.5), face=True)),
        ("a_b(eps) Maki + face + J0.5", Cfg(ab="eps", band=(3.0, 3.5),
                                           face=True, J=0.5)),
        ("a_b(eps) Saunders + face + J0.5", Cfg(ab="eps", band=(4.0, 5.0),
                                               face=True, J=0.5)),
    ]
    up = [6, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14, 16]
    for name, cfg in cfgs:
        out(f"   == {name}")
        out("     dir   kn   trim_mean  trim_p-p   z_mean   z_p-p")
        r_up = sweep_time(p, cfg, up)
        for kn, tm, tp, zm, zp in r_up:
            flag = "  <-- LIMIT CYCLE" if tp > 0.05 else ""
            out(f"     up  {kn:5.1f}   {tm:7.2f}   {tp:7.3f}   {zm:+.4f}  "
                f"{zp:.4f}{flag}")
        last = r_up[-1]
        down = up[::-1][1:]
        # restart from the planing end state of the up sweep
        r_dn = sweep_time(p, cfg, down, start=(last[3], th_of_trim(
            p, last[1])))
        for kn, tm, tp, zm, zp in r_dn:
            flag = "  <-- LIMIT CYCLE" if tp > 0.05 else ""
            out(f"     dn  {kn:5.1f}   {tm:7.2f}   {tp:7.3f}   {zm:+.4f}  "
                f"{zp:.4f}{flag}")
        out()


if __name__ == "__main__" and "--time" in sys.argv:
    # python studies/anyspeed_review_physics.py --time  (about 30 min)
    _h, _p = AU.build(captive_u=None)
    LINES.clear()
    part3(_p)
    with open(OUT.replace(".txt", "_time.txt"), "w", encoding="utf-8") as _f:
        _f.write("\n".join(LINES) + "\n")
