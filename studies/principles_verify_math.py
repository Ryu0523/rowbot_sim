#!/usr/bin/env python3
"""
Independent re-derivation of the math claims C4-C10, C19, C23 of the
principles ledger (studies/principles_ledger_checks.py, checks.txt).

Deliberately different methods from the ledger:
  C4/C5  indirect SOURCE-panel method on the full double body in an
         unbounded fluid (no image, no Green's identity), plus the exact
         Fourier series (telescoping sum) for the semicircle.
  C6/C7  wavemaker A(K, d) by splitting the integrand into smooth x
         {1, cos, sin} parts and Fourier (QAWF) tails; Kramers-Kronig in the
         variable kappa = w^2 d/g with a Cauchy-weight (QAWC) quadrature.
         Michell integral in lambda = cosh(u).
  C8     R(T) in the frequency domain, (2/pi) int B (1 - cos wT)/w^2 dw / A_inf
  C9     Savitsky lambda from the centre-of-pressure balance, own code.
  C10    JONSWAP fractions on a wider, finer grid.
  C19    complex amplification factor and a direct Heun run.

Read-only w.r.t. the plant. Output: studies/_cache/principles/verify/verify_math.txt
Run: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m studies.principles_verify_math
"""
import os
import sys
import time

import numpy as np
from scipy import integrate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "studies", "_cache", "principles", "verify")
G, RHO = 9.81, 1025.0
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


# ---------------------------------------------------------- source panels
def source_panel_added_mass(P, vn_fun):
    """Full closed contour P (M+1, 2), CCW, P[-1] == P[0]. Constant-strength
    source panels, collocation at midpoints, exterior Neumann problem
    dphi/dn = vn (n out of the body). Returns m_full / rho with the
    normalisation  m_full = -rho sum phi vn ds  (for unit body speed)."""
    a, b = P[:-1], P[1:]
    h = np.linalg.norm(b - a, axis=1)
    t = (b - a) / h[:, None]
    n = np.stack([t[:, 1], -t[:, 0]], -1)           # outward for CCW
    m = 0.5 * (a + b)
    vn = vn_fun(m, n)
    # local coordinates of every collocation point in every panel's frame
    r = m[:, None, :] - a[None, :, :]               # from panel start
    X = (r * t[None]).sum(-1)                       # along panel j
    Y = (r * n[None]).sum(-1)                       # normal to panel j
    x1, x2 = 0.0, h[None, :]
    u1, u2 = x1 - X, x2 - X
    r1s, r2s = u1 ** 2 + Y ** 2, u2 ** 2 + Y ** 2
    # velocity of int ln r ds in panel-local frame
    with np.errstate(divide="ignore", invalid="ignore"):
        vx = 0.5 * np.log(r1s / r2s)                # d/dX
    # subtended angle int Y/((s-X)^2+Y^2) ds = atan(u2/Y) - atan(u1/Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        Ys = np.where(np.abs(Y) < 1e-15, 1e-300, Y)
        vy = np.arctan(u2 / Ys) - np.arctan(u1 / Ys)
    N = len(h)
    idx = np.arange(N)
    vy[idx, idx] = np.pi          # self: limit from the fluid side (Y -> 0+)
    vx[idx, idx] = 0.0
    # global velocity, dotted with n_i
    vgx = vx * t[None, :, 0] + vy * n[None, :, 0]
    vgy = vx * t[None, :, 1] + vy * n[None, :, 1]
    Dn = (vgx * n[:, None, 0] + vgy * n[:, None, 1]) / (2 * np.pi)
    sig = np.linalg.solve(Dn, vn)
    # potential of int ln r ds: G(u) = u/2 ln(u^2+Y^2) - u + Y atan(u/Y)

    def Gf(u, Y_):
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.where(u * u + Y_ * Y_ > 0, np.log(u * u + Y_ * Y_), 0.0)
            at = np.where(np.abs(Y_) > 1e-15,
                          Y_ * np.arctan(u / np.where(np.abs(Y_) > 1e-15, Y_,
                                                       1.0)), 0.0)
        return 0.5 * u * lg - u + at
    Phi = (Gf(u2, Y) - Gf(u1, Y)) / (2 * np.pi)
    phi = Phi @ sig
    return float(-(phi * vn * h).sum())


def clustered(n):
    return 0.5 * (1 - np.cos(np.linspace(0, np.pi, n + 1)))


def polygon(verts, n_per):
    """Closed CCW polygon through verts (list), cosine-clustered panels."""
    pts = []
    V = list(verts) + [verts[0]]
    for p0, p1 in zip(V[:-1], V[1:]):
        s = clustered(n_per)[:-1]
        pts.append(np.array(p0)[None, :] + (np.array(p1) - np.array(p0))[
            None, :] * s[:, None])
    pts.append(np.array(V[-1])[None, :])
    return np.concatenate(pts)


def ellipse(ah, bv, n_half):
    th = np.concatenate([np.pi * clustered(n_half)[:-1],
                         np.pi + np.pi * clustered(n_half)])
    return np.stack([ah * np.cos(th), bv * np.sin(th)], -1)


VN = {
    "sway_lid": lambda m, n: n[:, 0],
    "sway_phi0": lambda m, n: n[:, 0] * np.where(m[:, 1] < 0, 1.0, -1.0),
    "heave_phi0": lambda m, n: n[:, 1],
}


def c4():
    say("=" * 78)
    say("C4  sway added mass with phi=0 on the free surface")
    # exact Fourier series, semicircle: m/(rho R^2) = (pi/2) sum beta_n^2/n
    k = np.arange(1, 2_000_001, dtype=float)
    S = (16 / np.pi ** 2) * np.sum(2 * k / (4 * k * k - 1) ** 2)
    say(f"   Fourier: sum beta_n^2/n = {S:.12f}; 4/pi^2 = {4 / np.pi ** 2:.12f}"
        f"  (telescoping: k/(4k^2-1)^2 = [1/(2k-1)^2 - 1/(2k+1)^2]/8)")
    say(f"   => m22_phi0 = (pi/2) rho R^2 x 4/pi^2 = {np.pi / 2 * S:.10f} rho R^2;"
        f" 2/pi = {2 / np.pi:.10f}")
    say("   conformal map ellipse -> circle keeps theta and the flux "
        "dpsi = U dz = U b cos(theta) dtheta, so every semi-ellipse of draught "
        "b has the semicircle(b) value")
    say("   source-panel, full double body, unbounded fluid (m_phys = m_full/2):")
    say("     a/b    lid/(pi/2 b^2)   phi0/(2/pi b^2)   phi0/lid   n_half")
    for ab in [0.1, 0.4, 1.0, 3.0]:
        for nh in [200, 400]:
            P = ellipse(ab, 1.0, nh)
            ml = source_panel_added_mass(P, VN["sway_lid"]) / 2
            mp = source_panel_added_mass(P, VN["sway_phi0"]) / 2
            say(f"     {ab:4.1f}   {ml / (np.pi / 2):.6f}        "
                f"{mp / (2 / np.pi):.6f}         {mp / ml:.6f}   {nh}")
    say(f"     4/pi^2 = {4 / np.pi ** 2:.6f}")


def c5(plant):
    say("=" * 78)
    say("C5  heave added mass, phi=0, about a FIXED waterline vs Wagner's c")
    tb = np.tan(np.radians(20.0))
    for nper in [100, 200, 400]:
        P = polygon([(1.0, 0.0), (0.0, tb), (-1.0, 0.0), (0.0, -tb)], nper)
        m = source_panel_added_mass(P, VN["heave_phi0"]) / 2
        say(f"   20 deg wedge, {nper:4d} panels/edge: m33_phi0 = {m:.5f} rho "
            f"c_g^2 = {m / (np.pi / 2):.5f} x (pi/2) rho c_g^2;  "
            f"(pi/2)^3/m = {(np.pi / 2) ** 3 / m:.4f}")
    # flat plate check (b -> 0): pi/2 rho c^2
    P = polygon([(1.0, 0.0), (0.0, 1e-4), (-1.0, 0.0), (0.0, -1e-4)], 400)
    say(f"   check, flat plate (rhombus 1e-4 thick): "
        f"{source_panel_added_mass(P, VN['heave_phi0']) / 2 / (np.pi / 2):.5f}"
        f" x (pi/2) rho c^2 (exact 1)")
    # scarab at rest
    from sim.planing_vessel import SP
    p = plant
    d = -(p.Z_rest + p.zk + SP * p.x * p.Th_rest)
    A_, c_, ma_, _, _ = p._section(d)
    tbp = p.tanb
    tot = 0.0
    for i in range(p.x.size):
        di = d[i]
        if di <= 1e-6:
            continue
        yc = p.yc[i]
        dch = yc * tbp
        if di <= dch:
            c = di / tbp
            verts = [(c, 0.0), (0.0, di), (-c, 0.0), (0.0, -di)]
        else:
            e = di - dch
            verts = [(yc, 0.0), (yc, e), (0.0, di), (-yc, e), (-yc, 0.0),
                     (-yc, -e), (0.0, -di), (yc, -e)]
        P = polygon(verts, 60)
        tot += source_panel_added_mass(P, VN["heave_phi0"]) / 2
    ex = tot * p.rho * p.dx
    v0 = ma_.sum() * p.dx
    say(f"   scarab195 at rest: v0 sum k_a pi/2 rho c^2 dx = {v0:.0f} kg "
        f"(k_a = {p.k_a}); exact 2D phi0 sections (source panels) = "
        f"{ex:.0f} kg; ratio {v0 / ex:.3f}")


# ---------------------------------------------------------- wavemaker
def wm_A_over_coef(kap):
    """int_0^inf S^2/(s (s^2+kap^2)) ds, S = sin s - (kap/s)(1-cos s);
    A = (2 rho d^2/pi) * this.  [0, s0] direct; tail by QAWF."""
    s0 = 30.0 + 0.0 * kap

    def f(s):
        S = np.sin(s) - kap / s * (1 - np.cos(s))
        return S * S / (s * (s * s + kap * kap))
    edges = np.linspace(0.0, s0, 61)
    head = sum(integrate.quad(f, lo, hi, epsabs=0, epsrel=1e-12, limit=200)[0]
               for lo, hi in zip(edges[:-1], edges[1:]))
    # S^2 = sin^2 - 2 kap sin(1-cos)/s + kap^2 (1-cos)^2/s^2
    #     = 1/2 - cos2s/2 - (2kap/s)(sin s - sin2s/2)
    #       + (kap^2/s^2)(3/2 - 2cos s + cos2s/2)
    w = lambda s: 1.0 / (s * (s * s + kap * kap))
    q = lambda g, wt, wv: integrate.quad(g, s0, np.inf, weight=wt, wvar=wv,
                                         limlst=200)[0]
    tail = integrate.quad(lambda s: w(s) * (0.5 + 1.5 * kap * kap / s ** 2),
                          s0, np.inf, epsrel=1e-13)[0]
    tail += q(lambda s: -0.5 * w(s) + 0.5 * kap * kap / s ** 2 * w(s), "cos",
              2.0)
    tail += q(lambda s: -2 * kap * kap / s ** 2 * w(s), "cos", 1.0)
    tail += q(lambda s: -2 * kap / s * w(s), "sin", 1.0)
    tail += q(lambda s: kap / s * w(s), "sin", 2.0)
    return head + tail


def wm_KK(kap):
    """A/A_inf - 1 from B alone: PV int h(k)/(k - kap) dk / ln 2,
    h = (1 - e^{-k})^2 / k^2  (B(w) = 2 rho w (1-e^{-Kd})^2/K^2)."""
    h = lambda k: ((-np.expm1(-k)) / k) ** 2 if k > 0 else 1.0
    M = 50.0 * max(kap, 1.0)
    pv = integrate.quad(h, 0.0, M, weight="cauchy", wvar=kap, limit=400,
                        epsrel=1e-12)[0]
    pv += integrate.quad(lambda k: h(k) / (k - kap), M, np.inf,
                         epsrel=1e-12)[0]
    return pv / np.log(2.0)


def c6_c7_wavemaker():
    say("=" * 78)
    say("C7a wavemaker: A/A_inf - 1 (direct eigenfunction sum) vs "
        "Kramers-Kronig from B (PV in kappa)")
    ln2 = np.log(2.0)
    # A_inf integral
    Iinf = integrate.quad(lambda s: (1 - np.cos(s)) ** 2 / s ** 3, 0, 30,
                          limit=400, epsrel=1e-13)[0]
    Iinf += integrate.quad(lambda s: 1.5 / s ** 3, 30, np.inf)[0]
    Iinf += integrate.quad(lambda s: -2 / s ** 3, 30, np.inf, weight="cos",
                           wvar=1.0)[0]
    Iinf += integrate.quad(lambda s: 0.5 / s ** 3, 30, np.inf, weight="cos",
                           wvar=2.0)[0]
    say(f"   int (1-cos s)^2/s^3 ds = {Iinf:.12f}, ln 2 = {ln2:.12f}")
    say("    Kd      A/A_inf    (A/Ainf-1)Kd/2   KK: A/Ainf-1      rel diff")
    for kap in [0.3, 1.0, 3.0, 10.0, 100.0, 1000.0]:
        r = wm_A_over_coef(kap) / ln2
        kk = wm_KK(kap)
        say(f"   {kap:7.1f}  {r:9.6f}   {(r - 1) * kap / 2:+9.5f}       "
            f"{kk:+.10f}   {(kk - (r - 1)) / abs(r - 1):+.2e}")
    say("   analytic: (A-A_inf)/A_inf -> -(2/kap) int sin s (1-cos s)/s^2 ds "
        "/ ln2 = -(2/kap) [Frullani int (cos s - cos 2s)/s = ln 2]/ln 2 = -2/kap")

    say("=" * 78)
    say("C6  additive composite error on the wavemaker with draught change "
        "(K fixed, d0 -> d = r d0)")
    say("    composite = A_inf(d) + A(K,d0) - A_inf(d0); error/A_inf(d); "
        "large-Kd estimate 2(r-1)/(K d0 r^2)")
    say("    K d0     r=0.7               r=1.3               r=1.6")
    for kd0 in [1.0, 3.0, 10.0, 30.0]:
        row = []
        a0 = wm_A_over_coef(kd0)                  # in units 2 rho d0^2/pi
        for r in [0.7, 1.3, 1.6]:
            tru = wm_A_over_coef(kd0 * r) * r * r
            comp = ln2 * r * r + a0 - ln2
            err = (comp - tru) / (ln2 * r * r)
            est = 2 * (r - 1) / (kd0 * r * r)
            row.append(f"{err:+8.4f} (est {est:+7.4f})")
        say(f"   {kd0:6.1f}   " + "   ".join(row))


# ---------------------------------------------------------- Michell
def michell(U, L, B, T, rho=RHO, g=G, n=400001):
    """Michell (1898): R = (4 rho g^2/(pi U^2)) int_1^inf |I|^2
    lam^2/sqrt(lam^2-1) dlam, I = int int f_x e^{k0 lam^2 z + i k0 lam x},
    in lam = cosh(u) (measure cosh^2 u du), dense trapezoid, vectorised."""
    k0 = g / U ** 2
    a = L / 2
    u_hi = np.arccosh(max(2.0, 80.0 / (k0 * a), 40.0 / np.sqrt(k0 * T)))
    u = np.linspace(0.0, u_hi, n)
    lam = np.cosh(u)
    mu = k0 * lam
    q = k0 * lam * lam
    # x: |int_{-a}^{a} (-2x/a^2) e^{i mu x} dx| = (4/a^2)(sin(mu a)/mu^2 - a cos(mu a)/mu)
    X = np.where(mu * a < 1e-3, (4 / a ** 2) * (mu * a ** 3 / 3),
                 (4 / a ** 2) * (np.sin(mu * a) / mu ** 2 - a * np.cos(mu * a) / mu))
    # z: int_{-T}^0 (1 - z^2/T^2) e^{q z} dz; series for small qT
    x = q * T
    e = np.exp(-x)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Zc = T * ((1 - e) / x - (2 / x ** 3 - e * (1 / x + 2 / x ** 2 + 2 / x ** 3)))
    Zs = T * (2 / 3 - x / 4 + x * x / 15 - x ** 3 / 72)
    Z = np.where(x < 2e-2, Zs, Zc)
    I = (B / 2) * X * Z
    return 4 * rho * g * g / (np.pi * U * U) * np.trapezoid(I * I * lam * lam, u)


def c7_c23_michell():
    say("=" * 78)
    say("C7b Michell, classic Wigley (L 1, B 0.1, T 0.0625): "
        "C = R 2 pi U^2/(rho g^2 A_wp^2) - ln(Fn^2 T/L)")
    L, B, T = 1.0, 0.1, 0.0625
    Awp = 2 * B * L / 3
    Cs = []
    for Fn in [0.5, 1.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
        U = Fn * np.sqrt(G * L)
        R = michell(U, L, B, T)
        C = R * 2 * np.pi * U * U / (RHO * G * G * Awp ** 2) - np.log(
            Fn * Fn * T / L)
        Cs.append((Fn, C))
        S_w = 0.1488   # not used for C
        say(f"   Fn {Fn:6.1f}   R/(rho g vol) {R / (RHO * G * 4 * L * B * T / 9):.4e}"
            f"   C {C:+.5f}")
    (f1, c1), (f2, c2) = Cs[-2], Cs[-1]
    say(f"   Richardson in 1/Fn^2 from Fn {f1:.0f},{f2:.0f}: C_inf = "
        f"{(c2 * f2 ** 2 - c1 * f1 ** 2) / (f2 ** 2 - f1 ** 2):.5f}")
    say("   C_w at Fn 0.5 for comparison with the ledger (S = 0.1488 L^2): "
        f"{michell(0.5 * np.sqrt(G), L, B, T) / (0.5 * RHO * G * 0.25 * 0.1488):.4e}")
    say("=" * 78)
    say("C23 Michell on the scarab-proportioned Wigley (L 5.4, B 2.025, T 0.385)")
    L, B, T = 5.4, 2.025, 0.385
    vol = 4 * L * B * T / 9
    for kn in [14, 20, 25, 35]:
        U = kn * 0.514444
        R = michell(U, L, B, T)
        say(f"   {kn:3d} kn  Fn_L {U / np.sqrt(G * L):.3f}  R_w {R:7.0f} N  "
            f"R_w/(rho g vol) {R / (RHO * G * vol):.4f}  R_w/(boat W 1400 kg) "
            f"{R / (1400 * G):.4f}")
    say(f"   v0 R_plant at 25 kn (audit_transom.txt) 2139 N -> R/W = "
        f"{2139 / (1400 * G):.4f}; Savitsky 2408 N -> {2408 / (1400 * G):.4f}")


# ---------------------------------------------------------- C8
def c8():
    say("=" * 78)
    say("C8  memory/inertia impulse ratio, frequency-domain form "
        "R(T) = (2/pi) int B (1-cos wT)/w^2 dw / A_inf")
    from hydro.retardation import taper
    z = np.load(os.path.join(HERE, "studies", "_cache", "anyspeed",
                             "bem_probe_lid_24_64.npz"))
    om, A33, B33 = z["omega"], z["A"][:, 2, 2], z["B"][:, 2, 2]
    L = 5.4
    for cut, Ainf in [(15.8, 4660.0), (12.4, 4757.0)]:
        k = om <= cut + 1e-9
        w, Bk = om[k], B33[k] * taper(om[k])
        wf = np.linspace(w[0], w[-1], 20001)
        Bf = np.interp(wf, w, Bk)
        # the ledger's trapezoid starts at w[0] = 0.5 (no 0..0.5 segment)
        def R(T, with_low=False):
            ww, bb = wf, Bf
            if with_low:
                w0 = np.linspace(1e-6, w[0], 2001)
                ww = np.concatenate([w0, wf[1:]])
                bb = np.concatenate([Bk[0] * w0 / w[0], Bf[1:]])
            return 2 / np.pi * np.trapezoid(bb * (1 - np.cos(ww * T)) / ww ** 2,
                                            ww) / Ainf
        out = []
        for kn in [10, 20, 30, 35]:
            T = L / (kn * 0.514444)
            out.append(f"{kn} kn {R(T):.3f}/{R(T, True):.3f}")
        say(f"   cut {cut}: A_inf {Ainf:.0f} (piecewise-linear B; without/with "
            f"a linear 0-0.5 rad/s segment): " + ", ".join(out))
        # speed where R = 0.05 and length sensitivity at 35 kn
        Ts = np.linspace(0.02, 0.5, 2000)
        Rs = np.array([R(T) for T in Ts])
        T05 = Ts[np.argmax(Rs >= 0.05)]
        say(f"      R = 0.05 at T = {T05:.3f} s -> {L / T05 / 0.514444:.0f} kn "
            f"(5.4 m), {2.5 / T05 / 0.514444:.0f} kn (2.5 m)")
        U35 = 35 * 0.514444
        for Lw in [5.4, 3.0, 2.5]:
            say(f"      35 kn, length {Lw} m: T {Lw / U35:.3f} s, R {R(Lw / U35):.3f}")
    # tail bound beyond the cut if B stayed at its last value
    say("   tail bound if B stayed at 5000 N s/m above 15.8 rad/s: "
        f"(2/pi) 5000 int_15.8^inf 2/w^2 dw / 4660 = "
        f"{2 / np.pi * 5000 * 2 / 15.8 / 4660:.3f} (absolute R units, upper bound)")


# ---------------------------------------------------------- C9
def c9(plant):
    say("=" * 78)
    say("C9  steady phi=0 parameter g L_k^2/(U^2 c) on the v0 branch")
    tab = {9: (2.41, 0.513), 14: (5.78, 0.427), 20: (3.51, 0.301),
           25: (2.36, 0.253), 35: (1.54, 0.221)}
    tb = np.tan(np.radians(20.0))
    yT = float(plant.yc[0])
    b = 0.83 * 2.44
    lcg = 0.38 * 5.74
    from scipy.optimize import brentq
    for kn, (trim, dT) in tab.items():
        U = kn * 0.514444
        Lk = min(5.4, dT / np.tan(np.radians(trim)))
        c = min(yT, 0.5 * np.pi * dT / tb)
        cv = U / np.sqrt(G * b)
        lam = brentq(lambda l: l * (0.75 - 1 / (5.21 * cv ** 2 / l ** 2 + 2.39))
                     - lcg / b, 0.05, 20)
        say(f"   {kn:3d} kn  L_k {Lk:.2f}  c {c:.3f}  param {G * Lk ** 2 / (U * U * c):6.2f}"
            f"   Savitsky lam {lam:.3f}  2 lam^2/C_V^2 {2 * lam ** 2 / cv ** 2:.3f}")
    # Fridsma
    from studies.exp_planing import fridsma_hull
    for case in "AB":
        hf, V = fridsma_hull(case)
        cv = V / np.sqrt(G * hf.b)
        lam = brentq(lambda l: l * (0.75 - 1 / (5.21 * cv ** 2 / l ** 2 + 2.39))
                     - hf.lcg / hf.b, 0.05, 20)
        say(f"   Fridsma {case}: C_V {cv:.3f}, lam {lam:.3f}, 2 lam^2/C_V^2 "
            f"{2 * lam ** 2 / cv ** 2:.3f}")


# ---------------------------------------------------------- C10
def c10():
    say("=" * 78)
    say("C10 JONSWAP (Hs 1, Tp 5, gamma 3.3): variance share with "
        "(w_e^2/g)(b/2) < 1, b = 2.025 m; grid 0.05-15 rad/s")
    w = np.linspace(0.05, 15.0, 300001)
    wp = 2 * np.pi / 5.0
    sig = np.where(w <= wp, 0.07, 0.09)
    S = w ** -5 * np.exp(-1.25 * (wp / w) ** 4) * 3.3 ** np.exp(
        -(w - wp) ** 2 / (2 * sig ** 2 * wp ** 2))
    k = w * w / G
    b = 0.83 * 2.44
    tot = np.trapezoid(S, w)
    say("    kn    mu=0     45     90    135    180")
    for kn in [0, 15, 25, 35]:
        U = kn * 0.514444
        row = []
        for mu in [0, 45, 90, 135, 180]:
            we = np.abs(w - k * U * np.cos(np.radians(mu)))
            row.append(np.trapezoid(S * (we * we / G * b / 2 < 1), w) / tot)
        say(f"   {kn:4d}  " + "  ".join(f"{v:5.3f}" for v in row))


# ---------------------------------------------------------- C19
def c19():
    say("=" * 78)
    say("C19 Heun amplification on x'' = -w^2 x")
    for wn, dt in [(3.94, 0.02), (4.9, 0.02)]:
        zz = 1j * wn * dt
        Rz = 1 + zz + zz * zz / 2
        n = int(round(100 / dt))
        say(f"   w {wn} dt {dt}: |R| = {abs(Rz):.10f}, sqrt(1+(w dt)^4/4) = "
            f"{np.sqrt(1 + (wn * dt) ** 4 / 4):.10f}; energy x {abs(Rz) ** (2 * n):.5f}"
            f" per 100 s")


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    say(f"principles_verify_math  {time.strftime('%Y-%m-%d %H:%M')}")
    c4()
    from hydro import hulls
    from sim.planing_vessel import PlaningVessel
    from sim.test_vessel import Monochromatic
    plant = PlaningVessel(hulls.get("scarab195"), Monochromatic(1.0, 0.0),
                          dt=0.02)
    c5(plant)
    c6_c7_wavemaker()
    c7_c23_michell()
    c8()
    c9(plant)
    c10()
    c19()
    say(f"(total {time.time() - t0:.0f} s)")
    with open(os.path.join(OUT, "verify_math.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
