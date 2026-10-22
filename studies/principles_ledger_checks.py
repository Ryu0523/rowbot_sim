#!/usr/bin/env python3
"""
Math checks behind the mechanism ledger of the planing plant (v0) and the
any-speed design (as1, learn/meta/anyspeed_design_en.md).

Read-only with respect to the plant: it imports the v0 plant and the hull
table, never edits them. Output: studies/_cache/principles/ledger/checks.txt

PARTS
  A  An exactly solvable linear radiation problem WITH gravity (Havelock's
     deep-water piston wavemaker): does [theory with g] - [its phi=0 limit]
     vanish at high frequency, at what rate, and does Kramers-Kronig (the
     identity behind Cummins with one A_inf) hold?
  B  2D boundary-element added masses with a free-surface image (phi=0 or
     rigid lid), validated on circle/ellipse; then (i) heave phi=0 of a 20 deg
     V section vs v0's Wagner formula, (ii) sway phi=0 vs rigid lid (v0's
     m22 = pi/2 rho d^2 is rigid-lid), (iii) scarab per-station sums at rest
     and at 25 kn, against v0 and the 3D BEM A_inf.
  C  Michell thin-ship wave resistance vs Froude number: R_w -> 0 as
     Fn -> inf, the asymptotic law, and the size in the 0.3-1.0 band.
  D  The divergence-theorem identity behind 'Xw is a second copy': exact 3D
     pressure quadrature on the scarab geometry in a long wave.
  E  Where is gravity NOT small? steady free-surface parameter on the v0
     branch; encounter-frequency spectrum by heading in the task sea.
  F  The additive composite [nonlinear phi=0] + [linear g] - [linear phi=0]:
     its error is the mixed term; measured on the exact problem of part A.
  G  Heun's amplitude factor (the integrator).

Run: .venv/Scripts/python.exe -m studies.principles_ledger_checks
"""
import os
import sys
import time

import numpy as np
from scipy import integrate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "studies", "_cache", "principles", "ledger")
G = 9.81
RHO = 1025.0
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


# =====================================================================  A
def wavemaker_A(K, d, rho=RHO):
    """Added mass per unit width of a deep-water piston of draught d on a
    wall (fluid on one side, wall rigid below -d), free-surface wavenumber
    K = w^2/g. Havelock's expansion: progressive e^{Kz} plus evanescent
    psi_k = k cos kz + K sin kz, norm (pi/2)(k^2+K^2):
        A = (2 rho/pi) int_0^inf [sin kd - (K/k)(1-cos kd)]^2/(k(k^2+K^2)) dk
    """
    f = lambda s: ((np.sin(s) - (K * d / s) * (1 - np.cos(s))) ** 2
                   / (s * (s * s + (K * d) ** 2)))
    # in s = k d; integrate piecewise (oscillatory)
    edges = np.concatenate([[0.0], np.pi * np.arange(1, 400)])
    tot = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        tot += integrate.quad(f, a, b, limit=200, epsabs=0, epsrel=1e-11)[0]
    # tail beyond s = 399 pi: integrand ~ [..]^2 / (s (s^2+(Kd)^2)), bounded
    # by (1 + 2Kd/s)^2/s^3 -> add its average (sin^2 ~ 1/2 etc.) analytically
    s0 = edges[-1]
    kd = K * d
    tail = integrate.quad(lambda s: (0.5 + 1.5 * (kd / s) ** 2)
                          / (s * (s * s + kd * kd)), s0, np.inf)[0]
    return 2 * rho / np.pi * d * d * (tot + tail)


def wavemaker_B(w, d, rho=RHO, g=G):
    K = w * w / g
    return 2 * rho * w * (1 - np.exp(-K * d)) ** 2 / K ** 2


def part_A():
    say("=" * 78)
    say("A  EXACT LINEAR RADIATION WITH GRAVITY: deep-water piston wavemaker "
        "(Havelock 1929; Wehausen & Laitone 1960 sec. 21)")
    d = 1.0
    i_ln2 = integrate.quad(lambda s: (1 - np.cos(s)) ** 2 / s ** 3, 0, 50,
                           limit=500)[0] + integrate.quad(
        lambda s: (1.5 - 2 * np.cos(s) + 0.5 * np.cos(2 * s)) / s ** 3, 50,
        np.inf, limit=500)[0]
    A_inf = 2 * RHO / np.pi * d * d * np.log(2.0)
    say(f"   phi=0 limit solved directly (sine transform): A_inf = (2/pi) rho "
        f"d^2 int(1-cos s)^2/s^3 ds;  int = {i_ln2:.10f}, ln 2 = "
        f"{np.log(2):.10f}")
    say("   K d     A/A_inf   (A-A_inf)/A_inf   x (Kd)/2 [-> -1]   "
        "B/(w A_inf)   x (Kd)^2 ln2/pi [-> 1]")
    rows = []
    for kd in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        K = kd / d
        w = np.sqrt(K * G)
        A = wavemaker_A(K, d)
        B = wavemaker_B(w, d)
        r = (A - A_inf) / A_inf
        rows.append((kd, A, B))
        say(f"   {kd:7.1f}  {A / A_inf:8.4f}   {r:+12.4e}     {r * kd / 2:+8.4f}"
            f"           {B / (w * A_inf):10.3e}    "
            f"{B / (w * A_inf) * kd * kd * np.log(2) / np.pi:8.4f}")
    say("   derived: (A - A_inf)/A_inf = -2/(Kd) + O(ln(Kd)/(Kd)^2)  "
        "[expansion of the integrand; Frullani gives the ln 2],")
    say("            B/(w A_inf) = pi/(ln2 (Kd)^2) (1 - e^{-Kd})^2 -> both "
        "gravity effects vanish as Kd = w^2 d/g -> inf.")
    # Kramers-Kronig: A(w) - A_inf = (2/pi) PV int_0^inf B(s)/(s^2 - w^2) ds
    say("   Kramers-Kronig check (the relation Cummins needs: A_inf from the "
        "same theory as B):")
    say("     K d    A - A_inf (direct)   (2/pi) PV int B/(s^2-w^2)   rel diff")
    for kd in [0.3, 1.0, 3.0, 10.0]:
        K = kd / d
        w = np.sqrt(K * G)
        Bw = wavemaker_B(w, d)
        S = 400.0 * w

        def fint(s):
            if abs(s - w) < 1e-9 * w:
                # derivative limit of (B(s)-B(w))/(s^2-w^2) at s=w
                h = 1e-6 * w
                return (wavemaker_B(w + h, d) - wavemaker_B(w - h, d)) / (
                    2 * h) / (2 * w)
            return (wavemaker_B(s, d) - Bw) / (s * s - w * w)
        pts = np.concatenate([np.linspace(1e-9, 3 * w, 61)[1:],
                              np.geomspace(3 * w, S, 120)[1:]])
        tot = 0.0
        lo = 1e-12
        for hi in pts:
            tot += integrate.quad(fint, lo, hi, limit=200, epsrel=1e-11,
                                  points=[w] if lo < w < hi else None)[0]
            lo = hi
        tot += Bw * np.log((S - w) / (S + w)) / (2 * w)   # PV int_0^S 1/(s^2-w^2)
        tot += integrate.quad(lambda s: wavemaker_B(s, d) / (s * s - w * w),
                              S, np.inf)[0]
        kk = 2 / np.pi * tot
        direct = wavemaker_A(K, d) - A_inf
        say(f"     {kd:5.1f}   {direct:14.6f}       {kk:14.6f}           "
            f"{(kk - direct) / abs(direct):+.2e}")
    return rows


# =====================================================================  B
def panel_added_mass(Y, Z, mode, fs):
    """2D added mass (per unit length, physical half only) of the wetted
    contour (Y, Z) (z <= 0, ordered from the +y waterline round the keel to
    the -y waterline) with the free surface z = 0 as an image plane:
    fs = 'phi0' (phi = 0, negative image) or 'lid' (rigid lid, positive
    image). Green's third identity with constant phi and dphi/dn per panel,
    exact panel integrals. mode 'heave' or 'sway'. Returns m / rho."""
    P = np.stack([Y, Z], -1)
    a, b = P[:-1], P[1:]
    h = np.linalg.norm(b - a, axis=1)
    t = (b - a) / h[:, None]
    n = np.stack([t[:, 1], -t[:, 0]], -1)        # into the body (see notes)
    m = 0.5 * (a + b)
    sgn = -1.0 if fs == "phi0" else 1.0
    N = len(h)

    def infl(cen, tt, nn, hh):
        # field points m (N,2) against panels (centre cen, tangent tt,
        # normal nn, length hh), all (N,...) -> S, D matrices (N, N)
        r = m[:, None, :] - cen[None, :, :]
        X = (r * tt[None]).sum(-1)
        Yl = (r * nn[None]).sum(-1)
        hp = 0.5 * hh[None, :]

        def F(s):
            rs2 = (s - X) ** 2 + Yl ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = 0.5 * np.log(np.where(rs2 > 0, rs2, 1.0))
                at = np.where(np.abs(Yl) > 1e-14,
                              Yl * np.arctan((s - X) / np.where(
                                  np.abs(Yl) > 1e-14, Yl, 1.0)), 0.0)
            return (s - X) * lr - s + at
        S = (F(hp) - F(-hp)) / (2 * np.pi)
        # d/dn_xi ln r = -Yl/r^2, integrated over the panel; 0 on the panel
        # itself (principal value; the 1/2 jump is explicit)
        D = -_angle(hp, X, Yl) / (2 * np.pi)
        return S, D

    S0, D0 = infl(m, t, n, h)
    # image panel: reflected centre, tangent, normal (z -> -z)
    R = np.array([1.0, -1.0])
    Si, Di = infl(m * R, t * R, n * R, h)
    S = S0 + sgn * Si
    D = D0 + sgn * Di
    k = 1 if mode == "heave" else 0
    q = n[:, k]                                   # dphi/dn, n into body
    Amat = 0.5 * np.eye(N) - D
    phi = np.linalg.solve(Amat, -S @ q)
    return float((phi * n[:, k] * h).sum())


def _angle(hp, X, Yl):
    """int_{-hp}^{hp} Yl / ((s-X)^2 + Yl^2) ds, continuous, 0 at Yl = 0
    inside the panel (principal value)."""
    Ys = np.where(np.abs(Yl) < 1e-14, 1.0, Yl)
    v = np.arctan((hp - X) / Ys) - np.arctan((-hp - X) / Ys)
    return np.where(np.abs(Yl) < 1e-14, 0.0, v)


def _cluster(n):
    """n+1 points on [0,1] clustered at both ends (cosine spacing)."""
    return 0.5 * (1 - np.cos(np.linspace(0, np.pi, n + 1)))


def contour_polyline(pts, n_per):
    """Densify a polyline (list of (y,z) vertices) with cosine-clustered
    panels on each edge."""
    Y, Z = [], []
    for (y0, z0), (y1, z1) in zip(pts[:-1], pts[1:]):
        s = _cluster(n_per)
        Y.append(y0 + (y1 - y0) * s[:-1])
        Z.append(z0 + (z1 - z0) * s[:-1])
    Y.append([pts[-1][0]])
    Z.append([pts[-1][1]])
    return np.concatenate(Y), np.concatenate(Z)


def section_contour(d, yc, tanb, n_per=60):
    """v0 section (V bottom to the chine at yc, vertical sides above) at keel
    immersion d, as a wetted contour from the +y waterline to the -y one."""
    dch = yc * tanb
    if d <= dch:
        c = d / tanb
        pts = [(c, 0.0), (0.0, -d), (-c, 0.0)]
    else:
        pts = [(yc, 0.0), (yc, -(d - dch)), (0.0, -d), (-yc, -(d - dch)),
               (-yc, 0.0)]
    return contour_polyline(pts, n_per)


def plate_sway_ratio_chebyshev(nmax=4000):
    """Sway added mass of a surface-piercing vertical plate, phi=0 over
    rigid lid, by the flat-plate Chebyshev series (double body = plate of
    half-length d with normal velocity sgn(z) for phi=0, uniform for the
    lid). Jump 2a sin(n th)/n <-> normal velocity U_{n-1}; m = rho pi a^2
    sum c_n^2/n, c_n = (2/pi) int_0^pi v(cos th) sin th sin(n th) dth."""
    th = np.linspace(0, np.pi, 200001)
    v = np.sign(np.cos(th))
    tot = 0.0
    for nn in range(1, nmax + 1):
        cn = 2 / np.pi * np.trapezoid(v * np.sin(th) * np.sin(nn * th), th)
        tot += cn * cn / nn
    return tot


def part_B(plant):
    say("=" * 78)
    say("B  2D ADDED MASS WITH A FREE-SURFACE IMAGE (BEM, Green's identity, "
        "constant panels, exact panel integrals)")
    # validation
    say("   validation (exact: half of the double body's added mass)")
    a = 1.0
    th = np.pi * _cluster(240)
    Yc, Zc = a * np.cos(th), -a * np.sin(th)
    say("     (rigid-lid HEAVE of a surface-piercing 2D body is ill-posed: a "
        "net volume flux, log-divergent as in part A at K->0; not used)")
    for mode, fs, ex in [("heave", "phi0", np.pi / 2),
                         ("sway", "lid", np.pi / 2)]:
        mval = panel_added_mass(Yc, Zc, mode, fs)
        say(f"     semicircle a=1  {mode:5s} {fs:4s}: {mval:.5f} rho a^2   "
            f"exact {ex:.5f}   rel {mval / ex - 1:+.2e}")
    ae, be = 1.0, 0.4
    Ye, Ze = ae * np.cos(th), -be * np.sin(th)
    for mode, fs, ex in [("heave", "phi0", np.pi / 2 * ae ** 2),
                         ("sway", "lid", np.pi / 2 * be ** 2)]:
        mval = panel_added_mass(Ye, Ze, mode, fs)
        say(f"     half-ellipse a=1 b=0.4 {mode:5s} {fs:4s}: {mval:.5f} rho   "
            f"exact {ex:.5f}   rel {mval / ex - 1:+.2e}")
    msc = panel_added_mass(Yc, Zc, "sway", "phi0")
    say(f"     semicircle sway phi0: {msc:.5f} rho a^2 = "
        f"{msc / (np.pi / 2):.5f} x the rigid-lid value; 4/pi^2 = "
        f"{4 / np.pi ** 2:.5f} (design review quotes 0.406)")
    for ae_ in [0.1, 3.0]:
        Yq, Zq = ae_ * np.cos(th), -np.sin(th)
        rq = panel_added_mass(Yq, Zq, "sway", "phi0") / panel_added_mass(
            Yq, Zq, "sway", "lid")
        say(f"     half-ellipse a/b = {ae_}: sway phi0/lid {rq:.5f}  (same "
            f"4/pi^2 for every semi-ellipse: m22_phi0 = (2/pi) rho d^2)")
    rat = plate_sway_ratio_chebyshev()
    # thin vertical slab by BEM, thickness 1% of draught
    dd, tt = 1.0, 0.01
    Ys, Zs = contour_polyline([(tt / 2, 0.0), (tt / 2, -dd), (-tt / 2, -dd),
                               (-tt / 2, 0.0)], 160)
    s_phi = panel_added_mass(Ys, Zs, "sway", "phi0")
    s_lid = panel_added_mass(Ys, Zs, "sway", "lid")
    say(f"     vertical plate sway phi0/lid: Chebyshev series {rat:.4f}; "
        f"BEM on a 1%-thick slab {s_phi / s_lid:.4f} (lid "
        f"{s_lid / (np.pi / 2):.4f} x pi/2 rho d^2)")

    # (i) V wedge heave
    say("   (i) heave, phi=0, V section of deadrise 20 deg, waterline "
        "half-beam c_g = d/tan(beta), NO chine (pure wedge):")
    tb = np.tan(np.radians(20.0))
    Yw, Zw = contour_polyline([(1.0, 0.0), (0.0, -tb), (-1.0, 0.0)], 200)
    mw = panel_added_mass(Yw, Zw, "heave", "phi0")
    say(f"     exact phi0: {mw:.4f} rho c_g^2 = {mw / (np.pi / 2):.4f} x "
        f"(pi/2 rho c_g^2);  v0/Wagner pile-up c = (pi/2) c_g gives "
        f"(pi/2)^3 = {(np.pi / 2) ** 3:.4f} rho c_g^2 = "
        f"{(np.pi / 2) ** 2:.4f} x (pi/2 rho c_g^2)")
    say(f"     -> for a section oscillating about a FIXED waterline, v0's "
        f"m_a is {(np.pi / 2) ** 3 / mw:.2f}x the exact phi=0 value "
        f"(Wagner's pi/2 belongs to water ENTRY, where the waterline rises)")

    # (iii) scarab stations at rest and at 25 kn
    p = plant
    from sim.planing_vessel import SP
    rows = {}
    for tag, (z, th_) in [("rest", (0.0, 0.0)),
                          ("25 kn", p.running_attitude(25 * 0.514444))]:
        d = -(p.Z_rest + z + p.zk + SP * p.x * (p.Th_rest + th_))
        A_, c_, ma_, _, _ = p._section(d)
        heave0 = sway0 = swayL = 0.0
        geo = 0.0
        for i in range(p.x.size):
            if d[i] <= 1e-6:
                continue
            Ys_, Zs_ = section_contour(d[i], p.yc[i], p.tanb, 50)
            heave0 += panel_added_mass(Ys_, Zs_, "heave", "phi0")
            sway0 += panel_added_mass(Ys_, Zs_, "sway", "phi0")
            swayL += panel_added_mass(Ys_, Zs_, "sway", "lid")
            cg = min(d[i] / p.tanb, p.yc[i])
            geo += np.pi / 2 * cg ** 2
        rho, dx = p.rho, p.dx
        m22v0 = (0.5 * np.pi * rho * np.maximum(d, 0) ** 2).sum() * dx
        rows[tag] = dict(ma_v0=ma_.sum() * dx, heave0=heave0 * rho * dx,
                         geo=geo * rho * dx, sway0=sway0 * rho * dx,
                         swayL=swayL * rho * dx, m22v0=m22v0)
        r = rows[tag]
        say(f"   (iii) scarab195, {tag}: sum over 48 stations (kg)")
        note = "" if tag == "rest" else ("   [at speed NOT comparable: the "
                                         "pile-up of steady entry is physical]")
        say(f"        heave: v0 m_a (Wagner c, capped) {r['ma_v0']:7.0f} | "
            f"pi/2 rho c_g^2 {r['geo']:7.0f} | exact 2D phi0 section "
            f"{r['heave0']:7.0f}{note}")
        say(f"        sway : v0 m22 = pi/2 rho d^2 {r['m22v0']:7.0f} | exact "
            f"2D rigid lid {r['swayL']:7.0f} | exact 2D phi0 {r['sway0']:7.0f}"
            f"  (phi0/lid {r['sway0'] / r['swayL']:.3f})")
    say("        3D BEM at rest (studies/_cache/anyspeed/memory_fit.txt, "
        "Ogilvie fit, err_A 0.13): A33_inf 4538 kg, A22_inf 318 kg")
    r = rows["rest"]
    say(f"        => at rest the factor v0/3D = {r['ma_v0'] / 4538:.2f} splits "
        f"into section/pile-up {r['ma_v0'] / r['heave0']:.2f} x 3D "
        f"{r['heave0'] / 4538:.2f}  (heave)")
    return rows


# =====================================================================  C
def _x_int(mu, a):
    """|int_{-a}^{a} x e^{i mu x} dx| (purely imaginary)."""
    mu = np.asarray(mu, float)
    small = np.abs(mu * a) < 1e-4
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 2 * (np.sin(mu * a) / mu ** 2 - a * np.cos(mu * a) / mu)
    return np.where(small, 2 * mu * a ** 3 / 3, v)


def _z_int(k, T):
    """int_{-T}^0 (1 - z^2/T^2) e^{k z} dz."""
    k = np.asarray(k, float)
    small = k * T < 1e-3
    e = np.exp(-k * T)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t1 = (1 - e) / k
        t2 = (2 / k ** 3 - e * (T ** 2 / k + 2 * T / k ** 2 + 2 / k ** 3)) \
            / T ** 2
        v = t1 - t2
    return np.where(small, 2 * T / 3 - k * T * T / 4 + k * k * T ** 3 / 15, v)


def michell_wigley(U, L, B, T, rho=RHO, g=G):
    """Michell (1898) wave resistance of the Wigley hull
    y = (B/2)(1-(2x/L)^2)(1-(z/T)^2):
        R = (4 rho g^2/(pi U^2)) int_1^inf |I|^2 lam^2/sqrt(lam^2-1) dlam
        I = int int f_x e^{k0 lam^2 z} e^{i k0 lam x} dx dz,  k0 = g/U^2
    (Wehausen & Laitone 1960 eq. 13.37; Newman 1977 sec. 6.19); in
    t = sqrt(lam^2-1) the measure is lam dt (no endpoint singularity)."""
    k0 = g / U ** 2
    a = L / 2

    def f(t):
        lam = np.sqrt(1 + t * t)
        I = (B / 2) * (8 / L ** 2) * _x_int(k0 * lam, a) * _z_int(
            k0 * lam * lam, T)
        return I * I * lam
    # integrate in log t over decades, up to where both factors have died
    t_hi = max(1e3, 1e3 / (k0 * L), 1e2 / np.sqrt(k0 * T))
    edges = np.concatenate([np.linspace(0, 1, 9),
                            np.geomspace(1, t_hi, 400)[1:]])
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        tot += integrate.quad(f, lo, hi, limit=200, epsrel=1e-10)[0]
    return 4 * rho * g * g / (np.pi * U * U) * tot


def part_C():
    say("=" * 78)
    say("C  MICHELL THIN-SHIP WAVE RESISTANCE vs Froude number (Wigley form)")
    # closed forms vs direct quadrature
    L, B, T = 1.0, 0.1, 0.0625
    for mu, k in [(3.0, 5.0), (40.0, 900.0)]:
        xq = integrate.quad(lambda x: x * np.sin(mu * x), -L / 2, L / 2,
                            limit=400)[0]
        zq = integrate.quad(lambda z: (1 - (z / T) ** 2) * np.exp(k * z), -T,
                            0)[0]
        say(f"   closed forms vs quadrature: X({mu}) {_x_int(mu, L / 2):.8e} "
            f"vs {xq:.8e};  Z({k}) {_z_int(k, T):.8e} vs {zq:.8e}")
    say("   (a) classic Wigley L/B 10, B/T 1.6: C_w = R_w/(1/2 rho U^2 S), "
        "S from the Wigley surface; asymptote derived below")
    # wetted surface of the Wigley hull (both sides), numerically
    def wetted(L, B, T):
        def dS(z, x):
            fx = (B / 2) * (-8 * x / L ** 2) * (1 - (z / T) ** 2)
            fz = (B / 2) * (1 - (2 * x / L) ** 2) * (-2 * z / T ** 2)
            return 2 * np.sqrt(1 + fx * fx + fz * fz)
        return integrate.dblquad(dS, -L / 2, L / 2, -T, 0)[0]
    S = wetted(L, B, T)
    Awp = 2 * B * L / 3
    vol = 4 * L * B * T / 9
    say(f"   S = {S:.4f} L^2, A_wp = {Awp:.4f}, volume = {vol:.5f}")
    say("   derived high-Fn law (waterline dominates, log range "
        "sqrt(k0/T) < mu < 1/L):")
    say("       R_w ~ (rho g^2 A_wp^2 / (2 pi U^2)) [ln(Fn^2 T/L) + C]")
    say("     Fn      C_w        R_w/(rho g vol)   R_w 2piU^2/(rho g^2 Awp^2)"
        " - ln(Fn^2 T/L)   [-> const C]")
    for Fn in [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0,
               50.0]:
        U = Fn * np.sqrt(G * L)
        R = michell_wigley(U, L, B, T)
        cw = R / (0.5 * RHO * U * U * S)
        lead = R * 2 * np.pi * U * U / (RHO * G * G * Awp ** 2) - np.log(
            Fn * Fn * T / L)
        say(f"   {Fn:5.1f}   {cw:10.3e}   {R / (RHO * G * vol):10.3e}        "
            f"   {lead:+9.4f}")
    say("   (b) scarab-proportioned Wigley (L 5.4, B 2.025, T 0.385 m; "
        "thin-ship at B/L 0.375 is OUTSIDE its assumption -- order of "
        "magnitude only)")
    L, B, T = 5.4, 2.025, 0.385
    vol = 4 * L * B * T / 9
    say("     kn     Fn_L   R_w/W (W = rho g vol, vol = 4LBT/9)")
    out = []
    for kn in [2, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 25, 35, 50]:
        U = kn * 0.514444
        R = michell_wigley(U, L, B, T)
        out.append((kn, U / np.sqrt(G * L), R / (RHO * G * vol)))
        say(f"   {kn:5.0f}   {U / np.sqrt(G * L):5.2f}   {R / (RHO * G * vol):.4f}")
    return out


# =====================================================================  D
def part_D(plant):
    say("=" * 78)
    say("D  DIVERGENCE-THEOREM IDENTITY behind 'Xw is a second copy' "
        "(scarab geometry, v0 kinematics, long-wave pressure p = rho g "
        "(eta(x) - z))")
    p = plant
    from sim.planing_vessel import SP
    rho = p.rho
    tb = p.tanb
    h = p.hull
    L, b, lcg = h.L, h.b, h.lcg
    # the chine from the geometry module (the plant's source), without the
    # floor; the side integral below substitutes xi = x0f + Lt sin(phi) on
    # the quarter-ellipse taper and the keel is straight, so the check is
    # for a family hull of that shape only (scarab195): refuse others
    from hydro.hull_geometry import form_of, PlaningHullForm
    fm = form_of(h)
    if not (isinstance(fm, PlaningHullForm) and fm.v0_member()
            and fm.stem_rise == 0.0):
        raise ValueError(f"part D is written for the family's elliptic "
                         f"taper and a straight keel; {h.name!r} is not")
    g = fm.stations()
    x0f = (1.0 - fm.taper) * L         # taper start, from the transom
    xs = -lcg                          # body x of the transom

    def yc_of(xi):                     # xi from the transom, continuous
        return float(g.chine(xi)) - g.eps

    results = []
    for tag, a_w, z_, th_ in [("calm, rest trim", 0.0, 0.0, 0.0),
                              ("calm, +3 deg extra trim", 0.0, 0.02,
                               np.radians(-3.0)),
                              ("wave a 0.10 m, lam 2L, rest", 0.10, 0.0, 0.0),
                              ("wave a 0.10 m, lam 1L, +2 deg", 0.10, 0.03,
                               np.radians(-2.0))]:
        lam = 2 * L if "2L" in tag else L
        kw = 2 * np.pi / lam
        ph = 0.7
        eta = lambda xb: a_w * np.cos(kw * xb + ph)            # body x
        eta_x = lambda xb: -a_w * kw * np.sin(kw * xb + ph)
        Th = p.Th_rest + th_
        zkeel = lambda xb: p.Z_rest + z_ - h.vcg + SP * xb * Th
        tau = SP * Th
        dfun = lambda xb: eta(xb) - zkeel(xb)

        # --- exact pressure integrals by independent quadrature
        # bottom (V part): -int dx int dy p dz_s/dx, dz_s/dx = tau
        def bottom_line(xb):
            xi = xb - xs
            d = dfun(xb)
            if d <= 0:
                return 0.0
            ymax = min(yc_of(xi), d / tb)
            f = lambda y: rho * G * (d - y * tb)
            return 2 * integrate.quad(f, 0, ymax)[0]
        # sides above the chine, x-force 2 int yc' (int p dz) dx; use the
        # substitution xi = x0f + (L - x0f) sin(phi) on the taper
        def side_p(xb, yc):
            d = dfun(xb)
            dch = yc * tb
            if d <= dch:
                return 0.0
            return rho * G * 0.5 * (d - dch) ** 2
        Fb = -tau * integrate.quad(bottom_line, xs, xs + L, limit=400,
                                   points=[xs + x0f])[0]
        Lt = L - x0f

        def side_int(phi_):
            xi = x0f + Lt * np.sin(phi_)
            yc = yc_of(xi)
            dyc = -0.5 * b * np.sin(phi_)          # yc'(xi) dxi/dphi
            return 2 * side_p(xi + xs, yc) * dyc
        Fs = integrate.quad(side_int, 0, np.pi / 2, limit=400)[0]
        # transom face (outward normal -x): +int p dA
        dT = dfun(xs)
        ycT = yc_of(0.0)

        def face_line(zz):            # zz = depth below the surface
            # half-width of the transom section at depth zz below the surface
            hgt = dT - zz             # height above the keel
            if hgt < 0:
                return 0.0
            return 2 * min(hgt / tb, ycT) * rho * G * zz
        Ff = integrate.quad(face_line, 0, max(dT, 0), limit=200)[0] \
            if dT > 0 else 0.0

        # closed-body FK: -rho g int A eta_x dx (A by quadrature as well)
        def area(xb):
            xi = xb - xs
            d = dfun(xb)
            if d <= 0:
                return 0.0
            yc = yc_of(xi)
            dch = yc * tb
            be = min(d, dch)
            return be * be / tb + 2 * yc * max(d - dch, 0)
        FK = -rho * G * integrate.quad(lambda xb: area(xb) * eta_x(xb), xs,
                                       xs + L, limit=400,
                                       points=[xs + x0f])[0]
        tot = Fb + Fs + Ff
        # --- v0's own discrete terms at its stations
        d_st = eta(p.x) - zkeel(p.x)
        A_st = p._section(d_st)[0]
        P_v0 = -(rho * G * A_st * tau).sum() * p.dx          # a_b = 1
        Xw_v0 = -(rho * G * A_st * eta_x(p.x)).sum() * p.dx  # a_b = 1
        say(f"   {tag}:")
        say(f"      quadrature: bottom {Fb:+9.2f}  sides(taper) {Fs:+8.2f}  "
            f"transom face {Ff:+9.2f}  sum {tot:+9.3f} N")
        say(f"      closed-body FK -rho g int A eta_x dx = {FK:+9.3f} N;  "
            f"sum - FK = {tot - FK:+.2e} N")
        say(f"      v0 discrete (a_b=1): bottom term -rho g sum A tau dx "
            f"{P_v0:+9.2f} (quadrature {Fb:+9.2f});  Xw {Xw_v0:+9.2f};  "
            f"v0 total wave part = bottom(wave)-bottom(calm) + Xw")
        results.append((tag, Fb, Fs, Ff, FK, P_v0, Xw_v0))
    # double count: wave cases vs calm at the same attitude
    say("   reading: the bottom pressure integral (= v0's bottom term) already "
        "contains the FK surge force; the identity")
    say("      bottom + sides + face = -rho g int A eta_x dx   (exactly, any "
        "trim/heave/wave)")
    say("   holds to quadrature error, so a separately added Xw = -rho g a_b "
        "int A eta_x dx is a second copy at a_b = 1.")
    return results


def part_D_v0_doublecount(plant):
    """v0's longitudinal FK in a captive regular head wave at the rest
    attitude, against the exact closed-body FK: first harmonics over a
    wave period. Terms: Xw (pv:300), the wave part of v0's bottom term
    -rho g sum A tau dx (inside Xp, pv:355; a_b = 1 below C_V 0.5)."""
    p = plant
    from sim.planing_vessel import SP
    rho = p.rho
    say("   v0 vs exact, captive at the rest attitude, regular head wave "
        "a = 0.02 m, first harmonics over one period (a_b = 1):")
    say("      lam/L   |FK exact|   |Xw|/|FK|   |v0 bottom wave part|/|FK|   "
        "|v0 total|/|FK|   phase(v0 total - FK) deg")
    L = p.hull.L
    Th = p.Th_rest
    tau = SP * Th
    dc = -(p.Z_rest + p.zk + SP * p.x * Th)
    bc = -(rho * G * p._section(dc)[0] * tau).sum() * p.dx
    phs = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    for lr in [1.0, 2.0, 4.0, 8.0]:
        kw = 2 * np.pi / (lr * L)
        xw, bw = [], []
        for ph in phs:
            eta = 0.02 * np.cos(kw * p.x + ph)
            eta_x = -0.02 * kw * np.sin(kw * p.x + ph)
            dw = eta + dc
            A = p._section(dw)[0]
            bw.append(-(rho * G * A * tau).sum() * p.dx - bc)
            xw.append(-(rho * G * A * eta_x).sum() * p.dx)
        xw, bw = np.array(xw), np.array(bw)
        h1 = lambda y: 2 * np.mean(y * np.exp(-1j * phs))
        FK = h1(xw)          # Xw equals the closed-body FK (part D above)
        tot = h1(xw + bw)
        say(f"      {lr:4.0f}    {abs(FK):8.2f} N   {abs(h1(xw)) / abs(FK):6.3f}"
            f"        {abs(h1(bw)) / abs(FK):8.3f}                  "
            f"{abs(tot) / abs(FK):6.3f}          "
            f"{np.degrees(np.angle(tot / FK)):+7.1f}")
    say("      v0 total = Xw + bottom wave part = 2 FK - d(face) - d(taper): "
        "the missing transom face is first order in")
    say("      the RELATIVE transom immersion (no k), the FK is first order in "
        "the slope (k a); so the captive error is not")
    say("      'twice FK' in general; it is Xw (a copy) plus the missing face. "
        "With free heave/pitch the relative immersion")
    say("      shrinks in long waves and the doubled slope force dominates "
        "(the audit's 1.9-2.3x, measured that way).")


# =====================================================================  E
def part_E(plant):
    say("=" * 78)
    say("E  WHERE GRAVITY IS NOT SMALL")
    p = plant
    tb = p.tanb
    yT = float(p.yc[0])
    say("   (a) steady flow: in 2D+t the cross-plane free-surface condition "
        "is U^2 phi_xx + g phi_z = 0;")
    say("       phi=0 needs g L_k^2 / (U^2 c) << 1 (L_k keel wetted length, "
        "c wetted half-beam at the transom)")
    say("       = 2 lam^2 / C_V^2 in Savitsky's variables with c = b/2. "
        "v0 branch from studies/_cache/anyspeed/audit_transom.txt:")
    tab = [(1, 2.01, 0.394), (3, 2.16, 0.374), (5, 2.32, 0.385),
           (7, 1.90, 0.446), (9, 2.41, 0.513), (10, 5.45, 0.513),
           (12, 6.38, 0.493), (14, 5.78, 0.427), (16, 4.76, 0.369),
           (18, 4.08, 0.331), (20, 3.51, 0.301), (25, 2.36, 0.253),
           (30, 1.71, 0.231), (35, 1.54, 0.221)]
    from hydro.planing import savitsky
    h = p.hull
    say("      kn   C_V    L_k m   c_T m   g L_k^2/(U^2 c)   | Savitsky lam  "
        "2 lam^2/C_V^2")
    for kn, trim, dT in tab:
        U = kn * 0.514444
        Lk = min(p.hull.L, dT / np.tan(np.radians(trim)))
        cT = min(yT, 0.5 * np.pi * dT / tb)
        par = G * Lk * Lk / (U * U * cT)
        cv = U / np.sqrt(G * h.b)
        try:
            sv = savitsky(U, h.mass, h.lcg, h.b, h.beta_deg)
            sv_s = f"{sv['lam']:6.2f}        {2 * sv['lam'] ** 2 / cv ** 2:6.2f}"
        except Exception:
            sv_s = "   --"
        say(f"    {kn:4.0f}  {cv:4.2f}   {Lk:5.2f}   {cT:5.3f}       "
            f"{par:8.2f}       | {sv_s}")
    # Fridsma for comparison
    from studies.exp_planing import fridsma_hull
    for case in "AB":
        hf, V = fridsma_hull(case)
        sv = savitsky(V, hf.mass, hf.lcg, hf.b, hf.beta_deg, rho=1000.0)
        cv = V / np.sqrt(G * hf.b)
        say(f"    Fridsma {case}: C_V {cv:.2f}, Savitsky lam {sv['lam']:.2f}, "
            f"2 lam^2/C_V^2 = {2 * sv['lam'] ** 2 / cv ** 2:.2f}  (the planing "
            f"validation of v0 sits here)")
    # (b) encounter frequency
    say("   (b) waves: the linear section problem's gravity correction scales "
        "as 1/(K_e b/2), K_e = w_e^2/g (part A: -2/(Kd));")
    say("       share of JONSWAP (Hs 1 m, Tp 5 s, gamma 3.3) wave-height "
        "variance with K_e b/2 < 1 (gravity O(1)), b = 2.025 m")
    w = np.linspace(0.3, 6.0, 4000)
    wp = 2 * np.pi / 5.0
    sig = np.where(w <= wp, 0.07, 0.09)
    gam = 3.3 ** np.exp(-(w - wp) ** 2 / (2 * sig ** 2 * wp ** 2))
    Sw = (5 / 16) * 1.0 ** 2 * wp ** 4 / w ** 5 * np.exp(-1.25 * (wp / w) ** 4) \
        * gam
    Sw /= np.trapezoid(Sw, w) / (1.0 ** 2 / 16)
    k = w * w / G
    say("      heading (0 = following, 180 = head):   0      45     90    "
        "135    180")
    for kn in [0, 5, 10, 15, 25, 35]:
        U = kn * 0.514444
        row = []
        for mu in [0, 45, 90, 135, 180]:
            we = np.abs(w - k * U * np.cos(np.radians(mu)))
            Ke = we * we / G
            frac = np.trapezoid(Sw * (Ke * h.b / 2 < 1.0), w) / np.trapezoid(
                Sw, w)
            row.append(frac)
        say(f"      {kn:3d} kn                              " +
            "  ".join(f"{v:5.2f}" for v in row))
    say("      -> 'vanishes at high speed' is true in head and bow seas "
        "(w_e grows with U), NOT in following/quartering seas,")
    say("         where w_e = w - kU cos(mu) passes through 0 at any speed "
        "(surf-riding band).")


# =====================================================================  F
def part_F():
    say("=" * 78)
    say("F  THE ADDITIVE COMPOSITE  F ~ F_NL,phi0 + (F_lin,g - F_lin,phi0): "
        "error = the mixed term")
    say("   exact model problem (part A): the 'nonlinearity' is the draught "
        "change d0 -> d during the motion; true A(K, d),")
    say("   composite A(inf, d) + [A(K, d0) - A(inf, d0)]; damping: true "
        "B(w, d), composite B(w, d0).")
    say("   error of the composite, (composite - true)/A_inf(d) for A and "
        "(composite - true)/(w A_inf(d)) for B:")
    d0 = 1.0
    say("     K d0 :  d/d0 =   0.7        1.0        1.3        1.6      "
        "|  B: 0.7      1.3      1.6")
    a_inf = lambda d: 2 * RHO / np.pi * d * d * np.log(2)
    for kd in [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        K = kd / d0
        w = np.sqrt(K * G)
        Ad0 = wavemaker_A(K, d0)
        out, outb = [], []
        for r in [0.7, 1.0, 1.3, 1.6]:
            d = r * d0
            tru = wavemaker_A(K, d)
            comp = a_inf(d) + Ad0 - a_inf(d0)
            out.append((comp - tru) / a_inf(d))
        for r in [0.7, 1.3, 1.6]:
            outb.append((wavemaker_B(w, d0) - wavemaker_B(w, r * d0))
                        / (w * a_inf(r * d0)))
        say(f"    {kd:6.1f}            " + "  ".join(f"{v:+9.2e}" for v in out)
            + "  | " + "  ".join(f"{v:+8.4f}" for v in outb))
    say("   -> exact at d = d0 (the linear regime) for every K; vanishes as "
        "K d0 -> inf; O(1) where gravity is O(1) AND the motion is large.")
    say("      This is a superposition (first-order cross-term neglected), "
        "not a matched-asymptotic composite: the two")
    say("      expansions are in different parameters (amplitude vs 1/Fn^2) "
        "and the common part is exact only if the")
    say("      linear theory and the linearised nonlinear theory are the SAME "
        "problem (geometry, body and transom conditions).")


# =====================================================================  G
def part_G():
    say("=" * 78)
    say("G  HEUN (RK2) AMPLITUDE FACTOR on x'' = -w^2 x: |R(i w dt)| = "
        "sqrt(1 + (w dt)^4/4)")
    for wn, dt in [(3.94, 0.02), (4.9, 0.02), (4.9, 0.004)]:
        s = np.array([1.0, 0.0])
        Mx = np.array([[0, 1], [-wn * wn, 0]])
        n = int(round(100 / dt))
        for _ in range(n):
            k1 = Mx @ s
            k2 = Mx @ (s + dt * k1)
            s = s + 0.5 * dt * (k1 + k2)
        E = 0.5 * (s[1] ** 2 + wn * wn * s[0] ** 2) / (0.5 * wn * wn)
        Eth = (1 + (wn * dt) ** 4 / 4) ** n
        say(f"   w {wn} dt {dt}: energy after 100 s {E:.5f} (theory "
            f"{Eth:.5f}), +{(E - 1) * 100:.2f}% per 100 s")


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    say(f"principles_ledger_checks  {time.strftime('%Y-%m-%d %H:%M')}")
    part_A()
    from hydro import hulls
    from sim.planing_vessel import PlaningVessel
    from sim.test_vessel import Monochromatic
    plant = PlaningVessel(hulls.get("scarab195"), Monochromatic(1.0, 0.0),
                          dt=0.02)
    part_B(plant)
    part_C()
    part_D(plant)
    part_D_v0_doublecount(plant)
    part_E(plant)
    part_F()
    part_G()
    say(f"(total {time.time() - t0:.0f} s)")
    with open(os.path.join(OUT, "checks.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
