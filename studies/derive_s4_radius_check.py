#!/usr/bin/env python3
"""
S4: validity radius of the second-order expansion on the Scarab 195 hull
(SECOND_ORDER_OPTIONS_2026-10-09.md section 8, the falsifying check of the
recommendation "option (iii)").

Test function: the EXACT hydrostatic + Froude-Krylov force of the incident
wave on the instantaneous wetted surface of the plant's own prismatic hull
(straight keel, 20 deg deadrise V bottom to the chine yc(x), vertical sides
above the chine, quarter-ellipse chine taper over the bow 35 % of L, flat
transom), at its rest attitude (sink Z_rest, trim Th_rest of
sim/planing_vessel.PlaningVessel._static via the V1 vessel, read-only). No
potential-flow solve. Pressure (D1'/fk_exact convention, depth decay measured
from the local surface so that p = 0 exactly on the free surface):

    p = -rho g z + rho g eta e^{-k eta} e^{k z},   eta(x, t) = A cos(w t + k x)

(variant "plain": p = -rho g z + rho g eta e^{k z}, V2's Airy convention; the
two have the SAME heave x wave second-order coefficient -- the stretching
correction is wave x wave -- and differ from third order on). The section
integrals (bottom: int p dy, int p y dy over the wetted half-width; transom
face and tapered bow sides: int p w(z) dz, int p w(z) z dz) are closed-form
(polynomial x exponential), so the only discretisation is the station sum
along x (480 sub-stations, 1.125 cm; the plant's 48 stations as a variant).

Inputs: prescribed heave z = a cos(w t) (up) at w = 3.9 rad/s and a head wave
eta = A cos(w t + k x) at the same frequency; relative-elevation amplitude
delta = a + A split EQUALLY, a = A = delta/2 (V2 used a = A). With both at the
same frequency the relative elevation at station x is
(delta/2)[cos(w t + k x) - cos(w t)], amplitude delta |sin(k x / 2)|: ~delta at
the transom (x = -2.18 m from the CG), 0 at the CG, 0.64 delta at the stem.

Measured: the exact double-input increment over one common period
    D(t) = F(a, A) - F(a, 0) - F(0, A) + F(0, 0)
against its second-order Taylor prediction D2(t) = a A c2(t), c2(t) =
D(t)/(a A) at the smallest amplitude (delta_ref = 0.5 mm; quadratic scaling
checked against 1 mm and 2 mm; and against V2's hf coefficient in
v2_results_quick.json). Ratios reported per amplitude: RMS over the period,
sum-frequency (2 w) component, mean component, for heave force and pitch
moment. Also: at which amplitude and at which stations the instantaneous
waterline crosses the chine line (and the keel).

Run: cd repo && .venv/Scripts/python.exe -m studies.derive_s4_radius_check
Outputs: studies/_cache/derive/s4_radius.txt, s4_radius.json, s4_radius.png
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "_cache", "derive")
G0 = 9.81
SP = -1.0                      # the plant's pitch sign (bow down positive th; d = e - (Z + z + zk + SP x Th))
OMEGA = 3.9
DELTAS = (0.01, 0.03, 0.05, 0.1, 0.2, 0.3)
DELTA_REF = (0.0005, 0.001, 0.002)   # smallest first = the Taylor coefficient; the others check the scaling
NT = 1600
SUB = 10                       # sub-stations per plant station (480 in all)


# ------------------------------------------------------------------ hull
class Hull:
    """Geometry at the rest attitude, from the plant (read-only)."""

    def __init__(self, sub=SUB):
        from studies.derive_v1_linear_consistency import make_vessel
        v = make_vessel(False, 0.0)
        h = v.hull
        self.v = v
        self.rho, self.tb = float(v.rho), float(v.tanb)
        self.L, self.b, self.lcg, self.vcg = float(h.L), float(h.b), float(h.lcg), float(h.vcg)
        self.taper = float(h.taper)
        self.Z_rest, self.Th_rest = float(v.Z_rest), float(v.Th_rest)
        # the plant: d = e - (Z + z + zk + SP x Th) => keel earth height z_k = Z + z + zk + SP x Th,
        # keel slope dz_k/dx = SP * Th (zk constant, stem_rise 0)
        self.kappa = SP * self.Th_rest
        self.m = float(v.m)
        n = len(v.x)
        edges = np.linspace(0.0, self.L, n * sub + 1)
        self.xi = 0.5 * (edges[:-1] + edges[1:])          # from the transom
        self.x = self.xi - self.lcg                        # body x from the CG
        self.dx = self.L / (n * sub)
        self.yc = self.yc_of(self.xi)
        self.dyc = self.yc_of(edges[1:]) - self.yc_of(edges[:-1])   # chine half-width change over each cell
        self.zkb = -self.vcg                               # keel, body frame (straight keel)
        self.x_T = -self.lcg                               # transom
        self.yc_T = self.yc_of(np.array(0.0))
        # the plant's own stations (for the crossing log)
        self.x48, self.yc48, self.xi48 = v.x, v.yc, v.x + self.lcg

    def yc_of(self, xi):
        x0 = (1.0 - self.taper) * self.L
        s = np.clip((np.asarray(xi, float) - x0) / max(self.L - x0, 1e-9), 0.0, 1.0)
        return 0.5 * self.b * np.sqrt(np.maximum(1.0 - s ** 2, 0.0)) + 1e-4

    def zkeel(self, x, z):
        """Keel earth height at body x for heave z (shape broadcast)."""
        return self.Z_rest + z + self.zkb + SP * x * self.Th_rest


# ------------------------------------------------------------------ closed-form integrals
def _anti_exp(z, k, mmax):
    """Antiderivatives J_m(z) = int z^m e^{k z} dz, m = 0..mmax (recursive)."""
    E = np.exp(k * z)
    J = [E / k]
    for m in range(1, mmax + 1):
        J.append(z ** m * E / k - m / k * J[m - 1])
    return J


def int_poly_exp(coef, k, z0, z1):
    """int_{z0}^{z1} (sum_m coef[m] z^m) e^{k z} dz, coef list of arrays/scalars."""
    J0 = _anti_exp(z0, k, len(coef) - 1)
    J1 = _anti_exp(z1, k, len(coef) - 1)
    return sum(c * (a - b) for c, a, b in zip(coef, J1, J0))


def int_poly(coef, z0, z1):
    return sum(c * (z1 ** (m + 1) - z0 ** (m + 1)) / (m + 1) for m, c in enumerate(coef))


def face_integrals(zk, H, yc, tb, k, gam, Zcg, rho):
    """Pressure p = rho g (-z + gam e^{k z}) on a V-section face (transom or a
    tapered side strip) seen in the (y, z) plane, width w(z) = 2 min((z - zk)/tb, yc)
    for the transom (mode 'transom') -- returns (int p w dz, int p w (z - Zcg) dz)
    over z in [zk, H]. Arrays over time."""
    zch = zk + yc * tb
    H1 = np.minimum(H, zch)
    wet = H > zk
    out_F = np.zeros_like(H)
    out_M = np.zeros_like(H)
    # piece 1: w = (2/tb)(z - zk) = w0 + w1 z, w0 = -2 zk/tb, w1 = 2/tb
    for lo, hi, w0, w1 in ((zk, H1, -2.0 * zk / tb, 2.0 / tb * np.ones_like(zk)),
                           (zch, np.maximum(H, zch), 2.0 * yc * np.ones_like(zk), np.zeros_like(zk))):
        # p w = rho g [(-w0 z - w1 z^2) + gam e^{kz} (w0 + w1 z)]
        F = int_poly([0.0, -w0, -w1], lo, hi) + gam * int_poly_exp([w0, w1], k, lo, hi)
        # p w (z - Zcg): poly (-w0 z - w1 z^2)(z - Zcg) = w0 Zcg z + (w1 Zcg - w0) z^2 - w1 z^3
        #                exp (w0 + w1 z)(z - Zcg) = -w0 Zcg + (w0 - w1 Zcg) z + w1 z^2
        M = int_poly([0.0, w0 * Zcg, w1 * Zcg - w0, -w1], lo, hi) \
            + gam * int_poly_exp([-w0 * Zcg, w0 - w1 * Zcg, w1], k, lo, hi)
        out_F = out_F + np.where(hi > lo, F, 0.0)
        out_M = out_M + np.where(hi > lo, M, 0.0)
    return rho * G0 * np.where(wet, out_F, 0.0), rho * G0 * np.where(wet, out_M, 0.0)


def side_integrals(zch, H, k, gam, Zcg, rho):
    """int_{zch}^{H} p dz and int p (z - Zcg) dz for the vertical side above the chine (per unit width)."""
    hi = np.maximum(H, zch)
    F = int_poly([0.0, -1.0], zch, hi) + gam * int_poly_exp([1.0], k, zch, hi)
    M = int_poly([0.0, Zcg, -1.0], zch, hi) + gam * int_poly_exp([-Zcg, 1.0], k, zch, hi)
    return rho * G0 * F, rho * G0 * M


def forces(hull, t, a, A, variant="stretched"):
    """Heave force (N, up), pitch moment about +y through the CG (N m; the plant's
    My = -sum x f dx sign), surge force (N), per time step; plus the per-station
    immersion d = H - z_k (nt, nx) and the transom immersion (nt,)."""
    w = OMEGA
    k = w * w / G0
    rho, tb = hull.rho, hull.tb
    x = hull.x[None, :]
    z = a * np.cos(w * t)[:, None]
    H = A * np.cos(w * t[:, None] + k * x)
    zk = hull.zkeel(x, z)
    Zcg = hull.Z_rest + z
    d = H - zk
    yc = hull.yc[None, :]
    yw = np.minimum(np.maximum(d, 0.0) / tb, yc)
    gam = H * np.exp(-k * H) if variant == "stretched" else H          # p = rho g (-z + gam e^{k z})
    # bottom: p(y) = rho g [alpha + beta y + gamma e^{c y}], z = zk + tb y
    alpha, beta, gamma, c = -zk, -tb, gam * np.exp(k * zk), k * tb
    ec = np.exp(c * yw)
    I0 = rho * G0 * (alpha * yw + 0.5 * beta * yw ** 2 + gamma * (ec - 1.0) / c)
    I1 = rho * G0 * (0.5 * alpha * yw ** 2 + beta * yw ** 3 / 3.0 + gamma * (ec * (c * yw - 1.0) + 1.0) / c ** 2)
    kap = hull.kappa
    fz = 2.0 * I0
    fx = -kap * fz
    my = 2.0 * (-kap * (zk - Zcg) * I0 - kap * tb * I1 - x * I0)
    Fz = fz.sum(1) * hull.dx
    My = my.sum(1) * hull.dx
    Fx = fx.sum(1) * hull.dx
    # tapered bow sides above the chine (where wetted)
    zch = zk + yc * tb
    sF, sM = side_integrals(zch, H, k, gam, Zcg, rho)
    wet_side = H > zch
    dyc = hull.dyc[None, :]
    Fx = Fx + (2.0 * dyc * np.where(wet_side, sF, 0.0)).sum(1)
    My = My + (2.0 * dyc * np.where(wet_side, sM, 0.0)).sum(1)
    # transom face
    HT = A * np.cos(w * t + k * hull.x_T)
    zkT = hull.zkeel(hull.x_T, z[:, 0])
    gamT = HT * np.exp(-k * HT) if variant == "stretched" else HT
    FT, MT = face_integrals(zkT, HT, hull.yc_T, tb, k, gamT, Zcg[:, 0], rho)
    Fx = Fx + FT
    My = My + MT
    return Fz, My, Fx, d, HT - zkT


def project(t, M, Om):
    if Om < 1e-9:
        return complex(np.mean(M))
    return complex(2.0 * np.mean(M * np.exp(-1j * Om * t)))


def brute_check(hull, say):
    """Closed forms against dense numerical quadrature of the same pressure at one instant."""
    rng = np.random.default_rng(1)
    w = OMEGA
    k = w * w / G0
    a, A, t0 = 0.05, 0.1, 0.37
    rho, tb = hull.rho, hull.tb
    worst = 0.0
    for variant in ("stretched", "plain"):
        Fz, My, Fx, d, dT = forces(hull, np.array([t0]), a, A, variant)
        # per-station numerical bottom integral for a few stations
        z = a * np.cos(w * t0)
        Zcg = hull.Z_rest + z
        for j in (0, 100, 300, 460, 479):
            xj = hull.x[j]
            H = A * np.cos(w * t0 + k * xj)
            zk = hull.zkeel(xj, z)
            gam = H * np.exp(-k * H) if variant == "stretched" else H
            yw = min(max(H - zk, 0.0) / tb, hull.yc[j])
            y = np.linspace(0.0, yw, 200001)
            zz = zk + tb * y
            p = rho * G0 * (-zz + gam * np.exp(k * zz))
            fz_num = 2.0 * np.trapezoid(p, y)
            my_num = 2.0 * np.trapezoid(p * (-hull.kappa * (zz - Zcg) - xj), y)
            # closed form for this station alone
            alpha, beta, gamma, c = -zk, -tb, gam * np.exp(k * zk), k * tb
            I0 = rho * G0 * (alpha * yw + 0.5 * beta * yw ** 2 + gamma * (np.exp(c * yw) - 1.0) / c)
            I1 = rho * G0 * (0.5 * alpha * yw ** 2 + beta * yw ** 3 / 3.0 + gamma * (np.exp(c * yw) * (c * yw - 1.0) + 1.0) / c ** 2)
            fz_cf = 2.0 * I0
            my_cf = 2.0 * (-hull.kappa * (zk - Zcg) * I0 - hull.kappa * tb * I1 - xj * I0)
            worst = max(worst, abs(fz_cf / fz_num - 1.0), abs(my_cf - my_num) / max(abs(my_num), 1.0))
        # transom face numerically
        HT = A * np.cos(w * t0 + k * hull.x_T)
        zkT = hull.zkeel(hull.x_T, z)
        gamT = HT * np.exp(-k * HT) if variant == "stretched" else HT
        zz = np.linspace(zkT, HT, 200001)
        wz = 2.0 * np.minimum((zz - zkT) / tb, hull.yc_T)
        p = rho * G0 * (-zz + gamT * np.exp(k * zz))
        FT_num = np.trapezoid(p * wz, zz)
        MT_num = np.trapezoid(p * wz * (zz - Zcg), zz)
        FT, MT = face_integrals(np.array([zkT]), np.array([HT]), hull.yc_T, tb, k, np.array([gamT]), np.array([Zcg]), rho)
        worst = max(worst, abs(FT[0] / FT_num - 1.0), abs(MT[0] / MT_num - 1.0))
        say(f"  closed form vs 200001-point trapezoid ({variant}): bottom stations and transom, worst relative "
            f"difference so far {worst:.2e}; transom face F_x {FT[0]:.1f} N, M_y {MT[0]:.1f} N m")
    return worst


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    t_start = time.time()
    say(f"S4 radius check: exact hydrostatic + Froude-Krylov force on the instantaneous wetted surface vs its "
        f"second-order Taylor term, Scarab 195 at rest, heave {OMEGA} x head wave {OMEGA} rad/s ({time.strftime('%Y-%m-%d %H:%M')})")
    hull = Hull()
    k = OMEGA ** 2 / G0
    say(f"  hull: L {hull.L} m, chine half-width {hull.yc_T:.3f} m, deadrise tan {hull.tb:.4f}, rest sink of CG {hull.Z_rest:.4f} m, "
        f"trim {np.degrees(hull.Th_rest):.3f} deg (plant sign; keel slope dz_k/dx = {hull.kappa:+.4f}), transom draught "
        f"{-hull.zkeel(hull.x_T, 0.0):.4f} m, stem keel draught {-hull.zkeel(hull.x[-1], 0.0):.4f} m")
    say(f"  wave number k {k:.4f} 1/m, wavelength {2 * np.pi / k:.3f} m; {len(hull.x)} sub-stations of {hull.dx:.5f} m; "
        f"{NT} time steps per period {2 * np.pi / OMEGA:.4f} s")
    T = 2.0 * np.pi / OMEGA
    t = np.linspace(0.0, T, NT, endpoint=False)

    # rest margin of the chine (keel immersion minus chine height) on the plant's 48 stations and the transom
    d0_48 = -hull.zkeel(hull.x48, 0.0)
    margin48 = d0_48 - hull.yc48 * hull.tb
    rel48 = np.abs(np.sin(0.5 * k * hull.x48))          # relative-elevation amplitude per unit delta
    d0_T = -hull.zkeel(hull.x_T, 0.0)
    marginT = d0_T - hull.yc_T * hull.tb
    relT = abs(np.sin(0.5 * k * hull.x_T))
    say("\n== rest geometry at the plant's 48 stations: x from transom (m), keel immersion d0 (m), chine margin d0 - yc tanb (m; "
        "+ = chine under water), relative-elevation amplitude per unit delta |sin(kx/2)|, and the delta at which the chine crosses "
        "= |margin| / |sin(kx/2)| ==")
    cross48 = np.where(rel48 > 1e-9, np.abs(margin48) / np.maximum(rel48, 1e-12), np.inf)
    keel48 = np.where(rel48 > 1e-9, d0_48 / np.maximum(rel48, 1e-12), np.inf)
    say(f"  transom face: d0 {d0_T:.4f}, margin {marginT:+.4f}, |sin| {relT:.3f}, chine crosses at delta {abs(marginT) / relT:.4f} m")
    for j in range(len(hull.x48)):
        say(f"  st {j:2d}  x_T {hull.xi48[j]:5.3f}  d0 {d0_48[j]:.4f}  margin {margin48[j]:+.4f}  |sin| {rel48[j]:.3f}  "
            f"delta_chine {cross48[j]:7.4f}  delta_keel {keel48[j]:7.3f}")

    say("\n== self-check of the closed-form integrals ==")
    worst_cf = brute_check(hull, say)

    # static checks
    Fz0, My0, Fx0, _, _ = forces(hull, t[:1], 0.0, 0.0)
    say(f"\n== rest: exact buoyancy {Fz0[0]:.1f} N vs weight {hull.m * G0:.1f} N ({(Fz0[0] / (hull.m * G0) - 1) * 100:+.2f} %; "
        f"the plant balanced its own 48-station area formula), pitch moment {My0[0]:+.1f} N m (incl. transom and keel-slope "
        f"terms the plant does not carry), surge {Fx0[0]:+.1f} N ==")

    res = dict(omega=OMEGA, k=k, split="a = A = delta/2", deltas=list(DELTAS), delta_ref=list(DELTA_REF), nt=NT,
               n_sub=len(hull.x), dx=hull.dx, closed_form_check=worst_cf, rest=dict(Fz=float(Fz0[0]), My=float(My0[0]),
               Fx=float(Fx0[0]), weight=hull.m * G0, margin48=margin48.tolist(), xT48=hull.xi48.tolist(),
               rel48=rel48.tolist(), marginT=float(marginT), relT=float(relT)), variants={})

    # V2's hf second-order coefficient (per unit a A) for reference
    v2 = None
    pth = os.path.join(OUT, "v2_results_quick.json")
    if os.path.exists(pth):
        R = json.load(open(pth, encoding="utf-8"))["res"]
        v2 = {}
        for p in R["pairs"]:
            if p["kind"] == "mixed" and p["motion"] == "heave" and p["w_motion"] == OMEGA and p["w_wave"] == OMEGA and "quad_hf" in p:
                tot = [complex(*c) for c in p["quad_hf"]["total"]]
                v2["mean" if p["diff"] else "sum"] = tot
        f1 = [r for r in R["first"] if r["omega"] == OMEGA][0]
        v2["T33"], v2["X3"], v2["T53"], v2["X5"] = (abs(complex(*f1["T"]["33"])), abs(complex(*f1["X"]["3"])),
                                                   abs(complex(*f1["T"]["53"])), abs(complex(*f1["X"]["5"])))

    # analytic mixed second-order coefficient of the bottom's vertical force (continuum along x, exact in y):
    # d2 f/(dz dH) per unit length = -(2 rho g / tanb) e^{k z_k}            on a V station (chine dry at rest)
    #                              = +(2 rho g / tanb) e^{k z_k} (e^{k d_ch} - 1) on a chine-wet station (fixed width)
    # z H = (aA/2)[cos(2wt + kx) + cos(kx)]  =>  mean per aA = 1/2 sum coef cos(kx) dx, 2w amplitude = 1/2 sum coef e^{ikx} dx
    zk0 = hull.zkeel(hull.x, 0.0)
    d0f = -zk0
    dchf = hull.yc * hull.tb
    Ek = np.exp(k * zk0)
    coef = np.where(d0f < dchf, -(2 * hull.rho * G0 / hull.tb) * Ek, (2 * hull.rho * G0 / hull.tb) * Ek * (np.exp(k * dchf) - 1.0))
    ana = {}
    say("\n== analytic second-order mixed coefficient (bottom vertical force, per unit a A), by hull region ==")
    for name, msk in (("V stations (chine dry)", d0f < dchf), ("chine-wet aft", (d0f >= dchf) & (hull.x < 0)),
                      ("chine-wet bow (taper)", (d0f >= dchf) & (hull.x > 0)), ("whole hull", np.ones_like(d0f, bool))):
        mF = 0.5 * np.sum((coef * np.cos(k * hull.x))[msk]) * hull.dx
        sF = 0.5 * np.sum((coef * np.exp(1j * k * hull.x))[msk]) * hull.dx
        mM = 0.5 * np.sum((-hull.x * coef * np.cos(k * hull.x))[msk]) * hull.dx
        sM = 0.5 * np.sum((-hull.x * coef * np.exp(1j * k * hull.x))[msk]) * hull.dx
        ana[name] = dict(length=float(msk.sum() * hull.dx), mean_F=float(mF), sum_F=complex(sF), mean_M=float(mM), sum_M=complex(sM))
        say(f"  {name:24s} length {msk.sum() * hull.dx:5.2f} m: heave force mean {mF:+8.0f}, 2w |{abs(sF):6.0f}| N/m^2;  "
            f"pitch moment (vertical force x arm only) mean {mM:+8.0f}, 2w |{abs(sM):6.0f}| N m/m^2")
    res["analytic_c2_bottom"] = ana

    for variant, hull_v in (("stretched", hull), ("plain", hull), ("coarse48", Hull(sub=1))):
        vkey = variant
        var = "stretched" if variant == "coarse48" else variant
        hull_use = hull_v
        say(f"\n==================== variant {vkey} ({len(hull_use.x)} stations, pressure {var}) ====================")

        def incr(a, A):
            FM = lambda a_, A_: np.stack(forces(hull_use, t, a_, A_, var)[:2], 1)
            return FM(a, A) - FM(a, 0.0) - FM(0.0, A) + FM(0.0, 0.0)
        # Taylor coefficient from the smallest amplitude, scaling check
        c2s = []
        for dref in DELTA_REF:
            a_ = A_ = 0.5 * dref
            c2s.append(incr(a_, A_) / (a_ * A_))
        c2 = c2s[0]
        say(f"  second-order coefficient c2(t) = D/(aA) from delta_ref {DELTA_REF[0] * 1e3:.2f} mm (a = A = {0.25 * DELTA_REF[0] * 1e3 * 2:.3f} mm):")
        comp = {}
        for name, Om in (("sum 2w", 2 * OMEGA), ("mean", 0.0)):
            X = [project(t, c2[:, c], Om) for c in range(2)]
            comp[name] = X
            s = f"    {name:>6}: heave force {X[0].real:+9.0f} {X[0].imag:+9.0f}i  |{abs(X[0]):8.0f}| N/m^2;  pitch moment " \
                f"{X[1].real:+9.0f} {X[1].imag:+9.0f}i  |{abs(X[1]):8.0f}| N m/m^2"
            if v2 is not None and variant == "stretched":
                ref = v2["sum"] if Om > 0 else v2["mean"]
                if Om > 0:
                    s += f"   | V2 hf: {ref[0].real:+.0f} {ref[0].imag:+.0f}i |{abs(ref[0]):.0f}|,  {ref[1].real:+.0f} {ref[1].imag:+.0f}i |{abs(ref[1]):.0f}|"
                else:
                    s += f"   | V2 hf (real part = mean): {ref[0].real:+.0f},  {ref[1].real:+.0f}"
            say(s)
        rms_c2 = np.sqrt(np.mean(c2 ** 2, 0))
        for dref, c in zip(DELTA_REF[1:], c2s[1:]):
            dev = np.sqrt(np.mean((c - c2) ** 2, 0)) / rms_c2
            say(f"    scaling check: c2 from delta {dref * 1e3:.1f} mm differs from the {DELTA_REF[0] * 1e3:.1f} mm one by "
                f"RMS {dev[0] * 100:.3f} % (force), {dev[1] * 100:.3f} % (moment)")
        rows = []
        say(f"\n  {'delta':>6} {'a=A':>6} | {'Fz: RMS%':>9} {'sum%':>7} {'mean%':>7} | {'My: RMS%':>9} {'sum%':>7} {'mean%':>7} | "
            f"{'|D2| Fz sum,mean (N)':>21} {'|D| Fz sum,mean':>17} | {'|D2| My sum,mean (N m)':>23} {'|D| My sum,mean':>17} | "
            f"{'D2/1st Fz':>9} {'My':>6} | chine-crossing stations (of 48) / transom / keel-dry stations")
        for delta in DELTAS:
            a_ = A_ = 0.5 * delta
            D = incr(a_, A_)
            D2 = a_ * A_ * c2
            ent = dict(delta=delta, a=a_, A=A_)
            rms = np.sqrt(np.mean((D - D2) ** 2, 0)) / np.sqrt(np.mean(D2 ** 2, 0))
            ent["rms_ratio"] = rms.tolist()
            for name, Om in (("sum", 2 * OMEGA), ("mean", 0.0)):
                X = np.array([project(t, D[:, c], Om) for c in range(2)])
                X2 = np.array([project(t, D2[:, c], Om) for c in range(2)])
                ent[name] = dict(exact=[complex(x) for x in X], second=[complex(x) for x in X2],
                                 ratio=(np.abs(X - X2) / np.abs(X2)).tolist())
            # crossings on the plant's 48 stations: relative elevation at the station over the period
            zeta48 = a_ * (np.cos(OMEGA * t[:, None] + k * hull.x48[None, :]) - np.cos(OMEGA * t)[:, None])
            m48 = margin48[None, :] + zeta48
            cross = np.where((m48.min(0) < 0) & (m48.max(0) > 0))[0]
            keel_dry = np.where((d0_48[None, :] + zeta48).min(0) < 0)[0]
            zT = a_ * (np.cos(OMEGA * t + k * hull.x_T) - np.cos(OMEGA * t))
            mT = marginT + zT
            crossT = bool(mT.min() < 0 < mT.max())
            # on the fine grid: fraction of the hull length where the chine crosses
            _, _, _, dfine, _ = forces(hull_use, t, a_, A_, var)
            mfine = dfine - (hull_use.yc * hull_use.tb)[None, :]
            frac = float(np.mean((mfine.min(0) < 0) & (mfine.max(0) > 0)))
            ent.update(cross_stations=cross.tolist(), cross_x_from_transom=hull.xi48[cross].tolist(), transom_crosses=crossT,
                       keel_dry_stations=keel_dry.tolist(), chine_cross_length_fraction=frac,
                       rel_elev_transom=delta * relT, rel_elev_bow=delta * rel48[-1])
            if v2 is not None:
                f1 = (v2["T33"] * a_ + v2["X3"] * A_, v2["T53"] * a_ + v2["X5"] * A_)
                ent["D2_over_first_order"] = [abs(ent["sum"]["second"][c]) / f1[c] for c in range(2)]
            rows.append(ent)
            s_cross = ",".join(str(i) for i in cross) if len(cross) else "-"
            s_keel = ",".join(str(i) for i in keel_dry) if len(keel_dry) else "-"
            d2f = ent["D2_over_first_order"] if v2 is not None else [float("nan")] * 2
            say(f"  {delta:6.3f} {a_:6.3f} | {rms[0] * 100:8.1f}% {ent['sum']['ratio'][0] * 100:6.1f}% {ent['mean']['ratio'][0] * 100:6.1f}% | "
                f"{rms[1] * 100:8.1f}% {ent['sum']['ratio'][1] * 100:6.1f}% {ent['mean']['ratio'][1] * 100:6.1f}% | "
                f"{abs(ent['sum']['second'][0]):9.1f} {abs(ent['mean']['second'][0]):9.1f} {abs(ent['sum']['exact'][0]):8.1f} {abs(ent['mean']['exact'][0]):8.1f} | "
                f"{abs(ent['sum']['second'][1]):10.1f} {abs(ent['mean']['second'][1]):10.1f} {abs(ent['sum']['exact'][1]):8.1f} {abs(ent['mean']['exact'][1]):8.1f} | "
                f"{d2f[0] * 100:8.2f}% {d2f[1] * 100:5.1f}% | {s_cross} / {'yes' if crossT else 'no'} / {s_keel}  (chine-crossing hull length {frac * 100:.0f} %)")
        res["variants"][vkey] = dict(n_stations=len(hull_use.x), pressure=var, c2=dict(
            sum=[complex(x) for x in comp["sum 2w"]], mean=[complex(x) for x in comp["mean"]]), rows=rows)
    if v2 is not None:
        res["v2_hf"] = dict(sum=v2["sum"], mean=v2["mean"], T33=v2["T33"], X3=v2["X3"], T53=v2["T53"], X5=v2["X5"])

    # verdict (primary variant, RMS and component ratios)
    prim = res["variants"]["stretched"]["rows"]
    by = {r["delta"]: r for r in prim}

    def worst(r):
        return max(max(r["rms_ratio"]), max(r["sum"]["ratio"]), max(r["mean"]["ratio"]))
    say("\n== verdict rule (document section 8): deviation < 10 % at 0.05 m, < 25 % at 0.1 m, > 50 % at 0.3 m; if already > 25 % "
        "at 0.05 m the radius claim is wrong ==")
    for d_, lim, sense in ((0.05, 0.10, "<"), (0.1, 0.25, "<"), (0.3, 0.5, ">")):
        r = by[d_]
        wv = worst(r)
        ok = (wv < lim) if sense == "<" else (wv > lim)
        say(f"  delta {d_}: worst ratio over (Fz, My) x (RMS, sum, mean) = {wv * 100:.1f} % ; predicted {sense} {lim * 100:.0f} % -> "
            f"{'holds' if ok else 'FAILS'}   [Fz RMS {r['rms_ratio'][0] * 100:.1f} %, My RMS {r['rms_ratio'][1] * 100:.1f} %]")
    w05 = worst(by[0.05])
    verdict = ("radius claim falsified: deviation > 25 % already at 0.05 m -> option (iii) demoted to mean force and centimetre amplitudes"
               if w05 > 0.25 else ("radius claim holds at 0.05 m (< 10 %)" if w05 < 0.10 else
                                   "intermediate: 10-25 % at 0.05 m (claim not falsified by the stated rule, but weaker than predicted)"))
    say(f"  => {verdict}")
    res["verdict"] = verdict
    res["t_wall"] = time.time() - t_start
    say(f"\nwall time {res['t_wall']:.1f} s")

    def _json(o):
        if isinstance(o, dict):
            return {str(k_): _json(v) for k_, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_json(v) for v in o]
        if isinstance(o, complex):
            return [o.real, o.imag]
        if isinstance(o, np.ndarray):
            return _json(o.tolist())
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    json.dump(_json(res), open(os.path.join(OUT, "s4_radius.json"), "w", encoding="utf-8"), indent=1)
    open(os.path.join(OUT, "s4_radius.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    plot(res, hull, os.path.join(OUT, "s4_radius.png"))


def plot(res, hull, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C1, C2 = "#2a78d6", "#eb6834"
    rows = res["variants"]["stretched"]["rows"]
    d = np.array([r["delta"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    for c, col, lab in ((0, C1, "heave force"), (1, C2, "pitch moment")):
        a.plot(d, [r["rms_ratio"][c] * 100 for r in rows], "-o", color=col, lw=2, ms=6, label=f"{lab}, RMS over period")
        a.plot(d, [r["sum"]["ratio"][c] * 100 for r in rows], "--s", color=col, lw=1.2, ms=5, alpha=0.7, label=f"{lab}, 2w component")
        a.plot(d, [r["mean"]["ratio"][c] * 100 for r in rows], ":^", color=col, lw=1.2, ms=5, alpha=0.7, label=f"{lab}, mean")
    for x_, y_ in ((0.05, 10), (0.1, 25), (0.3, 50)):
        a.plot([x_], [y_], "x", color="#555", ms=9, mew=2)
    a.text(0.052, 10, " predicted < 10 %", fontsize=8, color="#555", va="center")
    a.text(0.104, 25, " < 25 %", fontsize=8, color="#555", va="center")
    a.text(0.31, 50, " > 50 %", fontsize=8, color="#555", va="center")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("relative-elevation amplitude delta = a + A (m), a = A")
    a.set_ylabel("deviation from the second-order term (%)")
    a.set_title("Deviation of the exact double-input increment\nfrom its second-order Taylor term (HS + FK, heave 3.9 x wave 3.9)", fontsize=10)
    a.grid(True, which="both", color="#e5e5e5", lw=0.6)
    a.legend(fontsize=7, loc="lower right")
    b = ax[1]
    xi = np.array(res["rest"]["xT48"])
    margin = np.array(res["rest"]["margin48"])
    rel = np.array(res["rest"]["rel48"])
    cross = np.where(rel > 1e-9, np.abs(margin) / np.maximum(rel, 1e-12), np.nan)
    b.plot(xi, margin, "-", color="#555", lw=1.5, label="chine margin at rest (m; + = chine under water)")
    b.plot(xi, np.where(cross < 1, cross, np.nan), "-", color=C1, lw=2, label="delta at which the waterline crosses the chine (m)")
    for dd in (0.01, 0.03, 0.05, 0.1, 0.3):
        b.axhline(dd, color="#bbb", lw=0.6, ls="--")
        b.text(xi[-1], dd, f" {dd}", fontsize=7, color="#888", va="center")
    b.axhline(0, color="#999", lw=0.6)
    b.set_xlabel("x from the transom (m)")
    b.set_ylabel("m")
    b.set_ylim(-0.12, 0.45)
    b.set_title("Where the non-analytic point sits along the hull\n(48 plant stations; CG at x = 2.18 m has no relative motion)", fontsize=10)
    b.grid(True, color="#e5e5e5", lw=0.6)
    b.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
