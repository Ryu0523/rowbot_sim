#!/usr/bin/env python3
"""
V1 of learn/meta/HULL_FORCE_AUDIT_REVIEW_2026-10-09.md section 4: linear
baseline consistency of the corrected composite D1'

    F = F_HF + sum f^NL dx + [ F^{3D,lin} - sum f^lin dx ] + friction

at zero speed, small amplitude, for scarab195 at rest (ledger rows
B1.1-B1.12 of learn/meta/SIM_DATA_LEDGER_2026-10-08.md).

Read-only use of sim/ and hydro/ (another agent edits them concurrently):
the strip's dynamic section force is re-assembled here from the vessel's
own `_section` law and the formulas of ZarnickVessel.deriv (ZOPT_ZR path:
normal=False, mom='entry', cf='abs', chine='frozen', bm=1), and checked
against `deriv` to round-off before anything else is done (check 0).

Parts
  1  linearisation check of the strip's dynamic section force f^NL (no
     buoyancy): numerical Jacobian f^lin about the rest state, prescribed
     heave and pitch of amplitude 2/1/0.5/0.25 cm, residual f^NL - f^lin
     against a^2; linear coefficients (added mass, damping, stiffness) per
     term, Wagner (ZR) and MLM sections
  2  2-DOF heave-pitch time-domain model at U = 0 with exact sectional
     hydrostatics (exact lever arm), the BEM excitation of the probe table
     (Froude-Krylov + diffraction, head seas), the BEM radiation memory as
     state-space fits (hydro/retardation.fit_coefficient, as B1.12) minus
     the strip's own linear limit, plus the strip's f^NL; three configs:
     D1' / plain strip / pure BEM LTI; regular head waves a = 1 cm at the
     seven B1.3 frequencies; heave decay from 1 cm
  3  the strip linearised about the RUNNING state at 5 kn (first-order
     terms of the momentum and cross-flow terms) -- the 3-D forward-speed
     part of V1 is blocked (no scarab195 forward-speed table, no excitation
     in hydro/rankine/tables.py); see the write-up

Outputs: studies/_cache/derive/v1_linear_consistency.txt, v1_results.json,
v1_scaling.png, v1_rao.png. Write-up: v1_progress_zh.md (by hand).

Run: cd <repo> && .venv/Scripts/python.exe -m studies.derive_v1_linear_consistency
"""
import json
import math
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_cache", "derive")
os.makedirs(OUT, exist_ok=True)
G = 9.81
SP = -1.0                      # sim/vessel.SIGN_PITCH (dz = -x theta)
KNOT = 0.514444
RHO_FALLBACK = 1025.0
# B1.3 reference values (BEM, rest_rao.txt)
W_B13 = np.array([1.35, 2.20, 3.04, 3.89, 4.74, 5.59, 7.28])
HEAVE_B13 = np.array([0.967, 0.775, 0.435, 0.282, 0.142, 0.061, 0.015])
PITCH_B13 = np.array([0.999, 0.875, 0.528, 0.155, 0.064, 0.017, 0.002])
C_BEM = dict(C33=87738.0, C55=197509.0, C35=-19658.0)       # B1.1

_LINES = []


def say(s=""):
    print(s, flush=True)
    _LINES.append(s)


# --------------------------------------------------------------- vessels
def make_vessel(mlm, u=0.0):
    from hydro.hulls import scarab_195
    from sim.planing_zarnick import ZarnickVessel, ZOPT_ZR
    from sim.test_vessel import Monochromatic
    h = scarab_195()
    zopt = dict(ZOPT_ZR, mlm=bool(mlm))
    return ZarnickVessel(h, Monochromatic(1.0, 0.0), dt=0.005, zopt=zopt,
                         captive_u=float(u))


class Strip:
    """The strip's forces at surge speed u, split into the dynamic section
    force f^NL (added-mass, momentum, cross-flow; NO buoyancy) and the
    buoyancy, each per station (N/m), from the vessel's own `_section`."""

    def __init__(self, v, u=0.0):
        self.v, self.u = v, float(u)
        self.x, self.dx, self.rho = v.x, v.dx, v.rho
        self.ctr = np.broadcast_to(np.asarray(v._ctr(self.u), float), v.x.shape)
        self.ab = float(v._a_b(self.u))
        self.n = len(v.x)

    def immersion(self, z, th):
        v = self.v
        Th = v.Th_rest + th
        d = -(v.Z_rest + z + v.zk + SP * v.x * Th)
        return d, Th

    def dyn(self, xi):
        """xi = (z, th, w, q, wd, qd). Returns f (N/m per station) and the
        per-term dict (all N/m per station)."""
        z, th, w, q, wd, qd = xi
        v, u, x = self.v, self.u, self.x
        d, Th = self.immersion(z, th)
        tau = v.kp + SP * Th
        V = -(w + SP * x * q) + u * tau
        A, c, ma, dma, girth = v._section(d)
        ma = ma * self.ctr
        dma = dma * self.ctr
        mom = np.where(V > 0.0, dma * V * V, 0.0)
        cf = 0.5 * self.rho * v.c_dc * 2.0 * c * V * np.abs(V) * self.ctr
        iner = -ma * (wd + SP * x * qd)
        cor = ma * (2.0 * SP * u * q - u * u * v.kpp)
        f = iner + cor + mom + cf
        return f, dict(iner=iner, cor=cor, mom=mom, cf=cf, ma=ma, V=V, d=d)

    def added_mass_matrix(self, z, th):
        d, _ = self.immersion(z, th)
        ma = self.v._section(d)[2] * self.ctr
        x, dx = self.x, self.dx
        return np.array([[ma.sum(), (ma * SP * x).sum()],
                         [(ma * SP * x).sum(), (ma * x * x).sum()]]) * dx

    def buoy(self, z, th):
        d, _ = self.immersion(z, th)
        A = self.v._section(d)[0]
        return self.ab * self.rho * G * A * self.ctr

    def sum_force_moment(self, f):
        return np.array([f.sum(), (SP * self.x * f).sum()]) * self.dx


def exact_hydrostatics(strip, z, th):
    """Hydrostatic force and moment on the geometric wetted V sections for
    pitch about the CG: exact immersion of each body section (the water
    surface cuts the plane x = const in a horizontal line at body height
    (x sin Th - Z - z) / cos Th), exact lever arm (earth-horizontal offset
    x cos Th + z_c sin Th of each section's area centroid from the CG).
    The strip's own law (Strip.immersion) is the small-angle form of the
    same immersion. Returns (Fz - m g, My)."""
    v = strip.v
    Th = v.Th_rest + th
    d = (v.x * math.sin(Th) - v.Z_rest - z) / math.cos(Th) - v.zk
    tb, yc = v.tanb, v.yc
    dp = np.maximum(d, 0.0)
    dch = yc * tb
    below = np.minimum(dp, dch)
    A1 = below ** 2 / tb
    A2 = 2.0 * yc * np.maximum(dp - dch, 0.0)
    zc1 = v.zk + 2.0 / 3.0 * below
    zc2 = v.zk + dch + 0.5 * np.maximum(dp - dch, 0.0)
    A = A1 + A2
    zc = np.where(A > 0, (A1 * zc1 + A2 * zc2) / np.maximum(A, 1e-30), v.zk)
    F = strip.rho * G * A
    lever = v.x * math.cos(Th) + zc * math.sin(Th)
    Fz = F.sum() * strip.dx - v.m * G
    My = -(lever * F).sum() * strip.dx
    return np.array([Fz, My])


def jacobian(fun, x0, steps):
    """Central-difference Jacobian of fun (vector) at x0."""
    f0 = np.asarray(fun(x0))
    J = np.zeros((f0.size, len(x0)))
    for i, h in enumerate(steps):
        xp = np.array(x0, float)
        xm = xp.copy()
        xp[i] += h
        xm[i] -= h
        J[:, i] = (np.asarray(fun(xp)) - np.asarray(fun(xm))) / (2.0 * h)
    return J, f0


STEPS = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
TERMS = ("iner", "cor", "mom", "cf")


# ----------------------------------------------------- check 0: consistency
def check_deriv_consistency(strip, u, label):
    """My re-assembled strip forces against the vessel's own deriv."""
    v = strip.v
    worst = 0.0
    rng = np.random.default_rng(1)
    for _ in range(5):
        z, th = rng.normal(0, 0.02), rng.normal(0, 0.01)
        w, q = rng.normal(0, 0.1), rng.normal(0, 0.05)
        s = np.zeros(14)
        s[2], s[4], s[6], s[8], s[10] = z, th, u, w, q
        ds, aux = v.deriv(s, 0.0)
        wd, qd = ds[8], ds[10]
        f, _ = strip.dyn((z, th, w, q, wd, qd))
        b = strip.buoy(z, th)
        FM = strip.sum_force_moment(f + b)
        FM[0] -= v.m * G
        # the vessel's air drag acts 0.5 m above the CG (deriv: _load(-Da,
        # ..., z=0.5)) -> pitch moment -0.5 Da; friction and tow act at the
        # CG in ZOPT_ZR; the vessel's own buoyancy moment uses the strip
        # lever (SP x)
        Da = 0.5 * 1.225 * v.hull.cda * u * abs(u)
        FM[1] -= 0.5 * Da
        res = np.array([FM[0] - v.m * wd, FM[1] - v.Iy * qd])
        scale = np.array([v.m * G, v.m * G * v.L])
        worst = max(worst, float(np.max(np.abs(res) / scale)))
    say(f"  check 0 ({label}, u={u:.2f} m/s): re-assembled strip force vs "
        f"deriv, worst relative residual {worst:.1e} (must be ~1e-12)")
    return worst


# ------------------------------------------------- part 1: linearisation
def part1_linearisation(strip, label, x0, om, amps, res):
    """Jacobian about x0 and the a^2 scaling of f^NL - f^lin for prescribed
    heave and pitch at frequency om."""
    say(f"\n== part 1: linearisation of the strip's dynamic section force "
        f"({label}) ==")
    x0 = np.array(x0, float)
    Jall, f0 = jacobian(lambda xi: strip.dyn(xi)[0], x0, STEPS)
    J10, _ = jacobian(lambda xi: strip.dyn(xi)[0], x0, STEPS * 10)
    say(f"  Jacobian step check: max |J(h) - J(10h)| / max|J| = "
        f"{np.max(np.abs(Jall - J10)) / np.max(np.abs(Jall)):.1e}")
    Jterm = {k: jacobian(lambda xi, k=k: strip.dyn(xi)[1][k], x0, STEPS)[0]
             for k in TERMS}
    x, dx = strip.x, strip.dx
    lever = np.stack([np.ones_like(x), SP * x])          # (2, n): Fz, My

    def mats(J):
        # generalised 2x2 matrices: -dF/d(acc), -dF/d(vel), -dF/d(pos)
        Fz_My = lever @ J * dx                           # (2, 6)
        return dict(A=-Fz_My[:, 4:6], B=-Fz_My[:, 2:4], C=-Fz_My[:, 0:2])

    out = dict(label=label, x0=x0.tolist(), terms={})
    m_all = mats(Jall)
    for key, M in m_all.items():
        say(f"  total {key} (rows Fz, My; cols heave, pitch) =\n"
            f"    {M[0, 0]:12.4g} {M[0, 1]:12.4g}\n    {M[1, 0]:12.4g} {M[1, 1]:12.4g}")
        out[key] = M.tolist()
    for k in TERMS:
        mk = mats(Jterm[k])
        out["terms"][k] = {kk: vv.tolist() for kk, vv in mk.items()}
        say(f"  term {k:5s}: |A|max {np.max(np.abs(mk['A'])):.4g}  "
            f"|B|max {np.max(np.abs(mk['B'])):.4g}  |C|max {np.max(np.abs(mk['C'])):.4g}")
    d0 = strip.immersion(x0[0], x0[1])[0]
    _, aux0 = strip.dyn(x0)
    say(f"  rest immersion: transom {d0[0]:.3f} m, bow station {d0[-1]:.3f} m; "
        f"V0 (steady relative velocity) min {aux0['V'].min():+.4f} max "
        f"{aux0['V'].max():+.4f} m/s")
    # scaling: prescribed heave, then pitch (bow displacement = a)
    T = 2 * np.pi / om
    tt = np.linspace(0.0, T, 241)
    scal = {}
    for mode in ("heave", "pitch"):
        rows = []
        for a in amps:
            amp = a if mode == "heave" else a / float(x[-1])
            fmax = rmax = 0.0
            tmax = {k: 0.0 for k in TERMS}
            FMmax = np.zeros(2)
            for t in tt:
                s, c = math.sin(om * t), math.cos(om * t)
                dxi = np.zeros(6)
                i = 0 if mode == "heave" else 1
                dxi[i], dxi[i + 2], dxi[i + 4] = amp * s, amp * om * c, -amp * om * om * s
                xi = x0 + dxi
                f, aux = strip.dyn(xi)
                flin = f0 + Jall @ dxi
                r = f - flin
                fmax = max(fmax, float(np.max(np.abs(f - f0))))
                rmax = max(rmax, float(np.max(np.abs(r))))
                FMmax = np.maximum(FMmax, np.abs(strip.sum_force_moment(r)))
                for k in TERMS:
                    rk = aux[k] - aux0[k] - Jterm[k] @ dxi
                    tmax[k] = max(tmax[k], float(np.max(np.abs(rk))))
            rows.append(dict(a=a, f_max=fmax, resid_max=rmax,
                             resid_F=float(FMmax[0]), resid_M=float(FMmax[1]),
                             resid_terms=tmax))
        say(f"  prescribed {mode} at {om:.2f} rad/s (amplitude = "
            f"{'heave' if mode == 'heave' else 'bow displacement'}):")
        say(f"    {'a cm':>5}{'max|f-f0|':>11}{'max|res|':>10}{'res/a^2':>10}"
            f"{'resF N':>9}{'resM Nm':>9}  per-term residual: "
            + " ".join(f"{k:>8}" for k in TERMS))
        for r in rows:
            say(f"    {100 * r['a']:5.2f}{r['f_max']:11.4g}{r['resid_max']:10.3g}"
                f"{r['resid_max'] / r['a'] ** 2:10.4g}{r['resid_F']:9.3g}"
                f"{r['resid_M']:9.3g}  "
                + " ".join(f"{r['resid_terms'][k]:8.2g}" for k in TERMS))
        slopes = [math.log(rows[i]['resid_max'] / rows[i + 1]['resid_max'])
                  / math.log(rows[i]['a'] / rows[i + 1]['a'])
                  for i in range(len(rows) - 1) if rows[i + 1]['resid_max'] > 0]
        say(f"    scaling exponent of max|f^NL - f^lin| between successive "
            f"amplitudes: {', '.join(f'{s:.2f}' for s in slopes)} (2 = quadratic)")
        scal[mode] = dict(rows=rows, slopes=slopes)
    out["scaling"] = scal
    res[label] = out
    return Jall, out


# --------------------------------------------------- part 2: time domain
def load_bem():
    z = np.load(os.path.join(HERE, "_cache", "anyspeed",
                             "bem_probe_lid_24_64.npz"))
    om, A, B = z["omega"], z["A"], z["B"]
    F = z["F_re"] + 1j * z["F_im"]
    FK = z["FK_re"] + 1j * z["FK_im"]
    return dict(omega=om, A=A, B=B, F=F, FK=FK, directions=z["directions"])


def fit_memory(bem):
    from hydro.retardation import fit_coefficient
    om, A, B = bem["omega"], bem["A"], bem["B"]
    fits = {}
    for key, (i, j) in dict(hh=(2, 2), pp=(4, 4), hp=(2, 4)).items():
        Aij = 0.5 * (A[:, i, j] + A[:, j, i])
        Bij = 0.5 * (B[:, i, j] + B[:, j, i])
        fits[key] = fit_coefficient(om, Aij, Bij)
    say("  memory fits (hydro/retardation.fit_coefficient, order auto):")
    for k, f in fits.items():
        say(f"    {k}: order {f['order']}, err_K {f['err_K']:.3f}, err_B "
            f"{f['err_B']:.3f}, err_A {f['err_A']:.3f}, A_inf {f['A_inf']:.0f}, "
            f"max Re eig {f['max_real_eig']:.2f} 1/s, memory {f['memory_time']:.2f} s")
    return fits


class Memory2:
    """Block state space for the heave-pitch pair: mu = C x, x' = A x + B nu,
    nu = (w, q), mu = (heave force, pitch moment); the hp fit used twice."""

    def __init__(self, fits):
        blocks = [("hh", 0, 0), ("pp", 1, 1), ("hp", 0, 1), ("hp", 1, 0)]
        n = sum(len(fits[k]["Ar"]) for k, _, _ in blocks)
        self.A = np.zeros((n, n))
        self.B = np.zeros((n, 2))
        self.C = np.zeros((2, n))
        k0 = 0
        for key, i, j in blocks:
            f = fits[key]
            m = len(f["Ar"])
            self.A[k0:k0 + m, k0:k0 + m] = f["Ar"]
            self.B[k0:k0 + m, j] = f["Br"]
            self.C[i, k0:k0 + m] = f["Cr"]
            k0 += m
        self.n = n
        self.A_inf = np.array([[fits["hh"]["A_inf"], fits["hp"]["A_inf"]],
                               [fits["hp"]["A_inf"], fits["pp"]["A_inf"]]])
        self.max_eig = float(np.max(np.abs(np.linalg.eigvals(self.A))))


class Model2:
    """2-DOF heave-pitch model, surge held, U = 0.
    use_strip: + sum f^NL dx (dynamic strip force, nonlinear)
    use_bem:   + BEM radiation (A_inf + memory)
    both:      the bracket is [BEM - strip linear limit]: the strip's
               linear part (Jacobian about rest) is subtracted."""

    def __init__(self, strip, mem, J, use_strip, use_bem, Fexc=None, om=0.0):
        self.s, self.mem, self.J = strip, mem, J
        self.use_strip, self.use_bem = use_strip, use_bem
        self.sub = use_strip and use_bem
        v = strip.v
        self.M = np.diag([v.m, v.Iy])
        self.Fexc = Fexc           # complex (2,) for e^{-i om t}, N and N m
        self.om = om
        x, dx = strip.x, strip.dx
        lever = np.stack([np.ones_like(x), SP * x])
        FM = lever @ J * dx                              # (2, 6)
        self.A_lin = -FM[:, 4:6]
        self.K_lin = FM[:, 0:4]                           # on (z, th, w, q)
        self.x0 = np.zeros(6)
        self.f0 = strip.dyn(self.x0)[0]
        self.nm = mem.n if use_bem else 0

    def rhs(self, t, y):
        z, th, w, q = y[:4]
        xm = y[4:]
        s = self.s
        F = exact_hydrostatics(s, z, th)
        if self.Fexc is not None:
            F = F + np.real(self.Fexc * np.exp(-1j * self.om * t))
        Mass = self.M.copy()
        dxm = np.zeros(0)
        if self.use_bem:
            Mass = Mass + self.mem.A_inf
            F = F - self.mem.C @ xm
            dxm = self.mem.A @ xm + self.mem.B @ np.array([w, q])
        if self.use_strip:
            xi = np.array([z, th, w, q, 0.0, 0.0])
            f, _ = s.dyn(xi)                 # non-inertial part (acc = 0)
            F = F + s.sum_force_moment(f)
            Mass = Mass + s.added_mass_matrix(z, th)
        if self.sub:
            # minus the strip's own linear limit (about rest): the
            # non-inertial linear part and the linear added mass
            F = F - s.sum_force_moment(self.f0) - self.K_lin @ np.array([z, th, w, q])
            Mass = Mass - self.A_lin
        acc = np.linalg.solve(Mass, F)
        return np.concatenate([[w, q], acc, dxm])

    def run(self, y0, t_end, dt):
        n = int(round(t_end / dt))
        y = np.array(y0, float)
        t = 0.0
        rec = np.empty((n + 1, 4))
        rec[0] = y[:4]
        for i in range(n):
            k1 = self.rhs(t, y)
            k2 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = self.rhs(t + dt, y + dt * k3)
            y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            rec[i + 1] = y[:4]
        return np.arange(n + 1) * dt, rec


def amp_phase(t, y, om, n_last):
    T = 2 * np.pi / om
    m = t >= t[-1] - n_last * T
    Aa = np.column_stack([np.cos(om * t[m]), np.sin(om * t[m]), np.ones(m.sum())])
    c, *_ = np.linalg.lstsq(Aa, y[m], rcond=None)
    return float(np.hypot(c[0], c[1]))


def fd_rao(om_tab, A, B, C, M, F, idx, w):
    k = int(np.argmin(np.abs(om_tab - w)))
    w = om_tab[k]
    Z = -w ** 2 * (M + A[k][np.ix_(idx, idx)]) - 1j * w * B[k][np.ix_(idx, idx)] + C
    return np.linalg.solve(Z, F[k][idx]), w


def decay_stats(t, z):
    """Damped frequency and damping ratio from successive extrema of a
    decay record (half-cycle log decrement), first three half cycles."""
    ext = []
    for i in range(1, len(z) - 1):
        if (z[i] - z[i - 1]) * (z[i + 1] - z[i]) < 0:
            ext.append((t[i], z[i]))
    if len(ext) < 3:
        return float("nan"), float("nan"), ext
    # zero crossings
    zc = t[:-1][np.sign(z[:-1]) != np.sign(z[1:])]
    wd = math.pi / np.mean(np.diff(zc[:4])) if len(zc) >= 3 else float("nan")
    zetas = []
    for i in range(min(3, len(ext) - 1)):
        r = abs(ext[i][1]) / abs(ext[i + 1][1])
        if r > 1:
            dh = math.log(r)
            zetas.append(dh / math.sqrt(math.pi ** 2 + dh ** 2))
    return float(wd), float(np.mean(zetas)) if zetas else float("nan"), ext


def plant_reference(mlm, om, a, periods=40, dt=0.005):
    """The actual ZarnickVessel plant at captive u=0 in its own regular
    head wave (its own excitation), as rest_rao.py did for v0."""
    from sim.test_vessel import Monochromatic
    from hydro.hulls import scarab_195
    from sim.planing_zarnick import ZarnickVessel, ZOPT_ZR
    sea = Monochromatic(om, a, theta0=np.pi)
    v = ZarnickVessel(scarab_195(), sea, dt=dt, zopt=dict(ZOPT_ZR, mlm=bool(mlm)),
                      captive_u=0.0)
    s = np.zeros(14)
    T = 2 * np.pi / om
    n = int(periods * T / dt)
    t = np.arange(n) * dt
    rec = np.empty((n, 2))
    for i in range(n):
        s = v.step(s, t[i], 0.0, 0.0)
        s[6] = 0.0
        rec[i] = s[2], s[4]
    return amp_phase(t, rec[:, 0], om, 10), amp_phase(t, rec[:, 1], om, 10)


def part2_time_domain(strips, Js, bem, res, periods=25, dt=0.005, a=0.01):
    say("\n== part 2: zero-speed heave-pitch time-domain model (a = 1 cm, head seas) ==")
    t0 = time.time()
    fits = fit_memory(bem)
    mem = Memory2(fits)
    say(f"  memory state space: {mem.n} states, |eig|max {mem.max_eig:.1f} 1/s "
        f"(RK4 dt {dt} s: |eig| dt = {mem.max_eig * dt:.2f})")
    om_tab, A, B, F = bem["omega"], bem["A"], bem["B"], bem["F"]
    dirs = bem["directions"]
    i_head = int(np.argmin(np.abs(dirs - np.pi)))
    i_follow = int(np.argmin(np.abs(dirs - 0.0)))
    s0 = strips["ZR"]
    v = s0.v
    M = np.diag([v.m, v.Iy])
    idx = [2, 4]
    # hydrostatic stiffness of the exact sectional hydrostatics vs BEM C
    Chs = -jacobian(lambda p: exact_hydrostatics(s0, p[0], p[1]),
                    np.zeros(2), [1e-6, 1e-6])[0]
    Cbem = np.array([[C_BEM["C33"], C_BEM["C35"]], [C_BEM["C35"], C_BEM["C55"]]])
    say(f"  hydrostatic stiffness, exact sections (this model): C33 {Chs[0, 0]:.0f}, "
        f"C55 {Chs[1, 1]:.0f}, C35 {Chs[0, 1]:.0f} / {Chs[1, 0]:.0f}; BEM (B1.1): "
        f"{Cbem[0, 0]:.0f}, {Cbem[1, 1]:.0f}, {Cbem[0, 1]:.0f}; differences "
        f"{100 * (Chs[0, 0] / Cbem[0, 0] - 1):+.2f} %, {100 * (Chs[1, 1] / Cbem[1, 1] - 1):+.2f} %, "
        f"{100 * (Chs[0, 1] / Cbem[0, 1] - 1):+.2f} %")
    res["hydrostatics"] = dict(C_exact=Chs.tolist(), C_bem=Cbem.tolist())
    res["memory_fit"] = {k: dict(order=f["order"], err_K=f["err_K"], err_B=f["err_B"],
                                 err_A=f["err_A"], A_inf=f["A_inf"]) for k, f in fits.items()}
    res["memory_states"] = mem.n
    # the bracket's added-mass part
    for lab, s in strips.items():
        Al = Model2(s, mem, Js[lab], True, True).A_lin
        say(f"  {lab}: strip linear added mass A33 {Al[0, 0]:.0f} kg, A55 {Al[1, 1]:.0f} kg m2, "
            f"A35 {Al[0, 1]:.0f}; BEM A_inf 33/55/35 {mem.A_inf[0, 0]:.0f}/{mem.A_inf[1, 1]:.0f}/"
            f"{mem.A_inf[0, 1]:.0f}; bracket A_inf - A_lin: {mem.A_inf[0, 0] - Al[0, 0]:.0f} kg heave, "
            f"{mem.A_inf[1, 1] - Al[1, 1]:.0f} kg m2 pitch")
        res.setdefault("bracket_added_mass", {})[lab] = dict(
            A_lin=Al.tolist(), A_inf=mem.A_inf.tolist())
        # where does the effective heave mass of D1' vanish (frozen bracket)?
        zz = np.linspace(0.0, 0.5, 501)
        meff = [v.m + mem.A_inf[0, 0] - Al[0, 0] + s.added_mass_matrix(z, 0.0)[0, 0] for z in zz]
        meff = np.array(meff)
        iz = np.where(meff <= 0)[0]
        z_zero = float(zz[iz[0]]) if len(iz) else float("nan")
        say(f"    effective heave mass of D1' (frozen bracket) at rest {meff[0]:.0f} kg; "
            f"it reaches zero at a heave rise of {z_zero:.3f} m (rest transom draught "
            f"{s.immersion(0, 0)[0][0]:.3f} m)")
        res["bracket_added_mass"][lab]["z_mass_zero"] = z_zero

    # FD references
    say(f"\n  frequency-domain linear RAO (BEM A, B; F = FK + diffraction):")
    say(f"  {'w':>5}{'heave BEM-C head':>17}{'follow':>8}{'exact-C head':>13}"
        f"{'A/B fit':>9}{'B1.3':>7} | {'pitch head':>11}{'follow':>8}{'exact-C':>9}"
        f"{'fit':>7}{'B1.3':>7}")
    fd = {}
    for i, w in enumerate(W_B13):
        xh, wk = fd_rao(om_tab, A, B, Cbem, M, F[:, i_head, :], idx, w)
        xf, _ = fd_rao(om_tab, A, B, Cbem, M, F[:, i_follow, :], idx, w)
        xe, _ = fd_rao(om_tab, A, B, Chs, M, F[:, i_head, :], idx, w)
        # fit-reproduced A, B (what the state space represents)
        k = int(np.argmin(np.abs(om_tab - w)))
        Af = np.array([[fits["hh"]["A_fit"][k], fits["hp"]["A_fit"][k]],
                       [fits["hp"]["A_fit"][k], fits["pp"]["A_fit"][k]]])
        Bf = np.array([[fits["hh"]["B_fit"][k], fits["hp"]["B_fit"][k]],
                       [fits["hp"]["B_fit"][k], fits["pp"]["B_fit"][k]]])
        Zf = -wk ** 2 * (M + Af) - 1j * wk * Bf + Chs
        xfit = np.linalg.solve(Zf, F[k, i_head, idx])
        kw = wk ** 2 / G
        fd[float(wk)] = dict(heave_bemC=abs(xh[0]), pitch_bemC=abs(xh[1]) / kw,
                             heave_follow=abs(xf[0]), pitch_follow=abs(xf[1]) / kw,
                             heave_exactC=abs(xe[0]), pitch_exactC=abs(xe[1]) / kw,
                             heave_fit=abs(xfit[0]), pitch_fit=abs(xfit[1]) / kw)
        d = fd[float(wk)]
        say(f"  {wk:5.2f}{d['heave_bemC']:17.3f}{d['heave_follow']:8.3f}{d['heave_exactC']:13.3f}"
            f"{d['heave_fit']:9.3f}{HEAVE_B13[i]:7.3f} | {d['pitch_bemC']:11.3f}"
            f"{d['pitch_follow']:8.3f}{d['pitch_exactC']:9.3f}{d['pitch_fit']:7.3f}{PITCH_B13[i]:7.3f}")
    res["fd"] = fd
    say("  (B1.3's BEM column equals the direction-0 column: rest_rao.py took "
        "F[:, 0, :] = direction 0 = waves travelling +x = FOLLOWING seas for this "
        "hull, while its plant run used theta0 = pi = head seas.)")

    # time domain
    configs = [("D1'", True, True), ("plain strip", True, False),
               ("pure BEM LTI", False, True)]
    say(f"\n  time domain, RK4 dt {dt} s, {periods} periods, last 8 fitted; "
        f"excitation = BEM F (FK + diffraction) head seas, a = {a} m")
    td = {}
    for lab, s in strips.items():
        for cname, us, ub in configs:
            if not us and lab != "ZR":
                continue
            key = f"{cname}" if not us else f"{cname} {lab}"
            td[key] = {}
            say(f"  --- {key} ---")
            say(f"  {'w':>5}{'heave/a':>9}{'FD exact-C':>11}{'diff %':>8}{'B1.3':>7}{'diff %':>8}"
                f" | {'pitch/ka':>9}{'FD exact-C':>11}{'diff %':>8}{'B1.3':>7}{'diff %':>8}")
            for i, w in enumerate(W_B13):
                k = int(np.argmin(np.abs(om_tab - w)))
                wk = float(om_tab[k])
                Fe = F[k, i_head, idx] * a
                mdl = Model2(s, mem, Js[lab], us, ub, Fexc=Fe, om=wk)
                y0 = np.zeros(4 + mdl.nm)
                T = 2 * np.pi / wk
                t, rec = mdl.run(y0, periods * T, dt)
                hz = amp_phase(t, rec[:, 0], wk, 8) / a
                pz = amp_phase(t, rec[:, 1], wk, 8) / (wk ** 2 / G * a)
                ref = fd[wk]
                td[key][wk] = dict(heave=hz, pitch=pz)
                say(f"  {wk:5.2f}{hz:9.3f}{ref['heave_exactC']:11.3f}"
                    f"{100 * (hz / ref['heave_exactC'] - 1):+8.1f}{HEAVE_B13[i]:7.3f}"
                    f"{100 * (hz / HEAVE_B13[i] - 1):+8.1f} | {pz:9.3f}{ref['pitch_exactC']:11.3f}"
                    f"{100 * (pz / ref['pitch_exactC'] - 1):+8.1f}{PITCH_B13[i]:7.3f}"
                    f"{100 * (pz / PITCH_B13[i] - 1):+8.1f}")
    res["td"] = {k: {str(w): d for w, d in v.items()} for k, v in td.items()}

    # the actual plant with its own excitation (as B1.3's plant column)
    say(f"\n  the ZarnickVessel plant itself at captive u=0, its own head wave a = 1 cm "
        f"(long-wave FK via sectional buoyancy, ma*eta_tt, V with eta_t; 40 periods):")
    say(f"  {'w':>5}{'heave ZR':>9}{'MLM':>7}{'B1.3 v0':>8} | {'pitch ZR':>9}{'MLM':>7}{'B1.3 v0':>8}")
    v0h = [1.003, 0.870, 2.458, 0.075, 0.106, 0.073, 0.041]
    v0p = [1.071, 1.027, 1.040, 0.191, 0.049, 0.008, 0.006]
    plant = {}
    for i, w in enumerate(W_B13):
        k = int(np.argmin(np.abs(om_tab - w)))
        wk = float(om_tab[k])
        hz, pz = plant_reference(False, wk, a)
        hm, pm = plant_reference(True, wk, a)
        kw = wk ** 2 / G
        plant[wk] = dict(ZR=(hz / a, pz / (kw * a)), MLM=(hm / a, pm / (kw * a)))
        say(f"  {wk:5.2f}{hz / a:9.3f}{hm / a:7.3f}{v0h[i]:8.3f} | {pz / (kw * a):9.3f}"
            f"{pm / (kw * a):7.3f}{v0p[i]:8.3f}")
    res["plant_own_excitation"] = {str(w): d for w, d in plant.items()}

    # decay
    say(f"\n  heave decay from a 1 cm offset (no waves), damped frequency and damping "
        f"ratio from the first half cycles:")
    # BEM linear reference: iterate wn^2 = C33/(m + A33(wn))
    wn = 3.9
    for _ in range(20):
        A33 = np.interp(wn, om_tab, A[:, 2, 2])
        wn = math.sqrt(Chs[0, 0] / (v.m + A33))
    B33 = np.interp(wn, om_tab, B[:, 2, 2])
    zeta_fd = B33 / (2 * wn * (v.m + np.interp(wn, om_tab, A[:, 2, 2])))
    say(f"    BEM frequency-domain (uncoupled heave): wn {wn:.3f} rad/s, zeta {zeta_fd:.3f} "
        f"(ledger B1.2/B1.10: 3.94, 0.297)")
    dec = dict(fd=dict(wn=wn, zeta=zeta_fd))
    for lab, s in strips.items():
        for cname, us, ub in configs:
            if not us and lab != "ZR":
                continue
            key = f"{cname}" if not us else f"{cname} {lab}"
            mdl = Model2(s, mem, Js[lab], us, ub)
            y0 = np.zeros(4 + mdl.nm)
            y0[0] = 0.01
            t, rec = mdl.run(y0, 12.0, dt)
            wd, ze, ext = decay_stats(t, rec[:, 0])
            wn_td = wd / math.sqrt(max(1 - ze ** 2, 1e-9)) if ze == ze else float("nan")
            dec[key] = dict(wd=wd, zeta=ze, wn=wn_td,
                            peaks=[(float(a_), float(b_)) for a_, b_ in ext[:5]])
            say(f"    {key:22s}: damped freq {wd:.3f} rad/s, zeta {ze:.3f}, "
                f"undamped {wn_td:.3f} rad/s; first extrema (cm) "
                + ", ".join(f"{100 * e[1]:+.3f}" for e in ext[:4]))
    res["decay"] = dec
    say(f"  part 2 compute time {time.time() - t0:.0f} s")
    return fd, td, plant


# ------------------------------------------------- part 3: running state
def part3_running_state(res, u_kn=5.0):
    say(f"\n== part 3: the strip linearised about the RUNNING state at {u_kn:g} kn ==")
    t0 = time.time()
    u = u_kn * KNOT
    out = {}
    for lab, mlm in (("ZR", False), ("MLM", True)):
        v = make_vessel(mlm, u)
        z0, th0 = v.running_attitude(u)
        s = Strip(v, u)
        check_deriv_consistency(s, u, lab)
        x0 = np.array([z0, th0, 0.0, 0.0, 0.0, 0.0])
        say(f"  {lab}: running attitude at {u_kn:g} kn: rise {z0:+.4f} m, trim change "
            f"{math.degrees(th0):+.3f} deg (rest trim {-math.degrees(v.Th_rest):+.2f} deg bow up); "
            f"a_b blend {s.ab:.4f} (buoyancy x this factor), transom factor "
            f"{'1 (off)' if np.all(s.ctr == 1) else f'min {s.ctr.min():.3f}'}")
        Jall, o = part1_linearisation(s, f"{lab} running 5 kn", x0,
                                      om=3.24, amps=(0.02, 0.01, 0.005, 0.0025), res=out)
        _, aux0 = s.dyn(x0)
        n_entry = int((aux0["V"] > 0).sum())
        n_wet = int((aux0["d"] > 0).sum())
        say(f"  steady relative velocity U tan(tau): {aux0['V'][n_wet // 2]:+.4f} m/s at mid-length; "
            f"{n_entry} of {n_wet} wet stations count as 'entering' (momentum term on); "
            f"heave velocity amplitude at which the entry switch flips for a = 1 cm, 3.24 rad/s: "
            f"{0.01 * 3.24:.3f} m/s")
    res["running_5kn"] = out
    say(f"  part 3 compute time {time.time() - t0:.0f} s")


# ----------------------------------------------------------------- plots
def plots(res, strips_labels, fd, td):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
    # scaling
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    for ax, mode in zip(axs, ("heave", "pitch")):
        ci = 0
        for lab in strips_labels:
            rows = res[lab]["scaling"][mode]["rows"]
            a = np.array([r["a"] for r in rows]) * 100
            r_ = np.array([r["resid_max"] for r in rows])
            ax.loglog(a, r_, "-o", color=col[ci], lw=2, ms=5, label=f"{lab} (rest)")
            ci += 1
        for lab in res.get("running_5kn", {}):
            rows = res["running_5kn"][lab]["scaling"][mode]["rows"]
            a = np.array([r["a"] for r in rows]) * 100
            r_ = np.array([r["resid_max"] for r in rows])
            ax.loglog(a, r_, "--s", color=col[ci], lw=2, ms=5, label=lab)
            ci += 1
        rows = res["ZR"]["scaling"][mode]["rows"]
        a = np.array([r["a"] for r in rows]) * 100
        ref = rows[0]["resid_max"] * (a / a[0]) ** 2
        ax.loglog(a, ref, ":", color="#52514e", lw=1.5, label="slope 2 (a^2)")
        ax.set_xlabel("amplitude a, cm (pitch: bow displacement)")
        ax.set_ylabel("max |f_NL - f_lin|, N/m")
        ax.set_title(f"prescribed {mode}, strip section force")
        ax.grid(True, color="#e5e4e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v1_scaling.png"), dpi=130)
    plt.close(fig)
    # RAO
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    ws = np.array(sorted(fd))
    for ax, key, b13, nm in zip(axs, ("heave", "pitch"), (HEAVE_B13, PITCH_B13),
                                ("heave / a", "pitch / (k a)")):
        ax.plot(ws, [fd[w][f"{key}_exactC"] for w in ws], "-", color="#52514e", lw=2,
                label="BEM linear, frequency domain")
        ax.plot(W_B13, b13, "x", color="#0b0b0b", ms=6, label="B1.3 BEM column (following seas)")
        ci = 0
        for k, d in td.items():
            ww = np.array(sorted(d))
            ax.plot(ww, [d[w][key] for w in ww], "-o", color=col[ci], lw=1.6, ms=4, label=k)
            ci += 1
        ax.set_xlabel("wave frequency, rad/s")
        ax.set_ylabel(nm)
        ax.set_title(f"{key} response, U = 0, head seas, a = 1 cm")
        ax.grid(True, color="#e5e4e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v1_rao.png"), dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    t_all = time.time()
    try:
        from studies.parallel_runner import free_gb
        say(f"free RAM at start: {free_gb():.1f} GB")
    except Exception:
        pass
    res = {}
    strips, Js = {}, {}
    for lab, mlm in (("ZR", False), ("MLM", True)):
        v = make_vessel(mlm, 0.0)
        s = Strip(v, 0.0)
        strips[lab] = s
        if mlm:
            from sim.planing_zarnick import _mlm_constants
            Kv, Km, th = _mlm_constants(v.hull.beta_deg)
            say(f"  MLM constants (beta {v.hull.beta_deg:g} deg): K_m {Km:.4f} (Wagner pi/2 = "
                f"{math.pi / 2:.4f}), K_v {Kv:.4f} (Wagner pi^2/(2 tan b) = "
                f"{math.pi ** 2 / 2 / math.tan(math.radians(v.hull.beta_deg)):.4f})")
        say(f"  {lab}: rest CG height {v.Z_rest:+.4f} m, rest trim {-math.degrees(v.Th_rest):+.3f} deg "
            f"bow up, mass {v.m:.0f} kg, Iy {v.Iy:.0f} kg m2, {len(v.x)} stations dx {v.dx:.4f} m")
        res.setdefault("check0", {})[lab] = check_deriv_consistency(s, 0.0, lab)
        J, _ = part1_linearisation(s, lab, np.zeros(6), om=3.9,
                                   amps=(0.02, 0.01, 0.005, 0.0025), res=res)
        Js[lab] = J
    bem = load_bem()
    fd, td, plant = part2_time_domain(strips, Js, bem, res)
    part3_running_state(res)
    say("\n== part 4 (3-D forward speed): BLOCKED ==")
    say("  hydro/rankine/tables.py RadiationTable: radiation only (MODES surge, heave, pitch; "
        "no excitation, no diffraction, no encounter-frequency/following-sea handling); "
        "on disk only prismV wedge tables (L = 1 m) under studies/_cache/rankine, none for "
        "scarab195. The 5 kn comparison with a 3-D linear RAO cannot be made from existing tables.")
    plots(res, list(strips), fd, td)
    dt_all = time.time() - t_all
    say(f"\ntotal compute time {dt_all:.0f} s ({dt_all / 60:.1f} min)")
    res["compute_time_s"] = dt_all
    with open(os.path.join(OUT, "v1_linear_consistency.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_LINES) + "\n")

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    with open(os.path.join(OUT, "v1_results.json"), "w", encoding="utf-8") as f:
        json.dump(conv(res), f, indent=1)


if __name__ == "__main__":
    main()
