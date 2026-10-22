#!/usr/bin/env python3
"""
V1 (forward-speed half), steps 3-4 of learn/meta/HULL_FORCE_AUDIT_REVIEW_2026-10-09.md
section 4 row V1, on the 5 kn tables built by studies/derive_v1b_build.py:

  1  frequency-domain heave/pitch RAO of the 3-D linear model at 5 kn for the four
     wave cases (head / following seas, Tp 3 and 5 s), with the hydrostatics of the
     plant's exact V sections linearised about the running state;
  2  memory fit of the radiation (hydro/retardation.fit_coefficient on the
     stiffness- and B_inf-corrected pair A + C0/w^2, B - B_inf), the retardation
     function near the critical frequency g/(4U), the memory window needed;
  3  time-domain D1' (strip f^NL about the 5 kn running state + [3-D LTI - strip's
     numerical-Jacobian linear limit]) vs pure 3-D LTI vs plain strip, regular waves
     a = 1 cm, against the frequency-domain RAO;
  4  grid / domain / stern-closure sensitivity of the 3-D quantities at the four
     cases (whatever tables exist: coarse, base, fine, long, ramp variants).

Strip and hydrostatics are the previous agent's re-assembled forms
(studies/derive_v1_linear_consistency.py: Strip, exact_hydrostatics, jacobian),
verified against ZarnickVessel.deriv to round-off there.

Outputs: studies/_cache/derive/v1b_compare.txt, v1b_results.json, v1b_coeffs.png,
v1b_kernel.png, v1b_rao.png.

Run: .venv/Scripts/python.exe -m studies.derive_v1b_compare [--base base_r0.3] [--periods 25]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "_cache", "derive")
G = 9.81
SP = -1.0
_LINES = []


def say(s=""):
    print(s, flush=True)
    _LINES.append(s)


# ------------------------------------------------------------------ 3-D linear pieces
class Lin3D:
    """The 3-D LTI of one table, heave-pitch block (indices 1, 2 of modes (1, 3, 5))."""

    def __init__(self, rec):
        self.rec = rec
        ix = [1, 2]
        self.ix = ix
        self.omega = np.asarray(rec["omega"])
        self.A = np.asarray(rec["A"])[:, ix][:, :, ix]
        self.B = np.asarray(rec["B"])[:, ix][:, :, ix]
        self.A_inf = np.asarray(rec["A_inf"])[np.ix_(ix, ix)]
        self.B_inf = np.asarray(rec["B_inf"])[np.ix_(ix, ix)]
        self.C0 = np.asarray(rec["C0"])[np.ix_(ix, ix)]
        self.C_s = np.asarray(rec["C_s"])[np.ix_(ix, ix)]
        self.C_tot = self.C0 + self.C_s
        self.cases = rec["cases"]
        self.u = float(rec["plant"]["u"])

    def T(self, w, A=None, B=None):
        """-w^2 A(w) + i w B(w) from the table (nearest tabulated frequency) or from given matrices."""
        if A is None:
            k = int(np.argmin(np.abs(self.omega - w)))
            A, B = self.A[k], self.B[k]
        return -w ** 2 * A + 1j * w * B

    def case(self, key):
        c = self.cases[key]
        A = np.asarray(c["A_e"])[np.ix_(self.ix, self.ix)]
        B = np.asarray(c["B_e"])[np.ix_(self.ix, self.ix)]
        X = np.asarray(c["X"])[self.ix]
        return float(c["omega_e"]), float(c["k"]), A, B, X, c


def fd_rao(lin, M, Chs, key, A=None, B=None):
    we, k, Ae, Be, X, c = lin.case(key)
    if A is not None:
        Ae, Be = A, B
    Z = -we ** 2 * M + lin.T(we, Ae, Be) + Chs + lin.C_s
    xi = np.linalg.solve(Z, X)
    return dict(heave=abs(xi[0]), pitch=abs(xi[1]) / k, xi=xi, omega_e=we, k=k)


# ------------------------------------------------------------------ memory fit (4 entries)
def fit_memory(lin, uniform_only=True):
    from hydro.retardation import fit_coefficient
    om = lin.omega
    sel = np.ones(len(om), bool)
    if uniform_only:
        # keep the uniform 0.1 rad/s grid only (the three extra points around the critical frequency are left out)
        sel = np.abs(om * 10 - np.round(om * 10)) < 1e-6
        if sel.sum() < 10:
            sel = np.ones(len(om), bool)
    fits = {}
    for key, (i, j) in dict(hh=(0, 0), hp=(0, 1), ph=(1, 0), pp=(1, 1)).items():
        At = lin.A[sel, i, j] + lin.C0[i, j] / om[sel] ** 2          # stiffness removed: tends to A_inf
        Bt = lin.B[sel, i, j] - lin.B_inf[i, j]                      # integrable part of the damping
        f = fit_coefficient(om[sel], At, Bt)
        f["A_tilde"], f["B_tilde"], f["omega"] = At, Bt, om[sel]
        fits[key] = f
    return fits


class MemoryBlock:
    """Block state space of the four fitted entries: mu = C x, x' = A x + B nu, nu = (w, q)."""

    def __init__(self, fits):
        blocks = [("hh", 0, 0), ("hp", 0, 1), ("ph", 1, 0), ("pp", 1, 1)]
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
        self.A_inf_fit = np.array([[fits["hh"]["A_inf"], fits["hp"]["A_inf"]], [fits["ph"]["A_inf"], fits["pp"]["A_inf"]]])
        self.max_eig = float(np.max(np.abs(np.linalg.eigvals(self.A))))


def kernel_direct(lin, t, taper_hi=None):
    """K(t) of the table's B - B_inf by tables.memory_kernel (PCHIP on a fine grid, constant extension below the first
    frequency, zero above the last); heave-pitch block."""
    from hydro.rankine.tables import memory_kernel
    Bfull = np.asarray(lin.rec["B"])
    Binf = np.asarray(lin.rec["B_inf"])
    K, tail = memory_kernel(lin.omega, Bfull, Binf, t, taper=taper_hi)
    return K[:, lin.ix][:, :, lin.ix], tail


def window_error(t, K, w_list, i=0, j=0):
    """Relative error of B(w) when the convolution kernel is truncated at T: |int_T^inf K cos(w t) dt| / |int_0^inf|,
    for a list of frequencies; returns dict w -> (T for 5 %, T for 1 %)."""
    out = {}
    for w in w_list:
        full = np.trapezoid(K[:, i, j] * np.cos(w * t), t)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (K[1:, i, j] * np.cos(w * t[1:]) + K[:-1, i, j] * np.cos(w * t[:-1])) * np.diff(t))])
        rel = np.abs(full - cum) / max(abs(full), 1e-30)
        def first_below(th):
            ok = np.where(rel <= th)[0]
            # the LAST time the error exceeds th, then the next sample
            above = np.where(rel > th)[0]
            if len(above) == 0:
                return float(t[0])
            k = above[-1] + 1
            return float(t[k]) if k < len(t) else float("inf")
        out[float(w)] = (first_below(0.05), first_below(0.01), float(full))
    return out


# ------------------------------------------------------------------ strip at the running state
def strip_running(mlm):
    from studies.derive_v1_linear_consistency import make_vessel, Strip, jacobian, STEPS, exact_hydrostatics
    from studies.derive_v1b_build import U_KN, KNOT
    u = U_KN * KNOT
    v = make_vessel(mlm, u)
    z0, th0 = v.running_attitude(u)
    s = Strip(v, u)
    x0 = np.array([z0, th0, 0.0, 0.0, 0.0, 0.0])
    J, f0 = jacobian(lambda xi: s.dyn(xi)[0], x0, STEPS)
    lever = np.stack([np.ones_like(s.x), SP * s.x])
    FM = lever @ J * s.dx                                     # (2, 6) dF/d(z, th, w, q, wd, qd)
    A_lin = -FM[:, 4:6]
    K_lin = FM[:, 0:4]
    Chs = -jacobian(lambda q: exact_hydrostatics(s, q[0], q[1]), x0[:2], [1e-6, 1e-6])[0]
    return dict(v=v, strip=s, x0=x0, J=J, f0=f0, A_lin=A_lin, K_lin=K_lin, Chs=Chs,
                M=np.diag([v.m, v.Iy]), exact_hydrostatics=exact_hydrostatics)


class Model5:
    """2-DOF heave-pitch time-domain model at 5 kn about the plant's running state x0 = (z0, th0).
    use_strip: + sum f^NL dx (strip dynamic force at the running state, nonlinear, with the forward-speed terms)
    use_bem:   + 3-D LTI: A_inf, B_inf, memory, C0 + C_s about the slice attitude
    both:      the bracket [3-D LTI - strip linear limit]: the strip's numerical Jacobian about the running state
               (added mass, velocity and position terms) is subtracted.
    Each configuration removes its own static residual at x0, so x0 is its equilibrium."""

    def __init__(self, sr, lin, mem, use_strip, use_bem, Xc=None, om=0.0, amp=0.01):
        self.sr, self.lin, self.mem = sr, lin, mem
        self.s = sr["strip"]
        self.use_strip, self.use_bem = use_strip, use_bem
        self.sub = use_strip and use_bem
        self.M = sr["M"]
        self.x0 = sr["x0"][:2]
        self.Xc, self.om, self.amp = Xc, om, amp
        self.f0 = sr["f0"]
        self.nm = mem.n if use_bem else 0
        self.res0 = self._static(np.zeros(4 + self.nm))

    def _forces(self, t, y, with_exc=True):
        z, th, w, q = y[:4]
        xm = y[4:]
        s = self.s
        dx_ = np.array([z - self.x0[0], th - self.x0[1]])
        F = self.sr["exact_hydrostatics"](s, z, th)
        if with_exc and self.Xc is not None:
            F = F + self.amp * np.real(self.Xc * np.exp(1j * self.om * t))
        Mass = self.M.copy()
        dxm = np.zeros(0)
        vel = np.array([w, q])
        if self.use_bem:
            Mass = Mass + self.lin.A_inf
            F = F - self.lin.B_inf @ vel - self.lin.C_tot @ dx_ - self.mem.C @ xm
            dxm = self.mem.A @ xm + self.mem.B @ vel
        if self.use_strip:
            f, _ = s.dyn(np.array([z, th, w, q, 0.0, 0.0]))
            F = F + s.sum_force_moment(f)
            Mass = Mass + s.added_mass_matrix(z, th)
        if self.sub:
            F = F - s.sum_force_moment(self.f0) - self.sr["K_lin"] @ np.concatenate([dx_, vel])
            Mass = Mass - self.sr["A_lin"]
        return F, Mass, dxm

    def _static(self, y):
        y = np.array(y, float)
        y[0], y[1] = self.x0
        F, _, _ = self._forces(0.0, y, with_exc=False)
        return F

    def rhs(self, t, y):
        F, Mass, dxm = self._forces(t, y)
        acc = np.linalg.solve(Mass, F - self.res0)
        return np.concatenate([y[2:4], acc, dxm])

    def run(self, t_end, dt):
        n = int(round(t_end / dt))
        y = np.zeros(4 + self.nm)
        y[0], y[1] = self.x0
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
            if not np.all(np.isfinite(y)) or abs(y[0]) > 5.0:
                return np.arange(i + 2) * dt, rec[:i + 2], False
        return np.arange(n + 1) * dt, rec, True


def amp_phase(t, y, om, n_last):
    T = 2 * np.pi / om
    m = t >= t[-1] - n_last * T
    Aa = np.column_stack([np.cos(om * t[m]), np.sin(om * t[m]), np.ones(m.sum())])
    c, *_ = np.linalg.lstsq(Aa, y[m], rcond=None)
    return float(np.hypot(c[0], c[1]))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="base_r0.3")
    ap.add_argument("--others", default="smoke,fine_r0.3_n4,long_r0.3_n4,base_r0.15_n4,base_r0.6_n4,base_r0.3_n4_yc0.02")
    ap.add_argument("--periods", type=int, default=25)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--amp", type=float, default=0.01)
    ap.add_argument("--no-td", action="store_true")
    ap.add_argument("--tag", default="", help="suffix of the output files (v1b_compare<tag>.txt ...)")
    ap.add_argument("--td-cases", default="", help="comma list of cases for the time domain (default all four)")
    ap.add_argument("--periods-low", type=int, default=None, help="periods for the cases with omega_e < 1.2 rad/s")
    a = ap.parse_args()
    from studies.derive_v1b_build import load, OUT as OUTB
    t_all = time.time()
    res = dict(base=a.base)
    rec = load(a.base)
    lin = Lin3D(rec)
    p = rec["plant"]
    u = lin.u
    w_crit = G / (4 * u)
    say(f"== V1b: 5 kn ({u:.3f} m/s, Fn_L {u / math.sqrt(G * p['L']):.3f}), table {a.base}: grid {rec['grid']} M {rec['grid_kw']['M']}, "
        f"hull {rec['hull_mesh']}, NF {rec['NF']}, ramp {rec['ramp']} m; build {rec['t_build']:.0f} s, sweep {rec['t_sweep']:.0f} s "
        f"({len(lin.omega)} frequencies), total {rec['t_total']:.0f} s; peak RSS {rec['peak_rss_gb']:.2f} GB, GPU {rec['peak_gpu_gb']:.2f} GB")
    say(f"   critical frequency g/(4U) = {w_crit:.3f} rad/s; validity band of the grid [{rec['omega_lo']:.2f}, {rec['omega_hi']:.2f}] rad/s")
    say(f"   3-D limits (heave, pitch, coupling): A_inf {lin.A_inf[0, 0]:.0f} kg / {lin.A_inf[1, 1]:.0f} kg m2 / {lin.A_inf[0, 1]:.0f}, {lin.A_inf[1, 0]:.0f}; "
        f"B_inf {lin.B_inf[0, 0]:.1f} / {lin.B_inf[1, 1]:.1f} / {lin.B_inf[0, 1]:.1f}, {lin.B_inf[1, 0]:.1f}; "
        f"C0 {lin.C0[0, 0]:.0f} / {lin.C0[1, 1]:.0f} / {lin.C0[0, 1]:.0f}, {lin.C0[1, 0]:.0f}; "
        f"C_s {lin.C_s[0, 0]:.0f} / {lin.C_s[1, 1]:.0f} / {lin.C_s[0, 1]:.0f}, {lin.C_s[1, 0]:.0f}")
    res["limits"] = dict(A_inf=lin.A_inf.tolist(), B_inf=lin.B_inf.tolist(), C0=lin.C0.tolist(), C_s=lin.C_s.tolist())

    # ---- strip about the running state (ZR and MLM)
    srs = {lab: strip_running(mlm) for lab, mlm in (("ZR", False), ("MLM", True))}
    sr = srs["ZR"]
    Chs = sr["Chs"]
    M = sr["M"]
    say(f"\n== strip (plant) about the 5 kn running state ==")
    for lab, s_ in srs.items():
        say(f"  {lab}: running state z0 {s_['x0'][0]:+.4f} m, th0 {math.degrees(s_['x0'][1]):+.3f} deg; strip linear added mass "
            f"{s_['A_lin'][0, 0]:.0f} kg / {s_['A_lin'][1, 1]:.0f} kg m2 / {s_['A_lin'][0, 1]:.0f}; linear velocity terms (force per "
            f"heave vel, pitch rate; moment per heave vel, pitch rate) {s_['K_lin'][0, 2]:.0f} {s_['K_lin'][0, 3]:.0f} / "
            f"{s_['K_lin'][1, 2]:.0f} {s_['K_lin'][1, 3]:.0f}; position terms {s_['K_lin'][0, 0]:.0f} {s_['K_lin'][0, 1]:.0f} / "
            f"{s_['K_lin'][1, 0]:.0f} {s_['K_lin'][1, 1]:.0f}")
    say(f"  hydrostatic stiffness of the exact V sections about the running state: C33 {Chs[0, 0]:.0f} N/m, C55 {Chs[1, 1]:.0f} N m/rad, "
        f"C35 {Chs[0, 1]:.0f} / {Chs[1, 0]:.0f}; 3-D table's waterplane rho g Aw = {1025 * G * rec['Aw_run']:.0f} N/m (incl. ramp)")
    say(f"  bracket at 5 kn: added mass 3-D A_inf - strip A_lin = {lin.A_inf[0, 0] - sr['A_lin'][0, 0]:.0f} kg heave, "
        f"{lin.A_inf[1, 1] - sr['A_lin'][1, 1]:.0f} kg m2 pitch; damping B_inf - strip = {lin.B_inf[0, 0] + sr['K_lin'][0, 2]:.0f} / "
        f"{lin.B_inf[1, 1] + sr['K_lin'][1, 3]:.0f} (instantaneous part only; the memory adds B(w) - B_inf); "
        f"stiffness C0 + C_s - strip = {lin.C_tot[0, 0] + sr['K_lin'][0, 0]:.0f} / {lin.C_tot[1, 1] + sr['K_lin'][1, 1]:.0f}")
    res["strip"] = {lab: dict(x0=s_["x0"][:2].tolist(), A_lin=s_["A_lin"].tolist(), K_lin=s_["K_lin"].tolist()) for lab, s_ in srs.items()}
    res["C_hs"] = Chs.tolist()

    # ---- frequency-domain RAO at the four cases
    say(f"\n== frequency-domain RAO of the 3-D linear model at 5 kn (a -> 0), hydrostatics of the exact sections, C_s included ==")
    say(f"  {'case':>12}{'omega0':>8}{'omega_e':>8}{'tau_B':>7}{'heave/a':>9}{'pitch/ka':>10}{'|X3|':>8}{'|X5|/k':>9}{'FK3':>8}{'FK5/k':>9}"
        f"{'A33(we)':>9}{'B33(we)':>9}{'A55(we)':>9}{'B55(we)':>9}{'ramp FK':>9}")
    fd = {}
    for key in lin.cases:
        r = fd_rao(lin, M, Chs, key)
        we, k, Ae, Be, X, c = lin.case(key)
        Xfk = np.asarray(c["X_fk"])[lin.ix]
        fkr = np.asarray(c["X_fk_ramp"])[lin.ix]
        fd[key] = dict(heave=r["heave"], pitch=r["pitch"], omega_e=we, k=k, tau_B=float(c["tau_B"]),
                       X3=abs(X[0]), X5k=abs(X[1]) / k, FK3=abs(Xfk[0]), FK5k=abs(Xfk[1]) / k,
                       A33=Ae[0, 0], B33=Be[0, 0], A55=Ae[1, 1], B55=Be[1, 1], ramp_fk_share=abs(fkr[0]) / max(abs(Xfk[0]), 1e-9))
        d = fd[key]
        say(f"  {key:>12}{float(c['omega0']):8.3f}{we:8.3f}{d['tau_B']:7.3f}{d['heave']:9.3f}{d['pitch']:10.3f}{d['X3']:8.0f}{d['X5k']:9.0f}"
            f"{d['FK3']:8.0f}{d['FK5k']:9.0f}{d['A33']:9.0f}{d['B33']:9.0f}{d['A55']:9.0f}{d['B55']:9.0f}{d['ramp_fk_share']:9.3f}")
    res["fd"] = fd
    # zero-speed comparison values (previous agent, head seas, exact-C) at the nearest wave frequencies, for orientation
    say("  (for orientation, the zero-speed 3-D RAO of v1_progress_zh.md at wave frequency 2.20 rad/s was heave 0.769, pitch 0.858; "
        "at 1.35: 0.965, 0.992)")

    # ---- memory fit
    say(f"\n== memory fit (hydro/retardation.fit_coefficient on A + C0/w^2 and B - B_inf, uniform grid 0.2-8 rad/s step 0.1) ==")
    fits = fit_memory(lin)
    mem = MemoryBlock(fits)
    for k_, f in fits.items():
        say(f"  {k_}: order {f['order']}, err_K {f['err_K']:.3f}, err_B {f['err_B']:.3f}, err_A {f['err_A']:.3f} (band 0.3-2.5: B {f['err_B_band']:.3f}, "
            f"A {f['err_A_band']:.3f}); A_inf fit {f['A_inf']:.0f} (spread {f['A_inf_spread']:.3f}) vs solver "
            f"{lin.A_inf[{'hh': (0, 0), 'hp': (0, 1), 'ph': (1, 0), 'pp': (1, 1)}[k_]]:.0f}; memory time (2 %) {f['memory_time']:.1f} s; "
            f"slowest pole {f['max_real_eig']:.3f} 1/s")
    say(f"  state space: {mem.n} states, |eig|max {mem.max_eig:.1f} 1/s (RK4 dt {a.dt}: |eig| dt = {mem.max_eig * a.dt:.2f})")
    res["memory_fit"] = {k_: dict(order=int(f["order"]), err_K=f["err_K"], err_B=f["err_B"], err_A=f["err_A"], err_B_band=f["err_B_band"],
                                  err_A_band=f["err_A_band"], A_inf_fit=f["A_inf"], memory_time=f["memory_time"],
                                  max_real_eig=f["max_real_eig"]) for k_, f in fits.items()}
    res["memory_states"] = mem.n
    # retardation function directly from the table (to 80 s) and the window needed
    t_long = np.arange(0.0, 80.0 + 1e-9, 0.02)
    # cos^2 window from 5 to 8 rad/s: removes the Gibbs ringing of the hard cut at 8 rad/s, so the tail shows the
    # physics near the critical frequency and not the cut
    Kd, tail = kernel_direct(lin, t_long, taper_hi=(5.0, 8.0))
    peak = np.abs(Kd[:, 0, 0]).max()
    env = np.abs(Kd[:, 0, 0])
    # local maxima envelope after 5 s, fitted as c t^-p
    idx = [i for i in range(1, len(t_long) - 1) if env[i] > env[i - 1] and env[i] > env[i + 1] and t_long[i] > 5.0]
    pfit = float("nan")
    if len(idx) > 4:
        pfit = -np.polyfit(np.log(t_long[idx]), np.log(env[idx] + 1e-300), 1)[0]
    # dominant frequency of the tail
    tt = t_long > 10.0
    zc = np.where(np.diff(np.sign(Kd[tt, 0, 0])) != 0)[0]
    w_tail = math.pi / np.mean(np.diff(t_long[tt][zc])) if len(zc) > 3 else float("nan")
    say(f"  retardation function K33(t) from the table (PCHIP, 0-8 rad/s, cos^2 window 5-8 rad/s; B - B_inf at 8 rad/s is {tail:.3f} of its max before the window): peak {peak:.0f} N/m, "
        f"|K| < 2 % of peak after {t_long[np.max(np.where(env > 0.02 * peak)[0])]:.1f} s, < 0.5 % after "
        f"{t_long[np.max(np.where(env > 0.005 * peak)[0])] if np.any(env > 0.005 * peak) else float('nan'):.1f} s; envelope of the tail "
        f"(t > 5 s) decays like t^-{pfit:.2f}; tail oscillation frequency {w_tail:.3f} rad/s (critical g/(4U) = {w_crit:.3f})")
    wins = {}
    for (i, j, nm) in ((0, 0, "K33"), (1, 1, "K55")):
        wl = [0.843, 0.944, 1.671, 3.244]
        we_ = window_error(t_long, Kd, wl, i, j)
        wins[nm] = we_
        say(f"  window needed for {nm}: " + "; ".join(f"w_e {w:.3f}: {v[0]:.1f} s (5 %), {v[1]:.1f} s (1 %)" for w, v in we_.items()))
    res["kernel"] = dict(peak=peak, tail_power=pfit, w_tail=w_tail, windows=wins, t_2pct=float(t_long[np.max(np.where(env > 0.02 * peak)[0])]))

    # can a longer window / higher order fit the tail?  (information)
    from hydro.retardation import fit_coefficient
    say("\n  fit experiments (heave-heave and pitch-pitch): max error of B over 0.2-5.2 rad/s / of A, as a share of the range:")
    fx = {}
    for order, tmax in ((10, 30.0), (16, 30.0), (16, 60.0), (24, 60.0), (24, 90.0)):
        row = {}
        for k_ in ("hh", "pp"):
            f0 = fits[k_]
            try:
                f = fit_coefficient(f0["omega"], f0["A_tilde"], f0["B_tilde"], order=order, t_max=tmax)
                row[k_] = (f["err_B"], f["err_A"], f["err_B_band"], int(f["order"]), f["A_inf"])
            except Exception as e:      # noqa: BLE001
                row[k_] = (float("nan"), float("nan"), float("nan"), 0, float("nan"))
        fx[f"order{order}_t{tmax:g}"] = row
        say(f"    order {order:2d}, window {tmax:4.0f} s: hh err_B {row['hh'][0]:.3f} err_A {row['hh'][1]:.3f} (B in 0.3-2.5: {row['hh'][2]:.3f}, A_inf {row['hh'][4]:.0f}); "
            f"pp err_B {row['pp'][0]:.3f} err_A {row['pp'][1]:.3f} (B in 0.3-2.5: {row['pp'][2]:.3f}, A_inf {row['pp'][4]:.0f})")
    res["fit_experiments"] = fx
    # RAO without the hydrodynamic stiffness C0 + C_s (information: how much of the response is driven by the
    # unconverged static stiffness of the double-body model)
    say("\n  RAO with the hydrodynamic stiffness C0 + C_s removed (A + C0/w^2 used; information only):")
    fd0 = {}
    for key in lin.cases:
        we, k, Ae, Be, X, c = lin.case(key)
        Z = -we ** 2 * (M + Ae + lin.C0 / we ** 2) + 1j * we * Be + Chs
        xi = np.linalg.solve(Z, X)
        fd0[key] = dict(heave=abs(xi[0]), pitch=abs(xi[1]) / k)
        say(f"  {key:>12}: heave {abs(xi[0]):.3f} (with stiffness {fd[key]['heave']:.3f}), pitch {abs(xi[1]) / k:.3f} (with {fd[key]['pitch']:.3f})")
    res["fd_no_hydrostiff"] = fd0

    # RAO with the fitted A, B (what the state space represents) at the four cases
    say(f"\n  RAO with the fit-reproduced A, B at the encounter frequencies (separates the memory-fit error from the model):")
    fdfit = {}
    from hydro.retardation import transfer_function
    for key in lin.cases:
        we, k, Ae, Be, X, c = lin.case(key)
        Af = np.zeros((2, 2))
        Bf = np.zeros((2, 2))
        for k_, (i, j) in dict(hh=(0, 0), hp=(0, 1), ph=(1, 0), pp=(1, 1)).items():
            f = fits[k_]
            # exactly what the time-domain state space represents at omega_e: H(i w) of (Ar, Br, Cr) with the
            # solver's A_inf, B_inf and C0 (the same constants the time-domain model uses)
            H = transfer_function(np.array([we]), f["Ar"], f["Br"], f["Cr"])[0]
            Af[i, j] = lin.A_inf[i, j] + np.imag(H) / we - lin.C0[i, j] / we ** 2
            Bf[i, j] = lin.B_inf[i, j] + np.real(H)
        r = fd_rao(lin, M, Chs, key, A=Af, B=Bf)
        fdfit[key] = dict(heave=r["heave"], pitch=r["pitch"])
        say(f"  {key:>12}: heave {r['heave']:.3f} (exact {fd[key]['heave']:.3f}, {100 * (r['heave'] / fd[key]['heave'] - 1):+.1f} %), "
            f"pitch {r['pitch']:.3f} (exact {fd[key]['pitch']:.3f}, {100 * (r['pitch'] / fd[key]['pitch'] - 1):+.1f} %)")
    res["fd_fit"] = fdfit

    # ---- time domain
    td = {}
    if not a.no_td:
        say(f"\n== time domain, a = {a.amp} m, RK4 dt {a.dt} s, {a.periods} encounter periods, last 8 fitted; excitation = 3-D X (FK + diffraction) ==")
        configs = [("D1'", True, True), ("pure 3-D LTI", False, True), ("plain strip", True, False)]
        t0 = time.time()
        for lab, s_ in srs.items():
            for cname, us, ub in configs:
                if not us and lab != "ZR":
                    continue
                key = cname if not us else f"{cname} {lab}"
                td[key] = {}
                say(f"  --- {key} ---")
                say(f"  {'case':>12}{'heave/a':>9}{'FD':>8}{'diff %':>8}{'FD(fit)':>9}{'diff %':>8} | {'pitch/ka':>9}{'FD':>8}{'diff %':>8}{'FD(fit)':>9}{'diff %':>8}  note")
                for ck in lin.cases:
                    if a.td_cases and ck not in a.td_cases.split(","):
                        continue
                    we, k, Ae, Be, X, c = lin.case(ck)
                    mdl = Model5(s_, lin, mem, us, ub, Xc=X, om=we, amp=a.amp)
                    T = 2 * np.pi / we
                    nper = a.periods if (a.periods_low is None or we >= 1.2) else a.periods_low
                    t, rec_, ok = mdl.run(nper * T, a.dt)
                    if ok:
                        hz = amp_phase(t, rec_[:, 0] - mdl.x0[0], we, 8) / a.amp
                        pz = amp_phase(t, rec_[:, 1] - mdl.x0[1], we, 8) / (k * a.amp)
                        mean_z = float(np.mean(rec_[int(len(t) * 0.7):, 0]) - mdl.x0[0])
                        note = f"mean heave offset {100 * mean_z:+.2f} cm"
                    else:
                        hz = pz = float("nan")
                        note = f"DIVERGED at t = {t[-1]:.1f} s"
                    f_ = fd[ck]
                    ff = fdfit[ck]
                    if ok:
                        # convergence of the amplitude: last 8 periods vs the 8 before
                        hz_prev = amp_phase(t[t <= t[-1] - 8 * T], (rec_[:, 0] - mdl.x0[0])[t <= t[-1] - 8 * T], we, 8) / a.amp
                        note += f", heave amp over the previous 8 periods {hz_prev:.3f} ({nper} periods)"
                    td[key][ck] = dict(heave=hz, pitch=pz, ok=bool(ok), periods=int(nper), static_residual=mdl.res0.tolist())
                    say(f"  {ck:>12}{hz:9.3f}{f_['heave']:8.3f}{100 * (hz / f_['heave'] - 1):+8.1f}{ff['heave']:9.3f}{100 * (hz / ff['heave'] - 1):+8.1f} | "
                        f"{pz:9.3f}{f_['pitch']:8.3f}{100 * (pz / f_['pitch'] - 1):+8.1f}{ff['pitch']:9.3f}{100 * (pz / ff['pitch'] - 1):+8.1f}  {note}")
                r0 = Model5(s_, lin, mem, us, ub).res0
                say(f"    (static residual removed at the running state: {r0[0]:+.0f} N, {r0[1]:+.0f} N m)")
        say(f"  time-domain compute {time.time() - t0:.0f} s")
    res["td"] = td

    # ---- sensitivity across tables
    say(f"\n== sensitivity of the 3-D quantities at the four cases across grids / domain / stern closure ==")
    tabs = {a.base: rec}
    for lab in [x for x in a.others.split(",") if x]:
        try:
            tabs[lab] = load(lab)
        except Exception:
            pass
    say(f"  {'table':>18}{'M':>4}{'hull':>8}{'NF':>6}{'ramp':>5}{'band lo-hi':>12} | per case: A33(we) B33(we) A55(we) B55(we) |X3| |X5|/k heave/a pitch/ka")
    sens = {}
    for lab, r_ in tabs.items():
        li = Lin3D(r_)
        row = {}
        parts = []
        for ck in lin.cases:
            if ck not in li.cases:
                continue
            we, k, Ae, Be, X, c = li.case(ck)
            rr = fd_rao(li, M, Chs, ck)
            row[ck] = dict(A33=Ae[0, 0], B33=Be[0, 0], A55=Ae[1, 1], B55=Be[1, 1], X3=abs(X[0]), X5k=abs(X[1]) / k, heave=rr["heave"], pitch=rr["pitch"])
            parts.append(f"{ck}: {Ae[0, 0]:.0f} {Be[0, 0]:.0f} {Ae[1, 1]:.0f} {Be[1, 1]:.0f} {abs(X[0]):.0f} {abs(X[1]) / k:.0f} {rr['heave']:.3f} {rr['pitch']:.3f}")
        sens[lab] = dict(M=int(r_["grid_kw"]["M"]), hull=list(r_["hull_mesh"]), NF=int(r_["NF"]), ramp=float(r_["ramp"]),
                         band=[float(r_["omega_lo"]), float(r_["omega_hi"])], A_inf=np.asarray(r_["A_inf"]).tolist(),
                         C0=np.asarray(r_["C0"]).tolist(), C_s=np.asarray(r_["C_s"]).tolist(), cases=row,
                         bsym_worst=float(np.min(np.asarray(r_["bsym_rel"]))), t_total=float(r_["t_total"]),
                         peak_rss_gb=float(r_["peak_rss_gb"]), peak_gpu_gb=float(r_["peak_gpu_gb"]))
        say(f"  {lab:>18}{sens[lab]['M']:4d}{str(tuple(int(x) for x in sens[lab]['hull'])):>8}{sens[lab]['NF']:6d}{sens[lab]['ramp']:5.2f}"
            f"{sens[lab]['band'][0]:6.2f}-{sens[lab]['band'][1]:5.2f} | A_inf 33/55 {np.asarray(r_['A_inf'])[1, 1]:.0f}/{np.asarray(r_['A_inf'])[2, 2]:.0f}, "
            f"C0 33/55 {np.asarray(r_['C0'])[1, 1]:.0f}/{np.asarray(r_['C0'])[2, 2]:.0f}, C_s {np.asarray(r_['C_s'])[1, 1]:.0f}/{np.asarray(r_['C_s'])[2, 2]:.0f}, "
            f"worst passivity {sens[lab]['bsym_worst']:+.3f}, {sens[lab]['t_total']:.0f} s, RSS {sens[lab]['peak_rss_gb']:.2f} GB")
        for pt in parts:
            say(f"      {pt}")
    res["sensitivity"] = sens
    # A33, B33 on the common uniform frequencies of the tables that have them
    say(f"\n  A33 / B33 / A55 / B55 at common frequencies (tables with a sweep):")
    common = None
    for lab, r_ in tabs.items():
        om = np.round(np.asarray(r_["omega"]), 4)
        common = set(om) if common is None else common & set(om)
    common = sorted(common) if common else []
    pick = [w for w in common if any(abs(w - x) < 1e-6 for x in (0.3, 0.955, 1.0, 1.7, 3.2, 5.0, 8.0))] or common[:6]
    for w in pick:
        parts = []
        for lab, r_ in tabs.items():
            om = np.round(np.asarray(r_["omega"]), 4)
            k_ = int(np.argmin(np.abs(om - w)))
            A_, B_ = np.asarray(r_["A"])[k_], np.asarray(r_["B"])[k_]
            parts.append(f"{lab}: {A_[1, 1]:.0f} {B_[1, 1]:.0f} {A_[2, 2]:.0f} {B_[2, 2]:.0f}")
        say(f"    w_e {w:.3f}: " + " | ".join(parts))

    # ---- plots
    try:
        plots(lin, fits, mem, t_long, Kd, fd, td, tabs, w_crit)
    except Exception as e:      # noqa: BLE001
        say(f"plots failed: {e!r}")
    res["compute_time_s"] = time.time() - t_all
    say(f"\ntotal compare time {res['compute_time_s']:.0f} s")
    with open(os.path.join(OUT, f"v1b_compare{a.tag}.txt"), "w", encoding="utf-8") as f:
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
        if isinstance(o, complex):
            return [o.real, o.imag]
        return o
    with open(os.path.join(OUT, f"v1b_results{a.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(conv(res), f, indent=1)


def plots(lin, fits, mem, t_long, Kd, fd, td, tabs, w_crit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#52514e"]
    # coefficients
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.4))
    names = [("A33 (kg), stiffness part removed", 0, 0, "A"), ("B33 (N s/m)", 0, 0, "B"), ("A35 / A53 (kg m)", 0, 1, "A"),
             ("A55 (kg m2), stiffness part removed", 1, 1, "A"), ("B55 (N m s)", 1, 1, "B"), ("B35 / B53 (N s)", 0, 1, "B")]
    for ax, (nm, i, j, q) in zip(axs.ravel(), names):
        ci = 0
        for lab, r_ in tabs.items():
            li = Lin3D(r_)
            om = li.omega
            if q == "A":
                y = li.A[:, i, j] + li.C0[i, j] / om ** 2
                y2 = li.A[:, j, i] + li.C0[j, i] / om ** 2
            else:
                y = li.B[:, i, j]
                y2 = li.B[:, j, i]
            style = "-o" if lab == lin.rec["label"] else "--s"
            ax.plot(om, y, style, color=col[ci % len(col)], ms=3, lw=1.4, label=f"{lab} ({'ij' if i != j else ''})")
            if i != j:
                ax.plot(om, y2, ":", color=col[ci % len(col)], lw=1.4, label=f"{lab} (ji)")
            ci += 1
        if (i, j) == (0, 0) and q == "A":
            f = fits["hh"]
            ax.plot(f["omega"], f["A_fit"], "-", color="#0b0b0b", lw=0.8, label="state-space fit")
        if (i, j) == (0, 0) and q == "B":
            f = fits["hh"]
            ax.plot(f["omega"], f["B_fit"] + lin.B_inf[0, 0], "-", color="#0b0b0b", lw=0.8, label="state-space fit")
        if (i, j) == (1, 1) and q == "A":
            ax.plot(fits["pp"]["omega"], fits["pp"]["A_fit"], "-", color="#0b0b0b", lw=0.8)
        if (i, j) == (1, 1) and q == "B":
            ax.plot(fits["pp"]["omega"], fits["pp"]["B_fit"] + lin.B_inf[1, 1], "-", color="#0b0b0b", lw=0.8)
        ax.axvline(w_crit, color="#9a9a9a", lw=0.8, ls="--")
        ax.axvspan(lin.rec["omega_lo"], lin.rec["omega_hi"], color="#e5e4e0", alpha=0.5, lw=0)
        for ck, d in fd.items():
            ax.axvline(d["omega_e"], color="#c8c8c8", lw=0.6)
        ax.set_title(nm, fontsize=9)
        ax.set_xlabel("encounter frequency, rad/s")
        ax.grid(True, color="#eeeeee", lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle("scarab195 at 5 kn, double-body linearisation: dashed grey = g/(4U); shaded = validity band of the base grid; "
                 "thin lines = the four encounter frequencies", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v1b_coeffs.png"), dpi=130)
    plt.close(fig)
    # kernel
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axs[0]
    ax.plot(t_long, Kd[:, 0, 0], color=col[0], lw=1.2, label="K33 from the table (0-8 rad/s)")
    f = fits["hh"]
    ax.plot(f["t"], f["K_ss"], "--", color=col[1], lw=1.2, label=f"state-space fit (order {f['order']})")
    ax.set_xlim(0, 40)
    ax.set_xlabel("t, s")
    ax.set_ylabel("K33, N/m")
    ax.set_title("heave retardation function at 5 kn", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = axs[1]
    env = np.abs(Kd[:, 0, 0])
    ax.semilogy(t_long, env / env.max(), color=col[0], lw=1.0, label="|K33| / peak")
    ax.semilogy(t_long, np.abs(Kd[:, 1, 1]) / np.abs(Kd[:, 1, 1]).max(), color=col[2], lw=1.0, label="|K55| / peak")
    tt = t_long[t_long > 3]
    ax.semilogy(tt, 0.3 / tt, ":", color="#52514e", lw=1.0, label="1/t")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlabel("t, s")
    ax.set_title(f"decay of the tail (period of the tail 2 pi / {w_crit:.2f} = {2 * np.pi / w_crit:.1f} s)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for ax in axs:
        ax.grid(True, color="#eeeeee", lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v1b_kernel.png"), dpi=130)
    plt.close(fig)
    # RAO bars
    if td:
        keys = list(fd)
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, q, nm in zip(axs, ("heave", "pitch"), ("heave / a", "pitch / (k a)")):
            labels = ["3-D frequency domain"] + list(td)
            n = len(labels)
            x = np.arange(len(keys))
            wbar = 0.8 / n
            ax.bar(x - 0.4 + wbar / 2, [fd[k][q] for k in keys], wbar, color="#52514e", label=labels[0])
            for i_, (lab, d) in enumerate(td.items()):
                ax.bar(x - 0.4 + wbar * (i_ + 1.5), [d[k][q] for k in keys], wbar, color=col[i_ % len(col)], label=lab)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{k}\nw_e {fd[k]['omega_e']:.2f}" for k in keys], fontsize=7)
            ax.set_ylabel(nm)
            ax.set_title(f"{q} response at 5 kn, a = 1 cm", fontsize=9)
            ax.grid(True, axis="y", color="#eeeeee", lw=0.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axs[0].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "v1b_rao.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
