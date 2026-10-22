"""
Two-dimensional water entry of a symmetric wedge with a finite half-beam (hard chine): the nonlinear
cross-section model of stage B8 (learn/meta/RANKINE_STAGE1_2026-10-06.md, section 14).

Model (derived in 14.2; no fitted constant):
  * chine-dry stage ("slamming"): modified Logvinovich model (MLM; Korobkin 2004). The velocity potential
    is Wagner's flat-plate potential phi = -V sqrt(c^2 - y^2) on |y| < c, transferred to the body surface
    by a first-order Taylor shift, phi_b = -V [sqrt(c^2 - y^2) + f(y) - h]; the pressure is the full
    Bernoulli pressure on the body with the exact normal velocity. For a wedge f(y) = |y| tan(beta):
        p / rho = Vdot (R + f - h) + V c cdot / R - V^2/2 [sin^2(beta) + cos^2(beta) c^2 / R^2],
        R = sqrt(c^2 - y^2).                                                                    (W1)
    The half-width c follows Wagner's condition, for a wedge c = pi h / (2 tan(beta)) for ANY velocity
    history h(t) (W2); the force is the integral of (W1) from the keel to the first zero of p (the jet
    root, where MLM pressure turns negative), F = 2 int_0^a p dy.
  * chine-wetted stage (flow separation at the chine, y = l): Logvinovich's zero-pressure condition in
    the MLM form (Tassin, Korobkin & Cooker 2014): the flow leaves the chine, the pressure there is
    atmospheric; the same potential with a VIRTUAL half-width c > l, which follows from p(l) = 0 in (W1):
        dq/dt = V [sin^2 b sqrt(q) + cos^2 b (q + l^2)/sqrt(q)] - 2 (Vdot/V) sqrt(q) (sqrt(q) + l T - h),
        q = c^2 - l^2, q = 0 at chine wetting.                                                  (W3)
    The force is the integral of (W1) over the real bottom |y| <= l (and p >= 0).
  * linear reference (the cross-section of the linear 3-D theory of B7): Wagner's flat plate, half-width
    c_W = min(pi h/(2 T), l) (flow around the chine after wetting), force F_W = d/dt(rho pi c_W^2 V / 2).

Integrals are in closed form with y = c sin(theta) (14.2). Units: any consistent set; rho = 1.
"""
import math

import numpy as np


def _force_terms(c, cdot, V, Vdot, h, T, sb2, cb2, th):
    """2 int_0^{c sin th} p dy for (W1), closed form in theta (rho = 1)."""
    s, co = math.sin(th), math.cos(th)
    lnsec = math.log(1.0 / co + s / co) if co > 0 else float("inf")
    I = (V * c * cdot * th
         - 0.5 * V * V * sb2 * c * s
         - 0.5 * V * V * cb2 * c * lnsec
         + Vdot * (c * c * (0.5 * th + 0.25 * math.sin(2 * th)) + c * c * T * 0.5 * s * s - h * c * s))
    return 2.0 * I


def mlm_pressure(y, c, cdot, V, Vdot, h, T):
    """(W1) at |y| < c (rho = 1)."""
    b = math.atan(T)
    R2 = c * c - y * y
    R = np.sqrt(R2)
    return (Vdot * (R + np.abs(y) * T - h) + V * c * cdot / R
            - 0.5 * V * V * (math.sin(b) ** 2 + math.cos(b) ** 2 * c * c / R2))


def mlm_pressure_theta(th, c, cdot, V, Vdot, h, T):
    """(W1) at y = c sin(th), R = c cos(th) (no cancellation near the jet root)."""
    b = math.atan(T)
    co = np.cos(th)
    R = c * co
    return (Vdot * (R + c * np.abs(np.sin(th)) * T - h) + V * cdot / co
            - 0.5 * V * V * (math.sin(b) ** 2 + math.cos(b) ** 2 / (co * co)))


def _cutoff_theta(c, cdot, V, Vdot, h, T, th_max):
    """First zero of p(y = c sin th) going outward from the keel, or th_max if p > 0 up to it."""
    from scipy.optimize import brentq
    f = lambda th: float(mlm_pressure_theta(th, c, cdot, V, Vdot, h, T))
    if f(th_max) > 0.0:
        return th_max
    if f(0.0) <= 0.0:
        return 0.0
    return brentq(f, 0.0, th_max, xtol=1e-14)


def wedge_entry(T, l, h_of_t, V_of_t, Vd_of_t, t_end, n=4000, model="mlm"):
    """March one cross-section from h = 0 (keel at the undisturbed level) to t_end. T = tan(beta), l =
    half-beam. h, V = dh/dt, Vdot as functions of t (V > 0 downward). Returns arrays t, h, c (virtual half-
    width; = Wagner's before chine wetting), F (vertical force per unit length, rho = 1), F_W (linear
    flat-plate reference), and the chine-wetting time."""
    b = math.atan(T)
    sb2, cb2 = math.sin(b) ** 2, math.cos(b) ** 2
    t = np.linspace(0.0, t_end, n + 1)
    out = dict(t=t, h=np.zeros_like(t), c=np.zeros_like(t), F=np.zeros_like(t), F_W=np.zeros_like(t),
               a=np.zeros_like(t))
    q = 0.0
    t_cw = None
    # chine wetting: Wagner half-width reaches l at h = 2 l T / pi
    for k, tk in enumerate(t):
        h, V, Vd = h_of_t(tk), V_of_t(tk), Vd_of_t(tk)
        cW = math.pi * h / (2.0 * T)
        if cW < l or V <= 0.0:
            c, cdot = cW, math.pi * V / (2.0 * T)
            th_max = math.pi / 2 - 1e-15
        else:
            if t_cw is None:
                t_cw = tk
            if k > 0 and t_cw < tk:
                # integrate (W3) from the previous step (RK4 on q in sqrt-form: s = sqrt(q))
                q = _advance_q(q, t[k - 1], tk, T, l, h_of_t, V_of_t, Vd_of_t, sb2, cb2)
            sq = math.sqrt(max(q, 0.0))
            c = math.sqrt(l * l + q)
            # cdot from (W3)
            if sq > 0:
                qd = V * (sb2 * sq + cb2 * (q + l * l) / sq) - 2.0 * (Vd / V) * sq * (sq + l * T - h)
                cdot = 0.5 * qd / c
            else:
                cdot = float("inf")
            th_max = math.asin(min(l / c, 1.0)) if sq > 0 else math.pi / 2 - 1e-15
        if math.isinf(cdot):
            F = float("nan")
            th = th_max
        else:
            th = _cutoff_theta(c, cdot, V, Vd, h, T, th_max) if c > 0 else 0.0
            F = _force_terms(c, cdot, V, Vd, h, T, sb2, cb2, th) if c > 0 else 0.0
        cl = min(cW, l)
        cld = math.pi * V / (2.0 * T) if cW < l else 0.0
        out["h"][k], out["c"][k], out["F"][k] = h, c, F
        out["a"][k] = c * math.sin(th)
        out["F_W"][k] = math.pi * cl * cld * V + 0.5 * math.pi * cl * cl * Vd
    out["t_cw"] = t_cw
    return out


def _advance_q(q, t0, t1, T, l, h_of_t, V_of_t, Vd_of_t, sb2, cb2, nsub=20):
    """(W3) in the variable s = sqrt(q) (regular at s = 0 through s^2 ds/dt form): with u = s^3,
    du/dt = 3/2 s dq/dt = 3/2 V [sb2 s^2 + cb2 (s^2 + l^2)] - 3 (Vd/V) s^2 (s + l T - h)."""
    def rhs(t, u):
        s = max(u, 0.0) ** (1.0 / 3.0)
        V, Vd, h = V_of_t(t), Vd_of_t(t), h_of_t(t)
        return 1.5 * V * (sb2 * s * s + cb2 * (s * s + l * l)) - 3.0 * (Vd / V) * s * s * (s + l * T - h)
    u = max(q, 0.0) ** 1.5
    dt = (t1 - t0) / nsub
    tt = t0
    for _ in range(nsub):
        k1 = rhs(tt, u)
        k2 = rhs(tt + dt / 2, u + dt / 2 * k1)
        k3 = rhs(tt + dt / 2, u + dt / 2 * k2)
        k4 = rhs(tt + dt, u + dt * k3)
        u += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        tt += dt
    return max(u, 0.0) ** (2.0 / 3.0)


def constant_speed(T, l, h_end, V=1.0, n=4000):
    """Constant-velocity entry to depth h_end (keel below the undisturbed level)."""
    return wedge_entry(T, l, lambda t: V * t, lambda t: V, lambda t: 0.0, h_end / V, n=n)


def similarity_coefficients(beta_deg):
    """MLM chine-dry stage, constant V: F/(rho V^3 t), max p/(rho V^2/2) and the height of its location
    above the undisturbed level / (V t) (Zhao & Faltinsen 1993 definitions)."""
    T = math.tan(math.radians(beta_deg))
    V, t, l = 1.0, 1.0, 1e9
    h = V * t
    c = math.pi * h / (2 * T)
    cdot = math.pi * V / (2 * T)
    b = math.atan(T)
    th = _cutoff_theta(c, cdot, V, 0.0, h, T, math.pi / 2 - 1e-15)
    F = _force_terms(c, cdot, V, 0.0, h, T, math.sin(b) ** 2, math.cos(b) ** 2, th)
    ths = np.linspace(0, th, 200001)
    ys = c * np.sin(ths)
    p = mlm_pressure_theta(ths, c, cdot, V, 0.0, h, T)
    i = int(np.argmax(p))
    return dict(F=F / (V ** 3 * t), Cp_max=float(p[i]) / 0.5, z_max=(ys[i] * T - h) / (V * t),
                F_wagner=math.pi ** 3 / (4 * T * T))
