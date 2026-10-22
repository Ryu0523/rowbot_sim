#!/usr/bin/env python3
"""
Linearisation helpers for the frozen plant scarab195@as1 (control-side
analysis, learn/meta/STABILITY_MAP_2026-10-12.md). Read-only use of sim/,
hydro/, app/: nothing there is changed.

CONVENTIONS (stated once, used everywhere here)
  state s (14) = [x, y, z, phi, th, psi, u, v, w, p, q, r, T, delta]
      z   rise of the CG above its rest floating position, m (up positive)
      th  pitch from the rest attitude, rad, with SIGN_PITCH = -1: bow-up
          is NEGATIVE th. The trim angle reported everywhere is
          trim_deg = -(Th_rest + th) in degrees, bow-up POSITIVE (the
          hull's rest trim Th_rest is included, as sim/vessel_profile and
          the S1 checks do).
      u, v, w  body velocities m/s; p, q, r body rates rad/s
      T   the jet's thrust STATE, N "at the design speed" (the command's
          unit); c = T / t_max is the throttle fraction
      delta  nozzle angle, rad
  speeds in knots in the tables, 1 kn = 0.514444 m/s.
  calm water: the sea is Monochromatic(1, 0) (no waves), so the forces do
  not depend on x, y, psi or t.

THE PLANT'S INTERNAL STATE. With the as1 preset the plant has no hidden
dynamic field (plant.hidden_state() is None: the Wagner entry field exists
only with switch wagner_entry, not in as1). The chine-wetting position,
the entry/exit gate (V > 0), the transom wetted height and the number of
wet stations are instantaneous functions of the 14-vector. What remains
outside the vector: the two actuator delay lines (10 slots at dt 0.01 s),
the pump prime (relaxes towards clip(submergence / band, 0, 1); at every
equilibrium below the intake stays >= 5 band depths under water, so the
target is 1 and prime = 1 is a fixed point) and measurement flags
(_emerged, slam counters) that feed back into nothing. prep() puts the
actuators into their equilibrium state before every step evaluation.

THE ONE-STEP MAP. F(s) = plant.step(s, 0, T_cmd, delta_cmd, dt) with the
delay lines holding the commands: s_{k+1} = F(s_k). Its Jacobian Phi over
the 12 states STATES (x and y are excluded: they are pure integrators of
the body velocity and enter no force) is taken by central differences,
Phi[:, j] = (F(s + h e_j) - F(s - h e_j)) / 2h, h = eps x scale_j, scale
1 m, 1 rad, 1 m/s, 1 rad/s, and t_max for T. The input Jacobian B is the
derivative with respect to the DELAYED command (the delay lines filled
with the perturbed command), so B is the actuator lag's dt / tau on the
T and delta rows; the pure delay is carried separately (stability_ctrl).
Discrete eigenvalues z of Phi are converted to continuous rates
lambda = log(z) / dt; for a complex pair wn = |lambda|, zeta = -Re / |lambda|,
period 2 pi / Im; for a real one the time constant is -1 / Re.
"""
import copy
import math
import time

import numpy as np
from scipy.optimize import brentq, fsolve

KN = 0.514444
G = 9.81
IDX = dict(x=0, y=1, z=2, phi=3, th=4, psi=5, u=6, v=7, w=8, p=9, q=10,
           r=11, T=12, d=13)
STATES = ("z", "phi", "th", "psi", "u", "v", "w", "p", "q", "r", "T", "d")
VERT = ("z", "th", "u", "w", "q", "T")        # vertical plane + thrust lag
LAT = ("phi", "v", "p", "r", "d")              # lateral plane + nozzle lag
LAT_PSI = ("phi", "psi", "v", "p", "r", "d")   # with heading (closed loop)
HULL = "scarab195@as1"


def calm():
    from sim.test_vessel import Monochromatic
    return Monochromatic(1.0, 0.0)


def build(hull=None, dt=0.01, gz=True):
    """The frozen plant in calm water: hull.plant(calm, dt) and, with gz,
    its large-angle twin (sim.physics.gz_twin), as the evaluation runs it.
    `hull` None = hydro.hulls.get('scarab195@as1')."""
    from hydro import hulls
    from sim import physics
    h = hulls.get(HULL) if hull is None else hull
    p = h.plant(calm(), dt=dt)
    return physics.gz_twin(p) if gz else p


def scale_of(p, name):
    return float(p.prop.t_max) if name == "T" else 1.0


def prep(p, T, d=0.0):
    """Actuator internal state at an equilibrium with thrust command T and
    nozzle command d: delay lines full of the command, prime 1."""
    n = p.prop.delay.n
    p.prop.delay.buf = [float(T)] * n if n else None
    m = p.rudder.delay.n
    p.rudder.delay.buf = [float(d)] * m if m else None
    p.prop.prime = 1.0
    p._emerged = False
    if hasattr(p, "capsized"):
        p.capsized = False


def trim_deg(p, th):
    from sim.vessel import SIGN_PITCH
    return float(np.degrees(SIGN_PITCH * (p.Th_rest + th)))


def th_of_trim(p, trim):
    from sim.vessel import SIGN_PITCH
    return float(np.radians(trim) / SIGN_PITCH - p.Th_rest)


# ------------------------------------------------------------ equilibrium
def captive_F(p, u, v):
    """(w', q') of the captive equations at speed u, rise v[0], pitch v[1]
    (the plant's own captive copy: surge held by a force at the nozzle)."""
    ds, aux = p.captive_deriv(float(u), float(v[0]), float(v[1]))
    return np.array([ds[8], ds[10]]), aux


def jac2(p, u, v, eps=(1e-7, 1e-8)):
    J = np.zeros((2, 2))
    for i in range(2):
        h = np.zeros(2)
        h[i] = eps[i]
        J[:, i] = (captive_F(p, u, v + h)[0] - captive_F(p, u, v - h)[0]) \
            / (2 * eps[i])
    return J


def polish(p, u, v, tol=1e-11, it=40):
    """Newton with backtracking on the captive equations: (root, max|res|)."""
    v = np.asarray(v, float).copy()
    r = np.abs(captive_F(p, u, v)[0]).max()
    for _ in range(it):
        if r <= tol:
            break
        try:
            step = np.linalg.solve(jac2(p, u, v), -captive_F(p, u, v)[0])
        except np.linalg.LinAlgError:
            break
        lam = 1.0
        while lam > 1e-4:
            vn = v + lam * step
            rn = np.abs(captive_F(p, u, vn)[0]).max()
            if rn < r:
                v, r = vn, rn
                break
            lam *= 0.5
        else:
            break
    return v, float(r)


def thrust_for(p, u, R):
    """The thrust state T whose net jet force at speed u equals R (N):
    fx(T, 0, u) is monotone in T, bisected by brentq; prime 1."""
    prime, p.prop.prime = p.prop.prime, 1.0
    try:
        if R <= 0.0:
            return 0.0
        f = lambda T: p.prop.forces(T, 0.0, u)[0] - R   # noqa: E731
        hi = 3.0 * p.prop.t_max
        if f(hi) < 0.0:
            return float("nan")
        return float(brentq(f, 0.0, hi, xtol=1e-9, rtol=1e-13))
    finally:
        p.prop.prime = prime


def equilibrium(p, u, guess=None):
    """Calm-water equilibrium of the free plant at speed u (m/s): the
    captive (rise, pitch) root polished by Newton from `guess` (default the
    plant's own running_attitude captive run), the captive resistance R
    (the force that holds the speed, air drag included) and the thrust
    state T with fx(T, u) = R. Returns a dict with the 14-state vector."""
    u = float(u)
    if u <= 0.0:
        v, res = np.zeros(2), 0.0
        R, T = 0.0, 0.0
        _, aux = captive_F(p, 0.0, v)
    else:
        v0 = np.array(p.running_attitude(u) if guess is None else guess,
                      float)
        with np.errstate(all="ignore"):
            sol = fsolve(lambda w: captive_F(p, u, w)[0], v0, xtol=1e-13)
        v, res = polish(p, u, sol)
        if res > 1e-8:                      # fall back to the captive run
            v, res = polish(p, u, v0)
        _, aux = captive_F(p, u, v)
        R = float(aux["fx"])
        T = thrust_for(p, u, R)
    s = np.zeros(14)
    s[2], s[4], s[6], s[12] = v[0], v[1], u, T
    d = np.asarray(aux["d"])
    V = np.asarray(aux["V"])
    out = dict(u=u, kn=u / KN, z=float(v[0]), th=float(v[1]),
               trim_deg=trim_deg(p, v[1]), R=R, T=float(T),
               c=float(T / p.prop.t_max), resid=float(res),
               lw=float(aux["lw"]), n_wet=int((d > 0).sum()),
               n_entry=int((V > 0).sum()), s=s.tolist())
    if "hw" in aux:
        out.update(hw=float(aux["hw"]), dT=float(aux["dT"]))
    cap = getattr(p, "_taper_cap", None)
    out["n_chine_wet"] = int(cap.size) if cap is not None else None
    out["pressure"] = float(-aux["Xp"]) if "Xp" in aux else None
    out["friction"] = float(aux["Df"])
    out["air"] = float(aux["Da"])
    return out


def free_residual(p, s):
    """max |ds| over z..r of the FREE plant's derivative at s (x, y rates
    are the speed itself and are excluded), and the one-step map's change
    over z..delta. Both should be ~1e-12 at an equilibrium."""
    s = np.asarray(s, float)
    cu, p.captive_u = p.captive_u, None
    try:
        ds, _ = p.deriv(s, 0.0)
        prep(p, s[12], s[13])
        s1 = p.step(s.copy(), 0.0, s[12], s[13], p.dt)
    finally:
        p.captive_u = cu
    return float(np.abs(ds[2:12]).max()), float(np.abs(s1[2:] - s[2:]).max())


def census(p, u, zs=(-0.3, 0.45, 6), trims=(-4.0, 12.0, 6)):
    """All roots of the captive equations reached by fsolve from a grid of
    starts (rise x trim), deduplicated, each classified by the eigenvalues
    of the 4 x 4 (z, th, w, q) captive Jacobian: 'stable' if every real
    part is negative. Replicates the ledger's census (D2.9) cheaply."""
    roots = []
    for z in np.linspace(*zs):
        for tr in np.linspace(*trims):
            st = np.array([z, th_of_trim(p, tr)])
            with np.errstate(all="ignore"):
                sol, _, ier, _ = fsolve(lambda w: captive_F(p, u, w)[0], st,
                                        full_output=True, xtol=1e-12)
            if not np.all(np.isfinite(sol)):
                continue
            if np.abs(captive_F(p, u, sol)[0]).max() > 1e-3:
                continue
            v, r = polish(p, u, sol)
            if r > 1e-7:
                continue
            if any(abs(v[0] - q[0]) < 1e-4 and abs(v[1] - q[1]) < 1e-5
                   for q, _ in roots):
                continue
            roots.append((v, r))
    out = []
    for v, r in roots:
        J = captive_jac4(p, u, v[0], v[1])
        lam = np.linalg.eigvals(J)
        out.append(dict(z=float(v[0]), th=float(v[1]),
                        trim_deg=trim_deg(p, v[1]), resid=r,
                        max_re=float(lam.real.max()),
                        stable=bool(lam.real.max() < 0.0),
                        lam=[[float(l.real), float(l.imag)] for l in lam]))
    out.sort(key=lambda d: -d["z"])
    return out


def captive_jac4(p, u, z, th, eps=(1e-6, 1e-7, 1e-6, 1e-7)):
    """d(z', th', w', q')/d(z, th, w, q) of the captive equations."""
    def f4(v):
        s = np.zeros(14)
        s[2], s[4], s[6], s[8], s[10] = v[0], v[1], u, v[2], v[3]
        cu, p.captive_u = p.captive_u, float(u)
        try:
            # the plant's captive copy, as captive_deriv does, but with w, q
            from sim.planing_anyspeed import _StripAS
            v_ = _StripAS.__new__(_StripAS)
            v_.__dict__.update(p.__dict__)
            v_.sea = calm()
            v_._wave_arrays()
            v_.captive_u = float(u)
            ds, _ = v_.deriv(s, 0.0)
        finally:
            p.captive_u = cu
        return np.array([v[2], v[3], ds[8], ds[10]])
    x0 = np.array([z, th, 0.0, 0.0])
    J = np.zeros((4, 4))
    for i in range(4):
        h = np.zeros(4)
        h[i] = eps[i]
        J[:, i] = (f4(x0 + h) - f4(x0 - h)) / (2 * eps[i])
    return J


# ---------------------------------------------------------------- Jacobian
def step_map(p, s, Tcmd, dcmd, dt):
    prep(p, Tcmd, dcmd)
    return p.step(np.array(s, float), 0.0, float(Tcmd), float(dcmd), dt)


def step_jacobian(p, s, eps, dt=None, states=STATES):
    """(Phi, B): central-difference Jacobian of the one-step map over
    `states` (rows and columns in that order) and with respect to the two
    delayed commands [thrust, nozzle]."""
    dt = p.dt if dt is None else dt
    s = np.asarray(s, float)
    Tcmd, dcmd = float(s[12]), float(s[13])
    rows = [IDX[n] for n in states]
    n = len(states)
    Phi = np.zeros((n, n))
    for j, name in enumerate(states):
        i = IDX[name]
        h = eps * scale_of(p, name)
        sp, sm = s.copy(), s.copy()
        sp[i] += h
        sm[i] -= h
        fp = step_map(p, sp, Tcmd, dcmd, dt)
        fm = step_map(p, sm, Tcmd, dcmd, dt)
        Phi[:, j] = (fp[rows] - fm[rows]) / (2.0 * h)
    B = np.zeros((n, 2))
    hT = eps * p.prop.t_max
    fp = step_map(p, s, Tcmd + hT, dcmd, dt)
    fm = step_map(p, s, Tcmd - hT, dcmd, dt)
    B[:, 0] = (fp[rows] - fm[rows]) / (2.0 * hT)
    hd = eps
    fp = step_map(p, s, Tcmd, dcmd + hd, dt)
    fm = step_map(p, s, Tcmd, dcmd - hd, dt)
    B[:, 1] = (fp[rows] - fm[rows]) / (2.0 * hd)
    return Phi, B


def deriv_jacobian(p, s, eps, states=STATES):
    """Continuous Jacobian A_c = d(ds)/ds of the plant's vector field at s
    (free plant, actuator rows analytic: T' = (c - T)/tau, delta' =
    (cmd - delta)/tau_nozzle, linear away from the rate limits and the
    dead band)."""
    s = np.asarray(s, float)
    rows = [IDX[n] for n in states]
    n = len(states)
    A = np.zeros((n, n))
    cu, p.captive_u = p.captive_u, None
    try:
        for j, name in enumerate(states):
            i = IDX[name]
            h = eps * scale_of(p, name)
            sp, sm = s.copy(), s.copy()
            sp[i] += h
            sm[i] -= h
            dp, _ = p.deriv(sp, 0.0)
            dm, _ = p.deriv(sm, 0.0)
            A[:, j] = (dp[rows] - dm[rows]) / (2.0 * h)
    finally:
        p.captive_u = cu
    if "T" in states:
        k = states.index("T")
        A[k, k] = -1.0 / p.prop.tau
    if "d" in states:
        k = states.index("d")
        A[k, k] = -1.0 / p.rudder.tau
    return A


def indicators(p, s):
    """The plant's switches at state s: which stations are wet (d > 0),
    which are entering (V > 0), which have a chine-wetted part in the
    flux (_taper_cap), which have dm_a/dd > 0 (Wagner half-beam inside
    the chine), whether the transom face is wet (h_w > 0), the lateral
    substep count and the prime target."""
    s = np.asarray(s, float)
    cu, p.captive_u = p.captive_u, None
    try:
        ds, aux = p.deriv(s, 0.0)
        d = np.asarray(aux["d"])
        V = np.asarray(aux["V"])
        dp = np.maximum(d, 0.0)
        cw = 0.5 * np.pi * dp / p.tanb
        gate = (cw < p.yc) & (dp > 0)
        cap = getattr(p, "_taper_cap", np.zeros(0, int))
        lam = p._lateral_rate(s, d) * p.dt
        sub = p.prop_submergence(s[:6], 0.0)
    finally:
        p.captive_u = cu
    return dict(wet=frozenset(np.nonzero(d > 0)[0].tolist()),
                entry=frozenset(np.nonzero(V > 0)[0].tolist()),
                chine=frozenset(np.asarray(cap).tolist()),
                dma_gate=frozenset(np.nonzero(gate)[0].tolist()),
                transom_wet=bool(aux.get("hw", 0.0) > 0.0),
                n_sub=int(1 if lam <= 1.8 else math.ceil(lam / 0.5)),
                prime_target=float(np.clip(sub / p.prop.prime_band, 0, 1)))


def switch_changes(p, s, eps, states=STATES):
    """For each perturbed state: the switches that differ between
    s - h e_j and s + h e_j (empty = the linearisation crosses none)."""
    s = np.asarray(s, float)
    out = {}
    for name in states:
        i = IDX[name]
        h = eps * scale_of(p, name)
        sp, sm = s.copy(), s.copy()
        sp[i] += h
        sm[i] -= h
        a, b = indicators(p, sp), indicators(p, sm)
        diff = {}
        for k in a:
            if a[k] != b[k]:
                if isinstance(a[k], frozenset):
                    diff[k] = sorted((a[k] ^ b[k]))
                else:
                    diff[k] = [a[k], b[k]]
        out[name] = diff
    return out


# ------------------------------------------------------------------ modes
def to_continuous(z, dt):
    z = np.asarray(z, complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(z) / dt


def modes(lam):
    """Describe continuous eigenvalues: a list of dicts sorted by |lambda|,
    complex pairs once (imag > 0) with wn, zeta, period; real ones with
    the time constant."""
    out = []
    for l in lam:
        if l.imag < -1e-9:
            continue
        a = abs(l)
        if l.imag > 1e-9:
            out.append(dict(kind="pair", re=float(l.real), im=float(l.imag),
                            wn=float(a), zeta=float(-l.real / a),
                            period=float(2 * np.pi / l.imag)))
        else:
            out.append(dict(kind="real", re=float(l.real), im=0.0,
                            wn=float(a), zeta=(1.0 if l.real < 0 else -1.0),
                            tau=(float(-1.0 / l.real) if abs(l.real) > 1e-12
                                 else float("inf"))))
    out.sort(key=lambda m: m["wn"])
    return out


def block(M, states, names):
    k = [states.index(n) for n in names]
    return M[np.ix_(k, k)]


def coupling(M, states, a, b):
    """Frobenius norm of the a->b off-block of M relative to the norms of
    the diagonal blocks (identity removed for a step map)."""
    ia = [states.index(n) for n in a]
    ib = [states.index(n) for n in b]
    N = M - np.eye(M.shape[0]) if np.allclose(np.diag(M), 1.0, atol=0.5) \
        else M
    off = np.linalg.norm(N[np.ix_(ib, ia)]) + np.linalg.norm(N[np.ix_(ia, ib)])
    on = np.linalg.norm(N[np.ix_(ia, ia)]) + np.linalg.norm(N[np.ix_(ib, ib)])
    return float(off / max(on, 1e-300))


def analyse(Phi, dt, states=STATES):
    """Eigen-analysis of a one-step Jacobian: continuous eigenvalues of
    the vertical and lateral blocks and of the whole, as mode lists."""
    res = {}
    for lab, names in (("vertical", VERT), ("lateral", LAT)):
        Pb = block(Phi, states, names)
        lam = to_continuous(np.linalg.eigvals(Pb), dt)
        res[lab] = dict(states=list(names), modes=modes(lam),
                        lam=[[float(l.real), float(l.imag)] for l in lam])
    lam = to_continuous(np.linalg.eigvals(Phi), dt)
    res["all"] = dict(modes=modes(lam),
                      lam=[[float(l.real), float(l.imag)] for l in lam])
    res["coupling_vert_lat"] = coupling(Phi, states, VERT, LAT)
    return res


def linearity(P1, P2):
    """Per-column relative difference of two Jacobians of the same map:
    ||P1[:, j] - P2[:, j]|| / ||P1[:, j] - e_j|| (the non-identity part)."""
    n = P1.shape[0]
    out = []
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        den = np.linalg.norm(P1[:, j] - e)
        out.append(float(np.linalg.norm(P1[:, j] - P2[:, j])
                         / max(den, 1e-300)))
    return out


def tolist(a):
    return np.asarray(a, float).tolist()
