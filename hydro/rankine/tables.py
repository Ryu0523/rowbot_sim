"""
Radiation tables for the any-speed plant -- Track B step B5 interface (port
design WAVEBEM_PORT_DESIGN.md 8.5; stage note RANKINE_STAGE1_2026-10-06.md
sections 10.6 and 10.8). The tables are COMPUTED by the forward-speed LTI solver
(unsteady.py); nothing is fitted.

Force law consumed by the plant (B5 review, stage note 10.8.5; per speed U and
slice attitude qbar = (sinkage, trim) at which the table is sliced; symmetric
modes surge, heave, pitch; x forward, z up, pitch right-handed about +y, i.e.
positive = bow DOWN; moments about the table's reference point):

    F(t) = -A_inf a(t) - B_inf v(t) - int_0^t K(t - s) v(s) ds - (C0 + C_s) (q(t) - qbar)
           + F_steady(U, qbar)          (+ Froude-Krylov/hydrostatics, exact, plant side)

C_s (added after the B6 review, stage note 12.2 item 4; zero for the uniform-stream
tables): the displacement stiffness of the FROZEN base pressure rho (U^2 - |W|^2)/2
of the double-body linearisation (doublebody.frozen_pressure_stiffness) -- the
hull moving through the steady pressure field, which the radiation problem's
omega = 0 response C0 does not contain (B6 used C0 alone: incomplete).

i.e. the LTI system of the radiation solver about the slice attitude, exactly as
verified (check CU), plus the steady force at the slice attitude only. This is
the linear model's own consistent law: the frequency-domain identity of the same
system is
    -omega^2 A(omega) + i omega B(omega) = -omega^2 A_inf + i omega B_inf + C0 + i omega K^(omega),
    K(t) = (2/pi) int_0^inf (B(omega) - B_inf) cos(omega t) d omega   (integrable part),
    C0 = lim_{omega -> 0} [-omega^2 A + i omega B]        (static hydrodynamic stiffness),
and the static stiffness of the radiation problem is C0, NOT the attitude
derivative of F_steady: in the uniform-stream linearisation the two differ
(stage note 10.5 item 9, check C0B), so the B5 law "F_steady(U, q(t)) and no
C0" mixed two different linear models and is withdrawn. The plant chooses qbar
(e.g. the free-equilibrium attitude of the current speed) and re-slices when
qbar or U change; q - qbar is the oscillatory part of the attitude. Both force
forms are stored (direct pressure "D" and Ogilvie-Tuck "OT"); each kernel, B_inf
and C0 belong to their own form and must not be mixed (interp_slice(form=...)).

Plant kernels (stage note 10.8.6): above the frequency where the zero-speed
wavelength spans 4 cells of the table's grid (even the zero-speed waves are
unrepresentable there), the kernels are formed from
B_inf + W(omega) (B - B_inf) with a cos^2 window W from omega_t to 2 omega_t
(numerical device registered in EMPIRICAL_REGISTER_RANKINE.md). The arrays A, B
are stored unmodified; the KK distortion of the windowed kernel (|A_K - A| in the
validity band) is recorded per node (`taper_dA`).

Storage: one .npz per hull and physics version (arrays below, float64 /
complex128) plus a JSON sidecar with the metadata and the sha256 of the npz
(acceptance N3: cached by physics-version hash and hull name; old files are
never overwritten -- save_table refuses an existing path).

Arrays (nU speeds, ns sinkages, nt trims, nw frequencies, nk kernel times, m modes):
    U (nU,) [m/s], sink (ns,) [m, > 0 down], trim (nt,) [rad, > 0 bow UP, hull.py convention]
    omega (nw,) [rad/s]   (oscillation = encounter frequency)
    A, B        (nU, ns, nt, nw, m, m)    direct-pressure (D-form)
    A_inf, B_inf, C0, C_inf  (nU, ns, nt, m, m)
    t (nk,) [s], K (nU, ns, nt, nk, m, m)  (windowed plant kernel)
    A_ot, B_ot, B_inf_ot, C0_ot, C_inf_ot, K_ot   Ogilvie-Tuck (m-term) form
    omega_lo, omega_hi (nU, ns, nt)        validity band per node (forward-speed dispersion, 8 cells)
    omega_t (nU, ns, nt)                   start of the kernel window (zero-speed wavelength = 4 cells)
    passive, passive_ot (nU, ns, nt)       symmetric part of B(omega) >= 0 at every omega of the grid (flag)
    bsym_min, bsym_min_ot (nU, ns, nt)     min over omega of lambda_min(B_sym)/max abs lambda(B_sym)
    taper_dA, taper_dA_ot (nU, ns, nt)     max over the validity band of |A_K - A| / max |A| (KK distortion of
                                           the windowed kernel)
"""
import datetime
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

TABLE_VERSION = "rankine-radiation-3"      # 3: C_s in the plant law (B6 review)
MODES = (1, 3, 5)


@dataclass
class RadiationTable:
    hull_name: str
    L: float
    rho: float
    g: float
    ref: tuple                      # body-frame (x_ref, z_ref) of the moments
    U: np.ndarray
    sink: np.ndarray
    trim: np.ndarray
    omega: np.ndarray
    A: np.ndarray
    B: np.ndarray
    A_inf: np.ndarray
    B_inf: np.ndarray
    C0: np.ndarray
    C_inf: np.ndarray
    t: np.ndarray
    K: np.ndarray
    omega_lo: np.ndarray
    omega_hi: np.ndarray
    passive: np.ndarray
    A_ot: np.ndarray = None
    passive_ot: np.ndarray = None
    bsym_min: np.ndarray = None
    bsym_min_ot: np.ndarray = None
    B_ot: np.ndarray = None
    B_inf_ot: np.ndarray = None
    C0_ot: np.ndarray = None
    K_ot: np.ndarray = None
    C_inf_ot: np.ndarray = None
    omega_t: np.ndarray = None
    taper_dA: np.ndarray = None
    taper_dA_ot: np.ndarray = None
    C_s: np.ndarray = None
    modes: tuple = MODES
    meta: dict = field(default_factory=dict)

    ARRAYS = ("U", "sink", "trim", "omega", "A", "B", "A_inf", "B_inf", "C0", "C_inf", "t", "K", "omega_lo",
              "omega_hi", "passive", "A_ot", "B_ot", "B_inf_ot", "C0_ot", "K_ot", "passive_ot", "bsym_min",
              "bsym_min_ot", "C_inf_ot", "omega_t", "taper_dA", "taper_dA_ot", "C_s")


# ------------------------------------------------------------------ dispersion, bands, window
def along_track_wavenumbers(omega, U, g):
    """Wavenumbers of the waves radiated along the track by a source
    oscillating at omega in a stream of speed |U| (deep water; in the body
    frame (omega -+ U k)^2 = g k): returns (k0, k1, k2, k3, k4) with k0 =
    omega^2/g, k1 < k2 the roots of (omega - U k)^2 = g k and k3 < k4 those of
    (omega + U k)^2 = g k (NaN for tau = |U| omega/g >= 1/4). Closed forms
    (tau = |U| omega/g): k1 = 2 k0/((1 + 2 tau) + sqrt(1 + 4 tau)), k2 = (g +
    2 U omega + sqrt(g^2 + 4 g U omega))/(2 U^2), k3 = 2 k0/((1 - 2 tau) +
    sqrt(1 - 4 tau)), k4 = (g - 2 U omega + sqrt(g^2 - 4 g U omega))/(2 U^2);
    k2 > k4 > k3 >= k0 >= k1 for 0 < tau < 1/4."""
    omega = np.asarray(omega, float)
    U = abs(float(U))
    k0 = omega ** 2 / g
    if U == 0.0:
        return k0, k0, k0, k0, k0
    tau = U * omega / g
    k1 = 2 * k0 / ((1 + 2 * tau) + np.sqrt(1 + 4 * tau))
    k2 = (g + 2 * U * omega + np.sqrt(g * g + 4 * g * U * omega)) / (2 * U * U)
    with np.errstate(invalid="ignore"):
        ok = tau < 0.25
        r = np.sqrt(np.where(ok, 1 - 4 * tau, np.nan))
        k3 = np.where(ok, 2 * k0 / ((1 - 2 * tau) + r), np.nan)
        k4 = np.where(ok, (g - 2 * U * omega + g * r) / (2 * U * U), np.nan)
    return k0, k1, k2, k3, k4


def _kmax(omega, U, g):
    k0, k1, k2, k3, k4 = along_track_wavenumbers(omega, U, g)
    return np.nanmax(np.stack([np.atleast_1d(k0), np.atleast_1d(k2), np.atleast_1d(k4)]), axis=0)


def _solve_increasing(fun, target, w_hi=1e3):
    """Largest omega in [0, w_hi] with fun(omega) <= target for an increasing fun (bisection);
    0 if fun(0) > target."""
    if float(fun(0.0)) > target:
        return 0.0
    lo, hi = 0.0, w_hi
    if float(fun(hi)) <= target:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(fun(mid)) <= target:
            lo = mid
        else:
            hi = mid
    return lo


def validity_band(G, U, g, L, cells=8.0):
    """Validity band of a free-surface grid for the oscillation (encounter)
    frequency, with the forward-speed dispersion (registered rule, stage note
    10.8.6 and its amendment 1; EMPIRICAL_REGISTER_RANKINE.md): omega_hi = the
    largest omega at which the SHORTEST along-track wave, max(k0, k2, k4
    [tau < 1/4]), spans `cells` cells of the hull-region grid (dx = L_wl/M);
    omega_lo = the smallest omega at which every beach is at least one
    wavelength of the waves that reach it: the lateral beach of the transverse
    waves (k = k0 at 90 deg to the track), the downstream beach of the longest
    along-track wave k1, the upstream beach of k3 (only for tau < 1/4; no
    waves travel upstream above it). Values outside are solutions of the same
    causal system (Kramers-Kronig holds) but not resolution-verified. Returns
    (omega_lo, omega_hi); omega_hi = 0 means that even the steady Kelvin wave
    2 pi U^2/g is shorter than `cells` cells; omega_lo > omega_hi means an
    empty band."""
    dx = G["dx"]
    p = G["params"]
    w_hi = _solve_increasing(lambda w: _kmax(w, U, g)[0], 2 * np.pi / (cells * dx))

    def fits(w):
        k0, k1, k2, k3, k4 = along_track_wavenumbers(w, U, g)
        ok = float(k0) >= 2 * np.pi / (p["b_lat"] * L) and float(k1) >= 2 * np.pi / (p["b_dn"] * L)
        if np.isfinite(k3):
            ok = ok and float(k3) >= 2 * np.pi / (p["b_up"] * L)
        return ok
    lo, hi = 0.0, 1e3
    for _ in range(200):                 # each condition is monotone in omega
        mid = 0.5 * (lo + hi)
        if fits(mid):
            hi = mid
        else:
            lo = mid
    return float(hi), float(w_hi)


def kernel_window_start(G, U, g, cells=4.0):
    """omega_t: the frequency at which the ZERO-SPEED wavelength 2 pi g/omega^2 spans `cells`
    hull-region cells (above it even the waves of the zero-speed system are not representable on
    the grid; start of the kernel window; U is not used). Exploration B5R-E8 (stage note 10.8.6):
    starting the window at the dispersion-aware omega_hi (k2 waves, 8 cells) or at 4 cells of the k2
    wave removes resolved damping near the heave-pitch natural frequencies at low Fn, distorts
    A(omega) by 6-20 % through Kramers-Kronig and made the plant law unstable at one node."""
    return float(np.sqrt(2 * np.pi * g / (cells * G["dx"])))


def taper_window(omega, w_a, w_b):
    """1 for omega <= w_a, cos^2 roll-off to 0 at w_b, 0 beyond."""
    omega = np.asarray(omega, float)
    if w_b <= w_a:
        return (omega <= w_a).astype(float)
    s = np.clip((omega - w_a) / (w_b - w_a), 0.0, 1.0)
    return np.cos(0.5 * np.pi * s) ** 2


# ------------------------------------------------------------------ kernels
def memory_kernel(omega, B, B_inf, t, n_fine=200001, taper=None):
    """K(t) = (2/pi) int_0^inf W(w) (B(w) - B_inf) cos(w t) dw for B sampled on
    `omega` (ascending, (nw, m, m)): monotone cubic (PCHIP) interpolation of
    B - B_inf on a uniform fine grid from 0 (constant extension below
    omega[0]) to omega[-1], trapezoidal rule; beyond omega[-1] the integrand is
    taken as 0. taper = (w_a, w_b): window W of taper_window (None: W = 1).
    Returns (K, tail) with tail = max |W (B - B_inf)| at omega[-1] / max |B - B_inf|."""
    from scipy.interpolate import PchipInterpolator
    omega = np.asarray(omega, float)
    F = np.asarray(B, float) - np.asarray(B_inf, float)[None]
    wf = np.linspace(0.0, omega[-1], n_fine)
    Ff = PchipInterpolator(omega, F, axis=0, extrapolate=False)(np.maximum(wf, omega[0]))
    Ff[wf < omega[0]] = F[0]
    if taper is not None:
        Ff = Ff * taper_window(wf, *taper)[:, None, None]
    t = np.asarray(t, float)
    K = np.empty((len(t),) + F.shape[1:])
    dw = wf[1] - wf[0]
    wts = np.full(len(wf), dw)
    wts[0] = wts[-1] = 0.5 * dw
    for i, tt in enumerate(t):
        K[i] = (2 / np.pi) * np.tensordot(wts * np.cos(wf * tt), Ff, axes=(0, 0))
    Wend = taper_window(omega[-1:], *taper)[0] if taper is not None else 1.0
    tail = float(Wend * np.abs(F[-1]).max() / max(np.abs(F).max(), 1e-300))
    return K, tail


def kk_added_mass(omega, B, B_inf, A_inf, C0, wq, taper=None, n_fine=100001):
    """A_K(wq) of the (windowed) kernel by the forward-speed Kramers-Kronig
    relation A = A_inf - C0/w^2 + (2/pi) PV int (W (B - B_inf))/(w'^2 - w^2) dw'
    (subtraction of the integrand at w on [0, w_end]); used for the KK
    distortion of the kernel window (taper_dA)."""
    from scipy.interpolate import PchipInterpolator
    omega = np.asarray(omega, float)
    F = np.asarray(B, float) - np.asarray(B_inf, float)[None]
    wf = np.linspace(0.0, omega[-1], n_fine)
    Ff = PchipInterpolator(omega, F, axis=0, extrapolate=False)(np.maximum(wf, omega[0]))
    Ff[wf < omega[0]] = F[0]
    if taper is not None:
        Ff = Ff * taper_window(wf, *taper)[:, None, None]
    out = []
    we = wf[-1]
    m1, m2 = Ff.shape[1], Ff.shape[2]
    for w in np.atleast_1d(wq):
        fw = np.array([[np.interp(w, wf, Ff[:, i, j]) for j in range(m2)] for i in range(m1)])
        den = wf ** 2 - w * w
        with np.errstate(divide="ignore", invalid="ignore"):
            ig = np.where(np.abs(den)[:, None, None] > 1e-12, (Ff - fw[None]) / den[:, None, None], 0.0)
        I1 = np.trapezoid(ig, wf, axis=0)
        I2 = fw * np.log(abs((we - w) / (we + w))) / (2 * w)
        out.append(np.asarray(A_inf) - np.asarray(C0) / w ** 2 + 2 / np.pi * (I1 + I2))
    return np.array(out)


def static_stiffness(co, d_form=True):
    """C0 = lim_{omega -> 0} T(omega): the omega = 0 solve of the radiation
    problem with the displacement (m-term) body condition only (T at omega = 0
    is -rho int (-U d_x phi_k) n_j (D-form) or rho int phi_k m_j (OT-form)).
    Double-body coefficients (doublebody.DBCoefficients, stage B6): their own
    static_stiffness (D-form with W.grad)."""
    if hasattr(co, "static_stiffness"):
        return co.static_stiffness()
    s = co.s
    a = s.body.area
    m = len(co.modes)
    ph = np.stack([co.mv[k][1] for k in co.modes], 1).astype(complex)
    if s.U == 0:
        return np.zeros((m, m)), np.zeros((m, m))
    sol = s.solve_freq(0.0, ph)
    from .unsteady import hull_dx
    C = np.zeros((m, m))
    Cot = np.zeros((m, m))
    for c in range(m):
        phk = sol["phi_B"][:, c].real
        dxk = hull_dx(s.body, phk, ph[:, c].real, s.hull_shape)
        for r, j in enumerate(co.modes):
            nj, mj = co.mv[j]
            C[r, c] = -s.rho * 2 * np.sum((-s.U * dxk) * nj * a)
            Cot[r, c] = s.rho * 2 * np.sum(phk * mj * a)
    return C, Cot


# ------------------------------------------------------------------ build
def build_solver(hull, U, sink, trim, grid_kw, hull_mesh, rho=1025.0, g=9.81, ref=(0.0, 0.0), device=None,
                 linearisation="NK"):
    """One LTI system (unsteady.ForwardSpeedLTI with the wall-trace closure) at
    (U, sink, trim): hull panelled at the attitude, free-surface grid
    build_grid(hull, sink, trim, sign(U), **grid_kw) (U = 0: symmetric extents;
    U < 0: astern, mirrored grid). Returns (solver, Coefficients, grid).
    linearisation "DB" (stage B6): the double-body system of doublebody.py
    (conservative kinematic operator, Tuck-exact m-terms; D and OT forms
    coincide)."""
    if linearisation == "DB":
        from .doublebody import build_db_solver
        s, co, G, _ = build_db_solver(hull, U, sink, trim, grid_kw, hull_mesh, g=g, rho=rho, ref=ref, device=device)
        return s, co, G
    if linearisation != "NK":
        raise ValueError(linearisation)
    from .unsteady import build_grid, dx_operator, wall_trace_data, ForwardSpeedLTI, Coefficients
    sgn = 1.0 if U >= 0 else -1.0
    body = hull.mesh(*hull_mesh, sink=sink, trim=trim)
    kw = dict(grid_kw)
    if U == 0:
        kw["down"] = kw.get("up", 1.5)
        kw["b_dn"] = kw.get("b_up", 0.75)
    G = build_grid(hull, sink, trim, sgn, **kw)
    D, info = dx_operator(G, sgn, hull=hull, sink=sink, trim=trim)
    wt = wall_trace_data(info["wall"], hull, body, hull_mesh)
    s = ForwardSpeedLTI(body, G["fs"], D, G["nu"] * np.sqrt(g / hull.L), U, g, rho, device=device,
                        hull_shape=hull_mesh, wall=wt)
    refE = hull.to_earth(np.array(ref[0]), np.array(0.0), np.array(ref[1]), sink, trim)
    co = Coefficients(s, ref=(float(refE[0]), float(refE[2])))
    return s, co, G


def node_slice(hull, U, sink, trim, omega, t, grid_kw, hull_mesh, rho=1025.0, g=9.81, ref=(0.0, 0.0),
               device=None, linearisation="NK"):
    """All table quantities of ONE (U, sink, trim) node (used by build_radiation_table and by the direct
    mid-box check PVm): dict of A, B, A_ot, B_ot (nw, m, m), A_inf, B_inf, C_inf, B_inf_ot, C_inf_ot, C0,
    C0_ot, K, K_ot, omega_lo, omega_hi, omega_t, kernel tails, passivity, taper_dA."""
    omega = np.asarray(omega, float)
    s, co, G = build_solver(hull, U, sink, trim, grid_kw, hull_mesh, rho, g, ref, device, linearisation)
    m = len(MODES)
    out = dict(NF=s.NF)
    inst = co.instantaneous()
    out.update(A_inf=inst["A"], B_inf=inst["B"], C_inf=inst["C"], B_inf_ot=inst["B_ot"], C_inf_ot=inst["C_ot"])
    out["C0"], out["C0_ot"] = static_stiffness(co)
    if linearisation == "DB":
        from .doublebody import frozen_pressure_stiffness
        out["C_s"] = frozen_pressure_stiffness(co.db, co.ref, rho)
    else:
        out["C_s"] = np.zeros((m, m))
    A = np.zeros((len(omega), m, m))
    B, Aot, Bot = np.zeros_like(A), np.zeros_like(A), np.zeros_like(A)
    for iw, w in enumerate(omega):
        r = co.radiation(w)
        T, To = r["T"], r["T_ot"]
        A[iw], B[iw] = -T.real / w ** 2, T.imag / w
        Aot[iw], Bot[iw] = -To.real / w ** 2, To.imag / w
    out.update(A=A, B=B, A_ot=Aot, B_ot=Bot)
    wlo, whi = validity_band(G, U, g, hull.L)
    wt = kernel_window_start(G, U, g)
    out.update(omega_lo=wlo, omega_hi=whi, omega_t=wt)
    taper = (wt, 2.0 * wt) if wt > 0 else None
    out["K"], out["tail"] = memory_kernel(omega, B, inst["B"], t, taper=taper)
    out["K_ot"], out["tail_ot"] = memory_kernel(omega, Bot, inst["B_ot"], t, taper=taper)
    for key, Bx in (("", B), ("_ot", Bot)):
        Bs = 0.5 * (Bx + np.swapaxes(Bx, 1, 2))
        ev = np.linalg.eigvalsh(Bs)
        rel = ev[:, 0] / np.maximum(np.abs(ev).max(axis=1), 1e-300)
        out["bsym_min" + key] = float(rel.min())
        out["passive" + key] = bool(rel.min() >= -1e-9)
    # KK distortion of the windowed kernel inside the validity band
    for key, Ax, Bx, Bi, C0x in (("", A, B, inst["B"], out["C0"]), ("_ot", Aot, Bot, inst["B_ot"], out["C0_ot"])):
        band = (omega >= wlo) & (omega <= whi)
        if taper is None or not band.any():
            out["taper_dA" + key] = float("nan")
            continue
        Ak = kk_added_mass(omega, Bx, Bi, inst["A"], C0x, omega[band], taper=taper)
        Ak0 = kk_added_mass(omega, Bx, Bi, inst["A"], C0x, omega[band], taper=None)
        out["taper_dA" + key] = float(np.abs(Ak - Ak0).max() / max(np.abs(Ax[band]).max(), 1e-300))
    s.release()
    del s, co
    return out


def build_radiation_table(hull, U_list, sink_list, trim_list, omega, t, grid_kw, hull_mesh, rho=1025.0, g=9.81,
                          ref=(0.0, 0.0), device=None, verbose=False, linearisation="NK"):
    """Compute a RadiationTable (node_slice at every (U, sink, trim)). ref:
    moments about the body-frame point (x_ref, z_ref), carried with the
    attitude. omega in rad/s, t in s."""
    import time
    U_list = np.asarray(U_list, float)
    sink_list = np.asarray(sink_list, float)
    trim_list = np.asarray(trim_list, float)
    omega = np.asarray(omega, float)
    t = np.asarray(t, float)
    m = len(MODES)
    shp = (len(U_list), len(sink_list), len(trim_list))
    nw, nk = len(omega), len(t)
    arr = {k: np.zeros(shp + (nw, m, m)) for k in ("A", "B", "A_ot", "B_ot")}
    arr.update({k: np.zeros(shp + (m, m)) for k in ("A_inf", "B_inf", "C0", "C_inf", "B_inf_ot", "C0_ot",
                                                     "C_inf_ot", "C_s")})
    arr.update({k: np.zeros(shp + (nk, m, m)) for k in ("K", "K_ot")})
    arr.update({k: np.zeros(shp) for k in ("omega_lo", "omega_hi", "omega_t", "bsym_min", "bsym_min_ot", "taper_dA",
                                           "taper_dA_ot")})
    arr.update({k: np.zeros(shp, bool) for k in ("passive", "passive_ot")})
    tails = []
    t0 = time.time()
    for iu, U in enumerate(U_list):
        for i_s, sk in enumerate(sink_list):
            for i_t, tr in enumerate(trim_list):
                nd = node_slice(hull, U, sk, tr, omega, t, grid_kw, hull_mesh, rho, g, ref, device, linearisation)
                for k in arr:
                    arr[k][iu, i_s, i_t] = nd[k]
                tails += [nd["tail"], nd["tail_ot"]]
                if verbose:
                    print(f"  table U={U:.3f} s={sk:.4f} t={tr:.4f}: NF {nd['NF']} [{time.time() - t0:.0f} s]", flush=True)
    meta = dict(version=TABLE_VERSION, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"), grid_kw=grid_kw, hull_mesh=list(hull_mesh),
        force_law="F = -A_inf a - B_inf v - int K v ds - (C0 + C_s) (q - qbar) + F_steady(U, qbar) (stage note 10.8.5; "
        "C_s since the B6 review, 12.2 item 4; zero for NK)",
        forms="D (direct pressure) and OT (Ogilvie-Tuck); kernels, B_inf, C0, C_inf of one form only",
        modes=list(MODES), conventions="x forward, z up, pitch about +y (bow down positive); trim axis of the "
        "table in hull.py convention (bow up positive); moments about ref (body frame), F_rad = -T x",
        kernel_window="B_inf + W (B - B_inf), W cos^2 from omega_t (zero-speed wavelength = 4 cells) to "
        "2 omega_t (stage note 10.8.6)", kernel_tail_max=float(max(tails)) if tails else None,
        omega_units="rad/s", t_units="s",
        linearisation=("Neumann-Kelvin (uniform stream), stage note 10.2" if linearisation == "NK" else
                       "double body (steady base flow), stage note 11.2; D and OT forms coincide"),
        wall_closure="trace (stage note 10.8.2)" if linearisation == "NK" else "characteristic wall, mirror (11.2 item 3)")
    return RadiationTable(hull_name=hull.name, L=float(hull.L), rho=float(rho), g=float(g), ref=tuple(ref), U=U_list,
                          sink=sink_list, trim=trim_list, omega=omega, t=t, modes=MODES, meta=meta, **arr)


# ------------------------------------------------------------- storage
def save_table(tab, path):
    """npz + JSON sidecar (metadata, scalars, sha256 of the npz). Refuses to
    overwrite (acceptance N3)."""
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: tables are versioned, never overwritten")
    arrs = {k: getattr(tab, k) for k in RadiationTable.ARRAYS if getattr(tab, k) is not None}
    np.savez(path, **arrs)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    side = dict(hull_name=tab.hull_name, L=tab.L, rho=tab.rho, g=tab.g, ref=list(tab.ref), modes=list(tab.modes),
                meta=tab.meta, npz_sha256=sha)
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(side, f, indent=1, default=str)
    return sha


def load_table(path):
    side = json.load(open(os.path.splitext(path)[0] + ".json", encoding="utf-8"))
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    if sha != side["npz_sha256"]:
        raise ValueError(f"{path}: sha256 mismatch (file changed after it was written)")
    z = np.load(path)
    arrs = {k: (z[k] if k in z.files else None) for k in RadiationTable.ARRAYS}
    return RadiationTable(side["hull_name"], side["L"], side["rho"], side["g"], tuple(side["ref"]),
                          modes=tuple(side["modes"]), meta=side["meta"], **arrs)


def table_path(root, hull_name, physics_hash, label):
    """studies/_cache/rankine/<hull>@<physics>/<label>.npz (design 8.5)."""
    d = os.path.join(root, f"{hull_name}@{physics_hash}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{label}.npz")


# -------------------------------------------------------- plant side
def interp_slice(tab, U, sink, trim, form="D"):
    """Multilinear interpolation in (U, sink, trim) inside the table's box
    (outside: ValueError; the plant must not extrapolate). Returns dict(A_inf,
    B_inf, C0, C_inf, t, K) of ONE force form ("D" or "OT"; A_inf is common) and C_s (form-independent frozen
    base-pressure stiffness; zeros for tables without it)."""
    def axis_w(grid, v):
        grid = np.asarray(grid)
        if len(grid) == 1:
            if abs(v - grid[0]) > 1e-12 * max(1.0, abs(v)):
                raise ValueError("value outside a single-point table axis")
            return [(0, 1.0)]
        if v < grid[0] - 1e-12 or v > grid[-1] + 1e-12:
            raise ValueError(f"{v} outside the table axis [{grid[0]}, {grid[-1]}]")
        i = int(np.clip(np.searchsorted(grid, v) - 1, 0, len(grid) - 2))
        f = (v - grid[i]) / (grid[i + 1] - grid[i])
        return [(i, 1 - f), (i + 1, f)]
    if form not in ("D", "OT"):
        raise ValueError("form must be 'D' or 'OT'")
    wu, ws, wt = axis_w(tab.U, U), axis_w(tab.sink, sink), axis_w(tab.trim, trim)
    if form == "D":
        keys = dict(A_inf="A_inf", B_inf="B_inf", C0="C0", C_inf="C_inf", K="K")
    else:
        if tab.C_inf_ot is None:
            raise ValueError("table without C_inf_ot (written before the B5 review): OT slice not available")
        keys = dict(A_inf="A_inf", B_inf="B_inf_ot", C0="C0_ot", C_inf="C_inf_ot", K="K_ot")
    out = {}
    for k, src in keys.items():
        arr = getattr(tab, src)
        acc = 0.0
        for iu, a in wu:
            for i_s, b in ws:
                for i_t, c in wt:
                    acc = acc + a * b * c * arr[iu, i_s, i_t]
        out[k] = acc
    if tab.C_s is not None:
        acc = 0.0
        for iu, a in wu:
            for i_s, b in ws:
                for i_t, c in wt:
                    acc = acc + a * b * c * tab.C_s[iu, i_s, i_t]
        out["C_s"] = acc
    else:
        out["C_s"] = np.zeros_like(np.asarray(out["C0"]))
    out["t"] = tab.t
    out["form"] = form
    return out


class RadiationForce:
    """Plant-side evaluation of the documented force law (module docstring)
        F_rad(t) = -A_inf a - B_inf v - int K(t-s) v ds - (C0 + C_s) x  (C_s if the slice has it),
    x = q - qbar the displacement from the slice attitude (the plant adds
    F_steady(U, qbar)). One table slice (fixed U, qbar over the convolution
    window; the plant re-slices when U or qbar change). Convolution by the
    trapezoidal rule on the uniform step dt of the plant; the kernel is
    resampled on that step up to the table's last time."""

    def __init__(self, sl, dt, stiffness=True):
        self.dt = float(dt)
        self.A_inf, self.B_inf = np.asarray(sl["A_inf"]), np.asarray(sl["B_inf"])
        self.C0 = (np.asarray(sl["C0"]) + np.asarray(sl.get("C_s", 0.0))) if stiffness else None
        tk = np.asarray(sl["t"])
        n = int(np.floor(tk[-1] / dt)) + 1
        self.tk = dt * np.arange(n)
        Ksl = np.asarray(sl["K"])
        self.K = np.stack([np.stack([np.interp(self.tk, tk, Ksl[:, i, j]) for j in range(Ksl.shape[2])], -1)
                           for i in range(Ksl.shape[1])], 1)
        self._buf = np.zeros((max(1024, 2 * len(self.tk)), self.K.shape[2]))   # velocity history (grows)
        self._n = 0

    def convolution(self, v):
        """int K(t-s) v(s) ds with v the current velocity appended to the history (trapezoidal rule on the
        plant step; history kept in a growing array, newest last)."""
        if self._n == len(self._buf):
            self._buf = np.concatenate([self._buf, np.zeros_like(self._buf)])
        self._buf[self._n] = np.asarray(v, float)
        self._n += 1
        n = min(self._n, len(self.tk))
        V = self._buf[self._n - n:self._n][::-1]            # v(t - k dt), k = 0..n-1
        w = np.full(n, self.dt)
        w[0] = w[-1] = 0.5 * self.dt
        if n == 1:
            w[0] = 0.0
        return np.tensordot(self.K[:n], w[:, None] * V, axes=([0, 2], [0, 1]))

    def __call__(self, a, v, x=None):
        """a, v, x: current mode accelerations, velocities and displacements from the slice attitude (m,);
        returns F_rad (m,). x is required when the slice stiffness C0 is applied (stiffness=True)."""
        F = -self.A_inf @ np.asarray(a, float) - self.B_inf @ np.asarray(v, float) - self.convolution(v)
        if self.C0 is not None:
            if x is None:
                raise ValueError("RadiationForce(stiffness=True) needs the displacement x = q - qbar")
            F = F - self.C0 @ np.asarray(x, float)
        return F


def free_decay(sl, mass, C_hs, x0, dt, t_end, modes=(1, 2)):
    """Free decay of the plant law with hydrostatics for the selected modes
    (indices into the table's modes; default heave, pitch): (M + A_inf) a =
    -B_inf v - int K v ds - (C_hs + C0 + C_s) x, symplectic Euler, convolution as
    RadiationForce. mass, C_hs: (m, m) over the selected modes. Returns dict(t,
    x, v, E, Ep) with E = 1/2 v.(M + A_inf)_sym v + 1/2 x.(C_hs + C0 + C_s)_sym x
    (indefinite when the symmetric part of the non-symmetric stiffness is: B6
    review, finding B4) and the positive-definite norm Ep = 1/2 v.(M + A_inf)_sym v
    + 1/2 x.C_hs x (used by the corrected decay criterion, stage note 12.2 item 5)."""
    ix = np.array(modes)
    sub = {k: np.asarray(sl[k])[np.ix_(ix, ix)] for k in ("A_inf", "B_inf", "C0")}
    sub["C0"] = sub["C0"] + np.asarray(sl.get("C_s", np.zeros_like(np.asarray(sl["C0"]))))[np.ix_(ix, ix)]
    Ksub = np.asarray(sl["K"])[:, ix][:, :, ix]
    rf = RadiationForce(dict(A_inf=sub["A_inf"], B_inf=sub["B_inf"], C0=sub["C0"], K=Ksub, t=sl["t"]), dt,
                        stiffness=False)
    Mt = np.asarray(mass) + sub["A_inf"]
    Minv = np.linalg.inv(Mt)
    Ct = np.asarray(C_hs) + sub["C0"]
    Ms, Cs = 0.5 * (Mt + Mt.T), 0.5 * (Ct + Ct.T)
    n = int(round(t_end / dt))
    x = np.asarray(x0, float).copy()
    v = np.zeros_like(x)
    Ch = 0.5 * (np.asarray(C_hs, float) + np.asarray(C_hs, float).T)
    T, X, Vv, E, Ep = [], [], [], [], []
    for k in range(n + 1):
        T.append(k * dt)
        X.append(x.copy())
        Vv.append(v.copy())
        E.append(0.5 * v @ Ms @ v + 0.5 * x @ Cs @ x)
        Ep.append(0.5 * v @ Ms @ v + 0.5 * x @ Ch @ x)
        if k == n:
            break
        a = Minv @ (-sub["B_inf"] @ v - rf.convolution(v) - Ct @ x)
        v = v + dt * a
        x = x + dt * v
        if not np.all(np.isfinite(x)):
            break
    return dict(t=np.array(T), x=np.array(X), v=np.array(Vv), E=np.array(E), Ep=np.array(Ep))
