#!/usr/bin/env python3
"""
M3 -- linear 6-DOF Cummins model in the time domain.

    (M + A_inf) nu' + mu(t) + C eta = tau_exc(t)
    eta' = nu
    mu(t) = radiation memory, carried by state-space systems instead of a
            convolution

The memory is stored as one small linear system per (i,j) coefficient pair,
assembled into a single block-diagonal system so the whole vessel is one ODE:

    x'  = Ar x + Br nu
    mu  = Cr x

Zero forward speed here, deliberately. It is the case where the frequency
domain has an exact answer,

    [-w^2 (M + A(w)) + i w B(w) + C] xi = F_exc(w)

so the time-domain integration can be checked against something exact rather
than against another approximation. Forward speed, nonlinear Froude-Krylov and
viscous corrections come in M4, on top of a base that has been verified.
"""
import numpy as np

from hydro.retardation import fit_coefficient

DOF_NAMES = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]


_RAD_CACHE = {}


def radiation_memory(db, order="auto", rel_tol=1e-3, verbose=False):
    """Cached RadiationMemory.

    Fitting the 10 non-zero coefficient pairs costs a couple of seconds. The
    weight-tuning loop builds hundreds of episodes on the SAME hull, so without
    a cache the identification dominates the run time and nothing about it
    changes between episodes.
    """
    key = (id(db), order, rel_tol)
    if key not in _RAD_CACHE:
        _RAD_CACHE[key] = RadiationMemory(db, order=order, rel_tol=rel_tol,
                                          verbose=verbose)
    return _RAD_CACHE[key]


class RadiationMemory:
    """Block-diagonal state space reproducing the full 6x6 retardation matrix.

    Only pairs whose radiation damping is non-negligible are fitted; for a hull
    with port-starboard symmetry the longitudinal group (surge, heave, pitch)
    and the lateral group (sway, roll, yaw) do not couple, so roughly half the
    36 pairs are zero and fitting them would only add states carrying noise.
    """

    def __init__(self, db, order="auto", rel_tol=1e-3, verbose=False):
        self.db = db
        blocks, self.pairs = [], []
        n_state = 0
        b_scale = np.max(np.abs(db.B))

        for i in range(6):
            for j in range(6):
                Bij = db.B[:, i, j]
                if np.max(np.abs(Bij)) < rel_tol * b_scale:
                    continue
                f = fit_coefficient(db.omega, db.A[:, i, j], Bij, order=order)
                blocks.append((i, j, f))
                self.pairs.append((i, j, len(f["Ar"]), f["err_K"]))
                n_state += len(f["Ar"])

        self.n = n_state
        self.A = np.zeros((n_state, n_state))
        self.B = np.zeros((n_state, 6))
        self.C = np.zeros((6, n_state))
        self.A_inf = np.zeros((6, 6))
        k = 0
        for i, j, f in blocks:
            m = len(f["Ar"])
            self.A[k:k + m, k:k + m] = f["Ar"]
            self.B[k:k + m, j] = f["Br"]
            self.C[i, k:k + m] = f["Cr"]
            self.A_inf[i, j] = f["A_inf"]
            k += m

        # pairs with no radiation still have an infinite-frequency added mass;
        # take it from the high-frequency plateau of A(w)
        for i in range(6):
            for j in range(6):
                if self.A_inf[i, j] == 0.0:
                    self.A_inf[i, j] = db.A[-1, i, j]

        if verbose:
            print(f"  radiation memory: {len(blocks)} fitted pairs, "
                  f"{n_state} states")
            worst = max(self.pairs, key=lambda p: p[3])
            print(f"  worst K fit: {DOF_NAMES[worst[0]]}-{DOF_NAMES[worst[1]]}"
                  f" order {worst[2]}, err {100*worst[3]:.2f}%")

    def force(self, x):
        return self.C @ x

    def deriv(self, x, nu):
        return self.A @ x + self.B @ nu


class LinearVessel:
    """Freely floating vessel, linear, zero forward speed."""

    def __init__(self, db, order="auto", verbose=False):
        self.db = db
        self.rad = RadiationMemory(db, order=order, verbose=verbose)
        self.M = db.M                      # rigid-body inertia
        self.C_hs = db.C                   # hydrostatic restoring
        self.Mtot = self.M + self.rad.A_inf
        self.Minv = np.linalg.inv(self.Mtot)
        self.n_state = 12 + self.rad.n

    def split(self, s):
        return s[:6], s[6:12], s[12:]      # eta, nu, radiation states

    def deriv(self, s, tau):
        eta, nu, xr = self.split(s)
        mu = self.rad.force(xr)
        nu_dot = self.Minv @ (tau - self.C_hs @ eta - mu)
        return np.concatenate([nu, nu_dot, self.rad.deriv(xr, nu)])

    def step_rk4(self, s, t, dt, force_fn):
        k1 = self.deriv(s, force_fn(t))
        k2 = self.deriv(s + 0.5 * dt * k1, force_fn(t + 0.5 * dt))
        k3 = self.deriv(s + 0.5 * dt * k2, force_fn(t + 0.5 * dt))
        k4 = self.deriv(s + dt * k3, force_fn(t + dt))
        return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(self, force_fn, t_end, dt=0.01, s0=None):
        s = np.zeros(self.n_state) if s0 is None else np.array(s0, float)
        n = int(round(t_end / dt))
        t = np.arange(n + 1) * dt
        out = np.empty((n + 1, self.n_state))
        out[0] = s
        for k in range(n):
            s = self.step_rk4(s, t[k], dt, force_fn)
            out[k + 1] = s
        return t, out


# ------------------------------------------------------- frequency reference
def rao_frequency_domain(db, omega, direction_index=0, M=None):
    """Exact linear RAO: solve [-w^2(M+A) + i w B + C] xi = F_exc."""
    M = db.M if M is None else M
    k = int(np.argmin(np.abs(db.omega - omega)))
    w = db.omega[k]
    Z = (-w ** 2 * (M + db.A[k]) + 1j * w * db.B[k] + db.C)
    return np.linalg.solve(Z, db.F_exc[k, direction_index, :]), w


def excitation_series(db, omega, direction_index=0, amplitude=1.0,
                      sign=-1.0):
    """Regular-wave excitation force as a function of time.

    `sign` selects the time convention: -1 gives Re[F e^{-iwt}], +1 gives
    Re[F e^{+iwt}]. Which one matches the solver is determined empirically in
    the M3 verification rather than assumed -- getting it wrong flips every
    phase while leaving every amplitude untouched, so only a phase check
    can catch it.
    """
    k = int(np.argmin(np.abs(db.omega - omega)))
    F = db.F_exc[k, direction_index, :] * amplitude

    def f(t):
        return np.real(F * np.exp(sign * 1j * db.omega[k] * t))

    return f, db.omega[k]


def extract_amplitude_phase(t, y, omega, n_last_periods=6):
    """Least-squares fit of a cos/sin pair over the final cycles."""
    T = 2 * np.pi / omega
    mask = t >= (t[-1] - n_last_periods * T)
    tt, yy = t[mask], y[mask]
    Aa = np.column_stack([np.cos(omega * tt), np.sin(omega * tt),
                          np.ones_like(tt)])
    c, *_ = np.linalg.lstsq(Aa, yy, rcond=None)
    return np.hypot(c[0], c[1]), np.arctan2(-c[1], c[0])
