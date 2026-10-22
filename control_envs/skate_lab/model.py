"""Batched re-implementation of the toy's physics, linearisation tools and the
wave-frame (surf-riding) surge model.

Everything here is derived from ``skate_waves.SkateWavesEnv._substep``; the
equations are copied term by term (contacts, rolling resistance, drag, phase
advance, semi-implicit Euler) and evaluated for K rollouts at once with numpy.
``verify_against_env`` checks the two implementations step by step.

Coordinates
-----------
Full state  s = [x, y, th, vx, vy, om]  plus the wave phase ``phase``.
The wave is  h(x, t) = A sin(k x + phase),  phase += k c dt, so the board's
position on the wave is the *wave-frame coordinate*

    xi = k x + phase            (radians; crest at pi/2, trough at -pi/2)
    d xi / dt = k (vx + c)      (c = wave speed toward the board)

Riding the wave means vx = -c (moving with the wave) and xi constant.
The wave-frame state used for objective 3 is  z = [y, th, vx, vy, om, xi].
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .envpath import SkateConfig, SkateWavesEnv


# ----------------------------------------------------------------------------
# Batched physics
# ----------------------------------------------------------------------------
class SkateBatch:
    """K copies of the skateboard integrated in lock step (numpy, float64).

    Equations are identical to ``SkateWavesEnv._substep``; the per-rollout
    phase is kept as an array so rollouts may sit at different wave phases.
    """

    def __init__(self, cfg: SkateConfig, K: int):
        self.cfg = replace(cfg)
        self.K = int(K)
        c = self.cfg
        self.k = 2.0 * math.pi / c.wave_length
        self.inertia = c.mass * (c.deck_length ** 2 + c.deck_thickness ** 2) / 12.0
        self.state = np.zeros((self.K, 6))
        self.phase = np.zeros(self.K)
        self.n_contacts = np.zeros(self.K, dtype=np.int64)
        self._bx = np.array([[c.wheelbase / 2.0], [-c.wheelbase / 2.0]])

    # -- set/get ---------------------------------------------------------------
    def set(self, state, phase) -> None:
        self.state = np.broadcast_to(np.asarray(state, float), (self.K, 6)).copy()
        self.phase = np.broadcast_to(np.asarray(phase, float), (self.K,)).copy()

    @classmethod
    def from_env(cls, env: SkateWavesEnv, K: int) -> "SkateBatch":
        env = env.unwrapped
        b = cls(env.cfg, K)
        b.set(env.state, env.phase)
        b.n_contacts = np.full(K, env.n_contacts, dtype=np.int64)
        return b

    # -- wave ------------------------------------------------------------------
    def surface(self, wx, phase):
        c = self.cfg
        arg = self.k * wx + phase
        ca = np.cos(arg)
        hs = c.wave_amplitude * np.sin(arg)
        hx = c.wave_amplitude * self.k * ca
        ht = c.wave_amplitude * self.k * c.wave_speed * ca
        return hs, hx, ht

    # -- geometry -----------------------------------------------------------------
    def wheel_geometry(self, state, phase):
        """Per-wheel quantities, arrays of shape (2, K): wheel offsets rx, ry,
        wheel velocities, surface height/slope/vertical speed under the wheel,
        slope norm and penetration (positive = wheel pressed into the surface)."""
        c = self.cfg
        x, y, th, vx, vy, om = state.T
        cth, sth = np.cos(th), np.sin(th)
        bx = self._bx                      # (2,1): +wheelbase/2 (front), -wheelbase/2 (rear)
        by = -c.truck_height
        rx = cth * bx - sth * by           # offsets rotated into the world frame, (2,K)
        ry = sth * bx + cth * by
        wx, wy = x + rx, y + ry            # wheel centres
        wvx = vx - om * ry                 # wheel centre velocities
        wvy = vy + om * rx
        hs, hx, ht = self.surface(wx, phase)
        norm = np.sqrt(1.0 + hx * hx)
        penetration = c.wheel_radius - (wy - hs) / norm
        return rx, ry, wvx, wvy, hs, hx, ht, norm, penetration, cth, sth

    # -- forces (the right-hand side) -----------------------------------------
    def forces(self, state, phase, u):
        """Total force/torque on each board: returns Fx, Fy, tau, contacts.

        Both wheels are evaluated together (arrays (2, K)); the additions are
        done in the env's order so the two codes agree to rounding.
        """
        c = self.cfg
        rx, ry, wvx, wvy, hs, hx, ht, norm, penetration, cth, sth = self.wheel_geometry(state, phase)
        vx, vy, om = state[:, 3], state[:, 4], state[:, 5]
        nx, ny = -hx / norm, 1.0 / norm    # surface normal (pointing up)
        tx, ty = 1.0 / norm, hx / norm     # surface tangent (pointing +x)
        inc = penetration > 0.0
        vrel_x, vrel_y = wvx, wvy - ht     # surface points move vertically at dh/dt
        vn = vrel_x * nx + vrel_y * ny
        vt = vrel_x * tx + vrel_y * ty
        N = c.contact_stiffness * penetration - c.contact_damping * vn
        N = np.where(inc, np.maximum(N, 0.0), 0.0)        # push, never pull; 0 if no contact
        Ft = -c.rolling_resistance * N * np.tanh(vt / 0.05)
        fx = N * nx + Ft * tx
        fy = N * ny + Ft * ty
        Fx = fx[0] + fx[1]
        Fy = (-c.mass * c.gravity + fy[0]) + fy[1]
        tau = (rx[0] * fy[0] - ry[0] * fx[0]) + (rx[1] * fy[1] - ry[1] * fx[1])
        contacts = inc[0].astype(np.int64) + inc[1]
        F = u * c.max_force
        if c.force_only_when_grounded:
            F = np.where(contacts == 0, 0.0, F)
        if c.force_mode == "deck":
            Fx = Fx + F * cth
            Fy = Fy + F * sth
        else:
            Fx = Fx + F
        Fx = Fx - c.linear_drag * vx
        Fy = Fy - c.linear_drag * vy
        tau = tau - c.angular_drag * om
        return Fx, Fy, tau, contacts

    def substep(self, u, h: float):
        c = self.cfg
        Fx, Fy, tau, contacts = self.forces(self.state, self.phase, u)
        x, y, th, vx, vy, om = self.state.T.copy()
        vx = vx + Fx / c.mass * h
        vy = vy + Fy / c.mass * h
        om = om + tau / self.inertia * h
        x = x + vx * h
        y = y + vy * h
        th = th + om * h
        self.state = np.stack([x, y, th, vx, vy, om], axis=1)
        self.phase = self.phase + self.k * c.wave_speed * h
        self.n_contacts = contacts
        return contacts

    def step(self, u):
        """One control step (``substeps`` substeps) for all K boards.

        Returns a dict with per-board diagnostics for this step:
          contacts  wheels touching at the end of the step (env's info value)
          airborne  True if no wheel touched at the end of the step
          impact    closing speed (m/s) at a touchdown inside this step
                    (surface vertical velocity minus board vy, just before the
                    first wheel touched), 0 if no touchdown happened
          crashed   env's crash test evaluated at the end of the step
        """
        c = self.cfg
        u = np.clip(np.broadcast_to(np.asarray(u, float), (self.K,)), -1.0, 1.0)
        h = c.dt / c.substeps
        prev = self.n_contacts.copy()
        impact = np.zeros(self.K)
        for _ in range(c.substeps):
            vy_before = self.state[:, 4].copy()
            _, _, ht_before = self.surface(self.state[:, 0], self.phase)
            contacts = self.substep(u, h)
            touchdown = (prev == 0) & (contacts > 0)
            if touchdown.any():
                closing = np.maximum(ht_before - vy_before, 0.0)
                impact = np.where(touchdown, np.maximum(impact, closing), impact)
            prev = contacts
        return {
            "contacts": contacts,
            "airborne": contacts == 0,
            "impact": impact,
            "crashed": self.crashed(),
        }

    def crashed(self):
        c = self.cfg
        x, y, th = self.state[:, 0], self.state[:, 1], self.state[:, 2]
        too_tilted = np.cos(th) < math.cos(math.radians(c.crash_tilt_deg))
        below = y < c.wave_amplitude * np.sin(self.k * x + self.phase)
        return too_tilted | below

    def rollout(self, U):
        """Roll K boards along K action sequences U (K, H) from the current state.

        Returns dict of arrays over the horizon: state (K,H,6), phase (K,H),
        airborne (K,H), impact (K,H), crashed (K,H, cumulative) and the
        progress dx (K,H).
        """
        U = np.asarray(U, float)
        K, H = U.shape
        assert K == self.K
        out_state = np.empty((K, H, 6))
        out_phase = np.empty((K, H))
        airborne = np.empty((K, H), dtype=bool)
        impact = np.empty((K, H))
        crashed = np.empty((K, H), dtype=bool)
        dx = np.empty((K, H))
        dead = np.zeros(K, dtype=bool)
        for t in range(H):
            x0 = self.state[:, 0].copy()
            d = self.step(U[:, t])
            dead |= d["crashed"]
            out_state[:, t] = self.state
            out_phase[:, t] = self.phase
            airborne[:, t] = d["airborne"]
            impact[:, t] = d["impact"]
            crashed[:, t] = dead
            dx[:, t] = self.state[:, 0] - x0
        return {"state": out_state, "phase": out_phase, "airborne": airborne,
                "impact": impact, "crashed": crashed, "dx": dx, "u": U}

    def obs(self):
        """The env's 9-float observation for every board (float64)."""
        x, y, th, vx, vy, om = self.state.T
        hs, hx, _ = self.surface(x, self.phase)
        arg = self.k * x + self.phase
        return np.stack([x, y - hs, th, vx, vy, om, hx, np.sin(arg), np.cos(arg)], axis=1)


# ----------------------------------------------------------------------------
# Verification against the env
# ----------------------------------------------------------------------------
def verify_against_env(n_seq: int = 4, T: int = 300, seed: int = 0,
                       cfg: SkateConfig | None = None) -> dict:
    """Drive the env and the batched model with the same random action
    sequences and report the largest state deviation seen (per sequence)."""
    rng = np.random.default_rng(seed)
    cfg = cfg or SkateConfig()
    env = SkateWavesEnv(render_mode=None, config=cfg)
    worst = 0.0
    per_seq = []
    steps_run = []
    for s in range(n_seq):
        env.reset(seed=seed + s)
        model = SkateBatch.from_env(env, K=1)
        # smooth-ish random actions (random walk, clipped) so the board survives
        u = np.clip(np.cumsum(rng.normal(0, 0.15, T)) * 0.3 + rng.uniform(-0.5, 0.5), -1, 1)
        dev = 0.0
        n = 0
        for t in range(T):
            obs, r, term, trunc, info = env.step(np.array([u[t]]))
            d = model.step(np.array([u[t]]))
            dev = max(dev, float(np.max(np.abs(model.state[0] - env.state))),
                      abs(float(model.phase[0]) - env.phase))
            n += 1
            if int(d["contacts"][0]) != int(info["wheels_in_contact"]):
                raise AssertionError("contact count mismatch at step %d" % t)
            if bool(d["crashed"][0]) != bool(term):
                raise AssertionError("crash flag mismatch at step %d" % t)
            if term or trunc:
                break
        per_seq.append(dev)
        steps_run.append(n)
        worst = max(worst, dev)
    return {"max_abs_deviation": worst, "per_sequence": per_seq, "steps": steps_run}


# ----------------------------------------------------------------------------
# Numerical linearisation of the one-control-step map
# ----------------------------------------------------------------------------
def step_map(cfg: SkateConfig, S, u):
    """One control step of the full 7-vector S = [x, y, th, vx, vy, om, phase]
    for a batch S (K,7), u (K,). Returns S_next (K,7)."""
    S = np.atleast_2d(np.asarray(S, float))
    b = SkateBatch(cfg, S.shape[0])
    b.set(S[:, :6], S[:, 6])
    b.step(np.asarray(u, float))
    return np.concatenate([b.state, b.phase[:, None]], axis=1)


def linearize_step_map(cfg: SkateConfig, S0, u0: float, eps=1e-6):
    """Central differences of the one-control-step map about (S0, u0).

    Returns A (7x7), B (7,) with  S+ - f(S0,u0) ~= A (S - S0) + B (u - u0).
    eps may be a scalar or a per-state array (recommended: scale by state size).
    """
    S0 = np.asarray(S0, float).reshape(7)
    eps = np.broadcast_to(np.asarray(eps, float), (7,))
    pts = []
    for i in range(7):
        e = np.zeros(7)
        e[i] = eps[i]
        pts.append(S0 + e)
        pts.append(S0 - e)
    Sp = np.array(pts)
    U = np.full(Sp.shape[0], u0)
    nxt = step_map(cfg, Sp, U)
    A = np.empty((7, 7))
    for i in range(7):
        A[:, i] = (nxt[2 * i] - nxt[2 * i + 1]) / (2 * eps[i])
    du = 1e-4
    nu = step_map(cfg, np.array([S0, S0]), np.array([u0 + du, u0 - du]))
    B = (nu[0] - nu[1]) / (2 * du)
    return A, B


# ----------------------------------------------------------------------------
# Wave-frame coordinates (objective 3)
# ----------------------------------------------------------------------------
def xi_of(x, phase, k):
    return k * x + phase


def wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def wave_frame_rhs(cfg: SkateConfig, z, u):
    """Continuous-time right-hand side in wave-frame coordinates.

    z = [y, th, vx, vy, om, xi] (batched (K,6)), u (K,).
    dz/dt = [vy, om, Fx/m, Fy/m, tau/I, k (vx + c)].
    x is set to 0 and the phase to xi (only xi = k x + phase matters).
    """
    z = np.atleast_2d(np.asarray(z, float))
    b = SkateBatch(cfg, z.shape[0])
    state = np.zeros((z.shape[0], 6))
    state[:, 1:6] = z[:, 0:5]
    Fx, Fy, tau, _ = b.forces(state, z[:, 5], np.asarray(u, float))
    c = b.cfg
    return np.stack([z[:, 3], z[:, 4], Fx / c.mass, Fy / c.mass, tau / b.inertia,
                     b.k * (z[:, 2] + c.wave_speed)], axis=1)


def wave_frame_step(cfg: SkateConfig, z, u):
    """One control step in wave-frame coordinates (batched)."""
    z = np.atleast_2d(np.asarray(z, float))
    b = SkateBatch(cfg, z.shape[0])
    state = np.zeros((z.shape[0], 6))
    state[:, 1:6] = z[:, 0:5]
    b.set(state, z[:, 5])
    b.step(np.asarray(u, float))
    return np.concatenate([b.state[:, 1:6], (b.k * b.state[:, 0] + b.phase)[:, None]], axis=1)


def surface_rest_state(cfg: SkateConfig, xi: float):
    """(y, th) of a board resting on the surface at wave-frame position xi:
    both wheels pressed in by the quasi-static compression under half the
    weight (solved from the two-wheel geometry; the board spans a sizeable
    part of the wavelength, so the slope at the centre is not enough)."""
    from scipy.optimize import least_squares
    b = SkateBatch(cfg, 1)
    p0 = cfg.mass * cfg.gravity / (2 * cfg.contact_stiffness)
    k = b.k
    d = k * cfg.wheelbase / 2
    # chord through the two wheel contact points gives the pitch guess, its
    # mid-point the height guess
    h_f = cfg.wave_amplitude * math.sin(xi + d)
    h_r = cfg.wave_amplitude * math.sin(xi - d)
    th_guess = math.atan((h_f - h_r) / cfg.wheelbase)
    y_guess = 0.5 * (h_f + h_r) + (cfg.truck_height + cfg.wheel_radius) / math.cos(th_guess) - p0

    def resid(p):
        st = np.array([[0.0, p[0], p[1], 0.0, 0.0, 0.0]])
        pen = b.wheel_geometry(st, np.array([xi]))[8][:, 0]
        return pen - p0

    sol = least_squares(resid, [y_guess, th_guess], xtol=1e-14, ftol=1e-14, gtol=1e-14)
    if np.max(np.abs(sol.fun)) > 1e-9:
        raise RuntimeError(f"rest state not found at xi={xi:.3f}: residual {sol.fun}")
    return float(sol.x[0]), float(sol.x[1])


def wave_frame_equilibrium(cfg: SkateConfig, xi_star: float):
    """Riding equilibrium at wave phase xi_star: the board moves with the wave
    (vx = -c), sits on the surface (y, th fixed, vy = om = 0) and the push u*
    balances slope gravity, rolling resistance and drag.

    Solves  rhs(z, u) = 0  for (y, th, u) with vx = -c, vy = om = 0, xi = xi*.
    Returns (z_eq (6,), u_eq).
    """
    from scipy.optimize import least_squares
    c = cfg
    y0, th0 = surface_rest_state(cfg, xi_star)

    def resid(p):
        y, th, u = p
        z = np.array([[y, th, -c.wave_speed, 0.0, 0.0, xi_star]])
        f = wave_frame_rhs(cfg, z, np.array([u]))[0]
        return [f[2], f[3], f[4]]          # vx, vy, om accelerations must vanish

    sol = least_squares(resid, [y0, th0, 0.0], x_scale=[1e-3, 1e-3, 1.0],
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
    if np.max(np.abs(sol.fun)) > 1e-7:
        raise RuntimeError(f"no equilibrium at xi={xi_star:.3f}: residual {sol.fun}")
    y, th, u = sol.x
    z_eq = np.array([y, th, -c.wave_speed, 0.0, 0.0, xi_star])
    st = np.zeros((1, 6))
    st[0, 1:6] = z_eq[:5]
    contacts = int(SkateBatch(cfg, 1).forces(st, np.array([xi_star]), np.array([u]))[3][0])
    if contacts != 2:
        raise RuntimeError(f"equilibrium at xi={xi_star:.3f} has {contacts} wheels down (want 2)")
    if abs(u) > 1.0:
        raise RuntimeError(f"equilibrium at xi={xi_star:.3f} needs u*={u:+.2f} (beyond the +-1 limit)")
    return z_eq, float(u)


def jacobian_rhs(cfg: SkateConfig, z0, u0: float, eps=None):
    """Central-difference Jacobians of the continuous-time wave-frame RHS.
    Returns Ac (6x6), Bc (6,)."""
    z0 = np.asarray(z0, float).reshape(6)
    if eps is None:
        eps = np.array([1e-6, 1e-6, 1e-5, 1e-5, 1e-5, 1e-5])
    eps = np.broadcast_to(np.asarray(eps, float), (6,))
    pts = []
    for i in range(6):
        e = np.zeros(6)
        e[i] = eps[i]
        pts.append(z0 + e)
        pts.append(z0 - e)
    Z = np.array(pts)
    f = wave_frame_rhs(cfg, Z, np.full(Z.shape[0], u0))
    Ac = np.empty((6, 6))
    for i in range(6):
        Ac[:, i] = (f[2 * i] - f[2 * i + 1]) / (2 * eps[i])
    du = 1e-4
    fu = wave_frame_rhs(cfg, np.array([z0, z0]), np.array([u0 + du, u0 - du]))
    Bc = (fu[0] - fu[1]) / (2 * du)
    return Ac, Bc


def reduced_surge_model(cfg: SkateConfig, xi_star: float):
    """Linearised wave-frame surge dynamics about the riding equilibrium.

    Slow states  s = [xi - xi*, w]  with w = vx + c  (speed relative to the
    wave); fast states [y, th, vy, om] (contact springs and pitch) are
    residualised: their derivative is set to zero and they are solved for,
    i.e. A_s = A11 - A12 A22^-1 A21,  B_s = B1 - A12 A22^-1 B2.

    Returns dict with A2 (2x2), B2 (2,), the full Ac/Bc, the equilibrium,
    the open-loop eigenvalues of A2 and of the full Ac.
    """
    z_eq, u_eq = wave_frame_equilibrium(cfg, xi_star)
    Ac, Bc = jacobian_rhs(cfg, z_eq, u_eq)
    slow = [5, 2]          # xi, vx (w = vx + c differs from vx by a constant)
    fast = [0, 1, 3, 4]    # y, th, vy, om
    A11 = Ac[np.ix_(slow, slow)]
    A12 = Ac[np.ix_(slow, fast)]
    A21 = Ac[np.ix_(fast, slow)]
    A22 = Ac[np.ix_(fast, fast)]
    B1 = Bc[slow]
    B2 = Bc[fast]
    X = np.linalg.solve(A22, A21)
    Y = np.linalg.solve(A22, B2)
    A2 = A11 - A12 @ X
    B2r = B1 - A12 @ Y
    return {"xi_star": xi_star, "z_eq": z_eq, "u_eq": u_eq, "Ac": Ac, "Bc": Bc,
            "A2": A2, "B2": B2r, "eig_slow": np.linalg.eigvals(A2),
            "eig_full": np.linalg.eigvals(Ac), "eig_fast": np.linalg.eigvals(A22)}


def quasi_static_surge(cfg: SkateConfig, xi: float):
    """Textbook one-point quasi-static riding model (README derivation only):
    the board follows the surface at its centre, the normal force balances
    weight, the wheels roll backwards relative to the surface (vt < 0).

        0 = F cos a - N sin a + mu N cos a + c_d c      (horizontal balance)
        N (cos a + mu sin a) = m g                       (vertical balance)
        tan a = A k cos xi,   F = u* Fmax

    Open-loop surge stiffness  d(Fx)/d(xi) ~= m g A k sin xi: the equilibrium
    is unstable where sin xi > 0 (above the mean level, crest side) and stable
    where sin xi < 0 (trough side). Returns (u*, eigenvalues of the 2x2 model).
    """
    c = cfg
    k = 2 * math.pi / c.wave_length
    mu, g, m = c.rolling_resistance, c.gravity, c.mass

    def fx(xi_, w, u):
        hx = c.wave_amplitude * k * math.cos(xi_)
        a = math.atan(hx)
        N = m * g / (math.cos(a) + mu * math.sin(a))
        return (u * c.max_force * math.cos(a) - N * math.sin(a) + mu * N * math.cos(a)
                - c.linear_drag * (w - c.wave_speed)) / m

    hx = c.wave_amplitude * k * math.cos(xi)
    a = math.atan(hx)
    N = m * g / (math.cos(a) + mu * math.sin(a))
    u_star = (N * math.sin(a) - mu * N * math.cos(a) - c.linear_drag * c.wave_speed) / (
        c.max_force * math.cos(a))
    e = 1e-6
    dfdxi = (fx(xi + e, 0.0, u_star) - fx(xi - e, 0.0, u_star)) / (2 * e)
    dfdw = (fx(xi, e, u_star) - fx(xi, -e, u_star)) / (2 * e)
    A2 = np.array([[0.0, k], [dfdxi, dfdw]])
    B2 = np.array([0.0, c.max_force * math.cos(a) / m])
    return u_star, np.linalg.eigvals(A2), A2, B2


# ----------------------------------------------------------------------------
# Speed-controller plant (objective 1): the surge row of the step map
# ----------------------------------------------------------------------------
def surge_step_model(cfg: SkateConfig, v_star: float, xi0: float = math.pi / 2):
    """Discrete surge model for speed control, read off the numerically
    linearised one-step map at a representative state: board resting on the
    surface at wave-frame position xi0 (crest, slope zero) moving at vx = v*.

    Returns (a, b, S0, u0, A, B): vx+ ~= a vx + b u + (periodic wave disturbance).
    """
    y0, th0 = surface_rest_state(cfg, xi0)
    S0 = np.array([0.0, y0, th0, v_star, 0.0, 0.0, xi0])
    u0 = float(np.clip((cfg.linear_drag * v_star + cfg.rolling_resistance * cfg.mass * cfg.gravity)
                       / cfg.max_force, -1, 1))
    A, B = linearize_step_map(cfg, S0, u0,
                              eps=np.array([1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5, 1e-6]))
    return float(A[3, 3]), float(B[3]), S0, u0, A, B
