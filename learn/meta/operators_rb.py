#!/usr/bin/env python3
"""
The rigid-body family of error operators (M16): learn/meta/PRIOR_DERIVATION.md
D8, in the staged first version of D8.12 (the critique's items 1-19 resolved
there). A standalone module: nothing in data2 / operators uses it yet.

What a draw is. A draw is one imagined "true boat": the low-fidelity boat's
equations of motion with random changes to the terms every rigid hull has.
The changes act as an extra acceleration on the five velocity channels
(surge, sway, yaw rate, heave rate, pitch rate), the same place the v0
operators push. The low-fidelity physics itself is never touched.

Two time resolutions.
  rigid-body part   evaluated at EVERY plant substep (0.04 s) from the state
                    at the start of the substep and the plant's own surface
                    at its 5 x 3 stations at that instant (rb_accel). Most of
                    it is state feedback, which a control-step hold would
                    delay by half a step.
  residual + noise  once per control step (step), held over the substeps,
                    exactly like operators.Operator: the M15 operator of the
                    same seed, with its readout/filter rule scaled by w_res,
                    its coloured / white noise and bursts scaled by w_noise,
                    and its events, relays, constant offset and slow drift
                    kept at their own v0 size.

The rigid-body components (D8.12; T1 mass change, T2 from the mass change and
T6 radiation memory are deferred):
  T3   dissipation: each channel's own damping times a random smooth
       function of speed (leaning to more damping), a random point damper
       acting on the local sideways velocity (sway and yaw together), and
       quadratic damping. Heave and pitch damping act on the velocity
       RELATIVE to the low-fidelity boat's own water reference; the
       reference's rate is its true time rate along the path, i.e. the
       elevation rate at stations that MOVE with the hull (the encounter
       rate, review of M16 item 1), so a hull that follows the reference
       exactly gets no push at any wavelength.
  T4   restoring: a change of the low-fidelity boat's own restoring term,
       i.e. acting on heave and pitch measured from the surface reference the
       low-fidelity model already uses (mean elevation + z0, slope angle +
       th0), a heave-pitch coupling, a bounded dependence on the excursion,
       and a speed-dependent shift of the running attitude.
  T5   jet effectiveness: the jet force as a point force at a random point
       near the nozzle, minus exactly what the low-fidelity model applies
       (with a draw-level share kc ~ U[0, 1] of the axial loss T (cos
       nozzle - 1), which the low-fidelity surge lacks); the yaw inertia is
       drawn from rigid-body plus added inertia, and the Nomoto model's yaw
       forcing per newton is replaced by x / I_r with the yaw damping
       rescaled so the identified steady turn is kept. The Nomoto part is
       always on (structural); the rest w.p. P_T5.
  T2c  speed-proportional couplings every hull has and the reduced model
       lacks, WORKLESS together with the low-fidelity Coriolis term (review
       of M16 item 2): the rigid-body surge term m v r the low-fidelity
       surge lacks (always on, structural), plus ONE shared multiplier lam
       (either sign) on the added-mass Coriolis terms of Fossen's form:
       surge lam A22 v r, sway -lam A11 U r, yaw (Munk) -lam (A22 - A11)
       U v; in the vertical plane an antisymmetric heave / pitch velocity
       coupling ~ U (sized from the low-fidelity damping) and a pitch Munk
       stiffness ~ (U^2 - U_des^2) theta (not workless: a stiffness).
  Teven couplings allowed by port-starboard symmetry: surge, heave and pitch
       get quadratic forms of sway velocity, yaw rate and nozzle angle
       (speed loss, sinkage and trim in turns); sway and yaw damping get a
       bounded dependence on heave, pitch, heave rate and speed.
  T7   environmental loads: forces reading the waves the hull meets at that
       substep. Each load reads a random combination of the five M15
       spatial patterns (unit library std, the four non-mean ones
       independent of the mean), or each station reads its own local
       elevation; a random causal filter; applied at a random point of the
       wetted hull, along a random generalised direction (pure moments
       allowed), or as a load density along the hull. Heave and pitch parts
       have zero gain at zero frequency, so a boat in very long waves still
       follows the surface as the low-fidelity boat does.

Nothing tells a channel which wave quantity to read.

Shared batched functions (critique items 5 and 14): rb_accel is THE rigid-
body step; simulate_rb (batched rows, the relabel.simulate pattern),
RBInjector + RBPlant (the Mission / data2 path) and the preview all call it.
The operator's full state is a plain dict, so a branch from a mid-episode
moment starts from snapshot(st) (test_operators_rb test 6).
"""
import copy
import json
import os

import numpy as np
from scipy.linalg import expm

from learn.meta.operators import A_REF, CLIP, G_WAVE, N_IN, _logu, clip_push
from learn.meta.ops_m15 import OperatorM15, draw_pattern, pattern_projections

CHANNELS = ("surge", "sway", "yaw", "heave", "pitch")
RED_VEL = (2, 9, 8, 4, 6)          # reduced-state u, v, r, zdot, thdot
VEL_IDX = (6, 7, 11, 8, 10)        # the same in the 14-number plant state
KN = 0.514444
G = 9.81
# the task's speed range (data2's scenario draws speed targets in it); the
# smooth functions of speed live on xi in [-1, 1] over this range
U_RANGE = (16 * KN, 30 * KN)
RB_STREAM = 31                     # the rigid-body draws' stream [seed, 31]
COMPS = ("T3", "T4", "T5", "T2c", "Teven", "T7")
ALL = COMPS + ("T8", "T9")

# ---------------------------------------------------------------------------
# Declared hyper-parameters (D8.12). Every range is an assumption; only the
# preview and the target checks can confirm them. Scales are in units of the
# low-fidelity model's own terms or of the hull (L, T, B, rs = sqrt(L/10 m)).
P_RB_OFF = 0.05                    # all rigid-body parts off
W_RES = (0.1, 1.0)                 # T8: LogU weight of the old rule
W_NOISE = (0.1, 1.0)               # T9: LogU weight of the old noise
P_T3, P_T4, P_T5, P_T2C, P_TEVEN = 0.85, 0.7, 0.7, 0.8, 0.5
DAMP_C0 = (np.log(0.6), np.log(2.4))   # damping multiplier: P(> 1) = 0.63
STIFF_C0 = (np.log(0.64), np.log(1.56))
ENV_A = (0.02, 1.0)                # T7 amplitude x a_env (per library std)
ENV_KH = (0.05, 0.5)               # T7 horizontal / vertical load scale
LAM = (-1.0, 1.5)                  # T2c shared added-mass Coriolis multiplier
KC = (0.0, 1.0)                    # T5 share of the axial loss T (cos d - 1)
N_REJECT = 30                      # redraws before the rigid-body part is off
# nozzle actuators the lateral stability rule is checked with, (lag tau,
# delay, both in seconds or both in units of rs): ideal, the target-like
# member (lofi.act_target_like: 0.1 s, 0.12 s) and the slowest of the M10
# family (lofi.ACT_FAMILY: tau_noz <= 0.5 rs, delay <= 0.3 rs). Rate limits
# are not modelled (nonlinear); review of M16 item 7
ACT_CHECK = ((0.0, 0.0, False), (0.1, 0.12, False), (0.5, 0.3, True))


def default_params():
    """The Scarab 195 reduced model's parameters (the identified cache)."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    with open(os.path.join(root, "studies", "_cache",
                           "lofi_scarab195.json")) as f:
        return json.load(f)["params"]


def light_env(p=None, dt=0.04, sub=6):
    """The pieces of relabel._env() the rigid-body code needs, built from
    the parameters alone (no Mission): the reduced model, its stations, the
    substep and the limits. Ideal actuators."""
    from control.reduced import ReducedModel
    p = dict(default_params() if p is None else p)
    return dict(red=ReducedModel(p), dt=float(dt), sub=int(sub),
                t_max=float(p["t_max"]), rud_max=float(p["rud_max"]),
                u_ref=float(p["u_design"]), L=float(p["L"]),
                x_st=np.linspace(p["x_stern"], p["x_bow"], 5),
                y_off=np.array([-0.5 * p["B"], 0.0, 0.5 * p["B"]]))


def _smooth(rng, c0, sigma, p_const=0.4):
    """Coefficients of g(xi) = c0 + c1 xi + c2 (3 xi^2 - 1) / 2 (D8.3):
    w.p. p_const speed-independent, else c_k ~ N(0, sigma^2 rho^(2k)),
    rho ~ U[0.3, 0.8]."""
    if rng.random() < p_const:
        return np.array([float(c0), 0.0, 0.0])
    rho = rng.uniform(0.3, 0.8)
    return np.array([float(c0), rng.normal(0, sigma * rho),
                     rng.normal(0, sigma * rho ** 2)])


def _g(c, xi):
    return c[0] + c[1] * xi + c[2] * (1.5 * xi * xi - 0.5)


def _unit(rng, n):
    d = rng.normal(0, 1, n)
    return d / max(np.sqrt((d * d).sum()), 1e-12)


class _Kernel:
    """A random stable causal filter (T7), continuous time:
    order 1: H = (d0 + d1 s/w) / (1 + s/w);
    order 2: H = (d0 + d1 s/w) / (1 + 2 zeta s/w + s^2/w^2).
    Two outputs of the same state: y_full = H u, and y_hp, the same filter
    without d0 (zero gain at zero frequency) for heave and pitch.
    d0, d1 may be arrays (one numerator per load point, shared poles)."""

    def __init__(self, order, w, zeta, d0, d1):
        self.order, self.w, self.zeta = int(order), float(w), float(zeta)
        self.d0, self.d1 = np.atleast_1d(d0).astype(float), \
            np.atleast_1d(d1).astype(float)
        if self.order == 1:
            self.Ac = np.array([[-w]])
            self.Bc = np.array([w])
        else:
            self.Ac = np.array([[0.0, 1.0], [-w * w, -2 * zeta * w]])
            self.Bc = np.array([0.0, w * w])
        self._disc = {}

    def gain(self, om):
        """|H(j om)| of the full output, (n_points, len(om))."""
        s = 1j * np.asarray(om)[None] / self.w
        num = self.d0[:, None] + self.d1[:, None] * s
        den = (1 + s) if self.order == 1 else (1 + 2 * self.zeta * s + s * s)
        return np.abs(num / den)

    def scale(self, k):
        self.d0, self.d1 = self.d0 * k, self.d1 * k

    def outputs(self):
        """(C_full (P, n), D_full (P,), C_hp (P, n), D_hp (P,)) with y = C x
        + D u per load point."""
        P = len(self.d0)
        if self.order == 1:
            Cf = (self.d0 - self.d1)[:, None]
            Df = self.d1.copy()
            Ch = -self.d1[:, None]
            Dh = self.d1.copy()
        else:
            Cf = np.stack([self.d0, self.d1 / self.w], 1)
            Df = np.zeros(P)
            Ch = np.stack([np.zeros(P), self.d1 / self.w], 1)
            Dh = np.zeros(P)
        return Cf, Df, Ch, Dh

    def disc(self, dt):
        """Exact zero-order-hold discretisation (Ad, Bd) at step dt."""
        key = round(float(dt), 12)
        hit = self._disc.get(key)
        if hit is None:
            n = self.order
            M = np.zeros((n + 1, n + 1))
            M[:n, :n], M[:n, n] = self.Ac, self.Bc
            E = expm(M * dt)
            hit = (E[:n, :n].copy(), E[:n, n].copy())
            self._disc[key] = hit
        return hit


class OperatorRB:
    """One draw of the rigid-body family (module docstring).

    Same constructor arguments as operators.Operator, and the same step(st,
    s_raw, noise, rng, freeze, hold_relays) -> (rule, noise), both (B, 5),
    clipped by the caller with clip_push. The rigid-body part is
    rb_accel(st, sr, thr, noz, eta, etad, dt), called at every substep by
    RBInjector / simulate_rb. `p` is the reduced model's parameter dict
    (default: the Scarab cache); `enabled` a set of component names (ALL by
    default) for per-component previews.

    NOT a drop-in for Operator everywhere (review of M16 item 3):
      - the state is nested (residual operator's under st['res']); restore
        a slow state with set_slow(st, slow) (or the module function
        set_slow(op, st, slow), which works for both families), never by
        writing st['ga'] etc.;
      - run() raises: relabelled targets need the rigid-body part, which is
        state feedback and exists only in closed loop (D8.4); use
        simulate_rb. run_residual() is the T8 / T9 part alone, for callers
        that add the rigid-body part themselves;
      - the Mission hook is RBInjector on an RBPlant (rb_plant), not
        data2.ZOHInjector. D8.12 lists the data2 edits."""

    def __init__(self, seed, lib, dt=0.24, L=5.4, relay=False, null=None,
                 p=None, u_range=U_RANGE, rb_off=None):
        self.seed, self.dt = int(seed), float(dt)
        self.p = dict(default_params() if p is None else p)
        # T8 / T9: the M15 operator of the same seed (its own streams)
        self.res = OperatorM15(self.seed, lib, dt=dt, L=L, relay=relay,
                               null=null)
        rng = np.random.default_rng([self.seed, RB_STREAM])
        self.w_res = float(_logu(rng, *W_RES))
        self.w_noise = float(_logu(rng, *W_NOISE))
        self._consts(lib, u_range)
        self.rb_off = (rng.random() < P_RB_OFF) if rb_off is None \
            else bool(rb_off)
        self.n_reject = 0
        if not self.rb_off:
            for _ in range(N_REJECT):
                self._draw(rng)
                if self.stable():
                    break
                self.n_reject += 1
            else:
                self.rb_off = True
        if self.rb_off:
            self._draw_off()
        self._draw_env(rng, lib)
        self._pack()
        self.enabled = set(ALL)
        self.style = dict(self.res.style)
        self.style.update(op_family="rb", **self._style())

    # ------------------------------------------------------------ constants
    def _consts(self, lib, u_range):
        p = self.p
        self.L, self.T, self.B = float(p["L"]), float(p["T"]), float(p["B"])
        self.rs = np.sqrt(self.L / 10.0)
        self.m = float(p["m_coriolis"])               # rigid-body mass
        self.m_v = float(p["m_sway"])                 # rigid + A22
        self.m_u = float(p["m_surge"])                # the surge channel's
        #                                               own (effective) M0
        self.x_st = np.linspace(p["x_stern"], p["x_bow"], 5)
        self.y_off = np.array([-0.5 * self.B, 0.0, 0.5 * self.B])  # stbd..port
        self.xc = self.x_st - self.x_st.mean()
        self.den = float((self.xc ** 2).sum())
        self.x_n0 = float(p["x_stern"])               # nozzle at the transom
        self.u_lo, self.u_hi = float(u_range[0]), float(u_range[1])
        self.u_mid = 0.5 * (self.u_lo + self.u_hi)
        self.u_half = 0.5 * (self.u_hi - self.u_lo)
        self.u_des = float(p["u_design"])
        self.wh2, self.wp2 = p["wn_heave"] ** 2, p["wn_pitch"] ** 2
        self.D0 = np.array([
            2 * p["k_drag"] * self.u_mid / self.m_u,
            (p["k_lin_sway"] + p["k_sway"] * 0.08 * self.u_mid) / self.m_v,
            1.0 / p["tau_r"],
            2 * p["z_heave"] * p["wn_heave"],
            2 * p["z_pitch"] * p["wn_pitch"]])
        # typical speeds per channel (quadratic damping scale)
        gl = np.sqrt(G * self.L)
        self.v_typ = np.array([0.08 * self.u_mid,
                               0.08 * self.u_mid / (0.35 * self.L),
                               0.05 * gl, 0.05 * gl / (0.25 * self.L)])
        self.k_nf_tau = p["k_nomoto_f"] / p["tau_r"]  # yaw per newton, lofi
        # the library: pattern readings and the size of the waves
        mu, sd = np.asarray(lib["mu"], float), np.asarray(lib["sd"], float)
        S = np.asarray(lib["S"], float)
        Sf = S.reshape(-1, S.shape[-1])[:, :N_IN]
        self._S, self._Sf = S[..., :N_IN], Sf
        self.w_mu, self.w_sd = mu[list(G_WAVE)], sd[list(G_WAVE)]
        self.Q = pattern_projections(Sf, sd)
        Zw = Sf[:, list(G_WAVE)]
        raw = Zw * self.w_sd + self.w_mu
        self.sd_eta = float(raw.reshape(-1, 5, 3)[:, :, 1].mean(1).std())
        # the low-fidelity boat's own hydrostatic wave forcing per library
        # std of the mean elevation: the scale of T7's loads
        self.a_env = float(self.wh2 * self.sd_eta)
        # and of its pitch forcing per library std of the slope angle: a
        # load's pitch part is capped by this scale, so a point load near
        # the bow cannot turn the heave scale into a pitch 20x the
        # low-fidelity boat's own wave pitch (D8.7 risk)
        ec = raw.reshape(-1, 5, 3)[:, :, 1]
        self.sd_alpha = float(np.arctan((ec * self.xc).sum(1)
                                        / self.den).std())
        self.a_env_p = float(self.wp2 * self.sd_alpha)
        # local readings: each point's elevation made independent of the
        # mean reading and scaled to unit library std
        q0 = self.Q[0, list(G_WAVE)]
        m = Zw @ q0
        self.q0, self.q0_mu = q0, float(m.mean())
        mc = m - m.mean()
        self.loc_mu = Zw.mean(0)
        zc = Zw - self.loc_mu
        self.loc_beta = (zc * mc[:, None]).sum(0) / max(mc @ mc, 1e-12)
        self.loc_s = np.maximum((zc - mc[:, None] * self.loc_beta).std(0),
                                1e-9)

    # ---------------------------------------------------------------- draws
    def _draw_off(self):
        """Neutral values for every rigid-body parameter."""
        z3 = np.zeros(3)
        self.rho_r = np.sqrt(self.x_n0 / (self.m_v * self.k_nf_tau))
        self.I_r = self.m_v * self.rho_r ** 2
        self.a33, self.m_w = 1.0, 2.0 * self.m
        self.rho_q = 0.22 * self.L
        self.I_q = self.m_w * self.rho_q ** 2
        self.A11, self.A22 = 0.05 * self.m, max(self.m_v - self.m,
                                                 0.05 * self.m)
        self.l_w = self.x_st[-1] - self.x_st[0]
        self.f_r = 1.0
        self.t3 = self.t4 = self.t5 = self.t2c = self.teven = False
        self.cR, self.cS = z3.copy(), [z3.copy() for _ in range(4)]
        self.pd = None
        self.kap = np.zeros(4)
        self.cKz, self.cKq, self.cks, self.cka = z3.copy(), z3.copy(), \
            z3.copy(), z3.copy()
        self.cdz, self.cdq = z3.copy(), z3.copy()
        self.nl = None
        self.ceT, self.ceS, self.c3 = z3.copy(), z3.copy(), 0.0
        self.kc = 0.0
        self.x_j, self.z_j = self.x_n0, 0.0
        # T2c: lam the shared added-mass Coriolis multiplier, zq the
        # antisymmetric heave / pitch velocity coupling, M the pitch Munk
        # stiffness
        self.c2 = {k: z3.copy() for k in ("lam", "zq", "M")}
        self.lon = []
        self.lat = None

    def _draw(self, rng):
        """One draw of every rigid-body part except the loads (T7)."""
        L, T = self.L, self.T
        self._draw_off()
        # inertias (critique items 2, 4, 17): rigid body + added, per
        # episode; the yaw radius covers both the rigid-body estimate
        # (~0.25 L) and the Nomoto initial-slope value (~0.34 L)
        self.rho_r = float(_logu(rng, 0.22, 0.37)) * L
        self.I_r = self.m_v * self.rho_r ** 2
        self.a33 = float(_logu(rng, 0.3, 2.5))        # A33 / m
        self.m_w = self.m * (1.0 + self.a33)
        self.rho_q = float(rng.uniform(0.18, 0.27)) * L
        self.I_q = self.m_w * self.rho_q ** 2
        self.A11 = self.m * float(rng.uniform(0.02, 0.1))
        self.A22 = max(self.m_v - self.m, 0.05 * self.m)
        self.l_w = float(rng.uniform(0.4, 1.0)) * (self.x_st[-1]
                                                    - self.x_st[0])
        # the Nomoto structural part (item 4): yaw forcing per newton x/I_r
        # with the damping rescaled so the steady turn stays identified
        self.f_r = float((self.x_n0 / self.I_r) / self.k_nf_tau)
        # T3 dissipation
        if rng.random() < P_T3:
            self.t3 = True
            self.cR = _smooth(rng, rng.uniform(np.log(0.8), np.log(1.25)),
                              0.15)
            self.cS = [_smooth(rng, rng.uniform(*DAMP_C0), 0.2)
                       for _ in range(4)]
            if rng.random() < 0.5:
                c = float(_logu(rng, 0.05, 1.0)) * self.m_v * self.D0[1]
                self.pd = dict(x=float(rng.uniform(self.x_st[0],
                                                   self.x_st[-1])),
                               c=_smooth(rng, np.log(c), 0.2))
            if rng.random() < 0.3:
                self.kap = _logu(rng, 0.05, 1.0, 4) * self.D0[1:] \
                    / self.v_typ * (rng.random(4) < 0.6)
        # T4 restoring
        if rng.random() < P_T4:
            self.t4 = True
            if rng.random() < 0.5:
                self.cKz = _smooth(rng, rng.uniform(*STIFF_C0), 0.1)
            if rng.random() < 0.5:
                self.cKq = _smooth(rng, rng.uniform(*STIFF_C0), 0.1)
            if rng.random() < 0.5:
                self.cks = _smooth(rng, rng.uniform(-0.3, 0.3), 0.1)
            if rng.random() < 0.3:
                self.cka = _smooth(rng, rng.uniform(-0.15, 0.15), 0.05)
            if rng.random() < 0.6:
                self.cdz = np.array([rng.uniform(-0.2, 0.2) * T,
                                     rng.normal(0, 0.13 * T), 0.0])
                self.cdq = np.array([rng.uniform(-0.3, 0.3) * T / L,
                                     rng.normal(0, 0.2 * T / L), 0.0])
            if rng.random() < 0.3:
                self.nl = dict(gz=rng.uniform(-0.5, 0.5),
                               lz=rng.uniform(0.5, 2.0) * T,
                               gq=rng.uniform(-0.5, 0.5),
                               lq=rng.uniform(0.5, 2.0) * T / L)
        # T5 jet effectiveness
        if rng.random() < P_T5:
            self.t5 = True
            self.ceT = _smooth(rng, rng.uniform(np.log(0.8), np.log(1.2)),
                               0.1)
            self.ceS = _smooth(rng, rng.uniform(np.log(0.7), np.log(1.3)),
                               0.1)
            if rng.random() < 0.3:
                self.c3 = float(rng.uniform(-1, 1)) / self.p["rud_max"] ** 2
            self.x_j = self.x_n0 + float(rng.uniform(-0.1, 0.1)) * L
            self.z_j = float(rng.uniform(-1.5 * T, 0.0))
            # how much of the axial loss T (cos d - 1) of a deflected jet
            # the true boat has (the low-fidelity surge applies the full T)
            self.kc = float(rng.uniform(*KC))
        # T2c speed-proportional couplings (items 2, 17; review of M16
        # item 2): one shared added-mass multiplier (either sign), so the
        # lateral total is workless with the low-fidelity Coriolis term for
        # any value; the vertical velocity coupling is antisymmetric
        if rng.random() < P_T2C:
            self.t2c = True
            sv = np.sqrt(self.D0[3] * self.D0[4])
            for k, (lo, hi) in (("lam", LAM), ("M", (-0.3, 0.3))):
                if rng.random() < 0.7:
                    self.c2[k] = _smooth(rng, rng.uniform(lo, hi),
                                         0.3 * (hi - lo) / 2)
            if rng.random() < 0.7:
                c0 = float(rng.choice([-1, 1]) * _logu(rng, 0.05, 0.5) * sv)
                self.c2["zq"] = _smooth(rng, c0, 0.3 * abs(c0))
        # Teven: couplings even in the lateral quantities (item 15)
        if rng.random() < P_TEVEN:
            self.teven = True
            for ch in (0, 3, 4):
                if rng.random() < 0.5:
                    q = rng.normal(0, 1, 4) * (rng.random(4) < 0.6)
                    if not q.any():
                        q[rng.integers(4)] = 1.0
                    self.lon.append(dict(
                        ch=ch, q=q / np.sqrt((q * q).sum()),
                        W=float(_logu(rng, 0.01, 0.15)) * A_REF[ch],
                        shape=_smooth(rng, 0.0, 0.2)))
            if rng.random() < 0.5:
                on = rng.random(4) < 0.5
                if not on.any():
                    on[rng.integers(4)] = True
                Gs = []
                for _ in range(4):
                    Gm = rng.normal(0, 1, (2, 2))
                    Gs.append(Gm / max(np.linalg.norm(Gm, 2), 1e-12))
                beta = rng.dirichlet(np.ones(4)) * on
                beta = beta / beta.sum() * rng.uniform(0.1, 0.5)
                self.lat = dict(beta=beta, G=np.stack(Gs))

    def _draw_env(self, rng, lib):
        """T7: the environmental loads (items 3, 13, 18)."""
        self.modes = []
        if self.rb_off or rng.random() < 0.15:
            return
        n = min(1 + rng.poisson(1.2), 5)
        om_band = np.geomspace(0.3, 13.0, 4000)
        for _ in range(n):
            u = rng.random()
            kind = "point" if u < 0.45 else ("gen" if u < 0.65 else "local")
            order = 1 if rng.random() < 0.4 else 2
            w = float(_logu(rng, 0.7, 11.0)) / self.rs
            zeta = float(_logu(rng, 0.2, 1.5))
            # horizontal loads are a fraction kh of the vertical scale (a
            # hull's wetted normals are mostly vertical); declared range
            md = dict(kind=kind, A=float(_logu(rng, *ENV_A)) * self.a_env,
                      kh=float(_logu(rng, *ENV_KH)))
            if kind == "local":
                xh = self.x_st / (0.5 * self.L)
                xh = np.repeat(xh, 3)                 # 15 points, i * 3 + j
                a, b = rng.normal(0, 1, 2), rng.normal(0, 1, 2)
                K = _Kernel(order, w, zeta, a[0] + a[1] * xh,
                            b[0] + b[1] * xh)
                K.scale(1.0 / max(K.gain(om_band).max(), 1e-9))
                Xp = np.repeat(self.x_st, 3)
                Yp = np.tile(self.y_off, 5)
                wet = 1.0 / (1.0 + np.exp((Xp - self.x_st[0] - self.l_w)
                                          / (0.05 * self.L)))
                bz, by = rng.normal(0, 1, 2), rng.normal(0, 1, 2)
                f = np.stack([np.full(15, rng.normal(0, 0.3)),
                              by[0] + by[1] * xh, bz[0] + bz[1] * xh], 1)
                wf = wet[:, None] * f
                wf = wf / max(np.sqrt((wf ** 2).sum(1)).sum(), 1e-9)
                Zp = np.full(15, -0.5 * self.T)
                md.update(K=K, X=Xp, Y=Yp, Z=Zp, f=wf)
                # caps (review of M16 item 5): the local readings have the
                # mean removed, so bow and stern readings can have opposite
                # signs; the worst case over the signs of the 15 readings
                # (each at most 1 per library std after the gain
                # normalisation) is the sum of the absolute contributions,
                # not the in-phase resultant. Vertical factor sv first,
                # then a factor sp on the whole heave / pitch path so the
                # pitch moment including the horizontal lever Z Fx stays
                # under the pitch cap
                fz = wf[:, 2]
                hz = np.abs(fz).sum() / self.m_w
                pz = np.abs(Xp * fz).sum() / self.I_q
                md["sv"] = float(1.0 / (self.m_v * max(
                    hz, pz * self.a_env / self.a_env_p, 1e-12)))
                P = np.abs(Zp * md["kh"] * wf[:, 0]
                           - Xp * md["sv"] * fz).sum()
                md["sp"] = self._pcap(P)
            else:
                q = draw_pattern(rng, self.Q, self._Sf)
                qw = q[list(G_WAVE)]
                md["q"] = qw
                md["q_mu"] = float((self._Sf[:, list(G_WAVE)] @ qw).mean())
                d = rng.normal(0, 1, 2)
                K = _Kernel(order, w, zeta, d[0], d[1])
                K.scale(1.0 / max(K.gain(om_band).max(), 1e-9))
                md["K"] = K
                if kind == "point":
                    md["pt"] = np.array([
                        rng.uniform(self.x_st[0], self.x_st[0] + self.l_w),
                        rng.uniform(-0.5 * self.B, 0.5 * self.B),
                        rng.uniform(-1.5 * self.T, 0.0)])
                    md["d"] = _unit(rng, 3)
                    md["sv"] = self._vcap(md["pt"][0])
                    d = md["d"]
                    md["sp"] = self._pcap(abs(
                        md["pt"][2] * md["kh"] * d[0]
                        - md["pt"][0] * md["sv"] * d[2]))
                    md["quad"] = None
                    if rng.random() < 0.3:
                        h = _unit(rng, 2)
                        md["quad"] = dict(beta=float(rng.uniform(-1, 1)),
                                          h=h, y2=self._lib_y2(md))
                else:
                    md["g"] = _unit(rng, 5)
            md["out"] = md["K"].outputs()
            self.modes.append(md)

    def _vcap(self, x):
        """Factor on a vertical point load at x so that per unit amplitude
        its heave part is at most a_env and its pitch part at most
        a_env_p (in accelerations per library std)."""
        h = 1.0 / self.m_w
        pq = abs(x) / self.I_q
        return float(min(1.0, (self.a_env_p / self.a_env) * h / max(pq,
                                                                  1e-12)))

    def _pcap(self, lever):
        """Factor on a load's heave / pitch path so that its pitch
        acceleration per unit filtered reading, m_v A lever / I_q, is at
        most A a_env_p / a_env (lever = |z kh d_x - x sv d_z| for a point
        load, the sum of its absolute values over the points for a local
        one)."""
        return float(min(1.0, (self.a_env_p / self.a_env) * self.I_q
                         / (self.m_v * max(lever, 1e-12))))

    def _lib_y2(self, md):
        """Mean square of a pattern mode's filtered reading on the library
        (control-step resolution): the mean removed from a quadratic load."""
        K = md["K"]
        Ad, Bd = K.disc(self.dt)
        Cf, Df, _, _ = K.outputs()
        u = self._S[..., list(G_WAVE)] @ md["q"] - md["q_mu"]  # (B, T)
        x = np.zeros((u.shape[0], K.order))
        acc = []
        for k in range(u.shape[1]):
            y = x @ Cf[0] + Df[0] * u[:, k]
            x = x @ Ad.T + u[:, k][:, None] * Bd
            if k >= 40:
                acc.append(y ** 2)
        return float(np.mean(acc))

    def _pack(self):
        """All smooth-in-speed coefficient triples in one matrix, so a
        substep evaluates them with one product (speed)."""
        named = [("R", self.cR)] + [(f"S{i}", c) for i, c in
                                    enumerate(self.cS)]
        named += [("pd", self.pd["c"] if self.pd is not None
                   else np.zeros(3))]
        named += [("Kz", self.cKz), ("Kq", self.cKq), ("ks", self.cks),
                  ("ka", self.cka), ("dz", self.cdz), ("dq", self.cdq),
                  ("eT", self.ceT), ("eS", self.ceS)]
        named += [(f"c_{k}", v) for k, v in self.c2.items()]
        named += [(f"lon{i}", t["shape"]) for i, t in enumerate(self.lon)]
        self._ix = {n: i for i, (n, _) in enumerate(named)}
        self._C = np.stack([np.asarray(c, float) for _, c in named])

    def _gv(self, xi):
        basis = np.stack([np.ones_like(xi), xi, 1.5 * xi * xi - 0.5])
        return self._C @ basis

    # ------------------------------------------------------------ stability
    def lin_vertical(self, u):
        """The heave-pitch linear system at speed u in calm water about the
        running attitude, states (e_z, w, e_th, q): the low-fidelity
        oscillators plus every linear rigid-body term."""
        xi = float(np.clip((u - self.u_mid) / self.u_half, -1, 1))
        wh, wp = np.sqrt(self.wh2), np.sqrt(self.wp2)
        Kzz = self.wh2 * np.exp(_g(self.cKz, xi))
        Kqq = self.wp2 * np.exp(_g(self.cKq, xi))
        ks, ka = _g(self.cks, xi), _g(self.cka, xi)
        kzq, kqz = (ks + ka) * wh * wp, (ks - ka) * wh * wp
        S2z = np.exp(_g(self.cS[2], xi))
        S2q = np.exp(_g(self.cS[3], xi))
        # antisymmetric in the drawn inertias (workless): I_q = m_w rho_q^2
        cz = _g(self.c2["zq"], xi)
        cq = -cz * self.m_w * self.rho_q ** 2 / self.I_q
        cM = _g(self.c2["M"], xi) * (u * u - self.u_des ** 2) \
            / self.u_des ** 2
        r = u / self.u_mid
        return np.array([
            [0.0, 1.0, 0.0, 0.0],
            [-Kzz, -S2z * self.D0[3], -kzq * self.rho_q,
             cz * r * self.rho_q],
            [0.0, 0.0, 0.0, 1.0],
            [-kqz / self.rho_q, cq * r / self.rho_q,
             -Kqq - cM * self.wp2, -S2q * self.D0[4]]])

    def lin_lateral(self, u, tau=0.0, delay=0.0):
        """Sway, yaw rate and heading at speed u in closed loop with the
        episode autopilot's law (sim/env.Episode._steer, planing, no
        integral, linear nozzle), states (v, r, psi). With tau > 0 the
        nozzle follows the autopilot's command through a first-order lag
        tau and a first-order Pade delay (review of M16 item 7): states (v,
        r, psi, nozzle, Pade state)."""
        p = self.p
        xi = float(np.clip((u - self.u_mid) / self.u_half, -1, 1))
        T = p["k_drag"] * u * u
        wn = 0.30 * (p["u_design"] / self.L) / 0.45
        Kn = p["k_nomoto_f"] * p["k_jet_side"] * max(T, 0.15 * p["t_max"])
        a_ap = wn ** 2 * p["tau_r"] / Kn
        b_ap = max(2 * wn * p["tau_r"] - 1.0, 0.0) / Kn
        kl = p["k_jet_side"] * T                   # lift per rad of nozzle
        eS = np.exp(_g(self.ceS, xi))
        S2v = np.exp(_g(self.cS[0], xi))
        S2r = np.exp(_g(self.cS[1], xi))
        lam = _g(self.c2["lam"], xi)
        A = np.zeros((3, 3))
        # nozzle command = K x = -(a psi + b r); lift = kl nozzle
        Kc = np.array([0.0, -b_ap, -a_ap])
        bcol = np.array([eS * kl / self.m_v, self.x_j * eS * kl / self.I_r,
                         0.0])                           # d accel / d nozzle
        A[0, 0] -= S2v * p["k_lin_sway"] / self.m_v
        # the low-fidelity -m u r plus T2c's -lam A11 u r (workless form)
        A[0, 1] -= (p["m_coriolis"] + lam * self.A11) * u / self.m_v
        A[1, 1] -= S2r * self.f_r / p["tau_r"]
        A[1, 0] -= lam * (self.A22 - self.A11) * u / self.I_r
        if self.pd is not None:
            c = np.exp(_g(self.pd["c"], xi))
            J = np.array([1.0, self.pd["x"]])
            A[:2, :2] -= c * np.outer(J / np.array([self.m_v, self.I_r]), J)
        if self.lat is not None:
            n4 = np.tanh((u - self.u_mid) / self.u_half)
            X = self.lat["beta"][3] * n4 * self.lat["G"][3]
            Dh = np.sqrt(np.array([S2v * p["k_lin_sway"] / self.m_v,
                                   S2r * self.f_r / p["tau_r"]]))
            Bt = Dh[:, None] * X * Dh[None]
            Mh = np.sqrt(np.array([self.m_v, self.I_r]))
            A[:2, :2] -= Bt * Mh[None] / Mh[:, None]
        A[2, 1] = 1.0
        if tau <= 0.0:
            return A + np.outer(bcol, Kc)
        # nozzle' = (y - nozzle) / tau, y the delayed command:
        # y = -c + w, w' = -(2 / d) w + (4 / d) c (first-order Pade)
        Af = np.zeros((5, 5))
        Af[:3, :3] = A
        Af[:3, 3] = bcol
        Af[3, 3] = -1.0 / tau
        if delay > 0.0:
            Af[3, :3] = -Kc / tau
            Af[3, 4] = 1.0 / tau
            Af[4, :3] = (4.0 / delay) * Kc
            Af[4, 4] = -2.0 / delay
        else:
            Af[3, :3] = Kc / tau
            Af[4, 4] = -1.0                        # unused state
        return Af

    def stable(self):
        """Reject rule (items 7, 8, 18): at five speeds over the task range
        the vertical plane must be stable with every mode's damping ratio
        >= 0.05 (no porpoising), and the lateral plane must be stable in
        closed loop with the autopilot, for an ideal nozzle and for the
        nozzle lags / delays of ACT_CHECK up to the slowest of the M10
        family (open-loop course instability is allowed)."""
        acts = [(t * (self.rs if r else 1.0), d * (self.rs if r else 1.0))
                for t, d, r in ACT_CHECK]
        for u in np.linspace(self.u_lo, self.u_hi, 5):
            ev = np.linalg.eigvals(self.lin_vertical(u))
            if ev.real.max() >= 0:
                return False
            if (-ev.real / np.maximum(np.abs(ev), 1e-12)).min() < 0.05:
                return False
            for tau, d in acts:
                if np.linalg.eigvals(self.lin_lateral(u, tau, d)).real.max() \
                        >= 0:
                    return False
        return True

    # ------------------------------------------------------ the rigid body
    def _ref(self, eta, etad):
        """The low-fidelity model's own water reference (ReducedModel.step):
        heave k_wh * mean centre-line elevation + z0, pitch k_wp * the slope
        angle + th0, and their rates. etad must be the elevation rate at
        stations MOVING with the hull (RowSeas.stations with vel, RBPlant),
        so the rates returned are the reference's true time rates along
        the path (review of M16 item 1); the fixed-point rate would be at
        the wave frequency instead of the encounter frequency."""
        p = self.p
        ec, ed = eta[:, :, 1], etad[:, :, 1]
        sl = (ec * self.xc).sum(1) / self.den
        sld = (ed * self.xc).sum(1) / self.den
        kh, kp = p.get("k_wave_heave", 1.0), p.get("k_wave_pitch", 1.0)
        sp = p["sign_pitch"]
        z_ref = kh * ec.mean(1) + p.get("z0", 0.0)
        th_ref = kp * sp * np.arctan(sl) + p.get("th0", 0.0)
        return z_ref, th_ref, kh * ed.mean(1), kp * sp * sld / (1 + sl * sl)

    def _gen(self, Q):
        """Generalised forces (B, 5) -> accelerations."""
        return Q / np.array([self.m_u, self.m_v, self.I_r, self.m_w,
                             self.I_q])

    def point_accel(self, pt, F, F_hp=None):
        """Accelerations (B, 5) of a force F (B, 3) at hull point pt (x, y,
        z): the rigid-body Jacobian Q_u = Fx, Q_v = Fy, Q_r = x Fy - y Fx,
        Q_w = Fz, Q_q = z Fx - x Fz (D8.2); F_hp, if given, is used for the
        heave and pitch parts."""
        Fh = F if F_hp is None else F_hp
        x, y, z = pt
        Q = np.stack([F[:, 0], F[:, 1], x * F[:, 1] - y * F[:, 0], Fh[:, 2],
                      z * Fh[:, 0] - x * Fh[:, 2]], 1)
        return self._gen(Q)

    def rb_accel(self, st, sr, thr, noz, eta, etad, dt, parts=False):
        """THE rigid-body step, batched over B rows of this draw.

        sr (B, 10) reduced states at the start of the substep; thr, noz the
        thrust (N) and nozzle (rad) applied over it; eta, etad (B, 5, 3) the
        surface and its rate at the 5 x 3 stations at the start of the
        substep (the plant's own; index j = 0 is STARBOARD, y_off = -B/2;
        etad is the rate along the MOVING stations, see _ref);
        dt the substep. Advances the load filters in st. Returns the
        acceleration (B, 5) in channel order, or with parts=True a dict
        component -> (B, 5)."""
        sr = np.atleast_2d(np.asarray(sr, float))
        Bn = len(sr)
        out = {c: np.zeros((Bn, 5)) for c in COMPS}
        if self.rb_off:
            return out if parts else np.zeros((Bn, 5))
        eta = np.asarray(eta, float).reshape(Bn, 5, 3)
        etad = np.asarray(etad, float).reshape(Bn, 5, 3)
        thr = np.broadcast_to(np.asarray(thr, float), (Bn,))
        noz = np.broadcast_to(np.asarray(noz, float), (Bn,))
        p = self.p
        x, y, u, z, zd, th, thd, psi, r, v = sr.T
        xi = np.clip((u - self.u_mid) / self.u_half, -1.0, 1.0)
        z_ref, th_ref, zd_ref, thd_ref = self._ref(eta, etad)
        ez, eth = z - z_ref, th - th_ref
        wr, qr = zd - zd_ref, thd - thd_ref
        en = self.enabled
        gv, ix = self._gv(xi), self._ix
        drag0 = p["k_drag"] * u * np.abs(u)
        # ---------------------------------------------------------- T3
        if self.t3 and "T3" in en:
            a = out["T3"]
            S2 = np.exp(gv[ix["S0"]:ix["S0"] + 4])
            a[:, 0] = -(np.exp(gv[ix["R"]]) - 1.0) * drag0 / self.m_u
            a[:, 1] = -(S2[0] - 1.0) * (p["k_lin_sway"] * v + p["k_sway"]
                                        * v * np.abs(v)) / self.m_v
            a[:, 2] = -(S2[1] - 1.0) * self.f_r * r / p["tau_r"]
            a[:, 3] = -(S2[2] - 1.0) * self.D0[3] * wr
            a[:, 4] = -(S2[3] - 1.0) * self.D0[4] * qr
            if self.pd is not None:
                c = np.exp(gv[ix["pd"]])
                F = -c * (v + self.pd["x"] * r)
                a[:, 1] += F / self.m_v
                a[:, 2] += self.pd["x"] * F / self.I_r
            k = self.kap
            a[:, 1] -= k[0] * np.abs(v) * v
            a[:, 2] -= k[1] * np.abs(r) * r
            a[:, 3] -= k[2] * np.abs(wr) * wr
            a[:, 4] -= k[3] * np.abs(qr) * qr
        # ---------------------------------------------------------- T4
        if self.t4 and "T4" in en:
            a = out["T4"]
            wh, wp = np.sqrt(self.wh2), np.sqrt(self.wp2)
            Kzz = self.wh2 * np.exp(gv[ix["Kz"]])
            Kqq = self.wp2 * np.exp(gv[ix["Kq"]])
            ks, ka = gv[ix["ks"]], gv[ix["ka"]]
            kzq = (ks + ka) * wh * wp * self.rho_q     # accel per rad
            kqz = (ks - ka) * wh * wp / self.rho_q     # accel per m
            fz = fq = 1.0
            if self.nl is not None:
                fz = np.exp(self.nl["gz"] * np.tanh(ez / self.nl["lz"]))
                fq = np.exp(self.nl["gq"] * np.tanh(eth / self.nl["lq"]))
            dz, dq = gv[ix["dz"]], gv[ix["dq"]]
            a[:, 3] = -((Kzz * fz - self.wh2) * ez + kzq * eth) \
                + Kzz * dz + kzq * dq
            a[:, 4] = -((Kqq * fq - self.wp2) * eth + kqz * ez) \
                + Kqq * dq + kqz * dz
        # ---------------------------------------------------------- T5
        # the Nomoto part (x_j / I_r per newton, f_r) is always on; the
        # efficiencies, the lever z_j and the axial-loss share kc only when
        # t5 (all neutral otherwise, so surge, sway and pitch are then 0)
        if "T5" in en:
            a = out["T5"]
            lift0 = p["k_jet_side"] * np.maximum(thr, 0.0) * np.sin(noz)
            Fx = np.exp(gv[ix["eT"]]) * thr * (1.0 + self.kc
                                               * (np.cos(noz) - 1.0))
            Fy = np.exp(gv[ix["eS"]]) * lift0 * (1.0 + self.c3 * noz ** 2)
            a[:, 0] = (Fx - thr) / self.m_u
            a[:, 1] = (Fy - lift0) / self.m_v
            a[:, 2] = self.x_j * Fy / self.I_r - self.k_nf_tau * lift0 \
                - (self.f_r - 1.0) * r / p["tau_r"]
            a[:, 4] = self.z_j * (Fx - drag0) / self.I_q
        # --------------------------------------------------------- T2c
        # workless with the low-fidelity Coriolis term -m u r (sway): the
        # rigid-body m v r its surge lacks (always on) plus one shared
        # multiplier lam on Fossen's added-mass Coriolis terms, converted to
        # y to port, z up: surge +A22 v r, sway -A11 u r, yaw -(A22 - A11)
        # u v. Generalised power u m v r - v m u r + lam u v r (A22 - A11 -
        # (A22 - A11)) = 0 for any lam. The Munk forms hold at small drift
        # angles, so lam is scaled by the SAME scalar saturation factor
        # s(v) = vc tanh(v / vc) / v (drift 0.1 rad; s = 1 near v = 0, so
        # the linear check is exact); a common scalar keeps the power zero
        if "T2c" in en:
            a = out["T2c"]
            c = {k: gv[ix["c_" + k]] for k in self.c2}
            vc = 0.1 * np.maximum(np.abs(u), 1.0)
            av = np.abs(v)
            sat = np.where(av > 1e-9 * vc,
                           vc * np.tanh(av / vc) / np.maximum(av, 1e-300),
                           1.0)
            lam = c["lam"] * sat
            a[:, 0] = (self.m + lam * self.A22) * v * r / self.m_u
            a[:, 1] = -lam * self.A11 * u * r / self.m_v
            a[:, 2] = -lam * (self.A22 - self.A11) * u * v / self.I_r
            # vertical: antisymmetric in the drawn inertias (workless on
            # the velocities relative to the water reference)
            ur = u / self.u_mid
            a[:, 3] = c["zq"] * ur * self.rho_q * qr
            a[:, 4] = -c["zq"] * ur * self.m_w * self.rho_q * wr / self.I_q \
                - c["M"] * ((u * u - self.u_des ** 2) / self.u_des ** 2) \
                * self.wp2 * eth
        # ------------------------------------------------------- Teven
        if self.teven and "Teven" in en:
            a = out["Teven"]
            vs = 0.1 * self.u_mid
            nv, nr = v / vs, r * self.L / vs
            F = np.stack([nv * nv, nr * nr, nv * nr,
                          (noz / p["rud_max"]) ** 2], 1)
            F = F / (1.0 + np.abs(F) / 4.0)       # bounded: at most 4
            for i, t in enumerate(self.lon):
                a[:, t["ch"]] += t["W"] * np.exp(gv[ix[f"lon{i}"]]) * (
                    F @ t["q"])
            if self.lat is not None:
                n = np.stack([np.tanh(ez / (0.5 * self.T)),
                              np.tanh(eth / (self.T / self.L)),
                              np.tanh(wr / (0.05 * np.sqrt(G * self.L))),
                              np.tanh((u - self.u_mid) / self.u_half)], 1)
                X = np.einsum("bk,kij->bij", n * self.lat["beta"],
                              self.lat["G"])
                # scaled by the CURRENT linear damping of the two channels
                # (after T3 and T5), so the total stays >= half of it
                Dh = np.sqrt(np.stack([
                    np.exp(gv[ix["S0"]]) * p["k_lin_sway"] / self.m_v,
                    np.exp(gv[ix["S1"]]) * self.f_r / p["tau_r"]], 1))
                Mh = np.sqrt(np.array([self.m_v, self.I_r]))
                nut = np.stack([v, r], 1) * Mh
                Qt = -np.einsum("bij,bj->bi", Dh[:, :, None] * X
                                * Dh[:, None, :], nut)
                a[:, 1] += Qt[:, 0] / Mh[0]
                a[:, 2] += Qt[:, 1] / Mh[1]
        # ---------------------------------------------------------- T7
        if self.modes:
            a7 = out["T7"]
            z15 = (eta.reshape(Bn, 15) - self.w_mu) / self.w_sd
            on7 = "T7" in en
            for i, md in enumerate(self.modes):
                K = md["K"]
                Ad, Bd = K.disc(dt)
                Cf, Df, Ch, Dh = md["out"]
                xk = st["kx"][i]
                if md["kind"] == "local":
                    m = z15 @ self.q0 - self.q0_mu
                    uin = (z15 - self.loc_mu - m[:, None] * self.loc_beta) \
                        / self.loc_s                       # (B, 15)
                    yf = (xk * Cf[None]).sum(-1) + Df[None] * uin
                    yh = (xk * Ch[None]).sum(-1) + Dh[None] * uin
                    st["kx"][i] = np.einsum("bpn,mn->bpm", xk, Ad) \
                        + uin[..., None] * Bd
                else:
                    uin = z15 @ md["q"] - md["q_mu"]            # (B,)
                    yf = xk @ Cf[0] + Df[0] * uin
                    yh = xk @ Ch[0] + Dh[0] * uin
                    st["kx"][i] = xk @ Ad.T + uin[:, None] * Bd
                if on7:
                    a7 += self.load_accel(md, yf, yh)
        if parts:
            return out
        return sum(out.values())

    def load_accel(self, md, yf, yh):
        """Accelerations (B, 5) of one T7 load given its filtered readings:
        yf the full output (surge, sway, yaw path), yh the output without
        zero-frequency gain (heave, pitch path); (B,) for pattern loads,
        (B, 15) for local ones. The heave / pitch path carries the caps sv
        (vertical) and sp (whole path, pitch lever incl. z Fx)."""
        s = self.m_v * md["A"]
        if md["kind"] == "local":
            sh = s * md["kh"]
            f = md["f"]
            Fx, Fy = sh * f[:, 0] * yf, sh * f[:, 1] * yf
            Fxh = sh * md["sp"] * f[:, 0] * yh
            Fzh = s * md["sv"] * md["sp"] * f[:, 2] * yh
            Q = np.stack([
                Fx.sum(1), Fy.sum(1), (md["X"] * Fy - md["Y"] * Fx).sum(1),
                Fzh.sum(1), (md["Z"] * Fxh - md["X"] * Fzh).sum(1)], 1)
            return self._gen(Q)
        if md["kind"] == "gen":
            Mg = self.masses()
            kh = md["kh"]
            y5 = np.stack([kh * yf, kh * yf, kh * yf, yh,
                           yh * self.a_env_p / self.a_env * self.rho_q], 1)
            return md["A"] * md["g"] * np.sqrt(self.m_v / Mg) * y5
        d = md["d"] * np.array([md["kh"], md["kh"], md["sv"]])
        F = s * yf[:, None] * d[None]
        Fh = s * md["sp"] * yh[:, None] * d[None]
        if md["quad"] is not None:
            qd = md["quad"]
            yq = qd["beta"] * (yf ** 2 - qd["y2"])
            F[:, :2] += s * md["kh"] * yq[:, None] * qd["h"][None]
        return self.point_accel(md["pt"], F, Fh)

    def damping_accel(self, sr, thr=0.0, noz=0.0):
        """The dissipative parts only (T3, the lateral Teven term, T5's yaw
        damping rescale) plus the low-fidelity model's own damping, in calm
        water: what the test checks for non-positive power."""
        sr = np.atleast_2d(np.asarray(sr, float))
        Bn = len(sr)
        p = self.p
        x, y, u, z, zd, th, thd, psi, r, v = sr.T
        a0 = np.stack([-p["k_drag"] * u * np.abs(u) / self.m_u,
                       -(p["k_lin_sway"] * v + p["k_sway"] * v * np.abs(v))
                       / self.m_v, -r / p["tau_r"], -self.D0[3] * zd,
                       -self.D0[4] * thd], 1)
        eta = np.zeros((Bn, 5, 3))
        keep = self.enabled
        self.enabled = {"T3", "Teven"}
        lon, self.lon = self.lon, []
        try:
            st = dict(kx=[None] * len(self.modes))
            modes, self.modes = self.modes, []
            a = self.rb_accel(st, sr, thr, noz, eta, eta, 0.04)
        finally:
            self.enabled, self.lon, self.modes = keep, lon, modes
        a[:, 2] -= (self.f_r - 1.0) * r / p["tau_r"]
        return a0 + a

    def masses(self):
        """The drawn inertias used everywhere, (m_u, m_v, I_r, m_w, I_q)."""
        return np.array([self.m_u, self.m_v, self.I_r, self.m_w, self.I_q])

    # --------------------------------------------------------- the state
    def new_state(self, B=1, rng=None, init_s=None, rb_rows=None):
        """Fresh state: the residual operator's (B rows, operators.Operator
        .new_state; rng pre-rolls its noise) and the load filters at rest
        (rb_rows rows, default B)."""
        R = B if rb_rows is None else int(rb_rows)
        kx = []
        for md in self.modes:
            n = md["K"].order
            kx.append(np.zeros((R, 15, n)) if md["kind"] == "local"
                      else np.zeros((R, n)))
        res = self.res.new_state(B, rng=rng, init_s=init_s)
        return dict(res=res, kx=kx, k=res["k"])

    @staticmethod
    def snapshot(st):
        """A deep copy of the full state (filters, residual operator,
        its slow stream): a branch from this moment starts from it."""
        return copy.deepcopy(st)

    def slow_state(self, st):
        """The residual operator's slow state (operators.Operator
        .slow_state of st['res']); the rigid-body part has none (its load
        filters fade)."""
        return self.res.slow_state(st["res"])

    def set_slow(self, st, slow):
        """Restore a slow state (one row, broadcast), as operators.Operator
        .run and relabel._warm do for the flat state, into the nested one;
        st['k'] follows the residual's regime clock."""
        _set_slow_flat(st["res"], slow)
        st["k"] = st["res"]["k"]

    def run(self, *args, **kwargs):
        """Not available: a relabelled rigid-body target needs the
        rigid-body part, which is state feedback and exists only in closed
        loop (D8.4; review of M16 item 3). Use simulate_rb, or
        run_residual for the T8 / T9 part alone."""
        raise NotImplementedError(
            "OperatorRB.run: the rigid-body part is state feedback and "
            "cannot be relabelled on recorded inputs; use "
            "operators_rb.simulate_rb (closed loop), or run_residual for "
            "the T8 / T9 part alone (PRIOR_DERIVATION.md D8.12)")

    def run_residual(self, S_raw, slow=None, noise=False, rng=None, hold=0):
        """operators.Operator.run for the control-step part only (T8 + T9
        with this draw's weights), on the nested state. WITHOUT the
        rigid-body part: a caller that relabels must add it (simulate_rb).
        Returns rule (B, T, 5) [, noise]."""
        S_raw = np.asarray(S_raw, float)
        B, T, _ = S_raw.shape
        st = self.new_state(B, init_s=S_raw[:, 0])
        if slow is not None:
            self.set_slow(st, slow)
        R = np.zeros((B, T, 5))
        N = np.zeros((B, T, 5)) if noise else None
        for k in range(T):
            r, n = self.step(st, S_raw[:, k], noise=noise, rng=rng,
                             freeze=slow is not None, hold_relays=k < hold)
            R[:, k] = r
            if noise:
                N[:, k] = n
        return (R, N) if noise else R

    # ------------------------------------------------ control-step part
    def step(self, st, s_raw, noise=True, rng=None, freeze=False,
             hold_relays=False):
        """One control step of T8 + T9 (operators.Operator.step with the
        weights of D8.12): s_raw (B, 28) -> (rule (B, 5), noise (B, 5))."""
        op = self.res
        rs = st["res"]
        z = (op._ext(np.atleast_2d(s_raw)) - op.mu) / op.sd
        B = z.shape[0]
        k = rs["k"]
        yf = op._filters(rs, z)
        yn = (yf - op.f_mu) / op.f_s if len(op.filters) else yf
        X = np.concatenate([yn, z[:, op.dirs]], 1)
        if op.k_regime is not None and k >= op.k_regime and not freeze:
            rs["regime"][:] = True
        ym = op._ro_norm(op.ro, X)
        if op.k_regime is not None and rs["regime"].any():
            ym = np.where(rs["regime"][:, None], op._ro_norm(op.ro2, X), ym)
        w8 = self.w_res if "T8" in self.enabled else 0.0
        ym = w8 * ym
        for i, rl in enumerate(op.relays):
            if rl["src"][0] == "filter":
                xx = yn[:, rl["src"][1]]
            else:
                xx = rs["sb"][:, _slow_index(rl["src"][1])]
            h, age = rs["rl"][i], rs["rl_age"][i]
            if not hold_relays:
                up = (xx > rl["hi"]) & (h < 0.5) & (age >= rl["dwell"])
                dn = (xx < rl["lo"]) & (h > 0.5) & (age >= rl["dwell"])
                h = np.where(up, 1.0, np.where(dn, 0.0, h))
                age = np.where(up | dn, 0, age + 1)
            rs["rl"][i], rs["rl_age"][i] = h, age
            if "T8" in self.enabled:
                ym = ym + h[:, None] * op._ro_norm(rl["ro"], X)
        a = np.exp(rs["ga"])[:, None]
        on8 = 1.0 if "T8" in self.enabled else 0.0
        rule = op.A * a * 2.5 * np.tanh(np.sqrt(op.pi_m) * ym / 2.5) \
            + on8 * (op.b0 + rs["bb"])
        ev_noise = np.zeros((B, 5))
        for i, ev in enumerate(op.events):
            if ev["src"][0] == "filter":
                vv = yn[:, ev["src"][1]]
            else:
                yy, rs["ex"][i] = ev["src"][2].step(rs["ex"][i],
                                                    z @ ev["src"][1])
                vv = (yy - ev["mu"]) / ev["s"]
            prev = rs["ev_prev"][i]
            hit = (np.nan_to_num(prev, nan=np.inf) < ev["thr"]) & (
                vv >= ev["thr"])
            xx = np.where(np.isnan(prev), 0.0,
                          np.abs(vv - np.nan_to_num(prev)))
            size = np.expm1(np.minimum(ev["b"] * op._exp_rank(ev, xx),
                                       30.0)) / np.expm1(ev["b"] / 2)
            size = np.where(hit, np.minimum(size, 100.0), 0.0)
            mult = np.ones(B)
            if ev["mark_s"] > 0 and noise and rng is not None and hit.any():
                s_ = ev["mark_s"]
                mult = np.where(hit, np.exp(rng.normal(-0.5 * s_ ** 2, s_,
                                                       B)), 1.0)
            rs["ev_prev"][i] = vv
            rs["er"][i], r_rule = op._resp_step(ev, rs["er"][i], size)
            rs["en"][i], r_nz = op._resp_step(ev, rs["en"][i],
                                              size * (mult - 1.0))
            dvec = ev["H"] * op.A * ev["dir"]
            rule = rule + on8 * dvec[None] * r_rule[:, None]
            ev_noise += on8 * dvec[None] * r_nz[:, None]
        nz = np.zeros((B, 5))
        if noise:
            w9 = self.w_noise if "T9" in self.enabled else 0.0
            nz = w9 * op._noise(rs, z, a, B, rng) + ev_noise
        if not freeze:
            op._slow(rs, B, rng)
        rs["k"] = k + 1
        st["k"] = rs["k"]                 # one clock: the residual's
        return rule, nz

    # ------------------------------------------------------------ style
    def _style(self):
        return dict(
            rb_off=bool(self.rb_off), n_reject=int(self.n_reject),
            w_res=self.w_res, w_noise=self.w_noise,
            rho_r=float(self.rho_r), rho_q=float(self.rho_q),
            a33=float(self.a33), A11=float(self.A11), l_w=float(self.l_w),
            f_r=float(self.f_r), kc=float(self.kc),
            lam=float(self.c2["lam"][0]), t3=self.t3, t4=self.t4, t5=self.t5,
            t2c=self.t2c, teven=self.teven,
            damp_mult=[float(np.exp(c[0])) for c in self.cS],
            point_damper=self.pd is not None, nonlinear_k=self.nl is not None,
            n_loads=len(self.modes),
            load_kinds=[m["kind"] for m in self.modes],
            load_A=[m["A"] for m in self.modes])


def _slow_index(col):
    from learn.meta.operators import SLOW
    return SLOW.index(col)


def _set_slow_flat(st, slow):
    """operators.Operator.run's slow-state restore on a flat state."""
    for key in ("ga", "bb", "regime"):
        st[key][:] = slow[key][0]
    for i, r in enumerate(slow["rl"]):
        st["rl"][i][:] = r[0]
        st["rl_age"][i][:] = slow["rl_age"][i][0]
    st["k"] = slow["k"]


def set_slow(op, st, slow):
    """Family-neutral slow-state restore: OperatorRB.set_slow for the
    rigid-body family, the flat restore of operators.Operator.run /
    relabel._warm otherwise. relabel._warm and Operator.run should call
    this when the family flag lands (D8.12; not edited here, M15 owns
    those files)."""
    if hasattr(op, "set_slow"):
        op.set_slow(st, slow)
    else:
        _set_slow_flat(st, slow)


# ------------------------------------------------------------------ plant
class _Tap:
    """Wraps a ReducedModel and keeps the arguments of its last step: the
    substep's start state, applied thrust and nozzle, station elevations."""

    def __init__(self, model):
        self.model = model
        self.last = None

    def step(self, sr, thrust, rudder, eta, x_st, dt):
        self.last = (np.array(sr, float), float(thrust), float(rudder),
                     np.array(eta, float))
        return self.model.step(sr, thrust, rudder, eta, x_st, dt)

    def __getattr__(self, name):
        return getattr(self.model, name)


def rb_plant(plant):
    """An RBPlant copy of a sim.lofi.ReducedPlant (same parameters,
    geometry, sea and step). Defined here so sim/lofi.py stays untouched
    (the running jobs hash it)."""
    return RBPlant(plant.p, plant.geom, plant.sea, plant.dt)


def _rbplant_class():
    from sim.lofi import ReducedPlant

    class RBPlant(ReducedPlant):
        """A ReducedPlant that keeps, at every step, what the rigid-body
        injector needs, all at the START of the substep (the plant's own
        zero-order hold, critique item 9): the reduced state, the applied
        thrust and nozzle, and the elevation and its rate along the MOVING
        5 x 3 stations (review of M16 item 1). The elevation and the
        fixed-point rate the plant itself uses are returned unchanged (the
        same expressions, so the plant is bit-identical); the moving rate
        reuses the same phase evaluation."""

        def __init__(self, params, geom, sea, dt):
            super().__init__(params, geom, sea, dt)
            self.model = _Tap(self.model)
            self.last_etad = None
            self._s_step = None

        def step(self, s, t, thrust_cmd, rudder_cmd, dt=None):
            self._s_step = np.asarray(s, float)
            try:
                return super().step(s, t, thrust_cmd, rudder_cmd, dt)
            finally:
                self._s_step = None

        def _surface(self, X, Y, t):
            s = self._s_step
            n = 3 * len(self.x_st)
            if s is None or len(X) < n:
                # a call from outside step (data2._stations on a twin etc.)
                return super()._surface(X, Y, t)
            self._s_step = None               # the step's first call only
            Xd, Yd = station_velocity(s[0:1], s[1:2], s[5:6], s[6:7],
                                      s[7:8], s[11:12], self.x_st,
                                      self.y_off)
            Xd, Yd = Xd.reshape(n), Yd.reshape(n)
            if not self._fast:
                e, ed = super()._surface(X, Y, t)
                # central difference of the elevation along the moving
                # stations (a sea without its components)
                h = 1e-4
                ep = self.sea.eta(X[:n] + Xd * h, Y[:n] + Yd * h, t + h)
                em = self.sea.eta(X[:n] - Xd * h, Y[:n] - Yd * h, t - h)
                edm = (np.asarray(ep) - np.asarray(em)) / (2 * h)
            else:
                ph = (self._wk * (X[:, None] * self._wc
                                  + Y[:, None] * self._ws)
                      - self._ww * t + self._wphi)
                e = (self._wa * np.cos(ph)).sum(-1)
                ed = (self._waw * np.sin(ph)).sum(-1)
                enc = self._ww - self._wk * (Xd[:, None] * self._wc
                                             + Yd[:, None] * self._ws)
                edm = (self._wa * enc * np.sin(ph[:n])).sum(-1)
            self.last_etad = np.asarray(edm, float).reshape(
                1, len(self.x_st), 3).copy()
            return e, ed

        def with_sea(self, sea, params=None, **_ignored):
            p = dict(self.p)
            if params:
                p.update(params)
            return RBPlant(p, self.geom, sea, self.dt)

    return RBPlant


_RBPLANT = []


def RBPlant(params, geom, sea, dt):   # noqa: N802  (a class factory)
    """An RBPlant (see _rbplant_class); the class is built at first use so
    importing this module does not import sim.lofi."""
    if not _RBPLANT:
        _RBPLANT.append(_rbplant_class())
    return _RBPLANT[0](params, geom, sea, dt)


class RBInjector:
    """Mission's residual hook for the rigid-body family (the ZOHInjector of
    data2 for this family). Set rule / noise every control step from
    op.step + clip_push; at every substep it adds the rigid-body part at the
    substep's start (read off an RBPlant), clips as clip_push does and kicks
    the velocities."""

    def __init__(self, op, st):
        self.op, self.st = op, st
        self.rule, self.noise = np.zeros(5), np.zeros(5)
        self.last = np.zeros(5)
        self._acc, self._n = np.zeros(5), 0

    @property
    def r(self):
        """data2.ZOHInjector's interface (review of M16 item 3). Read: the
        mean push applied over the substeps since the last set (what data2
        records as R), or the held part before any substep. Set (data2's
        inj.r = rule + noise, both already clipped): the held part; the
        applied push clip(a + r) is the same as with rule / noise set
        separately, since clip_push's two parts always sum to clip(rule +
        noise)."""
        return self._acc / self._n if self._n else self.rule + self.noise

    @r.setter
    def r(self, value):
        self.rule = np.asarray(value, float).copy()
        self.noise = np.zeros(5)
        self._acc, self._n = np.zeros(5), 0

    def __call__(self, s, dt, plant=None):
        sr, thr, noz, eta = plant.model.last
        a = self.op.rb_accel(self.st, sr, np.array([thr]), np.array([noz]),
                             eta, plant.last_etad, dt)[0]
        rc, nc = clip_push(a + self.rule, self.noise)
        push = rc + nc
        s = s.copy()
        for c, i in enumerate(VEL_IDX):
            s[i] += push[c] * dt
        plant.last_cg_acc = float(plant.last_cg_acc + push[3])
        self.last = push
        self._acc, self._n = self._acc + push, self._n + 1
        return s


# --------------------------------------------------------- batched world
class RowSeas:
    """The seas of B rows (relabel.Seas) with the elevation RATE too."""

    def __init__(self, seas, x_st, y_off):
        n = max(s.a.size for s in seas)
        B = len(seas)
        z = lambda: np.zeros((B, n))                         # noqa: E731
        self.a, self.k, self.c, self.s, self.w, self.phi = (z() for _ in
                                                            range(6))
        for b, sea in enumerate(seas):
            m = sea.a.size
            self.a[b, :m], self.k[b, :m] = sea.a, sea.k
            self.c[b, :m], self.s[b, :m] = np.cos(sea.th), np.sin(sea.th)
            self.w[b, :m], self.phi[b, :m] = sea.w, sea.phi
        self.aw = self.a * self.w
        self.x_st, self.y_off = np.asarray(x_st), np.asarray(y_off)

    def stations(self, x, y, psi, t, rows=None, vel=None):
        """(B,) position, heading, time -> (eta, etad) (B, 5, 3).

        vel = (u, v, r) (B,) each: etad is the elevation rate along the
        stations MOVING with the hull, sum a (w - k (cos th X' + sin th Y'))
        sin(phase) with X', Y' the stations' earth velocities (the encounter
        rate; what rb_accel needs, review of M16 item 1). vel None: the rate
        at fixed earth points (sum a w sin(phase))."""
        sel = slice(None) if rows is None else rows
        cp, sp = np.cos(psi)[:, None, None], np.sin(psi)[:, None, None]
        X = x[:, None, None] + self.x_st[None, :, None] * cp \
            - self.y_off[None, None, :] * sp
        Y = y[:, None, None] + self.x_st[None, :, None] * sp \
            + self.y_off[None, None, :] * cp
        B = len(x)
        X, Y = X.reshape(B, -1, 1), Y.reshape(B, -1, 1)
        k, c, s = self.k[sel][:, None], self.c[sel][:, None], \
            self.s[sel][:, None]
        ph = k * (X * c + Y * s) - self.w[sel][:, None] \
            * np.asarray(t, float).reshape(-1, 1, 1) + self.phi[sel][:, None]
        e = (self.a[sel][:, None] * np.cos(ph)).sum(-1).reshape(B, 5, 3)
        if vel is None:
            ed = (self.aw[sel][:, None] * np.sin(ph)).sum(-1).reshape(B, 5,
                                                                    3)
            return e, ed
        u, v, r = (np.asarray(q, float) for q in vel)
        Xd, Yd = station_velocity(x, y, psi, u, v, r, self.x_st, self.y_off)
        Xd, Yd = Xd.reshape(B, -1, 1), Yd.reshape(B, -1, 1)
        enc = self.w[sel][:, None] - k * (Xd * c + Yd * s)
        ed = (self.a[sel][:, None] * enc * np.sin(ph)).sum(-1).reshape(B, 5,
                                                                     3)
        return e, ed


def station_velocity(x, y, psi, u, v, r, x_st, y_off):
    """Earth velocities (X', Y') (B, n_st, 3) of the stations at body
    offsets (x_st, y_off) of hulls at heading psi moving with body
    velocities (u, v) and yaw rate r, with the reduced model's own
    kinematics (x' = u cos psi - v sin psi, y' = u sin psi + v cos psi)."""
    psi = np.asarray(psi, float)
    cp, sp = np.cos(psi)[:, None, None], np.sin(psi)[:, None, None]
    u, v, r = (np.asarray(q, float)[:, None, None] for q in (u, v, r))
    xs, ys = np.asarray(x_st)[None, :, None], np.asarray(y_off)[None, None, :]
    Xd = u * cp - v * sp - r * (xs * sp + ys * cp)
    Yd = u * sp + v * cp + r * (xs * cp - ys * sp)
    return Xd, Yd


def to14(sr, thr, noz):
    """Reduced states (B, 10) + actuator positions -> plant states (B, 14)."""
    o = np.zeros((len(sr), 14))
    o[:, [0, 1, 2, 4, 5]] = sr[:, [0, 1, 3, 5, 7]]
    o[:, [6, 7, 8, 10, 11]] = sr[:, [2, 9, 4, 6, 8]]
    o[:, 12], o[:, 13] = thr, noz
    return o


def to_reduced(xs14):
    x = np.asarray(xs14, float)
    return np.stack([x[:, 0], x[:, 1], x[:, 6], x[:, 2], x[:, 8], x[:, 4],
                     x[:, 10], x[:, 5], x[:, 11], x[:, 7]], 1)


def s28_of(env, xs14, w15, U):
    """The operator's 28 raw inputs from plant states (B, 14), the 15 wave
    inputs (B, 15) and the command fractions (B, 2): residuals.inputs_of,
    heave and pitch about the running attitude, waves, commands."""
    p = env["red"].p
    s = np.asarray(xs14, float)
    return np.column_stack([
        s[:, 6] / env["u_ref"] - 1.0, s[:, 7], s[:, 11] * env["L"]
        / env["u_ref"], np.cos(s[:, 5]), np.sin(s[:, 5]),
        s[:, 12] / env["t_max"], s[:, 13] / env["rud_max"], s[:, 8],
        s[:, 10], (s[:, 2] - p.get("z0", 0.0)) / 0.2,
        (s[:, 4] - p.get("th0", 0.0)) / 0.05, np.asarray(w15).reshape(-1, 15),
        np.asarray(U).reshape(-1, 2)])


def mid_waves(seas, sr, t, h):
    """The elevations at the step's midpoint t + h, at the pose dead-
    reckoned from the step-start state (the M15 wave timing)."""
    x, y, u, psi, r, v = sr[:, 0], sr[:, 1], sr[:, 2], sr[:, 7], sr[:, 8], \
        sr[:, 9]
    c, s = np.cos(psi), np.sin(psi)
    e, _ = seas.stations(x + (u * c - v * s) * h, y + (u * s + v * c) * h,
                         psi + r * h, t + h)
    return e.reshape(len(sr), 15)


def simulate_rb(env, xs14, U, t0, seas, groups, n_steps, noise=True,
                freeze=False, waves="mid", parts=False, act=None):
    """Closed-loop batched rollout of B rows with the rigid-body family, the
    relabel.simulate pattern: the reduced model at every substep with the
    surface at the stations, the rigid-body part at the substep start, the
    control-step part held, velocity kicks after each substep.

    xs14 (B, 14) start states; U (B, n, 2) command fractions, or a callable
    U(step, xs14_now) -> (B, 2) (closed-loop commands); t0 (B,) start times;
    groups: list of dict(op, st, rows, rng) (rows index into B; st has
    len(rows) rows; rng the noise stream, None with noise off). waves: 'mid'
    (M15 timing) or 'start' for the control-step inputs. act: None = ideal
    actuators (the command acts at once); or a dict of (B,) arrays of
    lofi.ACT_KEYS (every row act_family 1, e.g. stacked lofi.draw_act
    draws): the M10 chain via lofi.act_step, the plant's own function, with
    the delay line filled with the initial positions (as the plant does at
    its first step); the rigid-body part sees the APPLIED thrust / nozzle.
    Returns dict(XS (B, n + 1, 14), U (B, n, 2), P (B, n, 5) mean applied
    push, clip (B, n, 5) substeps clipped, and with parts=True PARTS
    {comp: (B, n, 5)} mean rigid-body push per component, RULE / NOISE
    (B, n, 5) the held control-step parts)."""
    red, dt, sub = env["red"], env["dt"], env["sub"]
    dtc = dt * sub
    lim = CLIP * A_REF
    B = len(xs14)
    sr = to_reduced(xs14)
    thr = np.asarray(xs14, float)[:, 12].copy()
    noz = np.asarray(xs14, float)[:, 13].copy()
    t0 = np.broadcast_to(np.asarray(t0, float), (B,)).copy()
    XS = np.zeros((B, n_steps + 1, 14))
    UU = np.zeros((B, n_steps, 2))
    P = np.zeros((B, n_steps, 5))
    CL = np.zeros((B, n_steps, 5))
    PT = {c: np.zeros((B, n_steps, 5)) for c in COMPS} if parts else None
    RU = np.zeros((B, n_steps, 5))
    NZ = np.zeros((B, n_steps, 5))
    XS[:, 0] = to14(sr, thr, noz)
    if act is not None:
        from sim import lofi
        tmx, rmx = env["t_max"], env["rud_max"]
        A = {k_: np.broadcast_to(np.asarray(act[k_], float), (B,)).copy()
             for k_ in lofi.ACT_KEYS}
        if not (A["act_family"] > 0.5).all():
            raise ValueError("simulate_rb: act rows must be M10 draws")
        q = lofi.act_arrays(A, tmx, rmx)
        dly = q["delay"]                                   # (B, 2) substeps
        pos0 = np.stack([thr, noz], 1)
        vel = np.zeros((B, 2))
        CMD = np.zeros((B, n_steps, 2))
        rows_, cols_ = np.arange(B)[:, None], np.arange(2)[None, :]
    for k in range(n_steps):
        t = t0 + k * dtc
        xs_now = to14(sr, thr, noz)
        Uk = np.asarray(U(k, xs_now) if callable(U) else U[:, k], float)
        UU[:, k] = Uk
        if waves == "mid":
            w15 = mid_waves(seas, sr, t, 0.5 * dtc)
        else:
            w15 = seas.stations(sr[:, 0], sr[:, 1], sr[:, 7], t)[0].reshape(
                B, 15)
        s28 = s28_of(env, xs_now, w15, Uk)
        held_r, held_n = np.zeros((B, 5)), np.zeros((B, 5))
        for g in groups:
            rows = g["rows"]
            rr, nn = g["op"].step(g["st"], s28[rows], noise=noise,
                                  rng=g.get("rng"), freeze=freeze)
            if not noise:
                nn = np.zeros_like(rr)
            held_r[rows], held_n[rows] = clip_push(rr, nn)
        RU[:, k], NZ[:, k] = held_r, held_n
        cmd_t, cmd_r = Uk[:, 0] * env["t_max"], Uk[:, 1] * env["rud_max"]
        if act is not None:
            CMD[:, k, 0], CMD[:, k, 1] = cmd_t, cmd_r
        for kk in range(sub):
            tt = t + kk * dt
            if act is None:
                thr_app, noz_app = cmd_t, cmd_r
            else:
                # the command d substeps before global substep g belongs
                # to control step floor((g - d) / sub); before the start,
                # the initial positions
                gi = k * sub + kk - dly
                ci = np.maximum(gi, 0) // sub
                c_d = np.where(gi >= 0, CMD[rows_, ci, cols_], pos0)
                pos, vel, app = lofi.act_step(q, np.stack([thr, noz], 1),
                                              vel, c_d, dt)
                thr_app, noz_app = app[:, 0], app[:, 1]
                thr, noz = pos[:, 0], pos[:, 1]
            eta, etad = seas.stations(sr[:, 0], sr[:, 1], sr[:, 7], tt,
                                      vel=(sr[:, 2], sr[:, 9], sr[:, 8]))
            push = np.zeros((B, 5))
            for g in groups:
                rows = g["rows"]
                a = g["op"].rb_accel(g["st"], sr[rows], thr_app[rows],
                                     noz_app[rows], eta[rows], etad[rows], dt,
                                     parts=parts)
                if parts:
                    for c in COMPS:
                        PT[c][rows, k] += a[c] / sub
                    a = sum(a.values())
                rc, nc = clip_push(a + held_r[rows], held_n[rows])
                push[rows] = rc + nc
                CL[rows, k] += (np.abs(a + held_r[rows] + held_n[rows])
                                > lim)
            sr = red.step(sr, thr_app, noz_app, eta, env["x_st"], dt)[0]
            sr[:, list(RED_VEL)] += push * dt
            P[:, k] += push / sub
        if act is None:
            thr, noz = cmd_t, cmd_r
        XS[:, k + 1] = to14(sr, thr, noz)
    out = dict(XS=XS, U=UU, P=P, clip=CL, RULE=RU, NOISE=NZ)
    if parts:
        out["PARTS"] = PT
    return out


class Autopilot:
    """The data2 episode scenario and controllers, vectorised over rows:
    speed targets 16-30 kn held 10-40 s and heading targets held 20-60 s
    (informative rows), OU dithers on thrust and nozzle, the speed PI and
    sim/env.Episode._steer's heading law (planing, jet). For previews and
    tests; not bit-identical to data2 (no Mission)."""

    def __init__(self, env, B, rng, informative=None, dtc_nom=0.25):
        p = env["red"].p
        self.env, self.p, self.B, self.rng = env, p, B, rng
        self.dtn = dtc_nom
        self.inf = (rng.random(B) > 0.25) if informative is None else \
            np.broadcast_to(np.asarray(informative, bool), (B,)).copy()
        self.u_tgt = rng.uniform(16, 30, B) * KN
        self.psi_tgt = np.where(self.inf, rng.uniform(-np.pi, np.pi, B), 0.0)
        self.t_u = rng.uniform(10, 40, B)
        self.t_psi = rng.uniform(20, 60, B)
        self.integ = np.zeros(B)
        self.psi_i = np.zeros(B)
        self.dith = []
        for _ in range(2):
            self.dith.append(dict(
                on=self.inf & (rng.random(B) < 0.7),
                tau=np.exp(rng.uniform(np.log(0.5), np.log(5.0), B)),
                amp=rng.uniform(0, 0.15, B), x=np.zeros(B)))
        self.t = 0.0

    def __call__(self, k, xs):
        env, p, rng, B = self.env, self.p, self.rng, self.B
        t = k * env["dt"] * env["sub"]
        new_u = self.inf & (t >= self.t_u)
        self.u_tgt = np.where(new_u, rng.uniform(16, 30, B) * KN, self.u_tgt)
        self.t_u = np.where(new_u, t + rng.uniform(10, 40, B), self.t_u)
        new_p = self.inf & (t >= self.t_psi)
        self.psi_tgt = np.where(new_p, rng.uniform(-np.pi, np.pi, B),
                                self.psi_tgt)
        self.t_psi = np.where(new_p, t + rng.uniform(20, 60, B), self.t_psi)
        e_u = self.u_tgt - xs[:, 6]
        self.integ = np.clip(self.integ + e_u * self.dtn, -20, 20)
        for d in self.dith:
            d["x"] = np.where(d["on"], d["x"] - d["x"] / d["tau"] * self.dtn
                              + d["amp"] * np.sqrt(2 * self.dtn / d["tau"])
                              * rng.normal(0, 1, B), 0.0)
        kp = p["m_surge"] / 3.0
        thrust = np.clip(p["k_drag"] * self.u_tgt ** 2 + kp * e_u
                         + 0.1 * kp * self.integ + self.dith[0]["x"]
                         * p["t_max"], 0.0, p["t_max"])
        # sim/env.Episode._steer, planing jet boat
        rs = np.sqrt(env["L"] / 10.0)
        wn = 0.30 * (p["u_design"] / env["L"]) / 0.45
        t_now = np.maximum(thrust, 0.15 * p["t_max"])
        Kn = p["k_nomoto_f"] * p["k_jet_side"] * t_now
        T = max(p["tau_r"], 0.5 * rs)
        a = wn ** 2 * T / Kn
        b = max(2 * wn * T - 1.0, 0.0) / Kn
        err = (xs[:, 5] - self.psi_tgt + np.pi) % (2 * np.pi) - np.pi
        lim = p["rud_max"]
        raw = -(a * err + b * xs[:, 11] + (0.12 / rs) * a * self.psi_i)
        ok = np.abs(raw) < lim
        self.psi_i = np.where(ok, np.clip(self.psi_i + err * self.dtn,
                                          -6 * rs, 6 * rs), self.psi_i)
        rud = np.clip(np.clip(raw, -lim, lim) + self.dith[1]["x"] * lim,
                      -lim, lim)
        return np.stack([thrust / p["t_max"], rud / lim], 1)


def start_states(env, B, rng):
    """Plant states at the running attitude at a random task speed."""
    p = env["red"].p
    xs = np.zeros((B, 14))
    xs[:, 6] = rng.uniform(16, 30, B) * KN
    xs[:, 2] = p.get("z0", 0.0)
    xs[:, 4] = p.get("th0", 0.0)
    xs[:, 12] = p["k_drag"] * xs[:, 6] ** 2
    return xs


def sea_state(sea_dict, seed):
    """A SeaState from data2.sea_for's dict (as relabel._sea)."""
    from sim.wavefield import SeaState
    s = dict(sea_dict)
    return SeaState(s.pop("hs"), s.pop("tp"), theta0=s.pop("theta0"),
                    n_freq=s.pop("n_freq"), n_dir=s.pop("n_dir"),
                    seed=seed, **s)


def sea_dict(rng):
    """data2.sea_for(rng) (training seas), copied to keep this module free
    of data2's imports."""
    hs, tp = rng.uniform(0.5, 1.5), rng.uniform(4.0, 7.0)
    s, long = rng.uniform(2.0, 10.0), rng.random() < 0.25
    return dict(hs=float(hs), tp=float(tp),
                theta0=float(rng.uniform(0, 2 * np.pi)), n_freq=48,
                n_dir=1 if long else 8, jitter=True, spread_s=float(s))
