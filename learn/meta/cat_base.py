#!/usr/bin/env python3
"""
The force-catalogue error prior (learn/meta/PRIOR_D10_DRAFT.md, "D10"): the
framework every catalogue item plugs into. A standalone module: nothing in
the running pipelines (meta4d / meta5 / meta6) imports it; it imports
operators_rb / operators_gen read-only and subclasses them here.

What a draw is. One imagined "true boat" = the low-fidelity boat plus an
extra acceleration on its five velocity channels (surge u, sway v, yaw rate
r, heave rate zdot, pitch rate thdot) and, new here, a separate IMPULSE
channel (velocity jumps that are not clipped per substep; draft section 6
items 3 and 6). The low-fidelity physics is never touched. Three layers
(draft section 1), evaluated at every 0.04 s substep in this order:

  0. shared quantities (substep_quantities): station and keel immersion,
     relative normal velocity along the MOVING stations, wave orbital
     velocity from the low-fidelity sea's own components, intake immersion,
     dynamic pressure, trim relative to a hidden true running trim, the
     low-fidelity boat's own acceleration (heave / pitch analytic);
  1. environment items (stage 'env': E1 current, E4 wave packets, A1 / A2
     wind vector) may change the water the other terms see; then hidden
     degrees of freedom (stage 'hidden': H1 roll) may change the station
     immersion (+- y phi); the shared quantities are re-derived after each;
  2. layer 1: the D8.12 rigid-body family (OperatorRBL1 = operators_rb
     .OperatorRB with the stability rule replaced by divergence-only
     rejection) with the overlap switches of draft section 1 applied, read
     at the hydrodynamic (water-relative) velocities;
  3. layer 2: the catalogue items of stage 'force' (W, H2 / H3, A, B, E2 /
     E3), then the propulsion chain (stage 'prop', fixed order PROP_ORDER),
     then the contact gate (stage 'gate', W4), which needs the vertical
     hydrodynamic sums of the low-fidelity boat, layer 1 and the gated
     catalogue items;
  4. layer 3: the D9 general residual (OperatorGenL3 = operators_gen
     .OperatorGen with the declared share draw of draft section 5) plus the
     D9.1 rigid-body part: its inertia M_t maps every catalogue generalised
     force to accelerations and its exact Coriolis difference is always on.

Numerical rules (draft section 6 item 6) as helpers: exact exponential
updates for first-order states (exp_update), the exact 2 x 2 transition for
oscillators incl. negative stiffness / damping (osc_phi, osc_step),
linearised implicit quadratic drag (implicit_quad_drag), implicit momentum
impulses (implicit_impulse), soft saturation (soft_sat), lever-arm mapping
(lever_Q, to_acc).

Item interface (CatItem): draw(rng, ctx) -> params or None (switched off);
init_state(params, B, rng) -> dict; step(params, state, q, dt) -> ItemOut
(acc and / or generalised force Q via a lever arm, impulse dnu, extra
observed signals; the state dict is updated in place and returned). Item
modules are learn/meta/cat_*.py; they register with @register and add
their construction checks with @check(code) (studies/test_cat_items.py runs
them).

D10a staging (orchestrator): the network input format stays meta5's; the
observation model O1-O5 has its slot here (ObsItem, OBS_REGISTRY) but is
OFF in D10a data; the extra observed signals of draft section 11 (roll,
rpm, relative wind, speed through water, GNSS quality) are computed every
substep from their hidden states (or their "item off" defaults) and
returned in ItemOut.obs / CatDraw.substep()['obs'] for storage only.
"""
import copy
import glob
import importlib
import os

import numpy as np

from learn.meta import operators_gen as G
from learn.meta import operators_rb as R
from learn.meta.operators import A_REF, CLIP, _logu, clip_push

GRAV = 9.81
RHO_W = 1025.0
RHO_A = 1.225
KN = R.KN
DEG = np.pi / 180.0
DT_SUB = 0.04
CAT_STREAM = 71                  # every draw stream: [seed, 71, attempt, id]
SAT = 3.0                        # soft saturation: <= 3 x the nominal scale

# ---------------------------------------------------------------------------
# The catalogue (draft section 3). Stream ids are fixed by this order, so an
# item added later never shifts another item's draws.
CODES = tuple([f"W{i}" for i in range(1, 13)] + ["H1", "H2", "H3"]
              + ["A1", "A2", "A3"] + ["B1", "B2", "B3"]
              + [f"P{i}" for i in range(1, 15)] + ["E1", "E2", "E3", "E4"])
STAGES = ("env", "hidden", "force", "prop", "gate")

# item-level switch-on probabilities (draft tables 3.1-3.6 with section 11;
# all [assumption]). Where the draft gives sub-switches the item-level value
# is the probability that at least one sub-part is on; the item draws its
# sub-parts conditionally in draw_on.
P_ON = dict(
    W1=0.7, W2=0.3, W3=0.8, W4=1.0,
    W5=1 - 0.4 * 0.5,                      # (a) 0.6, (b) 0.5
    W6=0.6, W7=1 - 0.3 * 0.3,              # vertical 0.7, lateral 0.7
    W8=0.8, W9=0.7, W10=0.4, W11=0.6, W12=0.3,
    H1=0.7, H2=0.4, H3=0.15,
    A1=0.7, A2=1.0, A3=0.6,                # A2 "with A1"
    B1=0.8, B2=1 - 0.95 * 0.97 * 0.97, B3=0.3,
    P1=0.7, P2=1.0, P3=1.0, P4=0.3, P5=1.0,
    P6=1 - 0.5 * 0.4,                      # F_z part 0.5, nozzle pitch bias 0.6
    P7=1 - 0.7 * 0.7 * 0.8, P8=0.02, P9=1.0, P10=0.075, P11=0.03,
    P12=0.8 * (1 - 0.97 * 0.8 * 0.95 * 0.98),   # controller 0.8 x any rule
    P13=0.3, P14=0.5,
    E1=0.6, E2=0.1, E3=0.3, E4=0.3)
DEPENDS = dict(A2="A1", P2="P1", P13="P1")      # drawn only if the parent is on
WAVES_ONLY = ("W3", "W8")                        # "on when there are waves"
STRUCTURAL = ("W4",)                             # always on, also when sparse

# overlap switches (draft section 1, changes 1-12): which layer-1 terms a
# catalogue item turns off. Items with sub-parts override CatItem.overlaps.
ALWAYS_SWITCHES = ("stab_divergence_only",   # change 1 (OperatorRBL1)
                   "T8_off",                 # change 8: D9 residual instead
                   "T2c_rigid_off")          # D9.1 Coriolis replaces T2c m v r
OVERLAP = dict(
    W1=("T4_speed_const", "T4_nl_off",       # changes 2, 9
        "T4_coupling_off"),                  # D10.8: W1's moment carries
                                             # the linear heave-pitch slope
    W5=("T4_nl_off",),                       # change 9, when W5(a) is on
    P3=("T5_etaT_off",), P5=("T5_etaS_off",),    # change 3
    W8=("T7_quad_off",),                     # change 4
    W7=("T3_kap_lat_off", "T3_kap_vert_off"),    # changes 5, 10 (per part)
    W11=("T3_surge_low_gate",))              # change 12
# not a layer-1 switch: change 11 (old G12 observation noise off when the
# observation model is on) is a training flag, CatDraw.switches 'G12_off';
# changes 6-7 (M10 actuators, old T9 noise) keep the old terms unchanged
OBS_SWITCH = "G12_off"
T3_GATE_W = 1.0 * KN             # width of the T3-surge / W11 speed split
# layer-1 parts whose heave / pitch rows are vertical hydrodynamic forces
# (the W4 contact gate multiplies them; draft section 10 item 3)
L1_VERT = ("T2c", "T3", "T4", "T7", "Teven")

# the propulsion chain's order (draft 3.5, "推进各项的组合次序")
PROP_ORDER = ("P12", "M10", "P7", "P11",            # command -> nozzle
              "P3", "P10", "P1", "P13", "P2", "P9",  # thrust factors
              "P4",                                  # cavitation, last
              "P5", "P8", "P14", "P6")               # side force, bucket...
THRUST_FACTORS = ("P3", "P10", "P1", "P13", "P2", "P9")

# layer 3 share (draft section 5): (tier, probability)
L3_TIERS = (("small", 0.75), ("fail", 0.20), ("none", 0.05))
L3_SMALL_MEAN, L3_SMALL_MAX, L3_SMALL_KAPPA = 1.0, 3, (0.005, 0.1)
L3_FAIL_MEAN, L3_FAIL_MAX, L3_FAIL_KAPPA0 = 2.0, 8, (0.01, 1.0)
# sparse episodes (draft section 5, [assumption])
SPARSE_P, SPARSE_N = 0.15, (1, 3)
# hidden true running trim (draft section 1 table)
TAU0 = (2.0 * DEG, 6.0 * DEG)
TAU_MIN, TAU_SOFT = 0.5 * DEG, 0.5 * DEG
# stream ids inside [seed, CAT_STREAM, attempt, id]
SID_META, SID_L1, SID_L3 = 1, 3, 7
SID_ITEM, SID_ITEM_STATE = 100, 500

# extra observed signals (draft section 11), stored, not network inputs yet
OBS_EXTRA = ("roll", "roll_rate", "rpm", "wind_speed_rel", "wind_angle_rel",
             "stw_u", "stw_v", "gnss_q")
OBS_CODES = ("O1", "O2", "O3", "O4", "O5")
OBS_ON_D10A = False


# ------------------------------------------------------------------ helpers
def soft_sat(y, scale, k=SAT):
    """k scale tanh(y / (k scale)): at most k times the nominal scale."""
    s = k * np.asarray(scale, float)
    return s * np.tanh(np.asarray(y, float) / s)


def half_clip_nominal(ctx):
    """D10.8: the soft-saturation scale (5,) of the items the clip
    diagnostic found at clipped substeps (W1, W2, W5): their own cap
    SAT x nominal is at most half the clip limit CLIP x A_REF in every
    channel, and never above ctx.a_ref. General reason: one item alone
    must not reach the clip; clipping is left to rare coincidences of
    several items. With the default a_ref the caps are (6, 6, 3, 18, 4.5)
    instead of 3 x a_ref (about 9.7, 8.1, 5.3, 39, 5.9); only the size
    changes, the forms and their zero points do not."""
    return np.minimum(np.asarray(ctx.a_ref, float),
                      CLIP * A_REF / (2.0 * SAT))


def exp_update(x, x_inf, tau, dt):
    """Exact first-order update x' = (x_inf - x) / tau over dt (x_inf held):
    x_inf + (x - x_inf) exp(-dt / tau). Stable for any dt / tau."""
    a = np.exp(-dt / np.maximum(np.asarray(tau, float), 1e-12))
    return x_inf + (x - x_inf) * a


def exp_update_asym(x, x_inf, tau_up, tau_down, dt):
    """exp_update with tau_up when rising towards x_inf, tau_down when
    falling (the P1 suction state, the rising / falling actuator...)."""
    tau = np.where(np.asarray(x_inf) > np.asarray(x), tau_up, tau_down)
    return exp_update(x, x_inf, tau, dt)


def lin_damp_acc(c, v, dt):
    """Acceleration of an extra linear damping c (1/s) on v over one
    substep: exact exponential for c >= 0 (-v (1 - e^{-c dt}) / dt, so it
    never overshoots), explicit -c v for c < 0 (less damping)."""
    c = np.asarray(c, float)
    v = np.asarray(v, float)
    return np.where(c >= 0.0, -v * (-np.expm1(-np.maximum(c, 0.0) * dt)) / dt,
                    -c * v)


def ou_step(x, tau, dt, rng, size=None):
    """Exact unit-variance Ornstein-Uhlenbeck step (coloured noise)."""
    a = np.exp(-dt / np.maximum(tau, 1e-12))
    n = rng.normal(0.0, 1.0, np.shape(x) if size is None else size)
    return a * x + np.sqrt(1.0 - a * a) * n


def poisson_hit(rate, dt, rng, B):
    """(B,) bool: an event of a Poisson process of this rate in dt."""
    return rng.random(B) < -np.expm1(-np.asarray(rate, float) * dt)


def hyst_switch(state, x, up, down, tau, dt, key="h", skey="s"):
    """The two-threshold switch of draft section 2: the discrete state
    state[key] flips to 1 when x rises above `up`, to 0 when it falls below
    `down` (down < up); the smoothed value state[skey] follows it with an
    exact first-order lag tau. Returns the smoothed value."""
    h = np.where(x > up, 1.0, np.where(x < down, 0.0, state[key]))
    state[key] = h
    state[skey] = exp_update(state[skey], h, tau, dt)
    return state[skey]


_PHI_CACHE = {}


def osc_phi(k, c, dt):
    """Exact transition (..., 2, 2) of e'' = -k e - c e' over dt, for any
    real k (stiffness, omega^2; negative = unstable / bistable) and c
    (damping 2 zeta omega; negative = energy injection), elementwise over
    arrays: exp(A dt) = e^{-c dt / 2} [cosh(s dt) I + sinh(s dt) / s (A +
    c/2 I)], s = sqrt(c^2 / 4 - k) (complex allowed). Equal to
    control.reduced.ReducedModel._phi for k = wn^2, c = 2 zeta wn (checked in
    studies/test_cat_items.py)."""
    k = np.asarray(k, float)
    c = np.asarray(c, float)
    scalar = k.ndim == 0 and c.ndim == 0
    if scalar:
        key = (round(float(k), 12), round(float(c), 12), round(float(dt), 12))
        hit = _PHI_CACHE.get(key)
        if hit is not None:
            return hit
    k, c = np.broadcast_arrays(k, c)
    s = np.sqrt((0.25 * c * c - k).astype(complex))
    st = s * dt
    small = np.abs(st) < 1e-6
    s_safe = np.where(small, 1.0, s)
    shs = np.where(small, dt * (1.0 + st * st / 6.0), np.sinh(st) / s_safe)
    ch = np.cosh(st)
    e = np.exp(-0.5 * c * dt)
    P = np.empty(k.shape + (2, 2))
    P[..., 0, 0] = (e * (ch + 0.5 * c * shs)).real
    P[..., 0, 1] = (e * shs).real
    P[..., 1, 0] = (-e * k * shs).real
    P[..., 1, 1] = (e * (ch - 0.5 * c * shs)).real
    if scalar:
        _PHI_CACHE[key] = P
    return P


def osc_step(x, xd, k, c, dt, x_eq=0.0, acc=0.0):
    """One substep of a hidden oscillator: the linear part (k, c, about
    x_eq held over the substep) by the exact transition, the nonlinear and
    noise part `acc` added explicitly as a velocity kick (draft section 6
    item 6). Returns (x, xd)."""
    P = osc_phi(k, c, dt)
    e = np.asarray(x, float) - x_eq
    ne = P[..., 0, 0] * e + P[..., 0, 1] * xd
    nxd = P[..., 1, 0] * e + P[..., 1, 1] * xd + np.asarray(acc) * dt
    return x_eq + ne, nxd


def implicit_quad_drag(v, c, m_eff, dt):
    """Linearised implicit quadratic drag F = -c |v| v on a generalised
    mass m_eff: dv = -dt c |v| v / (m_eff + 2 dt c |v|) (D8.12, draft W7).
    Returns the acceleration dv / dt. |dv| < |v| / 2 and sign(dv) = -sign(v)
    for any dt, c >= 0: it never reverses v and never does positive work."""
    v = np.asarray(v, float)
    c = np.maximum(np.asarray(c, float), 0.0)
    return -c * np.abs(v) * v / (m_eff + 2.0 * dt * c * np.abs(v))


def implicit_impulse(V, m_eff, m_old, m_new):
    """Momentum-conserving update of an entry velocity V (positive = into
    the water) when the added mass at a point grows from m_old to m_new
    (draft W3, section 6 item 3): V+ = V (m_eff + m_old) / (m_eff + m_new).
    Returns (V+, J) with J = m_eff (V - V+) the impulse on the hull along
    the point's outward normal (upward for a bottom point). V+ never
    reverses; |V - V+| <= |V| when m_new >= m_old, independent of dt."""
    V = np.asarray(V, float)
    Vp = V * (m_eff + m_old) / (m_eff + m_new)
    return Vp, m_eff * (V - Vp)


def lever_Q(ctx, pt, F, th):
    """Generalised force (B, 5) of body forces F (B, 3) (x forward, y port,
    z up) at hull point pt = (x, y, z) from the reference point (scalars or
    (B,) arrays): (Fx, Fy, x Fy - y Fx, Fz + s_p theta Fx, s_p (x Fz - z
    Fx)). The heave row is the earth-vertical force (the low-fidelity heave
    coordinate), the pitch row is in the low-fidelity theta convention
    (hull point height z + s_p x theta). For s_p = -1 it equals
    operators_gen.OperatorGen._point_Q (checked in the harness)."""
    x, y, z = (np.asarray(a, float) for a in pt)
    F = np.atleast_2d(F)
    sp = ctx.sp
    Fx, Fy, Fz = F[:, 0], F[:, 1], F[:, 2]
    return np.stack([Fx, Fy, x * Fy - y * Fx, Fz + sp * th * Fx,
                     sp * (x * Fz - z * Fx)], 1)


def to_acc(ctx, Q):
    """Generalised forces (B, 5) -> accelerations with the draw's M_t."""
    return np.atleast_2d(Q) @ ctx.Minv.T


def unit_vertical(ctx, x):
    """Generalised direction (5,) of a unit upward body force at station x
    (lever_Q with F = (0, 0, 1): no yaw part, pitch s_p x)."""
    return np.array([0.0, 0.0, 0.0, 1.0, ctx.sp * float(x)])


def point_mass(ctx, g):
    """Generalised mass 1 / (g^T M_t^-1 g) along a generalised direction."""
    g = np.asarray(g, float)
    return 1.0 / float(g @ ctx.Minv @ g)


def softplus_floor(x, lo, s):
    """lo + s softplus((x - lo) / s): a smooth lower bound."""
    return lo + s * np.logaddexp(0.0, (np.asarray(x, float) - lo) / s)


# ------------------------------------------------------------------ context
class CatCtx:
    """Per-draw constants the items read: the low-fidelity parameters and
    geometry, reference scales, the shared inertia, hidden constants."""

    def __init__(self, p=None, keel=None, intake=None, v_slam=None,
                 has_waves=True):
        p = dict(R.default_params() if p is None else p)
        self.p = p
        self.L, self.T, self.B = float(p["L"]), float(p["T"]), float(p["B"])
        self.b = self.B                       # beam for Cv, q b^2 (low-fi B)
        self.rs = float(np.sqrt(self.L / 10.0))
        self.sp = float(p["sign_pitch"])
        self.z0, self.th0 = float(p.get("z0", 0.0)), float(p.get("th0", 0.0))
        self.x_st = np.linspace(p["x_stern"], p["x_bow"], 5)
        self.y_off = np.array([-0.5 * self.B, 0.0, 0.5 * self.B])  # stbd..port
        self.xc = self.x_st - self.x_st.mean()
        self.den = float((self.xc ** 2).sum())
        # keel height at the 5 stations in the rest frame (negative): the
        # geometry's sec.keel interpolated, or linear between -T (stern)
        # and -draft_bow (bow) (draft section 1 table)
        if keel is None:
            keel = np.interp(self.x_st, [self.x_st[0], self.x_st[-1]],
                             [-self.T, -float(p["draft_bow"])])
        self.keel5 = np.asarray(keel, float).reshape(5)
        # intake point (x, y, z) in the same frame as plant.intakes
        if intake is None:
            xi = self.x_st[0] + 0.15 * self.L
            intake = (xi, 0.0, float(np.interp(xi, self.x_st, self.keel5)))
        self.intake = np.asarray(intake, float).reshape(3)
        self.v_slam = float(p.get("v_slam", 0.093 * np.sqrt(GRAV * self.L))
                            if v_slam is None else v_slam)
        self.m = float(p["m_coriolis"])
        self.W = self.m * GRAV
        self.vol = self.m / RHO_W
        self.a_ref = G.a_ref_of(p)
        self.u_des = self.u_id = float(p["u_design"])
        self.u_max = float(p["u_max"])
        self.u_lo, self.u_hi = R.U_RANGE
        self.t_max, self.rud_max = float(p["t_max"]), float(p["rud_max"])
        self.rud_stall = float(p.get("rud_stall", self.rud_max))
        self.k_drag, self.k_js = float(p["k_drag"]), float(p["k_jet_side"])
        self.m_u, self.m_v = float(p["m_surge"]), float(p["m_sway"])
        self.wh, self.zh = float(p["wn_heave"]), float(p["z_heave"])
        self.wp, self.zp = float(p["wn_pitch"]), float(p["z_pitch"])
        self.kwh = float(p.get("k_wave_heave", 1.0))
        self.kwp = float(p.get("k_wave_pitch", 1.0))
        self.has_waves = bool(has_waves)
        self.F_b = G.freeboard(p)
        # calm running at the low-fidelity running attitude (z0, th0)
        hull = self.z0 + self.sp * self.x_st * self.th0
        self.hc_run = -hull - self.keel5                  # keel immersion
        self.hbar_run = float(np.maximum(self.hc_run, 0.0).mean())
        xi, yi, zi = self.intake
        self.h_run = float(-(self.z0 + self.sp * xi * self.th0) - zi)
        # neutral inertia (operators_rb._draw_off) until a layer-3 draw sets
        # the shared M_t
        I_r0 = abs(float(p["x_stern"])) / abs(p["k_nomoto_f"] / p["tau_r"])
        m_w = 2.0 * self.m
        self.set_inertia(np.diag([self.m_u, self.m_v, I_r0, m_w,
                                  m_w * (0.22 * self.L) ** 2]))
        # jet point and axial-loss share (layer 1 T5, if drawn)
        self.x_j, self.z_j, self.kc = float(p["x_stern"]), 0.0, 0.0
        self.tau0 = 4.0 * DEG
        self.tau_run = float(softplus_floor(self.tau0, TAU_MIN, TAU_SOFT))

    def set_inertia(self, Mt, inertia=None):
        self.Mt = np.asarray(Mt, float)
        self.Minv = np.linalg.inv(self.Mt)
        self.inertia = dict(inertia or {})

    def set_tau0(self, tau0):
        self.tau0 = float(tau0)
        self.tau_run = float(softplus_floor(self.tau0, TAU_MIN, TAU_SOFT))


def ctx_from_plant(plant, p=None, has_waves=True):
    """A CatCtx from a sim.lofi.ReducedPlant / RBPlant: its parameters, its
    keel (sec.keel interpolated to the 5 stations), its first intake and
    v_slam."""
    p = dict(plant.p if p is None else p)
    x_st = np.linspace(p["x_stern"], p["x_bow"], 5)
    sec = plant.sec
    o = np.argsort(np.asarray(sec.x, float))
    keel = np.interp(x_st, np.asarray(sec.x, float)[o],
                     np.asarray(sec.keel, float)[o])
    intake = plant.intakes[0] if getattr(plant, "intakes", None) else None
    return CatCtx(p, keel=keel, intake=intake,
                  v_slam=getattr(plant, "v_slam", None), has_waves=has_waves)


# ------------------------------------------------------------------ the sea
class CatSea:
    """The low-fidelity sea of B rows (the components of an operators_rb
    .RowSeas, or None for calm water) sampled ONCE per substep at the 5 x 3
    stations and the intake: elevation, its rate along the MOVING stations
    (the encounter rate, as RowSeas.stations(vel=...)), and the horizontal
    orbital velocity of deep-water waves, u_orb = sum a omega cos(phase)
    along each component's direction, in body axes (draft section 1 table,
    added after review)."""

    def __init__(self, rowseas, ctx):
        self.rs = rowseas
        self.ctx = ctx

    def sample(self, sr, t):
        sr = np.atleast_2d(np.asarray(sr, float))
        B = len(sr)
        ctx = self.ctx
        z = np.zeros((B, 5, 3))
        if self.rs is None:
            return dict(eta=z, etad=z.copy(), uorb=z.copy(), vorb=z.copy(),
                        eta_in=np.zeros(B))
        x, y, u, psi, r, v = sr[:, 0], sr[:, 1], sr[:, 2], sr[:, 7], \
            sr[:, 8], sr[:, 9]
        cp, sn = np.cos(psi)[:, None], np.sin(psi)[:, None]
        xs = np.concatenate([np.repeat(ctx.x_st, 3), [ctx.intake[0]]])
        ys = np.concatenate([np.tile(ctx.y_off, 5), [ctx.intake[1]]])
        X = x[:, None] + xs[None] * cp - ys[None] * sn           # (B, 16)
        Y = y[:, None] + xs[None] * sn + ys[None] * cp
        Xd = (u[:, None] * cp - v[:, None] * sn
              - r[:, None] * (xs[None] * sn + ys[None] * cp))
        Yd = (u[:, None] * sn + v[:, None] * cp
              + r[:, None] * (xs[None] * cp - ys[None] * sn))
        S = self.rs
        k, c, s = S.k[:, None], S.c[:, None], S.s[:, None]
        tt = np.broadcast_to(np.asarray(t, float), (B,))[:, None, None]
        ph = k * (X[..., None] * c + Y[..., None] * s) - S.w[:, None] * tt \
            + S.phi[:, None]
        cph, sph = np.cos(ph), np.sin(ph)
        a = S.a[:, None]
        e = (a * cph).sum(-1)                                     # (B, 16)
        enc = S.w[:, None] - k * (Xd[..., None] * c + Yd[..., None] * s)
        ed = (a * enc * sph).sum(-1)
        aw = (S.aw[:, None] * cph)
        ue, ve = (aw * c).sum(-1), (aw * s).sum(-1)               # earth
        ub = ue * cp + ve * sn
        vb = -ue * sn + ve * cp
        f = lambda q: q[:, :15].reshape(B, 5, 3)                  # noqa: E731
        return dict(eta=f(e), etad=f(ed), uorb=f(ub), vorb=f(vb),
                    eta_in=e[:, 15])


# ------------------------------------------------------- shared quantities
def lofi_accel_vertical(ctx, sr, eta):
    """The low-fidelity boat's own heave and pitch accelerations at the
    substep start, analytically (control/reduced.py ReducedModel.step:
    a_z = -wh^2 (z - eta_bar - z0) - 2 zh wh zdot, pitch the same about the
    slope target), (B, 2); no one-substep lag (draft section 1 table)."""
    sr = np.atleast_2d(sr)
    ec = np.asarray(eta, float)[:, :, 1]
    slope = (ec * ctx.xc).sum(1) / ctx.den
    eb = ctx.kwh * ec.mean(1) + ctx.z0
    al = ctx.kwp * ctx.sp * np.arctan(slope) + ctx.th0
    z, zd, th, thd = sr[:, 3], sr[:, 4], sr[:, 5], sr[:, 6]
    return np.stack([-ctx.wh ** 2 * (z - eb) - 2 * ctx.zh * ctx.wh * zd,
                     -ctx.wp ** 2 * (th - al) - 2 * ctx.zp * ctx.wp * thd], 1)


def substep_quantities(ctx, sr, thr, noz, sea, nu0=None, cmd=None,
                       dt=DT_SUB, t=None):
    """The shared per-substep quantities (draft section 1 table), from the
    START state sr (B, 10) of the substep, the applied actuators thr, noz
    (B,), a CatSea.sample dict `sea`, nu0 (B, 5) the low-fidelity boat's
    own velocity increment / dt over the substep (surge, sway, yaw used;
    heave, pitch replaced by the analytic values), cmd (B, 2) the physical
    commands (thrust N, nozzle rad) if known. Stage-'env' / 'hidden' items
    may then set q['eta_add'], q['etad_add'] (B, 5, 3), q['nu_c'] (B, 2)
    body-frame current, q['phi'], q['phid'] (B,) roll, and call
    derive_quantities(ctx, q) again."""
    sr = np.atleast_2d(np.asarray(sr, float))
    B = len(sr)
    q = dict(ctx=ctx, dt=float(dt), sr=sr, B=B,
             thr=np.broadcast_to(np.asarray(thr, float), (B,)).copy(),
             noz=np.broadcast_to(np.asarray(noz, float), (B,)).copy(),
             cmd=None if cmd is None else np.asarray(cmd, float).reshape(B, 2),
             t=t, eta0=np.asarray(sea["eta"], float),
             etad0=np.asarray(sea["etad"], float),
             uorb=np.asarray(sea["uorb"], float),
             vorb=np.asarray(sea["vorb"], float),
             eta_in0=np.asarray(sea["eta_in"], float),
             eta_add=np.zeros((B, 5, 3)), etad_add=np.zeros((B, 5, 3)),
             eta_in_add=np.zeros(B), nu_c=np.zeros((B, 2)),
             phi=np.zeros(B), phid=np.zeros(B))
    a_v = lofi_accel_vertical(ctx, sr, q["eta0"])
    nu = np.zeros((B, 5)) if nu0 is None else np.array(nu0, float).reshape(
        B, 5)
    nu[:, 3:5] = a_v
    q["nu0"], q["a_lofi_vert"] = nu, a_v
    return derive_quantities(ctx, q)


def derive_quantities(ctx, q):
    """(Re)compute the derived shared quantities of q in place."""
    sr = q["sr"]
    x, y, u, z, zd, th, thd, psi, r, v = sr.T
    sp = ctx.sp
    eta = q["eta0"] + q["eta_add"]
    etad = q["etad0"] + q["etad_add"]
    phi, phid = q["phi"], q["phid"]
    X, Y = ctx.x_st[None, :, None], ctx.y_off[None, None, :]
    hull = z[:, None, None] + sp * X * th[:, None, None] \
        + Y * phi[:, None, None]
    whull = zd[:, None, None] + sp * X * thd[:, None, None] \
        + Y * phid[:, None, None]
    q["eta"], q["etad"] = eta, etad
    q["d"] = eta - hull                           # station immersion
    q["dd"] = etad - whull                        # entry velocity (> 0 in)
    q["h"] = q["d"] - ctx.keel5[None, :, None]    # keel immersion
    q["hc"], q["ddc"] = q["h"][:, :, 1], q["dd"][:, :, 1]
    ec, edc = eta[:, :, 1], etad[:, :, 1]
    slope = (ec * ctx.xc).sum(1) / ctx.den
    q["slope"] = slope
    q["t_slope"] = ((eta[:, :, 2] - eta[:, :, 0]) / (2 * 0.5 * ctx.B)).mean(1)
    # the low-fidelity boat's own water reference and its rate along the
    # path (operators_rb.OperatorRB._ref, the same formulas)
    sld = (edc * ctx.xc).sum(1) / ctx.den
    q["z_ref"] = ctx.kwh * ec.mean(1) + ctx.z0
    q["th_ref"] = ctx.kwp * sp * np.arctan(slope) + ctx.th0
    q["zd_ref"] = ctx.kwh * edc.mean(1)
    q["thd_ref"] = ctx.kwp * sp * sld / (1 + slope * slope)
    q["e_z"], q["e_th"] = z - q["z_ref"], th - q["th_ref"]
    q["w_r"], q["q_r"] = zd - q["zd_ref"], thd - q["thd_ref"]
    # trim: hidden true running trim + pitch relative to the low-fidelity
    # running attitude (bow up positive) - the surface's own bow-up angle
    # (a hull that follows the surface keeps tau_e = tau0), soft floor
    q["trim_rel"] = sp * (th - ctx.th0)
    raw = ctx.tau0 + q["trim_rel"] - np.arctan(slope)
    q["tau_e"] = softplus_floor(raw, TAU_MIN, TAU_SOFT)
    # intake immersion (sim/lofi.py prop_submergence, with roll)
    xi, yi, zi = ctx.intake
    q["h_in"] = q["eta_in0"] + q["eta_in_add"] \
        - (z + sp * xi * th + yi * phi) - zi
    # water-relative (hydrodynamic) velocities and speed scales
    u_r, v_r = u - q["nu_c"][:, 0], v - q["nu_c"][:, 1]
    U_r = np.hypot(u_r, v_r)
    q["u_r"], q["v_r"], q["U_r"], q["U_g"] = u_r, v_r, U_r, np.hypot(u, v)
    q["qdyn"] = 0.5 * RHO_W * U_r ** 2
    q["qb2"] = q["qdyn"] * ctx.b ** 2
    q["Cv"] = U_r / np.sqrt(GRAV * ctx.b)
    q["Fn"] = U_r / np.sqrt(GRAV * ctx.L)
    q["Fn_vol"] = U_r / np.sqrt(GRAV * ctx.vol ** (1.0 / 3.0))
    return q


def sr_hydro(q):
    """The start state with (u, v) replaced by the water-relative ones:
    what layer 1 reads (E1: hydrodynamic terms read nu - nu_c)."""
    s = q["sr"].copy()
    s[:, 2], s[:, 9] = q["u_r"], q["v_r"]
    return s


def slam_events(ctx, q, emerged):
    """The low-fidelity boat's own slam rule (sim/lofi.py _measure) on the
    bow centre-line station: a slam when the bow was out of the water
    (keel immersion <= 0) and re-enters faster than v_slam. Returns (slam
    (B,) bool, entry speed (B,), new emerged flags)."""
    out = q["hc"][:, -1] <= 0.0
    vin = q["ddc"][:, -1]
    slam = emerged & ~out & (vin > ctx.v_slam)
    return slam, np.where(slam, vin, 0.0), out


def bow_height(ctx, sr, eta):
    """Bow height above the water (operators_gen.OperatorGen.bow_height):
    F_b + z + s_p x_b theta - the highest bow-station elevation."""
    sr = np.atleast_2d(sr)
    e = np.asarray(eta, float).reshape(len(sr), 5, 3)
    return ctx.F_b + sr[:, 3] + ctx.sp * ctx.x_st[-1] * sr[:, 5] \
        - e[:, 4, :].max(1)


safety_per_step = G.safety_per_step       # D9.6 as operators_gen has it


def obs_defaults(ctx, q):
    """The extra observed signals when their items are off: no roll, rpm
    from thrust ~ rpm^2, relative wind of still air, speed through water =
    over ground, GNSS good."""
    u, v = q["sr"][:, 2], q["sr"][:, 9]
    B = q["B"]
    return dict(roll=np.zeros(B), roll_rate=np.zeros(B),
                rpm=np.sqrt(np.maximum(q["thr"], 0.0) / ctx.t_max),
                wind_speed_rel=np.hypot(u, v),
                wind_angle_rel=np.arctan2(v, u),
                stw_u=q["u_r"].copy(), stw_v=q["v_r"].copy(),
                gnss_q=np.ones(B))


# ------------------------------------------------------------ item interface
class ItemOut:
    """What an item's step returns: acc (B, 5) accelerations and / or Q
    (B, 5) generalised forces (mapped with M_t), imp (B, 5) velocity jumps
    of the impulse channel (not clipped per substep), obs {name: (B,)}
    extra observed signals, state (the item's state dict)."""

    __slots__ = ("acc", "Q", "imp", "obs", "state")

    def __init__(self, acc=None, Q=None, imp=None, obs=None, state=None):
        self.acc, self.Q, self.imp = acc, Q, imp
        self.obs = {} if obs is None else obs
        self.state = state


class CatItem:
    """Base class of a catalogue item. Class attributes:
    code        catalogue code (CODES)
    stage       'env' | 'hidden' | 'force' | 'prop' | 'gate' (STAGES)
    gated       its heave / pitch parts are vertical hydrodynamic forces the
                contact gate W4 multiplies (W1, W5(a), W7 vertical)
    saturate    the framework soft-saturates its accelerations per channel
                at SAT x nominal(); off only for structural terms (W4)
    calm_zero   calm, steady, straight running gives exactly zero (draft
                7.1): checked by the harness at u_id and, with
                calm_params(), at all speeds"""

    code, stage = "", "force"
    gated, saturate, calm_zero = False, True, False

    def p_on(self, ctx):
        return P_ON[self.code]

    def draw(self, rng, ctx):
        """params, or None when the item is switched off this episode."""
        if rng.random() >= self.p_on(ctx):
            return None
        return self.draw_on(rng, ctx)

    def draw_on(self, rng, ctx):
        """params given that the item is on (sparse episodes call this)."""
        raise NotImplementedError

    def init_state(self, params, B=1, rng=None):
        return {}

    def step(self, params, state, q, dt):
        raise NotImplementedError

    def overlaps(self, params):
        return set(OVERLAP.get(self.code, ()))

    def nominal(self, params, ctx):
        """Per-channel acceleration scale (5,) of the soft saturation."""
        return ctx.a_ref

    def calm_params(self, params):
        """params for the all-speed calm-zero check (e.g. W1 without its
        speed extrapolation of the running trim)."""
        return params

    def adjust_ctx(self, params, ctx):
        """Draw-time change of the shared context (B1: masses, weight)."""


REGISTRY = {}
CHECKS = {}


def register(cls):
    """Class decorator of an item module: one instance per code."""
    if cls.code not in CODES:
        raise ValueError(f"register: unknown catalogue code {cls.code!r}")
    if cls.stage not in STAGES:
        raise ValueError(f"register: {cls.code} stage {cls.stage!r}")
    REGISTRY[cls.code] = cls()
    return cls


def check(code):
    """Decorator: a construction check fn(item, ctx) -> (ok, message) of
    one item (run by studies/test_cat_items.py)."""
    def deco(fn):
        CHECKS.setdefault(code, []).append(fn)
        return fn
    return deco


def load_items():
    """Import every item module learn/meta/cat_*.py (they register)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for f in sorted(glob.glob(os.path.join(here, "cat_*.py"))):
        name = os.path.splitext(os.path.basename(f))[0]
        if name != "cat_base":
            importlib.import_module(f"learn.meta.{name}")
    return REGISTRY


def item_accel(item, params, state, q, dt):
    """Run one item and map its output: (acc (B, 5) after the lever-arm
    mapping and the soft saturation, imp (B, 5), obs, acc before the
    saturation)."""
    ctx = q["ctx"]
    o = item.step(params, state, q, dt)
    B = q["B"]
    a = np.zeros((B, 5))
    if o.Q is not None:
        a = a + to_acc(ctx, o.Q)
    if o.acc is not None:
        a = a + np.asarray(o.acc, float).reshape(B, 5)
    raw = a
    if item.saturate:
        a = soft_sat(a, np.maximum(item.nominal(params, ctx), 1e-9))
    imp = np.zeros((B, 5)) if o.imp is None else np.asarray(o.imp, float)
    return a, imp, o.obs, raw


class ObsItem:
    """Slot of an observation-model item O1-O5 (draft section 4): applied to
    the recorded trajectory, never to the dynamics. OFF in D10a."""

    code = ""

    def draw(self, rng, ctx):
        raise NotImplementedError

    def apply(self, params, traj, rng):
        raise NotImplementedError


OBS_REGISTRY = {}


# ------------------------------------------------------------ layer 1 / 3
class OperatorRBL1(R.OperatorRB):
    """Layer 1: the D8.12 rigid-body family with its stability rule (every
    mode's damping ratio >= 0.05, closed-loop lateral stability) replaced
    by divergence-only rejection of the whole assembled draw (draft section
    1 change 1; D9.3): stable() accepts every draw, CatDraw is redrawn from
    the next attempt when the closed-loop acceptance run diverges."""

    def stable(self):
        return True


def apply_l1_switches(op, switches):
    """The overlap switches (draft section 1) on a drawn OperatorRBL1."""
    sw = set(switches)
    if "T8_off" in sw:
        op.enabled.discard("T8")
    if "T4_speed_const" in sw:           # constant stiffness multipliers,
        for c in (op.cKz, op.cKq):       # offset c0 only (changes 2, 9)
            c[1:] = 0.0
        op.cdz[1:] = 0.0
        op.cdq[1:] = 0.0
    if "T4_nl_off" in sw:
        op.nl = None
    if "T4_coupling_off" in sw:          # D10.8 (orchestrator decision on
        op.cks[:] = 0.0                  # D10.7 item 12): the heave-pitch
        op.cka[:] = 0.0                  # coupling k35 / k53 off with W1
    if "T5_etaT_off" in sw:
        op.ceT[:] = 0.0
    if "T5_etaS_off" in sw:
        op.ceS[:] = 0.0
        op.c3 = 0.0
    if "T7_quad_off" in sw:
        for md in op.modes:
            if md.get("quad") is not None:
                md["quad"] = None
    if "T3_kap_lat_off" in sw:
        op.kap[:2] = 0.0
    if "T3_kap_vert_off" in sw:
        op.kap[2:] = 0.0
    op._pack()
    return op


def l1_post(op, parts, sr_h, switches):
    """Switches applied to rb_accel's parts: remove T2c's rigid-body m v r
    (the D9.1 Coriolis difference carries it), and gate T3's surge part to
    the task speed range when W11 is on (change 12). In place."""
    if op.rb_off:
        return parts
    if "T2c_rigid_off" in switches and "T2c" in op.enabled:
        parts["T2c"][:, 0] -= op.m * sr_h[:, 9] * sr_h[:, 8] / op.m_u
    if "T3_surge_low_gate" in switches:
        parts["T3"][:, 0] *= 1.0 / (1.0 + np.exp(-(sr_h[:, 2] - op.u_lo)
                                                  / T3_GATE_W))
    return parts


def draw_tier(rng):
    return G._choice(rng, L3_TIERS)


class OperatorGenL3(G.OperatorGen):
    """Layer 3: the D9 general residual at the declared share (draft section
    5) and the D9.1 rigid-body part (M_t, the exact Coriolis difference).
    tier 'small': N ~ Poisson(1) forces (at most 3), each kappa ~ LogU[0.005,
    0.1]; 'fail': the D9.8 ranges (N = 1 + Poisson(2) <= 8, kappa0 ~
    LogU[0.01, 1] split by sqrt(Dirichlet)); 'none': no force (M_t and the
    Coriolis difference only). Streams [seed, CAT_STREAM, cat attempt,
    SID_L3, 1 + own attempt]; no old noise part (lib=None: the old T9 noise
    stays in layer 1); no own acceptance test (the whole draw is tested)."""

    def __init__(self, seed, tier, cat_attempt=0, p=None, geom=None, dt=0.24,
                 L=5.4):
        self.tier, self._cat_attempt = str(tier), int(cat_attempt)
        super().__init__(seed, lib=None, dt=dt, L=L, p=p, geom=geom,
                         accept=False)

    def _draw(self, attempt):
        rng = np.random.default_rng([self.seed, CAT_STREAM, self._cat_attempt,
                                     SID_L3, 1 + int(attempt)])
        self.attempt = int(attempt)
        self._draw_inertia(rng)
        self.psi_w = float(rng.uniform(0, 2 * np.pi))
        m = min(int(rng.poisson(G.N_HID_MEAN)), G.N_HID_MAX)
        self.hidden = [self._draw_hidden(rng) for _ in range(m)]
        self.groups = dict(state=list(range(0, 6)), act=[6, 7],
                           nu0=list(range(8, 13)), imm=list(range(13, 28)))
        if m:
            self.groups["hid"] = list(range(28, 28 + m))
        self.groups["dir"] = [28 + m, 29 + m]
        self.n_in = 30 + m
        if self.tier == "small":
            n = min(int(rng.poisson(L3_SMALL_MEAN)), L3_SMALL_MAX)
            kap = _logu(rng, *L3_SMALL_KAPPA, n) if n else np.zeros(0)
        elif self.tier == "fail":
            n = min(1 + int(rng.poisson(L3_FAIL_MEAN)), L3_FAIL_MAX)
            k0 = float(_logu(rng, *L3_FAIL_KAPPA0))
            kap = k0 * np.sqrt(rng.dirichlet(np.ones(n)))
        elif self.tier == "none":
            kap = np.zeros(0)
        else:
            raise ValueError(f"layer-3 tier {self.tier!r}")
        self.kappa0 = float(np.sqrt((np.asarray(kap) ** 2).sum()))
        self.forces = []
        if len(kap):
            Xl = self._lib_inputs()
            self.forces = [self._draw_force(rng, float(k), Xl) for k in kap]
            del Xl
        self.style = self._style()
        self.style.update(l3_tier=self.tier)


# ---------------------------------------------------------------- the draw
class CatDraw:
    """One draw of the D10 prior: layers 1-3 and the catalogue items.

    seed, attempt   streams [seed, CAT_STREAM, attempt, ...]; a draw whose
                    closed-loop acceptance run diverges is replaced by
                    next_attempt() (divergence only, D9.3 / D9.8 item 2)
    lib             the M15 input library dict (mu, sd, S) layer 1 needs
                    (its T7 loads and old T9 noise); None = no layer 1
    p, keel, intake low-fidelity parameters and geometry (CatCtx)
    items           codes allowed (default every registered item)
    force_items     {code: params or True}: items switched on regardless of
                    the draw (tests, per-item previews)
    sparse, tier    None = drawn; else forced
    obs_on          the observation model (OFF in D10a): adds 'G12_off'
    act             None (ideal) or a dict of lofi.ACT_KEYS scalars: the
                    episode's M10 actuator, needed only when a P12 rule
                    changes the command (the true actuator copy)"""

    def __init__(self, seed, lib=None, p=None, keel=None, intake=None,
                 v_slam=None, dtc=0.24, attempt=0, has_waves=True,
                 obs_on=False, items=None, force_items=None, sparse=None,
                 tier=None, layer1=True, act=None):
        self._kw = dict(lib=lib, p=p, keel=keel, intake=intake, v_slam=v_slam,
                        dtc=dtc, has_waves=has_waves, obs_on=obs_on,
                        items=items, force_items=force_items, sparse=sparse,
                        tier=tier, layer1=layer1, act=act)
        load_items()
        self.seed, self.attempt, self.dtc = int(seed), int(attempt), float(dtc)
        self.ctx = ctx = CatCtx(p, keel=keel, intake=intake, v_slam=v_slam,
                                has_waves=has_waves)
        self.act = act
        rm = self._rng(SID_META)
        ctx.set_tau0(rm.uniform(*TAU0))
        self.sparse = (rm.random() < SPARSE_P) if sparse is None \
            else bool(sparse)
        self.tier = draw_tier(rm) if tier is None else str(tier)
        # layer 3 first: its M_t is the shared inertia of every lever arm
        self.op_gen = OperatorGenL3(self.seed, self.tier, self.attempt,
                                    p=ctx.p, dt=dtc, L=ctx.L)
        ctx.set_inertia(self.op_gen.Mt, self.op_gen.inertia)
        self.on = self._draw_items(rm, items, force_items or {})
        self.switches = set(ALWAYS_SWITCHES)
        for code, prm in self.on.items():
            self.switches |= REGISTRY[code].overlaps(prm)
        if obs_on:
            self.switches.add(OBS_SWITCH)
        self.op_rb = None
        if layer1 and lib is not None:
            s1 = int(np.random.SeedSequence(
                [self.seed, CAT_STREAM, self.attempt, SID_L1]).generate_state(
                    1)[0])
            self.op_rb = apply_l1_switches(
                OperatorRBL1(s1, lib, dt=dtc, L=ctx.L, p=ctx.p),
                self.switches)
            if self.op_rb.t5 and not self.op_rb.rb_off:
                ctx.x_j, ctx.z_j = float(self.op_rb.x_j), float(
                    self.op_rb.z_j)
                ctx.kc = float(self.op_rb.kc)
        self.prop_on = [c for c in PROP_ORDER if c in self.on]
        self.reject_log = []

    def _rng(self, sid, *more):
        return np.random.default_rng([self.seed, CAT_STREAM, self.attempt,
                                      int(sid)] + [int(m) for m in more])

    def _draw_items(self, rm, items, force):
        ctx = self.ctx
        allowed = [c for c in CODES if c in REGISTRY
                   and (items is None or c in items)]
        on = {}
        chosen = None
        if self.sparse:
            pool = [c for c in allowed if c not in STRUCTURAL]
            n = int(rm.integers(SPARSE_N[0], SPARSE_N[1] + 1))
            chosen = set(rm.choice(pool, min(n, len(pool)), replace=False)) \
                if pool else set()
            for c in list(chosen):             # a dependent brings its parent
                if c in DEPENDS and DEPENDS[c] in allowed:
                    chosen.add(DEPENDS[c])
        # B1 (and any item that adjusts the context) before the others
        order = sorted(allowed, key=lambda c: 0 if c == "B1" else 1)
        for c in order:
            it = REGISTRY[c]
            rng = self._rng(SID_ITEM + CODES.index(c))
            if c in force:
                prm = force[c]
                on[c] = it.draw_on(rng, ctx) if prm is True else dict(prm)
            elif c in WAVES_ONLY and not ctx.has_waves:
                continue
            elif c in DEPENDS and DEPENDS[c] not in on:
                continue
            elif chosen is not None and c not in STRUCTURAL:
                if c in chosen:
                    on[c] = it.draw_on(rng, ctx)
            else:
                prm = it.draw(rng, ctx)
                if prm is not None:
                    on[c] = prm
            if c in on:
                it.adjust_ctx(on[c], ctx)
        return {c: on[c] for c in CODES if c in on}

    def next_attempt(self, reason=""):
        """The replacement draw after a divergence (same seed, next
        attempt stream); the rejection is logged in its reject_log."""
        nxt = CatDraw(self.seed, attempt=self.attempt + 1, **self._kw)
        nxt.reject_log = self.reject_log + [dict(
            attempt=self.attempt, reason=reason, tier=self.tier,
            sparse=self.sparse, items=sorted(self.on))]
        return nxt

    # ------------------------------------------------------------- state
    def new_state(self, B=1, hid_seed=0, rng=None):
        """Fresh state for B rows: every item's (own process stream
        [seed, CAT_STREAM, attempt, SID_ITEM_STATE + index, hid_seed]),
        layer 1's (rng pre-rolls its old noise), layer 3's, the slam
        detector's."""
        st = dict(k=0, items={}, emerged=np.zeros(B, bool), act=None)
        for c, prm in self.on.items():
            r = self._rng(SID_ITEM_STATE + CODES.index(c), hid_seed)
            st["items"][c] = REGISTRY[c].init_state(prm, B, rng=r)
        st["l1"] = None if self.op_rb is None else self.op_rb.new_state(
            B, rng=rng)
        st["l3"] = self.op_gen.new_state(B, hid_seed=hid_seed,
                                         forces=bool(self.op_gen.forces))
        return st

    snapshot = staticmethod(copy.deepcopy)

    def step(self, st, s_raw, noise=True, rng=None, freeze=False):
        """The control-step part (held over the substeps): layer 1's old
        noise part (T9, w_noise) with its rule part T8 off (so the rule is
        0). Returns (rule, noise) (B, 5), both to be clip_push'ed."""
        s_raw = np.atleast_2d(s_raw)
        if self.op_rb is None:
            z = np.zeros((len(s_raw), 5))
            st["k"] += 1
            return z, z.copy()
        r, n = self.op_rb.step(st["l1"], s_raw, noise=noise, rng=rng,
                               freeze=freeze)
        st["k"] = st["l1"]["k"]
        return r, n

    # ----------------------------------------------------------- substep
    def substep(self, st, sr, thr, noz, sea, nu0=None, cmd=None, dt=DT_SUB,
                t=None, freeze=False, parts=False):
        """THE substep of the D10 prior, batched over the B rows of st, in
        the layer order of the module docstring. sr (B, 10) the START state;
        thr, noz the low-fidelity boat's applied actuators; sea a
        CatSea.sample dict at the substep start; nu0 (B, 5) the low-fidelity
        velocity increment / dt over the substep (before any injection);
        cmd (B, 2) the physical commands. Returns dict(acc (B, 5) the rule
        part (clip_push with the held part), imp (B, 5) impulse-channel
        velocity jumps (added after the kick, not clipped), obs {OBS_EXTRA:
        (B,)}, slam (B,), q the shared quantities, and with parts=True
        PARTS {name: (B, 5)})."""
        ctx = self.ctx
        q = substep_quantities(ctx, sr, thr, noz, sea, nu0=nu0, cmd=cmd,
                               dt=dt, t=t)
        B = q["B"]
        acc, imp = np.zeros((B, 5)), np.zeros((B, 5))
        obs = obs_defaults(ctx, q)
        P = {} if parts else None
        q["a_cat_vert"] = np.zeros((B, 2))
        items = st["items"]

        def run(code):
            nonlocal acc, imp
            it = REGISTRY[code]
            a, dv, ob, _ = item_accel(it, self.on[code], items[code], q, dt)
            acc = acc + a
            imp = imp + dv
            obs.update(ob)
            if it.gated:
                q["a_cat_vert"] += a[:, 3:5]
            if parts:
                P[code] = a
                if dv.any():
                    P[code + ":imp"] = dv

        # 1. environment, then hidden degrees of freedom (re-derive after)
        for stage in ("env", "hidden"):
            codes = [c for c in self.on if REGISTRY[c].stage == stage]
            for c in codes:
                run(c)
            if codes:
                derive_quantities(ctx, q)
        slam, vslam, st["emerged"] = slam_events(ctx, q, st["emerged"])
        q["slam"], q["slam_v"] = slam, vslam
        # 2. layer 1 at the water-relative velocities, switches applied
        a1 = np.zeros((B, 5))
        q["a_l1_vert"] = np.zeros((B, 2))
        if self.op_rb is not None:
            sh = sr_hydro(q)
            pr = self.op_rb.rb_accel(st["l1"], sh, q["thr"], q["noz"],
                                     q["eta"], q["etad"], dt, parts=True)
            pr = l1_post(self.op_rb, pr, sh, self.switches)
            a1 = sum(pr.values())
            # every vertical hydrodynamic part of layer 1 (W4 gates them):
            # T3 / T4 / T7 and T2c's speed coupling and Teven's quadratic
            # loads (review 2026-09-30); not T5 (thrust) or T9 (old noise)
            q["a_l1_vert"] = sum((pr[c][:, 3:5] for c in L1_VERT if c in pr),
                                 np.zeros((B, 2)))
            if parts:
                P.update({"L1:" + c: v for c, v in pr.items()})
        acc = acc + a1
        # 3. layer 2: forces, the propulsion chain, the contact gate
        for c in self.on:
            if REGISTRY[c].stage == "force":
                run(c)
        if self.prop_on:
            a_p, ob = self._prop(st, q, dt)
            acc = acc + a_p
            obs.update(ob)
            if parts:
                P["prop"] = a_p
        for c in self.on:
            if REGISTRY[c].stage == "gate":
                run(c)
        # 4. layer 3 (forces on the low-fidelity quantities) + Coriolis
        if self.op_gen.forces:
            pg = self.op_gen.gen_accel(st["l3"], q["sr"], q["thr"], q["noz"],
                                       nu0 if nu0 is not None else q["nu0"],
                                       q["eta0"], dt, freeze=freeze,
                                       parts=True)
            cor = pg.pop("cor")
            a3 = sum(pg.values()) if pg else np.zeros((B, 5))
        else:
            cor, a3 = self.op_gen.coriolis(q["sr"]), np.zeros((B, 5))
        acc = acc + a3 + cor
        if parts:
            P["L3"], P["cor"] = a3, cor
        out = dict(acc=acc, imp=imp, obs=obs, slam=slam, q=q)
        if parts:
            out["PARTS"] = P
        return out

    def _prop(self, st, q, dt):
        """The propulsion chain in PROP_ORDER (draft 3.5): commands (P12) ->
        the actuator (M10: the low-fidelity boat's applied values, or a
        true-actuator copy when a rule changed the command) -> nozzle (P7,
        P11) -> T_pre = P3 x P10 x P1 x P13 x P2 x P9 -> P4 saturation ->
        side force (P5, default the low-fidelity law) -> bucket (P8) ->
        P14 -> P6 vertical force. Error = the jet force at the layer-1 jet
        point minus the low-fidelity boat's nominal one, through M_t."""
        ctx = q["ctx"]
        B = q["B"]
        cmd = q["cmd"] if q["cmd"] is not None else np.stack(
            [q["thr"], q["noz"]], 1)
        pc = dict(c_thr=cmd[:, 0].copy(), c_noz=cmd[:, 1].copy(),
                  cmd_changed=False, thr_act=q["thr"].copy(),
                  noz_act=q["noz"].copy(), u_r=q["u_r"], T=None,
                  rho=np.ones(B), p=np.ones(B), nu_fac=np.ones(B),
                  pulse=np.zeros(B), T_pre=None, T_eff=None, Fx=None,
                  Fy=None, Fz=np.zeros(B), dF=np.zeros((B, 2)))
        q["prop"] = pc
        obs = {}
        for code in PROP_ORDER:
            if code == "M10":
                if pc["cmd_changed"]:
                    self._true_actuator(st, pc, dt)
                pc["noz_eff"] = pc["noz_act"].copy()
                continue
            if code == "P4":
                self._t_pre(pc)
            if code == "P5":
                if pc["T_pre"] is None:
                    self._t_pre(pc)
                if pc["T_eff"] is None:
                    pc["T_eff"] = pc["T_pre"]
            if code in self.on:
                o = REGISTRY[code].step(self.on[code], st["items"][code], q,
                                        dt)
                obs.update(o.obs)
            if code == "P5" and pc["Fy"] is None:
                pc["Fy"] = ctx.k_js * np.maximum(pc["T_eff"], 0.0) * np.sin(
                    np.clip(pc["noz_eff"], -ctx.rud_stall, ctx.rud_stall))
        ax = lambda d: 1.0 + ctx.kc * (np.cos(d) - 1.0)          # noqa: E731
        Fx = pc["T_eff"] * ax(pc["noz_eff"]) if pc["Fx"] is None else pc["Fx"]
        Fx0 = q["thr"] * ax(q["noz"])
        Fy0 = ctx.k_js * np.maximum(q["thr"], 0.0) * np.sin(
            np.clip(q["noz"], -ctx.rud_stall, ctx.rud_stall))
        F = np.stack([Fx - Fx0 + pc["dF"][:, 0], pc["Fy"] - Fy0
                      + pc["dF"][:, 1], pc["Fz"]], 1)
        Q = lever_Q(ctx, (ctx.x_j, 0.0, ctx.z_j), F, q["sr"][:, 5])
        del q["prop"]
        return to_acc(ctx, Q), obs

    @staticmethod
    def _t_pre(pc):
        T = pc["thr_act"] if pc["T"] is None else pc["T"]
        pc["T_pre"] = T * pc["rho"] * pc["p"] * pc["nu_fac"] \
            * (1.0 + pc["pulse"])

    def _true_actuator(self, st, pc, dt):
        """The true boat's actuator fed with the changed command: ideal, or
        the episode's M10 chain (lofi.act_step, its own delay line)."""
        ctx = self.ctx
        c = np.stack([np.clip(pc["c_thr"], 0.0, ctx.t_max),
                      np.clip(pc["c_noz"], -ctx.rud_max, ctx.rud_max)], 1)
        if self.act is None:
            pc["thr_act"], pc["noz_act"] = c[:, 0], c[:, 1]
            return
        from sim import lofi
        qa = lofi.act_arrays(self.act, ctx.t_max, ctx.rud_max)
        D = int(qa["delay"].max())
        if st["act"] is None:
            pos = np.stack([pc["thr_act"], pc["noz_act"]], 1)
            st["act"] = dict(line=np.repeat(pos[:, None], max(D, 1), 1),
                             pos=pos, vel=np.zeros_like(pos))
        a = st["act"]
        full = np.concatenate([a["line"], c[:, None]], 1)
        cd = full[:, D - qa["delay"], [0, 1]] if D else c
        a["line"] = full[:, 1:]
        a["pos"], a["vel"], app = lofi.act_step(qa, a["pos"], a["vel"], cd,
                                                dt)
        pc["thr_act"], pc["noz_act"] = app[:, 0], app[:, 1]

    # ------------------------------------------------------------- style
    def style(self):
        def fl(v):
            if isinstance(v, (float, int, np.floating, np.integer)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                return {k: fl(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [fl(x) for x in v]
            return v
        return dict(prior_family="cat", seed=self.seed, attempt=self.attempt,
                    tier=self.tier, sparse=self.sparse,
                    tau0=self.ctx.tau0, items=sorted(self.on),
                    item_params={c: fl(p) for c, p in self.on.items()},
                    switches=sorted(self.switches),
                    l1=None if self.op_rb is None else dict(
                        rb_off=bool(self.op_rb.rb_off), t3=self.op_rb.t3,
                        t4=self.op_rb.t4, t5=self.op_rb.t5,
                        n_loads=len(self.op_rb.modes)),
                    l3=dict(n_forces=len(self.op_gen.forces),
                            kappa0=self.op_gen.kappa0),
                    n_reject=len(self.reject_log))
