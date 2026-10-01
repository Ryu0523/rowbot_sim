#!/usr/bin/env python3
"""
The general family of error operators: a rigid body acted on by ARBITRARY
causal forces (learn/meta/PRIOR_DERIVATION.md D9, as revised in D9.8 after
the critique). A standalone module with the interface of operators_rb.py so
a pipeline can switch families; nothing in data2 / operators uses it yet.

What a draw is. One imagined "true boat" = the low-fidelity boat plus an
extra acceleration on the five velocity channels (surge u, sway v, yaw rate
r, heave rate zdot, pitch rate thdot), the same place the other families
push. Rigid-body mechanics fixes only how a force becomes motion:
  - errors enter at the acceleration level only (positions follow by the
    exact kinematics of the low-fidelity step);
  - an unknown positive-definite inertia M_t = M_RB + M_A (rigid body with
    a random CG offset + a non-negative diagonal added part), anchored to
    the low-fidelity boat's own surge / sway mass and its jet-implied yaw
    inertia, turns point forces and distributed loads into accelerations
    (the lever arm fixes the channel ratios);
  - the exact rigid-body Coriolis / centripetal difference to the low-
    fidelity boat's -m u r (always on, a structural term).
The forces themselves are arbitrary causal functions of everything already
experienced (the Boyd-Chua structure): sparse random combinations of the
inputs (state, actual actuator positions, the low-fidelity boat's own
acceleration before injection, local immersion at the 5 x 3 stations,
hidden processes, heading relative to a random earth-fixed direction) ->
a bank of random stable filters (exact discretisation, pure delays,
two-threshold relays) -> a random small network readout -> soft
saturation -> a point force, a distributed load or a generic acceleration.
No hydrodynamic form, sign or symmetry. Inputs are standardised on a LOW-
FIDELITY-ONLY library (input_library: the low-fidelity boat alone in
closed loop, the data autopilot, training seas; D9.8 item 1), and so is
every filter output (each draw runs its own filters on the library); the
readout's reference cloud, biases and RMS normalisation come from those
library filter outputs, and relay thresholds are in units of the library
spread of the relay input. The force SIZE is set by kappa and the low-
fidelity boat's own per-channel accelerations a_ref (D9.3), not by the
input scaling.

Only safeguards: per-channel caps (|a_c| <= 3 kappa_i a_ref,c per force,
a_ref from the low-fidelity boat's own parameters), the unchanged D5 clip
(clip_push, checked, never a reason to reject), and an ACCEPTANCE TEST that
is the same for every draw (accept_draws: a perturbed calm-water row and a
standard-sea row, 60 s, the data autopilot, ideal actuators) whose ONLY
criterion is divergence (non-finite, u > 1.5 u_max, |theta| > 45 deg; D9.8
item 2). A rejected draw is redrawn from the next attempt stream of the
same seed; the episode's own conditions are untouched. A divergence inside
an episode truncates and flags the row (simulate_gen 'div'), it is not
redrawn (D9.8 item 7).

The old family's noise part (T9: coloured / white noise, bursts, with its
state-dependent envelope) is kept at the control step with weight w_noise;
its rule part (T8), relays and events are not (the arbitrary forces take
their place).

Safety quantities (D9.6 as revised in D9.8 item 10): per control step,
a_z,j = (zdot_j - zdot_{j-1}) / dt after injection at the low-fidelity
reference point (the nominal CG); APK = max_j, AMIN = min_j; HMIN = min over the substep ENDS
j = 1..6 of F_b + z + s_p x_b theta - max(elevation at the three bow
stations).
"""
import copy

import numpy as np
from scipy.linalg import expm

from learn.meta import operators_rb as R
from learn.meta.operators import A_REF, CLIP, _logu, clip_push

CHANNELS = R.CHANNELS
RED_VEL = R.RED_VEL
VEL_IDX = R.VEL_IDX
KN = R.KN
U_RANGE = R.U_RANGE
GEN_STREAM = 41                    # draws: [seed, 41] and [seed, 41, 1 + attempt]
HID_STREAM = 43                    # hidden processes' innovations
TEST_HID = 47                      # hidden stream id used in the acceptance test
set_slow = R.set_slow              # family-neutral slow-state restore

# ---------------------------------------------------------------------------
# Declared hyper-parameters (D9.8). All are assumptions; scales are in units
# of the low-fidelity boat's own terms or of the hull.
P_NONE = 0.05                      # no force at all (Coriolis difference only)
N_FORCE_MEAN, N_FORCE_MAX = 2.0, 8     # N = 1 + Poisson(2), at most 8
# draw-level size kappa0 ~ LogU, split over the forces by sqrt(Dirichlet)
# (D9.8 item 3)
KAPPA0 = (0.01, 1.0)
W_NOISE = (0.1, 1.0)               # weight of the old noise part (T9), LogU
P_KIND = (("point", 0.5), ("dist", 0.2), ("gen", 0.3))
P_POINT_DIR = (("body", 0.6), ("earth", 0.2), ("readout", 0.2))
P_GROUP = 0.5                      # each input group selected w.p. 0.5
N_COMB_MEAN, N_COMB_MAX = 2.0, 6   # combinations 1 + Poisson(2), each of 1-3 inputs
N_FILT_MEAN, N_FILT_MAX = 3.0, 8   # filters 1 + Poisson(3)
P_FILT = (("direct", 0.2), ("low", 0.25), ("lead", 0.15), ("res", 0.3),
          ("relay", 0.1))
P_DELAY = 0.2
TAU = (0.04, 30.0)                 # low-pass time constant, s (clamped bounds)
LEAD_TAU2 = (0.04, 3.0)            # lead: pole time constant, s
LEAD_RATIO = (1.0, 10.0)           # lead: zero / pole time constant ratio
OMEGA = (0.3, 20.0)                # resonance, rad/s (clamped bounds)
ZETA = (0.05, 2.0)
DELAY_S = (0.04, 1.0)              # pure delay, s
RELAY_TAU = (0.04, 3.0)            # relay input low-pass, s
# relay thresholds: library mean + (hi, hi - width) x the library std of the
# relay's low-passed input (D9.8 item 6)
RELAY_HI, RELAY_W = (-1.5, 1.5), (0.1, 1.5)
DT_SUB = 0.04                      # one substep: the fixed lower time bound
SAT = 3.0                          # soft saturation of every readout output
WIDTH, GAIN = (8, 64), (0.3, 3.0)
ACTS = ("tanh", "softplus", "relu", "abs")
P_TWO_LAYERS, P_PAIR, P_BIAS, BIAS_SHARE = 0.3, 0.3, 0.5, (0.0, 0.5)
N_MC = 1024                        # reference samples for the readout's RMS
N_HID_MEAN, N_HID_MAX = 1.5, 4
P_HID = (("ou", 0.4), ("drift", 0.2), ("jump", 0.2), ("switch", 0.2))
HID_OU_TAU, HID_DRIFT_TAU = (0.3, 300.0), (60.0, 3000.0)
HID_RATE, HID_DF, HID_CAP = (1 / 300, 1 / 10), (1.5, 5.0), 10.0
HID_DWELL = (2.0, 120.0)
TH_MAX = np.pi / 4                 # the divergence limit; bounds the -theta Fx term
# the Scarab hull (learn/repro/task.ctx() hull: vcg 0.55 m above the keel,
# depth 1.1 m); the low-fidelity boat carries the same geometry
GEOM = dict(vcg=0.55, depth=1.1)
# acceptance test (D9.8 item 2): the same for every draw, divergence only
N_REJECT = 30
TEST_STEPS, TEST_SKIP = 250, 42    # 60 s; the clip share recorded from 10 s
TEST_SPEED = 23 * KN
TEST_SEED = 424242
TEST_SEA = dict(hs=1.0, tp=5.5, theta0=0.0, n_freq=48, n_dir=8, jitter=True,
                spread_s=8.0)
TEST_SEA_SEED = 777001
# the input library (D9.8 item 1): the low-fidelity boat ALONE in closed
# loop (the data autopilot, informative rows, training seas, ideal
# actuators); fixed seeds, the same in every process
LIB_ROWS, LIB_STEPS, LIB_SKIP = 12, 250, 42    # 12 x 60 s; stats from 10 s
LIB_SEED = 515151
LIB_HID = 53                       # hidden stream id of a draw's library run
N_RAW = 28                         # library-standardised input columns
SD_FLOOR = 1e-9


def _choice(rng, table):
    u, acc = rng.random(), 0.0
    for name, pr in table:
        acc += pr
        if u < acc:
            return name
    return table[-1][0]


def _act(name, x):
    if name == "tanh":
        return np.tanh(x)
    if name == "softplus":
        return np.logaddexp(0.0, x)
    if name == "relu":
        return np.maximum(x, 0.0)
    return np.abs(x)


def freeboard(p, geom=None):
    """Static freeboard at the bow station: depth - bow draught (0.885 m
    for the Scarab: planing_vessel.py freeboard = depth + keel_rest[-1])."""
    g = dict(GEOM, **(geom or {}))
    return float(g["depth"] - p["draft_bow"])


def a_ref_of(p):
    """Per-channel reference accelerations from the low-fidelity boat's own
    parameters (D9.3): full thrust / surge mass; full-nozzle side force /
    sway mass; its yaw acceleration; the restoring of one draught in heave;
    and of one draught at half a length in pitch."""
    side = p["k_jet_side"] * p["t_max"] * np.sin(p["rud_max"])
    return np.array([p["t_max"] / p["m_surge"], side / p["m_sway"],
                     side * abs(p["k_nomoto_f"]) / p["tau_r"],
                     p["wn_heave"] ** 2 * p["T"],
                     p["wn_pitch"] ** 2 * p["T"] / (0.5 * p["L"])])


class OperatorGen:
    """One draw of the general family (module docstring).

    Same constructor arguments as operators_rb.OperatorRB (plus geom and
    accept), the same step(st, s_raw, noise, rng, freeze, hold_relays) ->
    (rule, noise) (B, 5) for the control-step part (here: rule 0, noise the
    old T9 part), the same snapshot / slow_state / set_slow / new_state. The
    substep part is gen_accel(st, sr, thr, noz, nu0, eta, dt), called by
    simulate_gen and GenInjector AFTER the low-fidelity step of the substep
    (nu0 = the low-fidelity boat's own acceleration over it, before any
    injection) with the substep's START state. accept=True runs the
    acceptance test now (accept_draws for many draws at once)."""

    def __init__(self, seed, lib=None, dt=0.24, L=5.4, relay=False,
                 null=None, p=None, u_range=U_RANGE, geom=None, accept=True):
        self.seed, self.dt = int(seed), float(dt)
        self.p = dict(R.default_params() if p is None else p)
        self.geom = dict(GEOM, **(geom or {}))
        self.res = None
        if lib is not None:
            from learn.meta.ops_m15 import OperatorM15
            self.res = OperatorM15(self.seed, lib, dt=dt, L=L, relay=relay,
                                   null=null)
        rng = np.random.default_rng([self.seed, GEN_STREAM])
        self.w_noise = float(_logu(rng, *W_NOISE))
        self._consts(u_range)
        self.attempt, self.n_reject, self.reject_log = 0, 0, []
        self.forces_off, self.accepted = False, False
        self._draw(0)
        if accept:
            accept_draws([self])

    # ------------------------------------------------------------ constants
    def _consts(self, u_range):
        p = self.p
        self.L, self.T, self.B = float(p["L"]), float(p["T"]), float(p["B"])
        self.rs = np.sqrt(self.L / 10.0)
        self.m0 = float(p["m_coriolis"])
        self.m_u, self.m_v = float(p["m_surge"]), float(p["m_sway"])
        self.x_st = np.linspace(p["x_stern"], p["x_bow"], 5)
        self.y_off = np.array([-0.5 * self.B, 0.0, 0.5 * self.B])
        self.xs15 = np.repeat(self.x_st, 3)
        self.x_b, self.sp = float(self.x_st[-1]), float(p["sign_pitch"])
        self.z0, self.th0 = float(p.get("z0", 0.0)), float(p.get("th0", 0.0))
        self.u_lo, self.u_hi = float(u_range[0]), float(u_range[1])
        self.u_mid = 0.5 * (self.u_lo + self.u_hi)
        self.u_half = 0.5 * (self.u_hi - self.u_lo)
        self.u_max = float(p["u_max"])
        self.t_max, self.rud_max = float(p["t_max"]), float(p["rud_max"])
        self.a_ref = a_ref_of(p)
        # the low-fidelity jet's yaw acceleration per newton of side force
        # at the nozzle (the transom station): its implied yaw inertia
        self.x_n0 = float(p["x_stern"])
        self.I_r0 = abs(self.x_n0) / abs(p["k_nomoto_f"] / p["tau_r"])
        self.F_b = freeboard(p, self.geom)
        vcg, D = self.geom["vcg"], self.geom["depth"]
        self.z_box = (-vcg - 0.5 * self.T, D - vcg)   # from the reference point
        self.x_box = (self.x_st[0] - 0.05 * self.L,
                      self.x_st[-1] + 0.05 * self.L)

    # ---------------------------------------------------------------- draws
    def _draw(self, attempt):
        """The whole boat for this attempt: inertia, earth direction, hidden
        processes, forces."""
        rng = np.random.default_rng([self.seed, GEN_STREAM, 1 + int(attempt)])
        self.attempt = int(attempt)
        self._draw_inertia(rng)
        self.psi_w = float(rng.uniform(0, 2 * np.pi))
        m = min(int(rng.poisson(N_HID_MEAN)), N_HID_MAX)
        self.hidden = [self._draw_hidden(rng) for _ in range(m)]
        self.groups = dict(state=list(range(0, 6)), act=[6, 7],
                           nu0=list(range(8, 13)), imm=list(range(13, 28)))
        if m:
            self.groups["hid"] = list(range(28, 28 + m))
        self.groups["dir"] = [28 + m, 29 + m]
        self.n_in = 30 + m
        self.kappa0 = float(_logu(rng, *KAPPA0))
        self.forces = []
        if rng.random() >= P_NONE:
            n = min(1 + int(rng.poisson(N_FORCE_MEAN)), N_FORCE_MAX)
            w = rng.dirichlet(np.ones(n))
            Xl = self._lib_inputs()          # this draw's inputs on the library
            self.forces = [self._draw_force(rng, self.kappa0 * np.sqrt(wi),
                                            Xl) for wi in w]
            del Xl
        self.style = self._style()

    def _lib_inputs(self):
        """This draw's full input matrix on the library (LIB_ROWS, T, n_in):
        the standardised library columns, the draw's hidden processes run
        on the library's time grid from their own stream [seed, 43,
        attempt, LIB_HID], the heading relative to psi_w."""
        L = input_library()
        R_, T = L["raw"].shape[:2]
        X = np.empty((R_, T, self.n_in))
        X[..., :N_RAW] = (L["raw"] - L["mu"]) / L["sd"]
        m = len(self.hidden)
        if m:
            st = self.new_state(R_, hid_seed=LIB_HID, forces=False)
            for t in range(T):
                X[:, t, N_RAW:N_RAW + m] = st["hid"]
                self._advance_hidden(st, DT_SUB)
        X[..., N_RAW + m] = np.cos(L["psi"] - self.psi_w)
        X[..., N_RAW + m + 1] = np.sin(L["psi"] - self.psi_w)
        return X

    def _draw_inertia(self, rng):
        """M_t = M_RB + M_A (D9.8 item 9): rigid mass below the low-fidelity
        sway / surge inertia, CG offset, radii of gyration, axis tilt;
        diagonal of M_t anchored to the low-fidelity boat (surge, sway) and
        its jet-implied yaw inertia; heave / pitch added parts drawn."""
        L, B, T = self.L, self.B, self.T
        m = float(rng.uniform(0.9 * self.m0, 0.97 * min(self.m_u, self.m_v)))
        xg = float(rng.uniform(-0.05, 0.05)) * L
        yg = float(rng.uniform(-0.03, 0.03)) * B
        zg = float(rng.uniform(-0.3, 0.3)) * T
        ryy, rzz = rng.uniform(0.22, 0.28, 2) * L
        rxx = float(rng.uniform(0.35, 0.45)) * B
        delta = np.deg2rad(rng.uniform(-5.0, 5.0))
        Ixz = 0.5 * m * (rzz ** 2 - rxx ** 2) * np.sin(2 * delta) \
            + m * xg * zg
        Iyy = m * (ryy ** 2 + xg ** 2 + zg ** 2)
        Izz = m * (rzz ** 2 + xg ** 2 + yg ** 2)
        I_r = max(self.I_r0 * float(_logu(rng, 0.8, 1.25)), 1.05 * Izz)
        m_w = m * (1.0 + float(_logu(rng, 0.3, 2.5)))
        I_q = Iyy * (1.0 + float(_logu(rng, 0.3, 2.5)))
        Mrb = np.diag([m, m, Izz, m, Iyy])
        # channels (u, v, r, w, q): the CG-offset couplings m S(r_g)
        for i, j, val in ((0, 2, -m * yg), (0, 4, m * zg), (1, 2, m * xg),
                          (3, 4, -m * xg)):
            Mrb[i, j] = Mrb[j, i] = val
        MA = np.diag([self.m_u - m, self.m_v - m, I_r - Izz, m_w - m,
                      I_q - Iyy])
        self.m, self.xg, self.yg, self.zg, self.Ixz = m, xg, yg, zg, Ixz
        self.Mt = Mrb + MA
        self.Minv = np.linalg.inv(self.Mt)
        self.inertia = dict(m=m, xg=xg, yg=yg, zg=zg, Ixz=float(Ixz),
                            I_r=float(I_r), m_w=float(m_w), I_q=float(I_q),
                            Izz=float(Izz), Iyy=float(Iyy))

    def _draw_hidden(self, rng):
        kind = _choice(rng, P_HID)
        if kind in ("ou", "drift"):
            tau = float(_logu(rng, *(HID_OU_TAU if kind == "ou"
                                     else HID_DRIFT_TAU)))
            return dict(kind=kind, tau=tau)
        if kind == "jump":
            return dict(kind=kind, rate=float(_logu(rng, *HID_RATE)),
                        df=float(rng.uniform(*HID_DF)))
        return dict(kind=kind, dwell=float(_logu(rng, *HID_DWELL)))

    def _tclamp(self, tau):
        return float(np.clip(tau * self.rs, DT_SUB, TAU[1]))

    def _draw_filters(self, rng, n_p):
        K = min(1 + int(rng.poisson(N_FILT_MEAN)), N_FILT_MAX)
        filt = []
        for _ in range(K):
            kind = _choice(rng, P_FILT)
            f = dict(kind=kind, src=int(rng.integers(n_p)), delay=0.0)
            if rng.random() < P_DELAY:
                f["delay"] = float(np.clip(_logu(rng, *DELAY_S) * self.rs,
                                           DELAY_S[0], DELAY_S[1]))
            if kind == "low":
                f["tau"] = self._tclamp(_logu(rng, *TAU))
            elif kind == "lead":
                f["tau2"] = self._tclamp(_logu(rng, *LEAD_TAU2))
                f["tau1"] = f["tau2"] * float(_logu(rng, *LEAD_RATIO))
            elif kind == "res":
                f["w"] = float(min(_logu(rng, *OMEGA) / self.rs, OMEGA[1]))
                f["zeta"] = float(_logu(rng, *ZETA))
            elif kind == "relay":
                f["tau"] = self._tclamp(_logu(rng, *RELAY_TAU))
                hi = float(rng.uniform(*RELAY_HI))
                f["hi"], f["lo"] = hi, hi - float(rng.uniform(*RELAY_W))
            filt.append(f)
        lin = [i for i, f in enumerate(filt) if f["kind"] != "relay"]
        rel = [i for i, f in enumerate(filt) if f["kind"] == "relay"]
        # the linear filters as one block state-space system (continuous)
        blocks = []
        for i in lin:
            f = filt[i]
            if f["kind"] == "direct":
                blocks.append((np.zeros((0, 0)), np.zeros(0), np.zeros(0), 1.0))
            elif f["kind"] == "low":
                t = f["tau"]
                blocks.append((np.array([[-1 / t]]), np.array([1 / t]),
                               np.array([1.0]), 0.0))
            elif f["kind"] == "lead":
                # H = (tau2/tau1)(tau1 s + 1)/(tau2 s + 1) = 1 + (tau2/tau1
                # - 1)/(tau2 s + 1): high-frequency gain 1 (the peak), DC
                # gain tau2/tau1 >= 0.1 (D9.8 item 5)
                t1, t2 = f["tau1"], f["tau2"]
                blocks.append((np.array([[-1 / t2]]), np.array([1 / t2]),
                               np.array([t2 / t1 - 1.0]), 1.0))
            else:
                w, z = f["w"], f["zeta"]
                peak = 1.0 / (2 * z * np.sqrt(1 - z * z)) if z < 0.7071 \
                    else 1.0
                blocks.append((np.array([[0.0, 1.0], [-w * w, -2 * z * w]]),
                               np.array([0.0, w * w]),
                               np.array([1.0 / peak, 0.0]), 0.0))
        n = sum(b[0].shape[0] for b in blocks)
        A, Bm = np.zeros((n, n)), np.zeros((n, len(lin)))
        C, D = np.zeros((len(lin), n)), np.zeros(len(lin))
        o = 0
        for j, (a, b, c, d) in enumerate(blocks):
            k = a.shape[0]
            A[o:o + k, o:o + k], Bm[o:o + k, j], C[j, o:o + k] = a, b, c
            D[j] = d
            o += k
        Sx = -np.linalg.solve(A, Bm) if n else np.zeros((0, len(lin)))
        return dict(K=K, list=filt, lin=np.array(lin, int),
                    rel=np.array(rel, int),
                    src=np.array([f["src"] for f in filt], int),
                    delay=np.array([f["delay"] for f in filt]),
                    A=A, Bm=Bm, C=C, D=D, Sx=Sx, n=n, disc={},
                    rel_tau=np.array([filt[i]["tau"] for i in rel]),
                    # thresholds in library-std units; made absolute (and
                    # y_mu / y_sd set) by _lib_filters
                    rel_hi_u=np.array([filt[i]["hi"] for i in rel]),
                    rel_lo_u=np.array([filt[i]["lo"] for i in rel]),
                    rel_hi=np.zeros(len(rel)), rel_lo=np.zeros(len(rel)),
                    y_mu=np.zeros(len(lin)), y_sd=np.ones(len(lin)))

    def _lib_filters(self, Wc, fl, Xl):
        """Run one force's delay lines, linear filters and relay low-passes
        on the library (exactly as _force_out does, dt = DT_SUB), then set
        fl['y_mu'], fl['y_sd'] (each linear filter output standardised on
        the library, from LIB_SKIP control steps on) and the relays'
        absolute thresholds (library mean + unit threshold x library std of
        the low-passed relay input). Returns the standardised library
        filter outputs (N, K) from LIB_SKIP on (relays +-1), the readout's
        reference cloud."""
        c = Xl @ Wc                                        # (R, T, n_p)
        Rn, T = c.shape[:2]
        Ad, Bd, d, ar = self._fdisc(fl, DT_SUB)
        tix = np.maximum(np.arange(T)[:, None] - d[None], 0)   # (T, K)
        uK = c[:, tix, fl["src"][None]]                    # (R, T, K)
        lin, rel = fl["lin"], fl["rel"]
        Y = np.empty((Rn, T, fl["K"]))
        if len(lin):
            ul = uK[:, :, lin]
            if fl["n"]:
                x = ul[:, 0] @ fl["Sx"].T
                for t in range(T):
                    Y[:, t, lin] = x @ fl["C"].T + ul[:, t] * fl["D"]
                    x = x @ Ad.T + ul[:, t] @ Bd.T
            else:
                Y[:, :, lin] = ul * fl["D"]
        s0 = LIB_SKIP * int(round(input_library()["sub"]))
        if len(lin):
            mu = Y[:, s0:, lin].mean((0, 1))
            sd = np.maximum(Y[:, s0:, lin].std((0, 1)), SD_FLOOR)
            fl["y_mu"], fl["y_sd"] = mu, sd
            Y[:, :, lin] = (Y[:, :, lin] - mu) / sd
        if len(rel):
            ur = uK[:, :, rel]
            XR = np.empty_like(ur)
            xr = ur[:, 0].copy()
            for t in range(T):
                XR[:, t] = xr                  # the state the output reads
                xr = ar * xr + (1 - ar) * ur[:, t]
            mu = XR[:, s0:].mean((0, 1))
            sd = np.maximum(XR[:, s0:].std((0, 1)), SD_FLOOR)
            fl["rel_hi"] = mu + fl["rel_hi_u"] * sd
            fl["rel_lo"] = mu + fl["rel_lo_u"] * sd
            h = np.where(XR[:, 0] > fl["rel_hi"], 1.0, -1.0)
            for t in range(T):
                Y[:, t, rel] = h
                if t + 1 < T:
                    h = np.where(XR[:, t + 1] > fl["rel_hi"], 1.0,
                                 np.where(XR[:, t + 1] < fl["rel_lo"], -1.0,
                                          h))
        return Y[:, s0:].reshape(-1, fl["K"])

    def _draw_readout(self, rng, K, n_out, Zlib=None):
        """A random small network on the K filter outputs (D9.8 item 4):
        biases through the reference cloud (N_MC samples of this force's
        standardised filter outputs on the library, Zlib; N(0, I) when
        None), one nonlinearity scale g, a linear share beta ~ U[0, 1], an
        optional pair product, RMS normalisation on the same cloud without
        centring, an explicit bias share."""
        act = ACTS[int(rng.integers(len(ACTS)))]
        g = float(_logu(rng, *GAIN))
        if Zlib is None:
            Z = rng.normal(0, 1, (N_MC, K))
        else:
            Z = Zlib[rng.integers(0, len(Zlib), N_MC)]
        layers, h = [], Z
        nl = 2 if rng.random() < P_TWO_LAYERS else 1
        for li in range(nl):
            nin = h.shape[1]
            H = int(round(float(_logu(rng, *WIDTH))))
            W = rng.normal(0, 1, (nin, H)) / np.sqrt(nin)
            Xs = h[rng.integers(0, N_MC, H)]            # (H, nin) samples
            b = (Xs * W.T).sum(1)
            h = _act(act, g * (h @ W - b))
            mu, sd = h.mean(0), np.maximum(h.std(0), 1e-9)
            layers.append(dict(W=W, b=b, mu=mu, sd=sd))
            if li < nl - 1:                 # a hidden layer feeds the next
                h = (h - mu) / sd
        # the last layer is not standardised (keeps its mean: a bias)
        V = rng.normal(0, 1, (h.shape[1], n_out)) / np.sqrt(h.shape[1])
        net = h @ V
        pair = None
        if rng.random() < P_PAIR and K >= 1:
            a, b = rng.integers(0, K, 2)
            rho = float(rng.uniform(0, 1))
            vp = rng.normal(0, 1, n_out)
            s0 = np.sqrt((net ** 2).mean(0)) + 1e-12
            net = np.sqrt(1 - rho) * net / s0 + np.sqrt(rho) * vp \
                * (Z[:, a] * Z[:, b])[:, None]
            pair = dict(a=int(a), b=int(b), rho=rho, vp=vp, s0=s0)
        rms = np.sqrt((net ** 2).mean(0)) + 1e-12
        Wl = rng.normal(0, 1, (K, n_out))
        Wl = Wl / np.maximum(np.sqrt((Wl ** 2).sum(0)), 1e-12)
        gam, sgn = 0.0, np.zeros(n_out)
        if rng.random() < P_BIAS:
            gam = float(rng.uniform(*BIAS_SHARE))
            sgn = rng.choice([-1.0, 1.0], n_out)
        ro = dict(act=act, g=g, layers=layers, V=V, pair=pair, rms=rms,
                  Wl=Wl, beta=float(rng.uniform(0, 1)), gamma=gam, sgn=sgn)
        ro["diag"] = readout_diag(ro, Z)
        return ro

    def _J(self, x, y, z):
        """Generalised force per unit body force components (5, 3), heave
        row the body z force (the -theta Fx term is added at run time)."""
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-y, x, 0.0],
                         [0.0, 0.0, 1.0], [z, 0.0, -x]])

    def _cap(self, Qs, fx):
        """A = kappa min_c a_ref,c / sum_o |M_t^-1 Q_o|_c (+ the worst-case
        -theta Fx heave term, theta up to TH_MAX): per unit amplitude."""
        g = sum(np.abs(self.Minv @ q) for q in Qs)
        g = g + TH_MAX * np.abs(fx).sum() * np.abs(self.Minv[:, 3])
        return float(np.min(self.a_ref / np.maximum(g, 1e-12)))

    def _draw_force(self, rng, kappa, Xl):
        kind = _choice(rng, P_KIND)
        names = list(self.groups)
        sel = [g for g in names if rng.random() < P_GROUP]
        if not sel:
            sel = [names[int(rng.integers(len(names)))]]
        n_p = min(1 + int(rng.poisson(N_COMB_MEAN)), N_COMB_MAX)
        Wc = np.zeros((self.n_in, n_p))
        for j in range(n_p):
            for _ in range(int(rng.integers(1, 4))):
                cols = self.groups[sel[int(rng.integers(len(sel)))]]
                Wc[cols[int(rng.integers(len(cols)))], j] += rng.normal()
            Wc[:, j] /= max(np.sqrt((Wc[:, j] ** 2).sum()), 1e-12)
        filt = self._draw_filters(rng, n_p)
        Zlib = self._lib_filters(Wc, filt, Xl)
        f = dict(kind=kind, kappa=float(kappa), groups=sel, Wc=Wc, filt=filt)
        if kind == "point":
            f["pt"] = np.array([rng.uniform(*self.x_box),
                                rng.uniform(-0.5 * self.B, 0.5 * self.B),
                                rng.uniform(*self.z_box)])
            f["dir"] = _choice(rng, P_POINT_DIR)
            J = self._J(*f["pt"])
            if f["dir"] == "body":
                d = rng.normal(0, 1, 3)
                f["d"] = d / max(np.sqrt((d * d).sum()), 1e-12)
                n_out, A = 1, self._cap([J @ f["d"]], [f["d"][0]])
            elif f["dir"] == "earth":
                f["psi_d"] = float(rng.uniform(0, 2 * np.pi))
                n_out, A = 1, self._cap([J[:, 0], J[:, 1]], [1.0])
            else:
                n_out, A = 3, self._cap([J[:, 0], J[:, 1], J[:, 2]], [1.0])
        elif kind == "dist":
            nb = int(rng.integers(2, 4))
            yd = float(rng.uniform(-0.5 * self.B, 0.5 * self.B))
            zd = float(rng.uniform(*self.z_box))
            d = rng.normal(0, 1, 3)
            d = d / max(np.sqrt((d * d).sum()), 1e-12)
            xq = np.linspace(*self.x_box, 41)
            xh = np.linspace(-1, 1, 41)
            P = np.stack([np.ones(41), xh, 1.5 * xh * xh - 0.5])[:nb]
            Qk = np.stack([np.mean([P[k, i] * (self._J(xq[i], yd, zd) @ d)
                                    for i in range(41)], 0)
                           for k in range(nb)])        # (nb, 5)
            Fxk = P.mean(1) * d[0]                     # body x force per unit
            f.update(Qk=Qk, Fxk=Fxk, d=d, yd=yd, zd=zd)
            n_out, A = nb, self._cap(list(Qk), Fxk)
        else:
            k = int(rng.integers(1, 6))
            G = rng.normal(0, 1, (5, k)) * _logu(rng, 0.05, 1.0, 5)[:, None]
            G = G / max(np.abs(G).sum(1).max(), 1e-12)   # max row L1 = 1
            f["Ga"] = kappa * self.a_ref[:, None] * G    # accelerations
            n_out, A = k, 1.0
        f["A"] = float(kappa * A) if kind != "gen" else 1.0
        f["n_out"] = n_out
        f["ro"] = self._draw_readout(rng, filt["K"], n_out, Zlib)
        return f

    def forces_off_now(self):
        """After N_REJECT rejections: no forces (the Coriolis difference and
        the noise stay); flagged in the style (D9.8 item 8)."""
        self.forces, self.forces_off = [], True
        self.style = self._style()

    # ------------------------------------------------------------- state
    def new_state(self, B=1, rng=None, init_s=None, hid_seed=0, forces=True,
                  diag=False):
        """Fresh state for B rows: the old noise part's (rng pre-rolls it),
        the hidden processes at a stationary draw (their own stream
        [seed, 43, attempt, hid_seed]) and the forces' filters / delay lines
        (filled from the first inputs at the first substep). forces=False:
        the hidden processes only (the library run). diag=True: every
        force also logs its standardised filter outputs per substep
        (st['fx'][i]['ylog'], a list of (B, K); for previews)."""
        st = dict(k=0, init=False)
        if forces and self.res is not None:
            st["res"] = self.res.new_state(B, rng=rng, init_s=init_s)
            st["k"] = st["res"]["k"]
        hr = np.random.default_rng([self.seed, HID_STREAM, self.attempt,
                                    int(hid_seed)])
        hid = np.zeros((B, len(self.hidden)))
        for i, h in enumerate(self.hidden):
            if h["kind"] in ("ou", "drift"):
                hid[:, i] = hr.normal(0, 1, B)
            elif h["kind"] == "jump":
                hid[:, i] = np.clip(hr.standard_t(h["df"], B), -HID_CAP,
                                    HID_CAP)
            else:
                hid[:, i] = hr.choice([-1.0, 1.0], B)
        st["hid"], st["rng_hid"] = hid, hr
        if not forces:
            return st
        Dm = int(round(DELAY_S[1] / DT_SUB)) + 1
        st["fx"] = [dict(x=np.zeros((B, f["filt"]["n"])),
                         hist=np.zeros((B, Dm, f["Wc"].shape[1])), ptr=0,
                         xr=np.zeros((B, len(f["filt"]["rel"]))),
                         h=np.zeros((B, len(f["filt"]["rel"]))),
                         **(dict(ylog=[]) if diag else {}))
                    for f in self.forces]
        return st

    @staticmethod
    def snapshot(st):
        """A deep copy of the full state (filters, delay lines, relays,
        hidden processes and their generator, the noise part): a branch
        from this moment starts from it."""
        return copy.deepcopy(st)

    def slow_state(self, st):
        """The old noise part's slow state (flat, as operators.Operator) plus
        the hidden processes' values under 'hid'."""
        out = self.res.slow_state(st["res"]) if self.res is not None \
            else dict(k=st["k"])
        out["hid"] = st["hid"].copy()
        return out

    def set_slow(self, st, slow):
        """Restore a slow state (row 0 broadcast); st['k'] follows."""
        if self.res is not None:
            R._set_slow_flat(st["res"], slow)
            st["k"] = st["res"]["k"]
        else:
            st["k"] = slow["k"]
        if "hid" in slow and st["hid"].size:
            st["hid"][:] = np.asarray(slow["hid"])[0]

    def run(self, *args, **kwargs):
        raise NotImplementedError(
            "OperatorGen.run: the forces are state feedback at every substep "
            "and cannot be relabelled on recorded inputs; use simulate_gen "
            "(closed loop), or run_residual for the noise part alone "
            "(PRIOR_DERIVATION.md D9.8)")

    def run_residual(self, S_raw, slow=None, noise=False, rng=None, hold=0):
        """The control-step part alone (noise; the rule is 0) on recorded
        inputs (B, T, 28): rule (B, T, 5) [, noise]."""
        S_raw = np.asarray(S_raw, float)
        B, T, _ = S_raw.shape
        st = self.new_state(B, init_s=S_raw[:, 0])
        if slow is not None:
            self.set_slow(st, slow)
        Rr, N = np.zeros((B, T, 5)), np.zeros((B, T, 5))
        for k in range(T):
            Rr[:, k], N[:, k] = self.step(st, S_raw[:, k], noise=noise,
                                          rng=rng, freeze=slow is not None)
        return (Rr, N) if noise else Rr

    # ------------------------------------------------ control-step part
    def step(self, st, s_raw, noise=True, rng=None, freeze=False,
             hold_relays=False):
        """One control step: rule 0 (no held rule in this family), noise =
        w_noise x the old family's noise part (operators.Operator._noise:
        coloured / white noise with its state envelope, bursts)."""
        s_raw = np.atleast_2d(np.asarray(s_raw, float))
        B = s_raw.shape[0]
        rule, nz = np.zeros((B, 5)), np.zeros((B, 5))
        if self.res is None:
            st["k"] = st["k"] + 1
            return rule, nz
        op, rs = self.res, st["res"]
        if noise:
            z = (op._ext(s_raw) - op.mu) / op.sd
            a = np.exp(rs["ga"])[:, None]
            nz = self.w_noise * op._noise(rs, z, a, B, rng)
        if not freeze:
            op._slow(rs, B, rng)
        rs["k"] = rs["k"] + 1
        st["k"] = rs["k"]
        return rule, nz

    # ------------------------------------------------------ substep part
    def raw_inputs(self, sr, thr, noz, nu0, eta):
        """The N_RAW physical input columns (B, 28), before standardisation:
        state (u, v, r, zdot, thdot, theta), actual actuators (thrust,
        nozzle), nu0 (5), local immersion at the 5 x 3 stations (surface
        minus hull point, eta - z - s_p x theta)."""
        sr = np.atleast_2d(sr)
        Bn = len(sr)
        x, y, u, z, zd, th, thd, psi, r, v = sr.T
        X = np.empty((Bn, N_RAW))
        X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5] = \
            u, v, r, zd, thd, th
        X[:, 6] = np.broadcast_to(thr, (Bn,))
        X[:, 7] = np.broadcast_to(noz, (Bn,))
        X[:, 8:13] = np.asarray(nu0, float).reshape(Bn, 5)
        e15 = np.asarray(eta, float).reshape(Bn, 15)
        X[:, 13:28] = e15 - z[:, None] - self.sp * self.xs15[None] \
            * th[:, None]
        return X

    def inputs(self, sr, thr, noz, nu0, eta, hid):
        """The force inputs (B, n_in) (D9.8 item 1): the 28 physical columns
        (raw_inputs) standardised by their mean and std on the low-fidelity-
        only library (input_library), the hidden processes (unit
        stationary std by construction), the heading relative to the
        earth-fixed direction (cos, sin)."""
        sr = np.atleast_2d(sr)
        Bn = len(sr)
        psi = sr[:, 7]
        L = input_library()
        X = np.empty((Bn, self.n_in))
        X[:, :N_RAW] = (self.raw_inputs(sr, thr, noz, nu0, eta) - L["mu"]) \
            / L["sd"]
        m = len(self.hidden)
        if m:
            X[:, 28:28 + m] = hid
        X[:, 28 + m] = np.cos(psi - self.psi_w)
        X[:, 29 + m] = np.sin(psi - self.psi_w)
        return X

    def _fdisc(self, fl, dt):
        key = round(float(dt), 12)
        hit = fl["disc"].get(key)
        if hit is None:
            n, k = fl["n"], len(fl["lin"])
            if n:
                Mx = np.zeros((n + k, n + k))
                Mx[:n, :n], Mx[:n, n:] = fl["A"], fl["Bm"]
                E = expm(Mx * dt)
                Ad, Bd = E[:n, :n].copy(), E[:n, n:].copy()
            else:
                Ad, Bd = np.zeros((0, 0)), np.zeros((0, k))
            d = np.clip(np.round(fl["delay"] / dt), 0,
                        int(round(DELAY_S[1] / DT_SUB))).astype(int)
            ar = np.exp(-dt / fl["rel_tau"]) if len(fl["rel"]) else \
                np.zeros(0)
            hit = (Ad, Bd, d, ar)
            fl["disc"][key] = hit
        return hit

    @staticmethod
    def readout(ro, yb):
        """The readout network on filter outputs yb (B, K) -> (B, n_out),
        soft-saturated at SAT."""
        h = yb
        nl = len(ro["layers"])
        for li, Ly in enumerate(ro["layers"]):
            h = _act(ro["act"], ro["g"] * (h @ Ly["W"] - Ly["b"]))
            if li < nl - 1:
                h = (h - Ly["mu"]) / Ly["sd"]
        net = h @ ro["V"]
        pr = ro["pair"]
        if pr is not None:
            net = np.sqrt(1 - pr["rho"]) * net / pr["s0"] + np.sqrt(
                pr["rho"]) * pr["vp"] * (yb[:, pr["a"]] * yb[:, pr["b"]])[:,
                                                                         None]
        net = net / ro["rms"]
        y = np.sqrt(ro["beta"]) * (yb @ ro["Wl"]) + np.sqrt(
            1 - ro["beta"]) * net
        y = np.sqrt(1 - ro["gamma"]) * y + np.sqrt(ro["gamma"]) * ro["sgn"]
        return SAT * np.tanh(y / SAT)

    def _force_out(self, f, fs, X, dt, init):
        """One force's readout (B, n_out) at this substep; advances its
        delay line, filters and relays."""
        fl = f["filt"]
        Ad, Bd, d, ar = self._fdisc(fl, dt)
        c = X @ f["Wc"]                                   # (B, n_p)
        Dm = fs["hist"].shape[1]
        if init:
            fs["hist"][:] = c[:, None, :]
            fs["ptr"] = 0
        else:
            fs["ptr"] = (fs["ptr"] + 1) % Dm
            fs["hist"][:, fs["ptr"]] = c
        uK = fs["hist"][:, (fs["ptr"] - d) % Dm, fl["src"]]   # (B, K)
        yb = np.empty_like(uK)
        lin, rel = fl["lin"], fl["rel"]
        if len(lin):
            ul = uK[:, lin]
            if init and fl["n"]:
                fs["x"] = ul @ fl["Sx"].T                 # steady state
            yb[:, lin] = (fs["x"] @ fl["C"].T + ul * fl["D"]
                          - fl["y_mu"]) / fl["y_sd"]     # library-standardised
            if fl["n"]:
                fs["x"] = fs["x"] @ Ad.T + ul @ Bd.T
        if len(rel):
            ur = uK[:, rel]
            if init:
                fs["xr"] = ur.copy()
                fs["h"] = np.where(ur > fl["rel_hi"], 1.0, -1.0)
            yb[:, rel] = fs["h"]
            fs["xr"] = ar * fs["xr"] + (1 - ar) * ur
            fs["h"] = np.where(fs["xr"] > fl["rel_hi"], 1.0,
                               np.where(fs["xr"] < fl["rel_lo"], -1.0,
                                        fs["h"]))
        if "ylog" in fs:
            fs["ylog"].append(yb.copy())
        return self.readout(f["ro"], yb)

    def point_accel(self, pt, F, th):
        """Accelerations (B, 5) of body forces F (B, 3) at hull point pt
        (x, y, z from the reference point): Q = (Fx, Fy, x Fy - y Fx,
        Fz - theta Fx (earth-vertical heave), z Fx - x Fz), then M_t^-1 Q."""
        return self._point_Q(pt, F, th) @ self.Minv.T

    @staticmethod
    def _point_Q(pt, F, th):
        x, y, z = pt
        return np.stack([F[:, 0], F[:, 1], x * F[:, 1] - y * F[:, 0],
                         F[:, 2] - th * F[:, 0], z * F[:, 0] - x * F[:, 2]], 1)

    def _force_Q(self, f, y, th, psi):
        A = f["A"]
        if f["kind"] == "point":
            if f["dir"] == "body":
                F = A * y[:, :1] * f["d"][None]
            elif f["dir"] == "earth":
                a = f["psi_d"] - psi
                F = A * y[:, :1] * np.stack([np.cos(a), np.sin(a),
                                             np.zeros_like(a)], 1)
            else:
                F = A * y
            return self._point_Q(f["pt"], F, th)
        Q = A * (y @ f["Qk"])
        Q[:, 3] -= th * A * (y @ f["Fxk"])
        return Q

    def coriolis(self, sr):
        """The rigid-body Coriolis / centripetal difference (D9.1 (d), D9.8
        item 9) as accelerations (B, 5): M_t^-1 (C_0 nu - C_RB(nu) nu), with
        w = zdot + u theta, q = thdot, CG offset and I_xz; the heave row in
        the earth-vertical coordinate (m (omega x v)_z cancels there); the
        low-fidelity boat's own term is m_coriolis u r in sway."""
        sr = np.atleast_2d(sr)
        u, zd, th, q, r, v = sr[:, 2], sr[:, 4], sr[:, 5], sr[:, 6], \
            sr[:, 8], sr[:, 9]
        w = zd + u * th
        m, xg, yg, zg, Ixz = self.m, self.xg, self.yg, self.zg, self.Ixz
        c1 = q * w - r * v
        C = np.stack([
            m * (c1 - (q * q + r * r) * xg),
            m * (r * u + r * q * zg - r * r * yg) - self.m0 * u * r,
            Ixz * q * r + m * (xg * r * u - yg * c1),
            m * (-q * q * zg + q * r * yg),
            -Ixz * r * r + m * (zg * c1 + xg * q * u)], 1)
        return -C @ self.Minv.T

    def _advance_hidden(self, st, dt):
        hid, rng = st["hid"], st["rng_hid"]
        Bn = hid.shape[0]
        for i, h in enumerate(self.hidden):
            k = h["kind"]
            if k in ("ou", "drift"):
                a = np.exp(-dt / h["tau"])
                hid[:, i] = a * hid[:, i] + np.sqrt(1 - a * a) \
                    * rng.normal(0, 1, Bn)
            elif k == "jump":
                hit = rng.random(Bn) < -np.expm1(-h["rate"] * dt)
                new = np.clip(rng.standard_t(h["df"], Bn), -HID_CAP, HID_CAP)
                hid[:, i] = np.where(hit, new, hid[:, i])
            else:
                hit = rng.random(Bn) < -np.expm1(-dt / h["dwell"])
                hid[:, i] = np.where(hit, -hid[:, i], hid[:, i])

    def gen_accel(self, st, sr, thr, noz, nu0, eta, dt, freeze=False,
                  parts=False):
        """THE substep step, batched over the B rows of st. sr (B, 10) the
        state at the START of the substep; thr, noz the applied actuator
        positions over it; nu0 (B, 5) the low-fidelity boat's own
        acceleration over it (before any injection); eta (B, 5, 3) the
        surface at the stations at its start. Advances filters, delay
        lines, relays and (unless freeze) the hidden processes. Returns
        the acceleration (B, 5), or with parts=True a dict {force index:
        (B, 5), 'cor': (B, 5)}."""
        sr = np.atleast_2d(np.asarray(sr, float))
        X = self.inputs(sr, thr, noz, nu0, eta, st["hid"])
        th, psi = sr[:, 5], sr[:, 7]
        init = not st["init"]
        Qm = np.zeros((len(sr), 5))
        acc = np.zeros((len(sr), 5))
        out = {}
        for i, f in enumerate(self.forces):
            y = self._force_out(f, st["fx"][i], X, dt, init)
            if f["kind"] == "gen":
                a = y @ f["Ga"].T
                acc += a
            else:
                Q = self._force_Q(f, y, th, psi)
                Qm += Q
                a = Q @ self.Minv.T if parts else None
            if parts:
                out[i] = a
        cor = self.coriolis(sr)
        st["init"] = True
        if not freeze and self.hidden:
            self._advance_hidden(st, dt)
        if parts:
            out["cor"] = cor
            return out
        return Qm @ self.Minv.T + acc + cor

    def bow_height(self, sr, eta):
        """Bow height above the water (B,): F_b + z + s_p x_b theta - the
        highest of the three bow-station elevations (D9.8 item 10)."""
        sr = np.atleast_2d(sr)
        e = np.asarray(eta, float).reshape(len(sr), 5, 3)
        return self.F_b + sr[:, 3] + self.sp * self.x_b * sr[:, 5] \
            - e[:, 4, :].max(1)

    # ------------------------------------------------------------ style
    def _style(self):
        fs = self.forces
        return dict(
            op_family="gen", attempt=int(self.attempt),
            n_reject=int(self.n_reject), forces_off=bool(self.forces_off),
            w_noise=self.w_noise, kappa0=float(self.kappa0),
            n_forces=len(fs), kinds=[f["kind"] for f in fs],
            kappas=[f["kappa"] for f in fs],
            groups=[list(f["groups"]) for f in fs],
            filters=[[g["kind"] for g in f["filt"]["list"]] for f in fs],
            beta=[f["ro"]["beta"] for f in fs],
            acts=[f["ro"]["act"] for f in fs],
            gain=[f["ro"]["g"] for f in fs],
            # the readout on its library cloud: std / RMS (small = nearly
            # constant) and the deviation from its linearisation at the
            # cloud mean / std (small = nearly linear)
            lib_std_rms=[f["ro"]["diag"]["std_rms"] for f in fs],
            lib_nonlin=[f["ro"]["diag"]["nonlin"] for f in fs],
            reject_reasons=[r["reason"] for r in self.reject_log],
            test_clip=float(getattr(self, "test_clip", np.nan)),
            hidden=[h["kind"] for h in self.hidden],
            psi_w=self.psi_w, **{k: float(v) for k, v in
                                 self.inertia.items()})


def readout_diag(ro, Z, eps=1e-4):
    """Descriptive numbers of a readout on a cloud Z (N, K) of its inputs:
    std_rms = mean over outputs of std / RMS of the output (the part that
    varies vs the constant part), nonlin = mean over outputs of std(y -
    linearisation at the cloud mean) / std(y) (central differences)."""
    y = OperatorGen.readout(ro, Z)
    zb = Z.mean(0, keepdims=True)
    y0 = OperatorGen.readout(ro, zb)
    K = Z.shape[1]
    J = np.zeros((K, y.shape[1]))
    for k in range(K):
        dz = np.zeros((1, K))
        dz[0, k] = eps
        J[k] = (OperatorGen.readout(ro, zb + dz)
                - OperatorGen.readout(ro, zb - dz))[0] / (2 * eps)
    dev = y - (y0 + (Z - zb) @ J)
    sd = y.std(0)
    rms = np.sqrt((y ** 2).mean(0)) + 1e-12
    return dict(std_rms=float((sd / rms).mean()),
                nonlin=float((dev.std(0) / np.maximum(sd, 1e-12)).mean()))


# ------------------------------------------------------ input library
class _Recorder:
    """A do-nothing group for simulate_gen that records the raw input
    columns and the heading at every substep (the library run)."""

    def __init__(self, holder):
        self.h, self.X, self.psi = holder, [], []

    def step(self, st, s_raw, noise=True, rng=None, freeze=False):
        z = np.zeros((len(np.atleast_2d(s_raw)), 5))
        return z, z

    def gen_accel(self, st, sr, thr, noz, nu0, eta, dt, freeze=False,
                  parts=False):
        sr = np.atleast_2d(sr)
        self.X.append(self.h.raw_inputs(sr, thr, noz, nu0, eta))
        self.psi.append(sr[:, 7].copy())
        z = np.zeros((len(sr), 5))
        return {"cor": z} if parts else z


def input_library(env=None):
    """The LOW-FIDELITY-ONLY input library (D9.8 item 1), computed once per
    process (deterministic, fixed seeds): LIB_ROWS rows of the low-fidelity
    boat alone in closed loop for LIB_STEPS control steps (the data
    autopilot with informative rows, training seas R.sea_dict, start at a
    random task speed, ideal actuators). Returns dict(raw (R, T, 28) the
    raw input columns per substep, psi (R, T), mu, sd (28,) over the
    substeps from LIB_SKIP control steps on, sub). No target data."""
    if "lib" in _TEST:
        return _TEST["lib"]
    env = _test_env() if env is None else env
    holder = OperatorGen.__new__(OperatorGen)
    holder.p = dict(env["red"].p)
    holder.geom = dict(GEOM)
    holder._consts(U_RANGE)
    rng = np.random.default_rng(LIB_SEED)
    B = LIB_ROWS
    xs = R.start_states(env, B, rng)
    seas = R.RowSeas([R.sea_state(R.sea_dict(np.random.default_rng(
        [LIB_SEED, i])), LIB_SEED + i) for i in range(B)], env["x_st"],
        env["y_off"])
    rec = _Recorder(holder)
    simulate_gen(env, xs, R.Autopilot(env, B, rng, informative=True),
                 np.zeros(B), seas, [dict(op=rec, st={}, rows=np.arange(B),
                                          rng=None)], LIB_STEPS, noise=False)
    raw = np.stack(rec.X, 1)
    psi = np.stack(rec.psi, 1)
    s0 = LIB_SKIP * int(env["sub"])
    flat = raw[:, s0:].reshape(-1, N_RAW)
    if not np.isfinite(flat).all():
        raise RuntimeError("input_library: non-finite low-fidelity run")
    _TEST["lib"] = dict(raw=raw, psi=psi, mu=flat.mean(0),
                        sd=np.maximum(flat.std(0), SD_FLOOR),
                        sub=int(env["sub"]))
    return _TEST["lib"]


# ------------------------------------------------------------ safety
def safety_per_step(az, hs, sub=6):
    """Per control step from per-substep series: az (B, n sub) the vertical
    accelerations after injection, hs (B, n sub + 1) the bow heights at the
    substep STARTS (the last one after the final substep). Returns APK,
    AMIN (max / min over the step's substeps) and HMIN (min over the
    substep ends j = 1..6, i.e. starts 1..6 of the chain), each (B, n)."""
    az, hs = np.atleast_2d(az), np.atleast_2d(hs)
    B, ns = az.shape
    n = ns // sub
    a = az[:, :n * sub].reshape(B, n, sub)
    h = np.stack([hs[:, k * sub + 1:(k + 1) * sub + 1] for k in range(n)], 1)
    return a.max(-1), a.min(-1), h.min(-1)


# ------------------------------------------------------------------ plant
def gen_plant(plant):
    """An RBPlant copy of a sim.lofi.ReducedPlant (operators_rb.rb_plant):
    it keeps the substep's start state, applied actuators and station
    elevations, which is all GenInjector needs."""
    return R.rb_plant(plant)


GenPlant = R.RBPlant


class GenInjector:
    """Mission's residual hook for the general family (data2.ZOHInjector /
    operators_rb.RBInjector interface). Set rule / noise (or r) every control
    step; at every substep, after the plant's own step, it reads the start
    state, applied actuators and elevations off the RBPlant, takes nu0 as
    the plant's own velocity increment over the substep, adds the forces,
    clips as clip_push does and kicks the velocities. It logs per substep
    the vertical acceleration after injection (az) and the bow height at
    the substep start (h) for the safety quantities (safety_per_step)."""

    def __init__(self, op, st, freeze=False):
        self.op, self.st, self.freeze = op, st, bool(freeze)
        self.rule, self.noise = np.zeros(5), np.zeros(5)
        self.last = np.zeros(5)
        self._acc, self._n = np.zeros(5), 0
        self.az, self.h = [], []

    @property
    def r(self):
        return self._acc / self._n if self._n else self.rule + self.noise

    @r.setter
    def r(self, value):
        self.rule = np.asarray(value, float).copy()
        self.noise = np.zeros(5)
        self._acc, self._n = np.zeros(5), 0

    def __call__(self, s, dt, plant=None):
        sr, thr, noz, eta = plant.model.last
        sr = np.atleast_2d(sr)
        post = R.to_reduced(np.asarray(s, float)[None])
        nu0 = (post[:, list(RED_VEL)] - sr[:, list(RED_VEL)]) / dt
        a = self.op.gen_accel(self.st, sr, np.array([thr]), np.array([noz]),
                              nu0, eta, dt, freeze=self.freeze)[0]
        rc, nc = clip_push(a + self.rule, self.noise)
        push = rc + nc
        s = s.copy()
        for c, i in enumerate(VEL_IDX):
            s[i] += push[c] * dt
        plant.last_cg_acc = float(plant.last_cg_acc + push[3])
        self.h.append(float(self.op.bow_height(sr, eta)[0]))
        self.az.append(plant.last_cg_acc)
        self.last = push
        self._acc, self._n = self._acc + push, self._n + 1
        return s


# --------------------------------------------------------- batched world
def simulate_gen(env, xs14, U, t0, seas, groups, n_steps, noise=True,
                 freeze=False, waves="mid", act=None, parts=False,
                 substeps=False, fb=None):
    """Closed-loop batched rollout with the general family (the simulate_rb
    pattern). Per substep: actuators, the surface at the stations at the
    substep start, the low-fidelity step, nu0 = its velocity increment / dt,
    the forces from the START state (gen_accel), clip_push with the held
    control-step part, the velocity kick. Rows in no group get no push.

    xs14 (B, 14), U (B, n, 2) or callable U(step, xs14_now), t0 (B,),
    seas an operators_rb.RowSeas, groups [dict(op, st, rows, rng)], act
    None (ideal) or stacked lofi.draw_act rows (the M10 chain). A row whose
    state leaves the envelope (non-finite, u > 1.5 u_max, |theta| > 45 deg)
    at the end of a control step is frozen at that step's start and
    flagged: div (B,) = the step, -1 if never (truncate, never redraw).

    Returns dict(XS (B, n+1, 14), U, P (B, n, 5) mean applied push, clip
    (B, n, 5) substeps with the total clipped, clip_rule (B, n) substeps
    with any channel of the rule part clipped, RULE / NOISE held parts,
    APK / AMIN / HMIN (B, n) the safety quantities, E15 (B, n, 15) the
    station elevations at each step start, div); parts=True adds PARTS
    [per group {force index | 'cor': (B_g, n, 5)}]; substeps=True adds
    SUB (B, n, sub, 10) reduced states after each substep (after the kick),
    PS (B, n, sub, 5) the pushes, AZ (B, n sub), HS (B, n sub + 1)."""
    red, dt, sub = env["red"], env["dt"], env["sub"]
    dtc = dt * sub
    lim = CLIP * A_REF
    p = red.p
    fb = freeboard(p) if fb is None else float(fb)
    x_b, s_p = float(env["x_st"][-1]), float(p["sign_pitch"])
    u_max = float(p["u_max"])
    B = len(xs14)
    rv = list(RED_VEL)
    sr = R.to_reduced(xs14)
    thr = np.asarray(xs14, float)[:, 12].copy()
    noz = np.asarray(xs14, float)[:, 13].copy()
    t0 = np.broadcast_to(np.asarray(t0, float), (B,)).copy()
    XS = np.zeros((B, n_steps + 1, 14))
    UU = np.zeros((B, n_steps, 2))
    P, CL = np.zeros((B, n_steps, 5)), np.zeros((B, n_steps, 5))
    CLR = np.zeros((B, n_steps))
    RU, NZ = np.zeros((B, n_steps, 5)), np.zeros((B, n_steps, 5))
    E15 = np.zeros((B, n_steps, 15))
    AZ = np.zeros((B, n_steps * sub))
    HS = np.zeros((B, n_steps * sub + 1))
    div = np.full(B, -1)
    alive = np.ones(B, bool)
    PT = [{} for _ in groups] if parts else None
    if substeps:
        SUB = np.zeros((B, n_steps, sub, 10))
        PS = np.zeros((B, n_steps, sub, 5))
    XS[:, 0] = R.to14(sr, thr, noz)
    if act is not None:
        from sim import lofi
        A = {k_: np.broadcast_to(np.asarray(act[k_], float), (B,)).copy()
             for k_ in lofi.ACT_KEYS}
        if not (A["act_family"] > 0.5).all():
            raise ValueError("simulate_gen: act rows must be M10 draws")
        q = lofi.act_arrays(A, env["t_max"], env["rud_max"])
        dly = q["delay"]
        pos0 = np.stack([thr, noz], 1)
        vel = np.zeros((B, 2))
        CMD = np.zeros((B, n_steps, 2))
        rows_, cols_ = np.arange(B)[:, None], np.arange(2)[None, :]
    for k in range(n_steps):
        t = t0 + k * dtc
        xs_now = R.to14(sr, thr, noz)
        Uk = np.asarray(U(k, xs_now) if callable(U) else U[:, k], float)
        UU[:, k] = Uk
        if waves == "mid":
            w15 = R.mid_waves(seas, sr, t, 0.5 * dtc)
        else:
            w15 = seas.stations(sr[:, 0], sr[:, 1], sr[:, 7], t)[0].reshape(
                B, 15)
        s28 = R.s28_of(env, xs_now, w15, Uk)
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
        sr_k, thr_k, noz_k = sr.copy(), thr.copy(), noz.copy()
        for kk in range(sub):
            tt = t + kk * dt
            if act is None:
                thr_app, noz_app = cmd_t, cmd_r
            else:
                gi = k * sub + kk - dly
                ci = np.maximum(gi, 0) // sub
                c_d = np.where(gi >= 0, CMD[rows_, ci, cols_], pos0)
                pos, vel, app = lofi.act_step(q, np.stack([thr, noz], 1),
                                              vel, c_d, dt)
                thr_app, noz_app = app[:, 0], app[:, 1]
                thr, noz = pos[:, 0], pos[:, 1]
            eta, _ = seas.stations(sr[:, 0], sr[:, 1], sr[:, 7], tt)
            if kk == 0:
                E15[:, k] = eta.reshape(B, 15)
            HS[:, k * sub + kk] = fb + sr[:, 3] + s_p * x_b * sr[:, 5] \
                - eta[:, 4, :].max(1)
            ns = red.step(sr, thr_app, noz_app, eta, env["x_st"], dt)[0]
            nu0 = (ns[:, rv] - sr[:, rv]) / dt
            push = np.zeros((B, 5))
            for j, g in enumerate(groups):
                rows = g["rows"]
                a = g["op"].gen_accel(g["st"], sr[rows], thr_app[rows],
                                      noz_app[rows], nu0[rows], eta[rows],
                                      dt, freeze=freeze, parts=parts)
                if parts:
                    for key, v in a.items():
                        acc_ = PT[j].setdefault(key, np.zeros(
                            (len(rows), n_steps, 5)))
                        acc_[:, k] += v / sub
                    a = sum(a.values())
                rc, nc = clip_push(a + held_r[rows], held_n[rows])
                push[rows] = rc + nc
                CL[rows, k] += (np.abs(a + held_r[rows] + held_n[rows])
                                > lim)
                CLR[rows, k] += (np.abs(a + held_r[rows]) > lim).any(-1)
            push[~alive] = 0.0
            zd_old = sr[:, 4].copy()
            sr = ns
            sr[:, rv] += push * dt
            AZ[:, k * sub + kk] = (sr[:, 4] - zd_old) / dt
            P[:, k] += push / sub
            if substeps:
                SUB[:, k, kk], PS[:, k, kk] = sr, push
        if act is None:
            thr, noz = np.array(cmd_t, float), np.array(cmd_r, float)
        with np.errstate(invalid="ignore"):
            bad = alive & (~np.isfinite(sr).all(1)
                           | (sr[:, 2] > 1.5 * u_max)
                           | (np.abs(sr[:, 5]) > TH_MAX))
        div[bad] = k
        alive &= ~bad
        dead = ~alive
        if dead.any():
            sr[dead], thr[dead], noz[dead] = sr_k[dead], thr_k[dead], \
                noz_k[dead]
            AZ[dead, k * sub:(k + 1) * sub] = np.nan
        XS[:, k + 1] = R.to14(sr, thr, noz)
    eta, _ = seas.stations(sr[:, 0], sr[:, 1], sr[:, 7], t0 + n_steps * dtc)
    HS[:, -1] = fb + sr[:, 3] + s_p * x_b * sr[:, 5] - eta[:, 4, :].max(1)
    for b in np.where(div >= 0)[0]:
        HS[b, div[b] * sub:] = np.nan
    APK, AMIN, HMIN = safety_per_step(AZ, HS, sub)
    out = dict(XS=XS, U=UU, P=P, clip=CL, clip_rule=CLR, RULE=RU, NOISE=NZ,
               APK=APK, AMIN=AMIN, HMIN=HMIN, E15=E15, div=div)
    if parts:
        out["PARTS"] = PT
    if substeps:
        out.update(SUB=SUB, PS=PS, AZ=AZ, HS=HS)
    return out


# ------------------------------------------------------ acceptance test
_TEST = {}


def _test_env():
    if "env" not in _TEST:
        _TEST["env"] = R.light_env()
    return _TEST["env"]


class _PairPilot:
    """The data autopilot, one identical instance (fixed seed) per draw's
    two test rows, so the test is the same whatever the batch."""

    def __init__(self, env, n):
        self.aps = [R.Autopilot(env, 2, np.random.default_rng(TEST_SEED),
                                informative=True) for _ in range(n)]

    def __call__(self, k, xs):
        return np.concatenate([ap(k, xs[2 * i:2 * i + 2])
                               for i, ap in enumerate(self.aps)])


class _CalmSeaRows:
    """Rows 0, 2, 4, ... calm water (surface 0), rows 1, 3, ... the standard
    sea: only the sea rows are evaluated (the RowSeas API)."""

    def __init__(self, sea, n, x_st, y_off):
        self.base = R.RowSeas([sea] * n, x_st, y_off)
        self.n = n

    def stations(self, x, y, psi, t, rows=None, vel=None):
        B = len(x)
        t = np.broadcast_to(np.asarray(t, float), (B,))
        e, ed = np.zeros((B, 5, 3)), np.zeros((B, 5, 3))
        s = slice(1, None, 2)
        v = None if vel is None else tuple(np.asarray(q)[s] for q in vel)
        e[s], ed[s] = self.base.stations(x[s], y[s], psi[s], t[s],
                                         rows=np.arange(B // 2), vel=v)
        return e, ed


def _test_world(env, n):
    """2 n rows: (calm, standard sea) per draw, start at TEST_SPEED at the
    running attitude with the nominal thrust; the calm rows start
    perturbed (bow a quarter draught up from heave and from pitch, some
    sway and yaw rate) so a self-excited motion shows without waves."""
    p = env["red"].p
    xs = np.zeros((2 * n, 14))
    xs[:, 6] = TEST_SPEED
    xs[:, 2], xs[:, 4] = p.get("z0", 0.0), p.get("th0", 0.0)
    xs[:, 12] = p["k_drag"] * TEST_SPEED ** 2
    T, x_b = float(p["T"]), float(env["x_st"][-1])
    xs[0::2, 2] += 0.25 * T
    xs[0::2, 4] += p["sign_pitch"] * 0.25 * T / x_b
    xs[0::2, 7], xs[0::2, 11] = 0.3, 0.05
    sea = R.sea_state(TEST_SEA, TEST_SEA_SEED)
    return xs, _CalmSeaRows(sea, n, env["x_st"], env["y_off"])


def acceptance_test(ops, env=None):
    """The acceptance test (D9.8 item 2), the same for every draw: per draw
    a perturbed calm-water row and a standard-sea row, 60 s, the data
    autopilot (fixed seed), ideal actuators, no noise part, hidden
    processes from the test stream. The ONLY criterion is divergence
    (non-finite, u > 1.5 u_max, |theta| > 45 deg in either row; simulate_gen
    'div'). Returns [(ok, reason)] per draw, reason 'diverged' or ''. Also
    sets op.test_clip: the share of substeps (from TEST_SKIP on, both rows)
    with the rule part clipped (recorded, never a reason to reject; D9.3
    item 2)."""
    env = _test_env() if env is None else env
    n = len(ops)
    xs, seas = _test_world(env, n)
    groups = [dict(op=o, st=o.new_state(2, hid_seed=TEST_HID),
                   rows=np.array([2 * i, 2 * i + 1]), rng=None)
              for i, o in enumerate(ops)]
    out = simulate_gen(env, xs, _PairPilot(env, n), np.zeros(2 * n), seas,
                       groups, TEST_STEPS, noise=False)
    res = []
    n_sub = 2 * (TEST_STEPS - TEST_SKIP) * env["sub"]
    for i, o in enumerate(ops):
        c, s = 2 * i, 2 * i + 1
        o.test_clip = float(out["clip_rule"][[c, s], TEST_SKIP:].sum()
                            / n_sub)
        if (out["div"][[c, s]] >= 0).any():
            res.append((False, "diverged"))
        else:
            res.append((True, ""))
    return res


def accept_draws(ops, env=None, max_reject=N_REJECT):
    """Run the acceptance test on every not-yet-accepted draw at once and
    redraw the rejected ones from their next attempt stream until all pass
    (a draw's outcome depends only on its own seed, never on the batch).
    After max_reject rejections a draw keeps no forces (forces_off). Each
    rejection is logged in op.reject_log (attempt, reason, draw summary)."""
    pending = [o for o in ops if not o.accepted]
    while pending:
        res = acceptance_test(pending, env)
        nxt = []
        for o, (ok, why) in zip(pending, res):
            if ok:
                o.accepted = True
                o.style = o._style()
                continue
            o.reject_log.append(dict(
                attempt=o.attempt, reason=why, kappa0=o.kappa0,
                n_forces=len(o.forces), kinds=[f["kind"] for f in o.forces],
                hidden=[h["kind"] for h in o.hidden],
                gain_max=max([f["ro"]["g"] for f in o.forces] or [0.0]),
                test_clip=float(getattr(o, "test_clip", np.nan))))
            o.n_reject += 1
            if o.n_reject >= max_reject:
                o.forces_off_now()
                o.accepted = True
            else:
                o._draw(o.attempt + 1)
                nxt.append(o)
        pending = nxt
    return ops


def build_ops(seeds, lib=None, env=None, **kw):
    """OperatorGen draws for many seeds, the acceptance test batched."""
    ops = [OperatorGen(s, lib, accept=False, **kw) for s in seeds]
    accept_draws(ops, env)
    return ops
