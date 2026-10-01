#!/usr/bin/env python3
"""
Can the synthetic operator family (learn/meta/operators.py, the prior the
one-step error model is trained on) EXPRESS the target's error function
(DEFECTS M11-M13)? Drawing operators at random cannot answer this: the
family is a huge function space, and a random draw essentially never lands
on any given function, whether that function is in the family or not
(studies/prior_mass_m13.py measures where the prior's MASS is, not what the
family can express). This study answers it by OPTIMISATION.

WHAT THE FAMILY PRODUCES. model3 is trained on e0 of the source splits
(data3.e0_from on the recorded states): the source plant (the reduced model
with the waves at the MPC's stations, the episode's actuators and the
operator's push) minus model0 (no waves, ideal actuators). The operator
only adds its push, so every training label is

    e0 = W_src + push_response(rule + noise)

W_src is the source plant's own wave (and actuator) response over the step;
push_response (data2 / relabel.push_response_red) maps the push through the
reduced model, mixing the channels by a state-dependent gain (the sway
response to a yaw push is the Coriolis term, growing with speed). So the
question is whether e0_target - W_src is push_response(rule) for some rule.
Here:
- W_src is the source plant replayed one step from every recorded target
  state (relabel.simulate, zero push) in the target episode's own sea
  (relabel._sea; the recorded elevations are checked against it), with
  IDEAL actuators (the family's actuator models are random per episode;
  with ideal ones W_src is the pure wave response), minus model0;
- push_response is linearised per recorded step: J (5 x 5) = the response
  to unit pushes from that state under that command (exact for the heave
  and pitch channels, a secant for the Coriolis cross terms; the recovery
  control measures the linearisation error on real rule + noise pushes).

FamilyFit below is a differentiable SUPERSET of one operator draw's rule
part: the same structure, with every continuous parameter trainable and
every discrete choice widened to a trainable mix. Its output is a PUSH,
mapped through J; it is fitted by gradient descent to e0 - W_src (channels
0-4: surge, sway, yaw rate, heave rate, pitch rate) from the target's own
inputs (data2.op_inputs: the 26 measured signals incl. the 15 wave
elevations, and the 2 commands), run recurrently over whole episodes from
rest.

THE NOISE PART THAT PAST ERRORS PREDICT. The family's noise is not all
unpredictable: its coloured part is a cascade of up to one resonant and one
first-order section (tau up to 30 s), the slow gain and bias drifts and the
per-checkpoint constants persist; an in-context network trained on the
family infers them from the error history, and that is part of what the
family expresses. After the input-only fit, a linear predictor of the
remaining error from its own past (NoisePred: per output and input channel
two real sections and one resonant pair, i.e. a steady-state Kalman
predictor of up to fourth order, fitted teacher-forced; conservative: the
rule is not refitted jointly) gives the family's one-step skill WITH
history, which is what the M11 networks (all reading the error history)
are compared with.

DECISION RULE. Every variant here is larger than one draw. So only a
FAILED fit is conclusive: if even the best member of the superset, with the
history predictor, leaves clearly more of the error than a network with the
same information (DEFECTS M11), the family cannot express the target. A
successful fit only shows that the superset expresses it; the 'draw'
variant (one draw's size) then checks whether a single family member does.
The M11 references differ in information: every M11 network reads
S[:, :11], the commands and the error history; 'wave' adds the 15
elevations at t (the inputs here), 'wmid' those half a step later on the
dead-reckoned hull. W_src itself sees the sea over the whole step, which is
wmid-like information; both are printed.

How each part of Operator is mirrored (M), widened (W), relaxed (R) or
omitted (O) in the 'family' and 'wide' variants:

  inputs      M  z = (s28 - lib mu) / lib sd, as Operator.step.
  projections W  K = 6 (the family's maximum), each a FREE 28-vector (the
                 family's are sparse, one to three input groups, smooth
                 along the hull for the waves: all special cases).
  pre-nonlin. W  alpha v + sum_j w2_j tanh(w1_j v + b1_j) + beta relu(v - t):
                 the family's three choices (identity, 8-unit tanh MLP,
                 thresholded ReLU) at once; initialised at the identity
                 (alpha 1, w2 ~ 0, beta 0).
  delay       W  a trainable mix of taps v_k .. v_{k-3} (the family picks
                 one delay in 0..3), initialised at tap 0; zeros before the
                 episode starts, as the family's delay line.
  sections    W  per projection one real first-order section
                 (x' = a x + (1 - a) v, a = exp(-dt / tau)) AND one
                 resonant pair (x' = r R(theta) x + [1, 0] v,
                 r = exp(-zeta wn dt), theta = min(wn sqrt(1 - zeta^2) dt,
                 0.97 pi), zeta <= 0.99), output y = c_real x + c_res . x
                 + d v: the family's one section of kind real / res / none
                 is the case of the other gains 0. Discretised at the
                 family's dt = 0.24 s exactly as _Sec does; stable by
                 construction (tau > 0, wn > 0, 0 < zeta < 0.99).
                 Initial poles drawn from the family's ranges.
  scheduling  W  a -> a^e, r -> r^e, theta -> theta e with
                 e = exp(w . sb), sb the family's four slow 5-s low-passes
                 (speed, cos / sin heading, thrust; started at the first
                 input, updated before the section, as _filters); the
                 family's kappa on ONE slow signal is a special case;
                 initialised at w = 0 (unscheduled).
  feature norm M  each filter output centred and scaled by its statistics
                 over the training steps from step 40 on (the family: over
                 the library after a 40-step burn-in). An affine map, so it
                 changes conditioning, not what the readout can express.
  direct in.  W  all 28 inputs (the family: up to 3 chosen ones).
  readout     W  an MLP over [normalised filter outputs, direct inputs]
                 with H units of EACH of the family's four activations
                 (tanh, ReLU, square, steep sigmoid), plus the bilinear
                 gain-scheduling term (u . filters)(V_c . direct) per
                 channel. A sigmoid unit is sigmoid(exp(ls) (w . X + b)):
                 as in the family, its slope (log-uniform in [3, 30] per
                 unit input sd at the start) is a parameter of its own, so
                 steep units are reachable (a raw weight moves only ~5 over
                 a run). The family's part normalisations, mix share beta
                 and _ro_norm are affine maps absorbed by W2 and the output
                 scale / bias.
  saturation  M  rule = A 2.5 tanh(ym / 2.5) + b per channel (A, b
                 trainable: amplitude and pi_m, and the constant offset b0).
  push map    M  the push (rule + events) enters e0 through J, as above.
  events      R  up to 2 (wide: 4) threshold events. Source: a trainable
                 mix of the normalised filter outputs PLUS an own projection
                 through its own sections (the family: one filter output OR
                 an own projection and section), normalised. The hard
                 upcrossing (v_{k-1} < thr <= v_k) becomes
                 sigmoid(s (v_k - thr)) sigmoid(s (thr - v_{k-1})) with a
                 trainable steepness s, so a gradient exists. The size
                 expm1(b E) / expm1(b / 2), E the crossing speed ranked
                 against the library's crossings (an empirical CDF), keeps
                 the expm1 form with E a trainable monotone map
                 e1 |dv| + e2 |dv|^2 (e1, e2 >= 0). Response: a one-step
                 spike plus a resonant section, both trainable, "acting in
                 the step of the hit, then ringing" as _resp_step; a
                 trainable direction (5) replaces H A dir. Starts switched
                 off (direction 0). A relaxation, not a strict superset:
                 the empirical CDF map is only approximated.
  regime      R  a second readout (initialised as a copy of the first)
                 mixed in by sigmoid((k - k_r) / w) with k the episode step
                 and w trainable (the family switches hard at step
                 k_regime = U(25, 70) s / dt, steps 104-292). k_r =
                 104 + 188 sigmoid(theta) lives on that scale: a raw k_r in
                 steps would move only ~5 steps over a run.
  relays      O  the prior draws of this question use relay=False (as
                 prior_mass_m13.py); their hysteresis memory also needs a
                 remembered mode no relaxation here reproduces faithfully.
  slow gain / R  in Operator.run they are 0; in a relabelled rule they are
  bias drift     per-checkpoint constants no input predicts: left to the
                 history predictor (a slow real section), with the constant
                 offset b0 kept in the rule (b).
  clip        O  the push's clip at 3 A_REF is never reached by the
                 target's errors (e0 sd ~ 1 m/s^2 against 6-36); the
                 control's targets are built unclipped (y = rule + noise,
                 as the control's definition says) so that the module can
                 represent them.
  noise part  R  coloured / white noise, state-dependent level (env),
                 bursts and the random part of event sizes do not follow
                 from the inputs; the part of them past errors predict is
                 the history predictor's (above).

WIDE variant: 3 real + 3 resonant sections per projection, 8 delay taps,
16 tanh units in the pre-nonlinearity, 64 units per activation in the
readout, 4 events with 2 resonant response sections each. Family vs wide
compares two supersets; it does not say whether the family's size limits
the fit.

DRAW variant (one draw's size): 16 readout units, each a trainable mix of
the four activations (sigmoid slope bounded to [3, 30]); 3 direct inputs,
each a softmax selection over the 28 whose temperature anneals from 1 to
0.03 over the first 60 % of the steps (early stopping only after that);
unit-norm projections with a group-lasso penalty (lam) over the input
groups state / command / wave; tau in [0.1, 30] s, wn in [0.3, 9] rad/s
(event responses [3, 11.5], zeta [0.1, 0.5]), scheduling on one slow
signal (a softmax choice) with |kappa| <= 0.25. Still wider than a draw in
the pre-nonlinearity mix and in one real + one resonant section per
projection. The fitted parameters (state_dict) of every fit are saved so
a solution can be checked against the family's ranges afterwards.

Data: C and Cb (studies/_cache/meta3), episodes split exactly as
info_target_m11.py (per split every 4th episode held out; of the rest
every 8th for early stopping; the others train). The loss is the MSE per
channel over valid steps from step 40 on, each channel divided by its
training variance. Adam, full batch, early stopping on the validation
episodes. Skill = remaining MSE / MSE of e0 (as prior_mass_m13.py and
DEFECTS M11) on the held-out episodes; also against e0 - W_src.

RECOVERY CONTROL (--control N): does the optimisation find family members
in this huge space at all? N operators are drawn from the training prior
(relay off, null off; by default picked from the first 80 seeds so that
resonant filters, events, a regime switch and an MLP pre-nonlinearity each
occur at least once; --consecutive takes seed0, seed0 + 1, ...) and run
WITH noise on the target's inputs. The target is y = push_response(rule +
noise), the exact nonlinear map from every recorded state (W_src would
cancel: it is known and subtracted); the superset (linear J) is fitted with
the same split and procedure. Reported per channel on the held-out steps:
the skill and the exact floor MSE(noise part) / MSE(y), the normalised gap
(skill - floor) / (1 - floor), the rule's own recovery error
||fit - push_response(rule)||^2 / ||push_response(rule)||^2, and the same
with history against the floor left by the history predictor fitted to the
true noise part. --floor_target F rescales each draw's noise per channel so
that the floor is about F (0.1: as the target's predictable share).

    python studies/family_fit_m13.py [--variant family|wide|draw]
    python studies/family_fit_m13.py --control 5 [--floor_target 0.1]
    python studies/family_fit_m13.py --smoke [--steps 20] [--control 1]
                                   (tiny, for tests; separate pickle)
"""
import argparse
import copy
import math
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache", "meta3")
LOG = os.path.join(C, "family_fit_m13.log")
NAMES = ("surge", "sway", "yaw", "heave", "pitch")
# DEFECTS M11 (studies/_cache/meta2/info_target_m11.log lines 12-20): held-out
# one-step skill of model3 fine-tuned on the target, C and Cb, per input set
M11 = dict(
    base=("S[:11] + commands + error history",
          [0.11, 0.12, 0.29, 0.08, 0.48], [0.08, 0.11, 0.33, 0.09, 0.47]),
    wave=("base + 15 elevations at t (the inputs here)",
          [0.04, 0.04, 0.09, 0.05, 0.34], [0.03, 0.04, 0.11, 0.05, 0.37]),
    wmid=("base + 15 elevations half a step on",
          [0.02, 0.02, 0.04, 0.02, 0.20], [0.02, 0.01, 0.04, 0.03, 0.23]),
    all=("base + roll + wave + wmid",
         [0.02, 0.01, 0.04, 0.01, 0.14], [0.01, 0.01, 0.04, 0.01, 0.17]))
DT = 0.24                                        # the family's control step
SLOW = (0, 3, 4, 5)                              # operators.SLOW
BURN = 40                                        # scored / normalised from
K_REG = (104.0, 292.0)       # the family's k_regime range, steps (25-70 s)
FEATS = ("res", "events", "regime", "mlp")       # control seed picking
VARIANTS = dict(
    family=dict(K=6, n_real=1, n_res=1, taps=4, n_pre=8, H=16, n_ev=2,
                n_ev_res=1),
    wide=dict(K=6, n_real=3, n_res=3, taps=8, n_pre=16, H=64, n_ev=4,
              n_ev_res=2),
    draw=dict(K=6, n_real=1, n_res=1, taps=4, n_pre=8, H=16, n_ev=2,
              n_ev_res=1, draw=True))
_SMOKE = [False]


def log(msg):
    line = time.strftime("%H:%M:%S ") + ("[smoke] " if _SMOKE[0] else "") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ------------------------------------------------------------------ data
def load_target():
    """C and Cb as one set (dict): raw operator inputs X (n, T, 28), e0
    channels 0-4 Y (n, T, 5), valid mask V (n, T) (e0 exists), split id
    (n,), recorded plant states XS (n, T, 14) and commands U (n, T, 2) at
    each step, the episodes' meta (their seas)."""
    from learn.meta.data2 import op_inputs
    lib = dict(np.load(os.path.join(C, "lib.npz")))
    out = {k: [] for k in ("X", "Y", "V", "split", "XS", "U")}
    meta = []
    for s, split in enumerate(("C", "Cb")):
        d = np.load(os.path.join(C, f"{split}.npz"))
        e0 = np.load(os.path.join(C, f"{split}_e0.npz"))["E0"]
        T = d["U"].shape[1]
        L = d["len"]
        # e0 exists for t < len - 1 (packed) or t < len (block splits)
        n_e = L - 1 if d["XS"].shape[1] == T else L
        out["X"].append(op_inputs(d["S"], d["U"]).astype(float))
        out["Y"].append(e0[..., :5].astype(float))
        out["V"].append(np.arange(T)[None] < n_e[:, None])
        out["split"].append(np.full(len(L), s))
        out["XS"].append(d["XS"][:, :T].astype(float))
        out["U"].append(d["U"].astype(float))
        meta += pickle.load(open(os.path.join(C, f"{split}_meta.pkl"), "rb"))
    T = min(x.shape[1] for x in out["X"])
    D = {k: np.concatenate([a if k == "split" else a[:, :T] for a in v])
         for k, v in out.items()}
    D.update(lib=lib, meta=meta)
    return D


def push_map(env, tg, P, chunk=20000):
    """The source plant's error response (n, T, 5) to pushes P (n, T, 5)
    held over each recorded target step: relabel.push_response_red, the
    reduced model stepped with and without the push from the recorded
    state under the recorded command (waves cancel). 0 where e0 does not
    exist."""
    from learn.meta.data2 import plant_to_reduced
    from learn.meta.relabel import push_response_red
    ii, tt = np.nonzero(tg["V"])
    out = np.zeros(P.shape)
    for s in range(0, len(ii), chunk):
        a, b = ii[s:s + chunk], tt[s:s + chunk]
        out[a, b] = push_response_red(
            env["red"], plant_to_reduced(tg["XS"][a, b]), tg["U"][a, b],
            P[a, b], env["t_max"], env["rud_max"], env["dt"], env["sub"],
            env["x_st"])
    return out


def push_jacobian(env, tg):
    """J (n, T, 5, 5): e0 per unit push of each channel, per recorded
    step (push_map of unit pushes; exact where the response is linear)."""
    n, T = tg["V"].shape
    J = np.zeros((n, T, 5, 5))
    for i in range(5):
        e = np.zeros((n, T, 5))
        e[..., i] = 1.0
        J[..., i] = push_map(env, tg, e)
    return J


def source_terms(env, tg):
    """W_src (n, T, 5): e0 channels 0-4 of the SOURCE plant (reduced model,
    waves of the episode's own sea at the stations, ideal actuators, no
    push) over one step from every recorded target state, i.e. the part of
    every source label that the operator does not produce. Also the
    relative rms mismatch between the recorded elevations and relabel's
    sea at the recorded stations (0 up to float32 rounding if the sea is
    the target's)."""
    from learn.meta import relabel
    from learn.meta.data3 import e0_from
    dtc = env["dt"] * env["sub"]
    n, T = tg["V"].shape
    W = np.zeros((n, T, 5))
    num = den = 0.0
    for i in range(n):
        tt = np.flatnonzero(tg["V"][i])
        B = len(tt)
        xs, u = tg["XS"][i, tt], tg["U"][i, tt]
        seas = relabel.Seas([relabel._sea(tg["meta"][i])] * B, env["x_st"],
                            env["y_off"])
        t0 = tt * dtc
        el = seas.stations(xs[:, 0], xs[:, 1], xs[:, 5], t0).reshape(B, 15)
        rec = tg["X"][i, tt, 11:26]
        num += ((el - rec) ** 2).sum()
        den += (rec ** 2).sum()
        _, xn = relabel.simulate(env, xs, u[:, None], t0, seas,
                                 np.zeros((B, 2)),
                                 pushes=np.zeros((B, 1, 5)))
        W[i, tt] = e0_from(env, xs, xn[:, 1], u)[:, :5]
    return W, float(np.sqrt(num / den))


def episode_split(split):
    """Exactly info_target_m11.run_variant's split (split 0 comes first)."""
    idx = np.arange(len(split))
    within = np.concatenate([np.arange((split == s).sum()) for s in (0, 1)])
    hold = idx[within % 4 == 3]
    rest = idx[within % 4 != 3]
    val = rest[np.arange(len(rest)) % 8 == 1]
    tr = rest[np.arange(len(rest)) % 8 != 1]
    return tr, val, hold


def slow_lowpass(Z):
    """The family's slow signals sb (B, T, 4): 5-s low-passes of z[SLOW],
    started at the first input and updated before each step's filters."""
    a = 1.0 - np.exp(-DT / 5.0)
    sb = Z[:, 0, list(SLOW)].copy()
    out = np.zeros(Z.shape[:2] + (len(SLOW),))
    for k in range(Z.shape[1]):
        sb += (Z[:, k, list(SLOW)] - sb) * a
        out[:, k] = sb
    return out


# ---------------------------------------------------------------- module
def _logu(g, lo, hi, shape):
    u = torch.rand(shape, generator=g, dtype=torch.float64)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def _to_raw(x, rng):
    """x > 0 -> its unconstrained parameter: log x (rng None), or the logit
    of x's place in [log lo, log hi] (bounded, log-uniform scale)."""
    if rng is None:
        return torch.log(x)
    lo, hi = math.log(rng[0]), math.log(rng[1])
    f = ((torch.log(x) - lo) / (hi - lo)).clamp(0.02, 0.98)
    return torch.log(f / (1 - f))


def _from_raw(p, rng):
    if rng is None:
        return torch.exp(p)
    lo, hi = math.log(rng[0]), math.log(rng[1])
    return torch.exp(lo + (hi - lo) * torch.sigmoid(p))


def _np_state(net):
    return {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}


class Bank(nn.Module):
    """m input channels, each feeding n_real real first-order sections and
    n_res resonant pairs (operators._Sec, discretised at DT the same way),
    optionally scheduled by the slow signals; output per channel
    y = sum c . x + d u. The pairs run as complex states: r R(theta) acting
    on (x0, x1) is multiplication of x0 + i x1 by r e^{i theta}.
    post: outputs read the state AFTER the update (an event response,
    _resp_step), else before it (a filter, _Sec.step).
    bounded: tau, wn, zeta kept inside the given (the family's) ranges, and
    the scheduling is the family's form, |kappa| <= 0.25 on one slow signal
    (a softmax choice); else all free (wn, tau > 0, zeta < 0.99, any w)."""

    def __init__(self, m, n_real, n_res, g, sched=True, wn=(0.3, 9.0),
                 zeta=(0.05, 1.0), tau=(0.1, 30.0), bounded=False):
        super().__init__()
        self.m, self.nr, self.ns = m, n_real, n_res
        self.bounded = bounded
        self.b_tau = tau if bounded else None
        self.b_wn = wn if bounded else None
        self.b_zeta = (zeta[0], min(zeta[1], 0.99)) if bounded else None
        self.tau_p = nn.Parameter(_to_raw(_logu(g, *tau, (m, n_real)),
                                          self.b_tau).float())
        self.wn_p = nn.Parameter(_to_raw(_logu(g, *wn, (m, n_res)),
                                         self.b_wn).float())
        z = _logu(g, *zeta, (m, n_res))
        if bounded:
            self.zeta_p = nn.Parameter(_to_raw(z, self.b_zeta).float())
        else:
            z = (z / 0.99).clamp(0.02, 0.98)
            self.zeta_p = nn.Parameter(torch.log(z / (1 - z)).float())
        self.c_real = nn.Parameter(torch.randn(m, n_real, generator=g))
        self.c_res = nn.Parameter(torch.randn(m, n_res, 2, generator=g))
        self.d = nn.Parameter(0.5 * torch.randn(m, generator=g))
        self.sched = sched
        n = n_real + n_res
        if sched and bounded:
            self.k_p = nn.Parameter(torch.zeros(m, n))
            self.s_p = nn.Parameter(torch.zeros(m, n, len(SLOW)))
        elif sched:
            self.w_s = nn.Parameter(torch.zeros(m, n, len(SLOW)))

    def poles(self):
        """tau (m, nr), wn (m, ns), zeta (m, ns)."""
        tau = _from_raw(self.tau_p, self.b_tau)
        wn = _from_raw(self.wn_p, self.b_wn)
        zeta = (_from_raw(self.zeta_p, self.b_zeta) if self.bounded
                else 0.99 * torch.sigmoid(self.zeta_p))
        return tau, wn, zeta

    def w_sched(self):
        """Scheduling weights (m, n, 4) on the slow signals."""
        if self.bounded:
            return 0.25 * torch.tanh(self.k_p)[..., None] * torch.softmax(
                self.s_p, -1)
        return self.w_s

    def forward(self, u, sb=None, post=False):
        """u (B, T, m), sb (B, T, 4) -> y (B, T, m)."""
        B, T, m = u.shape
        tau, wn, zeta = self.poles()
        log_a = -DT / tau                                     # log a, (m, nr)
        log_r = -zeta * wn * DT
        th = (wn * torch.sqrt(1 - zeta ** 2) * DT).clamp(max=0.97 * math.pi)
        if self.sched and sb is not None:
            e = torch.exp(torch.einsum("btk,mnk->btmn", sb, self.w_sched()))
            er, es = e[..., :self.nr], e[..., self.nr:]
        else:
            er = es = torch.ones(1, 1, 1, 1, device=u.device)
        lam_re = torch.exp(er * log_a)
        mag = torch.exp(es * log_r)
        ang = (es * th).clamp(max=0.97 * math.pi)
        lam_re = lam_re.expand(-1, T, m, self.nr) if lam_re.shape[1] == 1 \
            else lam_re
        mag = mag.expand(-1, T, m, self.ns) if mag.shape[1] == 1 else mag
        ang = ang.expand_as(mag)
        lam = torch.cat([torch.complex(lam_re, torch.zeros_like(lam_re)),
                         torch.complex(mag * torch.cos(ang),
                                       mag * torch.sin(ang))], -1)
        beta = torch.cat([1 - lam_re, torch.ones_like(mag)], -1)
        drive = torch.complex(beta * u[..., None], torch.zeros_like(beta))
        x = torch.zeros(B, m, self.nr + self.ns, dtype=lam.dtype,
                        device=u.device)
        lam_k, drv_k = lam.unbind(1), drive.expand(B, T, m, -1).unbind(1)
        if lam.shape[0] == 1:
            lam_k = [lk.expand(B, -1, -1) for lk in lam_k]
        xs = []
        for k in range(T):
            if not post:
                xs.append(x)
            x = lam_k[k] * x + drv_k[k]
            if post:
                xs.append(x)
        X = torch.stack(xs, 1)                             # (B, T, m, n)
        # Re(conj(c) x) = c0 x0 + c1 x1 for a pair, c x for a real section
        cv = torch.cat([torch.complex(self.c_real, torch.zeros_like(
            self.c_real)), torch.complex(self.c_res[..., 0],
                                         -self.c_res[..., 1])], -1)
        return (X * cv).real.sum(-1) + self.d * u


class Readout(nn.Module):
    """The family's readout widened: H units of each of the four
    activations (mix=False), or H units each a trainable mix of the four
    (mix=True, the draw variant); a sigmoid unit is
    sigmoid(slope (w . X + b)) with its own log-slope, as the family
    separates slope from direction; plus the bilinear term
    (u . filters)(V_c . direct) per channel."""

    SLOPE = (3.0, 30.0)          # the family's sigmoid slopes per input sd

    def __init__(self, n_in, K, n_dir, H, g, mix=False):
        super().__init__()
        self.H, self.mix = H, mix
        n_u = H if mix else 4 * H
        self.W1 = nn.Parameter(torch.randn(n_u, n_in, generator=g)
                               / math.sqrt(n_in))
        self.b1 = nn.Parameter(torch.randn(n_u, generator=g))
        self.b_ls = self.SLOPE if mix else None
        self.ls = nn.Parameter(_to_raw(_logu(g, *self.SLOPE, (H,)),
                                       self.b_ls).float())
        if mix:
            self.am = nn.Parameter(torch.zeros(H, 4))
        self.W2 = nn.Parameter(0.1 * torch.randn(5, n_u, generator=g)
                               / math.sqrt(n_u))
        self.u = nn.Parameter(0.1 * torch.randn(K, generator=g)
                              / math.sqrt(max(K, 1)))
        self.V = nn.Parameter(torch.randn(5, n_dir, generator=g)
                              / math.sqrt(n_dir))

    def forward(self, X, yn, d):
        H = self.H
        pre = X @ self.W1.T + self.b1
        slope = _from_raw(self.ls, self.b_ls)
        if self.mix:
            a = torch.stack([torch.tanh(pre), F.relu(pre), pre ** 2,
                             torch.sigmoid(slope * pre)], -1)
            h = (a * torch.softmax(self.am, -1)).sum(-1)
        else:
            h = torch.cat([torch.tanh(pre[..., :H]),
                           F.relu(pre[..., H:2 * H]),
                           pre[..., 2 * H:3 * H] ** 2,
                           torch.sigmoid(slope * pre[..., 3 * H:])], -1)
        return h @ self.W2.T + (yn @ self.u)[..., None] * (d @ self.V.T)


class FamilyFit(nn.Module):
    """The superset of one operator draw's rule part (module docstring).
    Its output is a push per channel, in units that J maps to the target's
    training sd."""

    def __init__(self, K, n_real, n_res, taps, n_pre, H, n_ev, n_ev_res,
                 draw=False, seed=0, n_in=28):
        super().__init__()
        from learn.meta.operators import G_CMD, G_STATE, G_WAVE
        g = torch.Generator().manual_seed(seed)
        self.K, self.n_ev, self.draw = K, n_ev, draw
        self.groups = (G_STATE, G_CMD, G_WAVE)
        self.temp = 1.0
        # projections and the pre-nonlinearity (near identity)
        self.P = nn.Parameter(torch.randn(n_in, K, generator=g)
                              / math.sqrt(n_in))
        self.pre_a = nn.Parameter(torch.ones(K))
        self.pre_w1 = nn.Parameter(1.5 * torch.randn(K, n_pre, generator=g))
        self.pre_b1 = nn.Parameter(torch.randn(K, n_pre, generator=g))
        self.pre_w2 = nn.Parameter(0.02 * torch.randn(K, n_pre, generator=g))
        self.relu_b = nn.Parameter(torch.zeros(K))
        self.relu_t = nn.Parameter(torch.ones(K))
        tp = torch.zeros(K, taps)
        tp[:, 0] = 1.0
        self.taps = nn.Parameter(tp)
        self.bank = Bank(K, n_real, n_res, g, bounded=draw)
        # direct inputs: all 28, or (draw) 3 softmax selections
        n_dir = 3 if draw else n_in
        if draw:
            self.sel = nn.Parameter(0.1 * torch.randn(n_in, 3, generator=g))
        # readout (and the regime's second one, a copy at the start)
        self.ro = Readout(K + n_dir, K, n_dir, H, g, mix=draw)
        self.ro2 = copy.deepcopy(self.ro)
        self.kr_p = nn.Parameter(torch.tensor(0.0))     # k_r mid-range
        self.log_kw = nn.Parameter(torch.tensor(math.log(30.0)))
        self.A = nn.Parameter(torch.ones(5))
        self.b = nn.Parameter(torch.zeros(5))
        # events
        if n_ev:
            self.ev_P = nn.Parameter(torch.randn(n_in, n_ev, generator=g)
                                     / math.sqrt(n_in))
            self.ev_bank = Bank(n_ev, n_real, n_res, g, bounded=draw)
            self.ev_w = nn.Parameter(torch.zeros(n_ev, K))
            self.ev_thr = nn.Parameter(1.5 + torch.rand(n_ev, generator=g))
            self.ev_ls = nn.Parameter(torch.full((n_ev,), math.log(4.0)))
            self.ev_e = nn.Parameter(torch.zeros(n_ev, 2))
            b0 = (0.35 - 0.05) / 0.95
            self.ev_b = nn.Parameter(torch.full((n_ev,), math.log(
                b0 / (1 - b0))))
            self.ev_resp = Bank(n_ev, 0, n_ev_res, g, sched=False,
                                wn=(3.0, 11.5), zeta=(0.1, 0.5),
                                bounded=draw)
            self.ev_dir = nn.Parameter(torch.zeros(n_ev, 5))
        for key, n in (("f", K), ("e", n_ev)):
            self.register_buffer(key + "_mu", torch.zeros(n))
            self.register_buffer(key + "_s", torch.ones(n))

    # --------------------------------------------- draw-variant helpers
    def _proj(self, P):
        """draw: unit-norm projections (their scale is absorbed by the
        pre-nonlinearity and the normalisation, so the group penalty acts
        on direction only)."""
        if not self.draw:
            return P
        return P / P.norm(dim=0, keepdim=True).clamp_min(1e-6)

    def penalty(self):
        """draw: group lasso of the unit projections, sum over groups of
        the group's norm minus 1 (0 for one group, up to sqrt(3) - 1 for
        three equal ones); else 0."""
        if not self.draw:
            return torch.zeros(())
        out = 0.0
        for P in [self.P] + ([self.ev_P] if self.n_ev else []):
            Pn = self._proj(P)
            out = out + sum(Pn[list(gr)].norm(dim=0) for gr in self.groups
                            ).sum() - Pn.shape[1]
        return out

    def set_progress(self, frac):
        """draw: anneal the direct-input selection temperature 1 -> 0.03
        over the first 60 % of the steps; True once settled (always for
        the other variants)."""
        if not self.draw:
            return True
        self.temp = math.exp(math.log(0.03) * min(frac / 0.6, 1.0))
        return frac >= 0.6

    def k_r(self):
        return K_REG[0] + (K_REG[1] - K_REG[0]) * torch.sigmoid(self.kr_p)

    @torch.no_grad()
    def describe(self):
        """What the fit chose, to check against the family's ranges."""
        tau, wn, zeta = self.bank.poles()
        d = dict(tau=tau.cpu().numpy(), wn=wn.cpu().numpy(),
                 zeta=zeta.cpu().numpy(), k_r=float(self.k_r()),
                 k_w=float(torch.exp(self.log_kw)),
                 sched=self.bank.w_sched().cpu().numpy())
        Pn = self._proj(self.P)
        sh = torch.stack([Pn[list(gr)].pow(2).sum(0) for gr in self.groups])
        d["group_share"] = (sh / sh.sum(0)).cpu().numpy()   # (3, K)
        if self.draw:
            w = torch.softmax(self.sel / self.temp, 0)
            d["direct"] = w.argmax(0).cpu().numpy()
            d["direct_w"] = w.max(0).values.cpu().numpy()
            d["act_mix"] = torch.softmax(self.ro.am, -1).cpu().numpy()
        return d

    # ------------------------------------------------------- forward
    def _norm(self, y, M, key, fit):
        """Centre and scale per column over the masked steps (fit: from
        this forward, stored for later; else the stored statistics)."""
        if fit:
            w = M[..., None].to(y.dtype)
            n = w.sum()
            mu = (y * w).sum((0, 1)) / n
            s = torch.sqrt(((y - mu) ** 2 * w).sum((0, 1)) / n + 1e-8)
            getattr(self, key + "_mu").copy_(mu.detach())
            getattr(self, key + "_s").copy_(s.detach())
        else:
            mu, s = getattr(self, key + "_mu"), getattr(self, key + "_s")
        return (y - mu) / s

    @staticmethod
    def _shift(v, d):
        return v if d == 0 else F.pad(v[:, :-d], (0, 0, d, 0))

    def forward(self, Z, SB, M, fit=False):
        """Z (B, T, 28) normalised inputs, SB (B, T, 4) slow signals, M
        (B, T) the steps whose statistics normalise (fit) -> push (B, T, 5)."""
        B, T, _ = Z.shape
        v = Z @ self._proj(self.P)
        v = (self.pre_a * v + (torch.tanh(v[..., None] * self.pre_w1
                                          + self.pre_b1) * self.pre_w2).sum(-1)
             + self.relu_b * F.relu(v - self.relu_t))
        vd = sum(self.taps[:, d] * self._shift(v, d)
                 for d in range(self.taps.shape[1]))
        yn = self._norm(self.bank(vd, SB), M, "f", fit)
        dr = Z @ torch.softmax(self.sel / self.temp, 0) if self.draw else Z
        X = torch.cat([yn, dr], -1)
        ym = self.ro(X, yn, dr)
        kk = torch.arange(T, device=Z.device, dtype=Z.dtype)
        gk = torch.sigmoid((kk - self.k_r()) / torch.exp(self.log_kw))
        ym = ym + gk[None, :, None] * (self.ro2(X, yn, dr) - ym)
        out = self.A * 2.5 * torch.tanh(ym / 2.5) + self.b
        if self.n_ev:
            src = self.ev_bank(Z @ self._proj(self.ev_P), SB) \
                + yn @ self.ev_w.T
            ve = self._norm(src, M, "e", fit)
            prev = self._shift(ve, 1)
            first = (kk >= 1).to(Z.dtype)[None, :, None]  # no hit at step 0
            s = torch.exp(self.ev_ls)
            hit = torch.sigmoid(s * (ve - self.ev_thr)) * torch.sigmoid(
                s * (self.ev_thr - prev)) * first
            dv = (ve - prev).abs() * first
            ep = F.softplus(self.ev_e)
            E = ep[:, 0] * dv + ep[:, 1] * dv ** 2
            b = 0.05 + 0.95 * torch.sigmoid(self.ev_b)
            size = (torch.expm1((b * E).clamp(max=30.0))
                    / torch.expm1(b / 2)).clamp(max=100.0)
            r = self.ev_resp(hit * size, post=True)
            out = out + r @ self.ev_dir
        return out


class NoisePred(nn.Module):
    """One-step linear prediction of the remaining error from its own past
    (module docstring): for every output channel c and input channel m a
    filter bank (two real sections, one resonant pair, a direct term) on
    res_m, delayed one step, summed over m. A steady-state Kalman predictor
    of the family's coloured noise (a cascade of at most one resonant and
    one real section, correlated across channels) is such a filter of
    order <= 3. Starts at zero prediction."""

    def __init__(self, g):
        super().__init__()
        self.bank = Bank(25, 2, 1, g, sched=False, wn=(0.3, 12.5),
                         zeta=(0.05, 0.7), tau=(0.1, 30.0))
        with torch.no_grad():
            for p in (self.bank.c_real, self.bank.c_res, self.bank.d):
                p.zero_()

    def forward(self, res):
        """res (B, T, 5) -> prediction (B, T, 5) from res_{<k}."""
        B, T, _ = res.shape
        u = F.pad(res[:, :-1], (0, 0, 1, 0)).repeat(1, 1, 5)  # c * 5 + m
        return self.bank(u).view(B, T, 5, 5).sum(-1)


# ------------------------------------------------------------------- fit
def _train(net, train_loss, val_loss, steps, lr, every, patience, tag,
           progress=None):
    """Adam with cosine decay, full batch, early stopping on val_loss().
    progress(frac) -> ready: before the model is ready (draw variant still
    annealing) the latest state is kept and patience does not count. A
    non-finite loss restores the best state WITH its optimiser state (a
    NaN gradient leaves NaN moments behind) and halves the step size."""
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps,
                                                       eta_min=lr / 30)
    ready = progress(0.0) if progress else True
    best = dict(v=val_loss(), it=0, bad=0, ready=ready,
                state=copy.deepcopy(net.state_dict()),
                opt=copy.deepcopy(opt.state_dict()))
    n_par = sum(p.numel() for p in net.parameters())
    log(f"  {tag}: {n_par} parameters; step 0 validation {best['v']:.4f}")
    t0 = time.time()
    it = 0
    for it in range(1, steps + 1):
        if progress:
            ready = progress(it / steps)
        loss = train_loss()
        if not torch.isfinite(loss):
            net.load_state_dict(best["state"])
            # a copy: load_state_dict keeps the moment tensors, which Adam
            # then updates in place (a second restore would get those)
            opt.load_state_dict(copy.deepcopy(best["opt"]))
            for pg in opt.param_groups:
                pg["lr"] *= 0.5
            log(f"  {tag}: non-finite loss at step {it}, restored step "
                f"{best['it']} (weights and Adam state), lr halved")
            continue
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % every == 0 or it == steps:
            v = val_loss()
            if not ready or v < best["v"] or not best["ready"]:
                best.update(v=v, it=it, bad=0, ready=ready,
                            state=copy.deepcopy(net.state_dict()),
                            opt=copy.deepcopy(opt.state_dict()))
            else:
                best["bad"] += 1
            log(f"  {tag}: step {it}  train {loss.item():.4f}  validation "
                f"{v:.4f}  ({(time.time() - t0) / it:.3f} s/step)")
            if best["bad"] >= patience:
                break
    net.load_state_dict(best["state"])
    log(f"  {tag}: kept step {best['it']} (validation {best['v']:.4f}) "
        f"after {it} steps, {time.time() - t0:.0f} s")
    return dict(kept=best["it"], val=best["v"], steps=it,
                time=time.time() - t0)


def fit_hist(res, W, tr, val, args, dev, tag):
    """NoisePred fitted teacher-forced to res (n, T, 5) (normalised, 0
    where e0 does not exist): loss on the training episodes' steps W,
    early stopping on the validation ones. -> predictions (n, T, 5)."""
    torch.manual_seed(args.seed)
    net = NoisePred(torch.Generator().manual_seed(args.seed)).to(dev)
    R = torch.tensor(res, dtype=torch.float32, device=dev)
    Wt = torch.tensor(W, device=dev)
    Rtr, Wtr, Rva, Wva = R[tr], Wt[tr], R[val], Wt[val]

    def mse(Rx, Wx):
        w = Wx[..., None].float()
        return (((net(Rx) - Rx) ** 2 * w).sum((0, 1)) / w.sum()).mean()

    def val_loss():
        with torch.no_grad():
            return mse(Rva, Wva).item()
    info = _train(net, lambda: mse(Rtr, Wtr), val_loss, args.hist_steps,
                  args.lr, args.every, args.patience, tag)
    with torch.no_grad():
        p = net(R).cpu().numpy()
    return p, dict(info, state=_np_state(net))


def fit(Z, SB, D, J, V, split, cfg, args, dev, tag, full=None, truth=None):
    """Fit FamilyFit so that J push matches D (n, T, 5) (the part of the
    error the rule must produce) on the training episodes, early stopping
    on the validation ones; then the history predictor on what remains.
    Held-out skills per channel (all held-out episodes, and per split)
    against D and, with full (e0), against e0. truth (control): the rule's
    exact error response, for the floor and the recovery error."""
    tr, val, hold = episode_split(split)
    W = V & (np.arange(Z.shape[1])[None] >= BURN)
    d_sd = D[tr][W[tr]].std(0) + 1e-9
    # push units: one unit moves its own channel by about one training sd
    gain = np.abs(np.diagonal(J[tr][W[tr]], axis1=1, axis2=2).mean(0))
    rs = d_sd / np.maximum(gain, 1e-3)
    Jn = J * rs[None, None, None, :] / d_sd[None, None, :, None]

    def tens(ii):
        f = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        return dict(z=f(Z[ii]), sb=f(SB[ii]), d=f(D[ii] / d_sd),
                    w=torch.tensor(W[ii], device=dev), j=f(Jn[ii]))
    Dtr, Dva, Dall = tens(tr), tens(val), tens(np.arange(len(Z)))
    log(f"  {tag}: {len(tr)} training / {len(val)} validation / "
        f"{len(hold)} held-out episodes")
    torch.manual_seed(args.seed)
    net = FamilyFit(**cfg, seed=args.seed).to(dev)

    def err_of(Dx, stats=False):
        push = net(Dx["z"], Dx["sb"], Dx["w"], fit=stats)
        return torch.einsum("btij,btj->bti", Dx["j"], push)

    def mse(pred, Dx):
        w = Dx["w"][..., None].float()
        return (((pred - Dx["d"]) ** 2 * w).sum((0, 1)) / w.sum()).mean()

    def train_loss():
        return mse(err_of(Dtr, True), Dtr) + args.lam * net.penalty()

    def val_loss():
        with torch.no_grad():
            net(Dtr["z"], Dtr["sb"], Dtr["w"], fit=True)   # train stats
            return mse(err_of(Dva), Dva).item()
    info = _train(net, train_loss, val_loss, args.steps, args.lr, args.every,
                  args.patience, tag, net.set_progress)
    with torch.no_grad():
        net(Dtr["z"], Dtr["sb"], Dtr["w"], fit=True)
        Pd = err_of(Dall).cpu().numpy() * d_sd
    res = np.where(V[..., None], (D - Pd) / d_sd, 0.0)
    ph, hinfo = fit_hist(res, W, tr, val, args, dev, tag + " history")
    Ph = ph * d_sd
    out = dict(info, hold=hold, d_sd=d_sd, rs=rs, state=_np_state(net),
               describe=net.describe(), hist=hinfo,
               rule_hold=Pd[hold].astype(np.float32),
               hist_hold=Ph[hold].astype(np.float32))
    Wh, Dh = W[hold], D[hold]
    for nm, sel in (("all", np.ones(len(hold), bool)),
                    ("C", split[hold] == 0), ("Cb", split[hold] == 1)):
        m = Wh & sel[:, None]
        rem = ((Dh - Pd[hold])[m] ** 2).sum(0)
        remh = ((Dh - Pd[hold] - Ph[hold])[m] ** 2).sum(0)
        den = (Dh[m] ** 2).sum(0)
        out["skillD_" + nm], out["skillDh_" + nm] = rem / den, remh / den
        if full is not None:
            # e0 - W_src - J push = D - J push: same remainder, e0's power
            dy = (full[hold][m] ** 2).sum(0)
            out["skill_" + nm], out["skillh_" + nm] = rem / dy, remh / dy
            out["skillW_" + nm] = den / dy
    if truth is not None:
        nc = D - truth
        m = Wh
        den = (Dh[m] ** 2).sum(0)
        out["floor"] = (nc[hold][m] ** 2).sum(0) / den
        out["gap"] = (out["skillD_all"] - out["floor"]) / (1 - out["floor"])
        out["rule_err"] = ((Pd[hold] - truth[hold])[m] ** 2).sum(0) / (
            truth[hold][m] ** 2).sum(0)
        pn, _ = fit_hist(np.where(V[..., None], nc / d_sd, 0.0), W, tr, val,
                         args, dev, tag + " true-noise history")
        out["floor_h"] = ((nc - pn * d_sd)[hold][m] ** 2).sum(0) / den
        out["gap_h"] = (out["skillDh_all"] - out["floor_h"]) / (
            1 - out["floor_h"])
        out["rc_hold"] = truth[hold].astype(np.float32)
    return out


def fmt(x):
    return " ".join(f"{v:.3f}" for v in x)


def save(key, res, path):
    allr = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    allr[key] = res
    tmp = path + ".tmp"
    pickle.dump(allr, open(tmp, "wb"))
    os.replace(tmp, path)


def log_describe(tag, d):
    """One line per fit of what it chose (full detail is in the pickle)."""
    gs = " ".join("/".join(f"{x:.2f}" for x in d["group_share"][:, k])
                  for k in range(d["group_share"].shape[1]))
    msg = (f"  {tag}: k_r {d['k_r']:.0f} (w {d['k_w']:.1f}); projection "
           f"group shares state/cmd/wave {gs}")
    if "direct" in d:
        msg += (f"; direct inputs {d['direct'].tolist()} (weights "
                f"{fmt(d['direct_w'])})")
    log(msg)


# ----------------------------------------------------------- control
def op_feats(op):
    st = op.style
    return dict(res="res" in st["types"], events=st["n_events"] > 0,
                regime=bool(st["regime"]), mlp="mlp" in st["pre"])


def pick_seeds(lib, env, n, seed0, window=80):
    """n control seeds among seed0 .. seed0 + window - 1: greedily the seed
    adding the most still-uncovered FEATS until all are covered (or n are
    picked), then the lowest remaining seeds."""
    from learn.meta.operators import Operator
    dtc = env["dt"] * env["sub"]
    cand = []
    for s in range(seed0, seed0 + window):
        f = op_feats(Operator(s, lib, dt=dtc, L=env["L"], relay=False,
                              null=False))
        cand.append((s, {k for k in FEATS if f[k]}))
    chosen, covered = [], set()
    while len(chosen) < n:
        left = [c for c in cand if c[0] not in chosen]
        s, f = max(left, key=lambda c: (len(c[1] - covered), -c[0]))
        if not f - covered:
            break
        chosen.append(s)
        covered |= f
    chosen += [c[0] for c in cand if c[0] not in chosen][:n - len(chosen)]
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=tuple(VARIANTS), default="family")
    ap.add_argument("--steps", type=int, default=None,
                    help="training steps (default 3000, smoke 20)")
    ap.add_argument("--hist_steps", type=int, default=None,
                    help="history predictor steps (default steps / 3)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam", type=float, default=0.01,
                    help="draw variant: group-lasso weight")
    ap.add_argument("--every", type=int, default=50,
                    help="validate every this many steps")
    ap.add_argument("--patience", type=int, default=12,
                    help="validations without improvement before stopping")
    ap.add_argument("--control", type=int, default=0,
                    help="run the recovery control on this many prior "
                         "draws instead of the target fit")
    ap.add_argument("--seed0", type=int, default=13_000_000,
                    help="first operator seed of the control")
    ap.add_argument("--consecutive", action="store_true",
                    help="control seeds seed0, seed0 + 1, ... (no picking)")
    ap.add_argument("--floor_target", type=float, default=0.0,
                    help="control: rescale the noise so the floor is ~this")
    ap.add_argument("--seed", type=int, default=0, help="init seed")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None,
                    help="results pickle name in the cache (parallel runs "
                         "each write their own)")
    args = ap.parse_args()
    if args.smoke:
        _SMOKE[0] = True
        args.steps = args.steps or 20
        args.hist_steps = args.hist_steps or args.steps
        args.every = min(args.every, 10)
        args.control = min(args.control, 1)
    args.steps = args.steps or 3000
    args.hist_steps = args.hist_steps or max(args.steps // 3, 1)
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    pkl = os.path.join(C, args.out or ("family_fit_m13_smoke.pkl"
                                       if args.smoke else "family_fit_m13.pkl"))
    cfg = VARIANTS[args.variant]
    from learn.meta import relabel
    env = relabel._env()
    tg = load_target()
    lib, X, Y, V, split = tg["lib"], tg["X"], tg["Y"], tg["V"], tg["split"]
    Z = (X - lib["mu"]) / lib["sd"]
    SB = slow_lowpass(Z)
    what = (f"control on {args.control} prior draws, floor target "
            f"{args.floor_target or 'none'}" if args.control
            else "target C + Cb")
    log(f"family_fit_m13 variant {args.variant} {cfg}, {what}, {args.steps} "
        f"+ {args.hist_steps} history steps, lr {args.lr}")
    t0 = time.time()
    J = push_jacobian(env, tg)
    Vm = V & (np.arange(V.shape[1])[None] >= BURN)
    dg = np.diagonal(J[Vm], axis1=1, axis2=2)
    log(f"push map J from the recorded states: diagonal mean "
        f"{fmt(dg.mean(0))}, sway per yaw push {J[Vm][:, 1, 2].mean():.3f} "
        f"(sd {J[Vm][:, 1, 2].std():.3f}) ({time.time() - t0:.0f} s)")
    if not args.control:
        t0 = time.time()
        Wsrc, mis = source_terms(env, tg)
        log(f"W_src (source plant, target sea, ideal actuators, no push): "
            f"recorded vs replayed elevations, relative rms {mis:.1e}; "
            f"W_src sd {fmt(Wsrc[Vm].std(0))}, e0 sd {fmt(Y[Vm].std(0))} "
            f"({time.time() - t0:.0f} s)")
        r = fit(Z, SB, Y - Wsrc, J, V, split, cfg, args, dev, "target",
                full=Y)
        save((args.variant, "target"), r, pkl)
        log_describe("target", r["describe"])
        log(f"held-out one-step skill [{' '.join(NAMES)}] (remaining MSE / "
            f"MSE of e0):")
        log(f"  W_src alone (rule 0)        {fmt(r['skillW_all'])}")
        for nm in ("all", "C", "Cb"):
            log(f"  superset, inputs only  {nm:<3}  {fmt(r['skill_' + nm])}")
        for nm in ("all", "C", "Cb"):
            log(f"  superset + history     {nm:<3}  {fmt(r['skillh_' + nm])}")
        log("  M11 networks (model3 fine-tuned on target data), C | Cb:")
        for k, (info, c, cb) in M11.items():
            log(f"    {k:<5} {fmt(c)} | {fmt(cb)}   inputs: {info}")
        log("  (a superset is larger than the family: only a clearly worse "
            "fit is conclusive; W_src sees the sea over the step, like wmid)")
        return
    from learn.meta.operators import Operator
    tr, val, hold = episode_split(split)
    seeds = ([args.seed0 + j for j in range(args.control)] if args.consecutive
             else pick_seeds(lib, env, args.control, args.seed0))
    res = []
    key = (args.variant, "control", args.floor_target or "prior")
    for j, seed in enumerate(seeds):
        op = Operator(seed, lib, dt=env["dt"] * env["sub"], L=env["L"],
                      relay=False, null=False)
        R, N = op.run(X, noise=True, rng=np.random.default_rng([seed, 7]))
        if args.floor_target > 0:
            m = Vm[tr]
            q = args.floor_target / (1 - args.floor_target)
            f = np.sqrt(q * (R[tr][m] ** 2).mean(0)
                        / ((N[tr][m] ** 2).mean(0) + 1e-12))
            N = N * f
        # the target unclipped (y = rule + noise), W_src left out: it is
        # known and subtracted, so it cancels
        D = push_map(env, tg, R + N)
        rc = push_map(env, tg, R)
        lin = np.einsum("ntij,ntj->nti", J, R + N)
        lin_err = ((lin - D)[Vm] ** 2).sum(0) / (D[Vm] ** 2).sum(0)
        st = op.style
        f_ = op_feats(op)
        has = " ".join(k for k in FEATS if f_[k]) or "none"
        log(f"control {j}: operator seed {seed} [{has}]: "
            f"K {st['K']} {st['types']}, pre {st['pre']}, {st['n_dir']} "
            f"direct, {st['n_events']} events, regime {st['regime']}, "
            f"bilinear share {st['beta']:.2f}; J vs exact push response, "
            f"relative error {fmt(lin_err)}")
        r = fit(Z, SB, D, J, V, split, cfg, args, dev, f"control {j}",
                truth=rc)
        r.update(seed=seed, style=st, feats=f_, lin_err=lin_err)
        res.append(r)
        log_describe(f"control {j}", r["describe"])
        log(f"  control {j}: held-out skill      {fmt(r['skillD_all'])}")
        log(f"  control {j}: exact floor         {fmt(r['floor'])}")
        log(f"  control {j}: rule recovery error {fmt(r['rule_err'])}")
        log(f"  control {j}: + history skill     {fmt(r['skillDh_all'])}")
        log(f"  control {j}: history floor       {fmt(r['floor_h'])}")
        save(key, res, pkl)
    log(f"recovery control per channel [{' '.join(NAMES)}]: (skill - floor) "
        f"/ (1 - floor) | same with history | rule recovery error:")
    for r in res:
        log(f"  seed {r['seed']}: {fmt(r['gap'])} | {fmt(r['gap_h'])} | "
            f"{fmt(r['rule_err'])}")


if __name__ == "__main__":
    main()
