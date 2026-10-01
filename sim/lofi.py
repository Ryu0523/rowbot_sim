#!/usr/bin/env python3
"""
The low-fidelity world: the MPC's own reduced model, run as a plant.

The sim-to-real study needs a SOURCE world where a controller is designed and
tuned cheaply, and a TARGET that differs from it the way a real boat will
differ from any model. Source = this. Target = the full plant (sim/vessel.py),
then a family of perturbed plants, eventually the boat.

It is the same equations the MPC predicts with (control/reduced.py), not a
third model, so "low fidelity" has one meaning in this project:

    surge   thrust against quadratic drag
    heave   second-order oscillator towards the mean surface under the hull
    pitch   second-order oscillator towards the surface slope along it
    sway    rudder lift, linear + quadratic damping, Coriolis
    yaw     first-order (Nomoto) response to the rudder

integrated at a finer step than the MPC's 0.5 s (dt = 0.1 s for the USV,
Froude-scaled), the surface re-sampled every step. With the identified
coefficients the MPC therefore has an exact model here, up to its own
discretisation -- the clean reference every gap is measured from.

What it does NOT have is the gap, by design: roll; the radiation memory
(frequency-dependent added mass and damping); nonlinear buoyancy and
Froude-Krylov; slam loads acting back on the motion; wave-induced sway and yaw;
added resistance, wind, propeller emergence. Its actuators are ideal unless
a parameter says otherwise: the old pair (tau_thrust, rud_rate: a thrust lag,
a nozzle rate limit), or, with act_family = 1, a random actuator chain per
channel (delay, command dead zone, gain and offset, backlash, a rate-limited
first- or second-order response; ACT_FAMILY, act_step; DEFECTS M10).

What it DOES share with the full plant is the measurement: bow immersion, the
Ochi slam count and the slam force are computed exactly as the plant computes
them -- same stations, same keel, same Wagner entry coefficient from the hull's
own sections -- only from this model's motion. The task and its scoring are
then identical in both worlds and only the dynamics differ, which is what a
transfer experiment has to hold fixed.

The interface is the part of NonlinearVessel that Episode, USVControlEnv and
seakeeping.summarise use, so every harness runs here unchanged:

    Episode(db, red, ..., fidelity="low")
    USVControlEnv(hull=..., fidelity="low", plant_params={...} or f(rng))
    config.plant_for(db, sea, hull, fidelity="low", params={...})

Parameters are identified from the full plant once per hull and cached in
studies/_cache/lofi_<hull>.json, keyed by what the calm plant DOES (as
learn/bc_only.py keys its demonstrations), so a change to the physics
re-identifies and a comment edit does not.

Run `python -m sim.lofi [HULL]` for the parameters, the step rate against the
full plant, and the same MPC run in both worlds.
"""
import json
import os
import time
from types import SimpleNamespace

import numpy as np

from control.reduced import ReducedModel
from sim import config

G = 9.81
# Bump when ReducedModel.identify changes what it measures: the fingerprint
# below sees the plant, not the identification procedure.
IDENT_VERSION = 1
DT_USV = 0.1                      # the 10 m USV's step, Froude-scaled

# The sim-to-real PRIOR: how far each parameter of this world may be from the
# vessel it stands for. "x" ranges multiply the identified value (sampled
# log-uniformly); "s" ranges are absolute seconds for the 10 m USV, scaled by
# sqrt(L / 10 m). The third field says what the range rests on. Most rest on
# nothing better than published scatter; the ones that rest on this project's
# own validation name the experiment.
PRIOR = {
    "wn_heave":     ("x", 0.90, 1.10, "no heave decay measured; generic"),
    "wn_pitch":     ("x", 0.90, 1.10, "no pitch decay measured; generic"),
    "z_heave":      ("x", 0.70, 1.40, "radiation-dominated, from the BEM; "
                                      "generic"),
    "z_pitch":      ("x", 0.70, 1.40, "as heave"),
    "k_wave_heave": ("x", 0.50, 2.00, "KCS T2015: short-wave response off "
                                      "by ~2x (DEFECTS F8)"),
    "k_wave_pitch": ("x", 0.50, 2.00, "as heave"),
    "k_drag":       ("x", 0.70, 1.40, "resistance is a placeholder, no "
                                      "trial"),
    "tau_u":        ("x", 0.70, 1.40, "mass + added mass over drag; follows "
                                      "k_drag"),
    "k_nomoto":     ("x", 0.82, 1.64, "35 deg turning radius feasible "
                                      "1.09-2.19 L about 1.79 L (DEFECTS)"),
    "tau_r":        ("x", 0.60, 1.60, "N_r' is a placeholder; same sweep"),
    "k_lift":       ("x", 0.75, 1.30, "rudder lift slope computed, not "
                                      "measured"),
    "k_lin_sway":   ("x", 0.50, 2.00, "Y_v' not identifiable from turning "
                                      "trials (DEFECTS)"),
    "tau_thrust":   ("s", 0.00, 2.00, "the full plant's thruster lag is "
                                      "1.5 s (sim/actuators.py)"),
    # a waterjet's steering channel in place of k_lift and k_nomoto
    "k_jet_side":   ("x", 0.70, 1.30, "jet side force per newton of thrust: "
                                      "pump flow and nozzle deflection "
                                      "efficiency unmeasured"),
    "k_nomoto_f":   ("x", 0.82, 1.64, "yaw per unit side force; N_r' is a "
                                      "placeholder, as for k_nomoto"),
    "rud_rate":     ("s", 0.35, 1.05, "nozzle rate, rad/s: 20-60 deg/s, "
                                      "the actuator is unknown"),
}
_RUDDER_ONLY = ("k_lift", "k_nomoto")
_JET_ONLY = ("k_jet_side", "k_nomoto_f", "rud_rate")


def prior_names(nominal):
    """The PRIOR entries that act on this vessel: a waterjet's steering
    parameters for a jet, a rudder's for a screw."""
    jet = nominal.get("steer_jet", 0.0) > 0.5
    drop = _RUDDER_ONLY if jet else _JET_ONLY
    return [n for n in PRIOR if n not in drop and n in nominal]

# The M10 actuator family (DEFECTS M10, learn/meta/PRIOR_DERIVATION.md D7).
# Per channel, prefix thr_ (thrust, N) or noz_ (nozzle, rad), in the order a
# command passes through them:
#   delay  transport delay, whole plant steps (an integer held in a float, so
#          the plant never re-rounds seconds with another dt)
#   dz     zero-centred command dead zone: |command| < dz acts as 0
#   gain, off   static gain and offset of the command
#   db     backlash: the actuator moves only once the command leaves a band
#          of +-db around its position, and then trails it by db
#   tau    response time constant, s (0 = instantaneous)
#   ratio  tau x ratio when moving to a LOWER value (asymmetric response)
#   zeta   0 = first order; > 0 = second-order servo, wn = 1 / tau, damping
#          zeta (< 1 overshoots)
#   rate   rate limit, units / s (0 = none)
# All neutral here. act_family > 0.5 switches ReducedPlant.step to this chain;
# otherwise the old tau_thrust / rud_rate block runs, unchanged. The ranges
# are in ACT_FAMILY, deliberately not in PRIOR: perturb() and every domain-
# randomisation user draw PRIOR's names, and must not start drawing these.
ACT_NEUTRAL = dict(delay=0.0, dz=0.0, gain=1.0, off=0.0, db=0.0, tau=0.0,
                   ratio=1.0, zeta=0.0, rate=0.0)
ACT_CH = ("thr", "noz")
ACT_KEYS = ("act_family",) + tuple(f"{c}_{k}" for c in ACT_CH
                                   for k in ACT_NEUTRAL)

# Added to the identified coefficients, all at the value that leaves the
# model exactly as the MPC has it: the sea drives heave and pitch at gain 1,
# the actuators are ideal. They exist so the prior above can move them.
EXTRA = dict(k_wave_heave=1.0, k_wave_pitch=1.0, tau_thrust=0.0,
             rud_rate=0.0,             # rud_rate 0 = no rate limit
             act_family=0.0,
             **{f"{c}_{k}": v for c in ACT_CH for k, v in ACT_NEUTRAL.items()})

# The family's distribution, per channel, drawn independently per episode:
# w.p. ACT_P_IDEAL the channel is ideal, else each component w.p. its first
# field. Seconds scale with rs = sqrt(L / 10 m), rates in rad/s with 1 / rs;
# "x scale" = times the channel's range (t_max; rud_max). The last field says
# what the range rests on: nothing measured on this boat or on VM 18, only
# generic small-craft actuator scales. The target's values (the planing
# plant, sim/planing_vessel.py) happen to lie inside; they did not set them.
ACT_P_IDEAL = 0.2
ACT_FAMILY = {
    "delay":    (0.5, "U", 0.0, 0.30, "s; drive-by-wire / servo-loop and "
                                      "hydraulic transport delays, tens to "
                                      "a few hundred ms; generic"),
    "tau_thr":  (0.7, "logU", 0.05, 2.0, "s; engine / pump spool-up of "
                                         "outboards, jets, small diesels "
                                         "(lofi.PRIOR's tau_thrust: 0-2 s)"),
    "tau_noz":  (0.7, "logU", 0.02, 0.5, "s; steering servo bandwidth "
                                         "~0.3-8 Hz; generic"),
    "ratio":    (0.3, "logU", 0.5, 2.0, "given a response: spool-up and "
                                        "spool-down, loaded and unloaded "
                                        "stroke differ; generic"),
    "zeta":     (0.3, "logU", 0.3, 1.5, "given a response: a position servo "
                                        "(over- or underdamped, overshoot "
                                        "below 1); not in the target"),
    "rate_thr": (0.7, "logU", 0.2, 3.0, "s for the full thrust range: "
                                        "throttle / governor slew; generic"),
    "rate_noz": (0.7, "logU", 0.25, 2.0, "rad/s, 14-115 deg/s: steering "
                                         "rams and servos (lofi.PRIOR's "
                                         "rud_rate: 20-60 deg/s)"),
    "dz":       (0.3, "U", 0.0, 0.03, "x scale; command dead band of a "
                                      "throttle / helm around zero"),
    "db":       (0.3, "U", 0.0, 0.03, "x scale; cable / linkage backlash; "
                                      "not in the target"),
    "gain":     (0.3, "logU", 0.9, 1.1, "command calibration error; not in "
                                        "the target (offset drawn with it)"),
    "off":      (None, "U", -0.03, 0.03, "x scale; drawn with the gain"),
}
N_SERVO = 4        # trapezoidal substeps of a second-order response per step


def _draw(rng, name):
    _, law, lo, hi, _ = ACT_FAMILY[name]
    if law == "logU":
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


def draw_act(rng, nominal, dt):
    """Overrides for one draw of the M10 actuator family (ACT_FAMILY), for a
    plant stepped at dt (the delay is rounded to whole steps of it). Every
    ACT_KEYS entry is set, act_family = 1."""
    rs = float(np.sqrt(nominal["L"] / 10.0))
    out = {"act_family": 1.0}
    for c, scale in (("thr", nominal["t_max"]), ("noz", nominal["rud_max"])):
        q = dict(ACT_NEUTRAL)
        if rng.random() >= ACT_P_IDEAL:
            if rng.random() < ACT_FAMILY["delay"][0]:
                q["delay"] = float(round(_draw(rng, "delay") * rs / dt))
            if rng.random() < ACT_FAMILY[f"tau_{c}"][0]:
                q["tau"] = _draw(rng, f"tau_{c}") * rs
                if rng.random() < ACT_FAMILY["ratio"][0]:
                    q["ratio"] = _draw(rng, "ratio")
                if rng.random() < ACT_FAMILY["zeta"][0]:
                    q["zeta"] = _draw(rng, "zeta")
            if rng.random() < ACT_FAMILY[f"rate_{c}"][0]:
                v = _draw(rng, f"rate_{c}")
                q["rate"] = scale / (v * rs) if c == "thr" else v / rs
            for n in ("dz", "db"):
                if rng.random() < ACT_FAMILY[n][0]:
                    q[n] = _draw(rng, n) * scale
            if rng.random() < ACT_FAMILY["gain"][0]:
                q["gain"] = _draw(rng, "gain")
                q["off"] = _draw(rng, "off") * scale
        out.update({f"{c}_{k}": float(v) for k, v in q.items()})
    return out


TARGET_DELAY_S = 0.1     # the planing plant's Waterjet / Nozzle delay


def target_delay_steps(dt):
    """The target-like delay in whole steps of dt: the SMALLEST whole-step
    delay not shorter than the target's 0.1 s (ceil, not round: 0.1 / 0.04
    = 2.5, and Python's round-half-even would give 2 steps = 0.08 s). At
    the source step 0.04 s (scarab195) this is 3 steps = 0.12 s, 0.02 s
    LONGER than the target's (the target plant, stepped at 0.02 s, has 5
    steps = 0.1 s exactly); an At delay at least the target's keeps "the
    stall is gone on At" from resting on an easier actuator than Cb's."""
    return float(np.ceil(TARGET_DELAY_S / dt - 1e-9))


def act_target_like(nominal, dt):
    """A fixed family member shaped like the target's actuators (the planing
    plant's Waterjet and Nozzle, sim/planing_vessel.py): nozzle delay
    target_delay_steps(dt) steps (0.12 s at dt = 0.04 s; the target has 0.1
    s), tau 0.1 s, 40 deg/s; thrust the same delay, tau 0.6 s, full range in
    0.3 s, dead band 50 N per 12 kN. For a TEST split only
    (studies/meta_step2.py 'At'), never for training."""
    out = {k: EXTRA[k] for k in ACT_KEYS}
    nd = target_delay_steps(dt)
    out.update(act_family=1.0, thr_delay=nd,
               thr_dz=nominal["t_max"] * 50.0 / 12000.0, thr_tau=0.6,
               thr_rate=nominal["t_max"] / 0.3,
               noz_delay=nd, noz_tau=0.1,
               noz_rate=float(np.radians(40.0)))
    return out


def act_arrays(p, t_max, rud_max):
    """The chain's per-channel parameters as arrays (..., 2) = (thrust,
    nozzle) from a parameter dict of scalars or of (B,) arrays, plus the
    travel limits lo / hi ([0, t_max], [-rud_max, rud_max]) and the delays
    as integers."""
    q = {k: np.stack(np.broadcast_arrays(
        np.asarray(p[f"thr_{k}"], float), np.asarray(p[f"noz_{k}"], float)),
        -1) for k in ACT_NEUTRAL}
    z = np.zeros_like(q["tau"])
    q["lo"] = z + np.array([0.0, -rud_max])
    q["hi"] = z + np.array([t_max, rud_max])
    q["delay"] = np.rint(q["delay"]).astype(int)
    return q


def act_step(q, pos, vel, c_d, dt):
    """One plant step of the M10 actuator chain, elementwise over arrays
    (..., 2) = (thrust, nozzle), with the delayed command c_d held over the
    step. The SAME function runs in ReducedPlant.step and in
    learn/meta/relabel.simulate, so a replay is exact by construction.

    Command: dead zone, gain and offset, clip to the travel; backlash gives
    the target the actuator moves to. First order: dx/dt = -clip(x / tau,
    +-rate) for x = position - target, solved exactly over the step (a ramp
    at the rate while |x| > rate tau, then exponential), and the force
    applied is the exact time average of the position over the step. This
    is continuous in every input: the design's "exponential, then clip the
    step" switched from the exponential average to the ramp midpoint at the
    limit, a jump of up to rate dt / 2 in the applied value (review of M10).
    Second order: x'' = -wn^2 x - 2 zeta wn x', N_SERVO trapezoidal
    substeps, the velocity clipped to the rate and the position to the
    travel; applied = the trapezoidal average.

    pos, vel: positions and servo velocities before the step. Returns (new
    positions, new velocities (0 for first-order channels), applied)."""
    c_z = np.where(np.abs(c_d) < q["dz"], 0.0, c_d)
    c_e = np.clip(q["gain"] * c_z + q["off"], q["lo"], q["hi"])
    err = c_e - pos
    err = np.sign(err) * np.maximum(np.abs(err) - q["db"], 0.0)
    tgt = pos + err
    tau = np.where(err < 0.0, q["tau"] * q["ratio"], q["tau"])
    rate = q["rate"]
    lim = rate > 0.0
    has_tau = tau > 0.0
    tt = np.where(has_tau, tau, 1.0)
    rt = np.where(lim, rate, 1.0)
    # first order: a = |x|, moving towards the target along sg
    a, sg = np.abs(err), np.sign(err)
    sat = lim & (a > rate * tau)
    t1 = np.where(sat, (a - rate * tau) / rt, 0.0)
    ramp = np.minimum(t1, dt)
    a_r = np.where(sat, np.where(t1 < dt, rate * tau, a - rate * dt), a)
    area1 = ramp * (a + a_r) / 2.0
    t2 = dt - ramp
    e2 = np.where(has_tau, np.exp(-t2 / tt), np.where(t2 > 0.0, 0.0, 1.0))
    a_1 = a_r * e2
    area2 = np.where(has_tau, a_r * tt * (1.0 - e2), 0.0)
    pos1 = tgt - sg * a_1
    app1 = tgt - sg * (area1 + area2) / dt
    so = (q["zeta"] > 0.0) & has_tau
    if not np.any(so):
        return pos1, np.zeros_like(pos1), app1
    # second order, trapezoidal rule (A-stable, keeps the overshoot)
    zeta, w = q["zeta"], 1.0 / tt
    h = dt / N_SERVO
    zwh, w2 = zeta * w * h, w * w
    det = 1.0 + zwh + 0.25 * w2 * h * h
    x, v = -err, np.where(so, vel, 0.0)
    area = np.zeros_like(x)
    for _ in range(N_SERVO):
        rx = x + 0.5 * h * v
        rv = v - 0.5 * h * (w2 * x + 2.0 * zeta * w * v)
        xn = ((1.0 + zwh) * rx + 0.5 * h * rv) / det
        vn = (rv - 0.5 * h * w2 * rx) / det
        dx = np.where(lim, np.clip(xn - x, -rate * h, rate * h), xn - x)
        vn = np.where(lim, np.clip(vn, -rate, rate), vn)
        pn = tgt + (x + dx)
        pc = np.clip(pn, q["lo"], q["hi"])
        vn = np.where(pc != pn, 0.0, vn)
        xn = pc - tgt
        area = area + 0.5 * h * (x + xn)
        x, v = xn, vn
    return (np.where(so, tgt + x, pos1), np.where(so, v, 0.0),
            np.where(so, tgt + area / dt, app1))


def default_dt(sc):
    """This world's step for a vessel with scales `sc` (config.scales_for):
    twice the full plant's. The heave and pitch updates are exact whatever the
    step, so it only has to resolve the wave, and 0.1 s is 1/50 of the
    shortest encounter period the USV meets."""
    return 2.0 * sc["dt"]


# ---------------------------------------------------------------- the plant
class ReducedPlant:
    """The reduced model with NonlinearVessel's interface.

    State: the full plant's layout, [eta(6), nu(6), thrust, rudder], with roll
    and roll rate held at zero, so from_plant_state, the observation and the
    metrics index it exactly as they index the plant.
    """

    def __init__(self, params, geom, sea, dt):
        self.p = dict(params)
        self.model = ReducedModel(self.p)
        self.geom, self.sea, self.dt = geom, sea, float(dt)
        # the attributes the harnesses read off a plant
        self.L, self.T, self.B, self.rho = geom.L, geom.T, geom.B, geom.rho
        self.sec, self.slam, self.thrusters = geom.sec, geom.slam, \
            geom.thrusters
        self.draft_bow, self.freeboard = geom.draft_bow, geom.freeboard
        self.v_slam = geom.v_slam
        self.prop = SimpleNamespace(t_max=self.p["t_max"],
                                    u_ref=self.p["u_design"])
        self.propulsor = geom.propulsor
        self.intakes = geom.intakes
        self.rudder = SimpleNamespace(max=self.p["rud_max"])
        self.n_state = 14
        self.n_rad = 0
        self.fidelity = "low"
        # the MPC's own stations: five from stern to bow, port/centre/stbd
        self.x_st = np.linspace(self.p["x_stern"], self.p["x_bow"], 5)
        self.y_off = np.array([-0.5 * self.B, 0.0, 0.5 * self.B])
        self._sp = float(self.p["sign_pitch"])
        # the wave field's components, read once: the surface and its
        # vertical velocity come from ONE phase evaluation per step
        need = ("a", "k", "w", "th", "phi")
        self._fast = all(hasattr(sea, n) for n in need)
        if self._fast:
            self._wa, self._wk, self._ww = sea.a, sea.k, sea.w
            self._wphi = sea.phi
            self._wc, self._ws = np.cos(sea.th), np.sin(sea.th)
            self._waw = sea.a * sea.w
        self.last_bow_acc = 0.0
        self.last_cg_acc = 0.0
        self.last_slam_force = 0.0
        self.peak_slam_force = 0.0
        self.last_rel_bow = 0.0
        self.slam_count = 0
        self._emerged = False
        # the M10 actuator chain (act_family > 0.5): its parameters, and its
        # hidden state -- the delay line (the last d_max commands, physical
        # units) and the servo velocities. The state is filled lazily at the
        # first step after construction or initial_state(), from the actual
        # positions then, so a new-family plant's step() is NOT a pure
        # function of (s, t, command): a probe from arbitrary states (as
        # data2.push_gain does on an old-style plant) must save and restore
        # it (act_state / set_act_state).
        self._act_on = self.p.get("act_family", 0.0) > 0.5
        if self._act_on:
            self._act_q = act_arrays(self.p, self.p["t_max"],
                                     self.p["rud_max"])
            self._act_d = self._act_q["delay"]
            self._act_dmax = int(self._act_d.max())
        self._line = self._vel = None

    # ------------------------------------------------------------ helpers
    def act_state(self):
        """A copy of the M10 actuator's hidden state (delay line, servo
        velocities); None before its first step."""
        if self._line is None:
            return None
        return self._line.copy(), self._vel.copy()

    def set_act_state(self, st):
        """Set the hidden state from act_state() (None: refill lazily)."""
        if st is None:
            self._line = self._vel = None
        else:
            self._line, self._vel = st[0].copy(), st[1].copy()

    def act_vel(self):
        """The servo velocities (thrust, nozzle) before the next step."""
        return np.zeros(2) if self._vel is None else self._vel.copy()
    def with_sea(self, sea, params=None, **_ignored):
        """The same vessel in other water; `wind=` etc. are accepted and
        ignored, as this world has no wind."""
        p = dict(self.p)
        if params:
            p.update(params)
        return ReducedPlant(p, self.geom, sea, self.dt)

    def unpack(self, s):
        return s[:6], s[6:12], s[12:12], s[12], s[13]

    def initial_state(self, u0=0.0):
        s = np.zeros(self.n_state)
        s[6] = u0
        self._line = self._vel = None       # refilled at the next step
        return s

    def _surface(self, X, Y, t):
        """Elevation and its rate at the points (X, Y)."""
        if not self._fast:
            return self.sea.eta(X, Y, t), self.sea.eta_dot(X, Y, t)
        ph = (self._wk * (X[:, None] * self._wc + Y[:, None] * self._ws)
              - self._ww * t + self._wphi)
        return ((self._wa * np.cos(ph)).sum(-1),
                (self._waw * np.sin(ph)).sum(-1))

    def prop_submergence(self, eta, t):
        """Water depth over the shallowest propeller centre (a waterjet's
        intake), as the plant computes it (roll is zero here). An
        observation only: nothing in this world loses thrust to it."""
        pts = np.asarray(self.geom.intakes, float)
        c, s_ = np.cos(eta[5]), np.sin(eta[5])
        xs = eta[0] + pts[:, 0] * c - pts[:, 1] * s_
        ys = eta[1] + pts[:, 0] * s_ + pts[:, 1] * c
        hull_z = eta[2] + self._sp * pts[:, 0] * eta[4] + pts[:, 1] * eta[3]
        return float(np.min(self.sea.eta(xs, ys, t) - hull_z - pts[:, 2]))

    # ----------------------------------------------------------- dynamics
    def step(self, s, t, thrust_cmd, rudder_cmd, dt=None):
        dt = self.dt if dt is None else dt
        p = self.p
        thr, rud = float(s[12]), float(s[13])

        if self._act_on:
            # The M10 chain (act_step). The delay line holds the last d_max
            # commands; filled with the actual positions at the first step,
            # then pushed once per step (as sim/actuators.Delay, whose
            # buffer starts with the first command instead).
            D = self._act_dmax
            if self._line is None:
                self._line = np.repeat(np.asarray(s[12:14], float)[None],
                                       D, 0)
                self._vel = np.zeros(2)
            full = np.concatenate([self._line, np.array(
                [[float(thrust_cmd), float(rudder_cmd)]])])
            c_d = full[D - self._act_d, (0, 1)]
            self._line = full[1:]
            pos, self._vel, app = act_step(self._act_q, np.array([thr, rud]),
                                           self._vel, c_d, dt)
            thr_new, rud_new = float(pos[0]), float(pos[1])
            thr_app, rud_app = float(app[0]), float(app[1])
        else:
            # Actuators. Ideal by default -- what the MPC assumes. With a
            # lag the force applied over the step is the exact average of
            # the first-order response, not its end value.
            tau = p["tau_thrust"]
            if tau > 0.0:
                e = np.exp(-dt / tau)
                thr_new = thrust_cmd + (thr - thrust_cmd) * e
                thr_app = thrust_cmd + (thr - thrust_cmd) * (tau / dt) \
                    * (1 - e)
            else:
                thr_new = thr_app = float(thrust_cmd)
            rate = p["rud_rate"]
            if rate > 0.0:
                rud_new = rud + float(np.clip(rudder_cmd - rud, -rate * dt,
                                              rate * dt))
                rud_app = 0.5 * (rud + rud_new)
            else:
                rud_new = rud_app = float(rudder_cmd)

        # The surface at the start of the step (a zero-order hold, as the MPC
        # holds it), at the MPC's 5 x 3 stations for the dynamics and at the
        # plant's own stations for the measurement, in one evaluation.
        c, s_ = np.cos(s[5]), np.sin(s[5])
        xd = s[0] + self.x_st[:, None] * c - self.y_off[None, :] * s_
        yd = s[1] + self.x_st[:, None] * s_ + self.y_off[None, :] * c
        xm = self.sec.x
        X = np.concatenate([xd.ravel(), s[0] + xm * c])
        Y = np.concatenate([yd.ravel(), s[1] + xm * s_])
        e_all, ed_all = self._surface(X, Y, t)
        n_d = xd.size
        eta_dyn = e_all[:n_d].reshape(1, len(self.x_st), 3)

        sr = self.model.from_plant_state(s)[None, :]
        ns, a_bow, _ = self.model.step(sr, thr_app, rud_app, eta_dyn,
                                       self.x_st, dt)
        self._measure(s, e_all[n_d:], ed_all[n_d:])
        self.last_bow_acc = float(a_bow[0])
        self.last_cg_acc = float((ns[0, 4] - s[8]) / dt)
        self.last_thr_app = float(thr_app)   # force applied over the step

        x, y, u, z, zd, th, thd, psi, r, v = ns[0]
        out = np.zeros(self.n_state)
        out[[0, 1, 2, 4, 5]] = x, y, z, th, psi
        out[[6, 7, 8, 10, 11]] = u, v, zd, thd, r
        out[12], out[13] = thr_new, rud_new
        return out

    def _measure(self, s, eta_m, etad_m):
        """Bow immersion, slam count and slam force, as the plant has them
        (NonlinearVessel.wave_forces / _update_slam / slam_load), from this
        model's heave and pitch. The plant samples the surface at the section
        centroid's depth for its immersion; the surface itself is used here,
        as the reduced model uses it."""
        x = self.sec.x
        z, th, zd, thd = s[2], s[4], s[8], s[10]
        d = eta_m - (z + self._sp * x * th)
        w_vessel = zd + self._sp * x * thd
        self.last_rel_bow = float(d[-1])
        bow_out = d[-1] <= self.sec.keel[-1]
        rel_v = w_vessel[-1] - etad_m[-1]
        if self._emerged and not bow_out and rel_v < -self.v_slam:
            self.slam_count += 1
        self._emerged = bool(bow_out)
        if self.geom.use_slam_load:
            v_entry = np.maximum(etad_m - w_vessel, 0.0)
            fz = float(np.sum(self.slam.wagner_slope(d) * v_entry ** 2)
                       * self.sec.dx)
            self.last_slam_force = fz
            self.peak_slam_force = max(self.peak_slam_force, abs(fz))


# ------------------------------------------------ identification and cache
_TEMPLATES = {}


def _key(db, hull):
    h = config.resolve(db, hull)
    return h, (h.name if h is not None else "wigley10")


def _cache_path(name):
    d = os.path.join(config.HERE, "studies", "_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"lofi_{name}.json")


def _fingerprint(plant):
    """What the calm plant DOES over three seconds: displaced in heave and
    pitch, thrust and rudder both moving. Every calm-water term the
    identification sees moves this number; an edited comment does not."""
    from sim.forces import Wind
    from sim.test_vessel import Monochromatic
    v = plant.with_sea(Monochromatic(1.0, 0.0), wind=Wind())
    k, rs = v.prop.t_max / 12000.0, float(np.sqrt(plant.L / 10.0))
    s = v.initial_state(4.0 * rs)
    s[2], s[4] = 0.3 * plant.T, 0.03
    for i in range(60):
        s = v.step(s, i * v.dt, (6000.0 + 2000.0 * np.sin(0.3 * i)) * k,
                   np.radians(12.0) * np.sin(0.2 * i), v.dt)
    return float(np.dot(s[:12], np.arange(1, 13)))


def template(db, hull=None, verbose=False):
    """(geometry, identified parameters) for the vessel of `db`, once per
    process; the parameters from the cache while the plant is unchanged."""
    h, name = _key(db, hull)
    hit = _TEMPLATES.get(name)
    if hit is not None:
        return hit
    plant, _ = config.calm_plant(h, db=db)
    geom = SimpleNamespace(
        name=name, L=float(plant.L), T=float(plant.T), B=float(plant.B),
        rho=float(plant.rho), sec=plant.sec, slam=plant.slam,
        thrusters=[tuple(map(float, t)) for t in plant.thrusters],
        intakes=[tuple(map(float, t)) for t in plant.intakes],
        propulsor=getattr(plant, "propulsor", "propeller"),
        draft_bow=float(plant.draft_bow), freeboard=float(plant.freeboard),
        v_slam=float(plant.v_slam),
        use_slam_load=bool(getattr(plant, "use_slam_load", True)))
    fp = _fingerprint(plant)
    path = _cache_path(name)
    p = None
    if os.path.exists(path):
        with open(path) as f:
            c = json.load(f)
        if (c.get("version") == IDENT_VERSION
                and abs(c.get("fingerprint", np.inf) - fp)
                <= 1e-9 * max(1.0, abs(fp))):
            p = c["params"]
    if p is None:
        if verbose:
            print(f"  identifying the low-fidelity model for {name} "
                  f"(cached afterwards) ...")
        p = {k: float(v) for k, v in ReducedModel.identify(plant).p.items()}
        with open(path, "w") as f:
            json.dump(dict(hull=name, version=IDENT_VERSION, fingerprint=fp,
                           params=p), f, indent=1)
    p = {**p, **EXTRA}
    _TEMPLATES[name] = (geom, p)
    return geom, p


def nominal_params(db=None, hull="wigley10"):
    """A copy of the identified parameters (with EXTRA at their neutral
    values) -- the low-fidelity world as the MPC models it."""
    if db is None:
        hull, db = config.load(hull)
    return dict(template(db, hull)[1])


def plant_for(db, sea, hull=None, dt=None, params=None):
    """The low-fidelity plant in `sea`; `params` overrides named coefficients
    (a typo raises rather than being silently ignored)."""
    geom, p0 = template(db, hull)
    p = dict(p0)
    if params:
        bad = set(params) - set(p0)
        if bad:
            raise KeyError(f"not parameters of the low-fidelity model: "
                           f"{sorted(bad)}")
        p.update(params)
    if dt is None:
        dt = DT_USV * np.sqrt(geom.L / 10.0)
    return ReducedPlant(p, geom, sea, dt)


# ------------------------------------------------------ the prior, sampled
def perturb(nominal, rng, scale=1.0, names=None):
    """Overrides drawn from PRIOR: multipliers log-uniform in their range,
    absolute values uniform. `scale` in [0, 1] shrinks every range towards
    the nominal (0 = nominal, 1 = the full prior)."""
    rs = float(np.sqrt(nominal["L"] / 10.0))
    out = {}
    for n in (prior_names(nominal) if names is None else names):
        kind, lo, hi, _ = PRIOR[n]
        if kind == "x":
            f = np.exp(scale * rng.uniform(np.log(lo), np.log(hi)))
            out[n] = float(nominal[n] * f)
        else:
            # rates are rad/s, scaled the other way (1/sqrt(lam))
            k = 1.0 / rs if n == "rud_rate" else rs
            val = rng.uniform(lo, hi) * k
            out[n] = float(nominal[n] + scale * (val - nominal[n]))
    return out


def from_unit(nominal, z, names):
    """Overrides from z in [-1, 1]^n, one entry per name, mapped onto the
    PRIOR range (log-scaled for multipliers) -- the box an optimiser or a
    policy searches when it adapts the model."""
    rs = float(np.sqrt(nominal["L"] / 10.0))
    out = {}
    for n, zi in zip(names, np.clip(np.asarray(z, float), -1.0, 1.0)):
        kind, lo, hi, _ = PRIOR[n]
        a = 0.5 * (zi + 1.0)
        if kind == "x":
            out[n] = float(nominal[n] * np.exp(np.log(lo)
                                               + a * np.log(hi / lo)))
        else:
            k = 1.0 / rs if n == "rud_rate" else rs
            out[n] = float((lo + a * (hi - lo)) * k)
    return out


# -------------------------------------------------------------------- main
def main(hull="wigley10"):
    from sim.env import Episode, score
    from sim.rl_env import USVControlEnv
    h, db = config.load(hull)
    t0 = time.perf_counter()
    geom, p = template(db, h, verbose=True)
    print(f"low-fidelity world for {geom.name}: "
          f"{time.perf_counter() - t0:.1f} s to build\n")
    print("  identified coefficients (the MPC's model):")
    for k in ("wn_heave", "z_heave", "wn_pitch", "z_pitch", "tau_u",
              "k_drag", "k_nomoto", "tau_r", "k_lin_sway"):
        print(f"    {k:>11} = {p[k]:.4g}")
    print(f"  prior: {len(PRIOR)} parameters can be randomised "
          f"(sim.lofi.PRIOR)\n")

    # the direct-control environment on both worlds, random actions
    rate = {}
    for fid in ("high", "low"):
        env = USVControlEnv(hull=h, db=db, fidelity=fid)
        obs, _ = env.reset(seed=0)
        rng = np.random.default_rng(0)
        n, t1 = 300, time.perf_counter()
        for _ in range(n):
            obs, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 2))
            if term or trunc:
                env.reset()
        rate[fid] = n / (time.perf_counter() - t1)
        print(f"  {fid:>4} fidelity: {rate[fid]:6.0f} control steps/s "
              f"= {rate[fid] * env.dt_ctrl:5.0f}x real time "
              f"(observation {obs.shape[0]} numbers)")
    print(f"  low / high = {rate['low'] / rate['high']:.1f}x\n")

    # the same MPC in both worlds: the tuning setup of learn/tune.py (MPC on
    # thrust, autopilot on heading), SS5 head seas, two wave realisations
    red = ReducedModel(dict(template(db, h)[1]))
    t_end = 120.0 * np.sqrt(p["L"] / 10.0)
    keys = ("u_mean", "acc_rms", "acc_p99", "rvm_rms", "slam_rate_ochi",
            "slam_impulse", "cross_rms")
    print(f"  same MPC, both worlds (SS5, {t_end:.0f} s, seeds 0-1):")
    print("    " + f"{'':>6}" + "".join(f"{k:>15}" for k in keys)
          + f"{'score':>9}")
    for fid in ("high", "low"):
        ms = []
        for sd in (0, 1):
            ep = Episode(db, red, hs=3.25, tp=9.7, seed=sd, u_ref=p["u_design"],
                         n_samples=128, n_freq=24, n_dir=5, use_rudder=False,
                         autopilot=True, hull=h, fidelity=fid)
            ms.append(ep.run(t_end))
        row = [np.mean([m[k] for m in ms]) for k in keys]
        sc = np.mean([score(m, p["u_design"]) for m in ms])
        print("    " + f"{fid:>6}" + "".join(f"{v:>15.3f}" for v in row)
              + f"{sc:>9.4f}")
    return geom, p


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2])            # python -m sim.lofi [HULL]
