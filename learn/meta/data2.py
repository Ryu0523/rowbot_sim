#!/usr/bin/env python3
"""
Episodes, library and relabelled targets for the v0 operator family
(learn/meta/operators.py; design in learn/meta/PRIOR_DERIVATION.md, D1-D4).

Sea (P0 of the derivation). Every episode draws its own jittered sea
(step0_preview_spec.WaveField(jitter=True), 48 x 8 components). The old
task sea is 24 fixed evenly spaced frequencies x 5 directions: at a fixed
point it repeats every 46 s. Training seas vary in state and direction:
  Hs U[0.5, 1.5] m, Tp U[4, 7] s, direction U[0, 360), spreading s U[2, 10],
  long-crested w.p. 0.25.
The target (world='high', split C) uses the task's sea state (Hs 1 m,
Tp 5 s, s = 8), with a random direction and the same jittered synthesis.

Scenario. As step 1 (data.py): speed targets 16-30 kn held 10-40 s, and
heading targets held 20-60 s. Now also an OU dither on the nozzle as well
as the thrust, each on w.p. 0.7 (tau LogU[0.5, 5] s, amplitude U[0, 0.15]
of range). A quarter of the episodes are steady (one speed, one heading, no
dither).

Operator (world 'low'). Its push is computed once per control step from
the measured inputs and the commands applied during the step, and held
over the substeps (a zero-order hold).

Actuators (world 'low'), chosen by the job key act_family:
- 'old' (default; the meta2 cache): w.p. 0.75 the thruster lag and the
  nozzle rate are drawn from lofi.PRIOR's ranges, else ideal.
- 'm10' (the meta3 cache; DEFECTS M10, PRIOR_DERIVATION.md D7): the random
  actuator family lofi.ACT_FAMILY per channel (delay, command dead zone,
  gain and offset, backlash, rate-limited first- or second-order response),
  drawn from its OWN stream [seed, 11]. The old draw is still made from the
  episode's stream and discarded, so sea, scenario and operator seed are
  those of the meta2 episode with the same seed; only the actuators differ.
- 'at' (test split At only): lofi.act_target_like, fixed target-shaped
  actuators.
- job['act_params'] (tests): these plant overrides verbatim.
The same parameters go to the plant AND the nominal twin, so the source
shows what the target's inputs show (the actuator state trailing its
command, S[k, 5:7] != U[k - 1]) without the actuator's behaviour becoming a
velocity error term; the twin copies the plant's actuator hidden state
(delay line, servo velocity) at every control step, so the two can never
drift apart. The actuator's own behaviour shows in the model's actuator
channels (data3.e0_from, e channels 8-9). Errors tied to commands come from
the operators, which read both the actuator states and the commands.
meta['act'] is exactly the dict of plant overrides (None = ideal, old).

Recorded per control step, raw:
  S   (26) the operator's / model's inputs: residuals.inputs_of (9), heave
      and pitch about the running attitude (2), and the 15 elevations at the
      MPC's stations (stern to bow, port / centre / starboard)
  XS  (14) plant state (for the model's own rollouts in the targets)
  U   (2)  commanded thrust / t_max, nozzle / rud_max
  E   (5)  observed one-step error: (boat - nominal twin) velocity change /
      step -- the target quantity, exactly as in step 1
  RR, RN (5) the operator's rule and noise parts (source only; diagnostic
      and relabelling)
  ACG      CG acceleration, g
  AV  (2)  the M10 servo velocities before the step (0 unless second order;
      a replay from a mid-episode moment needs them)
and the operator's slow state at the checkpoints CK. With job['raw64']
(tests) the records stay float64 and add R (the push applied, float64) and
CMD (thrust N, nozzle rad as the plant got them).

Targets (build_targets). At each checkpoint ck (steps 60, 120, ...,
while ck + H <= length) the operator's RULE is evaluated on M segments of
H = 20 steps:
- m = 0 is the episode's own future;
- m >= 1 are donors: another episode's inputs, 125 steps of warm-up and then
  H steps, with the operator's slow state copied from ck and held.
Targets are mapped to e units exactly by push_response: the source
plant stepped with and without the push. (The fixed gain G of push_gain is
kept as a diagnostic: in sway it misses the yaw-rate-dependent Coriolis
cross term by about 20%.) For every
segment the model's horizon inputs are also stored: the reduced model's own
rollout from the segment's first state under its recorded commands, with
no future waves (the MPC's convention, eta = 0 beyond the current step).
"""
import os

import numpy as np

from learn.meta.operators import A_REF, CLIP, Operator, clip_push
from learn.meta.residuals import VEL_IDX, inputs_of
from learn.repro.task import Mission, ctx

KN = 0.514444
T_EP = 90.0
H = 20                  # prediction horizon, control steps
WARM = 125              # donor warm-up, control steps (30 s)
CK0 = 60                # checkpoints every CK0 steps
M_DON = 12              # donor segments per checkpoint
N_FREQ, N_DIR = 48, 8


def sea_for(rng, target=False):
    if target:
        hs, tp, s, long = 1.0, 5.0, 8.0, False
    else:
        hs, tp = rng.uniform(0.5, 1.5), rng.uniform(4.0, 7.0)
        s, long = rng.uniform(2.0, 10.0), rng.random() < 0.25
    return dict(hs=float(hs), tp=float(tp),
                theta0=float(rng.uniform(0, 2 * np.pi)), n_freq=N_FREQ,
                n_dir=1 if long else N_DIR, jitter=True, spread_s=float(s))


class ZOHInjector:
    """Holds the operator's push r over a control step: velocity kicks
    after every plant substep (Mission's residual hook), heave part added
    to the measured CG acceleration."""

    def __init__(self):
        self.r = np.zeros(5)

    def __call__(self, s, dt, plant=None):
        s = s.copy()
        for c, i in enumerate(VEL_IDX):
            s[i] += self.r[c] * dt
        if plant is not None:
            plant.last_cg_acc = float(plant.last_cg_acc + self.r[3])
        return s


def _stations(twin, s, t):
    c_, s_ = np.cos(s[5]), np.sin(s[5])
    X = s[0] + twin.x_st[:, None] * c_ - twin.y_off[None, :] * s_
    Y = s[1] + twin.x_st[:, None] * s_ + twin.y_off[None, :] * c_
    eta, _ = twin._surface(X.ravel(), Y.ravel(), t)
    return eta


def low_dt():
    """The source plant's step (Mission's dt in world 'low'), known before
    a Mission exists: the M10 delays are drawn in whole steps of it."""
    from sim import config, lofi
    c = ctx()
    return float(lofi.default_dt(config.scales_for(c["db"], c["h"])))


def draw_actuators(job, rng, world):
    """The episode's actuator overrides (module docstring) and the family
    name. The old draw always consumes the episode's stream exactly as
    before (rng.random(), then two uniforms when it is < 0.75)."""
    from sim import lofi
    c = ctx()
    fam = job.get("act_family", "old")
    act = None
    if world == "low" and rng.random() < 0.75:
        rs = np.sqrt(c["p"]["L"] / 10.0)
        act = dict(tau_thrust=float(rs * rng.uniform(0.0, 2.0)),
                   rud_rate=float(rng.uniform(0.35, 1.05) / rs))
    if job.get("act_params") is not None:
        return dict(job["act_params"]), "params"
    if world != "low" or fam == "old":
        return act, "old"
    if fam == "m10":
        act = lofi.draw_act(np.random.default_rng([int(job["seed"]), 11]),
                            c["p"], low_dt())
    elif fam == "at":
        act = lofi.act_target_like(c["p"], low_dt())
    else:
        raise ValueError(f"act_family {fam!r}: 'old', 'm10' or 'at'")
    return act, fam


def episode(job):
    """job: seed, world ('low' | 'high'), lib (dict, source only),
    relay (bool), informative (None = random), T, act_family ('old' |
    'm10' | 'at'), act_params (dict, tests), raw64 (bool, tests)."""
    from sim import lofi
    c = ctx()
    rng = np.random.default_rng(job["seed"])
    world = job.get("world", "low")
    T = job.get("T", T_EP)
    raw64 = bool(job.get("raw64", False))
    sea = sea_for(rng, target=world == "high")
    act, fam = draw_actuators(job, rng, world)
    m = Mission(world, int(job["seed"]) + 50000, 0, t_end=T, track=0.0,
                sea=sea, plant_params=act)
    new_act = act is not None and act.get("act_family", 0.0) > 0.5
    if new_act and abs(m.dt - low_dt()) > 1e-12:
        raise RuntimeError(f"Mission dt {m.dt} != the dt the delays were "
                           f"drawn for ({low_dt()})")
    op = inj = None
    op_seed = int(job["seed"]) * 7 + 1
    if world == "low" and job.get("lib") is not None:
        # dt: the plant time a control step really covers (sub x dt =
        # 0.24 s; Mission's nominal dt_ctrl is 0.25 s)
        op = Operator(op_seed, job["lib"], dt=m.sub * m.dt, L=m.L,
                      relay=job.get("relay", False))
        inj = ZOHInjector()
        m.residual = inj
    rng_op = np.random.default_rng([op_seed, 1])
    st = None                  # built at the first step, from its inputs
    twin = lofi.plant_for(c["db"], m.sea, c["h"],
                          dt=m.dt if world == "low" else 2 * m.dt,
                          params=act)
    twin_sub = int(round(m.sub * m.dt / twin.dt))
    p = m.red.p
    informative = job.get("informative")
    if informative is None:
        informative = bool(rng.random() > 0.25)
    u_tgt = rng.uniform(16, 30) * KN
    psi_tgt = rng.uniform(-np.pi, np.pi) if informative else 0.0
    t_u = t_psi = 0.0
    integ = 0.0
    dth = dict(on=informative and rng.random() < 0.7,
               tau=np.exp(rng.uniform(np.log(0.5), np.log(5.0))),
               amp=rng.uniform(0, 0.15), x=0.0)
    dnz = dict(on=informative and rng.random() < 0.7,
               tau=np.exp(rng.uniform(np.log(0.5), np.log(5.0))),
               amp=rng.uniform(0, 0.15), x=0.0)
    kp = p["m_surge"] / 3.0
    keys = ("S", "XS", "U", "E", "RR", "RN", "ACG", "D", "AV")
    rec = {k: [] for k in keys + (("R", "CMD") if raw64 else ())}
    ck, slow = [], []
    while not m.done():
        t = m.t
        if informative and t >= t_u:
            u_tgt, t_u = rng.uniform(16, 30) * KN, t + rng.uniform(10, 40)
        if informative and t >= t_psi:
            psi_tgt = rng.uniform(-np.pi, np.pi)
            t_psi = t + rng.uniform(20, 60)
        m.ep.heading_ref = psi_tgt
        s0 = m.s.copy()
        e_u = u_tgt - s0[6]
        integ = float(np.clip(integ + e_u * m.dt_ctrl, -20, 20))
        for d in (dth, dnz):
            if d["on"]:
                d["x"] += (-d["x"] / d["tau"]) * m.dt_ctrl + d["amp"] \
                    * np.sqrt(2 * m.dt_ctrl / d["tau"]) * rng.normal()
        thrust = float(np.clip(p["k_drag"] * u_tgt ** 2 + kp * e_u
                               + 0.1 * kp * integ + dth["x"] * m.t_max,
                               0.0, m.t_max))
        rudder = float(np.clip(m.ep._steer(s0, thrust)
                               + dnz["x"] * m.rud_max, -m.rud_max,
                               m.rud_max))
        x9 = inputs_of(s0, m.u_ref, m.L, m.t_max, m.rud_max)
        zr = ((s0[2] - p.get("z0", 0.0)) / 0.2,
              (s0[4] - p.get("th0", 0.0)) / 0.05)
        w15 = _stations(twin, s0, t)
        s26 = np.concatenate([x9, zr, w15])
        s28 = np.concatenate([s26, [thrust / m.t_max, rudder / m.rud_max]])
        rr = rn = np.zeros(5)
        if op is not None:
            if st is None:
                st = op.new_state(1, rng=rng_op, init_s=s28[None])
            k = len(rec["S"])
            if k >= CK0 and k % CK0 == 0:
                ck.append(k)
                slow.append(op.slow_state(st))
            rr, rn = op.step(st, s28[None], noise=True, rng=rng_op)
            rr, rn = clip_push(rr[0], rn[0])
            inj.r = rr + rn
        av = m.plant.act_vel() if hasattr(m.plant, "act_vel") \
            else np.zeros(2)
        if new_act:
            # the twin starts the step with the plant's own actuator state
            twin.set_act_state(m.plant.act_state())
        stw = s0.copy()
        for kk in range(twin_sub):
            stw = twin.step(stw, t + kk * twin.dt, thrust, rudder, twin.dt)
        m.advance(thrust, rudder)
        if not m.finite:
            break
        dtc = m.sub * m.dt
        y = np.array([(m.s[i] - stw[i]) / dtc for i in VEL_IDX])
        rec["S"].append(s26)
        rec["XS"].append(s0)
        rec["U"].append((thrust / m.t_max, rudder / m.rud_max))
        rec["E"].append(y)
        rec["RR"].append(rr)
        rec["RN"].append(rn)
        rec["ACG"].append(m.last_acg)
        rec["D"].append((dth["x"], dnz["x"]))
        rec["AV"].append(av)
        if raw64:
            rec["R"].append(inj.r.copy() if inj is not None else np.zeros(5))
            rec["CMD"].append((thrust, rudder))
    out = {k: np.asarray(v, np.float64 if raw64 else np.float32)
           for k, v in rec.items()}
    out.update(finite=bool(m.finite), world=world, seed=int(job["seed"]),
               act=act, act_family=fam,
               op_seed=op_seed, relay=bool(job.get("relay", False)),
               informative=bool(informative), sea=sea,
               ck=np.asarray(ck, np.int32), slow=slow,
               style=None if op is None else op.style)
    return out


# ------------------------------------------------------------ library
def op_inputs(S, U):
    """The operator's 28 inputs: the 26 measured ones and the commands."""
    return np.concatenate([S, U], -1)


def build_library(eps, n_seq=16, T=250, start=60):
    """Normalisation (mu, sd of the operator's 28 inputs over all steps
    after 10 s) and a library of n_seq normalised input histories of T
    steps, from operator-free pilot episodes."""
    X = [op_inputs(e["S"], e["U"]) for e in eps]
    Sall = np.concatenate([x[40:] for x in X if len(x) > 60])
    mu, sd = Sall.mean(0), Sall.std(0) + 1e-6
    seqs = [x[start:start + T] for x in X if len(x) >= start + T][:n_seq]
    S = (np.stack(seqs) - mu) / sd
    return dict(mu=mu.astype(np.float64), sd=sd.astype(np.float64),
                S=S.astype(np.float64))


def push_gain(n_states=24, seed=0):
    """G (5 x 5): the observed one-step error e per unit push r held over
    one control step, e = G r. Measured on the source plant (the reduced
    model) by stepping it with and without the push from a spread of
    states, as the closed loop does. Heave and pitch are linear with
    constant coefficients, so their gains are exact. The sway response to
    a yaw push is the Coriolis term m u r and grows with speed, so G is
    fitted as G0 + G1 (u / u_ref - 1) (u / u_ref - 1 is input 0 of S).
    Returns (G (2, 5, 5) = [G0, G1], rms misfit of that fit (5, 5)).
    The probes restart the plant from arbitrary states, which is valid only
    because this Mission's plant has ideal old-style actuators: an M10
    plant carries a delay line and servo velocities between steps
    (lofi.ReducedPlant.act_state) that a probe would have to save and
    restore."""
    from sim import lofi
    c = ctx()
    rng = np.random.default_rng(seed)
    Gs, du = [], []
    for i in range(n_states):
        m = Mission("low", 900000 + i, 0, t_end=5.0, track=0.0,
                    sea=sea_for(rng))
        s = m.s.copy()
        s[6] = rng.uniform(16, 30) * KN
        s[5] = rng.uniform(-np.pi, np.pi)
        s[7], s[11] = rng.normal(0, 0.3), rng.normal(0, 0.1)
        thr, rud = 0.6 * m.t_max, rng.uniform(-0.5, 0.5) * m.rud_max
        t = rng.uniform(0, 50)
        G = np.zeros((5, 5))
        base = None
        for col in range(-1, 5):
            r = np.zeros(5)
            if col >= 0:
                r[col] = 1.0
            x = s.copy()
            for k in range(m.sub):
                x = m.plant.step(x, t + k * m.dt, thr, rud, m.dt)
                for ch, ix in enumerate(VEL_IDX):
                    x[ix] += r[ch] * m.dt
            v = np.array([x[ix] for ix in VEL_IDX])
            if col < 0:
                base = v
            else:
                G[:, col] = (v - base) / (m.sub * m.dt)
        Gs.append(G)
        du.append(s[6] / m.u_ref - 1.0)
    Gs, du = np.array(Gs), np.array(du)
    Xr = np.stack([np.ones_like(du), du], 1)
    coef, *_ = np.linalg.lstsq(Xr, Gs.reshape(len(du), -1), rcond=None)
    fit = (Xr @ coef).reshape(Gs.shape)
    return coef.reshape(2, 5, 5), np.sqrt(((Gs - fit) ** 2).mean(0))


def plant_to_reduced(xs14):
    """ReducedModel.from_plant_state, vectorised: (N, 14) -> (N, 10)."""
    x = np.asarray(xs14, float)
    return np.stack([x[:, 0], x[:, 1], x[:, 6], x[:, 2], x[:, 8], x[:, 4],
                     x[:, 10], x[:, 5], x[:, 11], x[:, 7]], 1)


RED_VEL = (2, 9, 8, 4, 6)       # u, v, r, heave rate, pitch rate (reduced)


def push_response(red, xs14, U, r, t_max, rud_max, dt, sub, x_st):
    """The observed error e (N, 5) that pushes r (N, 5), held over one
    control step, produce from plant states xs14 (N, 14) under commands U
    (N, 2): the source plant (the reduced model) stepped with and without
    the push, kicked after every substep as ZOHInjector does. Waves enter
    the model additively, so they cancel in the difference and are set to
    zero. Exact, including the speed- and yaw-rate-dependent cross terms
    (Coriolis) that a fixed gain matrix misses."""
    a = plant_to_reduced(xs14)
    b = a.copy()
    N = len(a)
    eta0 = np.zeros((N, len(x_st), 3))
    thr, rud = U[:, 0] * t_max, U[:, 1] * rud_max
    for _ in range(sub):
        a = red.step(a, thr, rud, eta0, x_st, dt)[0]
        b = red.step(b, thr, rud, eta0, x_st, dt)[0]
        for ch, col in enumerate(RED_VEL):
            a[:, col] = a[:, col] + r[:, ch] * dt
    return (a[:, list(RED_VEL)] - b[:, list(RED_VEL)]) / (sub * dt)


def apply_gain(G, r, du):
    """e = (G0 + G1 du) r; r (..., 5), du (...,) -> (..., 5)."""
    return r @ G[0].T + du[..., None] * (r @ G[1].T)


# ------------------------------------------------------------ packing
def pack(eps, T_max=None):
    """List of episode dicts -> dict of padded arrays + per-episode meta."""
    eps = [e for e in eps if len(e["E"]) > H + 10]
    T_max = T_max or max(len(e["E"]) for e in eps)
    n = len(eps)
    out = {}
    for k, d in (("S", 26), ("XS", 14), ("U", 2), ("E", 5), ("RR", 5),
                 ("RN", 5), ("ACG", None), ("D", 2), ("AV", 2)):
        shape = (n, T_max) if d is None else (n, T_max, d)
        a = np.zeros(shape, np.float32)
        for i, e in enumerate(eps):
            if k not in e:                  # AV: episodes made before M10
                continue
            L = min(len(e[k]), T_max)
            a[i, :L] = e[k][:L]
        out[k] = a
    out["len"] = np.array([min(len(e["E"]), T_max) for e in eps], np.int32)
    n_ck = (T_max - H) // CK0
    CK = np.full((n, n_ck), -1, np.int32)
    for i, e in enumerate(eps):
        for j in range(n_ck):
            k = CK0 * (j + 1)
            if k + H <= out["len"][i]:
                CK[i, j] = k
    out["CK"] = CK
    meta = [dict(seed=e["seed"], op_seed=e["op_seed"], relay=e["relay"],
                 act=e.get("act"), act_family=e.get("act_family", "old"),
                 informative=e["informative"], sea=e["sea"], world=e["world"],
                 finite=e["finite"], style=e["style"],
                 slow={int(k): s for k, s in zip(e["ck"], e["slow"])})
            for e in eps]
    return out, meta


# ------------------------------------------------------------ targets
def _reduced_rollout(red, xs14, U, t_max, rud_max, dt, sub, x_st, u_ref, L):
    """The MPC model's rollout from plant states xs14 (N, 14) under
    commands U (N, H, 2), with no waves: the 11 horizon inputs (N, H, 11)
    (inputs_of's 9 + heave and pitch about the running attitude), each the
    state BEFORE horizon step j, as S holds it. The thrust and nozzle
    columns are the actuator state: the measured one at j = 0, then the
    previous planned command (ideal actuators, the MPC's convention; exact
    in the source world, where S[k, 5:7] = U[k - 1])."""
    N, Hh, _ = U.shape
    p = red.p
    sr = np.stack([red.from_plant_state(x) for x in xs14])
    eta0 = np.zeros((N, len(x_st), 3))
    out = np.zeros((N, Hh, 11), np.float32)
    for j in range(Hh):
        thr = U[:, j, 0] * t_max
        rud = U[:, j, 1] * rud_max
        x, y, u, z, zd, th, thd, psi, r, v = sr.T
        if j == 0:
            a_thr, a_noz = xs14[:, 12] / t_max, xs14[:, 13] / rud_max
        else:
            a_thr, a_noz = U[:, j - 1, 0], U[:, j - 1, 1]
        out[:, j] = np.stack([u / u_ref - 1.0, v, r * L / u_ref,
                              np.cos(psi), np.sin(psi), a_thr, a_noz,
                              zd, thd,
                              (z - p.get("z0", 0.0)) / 0.2,
                              (th - p.get("th0", 0.0)) / 0.05], 1)
        for _ in range(sub):
            sr = red.step(sr, thr, rud, eta0, x_st, dt)[0]
    return out


_W = {}


def _targets_init(path, lib, G):
    d = np.load(path, allow_pickle=True)
    _W.update(S=d["S"], XS=d["XS"], U=d["U"], RR=d["RR"], len=d["len"],
              CK=d["CK"], lib=lib, G=G)
    import pickle
    _W["meta"] = pickle.load(open(path.replace(".npz", "_meta.pkl"), "rb"))


def _targets_one(i):
    from sim import lofi
    W = _W
    c = ctx()
    meta = W["meta"][i]
    if "mission" not in W:
        W["mission"] = Mission("low", 1, 0, t_end=5.0, track=0.0)
        W["twin"] = lofi.plant_for(c["db"], W["mission"].sea, c["h"],
                                   dt=W["mission"].dt)
    m, twin = W["mission"], W["twin"]
    rng = np.random.default_rng([meta["seed"], 99])
    CK = W["CK"][i]
    n_ck = len(CK)
    Y = np.zeros((n_ck, M_DON + 1, H, 5), np.float32)
    XH = np.zeros((n_ck, M_DON + 1, H, 11), np.float32)
    REF = np.full((n_ck, M_DON + 1, 2), -1, np.int32)
    src = meta["world"] == "low" and meta["style"] is not None
    op = (Operator(meta["op_seed"], W["lib"], dt=m.sub * m.dt, L=m.L,
                   relay=meta["relay"]) if src else None)
    n = len(W["len"])
    for c_, k in enumerate(CK):
        if k < 0:
            continue
        refs = [(i, k)]
        if src:
            for _ in range(M_DON):
                while True:
                    j = int(rng.integers(n))
                    if j != i and W["len"][j] >= WARM + H + 1:
                        break
                t0 = int(rng.integers(WARM, W["len"][j] - H + 1))
                refs.append((j, t0))
        REF[c_, :len(refs)] = refs
        xs = np.stack([W["XS"][j, t0] for j, t0 in refs])
        Us = np.stack([W["U"][j, t0:t0 + H] for j, t0 in refs])
        XH[c_, :len(refs)] = _reduced_rollout(
            m.red, xs, Us, m.t_max, m.rud_max, m.dt, m.sub, twin.x_st,
            m.u_ref, m.L)
        if not src:
            continue
        pr = dict(red=m.red, t_max=m.t_max, rud_max=m.rud_max, dt=m.dt,
                  sub=m.sub, x_st=twin.x_st)
        lim = CLIP * A_REF
        slow = meta["slow"][int(k)]
        # the own future, relabelled exactly as the donors: the fading state
        # from rest where the model's bank starts (k - WARM, or step 0), the
        # slow state frozen at the checkpoint, relays held until k, the rule
        # clipped alone
        a0 = max(0, k - WARM)
        R0 = np.clip(op.run(op_inputs(W["S"][i:i + 1, a0:k + H],
                                      W["U"][i:i + 1, a0:k + H]), slow=slow,
                            noise=False, hold=k - a0)[0, k - a0:], -lim, lim)
        Y[c_, 0] = push_response(r=R0, xs14=W["XS"][i, k:k + H],
                                 U=W["U"][i, k:k + H], **pr)
        Sd = np.stack([op_inputs(W["S"][j, t0 - WARM:t0 + H],
                                 W["U"][j, t0 - WARM:t0 + H])
                       for j, t0 in refs[1:]])
        R = np.clip(op.run(Sd, slow=slow, noise=False, hold=WARM), -lim, lim)
        XSd = np.stack([W["XS"][j, t0:t0 + H] for j, t0 in refs[1:]])
        Ud = np.stack([W["U"][j, t0:t0 + H] for j, t0 in refs[1:]])
        Y[c_, 1:] = push_response(
            r=R[:, WARM:].reshape(-1, 5), xs14=XSd.reshape(-1, 14),
            U=Ud.reshape(-1, 2), **pr).reshape(len(refs) - 1, H, 5)
    return i, Y, XH, REF


def build_targets(path, lib, G, procs=5, code=""):
    """Targets for every episode of a packed split (see module docstring).
    Writes <path>_targets.npz: SEG_Y (n, n_ck, M+1, H, 5), SEG_XH
    (n, n_ck, M+1, H, 11), SEG_REF (n, n_ck, M+1, 2) = (episode, start)."""
    from multiprocessing import Pool
    d = np.load(path, allow_pickle=True)
    n, n_ck = d["CK"].shape
    Y = np.zeros((n, n_ck, M_DON + 1, H, 5), np.float32)
    XH = np.zeros((n, n_ck, M_DON + 1, H, 11), np.float32)
    REF = np.full((n, n_ck, M_DON + 1, 2), -1, np.int32)
    del d
    with Pool(procs, initializer=_targets_init,
              initargs=(path, lib, G)) as pool:
        for i, y, xh, ref in pool.imap_unordered(_targets_one, range(n),
                                                 chunksize=8):
            Y[i], XH[i], REF[i] = y, xh, ref
    out = path.replace(".npz", "_targets.npz")
    import pickle
    seeds = np.array([m["seed"] for m in pickle.load(open(
        path.replace(".npz", "_meta.pkl"), "rb"))])
    tmp = out + ".tmp.npz"
    np.savez(tmp, SEG_Y=Y, SEG_XH=XH, SEG_REF=REF, seeds=seeds,
             code=np.array(code))
    os.replace(tmp, out)
    return out
