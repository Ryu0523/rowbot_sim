#!/usr/bin/env python3
"""
Episodes of the D10a force-catalogue error family (M17 pipeline, cache
meta7; learn/meta/PRIOR_D10_DRAFT.md, PRIOR_DERIVATION.md D10): everything
the meta7 data chunks are made from, so studies/meta_step7.code7() (the
stamp of every chunk and pack) hashes this module, the family modules
(operators_cat.py, cat_base.py and every item module learn/meta/cat_*.py),
the meta6 modules imported read-only and meta_step2.SOURCES. The branches
live in learn/meta/data7.py, outside that stamp.

family 'cat' (the job key 'family'): operators_cat.OperatorCat, three
layers (D8.12 rigid body with the overlap switches, the catalogue items,
the D9 residual at the declared shares).

What an episode is. episode6.episode with the catalogue in the loop and
nothing else changed: the same random stream in the same order (sea, the
M10 actuator draw from its own stream [seed, 11], scenario, dithers), so an
episode of seed s has the sea, actuators and scenario of the meta5 / meta6
episode of seed s; the same Mission, twin and records. The differences:
  - the plant wrapper is operators_cat.CatPlant (an RBPlant that also keeps
    each substep's start time and commands; bit-identical steps) and the
    Mission's residual hook is CatSafetyHook(CatInjector): the catalogue
    substep (state feedback at every 0.04 s substep, with its own sample
    of the plant's sea incl. the wave orbital velocity and the intake
    elevation), clip_push with the held control-step part (layer 1's old
    noise; the rule is 0), the velocity kick, then the IMPULSE channel
    (slams W3, debris E3; bounded per control step by the clip limit x
    0.24 s); the operator gets the episode's M10 actuator dict (the true-
    actuator copy a P12 rule needs);
  - a divergence truncates the episode (div), as episode6; the draw was
    acceptance-tested (divergence only) at construction and is never
    redrawn inside an episode;
  - extra per-step arrays: APK / AMIN / HMIN (episode6's definitions; the
    impulse is part of the vertical acceleration), PUSH (the mean clipped
    push), IMP (5, the impulse channel's mean acceleration over the step),
    SLAM (slams of the low-fidelity rule at the bow station in the step)
    and OBS (8, OBS_EXTRA = roll, roll rate, rpm, relative wind speed and
    angle, speed through water u / v, GNSS quality at the step's first
    substep): the draft section 11 signals, STORED only (D10a keeps the
    meta5 network inputs; the observation model O1-O5 is off);
  - snapshots as episode6 (operator state incl. every item's hidden state
    and generator, its noise stream, the plant's M10 actuator state, plant
    state, time).

The recorded arrays are episode5's / episode6's in the same formats, so
model_preview and studies/eval_preview_m15.py read the packs unchanged.
Meta (pack7): op_family 'm15' (the wave-input FORMAT), prior_family 'cat',
op_seed renamed op_seed_cat (every old rebuild path raises a KeyError).

No torch here: the episode and branch workers stay numpy-only.
"""
import copy

import numpy as np

from learn.meta import data2
from learn.meta import episode6 as E6
from learn.meta.data2 import (CK0, KN, T_EP, _stations, clip_push,
                              draw_actuators, low_dt, sea_for)
from learn.repro.task import Mission, ctx

FAMILIES7 = ("cat",)
SAFETY = E6.SAFETY
EXTRA7 = SAFETY + ("PUSH", "IMP", "SLAM", "OBS")
DIV_U, DIV_TH = E6.DIV_U, E6.DIV_TH


def obs_names():
    from learn.meta.operators_cat import OBS_EXTRA
    return OBS_EXTRA


# ------------------------------------------------------------ family
def make_op(family, op_seed, lib, dtc, L, relay=False, act=None):
    """The family's operator for a seed (deterministic: the acceptance test
    and its redraws run from the seed)."""
    if family != "cat":
        raise ValueError(f"family {family!r}: one of {FAMILIES7}")
    from learn.meta.operators_cat import OperatorCat
    return OperatorCat(op_seed, lib, dt=dtc, L=L, relay=relay, act=act)


def wrap_plant(plant):
    from learn.meta.operators_cat import cat_plant
    return cat_plant(plant)


def make_injector(family, op, st):
    from learn.meta.operators_cat import CatInjector
    return CatInjector(op, st)


class CatSafetyHook(E6.SafetyHook):
    """episode6.SafetyHook plus, per substep, the impulse channel's
    acceleration (imp), the extra observed signals (obs) and the slam
    flag (slam)."""

    def __init__(self, p, x_st, inj=None):
        super().__init__(p, x_st, inj)
        self.imp, self.obs, self.slam = [], [], []

    def __call__(self, s, dt, plant=None):
        s = super().__call__(s, dt, plant)
        self.imp.append(np.asarray(self.inj.last_imp, float).copy())
        self.obs.append(np.array([self.inj.obs[n] for n in obs_names()]))
        self.slam.append(float(self.inj.slam))
        return s


op_inputs6 = E6.op_inputs6
diverged = E6.diverged


def style7(op):
    return dict(op.style)


# ------------------------------------------------------------ episode
def episode(job):
    """episode6.episode with the catalogue in the loop (module docstring).
    Job keys as episode6 (seed, relay, T, act_family, act_params,
    informative, raw64, lib, family, snaps). Source world only."""
    from sim import lofi
    fam = job.get("family")
    if fam not in FAMILIES7:
        raise ValueError(f"family {fam!r}: one of {FAMILIES7}")
    world = job.get("world", "low")
    if world != "low" or job.get("lib") is None:
        raise ValueError("episode7: source episodes with a library only")
    c = ctx()
    rng = np.random.default_rng(job["seed"])
    T = job.get("T", T_EP)
    raw64 = bool(job.get("raw64", False))
    snap_at = {int(k) for k in job.get("snaps", ())}
    sea = sea_for(rng, target=False)
    act, act_fam = draw_actuators(job, rng, world)
    m = Mission(world, int(job["seed"]) + 50000, 0, t_end=T, track=0.0,
                sea=sea, plant_params=act)
    new_act = act is not None and act.get("act_family", 0.0) > 0.5
    if new_act and abs(m.dt - low_dt()) > 1e-12:
        raise RuntimeError(f"Mission dt {m.dt} != the dt the delays were "
                           f"drawn for ({low_dt()})")
    dtc = m.sub * m.dt
    op_seed = int(job["seed"]) * 7 + 1
    op = make_op(fam, op_seed, job["lib"], dtc, m.L,
                 relay=job.get("relay", False),
                 act=act if new_act else None)
    plant = wrap_plant(m.plant)
    m.plant = m.ep.plant = plant
    p = m.red.p
    hook = CatSafetyHook(p, plant.x_st)
    m.residual = hook
    rng_op = np.random.default_rng([op_seed, 1])
    st = None                  # built at the first step, from its inputs
    twin = lofi.plant_for(c["db"], m.sea, c["h"], dt=m.dt, params=act)
    twin_sub = int(round(m.sub * m.dt / twin.dt))
    h_mid = 0.5 * dtc
    cst = dict(u_ref=m.u_ref, L=m.L, t_max=m.t_max, rud_max=m.rud_max,
               z0=p.get("z0", 0.0), th0=p.get("th0", 0.0))
    u_max = float(p["u_max"])
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
    keys = ("S", "XS", "U", "E", "RR", "RN", "ACG", "D", "AV", "W_MID",
            "PUSH", "IMP", "SLAM", "OBS")
    rec = {k: [] for k in keys + (("R", "CMD") if raw64 else ())}
    ck, slow, snaps = [], [], {}
    div = -1
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
        cmd = [thrust / m.t_max, rudder / m.rud_max]
        s26, w_mid, s28 = op_inputs6(s0, t, twin, cst, h_mid, cmd)
        k = len(rec["S"])
        if st is None:
            st = op.new_state(1, rng=rng_op, init_s=s28[None])
            hook.inj = make_injector(fam, op, st)
        if k >= CK0 and k % CK0 == 0:
            ck.append(k)
            slow.append(op.slow_state(st))
        if k in snap_at:
            snaps[k] = dict(k=k, t=float(t), xs=s0.copy(),
                            st=op.snapshot(st), rng=copy.deepcopy(rng_op),
                            act=plant.act_state())
        rr, rn = op.step(st, s28[None], noise=True, rng=rng_op)
        rr, rn = clip_push(rr[0], rn[0])
        hook.set_held(rr, rn)
        av = m.plant.act_vel() if hasattr(m.plant, "act_vel") \
            else np.zeros(2)
        if new_act:
            twin.set_act_state(m.plant.act_state())
        stw = s0.copy()
        for kk in range(twin_sub):
            stw = twin.step(stw, t + kk * twin.dt, thrust, rudder, twin.dt)
        n0 = len(hook.push)
        m.advance(thrust, rudder)
        if not m.finite or diverged(m.s, u_max):
            div = k
            break
        y = np.array([(m.s[i] - stw[i]) / dtc for i in data2.VEL_IDX])
        rec["S"].append(s26)
        rec["XS"].append(s0)
        rec["U"].append(cmd)
        rec["E"].append(y)
        rec["RR"].append(rr)
        rec["RN"].append(rn)
        rec["ACG"].append(m.last_acg)
        rec["D"].append((dth["x"], dnz["x"]))
        rec["AV"].append(av)
        rec["W_MID"].append(w_mid)
        rec["PUSH"].append(np.mean(hook.push[n0:], 0))
        rec["IMP"].append(np.mean(hook.imp[n0:], 0))
        rec["SLAM"].append(float(np.sum(hook.slam[n0:])))
        rec["OBS"].append(hook.obs[n0])
        if raw64:
            rec["R"].append(np.mean(hook.push[n0:], 0))
            rec["CMD"].append((thrust, rudder))
    ft = np.float64 if raw64 else np.float32
    n, sub = len(rec["S"]), m.sub
    out = {k_: np.asarray(v, ft) for k_, v in rec.items()}
    n_obs = len(obs_names())
    if n == 0:
        for k_, w in (("S", 26), ("XS", 14), ("U", 2), ("E", 5), ("RR", 5),
                      ("RN", 5), ("D", 2), ("AV", 2), ("W_MID", 15),
                      ("PUSH", 5), ("IMP", 5), ("OBS", n_obs), ("R", 5),
                      ("CMD", 2)):
            if k_ in out:
                out[k_] = np.zeros((0, w), ft)
    hs = list(hook.h)
    if len(hs) < n * sub + 1:
        eta = _stations(twin, m.s, m.t)
        sr = data2.plant_to_reduced(m.s[None])[0]
        hs.append(hook.bow(sr, eta))
    az = np.asarray(hook.az[:n * sub], float).reshape(n, sub)
    hh = np.asarray(hs, float)
    out["APK"] = az.max(1).astype(ft) if n else np.zeros(0, ft)
    out["AMIN"] = az.min(1).astype(ft) if n else np.zeros(0, ft)
    out["HMIN"] = np.array([hh[q * sub + 1:(q + 1) * sub + 1].min()
                            for q in range(n)], ft)
    out.update(finite=bool(m.finite), world=world, seed=int(job["seed"]),
               act=act, act_family=act_fam, op_seed=op_seed,
               relay=bool(job.get("relay", False)),
               informative=bool(informative), sea=sea,
               ck=np.asarray(ck, np.int32), slow=slow,
               style=style7(op), family=fam, div=int(div),
               snaps=snaps)
    return out


_LIB7 = {}


def run_job(job):
    """A worker's episode (studies/meta_step7.py hands this to
    meta_step2.run_jobs): job['lib_path'] is loaded once per process; an
    exception comes back as dict(error, seed) so the chunk goes on."""
    j = dict(job)
    path = j.pop("lib_path", None)
    if j.pop("use_lib", False):
        if _LIB7.get("path") != path:
            d = np.load(path)
            _LIB7.update(path=path, lib=dict(mu=d["mu"], sd=d["sd"],
                                             S=d["S"]))
        j["lib"] = dict(_LIB7["lib"])
    try:
        return episode(j)
    except Exception as ex:                       # keep the chunk going
        import traceback
        return dict(error=repr(ex) + " | " + traceback.format_exc()[-600:],
                    seed=job["seed"])


# ------------------------------------------------------------ packing
def pack7(eps, T_max=None):
    """data2.pack plus W_MID (n, T, 15), APK / AMIN / HMIN / SLAM (n, T),
    PUSH / IMP (n, T, 5) and OBS (n, T, 8), float32, zero beyond len; one
    family per pack. Meta: op_family 'm15', prior_family 'cat', op_seed
    renamed op_seed_cat, div, obs_names. Returns (arrays, meta, snaps)."""
    kept = [e for e in eps if len(e["E"]) > data2.H + 10]
    fams = {e.get("family") for e in kept}
    if len(fams) > 1 or not fams <= set(FAMILIES7):
        raise ValueError(f"pack7: families {sorted(map(str, fams))}; the "
                         "catalogue family only")
    out, meta = data2.pack(eps, T_max)
    n, T = out["E"].shape[:2]
    n_obs = len(obs_names())
    for k, w in (("W_MID", 15), ("APK", None), ("AMIN", None),
                 ("HMIN", None), ("PUSH", 5), ("IMP", 5), ("SLAM", None),
                 ("OBS", n_obs)):
        a = np.zeros((n, T) if w is None else (n, T, w), np.float32)
        for i, e in enumerate(kept):
            L = min(len(e[k]), T)
            a[i, :L] = e[k][:L]
        out[k] = a
    snaps = {}
    for i, (m_, e) in enumerate(zip(meta, kept)):
        m_["op_seed_cat"] = m_.pop("op_seed")
        m_.update(op_family="m15", prior_family="cat", div=int(e["div"]),
                  obs_names=list(obs_names()))
        if e.get("snaps"):
            snaps[i] = e["snaps"]
    return out, meta, snaps
