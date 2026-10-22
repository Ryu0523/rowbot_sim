#!/usr/bin/env python3
"""
Several missions of the same boat (ROADMAP_2026-09-30 section 11.3: the
pretraining data the memory structures need): episode7's catalogue
episodes, with the episode's randomness split into what belongs to the
BOAT and what belongs to the MISSION.

  boat (law)    the error law (operators_cat.OperatorCat of op_seed =
                7 law + 1: the catalogue draw, which items are on, every
                item's hull / machinery parameters, layer 1 and layer 3's
                draws) and the M10 actuators (stream [law, 11]); faults
                keep a boat-lifetime clock: a fault (P8, P10, P11, P12)
                whose onset t_f fell in an earlier mission is present from
                the start of the later ones
  mission (mi)  the sea, the scenario and dithers (stream [law, mi, 8,
                TAG_SEA]); the Mission seed (law + 50000 + MISSION_STRIDE
                mi: the wave phases); the operator's noise (stream [op_seed,
                1, mi, TAG_OP]); every item's process stream and hidden-
                state start and layer 3's hidden processes (hid_seed
                HID0 + mi); layer 1's slow processes (stream [seed, 2, mi,
                TAG_SLOW]); the ENVIRONMENT parameters of the environment
                items that are on (ENV_KEYS: wind speed and direction, gust
                and squall statistics, current, water depth and bank,
                debris and wave-packet rates; stream [op_seed, TAG_ENV,
                attempt, 900 + item index, mi]) and layer 3's earth-fixed
                direction psi_w. Whether an environment item is on at all
                stays a property of the boat's draw (a limitation: a boat
                drawn without wind never meets wind). The acceptance test
                of the draw ran with mission 0's environment only; a later
                mission whose environment makes the boat diverge is
                truncated as any diverging episode.

Mission 0 uses episode7's streams and parameters exactly (rng seed law,
Mission seed law + 50000, operator stream [op_seed, 1], hid_seed 0, no
redraws, clocks at 0), so mission 0 of boat law IS episode7's episode of
seed law (studies/meta_step8.py reuses meta7's chunks for it after checking
the identity on a few boats per code version). The episode's id, the key of the
chunk files and of meta_step2.run_jobs, is episode_id(law, mi) = law +
MISSION_STRIDE mi (= law for mission 0).

Everything else is episode7's: imported, not copied, except episode()
(below: episode7.episode with the streams above, mission_env before the
operator state is built and mission_state after). Outputs carry law, mi,
hid_seed and env_redraw; episode7.pack7 packs them unchanged.

No torch here: the workers stay numpy-only.
"""
import copy

import numpy as np

from learn.meta import data2
from learn.meta.data2 import (CK0, KN, T_EP, _stations, clip_push,
                              draw_actuators, low_dt, sea_for)
from learn.meta.episode7 import (FAMILIES7, CatSafetyHook, diverged,
                                 make_injector, make_op, obs_names,
                                 op_inputs6, style7, wrap_plant)
from learn.meta.episode7 import pack7   # noqa: F401  (meta_step8's packer)
from learn.repro.task import Mission, ctx

MISSION_STRIDE = 1_000_000
HID0 = 1000                 # hid_seed of mission mi > 0: HID0 + mi (clear of
                            # operators_gen TEST_HID 47 and LIB_HID 53)
TAG_SEA, TAG_OP, TAG_ENV, TAG_SLOW = 0x5EA, 0x0B5, 0xE7, 0x51
# the environment keys of the environment items (None: every key); the
# other keys of these items (hull areas and shape, the shallow-water hull
# response, the bank force coefficients, debris damage) belong to the boat
ENV_KEYS = dict(A1=("U10", "psi_u", "wave_rel", "dpsi", "tau_d"),
                A2=None, E1=None,
                E2=("h_bar", "s_h", "l_h", "d_bar", "s_d", "l_d", "side",
                    "bank"),
                E3=("rate",), E4=("rate", "own", "p_trig"))


def episode_id(law, mi):
    return int(law) + MISSION_STRIDE * int(mi)


def mission_env(op, op_seed, mi):
    """Redraw, for mission mi > 0, the environment parameters of the
    environment items that are on and layer 3's earth-fixed direction
    (module docstring); returns {item: {key: value}} of what changed."""
    from learn.meta import cat_base as CB
    if mi == 0:
        return {}
    cat, out = op.cat, {}
    for c, keys in ENV_KEYS.items():
        if c not in cat.on:
            continue
        r = np.random.default_rng([op_seed, TAG_ENV, int(cat.attempt),
                                   900 + CB.CODES.index(c), int(mi)])
        fresh = CB.REGISTRY[c].draw_on(r, cat.ctx)
        ks = tuple(fresh) if keys is None else keys
        for k in ks:
            cat.on[c][k] = fresh[k]
        out[c] = {k: fresh[k] for k in ks}
    r = np.random.default_rng([op_seed, TAG_ENV, int(cat.attempt), 999,
                               int(mi)])
    cat.op_gen.psi_w = float(r.uniform(0, 2 * np.pi))
    out["psi_w"] = cat.op_gen.psi_w
    return out


def mission_state(op, st, mi, T):
    """For mission mi > 0, after new_state: layer 1's slow processes on a
    mission stream (their start redrawn as operators.Operator.new_state
    draws it) and the fault items' boat-lifetime clocks (module
    docstring)."""
    from learn.meta import cat_base as CB
    if mi == 0:
        return
    l1 = st.get("l1")
    if l1 is not None and l1["res"].get("rng_slow") is not None:
        res = op.cat.op_rb.res
        rs = np.random.default_rng([int(res.seed), 2, int(mi), TAG_SLOW])
        l1["res"]["rng_slow"] = rs
        B = len(np.atleast_1d(l1["res"].get("ga", np.zeros(1))))
        if res.gain_ou is not None:
            l1["res"]["ga"] = rs.normal(0, res.gain_ou[1], B)
        if res.bias_ou is not None:
            l1["res"]["bb"] = rs.normal(0, 1, (B, 5)) * res.bias_ou[1] \
                * res.A
    for c, ist in st["items"].items():
        if getattr(CB.REGISTRY[c], "clock", False):
            ist["t"] = float(mi) * float(T)


def episode(job):
    """episode7.episode for mission `mi` of boat `law` (module docstring).
    Job keys as episode7 plus law, mi; job['seed'] is the episode's id
    (episode_id(law, mi))."""
    from sim import lofi
    fam = job.get("family")
    if fam not in FAMILIES7:
        raise ValueError(f"family {fam!r}: one of {FAMILIES7}")
    world = job.get("world", "low")
    if world != "low" or job.get("lib") is None:
        raise ValueError("episode7: source episodes with a library only")
    c = ctx()
    law, mi = int(job["law"]), int(job["mi"])
    if int(job["seed"]) != episode_id(law, mi):
        raise ValueError(f"seed {job['seed']} != episode_id({law}, {mi})")
    # the mission's own stream: sea, the scenario, the dithers (mission 0:
    # episode7's stream of seed = law, so it is episode7's episode)
    rng = np.random.default_rng(law if mi == 0 else [law, mi, 8, TAG_SEA])
    T = job.get("T", T_EP)
    raw64 = bool(job.get("raw64", False))
    snap_at = {int(k) for k in job.get("snaps", ())}
    sea = sea_for(rng, target=False)
    # the boat's actuators: the M10 draw comes from the stream [law, 11]
    # (draw_actuators reads job['seed']); the old draw still consumes the
    # mission's stream as in episode7
    if job.get("act_params") is None and \
            job.get("act_family") not in ("m10", "at"):
        raise ValueError("episode8: the boat's actuators need act_family "
                         "'m10' or 'at' (the old draw uses the mission "
                         "stream)")
    act, act_fam = draw_actuators(dict(job, seed=law), rng, world)
    m = Mission(world, law + 50000 + MISSION_STRIDE * mi, 0, t_end=T,
                track=0.0, sea=sea, plant_params=act)
    new_act = act is not None and act.get("act_family", 0.0) > 0.5
    if new_act and abs(m.dt - low_dt()) > 1e-12:
        raise RuntimeError(f"Mission dt {m.dt} != the dt the delays were "
                           f"drawn for ({low_dt()})")
    dtc = m.sub * m.dt
    op_seed = law * 7 + 1                  # the boat's error law
    op = make_op(fam, op_seed, job["lib"], dtc, m.L,
                 relay=job.get("relay", False),
                 act=act if new_act else None)
    plant = wrap_plant(m.plant)
    m.plant = m.ep.plant = plant
    p = m.red.p
    hook = CatSafetyHook(p, plant.x_st)
    m.residual = hook
    # the operator's noise and hidden-state start: per mission
    rng_op = np.random.default_rng([op_seed, 1] if mi == 0
                                   else [op_seed, 1, mi, TAG_OP])
    hid_seed = 0 if mi == 0 else HID0 + mi
    env_redraw = mission_env(op, op_seed, mi)
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
            st = op.new_state(1, rng=rng_op, init_s=s28[None],
                              hid_seed=hid_seed)
            mission_state(op, st, mi, T)
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
               law=law, mi=mi, hid_seed=hid_seed, env_redraw=env_redraw,
               act=act, act_family=act_fam, op_seed=op_seed,
               relay=bool(job.get("relay", False)),
               informative=bool(informative), sea=sea,
               ck=np.asarray(ck, np.int32), slow=slow,
               style=style7(op), family=fam, div=int(div),
               snaps=snaps)
    return out


_LIB8 = {}


def run_job(job):
    """A worker's episode (meta_step2.run_jobs): job['lib_path'] loaded once
    per process; an exception comes back as dict(error, seed)."""
    j = dict(job)
    path = j.pop("lib_path", None)
    if j.pop("use_lib", False):
        if _LIB8.get("path") != path:
            d = np.load(path)
            _LIB8.update(path=path, lib=dict(mu=d["mu"], sd=d["sd"],
                                             S=d["S"]))
        j["lib"] = dict(_LIB8["lib"])
    try:
        return episode(j)
    except Exception as ex:                       # keep the chunk going
        import traceback
        return dict(error=repr(ex) + " | " + traceback.format_exc()[-600:],
                    seed=job["seed"])
