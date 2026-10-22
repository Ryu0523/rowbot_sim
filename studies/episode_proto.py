#!/usr/bin/env python3
"""
episode8's multi-mission catalogue episodes with two factors of the
train/test protocol switchable (studies/protocol_check.py):

  drive     'auto'   episode8's data autopilot: PI speed control towards
                     random speed targets (16-30 kn), random heading
                     targets, coloured dithers on thrust and nozzle, track 0
            'mpc'    the driving of the recorded gz missions
                     (studies/onestep_gz.record): the hand-tuned MPC's
                     thrust (weights theta(z_hand())) and the heading
                     autopilot's nozzle (Mission.mpc_command), the
                     measured-roll safe mode, along track leg LEGS[mi % 2]
                     (heading 0 or 45 deg to the waves from WAVES_FROM)
  sea_mode  'redraw' episode8's sea, drawn per mission (data2.sea_for)
            'task'   the task's sea in every mission (Hs 1.0 m, Tp 5 s,
                     waves from WAVES_FROM: the gz missions' spectrum);
                     only the wave phases differ (the Mission seed)

With drive 'auto' and sea_mode 'redraw' this is episode8.episode (the same
streams). The mission's random stream is consumed in the same order in
every condition up to the control loop. The boat's law, the actuators, the
error target and the operator streams are episode8's.
"""
import copy   # noqa: F401  (the episode body deep-copies snapshots)

import numpy as np

from learn.meta import data2
from learn.meta.data2 import (CK0, KN, T_EP, _stations, clip_push,  # noqa: F401
                              draw_actuators, low_dt, sea_for)
from learn.meta.episode7 import (FAMILIES7, CatSafetyHook, diverged,  # noqa: F401
                                 make_injector, make_op, obs_names,
                                 op_inputs6, style7, wrap_plant)
from learn.meta.episode8 import (HID0, MISSION_STRIDE, TAG_OP,  # noqa: F401
                                 TAG_SEA, episode_id, mission_env,
                                 mission_state)
from learn.meta.mpc_constrained import SafeMode
from learn.repro.task import LEGS, Mission, ctx
from studies.sim2real_jet import theta, z_hand


def episode(job):
    """episode8.episode with job['drive'] in ('auto', 'mpc') and
    job['sea_mode'] in ('redraw', 'task') (module docstring)."""
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
    drive = job.get("drive", "auto")
    sea_mode = job.get("sea_mode", "redraw")
    if drive not in ("auto", "mpc") or sea_mode not in ("redraw", "task"):
        raise ValueError(f"drive {drive!r} / sea_mode {sea_mode!r}")
    if sea_mode == "task":
        sea = None      # the task's sea; the wave phases from the Mission seed
    # the boat's actuators: the M10 draw comes from the stream [law, 11]
    # (draw_actuators reads job['seed']); the old draw still consumes the
    # mission's stream as in episode7
    if job.get("act_params") is None and \
            job.get("act_family") not in ("m10", "at"):
        raise ValueError("episode8: the boat's actuators need act_family "
                         "'m10' or 'at' (the old draw uses the mission "
                         "stream)")
    act, act_fam = draw_actuators(dict(job, seed=law), rng, world)
    if drive == "auto":
        m = Mission(world, law + 50000 + MISSION_STRIDE * mi, 0, t_end=T,
                    track=0.0, sea=sea, plant_params=act)
    else:               # the recorded gz missions' driving (onestep_gz.record)
        w_h, floor_h = theta(z_hand())
        leg = mi % 2
        m = Mission(world, law + 50000 + MISSION_STRIDE * mi, leg, t_end=T,
                    weights=w_h, floor=floor_h, track=LEGS[leg], sea=sea,
                    plant_params=act)
        safe = SafeMode(m.dt * m.sub)
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
        s0 = m.s.copy()
        if drive == "mpc":
            thrust, rudder = m.mpc_command()
            thrust, rudder, _ = safe(m.s[3], thrust, rudder)
            thrust = float(np.clip(thrust, 0.0, m.t_max))
            rudder = float(np.clip(rudder, -m.rud_max, m.rud_max))
        else:
            if informative and t >= t_u:
                u_tgt, t_u = rng.uniform(16, 30) * KN, t + rng.uniform(10, 40)
            if informative and t >= t_psi:
                psi_tgt = rng.uniform(-np.pi, np.pi)
                t_psi = t + rng.uniform(20, 60)
            m.ep.heading_ref = psi_tgt
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
    if sea is None:     # the task's sea, as relabel._sea rebuilds it (checked)
        sea = dict(hs=1.0, tp=5.0, theta0=float(np.pi), n_freq=24, n_dir=5)
    out.update(finite=bool(m.finite), world=world, seed=int(job["seed"]),
               law=law, mi=mi, hid_seed=hid_seed, env_redraw=env_redraw,
               act=act, act_family=act_fam, op_seed=op_seed,
               relay=bool(job.get("relay", False)),
               informative=bool(informative), sea=sea,
               ck=np.asarray(ck, np.int32), slow=slow,
               style=style7(op), family=fam, div=int(div),
               drive=drive, sea_mode=sea_mode,
               snaps=snaps)
    return out


_LIB = {}


def run_job(job):
    """A worker's episode (meta_step2.run_jobs), as episode8.run_job."""
    j = dict(job)
    path = j.pop("lib_path", None)
    if j.pop("use_lib", False):
        if _LIB.get("path") != path:
            d = np.load(path)
            _LIB.update(path=path, lib=dict(mu=d["mu"], sd=d["sd"],
                                            S=d["S"]))
        j["lib"] = dict(_LIB["lib"])
    try:
        return episode(j)
    except Exception as ex:                       # keep the chunk going
        import traceback
        return dict(error=repr(ex) + " | " + traceback.format_exc()[-600:],
                    seed=job["seed"])
