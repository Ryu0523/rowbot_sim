#!/usr/bin/env python3
"""
The episodes of the M15 operator family (DEFECTS M16, whose code, tag and
cache keep the name m15 / meta5; brief BRIEF_PREVIEW.md): everything the
data chunks are made from, and nothing else, so studies/meta_step5.code5()
(the stamp every chunk and pack carries) hashes exactly this module,
ops_m15.py and meta_step2.SOURCES. The operator rebuilds, branches and the
target splits' W_MID stay in data5.py (covered by meta_step5.sim5() only),
so editing them does not make the ~5 h of episodes stale. data5 re-exports
every name here; import from data5.

op_family (a flag of its own, separate from act_family):
- 'v0'  data2.episode exactly (test_m15 test 1: every record and key bit
        for bit, old / m10 with and without relays / at);
- 'm15' the operator's 15 wave inputs are W_MID, the elevations at the
        MPC's 5 x 3 stations at the step's midpoint t + dtc / 2 at the pose
        dead-reckoned from the step-start state (mid_pose), from the
        episode's own sea; the operator is ops_m15.OperatorM15 (pattern
        readings of the waves). S still records the elevations at t (what
        the boat measures); W_MID (15) is recorded next to it.

No torch here: the episode and branch workers stay numpy-only (mid_pose
uses torch only for tensors of a caller that loaded it; test_m15 test 1
checks that no episode imports it).
"""
import sys

import numpy as np

from learn.meta import data2
from learn.meta.data2 import (CK0, KN, T_EP, ZOHInjector, _stations,
                              clip_push, draw_actuators, inputs_of, low_dt,
                              sea_for)
from learn.meta.operators import Operator
from learn.meta.ops_m15 import OperatorM15
from learn.repro.task import Mission, ctx

OP_FAMILIES = ("v0", "m15")
# x, y, psi, u, v, r columns of the two state layouts
LAYOUTS = dict(plant14=(0, 1, 5, 6, 7, 11), red10=(0, 1, 7, 2, 9, 8))


# ------------------------------------------------------ mid-step pose
def mid_pose(s, layout, h):
    """The hull pose h seconds after the step start, dead-reckoned from the
    step-start state s (..., 14 or 10; numpy or torch): (x, y, psi)."""
    ix, iy, ip, iu, iv, ir = LAYOUTS[layout]
    # torch only when the caller already loaded it and passes a tensor:
    # the numpy callers (episodes, target_wmid, the branch workers) must
    # never load torch's ~3 GB of DLLs into every worker (Windows commit
    # charge, WinError 1455)
    torch = sys.modules.get("torch")
    xp = torch if torch is not None and torch.is_tensor(s) else np
    x, y, psi = s[..., ix], s[..., iy], s[..., ip]
    u, v, r = s[..., iu], s[..., iv], s[..., ir]
    c, sn = xp.cos(psi), xp.sin(psi)
    return x + (u * c - v * sn) * h, y + (u * sn + v * c) * h, psi + r * h


def station_xy(x, y, psi, x_st, y_off):
    """Station positions (..., 5, 3) of hulls at poses (...), as
    data2._stations / relabel.Seas.stations place them."""
    c, sn = np.cos(psi)[..., None, None], np.sin(psi)[..., None, None]
    x_st, y_off = np.asarray(x_st), np.asarray(y_off)
    X = x[..., None, None] + x_st[:, None] * c - y_off[None, :] * sn
    Y = y[..., None, None] + x_st[:, None] * sn + y_off[None, :] * c
    return X, Y


def mid_stations_twin(twin, s14, t, h):
    """W_MID (15,) of one step: the twin's surface at the stations of the
    pose mid_pose(s14) at time t + h."""
    xm, ym, pm = mid_pose(np.asarray(s14, float), "plant14", h)
    X, Y = station_xy(np.asarray(xm), np.asarray(ym), np.asarray(pm),
                      twin.x_st, twin.y_off)
    eta, _ = twin._surface(X.ravel(), Y.ravel(), t + h)
    return eta


# ------------------------------------------------------------ episode
def episode(job):
    """data2.episode with the operator family job['op_family'] ('v0'
    default = data2.episode exactly, 'm15'); the other job keys as
    data2.episode."""
    from sim import lofi
    fam_op = job.get("op_family", "v0")
    if fam_op not in OP_FAMILIES:
        raise ValueError(f"op_family {fam_op!r}: one of {OP_FAMILIES}")
    m15 = fam_op == "m15"
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
        cls = OperatorM15 if m15 else Operator
        op = cls(op_seed, job["lib"], dt=m.sub * m.dt, L=m.L,
                 relay=job.get("relay", False))
        inj = ZOHInjector()
        m.residual = inj
    rng_op = np.random.default_rng([op_seed, 1])
    st = None                  # built at the first step, from its inputs
    twin = lofi.plant_for(c["db"], m.sea, c["h"],
                          dt=m.dt if world == "low" else 2 * m.dt,
                          params=act)
    twin_sub = int(round(m.sub * m.dt / twin.dt))
    h_mid = 0.5 * m.sub * m.dt
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
    rec = {k: [] for k in keys + (("R", "CMD") if raw64 else ())
           + (("W_MID",) if m15 else ())}
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
        cmd = [thrust / m.t_max, rudder / m.rud_max]
        if m15:
            w_mid = mid_stations_twin(twin, s0, t, h_mid)
            s28 = np.concatenate([x9, zr, w_mid, cmd])
        else:
            s28 = np.concatenate([s26, cmd])
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
        y = np.array([(m.s[i] - stw[i]) / dtc for i in data2.VEL_IDX])
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
        if m15:
            rec["W_MID"].append(w_mid)
    out = {k: np.asarray(v, np.float64 if raw64 else np.float32)
           for k, v in rec.items()}
    if m15 and len(rec["W_MID"]) == 0:
        out["W_MID"] = np.zeros((0, 15), out["S"].dtype)
    out.update(finite=bool(m.finite), world=world, seed=int(job["seed"]),
               act=act, act_family=fam,
               op_seed=op_seed, relay=bool(job.get("relay", False)),
               informative=bool(informative), sea=sea,
               ck=np.asarray(ck, np.int32), slow=slow,
               style=None if op is None else op.style)
    if m15:
        out["op_family"] = "m15"
    return out


_LIB5 = {}


def run_job(job):
    """A worker's episode (studies/meta_step5.py hands this to
    meta_step2.run_jobs): job['lib_path'] is loaded once per process;
    an exception comes back as dict(error, seed) so the chunk goes on."""
    j = dict(job)
    path = j.pop("lib_path", None)
    if j.pop("use_lib", False):
        if _LIB5.get("path") != path:
            d = np.load(path)
            _LIB5.update(path=path, lib=dict(mu=d["mu"], sd=d["sd"],
                                             S=d["S"]))
        j["lib"] = dict(_LIB5["lib"])
    try:
        return episode(j)
    except Exception as ex:                       # keep the chunk going
        import traceback
        return dict(error=repr(ex) + " | " + traceback.format_exc()[-600:],
                    seed=job["seed"])


# ------------------------------------------------------------ packing
def pack5(eps, T_max=None):
    """data2.pack, plus W_MID (n, T, 15) when every episode has it (raises
    when only some do: a pack never mixes families), and in the meta
    op_family and, for M15 episodes, op_seed renamed op_seed_m15 (module
    docstring)."""
    kept = [e for e in eps if len(e["E"]) > data2.H + 10]
    has = ["W_MID" in e for e in kept]
    if any(has) and not all(has):
        raise ValueError(f"pack5: {sum(has)} of {len(kept)} episodes have "
                         "W_MID; one family per pack")
    fams = {e.get("op_family", "v0") for e in kept}
    if len(fams) > 1:
        raise ValueError(f"pack5: op families {sorted(fams)} in one pack")
    out, meta = data2.pack(eps, T_max)
    if all(has) and kept:
        n, T = out["E"].shape[:2]
        a = np.zeros((n, T, 15), np.float32)
        for i, e in enumerate(kept):
            L = min(len(e["W_MID"]), T)
            a[i, :L] = e["W_MID"][:L]
        out["W_MID"] = a
    for m_, e in zip(meta, kept):
        fam = e.get("op_family", "v0")
        if fam == "m15":
            m_["op_seed_m15"] = m_.pop("op_seed")
            m_["op_family"] = "m15"
    return out, meta
