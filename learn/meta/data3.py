#!/usr/bin/env python3
"""
Data for the one-step error model (learn/meta/model3.py; DEFECTS M7).

The model learns p(e_t | everything recorded before step t, the state at t,
the command applied during step t). The MPC gets multi-step predictions by
rolling its own reduced model forward and sampling e_t step by step.

THE ERROR IS DEFINED AGAINST THE MODEL THE MPC ROLLS OUT: model0_step, the
reduced model with no waves at any step (the MPC does not know future
waves) and ideal actuators, 6 substeps of 0.04 s = one 0.24-s control step,
no thrust floor. (control/mpc.py's MPPI does NOT do exactly this: one
0.25-s step, the measured wave at h = 0, a thrust floor. When the predictor
enters the MPC, the MPC must roll out with model0_step; on smoke data one
0.25-s step differs from it by about one e0 sd in sway.)

Ten channels per control step:

    e_t[0:8] = (true next state - model0's next state) / step
               for surge, sway, yaw rate, heave rate, pitch rate, heave and
               pitch displacement, heading (so a rolled-out state drifts in
               neither position nor heading)
    e_t[8:10] = (actuator position after the step - the command) / step
               for thrust and nozzle (fractions of range)

The waves' effect, any unknown mechanism and (through the first eight) the
actuator lag's consequences are in e; the last two let a rollout carry the
actuator positions the network was trained on (they trail the command)
without any lag model.
The recorded history and the imagined rollout then use the SAME definition,
so imagined states look like real ones (with the old true-wave error the
imagined heave / pitch would have been missing the wave response).

Also here:
- branches(): multi-step truth for the source splits (A, B). From a moment k
  of an episode, the operator's FULL internal state is recovered by
  replaying it from step 0 with the episode's own random stream. P commands
  sequences are fixed in advance, and each is simulated in closed loop with
  the full operator (rule + noise) in the episode's sea.
- build_tbranches(): the same with wide plans (wide_plans: full-range
  command steps) for an explicit job list -- training / validation branches
  for model3.train_rollout (DEFECTS M9) and the in-family test A_wbranches.
- episode_blocks(): target-world (high-fidelity) episodes whose commands are
  fixed in advance for 24-step blocks (no feedback inside a block), so a
  block's recorded future is an unconfounded multi-step truth for split C.
"""
import copy
import os
import pickle

import numpy as np

from learn.meta.data2 import (KN, T_EP, _stations, op_inputs, plant_to_reduced,
                              sea_for)
from learn.meta.operators import Operator, clip_push
from learn.meta.residuals import VEL_IDX, inputs_of

# reduced-state indices of the eight state error channels
E7 = (2, 9, 8, 4, 6, 3, 5, 7)   # u, v, r, heave rate, pitch rate, z, th, psi
CH7 = ("surge", "sway", "yaw", "heave", "pitch", "heave_pos", "pitch_pos",
       "heading", "thrust_act", "nozzle_act")
N_E = len(CH7)                  # 10 channels in all
HB = 24                          # horizon / block length, control steps


def model0_step(env, sr, U):
    """The MPC's model over one control step: no waves, ideal actuators.
    sr (N, 10) reduced states, U (N, 2) command fractions -> (N, 10)."""
    red = env["red"]
    eta0 = np.zeros((len(sr), len(env["x_st"]), 3))
    thr, rud = U[:, 0] * env["t_max"], U[:, 1] * env["rud_max"]
    out = sr.copy()
    for _ in range(env["sub"]):
        out = red.step(out, thr, rud, eta0, env["x_st"], env["dt"])[0]
    return out


def e0_from(env, xs_now, xs_next, U):
    """e (N, 10) from plant states before and after a step and the command
    applied during it."""
    a = plant_to_reduced(xs_now)
    b = plant_to_reduced(xs_next)
    m = model0_step(env, a, U)
    dtc = env["dt"] * env["sub"]
    act = np.asarray(xs_next, float)[:, 12:14] / np.array(
        [env["t_max"], env["rud_max"]])
    return np.concatenate([(b[:, list(E7)] - m[:, list(E7)]) / dtc,
                           (act - U) / dtc], 1)


def e0_split(path, chunk=100000):
    """E0 (n, T, 10) for a packed split; the last step of an episode has no
    next state and stays 0 (valid mask: t < len - 1)."""
    from learn.meta import relabel
    env = relabel._env()
    d = np.load(path)
    XS, U, L = d["XS"], d["U"], d["len"]
    n, T, _ = XS.shape
    E0 = np.zeros((n, T, N_E), np.float32)
    idx = np.array([(i, t) for i in range(n) for t in range(int(L[i]) - 1)])
    for s in range(0, len(idx), chunk):
        ii, tt = idx[s:s + chunk, 0], idx[s:s + chunk, 1]
        E0[ii, tt] = e0_from(env, XS[ii, tt], XS[ii, tt + 1], U[ii, tt])
    out = path.replace(".npz", "_e0.npz")
    np.savez(out + ".tmp.npz", E0=E0)
    os.replace(out + ".tmp.npz", out)
    return out


# ----------------------------------------------------------- plans
def make_plans(rng, u_now, P, H=HB, dt=0.24):
    """P command sequences (P, H, 2) fixed in advance from the current
    command u_now: plan 0 holds it, the others add an OU perturbation
    (tau 0.5-3 s, size up to 0.2 of range) and, w.p. 0.5, one step change
    (thrust +-0.25, nozzle +-0.4) at a random time: the variety an MPC's
    sampled plans and a probing manoeuvre cover."""
    out = np.repeat(np.asarray(u_now, float)[None, None], P, 0).repeat(H, 1)
    for p in range(1, P):
        for c, (lo, hi, st) in enumerate(((0.0, 1.0, 0.25), (-1.0, 1.0, 0.4))):
            tau = np.exp(rng.uniform(np.log(0.5), np.log(3.0)))
            amp = rng.uniform(0.0, 0.2)
            a = np.exp(-dt / tau)
            x, dev = 0.0, np.zeros(H)
            for j in range(H):
                x = a * x + amp * np.sqrt(1 - a * a) * rng.normal()
                dev[j] = x
            if rng.random() < 0.5:
                j0 = int(rng.integers(0, H))
                dev[j0:] += rng.uniform(-st, st)
            out[p, :, c] = np.clip(out[p, :, c] + dev, lo, hi)
    return out


def wide_plans(prng, u_now, P, H=HB, rng2=None):
    """make_plans' P plans (drawn UNCHANGED from prng, so the A / B / Cb
    rebuilds stay identical) plus, for each plan p >= 1 and each channel,
    w.p. 0.3 a full-range step: at j0 ~ U{0..H-1} the channel jumps to a new
    level ~ U(its range) and keeps make_plans' perturbation around it.
    The draws come from the separate stream rng2. Full nozzle reversals are
    7.3% of the steps in the target split Cb and 0.9% in the training
    commands (DEFECTS M8); these plans put them into the branch data.

    Returns the plans (P, H, 2) and a flag per plan (P,): True when the
    plan received at least one such step."""
    out = make_plans(prng, u_now, P, H)
    u0 = np.asarray(u_now, float)
    wide = np.zeros(P, bool)
    for p in range(1, P):
        for c, (lo, hi) in enumerate(((0.0, 1.0), (-1.0, 1.0))):
            if rng2.random() < 0.3:
                j0 = int(rng2.integers(0, H))
                level = rng2.uniform(lo, hi)
                out[p, j0:, c] = np.clip(level + (out[p, j0:, c] - u0[c]),
                                         lo, hi)
                wide[p] = True
    return out, wide


# -------------------------------------------------------- branches
def _replay_operator(meta, lib, X, k, env):
    """The operator of an episode and its full internal state before step k,
    by replaying it from step 0 on the recorded inputs with the episode's
    own random stream (as data2.episode drew it)."""
    op = Operator(meta["op_seed"], lib, dt=env["dt"] * env["sub"], L=env["L"],
                  relay=meta["relay"])
    rng = np.random.default_rng([meta["op_seed"], 1])
    st = op.new_state(1, rng=rng, init_s=X[0][None])
    for t in range(k):
        op.step(st, X[t][None], noise=True, rng=rng)
    return op, st, rng


def _tile_state(st, P):
    """Copy a one-row operator state into P rows. The coloured-noise cascade
    states nx hold rows b * 5 + channel, so they are tiled branch by branch
    (np.repeat would hand channel c's state to other channels)."""
    out = {}
    for key, v in st.items():
        if key == "rng_slow":
            out[key] = copy.deepcopy(v)
        elif key == "nx":
            out[key] = [np.tile(a, (P, 1)) for a in v]
        elif isinstance(v, np.ndarray):
            out[key] = np.repeat(v, P, axis=0)
        elif isinstance(v, list):
            out[key] = [None if a is None else np.repeat(a, P, axis=0)
                        for a in v]
        else:
            out[key] = v
    return out


def _branch_sim(i, k, plans, pushes=None):
    """Closed-loop truths of episode i from moment k under plans (P, H, 2):
    the operator's full state recovered by replay, tiled over the plans,
    each branch simulated with the full operator in the episode's sea (the
    M10 actuators' delay line from the recorded commands before k,
    relabel.hist_for). pushes (P, H, 5) replays given pushes instead of
    running the operator (tests: the deterministic part).
    Returns XS (P, H + 1, 14) and E0 (P, H, 10)."""
    from learn.meta import relabel as R
    W, env = R._W, R._env()
    P = len(plans)
    meta = W["meta"][i]
    X = op_inputs(W["S"][i], W["U"][i])
    op, st1, rng = _replay_operator(meta, W["lib"], X, k, env)
    st = _tile_state(st1, P)
    seas = R.Seas([R._sea(meta)] * P, env["x_st"], env["y_off"])
    act = R.act_rows([R._act(meta)] * P)
    t0 = np.full(P, k * env["dt"] * env["sub"])
    xs = np.repeat(W["XS"][i, k][None], P, 0).astype(float)
    uh, av = R.hist_for(env, W, [(i, k)] * P)
    if pushes is not None:
        _, XS = R.simulate(env, xs, plans, t0, seas, act, pushes=pushes,
                           u_hist=uh, av0=av)
    else:
        _, XS = R.simulate(env, xs, plans, t0, seas, act, op=op, st=st,
                           noise_rng=rng, u_hist=uh, av0=av)
    E0 = np.stack([e0_from(env, XS[:, j], XS[:, j + 1], plans[:, j])
                   for j in range(plans.shape[1])], 1)
    return XS, E0


def branches_one(args):
    """P closed-loop branches of H steps from moment k of episode i."""
    from learn.meta import relabel as R
    i, k, P, seed = args
    meta = R._W["meta"][i]
    # the plans' stream is independent of the operator's replay stream, so
    # drawing them before the replay leaves both unchanged
    prng = np.random.default_rng([meta["seed"], k, seed])
    plans = make_plans(prng, R._W["U"][i, k], P)
    XS, E0 = _branch_sim(i, k, plans)
    return i, k, plans.astype(np.float32), XS.astype(np.float32), \
        E0.astype(np.float32)


def tbranches_one(args):
    """branches_one with wide plans (wide_plans; its extra stream is seeded
    from the same job) and the per-plan full-range flag. The plan stream is
    seeded as in branches_one, so with the same seed the plans are A / B's
    plans plus the full-range steps."""
    from learn.meta import relabel as R
    i, k, P, seed = args
    meta = R._W["meta"][i]
    prng = np.random.default_rng([meta["seed"], k, seed])
    rng2 = np.random.default_rng([meta["seed"], k, seed, 7])
    plans, wide = wide_plans(prng, R._W["U"][i, k], P, rng2=rng2)
    XS, E0 = _branch_sim(i, k, plans)
    return i, k, plans.astype(np.float32), XS.astype(np.float32), \
        E0.astype(np.float32), wide


def build_branches(path, lib, n_eps=60, moments=(120, 240), P=8, procs=5,
                   seed=0):
    """Multi-step truth for a source split -> <split>_branches.npz."""
    from multiprocessing import Pool

    from learn.meta import relabel as R
    d = np.load(path)
    L = d["len"]
    del d
    rng = np.random.default_rng(seed)
    meta = pickle.load(open(path.replace(".npz", "_meta.pkl"), "rb"))
    ok = [i for i in range(len(L)) if L[i] >= max(moments) + HB + 1
          and not meta[i]["style"]["null"]]
    eps = rng.choice(ok, size=min(n_eps, len(ok)), replace=False)
    jobs = [(int(i), int(k), P, seed) for i in eps for k in moments]
    res = []
    if procs <= 1:                  # in this process (one-process machines)
        R._init(path, lib)
        R._W.pop("seas", None)          # keyed by episode seed, per split
        res = [branches_one(j) for j in jobs]
    else:
        with Pool(procs, initializer=R._init, initargs=(path, lib)) as pool:
            for r in pool.imap_unordered(branches_one, jobs):
                res.append(r)
    res.sort(key=lambda r: (r[0], r[1]))
    out = path.replace(".npz", "_branches.npz")
    np.savez(out + ".tmp.npz", ep=np.array([r[0] for r in res]),
             k=np.array([r[1] for r in res]),
             U=np.stack([r[2] for r in res]), XS=np.stack([r[3] for r in res]),
             E0=np.stack([r[4] for r in res]))
    os.replace(out + ".tmp.npz", out)
    return out


def build_tbranches(path, lib, jobs, out, procs=4):
    """Wide-plan branches for an explicit job list [(i, k, P, seed)] of the
    split at `path` -> `out` with ep, k, U (N, P, H, 2), XS (N, P, H + 1,
    14), E0 (N, P, H, 10) like <split>_branches.npz, plus wide (N, P).
    relabel._init loads the whole split in every worker (~0.5 GB for the
    training split), hence at most 4 workers; procs <= 1 runs in this
    process (tests)."""
    from learn.meta import relabel as R
    res = []
    if procs <= 1:
        R._init(path, lib)
        R._W.pop("seas", None)          # keyed by episode seed, per split
        res = [tbranches_one(j) for j in jobs]
    else:
        from multiprocessing import Pool
        with Pool(min(procs, 4), initializer=R._init,
                  initargs=(path, lib)) as pool:
            for r in pool.imap_unordered(tbranches_one, jobs):
                res.append(r)
    res.sort(key=lambda r: (r[0], r[1]))
    # a non-finite truth anywhere (full-range steps are new inputs to the
    # simulator) would make the ES scale NaN and skip every update: drop the
    # whole moment, so the (N, P, H, ...) arrays stay rectangular
    n_all = len(res)
    res = [r for r in res
           if np.isfinite(r[3]).all() and np.isfinite(r[4]).all()]
    if len(res) < n_all:
        print(f"build_tbranches {os.path.basename(out)}: dropped "
              f"{n_all - len(res)} of {n_all} moments with non-finite "
              f"XS / E0", flush=True)
    if not res:
        raise RuntimeError(f"build_tbranches {out}: all {n_all} moments "
                           f"non-finite")
    np.savez(out + ".tmp.npz", ep=np.array([r[0] for r in res]),
             k=np.array([r[1] for r in res]),
             U=np.stack([r[2] for r in res]), XS=np.stack([r[3] for r in res]),
             E0=np.stack([r[4] for r in res]),
             wide=np.stack([r[5] for r in res]))
    os.replace(out + ".tmp.npz", out)
    return out


# ------------------------------------------------- block-planned episodes
def episode_blocks(job):
    """A target-world episode driven by commands fixed in advance for
    24-step blocks. At each block start the scripted controller's command
    (as data2.episode's) is the base of a plan from make_plans (a random
    one of 4), executed without feedback until the next block start.
    Records S, XS, U, E like data2.episode (E: the old true-wave error,
    kept for comparison).

    job['p_hold'] (split Cbh; default 0 = Cb unchanged): w.p. p_hold a
    block's NOZZLE column is held at the previous block's last nozzle
    command for all 24 steps (the first block: the controller's), thrust
    keeps its plan, so many blocks have the nozzle command constant and the
    actual nozzle at rest on it. The choice uses its own stream [seed, 9],
    so the main stream (sea, controller, plans) is Cb's."""
    from learn.repro.task import Mission, ctx
    from sim import lofi
    c = ctx()
    rng = np.random.default_rng(job["seed"])
    T = job.get("T", T_EP)
    sea = sea_for(rng, target=True)
    m = Mission("high", int(job["seed"]) + 50000, 0, t_end=T, track=0.0,
                sea=sea)
    twin = lofi.plant_for(c["db"], m.sea, c["h"], dt=2 * m.dt)
    twin_sub = int(round(m.sub * m.dt / twin.dt))
    p = m.red.p
    u_tgt = rng.uniform(16, 30) * KN
    psi_tgt = rng.uniform(-np.pi, np.pi)
    t_u = t_psi = 0.0
    integ = 0.0
    kp = p["m_surge"] / 3.0
    rec = {k: [] for k in ("S", "XS", "U", "E")}
    plan, j = None, HB
    p_hold = float(job.get("p_hold", 0.0))
    hrng = np.random.default_rng([int(job["seed"]), 9])
    while not m.done():
        t = m.t
        s0 = m.s.copy()
        if j >= HB:
            if t >= t_u:
                u_tgt, t_u = rng.uniform(16, 30) * KN, t + rng.uniform(10, 40)
            if t >= t_psi:
                psi_tgt = rng.uniform(-np.pi, np.pi)
                t_psi = t + rng.uniform(20, 60)
            m.ep.heading_ref = psi_tgt
            e_u = u_tgt - s0[6]
            integ = float(np.clip(integ + e_u * m.dt_ctrl * HB, -20, 20))
            thr0 = float(np.clip(p["k_drag"] * u_tgt ** 2 + kp * e_u
                                 + 0.1 * kp * integ, 0.0, m.t_max))
            rud0 = float(np.clip(m.ep._steer(s0, thr0), -m.rud_max,
                                 m.rud_max))
            base = (thr0 / m.t_max, rud0 / m.rud_max)
            prev = None if plan is None else float(plan[-1, 1])
            plan = make_plans(rng, base, 4)[int(rng.integers(4))]
            if p_hold > 0 and hrng.random() < p_hold:
                plan = plan.copy()
                plan[:, 1] = base[1] if prev is None else prev
            j = 0
        thrust, rudder = plan[j, 0] * m.t_max, plan[j, 1] * m.rud_max
        j += 1
        x9 = inputs_of(s0, m.u_ref, m.L, m.t_max, m.rud_max)
        zr = ((s0[2] - p.get("z0", 0.0)) / 0.2,
              (s0[4] - p.get("th0", 0.0)) / 0.05)
        s26 = np.concatenate([x9, zr, _stations(twin, s0, t)])
        stw = s0.copy()
        for kk in range(twin_sub):
            stw = twin.step(stw, t + kk * twin.dt, thrust, rudder, twin.dt)
        m.advance(thrust, rudder)
        if not m.finite:
            break
        dtc = m.sub * m.dt
        rec["S"].append(s26)
        rec["XS"].append(s0)
        rec["U"].append((thrust / m.t_max, rudder / m.rud_max))
        rec["E"].append([(m.s[i] - stw[i]) / dtc for i in VEL_IDX])
        last = m.s.copy()
    # the state after the last RECORDED step (not a non-finite one)
    rec["XS"].append(last if rec["S"] else m.s.copy())
    out = {k: np.asarray(v, np.float32) for k, v in rec.items()}
    out.update(finite=bool(m.finite), seed=int(job["seed"]), sea=sea)
    return out
