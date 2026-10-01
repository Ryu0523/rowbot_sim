#!/usr/bin/env python3
"""
Data of the M15 operator family (DEFECTS M16, whose code, tag and cache
keep the name m15 / meta5; brief BRIEF_PREVIEW.md): in the training world,
the waves the hull meets during a step carry information about that step's
error.

The episodes themselves (op_family 'v0' = data2.episode exactly, 'm15' =
the operator reads W_MID through pattern readings; mid_pose, station_xy,
mid_stations_twin, episode, run_job, pack5) live in learn/meta/episode5.py,
re-exported here: meta_step5.code5(), the stamp of every data chunk, hashes
that module and ops_m15.py only, so edits to the code below (rebuilds,
branches, target W_MID; covered by meta_step5.sim5()) do not make the
episodes stale.

All code of the new family lives here, in episode5.py and in ops_m15.py,
outside studies/meta_step2.SOURCES, so the running v0 jobs' code hashes and
files are untouched. Every old path that rebuilds an operator from a meta
(data3._replay_operator, relabel._one, data2._targets_one) reads
meta['op_seed']; an M15 meta names it 'op_seed_m15' instead, so those paths
fail with a KeyError on M15 data instead of silently rebuilding a v0
operator on step-start elevations. The M15 versions are here: op_inputs5,
make_operator, replay_operator, simulate_mid, branch_sim5 /
build_branches5.

The mid-step pose (mid_pose) is ONE function for all four users: the
episode (plant state, 14), the target splits' rebuilt W_MID (plant state),
simulate_mid (reduced state, 10) and the network's rollout hook (reduced
state, torch): from the step-start state, x and y advance with the
earth-frame velocity of (u, v) at the step-start heading, psi with r, over
h = dtc / 2, dtc = sub x dt = 0.24 s (the plant time a control step
covers; Mission.dt_ctrl = 0.25 s is only the dither's step). It uses
nothing from the step's outcome (diag_wave_preview_m11's 'mid' used the
realised path and is not this).
"""
import os
import pickle

import numpy as np

from learn.meta.data2 import op_inputs
from learn.meta.episode5 import (LAYOUTS, OP_FAMILIES, episode,  # noqa: F401
                                 mid_pose, mid_stations_twin, pack5, run_job,
                                 station_xy)
from learn.meta.operators import Operator
from learn.meta.ops_m15 import OperatorM15


def sea_eval(sea, X, Y, t):
    """Elevation of a SeaState at points X, Y (...) and times t
    (broadcastable), the same sum as twin._surface / relabel.Seas."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    ph = sea.k * (X[..., None] * np.cos(sea.th) + Y[..., None]
                  * np.sin(sea.th)) - sea.w * np.asarray(t, float)[
        ..., None] + sea.phi
    return (sea.a * np.cos(ph)).sum(-1)


# ---------------------------------------------------- operator rebuild
def op_family_of(meta):
    return meta.get("op_family", "v0")


def op_inputs5(S, U, W_MID=None, family="v0"):
    """The operator's 28 inputs of a family: v0 = data2.op_inputs(S, U);
    m15 = S's first 11, W_MID in columns 11:26, the commands."""
    if family == "v0":
        return op_inputs(S, U)
    if family != "m15":
        raise ValueError(f"op family {family!r}")
    if W_MID is None:
        raise ValueError("op_inputs5: the m15 family needs W_MID")
    S = np.asarray(S)
    return np.concatenate([S[..., :11], W_MID, U], -1)


def make_operator(meta, lib, env):
    """The operator of an episode's meta, of its family."""
    dt, L = env["dt"] * env["sub"], env["L"]
    if op_family_of(meta) == "m15":
        return OperatorM15(meta["op_seed_m15"], lib, dt=dt, L=L,
                           relay=meta["relay"])
    return Operator(meta["op_seed"], lib, dt=dt, L=L, relay=meta["relay"])


def op_seed_of(meta):
    return meta["op_seed_m15"] if op_family_of(meta) == "m15" \
        else meta["op_seed"]


def replay_operator(meta, lib, X, k, env):
    """data3._replay_operator for either family: the operator and its full
    state before step k, replayed from step 0 on its recorded inputs X
    (op_inputs5) with the episode's own random stream."""
    op = make_operator(meta, lib, env)
    rng = np.random.default_rng([op_seed_of(meta), 1])
    st = op.new_state(1, rng=rng, init_s=X[0][None])
    for t in range(k):
        op.step(st, X[t][None], noise=True, rng=rng)
    return op, st, rng


class _MidSeas:
    """relabel.Seas that remembers the pose and time of its last
    stations() call (the step-start call relabel.simulate makes right
    before it runs the operator)."""

    def __init__(self, seas):
        self.base = seas
        self.last = None

    def stations(self, x, y, psi, t):
        self.last = (x, y, psi, t)
        return self.base.stations(x, y, psi, t)


class _OpMid:
    """An operator whose wave inputs are replaced, at each step of
    relabel.simulate, by W_MID of the SIMULATED step-start state: position
    and heading from the step-start stations() call, u, v, r from the
    operator's own inputs (columns 0-2: u / u_ref - 1, v, r L / u_ref), at
    t + dtc / 2 (mid_pose, reduced layout)."""

    def __init__(self, op, mseas, env):
        self.op, self.mseas, self.env = op, mseas, env

    def step(self, st, s28, **kw):
        env = self.env
        x, y, psi, t = self.mseas.last
        s28 = np.array(s28, float)
        u = (s28[:, 0] + 1.0) * env["u_ref"]
        v = s28[:, 1]
        r = s28[:, 2] * env["u_ref"] / env["L"]
        sr = np.zeros((len(x), 10))
        sr[:, 0], sr[:, 1], sr[:, 7] = x, y, psi
        sr[:, 2], sr[:, 9], sr[:, 8] = u, v, r
        h = 0.5 * env["dt"] * env["sub"]
        xm, ym, pm = mid_pose(sr, "red10", h)
        s28[:, 11:26] = self.mseas.base.stations(xm, ym, pm, t + h).reshape(
            len(x), 15)
        return self.op.step(st, s28, **kw)


def simulate_mid(env, xs14, U, t0, seas, act, op=None, st=None,
                 noise_rng=None, u_hist=None, av0=None, family="m15"):
    """relabel.simulate with the operator of `family`: m15 runs it on the
    W_MID of the simulated states (_OpMid); v0 is relabel.simulate."""
    from learn.meta import relabel as R
    if family == "v0" or op is None:
        return R.simulate(env, xs14, U, t0, seas, act, op=op, st=st,
                          noise_rng=noise_rng, u_hist=u_hist, av0=av0)
    ms = _MidSeas(seas)
    return R.simulate(env, xs14, U, t0, ms, act, op=_OpMid(op, ms, env),
                      st=st, noise_rng=noise_rng, u_hist=u_hist, av0=av0)


# ------------------------------------------------------------ branches
def init_w(path, lib):
    """relabel._init plus the split's W_MID (None when it has none)."""
    from learn.meta import relabel as R
    R._init(path, lib)
    d = np.load(path)
    R._W["W_MID"] = d["W_MID"] if "W_MID" in d.files else None
    R._W.pop("seas", None)


def branch_sim5(i, k, plans, pushes=None):
    """data3._branch_sim for either family (relabel._W from init_w)."""
    from learn.meta import data3
    from learn.meta import relabel as R
    W, env = R._W, R._env()
    P = len(plans)
    meta = W["meta"][i]
    fam = op_family_of(meta)
    X = op_inputs5(W["S"][i], W["U"][i],
                   None if W.get("W_MID") is None else W["W_MID"][i], fam)
    op, st1, rng = replay_operator(meta, W["lib"], X, k, env)
    st = data3._tile_state(st1, P)
    seas = R.Seas([R._sea(meta)] * P, env["x_st"], env["y_off"])
    act = R.act_rows([R._act(meta)] * P)
    t0 = np.full(P, k * env["dt"] * env["sub"])
    xs = np.repeat(W["XS"][i, k][None], P, 0).astype(float)
    uh, av = R.hist_for(env, W, [(i, k)] * P)
    if pushes is not None:
        _, XS = R.simulate(env, xs, plans, t0, seas, act, pushes=pushes,
                           u_hist=uh, av0=av)
    else:
        _, XS = simulate_mid(env, xs, plans, t0, seas, act, op=op, st=st,
                             noise_rng=rng, u_hist=uh, av0=av, family=fam)
    E0 = np.stack([data3.e0_from(env, XS[:, j], XS[:, j + 1], plans[:, j])
                   for j in range(plans.shape[1])], 1)
    return XS, E0


def branches_one5(args):
    """data3.branches_one with branch_sim5 (same plan stream)."""
    from learn.meta import data3
    from learn.meta import relabel as R
    i, k, P, seed = args
    meta = R._W["meta"][i]
    prng = np.random.default_rng([meta["seed"], k, seed])
    plans = data3.make_plans(prng, R._W["U"][i, k], P)
    XS, E0 = branch_sim5(i, k, plans)
    return i, k, plans.astype(np.float32), XS.astype(np.float32), \
        E0.astype(np.float32)


def build_branches5(path, lib, n_eps=60, moments=(120, 240), P=8, procs=4,
                    seed=0):
    """data3.build_branches (same episodes, moments, plans) with the
    operator of each episode's family -> <split>_branches.npz."""
    from learn.meta import data3
    d = np.load(path)
    L = d["len"]
    del d
    rng = np.random.default_rng(seed)
    meta = pickle.load(open(path.replace(".npz", "_meta.pkl"), "rb"))
    ok = [i for i in range(len(L)) if L[i] >= max(moments) + data3.HB + 1
          and not meta[i]["style"]["null"]]
    eps = rng.choice(ok, size=min(n_eps, len(ok)), replace=False)
    jobs = [(int(i), int(k), P, seed) for i in eps for k in moments]
    if procs <= 1:
        init_w(path, lib)
        res = [branches_one5(j) for j in jobs]
    else:
        from multiprocessing import Pool
        res = []
        with Pool(min(procs, 4), initializer=init_w,
                  initargs=(path, lib)) as pool:
            for r in pool.imap_unordered(branches_one5, jobs):
                res.append(r)
    res.sort(key=lambda r: (r[0], r[1]))
    out = path.replace(".npz", "_branches.npz")
    np.savez(out + ".tmp.npz", ep=np.array([r[0] for r in res]),
             k=np.array([r[1] for r in res]),
             U=np.stack([r[2] for r in res]), XS=np.stack([r[3] for r in res]),
             E0=np.stack([r[4] for r in res]))
    os.replace(out + ".tmp.npz", out)
    return out


# ------------------------------------------------- target splits' W_MID
def target_wmid(path, env):
    """W_MID (n, T, 15) raw of a (copied) target split from its REBUILT sea
    (relabel._sea of the meta) at the dead-reckoned mid-step poses of its
    recorded states, at t = k dtc + dtc / 2 (the same mid_pose as the
    source episodes); rows t >= len stay 0."""
    from learn.meta import relabel as R
    d = np.load(path)
    meta = pickle.load(open(path.replace(".npz", "_meta.pkl"), "rb"))
    XS, L = d["XS"].astype(float), d["len"]
    n, T = d["U"].shape[:2]
    dtc = env["dt"] * env["sub"]
    h = 0.5 * dtc
    out = np.zeros((n, T, 15), np.float32)
    for i in range(n):
        Li = int(L[i])
        sea = R._sea(meta[i])
        xm, ym, pm = mid_pose(XS[i, :Li], "plant14", h)
        X, Y = station_xy(xm, ym, pm, env["x_st"], env["y_off"])
        tt = (np.arange(Li) * dtc + h)[:, None, None]
        out[i, :Li] = sea_eval(sea, X, Y, tt).reshape(Li, 15)
    R._W.pop("seas", None)
    return out
