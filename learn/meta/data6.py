#!/usr/bin/env python3
"""
Branches of the rigid-body error families (meta6; learn/meta/episode6.py
for the episodes). Outside studies/meta_step6.code6() (the chunk stamp),
inside sim6(), so editing this file never makes the episodes stale.

Why not the meta5 path. data5.build_branches5 recovers the operator's state
at the branch moment by REPLAYING it on the episode's recorded inputs, and
data2.build_targets / relabel._warm evaluate an operator on recorded
(donor) inputs with a slow state written into it. Both need an operator
that is a function of the recorded inputs. The rigid-body families are
state feedback at every substep (OperatorGen.run / OperatorRB.run raise),
so here:
  - the state at the moment k is not replayed but taken from the
    snapshot the episode kept at k (episode6: operator state incl. filters,
    delay lines, hidden processes and the old noise part's slow state, its
    random stream, the plant's M10 actuator state, plant state, time);
  - every branch is a CLOSED-LOOP continuation of the episode in its own
    sea with the recorded actuators: the same substep loop as
    Mission.advance (the plant wrapper's step, then the family's injector),
    the operator rebuilt from its seed (episode6.make_op) and its
    control-step part computed from the simulated states' 28 M15 inputs
    (W_MID of the simulated pose);
  - every plan continues the episode's own noise stream (a copy of the
    snapshot's stream per plan: common random numbers across plans), so
    the plan equal to the recorded commands reproduces the recorded
    continuation (studies/test_m16.py);
  - no slow state is written anywhere (the snapshot carries it); donor
    targets and relabel's warm-up are not part of the meta5 / meta6 phases.

build_branches6 picks episodes and moments and draws plans exactly as
data5.build_branches5 (same rng, same make_plans stream), and writes the
same file format (ep, k, U, XS, E0) plus div (P,) per row. A branch that
leaves the envelope (continue_world: div = the step, its state then held,
as simulate_gen freezes a row) has no meaningful E0 after that step, and
model_preview / eval_preview_m15 read only U, XS, E0; so every MOMENT with
any diverged plan is dropped whole (the P plans stay rectangular) and
counted in n_dropped (with the dropped (ep, k) in dropped_ep / dropped_k).
No torch.
"""
import copy
import os
import pickle

import numpy as np

from learn.meta import episode6 as E6
from learn.meta.operators import clip_push


def rebuild_op(meta, lib, env):
    fam = meta["prior_family"]
    return E6.make_op(fam, meta[f"op_seed_{fam}"], lib,
                      env["dt"] * env["sub"], env["L"],
                      relay=meta.get("relay", False))


def base_plant(meta, env):
    """The episode's low-fidelity plant (its rebuilt sea, its actuators),
    as Mission builds it for world 'low'."""
    from learn.meta import relabel as R
    from learn.repro.task import ctx
    from sim import lofi
    c = ctx()
    return lofi.plant_for(c["db"], R._sea(meta), c["h"], dt=env["dt"],
                          params=meta.get("act"))


def continue_world(op, meta, snap, plans, env, base=None):
    """Closed-loop continuations of an episode from its snapshot under
    plans (P, H, 2) of command fractions: XS (P, H + 1, 14), E0 (P, H, 10),
    div (P,)."""
    from learn.meta import data3
    fam = meta["prior_family"]
    base = base_plant(meta, env) if base is None else base
    p = env["red"].p
    dt, sub = env["dt"], env["sub"]
    h_mid = 0.5 * dt * sub
    cst = dict(u_ref=env["u_ref"], L=env["L"], t_max=env["t_max"],
               rud_max=env["rud_max"], z0=p.get("z0", 0.0),
               th0=p.get("th0", 0.0))
    u_max = float(p["u_max"])
    plans = np.asarray(plans, float)
    P, H = plans.shape[:2]
    XS = np.zeros((P, H + 1, 14))
    div = np.full(P, -1)
    for b in range(P):
        plant = E6.wrap_plant(base)
        plant.set_act_state(snap["act"])
        st = op.snapshot(snap["st"])
        rng = copy.deepcopy(snap["rng"])
        hook = E6.SafetyHook(p, plant.x_st, E6.make_injector(fam, op, st))
        s, t = np.array(snap["xs"], float), float(snap["t"])
        XS[b, 0] = s
        for j in range(H):
            if div[b] >= 0:
                XS[b, j + 1] = s
                continue
            cmd = plans[b, j]
            thrust, rudder = cmd[0] * env["t_max"], cmd[1] * env["rud_max"]
            _, _, s28 = E6.op_inputs6(s, t, plant, cst, h_mid, cmd)
            rr, rn = op.step(st, s28[None], noise=True, rng=rng)
            hook.set_held(*clip_push(rr[0], rn[0]))
            s_k = s.copy()
            for _ in range(sub):
                s = plant.step(s, t, thrust, rudder, dt)
                s = hook(s, dt, plant)
                t += dt
                if not np.all(np.isfinite(s)):
                    break
            if E6.diverged(s, u_max):
                div[b] = j
                s = s_k
            XS[b, j + 1] = s
    E0 = np.stack([data3.e0_from(env, XS[:, j], XS[:, j + 1], plans[:, j])
                   for j in range(H)], 1)
    return XS, E0, div


# ------------------------------------------------------------ builder
_B = {}


def _binit(path, lib):
    d = np.load(path)
    _B.update(U=d["U"], lib=lib,
              meta=pickle.load(open(path.replace(".npz", "_meta.pkl"),
                                    "rb")),
              snaps=pickle.load(open(path.replace(".npz", "_snaps.pkl"),
                                     "rb")))


def _episode_branches(args):
    """All moments of one episode (the operator is rebuilt once)."""
    from learn.meta import data3
    from learn.meta import relabel as R
    i, moments, P, seed = args
    env = R._env()
    meta = _B["meta"][i]
    op = rebuild_op(meta, _B["lib"], env)
    base = base_plant(meta, env)
    out = []
    for k in moments:
        prng = np.random.default_rng([meta["seed"], k, seed])
        plans = data3.make_plans(prng, _B["U"][i, k], P)
        XS, E0, div = continue_world(op, meta, _B["snaps"][i][k], plans,
                                     env, base)
        out.append((i, k, plans.astype(np.float32), XS.astype(np.float32),
                    E0.astype(np.float32), div))
    R._W.pop("seas", None)
    return out


def build_branches6(path, lib, n_eps=60, moments=(120, 240), P=8, procs=1,
                    seed=0):
    """data5.build_branches5's episodes, moments and plans, continued from
    the episodes' snapshots -> <split>_branches.npz (+ div)."""
    from learn.meta import data3
    d = np.load(path)
    L = d["len"]
    del d
    rng = np.random.default_rng(seed)
    meta = pickle.load(open(path.replace(".npz", "_meta.pkl"), "rb"))
    snaps = pickle.load(open(path.replace(".npz", "_snaps.pkl"), "rb"))
    ok = [i for i in range(len(L)) if L[i] >= max(moments) + data3.HB + 1
          and not meta[i]["style"]["null"]
          and all(k in snaps.get(i, {}) for k in moments)]
    del snaps
    eps = rng.choice(ok, size=min(n_eps, len(ok)), replace=False)
    jobs = [(int(i), tuple(int(k) for k in moments), P, seed) for i in eps]
    res = []
    if procs <= 1:
        _binit(path, lib)
        for j in jobs:
            res += _episode_branches(j)
    else:
        from multiprocessing import Pool
        with Pool(min(procs, 2), initializer=_binit,
                  initargs=(path, lib)) as pool:
            for r in pool.imap_unordered(_episode_branches, jobs):
                res += r
    res.sort(key=lambda r: (r[0], r[1]))
    if not res:
        raise RuntimeError(f"build_branches6: no branches for {path}")
    bad = [r for r in res if (np.asarray(r[5]) >= 0).any()]
    keep = [r for r in res if not (np.asarray(r[5]) >= 0).any()]
    tpl = res[0]
    out = path.replace(".npz", "_branches.npz")

    def stack(j, dt):
        if keep:
            return np.stack([r[j] for r in keep])
        return np.zeros((0,) + np.shape(tpl[j]), dt)

    np.savez(out + ".tmp.npz", ep=np.array([r[0] for r in keep], int),
             k=np.array([r[1] for r in keep], int),
             U=stack(2, np.float32), XS=stack(3, np.float32),
             E0=stack(4, np.float32), div=stack(5, int),
             n_dropped=np.array(len(bad)),
             dropped_ep=np.array([r[0] for r in bad], int),
             dropped_k=np.array([r[1] for r in bad], int))
    os.replace(out + ".tmp.npz", out)
    return out
