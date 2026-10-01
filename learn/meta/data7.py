#!/usr/bin/env python3
"""
Branches of the D10a catalogue family (meta7; learn/meta/episode7.py for
the episodes). Outside studies/meta_step7.code7() (the chunk stamp),
inside sim7(), so editing this file never makes the episodes stale.

The data6 procedure for the catalogue family, unchanged in substance: the
state at the branch moment is the snapshot the episode kept (every item's
hidden state and generator, layer 1 incl. the old noise part's slow state,
layer 3's filters, the slam detector, the true-actuator copy, the noise
stream, the plant's M10 actuator state, plant state, time); every branch is
a CLOSED-LOOP continuation in the episode's own sea with the recorded
actuators through the same substep loop as Mission.advance (CatPlant, then
CatSafetyHook(CatInjector)), the operator rebuilt from its seed and the
episode's actuator dict (episode7.make_op); every plan continues a copy of
the snapshot's noise stream (common random numbers), so the plan equal to
the recorded commands reproduces the recorded continuation
(studies/test_m17.py). Moments, plans and the file format (ep, k, U, XS,
E0, div, n_dropped, dropped_ep / dropped_k) are data6's (data5's
episodes, moments and make_plans stream); a moment with any diverged plan
is dropped whole. No torch.
"""
import copy
import os
import pickle

import numpy as np

from learn.meta import data6 as D6
from learn.meta import episode7 as E7
from learn.meta.operators import clip_push


def rebuild_op(meta, lib, env):
    act = meta.get("act")
    new_act = act is not None and act.get("act_family", 0.0) > 0.5
    return E7.make_op(meta["prior_family"], meta["op_seed_cat"], lib,
                      env["dt"] * env["sub"], env["L"],
                      relay=meta.get("relay", False),
                      act=act if new_act else None)


base_plant = D6.base_plant


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
        plant = E7.wrap_plant(base)
        plant.set_act_state(snap["act"])
        st = op.snapshot(snap["st"])
        rng = copy.deepcopy(snap["rng"])
        hook = E7.CatSafetyHook(p, plant.x_st,
                                E7.make_injector(fam, op, st))
        s, t = np.array(snap["xs"], float), float(snap["t"])
        XS[b, 0] = s
        for j in range(H):
            if div[b] >= 0:
                XS[b, j + 1] = s
                continue
            cmd = plans[b, j]
            thrust, rudder = cmd[0] * env["t_max"], cmd[1] * env["rud_max"]
            _, _, s28 = E7.op_inputs6(s, t, plant, cst, h_mid, cmd)
            rr, rn = op.step(st, s28[None], noise=True, rng=rng)
            hook.set_held(*clip_push(rr[0], rn[0]))
            s_k = s.copy()
            for _ in range(sub):
                s = plant.step(s, t, thrust, rudder, dt)
                s = hook(s, dt, plant)
                t += dt
                if not np.all(np.isfinite(s)):
                    break
            if E7.diverged(s, u_max):
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


def build_branches7(path, lib, n_eps=60, moments=(120, 240), P=8, procs=1,
                    seed=0):
    """data6.build_branches6 for the catalogue family ->
    <split>_branches.npz (+ div)."""
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
        raise RuntimeError(f"build_branches7: no branches for {path}")
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
