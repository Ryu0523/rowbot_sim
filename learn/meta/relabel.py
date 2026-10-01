#!/usr/bin/env python3
"""
CLOSED-LOOP relabelled targets (replaces data2.build_targets' open-loop
ones; DEFECTS M3).

The MPC asks: from this state, with this plan, what errors will the boat
meet, given that the errors themselves push it off the model's path? The
first targets evaluated operator i's rule on another episode's RECORDED
states. Those states had been pushed by that episode's own operator and
noise, and within one or two steps (heave and pitch rates react at once)
they no longer had anything to do with operator i. A measured check on
split A (100 checkpoints x 3 donors) found that the open- and closed-loop
labels differ by 66-98% of the target's power, with 0.000 at step 0, 40% at
step 1 and 91% at step 2.

Here every segment is re-simulated in the source world, batched over the
segments of a checkpoint:
- the start state is the segment's recorded state;
- the commands are its recorded commands (the plan);
- its own episode's sea and actuator lag are used;
- operator i's RULE runs in the loop: noise off, slow state frozen at the
  checkpoint, fading state warmed on the recorded inputs before the start,
  relays held through the warm-up.

At every step the target is e = the push response of the rule's push from
the simulated state (exact: the model stepped with and without it). The
simulator replicates sim/lofi.ReducedPlant.step (the old actuator lag and
rate limit, or the M10 actuator chain through the plant's own
lofi.act_step, its delay line rebuilt from the recorded commands before the
start (hist_for); the surface at the MPC's stations at each substep, the
reduced model, velocity kicks after each substep as data2.ZOHInjector);
replaying an episode's recorded push reproduces its recorded states
(check_replay).
"""
import os
import pickle

import numpy as np

from learn.meta.data2 import (CK0, H, M_DON, WARM, RED_VEL, _reduced_rollout,
                              op_inputs, plant_to_reduced)
from learn.meta.operators import A_REF, CLIP, Operator, clip_push


class Seas:
    """The seas of B rows, padded to a common component count, evaluated
    at the MPC's 5 x 3 stations of each row's hull."""

    def __init__(self, seas, x_st, y_off):
        n = max(s.a.size for s in seas)
        B = len(seas)
        self.a, self.k = np.zeros((B, n)), np.zeros((B, n))
        self.c, self.s = np.zeros((B, n)), np.zeros((B, n))
        self.w, self.phi = np.zeros((B, n)), np.zeros((B, n))
        for b, sea in enumerate(seas):
            m = sea.a.size
            self.a[b, :m], self.k[b, :m] = sea.a, sea.k
            self.c[b, :m], self.s[b, :m] = np.cos(sea.th), np.sin(sea.th)
            self.w[b, :m], self.phi[b, :m] = sea.w, sea.phi
        self.x_st, self.y_off = np.asarray(x_st), np.asarray(y_off)

    def stations(self, x, y, psi, t):
        """(B,) position, heading, time -> elevations (B, 5, 3)."""
        cp, sp = np.cos(psi)[:, None, None], np.sin(psi)[:, None, None]
        X = x[:, None, None] + self.x_st[None, :, None] * cp \
            - self.y_off[None, None, :] * sp
        Y = y[:, None, None] + self.x_st[None, :, None] * sp \
            + self.y_off[None, None, :] * cp
        B = len(x)
        X, Y = X.reshape(B, -1, 1), Y.reshape(B, -1, 1)
        ph = self.k[:, None] * (X * self.c[:, None] + Y * self.s[:, None]) \
            - self.w[:, None] * t[:, None, None] + self.phi[:, None]
        return (self.a[:, None] * np.cos(ph)).sum(-1).reshape(B, 5, 3)


def push_response_red(red, sr, U, r, t_max, rud_max, dt, sub, x_st):
    """data2.push_response from reduced states (B, 10)."""
    a, b = sr.copy(), sr.copy()
    eta0 = np.zeros((len(sr), len(x_st), 3))
    thr, rud = U[:, 0] * t_max, U[:, 1] * rud_max
    for _ in range(sub):
        a = red.step(a, thr, rud, eta0, x_st, dt)[0]
        b = red.step(b, thr, rud, eta0, x_st, dt)[0]
        a[:, list(RED_VEL)] += r * dt
    return (a[:, list(RED_VEL)] - b[:, list(RED_VEL)]) / (sub * dt)


N_HIST = 2      # control steps of command history hist_for hands simulate


def act_rows(acts):
    """Per-row actuator parameter dicts (_act) -> one dict of (B,) arrays,
    as simulate takes them (np.array of the dicts would be an object
    array)."""
    return {k: np.array([a[k] for a in acts], float) for k in acts[0]}


def _act_arrays(act, B):
    """simulate's act argument -> dict of (B,) arrays over every actuator
    parameter. A (B, 2) array is the pre-M10 form (tau_thrust, rud_rate)."""
    from sim import lofi
    keys = ("tau_thrust", "rud_rate") + lofi.ACT_KEYS
    if not isinstance(act, dict):
        act = np.asarray(act, float)
        act = dict(tau_thrust=act[:, 0], rud_rate=act[:, 1])
    return {k: np.broadcast_to(np.asarray(act.get(k, lofi.EXTRA[k]), float),
                               (B,)).copy() for k in keys}


def simulate(env, xs14, U, t0, seas, act, pushes=None, op=None, st=None,
             noise_rng=None, u_hist=None, av0=None):
    """Closed-loop source-world rollout of B rows over U.shape[1] steps.

    xs14 (B, 14) start plant states; U (B, n, 2) command fractions; t0 (B,)
    start times, s; act: dict of (B,) arrays (act_rows of _act), or the
    pre-M10 (B, 2) = (tau_thrust, rud_rate), 0 = ideal. Per row, the old
    actuators run for act_family 0 (exactly as before, bit for bit), the
    M10 chain (lofi.act_step, the plant's own function) for act_family 1.
    M10 rows with a delay need u_hist (B, n_hist, 2): the command fractions
    of the n_hist control steps before the start, left-padded with the
    episode's INITIAL actuator positions / scale (what the plant filled its
    delay line with; hist_for builds it). Second-order rows need av0 (B, 2),
    the servo velocities at the start (hist_for). Missing ones raise.
    The push each step is pushes[:, s] (a replay) or op's clipped rule on
    the simulated inputs; with noise_rng, the FULL operator (rule + noise,
    slow processes drifting) drawing on that stream. Returns e (B, n, 5)
    and the plant states (B, n + 1, 14). e is push_response_red: the push's
    effect on the reduced model with no waves and the COMMANDED actuators
    held, so it is exact only for ideal actuators and small pushes; the
    recorded E (data2) is plant minus twin with waves and the actual
    actuators. Nothing downstream of step 3 uses this e (it takes XS)."""
    from sim import lofi
    red, p = env["red"], env["red"].p
    dt, sub = env["dt"], env["sub"]
    dtc = dt * sub
    tmx, rmx = env["t_max"], env["rud_max"]
    B, n, _ = U.shape
    sr = plant_to_reduced(xs14)
    thr = xs14[:, 12].astype(float).copy()
    noz = xs14[:, 13].astype(float).copy()
    A = _act_arrays(act, B)
    tau, rate = A["tau_thrust"], A["rud_rate"]
    new = A["act_family"] > 0.5
    if new.any():
        q = lofi.act_arrays(A, tmx, rmx)
        dly = np.where(new[:, None], q["delay"], 0)
        uh = np.zeros((B, 0, 2)) if u_hist is None else np.asarray(u_hist,
                                                                   float)
        n_hist = uh.shape[1]
        if (dly > n_hist * sub).any():
            raise ValueError(f"simulate: M10 delays up to {dly.max()} steps "
                             f"need u_hist covering them (got {n_hist} "
                             f"control steps of {sub})")
        Ucat = np.concatenate([uh, np.asarray(U, float)], 1) \
            * np.array([tmx, rmx])
        so = new[:, None] & (q["zeta"] > 0.0)
        if so.any() and av0 is None:
            raise ValueError("simulate: second-order M10 rows need av0")
        vel = np.zeros((B, 2)) if av0 is None else \
            np.asarray(av0, float).copy()
        rows = np.arange(B)[:, None]
        cols = np.arange(2)[None, :]
    lim = CLIP * A_REF
    E = np.zeros((B, n, 5))
    XS = np.zeros((B, n + 1, 14))

    def to14(sr, thr, noz):
        o = np.zeros((len(sr), 14))
        o[:, [0, 1, 2, 4, 5]] = sr[:, [0, 1, 3, 5, 7]]
        o[:, [6, 7, 8, 10, 11]] = sr[:, [2, 9, 4, 6, 8]]
        o[:, 12], o[:, 13] = thr, noz
        return o

    XS[:, 0] = to14(sr, thr, noz)
    for s in range(n):
        t = t0 + s * dtc
        if pushes is not None:
            r = pushes[:, s]
        else:
            w15 = seas.stations(sr[:, 0], sr[:, 1], sr[:, 7], t)
            x, y, u, z, zd, th, thd, psi, rr_, v = sr.T
            s28 = np.column_stack([
                u / env["u_ref"] - 1.0, v, rr_ * env["L"] / env["u_ref"],
                np.cos(psi), np.sin(psi), thr / tmx, noz / rmx, zd, thd,
                (z - p.get("z0", 0.0)) / 0.2, (th - p.get("th0", 0.0)) / 0.05,
                w15.reshape(B, 15), U[:, s]])
            if noise_rng is None:
                r, _ = op.step(st, s28, noise=False, freeze=True)
                r = np.clip(r, -lim, lim)
            else:
                rr, nz = op.step(st, s28, noise=True, rng=noise_rng)
                rr, nz = clip_push(rr, nz)
                r = rr + nz
        E[:, s] = push_response_red(red, sr, U[:, s], r, tmx, rmx, dt, sub,
                                    env["x_st"])
        cmd_t, cmd_r = U[:, s, 0] * tmx, U[:, s, 1] * rmx
        for kk in range(sub):
            # actuators exactly as ReducedPlant.step
            lag = tau > 0
            e_ = np.exp(-dt / np.where(lag, tau, 1.0))
            thr_new = np.where(lag, cmd_t + (thr - cmd_t) * e_, cmd_t)
            thr_app = np.where(lag, cmd_t + (thr - cmd_t) * (
                np.where(lag, tau, 1.0) / dt) * (1 - e_), cmd_t)
            lim_r = rate > 0
            noz_new = np.where(lim_r, noz + np.clip(cmd_r - noz, -rate * dt,
                                                    rate * dt), cmd_r)
            noz_app = np.where(lim_r, 0.5 * (noz + noz_new), cmd_r)
            if new.any():
                # the M10 chain: the command d steps before global substep
                # g = s sub + kk is control step floor((g - d) / sub)
                g = s * sub + kk
                c_d = Ucat[rows, n_hist + (g - dly) // sub, cols]
                pos, vel_n, app = lofi.act_step(
                    q, np.stack([thr, noz], 1), vel, c_d, dt)
                vel = np.where(new[:, None], vel_n, vel)
                thr_new = np.where(new, pos[:, 0], thr_new)
                noz_new = np.where(new, pos[:, 1], noz_new)
                thr_app = np.where(new, app[:, 0], thr_app)
                noz_app = np.where(new, app[:, 1], noz_app)
            eta = seas.stations(sr[:, 0], sr[:, 1], sr[:, 7], t + kk * dt)
            sr = red.step(sr, thr_app, noz_app, eta, env["x_st"], dt)[0]
            sr[:, list(RED_VEL)] += r * dt
            thr, noz = thr_new, noz_new
        XS[:, s + 1] = to14(sr, thr, noz)
    return E, XS


# ----------------------------------------------------------- workers
_W = {}


def _env():
    if "env" not in _W:
        from learn.repro.task import Mission, ctx
        from sim import lofi
        c = ctx()
        m = Mission("low", 1, 0, t_end=5.0, track=0.0)
        twin = lofi.plant_for(c["db"], m.sea, c["h"], dt=m.dt)
        _W["env"] = dict(red=m.red, dt=m.dt, sub=m.sub, t_max=m.t_max,
                         rud_max=m.rud_max, u_ref=m.u_ref, L=m.L,
                         x_st=twin.x_st, y_off=twin.y_off, mission=m)
    return _W["env"]


def _sea(meta_j):
    from sim.wavefield import SeaState
    key = meta_j["seed"]
    cache = _W.setdefault("seas", {})
    if key not in cache:
        s = dict(meta_j["sea"])
        cache[key] = SeaState(s.pop("hs"), s.pop("tp"),
                              theta0=s.pop("theta0"), n_freq=s.pop("n_freq"),
                              n_dir=s.pop("n_dir"), seed=key + 50000, **s)
    return cache[key]


def _act(meta_j):
    """An episode's full actuator parameter set (dict of floats: the old
    tau_thrust / rud_rate and every lofi.ACT_KEYS entry, neutral where the
    meta does not set them; meta without act_family is old-style). Stack
    rows with act_rows."""
    from sim import lofi
    a = meta_j.get("act") or {}
    return {k: float(a.get(k, lofi.EXTRA[k]))
            for k in ("tau_thrust", "rud_rate") + lofi.ACT_KEYS}


def hist_for(env, W, rows, n_hist=N_HIST):
    """simulate's u_hist (B, n_hist, 2) and av0 (B, 2) for rows [(episode,
    start step)] of the packed split in W: the recorded commands of the
    n_hist steps before the start, left-padded with the episode's initial
    actuator positions as fractions (XS[j, 0, 12:14] / (t_max, rud_max): the
    plant filled its delay line with them), and the recorded servo
    velocities at the start (zero for files without AV, made before M10,
    which hold no M10 rows)."""
    sc = np.array([env["t_max"], env["rud_max"]])
    B = len(rows)
    uh = np.zeros((B, n_hist, 2))
    av = np.zeros((B, 2))
    for b, (j, t0) in enumerate(rows):
        pad = np.asarray(W["XS"][j, 0, 12:14], float) / sc
        for q in range(n_hist):
            k = t0 - n_hist + q
            uh[b, q] = W["U"][j, k] if k >= 0 else pad
        if W.get("AV") is not None:
            av[b] = W["AV"][j, t0]
    return uh, av


def _init(path, lib):
    d = np.load(path, allow_pickle=True)
    _W.update(S=d["S"], XS=d["XS"], U=d["U"], len=d["len"], CK=d["CK"],
              AV=d["AV"] if "AV" in d.files else None,
              lib=lib, meta=pickle.load(open(path.replace(".npz",
                                                          "_meta.pkl"), "rb")))


def _warm(op, slow, X_rows, starts, ends):
    """Operator state for rows warmed on recorded inputs X_rows[r][starts:
    ends] (equal lengths), slow state frozen, relays held."""
    B = len(X_rows)
    st = op.new_state(B, init_s=np.stack([X_rows[r][starts[r]]
                                          for r in range(B)]))
    for key in ("ga", "bb", "regime"):
        st[key][:] = slow[key][0]
    for q, rl in enumerate(slow["rl"]):
        st["rl"][q][:] = rl[0]
        st["rl_age"][q][:] = slow["rl_age"][q][0]
    st["k"] = slow["k"]
    n = ends[0] - starts[0]
    for k in range(n):
        op.step(st, np.stack([X_rows[r][starts[r] + k] for r in range(B)]),
                noise=False, freeze=True, hold_relays=True)
    return st


def _one(i):
    W, env = _W, _env()
    meta = W["meta"][i]
    CK = W["CK"][i]
    n_ck = len(CK)
    n = len(W["len"])
    Y = np.zeros((n_ck, M_DON + 1, H, 5), np.float32)
    XH = np.zeros((n_ck, M_DON + 1, H, 11), np.float32)
    REF = np.full((n_ck, M_DON + 1, 2), -1, np.int32)
    src = meta["world"] == "low" and meta["style"] is not None
    op = (Operator(meta["op_seed"], W["lib"], dt=env["dt"] * env["sub"],
                   L=env["L"], relay=meta["relay"]) if src else None)
    rng = np.random.default_rng([meta["seed"], 99])      # data2's donors
    dtc = env["dt"] * env["sub"]
    for c_, k in enumerate(CK):
        if k < 0:
            continue
        refs = [(i, int(k))]
        if src:
            for _ in range(M_DON):
                while True:
                    j = int(rng.integers(n))
                    if j != i and W["len"][j] >= WARM + H + 1:
                        break
                refs.append((j, int(rng.integers(WARM, W["len"][j] - H + 1))))
        REF[c_, :len(refs)] = refs
        xs = np.stack([W["XS"][j, t0] for j, t0 in refs])
        Us = np.stack([W["U"][j, t0:t0 + H] for j, t0 in refs])
        XH[c_, :len(refs)] = _reduced_rollout(
            env["red"], xs, Us, env["t_max"], env["rud_max"], env["dt"],
            env["sub"], env["x_st"], env["u_ref"], env["L"])
        if not src:
            continue
        slow = meta["slow"][int(k)]
        X = {j: op_inputs(W["S"][j], W["U"][j]) for j, _ in refs}
        # own future: warm from max(0, k - WARM); donors: WARM steps
        groups = [[0], list(range(1, len(refs)))]
        for g in groups:
            rows = [refs[q] for q in g]
            if rows[0][1] - WARM >= 0:
                starts = [t0 - WARM for _, t0 in rows]
            else:
                starts = [0 for _ in rows]
            ends = [t0 for _, t0 in rows]
            st = _warm(op, slow, [X[j] for j, _ in rows], starts, ends)
            seas = Seas([_sea(W["meta"][j]) for j, _ in rows], env["x_st"],
                        env["y_off"])
            act = act_rows([_act(W["meta"][j]) for j, _ in rows])
            t0s = np.array([t0 * dtc for _, t0 in rows])
            uh, av = hist_for(env, W, rows)
            E, _ = simulate(env, xs[g], Us[g], t0s, seas, act, op=op, st=st,
                            u_hist=uh, av0=av)
            Y[c_, g] = E
    return i, Y, XH, REF


def build_targets(path, lib, procs=5, code=""):
    """Closed-loop SEG_Y, SEG_XH, SEG_REF for a packed split, written to
    <split>_targets.npz (label='closed')."""
    from multiprocessing import Pool
    d = np.load(path, allow_pickle=True)
    n, n_ck = d["CK"].shape
    del d
    Y = np.zeros((n, n_ck, M_DON + 1, H, 5), np.float32)
    XH = np.zeros((n, n_ck, M_DON + 1, H, 11), np.float32)
    REF = np.full((n, n_ck, M_DON + 1, 2), -1, np.int32)
    with Pool(procs, initializer=_init, initargs=(path, lib)) as pool:
        for i, y, xh, ref in pool.imap_unordered(_one, range(n), chunksize=4):
            Y[i], XH[i], REF[i] = y, xh, ref
    seeds = np.array([m["seed"] for m in pickle.load(open(
        path.replace(".npz", "_meta.pkl"), "rb"))])
    out = path.replace(".npz", "_targets.npz")
    tmp = out + ".tmp.npz"
    np.savez(tmp, SEG_Y=Y, SEG_XH=XH, SEG_REF=REF, seeds=seeds,
             code=np.array(code), label=np.array("closed"))
    os.replace(tmp, out)
    return out


REPLAY_TOL = 1e-5     # relative, on float32 records (rounding ~6e-8)


def check_replay(path, lib, n_eps=4, k=120, tol=REPLAY_TOL):
    """Replay the recorded push of a few episodes from a recorded state:
    the simulated states must match the recorded ones. The packed records
    are float32 (states, commands and the push RR + RN, which the plant got
    in float64), so the pass criterion is a float32-level tolerance: max
    over rows and columns of |XS_sim - XS_rec| / max|XS_rec| <= tol. e is
    reported only (simulate's e is not the recorded E by construction; see
    simulate). Returns dict(xs = per-episode (14,) relative errors, e =
    per-episode relative e errors, max_xs, ok)."""
    _init(path, lib)
    env = _env()
    xs_err, e_err = [], []
    d = np.load(path)
    for i in range(n_eps):
        m = _W["meta"][i]
        if m["world"] != "low":
            continue
        push = (d["RR"][i, k:k + H] + d["RN"][i, k:k + H])[None]
        seas = Seas([_sea(m)], env["x_st"], env["y_off"])
        uh, av = hist_for(env, _W, [(i, k)])
        E, XS = simulate(env, _W["XS"][i, k][None], _W["U"][i, k:k + H][None],
                         np.array([k * env["dt"] * env["sub"]]), seas,
                         act_rows([_act(m)]), pushes=push, u_hist=uh,
                         av0=av)
        rec = _W["XS"][i, k:k + H + 1]
        sc = np.abs(rec).max(0) + 1e-9
        xs_err.append(np.abs(XS[0] - rec).max(0) / sc)
        e_rec = d["E"][i, k:k + H]
        e_err.append(float(np.abs(E[0] - e_rec).max()
                           / (np.abs(e_rec).max() + 1e-9)))
    mx = float(max(e.max() for e in xs_err)) if xs_err else np.nan
    return dict(xs=xs_err, e=e_err, max_xs=mx, ok=bool(mx <= tol))
