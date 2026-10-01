#!/usr/bin/env python3
"""
Episodes for learning to infer residuals.

Each episode: the Scarab task's sea (sea state 3, waves from a fixed
direction), the source world with its nominal coefficients, and -- in the
source -- one random residual function (residuals.py). The boat is driven
to be INFORMATIVE: a speed loop chases piecewise-constant speed targets
(16-30 kn, held 10-40 s) with a slow random thrust dither, and the
autopilot chases random headings (held 20-60 s). A third of the episodes
run one speed and one heading with no dither -- the uninformative case, in
which the posterior must stay broad about everything the boat did not do.

Recorded per control step, only what a boat can measure (plus diagnostics):

  X   the residual's inputs (residuals.inputs_of)
  O   encoder tokens: X, heave and pitch about the running attitude, CG
      acceleration, the commands, and the observed error Y
  Y   (velocity change of the boat - velocity change of a nominal twin
      stepped from the same state with the same commands) / control step,
      in the five channels: the error of the reduced model's EQUATIONS,
      integrated finely with the true wave, over one control step. (Not the
      MPC's whole one-step error: that also contains the MPC's coarse
      0.25 s zero-order-hold step and its ignorance of the coming waves --
      measured separately in split C, see studies/meta_step1.py.) In the
      source this is the injected residual after the boat's own fast
      response within the step (heave and pitch are stiff: the observed
      error there is a filtered version of the push)
  F   the injected residual itself, step mean (diagnostic only)

and, in the source, the TRUTH over the whole input box, in Y's units:

  P   128 probe states per episode (the 9 residual inputs + heave and pitch
      about the running attitude): half near visited states, half spread
      over the input box
  YP  at checkpoints every 120 steps: the Y each probe state would produce --
      two copies of the plant stepped one control step from that state,
      with and without the residual (its memory frozen at that moment)

In the target world (world='high') there is no injected residual: Y is the
full plant's own difference from the model's equations -- the gap step 1
is tested on, never trained on.
"""
import numpy as np

from learn.meta.residuals import (Injector, ResidualFunction, VEL_IDX,
                                  inputs_of)
from learn.repro.task import LEGS, Mission, ctx

KN = 0.514444
T_EP = 90.0
CK_STEPS = 120                  # checkpoints every 120 control steps (28.8 s)
N_PROBE = 128


def _probes(rng, X, Z, n=N_PROBE):
    """Probe states (n, 11): half near visited ones, half over the box."""
    k = n // 2
    i = rng.integers(len(X), size=k)
    near = np.hstack([X[i], Z[i]]) + rng.normal(0, 0.05, (k, 11))
    a = rng.uniform(-np.pi, np.pi, n - k)
    box = np.column_stack([
        rng.uniform(-0.45, 0.25, n - k), rng.normal(0, 0.5, n - k),
        rng.normal(0, 0.5, n - k), np.cos(a), np.sin(a),
        rng.uniform(0.1, 1.0, n - k), rng.uniform(-1, 1, n - k),
        rng.normal(0, 1.0, n - k), rng.normal(0, 0.3, n - k),
        rng.normal(0, 1.0, n - k), rng.normal(0, 1.0, n - k)])
    return np.vstack([near, box])


def _probe_truth(m, fn, twin, P, x0, y0, t):
    """Y at each probe state: the model's equations stepped one control
    step from that state with and without the residual (its memory frozen),
    all probes at once. The same arithmetic as the twin's ReducedPlant.step
    (ideal actuators in the nominal model), vectorised over the probes."""
    p = m.red.p
    red = twin.model
    N = len(P)
    u = (P[:, 0] + 1.0) * m.u_ref
    psi = np.arctan2(P[:, 4], P[:, 3])
    sr = np.zeros((N, 10))
    sr[:, 0], sr[:, 1] = x0, y0
    sr[:, 2] = u
    sr[:, 3] = p.get("z0", 0.0) + 0.2 * P[:, 9]
    sr[:, 4] = P[:, 7]
    sr[:, 5] = p.get("th0", 0.0) + 0.05 * P[:, 10]
    sr[:, 6] = P[:, 8]
    sr[:, 7] = psi
    sr[:, 8] = P[:, 2] * m.u_ref / m.L
    sr[:, 9] = P[:, 1]
    thr = P[:, 5] * m.t_max
    rud = P[:, 6] * m.rud_max
    xs, yo = twin.x_st, twin.y_off

    def step(a, tk):
        c, s_ = np.cos(a[:, 7]), np.sin(a[:, 7])
        X = a[:, 0, None, None] + xs[None, :, None] * c[:, None, None] \
            - yo[None, None, :] * s_[:, None, None]
        Y = a[:, 1, None, None] + xs[None, :, None] * s_[:, None, None] \
            + yo[None, None, :] * c[:, None, None]
        eta, _ = twin._surface(X.ravel(), Y.ravel(), tk)
        return red.step(a, thr, rud, eta.reshape(N, len(xs), 3), xs,
                        m.dt)[0]

    a, b = sr.copy(), sr.copy()
    for k in range(m.sub):
        tk = t + k * m.dt
        a, b = step(a, tk), step(b, tk)
        x9 = np.column_stack([a[:, 2] / m.u_ref - 1.0, a[:, 9],
                              a[:, 8] * m.L / m.u_ref, np.cos(a[:, 7]),
                              np.sin(a[:, 7]), thr / m.t_max,
                              rud / m.rud_max, a[:, 4], a[:, 6]])
        r = fn.value(x9)
        for ch, col in enumerate((2, 9, 8, 4, 6)):
            a[:, col] += r[:, ch] * m.dt
    dtc = m.sub * m.dt
    return np.column_stack([(a[:, col] - b[:, col]) / dtc
                            for col in (2, 9, 8, 4, 6)])


def episode(job):
    """job: seed, world ('low'|'high'), exclude / force (residual styles),
    informative (bool or None = random), T."""
    from sim import lofi
    c = ctx()
    rng = np.random.default_rng(job["seed"])
    world = job.get("world", "low")
    T = job.get("T", T_EP)
    # the wave field's seed: unique per episode across all splits (train
    # 1.., A 100000.., B 200000.., C 300000..; a modulo here once made the
    # test splits reuse the training seas)
    m = Mission(world, int(job["seed"]) + 50000, 0, t_end=T, track=LEGS[0])
    fn = None
    if world == "low":
        fn = ResidualFunction(rng, exclude=job.get("exclude", ()),
                              force=job.get("force", ()))
        m.residual = Injector(fn, m.u_ref, m.L, m.t_max, m.rud_max)
    # the nominal twin: the model's equations, same sea, the source step
    twin = lofi.plant_for(c["db"], m.sea, c["h"],
                          dt=m.dt if world == "low" else 2 * m.dt)
    twin_sub = int(round(m.sub * m.dt / twin.dt))
    p = m.red.p
    informative = job.get("informative")
    if informative is None:
        informative = bool(rng.random() > 1 / 3)
    u_tgt = rng.uniform(16, 30) * KN
    psi_tgt = rng.uniform(-np.pi, np.pi) if informative else 0.0
    t_u = t_psi = 0.0
    dither, integ = 0.0, 0.0
    kp = p["m_surge"] / 3.0
    X, O, A, Y, F, Zs = [], [], [], [], [], []
    ck, YP = [], []
    P = None
    while not m.done():
        t = m.t
        if informative and t >= t_u:
            u_tgt, t_u = rng.uniform(16, 30) * KN, t + rng.uniform(10, 40)
        if informative and t >= t_psi:
            psi_tgt = rng.uniform(-np.pi, np.pi)
            t_psi = t + rng.uniform(20, 60)
        m.ep.heading_ref = psi_tgt
        s0 = m.s.copy()
        e = u_tgt - s0[6]
        integ = float(np.clip(integ + e * m.dt_ctrl, -20, 20))
        if informative:
            dither += (-dither / 3.0) * m.dt_ctrl + 0.08 * m.t_max \
                * np.sqrt(2 * m.dt_ctrl / 3.0) * rng.normal()
        thrust = float(np.clip(p["k_drag"] * u_tgt ** 2 + kp * e
                               + 0.1 * kp * integ + dither, 0.0, m.t_max))
        rudder = m.ep._steer(s0, thrust)
        x = inputs_of(s0, m.u_ref, m.L, m.t_max, m.rud_max)
        zr = ((s0[2] - p.get("z0", 0.0)) / 0.2,
              (s0[4] - p.get("th0", 0.0)) / 0.05)
        st = s0.copy()
        for k in range(twin_sub):
            st = twin.step(st, t + k * twin.dt, thrust, rudder, twin.dt)
        m.advance(thrust, rudder)
        if not m.finite:
            break
        fa = m.residual.take_mean() if fn is not None else np.zeros(5)
        dtc = m.sub * m.dt
        y = np.array([(m.s[i] - st[i]) / dtc for i in VEL_IDX])
        X.append(x)
        Zs.append(zr)
        O.append(np.concatenate([x, zr, [m.last_acg, thrust / m.t_max,
                                         rudder / m.rud_max], y]))
        A.append((thrust / m.t_max, rudder / m.rud_max))
        Y.append(y)
        F.append(fa)
        if fn is not None and len(X) % CK_STEPS == 0:
            if P is None:
                P = _probes(rng, np.asarray(X), np.asarray(Zs))
            ck.append(len(X))
            YP.append(_probe_truth(m, fn, twin, P, m.s[0], m.s[1], m.t))
    style = None
    if fn is not None:
        style = dict(fn.style)
        if fn.hyst is not None:
            n = max(fn.n_steps, 1)
            style.update(toggles=int(fn.toggles),
                         on_frac=float(fn.on_steps / n),
                         dwell_s=float(m.t / max(fn.toggles, 1)))
    return dict(X=np.asarray(X, np.float32), O=np.asarray(O, np.float32),
                A=np.asarray(A, np.float32), Y=np.asarray(Y, np.float32),
                F=np.asarray(F, np.float32),
                P=None if P is None else P.astype(np.float32),
                YP=np.asarray(YP, np.float32), ck=np.asarray(ck, np.int32),
                finite=bool(m.finite), world=world,
                informative=bool(informative), style=style)
