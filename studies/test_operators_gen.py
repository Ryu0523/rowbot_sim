#!/usr/bin/env python3
"""
Tests of the general operator family (learn/meta/operators_gen.py;
PRIOR_DERIVATION.md D9.8). One process, a few minutes.

  1  reproducible from the seed (draw, acceptance, outputs); another seed
     differs
  2  branch replay (float64): continuing from a snapshot at step k equals
     the run's own continuation (no noise; with noise) to <= 1e-9
  3  the Mission path (RBPlant + GenInjector, one row) equals simulate_gen
     to <= 1e-9, and so do its safety quantities
  4  forces enter only at the acceleration level: after one substep the
     positions equal the low-fidelity boat's, the velocities differ by
     exactly push x dt
  5  a point force gives the lever-arm ratios (sway : yaw = 1/m_v : x/I_r,
     heave : pitch = 1/m_w : -x/I_q) with the repo's signs (a port force
     ahead of the CG turns the bow to port, r > 0; an upward force at the
     bow lifts it, pitch rate < 0); with the CG offset, M_t a = Q exactly
  6  magnitude bounds: every force, every channel |a_c| <= 3 kappa_i
     a_ref,c on random and extreme inputs (bound by construction; the
     worst case is exercised)
  7  acceptance rule (divergence only): no forces -> accepted; a strong
     pitch anti-damping force -> 'diverged'; a constant surge push beyond
     the speed limit -> 'diverged'; a large constant heave push that
     stays bounded -> accepted (no shape rule), its clip share recorded
 10  input library: the standardised library columns have mean 0 / std 1;
     a draw's filter outputs on the library have unit std; relays switch
     on the library for a sizeable share of relay filters
  8  clip rule: a huge rule is clipped at CLIP x A_REF per substep, the
     total push is clip(rule + noise), the rule-clip counter counts it
  9  safety quantities (APK, AMIN, HMIN) equal an independent computation
     from the substep trajectory and the sea's own surface function

    python studies/test_operators_gen.py
"""
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from learn.meta import operators_gen as G
from learn.meta import operators_rb as R
from learn.meta.operators import A_REF, CLIP, clip_push

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L_ = np.load(os.path.join(ROOT, "studies", "_cache", "meta3", "lib.npz"))
LIB = {k: L_[k] for k in L_.files}
ENV = R.light_env()
LIM = CLIP * A_REF
FAIL = []


def check(name, ok, msg=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {msg}")
    if not ok:
        FAIL.append(name)


def seas_for(B, seed0=0):
    return R.RowSeas([R.sea_state(R.sea_dict(np.random.default_rng(
        [seed0, i])), 50000 + i) for i in range(B)], ENV["x_st"],
        ENV["y_off"])


def rich_op(seed, lib=None):
    """A draw with several forces, a filter with memory and a hidden
    process (to exercise the state), not acceptance-tested."""
    o = G.OperatorGen(seed, lib, accept=False)
    while True:
        kinds = [g["kind"] for f in o.forces for g in f["filt"]["list"]]
        if len(o.forces) >= 3 and o.hidden and ("res" in kinds or
                                                "low" in kinds):
            return o
        o = G.OperatorGen(o.seed + 1, lib, accept=False)


def manual_gen(op, col, a5, gain=1.0, bias=None):
    """A generic force whose single readout is SAT tanh(gain X[col] / SAT)
    (bias=None) or the constant SAT tanh(bias / SAT), times the
    acceleration vector a5: a direct filter (not standardised: y_mu 0,
    y_sd 1) and a purely linear readout. X is the library-standardised
    input."""
    rng = np.random.default_rng(0)
    Wc = np.zeros((op.n_in, 1))
    Wc[col, 0] = gain
    filt = dict(K=1, list=[dict(kind="direct", src=0, delay=0.0)],
                lin=np.array([0]), rel=np.array([], int), src=np.array([0]),
                delay=np.zeros(1), A=np.zeros((0, 0)), Bm=np.zeros((0, 1)),
                C=np.zeros((1, 0)), D=np.ones(1), Sx=np.zeros((0, 1)), n=0,
                disc={}, rel_tau=np.zeros(0), rel_hi=np.zeros(0),
                rel_lo=np.zeros(0), rel_hi_u=np.zeros(0),
                rel_lo_u=np.zeros(0), y_mu=np.zeros(1), y_sd=np.ones(1))
    ro = op._draw_readout(rng, 1, 1)
    ro.update(beta=1.0, Wl=np.ones((1, 1)), gamma=0.0, sgn=np.zeros(1))
    if bias is not None:
        Wc[:] = 0.0
        ro.update(gamma=1.0, sgn=np.array([float(bias)]))
    return dict(kind="gen", kappa=1.0, groups=["manual"], Wc=Wc, filt=filt,
                Ga=np.asarray(a5, float)[:, None], A=1.0, n_out=1, ro=ro)


def one_row_groups(ops, noise, lib_noise=True):
    return [dict(op=o, st=o.new_state(1, rng=np.random.default_rng(
        [o.seed, 1])), rows=np.array([i]),
        rng=np.random.default_rng([o.seed, 2]) if noise else None)
        for i, o in enumerate(ops)]


# ------------------------------------------------------------------ 1
def test_reproducible():
    print("1. reproducible from the seed")
    outs = []
    for seed in (11, 11, 12):
        o = G.OperatorGen(seed, LIB)            # with the acceptance test
        st = o.new_state(3, rng=np.random.default_rng([seed, 1]))
        nrng = np.random.default_rng([seed, 1])
        r2 = np.random.default_rng(5)
        acc = []
        for k in range(40):
            sr = np.zeros((3, 10))
            sr[:, 2] = 11 + r2.normal(0, 1, 3)
            sr[:, 9] = r2.normal(0, 1, 3)
            sr[:, 3] = o.p["z0"] + r2.normal(0, 0.1, 3)
            eta = r2.normal(0, 0.3, (3, 5, 3))
            nu0 = r2.normal(0, 1, (3, 5))
            acc.append(o.gen_accel(st, sr, 3000.0, 0.1, nu0, eta, 0.04))
            s28 = LIB["mu"] + r2.normal(0, 1, (3, 28)) * LIB["sd"]
            acc.extend(o.step(st, s28, noise=True, rng=nrng))
        outs.append((o.style, np.concatenate(acc), o.n_reject))
    same = outs[0][0] == outs[1][0] and np.array_equal(outs[0][1],
                                                       outs[1][1])
    check("same seed -> identical draw, acceptance and outputs", same,
          f"(rejections {outs[0][2]}, {outs[1][2]})")
    check("another seed -> different outputs",
          not np.allclose(outs[0][1], outs[2][1]))


# ------------------------------------------------------------------ 2
def test_branch_replay():
    print("2. branch replay from a snapshot (float64)")
    o = rich_op(4242, LIB)
    B, n, k = 3, 60, 25
    rng = np.random.default_rng(8)
    xs = R.start_states(ENV, B, rng)
    U = np.stack([np.clip(0.5 + 0.2 * rng.standard_normal((B, n)), 0, 1),
                  np.clip(0.3 * rng.standard_normal((B, n)), -1, 1)], -1)
    seas = seas_for(B, 8)
    for noise in (False, True):
        def fresh():
            return [dict(op=o, st=o.new_state(B, rng=np.random.default_rng(9)),
                         rows=np.arange(B),
                         rng=np.random.default_rng(10) if noise else None)]
        g = fresh()
        a = G.simulate_gen(ENV, xs, U[:, :k], np.zeros(B), seas, g, k,
                           noise=noise)
        snap = dict(g[0], st=o.snapshot(g[0]["st"]),
                    rng=copy.deepcopy(g[0]["rng"]))
        a2 = G.simulate_gen(ENV, a["XS"][:, -1], U[:, k:],
                            np.full(B, k * 0.24), seas, [snap], n - k,
                            noise=noise)
        full = G.simulate_gen(ENV, xs, U, np.zeros(B), seas, fresh(), n,
                              noise=noise)
        err = float(np.abs(full["XS"][:, k:] - a2["XS"]).max())
        err_s = float(np.nanmax(np.abs(full["APK"][:, k:] - a2["APK"])))
        check(f"continuation from step {k} = own continuation "
              f"({'with noise' if noise else 'no noise'})",
              err <= 1e-9 and err_s <= 1e-9,
              f"(max |diff| states {err:.1e}, APK {err_s:.1e})")


# ------------------------------------------------------------------ 3
def test_mission_path():
    print("3. RBPlant + GenInjector (the Mission path) vs simulate_gen")
    from learn.meta import relabel
    env = relabel._env()
    o = rich_op(5151, LIB)
    sea = R.sea_state(R.sea_dict(np.random.default_rng(12)), 50012)
    plant = G.gen_plant(env["mission"].plant).with_sea(
        sea, params=dict(tau_thrust=0.0, rud_rate=0.0, act_family=0.0))
    rng = np.random.default_rng(13)
    n = 40
    U = np.stack([np.clip(0.5 + 0.2 * rng.standard_normal(n), 0, 1),
                  np.clip(0.3 * rng.standard_normal(n), -1, 1)], -1)
    xs0 = R.start_states(env, 1, rng)[0]
    seas = R.RowSeas([sea], env["x_st"], env["y_off"])
    st = o.new_state(1, rng=np.random.default_rng(14))
    inj = G.GenInjector(o, st)
    nrng = np.random.default_rng(15)
    s, t = xs0.copy(), 0.0
    XS = [s.copy()]
    dt, sub = env["dt"], env["sub"]
    for k in range(n):
        sr = R.to_reduced(s[None])
        w15 = R.mid_waves(seas, sr, np.array([t]), 0.5 * dt * sub)
        s28 = R.s28_of(env, s[None], w15, U[k][None])
        rr, rn = o.step(st, s28, noise=True, rng=nrng)
        inj.rule, inj.noise = clip_push(rr[0], rn[0])
        for _ in range(sub):
            s = plant.step(s, t, U[k, 0] * env["t_max"],
                           U[k, 1] * env["rud_max"], dt)
            s = inj(s, dt, plant)
            t += dt
        XS.append(s.copy())
    XS = np.array(XS)
    o2 = G.OperatorGen(o.seed, LIB, accept=False)
    g = [dict(op=o2, st=o2.new_state(1, rng=np.random.default_rng(14)),
              rows=np.array([0]), rng=np.random.default_rng(15))]
    out = G.simulate_gen(env, xs0[None], U[None], np.zeros(1), seas, g, n)
    cols = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]
    err = float(np.abs(out["XS"][0][:, cols] - XS[:, cols]).max())
    check("Mission path = batched simulator", err <= 1e-9,
          f"(max |diff| {err:.1e})")
    apk, amin, hmin = G.safety_per_step(np.array(inj.az)[None],
                                        np.array(inj.h + [np.nan])[None])
    e = max(float(np.abs(apk - out["APK"]).max()),
            float(np.abs(amin - out["AMIN"]).max()),
            float(np.abs(hmin[:, :-1] - out["HMIN"][:, :-1]).max()))
    check("Mission path safety quantities = batched simulator's", e <= 1e-9,
          f"(max |diff| {e:.1e})")


# ------------------------------------------------------------------ 4
def test_acceleration_level():
    print("4. forces enter only at the acceleration level")
    o = rich_op(6060)
    B = 4
    rng = np.random.default_rng(21)
    xs = R.start_states(ENV, B, rng)
    xs[:, 7] = rng.normal(0, 0.5, B)
    U = np.full((B, 1, 2), 0.5)
    seas = seas_for(B, 21)
    with_ = G.simulate_gen(ENV, xs, U, np.zeros(B), seas,
                           [dict(op=o, st=o.new_state(B), rows=np.arange(B),
                                 rng=None)], 1, noise=False, substeps=True)
    alone = G.simulate_gen(ENV, xs, U, np.zeros(B), seas, [], 1,
                           noise=False, substeps=True)
    a, b = with_["SUB"][:, 0, 0], alone["SUB"][:, 0, 0]
    pos = [0, 1, 3, 5, 7]
    e_pos = float(np.abs(a[:, pos] - b[:, pos]).max())
    kick = with_["PS"][:, 0, 0] * ENV["dt"]
    e_vel = float(np.abs(a[:, list(R.RED_VEL)] - b[:, list(R.RED_VEL)]
                         - kick).max())
    check("first substep: positions unchanged, velocities + push dt",
          e_pos == 0.0 and e_vel <= 1e-12 and np.abs(kick).max() > 0,
          f"(|dpos| {e_pos:.1e}, |dvel - push dt| {e_vel:.1e}, "
          f"max |push dt| {np.abs(kick).max():.1e})")


# ------------------------------------------------------------------ 5
def test_lever_arm():
    print("5. point force: lever-arm ratios and signs")
    o = G.OperatorGen(7070, accept=False)
    full = (o.Mt.copy(), o.Minv.copy())
    o.Mt = np.diag(np.diag(o.Mt))
    o.Minv = np.linalg.inv(o.Mt)
    m_v, I_r, m_w, I_q = o.Mt[1, 1], o.Mt[2, 2], o.Mt[3, 3], o.Mt[4, 4]
    th = np.zeros(1)
    ok = True
    for x in (2.0, -1.5):
        a = o.point_accel((x, 0.0, 0.0), np.array([[0.0, 1000.0, 0.0]]),
                          th)[0]
        ok &= np.isclose(a[2] / a[1], x * m_v / I_r, rtol=1e-12)
        ok &= (a[2] > 0) == (x > 0) and a[1] > 0
        a = o.point_accel((x, 0.0, 0.0), np.array([[0.0, 0.0, 1000.0]]),
                          th)[0]
        ok &= np.isclose(a[4] / a[3], -x * m_w / I_q, rtol=1e-12)
        ok &= (a[4] < 0) == (x > 0) and a[3] > 0
    # a forward force above the reference point pitches the bow down
    a = o.point_accel((0.0, 0.0, 0.5), np.array([[1000.0, 0.0, 0.0]]), th)[0]
    ok &= a[4] > 0
    check("ratios x m_v / I_r and -x m_w / I_q, signs of r and pitch rate",
          bool(ok))
    o.Mt, o.Minv = full
    rng = np.random.default_rng(3)
    F = rng.normal(0, 500, (6, 3))
    thv = rng.normal(0, 0.1, 6)
    pt = (1.2, -0.4, -0.3)
    a = o.point_accel(pt, F, thv)
    Q = np.stack([F[:, 0], F[:, 1], pt[0] * F[:, 1] - pt[1] * F[:, 0],
                  F[:, 2] - thv * F[:, 0], pt[2] * F[:, 0] - pt[0] * F[:, 2]],
                 1)
    e = float(np.abs(a @ o.Mt.T - Q).max() / np.abs(Q).max())
    pd = bool(np.linalg.eigvalsh(o.Mt).min() > 0)
    check("with CG offset: M_t a = Q (heave earth-vertical), M_t > 0",
          e < 1e-12 and pd, f"(rel err {e:.1e})")


# ------------------------------------------------------------------ 6
def test_magnitudes():
    print("6. magnitude bounds per force and channel")
    worst = 0.0
    n_forces = 0
    rng = np.random.default_rng(31)
    for s in range(30):
        o = G.OperatorGen(8000 + s, accept=False)
        if not o.forces:
            continue
        n_forces += len(o.forces)
        B = 16
        st = o.new_state(B)
        for k in range(120):
            big = 30.0 if k % 40 > 30 else 1.0          # extreme stretches
            sr = np.zeros((B, 10))
            sr[:, 2] = rng.uniform(3, 30, B)
            sr[:, 9] = rng.normal(0, 2, B) * big
            sr[:, 8] = rng.normal(0, 0.5, B) * big
            sr[:, 4] = rng.normal(0, 1, B) * big
            sr[:, 6] = rng.normal(0, 0.7, B) * big
            sr[:, 5] = rng.uniform(-G.TH_MAX, G.TH_MAX, B)
            sr[:, 3] = rng.normal(0, 0.5, B)
            sr[:, 7] = rng.uniform(-np.pi, np.pi, B)
            eta = rng.normal(0, 0.7, (B, 5, 3)) * big
            nu0 = rng.normal(0, 1, (B, 5)) * o.a_ref * big
            st["hid"][:] = rng.normal(0, 3, st["hid"].shape) * big
            parts = o.gen_accel(st, sr, rng.uniform(0, o.t_max, B),
                                rng.uniform(-o.rud_max, o.rud_max, B), nu0,
                                eta, 0.04, parts=True)
            for i, f in enumerate(o.forces):
                r = np.abs(parts[i]) / (G.SAT * f["kappa"] * o.a_ref)
                worst = max(worst, float(r.max()))
    check("|a_c| <= 3 kappa_i a_ref,c for every force", worst <= 1 + 1e-9,
          f"(worst ratio {worst:.3f} over {n_forces} forces)")


# ------------------------------------------------------------------ 7
def test_acceptance():
    print("7. acceptance rule")
    base = G.OperatorGen(9090, accept=False)
    ops = []
    sd = G.input_library()["sd"]
    for tag in ("none", "pitch", "surge", "heave"):
        o = copy.deepcopy(base)
        o.forces = []
        if tag == "pitch":
            # pitch acceleration + 6 thdot (the low-fidelity boat's own
            # pitch damping is 2 zeta w = 1.15 / s), saturating at 3 x
            # 3 a_ref: strong anti-damping. X[4] = (thdot - mu) / sd
            g = 6.0 * sd[4] / (3.0 * o.a_ref[4])
            o.forces = [manual_gen(o, 4, [0, 0, 0, 0, 3.0 * o.a_ref[4]],
                                   gain=g)]
        elif tag == "surge":
            o.forces = [manual_gen(o, 0, [11.0, 0, 0, 0, 0], bias=10.0)]
        elif tag == "heave":
            # a constant 3 x 30 m/s^2 upward push: clipped at 36 m/s^2 on
            # every substep, the boat sits high but does not diverge
            o.forces = [manual_gen(o, 0, [0, 0, 0, 30.0, 0], bias=10.0)]
        ops.append(o)
    res = G.acceptance_test(ops)
    print(f"     results: {res}; heave clip share {ops[3].test_clip:.2f}")
    check("no forces -> accepted", res[0][0])
    check("pitch anti-damping -> diverged", res[1] == (False, "diverged"))
    check("surge push beyond the speed limit -> diverged",
          res[2] == (False, "diverged"))
    check("bounded large heave push -> accepted, clip share recorded",
          res[3][0] and ops[3].test_clip > 0.9, f"({res[3]})")


# ------------------------------------------------------------------ 8
def test_clip():
    print("8. clip rule")
    o = G.OperatorGen(9191, LIB, accept=False)
    o.forces = [manual_gen(o, 0, [0, 0, 0, 100.0, 0], bias=10.0)]
    B, n = 2, 10
    rng = np.random.default_rng(41)
    xs = R.start_states(ENV, B, rng)
    U = np.full((B, n, 2), 0.5)
    g = [dict(op=o, st=o.new_state(B, rng=np.random.default_rng(1)),
              rows=np.arange(B), rng=np.random.default_rng(2))]
    out = G.simulate_gen(ENV, xs, U, np.zeros(B), seas_for(B, 41), g, n,
                         substeps=True, parts=True)
    PS = out["PS"]
    rule = 100.0 * G.SAT * np.tanh(10.0 / G.SAT)
    cor = out["PARTS"][0]["cor"]                        # step means
    ok_bound = bool((np.abs(PS) <= LIM + 1e-12).all())
    exp_h = np.clip(rule + cor[:, :, 3:4] + out["NOISE"][:, :, 3:4], -LIM[3],
                    LIM[3])
    ok_h = bool(np.allclose(PS[..., 3], exp_h, atol=1e-6))
    ok_cnt = bool((out["clip_rule"] == 6).all())
    check("push within CLIP x A_REF; heave = clip(rule + noise); every "
          "substep counted as rule-clipped", ok_bound and ok_h and ok_cnt,
          f"(max |push| / lim {np.abs(PS / LIM).max():.3f})")


# ------------------------------------------------------------------ 9
def test_safety():
    print("9. safety quantities vs an independent computation")
    o = rich_op(9292, LIB)
    B, n = 3, 30
    rng = np.random.default_rng(51)
    xs = R.start_states(ENV, B, rng)
    U = np.stack([np.clip(0.5 + 0.2 * rng.standard_normal((B, n)), 0, 1),
                  np.clip(0.3 * rng.standard_normal((B, n)), -1, 1)], -1)
    raw = [R.sea_state(R.sea_dict(np.random.default_rng([51, i])), 50000 + i)
           for i in range(B)]
    seas = R.RowSeas(raw, ENV["x_st"], ENV["y_off"])
    g = [dict(op=o, st=o.new_state(B, rng=np.random.default_rng(1)),
              rows=np.arange(B), rng=np.random.default_rng(2))]
    out = G.simulate_gen(ENV, xs, U, np.zeros(B), seas, g, n, substeps=True)
    SUB, dt, sub = out["SUB"], ENV["dt"], ENV["sub"]
    zd = np.concatenate([xs[:, None, 8], SUB[:, :, :, 4].reshape(B, -1)], 1)
    az = np.diff(zd, axis=1).reshape(B, n, sub) / dt
    e_a = max(float(np.abs(az.max(-1) - out["APK"]).max()),
              float(np.abs(az.min(-1) - out["AMIN"]).max()))
    p = ENV["red"].p
    fb, x_b, sp = G.freeboard(p), ENV["x_st"][-1], p["sign_pitch"]
    H = np.zeros((B, n))
    for b in range(B):
        for k in range(n):
            hj = []
            for j in range(sub):
                s = SUB[b, k, j]
                tt = (k * sub + j + 1) * dt
                c, sn = np.cos(s[7]), np.sin(s[7])
                X = s[0] + x_b * c - ENV["y_off"] * sn
                Y = s[1] + x_b * sn + ENV["y_off"] * c
                e = np.asarray(raw[b].eta(X, Y, tt), float)
                hj.append(fb + s[3] + sp * x_b * s[5] - e.max())
            H[b, k] = min(hj)
    e_h = float(np.abs(H - out["HMIN"]).max())
    check("APK / AMIN = max / min of the substep zdot differences",
          e_a <= 1e-9, f"(max |diff| {e_a:.1e})")
    check("HMIN = min over substep ends of the bow height (sea.eta)",
          e_h <= 1e-8, f"(max |diff| {e_h:.1e} m)")


# ----------------------------------------------------------------- 10
def test_library():
    print("10. input library and library-standardised filter outputs")
    L = G.input_library()
    o = G.OperatorGen(0, accept=False)
    s0 = G.LIB_SKIP * L["sub"]
    Xs = (L["raw"][:, s0:] - L["mu"]) / L["sd"]
    e_col = max(float(np.abs(Xs.mean((0, 1))).max()),
                float(np.abs(Xs.std((0, 1)) - 1).max()))
    check("library columns standardised (mean 0, std 1)", e_col < 1e-9,
          f"(max dev {e_col:.1e}; library {L['raw'].shape})")
    worst, n_rel, n_sw = 0.0, 0, 0
    for s in range(12):
        o = G.OperatorGen(3000 + s, accept=False)
        if not o.forces:
            continue
        Xl = o._lib_inputs()
        for f in o.forces:
            fl = copy.deepcopy(f["filt"])
            Y = o._lib_filters(f["Wc"], fl, Xl)
            lin, rel = fl["lin"], fl["rel"]
            same = (np.allclose(fl["y_sd"], f["filt"]["y_sd"])
                    and np.allclose(fl["rel_hi"], f["filt"]["rel_hi"]))
            if len(lin):
                worst = max(worst, float(np.abs(Y[:, lin].std(0) - 1).max()),
                            0.0 if same else 1.0)
            for j in rel:
                n_rel += 1
                n_sw += int(np.ptp(Y[:, j]) > 0)
    check("filter outputs on the library: std 1 (linear filters)",
          worst < 1e-6, f"(max |std - 1| {worst:.1e})")
    check("relays switch on the library for >= 30% of relay filters",
          n_rel == 0 or n_sw >= 0.3 * n_rel, f"({n_sw} of {n_rel})")


if __name__ == "__main__":
    t0 = time.time()
    for fn in (test_reproducible, test_branch_replay, test_mission_path,
               test_acceleration_level, test_lever_arm, test_magnitudes,
               test_acceptance, test_clip, test_safety, test_library):
        t = time.time()
        fn()
        print(f"     ({time.time() - t:.0f} s)")
    print(f"\n{len(FAIL)} failed ({', '.join(FAIL) if FAIL else '-'}); "
          f"{time.time() - t0:.0f} s")
    sys.exit(1 if FAIL else 0)
