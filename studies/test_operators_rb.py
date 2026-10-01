#!/usr/bin/env python3
"""
Tests of the rigid-body operator family (learn/meta/operators_rb.py;
PRIOR_DERIVATION.md D8.12). Minutes, one process.

Checks marked [regression] only confirm the construction (they are true by
how the draws are built and would not catch a wrong formula); they stay as
guards against edits. The others are behavioural: they exercise the code
path on states or loads where a wrong formula would show (review of M16
item 8).

  1  on random draws:
     [regression] inertias positive and ordered; load filters stable, zero
     heave / pitch gain at zero frequency, band gain <= 1
     behavioural: the total damping never does positive work; the T2c
     couplings plus the low-fidelity Coriolis term do zero work (sway
     saturation included); point-load signs and rigid-body ratios; every
     drawn load's heave and pitch accelerations stay under their caps in
     the worst case over its readings (pitch lever z Fx included); T5 at
     the nominal draw is zero, and its axial-loss share gives exactly the
     declared surge term; a hull that tracks the low-fidelity water
     reference along a moving path (12 m/s, long and short regular waves,
     head / following / beam, straight and turning) gets zero vertical
     push; the stability rule rejects an undamped draw; rb_off and
     disabled T8 / T9 give zeros
  2  reproducible from the seed
  3  magnitudes: 60 s closed loop, 40 draws, rigid-body push alone: per
     channel sizes; clipped control steps (any channel, any substep; the
     D5 rule) < 0.1%
  4  90 s closed loop with random draws and the M10 actuator family
     (lofi.draw_act): finite, rigid-body part bounded against the low-
     fidelity boat alone
  5  branch replay (float64): continuing from a snapshot at step k equals
     the run's own continuation (rule only, and with noise) to <= 1e-9
  6  the Mission path (RBPlant + RBInjector, single row) equals
     simulate_rb to <= 1e-9
  7  interface: run() raises, set_slow restores the nested state and its
     clock, RBInjector.r behaves as data2's ZOHInjector.r

    python studies/test_operators_rb.py [--quick]
"""
import argparse
import copy
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from learn.meta import operators_rb as R
from learn.meta.operators import A_REF, CLIP, clip_push

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
L_ = np.load(os.path.join(ROOT, "studies", "_cache", "meta3", "lib.npz"))
LIB = {k: L_[k] for k in L_.files}
ENV = R.light_env()
FAIL = []


def check(name, ok, msg=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {msg}")
    if not ok:
        FAIL.append(name)


def seas_for(B, seed0=0):
    return R.RowSeas([R.sea_state(R.sea_dict(np.random.default_rng(
        [seed0, i])), 50000 + i) for i in range(B)], ENV["x_st"],
        ENV["y_off"])


def groups_for(ops, noise=True):
    return [dict(op=o, st=o.new_state(1, rng=np.random.default_rng(
        [o.seed, 1])), rows=np.array([i]),
        rng=np.random.default_rng([o.seed, 1]) if noise else None)
        for i, o in enumerate(ops)]


def act_rows(seeds):
    """Stacked M10 actuator draws (data2's 'm10' family), one per row."""
    from sim import lofi
    p = ENV["red"].p
    ds = [lofi.draw_act(np.random.default_rng([int(s), 11]), p, ENV["dt"])
          for s in seeds]
    return {k: np.array([d[k] for d in ds]) for k in lofi.ACT_KEYS}


def path(t, u, v, r, psi0):
    """Poses of a hull moving with constant body velocities (u, v) and yaw
    rate r (the reduced model's kinematics), from the origin."""
    psi = psi0 + r * t
    if abs(r) < 1e-12:
        c, s = np.cos(psi0), np.sin(psi0)
        return (u * c - v * s) * t, (u * s + v * c) * t, psi
    x = (u * np.sin(psi) + v * np.cos(psi)
         - u * np.sin(psi0) - v * np.cos(psi0)) / r
    y = (-u * np.cos(psi) + v * np.sin(psi)
         + u * np.cos(psi0) - v * np.sin(psi0)) / r
    return x, y, psi


# ------------------------------------------------------------------ 1
def test_constraints():
    print("1. constraints on random draws")
    rng = np.random.default_rng(0)
    n = 30 if args.quick else 80
    ops = [R.OperatorRB(1000 + s, LIB) for s in range(n)]
    on = [o for o in ops if not o.rb_off]
    m_ok = all((o.masses() > 0).all() and o.A11 < o.A22 and o.m_w > o.m
               for o in on)
    check("[regression] inertias positive, A11 < A22, m_w > m", m_ok)
    # total damping (low-fidelity + dissipative rigid-body parts) never does
    # positive work, in the metric of the drawn inertias
    worst = -np.inf
    for o in on:
        sr = np.zeros((400, 10))
        sr[:, 2] = rng.uniform(o.u_lo - 3, o.u_hi + 3, 400)
        sr[:, 9] = rng.normal(0, 2.0, 400)
        sr[:, 8] = rng.normal(0, 0.6, 400)
        sr[:, 4] = rng.normal(0, 1.5, 400)
        sr[:, 6] = rng.normal(0, 0.4, 400)
        sr[:, 3] = o.p["z0"] + rng.normal(0, 0.3, 400)
        sr[:, 5] = o.p["th0"] + rng.normal(0, 0.05, 400)
        a = o.damping_accel(sr)
        nu = sr[:, [2, 9, 8, 4, 6]]
        pw = (a * nu * o.masses()).sum(1)
        scale = (np.abs(a * nu) * o.masses()).sum(1) + 1e-9
        worst = max(worst, float((pw / scale).max()))
    check("damping power <= 0", worst <= 1e-12, f"(max relative {worst:+.2e})")
    # T2c + the low-fidelity Coriolis term (-m u r in sway) do no work, for
    # any drawn multiplier, including drift angles where the Munk terms
    # saturate (eta = 0 and the hull on its running attitude, so the pitch
    # Munk stiffness, a stiffness, is zero)
    worst = 0.0
    for o in on[:30]:
        o = copy.deepcopy(o)
        o.t2c = True
        o.c2["lam"] = np.array([rng.uniform(*R.LAM), rng.normal(0, 0.3),
                                rng.normal(0, 0.1)])
        o.c2["zq"] = np.array([rng.normal(0, 3.0), rng.normal(0, 1.0), 0.0])
        o.c2["M"] = np.array([0.2, 0.1, 0.0])
        o._pack()
        o.enabled = {"T2c"}
        B = 400
        sr = np.zeros((B, 10))
        sr[:, 2] = rng.uniform(2.0, 20.0, B)
        sr[:, 9] = rng.normal(0, 3.0, B)
        sr[:, 8] = rng.normal(0, 0.6, B)
        sr[:, 4] = rng.normal(0, 1.5, B)
        sr[:, 6] = rng.normal(0, 0.4, B)
        sr[:, 3], sr[:, 5] = o.p["z0"], o.p["th0"]
        z = np.zeros((B, 5, 3))
        a = o.rb_accel(o.new_state(B), sr, 3000.0, 0.1, z, z, 0.04)
        nu = sr[:, [2, 9, 8, 4, 6]]
        lofi_pw = -o.m * sr[:, 2] * sr[:, 8] * sr[:, 9]
        pw = (a * nu * o.masses()).sum(1) + lofi_pw
        scale = (np.abs(a * nu) * o.masses()).sum(1) + np.abs(lofi_pw)
        worst = max(worst, float(np.abs(pw / scale).max()))
    check("T2c + low-fidelity Coriolis: zero power", worst < 1e-12,
          f"(max relative {worst:.1e})")
    # load filters
    kmax, dc, g_ok = 0.0, 0.0, True
    for o in on:
        for md in o.modes:
            K = md["K"]
            Ad, Bd = K.disc(0.04)
            kmax = max(kmax, float(np.abs(np.linalg.eigvals(Ad)).max()))
            g_ok &= bool(K.gain(np.linspace(0.3, 13.0, 1000)).max()
                         <= 1.0 + 1e-3)
            Cf, Df, Ch, Dh = md["out"]
            x = np.zeros((1, K.order))
            for _ in range(4000):              # 160 s of a constant input
                yh = x @ Ch[0] + Dh[0]
                x = x @ Ad.T + Bd
            dc = max(dc, float(abs(yh[0])))
    check("[regression] load filters stable", kmax < 1.0,
          f"(max |eig| {kmax:.4f})")
    check("[regression] heave / pitch load gain 0 at zero frequency",
          dc < 1e-6, f"(|y_hp| after 160 s of a step {dc:.1e})")
    check("[regression] band gain <= 1", g_ok)
    # load caps: worst case over each drawn load's readings (1 per library
    # std at every point, signs free), through load_accel itself
    ratio_h, ratio_p, n_sp, n_md = 0.0, 0.0, 0, 0
    for o in on:
        for md in o.modes:
            n_md += 1
            n_sp += md.get("sp", 1.0) < 1.0
            if md["kind"] == "local":
                a = o.load_accel(md, np.zeros((15, 15)), np.eye(15))
                h, q = np.abs(a[:, 3]).sum(), np.abs(a[:, 4]).sum()
            else:
                a = o.load_accel(md, np.zeros(1), np.ones(1))[0]
                h, q = abs(a[3]), abs(a[4])
            ratio_h = max(ratio_h, h / md["A"])
            ratio_p = max(ratio_p, q / (md["A"] * o.a_env_p / o.a_env))
    check("every load: heave <= A and pitch <= A a_env_p / a_env per unit "
          "reading (worst case)", ratio_h <= 1 + 1e-9 and ratio_p <= 1 + 1e-9,
          f"(max ratios {ratio_h:.3f} / {ratio_p:.3f}; {n_sp} of {n_md} loads"
          f" limited by the pitch-lever cap)")
    # point-load signs and ratios
    o = on[0]
    F = np.array([[0.0, 0.0, 100.0]])
    a = o.point_accel((2.0, 0.0, 0.0), F)[0]
    check("bow-up force: heave +, pitch - (bow rises)",
          a[3] > 0 and a[4] < 0)
    check("heave : pitch = 1 : -x m_w / I_q",
          abs(a[4] / a[3] + 2.0 * o.m_w / o.I_q) < 1e-12)
    a = o.point_accel((-2.0, 0.0, 0.0), np.array([[0.0, 100.0, 0.0]]))[0]
    check("stern force to port: sway +, yaw -", a[1] > 0 and a[2] < 0)
    check("sway : yaw = 1 : x m_v / I_r",
          abs(a[2] / a[1] + 2.0 * o.m_v / o.I_r) < 1e-12)
    a = o.point_accel((0.0, 0.0, -0.3), np.array([[100.0, 0.0, 0.0]]))[0]
    check("thrust below the CG: bow up (pitch accel < 0)", a[4] < 0)
    # T5 at the nominal draw, and the declared axial-loss term
    o = copy.deepcopy(on[1])
    o.ceT[:] = 0.0
    o.ceS[:] = 0.0
    o.c3, o.x_j, o.z_j, o.kc = 0.0, o.x_n0, 0.0, 0.0
    o.rho_r = np.sqrt(o.x_n0 / (o.m_v * o.k_nf_tau))
    o.I_r = o.m_v * o.rho_r ** 2
    o.f_r = (o.x_n0 / o.I_r) / o.k_nf_tau
    o._pack()
    o.enabled = {"T5"}
    sr = np.zeros((50, 10))
    sr[:, 2] = rng.uniform(8, 15, 50)
    sr[:, 8] = rng.normal(0, 0.3, 50)
    thr = rng.uniform(0, o.p["t_max"], 50)
    noz = rng.uniform(-0.4, 0.4, 50)
    z53 = np.zeros((50, 5, 3))
    a1 = o.rb_accel(o.new_state(50), sr, thr, noz, z53, z53, 0.04)
    check("T5 zero at the nominal draw (kc = 0, nozzle over, all channels)",
          np.abs(a1).max() < 1e-12, f"({np.abs(a1).max():.1e})")
    o.kc = 0.6
    a2 = o.rb_accel(o.new_state(50), sr, thr, noz, z53, z53, 0.04)
    want = 0.6 * thr * (np.cos(noz) - 1.0) / o.m_u
    err = max(float(np.abs(a2[:, 0] - want).max()),
              float(np.abs(a2[:, 1:]).max()))
    check("T5 axial-loss share: surge = kc T (cos d - 1) / m_u, rest 0",
          err < 1e-12, f"(max |diff| {err:.1e}; surge up to "
          f"{np.abs(want).max():.2f} m/s^2)")
    # a hull that tracks the low-fidelity water reference along a MOVING
    # path gets no vertical push: its displacement and rate equal the
    # reference's, the rate taken by finite differences along the path
    # (independent of the code's encounter-rate formula)
    worst_rate, worst_push, old_push = 0.0, 0.0, 0.0
    tt = np.arange(0.0, 16.0, 0.04)
    hfd = 1e-4
    cases = [(np.pi, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5 * np.pi, 0.0, 0.0),
             (2.0, 0.8, 0.1)]
    for o in on[:8]:
        o = copy.deepcopy(o)
        o.t3 = o.t4 = o.t2c = True
        o.cS[2] = np.array([np.log(2.0), 0.1, 0.0])
        o.cS[3] = np.array([np.log(2.0), 0.1, 0.0])
        o.kap = np.array([0.0, 0.0, 0.3, 0.3])
        o.cKz = np.array([np.log(1.3), 0.0, 0.0])
        o.cKq = np.array([np.log(0.8), 0.0, 0.0])
        o.c2["zq"] = np.array([2.0, 0.3, 0.0])
        o.c2["M"] = np.array([0.2, 0.0, 0.0])
        o.cdz[:] = 0.0
        o.cdq[:] = 0.0
        o._pack()
        o.enabled = {"T3", "T4", "T2c"}
        for lam_w in (100.0, 30.0):
            k = 2 * np.pi / lam_w
            for th, v, r in cases:
                sea = SimpleNamespace(a=np.array([0.5]), k=np.array([k]),
                                      th=np.array([th]),
                                      w=np.array([np.sqrt(9.81 * k)]),
                                      phi=np.array([0.3]))
                B = len(tt)
                seas = R.RowSeas([sea] * B, ENV["x_st"], ENV["y_off"])
                u = np.full(B, 12.0)
                vv, rr = np.full(B, v), np.full(B, r)

                def ref(t):
                    x, y, psi = path(t, 12.0, v, r, 0.0)
                    e, ed = seas.stations(x, y, psi, t, vel=(u, vv, rr))
                    return (x, y, psi), o._ref(e, ed), (e, ed)
                pose, (zr, tr, zdr, tdr), (e, ed) = ref(tt)
                _, (zp, tp, _, _), _ = ref(tt + hfd)
                _, (zm, tm, _, _), _ = ref(tt - hfd)
                zd_fd, td_fd = (zp - zm) / (2 * hfd), (tp - tm) / (2 * hfd)
                # relative to the channel's size (floor: in beam seas the
                # centre-line slope, and so the pitch reference, is 0)
                worst_rate = max(worst_rate, float(
                    np.abs(zdr - zd_fd).max() / (np.abs(zd_fd).max() + 1e-3)),
                    float(np.abs(tdr - td_fd).max()
                          / (np.abs(td_fd).max() + 1e-3)))
                sr = np.zeros((B, 10))
                sr[:, 0], sr[:, 1], sr[:, 7] = pose
                sr[:, 2], sr[:, 9], sr[:, 8] = u, vv, rr
                sr[:, 3], sr[:, 4], sr[:, 5], sr[:, 6] = zr, zd_fd, tr, td_fd
                a = o.rb_accel(o.new_state(B), sr, 3000.0, 0.0, e, ed, 0.04)
                worst_push = max(worst_push, float(np.abs(a[:, 3:]).max()))
                # the same hull with the fixed-point rate (the old input)
                ef = seas.stations(*pose, tt)[1]
                a = o.rb_accel(o.new_state(B), sr, 3000.0, 0.0, e, ef, 0.04)
                old_push = max(old_push, float(np.abs(a[:, 3:]).max()))
    check("reference rate = finite difference along the moving path",
          worst_rate < 1e-6, f"(max relative {worst_rate:.1e})")
    check("zero vertical push on a hull tracking the moving reference",
          worst_push < 1e-6, f"({worst_push:.1e} m/s^2; with the fixed-"
          f"point rate it was {old_push:.2f})")
    # stability rule
    o = copy.deepcopy(on[2])
    ok_before = o.stable()
    o.cS[2] = np.array([np.log(1e-3), 0.0, 0.0])
    o.cS[3] = np.array([np.log(1e-3), 0.0, 0.0])
    o.c2["zq"] = np.zeros(3)
    check("stability rule: accepted draw stable, undamped draw rejected",
          ok_before and not o.stable())
    rej = np.mean([o.n_reject > 0 for o in ops])
    print(f"     draws needing a redraw: {rej:.2f}; rb off: "
          f"{np.mean([o.rb_off for o in ops]):.2f}")
    # rb_off and disabled T8 / T9
    o = R.OperatorRB(7, LIB, rb_off=True)
    st = o.new_state(4, rng=np.random.default_rng(1))
    eta = rng.normal(0, 0.4, (4, 5, 3))
    a = o.rb_accel(st, sr[:4], 3000.0, 0.1, eta, eta, 0.04)
    o.enabled = set(R.COMPS)
    s28 = np.tile(LIB["mu"], (4, 1))
    r_, n_ = o.step(st, s28, noise=True, rng=np.random.default_rng(2))
    check("rb_off: zero rigid-body push; T8 / T9 off: zero step",
          np.abs(a).max() == 0 and np.abs(r_).max() == 0
          and np.abs(n_).max() == 0)
    return ops


# ------------------------------------------------------------------ 2
def test_reproducible():
    print("2. reproducible from the seed")
    outs = []
    for seed in (11, 11, 12):
        o = R.OperatorRB(seed, LIB)
        st = o.new_state(3, rng=np.random.default_rng([seed, 1]))
        nrng = np.random.default_rng([seed, 1])
        r2 = np.random.default_rng(5)
        acc = []
        for k in range(40):
            sr = np.zeros((3, 10))
            sr[:, 2] = 11 + r2.normal(0, 1, 3)
            sr[:, 9] = r2.normal(0, 1, 3)
            sr[:, 3] = o.p["z0"] + r2.normal(0, 0.2, 3)
            eta = r2.normal(0, 0.3, (3, 5, 3))
            acc.append(o.rb_accel(st, sr, 3000.0, 0.1, eta, eta, 0.04))
            s28 = LIB["mu"] + r2.normal(0, 1, (3, 28)) * LIB["sd"]
            acc.extend(o.step(st, s28, noise=True, rng=nrng))
        outs.append((o.style, np.concatenate(acc)))
    same = outs[0][0] == outs[1][0] and np.array_equal(outs[0][1],
                                                       outs[1][1])
    diff = not np.allclose(outs[0][1], outs[2][1])
    check("same seed -> identical draw and outputs", same)
    check("another seed -> different outputs", diff)


# ------------------------------------------------------------------ 3
def test_magnitudes():
    print("3. magnitudes (60 s closed loop, rigid-body part alone)")
    n = 16 if args.quick else 40
    ops = [R.OperatorRB(2000 + s, LIB) for s in range(n)]
    for o in ops:
        o.enabled = set(R.COMPS)
    rng = np.random.default_rng(4)
    xs = R.start_states(ENV, n, rng)
    out = R.simulate_rb(ENV, xs, R.Autopilot(ENV, n, rng), np.zeros(n),
                        seas_for(n, 4), groups_for(ops, noise=False), 250,
                        noise=False, parts=True)
    P = out["P"][:, 40:]
    rms = np.sqrt((P ** 2).mean(1))
    q = np.quantile(rms, [0.5, 0.95], axis=0)
    print("     per-draw rms of the rigid-body push, median / 95%: "
          + ", ".join(f"{c} {a:.2f}/{b:.2f}" for c, a, b in
                      zip(R.CHANNELS, q[0], q[1])))
    check("[regression] 95% of draws below CLIP x A_REF",
          (q[1] < CLIP * A_REF).all())
    # the D5 rule: control steps with any channel clipped at any substep
    step_clip = float((out["clip"][:, 40:].sum(-1) > 0).mean())
    check("clipped control steps (D5 rule) < 0.1%", step_clip < 1e-3,
          f"({step_clip:.2e})")
    return out


# ------------------------------------------------------------------ 4
def test_closed_loop():
    print("4. 90 s closed loop, random draws, M10 actuator family: low-"
          "fidelity alone, rigid-body part, all parts")
    n = 8 if args.quick else 16
    ext = {}
    act = act_rows(3000 + np.arange(n))
    slow = np.mean([act["noz_tau"][i] > 0.2 or act["noz_delay"][i] >= 4
                    for i in range(n)])
    print(f"     nozzle: tau {act['noz_tau'].min():.2f}-"
          f"{act['noz_tau'].max():.2f} s, delay up to "
          f"{act['noz_delay'].max():.0f} substeps, rate "
          f"{np.where(act['noz_rate'] > 0, act['noz_rate'], np.inf).min():.2f}"
          f" rad/s min; share of slow nozzles {slow:.2f}")
    for tag, en in (("none", set()), ("rb", set(R.COMPS)),
                    ("all", set(R.ALL))):
        ops = [R.OperatorRB(3000 + s, LIB) for s in range(n)]
        for o in ops:
            o.enabled = en
        rng = np.random.default_rng(6)
        xs = R.start_states(ENV, n, rng)
        t = time.time()
        out = R.simulate_rb(ENV, xs, R.Autopilot(ENV, n, rng), np.zeros(n),
                            seas_for(n, 6), groups_for(ops), 375, act=act)
        XS = out["XS"]
        ext[tag] = dict(
            fin=bool(np.isfinite(XS).all()),
            v=np.abs(XS[:, :, 7]).max(1),
            z=np.abs(XS[:, :, 2] - ENV["red"].p["z0"]).max(1),
            th=np.abs(XS[:, :, 4]).max(1), u=XS[:, :, 6].min(1),
            clip=float((out["clip"].sum(-1) > 0).mean()))
        e = ext[tag]
        print(f"     {tag:<4} {time.time() - t:3.0f} s: max |v| {e['v'].max():.2f}"
              f" m/s, max |z - z0| {e['z'].max():.2f} m, max |pitch| "
              f"{e['th'].max():.2f} rad, min speed {e['u'].min():.1f} m/s, "
              f"clipped steps {e['clip']:.1e}")
    a, b = ext["none"], ext["rb"]
    check("finite", all(e["fin"] for e in ext.values()))
    # the rigid-body part must not make the boat diverge where the low-
    # fidelity boat alone does not (its own hard turns reach ~6 m/s of
    # sway; the old family's offsets can stop it, which is kept as is)
    check("rigid-body part bounded vs the low-fidelity boat alone",
          (b["v"] <= 1.5 * a["v"] + 1.5).all() and b["th"].max() < 0.6
          and b["z"].max() < 3.0 and b["u"].min() > 2.0)


# ------------------------------------------------------------------ 5
def test_branch_replay():
    print("5. branch replay from a snapshot (float64)")
    o = R.OperatorRB(4242, LIB)
    while o.rb_off or not o.modes:
        o = R.OperatorRB(o.seed + 1, LIB)
    B, n, k = 3, 60, 25
    rng = np.random.default_rng(8)
    xs = R.start_states(ENV, B, rng)
    U = np.stack([np.clip(0.5 + 0.2 * rng.standard_normal((B, n)), 0, 1),
                  np.clip(0.3 * rng.standard_normal((B, n)), -1, 1)], -1)
    seas = seas_for(B, 8)
    for noise in (False, True):
        st = o.new_state(B, rng=np.random.default_rng(9))
        g = [dict(op=o, st=st, rows=np.arange(B),
                  rng=np.random.default_rng(10) if noise else None)]
        a = R.simulate_rb(ENV, xs, U[:, :k], np.zeros(B), seas, g, k,
                          noise=noise, freeze=not noise)
        snap = dict(g[0], st=o.snapshot(g[0]["st"]),
                    rng=copy.deepcopy(g[0]["rng"]))
        a2 = R.simulate_rb(ENV, a["XS"][:, -1], U[:, k:], np.full(B, k * 0.24),
                           seas, [snap], n - k, noise=noise,
                           freeze=not noise)
        st = o.new_state(B, rng=np.random.default_rng(9))
        g = [dict(op=o, st=st, rows=np.arange(B),
                  rng=np.random.default_rng(10) if noise else None)]
        full = R.simulate_rb(ENV, xs, U, np.zeros(B), seas, g, n,
                             noise=noise, freeze=not noise)
        err = float(np.abs(full["XS"][:, k:] - a2["XS"]).max())
        check(f"continuation from step {k} = own continuation "
              f"({'rule + noise' if noise else 'rule only'})", err <= 1e-9,
              f"(max |diff| {err:.1e})")


# ------------------------------------------------------------------ 6
def test_mission_path():
    print("6. RBPlant + RBInjector (the Mission path) vs simulate_rb")
    from learn.meta import relabel
    env = relabel._env()
    o = R.OperatorRB(5151, LIB)
    while o.rb_off or not o.modes:
        o = R.OperatorRB(o.seed + 1, LIB)
    sea = R.sea_state(R.sea_dict(np.random.default_rng(12)), 50012)
    plant = R.rb_plant(env["mission"].plant).with_sea(
        sea, params=dict(tau_thrust=0.0, rud_rate=0.0, act_family=0.0))
    rng = np.random.default_rng(13)
    n = 40
    U = np.stack([np.clip(0.5 + 0.2 * rng.standard_normal(n), 0, 1),
                  np.clip(0.3 * rng.standard_normal(n), -1, 1)], -1)
    xs0 = R.start_states(env, 1, rng)[0]
    seas = R.RowSeas([sea], env["x_st"], env["y_off"])
    st = o.new_state(1, rng=np.random.default_rng(14))
    inj = R.RBInjector(o, st)
    nrng = np.random.default_rng(15)
    s = xs0.copy()
    t = 0.0
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
    o2 = R.OperatorRB(o.seed, LIB)
    g = [dict(op=o2, st=o2.new_state(1, rng=np.random.default_rng(14)),
              rows=np.array([0]), rng=np.random.default_rng(15))]
    out = R.simulate_rb(env, xs0[None], U[None], np.zeros(1), seas, g, n)
    cols = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]
    err = float(np.abs(out["XS"][0][:, cols] - XS[:, cols]).max())
    check("Mission path = batched simulator", err <= 1e-9,
          f"(max |diff| {err:.1e})")


# ------------------------------------------------------------------ 7
def test_interface():
    print("7. interface (run, set_slow, RBInjector.r)")
    o = R.OperatorRB(6161, LIB)
    try:
        o.run(np.zeros((1, 3, 26)))
        raised = False
    except NotImplementedError:
        raised = True
    check("run() raises NotImplementedError (use simulate_rb)", raised)
    rng = np.random.default_rng(16)
    st = o.new_state(2, rng=np.random.default_rng(17))
    for _ in range(30):
        o.step(st, LIB["mu"] + rng.normal(0, 1, (2, 28)) * LIB["sd"],
               noise=True, rng=rng)
    slow = {k: (v[:1] if isinstance(v, np.ndarray) else
                [x[:1] for x in v] if isinstance(v, list) else v)
            for k, v in o.slow_state(st).items()}
    st2 = o.new_state(3)
    R.set_slow(o, st2, slow)
    ok = (st2["k"] == st2["res"]["k"] == slow["k"]
          and np.allclose(st2["res"]["ga"], slow["ga"][0])
          and np.allclose(st2["res"]["bb"], slow["bb"][0]))
    check("set_slow restores the nested slow state and one clock", ok,
          f"(k = {st2['k']})")
    inj = R.RBInjector(o, o.new_state(1))
    inj.r = np.arange(5.0)
    ok = np.array_equal(inj.r, np.arange(5.0)) and not inj.noise.any()
    check("RBInjector.r: set = held part, read before a substep = held", ok)


if __name__ == "__main__":
    t0 = time.time()
    test_constraints()
    test_reproducible()
    test_magnitudes()
    test_closed_loop()
    test_branch_replay()
    test_mission_path()
    test_interface()
    print(f"\n{len(FAIL)} failed ({', '.join(FAIL) if FAIL else '-'}); "
          f"{time.time() - t0:.0f} s")
    sys.exit(1 if FAIL else 0)
