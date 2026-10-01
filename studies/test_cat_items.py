#!/usr/bin/env python3
"""
Construction checks of the D10 force catalogue (learn/meta/cat_base.py,
items learn/meta/cat_*.py; learn/meta/PRIOR_D10_DRAFT.md 7.1 and section 6
item 6). One process, well under a minute, a few MB.

Framework (always):
  F1  numerical helpers: exp_update stable for dt / tau = 50; osc_phi equals
      control.reduced.ReducedModel._phi and scipy expm (negative stiffness
      and damping too); a stiff lightly damped oscillator (9 rad/s, zeta
      0.005) never gains energy; implicit quadratic drag never reverses the
      velocity and never does positive work; the implicit impulse never
      reverses the entry velocity and |dV| <= V; lever_Q equals
      operators_gen.OperatorGen._point_Q (s_p = -1); a unit impulse at a
      point changes that point's velocity by 1 / point_mass
  F2  CatSea: elevation and moving-station rate equal operators_rb.RowSeas
      .stations(vel=...); the orbital velocity of a single component is
      a omega cos(phase) along its direction, rotated into body axes
  F3  shared quantities at calm steady running (any speed): keel immersion
      = the calm-running value (> 0 at every station), entry velocity 0,
      tau_e = tau_run, intake immersion = h_run > 0, the low-fidelity
      boat's own heave / pitch acceleration 0
  F4  tables: P_ON for every code, the propulsion order of draft 3.5,
      layer-3 tier and sparse-episode frequencies, sparse episodes switch on
      1-3 items
  F5  assembly (needs studies/_cache/meta5/lib.npz, read only): overlap
      switches applied to layer 1 (T8 off -> held rule exactly 0, T4 speed
      polynomial constant, bounded tanh stiffness off, heave-pitch
      coupling off with W1, T2c rigid m v r removed), parts sum to the total, a short closed loop in a sea stays
      finite
Per registered item:
  I1  draw deterministic from its stream; switch-on rate = P_ON
  I2  random and extreme states (airborne, deep, fast, slow) in a sea:
      finite outputs of the right shape; |a| <= SAT x nominal after the
      framework's soft saturation
  I3  calm_zero items: exactly zero in calm steady straight running at u_id
      (raw draws) and at all speeds (calm_params); stateful items over 1 s
  I4  closed loop with the item alone: a calm steady row stays at rest
      (calm_zero items), sea rows finite
  I5  the item module's own checks (@cat_base.check)

    python studies/test_cat_items.py [--items W1,...] [--no-assembly]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from learn.meta import cat_base as CB                         # noqa: E402
from learn.meta import operators_gen as G                     # noqa: E402
from learn.meta import operators_rb as R                      # noqa: E402

RES = []


def rep(name, ok, msg=""):
    RES.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}", flush=True)


def guard(name, fn, *a):
    try:
        ok, msg = fn(*a)
    except Exception as e:                    # a crash is a failure
        ok, msg = False, f"{type(e).__name__}: {e}"
    rep(name, ok, msg)


def a_sea(ctx, B, seed=11, hs=None):
    rng = np.random.default_rng(seed)
    d = R.sea_dict(rng)
    if hs is not None:
        d["hs"] = hs
    sea = R.sea_state(d, seed)
    return R.RowSeas([sea] * B, ctx.x_st, ctx.y_off)


def running(ctx, u, B=None):
    u = np.atleast_1d(np.asarray(u, float))
    sr = np.zeros((len(u), 10))
    sr[:, 2], sr[:, 3], sr[:, 5] = u, ctx.z0, ctx.th0
    return sr


# ------------------------------------------------------------ framework
def f1_helpers(ctx):
    from scipy.linalg import expm
    from control.reduced import ReducedModel
    msgs = []
    x = CB.exp_update(np.array([1.0, -3.0]), 0.5, 0.02, 0.04 * 50)
    if not np.all(np.abs(x - 0.5) <= np.abs(np.array([0.5, -3.5]))):
        return False, "exp_update grows"
    red = ReducedModel(ctx.p)
    d = 0.0
    for wn, z in ((ctx.wh, ctx.zh), (ctx.wp, ctx.zp), (9.0, 0.005)):
        d = max(d, np.abs(CB.osc_phi(wn * wn, 2 * z * wn, 0.04)
                          - red._phi(wn, z, 0.04)).max())
    for k, c in ((-4.0, 0.3), (2.0, -0.5), (0.0, 0.0), (1e-14, 2e-7)):
        A = np.array([[0.0, 1.0], [-k, -c]])
        d = max(d, np.abs(CB.osc_phi(k, c, 0.04) - expm(A * 0.04)).max())
    kk = np.array([4.0, -1.0, 25.0])
    cc = np.array([0.1, 0.2, -0.3])
    Pb = CB.osc_phi(kk, cc, 0.04)
    for i in range(3):
        d = max(d, np.abs(Pb[i] - CB.osc_phi(kk[i], cc[i], 0.04)).max())
    if d > 1e-10:
        return False, f"osc_phi differs by {d:.2e}"
    msgs.append(f"osc_phi max diff {d:.1e}")
    xo, vo = 1.0, 0.0
    w = 9.0
    e0 = 0.5 * (w * w * xo ** 2 + vo ** 2)
    emax = e0
    for _ in range(20000):
        xo, vo = CB.osc_step(xo, vo, w * w, 2 * 0.005 * w, 0.04)
        emax = max(emax, 0.5 * (w * w * xo ** 2 + vo ** 2))
    if emax > e0 * (1 + 1e-9):
        return False, "oscillator gains energy"
    rng = np.random.default_rng(1)
    v = rng.normal(0, 5, 10000)
    c = np.exp(rng.uniform(-3, 8, 10000))
    a = CB.implicit_quad_drag(v, c, 1400.0, 0.04)
    if not (np.all(a * v <= 0) and np.all(np.abs(a * 0.04) < 0.5 * np.abs(v)
                                          + 1e-15)):
        return False, "implicit drag reverses or does positive work"
    V = np.abs(rng.normal(0, 4, 10000))
    mo = np.exp(rng.uniform(0, 8, 10000))
    mn = mo + np.exp(rng.uniform(0, 9, 10000))
    Vp, J = CB.implicit_impulse(V, 1400.0, mo, mn)
    if not (np.all(Vp >= 0) and np.all(Vp <= V) and np.all(J >= 0)):
        return False, "implicit impulse reverses"
    pt = (rng.uniform(-2, 3), rng.uniform(-1, 1), rng.uniform(-0.5, 0.5))
    F = rng.normal(0, 1, (7, 3))
    th = rng.normal(0, 0.05, 7)
    if ctx.sp != -1.0:
        return False, "check written for s_p = -1"
    dq = np.abs(CB.lever_Q(ctx, pt, F, th)
                - G.OperatorGen._point_Q(pt, F, th)).max()
    if dq > 1e-12:
        return False, f"lever_Q != operators_gen._point_Q ({dq:.1e})"
    g = CB.unit_vertical(ctx, 2.0)
    dnu = ctx.Minv @ g * 1.0                 # unit impulse
    if abs(g @ dnu - 1.0 / CB.point_mass(ctx, g)) > 1e-12:
        return False, "point_mass inconsistent"
    return True, "; ".join(msgs + ["drag, impulse, lever, point mass ok"])


def f2_sea(ctx):
    B = 6
    rs = a_sea(ctx, B)
    rng = np.random.default_rng(2)
    sr = running(ctx, rng.uniform(5, 25, B))
    sr[:, 0], sr[:, 1] = rng.uniform(-50, 50, (2, B))
    sr[:, 7], sr[:, 8], sr[:, 9] = rng.uniform(-3, 3, B), \
        rng.uniform(-0.3, 0.3, B), rng.uniform(-1, 1, B)
    t = rng.uniform(0, 100, B)
    s = CB.CatSea(rs, ctx).sample(sr, t)
    e, ed = rs.stations(sr[:, 0], sr[:, 1], sr[:, 7], t,
                        vel=(sr[:, 2], sr[:, 9], sr[:, 8]))
    d1 = max(np.abs(s["eta"] - e).max(), np.abs(s["etad"] - ed).max())
    # one component: u_orb along its direction = omega * eta
    one = type("S", (), {})()
    a, k, w, th0, ph0 = 0.7, 0.12, np.sqrt(9.81 * 0.12), 0.9, 0.3
    for nm, val in (("a", a), ("k", k), ("w", w), ("c", np.cos(th0)),
                    ("s", np.sin(th0)), ("phi", ph0)):
        setattr(one, nm, np.full((B, 1), val))
    one.aw = one.a * one.w
    s1 = CB.CatSea(one, ctx).sample(sr, t)
    psi = sr[:, 7][:, None, None]
    ub = w * s1["eta"] * np.cos(th0 - psi)
    vb = w * s1["eta"] * np.sin(th0 - psi)
    d2 = max(np.abs(s1["uorb"] - ub).max(), np.abs(s1["vorb"] - vb).max())
    return max(d1, d2) < 1e-9, f"RowSeas diff {d1:.1e}, orbital {d2:.1e}"


def f3_quantities(ctx):
    sr = running(ctx, np.linspace(3.0, 30.0, 12))
    sea = CB.CatSea(None, ctx).sample(sr, 0.0)
    q = CB.substep_quantities(ctx, sr, 0.0, 0.0, sea)
    bad = []
    if not np.allclose(q["hc"], ctx.hc_run[None], atol=1e-12):
        bad.append("hc")
    if not (ctx.hc_run > 0).all():
        bad.append(f"hc_run {ctx.hc_run}")
    if np.abs(q["dd"]).max() > 1e-12:
        bad.append("dd")
    if np.abs(q["tau_e"] - ctx.tau_run).max() > 1e-12:
        bad.append("tau_e")
    if np.abs(q["h_in"] - ctx.h_run).max() > 1e-12 or ctx.h_run <= 0:
        bad.append("h_in")
    if np.abs(q["a_lofi_vert"]).max() > 1e-12:
        bad.append("a_lofi_vert")
    msg = (f"hc_run {np.round(ctx.hc_run, 3).tolist()} m, h_run "
           f"{ctx.h_run:.3f} m, tau_run {np.degrees(ctx.tau_run):.2f} deg")
    return not bad, (("bad: " + ", ".join(bad) + "; ") if bad else "") + msg


def f4_tables(ctx):
    if set(CB.P_ON) != set(CB.CODES):
        return False, "P_ON does not cover CODES"
    P = [c for c in CB.PROP_ORDER if c != "M10"]
    if sorted(P) != sorted(f"P{i}" for i in range(1, 15)):
        return False, "PROP_ORDER must hold every P item once"
    ix = CB.PROP_ORDER.index
    chain = ("P12", "M10", "P7", "P3", "P10", "P1", "P2", "P9", "P4", "P5",
             "P8", "P14")
    if not all(ix(a) < ix(b) for a, b in zip(chain, chain[1:])) or \
            ix("P7") > ix("P3") or ix("P11") > ix("P3"):
        return False, "propulsion order differs from draft 3.5"
    if abs(sum(p for _, p in CB.L3_TIERS) - 1.0) > 1e-12:
        return False, "tier shares do not sum to 1"
    rng = np.random.default_rng(5)
    n = 20000
    tiers = [CB.draw_tier(rng) for _ in range(n)]
    worst = max(abs(tiers.count(t) / n - p) / np.sqrt(p * (1 - p) / n)
                for t, p in CB.L3_TIERS)
    if worst > 4:
        return False, f"tier frequencies off ({worst:.1f} sigma)"
    # sparse episodes, W1 the only item offered
    ns, n_on, n_sp_on, N = 0, 0, 0, 600
    for s in range(N):
        d = CB.CatDraw(s, tier="none", items=("W1",), layer1=False)
        ns += d.sparse
        n_on += "W1" in d.on
        n_sp_on += d.sparse and "W1" in d.on
    f_sp = ns / N
    p_w1 = CB.SPARSE_P + (1 - CB.SPARSE_P) * CB.P_ON["W1"]
    z1 = abs(f_sp - CB.SPARSE_P) / np.sqrt(CB.SPARSE_P * 0.85 / N)
    z2 = abs(n_on / N - p_w1) / np.sqrt(p_w1 * (1 - p_w1) / N)
    ok = z1 < 4 and z2 < 4 and n_sp_on == ns
    return ok, (f"tiers {worst:.1f} sigma; sparse share {f_sp:.3f} "
                f"(0.15); W1 on {n_on / N:.3f} (expect {p_w1:.3f}); "
                f"W1 on in every sparse draw: {n_sp_on == ns}")


def mini_loop(ctx, fn, rowseas, sr0, thr, noz, n_sub, dt=CB.DT_SUB,
              calm=None):
    """The closed-loop pattern of simulate_gen with ideal, constant
    actuators: sea at the substep start, the low-fidelity step, nu0, the
    injected acceleration clipped (clip_push), kick + impulse. calm (B,)
    bool: rows whose sea is masked to zero (for both the low-fidelity boat
    and the injection)."""
    from control.reduced import ReducedModel
    red = ReducedModel(ctx.p)
    cs = CB.CatSea(rowseas, ctx)
    sr = sr0.copy()
    rv = list(R.RED_VEL)
    pmax, S = 0.0, [sr.copy()]
    for j in range(n_sub):
        t = j * dt
        sea = cs.sample(sr, t)
        if calm is not None:
            sea = {k: np.where(calm.reshape((-1,) + (1,) * (v.ndim - 1)),
                               0.0, v) for k, v in sea.items()}
        ns = red.step(sr, thr, noz, sea["eta"], ctx.x_st, dt)[0]
        nu0 = (ns[:, rv] - sr[:, rv]) / dt
        acc, imp = fn(sr, thr, noz, sea, nu0, t)
        rc, _ = CB.clip_push(acc, np.zeros_like(acc))
        ns[:, rv] += rc * dt + imp
        pmax = max(pmax, float(np.abs(rc).max()))
        sr = ns
        S.append(sr.copy())
    return np.stack(S, 1), pmax


def f5_assembly(ctx):
    path = os.path.join(HERE, "_cache", "meta5", "lib.npz")
    if not os.path.exists(path):
        return True, "skipped (no meta5 lib.npz)"
    with np.load(path) as z:
        lib = dict(mu=z["mu"], sd=z["sd"], S=z["S"])
    d = None
    for s in range(40):
        c = CB.CatDraw(s, lib=lib, tier="none", items=("W1",),
                       force_items={"W1": True}, sparse=False)
        o = c.op_rb
        if o is not None and not o.rb_off and o.t4 and o.t3:
            d = c
            break
    if d is None:
        return False, "no draw with layer 1 T3 + T4 in 40 seeds"
    o = d.op_rb
    bad = []
    if "T8" in o.enabled:
        bad.append("T8 on")
    if np.abs(o.cKz[1:]).max() + np.abs(o.cKq[1:]).max() > 0 or \
            abs(o.cdz[1]) + abs(o.cdq[1]) > 0:
        bad.append("T4 speed polynomial")
    if o.nl is not None:
        bad.append("T4 tanh stiffness")
    if np.abs(o.cks).max() + np.abs(o.cka).max() > 0:
        bad.append("T4 heave-pitch coupling")
    B = 4
    rs = a_sea(ctx, B, seed=21)
    st = d.new_state(B, rng=np.random.default_rng(3))
    sr = running(d.ctx, [9.0, 13.0, 17.0, 21.0])
    sr[:, 9], sr[:, 8] = 0.3, 0.1
    sea = CB.CatSea(rs, d.ctx).sample(sr, 0.0)
    thr = d.ctx.k_drag * sr[:, 2] ** 2
    noz = np.zeros(B)
    # T2c's rigid m v r removed: layer 1 raw minus post
    raw = o.rb_accel(d.new_state(B)["l1"], sr, thr, noz, sea["eta"],
                     sea["etad"], CB.DT_SUB, parts=True)
    want = raw["T2c"][:, 0] - o.m * 0.3 * 0.1 / o.m_u
    out = d.substep(st, sr, thr, noz, sea, parts=True)
    got = out["PARTS"]["L1:T2c"][:, 0]
    if np.abs(got - want).max() > 1e-12:
        bad.append("T2c rigid term")
    tot = sum(v for k, v in out["PARTS"].items() if not k.endswith(":imp"))
    if np.abs(tot - out["acc"]).max() > 1e-9:
        bad.append("parts do not sum to acc")
    env = R.light_env(d.ctx.p)
    U = np.stack([thr / d.ctx.t_max, noz], 1)
    s28 = R.s28_of(env, R.to14(sr, thr, noz), sea["eta"].reshape(B, 15), U)
    rule, nz = d.step(st, s28, noise=True, rng=np.random.default_rng(4))
    if np.abs(rule).max() != 0.0:
        bad.append("held rule not 0 with T8 off")
    st = d.new_state(B, rng=np.random.default_rng(3))

    def fn(s, th_, nz_, sea_, nu0, t):
        r = d.substep(st, s, th_, nz_, sea_, nu0=nu0)
        return r["acc"], r["imp"]
    S, pmax = mini_loop(d.ctx, fn, rs, sr, thr, noz, 125)
    if not np.isfinite(S).all():
        bad.append("closed loop not finite")
    return not bad, ((", ".join(bad) + "; ") if bad else "") + (
        f"seed {d.seed}: switches {sorted(d.switches)}; 5 s closed loop "
        f"max |push| {pmax:.2f}")


# ------------------------------------------------------------ per item
def i1_draw(item, ctx):
    a = item.draw(np.random.default_rng(7), ctx)
    b = item.draw(np.random.default_rng(7), ctx)
    same = (a is None and b is None) or (a is not None and b is not None and
                                         all(np.allclose(a[k], b[k])
                                             for k in a))
    rng = np.random.default_rng(8)
    n = 2000
    on = sum(item.draw(rng, ctx) is not None for _ in range(n))
    p = item.p_on(ctx)
    sd = np.sqrt(max(p * (1 - p), 1e-12) / n)
    ok = same and (abs(on / n - p) < 4 * sd if 0 < p < 1 else on == p * n)
    return ok, f"deterministic {same}; on {on / n:.3f} (P_ON {p:.3f})"


def rand_states(ctx, rng, B):
    sr = np.zeros((B, 10))
    sr[:, 0], sr[:, 1] = rng.uniform(-100, 100, (2, B))
    sr[:, 2] = rng.uniform(0.5, 30.0, B)
    sr[:, 3] = ctx.z0 + rng.uniform(-0.6, 0.8, B)
    sr[:, 4] = rng.uniform(-4, 4, B)
    sr[:, 5] = ctx.th0 + rng.uniform(-0.15, 0.15, B)
    sr[:, 6] = rng.uniform(-1.5, 1.5, B)
    sr[:, 7] = rng.uniform(-np.pi, np.pi, B)
    sr[:, 8] = rng.uniform(-0.6, 0.6, B)
    sr[:, 9] = rng.uniform(-2, 2, B)
    sr[:8, 3] = ctx.z0 + 1.5                   # airborne
    sr[8:16, 3] = ctx.z0 - 0.8                 # deep
    return sr


def i2_states(item, ctx):
    rng = np.random.default_rng(9)
    B = 256
    rs = a_sea(ctx, B, seed=31, hs=2.0)
    worst, sat = 0.0, 0.0
    for _ in range(10):
        prm = item.draw_on(rng, ctx)
        st = item.init_state(prm, B, rng=np.random.default_rng(1))
        sr = rand_states(ctx, rng, B)
        sea = CB.CatSea(rs, ctx).sample(sr, rng.uniform(0, 100, B))
        thr = rng.uniform(0, ctx.t_max, B)
        noz = rng.uniform(-ctx.rud_max, ctx.rud_max, B)
        q = CB.substep_quantities(ctx, sr, thr, noz, sea,
                                  cmd=np.stack([thr, noz], 1))
        a, imp, obs, raw = CB.item_accel(item, prm, st, q, CB.DT_SUB)
        if a.shape != (B, 5) or imp.shape != (B, 5):
            return False, f"shapes {a.shape} {imp.shape}"
        if not (np.isfinite(a).all() and np.isfinite(imp).all()
                and all(np.isfinite(v).all() for v in obs.values())):
            return False, "non-finite output"
        nom = item.nominal(prm, ctx)
        if item.saturate:
            worst = max(worst, float((np.abs(a) / nom).max()))
        sat = max(sat, float((np.abs(raw) / nom).max()))
    ok = worst <= CB.SAT * (1 + 1e-12)
    return ok, (f"max |a| / nominal {worst:.2f} after saturation (<= "
                f"{CB.SAT}), {sat:.2f} before, over random / airborne / "
                "deep states")


def i3_calm(item, ctx):
    if not item.calm_zero:
        return True, "not a calm-zero item"
    rng = np.random.default_rng(10)
    worst = 0.0
    speeds = np.linspace(0.5, 2.0, 7) * ctx.u_id
    for _ in range(50):
        prm = item.draw_on(rng, ctx)
        for sp_, pr in (([ctx.u_id], prm), (speeds, item.calm_params(prm))):
            sr = running(ctx, sp_)
            B = len(sr)
            st = item.init_state(pr, B, rng=np.random.default_rng(2))
            sea = CB.CatSea(None, ctx).sample(sr, 0.0)
            thr = ctx.k_drag * sr[:, 2] ** 2
            for _k in range(25):
                q = CB.substep_quantities(ctx, sr, thr, 0.0, sea)
                a, imp, _, _ = CB.item_accel(item, pr, st, q, CB.DT_SUB)
                worst = max(worst, float(np.abs(a).max()),
                            float(np.abs(imp).max()))
    return worst < 1e-9, f"max |a| in calm steady running {worst:.1e}"


def i4_loop(item, ctx):
    rng = np.random.default_rng(12)
    prm = item.draw_on(rng, ctx)
    B = 4
    rs = a_sea(ctx, B, seed=41)
    sr = running(ctx, [ctx.u_id, 9.0, 15.0, 22.0])
    thr = ctx.k_drag * sr[:, 2] ** 2
    noz = np.zeros(B)
    st = item.init_state(prm, B, rng=np.random.default_rng(3))
    calm = np.array([True, False, False, False])

    def fn(s, th_, nz_, sea_, nu0, t):
        q = CB.substep_quantities(ctx, s, th_, nz_, sea_, nu0=nu0)
        a, imp, _, _ = CB.item_accel(item, prm, st, q, CB.DT_SUB)
        return a, imp
    # row 0 calm (the sea masked to 0), rows 1-3 in the sea, 10 s
    S, pmax = mini_loop(ctx, fn, rs, sr, thr, noz, 250, calm=calm)
    fin = np.isfinite(S).all()
    drift = float(np.abs(S[0, -1] - S[0, 0])[[3, 4, 5, 6, 8, 9]].max())
    ok = fin and (drift < 1e-9 or not item.calm_zero)
    return ok, (f"finite {fin}; calm row drift {drift:.1e}; max |push| in "
                f"the sea {pmax:.2f} m/s^2 or rad/s^2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="")
    ap.add_argument("--no-assembly", action="store_true")
    a = ap.parse_args()
    try:
        import psutil
        print(f"free RAM {psutil.virtual_memory().available / 2 ** 30:.2f} "
              "GiB", flush=True)
    except ImportError:
        pass
    t0 = time.time()
    CB.load_items()
    ctx = CB.CatCtx()
    for nm, fn in (("F1 helpers", f1_helpers), ("F2 sea", f2_sea),
                   ("F3 shared quantities", f3_quantities),
                   ("F4 tables", f4_tables)):
        guard(nm, fn, ctx)
    if not a.no_assembly:
        guard("F5 assembly", f5_assembly, ctx)
    want = [c for c in a.items.split(",") if c] or list(CB.REGISTRY)
    for code in [c for c in CB.CODES if c in want]:
        item = CB.REGISTRY.get(code)
        if item is None:
            rep(f"{code}", False, "not registered")
            continue
        for nm, fn in (("I1 draw", i1_draw), ("I2 states", i2_states),
                       ("I3 calm zero", i3_calm), ("I4 closed loop", i4_loop)):
            guard(f"{code} {nm}", fn, item, ctx)
        for fn in CB.CHECKS.get(code, []):
            guard(f"{code} I5 {fn.__name__}", fn, item, ctx)
    n_bad = sum(not ok for _, ok in RES)
    print(f"{len(RES) - n_bad}/{len(RES)} passed, {time.time() - t0:.1f} s",
          flush=True)
    return 1 if n_bad else 0



# ------------------------------------------------ group obs (O1-O5)
# Appended by the observation-model group (learn/meta/cat_obs.py). The O
# items live in cat_base.OBS_REGISTRY, which main()'s item loop does not
# reach, so this block wraps main(): afterwards the O codes named by
# --items (all of them when --items is empty; none when it names only
# catalogue codes) run
#   OI1  draw deterministic from its stream, on-rate = P_ON_OBS
#   OI2  the item alone on a low-fidelity closed-loop record in a sea (plus
#        a roll signal): finite, right shapes, the record unchanged (never
#        fed back); median rms of observed - true per channel
#   I5   the item module's own checks (@cat_base.check)
# and once
#   OA   all items together (ObsModel on=True, 10 draws): finite, record
#        unchanged; the default ObsModel is OFF (D10a) and returns the truth
#        exactly; the pipeline with no item reproduces the truth; median
#        noise sd of the observed velocity differences / 0.24 s (what e0
#        would carry)
def _obs_record(ctx, T=30.0):
    B = 2
    rs = a_sea(ctx, B, seed=41)
    sr = running(ctx, [ctx.u_id, 15.0])
    thr = ctx.k_drag * sr[:, 2] ** 2
    noz = np.zeros(B)

    def zero(s, th_, nz_, sea_, nu0, t):
        return np.zeros((len(s), 5)), np.zeros((len(s), 5))
    S, _ = mini_loop(ctx, zero, rs, sr, thr, noz, int(round(T / CB.DT_SUB)))
    x = S[1].copy()
    N = len(x)
    t = np.arange(N) * CB.DT_SUB
    u, v = x[:, 2], x[:, 9]
    extra = dict(roll=0.05 * np.sin(1.3 * t),
                 roll_rate=0.065 * np.cos(1.3 * t),
                 rpm=np.full(N, np.sqrt(thr[1] / ctx.t_max)),
                 wind_speed_rel=np.hypot(u, v), wind_angle_rel=np.arctan2(v, u),
                 stw_u=u.copy(), stw_v=v.copy(), gnss_q=np.ones(N))
    return dict(t=t, sr=x, cmd=np.tile([thr[1], 0.0], (N, 1)), extra=extra)


def _obs_same(a, b):
    return all(np.array_equal(a[k], b[k]) if not isinstance(a[k], dict)
               else _obs_same(a[k], b[k]) for k in a)


_OBS_CH = (("u", 2), ("v", 9), ("r", 8), ("zd", 4), ("th", 5))


def _obs_rms(o, tr):
    d = o["sr"] - tr["sr"][o["k"]]
    return np.sqrt((d ** 2).mean(0))


def oi1_draw(item, ctx):
    a = item.draw(np.random.default_rng(7), ctx)
    b = item.draw(np.random.default_rng(7), ctx)
    same = a is not None and b is not None and all(
        np.allclose(a[k], b[k]) for k in a)
    rng = np.random.default_rng(8)
    n = 500
    on = sum(item.draw(rng, ctx) is not None for _ in range(n))
    p = item.p_on(ctx)
    return same and on == round(p * n), \
        f"deterministic {same}; on {on / n:.3f} (P_ON_OBS {p:.2f})"


def oi2_apply(item, ctx):
    import copy
    tr = _obs_record(ctx)
    keep = copy.deepcopy(tr)
    rng = np.random.default_rng(9)
    R = []
    for s in range(10):
        prm = item.draw_on(rng, ctx)
        o = item.apply(prm, tr, np.random.default_rng(s), ctx=ctx)
        M = len(o["k"])
        if o["sr"].shape != (M, 10) or o["cmd"].shape != (M, 2) or \
                any(v.shape != (M,) for v in o["extra"].values()):
            return False, "shapes"
        if not (np.isfinite(o["sr"]).all() and np.isfinite(o["cmd"]).all()
                and all(np.isfinite(v).all() for v in o["extra"].values())):
            return False, "non-finite output"
        R.append(_obs_rms(o, tr))
    if not _obs_same(tr, keep):
        return False, "the recorded trajectory was modified"
    med = np.median(R, 0)
    return True, "median rms obs - truth: " + ", ".join(
        f"{n} {med[i]:.3g}" for n, i in _OBS_CH) + "; record unchanged"


def oa_all(ctx):
    import copy
    from learn.meta import cat_obs as CO
    tr = _obs_record(ctx)
    keep = copy.deepcopy(tr)
    off = CO.ObsModel(3, ctx).observe(tr)
    if CB.OBS_ON_D10A or off["obs_on"] or not np.array_equal(
            off["sr"], tr["sr"][off["k"]]):
        return False, "the default model is not OFF / not the truth"
    none = CO.run_obs(ctx, tr, {})
    d0 = np.abs(none["sr"] - tr["sr"][none["k"]]).max()
    R, E = [], []
    for s in range(10):
        m = CO.ObsModel(s, ctx, on=True)
        o = m.observe(tr, noise_seed=1)
        if not (np.isfinite(o["sr"]).all() and all(
                np.isfinite(v).all() for v in o["extra"].values())):
            return False, f"seed {s}: non-finite output"
        R.append(_obs_rms(o, tr))
        e = np.diff(o["sr"] - tr["sr"][o["k"]], axis=0) / 0.24
        E.append(e.std(0))
    ok = d0 < 1e-9 and _obs_same(tr, keep)
    mr, me = np.median(R, 0), np.median(E, 0)
    return ok, (f"off = truth; no-item pipeline error {d0:.1e}; median rms "
                + ", ".join(f"{n} {mr[i]:.3g}" for n, i in _OBS_CH)
                + "; median e0 noise sd " + ", ".join(
                    f"{n} {me[i]:.3g}" for n, i in _OBS_CH if n != "th")
                + "; record unchanged " + str(_obs_same(tr, keep)))


_main_before_obs = main


def main():
    rc = _main_before_obs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="")
    a, _ = ap.parse_known_args()
    want = [c for c in a.items.split(",") if c]
    codes = [c for c in CB.OBS_CODES if not want or c in want]
    if not codes:
        return rc
    t0 = time.time()
    from learn.meta import cat_obs  # noqa: F401  (registers O1-O5)
    ctx = CB.CatCtx()
    for code in codes:
        item = CB.OBS_REGISTRY.get(code)
        if item is None:
            rep(code, False, "not registered")
            continue
        guard(f"{code} OI1 draw", oi1_draw, item, ctx)
        guard(f"{code} OI2 apply", oi2_apply, item, ctx)
        for fn in CB.CHECKS.get(code, []):
            guard(f"{code} I5 {fn.__name__}", fn, item, ctx)
    guard("OA all observation items", oa_all, ctx)
    n_bad = sum(not ok for _, ok in RES)
    print(f"{len(RES) - n_bad}/{len(RES)} passed incl. the obs group "
          f"({time.time() - t0:.1f} s for it)", flush=True)
    return 1 if n_bad else rc


if __name__ == "__main__":
    sys.exit(main())
