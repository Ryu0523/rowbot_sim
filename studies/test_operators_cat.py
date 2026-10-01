#!/usr/bin/env python3
"""
Tests of the D10a catalogue family (learn/meta/operators_cat.py; draft
learn/meta/PRIOR_D10_DRAFT.md; PRIOR_DERIVATION.md D10). One process.

  1  reproducible from the seed: the draw, the acceptance test and its
     redraws, and a closed-loop rollout are identical for the same seed;
     another seed differs
  2  branch replay (float64): continuing simulate_cat from a snapshot at
     step k equals the run's own continuation (with noise) to <= 1e-9
  3  the Mission path (CatPlant + CatInjector, one row, a draw with slams
     W3 forced on in a sea) equals simulate_cat to <= 1e-9: states, safety
     quantities, the impulse channel, the extra observed signals
  4  the full construction harness studies/test_cat_items.py passes (run
     in this process)
  5  calm, steady, straight running at the identification speed: every
     calm-zero catalogue item (and the gate W4) gives zero (<= 1e-12: the
     propulsion items' own check allows round-off 2e-15), in a closed loop
     of 2 s, and the state does not drift
  6  clip and impulse-channel rules (a stub operator): a huge rule is
     clipped at CLIP x A_REF per substep and counted; impulses are added
     after the kick unclipped per substep, their control-step sum is
     bounded by CLIP x A_REF x 0.24 s (counted when it acts), the control-
     step-average check counts |P + IMP| > CLIP x A_REF, and the vertical
     acceleration of the safety quantities includes the impulse
  7  divergence-only acceptance: layer 1's stability rule is off
     (OperatorRBL1.stable), a draw made to diverge (strong pitch anti-
     damping on its first attempt) is rejected with reason 'diverged', its
     items logged, and replaced by the next attempt; a large constant but
     bounded push (heave 0.2 g, sway 0.3 x its clip) is accepted (no shape
     rule)

    python studies/test_operators_cat.py [--skip-harness]
"""
import argparse
import copy
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from learn.meta import cat_base as CB  # noqa: E402
from learn.meta import operators_cat as OC  # noqa: E402
from learn.meta import operators_gen as G  # noqa: E402
from learn.meta import operators_rb as R  # noqa: E402
from learn.meta.operators import A_REF, CLIP, clip_push  # noqa: E402

L_ = np.load(os.path.join(ROOT, "studies", "_cache", "meta2", "lib.npz"))
LIB = {k: L_[k] for k in ("mu", "sd", "S")}
ENV = R.light_env()
LIM = CLIP * A_REF
FAIL = []


def check(name, ok, msg=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {msg}", flush=True)
    if not ok:
        FAIL.append(name)


def wait_ram(min_gb=3.0, poll=60, max_wait=1800):
    try:
        import psutil
    except ImportError:
        return True
    t0 = time.time()
    while psutil.virtual_memory().available / 1e9 < min_gb:
        if time.time() - t0 > max_wait:
            return False
        time.sleep(poll)
    return True


def seas_for(B, seed0=0, hs=None):
    ds = []
    for i in range(B):
        d = R.sea_dict(np.random.default_rng([seed0, i]))
        if hs is not None:
            d["hs"] = hs
        ds.append(R.sea_state(d, 50000 + i))
    return R.RowSeas(ds, ENV["x_st"], ENV["y_off"])


def rand_U(rng, B, n):
    return np.stack([np.clip(0.5 + 0.2 * rng.standard_normal((B, n)), 0, 1),
                     np.clip(0.3 * rng.standard_normal((B, n)), -1, 1)], -1)


def rich_op(seed, **kw):
    """A dense draw (not sparse, small layer-3 tier, several items), not
    acceptance-tested."""
    kw = dict(dict(sparse=False, tier="small"), **kw)
    return OC.OperatorCat(seed, LIB, accept=False, **kw)


# ------------------------------------------------------------------ 1
def test_reproducible():
    print("1. reproducible from the seed", flush=True)
    outs = []
    for seed in (21, 21, 22):
        o = OC.OperatorCat(seed, LIB)             # with the acceptance test
        B, n = 2, 30
        rng = np.random.default_rng(3)
        xs = R.start_states(ENV, B, rng)
        g = [dict(op=o, st=o.new_state(B, rng=np.random.default_rng(4)),
                  rows=np.arange(B), rng=np.random.default_rng(5))]
        out = OC.simulate_cat(ENV, xs, rand_U(rng, B, n), np.zeros(B),
                              seas_for(B, 6), g, n)
        outs.append((o.style, out["XS"], o.n_reject))
    same = outs[0][0] == outs[1][0] and np.array_equal(outs[0][1],
                                                       outs[1][1])
    check("same seed -> identical draw, acceptance and rollout", same,
          f"(rejections {outs[0][2]}, {outs[1][2]}; items "
          f"{len(outs[0][0]['items'])})")
    check("another seed -> different draw and rollout",
          outs[0][0]["items"] != outs[2][0]["items"]
          or not np.allclose(outs[0][1], outs[2][1]))


# ------------------------------------------------------------------ 2
def test_branch():
    print("2. branch replay from a snapshot (float64)", flush=True)
    o = rich_op(4242, force_items={"W3": True, "H1": True, "P1": True})
    B, n, k = 2, 50, 20
    rng = np.random.default_rng(8)
    xs = R.start_states(ENV, B, rng)
    U = rand_U(rng, B, n)
    seas = seas_for(B, 8)

    def fresh():
        return [dict(op=o, st=o.new_state(B, rng=np.random.default_rng(9)),
                     rows=np.arange(B), rng=np.random.default_rng(10))]
    g = fresh()
    a = OC.simulate_cat(ENV, xs, U[:, :k], np.zeros(B), seas, g, k)
    snap = dict(g[0], st=o.snapshot(g[0]["st"]),
                rng=copy.deepcopy(g[0]["rng"]))
    a2 = OC.simulate_cat(ENV, a["XS"][:, -1], U[:, k:], np.full(B, k * 0.24),
                         seas, [snap], n - k)
    full = OC.simulate_cat(ENV, xs, U, np.zeros(B), seas, fresh(), n)
    err = float(np.abs(full["XS"][:, k:] - a2["XS"]).max())
    err_s = float(np.nanmax(np.abs(full["APK"][:, k:] - a2["APK"])))
    err_o = float(np.abs(full["OBS"][:, k:] - a2["OBS"]).max())
    check(f"continuation from step {k} = own continuation (with noise)",
          err <= 1e-9 and err_s <= 1e-9 and err_o <= 1e-9,
          f"(max |diff| states {err:.1e}, APK {err_s:.1e}, OBS {err_o:.1e};"
          f" items {sorted(o.on)})")


# ------------------------------------------------------------------ 3
def test_mission_path():
    print("3. CatPlant + CatInjector (the Mission path) vs simulate_cat",
          flush=True)
    from learn.meta import relabel
    env = relabel._env()
    kw = dict(force_items={"W3": True, "W1": True, "H1": True, "P1": True})
    o = rich_op(5151, **kw)
    d = R.sea_dict(np.random.default_rng(12))
    d["hs"] = max(d["hs"], 1.2)
    sea = R.sea_state(d, 50012)
    plant = OC.cat_plant(env["mission"].plant).with_sea(
        sea, params=dict(tau_thrust=0.0, rud_rate=0.0, act_family=0.0))
    rng = np.random.default_rng(13)
    n = 60
    U = rand_U(rng, 1, n)[0]
    U[:, 0] = np.clip(U[:, 0] + 0.2, 0, 1)
    xs0 = R.start_states(env, 1, rng)[0]
    seas = R.RowSeas([sea], env["x_st"], env["y_off"])
    st = o.new_state(1, rng=np.random.default_rng(14))
    inj = OC.CatInjector(o, st)
    nrng = np.random.default_rng(15)
    s, t = xs0.copy(), 0.0
    XS, OBS, IMP = [s.copy()], [], []
    dt, sub = env["dt"], env["sub"]
    for k in range(n):
        sr = R.to_reduced(s[None])
        w15 = R.mid_waves(seas, sr, np.array([t]), 0.5 * dt * sub)
        s28 = R.s28_of(env, s[None], w15, U[k][None])
        rr, rn = o.step(st, s28, noise=True, rng=nrng)
        inj.rule, inj.noise = clip_push(rr[0], rn[0])
        im = np.zeros(5)
        for kk in range(sub):
            s = plant.step(s, t, U[k, 0] * env["t_max"],
                           U[k, 1] * env["rud_max"], dt)
            s = inj(s, dt, plant)
            if kk == 0:
                OBS.append([inj.obs[nm] for nm in OC.OBS_EXTRA])
            im += inj.last_imp / sub
            t += dt
        XS.append(s.copy())
        IMP.append(im)
    XS = np.array(XS)
    o2 = rich_op(5151, **kw)
    g = [dict(op=o2, st=o2.new_state(1, rng=np.random.default_rng(14)),
              rows=np.array([0]), rng=np.random.default_rng(15))]
    out = OC.simulate_cat(env, xs0[None], U[None], np.zeros(1), seas, g, n)
    cols = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]
    err = float(np.abs(out["XS"][0][:, cols] - XS[:, cols]).max())
    n_imp = int((np.abs(out["IMP"][0]).sum(-1) > 0).sum())
    check("Mission path = batched simulator", err <= 1e-9,
          f"(max |diff| {err:.1e}; steps with impulses {n_imp} of {n}, "
          f"slams {int(out['SLAM'].sum())})")
    apk, amin, hmin = G.safety_per_step(np.array(inj.az)[None],
                                        np.array(inj.h + [np.nan])[None])
    e = max(float(np.abs(apk - out["APK"]).max()),
            float(np.abs(amin - out["AMIN"]).max()),
            float(np.abs(hmin[:, :-1] - out["HMIN"][:, :-1]).max()))
    check("Mission path safety quantities (impulse included) = batched",
          e <= 1e-9, f"(max |diff| {e:.1e})")
    e2 = max(float(np.abs(np.array(OBS) - out["OBS"][0]).max()),
             float(np.abs(np.array(IMP) - out["IMP"][0]).max()))
    check("Mission path observed signals and impulse channel = batched",
          e2 <= 1e-9, f"(max |diff| {e2:.1e}; roll range "
          f"{np.degrees(out['OBS'][0, :, 0]).min():.2f}..."
          f"{np.degrees(out['OBS'][0, :, 0]).max():.2f} deg)")


# ------------------------------------------------------------------ 4
def test_harness():
    print("4. the construction harness studies/test_cat_items.py",
          flush=True)
    from studies import test_cat_items as H
    argv = sys.argv
    sys.argv = [os.path.join(ROOT, "studies", "test_cat_items.py")]
    t0 = time.time()
    try:
        rc = H.main()
    finally:
        sys.argv = argv
    n_bad = sum(not ok for _, ok in H.RES)
    check("test_cat_items.py passes", rc == 0 and n_bad == 0,
          f"({len(H.RES) - n_bad}/{len(H.RES)} checks, "
          f"{time.time() - t0:.0f} s)")


# ------------------------------------------------------------------ 5
def calm_world(B, u):
    p = ENV["red"].p
    xs = np.zeros((B, 14))
    xs[:, 6] = u
    xs[:, 2], xs[:, 4] = p.get("z0", 0.0), p.get("th0", 0.0)
    xs[:, 12] = p["k_drag"] * u ** 2
    seas = seas_for(B, 3)
    seas.a[:] = 0.0
    seas.aw[:] = 0.0
    return xs, seas


def test_calm_zero():
    print("5. calm steady straight running: calm-zero items zero",
          flush=True)
    CB.load_items()
    codes = tuple(c for c in CB.CODES if c in CB.REGISTRY
                  and (CB.REGISTRY[c].calm_zero or c == "W4"))
    p = ENV["red"].p
    u = float(p["u_design"])
    worst, drift, n_items, seeds = 0.0, 0.0, 0, 12
    for s in range(seeds):
        o = OC.OperatorCat(700 + s, None, accept=False, items=codes,
                           sparse=False, tier="none",
                           force_items={c: True for c in codes})
        B = 1
        xs, seas = calm_world(B, u)
        n = 9
        U = np.zeros((B, n, 2))
        U[:, :, 0] = xs[0, 12] / ENV["t_max"]
        g = [dict(op=o, st=o.new_state(B), rows=np.arange(B), rng=None)]
        out = OC.simulate_cat(ENV, xs, U, np.zeros(B), seas, g, n,
                              noise=False, parts=True)
        tot = sum(v for k, v in out["PARTS"][0].items() if k != "cor")
        worst = max(worst, float(np.abs(tot).max()),
                    float(np.abs(out["IMP"]).max()))
        drift = max(drift, float(np.abs(out["XS"][:, -1, [2, 4, 6, 7, 8,
                                                          10, 11]]
                                        - xs[:, [2, 4, 6, 7, 8, 10, 11]])
                                 .max()))
        n_items = max(n_items, len(o.on))
    check(f"{len(codes)} calm-zero items + W4 give zero in a 2 s "
          f"closed loop at u_id ({seeds} draws)", worst <= 1e-12,
          f"(max |catalogue part| {worst:.1e}; items {codes})")
    check("the calm steady row does not drift", drift <= 1e-12,
          f"(max |state change| {drift:.1e})")


# ------------------------------------------------------------------ 6
class _Stub:
    """A minimal operator for simulate_cat: a constant rule acceleration
    and a constant impulse per substep."""

    def __init__(self, acc, imp):
        self.acc, self.imp = np.asarray(acc, float), np.asarray(imp, float)
        self.ctx = CB.CatCtx()

    def step(self, st, s_raw, noise=True, rng=None, freeze=False):
        z = np.zeros((len(np.atleast_2d(s_raw)), 5))
        return z, z.copy()

    def bind_sea(self, st, rs):
        pass

    def cat_accel(self, st, sr, thr, noz, nu0, sea, dt, cmd=None, t=None,
                  freeze=False, parts=False):
        B = len(np.atleast_2d(sr))
        return dict(acc=np.tile(self.acc, (B, 1)),
                    imp=np.tile(self.imp, (B, 1)),
                    obs={nm: np.zeros(B) for nm in OC.OBS_EXTRA},
                    slam=np.zeros(B, bool))


def test_clip_impulse():
    print("6. clip rule and the impulse channel", flush=True)
    B, n = 1, 3
    dt, sub = ENV["dt"], ENV["sub"]
    xs, seas = calm_world(B, 12.0)
    U = np.zeros((B, n, 2))
    U[:, :, 0] = xs[0, 12] / ENV["t_max"]
    # a huge rule: clipped per substep, counted
    big = OC.simulate_cat(ENV, xs, U, np.zeros(B), seas, [dict(
        op=_Stub(5 * LIM, np.zeros(5)), st={}, rows=np.arange(B))], n,
        noise=False)
    check("a huge rule is clipped at CLIP x A_REF per substep and counted",
          np.allclose(big["P"][0], LIM) and (big["clip_rule"] == sub).all()
          and (big["clip"] == sub).all(), f"(P {big['P'][0, 0]})")
    # a small impulse: added exactly, not clipped
    imp = np.array([0.0, 0.0, 0.0, 0.05, 0.0])
    ref = OC.simulate_cat(ENV, xs, U, np.zeros(B), seas, [dict(
        op=_Stub(np.zeros(5), np.zeros(5)), st={}, rows=np.arange(B))], n,
        noise=False, substeps=True)
    one = OC.simulate_cat(ENV, xs, U, np.zeros(B), seas, [dict(
        op=_Stub(np.zeros(5), imp), st={}, rows=np.arange(B))], 1,
        noise=False, substeps=True)
    dzd = one["SUB"][0, 0, 0, 4] - ref["SUB"][0, 0, 0, 4]
    check("an impulse is added after the kick, exactly (one substep)",
          abs(dzd - 0.05) <= 1e-12 and np.allclose(one["IMP"][0, 0],
                                                    imp * sub / (dt * sub))
          and one["clip_imp"].sum() == 0,
          f"(heave-rate change {dzd:.6f}, IMP {one['IMP'][0, 0, 3]:.3f})")
    az = one["AZ"][0, 0] - ref["AZ"][0, 0]
    check("the vertical acceleration of the safety quantities includes "
          "the impulse", abs(az - 0.05 / dt) <= 1e-9, f"(+{az:.3f} m/s^2)")
    # a huge impulse: the control-step sum is bounded, counted; the
    # control-step-average check counts |P + IMP| > LIM
    hug = OC.simulate_cat(ENV, xs, U, np.zeros(B), seas, [dict(
        op=_Stub(np.zeros(5), np.array([0, 0, 0, 5.0, 0])), st={},
        rows=np.arange(B))], n, noise=False)
    check("a huge impulse: control-step mean bounded at CLIP x A_REF, "
          "counted, flagged by the control-step-average check",
          np.allclose(hug["IMP"][0, :, 3], LIM[3])
          and (hug["clip_imp"][0] > 0).all()
          and (hug["clip_rule"] == 0).all(),
          f"(IMP heave {hug['IMP'][0, 0, 3]:.1f} = LIM {LIM[3]:.1f})")
    both = OC.simulate_cat(ENV, xs, U, np.zeros(B), seas, [dict(
        op=_Stub(np.array([0, 0, 0, 30.0, 0]), np.array([0, 0, 0, 0.5, 0])),
        st={}, rows=np.arange(B))], n, noise=False)
    check("rule + impulse above the limit on the control-step average is "
          "counted (never rejected)",
          both["clip_avg"][0, :, 3].all() and (both["div"] < 0).all(),
          f"(P {both['P'][0, 0, 3]:.1f} + IMP {both['IMP'][0, 0, 3]:.1f} "
          f"> {LIM[3]:.0f})")


# ------------------------------------------------------------------ 7
class _Diverge(OC.OperatorCat):
    """Attempt 0 gets a strong pitch anti-damping acceleration (+80 x
    pitch rate + a kick), later attempts are the plain draw."""

    def cat_accel(self, st, sr, thr, noz, nu0, sea, dt, cmd=None, t=None,
                  freeze=False, parts=False):
        o = super().cat_accel(st, sr, thr, noz, nu0, sea, dt, cmd=cmd, t=t,
                              freeze=freeze, parts=parts)
        if self.cat.attempt == 0:
            o["acc"] = o["acc"].copy()
            o["acc"][:, 4] += 80.0 * np.atleast_2d(sr)[:, 6] + 0.5
        return o


class _Heave(OC.OperatorCat):
    """A constant push, heave 2 m/s^2 (0.2 g: the hull stays wet) and sway
    3.6 m/s^2 (0.3 x its clip limit): large, bounded, of no hydrodynamic
    shape."""

    def cat_accel(self, st, sr, thr, noz, nu0, sea, dt, cmd=None, t=None,
                  freeze=False, parts=False):
        o = super().cat_accel(st, sr, thr, noz, nu0, sea, dt, cmd=cmd, t=t,
                              freeze=freeze, parts=parts)
        o["acc"] = o["acc"].copy()
        o["acc"][:, 3] += 2.0
        o["acc"][:, 1] += 0.3 * LIM[1]
        return o


def test_acceptance():
    print("7. divergence-only acceptance", flush=True)
    o = rich_op(6061)
    check("layer 1's stability rule is off (OperatorRBL1.stable accepts)",
          o.cat.op_rb is not None and o.cat.op_rb.stable() is True
          and isinstance(o.cat.op_rb, CB.OperatorRBL1))
    kw = dict(sparse=False, tier="none")
    d = _Diverge(6062, LIB, accept=False, **kw)
    items0 = sorted(d.on)
    OC.accept_cats([d])
    log = d.reject_log
    check("a diverging draw is rejected ('diverged'), its items logged, "
          "replaced by the next attempt",
          d.n_reject >= 1 and log and log[0]["reason"] == "diverged"
          and log[0]["items"] == items0 and d.cat.attempt == d.n_reject
          and d.accepted and d.style["reject_items"][0] == items0,
          f"(rejections {d.n_reject}, now attempt {d.cat.attempt})")
    h = _Heave(6063, LIB, accept=False, **kw)
    OC.accept_cats([h])
    check("a large constant bounded push (heave 0.2 g, sway 0.3 x its "
          "clip) is accepted (no shape rule); clip share recorded",
          h.n_reject == 0,
          f"(rejections {h.n_reject}, rule-clip share {h.test_clip:.2e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-harness", action="store_true")
    a = ap.parse_args()
    if not wait_ram():
        print("SKIPPED: free RAM below 3 GB for 30 min")
        sys.exit(2)
    t0 = time.time()
    test_clip_impulse()
    test_calm_zero()
    test_acceptance()
    test_branch()
    test_mission_path()
    test_reproducible()
    if not a.skip_harness:
        test_harness()
    print(f"{'ALL OK' if not FAIL else 'FAILED: ' + ', '.join(FAIL)} "
          f"({time.time() - t0:.0f} s)", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
