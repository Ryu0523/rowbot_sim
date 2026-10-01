#!/usr/bin/env python3
"""
Prior-predictive preview of the D10a catalogue family (learn/meta/
operators_cat.py; draft learn/meta/PRIOR_D10_DRAFT.md 7.1; PRIOR_DERIVATION
.md D10). STANDALONE: no data pipeline, nothing written except the log.
Descriptive only (no statement about information or ceilings; target
statistics are a check, never a fit).

CLOSED LOOP (the catalogue is state feedback at every substep): N fresh
episodes in the reduced world under the SAME conditions as
prior_preview_gen.py (seed0 777: same seas, start states, autopilot, M10
actuator draws; operator seeds seed0 * 1000 + i drawn from the catalogue
family and acceptance-tested in batches). e0 = data3.e0_from on the
simulated states. The statistics functions are prior_preview_gen's
(imported read-only).

Reported:
  - acceptance: rejections per draw, rejected share of attempts PER ITEM,
    draws left with forces off; layer-3 tiers, sparse draws, items per draw;
    episodes truncated by divergence
  - push size, the D5 clip check (control steps with any channel clipped),
    the rule-clip share of substeps, the impulse channel (steps where its
    control-step bound acted, steps where |mean push + impulse| exceeds the
    clip limit), the share of heave rule peaks above half the limit that
    were clipped, the injected power above 6 Hz
  - the variance share of each layer per channel: variance over the
    control steps (40..) of each part's control-step mean, summed over the
    episodes, divided by the sum over the parts (covariances ignored):
    layer 1, the catalogue groups W / H / A / B / E and the propulsion chain
    P, layer 3, the rigid-body Coriolis difference, the held old noise
  - the drawn total heave stiffness (draft 6 item 6): per draw the maximum
    over the task speeds of w_h^2 x W1 s^p_K x W5(a) largest local
    stiffness ratio x W5(b) k_m x T4 exp(cKz) (x its bounded tanh factor) x
    B1 (1 - e_m eps)^2, and the share of draws with omega dt_sub > 1
  - trigger rates: porpoising by mechanism (W2 net self-excitation at small
    amplitude: van der Pol rho_n gate > 1, flutter rho_f gate^2 > 1, at the
    substep's water-relative speed; share of time, episodes with >= 10 s of
    it), W2's push size in waves (a 10 s window with the W2 pitch
    acceleration RMS above 0.1 a_ref; NOT self-excitation: in waves the
    push is large also when W2 only lowers the damping), sustained heel (a
    10 s window with
    |mean roll| > 2 deg), chine walking (a 10 s window with roll std > 2
    deg and >= 3 sign changes), intake ventilation (P1's OWN suction below
    0.9: share of time, entries per hour; P1's latch share; P13's air
    factor below 0.9 separately), airborne time (all five centre-line
    keel immersions <= 0), slams per hour (the low-fidelity bow-station
    rule), faults occurring in the episode (P8, P10, P11 on with the fault
    time inside the episode; a P12 rule changing the command)
  - the per-episode statistics of prior_check_m15 with the target median
    (C + Cb) and its percentile; the safety quantities exact / coarse
  - SUMMARY: target median percentile (share below 5% / above 95%) for the
    old projection family (log), D8 stage 1 closed loop (v2 log), the
    general D9 family (prior_preview_gen_k1.log) and D10a (this run)

    python studies/prior_preview_cat.py [--closed 64] [--T 375]
        [--batch 32] [--log studies/_cache/meta3/prior_preview_cat.log]
"""
import argparse
import collections
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="meta3")
ap.add_argument("--closed", type=int, default=64)
ap.add_argument("--T", type=int, default=375)
ap.add_argument("--seed0", type=int, default=777)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--act", default="m10", choices=("m10", "ideal"))
ap.add_argument("--log", default=None)
args = ap.parse_args()
_argv = sys.argv
sys.argv = [os.path.join(ROOT, "studies", "prior_preview_gen.py"), "--cache",
            args.cache]
from studies import prior_preview_gen as PG  # noqa: E402
sys.argv = _argv
from learn.meta import operators_cat as OC  # noqa: E402
from learn.meta import operators_gen as G  # noqa: E402
from learn.meta import operators_rb as R  # noqa: E402
from learn.meta.operators import A_REF, CLIP  # noqa: E402
from learn.meta import cat_base as CB  # noqa: E402
from learn.meta.operators_rb import _g as _g_rb  # noqa: E402

CB.load_items()
W2C = CB.REGISTRY["W2"]

C = os.path.join(ROOT, "studies", "_cache", args.cache)
LOG = args.log or os.path.join(C, "prior_preview_cat.log")
CH = R.CHANNELS
K0 = PG.K0
LIM = CLIP * A_REF
GROUPS = ("L1", "W", "H", "A", "B", "E", "P", "L3", "cor", "noise")
_OUT = []


def out(s=""):
    print(s, flush=True)
    _OUT.append(s)


def group_of(key):
    if key.startswith("L1:"):
        return "L1"
    if key in ("L3", "cor"):
        return key
    if key == "prop":
        return "P"
    c = key.split(":")[0]
    return c[0] if c[0] in "WHABEP" else "other"


def ctx_wh(env):
    return float(env["red"].p["wn_heave"])


def windows(x, w):
    """(n - w + 1, w) sliding windows of a 1-D series (w >= 1)."""
    n = len(x)
    if n < w:
        return np.zeros((0, w))
    idx = np.arange(w)[None] + np.arange(n - w + 1)[:, None]
    return x[idx]


def parse_gen_k1():
    """prior_preview_gen_k1.log SUMMARY, last column: stat -> (pct, below,
    above)."""
    res = {}
    p = os.path.join(C, "prior_preview_gen_k1.log")
    if not os.path.exists(p):
        return res
    on = False
    pat = re.compile(r"([\d.]+)% \(([\d.]+)/([\d.]+)\)\s*$")
    for line in open(p, encoding="utf-8"):
        if line.startswith("SUMMARY"):
            on = True
            continue
        if on and line.strip() and not line.strip().startswith(
                "statistic"):
            m = pat.search(line)
            if m:
                res[line.split()[0]] = tuple(float(x) for x in m.groups())
    return res


def heave_stiffness_max(o, n_u=9):
    """Draft 6 item 6: the largest drawn total heave stiffness (rad^2/s^2)
    of one draw over the task speeds: w_h^2 x W1 s^p_K x W5(a) largest local
    stiffness ratio (largest plateau / its value at calm running, over the
    stations) x W5(b) max(k_m, 1) x T4 exp(cKz(U)) x exp(|g_z|) (T4's
    bounded tanh stiffness, when kept) x B1 (1 - e_m eps)^2."""
    ctx, on = o.ctx, o.on
    U = np.linspace(ctx.u_lo, ctx.u_hi, n_u)
    k = np.full(n_u, ctx.wh ** 2)
    if "W1" in on:
        s = np.maximum(U / ctx.u_id, ctx.u_lo / ctx.u_id)
        k = k * s ** on["W1"]["p_K"]
    if "W5" in on:
        prm = on["W5"]
        if prm["a_on"]:
            W5C = CB.REGISTRY["W5"]
            kr = W5C.kappa(prm, np.broadcast_to(ctx.hc_run, (1, 5)))[0][0]
            pl = 1.0 + np.concatenate([[0.0], np.cumsum(W5C.kinks(prm)[1])])
            k = k * float((pl.max() / kr).max())
        if prm["b_on"]:
            k = k * max(float(prm["k_m"]), 1.0)
    rb = getattr(o.cat, "op_rb", None)
    if rb is not None and "T4" in getattr(rb, "enabled", ()):
        xi = np.clip((U - rb.u_mid) / rb.u_half, -1.0, 1.0)
        k = k * np.exp(_g_rb(rb.cKz, xi))
        if getattr(rb, "nl", None) is not None:
            k = k * np.exp(abs(float(rb.nl["gz"])))
    if "B1" in on:
        k = k * (1.0 - on["B1"]["em"] * on["B1"]["eps"]) ** 2
    return float(k.max())


# --------------------------------------------------------------- closed
class Diag:
    """Per-substep diagnostics of a batch (the simulate_cat hook): for each
    group (one row) airborne substeps, suction below 0.9 (substeps and
    entries), nozzle deviating from the actuator by > 2 deg, a P12 rule
    changing the command."""

    def __init__(self, n_groups, k0, sub, ops=None):
        self.k0, self.sub = k0, sub
        z = lambda: np.zeros(n_groups)                       # noqa: E731
        self.air, self.n, self.vent, self.vent_in = z(), z(), z(), z()
        self.latch, self.vent13, self.porp = z(), z(), z()
        self.noz_dev, self.cmd_chg = z(), np.zeros(n_groups, bool)
        self.prev_vent = np.zeros(n_groups, bool)
        # W2's (params, ctx) per group: net self-excitation at the substep
        self.w2 = [None if o is None or "W2" not in o.on else
                   (o.on["W2"], o.ctx) for o in (ops or [None] * n_groups)]

    def __call__(self, k, kk, j, g, o):
        q = o["q"]
        if k < self.k0:
            return
        self.n[j] += 1
        self.air[j] += float((q["hc"][0] <= 0.0).all())
        if self.w2[j] is not None:
            prm, ctx = self.w2[j]
            gam = float(W2C.gate(prm, ctx, float(q["U_r"][0])))
            gain = (prm["rho_f"] * gam ** 2 if prm["flutter"] > 0.5
                    else prm["rho_n"] * gam)
            self.porp[j] += float(gain > 1.0)
        pd = q.get("prop_diag")
        if pd is not None:
            v = bool(pd["p1"][0] < 0.9)            # P1's own suction
            self.vent13[j] += float(pd["p13"][0] < 0.9)
            self.latch[j] += float(bool(pd["lost"][0]))
            self.vent[j] += v
            self.vent_in[j] += v and not self.prev_vent[j]
            self.prev_vent[j] = v
            self.noz_dev[j] += float(abs(pd["noz_eff"][0] - pd["noz_act"][0])
                                     > np.radians(2.0))
            self.cmd_chg[j] |= bool(pd["cmd_changed"])


def closed(lib, env):
    from learn.meta.data3 import e0_from
    from sim import lofi
    p = env["red"].p
    fb = float(env["mission"].plant.freeboard)
    x_b, sp = float(env["x_st"][-1]), float(p["sign_pitch"])
    a_ref = G.a_ref_of(p)
    rows_all, info = [], collections.defaultdict(list)
    var = collections.defaultdict(lambda: np.zeros(5))
    att, rej = collections.Counter(), collections.Counter()
    trig = collections.defaultdict(list)
    n_eps, seed0 = args.closed, args.seed0
    for g0 in range(0, n_eps, args.batch):
        B = min(args.batch, n_eps - g0)
        rng = np.random.default_rng([seed0, g0])
        t0 = time.time()
        ops = OC.build_ops([int(seed0 * 1000 + g0 + i) for i in range(B)],
                           lib, dt=env["dt"] * env["sub"], L=env["L"])
        t_acc = time.time() - t0
        for o in ops:
            info["k_heave"].append(heave_stiffness_max(o))
            info["n_reject"].append(o.n_reject)
            info["off"].append(o.forces_off)
            info["tier"].append(o.cat.tier)
            info["sparse"].append(o.cat.sparse)
            info["n_items"].append(len(o.on))
            for c in o.on:
                att[c] += 1
            for r in o.reject_log:
                for c in r["items"]:
                    att[c] += 1
                    rej[c] += 1
        groups = [dict(op=o, st=o.new_state(1, rng=np.random.default_rng(
            [o.seed, 1])), rows=np.array([i]),
            rng=np.random.default_rng([o.seed, 1])) for i, o in enumerate(ops)]
        seas = R.RowSeas([R.sea_state(R.sea_dict(np.random.default_rng(
            [seed0, g0, i])), 50000 + g0 + i) for i in range(B)],
            env["x_st"], env["y_off"])
        xs = R.start_states(env, B, rng)
        act = None
        if args.act == "m10":
            ds = [lofi.draw_act(np.random.default_rng([o.seed, 11]), p,
                                env["dt"]) for o in ops]
            act = {k: np.array([d[k] for d in ds]) for k in lofi.ACT_KEYS}
        dg = Diag(B, K0, env["sub"], ops)
        t1 = time.time()
        res = OC.simulate_cat(env, xs, R.Autopilot(env, B, rng), np.zeros(B),
                              seas, groups, args.T, act=act, substeps=True,
                              fb=fb, parts=True, hook=dg)
        XS, UU = res["XS"], res["U"]
        for i in range(B):
            dv = int(res["div"][i])
            info["div"].append(dv)
            n = args.T if dv < 0 else dv
            o = ops[i]
            info["test_clip"].append(o.test_clip)
            if n - K0 < 60:
                continue
            hours = (n - K0) * env["dt"] * env["sub"] / 3600.0
            # variance shares per layer
            pt = res["PARTS"][i]
            gsum = collections.defaultdict(lambda: np.zeros((n - K0, 5)))
            for key, v in pt.items():
                gsum[group_of(key)] += v[0, K0:n]
            gsum["noise"] = res["NOISE"][i, K0:n]
            for gname, v in gsum.items():
                var[gname] += v.var(0)
            # triggers
            w = int(round(10.0 / (env["dt"] * env["sub"])))
            if "W2" in pt:
                ww = windows(pt["W2"][0, K0:n, 4], w)
                trig["w2_push"].append(bool(len(ww) and (np.sqrt(
                    (ww ** 2).mean(1)) > 0.1 * a_ref[4]).any()))
                trig["W2_on"].append(True)
                trig["W2_flutter"].append(bool(o.on["W2"]["flutter"] > 0.5))
            else:
                trig["w2_push"].append(False)
                trig["W2_on"].append(False)
                trig["W2_flutter"].append(False)
            t_self = dg.porp[i] * env["dt"]            # seconds
            trig["porp_share"].append(dg.porp[i] / max(dg.n[i], 1))
            trig["porp"].append(bool(t_self >= 10.0))
            roll = res["OBS"][i, K0:n, 0]
            wr = windows(roll, w)
            sgn = np.diff(np.sign(wr - wr.mean(1, keepdims=True)), axis=1)
            heel = bool(len(wr) and (np.abs(wr.mean(1))
                                     > np.radians(2.0)).any())
            cw = bool(len(wr) and ((wr.std(1) > np.radians(2.0))
                                   & ((sgn != 0).sum(1) >= 3)).any())
            h1 = o.on.get("H1")
            trig["heel"].append(heel)
            trig["cw"].append(cw)
            trig["H1_on"].append(h1 is not None)
            trig["H1_persist"].append(bool(h1 and h1.get("persist")))
            trig["H1_cw"].append(bool(h1 and h1.get("cw")))
            trig["roll_max"].append(float(np.degrees(np.abs(roll).max())))
            trig["air"].append(dg.air[i] / max(dg.n[i], 1))
            trig["vent"].append(dg.vent[i] / max(dg.n[i], 1))
            trig["latch"].append(dg.latch[i] / max(dg.n[i], 1))
            trig["vent13"].append(dg.vent13[i] / max(dg.n[i], 1))
            trig["P13_on"].append("P13" in o.on)
            trig["vent_ph"].append(dg.vent_in[i] / hours)
            trig["P1_on"].append("P1" in o.on)
            trig["slam_ph"].append(res["SLAM"][i, K0:n].sum() / hours)
            T_s = n * env["dt"] * env["sub"]
            for c in ("P8", "P10", "P11"):
                prm = o.on.get(c)
                trig[c].append(bool(prm is not None and prm["t_f"] < T_s))
            trig["P12"].append(bool(dg.cmd_chg[i]))
            trig["noz_dev"].append(dg.noz_dev[i] / max(dg.n[i], 1))
            trig["imp_steps"].append(float((np.abs(res["IMP"][i, K0:n])
                                            .sum(-1) > 0).mean()))
            # statistics (prior_preview_gen's)
            e0 = e0_from(env, XS[i, :n], XS[i, 1:n + 1], UU[i, :n])[:, :5]
            r = PG.stats_one(e0[K0:], XS[i, K0:n])
            if r is None:
                continue
            r.update(PG.safety_stats(res["APK"][i, K0:n],
                                     res["HMIN"][i, K0:n], "x"))
            a, h = PG.coarse(XS[i], res["E15"][i, :, 12:15], fb, x_b, sp, n)
            r.update(PG.safety_stats(a, h, "c"))
            rows_all.append(r)
            ps = res["PS"][i, K0:n].reshape(-1, 5)
            ps = ps - ps.mean(0)
            pw = np.abs(np.fft.rfft(ps, axis=0)) ** 2
            f = np.fft.rfftfreq(len(ps), env["dt"])
            info["hf"].append(pw[f > 6.0].sum(0) / np.maximum(pw.sum(0),
                                                              1e-30))
            info["rms_push"].append(np.sqrt((res["P"][i, K0:n] ** 2)
                                            .mean(0)))
            info["rms_imp"].append(np.sqrt((res["IMP"][i, K0:n] ** 2)
                                           .mean(0)))
            info["clip_steps"].append(float((res["clip"][i, K0:n].sum(-1)
                                             > 0).mean()))
            info["rule_clip"].append(float(res["clip_rule"][i, K0:n].sum()
                                           / ((n - K0) * env["sub"])))
            info["clip_imp"].append(float((res["clip_imp"][i, K0:n] > 0)
                                          .mean()))
            info["clip_avg"].append(float(res["clip_avg"][i, K0:n].any(-1)
                                          .mean()))
            info["hpk"].append(float(res["hpk"][i, K0:n].sum()))
            info["hpk_clip"].append(float(res["hpk_clip"][i, K0:n].sum()))
        out(f"  batch {g0}: {B} draws, acceptance {t_acc:.0f} s, "
            f"simulation {time.time() - t1:.0f} s")
    return rows_all, info, var, att, rej, trig, fb, x_b, sp


def main():
    from learn.meta import relabel
    t0 = time.time()
    try:
        import psutil
        free = psutil.virtual_memory().available / 1e9
        t_w = time.time()
        while free < 3.0 and time.time() - t_w < 1800:
            time.sleep(60)
            free = psutil.virtual_memory().available / 1e9
        if free < 3.0:
            print("SKIPPED: free RAM below 3 GB for 30 min")
            sys.exit(2)
    except ImportError:
        pass
    Lb = np.load(os.path.join(C, "lib.npz"))
    lib = {k: Lb[k] for k in ("mu", "sd", "S")}
    env = relabel._env()
    p = env["red"].p
    out(f"prior_preview_cat: {args.closed} closed-loop episodes of "
        f"{args.T} steps, seed0 {args.seed0}, {args.act} actuators "
        f"(the conditions of prior_preview_gen.py); a_ref "
        + ", ".join(f"{c} {a:.2f}" for c, a in zip(CH, G.a_ref_of(p))))
    rows, info, var, att, rej, trig, fb, x_b, sp = closed(lib, env)
    tC, acgC = PG.split_stats("C", 64, fb, x_b, sp)
    tCb, _ = PG.split_stats("Cb", 64, fb, x_b, sp)
    target = tC + tCb
    out(f"\ntarget: {len(target)} episodes (C + Cb); prior: {len(rows)} "
        f"episodes; total {time.time() - t0:.0f} s")
    nr = np.array(info["n_reject"])
    out(f"acceptance (divergence only): {len(nr)} draws, rejections per draw"
        f" mean {nr.mean():.2f} (max {nr.max()}), rejected attempts / all "
        f"attempts {nr.sum() / (nr.sum() + len(nr)):.3f}; left with forces "
        f"off {int(np.sum(info['off']))}")
    worst = sorted(((rej[c] / att[c], c) for c in att if rej[c]),
                   reverse=True)
    out("  rejected share of attempts per item (items in any rejected "
        "attempt): " + (", ".join(f"{c} {r:.2f} ({rej[c]}/{att[c]})"
                                  for r, c in worst) or "none"))
    tc = collections.Counter(info["tier"])
    out(f"  layer-3 tiers {dict(tc)}; sparse draws "
        f"{int(np.sum(info['sparse']))}; items per draw median "
        f"{np.median(info['n_items']):.0f} (10-90% "
        f"{np.quantile(info['n_items'], 0.1):.0f}-"
        f"{np.quantile(info['n_items'], 0.9):.0f})")
    dv = np.array(info["div"])
    out(f"episodes truncated by divergence: {(dv >= 0).sum()} of {len(dv)}")
    q = np.median(np.array(info["rms_push"]), 0)
    out("push size (per-episode rms, median): "
        + "  ".join(f"{c} {x:.2f}" for c, x in zip(CH, q)))
    qi = np.median(np.array(info["rms_imp"]), 0)
    qi9 = np.quantile(np.array(info["rms_imp"]), 0.9, axis=0)
    out("impulse channel (per-episode rms of the control-step mean, median /"
        " 90%): " + "  ".join(f"{c} {x:.2f}/{y:.2f}" for c, x, y in
                              zip(CH, qi, qi9)))
    out(f"D5 clip check: control steps with any channel clipped at any "
        f"substep {np.mean(info['clip_steps']):.1e} (limit 1e-3); rule "
        f"part clipped on {np.mean(info['rule_clip']):.1e} of substeps")
    out(f"impulse channel: control steps where its bound acted "
        f"{np.mean(info['clip_imp']):.1e}; control steps with |mean push + "
        f"impulse| > CLIP x A_REF (the control-step-average check) "
        f"{np.mean(info['clip_avg']):.1e}")
    hp, hc = np.sum(info["hpk"]), np.sum(info["hpk_clip"])
    out(f"heave rule peaks: substeps above LIM/2 = {LIM[3] / 2:.0f} m/s^2: "
        f"{int(hp)}, of them clipped (> {LIM[3]:.0f}) {int(hc)} "
        f"({hc / max(hp, 1):.2f})")
    kh = np.array(info["k_heave"])
    om = np.sqrt(np.maximum(kh, 0.0)) * env["dt"]
    out(f"drawn total heave stiffness (draft 6 item 6, max over the task "
        f"speeds; omega dt_sub, dt_sub {env['dt']:.2f} s): median "
        f"{np.median(om):.3f}, 90% {np.quantile(om, 0.9):.3f}, max "
        f"{om.max():.3f}; draws with omega dt > 1: {(om > 1).mean():.3f} "
        f"(low-fidelity alone {ctx_wh(env) * env['dt']:.3f})")
    hf = np.median(np.array(info["hf"]), 0)
    out("injected acceleration power above 6 Hz (median share): "
        + "  ".join(f"{c} {x:.3f}" for c, x in zip(CH, hf)))
    tot = sum(var[g] for g in var)
    out("\nVARIANCE SHARE per channel (sum over episodes of each part's "
        "variance over the control steps / sum over the parts)")
    out(f"  {'part':<8} " + " ".join(f"{c:>7}" for c in CH))
    for gname in GROUPS + tuple(g for g in var if g not in GROUPS):
        if gname not in var:
            continue
        out(f"  {gname:<8} " + " ".join(f"{x:7.3f}" for x in
                                        var[gname] / np.maximum(tot, 1e-30)))
    l3 = var["L3"] / np.maximum(tot, 1e-30)
    bad = [c for c, x in zip(CH, l3) if x > 0.3]
    out("  layer 3 above 0.3 of a channel's variance (draft section 5 "
        "reason 4: then strengthen the catalogue there): "
        + (", ".join(bad) if bad else "none"))
    T_ = lambda k: np.array(trig[k], float)                   # noqa: E731
    n_ep = len(trig["porp"])
    out(f"\nTRIGGER RATES ({n_ep} episodes, steps 40..)")
    w2 = T_("W2_on") > 0
    fl = T_("W2_flutter") > 0
    vdp = w2 & ~fl
    out(f"  porpoising by mechanism (W2 net self-excitation, van der Pol "
        f"rho_n gate > 1 / flutter rho_f gate^2 > 1, for >= 10 s): "
        f"{T_('porp').mean():.3f} of all episodes; "
        f"{T_('porp')[w2].mean() if w2.any() else np.nan:.3f} of the "
        f"{int(w2.sum())} with W2 on (van der Pol "
        f"{T_('porp')[vdp].mean() if vdp.any() else np.nan:.3f} of "
        f"{int(vdp.sum())}, flutter "
        f"{T_('porp')[fl].mean() if fl.any() else np.nan:.3f} of "
        f"{int(fl.sum())}); time share with W2 on mean "
        f"{T_('porp_share')[w2].mean() if w2.any() else np.nan:.3f}")
    out(f"  W2 push size in waves (pitch push 10 s RMS > 0.1 a_ref; not "
        f"self-excitation): {T_('w2_push').mean():.3f} of all, "
        f"{T_('w2_push')[w2].mean() if w2.any() else np.nan:.3f} of W2 on")
    h1, hp_, hc_ = T_("H1_on") > 0, T_("H1_persist") > 0, T_("H1_cw") > 0
    out(f"  sustained heel (10 s |mean roll| > 2 deg): {T_('heel').mean():.3f}"
        f" of all; {T_('heel')[hp_].mean() if hp_.any() else np.nan:.3f} of"
        f" the {int(hp_.sum())} with H1's two stable heels drawn")
    out(f"  chine walking (10 s roll std > 2 deg, >= 3 sign changes): "
        f"{T_('cw').mean():.3f} of all; "
        f"{T_('cw')[hc_].mean() if hc_.any() else np.nan:.3f} of the "
        f"{int(hc_.sum())} with chine walking drawn; H1 on in "
        f"{int(h1.sum())}; max |roll| median "
        f"{np.median(T_('roll_max')):.2f} deg (90% "
        f"{np.quantile(T_('roll_max'), 0.9):.2f})")
    p1 = T_("P1_on") > 0
    p13 = T_("P13_on") > 0
    out(f"  intake ventilation (P1's own suction < 0.9): time share median /"
        f" 90% "
        f"{np.median(T_('vent')[p1]) if p1.any() else np.nan:.4f} / "
        f"{np.quantile(T_('vent')[p1], 0.9) if p1.any() else np.nan:.4f}, "
        f"entries per hour mean "
        f"{T_('vent_ph')[p1].mean() if p1.any() else np.nan:.1f}; latch "
        f"(suction lost) time share median / 90% "
        f"{np.median(T_('latch')[p1]) if p1.any() else np.nan:.4f} / "
        f"{np.quantile(T_('latch')[p1], 0.9) if p1.any() else np.nan:.4f} "
        f"({int(p1.sum())} episodes with P1)")
    out(f"  P13 slam-air factor < 0.9: time share median / 90% "
        f"{np.median(T_('vent13')[p13]) if p13.any() else np.nan:.4f} / "
        f"{np.quantile(T_('vent13')[p13], 0.9) if p13.any() else np.nan:.4f}"
        f" ({int(p13.sum())} episodes with P13)")
    out(f"  airborne time (all five keel immersions <= 0): mean "
        f"{T_('air').mean():.4f}, 90% {np.quantile(T_('air'), 0.9):.4f}, "
        f"max {T_('air').max():.4f}")
    out(f"  slams per hour (low-fidelity bow rule): median "
        f"{np.median(T_('slam_ph')):.0f}, 90% "
        f"{np.quantile(T_('slam_ph'), 0.9):.0f}; control steps with an "
        f"impulse median {np.median(T_('imp_steps')):.3f}")
    out(f"  faults in the episode: P8 stuck bucket {T_('P8').mean():.3f}, "
        f"P10 thrust loss {T_('P10').mean():.3f}, P11 nozzle "
        f"{T_('P11').mean():.3f}, P12 rule changed the command "
        f"{T_('P12').mean():.3f}; nozzle > 2 deg off the actuator (P7 / "
        f"P11) time share mean {T_('noz_dev').mean():.4f}")
    old, d8, gen = PG.parse_old(), PG.parse_d8(), parse_gen_k1()
    keys = [k for k in target[0] if not k.startswith(("Apk", "Hmin"))]
    out("\nCLOSED LOOP, D10a catalogue family (m10 actuators)")
    out(f"  {'statistic':<11} {'old (log) 5/50/95':>23} | "
        f"{'cat 5/50/95':>23} | target med | target pct in cat")
    for k in keys:
        pv = np.array([r.get(k, np.nan) for r in rows])
        tv = np.array([r[k] for r in target])
        o = old.get(k)
        qq = np.nanquantile(pv, [0.05, 0.5, 0.95])
        pl = PG.pct_line(pv, tv)
        out(f"  {k:<11} " + (f"{o[0]:+7.2f} {o[1]:+7.2f} {o[2]:+7.2f}"
                             if o else " " * 23)
            + f" | {qq[0]:+7.2f} {qq[1]:+7.2f} {qq[2]:+7.2f} | "
            f"{np.nanmedian(tv):+9.3f} | {pl[0]:5.1f}% ({pl[1]:.2f}/"
            f"{pl[2]:.2f})")
    out("\nSAFETY QUANTITIES per episode (steps 40..): exact = substeps "
        "(prior only, impulses included); coarse = control steps, same in "
        "both worlds")
    out(f"  {'statistic':<14} {'cat exact 5/50/95':>23} | "
        f"{'cat coarse 5/50/95':>23} | target coarse med | target pct in "
        "cat coarse")
    for base in ("Apk_q50", "Apk_q95", "Apk_max", "Hmin_q50", "Hmin_q05",
                 "Hmin_min"):
        px = np.array([r.get(base + "_x", np.nan) for r in rows])
        pc = np.array([r.get(base + "_c", np.nan) for r in rows])
        tv = np.array([r.get(base + "_c", np.nan) for r in target])
        qx = np.nanquantile(px, [0.05, 0.5, 0.95])
        qc = np.nanquantile(pc, [0.05, 0.5, 0.95])
        pl = PG.pct_line(pc, tv)
        out(f"  {base:<14} {qx[0]:+7.2f} {qx[1]:+7.2f} {qx[2]:+7.2f} | "
            f"{qc[0]:+7.2f} {qc[1]:+7.2f} {qc[2]:+7.2f} | "
            f"{np.nanmedian(tv):+17.3f} | {pl[0]:5.1f}% ({pl[1]:.2f}/"
            f"{pl[2]:.2f})")
    out("\nSUMMARY: target median percentile (share below 5% / above 95%)")
    out(f"  {'statistic':<11} {'old projection (log)':>22} | "
        f"{'D8 stage 1, closed v2':>22} | {'general D9 (k1 log)':>22} | "
        f"{'D10a (this run)':>22}")
    for k in keys:
        o, d, gg = old.get(k), d8.get(k), gen.get(k)
        pv = np.array([r.get(k, np.nan) for r in rows])
        tv = np.array([r[k] for r in target])
        pl = PG.pct_line(pv, tv)
        f = lambda z: (f"{z[0]:5.1f}% ({z[1]:.2f}/{z[2]:.2f})"   # noqa
                       if z else "-")
        out(f"  {k:<11} {f(o[3:] if o else None):>22} | {f(d):>22} | "
            f"{f(gg):>22} | {f(pl):>22}")
    with open(LOG + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(_OUT) + "\n")
    os.replace(LOG + ".tmp", LOG)
    print(f"wrote {LOG}")


if __name__ == "__main__":
    main()
