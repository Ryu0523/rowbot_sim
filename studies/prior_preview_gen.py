#!/usr/bin/env python3
"""
Prior-predictive preview of the general operator family (learn/meta/
operators_gen.py; PRIOR_DERIVATION.md D9.8). STANDALONE: no data pipeline,
nothing written except the log. Descriptive only (no statement about
information or ceilings; target statistics are a check, never a fit).

CLOSED LOOP only (the forces are state feedback at every substep): N fresh
episodes in the reduced world, the same conditions as prior_preview_rb.py
--closed (seed0 777: same seas, start states, autopilot, M10 actuator draws;
operator seeds seed0 * 1000 + i, drawn from the general family and
acceptance-tested in batches). e0 = data3.e0_from on the simulated states.

Reported:
  - acceptance: rejections per draw, reasons, draws left without forces;
    episodes truncated by divergence
  - push size, the D5 clip check, the rule-clip share, the share of the
    injected acceleration's power above 6 Hz (D9.8 item 5)
  - the per-episode statistics of prior_check_m15 (e0 channels 0-4) with
    the old family's quantiles (prior_check_m15.log), the target median
    (C + Cb) and the target episodes' median percentile in the new prior
  - the safety quantities per episode (steps 40..): APK q50 / q95 / max,
    HMIN q50 / q05 / min, EXACT (substeps, D9.8) for the prior; the target
    cache has control-step states only, so both are also computed COARSE
    in the same way in both worlds (A_c = zdot difference over one control
    step / 0.24 s; H_c = the bow height at the control-step starts), and
    the target percentile is taken in the coarse prior
  - a summary table: target median percentile (share below 5% / above 95%)
    for the old projection family (log), D8 stage 1 closed loop (v2 log)
    and the general family
  - how varied / nonlinear the forces are IN CLOSED LOOP (steps 40..):
    std of each standardised linear filter output, share of relays that
    switch, per force std / RMS of the readout output (small = nearly
    constant) and std(output - its linearisation at the mean input) /
    std(output) (small = nearly linear); the same two numbers on the
    library cloud (draw time, style); the acceptance test's rule-clip share

    python studies/prior_preview_gen.py [--closed 64] [--T 375]
"""
import argparse
import collections
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from learn.meta import operators_gen as G
from learn.meta import operators_rb as R

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="meta3")
ap.add_argument("--closed", type=int, default=64)
ap.add_argument("--T", type=int, default=375)
ap.add_argument("--seed0", type=int, default=777)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--act", default="m10", choices=("m10", "ideal"))
args = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "studies", "_cache", args.cache)
CH = R.CHANNELS
VEL = (6, 7, 11, 8, 10)
K0 = 40                          # statistics from step 40 (as prior_check)


# ------------------------------------------------ prior_check_m15's stats
def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-30))


def slope(y, x):
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / (x @ x + 1e-30) * (x.std() + 1e-30)
                 / (y.std() + 1e-30))


def stats_one(e, xs):
    """prior_check_m15.stats_of for one episode (as prior_preview_rb)."""
    if len(e) < 60 or not np.isfinite(e).all():
        return None
    ok = e.std(0) > 1e-9
    r = {}
    for c, nm in enumerate(CH):
        r[f"rms_{nm}"] = float(np.sqrt((e[:, c] ** 2).mean()))
        r[f"ac1_{nm}"] = corr(e[1:, c], e[:-1, c])
        r[f"damp_{nm}"] = corr(e[:, c], xs[:, VEL[c]])
    r["sync_sy"] = corr(e[:, 1], e[:, 2])
    r["sync_hp"] = corr(e[:, 3], e[:, 4])
    r["slope_th"] = slope(e[:, 4], xs[:, 4])
    r["slope_u"] = slope(e[:, 4], xs[:, 6])
    for c, nm in enumerate(CH):
        if not ok[c]:
            r[f"ac1_{nm}"] = r[f"damp_{nm}"] = np.nan
    if not (ok[1] and ok[2]):
        r["sync_sy"] = np.nan
    if not (ok[3] and ok[4]):
        r["sync_hp"] = np.nan
    if not ok[4]:
        r["slope_th"] = r["slope_u"] = np.nan
    return r


def safety_stats(a, h, tag):
    if len(a) < 20 or not (np.isfinite(a).all() and np.isfinite(h).all()):
        return {}
    return {f"Apk_q50_{tag}": float(np.quantile(a, 0.5)),
            f"Apk_q95_{tag}": float(np.quantile(a, 0.95)),
            f"Apk_max_{tag}": float(a.max()),
            f"Hmin_q50_{tag}": float(np.quantile(h, 0.5)),
            f"Hmin_q05_{tag}": float(np.quantile(h, 0.05)),
            f"Hmin_min_{tag}": float(h.min())}


def coarse(XS, E_bow, fb, x_b, sp, n):
    """Control-step versions, the same in both worlds: A_c[k] = (zdot_{k+1}
    - zdot_k) / 0.24, H_c[k] = F_b + z_k + s_p x_b theta_k - max of the
    three bow-station elevations at the step start; k = 40 .. n - 1."""
    a = (XS[K0 + 1:n + 1, 8] - XS[K0:n, 8]) / 0.24
    h = fb + XS[K0:n, 2] + sp * x_b * XS[K0:n, 4] - E_bow[K0:n].max(1)
    return a, h


def split_stats(split, n_max, fb, x_b, sp):
    d = np.load(os.path.join(C, f"{split}.npz"))
    e0 = np.load(os.path.join(C, f"{split}_e0.npz"))["E0"]
    L, XS, S = d["len"], d["XS"], d["S"]
    n_e = L - 1 if XS.shape[1] == d["U"].shape[1] else L
    rows, acg = [], []
    for i in range(min(n_max, len(n_e))):
        n = int(n_e[i])
        if n < 100:
            continue
        r = stats_one(e0[i, K0:n, :5].astype(float),
                      XS[i, K0:n].astype(float))
        if r is None:
            continue
        a, h = coarse(XS[i].astype(float), S[i, :, 23:26].astype(float),
                      fb, x_b, sp, min(n, XS.shape[1] - 1))
        r.update(safety_stats(a, h, "c"))
        rows.append(r)
        if "ACG" in d.files:
            acg.append(float(np.quantile(np.abs(d["ACG"][i, K0:n]), 0.95)))
    return rows, acg


def parse_old():
    """prior_check_m15.log: stat -> (q5, q50, q95, pct, below, above)."""
    out = {}
    p = os.path.join(C, "prior_check_m15.log")
    pat = re.compile(r"^\s+(\w+)\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)\s+"
                     r"[-+\d.]+\s+([\d.]+)%\s+\(([\d.]+) / ([\d.]+)\)")
    for line in open(p):
        m = pat.match(line)
        if m:
            g = m.groups()
            out[g[0]] = tuple(float(x) for x in g[1:])
    return out


def parse_d8():
    """prior_preview_rb_v2.log, section 'CLOSED LOOP in the reduced world
    (m10 actuators)': stat -> (pct, below, above)."""
    out = {}
    p = os.path.join(C, "prior_preview_rb_v2.log")
    if not os.path.exists(p):
        return out
    on = False
    pat = re.compile(r"^\s+(\w+)\s.*\|\s+[-+\d.]+\s+\|\s+([\d.]+)% "
                     r"\(([\d.]+)/([\d.]+)\)\s*$")
    for line in open(p):
        if line.startswith("CLOSED LOOP in the reduced world"):
            on = True
            continue
        if on and line.startswith("  closed "):
            break
        if on:
            m = pat.match(line)
            if m:
                out[m.group(1)] = tuple(float(x) for x in m.groups()[1:])
    return out


def pct_line(p, t):
    p = p[np.isfinite(p)]
    t = t[np.isfinite(t)]
    if len(p) < 3 or len(t) == 0:
        return None
    pct = np.array([(p < v).mean() for v in t]) * 100
    return float(np.median(pct)), float((pct < 5).mean()), \
        float((pct > 95).mean())


# --------------------------------------------------------------- closed
def closed(lib, env):
    from learn.meta.data3 import e0_from
    from sim import lofi
    p = env["red"].p
    fb = float(env["mission"].plant.freeboard)
    x_b, sp = float(env["x_st"][-1]), float(p["sign_pitch"])
    rows_all, info = [], collections.defaultdict(list)
    reasons = collections.Counter()
    n_eps, seed0 = args.closed, args.seed0
    for g0 in range(0, n_eps, args.batch):
        B = min(args.batch, n_eps - g0)
        rng = np.random.default_rng([seed0, g0])
        t0 = time.time()
        ops = G.build_ops([int(seed0 * 1000 + g0 + i) for i in range(B)],
                          lib, dt=env["dt"] * env["sub"], L=env["L"])
        t_acc = time.time() - t0
        for o in ops:
            info["n_reject"].append(o.n_reject)
            info["off"].append(o.forces_off)
            info["kappa0"].append(o.kappa0)
            info["n_forces"].append(len(o.forces))
            for r in o.reject_log:
                reasons[r["reason"]] += 1
                info["rej_kappa0"].append(r["kappa0"])
        groups = [dict(op=o, st=o.new_state(1, rng=np.random.default_rng(
            [o.seed, 1]), diag=True), rows=np.array([i]),
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
        t1 = time.time()
        out = G.simulate_gen(env, xs, R.Autopilot(env, B, rng), np.zeros(B),
                             seas, groups, args.T, act=act, substeps=True,
                             fb=fb)
        XS, UU = out["XS"], out["U"]
        for i in range(B):
            dv = int(out["div"][i])
            info["div"].append(dv)
            n = args.T if dv < 0 else dv
            o = ops[i]
            info["test_clip"].append(o.style["test_clip"])
            info["lib_std_rms"] += o.style["lib_std_rms"]
            info["lib_nonlin"] += o.style["lib_nonlin"]
            for f, fs in zip(o.forces, groups[i]["st"]["fx"]):
                Y = np.concatenate(fs["ylog"][K0 * env["sub"]:
                                              n * env["sub"]], 0)
                fs["ylog"] = []
                if len(Y) < 60 * env["sub"]:
                    continue
                lin, rel = f["filt"]["lin"], f["filt"]["rel"]
                info["filt_sd"] += list(Y[:, lin].std(0))
                info["relay_sw"] += [float(np.ptp(Y[:, j]) > 0) for j in rel]
                dg = G.readout_diag(f["ro"], Y)
                info["cl_std_rms"].append(dg["std_rms"])
                info["cl_nonlin"].append(dg["nonlin"])
            if n - K0 < 60:
                continue
            e0 = e0_from(env, XS[i, :n], XS[i, 1:n + 1], UU[i, :n])[:, :5]
            r = stats_one(e0[K0:], XS[i, K0:n])
            if r is None:
                continue
            r.update(safety_stats(out["APK"][i, K0:n], out["HMIN"][i, K0:n],
                                  "x"))
            a, h = coarse(XS[i], out["E15"][i, :, 12:15], fb, x_b, sp, n)
            r.update(safety_stats(a, h, "c"))
            rows_all.append(r)
            ps = out["PS"][i, K0:n].reshape(-1, 5)
            ps = ps - ps.mean(0)
            pw = np.abs(np.fft.rfft(ps, axis=0)) ** 2
            f = np.fft.rfftfreq(len(ps), env["dt"])
            info["hf"].append(pw[f > 6.0].sum(0) / np.maximum(pw.sum(0),
                                                              1e-30))
            info["rms_push"].append(np.sqrt((out["P"][i, K0:n] ** 2)
                                            .mean(0)))
            info["clip_steps"].append(float((out["clip"][i, K0:n].sum(-1)
                                             > 0).mean()))
            info["rule_clip"].append(float(out["clip_rule"][i, K0:n].sum()
                                           / ((n - K0) * env["sub"])))
        print(f"  batch {g0}: {B} draws, acceptance {t_acc:.0f} s, "
              f"simulation {time.time() - t1:.0f} s", flush=True)
    return rows_all, info, reasons, fb, x_b, sp


def main():
    from learn.meta import relabel
    t0 = time.time()
    Lb = np.load(os.path.join(C, "lib.npz"))
    lib = {k: Lb[k] for k in Lb.files}
    env = relabel._env()
    p = env["red"].p
    print(f"freeboard: plant {env['mission'].plant.freeboard:.3f} m, "
          f"module {G.freeboard(p):.3f} m; a_ref "
          + ", ".join(f"{c} {a:.2f}" for c, a in zip(CH, G.a_ref_of(p))))
    rows, info, reasons, fb, x_b, sp = closed(lib, env)
    tC, acgC = split_stats("C", 64, fb, x_b, sp)
    tCb, _ = split_stats("Cb", 64, fb, x_b, sp)
    target = tC + tCb
    print(f"\ntarget: {len(target)} episodes (C + Cb); prior: {len(rows)} "
          f"episodes; total {time.time() - t0:.0f} s")
    nr = np.array(info["n_reject"])
    print(f"acceptance: {len(nr)} draws, rejections per draw mean "
          f"{nr.mean():.2f} (max {nr.max()}), rejected attempts / all "
          f"attempts {nr.sum() / (nr.sum() + len(nr)):.2f}; reasons "
          f"{dict(reasons)}; left without forces {int(np.sum(info['off']))}")
    if info["rej_kappa0"]:
        print(f"  kappa0 median: rejected attempts "
              f"{np.median(info['rej_kappa0']):.3f}, accepted draws "
              f"{np.median(info['kappa0']):.3f}; forces per accepted draw "
              f"mean {np.mean(info['n_forces']):.1f}")
    dv = np.array(info["div"])
    print(f"episodes truncated by divergence: {(dv >= 0).sum()} of {len(dv)}")
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]

    def qline(name, v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return f"  {name:<34} -"
        return (f"  {name:<34} " + " ".join(f"{x:5.2f}" for x in
                                           np.quantile(v, qs))
                + f"   (n {len(v)})")
    print("\nFORCES IN CLOSED LOOP (steps 40..), quantiles 10/25/50/75/90%")
    print(qline("filter output std (linear, unit=lib)", info["filt_sd"]))
    print(qline("force output std / RMS", info["cl_std_rms"]))
    print(qline("force nonlinearity (dev / std)", info["cl_nonlin"]))
    print(qline("  on the library: std / RMS", info["lib_std_rms"]))
    print(qline("  on the library: nonlinearity", info["lib_nonlin"]))
    print(qline("acceptance test rule-clip share", info["test_clip"]))
    sw = np.asarray(info["relay_sw"])
    print(f"  relays that switch in closed loop: {int(sw.sum())} of "
          f"{len(sw)}")
    for nm, v, thr in (("std / RMS", info["cl_std_rms"], (0.1, 0.3)),
                       ("nonlinearity", info["cl_nonlin"], (0.1, 0.3))):
        v = np.asarray(v, float)
        print(f"  share of forces with {nm} below {thr[0]}: "
              f"{(v < thr[0]).mean():.2f}, below {thr[1]}: "
              f"{(v < thr[1]).mean():.2f}")
    q = np.median(np.array(info["rms_push"]), 0)
    print("push size (per-episode rms, median): "
          + "  ".join(f"{c} {x:.2f}" for c, x in zip(CH, q)))
    print(f"D5 clip check: control steps with any channel clipped at any "
          f"substep {np.mean(info['clip_steps']):.1e} (limit 1e-3); rule "
          f"part clipped on {np.mean(info['rule_clip']):.1e} of substeps")
    hf = np.median(np.array(info["hf"]), 0)
    print("injected acceleration power above 6 Hz (median share): "
          + "  ".join(f"{c} {x:.3f}" for c, x in zip(CH, hf)))
    if acgC:
        print(f"(info: high-fidelity ACG, an instantaneous value at its step "
              f"start, not the same definition: median per-episode q95 "
              f"|ACG| {np.median(acgC):.2f} g)")
    old = parse_old()
    d8 = parse_d8()
    keys = [k for k in target[0] if not k.startswith(("Apk", "Hmin"))]
    print("\nCLOSED LOOP, general family (m10 actuators)")
    print(f"  {'statistic':<11} {'old (log) 5/50/95':>23} | "
          f"{'gen 5/50/95':>23} | target med | target pct in gen")
    for k in keys:
        pv = np.array([r.get(k, np.nan) for r in rows])
        tv = np.array([r[k] for r in target])
        o = old.get(k)
        qq = np.nanquantile(pv, [0.05, 0.5, 0.95])
        pl = pct_line(pv, tv)
        print(f"  {k:<11} " + (f"{o[0]:+7.2f} {o[1]:+7.2f} {o[2]:+7.2f}"
                               if o else " " * 23)
              + f" | {qq[0]:+7.2f} {qq[1]:+7.2f} {qq[2]:+7.2f} | "
              f"{np.nanmedian(tv):+9.3f} | {pl[0]:5.1f}% ({pl[1]:.2f}/"
              f"{pl[2]:.2f})")
    print("\nSAFETY QUANTITIES per episode (steps 40..): exact = substeps "
          "(prior only); coarse = control steps, same in both worlds")
    print(f"  {'statistic':<14} {'gen exact 5/50/95':>23} | "
          f"{'gen coarse 5/50/95':>23} | target coarse med | target pct in "
          "gen coarse")
    for base in ("Apk_q50", "Apk_q95", "Apk_max", "Hmin_q50", "Hmin_q05",
                 "Hmin_min"):
        px = np.array([r.get(base + "_x", np.nan) for r in rows])
        pc = np.array([r.get(base + "_c", np.nan) for r in rows])
        tv = np.array([r.get(base + "_c", np.nan) for r in target])
        qx = np.nanquantile(px, [0.05, 0.5, 0.95])
        qc = np.nanquantile(pc, [0.05, 0.5, 0.95])
        pl = pct_line(pc, tv)
        print(f"  {base:<14} {qx[0]:+7.2f} {qx[1]:+7.2f} {qx[2]:+7.2f} | "
              f"{qc[0]:+7.2f} {qc[1]:+7.2f} {qc[2]:+7.2f} | "
              f"{np.nanmedian(tv):+17.3f} | {pl[0]:5.1f}% ({pl[1]:.2f}/"
              f"{pl[2]:.2f})")
    print("\nSUMMARY: target median percentile (share below 5% / above 95%)")
    print(f"  {'statistic':<11} {'old projection (log)':>22} | "
          f"{'D8 stage 1, closed v2':>22} | {'general (this run)':>22}")
    for k in keys:
        o, d = old.get(k), d8.get(k)
        pv = np.array([r.get(k, np.nan) for r in rows])
        tv = np.array([r[k] for r in target])
        pl = pct_line(pv, tv)
        f = lambda z: (f"{z[0]:5.1f}% ({z[1]:.2f}/{z[2]:.2f})"   # noqa
                       if z else "-")
        print(f"  {k:<11} {f(o[3:] if o else None):>22} | {f(d):>22} | "
              f"{f(pl):>22}")


if __name__ == "__main__":
    main()
