#!/usr/bin/env python3
"""
Can the planners of studies/final_eval.py represent and choose a throttle
locked to the wave phase ("push while the bow climbs, ease off at the
crest")? Two parts, both light:

  consts  the numbers the argument needs: control step, knot grid, reduced
          surge model, the plant's thrust chain (delay, lag, rate limit,
          dead band), encounter periods (deep water, Tp 5 s; and the
          spectral mean of the actual sea), how much of a sinusoid at the
          encounter period the 7-knot linear spline can hold, what the warm
          start's one-step shift leaves of it, its W_JUMP price, and the
          thrust chain's response to a sinusoidal command at that period
  score   from closed-loop states of the final_eval mission (the hand MPC
          drives seed s, leg l), the cost every planner gives the SAME
          designed plans: constants and sinusoids at the encounter period
          with 8 phases relative to the bow-wave crest. hand: control/mpc.py
          rollout_cost split per term; m0 and cat_p_prior:
          learn/meta/mpc_constrained.ConstrainedMPPI with its sampled
          candidates replaced by the designed ones (its own rollouts,
          constraints, W_JUMP), cost parts captured per candidate

Nothing tracked is modified; outputs in studies/_cache/wave_phase/.

    python -m studies.wave_phase_planners --part consts
    python -m studies.wave_phase_planners --part score --seeds 0 --legs 0,1
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np   # noqa: E402

KN = 0.514444
G = 9.81
OUT = os.path.join(HERE, "_cache", "wave_phase")


def log(msg, f=None):
    print(msg, flush=True)
    if f is not None:
        f.write(msg + "\n")
        f.flush()


def omega_e(U, leg, tp=5.0):
    """Peak encounter frequency (rad/s), deep water, waves propagating
    towards pi (task.WAVES_FROM), boat heading task.LEGS[leg]."""
    w = 2 * math.pi / tp
    k = w * w / G
    psi = (0.0, math.pi / 4)[leg]
    return w - k * U * math.cos(math.pi - psi)


def spectral_encounter(sea, U, psi):
    """(peak, mean zero-crossing) encounter periods of a SeaState's
    components for a boat at speed U on heading psi."""
    a, w, k, th = (np.asarray(v, float) for v in (sea.a, sea.w, sea.k,
                                                   sea.th))
    we = np.abs(w - k * U * np.cos(th - psi))
    m0 = (a * a).sum() / 2
    m2 = (a * a * we * we).sum() / 2
    hist, edges = np.histogram(we, bins=60, weights=a * a)
    i = int(np.argmax(hist))
    wpk = 0.5 * (edges[i] + edges[i + 1])
    return 2 * math.pi / wpk, 2 * math.pi * math.sqrt(m0 / m2)


# ------------------------------------------------------------------ consts
def part_consts(f):
    from learn.meta import mpc_learned as ML
    from learn.meta import mpc_constrained as MC
    from studies import final_eval as FE
    from studies.sim2real_jet import theta, z_hand
    env = ML.env()
    p = env["red"].p
    dtc = env["dt"] * env["sub"]
    H, NK = ML.H, ML.N_KNOTS
    E, Sh = ML.knot_matrices()
    out = dict(dt=env["dt"], sub=env["sub"], dtc=dtc, H=H, n_knots=NK,
               t_max=env["t_max"], u_ref=env["u_ref"], L=env["L"])
    for kk in ("tau_u", "k_drag", "m_surge", "u_design", "dt_ctrl",
               "wn_heave", "z_heave", "wn_pitch", "z_pitch", "z0", "th0",
               "u_ss", "steer_jet", "planing"):
        out["red_" + kk] = float(p.get(kk, float("nan")))
    kt = np.linspace(0, H - 1, NK)
    out["knot_spacing_steps"] = float(kt[1] - kt[0])
    out["knot_spacing_s_constrained"] = float((kt[1] - kt[0]) * dtc)
    out["horizon_s_constrained"] = H * dtc
    # the hand MPC as final_eval builds it
    w, floor = theta(z_hand())
    m = FE.make_mission(0, 0, 120.0, weights=w, floor=floor)
    c = m.ep.ctrl
    out.update(hand_dt=c.dt, hand_H=c.H, hand_n_knots=c.n_knots, hand_K=c.K,
               hand_sigma=c.sigma.tolist(), hand_lam=c.lam,
               hand_floor=c.thrust_floor, hand_t_preview=c.pv.t_preview,
               hand_rel_noise=getattr(c.pv, "pv", c.pv).rel_noise,
               hand_weights=dict(zip(
                   ["w_acc", "w_slam", "w_speed", "w_track", "w_heading",
                    "w_dthrust", "w_drudder"], [float(x) for x in c.w])),
               hand_u_ref=c.u_ref, hand_use_rudder=c.use_rudder,
               hand_knot_spacing_s=float((c.knot_t[1] - c.knot_t[0]) * c.dt))
    pr = m.plant.prop
    out.update(plant=type(m.plant).__name__, plant_dt=m.dt, plant_sub=m.sub,
               prop_type=type(pr).__name__, prop_tau=pr.tau,
               prop_rate_max_frac_per_s=pr.rate_max / pr.t_max,
               prop_dead_band_frac=pr.dead_band / pr.t_max,
               prop_delay_steps=pr.delay.n, prop_delay_s=pr.delay.n * m.dt,
               prop_t_max=pr.t_max, prop_t_min=pr.t_min,
               nozzle_tau=m.plant.rudder.tau if hasattr(m.plant.rudder, "tau")
               else None)
    out.update(K_CAND=ML.K_CAND, SIGMA=ML.SIGMA, LAM=ML.LAM, W_JUMP=ML.W_JUMP,
               S_final_eval=FE.OPT["S"], K_final_eval=FE.OPT["K"],
               fc_final_eval=dict(lam=FE.OPT["fc_lam"], hp=FE.OPT["fc_hp"]),
               limits=MC.Limits().__dict__)
    for kk, v in out.items():
        log(f"  {kk:>28} = {v}", f)

    # encounter periods
    log("\n  encounter (deep water, Tp 5 s peak; spectral = seed-0 sea "
        "components, a^2 weighted)", f)
    enc = {}
    for kn in (20.0, 28.0):
        U = kn * KN
        for leg in (0, 1):
            we = omega_e(U, leg)
            Te = 2 * math.pi / we
            tpk, tz = spectral_encounter(m.sea, U, (0.0, math.pi / 4)[leg])
            enc[f"{kn:.0f}kn_leg{leg}"] = dict(we=we, Te=Te, Te_spec_peak=tpk,
                                              Te_spec_tz=tz)
            log(f"    {kn:.0f} kn leg {leg}: we {we:.3f} rad/s, Te {Te:.3f} s "
                f"(spectral peak {tpk:.2f} s, mean zero-crossing {tz:.2f} s); "
                f"knots per period {Te / out['knot_spacing_s_constrained']:.2f}"
                f" (constrained), {Te / out['hand_knot_spacing_s']:.2f} (hand)",
                f)
    out["encounter"] = enc
    lam_w = 2 * math.pi / ((2 * math.pi / 5.0) ** 2 / G)
    out["wavelength_m"] = lam_w
    log(f"    wavelength at Tp 5 s: {lam_w:.2f} m", f)

    # knot-spline representability, warm-start attenuation, W_JUMP
    log("\n  7-knot linear spline vs cos(we t + phi) over the 24-step grid "
        "(mean over 36 phases)", f)
    rep = {}
    for name, e in enc.items():
        for which, dt_ in (("constrained", dtc), ("hand", c.dt)):
            tj = (np.arange(H) + 0.5) * dt_
            cap, att1, att4, wj = [], [], [], []
            for ph in np.linspace(0, 2 * np.pi, 36, endpoint=False):
                s = np.cos(e["we"] * tj + ph)
                kn_, *_ = np.linalg.lstsq(E, s, rcond=None)
                fit = E @ kn_
                cap.append(1 - np.sum((s - fit) ** 2) / np.sum(s ** 2))
                # warm start: one shift vs the ideal next plan (the same
                # sinusoid one step later, fitted)
                s1 = np.cos(e["we"] * (tj + dt_) + ph)
                k1, *_ = np.linalg.lstsq(E, s1, rcond=None)
                sh = kn_.copy()
                sh = Sh @ sh
                att1.append(np.std((E @ sh)[:20]) / np.std((E @ k1)[:20]))
                sh4 = kn_.copy()
                for _ in range(4):
                    sh4 = Sh @ sh4
                s4 = np.cos(e["we"] * (tj + 4 * dt_) + ph)
                k4, *_ = np.linalg.lstsq(E, s4, rcond=None)
                att4.append(np.std((E @ sh4)[:12]) / np.std((E @ k4)[:12]))
                wj.append(float(np.sum(np.diff(fit) ** 2)))
            rep[f"{name}_{which}"] = dict(
                captured=float(np.mean(cap)), captured_min=float(np.min(cap)),
                shift1_amp_ratio=float(np.mean(att1)),
                shift4_amp_ratio=float(np.mean(att4)),
                sum_dU2_unit_amp=float(np.mean(wj)))
            r = rep[f"{name}_{which}"]
            log(f"    {name:>12} {which:>11}: captured {r['captured']:.3f} "
                f"(min {r['captured_min']:.3f}), after 1 warm-start shift "
                f"{r['shift1_amp_ratio']:.2f} of the ideal amplitude, after "
                f"4 {r['shift4_amp_ratio']:.2f}; sum (dU)^2 per unit "
                f"amplitude^2 {r['sum_dU2_unit_amp']:.3f}", f)
    out["spline"] = rep
    # alternating knots (the highest frequency the spline holds)
    alt = np.array([(-1) ** i for i in range(NK)], float)
    out["alt_knot_period_s"] = 2 * out["knot_spacing_s_constrained"]
    out["alt_sum_dU2_unit_amp"] = float(np.sum(np.diff(E @ alt) ** 2))
    out["alt_shift1_ratio"] = float(np.std((E @ (Sh @ alt))[:20])
                                    / np.std((E @ alt)[:20]))
    log(f"    alternating knots: period {out['alt_knot_period_s']:.3f} s, "
        f"sum (dU)^2 per unit amplitude^2 {out['alt_sum_dU2_unit_amp']:.3f}, "
        f"one shift keeps {out['alt_shift1_ratio']:.2f} of its amplitude", f)
    for A in (0.1, 0.25, 0.5):
        log(f"    W_JUMP x sum(dU)^2, amplitude {A} (thrust fraction): "
            + ", ".join(f"{nm} {ML.W_JUMP * A * A * r['sum_dU2_unit_amp']:.3f}"
                        for nm, r in rep.items() if nm.endswith("constrained")),
            f)
    log(f"    hand w_dthrust {c.w[5]:.2f} x sum(dU)^2, amplitude 0.25: "
        + ", ".join(f"{nm} {c.w[5] * 0.0625 * r['sum_dU2_unit_amp']:.3f}"
                    for nm, r in rep.items() if nm.endswith("hand")), f)

    # the plant's thrust chain at the encounter frequency
    log("\n  plant thrust chain (its own Waterjet.advance at 0.02 s, command "
        "held 0.24 s): thrust state vs cos(we t) command", f)
    import copy
    resp = {}
    for name, e in enc.items():
        prop = copy.deepcopy(m.plant.prop)
        prop.delay.buf = None
        T = 0.5 * prop.t_max
        ts, xs = [], []
        n_tot = int(round(80.0 / m.dt))
        cmd = 0.5 * prop.t_max
        for i in range(n_tot):
            t = i * m.dt
            if i % m.sub == 0:
                cmd = (0.5 + 0.3 * math.cos(e["we"] * t)) * prop.t_max
            T = prop.advance(T, cmd, m.dt)
            ts.append(t + m.dt)
            xs.append(T / prop.t_max)
        ts, xs = np.array(ts), np.array(xs)
        sel = ts > 20.0
        X = np.column_stack([np.ones(sel.sum()), np.cos(e["we"] * ts[sel]),
                             np.sin(e["we"] * ts[sel])])
        b = np.linalg.lstsq(X, xs[sel], rcond=None)[0]
        amp = math.hypot(b[1], b[2]) / 0.3
        lag = math.degrees(math.atan2(b[2], b[1]))
        th_amp = 1 / math.sqrt(1 + (e["we"] * prop.tau) ** 2)
        th_lag = math.degrees(math.atan(e["we"] * prop.tau)
                              + e["we"] * (prop.delay.n * m.dt
                                           + 0.5 * m.sub * m.dt))
        resp[name] = dict(gain=amp, lag_deg=lag, lag_s=math.radians(lag)
                          / e["we"], gain_theory=th_amp, lag_theory=th_lag)
        log(f"    {name:>12}: gain {amp:.3f} (first-order {th_amp:.3f}), "
            f"lag {lag:.1f} deg = {resp[name]['lag_s']:.3f} s (lag + delay + "
            f"half a held step: {th_lag:.1f} deg)", f)
    out["thrust_chain"] = resp
    # surge response of the reduced model (what m0 / hand roll out)
    log("\n  reduced-model surge response to a thrust oscillation (ideal "
        "actuator, linearised about u)", f)
    sr_ = {}
    for name, e in enc.items():
        U = float(name.split("kn")[0]) * KN
        tau_lin = p["m_surge"] / (2 * p["k_drag"] * U)
        du = 0.25 * env["t_max"] / (p["m_surge"]
                                    * math.sqrt(e["we"] ** 2
                                                + tau_lin ** -2))
        dx = du / e["we"]
        sr_[name] = dict(tau_lin=tau_lin, du_amp_A025=du, dx_amp_A025=dx,
                         wave_phase_deg=math.degrees(dx * 2 * math.pi
                                                     / lam_w))
        log(f"    {name:>12}: linear surge time constant {tau_lin:.2f} s; "
            f"amplitude 0.25 t_max -> du {du:.3f} m/s ({du / KN:.2f} kn), "
            f"dx {dx:.3f} m = {sr_[name]['wave_phase_deg']:.2f} deg of wave "
            "phase", f)
    out["surge_resp"] = sr_
    json.dump(out, open(os.path.join(OUT, "planners_consts.json"), "w"), indent=1,
              default=float)


# ------------------------------------------------------------------- score
class FakeRng:
    """Stands in for ConstrainedMPPI.rng[b]: normal() returns the noise
    that turns the controller's current nominal into the designed knots
    (cand = clip(nominal + noise * sigma, floor, 1) = the designed knots
    when they are inside [floor, 1])."""

    def __init__(self, ctrl, b, knots):
        self.ctrl, self.b, self.knots = ctrl, b, np.asarray(knots, float)

    def normal(self, loc, scale, size):
        assert tuple(size) == self.knots.shape, (size, self.knots.shape)
        return (self.knots - self.ctrl.nominal[self.b][None]) / self.ctrl.sigma


def bow_wave_phase(m, we, H, dtc, x_b):
    """Phase theta (rad) of the true bow elevation along a dead-reckoned
    constant-speed track over the horizon, fitted as A cos(we t - theta):
    the crest is at the bow when we t = theta (t from now)."""
    from learn.meta.data5 import sea_eval
    s = m.s
    u, psi = float(s[6]), float(s[5])
    t = (np.arange(H) + 0.5) * dtc
    X = s[0] + u * t * math.cos(psi) + x_b * math.cos(psi)
    Y = s[1] + u * t * math.sin(psi) + x_b * math.sin(psi)
    eta = sea_eval(m.sea, X, Y, m.t + t)
    A = np.column_stack([np.ones(H), np.cos(we * t), np.sin(we * t)])
    b = np.linalg.lstsq(A, eta, rcond=None)[0]
    r2 = 1 - np.sum((eta - A @ b) ** 2) / max(np.sum((eta - eta.mean()) ** 2),
                                              1e-12)
    return math.atan2(b[2], b[1]), math.hypot(b[1], b[2]), r2, eta


def designs(E, we, theta, H, dt_):
    """[(name, knots (7,), mean, amp, lead_deg)]: constants, and sinusoids
    mean + amp cos(we t - theta + lead) (thrust peak `lead` before the bow
    crest), knot least squares, clipped to [0, 1]."""
    out = []
    for lev in (0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
        out.append((f"const{lev:.2f}", np.full(E.shape[1], lev), lev, 0.0,
                    float("nan")))
    t = (np.arange(H) + 0.5) * dt_
    for mean, amp in ((0.6, 0.3), (0.5, 0.5)):
        for lead in range(0, 360, 45):
            s = mean + amp * np.cos(we * t - theta + math.radians(lead))
            kn_, *_ = np.linalg.lstsq(E, s, rcond=None)
            out.append((f"sin{mean:.1f}+-{amp:.1f}@{lead:03d}",
                        np.clip(kn_, 0.0, 1.0), mean, amp, float(lead)))
    return out


def score_constrained(ctrl, k, info, plans, MC, ML, torch):
    """Cost parts per designed plan from one ConstrainedMPPI call."""
    knots = np.stack([p[1] for p in plans])
    ctrl.rng[0] = FakeRng(ctrl, 0, knots)
    cap = {}
    orig = MC.constraint_cost

    def wrap(prog, peaks, bow, yaw, ytr, *a, **kw):
        c, parts = orig(prog, peaks, bow, yaw, ytr, *a, **kw)
        cap.update(c=c.detach().cpu().numpy(),
                   parts={n: v.detach().cpu().numpy() for n, v in
                          parts.items()},
                   peaks=peaks.detach().cpu().numpy(),
                   bow=bow.detach().cpu().numpy(),
                   prog=prog.detach().cpu().numpy())
        return c, parts
    MC.constraint_cost = wrap
    try:
        _, diag = ctrl([0], k, [info])
    finally:
        MC.constraint_cost = orig
    thr = knots @ ctrl.E.T
    jump = ML.W_JUMP * np.sum(np.diff(thr, axis=1) ** 2, 1)
    rows = []
    for i, p in enumerate(plans):
        pk = cap["peaks"][0, i]
        rows.append(dict(
            plan=p[0], mean=p[2], amp=p[3], lead=p[4],
            speed=float(cap["parts"]["speed"][0, i]),
            **{n: float(cap["parts"][n][0, i]) for n in cap["parts"]
               if n.startswith("g_")},
            pen=float(cap["parts"]["pen"][0, i]), jump=float(jump[i]),
            total=float(cap["c"][0, i] + jump[i]),
            peak_mean_g=float(np.nanmean(pk)),
            peak_max_g=float(np.nanmax(pk)) if np.isfinite(pk).any()
            else float("nan"),
            n_nonfinite_peaks=int((~np.isfinite(pk)).sum()),
            bow_min=float(np.nanmin(cap["bow"][0, i])),
            prog_mean=float(np.nanmean(cap["prog"][0, i])),
            peak_by_step=np.nanmean(pk, 0).tolist(),
            bow_by_step=np.nanmean(cap["bow"][0, i], 0).tolist()))
    return rows


def score_hand(m, plans_h, ctrl):
    """control/mpc.py rollout_cost per weighted term for designed thrust
    sequences (rudder column 0, as use_rudder=False plans it)."""
    sr = m.track_state()
    H = ctrl.H
    seq = np.zeros((len(plans_h), 2, H))
    for i, p in enumerate(plans_h):
        cand = np.zeros((1, 2, ctrl.n_knots))
        cand[0, 0] = np.clip(p[1], ctrl.thrust_floor, 1.0)
        seq[i] = ctrl._expand(cand)[0]
    w0 = ctrl.w.copy()
    names = ["acc", "slam", "speed", "track", "heading", "dthrust",
             "drudder"]
    terms = {}
    try:
        for j, nm in enumerate(names):
            ctrl.w = np.zeros_like(w0)
            ctrl.w[j] = 1.0
            terms[nm] = w0[j] * ctrl.rollout_cost(sr, seq, m.t, 0.0)
        ctrl.w = w0
        tot = ctrl.rollout_cost(sr, seq, m.t, 0.0)
    finally:
        ctrl.w = w0
    rows = []
    for i, p in enumerate(plans_h):
        rows.append(dict(plan=p[0], mean=p[2], amp=p[3], lead=p[4],
                         total=float(tot[i]),
                         **{nm: float(terms[nm][i]) for nm in names}))
    assert np.allclose(sum(terms[n] for n in names), tot, rtol=1e-9,
                       atol=1e-9)
    return rows


def summarize(rows, keys):
    """Best constant vs best sinusoid by total; spread over the 8 leads of
    each sinusoid family per key."""
    const = [r for r in rows if r["plan"].startswith("const")]
    sins = [r for r in rows if r["plan"].startswith("sin")]
    bc = min(const, key=lambda r: r["total"])
    bs = min(sins, key=lambda r: r["total"])
    out = dict(best_const=bc["plan"], best_const_total=bc["total"],
               best_sin=bs["plan"], best_sin_total=bs["total"],
               sin_beats_all_const=bs["total"] < bc["total"])
    for fam in sorted({r["plan"].split("@")[0] for r in sins}):
        fr = [r for r in sins if r["plan"].startswith(fam + "@")]
        mean = fr[0]["mean"]
        ref = [r for r in const if abs(r["mean"] - mean) < 1e-9]
        for kk in keys:
            v = np.array([r[kk] for r in fr])
            d = dict(min=float(v.min()), max=float(v.max()),
                     argmin_lead=fr[int(np.argmin(v))]["lead"])
            if ref:
                d["const_same_mean"] = ref[0][kk]
            out[f"{fam}:{kk}"] = d
    return out


def part_score(f, seeds, legs, moments, rows_want):
    import torch
    from learn.meta import mpc_constrained as MC
    from learn.meta import mpc_learned as ML
    from learn.meta import model_preview as MP
    from studies import final_eval as FE
    from studies.sim2real_jet import theta, z_hand
    env = ML.env()
    E, _ = ML.knot_matrices()
    H = ML.H
    turn = FE.phase_turn()
    net = head = ck = None
    if "cat_p_prior" in rows_want:
        from learn.meta.safety_head import load_head
        dev = FE.device()
        spec, why = FE.resolve_row("cat_p_prior")
        if spec is None:
            raise SystemExit(why)
        ck = torch.load(spec["ckpt"], map_location=dev, weights_only=False)
        head, _ = load_head(spec["head"], dev)
        net = MP.load_variant(ck, dev)
        log(f"  cat_p_prior: {spec['ckpt']} (ctx {ck.get('ctx')}), head "
            f"{spec['head']}, device {dev}", f)
    res = []
    for seed in seeds:
        for leg in legs:
            w, floor = theta(z_hand())
            m = FE.make_mission(seed, leg, 120.0, weights=w, floor=floor)
            ML.check_env(m, env)
            geo = FE.geometry(m, env)
            rec = FE.Recorder(m, env)
            m.residual = rec
            dtc = m.dt * m.sub
            lim = MC.Limits()
            jobs = [dict(seed=seed, leg=leg)]
            live = None
            ctl = {}
            n_pl = 22
            ctl["m0"] = MC.ConstrainedMPPI(jobs, [m], env, lim, geo, K=n_pl,
                                           S=FE.OPT["S"], steer="auto")
            if net is not None:
                live = MC.LiveDataW(1, m.n_ctrl + 1, ck["stats"], FE.device(),
                                    ML.mission_consts(m), [m.sea], env)
                live.start(0, m.s, m.t)
                ctx = int(ck.get("ctx", 256))
                ctl["cat_p_prior"] = MC.ConstrainedMPPI(
                    jobs, [m], env, lim, geo, net=net, live=live, head=head,
                    variant="p", nh=ctx - ML.M.HB, K=n_pl, S=FE.OPT["S"],
                    fc=dict(lam=FE.OPT["fc_lam"], hp=FE.OPT["fc_hp"]),
                    steer="auto")
            past = []
            n_win = int(round(lim.win_s / dtc))
            k = 0
            t0 = time.time()
            for km in moments:
                while k < km:
                    thrust, rud = m.mpc_command()
                    u_app = (thrust / m.t_max, rud / m.rud_max)
                    s0 = m.s.copy()
                    n0 = len(rec.S)
                    m.advance(thrust, rud)
                    apk, hmin = FE.step_safety(rec, n0, s0, geo, m.sub)
                    past.append(apk)
                    if live is not None:
                        live.push(0, u_app, m.s, m.t, apk, hmin)
                    k += 1
                U = float(m.s[6])
                we = omega_e(U, leg)
                th, A_b, r2, _ = bow_wave_phase(m, we, H, dtc, geo["x_b"])
                info = dict(past=np.asarray(past[-n_win:]) / G, u_meas=U,
                            r_max=MC.r_max_of(turn, U),
                            bow=FE.bow_stats(rec, lim.win_s), infl=1.0)
                plans = designs(E, we, th, H, dtc)
                entry = dict(seed=seed, leg=leg, k=k, u_kn=U / KN, we=we,
                             Te=2 * math.pi / we, bow_amp=A_b, bow_fit_r2=r2,
                             bow_omega_meas=info["bow"][2],
                             thrust_state=float(m.s[12] / m.t_max),
                             past_a110_g=None, rows={})
                # hand: its own 0.25-s grid
                hc = m.ep.ctrl
                th_h, _, _, _ = bow_wave_phase(m, we, hc.H, hc.dt, geo["x_b"])
                plans_h = designs(E, we, th_h, hc.H, hc.dt)
                if "hand" in rows_want:
                    entry["rows"]["hand"] = score_hand(m, plans_h, hc)
                for nm in ("m0", "cat_p_prior"):
                    if nm in rows_want and nm in ctl:
                        tc = time.perf_counter()
                        entry["rows"][nm] = score_constrained(
                            ctl[nm], k, info, plans,
                            MC, ML, torch)
                        entry[f"t_{nm}"] = time.perf_counter() - tc
                entry["summary"] = {}
                for nm, rr in entry["rows"].items():
                    keys = (["acc", "slam", "speed", "track", "heading",
                             "dthrust", "total"] if nm == "hand" else
                            ["speed", "g_a110", "g_apk", "g_bow", "g_yaw",
                             "g_y", "pen", "jump", "total", "peak_mean_g",
                             "peak_max_g"])
                    entry["summary"][nm] = summarize(rr, keys)
                res.append(entry)
                log(f"\n  seed {seed} leg {leg} k {k}: {U / KN:.1f} kn, "
                    f"Te {2 * math.pi / we:.2f} s (measured bow zero-crossing "
                    f"{2 * math.pi / info['bow'][2]:.2f} s), bow fit r2 "
                    f"{r2:.2f}, thrust state {entry['thrust_state']:.2f} "
                    f"({time.time() - t0:.0f} s)", f)
                for nm, rr in entry["rows"].items():
                    log(f"    [{nm}]", f)
                    cols = (["acc", "slam", "speed", "track", "heading",
                             "dthrust"] if nm == "hand" else
                            ["speed", "g_a110", "g_apk", "g_bow", "g_yaw",
                             "g_y", "jump", "peak_mean_g"])
                    log("      " + f"{'plan':>18}" + "".join(
                        f"{c_:>11}" for c_ in cols) + f"{'total':>12}", f)
                    for r in rr:
                        log("      " + f"{r['plan']:>18}" + "".join(
                            f"{r[c_]:11.4f}" for c_ in cols)
                            + f"{r['total']:12.3f}", f)
                    s_ = entry["summary"][nm]
                    log(f"      best constant {s_['best_const']} "
                        f"{s_['best_const_total']:.3f}; best sinusoid "
                        f"{s_['best_sin']} {s_['best_sin_total']:.3f}; "
                        f"sinusoid beats every constant: "
                        f"{s_['sin_beats_all_const']}", f)
    json.dump(res, open(os.path.join(OUT, "planner_scores.json"), "w"),
              indent=0, default=float)
    # pooled summary
    log("\n  pooled over moments", f)
    for nm in rows_want:
        ents = [e for e in res if nm in e["rows"]]
        if not ents:
            continue
        n_win_ = sum(e["summary"][nm]["sin_beats_all_const"] for e in ents)
        log(f"    {nm}: a sinusoid beat every constant in {n_win_} of "
            f"{len(ents)} moments; best constants "
            f"{[e['summary'][nm]['best_const'] for e in ents]}", f)
        keys = (["acc", "slam"] if nm == "hand" else
                ["g_a110", "g_apk", "g_bow", "peak_mean_g", "peak_max_g"])
        for fam in ("sin0.6+-0.3", "sin0.5+-0.5"):
            for kk in keys:
                spans = [e["summary"][nm][f"{fam}:{kk}"]["max"]
                         - e["summary"][nm][f"{fam}:{kk}"]["min"]
                         for e in ents]
                diffs = [e["summary"][nm][f"{fam}:{kk}"]["min"]
                         - e["summary"][nm][f"{fam}:{kk}"].get(
                             "const_same_mean", float("nan")) for e in ents]
                leads = [e["summary"][nm][f"{fam}:{kk}"]["argmin_lead"]
                         for e in ents]
                log(f"      {fam:>12} {kk:>12}: spread over the 8 leads "
                    f"median {np.median(spans):.4g} max {np.max(spans):.4g}; "
                    f"best lead minus the constant of the same mean median "
                    f"{np.nanmedian(diffs):.4g}; best leads {leads}", f)


# ------------------------------------------------------------------- truth
def gen_states(ctrl):
    return dict(gen=[g.get_state() for g in ctrl.gen],
                gen_u=[g.get_state() for g in ctrl.gen_u],
                gen_w=ctrl.gen_w.get_state(), gen_b=ctrl.gen_b.get_state())


def set_gen_states(ctrl, st):
    for g, s in zip(ctrl.gen, st["gen"]):
        g.set_state(s)
    for g, s in zip(ctrl.gen_u, st["gen_u"]):
        g.set_state(s)
    ctrl.gen_w.set_state(st["gen_w"])
    ctrl.gen_b.set_state(st["gen_b"])


def a110_np(past_g, fut_g, k=2.0, thr=0.3):
    v = np.concatenate([k * np.asarray(past_g), k * np.asarray(fut_g)])
    v = np.sort(v[v >= thr])[::-1]
    if not len(v):
        return 0.0
    return float(v[:max(1, math.ceil(len(v) / 10))].mean())


def plant_truth(m, plans, E, geo, info, lim, turn, H, MC, FE):
    """Open-loop plant futures of the designed plans from the mission's
    current state (deep copies; the heading autopilot sets the nozzle as on
    the boat): per-step APK (g) and HMIN (m), and the planner's constraint
    formula applied to this one realised future."""
    import copy
    out = []
    for p in plans:
        mm = copy.deepcopy(m)
        rr = mm.residual
        thr = np.clip(E @ p[1], 0.0, 1.0)
        c, s_ = math.cos(mm.phi), math.sin(mm.phi)
        x0 = c * mm.s[0] + s_ * mm.s[1]
        apk, hmin, yaw, ytr = [], [], [], []
        for j in range(H):
            T = float(thr[j]) * mm.t_max
            rud = mm.ep._steer(mm.s, T)
            s0 = mm.s.copy()
            n0 = len(rr.S)
            mm.advance(T, rud)
            if not mm.finite or getattr(mm.plant, "capsized", False):
                break
            a, h = FE.step_safety(rr, n0, s0, geo, mm.sub)
            apk.append(a / G)
            hmin.append(h)
            yaw.append(abs(float(mm.s[11])))
            ytr.append(abs(-s_ * mm.s[0] + c * mm.s[1]))
        prog = c * mm.s[0] + s_ * mm.s[1] - x0
        dtc = mm.dt * mm.sub
        g = dict(a110=a110_np(info["past"], apk, lim.k, lim.peak_min)
                 / lim.a110 - 1.0,
                 apk=lim.k * max(apk) / lim.apk - 1.0,
                 bow=(geo["fb"] * lim.bow_frac - min(hmin))
                 / (geo["fb"] * lim.bow_frac),
                 yaw=max(yaw) / info["r_max"] - 1.0,
                 y=max(ytr) / (lim.y_lengths * geo["L"]) - 1.0)
        pen = sum((lim.big + lim.w_g * min(v, lim.g_clip) ** 2) if v > 0
                  else 0.0 for v in g.values())
        speed = H * dtc - prog / geo["u_ref"]
        jump = 0.3 * float(np.sum(np.diff(thr) ** 2))
        out.append(dict(plan=p[0], n=len(apk), apk_by_step=apk,
                        hmin_by_step=hmin, apk_max=max(apk),
                        apk_mean=float(np.mean(apk)), hmin_min=min(hmin),
                        prog=float(prog), speed=speed,
                        **{"g_" + k_: float(v) for k_, v in g.items()},
                        pen=pen, jump=jump, total=speed + pen + jump))
    return out


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok])[0])


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def part_truth(f, seeds, legs, moments, n_rep=3):
    """cat_p_prior's preferences among the designed plans: stable across
    random draws? moved by the preview (hp 24 vs 1, identical draws)? in
    line with the plant's own open-loop futures of the same plans? (m0
    alongside: its proxy quantities against the same truth.)"""
    import torch
    from learn.meta import mpc_constrained as MC
    from learn.meta import mpc_learned as ML
    from learn.meta import model_preview as MP
    from learn.meta.safety_head import load_head
    from studies import final_eval as FE
    from studies.sim2real_jet import theta, z_hand
    env = ML.env()
    E, _ = ML.knot_matrices()
    H = ML.H
    turn = FE.phase_turn()
    dev = FE.device()
    spec, why = FE.resolve_row("cat_p_prior")
    ck = torch.load(spec["ckpt"], map_location=dev, weights_only=False)
    head, _ = load_head(spec["head"], dev)
    net = MP.load_variant(ck, dev)
    res = []
    for seed in seeds:
        for leg in legs:
            w, floor = theta(z_hand())
            m = FE.make_mission(seed, leg, 120.0, weights=w, floor=floor)
            geo = FE.geometry(m, env)
            rec = FE.Recorder(m, env)
            m.residual = rec
            dtc = m.dt * m.sub
            lim = MC.Limits()
            jobs = [dict(seed=seed, leg=leg)]
            live = MC.LiveDataW(1, m.n_ctrl + 1, ck["stats"], dev,
                                ML.mission_consts(m), [m.sea], env)
            live.start(0, m.s, m.t)
            ctx = int(ck.get("ctx", 256))
            n_pl = 22
            cat = MC.ConstrainedMPPI(
                jobs, [m], env, lim, geo, net=net, live=live, head=head,
                variant="p", nh=ctx - ML.M.HB, K=n_pl, S=FE.OPT["S"],
                fc=dict(lam=FE.OPT["fc_lam"], hp=FE.OPT["fc_hp"]),
                steer="auto")
            m0 = MC.ConstrainedMPPI(jobs, [m], env, lim, geo, K=n_pl,
                                    S=FE.OPT["S"], steer="auto")
            past = []
            n_win = int(round(lim.win_s / dtc))
            k = 0
            t0 = time.time()
            for km in moments:
                while k < km:
                    thrust, rud = m.mpc_command()
                    u_app = (thrust / m.t_max, rud / m.rud_max)
                    s0 = m.s.copy()
                    n0 = len(rec.S)
                    m.advance(thrust, rud)
                    apk, hmin = FE.step_safety(rec, n0, s0, geo, m.sub)
                    past.append(apk)
                    live.push(0, u_app, m.s, m.t, apk, hmin)
                    k += 1
                U = float(m.s[6])
                we = omega_e(U, leg)
                th, A_b, r2, eta_b = bow_wave_phase(m, we, H, dtc, geo["x_b"])
                info = dict(past=np.asarray(past[-n_win:]) / G, u_meas=U,
                            r_max=MC.r_max_of(turn, U),
                            bow=FE.bow_stats(rec, lim.win_s), infl=1.0)
                plans = designs(E, we, th, H, dtc)
                truth = plant_truth(m, plans, E, geo, info, lim, turn, H, MC,
                                    FE)
                st0 = gen_states(cat)
                reps = [score_constrained(cat, k, info, plans, MC, ML, torch)
                        for _ in range(n_rep)]
                set_gen_states(cat, st0)
                cat.fc["hp"] = 1
                try:
                    nopv = score_constrained(cat, k, info, plans, MC, ML,
                                             torch)
                finally:
                    cat.fc["hp"] = FE.OPT["fc_hp"]
                r_m0 = score_constrained(m0, k, info, plans, MC, ML, torch)
                ent = dict(seed=seed, leg=leg, k=k, u_kn=U / KN,
                           Te=2 * math.pi / we, bow_amp=A_b, bow_r2=r2,
                           plans=[p[0] for p in plans], truth=truth,
                           cat_reps=reps, cat_nopreview=nopv, m0=r_m0)
                # stability across draws
                tot = np.array([[r["total"] for r in rr] for rr in reps])
                rs = [spearman(tot[i], tot[j]) for i in range(n_rep)
                      for j in range(i + 1, n_rep)]
                best = [plans[int(np.argmin(t))][0] for t in tot]
                ent["rep_spearman_total"] = rs
                ent["rep_best"] = best
                for kk in ("g_apk", "g_bow", "g_a110", "g_y", "peak_mean_g"):
                    v = np.array([[r[kk] for r in rr] for rr in reps])
                    ent[f"rep_spearman_{kk}"] = [
                        spearman(v[i], v[j]) for i in range(n_rep)
                        for j in range(i + 1, n_rep)]
                # preview effect with identical draws (reps[0] vs nopv)
                pk1 = np.array([r["peak_by_step"] for r in reps[0]])
                pk0 = np.array([r["peak_by_step"] for r in nopv])
                bw1 = np.array([r["bow_by_step"] for r in reps[0]])
                bw0 = np.array([r["bow_by_step"] for r in nopv])
                ent["preview_dpeak_by_step"] = np.nanmean(
                    np.abs(pk1 - pk0), 0).tolist()
                ent["preview_dbow_by_step"] = np.nanmean(
                    np.abs(bw1 - bw0), 0).tolist()
                ent["preview_spearman_total"] = spearman(
                    [r["total"] for r in reps[0]], [r["total"] for r in nopv])
                ent["preview_best"] = [plans[int(np.argmin(
                    [r["total"] for r in reps[0]]))][0], plans[int(np.argmin(
                        [r["total"] for r in nopv]))][0]]
                # predicted per-step peaks vs the plant's (per plan, then
                # mean); and the bow height
                tr_apk = [t["apk_by_step"] for t in truth]
                tr_h = [t["hmin_by_step"] for t in truth]
                ent["corr_step_peak_pv"] = float(np.nanmean([
                    pearson(pk1[i], tr_apk[i]) for i in range(len(plans))]))
                ent["corr_step_peak_nopv"] = float(np.nanmean([
                    pearson(pk0[i], tr_apk[i]) for i in range(len(plans))]))
                ent["corr_step_bow_pv"] = float(np.nanmean([
                    pearson(bw1[i], tr_h[i]) for i in range(len(plans))]))
                ent["corr_step_bow_nopv"] = float(np.nanmean([
                    pearson(bw0[i], tr_h[i]) for i in range(len(plans))]))
                pkm0 = np.array([r["peak_by_step"] for r in r_m0])
                bwm0 = np.array([r["bow_by_step"] for r in r_m0])
                ent["corr_step_peak_m0"] = float(np.nanmean([
                    pearson(pkm0[i], tr_apk[i]) for i in range(len(plans))]))
                ent["corr_step_bow_m0"] = float(np.nanmean([
                    pearson(bwm0[i], tr_h[i]) for i in range(len(plans))]))
                # across plans: predicted vs true
                for nm, rr in (("cat", reps[0]), ("cat_nopv", nopv),
                               ("m0", r_m0)):
                    ent[f"xplan_{nm}"] = dict(
                        total=spearman([r["total"] for r in rr],
                                       [t["total"] for t in truth]),
                        g_apk=spearman([r["g_apk"] for r in rr],
                                       [t["g_apk"] for t in truth]),
                        peak_mean=spearman([r["peak_mean_g"] for r in rr],
                                           [t["apk_mean"] for t in truth]),
                        g_bow=spearman([r["g_bow"] for r in rr],
                                       [t["g_bow"] for t in truth]),
                        prog=spearman([r["prog_mean"] for r in rr],
                                      [t["prog"] for t in truth]))
                ib = int(np.argmin([r["total"] for r in reps[0]]))
                it = int(np.argmin([t["total"] for t in truth]))
                ent["cat_pick"] = plans[ib][0]
                ent["cat_pick_true_total"] = truth[ib]["total"]
                ent["true_best"] = plans[it][0]
                ent["true_best_total"] = truth[it]["total"]
                cs = [i for i, p in enumerate(plans) if p[0].startswith("c")]
                ic = min(cs, key=lambda i: truth[i]["total"])
                ent["true_best_const"] = plans[ic][0]
                ent["true_best_const_total"] = truth[ic]["total"]
                res.append(ent)
                log(f"\n  seed {seed} leg {leg} k {k}: {U / KN:.1f} kn, Te "
                    f"{2 * math.pi / we:.2f} s ({time.time() - t0:.0f} s)", f)
                log(f"    cat repeat draws: Spearman of totals "
                    f"{[round(x, 2) for x in rs]}, best plan per draw {best}",
                    f)
                log("    cat repeat-draw Spearman per term: " + ", ".join(
                    f"{kk} {[round(x, 2) for x in ent['rep_spearman_' + kk]]}"
                    for kk in ("g_apk", "g_bow", "g_a110", "g_y",
                               "peak_mean_g")), f)
                log(f"    preview on (hp 24) vs off beyond step 0 (hp 1), "
                    f"same draws: Spearman of totals "
                    f"{ent['preview_spearman_total']:.2f}; best "
                    f"{ent['preview_best']}; mean |d peak| (g) by step "
                    + " ".join(f"{x:.3f}" for x in
                               ent["preview_dpeak_by_step"]), f)
                log("      mean |d bow height| (m) by step " + " ".join(
                    f"{x:.3f}" for x in ent["preview_dbow_by_step"]), f)
                log(f"    per-step correlation predicted vs plant (mean over "
                    f"plans): peak cat {ent['corr_step_peak_pv']:.2f} "
                    f"(no preview {ent['corr_step_peak_nopv']:.2f}, m0 "
                    f"{ent['corr_step_peak_m0']:.2f}); bow height cat "
                    f"{ent['corr_step_bow_pv']:.2f} (no preview "
                    f"{ent['corr_step_bow_nopv']:.2f}, m0 "
                    f"{ent['corr_step_bow_m0']:.2f})", f)
                for nm in ("cat", "cat_nopv", "m0"):
                    log(f"    across the 22 plans, Spearman predicted vs plant "
                        f"[{nm}]: " + ", ".join(
                            f"{kk} {v:.2f}" for kk, v in
                            ent[f"xplan_{nm}"].items()), f)
                log(f"    plant open loop: best plan {ent['true_best']} "
                    f"(total {ent['true_best_total']:.1f}), best constant "
                    f"{ent['true_best_const']} "
                    f"({ent['true_best_const_total']:.1f}); cat's pick "
                    f"{ent['cat_pick']} -> plant total "
                    f"{ent['cat_pick_true_total']:.1f}", f)
                log("      plant per plan: " + "; ".join(
                    f"{t['plan']} apk_max {t['apk_max']:.2f} g, hmin "
                    f"{t['hmin_min']:.2f} m, prog {t['prog']:.1f} m"
                    for t in truth), f)
    json.dump(res, open(os.path.join(OUT, "planner_truth.json"), "w"),
              indent=0, default=float)
    log("\n  pooled", f)
    keys = ["rep_spearman_total", "rep_spearman_g_apk", "rep_spearman_g_bow",
            "rep_spearman_g_a110", "rep_spearman_g_y",
            "rep_spearman_peak_mean_g"]
    for kk in keys:
        v = [x for e in res for x in e[kk]]
        log(f"    {kk}: median {np.nanmedian(v):.2f} (range "
            f"{np.nanmin(v):.2f} .. {np.nanmax(v):.2f})", f)
    same = sum(len(set(e["rep_best"])) == 1 for e in res)
    log(f"    cat picks the same plan in all {n_rep} draws at {same} of "
        f"{len(res)} moments", f)
    for kk in ("corr_step_peak_pv", "corr_step_peak_nopv", "corr_step_peak_m0",
               "corr_step_bow_pv", "corr_step_bow_nopv", "corr_step_bow_m0",
               "preview_spearman_total"):
        v = [e[kk] for e in res]
        log(f"    {kk}: median {np.nanmedian(v):.2f} (range "
            f"{np.nanmin(v):.2f} .. {np.nanmax(v):.2f})", f)
    dp = np.array([e["preview_dpeak_by_step"] for e in res])
    db = np.array([e["preview_dbow_by_step"] for e in res])
    log("    preview |d peak| (g) by step, mean over moments: " + " ".join(
        f"{x:.3f}" for x in np.nanmean(dp, 0)), f)
    log("    preview |d bow| (m) by step, mean over moments: " + " ".join(
        f"{x:.3f}" for x in np.nanmean(db, 0)), f)
    for nm in ("cat", "cat_nopv", "m0"):
        for kk in ("total", "g_apk", "peak_mean", "g_bow", "prog"):
            v = [e[f"xplan_{nm}"][kk] for e in res]
            log(f"    across plans [{nm}] {kk}: median {np.nanmedian(v):.2f} "
                f"(range {np.nanmin(v):.2f} .. {np.nanmax(v):.2f})", f)
    n_sin = sum(e["true_best"].startswith("sin") for e in res)
    log(f"    plant open loop: a sinusoid is the best plan at {n_sin} of "
        f"{len(res)} moments; truth {[e['true_best'] for e in res]}", f)
    gain = [e["true_best_const_total"] - e["cat_pick_true_total"] for e in res]
    log(f"    cat's pick vs the plant-best constant (plant totals, + = cat's "
        f"pick better): {[round(x, 1) for x in gain]}", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="consts")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--legs", default="0,1")
    ap.add_argument("--moments", default="250,262,274,286,298,310")
    ap.add_argument("--rows", default="hand,m0,cat_p_prior")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"planners_{a.part}.log"), "w",
              encoding="utf-8") as f:
        if a.part == "consts":
            part_consts(f)
        elif a.part == "truth":
            part_truth(f, [int(x) for x in a.seeds.split(",")],
                       [int(x) for x in a.legs.split(",")],
                       [int(x) for x in a.moments.split(",")])
        else:
            part_score(f, [int(x) for x in a.seeds.split(",")],
                       [int(x) for x in a.legs.split(",")],
                       [int(x) for x in a.moments.split(",")],
                       a.rows.split(","))


if __name__ == "__main__":
    main()
