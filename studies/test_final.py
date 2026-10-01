#!/usr/bin/env python3
"""
Short unit tests of the final-comparison parts (learn/meta/safety_head.py,
mpc_constrained.py, online_stream.py, studies/final_eval.py); one process,
about a minute or two; data read only: studies/_cache/meta5 (model_w.pt,
model3.pt) and meta6_smoke (train split with APK / HMIN, model_w / model_p).

   1  quantile tools: inv_cdf returns the quantiles at their levels, is
      monotone, has exponential tails; inflate keeps the median; the head's
      quantiles are monotone; pinball is minimal at the true quantile
   2  constraint math: A1/10 (past + horizon, threshold, top 10%), worst-
      alpha means, constraint_cost (no violation = speed only; one
      violation = BIG + W_G g^2; all candidates violating are ordered by
      g^2; non-finite samples are maximal violations)
   3  statistical bow form (mean, RMS), SafeMode in / out, r_max_of
   4  rollout_tap = model_preview.rollout_core_w bit for bit (meta5 'w' with
      the wave hook, meta5 'a' without, meta6_smoke 'p' with preview)
   5  LiveDataW: the measured elevations / W_MID / mid poses equal direct
      sea evaluations, normalised with the checkpoint's w_mu / w_sd
   6  online: LoRA starts as the prior exactly; a learner step changes only
      the flow head, the adapters and the safety head; Monitor inflates on
      misses and recovers to 1 on hits
   7  safety head on meta6_smoke: subset loader, 40 training steps, save /
      load round trip (same quantiles), monotone quantiles, calibration
      table present
   8  step_safety on a hand-built record (0.04 s grid), fallback_events on
      a synthetic run (one 2 g spike -> one peak, A1/10, bow wet entry)
   9  closed-loop code path with the head (meta6_smoke 'w' + the test-7
      head, online mode, 1.5 s) and m0 (1.5 s): finite, records complete

    python -m studies.test_final [1 2 ...]
"""
import argparse
import math
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np   # noqa: E402
import torch   # noqa: E402

from learn.meta import model3 as M   # noqa: E402

META5 = os.path.join(HERE, "_cache", "meta5")
SMOKE6 = os.path.join(HERE, "_cache", "meta6_smoke")
RES = {}


def report(name, ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
    RES.setdefault("fails", [])
    if not ok:
        RES["fails"].append(name)


def dev():
    if "dev" not in RES:
        RES["dev"] = M.device(0.35)
    return RES["dev"]


def live_mission(stats, n_steps=30, seed=0):
    """A LiveDataW fed a short high-fidelity mission (constant thrust, the
    autopilot's nozzle)."""
    from learn.meta import mpc_constrained as MC
    from learn.meta import mpc_learned as ML
    from learn.repro.task import Mission
    m = Mission("high", seed, 0, t_end=20.0, track=0.0)
    env = ML.env()
    live = MC.LiveDataW(1, m.n_ctrl + 1, stats, dev(), ML.mission_consts(m),
                        [m.sea], env)
    live.start(0, m.s, m.t)
    for k in range(n_steps):
        thr = 0.6 * m.t_max
        rud = m.ep._steer(m.s, thr)
        m.advance(thr, rud)
        live.push(0, (thr / m.t_max, rud / m.rud_max), m.s, m.t, 1.0, 0.5)
    return m, live, env


def load_ck(path):
    from learn.meta import model_preview as MP
    ck = torch.load(path, map_location=dev(), weights_only=False)
    return ck, MP.load_variant(ck, dev())


# ------------------------------------------------------------------ 1
def test_1():
    from learn.meta.safety_head import QS, SafetyHead, inflate, inv_cdf, \
        pinball
    q = torch.tensor([[-2.0, -1.0, -0.6, -0.2, 0.1, 0.4, 0.9, 1.3, 2.5]],
                     dtype=torch.float64)
    at = inv_cdf(q.expand(len(QS), -1), torch.tensor(QS,
                                                     dtype=torch.float64))
    u = torch.linspace(1e-5, 1 - 1e-5, 2001, dtype=torch.float64)
    v = inv_cdf(q.expand(len(u), -1), u)
    mono = bool((v[1:] >= v[:-1] - 1e-12).all())
    b_hi = (2.5 - 1.3) / math.log(0.05 / 0.01)
    tail = float(inv_cdf(q, torch.tensor([0.999], dtype=torch.float64)))
    ok_tail = abs(tail - (2.5 + b_hi * math.log(0.01 / 0.001))) < 1e-9
    inf = inflate(q, 2.0)
    ok_inf = abs(float(inf[0, 4] - q[0, 4])) < 1e-12 and abs(
        float(inf[0, 8] - (0.1 + 2 * 2.4))) < 1e-12
    head = SafetyHead(d=8)
    qq = head(torch.randn(50, 8), torch.randn(50, M.C7))
    mono_h = bool((qq[..., 1:] >= qq[..., :-1]).all())
    y = torch.randn(20000, 2, dtype=torch.float64)
    true_q = torch.tensor(np.quantile(y.numpy(), QS, axis=0).T)
    l0 = pinball(true_q.expand(20000, 2, -1), y, QS).mean()
    l1 = pinball((true_q + 0.1).expand(20000, 2, -1), y, QS).mean()
    ok = (torch.allclose(at, q[0]) and mono and ok_tail and ok_inf and mono_h
          and float(l0) < float(l1))
    report("1 quantile tools", ok, f"levels exact {torch.allclose(at, q[0])}, "
           f"monotone {mono}, exp tail {ok_tail}, inflate {ok_inf}, head "
           f"monotone {mono_h}, pinball {float(l0):.4f} < {float(l1):.4f}")


# ------------------------------------------------------------------ 2
def test_2():
    from learn.meta import mpc_constrained as MC
    lim = MC.Limits()
    past = torch.tensor([0.2, 1.0, 2.0], dtype=torch.float64)
    hor = torch.tensor([[0.1, 3.0, 0.5]], dtype=torch.float64)
    # values >= 0.3: 1.0, 2.0, 3.0, 0.5 -> n 4, top ceil(0.4) = 1 -> 3.0
    v1 = float(MC.a110(hor, past, 0.3)[0])
    many = torch.linspace(0.31, 2.0, 25, dtype=torch.float64)[None]
    v2 = float(MC.a110(many, torch.zeros(0, dtype=torch.float64), 0.3)[0])
    srt = np.sort(np.linspace(0.31, 2.0, 25))[::-1]
    ok_a = abs(v1 - 3.0) < 1e-12 and abs(v2 - srt[:3].mean()) < 1e-12
    x = torch.arange(20, dtype=torch.float64)[None]
    ok_c = (float(MC.cvar_hi(x, 0.1)) == 18.5
            and float(MC.cvar_lo(x, 0.1)) == 0.5)
    K, S, Hh = 3, 4, 24
    fb, L, u_ref, hd = 0.885, 5.8, 12.0, Hh * 0.24
    prog = torch.full((1, K, S), 100.0, dtype=torch.float64)
    prog[0, 1] = 120.0
    prog[0, 2] = 140.0
    peaks = torch.full((1, K, S, Hh), 0.1, dtype=torch.float64)
    bow = torch.full((1, K, S, Hh), 0.8, dtype=torch.float64)
    yaw = torch.zeros(1, K, S, Hh, dtype=torch.float64)
    ytr = torch.zeros(1, K, S, Hh, dtype=torch.float64)
    pa = [torch.zeros(0, dtype=torch.float64)]
    c, parts = MC.constraint_cost(prog, peaks, bow, yaw, ytr, pa, [0.5],
                                  lim, fb, L, u_ref, hd)
    ok_free = torch.allclose(c, parts["speed"]) and bool(
        (parts["pen"] == 0).all()) and int(torch.argmin(c[0])) == 2
    # candidate 2 (fastest) has one 4 g step: k x 4 = 8 g breaks the single
    # peak (7 g) and, as its only peak >= 0.3 g, the A1/10 (3 g)
    pk2 = peaks.clone()
    pk2[0, 2, 0, 5] = 4.0
    c2, p2 = MC.constraint_cost(prog, pk2, bow, yaw, ytr, pa, [0.5], lim,
                                fb, L, u_ref, hd)
    g1, g2 = 8.0 / 7.0 - 1, 8.0 / 3.0 - 1
    want = 2 * lim.big + lim.w_g * (g1 * g1 + g2 * g2)
    ok_one = (abs(float(p2["pen"][0, 2]) - want) < 1e-9
              and int(torch.argmin(c2[0])) == 1
              and float(p2["pen"][0, :2].abs().sum()) == 0.0)
    # all violate the bow limit; the least violation must win despite speed
    bw = bow.clone()
    bw[0, 0] = 0.20
    bw[0, 1] = 0.15
    bw[0, 2] = 0.10
    c3, p3 = MC.constraint_cost(prog, peaks, bw, yaw, ytr, pa, [0.5], lim,
                                fb, L, u_ref, hd)
    ok_all = int(torch.argmin(c3[0])) == 0 and bool((p3["pen"] > 0).all())
    # a non-finite sample is a maximal (clipped) violation
    bw4 = bow.clone()
    bw4[0, 1, 2, 3] = float("nan")
    c4, p4 = MC.constraint_cost(prog, peaks, bw4, yaw, ytr, pa, [0.5], lim,
                                fb, L, u_ref, hd)
    ok_nan = (float(p4["g_bow"][0, 1]) == lim.g_clip
              and bool(torch.isfinite(c4).all()))
    ok = ok_a and ok_c and ok_free and ok_one and ok_all and ok_nan
    report("2 constraint math", ok, f"A1/10 {v1:.2f} / {v2:.3f}, cvar "
           f"{ok_c}, free {ok_free}, one violation {ok_one}, all violate -> "
           f"least g {ok_all}, NaN {ok_nan}")


# ------------------------------------------------------------------ 3
def test_3():
    from learn.meta import mpc_constrained as MC
    g = torch.Generator().manual_seed(0)
    eta = MC.bow_statistical(0.3, 0.25, 3.0, 1, 40000, 24, 0.24, g)
    mu, sd = float(eta.mean()), float(eta.std())
    ok_b = abs(mu - 0.3) < 0.01 and abs(sd - 0.25) < 0.01
    sm = MC.SafeMode(0.24)
    th = []
    for roll in [0.0, math.radians(25)] + [math.radians(5)] * 9:
        t, r, on = sm(roll, 100.0, 0.2)
        th.append((t, r, on))
    ok_s = (th[0] == (100.0, 0.2, False) and th[1] == (50.0, 0.0, True)
            and all(x[2] for x in th[2:9]) and not th[10][2])
    tab = dict(u=[10.0, 5.0, 15.0], r=[0.4, 0.3, 0.5])
    ok_r = (abs(MC.r_max_of(tab, 7.5) - 0.35) < 1e-12
            and MC.r_max_of(tab, 30.0) == 0.5)
    report("3 bow form / safe mode / r_max", ok_b and ok_s and ok_r,
           f"eta mean {mu:.3f} (0.3) rms {sd:.3f} (0.25); safe mode "
           f"{[x[2] for x in th]}; r_max {ok_r}")


# ------------------------------------------------------------------ 4
def test_4():
    from learn.meta import model_preview as MP
    from learn.meta import mpc_constrained as MC
    out = []
    ok = True
    for path, var in ((os.path.join(META5, "model_w.pt"), "w"),
                      (os.path.join(META5, "model3.pt"), "a"),
                      (os.path.join(SMOKE6, "model_p.pt"), "p")):
        ck, net = load_ck(path)
        m, live, env = live_mission(ck["stats"], 30)
        ctx = int(ck.get("ctx", M.W_CTX))
        nh = ctx - M.HB
        k = 30
        P, S = 3, 2
        g = torch.Generator().manual_seed(5)
        base = torch.randn((1, 1, S, M.HB, M.C7), generator=g).expand(
            1, P, S, M.HB, M.C7)
        plans = torch.rand((1, P, M.HB, 2), generator=g,
                           dtype=torch.float64)
        plans[..., 1] = 0.0
        waves = []
        for _ in range(2):
            if var == "a":
                waves.append(None)
            else:
                waves.append(MP.WaveRoll(live, [0], [k], 0.1,
                                         8 if var == "p" else 0, 0.01,
                                         torch.Generator().manual_seed(9)))
        e1, s1, a1 = MP.rollout_core_w(net, live, [0], [k], plans,
                                       m.s[None], env, S, base=base,
                                       wave=waves[0], nh=nh)
        e2, s2, a2, h2 = MC.rollout_tap(net, live, [0], [k], plans,
                                        m.s[None], env, S, base,
                                        wave=waves[1], nh=nh)
        same = (torch.equal(e1, e2) and torch.equal(s1, s2)
                and torch.equal(a1, a2))
        ok = ok and same and tuple(h2.shape) == (1, P, S, M.HB, net.d)
        out.append(f"{var}: identical {same}, h {tuple(h2.shape)}")
    report("4 rollout_tap = rollout_core_w", ok, "; ".join(out))


# ------------------------------------------------------------------ 5
def test_5():
    from learn.meta.data5 import sea_eval
    from learn.meta.episode5 import mid_pose, station_xy
    ck, _ = load_ck(os.path.join(META5, "model_w.pt"))
    m, live, env = live_mission(ck["stats"], 12)
    st = ck["stats"]
    worst = 0.0
    for k in (0, 5, 12):
        xs = live.XS[0, k]
        t = k * live.dtc
        X, Y = station_xy(np.asarray(xs[0]), np.asarray(xs[1]),
                          np.asarray(xs[5]), env["x_st"], env["y_off"])
        w = sea_eval(m.sea, X, Y, t).reshape(15)
        xm, ym, pm = mid_pose(xs, "plant14", live.h)
        Xm, Ym = station_xy(np.asarray(xm), np.asarray(ym), np.asarray(pm),
                            env["x_st"], env["y_off"])
        wm = sea_eval(m.sea, Xm, Ym, t + live.h).reshape(15)
        worst = max(worst,
                    float(np.abs(live.Wm[0, k].cpu().numpy()
                                 - (w - st["w_mu"]) / st["w_sd"]).max()),
                    float(np.abs(live.Wmid[0, k].cpu().numpy()
                                 - (wm - st["w_mu"]) / st["w_sd"]).max()),
                    float(np.abs(live.PM[0, k].cpu().numpy()
                                 - np.array([xm, ym, pm])).max()))
    ok = worst < 1e-4 and abs(float(m.t) - 12 * live.dtc) < 1e-9
    report("5 LiveDataW wave columns", ok, f"max |diff| {worst:.1e}")


# ------------------------------------------------------------------ 6
def test_6():
    from learn.meta import online_stream as OS
    from learn.meta.safety_head import SafetyHead
    ck, net = load_ck(os.path.join(META5, "model_w.pt"))
    m, live, env = live_mission(ck["stats"], 40)
    ctx = int(ck.get("ctx", M.W_CTX))
    tok, _, _ = OS._tokens_measured(live, 0, 0, ctx, "w",
                                    dict(lam=0.0, hp=0, msd=0.01),
                                    torch.Generator().manual_seed(1))
    with torch.no_grad():
        h0 = net.encode(tok)
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    head = SafetyHead(d=net.d).to(dev())
    hb = {k: v.detach().clone() for k, v in head.state_dict().items()}
    lr = OS.OnlineLearner(net, live, 0, ctx, "w", head=head, seed=0)
    with torch.no_grad():
        h1 = net.encode(tok)
    same0 = torch.allclose(h0, h1, atol=1e-6)
    for k in range(30, 40):
        lr.step(k)
    after = net.state_dict()
    changed, wrong = [], []
    for k, v in before.items():
        k2 = k if k in after else k.replace(".weight", ".parametrizations."
                                            "weight.original")
        if not torch.equal(v, after[k2]):
            (changed if k.startswith("head.") else wrong).append(k)
    lora_moved = any(float(p.abs().sum()) > 0 for nm, p in
                     net.named_parameters() if nm.endswith(".B"))
    head_moved = any(not torch.equal(v, head.state_dict()[k])
                     for k, v in hb.items() if not k.startswith("y_"))
    mon = OS.Monitor(net, live, 0, ctx, "w")
    pred = dict(qe=np.zeros((5, M.C7)))
    pred["qe"][0], pred["qe"][1], pred["qe"][3], pred["qe"][4] = \
        -1e-9, -1e-9, 1e-9, 1e-9
    for _ in range(40):
        mon.update(pred, 35)                  # the truth is outside
    up = mon.infl
    pred_ok = dict(qe=np.zeros((5, M.C7)))
    pred_ok["qe"][0], pred_ok["qe"][1] = -1e9, -1e9
    pred_ok["qe"][3], pred_ok["qe"][4] = 1e9, 1e9
    for _ in range(400):
        mon.update(pred_ok, 35)
    p1 = mon.predict(39, (0.6, 0.0))
    ok = (same0 and changed and not wrong and lora_moved and head_moved
          and up > 1.3 and mon.infl == 1.0 and p1["qe"].shape == (5, M.C7))
    report("6 online learner / monitor", ok,
           f"LoRA no-op at start {same0}; changed head tensors "
           f"{len(changed)}, other trunk tensors changed {wrong[:3]}; LoRA "
           f"moved {lora_moved}; safety head moved {head_moved}; inflation "
           f"after misses {up:.2f}, after hits {mon.infl:.2f}; "
           f"learner {lr.summary()}")


# ------------------------------------------------------------------ 7
def test_7():
    from learn.meta import safety_head as SH
    ck, net = load_ck(os.path.join(SMOKE6, "model_w.pt"))
    L = int(ck.get("ctx", M.W_CTX))
    D = SH.load_split(SMOKE6, "train", dev(), ck["stats"], n_max=24)
    head, info = SH.train_head(net, D, "w", L, steps=40, batch=8, check=20,
                               log=lambda *a: None)
    tmp = os.path.join(tempfile.gettempdir(), "test_final_safety_w.pt")
    SH.save_head(head, info, tmp)
    h2, _ = SH.load_head(tmp, dev())
    hh = torch.randn(30, net.d, device=dev())
    ee = torch.randn(30, M.C7, device=dev())
    with torch.no_grad():
        q1, q2 = head(hh, ee), h2(hh, ee)
    mono = bool((q1[..., 1:] >= q1[..., :-1]).all())
    cal = info["held_out"]["cal"]
    ok = (torch.equal(q1, q2) and mono and len(cal) == 2
          and len(cal[0]) == len(SH.QS) and D.Y.shape[-1] == 2)
    RES["head7"] = (tmp, ck, net)
    report("7 safety head (meta6_smoke)", ok,
           f"{D.n} episodes, ctx {L}, round trip {torch.equal(q1, q2)}, "
           f"monotone {mono}, held-out cov90 "
           f"{np.round(info['held_out']['cov90'], 2)}, APK > q99 "
           f"{info['held_out']['apk_above_q99']:.3f}")


# ------------------------------------------------------------------ 8
def test_8():
    from types import SimpleNamespace
    from studies import final_eval as FE
    geo = dict(fb=0.885, x_b=3.16, s_p=-1.0, L=5.8, u_ref=12.0)
    sub, dt = 12, 0.02
    rec = SimpleNamespace(m=SimpleNamespace(dt=dt), S=[], EB=[])
    s0 = np.zeros(14)
    w = np.cumsum(np.r_[np.zeros(4), 0.4, np.zeros(7)])   # one jump of 0.4
    for i in range(sub):
        s = np.zeros(14)
        s[8] = w[i]
        s[2] = 0.01 * i
        rec.S.append(s)
        rec.EB.append(np.array([0.1, 0.3 if i == 7 else 0.2, 0.0]))
    apk, hmin = FE.step_safety(rec, 0, s0, geo, sub)
    ok_s = abs(apk - 0.4 / 0.04) < 1e-9 and abs(
        hmin - min(0.885 + 0.01 * (i - 1) - (0.3 if i - 1 == 7 else 0.2)
                   for i in range(2, 13, 2))) < 1e-12
    n = 3000
    t = np.arange(1, n + 1) * dt
    a = np.zeros(n)
    a[1500:1505] = 2.0 * 9.81 / 2.0           # 2 g at k = 2 after filtering
    S = np.zeros((n, 14))
    S[:, 6] = 12.0
    bow = np.full(n, 0.5)
    bow[2000:2010] = -0.1
    tr = dict(dt=dt, t=t, s=S, a_cg=a, a_bow=a, n_sub=np.ones(n),
              eta_bow=np.zeros((n, 3)), bow_h=bow, intake=np.full(n, 0.2),
              thr_cmd=np.zeros(n), noz_cmd=np.zeros(n),
              safe=np.zeros(n, bool), finite=True, t_end=float(t[-1]),
              L=5.8, B=2.0, u_ref=12.0, u_max=23.5, freeboard=0.885,
              x_bow=3.16, sign_pitch=-1.0, noz_max=0.436, steer_sign=-1.0,
              phi=0.0, wave_dir=math.pi, wave_c=7.8, n_sub_max=64, g=9.81)
    e = FE.fallback_events(tr, 2.0)
    ok_e = (abs(e["a110"] - e["peak_max"]) < 1e-12 and e["peak_max"] > 1.0
            and e["n_ge7"] == 0 and e["bow_wet"] == 1
            and e["bow_burial"] == 0 and e["numeric"] == 0
            and e["y_within_2L"] == 1.0 and e["intake_out"] == 0)
    report("8 step_safety / fallback events", ok_s and ok_e,
           f"APK {apk:.2f} m/s2, HMIN {hmin:.3f} m; events peak "
           f"{e['peak_max']:.2f} g, A1/10 {e['a110']:.2f}, bow wet "
           f"{e['bow_wet']}, burial {e['bow_burial']}")


# ------------------------------------------------------------------ 9
def test_9():
    from studies import final_eval as FE
    from learn.meta.safety_head import load_head
    if "head7" not in RES:
        test_7()
    tmp, ck, _ = RES["head7"]
    head, _ = load_head(tmp, dev())
    FE.OPT.update(tag="test_final", S=4, K=32)
    turn = dict(u=[5.0, 15.0], r=[0.3, 0.5])
    spec = dict(name="gen_w_online", kind="net", fam="gen", variant="w",
                mode="online", cache=SMOKE6, ckpt=None, head=tmp)
    t0 = time.time()
    r = FE.run_mission(spec, 0, 1, 1.5, turn, {spec["name"]: (ck, head)})
    t1 = time.time()
    r0 = FE.run_mission(dict(name="m0", kind="m0"), 0, 0, 1.5, turn, {})
    keys = ("kn", "events", "t_step", "monitor", "learner", "plan")
    ok = (r["finite"] and r0["finite"] and all(k in r for k in keys)
          and r["plan"]["kind"] == "head" and r0["plan"]["kind"] == "proxy"
          and set(r["events"]) == {"1.5", "2.0", "2.5"})
    report("9 closed-loop code path", ok,
           f"head/online: {r['n_steps']} steps, {r['kn']:.1f} kn, "
           f"{r['t_step']:.2f} s/step, monitor n {r['monitor']['n']}, "
           f"learner n {r['learner']['n']} ({t1 - t0:.0f} s); m0: "
           f"{r0['n_steps']} steps, {r0['kn']:.1f} kn, "
           f"{r0['t_step']:.3f} s/step; events from "
           f"{r['events']['2.0']['source']}")


# ------------------------------------------------------------------ 10
def test_10():
    """scratch row: untrained start, every weight learns, no pull; carry:
    a two-mission chain through phase_eval, the second mission starting
    from the first one's learned state, and the saved state round trip."""
    from studies import final_eval as FE
    from learn.meta.safety_head import load_head
    if "head7" not in RES:
        test_7()
    tmp, ck, _ = RES["head7"]
    head, _ = load_head(tmp, dev())
    FE.OPT.update(tag="test_final_10", S=4, K=32)
    turn = dict(u=[5.0, 15.0], r=[0.3, 0.5])
    spec = dict(name="gen_w_scratch", kind="net", fam="gen", variant="w",
                mode="scratch", cache=SMOKE6, ckpt=None, head=tmp)
    nets = {spec["name"]: (ck, head)}
    n1, h1 = FE.build_net(spec, ck, head, dev(), 0, 0)
    n2, _ = FE.build_net(spec, ck, head, dev(), 0, 0)
    n3, _ = FE.build_net(spec, ck, head, dev(), 1, 0)
    sd1, sd2, sd3 = n1.state_dict(), n2.state_dict(), n3.state_dict()
    seeded = all(torch.equal(sd1[k], sd2[k]) for k in sd1)
    fresh = not all(torch.equal(sd1[k], ck["net"][k].to(sd1[k].device))
                    for k in sd1) and not all(torch.equal(sd1[k], sd3[k])
                                              for k in sd1)
    w1, w0 = next(h1.net.parameters()), next(head.net.parameters())
    scale = torch.equal(h1.y_sd, head.y_sd) and not torch.equal(w1, w0)
    r = FE.run_mission(spec, 0, 0, 1.5, turn, nets)
    ok_s = (r["finite"] and r["learner"]["pull_last60"] == 0.0
            and r["learner"]["dev_sq"] > 0)
    # carry chain through phase_eval (results under the test tag)
    import shutil
    shutil.rmtree(FE.out_dir(), ignore_errors=True)
    cspec = dict(spec, name="gen_w_online_carry", mode="online", carry=True)
    orig = FE.resolve_row
    FE.resolve_row = lambda nm: (cspec, None)
    orig_turn = FE.phase_turn
    FE.phase_turn = lambda force=False: turn
    orig_load = torch.load
    try:
        import learn.meta.safety_head as SH
        orig_lh = SH.load_head
        SH.load_head = lambda p, d: (head, {})
        cspec["ckpt"] = os.path.join(FE.out_dir(), "ck.pt")
        torch.save(ck, cspec["ckpt"])
        FE.phase_eval([cspec["name"]], [0], [0, 1], 1.5)
        R = FE.load_results()["rows"][cspec["name"]]
        c = orig_load(os.path.join(FE.out_dir(),
                                   f"carry_{cspec['name']}.pt"),
                      map_location=dev(), weights_only=False)
    finally:
        FE.resolve_row, FE.phase_turn = orig, orig_turn
        SH.load_head = orig_lh
    nb = [R[k]["net"]["n_before"] for k in ("s0_l0", "s0_l1")]
    ok_c = (nb[0] == 0 and nb[1] == R["s0_l0"]["learner"]["n"] > 0
            and c["last"] == "s0_l1" and c["state"]["n"]
            == R["s0_l1"]["learner"]["n"] and "_carry_state" not in
            R["s0_l1"])
    ok = seeded and fresh and scale and ok_s and ok_c
    report("10 scratch row / carry chain", ok,
           f"seeded {seeded}, untrained {fresh}, head scale only {scale}; "
           f"scratch: no pull {r['learner']['pull_last60'] == 0.0}, moved "
           f"{r['learner']['dev_sq']:.2e}; carry n_before {nb}, saved "
           f"after {c['last']} (n {c['state']['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tests", nargs="*", type=int)
    a = ap.parse_args()
    which = a.tests or list(range(1, 11))
    t0 = time.time()
    for i in which:
        try:
            globals()[f"test_{i}"]()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            report(f"{i}", False, repr(ex))
    print(f"{len(which) - len(RES.get('fails', []))}/{len(which)} passed in "
          f"{time.time() - t0:.0f} s; failed {RES.get('fails', [])}")


if __name__ == "__main__":
    main()
