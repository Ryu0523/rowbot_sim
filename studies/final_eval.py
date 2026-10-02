#!/usr/bin/env python3
"""
The final closed-loop comparison on the high-fidelity planing target
(ROADMAP_2026-09-30 sections 9-10, "judge by the final result"): mean
along-track speed under the safety constraints, and safety events per
hour.

Missions: those of studies/mpc_compare.py -- learn/repro/task.py in the
turned-track geometry (waves from pi, legs 0 = head seas and 1 = 45 deg
off the bow), sea state 3 (Hs 1.0 m, Tp 5 s), task.EVAL_SEEDS x both legs,
'120 s' each (480 control steps of 0.24 s = 115.2 s: Mission counts steps
with the nominal 0.25 s), the same missions for every row (paired).

Rows (each skipped with a message when its checkpoint is missing):
  hand              control/mpc.py with the hand weights (z_hand), as the
                    repro baselines run it (its own internal impact price)
  m0                the constrained MPPI with the e = 0 model only
  <fam>_<v>_prior   the constrained MPPI with the trained net of family
                    fam (proj = meta5 old projection, gen = meta6 general
                    D9, cat = meta7 catalogue D10), variant v (a: no waves,
                    w: elevations at t, p: + preview), prior weights;
                    mismatch trigger on (spread inflation)
  <fam>_<v>_online  the same plus streaming online learning (head + LoRA,
                    pulled to the prior), learn/meta/online_stream.py
  <fam>_<v>_scratch the "no prior" row: the same network shape, MPC,
                    trigger and online learning, but starting from untrained
                    weights (seeded per mission) with every weight learning
                    and no pull; it shares only the checkpoint's input /
                    output normalisation and, for gen / cat, a fresh safety
                    head with the trained head's output scale. scratch -
                    online = what the prior is worth in closed loop
  <fam>_<v>_bayes   the prior net plus learn/meta/bayes_adapter: a Bayesian
                    low-rank adapter on the trunk output whose posterior
                    N(mu, P) narrows by one extended-Kalman step per
                    observation (each used once); every sampled future
                    keeps its own adapter draw for the whole horizon
  ..._online_carry / ..._scratch_carry / ..._bayes_carry
                    one boat over consecutive missions (in the order of
                    --missions x --legs): what was learned (weights,
                    optimiser, data count; not the inflation) carries over
                    instead of starting each mission afresh. The state after
                    every mission is saved (carry_<row>.pt) so a stopped
                    run resumes where it was
Steering (--steer): auto (default) = the heading autopilot sets the nozzle
in the rollouts and on the boat, the MPPI plans the thrust; free = the MPPI
plans the nozzle too (learn/meta/mpc_constrained.py, steer='free'), for
every row but hand (its own controller). --y-lengths sets the planner's
cross-track limit (boat lengths, default 2). A non-default setting appends
'steerfree' / 'y<n>' to the tag, so its results never mix with another's.
proj rows have no safety head (meta5 has no APK / HMIN arrays): their
acceleration constraint uses the predicted step-mean heave acceleration
and their bow constraint the bow height from the predicted heave and pitch
(+ the forecast for p, else the statistical form). gen / cat rows use the
safety head <cache>/safety_<v>.pt (phase heads).

Every row: the same measured-roll safe mode (|roll| > 20 deg: throttle
halved, nozzle centred; out after 2 s below 10 deg), the same plant, the
same evaluation. The plant (--plant, default gz when it imports) is
sim/planing_vessel_gz.PlaningVesselGZ (large-angle righting arm, can
capsize), built with from_plant(Mission's plant, record=True); a mission
stops at plant.capsized. The planner works at k = 2.0; the events are
studies/safety_events.safety_events(plant.trajectory(), k=2.0,
k_report=(1.5, 2.5), track=phi), kept whole per mission (events_raw) and
flattened per k for the report (with --plant default, which records
nothing, this file's fallback counter; each record says which).

Phases (results in studies/_cache/final_eval[/<tag>], written atomically):
  turn    the steady full-nozzle turning rate vs speed on the plant (calm
          water, six throttle settings; cached per plant): r_max(u)
  heads   safety heads for the gen / cat nets that exist (skips the rest)
  eval    the rows x missions (resumable: finished missions are skipped)
  report  the table

    python -m studies.final_eval --phase turn,heads,eval,report \\
        --rows hand,m0,proj_p_prior,proj_p_online,gen_p_prior,gen_p_online,\\
gen_p_scratch,cat_p_prior,cat_p_online,cat_p_scratch [--plant default|gz] \\
        [--missions 0-7] [--T 120]
    python -m studies.final_eval --tag carry --phase eval,report \\
        --rows cat_p_online_carry,cat_p_scratch_carry
"""
import argparse
import glob
import json
import math
import os
import pickle
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np   # noqa: E402

from learn.repro import task   # noqa: E402

KN = 0.514444
FAMILIES = dict(proj="meta5", gen="meta6", cat="meta7")
K_EVAL = (1.5, 2.0, 2.5)
OPT = dict(tag="", plant="gz", T=task.T_EVAL, S=8, K=128, device="auto",
           fc_lam=0.1, fc_hp=24, trigger=True, min_gb=3.0, steer="auto",
           y_lengths=2.0)


# ------------------------------------------------------------ plumbing
def out_dir():
    d = os.path.join(HERE, "_cache", "final_eval", OPT["tag"]) \
        if OPT["tag"] else os.path.join(HERE, "_cache", "final_eval")
    os.makedirs(d, exist_ok=True)
    return d


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(out_dir(), "run.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def atomic_pickle(obj, path):
    pickle.dump(obj, open(path + ".tmp", "wb"))
    os.replace(path + ".tmp", path)


def atomic_json(obj, path):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def free_gb():
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


def wait_ram(need=None, poll=60, max_wait=1800):
    """At least `need` GB free (poll every 60 s up to 30 min); False if
    never."""
    need = OPT["min_gb"] if need is None else need
    t0 = time.time()
    while True:
        g = free_gb()
        if g >= need:
            return True
        if time.time() - t0 >= max_wait:
            log(f"  only {g:.1f} GB free after {max_wait / 60:.0f} min "
                f"(need {need} GB): skipping")
            return False
        log(f"  {g:.1f} GB free, waiting for {need} GB")
        time.sleep(poll)


def gpu_free_gb():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().split("\n")[0]) / 1024.0
    except Exception:
        return 0.0


_DEV = {}


def device():
    if "dev" not in _DEV:
        import torch
        want = OPT["device"]
        if want == "cpu" or not torch.cuda.is_available():
            _DEV["dev"] = torch.device("cpu")
        elif want == "cuda" or gpu_free_gb() >= 2.0:
            torch.cuda.set_per_process_memory_fraction(
                OPT.get("gpu_frac", 0.35))
            _DEV["dev"] = torch.device("cuda")
        else:
            log("  GPU has < 2 GB free: CPU")
            _DEV["dev"] = torch.device("cpu")
    return _DEV["dev"]


# ------------------------------------------------------------ plant
def plant_factory(name):
    """f(sea, dt) -> the target plant. 'default': sim/planing_vessel.py
    through config.plant_for (linear roll: cannot capsize); 'gz':
    sim/planing_vessel_gz.plant_for (the large-angle righting arm)."""
    from sim import config
    c = task.ctx()
    if name == "default":
        return lambda sea, dt: config.plant_for(c["db"], sea, c["h"], dt=dt,
                                                fidelity="high")
    if name != "gz":
        raise SystemExit(f"--plant {name!r}: default or gz")
    try:
        from sim import planing_vessel_gz as gz
    except ImportError as ex:
        raise SystemExit(f"--plant gz: sim/planing_vessel_gz.py not "
                         f"importable ({ex})")
    return lambda sea, dt: gz.plant_for(c["db"], sea, c["h"], dt=dt)


def make_mission(seed, leg, T, weights=None, floor=0.0):
    m = task.Mission("high", seed, leg, t_end=T, weights=weights, floor=floor,
                     track=task.LEGS[leg])
    if OPT["plant"] == "gz":
        # the GZ twin of the plant the Mission built, recording for
        # studies/safety_events.py (as its own mission_gz does)
        from sim.planing_vessel_gz import from_plant
        pl = from_plant(m.plant, record=True)
        for a in ("L",):
            assert abs(float(getattr(pl, a)) - m.L) < 1e-9, a
        assert abs(float(pl.prop.t_max) - m.t_max) < 1e-6
        assert abs(float(pl.rudder.max) - m.rud_max) < 1e-9
        m.plant = m.ep.plant = pl
        m.s = pl.initial_state(0.8 * m.u_ref)
        m.s[5] = m.phi
    return m


# ------------------------------------------------------------ turning
def turn_path():
    return os.path.join(HERE, "_cache", "final_eval",
                        f"turn_{OPT['plant']}.json")


def phase_turn(force=False):
    """r_max(u): steady yaw rate with the nozzle hard over, calm water, six
    throttle settings (20 s straight, 25 s turning, means over the last
    8 s). Cached per plant."""
    path = turn_path()
    if os.path.exists(path) and not force:
        return json.load(open(path))
    from sim.test_vessel import Monochromatic
    c = task.ctx()
    m = task.Mission("high", 0, 0, t_end=1.0, track=0.0)
    dt = m.dt
    plant = plant_factory(OPT["plant"])(Monochromatic(1.0, 0.0), dt)
    t_max, rud = float(plant.prop.t_max), float(plant.rudder.max)
    U, R, Fs = [], [], []
    t0 = time.time()
    for f in (0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
        s = plant.initial_state(0.8 * float(c["p"]["u_design"]))
        t, us, rs = 0.0, [], []
        for i in range(int(round(45.0 / dt))):
            r_cmd = rud if t >= 20.0 else 0.0
            s = plant.step(s, t, f * t_max, r_cmd, dt)
            t += dt
            if t >= 37.0:
                us.append(s[6])
                rs.append(abs(s[11]))
        if not np.all(np.isfinite(s)):
            log(f"  turn: throttle {f} went non-finite, dropped")
            continue
        U.append(float(np.mean(us)))
        R.append(float(np.mean(rs)))
        Fs.append(f)
    tab = dict(u=U, r=R, throttle=Fs, plant=OPT["plant"], dt=dt,
               note="steady full-nozzle turn in calm water")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    atomic_json(tab, path)
    log(f"  [turn] {OPT['plant']}: " + ", ".join(
        f"{u / KN:.1f} kn -> {math.degrees(r):.1f} deg/s"
        for u, r in zip(U, R)) + f" ({time.time() - t0:.0f} s)")
    return tab


# ------------------------------------------------------------ rows
def ckpt_path(fam, v):
    cache = os.path.join(HERE, "_cache", FAMILIES[fam])
    return cache, os.path.join(cache, "model3.pt" if v == "a"
                               else f"model_{v}.pt")


def resolve_row(name):
    """Row spec or (None, reason)."""
    if name in ("hand", "m0"):
        return dict(name=name, kind=name), None
    parts = name.split("_")
    carry = len(parts) == 4 and parts[3] == "carry"
    if len(parts) != 3 and not carry:
        return None, f"unknown row {name!r}"
    fam, v, mode = parts[:3]
    if fam not in FAMILIES or v not in ("a", "w", "p") or mode not in (
            "prior", "online", "scratch", "bayes") or (
            carry and mode == "prior"):
        return None, f"unknown row {name!r}"
    cache, ck = ckpt_path(fam, v)
    if not os.path.exists(ck):
        return None, f"checkpoint {os.path.relpath(ck, ROOT)} does not exist"
    spec = dict(name=name, kind="net", fam=fam, variant=v, mode=mode,
                carry=carry, cache=cache, ckpt=ck, head=None)
    if fam != "proj":
        from learn.meta.safety_head import head_path
        hp = head_path(cache, v)
        if not os.path.exists(hp):
            return None, (f"safety head {os.path.relpath(hp, ROOT)} missing "
                          "(run --phase heads)")
        import torch
        info = torch.load(hp, map_location="cpu",
                          weights_only=False).get("info", {})
        if abs(info.get("ckpt_mtime", -1) - os.path.getmtime(ck)) > 1e-3:
            return None, (f"safety head {os.path.relpath(hp, ROOT)} is "
                          "older than its checkpoint (rerun --phase heads)")
        spec["head"] = hp
    return spec, None


def phase_heads(variants):
    from learn.meta.safety_head import head_path, train_for_cache
    for fam in ("gen", "cat"):
        for v in variants:
            cache, ck = ckpt_path(fam, v)
            if not os.path.exists(ck):
                log(f"  [heads] {fam}_{v}: no checkpoint "
                    f"{os.path.relpath(ck, ROOT)}, skipped")
                continue
            hp = head_path(cache, v)
            if os.path.exists(hp):
                import torch
                info = torch.load(hp, map_location="cpu",
                                  weights_only=False).get("info", {})
                if abs(info.get("ckpt_mtime", -1)
                       - os.path.getmtime(ck)) < 1e-3:
                    log(f"  [heads] {fam}_{v}: up to date")
                    continue
            if not wait_ram():
                continue
            train_for_cache(cache, v, device(), log=log)


# ------------------------------------------------------------ recording
def _sea_eval(sea, X, Y, t):
    from learn.meta.data5 import sea_eval
    return sea_eval(sea, X, Y, t)


class Recorder:
    """Mission's residual hook (returns the state unchanged): per plant
    step the state after it, the plant's CG / bow accelerations, its
    substep count, the incident wave at the three bow points (x_st[-1],
    y_off) and the water over the jet intake."""

    def __init__(self, m, env, t0=0.0):
        self.m, self.t = m, float(t0)
        self.x_b = float(env["x_st"][-1])
        self.y_off = np.asarray(env["y_off"], float)
        self.S, self.T, self.A, self.AB, self.NS, self.EB, self.IN = \
            [], [], [], [], [], [], []
        self.safe = []

    def __call__(self, s, dt, plant):
        self.t += dt
        self.S.append(np.array(s, float))
        self.T.append(self.t)
        self.A.append(float(plant.last_cg_acc))
        self.AB.append(float(getattr(plant, "last_bow_acc", np.nan)))
        self.NS.append(int(getattr(plant, "last_n_sub", 1)))
        c, sn = math.cos(s[5]), math.sin(s[5])
        X = s[0] + self.x_b * c - self.y_off * sn
        Y = s[1] + self.x_b * sn + self.y_off * c
        self.EB.append(_sea_eval(self.m.sea, X, Y, self.t))
        if getattr(plant, "record", False):
            self.IN.append(np.nan)        # the plant's own log has it
        elif hasattr(plant, "prop_submergence") and np.all(np.isfinite(s)):
            self.IN.append(float(plant.prop_submergence(s[:6], self.t)))
        else:
            self.IN.append(np.nan)
        return s


def step_safety(rec, n0, s0, geo, sub):
    """APK (m/s^2) and HMIN (m) of the control step whose plant steps are
    rec[n0:n0 + sub] (start state s0): the D9.6 / D9.8 definitions on the
    0.04 s grid -- a_j = (w_2j - w_2j-2) / 0.04 and the bow height (roll 0,
    the highest of the three bow points) at the ends j = 1..6."""
    w = [s0[8]] + [rec.S[n0 + i][8] for i in range(sub)]
    dt2 = 2 * rec.m.dt
    a = [(w[i] - w[i - 2]) / dt2 for i in range(2, sub + 1, 2)]
    hb = [geo["fb"] + rec.S[n0 + i - 1][2]
          + geo["s_p"] * geo["x_b"] * rec.S[n0 + i - 1][4]
          - float(np.max(rec.EB[n0 + i - 1])) for i in range(2, sub + 1, 2)]
    return float(max(a)), float(min(hb))


def bow_stats(rec, win_s):
    """(mean, sd, zero-crossing angular frequency) of the measured highest
    bow-point elevation over the last win_s seconds."""
    if not rec.EB:
        return 0.0, 0.0, 2 * math.pi / 2.0
    n = int(round(win_s / rec.m.dt))
    x = np.max(np.asarray(rec.EB[-n:]), 1)
    mu, sd = float(x.mean()), float(x.std())
    d = x - mu
    nc = int(np.sum((d[:-1] < 0) & (d[1:] >= 0)))
    dur = max(len(x) * rec.m.dt, rec.m.dt)
    return mu, sd, 2 * math.pi * max(nc, 1) / dur


def geometry(m, env):
    from learn.meta.operators_gen import freeboard
    p = m.red.p
    fb = float(getattr(m.plant, "freeboard", freeboard(p)))
    return dict(fb=fb, x_b=float(env["x_st"][-1]),
                s_p=float(p["sign_pitch"]), L=float(m.L),
                u_ref=float(m.u_ref))


# ------------------------------------------------------------ one mission
def build_net(spec, ck, head0, dev, seed, leg):
    """(net, head) a mission of a net row starts from: the checkpoint's
    trained weights and a copy of the trained safety head; for scratch rows
    an untrained network of the same shape (seeded by the row's mission, or
    by the first mission of a carry chain) and a fresh head that keeps only
    the trained head's output scale (y_mu / y_sd)."""
    import copy
    import torch
    from learn.meta import model_preview as MP
    from learn.meta import mpc_learned as ML
    if spec["mode"] != "scratch":
        head = copy.deepcopy(head0) if head0 is not None else None
        return MP.load_variant(ck, dev), head
    from learn.meta import model3 as M3
    from learn.meta.safety_head import SafetyHead
    torch.manual_seed(ML.row_seed(seed, leg, 19))
    net = (MP.NetP() if ck.get("variant", "a") in ("w", "p")
           else M3.Net()).to(dev)
    net.eval()
    head = None
    if head0 is not None:
        head = SafetyHead(d=net.d, qs=head0.qs).to(dev)
        head.y_mu.copy_(head0.y_mu)
        head.y_sd.copy_(head0.y_sd)
        head.eval()
    return net, head


def run_mission(spec, seed, leg, T, turn, nets, carry=None):
    """One closed-loop mission of a row; returns its record. carry (carry
    rows): None for the first mission of the chain, else dict(state=the
    learner's state after the previous mission, seed, leg of the chain's
    first mission); the record then holds the new state under
    '_carry_state' (taken out before the record is stored)."""
    import torch
    from learn.meta import mpc_constrained as MC
    from learn.meta import mpc_learned as ML
    from studies.sim2real_jet import theta, z_hand
    env = ML.env()
    kind = spec["kind"]
    if kind == "hand":
        w, floor = theta(z_hand())
        m = make_mission(seed, leg, T, weights=w, floor=floor)
    else:
        m = make_mission(seed, leg, T)
        ML.check_env(m, env)
    geo = geometry(m, env)
    rec = Recorder(m, env)
    m.residual = rec
    # the control step the plant actually advances: sub plant steps (12 x
    # 0.02 = 0.24 s), not the nominal m.dt_ctrl (0.25 s, which only sets
    # n_ctrl: a '120 s' mission is 480 steps = 115.2 s)
    dtc = m.dt * m.sub
    safe = MC.SafeMode(dtc)
    lim = MC.Limits(y_lengths=OPT["y_lengths"])
    jobs = [dict(seed=seed, leg=leg)]
    ctrl = mon = learner = live = None
    net_info = {}
    if kind != "hand":
        net = head = None
        variant, nh = "a", None
        if kind == "net":
            dev = device()
            ck, head0 = nets[spec["name"]]
            # a carry chain builds every mission's learner like its first
            # mission's, then loads the saved state
            s1, l1 = ((carry["seed"], carry["leg"]) if carry else (seed, leg))
            net, head = build_net(spec, ck, head0, dev, s1, l1)
            variant = spec["variant"]
            ctx = int(ck.get("ctx", 256))
            nh = ctx - ML.M.HB
            live = MC.LiveDataW(1, m.n_ctrl + 1, ck["stats"], dev,
                                ML.mission_consts(m), [m.sea], env)
            live.start(0, m.s, m.t)
            fc = dict(lam=OPT["fc_lam"], hp=OPT["fc_hp"], msd=0.01)
            if spec["mode"] == "bayes":
                from learn.meta.bayes_adapter import BayesLearner
                learner = BayesLearner(net, live, 0, ctx, variant, head=head,
                                       seed=ML.row_seed(s1, l1, 17),
                                       fc=fc)
                if carry:
                    learner.load_state(carry["state"])
            elif spec["mode"] in ("online", "scratch"):
                from learn.meta.online_stream import OnlineLearner
                learner = OnlineLearner(net, live, 0, ctx, variant, head=head,
                                        seed=ML.row_seed(s1, l1, 17),
                                        scratch=spec["mode"] == "scratch")
                if carry:
                    learner.load_state(carry["state"])
            mon = MC_monitor(net, live, ctx, variant, head, fc, seed, leg)
            net_info = dict(ctx=ctx, variant=variant, head=head is not None,
                            mode=spec["mode"], carry=bool(spec.get("carry")),
                            n_before=learner.n if learner else 0)
        ctrl = MC.ConstrainedMPPI(jobs, [m], env, lim, geo, net=net,
                                  live=live, head=head, variant=variant,
                                  nh=nh, K=OPT["K"], S=OPT["S"],
                                  fc=dict(lam=OPT["fc_lam"],
                                          hp=OPT["fc_hp"]),
                                  steer=OPT["steer"])
    t_ctl = t_lrn = 0.0
    n_steps = 0
    past, diags = [], []
    n_win = int(round(lim.win_s / dtc))
    cmd_thr, cmd_noz, safe_on = [], [], []
    hdg = []                     # heading minus track angle per step (rad)
    k = 0
    while not m.done():
        tc = time.perf_counter()
        if kind == "hand":
            thrust, rud = m.mpc_command()
        else:
            info = dict(past=np.asarray(past[-n_win:]) / task.G,
                        u_meas=float(m.s[6]),
                        r_max=MC.r_max_of(turn, float(m.s[6])),
                        bow=bow_stats(rec, lim.win_s),
                        infl=mon.infl if (mon is not None
                                          and OPT["trigger"]) else 1.0)
            cmds, dg = ctrl([0], k, [info])
            thrust = float(cmds[0]) * m.t_max
            if OPT["steer"] == "free":
                rud = dg[0]["nozzle"] * m.rud_max
            else:
                rud = m.ep._steer(m.s, thrust)
            diags.append(dg[0])
        t_ctl += time.perf_counter() - tc
        thrust, rud, on = safe(m.s[3], thrust, rud)
        u_app = (thrust / m.t_max, rud / m.rud_max)
        tl = time.perf_counter()
        pred = mon.predict(k, u_app) if mon is not None else None
        t_lrn += time.perf_counter() - tl
        s0 = m.s.copy()
        n0 = len(rec.S)
        m.advance(thrust, rud)
        cmd_thr += [thrust] * (len(rec.S) - n0)
        cmd_noz += [rud] * (len(rec.S) - n0)
        safe_on += [on] * (len(rec.S) - n0)
        n_steps += 1
        if not m.finite or getattr(m.plant, "capsized", False):
            break
        apk, hmin = step_safety(rec, n0, s0, geo, m.sub)
        past.append(apk)
        hdg.append(math.remainder(float(m.s[5]) - float(m.phi), 2 * math.pi))
        tl = time.perf_counter()
        if live is not None:
            live.push(0, u_app, m.s, m.t, apk, hmin)
            if mon is not None:
                mon.update(pred, k)
            if learner is not None:
                learner.step(k)
        t_lrn += time.perf_counter() - tl
        k += 1
        if k % 40 == 0:
            log(f"    {spec['name']} s{seed} l{leg} step {k}/{m.n_ctrl}: "
                f"{m.s[6] / KN:.1f} kn, {(t_ctl + t_lrn) / n_steps:.2f} s "
                f"per step" + (f", infl {mon.infl:.2f}" if mon else ""))
    met = m.metrics()
    events, raw = events_for(m, rec, geo, cmd_thr, cmd_noz, safe_on)
    r = dict(seed=seed, leg=leg, row=spec["name"], **met,
             capsized=bool(getattr(m.plant, "capsized", False)),
             events_raw=raw,
             kn=met["u_along"] / KN if np.isfinite(met["u_along"])
             else float("nan"),
             events=events, t_ctrl_step=t_ctl / max(n_steps, 1),
             t_learn_step=t_lrn / max(n_steps, 1),
             t_step=(t_ctl + t_lrn) / max(n_steps, 1), n_steps=n_steps,
             safe_entries=safe.n_entries, safe_steps=safe.n_on,
             apk_mean_g=float(np.mean(past)) / task.G if past else np.nan,
             plant=OPT["plant"], net=net_info, steer=OPT["steer"],
             hdg_dev_mean_deg=(math.degrees(float(np.mean(hdg))) if hdg
                               else float("nan")),
             hdg_dev_abs_deg=(math.degrees(float(np.mean(np.abs(hdg))))
                              if hdg else float("nan")))
    if diags:
        r["plan"] = dict(
            n_ok_mean=float(np.mean([d["n_ok"] for d in diags])),
            all_violate_share=float(np.mean([d["n_ok"] == 0
                                             for d in diags])),
            **{nm: float(np.mean([d[nm] > 0 for d in diags]))
               for nm in diags[0] if nm.startswith("g_")},
            kind=diags[0]["kind"])
        if "nozzle" in diags[0]:
            r["plan"]["nozzle_abs_mean"] = float(np.mean(
                [abs(d["nozzle"]) for d in diags]))
    if ctrl is not None:
        r["n_bad"], r["n_samp"] = int(ctrl.n_bad[0]), int(ctrl.n_samp[0])
    if mon is not None:
        r["monitor"] = mon.summary()
    if learner is not None:
        r["learner"] = learner.summary()
        if spec.get("carry"):
            r["_carry_state"] = learner.state()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return r


def MC_monitor(net, live, ctx, variant, head, fc, seed, leg):
    from learn.meta import mpc_learned as ML
    from learn.meta.online_stream import Monitor
    return Monitor(net, live, 0, ctx, variant, head=head, fc=fc,
                   seed=ML.row_seed(seed, leg, 13))


# ------------------------------------------------------------ events
def trajectory(m, rec, geo, cmd_thr, cmd_noz, safe_on):
    """The closed-loop record handed to the event counter: per plant step
    t, s (14), a_cg / a_bow (m/s^2), n_sub, eta_bow (3), bow_h (roll 0),
    intake (water over the intake top, m), thr_cmd (N), noz_cmd (rad),
    safe; constants of the boat, the track, the sea."""
    p = m.red.p
    n = len(rec.S)
    S = np.asarray(rec.S, float).reshape(n, 14)
    EB = np.asarray(rec.EB, float).reshape(n, 3)
    bow = geo["fb"] + S[:, 2] + geo["s_p"] * geo["x_b"] * S[:, 4] \
        - EB.max(1) if n else np.zeros(0)
    tp = float(getattr(m.sea, "tp", task.TP))
    return dict(dt=m.dt, t=np.asarray(rec.T), s=S, a_cg=np.asarray(rec.A),
                a_bow=np.asarray(rec.AB), n_sub=np.asarray(rec.NS),
                eta_bow=EB, bow_h=bow, intake=np.asarray(rec.IN),
                thr_cmd=np.asarray(cmd_thr), noz_cmd=np.asarray(cmd_noz),
                safe=np.asarray(safe_on, bool), finite=bool(m.finite),
                t_end=float(m.t), L=float(m.L),
                B=float(getattr(m.plant, "B", np.nan)),
                u_ref=float(m.u_ref), u_max=float(p["u_max"]),
                freeboard=geo["fb"], x_bow=geo["x_b"],
                sign_pitch=geo["s_p"], noz_max=float(m.rud_max),
                steer_sign=float(np.sign(p.get("k_nomoto_f", -1.0)
                                         * p.get("k_jet_side", 1.0))),
                phi=float(m.phi), wave_dir=float(m.theta0),
                wave_c=task.G * tp / (2 * math.pi),
                n_sub_max=64, g=task.G)


def events_for(m, rec, geo, cmd_thr, cmd_noz, safe_on):
    """The run's safety events: ({'1.5', '2.0', '2.5': flat dict}, raw).
    With a recording GZ plant and studies/safety_events.py: its
    safety_events(plant.trajectory(), k=2.0, k_report=(1.5, 2.5),
    track=phi), kept whole as `raw` and flattened per k for the report;
    otherwise this file's fallback counter on the Recorder's trajectory
    (raw None)."""
    se = None
    if getattr(m.plant, "record", False):
        try:
            from studies import safety_events as se
        except ImportError:
            se = None
    if se is not None:
        raw = se.safety_events(m.plant.trajectory(), k=2.0,
                               k_report=(1.5, 2.5), track=float(m.phi))
        return {str(kk): flatten_events(raw, kk) for kk in K_EVAL}, raw
    traj = trajectory(m, rec, geo, cmd_thr, cmd_noz, safe_on)
    why = ("studies/safety_events.py absent" if getattr(
        m.plant, "record", False) else "plant without recording")
    out = {}
    for kk in K_EVAL:
        out[str(kk)] = fallback_events(traj, kk)
        out[str(kk)]["source"] = f"fallback ({why})"
    return out, None


def flatten_events(raw, k):
    """The report's flat view of studies/safety_events output at one k."""
    a = raw["acc"][k]
    h = raw["hours"]
    per = (lambda c: c / h) if h > 0 else (lambda c: float("nan"))  # noqa
    mg = raw.get("mgn328") or {}
    out = dict(hours=h, k=k, source="studies.safety_events",
               a110=a["A110_run_g"], peak_max=a["peak_max_g"],
               a110_win_over_share=a["share_windows_over_3g"],
               a110_partial=bool(a["window_partial"]),
               n_ge7=a["peaks_ge_7g"], n_ge10=a["peaks_ge_10g"],
               bow_wet=raw["bow"]["wet"], bow_burial=raw["bow"]["buried"],
               bow_min=raw["bow"]["min_height"],
               broach=raw["yaw"]["broach"],
               heading_gt10=raw["yaw"]["heading_dev_over_10"],
               roll_gt20=raw["roll"]["over_20"],
               roll_gt_fail=raw["roll"]["over_fail"],
               roll_max_deg=raw["roll"]["max_deg"],
               capsize=int(bool(raw["capsize"]["capsized"])),
               intake_out=raw["intake"]["emergence"],
               numeric=int(bool(raw["numerical"]["any"])),
               y_within_2L=raw["track"]["share_within_2L"],
               y_fail_5L=int(bool(raw["track"]["fail_5L"])),
               surf_band_share=mg.get("share", float("nan")))
    for nm in ("n_ge7", "n_ge10", "bow_wet", "bow_burial", "broach",
               "heading_gt10", "roll_gt20", "roll_gt_fail", "intake_out"):
        out[nm + "_per_h"] = per(out[nm])
    return out


def _entries(mask):
    m = np.asarray(mask, bool)
    return int(np.sum(m[1:] & ~m[:-1]) + (1 if len(m) and m[0] else 0))


def _runs(mask, n_min):
    """Runs of True at least n_min long."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(int))
    st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return int(np.sum((en - st) >= n_min))


def fallback_events(tr, k):
    """The section 9 evaluation events from a trajectory (a minimal stand-in
    for studies/safety_events.py): accelerations as NSWCCD (k x a_cg,
    gravity removed, 10 Hz 4-pole low-pass, peaks >= 0.3 g at least 0.1 s
    apart), A1/10 over the run and the share of 60-s windows (1-s steps)
    above 3.0 g, peaks >= 7 g and >= 10 g; bow wet entries (bow height
    < 0, roll 0) and burials (< -F_b / 4, or wet for 0.5 s); broaches
    (nozzle >= 95% hard over against the heading error while |heading
    error| and |yaw rate| grow, 0.5 s); heading error > 10 deg entries;
    roll > 20 / 40 deg entries, capsize (> 90 deg); intake emergence
    entries; numerical events; time in the MGN 328 surf-riding band;
    cross-track share within 2 L and the 5 L failure."""
    from scipy.signal import butter, find_peaks, sosfilt
    dt, n = tr["dt"], len(tr["t"])
    hours = max(tr["t_end"], dt) / 3600.0
    out = dict(hours=hours, k=k)
    if n < 10:
        return out
    S = tr["s"]
    fin = np.all(np.isfinite(S), 1)
    a = np.nan_to_num(k * tr["a_cg"] / tr["g"])
    af = sosfilt(butter(4, 10.0, fs=1.0 / dt, output="sos"), a)
    pk, _ = find_peaks(af, height=0.3, distance=max(1, int(round(0.1 / dt))))
    vals = af[pk]
    top = np.sort(vals)[::-1]
    out["a110"] = float(top[:max(1, int(math.ceil(len(top) / 10)))].mean()) \
        if len(top) else 0.0
    out["peak_max"] = float(top[0]) if len(top) else 0.0
    out["n_ge7"] = int(np.sum(vals >= 7.0))
    out["n_ge10"] = int(np.sum(vals >= 10.0))
    tpk = tr["t"][pk]
    over, nw = 0, 0
    for t1 in np.arange(60.0, tr["t_end"] + 1e-9, 1.0):
        v = np.sort(vals[(tpk > t1 - 60.0) & (tpk <= t1)])[::-1]
        nw += 1
        if len(v) and v[:max(1, int(math.ceil(len(v) / 10)))].mean() > 3.0:
            over += 1
    out["a110_win_over_share"] = over / nw if nw else float("nan")
    bow = tr["bow_h"]
    wet = bow < 0
    n05 = int(round(0.5 / dt))
    out["bow_wet"] = _entries(wet)
    # a wet spell is a burial when it lasts 0.5 s or goes below -F_b / 4
    m_ = np.r_[False, wet, False]
    st_, en_ = (np.nonzero(np.diff(m_.astype(int)) == 1)[0],
                np.nonzero(np.diff(m_.astype(int)) == -1)[0])
    out["bow_burial"] = int(sum(
        (e_ - s_) >= n05 or bow[s_:e_].min() < -0.25 * tr["freeboard"]
        for s_, e_ in zip(st_, en_)))
    err = (S[:, 5] - tr["phi"] + np.pi) % (2 * np.pi) - np.pi
    r = S[:, 11]
    corr = -tr["steer_sign"] * np.sign(err)
    counter = S[:, 13] * corr >= 0.95 * tr["noz_max"]
    grow = np.r_[False, (np.diff(np.abs(err)) > 0) & (np.diff(np.abs(r)) > 0)
                 & (r[1:] * err[1:] > 0)]
    out["broach"] = _runs(counter & grow, n05)
    out["heading_gt10"] = _entries(np.abs(err) > math.radians(10.0))
    roll = np.abs(np.degrees(S[:, 3]))
    out["roll_gt20"] = _entries(roll > 20.0)
    out["roll_gt_fail"] = _entries(roll > 40.0)
    out["capsize"] = int(np.any(roll > 90.0))
    inn = tr["intake"]
    out["intake_out"] = _entries(np.isfinite(inn) & (inn < 0)) \
        if np.isfinite(inn).any() else None
    out["numeric"] = int((not tr["finite"]) or np.any(~fin)
                         or np.any(tr["n_sub"] >= tr["n_sub_max"])
                         or np.any(S[fin, 6] > 1.5 * tr["u_max"])
                         or np.any(np.abs(S[fin, 4]) > math.pi / 4))
    vg = np.stack([S[:, 6] * np.cos(S[:, 5]) - S[:, 7] * np.sin(S[:, 5]),
                   S[:, 6] * np.sin(S[:, 5]) + S[:, 7] * np.cos(S[:, 5])], 1)
    ua = vg @ np.array([math.cos(tr["wave_dir"]), math.sin(tr["wave_dir"])])
    q = ua / tr["wave_c"]
    out["surf_band_share"] = float(np.mean((q >= 0.7) & (q <= 1.15)))
    y = -math.sin(tr["phi"]) * S[:, 0] + math.cos(tr["phi"]) * S[:, 1]
    out["y_within_2L"] = float(np.mean(np.abs(y) <= 2 * tr["L"]))
    out["y_fail_5L"] = int(np.any(np.abs(y) > 5 * tr["L"]))
    for nm in ("n_ge7", "n_ge10", "bow_wet", "bow_burial", "broach",
               "heading_gt10", "roll_gt20", "roll_gt_fail", "intake_out"):
        if out.get(nm) is not None:
            out[nm + "_per_h"] = out[nm] / hours
    return out


# ------------------------------------------------------------ phases
def load_results():
    """results.pkl plus the per-mission part files the parallel workers
    write (parts/<row>__<mission>.pkl, merged here; phase_eval folds them
    into results.pkl and removes them)."""
    p = os.path.join(out_dir(), "results.pkl")
    R = pickle.load(open(p, "rb")) if os.path.exists(p) else dict(rows={})
    for fp in sorted(glob.glob(os.path.join(out_dir(), "parts", "*.pkl"))):
        try:
            name, key, r = pickle.load(open(fp, "rb"))
        except Exception:
            continue                    # a part still being written
        R["rows"].setdefault(name, {})[key] = r
    return R


def _fold_parts(R):
    """Write R (with the parts merged by load_results) and drop the parts
    it now holds."""
    atomic_pickle(R, os.path.join(out_dir(), "results.pkl"))
    for fp in glob.glob(os.path.join(out_dir(), "parts", "*.pkl")):
        try:
            name, key, _ = pickle.load(open(fp, "rb"))
        except Exception:
            continue
        if key in R["rows"].get(name, {}):
            os.remove(fp)


def _prepare(spec, rr, missions, legs):
    """(todo, order, carry) of one row, or None (logged) when it has
    nothing to do or its carry chain does not match. carry: None (no
    carry row, or its chain starts now) or "load" (_run_unit loads the
    saved state on its own device, so the parent never touches the GPU)."""
    import torch
    todo = [(s, lg) for s in missions for lg in legs
            if f"s{s}_l{lg}" not in rr]
    if not todo:
        log(f"  [eval] {spec['name']}: done")
        return None
    order = [f"s{s}_l{lg}" for s in missions for lg in legs]
    cpath = os.path.join(out_dir(), f"carry_{spec['name']}.pt")
    carry = None
    if spec.get("carry"):
        done = [key for key in order if key in rr]
        if done != order[:len(done)]:
            log(f"  [eval] {spec['name']}: finished missions {done} are "
                f"not a prefix of the chain {order} (other --missions / "
                f"--legs than before?): skipped; use another --tag")
            return None
        if done:
            c = torch.load(cpath, map_location="cpu", weights_only=False)
            if c.get("last") != done[-1] or c.get("order", [])[
                    :len(done)] != done:
                log(f"  [eval] {spec['name']}: {cpath} does not match "
                    f"the finished missions: skipped")
                return None
            carry = "load"
    return todo, order, carry


_NETS = {}


def _row_nets(spec):
    """{row: (checkpoint, head)} for a net row, cached per process (one
    row's net at a time)."""
    import torch
    if spec["kind"] != "net":
        return {}
    if spec["name"] not in _NETS:
        dev = device()
        ck = torch.load(spec["ckpt"], map_location=dev, weights_only=False)
        head = None
        if spec["head"]:
            from learn.meta.safety_head import load_head
            head, _ = load_head(spec["head"], dev)
        _NETS.clear()
        _NETS[spec["name"]] = (ck, head)
    return _NETS


def _run_unit(spec, todo, order, carry, T, turn, sink):
    """Run the missions `todo` of one row in order (a carry row's chain:
    the learner state passes from mission to mission and is saved after
    each); sink(key, record) stores each record. False when RAM ran out."""
    import torch
    nets = _row_nets(spec)
    cpath = os.path.join(out_dir(), f"carry_{spec['name']}.pt")
    if carry == "load":
        carry = torch.load(cpath, map_location=device(), weights_only=False)
    for s, lg in todo:
        if not wait_ram():
            return False
        t0 = time.time()
        r = run_mission(spec, s, lg, T, turn, nets, carry=carry)
        r["wall_s"] = time.time() - t0
        key = f"s{s}_l{lg}"
        if spec.get("carry"):
            carry = dict(state=r.pop("_carry_state"), order=order,
                         last=key, seed=(carry or dict(seed=s))["seed"],
                         leg=(carry or dict(leg=lg))["leg"])
            torch.save(carry, cpath + ".tmp")
            os.replace(cpath + ".tmp", cpath)
        sink(key, r)
        e = r["events"]["2.0"]
        log(f"  [eval] {spec['name']} s{s} l{lg}: {r['kn']:.1f} kn, "
            f"A1/10 {e.get('a110', float('nan')):.2f} g (k 2), "
            f">=7g {e.get('n_ge7')}, bow wet {e.get('bow_wet')}, "
            f"broach {e.get('broach')}, roll>20 {e.get('roll_gt20')}, "
            f"{r['t_step']:.2f} s/step (ctrl {r['t_ctrl_step']:.2f}), "
            f"{r['wall_s']:.0f} s wall")
    return True


def _worker_init(opt, threads):
    import torch
    OPT.update(opt)
    torch.set_num_threads(max(1, int(threads)))
    task.ctx()


def _worker(spec, todo, order, carry, T, turn):
    """One unit in a worker process: each record goes to its own part
    file (the parent folds them into results.pkl)."""
    pdir = os.path.join(out_dir(), "parts")
    os.makedirs(pdir, exist_ok=True)

    def sink(key, r):
        fp = os.path.join(pdir, f"{spec['name']}__{key}.pkl")
        atomic_pickle((spec["name"], key, r), fp)
    try:
        return _run_unit(spec, todo, order, carry, T, turn, sink)
    except Exception:
        import traceback
        log(f"  [eval] {spec['name']} {todo}: worker failed\n"
            + traceback.format_exc())
        return False


def phase_eval(rows, missions, legs, T, procs=1):
    """procs 1: every row's missions in turn in this process. procs > 1:
    units (one mission of a row; a carry row's whole chain) in procs
    worker processes, torch threads split between them; each new unit
    waits for OPT['min_gb'] free RAM."""
    R = load_results()
    R.setdefault("meta", {}).update(plant=OPT["plant"], T=T, S=OPT["S"],
                                    K=OPT["K"], fc_lam=OPT["fc_lam"],
                                    fc_hp=OPT["fc_hp"],
                                    trigger=OPT["trigger"],
                                    steer=OPT["steer"],
                                    y_lengths=OPT["y_lengths"])
    _fold_parts(R)
    turn = phase_turn()
    specs = []
    for nm in rows:
        spec, why = resolve_row(nm)
        if spec is None:
            log(f"  [eval] row {nm} skipped: {why}")
            continue
        specs.append(spec)
    units = []
    for spec in specs:
        prep = _prepare(spec, R["rows"].setdefault(spec["name"], {}),
                        missions, legs)
        if prep is None:
            continue
        todo, order, carry = prep
        if spec.get("carry") or procs <= 1:
            units.append((spec, todo, order, carry))
        else:
            units += [(spec, [m_], order, None) for m_ in todo]
    if procs <= 1:
        for spec, todo, order, carry in units:
            rr = R["rows"][spec["name"]]

            def sink(key, r, rr=rr):
                rr[key] = r
                atomic_pickle(R, os.path.join(out_dir(), "results.pkl"))
            if not _run_unit(spec, todo, order, carry, T, turn, sink):
                return
        return
    import multiprocessing as mp
    from concurrent.futures import (FIRST_COMPLETED, ProcessPoolExecutor,
                                    wait)
    threads = max(1, (os.cpu_count() or procs) // procs)
    log(f"  [eval] {len(units)} units in {procs} processes, {threads} torch "
        f"threads each")
    opt = dict(OPT)
    opt["gpu_frac"] = min(0.35, 0.9 / procs)
    with ProcessPoolExecutor(max_workers=procs,
                             mp_context=mp.get_context("spawn"),
                             initializer=_worker_init,
                             initargs=(opt, threads)) as ex:
        pending, queue = set(), list(units)
        while queue or pending:
            while queue and len(pending) < procs:
                if not wait_ram():
                    queue = []
                    break
                spec, todo, order, carry = queue.pop(0)
                pending.add(ex.submit(_worker, spec, todo, order, carry, T,
                                      turn))
                if queue and len(pending) < procs:
                    time.sleep(20)      # let the new process load first
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            _fold_parts(load_results())
    _fold_parts(load_results())


def phase_report():
    R = load_results()
    lines = [f"final_eval report ({OPT['plant']} plant): rows x missions; "
             "events summed over missions, rates per hour of mission time"]
    summ = {}
    for name, rr in R["rows"].items():
        recs = list(rr.values())
        if not recs:
            continue
        hours = sum(r["events"]["2.0"].get("hours", 0.0) for r in recs)
        row = dict(n=len(recs), hours=hours,
                   kn=float(np.nanmean([r["kn"] for r in recs])),
                   kn_sd=float(np.nanstd([r["kn"] for r in recs])),
                   finite=int(sum(bool(r["finite"]) for r in recs)),
                   t_step=float(np.mean([r["t_step"] for r in recs])),
                   t_ctrl=float(np.mean([r["t_ctrl_step"] for r in recs])),
                   source=recs[0]["events"]["2.0"].get("source"))
        for kk in K_EVAL:
            ev = [r["events"][str(kk)] for r in recs]
            d = {}
            for nm in ev[0]:
                vals = [e.get(nm) for e in ev]
                if all(isinstance(v, (int, float, np.integer, np.floating))
                       and v is not None for v in vals) and nm not in (
                        "k", "hours"):
                    if nm.endswith("_per_h"):
                        continue
                    if nm in ("a110", "peak_max", "a110_win_over_share",
                              "surf_band_share", "y_within_2L"):
                        d[nm] = float(np.nanmean(vals))
                    elif nm == "bow_min":
                        d[nm] = float(np.min(vals))
                    elif nm == "roll_max_deg":
                        d[nm] = float(np.max(vals))
                    else:
                        d[nm] = float(np.sum(vals))
                        d[nm + "_per_h"] = d[nm] / max(hours, 1e-9)
            row[f"k{kk}"] = d
        mons = [r["monitor"] for r in recs if "monitor" in r]
        if mons:
            row["monitor"] = {k_: float(np.nanmean([m_[k_] for m_ in mons]))
                              for k_ in mons[0]}
        summ[name] = row
        e = row["k2.0"]
        lines.append(
            f"  {name:<16} n {row['n']:2d}  {row['kn']:5.1f} +- "
            f"{row['kn_sd']:.1f} kn  A1/10 {e.get('a110', np.nan):.2f} g  "
            f">=7g/h {e.get('n_ge7_per_h', np.nan):.1f}  >=10g "
            f"{e.get('n_ge10', np.nan):.0f}  bow wet/h "
            f"{e.get('bow_wet_per_h', np.nan):.1f}  burial "
            f"{e.get('bow_burial', np.nan):.0f}  broach "
            f"{e.get('broach', np.nan):.0f}  roll>20 "
            f"{e.get('roll_gt20', np.nan):.0f} (max "
            f"{e.get('roll_max_deg', np.nan):.1f} deg) >fail "
            f"{e.get('roll_gt_fail', np.nan):.0f}  capsize "
            f"{e.get('capsize', np.nan):.0f}  intake/h "
            f"{e.get('intake_out_per_h', np.nan):.0f}  numeric "
            f"{e.get('numeric', np.nan):.0f}  y<2L "
            f"{e.get('y_within_2L', np.nan):.2f}  {row['t_step']:.2f} s/step")
        for kk in (1.5, 2.5):
            e2 = row[f"k{kk}"]
            lines.append(f"  {'':<16} k {kk}: A1/10 {e2.get('a110', np.nan):.2f}"
                         f" g, windows over 3 g "
                         f"{e2.get('a110_win_over_share', np.nan):.2f}, "
                         f">=7g {e2.get('n_ge7', np.nan):.0f}, >=10g "
                         f"{e2.get('n_ge10', np.nan):.0f}")
        hd = [r for r in recs if "hdg_dev_abs_deg" in r]
        if hd:
            row["hdg_dev_abs_deg"] = float(np.nanmean(
                [r["hdg_dev_abs_deg"] for r in hd]))
            row["hdg_dev_mean_deg"] = float(np.nanmean(
                [r["hdg_dev_mean_deg"] for r in hd]))
            lines.append(f"  {'':<16} steer {hd[0].get('steer', 'auto')}: "
                         f"|heading - track| {row['hdg_dev_abs_deg']:.1f} deg"
                         f" (mean signed {row['hdg_dev_mean_deg']:+.1f})")
        if "monitor" in row:
            mm = row["monitor"]
            lines.append(f"  {'':<16} monitor: cov90 vel "
                         f"{mm['cov90_vel']:.2f}, cov99 vel "
                         f"{mm['cov99_vel']:.2f}, APK out90 "
                         f"{mm['apk_out90']:.2f}, APK > q99 "
                         f"{mm['apk_above99']:.3f}, inflation mean "
                         f"{mm['infl_mean']:.2f} max {mm['infl_max']:.2f}")
        if name.endswith("_carry"):
            seq = sorted(recs, key=lambda r_: r_["net"].get("n_before", 0))
            row["chain"] = [dict(mission=f"s{r_['seed']}_l{r_['leg']}",
                                 kn=r_["kn"], n_before=r_["net"]["n_before"],
                                 a110=r_["events"]["2.0"].get("a110"))
                            for r_ in seq]
            lines.append(f"  {'':<16} chain (kn, A1/10 g at k 2): " + ", ".join(
                f"{c['kn']:.1f}/{c['a110'] if c['a110'] is not None else np.nan:.2f}"
                for c in row["chain"]))
        if hours > 0:
            lines.append(f"  {'':<16} {hours:.3f} h: 0 capsizes bound "
                         f"the rate below {3.0 / hours:.0f} per hour (95%)")
    txt = "\n".join(lines)
    log(txt)
    atomic_json(summ, os.path.join(out_dir(), "summary.json"))
    with open(os.path.join(out_dir(), "report.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    return summ


def parse_range(s):
    out = []
    for part in str(s).split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="turn,eval,report")
    ap.add_argument("--rows", default="hand,m0,proj_p_prior,proj_p_online,"
                    "gen_p_prior,gen_p_online,gen_p_bayes,gen_p_scratch,"
                    "cat_p_prior,cat_p_online,cat_p_bayes,cat_p_scratch",
                    help="also ..._online_carry / ..._scratch_carry (one "
                    "boat over consecutive missions; best with its own "
                    "--tag)")
    ap.add_argument("--head-variants", default="p",
                    help="variants whose gen / cat safety heads phase "
                    "heads trains")
    ap.add_argument("--missions", default="0-7")
    ap.add_argument("--legs", default="0,1")
    ap.add_argument("--T", type=float, default=task.T_EVAL)
    ap.add_argument("--plant", default="auto",
                    help="gz (sim/planing_vessel_gz.py, the default when it "
                    "imports) or default (sim/planing_vessel.py)")
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--fc-lam", type=float, default=0.1)
    ap.add_argument("--fc-hp", type=int, default=24)
    ap.add_argument("--no-trigger", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    ap.add_argument("--force-turn", action="store_true")
    ap.add_argument("--steer", default="auto", choices=("auto", "free"),
                    help="auto: the heading autopilot sets the nozzle (rollouts "
                    "and boat); free: the MPPI plans the nozzle too "
                    "(learn/meta/mpc_constrained.py). The hand row keeps its "
                    "own controller either way")
    ap.add_argument("--procs", type=int, default=1,
                    help="worker processes for phase eval (one mission of a "
                    "row, or a carry row's chain, per unit); each needs "
                    "about 1-2 GB; a new unit waits for --min-gb free")
    ap.add_argument("--min-gb", type=float, default=3.0,
                    help="free RAM (GB) a new mission waits for")
    ap.add_argument("--y-lengths", type=float, default=2.0,
                    help="the planner's cross-track limit in boat lengths")
    a = ap.parse_args()
    # results of another steering or corridor never share a directory (a
    # resumed run would skip missions finished under the other setting)
    suffix = ([] if a.steer == "auto" else ["steerfree"]) + (
        [] if a.y_lengths == 2.0 else [f"y{a.y_lengths:g}"])
    if suffix:
        a.tag = "_".join(([a.tag] if a.tag else []) + suffix)
    if a.plant == "auto":
        try:
            import sim.planing_vessel_gz   # noqa: F401
            a.plant = "gz"
        except ImportError:
            a.plant = "default"
    OPT.update(tag=a.tag, plant=a.plant, T=a.T, S=a.S, K=a.K,
               device=a.device, fc_lam=a.fc_lam, fc_hp=a.fc_hp,
               trigger=not a.no_trigger, steer=a.steer, min_gb=a.min_gb,
               y_lengths=a.y_lengths)
    phases = a.phase.split(",")
    log(f"final_eval phases {phases} rows {a.rows} missions {a.missions} "
        f"legs {a.legs} T {a.T} plant {a.plant} S {a.S} K {a.K} "
        f"steer {a.steer} corridor {a.y_lengths:g} L tag {a.tag!r} "
        f"({free_gb():.1f} GB free)")
    task.ctx()
    if "turn" in phases:
        phase_turn(force=a.force_turn)
    if "heads" in phases:
        phase_heads(a.head_variants.split(","))
    if "eval" in phases:
        phase_eval(a.rows.split(","), parse_range(a.missions),
                   parse_range(a.legs), a.T, procs=a.procs)
    if "report" in phases:
        phase_report()


if __name__ == "__main__":
    main()
