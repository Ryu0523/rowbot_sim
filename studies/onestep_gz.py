#!/usr/bin/env python3
"""
Fast one-step evaluation of the error model on recorded missions of the gz
plant (learn/meta/ROADMAP_2026-09-30.md section 11): the yardstick for the
candidates there (capacity, memory structures, online learning), in
minutes instead of the hours of a closed-loop final_eval run.

The missions are driven by the hand controller (final_eval's hand row:
no net in the loop, deterministic), recorded once and replayed: every net
sees the same observations. Default split: seeds 0-3 x legs 0, 1 train
(the missions the closed-loop learner rows ran), seeds 4-7 held out.

  record  the hand row's loop of final_eval.run_mission, saving what a
          LiveDataW needs (states, times, applied commands, per-step APK /
          HMIN, the sea); cache studies/_cache/onestep_gz/streams_T<T>.pkl
  eval    rows frozen on the held-out missions. Every step k >= K0 is
          scored once, before anything learns from it, with the deployment
          context: the window ends at the scored step's chunk (STRIDE
          steps), so a step k >= 256 sees >= 256 - STRIDE steps of history
          and a step k < 256 all of its history. Tokens all measured
          (moment beyond the window, model_preview.fixed_cfg), preview at
          final_eval's forecast level. Common random numbers: per mission
          and step the same flow base noise, PIT jitter and safety-head
          uniforms for every row (any variant or context length), and the
          same token perturbations for rows of the same variant and
          context, so differences between rows are paired.
          Learner rows are not accepted here (nothing learned yet: use
          online); the reference row must be a prior row.
  online  a learner (online: LoRA + flow head + safety head; bayes: the
          low-rank adapter) learns over the train missions in order, one
          update per control step, its state carried from mission to
          mission as final_eval's carry rows do (state() / load_state()
          into a freshly built net), --passes times; then 'eval' of the
          frozen state on the held-out missions.

Scores per channel (surge sway yaw heave pitch first): skill = MSE of the
sample mean (sample-mean variance removed) / MSE of predicting 0, pooled
over the scored steps (model3.Tally); coverage of the central 90 % by the
randomised PIT (exactly 0.90 when calibrated, any sample count); CRPS
relative to a Gaussian with the truth's RMS. Against the reference row:
the MSE ratio per channel, pooled and per mission. Safety head (if the
row has one): teacher-forced pinball loss (true e, as the learners train
it) and the coverage of the predictive draws of APK and HMIN (sampled e,
as the Monitor and the MPC use them; no inflation).

    python -m studies.onestep_gz --phase record
    python -m studies.onestep_gz --phase eval --rows cat_p_prior,gen_p_prior
    python -m studies.onestep_gz --phase online --rows cat_p_online \\
        --tau 0.02,0.1 [--passes 1]
    python -m studies.onestep_gz --phase online --rows cat_p_bayes \\
        --bayes-tau 0.05,1.0
    python -m studies.onestep_gz --phase eval --rows cat_p_prior \\
        --memnets rec=studies/_cache/meta8/mem_recurrent.pt

  batch   batch training on target data (a diagnostic, the M11 question
          with today's nets): a copy of each net (--ref row, --ckpts) is
          trained on windows of the first N of the train missions that are
          not validation missions (--batch-sizes), with the pretraining
          loss and token convention (model_preview.train_tokens_p), only
          the flow head or the whole net (--batch-parts); the step with the
          lowest flow-matching loss on the validation missions (--batch-val)
          is kept; then all are scored frozen on the held-out missions.
          --batch-heads adds other learners on the same data: headfresh
          (a freshly initialised flow head), mse (a mean head on the
          squared error, constant spread from the validation missions),
          gauss (mean and variance, Gaussian likelihood), ens<K> (K flow
          heads fine-tuned with different seeds, their mixture), scratch
          (the whole net from random weights); --calibrate adds a copy of
          each with its spread widened per channel about the mean until
          the validation missions' 90 % coverage is reached.
  ttt     test-time training inside one mission (no pull): for every
          held-out mission a copy of the --ref net is trained on the
          mission's own first K steps (--ttt-split; windows of 128 tokens
          inside [0, K), the pretraining loss and token convention, a fixed
          number of steps --ttt-steps, head only or full --ttt-parts) and
          scored on steps >= K of the same mission, against the frozen net
          on the same steps (paired); optionally (--ttt-cross N) a copy
          trained on the first N train missions as in 'batch' is scored on
          the same steps for comparison.
  --windows  every row (or the --windows-rows) also scored with windows
          of L tokens and one scored step per window, so each scored step
          has min(k, L - 1) steps of history (L = 1: the current token
          only).

Memory networks (--memnets, studies/train_mem.py): the memory is written
once from the last --mem-missions train missions in order, then the
held-out missions are scored with it; each also as '<label> empty' (the
same net, memory left empty), so the memory's own effect is paired.

Cost on this machine (one RTX 3080 Ti laptop GPU): frozen rows about 20 s
to build the held-out missions' data plus about 10 s per row; a learner
about 4-5 min per pass over the 8 train missions (one update per step).
A closed-loop final_eval mission of a learned row takes 10-20 min.
"""
import argparse
import copy
import os
import pickle
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
import numpy as np   # noqa: E402
import torch   # noqa: E402  (before final_eval, which caps OMP threads)

from studies import final_eval as FE   # noqa: E402
from learn.meta import model3 as M   # noqa: E402
from learn.meta import model_preview as MP   # noqa: E402
from learn.meta import mpc_constrained as MC   # noqa: E402
from learn.meta import mpc_learned as ML   # noqa: E402
from studies import train_mem as TM   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache", "onestep_gz")
K0, STRIDE, L_CTX, N_SAMP = 40, 16, 256, 128
CH = ("surge", "sway", "yaw", "heave", "pitch")


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def parse_missions(s):
    """'0-3' -> [(0, 0), (0, 1), ..., (3, 1)] (seeds x legs 0, 1)."""
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        for seed in range(int(a), int(b or a) + 1):
            out += [(seed, 0), (seed, 1)]
    return out


# ------------------------------------------------------------ record
# --record-sea jitter: the task's spectrum (Hs 1.0 m, Tp 5 s, spreading 8,
# waves from WAVES_FROM) synthesised as the training seas are (data2.sea_for:
# 48 jittered frequencies x 8 jittered directions) instead of the task's
# line spectrum (24 x 5 fixed components, the same in every mission, which
# repeats every 46 s at a fixed point). Seeds >= JITTER_SEED0 only, so the
# two kinds of mission never share a key.
JITTER_SEED0 = 100
JITTER_SEA = dict(hs=1.0, tp=5.0, theta0=float(np.pi), n_freq=48, n_dir=8,
                  jitter=True, spread_s=8.0)
# --record-sea jitter_dir: as jitter, and the mean wave direction drawn per
# seed (uniform, stream [seed, 0xD1]; both legs of a seed share it), so the
# boat's heading no longer tells the relative wave direction. Seeds >= 200.
JITTER_DIR_SEED0 = 200


def sea_of(kind, seed):
    if kind == "task":
        return None
    if kind == "jitter":
        return dict(JITTER_SEA)
    th = float(np.random.default_rng([seed, 0xD1]).uniform(0, 2 * np.pi))
    return dict(JITTER_SEA, theta0=th)


def record(seed, leg, T, sea=None):
    """final_eval.run_mission's hand loop, keeping the raw stream."""
    from studies.sim2real_jet import theta, z_hand
    env = ML.env()
    w, floor = theta(z_hand())
    th0 = None if sea is None else float(sea["theta0"])
    m = FE.make_mission(seed, leg, T, weights=w, floor=floor,
                        sea=None if sea is None else dict(sea))
    ML.check_env(m, env)
    geo = FE.geometry(m, env)
    rec = FE.Recorder(m, env)
    m.residual = rec
    safe = MC.SafeMode(m.dt * m.sub)
    xs, ts, us, apks, hmins, n_safe = [m.s.copy()], [m.t], [], [], [], 0
    while not m.done():
        thrust, rud = m.mpc_command()
        thrust, rud, on = safe(m.s[3], thrust, rud)
        s0, n0 = m.s.copy(), len(rec.S)
        m.advance(thrust, rud)
        if not m.finite or getattr(m.plant, "capsized", False):
            break
        apk, hmin = FE.step_safety(rec, n0, s0, geo, m.sub)
        us.append((thrust / m.t_max, rud / m.rud_max))
        xs.append(m.s.copy())
        ts.append(m.t)
        apks.append(apk)
        hmins.append(hmin)
        n_safe += int(on)
    sea = {a: np.array(getattr(m.sea, a), float)
           for a in ("a", "k", "th", "w", "phi")}
    met = m.metrics()
    return dict(seed=seed, leg=leg, n=len(us), n_ctrl=m.n_ctrl,
                sea_kind="task" if sea is None else "jitter",
                theta0=th0,
                xs=np.array(xs), t=np.array(ts), u=np.array(us, float),
                apk=np.array(apks), hmin=np.array(hmins), sea=sea,
                consts=ML.mission_consts(m), n_safe=n_safe,
                kn=met["u_along"] / FE.KN)


def streams_path(T):
    return os.path.join(CACHE, f"streams_T{T:g}.pkl")


def phase_record(missions, T, sea_kind="task"):
    os.makedirs(CACHE, exist_ok=True)
    p = streams_path(T)
    S = pickle.load(open(p, "rb")) if os.path.exists(p) else {}
    for key in missions:
        if key in S:
            continue
        t0 = time.time()
        lo, hi = dict(task=(0, JITTER_SEED0),
                      jitter=(JITTER_SEED0, JITTER_DIR_SEED0),
                      jitter_dir=(JITTER_DIR_SEED0, 10 ** 9))[sea_kind]
        if not lo <= key[0] < hi:
            raise SystemExit(f"mission {key}: {sea_kind} missions use seeds "
                             f"in [{lo}, {hi})")
        S[key] = record(*key, T, sea=sea_of(sea_kind, key[0]))
        log(f"recorded s{key[0]}_l{key[1]}: {S[key]['n']} steps, "
            f"{S[key]['kn']:.3f} kn, safe steps {S[key]['n_safe']}, "
            f"{time.time() - t0:.0f} s")
        pickle.dump(S, open(p + ".tmp", "wb"))
        os.replace(p + ".tmp", p)
    return S


def load_streams(missions, T):
    p = streams_path(T)
    S = pickle.load(open(p, "rb")) if os.path.exists(p) else {}
    miss = [k for k in missions if k not in S]
    if miss:
        raise SystemExit(f"missions {miss} not recorded at T {T:g} (run "
                         "--phase record with them)")
    return S


# ------------------------------------------------------------ live data
def new_live(st, stats, dev):
    """An empty LiveDataW of one recorded mission (final_eval.py:486-488)."""
    live = MC.LiveDataW(1, st["n_ctrl"] + 1, stats, dev, st["consts"],
                        [SimpleNamespace(**st["sea"])], ML.env())
    live.start(0, st["xs"][0], float(st["t"][0]))
    return live


def push(live, st, k):
    live.push(0, st["u"][k], st["xs"][k + 1], float(st["t"][k + 1]),
              float(st["apk"][k]), float(st["hmin"][k]))


def full_live(st, stats, dev):
    live = new_live(st, stats, dev)
    for k in range(st["n"]):
        push(live, st, k)
    return live


# ------------------------------------------------------------ rows
def load_row(name, dev):
    spec, why = FE.resolve_row(name)
    if spec is None or spec["kind"] != "net":
        raise SystemExit(f"row {name}: {why or 'not a net row'}")
    ck = torch.load(spec["ckpt"], map_location=dev, weights_only=False)
    head0 = None
    if spec["head"]:
        from learn.meta.safety_head import load_head
        head0, _ = load_head(spec["head"], dev)
    return spec, ck, head0


def build(spec, ck, head0, dev, live, first, learner_kw=None, state=None,
          attach=True):
    """net, head and (mode online / bayes, attach) the learner exactly as
    final_eval.run_mission builds them for a mission whose chain starts at
    `first` (seed, leg); learner_kw: extra constructor arguments (tau);
    state: a learner state to load. attach False: the pretrained net and
    head only (a learner row that has learned nothing is its prior)."""
    net, head = FE.build_net(spec, ck, head0, dev, *first)
    ctx = int(ck.get("ctx", L_CTX))
    fc = dict(lam=FE.OPT["fc_lam"], hp=FE.OPT["fc_hp"], msd=0.01)
    seed = ML.row_seed(first[0], first[1], 17)
    learner = None
    if not attach:
        pass
    elif spec["mode"] == "bayes":
        from learn.meta.bayes_adapter import BayesLearner
        learner = BayesLearner(net, live, 0, ctx, spec["variant"], head=head,
                               seed=seed, fc=fc, **(learner_kw or {}))
    elif spec["mode"] in ("online", "scratch"):
        from learn.meta.online_stream import OnlineLearner
        learner = OnlineLearner(net, live, 0, ctx, spec["variant"],
                                head=head, seed=seed,
                                scratch=spec["mode"] == "scratch",
                                **(learner_kw or {}))
    if learner is not None and state is not None:
        learner.load_state(state)
    return SimpleNamespace(net=net, head=head, learner=learner, ctx=ctx)


# ------------------------------------------------------------ scoring
def windows(n, L=L_CTX, k0=K0, stride=STRIDE):
    """(a0, lo, hi): windows [a0, a0 + L) scoring steps [lo, hi), every
    step in [k0, n) once; after the first window each scored step has >=
    L - stride steps of history."""
    if L > k0:
        out, lo = [(0, k0, min(n, L))], L
    else:                       # short windows: score from k0 on
        out, lo = [], k0
    while lo < n:
        hi = min(n, lo + stride)
        out.append((hi - L, lo, hi))
        lo = hi
    return out


# --eval-obs: score with the pretraining's observation noise on the error
# history (model3.obs_sd), as every training window has it; default off,
# the deployment case (exact recorded errors)
EVAL_OBS = False


@torch.no_grad()
def score_mission(v, live, st, variant, gen_seed, n_samp=N_SAMP):
    """Per-step arrays of one frozen row on one mission (module docstring).
    Three generators restart from gen_seed for every row: the token
    perturbations; the common random numbers (flow base noise, PIT
    jitter, the safety head's uniforms), drawn for every step 0..n-1 of
    the mission and indexed by the scored step, so rows of any variant or
    context length get the same numbers at the same step; the Bayes
    adapter draws. Steps whose truth is not finite are left out."""
    from learn.meta.safety_head import pinball, inv_cdf
    dev, net = live.dev, v.net
    L = min(L_CTX, v.ctx)
    win = windows(st["n"], L, max(K0, getattr(v, "k0", K0)),
                  getattr(v, "stride", STRIDE))
    a = torch.tensor([w[0] for w in win], device=dev)
    ii = torch.zeros_like(a)
    g = torch.Generator().manual_seed(gen_seed)
    gc = torch.Generator().manual_seed(gen_seed + 1)
    ga = torch.Generator().manual_seed(gen_seed + 2)
    n = int(st["n"])
    z0_all = torch.randn((n, n_samp, M.C7), generator=gc)
    u01_all = torch.rand((n, n_samp, 2), generator=gc, dtype=torch.float64)
    ur_all = torch.rand((n, M.C7), generator=gc)
    uy_all = torch.rand((n, 2), generator=gc, dtype=torch.float64)
    hp = 0 if variant == "w" else int(FE.OPT["fc_hp"])
    lam = 0.0 if variant == "w" else float(FE.OPT["fc_lam"])
    if variant == "a":
        tok, tgt, ok = M.window_tokens(live, ii, a, L)
    else:
        cfg = MP.fixed_cfg(live, a + L, lam, hp, 0.01, g)
        tok, tgt, ok = MP.tokens_p(live, ii, a, L, cfg, g, obs=EVAL_OBS)
    mem = getattr(v, "mem", None)               # a memory network's memory
    if mem is None:
        h = net.encode(tok)                                 # (nw, L, d)
    else:
        from learn.meta import memory_net as MN
        h = net.encode(tok, MN.expand(mem, len(win)))
    pos = a[:, None] + torch.arange(L, device=dev)[None]
    lo = torch.tensor([w[1] for w in win], device=dev)[:, None]
    hi = torch.tensor([w[2] for w in win], device=dev)[:, None]
    sel = ok & (pos >= lo) & (pos < hi)
    sel = sel & torch.isfinite(tgt).all(-1)
    hs, ys, ks = h[sel], tgt[sel], pos[sel]
    order = torch.argsort(ks)
    hs, ys, ks = hs[order], ys[order], ks[order]
    yq = live.Y[0, ks]
    N = len(ks)
    kc = ks.cpu()
    z0, u01 = z0_all[kc].to(dev), u01_all[kc].to(dev)
    ur, uy = ur_all[kc].to(dev), uy_all[kc].to(dev)
    ad = getattr(net, "bayes", None)
    out = {k: [] for k in ("se", "var", "pit", "crps", "pin", "pit_y")}
    for c0 in range(0, N, 512):
        hc = hs[c0:c0 + 512]
        yc = ys[c0:c0 + 512]
        nc = len(hc)
        hm = ad.apply_mean(hc) if ad is not None else hc
        e_m = net.sample_grad(hm[:, None].expand(nc, n_samp, -1),
                              z0[c0:c0 + nc])               # (nc, S, 10)
        if ad is not None:
            Bs = ad.draw(n_samp, ga)                        # (S, d, r)
            hh = hc[:, None] + torch.einsum("nr,sdr->nsd", hc @ ad.A.T, Bs)
            e_s = net.sample_grad(hh, z0[c0:c0 + nc])
        else:
            hh, e_s = hm[:, None].expand(nc, n_samp, -1), e_m
        mean = e_m.mean(1)
        out["se"].append((mean - yc) ** 2)
        out["var"].append(e_m.var(1))
        r = (e_s < yc[:, None]).sum(1).float()
        out["pit"].append((r + ur[c0:c0 + nc]) / (n_samp + 1))
        out["crps"].append(M.crps_samples(e_s, yc))
        if v.head is not None:
            yy = yq[c0:c0 + nc]
            fin = torch.isfinite(yy).all(-1)
            q = v.head(hm, yc, raw=False)
            yn = (yy - v.head.y_mu) / v.head.y_sd
            pin = pinball(q, yn, v.head.qs)                 # (nc, 2)
            out["pin"].append(torch.where(fin[:, None], pin,
                                          torch.full_like(pin, float("nan"))))
            qq = v.head(hh, e_s).double()                   # (nc, S, 2, nq)
            draws = inv_cdf(qq, u01[c0:c0 + nc])            # (nc, S, 2)
            rr = (draws < yy.double()[:, None]).sum(1).double()
            py = (rr + uy[c0:c0 + nc]) / (n_samp + 1)
            out["pit_y"].append(torch.where(fin[:, None], py,
                                            torch.full_like(py, float("nan"))))
    width = dict(se=M.C7, var=M.C7, pit=M.C7, crps=M.C7, pin=2, pit_y=2)
    res = {k: (torch.cat(x).double().cpu().numpy() if x else
               np.zeros((0, width[k])))
           for k, x in out.items() if x or k not in ("pin", "pit_y")}
    res["e"] = ys.double().cpu().numpy()
    res["k"] = ks.cpu().numpy()
    return res


def summarise(per, ref=None, n_samp=N_SAMP):
    """Pooled scores of per-mission step arrays; ref: the reference row's
    per-mission arrays (same missions) for the paired MSE ratios."""
    keys = sorted(per)
    cat = lambda name: np.concatenate([per[k][name] for k in keys])  # noqa
    mse = lambda d: (d["se"] - d["var"] / n_samp).sum(0)  # noqa: E731
    se = sum(mse(per[k]) for k in keys)
    e2 = sum((per[k]["e"] ** 2).sum(0) for k in keys)
    pit = cat("pit")
    rms = np.sqrt((cat("e") ** 2).mean(0)) + 1e-12
    c0 = M.gauss_crps_np(rms, cat("e")).sum(0)
    s = dict(skill=se / np.maximum(e2, 1e-12),
             cov90=((pit >= 0.05) & (pit <= 0.95)).mean(0),
             crps=cat("crps").sum(0) / np.maximum(c0, 1e-12), n=len(pit))
    if all("pin" in per[k] for k in keys):
        py = cat("pit_y")
        py = py[np.isfinite(py).all(1)]
        s.update(pin=np.nanmean(cat("pin"), 0),
                 cov90_y=((py >= 0.05) & (py <= 0.95)).mean(0),
                 above99_apk=(py[:, 0] > 0.99).mean(),
                 below01_hmin=(py[:, 1] < 0.01).mean())
    if ref is not None:
        rse = sum(mse(ref[k]) for k in keys)
        s["ratio"] = se / np.maximum(rse, 1e-12)
        pm = np.array([mse(per[k]) / np.maximum(mse(ref[k]), 1e-12)
                       for k in keys])
        s["ratio_missions"] = pm
        if "pin" in s and all("pin" in ref[k] for k in keys):
            s["pin_ratio"] = s["pin"] / np.nanmean(np.concatenate(
                [ref[k]["pin"] for k in keys]), 0)
    return s


def show(name, s):
    f = lambda x: " ".join(f"{v:6.3f}" for v in np.atleast_1d(x))  # noqa
    line = (f"  {name:28s} skill {f(s['skill'][:5])} | cov90 "
            f"{f(s['cov90'][:5])} | crps {f(s['crps'][:5])}")
    if "ratio" in s:
        pm = s["ratio_missions"][:, :5].mean(1)
        line += (f"\n  {'':28s} MSE / reference {f(s['ratio'][:5])} | mean "
                 f"{s['ratio'][:5].mean():.3f}, better in {(pm < 1).sum()}"
                 f"/{len(pm)} missions")
    if "pin" in s:
        line += (f"\n  {'':28s} safety head: pinball [APK HMIN] "
                 f"{f(s['pin'])}" + (f" (/ reference {f(s['pin_ratio'])})"
                                     if "pin_ratio" in s else "")
                 + f" | draws cov90 {f(s['cov90_y'])}, APK above q99 "
                 f"{s['above99_apk']:.4f}, HMIN below q01 "
                 f"{s['below01_hmin']:.4f} (nominal 0.90, 0.01, 0.01)")
    print(line, flush=True)


def evaluate(cands, held, S, dev, ref_name=None, first=(0, 0)):
    """cands: list of (label, spec, ck, head0, learner_kw, state) frozen;
    scored on the held missions; first: the (seed, leg) the states' chain
    started from (it seeds the adapters); returns {label: summary}."""
    lives, out, per_all = {}, {}, {}
    import hashlib

    def stats_key(st_):
        h = hashlib.sha1()
        for k_ in sorted(st_):
            h.update(k_.encode())
            h.update(np.ascontiguousarray(np.asarray(st_[k_])).tobytes())
        return h.hexdigest()
    short = [k for k in held if S[k]["n"] <= K0]
    if short:
        log(f"  missions with <= {K0} steps left out: {short}")
        held = [k for k in held if k not in short]
    for label, spec, ck, head0, kw, state in cands:
        sk = stats_key(ck["stats"])
        per = {}
        kw = dict(kw or {})
        win_l = kw.pop("window", None)
        stride = kw.pop("stride", STRIDE)
        kw = kw or None
        for key in held:
            st = S[key]
            if (sk, key) not in lives:
                lives[(sk, key)] = full_live(st, ck["stats"], dev)
            live = lives[(sk, key)]
            if kw and "net" in kw:                  # phase batch
                v = SimpleNamespace(net=kw["net"], head=None, learner=None,
                                    ctx=L_CTX)
            elif kw and "memnet" in kw:
                net_m, mem = kw["memnet"]
                v = SimpleNamespace(net=net_m, head=None, learner=None,
                                    ctx=L_CTX, mem=mem)
            else:
                v = build(spec, ck, head0, dev, live, first, learner_kw=kw,
                          state=state, attach=state is not None)
            if win_l:
                v.ctx = min(v.ctx, win_l)
            v.stride = stride
            per[key] = score_mission(v, live, st, spec["variant"],
                                     ML.row_seed(key[0], key[1], 43))
            del v
        per_all[label] = per
    ref = per_all.get(ref_name)
    for label in per_all:
        out[label] = summarise(per_all[label],
                               ref if label != ref_name else None)
        show(label, out[label])
    return out, per_all


# ------------------------------------------------------------ memory nets
@torch.no_grad()
def gz_memory(net, S, train, stats, dev, variant):
    """A memory network's memory after the given train missions, written
    segment by segment in mission order (studies/train_mem.py's segments,
    tokens all measured, no history noise: the deployment convention).
    Returns (memory, number of writes)."""
    from learn.meta import memory_net as MN
    mem = net.empty(1, dev)
    g = torch.Generator().manual_seed(4242)
    n_w = 0
    for key in train:
        st = S[key]
        live = full_live(st, stats, dev)
        for a, b in MN.segments(int(st["n"]), TM.SEG):
            tok, ok = TM.seg_tokens(live, [0], [a], TM.SEG, g, variant,
                                    obs=False)
            t = a + torch.arange(TM.SEG, device=dev)[None]
            mem = net.write(tok, ok & (t < b), mem)
            n_w += 1
        del live
    return mem, n_w


# ------------------------------------------------------------ batch
def batch_fit(ck, dev, S, train, val, part, steps, batch=32, lr=None,
              seed=0, check=100):
    """A copy of the checkpoint's net trained on random windows of the
    train missions under the pretraining loss and token convention
    (model_preview.train_tokens_p; window starts uniform over the
    mission), part 'head' (the flow head only) or 'full'; the weights at
    the check with the lowest flow-matching loss on the val missions are
    kept. Returns (net in eval mode, info)."""
    if part == "scratch":
        torch.manual_seed(seed)
        net = MP.NetP(**dict(ck.get("arch") or {})).to(dev)
    else:
        net = MP.load_variant(ck, dev)
    variant = ck.get("variant", "p")
    L = min(L_CTX, int(ck.get("ctx", L_CTX)))
    for q in net.parameters():
        q.requires_grad_(part in ("full", "scratch"))
    if part in ("head", "headfresh"):
        if part == "headfresh":
            torch.manual_seed(seed)
            for m in net.head:
                if isinstance(m, torch.nn.Linear):
                    m.reset_parameters()
        for q in net.head.parameters():
            q.requires_grad_(True)
    params = [q for q in net.parameters() if q.requires_grad]
    lr = lr or (1e-4 if part == "full" else 3e-4)
    opt = torch.optim.AdamW(params, lr, weight_decay=1e-4)
    lt = [(full_live(S[k], ck["stats"], dev), int(S[k]["n"])) for k in train]
    lv = [(full_live(S[k], ck["stats"], dev), int(S[k]["n"])) for k in val]

    def loss_on(lives, gen, B):
        tok, tgt, ok = batch_tokens(lives, gen, B, L, variant, dev)
        return M.fm_loss_h(net, net.encode(tok), tgt, ok, gen)

    @torch.no_grad()
    def val_loss():
        net.eval()
        out = float(np.mean([float(loss_on(lv, torch.Generator()
                                           .manual_seed(900 + i), batch))
                             for i in range(8)]))
        net.train()
        return out

    g = torch.Generator().manual_seed(seed)
    v0 = val_loss()
    best = dict(v=v0, it=-1, state=copy.deepcopy(net.state_dict()))
    net.train()
    for it in range(steps):
        loss = loss_on(lt, g, batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (it + 1) % check == 0 or it == steps - 1:
            v = val_loss()
            if v < best["v"]:
                best.update(v=v, it=it + 1,
                            state=copy.deepcopy(net.state_dict()))
    net.load_state_dict(best["state"])
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    del lt, lv
    return net, dict(val0=v0, val=best["v"], it=best["it"])


def batch_tokens(lives, gen, B, L, variant, dev):
    """B training windows spread over the missions [(live, n)]: tokens,
    targets and the loss mask (model_preview.train_tokens_p; window starts
    uniform over the mission; non-finite targets masked)."""
    mi = torch.randint(0, len(lives), (B,), generator=gen)
    toks, tgts, oks = [], [], []
    for j, (live, n) in enumerate(lives):
        cnt = int((mi == j).sum())
        if cnt == 0:
            continue
        a = torch.randint(0, max(0, n - L) + 1, (cnt,), generator=gen).to(dev)
        tok, tgt, ok, _ = MP.train_tokens_p(live, torch.zeros_like(a), a, L,
                                            gen, variant)
        fin = torch.isfinite(tgt).all(-1)
        toks.append(tok)
        tgts.append(torch.nan_to_num(tgt))
        oks.append(ok & fin)
    return torch.cat(toks), torch.cat(tgts), torch.cat(oks)


class HeadNet(torch.nn.Module):
    """A frozen trunk with another output head (phase batch): 'mse' a mean
    head trained on the squared error, its spread a constant per channel
    from the validation missions' residuals; 'gauss' mean and log-variance
    heads trained on the Gaussian negative log-likelihood. Both in
    normalised units (the scored space; the flow head works in asinh
    space), the same layer sizes as the flow head, sampled with the flow's
    base noise so the scoring is unchanged."""

    def __init__(self, base, kind):
        super().__init__()
        self.base, self.kind, self.d = base, kind, base.d
        self.mu = M.mlp(base.d, (512, 512, 512), M.C7)
        self.lv = (M.mlp(base.d, (512, 512, 512), M.C7) if kind == "gauss"
                   else None)
        self.register_buffer("sd", torch.ones(M.C7))

    def encode(self, tok):
        return self.base.encode(tok)

    def stats(self, h):
        mu = self.mu(h)
        if self.lv is None:
            return mu, self.sd.expand_as(mu)
        return mu, torch.exp(0.5 * self.lv(h).clamp(-12.0, 8.0))

    def sample_grad(self, h, z0, steps=24):
        mu, sd = self.stats(h)
        return mu + sd * z0


class EnsNet(torch.nn.Module):
    """Nets sharing one frozen trunk; the samples are split evenly over the
    members (the ensemble's predictive mixture)."""

    def __init__(self, nets):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self.d = nets[0].d

    def encode(self, tok):
        return self.nets[0].encode(tok)

    def sample_grad(self, h, z0, steps=24):
        out = torch.empty_like(z0)
        idx = torch.arange(z0.shape[-2], device=z0.device)
        for net, ix in zip(self.nets, idx.chunk(len(self.nets))):
            out[..., ix, :] = net.sample_grad(h[..., ix, :], z0[..., ix, :],
                                              steps)
        return out


class ScaledNet(torch.nn.Module):
    """A net's predictive samples widened about their mean by s (C7,), per
    channel: a post-hoc calibration of the spread."""

    def __init__(self, net, s):
        super().__init__()
        self.net, self.d = net, net.d
        self.register_buffer("s", torch.as_tensor(
            s, dtype=torch.float32, device=next(net.parameters()).device))

    def encode(self, tok):
        return self.net.encode(tok)

    def sample_grad(self, h, z0, steps=24):
        e = self.net.sample_grad(h, z0, steps)
        m = e.mean(-2, keepdim=True)
        return m + self.s * (e - m)


def head_fit(ck, dev, S, train, val, kind, steps, batch=32, lr=3e-4, seed=0,
             check=100):
    """HeadNet of kind 'mse' or 'gauss' on the checkpoint's frozen trunk,
    trained on the train missions (batch_tokens), the check with the lowest
    validation loss kept; 'mse' then gets its per-channel spread from the
    validation residuals. Returns (net, info)."""
    base = MP.load_variant(ck, dev)
    for q in base.parameters():
        q.requires_grad_(False)
    torch.manual_seed(seed)
    net = HeadNet(base, kind).to(dev)
    variant = ck.get("variant", "p")
    L = min(L_CTX, int(ck.get("ctx", L_CTX)))
    lt = [(full_live(S[k], ck["stats"], dev), int(S[k]["n"])) for k in train]
    lv = [(full_live(S[k], ck["stats"], dev), int(S[k]["n"])) for k in val]
    params = [q for q in net.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr, weight_decay=1e-4)

    def loss_on(lives, gen, B):
        tok, tgt, ok = batch_tokens(lives, gen, B, L, variant, dev)
        with torch.no_grad():
            h = base.encode(tok)
        mu, sd = net.stats(h)
        if kind == "mse":
            per = ((mu - tgt) ** 2).mean(-1)
        else:
            per = (torch.log(sd) + 0.5 * ((mu - tgt) / sd) ** 2).mean(-1)
        return (per * ok).sum() / ok.sum().clamp(min=1)

    @torch.no_grad()
    def val_loss():
        return float(np.mean([float(loss_on(lv, torch.Generator()
                                            .manual_seed(900 + i), batch))
                              for i in range(8)]))

    g = torch.Generator().manual_seed(seed)
    best = dict(v=float("inf"), it=0, state=None)
    for it in range(steps):
        loss = loss_on(lt, g, batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (it + 1) % check == 0 or it == steps - 1:
            v = val_loss()
            if v < best["v"]:
                best.update(v=v, it=it + 1,
                            state=copy.deepcopy(net.state_dict()))
    net.load_state_dict(best["state"])
    if kind == "mse":
        with torch.no_grad():
            se, cnt = torch.zeros(M.C7, device=dev), 0.0
            for i in range(8):
                tok, tgt, ok = batch_tokens(lv, torch.Generator()
                                            .manual_seed(900 + i), batch, L,
                                            variant, dev)
                mu, _ = net.stats(base.encode(tok))
                se += (((mu - tgt) ** 2) * ok[..., None]).sum((0, 1))
                cnt += float(ok.sum())
            net.sd.copy_(torch.sqrt(se / max(cnt, 1.0)))
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    return net, dict(val=best["v"], it=best["it"])


def calibrate(net, S, val, stats, dev, variant,
              grid=(1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)):
    """Per-channel spread factors (C7,): for each channel the smallest grid
    value whose 90 % coverage on the validation missions reaches 0.90 (the
    largest if none does). Returns (factors, validation coverage at them)."""
    lives = {k: full_live(S[k], stats, dev) for k in val}
    s = np.full(M.C7, grid[-1])
    cov_at = np.zeros(M.C7)
    done = np.zeros(M.C7, bool)
    for f in grid:
        v = SimpleNamespace(net=ScaledNet(net, np.full(M.C7, f)), head=None,
                            learner=None, ctx=L_CTX)
        per = {k: score_mission(v, lives[k], S[k], variant,
                                ML.row_seed(k[0], k[1], 47)) for k in val}
        cov = summarise(per)["cov90"]
        new = (~done) & (cov >= 0.90)
        s[new], cov_at[new] = f, cov[new]
        done |= new
        cov_at[~done] = cov[~done]
        if done.all():
            break
    return s, cov_at


def ttt_fit(ck, dev, live, n_train, part, steps, L=128, batch=32, seed=0):
    """A copy of the net trained on windows of L tokens lying inside the
    first n_train steps of one mission (live), a fixed number of steps."""
    net = MP.load_variant(ck, dev)
    variant = ck.get("variant", "p")
    for q in net.parameters():
        q.requires_grad_(part == "full")
    if part == "head":
        for q in net.head.parameters():
            q.requires_grad_(True)
    params = [q for q in net.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, 1e-4 if part == "full" else 3e-4,
                            weight_decay=1e-4)
    L = min(L, n_train)
    g = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(steps):
        a = torch.randint(0, n_train - L + 1, (batch,), generator=g).to(dev)
        tok, tgt, ok, _ = MP.train_tokens_p(live, torch.zeros_like(a), a, L,
                                            g, variant)
        ok = ok & torch.isfinite(tgt).all(-1)
        loss = M.fm_loss_h(net, net.encode(tok), torch.nan_to_num(tgt), ok,
                           g)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    return net


def phase_ttt(args, S, train, held, spec, ck, dev):
    """Test-time training inside a mission (module docstring, 'ttt')."""
    splits = [int(x) for x in args.ttt_split.split(",") if x]
    steps_l = [int(x) for x in args.ttt_steps.split(",") if x]
    parts = [x for x in args.ttt_parts.split(",") if x]
    cross = None
    if args.ttt_cross:
        val = parse_missions(args.batch_val)
        pool = [k for k in train if k not in val]
        cross = batch_fit(ck, dev, S, pool[:args.ttt_cross], val, "full",
                          args.batch_steps)[0]
    frozen = MP.load_variant(ck, dev)
    for K in splits:
        per = {}
        for key in held:
            st = S[key]
            live = full_live(st, ck["stats"], dev)
            seed = ML.row_seed(key[0], key[1], 43)
            rows = [("frozen", frozen)]
            if cross is not None:
                rows.append((f"cross-mission full {args.ttt_cross}", cross))
            for part in parts:
                for nst in steps_l:
                    rows.append((f"ttt {part} {nst} steps",
                                 ttt_fit(ck, dev, live, K, part, nst)))
            for lab, net in rows:
                v = SimpleNamespace(net=net, head=None, learner=None,
                                    ctx=L_CTX, k0=K)
                per.setdefault(lab, {})[key] = score_mission(
                    v, live, st, spec["variant"], seed)
            del live
        log(f"first {K} steps of each held-out mission learned, steps >= "
            f"{K} scored")
        ref = per["frozen"]
        for lab in per:
            show(f"{lab} (K {K})", summarise(per[lab],
                                             ref if lab != "frozen" else None))


# ------------------------------------------------------------ online
def learn_chain(spec, ck, head0, train, S, dev, learner_kw, passes=1):
    """The learner over the train missions in order, carried (final_eval's
    carry rows: a fresh net per mission seeded by the chain's first
    mission, then load_state); returns the final state."""
    state, first = None, train[0]
    for p in range(passes):
        for key in train:
            st = S[key]
            live = new_live(st, ck["stats"], dev)
            v = build(spec, ck, head0, dev, live, first, learner_kw, state)
            for k in range(st["n"]):
                push(live, st, k)
                v.learner.step(k)
            state = copy.deepcopy(v.learner.state())
            del v, live
        log(f"  {spec['name']} {learner_kw or ''}: pass {p + 1} over "
            f"{len(train)} missions done")
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="eval",
                    choices=("record", "eval", "online", "batch", "ttt"))
    ap.add_argument("--rows", default="cat_p_prior")
    ap.add_argument("--ref", default="cat_p_prior",
                    help="reference row for the paired MSE ratios")
    ap.add_argument("--train", default="0-3")
    ap.add_argument("--held", default="4-7")
    ap.add_argument("--T", type=float, default=120.0)
    ap.add_argument("--eval-obs", action="store_true",
                    help="eval: observation noise on the error history as in "
                         "pretraining")
    ap.add_argument("--record-sea", default="task",
                    choices=("task", "jitter", "jitter_dir"),
                    help="record: the task's line-spectrum sea or the same "
                         "spectrum jittered as in training (seeds >= 100)")
    ap.add_argument("--tau", default="",
                    help="online rows: comma list of the pull sd TAU "
                         "(online_stream default 0.02)")
    ap.add_argument("--bayes-tau", default="",
                    help="bayes rows: comma list of the adapter's initial "
                         "tau (bayes_adapter default 0.05; 1.0 gives the "
                         "documented prior width, ROADMAP section 11.1)")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--memnets", default="",
                    help="memory networks (studies/train_mem.py): comma "
                         "list of label=path; the memory is written from "
                         "the train missions in order, then the held-out "
                         "missions are scored with it (ROADMAP 11.3)")
    ap.add_argument("--mem-missions", type=int, default=2,
                    help="memory networks: write the memory from the last "
                         "N train missions (0: all). These missions have 4 "
                         "segments each, training at most 9 writes: 2 (8 "
                         "writes) stays in the trained range, 3 is 12")
    ap.add_argument("--ckpts", default="",
                    help="extra frozen candidates by checkpoint: comma list "
                         "of label=path (e.g. a wider net, ROADMAP 11.2); "
                         "scored like the reference row, without a safety "
                         "head")
    ap.add_argument("--batch-parts", default="head,full",
                    help="batch: what is trained (head, full)")
    ap.add_argument("--batch-sizes", default="2,4,6",
                    help="batch: numbers of train missions to train on")
    ap.add_argument("--batch-val", default="3",
                    help="batch: seeds of the validation missions (taken "
                         "out of --train)")
    ap.add_argument("--batch-steps", type=int, default=1500)
    ap.add_argument("--batch-heads", default="",
                    help="batch: other learners on the largest size: "
                         "headfresh, mse, gauss, ens<K>, scratch")
    ap.add_argument("--calibrate", action="store_true",
                    help="batch: add a spread-calibrated copy of each "
                         "trained row (factors from the validation "
                         "missions)")
    ap.add_argument("--windows", default="",
                    help="also score rows with these window lengths (one "
                         "scored step per window), e.g. 1,16,64,256")
    ap.add_argument("--windows-rows", default="",
                    help="the labels (comma list) --windows applies to; "
                         "default all rows")
    ap.add_argument("--ttt-split", default="120,240",
                    help="ttt: steps of each mission learned before scoring")
    ap.add_argument("--ttt-steps", default="100,300")
    ap.add_argument("--ttt-parts", default="head,full")
    ap.add_argument("--ttt-cross", type=int, default=0,
                    help="ttt: also a copy trained on this many train "
                         "missions (as in batch), for comparison")
    ap.add_argument("--save", default="", help="pickle the summaries here")
    args = ap.parse_args()
    FE.out_dir = lambda: CACHE           # nothing of final_eval writes
    FE.log = log
    global EVAL_OBS
    EVAL_OBS = bool(args.eval_obs)
    train, held = parse_missions(args.train), parse_missions(args.held)
    if args.phase == "record":
        phase_record(train + held, args.T, args.record_sea)
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = load_streams(held + (train if args.phase in ("online", "batch",
                                                     "ttt")
                             or args.memnets else []),
                     args.T)
    rows = list(dict.fromkeys(r for r in args.rows.split(",") if r))
    loaded = {}
    cands = []
    for name in sorted(set(rows + ([args.ref] if args.ref else []))):
        loaded[name] = load_row(name, dev)
        mode = loaded[name][0]["mode"]
        if mode != "prior" and (args.phase in ("eval", "batch", "ttt")
                                or name == args.ref):
            raise SystemExit(
                f"row {name}: a learner row is only meaningful after it has "
                "learned (--phase online), and the reference must be a "
                "prior row")
    if args.phase == "ttt":
        phase_ttt(args, S, train, held, loaded[args.ref][0],
                  loaded[args.ref][1], dev)
        return
    if args.ref:
        cands.append((args.ref, *loaded[args.ref], None, None))
    t0 = time.time()
    for name in rows:
        if name == args.ref:
            continue
        spec, ck, head0 = loaded[name]
        if args.phase in ("eval", "batch") or spec["mode"] == "prior":
            cands.append((name, spec, ck, head0, None, None))
            continue
        arg = args.bayes_tau if spec["mode"] == "bayes" else args.tau
        taus = [float(x) for x in arg.split(",") if x] or [None]
        for tau in taus:
            kw = dict(tau=tau) if tau is not None else None
            state = learn_chain(spec, ck, head0, train, S, dev, kw,
                                args.passes)
            label = f"{name}" + (f" tau {tau:g}" if tau is not None else "") \
                + (f" x{args.passes}" if args.passes > 1 else "")
            cands.append((label, spec, ck, head0, kw, state))
    for item in [x for x in args.memnets.split(",") if x]:
        label, _, path = item.partition("=")
        if not path or not os.path.exists(path) or not args.ref:
            raise SystemExit(f"--memnets {item}: no such checkpoint (or no "
                             "--ref)")
        net_m, st_m, ck_m = TM.load_memnet(path, dev)
        if st_m.get("seg", TM.SEG) != TM.SEG:
            raise SystemExit(f"--memnets {item}: trained with SEG "
                             f"{st_m.get('seg')}, this code writes {TM.SEG}")
        src = train[-args.mem_missions:] if args.mem_missions else train
        mem, n_w = gz_memory(net_m, S, src, ck_m["stats"], dev,
                             st_m["variant"])
        spec = dict(loaded[args.ref][0], name=label,
                    ckpt=st_m["base_ckpt"], head=None, mode="prior",
                    variant=st_m["variant"])
        cands.append((label, spec, ck_m, None, dict(memnet=(net_m, mem)),
                      None))
        # the same net with its memory left empty: the paired control
        if net_m.kind != "none":
            cands.append((label + " empty", spec, ck_m, None,
                          dict(memnet=(net_m, net_m.empty(1, dev))), None))
        m0 = mem[0]
        log(f"  {label}: memory written from the last {len(src)} train "
            f"missions ({n_w} writes; memory "
            f"{'none' if m0 is None else tuple(m0.shape)})")
    for item in [x for x in args.ckpts.split(",") if x]:
        label, _, path = item.partition("=")
        if not path or not os.path.exists(path):
            raise SystemExit(f"--ckpts {item}: no such checkpoint")
        if not args.ref:
            raise SystemExit("--ckpts needs --ref (its row supplies the "
                             "family and variant conventions)")
        ck = torch.load(path, map_location=dev, weights_only=False)
        spec = dict(loaded[args.ref][0], name=label, ckpt=path, head=None,
                    mode="prior", variant=ck.get("variant", "p"))
        cands.append((label, spec, ck, None, None, None))
    if args.phase == "batch":
        val = parse_missions(args.batch_val)
        pool = [k for k in train if k not in val]
        bases = [(c[0], c[1], c[2]) for c in list(cands)]
        for label, spec, ck in bases:
            for part in [x for x in args.batch_parts.split(",") if x]:
                for n in [int(x) for x in args.batch_sizes.split(",") if x]:
                    if n > len(pool):
                        raise SystemExit(f"--batch-sizes {n}: only "
                                         f"{len(pool)} train missions")
                    tb = time.time()
                    net, info = batch_fit(ck, dev, S, pool[:n], val, part,
                                          args.batch_steps)
                    lab = f"{label} {part} {n} missions"
                    log(f"  {lab}: validation flow-matching loss "
                        f"{info['val0']:.4f} -> {info['val']:.4f} (step "
                        f"{info['it']}), {time.time() - tb:.0f} s")
                    cands.append((lab, dict(spec, name=lab, head=None,
                                            mode="prior"), ck, None,
                                  dict(net=net), None))
            n = max(int(x) for x in args.batch_sizes.split(",") if x)
            for kind in [x for x in args.batch_heads.split(",") if x]:
                tb = time.time()
                if kind in ("mse", "gauss"):
                    net, info = head_fit(ck, dev, S, pool[:n], val, kind,
                                         args.batch_steps)
                elif kind.startswith("ens"):
                    nets = [batch_fit(ck, dev, S, pool[:n], val, "head",
                                      args.batch_steps, seed=j)[0]
                            for j in range(int(kind[3:]))]
                    net, info = EnsNet(nets), dict(val=float("nan"), it=-1)
                else:                                 # headfresh, scratch
                    net, info = batch_fit(ck, dev, S, pool[:n], val, kind,
                                          max(args.batch_steps, 3000)
                                          if kind == "scratch"
                                          else args.batch_steps)
                lab = f"{label} {kind} {n} missions"
                log(f"  {lab}: validation loss {info['val']:.4f} (step "
                    f"{info['it']}), {time.time() - tb:.0f} s")
                cands.append((lab, dict(spec, name=lab, head=None,
                                        mode="prior"), ck, None,
                              dict(net=net), None))
        if args.calibrate:
            for c in [c for c in cands if c[4] and "net" in c[4]]:
                sc, cv = calibrate(c[4]["net"], S, val, c[2]["stats"], dev,
                                   c[1]["variant"])
                log(f"  {c[0]}: spread factors {np.round(sc[:5], 2)} "
                    f"(validation coverage {np.round(cv[:5], 3)})")
                cands.append((c[0] + " calibrated", c[1], c[2], None,
                              dict(net=ScaledNet(c[4]["net"], sc)), None))
    if args.windows:
        sel = [x for x in args.windows_rows.split(",") if x]
        for c in list(cands):
            if sel and c[0] not in sel:
                continue
            for w in [int(x) for x in args.windows.split(",") if x]:
                cands.append((f"{c[0]} history {w - 1}", c[1], c[2], c[3],
                              dict(c[4] or {}, window=w, stride=1), c[5]))
    log(f"learning {time.time() - t0:.0f} s; scoring {len(cands)} rows on "
        f"{len(held)} held-out missions")
    t0 = time.time()
    out, _ = evaluate(cands, held, S, dev, ref_name=args.ref or None,
                      first=train[0])
    log(f"scoring {time.time() - t0:.0f} s")
    if args.save:
        pickle.dump(dict(args=vars(args), summaries=out),
                    open(args.save, "wb"))


if __name__ == "__main__":
    main()
