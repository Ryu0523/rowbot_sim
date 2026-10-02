#!/usr/bin/env python3
"""
"As fast as possible under the safety constraints" (ROADMAP_2026-09-30
section 9; DEFECTS M17 for why the weighted impact price is gone): an MPPI
over thrust knots whose objective is the mean along-track progress over the
horizon and whose safety terms are constraints, computed from the sampled
futures of a learned one-step error model (or of the e = 0 model, row m0).

Structure kept from learn/meta/mpc_learned.LearnedMPPI (read-only import):
7 thrust knots over the 24-step horizon, warm start shifted by one step,
softmax weights (lam 0.6), knot noise 0.25, 128 candidates, the nozzle set
by the heading autopilot in the rollouts (SteerT) and on the boat
(steer='auto'; steer='free' below), common
random numbers (one base draw per row and call, shared by all candidates),
the per-row random streams (seed, leg, 5) knots / (seed, leg, 7) flow base
noise; new streams (seed, leg, 9) for the safety-quantity draws and
(seed, leg, 11) for the wave hook. ImpactLink / task_cost are not used.

Per candidate c, with S sampled futures (worst-alpha mean = the mean of the
worst ceil(alpha S) futures, alpha = 0.1; at S <= 10 the worst one):

  speed   H dtc - mean_s(progress_s) / u_ref (progress = along-track
          distance at the horizon end minus now; no cap at u_ref)
  g_a110  k A1/10 / 3.0 g - 1: per future, A1/10 = mean of the top 10% of
          the peaks >= 0.3 g in {measured per-step peaks of the past 60 s}
          U {the future's per-step peaks}, all x k; worst-alpha mean
  g_apk   k max_{s,j} peak / 7.0 g - 1 (the worst sampled future)
  g_bow   (F_b / 4 - h) / (F_b / 4), h = worst-alpha mean of the futures'
          horizon minimum bow height
  g_yaw   max_j |r| / r_max(u) - 1, worst-alpha mean; r_max = the steady
          full-nozzle turning rate of the boat at the MEASURED speed (a
          speed the plan cannot move), from a turning run on the plant
  g_y     max_j |y| / (2 L) - 1, worst-alpha mean
  cost    speed + W_JUMP sum (dU)^2 + sum_g [g > 0] (BIG + W_G g^2)

steer='free': the planner owns the heading. Every candidate also carries 7
nozzle knots (fractions in [-1, 1], knot noise SIGMA_N, warm start shifted
like the thrust's, the first plan straight at 0); the rollouts take the
plan's nozzle column as it is (no autopilot) and the boat gets the
weighted plan's first nozzle command (diag 'nozzle'). Nothing else
changes: the same objective (along-track progress; a heading off the track
counts only its cos share), the same constraints (yaw rate <= r_max(u),
cross-track <= y_lengths L, accelerations, bow), W_JUMP on the nozzle
jumps as on the thrust's. The heading in the rollouts then comes from the
error model's yaw predictions alone (no autopilot pulling it back inside
the horizon); replanning every step is the only feedback.

g <= 0: no penalty. BIG = 100 exceeds any speed difference (the speed term
spans about -3 .. 6). W_G = BIG as well: when every candidate violates,
BIG cancels in the softmax and the ordering is by g^2 (the roadmap's
stated intent, "only the g^2 ordering remains"); an unscaled g^2 would be
outweighed by the speed term's spread. Every g is clipped at 10 and a
non-finite sample is a maximal violation (a runaway draw is shared by all
candidates and must not decide between them). The same horizon (24 steps)
for every term.

Per-step peaks and bow heights per sampled future:
  head    (meta6 / meta7 nets) learn/meta/safety_head: quantiles of
          (APK, HMIN) from the trunk output and the sampled error of the
          step, one draw per (future, step) by the inverse CDF from the
          row's stream (shared by all candidates), widened about the median
          by the mismatch inflation
  proxy   (meta5 nets, m0) peak = the step-mean heave acceleration of the
          rolled states, (zd_{j+1} - zd_j) / dtc (upward +, a lower bound
          of the step's peak); bow height = F_b + z_j + s_p x_b th_j
          - eta_bow,j at the end of step j (roll 0), with eta_bow,j:
            p nets: the forecast (model_preview.fc_eval with the call's
                    forecast-error draw) at the rolled end-of-step pose,
                    highest of the three bow points, while j < hp;
            otherwise (and beyond hp): the statistical relative-motion
                    form -- a narrow-band Gaussian wave at the bow with the
                    RMS and mean zero-crossing frequency measured over the
                    past 60 s (Rayleigh amplitude, uniform phase: the
                    Rayleigh exceedance statistics NORDFORSK / STANAG use),
                    max(16 / S, 1) draws per sampled future

Also here: LiveDataW (mpc_learned.LiveData plus the wave columns
model_preview's nets read, the per-step safety truths and the measured
elevations), rollout_tap (model_preview._rollout_core_w line for line,
also returning the trunk outputs), SafeMode (the measured-roll protection,
not part of planning) and r_max interpolation.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from learn.meta import model3 as M
from learn.meta import mpc_learned as ML
from learn.meta.episode5 import mid_pose, station_xy
from learn.repro.task import G

H = M.HB
K_CAND, N_KNOTS, SIGMA, LAM, W_JUMP = (ML.K_CAND, ML.N_KNOTS, ML.SIGMA,
                                       ML.LAM, ML.W_JUMP)
SIGMA_N = 0.25          # nozzle knot noise (fraction of full lock), free
                        # steering; the thrust's value [assumption]
STEER_MODES = ("auto", "free")
S_DEFAULT = 8
S_BOW = 16          # wave draws per call for the statistical bow form


@dataclass
class Limits:
    """Section 9 limits (on k x simulated accelerations)."""
    k: float = 2.0
    a110: float = 3.0           # g, A1/10 running limit
    apk: float = 7.0            # g, single-peak limit
    peak_min: float = 0.3       # g, smaller maxima are not peaks
    win_s: float = 60.0         # past window of the A1/10, s
    bow_frac: float = 0.25      # of the static bow freeboard
    y_lengths: float = 2.0      # cross-track limit in boat lengths
    alpha: float = 0.1          # worst share of the sampled futures
    big: float = 100.0
    w_g: float = 100.0
    g_clip: float = 10.0
    names: tuple = field(default=("a110", "apk", "bow", "yaw", "y"))


# ------------------------------------------------------------ live data
def sea_array(sea):
    """(NC, 6) float32 [a, k, cos th, sin th, w, phi] of a SeaState (the
    layout of model_preview.sea_arrays)."""
    a = np.atleast_1d(np.asarray(sea.a, float))
    th = np.atleast_1d(np.asarray(sea.th, float))
    return np.stack([a, np.atleast_1d(np.asarray(sea.k, float)), np.cos(th),
                     np.sin(th), np.atleast_1d(np.asarray(sea.w, float)),
                     np.atleast_1d(np.asarray(sea.phi, float))],
                    1).astype(np.float32)


def sea_eval(sea, X, Y, t):
    from learn.meta.data5 import sea_eval as se
    return se(sea, X, Y, t)


class LiveDataW(ML.LiveData):
    """mpc_learned.LiveData plus what model_preview.wave_cols / WaveRoll
    read off a DataP (Wm, Wmid normalised by the checkpoint's w_mu / w_sd,
    PM the dead-reckoned mid-step poses, SEA, x_st, y_off, dtc, h), the
    raw measured elevations W15 at the 15 stations at t_k and WMID15, and
    the per-step safety truths APK (m/s^2) / HMIN (m) of the plant (Y,
    tensor (n, T, 2), for the online head). Wave arrays at step k are set
    from the measured plant state at k (start / push). Stats without w_mu
    (variant a nets): the normalised wave tensors stay zero."""

    def __init__(self, n, T, stats, dev, consts, seas, env):
        super().__init__(n, T, stats, dev, consts)
        T = self.T
        self.env = env
        self.seas = list(seas)
        self.x_st_np = np.asarray(env["x_st"], float)
        self.y_off_np = np.asarray(env["y_off"], float)
        self.dtc = float(env["dt"] * env["sub"])
        self.h = 0.5 * self.dtc
        self.W15 = np.zeros((n, T, 15), np.float32)
        self.WMID15 = np.zeros((n, T, 15), np.float32)
        self.APK = np.full((n, T), np.nan, np.float32)
        self.HMIN = np.full((n, T), np.nan, np.float32)
        f32 = dict(dtype=torch.float32, device=dev)
        self.Wm = torch.zeros(n, T, 15, **f32)
        self.Wmid = torch.zeros(n, T, 15, **f32)
        self.PM = torch.zeros(n, T, 3, **f32)
        self.Y = torch.zeros(n, T, 2, **f32)
        self.has_w = "w_mu" in stats
        nc = max(len(np.atleast_1d(s.a)) for s in self.seas)
        sea = np.zeros((n, nc, 6), np.float32)
        for i, s in enumerate(self.seas):
            q = sea_array(s)
            sea[i, :len(q)] = q
        self.SEA = torch.tensor(sea, **f32)
        self.x_st = torch.tensor(self.x_st_np, **f32)
        self.y_off = torch.tensor(self.y_off_np, **f32)
        if self.has_w:
            self.w_mu = torch.tensor(np.asarray(stats["w_mu"]), **f32)
            self.w_sd = torch.tensor(np.asarray(stats["w_sd"]), **f32)

    def _waves(self, b, k, xs, t):
        xs = np.asarray(xs, float)
        X, Y = station_xy(np.asarray(xs[0]), np.asarray(xs[1]),
                          np.asarray(xs[5]), self.x_st_np, self.y_off_np)
        w15 = sea_eval(self.seas[b], X, Y, t).reshape(15)
        xm, ym, pm = mid_pose(xs, "plant14", self.h)
        Xm, Ym = station_xy(np.asarray(xm), np.asarray(ym), np.asarray(pm),
                            self.x_st_np, self.y_off_np)
        wmid = sea_eval(self.seas[b], Xm, Ym, t + self.h).reshape(15)
        self.W15[b, k], self.WMID15[b, k] = w15, wmid
        self.PM[b, k] = torch.tensor([xm, ym, pm], dtype=torch.float32)
        if self.has_w:
            mu, sd = self.stats["w_mu"], self.stats["w_sd"]
            self.Wm[b, k] = torch.from_numpy(
                ((w15 - mu) / sd).astype(np.float32)).to(self.dev)
            self.Wmid[b, k] = torch.from_numpy(
                ((wmid - mu) / sd).astype(np.float32)).to(self.dev)

    def start(self, b, xs0, t0=0.0):
        super().start(b, xs0)
        self._waves(b, 0, xs0, t0)

    def push(self, b, u, xs_next, t_next=None, apk=np.nan, hmin=np.nan):
        k = int(self.k[b])
        super().push(b, u, xs_next)
        self.APK[b, k], self.HMIN[b, k] = apk, hmin
        self.Y[b, k] = torch.tensor([apk, hmin], dtype=torch.float32)
        if t_next is None:
            t_next = (k + 1) * self.dtc
        self._waves(b, k + 1, xs_next, t_next)


# ------------------------------------------------------------ rollout
@torch.no_grad()
def rollout_tap(net, D, ii, k, plans, xs_k, env, S, base, steer=None,
                wave=None, nh=None, ode_steps=24):
    """model_preview._rollout_core_w line for line (no gradients, no
    observation noise, no feedback damping) that also returns the trunk
    outputs of the generated tokens: (e (B, P, S, H, 10), states (B, P, S,
    H, 10), actuators (B, P, S, H, 2), h (B, P, S, H, d)).
    studies/test_final.py checks it against rollout_core_w bit for bit."""
    from learn.meta.data3 import E7
    dev, st = D.dev, D.stats
    iit = torch.as_tensor(ii, device=dev).long()
    kt = torch.as_tensor(k, device=dev).long()
    pl = torch.as_tensor(plans, dtype=torch.float64, device=dev)
    B, P, Hh, _ = pl.shape
    NH = M.W_CTX - M.HB if nh is None else int(nh)
    assert Hh <= M.HB and 1 <= NH <= M.W_CTX - M.HB
    a = (kt - NH).clamp(min=0)
    Lh = kt - a
    tok_h, _, _ = M.window_tokens(D, iit, a, NH)
    if wave is not None:
        tok_h = torch.cat([tok_h, wave.history(D, iit, a, NH)], -1)
    hist, _ = M.kv_history(net, tok_h)
    hide = torch.arange(NH, device=dev)[None] >= Lh[:, None]
    e_prev = D.E[iit, (kt - 1).clamp(min=0)] * (kt >= 1)[:, None]
    e_prev = e_prev[:, None, None].expand(B, P, S, M.C7)
    xs = torch.as_tensor(xs_k, dtype=torch.float64, device=dev)
    amax = torch.tensor([env["t_max"], env["rud_max"]], dtype=torch.float64,
                        device=dev)
    act = (xs[:, 12:14] / amax)[:, None, None].expand(B, P, S, 2)
    sr = M.plant_to_reduced_t(xs)[:, None, None].expand(B, P, S, 10)
    base = base.to(dev)
    u_mu = torch.as_tensor(st["u_mu"], dtype=torch.float64, device=dev)
    u_sd = torch.as_tensor(st["u_sd"], dtype=torch.float64, device=dev)
    Un = ((pl - u_mu) / u_sd).float()
    e_sd = torch.as_tensor(st["e_sd"], device=dev)
    sel = torch.zeros(len(E7), 10, dtype=torch.float64, device=dev)
    sel[torch.arange(len(E7)), torch.tensor(E7)] = 1.0
    lo = torch.tensor([0.0, -1.0], dtype=torch.float64, device=dev)
    hi = torch.tensor([1.0, 1.0], dtype=torch.float64, device=dev)
    dtc = env["dt"] * env["sub"]
    m0 = M.model0t(env)
    n_st = len(E7)
    cache = [([], []) for _ in net.tf.layers]
    outs, states, acts, hs = [], [], [], []
    for j in range(Hh):
        X = M.state_tokens_t(sr, act, st, env).float()
        first = ((Lh + j) == 0).float().view(B, 1, 1).expand(B, P, S)
        Uj = pl[:, :, None, j]
        if steer is None:
            Un_j = Un[:, :, None, j].expand(B, P, S, 2)
        else:
            Uj = steer(j, sr, Uj)
            Un_j = ((Uj - u_mu) / u_sd).float()
        tok = M.make_tokens(X, Un_j, e_prev, first)
        if wave is not None:
            tok = torch.cat([tok, wave(j, sr, Uj).float()], -1)
        h = M.kv_step(net, tok, net.pos[Lh + j].view(B, 1, 1, -1), hist,
                      hide, cache)
        if getattr(net, "bayes", None) is not None:
            # learn/meta/bayes_adapter: future s keeps its own adapter
            # sample for the whole horizon
            h = net.bayes.apply(h)
        e = net.sample_grad(h, base[:, :, :, j], ode_steps)
        e_raw = (e * e_sd).double()
        sr = m0(sr, Uj) + (e_raw[..., :n_st] * dtc) @ sel
        act = torch.maximum(torch.minimum(
            Uj + e_raw[..., n_st:n_st + 2] * dtc, hi), lo)
        outs.append(e)
        states.append(sr)
        acts.append(act)
        hs.append(h)
        e_prev = e
    return (torch.stack(outs, 3), torch.stack(states, 3),
            torch.stack(acts, 3), torch.stack(hs, 3))


# ------------------------------------------------------------ constraints
def cvar_hi(x, alpha):
    """Mean of the largest ceil(alpha S) values along the last axis."""
    m = max(1, math.ceil(alpha * x.shape[-1] - 1e-9))
    return x.topk(m, -1).values.mean(-1)


def cvar_lo(x, alpha):
    return -cvar_hi(-x, alpha)


def a110(peaks, past, thr):
    """Per-future A1/10 (g): the mean of the top 10% (at least one) of the
    values >= thr among past (n_p,) and the future's peaks (..., H), both
    already scaled by k; 0 when there is no peak."""
    n_p = past.shape[-1]
    vals = torch.cat([past.to(peaks).expand(*peaks.shape[:-1], n_p), peaks],
                     -1)
    keep = vals >= thr
    n = keep.sum(-1)
    srt = torch.where(keep, vals, torch.zeros_like(vals)).sort(
        -1, descending=True).values
    m = torch.ceil(n.to(vals.dtype) / 10.0).clamp(min=1).long()
    top = srt.cumsum(-1).gather(-1, (m - 1)[..., None])[..., 0] / m
    return torch.where(n > 0, top, torch.zeros_like(top))


def constraint_cost(prog, peaks, bow, yaw, ytr, past, r_max, lim, fb, L,
                    u_ref, h_dtc):
    """Planner cost per candidate (module docstring) for nb rows:
    prog (nb, K, Sp) along-track progress per future (m); peaks (nb, K, Sa,
    H) per-step peak acceleration (g, simulated: k is applied here); bow
    (nb, K, Sb, H) bow height (m); yaw (nb, K, Sp, H) yaw rate (rad/s);
    ytr (nb, K, Sp, H) cross-track (m); past [nb tensors (n_p,)] measured
    per-step peaks of the past window (g, simulated); r_max (nb,) rad/s.
    Non-finite entries are maximal violations. Returns (cost (nb, K),
    parts {speed, g_<name>, pen})."""
    big = 1e6
    prog = torch.nan_to_num(prog, nan=0.0, posinf=0.0, neginf=0.0)
    peaks = torch.nan_to_num(peaks, nan=big, posinf=big, neginf=big)
    bow = torch.nan_to_num(bow, nan=-big, posinf=-big, neginf=-big)
    yaw = torch.nan_to_num(yaw, nan=big, posinf=big, neginf=big)
    ytr = torch.nan_to_num(ytr, nan=big, posinf=big, neginf=big)
    speed = h_dtc - prog.mean(-1) / u_ref
    pk = lim.k * peaks
    g = {}
    g["a110"] = torch.stack([
        cvar_hi(a110(pk[i], lim.k * past[i], lim.peak_min), lim.alpha)
        for i in range(pk.shape[0])]) / lim.a110 - 1.0
    g["apk"] = pk.amax((-2, -1)) / lim.apk - 1.0
    hb = fb * lim.bow_frac
    g["bow"] = (hb - cvar_lo(bow.amin(-1), lim.alpha)) / hb
    rm = torch.as_tensor(r_max, dtype=yaw.dtype, device=yaw.device).view(
        -1, 1)
    g["yaw"] = cvar_hi(yaw.abs().amax(-1), lim.alpha) / rm - 1.0
    g["y"] = cvar_hi(ytr.abs().amax(-1), lim.alpha) / (lim.y_lengths * L) \
        - 1.0
    pen = torch.zeros_like(speed)
    parts = dict(speed=speed)
    for nm in lim.names:
        gc = g[nm].to(speed.dtype).clamp(max=lim.g_clip)
        parts["g_" + nm] = gc
        pen = pen + torch.where(gc > 0, lim.big + lim.w_g * gc ** 2,
                                torch.zeros_like(gc))
    parts["pen"] = pen
    return speed + pen, parts


def bow_statistical(mu, sd, omega, S, Sw, Hh, dtc, gen):
    """eta at the bow (S * Sw, H) for one row: mu + A cos(omega t_j + phi),
    t_j = (j + 1) dtc, A Rayleigh with parameter sd, phi uniform; draws from
    gen (the row's stream), the same for every candidate."""
    n = S * Sw
    u1 = torch.rand(n, generator=gen, dtype=torch.float64).clamp(min=1e-12)
    ph = 2 * math.pi * torch.rand(n, generator=gen, dtype=torch.float64)
    A = sd * torch.sqrt(-2.0 * torch.log(u1))
    t = (torch.arange(Hh, dtype=torch.float64) + 1.0) * dtc
    return mu + A[:, None] * torch.cos(omega * t[None] + ph[:, None])


def r_max_of(table, u):
    """Steady full-nozzle yaw rate (rad/s) at surge speed u from a turning
    table {'u': [...], 'r': [...]} (linear, held at the ends)."""
    uu, rr = np.asarray(table["u"], float), np.asarray(table["r"], float)
    o = np.argsort(uu)
    return float(np.interp(u, uu[o], rr[o]))


# ------------------------------------------------------------ controller
class ConstrainedMPPI:
    """The constrained MPPI for B mission rows in lockstep (module
    docstring). net None: rollout_zero (the e = 0 model, S = 1, proxy
    quantities). head: a SafetyHead for the trunk (else proxy). variant:
    'a' (no wave hook), 'w' (measured elevations, no preview), 'p'
    (preview at forecast level fc['lam'] for fc['hp'] steps). geo:
    dict(fb, x_b, s_p, L, u_ref). Call with the live rows, the step k and
    per-row info dicts (past (n_p,) measured per-step peaks of the past
    window in g, u_meas, r_max, bow (mu, sd, omega), infl); returns the
    thrust commands (fractions) and diagnostics of the chosen plan (with
    steer='free' also the nozzle command, fraction of full lock, as diag
    'nozzle'; module docstring)."""

    def __init__(self, jobs, missions, en, lim, geo, net=None, live=None,
                 head=None, variant="a", nh=None, spread=1.0, K=K_CAND,
                 S=S_DEFAULT, sigma=SIGMA, lam=LAM, floor=0.0, fc=None,
                 steer="auto", sigma_n=SIGMA_N):
        if steer not in STEER_MODES:
            raise ValueError(f"steer {steer!r} not in {STEER_MODES}")
        self.steer, self.sigma_n = steer, float(sigma_n)
        self.jobs, self.ms, self.en, self.lim, self.geo = (jobs, missions,
                                                           en, lim, geo)
        self.net, self.live, self.head = net, live, head
        self.variant, self.nh = variant, nh
        self.B = len(jobs)
        self.K, self.S = K, (S if net is not None else 1)
        self.sigma, self.lam, self.floor = sigma, lam, floor
        self.spread = float(spread)
        self.fc = {**dict(lam=0.1, hp=H, msd=0.01), **(fc or {})}
        self.E, self.Sh = ML.knot_matrices()
        self.nominal = np.full((self.B, N_KNOTS), 0.5)
        self.nominal_n = np.zeros((self.B, N_KNOTS))       # steer 'free'
        self.rng = [np.random.default_rng([j["seed"], j["leg"], 5])
                    for j in jobs]
        self.gen = [torch.Generator().manual_seed(
            ML.row_seed(j["seed"], j["leg"], 7)) for j in jobs]
        self.gen_u = [torch.Generator().manual_seed(
            ML.row_seed(j["seed"], j["leg"], 9)) for j in jobs]
        self.gen_w = torch.Generator().manual_seed(
            ML.row_seed(jobs[0]["seed"], jobs[0]["leg"], 11))
        self.gen_b = torch.Generator().manual_seed(
            ML.row_seed(jobs[0]["seed"], jobs[0]["leg"], 23))
        self.dev = live.dev if net is not None else torch.device("cpu")
        self.dtc = en["dt"] * en["sub"]
        self.n_bad = np.zeros(self.B, int)
        self.n_samp = np.zeros(self.B, int)
        self.last = None

    # -------------------------------------------------------------- pieces
    def _wave(self, rows, k):
        if self.net is None or self.variant == "a":
            return None
        from learn.meta.model_preview import WaveRoll
        hp = 0 if self.variant == "w" else int(self.fc["hp"])
        lam = 0.0 if self.variant == "w" else float(self.fc["lam"])
        return WaveRoll(self.live, rows, [k] * len(rows), lam, hp,
                        self.fc["msd"], self.gen_w)

    def _forecast_bow(self, wave, st, rows, k):
        """eta at the bow (nb, K, S, H) forecast at the rolled end-of-step
        poses with the call's forecast-error draw (wave.cfg): the highest
        of the three bow points; NaN beyond the preview horizon."""
        from learn.meta.model_preview import fc_eval
        D = self.live
        nb, K, S, Hh, _ = st.shape
        x, y, psi = st[..., 0], st[..., 1], st[..., 7]
        c, s = torch.cos(psi)[..., None], torch.sin(psi)[..., None]
        yo = D.y_off.to(st.dtype)
        xb = float(self.geo["x_b"])
        X = (x[..., None] + xb * c - yo * s).float().reshape(nb, -1, 3)
        Y = (y[..., None] + xb * s + yo * c).float().reshape(nb, -1, 3)
        j = torch.arange(Hh, device=st.device)
        tm = ((k + j + 1).float() * D.dtc).view(1, 1, 1, Hh).expand(
            nb, K, S, Hh).reshape(nb, -1)
        lead = ((j + 1).float() * D.dtc).view(1, 1, 1, Hh).expand(
            nb, K, S, Hh).reshape(nb, -1)
        sea = D.SEA[torch.as_tensor(rows, device=D.dev)]
        _, val = fc_eval(sea, X, Y, tm, lead, wave.cfg, full=True)
        eta = val.amax(-1).view(nb, K, S, Hh).double()
        on = (j < int(self.fc["hp"])).view(1, 1, 1, Hh)
        return torch.where(on, eta, torch.full_like(eta, float("nan")))

    def _bow_from_states(self, st, eta):
        g = self.geo
        return g["fb"] + st[..., 3] + g["s_p"] * g["x_b"] * st[..., 5] - eta

    # ---------------------------------------------------------------- call
    def __call__(self, rows, k, info):
        nb, K, S = len(rows), self.K, self.S
        free = self.steer == "free"
        cand, cand_n, bases = [], [], []
        for b in rows:
            noise = self.rng[b].normal(0.0, 1.0, (K, N_KNOTS)) * self.sigma
            cand.append(np.clip(self.nominal[b][None] + noise, self.floor,
                                1.0))
            if free:
                nz = self.rng[b].normal(0.0, 1.0, (K, N_KNOTS)) * self.sigma_n
                cand_n.append(np.clip(self.nominal_n[b][None] + nz, -1.0,
                                      1.0))
            if self.net is not None:
                bases.append(torch.randn((S, H, M.C7), generator=self.gen[b]))
        cand = np.stack(cand)
        thr = cand @ self.E.T
        plans = np.zeros((nb, K, H, 2))
        plans[..., 0] = thr
        ms = [self.ms[b] for b in rows]
        xs = np.stack([m.s for m in ms])
        phi = np.array([m.phi for m in ms])
        if free:
            cand_n = np.stack(cand_n)
            plans[..., 1] = cand_n @ self.E.T
            steer = None
        else:
            steer = ML.SteerT(ms[0].ep, phi, [m.ep._psi_i for m in ms],
                              self.en)
        infl = np.array([float(info[i].get("infl", 1.0)) for i in range(nb)])
        wave = None
        if self.net is not None:
            base = torch.stack(bases)[:, None] * torch.as_tensor(
                self.spread * infl, dtype=torch.float32).view(nb, 1, 1, 1, 1)
            base = base.to(self.dev).expand(nb, K, S, H, M.C7)
            wave = self._wave(rows, k)
            if getattr(self.net, "bayes", None) is not None:
                self.net.bayes.resample(S, self.gen_b)
            with _cached(self.net):
                e, st, _, hs = rollout_tap(
                    self.net, self.live, list(rows), [k] * nb,
                    torch.tensor(plans, dtype=torch.float64), xs, self.en, S,
                    base, steer=steer, wave=wave, nh=self.nh)
        else:
            e, st, _ = ML.rollout_zero(torch.tensor(plans,
                                                    dtype=torch.float64),
                                       xs, self.en, steer)
            hs = None
        dev = st.device
        xs_t = torch.as_tensor(xs, dtype=torch.float64, device=dev)
        # progress, yaw rate, cross-track
        ph = torch.as_tensor(phi, dtype=torch.float64, device=dev).view(
            nb, 1, 1, 1)
        cp, sp = torch.cos(ph), torch.sin(ph)
        along0 = (cp[:, 0, 0, 0] * xs_t[:, 0] + sp[:, 0, 0, 0] * xs_t[:, 1])
        prog = (cp[..., 0] * st[..., -1, 0] + sp[..., 0] * st[..., -1, 1]
                - along0.view(nb, 1, 1))
        ytr = -sp * st[..., 0] + cp * st[..., 1]
        yaw = st[..., 8]
        # per-step peaks (g) and bow heights (m)
        u_draw = [torch.rand((S, H, 2), generator=self.gen_u[b],
                             dtype=torch.float64) for b in rows]
        if self.head is not None and hs is not None:
            from learn.meta.safety_head import inflate, inv_cdf
            with torch.no_grad():
                q = self.head(hs, e).double()           # (nb,K,S,H,2,nq)
            q = inflate(q, torch.as_tensor(infl, dtype=q.dtype, device=dev)
                        .view(nb, 1, 1, 1, 1))
            u = torch.stack(u_draw).to(dev)[:, None].expand(nb, K, S, H, 2)
            vals = inv_cdf(q, u)
            peaks = vals[..., 0] / G
            bow = vals[..., 1]
            kind = "head"
        else:
            zd0 = xs_t[:, 8].view(nb, 1, 1, 1).expand(nb, K, S, 1)
            zd = torch.cat([zd0, st[..., 4]], -1)
            peaks = (zd[..., 1:] - zd[..., :-1]) / (self.dtc * G)
            eta = None
            if wave is not None and self.variant == "p":
                eta = self._forecast_bow(wave, st, rows, k)
            Sw = max(1, math.ceil(S_BOW / S))
            stat = torch.stack([bow_statistical(
                *info[i]["bow"], S, Sw, H, self.dtc, self.gen_u[b])
                for i, b in enumerate(rows)]).to(dev)   # (nb, S*Sw, H)
            stat = stat.view(nb, 1, S, Sw, H).expand(nb, K, S, Sw, H)
            if eta is not None:
                eta = torch.where(torch.isnan(eta)[..., None, :].expand_as(
                    stat), stat, eta[:, :, :, None].expand_as(stat))
            else:
                eta = stat
            st_b = st[:, :, :, None].expand(nb, K, S, Sw, H, 10)
            bow = self._bow_from_states(st_b, eta).reshape(nb, K, S * Sw, H)
            kind = "proxy"
        past = [torch.as_tensor(np.asarray(info[i]["past"], float),
                                dtype=torch.float64, device=dev)
                for i in range(nb)]
        r_max = [info[i]["r_max"] for i in range(nb)]
        c, parts = constraint_cost(prog, peaks, bow, yaw, ytr, past, r_max,
                                   self.lim, self.geo["fb"], self.geo["L"],
                                   self.geo["u_ref"], H * self.dtc)
        thr_t = torch.as_tensor(thr, dtype=torch.float64, device=dev)
        c = c + W_JUMP * ((thr_t[..., 1:] - thr_t[..., :-1]) ** 2).sum(-1)
        if free:
            noz_t = torch.as_tensor(plans[..., 1], dtype=torch.float64,
                                    device=dev)
            c = c + W_JUMP * ((noz_t[..., 1:] - noz_t[..., :-1]) ** 2).sum(-1)
        bad = ~torch.isfinite(st).all(-1).all(-1)           # (nb, K, S)
        C = c.cpu().numpy()
        cmds = np.zeros(nb)
        diag = []
        for i, b in enumerate(rows):
            self.n_bad[b] += int(bad[i].sum())
            self.n_samp[b] += int(bad[i].numel())
            if np.all(np.isfinite(C[i])):
                w = np.exp(-(C[i] - C[i].min()) / self.lam)
                self.nominal[b] = (w[:, None] * cand[i]).sum(0) / w.sum()
                if free:
                    self.nominal_n[b] = (w[:, None] * cand_n[i]).sum(0) \
                        / w.sum()
                best = int(np.argmin(C[i]))
            else:
                best = 0
            cmds[i] = self.nominal[b, 0]
            self.nominal[b] = self.nominal[b] @ self.Sh.T
            extra = {}
            if free:
                extra["nozzle"] = float(self.nominal_n[b, 0])
                self.nominal_n[b] = self.nominal_n[b] @ self.Sh.T
            diag.append(dict(**extra,
                kind=kind, best=best,
                n_ok=int((parts["pen"][i] == 0).sum()),
                **{nm: float(parts[nm][i, best]) for nm in parts
                   if nm.startswith("g_")},
                speed=float(parts["speed"][i, best])))
        return cmds, diag


class _cached:
    """torch.nn.utils.parametrize.cached() when the net carries
    parametrisations (online LoRA adapters), else nothing."""

    def __init__(self, net):
        self.net = net
        self.cm = None

    def __enter__(self):
        from torch.nn.utils import parametrize
        if any(parametrize.is_parametrized(m) for m in self.net.modules()):
            self.cm = parametrize.cached()
            self.cm.__enter__()

    def __exit__(self, *a):
        if self.cm is not None:
            self.cm.__exit__(*a)


# ------------------------------------------------------------ protection
class SafeMode:
    """Measured-roll protection (section 9, not part of planning): |roll|
    above 20 deg -> throttle halved, nozzle centred; exit after 2 s with
    |roll| below 10 deg."""

    def __init__(self, dt_ctrl, enter_deg=20.0, exit_deg=10.0, hold_s=2.0):
        self.dt, self.enter = dt_ctrl, math.radians(enter_deg)
        self.exit, self.hold = math.radians(exit_deg), hold_s
        self.on, self.calm, self.n_on, self.n_entries = False, 0.0, 0, 0

    def __call__(self, roll, thrust, rud):
        a = abs(float(roll))
        if not self.on and a > self.enter:
            self.on, self.calm = True, 0.0
            self.n_entries += 1
        elif self.on:
            self.calm = self.calm + self.dt if a < self.exit else 0.0
            if self.calm >= self.hold:
                self.on = False
        if self.on:
            self.n_on += 1
            return 0.5 * thrust, 0.0, True
        return thrust, rud, False
