#!/usr/bin/env python3
"""
The learned one-step error model inside the MPC (design: the scratchpad's
DESIGN_MPC.md; the model: learn/meta/model3.py, DEFECTS M7-M11).

The MPC keeps control/mpc.py's MPPI structure (7 thrust knots over a
24-step horizon, warm start by one step, softmax weights with lam 0.6, the
first command executed, 128 candidates as the task runs it) but rolls its
candidates out with the MPC-convention reduced model PLUS sampled learned
errors: model3.rollout_core, the K candidates as the P plans of ONE moment
(the current step of each live episode), S samples each, the SAME base
noise for every candidate (common random numbers, RELATED_WORK_WEAKNESSES
W6). The cost is the task's own objective computed from rolled states only
(D6: no wave inputs):

  speed    max(0, (U - u_along) / U), as task.cost_parts
  impact   ImpactLink(step-mean |a_cg|): the task prices |a_cg| above 1 g
           at every 0.02 s plant step, a rolled state only gives the MEAN
           CG acceleration over a 0.24 s control step (exactly: plant w-dot
           is last_cg_acc and reduced zd is w, so |zd_{j+1} - zd_j| / dtc
           is |mean a_cg| over step j). The link K max(0, abar / A(u) - 1)^2
           maps that to the task's impact cost; (A, K) are fitted on target
           runs across the speed range (fit_impact_link) -- part of the task
           definition, a sensor-resolution conversion, the same for every
           variant. A(u) may depend on u = the MEASURED surge at the call,
           the same for every horizon step: a speed the plan cannot change
           (a threshold read from rolled speeds would let the planner cut
           its impact cost by speeding up, whatever the accelerations).
  track    K_Y huber(y / (5 L)), as task.cost_parts
  smooth   W_JUMP sum (U_{j+1} - U_j)^2 on the thrust plan (planner only)

The nozzle is NOT planned (no MPC on this task plans it): the heading
autopilot sets it every step from the measured state, and in the rollouts
SteerT, a torch port of Episode._steer, sets it per rolled sample through
rollout_core's `steer` hook.

Also here: LiveData (the live episode's history as a Data3-like object,
exactly the tokens window_tokens makes of a recorded episode), Pool3 (the
collected target episodes of the online variant), adapt_head (head-only
fine-tuning, DEFECTS M11), calibrate_spread (one rule for every net: the
sampler's base-noise scale whose one-step 90% coverage is closest to 0.90)
and horizon_check / impact_horizon (rolled vs true impact exceedance,
link cost and speed cost per horizon step on the unconfounded Cb blocks,
and the one impact horizon they allow).

Everything that rolls model0 or computes e0 uses env() = relabel._env()
(0.04 s x 6 substeps, what every target label was computed with), never an
env built from the live 0.02 s x 12 Mission.
"""
import math
import time

import numpy as np
import torch
from torch import nn

from learn.meta import model3 as M
from learn.repro.task import A_LIM, G, K_A, K_Y, LEGS, Y_SCALE

H = M.HB                  # horizon, control steps
N_KNOTS = 7
K_CAND = 128              # candidates (task.EP_KW n_samples)
S_SAMP = 4                # samples per candidate
SIGMA = 0.25              # knot noise on thrust
LAM = 0.6                 # MPPI temperature (hand MPC, C0)
W_JUMP = 0.3              # thrust smoothness (hand's w_dthrust, C0's W_JUMP)
# the cap of every sample cost, and the cost of a non-finite sample: a
# runaway rollout that overflowed must never look cheaper than a large
# finite one, and one runaway base draw (shared by all candidates, CRN)
# must not decide between candidates by how far each one ran away
BAD_COST = 1e3
KN = 0.514444
# spread scales searched by calibrate_spread (design: 1..3; 4 added for the
# fully fine-tuned oracle, whose coverage sat at 0.25-0.4, DEFECTS M10)
SCALES = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
COST_CH = (0, 3)          # e channels the cost reads: surge, heave rate
VEL_CH = (0, 1, 2, 3, 4)  # the 5 velocity channels (M11's mean)


def env():
    """THE env of every model0 rollout and e0 here (relabel._env(), cached
    there: one object per process)."""
    from learn.meta import relabel
    return relabel._env()


def check_env(m, en):
    """The live target Mission and the model's env agree on the control step
    and the actuator scales (fractions mean the same thing)."""
    assert abs(en["dt"] * en["sub"] - m.sub * m.dt) < 1e-12, \
        (en["dt"], en["sub"], m.dt, m.sub)
    assert abs(en["t_max"] - m.t_max) < 1e-9 * m.t_max, (en["t_max"], m.t_max)
    assert abs(en["rud_max"] - m.rud_max) < 1e-12, (en["rud_max"], m.rud_max)


def row_seed(*keys):
    """A 64-bit seed from integer keys (np.random.SeedSequence)."""
    return int(np.random.SeedSequence([int(k) for k in keys]).generate_state(
        1, np.uint64)[0])


def load_net(ck, dev, head=None):
    """model3.Net from a checkpoint dict or path, eval mode; head: a state
    dict of net.head to load on top (an online adaptation)."""
    if isinstance(ck, str):
        ck = torch.load(ck, weights_only=False)
    net = M.Net().to(dev)
    net.load_state_dict(ck["net"])
    if head is not None:
        net.head.load_state_dict(head)
    net.eval()
    return net


# ------------------------------------------------------------ knots
def knot_matrices(horizon=H, n=N_KNOTS):
    """(E (horizon, n), Sh (n, n)): MPPIController._expand as knots @ E.T and
    _shift as knots @ Sh.T (np.interp is linear in the knot values, so both
    are fixed matrices, built by applying them to the unit vectors)."""
    kt = np.linspace(0, horizon - 1, n)
    grid = np.arange(horizon)
    E = np.zeros((horizon, n))
    Sh = np.zeros((n, n))
    for i in range(n):
        full = np.interp(grid, kt, np.eye(n)[i])
        E[:, i] = full
        Sh[:, i] = np.interp(kt, grid, np.r_[full[1:], full[-1]])
    return E, Sh


# ------------------------------------------------------------ autopilot
class SteerT:
    """Episode._steer (the heading autopilot) in torch float64, per rolled
    sample, for rollout_core's `steer` hook: returns the plan's command with
    the nozzle column replaced by the autopilot's output for the rolled
    state, including the conditional integral (one per sample, started at
    the live Episode's _psi_i). Constants from the live Episode (reduced
    parameters, plant length and nozzle limit, dt_ctrl 0.25 s, u_ref);
    heading references phi (B,) per row; commands are fractions, thrust
    in N = fraction x env t_max as Model0T reads it.

    psi and r are reduced columns 7 and 8 in the world frame."""

    def __init__(self, ep, phi, psi_i0, en, keep=False):
        p = ep.reduced.p
        L = float(ep.plant.L)
        self.rs = math.sqrt(L / 10.0)
        self.planing = p.get("planing", 0.0) > 0.5
        self.wn = (0.30 * (ep.u_ref / L) / 0.45 if self.planing
                   else 0.30 / self.rs)
        self.lim = float(ep.plant.rudder.max)
        self.jet = p.get("steer_jet", 0.0) > 0.5
        if self.jet:
            self.kf = (float(p["k_nomoto_f"]), float(p["k_jet_side"]))
            self.t_floor = 0.15 * float(p["t_max"])
        else:
            self.K = float(p.get("k_nomoto", -1.0))
        self.T = max(float(p.get("tau_r", 3.0)), 0.5 * self.rs)
        self.dt_ctrl = float(ep.dt_ctrl)
        self.t_max, self.rud_max = float(en["t_max"]), float(en["rud_max"])
        self.phi = torch.as_tensor(np.asarray(phi, float), dtype=torch.float64)
        self.psi_i0 = torch.as_tensor(np.asarray(psi_i0, float),
                                      dtype=torch.float64)
        self.keep = keep
        self.reset()

    def reset(self, psi_i0=None):
        """Every sample's integral back to the live value(s) (B,)."""
        if psi_i0 is not None:
            self.psi_i0 = torch.as_tensor(np.asarray(psi_i0, float),
                                          dtype=torch.float64)
        self.psi_i = None
        self.log = []

    def __call__(self, j, sr, Uj):
        dev = sr.device
        B = sr.shape[0]
        nb = sr.dim() - 2                      # plan / sample dims
        view = (B,) + (1,) * nb
        if j == 0 or self.psi_i is None:
            self.psi_i = self.psi_i0.to(dev).view(view)
        phi = self.phi.to(dev).view(view)
        thr = Uj[..., 0].expand(sr.shape[:-1])
        zeta = 1.0
        if self.jet:
            t_now = torch.clamp(thr * self.t_max, min=self.t_floor)
            K = self.kf[0] * self.kf[1] * t_now
        else:
            K = self.K
        wn, T = self.wn, self.T
        a = wn ** 2 * T / K
        if self.planing:
            b = max(2 * zeta * wn * T - 1.0, 0.0) / K
        else:
            b = (2 * zeta * wn * T - 1.0) / K
        err = torch.remainder(sr[..., 7] - phi + math.pi, 2 * math.pi) \
            - math.pi
        raw = -(a * err + b * sr[..., 8] + (0.12 / self.rs) * a * self.psi_i)
        ok = raw.abs() < self.lim
        self.psi_i = torch.where(
            ok, torch.clamp(self.psi_i + err * self.dt_ctrl,
                            -6.0 * self.rs, 6.0 * self.rs), self.psi_i)
        noz = torch.clamp(raw, -self.lim, self.lim) / self.rud_max
        out = torch.stack([thr, noz], -1)
        if self.keep:
            self.log.append(out)
        return out


# ------------------------------------------------------------ rollouts
def rollout_zero(plans, xs_k, en, steer=None):
    """The e = 0 rollout (variant m0): exactly what rollout_core does when
    every sampled error is 0 -- s_{j+1} = Model0T(s_j, U_j), the actuator
    column clip(U_j) -- in float64 on the plans' device (the CPU for m0),
    S = 1. Returns rollout_core's shapes: e (B, P, 1, H, 10) zeros, states
    (B, P, 1, H, 10), actuator positions (B, P, 1, H, 2)."""
    pl = torch.as_tensor(plans, dtype=torch.float64)
    dev = pl.device
    B, P, Hh, _ = pl.shape
    xs = torch.as_tensor(np.asarray(xs_k, float), dtype=torch.float64,
                         device=dev)
    sr = M.plant_to_reduced_t(xs)[:, None, None].expand(B, P, 1, 10)
    lo = torch.tensor([0.0, -1.0], dtype=torch.float64, device=dev)
    hi = torch.tensor([1.0, 1.0], dtype=torch.float64, device=dev)
    m0 = M.model0t(en)
    states, acts = [], []
    for j in range(Hh):
        Uj = pl[:, :, None, j]
        if steer is not None:
            Uj = steer(j, sr, Uj)
        sr = m0(sr, Uj)
        act = torch.maximum(torch.minimum(Uj, hi), lo)
        states.append(sr.expand(B, P, 1, 10))
        acts.append(act.expand(B, P, 1, 2))
    return (torch.zeros(B, P, 1, Hh, M.C7, device=dev),
            torch.stack(states, 3), torch.stack(acts, 3))


# ------------------------------------------------------------ the cost
class ImpactLink:
    """Step-mean |a_cg| (g) -> the task's impact cost per second:
    K max(0, abar / A(u) - 1)^2, A(u) = clip(A0 + A1 (u / kn - pivot),
    a_min, a_max) with u the measured surge when the plan is made (the
    window's start speed in the fit), never a rolled speed (A1 = 0: one
    threshold). j_imp: the impact term counts only horizon steps j < j_imp
    (horizon_check's fix; 24 = all)."""

    def __init__(self, A0, K, A1=0.0, pivot=23.0, a_min=0.15, a_max=1.5,
                 j_imp=H):
        self.A0, self.K, self.A1 = float(A0), float(K), float(A1)
        self.pivot, self.a_min, self.a_max = float(pivot), a_min, a_max
        self.j_imp = int(j_imp)

    def A(self, u):
        if self.A1 == 0.0:
            return self.A0
        a = self.A0 + self.A1 * (u / KN - self.pivot)
        if isinstance(a, torch.Tensor):
            return a.clamp(self.a_min, self.a_max)
        return np.clip(a, self.a_min, self.a_max)

    def cost(self, abar, u):
        x = abar / self.A(u) - 1.0
        if isinstance(x, torch.Tensor):
            return self.K * x.clamp(min=0.0) ** 2
        return self.K * np.maximum(x, 0.0) ** 2

    def to_dict(self):
        return dict(A0=self.A0, K=self.K, A1=self.A1, pivot=self.pivot,
                    a_min=self.a_min, a_max=self.a_max, j_imp=self.j_imp)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def huber_t(d):
    """task.huber for tensors."""
    a = d.abs()
    return torch.where(a < 2.0, d * d, 2.0 * (2.0 * a - 2.0))


def task_cost(states, plans_thr, xs_k, phi, link, en, u_ref, L):
    """Per-sample planner cost of rolled states: states (B, P, S, H, 10)
    after steps j = 0..H-1, the thrust plans (B, P, H), the measured plant
    states at k (B, 14), track angles phi (B,). Returns c (B, P, S) and the
    parts (speed, impact, track) (B, P, S), each integrated with dtc (per-
    second rates x 0.24 s, the task score's scale). The heave-rate change
    of step j is taken from the state before it (the measured one for
    j = 0); the impact threshold A(u) reads the MEASURED surge xs_k[6] for
    every step, so no candidate can move it."""
    dev = states.device
    dtc = en["dt"] * en["sub"]
    B, P, S, Hh, _ = states.shape
    xs = torch.as_tensor(np.asarray(xs_k, float), dtype=torch.float64,
                         device=dev)
    ph = torch.as_tensor(np.asarray(phi, float), dtype=torch.float64,
                         device=dev).view(B, 1, 1, 1)
    x, y, u = states[..., 0], states[..., 1], states[..., 2]
    zd, psi, v = states[..., 4], states[..., 7], states[..., 9]
    d = psi - ph
    u_al = u * torch.cos(d) - v * torch.sin(d)
    c_speed = ((u_ref - u_al) / u_ref).clamp(min=0.0)
    first = lambda c: xs[:, c].view(B, 1, 1, 1).expand(B, P, S, 1)  # noqa
    zd_prev = torch.cat([first(8), zd[..., :-1]], -1)
    abar = (zd - zd_prev).abs() / (dtc * G)
    c_imp = link.cost(abar, first(6))
    if link.j_imp < Hh:
        c_imp = c_imp * (torch.arange(Hh, device=dev) < link.j_imp)
    yt = -torch.sin(ph) * x + torch.cos(ph) * y
    c_track = K_Y * huber_t(yt / (Y_SCALE * L))
    parts = [(c * dtc).sum(-1) for c in (c_speed, c_imp, c_track)]
    thr = torch.as_tensor(plans_thr, dtype=torch.float64, device=dev)
    smooth = W_JUMP * ((thr[..., 1:] - thr[..., :-1]) ** 2).sum(-1)
    return parts[0] + parts[1] + parts[2] + smooth[..., None], parts


# ------------------------------------------------ the link from target runs
class StepLog:
    """Per control step: surge speed and heave rate (the state at step k,
    k = 0..n), the step's mean plant-step task impact cost (the score's
    impact part over the step), the actual thrust fraction after the step.
    Call with the mission once at the start and after every finite step
    (cmpc.run_episode's on_step)."""

    def __init__(self):
        self.u, self.zd, self.c_imp, self.thr = [], [], [], []
        self.dtc = None

    def __call__(self, m):
        if self.u:
            self.c_imp.append(float(m.last_parts[1]))
            self.thr.append(float(m.s[12] / m.t_max))
        else:
            self.dtc = m.sub * m.dt
        self.u.append(float(m.s[6]))
        self.zd.append(float(m.s[8]))

    def arrays(self):
        """dict(u (n+1), zd (n+1), c_imp (n), abar (n) = |dzd| / (dtc g),
        thr (n)), float32."""
        dtc = self.dtc
        zd = np.asarray(self.zd)
        return dict(u=np.asarray(self.u, np.float32),
                    zd=zd.astype(np.float32),
                    c_imp=np.asarray(self.c_imp, np.float32),
                    abar=(np.abs(np.diff(zd)) / (dtc * G)).astype(np.float32),
                    thr=np.asarray(self.thr, np.float32))


SPEED_BINS = np.arange(14.0, 34.1, 2.0)      # kn


def link_windows(recs, win=H, stride=8):
    """Windows of `win` control steps from step logs [dict(arrays, leg)]:
    y (per-step mean task impact cost), abar (win,), u_start (win,) (the
    speed before each step), the window's index speed (its first step's
    START speed, never its own mean: impacts slow the boat within the
    window, J2.1), the leg."""
    Y, AB, US, U0, LG = [], [], [], [], []
    for r in recs:
        n = len(r["c_imp"])
        for a in range(0, n - win + 1, stride):
            Y.append(float(np.mean(r["c_imp"][a:a + win])))
            AB.append(r["abar"][a:a + win])
            US.append(r["u"][a:a + win])
            U0.append(float(r["u"][a]))
            LG.append(int(r["leg"]))
    return dict(y=np.asarray(Y), abar=np.asarray(AB, float),
                u=np.asarray(US, float), u0=np.asarray(U0),
                leg=np.asarray(LG))


def _cells(w, n_min):
    """(leg, speed bin) cells with at least n_min windows: list of (leg,
    bin index, member mask)."""
    b = np.digitize(w["u0"] / KN, SPEED_BINS)
    out = []
    for lg in np.unique(w["leg"]):
        for bi in np.unique(b[w["leg"] == lg]):
            msk = (w["leg"] == lg) & (b == bi)
            if msk.sum() >= n_min:
                out.append((int(lg), int(bi), msk))
    return out


def _link_eval(w, link, cells):
    """Predicted per-step mean cost of each window under `link` (the
    threshold at the window's start speed for all its steps, as the planner
    reads it at the measured speed), and the cell table [(leg, bin lo kn,
    n, true, pred)]."""
    pred = link.cost(w["abar"], w["u0"][:, None]).mean(1)
    tab = [(lg, float(SPEED_BINS[bi - 1]) if bi > 0 else float("nan"),
            int(m.sum()), float(w["y"][m].mean()), float(pred[m].mean()))
           for lg, bi, m in cells]
    return pred, tab


def link_quality(tab, y_mean):
    """(weighted RMS of the cell errors / overall mean cost, worst growth
    ratio predicted / true between each leg's lowest and highest cell, per
    leg ratios). The growth ratio is what the throttle decision reads."""
    n = np.array([t[2] for t in tab], float)
    err = np.array([t[4] - t[3] for t in tab])
    rms = float(np.sqrt((n * err ** 2).sum() / n.sum())) / max(y_mean, 1e-12)
    ratios = {}
    for lg in sorted({t[0] for t in tab}):
        rows = sorted([t for t in tab if t[0] == lg], key=lambda t: t[1])
        if len(rows) >= 2:
            gt, gp = rows[-1][3] - rows[0][3], rows[-1][4] - rows[0][4]
            if abs(gt) > 0.1 * max(y_mean, 1e-12):
                ratios[lg] = gp / gt
    worst = max([abs(math.log(max(r, 1e-6))) for r in ratios.values()],
                default=0.0)
    return rms, math.exp(worst), ratios


def fit_impact_link(recs, n_min=15, a_grid=None, a1_grid=None, rms_tol=0.2,
                    growth_tol=1.25, log=print):
    """(A, K) of the impact link from target step logs across the speed
    range (hand MPC and C0 with planning-speed offsets). Per candidate A: K
    so that the overall mean predicted cost equals the true one; A chosen
    to reproduce the cells (leg x 2-kn bins of the window's START speed):
    the least n-weighted squared error of the cell means. The single
    threshold is kept when it tracks the cells (weighted RMS error <=
    rms_tol x the mean cost AND every leg's growth from its slowest to its
    fastest cell within a factor growth_tol); otherwise A is made linear in
    the window's START speed (A1 != 0; the planner reads it at the measured
    speed), fitted the same way. info['edge'] flags an A0 at an end of
    a_grid (a constrained fit: K then prices rare large accelerations very
    steeply). Returns (ImpactLink, info)."""
    w = link_windows(recs)
    cells = _cells(w, n_min)
    y_mean = float(w["y"].mean())
    a_grid = np.arange(0.20, 1.001, 0.025) if a_grid is None else a_grid
    # rounded: arange's ~1e-17 in place of 0 would take ImpactLink's
    # speed-dependent path for what is one threshold
    a1_grid = (np.round(np.arange(-0.06, 0.0601, 0.005), 6)
               if a1_grid is None else a1_grid)

    def fit(a1s):
        best = None
        for a1 in a1s:
            for a0 in a_grid:
                lk = ImpactLink(a0, 1.0, A1=a1)
                f = lk.cost(w["abar"], w["u0"][:, None]).mean(1)
                if f.sum() <= 0:
                    continue
                lk.K = float(w["y"].sum() / f.sum())
                _, tab = _link_eval(w, lk, cells)
                n = np.array([t[2] for t in tab], float)
                sse = float((n * (np.array([t[4] - t[3] for t in tab])) ** 2
                             ).sum() / max(n.sum(), 1))
                if best is None or sse < best[0]:
                    best = (sse, lk, tab)
        return best

    sse1, lk1, tab1 = fit([0.0])
    rms1, gr1, rat1 = link_quality(tab1, y_mean)
    info = dict(n_windows=len(w["y"]), y_mean=y_mean, single=dict(
        link=lk1.to_dict(), rms=rms1, growth=gr1, ratios=rat1, cells=tab1))
    ok = rms1 <= rms_tol and gr1 <= growth_tol
    lk = lk1
    if not ok:
        sse2, lk2, tab2 = fit(a1_grid)
        rms2, gr2, rat2 = link_quality(tab2, y_mean)
        info["speed"] = dict(link=lk2.to_dict(), rms=rms2, growth=gr2,
                             ratios=rat2, cells=tab2)
        lk = lk2
    # the slope search may itself land on A1 = 0: then it is one threshold
    info["chosen"] = "single" if ok or lk.A1 == 0.0 else "speed"
    info["edge"] = bool(min(abs(lk.A0 - a_grid[0]),
                            abs(lk.A0 - a_grid[-1])) < 1e-9)
    info["a_grid"] = (float(a_grid[0]), float(a_grid[-1]))
    pred, _ = _link_eval(w, lk, cells)
    info["corr"] = float(np.corrcoef(pred, w["y"])[0, 1]) \
        if pred.std() > 0 and w["y"].std() > 0 else float("nan")
    info["per_leg"] = {int(lg): (float(w["y"][w["leg"] == lg].mean()),
                                 float(pred[w["leg"] == lg].mean()))
                       for lg in np.unique(w["leg"])}
    log(format_link(info))
    return lk, info


def format_link(info):
    lines = [f"  impact link: {info['n_windows']} windows, mean task impact "
             f"cost {info['y_mean']:.4f} per step; chosen {info['chosen']}"]
    for key in ("single", "speed"):
        if key not in info:
            continue
        d = info[key]
        lk = d["link"]
        lines.append(f"    {key}: A0 {lk['A0']:.3f} g, A1 {lk['A1']:+.3f} g/kn,"
                     f" K {lk['K']:.3f}; cell RMS error {d['rms']:.2f} x mean,"
                     f" worst growth ratio {d['growth']:.2f} (per leg "
                     + ", ".join(f"{k}: {v:.2f}" for k, v in
                                 d["ratios"].items()) + ")")
        for lg, lo, n, yt, yp in d["cells"]:
            lines.append(f"      leg {lg} {lo:4.0f}-{lo + 2:.0f} kn n {n:4d}: "
                         f"true {yt:.4f} link {yp:.4f}")
    if info.get("edge"):
        lo, hi = info["a_grid"]
        lines.append(f"    WARNING: the chosen A0 sits at an end of the fit "
                     f"grid [{lo:.2f}, {hi:.2f}] g: a constrained fit, K "
                     "prices rare large step accelerations steeply")
    if "corr" in info:
        lines.append(f"    window correlation {info['corr']:.2f}; per leg "
                     "(true, link) " + ", ".join(
                         f"{k}: {a:.4f}/{b:.4f}"
                         for k, (a, b) in info["per_leg"].items()))
    return "\n".join(lines)


# ------------------------------------------------------ live history
def mission_consts(m):
    """What the state inputs need from a target Mission (as
    data3.episode_blocks recorded them)."""
    p = m.red.p
    return dict(u_ref=float(m.u_ref), L=float(m.L), t_max=float(m.t_max),
                rud_max=float(m.rud_max), z0=float(p.get("z0", 0.0)),
                th0=float(p.get("th0", 0.0)))


def inputs11(xs, c):
    """The recorded S[:11] of a plant state: residuals.inputs_of + zr, world
    frame, actual actuator positions, float32 (as episode_blocks stores)."""
    from learn.meta.residuals import inputs_of
    x9 = inputs_of(xs, c["u_ref"], c["L"], c["t_max"], c["rud_max"])
    zr = ((xs[2] - c["z0"]) / 0.2, (xs[4] - c["th0"]) / 0.05)
    return np.concatenate([x9, zr]).astype(np.float32)


def _nx(s11, st):
    return (s11 - st["x_mu"]) / st["x_sd"]


def _nu(u32, st):
    return (u32 - st["u_mu"]) / st["u_sd"]


def _ne(e32, st):
    return e32 / st["e_sd"]


class LiveData:
    """The live episodes of a lockstep group as a Data3-like object that
    window_tokens / rollout_core index (rows b = episodes, growing k):
    X (n, T, 11), U (n, T, 2), E (n, T, 10) float32 normalised with the
    checkpoint's stats, valid (n, T), len (n,), n, T, dev, stats; numpy
    raw mirrors S11, U32, E0, XS. X[b, k] from the measured plant state at
    k; after the plant advanced step k with command u (fractions):
    U[b, k], E[b, k] = e0_from(env, xs_k, xs_{k+1}, u) / e_sd (float32 cast
    first, as the recorded splits), valid[b, k]. The first-flag and e_prev
    conventions are window_tokens' own; rollout_core hides positions >= k
    (not yet filled)."""

    def __init__(self, n, T, stats, dev, consts):
        # at least one context window: window_tokens clamps its positions
        # to T - 1 but not the previous-error index t - 1
        T = max(int(T), M.W_CTX + 1)
        self.n, self.T, self.dev, self.stats, self.c = n, T, dev, stats, consts
        z = lambda *s: torch.zeros(*s, device=dev)   # noqa: E731
        self.X, self.U, self.E = z(n, T, M.D_X), z(n, T, 2), z(n, T, M.C7)
        self.valid = torch.zeros(n, T, dtype=torch.bool, device=dev)
        self.len = torch.zeros(n, dtype=torch.long, device=dev)
        self.S11 = np.zeros((n, T, M.D_X), np.float32)
        self.U32 = np.zeros((n, T, 2), np.float32)
        self.E0 = np.zeros((n, T, M.C7), np.float32)
        self.XS = np.zeros((n, T, 14))
        self.Uraw = self.U32
        self.k = np.zeros(n, int)

    def start(self, b, xs0):
        self.XS[b, 0] = xs0
        self.S11[b, 0] = inputs11(np.asarray(xs0, float), self.c)
        self.X[b, 0] = torch.from_numpy(_nx(self.S11[b, 0],
                                            self.stats)).to(self.dev)

    def push(self, b, u, xs_next):
        from learn.meta.data3 import e0_from
        k = int(self.k[b])
        u = np.asarray(u, float)
        xs_next = np.asarray(xs_next, float)
        e0 = e0_from(env(), self.XS[b, k][None], xs_next[None], u[None])[0]
        self.U32[b, k] = u.astype(np.float32)
        self.E0[b, k] = e0.astype(np.float32)
        self.XS[b, k + 1] = xs_next
        self.S11[b, k + 1] = inputs11(xs_next, self.c)
        st, dev = self.stats, self.dev
        self.U[b, k] = torch.from_numpy(_nu(self.U32[b, k], st)).to(dev)
        self.E[b, k] = torch.from_numpy(_ne(self.E0[b, k], st)).to(dev)
        self.X[b, k + 1] = torch.from_numpy(_nx(self.S11[b, k + 1],
                                                st)).to(dev)
        self.valid[b, k] = True
        self.k[b] = k + 1
        self.len[b] = k + 1

    def episode(self, b):
        """The raw record of row b for Pool3."""
        n = int(self.k[b])
        return dict(s11=self.S11[b, :n + 1].copy(), u=self.U32[b, :n].copy(),
                    e0=self.E0[b, :n].copy(), xs=self.XS[b, :n + 1].copy(),
                    len=n)


class Pool3:
    """Collected target episodes (LiveData.episode records) as a Data3-like
    object with Data3's conventions: X zero at t >= len, valid t < len (a
    block-split style record: the final state is stored)."""

    def __init__(self, eps, stats, dev):
        self.eps = list(eps)
        self.stats, self.dev = stats, dev
        n = len(self.eps)
        T = max(max(e["len"] for e in self.eps) + 1, M.W_CTX + 1)
        S11 = np.zeros((n, T, M.D_X), np.float32)
        U = np.zeros((n, T, 2), np.float32)
        E0 = np.zeros((n, T, M.C7), np.float32)
        XS = np.zeros((n, T, 14))
        L = np.array([e["len"] for e in self.eps])
        for i, e in enumerate(self.eps):
            S11[i, :e["len"] + 1], XS[i, :e["len"] + 1] = e["s11"], e["xs"]
            U[i, :e["len"]], E0[i, :e["len"]] = e["u"], e["e0"]
        valid = np.arange(T)[None] < L[:, None]
        t = lambda a, **k: torch.tensor(a, device=dev, **k)   # noqa: E731
        self.X = t(_nx(S11, stats) * valid[..., None], dtype=torch.float32)
        self.U = t(_nu(U, stats), dtype=torch.float32)
        self.E = t(_ne(E0, stats) * valid[..., None], dtype=torch.float32)
        self.valid = t(valid)
        self.len = t(L)
        self.n, self.T = n, T
        self.XS, self.Uraw = XS, U

    def save(self, path):
        np.savez(path + ".tmp.npz", n=len(self.eps),
                 **{f"{k}_{i}": e[k] for i, e in enumerate(self.eps)
                    for k in ("s11", "u", "e0", "xs")})
        import os
        os.replace(path + ".tmp.npz", path)

    @staticmethod
    def load_eps(path):
        d = np.load(path)
        return [dict(s11=d[f"s11_{i}"], u=d[f"u_{i}"], e0=d[f"e0_{i}"],
                     xs=d[f"xs_{i}"], len=len(d[f"u_{i}"]))
                for i in range(int(d["n"]))]


# ------------------------------------------------------ the controller
class LearnedMPPI:
    """MPPI over thrust knots for B episode rows in lockstep (control/
    mpc.py's structure). Per row: nominal knots (start 0.5), knot noise
    from np.random.default_rng([seed, leg, 5]), base noise from its own
    torch CPU generator seeded (seed, leg, 7), drawn row by row (a row's
    draws never depend on the group it runs in). Per call (rows all at
    step k): candidates = clip(nominal + noise, floor, 1), expanded to 24
    steps, rolled out (net: rollout_core with the rows' LiveData history,
    S samples with ONE base draw per row shared by all candidates, scaled by
    the spread s -- a scalar or one per horizon step; net None: rollout_zero,
    the e = 0 model), the nozzle by SteerT, cost = mean over samples of
    task_cost with every sample cost capped at BAD_COST and a non-finite
    one set to BAD_COST (counted per row: n_bad non-finite, n_clip capped
    finite, n_samp all); softmax weights, nominal = weighted mean, command
    = its first step, then the one-step shift."""

    def __init__(self, jobs, missions, en, link, net=None, live=None,
                 spread=1.0, K=K_CAND, S=S_SAMP, sigma=SIGMA, lam=LAM,
                 floor=0.0):
        self.jobs, self.ms, self.en, self.link = jobs, missions, en, link
        self.net, self.live = net, live
        self.B = len(jobs)
        self.K, self.S = K, S if net is not None else 1
        self.sigma, self.lam, self.floor = sigma, lam, floor
        self.E, self.Sh = knot_matrices()
        self.nominal = np.full((self.B, N_KNOTS), 0.5)
        self.rng = [np.random.default_rng([j["seed"], j["leg"], 5])
                    for j in jobs]
        self.gen = [torch.Generator().manual_seed(row_seed(j["seed"],
                                                           j["leg"], 7))
                    for j in jobs]
        sp = np.atleast_1d(np.asarray(spread, float))
        self.spread = torch.tensor(sp if sp.size > 1 else sp[0],
                                   dtype=torch.float32)
        self.n_bad = np.zeros(self.B, int)
        self.n_clip = np.zeros(self.B, int)
        self.n_samp = np.zeros(self.B, int)
        self.capture = False
        self.last = None
        self.dev = live.dev if net is not None else torch.device("cpu")

    def draws(self, b):
        """Row b's knot noise (K, 7) and base noise (S, H, 10) for this
        step, from its own streams."""
        noise = self.rng[b].normal(0.0, 1.0, (self.K, N_KNOTS)) * self.sigma
        base = None
        if self.net is not None:
            base = torch.randn((self.S, H, M.C7), generator=self.gen[b])
        return noise, base

    def base_tensor(self, bases):
        """(nb, K, S, H, 10): each row's draw x spread, expanded (a view)
        over the candidates -- common random numbers."""
        bs = torch.stack(bases)[:, None]
        sp = self.spread
        bs = bs * (sp.view(1, 1, 1, H, 1) if sp.dim() else sp)
        nb = bs.shape[0]
        return bs.to(self.dev).expand(nb, self.K, self.S, H, M.C7)

    def __call__(self, rows, k):
        nb = len(rows)
        cand, bases = [], []
        for b in rows:
            noise, base = self.draws(b)
            cand.append(np.clip(self.nominal[b][None] + noise, self.floor,
                                1.0))
            bases.append(base)
        cand = np.stack(cand)                                # (nb, K, 7)
        thr = cand @ self.E.T                                # (nb, K, H)
        plans = np.zeros((nb, self.K, H, 2))
        plans[..., 0] = thr
        ms = [self.ms[b] for b in rows]
        xs = np.stack([m.s for m in ms])
        phi = np.array([m.phi for m in ms])
        steer = SteerT(ms[0].ep, phi, [m.ep._psi_i for m in ms], self.en)
        if self.net is not None:
            pl = torch.tensor(plans, dtype=torch.float64)
            base = self.base_tensor(bases)
            _, st, _ = M.rollout_core(self.net, self.live, list(rows),
                                      [k] * nb, pl, xs, self.en, self.S,
                                      base=base, steer=steer)
        else:
            _, st, _ = rollout_zero(torch.tensor(plans, dtype=torch.float64),
                                    xs, self.en, steer)
        m0 = ms[0]
        c, parts = task_cost(st, thr, xs, phi, self.link, self.en, m0.u_ref,
                             m0.L)
        c_raw = c
        bad = ~torch.isfinite(c)
        clip = ~bad & (c > BAD_COST)
        c = torch.where(bad | clip, torch.full_like(c, BAD_COST), c)
        C = c.mean(-1).cpu().numpy()                         # (nb, K)
        cmds = np.zeros(nb)
        for i, b in enumerate(rows):
            self.n_bad[b] += int(bad[i].sum())
            self.n_clip[b] += int(clip[i].sum())
            self.n_samp[b] += int(bad[i].numel())
            if bool(bad[i].all()):
                pass                            # keep the warm start
            else:
                w = np.exp(-(C[i] - C[i].min()) / self.lam)
                self.nominal[b] = (w[:, None] * cand[i]).sum(0) / w.sum()
            cmds[i] = self.nominal[b, 0]
            self.nominal[b] = self.nominal[b] @ self.Sh.T
        if self.capture:
            self.last = dict(rows=list(rows), k=k, plans=plans, xs=xs,
                             phi=phi, psi_i=[m.ep._psi_i for m in ms],
                             base_rows=None if self.net is None else
                             torch.stack(bases),
                             states=st.cpu(), cost=c_raw.cpu(),
                             cost_used=c.cpu(), cand=cand,
                             cmds=cmds.copy())
        return cmds


def gpu_util():
    """nvidia-smi's utilisation and memory in use (a contention snapshot)."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True,
            text=True, timeout=10).stdout.strip().split(",")
        return dict(util=float(out[0]), mem_mb=float(out[1]))
    except Exception:
        return dict(util=float("nan"), mem_mb=float("nan"))


def run_group(jobs, en, link, net=None, stats=None, spread=1.0, T=120.0,
              collect=False, log=print, tag=""):
    """The new controller on B target missions in lockstep: jobs [dict(seed,
    leg)], Mission("high", seed, leg, t_end=T, track=LEGS[leg]); net None =
    m0. One controller call per control step for all live rows; the plants
    step serially; thrust = command x t_max, nozzle = the real autopilot
    (m.ep._steer), m.advance, LiveData.push. Returns one row per job:
    metrics, step log, controller wall time per call and per episode-step,
    non-finite sample costs, the recorded episode (collect) and the GPU
    peak."""
    from learn.repro.task import Mission
    ms = [Mission("high", j["seed"], j["leg"], t_end=T,
                  track=LEGS[j["leg"]]) for j in jobs]
    check_env(ms[0], en)
    B, n_ctrl = len(jobs), ms[0].n_ctrl
    dev = next(net.parameters()).device if net is not None else \
        torch.device("cpu")
    live = None
    if net is not None or collect:
        live = LiveData(B, n_ctrl + 1, stats, dev, mission_consts(ms[0]))
        for b, m in enumerate(ms):
            live.start(b, m.s)
    logs = [StepLog() for _ in jobs]
    for m, lg in zip(ms, logs):
        lg(m)
    ctrl = LearnedMPPI(jobs, ms, en, link, net=net, live=live, spread=spread)
    snap = gpu_util()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_call, n_rows = [], []
    t0 = time.time()
    k = 0
    while True:
        rows = [b for b in range(B) if not ms[b].done()]
        if not rows:
            break
        tc = time.perf_counter()
        cmds = ctrl(rows, k)
        t_call.append(time.perf_counter() - tc)
        n_rows.append(len(rows))
        for b, c in zip(rows, cmds):
            m = ms[b]
            thrust = float(c) * m.t_max
            rud = m.ep._steer(m.s, thrust)
            m.advance(thrust, rud)
            if m.finite:
                if live is not None:
                    live.push(b, (thrust / m.t_max, rud / m.rud_max), m.s)
                logs[b](m)
        k += 1
        if k % 60 == 0:
            log(f"    {tag} step {k}/{n_ctrl}: {np.mean(t_call[-60:]):.2f} s "
                f"per call ({len(rows)} rows), {time.time() - t0:.0f} s")
    tcall = np.asarray(t_call)
    per_ep = tcall / np.asarray(n_rows)
    gpu = (torch.cuda.max_memory_allocated() / 2 ** 30
           if dev.type == "cuda" else 0.0)
    out = []
    for b, (j, m) in enumerate(zip(jobs, ms)):
        r = dict(seed=j["seed"], leg=j["leg"], **m.metrics(),
                 log=logs[b].arrays(), n_bad=int(ctrl.n_bad[b]),
                 n_clip=int(ctrl.n_clip[b]), n_samp=int(ctrl.n_samp[b]),
                 t_call_mean=float(tcall.mean()),
                 t_call_p95=float(np.percentile(tcall, 95)),
                 t_step_mean=float(per_ep.mean()),
                 t_step_p95=float(np.percentile(per_ep, 95)),
                 group=len(jobs), gpu_peak_gb=gpu, gpu_snap=snap,
                 spread=np.asarray(spread).tolist())
        r["kn"] = r["u_along"] / KN if np.isfinite(r["u_along"]) else np.nan
        if collect:
            r["episode"] = live.episode(b)
        out.append(r)
    return out


# ------------------------------------------------ spread and adaptation
@torch.no_grad()
def one_step_tally(net, D, eps, s, lo, n_samp=64, seed=0, L=M.W_CTX):
    """One-step Tally of net on episodes eps of D at every position >= lo
    with the full recorded history before it (windows [0, L) and
    [len - L, len), as adapt_calib_m11.score), the flow sampled through
    sample_grad with base noise x s (the controller's sampler)."""
    dev = D.dev
    gen = torch.Generator(device=dev).manual_seed(seed)
    tal = M.Tally((M.C7,))
    for i, a0, lw, hw in _val_windows(D, eps, lo, L):
        ii = torch.tensor([i], device=dev)
        a = torch.full_like(ii, a0)
        tok, tgt, ok = M.window_tokens(D, ii, a, L)
        h = net.encode(tok)
        pos = torch.arange(L, device=dev) + a0
        sel = ok & (pos[None] >= lw) & (pos[None] < hw)
        hs, ys = h[sel], tgt[sel]
        for c0 in range(0, len(hs), 1024):
            hh = hs[c0:c0 + 1024]
            y0 = s * torch.randn((len(hh), n_samp, M.C7), device=dev,
                                 generator=gen)
            smp = net.sample_grad(hh[:, None].expand(-1, n_samp, -1), y0)
            tal.add(smp, ys[c0:c0 + 1024])
    return tal


def calibrate_spread(net, D, eps, lo, scales=SCALES, ch=COST_CH, log=None):
    """THE spread rule, the same for every net: the base-noise scale s whose
    one-step 90% coverage (randomised PIT) on D's episodes eps at positions
    >= lo is closest to 0.90, the coverage averaged over the channels the
    cost reads (surge, heave rate; the 5-velocity-channel mean reported
    next to it). Returns (s, table {s: dict(cov_cost, cov5, cov, skill)})."""
    tab = {}
    for s in scales:
        r = one_step_tally(net, D, eps, s, lo).summary()
        tab[s] = dict(cov_cost=float(r["cov"][list(ch)].mean()),
                      cov5=float(r["cov"][list(VEL_CH)].mean()),
                      cov=r["cov"].tolist(), skill=r["skill"].tolist())
    s_best = min(scales, key=lambda s: abs(tab[s]["cov_cost"] - 0.9))
    if log is not None:
        log("    spread: " + ", ".join(
            f"s {s}: cov {t['cov_cost']:.2f} (5-ch {t['cov5']:.2f})"
            for s, t in tab.items()) + f" -> s = {s_best}")
    return s_best, tab


def _val_windows(D, eps, lo, L=M.W_CTX):
    """[(episode, a, lo_w, hi_w)] of the scoring windows: per episode [0, L)
    scoring [lo, L) and [len - L, len) scoring [max(L, lo), len); lo an int
    or a dict episode -> lo."""
    out = []
    for i in eps:
        Tn = int(D.len[int(i)])
        li = lo[int(i)] if isinstance(lo, dict) else lo
        for a0, lw, hw in ((0, li, min(L, Tn)), (max(0, Tn - L),
                                                  max(L, li), Tn)):
            if hw > lw:
                out.append((int(i), a0, lw, hw))
    return out


@torch.no_grad()
def _val_fm(net, D, wins, reps=4, L=M.W_CTX):
    """The FM loss (training observation noise, a fixed stream) on the
    validation windows' scored positions."""
    gen = torch.Generator(device="cpu").manual_seed(4321)
    dev = D.dev
    ii = torch.tensor([w[0] for w in wins], device=dev)
    a = torch.tensor([w[1] for w in wins], device=dev)
    lw = torch.tensor([w[2] for w in wins], device=dev)[:, None]
    hw = torch.tensor([w[3] for w in wins], device=dev)[:, None]
    pos = a[:, None] + torch.arange(L, device=dev)[None]
    tot = 0.0
    for _ in range(reps):
        tok, tgt, ok = M.window_tokens(D, ii, a, L,
                                       M.obs_sd(len(ii), dev, gen), gen)
        ok = ok & (pos >= lw) & (pos < hw)
        tot += M.fm_loss_h(net, net.encode(tok), tgt, ok, gen).item()
    return tot / reps


def adapt_head(ck, pool, holdout=None, split_frac=0.7, steps=4000,
               check=250, patience=4, lr=1e-4, batch=24, seed=0, log=print,
               val_lo=40):
    """Head-only adaptation (DEFECTS M11) of the checkpoint ck (restarted
    every time) on the Pool3 `pool`, then the spread by calibrate_spread.

    holdout None (fewer than 4 episodes): a time split inside each
    episode, steps < split_frac x len train, validation scores the steps
    after with the full history before them. holdout = whole episodes (one
    per leg): the others train on complete episodes, validation (early
    stopping and s) scores the held-out episodes from position val_lo.
    Only net.head trains (AdamW lr 1e-4, wd 0, `batch` windows of 256, the
    FM loss with the training observation noise, clip 1.0); validation FM
    every `check` steps, patience `patience`, keep the best (step 0
    included). Returns (net, info) with the head state, s, the validation
    coverage table and the UNADAPTED net's one-step coverage on the same
    positions (at s = 1 and at its own best s)."""
    dev = pool.dev
    net = load_net(ck, dev)
    base_net = load_net(ck, dev)
    for p in net.parameters():
        p.requires_grad_(False)
    train_p = list(net.head.parameters())
    for p in train_p:
        p.requires_grad_(True)
    n = pool.n
    Lmax = pool.len.cpu()
    if holdout is None:
        tr = torch.arange(n)
        cut = (Lmax.float() * split_frac).long()
        lim = cut
        va, lo = list(range(n)), {i: int(cut[i]) for i in range(n)}
    else:
        hs = set(int(i) for i in holdout)
        tr = torch.tensor([i for i in range(n) if i not in hs])
        lim = Lmax
        va, lo = sorted(hs), val_lo
    wins = _val_windows(pool, va, lo)
    lim_d = lim.to(dev)
    L = M.W_CTX
    opt = torch.optim.AdamW(train_p, lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    snap = lambda: {k_: x.detach().clone()  # noqa: E731
                    for k_, x in net.head.state_dict().items()}
    v0 = _val_fm(net, pool, wins)
    best = dict(v=v0, it=0, state=snap(), bad=0)
    t0 = time.time()
    it = 0
    # every training window is W_CTX long, whatever the episode lengths: a
    # short episode (one that went non-finite early) gets start 0 from
    # fm_draw's clamp and its positions >= lim are masked below (the
    # encoder is causal, so nothing scored sees them). One window length
    # from the shortest episode would shrink every window of every later
    # adaptation.
    n_short = int((lim[tr] < L + 1).sum())
    for it in range(1, steps + 1):
        ii, a = M.fm_draw(tr, batch, g, lim, L, dev)
        tok, tgt, ok = M.window_tokens(pool, ii, a, L,
                                       M.obs_sd(len(ii), dev, g), g)
        pos = a[:, None] + torch.arange(L, device=dev)[None]
        ok = ok & (pos < lim_d[ii][:, None])
        with torch.no_grad():
            h = net.encode(tok)
        loss = M.fm_loss_h(net, h, tgt, ok, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(train_p, 1.0)
        opt.step()
        if it % check == 0 or it == steps:
            v = _val_fm(net, pool, wins)
            if v < best["v"]:
                best.update(v=v, it=it, state=snap(), bad=0)
            else:
                best["bad"] += 1
            if best["bad"] >= patience:
                break
    net.head.load_state_dict(best["state"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    s, tab = calibrate_spread(net, pool, va, lo)
    s0, tab0 = calibrate_spread(base_net, pool, va, lo)
    info = dict(best_it=best["it"], val_fm0=v0, val_fm=best["v"],
                steps_run=it, s=s, cov=tab, prior_s=s0, prior_cov=tab0,
                n_eps=n, holdout=None if holdout is None else sorted(hs),
                val_lo=lo, time_s=time.time() - t0, win=L,
                n_short=n_short,
                head={k_: x.cpu() for k_, x in best["state"].items()})
    how = (f"time split at {split_frac:.0%} of each episode"
           if holdout is None else f"held out {sorted(hs)}")
    if n_short:
        how += (f"; {n_short} training episode(s) shorter than one window, "
                "masked")
    log(f"    adapt: {n} episodes ({how}), kept step {best['it']} of {it} (val FM {v0:.4f} -> {best['v']:.4f});"
        f" s = {s}: cov {tab[s]['cov_cost']:.2f} (5-ch {tab[s]['cov5']:.2f}),"
        f" at s = 1 {tab[1.0]['cov_cost']:.2f}; unadapted at s = 1 "
        f"{tab0[1.0]['cov_cost']:.2f} (5-ch {tab0[1.0]['cov5']:.2f}), its own "
        f"s {s0}; {info['time_s']:.0f} s")
    return net, info


# ------------------------------------------------ horizon check on Cb
def cb_blocks(D, k0=48, stride=H, max_eps=None):
    """Block starts of a block split (commands fixed in advance for 24
    steps: an unconfounded multi-step truth): (ep, k) arrays."""
    ep, kk = [], []
    n = D.n if max_eps is None else min(D.n, max_eps)
    for i in range(n):
        for k in range(k0, int(D.len[i]) - H + 1, stride):
            ep.append(i)
            kk.append(k)
    return np.array(ep), np.array(kk)


def _costs_of(states, xs0, link, en, u_ref):
    """Per-step link cost, speed cost and exceedance (numpy, (N, H)) of
    state sequences (N, H, 10) (reduced) from start states xs0 (N, 14):
    the impact threshold at the block's start speed for every step (as
    task_cost reads it at the measured speed), exceedance = 1 where the
    step-mean |a_cg| is above it (the link cost is > 0), the speed term
    with phi = the heading at the block start (the Cb blocks have no
    track)."""
    dtc = en["dt"] * en["sub"]
    zd = np.concatenate([xs0[:, None, 8], states[..., 4]], 1)
    abar = np.abs(np.diff(zd, axis=1)) / (dtc * G)
    u0 = xs0[:, None, 6]
    lk = link.cost(abar, u0)
    exc = (abar > link.A(u0)).astype(float)
    d = states[..., 7] - xs0[:, None, 5]
    ua = states[..., 2] * np.cos(d) - states[..., 9] * np.sin(d)
    return lk, np.maximum(0.0, (u_ref - ua) / u_ref), exc


@torch.no_grad()
def horizon_check(models, D, en, link, u_ref, n_samp=16, batch=48,
                  max_eps=None, log=print):
    """Rolled vs true cost per horizon step on the Cb blocks (steer None,
    the block plans): for each model name -> (net, spread s) (net None:
    m0, the deterministic e = 0 rollout, one sample), the sample means of
    the impact-link cost, of the exceedance indicator and of the speed
    cost at step j, against the same functions of the TRUE next states,
    per block (with the block start speed for binning). The same base
    draws for every net (x its s), whatever set of models a call gets.
    Returns dict(true_imp, true_spd, true_exc, u0 (N,), ep, k, per model
    (imp, spd, exc) (N, H))."""
    from learn.meta.data2 import plant_to_reduced
    ep, kk = cb_blocks(D, max_eps=max_eps)
    N = len(ep)
    XS = D.XS.astype(float)
    xs0 = XS[ep, kk]
    U = D.Uraw.astype(float)
    plans = np.stack([U[i, k:k + H] for i, k in zip(ep, kk)])[:, None]
    truth = plant_to_reduced(np.stack([XS[i, k + 1:k + H + 1]
                                       for i, k in zip(ep, kk)]).reshape(
        -1, 14)).reshape(N, H, 10)
    t_imp, t_spd, t_exc = _costs_of(truth, xs0, link, en, u_ref)
    gen = torch.Generator().manual_seed(77)
    base = torch.randn((N, 1, n_samp, H, M.C7), generator=gen)
    out = dict(true_imp=t_imp, true_spd=t_spd, true_exc=t_exc, u0=xs0[:, 6],
               ep=ep, k=kk)
    for name, (net, s) in models.items():
        t0 = time.time()
        if net is None:
            _, st, _ = rollout_zero(torch.tensor(plans), xs0, en)
            imp, spd, exc = _costs_of(st[:, 0, 0].numpy(), xs0, link, en,
                                      u_ref)
        else:
            imp, spd, exc = (np.zeros((N, H)) for _ in range(3))
            for b0 in range(0, N, batch):
                sl = slice(b0, min(N, b0 + batch))
                _, st, _ = M.rollout_core(net, D, ep[sl], kk[sl], plans[sl],
                                          xs0[sl], en, n_samp,
                                          base=(base[sl] * s).to(D.dev))
                st = st[:, 0].cpu().numpy()              # (nb, S, H, 10)
                nb = st.shape[0]
                parts = _costs_of(st.reshape(nb * n_samp, H, 10),
                                  np.repeat(xs0[sl], n_samp, 0), link, en,
                                  u_ref)
                for dst, v in zip((imp, spd, exc), parts):
                    dst[sl] = v.reshape(nb, n_samp, H).mean(1)
        out[name] = (imp, spd, exc)
        log(f"    horizon check {name} (s = {s}): {N} blocks, "
            f"{time.time() - t0:.0f} s")
    return out


def impact_horizon(hc, names, decide=None, n_min=30, level_tol=1.5,
                   growth_tol=1.5, j_floor=4, smooth=3, min_events=5):
    """The horizon fix from horizon_check: ONE impact horizon for every
    new-controller row (the same objective in every row, so the paired
    differences measure the error model, not the cost), decided on the
    models `decide` (default: all `names`; m0 is reported, not deciding).

    Per model and step j the decision reads the EXCEEDANCE probability
    P(step-mean |a_cg| > A(u)), rolled (sample and block mean) against
    true (block mean): a bounded, tail-robust statistic (the mean link
    cost K max(0, x)^2 with the fitted K is dominated by a few tail
    samples; it is reported, not decided on). level = (expected rolled
    events + 1) / (true events + 1) over the blocks (the +1 keeps a step
    with ~no true events finite without hiding a model that predicts many);
    growth = the ratio of the exceedance increase between the slowest and
    the fastest populated start-speed bin, only where the truth rises by
    >= min_events events. Both from 3-step moving averages of the means
    (only the steps that exist at the ends). A step is 'bad' when either
    leaves [1 / tol, tol]; first_bad per model.

    j_imp = the first bad step over the deciding models, 24 if none, at
    least j_floor. early = {model: first_bad} for the deciding models that
    fail BELOW j_floor: a horizon cut cannot fix those (the kept steps are
    the bad ones) and the caller must stop or mark the rows as not
    interpretable. Returns (j_imp, table per model, early)."""
    decide = list(names) if decide is None else list(decide)
    b = np.digitize(hc["u0"] / KN, SPEED_BINS)
    bins = [bi for bi in np.unique(b) if (b == bi).sum() >= n_min]
    ker = np.ones(smooth)
    cnt = np.convolve(np.ones(H), ker, "same")
    ma = lambda x: np.convolve(x, ker, "same") / cnt   # noqa: E731
    ti, te = hc["true_imp"], hc["true_exc"]
    N = len(te)
    tab = {}
    first_bad = H
    for nm in names:
        imp, _, exc = hc[nm]
        lev = (ma(exc.mean(0)) * N + 1.0) / (ma(te.mean(0)) * N + 1.0)
        mean_ratio = ma(imp.mean(0)) / np.maximum(ma(ti.mean(0)), 1e-12)
        gro = np.full(H, np.nan)
        if len(bins) >= 2:
            lo_m, hi_m = b == bins[0], b == bins[-1]
            gt = ma(te[hi_m].mean(0) - te[lo_m].mean(0))
            gp = ma(exc[hi_m].mean(0) - exc[lo_m].mean(0))
            okg = gt * min(lo_m.sum(), hi_m.sum()) >= min_events
            gro = np.where(okg, gp / np.where(okg, gt, 1.0), np.nan)
        bad_l = np.abs(np.log(np.maximum(lev, 1e-9))) > math.log(level_tol)
        bad_g = np.abs(np.log(np.maximum(np.nan_to_num(gro, nan=1.0),
                                         1e-9))) > math.log(growth_tol)
        bad = bad_l | bad_g
        fb = int(np.argmax(bad)) if bad.any() else H
        if nm in decide:
            first_bad = min(first_bad, fb)
        tab[nm] = dict(level=lev.tolist(), mean_ratio=mean_ratio.tolist(),
                       growth=gro.tolist(), first_bad=fb,
                       decides=nm in decide)
    j_imp = H if first_bad >= H else max(j_floor, first_bad)
    early = {nm: tab[nm]["first_bad"] for nm in decide
             if tab[nm]["first_bad"] < j_floor}
    return j_imp, tab, early


def link_tail(hc, link, en):
    """How much the fitted link leans on rare samples: the share of the
    true link cost on the Cb blocks carried by the top 1% of step values,
    the true exceedance rate, and the per-step cost of a step at 1.5 A and
    at 2 A (K x dtc x 0.25, K x dtc)."""
    dtc = en["dt"] * en["sub"]
    v = np.sort(np.asarray(hc["true_imp"], float).ravel())[::-1]
    top = float(v[:max(1, len(v) // 100)].sum() / max(v.sum(), 1e-12))
    return dict(top1_share=top, exc_rate=float(np.mean(hc["true_exc"])),
                cost_15A=link.K * dtc * 0.25, cost_2A=link.K * dtc)


def format_horizon(hc, tab, names, link=None, en=None):
    ti, ts, te = hc["true_imp"], hc["true_spd"], hc["true_exc"]
    blocks = ((0, 1), (1, 5), (5, 12), (12, H))
    lines = []
    if link is not None:
        t = link_tail(hc, link, en)
        lines.append(
            f"  impact link tail sensitivity: A0 {link.A0:.3f} g, A1 "
            f"{link.A1:+.3f} g/kn, K {link.K:.1f}; one step at 1.5 A costs "
            f"{t['cost_15A']:.2f}, at 2 A {t['cost_2A']:.2f}; on the Cb "
            f"blocks the true exceedance rate is {t['exc_rate']:.3f} per step"
            f" and the top 1% of steps carry {t['top1_share']:.0%} of the "
            "true link cost")
    lines.append("  horizon check (Cb blocks, rolled vs truth) at steps 0 / "
                 "1-4 / 5-11 / 12-23: exceedance P(abar > A) [decides], mean "
                 "link cost [reported], speed cost")
    for nm in names:
        imp, spd, exc = hc[nm]
        le = " ".join(f"{exc[:, a:b].mean():.3f}/{te[:, a:b].mean():.3f}"
                      for a, b in blocks)
        li = " ".join(f"{imp[:, a:b].mean():.4f}/{ti[:, a:b].mean():.4f}"
                      for a, b in blocks)
        ls = " ".join(f"{spd[:, a:b].mean():.3f}/{ts[:, a:b].mean():.3f}"
                      for a, b in blocks)
        gr = np.asarray(tab[nm]["growth"], float)
        with np.errstate(all="ignore"):
            gtxt = " ".join(f"{np.nanmean(gr[a:b]):.2f}" if
                            np.isfinite(gr[a:b]).any() else "nan"
                            for a, b in blocks)
        lines.append(f"    {nm}{'' if tab[nm]['decides'] else ' (reported)'}"
                     f": exceedance {le} | impact link {li} | speed {ls} | "
                     f"growth ratio {gtxt} | first bad step "
                     f"{tab[nm]['first_bad']}")
    return "\n".join(lines)
