#!/usr/bin/env python3
"""
Cheaper uses of the learned error model inside the SAME MPPI controller
(learn/meta/RELATED_WORK_CONTROL_USE.md, candidate variants C and A; the
full controller: learn/meta/mpc_learned.LearnedMPPI).

Every variant here IS LearnedMPPI -- the same 7 knots, K candidates, warm
start and one-step shift, knot-noise and base-noise streams (drawn by the
parent's draws(), so a row's knot noise at a call is the parent's), S
samples, spread, SteerT nozzle, task_cost with the impact link and its
horizon j_imp, BAD_COST capping, softmax with lam -- except for WHERE the
candidates' errors come from. The candidates are rolled by Model0T plus
GIVEN per-sample errors (rollout_with_e: rollout_core's own state update,
on the CPU in float64, as the m0 path), and the learned model is called
once per control step on a few plans only:

  hold     rollout_core over ONE step along the warm-start plan's first
           command, S samples with the row's base draw at horizon step 0
           (x spread): the step-0 error samples e0_s. Candidate c, sample
           s, horizon step j gets e0_s w_j with w_j = exp(-j dtc / tau)
           (tau = 1 s by default) or w_j = 1 (tau = inf: held constant,
           the offset-free MPC baseline). S sequences of depth 1 per call.
  nominal  rollout_core over the full horizon along the warm-start plan, S
           samples with the row's base draw (the parent's base for this
           row and step): the per-sample error trajectories e_s,j, added
           to every candidate (a plan-independent correction). S
           sequences of depth 24.
  sens     nominal plus 7 plans, each the warm-start plan with ONE knot
           nudged by delta (+delta, or -delta where +delta would leave
           [floor, 1]), all 8 in one rollout_core call on the same base
           (common random numbers: the difference is the plan's effect on
           each sample path, no sampling noise). D_i = (e^(i) - e^nom) /
           delta_i per sample, step and channel; candidate c gets
               e = e^nom + clip(sum_i D_i clip(dk_i, -r, r), +-c_max e_sd)
           with dk = its knots - the warm-start knots. Two guards, both
           because a candidate can sit far from the warm start (knot sd
           0.25, thrust up to +-0.5 away) and a finite difference across
           an event (a jump present in one plan's sample path and not the
           other's) is large:
             trust  r = SIGMA (one knot-noise sd): beyond it the linear
                    model is held at its boundary value, not extrapolated;
             cap    each correction element at most c_max = 2 training
                    error sd of its channel (the net's e_sd).
           8 S sequences of depth 24.

hold's step-0 errors depend on the warm-start command only; nominal's on
the warm-start plan only; sens adds the first-order plan effect. All three
keep S separate sample paths for the cost (the impact tail stays priced
per sample, never on a mean), with the same base draw for every candidate.

run_group_v runs a variant through mpc_learned.run_group itself (its loop,
timing and metrics, not a copy: the controller class is swapped for the
call), and adds n_seq (learned-model sequences per call and row), depth
(their length in steps) and the variant's settings to each row.
"""
import math

import numpy as np
import torch

from learn.meta import model3 as M
from learn.meta import mpc_learned as ML

VARIANTS = ("hold", "nominal", "sens")
TAU = 1.0                  # s, hold's decay time constant (inf: constant)
DELTA = 0.05               # sens: knot nudge (thrust fraction)
C_MAX = 2.0                # sens: correction cap, in training error sd


def decay_weights(tau, en, horizon=ML.H):
    """w_j = exp(-j dtc / tau), j = 0..H-1 (w_0 = 1 exactly); tau = inf or
    None: all 1 (held constant). float64 (H,)."""
    if tau is None or math.isinf(tau):
        return torch.ones(horizon, dtype=torch.float64)
    dtc = en["dt"] * en["sub"]
    return torch.exp(-torch.arange(horizon, dtype=torch.float64) * dtc
                     / float(tau))


def rollout_with_e(plans, xs_k, en, e_raw, steer=None):
    """Model0T rollouts of plans (B, P, H, 2) from the plant states xs_k
    (B, 14) plus GIVEN raw errors e_raw (B, Pe, S, H, 10) float64 (Pe = 1:
    the same errors for every plan, broadcast; or Pe = P), on e_raw's
    device: exactly rollout_core's update, s_{j+1} = Model0T(s_j, U_j) +
    (e_raw_j[:8] dtc) @ sel on the E7 columns and the actuator column
    clip(U_j + e_raw_j[8:] dtc), U_j the plan's command or steer's per-
    sample output. e_raw = 0 gives rollout_zero's states exactly (x + 0.0
    = x), repeated over S. Returns (states (B, P, S, H, 10), actuator
    positions (B, P, S, H, 2))."""
    from learn.meta.data3 import E7
    dev = e_raw.device
    pl = torch.as_tensor(plans, dtype=torch.float64, device=dev)
    B, P, Hh, _ = pl.shape
    S = e_raw.shape[2]
    assert e_raw.shape[3] >= Hh and e_raw.shape[1] in (1, P), e_raw.shape
    xs = torch.as_tensor(np.asarray(xs_k, float), dtype=torch.float64,
                         device=dev)
    sr = M.plant_to_reduced_t(xs)[:, None, None].expand(B, P, S, 10)
    n_st = len(E7)
    sel = torch.zeros(n_st, 10, dtype=torch.float64, device=dev)
    sel[torch.arange(n_st), torch.tensor(E7)] = 1.0
    lo = torch.tensor([0.0, -1.0], dtype=torch.float64, device=dev)
    hi = torch.tensor([1.0, 1.0], dtype=torch.float64, device=dev)
    dtc = en["dt"] * en["sub"]
    m0 = M.model0t(en)
    states, acts = [], []
    for j in range(Hh):
        Uj = pl[:, :, None, j]
        if steer is not None:
            Uj = steer(j, sr, Uj)
        ej = e_raw[:, :, :, j]
        sr = m0(sr, Uj) + (ej[..., :n_st] * dtc) @ sel
        act = torch.maximum(torch.minimum(Uj + ej[..., n_st:n_st + 2] * dtc,
                                          hi), lo)
        states.append(sr.expand(B, P, S, 10))
        acts.append(act.expand(B, P, S, 2))
    return torch.stack(states, 3), torch.stack(acts, 3)


def sens_combine(e8, dsign, dk, trust, cap):
    """sens's candidate errors from one 8-plan rollout: e8 (B, 8, S, H, C)
    raw (plan 0 the warm start, plan 1 + i knot i nudged by dsign (B, 7)),
    dk (B, K, 7) the candidates' knot offsets from the warm start, trust
    the knot-offset clip r, cap (C,) the per-channel correction cap.
    Returns (e (B, K, S, H, C), D (B, 7, S, H, C)); D = 0 gives e = e^nom
    exactly."""
    e_nom = e8[:, :1]
    D = (e8[:, 1:] - e_nom) / dsign[:, :, None, None, None]
    d = torch.as_tensor(dk, dtype=e8.dtype, device=e8.device).clamp(
        -trust, trust)
    corr = torch.einsum("bki,bishc->bkshc", d, D)
    cap = torch.as_tensor(cap, dtype=e8.dtype, device=e8.device)
    corr = torch.maximum(torch.minimum(corr, cap), -cap)
    return e_nom + corr, D


class VariantMPPI(ML.LearnedMPPI):
    """LearnedMPPI whose candidates get their errors from `variant` (hold,
    nominal, sens; module docstring) instead of one learned rollout per
    candidate. Same constructor as the parent plus the variant's settings;
    net is required (the e = 0 controller is the parent with net None)."""

    def __init__(self, jobs, missions, en, link, net=None, live=None,
                 variant="nominal", tau=TAU, delta=DELTA, trust=ML.SIGMA,
                 c_max=C_MAX, **kw):
        assert net is not None, "a variant needs a learned net"
        assert variant in VARIANTS, variant
        super().__init__(jobs, missions, en, link, net=net, live=live, **kw)
        self.variant, self.tau, self.delta = variant, tau, float(delta)
        self.trust, self.c_max = float(trust), float(c_max)
        self.w = decay_weights(tau, en)
        e_sd = torch.as_tensor(live.stats["e_sd"])
        self.e_sd_dev = e_sd.to(self.dev)           # rollout_core's cast
        self.cap = (self.c_max * e_sd).double()
        self.n_plans = dict(hold=1, nominal=1, sens=1 + ML.N_KNOTS)[variant]
        self.depth = 1 if variant == "hold" else ML.H
        self.n_seq = self.n_plans * self.S

    def row_base(self, bases, P):
        """(nb, P, S, H, 10): each row's draw x spread (as base_tensor),
        expanded over P plans."""
        bs = torch.stack(bases)[:, None]
        sp = self.spread
        bs = bs * (sp.view(1, 1, 1, ML.H, 1) if sp.dim() else sp)
        nb = bs.shape[0]
        return bs.to(self.dev).expand(nb, P, self.S, ML.H, M.C7)

    def _learned(self, rows, k, knots, base, xs, steer, Hh):
        """rollout_core on the plans expanded from knots (nb, P, 7), their
        first Hh steps: raw errors (nb, P, S, Hh, 10) float64 on the CPU,
        as rollout_core scales them (e x e_sd in float32, then float64)."""
        nb, P, _ = knots.shape
        plans = np.zeros((nb, P, ML.H, 2))
        plans[..., 0] = knots @ self.E.T
        pl = torch.tensor(plans[:, :, :Hh], dtype=torch.float64)
        e, _, _ = M.rollout_core(self.net, self.live, list(rows), [k] * nb,
                                 pl, xs, self.en, self.S,
                                 base=base[..., :Hh, :], steer=steer)
        return (e * self.e_sd_dev).double().cpu()

    def errors(self, rows, k, cand, bases, xs, steer):
        """The candidates' raw errors (nb, 1 or K, S, H, 10) on the CPU and
        a dict of what produced them (for capture and tests)."""
        warm = self.nominal[list(rows)].copy()                  # (nb, 7)
        nb = len(rows)
        if self.variant == "hold":
            e0 = self._learned(rows, k, warm[:, None], self.row_base(
                bases, 1), xs, steer, 1)                   # (nb, 1, S, 1, C)
            return e0 * self.w.view(1, 1, 1, ML.H, 1), dict(warm=warm, e0=e0)
        if self.variant == "nominal":
            e = self._learned(rows, k, warm[:, None], self.row_base(bases, 1),
                              xs, steer, ML.H)
            return e, dict(warm=warm)
        up = warm + self.delta <= 1.0
        dsign = np.where(up, self.delta, -self.delta)           # (nb, 7)
        knots = np.repeat(warm[:, None], 1 + ML.N_KNOTS, 1)
        idx = np.arange(ML.N_KNOTS)
        knots[:, 1 + idx, idx] += dsign
        e8 = self._learned(rows, k, knots, self.row_base(
            bases, 1 + ML.N_KNOTS), xs, steer, ML.H)
        e, D = sens_combine(e8, torch.as_tensor(dsign), cand - warm[:, None],
                            self.trust, self.cap)
        return e, dict(warm=warm, e8=e8, D=D, dsign=dsign)

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
        plans = np.zeros((nb, self.K, ML.H, 2))
        plans[..., 0] = thr
        ms = [self.ms[b] for b in rows]
        xs = np.stack([m.s for m in ms])
        phi = np.array([m.phi for m in ms])
        psi_i = [m.ep._psi_i for m in ms]
        steer_l = ML.SteerT(ms[0].ep, phi, psi_i, self.en)
        e_raw, info = self.errors(rows, k, cand, bases, xs, steer_l)
        steer = ML.SteerT(ms[0].ep, phi, psi_i, self.en)
        st, _ = rollout_with_e(torch.tensor(plans, dtype=torch.float64), xs,
                               self.en, e_raw, steer)
        # from here on LearnedMPPI.__call__ line by line
        m0 = ms[0]
        c, parts = ML.task_cost(st, thr, xs, phi, self.link, self.en,
                                m0.u_ref, m0.L)
        c_raw = c
        bad = ~torch.isfinite(c)
        clip = ~bad & (c > ML.BAD_COST)
        c = torch.where(bad | clip, torch.full_like(c, ML.BAD_COST), c)
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
                             phi=phi, psi_i=psi_i,
                             base_rows=torch.stack(bases), e_raw=e_raw,
                             states=st, cost=c_raw, cost_used=c, cand=cand,
                             cmds=cmds.copy(), **info)
        return cmds


def settings(variant, tau=TAU, delta=DELTA, trust=ML.SIGMA, c_max=C_MAX):
    """The settings that define a variant's row (for the cache key)."""
    d = dict(variant=variant)
    if variant == "hold":
        d["tau"] = None if tau is None or math.isinf(tau) else float(tau)
    if variant == "sens":
        d.update(delta=float(delta), trust=float(trust), c_max=float(c_max))
    return d


def run_group_v(jobs, en, link, variant, net, stats, spread=1.0, T=120.0,
                log=print, tag="", **kw):
    """mpc_learned.run_group with VariantMPPI(variant, **kw) as the
    controller (the class is swapped for the call only, so the loop, the
    timing and the metrics are run_group's own). Rows get n_seq (learned-
    model sequences per call and row), depth (their length in steps),
    n_seq_steps = n_seq x depth and the variant's settings."""
    made = []

    def factory(*a, **k):
        c = VariantMPPI(*a, variant=variant, **kw, **k)
        made.append(c)
        return c

    old = ML.LearnedMPPI
    ML.LearnedMPPI = factory
    try:
        rows = ML.run_group(jobs, en, link, net=net, stats=stats,
                            spread=spread, T=T, log=log, tag=tag)
    finally:
        ML.LearnedMPPI = old
    c = made[0]
    st = settings(variant, tau=c.tau, delta=c.delta, trust=c.trust,
                  c_max=c.c_max)
    for r in rows:
        r.update(n_seq=c.n_seq, depth=c.depth,
                 n_seq_steps=c.n_seq * c.depth, **st)
    return rows
