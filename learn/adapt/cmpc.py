#!/usr/bin/env python3
"""
The corrected MPC and the ways it can probe.

The MPC plans against the OPERATOR'S cost (learn/repro/task.py: speed
lost along the track, impacts above 1 g, cross-track) -- the objective is
known; what is unknown is how the boat responds. Two learned corrections
carry that (learn/adapt/blr.py):

  speed loss   an extra surge acceleration, r_u(speed, heading, thrust),
               added to the reduced model in every rollout. Learned from
               what the speed log does against what the model predicted.
  impacts      the expected impact cost per unit time, e(speed, heading),
               used directly in the cost. Learned from the accelerometer,
               one sample per ~2 s segment (a wave encounter or so). It
               is a STATISTIC, not a state: with no preview single impacts
               cannot be predicted, but how their average grows with speed
               and depends on the heading can.

Before any target data both corrections come from the source world (the
prior, studies/adapt_matrix.py), so the MPC starts from the low-fidelity
answer and the target data correct it.

Probing (only while learning; evaluation is always the posterior mean):
  none    plan with the posterior mean
  dither  the planning speed target moves by -4/-2/0/+2 kn every 20 s,
          amplitude falling to zero over the learning episodes
  etc     explore-then-commit: the first 4 learning episodes run a fixed
          schedule of speed offsets (30 s each), then plan with the mean
  ucb     optimism: plan with the impact cost one std BELOW the mean
  ts      Thompson sampling: every 30 s draw both corrections from their
          posteriors and plan as if the draw were true
  info    an information bonus: minus lambda x the fraction by which the
          plan's observations would shrink the variance of the slope
          d(impact cost)/d(speed) at the current heading -- the quantity
          the speed decision depends on; scaled by the share of the
          learning budget left, so it fades as learning ends
Every strategy, 'none' included, is guarded by a pessimistic bound: plans
whose impact cost at mean + 2 std exceeds E_LIM are penalised.

Two things the first review of this code found (both fixed):
  * An impact sample must be indexed by inputs fixed BEFORE its impacts:
    a slam slows the boat within the same 2 s, so pairing a segment's
    impacts with its own mean speed makes heavy segments look slow and
    the fitted slope comes out "faster = fewer impacts" -- the opposite of
    what controlled speed offsets show. Rows now carry the speed and
    intensity at the segment's START, and also the mean speed over the
    30 s before it ('slow' regressor: the speed level the MPC chooses,
    less entangled with single wave groups). A correction picks one.
  * The operator's speed term is one-sided (nothing is gained above
    25 kn) and thrust was free, so a lower planning target gave no reason
    to slow down: speed offsets only worked upwards, and the source
    prior's data all sat near 29 kn at half throttle. Planning now also
    prices running ABOVE its target (W_OVER, quadratic; planning only --
    the score never sees it), so the boat settles at its target and speed
    offsets act in both directions. (A small fuel price alone, W_THR, was
    tried first: its cost differences sat far below the MPPI temperature
    and the boat still ran at ~28 kn with a 21 kn target.)
"""
from collections import deque

import numpy as np

from control.mpc import MPPIController
from learn.adapt.blr import BLR
from learn.adapt.features import Features
from learn.repro.task import (A_LIM, G, K_A, K_Y, LEGS, Y_SCALE, Mission,
                              ctx)

STRATEGIES = ("none", "dither", "etc", "ucb", "ts", "info")
SEG = 8                        # control steps per impact sample (~1.9 s)
KN = 0.514444
E_LIM = 0.2                    # impact cost per second, pessimistic bound
BETA = 1.0                     # optimism, in posterior stds
LAM_INFO = 2.0                 # information bonus, in cost units
W_JUMP = 0.3                   # thrust smoothness (the hand MPC's value)
W_THR = 0.02                   # planning-only fuel price, per s at full thrust
W_OVER = 2.0                   # planning-only price of running above the target
SLOW_S = 30.0                  # window of the 'slow' speed regressor, s
DITHER_KN = (-4.0, -2.0, 0.0, 2.0)
ETC_EPISODES = 4


# ------------------------------------------------------------- the model
def broad_prior(feat, std, sigma2, grid=None):
    """N(0, tau^2 I) with tau chosen so the prior predictive std is `std`
    over typical inputs -- 'the correction could be about this big'."""
    un = np.linspace(-0.5, 0.2, 8)
    psi = np.linspace(-np.pi, np.pi, 9)
    thr = np.linspace(0.3, 0.9, 3)
    inten = np.linspace(0.6, 2.0, 4)
    U, P, T, I = np.meshgrid(un, psi, thr, inten)
    F = feat(U.ravel(), P.ravel(), T.ravel(), I.ravel())
    tau2 = std ** 2 / np.mean(np.sum(F ** 2, axis=1))
    return np.zeros(feat.dim), tau2 * np.eye(feat.dim), sigma2


class Corrections:
    """The two corrections and their features. `prior` = (imp, srg), each
    (m0, S0, sigma2); None = broad priors around zero."""

    IMP_STD, IMP_S2 = 0.05, 1e-2       # impact cost per s; per-segment noise
    SRG_STD, SRG_S2 = 0.5, 0.25        # m/s^2

    def __init__(self, kind="hand", intensity=False, prior=None,
                 regressor="fast"):
        self.kind, self.intensity = kind, intensity
        self.regressor = regressor          # 'fast' | 'slow' speed input
        self.f_imp = Features(kind, "impact", intensity, seed=11)
        self.f_srg = Features(kind, "surge", intensity, seed=12)
        if prior is None:
            prior = (broad_prior(self.f_imp, self.IMP_STD, self.IMP_S2),
                     broad_prior(self.f_srg, self.SRG_STD, self.SRG_S2))
        (mi, Si, si), (ms, Ss, ss) = prior
        self.imp = BLR(mi, Si, si, sigma2_floor=1e-4)
        self.srg = BLR(ms, Ss, ss, sigma2_floor=1e-3)

    def frozen(self):
        """The posteriors alone (no data): what an evaluation needs."""
        return dict(kind=self.kind, intensity=self.intensity,
                    regressor=self.regressor,
                    imp=(self.imp.m, self.imp.S, self.imp.sigma2),
                    srg=(self.srg.m, self.srg.S, self.srg.sigma2))

    @classmethod
    def from_frozen(cls, fz):
        return cls(fz["kind"], fz["intensity"], prior=(fz["imp"], fz["srg"]),
                   regressor=fz.get("regressor", "fast"))

    # raw rows -> the regressions (the same raw data fits any feature set)
    #   surge rows:  (un, psi, thrust fraction, intensity, residual)
    #   impact rows: (un at segment start, un over the 30 s before it,
    #                 psi at start, intensity at start, mean impact sample)
    def add_raw(self, surge_rows, imp_rows):
        if len(surge_rows):
            r = np.asarray(surge_rows)
            self.srg.add(self.f_srg(r[:, 0], r[:, 1], r[:, 2], r[:, 3]),
                         r[:, 4])
        if len(imp_rows):
            r = np.asarray(imp_rows)
            un = r[:, 1] if self.regressor == "slow" else r[:, 0]
            self.imp.add(self.f_imp(un, r[:, 2], None, r[:, 3]), r[:, 4])


def prior_from(post, widen_std, feat, s2_min):
    """A target prior from a source posterior: same mean, the source's
    covariance widened by a broad N(0, tau^2 I) -- 'the target may differ
    from the source by about widen_std'. The noise level starts no lower
    than s2_min: the source's is ~100x smaller than the target's (almost no
    impacts there), and until it is re-estimated every target row would
    count ~100x too much."""
    _, S_b, _ = broad_prior(feat, widen_std, 1.0)
    return post.m.copy(), post.S + S_b, max(post.sigma2, s2_min)


# ------------------------------------------------------------ the MPC
class CorrectedMPC(MPPIController):
    def __init__(self, model, preview, corr, strategy="none", **kw):
        super().__init__(model, preview, **kw)
        self.corr = corr
        self.strategy = strategy
        self.ctx = dict(phi=0.0, inten=0.0, explore=False, du_ref=0.0,
                        frac_left=1.0, un_now=0.0, psi_now=0.0)
        self.w_imp = self.w_srg = None          # Thompson draws

    def rollout_cost(self, s0, seq, t0, track_ref):
        K = seq.shape[0]
        s = np.repeat(s0[None, :], K, axis=0)
        cost = np.zeros(K)
        p, c = self.m.p, self.ctx
        U, L = self.u_ref, p["L"]
        u_tgt = U + c["du_ref"]
        explore = c["explore"]
        w_s = self.corr.srg.m if self.w_srg is None else self.w_srg
        w_i = None if self.w_imp is None else self.w_imp
        info = explore and self.strategy == "info"
        rows = []
        for h in range(self.H):
            t = t0 + h * self.dt
            psi_h = s[:, 7:8]
            ch, sh = np.cos(psi_h), np.sin(psi_h)
            xb = self.x_st[None, :, None]
            yb = self.y_off[None, None, :]
            xs = s[:, 0:1, None] + xb * ch[..., None] - yb * sh[..., None]
            ys = s[:, 1:2, None] + xb * sh[..., None] + yb * ch[..., None]
            eta = self.pv.at(xs, ys, t, h * self.dt)
            thr = np.clip(seq[:, 0, h], self.thrust_floor, 1.0)
            rud = np.clip(seq[:, 1, h], -1.0, 1.0) * self.rud_max
            # the impact rate of this step at the speed it starts from --
            # as the samples it was learnt from are indexed
            un0 = s[:, 2] / U - 1.0
            psi0 = s[:, 7] + c["phi"]           # world heading: the context
            Fi = self.corr.f_imp(un0, psi0, None, c["inten"])
            s, _, _ = self.m.step(s, thr * self.t_max, rud, eta, self.x_st,
                                  self.dt)
            un = s[:, 2] / U - 1.0
            psi = s[:, 7] + c["phi"]
            s[:, 2] += self.corr.f_srg(un, psi, thr, c["inten"]) @ w_s \
                * self.dt
            mean, std = self.corr.imp.predict(Fi)
            if w_i is not None:
                e = Fi @ w_i
            elif explore and self.strategy == "ucb":
                e = mean - BETA * std
            else:
                e = mean
            e = np.maximum(e, 0.0)
            ua = s[:, 2] * np.cos(s[:, 7]) - s[:, 9] * np.sin(s[:, 7])
            cost += (np.maximum(0.0, (u_tgt - ua) / U) + K_A * e
                     + W_OVER * np.maximum(0.0, (ua - u_tgt) / U) ** 2
                     + K_Y * (s[:, 1] / (Y_SCALE * L)) ** 2 + W_THR * thr
                     + 10.0 * np.maximum(0.0, mean + 2.0 * std - E_LIM)
                     ) * self.dt
            if info and h % 4 == 3:
                rows.append(Fi)
        cost += W_JUMP * np.sum(np.diff(seq[:, 0], axis=1) ** 2, axis=1)
        if info and rows:
            cvec = self.corr.f_imp.d_speed(c["un_now"], c["psi_now"], None,
                                           c["inten"])[0]
            v0 = float(cvec @ self.corr.imp.S0 @ cvec) + 1e-12
            # each row stands for 4 control steps, half an impact sample
            s2 = self.corr.imp.sigma2 * SEG / 4.0
            drop = sum(self.corr.imp.slope_var_drop(F, cvec, s2)
                       for F in rows)
            cost -= LAM_INFO * c["frac_left"] * drop / v0
        return cost


# ------------------------------------------------------------ an episode
def run_episode(corr, strategy, world, seed, leg, t_end, learn, explore,
                ep_idx=0, n_learn=1, rng_seed=0, du_const=None,
                collect_raw=False, on_step=None):
    """One episode of the corrected MPC. Returns (metrics, raw rows).

    learn    update the corrections online (every SEG control steps)
    explore  run the probing strategy (else: plan with the posterior mean)
    du_const planning speed offsets as a fixed schedule [(t_from, kn), ...]
             (source / data-collection runs), overriding the strategy
    on_step  f(mission) once before the first step and after every finite
             control step (a recorder, e.g. studies/mpc_compare's per-step
             speed / heave-rate log); it must not change the mission
    """
    ctx()
    rng = np.random.default_rng(rng_seed)

    def factory(red, pv, ep):
        c0 = ep.ctrl
        return CorrectedMPC(red, pv, corr, strategy, weights=c0.w,
                            dt_ctrl=c0.dt, u_ref=c0.u_ref,
                            seed=int(rng.integers(1 << 30)), n_samples=c0.K,
                            use_rudder=False)

    m = Mission(world, seed, leg, t_end=t_end, ctrl_factory=factory,
                track=LEGS[leg])
    ctrl, p = m.ep.ctrl, m.red.p
    frac_left = max(0.0, 1.0 - ep_idx / max(n_learn, 1))
    ctrl.ctx.update(phi=m.phi, explore=explore, frac_left=frac_left)
    hv = deque(maxlen=max(int(round(5.0 / m.dt_ctrl)), 2))
    us = deque(maxlen=max(int(round(SLOW_S / m.dt_ctrl)), 2))
    dtc = m.sub * m.dt
    seg_start = None
    srg_rows, imp_rows, seg = [], [], []
    raw_s, raw_i = [], []
    du, t_next = 0.0, 0.0
    amp = frac_left
    if on_step is not None:
        on_step(m)                      # the state at step 0
    while not m.done():
        t = m.t
        # the planning speed offset
        if du_const is not None:
            du = [kn for t0, kn in du_const if t >= t0][-1] * KN
        elif explore and strategy == "dither":
            if t >= t_next:
                du = amp * rng.choice(DITHER_KN) * KN
                t_next = t + 20.0
        elif explore and strategy == "etc" and ep_idx < ETC_EPISODES:
            du = DITHER_KN[int(t // 30.0) % len(DITHER_KN)] * KN
        else:
            du = 0.0
        if explore and strategy == "ts" and t >= t_next:
            ctrl.w_imp = corr.imp.sample(rng)
            ctrl.w_srg = corr.srg.sample(rng)
            t_next = t + 30.0
        s0 = m.s.copy()
        hv.append(s0[8])
        us.append(s0[6])
        inten = float(np.sqrt(np.mean(np.square(hv))))
        if not seg:
            # inputs of the coming impact sample, fixed before its impacts
            seg_start = (s0[6] / m.u_ref - 1.0,
                         float(np.mean(us)) / m.u_ref - 1.0, s0[5], inten)
        ctrl.ctx.update(inten=inten, du_ref=du,
                        un_now=s0[6] / m.u_ref - 1.0, psi_now=s0[5])
        thr_cmd, rud = m.mpc_command()
        m.advance(thr_cmd, rud)
        if not m.finite:
            break
        if on_step is not None:
            on_step(m)
        s1 = m.s
        # speed-loss sample: measured surge acceleration minus the model's,
        # with the thrust the plant actually applied over the step (the
        # first step of an episode is a start-up transient: skipped)
        thr_app = m.last_thr_app
        a_mod = (thr_app - p["k_drag"] * s0[6] * abs(s0[6])) / p["m_surge"]
        if m.k > 1:
            srg_rows.append((s0[6] / m.u_ref - 1.0, s0[5], thr_app / m.t_max,
                             inten, (s1[6] - s0[6]) / dtc - a_mod))
        # impact sample: this step's cost above the limit (task units / K_A)
        a = np.asarray(m._acg[-m.sub:])
        seg.append(float(np.mean(np.maximum(0.0, a / A_LIM - 1.0) ** 2)))
        if len(seg) == SEG:
            imp_rows.append(seg_start + (float(np.mean(seg)),))
            seg = []
            if learn:
                corr.add_raw(srg_rows, imp_rows)
            if collect_raw:
                raw_s += srg_rows
                raw_i += imp_rows
            srg_rows, imp_rows = [], []
    ctrl.w_imp = ctrl.w_srg = None
    raw = (raw_s, raw_i) if collect_raw else None
    return m.metrics(), raw
