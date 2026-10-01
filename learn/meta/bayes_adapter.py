#!/usr/bin/env python3
"""
Narrowing a prior with observations, made explicit (learn/meta/
THEORY_ONLINE_FLOW_2026-10-01.md T11): a Bayesian low-rank adapter on the
trunk output of a trained one-step net,

    h' = h + B (A h),    A (r x d) fixed random N(0, 1/d),  theta = vec(B),
    prior theta ~ N(0, TAU_B^2 I)   (d r = 192 x 8 = 1536 numbers),

whose posterior N(mu, P) is updated once per new observation by an
extended-Kalman (Gauss-Newton / Laplace) step on the flow-matching
residuals of that observation:

    r(theta) = v_theta(y_s, s, h') - (y1 - y0)   (R draws of (y0, s), C7 each)
    J = dr/dtheta at mu,   S = J P J^T + R sigma^2 I,   K = P J^T S^-1
    mu <- mu - K r(mu),    P <- P - K J P,    P <- P + Q_RW TAU_B^2 I

* every observation is used exactly once (no window re-use), so the
  precision counts data the way "prior + n observations" does;
* the flow-matching loss is not a likelihood: the update is the Gauss-
  Newton step of a generalised-Bayes posterior with loss
  sum r^2 / (2 sigma^2); sigma^2 per channel is a running mean of the
  squared residuals, and the R noise draws of one observation share one
  observation's weight (variance x R);
* the prior width is set in OUTPUT units (TAU_V), not in parameter units:
  before the first update P0 is rescaled so that the prior's spread of the
  predicted velocity is TAU_V x the residual noise of that observation
  (only the width; the centre stays at the prior net);
* Q_RW adds a random-walk floor so the width does not shrink below the
  level a slowly changing boat allows (the Kalman steady state);
* rollouts: resample(S) draws S adapters from N(mu, P); apply() gives
  future s its own adapter for the whole horizon, so the not-yet-known
  part of the law accumulates as H^2 (T11), not H.

Only the adapter is learned here; the trunk and the flow head stay at the
prior weights (no window re-fit, T5.2). The safety head, when present, gets
a point update on the newest observation only (pinball loss, pulled to its
prior with 1 / n, n = observations seen).
"""
import math

import numpy as np
import torch
from torch import nn

from learn.meta import model3 as M

RANK = 8
TAU_B = 0.05          # initial parameter-space width, rescaled below
TAU_V = 0.5           # prior width in output units: the adapter's prior
                      # spread in the predicted velocity = TAU_V x the
                      # residual noise, i.e. the prior is worth ~1/TAU_V^2
                      # = 4 observations per excited direction [assumption]
REPS_B = 8
Q_RW = 1e-4
SIG_EMA = 0.02
LR_SAFETY_B = 1e-3
TAU_SAFETY = 0.02


def ekf_update(mu, P, J, r, rvar):
    """One extended-Kalman step. mu (p,), P (p, p), J (m, p) the Jacobian
    of the residuals r (m,) at mu, rvar (m,) their variances. Returns
    (mu', P'). Exact Bayes for residuals linear in the parameters."""
    PJt = P @ J.T
    S = J @ PJt + torch.diag(rvar)
    K = torch.linalg.solve(S, PJt.T).T
    mu = mu - K @ r
    P = P - K @ PJt.T
    return mu, 0.5 * (P + P.T)


class BayesAdapter:
    """The adapter and its posterior (module docstring)."""

    def __init__(self, d, dev, rank=RANK, tau=TAU_B, seed=0):
        g = torch.Generator().manual_seed(int(seed))
        self.d, self.r, self.dev, self.tau = d, rank, dev, float(tau)
        self.A = (torch.randn(rank, d, generator=g) / math.sqrt(d)).to(dev)
        p = d * rank
        self.mu = torch.zeros(p, dtype=torch.float64, device=dev)
        self.P = torch.eye(p, dtype=torch.float64, device=dev) * self.tau ** 2
        self.samples = None                    # (S, d, r) for the next apply

    # ---------------------------------------------------------- draws
    def _chol(self):
        p = self.P.shape[0]
        jit = 1e-12 * self.tau ** 2
        for _ in range(6):
            try:
                return torch.linalg.cholesky(
                    self.P + jit * torch.eye(p, dtype=self.P.dtype,
                                             device=self.P.device))
            except RuntimeError:
                jit *= 100.0
        w, V = torch.linalg.eigh(self.P)
        return V * w.clamp(min=0).sqrt()

    def draw(self, n, gen):
        """n adapters (n, d, r) from N(mu, P)."""
        z = torch.randn(n, self.mu.numel(), generator=gen,
                        dtype=torch.float64).to(self.dev)
        th = self.mu + z @ self._chol().T
        return th.view(n, self.d, self.r).float()

    def resample(self, n, gen):
        """Fix n adapters for the next rollout: future s uses samples[s]."""
        self.samples = self.draw(n, gen)

    def mean_B(self):
        return self.mu.view(self.d, self.r).float()

    # ---------------------------------------------------------- apply
    def apply(self, h):
        """h (..., S, d) -> h' with future s using samples[s] (mean adapter
        when no samples are set or their count differs from S)."""
        Ah = h @ self.A.T                                     # (..., S, r)
        if self.samples is not None and h.dim() >= 2 \
                and h.shape[-2] == self.samples.shape[0]:
            return h + torch.einsum("...sr,sdr->...sd", Ah, self.samples)
        return h + Ah @ self.mean_B().T

    def apply_mean(self, h):
        return h + (h @ self.A.T) @ self.mean_B().T

    def sample_h(self, h, n, gen):
        """n adapted copies (n, d) of one condition h (d,), one adapter each
        (the predictive's epistemic part); does not touch self.samples."""
        Bs = self.draw(n, gen)
        Ah = self.A @ h                                       # (r,)
        return h[None] + torch.einsum("r,ndr->nd", Ah, Bs)

    def width(self):
        """trace(P) / trace(P0): 1 at the prior, -> 0 as data narrow it."""
        p = self.P.shape[0]
        return float(torch.trace(self.P) / (p * self.tau ** 2))


class BayesLearner:
    """Predict-then-learn with the Bayesian adapter for one live row. Same
    interface as online_stream.OnlineLearner (step, rebind, state,
    load_state, summary); attaches the adapter as net.bayes, which
    mpc_constrained.rollout_tap and online_stream.Monitor use."""

    def __init__(self, net, live, b, ctx, variant, head=None, seed=0,
                 tau=TAU_B, rank=RANK, reps=REPS_B, q_rw=Q_RW, fc=None):
        self.net, self.D, self.b, self.ctx = net, live, b, int(ctx)
        self.variant, self.head = variant, head
        self.reps, self.q = int(reps), float(q_rw)
        self.fc = {**dict(lam=0.1, hp=M.HB, msd=0.01), **(fc or {})}
        for p in net.parameters():
            p.requires_grad_(False)
        dev = next(net.parameters()).device
        self.ad = BayesAdapter(net.d, dev, rank, tau, seed=seed + 31)
        net.bayes = self.ad
        self.sig2 = None
        self.snr0 = self.width_data = float("nan")
        self.scaled = False
        self.gen = torch.Generator().manual_seed(seed + 103)
        self.n = self.n0 = 0
        if head is not None:
            for p in head.parameters():
                p.requires_grad_(True)
            self.hprior = [p.detach().clone() for p in head.parameters()]
            self.hopt = torch.optim.Adam(head.parameters(), LR_SAFETY_B)
        self.log = dict(fm=[], head=[], width=[])

    # ------------------------------------------------------------ data
    def _h_at(self, k):
        """Trunk output at token k of the measured window ending at k (the
        Monitor's convention), no gradient."""
        from learn.meta.online_stream import _tokens_measured
        D, b, L = self.D, self.b, self.ctx
        a = max(0, k - L + 1)
        tok, _, _ = _tokens_measured(D, b, a, L, self.variant, self.fc,
                                     self.gen)
        with torch.no_grad():
            return self.net.encode(tok)[0, k - a]

    def _residual_fn(self, h, yt, tau, target):
        net, A, d, r = self.net, self.ad.A, self.ad.d, self.ad.r
        R = yt.shape[0]

        def res(theta):
            Bm = theta.view(d, r).to(h.dtype)
            hp = h + Bm @ (A @ h)
            v = net.velocity(yt, tau, hp.expand(R, d))
            return (v - target).reshape(-1)
        return res

    # ------------------------------------------------------------ step
    def step(self, k):
        """One posterior update with the observation of step k (call after
        the plant advanced step k and the live data were pushed)."""
        D, b = self.D, self.b
        self.n = self.n0 + k + 1
        if not np.isfinite(D.E0[b, k]).all():
            return
        h = self._h_at(k)
        dev = h.device
        tgt = D.E[b, k].to(dev)
        R = self.reps
        y1 = torch.asinh(tgt / M.KAPPA)[None].expand(R, M.C7)
        y0 = torch.randn(R, M.C7, generator=self.gen).to(dev)
        tau = torch.rand(R, generator=self.gen).to(dev)
        yt = (1 - tau[:, None]) * y0 + tau[:, None] * y1
        res = self._residual_fn(h, yt, tau, y1 - y0)
        mu32 = self.ad.mu.float()
        r0 = res(mu32).detach()
        J = torch.func.jacrev(res)(mu32).detach()
        rr = r0.view(R, M.C7)
        ms = (rr.double() ** 2).mean(0)
        self.sig2 = ms if self.sig2 is None else \
            (1 - SIG_EMA) * self.sig2 + SIG_EMA * ms
        rvar = (self.sig2.clamp(min=1e-6) * R).repeat(R)
        Jd = J.double()
        if not self.scaled:
            # prior width in output units (module docstring, TAU_V)
            g2 = float((Jd * Jd).sum()) / self.ad.tau ** 2  # per unit tau^2
            t2 = TAU_V ** 2 * float(rvar.sum() / R) / max(g2, 1e-300)
            self.ad.P = self.ad.P * (t2 / self.ad.tau ** 2)
            self.ad.tau = math.sqrt(t2)
            self.scaled = True
        # diagnostics on the directions this observation excites: the
        # prior's predictive spread vs the noise (signal-to-noise of one
        # observation), and the width left there now relative to the prior
        p0 = float((Jd * Jd).sum()) * self.ad.tau ** 2
        self.snr0 = p0 / float(rvar.sum() / R)   # per observation
        mu, P = ekf_update(self.ad.mu, self.ad.P, Jd, r0.double(), rvar)
        self.width_data = float(torch.einsum("mp,pq,mq->", Jd, P, Jd)) \
            / max(p0, 1e-300)
        P = P + self.q * self.ad.tau ** 2 * torch.eye(
            P.shape[0], dtype=P.dtype, device=P.device)
        self.ad.mu, self.ad.P = mu, P
        self.log["fm"].append(float(ms.mean()))
        self.log["width"].append(self.ad.width())
        if self.head is not None:
            self._head_step(h, tgt, k)

    def _head_step(self, h, tgt, k):
        from learn.meta.safety_head import pinball
        y = self.D.Y[self.b, k]
        if not bool(torch.isfinite(y).all()):
            return
        hd = self.head
        with torch.enable_grad():
            q = hd(self.ad.apply_mean(h)[None], tgt[None], raw=False)
            yn = (y.to(h.device) - hd.y_mu) / hd.y_sd
            lh = pinball(q, yn[None], hd.qs).sum()
            pull = sum(((p - p0) ** 2).sum() for p, p0 in
                       zip(hd.parameters(), self.hprior)) / (
                2 * TAU_SAFETY ** 2 * max(self.n, 1))
            loss = lh + pull
            self.hopt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(hd.parameters()), 1.0)
            self.hopt.step()
        self.log["head"].append(float(lh))

    # ------------------------------------------------------------ carry
    def rebind(self, live, b=0):
        self.D, self.b, self.n0 = live, b, self.n

    def state(self):
        return dict(mu=self.ad.mu, P=self.ad.P, sig2=self.sig2, n=self.n,
                    tau=self.ad.tau, scaled=self.scaled,
                    head=(self.head.state_dict() if self.head is not None
                          else None),
                    hopt=(self.hopt.state_dict() if self.head is not None
                          else None))

    def load_state(self, st):
        self.ad.mu, self.ad.P = st["mu"], st["P"]
        self.ad.tau, self.scaled = float(st["tau"]), bool(st["scaled"])
        self.sig2 = st["sig2"]
        self.n = self.n0 = int(st["n"])
        if self.head is not None and st.get("head") is not None:
            self.head.load_state_dict(st["head"])
            self.hopt.load_state_dict(st["hopt"])

    def summary(self):
        f = lambda x: float(np.mean(x[-60:])) if x else float("nan")  # noqa
        return dict(n=self.n, fm_last60=f(self.log["fm"]),
                    head_last60=f(self.log["head"]),
                    width=self.ad.width(), width_data=self.width_data,
                    snr0=self.snr0,
                    mu_norm=float(self.ad.mu.norm()),
                    pull_last60=0.0, dev_sq=float((self.ad.mu ** 2).sum()))
