#!/usr/bin/env python3
"""
Streaming online learning with a mismatch trigger (ROADMAP_2026-09-30
section 10, layers 2 and 3), for one live mission row at a time.

Every control step k, in time order:
  1. predict  (Monitor.predict, after the MPC chose the command u_k, before
              the plant moves): the trunk over the recorded window ending
              at token k (state x_k, the command actually applied, e_{k-1};
              for w / p nets the measured elevations, the preview at lead
              dtc / 2), 128 flow samples of e_k with the base noise x
              spread x the current inflation, and (head rows) the safety
              head's (APK_k, HMIN_k) for each sampled e_k, one inverse-CDF
              draw each -- the predictive distribution the MPC plans with;
  2. observe  (Monitor.update, after the plant step): the observed e_k
              against the predicted 90% and 99% bands of the five
              velocity channels; for head rows APK_k / HMIN_k against their
              90% bands and the tails (APK above q99, HMIN below q01);
  3. learn    (OnlineLearner.step, online rows only): one gradient step on
              the window ending at k (the flow head's FM loss, R noise
              draws per token, the trunk's own training token convention,
              so rollout-segment tokens are trained too; the safety head's
              pinball loss on the plant's observed APK / HMIN), the trunk
              frozen apart from rank-4 adapters on every layer's
              feed-forward weights.

Parameters that learn: the flow head (net.head), the LoRA adapters, the
safety head. The pull to the prior is a Gaussian prior on the parameter
deviations with sd TAU: per-token loss + |theta - theta_0|^2 / (2 TAU^2 n),
n = steps observed so far (the MAP objective of n tokens, so the pull
weakens as 1 / n as data accumulates). TAU, the learning rates and R are
set a priori (not fitted on target data).

Mismatch trigger: over the last N_WIN = 60 steps (15 s; at least 20), the
share of observations outside the predicted 90% bands (velocity channels
averaged; plus APK / HMIN for head rows), and over the last N_TAIL = 240
steps (at least 60) the share outside the 99% bands / beyond the tail
quantiles. "Clearly above" = above the nominal share by more than two
binomial standard deviations (0.10 + 2 sqrt(0.09 / n); 0.01 + 2 sqrt(
0.0099 / n)). While either is clearly above, the inflation of the spread
the constraints use grows by UP per step (max S_MAX); once both are at or
below nominal it shrinks by DOWN per step back to 1 ("until it recovers").
For proxy rows (no head) the APK prediction is a step mean (a lower bound
of the peak by construction): recorded, never used by the trigger.
"""
import math
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

from learn.meta import model3 as M

VEL_CH = (0, 1, 2, 3, 4)
TAU = 0.02
LR_HEAD, LR_LORA, LR_SAFETY = 1e-4, 1e-3, 1e-3
RANK, REPS = 4, 8
N_WIN, N_TAIL, N_MIN, N_MIN_TAIL = 60, 240, 20, 60
UP, DOWN, S_MAX = 1.03, 1.01, 3.0
N_PRED = 128


# ------------------------------------------------------------ adapters
class LoRA(nn.Module):
    """W -> W + B A (rank r), B zero at the start (the prior exactly)."""

    def __init__(self, n_out, n_in, r, gen):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, n_in, generator=gen)
                              / math.sqrt(n_in))
        self.B = nn.Parameter(torch.zeros(n_out, r))

    def forward(self, W):
        return W + self.B.to(W.dtype) @ self.A.to(W.dtype)


def add_lora(net, r=RANK, seed=0):
    """Rank-r adapters on linear1 / linear2 of every transformer layer (a
    parametrisation of .weight, so encode, kv_history and kv_step all see
    the adapted weights). Returns the adapter parameters."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for lay in net.tf.layers:
        for lin in (lay.linear1, lay.linear2):
            W = lin.weight
            mod = LoRA(W.shape[0], W.shape[1], r, gen).to(W.device)
            parametrize.register_parametrization(lin, "weight", mod)
            out += [mod.A, mod.B]
    return out


# ------------------------------------------------------------ tokens
def _tokens_measured(D, b, a, L, variant, fc, gen):
    """The window [a, a + L) of row b with every token measured (the
    one-step convention of model_preview.eval_one_step_p; model3's tokens
    for 'a')."""
    from learn.meta import model_preview as MP
    ii = torch.tensor([b], device=D.dev)
    aa = torch.tensor([a], device=D.dev)
    if variant == "a":
        tok, tgt, ok = M.window_tokens(D, ii, aa, L)
    else:
        hp = 0 if variant == "w" else int(fc["hp"])
        lam = 0.0 if variant == "w" else float(fc["lam"])
        cfg = MP.fixed_cfg(D, [a + L], lam, hp, fc["msd"], gen)
        tok, tgt, ok = MP.tokens_p(D, ii, aa, L, cfg, gen, obs=False)
    return tok, tgt, ok


# ------------------------------------------------------------ monitor
class Monitor:
    """One-step predictive check and the spread inflation (module
    docstring)."""

    def __init__(self, net, live, b, ctx, variant, head=None, spread=1.0,
                 fc=None, seed=0):
        self.net, self.D, self.b, self.ctx = net, live, b, int(ctx)
        self.variant, self.head, self.spread = variant, head, float(spread)
        self.fc = {**dict(lam=0.1, hp=M.HB, msd=0.01), **(fc or {})}
        self.gen = torch.Generator().manual_seed(seed)
        self.infl = 1.0
        self.h90, self.h99 = deque(maxlen=N_WIN), deque(maxlen=N_TAIL)
        self.log = dict(miss90=[], miss99=[], apk90=[], apk99=[],
                        hmin90=[], hmin01=[], infl=[], trig=[])

    @torch.no_grad()
    def predict(self, k, u_frac):
        """Predictive quantiles of step k under the applied command u_frac
        (thrust, nozzle fractions)."""
        D, b, L = self.D, self.b, self.ctx
        st = D.stats
        un = (np.asarray(u_frac, np.float32) - st["u_mu"]) / st["u_sd"]
        old = D.U[b, k].clone()
        D.U[b, k] = torch.as_tensor(un, dtype=torch.float32, device=D.dev)
        a = max(0, k - L + 1)
        tok, _, _ = _tokens_measured(D, b, a, L, self.variant, self.fc,
                                     self.gen)
        D.U[b, k] = old
        h = self.net.encode(tok)[0, k - a]
        y0 = torch.randn((N_PRED, M.C7), generator=self.gen) \
            * self.spread * self.infl
        e = self.net.sample_grad(h.expand(N_PRED, -1), y0.to(D.dev))
        lv = torch.tensor([0.005, 0.05, 0.5, 0.95, 0.995], device=D.dev)
        out = dict(qe=torch.quantile(e, lv, dim=0).cpu().numpy())  # (5, 10)
        if self.head is not None:
            from learn.meta.safety_head import inflate, inv_cdf
            q = inflate(self.head(h.expand(N_PRED, -1), e).double(),
                        self.infl)
            u = torch.rand((N_PRED, 2), generator=self.gen,
                           dtype=torch.float64).to(D.dev)
            v = inv_cdf(q, u)                                # (n, 2)
            lq = torch.tensor([0.01, 0.05, 0.95, 0.99], dtype=v.dtype,
                              device=D.dev)
            out["qs"] = torch.quantile(v, lq, dim=0).cpu().numpy()  # (4, 2)
        else:
            # proxy: the step-mean heave acceleration of each sample, the
            # e = 0 model's step plus the sampled heave-rate error
            env = D.env
            xs = torch.as_tensor(D.XS[b, k], dtype=torch.float64)
            sr = M.plant_to_reduced_t(xs)[None]
            U = torch.as_tensor(np.asarray(u_frac, float))[None]
            zd1 = float(M.model0t(env)(sr, U)[0, 4])
            dtc = env["dt"] * env["sub"]
            a = (zd1 - float(xs[8])) / dtc + e[:, 3].double() * float(
                st["e_sd"][3])
            out["proxy_apk"] = torch.quantile(a, torch.tensor(
                [0.05, 0.95, 0.99], dtype=torch.float64,
                device=a.device)).cpu().numpy()
        return out

    def update(self, pred, k):
        """Score the prediction of step k against what the plant did
        (D.E[b, k], D.APK[b, k], D.HMIN[b, k]) and move the inflation.
        Returns the inflation for the next step."""
        D, b = self.D, self.b
        e = D.E[b, k].cpu().numpy()
        qe = pred["qe"]
        ch = list(VEL_CH)
        m90 = float(np.mean((e[ch] < qe[1, ch]) | (e[ch] > qe[3, ch])))
        m99 = float(np.mean((e[ch] < qe[0, ch]) | (e[ch] > qe[4, ch])))
        ind90, ind99 = [m90], [m99]
        apk, hmin = float(D.APK[b, k]), float(D.HMIN[b, k])
        lg = self.log
        lg["miss90"].append(m90)
        lg["miss99"].append(m99)
        if "qs" in pred and np.isfinite(apk) and np.isfinite(hmin):
            q = pred["qs"]
            a90 = float(apk < q[1, 0] or apk > q[2, 0])
            a99 = float(apk > q[3, 0])
            h90 = float(hmin < q[1, 1] or hmin > q[2, 1])
            h01 = float(hmin < q[0, 1])
            ind90 += [a90, h90]
            ind99 += [a99, h01]
            lg["apk90"].append(a90)
            lg["apk99"].append(a99)
            lg["hmin90"].append(h90)
            lg["hmin01"].append(h01)
        elif "proxy_apk" in pred and np.isfinite(apk):
            lg["apk90"].append(float(apk > pred["proxy_apk"][1]))
            lg["apk99"].append(float(apk > pred["proxy_apk"][2]))
        self.h90.append(float(np.mean(ind90)))
        self.h99.append(float(np.mean(ind99)))
        n90, n99 = len(self.h90), len(self.h99)
        p90, p99 = float(np.mean(self.h90)), float(np.mean(self.h99))
        hi90 = n90 >= N_MIN and p90 > 0.10 + 2.0 * math.sqrt(0.09 / n90)
        hi99 = n99 >= N_MIN_TAIL and p99 > 0.01 + 2.0 * math.sqrt(
            0.0099 / n99)
        if hi90 or hi99:
            self.infl = min(S_MAX, self.infl * UP)
        elif p90 <= 0.10 and (n99 < N_MIN_TAIL or p99 <= 0.01):
            self.infl = max(1.0, self.infl / DOWN)
        lg["infl"].append(self.infl)
        lg["trig"].append(bool(hi90 or hi99))
        return self.infl

    def summary(self):
        lg = self.log
        f = lambda x: float(np.mean(x)) if x else float("nan")  # noqa: E731
        return dict(cov90_vel=1.0 - f(lg["miss90"]),
                    cov99_vel=1.0 - f(lg["miss99"]),
                    apk_out90=f(lg["apk90"]), apk_above99=f(lg["apk99"]),
                    hmin_out90=f(lg["hmin90"]), hmin_below01=f(lg["hmin01"]),
                    infl_mean=f(lg["infl"]),
                    infl_max=max(lg["infl"]) if lg["infl"] else float("nan"),
                    trig_share=f(lg["trig"]), n=len(lg["miss90"]))


# ------------------------------------------------------------ learner
def fm_loss_rep(net, h, tgt, ok, gen, reps=REPS):
    """model3.fm_loss_h with `reps` noise draws per token."""
    return M.fm_loss_h(net, h.repeat(reps, 1, 1), tgt.repeat(reps, 1, 1),
                       ok.repeat(reps, 1), gen)


class OnlineLearner:
    """Predict-then-learn adaptation of one row (module docstring). Adds
    the LoRA adapters to `net` in place (a fresh net per mission)."""

    def __init__(self, net, live, b, ctx, variant, head=None, tau=TAU,
                 seed=0):
        from learn.meta.safety_head import QS
        self.net, self.D, self.b, self.ctx = net, live, b, int(ctx)
        self.variant, self.head, self.tau = variant, head, float(tau)
        self.qs = QS
        for p in net.parameters():
            p.requires_grad_(False)
        lora = add_lora(net, seed=seed)
        groups = [dict(params=list(net.head.parameters()), lr=LR_HEAD),
                  dict(params=lora, lr=LR_LORA)]
        self.params = list(net.head.parameters()) + lora
        if head is not None:
            groups.append(dict(params=list(head.parameters()), lr=LR_SAFETY))
            self.params += list(head.parameters())
        for p in self.params:
            p.requires_grad_(True)
        self.prior = [p.detach().clone() for p in self.params]
        self.opt = torch.optim.Adam(groups)
        self.gen = torch.Generator().manual_seed(seed + 101)
        self.n = 0
        self.log = dict(fm=[], head=[], pull=[])

    def step(self, k):
        """One gradient step after the plant advanced step k."""
        D, b, L = self.D, self.b, self.ctx
        self.n = k + 1
        if not np.isfinite(D.E0[b, k]).all():
            return
        a = max(0, k + 1 - L)
        ii = torch.tensor([b], device=D.dev)
        aa = torch.tensor([a], device=D.dev)
        net = self.net
        with torch.enable_grad():
            if self.variant == "a":
                tok, tgt, ok = M.window_tokens(D, ii, aa, L, M.obs_sd(
                    1, D.dev, self.gen), self.gen)
            else:
                from learn.meta import model_preview as MP
                tok, tgt, ok, _ = MP.train_tokens_p(D, ii, aa, L, self.gen,
                                                    self.variant)
            h = net.encode(tok)
            loss_fm = fm_loss_rep(net, h, tgt, ok, self.gen)
            loss = loss_fm
            lh = torch.zeros(())
            if self.head is not None:
                from learn.meta.safety_head import pinball
                t = (aa[:, None] + torch.arange(L, device=D.dev)[None]
                     ).clamp(max=D.T - 1)
                y = D.Y[ii[:, None], t]
                okh = ok & torch.isfinite(y).all(-1)
                if okh.any():
                    q = self.head(h.detach(), tgt, raw=False)
                    hd = self.head
                    yn = (torch.nan_to_num(y) - hd.y_mu) / hd.y_sd
                    lh = (pinball(q, yn, self.qs).sum(-1) * okh).sum() \
                        / okh.sum()
                    loss = loss + lh
            pull = sum(((p - p0) ** 2).sum() for p, p0 in
                       zip(self.params, self.prior)) / (2 * self.tau ** 2
                                                        * self.n)
            loss = loss + pull
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
        self.log["fm"].append(float(loss_fm))
        self.log["head"].append(float(lh))
        self.log["pull"].append(float(pull))

    def summary(self):
        f = lambda x: float(np.mean(x[-60:])) if x else float("nan")  # noqa
        dev = sum(float(((p - p0) ** 2).sum()) for p, p0 in
                  zip(self.params, self.prior))
        return dict(n=self.n, fm_last60=f(self.log["fm"]),
                    head_last60=f(self.log["head"]),
                    pull_last60=f(self.log["pull"]), dev_sq=dev)
