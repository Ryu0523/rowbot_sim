#!/usr/bin/env python3
"""
PPO (stable-baselines3) with SG-RL's solver-gradient guidance, and the
training helpers every reproduction shares.

SG-RL (Zarrouki et al., arXiv 2609.01061, Sec. V) stores with every
transition the normalised gradient g_t of a surrogate tracking loss with
respect to the MPC weights (here envs.MPPISens), and uses it in one of four
ways, each tested alone in the paper. Two are implemented:

  "sca"  step-size scaling (their best on Monza, 60% fewer samples than
         PPO): per minibatch, with g_RL the PPO gradient of the actor and
         g~ the gradient of  l = mean_i mu(o_i) . g_i,
             rho   = g~ . g_RL / (|g_RL|^2 + eps)
             alpha = clip(1 + lambda rho, 0, alpha_max)
         and the actor's step is scaled by alpha -- direction unchanged.
         SB3 uses Adam, which normalises gradient scale away, so alpha
         scales the actor's learning rate for that minibatch (the paper's
         Eq. 22 is written for plain SGD).
  "los"  auxiliary loss (their best with SCA on Yas Marina, 70.6% fewer):
             L_PPO + lambda E[ w_t |mu(o_t) - (mu_old,t - eta g_t)|^2 ],
             w_t = clip(-A_t, 0, w_max)
         i.e. where the advantage says the action was bad, pull the mean
         along the solver's descent direction.

lambda is annealed linearly to zero (the paper anneals it; the schedule is
not given). g is in the policy's action box (envs.MPPISens maps it there,
the paper does not say how it handled the box mapping). Coefficients the
paper leaves out: lambda0 = 1, alpha_max = 3, eta = 0.1, w_max = 2.
"""
import functools
import json
import os
from typing import NamedTuple

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.nn import functional as F


# ------------------------------------------------------------------ buffer
class SGSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    sens: th.Tensor
    mu_old: th.Tensor


class SensBuffer(RolloutBuffer):
    """RolloutBuffer that also keeps, per transition, the solver gradient
    and the policy mean at collection time (filled by SensCallback)."""

    def reset(self):
        super().reset()
        shape = (self.buffer_size, self.n_envs, self.action_dim)
        self.sens = np.zeros(shape, np.float32)
        self.mu_old = np.zeros(shape, np.float32)

    def get(self, batch_size=None):
        assert self.full
        n = self.buffer_size * self.n_envs
        idx = np.random.permutation(n)
        if not self.generator_ready:
            for k in ("observations", "actions", "values", "log_probs",
                      "advantages", "returns", "sens", "mu_old"):
                self.__dict__[k] = self.swap_and_flatten(self.__dict__[k])
            self.generator_ready = True
        batch_size = n if batch_size is None else batch_size
        for i in range(0, n, batch_size):
            yield self._get_samples(idx[i:i + batch_size])

    def _get_samples(self, inds, env=None):
        data = (self.observations[inds], self.actions[inds],
                self.values[inds].flatten(), self.log_probs[inds].flatten(),
                self.advantages[inds].flatten(), self.returns[inds].flatten(),
                self.sens[inds], self.mu_old[inds])
        return SGSamples(*map(self.to_torch, data))


class SensCallback(BaseCallback):
    """Copies info['sens'] and the collection-time policy mean into the
    buffer slot the current transition is about to occupy."""

    def _on_step(self):
        buf = self.model.rollout_buffer
        if not isinstance(buf, SensBuffer):
            return True
        pos = buf.pos
        infos = self.locals["infos"]
        buf.sens[pos] = np.stack([i.get("sens", np.zeros(buf.action_dim,
                                                         np.float32))
                                  for i in infos])
        with th.no_grad():
            d = self.model.policy.get_distribution(self.locals["obs_tensor"])
            buf.mu_old[pos] = d.distribution.mean.cpu().numpy()
        return True


class MetricsCallback(BaseCallback):
    """Collects the task metrics of every finished training episode (the
    learning curve) and saves them with the timestep."""

    def __init__(self, path=None, every=20):
        super().__init__()
        self.path, self.every, self.rows, self._n = path, every, [], 0

    def _on_step(self):
        for i in self.locals["infos"]:
            m = i.get("metrics")
            if m is not None:
                self.rows.append(dict(t=int(self.num_timesteps),
                                      score=m["score"], u=m["u_along"],
                                      acc=m["acc_cg_p99"],
                                      lost=bool(m.get("lost", False)),
                                      len=float(m["t_end"])))
                self._n += 1
                if self.path and self._n % self.every == 0:
                    self.save()
        return True

    def save(self):
        if self.path:
            with open(self.path, "w") as f:
                json.dump(self.rows, f)


# --------------------------------------------------------------------- PPO
class SGPPO(PPO):
    def __init__(self, *a, sg_mode=None, sg_lambda=1.0, sg_alpha_max=3.0,
                 sg_eta=0.1, sg_wmax=2.0, **kw):
        self.sg_mode = sg_mode
        self.sg_lambda, self.sg_alpha_max = sg_lambda, sg_alpha_max
        self.sg_eta, self.sg_wmax = sg_eta, sg_wmax
        if sg_mode is not None:
            kw.setdefault("rollout_buffer_class", SensBuffer)
        super().__init__(*a, **kw)
        self.sg_log = []

    def _actor_params(self):
        return (list(self.policy.mlp_extractor.policy_net.parameters())
                + list(self.policy.action_net.parameters()))

    def _setup_model(self):
        super()._setup_model()
        if self.sg_mode == "sca":
            # separate the actor so only its step is scaled
            actor = self._actor_params()
            if hasattr(self.policy, "log_std"):
                actor = actor + [self.policy.log_std]
            ids = {id(p) for p in actor}
            rest = [p for p in self.policy.parameters() if id(p) not in ids]
            self.policy.optimizer = self.policy.optimizer_class(
                [{"params": actor}, {"params": rest}], lr=self.lr_schedule(1),
                **self.policy.optimizer_kwargs)

    def train(self):
        if self.sg_mode is None:
            return super().train()
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        lr = self.policy.optimizer.param_groups[0]["lr"]
        clip_range = self.clip_range(self._current_progress_remaining)
        lam = self.sg_lambda * self._current_progress_remaining
        pg_l, v_l, e_l, alphas, rhos, aux_l = [], [], [], [], [], []
        actor = self._actor_params()
        for _ in range(self.n_epochs):
            for d in self.rollout_buffer.get(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(
                    d.observations, d.actions)
                values = values.flatten()
                adv = d.advantages
                if self.normalize_advantage and len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = th.exp(log_prob - d.old_log_prob)
                policy_loss = -th.min(adv * ratio, adv * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range)).mean()
                value_loss = F.mse_loss(d.returns, values)
                ent = -th.mean(entropy) if entropy is not None \
                    else th.mean(log_prob)
                loss = policy_loss + self.ent_coef * ent \
                    + self.vf_coef * value_loss
                mask = (d.sens.abs().sum(-1) > 0).float()
                if self.sg_mode == "los" and lam > 0:
                    mu = self.policy.get_distribution(
                        d.observations).distribution.mean
                    tgt = d.mu_old - self.sg_eta * d.sens
                    w = th.clamp(-adv, 0.0, self.sg_wmax) * mask
                    aux = (w * ((mu - tgt) ** 2).sum(-1)).mean()
                    loss = loss + lam * aux
                    aux_l.append(aux.item())
                self.policy.optimizer.zero_grad()
                loss.backward()
                alpha = 1.0
                if self.sg_mode == "sca" and lam > 0:
                    g_rl = [p.grad.detach().clone() if p.grad is not None
                            else th.zeros_like(p) for p in actor]
                    mu = self.policy.get_distribution(
                        d.observations).distribution.mean
                    ell = (mu * d.sens).sum(-1).mean()
                    g_t = th.autograd.grad(ell, actor, allow_unused=True)
                    num = sum((a * b).sum() for a, b in zip(g_t, g_rl)
                              if a is not None)
                    den = sum((b * b).sum() for b in g_rl) + 1e-12
                    rho = float(num / den)
                    alpha = float(np.clip(1.0 + lam * rho, 0.0,
                                          self.sg_alpha_max))
                    rhos.append(rho)
                    self.policy.optimizer.param_groups[0]["lr"] = lr * alpha
                alphas.append(alpha)
                th.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                            self.max_grad_norm)
                self.policy.optimizer.step()
                self.policy.optimizer.param_groups[0]["lr"] = lr
                pg_l.append(policy_loss.item())
                v_l.append(value_loss.item())
                e_l.append(ent.item())
            self._n_updates += 1
        ev = explained_variance(self.rollout_buffer.values.flatten(),
                                self.rollout_buffer.returns.flatten())
        row = dict(t=int(self.num_timesteps), lam=lam,
                   alpha=float(np.mean(alphas)),
                   clamp=float(np.mean([a in (0.0, self.sg_alpha_max)
                                        for a in alphas])),
                   rho=float(np.mean(rhos)) if rhos else 0.0,
                   aux=float(np.mean(aux_l)) if aux_l else 0.0,
                   ev=float(ev))
        self.sg_log.append(row)
        self.logger.record("train/policy_gradient_loss", np.mean(pg_l))
        self.logger.record("train/value_loss", np.mean(v_l))
        self.logger.record("train/explained_variance", ev)
        self.logger.record("train/sg_alpha", row["alpha"])


# ----------------------------------------------------------------- helpers
def make_env(name, kw, seed):
    from learn.repro import envs
    return getattr(envs, name)(seed=seed, **kw)


def vec_env(name, kw, n_envs, seed=0):
    """SubprocVecEnv of learn.repro.envs.<name>(**kw), one process per env
    (each imports torch through stable-baselines3: ~0.3 GB apiece)."""
    fns = [functools.partial(make_env, name, kw, seed * 100 + i)
           for i in range(n_envs)]
    return SubprocVecEnv(fns, start_method="spawn")


class NumpyMLP:
    """The deterministic action of an SB3 MlpPolicy (flatten -> tanh MLP ->
    linear), in numpy, so an evaluation pool needs no torch."""

    def __init__(self, policy):
        self.W, self.b = [], []
        for m in policy.mlp_extractor.policy_net:
            if isinstance(m, th.nn.Linear):
                self.W.append(m.weight.detach().cpu().numpy().T)
                self.b.append(m.bias.detach().cpu().numpy())
        a = policy.action_net
        self.Wa = a.weight.detach().cpu().numpy().T
        self.ba = a.bias.detach().cpu().numpy()
        self.tanh = any(isinstance(m, th.nn.Tanh)
                        for m in policy.mlp_extractor.policy_net)

    def __call__(self, obs, *_):
        h = np.asarray(obs, float)
        for W, b in zip(self.W, self.b):
            h = h @ W + b
            h = np.tanh(h) if self.tanh else np.maximum(h, 0.0)
        return np.clip(h @ self.Wa + self.ba, -1.0, 1.0)


def out_dir(name):
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(here, "..", "..", "studies", "_cache", "repro", name)
    os.makedirs(d, exist_ok=True)
    return os.path.abspath(d)
