#!/usr/bin/env python3
"""
RMA (Kumar, Fu, Pathak, Malik, RSS 2021) and Jiang, Bi, D'Andrea,
Ramachandran (arXiv 2607.02037, 2026): a direct-control policy that is told
the vessel's parameters in training and learns to estimate them from its
own recent history at run time.

Both methods, as published:
  phase 1  PPO on a simulator with randomised parameters e. An encoder mu
           maps e to a latent z; the policy acts on [observation, z]. Encoder
           and policy are trained together, end to end.
  phase 2  an adaptation module estimates z from the recent history of
           observations and actions. It is trained by regression onto the
           frozen encoder's z, on rollouts of the frozen policy driven by the
           module's OWN estimates (DAgger-like, so it learns on the states
           its errors lead to). No action imitation, no RL fine-tuning.
  deploy   policy + adaptation module, no real data at all.

Where they differ, and what is kept:
                     RMA                          Jiang et al.
  latent z           8-d, bottleneck              d_z = d_e (no bottleneck)
  encoder mu         MLP 256-128                  MLP 128-128
  policy / critic    MLP 128-128-128              MLP 128-128-64
  history module     1-D CNN over the last 50     GRU(128) + MLP 128, the
                     steps                        history lives in its state
  PPO                SB3 defaults                 gamma 0.96, clip 0.1, lr
                                                  3e-4 -> 3e-5, entropy
                                                  0.005, max grad 0.7,
                                                  target KL 0.02, 8 mini-
                                                  batches, 10 epochs

What had to change for this task:
  * e is the source vessel's coefficients on the lofi.PRIOR box (14 numbers:
    heave/pitch natural frequency, damping and wave gain, drag, yaw time
    constant and gain, sway damping, thruster lag, jet side-force gain,
    nozzle rate; task.to_unit). RMA's e is mass, friction, motor strength;
    Jiang's is the per-unit-command form of a fully actuated 3-DoF model.
  * The action is thrust and nozzle angle (a single jet has no side force
    of its own); Jiang's is body-frame Fx, Fy, Mz.
  * The task is the operator cost of task.py (speed made good, impacts,
    cross-track) rather than walking or trajectory tracking; the reward is
    minus that cost plus a small action-jump penalty (envs.DirectEnv).
  * Jiang's curriculum on the parameter ranges is not used: the PRIOR is
    only +-30% wide where theirs spans four decades.
  * Budgets are what the laptop can do: 2M PPO steps (Jiang: 6e7; RMA
    ~1e9) and ~120k adaptation steps.
"""
import json
import os

import numpy as np
import torch as th
from torch import nn

from learn.repro import task
from learn.repro.envs import DirectAgent
from learn.repro.ppo_sg import MetricsCallback, out_dir, vec_env

VARIANTS = {
    "rma": dict(z_dim=8, enc=(256, 128), pi=[128, 128, 128],
                vf=[128, 128, 128], act=nn.ReLU,
                ppo=dict(learning_rate=3e-4, n_steps=1024, batch_size=1536,
                         n_epochs=10, gamma=0.99, gae_lambda=0.95,
                         clip_range=0.2, ent_coef=0.0),
                adapter="cnn", hist=50),
    "jiang": dict(z_dim=None, enc=(128, 128), pi=[128, 128, 64],
                  vf=[128, 128, 64], act=nn.Tanh,
                  ppo=dict(learning_rate=None, n_steps=1024, batch_size=768,
                           n_epochs=10, gamma=0.96, gae_lambda=0.95,
                           clip_range=0.1, ent_coef=0.005, vf_coef=0.5,
                           max_grad_norm=0.7, target_kl=0.02),
                  adapter="gru", hist=None),
}


def n_priv():
    return len(task.ctx()["names"])


# ------------------------------------------------------------ networks
def mlp(n_in, sizes, n_out, act):
    layers, n = [], n_in
    for s in sizes:
        layers += [nn.Linear(n, s), act()]
        n = s
    layers.append(nn.Linear(n, n_out))
    return nn.Sequential(*layers)


try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class PrivExtractor(BaseFeaturesExtractor):
        """obs = [o | e] -> [o | mu(e)]: the environment-factor encoder."""

        def __init__(self, observation_space, n_e, z_dim, enc, act):
            n = observation_space.shape[0]
            super().__init__(observation_space, features_dim=n - n_e + z_dim)
            self.n_o = n - n_e
            self.mu = mlp(n_e, enc, z_dim, act)

        def forward(self, obs):
            return th.cat([obs[:, :self.n_o], self.mu(obs[:, self.n_o:])], 1)
except ImportError:                                  # pragma: no cover
    PrivExtractor = None


class CNNAdapter(nn.Module):
    """RMA's adaptation module: per-step embedding, three 1-D convolutions
    over the last `hist` steps, linear to z."""

    def __init__(self, n_in, z_dim, hist=50):
        super().__init__()
        self.hist = hist
        self.embed = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU())
        self.conv = nn.Sequential(
            nn.Conv1d(32, 32, 8, stride=4), nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1), nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1), nn.ReLU())
        with th.no_grad():
            n = self.conv(th.zeros(1, 32, hist)).numel()
        self.head = nn.Linear(n, z_dim)

    def forward(self, h):                       # h: (B, hist, n_in)
        x = self.embed(h).transpose(1, 2)
        return self.head(self.conv(x).flatten(1))


class GRUAdapter(nn.Module):
    """Jiang et al.'s adapter: GRU(128), then an MLP to z."""

    def __init__(self, n_in, z_dim, hidden=128):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(),
                                  nn.Linear(128, z_dim))

    def forward(self, seq, h0=None):            # seq: (B, T, n_in)
        y, h = self.gru(seq, h0)
        return self.head(y), h


# -------------------------------------------------------------- phase 1
def train_teacher(variant, steps=2_000_000, n_envs=6, seed=0):
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import get_linear_fn
    v = VARIANTS[variant]
    d = out_dir(variant)
    n_e = n_priv()
    z_dim = v["z_dim"] or n_e
    ppo = dict(v["ppo"])
    if ppo["learning_rate"] is None:            # Jiang: 3e-4 -> 3e-5
        ppo["learning_rate"] = get_linear_fn(3e-4, 3e-5, 1.0)
    env = vec_env("DirectEnv", dict(world="low", randomize=True, priv=True),
                  n_envs, seed)
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                policy_kwargs=dict(
                    features_extractor_class=PrivExtractor,
                    features_extractor_kwargs=dict(n_e=n_e, z_dim=z_dim,
                                                   enc=v["enc"],
                                                   act=v["act"]),
                    net_arch=dict(pi=v["pi"], vf=v["vf"]),
                    activation_fn=v["act"]), **ppo)
    cb = MetricsCallback(os.path.join(d, "curve_teacher.json"))
    model.learn(steps, callback=cb)
    cb.save()
    model.save(os.path.join(d, "teacher.zip"))
    env.close()
    return model


def _actor(model):
    """(z(e), act([o, z]) -> mean action) of a trained teacher."""
    pol = model.policy
    ext = pol.features_extractor

    def z_of(e):
        with th.no_grad():
            return ext.mu(th.as_tensor(e, dtype=th.float32))

    def act(o, z):
        with th.no_grad():
            f = th.cat([th.as_tensor(o, dtype=th.float32), z], -1)
            return pol.action_net(pol.mlp_extractor.forward_actor(f))
    return z_of, act, ext.n_o


# -------------------------------------------------------------- phase 2
def train_adapter(variant, model, iters=20, steps_per_iter=1024, n_envs=6,
                  seed=1, epochs=4, noise=0.1):
    """DAgger-like latent regression (both papers' phase 2)."""
    v = VARIANTS[variant]
    d = out_dir(variant)
    z_of, act, n_o = _actor(model)
    n_e = n_priv()
    z_dim = v["z_dim"] or n_e
    if v["adapter"] == "cnn":
        net = CNNAdapter(n_o, z_dim, v["hist"])
        H = v["hist"]
    else:
        net = GRUAdapter(n_o, z_dim)
        H = None
    opt = th.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = th.optim.lr_scheduler.LambdaLR(
        opt, lambda i: 1.0 - 0.9 * min(i / max(iters * epochs, 1), 1.0))
    env = vec_env("DirectEnv", dict(world="low", randomize=True, priv=True,
                                    lost_penalty=0.0), n_envs, seed)
    rng = np.random.default_rng(seed)
    obs = env.reset()
    log = []
    hist = np.zeros((n_envs, H or 1, n_o), np.float32)
    h_gru = None
    seqs = [[] for _ in range(n_envs)]          # current episode, per env
    data = []                                   # finished (O, Z) sequences
    for it in range(iters):
        net.eval()
        err = []
        for _ in range(steps_per_iter):
            o, e = obs[:, :n_o], obs[:, n_o:]
            z_true = z_of(e)
            with th.no_grad():
                if H:
                    hist = np.roll(hist, -1, axis=1)
                    hist[:, -1] = o
                    z_hat = net(th.as_tensor(hist))
                else:
                    y, h_gru = net.gru(th.as_tensor(o[:, None, :]), h_gru)
                    z_hat = net.head(y[:, -1])
            err.append(float(((z_hat - z_true) ** 2).mean()))
            a = act(o, z_hat).numpy()
            a = np.clip(a + noise * rng.normal(size=a.shape), -1, 1)
            for i in range(n_envs):
                seqs[i].append((o[i].copy(), z_true[i].numpy()))
            obs, _, dones, _ = env.step(a)
            for i in np.flatnonzero(dones):
                data.append(seqs[i])
                seqs[i] = []
                hist[i] = 0.0
                if h_gru is not None:
                    h_gru[:, i] = 0.0
        # regression on everything gathered so far (DAgger aggregates)
        net.train()
        loss_it = []
        for _ in range(epochs):
            # the GRU learns from episode starts: finished episodes only
            pool_ = data + ([s for s in seqs if len(s) > 8] if H else [])
            for O, Z in _batches(pool_, H, rng):
                if H:
                    pred = net(O)
                    loss = ((pred - Z) ** 2).mean()
                else:
                    pred, _ = net(O)
                    loss = ((pred - Z) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                loss_it.append(loss.item())
            sched.step()
        log.append(dict(it=it, rollout_z_mse=float(np.mean(err)),
                        train_mse=float(np.mean(loss_it)),
                        n_seq=len(data)))
        print(f"    {variant} adapter it {it:2d}: rollout z-mse "
              f"{np.mean(err):.4f}  train {np.mean(loss_it):.4f}  "
              f"[{task.avail_gb():.1f} GB free]", flush=True)
    env.close()
    th.save(net.state_dict(), os.path.join(d, "adapter.pt"))
    with open(os.path.join(d, "adapter_log.json"), "w") as f:
        json.dump(log, f)
    return net


def _batches(seqs, H, rng, batch=256, seq_len=64):
    """CNN: windows of H steps ending at random times (zero-padded at the
    start of an episode). GRU: random contiguous chunks of seq_len steps
    from episode starts, so the recurrent state is learnt from zero."""
    if not seqs:
        return
    if H:
        pool = [(k, t) for k, s in enumerate(seqs) for t in range(len(s))]
        idx = rng.permutation(len(pool))[:batch * 16]
        for i in range(0, len(idx), batch):
            O, Z = [], []
            for j in idx[i:i + batch]:
                k, t = pool[j]
                s = seqs[k]
                w = np.zeros((H, len(s[0][0])), np.float32)
                lo = max(0, t - H + 1)
                arr = np.array([x[0] for x in s[lo:t + 1]], np.float32)
                w[H - len(arr):] = arr
                O.append(w)
                Z.append(s[t][1])
            yield th.as_tensor(np.array(O)), th.as_tensor(np.array(Z))
    else:
        order = rng.permutation(len(seqs))
        for i in range(0, len(order), 32):
            O, Z = [], []
            for k in order[i:i + 32]:
                s = seqs[k][:seq_len * 4]
                O.append(np.array([x[0] for x in s], np.float32))
                Z.append(np.array([x[1] for x in s], np.float32))
            T = min(len(x) for x in O)
            yield (th.as_tensor(np.array([x[:T] for x in O])),
                   th.as_tensor(np.array([x[:T] for x in Z])))


# --------------------------------------------------------------- deploy
class LatentPolicy:
    """Frozen policy + adaptation module, for envs.rollout('direct', ...).
    `z_fixed` replaces the estimate by a constant (ablation: the policy with
    the nominal vessel's z, i.e. no adaptation)."""

    def __init__(self, variant, model, net, z_fixed=None):
        self.v = VARIANTS[variant]
        self.z_of, self.act, self.n_o = _actor(model)
        self.net = net.eval() if net is not None else None
        self.z_fixed = z_fixed
        self.reset()

    def reset(self):
        H = self.v["hist"]
        self.hist = np.zeros((1, H or 1, self.n_o), np.float32)
        self.h = None

    def __call__(self, obs, agent=None):
        o = np.asarray(obs, np.float32)[None, :self.n_o]
        if self.z_fixed is not None:
            z = th.as_tensor(self.z_fixed, dtype=th.float32)[None]
        else:
            with th.no_grad():
                if self.v["hist"]:
                    self.hist = np.roll(self.hist, -1, axis=1)
                    self.hist[:, -1] = o
                    z = self.net(th.as_tensor(self.hist))
                else:
                    y, self.h = self.net.gru(th.as_tensor(o[:, None]), self.h)
                    z = self.net.head(y[:, -1])
        return self.act(o, z).numpy()[0]


def load(variant):
    from stable_baselines3 import PPO
    d = out_dir(variant)
    v = VARIANTS[variant]
    model = PPO.load(os.path.join(d, "teacher.zip"), device="cpu")
    n_e = n_priv()
    z_dim = v["z_dim"] or n_e
    n_o = DirectAgent.n_obs()
    net = CNNAdapter(n_o, z_dim, v["hist"]) if v["adapter"] == "cnn" \
        else GRUAdapter(n_o, z_dim)
    p = os.path.join(d, "adapter.pt")
    if os.path.exists(p):
        net.load_state_dict(th.load(p))
    else:
        net = None
    return model, net
