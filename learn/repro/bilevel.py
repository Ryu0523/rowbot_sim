#!/usr/bin/env python3
"""
Bi-level RL for sim-to-real (Anand, Sawant, Parmas, Hoffmann, Reinhardt,
Gros; arXiv 2510.17709 v2, RLJ 2026), on this task.

The method, as published:
  inner  PPO trains the policy phi in the SIMULATOR, whose dynamics and
         reward are parameterised by theta; warm-started across outer
         iterations, never converged (their quadrotor: 20 PPO updates per
         outer step).
  outer  theta moves to maximise the policy's return on the REAL system:
             dJ_real/dtheta = (dphi*/dtheta)^T g,  dphi*/dtheta = -A^-1 B
         g = E_real[grad_phi log pi * A_real] is the only thing estimated
         from real data (a PPO-style policy gradient with a real critic);
         A = d(sim policy gradient)/dphi, B = d(sim policy gradient)/dtheta
         come from simulation. Their quadrotor used "a first-order
         approximation, neglecting the Hessian term".

What is done here, and why:
  * First-order: A ~ -c I (the sim objective is at or near a maximum in
    phi, so its Hessian is negative definite), giving the ascent direction
        dJ_real/dtheta ~ B^T g = d/dtheta [ g . simPG(phi, theta) ]
    -- move the simulator so that ITS policy gradient points where the
    real one does. The paper's B needs grad_theta log P_theta, a transition
    DENSITY, which a deterministic simulator does not have (the paper does
    not say how it handled its deterministic quadrotor). B^T g is taken
    instead by central finite differences in theta with common random
    numbers (same waves, same headings, same action noise at +delta and
    -delta), 2 n_theta simulated policy-gradient estimates per outer step.
  * theta = 10 of the source world's coefficients (lofi.PRIOR box: drag,
    thruster lag, jet side-force and yaw gains, yaw time constant, sway
    damping, heave/pitch wave gains and damping) + the prices of the
    simulator reward's three parts (speed, impacts, cross-track), each
    x1/4 .. x4. The paper moves dynamics and reward parameters too.
  * g: 4 target episodes per outer step (2 seeds x 2 headings, 120 s),
    stochastic policy; a real critic, warm-started from the simulator's and
    refitted on all target data so far (Monte-Carlo returns); GAE(0.95).
  * 8 outer steps -> 32 target episodes, the budget of the few-shot
    methods in studies/sim2real_jet.py. Step 0.15 on the unit box along
    the normalised direction (the toy examples' 0.1 on raw gradients).
  * Policy: direct control (thrust, nozzle), MLP 128-128, as the paper's
    PPO policies; pre-trained 1.5M steps in the nominal source world (the
    paper's "in-sim policy", also reported on its own).
"""
import copy
import json
import os

import numpy as np
import torch as th

from learn.repro import task
from learn.repro.envs import DirectAgent
from learn.repro.ppo_sg import MetricsCallback, out_dir, vec_env
from learn.repro.task import Mission, Runner, ctx

DYN_KEYS = ("k_drag", "tau_thrust", "k_jet_side", "k_nomoto_f", "tau_r",
            "k_lin_sway", "k_wave_heave", "k_wave_pitch", "z_heave",
            "z_pitch")
N_REW = 3
REW_SPAN = 4.0                 # reward prices move within x1/4 .. x4
C_JUMP = 0.02
ALIVE = 1.0
PPO_KW = dict(learning_rate=3e-4, n_steps=1024, batch_size=1536, n_epochs=10,
              gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0)
NET = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))


def theta0():
    c = ctx()
    return np.r_[task.to_unit(c["p"], {}, DYN_KEYS), np.zeros(N_REW)]


def decode(th_):
    """theta -> (source overrides, reward prices)."""
    from sim import lofi
    c = ctx()
    th_ = np.clip(np.asarray(th_, float), -1.0, 1.0)
    over = lofi.from_unit(c["p"], th_[:len(DYN_KEYS)], DYN_KEYS)
    return over, REW_SPAN ** th_[len(DYN_KEYS):]


# ------------------------------------------------------ numpy Gaussian
class NumpyGauss:
    """An SB3 MlpPolicy's Gaussian in numpy (raw mean, no clipping, and
    the std), so rollout workers need no torch."""

    def __init__(self, policy):
        self.W, self.b = [], []
        for m in policy.mlp_extractor.policy_net:
            if isinstance(m, th.nn.Linear):
                self.W.append(m.weight.detach().numpy().T.copy())
                self.b.append(m.bias.detach().numpy().copy())
        self.Wa = policy.action_net.weight.detach().numpy().T.copy()
        self.ba = policy.action_net.bias.detach().numpy().copy()
        self.std = np.exp(policy.log_std.detach().numpy().copy())

    def mean(self, o):
        h = np.asarray(o, float)
        for W, b in zip(self.W, self.b):
            h = np.tanh(h @ W + b)
        return h @ self.Wa + self.ba


def rollout_job(job):
    """One episode with a stochastic policy -> the trajectory.
    job: pol (NumpyGauss), world, seed, head, noise (seed), t_end,
         theta (source only) or None, real (bool: task reward)."""
    ctx()
    pp, w = (decode(job["theta"]) if job.get("theta") is not None
             else (None, None))
    # the source's lateral push (task.LateralOU, as the policy was trained
    # with), on its own noise stream so +-delta runs meet the same push
    dist = (task.LateralOU(np.random.default_rng(job["noise"] + 99991))
            if job["world"] == "low" else None)
    m = Mission(job["world"], job["seed"], job["head"], plant_params=pp,
                t_end=job["t_end"], disturb=dist)
    ag = DirectAgent(m)
    pol = job["pol"]
    rng = np.random.default_rng(job["noise"])
    O, A, R = [], [], []
    while not m.done():
        o = ag.obs()
        a = pol.mean(o) + pol.std * rng.normal(size=2)
        r, jump = ag.act(a, None if job.get("real") else w)
        if not job.get("real"):
            # the simulator's reward exactly as the inner PPO sees it
            # (envs.DirectEnv: jump penalty and the per-step alive bonus)
            r = r - C_JUMP * jump + ALIVE
        O.append(o)
        A.append(a)
        R.append(r if m.finite else -task.BAD_SCORE)
    return dict(obs=np.array(O, np.float32), act=np.array(A, np.float32),
                rew=np.array(R, np.float32), last=ag.obs(),
                finite=m.finite, metrics=m.metrics())


# ----------------------------------------------------- policy gradients
def _actor_params(policy):
    return (list(policy.mlp_extractor.policy_net.parameters())
            + list(policy.action_net.parameters()) + [policy.log_std])


def _gae(rew, v, v_last, gamma, lam, finite):
    adv = np.zeros_like(rew)
    nxt, g = (v_last if finite else 0.0), 0.0
    for t in range(len(rew) - 1, -1, -1):
        delta = rew[t] + gamma * nxt - v[t]
        g = delta + gamma * lam * g
        adv[t] = g
        nxt = v[t]
    return adv


def policy_gradient(policy, trajs, value_fn, gamma=0.99, lam=0.95):
    """grad_phi mean_t[log pi(a_t|s_t) A_t] over the trajectories (raw GAE
    advantages, so estimates at different theta are comparable)."""
    O, A, ADV = [], [], []
    for tr in trajs:
        with th.no_grad():
            v = value_fn(th.as_tensor(tr["obs"])).numpy().ravel()
            vl = float(value_fn(th.as_tensor(tr["last"][None])).item())
        ADV.append(_gae(tr["rew"], v, vl, gamma, lam, tr["finite"]))
        O.append(tr["obs"])
        A.append(tr["act"])
    O, A = th.as_tensor(np.concatenate(O)), th.as_tensor(np.concatenate(A))
    adv = th.as_tensor(np.concatenate(ADV))
    _, logp, _ = policy.evaluate_actions(O, A)
    params = _actor_params(policy)
    g = th.autograd.grad((logp * adv).mean(), params)
    return th.cat([x.flatten() for x in g]).numpy()


class RealCritic:
    """A value function for the target: the simulator's critic, refitted
    on every target episode so far (Monte-Carlo returns)."""

    def __init__(self, policy, gamma=0.99):
        self.net = th.nn.Sequential(copy.deepcopy(
            policy.mlp_extractor.value_net), copy.deepcopy(policy.value_net))
        self.gamma, self.O, self.G = gamma, [], []
        self.opt = th.optim.Adam(self.net.parameters(), lr=1e-3)

    def __call__(self, o):
        return self.net(o)

    def add(self, trajs):
        for tr in trajs:
            g, G = 0.0, np.zeros_like(tr["rew"])
            for t in range(len(tr["rew"]) - 1, -1, -1):
                g = tr["rew"][t] + self.gamma * g
                G[t] = g
            self.O.append(tr["obs"])
            self.G.append(G)
        O = th.as_tensor(np.concatenate(self.O))
        G = th.as_tensor(np.concatenate(self.G))[:, None]
        for _ in range(200):
            i = th.randint(0, len(O), (512,))
            loss = ((self.net(O[i]) - G[i]) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


# ------------------------------------------------------------ the method
def run(pretrain=1_500_000, outer=8, inner_updates=20, n_envs=6, workers=6,
        delta=0.2, step=0.15, fd_seeds=3, seed=0):
    from stable_baselines3 import PPO
    d = out_dir("bilevel")
    state_path = os.path.join(d, "state.json")
    S = json.load(open(state_path)) if os.path.exists(state_path) else {}
    th_ = np.array(S.get("theta", theta0()))
    env = vec_env("DirectEnv", dict(world="low", randomize=False),
                  n_envs, seed)
    pre = os.path.join(d, "insim.zip")
    if os.path.exists(pre):
        model = PPO.load(pre, env=env, device="cpu")
    else:
        print("  bi-level: pre-training in the nominal source world",
              flush=True)
        model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                    policy_kwargs=NET, **PPO_KW)
        cb = MetricsCallback(os.path.join(d, "curve_insim.json"))
        model.learn(pretrain, callback=cb)
        cb.save()
        model.save(pre)
    pool = Runner(workers, fn=rollout_job)
    real_critic = RealCritic(model.policy)
    hist = S.get("hist", [])
    k0 = len(hist)
    if k0:
        # resume: the moved simulator and the policy trained in it
        model = PPO.load(os.path.join(d, "policy.zip"), env=env, device="cpu")
        real_critic = RealCritic(model.policy)
        over, w = decode(th_)
        env.env_method("set_params", over)
        env.env_method("set_reward_w", w.tolist())
    try:
        for k in range(k0, outer):
            pol = NumpyGauss(model.policy)
            # 1. target rollouts: the only real data
            real = pool([dict(pol=pol, world="high", seed=9000 + 2 * k + s,
                              head=h, noise=7000 + 4 * k + 2 * s + h,
                              t_end=task.T_EVAL, real=True)
                         for s in range(2) for h in range(2)])
            real_critic.add(real)
            g = policy_gradient(model.policy, real, real_critic)
            # 2. B^T g by central differences in theta, common random
            #    numbers across +-delta
            base = [dict(pol=pol, world="low", seed=500 + 3 * k + s, head=h,
                         noise=100 * k + 2 * s + h, t_end=task.T_TRAIN)
                    for s in range(fd_seeds) for h in range(2)]
            jobs = []
            for j in range(len(th_)):
                for sg in (1.0, -1.0):
                    tj = th_.copy()
                    tj[j] = np.clip(tj[j] + sg * delta, -1.0, 1.0)
                    jobs += [dict(b, theta=tj.tolist()) for b in base]
            res = pool(jobs)
            vf = model.policy.predict_values
            nb = len(base)
            s_dir = np.zeros(len(th_))
            for j in range(len(th_)):
                lo_, hi_ = (res[(2 * j + 1) * nb:(2 * j + 2) * nb],
                            res[2 * j * nb:(2 * j + 1) * nb])
                dp = (policy_gradient(model.policy, hi_, vf)
                      - policy_gradient(model.policy, lo_, vf))
                span = (min(th_[j] + delta, 1.0) - max(th_[j] - delta, -1.0))
                s_dir[j] = float(g @ dp) / max(span, 1e-6)
            nrm = np.linalg.norm(s_dir)
            th_ = np.clip(th_ + step * s_dir / (nrm + 1e-12), -1.0, 1.0)
            over, w = decode(th_)
            # 3. inner loop: PPO in the moved simulator, warm start
            env.env_method("set_params", over)
            env.env_method("set_reward_w", w.tolist())
            model.learn(inner_updates * n_envs * PPO_KW["n_steps"],
                        reset_num_timesteps=False)
            m_real = [r["metrics"] for r in real]
            hist.append(dict(k=k, theta=th_.tolist(), dir=s_dir.tolist(),
                             reward_w=w.tolist(),
                             over={a: float(b) for a, b in over.items()},
                             real_score=float(np.mean(
                                 [m["score"] for m in m_real])),
                             real_u=float(np.mean([m["u_along"]
                                                   for m in m_real]))))
            print(f"    bi-level outer {k}: target score (stochastic) "
                  f"{hist[-1]['real_score']:.3f}  reward prices "
                  f"{np.round(w, 2).tolist()}  "
                  f"[{task.avail_gb():.1f} GB free]", flush=True)
            model.save(os.path.join(d, "policy.zip"))
            S.update(theta=th_.tolist(), hist=hist,
                     target_episodes=4 * (k + 1))
            json.dump(S, open(state_path, "w"))
    finally:
        pool.close()
        env.close()
    return model


def load():
    from stable_baselines3 import PPO
    d = out_dir("bilevel")
    out = {}
    for tag, f in (("insim", "insim.zip"), ("bilevel", "policy.zip")):
        p = os.path.join(d, f)
        if os.path.exists(p):
            out[tag] = PPO.load(p, device="cpu")
    return out
