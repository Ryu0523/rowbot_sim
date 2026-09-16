#!/usr/bin/env python3
"""
Clone the MPC and evaluate the clone -- before any fine-tuning touches it.

The combined behaviour-cloning-then-PPO run produced a policy that spins:
heading RMS of 385 and 795 degrees on two held-out seas. That single number
cannot say which half failed, and the two halves fail for completely different
reasons and want completely different fixes:

  the clone never worked      distribution shift. Cloning matches the expert on
                              the expert's OWN states; small errors compound,
                              the vessel drifts somewhere the MPC never went,
                              and the clone has never seen it. The fix is
                              DAgger -- ask the expert what it would do on the
                              CLONE's states, not its own.

  fine-tuning destroyed it    behaviour cloning trains the action MEAN and
                              leaves log_std at its initialisation, so the
                              policy still samples with std = 1.0 on actions
                              bounded in [-1, 1]. The first PPO rollouts are
                              then near-random regardless of how good the mean
                              is, and ent_coef actively resists shrinking it.
                              The fix is two lines: set log_std small after
                              cloning, and drop the entropy bonus when starting
                              from a good policy.

So this script stops after cloning and measures the deterministic clone against
the MPC it copied. Demonstrations are cached to disk, because collecting them
costs four minutes and every iteration after the first should not pay it again.

Run: python -m learn.bc_only [--demos 30000] [--epochs 60] [--hull NAME]
"""
import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

CACHE = "bc_demos.npz"


def cache_path(hull="wigley10"):
    """One demonstration cache per vessel -- a cache recorded on one hull is
    no use on another, and the fingerprint would only say so after the fact."""
    return CACHE if hull == "wigley10" else f"bc_demos_{hull}.npz"


def plant_fingerprint(hull="wigley10"):
    """Identify the plant these demonstrations were recorded against -- by
    what it DOES, not by what one of its input files contains.

    The first version hashed db.M and db.C from disk. It caught the inertia
    correction (DEFECTS A30) and would have missed everything after it: wind,
    the full Coriolis matrix, the rudder inflow angle, K_T(J), the symmetry
    fix and the re-identified hull derivatives all changed the plant's
    behaviour without touching that file, and a stale cache would have been
    reused in silence. Extending a list of watched inputs every time the
    physics moves is exactly the maintenance that gets forgotten.

    So the plant is run instead: a fixed sea, fixed commands that exercise
    thrust AND rudder, a few seconds, and the resulting state is the
    fingerprint. Any change to the physics moves it; comment edits do not.
    Costs about a second.
    """
    from sim import config
    from sim.wavefield import SeaState
    h, db = config.load(hull)
    sea = SeaState(3.25, 9.7, n_freq=8, n_dir=3, seed=12345)
    p = config.plant_for(db, sea, h)
    # the USV's commands, scaled to this vessel: 1.0 and 1.0 for the USV
    k, rs = p.prop.t_max / 12000.0, float(np.sqrt(h.L / 10.0))
    s = p.initial_state(4.0 * rs)
    for i in range(60):
        s = p.step(s, i * p.dt, (6000.0 + 2000.0 * np.sin(0.3 * i)) * k,
                   np.radians(12.0) * np.sin(0.2 * i), p.dt)
    return float(np.dot(s[:12], np.arange(1, 13)))


def get_demos(n, preview, refresh=False, hull="wigley10"):
    from learn.ppo_bc import collect_demos
    CACHE = cache_path(hull)
    fp = plant_fingerprint(hull)
    if os.path.exists(CACHE) and not refresh:
        d = np.load(CACHE)
        old = float(d["fingerprint"]) if "fingerprint" in d.files else None
        # rtol loose enough for BLAS-level differences between machines, far
        # tighter than any physics change moves a 3 s trajectory
        stale = old is None or not np.isclose(old, fp, rtol=1e-6)
        if stale:
            was = "unstamped" if old is None else f"{old:.6g}"
            print(f"  {CACHE} was recorded against a DIFFERENT plant "
                  f"({was} vs {fp:.6g}) -- recollecting")
        elif len(d["X"]) >= n:
            print(f"  using cached demonstrations {d['X'].shape}")
            return d["X"][:n], d["Y"][:n]
    print(f"  collecting {n:,} MPC demonstration steps ...")
    X, Y = collect_demos(n, preview, hull=hull)
    np.savez_compressed(CACHE, X=X, Y=Y, fingerprint=fp)
    print(f"  cached -> {CACHE}")
    return X, Y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--preview", type=float, default=None,
                    help="s; default 8 s for the USV, Froude-scaled")
    ap.add_argument("--std", type=float, default=0.15)
    ap.add_argument("--out", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--hull", default="wigley10",
                    help="a vessel from hydro/hulls.py")
    a = ap.parse_args()
    a.out = a.out or ("bc_clone" if a.hull == "wigley10"
                      else f"bc_clone_{a.hull}")

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import (DummyVecEnv, VecNormalize,
                                                  VecFrameStack)
    from learn.ppo_train import TRAIN_SEEDS, N_STACK, make_env, _time_scale
    torch.set_num_threads(1)

    X, Y = get_demos(a.demos, a.preview, a.refresh, hull=a.hull)
    mean, std = X.mean(0), X.std(0) + 1e-8
    Xn = np.clip((X - mean) / std, -10, 10).astype(np.float32)

    venv = DummyVecEnv([make_env(TRAIN_SEEDS, a.preview, rank=0,
                                 hull=a.hull)])
    venv = VecFrameStack(venv, n_stack=N_STACK)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, clip_reward=10.0, gamma=0.995)
    venv.obs_rms.mean = mean.astype(np.float64)
    venv.obs_rms.var = (std ** 2).astype(np.float64)
    venv.obs_rms.count = float(len(X))

    model = PPO("MlpPolicy", venv, seed=0, verbose=0,
                policy_kwargs=dict(net_arch=[128, 128]))
    dev = model.policy.device
    Xt = torch.as_tensor(Xn, device=dev)
    Yt = torch.as_tensor(Y, device=dev)
    opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    n, bs = len(Xt), 256
    print(f"\n  cloning {a.epochs} epochs on {n:,} pairs ...")
    for ep in range(a.epochs):
        idx = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            d = model.policy.get_distribution(Xt[j])
            loss = torch.nn.functional.mse_loss(d.distribution.mean, Yt[j])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(j)
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"    epoch {ep:>3}  action MSE {tot/n:.5f}")

    # Cloning trains the mean and leaves log_std where it started, so the
    # policy would still sample with std ~ 1 on a [-1,1] action. Pin it small.
    with torch.no_grad():
        model.policy.log_std.fill_(float(np.log(a.std)))
    print(f"  log_std pinned to std = {a.std}")
    model.save(a.out)
    venv.save(a.out + "_vecnorm.pkl")

    # ---- does the clone actually fly? -------------------------------------
    import studies.rl_vs_mpc as R
    from sim import config
    from control.reduced import ReducedModel
    from control.mpc import MPPIController, PreviewProvider
    red = ReducedModel.identify(config.calm_plant(a.hull)[0])

    def mpc_pol(u_ref=None):
        u_ref = red.p["u_design"] if u_ref is None else u_ref
        st = {}
        def setup(env):
            st["c"] = MPPIController(red, PreviewProvider(env.sea,
                                                          env.t_preview,
                                                          0.0, 0),
                                     dt_ctrl=env.dt_ctrl, u_ref=u_ref, seed=0,
                                     n_samples=192, use_rudder=True)
            st["t"] = 0.0
        def f(o, e):
            c = st["c"]; cmd = c(red.from_plant_state(e._s), st["t"])
            st["t"] += e.dt_ctrl
            thr, rud = c.to_actuator(cmd)
            return np.array([2 * (thr / e._plant.prop.t_max) - 1,
                             rud / e._plant.rudder.max])
        return f, setup

    SEEDS = list(range(100, 106))
    cf, cs = R.ppo_policy(a.out)
    mf, ms = mpc_pol()
    print(f"\n  deterministic clone vs the MPC it copied, {len(SEEDS)} "
          f"HELD-OUT seas\n")
    print(f"  {'controller':<14}{'u':>6}{'acc_p99':>9}{'rvm':>7}{'slamImp':>9}"
          f"{'cross':>8}{'hdg':>8}")
    out = {}
    for nm, pol, st in (("MPC", mf, ms), ("BC clone", cf, cs)):
        r = [R.rollout(pol, sd, t_end=150.0 * _time_scale(a.hull),
                       t_preview=a.preview, reset=st, hull=a.hull)
             for sd in SEEDS]
        g = {k: float(np.mean([x[k] for x in r]))
             for k in ("u_mean", "acc_p99", "rvm_rms", "slam_impulse",
                       "cross_rms", "heading_rms")}
        out[nm] = g
        print(f"  {nm:<14}{g['u_mean']:>6.2f}{g['acc_p99']:>9.3f}"
              f"{g['rvm_rms']:>7.3f}{g['slam_impulse']:>9.1f}"
              f"{g['cross_rms']:>7.2f}m{g['heading_rms']:>7.1f}d")
    c, m = out["BC clone"], out["MPC"]
    ok = (c["heading_rms"] < 5 * m["heading_rms"]
          and c["cross_rms"] < 5 * m["cross_rms"])
    print("\n  " + "-" * 62)
    if ok:
        print("  The clone flies. Whatever went wrong happened in FINE-TUNING,")
        print("  and the culprit is the action noise: cloning leaves log_std at")
        print("  1.0, so the first PPO rollouts sample near-randomly and throw")
        print("  the clone away before it can be improved.")
    else:
        print("  The clone does NOT fly on its own. This is distribution shift:")
        print("  it matches the MPC on the MPC's states, drifts off them, and")
        print("  has never seen where it ends up. Fine-tuning is not the")
        print("  problem, and the fix is DAgger -- relabel the CLONE's own")
        print("  states with what the MPC would have done there.")
    return out


if __name__ == "__main__":
    main()
