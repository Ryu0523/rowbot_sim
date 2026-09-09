#!/usr/bin/env python3
"""
PPO against MPC, on seas neither of them was tuned on.

Four controllers, ONE measurement path. That last part is the whole design:
every earlier comparison in this project ran each controller through its own
harness, and at least one conclusion turned out to be an artefact of the
harness rather than the controller. Here each policy is just a function
(observation -> action) driven by the same environment loop, and every metric
comes from the same `sim/seakeeping.py` code.

    constant thrust   the no-information baseline. Beat this or go home.
    autopilot         constant thrust, classical Nomoto heading hold
    MPC               192 rollouts every 0.5 s through the 10-state reduced
                      model, steering, with 8 s of preview
    PPO               a network trained against the full 110-state plant, with
                      the same 8 s of preview in its observation

EVALUATION SEEDS ARE DISJOINT FROM TRAINING SEEDS. PPO trained on 0-23 and is
scored on 100-111. The MPC has no training seeds, which is an advantage it gets
to keep -- that asymmetry is the honest point of the comparison, not a flaw in
it: a controller you can deploy without training is worth something.

Nothing here is compared at equal speed, because all four are free to choose
their own speed and the objective already prices that in. `u_mean` is reported
so the trade is visible.

Run: python -m studies.rl_vs_mpc
"""
import json
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from hydro import bem
from sim.env import FREEBOARD, score
from sim.rl_env import USVControlEnv
from sim.vessel import NonlinearVessel
from sim.test_vessel import Monochromatic
from sim import seakeeping
from control.reduced import ReducedModel
from control.mpc import MPPIController, PreviewProvider

G = 9.81
EVAL_SEEDS = list(range(100, 112))
T_END = 200.0
T_PREVIEW = 8.0
KEYS = ("acc_rms", "acc_p99", "fatigue", "rvm_rms", "rvv_rms",
        "slam_impulse", "slam_rate_ochi", "wet_rate")


def rollout(policy, seed, t_end=T_END, t_preview=T_PREVIEW, reset=None):
    """One episode, one policy, one set of measurements."""
    env = USVControlEnv(seeds=(seed,), t_end=t_end, t_preview=t_preview,
                        n_freq=16, n_dir=4)
    obs, _ = env.reset(seed=0)
    if reset is not None:
        reset(env)
    acc, rel, sf, spd, yaw, cross, rol = [], [], [], [], [], [], []
    while True:
        a = policy(obs, env)
        obs, _, term, trunc, _ = env.step(a)
        s, p = env._s, env._plant
        acc.append(p.last_bow_acc / G); rel.append(p.last_rel_bow)
        sf.append(p.last_slam_force); spd.append(s[6])
        yaw.append(s[5]); cross.append(s[1]); rol.append(s[3])
        if term or trunc:
            break
    dt = env.dt_ctrl
    m = seakeeping.summarise(rel, acc, dt, env._plant.T, FREEBOARD,
                             env._plant.v_slam)
    m.update(u_mean=float(np.mean(spd)),
             slam_impulse=float(np.sum(sf) * dt / (t_end / 60.0) / 1e3),
             heading_rms=float(np.degrees(np.sqrt(np.mean(np.square(yaw))))),
             cross_rms=float(np.sqrt(np.mean(np.square(cross)))),
             roll_rms=float(np.degrees(np.std(rol))),
             slam_per_min=float(env._plant.slam_count / (t_end / 60.0)),
             cross_rms_=0.0, finite=True)
    return m


# ---------------------------------------------------------------- policies
def const_policy(frac=0.55):
    a = np.array([2 * frac - 1, 0.0])
    return lambda o, e: a


def autopilot_policy(frac=0.55, kp=0.8, kd=1.2):
    def f(o, e):
        s = e._s
        err = (s[5] + np.pi) % (2 * np.pi) - np.pi
        rud = np.clip(kp * err + kd * s[11], -1, 1)
        return np.array([2 * frac - 1, rud])
    return f


def mpc_policy(db, red, seed):
    ctrl = {}

    def setup(env):
        pv = PreviewProvider(env.sea, T_PREVIEW, 0.0, seed)
        ctrl["c"] = MPPIController(red, pv, dt_ctrl=env.dt_ctrl, u_ref=4.5,
                                   seed=seed, n_samples=192, use_rudder=True)
        ctrl["t"] = 0.0

    def f(o, e):
        c = ctrl["c"]
        sr = red.from_plant_state(e._s)
        cmd = c(sr, ctrl["t"])
        ctrl["t"] += e.dt_ctrl
        thr, rud = c.to_actuator(cmd)
        return np.array([2 * (thr / e._plant.prop.t_max) - 1,
                         rud / e._plant.rudder.max])
    return f, setup


def ppo_policy(path="ppo_usv", n_stack=6):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    import pickle
    model = PPO.load(path, device="cpu")
    with open(path + "_vecnorm.pkl", "rb") as fh:
        vn = pickle.load(fh)
    buf = {}

    def setup(env):
        buf["s"] = None

    def f(o, e):
        # reproduce VecFrameStack: newest frame last
        if buf["s"] is None:
            buf["s"] = np.tile(o, n_stack)
        else:
            buf["s"] = np.concatenate([buf["s"][len(o):], o])
        x = np.clip((buf["s"] - vn.obs_rms.mean)
                    / np.sqrt(vn.obs_rms.var + vn.epsilon), -10, 10)
        a, _ = model.predict(x.astype(np.float32), deterministic=True)
        return np.asarray(a, float).ravel()
    return f, setup


def main():
    db = bem.load("hydro_wigley_10m.npz")
    red = ReducedModel.identify(
        NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L), db)

    entries = [("constant thrust", const_policy(), None),
               ("autopilot", autopilot_policy(), None)]
    mf, ms = mpc_policy(db, red, 0)
    entries.append(("MPC (steering, 8 s preview)", mf, ms))
    try:
        pf, ps = ppo_policy()
        entries.append(("PPO (8 s preview)", pf, ps))
    except Exception as exc:
        print(f"  [no PPO model: {exc}]")

    print(f"{len(EVAL_SEEDS)} HELD-OUT seas x {T_END:.0f} s "
          f"(PPO trained on seeds 0-23, scored on 100-111)\n")
    rows = []
    for name, pol, setup in entries:
        runs = [rollout(pol, sd, reset=setup) for sd in EVAL_SEEDS]
        m = {k: float(np.mean([r[k] for r in runs]))
             for k in KEYS + ("u_mean", "heading_rms", "cross_rms",
                              "roll_rms", "slam_per_min")}
        m.update({k + "_se": float(np.std([r[k] for r in runs], ddof=1)
                                   / np.sqrt(len(runs)))
                  for k in KEYS + ("u_mean",)})
        m["J"] = float(np.mean([score(r, 4.5) for r in runs]))
        m["J_se"] = float(np.std([score(r, 4.5) for r in runs], ddof=1)
                          / np.sqrt(len(runs)))
        m["name"] = name
        rows.append(m)
        print(f"  {name:<30} done")

    print(f"\n  {'controller':<30}{'J':>8}{'+-':>6}{'u':>7}{'acc_p99':>9}"
          f"{'rvm':>7}{'slamImp':>9}{'wet':>7}{'hdg':>7}{'cross':>7}")
    for m in rows:
        print(f"  {m['name']:<30}{m['J']:>8.3f}{m['J_se']:>6.3f}"
              f"{m['u_mean']:>7.2f}{m['acc_p99']:>9.3f}{m['rvm_rms']:>7.3f}"
              f"{m['slam_impulse']:>9.1f}{m['wet_rate']:>7.2f}"
              f"{m['heading_rms']:>6.1f}d{m['cross_rms']:>6.2f}m")

    base = rows[0]
    print(f"\n  against constant thrust, on the operator objective J:")
    for m in rows[1:]:
        d = 100 * (m["J"] - base["J"]) / abs(base["J"])
        dse = 100 * np.sqrt(m["J_se"] ** 2 + base["J_se"] ** 2) / abs(base["J"])
        print(f"    {m['name']:<30}{d:>+8.1f}% +-{dse:.1f}"
              f"   {'RESOLVED' if abs(d) > 2 * dse else 'not resolved'}")
    json.dump(rows, open("rl_vs_mpc.json", "w"), indent=1, default=float)
    return rows


if __name__ == "__main__":
    main()
