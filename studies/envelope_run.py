#!/usr/bin/env python3
"""
One run of the achievable-envelope study (studies/envelope_pass1.py): the
frozen plant scarab195@as1 on its large-angle twin, held at a set speed by
the test scaffolding's P+I throttle (studies/app_randomization_test._run:
k_p 0.15, k_i 0.02 of t_max per m/s and per control step) and on a set
heading by the project's heading autopilot (UsvEnv action mode 'heading'),
in the frozen scenario builder's sea (app/frozen.scenario: JONSWAP gamma
3.3, 24 frequencies, 5 directions), for SETTLE + RUN seconds; the first
SETTLE seconds are dropped from every statistic.

Read-only use of sim/, hydro/, app/. The env is built with record=False
(the full record keeps the physics' hidden state per plant step, far too
large for 66 000 steps) and a plant hook collects what the study needs
after every plant step:

    t, a_cg, a_station (raw m/s^2), heave z, roll phi, pitch th, speed u,
    yaw rate r, the bow station's relative immersion (plant.last_rel_bow:
    the keel's immersion at the stem minus its value at rest, so the
    running bow freeboard is plant.freeboard - last_rel_bow) and the
    plant's slam counter (plant.slam_count: bow re-entries faster than
    v_slam after an emergence).

Outputs of run_one(arg): a dict of scalar results and arrays (the filtered
CG and bow-gauge accelerations in g at the plant step, the slow channels
decimated), saved by the driver as one .npz per run.

The bow gauge is the output spec's (app/accel.py): 0.1 L aft of the stem,
interpolated between the CG and the plant's bow station; the filter the
spec's zero-phase 2nd-order Butterworth at 20 Hz.
"""
import math
import os
import time
import warnings

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

KN = 0.514444
G = 9.81
K_P, K_I = 0.15, 0.02          # studies/app_randomization_test._run
I_CLIP = 2.0                   # the scaffolding's integrator clip (m/s-steps)
SETTLE = 60.0
RUN = 600.0
FB_FRAC = 0.25                 # bow immersion: freeboard below this x static
DECIM = 10                     # slow channels kept every DECIM-th plant step


class Hook:
    """Plant hook (UsvEnv plant_hooks): collects the study's channels after
    every plant step; returns None (the state is not changed)."""

    def __init__(self):
        self.rows = []

    def __call__(self, s, dt, plant):
        self.rows.append((
            float(plant.last_cg_acc),
            float(getattr(plant, "last_bow_acc", np.nan)),
            float(s[2]), float(s[3]), float(s[4]), float(s[6]),
            float(s[11]),
            float(getattr(plant, "last_rel_bow", np.nan)),
            float(getattr(plant, "slam_count", np.nan))))
        return None


def make_env(arg):
    """UsvEnv for a run argument {hs, tp, heading_rad, speed_kn, seed,
    sea_seed, randomization (dict or None), duration}."""
    from app import frozen
    from app.env import UsvEnv
    usv = dict(sea_seed=int(arg.get("sea_seed", arg["seed"])))
    sc = frozen.scenario(hs=float(arg["hs"]), tp=float(arg["tp"]),
                         theta0=math.pi, heading=float(arg["heading_rad"]),
                         duration=float(arg.get("duration", SETTLE + RUN)),
                         seed=int(arg["seed"]),
                         speed=float(arg["speed_kn"]) * KN, **usv)
    rz = arg.get("randomization")
    hook = Hook()
    env = UsvEnv(sc, action_mode="heading", record=False,
                 plant_hooks=[hook], randomization=rz)
    return env, hook


def run_one(arg):
    """One episode; returns the study's results for it (see the module
    docstring). `arg` also carries 'name' (the run's label) and 'randomize'
    (False / True: the env draws from its Randomization with the episode
    seed)."""
    warnings.filterwarnings("ignore")
    from app import accel
    t_wall0 = time.perf_counter()
    env, hook = make_env(arg)
    t_build0 = time.perf_counter()
    obs, info = env.reset(seed=int(arg["seed"]),
                          options={"randomize": bool(arg.get("randomize",
                                                             False))})
    t_build = time.perf_counter() - t_build0
    u_ref = float(arg["speed_kn"]) * KN
    heading = float(arg["heading_rad"])
    thr0 = float(env.s[12] / env.t_max)
    i_clip = float(arg.get("i_clip", I_CLIP))
    acc_err = 0.0
    thr_log = []
    reason = None
    t_step0 = time.perf_counter()
    while True:
        err = u_ref - float(env.s[6])
        acc_err = min(i_clip, max(-i_clip, acc_err + err))
        thr = min(1.0, max(0.0, thr0 + K_P * err + K_I * acc_err))
        thr_log.append(thr)
        obs, r, term, trunc, info = env.step([thr, heading])
        if term or trunc:
            reason = info["reason"]
            break
    t_step = time.perf_counter() - t_step0
    rows = np.asarray(hook.rows, float)
    n = len(rows)
    dt = env.dt
    t = dt * np.arange(1, n + 1)
    settle = float(arg.get("settle", SETTLE))
    # the output spec's series: bow gauge, g, zero-phase filter at 20 Hz
    a_cg, a_st = rows[:, 0], rows[:, 1]
    bow = accel.bow_gauge(a_cg, a_st, env.x_gauge, env.x_station)
    q = np.column_stack([a_cg, bow]) / G
    qf = accel.lowpass_offline(q, dt, env.filter_hz)
    keep = t > settle + 1e-9
    fb0 = float(getattr(env.plant, "freeboard", np.nan))
    fb = fb0 - rows[:, 7]
    out = dict(name=arg["name"], arg={k: v for k, v in arg.items()
                                     if k != "randomization"},
               dt=dt, n_plant=int(n), t_end=float(t[-1]) if n else 0.0,
               settle=settle, t_stat=float(keep.sum() * dt),
               reason=reason, capsized=bool(env.capsized),
               finite=bool(env.finite),
               wall_total=time.perf_counter() - t_wall0,
               wall_build=t_build, wall_step=t_step,
               ms_per_step=1e3 * t_step / max(n, 1),
               x_gauge=float(env.x_gauge), x_station=float(env.x_station),
               freeboard_static=fb0, t_max=float(env.t_max), thr0=thr0,
               draw=(None if env.draw is None else dict(env.draw)),
               hull_applied=dict(env.hull_applied),
               plant_source=env.plant_source,
               sea_params={k: (float(v) if isinstance(v, (int, float))
                               else v) for k, v in
                           (env.sea_params or {}).items()},
               sea_seed=int(env.sea_seed), seed=int(env.seed_))
    if keep.sum() < 10:
        out.update(stats=None)
        return out
    w = qf[keep]
    st = {}
    for j, nm in enumerate(("cg", "bow")):
        pk = accel.excursion_peaks(w[:, j])
        ps = accel.peak_stats(w[:, j])
        st[nm] = dict(mean=ps["mean"], a110=ps["a110"], max=ps["max"],
                      n=ps["n"], peaks=np.sort(pk)[::-1])
    u = rows[keep, 5]
    phi = rows[keep, 3]
    rr = rows[keep, 6]
    fbk = fb[keep]
    thr_a = np.asarray(thr_log, float)
    nk = int(round(settle / env.dt_step))
    below = fbk < FB_FRAC * fb0
    n_imm = int(np.sum(below[1:] & ~below[:-1]) + (1 if below[0] else 0))
    slam = rows[:, 8]
    k0 = int(np.searchsorted(t, settle))
    out.update(stats=dict(
        u_mean=float(u.mean()), u_sd=float(u.std()),
        u_min=float(u.min()), u_max=float(u.max()),
        u_mean_kn=float(u.mean() / KN),
        thr_mean=float(thr_a[nk:].mean()) if len(thr_a) > nk else
        float("nan"),
        thr_sat_frac=float(np.mean(thr_a[nk:] >= 1.0 - 1e-9))
        if len(thr_a) > nk else float("nan"),
        roll_max_deg=float(np.degrees(np.abs(phi).max())),
        roll_rms_deg=float(np.degrees(np.sqrt(np.mean(phi ** 2)))),
        yaw_rate_max_dps=float(np.degrees(np.abs(rr).max())),
        yaw_rate_rms_dps=float(np.degrees(np.sqrt(np.mean(rr ** 2)))),
        fb_min=float(np.nanmin(fbk)), fb_mean=float(np.nanmean(fbk)),
        n_immersion=n_imm,
        immersion_per_hour=float(n_imm * 3600.0 / (keep.sum() * dt)),
        slam_count=int(slam[-1] - slam[k0]) if np.isfinite(slam[-1])
        else None,
        slam_per_hour=(float((slam[-1] - slam[k0]) * 3600.0
                             / (keep.sum() * dt))
                       if np.isfinite(slam[-1]) else None),
        heave_sd=float(rows[keep, 2].std()),
        pitch_sd_deg=float(np.degrees(rows[keep, 4].std())),
        cg=st["cg"], bow=st["bow"]))
    # the series kept (acceleration at the plant step, the rest decimated)
    out.update(series=dict(
        acc_f=qf.astype(np.float32),
        t_slow=t[::DECIM].astype(np.float32),
        u=rows[::DECIM, 5].astype(np.float32),
        roll=rows[::DECIM, 3].astype(np.float32),
        pitch=rows[::DECIM, 4].astype(np.float32),
        heave=rows[::DECIM, 2].astype(np.float32),
        yaw_rate=rows[::DECIM, 6].astype(np.float32),
        freeboard=fb[::DECIM].astype(np.float32),
        slam_count=rows[::DECIM, 8].astype(np.float32),
        throttle=thr_a.astype(np.float32)))
    return out


if __name__ == "__main__":
    # smoke test: 20 kn head seas, 30 s settle + 60 s
    import sys
    kn = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    r = run_one(dict(name="smoke", hs=1.0, tp=5.0, heading_rad=0.0,
                     speed_kn=kn, seed=0, duration=30.0 + T, settle=30.0))
    s = r["stats"]
    print(f"steps {r['n_plant']}  {r['ms_per_step']:.2f} ms/step  build "
          f"{r['wall_build']:.1f} s  total {r['wall_total']:.1f} s  "
          f"reason {r['reason']}")
    print(f"freeboard static {r['freeboard_static']:.3f} m  thr0 {r['thr0']:.3f}")
    if s:
        print(f"u mean {s['u_mean_kn']:.2f} kn (sd {s['u_sd']:.2f} m/s) thr "
              f"mean {s['thr_mean']:.3f} sat {s['thr_sat_frac']:.2f}")
        print(f"cg a110 {s['cg']['a110']:.2f} g max {s['cg']['max']:.2f} n "
              f"{s['cg']['n']} | bow a110 {s['bow']['a110']:.2f} max "
              f"{s['bow']['max']:.2f}")
        print(f"roll max {s['roll_max_deg']:.2f} deg  yaw rate max "
              f"{s['yaw_rate_max_dps']:.2f} deg/s  fb min {s['fb_min']:.3f} "
              f"mean {s['fb_mean']:.3f}  immersions {s['n_immersion']}  "
              f"slams {s['slam_count']}")
