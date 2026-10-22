#!/usr/bin/env python3
"""
Tests of the frozen configuration, the domain randomisation and the output
spec in the product-layer environment (learn/meta/PRODUCT_LAYER.md, section
9; learn/meta/SIM_FINAL_2026-10-09.md sections 4.1, 5, 6).

    python -m studies.app_randomization_test               all groups
    python -m studies.app_randomization_test --groups frozen,randomization
    python -m studies.app_randomization_test --full        + the 0.005 s
                                                           reference run

Groups (run one after the other in this process; plant-only, no policy):
  frozen         the default env is the frozen configuration: scarab195@as1,
                 PlaningVesselAS (gz twin), the as1 preset, dt 0.01 s, 25
                 sub-steps; a v0 hull keeps 0.02 s / 12; usv.dt 0.005 gives
                 50; Gymnasium check_env; the physics stamp in info and in
                 the record; the filter's coefficients and causal output
                 equal scipy's; the statistics helpers against the harness
                 (studies/exp_planing.excursion_peaks) and on known signals
  randomization  3 episodes with randomisation on and 1 off: finite outputs,
                 the draw in info and header, the episode's hull copy and
                 stamp, ms per plant step; randomize=False with a fixed seed
                 reproduces the first 50 control steps bit for bit in two
                 fresh envs; randomize=True with the same seed reproduces
                 the draw and the run; a randomised record replays bit for
                 bit (the draw re-applied from the header); the output scales
                 leave the physics untouched; the thrust gain reaches the
                 plant; the short-wave components are the ones scaled; the
                 config forms, refusals and the train / eval streams
  hull_every     (2026-10-10) Randomization.hull_every and the plant cache
                 (app/randomize.py HULL_EVERY, PLANT CACHE): the LRU itself;
                 hull_every = 3 over six randomised episodes (whose hull is
                 in force, the other draws each episode's own, header and
                 info fields, the plant rebuilt on episodes 0 and 3 only); a
                 skipped-hull episode's record replays bit for bit; cache
                 isolation (two episodes on one cached hull with different
                 seas equal a fresh env's single episodes bit for bit; the
                 prototype is never stepped); the cache off gives the same
                 bits as the cache; the reset costs (miss, hit, hull_every
                 = 10 amortised, sea-only, off)
  reference      20 kn, head seas, Hs 1.0 m, Tp 5 s (the evaluation sea),
                 60 s at dt 0.01 on the frozen plant, throttle held by a
                 proportional speed hold inside the test: the filtered CG /
                 bow-gauge acceleration statistics (mean of the positive
                 peaks, A1/10, max, count; raw beside) and ms per plant step
                 -- the control side's reference numbers

Results: studies/_cache/app/randomization_results.json.
"""
import argparse
import json
import math
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np   # noqa: E402

from studies.app_test import Checks, bits_equal   # noqa: E402

OUT = os.path.join(ROOT, "studies", "_cache", "app")
KN = 0.514444


def _run(env, action, n_max=None, hold_kn=None, k_p=0.15, k_i=0.02):
    """Step `env` with a constant action (or a speed hold on the throttle:
    the calm-water thrust fraction of the start speed plus k_p per m/s of
    speed error and an integral k_i per m/s per control step -- test
    scaffolding, not a policy) until the episode ends or n_max control
    steps; returns (obs list, info list, wall seconds of the stepping)."""
    obs_l, info_l = [], []
    t0 = time.perf_counter()
    k = 0
    thr0 = float(env.s[12] / env.t_max) if hold_kn is not None else None
    acc_err = 0.0
    while True:
        if hold_kn is not None:
            err = hold_kn * KN - float(env.s[6])
            acc_err = min(2.0, max(-2.0, acc_err + err))
            a = [min(1.0, max(0.0, thr0 + k_p * err + k_i * acc_err)), 0.0]
        else:
            a = action
        obs, r, term, trunc, info = env.step(a)
        obs_l.append(obs)
        info_l.append(info)
        k += 1
        if term or trunc or (n_max is not None and k >= n_max):
            break
    return obs_l, info_l, time.perf_counter() - t0


def _finite_obs(obs):
    return all(np.all(np.isfinite(v)) for v in obs.values())


# ================================================================== frozen
def g_frozen():
    C = Checks("frozen")
    from app import accel, frozen
    from app.env import UsvEnv
    from app.scenario import straight
    from sim import physics
    from sim.planing_anyspeed import PRESETS

    env = UsvEnv(frozen.scenario(duration=1.0, seed=3))
    C.ok("default model is the frozen selector, on the gz twin",
         env.hull_key == frozen.MODEL and env.plant_kind == "gz"
         and type(env.plant).__name__ == "PlaningVesselAS",
         f"{env.hull_key}, {env.plant_kind}, {type(env.plant).__name__}")
    C.ok("as1 preset switches, physics as1",
         tuple(env.plant.switches) == PRESETS["as1"]
         and physics.of(env.plant) == "as1", env.plant.switches)
    C.ok("steps: dt 0.01 s, 25 sub-steps, control step 0.25 s",
         env.dt == 0.01 and env.sub == 25 and abs(env.dt_step - 0.25) < 1e-12
         and env.frozen, (env.dt, env.sub, env.dt_step))
    C.ok("observation has the 'accel' channel (2,)",
         "accel" in env.observation_space.spaces
         and env.observation_space["accel"].shape == (2,),
         sorted(env.observation_space.spaces))
    obs, info = env.reset(seed=3)
    st = physics.stamp(env.plant)
    C.ok("info['physics'] is sim.physics.stamp(plant)",
         info["physics"] == st and st["physics"] == "as1"
         and st["hull"] == frozen.MODEL, st)
    obs_l, info_l, wall = _run(env, [0.6, 0.0])
    rec = env.record()
    h = rec.header
    C.ok("record header: stamp, nominal stamp, frozen flag, filter, gauge",
         h["physics"] == st and h["physics_nominal"] == st and h["frozen"]
         and h["filter_hz"] == 20.0 and h["randomization"] is None
         and abs(h["x_bow_gauge"] - (env.hull.L - env.hull.lcg
                                      - 0.1 * env.hull.L)) < 1e-12
         and h["x_bow_station"] == float(env.plant.x[-1]),
         {k: h[k] for k in ("frozen", "filter_hz", "x_bow_gauge",
                            "x_bow_station", "dt", "sub")})
    C.ok("record: raw a_cg / a_bow (m/s^2) and acc_obs (g, filtered) rows",
         rec.plant["acc_obs"].shape == (len(rec.plant["t"]), 2)
         and np.all(np.isfinite(rec.plant["acc_obs"]))
         and np.all(np.isfinite(rec.plant["a_cg"])),
         f"{len(rec.plant['t'])} rows; last acc_obs {rec.plant['acc_obs'][-1]}")

    def gym_check():
        from gymnasium.utils.env_checker import check_env
        e = UsvEnv(frozen.scenario(duration=1.0, seed=1))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_env(e, skip_render_check=True)
        e2 = UsvEnv(frozen.scenario(duration=1.0, seed=1, obs_truth=True))
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            check_env(e2, skip_render_check=True)
        return f"{len(set(str(x.message)[:80] for x in w + w2))} warnings"
    C.run("Gymnasium check_env on the frozen env (direct; with obs_truth)",
          gym_check)

    e0 = UsvEnv(straight("scarab195", duration=1.0, seed=1))
    C.ok("a v0 hull keeps its own steps (0.02 s, 12 sub-steps)",
         e0.dt == 0.02 and e0.sub == 12 and not e0.frozen,
         (e0.dt, e0.sub))
    e5 = UsvEnv(frozen.scenario(duration=1.0, seed=1, dt=0.005))
    C.ok("usv.dt 0.005 (A1/10 and extremes): 50 sub-steps",
         e5.dt == 0.005 and e5.sub == 50 and not e5.frozen, (e5.dt, e5.sub))
    e6 = UsvEnv(frozen.scenario(duration=1.0, seed=1), dt=0.02,
                dt_ctrl=0.25)
    C.ok("UsvEnv(dt=0.02, dt_ctrl=0.25) (an old as1 record's steps): 12",
         e6.dt == 0.02 and e6.sub == 12, (e6.dt, e6.sub))

    # ------------------------------------------------------------ filter
    def filt_coeff():
        from scipy.signal import butter, lfilter, lfilter_zi
        worst = 0.0
        for fs in (100.0, 200.0, 50.0):
            b, a = accel.butter2_lowpass(20.0, fs)
            bs, as_ = butter(2, 20.0 / (0.5 * fs))
            worst = max(worst, float(np.max(np.abs(b - bs))),
                        float(np.max(np.abs(a - as_))))
            rng = np.random.default_rng(0)
            x = np.cumsum(rng.normal(size=400)) + 3.0
            f = accel.CausalLowpass(20.0, fs, n=1)
            y = np.array([float(f(v)) for v in x])
            zi = lfilter_zi(bs, as_) * x[0]
            ys, _ = lfilter(bs, as_, x, zi=zi)
            worst = max(worst, float(np.max(np.abs(y - ys))))
        assert worst < 1e-10, worst
        f = accel.CausalLowpass(20.0, 100.0, n=2)
        c = np.array([0.7, -1.2])
        same = all(np.allclose(f(c), c, rtol=0, atol=1e-14) for _ in range(5))
        assert same, "a constant input must come out unchanged (to round-off)"
        fi = accel.CausalLowpass(20.0, 40.0, n=1)
        assert fi.identity and fi(np.array([2.0]))[0] == 2.0
        return f"max |diff| to scipy {worst:.2e}; constant unchanged; " \
               f"identity above Nyquist"
    C.run("filter: closed-form coefficients and causal output == scipy "
          "(butter, lfilter with lfilter_zi)", filt_coeff)

    def stats_check():
        from studies.exp_planing import excursion_peaks as harness_peaks
        rng = np.random.default_rng(1)
        y = np.sin(np.linspace(0, 40 * math.pi, 4000)) * (1 + 0.3 * rng.normal(
            size=4000)) + 0.2 * rng.normal(size=4000)
        a, b = accel.excursion_peaks(y), harness_peaks(y)
        assert np.array_equal(a, b), (len(a), len(b))
        st = accel.irregular_stats(np.column_stack([y, 2 * y]), drop=0.1)
        assert st["bow_n"] == st["cg_n"] and abs(
            st["bow_mean"] - 2 * st["cg_mean"]) < 1e-12
        assert st["cg_a110"] >= st["cg_mean"] and st["cg_max"] >= st["cg_a110"]
        om = 2 * math.pi / 1.3
        dt = 0.01
        t = np.arange(0, 20, dt)
        q = np.column_stack([0.5 * np.sin(om * t), 1.5 * np.sin(om * t)])
        rs = accel.regular_stats(q, dt, om, drop=0.5)
        assert abs(rs["cg_cycmean"] - 0.5) < 1e-3 and abs(
            rs["bow_cycmean"] - 1.5) < 3e-3, rs
        pk = accel.excursion_peaks(y)
        rate = accel.exceedance_per_hour(pk, 1.0, 40.0)
        assert rate == float((pk > 1.0).sum()) * 90.0
        g = accel.bow_gauge(1.0, 3.0, 1.0, 2.0)
        assert g == 2.0
        return f"{len(a)} peaks == harness; cycle means {rs['cg_cycmean']:.4f}" \
               f" / {rs['bow_cycmean']:.4f}; {st['cg_n']} excursions"
    C.run("statistics: excursion peaks == studies/exp_planing's, A1/10 and "
          "regular cycle means on known signals, event rate, gauge", stats_check)

    def rec_stats():
        t, q = accel.gauge_series(rec)
        t2, qf = accel.filtered_series(rec)
        st = accel.record_stats(rec)
        assert q.shape == qf.shape == (len(rec.plant["t"]), 2)
        assert np.all(np.isfinite(qf)) and st["fc_hz"] == 20.0
        # the gauge from the recorded CG and station values
        xg, xs = rec.header["x_bow_gauge"], rec.header["x_bow_station"]
        bow = rec.plant["a_cg"] + (rec.plant["a_bow"] - rec.plant["a_cg"]) \
            * (xg / xs)
        assert np.allclose(q[:, 1] * 9.81, bow, atol=1e-12)
        return {k: round(v, 3) if isinstance(v, float) else v
                for k, v in st.items() if k.startswith(("cg", "raw_cg"))}
    C.run("record helpers: gauge_series, filtered_series, record_stats",
          rec_stats)
    return C.result()


# =========================================================== randomization
def g_randomization():
    C = Checks("randomization")
    from app import frozen, randomize
    from app.env import UsvEnv
    from app.record import replay
    from sim import physics
    from learn.repro.task import EVAL_SEEDS

    rz = randomize.Randomization(seed=11)
    sc = frozen.scenario(duration=4.0, seed=100)
    env = UsvEnv(sc, randomization=rz)
    nominal = physics.stamp(env.plant)
    rows = []
    draws = []
    for i, (seed, on) in enumerate([(100, True), (101, True), (102, True),
                                    (100, False)]):
        obs, info = env.reset(seed=seed, options={"randomize": on})
        d = info["randomization"]
        obs_l, info_l, wall = _run(env, [0.7, 0.0])
        rec = env.record()
        n = len(rec.plant["t"])
        ms = wall / n * 1e3
        fin = (all(_finite_obs(o) for o in obs_l) and all(
            i_["finite"] for i_ in info_l) and np.all(np.isfinite(
                rec.plant["acc_obs"])) and np.all(np.isfinite(rec.plant["s"])))
        h = rec.header
        rows.append(dict(seed=seed, on=on, finite=bool(fin), ms_per_step=ms,
                         build_s=h["plant_build_s"], hull=h["episode_hull"],
                         stamp=h["physics"]["physics_sha1"][:12],
                         n_hull_fields=len(h["randomization_applied"]["hull"]),
                         unapplied=h["randomization_applied"]["unapplied"],
                         reason=h["reason"]))
        if on:
            draws.append(d)
            C.ok(f"episode seed {seed}, randomisation on: finite outputs, "
                 f"draw in info and header, hull copy and its own stamp",
                 fin and d is not None and h["randomization"] == d
                 and set(d) == set(randomize.PARAMS)
                 and h["episode_hull"].endswith("~rz@as1")
                 and h["physics"]["physics_sha1"] != nominal["physics_sha1"]
                 and h["physics_nominal"] == nominal
                 and h["randomization_applied"]["unapplied"] == [],
                 f"{ms:.2f} ms/plant step ({n} steps, {env.sub} sub-steps), "
                 f"plant build {h['plant_build_s']:.1f} s, "
                 f"{randomize.describe(d)[:160]}")
        else:
            C.ok(f"episode seed {seed}, randomisation off: finite, no draw, "
                 f"nominal hull and stamp",
                 fin and d is None and h["randomization"] is None
                 and h["episode_hull"] == frozen.MODEL
                 and h["physics"] == nominal
                 and h["physics_nominal"] == nominal,
                 f"{ms:.2f} ms/plant step ({n} steps)")
    C.ok("the three draws differ", len({json.dumps(d, sort_keys=True)
                                        for d in draws}) == 3,
         [round(d["mass"], 1) for d in draws])
    C.ok("every draw inside its range",
         all(getattr(rz, k).lo <= v <= getattr(rz, k).hi
             for d in draws for k, v in d.items()))
    env.close()

    # ------------------------------------------- deterministic regression
    def regression():
        a = UsvEnv(frozen.scenario(duration=30.0, seed=7))
        b = UsvEnv(frozen.scenario(duration=30.0, seed=7))
        a.reset(seed=7, options={"randomize": False})
        b.reset(seed=7, options={"randomize": False})
        oa, ia, _ = _run(a, [0.65, 0.1], n_max=50)
        ob, ib, _ = _run(b, [0.65, 0.1], n_max=50)
        same = all(bits_equal(x[k], y[k]) for x, y in zip(oa, ob)
                   for k in x) and all(
            bits_equal(x["state"], y["state"]) and x["reward"] == y["reward"]
            and bits_equal(x["accel"], y["accel"]) for x, y in zip(ia, ib))
        ra, rb = a.record(), b.record()
        assert same and ra.fingerprint() == rb.fingerprint(), ra.diff(rb)[:3]
        return f"50 control steps ({len(ra.plant['t'])} plant rows) " \
               f"identical; fingerprint {ra.fingerprint()[:16]}"
    C.run("randomize=False, seed 7: the first 50 control steps reproduce "
          "bit for bit in two fresh envs (obs, states, rewards, record)",
          regression)

    def rand_det():
        a = UsvEnv(frozen.scenario(duration=20.0, seed=5), randomization=rz)
        b = UsvEnv(frozen.scenario(duration=20.0, seed=5), randomization=rz)
        _, ia0 = a.reset(seed=21, options={"randomize": True})
        _, ib0 = b.reset(seed=21, options={"randomize": True})
        assert ia0["randomization"] == ib0["randomization"]
        oa, ia, _ = _run(a, [0.7, 0.0], n_max=12)
        ob, ib, _ = _run(b, [0.7, 0.0], n_max=12)
        assert all(bits_equal(x["state"], y["state"]) for x, y in zip(ia, ib))
        ra, rb = a.record(), b.record()
        assert ra.fingerprint() == rb.fingerprint(), ra.diff(rb)[:3]
        return f"same draw, same run ({len(ra.plant['t'])} rows)"
    C.run("randomize=True, same seed twice: the same draw and the same run",
          rand_det)

    def rand_replay():
        e = UsvEnv(frozen.scenario(duration=5.0, seed=9), randomization=rz)
        e.reset(seed=33, options={"randomize": True})
        _run(e, [0.75, -0.2])
        rec = e.record()
        back = replay(rec)
        assert back.header["randomization"] == rec.header["randomization"]
        assert back.fingerprint(physical=True) == rec.fingerprint(
            physical=True), rec.diff(back, physical=True)[:3]
        # the plant got the command times the gain
        g = rec.header["randomization"]["thrust_gain"]
        thr_c = np.repeat(rec.ctrl["thrust"], e.sub)[:len(rec.plant["t"])]
        assert np.allclose(rec.plant["thrust"], thr_c * g, rtol=0, atol=1e-9)
        return f"physical fingerprint {rec.fingerprint(physical=True)[:16]}; " \
               f"thrust gain {g:.4f} applied"
    C.run("a randomised record replays bit for bit (the draw re-applied "
          "from the header); the thrust gain reaches the plant", rand_replay)

    def outputs_only():
        e0 = UsvEnv(frozen.scenario(duration=4.0, seed=4, obs_truth=True))
        e0.reset(seed=4)
        o0, i0, _ = _run(e0, [0.7, 0.0])
        r0 = e0.record()
        e1 = UsvEnv(frozen.scenario(duration=4.0, seed=4, obs_truth=True))
        e1.reset(seed=4, options={"randomize": {"acc_scale": 1.3,
                                                "motion_scale": 0.8}})
        o1, i1, _ = _run(e1, [0.7, 0.0])
        r1 = e1.record()
        assert bits_equal(r0.plant["s"], r1.plant["s"]) and bits_equal(
            r0.plant["a_cg"], r1.plant["a_cg"]) and bits_equal(
            r0.plant["thrust"], r1.plant["thrust"])
        assert np.allclose(r1.plant["acc_obs"], 1.3 * r0.plant["acc_obs"],
                           rtol=1e-12, atol=1e-12)
        tr0 = np.array([o["truth"] for o in o0])
        tr1 = np.array([o["truth"] for o in o1])
        assert np.allclose(tr1[:, 3], 0.8 * tr0[:, 3]) and np.allclose(
            tr1[:, 5], 0.8 * tr0[:, 5]) and np.array_equal(tr1[:, 7], tr0[:, 7])
        assert all(bits_equal(a["state"], b["state"]) for a, b in zip(i0, i1))
        imu0 = np.array([o["imu"] for o in o0])
        imu1 = np.array([o["imu"] for o in o1])
        assert not np.array_equal(imu0, imu1), "the sensors see the scaled motion"
        return "states, a_cg and thrust identical; acc_obs x1.3; truth " \
               "heave / pitch x0.8; info['state'] untouched; IMU differs"
    C.run("output scales (acc_scale, motion_scale) change the observation "
          "and acc_obs only, never the physics or info['state']", outputs_only)

    def short_waves():
        e0 = UsvEnv(frozen.scenario(duration=1.0, seed=2))
        e0.reset(seed=2)
        e1 = UsvEnv(frozen.scenario(duration=1.0, seed=2))
        e1.reset(seed=2, options={"randomize": {"short_wave": 0.6}})
        lam = 2 * math.pi / np.asarray(e0.sea.k)
        short = lam < 1.5 * e0.L
        a0, a1 = np.asarray(e0.sea.a), np.asarray(e1.sea.a)
        assert short.any() and (~short).any()
        assert np.allclose(a1[short], 0.6 * a0[short]) and np.array_equal(
            a1[~short], a0[~short])
        assert e1.n_short_scaled == int(short.sum())
        assert np.array_equal(e0.sea.phi, e1.sea.phi)
        return f"{int(short.sum())} of {len(lam)} components scaled " \
               f"(wavelength < 1.5 L = {1.5 * e0.L:.2f} m)"
    C.run("short_wave scales exactly the components with wavelength < 1.5 L",
          short_waves)

    def sea_draw():
        e = UsvEnv(frozen.scenario(duration=1.0, seed=2))
        e.reset(seed=2, options={"randomize": {"hs": 1.1, "tp": 0.8,
                                               "direction_deg": 10.0,
                                               "gamma": 2.0}})
        sp = e.header()["randomization_applied"]["sea"]
        assert abs(sp["hs"] - 1.1) < 1e-12 and abs(sp["tp"] - 4.0) < 1e-12
        assert abs(sp["theta0"] - (math.pi + math.radians(10))) < 1e-12
        assert sp["gamma"] == 2.0 and abs(e.sea.hs - 1.1) < 1e-12
        assert e.header()["episode_hull"] == frozen.MODEL   # no rebuild
        return sp
    C.run("sea draw: Hs, Tp factors, direction shift, gamma reach the "
          "SeaState; no plant rebuild without hull fields", sea_draw)

    def config_forms():
        r = randomize.Randomization.of({"seed": 3, "mass": [1380, 1450],
                                        "rho": False, "hs": {"lo": 0.95,
                                                             "hi": 1.05}})
        assert r.mass == randomize.Param(1380.0, 1450.0) and not r.rho.on
        assert r.hs.lo == 0.95 and r.enabled().count("rho") == 0
        d_all = randomize.Randomization(seed=3).draw(42)
        d_off = randomize.Randomization(seed=3).without("mass", "hs").draw(42)
        assert "mass" not in d_off and all(d_off[k] == d_all[k] for k in d_off)
        only = randomize.Randomization(seed=3).only("thrust_gain")
        assert only.enabled() == ["thrust_gain"] and not only.rebuilds_plant()
        assert randomize.Randomization(seed=3).rebuilds_plant()
        tr = randomize.Randomization(seed=3, stream="train").draw(5)
        ev = randomize.Randomization(seed=3, stream="eval").draw(5)
        assert tr != ev
        bad = 0
        for x in ({"massive": [1, 2]}, {"mass": [2, 1]}, {"stream": "x"},
                  {"seed": -1}, {"rho": [0, 1000]}):
            try:
                randomize.Randomization.of(x)
            except (KeyError, ValueError):
                bad += 1
        assert bad == 5
        for x in ({"nope": 1.0}, {"mass": float("nan")}, {"mass": True}):
            try:
                randomize.check_draw(x)
            except (KeyError, ValueError, TypeError):
                bad += 1
        assert bad == 8
        rz2 = randomize.Randomization.of(r.to_dict())
        assert rz2 == r
        return f"{len(randomize.PARAMS)} parameters; flags do not shift " \
               f"the stream; train != eval; 8 refusals; to_dict round trip"
    C.run("config: dict forms, without / only, stable draws under flags, "
          "train vs eval streams, refusals", config_forms)

    def seeds():
        e = UsvEnv(frozen.scenario(duration=1.0, seed=0),
                   randomization=randomize.Randomization(seed=1))
        e.reset(seed=0)
        drawn = [e.episode_seed() for _ in range(3000)]
        assert not any(s in EVAL_SEEDS for s in drawn)
        e2 = UsvEnv(frozen.scenario(duration=1.0, seed=0,
                                    randomization={"seed": 1,
                                                   "stream": "eval"}))
        assert e2.rz.stream == "eval"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            e.reset(seed=EVAL_SEEDS[2], options={"randomize": True})
        assert any("evaluation seed" in str(x.message) for x in w)
        return f"3000 training draws avoid EVAL_SEEDS {list(EVAL_SEEDS)}; " \
               f"the scenario's usv.randomization is read; a train reset on " \
               f"an evaluation seed warns"
    C.run("seeds: training draws never return an evaluation seed; the "
          "scenario carries the config; warning on a train/eval clash", seeds)

    def refusals():
        e = UsvEnv(frozen.scenario(duration=1.0, seed=0))
        n = 0
        for opt in ({"randomize": {"mass": "x"}}, {"randomise": True},
                    {"randomize": {"nope": 1.0}}):
            try:
                e.reset(seed=0, options=opt)
            except (KeyError, ValueError, TypeError):
                n += 1
        assert n == 3
        return "3 refusals before anything is stepped"
    C.run("reset options refused: a non-numeric draw, an unknown option, an "
          "unknown parameter", refusals)
    return C.result(rows=rows)


# ============================================================== hull_every
def g_hull_every():
    """Randomization.hull_every and the plant cache (PRODUCT_LAYER section
    9.8; app/randomize.py HULL_EVERY, PLANT CACHE)."""
    C = Checks("hull_every")
    from app import frozen, plantstate, randomize
    from app.env import UsvEnv
    from app.record import replay

    def lru():
        c = randomize.PlantCache(2)
        c.put(("a",), 1)
        c.put(("b",), 2)
        assert c.get(("a",)) == 1                 # a is the most recent now
        c.put(("c",), 3)                          # evicts b
        assert c.get(("b",)) is None and len(c) == 2
        assert set(c.keys()) == {("a",), ("c",)} and ("c",) in c
        c.resize(1)
        assert len(c) == 1 and c.get(("c",)) == 3 and c.get(("a",)) is None
        assert (c.hits, c.misses) == (2, 2)
        off = randomize.PlantCache(0)
        off.put(("a",), 1)
        assert off.get(("a",)) is None and len(off) == 0
        r = randomize.Randomization.of({"seed": 1, "hull_every": 3,
                                        "plant_cache": 2})
        assert r.hull_every == 3 and r.plant_cache == 2
        assert randomize.Randomization.of(r.to_dict()) == r
        assert r.with_(hull_every=5).hull_every == 5
        assert randomize.Randomization().hull_every == 1
        assert randomize.Randomization().plant_cache == 8
        bad = 0
        for x in ({"hull_every": 0}, {"plant_cache": -1},
                  {"hull_every": 1.5}, {"hull_every": True}):
            try:
                randomize.Randomization.of(x)
            except (ValueError, TypeError):
                bad += 1
        assert bad == 4
        k = randomize.hull_key({"mass": 1400.0, "hs": 1.0, "lcg": 1.01})
        assert k == (("mass", 1400.0), ("lcg", 1.01))
        return ("eviction order, resize, hits / misses counted, size 0 = "
                "off; hull_every and plant_cache in the config (dict form, "
                "to_dict round trip, with_); 4 refusals; hull_key is the "
                "hull fields in PARAMS order")
    C.run("the plant cache is an LRU; the config carries hull_every and "
          "plant_cache; refusals; hull_key", lru)

    # ------------------------------------ hull_every = 3, six episodes
    rz = randomize.Randomization(seed=11, hull_every=3)
    env = UsvEnv(frozen.scenario(duration=2.0, seed=300), randomization=rz)
    seeds = [300, 301, 302, 303, 304, 305]
    recs, infos, costs = [], [], []
    for s in seeds:
        t0 = time.perf_counter()
        obs, info = env.reset(seed=s, options={"randomize": True})
        costs.append(time.perf_counter() - t0)
        _run(env, [0.7, 0.0], n_max=4)
        recs.append(env.record())
        infos.append(info)

    def every():
        for i, (s, r, info) in enumerate(zip(seeds, recs, infos)):
            own = rz.draw(s)
            src_seed = seeds[(i // 3) * 3]
            hull_src = randomize.hull_fields(rz.draw(src_seed))
            d, hf = r.header["randomization"], r.header[
                "randomization_hull_from"]
            assert info["randomization_hull_from"] == hf, (i, hf)
            assert hf == dict(seed=src_seed, episode=(i // 3) * 3,
                              this_episode=i, hull_every=3), (i, hf)
            assert {k: d[k] for k in randomize.HULL} == hull_src, i
            assert all(d[k] == own[k] for k in own
                       if k not in randomize.HULL), i
            assert list(d) == list(own), i              # key order kept
            assert r.header["randomization_applied"]["hull"]["mass"] == \
                hull_src["mass"]
            assert r.header["plant_cache"]["source"] == (
                "built" if i % 3 == 0 else "cache"), (i, r.header[
                    "plant_cache"])
            assert r.header["plant_cache"]["max"] == 8
            assert np.all(np.isfinite(r.plant["s"]))
            assert r.header["episode_hull"].endswith("~rz@as1")
        st = [r.header["physics"]["physics_sha1"] for r in recs]
        assert st[0] == st[1] == st[2] and st[3] == st[4] == st[5]
        assert st[0] != st[3]
        return (f"episodes 0-2 hull of seed 300 (stamp {st[0][:8]}), 3-5 of "
                f"seed 303 ({st[3][:8]}); the 8 other values each episode's "
                f"own; header and info 'randomization_hull_from' agree")
    C.run("hull_every=3: episodes 0-2 run the hull drawn for episode 0, 3-5 "
          "the one drawn for episode 3; the sea, gain and output values are "
          "each episode's own (the stream does not shift); header and info "
          "say whose hull draw is in force", every)

    def rebuilt():
        b = [r.header["plant_build_s"] for r in recs]
        assert b[0] > 0.5 and b[3] > 0.5, b
        assert all(x < 0.25 for x in (b[1], b[2], b[4], b[5])), b
        assert env._plants.misses == 2 and env._plants.hits == 4
        return (f"reset wall s {[round(c, 3) for c in costs]}; "
                f"plant_build_s {[round(x, 3) for x in b]}; mean reset over "
                f"the six {np.mean(costs):.2f} s")
    C.run("the plant is built on episodes 0 and 3 only; the other four take "
          "a fresh plant from the cached prototype in milliseconds", rebuilt)

    def skipped_replays():
        r = recs[4]
        back = replay(r)
        assert back.header["randomization"] == r.header["randomization"]
        assert back.fingerprint(physical=True) == r.fingerprint(
            physical=True), r.diff(back, physical=True)[:3]
        return (f"episode 4 (hull of seed 303, sea of seed 304): physical "
                f"fingerprint {r.fingerprint(physical=True)[:16]}")
    C.run("a record of an episode running an earlier episode's hull replays "
          "bit for bit (its draw carries the hull values in force)",
          skipped_replays)

    def pristine():
        key = randomize.hull_key(recs[-1].header["randomization"])
        e = env._plants.get(key)
        proto = e["plant"]
        aux = plantstate.aux_row(proto)
        assert proto is not env.plant
        assert all(np.isnan(x) for x in aux[len(plantstate.SCALARS):])
        assert proto.slam_count == 0 and proto.hull is env.plant.hull
        return ("the cached prototype is not the episode's plant, its delay "
                "lines were never primed, slam count 0; the episode's plant "
                "shares the hull object (the physics' caches key on it)")
    C.run("the cached prototype is pristine (never stepped); the episode's "
          "plant is another object on the same hull", pristine)
    env.close()

    # ------------------------------------------------ cache isolation
    rz1 = randomize.Randomization(seed=11)
    dA = rz1.draw(400)
    dB = {**rz1.draw(401), **randomize.hull_fields(dA)}   # A's hull, B's sea
    e = UsvEnv(frozen.scenario(duration=3.0, seed=400), randomization=rz1)
    e.reset(seed=400, options={"randomize": dA})
    _run(e, [0.75, 0.1])
    rA = e.record()

    def isolation():
        e.reset(seed=401, options={"randomize": dB})
        _run(e, [0.6, -0.2])
        rB = e.record()
        e.reset(seed=400, options={"randomize": dA})
        _run(e, [0.75, 0.1])
        rA2 = e.record()
        assert rA.header["plant_cache"]["source"] == "built"
        assert rB.header["plant_cache"]["source"] == "cache"
        assert rA2.header["plant_cache"]["source"] == "cache"
        assert rA.fingerprint() == rA2.fingerprint(), rA.diff(rA2)[:3]
        f = UsvEnv(frozen.scenario(duration=3.0, seed=400),
                   randomization=rz1)
        f.reset(seed=401, options={"randomize": dB})
        _run(f, [0.6, -0.2])
        rB_ref = f.record()
        assert rB_ref.header["plant_cache"]["source"] == "built"
        assert rB.fingerprint() == rB_ref.fingerprint(), \
            rB.diff(rB_ref)[:3]
        assert rB.header["episode_hull_sha1"] == rA.header[
            "episode_hull_sha1"] and rB.header["sea_seed"] != rA.header[
                "sea_seed"]
        return (f"A (seed 400) built, B (A's hull, seed 401's sea) and A "
                f"again from the cache: B == a fresh env's B "
                f"({rB.fingerprint()[:16]}), A again == A "
                f"({rA.fingerprint()[:16]}); {len(rA.plant['t'])} plant "
                f"rows each")
    C.run("cache isolation: two episodes on one cached hull with different "
          "seas reproduce a fresh env's single-episode results bit for "
          "bit; the first draw again reproduces the first episode",
          isolation)

    def cache_off():
        g = UsvEnv(frozen.scenario(duration=3.0, seed=400),
                   randomization=rz1.with_(plant_cache=0))
        g.reset(seed=400, options={"randomize": dA})
        _run(g, [0.75, 0.1])
        r1 = g.record()
        t0 = time.perf_counter()
        g.reset(seed=400, options={"randomize": dA})
        t_again = time.perf_counter() - t0
        _run(g, [0.75, 0.1])
        r2 = g.record()
        assert r1.header["plant_cache"] == dict(source="built", n=0, max=0)
        assert r2.header["plant_cache"]["source"] == "built"
        assert r1.fingerprint() == rA.fingerprint(), r1.diff(rA)[:3]
        assert r2.fingerprint() == rA.fingerprint(), r2.diff(rA)[:3]
        return (f"plant_cache=0: both episodes built anew (the second in "
                f"{t_again:.2f} s: the righting-arm table again, the steady "
                f"attitudes from the physics' manifest-keyed cache) and "
                f"both equal the cached run {rA.fingerprint()[:16]}")
    C.run("plant_cache=0 turns the cache off; an episode built anew gives "
          "the same bits as one from the cache (the cache changes no "
          "result)", cache_off)
    e.close()

    # ------------------------------------------------------- reset costs
    def reset_costs():
        rz2 = randomize.Randomization(seed=12, hull_every=10)
        g = UsvEnv(frozen.scenario(duration=1.0, seed=500),
                   randomization=rz2)
        g.reset(seed=499, options={"randomize": False})  # the nominal
        t = []
        for s in range(500, 510):
            t0 = time.perf_counter()
            g.reset(seed=s, options={"randomize": True})
            t.append(time.perf_counter() - t0)
        assert g.hull_from["seed"] == 500 and g._plants.hits == 9
        sea_only = {k: v for k, v in rz2.draw(600).items()
                    if k not in randomize.HULL}
        t0 = time.perf_counter()
        g.reset(seed=600, options={"randomize": sea_only})
        t_sea = time.perf_counter() - t0
        assert g.plant_source == "nominal" and g.hull_from is None
        t0 = time.perf_counter()
        g.reset(seed=601, options={"randomize": False})
        t_off = time.perf_counter() - t0
        out = dict(miss_s=round(t[0], 3), hit_mean_ms=round(
            1e3 * float(np.mean(t[1:])), 1), hit_max_ms=round(
            1e3 * max(t[1:]), 1), hull_every_10_amortised_s=round(
            float(np.mean(t)), 3), sea_only_ms=round(1e3 * t_sea, 1),
            off_ms=round(1e3 * t_off, 1))
        C.costs = out
        return out
    C.run("reset cost: a hull draw with a cache miss, the nine hits after "
          "it (hull_every=10), the amortised mean, a sea-only draw, "
          "randomisation off", reset_costs)
    return C.result(costs=getattr(C, "costs", None))


# =============================================================== reference
def g_reference(full=False):
    C = Checks("reference")
    from app import accel, frozen
    from app.env import UsvEnv
    out = {}
    for dt in ((0.01, 0.005) if full else (0.01,)):
        sc = frozen.scenario(hs=1.0, tp=5.0, theta0=math.pi, heading=0.0,
                             duration=60.0, seed=0, speed=20 * KN, dt=dt)
        env = UsvEnv(sc)
        env.reset(seed=0)
        obs_l, info_l, wall = _run(env, None, hold_kn=20.0)
        rec = env.record()
        n = len(rec.plant["t"])
        u = rec.plant["s"][:, 6] / KN
        w = int(0.1 * n)
        st = accel.record_stats(rec)
        pk_cg = accel.excursion_peaks(accel.drop_start(
            accel.filtered_series(rec)[1][:, 0]))
        res = dict(dt=dt, sub=env.sub, n_plant_steps=n,
                   ms_per_plant_step=wall / n * 1e3,
                   t_end=float(rec.header["t_end"]), reason=rec.header["reason"],
                   u_mean_kn=float(u[w:].mean()), u_min_kn=float(u[w:].min()),
                   u_max_kn=float(u[w:].max()),
                   stats={k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in st.items()},
                   cg_peaks_over_1g_per_hour=accel.exceedance_per_hour(
                       pk_cg, 1.0, 0.9 * float(rec.header["t_end"])),
                   stamp=rec.header["physics"]["physics_sha1"][:12])
        out[str(dt)] = res
        C.ok(f"20 kn head seas Hs 1.0 m Tp 5 s, 60 s at dt {dt}: finite, "
             f"speed held near 20 kn",
             rec.header["finite"] and rec.header["reason"] == "duration"
             and abs(res["u_mean_kn"] - 20.0) < 2.0,
             f"u mean {res['u_mean_kn']:.2f} kn ({res['u_min_kn']:.1f}-"
             f"{res['u_max_kn']:.1f}); {res['ms_per_plant_step']:.2f} ms/plant "
             f"step; filtered CG mean of positive peaks "
             f"{st['cg_mean']:.3f} g, A1/10 {st['cg_a110']:.3f} g, max "
             f"{st['cg_max']:.3f} g, {st['cg_n']} peaks; bow mean "
             f"{st['bow_mean']:.3f} g, A1/10 {st['bow_a110']:.3f} g, max "
             f"{st['bow_max']:.3f} g; raw CG mean {st['raw_cg_mean']:.3f} g, "
             f"bow {st['raw_bow_mean']:.3f} g; CG peaks over 1 g per hour "
             f"{res['cg_peaks_over_1g_per_hour']:.0f}")
        env.close()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "reference_20kn_head_ss3.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    return C.result(reference=out)


GROUPS = dict(frozen=g_frozen, randomization=g_randomization,
              hull_every=g_hull_every, reference=g_reference)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=None)
    ap.add_argument("--full", action="store_true",
                    help="also the 0.005 s reference run")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    names = a.groups.split(",") if a.groups else list(GROUPS)
    t0 = time.time()
    out = []
    for g in names:
        fn = GROUPS[g]
        try:
            r = fn(a.full) if g == "reference" else fn()
        except Exception as ex:                      # noqa: BLE001
            import traceback
            r = dict(group=g, ok=False, n=1, seconds=0.0, checks=[dict(
                check="group crashed", ok=False,
                detail=f"{type(ex).__name__}: {ex}\n"
                       f"{traceback.format_exc()[-2000:]}")])
        out.append(r)
    wall = time.time() - t0
    n_ok = sum(c["ok"] for r in out for c in r["checks"])
    n_all = sum(len(r["checks"]) for r in out)
    print(f"\n=== randomization tests: {n_ok}/{n_all} checks pass, "
          f"{wall:.0f} s wall ===")
    for r in out:
        print(f"\n[{'PASS' if r['ok'] else 'FAIL'}] {r['group']} "
              f"({r['seconds']:.0f} s)")
        for c in r["checks"]:
            d = c["detail"].replace("\n", " ")
            print(f"   {'ok ' if c['ok'] else 'BAD'} {c['check']}"
                  + (f" [{c['s']:g} s]" if c.get("s") else "")
                  + (f" -- {d[:600]}" if d else ""))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "randomization_results.json"), "w") as f:
        json.dump(dict(wall_s=wall, n_ok=n_ok, n=n_all, groups=out,
                       when=time.strftime("%Y-%m-%d %H:%M:%S")),
                  f, indent=1, default=str)
    return 0 if n_ok == n_all else 1


if __name__ == "__main__":
    sys.exit(main())
