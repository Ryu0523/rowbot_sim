#!/usr/bin/env python3
"""
Tests of the product layer (learn/meta/PRODUCT_LAYER.md section 1, checks
C-P1a ... C-P7), in independent groups run in parallel processes sized by
free RAM (studies/parallel_runner.py; ACCEPTANCE T1-T4).

    python -m studies.app_test                 quick tier (default)
    python -m studies.app_test --tier full     + longer m0 run through the
                                               runner, a second e2e seed
    python -m studies.app_test --groups api,sensors

Groups (plant-only unless marked; closed-loop MPC only where the controller
interface is what is tested: the equivalence groups and the end-to-end
demos, short missions):
  api         Gymnasium check_env; spaces; reset determinism; actions;
              hull and physics selection by key; scenario files; geodesy
  equiv       app env + hand MPC == learn/repro/task.Mission (lin plant) and
              studies/final_eval.make_mission (gz plant), bit for bit
  replay      save / load fingerprint; command replay == original (physical
              fingerprint); det_a / det_b: the same run in two more
              processes gives the same fingerprint
  sensors     noise off == truth; noise statistics (chi-square, binomial
              bands at 99.9 %); latency; bias, random walk, scale; seeding
  interfaces  websocket / UDP / HTTP sessions == direct env bit for bit;
              server-sent events of a live run; origin and path checks;
              message shapes == LOTUSim's .msg field lists; ROS 2 bridge
              imports without rclpy
  e2e:<file>  a demo scenario end to end on its hull: run, record, viewer
              JSON, Blender scene, command replay
  robust      the failure modes of the 2026-10-06 review: path traversal
              (backslash, encoded, drive), concurrent HTTP sessions, NaN /
              inf actions on every interface, a policy failing mid-run
              (partial record, live run saved), LOTUSim waypoint syntax,
              UDP retries (seq), sensor / scenario / policy validation at
              load, R1 warnings, Blender export windows, strict JSON,
              websocket protocol errors, live-run limits (429), record
              staleness, reset() seeding, sensor-driven policies, hull
              listing (plant-only; one 10-step hand run with sensors)
  resume      the reset / replay contract (2026-10-06, P1c): a record of
              scarab195@as1 + wagner_entry (test switches; its hidden
              state) resumed from any plant row and any control step ==
              the record bit for bit; the plant's own trajectory()['hidden']
              agrees; negative controls (no hidden state, no delay lines);
              the completeness audit (callables and closures included,
              with a self-test); the exact codec; v0 closed loop resumed
              with the same scripted policy == the original, every array;
              refusals
  resume_more the as1 preset + all four P1b switches resumed and audited;
              safe mode's state restored on resume (lowered thresholds)
  codehash    every repository module an as1 plant with all switches
              loads or runs is in its code hash (fresh process)
  torch       (imports torch) SafeMode copy == mpc_constrained.SafeMode;
              hand and m0 through the app == studies/final_eval.run_mission
              (metrics equal to the last bit); m0 refuses another hull;
              a learned row (cat_p_prior) runs 2 control steps

Results: studies/_cache/app/test_results.json; test records in
studies/_cache/app/test_runs/.
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np   # noqa: E402

OUT = os.path.join(ROOT, "studies", "_cache", "app")
TEST_RUNS = os.path.join(OUT, "test_runs")
LOTUS_MSG = os.path.join(os.path.dirname(ROOT), "ext_sims", "LOTUSim",
                         "interfaces")


class Checks:
    def __init__(self, name):
        self.name, self.items, self.t0 = name, [], time.time()

    def ok(self, what, cond, detail=""):
        self.items.append(dict(check=what, ok=bool(cond), detail=str(detail)))
        return bool(cond)

    def run(self, what, fn):
        """A check whose failure is an exception (its wall time kept)."""
        t0 = time.time()
        try:
            d = fn()
            self.items.append(dict(check=what, ok=True,
                                   detail="" if d is None else str(d),
                                   s=round(time.time() - t0, 1)))
        except Exception as ex:                    # noqa: BLE001
            self.items.append(dict(check=what, ok=False,
                                   detail=f"{type(ex).__name__}: {ex}\n"
                                   + traceback.format_exc()[-1500:],
                                   s=round(time.time() - t0, 1)))

    def result(self, **extra):
        return dict(group=self.name, ok=all(i["ok"] for i in self.items),
                    n=len(self.items), seconds=time.time() - self.t0,
                    checks=self.items, **extra)


def bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def chi2_band(n, p=0.999):
    from scipy.stats import chi2
    a = (1 - p) / 2
    return chi2.ppf(a, n) / n, chi2.ppf(1 - a, n) / n


# ===================================================================== api
def g_api():
    C = Checks("api")
    from app import geo
    from app.env import UsvEnv
    from app.scenario import Scenario, ScenarioError, load, save, straight

    def gym_check():
        from gymnasium.utils.env_checker import check_env
        env = UsvEnv(straight("scarab195", duration=2.4, seed=1))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_env(env, skip_render_check=True)
        env2 = UsvEnv(straight("scarab195", duration=2.4, seed=1,
                               obs_truth=True), action_mode="heading")
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            check_env(env2, skip_render_check=True)
        msgs = sorted({str(x.message)[:90] for x in w + w2})
        return f"{len(msgs)} distinct warnings: {msgs}"
    C.run("C-P1a gymnasium check_env (direct and heading mode, obs_truth)",
          gym_check)

    env = UsvEnv(straight("scarab195", duration=4.8, seed=1))
    o1, i1 = env.reset(seed=7)
    s1 = env.s.copy()
    o2, i2 = env.reset(seed=7)
    C.ok("reset(seed) twice: same obs and state, bit for bit",
         all(bits_equal(o1[k], o2[k]) for k in o1) and bits_equal(s1, env.s))
    phi7 = np.asarray(env.sea.phi).copy()
    env.reset(seed=8)
    C.ok("another seed: another sea", not bits_equal(phi7, env.sea.phi))
    C.ok("obs in observation_space", env.observation_space.contains(o1))
    env.reset(seed=7)
    n, term, trunc = 0, False, False
    while not (term or trunc):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        n += 1
    C.ok("an episode truncates after n_ctrl steps", trunc and not term
         and n == env.n_ctrl == 20, f"n {n}, n_ctrl {env.n_ctrl}")
    C.ok("info carries t, state, pose physics stamp, reason",
         {"t", "state", "leg", "cost_parts", "physics", "reason"} <=
         set(info) and info["reason"] == "duration")

    def after_end():
        try:
            env.step([0.5, 0.0])
        except RuntimeError as ex:
            return f"raises: {ex}"
        raise AssertionError("step after the end did not raise")
    C.run("step after the end raises", after_end)

    env.reset(seed=7)
    _, r1, *_ = env.step({"thrust": 0.5 * env.t_max, "nozzle_rad": 0.0})
    s_a = env.s.copy()
    env.reset(seed=7)
    _, r2, *_ = env.step([0.5, 0.0])
    C.ok("dict action (N, rad) == array action (fractions)",
         bits_equal(s_a, env.s))
    env.reset(seed=7)
    env.step([5.0, -3.0])
    C.ok("array actions are clipped to the action space",
         env.last_cmd["thrust"] == env.t_max and
         env.last_cmd["nozzle"] == -env.rud_max)

    def bad_key():
        try:
            env.reset(seed=7)
            env.step({"thrust": 1.0, "rudder": 0.1})
        except ValueError as ex:
            return str(ex)
        raise AssertionError("unknown action key accepted")
    C.run("unknown action key raises", bad_key)

    eh = UsvEnv(straight("scarab195", duration=2.4, seed=1),
                action_mode="heading")
    eh.reset()
    for _ in range(5):
        eh.step([0.7, 0.2])
    C.ok("heading mode: the autopilot steers toward the heading",
         eh.autopilot is not None and eh.last_cmd["nozzle"] != 0.0,
         f"nozzle {eh.last_cmd['nozzle']:.4f} rad")

    # hull and physics selection by key
    from sim import physics
    table = [("scarab195", "gz", "PlaningVesselGZ", "v0"),
             ("scarab195", "lin", "PlaningVessel", "v0"),
             ("scarab195@as1", "gz", "PlaningVesselAS", "as1"),
             ("scarab195@as1", "lin", "PlaningVesselASLin", "as1"),
             ("probe_hullB", "gz", "PlaningVesselGZ", "v0"),
             ("probe_hullB@as1", "gz", "PlaningVesselAS", "as1")]
    for key, kind, cls, ver in table:
        def sel(key=key, kind=kind, cls=cls, ver=ver):
            e = UsvEnv(straight(key, plant=kind, duration=0.48))
            e.reset()
            got = (type(e.plant).__name__, physics.verify(e.plant, e.hull),
                   e.hull.name)
            assert got == (cls, ver, key), got
            e.step([0.6, 0.0])
            assert np.all(np.isfinite(e.s))
            return f"{got}, stamp {physics.stamp(e.plant)['physics_sha1'][:10]}"
        C.run(f"C-P1b {key} plant {kind} -> {cls} ({ver})", sel)

    def unknown():
        try:
            UsvEnv(straight("no_such_boat"))
        except KeyError as ex:
            return str(ex)[:80]
        raise AssertionError("unknown hull accepted")
    C.run("unknown hull key raises KeyError", unknown)

    def not_planing():
        try:
            UsvEnv(straight("wigley10"))
        except NotImplementedError as ex:
            return str(ex)[:80]
        raise AssertionError("displacement hull accepted")
    C.run("a displacement hull is refused clearly", not_planing)

    # scenario files
    def roundtrip():
        import tempfile
        out = []
        for f in ("demo_scarab195.json", "demo_hullB.json",
                  "demo_scarab195_as1.json", "eval_head_seas.json"):
            a = load(f)
            p = os.path.join(tempfile.gettempdir(), "app_test_" + f)
            save(a, p)
            b = load(p)
            assert a == b and a.sha256() == b.sha256(), f
            out.append(f"{f}: {a.hull_key}, {a.track_spec()['kind']}")
        return "; ".join(out)
    C.run("C-P3 load -> save -> load is identity (all scenario files)",
          roundtrip)
    sc = load("demo_scarab195.json")
    C.ok("wave_direction 90 (from the east) -> theta0 == pi exactly",
         sc.theta0() == math.pi, sc.theta0())
    C.ok("agent heading 45 deg -> pi/4 exactly",
         geo.deg_to_rad_wrapped(45.0) == math.pi / 4)
    C.ok("LOTUSim keys present in the demo scenario",
         {"name", "reference_position", "environment", "agents"} <=
         set(sc.raw) and {"model", "position", "heading",
                          "waypoint_interface"} <= set(sc.raw_agent))
    bad = {"current": dict(environment=dict(ocean=dict(current_speed=0.5)),
                           agents=dict(a=dict())),
           "wind": dict(environment=dict(atmosphere=dict(wind_speed=3.0)),
                        agents=dict(a=dict())),
           "two agents": dict(agents=dict(a=dict(), b=dict())),
           "two wave directions": dict(environment=dict(ocean=dict(
               wave_direction=90.0, theta0=3.0)), agents=dict(a=dict())),
           "unknown key": dict(agents=dict(a=dict(colour="red")))}
    for what, doc in bad.items():
        def refuse(doc=doc):
            try:
                Scenario(doc)
            except ScenarioError as ex:
                return str(ex)[:70]
            raise AssertionError("accepted")
        C.run(f"scenario refused: {what}", refuse)

    # geodesy
    fr = geo.LocalFrame(1.2605794, 103.7516212, 0.0)
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(200):
        e, n_, u = rng.uniform(-10e3, 10e3, 2).tolist() + [
            float(rng.uniform(-5, 50))]
        lat, lon, h = fr.to_geodetic(e, n_, u)
        err = max(err, float(np.max(np.abs(np.array(fr.to_enu(lat, lon, h))
                                           - (e, n_, u)))))
    C.ok("ENU -> geodetic -> ENU round trip over +-10 km < 1e-6 m", err < 1e-6,
         f"{err:.2e} m")
    p0 = geo.geodetic_to_ecef(0.0, 0.0, 0.0)
    p9 = geo.geodetic_to_ecef(90.0, 0.0, 0.0)
    C.ok("ECEF of (0,0,0) = (a,0,0) and of the pole = (0,0,b)",
         np.allclose(p0, [geo.A_WGS84, 0, 0], atol=1e-6)
         and np.allclose(p9, [0, 0, geo.B_WGS84], atol=1e-6))
    lat1, _, _ = fr.to_geodetic(0.0, 1000.0, 0.0)
    M = geo.A_WGS84 * (1 - geo.E2) / (1 - geo.E2 * math.sin(math.radians(
        1.2605794)) ** 2) ** 1.5
    C.ok("1 km north = 1000 / M (meridian radius) in latitude, 1e-9 deg",
         abs((lat1 - 1.2605794) - math.degrees(1000.0 / M)) < 1e-9,
         f"{(lat1 - 1.2605794) - math.degrees(1000.0 / M):.1e} deg")
    q_err = 0.0
    for _ in range(100):
        r, p, y = rng.uniform(-3, 3, 3)
        q_err = max(q_err, float(np.abs(geo.quat_to_rot(geo.quat_zyx(r, p, y))
                                        - geo.rot_zyx(r, p, y)).max()))
    C.ok("quaternion == rotation matrix (ZYX, FLU -> ENU) to 1e-12",
         q_err < 1e-12, f"{q_err:.1e}")
    C.ok("yaw 0 (east) = bearing 90; yaw pi/2 (north) = bearing 0",
         geo.yaw_to_bearing_deg(0.0) == 90.0 and
         abs(geo.yaw_to_bearing_deg(math.pi / 2)) % 360 < 1e-12)
    R = geo.rot_zyx(0.0, 0.1, 0.0)
    C.ok("positive pitch puts the bow down (FLU, as the plant's SP = -1)",
         (R @ [1, 0, 0])[2] < 0)
    return C.result()


# ===================================================================== equiv
def g_equiv():
    C = Checks("equiv")
    from app.env import UsvEnv
    from app.policies import HandMPCPolicy
    from app.scenario import straight
    from learn.repro import task
    from studies import final_eval as FE
    from studies.sim2real_jet import theta, z_hand
    N = 40
    for plant in ("lin", "gz"):
        for seed, leg in ((0, 0), (3, 1)):
            def one(plant=plant, seed=seed, leg=leg):
                w, floor = theta(z_hand())
                if plant == "lin":
                    m = task.Mission("high", seed, leg, t_end=N * 0.25,
                                     weights=w, floor=floor,
                                     track=task.LEGS[leg])
                else:
                    FE.OPT["plant"] = "gz"
                    m = FE.make_mission(seed, leg, N * 0.25, weights=w,
                                        floor=floor)
                S1, C1 = [m.s.copy()], []
                while not m.done():
                    thr, rud = m.mpc_command()
                    C1.append((thr, rud))
                    m.advance(thr, rud)
                    S1.append(m.s.copy())
                sc = straight("scarab195", hs=task.HS, tp=task.TP,
                              theta0=task.WAVES_FROM, heading=task.LEGS[leg],
                              duration=N * 0.24, seed=seed, plant=plant)
                env = UsvEnv(sc, action_mode="heading")
                obs, info = env.reset()
                pol = HandMPCPolicy(guidance="hold")
                pol.reset(env)
                S2, C2 = [env.s.copy()], []
                while True:
                    obs, r, te, tr, info = env.step(pol.act(obs, info))
                    C2.append((env.last_cmd["thrust"], env.last_cmd["nozzle"]))
                    S2.append(env.s.copy())
                    if te or tr:
                        break
                ok = bits_equal(S1, S2) and bits_equal(C1, C2)
                assert ok, f"max |ds| {np.abs(np.array(S1) - np.array(S2)).max()}"
                return (f"{len(S2) - 1} control steps, states and commands "
                        f"identical; plant {type(env.plant).__name__}")
            C.run(f"C-P1c hand MPC == {'Mission' if plant == 'lin' else 'final_eval.make_mission'} "
                  f"({plant}), seed {seed} leg {leg}", one)
    return C.result()


# ===================================================================== replay
def _demo_short(duration=12.0):
    from app.runner import run
    return run("demo_scarab195.json", duration=duration)


def g_det(tag):
    """The same short demo run in its own process: its fingerprint."""
    rec = _demo_short()
    return dict(group=tag, ok=True, n=0, seconds=0.0, checks=[],
                fingerprint=rec.fingerprint())


def g_replay():
    C = Checks("replay")
    from app.record import Record, replay
    rec = _demo_short()
    fp = rec.fingerprint()
    os.makedirs(TEST_RUNS, exist_ok=True)
    path = rec.save(os.path.join(TEST_RUNS, "replay_test.npz"))
    back = Record.load(path)
    C.ok("C-P4b(iii) save -> load preserves the fingerprint",
         back.fingerprint() == fp, fp[:16])
    hd = dict(back.header)
    hd.pop("fingerprint", None)
    C.ok("save -> load preserves the header",
         json.dumps(hd, sort_keys=True, default=str) ==
         json.dumps(rec.header, sort_keys=True, default=str))
    C.ok("C-P4a the record holds trajectory, sea, commands, sensors, stamp",
         rec.plant["s"].shape[1] == 14 and len(rec.header["sea"]["phi"]) > 0
         and len(rec.ctrl["thrust"]) == len(rec.ctrl["t"])
         and set(rec.sensors) >= {"imu", "gnss", "heading", "speed_log",
                                  "wave_probe", "preview"}
         and rec.header["physics"]["physics"] == "v0"
         and rec.header["git"].get("commit"),
         f"{len(rec.plant['t'])} plant steps, {len(rec.ctrl['t'])} control "
         f"steps, {len(rec.header['sea']['a'])} sea components, commit "
         f"{(rec.header['git'].get('commit') or '')[:10]}")
    t0 = time.time()
    rep = replay(rec)
    same = rep.fingerprint(physical=True) == rec.fingerprint(physical=True)
    C.ok("C-P4b(ii) command replay reproduces every state and sensor sample",
         same, f"{time.time() - t0:.1f} s; differing: {rec.diff(rep)[:6]}")
    C.ok("replay reaches the same end", rep.header["reason"] ==
         rec.header["reason"] and rep.header["t_end"] == rec.header["t_end"])
    return C.result(fingerprint=fp)


# ===================================================================== resume
# the P1b switches (sim/planing_anyspeed: implemented, not in the preset)
P1B = ("fk_exact", "orbital", "wagner_entry", "lateral_phi0")


def _switches_plus(*extra):
    """The CURRENT as1 preset plus `extra`, in the plant's own order
    (sim/planing_anyspeed.resolve: SWITCHES order, no duplicates) -- so
    'the preset + wagner_entry' stays that when the preset grows."""
    from sim import physics
    from sim.planing_anyspeed import resolve
    return resolve("as1", tuple(physics.preset("as1")) + tuple(extra))


def _wagner_test():
    """A plant with a hidden state: the as1 preset plus P1b's wagner_entry
    (its flow-state field, sim/planing_anyspeed hidden_state())."""
    return _switches_plus("wagner_entry")


def _wagner_record(duration=12.0, switches=None):
    """A short run of scarab195@as1 + `switches` (default: the preset +
    wagner_entry) in the demo sea, direct actions (scripted throttle and
    nozzle schedules: the delay lines carry changing commands; no reduced
    model needed), the GZ plant recording its own trajectory
    (record_plant: trajectory()['hidden'])."""
    from app.env import UsvEnv
    from app.runner import run
    from app.scenario import load
    sc = load("demo_scarab195_as1.json").with_overrides(
        duration=duration, record_plant=True,
        policy={"type": "scripted", "throttle": [[0, 0.6], [6, 0.95],
                                                 [9, 0.5]],
                "nozzle": [[0, 0.0], [5, 0.4], [8, -0.3]]})
    env = UsvEnv(sc, action_mode="direct",
                 test_switches=switches or _wagner_test())
    rec = run(sc, env=env)
    return rec, env


def _audit_at(rec, j):
    """(original, restored, s) at plant row j after the same next step: the
    original is a fresh env's plant stepped from the reset to row j with
    the recorded commands (no copy: a deep copy's closures still point at
    the plant it was copied from), the restored one is record.plant_at."""
    from app.record import env_of, plant_at
    P, h = rec.plant, rec.header
    e = env_of(rec, record=False)
    e.resume(rec, 0, strict=True)
    p, s, t = e.plant, np.array(rec.s0), 0.0
    for i in range(j + 1):
        s = p.step(s, t, float(P["thrust"][i]), float(P["nozzle"][i]),
                   h["dt"])
        t += h["dt"]
    assert bits_equal(s, P["s"][j]), \
        f"row {j}: stepping from the reset does not reach the record"
    rest, s2, t2 = plant_at(rec, j)
    c = (float(P["thrust"][j + 1]), float(P["nozzle"][j + 1]))
    a1 = p.step(np.array(s), t, *c, h["dt"])
    a2 = rest.step(s2, t2, *c, h["dt"])
    assert bits_equal(a1, a2) and t == t2, f"row {j}: the next step differs"
    return p, rest, a1


def _same_exact(a, b):
    """a and b are the same value to the bit: types, dtypes, shapes,
    memory order (C / Fortran) and bytes, recursively."""
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return (a.dtype == b.dtype and a.shape == b.shape
                and a.flags.f_contiguous == b.flags.f_contiguous
                and a.flags.c_contiguous == b.flags.c_contiguous
                and a.tobytes(order="A") == b.tobytes(order="A"))
    if isinstance(a, np.generic):
        return a.dtype == b.dtype and a.tobytes() == b.tobytes()
    if isinstance(a, float):
        import struct
        return struct.pack("<d", a) == struct.pack("<d", b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_exact(x, y)
                                        for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_exact(a[k], b[k])
                                            for k in a)
    return a == b


def _step_from(rec, row, n, plant=None, s=None, t=None):
    """Step a plant restored at plant row `row` (record.plant_at) n steps
    with the recorded commands; (states, a_cg) of rows row+1 .. row+n."""
    from app.record import plant_at
    if plant is None:
        plant, s, t = plant_at(rec, row)
    P, dt = rec.plant, rec.header["dt"]
    S, A = [], []
    for j in range(row + 1, min(row + 1 + n, len(P["t"]))):
        s = plant.step(s, t, float(P["thrust"][j]), float(P["nozzle"][j]),
                       dt)
        t += dt
        S.append(np.array(s))
        A.append(plant.last_cg_acc)
    return np.array(S), np.array(A)


def g_resume():
    """The reset / replay contract (ACCEPTANCE P4; app/plantstate.py,
    UsvEnv.resume, record.replay(start=), record.plant_at): resuming from
    any recorded row reproduces the rest of the run bit for bit, with a
    plant whose step depends on a hidden state (wagner_entry) and with v0's
    carried actuator state; negative controls show both matter; the
    completeness audit finds no uncarried attribute; refusals."""
    C = Checks("resume")
    import copy
    from app import plantstate
    from app.record import (Record, RecordMismatch, plant_at, replay)
    t0 = time.time()
    wt = _wagner_test()
    rec, env = _wagner_record()
    P, h = rec.plant, rec.header
    hid = plantstate.loads_rows(P["hidden"]) if "hidden" in P else []
    states = [set(x["entry_state"][1].tolist()) if x and x["entry_state"]
              else set() for x in hid]
    rows_F = [j for j, st in enumerate(states) if 2 in st]
    rows_X = [j for j, st in enumerate(states) if 1 in st]
    C.ok("the wagner_entry record carries a non-trivial hidden state at "
         "every plant row (plant['hidden']; F and X planes present)",
         len(hid) == len(P["t"]) and all(x is not None for x in hid)
         and len(rows_F) > 20 and len(rows_X) > 20
         and h["physics"]["switches"] == list(wt)
         and h["test_switches"] == list(wt)
         and h["hidden_state"],
         f"{len(P['t'])} rows ({time.time() - t0:.1f} s); rows with F "
         f"planes {len(rows_F)}, with X {len(rows_X)}; aux "
         f"{P['aux'].shape}; u {P['s'][:, 6].min():.1f}..."
         f"{P['s'][:, 6].max():.1f} m/s; switches {list(wt)}")
    traj = env.plant.trajectory()
    th = traj["hidden"]
    C.ok("the plant's own trajectory()['hidden'][j + 1] == the record's "
         "row j, bit for bit (and row 0 of the plant's log is None: the "
         "first step starts from the steady field)",
         len(th) == len(hid) and th[0] is None and all(
             plantstate.same(th[j + 1], hid[j])
             for j in range(len(hid) - 1)),
         f"{len(th)} rows compared; es_resets {env.plant.es_resets}")
    # rows chosen where the field has F planes, inside control steps
    sub = h["sub"]
    m = len(rec.ctrl["t"])
    jF = rows_F[len(rows_F) // 2]
    rows = sorted({rows_F[0], jF, jF - jF % sub + sub // 2,
                   len(P["t"]) - 40})

    def plant_rows():
        out = []
        for j in rows:
            S, A = _step_from(rec, j, 40)
            n = len(S)
            ok = bits_equal(S, P["s"][j + 1:j + 1 + n]) and bits_equal(
                A, P["a_cg"][j + 1:j + 1 + n])
            if not ok:
                raise AssertionError(f"row {j}: differs after resume")
            out.append(f"{j} ({j % sub}/{sub} into its control step)")
        return "plant rows " + ", ".join(out) + ": 40 steps each, s and " \
            "a_cg identical"
    C.run("plant_at(rec, j) + the recorded commands == the record, any "
          "plant row (inside a control step too)", plant_rows)

    def from_traj():
        # the plant's own log (pre-step convention) as the restore source
        j = jF
        plant, s, t = plant_at(rec, j)
        plant.set_hidden_state(th[j + 1])
        S, _ = _step_from(rec, j, 40, plant, s, t)
        assert bits_equal(S, P["s"][j + 1:j + 41]), "differs"
        return f"row {j}: restored from trajectory()['hidden'][{j + 1}]"
    C.run("restoring from the plant's trajectory()['hidden'] gives the same",
          from_traj)

    def negative():
        j = jF
        plant, s, t = plant_at(rec, j)
        plantstate.set_hidden(plant, None)          # steady field instead
        S, _ = _step_from(rec, j, 40, plant, s, t)
        d_h = float(np.abs(S - P["s"][j + 1:j + 41]).max())
        # the actuators' delay lines: the row with the largest spread of
        # pending nozzle commands
        cols = h["state_cols"]
        ib = [i for i, c in enumerate(cols) if c.startswith("rudder.delay")]
        spread = np.ptp(P["aux"][:, ib], axis=1)
        jd = int(np.argmax(spread))
        plant, s, t = plant_at(rec, jd)
        for o in ("prop", "rudder"):
            getattr(plant, o).delay.buf = None      # as a fresh plant
        S2, _ = _step_from(rec, jd, 40, plant, s, t)
        d_a = float(np.abs(S2 - P["s"][jd + 1:jd + 41]).max())
        assert d_h > 0.0 and d_a > 0.0, (d_h, d_a)
        return (f"without the hidden state (row {j}): max |ds| {d_h:.2e} "
                f"over 40 steps; without the delay lines (row {jd}, pending "
                f"nozzle commands spread {np.degrees(spread[jd]):.1f} deg): "
                f"{d_a:.2e}")
    C.run("negative controls: dropping the hidden state or the delay lines "
          "changes the continuation (the checks above are not vacuous)",
          negative)

    def audit():
        out = []
        for j in (jF, len(P["t"]) // 3):
            orig, rest, _ = _audit_at(rec, j)
            diff, one, n = plantstate.audit(orig, rest, count=True)
            if diff:
                raise AssertionError(f"row {j}: uncarried {diff[:10]}")
            out.append(f"{j} ({n} attribute paths; built lazily in one "
                       f"plant only: {one or 'none'})")
        return (f"rows {', '.join(out)}: after the same next step every "
                f"attribute of the restored plant equals the original's "
                f"(scratch, logs and last-step outputs aside)")
    C.run("completeness audit: nothing the plant keeps between steps is "
          "left out (app/plantstate.audit)", audit)

    def audit_self():
        # the audit sees the delay lines -- callable objects (Delay defines
        # __call__); the plant's functions are markers, never entered, so a
        # deep copy reads as equal although its closures still point at the
        # original's objects (app/plantstate, FUNCTIONS ARE MARKERS)
        j = jF
        orig, rest, _ = _audit_at(rec, j)
        b = copy.deepcopy(rest)
        same = plantstate.audit(rest, b)
        assert same == [], f"an unaltered deep copy differs: {same[:6]}"
        out = []
        for o, f in (("prop", lambda v: v + 1000.0),
                     ("rudder", lambda v: -v + 0.1)):
            b = copy.deepcopy(rest)
            d = getattr(b, o).delay
            assert d.buf, f"{o}.delay.buf is empty at row {j}"
            d.buf = [f(v) for v in d.buf]
            diff = plantstate.audit(rest, b)
            hit = [k for k in diff if k.startswith(f"plant.{o}.delay.buf")]
            if not hit or len(hit) != len(diff):
                raise AssertionError(f"{o}.delay.buf altered: audit says "
                                     f"{diff[:6]}")
            out.append(f"{o}: {hit[0]}..")
        fa = {}
        plantstate._flat(rest, "plant", fa, set(), 0, frozenset([id(rest)]))
        assert any(k.startswith("plant.prop.delay.buf") for k in fa)
        # functions are markers (plantstate, FUNCTIONS ARE MARKERS): the
        # field memory _esm[0] -- a closure over the plant's geometry, kept
        # since 2026-10-07, present with wagner_entry -- is one path with a
        # marker value and nothing under it (walked from a deep copy it
        # reached the ORIGINAL's geometry and read as 29 differences)
        marks = {k: v for k, v in fa.items() if isinstance(v, str)
                 and v.startswith(("<function", "<method", "<partial"))}
        assert "plant._esm[0]" in marks, sorted(k for k in fa if "_esm" in k)
        assert not any("<" in k for k in fa), [k for k in fa if "<" in k][:5]
        assert "plant.slam.wagner_slope" in marks
        # swapping a function for another one is named, through the marker
        b = copy.deepcopy(rest)
        b.slam.wagner_slope = lambda d: 0.0 * d
        diff = plantstate.audit(rest, b)
        assert diff == ["plant.slam.wagner_slope"], diff
        # the marker of a rebuilt plant's own closure (another function
        # object, the same code) equals this plant's
        fo = {}
        plantstate._flat(orig, "plant", fo, set(), 0, frozenset([id(orig)]))
        assert orig._esm[0] is not rest._esm[0]
        assert fo["plant._esm[0]"] == fa["plant._esm[0]"]
        return (f"row {j}: unaltered deep copy -> []; altered delay lines "
                f"named ({'; '.join(out)}); {len(fa)} paths, {len(marks)} of "
                f"them function markers (e.g. plant._esm[0] = "
                f"{marks['plant._esm[0]'][:70]}...); a swapped function is "
                f"named; the marker of a rebuilt plant's closure is equal")
    C.run("the audit sees callable objects (the delay lines) and writes the "
          "plant's functions as markers, never entering their closures: "
          "altering a delay line or swapping a function on a deep copy is "
          "named, an unaltered deep copy (its closures pointing at the "
          "original's geometry) is not", audit_self)

    def codec():
        # the exact codec on values hidden_state() does not hold today
        cases = dict(
            zero_d=np.array(3.0), fortran=np.asfortranarray(
                np.arange(12.0).reshape(3, 4)),
            bool_=np.array([True, False, True]),
            int8=np.array([-128, 0, 127], np.int8),
            nan_neg0=np.array([np.nan, -0.0, np.inf, -np.nan]),
            py=(float("nan"), -0.0, 1e-310, 2 ** 70, True, None, "s"),
            npscal=(np.float64(-0.0), np.int8(-3), np.bool_(True),
                    np.float32(np.nan)),
            nested={"a": [np.zeros((0, 3)), (1, {"b": np.uint16(7)})]})
        bad = []
        for name, v in cases.items():
            back = plantstate.decode(json.loads(json.dumps(
                plantstate.encode(v))))
            if not _same_exact(v, back):
                bad.append(name)
        try:
            plantstate.encode({"x": np.array([1.0, None], object)})
            bad.append("object array accepted")
        except TypeError:
            pass
        assert not bad, bad
        return (f"{len(cases)} cases round-trip with dtype, shape (0-d), "
                f"memory order (Fortran), bits (NaN payload, -0.0) and "
                f"types; an object array raises TypeError at encode")
    C.run("the exact codec: 0-d, Fortran-ordered, bool, int8, NaN / -0.0, "
          "numpy scalars, nested; object arrays refused at once", codec)

    def env_rows():
        out = []
        for k in (1, m // 3, m - 2):
            rep = replay(rec, start=k)
            if rep.fingerprint(physical=True) != rec.fingerprint(
                    physical=True):
                raise AssertionError(f"start {k}: {rec.diff(rep, True)[:4]}")
            out.append(str(k))
        return (f"replay from control steps {', '.join(out)} of {m}: "
                f"physical fingerprint {rec.fingerprint(True)[:16]}")
    C.run("record.replay(rec, start=k) == the record (states, hidden "
          "state, actuator state, sensors, reward)", env_rows)

    def v0_closed_loop():
        # v0 scarab195, heading mode (the env's autopilot), safe mode on: a
        # scripted policy resumed at k continues the original record bit
        # for bit -- the env's controller state is restored too
        from app.runner import run
        from app.scenario import load
        sc = load("demo_scarab195.json").with_overrides(
            duration=10.0, safe_mode=True,
            policy={"type": "scripted", "throttle": [[0, 0.7], [5, 0.9]],
                    "heading": "track"})
        a = run(sc)
        k = 17
        b = run(Scenario(a.header["scenario"]), resume=(a, k))
        same = b.fingerprint() == a.fingerprint()
        rep = replay(a, start=k)
        same_rep = rep.fingerprint(True) == a.fingerprint(True)
        if not (same and same_rep):
            raise AssertionError(f"resume: {a.diff(b)[:4]}; replay: "
                                 f"{a.diff(rep, True)[:4]}")
        return (f"v0, {len(a.ctrl['t'])} control steps, resumed at {k}: "
                f"full fingerprint {a.fingerprint()[:16]} (every array, the "
                f"policy's actions and the autopilot state included); "
                f"aux columns {a.plant['aux'].shape[1]}, no hidden state "
                f"('hidden' in plant: {'hidden' in a.plant})")
    from app.scenario import Scenario
    C.run("v0 closed loop: run(..., resume=(rec, k)) with the same scripted "
          "policy == the original, every array", v0_closed_loop)

    def refusals():
        from app.env import UsvEnv
        msgs = []
        e = UsvEnv(rec.header["scenario"], action_mode="direct")
        try:
            e.resume(rec, 5)
            raise AssertionError("resumed without the test switches")
        except RecordMismatch as ex:
            msgs.append(str(ex)[:60])
        old = Record(dict(rec.header, format="usv-app-record/1"), rec.s0,
                     {k: v for k, v in rec.plant.items()
                      if k not in ("aux", "hidden")}, rec.ctrl, rec.sensors)
        try:
            replay(old, start=3)
            raise AssertionError("resumed a format-1 record mid-run")
        except RecordMismatch as ex:
            msgs.append(str(ex)[:60])
        assert replay(old, start=0).fingerprint(True)  # from the start: ok
        st = dict(rec.header["physics"], physics_sha1="0" * 40)
        bad = Record(dict(rec.header, physics=st), rec.s0, rec.plant,
                     rec.ctrl, rec.sensors)
        try:
            replay(bad, start=3)
            raise AssertionError("resumed a record of other physics code")
        except RecordMismatch as ex:
            msgs.append(str(ex)[:80])
        for k in (-1, m, 2.5):
            try:
                replay(rec, start=k)
                raise AssertionError(f"start {k} accepted")
            except ValueError:
                pass
        return "; ".join(msgs)
    C.run("refusals: other test switches, a format-1 record mid-run, other "
          "physics code, rows out of range", refusals)

    def saved():
        os.makedirs(TEST_RUNS, exist_ok=True)
        path = rec.save(os.path.join(TEST_RUNS, "resume_wagner.npz"))
        back = Record.load(path)
        assert back.fingerprint() == rec.fingerprint()
        rep = replay(back, start=m // 2)
        assert rep.fingerprint(True) == rec.fingerprint(True)
        return (f"{os.path.getsize(path) / 1e3:.0f} kB; hidden "
                f"{P['hidden'].nbytes / 1e3:.0f} kB raw JSON")
    C.run("save -> load keeps the state rows: a loaded record resumes "
          "exactly", saved)
    return C.result()


# ============================================================== resume_more
def g_resume_more():
    """More of the reset / replay contract: a record on the as1 preset plus
    all four P1b switches (fk_exact, orbital, wagner_entry, lateral_phi0)
    resumed from plant rows and control steps, audited; safe mode's state
    restored on resume (v0, the env's SafeMode with lowered thresholds so
    that it engages in a short run)."""
    C = Checks("resume_more")
    from app import plantstate
    from app.record import replay
    t0 = time.time()
    sw = _switches_plus(*P1B)
    rec, env = _wagner_record(duration=8.0, switches=sw)
    P, h = rec.plant, rec.header
    hid = plantstate.loads_rows(P["hidden"]) if "hidden" in P else []
    n, m, sub = len(P["t"]), len(rec.ctrl["t"]), h["sub"]
    C.ok("a record on the as1 preset + all four P1b switches: the hidden "
         "state is carried at every plant row",
         h["test_switches"] == list(sw) and len(hid) == n
         and all(x is not None for x in hid) and all(
             q in sw for q in P1B),
         f"{n} rows, {m} control steps ({time.time() - t0:.1f} s); "
         f"switches {list(sw)}; u {P['s'][:, 6].min():.1f}.."
         f"{P['s'][:, 6].max():.1f} m/s")
    rows = sorted({n // 4 + sub // 2, n // 2 + 1, n - 30})

    def plant_rows():
        out = []
        for j in rows:
            S, A = _step_from(rec, j, 30)
            k = len(S)
            if not (bits_equal(S, P["s"][j + 1:j + 1 + k])
                    and bits_equal(A, P["a_cg"][j + 1:j + 1 + k])):
                raise AssertionError(f"row {j}: differs after resume")
            out.append(f"{j} ({j % sub}/{sub} into its control step)")
        return "plant rows " + ", ".join(out) + ": 30 steps each, s and " \
            "a_cg identical"
    C.run("plant_at(rec, j) + the recorded commands == the record (preset "
          "+ the four P1b switches)", plant_rows)

    def audit():
        j = n // 2
        orig, rest, _ = _audit_at(rec, j)
        diff, one, cnt = plantstate.audit(orig, rest, count=True)
        if diff:
            raise AssertionError(f"row {j}: uncarried {diff[:10]}")
        return (f"row {j}: {cnt} attribute paths equal after the same next "
                f"step; built lazily in one plant only: {one or 'none'}")
    C.run("completeness audit (preset + the four P1b switches)", audit)

    def env_rows():
        out = []
        for k in (1, m // 2, m - 2):
            rep = replay(rec, start=k)
            if rep.fingerprint(physical=True) != rec.fingerprint(
                    physical=True):
                raise AssertionError(f"start {k}: {rec.diff(rep, True)[:4]}")
            out.append(str(k))
        return (f"replay from control steps {', '.join(out)} of {m}: "
                f"physical fingerprint {rec.fingerprint(True)[:16]}")
    C.run("record.replay(rec, start=k) == the record (preset + the four "
          "P1b switches)", env_rows)

    def safe_mode():
        # the env's safe mode (enter 20 deg, exit 10 deg, hold 2 s) never
        # engages in the demo sea; with enter 1 deg, exit 0.8 deg, hold 1 s
        # it switches on and off, and a resume must restore its state
        # (on, calm time, counters) for the closed loop to continue exactly
        import app.env as E
        from app.runner import run
        from app.scenario import Scenario, load

        class LowSafe(E.SafeMode):
            def __init__(self, dt_ctrl, enter_deg=1.0, exit_deg=0.8,
                         hold_s=1.0):
                super().__init__(dt_ctrl, enter_deg, exit_deg, hold_s)
        keep = E.SafeMode
        E.SafeMode = LowSafe
        try:
            sc = load("demo_scarab195.json").with_overrides(
                duration=10.0, safe_mode=True,
                policy={"type": "scripted", "throttle": [[0, 0.7],
                                                         [5, 0.9]],
                        "heading": "track"})
            a = run(sc)
            ss = np.asarray(a.ctrl["safe_state"])
            mm = len(ss)
            on_calm = [k for k in range(1, mm - 1)
                       if ss[k - 1, 0] == 1 and ss[k - 1, 1] > 0]
            on_zero = [k for k in range(1, mm - 1)
                       if ss[k - 1, 0] == 1 and ss[k - 1, 1] == 0]
            off_after = [k for k in range(1, mm - 1)
                         if ss[k - 1, 0] == 0 and ss[k - 1, 3] > 0]
            assert on_calm and on_zero, "safe mode never on with calm > 0"
            picks = sorted({on_calm[len(on_calm) // 2], on_zero[0]}
                           | ({off_after[0]} if off_after else set()))
            out = []
            for k in picks:
                b = run(Scenario(a.header["scenario"]), resume=(a, k))
                if b.fingerprint() != a.fingerprint():
                    raise AssertionError(f"resume at {k}: {a.diff(b)[:4]}")
                rep = replay(a, start=k)
                if rep.fingerprint(True) != a.fingerprint(True):
                    raise AssertionError(f"replay from {k}: "
                                         f"{a.diff(rep, True)[:4]}")
                out.append(f"{k} (on {int(ss[k - 1, 0])}, calm "
                           f"{ss[k - 1, 1]:.2f} s, entries "
                           f"{int(ss[k - 1, 3])})")
            return (f"v0, {mm} control steps, safe mode on in "
                    f"{int(np.sum(a.ctrl['safe']))} of them, "
                    f"{int(ss[-1, 3])} entries; resumed at "
                    f"{'; '.join(out)}: full fingerprint "
                    f"{a.fingerprint()[:16]} each time")
        finally:
            E.SafeMode = keep
    C.run("safe mode's state is restored on resume (v0 closed loop, safe "
          "mode on with calm time > 0 at the resume row): run(..., "
          "resume=(rec, k)) == the original, every array", safe_mode)
    return C.result()


# ================================================================= codehash
CODEHASH_PROBE = r"""
import json, os, sys
ROOT = sys.argv[1]
sys.path.insert(0, ROOT)
import numpy as np
M = sys.monitoring
TOOL = M.PROFILER_ID
M.use_tool_id(TOOL, "app_test codehash")
ran = {}
def on_start(code, offset):
    ran.setdefault(code.co_filename, set()).add(code.co_qualname)
    return M.DISABLE
M.register_callback(TOOL, M.events.PY_START, on_start)
from hydro import hulls
from sim import physics
from sim.planing_anyspeed import SWITCHES
from sim.wavefield import SeaState
allsw = tuple(k for k, v in SWITCHES.items() if v["implemented"])
cases = [("scarab195@as1", "gz"), ("planing_b@as1", "plant")]
H = {k: hulls.get(k) for k, _ in cases}
sea = SeaState(1.0, 5.0, seed=1)
before = set(sys.modules)
M.set_events(TOOL, M.events.PY_START)
for key, kind in cases:
    h = H[key]
    cls = physics.gz_class(h) if kind == "gz" else physics.plant_class(h)
    p = cls(h, sea, dt=h.scales()["dt"], test_switches=allsw)
    for u in (2.0, 12.0):
        s = np.zeros(14)
        s[6] = u
        s[12] = 0.5 * p.prop.t_max
        t = 0.0
        for i in range(4):
            s = p.step(s, t, 0.7 * p.prop.t_max, 0.2 * p.rudder.max, p.dt)
            t += p.dt
M.set_events(TOOL, 0)
def rel(f):
    a = os.path.abspath(f)
    if (not os.path.isfile(a) or not a.startswith(os.path.abspath(ROOT))
            or ".venv" in a):
        return None
    return os.path.relpath(a, ROOT).replace(os.sep, "/")
out = dict(switches=list(allsw), hashed=list(physics.VERSIONS["as1"]["code"]),
           ran={}, loaded=[])
for f, q in ran.items():
    r = rel(f)
    if r:
        out["ran"][r] = sorted(q)
for n, mod in sys.modules.items():
    f = getattr(mod, "__file__", None)
    if n not in before and f and rel(f):
        out["loaded"].append(rel(f))
print("CODEHASH " + json.dumps(out))
"""

# repository modules an as1 plant loads or runs that are not in its code
# list, each with why it is not numerics of a plant step, and the code of it
# that may run (None: any)
CODEHASH_NOT_NUMERIC = {
    "sim/config.py": ("imported by the GZ plant module's factory; no "
                      "function of it runs in a build or a step",
                      {"<module>"}),
}


def g_codehash():
    """The code hash of as1 (sim/physics.VERSIONS['as1']['code']: what a
    record's stamp and every cache tag carry) covers every repository
    module an as1 plant loads or runs: a fresh process builds the GZ plant
    of scarab195@as1 and the linear plant of planing_b@as1 with EVERY
    implemented switch on and steps each at 2 and 12 m/s, recording the
    files whose code starts (sys.monitoring) and the modules it imports;
    each must be in the list or in CODEHASH_NOT_NUMERIC. A module missing
    from the list is a stale-record hole: editing it changes neither the
    stamp nor the cache tags, so env.resume / replay(start=k) would accept
    an older record and the caches would be reused (review of P1c)."""
    C = Checks("codehash")
    import subprocess
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", CODEHASH_PROBE, ROOT],
                       capture_output=True, text=True, timeout=600,
                       cwd=ROOT)
    line = [x for x in r.stdout.splitlines() if x.startswith("CODEHASH ")]
    if not line:
        C.ok("the probe ran", False, (r.stderr or r.stdout)[-1500:])
        return C.result()
    d = json.loads(line[-1][len("CODEHASH "):])
    hashed = set(d["hashed"])
    seen = sorted(set(d["ran"]) | set(d["loaded"]))
    missing, allowed = [], []
    for f in seen:
        if f in hashed:
            continue
        why = CODEHASH_NOT_NUMERIC.get(f)
        code = set(d["ran"].get(f, ["<module>"]))
        if why is not None and (why[1] is None or code <= why[1]):
            allowed.append(f"{f} ({why[0]})")
        else:
            missing.append(f"{f} (ran: {sorted(code)[:6]})")
    C.ok("every repository module an as1 plant with all switches loads or "
         "runs is in sim/physics.VERSIONS['as1']['code'] (or named as not "
         "numeric)", not missing,
         f"{time.time() - t0:.0f} s; switches {d['switches']}; NOT HASHED: "
         f"{missing or 'none'}; hashed and seen: "
         f"{sorted(hashed & set(seen))}; allowed: {allowed or 'none'}")
    from app import codeguard
    mods = set(m.replace(".", "/") + ".py" for m in codeguard.modules_of(
        "as1"))
    C.ok("app/codeguard.load_all imports every module of the list (so a "
         "server loads all of it with the hash)", mods == hashed,
         sorted(mods ^ hashed))
    return C.result()


# ===================================================================== sensors
def _sensor_run(sensors, duration=30.0, seed=2, sensor_seed=None, hs=1.0):
    from app.env import UsvEnv
    from app.runner import run
    from app.scenario import straight
    sc = straight("scarab195", hs=hs, duration=duration, seed=seed,
                  plant="gz", sensor_seed=sensor_seed)
    raw = sc.to_dict()
    raw["agents"]["usv0"]["sensors"] = sensors
    from app.scenario import Scenario
    sc = Scenario(raw)
    env = UsvEnv(sc, action_mode="heading")
    rec = run(sc, {"type": "scripted", "throttle": [[0, 0.7], [15, 0.9]],
                   "heading": [[0, 0], [10, 0], [20, 30]]}, env=env)
    return env, rec


def g_sensors():
    C = Checks("sensors")
    from app import geo
    from app.sensors import world_velocity
    zero = {k: {"latency_s": 0.0} for k in ("imu", "gnss", "heading",
                                            "speed_log", "wave_probe",
                                            "preview")}
    zero["noise"] = False
    env, rec = _sensor_run(zero, duration=8.0)
    T, S = rec.states_with_start()
    p = env.plant
    th, zr = float(p.Th_rest), float(p.Z_rest)
    by_t = {round(t, 9): i for i, t in enumerate(T)}

    def rows(name):
        a = rec.sensors[name]
        idx = np.array([by_t[round(t, 9)] for t in a["t_meas"]])
        return a, idx
    a, idx = rows("imu")
    att = np.stack([S[idx, 3], th + S[idx, 4],
                    np.array([math.remainder(y, 2 * math.pi)
                              for y in S[idx, 5]])], 1)
    gyro = np.array([geo.euler_rates_to_body(S[i, 3], th + S[i, 4], S[i, 9],
                                             S[i, 10], S[i, 11]) for i in idx])
    f = np.zeros((len(idx), 3))
    for j, i in enumerate(idx):
        if i == 0:
            acc = np.zeros(3)
        else:
            acc = (world_velocity(S[i]) - world_velocity(S[i - 1])) / \
                (T[i] - T[i - 1])
        R = geo.rot_zyx(S[i, 3], th + S[i, 4], S[i, 5])
        f[j] = R.T @ (acc + [0, 0, 9.81])
    e_imu = max(np.abs(a["values"][:, 6:9] - att).max(),
                np.abs(a["values"][:, 3:6] - gyro).max(),
                np.abs(a["values"][:, 0:3] - f).max())
    C.ok("C-P5 noise off: IMU == specific force from the plant's velocity, "
         "body rates, attitude (1e-9)", e_imu < 1e-9 and len(idx) == len(T),
         f"max err {e_imu:.1e}, {len(idx)} samples (every plant step)")
    a, idx = rows("gnss")
    enu = np.stack([S[idx, 0], S[idx, 1], zr + S[idx, 2]], 1)
    e_g = np.abs(a["values"][:, 3:6] - enu).max()
    back = np.array([env.frame.to_enu(*v[:3]) for v in a["values"]])
    e_geo = np.abs(back - a["values"][:, 3:6]).max()
    vel = np.array([world_velocity(S[i]) for i in idx])
    C.ok("noise off: GNSS ENU == plant position; lat/lon/alt -> ENU < 1 mm; "
         "velocity == plant velocity",
         e_g < 1e-9 and e_geo < 1e-3 and np.abs(a["values"][:, 6:9]
                                                - vel).max() < 1e-12,
         f"ENU {e_g:.1e} m, geodetic round trip {e_geo:.1e} m, "
         f"{len(idx)} samples at {env.sensors['gnss'].rate_hz:g} Hz")
    a, idx = rows("heading")
    yaw = np.array([math.remainder(y, 2 * math.pi) for y in S[idx, 5]])
    C.ok("noise off: heading == yaw, bearing == 90 - yaw (deg)",
         np.abs(a["values"][:, 0] - yaw).max() < 1e-12 and np.abs(
             a["values"][:, 1] - (90 - np.degrees(yaw)) % 360).max() < 1e-9)
    a, idx = rows("speed_log")
    C.ok("noise off: speed log == u", bits_equal(a["values"][:, 0], S[idx, 6]))
    a, idx = rows("wave_probe")
    pts = env.sensors["wave_probe"].pts
    ok = True
    for j, i in enumerate(idx):
        c, s_ = math.cos(S[i, 5]), math.sin(S[i, 5])
        X = S[i, 0] + pts[:, 0] * c - pts[:, 1] * s_
        Y = S[i, 1] + pts[:, 0] * s_ + pts[:, 1] * c
        ok &= np.abs(env.sea.eta(X, Y, T[i]) - a["values"][j]).max() < 1e-12
    C.ok("noise off: wave probe == the sea's elevation at the stem and CG",
         ok)
    a, idx = rows("preview")
    d = env.sensors["preview"].dist + env.x_stem
    ok = all(np.abs(env.sea.eta(S[i, 0] + d * math.cos(S[i, 5]),
                                S[i, 1] + d * math.sin(S[i, 5]), T[i])
                    - a["values"][j]).max() < 1e-12
             for j, i in enumerate(idx))
    C.ok("noise off: preview == the sea ahead at 1..5 L", ok)

    # statistics
    cfg = {"imu": dict(sigma_f=0.1, sigma_w=0.01, sigma_att=0.01,
                       sigma_yaw=0.02, bias_f=0.0, bias_w=0.0, rw_f=0.0,
                       rw_w=0.0, dropout=0.2, latency_s=0.0),
           "gnss": dict(sigma_h=1.0, sigma_v=2.0, sigma_vel=0.05, bias_h=0.0,
                        rw_h=0.0, dropout=0.1, latency_s=0.1),
           "heading": dict(sigma=0.01, bias=0.0, latency_s=0.04),
           "speed_log": dict(sigma=0.1, bias=0.0, scale_sd=0.0),
           "wave_probe": dict(sigma=0.02), "preview": dict(sigma0=0.02,
                                                           sigma_per_m=0.002)}
    cfg0 = {k: dict(v) for k, v in cfg.items()}
    cfg0["noise"] = False
    env1, r1 = _sensor_run(cfg, sensor_seed=11)
    env0, r0 = _sensor_run(cfg0, sensor_seed=11)
    C.ok("sensor noise does not change the trajectory",
         bits_equal(r1.plant["s"], r0.plant["s"]))
    lo, hi = None, None
    worst = []
    from scipy.stats import binom
    for name, sn in env1.sensors.items():
        a1, a0 = r1.sensors[name], r0.sensors[name]
        res = a1["values"] - a0["values"]
        if name == "heading":
            res[:, 0] = np.array([math.remainder(x, 2 * math.pi)
                                  for x in res[:, 0]])
        n = len(res)
        lo, hi = chi2_band(n)
        for c, ch in enumerate(sn.channels):
            sg = sn.sigma[c]
            if sg == 0 or (name == "heading" and ch == "bearing_deg"):
                continue
            ratio = float(np.mean(res[:, c] ** 2) / sg ** 2)
            worst.append((name, ch, n, round(ratio, 3)))
            C.ok(f"C-P5 {name}.{ch}: mean square / sigma^2 in the chi-square "
                 f"99.9 % band", lo <= ratio <= hi,
                 f"n {n}, ratio {ratio:.3f}, band [{lo:.3f}, {hi:.3f}]")
        k = int((~a1["valid"]).sum())
        p_ = sn.dropout
        if p_ > 0:
            klo, khi = binom.ppf(0.0005, n, p_), binom.ppf(0.9995, n, p_)
            C.ok(f"{name}: dropout count in the binomial 99.9 % band",
                 klo <= k <= khi, f"{k} of {n}, p {p_}, band [{klo:.0f}, "
                                  f"{khi:.0f}]")
        lat = a1["t_avail"] - a1["t_meas"]
        C.ok(f"{name}: latency exactly {sn.latency_s:g} s on every sample",
             np.allclose(lat, sn.latency_s, atol=1e-12, rtol=0),
             f"configured {sn.cfg.get('latency_s')}, actual {sn.latency_s:g}")
    g = r1.sensors["gnss"]["values"]
    back = np.array([env1.frame.to_enu(*v[:3]) for v in g])
    C.ok("noisy GNSS: lat/lon/alt are the noisy ENU position, converted",
         np.abs(back - g[:, 3:6]).max() < 1e-3)
    # bias, random walk and scale over many resets / one long run
    from app.sensors import build
    env_b = env0
    s0 = r0.s0
    sens = build({"imu": dict(sigma_f=0, sigma_w=0, sigma_att=0, sigma_yaw=0,
                              bias_f=0.05, bias_w=0.0, rw_f=0, rw_w=0),
                  "speed_log": dict(sigma=0.0, scale_sd=0.05)}, env_b)
    B, SC = [], []
    for k in range(400):
        sens["imu"].reset(1000 + k, env_b, 0.0, s0)
        sens["speed_log"].reset(1000 + k, env_b, 0.0, s0)
        B += list(sens["imu"].b[:3])
        SC.append(sens["speed_log"].scale[0] - 1.0)
    lo, hi = chi2_band(len(B))
    rb = float(np.mean(np.square(B)) / 0.05 ** 2)
    C.ok("constant bias b0 ~ N(0, bias_sd^2) over 400 resets (chi-square)",
         lo <= rb <= hi, f"ratio {rb:.3f}, band [{lo:.3f}, {hi:.3f}]")
    lo, hi = chi2_band(len(SC))
    rs = float(np.mean(np.square(SC)) / 0.05 ** 2)
    C.ok("scale factor - 1 ~ N(0, scale_sd^2) over 400 resets (chi-square)",
         lo <= rs <= hi, f"ratio {rs:.3f}, band [{lo:.3f}, {hi:.3f}]")
    cfg_rw = {"imu": dict(sigma_f=0, sigma_w=0, sigma_att=0, sigma_yaw=0,
                          bias_f=0, bias_w=0, rw_f=0, rw_w=0.01)}
    cfg_rw0 = {"imu": dict(cfg_rw["imu"]), "noise": False}
    e_rw, r_rw = _sensor_run(cfg_rw, duration=20.0, sensor_seed=5)
    _, r_rw0 = _sensor_run(cfg_rw0, duration=20.0, sensor_seed=5)
    bw = r_rw.sensors["imu"]["values"][:, 3:6] - \
        r_rw0.sensors["imu"]["values"][:, 3:6]
    inc = np.diff(bw, axis=0).ravel()
    h = e_rw.sensors["imu"].period * e_rw.dt
    lo, hi = chi2_band(len(inc))
    rr = float(np.mean(inc ** 2) / (0.01 ** 2 * h))
    C.ok("random-walk bias increments ~ N(0, rw^2 h) (chi-square)",
         lo <= rr <= hi, f"n {len(inc)}, ratio {rr:.3f}, band [{lo:.3f}, "
                         f"{hi:.3f}]")
    # seeding
    _, r1b = _sensor_run(cfg, sensor_seed=11)
    _, r1c = _sensor_run(cfg, sensor_seed=12)
    same = all(bits_equal(r1.sensors[n]["values"], r1b.sensors[n]["values"])
               for n in r1.sensors)
    diff = all(not bits_equal(r1.sensors[n]["values"],
                              r1c.sensors[n]["values"]) for n in r1.sensors)
    C.ok("same sensor seed -> same readings; another -> other readings, "
         "same trajectory", same and diff and bits_equal(r1.plant["s"],
                                                         r1c.plant["s"]))
    return C.result(stats=worst)


# ===================================================================== interfaces
def _msg_fields(path):
    out = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.split("#", 1)[0].strip()
        if not ln or "=" in ln:
            continue
        out.append(ln.split()[1])
    return out


def g_interfaces():
    C = Checks("interfaces")
    import http.client
    import socket
    from app import lotus, wsproto
    from app.env import UsvEnv
    from app.scenario import straight
    from app import server as S
    from app.server import AppServer
    os.makedirs(TEST_RUNS, exist_ok=True)
    # the live run of sse_case must not get a 429 because other work on
    # this machine holds RAM (the gate itself: robust group)
    S.RESERVE_GB = 0.2
    srv = AppServer(port=0, udp_port=0, runs_dir=TEST_RUNS)
    srv.start()
    port, uport = srv.port, srv.udp.port
    doc = straight("scarab195", duration=4.8, seed=5).to_dict()
    cmds = [lotus.cmd_array(throttle=0.6 + 0.1 * math.sin(k),
                            **{"nozzle(deg)": 8 * math.sin(0.4 * k)})
            for k in range(20)]
    direct = UsvEnv(doc, action_mode="direct")
    direct.reset(seed=5)
    ref = []
    for c in cmds:
        direct.step(lotus.parse_cmd(c))
        ref.append(lotus.vessel_position_array(direct)["vessels"][0])

    def via(call):
        rep = call(dict(op="reset", scenario=doc))
        assert rep["op"] == "reset_ok", rep
        got = []
        for c in cmds:
            rep = call(dict(op="step", **c))
            assert rep["op"] == "state", rep
            got.append(rep["poses"]["vessels"][0])
        return got, rep

    def ws_case():
        cl = wsproto.Client("127.0.0.1", port, "/ws")
        try:
            got, last = via(cl.call)
        finally:
            cl.close()
        assert got == ref, "poses differ"
        assert last["truncated"] and last["reason"] == "duration"
        return f"20 steps over the websocket == direct env (poses ==), " \
               f"last t {last['t']:.2f}"
    C.run("C-P6 websocket lockstep session == direct env, bit for bit",
          ws_case)

    def udp_case():
        sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sk.settimeout(60)

        def call(m):
            sk.sendto(json.dumps(m).encode(), ("127.0.0.1", uport))
            return json.loads(sk.recvfrom(65535)[0])
        try:
            got, last = via(call)
        finally:
            sk.close()
        assert got == ref, "poses differ"
        return "20 steps over UDP == direct env"
    C.run("C-P6 UDP lockstep session == direct env, bit for bit", udp_case)

    def http_post(path, body, origin=None):
        h = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
        hd = {"Content-Type": "application/json"}
        if origin:
            hd["Origin"] = origin
        h.request("POST", path, json.dumps(body), hd)
        r = h.getresponse()
        return r.status, json.loads(r.read() or b"{}")

    def http_get(path):
        h = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
        h.request("GET", path)
        r = h.getresponse()
        return r.status, r.read()

    def http_session():
        st, rep = http_post("/api/session", dict(op="reset", scenario=doc))
        assert st == 200 and rep["op"] == "reset_ok", rep
        sid = rep["session"]
        st, rep = http_post("/api/session", dict(op="step", session=sid,
                                                 **cmds[0]))
        assert rep["poses"]["vessels"][0] == ref[0], rep
        assert set(rep["poses"]) == {"header", "vessels"} and "usv" in rep
        return f"reset + step over HTTP (session {sid[:8]}) == direct env"
    C.run("HTTP session endpoint == direct env", http_session)

    def sse_case():
        st, rep = http_post("/api/run", dict(
            scenario="demo_scarab195.json", duration=4.8, rate=0,
            policy={"type": "scripted", "throttle": 0.7}))
        assert st == 200, rep
        rid = rep["run_id"]
        h = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
        h.request("GET", f"/api/stream/{rid}")
        r = h.getresponse()
        assert r.getheader("Content-Type") == "text/event-stream"
        kinds, data, cur = [], {}, None
        while True:
            ln = r.readline().decode()
            if not ln:
                break
            ln = ln.rstrip("\n")
            if ln.startswith("event: "):
                cur = ln[7:]
            elif ln.startswith("data: "):
                kinds.append(cur)
                data.setdefault(cur, []).append(json.loads(ln[6:]))
                if cur in ("end", "error"):
                    break
        assert kinds[0] == "header" and kinds[-1] == "end", kinds[-3:]
        hdr = data["header"][0]
        n_fr = sum(len(b["frames"]["t"]) for b in data["frames"])
        assert {"meta", "sea", "hull", "frames"} <= set(hdr)
        rec = os.path.join(ROOT, data["end"][0]["record"])
        assert os.path.exists(rec), rec
        st, vj = http_get(f"/api/run/{rid}/viewer.json")
        vj = json.loads(vj)
        assert st == 200 and len(vj["frames"]["t"]) == n_fr + 1
        assert vj["meta"]["server_code_stale"] == [] and vj["hull"]["x_tab"]
        return (f"{kinds.count('frames')} frame batches ({n_fr} plant steps) "
                f"+ header + end; record {os.path.relpath(rec, ROOT)}; "
                f"viewer.json {len(vj['frames']['t'])} frames")
    C.run("C-P6 server-sent events of a live run: header, frames, end; "
          "record and viewer JSON", sse_case)

    def guards():
        st, rep = http_post("/api/run", dict(scenario="demo_scarab195.json"),
                            origin="http://evil.example")
        assert st == 403, (st, rep)
        st, rep = http_post("/api/run", dict(scenario="demo_scarab195.json",
                                             policy="os:system"))
        assert st == 400 and "callable" in rep["error"], (st, rep)
        st, rep = http_post("/api/run", dict(scenario="../../x.json"))
        assert st == 400, (st, rep)
        st, _ = http_get("/app/../sim/physics.py")
        st2, _ = http_get("/hydro/hulls.py")
        st3, _ = http_get("/viewer/..%5capp%5cserver.py")
        st4, _ = http_get("/viewer/%2e%2e/app/server.py")
        assert st == 404 and st2 == 404 and st3 == 404 and st4 == 404, \
            (st, st2, st3, st4)
        return "cross-origin POST 403, callable policy 400, scenario path " \
               "traversal 400, files outside viewer/ app/runs/ " \
               "app/scenarios/ 404 (also encoded backslash and dot forms; " \
               "raw backslashes: robust group)"
    C.run("server guards: origin, callable policies, paths", guards)

    def shapes():
        env = UsvEnv(doc, action_mode="direct")
        env.reset()
        env.step([0.7, 0.0])
        pa = lotus.vessel_position_array(env)
        out = []
        f_vpa = _msg_fields(os.path.join(LOTUS_MSG, "lotusim_msgs", "msg",
                                         "VesselPositionArray.msg"))
        f_vp = _msg_fields(os.path.join(LOTUS_MSG, "lotusim_msgs", "msg",
                                        "VesselPosition.msg"))
        f_vc = _msg_fields(os.path.join(LOTUS_MSG, "lotusim_msgs", "msg",
                                        "VesselCmd.msg"))
        f_gps = _msg_fields(os.path.join(LOTUS_MSG, "lotusim_sensor_msgs",
                                         "msg", "GPS.msg"))
        assert set(pa) == set(f_vpa), (set(pa), f_vpa)
        assert set(pa["vessels"][0]) == set(f_vp), (pa["vessels"][0], f_vp)
        assert set(pa["vessels"][0]["pose"]) == {"position", "orientation"}
        g = lotus.gps(env)
        assert set(g) == set(f_gps), (set(g), f_gps)
        c = lotus.cmd_array(throttle=0.5, **{"heading(deg)": 10})["cmds"][0]
        assert set(c) <= set(f_vc), (set(c), f_vc)
        out.append(f"VesselPositionArray {f_vpa}, VesselPosition {f_vp}, "
                   f"GPS {f_gps}, VesselCmd {f_vc}")
        q = pa["vessels"][0]["pose"]["orientation"]
        from app.geo import quat_to_rot
        R = quat_to_rot((q["x"], q["y"], q["z"], q["w"]))
        yaw = math.atan2(R[1, 0], R[0, 0])
        assert abs(math.remainder(yaw - env.s[5], 2 * math.pi)) < 1e-9
        return "; ".join(out) + "; orientation's body x is the heading"
    if os.path.isdir(LOTUS_MSG):
        C.run("C-P6 message shapes == LOTUSim .msg field lists", shapes)
    else:
        C.ok("LOTUSim .msg files available for the shape check", False,
             LOTUS_MSG)

    def cmd_errors():
        s = lotus.Session()
        s.handle(dict(op="reset", scenario=doc))
        r1 = s.handle(dict(op="step", cmd_string=json.dumps(
            {"throttle": 0.5, "rudder": 3})))
        r2 = s.handle(dict(op="step", cmds=[
            dict(vessel_name="usv0", cmd_string='{"throttle": 0.5, '
                 '"nozzle": 0}')] * 2))
        r3 = s.handle(dict(op="step", cmd_string=json.dumps(
            {"jet(T)": 2000.0, "heading(deg)": 20})))
        assert r1["op"] == "error" and r2["op"] == "error" and \
            r3["op"] == "state", (r1, r2, r3)
        return f"{r1['error'][:60]} | {r2['error'][:60]}"
    C.run("cmd_string: unknown key and duplicate vessel commands refused; "
          "jet(T) + heading(deg) accepted", cmd_errors)

    def ros():
        import importlib
        rb = importlib.import_module("app.ros2_bridge")
        assert hasattr(rb, "build_node") and hasattr(rb, "main")
        return f"imports; rclpy available here: {rb.available()}"
    C.run("ROS 2 bridge module imports without rclpy", ros)
    srv.shutdown()
    return C.result()


# ===================================================================== e2e
def g_e2e(fname):
    C = Checks(f"e2e:{fname}")
    from app import export
    from app.record import replay
    from app.runner import run
    from app.scenario import load
    os.makedirs(TEST_RUNS, exist_ok=True)
    sc = load(fname)
    t0 = time.time()
    rec = run(sc)
    wall = time.time() - t0
    summ = rec.summary()
    name = sc.name
    rec.save(os.path.join(TEST_RUNS, name + ".npz"))
    C.ok("C-P7 runs to the end with finite states",
         rec.header["finite"] and rec.header["reason"] in (
             "track_complete", "duration"),
         f"{summ}; {wall:.0f} s wall, policy "
         f"{rec.header['policy']['t_policy_per_step']:.3f} s/step")
    from app.record import code_changes
    ch = code_changes(rec)
    C.ok("the record's physics stamp and hull sha1 match the current code",
         ch == [], ch)
    key = sc.hull_key
    C.ok("physics version as the key says",
         rec.header["physics"]["physics"] == ("as1" if "@as1" in key
                                              else "v0"),
         rec.header["physics"])
    if sc.track_spec()["kind"] == "waypoints":
        C.ok("the waypoint course is completed", rec.header["reason"] ==
             "track_complete", f"leg {int(rec.ctrl['leg'][-1])} of "
                               f"{rec.header['track']['points'].__len__() - 1}"
                               f", t {rec.header['t_end']:.1f} s")
    vj = export.viewer_json(rec, os.path.join(TEST_RUNS,
                                              name + ".viewer.json"))
    C.ok("viewer JSON: every plant step + the start, the run's sea, the "
         "hull's stations",
         len(vj["frames"]["t"]) == len(rec.plant["t"]) + 1 and
         len(vj["sea"]["a"]) == len(rec.header["sea"]["a"]) and
         len(vj["hull"]["x"]) == len(vj["hull"]["yc"]),
         f"{len(vj['frames']['t'])} frames, {len(vj['sea']['a'])} "
         f"components, hull {vj['hull']['name']} L {vj['hull']['L']}")

    def hull_lines():
        # the viewers draw from x_fine / yc_fine / zk_fine: they must be
        # hydro/hull_geometry's lines of this hull at export.line_points,
        # keyed by its cache tag, the v0 family formula (taper / stem_rise)
        # not needed; and the drawn lines must be close to the module
        # everywhere, the stem included
        from app.hulls import resolve
        from hydro.hull_geometry import stations_of
        from sim import physics
        H = vj["hull"]
        hh = resolve(sc.hull_spec)
        g = stations_of(hh)
        xf = export.line_points(g)
        assert np.array_equal(np.asarray(H["x_fine"]), xf - hh.lcg)
        assert np.array_equal(np.asarray(H["yc_fine"]), g.chine(xf))
        assert np.array_equal(np.asarray(H["zk_fine"]),
                              -hh.vcg + g.keel(xf)[0])
        assert "taper" not in H and "stem_rise" not in H
        assert H["cache_tag"] == physics.cache_tag(hh)
        assert vj["meta"]["hull_tag"] == export.hull_tag(hh)
        # deviation from the module on a dense reference: 20 001 uniform
        # points and 2 000 more spaced geometrically toward the stem (down
        # to 1e-9 L from it)
        L = float(g.L)
        ref = np.union1d(np.linspace(0.0, L, 20001),
                         L - L * np.logspace(-9, 0, 2000))
        yc_m, zk_m = g.chine(ref), g.keel(ref)[0]

        def dev(x, yc, zk=None):
            e = float(np.abs(np.interp(ref, x, yc) - yc_m).max())
            if zk is not None:
                e = max(e, float(np.abs(np.interp(ref, x, zk) - zk_m).max()))
            return e
        HX = np.asarray(H["x_fine"])                   # body x
        HY, HZ = np.asarray(H["yc_fine"]), np.asarray(H["zk_fine"])
        e_exp = dev(HX + hh.lcg, HY, HZ + hh.vcg)
        # the web mesh (viewer/demo.html, proto_web.html: XST = the exported
        # points and those of 141 uniform ones not within 1e-9 m of one;
        # vertices at ycOf / zkOf, linear between the exported points)
        XT, XS = H["x_transom"], H["x_stem"]
        hx = [x for x in HX.tolist() if XT <= x <= XS]
        u = [XT + (XS - XT) * i / 140 for i in range(141)]
        u = [x for x in u if not any(abs(y - x) <= 1e-9 for y in hx)]
        xst = np.asarray(sorted(hx + u))
        e_mesh = dev(xst + hh.lcg, np.interp(xst, HX, HY),
                     np.interp(xst, HX, HZ) + hh.vcg)
        # the water shader's chine table
        xt, yt = np.asarray(H["x_tab"]), np.asarray(H["yc_tab"])
        e_tab = dev(xt + hh.lcg, yt)
        # for comparison: the 201 uniform points of the previous export
        xu = np.linspace(0.0, L, 201)
        e_u = dev(xu, g.chine(xu), g.keel(xu)[0])
        tol = H["line_tol"]
        assert e_exp <= 2 * tol and e_mesh <= 2 * tol, (e_exp, e_mesh)
        assert len(xt) <= 64 and e_tab < 5e-3, (len(xt), e_tab)
        x0 = (1 - hh.taper) * hh.L
        sv = np.clip((xf - x0) / (hh.L - x0), 0, 1)
        yv = 0.5 * hh.b * np.sqrt(np.maximum(1 - sv * sv, 0))
        zv = -hh.vcg + hh.stem_rise * sv * sv
        return (f"{len(xf)} points, tag {H['cache_tag']}; largest deviation "
                f"from the module (stem included): export "
                f"{1e3 * e_exp:.2f} mm, web mesh ({len(xst)} stations) "
                f"{1e3 * e_mesh:.2f} mm, shader table ({len(xt)} points) "
                f"{1e3 * e_tab:.2f} mm; the previous 201 uniform points "
                f"{1e3 * e_u:.1f} mm; the v0 family formula would be off by "
                f"{np.abs(yv - H['yc_fine']).max():.3f} m (chine), "
                f"{np.abs(zv - H['zk_fine']).max():.3f} m (keel)")
    C.run("viewer hull = hydro/hull_geometry's chine and keel (x_fine, "
          "yc_fine, zk_fine) under the hull's cache tag; export, web mesh "
          "and shader table close to the module everywhere (stem "
          "included)", hull_lines)
    tm = 0.5 * rec.header["t_end"]
    bs = export.blender_scene(rec, os.path.join(TEST_RUNS,
                                                name + ".scene.json"),
                              tm - 2.0, tm + 2.0, fps=30)
    C.ok("Blender scene.json (render.py's format): 4 s at 30 fps, track, "
         "plant steps; rigid pose vs the plant's keel formula",
         len(bs["frames"]) == 121 and {"meta", "sea", "hull", "frames",
                                       "track", "steps"} == set(bs)
         and bs["meta"]["check_keel_rigid_vs_plant_m"] < 0.05,
         f"{len(bs['frames'])} frames, {len(bs['steps'])} steps, keel "
         f"rigid vs plant {1000 * bs['meta']['check_keel_rigid_vs_plant_m']:.1f}"
         f" mm")
    t0 = time.time()
    rep = replay(rec)
    C.ok("command replay reproduces the run (physical fingerprint)",
         rep.fingerprint(physical=True) == rec.fingerprint(physical=True),
         f"{time.time() - t0:.0f} s; {rec.diff(rep)[:4]}")
    t0 = time.time()
    k = len(rec.ctrl["t"]) // 2
    rep = replay(rec, start=k)
    C.ok("replay resumed from the middle (control step k) reproduces the "
         "rest of the run (physical fingerprint)",
         rep.fingerprint(physical=True) == rec.fingerprint(physical=True),
         f"k {k} of {len(rec.ctrl['t'])}, {time.time() - t0:.0f} s; "
         f"{rec.diff(rep, True)[:4]}")
    return C.result(summary=summ, wall_s=wall)


# ===================================================================== robust
def _raw_http(port, request):
    """Send raw bytes, return (status line, body bytes)."""
    import socket
    with socket.create_connection(("127.0.0.1", port), timeout=30) as sk:
        sk.sendall(request)
        buf = b""
        while True:
            c = sk.recv(65536)
            if not c:
                break
            buf += c
    head, _, body = buf.partition(b"\r\n\r\n")
    return head.split(b"\r\n", 1)[0].decode(errors="replace"), body


def _ws_raw(port):
    """A raw websocket connection after the handshake (socket, file)."""
    import base64
    import socket
    sk = socket.create_connection(("127.0.0.1", port), timeout=30)
    key = base64.b64encode(os.urandom(16)).decode()
    sk.sendall((f"GET /ws HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13"
                f"\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        head += sk.recv(1)
    assert b" 101 " in head, head
    return sk, sk.makefile("rwb", buffering=0)


def _close_code(f):
    """Read frames until a close frame; its status code."""
    import struct
    from app import wsproto
    while True:
        fin, op, data = wsproto.read_frame(f)
        if op == wsproto.OP_CLOSE:
            return struct.unpack(">H", data[:2])[0] if len(data) >= 2 \
                else None


def g_robust():
    """The failure modes of the review (plant-only, no MPC but one short
    hand run with sensors): path traversal, concurrent HTTP sessions,
    non-finite actions on every interface, a policy failing mid-run, the
    LOTUSim waypoint syntax, UDP retries, sensor and scenario validation,
    Blender export windows, JSON, websocket protocol errors, live-run
    limits, record staleness, reset seeding, sensor-driven policies."""
    C = Checks("robust")
    import http.client
    import socket
    import struct
    import threading
    from app import lotus, server as S, wsproto
    from app.env import ActionError, UsvEnv
    from app.record import Record, code_changes
    from app.runner import run
    from app.scenario import Scenario, ScenarioError, load, straight
    os.makedirs(TEST_RUNS, exist_ok=True)

    # ---------------------------------------------------- static paths
    bad_paths = ["/viewer/..\\.git\\config", "/viewer/..\\app\\server.py",
                 "/viewer/..\\learn\\meta\\ACCEPTANCE_2026-10-06.md",
                 "/viewer/..%5c.git%5cconfig", "/viewer/%2e%2e/app/server.py",
                 "/viewer/..%2fapp%2fserver.py", "/viewer/%252e%252e/x",
                 "/viewer/C:/Windows/win.ini", "/app/runs/../server.py",
                 "/viewer/./../app/server.py", "/viewer/demo.html%00.png",
                 "/app/scenarios/..\\..\\app\\env.py"]

    def static_unit():
        got = {p: S.safe_static(p) for p in bad_paths}
        assert not any(got.values()), {k: v for k, v in got.items() if v}
        ok = S.safe_static("/viewer/demo.html")
        assert ok and ok.endswith("demo.html"), ok
        assert S.safe_static("/app/scenarios/demo_scarab195.json")
        return f"{len(bad_paths)} traversal forms -> None; viewer/demo.html " \
               f"and a scenario file resolve"
    C.run("safe_static refuses backslash, encoded, drive and '..' forms",
          static_unit)

    srv = S.AppServer(port=0, udp_port=0, runs_dir=TEST_RUNS)
    srv.start()
    port, uport = srv.port, srv.udp.port
    try:
        def static_live():
            codes = {}
            for pth in bad_paths:
                for meth in ("GET", "HEAD"):
                    st, body = _raw_http(port, (
                        f"{meth} {pth} HTTP/1.1\r\nHost: x\r\n"
                        f"Connection: close\r\n\r\n").encode())
                    codes[(meth, pth)] = st
                    assert " 404 " in st + " ", (meth, pth, st, body[:80])
            st, _ = _raw_http(port, b"GET /viewer/demo.html HTTP/1.1\r\n"
                                    b"Host: x\r\nConnection: close\r\n\r\n")
            assert " 200 " in st, st
            return f"{len(codes)} raw GET/HEAD requests -> 404; " \
                   f"viewer/demo.html -> 200"
        C.run("server: traversal over raw sockets (GET and HEAD) -> 404",
              static_live)

        doc = straight("scarab195", duration=4.8, seed=5).to_dict()

        def post(path, body, timeout=120):
            h = http.client.HTTPConnection("127.0.0.1", port,
                                           timeout=timeout)
            h.request("POST", path, json.dumps(body),
                      {"Content-Type": "application/json"})
            r = h.getresponse()
            out = r.status, json.loads(r.read() or b"{}")
            h.close()
            return out

        # ------------------------------------------ stale server code
        def stale_server():
            # as if sim/planing_anyspeed.py had been edited after this
            # server loaded it: the process's as1 hash no longer matches
            # the disk (app/codeguard.py) -> new live as1 runs and session
            # resets are refused, v0 is not affected, the viewer meta says it
            from sim import physics
            from app import codeguard
            keep = physics._CODE["as1"]
            physics._CODE["as1"] = "0" * 40
            try:
                note = codeguard.stale("as1")
                assert note and codeguard.stale("v0") is None, note
                st, out = post("/api/run", {"scenario":
                                            "demo_scarab195_as1.json",
                                            "duration": 1.0})
                assert st == 409 and "StaleCode" in out.get("error", ""),                     (st, out)
                st, rep = post("/api/session", dict(
                    op="reset", scenario="demo_scarab195_as1.json"))
                assert rep.get("op") == "error" and "StaleCode" in                     rep.get("error", ""), rep
                st, rep = post("/api/session", dict(op="reset",
                                                    scenario=doc))
                assert rep.get("op") == "reset_ok", rep       # v0: runs
                assert codeguard.notes(["as1"]) == [note]
            finally:
                physics._CODE["as1"] = keep
            # the viewer data say it at serve time (not as cached): a v0
            # record's meta.server_code_stale is [] now, and names v0 while
            # the process's v0 hash is replaced
            r0 = run(straight("scarab195", duration=0.48, seed=4),
                     {"type": "scripted", "throttle": 0.6})
            r0.save(os.path.join(TEST_RUNS, "robust_stale_v0.npz"))

            def get_meta():
                h = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
                h.request("GET", "/api/run/robust_stale_v0/viewer.json")
                r = h.getresponse()
                d = json.loads(r.read())
                h.close()
                return r.status, d
            st, d0 = get_meta()
            assert st == 200 and d0["meta"]["server_code_stale"] == [] and                 d0["hull"].get("x_tab"), (st, d0["meta"].get(
                    "server_code_stale"))
            keep0 = physics._CODE["v0"]
            physics._CODE["v0"] = "0" * 40
            try:
                st, d1 = get_meta()
            finally:
                physics._CODE["v0"] = keep0
            assert st == 200 and d1["meta"]["server_code_stale"] and                 "v0" in d1["meta"]["server_code_stale"][0], d1["meta"]
            assert codeguard.stale("as1") is None or physics.code_changed(
                "as1")
            return f"as1 refused (409; session reset -> StaleCode); v0 "                    f"session runs; viewer meta.server_code_stale [] when "                    f"fresh, names v0 when its hash is replaced (set at "                    f"serve time); note: {note[:90]}"
        C.run("a server whose physics code changed on disk refuses new live "
              "runs of that version and says so (app/codeguard)",
              stale_server)

        # ------------------------------------------ concurrent sessions
        def concurrent():
            n_cl, n_st = 4, 5
            cmds = {c: [lotus.cmd_array(throttle=0.5 + 0.1 * c,
                                        **{"nozzle(deg)": 3.0 * (c - 1.5)})
                        for _ in range(n_st)] for c in range(n_cl)}
            sids, errs = {}, []
            for c in range(n_cl):
                st, rep = post("/api/session", dict(op="reset", scenario=doc))
                assert rep["op"] == "reset_ok" and rep.get("session"), rep
                sids[c] = rep["session"]
            assert len(set(sids.values())) == n_cl
            got = {c: [] for c in range(n_cl)}

            def client(c):
                for k in range(n_st):
                    st, rep = post("/api/session", dict(
                        op="step", session=sids[c], **cmds[c][k]))
                    if rep.get("op") != "state":
                        errs.append(rep)
                        return
                    got[c].append((rep["k"], rep["poses"]["vessels"][0]))
            th = [threading.Thread(target=client, args=(c,))
                  for c in range(n_cl)]
            for t in th:
                t.start()
            for t in th:
                t.join()
            assert not errs, errs[:2]
            for c in range(n_cl):
                assert [k for k, _ in got[c]] == list(range(1, n_st + 1)), \
                    got[c]
                env = UsvEnv(doc, action_mode="direct")
                env.reset(seed=5)
                for k in range(n_st):
                    env.step(lotus.parse_cmd(cmds[c][k]))
                    assert got[c][k][1] == lotus.vessel_position(env), (c, k)
            st, r1 = post("/api/session", dict(op="step", **cmds[0][0]))
            st, r2 = post("/api/session", dict(op="step", session="nope",
                                               **cmds[0][0]))
            assert r1["op"] == "error" and r2["op"] == "error", (r1, r2)
            return (f"{n_cl} clients x {n_st} concurrent steps: each k = "
                    f"1..{n_st}, poses == its own direct env; no session -> "
                    f"'{r1['error'][:40]}'; unknown -> '{r2['error'][:40]}'")
        C.run("HTTP sessions: one per client, concurrent steps never mix",
              concurrent)

        # ------------------------------------------ non-finite actions
        def nonfinite():
            env = UsvEnv(doc, action_mode="direct")
            env.reset(seed=5)
            s0 = env.s.copy()
            bad = [[float("nan"), 0.0], [0.5, float("inf")],
                   {"throttle": float("nan"), "nozzle": 0.0},
                   {"thrust": 1000.0, "heading": float("nan")},
                   {"throttle": "fast", "nozzle": 0.0}, "abc"]
            for a in bad:
                try:
                    env.step(a)
                    raise AssertionError(f"accepted {a!r}")
                except ActionError:
                    pass
            assert bits_equal(s0, env.s) and env.k == 0 and \
                env.reason is None
            env.step([0.5, 0.0])
            for cs in ('{"throttle": NaN, "nozzle": 0}',
                       '{"throttle": 0.5, "heading(deg)": Infinity}',
                       '{"jet(T)": "x", "nozzle": 0}'):
                try:
                    lotus.parse_cmd({"cmd_string": cs})
                    raise AssertionError(f"parse_cmd accepted {cs}")
                except ValueError:
                    pass
            nan_cmd = dict(cmd_string='{"throttle": NaN, "nozzle": 0}')
            ok_cmd = lotus.cmd_array(throttle=0.6, nozzle=0.0)
            out = []
            cl = wsproto.Client("127.0.0.1", port, "/ws")
            try:
                assert cl.call(dict(op="reset", scenario=doc))["op"] == \
                    "reset_ok"
                r = cl.call(dict(op="step", **nan_cmd))
                assert r["op"] == "error" and "non-finite" in r["error"], r
                r = cl.call(dict(op="step", **ok_cmd))
                assert r["op"] == "state" and r["k"] == 1, r
                out.append("websocket")
            finally:
                cl.close()
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sk.settimeout(60)
            try:
                def ucall(m):
                    sk.sendto(json.dumps(m).encode(), ("127.0.0.1", uport))
                    return json.loads(sk.recvfrom(65535)[0])
                assert ucall(dict(op="reset", scenario=doc))["op"] == \
                    "reset_ok"
                r = ucall(dict(op="step", **nan_cmd))
                assert r["op"] == "error" and "non-finite" in r["error"], r
                r = ucall(dict(op="step", **ok_cmd))
                assert r["op"] == "state" and r["k"] == 1, r
                out.append("UDP")
            finally:
                sk.close()
            st, rep = post("/api/session", dict(op="reset", scenario=doc))
            sid = rep["session"]
            st, r = post("/api/session", dict(op="step", session=sid,
                                              action={"throttle": 0.5,
                                                      "nozzle": None}))
            assert r["op"] == "error", r
            st, r = post("/api/session", dict(op="step", session=sid,
                                              **nan_cmd))
            assert r["op"] == "error" and "non-finite" in r["error"], r
            st, r = post("/api/session", dict(op="step", session=sid,
                                              **ok_cmd))
            assert r["op"] == "state" and r["k"] == 1, r
            out.append("HTTP")
            return (f"{len(bad)} bad env actions refused, state untouched; "
                    f"NaN / inf / string cmd_strings refused; over "
                    f"{', '.join(out)}: error reply, then the next valid "
                    f"step is k = 1")
        C.run("non-finite actions refused on env, cmd_string, websocket, "
              "UDP, HTTP; the episode stays alive", nonfinite)

        # ------------------------------------------ UDP retries
        def udp_seq():
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sk.settimeout(60)
            try:
                def ucall(m):
                    sk.sendto(json.dumps(m).encode(), ("127.0.0.1", uport))
                    return json.loads(sk.recvfrom(65535)[0])
                r = ucall(dict(op="reset", scenario=doc, seq=0))
                assert r["op"] == "reset_ok" and r["seq"] == 0, r
                step = dict(op="step", seq=1,
                            **lotus.cmd_array(throttle=0.7, nozzle=0.0))
                a = ucall(step)
                b = ucall(step)                  # the "lost reply" resend
                assert a["k"] == 1 and b["k"] == 1 and b.get("retry"), \
                    (a["k"], b["k"])
                g = ucall(dict(step, seq=3))
                assert g["op"] == "error" and "out of order" in g["error"], g
                c = ucall(dict(step, seq=2))
                assert c["k"] == 2, c
                return ("seq 1 sent twice -> k 1 both times (second marked "
                        "retry); seq 3 after 1... refused (gap); seq 2 -> k 2")
            finally:
                sk.close()
        C.run("UDP seq: a resent step is answered from cache, gaps refused",
              udp_seq)

        # ------------------------------------------ websocket protocol
        def ws_proto():
            out = []
            sk, f = _ws_raw(port)        # invalid UTF-8 in a text frame
            try:
                payload = b"\xff\xfe{}"
                key = os.urandom(4)
                f.write(bytes([0x81, 0x80 | len(payload)]) + key + bytes(
                    b ^ key[i % 4] for i, b in enumerate(payload)))
                code = _close_code(f)
                assert code == 1007, code
                out.append("invalid UTF-8 -> 1007")
            finally:
                sk.close()
            sk, f = _ws_raw(port)        # unmasked client frame
            try:
                f.write(bytes([0x81, 2]) + b"{}")
                code = _close_code(f)
                assert code == 1002, code
                out.append("unmasked -> 1002")
            finally:
                sk.close()
            sk, f = _ws_raw(port)        # continuation without a start
            try:
                key = os.urandom(4)
                f.write(bytes([0x80, 0x82]) + key + bytes(
                    b ^ key[i % 4] for i, b in enumerate(b"{}")))
                code = _close_code(f)
                assert code == 1002, code
                out.append("continuation without a start -> 1002")
            finally:
                sk.close()
            old = wsproto.MAX_MSG
            sk, f = _ws_raw(port)        # fragments over the cap, FIN clear
            wsproto.MAX_MSG = 1000
            try:
                for i in range(3):
                    data = b"x" * 600
                    key = os.urandom(4)
                    b0 = (wsproto.OP_TEXT if i == 0 else wsproto.OP_CONT)
                    f.write(bytes([b0, 0x80 | 126]) + struct.pack(">H", 600)
                            + key + bytes(b ^ key[j % 4] for j, b in
                                          enumerate(data)))
                code = _close_code(f)
                assert code == 1009, code
                out.append("non-final fragments over the cap -> 1009")
            finally:
                wsproto.MAX_MSG = old
                sk.close()
            cl = wsproto.Client("127.0.0.1", port, "/ws")   # still serving
            try:
                assert cl.call(dict(op="state"))["op"] == "error"
            finally:
                cl.close()
            return "; ".join(out) + "; the server still serves"
        C.run("websocket: protocol errors close with 1007 / 1002 / 1009, "
              "never HTTP inside the stream", ws_proto)

        # ------------------------------------------ live-run limits
        def limits():
            old = S.MAX_LIGHT, S.MAX_HEAVY, S.RESERVE_GB
            S.MAX_LIGHT, S.MAX_HEAVY = 1, 0
            try:
                # the RAM gate (with an impossible reserve) ...
                S.RESERVE_GB = 1e6
                st0, rep0 = post("/api/run", dict(
                    scenario="demo_scarab195.json", duration=2.4,
                    policy={"type": "scripted", "throttle": 0.6}))
                assert st0 == 429 and "free RAM" in rep0["error"], rep0
                # ... and the count limits (a small reserve, so that other
                # work on this machine cannot fail the check)
                S.RESERVE_GB = 0.2
                st, rep = post("/api/run", dict(
                    scenario="demo_scarab195.json", duration=30.0, rate=1.0,
                    policy={"type": "scripted", "throttle": 0.6}))
                assert st == 200, (st, rep)
                rid = rep["run_id"]
                st2, rep2 = post("/api/run", dict(
                    scenario="demo_scarab195.json", duration=30.0, rate=1.0,
                    policy={"type": "scripted", "throttle": 0.6}))
                st3, rep3 = post("/api/run", dict(
                    scenario="eval_head_seas.json", policy={"type": "m0"}))
                post(f"/api/stop/{rid}", {})
                assert st2 == 429 and st3 == 429, (st2, rep2, st3, rep3)
                st4, rep4 = post("/api/run", dict(
                    scenario="demo_scarab195.json",
                    policy={"type": "scripted", "throttle": "fast"}))
                assert st4 == 400 and "throttle" in rep4["error"], rep4
                inline = straight("scarab195", duration=2.4).to_dict()
                inline["agents"]["usv0"]["policy"] = {
                    "type": "callable", "target": "os:getcwd"}
                st5, rep5 = post("/api/run", dict(scenario=inline))
                assert st5 == 400 and "callable" in rep5["error"], rep5
                return (f"RAM gate 429 ('{rep0['error'][:40]}'); second "
                        f"light run 429 ('{rep2['error'][:50]}'), m0 429; "
                        f"scripted 'fast' 400 at POST; an inline "
                        f"scenario's callable policy 400")
            finally:
                S.MAX_LIGHT, S.MAX_HEAVY, S.RESERVE_GB = old
        C.run("live runs: limits answer 429; bad specs and scenario "
              "callables refused at POST", limits)

        def save_no_overwrite():
            cl = wsproto.Client("127.0.0.1", port, "/ws")
            try:
                cl.call(dict(op="reset", scenario=doc))
                cl.call(dict(op="step", **lotus.cmd_array(throttle=0.5,
                                                         nozzle=0.0)))
                a = cl.call(dict(op="save", name="robust_save"))
                b = cl.call(dict(op="save", name="robust_save"))
            finally:
                cl.close()
            assert a["op"] == b["op"] == "saved" and a["path"] != b["path"]
            for r in (a, b):
                os.remove(os.path.join(ROOT, r["path"]))
            return f"{a['path']} then {b['path']}"
        C.run("session save never overwrites a record", save_no_overwrite)
    finally:
        srv.shutdown()

    # ---------------------------------------------------- dumps
    def dumps_case():
        txt = S.dumps({"e": "cmd_string keys ['NaNx'] -Infinity Infinity",
                       "x": float("nan"), "y": np.float32(np.inf),
                       "z": np.array([1.0, -np.inf]), "t": (1, 2.5)})

        def no_const(c):
            raise ValueError(c)
        d = json.loads(txt, parse_constant=no_const)
        assert d["e"] == "cmd_string keys ['NaNx'] -Infinity Infinity"
        assert d["x"] is None and d["y"] is None and d["z"] == [1.0, None]
        return txt
    C.run("dumps: non-finite numbers -> null, strings untouched, strict "
          "JSON", dumps_case)

    # ---------------------------------------------------- policy failure
    class Boom:
        action_mode = "direct"
        name = "boom"
        closed = False

        def __init__(self, k_fail, nan=False):
            self.k_fail, self.nan = k_fail, nan

        def reset(self, env):
            self.env = env

        def act(self, obs, info):
            if self.env.k == self.k_fail:
                if self.nan:
                    return [float("nan"), 0.0]
                raise RuntimeError("boom at step %d" % self.k_fail)
            return [0.6, 0.0]

        def close(self):
            self.closed = True

    def policy_error():
        sc = straight("scarab195", duration=4.8, seed=2)
        p1 = Boom(5)
        rec = run(sc, p1)
        assert rec.header["reason"].startswith(
            "policy_error: RuntimeError: boom at step 5"), rec.header["reason"]
        assert len(rec.ctrl["t"]) == 5 and p1.closed
        assert "boom" in rec.header["error"]["traceback"]
        p2 = Boom(3, nan=True)
        rec2 = run(sc, p2)
        assert rec2.header["reason"].startswith("policy_error: ActionError")
        assert len(rec2.ctrl["t"]) == 3 and p2.closed
        try:
            run(sc, Boom(2), raise_on_error=True)
            raise AssertionError("raise_on_error did not raise")
        except RuntimeError:
            pass
        live = S.LiveRun("robust_live_boom", sc, Boom(4), rate=0.0,
                         runs_dir=TEST_RUNS)
        live.run()                       # in this thread
        kinds = [k for k, _ in live.events]
        end = live.events[-1][1]
        assert kinds[-1] == "end" and os.path.exists(live.record_path), kinds
        assert end["reason"].startswith("policy_error"), end
        back = Record.load(live.record_path)
        assert len(back.ctrl["t"]) == 4
        return (f"partial records of 5 and 3 steps, reasons "
                f"'{rec.header['reason'][:45]}' / "
                f"'{rec2.header['reason'][:40]}'; close() ran; live run "
                f"saved {os.path.basename(live.record_path)} and pushed end")
    C.run("a policy failing mid-run: partial record, reason, traceback, "
          "close(); live run saves it", policy_error)

    # ---------------------------------------------------- LOTUSim syntax
    def lotus_waypoints():
        ref = {"latitude": 48.8566, "longitude": 2.3522, "altitude": 0.0}

        def doc(wp, pos=None):
            return {"name": "lt", "reference_position": ref,
                    "agents": {"usv0": dict(model="scarab195",
                                            position=pos or dict(ref),
                                            heading=30,
                                            waypoint_interface=wp)}}
        pts = [{"lat": 48.8576, "lng": 2.3522}, {"lat": 48.8576,
                                                 "lng": 2.3552}]
        a = Scenario(doc({"enabled": True, "loop": False, "mode":
                          "waypoints", "waypoints": pts}))
        ts = a.track_spec()
        want = [a.frame.to_enu(p["lat"], p["lng"], 0.0)[:2] for p in pts]
        assert ts["kind"] == "waypoints" and np.allclose(ts["points"], want)
        b = Scenario(doc({"loop": True, "mode": "circle",
                          "circle": {"radius": 9}}))      # demo.yaml's block
        tb = b.track_spec()
        r = np.hypot(*np.asarray(tb["points"]).T)
        assert len(tb["points"]) == 8 and np.allclose(r, 9.0) and tb["loop"]
        c = Scenario(doc({"mode": "line", "line": {"direction": 0.5,
                                                   "length": 300.0}}))
        tc = c.track_spec()["points"]
        assert np.allclose(tc, [[300 * math.cos(0.5), 300 * math.sin(0.5)]])
        refused = []
        for wp in ({"enabled": True, "path": [[1.2635, 103.7516]]},
                   {"enabled": True},
                   {"mode": "waypoints", "waypoints": [{"lat": 99.0,
                                                        "lng": 0.0}]},
                   {"mode": "spiral"},
                   {"path_enu": [[10, 0]], "radius": -5},
                   {"path_enu": [[10, 0]], "speed": 3}):
            try:
                Scenario(doc(wp)).track_spec()
                raise AssertionError(f"accepted {wp}")
            except ScenarioError as ex:
                refused.append(str(ex)[:50])
        d = Scenario(doc({"enabled": False, "mode": "circle",
                          "circle": {"radius": 9}}))
        assert d.track_spec()["kind"] == "line"
        return (f"waypoints [{{lat, lng}}] -> ENU, circle r 9 -> 8 points, "
                f"line -> 1 leg, block without 'enabled' is on; refused: "
                f"{refused}")
    C.run("scenario: LOTUSim waypoint_interface (waypoints / circle / line, "
          "enabled by default); bare [lat, lon] path refused",
          lotus_waypoints)

    # ---------------------------------------------------- validation
    def validation():
        base = straight("scarab195", duration=4.8).to_dict()

        def with_(path, val):
            d = json.loads(json.dumps(base))
            node = d
            for k in path[:-1]:
                node = node.setdefault(k, {})
            node[path[-1]] = val
            return d
        cases = {
            "n_freq 0": (("environment", "ocean", "n_freq"), 0),
            "n_dir 0": (("environment", "ocean", "n_dir"), 0),
            "seed 3.7": (("usv", "seed"), 3.7),
            "seed -1": (("usv", "seed"), -1),
            "wave_height 'abc'": (("environment", "ocean", "wave_height"),
                                  "abc"),
            "duration 'x'": (("usv", "duration"), "x"),
            "dropout 1.5": (("agents", "usv0", "sensors"),
                            {"gnss": {"dropout": 1.5}}),
            "rate_hz 0": (("agents", "usv0", "sensors"),
                          {"imu": {"rate_hz": 0}}),
            "rate_hz -10": (("agents", "usv0", "sensors"),
                            {"imu": {"rate_hz": -10}}),
            "latency -1": (("agents", "usv0", "sensors"),
                           {"gnss": {"latency_s": -1}}),
            "preview 10.001 / 10.002": (("agents", "usv0", "sensors"),
                                        {"preview": {"distances":
                                                     [10.001, 10.002]}}),
            "unknown policy type": (("agents", "usv0", "policy"),
                                    {"type": "magic"}),
            "scripted throttle 'fast'": (("agents", "usv0", "policy"),
                                         {"type": "scripted",
                                          "throttle": "fast"}),
            "learned without a row": (("agents", "usv0", "policy"),
                                      {"type": "learned"}),
            "hand guidance 'zigzag'": (("agents", "usv0", "policy"),
                                       {"type": "hand",
                                        "guidance": "zigzag"}),
        }
        msgs = {}
        for what, (path, val) in cases.items():
            try:
                Scenario(with_(path, val))
                raise AssertionError(f"accepted: {what}")
            except ScenarioError as ex:
                msgs[what] = str(ex)[:70]
        for what, key in (("wave_height 'abc'", "wave_height"),
                          ("duration 'x'", "duration"), ("seed 3.7", "seed"),
                          ("n_freq 0", "n_freq")):
            assert key in msgs[what], (what, msgs[what])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sc = Scenario(with_(("environment", "ocean", "wave_height"), 5.0))
        assert sc.warnings and any("R1" in str(x.message) for x in w)
        env = UsvEnv(sc)
        env.reset()
        env.step([0.5, 0.0])
        hd = env.header()
        assert hd["scenario_warnings"] == sc.warnings
        # dropout 1.0 is valid: nothing is ever delivered valid
        s1 = Scenario(with_(("agents", "usv0", "sensors"),
                            {"gnss": {"dropout": 1.0, "latency_s": 0.0},
                             "speed_log": {"rate_hz": None},
                             "preview": {"distances": [10.2, 10.4]}}))
        e1 = UsvEnv(s1)
        obs, _ = e1.reset()
        for _ in range(3):
            obs, *_ = e1.step([0.5, 0.0])
        assert obs["gnss"][0] == 0.0 and obs["gnss"][1] == -1.0
        assert e1.sensors["speed_log"].period == 1
        assert e1.sensors["preview"].channels == ("eta_10.2m", "eta_10.4m")
        assert len(e1.sensors["preview"].reading()) == 4
        return (f"{len(cases)} bad scenarios refused at load, e.g. "
                f"{[msgs[k] for k in ('n_freq 0', 'seed 3.7', 'dropout 1.5')]}"
                f"; Hs 5 m loads with an R1 warning in the header; "
                f"dropout 1.0 -> never valid; rate null -> every step; "
                f"preview 10.2 / 10.4 -> {e1.sensors['preview'].channels}")
    C.run("validation at load: numbers, seeds, sensors, policy specs; R1 "
          "warnings recorded", validation)

    # ---------------------------------------------------- Blender export
    def blender_windows():
        from app import export
        from app import __main__ as cli
        rec = run(straight("scarab195", duration=2.4), {"type": "scripted",
                                                        "throttle": 0.6})
        p = os.path.join(TEST_RUNS, "robust_bl.npz")
        rec.save(p)
        out = os.path.join(TEST_RUNS, "robust_bl.scene.json")
        bad = []
        for t0, t1 in ((200.0, 210.0), (1.0, 1.0), (2.0, 1.0), (-1.0, 1.0),
                       (0.5, 5.0)):
            try:
                export.blender_scene(rec, out, t0, t1)
                raise AssertionError(f"window {t0, t1} accepted")
            except ValueError as ex:
                bad.append(str(ex)[:40])
        sc = export.blender_scene(rec, out, 0.5, 2.0, fps=10)
        assert len(sc["frames"]) == 16
        try:
            import contextlib
            import io
            with contextlib.redirect_stderr(io.StringIO()):
                cli.main(["export", p])
            raise AssertionError("export with no output flag accepted")
        except SystemExit as ex:
            assert ex.code == 2
        mtime = os.path.getmtime(out)
        try:
            with contextlib.redirect_stderr(io.StringIO()) as err:
                cli.main(["export", p, "--blender", out, "--t0", "200",
                          "--t1", "210"])
            raise AssertionError("CLI window accepted")
        except SystemExit as ex:
            assert ex.code == 2 and "window" in err.getvalue(), \
                err.getvalue()
        assert os.path.getmtime(out) == mtime     # nothing written
        cmd = export.blender_command(out, os.path.join(TEST_RUNS, "fr"),
                                     clip=(0, 15), mp4=os.path.join(
                                         TEST_RUNS, "x.mp4"))
        assert "--clip" in cmd and "0:15" in cmd and "--mp4" in cmd
        return (f"windows refused: {bad}; 0.5-2 s at 10 fps = 16 frames; "
                f"no-flag export exits 2; clip command {cmd[0]}")
    C.run("Blender export: window checked (0 <= t0 < t1 <= t_end), "
          "export needs an output, clip/mp4 command", blender_windows)

    # ---------------------------------------------------- records
    def staleness():
        rec = run(straight("scarab195", duration=0.96, seed=4),
                  {"type": "scripted", "throttle": 0.6})
        assert code_changes(rec) == []
        fp = rec.fingerprint()
        h2 = json.loads(json.dumps(rec.header))
        h2["physics"]["physics_sha1"] = "0" * 40
        r2 = Record(h2, rec.s0, rec.plant, rec.ctrl, rec.sensors)
        ch = code_changes(r2)
        assert ch and "physics code changed" in ch[0], ch
        assert r2.fingerprint() != fp and \
            r2.fingerprint(physical=True) != rec.fingerprint(physical=True)
        h3 = json.loads(json.dumps(rec.header))
        h3["sea"]["phi"][0] += 1e-9
        assert Record(h3, rec.s0, rec.plant, rec.ctrl,
                      rec.sensors).fingerprint() != fp
        h4 = json.loads(json.dumps(rec.header))
        h4["hull_sha1"] = "f" * 40
        assert any("hull" in c for c in code_changes(
            Record(h4, rec.s0, rec.plant, rec.ctrl, rec.sensors)))
        return f"stamp / sea / hull edits change the fingerprint; " \
               f"code_changes: {ch}"
    C.run("records: the fingerprint covers stamp, hull, seeds and sea; "
          "code changes are named", staleness)

    # ---------------------------------------------------- reset seeding
    def seeding():
        env = UsvEnv(straight("scarab195", duration=0.48, seed=1))
        o, i = env.reset(seed=11)
        seq1, seas = [i["seed"]], [np.asarray(env.sea.phi).copy()]
        for _ in range(3):
            o, i = env.reset()
            seq1.append(i["seed"])
            seas.append(np.asarray(env.sea.phi).copy())
        env.reset(seed=11)
        seq2 = [11] + [env.reset()[1]["seed"] for _ in range(3)]
        assert seq1 == seq2 and len(set(seq1)) == 4
        assert len({s.tobytes() for s in seas}) == 4
        e2 = UsvEnv(straight("scarab195", duration=0.48, seed=1))
        first = e2.reset()[1]["seed"]
        again = e2.reset()[1]["seed"]
        assert first == 1 and again != 1
        return f"reset(11) + 3 x reset() -> seeds {seq1} (twice), 4 seas; " \
               f"first plain reset uses the scenario seed {first}"
    C.run("reset(): new sea per episode, reproducible from the first seed",
          seeding)

    # ---------------------------------------------------- sensors in loop
    def sensed():
        sens = {"gnss": {"sigma_h": 3.0, "dropout": 0.2},
                "heading": {"sigma": 0.05}}

        def sc_with(noise):
            d = straight("scarab195", duration=9.6, seed=3).to_dict()
            d["agents"]["usv0"]["sensors"] = dict(sens, noise=noise)
            d["agents"]["usv0"]["track"] = {"heading": 10.0}
            return Scenario(d)
        pol = {"type": "scripted", "throttle": 0.7, "heading": "track",
               "state_source": "sensors"}
        a = run(sc_with(True), pol)
        b = run(sc_with(False), pol)
        t = run(sc_with(True), dict(pol, state_source="truth"))
        t0 = run(sc_with(False), dict(pol, state_source="truth"))
        assert not bits_equal(a.plant["s"], b.plant["s"])
        assert bits_equal(t.plant["s"], t0.plant["s"])
        h = run(sc_with(True), {"type": "hand", "frame": "heading",
                                "state_source": "sensors"}, duration=2.4)
        assert h.header["finite"] and h.header["policy"]["summary"][
            "state_source"] == "sensors"
        from app.export import viewer_meta, state_source
        assert state_source(a.header["policy"]["spec"],
                            a.header["policy"]["summary"]) == "sensors"
        dy = float(np.abs(a.plant["s"][:, 1] - b.plant["s"][:, 1]).max())
        return (f"scripted on sensors: GNSS noise moves the boat (max "
                f"|dy| {dy:.2f} m over 9.6 s); on truth it does not; hand "
                f"(heading frame, sensors) runs 10 steps, finite")
    C.run("state_source 'sensors': the sensors' errors reach the loop",
          sensed)

    def hulls_list():
        from app.hulls import listing
        rows = listing()
        L = {k: ok for k, _, ok, _ in rows}
        N = {k: n for k, _, _, n in rows}
        assert L["scarab195"] and L["probe_hullB"] and L["scarab195@as1"]
        assert L["probe_hullB@as1"] and L["planing_b@as1"]
        # planing_b: v0 cannot honour it, the listing says so and why
        assert not L["planing_b"] and "planing_b@as1" in N["planing_b"]
        # said once, and v0's defaults are not called hard-coded
        nb = N["planing_b"]
        assert nb.count("v0 refuses") == 1 and "as1 only" not in nb, nb
        assert "other inputs differ from v0's fixed values" in nb, nb
        assert "v0" not in N["planing_b@as1"], N["planing_b@as1"]
        assert not L["wigley10"] and not L["kcs_2_10"]
        assert "wigley10@as1" not in L      # versions are the planing plant's
        from app.env import UsvEnv
        try:
            UsvEnv({"name": "x", "agents": {"usv0": {
                "model": "planing_b", "waypoint_interface": {
                    "path_enu": [[100, 0]]}}}})
            raise AssertionError("planing_b (v0) accepted")
        except ValueError as ex:
            assert "planing_b@as1" in str(ex), ex
        return (f"runnable: {sorted(k for k, v in L.items() if v)}; "
                f"planing_b: {N['planing_b'][-60:]}")
    C.run("hull listing says which keys can run", hulls_list)
    return C.result()


# ===================================================================== torch
def _mission_metrics(rec, phi):
    """learn/repro/task.Mission.metrics() recomputed from a record, in
    Mission's order of floating-point operations."""
    from learn.repro.task import G, cost_parts
    h = rec.header
    L, u_ref = h["L"], float(h["u_ref"])
    S, A, AB = rec.plant["s"], rec.plant["a_cg"], rec.plant["a_bow"]
    cp, sp = np.cos(phi), np.sin(phi)
    tot = np.zeros(3)
    acg, abow, us, ua_, ys, psis = [], [], [], [], [], []
    sub = int(h["sub"])
    for k in range(len(S) // sub):
        parts = np.zeros(3)
        for i in range(k * sub, (k + 1) * sub):
            s = S[i]
            a = A[i] / G
            d = s[5] - phi
            ua = s[6] * np.cos(d) - s[7] * np.sin(d)
            yt = -sp * s[0] + cp * s[1]
            parts += cost_parts(ua, a, yt, L, u_ref)
            acg.append(abs(a))
            abow.append(abs(AB[i]) / G)
            us.append(s[6])
            ua_.append(ua)
            ys.append(yt)
            psis.append((s[5] - phi + np.pi) % (2 * np.pi) - np.pi)
        tot += parts
    n = len(acg)
    parts = tot / n
    acg, y = np.asarray(acg), np.asarray(ys)
    return dict(score=float(parts.sum()), c_speed=float(parts[0]),
                c_impact=float(parts[1]), c_track=float(parts[2]),
                u_mean=float(np.mean(us)), u_along=float(np.mean(ua_)),
                acc_cg_p99=float(np.percentile(acg, 99)),
                acc_bow_p99=float(np.percentile(abow, 99)),
                cross_rms=float(np.sqrt(np.mean(y ** 2))),
                heading_rms=float(np.sqrt(np.mean(np.square(psis)))))


def g_torch(full=False):
    C = Checks("torch")
    import torch  # noqa: F401
    from app.env import SafeMode as AppSafe
    from app.policies import quiet_fe
    from app.runner import run
    from app.scenario import straight
    from learn.meta.mpc_constrained import SafeMode as MCSafe
    from learn.repro import task
    rng = np.random.default_rng(0)
    a, b = AppSafe(0.24), MCSafe(0.24)
    roll = np.cumsum(rng.normal(0, 0.05, 5000))
    roll = 0.6 * np.sin(roll)
    outs_a = [a(r, 1000.0 + k, 0.1 * k) for k, r in enumerate(roll)]
    outs_b = [b(r, 1000.0 + k, 0.1 * k) for k, r in enumerate(roll)]
    C.ok("app SafeMode == learn/meta/mpc_constrained.SafeMode (5000 steps)",
         outs_a == outs_b and a.n_entries == b.n_entries and a.n_entries > 0,
         f"{a.n_entries} entries, {a.n_on} steps on")
    from studies import final_eval as FE0
    saved = (FE0.log, FE0.out_dir, dict(FE0.OPT))
    with quiet_fe(plant="gz", device="auto") as FE:
        turn = FE.phase_turn()
    C.ok("quiet_fe puts final_eval's log, out_dir and OPT back on exit",
         (FE0.log, FE0.out_dir, FE0.OPT) == saved)
    # the comparisons below log into the app's cache too (put back at the end)
    quiet = quiet_fe(plant="gz", device="auto")
    FE = quiet.__enter__()
    keys = ("score", "c_speed", "c_impact", "c_track", "u_mean", "u_along",
            "acc_cg_p99", "acc_bow_p99", "cross_rms", "heading_rms")
    for row, seed, leg, T in (("hand", 2, 1, 10.0), ("m0", 1, 0, 5.0)):
        def cmp(row=row, seed=seed, leg=leg, T=T):
            spec, _ = FE.resolve_row(row)
            t0 = time.time()
            ref = FE.run_mission(spec, seed, leg, T, turn, {})
            t_fe = time.time() - t0
            n = int(round(T / 0.25))
            sc = straight("scarab195", hs=task.HS, tp=task.TP,
                          theta0=task.WAVES_FROM, heading=task.LEGS[leg],
                          duration=n * 0.24, seed=seed, plant="gz",
                          safe_mode=True)
            pol = ({"type": "hand", "guidance": "hold"} if row == "hand"
                   else {"type": "m0", "leg_seed": leg})
            t0 = time.time()
            rec = run(sc, pol)
            t_app = time.time() - t0
            got = _mission_metrics(rec, task.LEGS[leg])
            bad = {k: (ref[k], got[k]) for k in keys if ref[k] != got[k]}
            assert not bad, bad
            return (f"{n} steps; score {got['score']:.6f}, u_mean "
                    f"{got['u_mean']:.6f}: all {len(keys)} metrics == "
                    f"run_mission's; {t_fe:.0f} s vs {t_app:.0f} s wall")
        C.run(f"C-P2a {row} through the app == final_eval.run_mission "
              f"(seed {seed}, leg {leg}), metrics to the last bit", cmp)

    def refuse():
        try:
            run(straight("probe_hullB", duration=0.48), {"type": "m0"})
        except ValueError as ex:
            return str(ex)[:120]
        raise AssertionError("m0 ran on probe_hullB")
    C.run("m0 refuses a hull with other actuator scales (probe_hullB)",
          refuse)

    def learned():
        t0 = time.time()
        rec = run("eval_head_seas.json", {"type": "learned",
                                          "row": "cat_p_prior"},
                  duration=0.48)
        s = rec.header["policy"]["summary"]
        assert rec.header["finite"] and len(rec.ctrl["t"]) == 2
        return (f"2 steps, {time.time() - t0:.0f} s wall incl. loading; "
                f"net {s.get('net')}; "
                f"{rec.header['policy']['t_policy_per_step']:.2f} s/step")
    C.run("C-P2a learned row cat_p_prior runs through the runner (2 steps)",
          learned)
    if full:
        def m0_long():
            rec = run("eval_head_seas.json", duration=30.0)
            return rec.summary()
        C.run("full tier: m0 30 s through the runner", m0_long)
    quiet.__exit__(None, None, None)
    return C.result()


# ===================================================================== main
GROUPS = dict(api=g_api, equiv=g_equiv, replay=g_replay, sensors=g_sensors,
              interfaces=g_interfaces, robust=g_robust, resume=g_resume,
              resume_more=g_resume_more, codehash=g_codehash)
E2E = ("demo_scarab195.json", "demo_hullB.json", "demo_scarab195_as1.json",
       "demo_planing_b_as1.json")


def run_group(arg):
    name, full = arg
    warnings.filterwarnings("ignore")
    try:
        if name in GROUPS:
            return GROUPS[name]()
        if name in ("det_a", "det_b"):
            return g_det(name)
        if name.startswith("e2e:"):
            return g_e2e(name[4:])
        if name == "torch":
            return g_torch(full)
        raise KeyError(name)
    except Exception as ex:                          # noqa: BLE001
        return dict(group=name, ok=False, n=1, seconds=0.0, checks=[dict(
            check="group crashed", ok=False,
            detail=f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="quick", choices=("quick", "full"))
    ap.add_argument("--groups", default=None)
    ap.add_argument("--reserve", type=float, default=3.0)
    a = ap.parse_args()
    from studies.parallel_runner import free_gb, run_parallel
    full = a.tier == "full"
    light = list(GROUPS) + ["det_a", "det_b"] + [f"e2e:{f}" for f in E2E]
    heavy = ["torch"]
    if a.groups:
        want = a.groups.split(",")
        light = [g for g in light if g in want or g.split(":")[0] in want]
        heavy = [g for g in heavy if g in want]
    t0 = time.time()
    print(f"free RAM {free_gb():.1f} GB; light groups {light}; torch "
          f"group {heavy}", flush=True)
    res = run_parallel(run_group, [(g, full) for g in light],
                       gb_per_job=0.4, reserve_gb=a.reserve)
    t_light = time.time() - t0
    if heavy:
        res += run_parallel(run_group, [(g, full) for g in heavy],
                            gb_per_job=2.5, reserve_gb=a.reserve)
    out = []
    for g, r in zip(light + heavy, res):
        if isinstance(r, Exception):
            r = dict(group=g, ok=False, n=1, seconds=0.0, checks=[dict(
                check="worker exception", ok=False, detail=repr(r))])
        out.append(r)
    fps = {r["group"]: r.get("fingerprint") for r in out
           if r.get("fingerprint")}
    if {"replay", "det_a", "det_b"} <= set(fps):
        same = len(set(fps[k] for k in ("replay", "det_a", "det_b"))) == 1
        out.append(dict(group="determinism", ok=same, n=1, seconds=0.0,
                        checks=[dict(check="C-P4b(i) the same run in three "
                                     "processes: one fingerprint", ok=same,
                                     detail=str({k: v[:16] for k, v in
                                                 fps.items()}))]))
    wall = time.time() - t0
    n_ok = sum(c["ok"] for r in out for c in r["checks"])
    n_all = sum(len(r["checks"]) for r in out)
    print(f"\n=== app tests ({a.tier} tier): {n_ok}/{n_all} checks pass, "
          f"{wall:.0f} s wall (light groups {t_light:.0f} s) ===")
    for r in out:
        print(f"\n[{'PASS' if r['ok'] else 'FAIL'}] {r['group']} "
              f"({r['seconds']:.0f} s)")
        for c in r["checks"]:
            d = c["detail"].replace("\n", " ")
            print(f"   {'ok ' if c['ok'] else 'BAD'} {c['check']}"
                  + (f" [{c['s']:g} s]" if c.get("s") else "")
                  + (f" -- {d[:400]}" if d else ""))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "test_results.json"), "w") as f:
        json.dump(dict(tier=a.tier, wall_s=wall, n_ok=n_ok, n=n_all,
                       groups=out, when=time.strftime("%Y-%m-%d %H:%M:%S")),
                  f, indent=1, default=str)
    return 0 if n_ok == n_all else 1


if __name__ == "__main__":
    sys.exit(main())
