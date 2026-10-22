"""
Policies the runner can drive the boat with -- any of them through one
interface:

    class Policy:
        action_mode            "direct" or "heading" (array semantics)
        reset(env)             once per episode, after env.reset()
        act(obs, info) -> a    an array (env.action_space) or a dict with
                               physical units (thrust N, nozzle_rad, heading)
        observe(info)          after every env.step (optional)
        summary() -> dict      for the record (optional)

    make_policy(spec)          spec: a dict {"type": ...}, a string
                               ("hand", "m0", "learned:cat_p_prior",
                               "scripted", "pkg.mod:function"), a Policy, or
                               any callable f(obs, info) -> action

TYPES
  scripted  throttle schedule [[t, fraction], ...] (or a constant) and a
            heading: "track" (the track's guidance, default), a constant or a
            schedule in degrees (ENU, CCW from east); or a nozzle schedule
            (fractions) for open-loop steering
  hand      control/mpc.MPPIController with the hand weights
            (studies/sim2real_jet.theta(z_hand())), thrust only, the
            heading autopilot on the track's guidance -- the evaluation's
            `hand` row (learn/repro/task.Mission.mpc_command), on any hull
            (its own identified reduced model)
  m0        learn/meta/mpc_constrained.ConstrainedMPPI with the e = 0 model
  learned   a studies/final_eval.py net row ("cat_p_prior", "gen_p_online",
            ...): its builders (resolve_row, _row_nets, build_net,
            LiveDataW, Monitor, OnlineLearner / BayesLearner) stepped one
            control step at a time, the per-step body of
            final_eval.run_mission
  callable  any f(obs, info) -> action; "pkg.mod:function" imports it

WHAT THE BUILT-IN POLICIES SEE. scripted and hand take state_source:
  "truth"    (default) the plant's state (env.s) and the env's heading
             autopilot on it -- the evaluation's setting; the scenario's
             sensors are recorded but do not reach the loop
  "sensors"  app/estimator.SensorEstimator's estimate from the GNSS, IMU,
             heading sensor and speed log (noise, bias, latency, dropout
             included): guidance, the MPC's state and the policy's own copy
             of the heading autopilot all run on it, and the nozzle is sent
             as a direct command
m0 and learned rows always run on the truth (final_eval's setting).
Callables and external clients read the observation, i.e. the sensors.

m0 and learned rows plan with scarab195's identified model and actuator
scales (learn/meta/mpc_learned.env()); on a hull whose scales differ they
refuse to start (mpc_learned.check_env), and they plan in the evaluation's
straight-track frame, so they need a straight track from the origin.
"""
import importlib
import math

import numpy as np


class Policy:
    action_mode = "direct"
    name = "policy"
    label = None

    def reset(self, env):
        self.env = env

    def act(self, obs, info):
        raise NotImplementedError

    def observe(self, info):
        pass

    def summary(self):
        return {}

    def close(self):
        pass


# ------------------------------------------------------------ helpers
def check_schedule(spec, what):
    """A constant or [[t, value], ...] with finite numbers and increasing
    times (ValueError otherwise); returns it."""
    if spec is None:
        return spec
    if isinstance(spec, bool):
        raise ValueError(f"{what}: a number or [[t, value], ...], not "
                         f"{spec!r}")
    if isinstance(spec, (int, float)):
        if not math.isfinite(spec):
            raise ValueError(f"{what}: a finite number, not {spec!r}")
        return spec
    try:
        a = np.asarray(spec, float)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: a number or [[t, value], ...], not "
                         f"{spec!r}") from None
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 1:
        raise ValueError(f"{what}: [[t, value], ...] pairs, not {spec!r}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{what}: finite numbers, not {spec!r}")
    if np.any(np.diff(a[:, 0]) <= 0):
        raise ValueError(f"{what}: times must increase, not {spec!r}")
    return spec


def schedule(spec, t, default=None):
    """A constant or a piecewise-linear [[t, value], ...] at time t."""
    if spec is None:
        return default
    if isinstance(spec, (int, float)):
        return float(spec)
    a = np.asarray(spec, float).reshape(-1, 2)
    return float(np.interp(t, a[:, 0], a[:, 1]))


def _check_state_source(v):
    if v not in ("truth", "sensors"):
        raise ValueError(f"state_source: 'truth' or 'sensors', not {v!r}")
    return v


def _check_guidance(guidance, lookahead):
    if guidance not in ("los", "hold"):
        raise ValueError(f"guidance: 'los' or 'hold', not {guidance!r}")
    if lookahead is not None and (isinstance(lookahead, bool) or not (
            isinstance(lookahead, (int, float)) and lookahead > 0)):
        raise ValueError(f"lookahead: metres > 0 or null, not {lookahead!r}")


class _OwnAutopilot:
    """state_source 'sensors': the policy's copy of the env's heading
    autopilot (app/models.Autopilot, the same gains), run on the estimate;
    the nozzle goes to the env as a direct command."""

    def setup(self, env):
        from .estimator import SensorEstimator
        from .models import Autopilot
        self.est = SensorEstimator(env)
        p = env.reduced.p
        self.ap = Autopilot(env.plant, env.reduced,
                            float(p.get("u_design", env.u_ref)),
                            env.dt_ctrl)

    def steer(self, s_est, thrust, heading):
        return self.ap(s_est, thrust, heading)


class TrackPreview:
    """control/mpc.PreviewProvider seen from a track leg's frame (origin
    ox, oy, angle phi): the MPC plans with its leg along +x, the sea stays
    where it is. For a leg through the origin it is task.RotPreview."""

    def __init__(self, pv, ox, oy, phi):
        self.pv, self.ox, self.oy = pv, float(ox), float(oy)
        self.sea, self.t_preview = pv.sea, pv.t_preview
        self.c, self.s = np.cos(phi), np.sin(phi)

    def at(self, x, y, t, t_ahead):
        return self.pv.at(self.ox + self.c * x - self.s * y,
                          self.oy + self.s * x + self.c * y, t, t_ahead)


# ------------------------------------------------------------ scripted
class ScriptedPolicy(Policy, _OwnAutopilot):
    name = "scripted"

    def __init__(self, throttle=0.8, heading="track", nozzle=None,
                 guidance="los", lookahead=None, state_source="truth"):
        check_schedule(throttle, "throttle")
        if throttle is None:
            raise ValueError("throttle: a number or [[t, fraction], ...]")
        if heading != "track":
            check_schedule(heading, "heading (deg, or 'track')")
        check_schedule(nozzle, "nozzle (fraction)")
        _check_guidance(guidance, lookahead)
        self.state_source = _check_state_source(state_source)
        self.throttle, self.heading, self.nozzle = throttle, heading, nozzle
        self.guidance, self.lookahead = guidance, lookahead
        self.action_mode = "direct" if nozzle is not None else "heading"
        self.label = "scripted" + (" (sensors)" if state_source == "sensors"
                                   else "")

    def reset(self, env):
        self.env = env
        if self.state_source == "sensors" and self.nozzle is None:
            self.setup(env)

    def act(self, obs, info):
        env, t = self.env, info["t"]
        thr = schedule(self.throttle, t)
        if self.nozzle is not None:
            return {"throttle": thr, "nozzle": schedule(self.nozzle, t)}
        sensed = self.state_source == "sensors"
        s = self.est.update() if sensed else env.s
        if self.heading == "track":
            h = env.track.guidance(s[0], s[1], self.guidance,
                                   self.lookahead, env.L)
        else:
            h = math.radians(schedule(self.heading, t))
        if sensed:
            # the thrust env._command makes of a throttle, which is what
            # its autopilot is scheduled on
            thrust = float(np.clip(float(thr) * env.t_max, env.t_min,
                                   env.t_max))
            return {"thrust": thrust,
                    "nozzle_rad": self.steer(s, thrust, h)}
        return {"throttle": thr, "heading": h}

    def summary(self):
        return dict(throttle=self.throttle, heading=self.heading,
                    nozzle=self.nozzle, guidance=self.guidance,
                    state_source=self.state_source)


# ------------------------------------------------------------ hand MPC
class HandMPCPolicy(Policy, _OwnAutopilot):
    """learn/repro/task.Mission.mpc_command on any hull and any track.

    frame -- which frame the MPC plans its thrust in:
      "leg"      the current leg's track frame (x along the leg, y its
                 cross-track, heading minus the leg angle): exactly Mission,
                 and the default
      "heading"  a frame through the boat along its own heading, so the
                 plan starts on its line with zero heading error. The MPC
                 here plans thrust only and its rollouts hold the nozzle at
                 zero (use_rudder=False), so they cannot know that the
                 autopilot is turning the boat: after a course change the
                 leg frame shows a large heading error, the rollouts
                 predict the cross-track error growing with speed, and the
                 plan cuts the thrust -- which takes the jet's steering
                 authority with it (seen on probe_hullB at a 45 deg turn:
                 throttle ~1 %, nozzle hard over, 6 kn). In the heading
                 frame the MPC schedules thrust for speed and seakeeping
                 only and leaves the course to the autopilot, the division
                 of work sim/env.Episode describes. On a straight track
                 with a small heading error the two frames plan almost the
                 same thrust.
    """
    name = "hand"
    action_mode = "heading"

    def __init__(self, guidance="los", lookahead=None, weights=None,
                 floor=None, n_samples=128, seed=None, t_preview=0.0,
                 frame="leg", time_ref=None, state_source="truth"):
        if frame not in ("leg", "heading"):
            raise ValueError("frame: 'leg' (Mission's) or 'heading'")
        _check_guidance(guidance, lookahead)
        self.state_source = _check_state_source(state_source)
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) \
                or n_samples < 1:
            raise ValueError(f"n_samples: an integer >= 1, not "
                             f"{n_samples!r}")
        if time_ref not in (None, "froude") and not (
                isinstance(time_ref, (int, float)) and time_ref > 0):
            raise ValueError(f"time_ref: null, 'froude' or seconds > 0, not "
                             f"{time_ref!r}")
        self.frame = frame
        parts = (["heading frame"] if frame == "heading" else []) + \
            (["sensors"] if state_source == "sensors" else [])
        self.label = "hand" + (f" ({', '.join(parts)})" if parts else "")
        # control/mpc.py's time-scale-free objective (None = v0's, as
        # Mission; "froude" = froude_time_ref(L) of the hull; or seconds)
        self.time_ref = time_ref
        self.guidance, self.lookahead = guidance, lookahead
        self.weights, self.floor = weights, floor
        self.n_samples, self.seed, self.t_preview = n_samples, seed, \
            t_preview

    def reset(self, env):
        from control.mpc import MPPIController, PreviewProvider
        from studies.sim2real_jet import theta, z_hand
        self.env = env
        w, floor = theta(z_hand())
        if self.weights is not None:
            w = np.asarray(self.weights, float)
        if self.floor is not None:
            floor = float(self.floor)
        red = env.reduced
        p = red.p
        seed = env.seed_ if self.seed is None else int(self.seed)
        # as sim/env.Episode builds them (Mission: n_samples 128, no
        # rudder, the hand floor)
        self.pv0 = PreviewProvider(env.sea, self.t_preview, 0.0, seed)
        kw = {}
        if self.time_ref is not None:
            tr = self.time_ref
            if tr == "froude":
                from control.mpc import froude_time_ref
                tr = froude_time_ref(float(p["L"]))
            kw["time_ref"] = float(tr)
        self.ctrl = MPPIController(red, self.pv0, weights=w,
                                   dt_ctrl=env.dt_ctrl, u_ref=p["u_design"],
                                   seed=seed, n_samples=self.n_samples,
                                   use_rudder=False, thrust_floor=floor,
                                   **kw)
        self.red = red
        self._leg = None
        env.ensure_autopilot()
        if self.state_source == "sensors":
            self.setup(env)

    def _preview(self):
        tr = self.env.track
        key = (tr.leg, tr.ox, tr.oy, tr.phi)
        if key != self._leg:
            self._leg = key
            if tr.ox == 0.0 and tr.oy == 0.0:
                if tr.phi == 0.0:
                    self.ctrl.pv = self.pv0
                else:
                    from learn.repro.task import RotPreview
                    self.ctrl.pv = RotPreview(self.pv0, tr.phi)
            else:
                self.ctrl.pv = TrackPreview(self.pv0, tr.ox, tr.oy, tr.phi)

    def act(self, obs, info):
        env = self.env
        sensed = self.state_source == "sensors"
        s = self.est.update() if sensed else env.s
        if self.frame == "heading":
            self.ctrl.pv = TrackPreview(self.pv0, s[0], s[1], s[5])
            sr = self.red.from_plant_state(s)
            sr[0], sr[1], sr[7] = 0.0, 0.0, 0.0
        else:
            self._preview()
            sr = env.track_state(self.red, None if not sensed else s)
        cmd = self.ctrl(sr, env.t)
        thrust, _ = self.ctrl.to_actuator(cmd)
        h = env.track.guidance(s[0], s[1], self.guidance,
                               self.lookahead, env.L)
        if sensed:
            return {"thrust": thrust, "nozzle_rad": self.steer(s, thrust, h)}
        return {"thrust": thrust, "heading": h}

    def summary(self):
        return dict(weights=list(map(float, self.ctrl.w)),
                    floor=float(self.ctrl.thrust_floor),
                    n_samples=self.n_samples, guidance=self.guidance,
                    frame=self.frame, time_ref=self.time_ref,
                    state_source=self.state_source)


# ------------------------------------------------------------ m0 / learned
class MissionView:
    """learn/repro/task.Mission's attributes over a UsvEnv: what
    studies/final_eval and learn/meta/mpc_constrained read from a mission."""

    def __init__(self, env):
        self.env = env
        env.ensure_autopilot()

    s = property(lambda self: self.env.s)
    t = property(lambda self: self.env.t)
    phi = property(lambda self: float(self.env.track.phi))
    ep = property(lambda self: self.env.autopilot)
    red = property(lambda self: self.env.reduced)
    plant = property(lambda self: self.env.plant)
    sea = property(lambda self: self.env.sea)
    L = property(lambda self: self.env.L)
    u_ref = property(lambda self: float(self.env.reduced.p["u_design"]))
    t_max = property(lambda self: self.env.t_max)
    rud_max = property(lambda self: self.env.rud_max)
    dt = property(lambda self: self.env.dt)
    sub = property(lambda self: self.env.sub)
    dt_ctrl = property(lambda self: self.env.dt_ctrl)
    n_ctrl = property(lambda self: self.env.n_ctrl)
    theta0 = property(lambda self: float(self.env.sea.theta0))
    finite = property(lambda self: self.env.finite)


class quiet_fe:
    """with quiet_fe(**opt) as FE: studies/final_eval with its log and
    output directory redirected to the app's cache and FE.OPT updated with
    opt -- all three put back on exit, so nothing else in the process that
    calls final_eval logs into the app's cache or sees the app's options.
    (Only the set-up of a row reads them: the per-step helpers the policy
    calls afterwards, bow_stats and step_safety, read neither.)"""

    def __init__(self, **opt):
        self.opt = opt

    def __enter__(self):
        import os
        from studies import final_eval as FE
        from . import CACHE
        d = os.path.join(CACHE, "fe_log")

        def log(msg):
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "run.log"), "a",
                      encoding="utf-8") as f:
                f.write(msg + "\n")
        self.FE = FE
        self.saved = (FE.log, FE.out_dir, dict(FE.OPT))
        FE.log = log
        FE.out_dir = lambda: (os.makedirs(d, exist_ok=True), d)[1]
        FE.OPT.update(self.opt)
        return FE

    def __exit__(self, *exc):
        FE = self.FE
        FE.log, FE.out_dir = self.saved[0], self.saved[1]
        FE.OPT.clear()
        FE.OPT.update(self.saved[2])
        return False


class ConstrainedPolicy(Policy):
    """m0 or a learned row of studies/final_eval.py, one control step at a
    time: act() is the part of run_mission's loop before m.advance, and
    observe() the part after it."""
    action_mode = "heading"

    def __init__(self, row="m0", steer="auto", K=128, S=8, y_lengths=2.0,
                 fc_lam=0.1, fc_hp=24, trigger=True, device="auto",
                 leg_seed=0):
        self.row, self.steer = row, steer
        self.K, self.S, self.y_lengths = K, S, y_lengths
        self.fc_lam, self.fc_hp, self.trigger = fc_lam, fc_hp, trigger
        self.device, self.leg_seed = device, int(leg_seed)
        if not isinstance(row, str) or not row:
            raise ValueError(f"row: a final_eval row name, not {row!r}")
        if steer not in ("auto", "free"):
            raise ValueError(f"steer: 'auto' or 'free', not {steer!r}")
        self.name = row
        self.label = row
        if steer == "free":
            self.action_mode = "direct"

    def reset(self, env):
        import torch
        self.torch = torch
        with quiet_fe(device=self.device, fc_lam=self.fc_lam,
                      fc_hp=self.fc_hp, steer=self.steer, K=self.K,
                      S=self.S, y_lengths=self.y_lengths,
                      trigger=self.trigger) as FE:
            self._reset(env, FE)

    def _reset(self, env, FE):
        import torch
        from learn.meta import mpc_constrained as MC
        from learn.meta import mpc_learned as ML
        self.env = env
        tr = env.track
        if not tr.line or tr.ox != 0.0 or tr.oy != 0.0:
            raise ValueError(f"{self.row}: plans in the evaluation's "
                             f"straight-track frame; give a straight track "
                             f"from the origin (waypoints: use 'hand')")
        spec, why = FE.resolve_row(self.row)
        if spec is None:
            raise ValueError(f"row {self.row!r}: {why}")
        if spec.get("carry"):
            raise ValueError("carry rows chain missions; run them through "
                             "studies/final_eval.py")
        self.spec = spec
        en = ML.env()
        m = MissionView(env)
        try:
            ML.check_env(m, en)
        except AssertionError as ex:
            raise ValueError(
                f"{self.row} plans with scarab195's model and actuator "
                f"scales (control step, t_max, nozzle limit); "
                f"{env.hull_key} differs: {ex}") from None
        self.m, self.en = m, en
        self.geo = FE.geometry(m, en)
        self.rec = FE.Recorder(m, en)
        env.episode_hooks.append(self.rec)
        self.dtc = m.dt * m.sub
        self.lim = MC.Limits(y_lengths=self.y_lengths)
        from .models import turn_table
        self.turn = turn_table(env.hull, env.plant_kind)
        seed, leg = env.seed_, self.leg_seed
        jobs = [dict(seed=seed, leg=leg)]
        net = head = live = None
        self.mon = self.learner = None
        variant, nh = "a", None
        self.net_info = {}
        if spec["kind"] == "net":
            dev = FE.device()
            ck, head0 = FE._row_nets(spec)[spec["name"]]
            net, head = FE.build_net(spec, ck, head0, dev, seed, leg)
            variant = spec["variant"]
            ctx = int(ck.get("ctx", 256))
            nh = ctx - ML.M.HB
            live = MC.LiveDataW(1, m.n_ctrl + 1, ck["stats"], dev,
                                ML.mission_consts(m), [m.sea], en)
            live.start(0, m.s, m.t)
            fc = dict(lam=self.fc_lam, hp=self.fc_hp, msd=0.01)
            if spec["mode"] == "bayes":
                from learn.meta.bayes_adapter import BayesLearner
                self.learner = BayesLearner(net, live, 0, ctx, variant,
                                            head=head,
                                            seed=ML.row_seed(seed, leg, 17),
                                            fc=fc)
            elif spec["mode"] in ("online", "scratch"):
                from learn.meta.online_stream import OnlineLearner
                self.learner = OnlineLearner(
                    net, live, 0, ctx, variant, head=head,
                    seed=ML.row_seed(seed, leg, 17),
                    scratch=spec["mode"] == "scratch")
            self.mon = FE.MC_monitor(net, live, ctx, variant, head, fc, seed,
                                     leg)
            self.net_info = dict(ctx=ctx, variant=variant,
                                 head=head is not None, mode=spec["mode"])
        self.live = live
        self.ctrl = MC.ConstrainedMPPI(
            jobs, [m], en, self.lim, self.geo, net=net, live=live, head=head,
            variant=variant, nh=nh, K=self.K, S=self.S,
            fc=dict(lam=self.fc_lam, hp=self.fc_hp), steer=self.steer)
        self.past = []
        self.n_win = int(round(self.lim.win_s / self.dtc))
        self.k = 0
        self.torch = torch

    def act(self, obs, info):
        from learn.meta import mpc_constrained as MC
        from studies import final_eval as FE
        m = self.m
        ci = dict(past=np.asarray(self.past[-self.n_win:]) / 9.81,
                  u_meas=float(m.s[6]),
                  r_max=MC.r_max_of(self.turn, float(m.s[6])),
                  bow=FE.bow_stats(self.rec, self.lim.win_s),
                  infl=self.mon.infl if (self.mon is not None
                                         and self.trigger) else 1.0)
        cmds, dg = self.ctrl([0], self.k, [ci])
        thrust = float(cmds[0]) * m.t_max
        self._s0 = m.s.copy()
        self._n0 = len(self.rec.S)
        if self.steer == "free":
            return {"thrust": thrust,
                    "nozzle_rad": dg[0]["nozzle"] * m.rud_max}
        return {"thrust": thrust, "heading": m.phi}

    def observe(self, info):
        from studies import final_eval as FE
        env, m = self.env, self.m
        c = env.last_cmd
        u_app = (c["thrust"] / m.t_max, c["nozzle"] / m.rud_max)
        pred = self.mon.predict(self.k, u_app) if self.mon is not None \
            else None
        if not env.finite or env.capsized:
            return
        apk, hmin = FE.step_safety(self.rec, self._n0, self._s0, self.geo,
                                   m.sub)
        self.past.append(apk)
        if self.live is not None:
            self.live.push(0, u_app, m.s, m.t, apk, hmin)
            if self.mon is not None:
                self.mon.update(pred, self.k)
            if self.learner is not None:
                self.learner.step(self.k)
        self.k += 1

    def summary(self):
        out = dict(row=self.row, steer=self.steer, K=self.K, S=self.S,
                   y_lengths=self.y_lengths, net=self.net_info,
                   n_bad=int(self.ctrl.n_bad[0]),
                   n_samp=int(self.ctrl.n_samp[0]))
        if self.mon is not None:
            out["monitor"] = self.mon.summary()
        if self.learner is not None:
            out["learner"] = self.learner.summary()
        return out

    def close(self):
        torch = getattr(self, "torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


# ------------------------------------------------------------ callables
class CallablePolicy(Policy):
    name = "callable"

    def __init__(self, fn, action_mode="direct", name=None):
        self.fn, self.action_mode = fn, action_mode
        self.name = name or getattr(fn, "__name__", "callable")

    def act(self, obs, info):
        return self.fn(obs, info)

    def summary(self):
        return dict(target=self.name)


def _import(target):
    mod, _, attr = target.partition(":")
    obj = importlib.import_module(mod)
    for a in attr.split("."):
        obj = getattr(obj, a)
    return obj


def make_policy(spec):
    """A Policy from a spec (module docstring)."""
    if spec is None:
        spec = {"type": "scripted"}
    if isinstance(spec, Policy):
        return spec
    if not isinstance(spec, (str, dict)) and hasattr(spec, "act") and \
            hasattr(spec, "reset"):
        return spec                       # any object with the interface
    if callable(spec) and not isinstance(spec, (str, dict)):
        return CallablePolicy(spec)
    if not isinstance(spec, (str, dict)):
        raise ValueError(f"policy: a name, a {{'type': ...}} object, a "
                         f"Policy or a callable, not {spec!r}")
    if isinstance(spec, str):
        if spec.startswith("learned:"):
            spec = {"type": "learned", "row": spec.split(":", 1)[1]}
        elif ":" in spec:
            spec = {"type": "callable", "target": spec}
        else:
            spec = {"type": spec}
    spec = dict(spec)
    kind = spec.pop("type", "scripted")
    if kind == "scripted":
        return ScriptedPolicy(**spec)
    if kind == "hand":
        return HandMPCPolicy(**spec)
    if kind == "m0":
        if "row" in spec:
            raise ValueError("m0: no 'row' (use type 'learned' for a net "
                             "row)")
        return ConstrainedPolicy(row="m0", **spec)
    if kind == "learned":
        if "row" not in spec:
            raise ValueError("learned: give the final_eval row, e.g. "
                             "{'type': 'learned', 'row': 'cat_p_prior'}")
        return ConstrainedPolicy(**spec)
    if kind == "callable":
        target = spec.pop("target")
        obj = _import(target)
        if isinstance(obj, type):
            obj = obj(**spec.pop("kwargs", {}))
        if hasattr(obj, "act") and hasattr(obj, "reset"):
            return obj
        return CallablePolicy(obj, action_mode=spec.get("action_mode",
                                                        "direct"),
                              name=target)
    raise ValueError(f"policy type {kind!r}: scripted | hand | m0 | "
                     f"learned | callable")


def label(spec):
    """A short human label of a policy spec (viewer HUD, run ids): the
    policy's own label when it has one, e.g. 'hand (heading frame)'."""
    try:
        pol = make_policy(spec) if not (isinstance(spec, dict) and
                                        spec.get("type") == "callable") \
            and not (isinstance(spec, str) and ":" in spec and
                     not spec.startswith("learned:")) else None
    except Exception:                                # noqa: BLE001
        pol = None
    if pol is not None:
        return getattr(pol, "label", None) or getattr(pol, "name", None) \
            or type(pol).__name__
    if isinstance(spec, dict):
        return str(spec.get("target") or spec.get("type") or "policy")
    if isinstance(spec, str):
        return spec
    return getattr(spec, "__name__", type(spec).__name__)


def spec_of(spec):
    """A JSON-able description of a policy spec (for records)."""
    if spec is None or isinstance(spec, (str, dict)):
        return spec
    if isinstance(spec, Policy):
        return {"type": getattr(spec, "name", type(spec).__name__),
                "class": type(spec).__name__}
    return {"type": "callable", "target": getattr(spec, "__qualname__",
                                                  repr(spec))}
