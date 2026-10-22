"""
UsvEnv: the project's planing plant behind a Gymnasium-style interface.

    env = UsvEnv("app/scenarios/demo_scarab195.json")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step([0.8, 0.0])

WHAT ONE STEP IS. One control step: `sub` plant steps of `dt`, sub =
round(dt_ctrl / dt) from Hull.scales() (12 x 0.02 s = 0.24 s for scarab195,
exactly learn/repro/task.Mission's loop). The plant is the hull key's
physics version (sim/physics.py): 'gz' builds the large-angle twin that the
final evaluation runs (physics.gz_twin of the hull's plant), 'lin' the
hull's own plant (what Mission steps). The sea is sim/wavefield.SeaState
with the scenario's spectrum and seed.

SEEDS (Gymnasium's rule)
  reset(seed=s) builds the episode from s. The first reset without a seed
  uses the scenario's usv.seed. A later reset() without a seed draws the
  episode's seed from self.np_random (seeded by the last reset that was
  given one, or by the scenario's seed), so an RL loop that calls reset()
  sees a new sea every episode, and the whole sequence is reproducible from
  the first seed. The episode's seed is info["seed"] and the record's
  header["seed"]; replay rebuilds the episode from it.

ACTIONS
  array, action_mode "direct":  [throttle, nozzle], fractions of t_max (down
                                to t_min / t_max = -1/3, reverse) and of full
                                nozzle lock
  array, action_mode "heading": [throttle, heading], heading ENU rad; the
                                project's heading autopilot
                                (sim/env.Episode._steer via app/models)
                                sets the nozzle
  dict (any mode): thrust [N] or throttle; nozzle_rad, nozzle (fraction) or
                   heading [rad] -- physical units pass through unchanged,
                   which keeps the model-based policies bit-exact
  A value that is not a finite number (NaN, inf) raises ValueError("non-
  finite action ...") before anything is stepped: the episode stays as it
  was (a NaN must not read as a physics divergence).
OBSERVATION  Dict of sensors (app/sensors.py), each [valid, age, values...];
             plus "truth" = [t, state(14)] when usv.obs_truth is set.
REWARD       minus learn/repro/task's operator cost (speed made good along
             the current leg, CG impacts above A_LIM, cross-track per
             Y_SCALE L), the mean over the control step's plant steps --
             Mission.advance's number on a straight track.
END          terminated: capsize (GZ plant), a non-finite state, or the last
             waypoint reached; truncated: the scenario's duration.
SAFE MODE    usv.safe_mode: final_eval's measured-roll protection (|roll| >
             20 deg: throttle halved, nozzle centred; out after 2 s below
             10 deg), applied after the policy and the autopilot.

The env records every plant step, control step and sensor sample
(record=True); env.record() returns the app.record.Record of the episode.

RESUMING FROM A RECORDED ROW (ACCEPTANCE P4; app/plantstate.py)
  env.resume(rec, k)  -- or reset(options={"record": rec, "row": k}) --
  puts the episode where `rec` was after its control step k (0 = the
  start): the same scenario, seed and sea (checked), the plant at its row
  (state, the physics' hidden state -- plant.hidden_state(), e.g. the
  wagner_entry flow-state field -- and v0's carried actuator state, from
  the record's plant['hidden'] and plant['aux']), the sensors, the track
  and the reward replayed on the recorded rows (no plant step), the env's
  autopilot and safe mode from ctrl['ap'] and ctrl['safe_state']. The
  record lists hold the rows before k, so the episode's record is whole.
  Fed the recorded commands from there, every later row comes back bit for
  bit (app.record.replay(rec, start=k)); a policy whose action depends only
  on (t, state, track) -- scripted -- continues the original run bit for
  bit (app.runner.run(..., resume=(rec, k))). A policy's own memory (an
  MPC's warm start, a learner) is not in a record: it starts fresh.
  A record made by other physics code, scenario or test switches is
  refused (record.RecordMismatch, naming the difference).

TEST SWITCHES  UsvEnv(..., test_switches=(...)) builds an as plant on
  switches other than its version's preset (sim/planing_anyspeed
  test_switches=; for tests such as a wagner_entry plant). Never from a
  scenario file: a product run is its hull key's preset. The record's
  header carries them and its stamp says them; replay and resume rebuild
  with them.

CODE ON DISK  Building an env loads every version's physics code with its
  hash (app/codeguard.load_all), so a record's stamp names the code that
  ran. If the files on disk changed after this process loaded them, the
  env warns once (RuntimeWarning); UsvEnv(..., refuse_stale=True) raises
  codeguard.StaleCode instead (the server's sessions do).

FROZEN CONFIGURATION (2026-10-09, learn/meta/SIM_FINAL_2026-10-09.md;
  app/frozen.py). The scenario's model defaults to 'scarab195@as1' on the
  large-angle ('gz') plant; an as-version hull steps at 0.01 s with a
  0.25 s control step (25 sub-steps) unless the scenario's usv.dt /
  usv.dt_ctrl (or UsvEnv(dt=, dt_ctrl=)) say otherwise; a v0 hull keeps its
  own steps (0.02 s / 12), so every v0 result reproduces as before. Every
  record and info carries sim.physics.stamp(plant) ('physics'); the header
  also carries the nominal key's stamp ('physics_nominal') and the
  randomisation draw ('randomization'); header['frozen'] says whether the
  episode ran the frozen configuration exactly.

RANDOMISATION (app/randomize.py; SIM_FINAL section 5)
  UsvEnv(..., randomization=Randomization(seed=7)) -- or the scenario's
  usv.randomization dict -- holds the ranges and the enable flags;
  reset(options={"randomize": True}) draws one sample for the episode and
  applies it: hull fields -> a renamed copy of the hull on the same physics
  version, the plant built for it (about 9 s for a hull the process has
  not seen: steady attitude at two speeds and the righting-arm table;
  milliseconds from the plant cache, below); sea parameters -> the
  SeaState (and the short-wave component amplitudes); the thrust gain ->
  the commanded thrust before the jet; the output scales -> the
  observation and the record's acc_obs only.
  reset(options={"randomize": draw_dict}) applies a given draw (replay);
  reset(options={"randomization": cfg}) replaces the config for this and
  later episodes. The draw is info["randomization"] and header
  ["randomization"]; the controller-side models (reduced model, autopilot,
  action scaling) stay the NOMINAL hull's. Train / eval seed streams: the
  config's `stream` enters the draw's RNG, and in the 'train' stream the
  automatic episode seeds (reset() without a seed) never return one of
  learn/repro/task.EVAL_SEEDS.
  hull_every and the plant cache (2026-10-10; app/randomize.py, HULL_EVERY
  and PLANT CACHE): with Randomization.hull_every = N the hull fields are
  redrawn (and the plant rebuilt) only on every N-th randomised episode of
  this env, the sea / gain / output fields every episode, the stream
  unchanged; info and header 'randomization_hull_from' say whose hull draw
  is in force. Plants built for randomised hulls are kept in an LRU
  (Randomization.plant_cache entries, default 8) keyed by the hull-draw
  tuple; the cached prototype is pristine and every episode gets a fresh
  plant from it (with_sea), so a repeated draw (hull_every, an evaluation
  set, a replay) costs milliseconds; header 'plant_cache' says whether the
  episode's plant was 'nominal', 'built' or from the 'cache'.

OUTPUT SPEC (app/accel.py; SIM_FINAL section 6)
  obs["accel"] = [a_cg, a_bow] in g: the CG and bow-gauge (0.1 L aft of the
  stem, interpolated from the CG and the plant's bow station) vertical
  accelerations through a causal 2nd-order Butterworth low-pass at
  usv.filter_hz (default 20 Hz full scale), times the acc_scale draw. The
  record keeps the raw m/s^2 (plant a_cg, a_bow) for offline statistics
  (zero-phase filter, app.accel.record_stats) and what the observation saw
  (plant acc_obs). The reward uses the raw CG acceleration, as Mission.
"""
import copy
import math
import os
import platform
import subprocess
import time
import warnings

import numpy as np

from . import ROOT, geo, hulls, sensors as sensors_mod
from . import accel as accel_mod, frozen, randomize as rz_mod
from .scenario import load as load_scenario
from .track import Track

try:
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:                                  # pragma: no cover
    gym, spaces, _Base = None, None, object

G = 9.81


class ActionError(ValueError):
    """step() refused the action (malformed or not finite); nothing was
    stepped and the episode goes on."""


class SafeMode:
    """learn/meta/mpc_constrained.SafeMode, the same rule and numbers
    (studies/app_test.py checks the two give the same output on random roll
    sequences). Kept here because importing mpc_constrained imports torch,
    which a plant-only product process should not need."""

    def __init__(self, dt_ctrl, enter_deg=20.0, exit_deg=10.0, hold_s=2.0):
        self.dt, self.enter = dt_ctrl, math.radians(enter_deg)
        self.exit, self.hold = math.radians(exit_deg), hold_s
        self.on, self.calm, self.n_on, self.n_entries = False, 0.0, 0, 0

    def __call__(self, roll, thrust, rud):
        a = abs(float(roll))
        if not self.on and a > self.enter:
            self.on, self.calm = True, 0.0
            self.n_entries += 1
        elif self.on:
            self.calm = self.calm + self.dt if a < self.exit else 0.0
            if self.calm >= self.hold:
                self.on = False
        if self.on:
            self.n_on += 1
            return 0.5 * thrust, 0.0, True
        return thrust, rud, False


def git_version():
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain",
                                "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5)
        return dict(commit=rev.stdout.strip(),
                    dirty=bool(dirty.stdout.strip()))
    except Exception:                                # noqa: BLE001
        return dict(commit=None, dirty=None)


_GIT = {}


class UsvEnv(_Base):
    metadata = {"render_modes": []}

    def __init__(self, scenario="demo_scarab195.json", *, seed=None,
                 duration=None, model=None, action_mode=None, record=True,
                 plant_hooks=None, test_switches=None, refuse_stale=False,
                 randomization=None, dt=None, dt_ctrl=None):
        sc = load_scenario(scenario)
        if seed is not None or duration is not None or model is not None:
            sc = sc.with_overrides(seed=seed, duration=duration, model=model)
        self.scenario = sc
        self.hull = hulls.resolve(sc.hull_spec)
        self._base = None                 # the v0 hull the copies start from
        if not hulls.is_planing(self.hull):
            raise NotImplementedError(
                f"{self.hull.name!r} is not a planing hull: the product layer "
                f"runs the planing plant and its physics versions")
        ok, why = hulls.runnable_hull(self.hull)
        if not ok:
            raise ValueError(f"hull {self.hull.name!r} cannot run: {why}")
        # load every version's physics code with its hash now (a notebook
        # that imported sim.physics long ago is told if the disk moved on)
        from . import codeguard
        from sim import physics as _physics
        codeguard.check(_physics.version(self.hull), refuse=refuse_stale)
        self.hull_key = self.hull.name
        self.test_switches = (None if test_switches is None
                              else tuple(test_switches))
        scl = self.hull.scales()
        u = sc.usv
        # steps: the caller's, else the scenario's, else the frozen
        # configuration's for an as version / the hull's own for v0
        dt0, dtc0 = frozen.default_steps(self.hull)
        self.dt = float(dt if dt is not None else
                        u["dt"] if u.get("dt") is not None else dt0)
        self.dt_ctrl = float(dt_ctrl if dt_ctrl is not None else
                             u["dt_ctrl"] if u.get("dt_ctrl") is not None
                             else dtc0)                # nominal (Episode's)
        self.sub = max(int(round(self.dt_ctrl / self.dt)), 1)
        self.dt_step = self.dt * self.sub           # the actual control step
        self.u_ref = float(scl["u_design"])
        self.plant_kind = sc.agent["plant"]
        self.action_mode = action_mode or sc.agent["action_mode"]
        self.frame = sc.frame
        self.n_ctrl = int(math.ceil(sc.duration / self.dt_step - 1e-9))
        self.record_on = bool(record)
        self.user_hooks = list(plant_hooks or [])
        self.episode_hooks = []
        self._reduced = None
        self.autopilot = None
        # the output filter (app/accel.py) and the randomisation config
        self.filter_hz = float(u["filter_hz"] if u.get("filter_hz")
                               is not None else frozen.FILTER_HZ)
        self.rz = rz_mod.Randomization.of(
            randomization if randomization is not None
            else u.get("randomization"))
        self.draw = None
        self.thrust_gain, self.acc_scale, self.motion_scale = 1.0, 1.0, 1.0
        self.hull_applied, self.hull_unapplied = {}, []
        self.sea_params, self.n_short_scaled, self.t_build = None, 0, 0.0
        # the plant cache and the hull draw in force (module docstring,
        # RANDOMISATION; app/randomize.py HULL_EVERY, PLANT CACHE)
        self._plants = rz_mod.PlantCache(self.rz.plant_cache)
        self._hull_inforce, self._n_rz = None, 0
        self.hull_from, self.plant_source = None, "nominal"
        # constants from a calm-water plant (the sea comes with reset)
        from sim.test_vessel import Monochromatic
        self._build_plant(Monochromatic(1.0, 0.0))
        p = self.plant
        self.L = float(p.L)
        self.t_max = float(p.prop.t_max)
        self.t_min = float(getattr(p.prop, "t_min", -self.t_max / 3.0))
        self.rud_max = float(p.rudder.max)
        self.x_stem = float(self.hull.L - self.hull.lcg)
        self.x_transom = float(-self.hull.lcg)
        self.x_gauge = accel_mod.gauge_x(self.hull)
        self.x_station = float(p.x[-1])
        self.frozen = frozen.is_frozen(self.hull_key, self.plant_kind,
                                       self.dt, self.dt_ctrl)
        self.sensor_cfg = sc.sensor_config()
        self.sensors = sensors_mod.build(self.sensor_cfg, self)
        lo = self.t_min / self.t_max
        if spaces is not None:
            if self.action_mode == "direct":
                self.action_space = spaces.Box(
                    np.array([lo, -1.0]), np.array([1.0, 1.0]),
                    dtype=np.float64)
            else:
                self.action_space = spaces.Box(
                    np.array([lo, -math.pi]), np.array([1.0, math.pi]),
                    dtype=np.float64)
            d = {n: spaces.Box(-np.inf, np.inf, (2 + len(s.channels),),
                               dtype=np.float64)
                 for n, s in self.sensors.items()}
            # the output spec's filtered CG and bow accelerations, g
            d["accel"] = spaces.Box(-np.inf, np.inf, (2,), dtype=np.float64)
            if sc.usv.get("obs_truth"):
                d["truth"] = spaces.Box(-np.inf, np.inf, (15,),
                                        dtype=np.float64)
            self.observation_space = spaces.Dict(d)
        self.s = None
        self.t = 0.0
        self.seed_ = None
        self.error = None
        self.resumed_from = None
        self._n_resets = 0
        self.acc_obs = np.zeros(2)
        self.filt = None
        if gym is None:
            self.np_random = None

    # ------------------------------------------------------------ build
    def _build_plant(self, sea, hull=None):
        """The plant for `hull` (default: the nominal hull of the key) in
        `sea`: the hull's version through its own plant() dispatch, the
        large-angle twin through physics.gz_twin for plant kind 'gz'."""
        from sim import config, physics
        hull = self.hull if hull is None else hull
        kw = {} if self.test_switches is None else {
            physics.TEST_KW: self.test_switches}
        base = config.plant_for(None, sea, hull, dt=self.dt,
                                fidelity="high", **kw)
        if self.plant_kind == "gz":
            rec = bool(self.scenario.usv.get("record_plant", False))
            plant = physics.gz_twin(base, record=rec)
        else:
            plant = base
        physics.verify(plant, hull,
                       allow_test_switches=self.test_switches is not None)
        self.plant = plant
        return plant

    def base_hull(self):
        """The v0 hull the nominal hull was made from (the start of a
        randomised copy): the registered / extra / inline hull of the key's
        base name, or the nominal hull itself on v0."""
        if self._base is None:
            from sim import physics
            base, ver = physics.split(self.hull.name)
            spec = self.scenario.hull_spec
            if ver == physics.DEFAULT:
                self._base = self.hull
            elif isinstance(spec, str):
                self._base = hulls.resolve(base)
            elif isinstance(spec, dict):
                self._base = hulls.resolve({**spec, "name": base})
            else:
                h = copy.deepcopy(self.hull)
                h.name = base
                if "physics" in vars(h):
                    del h.physics
                self._base = h
        return self._base

    def _episode_hull(self, draw):
        """(hull, applied, unapplied) for the episode: the nominal hull when
        the draw has no hull field, else a renamed copy on the key's
        version (app/randomize.hull_for). A fresh copy every call: the
        plant cache (_episode_plant) is what keeps one per hull draw."""
        if not draw or not rz_mod.hull_fields(draw):
            return self.hull, {}, []
        from sim import physics
        h, applied, unapplied = rz_mod.hull_for(
            self.base_hull(), physics.version(self.hull), draw)
        return h, applied, unapplied

    def _episode_plant(self, draw):
        """The episode's plant in self.sea; sets hull_applied,
        hull_unapplied and plant_source. The nominal hull's plant when the
        draw has no hull field ('nominal'); else the plant of the hull copy
        for the draw's hull fields, through the plant cache (app/randomize
        .PlantCache, keyed by the hull-draw tuple): a miss builds the copy's
        plant once and stores it as the pristine prototype ('built'), a hit
        reuses the stored one ('cache'); in both cases the episode gets a
        FRESH plant made from the prototype in the episode's sea
        (plant.with_sea: a new object built for the same hull object, so the
        physics' own steady-attitude and righting-arm caches hit), never the
        prototype itself. The hull in force under hull_every keeps its entry
        beside the cache, so it is reused even with the cache off."""
        from sim import physics
        self.hull_applied, self.hull_unapplied = {}, []
        if not draw or not rz_mod.hull_fields(draw):
            self.plant_source = "nominal"
            return self._build_plant(self.sea)
        key = rz_mod.hull_key(draw)
        f = self._hull_inforce
        e = self._plants.get(key)
        if e is None and f is not None and f.get("key") == key:
            e = f.get("entry")
        if e is None:
            hull, applied, unapplied = self._episode_hull(draw)
            proto = self._build_plant(self.sea, hull)
            e = dict(hull=hull, plant=proto, applied=applied,
                     unapplied=unapplied)
            self._plants.put(key, e)
            self.plant_source = "built"
        else:
            self.plant_source = "cache"
        if f is not None and f.get("key") == key:
            f["entry"] = e
        self.hull_applied = dict(e["applied"])
        self.hull_unapplied = list(e["unapplied"])
        with_sea = getattr(e["plant"], "with_sea", None)
        if with_sea is None:                 # a plant class without one
            return self._build_plant(self.sea, e["hull"])
        plant = with_sea(self.sea)
        physics.verify(plant, e["hull"],
                       allow_test_switches=self.test_switches is not None)
        self.plant = plant
        return plant

    def _draw_in_force(self, full):
        """(draw, hull_from) of a randomised episode under hull_every
        (app/randomize.py, HULL_EVERY): the i-th randomised episode of this
        env (i from 0) runs its own hull fields when i is a multiple of
        hull_every, and they stay in force; otherwise it runs the ones in
        force (its own hull numbers were drawn and are dropped: the other
        parameters' stream is the same whatever hull_every is). hull_from
        says whose hull draw the episode runs; None when no hull field is
        on."""
        i = self._n_rz
        self._n_rz += 1
        hull = rz_mod.hull_fields(full)
        if not hull:
            return full, None
        n = int(self.rz.hull_every)
        f = self._hull_inforce
        if f is None or n <= 1 or i % n == 0:
            f = self._hull_inforce = dict(seed=self.seed_, episode=i,
                                          hull=dict(hull),
                                          key=rz_mod.hull_key(hull),
                                          entry=None)
        draw = dict(full)
        draw.update(f["hull"])
        return draw, dict(seed=f["seed"], episode=f["episode"],
                          this_episode=i, hull_every=n)

    @property
    def reduced(self):
        """The identified reduced model of this hull (app/models.py)."""
        if self._reduced is None:
            from .models import reduced_model
            self._reduced = reduced_model(self.hull)
        return self._reduced

    def ensure_autopilot(self):
        if self.autopilot is None:
            from .models import Autopilot
            p = self.reduced.p
            # Episode's u_ref is the identified u_design (Mission passes it)
            self.autopilot = Autopilot(self.plant, self.reduced,
                                       float(p.get("u_design", self.u_ref)),
                                       self.dt_ctrl)
            self._apply_ap()
        return self.autopilot

    def _apply_ap(self):
        """The autopilot state a resume restored (integrator, reference)."""
        if getattr(self, "_ap_pending", None) is not None:
            self.autopilot._psi_i, self.autopilot.heading_ref = \
                self._ap_pending
            self._ap_pending = None

    # ------------------------------------------------------------ gym API
    def _seed_rng(self, seed):
        if gym is not None:
            super().reset(seed=seed)
        else:
            self.np_random = np.random.default_rng(seed)

    def episode_seed(self, seed=None):
        """The seed of the next episode (module docstring, SEEDS); seeds
        np_random when a seed is given or on the first reset."""
        if seed is not None:
            if isinstance(seed, bool) or int(seed) != seed or seed < 0:
                raise ValueError(f"seed: an integer >= 0, not {seed!r}")
            seed = int(seed)
            self._seed_rng(seed)
            return seed
        if self._n_resets == 0:
            self._seed_rng(self.scenario.seed)
            return self.scenario.seed
        if gym is not None:
            super().reset(seed=None)
        s = int(self.np_random.integers(2 ** 31))
        if self.rz.stream == "train":
            # the evaluation's seeds are never drawn for training
            from learn.repro.task import EVAL_SEEDS
            while s in EVAL_SEEDS:
                s = int(self.np_random.integers(2 ** 31))
        return s

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if "record" in options or "row" in options:
            bad = set(options) - {"record", "row"}
            if bad or "record" not in options:
                raise ValueError("reset options: {'record': Record or "
                                 "path, 'row': control step} to resume")
            if seed is not None:
                raise ValueError("resuming takes the record's seed; do not "
                                 "pass seed=")
            return self.resume(options["record"], options.get("row", 0))
        rz_opt = options.pop("randomize", None)
        rz_cfg = options.pop("randomization", None)
        if options:
            raise ValueError(f"reset options {sorted(options)}: 'randomize' "
                             f"(True / False / a draw dict), "
                             f"'randomization' (a config), or 'record' + "
                             f"'row' to resume")
        if rz_cfg is not None:
            self.rz = rz_mod.Randomization.of(rz_cfg)
            self._plants.resize(self.rz.plant_cache)
            self._hull_inforce, self._n_rz = None, 0
        sc = self.scenario
        self.resumed_from = None
        self.seed_ = self.episode_seed(seed)
        self._n_resets += 1
        u = sc.usv
        # the episode's randomisation draw (module docstring)
        self.hull_from = None
        if rz_opt is None or rz_opt is False:
            draw = None
        elif rz_opt is True:
            draw, self.hull_from = self._draw_in_force(
                self.rz.draw(self.seed_))
            if self.rz.stream == "train" and seed is not None:
                from learn.repro.task import EVAL_SEEDS
                if seed in EVAL_SEEDS:
                    warnings.warn(f"reset(seed={seed}) in the 'train' stream:"
                                  f" {seed} is an evaluation seed "
                                  f"(learn/repro/task.EVAL_SEEDS)",
                                  RuntimeWarning, stacklevel=2)
        else:
            draw = rz_mod.check_draw(rz_opt)
        self.draw = draw
        self.thrust_gain, self.acc_scale, self.motion_scale = \
            rz_mod.scales(draw)
        self.sea_seed = int(u["sea_seed"]) if u.get("sea_seed") is not None \
            else self.seed_
        self.sensor_seed = int(u["sensor_seed"]) if u.get(
            "sensor_seed") is not None else int(np.random.SeedSequence(
                [self.seed_, 0x5E45]).generate_state(1)[0])
        from sim.wavefield import SeaState
        hs, tp, kw = sc.sea_kw()
        if draw:
            hs, tp, kw = rz_mod.sea_params(draw, hs, tp, kw)
        self.sea = SeaState(hs, tp, seed=self.sea_seed, **kw)
        self.sea_params = dict(hs=hs, tp=tp, **kw)
        self.n_short_scaled = 0
        if draw and "short_wave" in draw:
            self.n_short_scaled = rz_mod.scale_short_waves(
                self.sea, self.L, draw["short_wave"], frozen.SHORT_WAVE_RATIO)
        t0b = time.perf_counter()
        plant = self._episode_plant(draw)
        self.t_build = time.perf_counter() - t0b
        self.x_gauge = accel_mod.gauge_x(plant.hull)
        self.x_station = float(plant.x[-1])
        self.filt = accel_mod.CausalLowpass(self.filter_hz, 1.0 / self.dt,
                                            n=2)
        self.acc_obs = np.zeros(2)
        u0 = sc.agent.get("speed")
        u0 = 0.8 * self.u_ref if u0 is None else float(u0)
        s = plant.initial_state(u0)
        x0, y0 = sc.start_xy()
        if x0 != 0.0 or y0 != 0.0:
            s[0], s[1] = x0, y0
        s[5] = sc.start_yaw()
        self.s = s
        self.s0 = s.copy()
        self.t = 0.0
        self.k = 0
        self.n_plant = 0
        self.track = Track(sc.track_spec(), (x0, y0))
        self.autopilot = None
        self._ap_pending = None
        if self.action_mode == "heading":
            self.ensure_autopilot()
        self.safe = SafeMode(self.dt_step) if u.get("safe_mode") else None
        self.episode_hooks = []
        so = self.obs_state(s)
        for sn in self.sensors.values():
            sn.reset(self.sensor_seed, self, 0.0, so)
            sn.deliver(0)
        self.finite, self.capsized, self.reason = True, False, None
        self.error = None
        self.last_reward, self.last_parts = 0.0, np.zeros(3)
        self.last_cmd = dict(thrust=float(s[12]), nozzle=float(s[13]),
                             heading=float("nan"), safe=False)
        self._rec_reset()
        self._t_wall0 = time.perf_counter()
        return self._obs(), self._info()

    def obs_state(self, s):
        """The state the observation side sees: heave, pitch and their
        rates times the motion_scale draw (the plant's own state when the
        scale is 1: the same object, no copy)."""
        m = self.motion_scale
        if m == 1.0:
            return s
        so = np.array(s, float)
        so[2] *= m
        so[8] *= m
        so[4] *= m
        so[10] *= m
        return so

    def _acc_update(self, a_cg, a_station):
        """The output spec's acceleration channel after a plant step: the
        CG and the bow gauge in g through the causal filter, times the
        acc_scale draw."""
        bow = accel_mod.bow_gauge(float(a_cg), float(a_station),
                                  self.x_gauge, self.x_station)
        f = self.filt(np.array([float(a_cg) / G, bow / G]))
        if self.acc_scale != 1.0:
            f = f * self.acc_scale
        self.acc_obs = f
        return f

    def step(self, action):
        if self.s is None:
            raise RuntimeError("call reset() first")
        if self.reason is not None:
            raise RuntimeError(f"the episode has ended ({self.reason}); "
                               f"call reset()")
        t0w = time.perf_counter()
        try:
            thrust, nozzle, heading, given = self._command(action)
        except ActionError:
            raise
        except (ValueError, TypeError) as ex:
            raise ActionError(str(ex)) from ex
        safe_on = False
        if self.safe is not None:
            thrust, nozzle, safe_on = self.safe(self.s[3], thrust, nozzle)
        self.last_cmd = dict(thrust=thrust, nozzle=nozzle, heading=heading,
                             safe=safe_on)
        from learn.repro.task import cost_parts
        plant, tr = self.plant, self.track
        parts = np.zeros(3)
        n = 0
        hooks = self.user_hooks + self.episode_hooks
        # the thrust the plant gets: the command times the thrust_gain draw
        # (1 when randomisation is off: the same float)
        thrust_p = thrust if self.thrust_gain == 1.0 else \
            thrust * self.thrust_gain
        for _ in range(self.sub):
            self.s = plant.step(self.s, self.t, thrust_p, nozzle, self.dt)
            for hk in hooks:
                out = hk(self.s, self.dt, plant)
                if out is not None:
                    self.s = out
            self.t += self.dt
            self.n_plant += 1
            s = self.s
            if not np.all(np.isfinite(s)):
                self.finite = False
                self._rec_plant(s, thrust_p, nozzle, np.full(2, np.nan))
                break
            a = plant.last_cg_acc / G
            ua, yt = tr.along_cross(s)
            parts += cost_parts(ua, a, yt, self.L, self.u_ref)
            n += 1
            acc = self._acc_update(plant.last_cg_acc,
                                   getattr(plant, "last_bow_acc", np.nan))
            self._rec_plant(s, thrust_p, nozzle, acc)
            so = self.obs_state(s)
            for sn in self.sensors.values():
                sn.sample(self, self.n_plant, self.t, so)
            if getattr(plant, "capsized", False):
                self.capsized = True
                break
        self.k += 1
        for sn in self.sensors.values():
            sn.deliver(self.n_plant)
        if self.finite:
            tr.update(self.s[0], self.s[1])
        reward = -float(parts.sum() / max(n, 1)) if n else -10.0
        self.last_reward = reward
        self.last_parts = parts / max(n, 1)
        terminated = (not self.finite) or self.capsized or tr.complete
        truncated = (not terminated) and self.k >= self.n_ctrl
        if terminated:
            self.reason = ("nonfinite" if not self.finite else
                           "capsized" if self.capsized else "track_complete")
        elif truncated:
            self.reason = "duration"
        self._rec_ctrl(given, thrust, nozzle, heading, reward, safe_on,
                       time.perf_counter() - t0w)
        return self._obs(), reward, bool(terminated), bool(truncated), \
            self._info()

    def close(self):
        pass

    # ------------------------------------------------------------ resume
    def resume(self, rec, k, strict=None):
        """Put the episode where `rec` was after its control step k
        (module docstring, RESUMING). Returns (obs, info) as reset().
        strict (default: k > 0) also refuses a record made by other physics
        code or with another sea; k = 0 without it is a plain reset with
        the record's seed (a replay from the start on changed code is
        allowed: it shows what changed)."""
        from . import plantstate
        from .record import Record, RecordMismatch, code_changes
        if isinstance(rec, (str, os.PathLike)):
            rec = Record.load(rec)
        h, P, C = rec.header, rec.plant, rec.ctrl
        m = len(C["t"])
        if isinstance(k, bool) or int(k) != k:
            raise ValueError(f"row: a control step 0..{m - 1}, not {k!r}")
        k = int(k)
        if not 0 <= k < m:
            raise ValueError(f"row {k}: the record has control steps "
                             f"0..{m - 1} to resume from")
        why = []
        if h.get("scenario_sha256") != self.scenario.sha256():
            why.append("the scenario differs from the record's")
        if tuple(h.get("test_switches") or ()) != \
                tuple(self.test_switches or ()):
            why.append(f"test switches {h.get('test_switches')} in the "
                       f"record, {self.test_switches} here")
        if (h.get("dt"), h.get("sub")) != (self.dt, self.sub):
            why.append(f"steps (dt, sub) {(h.get('dt'), h.get('sub'))} in "
                       f"the record, {(self.dt, self.sub)} here")
        if h.get("plant_kind") != self.plant_kind:
            why.append(f"plant {h.get('plant_kind')!r} in the record, "
                       f"{self.plant_kind!r} here")
        if why:
            raise RecordMismatch("cannot resume: " + "; ".join(why))
        # the record's randomisation draw is applied again (the same hull
        # copy, sea, gain and output scales), none when it had none
        opts = {"randomize": dict(h["randomization"])} \
            if h.get("randomization") else None
        obs, info = self.reset(seed=h["seed"], options=opts)
        if strict is None:
            strict = k > 0
        st = self.stamp()
        if strict and st.get("physics_sha1") != (h.get("physics") or {}).get(
                "physics_sha1"):
            raise RecordMismatch(
                "cannot resume: the physics that made the record is not "
                "the code here (" + "; ".join(code_changes(rec) or [
                    f"stamp {h.get('physics')} vs {st}"]) + ")")
        if strict and not (np.array_equal(self.s0, rec.s0) and all(
                np.array_equal(np.asarray(getattr(self.sea, c)),
                               np.asarray(h["sea"][c])) for c in
                ("a", "k", "th", "w", "phi"))):
            raise RecordMismatch("cannot resume: the reset state or the sea "
                                 "built from the record's seed differs")
        self.resumed_from = dict(fingerprint=rec.fingerprint(), row=k,
                                 source=h.get("created"))
        if k == 0:
            return obs, info
        if "aux" not in P:
            raise RecordMismatch(
                f"the record ({h.get('format')}) carries no plant state "
                f"rows (plant['aux']): it can be replayed from the start "
                f"only")
        n = k * self.sub
        if len(P["t"]) < n:
            raise RecordMismatch(f"the record has {len(P['t'])} plant rows,"
                                 f" control step {k} needs {n}")
        cols = plantstate.columns(self.plant)
        if list(h.get("state_cols") or []) != cols:
            raise RecordMismatch(f"plant state columns {h.get('state_cols')}"
                                 f" in the record, {cols} here")
        hid = plantstate.loads_rows(P["hidden"]) if "hidden" in P else None
        # the env side, replayed on the recorded rows (no plant step); the
        # output filter is fed the recorded raw accelerations, so its state
        # (and the acc_obs rows) are the live run's bit for bit
        tr, j = self.track, 0
        for c in range(k):
            for _ in range(self.sub):
                self.s = np.array(P["s"][j], float)
                self.t = float(P["t"][j])
                self.n_plant = j + 1
                acc = self._acc_update(P["a_cg"][j], P["a_bow"][j]) \
                    if np.all(np.isfinite(self.s)) else np.full(2, np.nan)
                if self.record_on:
                    self._P_row(P, hid, j, acc)
                so = self.obs_state(self.s)
                for sn in self.sensors.values():
                    sn.sample(self, self.n_plant, self.t, so)
                j += 1
            for sn in self.sensors.values():
                sn.deliver(self.n_plant)
            tr.update(self.s[0], self.s[1])
            if self.record_on:
                self._C_row(C, c)
        for name, sn in self.sensors.items():
            mine, theirs = sn.arrays(), rec.sensors.get(name)
            nn = len(mine["t_meas"])
            if theirs is None or any(
                    not np.array_equal(mine[q], np.asarray(theirs[q])[:nn])
                    for q in mine):
                raise RecordMismatch(f"cannot resume: sensor {name!r} "
                                     f"replayed on the recorded rows differs "
                                     f"from the record")
        if int(C["leg"][k - 1]) != tr.leg:
            raise RecordMismatch("cannot resume: the track replayed on the "
                                 "recorded rows is on another leg")
        self.k = k
        # the plant at row n - 1
        plantstate.restore(self.plant, self.s,
                           None if hid is None else hid[n - 1],
                           P["aux"][n - 1])
        # the env's controller state after control step k - 1
        ap = C["ap"][k - 1] if "ap" in C else None
        if ap is not None and np.all(np.isfinite(ap)):
            # applied now if the autopilot exists (heading mode), else when
            # a dict action with a heading first builds it
            self._ap_pending = (float(ap[0]), float(ap[1]))
            if self.autopilot is not None:
                self._apply_ap()
        ss = C["safe_state"][k - 1] if "safe_state" in C else None
        if self.safe is not None and ss is not None and np.all(
                np.isfinite(ss)):
            self.safe.on, self.safe.calm = bool(ss[0]), float(ss[1])
            self.safe.n_on, self.safe.n_entries = int(ss[2]), int(ss[3])
        hd = float(C["heading"][k - 1])
        self.last_cmd = dict(thrust=float(C["thrust"][k - 1]),
                             nozzle=float(C["nozzle"][k - 1]), heading=hd,
                             safe=bool(C["safe"][k - 1]))
        self.last_reward = float(C["reward"][k - 1])
        self.last_parts = np.array(C["parts"][k - 1], float)
        return self._obs(), self._info()

    # ------------------------------------------------------------ actions
    def _command(self, action):
        """(thrust N, nozzle rad, heading ref or nan, normalised record of
        what was given). Raises ValueError on a malformed or non-finite
        action BEFORE anything changes (the autopilot's integrator
        included)."""
        from .lotus import finite_number
        heading = float("nan")
        if isinstance(action, dict):
            bad = set(action) - {"thrust", "throttle", "nozzle_rad",
                                 "nozzle", "heading"}
            if bad:
                raise ValueError(f"action keys {sorted(bad)}: thrust | "
                                 f"throttle, nozzle_rad | nozzle | heading")
            action = {k: finite_number(v, k) for k, v in action.items()}
            if "thrust" in action and "throttle" in action:
                raise ValueError("action: thrust or throttle, not both")
            if "thrust" in action:
                thrust = float(action["thrust"])
            elif "throttle" in action:
                thrust = float(action["throttle"]) * self.t_max
            else:
                raise ValueError("action needs thrust (N) or throttle")
            thrust = float(np.clip(thrust, self.t_min, self.t_max))
            nk = [k for k in ("nozzle_rad", "nozzle", "heading")
                  if k in action]
            if len(nk) != 1:
                raise ValueError("exactly one of nozzle_rad, nozzle, heading")
            if nk[0] == "heading":
                heading = float(action["heading"])
                nozzle = self.ensure_autopilot()(self.s, thrust, heading)
            elif nk[0] == "nozzle":
                nozzle = float(np.clip(action["nozzle"], -1, 1)) * \
                    self.rud_max
            else:
                nozzle = float(np.clip(action["nozzle_rad"], -self.rud_max,
                                       self.rud_max))
            return thrust, nozzle, heading, (thrust / self.t_max,
                                             nozzle / self.rud_max)
        try:
            a = np.asarray(action, float).reshape(-1)
        except (TypeError, ValueError):
            raise ValueError(f"action: [throttle, nozzle | heading] numbers, "
                             f"not {action!r}") from None
        if a.size != 2:
            raise ValueError("action: [throttle, nozzle | heading]")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"non-finite action {a.tolist()}")
        lo = self.t_min / self.t_max
        thr = float(np.clip(a[0], lo, 1.0))
        thrust = thr * self.t_max
        if self.action_mode == "direct":
            nozzle = float(np.clip(a[1], -1.0, 1.0)) * self.rud_max
        else:
            heading = float(a[1])
            nozzle = self.ensure_autopilot()(self.s, thrust, heading)
        return thrust, nozzle, heading, (float(a[0]), float(a[1]))

    # ------------------------------------------------------------ views
    def _obs(self):
        o = {n: sn.obs(self.t) for n, sn in self.sensors.items()}
        o["accel"] = np.array(self.acc_obs, float)
        if self.scenario.usv.get("obs_truth"):
            o["truth"] = np.concatenate([[self.t], self.obs_state(self.s)])
        return o

    def pose(self):
        """ENU pose of the CG (z = height above calm water), FLU
        orientation with the total pitch, geodetic point, body velocity."""
        s, p = self.s, self.plant
        pitch = p.Th_rest + s[4]
        e, nn, uu = float(s[0]), float(s[1]), float(p.Z_rest + s[2])
        lat, lon, alt = self.frame.to_geodetic(e, nn, uu)
        w = geo.euler_rates_to_body(s[3], pitch, s[9], s[10], s[11])
        return dict(t=self.t, position=(e, nn, uu),
                    rpy=(float(s[3]), float(pitch), float(s[5])),
                    quaternion=geo.quat_zyx(s[3], pitch, s[5]),
                    geo=(lat, lon, alt),
                    lin_vel_body=(float(s[6]), float(s[7]), float(s[8])),
                    ang_vel_body=tuple(float(v) for v in w))

    def track_state(self, red, s=None):
        """Mission.track_state for the current leg: the reduced state with
        x along the leg, y its cross-track, heading minus the leg angle (of
        the plant state, or of `s`, e.g. a sensor estimate)."""
        sr = red.from_plant_state(self.s if s is None else s)
        tr = self.track
        if tr.phi != 0.0 or tr.ox != 0.0 or tr.oy != 0.0:
            x, y = sr[0], sr[1]
            if tr.ox != 0.0 or tr.oy != 0.0:
                x, y = x - tr.ox, y - tr.oy
            sr[0] = tr.cp * x + tr.sp * y
            sr[1] = -tr.sp * x + tr.cp * y
            sr[7] = sr[7] - tr.phi
        return sr

    def _info(self):
        tr = self.track
        ua, yt = tr.along_cross(self.s) if self.finite else (None, None)
        return dict(t=self.t, k=self.k, seed=self.seed_,
                    state=self.s.copy(), leg=tr.leg,
                    n_legs=tr.n_legs, track_complete=tr.complete,
                    phi=float(tr.phi),
                    u_along=None if ua is None else float(ua),
                    cross=None if yt is None else float(yt),
                    cost_parts=self.last_parts.copy(),
                    reward=self.last_reward, capsized=self.capsized,
                    finite=self.finite, reason=self.reason,
                    cmd=self._cmd_info(), hull=self.hull_key,
                    physics=self.stamp(),
                    accel=np.array(self.acc_obs, float),
                    randomization=(None if self.draw is None
                                   else dict(self.draw)),
                    randomization_hull_from=(None if self.hull_from is None
                                             else dict(self.hull_from)))

    def _cmd_info(self):
        """last_cmd for info: no NaN (a NaN never equals itself, and a
        Gym info must compare equal across equal runs)."""
        c = dict(self.last_cmd)
        if c.get("heading") is not None and math.isnan(c["heading"]):
            c["heading"] = None
        return c

    def stamp(self):
        from sim import physics
        return physics.stamp(self.plant)

    # ------------------------------------------------------------ record
    # plant rows: thrust is what the PLANT got (the command times the
    # thrust_gain draw); acc_obs the output spec's filtered CG and bow
    # accelerations in g as the observation saw them (times acc_scale);
    # a_cg, a_bow the raw m/s^2 of the CG and the bow station
    P_KEYS = ("t", "s", "a_cg", "a_bow", "n_sub", "thrust", "nozzle", "aux")
    C_KEYS = ("t", "given", "thrust", "nozzle", "heading", "reward", "parts",
              "leg", "safe", "wall", "ap", "safe_state")

    def _rec_reset(self):
        self._P = {k: [] for k in self.P_KEYS}
        self._P["hidden"] = []
        self._P["acc_obs"] = []
        self._C = {k: [] for k in self.C_KEYS}

    def _rec_plant(self, s, thrust, nozzle, acc):
        if not self.record_on:
            return
        from . import plantstate
        P, p = self._P, self.plant
        P["t"].append(self.t)
        P["s"].append(np.array(s, float))
        P["a_cg"].append(float(p.last_cg_acc))
        P["a_bow"].append(float(getattr(p, "last_bow_acc", np.nan)))
        P["n_sub"].append(int(getattr(p, "last_n_sub", 1)))
        P["thrust"].append(thrust)
        P["nozzle"].append(nozzle)
        P["acc_obs"].append(np.array(acc, float))
        # the plant's state outside s after this step (app/plantstate.py)
        P["aux"].append(plantstate.aux_row(p))
        P["hidden"].append(plantstate.hidden(p))

    def _P_row(self, P, hid, j, acc):
        """Row j of a record's plant arrays into this episode's lists (a
        resume); acc the filter's output recomputed on the row."""
        for k in self.P_KEYS:
            v = P[k][j]
            self._P[k].append(np.array(v, float) if k in ("s", "aux") else
                              (int(v) if k == "n_sub" else float(v)))
        self._P["acc_obs"].append(np.array(acc, float))
        self._P["hidden"].append(None if hid is None else hid[j])

    def _C_row(self, C, c):
        for k in self.C_KEYS:
            v = C[k][c] if k in C else np.full(
                {"ap": 2, "safe_state": 4}.get(k, 1), np.nan)
            if k in ("given", "parts", "ap", "safe_state"):
                v = np.array(v, float)
            elif k == "leg":
                v = int(v)
            elif k == "safe":
                v = bool(v)
            else:
                v = float(v)
            self._C[k].append(v)

    def _rec_ctrl(self, given, thrust, nozzle, heading, reward, safe, wall):
        if not self.record_on:
            return
        C = self._C
        C["t"].append(self.t)
        C["given"].append(given)
        C["thrust"].append(thrust)
        C["nozzle"].append(nozzle)
        C["heading"].append(heading)
        C["reward"].append(reward)
        C["parts"].append(self.last_parts.copy())
        C["leg"].append(self.track.leg)
        C["safe"].append(bool(safe))
        C["wall"].append(wall)
        # the env's controller state after this step (resume)
        ap = self.autopilot
        C["ap"].append([np.nan, np.nan] if ap is None else
                       [float(ap._psi_i), float(ap.heading_ref)])
        sm = self.safe
        C["safe_state"].append([np.nan] * 4 if sm is None else
                               [float(sm.on), float(sm.calm),
                                float(sm.n_on), float(sm.n_entries)])

    def header(self, policy=None):
        sc, sea, p = self.scenario, self.sea, self.plant
        hs, tp, kw = sc.sea_kw()
        if not _GIT:
            _GIT.update(git_version())
        from sim import physics
        from . import plantstate
        st = self.stamp()
        nominal = physics.stamp_for(self.hull, st["physics"],
                                    st.get("switches", ()))
        return dict(
            format="usv-app-record/2",
            scenario=sc.to_dict(), scenario_sha256=sc.sha256(),
            scenario_source=sc.source, hull_key=self.hull_key,
            hull_sha1=physics.hull_sha1(self.hull),
            cache_tag=physics.cache_tag(self.hull),
            # the stamp of the plant that ran (a randomised episode's copy
            # of the hull has its own manifest), the nominal key's beside it
            physics=st, physics_nominal=nominal,
            episode_hull=p.hull.name,
            episode_hull_sha1=physics.hull_sha1(p.hull),
            frozen=bool(self.frozen),
            randomization=(None if self.draw is None else dict(self.draw)),
            randomization_config=self.rz.to_dict(),
            randomization_applied=dict(hull=dict(self.hull_applied),
                                       unapplied=list(self.hull_unapplied),
                                       sea=self.sea_params,
                                       n_short_wave_scaled=self.n_short_scaled,
                                       thrust_gain=self.thrust_gain,
                                       acc_scale=self.acc_scale,
                                       motion_scale=self.motion_scale),
            plant_build_s=float(getattr(self, "t_build", 0.0)),
            # whose hull draw the episode runs (hull_every) and where its
            # plant came from (the plant cache)
            randomization_hull_from=(None if self.hull_from is None
                                     else dict(self.hull_from)),
            plant_cache=dict(source=self.plant_source, n=len(self._plants),
                             max=int(self._plants.maxsize)),
            filter_hz=self.filter_hz, filter_order=accel_mod.ORDER,
            x_bow_gauge=self.x_gauge, x_bow_station=self.x_station,
            plant_class=type(p).__name__,
            test_switches=(None if self.test_switches is None
                           else list(self.test_switches)),
            state_format=plantstate.FORMAT,
            state_cols=plantstate.columns(p),
            hidden_state=bool(any(r is not None
                                  for r in self._P.get("hidden", []))),
            resumed_from=self.resumed_from,
            plant_kind=self.plant_kind, action_mode=self.action_mode,
            seed=self.seed_, sea_seed=self.sea_seed,
            sensor_seed=self.sensor_seed,
            sea=dict(hs=hs, tp=tp, **{k: (list(v) if isinstance(v, tuple)
                                          else v) for k, v in kw.items()},
                     a=np.asarray(sea.a).tolist(),
                     k=np.asarray(sea.k).tolist(),
                     th=np.asarray(sea.th).tolist(),
                     w=np.asarray(sea.w).tolist(),
                     phi=np.asarray(sea.phi).tolist()),
            dt=self.dt, sub=self.sub, dt_ctrl=self.dt_ctrl,
            dt_step=self.dt_step, n_ctrl=self.n_ctrl, u_ref=self.u_ref,
            L=self.L, t_max=self.t_max, t_min=self.t_min,
            rud_max=self.rud_max, Z_rest=float(p.Z_rest),
            Th_rest=float(p.Th_rest), x_stem=self.x_stem,
            reference_position=self.frame.as_dict(),
            track=self.track.describe(),
            sensors={n: s.describe() for n, s in self.sensors.items()},
            safe_mode=self.safe is not None,
            policy=policy, reason=self.reason, error=self.error,
            scenario_warnings=list(sc.warnings),
            t_end=self.t, k=self.k,
            capsized=self.capsized, finite=self.finite,
            git=dict(_GIT), python=platform.python_version(),
            numpy=np.__version__,
            created=time.strftime("%Y-%m-%dT%H:%M:%S"),
            wall_s=time.perf_counter() - self._t_wall0)

    def record(self, policy=None):
        """The app.record.Record of the episode so far."""
        from .record import Record
        if not self.record_on:
            raise RuntimeError("built with record=False")
        from . import plantstate
        P, C = self._P, self._C
        n = len(P["t"])
        ncol = len(plantstate.columns(self.plant))
        plant = dict(
            t=np.asarray(P["t"], float),
            s=np.asarray(P["s"], float).reshape(n, 14),
            a_cg=np.asarray(P["a_cg"], float),
            a_bow=np.asarray(P["a_bow"], float),
            n_sub=np.asarray(P["n_sub"], np.int64),
            thrust=np.asarray(P["thrust"], float),
            nozzle=np.asarray(P["nozzle"], float),
            acc_obs=np.asarray(P["acc_obs"], float).reshape(n, 2),
            aux=np.asarray(P["aux"], float).reshape(n, ncol))
        if any(r is not None for r in P["hidden"]):
            # the physics' hidden state after every step, exact (JSON
            # bytes; app/plantstate.loads_rows); absent when the plant has
            # none (v0, the as1 preset), so those records hash as before
            plant["hidden"] = plantstate.dumps_rows(P["hidden"])
        m = len(C["t"])
        ctrl = dict(
            t=np.asarray(C["t"], float),
            given=np.asarray(C["given"], float).reshape(m, 2),
            thrust=np.asarray(C["thrust"], float),
            nozzle=np.asarray(C["nozzle"], float),
            heading=np.asarray(C["heading"], float),
            reward=np.asarray(C["reward"], float),
            parts=np.asarray(C["parts"], float).reshape(m, 3),
            leg=np.asarray(C["leg"], np.int64),
            safe=np.asarray(C["safe"], bool),
            wall=np.asarray(C["wall"], float),
            ap=np.asarray(C["ap"], float).reshape(m, 2),
            safe_state=np.asarray(C["safe_state"], float).reshape(m, 4))
        sens = {n_: s.arrays() for n_, s in self.sensors.items()}
        return Record(self.header(policy), np.asarray(self.s0, float),
                      plant, ctrl, sens)
