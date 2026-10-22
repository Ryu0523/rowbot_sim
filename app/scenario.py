"""
Scenario files: what a run is, in one JSON (or YAML, when PyYAML is
installed) document -- hull key and physics version, sea, start, track or
waypoints, duration, seeds, sensors, policy.

KEY NAMES. Where LOTUSim has a key (LOTUSim/assets/scenarios/demo.yaml,
systems/lotusim_mas/src/scenario_parser.cpp) the same key means the same
thing here: name, description, version, reference_position {latitude,
longitude, altitude}, environment.ocean {wave_height, wave_period,
wave_direction, wave_spectrum, current_speed}, environment.atmosphere
{wind_speed}, agents.<name> {model, position, heading, waypoint_interface}.
LOTUSim's parser reads none of the environment (LOTUSIM_GAP_2026-10-05.md);
here every ocean key is used and the ones the planing plant cannot honour
(current, wind) are refused unless zero.

WAYPOINT_INTERFACE (LOTUSim's scenario_parser.cpp appendWaypointFollower
and systems/waypoint_follower read the same keys the same way):
  enabled   default TRUE when the block is present (LOTUSim's default)
  loop      default false
  mode      "waypoints": waypoints [{lat, lng}, ...] (WGS84 degrees,
                         converted on the tangent plane at
                         reference_position); also accepted per point:
                         {latitude, longitude} or {enu: [x, y]}
            "line":      line {direction (rad, ENU, CCW from east -- an
                         SDF angle in LOTUSim), length (m)}: one leg from
                         the start
            "circle":    circle {radius (m)}: LOTUSim's eight points
                         start + radius (cos k pi/4, sin k pi/4), k = 0..7
  ours      path_enu [[x, y], ...] (metres, ENU), or path [{enu: [x, y]}
            | {latitude, longitude}, ...] -- bare lists under "path" are
            refused (a [lat, lon] pair must not be read as metres);
            radius (m, > 0, default 20): a leg ends within it of its end
            point (LOTUSim's range_tolerance is accepted as the same thing)
  One of mode / waypoints / path / path_enu must say where to go.

VALIDATION (at load; ScenarioError names the key): numbers are finite and
in range (wave_height >= 0, wave_period > 0, n_freq and n_dir integers >= 1,
band 0 < lo < hi, seeds integers >= 0, duration > 0, radius > 0, ...);
sensors (app/sensors.validate_config: 0 <= dropout <= 1, rate_hz > 0 or
null = every plant step, latency_s >= 0, distinct preview distances); the
policy spec (app/policies.make_policy, schedules dry-run; a callable's
target is checked for its form only, never imported here). Outside the
validated range of ACCEPTANCE R1 (Hs <= 1.5 m, Tp 3-10 s, speed -3..40 kn)
the scenario loads with a warning (ScenarioWarning), listed in
Scenario.warnings and in every record's header.

CONVENTIONS (app/geo.py). ENU world, FLU body. agent heading: degrees
counter-clockwise from east (LOTUSim: "relative to East"), or heading_rad.
wave_direction: degrees clockwise from true north the waves come FROM; or
the plant's own theta0 (radians, ENU direction of travel) -- exactly one of
the two. Positions: {"latitude", "longitude"} (converted on the WGS84
tangent plane at reference_position) or {"enu": [x, y]} metres.

Exactly one agent: this layer drives one boat (multi-vessel is out of scope,
ACCEPTANCE P).

    sc = load("app/scenarios/demo_scarab195.json")
    sc.hull_key, sc.sea_kw(), sc.duration, sc.seed ...
    save(sc, "copy.json")           # canonical JSON; load(save(x)) == x
"""
import copy
import hashlib
import json
import math
import os
import warnings

from . import geo

KN = 0.514444
R1 = dict(hs_max=1.5, tp=(3.0, 10.0), speed_kn=(-3.0, 40.0))

DEFAULTS = {
    "name": "unnamed",
    "description": "",
    "version": "0.1",
    "reference_position": {"latitude": 0.0, "longitude": 0.0,
                           "altitude": 0.0},
    "environment": {
        "ocean": {"wave_height": 0.0, "wave_period": 5.0,
                  "wave_spectrum": "JONSWAP", "gamma": 3.3,
                  "spreading_s": 8, "n_freq": 24, "n_dir": 5,
                  "band": [0.5, 3.0], "jitter": False,
                  "current_speed": 0.0},
        "atmosphere": {"wind_speed": 0.0},
    },
    # dt / dt_ctrl null: the frozen configuration's steps for an as version
    # (app/frozen.py: 0.01 s, 0.25 s), the hull's own for v0; filter_hz
    # null: the frozen 20 Hz; randomization: null or a dict for
    # app/randomize.Randomization.of (applied when reset(options=
    # {"randomize": True}) asks)
    "usv": {"duration": 60.0, "seed": 0, "sea_seed": None,
            "sensor_seed": None, "safe_mode": False, "obs_truth": False,
            "record_plant": False, "dt": None, "dt_ctrl": None,
            "filter_hz": None, "randomization": None},
}
AGENT_DEFAULTS = {
    "model": "scarab195@as1",           # the frozen configuration's (app/frozen)
    "plant": "gz",
    "position": {"enu": [0.0, 0.0]},
    "heading": 0.0,
    "speed": None,
    "action_mode": "direct",
    "waypoint_interface": None,
    "track": None,
    "sensors": {},
    "policy": None,
}
OCEAN_KEYS = {"wave_height", "wave_period", "wave_direction", "theta0",
              "wave_spectrum", "gamma", "spreading_s", "n_freq", "n_dir",
              "band", "jitter", "current_speed", "current_direction",
              "water_temperature", "salinity", "sea_state"}
AGENT_KEYS = set(AGENT_DEFAULTS) | {"heading_rad", "guidance"}
USV_KEYS = set(DEFAULTS["usv"])
TOP_KEYS = {"name", "description", "version", "author", "tags",
            "time_of_day", "date", "reference_position", "environment",
            "agents", "usv"}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class ScenarioError(ValueError):
    pass


class ScenarioWarning(UserWarning):
    """A scenario outside the validated range (ACCEPTANCE R1)."""


WP_KEYS = {"enabled", "loop", "mode", "waypoints", "line", "circle", "path",
           "path_enu", "radius", "range_tolerance"}


def _number(v, key, lo=None, hi=None, lo_open=False, integer=False):
    """float(v) (int when integer) for scenario key `key`, checked."""
    if isinstance(v, bool) or v is None:
        raise ScenarioError(f"{key}: a number, not {v!r}")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ScenarioError(f"{key}: a number, not {v!r}") from None
    if not math.isfinite(x):
        raise ScenarioError(f"{key}: a finite number, not {v!r}")
    if integer:
        if x != int(x):
            raise ScenarioError(f"{key}: an integer, not {v!r}")
        x = int(x)
    if lo is not None and (x <= lo if lo_open else x < lo):
        raise ScenarioError(f"{key}: {'>' if lo_open else '>='} {lo:g}, "
                            f"not {v!r}")
    if hi is not None and x > hi:
        raise ScenarioError(f"{key}: <= {hi:g}, not {v!r}")
    return x


def as_seed(v, key="usv.seed"):
    return _number(v, key, lo=0, integer=True)


class Scenario:
    """A validated scenario. `raw` is the document as given (what save()
    writes back); everything else is derived from it."""

    def __init__(self, doc, source=None):
        if not isinstance(doc, dict):
            raise ScenarioError("a scenario is a JSON object")
        bad = set(doc) - TOP_KEYS
        if bad:
            raise ScenarioError(f"unknown top-level keys {sorted(bad)}")
        self.raw = copy.deepcopy(doc)
        self.source = source
        d = _merge(DEFAULTS, doc)
        agents = doc.get("agents") or {}
        if len(agents) != 1:
            raise ScenarioError(f"exactly one agent expected, got "
                                f"{len(agents)} (one boat per run)")
        (self.agent_name, a0), = agents.items()
        bad = set(a0) - AGENT_KEYS
        if bad:
            raise ScenarioError(f"agent {self.agent_name!r}: unknown keys "
                                f"{sorted(bad)}")
        self.agent = _merge(AGENT_DEFAULTS, a0)
        oc = d["environment"].get("ocean", {})
        bad = set(oc) - OCEAN_KEYS
        if bad:
            raise ScenarioError(f"environment.ocean: unknown keys "
                                f"{sorted(bad)}")
        bad = set(d["usv"]) - USV_KEYS
        if bad:
            raise ScenarioError(f"usv: unknown keys {sorted(bad)}")
        self.doc = d
        self.warnings = []
        self._validate()
        for w in self.warnings:
            warnings.warn(f"scenario {self.name!r}: {w}", ScenarioWarning,
                          stacklevel=3)

    # ------------------------------------------------------------ fields
    def _validate(self):
        oc = self.ocean
        if "wave_direction" in self.raw_ocean and "theta0" in self.raw_ocean:
            raise ScenarioError("give wave_direction (deg, from, clockwise "
                                "from north) or theta0 (rad, ENU travel "
                                "direction), not both")
        cs = _number(oc.get("current_speed", 0.0) or 0.0,
                     "environment.ocean.current_speed")
        if cs != 0.0:
            raise ScenarioError("current is not modelled by the planing "
                                "plant: current_speed must be 0")
        wind = self.doc["environment"].get("atmosphere", {}) or {}
        ws = _number(wind.get("wind_speed", 0.0) or 0.0,
                     "environment.atmosphere.wind_speed")
        if ws != 0.0:
            raise ScenarioError("wind is not modelled by the planing plant "
                                "(only still-air drag): wind_speed must be 0")
        if str(oc.get("wave_spectrum", "JONSWAP")).upper() not in (
                "JONSWAP",):
            raise ScenarioError("wave_spectrum: only JONSWAP (with gamma; "
                                "gamma 1 is Pierson-Moskowitz)")
        k = "environment.ocean."
        hs = _number(oc["wave_height"], k + "wave_height", lo=0.0)
        tp = _number(oc["wave_period"], k + "wave_period", lo=0.0,
                     lo_open=True)
        for key in ("wave_direction", "theta0"):
            if key in self.raw_ocean:
                _number(self.raw_ocean[key], k + key)
        _number(oc.get("gamma", 3.3), k + "gamma", lo=0.0, lo_open=True)
        _number(oc.get("spreading_s", 8), k + "spreading_s", lo=0.0,
                lo_open=True)
        _number(oc["n_freq"], k + "n_freq", lo=1, integer=True)
        _number(oc["n_dir"], k + "n_dir", lo=1, integer=True)
        band = oc.get("band", (0.5, 3.0))
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise ScenarioError(f"{k}band: [lo, hi] (multiples of the peak "
                                f"frequency), not {band!r}")
        b0 = _number(band[0], k + "band[0]", lo=0.0, lo_open=True)
        b1 = _number(band[1], k + "band[1]")
        if b1 <= b0:
            raise ScenarioError(f"{k}band: lo < hi, not {band!r}")
        r = self.doc["reference_position"]
        if not isinstance(r, dict):
            raise ScenarioError("reference_position: {latitude, longitude, "
                                "altitude}")
        _number(r.get("latitude", 0.0), "reference_position.latitude",
                lo=-90.0, hi=90.0)
        _number(r.get("longitude", 0.0), "reference_position.longitude",
                lo=-180.0, hi=360.0)
        _number(r.get("altitude", 0.0), "reference_position.altitude")
        if self.agent["plant"] not in ("gz", "lin"):
            raise ScenarioError("agent plant: 'gz' (large-angle roll, can "
                                "capsize; the evaluation's) or 'lin'")
        if self.agent["action_mode"] not in ("direct", "heading"):
            raise ScenarioError("action_mode: 'direct' or 'heading'")
        u = self.usv
        _number(u["duration"], "usv.duration", lo=0.0, lo_open=True)
        as_seed(u["seed"], "usv.seed")
        for key in ("sea_seed", "sensor_seed"):
            if u.get(key) is not None:
                as_seed(u[key], f"usv.{key}")
        for key in ("safe_mode", "obs_truth", "record_plant"):
            if not isinstance(u.get(key, False), bool):
                raise ScenarioError(f"usv.{key}: true or false")
        for key in ("dt", "dt_ctrl", "filter_hz"):
            if u.get(key) is not None:
                _number(u[key], f"usv.{key}", lo=0.0, lo_open=True)
        if u.get("dt") is not None and u.get("dt_ctrl") is not None \
                and u["dt_ctrl"] < u["dt"]:
            raise ScenarioError("usv.dt_ctrl must be >= usv.dt")
        if u.get("randomization") is not None:
            from .randomize import Randomization
            try:
                Randomization.of(u["randomization"])
            except (TypeError, ValueError, KeyError) as ex:
                raise ScenarioError(f"usv.randomization: {ex}") from None
        if "heading_rad" in self.agent and "heading" in self.raw_agent:
            raise ScenarioError("heading (deg) or heading_rad, not both")
        ag = f"agents.{self.agent_name}."
        if "heading_rad" in self.agent:
            _number(self.agent["heading_rad"], ag + "heading_rad")
        else:
            _number(self.agent.get("heading", 0.0), ag + "heading")
        speed = self.agent.get("speed")
        if speed is not None:
            speed = _number(speed, ag + "speed")
        self.start_xy()
        wp = self.agent.get("waypoint_interface")
        if wp is not None and not isinstance(wp, dict):
            raise ScenarioError(f"{ag}waypoint_interface: an object")
        if self._wp_enabled() and self.agent.get("track"):
            raise ScenarioError("give waypoints or a straight track, not "
                                "both")
        self.track_spec()                       # checks the waypoints
        try:
            from .sensors import validate_config
            validate_config(self.sensor_config())
        except ValueError as ex:
            raise ScenarioError(f"{ag}sensors: {ex}") from None
        self._check_policy(self.agent.get("policy"))
        # outside the validated range (ACCEPTANCE R1): load, but say so
        if hs > R1["hs_max"]:
            self.warnings.append(
                f"wave_height {hs:g} m is above the validated {R1['hs_max']:g}"
                f" m (ACCEPTANCE R1)")
        if not R1["tp"][0] <= tp <= R1["tp"][1]:
            self.warnings.append(
                f"wave_period {tp:g} s is outside the validated "
                f"{R1['tp'][0]:g}-{R1['tp'][1]:g} s (ACCEPTANCE R1)")
        if speed is not None and not (R1["speed_kn"][0] * KN <= speed <=
                                      R1["speed_kn"][1] * KN):
            self.warnings.append(
                f"initial speed {speed:g} m/s ({speed / KN:.1f} kn) is "
                f"outside the validated {R1['speed_kn'][0]:g} to "
                f"{R1['speed_kn'][1]:g} kn (ACCEPTANCE R1)")

    def _check_policy(self, spec):
        """The agent's policy spec builds (make_policy; schedules are
        dry-run in the constructors). A callable's target is checked for
        its form only: loading a scenario never imports code."""
        if spec is None:
            return
        where = f"agents.{self.agent_name}.policy"
        if isinstance(spec, str):
            if ":" in spec and not spec.startswith("learned:"):
                mod, _, attr = spec.partition(":")
                if not mod or not attr:
                    raise ScenarioError(f"{where}: 'pkg.mod:function', not "
                                        f"{spec!r}")
                return
        elif isinstance(spec, dict):
            if spec.get("type") == "callable":
                t = spec.get("target")
                if not isinstance(t, str) or ":" not in t:
                    raise ScenarioError(f"{where}: a callable needs target "
                                        f"'pkg.mod:function'")
                return
        else:
            raise ScenarioError(f"{where}: a name or an object "
                                f"{{'type': ...}}, not {spec!r}")
        from .policies import make_policy
        try:
            make_policy(spec)
        except (TypeError, ValueError, KeyError) as ex:
            raise ScenarioError(f"{where}: {type(ex).__name__}: {ex}") \
                from None

    @property
    def raw_ocean(self):
        return (self.raw.get("environment") or {}).get("ocean") or {}

    @property
    def raw_agent(self):
        return (self.raw.get("agents") or {})[self.agent_name]

    @property
    def name(self):
        return str(self.doc["name"])

    @property
    def ocean(self):
        return self.doc["environment"]["ocean"]

    @property
    def usv(self):
        return self.doc["usv"]

    @property
    def hull_spec(self):
        return self.agent["model"]

    @property
    def hull_key(self):
        from .hulls import key_of
        return key_of(self.agent["model"])

    @property
    def duration(self):
        return float(self.usv["duration"])

    @property
    def seed(self):
        return as_seed(self.usv["seed"])

    @property
    def frame(self):
        r = self.doc["reference_position"]
        return geo.LocalFrame(r.get("latitude", 0.0), r.get("longitude", 0.0),
                              r.get("altitude", 0.0))

    def theta0(self):
        oc = self.raw_ocean
        if "theta0" in oc:
            return float(oc["theta0"])
        return geo.wave_from_deg_to_theta0(oc.get("wave_direction", 90.0))

    def sea_kw(self):
        """(hs, tp, SeaState keywords) of the scenario's sea (seed apart)."""
        oc = self.ocean
        kw = dict(theta0=self.theta0(), n_freq=int(oc["n_freq"]),
                  n_dir=int(oc["n_dir"]))
        # only pass what differs from the SeaState defaults, so a scenario
        # that does not mention them builds the sea the way Mission does
        if float(oc.get("gamma", 3.3)) != 3.3:
            kw["gamma"] = float(oc["gamma"])
        if float(oc.get("spreading_s", 8)) != 8:
            kw["spread_s"] = float(oc["spreading_s"])
        if tuple(oc.get("band", (0.5, 3.0))) != (0.5, 3.0):
            kw["band"] = tuple(float(b) for b in oc["band"])
        if oc.get("jitter"):
            kw["jitter"] = True
        return float(oc["wave_height"]), float(oc["wave_period"]), kw

    def _xy(self, p, key="position", metres_list=False):
        """(east, north) metres of a point: {"enu": [x, y]}, {"latitude",
        "longitude"} or LOTUSim's {"lat", "lng"}; a bare [x, y] only where
        the key says metres (path_enu)."""
        if metres_list and isinstance(p, (list, tuple)) and len(p) == 2:
            return (_number(p[0], f"{key}[0]"), _number(p[1], f"{key}[1]"))
        if isinstance(p, dict) and "enu" in p:
            e = p["enu"]
            if not isinstance(e, (list, tuple)) or len(e) not in (2, 3):
                raise ScenarioError(f"{key}.enu: [x, y] metres, not {e!r}")
            return (_number(e[0], f"{key}.enu[0]"),
                    _number(e[1], f"{key}.enu[1]"))
        for la, lo in (("latitude", "longitude"), ("lat", "lng")):
            if isinstance(p, dict) and la in p:
                lat = _number(p[la], f"{key}.{la}", lo=-90.0, hi=90.0)
                lon = _number(p.get(lo), f"{key}.{lo}", lo=-180.0, hi=360.0)
                alt = _number(p.get("altitude", 0.0), f"{key}.altitude")
                e, n, _ = self.frame.to_enu(lat, lon, alt)
                return e, n
        raise ScenarioError(f"{key} {p!r}: {{'enu': [x, y]}} (metres) or "
                            f"{{'latitude', 'longitude'}} / {{'lat', 'lng'}}"
                            f" (degrees)")

    def start_xy(self):
        return self._xy(self.agent["position"],
                        f"agents.{self.agent_name}.position")

    def start_yaw(self):
        if "heading_rad" in self.agent:
            return float(self.agent["heading_rad"])
        return geo.deg_to_rad_wrapped(self.agent.get("heading", 0.0))

    def _wp_enabled(self):
        wp = self.agent.get("waypoint_interface")
        if not isinstance(wp, dict):
            return False
        en = wp.get("enabled", True)          # LOTUSim's default
        if not isinstance(en, bool):
            raise ScenarioError("waypoint_interface.enabled: true or false")
        return en

    def _waypoints(self):
        wp = self.agent["waypoint_interface"]
        key = f"agents.{self.agent_name}.waypoint_interface"
        bad = set(wp) - WP_KEYS
        if bad:
            raise ScenarioError(f"{key}: unknown keys {sorted(bad)} (known: "
                                f"{sorted(WP_KEYS)})")
        loop = wp.get("loop", False)
        if not isinstance(loop, bool):
            raise ScenarioError(f"{key}.loop: true or false")
        if "radius" in wp and "range_tolerance" in wp:
            raise ScenarioError(f"{key}: radius or range_tolerance (the "
                                f"same thing), not both")
        rk = "range_tolerance" if "range_tolerance" in wp else "radius"
        radius = _number(wp.get(rk, 20.0), f"{key}.{rk}", lo=0.0,
                         lo_open=True)
        given = [k for k in ("path", "path_enu") if k in wp]
        mode = wp.get("mode")
        if mode is None and "waypoints" in wp:
            mode = "waypoints"
        if given and mode is not None:
            raise ScenarioError(f"{key}: LOTUSim's mode / waypoints or our "
                                f"{given[0]}, not both")
        if len(given) > 1:
            raise ScenarioError(f"{key}: path or path_enu, not both")
        x0, y0 = self.start_xy()

        def seq(name):
            v = wp.get(name)
            if not isinstance(v, (list, tuple)) or not v:
                raise ScenarioError(f"{key}.{name}: a non-empty list")
            return v
        if given == ["path_enu"]:
            pts = [self._xy(p, f"{key}.path_enu[{i}]", metres_list=True)
                   for i, p in enumerate(seq("path_enu"))]
        elif given == ["path"]:
            pts = [self._xy(p, f"{key}.path[{i}]")
                   for i, p in enumerate(seq("path"))]
        elif mode == "waypoints":
            pts = [self._xy(p, f"{key}.waypoints[{i}]")
                   for i, p in enumerate(seq("waypoints"))]
        elif mode == "line":
            ln = wp.get("line")
            if not isinstance(ln, dict):
                raise ScenarioError(f"{key}.line: {{direction (rad), length "
                                    f"(m)}}")
            d = _number(ln.get("direction"), f"{key}.line.direction")
            L = _number(ln.get("length"), f"{key}.line.length", lo=0.0,
                        lo_open=True)
            pts = [(x0 + L * math.cos(d), y0 + L * math.sin(d))]
        elif mode == "circle":
            c = wp.get("circle")
            if not isinstance(c, dict):
                raise ScenarioError(f"{key}.circle: {{radius (m)}}")
            r = _number(c.get("radius"), f"{key}.circle.radius", lo=0.0,
                        lo_open=True)
            # LOTUSim's waypoint_follower: start + r (cos a, sin a),
            # a = 0, pi/4, ... 7 pi/4 (the circle is centred on the start)
            pts = [(x0 + r * math.cos(2 * math.pi * i / 8),
                    y0 + r * math.sin(2 * math.pi * i / 8))
                   for i in range(8)]
        elif mode is None:
            raise ScenarioError(f"{key} is enabled (LOTUSim's default when "
                                f"the block is present) but says nowhere to "
                                f"go: give mode waypoints / line / circle, "
                                f"or path_enu / path")
        else:
            raise ScenarioError(f"{key}.mode {mode!r}: waypoints | line | "
                                f"circle")
        return dict(kind="waypoints", points=pts, radius=radius, loop=loop)

    def track_spec(self):
        """dict(kind='line', heading) or dict(kind='waypoints', points,
        radius, loop)."""
        if self._wp_enabled():
            return self._waypoints()
        tr = self.agent.get("track") or {}
        if not isinstance(tr, dict) or set(tr) - {"heading", "heading_rad"}:
            raise ScenarioError(f"agents.{self.agent_name}.track: "
                                f"{{heading (deg)}} or {{heading_rad}}")
        if "heading_rad" in tr:
            h = _number(tr["heading_rad"], "track.heading_rad")
        elif "heading" in tr:
            h = geo.deg_to_rad_wrapped(_number(tr["heading"],
                                               "track.heading"))
        else:
            h = self.start_yaw()
        return dict(kind="line", heading=h)

    def sensor_config(self):
        return copy.deepcopy(self.agent.get("sensors") or {})

    def policy_spec(self):
        return copy.deepcopy(self.agent.get("policy"))

    # ------------------------------------------------------------ edits
    def with_overrides(self, seed=None, duration=None, model=None,
                       policy=None, **usv):
        raw = copy.deepcopy(self.raw)
        u = raw.setdefault("usv", {})
        if seed is not None:
            u["seed"] = as_seed(seed, "seed")
        if duration is not None:
            u["duration"] = _number(duration, "duration", lo=0.0,
                                    lo_open=True)
        for k, v in usv.items():
            u[k] = v
        ag = raw["agents"][self.agent_name]
        if model is not None:
            if not isinstance(model, (str, dict)):
                raise ScenarioError(f"model: a hull key or an inline hull, "
                                    f"not {model!r}")
            ag["model"] = model
        if policy is not None:
            ag["policy"] = policy
        return Scenario(raw, self.source)

    def to_dict(self):
        return copy.deepcopy(self.raw)

    def sha256(self):
        return hashlib.sha256(canonical(self.raw).encode()).hexdigest()

    def __eq__(self, other):
        return isinstance(other, Scenario) and canonical(self.raw) == \
            canonical(other.raw)

    def __repr__(self):
        return (f"Scenario({self.name!r}, hull {self.hull_key!r}, "
                f"{self.duration:g} s, seed {self.seed})")


def canonical(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _yaml():
    try:
        import yaml
        return yaml
    except ImportError:
        return None


def load(src):
    """A Scenario from a path (.json, .yaml/.yml), a dict, or a Scenario."""
    if isinstance(src, Scenario):
        return src
    if isinstance(src, dict):
        return Scenario(src)
    path = os.fspath(src)
    if not os.path.isabs(path) and not os.path.exists(path):
        from . import ROOT, SCENARIOS
        for base in (SCENARIOS, ROOT):
            cand = os.path.join(base, path)
            if os.path.exists(cand):
                path = cand
                break
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        y = _yaml()
        if y is None:
            raise ScenarioError("PyYAML is not installed: use the JSON form "
                                "(same keys)")
        doc = y.safe_load(text)
    else:
        doc = json.loads(text)
    return Scenario(doc, source=os.path.abspath(path))


def save(sc, path):
    sc = load(sc)
    with open(path, "w", encoding="utf-8") as f:
        if path.lower().endswith((".yaml", ".yml")):
            y = _yaml()
            if y is None:
                raise ScenarioError("PyYAML is not installed")
            y.safe_dump(sc.to_dict(), f, sort_keys=False)
        else:
            json.dump(sc.to_dict(), f, indent=1)
    return path


def straight(model=AGENT_DEFAULTS["model"], hs=1.0, tp=5.0, theta0=math.pi,
             heading=0.0, duration=60.0, seed=0, plant="gz", **usv):
    """A minimal straight-track scenario in code (tests, CLI); the model
    defaults to the frozen configuration's (app/frozen.MODEL)."""
    doc = dict(
        name=f"straight_{model}",
        environment=dict(ocean=dict(wave_height=hs, wave_period=tp,
                                    theta0=theta0)),
        agents={"usv0": dict(model=model, plant=plant, heading_rad=heading)},
        usv=dict(duration=duration, seed=seed, **usv))
    return Scenario(doc)
