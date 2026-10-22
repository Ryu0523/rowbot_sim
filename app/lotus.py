"""
LOTUSim-shaped JSON messages, and the lockstep session every external
interface (websocket, UDP, HTTP, the ROS 2 bridge) shares.

SHAPES (LOTUSim/interfaces/lotusim_msgs and lotusim_sensor_msgs, as ROS 2
messages rendered to JSON field by field):

  VesselPositionArray  {header {stamp {sec, nanosec}, frame_id},
                        vessels [{vessel_name, model_name,
                                  pose {position {x, y, z},
                                        orientation {x, y, z, w}},
                                  geo_point {latitude, longitude,
                                             altitude}}]}
      pose: ENU position of the CG (z = height above calm water),
      orientation FLU (LOTUSim: "Orientation uses front-left-up") with the
      total pitch. The "poses" object is EXACTLY {header, vessels}, so a
      strict ROS converter (rosidl_runtime_py.set_message_fields,
      rosbridge) accepts it as it is. LOTUSim's message has no velocity and
      no time base beyond the header; our extras (velocity in the body
      frame, actuators, sim time, leg, reward, end flags) are the reply's
      sibling key "usv" (usv_extras), never inside "poses".
  VesselCmdArray       {cmds [{header?, vessel_name, entity?, cmd_string}]}
      cmd_string is a JSON object in xdyn's "name(unit)" style:
        "jet(T)"        thrust, N                 or "throttle": fraction
        "nozzle(rad)" / "nozzle(deg)"            or "nozzle": fraction
        "heading(rad)" / "heading(deg)"  ENU, CCW from east (LOTUSim's
                        agent heading convention): the autopilot steers
      Unknown keys are an error (a typo must not be ignored), and so is a
      value that is not a finite number (NaN, inf, a string): the step is
      refused with an error reply and the episode stays as it was.
  GPS                  {header, status {status, service}, latitude,
                        longitude, altitude, position_covariance[9],
                        position_covariance_type, linear_velocity {x,y,z}}
                        (velocity ENU, LOTUSim's addition to NavSatFix)
  Imu                  sensor_msgs/Imu: orientation, angular_velocity,
                        linear_acceleration (+ covariances)

SESSION PROTOCOL (one JSON object per message)
  {"op": "reset", "scenario": "demo_scarab195.json" | {...},
   "seed": 0, "duration": 60, "model": "scarab195@as1"}   -> reset_ok
  {"op": "step", "cmds": [{"vessel_name": "usv0", "cmd_string": "..."}]}
  {"op": "step", "cmd_string": "..."}  or  {"op": "step", "action": {...}}
                                                           -> state
  {"op": "state"}                                          -> state
  {"op": "save", "name": "my_run"}                         -> saved
  {"op": "close"}
Replies: {"op": "state", "t", "k", "poses": VesselPositionArray, "usv":
our extras, "gps": GPS, "imu": Imu, "sensors": {name: last delivered
reading}, "reward", "terminated", "truncated", "reason"}; errors {"op":
"error", "error"}.

SESSIONS AND RETRIES (SessionTable; HTTP and UDP -- a websocket connection
is its own session)
  session  HTTP: the reset reply carries "session": an id; every later
           message of that client carries it ("session": id). Each session
           has its own lock, so concurrent clients never step each other's
           boat. Sessions idle for idle_s (default 600 s) are dropped.
           UDP: the client address is the session.
  seq      optional integer in any message, echoed in the reply. A message
           whose seq equals the last one of its session is a RETRY: the
           cached reply is sent again and nothing is stepped (a lost reply
           must not step the boat twice). Otherwise seq must be the last + 1
           (a reset may start any sequence); a gap is refused with an error
           and nothing is done.
"""
import json
import math
import os
import re
import threading
import time
import uuid

from . import RUNS, SCENARIOS, geo

CMD_KEYS = {"jet(T)", "throttle", "nozzle(rad)", "nozzle(deg)", "nozzle",
            "heading(rad)", "heading(deg)"}


def stamp(t):
    sec = int(math.floor(t))
    ns = int(round((t - sec) * 1e9))
    if ns >= 1_000_000_000:
        sec, ns = sec + 1, ns - 1_000_000_000
    return dict(sec=sec, nanosec=ns)


def header(t, frame_id="enu"):
    return dict(stamp=stamp(t), frame_id=frame_id)


def vessel_position(env, vessel_name="usv0"):
    p = env.pose()
    e, n, u = p["position"]
    qx, qy, qz, qw = p["quaternion"]
    lat, lon, alt = p["geo"]
    return dict(vessel_name=vessel_name, model_name=env.hull_key,
                pose=dict(position=dict(x=e, y=n, z=u),
                          orientation=dict(x=qx, y=qy, z=qz, w=qw)),
                geo_point=dict(latitude=lat, longitude=lon, altitude=alt))


def vessel_position_array(env, vessel_name="usv0"):
    """lotusim_msgs/VesselPositionArray as JSON: exactly {header,
    vessels}."""
    return dict(header=header(env.t), vessels=[vessel_position(
        env, vessel_name)])


def usv_extras(env):
    """What LOTUSim's message lacks (the reply's sibling key "usv")."""
    p = env.pose()
    c = env.last_cmd
    lv, av = p["lin_vel_body"], p["ang_vel_body"]
    return dict(t=env.t, k=env.k,
                twist=dict(linear=dict(x=lv[0], y=lv[1], z=lv[2]),
                           angular=dict(x=av[0], y=av[1], z=av[2])),
                rpy=dict(roll=p["rpy"][0], pitch=p["rpy"][1],
                         yaw=p["rpy"][2]),
                actuators=dict(thrust=float(env.s[12]),
                               nozzle=float(env.s[13]),
                               thrust_cmd=float(c["thrust"]),
                               nozzle_cmd=float(c["nozzle"]),
                               heading_ref=_num(c.get("heading")),
                               safe_mode=bool(c.get("safe"))),
                leg=env.track.leg, n_legs=env.track.n_legs,
                reward=env.last_reward, capsized=env.capsized,
                finite=env.finite, reason=env.reason,
                physics=env.stamp()["physics"])


def _num(x):
    x = float(x)
    return None if math.isnan(x) else x


def finite_number(v, what):
    """float(v), refusing what is not a finite number (NaN, inf, bool,
    strings)."""
    if isinstance(v, bool):
        raise ValueError(f"{what}: a number, not {v!r}")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: a number, not {v!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"non-finite action: {what} = {x}")
    return x


def gps(env, name="gnss"):
    sn = env.sensors.get(name)
    r = sn.reading() if sn is not None else None
    if r is None:
        return None
    sh, sv = sn.sigma[3], sn.sigma[5]
    return dict(header=header(r["t"], "gnss"),
                status=dict(status=0 if r["valid"] else -1, service=1),
                latitude=r["lat"], longitude=r["lon"], altitude=r["alt"],
                position_covariance=[sh * sh, 0, 0, 0, sh * sh, 0, 0, 0,
                                     sv * sv],
                position_covariance_type=2,
                linear_velocity=dict(x=r["ve"], y=r["vn"], z=r["vu"]))


def imu(env, name="imu"):
    sn = env.sensors.get(name)
    r = sn.reading() if sn is not None else None
    if r is None:
        return None
    qx, qy, qz, qw = geo.quat_zyx(r["roll"], r["pitch"], r["yaw"])
    sa, sw, sf = sn.sigma[6], sn.sigma[3], sn.sigma[0]
    return dict(header=header(r["t"], "base_link"),
                orientation=dict(x=qx, y=qy, z=qz, w=qw),
                orientation_covariance=[sa * sa, 0, 0, 0, sa * sa, 0, 0, 0,
                                        sn.sigma[8] ** 2],
                angular_velocity=dict(x=r["wx"], y=r["wy"], z=r["wz"]),
                angular_velocity_covariance=[sw * sw, 0, 0, 0, sw * sw, 0, 0,
                                             0, sw * sw],
                linear_acceleration=dict(x=r["fx"], y=r["fy"], z=r["fz"]),
                linear_acceleration_covariance=[sf * sf, 0, 0, 0, sf * sf, 0,
                                                0, 0, sf * sf],
                valid=bool(r["valid"]))


def parse_cmd(msg, env=None, vessel_name="usv0"):
    """An env action (dict with physical units / fractions) from a
    VesselCmdArray, a VesselCmd, a cmd_string or an action dict."""
    if isinstance(msg, str):
        msg = {"cmd_string": msg}
    if "action" in msg:
        return msg["action"]
    if "cmds" in msg:
        mine = [c for c in msg["cmds"]
                if c.get("vessel_name") in (None, "", vessel_name)]
        if len(mine) != 1:
            raise ValueError(f"VesselCmdArray: expected one command for "
                             f"{vessel_name!r}, got {len(mine)}")
        msg = mine[0]
    cs = msg.get("cmd_string")
    if cs is None:
        raise ValueError("no cmd_string / cmds / action in the message")
    d = json.loads(cs) if isinstance(cs, str) else dict(cs)
    if not isinstance(d, dict):
        raise ValueError("cmd_string: a JSON object")
    bad = set(d) - CMD_KEYS
    if bad:
        raise ValueError(f"cmd_string keys {sorted(bad)} unknown; use "
                         f"{sorted(CMD_KEYS)}")
    a = {}
    if "jet(T)" in d and "throttle" in d:
        raise ValueError("cmd_string: jet(T) or throttle, not both")
    if "jet(T)" in d:
        a["thrust"] = finite_number(d["jet(T)"], "jet(T)")
    elif "throttle" in d:
        a["throttle"] = finite_number(d["throttle"], "throttle")
    else:
        raise ValueError("cmd_string needs jet(T) or throttle")
    steer = [k for k in ("nozzle(rad)", "nozzle(deg)", "nozzle",
                         "heading(rad)", "heading(deg)") if k in d]
    if len(steer) != 1:
        raise ValueError("cmd_string needs exactly one of nozzle(rad), "
                         "nozzle(deg), nozzle, heading(rad), heading(deg)")
    k = steer[0]
    v = finite_number(d[k], k)
    if k == "nozzle(rad)":
        a["nozzle_rad"] = v
    elif k == "nozzle(deg)":
        a["nozzle_rad"] = math.radians(v)
    elif k == "nozzle":
        a["nozzle"] = v
    elif k == "heading(rad)":
        a["heading"] = v
    else:
        a["heading"] = geo.deg_to_rad_wrapped(v)
    return a


def cmd_array(vessel_name="usv0", **cmd):
    """A VesselCmdArray dict for a client: cmd_array(throttle=0.7,
    **{'heading(deg)': 0})."""
    return dict(cmds=[dict(vessel_name=vessel_name,
                           cmd_string=json.dumps(cmd))])


_SAFE = re.compile(r"^[A-Za-z0-9_.@\-]+$")


def scenario_path(name):
    """A scenario file inside app/scenarios (no path traversal)."""
    base = os.path.basename(str(name))
    if base != str(name) or not _SAFE.match(base):
        raise ValueError(f"scenario {name!r}: a file name in app/scenarios")
    if not base.endswith((".json", ".yaml", ".yml")):
        base += ".json"
    p = os.path.join(SCENARIOS, base)
    if not os.path.exists(p):
        raise ValueError(f"no scenario {base} in app/scenarios")
    return p


class Session:
    """One external client's lockstep episode. Not thread-safe by itself:
    SessionTable (HTTP, UDP) holds one lock per session, and a websocket
    connection is served by one thread."""

    def __init__(self, vessel_name="usv0", allow_inline=True, runs_dir=RUNS,
                 refuse_stale=False):
        self.env = None
        self.vessel_name = vessel_name
        self.allow_inline = allow_inline
        self.runs_dir = runs_dir
        # a long-running server refuses a reset on physics code that changed
        # on disk since it loaded it (app/codeguard.py)
        self.refuse_stale = refuse_stale
        self._key = None

    def reply_state(self, reward=None, term=False, trunc=False):
        env = self.env
        return dict(op="state", t=env.t, k=env.k,
                    poses=vessel_position_array(env, self.vessel_name),
                    usv=usv_extras(env),
                    gps=gps(env), imu=imu(env),
                    sensors={n: s.reading() for n, s in env.sensors.items()},
                    reward=env.last_reward if reward is None else reward,
                    terminated=bool(term), truncated=bool(trunc),
                    reason=env.reason)

    def handle(self, msg):
        try:
            if isinstance(msg, (bytes, str)):
                msg = json.loads(msg)
            op = msg.get("op")
            if op == "reset":
                return self._reset(msg)
            if op == "step":
                if self.env is None:
                    raise ValueError("reset first")
                a = parse_cmd(msg, self.env, self.vessel_name)
                obs, r, te, tr, info = self.env.step(a)
                return self.reply_state(r, te, tr)
            if op == "state":
                if self.env is None:
                    raise ValueError("reset first")
                return self.reply_state()
            if op == "save":
                if self.env is None:
                    raise ValueError("reset first")
                name = re.sub(r"[^A-Za-z0-9_\-]", "_",
                              str(msg.get("name", "session")))[:100]
                rec = self.env.record(policy=dict(spec={"type": "external"},
                                                  name="external"))
                # never overwrite a record (the demos live in the same
                # folder): name, name_2, name_3 ...
                base, n = name, 1
                while os.path.exists(os.path.join(self.runs_dir,
                                                  name + ".npz")):
                    n += 1
                    name = f"{base}_{n}"
                path = rec.save(os.path.join(self.runs_dir, name + ".npz"))
                return dict(op="saved", name=name, path=os.path.relpath(
                    path, os.path.dirname(os.path.dirname(RUNS))).replace(
                        "\\", "/"), fingerprint=rec.fingerprint())
            if op == "close":
                self.env = None
                return dict(op="closed")
            raise ValueError(f"op {op!r}: reset | step | state | save | "
                             f"close")
        except Exception as ex:                      # noqa: BLE001
            return dict(op="error", error=f"{type(ex).__name__}: {ex}")

    def _reset(self, msg):
        from .env import UsvEnv
        from .scenario import load
        sc = msg.get("scenario", "demo_scarab195.json")
        if isinstance(sc, dict):
            if not self.allow_inline:
                raise ValueError("inline scenarios are not accepted here")
            sc = load(sc)
        else:
            sc = load(scenario_path(sc))
        sc = sc.with_overrides(seed=msg.get("seed"),
                               duration=msg.get("duration"),
                               model=msg.get("model"))
        key = sc.sha256()
        if self.refuse_stale:
            from . import codeguard, hulls
            from sim import physics
            codeguard.check(physics.version(hulls.resolve(sc.hull_spec)),
                            refuse=True)
        if self.env is None or key != self._key:
            self.env = UsvEnv(sc, action_mode="direct")
            self._key = key
        obs, info = self.env.reset(seed=sc.seed)
        env = self.env
        out = self.reply_state()
        out.update(op="reset_ok", info=dict(
            hull=env.hull_key, physics=env.stamp(), plant=type(
                env.plant).__name__, dt=env.dt, sub=env.sub,
            dt_step=env.dt_step, n_ctrl=env.n_ctrl, t_max=env.t_max,
            t_min=env.t_min, nozzle_max=env.rud_max,
            reference_position=env.frame.as_dict(),
            track=env.track.describe(), seed=env.seed_,
            sea=dict(hs=env.sea.hs, tp=env.sea.tp,
                     theta0=float(env.sea.theta0),
                     n_components=int(env.sea.a.size)),
            cmd_keys=sorted(CMD_KEYS)))
        return out


class _Slot:
    __slots__ = ("session", "lock", "last", "seq", "reply")

    def __init__(self, session):
        self.session, self.lock = session, threading.Lock()
        self.last, self.seq, self.reply = time.monotonic(), None, None


class SessionTable:
    """Sessions of many clients behind one endpoint (HTTP, UDP): one
    Session and one lock per key, idle eviction, and the seq rule of the
    module docstring (idempotent retries, gaps refused).

    handle(key, msg) -> reply dict. key None with op 'reset' makes a new
    session (an uuid); `new_key` then says how keys are made."""

    def __init__(self, factory, idle_s=600.0, max_sessions=32):
        self.factory = factory
        self.idle_s, self.max_sessions = float(idle_s), int(max_sessions)
        self._d = {}
        self._lock = threading.Lock()

    def __len__(self):
        return len(self._d)

    def evict(self, now=None):
        """Drop sessions idle for more than idle_s (not while in use)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            for k, sl in list(self._d.items()):
                if now - sl.last > self.idle_s and not sl.lock.locked():
                    del self._d[k]

    def _slot(self, key, create):
        with self._lock:
            sl = self._d.get(key)
            if sl is None and create:
                if len(self._d) >= self.max_sessions:
                    raise ValueError(f"too many sessions ({len(self._d)}); "
                                     f"close one ('op': 'close') or wait "
                                     f"{self.idle_s:.0f} s for idle ones "
                                     f"to expire")
                sl = self._d[key] = _Slot(self.factory())
            return sl

    def handle(self, key, msg, create_any=False):
        """create_any: any message may open a session (UDP: the address is
        the key); otherwise only a reset without a key does (HTTP)."""
        self.evict()
        seq = None
        try:
            if isinstance(msg, (bytes, bytearray, str)):
                msg = json.loads(msg)
            if not isinstance(msg, dict):
                raise ValueError("a message is a JSON object")
            op = msg.get("op")
            seq = msg.get("seq")
            if seq is not None and (isinstance(seq, bool) or
                                    not isinstance(seq, int)):
                raise ValueError(f"seq: an integer, not {seq!r}")
            if key is None:
                if op != "reset":
                    raise ValueError("no session: send a reset first and "
                                     "carry its 'session' id in every "
                                     "message")
                key = uuid.uuid4().hex
                sl = self._slot(key, True)
            else:
                sl = self._slot(key, create_any)
                if sl is None:
                    raise ValueError(f"unknown or expired session {key!r} "
                                     f"(sessions idle for {self.idle_s:.0f}"
                                     f" s are dropped): reset again")
        except Exception as ex:                      # noqa: BLE001
            out = dict(op="error", error=f"{type(ex).__name__}: {ex}")
            if seq is not None:
                out["seq"] = seq
            return out
        with sl.lock:
            sl.last = time.monotonic()
            if seq is not None and sl.seq is not None:
                if seq == sl.seq and sl.reply is not None:
                    return dict(sl.reply, retry=True)
                if op != "reset" and seq != sl.seq + 1:
                    return dict(op="error", seq=seq, session=_pub(key),
                                error=f"seq {seq} out of order: expected "
                                      f"{sl.seq + 1} (or {sl.seq} to get "
                                      f"the last reply again); nothing "
                                      f"was done")
            out = sl.session.handle(msg)
            if out.get("op") == "error" and sl.session.env is None:
                # a session that never got an episode (its first reset
                # failed, or junk from a new address) is not kept
                with self._lock:
                    self._d.pop(key, None)
                if seq is not None:
                    out["seq"] = seq
                return out
            out["session"] = _pub(key)
            if seq is not None:
                out["seq"] = seq
                sl.seq, sl.reply = seq, out
            if op == "close":
                with self._lock:
                    self._d.pop(key, None)
            sl.last = time.monotonic()
            return out


def _pub(key):
    return key if isinstance(key, str) else ":".join(map(str, key))
