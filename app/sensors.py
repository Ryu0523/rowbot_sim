"""
Sensor models: what a controller on the real boat would read instead of the
plant's state.

    imu         specific force (body FLU, m/s^2), body angular rate (rad/s),
                attitude roll / pitch / yaw (rad; an AHRS output)
    gnss        latitude, longitude, altitude (WGS84) and the same position
                and velocity in ENU (m, m/s)
    heading     yaw (rad, ENU) and the compass bearing (deg, CW from north)
    speed_log   speed through the water along the hull (m/s)
    wave_probe  incident wave elevation at body-fixed points (m)
    preview     incident wave elevation at points ahead along the heading
                (a wave radar / lidar; m), error growing with range

WHAT IS MEASURED, EXACTLY
  Specific force is what an accelerometer integrates: over a sample interval
  of length h it reports (V(t) - V(t - h)) / h - g, g = (0, 0, -9.81) in ENU,
  rotated into the body frame at the end of the interval. V is the world
  velocity of the plant's kinematics (u cos psi - v sin psi, u sin psi +
  v cos psi, w). Angular rate: the plant integrates the Euler angles with
  rates (p, q, r) (sim/planing_vessel.deriv: d(roll, pitch, yaw)/dt = p, q,
  r), so the gyro reports the body rate of those Euler rates through the
  exact ZYX kinematic matrix (app/geo.euler_rates_to_body). Attitude uses
  the TOTAL pitch Th_rest + pitch (positive bow down, FLU). GNSS height is
  the CG (plus lever arm) above calm water: Z_rest + z.

ERRORS (every one seeded; noise off = the truth)
  value = scale * truth + bias + noise, per channel, where
  noise ~ N(0, sigma^2) independent per sample; bias = b0 + random walk,
  b0 ~ N(0, bias_sd^2) drawn at reset, the walk adding N(0, bias_rw^2 h)
  per sample interval h; scale = 1 + N(0, scale_sd^2) at reset (speed log).
  A sample is lost with probability `dropout` (delivered with valid = 0).
  Latency: a sample taken at t is delivered at t + latency.
  Sampling and latency live on the plant-step grid: the period is
  round(1 / (rate dt)) plant steps and the latency round(latency / dt)
  steps -- exact and deterministic (the actual values are reported).
  The random draws per sample are always taken, in a fixed order (walk,
  noise, dropout), so a dropout never shifts a stream; each sensor has its
  own stream, np.random.default_rng([sensor_seed, index]), independent of
  the sea's and the controllers'.

CONFIG (scenario agents.<name>.sensors.<sensor>; null disables a sensor;
"noise": false at the top switches every random error off). Checked by
validate_config when the scenario loads: rate_hz > 0 or null (= every
plant step), latency_s >= 0, 0 <= dropout <= 1, every sigma / bias / walk /
scale >= 0, preview distances >= 0 and distinct to the centimetre (their
channels are named eta_<distance>m).
  common   rate_hz, latency_s, dropout
  imu      sigma_f, sigma_w, sigma_att (roll/pitch), sigma_yaw, bias_f,
           bias_w, rw_f, rw_w
  gnss     sigma_h, sigma_v, sigma_vel, bias_h, rw_h, lever [x, y, z]
  heading  sigma, bias
  speed_log sigma, bias, scale_sd
  wave_probe sigma, points [[xb, yb], ...] (body, from the CG; default the
           stem and the CG)
  preview  sigma0, sigma_per_m, distances [m] (default 1..5 boat lengths)
"""
import collections
import math

import numpy as np

from . import geo

G = 9.81
D2R = math.pi / 180.0

DEFAULTS = {
    "imu": dict(rate_hz=50.0, latency_s=0.0, dropout=0.0, sigma_f=0.05,
                sigma_w=0.002, sigma_att=0.2 * D2R, sigma_yaw=0.5 * D2R,
                bias_f=0.02, bias_w=5e-4, rw_f=0.0, rw_w=1e-4),
    "gnss": dict(rate_hz=10.0, latency_s=0.1, dropout=0.01, sigma_h=1.0,
                 sigma_v=2.0, sigma_vel=0.05, bias_h=0.5, rw_h=0.05,
                 lever=[0.0, 0.0, 0.0]),
    "heading": dict(rate_hz=10.0, latency_s=0.04, dropout=0.0,
                    sigma=0.3 * D2R, bias=0.5 * D2R),
    "speed_log": dict(rate_hz=5.0, latency_s=0.0, dropout=0.0, sigma=0.1,
                      bias=0.0, scale_sd=0.02),
    "wave_probe": dict(rate_hz=10.0, latency_s=0.0, dropout=0.0,
                       sigma=0.02, points=None),
    "preview": dict(rate_hz=5.0, latency_s=0.0, dropout=0.0, sigma0=0.02,
                    sigma_per_m=0.002, distances=None),
}
ORDER = ("imu", "gnss", "heading", "speed_log", "wave_probe", "preview")


def world_velocity(s):
    c, sn = math.cos(s[5]), math.sin(s[5])
    return np.array([s[6] * c - s[7] * sn, s[6] * sn + s[7] * c, s[8]])


class Sensor:
    kind = ""

    def __init__(self, name, cfg, env, index):
        self.name, self.index = name, int(index)
        self.cfg = dict(cfg)
        dt = env.dt
        rate = cfg.get("rate_hz")
        self.period = 1 if rate is None else max(1, int(round(
            1.0 / (float(rate) * dt))))
        self.lat_steps = max(0, int(round(float(cfg.get("latency_s", 0.0))
                                          / dt)))
        self.rate_hz = 1.0 / (self.period * dt)
        self.latency_s = self.lat_steps * dt
        self.dropout = float(cfg.get("dropout", 0.0))
        self.dt = dt
        self.setup(env)
        n = len(self.channels)
        self.sigma = np.broadcast_to(np.asarray(self.sigma, float), (n,)).copy()
        self.bias_sd = np.broadcast_to(np.asarray(self.bias_sd, float),
                                       (n,)).copy()
        self.bias_rw = np.broadcast_to(np.asarray(self.bias_rw, float),
                                       (n,)).copy()
        self.scale_sd = np.broadcast_to(np.asarray(getattr(
            self, "scale_sd_ch", 0.0), float), (n,)).copy()

    # subclasses: channels, setup(env) sets sigma / bias_sd / bias_rw,
    # truth(env, t, s) -> array
    def setup(self, env):
        raise NotImplementedError

    def truth(self, env, t, s):
        raise NotImplementedError

    def wrap_channels(self):
        return ()

    # -------------------------------------------------------- the loop
    def reset(self, seed, env, t0, s0):
        self.rng = np.random.default_rng([int(seed), self.index])
        n = len(self.channels)
        self.b = self.rng.normal(0.0, 1.0, n) * self.bias_sd
        self.scale = 1.0 + self.rng.normal(0.0, 1.0, n) * self.scale_sd
        self.queue = collections.deque()
        self.log_t, self.log_ta, self.log_ok, self.log_v = [], [], [], []
        self.last = None            # (t_meas, valid, values) last delivered
        self.last_valid = None      # (t_meas, values) last valid delivered
        self.t_prev = t0
        self.prep(env, t0, s0)
        self.sample(env, 0, t0, s0)

    def prep(self, env, t0, s0):
        pass

    def errors(self, h):
        n = len(self.channels)
        self.b = self.b + self.rng.normal(0.0, 1.0, n) * self.bias_rw * \
            math.sqrt(max(h, 0.0))
        noise = self.rng.normal(0.0, 1.0, n) * self.sigma
        ok = bool(self.rng.random() >= self.dropout)
        return noise, ok

    def measure(self, env, t, s, h):
        tr = np.asarray(self.truth(env, t, s), float)
        noise, ok = self.errors(h)
        v = self.scale * tr + self.b + noise
        for i in self.wrap_channels():
            v[i] = math.remainder(v[i], 2 * math.pi)
        return v, ok

    def sample(self, env, step, t, s):
        """Called at every plant step `step` (0 = the reset state)."""
        if step % self.period:
            return
        h = t - self.t_prev
        v, ok = self.measure(env, t, s, h)
        self.t_prev = t
        self.queue.append((step + self.lat_steps, t, ok, v))
        self.log_t.append(t)
        self.log_ta.append(t + self.latency_s)
        self.log_ok.append(ok)
        self.log_v.append(v)

    def deliver(self, step):
        """Deliver every sample due by plant step `step`."""
        while self.queue and self.queue[0][0] <= step:
            _, tm, ok, v = self.queue.popleft()
            self.last = (tm, ok, v)
            if ok:
                self.last_valid = (tm, v)

    def obs(self, t):
        """[valid, age, values...]: the values of the last VALID delivered
        sample and its age; valid says whether the latest delivery was
        valid. Never delivered: valid 0, age -1, zeros."""
        n = len(self.channels)
        if self.last_valid is None:
            return np.concatenate([[0.0, -1.0], np.zeros(n)])
        tm, v = self.last_valid
        return np.concatenate([[1.0 if self.last[1] else 0.0, t - tm], v])

    def reading(self):
        """The last delivered sample as a dict (None before the first)."""
        if self.last is None:
            return None
        tm, ok, v = self.last
        return dict(t=tm, valid=ok, **{c: float(x) for c, x in
                                       zip(self.channels, v)})

    def arrays(self):
        n = len(self.channels)
        return dict(t_meas=np.asarray(self.log_t, float),
                    t_avail=np.asarray(self.log_ta, float),
                    valid=np.asarray(self.log_ok, bool),
                    values=(np.asarray(self.log_v, float).reshape(-1, n)))

    def describe(self):
        return dict(kind=self.kind, channels=list(self.channels),
                    rate_hz=self.rate_hz, latency_s=self.latency_s,
                    dropout=self.dropout, sigma=self.sigma.tolist(),
                    bias_sd=self.bias_sd.tolist(),
                    bias_rw=self.bias_rw.tolist(),
                    scale_sd=self.scale_sd.tolist(), cfg=self.cfg)


class IMU(Sensor):
    kind = "imu"
    channels = ("fx", "fy", "fz", "wx", "wy", "wz", "roll", "pitch", "yaw")

    def setup(self, env):
        c = self.cfg
        self.sigma = [c["sigma_f"]] * 3 + [c["sigma_w"]] * 3 + \
            [c["sigma_att"]] * 2 + [c["sigma_yaw"]]
        self.bias_sd = [c["bias_f"]] * 3 + [c["bias_w"]] * 3 + [0.0] * 3
        self.bias_rw = [c["rw_f"]] * 3 + [c["rw_w"]] * 3 + [0.0] * 3

    def wrap_channels(self):
        return (8,)

    def prep(self, env, t0, s0):
        self.v_prev = world_velocity(s0)

    def truth(self, env, t, s):
        v = world_velocity(s)
        h = t - self.t_prev
        a = (v - self.v_prev) / h if h > 0 else np.zeros(3)
        self.v_prev = v
        pitch = env.plant.Th_rest + s[4]
        R = geo.rot_zyx(s[3], pitch, s[5])
        f = R.T @ (a + np.array([0.0, 0.0, G]))
        w = geo.euler_rates_to_body(s[3], pitch, s[9], s[10], s[11])
        return np.concatenate([f, w, [s[3], pitch,
                                      math.remainder(s[5], 2 * math.pi)]])


class GNSS(Sensor):
    kind = "gnss"
    channels = ("lat", "lon", "alt", "e", "n", "u", "ve", "vn", "vu")

    def setup(self, env):
        c = self.cfg
        self.sigma = [0.0] * 3 + [c["sigma_h"]] * 2 + [c["sigma_v"]] + \
            [c["sigma_vel"]] * 3
        self.bias_sd = [0.0] * 3 + [c["bias_h"]] * 2 + [0.0] * 4
        self.bias_rw = [0.0] * 3 + [c["rw_h"]] * 2 + [0.0] * 4
        self.lever = np.asarray(c.get("lever") or [0.0, 0.0, 0.0], float)
        self.frame = env.frame

    def truth(self, env, t, s):
        pitch = env.plant.Th_rest + s[4]
        R = geo.rot_zyx(s[3], pitch, s[5])
        p = np.array([s[0], s[1], env.plant.Z_rest + s[2]]) + R @ self.lever
        lat, lon, alt = self.frame.to_geodetic(*p)
        return np.concatenate([[lat, lon, alt], p, world_velocity(s)])

    def measure(self, env, t, s, h):
        tr = np.asarray(self.truth(env, t, s), float)
        noise, ok = self.errors(h)
        v = tr + self.b + noise
        # the geodetic channels are the noisy ENU position, converted
        v[0], v[1], v[2] = self.frame.to_geodetic(v[3], v[4], v[5])
        return v, ok


class Heading(Sensor):
    kind = "heading"
    channels = ("yaw", "bearing_deg")

    def setup(self, env):
        c = self.cfg
        self.sigma = [c["sigma"], 0.0]
        self.bias_sd = [c["bias"], 0.0]
        self.bias_rw = [0.0, 0.0]

    def truth(self, env, t, s):
        y = math.remainder(s[5], 2 * math.pi)
        return np.array([y, geo.yaw_to_bearing_deg(y)])

    def measure(self, env, t, s, h):
        tr = np.asarray(self.truth(env, t, s), float)
        noise, ok = self.errors(h)
        y = math.remainder(tr[0] + self.b[0] + noise[0], 2 * math.pi)
        return np.array([y, geo.yaw_to_bearing_deg(y)]), ok


class SpeedLog(Sensor):
    kind = "speed_log"
    channels = ("stw",)

    def setup(self, env):
        c = self.cfg
        self.sigma, self.bias_sd, self.bias_rw = [c["sigma"]], [c["bias"]], \
            [0.0]
        self.scale_sd_ch = [c.get("scale_sd", 0.0)]

    def truth(self, env, t, s):
        return np.array([s[6]])


class WaveProbe(Sensor):
    kind = "wave_probe"

    def setup(self, env):
        c = self.cfg
        pts = c.get("points")
        if not pts:
            pts = [[env.x_stem, 0.0], [0.0, 0.0]]
        self.pts = np.asarray(pts, float).reshape(-1, 2)
        self.channels = tuple(f"eta{i}" for i in range(len(self.pts)))
        self.sigma, self.bias_sd, self.bias_rw = c["sigma"], 0.0, 0.0

    def truth(self, env, t, s):
        c, sn = math.cos(s[5]), math.sin(s[5])
        X = s[0] + self.pts[:, 0] * c - self.pts[:, 1] * sn
        Y = s[1] + self.pts[:, 0] * sn + self.pts[:, 1] * c
        return np.asarray(env.sea.eta(X, Y, t), float).reshape(-1)


class Preview(Sensor):
    kind = "preview"

    def setup(self, env):
        c = self.cfg
        d = c.get("distances")
        if not d:
            d = [env.L * k for k in (1, 2, 3, 4, 5)]
        self.dist = np.asarray(d, float)
        self.channels = preview_names(self.dist)
        self.sigma = c["sigma0"] + c["sigma_per_m"] * self.dist
        self.bias_sd, self.bias_rw = 0.0, 0.0

    def truth(self, env, t, s):
        c, sn = math.cos(s[5]), math.sin(s[5])
        X = s[0] + (env.x_stem + self.dist) * c
        Y = s[1] + (env.x_stem + self.dist) * sn
        return np.asarray(env.sea.eta(X, Y, t), float).reshape(-1)


def preview_names(dist):
    """eta_<d>m, d to the centimetre; distances that would share a name are
    refused (a reading() dict would lose one)."""
    names = tuple(f"eta_{round(float(x), 2):g}m" for x in dist)
    if len(set(names)) != len(names):
        raise ValueError(f"preview distances {list(map(float, dist))} give "
                         f"duplicate channels {list(names)}: make them "
                         f"distinct to the centimetre")
    return names


KINDS = dict(imu=IMU, gnss=GNSS, heading=Heading, speed_log=SpeedLog,
             wave_probe=WaveProbe, preview=Preview)
NOISE_KEYS = ("sigma_f", "sigma_w", "sigma_att", "sigma_yaw", "bias_f",
              "bias_w", "rw_f", "rw_w", "sigma_h", "sigma_v", "sigma_vel",
              "bias_h", "rw_h", "sigma", "bias", "scale_sd", "sigma0",
              "sigma_per_m", "dropout")


def _num(v, what, lo=None, hi=None, lo_open=False):
    if isinstance(v, bool) or v is None:
        raise ValueError(f"{what}: a number, not {v!r}")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: a number, not {v!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"{what}: a finite number, not {v!r}")
    if lo is not None and (x <= lo if lo_open else x < lo):
        raise ValueError(f"{what}: {'>' if lo_open else '>='} {lo:g}, not "
                         f"{v!r}")
    if hi is not None and x > hi:
        raise ValueError(f"{what}: <= {hi:g}, not {v!r}")
    return x


def validate_config(config):
    """[(name, kind, index, cfg)] of a sensors config, every value checked
    (ValueError naming the sensor and the key)."""
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("sensors: an object {name: {...} | null}")
    config = dict(config)
    noise_on = config.pop("noise", True)
    if not isinstance(noise_on, bool):
        raise ValueError("sensors.noise: true or false")
    out = []
    names = list(ORDER) + sorted(k for k in config if k not in ORDER)
    for i, name in enumerate(names):
        if name in config and config[name] is None:
            continue
        user = config.get(name) or {}
        if not isinstance(user, dict):
            raise ValueError(f"sensor {name!r}: an object or null")
        user = dict(user)
        kind = user.pop("kind", name)
        if kind not in KINDS:
            raise ValueError(f"sensor {name!r}: unknown kind {kind!r} "
                             f"(one of {sorted(KINDS)})")
        cfg = dict(DEFAULTS[kind])
        bad = set(user) - set(cfg)
        if bad:
            raise ValueError(f"sensor {name!r}: unknown keys {sorted(bad)}")
        cfg.update(user)
        w = f"sensor {name!r}"
        if cfg.get("rate_hz") is not None:
            _num(cfg["rate_hz"], f"{w} rate_hz (or null = every plant "
                                 f"step)", lo=0.0, lo_open=True)
        _num(cfg.get("latency_s", 0.0), f"{w} latency_s", lo=0.0)
        _num(cfg.get("dropout", 0.0), f"{w} dropout", lo=0.0, hi=1.0)
        for k in NOISE_KEYS:
            if k in cfg and k != "dropout":
                _num(cfg[k], f"{w} {k}", lo=0.0)
        if kind == "gnss":
            lv = cfg.get("lever") or [0.0, 0.0, 0.0]
            if not isinstance(lv, (list, tuple)) or len(lv) != 3:
                raise ValueError(f"{w} lever: [x, y, z] metres (body)")
            for j, v in enumerate(lv):
                _num(v, f"{w} lever[{j}]")
        if kind == "wave_probe" and cfg.get("points") is not None:
            pts = cfg["points"]
            if not isinstance(pts, (list, tuple)) or not pts:
                raise ValueError(f"{w} points: [[xb, yb], ...] or null")
            for j, pt in enumerate(pts):
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    raise ValueError(f"{w} points[{j}]: [xb, yb] metres")
                for v in pt:
                    _num(v, f"{w} points[{j}]")
        if kind == "preview" and cfg.get("distances") is not None:
            d = cfg["distances"]
            if not isinstance(d, (list, tuple)) or not d:
                raise ValueError(f"{w} distances: [m, ...] or null")
            preview_names([_num(x, f"{w} distances[{j}]", lo=0.0)
                           for j, x in enumerate(d)])
        if not noise_on:
            for k in NOISE_KEYS:
                if k in cfg:
                    cfg[k] = 0.0
        out.append((name, kind, i, cfg))
    return out


def build(config, env):
    """The scenario's sensors (dict name -> Sensor), in ORDER."""
    return {name: KINDS[kind](name, cfg, env, i)
            for name, kind, i, cfg in validate_config(config)}
