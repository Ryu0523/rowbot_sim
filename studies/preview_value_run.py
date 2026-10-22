#!/usr/bin/env python3
"""
One run of the preview-value study (studies/preview_value.py): what is a
perfect preview of the oncoming waves worth to the frozen plant
scarab195@as1, on the budgets of the envelope study?

The set-up is the envelope study's (studies/envelope_run.py: the frozen
plant on its large-angle twin, P+I throttle, the project's heading
autopilot, 60 s settling dropped + 600 s of statistics) with ONE change:
the speed the P+I loop tracks is no longer a constant but a two-level
reference chosen every control step by a "speed reference" rule,

    v_ref = v_lo  while a dangerous wave is expected (or being met),
    v_ref = v_hi  otherwise,

and the rules differ only in WHAT THEY KNOW:

  const          v_hi always (the envelope study's constant-speed runs)
  preview_h      PERFECT PREVIEW: the elevation the bow will meet during the
                 next `horizon` seconds if speed and heading are held
                 (sampled from the same SeaState that forces the plant, at
                 the bow's future positions); the largest zero-crossing wave
                 height in that window above thr x Hs triggers v_lo until
                 that crest has passed (+ hold s)
  preview_group  PERFECT PREVIEW of a longer window; the window's RMS
                 elevation above thr x Hs/4 (a wave group) triggers v_lo for
                 `hold` s (renewed while the group lasts)
  ar_h           PAST ONLY: the elevation at the bow up to now, known
                 exactly (an upper bound for anything the hull's motion can
                 give), extrapolated `horizon` s ahead by a least-squares
                 autoregressive model refitted every step; then the
                 preview_h criterion on the forecast
  reactive       NO PREVIEW: the bow's emergence speed (the plant's
                 last_rel_bow falling faster than thr m/s) triggers v_lo for
                 `hold` s -- what a rule on the current motion can do

The same two registered wave seeds and the same sea as the envelope runs,
so the constant-speed rows of studies/_cache/envelope are the baseline.

Per run the outputs of the envelope study (filtered CG and bow-gauge
accelerations, peaks, speed, roll, yaw rate, freeboard, slams) plus: the
reference-speed log, the trigger log (time, flag, metric, lead), the time
fraction at v_lo, the number of slow-downs, the CG pose every POSE_DECIM
plant steps (for the bow-camera study) and the sea's component arrays
(a, w, th, phi; the surface can be rebuilt offline without the plant).

Read-only use of sim/, hydro/, app/ and studies/envelope_run.
"""
import math
import os
import time
import warnings
from collections import deque

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from studies import envelope_run as R           # noqa: E402

KN, G = R.KN, R.G
K_P, K_I = R.K_P, R.K_I
SETTLE, RUN = R.SETTLE, R.RUN
FB_FRAC, DECIM = R.FB_FRAC, R.DECIM
POSE_DECIM = 5                 # pose kept every 5 plant steps = 0.05 s
KINDS = ("const", "preview_h", "preview_group", "ar_h", "reactive",
         "learned_future", "learned_image", "learned_motion")
# learned_future / learned_image: a predictor of the largest filtered CG
# acceleration in the next 5 s, trained by studies/bow_camera_train on the
# exact future wave at the bow + the motion history (privileged), or on the
# stabilised bow-camera range image + the motion history; v_lo while the
# predicted value exceeds thr (g). `net` in the run argument names the
# saved net (studies/_cache/camera/net_<name>.pt).


class Hook(R.Hook):
    """The envelope hook's channels plus the full pose at every plant step
    (x, y, z, roll, pitch, yaw, u, r, thrust state, nozzle state)."""

    def __init__(self):
        super().__init__()
        self.pose = []

    def __call__(self, s, dt, plant):
        super().__call__(s, dt, plant)
        self.pose.append((float(s[0]), float(s[1]), float(s[2]),
                          float(s[3]), float(s[4]), float(s[5]),
                          float(s[6]), float(s[11]), float(s[12]),
                          float(s[13])))
        return None


class Encounter:
    """The linear surface of a SeaState, evaluated at the bow's positions.

    `future(s, t)`: eta at the bow's positions tau seconds ahead if the
    vessel holds its speed and heading (tau on the sampled horizon);
    `now(s, t)`: eta at the bow now. The component arrays are copied from
    the sea object AFTER the env's reset, so any short-wave scaling of the
    randomisation is included."""

    def __init__(self, sea, x_bow, horizon, step):
        self.a = np.asarray(sea.a, float)
        self.k = np.asarray(sea.k, float)
        self.w = np.asarray(sea.w, float)
        self.phi = np.asarray(sea.phi, float)
        th = np.asarray(sea.th, float)
        self.cth, self.sth = np.cos(th), np.sin(th)
        self.x_bow = float(x_bow)
        self.step = float(step)
        self.tau = np.arange(0.0, horizon + 1e-9, step)

    def eta(self, x, y, t):
        x = np.atleast_1d(np.asarray(x, float))[:, None]
        y = np.atleast_1d(np.asarray(y, float))[:, None]
        t = np.atleast_1d(np.asarray(t, float))[:, None]
        ph = self.k * (x * self.cth + y * self.sth) - self.w * t + self.phi
        return (self.a * np.cos(ph)).sum(-1)

    def future(self, s, t):
        u = max(float(s[6]), 0.0)
        d = self.x_bow + u * self.tau
        return self.eta(s[0] + d * math.cos(s[5]), s[1] + d * math.sin(s[5]),
                        t + self.tau)

    def now(self, s, t):
        return float(self.eta(s[0] + self.x_bow * math.cos(s[5]),
                              s[1] + self.x_bow * math.sin(s[5]), t)[0])


def next_wave_height(eta, step):
    """(H, tau_crest) of the largest zero-crossing wave in the sampled
    elevation `eta` (max minus min between consecutive up-crossings of
    zero; the partial waves at both ends count), tau_crest the time of
    its crest from the start of the window."""
    e = np.asarray(eta, float)
    up = np.flatnonzero((e[:-1] < 0.0) & (e[1:] >= 0.0)) + 1
    edges = np.concatenate([[0], up, [len(e)]])
    best, tau = 0.0, 0.0
    for i0, i1 in zip(edges[:-1], edges[1:]):
        if i1 - i0 < 2:
            continue
        seg = e[i0:i1]
        h = float(seg.max() - seg.min())
        if h > best:
            best, tau = h, step * (i0 + int(np.argmax(seg)))
    return best, tau


def ar_forecast(hist, order, n_ahead, ridge=1e-6):
    """Least-squares AR(order) fit of the series `hist` (oldest first) and
    its recursive forecast n_ahead steps; None when the history is short."""
    h = np.asarray(hist, float)
    n = len(h)
    if n < 2 * order + 8:
        return None
    X = np.column_stack([h[order - j - 1:n - j - 1] for j in range(order)])
    y = h[order:]
    A = X.T @ X + ridge * np.trace(X.T @ X) / order * np.eye(order)
    try:
        c = np.linalg.solve(A, X.T @ y)
    except np.linalg.LinAlgError:
        return None
    buf = list(h[-order:][::-1])             # newest first
    out = np.empty(n_ahead)
    for i in range(n_ahead):
        v = float(np.dot(c, buf[:order]))
        out[i] = v
        buf.insert(0, v)
    return out


class LearnedTrigger:
    """The impact predictor of studies/bow_camera_train in the loop: inputs
    built exactly as the training frames (studies/bow_camera.process):
    future wave at the bow at 0.25 .. 10 s in Hs/4; heave / (Hs/4), pitch
    / 0.05, (speed - 8) / 4 over the last 10 s; the range image as
    (log r - log 50) / 1.5. Output: the predicted largest filtered CG
    acceleration (g) in the next 5 s."""

    def __init__(self, net_name, kind, env, enc, hs):
        import torch
        from studies import bow_camera_train as BT
        from studies import bow_camera as BC
        d = torch.load(os.path.join(BC.OUT, f"net_{net_name}.pt"),
                       weights_only=False)
        self.net = BT.build(d["kind"])
        self.net.load_state_dict(d["state"])
        self.net.eval()
        self.ym, self.ys = d["ym"], d["ys"]
        self.use_image = kind == "learned_image"
        self.sc = hs / 4.0
        self.enc = enc
        self.fut_enc = Encounter(env.sea, env.x_station, 10.0, 0.25)
        self.motion = deque(maxlen=40)
        self.past = deque(maxlen=80)          # the bow elevation, 20 s
        if self.use_image:
            self.cam = BC.Camera()
            self.surf = BC.Surface(enc.a, enc.w, np.arctan2(enc.sth, enc.cth),
                                   enc.phi)
            p = env.plant
            self.x_st = float(env.x_station)
            self.Z_rest, self.Th_rest = float(p.Z_rest), float(p.Th_rest)
            self.z_cam = float(p.freeboard) - self.Z_rest + 1.0
            self.rot = BC.rot_zyx
        self.dummy_img = np.zeros((1, 1, 24, 32), np.float32)

    def __call__(self, s, t):
        import torch
        self.motion.append((float(s[2]) / self.sc, float(s[4]) / 0.05,
                            (float(s[6]) - 8.0) / 4.0))
        self.past.append(self.fut_enc.now(s, t) / self.sc)
        if len(self.past) < 80:
            return float("nan")
        fut = (self.fut_enc.future(s, t)[1:] / self.sc).astype(np.float32)
        mo = np.asarray(self.motion, np.float32).T.reshape(1, -1)
        pa = np.asarray(self.past, np.float32).reshape(1, -1)
        if self.use_image:
            R = self.rot(s[3], self.Th_rest + s[4], s[5])
            c = np.array([s[0], s[1], self.Z_rest + s[2]]) + R @ np.array(
                [self.x_st, 0.0, self.z_cam])
            lr, _, _, _ = self.cam.frame(self.surf, c, float(s[5]), t)
            img = ((lr - math.log(50.0)) / 1.5).astype(np.float32)[None, None]
        else:
            img = self.dummy_img
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a))   # noqa: E731
        with torch.no_grad():
            out = self.net(T(img), T(pa), T(mo), T(fut[None])).numpy()[0]
        out = out * self.ys + self.ym
        return float(np.expm1(out[41]))          # imp in the next 5 s, g


class SpeedRef:
    """The two-level speed reference (module docstring). `__call__(s, t,
    emergence)` returns v_ref in m/s; `emergence` is the bow's emergence
    speed in m/s (positive = rising out of the water), used by 'reactive'
    only. Logs (t, triggered, metric, lead, v_ref) per step."""

    def __init__(self, kind, v_hi, v_lo, thr, hs, enc, dt_step, horizon,
                 hold=1.0, ar_order=16, ar_hist_s=40.0, learned=None):
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind, self.v_hi, self.v_lo = kind, float(v_hi), float(v_lo)
        self.thr, self.hs, self.enc = float(thr), float(hs), enc
        self.step, self.horizon, self.hold = float(dt_step), float(horizon), \
            float(hold)
        self.n_ahead = int(round(horizon / dt_step))
        self.ar_order = int(ar_order)
        self.hist = deque(maxlen=int(round(ar_hist_s / dt_step)))
        self.learned = learned
        self.until = -1.0
        self.log = []

    def __call__(self, s, t, emergence):
        trig, metric, lead = False, float("nan"), float("nan")
        k = self.kind
        if k in ("learned_future", "learned_image", "learned_motion"):
            metric = self.learned(s, t)
            trig = np.isfinite(metric) and metric > self.thr
            lead = 2.5
        elif k == "preview_h":
            e = self.enc.future(s, t)
            H, lead = next_wave_height(e, self.step)
            metric = H / self.hs
            trig = metric > self.thr
        elif k == "preview_group":
            e = self.enc.future(s, t)
            metric = float(e.std()) / (self.hs / 4.0)
            trig = metric > self.thr
            lead = 0.0
        elif k == "ar_h":
            self.hist.append(self.enc.now(s, t))
            f = ar_forecast(self.hist, self.ar_order, self.n_ahead)
            if f is not None:
                H, lead = next_wave_height(f, self.step)
                metric = H / self.hs
                trig = metric > self.thr
        elif k == "reactive":
            metric = float(emergence)
            trig = metric > self.thr
            lead = 0.0
        if trig:
            self.until = max(self.until, t + lead + self.hold)
        v = self.v_lo if t < self.until else self.v_hi
        self.log.append((t, float(trig), metric, lead, v))
        return v


def make_env(arg):
    """UsvEnv for a run argument (studies/envelope_run.make_env with this
    module's hook)."""
    from app import frozen
    from app.env import UsvEnv
    usv = dict(sea_seed=int(arg.get("sea_seed", arg["seed"])))
    sc = frozen.scenario(hs=float(arg["hs"]), tp=float(arg["tp"]),
                         theta0=math.pi, heading=float(arg["heading_rad"]),
                         duration=float(arg.get("duration", SETTLE + RUN)),
                         seed=int(arg["seed"]),
                         speed=float(arg["v_hi_kn"]) * KN, **usv)
    hook = Hook()
    env = UsvEnv(sc, action_mode="heading", record=False,
                 plant_hooks=[hook], randomization=arg.get("randomization"))
    return env, hook


def run_one(arg):
    """One episode. `arg`: name, hs, tp, heading_rad, seed, sea_seed, kind,
    v_hi_kn, v_lo_kn, thr, horizon, hold, [i_clip, thr_lo, randomize,
    randomization, duration, settle, ar_order]."""
    warnings.filterwarnings("ignore")
    from app import accel
    t_wall0 = time.perf_counter()
    env, hook = make_env(arg)
    t_build0 = time.perf_counter()
    obs, info = env.reset(seed=int(arg["seed"]),
                          options={"randomize": bool(arg.get("randomize",
                                                             False))})
    t_build = time.perf_counter() - t_build0
    hs_eff = float(env.sea_params["hs"])
    horizon = float(arg.get("horizon", 8.0))
    enc = Encounter(env.sea, env.x_station, horizon, env.dt_step)
    learned = None
    if arg["kind"] in ("learned_future", "learned_image", "learned_motion"):
        learned = LearnedTrigger(arg["net"], arg["kind"], env, enc, hs_eff)
    pol = SpeedRef(arg["kind"], float(arg["v_hi_kn"]) * KN,
                   float(arg.get("v_lo_kn", arg["v_hi_kn"])) * KN,
                   float(arg.get("thr", 0.0)), hs_eff, enc, env.dt_step,
                   horizon, hold=float(arg.get("hold", 1.0)),
                   ar_order=int(arg.get("ar_order", 16)), learned=learned)
    heading = float(arg["heading_rad"])
    thr0 = float(env.s[12] / env.t_max)
    i_clip = float(arg.get("i_clip", 10.0))
    thr_lo = float(arg.get("thr_lo", 0.0))
    acc_err = 0.0
    thr_log, vref_log = [], []
    reason = None
    sub = env.sub
    t_step0 = time.perf_counter()
    while True:
        rows = hook.rows
        if len(rows) > sub:
            emergence = -(rows[-1][7] - rows[-1 - sub][7]) / env.dt_step
        else:
            emergence = 0.0
        v_ref = pol(env.s, env.t, emergence)
        err = v_ref - float(env.s[6])
        acc_err = min(i_clip, max(-i_clip, acc_err + err))
        thr = min(1.0, max(thr_lo, thr0 + K_P * err + K_I * acc_err))
        thr_log.append(thr)
        vref_log.append(v_ref)
        obs, r, term, trunc, info = env.step([thr, heading])
        if term or trunc:
            reason = info["reason"]
            break
    t_step = time.perf_counter() - t_step0
    rows = np.asarray(hook.rows, float)
    pose = np.asarray(hook.pose, float)
    n = len(rows)
    dt = env.dt
    t = dt * np.arange(1, n + 1)
    settle = float(arg.get("settle", SETTLE))
    a_cg, a_st = rows[:, 0], rows[:, 1]
    bow = accel.bow_gauge(a_cg, a_st, env.x_gauge, env.x_station)
    q = np.column_stack([a_cg, bow]) / G
    qf = accel.lowpass_offline(q, dt, env.filter_hz)
    keep = t > settle + 1e-9
    fb0 = float(getattr(env.plant, "freeboard", np.nan))
    fb = fb0 - rows[:, 7]
    out = dict(name=arg["name"], arg={k: v for k, v in arg.items()
                                     if k != "randomization"},
               dt=dt, dt_step=float(env.dt_step), n_plant=int(n),
               t_end=float(t[-1]) if n else 0.0,
               settle=settle, t_stat=float(keep.sum() * dt),
               reason=reason, capsized=bool(env.capsized),
               finite=bool(env.finite),
               wall_total=time.perf_counter() - t_wall0,
               wall_build=t_build, wall_step=t_step,
               ms_per_step=1e3 * t_step / max(n, 1),
               x_gauge=float(env.x_gauge), x_station=float(env.x_station),
               L=float(env.L), Z_rest=float(env.plant.Z_rest),
               Th_rest=float(env.plant.Th_rest),
               freeboard_static=fb0, t_max=float(env.t_max), thr0=thr0,
               hs_eff=hs_eff,
               draw=(None if env.draw is None else dict(env.draw)),
               hull_applied=dict(env.hull_applied),
               plant_source=env.plant_source,
               sea_params={k: (float(v) if isinstance(v, (int, float))
                               else v) for k, v in
                           (env.sea_params or {}).items()},
               sea_seed=int(env.sea_seed), seed=int(env.seed_))
    log = np.asarray(pol.log, float)
    t_ctrl = env.dt_step * np.arange(1, len(thr_log) + 1)
    kc = t_ctrl > settle + 1e-9
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
    vr = np.asarray(vref_log, float)
    below = fbk < FB_FRAC * fb0
    n_imm = int(np.sum(below[1:] & ~below[:-1]) + (1 if below[0] else 0))
    slam = rows[:, 8]
    k0 = int(np.searchsorted(t, settle))
    low = vr[kc] < pol.v_hi - 1e-9
    n_slow = int(np.sum(low[1:] & ~low[:-1]) + (1 if low[0] else 0))
    T = float(keep.sum() * dt)
    out.update(stats=dict(
        u_mean=float(u.mean()), u_sd=float(u.std()),
        u_min=float(u.min()), u_max=float(u.max()),
        u_mean_kn=float(u.mean() / KN),
        thr_mean=float(thr_a[kc].mean()),
        thr_sat_frac=float(np.mean(thr_a[kc] >= 1.0 - 1e-9)),
        thr_zero_frac=float(np.mean(thr_a[kc] <= thr_lo + 1e-9)),
        low_frac=float(low.mean()), n_slow=n_slow,
        slow_per_hour=float(n_slow * 3600.0 / T),
        trig_frac=float(np.nanmean(log[kc, 1])) if len(log) else 0.0,
        roll_max_deg=float(np.degrees(np.abs(phi).max())),
        roll_rms_deg=float(np.degrees(np.sqrt(np.mean(phi ** 2)))),
        yaw_rate_max_dps=float(np.degrees(np.abs(rr).max())),
        yaw_rate_rms_dps=float(np.degrees(np.sqrt(np.mean(rr ** 2)))),
        fb_min=float(np.nanmin(fbk)), fb_mean=float(np.nanmean(fbk)),
        n_immersion=n_imm,
        immersion_per_hour=float(n_imm * 3600.0 / T),
        slam_count=int(slam[-1] - slam[k0]) if np.isfinite(slam[-1])
        else None,
        slam_per_hour=(float((slam[-1] - slam[k0]) * 3600.0 / T)
                       if np.isfinite(slam[-1]) else None),
        heave_sd=float(rows[keep, 2].std()),
        pitch_sd_deg=float(np.degrees(rows[keep, 4].std())),
        cg=st["cg"], bow=st["bow"]))
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
        throttle=thr_a.astype(np.float32),
        v_ref=vr.astype(np.float32),
        t_ctrl=t_ctrl.astype(np.float32),
        trig_log=log.astype(np.float32),
        pose=pose[::POSE_DECIM].astype(np.float32),
        sea_a=enc.a, sea_w=enc.w, sea_th=np.arctan2(enc.sth, enc.cth),
        sea_phi=enc.phi))
    return out


if __name__ == "__main__":
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else "preview_h"
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    r = run_one(dict(name="smoke", hs=1.0, tp=5.0, heading_rad=0.0,
                     seed=0, sea_seed=0, kind=kind, v_hi_kn=20.0,
                     v_lo_kn=15.0, thr=1.3, horizon=8.0, hold=1.0,
                     duration=30.0 + T, settle=30.0))
    s = r["stats"]
    print(f"{kind}: steps {r['n_plant']}  {r['ms_per_step']:.2f} ms/step  "
          f"build {r['wall_build']:.1f} s  total {r['wall_total']:.1f} s  "
          f"reason {r['reason']}")
    if s:
        print(f"u mean {s['u_mean_kn']:.2f} kn  low frac {s['low_frac']:.2f}"
              f"  slow-downs {s['n_slow']}  trig frac {s['trig_frac']:.2f}"
              f"  thr mean {s['thr_mean']:.2f} zero {s['thr_zero_frac']:.2f}")
        print(f"cg a110 {s['cg']['a110']:.2f} g max {s['cg']['max']:.2f} n "
              f"{s['cg']['n']} | bow a110 {s['bow']['a110']:.2f} | "
              f"immersions {s['n_immersion']}")
        lg = r["series"]["trig_log"]
        m = lg[:, 2]
        print(f"metric: median {np.nanmedian(m):.2f} max {np.nanmax(m):.2f}"
              f"; pose rows {len(r['series']['pose'])}; sea comps "
              f"{len(r['series']['sea_a'])}")
