#!/usr/bin/env python3
"""
The one task every reproduction is run on, and the one way it is scored.

Vessel   scarab195 (hydro/hulls.py): the planing, waterjet stand-in for the
         VM 18 (DEFECTS H6-H7).
Source   the low-fidelity world (sim/lofi.py): the MPC's own reduced model
         run as a plant, optionally with its coefficients drawn from
         lofi.PRIOR (domain randomisation).
Target   the full planing plant (sim/planing_vessel.py) -- the "real boat"
         of the study. It has what the source lacks: speed lost to the
         waves, sway and yaw pushed by them, the pump losing its prime.
Mission  sea state 3 (Hs 1.0 m, Tp 5 s, as studies/exp_planing.py P4),
         head seas and 45 deg off the bow, holding 25 kn on a straight
         track (+x). Episodes: 60 s in training, 120 s in evaluation.
Scored   by the operator's cost, sampled at every plant step and averaged:

    c = max(0, (U - u_along) / U)                speed lost along the track
      + K_A max(0, |a_cg| / A_LIM - 1)^2         impacts above the limit
      + K_Y huber(y / (5 L))                     cross-track

  u_along is the speed made good along the track (a policy that turned
  round would otherwise be paid for its speed); a_cg the vertical
  acceleration at the CG, in g -- the standard impact measure for planing
  craft; y the cross-track error. The RL rewards are exactly -c, so the
  number an RL method optimises and the number every method is judged on
  are the same.

  THE PRICES ARE A CHOICE, NOT A MEASUREMENT (flagged to the user as the
  open question of stage 3). A_LIM = 1 g: the model under-predicts impact
  accelerations by 1.5-2.5x (DEFECTS H7), so this is roughly 1.5-2.5 g on
  the water. K_A = 5 makes impacts and speed loss cost about the same for
  the hand-tuned MPC in head seas in the target (0.26 against 0.15, both
  seeds); in the source the same controller pays almost nothing for
  impacts (0.01). The evaluation (studies/repro_baselines.py, DEFECTS I)
  then showed what the target wants: push harder 45 deg off the bow,
  where impacts stay small, and NOT in head seas, where pushing costs more
  in impacts than it saves in speed. The source says "push" everywhere --
  a sim-to-real gap in the DECISION, and a heading-dependent one, which is
  what the adaptation methods are supposed to close. Cross-track is priced
  per five boat lengths: the target drifts 2-13 m in 120 s under the
  autopilot, which is not what an operator cares about on a 1.5 km leg.

Evaluation: 8 wave seeds x 2 headings = 16 episodes of 120 s in the
target, the same 16 for every method (paired).
"""
import contextlib
import ctypes
import io
import multiprocessing as mp
import warnings

import numpy as np

HULL = "scarab195"
HS, TP = 1.0, 5.0
HEADINGS = (np.pi, np.pi - np.pi / 4)
T_TRAIN, T_EVAL = 60.0, 120.0
EP_KW = dict(n_samples=128, n_freq=24, n_dir=5)
A_LIM, K_A = 1.0, 5.0
Y_SCALE, K_Y = 5.0, 0.4
EVAL_SEEDS = tuple(range(8))
# The same two relative wave directions with the SEA fixed and the TRACK
# turned (Mission(track=...)): waves always from WAVES_FROM, legs heading
# 0 (head seas) and 45 deg (45 deg off the bow). Physically the task above;
# the difference is that the boat's own heading now says which leg it is
# on, so a correction indexed by heading needs no wave-direction sensor.
WAVES_FROM = np.pi
LEGS = (0.0, np.pi / 4)
G = 9.81
BAD_SCORE = 10.0            # a run that went non-finite


def huber(d):
    """Quadratic near the track, linear beyond two units (rl_env's form)."""
    a = abs(d)
    return d * d if a < 2.0 else 2.0 * (2.0 * a - 2.0)


def cost_parts(u_along, a_cg_g, y, L, u_ref):
    """(speed, impact, track) parts of the per-step operator cost."""
    return (max(0.0, (u_ref - u_along) / u_ref),
            K_A * max(0.0, abs(a_cg_g) / A_LIM - 1.0) ** 2,
            K_Y * huber(y / (Y_SCALE * L)))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def avail_gb():
    """Available physical memory, GB (Windows; no psutil in this venv)."""
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 30
    except Exception:
        return float("nan")


# ----------------------------------------------------------------- context
_CTX = {}


def ctx():
    """(hull, db, identified low-fi parameters, PRIOR names), once per
    process."""
    if not _CTX:
        warnings.filterwarnings("ignore")
        from sim import config, lofi
        with contextlib.redirect_stdout(io.StringIO()):
            h, db = config.load(HULL)
            _, p = lofi.template(db, h)
        _CTX.update(h=h, db=db, p=dict(p), names=lofi.prior_names(p))
    return _CTX


def to_unit(nominal, params, names):
    """lofi.from_unit inverted: coefficients -> z in [-1, 1] per name (the
    'privileged' description of a randomised vessel). A coefficient at a
    value outside its PRIOR range (the neutral 0 of tau_thrust / rud_rate)
    maps outside [-1, 1] and is clipped to +-1.5."""
    from sim.lofi import PRIOR
    rs = float(np.sqrt(nominal["L"] / 10.0))
    out = []
    for n in names:
        kind, lo, hi, _ = PRIOR[n]
        v = params.get(n, nominal[n])
        if kind == "x":
            # the RATIO: some coefficients are negative (k_nomoto_f)
            ratio = v / nominal[n] if nominal[n] != 0 else 1.0
            a = (np.log(max(ratio, 1e-12)) - np.log(lo)) / np.log(hi / lo)
        else:
            k = 1.0 / rs if n == "rud_rate" else rs
            a = (v / k - lo) / (hi - lo)
        out.append(float(np.clip(2.0 * a - 1.0, -1.5, 1.5)))
    return np.array(out)


def draw_vessel(rng, scale=1.0):
    """A source-world vessel drawn from lofi.PRIOR: (overrides, z)."""
    from sim import lofi
    c = ctx()
    pp = lofi.perturb(c["p"], rng, scale=scale)
    return pp, to_unit(c["p"], pp, c["names"])


class LateralOU:
    """A wind-and-wave-like push in sway and yaw for the SOURCE world, which
    has none of its own (sim/lofi.py). Jiang et al.'s randomisation: an
    Ornstein-Uhlenbeck force and moment in the body frame, bounded at 10%
    of what the actuator can do. Here 10% of the jet's largest sway
    acceleration and yaw acceleration (full thrust, nozzle hard over, on
    the identified model); correlation time 5 s, stationary spread half the
    bound. Their lambda and sigma are not given.

    Without it a direct-control policy trained in the source never meets a
    heading disturbance and never learns to correct one; the MPC stack has
    the classical autopilot for that job and does not use this."""

    def __init__(self, rng, frac=0.1, tau=5.0):
        p = ctx()["p"]
        lift = p["k_jet_side"] * p["t_max"] * np.sin(p["rud_max"])
        self.bound = frac * np.array([lift / p["m_sway"],
                                      abs(p["k_nomoto_f"] * lift)
                                      / p["tau_r"]])
        self.lam = 1.0 / tau
        self.sig = 0.5 * self.bound * np.sqrt(2.0 * self.lam)
        self.d = np.zeros(2)
        self.rng = rng

    def __call__(self, s, dt):
        self.d += (-self.lam * self.d * dt
                   + self.sig * np.sqrt(dt) * self.rng.normal(size=2))
        self.d = np.clip(self.d, -self.bound, self.bound)
        s = s.copy()
        s[7] += self.d[0] * dt
        s[11] += self.d[1] * dt
        return s


class RotPreview:
    """A PreviewProvider seen from a track frame turned by phi: the MPC
    plans with the track along +x, the sea stays where it is."""

    def __init__(self, pv, phi):
        self.pv, self.phi = pv, float(phi)
        self.sea, self.t_preview = pv.sea, pv.t_preview
        self.c, self.s = np.cos(self.phi), np.sin(self.phi)

    def at(self, x, y, t, t_ahead):
        return self.pv.at(self.c * x - self.s * y, self.s * x + self.c * y,
                          t, t_ahead)


# ----------------------------------------------------------------- mission
class Mission:
    """One episode, stepped one control step (0.25 s) at a time.

    Built on sim.env.Episode so the plant, the MPC and the heading
    autopilot are exactly the ones every other study uses; this class only
    owns the loop, so a policy can act between control steps. With the
    MPC's command and constant weights it is Episode.run.

    world        "low" (source) or "high" (target)
    head         index into HEADINGS
    plant_params source-world overrides (randomised vessel), or None
    model_over   overrides of the MPC's internal model
    ctrl_factory f(reduced, preview, episode) -> controller, replacing
                 the Episode's MPPIController (SG-RL's sensitivity MPC)
    disturb      f(state, dt) -> state after every plant step (LateralOU),
                 or None
    track        None: waves turned by HEADINGS[head], track +x (the task
                 as first defined). An angle phi: waves from WAVES_FROM,
                 the boat starts on and holds heading phi; costs, the
                 observation and the MPC all work in the track frame.
    """

    def __init__(self, world, seed, head, plant_params=None,
                 t_end=T_TRAIN, weights=None, floor=0.0, model_over=None,
                 ctrl_factory=None, disturb=None, track=None, residual=None,
                 sea=None):
        from control.reduced import ReducedModel
        from sim.env import Episode
        c = ctx()
        p = dict(c["p"])
        if model_over:
            p.update(model_over)
        self.red = ReducedModel(p)
        self.head = int(head)
        self.phi = 0.0 if track is None else float(track)
        theta0 = HEADINGS[self.head] if track is None else WAVES_FROM
        # sea: None = the task's sea. A dict overrides any of hs, tp,
        # theta0, n_freq, n_dir and passes the rest (jitter, spread_s, band,
        # gamma) to the SeaState -- learn/meta's jittered random seas.
        sea = dict(sea or {})
        hs, tp = sea.pop("hs", HS), sea.pop("tp", TP)
        theta0 = sea.pop("theta0", theta0)
        kw = dict(EP_KW)
        for n in ("n_freq", "n_dir"):
            if n in sea:
                kw[n] = sea.pop(n)
        with contextlib.redirect_stdout(io.StringIO()):
            self.ep = Episode(c["db"], self.red, hs=hs, tp=tp,
                              seed=int(seed), u_ref=p["u_design"],
                              weights=weights, hull=c["h"],
                              theta0=theta0, fidelity=world,
                              plant_params=plant_params, thrust_floor=floor,
                              use_rudder=False, autopilot=True,
                              heading_ref=self.phi, sea_kw=sea or None, **kw)
        if self.phi != 0.0:
            self.ep.pv = RotPreview(self.ep.pv, self.phi)
            self.ep.ctrl.pv = self.ep.pv
        if ctrl_factory is not None:
            self.ep.ctrl = ctrl_factory(self.red, self.ep.pv, self.ep)
        self.world = world
        self.disturb = disturb
        # residual(state, dt, plant) -> state after every plant step: an
        # unknown extra acceleration (learn/meta/residuals.Injector)
        self.residual = residual
        self.plant = self.ep.plant
        self.sea = self.ep.sea
        self.theta0 = theta0
        self._cp, self._sp = np.cos(self.phi), np.sin(self.phi)
        self.u_ref = float(p["u_design"])
        self.L = float(self.plant.L)
        self.t_max = float(self.plant.prop.t_max)
        self.rud_max = float(self.plant.rudder.max)
        self.dt, self.sub = self.ep.dt, self.ep.sub
        self.dt_ctrl = self.ep.dt_ctrl
        self.n_ctrl = int(round(t_end / self.dt_ctrl))
        self.s = self.plant.initial_state(0.8 * self.u_ref)
        self.s[5] = self.phi
        self.t, self.k = 0.0, 0
        self.finite = True
        self._parts = np.zeros(3)
        self._n = 0
        self._acg, self._abow, self._u, self._ua, self._y, self._psi = \
            [], [], [], [], [], []
        self.last_amax = 0.0            # max |a_cg| over the last step, g
        self.last_acg = 0.0

    # ---------------------------------------------------------- stepping
    def mpc_command(self):
        """The MPC's thrust and the autopilot's nozzle, as Episode.run."""
        sr = self.track_state()
        cmd = self.ep.ctrl(sr, self.t)
        thrust, rudder = self.ep.ctrl.to_actuator(cmd)
        return thrust, self.ep._steer(self.s, thrust)

    def track_state(self):
        """The reduced state in the track frame (x along the track)."""
        sr = self.red.from_plant_state(self.s)
        if self.phi != 0.0:
            x, y = sr[0], sr[1]
            sr[0] = self._cp * x + self._sp * y
            sr[1] = -self._sp * x + self._cp * y
            sr[7] = sr[7] - self.phi
        return sr

    def along_cross(self, s):
        """(speed made good along the track, cross-track error)."""
        d = s[5] - self.phi
        return (s[6] * np.cos(d) - s[7] * np.sin(d),
                -self._sp * s[0] + self._cp * s[1])

    def advance(self, thrust, rudder):
        """One control step. Returns the mean operator cost over it."""
        parts = np.zeros(3)
        amax = 0.0
        thr_app = 0.0
        for _ in range(self.sub):
            thr_pre = float(self.s[12])
            self.s = self.plant.step(self.s, self.t, thrust, rudder, self.dt)
            # the thrust that acted over this substep: the low-fidelity
            # plant says; the full plant applies its thrust state as it
            # stood at the start of the substep
            thr_app += getattr(self.plant, "last_thr_app", thr_pre)
            if self.disturb is not None:
                self.s = self.disturb(self.s, self.dt)
            if self.residual is not None:
                self.s = self.residual(self.s, self.dt, self.plant)
            self.t += self.dt
            s = self.s
            if not np.all(np.isfinite(s)):
                self.finite = False
                break
            a = self.plant.last_cg_acc / G
            ua, yt = self.along_cross(s)
            parts += cost_parts(ua, a, yt, self.L, self.u_ref)
            amax = max(amax, abs(a))
            self._acg.append(abs(a))
            self._abow.append(abs(self.plant.last_bow_acc) / G)
            self._u.append(s[6])
            self._ua.append(ua)
            self._y.append(yt)
            self._psi.append(wrap(s[5] - self.phi))
        self.k += 1
        self.last_amax, self.last_acg = amax, self.plant.last_cg_acc / G
        n = max(len(self._acg) - self._n, 1)
        self._n = len(self._acg)
        self._parts += parts
        self.last_parts = parts / n     # (speed, impact, track), this step
        self.last_thr_app = thr_app / self.sub
        return float(parts.sum() / n)

    def done(self):
        return self.k >= self.n_ctrl or not self.finite

    # ----------------------------------------------------------- metrics
    def metrics(self):
        n = max(len(self._acg), 1)
        if not self.finite or not self._acg:
            return dict(score=BAD_SCORE, finite=False, u_mean=float("nan"),
                        u_along=float("nan"), acc_cg_p99=float("nan"),
                        acc_bow_p99=float("nan"), cross_rms=float("nan"),
                        heading_rms=float("nan"), c_speed=float("nan"),
                        c_impact=float("nan"), c_track=float("nan"),
                        t_end=float(self.t))
        acg, y = np.asarray(self._acg), np.asarray(self._y)
        parts = self._parts / n
        return dict(score=float(parts.sum()), finite=True,
                    c_speed=float(parts[0]), c_impact=float(parts[1]),
                    c_track=float(parts[2]),
                    u_mean=float(np.mean(self._u)),
                    u_along=float(np.mean(self._ua)),
                    acc_cg_p99=float(np.percentile(acg, 99)),
                    acc_bow_p99=float(np.percentile(self._abow, 99)),
                    cross_rms=float(np.sqrt(np.mean(y ** 2))),
                    heading_rms=float(np.sqrt(np.mean(np.square(self._psi)))),
                    t_end=float(self.t))

    # ------------------------------------------------------ observations
    def features(self):
        """The vessel's own state, scaled to order one: what any policy
        here is allowed to see. Speed made good, sway and yaw rate,
        heading and cross-track error, heave and pitch about the running
        attitude, the CG acceleration (last sample and worst of the last
        step), the actuators' actual state, the water at stern / midships /
        bow, and where the waves come from relative to the bow."""
        s, p = self.s, self.red.p
        ua, yt = self.along_cross(s)
        rel = self.theta0 - s[5]
        off = np.array([p["x_stern"], 0.0, p["x_bow"]])
        cp, sp = np.cos(s[5]), np.sin(s[5])
        wave = self.sea.eta(s[0] + off * cp, s[1] + off * sp, self.t) \
            / (HS / 4.0)
        f = np.concatenate([[
            ua / self.u_ref - 1.0,
            s[7] / 1.0,
            s[11] * self.L / self.u_ref,
            wrap(s[5] - self.phi) / 0.2,
            yt / (Y_SCALE * self.L),
            (s[2] - p.get("z0", 0.0)) / 0.2,
            s[8] / 1.0,
            (s[4] - p.get("th0", 0.0)) / 0.05,
            s[10] / 0.3,
            self.last_acg,
            self.last_amax / A_LIM,
            s[12] / self.t_max,
            s[13] / self.rud_max,
            np.cos(rel), np.sin(rel)], wave])
        # a runaway state must never reach a network as inf / nan
        return np.clip(np.nan_to_num(f, nan=0.0), -20.0, 20.0).astype(
            np.float32)


N_FEATURES = 18


# ------------------------------------------------ fixed-parameter MPC jobs
def run_job(job):
    """dict(z, world, seed, head, plant=None, model=None) -> metrics, the
    job format of studies/sim2real_jet.py, so its CMA-ES and GP search run
    unchanged on this task (the reference rows of the comparison)."""
    from studies.sim2real_jet import theta
    w, floor = theta(job["z"])
    track = LEGS[job["head"]] if job.get("geometry") == "track" else None
    m = Mission(job["world"], job["seed"], job["head"],
                plant_params=job.get("plant"), t_end=job.get("t_end", T_EVAL),
                weights=w, floor=floor, model_over=job.get("model"),
                track=track)
    while not m.done():
        m.advance(*m.mpc_command())
    return m.metrics()


def eval_jobs(z, world="high", seeds=EVAL_SEEDS, model=None):
    return [dict(z=list(map(float, z)), world=world, seed=int(s), head=h,
                 model=model) for s in seeds for h in range(len(HEADINGS))]


class Runner:
    """A process pool for episode jobs; refuses to start work with less
    than 1.5 GB of RAM free (the 16 GB laptop froze twice)."""

    def __init__(self, workers, fn=run_job):
        self.workers, self.fn = workers, fn
        self.pool = mp.Pool(workers, initializer=ctx)
        self.n = {"low": 0, "high": 0}

    def __call__(self, jobs):
        if avail_gb() < 1.5:
            raise MemoryError(f"only {avail_gb():.1f} GB of RAM left")
        for j in jobs:
            self.n[j.get("world", "high")] += 1
        return self.pool.map(self.fn, jobs, chunksize=1)

    def close(self):
        self.pool.close()
        self.pool.join()
