"""
Domain randomisation for training on the frozen plant
(learn/meta/SIM_FINAL_2026-10-09.md section 5: what to randomise, the
ranges, and by which route -- a field of a copied hull, an input
perturbation, or an output perturbation). Nothing under sim/ or hydro/ is
changed: hull fields go through a renamed copy of the hull on its physics
version (sim.physics.variant, so its caches are its own), the transom
threshold through hydro/vessel_spec's per-hull override, the sea through
SeaState's parameters and the component amplitudes, the thrust gain and the
output scales in the environment.

    rz = Randomization(seed=7)                       every parameter on
    rz = Randomization(seed=7).without("rho", "cda") some off
    rz = Randomization.of({"seed": 7, "mass": [1380, 1450], "rho": False})
    draw = rz.draw(episode_seed)                     {name: value} of the
                                                     parameters that are on
    env.reset(seed=s, options={"randomize": True})   draws and applies one
    env.reset(seed=s, options={"randomize": draw})   applies a given draw

THE DRAW. One uniform sample per parameter per episode from
np.random.default_rng([seed, episode_seed, stream]), in the fixed order
PARAMS; a parameter that is off is still drawn and dropped, so switching
one off does not shift the others. `stream` is 'train' or 'eval': the two
give different draws for the same episode seed, and training seeds never
return an evaluation seed (learn/repro/task.EVAL_SEEDS; app/env.py).

PARAMETERS (route; basis). Hull fields rebuild the plant; the others cost
nothing. A rebuild for a hull the process has not seen costs about 9 s on
this machine (measured 2026-10-10: the steady attitude at the design
speed 3.9 s, the righting-arm table 2.0 s, the steady attitude at the
initial speed 3.2 s -- all inside sim/, keyed by the hull's manifest or
by id(hull)); the same hull OBJECT again costs milliseconds, which is what
hull_every and the plant cache below make use of.
  mass            kg                           hull field; data (two test loads)
  lcg             x nominal LCG (scarab195: 0.34-0.42 LOA)   hull field; data
  vcg             m added to the nominal VCG   hull field; judgement
  cda             m^2, air drag C_D A          hull field; data (top-speed MC)
  rho             kg/m^3, water density        hull field; data (fresh, sea)
  c_dc            x nominal crossflow drag     hull field; judgement (E3 +-30 %)
  transom_ft_crit transom ventilation threshold, hydro/vessel_spec per-hull
                  override (as versions only; v0 has no transom state)   data
  roll_zeta       roll damping ratio           hull field; judgement (E6)
  cd_lat          lateral crossflow drag       hull field; literature (E5)
  jet_tau         s, jet thrust time constant  hull field jet.tau; judgement
  jet_delay       s, jet command delay         hull field jet.delay; judgement
  hs              x significant wave height    sea input; judgement
  tp              x peak period                sea input; data (Fridsma 1971)
  direction_deg   deg added to the wave direction   sea input; judgement
  gamma           JONSWAP peakedness           sea input; judgement
  short_wave      x amplitude of components with wavelength < 1.5 L
                                               sea input; data (short-wave heave)
  thrust_gain     x commanded thrust before the jet (the jet still saturates
                  at its own limits)           input; data (calm resistance)
  acc_scale       x the filtered CG and bow accelerations in the observation
                  and plant['acc_obs'] (not the physics, not the reward)
                                               output; data (irregular waves)
  motion_scale    x heave, pitch and their rates in every observation (the
                  sensors' input and 'truth'; not info['state'])
                                               output; data (regular waves)
Not randomised (SIM_FINAL): k_a, a_b (not advised), the static added mass,
the radiation damping and the post-chine-wetting section force (no knob;
acc_scale stands in for the last).

HULL_EVERY (2026-10-10). Randomization.hull_every = N (default 1): the hull
fields are redrawn, and the plant rebuilt, only on every N-th randomised
episode of an env (the i-th randomised reset, i from 0, when i is a
multiple of N); the episodes between run the hull draw in force and their
own sea, gain and output draws. The stream does not shift: every episode
still draws all PARAMS in order, the hull numbers of a skipped episode are
drawn and dropped, exactly as a parameter that is off. The draw an
episode runs (info / header 'randomization') carries the hull values in
force, so a record replays from its header alone; header and info
'randomization_hull_from' say whose hull draw it is ({'seed': the episode
seed that drew it, 'episode': its index among the env's randomised
episodes, 'this_episode', 'hull_every'}). Resets with randomize=False or
with a given draw dict neither count nor change the hull in force; a new
config (reset options 'randomization') starts the count again.

PLANT CACHE (2026-10-10). Randomization.plant_cache = M (default 8, 0 =
off): UsvEnv keeps the last M plants built for randomised hulls in an LRU
(PlantCache) keyed by the hull-draw tuple (hull_key: the hull fields and
their exact values). An entry holds the hull copy and a PRISTINE
prototype plant that is never stepped; every episode -- a miss as well as
a hit -- gets a fresh plant made from the prototype in the episode's sea
(plant.with_sea: a new object built for the same hull object, so the
physics' own steady-attitude and righting-arm caches hit and the build
takes milliseconds), so no episode can leak state into another or into
the cache (checked bit for bit in studies/app_randomization_test.py,
group hull_every). The hull in force under hull_every keeps its entry
even when the cache is off. Header 'plant_cache' = {'source': 'nominal' |
'built' | 'cache', 'n', 'max'}.
"""
import copy
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace

import numpy as np

STREAMS = {"train": 1, "eval": 2}
SHORT_WAVE_RATIO = 1.5


@dataclass(frozen=True)
class Param:
    lo: float
    hi: float
    on: bool = True

    def sample(self, rng):
        return float(rng.uniform(self.lo, self.hi))


HULL = ("mass", "lcg", "vcg", "cda", "rho", "c_dc", "transom_ft_crit",
        "roll_zeta", "cd_lat", "jet_tau", "jet_delay")
SEA = ("hs", "tp", "direction_deg", "gamma", "short_wave")
INPUT = ("thrust_gain",)
OUTPUT = ("acc_scale", "motion_scale")
PARAMS = HULL + SEA + INPUT + OUTPUT
# how each parameter is applied (for the record and the documentation)
ROUTE = {**{k: "hull field" for k in HULL}, **{k: "sea input" for k in SEA},
         **{k: "input" for k in INPUT}, **{k: "output" for k in OUTPUT}}
ROUTE["transom_ft_crit"] = "hull override (hydro/vessel_spec.TRANSOM_CLOSURE)"
UNITS = dict(mass="kg", lcg="x nominal", vcg="m added", cda="m^2",
             rho="kg/m^3", c_dc="x nominal", transom_ft_crit="-",
             roll_zeta="-", cd_lat="-", jet_tau="s", jet_delay="s",
             hs="x", tp="x", direction_deg="deg added", gamma="-",
             short_wave="x amplitude", thrust_gain="x", acc_scale="x",
             motion_scale="x")


@dataclass
class Randomization:
    seed: int = 0
    stream: str = "train"
    # hull fields redrawn (plant rebuilt) every N randomised episodes
    hull_every: int = 1
    # built plants kept for repeated hull draws (LRU entries; 0 = off)
    plant_cache: int = 8
    # hull fields (the plant is rebuilt for the episode)
    mass: Param = Param(1340.0, 1490.0)
    lcg: Param = Param(0.34 / 0.38, 0.42 / 0.38)
    vcg: Param = Param(-0.1, 0.1)
    cda: Param = Param(1.2, 2.2)
    rho: Param = Param(1000.0, 1025.0)
    c_dc: Param = Param(0.7, 1.3)
    transom_ft_crit: Param = Param(2.0, 4.5)
    roll_zeta: Param = Param(0.03, 0.20)
    cd_lat: Param = Param(0.8, 1.2)
    jet_tau: Param = Param(0.4, 0.8)
    jet_delay: Param = Param(0.05, 0.15)
    # sea (input perturbations)
    hs: Param = Param(0.9, 1.1)
    tp: Param = Param(0.8, 1.25)
    direction_deg: Param = Param(-15.0, 15.0)
    gamma: Param = Param(1.0, 3.3)
    short_wave: Param = Param(0.6, 1.4)
    # actuator (input perturbation)
    thrust_gain: Param = Param(0.85, 1.15)
    # outputs (observation / record outputs, not the physics)
    acc_scale: Param = Param(0.7, 1.3)
    motion_scale: Param = Param(0.8, 1.2)

    # ------------------------------------------------------------ forms
    @classmethod
    def of(cls, x):
        """A Randomization from None (defaults), one, or a dict: {'seed',
        'stream', 'hull_every', 'plant_cache', NAME: [lo, hi] | {'lo', 'hi',
        'on'} | true | false}."""
        if x is None:
            return cls()
        if isinstance(x, cls):
            return x.validate()
        if not isinstance(x, dict):
            raise TypeError(f"randomization: a Randomization or a dict, not "
                            f"{type(x).__name__}")
        d = dict(x)
        kw = dict(seed=d.pop("seed", 0), stream=d.pop("stream", "train"),
                  hull_every=d.pop("hull_every", 1),
                  plant_cache=d.pop("plant_cache", 8))
        for name in PARAMS:
            if name not in d:
                continue
            v = d.pop(name)
            dflt = getattr(cls, name)
            if isinstance(v, bool):
                kw[name] = replace(dflt, on=v)
            elif isinstance(v, (list, tuple)) and len(v) == 2:
                kw[name] = Param(float(v[0]), float(v[1]), True)
            elif isinstance(v, dict):
                bad = set(v) - {"lo", "hi", "on"}
                if bad:
                    raise KeyError(f"randomization.{name}: unknown keys "
                                   f"{sorted(bad)}")
                kw[name] = Param(float(v.get("lo", dflt.lo)),
                                 float(v.get("hi", dflt.hi)),
                                 bool(v.get("on", True)))
            else:
                raise ValueError(f"randomization.{name}: [lo, hi], "
                                 f"{{lo, hi, on}}, true or false, not {v!r}")
        if d:
            raise KeyError(f"randomization: unknown parameters {sorted(d)}; "
                           f"known: {list(PARAMS)}")
        return cls(**kw).validate()

    def validate(self):
        if isinstance(self.seed, bool) or int(self.seed) != self.seed \
                or self.seed < 0:
            raise ValueError(f"randomization.seed: an integer >= 0, not "
                             f"{self.seed!r}")
        if self.stream not in STREAMS:
            raise ValueError(f"randomization.stream: one of "
                             f"{sorted(STREAMS)}, not {self.stream!r}")
        for name, lo in (("hull_every", 1), ("plant_cache", 0)):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) \
                    or v < lo:
                raise ValueError(f"randomization.{name}: an integer >= {lo},"
                                 f" not {v!r}")
        for name in PARAMS:
            p = getattr(self, name)
            if not isinstance(p, Param):
                raise TypeError(f"randomization.{name}: a Param")
            if not (math.isfinite(p.lo) and math.isfinite(p.hi)
                    and p.lo <= p.hi):
                raise ValueError(f"randomization.{name}: lo <= hi, finite, "
                                 f"not {p}")
        for name in ("mass", "rho", "cda", "jet_tau", "transom_ft_crit",
                     "thrust_gain", "acc_scale", "motion_scale", "hs", "tp",
                     "gamma", "short_wave", "lcg", "c_dc", "cd_lat"):
            if getattr(self, name).lo <= 0.0:
                raise ValueError(f"randomization.{name}: lo > 0")
        if self.jet_delay.lo < 0.0 or self.roll_zeta.lo < 0.0:
            raise ValueError("randomization: jet_delay and roll_zeta >= 0")
        return self

    def to_dict(self):
        d = dict(seed=int(self.seed), stream=self.stream,
                 hull_every=int(self.hull_every),
                 plant_cache=int(self.plant_cache))
        for name in PARAMS:
            d[name] = asdict(getattr(self, name))
        return d

    # ------------------------------------------------------------ edits
    def with_(self, **changes):
        """A copy with parameters changed: NAME=False / True (off / on),
        NAME=(lo, hi), NAME=Param(...); seed=, stream=, hull_every=,
        plant_cache=."""
        kw = {}
        for k, v in changes.items():
            if k in ("seed", "stream", "hull_every", "plant_cache"):
                kw[k] = v
            elif k in PARAMS:
                p = getattr(self, k)
                if isinstance(v, bool):
                    kw[k] = replace(p, on=v)
                elif isinstance(v, Param):
                    kw[k] = v
                else:
                    lo, hi = v
                    kw[k] = Param(float(lo), float(hi), p.on)
            else:
                raise KeyError(f"randomization: unknown parameter {k!r}")
        return replace(self, **kw).validate()

    def without(self, *names):
        return self.with_(**{n: False for n in names})

    def only(self, *names):
        bad = set(names) - set(PARAMS)
        if bad:
            raise KeyError(f"randomization: unknown parameters {sorted(bad)}")
        return self.with_(**{n: (n in names) for n in PARAMS})

    def enabled(self):
        return [n for n in PARAMS if getattr(self, n).on]

    def rebuilds_plant(self):
        """True when a hull field is on (the plant is built per episode)."""
        return any(getattr(self, n).on for n in HULL)

    # ------------------------------------------------------------ draw
    def rng(self, episode_seed):
        return np.random.default_rng([int(self.seed), int(episode_seed),
                                      STREAMS[self.stream]])

    def draw(self, episode_seed):
        """{name: value} for this episode: every parameter sampled in the
        order PARAMS (so flags do not shift the stream), the ones that are
        off dropped."""
        self.validate()
        rng = self.rng(episode_seed)
        out = {}
        for name in PARAMS:
            p = getattr(self, name)
            u = p.sample(rng)
            if p.on:
                out[name] = u
        return out


def check_draw(draw):
    """A draw given to reset(): known names, finite numbers."""
    if not isinstance(draw, dict):
        raise TypeError("randomize: True, False or a draw dict")
    bad = set(draw) - set(PARAMS)
    if bad:
        raise KeyError(f"randomize: unknown parameters {sorted(bad)}")
    out = {}
    for k, v in draw.items():
        if isinstance(v, bool) or not isinstance(v, (int, float, np.number)) \
                or not math.isfinite(float(v)):
            raise ValueError(f"randomize: {k} = {v!r} is not a finite number")
        out[k] = float(v)
    return out


# ------------------------------------------------------------- apply: hull
def hull_fields(draw):
    return {k: v for k, v in draw.items() if k in HULL}


def hull_key(draw):
    """The hull-draw tuple of a draw: ((name, value), ...) of its hull
    fields in PARAMS order, values the exact floats -- the plant cache's
    key (two draws with the same hull values build the same hull copy)."""
    return tuple((k, float(draw[k])) for k in HULL if k in draw)


class PlantCache:
    """A small LRU of entries keyed by hull_key (module docstring, PLANT
    CACHE): get(key) -> entry or None (and marks it most recent), put(key,
    entry) (evicts the least recent beyond maxsize; drops at once when
    maxsize is 0), resize(maxsize), len(), keys(); hits / misses counted.
    The entries UsvEnv stores are {'hull', 'plant' (the pristine
    prototype), 'applied', 'unapplied'}."""

    def __init__(self, maxsize=8):
        self._d = OrderedDict()
        self.maxsize = 0
        self.hits = self.misses = 0
        self.resize(maxsize)

    def resize(self, maxsize):
        m = int(maxsize)
        if isinstance(maxsize, bool) or m != maxsize or m < 0:
            raise ValueError(f"plant cache size: an integer >= 0, not "
                             f"{maxsize!r}")
        self.maxsize = m
        while len(self._d) > m:
            self._d.popitem(last=False)

    def get(self, key):
        e = self._d.get(key)
        if e is None:
            self.misses += 1
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return e

    def put(self, key, entry):
        if self.maxsize == 0:
            return
        self._d[key] = entry
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def keys(self):
        return list(self._d)

    def __len__(self):
        return len(self._d)

    def __contains__(self, key):
        return key in self._d


def apply_hull(base, draw):
    """(hull copy, applied, unapplied): a deep copy of the v0 hull `base`
    with the draw's hull fields set. `applied` maps field -> value; a
    parameter the hull cannot carry (transom_ft_crit on v0) is listed in
    `unapplied`."""
    h = copy.deepcopy(base)
    h.T = None                  # the static draft: the new plant sets its own
    applied, unapplied = {}, []
    d = hull_fields(draw)
    if "mass" in d:
        h.mass = float(d["mass"])
        applied["mass"] = h.mass
    if "lcg" in d:
        h.lcg = float(base.lcg * d["lcg"])
        applied["lcg"] = h.lcg
    if "vcg" in d:
        h.vcg = float(base.vcg + d["vcg"])
        applied["vcg"] = h.vcg
    if "cda" in d:
        h.cda = float(d["cda"])
        applied["cda"] = h.cda
    if "rho" in d:
        h.rho = float(d["rho"])
        applied["rho"] = h.rho
    if "c_dc" in d:
        h.c_dc = float(base.c_dc * d["c_dc"])
        applied["c_dc"] = h.c_dc
    if "roll_zeta" in d:
        h.roll_zeta = float(d["roll_zeta"])
        applied["roll_zeta"] = h.roll_zeta
    if "cd_lat" in d:
        h.cd_lat = float(d["cd_lat"])
        applied["cd_lat"] = h.cd_lat
    if "jet_tau" in d or "jet_delay" in d:
        h.jet = dict(h.jet)
        if "jet_tau" in d:
            h.jet["tau"] = float(d["jet_tau"])
            applied["jet.tau"] = h.jet["tau"]
        if "jet_delay" in d:
            h.jet["delay"] = float(d["jet_delay"])
            applied["jet.delay"] = h.jet["delay"]
    if "transom_ft_crit" in d:
        applied["transom_closure.F_T_crit"] = float(d["transom_ft_crit"])
        tc = dict(getattr(h, "transom_closure", None) or {})
        tc["F_T_crit"] = float(d["transom_ft_crit"])
        h.transom_closure = tc
    if not (h.mass > 0 and 0.0 < h.lcg < h.L and 0.0 < h.vcg < h.depth):
        raise ValueError(f"randomised hull out of range: mass {h.mass}, lcg "
                         f"{h.lcg} (L {h.L}), vcg {h.vcg} (depth {h.depth})")
    return h, applied, unapplied


def hull_for(base, version, draw, name=None):
    """The episode's hull: None when the draw has no hull field; else a
    renamed copy of the v0 hull `base` on `version` (physics.variant for an
    as version, so its caches are keyed by its own manifest; a renamed v0
    copy for v0). Returns (hull, applied, unapplied)."""
    if not hull_fields(draw):
        return None, {}, []
    from sim import physics
    h, applied, unapplied = apply_hull(base, draw)
    nm = f"{base.name}~rz" if name is None else str(name)
    if version == physics.DEFAULT:
        if "transom_closure" in vars(h):
            # v0 has no transom state: the threshold is not a knob there
            del h.transom_closure
            applied.pop("transom_closure.F_T_crit", None)
            unapplied.append("transom_ft_crit (v0 has no transom state)")
        h.name = nm
        return h, applied, unapplied
    return physics.variant(h, version, name=nm), applied, unapplied


# -------------------------------------------------------------- apply: sea
def sea_params(draw, hs, tp, kw):
    """(hs, tp, SeaState keywords) with the draw's sea perturbations: Hs
    and Tp factors, the direction shift (theta0 in radians), gamma."""
    kw = dict(kw)
    if "hs" in draw:
        hs = float(hs * draw["hs"])
    if "tp" in draw:
        tp = float(tp * draw["tp"])
    if "direction_deg" in draw:
        kw["theta0"] = float(kw.get("theta0", math.pi)
                             + math.radians(draw["direction_deg"]))
    if "gamma" in draw:
        kw["gamma"] = float(draw["gamma"])
    return hs, tp, kw


def scale_short_waves(sea, L, factor, ratio=SHORT_WAVE_RATIO):
    """Multiply the amplitude of every component whose wavelength is below
    `ratio` x L by `factor` (in place, before the plant is built). Returns
    the number of components scaled."""
    lam = 2.0 * math.pi / np.asarray(sea.k, float)
    mask = lam < ratio * float(L)
    if mask.any():
        a = np.array(sea.a, float)
        a[mask] = a[mask] * float(factor)
        sea.a = a
    return int(mask.sum())


# ----------------------------------------------------------- apply: scales
def scales(draw):
    """(thrust_gain, acc_scale, motion_scale) of a draw (1.0 when off)."""
    d = draw or {}
    return (float(d.get("thrust_gain", 1.0)), float(d.get("acc_scale", 1.0)),
            float(d.get("motion_scale", 1.0)))


def describe(draw):
    if not draw:
        return "randomisation off"
    return ", ".join(f"{k} {v:.4g}" for k, v in draw.items())


def table():
    """[(name, lo, hi, unit, route)] of the default ranges (documentation,
    `python -m app randomization`)."""
    rz = Randomization()
    return [(n, getattr(rz, n).lo, getattr(rz, n).hi, UNITS[n], ROUTE[n])
            for n in PARAMS]
