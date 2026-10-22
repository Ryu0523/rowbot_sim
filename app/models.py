"""
What the model-based parts of the product layer need from a hull, built
once and cached: the identified reduced model (control/reduced.py) and the
steady turning table.

REDUCED MODEL. Exactly what learn/repro/task.ctx() gets from
sim/lofi.template: ReducedModel.identify on the calm plant, plus
lofi.EXTRA. A file is accepted by lofi's rule (IDENT_VERSION and the
plant's three-second calm-water fingerprint match), but only a file made
by the same code: this layer's own studies/_cache/app/lofi_<cache tag>.json
(the tag carries the version's code hash, the hull and the switches) and,
for a v0 hull only -- v0's code is frozen, so a name-keyed file of it was
made by this code -- the project's studies/_cache/lofi_<name>.json and
studies/_cache/principles/hull_swap/lofi_<name>.json. Otherwise it
identifies (about a minute) and writes its own file; it never writes the
project's caches nor another tag's file (ACCEPTANCE N3).
WHY NOT ACROSS CODE STATES. A first version copied an earlier tag's file of
the same hull when the fingerprint matched. The fingerprint is a 3-second
calm-water run at one speed: it separates the presets (35.6408 on the as1
preset, 36.2912 on the two-switch preset, 36.5027 with none -- v0's), but
an edit that changes only regimes it never reaches (planing speed, waves,
turning) would have been copied across tags unseen. Every as1 code edit
now costs one identification per hull, on first use.

AUTOPILOT. sim/env.Episode._steer is the project's heading autopilot (gains
placed on the identified Nomoto model, conditional integral). It is called
here as it is, on a small object that carries the attributes it reads, so
the product layer's heading mode IS that code, not a copy.

TURN TABLE. studies/final_eval.phase_turn's r_max(u) (steady full-nozzle
turn, calm water, six throttle settings): read from
studies/_cache/final_eval/turn_gz.json for scarab195 (v0, its pinned
manifest: physics.cache_tag is None) on the GZ plant, else computed the same
way on the hull's own plant and cached in
studies/_cache/app/turn_<cache tag>_<plant>.json.

CACHE KEYS (ACCEPTANCE N3). Files are named by sim/physics.cache_tag(hull)
-- 'NAME-<key_sha1[:12]>', the physics sha1 of the hull's version code,
its inputs and its preset switches; the legacy name for scarab195 on v0
with its pinned manifest (tag None) -- as studies/final_eval.turn_path and
sim/vessel_profile.py name theirs: a new as1 code state writes a new file
and never rewrites the file of another. The process caches are keyed the
same way, so an inline hull that reuses a name, or an @as1 hull after the
physics track edits its code, never gets a stale model or table. (The
reduced-model files also carry lofi's own validity check, the plant's
three-second fingerprint.)
"""
import json
import math
import os

import numpy as np

from . import CACHE, ROOT

_RED = {}


def _atomic_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def tag_of(hull):
    """The cache-file key of `hull`: sim/physics.cache_tag, or the plain
    name for the legacy hull (tag None)."""
    from sim import physics
    tag = physics.cache_tag(hull)
    if tag is not None:
        return tag
    return "".join(c if c.isalnum() or c in "-_.@" else "_"
                   for c in hull.name)


def reduced_params(hull, verbose=False):
    """The identified reduced-model parameters of `hull` (with
    lofi.EXTRA), as task.ctx()['p'] holds them for scarab195."""
    name = hull.name
    key = (name, tag_of(hull))
    if key in _RED:
        return dict(_RED[key])
    from sim import config, lofi
    from control.reduced import ReducedModel
    plant, _ = config.calm_plant(hull, db=None)
    fp = lofi._fingerprint(plant)
    own = os.path.join(CACHE, f"lofi_{tag_of(hull)}.json")
    cands = [own,
             os.path.join(ROOT, "studies", "_cache", f"lofi_{name}.json"),
             os.path.join(ROOT, "studies", "_cache", "principles",
                          "hull_swap", f"lofi_{name}.json")]
    from sim import physics
    if physics.version(hull) != physics.DEFAULT:
        # not frozen: only the file of this code state (module docstring)
        cands = [own]
    p = None
    for path in cands:
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                c = json.load(f)
        except (OSError, ValueError):
            continue
        if (c.get("version") == lofi.IDENT_VERSION
                and abs(c.get("fingerprint", np.inf) - fp)
                <= 1e-9 * max(1.0, abs(fp))):
            p = c["params"]
            break
    if p is None:
        if verbose:
            print(f"  identifying the reduced model for {name} (cached in "
                  f"{os.path.relpath(own, ROOT)}) ...", flush=True)
        p = {k: float(v) for k, v in ReducedModel.identify(plant).p.items()}
        _atomic_json(dict(hull=name, cache_tag=tag_of(hull),
                          version=lofi.IDENT_VERSION,
                          fingerprint=fp, params=p), own)
    p = {**p, **lofi.EXTRA}
    _RED[key] = dict(p)
    return dict(p)


def reduced_model(hull, verbose=False):
    from control.reduced import ReducedModel
    return ReducedModel(reduced_params(hull, verbose))


class Autopilot:
    """sim/env.Episode._steer on the attributes it reads. dt_ctrl is the
    NOMINAL control step (Hull.scales()['dt_ctrl'], 0.25 s for scarab195)
    as Episode passes it, not sub x dt."""

    def __init__(self, plant, reduced, u_ref, dt_ctrl, heading_ref=0.0):
        self.plant, self.reduced = plant, reduced
        self.u_ref, self.dt_ctrl = float(u_ref), float(dt_ctrl)
        self.heading_ref = heading_ref
        self._psi_i = 0.0

    def _steer(self, s, thrust=None, wn=None, zeta=1.0, lim=None):
        from sim.env import Episode
        return Episode._steer(self, s, thrust, wn=wn, zeta=zeta, lim=lim)

    def __call__(self, s, thrust, heading_ref):
        self.heading_ref = heading_ref
        return self._steer(s, thrust)


THROTTLES = (0.3, 0.45, 0.6, 0.75, 0.9, 1.0)


def turn_table(hull, plant_kind="gz", u0=None):
    """r_max(u) as studies/final_eval.phase_turn computes it, for this hull
    and plant kind."""
    from sim import config, physics
    path = os.path.join(CACHE, f"turn_{tag_of(hull)}_{plant_kind}.json")
    proj = os.path.join(ROOT, "studies", "_cache", "final_eval",
                        "turn_gz.json")
    if hull.name == "scarab195" and plant_kind == "gz" and \
            physics.cache_tag(hull) is None and os.path.exists(proj):
        with open(proj) as f:
            return json.load(f)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    from sim.test_vessel import Monochromatic
    calm = Monochromatic(1.0, 0.0)
    sc = hull.scales()
    base = config.plant_for(None, calm, hull, dt=sc["dt"], fidelity="high")
    plant = physics.gz_twin(base) if plant_kind == "gz" else base
    dt = plant.dt
    t_max, rud = float(plant.prop.t_max), float(plant.rudder.max)
    u_start = 0.8 * float(sc["u_design"] if u0 is None else u0)
    U, R, Fs = [], [], []
    for f in THROTTLES:
        s = plant.initial_state(u_start)
        t, us, rs = 0.0, [], []
        for _ in range(int(round(45.0 / dt))):
            r_cmd = rud if t >= 20.0 else 0.0
            s = plant.step(s, t, f * t_max, r_cmd, dt)
            t += dt
            if t >= 37.0:
                us.append(s[6])
                rs.append(abs(s[11]))
        if not np.all(np.isfinite(s)):
            continue
        U.append(float(np.mean(us)))
        R.append(float(np.mean(rs)))
        Fs.append(f)
    tab = dict(u=U, r=R, throttle=Fs, plant=plant_kind, dt=dt,
               hull=hull.name,
               note="steady full-nozzle turn in calm water "
                    "(studies/final_eval.phase_turn's procedure)")
    _atomic_json(tab, path)
    return tab


def deg(x):
    return math.degrees(x)
