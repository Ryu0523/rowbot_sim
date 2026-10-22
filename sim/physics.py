#!/usr/bin/env python3
"""
Which physics a planing plant runs -- chosen by one thing only, the hull key.

WHY VERSIONS AND NOT EDITS
The planing plant as it stands ("v0": sim/planing_vessel.py,
sim/planing_vessel_gz.py, sim/waterjet.py, sim/actuators.py) is not credible
below planing speed (learn/meta/anyspeed_design_en.md, section 1), and the
cure is new physics. But v0 carries every result so far, bit for bit: the
meta2-meta8 caches accept their files by a hash of sim/planing_vessel.py
(studies/meta_step2.py SOURCES), the reduced model the MPC uses
(studies/_cache/lofi_scarab195.json) was identified on it, the final
evaluation and its turning tables ran on it. So v0 is frozen and stays the
default, and the new physics is a new version beside it, asked for by name:

    hydro.hulls.get("scarab195")        v0, built exactly as before
    hydro.hulls.get("scarab195@as1")    the same vessel on the any-speed
                                        plant, sim/planing_anyspeed.py

ONE SELECTOR
The version is part of the hull's NAME, and PlaningHull.plant() asks
plant_class() here which class to build; nothing else selects. Every cache
keyed by hull name then keeps the versions apart by itself: the reduced
model (lofi_scarab195@as1.json -- sim/lofi.py rewrites lofi_{name}.json when
its fingerprint does not match, so a v0 name must never reach an as1 plant),
the plants' steady-state, GZ and top-speed caches. A first draft also let an
environment variable select the physics; the validation review found that
the two selectors had no precedence, and that a pipeline accepting its
caches by code hash alone would have mixed an env-selected plant with v0
data. So the environment variable USV_HULL carries nothing but a hull KEY to
worker processes (use() sets it, hull_key() reads it), and choosing a class
for a hull that disagrees with it raises: a worker told 'scarab195@as1' that
builds the plain 'scarab195' stops instead of quietly running v0. A hull's
name and its `physics` attribute must agree (version()), so no v0 name ever
yields another plant.

as1 HAS TWO CLASSES, AS v0 HAS
v0 has two plants, PlaningVessel (linear roll; what PlaningHull.plant()
builds) and its large-angle twin PlaningVesselGZ (what the evaluation runs).
as1 mirrors them: PlaningVesselASLin is what PlaningHull.plant() builds for
an as1 hull, PlaningVesselAS is its GZ twin (gz_twin, gz_plant_for). With
one class only, everything built through hull.plant -- Mission's base plant,
sim/env.Episode, sim/rl_env, the reduced-model identification -- would have
switched roll model and capsize behaviour together with the physics, and an
as1 reduced model could never reproduce v0's fingerprint (review of S0/S1).

WHAT THE FROZEN FILES CANNOT KNOW
sim/planing_vessel_gz.from_plant and plant_for build the v0 PlaningVesselGZ
whatever the hull says, and they cannot be edited. gz_twin() and
gz_plant_for() are their hull-aware versions (for a v0 hull they call the
frozen ones); verify() raises when a plant does not run the physics its hull
names -- it catches the frozen from_plant applied to an as1 plant -- or runs
switches other than its version's preset (test_switches=); stamp() is what
a record should carry about the plant that made it, with the code hash of
the files this process imported (taken once, at import), not of whatever is
on disk when the record is written.

HULLS THE FROZEN PLANT CANNOT HONOUR (P0, the modular split)
A hull may now carry inputs v0 does not read: a geometry outside v0's family
member (hydro/hull_geometry), plant constants other than v0's literals, the
jet time constants v0 hard-codes (hydro/vessel_spec). v0 would build it
from its own formulas and run a different boat under the same name, so the
v0 dispatch refuses such a hull (hydro/vessel_spec.v0_representable); it
runs as 'NAME@as1'.

CACHE KEYS (ACCEPTANCE N3)
key_sha1(hull) is the physics_sha1 that a plant built for `hull` on its
version's preset stamps (code hash, hull manifest, switches), without
building it; cache_tag(hull) is what a cache file is named by: None for a
hull in LEGACY (scarab195 with its pinned manifest -- its caches predate
keying and keep their names, so v0 results and files are untouched), else
'NAME-<sha1[:12]>'. studies/final_eval.turn_path and sim/vessel_profile.py
use it.
"""
import copy
import hashlib
import importlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = "USV_HULL"
DEFAULT = "v0"
# the files whose bytes are each version's numerics (stamp, code_sha1)
V0_CODE = ("sim/planing_vessel.py", "sim/planing_vessel_gz.py",
           "sim/waterjet.py", "sim/actuators.py")
VERSIONS = {
    "v0": dict(plant=("sim.planing_vessel", "PlaningVessel"),
               gz=("sim.planing_vessel_gz", "PlaningVesselGZ"),
               code=V0_CODE),
    # in development until step S9 releases it: its preset of switches
    # (sim/planing_anyspeed.PRESETS) grows with each step, and stamp()
    # records the switches and the code so that nothing mixes meanwhile
    # (hydro/planing.py: PlaningHull and its scales(), the step of a plain
    # hull; sim/test_vessel.py: the calm sea of the captive runs)
    "as1": dict(plant=("sim.planing_anyspeed", "PlaningVesselASLin"),
                gz=("sim.planing_anyspeed", "PlaningVesselAS"),
                code=V0_CODE + ("sim/planing_anyspeed.py", "sim/physics.py",
                                # P1c: fk_exact's integrals (sim/
                                # planing_anyspeed._fk_b calls them)
                                "sim/as_kernels.py",
                                # P1c review: the batch mode; its records
                                # carry the plants' stamps, so its code is
                                # in the hash they carry
                                "sim/planing_batch.py",
                                "hydro/hull_geometry.py",
                                "hydro/vessel_spec.py", "hydro/planing.py",
                                "hydro/section2d.py",
                                "sim/test_vessel.py")),
}


# ------------------------------------------------------------- hull keys
def split(key):
    """(vessel, version) of a hull key: 'scarab195@as1' -> ('scarab195',
    'as1'); a plain name is v0. 'NAME@v0' is refused: v0 has one spelling,
    the plain name, so its caches have one name."""
    base, at, ver = str(key).partition("@")
    if not at:
        return base, DEFAULT
    if not base or ver == DEFAULT or ver not in VERSIONS:
        raise KeyError(f"hull key {key!r}: write NAME for v0 or NAME@VERSION "
                       f"with VERSION one of "
                       f"{sorted(v for v in VERSIONS if v != DEFAULT)}")
    return base, ver


def version(hull):
    """The physics version `hull` asks for. Its name and its `physics`
    attribute must say the same: a hull named without '@' is v0 whatever
    else it carries."""
    ver = getattr(hull, "physics", DEFAULT)
    name = getattr(hull, "name", None)
    named = split(name)[1] if isinstance(name, str) else DEFAULT
    if ver != named:
        raise ValueError(f"hull {name!r} carries physics {ver!r}: a hull's "
                         f"version is set only through its key "
                         f"(hydro.hulls.get('NAME@VERSION'), variant())")
    return ver


def variant(hull, ver, name=None):
    """A copy of the v0 planing hull `hull` on physics version `ver`, named
    NAME@ver -- what hydro.hulls.get('NAME@ver') returns; also the way to put
    a hull built outside the registry (a validation model) on a version.

    The key is a cache key (lofi_NAME@ver.json, steady states), so a hull
    that differs from the registered hull of the same name (a deep copy with
    other stations, say) is refused unless it is given a distinct `name`;
    `name` may not be a registered name."""
    if ver == DEFAULT or ver not in VERSIONS:
        raise KeyError(f"physics version {ver!r}: one of "
                       f"{sorted(v for v in VERSIONS if v != DEFAULT)}")
    if getattr(hull, "kind", None) != "planing":
        raise ValueError(f"{getattr(hull, 'name', hull)!r} is not a planing "
                         f"hull; physics versions are the planing plant's")
    if version(hull) != DEFAULT:
        raise ValueError(f"{hull.name!r} is already a variant")
    from hydro import hulls as _hulls
    base = hull.name if name is None else str(name)
    if "@" in base:
        raise ValueError(f"name {base!r}: no '@' (the version is added)")
    if name is not None and base in _hulls.REGISTRY:
        raise ValueError(f"name {base!r} is a registered hull; give the "
                         f"modified hull a name of its own")
    if name is None and base in _hulls.REGISTRY:
        reg = _hulls.REGISTRY[base]()
        if hull_sha1(hull) != hull_sha1(reg):
            f0, f1 = hull_fields(hull), hull_fields(reg)
            diff = sorted(k for k in set(f0) | set(f1)
                          if f0.get(k) != f1.get(k))
            raise ValueError(
                f"{base!r} differs from the registered hull of that name "
                f"(fields {diff}); its {ver} key would share caches with the "
                f"registered one -- pass variant(hull, {ver!r}, name=...)")
    h = copy.deepcopy(hull)
    h.name, h.physics = f"{base}@{ver}", ver
    return h


# --------------------------------------------------- worker processes
def use(key):
    """Hand the hull key `key` to the worker processes started from now on
    (they inherit the environment); None clears it. Returns the previous
    value, so a caller can put it back."""
    prev = os.environ.get(ENV)
    if key is None:
        os.environ.pop(ENV, None)
    else:
        split(key)
        os.environ[ENV] = str(key)
    return prev


def hull_key(default):
    """The hull key handed down by use(), else `default`."""
    key = os.environ.get(ENV) or default
    split(key)
    return key


def check_env(hull):
    """Raise if a hull key was handed down (use()) and `hull` is not it."""
    key = os.environ.get(ENV)
    name = getattr(hull, "name", None)
    if key and key != name:
        raise RuntimeError(f"{ENV}={key!r}, but a plant is being built for "
                           f"hull {name!r}: the environment carries the "
                           f"hull key to workers and must agree with the "
                           f"hull in hand")


# ------------------------------------------------------------- dispatch
def _cls(hull, which):
    ver = version(hull)
    check_env(hull)
    if ver == DEFAULT:
        from hydro.vessel_spec import v0_representable
        ok, why = v0_representable(hull)
        if not ok:
            raise ValueError(f"hull {getattr(hull, 'name', hull)!r}: the "
                             f"frozen v0 plant cannot honour its inputs "
                             f"({'; '.join(why)}); build it as "
                             f"'{getattr(hull, 'name', hull)}@as1'")
    mod, name = VERSIONS[ver][which]
    return getattr(importlib.import_module(mod), name)


def plant_class(hull):
    """The class PlaningHull.plant() builds for `hull`."""
    return _cls(hull, "plant")


def gz_class(hull):
    """The large-angle (GZ, capsizing) plant class for `hull`."""
    return _cls(hull, "gz")


TEST_KW = "test_switches"


def _no_test_kw(hull, kw):
    """A v0 plant swallows unknown keywords (PlaningVessel(**_ignored)), so
    test_switches= given for a v0 hull would silently run v0: refuse it."""
    for k in (TEST_KW, "switches"):
        if k in kw and version(hull) == DEFAULT:
            raise TypeError(f"{k}= is for the as plants' tests; "
                            f"{hull.name!r} is a v0 hull")


def build(hull, sea, **kw):
    """The plant PlaningHull.plant() builds: plant_class(hull)(hull, sea,
    **kw), refusing test_switches= for a v0 hull."""
    _no_test_kw(hull, kw)
    return plant_class(hull)(hull, sea, **kw)


def gz_plant_for(db, sea, hull=None, dt=None, **kw):
    """sim/planing_vessel_gz.plant_for, choosing the class by the hull."""
    from sim import config
    h = config.resolve(db, hull)
    if getattr(h, "kind", None) != "planing":
        raise ValueError("the GZ plant is the planing plant's; "
                         f"{getattr(h, 'name', h)!r} is not a planing hull")
    _no_test_kw(h, kw)
    return gz_class(h)(h, sea, dt=h.scales()["dt"] if dt is None else dt,
                       **kw)


def gz_twin(plant, **kw):
    """sim/planing_vessel_gz.from_plant, choosing the class by the hull: the
    large-angle twin of a built plant, same hull, sea, time step and options
    (an as plant's test switches included). For a v0 hull it IS the frozen
    from_plant, unchanged -- which drops a prop= or rudder= the plant was
    built with, as v0 always did. An as plant hands fresh copies of them on
    (sim/planing_anyspeed, 'ACTUATORS PASSED IN ARE KEPT': the one place
    where an all-off as plant differs from v0)."""
    cls = gz_class(plant.hull)
    _no_test_kw(plant.hull, kw)
    if version(plant.hull) == DEFAULT:
        from sim.planing_vessel_gz import from_plant
        return from_plant(plant, **kw)
    opts = plant._rebuild_kw(**kw)
    if plant.captive_u is not None:
        opts.setdefault("captive_u", plant.captive_u)
    return cls(plant.hull, plant.sea, **opts)


# ------------------------------------------------------------ provenance
def of(plant):
    """The physics a built plant runs; 'v0' for one that does not say."""
    return getattr(plant, "physics", DEFAULT)


def verify(plant, hull=None, allow_test_switches=False):
    """Raise unless `plant` runs the physics its hull names (and is built
    for `hull`, when given), on that version's preset of switches. A v0
    class built for an as1 hull -- the frozen
    sim/planing_vessel_gz.from_plant applied to an as1 plant -- fails here,
    so does a v0 class built directly for a hull v0 cannot honour
    (hydro/vessel_spec.v0_representable), and so does an as plant built
    with test_switches= unless the caller says it is a test
    (allow_test_switches=True)."""
    want = version(plant.hull)
    if hull is not None and getattr(hull, "name", hull) != plant.hull.name:
        raise RuntimeError(f"plant built for {plant.hull.name!r}, expected "
                           f"{getattr(hull, 'name', hull)!r}")
    got = of(plant)
    if got != want:
        raise RuntimeError(f"{type(plant).__name__} runs physics {got!r} but "
                           f"its hull {plant.hull.name!r} asks for {want!r} "
                           f"(build twins with physics.gz_twin, not the "
                           f"frozen sim/planing_vessel_gz.from_plant)")
    if got == DEFAULT and getattr(plant.hull, "kind", None) == "planing":
        # a frozen constructor called directly (PlaningVessel(hull, ...))
        # skips the dispatch's refusal and builds v0's geometry from the
        # hull fields whatever the hull describes
        from hydro.vessel_spec import v0_representable
        ok, why = v0_representable(plant.hull)
        if not ok:
            raise RuntimeError(f"{type(plant).__name__} was built for "
                               f"{plant.hull.name!r}, which the frozen v0 "
                               f"plant cannot honour ({'; '.join(why)}): it "
                               f"ran another boat; build through hull.plant /"
                               f" physics.build")
    if got != DEFAULT and not allow_test_switches:
        from sim.planing_anyspeed import PRESETS
        sw = tuple(getattr(plant, "switches", ()))
        if sw != PRESETS[got]:
            raise RuntimeError(f"{type(plant).__name__} runs switches {sw}, "
                               f"not the {got} preset {PRESETS[got]} "
                               f"(built with {TEST_KW}=; pass "
                               f"allow_test_switches=True for a test)")
    return got


def code_sha1_disk(ver):
    """sha1 over the bytes of the files that make version `ver`'s
    numerics, as they are on disk now."""
    h = hashlib.sha1()
    for f in VERSIONS[ver]["code"]:
        h.update(f.encode())
        with open(os.path.join(ROOT, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


_CODE = {}


def code_sha1(ver):
    """code_sha1_disk(ver) as it was when this process first imported this
    module (taken at import, at the end of the module): the hash of the code
    that runs. A file edited while a long job runs does not change what its
    records carry; code_changed() says whether it happened."""
    if ver not in _CODE:
        _CODE[ver] = code_sha1_disk(ver)
    return _CODE[ver]


def code_changed(ver):
    """True if version `ver`'s files on disk no longer hash to code_sha1."""
    return code_sha1_disk(ver) != code_sha1(ver)


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in sorted(v.items(), key=str)}
    if isinstance(v, (set, frozenset)):
        return sorted(_plain(x) for x in v)
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return repr(v)


def hull_fields(hull):
    """The hull's inputs as JSON values: every instance field but T, which
    the plant derives and writes back on its first build. A v0 hull has no
    `physics` field (PlaningHull's class default), so this manifest of the
    plain hull did not change when versions were added."""
    return {k: _plain(v) for k, v in sorted(vars(hull).items()) if k != "T"}


def hull_sha1(hull):
    return hashlib.sha1(json.dumps(hull_fields(hull), sort_keys=True)
                        .encode()).hexdigest()


def stamp_for(hull, ver, switches):
    """The stamp of a plant of version `ver` with `switches` built for
    `hull` (stamp), computed from them alone.

    Every switch is in the hash, the implementation-only ones too
    (sim/planing_anyspeed: fast_path, which changes no bit -- step P1c,
    checks C3, C7). So two sets that differ only by such a switch get
    different physics_sha1 and separate caches although their results are
    the same bits. That duplication is intended (P1c review): the hash then
    never depends on a claim of bit-identity holding in cases no check
    covered, and only test runs build the non-preset set (key_sha1, the
    cache key, is the preset's)."""
    sw = list(switches)
    h = hashlib.sha1()
    for part in (code_sha1(ver), hull_sha1(hull), json.dumps(sw)):
        h.update(part.encode())
    return dict(physics=ver, switches=sw, hull=hull.name,
                physics_sha1=h.hexdigest())


def stamp(plant, fresh=False):
    """What a record should carry about the plant that made it: version,
    switches, hull key and one sha1 over the version's code (as imported,
    code_sha1), the hull's inputs and the switches. A record without it is
    v0. An as plant takes its stamp when it is built (plant._stamp) and this
    returns that one; fresh=True computes it anew."""
    if not fresh and getattr(plant, "_stamp", None) is not None:
        return dict(plant._stamp)
    return stamp_for(plant.hull, of(plant),
                     getattr(plant, "switches", ()))


def preset(ver):
    """The switches a plant of version `ver` runs on its preset."""
    if ver == DEFAULT:
        return ()
    from sim.planing_anyspeed import PRESETS, resolve
    return resolve(ver, PRESETS[ver])


def key_sha1(hull):
    """The physics_sha1 a plant built for `hull` on its version's preset
    stamps (code as imported, hull manifest, switches) -- without building
    it."""
    ver = version(hull)
    return stamp_for(hull, ver, preset(ver))["physics_sha1"]


# hulls whose caches predate keying: name -> the manifest sha1 they were
# made with (studies/anyspeed_regress_as.HULLS is this dict: its freeze tier
# checks the registered hull against it). Their cache files keep their
# legacy names and are never rewritten by keyed code.
LEGACY = {"scarab195": "8bf0b328c3d8147d797dfaef87ad64862c50e17c"}


def cache_tag(hull):
    """None for a LEGACY hull (v0, its pinned manifest): its caches keep
    their names. Else 'NAME-<key_sha1[:12]>', the name made file-safe."""
    if version(hull) == DEFAULT and LEGACY.get(hull.name) == hull_sha1(hull):
        return None
    safe = "".join(c if c.isalnum() or c in "-_.@" else "_"
                   for c in hull.name)
    return f"{safe}-{key_sha1(hull)[:12]}"


# the code hashes of the process, taken when it first imports this module
for _v in VERSIONS:
    code_sha1(_v)
