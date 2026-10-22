"""
Hull keys for the product layer.

    resolve("scarab195")          v0 physics, hydro.hulls.get as everywhere
    resolve("scarab195@as1")      the same vessel on the any-speed plant
    resolve("probe_hullB")        the second, NOT similar parametric hull of
                                  studies/principles_hull_swap.py (L 6.6 m,
                                  deadrise 16 deg), built by its make_hull
    resolve("probe_hullB@as1")
    resolve({"name": "myboat", "L": 6.0, "b": 2.0, ...})   inline PlaningHull

The version grammar and the variant copy are sim/physics.py's (split,
variant), so a hull's key is its name and every cache keyed by name keeps
the versions apart (ACCEPTANCE N3). hydro/hulls.py is not edited: the extra
hulls live in EXTRA here.
"""
import copy
import hashlib
import json

REG_DOC = {
    "planing_b": "second parametric planing hull of hydro/hulls.py: "
                 "L 6.6 m, deadrise 16 deg, transom 10 % narrower than the "
                 "beam, raised forefoot",
}
EXTRA_DOC = {
    "probe_hullB": "second parametric planing hull of "
                   "studies/principles_hull_swap.py (make_hull('hullB')): "
                   "L 6.6 m, b 2.0 m, deadrise 16 deg, 1900 kg, LCG 0.36 L",
}


def _probe_hullB():
    from studies.principles_hull_swap import make_hull
    return make_hull("hullB", 1.0)[0]


EXTRA = {"probe_hullB": _probe_hullB}


def _inline(spec):
    from hydro.planing import PlaningHull
    from hydro import hulls as reg
    from sim import physics
    spec = dict(spec)
    name = str(spec.pop("name", ""))
    if not name:
        raise KeyError("an inline hull needs a 'name'")
    base, ver = physics.split(name)
    if base in reg.REGISTRY or base in EXTRA:
        raise KeyError(f"inline hull name {base!r} is a registered hull; "
                       f"give it a name of its own")
    h = PlaningHull(name=base, **spec)
    return h if ver == physics.DEFAULT else physics.variant(h, ver)


_RESOLVED = {}


def resolve(key):
    """The hull for a key (str), an inline spec (dict) or a hull object.

    One hull object per key and process (sim/config.resolve: "Resolve ONCE
    and pass the result on"): the plant caches its steady running attitude
    and its righting-arm table per hull object (keys with id(hull)), so
    every env after the first on a key builds its plant in milliseconds
    instead of seconds -- the same numbers, computed once. Inline specs are
    keyed by their canonical JSON, so two specs that share a name but not
    their numbers stay apart."""
    if isinstance(key, dict):
        ck = "inline:" + json.dumps(key, sort_keys=True, default=str)
        if ck not in _RESOLVED:
            _RESOLVED[ck] = _inline(key)
        return _RESOLVED[ck]
    if not isinstance(key, str):
        return key
    if key not in _RESOLVED:
        _RESOLVED[key] = _resolve_key(key)
    return _RESOLVED[key]


def _resolve_key(key):
    from hydro import hulls as reg
    from sim import physics
    base, ver = physics.split(key)
    if base in reg.REGISTRY:
        return reg.get(key)
    if base in EXTRA:
        h = EXTRA[base]()
        return h if ver == physics.DEFAULT else physics.variant(h, ver)
    raise KeyError(f"unknown hull {key!r}; known: "
                   f"{sorted(reg.REGISTRY) + sorted(EXTRA)} "
                   f"(append @VERSION for another physics version)")


def key_of(spec):
    """The hull key a spec stands for (the name of the resolved hull)."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return str(spec.get("name"))
    return getattr(spec, "name", repr(spec))


def spec_sha1(spec):
    """sha1 of a hull spec as written (inline dicts canonicalised)."""
    s = json.dumps(spec, sort_keys=True, default=str) \
        if isinstance(spec, dict) else str(spec)
    return hashlib.sha1(s.encode()).hexdigest()


_RUNNABLE = {}


def runnable_hull(h):
    """(True, version) when UsvEnv can run the resolved hull `h`: a planing
    hull whose physics version the dispatch accepts (sim/physics
    plant_class and gz_class -- the selector itself, so the frozen v0
    plant's refusal of a hull it cannot honour, hydro/vessel_spec
    .v0_representable, is the same refusal here); else (False, why).
    It does not build a plant: a refusal of the plant's constructor (an
    inline hull with transom_correction=True on as1, say) is raised by
    UsvEnv, which builds one at once. Building calm plants here would put
    4-7 s per key on the listing (both classes, the steady attitude and
    the righting-arm table), and every registered and extra hull builds."""
    if not is_planing(h):
        return False, (f"{getattr(h, 'kind', None) or 'displacement'} hull:"
                       f" the planing plant cannot run it")
    from sim import physics
    try:
        physics.plant_class(h)
        physics.gz_class(h)
    except (ValueError, RuntimeError, KeyError) as ex:
        return False, str(ex)
    return True, physics.version(h)


def runnable(key):
    """(True, note) when UsvEnv can run the hull key `key`, else (False,
    why). Builds the hull once per process (a few seconds for the meshed
    ones), then cached; never builds a plant."""
    if key not in _RUNNABLE:
        import logging
        logging.disable(logging.WARNING)        # mesh builders' chatter
        try:
            ok, why = runnable_hull(resolve(key))
        except Exception as ex:                      # noqa: BLE001
            ok, why = False, f"does not build: {type(ex).__name__}: {ex}"
        finally:
            logging.disable(logging.NOTSET)
        _RUNNABLE[key] = (ok, why)
    return _RUNNABLE[key]


def _short(why, name):
    """A one-line reason for the listing."""
    if "cannot honour" in why:
        i = why.find("(")
        j = why.rfind(")")
        parts = (why[i + 1:j] if 0 <= i < j else why).split("; ")
        more = (f", and {len(parts) - 1} other input"
                f"{'s' if len(parts) > 2 else ''} differ"
                f"{'' if len(parts) > 2 else 's'} from v0's fixed values"
                if len(parts) > 1 else "")
        return f"v0 refuses it ({parts[0]}{more}): run it as {name}@as1"
    return why


def listing():
    """[(key, physics version, runnable, note)] of every key this layer
    knows: each registered or extra hull as NAME (v0) and, for a planing
    hull, NAME@VERSION for every other version; runnable = UsvEnv accepts
    the key (runnable_hull: a planing hull the dispatch builds -- e.g.
    planing_b only as planing_b@as1)."""
    from hydro import hulls as reg
    from sim import physics
    vers = [v for v in physics.VERSIONS if v != physics.DEFAULT]
    out = []
    for name in list(reg.REGISTRY) + list(EXTRA):
        doc = EXTRA_DOC.get(name) or REG_DOC.get(name, "hydro/hulls.py")
        ok, why = runnable(name)
        out.append((name, physics.DEFAULT, ok,
                    doc + ("" if ok else f"; {_short(why, name)}")))
        try:
            planing = is_planing(resolve(name))
        except Exception:                            # noqa: BLE001
            planing = False
        if not planing:
            continue
        for v in vers:
            key = f"{name}@{v}"
            ok, why = runnable(key)
            out.append((key, v, ok, doc + ("" if ok else f"; {why}")))
    return out


def is_planing(h):
    return getattr(h, "kind", None) == "planing"


def copy_hull(h):
    return copy.deepcopy(h)
