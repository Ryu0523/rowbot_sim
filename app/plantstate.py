"""
The plant's state outside its 14-number vector, so that a run can be
resumed from any recorded row bit for bit (ACCEPTANCE P4).

WHAT A PLANT STEP DEPENDS ON BESIDES s
  physics-owned  plant.hidden_state() / set_hidden_state()
                 (sim/planing_anyspeed._ASMixin; switch wagner_entry: the
                 planes' flow-state field and the state the last step
                 returned; None on the as1 preset; a v0 plant has neither
                 method). The app carries it OPAQUE through the exact codec
                 below (encode / decode), so whatever a later physics step
                 adds to hidden_state() is carried as well, unread.
  v0's own       what sim/planing_vessel.py and sim/planing_vessel_gz.py
                 keep between steps and do not expose (AUX): the jet's and
                 the nozzle's command delay lines (sim/actuators.Delay.buf),
                 the jet's prime (sim/waterjet.Waterjet.prime), the slam
                 detector (_emerged, slam_count), the running peaks
                 (peak_slam_force, peak_slam_force_full) and the GZ plant's
                 capsize (capsized, capsize_t; _capsize_state is the state
                 of that row).
  Everything else a plant keeps is a constant of its construction, a
  scratch value its derivative recomputes before use (SCRATCH), an output
  of the last step that the next step overwrites (last_*), a table built
  lazily on first use, or a log. audit() checks that claim: it compares
  every attribute of two plants (recursively into the actuators and every
  object of sim / hydro, callable ones such as the delay lines included),
  and studies/app_test.py runs it on a plant restored at a row against a
  deep copy of the original plant at that row, after each has taken the
  same next step -- a hidden state added later and not carried shows up
  there by name, and the end-to-end resume checks fail with it.

FUNCTIONS ARE MARKERS, NOT WALKED (2026-10-10). A plain function, a bound
method or a functools.partial a plant keeps (the geometry's chine / keel
functions, slam.wagner_slope, the field memory _esm's chine function
since 2026-10-07) is written as one stable marker -- module, qualified
name and a hash of its code (bytecode, names, constants; nested code
recursively) -- and its default arguments, closure cells, bound instance
and partial arguments are NOT entered. Reason: copy.deepcopy does not
copy functions, so a deep copy's closures point at the ORIGINAL plant's
objects (the geometry `self._geom` behind _esm[0]); walking them from
the copy reached the original's geometry under a path the original did
not have (its own geometry was already `seen` under plant._geom), and an
unaltered deep copy read as 29 differences. Consequence: an object that
is reachable ONLY through a function's closure or defaults is not
audited. Every such object in the plants today is also an attribute
(the geometry is plant._geom, keel_rest is in the transom namespace, the
closed-over `self` is the plant); a closure over a new, otherwise
unreachable mutable object would have to be exposed as an attribute to
be covered. Swapping a function for another one (other code or name) is
still a named difference, through the marker. Objects that merely define
__call__ (sim/actuators.Delay, filters, rate limiters) are not functions
and are walked as before.

ROWS. Row j is the plant right after plant step j: the record's
plant arrays t[j], s[j] (app/record.py), its aux[j] and hidden[j]. Row -1
is the reset state (a fresh plant: nothing to restore). The plant's own
log, trajectory()['hidden'][j] (GZ plant built with record=True), is the
hidden state BEFORE the step from s[j] -- None where that step started
from the steady field -- so, with no hook editing the state between steps,
trajectory()['hidden'][j + 1] equals row j (checked bit for bit in
app_test, group resume).
"""
import base64
import functools
import hashlib
import json
import math
import struct
import types

import numpy as np

FORMAT = "usv-app-state/1"
# v0 attributes carried between steps (module docstring), as columns
SCALARS = (("prop", "prime"), (None, "_emerged"), (None, "slam_count"),
           (None, "peak_slam_force"), (None, "peak_slam_force_full"),
           (None, "capsized"), (None, "capsize_t"))
DELAYS = ("prop", "rudder")
# per-derivative or per-step scratch: recomputed before it is read
SCRATCH = ("_w_last", "_trs", "_taper_cap", "_aux0", "_jet_f", "_es_run")
# logs and counters that do not feed back
LOGS = ("_log", "_es_log", "es_resets", "record")


# ------------------------------------------------------------ codec
def encode(obj):
    """An exact, JSON-able form of a nested value (dict / list / tuple /
    ndarray / numpy scalar / float / int / bool / str / None): arrays keep
    dtype, shape (0-d too), memory order (C or Fortran; any other strided
    view is stored C-ordered, its values exact) and bytes; floats their bits
    (float.hex; a NaN by its 8 bytes, payload and sign kept); tuples stay
    tuples. Object arrays and anything else raise TypeError at once -- not
    later, when a resume reads them back."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.ndarray):
        if obj.dtype.hasobject:
            raise TypeError(f"hidden state: an object array ({obj.dtype}) "
                            f"cannot be carried exactly (its bytes are "
                            f"pointers)")
        fortran = (obj.ndim > 1 and obj.flags.f_contiguous
                   and not obj.flags.c_contiguous)
        order = "F" if fortran else "C"
        a = np.array(obj, order=order, copy=True)
        out = {"nd": a.dtype.str, "shape": list(obj.shape),
               "b64": base64.b64encode(a.tobytes(order=order)).decode()}
        if fortran:
            out["order"] = "F"
        return out
    if isinstance(obj, np.generic):
        if obj.dtype.hasobject:
            raise TypeError("hidden state: a numpy object scalar cannot be "
                            "carried exactly")
        return {"np": obj.dtype.str,
                "b64": base64.b64encode(obj.tobytes()).decode()}
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return {"fbits": struct.pack("<d", obj).hex()}
        return {"f": obj.hex()}
    if isinstance(obj, tuple):
        return {"tuple": [encode(x) for x in obj]}
    if isinstance(obj, list):
        return [encode(x) for x in obj]
    if isinstance(obj, dict):
        bad = [k for k in obj if not isinstance(k, str)]
        if bad:
            raise TypeError(f"hidden state: dict keys must be str, not "
                            f"{bad[:3]!r}")
        return {"dict": {k: encode(v) for k, v in obj.items()}}
    raise TypeError(f"hidden state: cannot carry a {type(obj).__name__} "
                    f"exactly (extend app/plantstate.encode)")


def decode(obj):
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, list):
        return [decode(x) for x in obj]
    if "nd" in obj:
        order = obj.get("order", "C")
        a = np.frombuffer(base64.b64decode(obj["b64"]),
                          dtype=np.dtype(obj["nd"]))
        return a.reshape(obj["shape"], order=order).copy(order=order)
    if "np" in obj:
        return np.frombuffer(base64.b64decode(obj["b64"]),
                             dtype=np.dtype(obj["np"]))[0]
    if "f" in obj:
        return float.fromhex(obj["f"])
    if "fbits" in obj:
        return struct.unpack("<d", bytes.fromhex(obj["fbits"]))[0]
    if "tuple" in obj:
        return tuple(decode(x) for x in obj["tuple"])
    if "dict" in obj:
        return {k: decode(v) for k, v in obj["dict"].items()}
    raise ValueError(f"hidden state: unknown encoded form {sorted(obj)}")


def dumps_rows(rows):
    """The per-row hidden states (a list, None allowed) as uint8 bytes of
    canonical JSON -- what a record stores as plant['hidden']."""
    txt = json.dumps([encode(r) for r in rows], separators=(",", ":"),
                     sort_keys=True)
    return np.frombuffer(txt.encode(), np.uint8).copy()


def loads_rows(arr):
    return [decode(r) for r in json.loads(bytes(np.asarray(arr, np.uint8))
                                          .decode())]


def same(a, b):
    """True when two (decoded) hidden states are the same bits."""
    return json.dumps(encode(a), sort_keys=True) == json.dumps(
        encode(b), sort_keys=True)


# ------------------------------------------------------------ the plant
def hidden(plant):
    """plant.hidden_state(), or None for a plant without one (v0)."""
    f = getattr(plant, "hidden_state", None)
    return f() if f is not None else None


def set_hidden(plant, h):
    f = getattr(plant, "set_hidden_state", None)
    if f is None:
        if h is not None:
            raise ValueError(f"{type(plant).__name__} has no hidden state "
                             f"to restore (the record was made by a plant "
                             f"that had one)")
        return
    f(h)


def _get(plant, owner, name):
    o = plant if owner is None else getattr(plant, owner, None)
    return getattr(o, name, None) if o is not None else None


def delay_len(plant, owner):
    d = getattr(getattr(plant, owner, None), "delay", None)
    return int(getattr(d, "n", 0)) if d is not None else 0


def columns(plant):
    """The names of the aux columns for `plant` (fixed by its actuators'
    delay lengths)."""
    cols = [n if o is None else f"{o}.{n}" for o, n in SCALARS]
    for o in DELAYS:
        cols += [f"{o}.delay.buf[{i}]" for i in range(delay_len(plant, o))]
    return cols


def aux_row(plant):
    """v0's carried state (AUX) as one float row: NaN where the plant does
    not have the attribute (or a delay line has not been primed yet)."""
    out = []
    for o, n in SCALARS:
        v = _get(plant, o, n)
        out.append(np.nan if v is None else float(v))
    for o in DELAYS:
        n = delay_len(plant, o)
        buf = getattr(getattr(getattr(plant, o, None), "delay", None),
                      "buf", None)
        if buf is None:
            out += [np.nan] * n
        else:
            if len(buf) != n:
                raise RuntimeError(f"{o}.delay.buf holds {len(buf)} values, "
                                   f"its delay is {n} steps")
            out += [float(x) for x in buf]
    return out


def set_aux(plant, row, s):
    """Restore aux_row() on `plant`; `s` is the state of that row (the GZ
    plant holds it once capsized)."""
    row = np.asarray(row, float)
    k = 0
    for o, n in SCALARS:
        v = row[k]
        k += 1
        tgt = plant if o is None else getattr(plant, o, None)
        if tgt is None or not hasattr(tgt, n):
            if not np.isnan(v):
                raise ValueError(f"the record carries {o or 'plant'}.{n} "
                                 f"but this plant has none")
            continue
        cur = getattr(tgt, n)
        if n == "capsize_t":
            setattr(tgt, n, None if np.isnan(v) else float(v))
        elif isinstance(cur, (bool, np.bool_)):
            setattr(tgt, n, bool(v))
        elif isinstance(cur, (int, np.integer)):
            setattr(tgt, n, int(v))
        else:
            setattr(tgt, n, float(v))
    for o in DELAYS:
        n = delay_len(plant, o)
        vals = row[k:k + n]
        k += n
        d = getattr(getattr(plant, o, None), "delay", None)
        if d is None or n == 0:
            continue
        d.buf = None if np.isnan(vals).all() else [float(x) for x in vals]
    if k != len(row):
        raise ValueError(f"aux row of {len(row)} values, this plant's "
                         f"columns are {k}")
    if getattr(plant, "capsized", False):
        plant._capsize_state = np.array(s, float).copy()


def restore(plant, s, hidden_row, aux):
    """Put `plant` (freshly reset, same hull, sea and switches) at a
    recorded row: its hidden state and v0's carried state. The caller
    steps on from `s`."""
    set_hidden(plant, hidden_row)
    set_aux(plant, aux, s)


# ------------------------------------------------------------ the audit
# what the audit walks into: objects of the plant's own packages (and
# SimpleNamespace, module 'types'), callable or not -- a Delay line, a
# filter or a rate limiter that defines __call__ keeps state like any other
# object. A function, a bound method or a partial is written as a marker
# (module docstring, FUNCTIONS ARE MARKERS): never entered. Classes,
# modules and builtins are skipped.
_OWN = ("sim", "hydro", "types")
_SKIP = (type, types.ModuleType, types.BuiltinFunctionType,
         types.BuiltinMethodType, types.MethodWrapperType,
         types.WrapperDescriptorType, types.MethodDescriptorType)
_CALLABLE = (types.FunctionType, types.MethodType, functools.partial)


def _own(o):
    return str(getattr(o, "__module__", None) or "").split(".")[0] in _OWN


def _code_sig(code, h=None):
    """sha1 over what a code object computes: bytecode, names, argument
    count, free variables and constants (nested code objects recursively,
    so a lambda inside is covered without its memory address)."""
    top = h is None
    h = hashlib.sha1() if top else h
    h.update(code.co_code)
    h.update(repr((code.co_argcount, code.co_names, code.co_varnames,
                   code.co_freevars)).encode())
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            _code_sig(c, h)
        else:
            h.update(repr(c).encode())
    return h.hexdigest()[:12] if top else None


def callable_marker(o, roots=frozenset()):
    """The stable marker written for a function, bound method or partial:
    '<function MODULE.QUALNAME code:HASH>'; a bound method names its
    instance's type ('the plant' when the instance is a plant being
    compared); a partial names its function and the arguments it carries
    (their count and keyword names). Equal for a function and any copy or
    rebuild of it (functions are not copied by deepcopy; a rebuilt plant's
    closures are new objects with the same code)."""
    if isinstance(o, functools.partial):
        return (f"<partial {callable_marker(o.func, roots)} "
                f"args:{len(o.args)} kw:{sorted(o.keywords or {})}>")
    if isinstance(o, types.MethodType):
        inst = o.__self__
        who = "the plant" if id(inst) in roots else type(inst).__name__
        return f"<method {callable_marker(o.__func__, roots)} of {who}>"
    code = getattr(o, "__code__", None)
    sig = _code_sig(code) if code is not None else "none"
    return (f"<function {getattr(o, '__module__', None)}."
            f"{getattr(o, '__qualname__', type(o).__name__)} code:{sig}>")


def _flat(o, path, out, seen, depth, roots=frozenset()):
    """Every comparable value under `o` into out[path]. `roots`: the ids of
    the plants being compared -- a reference back to one of them from
    inside (a bound method's instance) is the plant, whose state is walked
    from the top, so it is written as a marker. Functions, methods and
    partials are markers (callable_marker), never entered."""
    if depth > 6:
        return
    if depth > 0 and id(o) in roots:
        out[path] = "<the plant>"
        return
    if o is None or isinstance(o, (bool, int, float, str, np.generic)):
        out[path] = repr(o) if not isinstance(o, float) else o.hex()
        return
    if isinstance(o, np.ndarray):
        out[path] = (o.dtype.str, o.shape, o.tobytes())
        return

    def rec(v, p):
        _flat(v, p, out, seen, depth + 1, roots)
    if isinstance(o, (list, tuple)):
        out[path + "#len"] = len(o)
        for i, x in enumerate(o[:4096]):
            rec(x, f"{path}[{i}]")
        return
    if isinstance(o, dict):
        for k in sorted(o, key=str):
            rec(o[k], f"{path}{{{k}}}")
        return
    if id(o) in seen or isinstance(o, _SKIP):
        return
    if isinstance(o, _CALLABLE):
        # a marker, never entered (module docstring, FUNCTIONS ARE MARKERS)
        out[path] = callable_marker(o, roots)
        return
    if hasattr(o, "__dict__") and _own(type(o)):
        seen.add(id(o))
        for k, v in vars(o).items():
            if k in ("sea", "hull", "db") or k in SCRATCH or k in LOGS                     or k.startswith("last_"):
                continue
            rec(v, f"{path}.{k}")


def audit(a, b, count=False):
    """[attribute path] where plants `a` and `b` differ, scratch values,
    logs and last-step outputs left out (module docstring). A table a plant
    builds lazily on first use (`self.__dict__.get(name)`, then set: its 2D
    section table _sec2d, the field's geometry _esg) exists only in a plant
    whose steps have needed it; such an attribute, present in one plant
    only, is not a difference here. Compare two plants after each has taken
    the same step from the row, from a row where the tables are in use, and
    they are compared too. count=True: (differences, [attributes present in
    one plant only], number of paths compared)."""
    fa, fb = {}, {}
    roots = frozenset((id(a), id(b)))
    _flat(a, "plant", fa, set(), 0, roots)
    _flat(b, "plant", fb, set(), 0, roots)
    top = lambda k: k.split(".")[1].split("[")[0].split("{")[0].split("#")[0]
    ta, tb = {top(k) for k in fa}, {top(k) for k in fb}
    one = sorted(ta ^ tb)
    d = sorted(k for k in set(fa) | set(fb)
               if fa.get(k) != fb.get(k) and top(k) not in one)
    return (d, one, len(set(fa) | set(fb))) if count else d
