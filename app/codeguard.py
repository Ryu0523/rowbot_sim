"""
Is the physics code this process RUNS the code on disk?

sim/physics.code_sha1(ver) is the hash of a version's files taken when the
process first imports sim/physics.py; a record's stamp carries it. A
module is compiled when it is first imported, and the plant modules are
imported lazily -- sim/planing_anyspeed.py when the first as1 plant is
built, some of its helpers only inside the step that needs them. A process
that imported sim.physics at start-up and built its first as1 plant an
hour later, after an edit, would run the edited code under the old hash
(review of the P1c round: the server on 8766 served and recorded that way).

load_all() closes that window: it imports sim/physics.py and every module
of every version's code list (sim/physics.VERSIONS[ver]['code']) at once,
so the hash and the compiled code are taken together. stale(ver) then says
whether the disk has moved on since (sim/physics.code_changed, plus a
change caught between the two imports); check(ver, refuse) raises
StaleCode or warns once. app/server.py refuses new live runs of a stale
version (HTTP 409, a session's reset replies with the error) and reports
it in the viewer meta ('server_code_stale'); UsvEnv calls load_all() and
warns (a notebook keeps working; its records carry the hash of the code
that ran, which is right, but not the disk's).

What the code lists must contain is studies/app_test.py's group
'codehash': every repository module a plant of each version loads or runs
with all its switches on, in a fresh process, is in its version's list or
named there as not numeric.
"""
import importlib
import threading
import warnings

_LOCK = threading.Lock()
_LOADED = {}            # ver -> True if the disk differed right after load
_WARNED = set()


class StaleCode(RuntimeError):
    """The process runs physics code older than the files on disk."""


def modules_of(ver):
    """The importable module names of version `ver`'s code list."""
    from sim import physics
    return [f[:-3].replace("/", ".") for f in physics.VERSIONS[ver]["code"]
            if f.endswith(".py")]


def load_all():
    """Import sim/physics.py and every version's code modules now (once per
    process); returns {ver: stale at load}."""
    with _LOCK:
        if _LOADED:
            return dict(_LOADED)
        from sim import physics
        for ver in physics.VERSIONS:
            for m in modules_of(ver):
                importlib.import_module(m)
        for ver in physics.VERSIONS:
            _LOADED[ver] = physics.code_changed(ver)
        return dict(_LOADED)


def stale(ver):
    """None when version `ver`'s files on disk hash to what this process
    loaded; else a sentence saying they differ (restart the process)."""
    from sim import physics
    load_all()
    if not (_LOADED.get(ver) or physics.code_changed(ver)):
        return None
    now, disk = physics.code_sha1(ver)[:12], physics.code_sha1_disk(ver)[:12]
    return (f"this process runs {ver} physics code {now}, the files on "
            f"disk now hash to {disk}: restart it to run (and record) the "
            f"current code")


def notes(vers=None):
    """[stale(ver)] for the versions given (default: all), Nones left out."""
    from sim import physics
    vers = list(physics.VERSIONS) if vers is None else list(vers)
    return [n for n in (stale(v) for v in vers) if n]


def check(ver, refuse=False):
    """Raise StaleCode (refuse=True) or warn once when `ver` is stale."""
    note = stale(ver)
    if note is None:
        return
    if refuse:
        raise StaleCode(note)
    if ver not in _WARNED:
        _WARNED.add(ver)
        warnings.warn(note, RuntimeWarning, stacklevel=3)
