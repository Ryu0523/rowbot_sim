"""
The app server: the browser demo, live runs, and the external JSON
interfaces, standard library only.

    python -m app serve [--port 8766] [--udp 8767]
    open http://127.0.0.1:8766/            (viewer/demo.html)

HTTP (127.0.0.1 only)
  GET  /                          -> viewer/demo.html
  GET  /viewer/..., /app/runs/..., /app/scenarios/...   static files
  GET  /api/hulls                 hull keys
  GET  /api/scenarios             scenario files
  GET  /api/runs                  finished and live runs
  GET  /api/run/<id>/viewer.json  a run's viewer data (built from the record)
  GET  /api/stream/<id>           server-sent events of a run: "header"
                                  (meta, sea, hull, first frame), "frames"
                                  (one batch per control step), "end"
  POST /api/run                   {"scenario": "demo_scarab195.json",
                                   "policy": {...}, "rate": 1.0, "seed",
                                   "duration", "model"} -> {"run_id"};
                                  429 when the live-run limits are reached
                                  (MAX_LIGHT light runs, MAX_HEAVY m0 /
                                  learned run, free RAM above a reserve);
                                  learned rows always run at rate 0 (about
                                  10x slower than real time: follow, then
                                  replay)
  POST /api/stop/<id>
  POST /api/snapshot              {"name", "png": data URL} -> saves
                                  viewer/demo_shots/<name>.png
  POST /api/session               one lockstep message (app/lotus.Session);
                                  the reset reply carries "session": an id
                                  that every later message must carry
                                  (app/lotus.SessionTable: one lock per
                                  session, idle ones expire, optional "seq"
                                  for idempotent retries)
WEBSOCKET  /ws                    lockstep JSON session per connection
UDP        --udp PORT             lockstep JSON session per client address
                                  (SessionTable; "seq" makes a resent
                                  datagram safe; each datagram is served by
                                  a worker thread, so one client's slow
                                  reset does not block the others)

SAFETY. Bound to 127.0.0.1. POSTs and websocket upgrades from a browser
must come from this server's own origin (a page elsewhere cannot drive
it). The HTTP API takes scenario NAMES from app/scenarios (or inline
scenario objects) and refuses "callable" policies, which import code --
those are for Python and the command line -- whether they come in the
request or in the scenario's own policy. Static files come only from
viewer/, app/runs/ and app/scenarios/ (safe_static: no backslashes, colons,
percent signs or '..' components after unquoting; the real path must lie
inside one of the three folders).

CODE ON DISK. The server loads every version's physics code with its hash
when it starts (app/codeguard.load_all). If a version's files change on
disk afterwards, a POST /api/run or a session reset of that version is
refused (409 / an "error" reply naming StaleCode: restart the server), and
viewer data says so in meta.server_code_stale (the page's title adds
"server runs old physics code: restart it").
"""
import base64
import json
import math
import mimetypes
import os
import re
import socket
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import numpy as np

from . import ROOT, RUNS, SCENARIOS
from . import lotus, wsproto
from .codeguard import StaleCode

SHOTS = os.path.join(ROOT, "viewer", "demo_shots")
STATIC = ("viewer/", "app/runs/", "app/scenarios/")
_ID = re.compile(r"^[A-Za-z0-9_.@\-]+$")


# live-run limits (start_run): running at the same time, and the free RAM
# (GB) a new run must leave (studies/parallel_runner.free_gb)
MAX_LIGHT, MAX_HEAVY = 4, 1
GB_LIGHT, GB_HEAVY, RESERVE_GB = 0.3, 3.0, 3.0
KEEP_FINISHED = 20           # finished live runs kept in memory (all on disk)
EVENTS_GRACE_S = 30.0        # a finished run's events kept for late readers


def _clean(o):
    """o with every non-finite float (Python or numpy) -> None, numpy
    arrays and scalars -> lists and numbers, tuples and sets -> lists."""
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else str(k)): _clean(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if o is None or isinstance(o, (str, int, bool)):
        return o
    return repr(o)


def dumps(o):
    """JSON text of o; NaN and infinities become null (strict JSON) without
    touching any string."""
    return json.dumps(_clean(o), allow_nan=False)


def safe_static(path, root=ROOT):
    """The real path of a static file under viewer/, app/runs/ or
    app/scenarios/, or None. `path` is the URL path (not unquoted)."""
    rel = unquote(path).lstrip("/")
    if (not rel.startswith(STATIC) or "\\" in rel or ":" in rel
            or "%" in rel or "\x00" in rel
            or any(p in ("..", ".") for p in rel.split("/"))):
        return None
    full = os.path.realpath(os.path.join(root, *rel.split("/")))
    for d in STATIC:
        allowed = os.path.realpath(os.path.join(root, d))
        try:
            if os.path.commonpath([full, allowed]) == allowed:
                return full if os.path.isfile(full) else None
        except ValueError:                      # another drive
            continue
    return None


def is_callable_spec(pol):
    """A policy spec that would import code (refused over HTTP)."""
    if isinstance(pol, str):
        return ":" in pol and not pol.startswith("learned:")
    if isinstance(pol, dict):
        return pol.get("type") == "callable"
    return pol is not None


def is_heavy_spec(pol):
    """m0 and learned rows (torch, the GPU, GBs); the rest is light."""
    if isinstance(pol, str):
        return pol == "m0" or pol.startswith("learned:")
    if isinstance(pol, dict):
        return pol.get("type") in ("m0", "learned")
    return False


def is_learned_spec(pol):
    if isinstance(pol, str):
        return pol.startswith("learned:")
    return isinstance(pol, dict) and pol.get("type") == "learned"


class Busy(Exception):
    """The live-run limits are reached (HTTP 429)."""


class _HTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that does not print a traceback when a browser
    drops a keep-alive connection (navigating away): that is not an
    error."""
    daemon_threads = True

    def handle_error(self, request, client_address):
        import sys
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


# ------------------------------------------------------------ live runs
class LiveRun(threading.Thread):
    """One run in a thread: the runner with an on_step that turns every
    control step into a frame batch for the stream."""

    def __init__(self, run_id, scenario, policy, rate=1.0, seed=None,
                 duration=None, model=None, runs_dir=RUNS):
        super().__init__(daemon=True)
        self.runs_dir = runs_dir
        self.run_id = run_id
        self.scenario, self.policy, self.rate = scenario, policy, rate
        self.seed, self.duration, self.model = seed, duration, model
        self.events = []
        self.cond = threading.Condition()
        self.status = "starting"
        self.error = None
        self.stop_flag = False
        self.record_path = None
        self.summary = None
        self._n_sent = 0
        self.t_started = time.time()
        self.t_finished = None
        self.readers = 0
        self.heavy = is_heavy_spec(policy)
        self.end_event = None
        self._policy_name = None

    def push(self, kind, data):
        with self.cond:
            self.events.append((kind, data))
            if kind in ("end", "error"):
                self.end_event = (kind, data)
            self.cond.notify_all()

    @property
    def finished(self):
        return self.status in ("done", "error")

    def prune(self, now=None):
        """Drop the frame events of a finished run nobody is reading (its
        record is on disk; a late reader gets a 'replay' event)."""
        now = time.time() if now is None else now
        with self.cond:
            if (self.finished and self.readers == 0 and self.events and
                    self.t_finished is not None and
                    now - self.t_finished > EVENTS_GRACE_S):
                self.events = []

    def _header(self, env):
        from sim import physics
        from . import codeguard, export
        h = env.hull
        geo = physics.gz_class(h)(h, env.sea, dt=env.dt)
        X0, Y0 = float(env.s0[0]), float(env.s0[1])
        tr = env.track.describe()
        tr["points"] = [[x - X0, y - Y0] for x, y in tr["points"]]
        sea = SimpleNamespace(**{k: np.asarray(getattr(env.sea, k), float)
                                 for k in ("a", "k", "th", "w", "phi")})
        meta = dict(run=self.run_id, scenario=env.scenario.name,
                    hull_key=env.hull_key, physics=env.stamp()["physics"],
                    plant=type(env.plant).__name__,
                    policy=self._policy_name,
                    state_source=export.state_source(self.policy),
                    hs=float(env.sea.hs),
                    tp=float(env.sea.tp), theta0=float(env.sea.theta0),
                    dt=env.dt, t0=0.0, t1=env.n_ctrl * env.dt_step,
                    t_first=0.0, origin=[X0, Y0], g=9.81,
                    reference_position=env.frame.as_dict(), track=tr,
                    live=True, seed=env.seed_, stills=[],
                    t_max=env.t_max, nozzle_max=env.rud_max,
                    hull_tag=export.hull_tag(h),
                    server_code_stale=codeguard.notes(
                        [physics.version(h)]))
        self._geo, self._sea, self._X0, self._Y0 = geo, sea, X0, Y0
        return dict(meta=meta, sea=export.rebased_sea(sea, X0, Y0),
                    hull=export.hull_dict(h, geo),
                    frames=self._frames(env, 0, initial=True))

    def _frames(self, env, i0, initial=False):
        from . import export
        P = env._P
        if initial:
            T = np.array([0.0])
            S = env.s0[None].copy()
            a_cg = a_bow = np.zeros(1)
            thr, noz = S[:, 12], S[:, 13]
        else:
            T = np.asarray(P["t"][i0:], float)
            S = np.asarray(P["s"][i0:], float).reshape(-1, 14)
            a_cg = np.asarray(P["a_cg"][i0:], float)
            a_bow = np.asarray(P["a_bow"][i0:], float)
            thr = np.asarray(P["thrust"][i0:], float)
            noz = np.asarray(P["nozzle"][i0:], float)
        if not len(T):
            return None
        fr = export.frames_of(None, self._geo, self._sea, S, T, a_cg, a_bow,
                              thr, noz, self._X0, self._Y0)
        return fr

    def run(self):
        from .policies import label
        from .runner import run as run_episode
        try:
            pol = self.policy
            self._policy_name = label(pol)
            self.status = "running"

            def on_step(env, info):
                if self._n_sent == 0:
                    self.push("header", self._header(env))
                fr = self._frames(env, self._n_sent)
                self._n_sent = len(env._P["t"])
                if fr is not None:
                    self.push("frames", dict(
                        frames=fr, k=env.k, leg=info["leg"],
                        reward=info["reward"], reason=info["reason"]))

            # a policy that raises mid-run gives a partial record whose
            # reason says so (runner.run); it is saved like any other
            rec = run_episode(self.scenario, pol, seed=self.seed,
                              duration=self.duration, model=self.model,
                              rate=self.rate, on_step=on_step,
                              stop=lambda: self.stop_flag)
            path = os.path.join(self.runs_dir, self.run_id + ".npz")
            rec.save(path)
            self.record_path = path
            self.summary = rec.summary()
            err = rec.header.get("error")
            if err:
                self.error = rec.header["reason"]
            self.status = "done"
            self.t_finished = time.time()
            self.push("end", dict(summary=self.summary, record=os.path.relpath(
                path, ROOT).replace("\\", "/"), reason=rec.header["reason"],
                error=(err or {}).get("message")))
        except Exception as ex:                       # noqa: BLE001
            # set-up failures (scenario, policy reset): nothing to record
            self.status = "error"
            self.t_finished = time.time()
            self.error = f"{type(ex).__name__}: {ex}"
            traceback.print_exc()
            self.push("error", dict(error=self.error))


# ------------------------------------------------------------ the server
class AppServer:
    def __init__(self, host="127.0.0.1", port=8766, udp_port=None,
                 verbose=False, runs_dir=RUNS):
        self.host, self.port, self.udp_port = host, port, udp_port
        self.runs_dir = runs_dir
        self.runs = {}
        self.verbose = verbose
        self.httpd = _HTTPServer((host, port), self._handler())
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.udp = None
        self.sessions = lotus.SessionTable(
            lambda: lotus.Session(allow_inline=True, runs_dir=runs_dir,
                                  refuse_stale=True))
        self._runs_lock = threading.Lock()
        # the physics code this server runs is loaded now, all of it, with
        # its hash (app/codeguard.py): a file edited later makes the server
        # stale -- new live runs of that version are refused (409) and the
        # viewer data says so -- instead of quietly running the edit under
        # the old stamp
        from . import codeguard
        codeguard.load_all()

    def origin_ok(self, origin, host_header):
        if not origin:
            return True                      # not a browser cross-site call
        o = urlparse(origin)
        return o.hostname in ("127.0.0.1", "localhost") and \
            (o.port == self.port)

    def start(self, background=True):
        if self.udp_port is not None:
            self.udp = UdpBridge(self.host, self.udp_port,
                                 runs_dir=self.runs_dir)
            self.udp.start()
        if background:
            t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            t.start()
            return t
        self.httpd.serve_forever()

    def shutdown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.udp is not None:
            self.udp.stop()
        for r in self.runs.values():
            r.stop_flag = True

    # -------------------------------------------------------- runs
    def _janitor(self):
        """Prune finished runs' events and keep at most KEEP_FINISHED
        finished runs in memory (their records stay on disk)."""
        now = time.time()
        with self._runs_lock:
            for r in self.runs.values():
                r.prune(now)
            done = [rid for rid, r in self.runs.items()
                    if r.finished and r.readers == 0]
            for rid in done[:max(0, len(done) - KEEP_FINISHED)]:
                del self.runs[rid]

    def start_run(self, req):
        """A live run thread for a POST /api/run request: (id, note)."""
        from .policies import label, make_policy
        from .scenario import load
        from studies.parallel_runner import free_gb
        self._janitor()
        if not isinstance(req, dict):
            raise ValueError("a JSON object")
        pol = req.get("policy")
        if pol is not None and not isinstance(pol, (str, dict)):
            raise ValueError("policy: a name or a JSON object")
        if is_callable_spec(pol):
            raise ValueError("callable policies are not accepted over HTTP")
        sc = req.get("scenario", "demo_scarab195.json")
        sc = load(sc) if isinstance(sc, dict) else load(
            lotus.scenario_path(sc))
        if pol is None:
            pol = sc.policy_spec()
            if is_callable_spec(pol):
                raise ValueError("the scenario's policy is a callable: not "
                                 "accepted over HTTP (run it from Python "
                                 "or the command line)")
        make_policy(pol)                        # a bad spec fails here
        from sim import physics
        from . import codeguard, hulls
        codeguard.check(physics.version(hulls.resolve(sc.hull_spec)),
                        refuse=True)            # StaleCode: HTTP 409
        sc.with_overrides(seed=req.get("seed"), duration=req.get("duration"),
                          model=req.get("model"))   # and bad overrides
        rate = float(req.get("rate", 1.0))
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("rate: >= 0 (0 = as fast as it runs)")
        note = None
        if is_learned_spec(pol) and rate > 0:
            rate, note = 0.0, ("learned rows run about 10x slower than real "
                               "time: rate set to 0 (follow it live, then "
                               "replay the record)")
        heavy = is_heavy_spec(pol)
        with self._runs_lock:
            running = [r for r in self.runs.values() if not r.finished]
            n_heavy = sum(r.heavy for r in running)
            n_light = len(running) - n_heavy
            if heavy and n_heavy >= MAX_HEAVY:
                raise Busy(f"{n_heavy} m0 / learned run(s) already running "
                           f"(at most {MAX_HEAVY}); stop it or wait")
            if not heavy and n_light >= MAX_LIGHT:
                raise Busy(f"{n_light} live runs already running (at most "
                           f"{MAX_LIGHT}); stop one or wait")
            need = (GB_HEAVY if heavy else GB_LIGHT) + RESERVE_GB
            free = free_gb()
            if free < need:
                raise Busy(f"{free:.1f} GB free RAM; a new "
                           f"{'m0 / learned' if heavy else 'live'} run "
                           f"needs {need:.1f} GB (incl. a {RESERVE_GB:g} GB "
                           f"reserve)")
            pname = re.sub(r"[^A-Za-z0-9]+", "_", label(pol)).strip("_")
            rid = re.sub(r"[^A-Za-z0-9_.@\-]", "_",
                         f"{time.strftime('%Y%m%d_%H%M%S')}_{sc.name}_"
                         f"{pname}")
            n = 1
            base = rid
            while rid in self.runs or os.path.exists(os.path.join(
                    self.runs_dir, rid + ".npz")):
                n += 1
                rid = f"{base}_{n}"
            run = LiveRun(rid, sc, pol, rate=rate, seed=req.get("seed"),
                          duration=req.get("duration"),
                          model=req.get("model"), runs_dir=self.runs_dir)
            self.runs[rid] = run
        run.start()
        return rid, note

    def list_runs(self):
        self._janitor()
        out = []
        for rid, r in list(self.runs.items()):
            out.append(dict(id=rid, status=r.status, live=True,
                            summary=r.summary, error=r.error))
        if os.path.isdir(self.runs_dir):
            for f in sorted(os.listdir(self.runs_dir)):
                if f.endswith(".npz") and f[:-4] not in self.runs:
                    out.append(dict(id=f[:-4], status="done", live=False))
        return out

    def viewer_cache_path(self, rid, header):
        """Where the server keeps a run's viewer JSON: keyed by the run AND
        the cache tag of the hull as the current code builds it
        (export.hull_tag: sim/physics.cache_tag -- version code, hull
        inputs, preset switches). The JSON draws the CURRENT hull and
        reports what changed since the record (record.code_changes), so a
        new physics code state or hull gets a new file and an older file
        is never rewritten or served for it (ACCEPTANCE N3)."""
        from . import CACHE, export, hulls
        from .scenario import Scenario
        tag = export.hull_tag(hulls.resolve(
            Scenario(header["scenario"]).hull_spec))
        safe = re.sub(r"[^A-Za-z0-9_.@-]+", "_", tag)
        # one folder per runs folder: two servers on other folders (the
        # tests') never share a run id's file
        import hashlib
        rd = hashlib.sha1(os.path.realpath(self.runs_dir).encode()
                          ).hexdigest()[:10]
        # export.VIEWER_FORMAT: a file written by an older exporter (201
        # uniform hull points, no shader table) is not served for a newer
        # one -- and not rewritten either
        return os.path.join(CACHE, "viewer", rd,
                            f"{rid}.{safe}.f{export.VIEWER_FORMAT}.json")

    def viewer_data(self, rid):
        if not _ID.match(rid):
            raise ValueError("bad run id")
        vj = os.path.join(self.runs_dir, rid + ".viewer.json")
        npz = os.path.join(self.runs_dir, rid + ".npz")
        if not os.path.exists(npz):
            # a viewer export without its record (written by `python -m
            # app export`): served as it is
            if os.path.exists(vj):
                with open(vj, "rb") as f:
                    return f.read()
            raise FileNotFoundError(rid)
        with np.load(npz, allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
        cp = self.viewer_cache_path(rid, header)
        if not (os.path.exists(cp)
                and os.path.getmtime(cp) >= os.path.getmtime(npz)):
            from . import export
            from .record import Record
            tmp = f"{cp}.{os.getpid()}.{threading.get_ident()}.tmp"
            export.viewer_json(Record.load(npz), tmp, run=rid)
            os.replace(tmp, cp)
        with open(cp, "rb") as f:
            data = json.loads(f.read())
        # whether THIS server's code is stale is a fact of now, not of the
        # time the cached file was written
        from sim import physics
        from . import codeguard
        ver = (header.get("physics") or {}).get("physics", physics.DEFAULT)
        data["meta"]["server_code_stale"] = codeguard.notes(
            [ver] if ver in physics.VERSIONS else None)
        return json.dumps(data, separators=(",", ":")).encode()

    # -------------------------------------------------------- handler
    def _handler(self):
        app = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *a):
                if app.verbose:
                    super().log_message(fmt, *a)

            def _send(self, code, body, ctype="application/json"):
                if isinstance(body, (dict, list)):
                    body = dumps(body)
                if isinstance(body, str):
                    body = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _read_body(self):
                """Read the whole request body first, whatever the answer
                will be: an unread body would be parsed as the next request
                on a keep-alive connection."""
                n = int(self.headers.get("Content-Length", 0) or 0)
                if n > 32 * 2 ** 20:
                    self.close_connection = True
                    raise ValueError("request too large")
                self._raw = self.rfile.read(n) if n else b""

            def _body(self):
                return json.loads(self._raw or b"{}")

            def _origin_ok(self):
                return app.origin_ok(self.headers.get("Origin"),
                                     self.headers.get("Host"))

            def do_GET(self):
                self._upgraded = False
                try:
                    self._get()
                except (BrokenPipeError, ConnectionError):
                    pass
                except Exception as ex:               # noqa: BLE001
                    if self._upgraded:   # never HTTP into a websocket
                        self.close_connection = True
                        return
                    if isinstance(ex, FileNotFoundError):
                        self._send(404, dict(error=f"not found: {ex}"))
                    else:
                        self._send(400, dict(error=f"{type(ex).__name__}: "
                                                   f"{ex}"))

            def _get(self):
                path = urlparse(self.path).path
                if path in ("/", "/index.html"):
                    self.send_response(302)
                    self.send_header("Location", "/viewer/demo.html")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if path == "/ws":
                    self._websocket()
                    return
                if path == "/api/hulls":
                    from .hulls import listing
                    self._send(200, [dict(key=k, physics=v, runnable=r,
                                          note=n)
                                     for k, v, r, n in listing()])
                    return
                if path == "/api/scenarios":
                    out = []
                    for f in sorted(os.listdir(SCENARIOS)):
                        if f.endswith(".json"):
                            try:
                                with open(os.path.join(SCENARIOS, f),
                                          encoding="utf-8") as fh:
                                    d = json.load(fh)
                                out.append(dict(file=f, name=d.get("name"),
                                                description=d.get(
                                                    "description")))
                            except Exception:          # noqa: BLE001
                                out.append(dict(file=f))
                        elif f.endswith((".yaml", ".yml")):
                            out.append(dict(file=f))
                    self._send(200, out)
                    return
                if path == "/api/runs":
                    self._send(200, app.list_runs())
                    return
                m = re.match(r"^/api/run/([^/]+)/viewer\.json$", path)
                if m:
                    self._send(200, app.viewer_data(m.group(1)))
                    return
                m = re.match(r"^/api/stream/([^/]+)$", path)
                if m:
                    self._stream(m.group(1))
                    return
                self._static(path)

            def _websocket(self):
                if not self._origin_ok():
                    self._send(403, dict(error="cross-origin"))
                    return
                ws = wsproto.handshake(self)
                if ws is None:
                    return
                self._upgraded = True
                self.close_connection = True
                sess = lotus.Session(runs_dir=app.runs_dir,
                                     refuse_stale=True)
                while True:
                    try:
                        msg = ws.recv()
                    except wsproto.ProtocolError as ex:
                        ws.close(ex.code, ex.reason)
                        self._drain()
                        return
                    except (ConnectionError, OSError):
                        return
                    if msg is None:
                        return
                    try:
                        ws.send(dumps(sess.handle(msg)))
                    except (ConnectionError, OSError):
                        return
                    except Exception as ex:       # noqa: BLE001
                        ws.close(1011, f"{type(ex).__name__}")
                        return

            def _drain(self, timeout=1.0):
                """After our close frame: stop writing and read what the
                peer still sends (its close frame, or the rest of a bad
                frame) until it closes or `timeout` -- closing a socket with
                unread data makes Windows send RST, which can discard the
                close frame before the peer reads it."""
                try:
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_WR)
                    self.connection.settimeout(timeout)
                    t_end = time.time() + timeout
                    while time.time() < t_end:
                        if not self.connection.recv(65536):
                            break
                except OSError:
                    pass

            def _static(self, path):
                full = safe_static(path)
                if full is None:
                    self._send(404, dict(error="not found"))
                    return
                ctype = mimetypes.guess_type(full)[0] or \
                    "application/octet-stream"
                with open(full, "rb") as f:
                    body = f.read()
                self._send(200, body, ctype)

            def _stream(self, rid):
                run = app.runs.get(rid)
                if run is None:
                    self._send(404, dict(error=f"no live run {rid}"))
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                with run.cond:
                    pruned = run.finished and not run.events
                    if not pruned:
                        run.readers += 1
                if pruned:
                    # its frames were dropped: point at the record
                    kind, data = run.end_event or ("error", {})
                    try:
                        self.wfile.write(("event: replay\ndata: " + dumps(
                            dict(run=rid, kind=kind, end=data,
                                 viewer=f"/api/run/{rid}/viewer.json"))
                            + "\n\n").encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionError, OSError):
                        pass
                    return
                try:
                    self._stream_events(run)
                finally:
                    with run.cond:
                        run.readers -= 1

            def _stream_events(self, run):
                i = 0
                last = time.time()
                try:
                    while True:
                        with run.cond:
                            while i >= len(run.events):
                                if not run.cond.wait(timeout=10.0):
                                    break
                            batch = run.events[i:]
                        for kind, data in batch:
                            self.wfile.write(
                                f"event: {kind}\ndata: {dumps(data)}\n\n"
                                .encode())
                            i += 1
                            if kind in ("end", "error"):
                                self.wfile.flush()
                                return
                        if not batch and time.time() - last > 9.0:
                            self.wfile.write(b": keep-alive\n\n")
                        last = time.time()
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionError, OSError):
                    return

            def do_HEAD(self):
                path = urlparse(self.path).path
                ok = path in ("/", "/index.html") or \
                    safe_static(path) is not None
                self.send_response(200 if ok else 404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                try:
                    self._read_body()
                    if not self._origin_ok():
                        self._send(403, dict(error="cross-origin request"))
                        return
                    path = urlparse(self.path).path
                    if path == "/api/run":
                        rid, note = app.start_run(self._body())
                        self._send(200, dict(run_id=rid, note=note))
                        return
                    m = re.match(r"^/api/stop/([^/]+)$", path)
                    if m:
                        r = app.runs.get(m.group(1))
                        if r:
                            r.stop_flag = True
                        self._send(200, dict(ok=bool(r)))
                        return
                    if path == "/api/snapshot":
                        b = self._body()
                        name = re.sub(r"[^A-Za-z0-9_.\-]", "_",
                                      str(b.get("name", "shot")))[:120]
                        png = str(b.get("png", ""))
                        if not png.startswith("data:image/png;base64,"):
                            raise ValueError("png: a data:image/png URL")
                        os.makedirs(SHOTS, exist_ok=True)
                        p = os.path.join(SHOTS, name + ".png")
                        with open(p, "wb") as f:
                            f.write(base64.b64decode(png.split(",", 1)[1]))
                        self._send(200, dict(saved=os.path.relpath(
                            p, ROOT).replace("\\", "/"),
                            bytes=os.path.getsize(p)))
                        return
                    if path == "/api/session":
                        try:
                            b = self._body()
                        except ValueError as ex:
                            self._send(200, dict(op="error", error=(
                                f"{type(ex).__name__}: {ex}")))
                            return
                        key = b.get("session") if isinstance(b, dict) \
                            else None
                        self._send(200, app.sessions.handle(key, b))
                        return
                    self._send(404, dict(error="not found"))
                except (BrokenPipeError, ConnectionError):
                    pass
                except Busy as ex:
                    self._send(429, dict(error=str(ex)))
                except StaleCode as ex:
                    self._send(409, dict(error=f"StaleCode: {ex}"))
                except Exception as ex:               # noqa: BLE001
                    self._send(400, dict(error=f"{type(ex).__name__}: "
                                               f"{ex}"))

        return H


class UdpBridge(threading.Thread):
    """Lockstep JSON sessions over UDP, one per client (host, port), in a
    lotus.SessionTable: per-session locks, idle eviction, and "seq" for
    retries (a datagram resent after a lost reply returns the cached reply
    and steps nothing). Each datagram is handled by a worker thread, so a
    client's reset (seconds, cold) does not block the others; one session's
    messages are still served one at a time."""

    def __init__(self, host="127.0.0.1", port=8767, runs_dir=RUNS,
                 idle_s=600.0, workers=8):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.5)
        self.port = self.sock.getsockname()[1]
        self.sessions = lotus.SessionTable(
            lambda: lotus.Session(allow_inline=True, runs_dir=runs_dir,
                                  refuse_stale=True),
            idle_s=idle_s)
        self.pool = ThreadPoolExecutor(max_workers=workers,
                                       thread_name_prefix="udp")
        self._stop = threading.Event()

    def _serve(self, data, addr):
        out = dumps(self.sessions.handle(addr, data, create_any=True)) \
            .encode()
        if len(out) > 65000:
            out = dumps(dict(op="error", error="reply exceeds one "
                             "datagram; use the websocket")).encode()
        try:
            self.sock.sendto(out, addr)
        except OSError:
            pass

    def run(self):
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                self.sessions.evict()
                continue
            except OSError:
                break
            try:
                self.pool.submit(self._serve, data, addr)
            except RuntimeError:                     # pool shut down
                break

    def stop(self):
        self._stop.set()
        self.pool.shutdown(wait=False)
        try:
            self.sock.close()
        except OSError:
            pass


def serve(port=8766, udp_port=None, verbose=True):
    srv = AppServer(port=port, udp_port=None, verbose=verbose)
    if udp_port is not None:
        srv.udp = UdpBridge(srv.host, udp_port, runs_dir=srv.runs_dir)
        srv.udp.start()
    print(f"app server on http://127.0.0.1:{srv.port}/  (websocket "
          f"ws://127.0.0.1:{srv.port}/ws"
          + (f", UDP 127.0.0.1:{srv.udp.port}" if srv.udp else "") + ")",
          flush=True)
    srv.httpd.serve_forever()
