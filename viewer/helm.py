#!/usr/bin/env python3
"""
Take the helm of the REAL plant -- all 110 states, driven from a browser.

The viewer's built-in drive mode integrates the 10-state reduced model, because
a browser cannot run the full solve. That was a compromise about WHERE the
physics runs, not about whether it can run: the plant does 760 steps/s in
numpy, which is 38x real time at dt = 0.05. There is a factor of thirty-eight
of headroom, so a person can drive the genuine article -- it just has to run in
Python and send the result to the browser.

    python -m viewer.helm            then open http://127.0.0.1:8770

The browser keeps its job (rendering, input) and loses the physics. What you
steer is then EXACTLY the environment `sim/rl_env.py` trains on, so a human lap
and a learned policy are directly comparable -- which is the point. A human
baseline on the identical environment is worth more than any amount of arguing
about whether a policy is good.

WHAT COMES BACK that the reduced model could not give you:

    slam force        peaks near half the vessel's weight, and you feel the
                      bow being stopped rather than the bow acceleration alone
    roll              a whole degree of freedom the reduced model does not have
    added resistance  the speed you lose to waves, which is why the throttle
                      does not mean what it means in calm water
    fluid memory      2.2-2.9 s of it, rather than a fitted damping ratio

No dependencies beyond the standard library: the plant runs in a background
thread at wall-clock pace and the browser polls over plain HTTP. Localhost
round trips are well under a millisecond, so at 30 Hz the transport is not the
bottleneck -- and this keeps the whole thing a `python -m` away rather than a
package install.
"""
import json
import os
import threading
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import numpy as np

from hydro import bem
from sim.vessel import NonlinearVessel
from sim.wavefield import SeaState

HERE = os.path.dirname(os.path.abspath(__file__))
G = 9.81
DT = 0.05
# must match studies/export_viewer.py, or the water the browser draws is not
# the water the plant is being pushed by
N_FREQ, N_DIR = 16, 4
HS, TP = 3.25, 9.7


class Helm:
    """The plant, running in real time, taking commands from anywhere."""

    def __init__(self, db, seed=0, u0=3.5):
        self.db = db
        self.lock = threading.Lock()
        self.cmd = dict(thrust=0.45, rudder=0.0)
        self.seed = seed
        self.u0 = u0
        self._reset_flag = True
        self.state = {}
        self.running = True

    def _build(self):
        sea = SeaState(HS, TP, n_freq=N_FREQ, n_dir=N_DIR, seed=self.seed)
        self.plant = NonlinearVessel(
            self.db, sea, L=self.db.L,
            B=float(self.db.attrs.get("B", 2.5)),
            T=float(self.db.attrs.get("T", 0.8)), dt=DT)
        self.s = self.plant.initial_state(self.u0)
        self.t = 0.0
        self.cost = 0.0
        self.dist = 0.0
        self.slam0 = 0

    def loop(self):
        self._build()
        next_t = time.perf_counter()
        while self.running:
            with self.lock:
                if self._reset_flag:
                    self._build()
                    self._reset_flag = False
                c = dict(self.cmd)
            thrust = float(np.clip(c["thrust"], 0, 1)) * self.plant.prop.t_max
            rudder = float(np.clip(c["rudder"], -1, 1)) * self.plant.rudder.max
            self.s = self.plant.step(self.s, self.t, thrust, rudder, DT)
            self.t += DT

            s = self.s
            a = self.plant.last_bow_acc / G
            # progress along +x, the direction the destination lies in
            vmg = float(s[6] * np.cos(s[5]) - s[7] * np.sin(s[5]))
            self.dist += max(vmg, 0.0) * DT
            self.cost += a * a * DT
            with self.lock:
                self.state = dict(
                    t=self.t, x=float(s[0]), y=float(s[1]), z=float(s[2]),
                    roll=float(s[3]), pitch=float(s[4]), yaw=float(s[5]),
                    u=float(s[6]), v=float(s[7]), r=float(s[11]),
                    acc=float(a), vmg=vmg,
                    thr=float(self.plant.unpack(s)[3]
                              / self.plant.prop.t_max),
                    rud=float(self.plant.unpack(s)[4]),
                    slam_f=float(self.plant.last_slam_force / 1e3),
                    slams=int(self.plant.slam_count),
                    rel_bow=float(self.plant.last_rel_bow),
                    cost=float(self.cost / self.dist * 100)
                    if self.dist > 5 else None,
                    seed=self.seed)
            # pace to wall clock; if we fall behind, skip rather than spiral
            next_t += DT
            lag = next_t - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.5:
                next_t = time.perf_counter()


def make_handler(helm):
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=HERE, **kw)

        def log_message(self, *a):
            pass

        def _json(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/state"):
                with helm.lock:
                    return self._json(helm.state)
            return super().do_GET()

        def do_POST(self):
            if not self.path.startswith("/cmd"):
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                d = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                d = {}
            with helm.lock:
                if d.get("reset"):
                    helm._reset_flag = True
                    if "seed" in d:
                        helm.seed = int(d["seed"])
                if "thrust" in d:
                    helm.cmd["thrust"] = float(d["thrust"])
                if "rudder" in d:
                    helm.cmd["rudder"] = float(d["rudder"])
            self._json({"ok": True})
    return H


def main(port=8770, seed=0):
    db = bem.load("hydro_wigley_10m.npz")
    helm = Helm(db, seed=seed)
    threading.Thread(target=helm.loop, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(helm))
    n_state = helm.db and NonlinearVessel(
        db, SeaState(HS, TP, n_freq=N_FREQ, n_dir=N_DIR, seed=seed),
        L=db.L, dt=DT).n_state
    print(f"  plant: {n_state} states, running at wall-clock pace")
    print(f"  open  http://127.0.0.1:{port}/seaway_full.html?live")
    print("  the page will say LIVE PLANT once it finds this server\n")
    print("  W/S throttle, A/D rudder, space centres, R restarts")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        helm.running = False
        print("\n  stopped")


if __name__ == "__main__":
    main()
