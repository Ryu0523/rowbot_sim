"""
The product layer for one boat (learn/meta/PRODUCT_LAYER.md): what LOTUSim
offers around a simulator and this project lacked, built on the project's
own plant and controllers without changing either.

    import app
    env = app.UsvEnv("app/scenarios/demo_scarab195.json")      # Gym-style
    rec = app.run("app/scenarios/demo_scarab195.json", {"type": "hand"})
    rec.save("app/runs/demo.npz"); app.export.viewer_json(rec, "x.json")

    python -m app hulls | run | replay | export | serve     (app/__main__.py)

Modules
    hulls      hull keys NAME[@VERSION], the extra test hulls, inline hulls
    geo        WGS84 geodetic <-> ENU, quaternions (ENU world, FLU body)
    scenario   JSON (YAML when available) scenario files, LOTUSim key names
    track      straight tracks and waypoint legs, line-of-sight guidance
    sensors    IMU, GNSS, heading, speed log, wave probe, wave preview
    models     identified reduced model (cached), the heading autopilot
    env        UsvEnv: reset / step / spaces, recording
    policies   scripted, hand MPC, m0, learned rows, any callable
    runner     run(scenario, policy) -> Record
    record     one record per run, fingerprint, command replay (from any
               row: replay(rec, start=k), plant_at(rec, j))
    plantstate the plant's state outside s (hidden state, actuators) for
               resuming from a recorded row
    codeguard  load every version's physics code with its hash at once;
               is the code on disk still the code this process runs?
    frozen     the frozen final simulator configuration (SIM_FINAL
               2026-10-09): selector, plant kind, steps, filter, seeds
    randomize  domain randomisation (SIM_FINAL section 5): config, draw,
               how each parameter is applied
    accel      the 20 Hz causal / zero-phase filter, the bow gauge, the
               acceleration statistics (SIM_FINAL section 6)
    export     viewer JSON (viewer/demo.html), Blender scene.json
    lotus      LOTUSim-shaped JSON messages
    wsproto    a small RFC 6455 websocket (stdlib only)
    server     HTTP + server-sent events + websocket + UDP
    ros2_bridge  optional ROS 2 node (not required to run here)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

RUNS = os.path.join(ROOT, "app", "runs")
SCENARIOS = os.path.join(ROOT, "app", "scenarios")
CACHE = os.path.join(ROOT, "studies", "_cache", "app")

__version__ = "0.1"


def __getattr__(name):
    # lazy: importing `app` must not import the plant
    if name == "UsvEnv":
        from .env import UsvEnv
        return UsvEnv
    if name == "run":
        from .runner import run
        return run
    if name == "load_scenario":
        from .scenario import load
        return load
    if name == "make_policy":
        from .policies import make_policy
        return make_policy
    if name in ("export", "record", "hulls", "geo", "scenario", "sensors",
                "track", "lotus", "runner", "policies", "env", "server",
                "models", "wsproto", "plantstate", "accel", "randomize",
                "frozen"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
