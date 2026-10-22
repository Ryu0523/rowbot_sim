"""
Command line of the product layer.

    python -m app hulls
    python -m app run demo_scarab195.json [--policy hand] [--seed 3]
                  [--duration 60] [--model probe_hullB@as1] [--out NAME]
                  [--blender T0:T1]
    python -m app replay app/runs/NAME.npz
    python -m app export app/runs/NAME.npz [--viewer out.json]
                  [--blender scene.json --t0 40 --t1 44 --fps 30]
                  [--video out.mp4 [--render] [--res 960x540]
                   [--samples 16] [--engine EEVEE]]
    python -m app serve [--port 8766] [--udp 8767]

`run` writes app/runs/NAME.npz (the record) and NAME.viewer.json (open
http://127.0.0.1:8766/viewer/demo.html?run=NAME with the server running, or
viewer/demo.html?data=../app/runs/NAME.viewer.json from any static server).
--policy takes a JSON object or a short form: scripted | hand | m0 |
learned:<row> | pkg.mod:function.

`export` needs at least one of --viewer / --blender. --video (with
--blender) prints the Blender command that renders the window as PNG
frames plus an MP4 (viewer/proto_blender/render.py --clip); --render also
runs it (Blender located by app.export.find_blender: $BLENDER, PATH, or
Program Files/Blender Foundation).

`replay` first says whether the physics code or the hull changed since the
record was made (app.record.code_changes), then compares the replay with
the record (physical fingerprint).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def _policy(s):
    if s is None:
        return None
    s = s.strip()
    if s.startswith("{"):
        return json.loads(s)
    return s


def _window(s):
    try:
        a, b = (float(x) for x in s.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"T0:T1 seconds, not {s!r}")
    return a, b


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m app")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hulls")
    r = sub.add_parser("run")
    r.add_argument("scenario")
    r.add_argument("--policy")
    r.add_argument("--seed", type=int)
    r.add_argument("--duration", type=float)
    r.add_argument("--model")
    r.add_argument("--out")
    r.add_argument("--blender", type=_window,
                   help="T0:T1 also writes NAME.scene.json")
    r.add_argument("--quiet", action="store_true")
    p = sub.add_parser("replay")
    p.add_argument("record")
    p.add_argument("--start", type=int, default=0,
                   help="resume after this control step (format-2 records)")
    e = sub.add_parser("export")
    e.add_argument("record")
    e.add_argument("--viewer")
    e.add_argument("--blender", help="scene.json for render.py")
    e.add_argument("--t0", type=float)
    e.add_argument("--t1", type=float)
    e.add_argument("--fps", type=float, default=30.0)
    e.add_argument("--video", help="MP4 path (with --blender): the "
                   "window's frames rendered as a clip")
    e.add_argument("--render", action="store_true",
                   help="run Blender now (else only print the command)")
    e.add_argument("--engine", default="EEVEE")
    e.add_argument("--res", default="960x540")
    e.add_argument("--samples", type=int, default=16)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8766)
    s.add_argument("--udp", type=int)
    a = ap.parse_args(argv)

    if a.cmd == "hulls":
        from app.hulls import listing
        print(f"  {'key':<22} {'phys':<5} {'runs':<5} note")
        for k, v, ok, n in listing():
            print(f"  {k:<22} {v:<5} {'yes' if ok else 'no':<5} {n}")
        print("  (runs = UsvEnv accepts the key: a planing hull the physics "
              "selector builds on that version; 'no' gives the reason -- "
              "displacement hulls, or a hull v0 cannot honour, which runs "
              "as NAME@as1)")
        return 0
    if a.cmd == "run":
        from app import RUNS, export
        from app.runner import run
        from app.scenario import load
        t0 = time.time()
        sc = load(a.scenario)
        rec = run(sc, _policy(a.policy), seed=a.seed, duration=a.duration,
                  model=a.model, verbose=not a.quiet)
        label = export.policy_label(rec.header)
        name = a.out or re.sub(r"[^A-Za-z0-9_.@\-]+", "_",
                               f"{sc.name}_{label}_s{rec.header['seed']}")
        path = rec.save(os.path.join(RUNS, name + ".npz"))
        export.viewer_json(rec, os.path.join(RUNS, name + ".viewer.json"),
                           run=name)
        if a.blender:
            b0, b1 = a.blender
            export.blender_scene(rec, os.path.join(RUNS,
                                                   name + ".scene.json"),
                                 b0, b1)
        print(json.dumps(rec.summary(), indent=1))
        if rec.header.get("error"):
            print(f"the run ended early: {rec.header['reason']}")
        print(f"record {path} ({time.time() - t0:.1f} s wall)")
        return 0 if not rec.header.get("error") else 2
    if a.cmd == "replay":
        from app.record import Record, code_changes, replay
        rec = Record.load(a.record)
        changes = code_changes(rec)
        for c in changes:
            print(f"NOTE: {c}")
        t0 = time.time()
        try:
            rep = replay(rec, start=a.start)
        except ValueError as ex:          # RecordMismatch, a bad --start
            print(f"cannot replay from control step {a.start}: {ex}")
            return 2
        same = rep.fingerprint(physical=True) == rec.fingerprint(
            physical=True)
        diff = rec.diff(rep, physical=True)
        print(f"command replay of {a.record}"
              f"{f' from control step {a.start}' if a.start else ''}: "
              f"physical fingerprint "
              f"{'IDENTICAL' if same else 'DIFFERENT'} "
              f"({time.time() - t0:.1f} s)")
        if not same:
            if changes and not diff:
                print("   every state, command and sensor sample is still "
                      "bit-identical; only the code / hull identity "
                      "differs (above)")
            elif changes:
                print("   the trajectory differs -- expected, since the "
                      "code that made the record changed (above)")
            for k, d in diff[:20]:
                print("  ", k, d)
        return 0 if same else 1
    if a.cmd == "export":
        from app import export
        from app.record import Record, code_changes
        if not (a.viewer or a.blender):
            ap.error("export: give --viewer OUT.json and/or --blender "
                     "SCENE.json (nothing to do otherwise)")
        if a.video and not a.blender:
            ap.error("export: --video renders the --blender scene; give "
                     "--blender SCENE.json too")
        if a.render and not a.video:
            ap.error("export: --render needs --video OUT.mp4")
        rec = Record.load(a.record)
        for c in code_changes(rec):
            print(f"NOTE: {c} -- the export draws the current hull")
        if a.blender:
            # check the window before anything is written
            try:
                export.clip_window(float(rec.header["t_end"]), a.t0, a.t1)
            except ValueError as ex:
                ap.error(f"export: {ex}")
        if a.viewer:
            export.viewer_json(rec, a.viewer)
            print(f"viewer data {a.viewer}")
        if a.blender:
            sc = export.blender_scene(rec, a.blender, a.t0, a.t1, a.fps)
            print(f"Blender scene {a.blender} ({len(sc['frames'])} frames, "
                  f"{sc['meta']['t0']:g}-{sc['meta']['t1']:g} s)")
            if a.video:
                frames_dir = os.path.splitext(a.video)[0] + "_frames"
                cmd = export.blender_command(
                    a.blender, frames_dir, engine=a.engine, res=a.res,
                    samples=a.samples, clip=(0, len(sc["frames"]) - 1),
                    mp4=a.video)
            else:
                cmd = export.blender_command(a.blender,
                                             a.blender[:-5] + ".png",
                                             engine=a.engine, res=a.res,
                                             samples=a.samples)
            print("render with:\n  " + subprocess.list2cmdline(cmd))
            if a.render:
                if not os.path.isfile(cmd[0]) and not export.find_blender():
                    print("Blender not found (set BLENDER to blender.exe)")
                    return 3
                t0 = time.time()
                sys.stdout.flush()
                rc = subprocess.run(cmd).returncode
                print(f"Blender exit {rc} ({time.time() - t0:.0f} s); "
                      f"video {a.video}: "
                      f"{os.path.getsize(a.video) if os.path.exists(a.video) else 'missing'}"
                      f" bytes")
                return rc
        return 0
    if a.cmd == "serve":
        from app.server import serve
        serve(port=a.port, udp_port=a.udp)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
