"""Which way, and in which unit, does the DTMB's rudder work in LOTUSim?

Spawns a DTMB hull, holds a steady throttle, applies the rudder command
+BETA for --seconds then -BETA, and prints the yaw rate. Run twice: with
BETA = 20 (degrees if LOTUSim passes the value in the YAML's unit) and
BETA = 0.35 (radians if it passes SI). The one that gives a steady, opposite
yaw rate for the two signs is the right unit.

    nix develop path:control_envs/lotusim_setup --command python3 -u \
        control_envs/lotusim_rudder_test.py --beta 20 --seconds 40
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control_envs.lotusim_env import LotusimLink, surface_xdyn_param   # noqa: E402

SPAWN_LAT, SPAWN_LON = 1.2605794416293148, 103.7516212463379


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=20.0, help="rudder command value sent as-is")
    ap.add_argument("--rpm", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=40.0)
    args = ap.parse_args()
    thr = ("PSPropRudd", "SBPropRudd")
    link = LotusimLink("rudder_test")
    name = link.spawn("dtmb_hull", SPAWN_LAT, SPAWN_LON, 0.0,
                      lotus_param=surface_xdyn_param(12345, thr))
    print("vessel:", name, flush=True)
    time.sleep(3.0)

    def cmd(beta):
        c = {}
        for t in thr:
            c[f"{t}(rpm)"] = args.rpm
            c[f"{t}(P/D)"] = 0.79
            c[f"{t}(beta)"] = beta
        return c

    pose = link.wait_pose(-1.0, timeout=20.0)
    pose = link.wait_pose(pose[0] + 1.0, timeout=20.0)
    try:
        # first get way on with the rudder centred
        t0 = pose[0]
        while pose[0] - t0 < 20.0:
            link.send(cmd(0.0))
            pose = link.wait_pose(pose[0] + 1.0 - 1e-9, timeout=10.0)
        for beta in (+args.beta, -args.beta):
            t0, yaw0 = pose[0], pose[3]
            while pose[0] - t0 < args.seconds:
                link.send(cmd(beta))
                prev = pose
                pose = link.wait_pose(pose[0] + 1.0 - 1e-9, timeout=10.0)
                r = math.remainder(pose[3] - prev[3], 2 * math.pi) / max(pose[0] - prev[0], 1e-6)
                k = int(round(pose[0] - t0))
                if k % 5 == 0:
                    sp = math.hypot(pose[1] - prev[1], pose[2] - prev[2]) / max(pose[0] - prev[0], 1e-6)
                    print(f"beta {beta:+6.2f}: t={pose[0] - t0:5.1f}s speed {sp:4.2f} m/s "
                          f"yaw={math.degrees(pose[3]):7.1f} deg  yaw rate {math.degrees(r):+6.2f} deg/s",
                          flush=True)
            print(f"beta {beta:+6.2f}: yaw change over {args.seconds:.0f} s = "
                  f"{math.degrees(math.remainder(pose[3] - yaw0, 2 * math.pi)):+.1f} deg", flush=True)
    finally:
        link.send(cmd(0.0))
        link.delete()


if __name__ == "__main__":
    main()
