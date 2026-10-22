"""End-to-end check of the LOTUSim Gymnasium environment against a running
LOTUSim (Gazebo + ROS 2) with an xdyn physics server for surface vessels.

Start the simulation first (two terminals, see lotusim_setup/SETUP.txt or just
`bash lotusim_setup/run_sim.sh`), then run this inside the Python dev shell:

    nix develop path:control_envs/lotusim_setup --command python3 control_envs/lotusim_gym_demo.py

What it does:
  1. spawns a DTMB hull (twin propeller+rudder, physics by xdyn on port 12345)
  2. wraps it in LotusimEnv (obs: speed, yaw rate, heading error; action: throttle, rudder)
  3. drives it for --seconds with a simple P controller and prints the speed
  4. deletes the vessel on exit
The vessel is a 142 m destroyer model (the only surface model shipped with
controllable xdyn forces), so it accelerates slowly; a USV model with its own
xdyn YAML plugs in through --model/--yaml/--thrusters.
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from control_envs.lotusim_env import (LotusimEnv, LotusimLink, surface_xdyn_param,
                                      twin_screw_cmd)

# Same spawn point as LOTUSim's own python examples (Singapore strait).
SPAWN_LAT, SPAWN_LON = 1.2605794416293148, 103.7516212463379


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dtmb_hull", help="folder name under assets/models")
    ap.add_argument("--name", default="gym_dtmb", help="vessel name to request")
    ap.add_argument("--thrusters", default="PSPropRudd,SBPropRudd",
                    help="controlled-force names from the vessel's xdyn YAML")
    ap.add_argument("--port", type=int, default=12345, help="xdyn websocket port (surface)")
    ap.add_argument("--rpm-max", type=float, default=30.0, help="shaft speed at throttle 1 (xdyn units)")
    ap.add_argument("--target-speed", type=float, default=3.0, help="m/s")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=0.5, help="control step (s)")
    ap.add_argument("--keep", action="store_true", help="do not delete the vessel at the end")
    args = ap.parse_args()

    thrusters = tuple(t.strip() for t in args.thrusters.split(","))
    link = LotusimLink(args.name)
    print("spawning", args.model, "...")
    name = link.spawn(args.model, SPAWN_LAT, SPAWN_LON, 0.0,
                      lotus_param=surface_xdyn_param(args.port, thrusters))
    print("vessel name:", name)
    time.sleep(1.0)
    pose = link.wait_pose(-1.0, timeout=20.0)
    print("first pose: t=%.2f x=%.1f y=%.1f yaw=%.2f" % pose)

    env = LotusimEnv(link, dt=args.dt, episode_s=args.seconds, target_speed=args.target_speed,
                     target_heading=pose[3], cmd_fn=twin_screw_cmd(thrusters, rpm_max=args.rpm_max),
                     reset_pose=(pose[1], pose[2]))
    try:
        obs, _ = env.reset(seed=0)
        total, t0 = 0.0, time.time()
        for k in range(env.max_steps):
            throttle = float(np.clip(0.5 + 0.3 * (args.target_speed - obs[0]), 0.0, 1.0))
            rudder = float(np.clip(-1.0 * math.atan2(obs[2], obs[3]), -1.0, 1.0))
            obs, r, term, trunc, info = env.step([throttle, rudder])
            total += r
            if k % max(1, int(5.0 / args.dt)) == 0:
                print("t=%5.1fs speed=%5.2f m/s yaw_rate=%6.3f heading_err=%6.3f throttle=%.2f rudder=%.2f"
                      % (k * args.dt, obs[0], obs[1], math.atan2(obs[2], obs[3]), throttle, rudder))
            if term or trunc:
                break
        print("done: %d steps in %.1f s wall, return %.1f, final speed %.2f m/s (target %.2f)"
              % (k + 1, time.time() - t0, total, obs[0], args.target_speed))
    finally:
        if not args.keep:
            print("deleting vessel")
            link.delete()


if __name__ == "__main__":
    main()
