"""Waypoint-following demo for the LOTUSim 3-D environment (for a video).

A vessel is spawned in LOTUSim and steered round a closed course (a square of
side --side metres by default, or a figure of eight) by a simple two-loop
controller written in a few lines:

    speed loop     throttle = k_p (v_ref - v) + k_i * integral     (PI)
    heading loop   rudder   = k_h * heading_error + k_r * yaw_rate  (P + rate)
    guidance       heading reference = bearing from the vessel to the next
                   waypoint; the next waypoint becomes active when the vessel
                   is within --accept metres of the current one, or has
                   passed abeam of it (the ship's turning circle is wide)

Everything the video needs is printed and logged: the active waypoint, speed,
heading error, rudder; the track is saved to a JSON file for a plot.

Start the simulation first (lotusim_setup/run_sim.sh --gui), then:

    nix develop path:control_envs/lotusim_setup --command \
        python3 control_envs/lotusim_waypoint_demo.py --side 300 --target-speed 4

Without LOTUSim (any machine):   python control_envs/lotusim_waypoint_demo.py --fake
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from control_envs.lotusim_env import (FakeLink, surface_xdyn_param,
                                      single_propeller_cmd, twin_screw_cmd)

SPAWN_LAT, SPAWN_LON = 1.2605794416293148, 103.7516212463379


def course(kind, side, x0, y0, yaw0):
    """Waypoints (x, y) in LOTUSim's world frame, starting ahead of the
    vessel: a square, or a figure of eight made of two squares."""
    c, s = math.cos(yaw0), math.sin(yaw0)

    def local(dx, dy):              # forward dx, left dy -> world
        return (x0 + dx * c - dy * s, y0 + dx * s + dy * c)
    if kind == "square":
        pts = [(side, 0), (side, side), (0, side), (0, 0)]
    elif kind == "eight":
        pts = [(side, 0), (side, side), (0, side), (0, 0),
               (side, 0), (side, -side), (0, -side), (0, 0)]
    else:
        raise ValueError(kind)
    return [local(*p) for p in pts]


def wrap(a):
    return math.remainder(a, 2 * math.pi)


class Controller:
    """Speed PI + heading P-with-rate, both with limits (module docstring)."""

    def __init__(self, dt, k_p=0.3, k_i=0.05, k_h=1.2, k_r=0.8, thr_min=0.1):
        self.dt, self.k_p, self.k_i, self.k_h, self.k_r = dt, k_p, k_i, k_h, k_r
        self.thr_min = thr_min
        self.i_err = 0.0

    def __call__(self, v_ref, v, heading_err, yaw_rate):
        self.i_err = float(np.clip(self.i_err + (v_ref - v) * self.dt, -5, 5))
        throttle = float(np.clip(self.thr_min + self.k_p * (v_ref - v)
                                 + self.k_i * self.i_err, 0.0, 1.0))
        rudder = float(np.clip(-self.k_h * heading_err - self.k_r * yaw_rate,
                               -1.0, 1.0))
        return throttle, rudder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="toy boat, no LOTUSim")
    ap.add_argument("--course", default="square", choices=("square", "eight"))
    ap.add_argument("--side", type=float, default=300.0, help="metres")
    ap.add_argument("--accept", type=float, default=40.0,
                    help="waypoint reached within this distance (m)")
    ap.add_argument("--target-speed", type=float, default=4.0, help="m/s")
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--dt", type=float, default=0.5, help="control step (s)")
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--model", default="dtmb_hull")
    ap.add_argument("--thrusters", default="PSPropRudd,SBPropRudd")
    ap.add_argument("--port", type=int, default=12345)
    ap.add_argument("--rpm-max", type=float, default=80.0)
    ap.add_argument("--rudder-sign", type=float, default=1.0,
                    help="+1 (positive rudder turns to port, as measured on"
                         " the DTMB and on the toy boat); -1 to flip")
    ap.add_argument("--log", default="control_envs/_out/lotusim_track.json")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    sign = args.rudder_sign
    if args.fake:
        link = FakeLink(dt=0.05)
        cmd_fn = single_propeller_cmd()
        name = "fake_boat"
        args.side = min(args.side, 60.0)          # the toy boat does ~3 m/s
        args.accept = min(args.accept, 8.0)
        args.target_speed = min(args.target_speed, 2.5)
    else:
        from control_envs.lotusim_env import LotusimLink
        thrusters = tuple(t.strip() for t in args.thrusters.split(","))
        link = LotusimLink("wp_demo")
        print("spawning", args.model, "...")
        name = link.spawn(args.model, SPAWN_LAT, SPAWN_LON, 0.0,
                          lotus_param=surface_xdyn_param(args.port, thrusters))
        print("vessel name:", name)
        cmd_fn = twin_screw_cmd(thrusters, rpm_max=args.rpm_max)
        time.sleep(1.0)

    pose = link.wait_pose(-1.0, timeout=20.0)
    if not args.fake:
        # the first poses after a spawn can be the placement before xdyn
        # takes over; wait for the vessel to settle before laying the course
        time.sleep(3.0)
        pose = link.wait_pose(pose[0] + 2.0, timeout=20.0)
    t, x, y, yaw = pose
    wps = course(args.course, args.side, x, y, yaw) * args.laps
    x0_, y0_ = x, y
    print(f"start x={x:.1f} y={y:.1f} yaw={math.degrees(yaw):.0f} deg; "
          f"{len(wps)} waypoints, side {args.side:g} m")
    ctrl = Controller(args.dt)
    log = []
    k_wp, t_start = 0, t
    prev = pose
    n_steps = int(args.seconds / args.dt)
    try:
        for k in range(n_steps):
            t, x, y, yaw = pose
            tx, ty = wps[k_wp]
            dist = math.hypot(tx - x, ty - y)
            # the waypoint counts as reached inside the acceptance circle,
            # or once the vessel has passed abeam of it (the leg from the
            # previous waypoint projects negative): a ship whose turning
            # circle is wider than the course would otherwise orbit it
            px, py = wps[k_wp - 1] if k_wp > 0 else (x0_, y0_)
            lx, ly = tx - px, ty - py
            ll = math.hypot(lx, ly) or 1.0
            along = ((tx - x) * lx + (ty - y) * ly) / ll
            if dist < args.accept or along < 0.0:
                k_wp += 1
                if k_wp >= len(wps):
                    print(f"course complete at t={t - t_start:.0f} s")
                    break
                tx, ty = wps[k_wp]
                dist = math.hypot(tx - x, ty - y)
            bearing = math.atan2(ty - y, tx - x)
            h_err = wrap(yaw - bearing)
            dt_m = max(t - prev[0], 1e-6)
            speed = math.hypot(x - prev[1], y - prev[2]) / dt_m
            yaw_rate = wrap(yaw - prev[3]) / dt_m
            throttle, rudder = ctrl(args.target_speed, speed, h_err, yaw_rate)
            link.send(cmd_fn(throttle, sign * rudder))
            log.append(dict(t=t - t_start, x=x, y=y, yaw=yaw, speed=speed,
                            wp=k_wp, dist=dist, h_err=h_err,
                            throttle=throttle, rudder=rudder))
            if k % max(1, int(5.0 / args.dt)) == 0:
                print(f"t={t - t_start:6.1f}s wp {k_wp + 1}/{len(wps)} dist "
                      f"{dist:6.1f} m speed {speed:4.2f} m/s heading err "
                      f"{math.degrees(h_err):6.1f} deg throttle {throttle:.2f} "
                      f"rudder {rudder:+.2f}")
            prev = pose
            pose = link.wait_pose(t + args.dt - 1e-9)
    finally:
        link.send(cmd_fn(0.0, 0.0))
        out = Path(args.log)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(dict(vessel=name, course=args.course, side=args.side,
                           waypoints=wps, target_speed=args.target_speed,
                           log=log), f)
        print(f"track saved: {out} ({len(log)} rows)")
        if not args.fake and not args.keep:
            print("deleting vessel")
            link.delete()


if __name__ == "__main__":
    main()
