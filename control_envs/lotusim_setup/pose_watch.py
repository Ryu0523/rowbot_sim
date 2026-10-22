"""Print every vessel's x, y, yaw and the derived speed / yaw rate once a
second for --seconds, from /lotusim/poses. Run inside the dev shell."""
import argparse
import math
import time

import rclpy
from lotusim_msgs.msg import VesselPositionArray


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()
    rclpy.init()
    node = rclpy.create_node("pose_watch")
    last = {}

    def cb(msg):
        for v in msg.vessels:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            p = v.pose.position
            yaw = yaw_of(v.pose.orientation)
            prev = last.get(v.vessel_name)
            if prev is None or t - prev[0] >= 1.0:
                if prev is not None:
                    dt = t - prev[0]
                    sp = math.hypot(p.x - prev[1], p.y - prev[2]) / dt
                    r = math.remainder(yaw - prev[3], 2 * math.pi) / dt
                    print(f"{v.vessel_name}: t={t:9.1f} x={p.x:9.1f} y={p.y:9.1f} yaw={math.degrees(yaw):7.1f} deg "
                          f"speed={sp:4.2f} m/s yaw rate={math.degrees(r):+5.2f} deg/s", flush=True)
                last[v.vessel_name] = (t, p.x, p.y, yaw)

    node.create_subscription(VesselPositionArray, "/lotusim/poses", cb, 10)
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
