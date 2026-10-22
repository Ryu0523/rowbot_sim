"""
Optional ROS 2 bridge: the product layer's boat as a ROS 2 node with
LOTUSim's topic names. Written and documented, NOT required here (this
machine has no ROS 2; ACCEPTANCE P6 asks for the module, not an install).
Importing this module never needs rclpy; build_node() does.

WHAT IT DOES
  One node `usv_app_bridge` that owns a UsvEnv (in process, lockstep) and,
  every control step of the scenario (0.24 s for scarab195):
    publishes  /lotusim/poses              lotusim_msgs/VesselPositionArray
                                           (std_msgs/String with the same
                                           JSON when lotusim_msgs is not
                                           built in the workspace)
               /<vessel>/imu               sensor_msgs/Imu
               /<vessel>/gnss              sensor_msgs/NavSatFix
               /<vessel>/odom              nav_msgs/Odometry (ENU / FLU)
               /clock                      rosgraph_msgs/Clock (sim time)
    subscribes /lotusim/vessel_cmd_array   lotusim_msgs/VesselCmdArray (or a
                                           std_msgs/String holding its JSON);
                                           cmd_string keys as app/lotus.py
  then steps the env once with the latest command. With use_sim_time and
  /clock every subscriber sees the simulation's own time, which LOTUSim's
  bridge lacks (LOTUSIM_GAP_2026-10-05.md). Command-line flags (argparse,
  not ROS parameters):
    --rate R            R x real time (default 1); --rate 0 steps as fast
                        as the timer allows (1 ms period)
    --wait-for-command  lockstep: the node steps only when a new command
                        has arrived, one command per step. Without it the
                        command applied at step k is the last one that
                        arrived before it, so results depend on message
                        timing unless the client answers every pose before
                        the next step.
  When the episode ends (terminated or truncated) the node stops its timer
  and logs the reason once; restart it for a new episode.

USE (on a machine with ROS 2 Humble or newer and this repository)
    source /opt/ros/humble/setup.bash
    # optional: build LOTUSim's interfaces for the typed messages
    #   colcon build --packages-select lotusim_msgs lotusim_sensor_msgs
    python -m app.ros2_bridge --scenario demo_scarab195.json \\
        [--vessel usv0] [--rate 1.0] [--wait-for-command]
    ros2 topic echo /lotusim/poses
    ros2 topic pub /lotusim/vessel_cmd_array std_msgs/String \\
        "{data: '{\\"cmds\\": [{\\"vessel_name\\": \\"usv0\\", \\"cmd_string\\": \\"{\\\\\\"throttle\\\\\\": 0.7, \\\\\\"heading(deg)\\\\\\": 0}\\"}]}'}"

ALTERNATIVE WITHOUT ROS ON THIS MACHINE
  Run `python -m app serve` here and the bridge's network mode on the ROS
  machine (--remote ws://HOST:8766/ws): the node then forwards commands and
  poses over the websocket protocol of app/lotus.Session instead of
  stepping a local env (same messages, same lockstep). The app server binds
  127.0.0.1 only, so reach it through a tunnel (e.g. ssh -L 8766:127.0.0.1:
  8766 this-machine) and give --remote ws://127.0.0.1:8766/ws.
"""
import argparse
import json
import threading


def available():
    """True when rclpy imports (a ROS 2 environment is sourced)."""
    try:
        import rclpy  # noqa: F401
        return True
    except ImportError:
        return False


def _typed(module, name):
    try:
        mod = __import__(module, fromlist=[name])
        return getattr(mod, name)
    except (ImportError, AttributeError):
        return None


def to_ros_pose_array(msg_json, VesselPositionArray, now):
    """Fill a lotusim_msgs/VesselPositionArray from app/lotus's JSON."""
    out = VesselPositionArray()
    out.header.stamp = now
    out.header.frame_id = msg_json["header"]["frame_id"]
    VP = type(out.vessels).__args__[0] if hasattr(type(out.vessels),
                                                   "__args__") else None
    for v in msg_json["vessels"]:
        vp = VP() if VP else _typed("lotusim_msgs.msg", "VesselPosition")()
        vp.vessel_name, vp.model_name = v["vessel_name"], v["model_name"]
        p, q = v["pose"]["position"], v["pose"]["orientation"]
        vp.pose.position.x, vp.pose.position.y, vp.pose.position.z = \
            p["x"], p["y"], p["z"]
        (vp.pose.orientation.x, vp.pose.orientation.y,
         vp.pose.orientation.z, vp.pose.orientation.w) = (q["x"], q["y"],
                                                          q["z"], q["w"])
        g = v["geo_point"]
        vp.geo_point.latitude, vp.geo_point.longitude = g["latitude"], \
            g["longitude"]
        vp.geo_point.altitude = g["altitude"]
        out.vessels.append(vp)
    return out


def build_node(scenario="demo_scarab195.json", vessel="usv0", rate=1.0,
               wait_for_command=False, remote=None):
    """The rclpy node (requires a sourced ROS 2 environment)."""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from sensor_msgs.msg import Imu, NavSatFix
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from builtin_interfaces.msg import Time
    from . import lotus

    VPA = _typed("lotusim_msgs.msg", "VesselPositionArray")
    VCA = _typed("lotusim_msgs.msg", "VesselCmdArray")

    class Bridge(Node):
        def __init__(self):
            super().__init__("usv_app_bridge")
            if remote:
                from .wsproto import Client
                from urllib.parse import urlparse
                u = urlparse(remote)
                self.ws = Client(u.hostname, u.port or 8766, u.path or "/ws")
                self.session = None
                rep = self.ws.call({"op": "reset", "scenario": scenario})
                self.dt = rep["info"]["dt_step"]
            else:
                from .env import UsvEnv
                self.ws = None
                self.session = lotus.Session(vessel_name=vessel)
                rep = self.session.handle({"op": "reset",
                                           "scenario": scenario})
                if rep.get("op") == "error":
                    raise RuntimeError(rep["error"])
                self.dt = self.session.env.dt_step
            self.state = rep
            self.cmd = None
            self.lock = threading.Lock()
            self.p_pose = self.create_publisher(VPA or String,
                                                "/lotusim/poses", 10)
            self.p_imu = self.create_publisher(Imu, f"/{vessel}/imu", 10)
            self.p_gnss = self.create_publisher(NavSatFix, f"/{vessel}/gnss",
                                                10)
            self.p_odom = self.create_publisher(Odometry, f"/{vessel}/odom",
                                                10)
            self.p_clock = self.create_publisher(Clock, "/clock", 10)
            self.create_subscription(VCA or String,
                                     "/lotusim/vessel_cmd_array",
                                     self.on_cmd, 10)
            period = self.dt / rate if rate > 0 else 0.001
            self.timer = self.create_timer(period, self.tick)
            self.publish(self.state)

        def on_cmd(self, msg):
            if VCA is not None and not isinstance(msg, String):
                d = dict(cmds=[dict(vessel_name=c.vessel_name,
                                    cmd_string=c.cmd_string)
                               for c in msg.cmds])
            else:
                d = json.loads(msg.data)
            with self.lock:
                self.cmd = d

        def tick(self):
            if self.state.get("terminated") or self.state.get("truncated"):
                # the episode is over: stop stepping (logged once)
                self.timer.cancel()
                self.get_logger().info(
                    f"episode ended ({self.state.get('reason')}); timer "
                    f"stopped -- restart the node for a new episode")
                return
            with self.lock:
                cmd, self.cmd = self.cmd, (None if wait_for_command
                                           else self.cmd)
            if cmd is None:
                return              # nothing (new) commanded yet
            msg = dict(op="step", **cmd)
            rep = (self.ws.call(msg) if self.ws else
                   self.session.handle(msg))
            if rep.get("op") == "error":
                self.get_logger().error(rep["error"])
                return
            self.state = rep
            self.publish(rep)

        def _stamp(self, t):
            # app/lotus.stamp carries a rounded-up nanosecond into sec
            st = lotus.stamp(t)
            s = Time()
            s.sec, s.nanosec = st["sec"], st["nanosec"]
            return s

        def publish(self, rep):
            t = float(rep["t"])
            now = self._stamp(t)
            c = Clock()
            c.clock = now
            self.p_clock.publish(c)
            poses = rep["poses"]          # exactly {header, vessels}
            if VPA is not None:
                self.p_pose.publish(to_ros_pose_array(poses, VPA, now))
            else:
                self.p_pose.publish(String(data=json.dumps(poses)))
            v = poses["vessels"][0]
            ext = rep["usv"]              # our extras: the reply's sibling
            od = Odometry()
            od.header.stamp, od.header.frame_id = now, "enu"
            od.child_frame_id = f"{vessel}/base_link"
            p, q = v["pose"]["position"], v["pose"]["orientation"]
            od.pose.pose.position.x, od.pose.pose.position.y, \
                od.pose.pose.position.z = p["x"], p["y"], p["z"]
            od.pose.pose.orientation.x, od.pose.pose.orientation.y, \
                od.pose.pose.orientation.z, od.pose.pose.orientation.w = \
                q["x"], q["y"], q["z"], q["w"]
            tw = ext["twist"]
            od.twist.twist.linear.x, od.twist.twist.linear.y, \
                od.twist.twist.linear.z = (tw["linear"]["x"],
                                           tw["linear"]["y"],
                                           tw["linear"]["z"])
            od.twist.twist.angular.x, od.twist.twist.angular.y, \
                od.twist.twist.angular.z = (tw["angular"]["x"],
                                            tw["angular"]["y"],
                                            tw["angular"]["z"])
            self.p_odom.publish(od)
            g = rep.get("gps")
            if g:
                n = NavSatFix()
                n.header.stamp, n.header.frame_id = self._stamp(
                    g["header"]["stamp"]["sec"]
                    + 1e-9 * g["header"]["stamp"]["nanosec"]), "gnss"
                n.status.status = g["status"]["status"]
                n.status.service = g["status"]["service"]
                n.latitude, n.longitude, n.altitude = g["latitude"], \
                    g["longitude"], g["altitude"]
                n.position_covariance = [float(x) for x in
                                         g["position_covariance"]]
                n.position_covariance_type = g["position_covariance_type"]
                self.p_gnss.publish(n)
            m = rep.get("imu")
            if m:
                i = Imu()
                i.header.stamp, i.header.frame_id = now, f"{vessel}/imu"
                o, w, a = m["orientation"], m["angular_velocity"], \
                    m["linear_acceleration"]
                i.orientation.x, i.orientation.y, i.orientation.z, \
                    i.orientation.w = o["x"], o["y"], o["z"], o["w"]
                i.angular_velocity.x, i.angular_velocity.y, \
                    i.angular_velocity.z = w["x"], w["y"], w["z"]
                i.linear_acceleration.x, i.linear_acceleration.y, \
                    i.linear_acceleration.z = a["x"], a["y"], a["z"]
                i.orientation_covariance = [float(x) for x in
                                            m["orientation_covariance"]]
                i.angular_velocity_covariance = [
                    float(x) for x in m["angular_velocity_covariance"]]
                i.linear_acceleration_covariance = [
                    float(x) for x in m["linear_acceleration_covariance"]]
                self.p_imu.publish(i)

    return Bridge()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m app.ros2_bridge")
    ap.add_argument("--scenario", default="demo_scarab195.json")
    ap.add_argument("--vessel", default="usv0")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--wait-for-command", action="store_true")
    ap.add_argument("--remote", help="ws://HOST:8766/ws of an app server")
    a = ap.parse_args(argv)
    if not available():
        raise SystemExit("rclpy is not importable: source a ROS 2 "
                         "environment first (see this module's docstring)")
    import rclpy
    rclpy.init()
    node = build_node(a.scenario, a.vessel, a.rate, a.wait_for_command,
                      a.remote)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
