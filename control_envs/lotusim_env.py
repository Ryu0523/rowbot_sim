"""
A minimal Gymnasium environment for one vessel in LOTUSim.

LOTUSim runs its own simulation loop (Gazebo + the xdyn physics server) and
talks over ROS 2. This env only does three things:
  - send the action as a JSON command on   /lotusim/vessel_cmd_array
  - read the vessel pose from              /lotusim/poses
  - teleport the vessel back on reset via  /lotusim/mas_cmd  (MOVE command)
Topic, message and command names are taken from LOTUSim's own example
examples/python-scripts/controlling_ships.py and its msg definitions.

Requirements: a running LOTUSim (Linux / WSL2, ROS 2 sourced) with the vessel
already spawned, and `lotusim_msgs` on the Python path. Without ROS 2 you can
still run the env with FakeLink below (a toy boat), which is how this file is
tested on Windows.

Usage:
    env = LotusimEnv(LotusimLink("my_boat"))      # real LOTUSim
    env = LotusimEnv(FakeLink())                  # no ROS needed
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step([0.5, 0.0])
"""
import json
import math
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# --------------------------------------------------------------------------
# 1. The link to LOTUSim: send a command, read a pose, move the vessel.
#    Kept separate from the env so the env can be tested without ROS.
# --------------------------------------------------------------------------
class LotusimLink:
    """ROS 2 connection to one LOTUSim vessel."""

    def __init__(self, vessel_name, namespace="/lotusim"):
        import rclpy                                  # only needed here
        from rclpy.node import Node
        from rclpy.action import ActionClient
        from lotusim_msgs.msg import VesselPositionArray
        from lotusim_msgs.action import MASCmd

        if not rclpy.ok():
            rclpy.init()
        self.rclpy = rclpy
        self.vessel = vessel_name
        self.node = Node("gym_lotusim_" + vessel_name)
        self.pose = None                              # (t, x, y, yaw) latest
        self.node.create_subscription(
            VesselPositionArray, namespace + "/poses", self._on_poses, 10)
        from lotusim_msgs.msg import VesselCmdArray
        self.cmd_pub = self.node.create_publisher(
            VesselCmdArray, namespace + "/vessel_cmd_array", 10)
        self.mas = ActionClient(self.node, MASCmd, namespace + "/mas_cmd")

    def _on_poses(self, msg):
        for v in msg.vessels:
            if v.vessel_name == self.vessel:
                t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
                q = v.pose.orientation                # front-left-up frame
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y * q.y + q.z * q.z))
                self.pose = (t, v.pose.position.x, v.pose.position.y, yaw)

    def send(self, cmd: dict):
        from lotusim_msgs.msg import VesselCmd, VesselCmdArray
        c = VesselCmd()
        c.vessel_name = self.vessel
        c.cmd_string = json.dumps(cmd)                # e.g. {"propeller(rpm)": 300}
        arr = VesselCmdArray()
        arr.cmds.append(c)
        self.cmd_pub.publish(arr)

    def wait_pose(self, after_t, timeout=5.0):
        """Spin until a pose newer than `after_t` (sim time) arrives."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.rclpy.spin_once(self.node, timeout_sec=0.01)
            if self.pose is not None and self.pose[0] > after_t:
                return self.pose
        raise TimeoutError("no pose from LOTUSim for " + self.vessel)

    def move(self, x, y, yaw):
        """Teleport the vessel (LOTUSim MASCmd.MOVE_CMD)."""
        from lotusim_msgs.msg import MASCmd as MASCmdMsg
        m = MASCmdMsg()
        m.cmd_type = MASCmdMsg.MOVE_CMD
        m.vessel_name = self.vessel
        m.vessel_position.position.x, m.vessel_position.position.y = x, y
        m.vessel_position.orientation.z = math.sin(yaw / 2)
        m.vessel_position.orientation.w = math.cos(yaw / 2)
        self._mas_goal(m)

    def _mas_goal(self, m, timeout=10.0):
        """Send one MASCmd goal and wait for its result; returns the result msg."""
        from lotusim_msgs.action import MASCmd
        goal = MASCmd.Goal()
        goal.cmd = m
        if not self.mas.wait_for_server(timeout_sec=timeout):
            raise TimeoutError("LOTUSim action server /lotusim/mas_cmd not found")
        fut = self.mas.send_goal_async(goal)
        self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout)
        handle = fut.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("LOTUSim rejected the command")
        res = handle.get_result_async()
        self.rclpy.spin_until_future_complete(self.node, res, timeout_sec=timeout)
        return res.result().result if res.result() is not None else None

    def spawn(self, model_name, lat, lon, alt=0.0, lotus_param=None, heading=0.0):
        """Create a vessel from a model in LOTUSim's assets (MASCmd.CREATE_CMD).

        `lotus_param` is the <lotus_param> XML string that tells LOTUSim how the
        vessel is driven (see SURFACE_XDYN_PARAM below). Returns the name LOTUSim
        gave the vessel (it appends an index) and makes it this link's vessel.
        """
        from geographic_msgs.msg import GeoPoint
        from lotusim_msgs.msg import MASCmd as MASCmdMsg
        m = MASCmdMsg()
        m.cmd_type = MASCmdMsg.CREATE_CMD
        m.model_name = model_name
        m.vessel_name = self.vessel
        g = GeoPoint()
        g.latitude, g.longitude, g.altitude = float(lat), float(lon), float(alt)
        m.geo_point = g
        m.heading = float(heading)
        m.sdf_string = lotus_param or ""
        result = self._mas_goal(m, timeout=30.0)
        name = getattr(result, "name", "") if result is not None else ""
        if name:
            self.vessel = name
        return self.vessel

    def delete(self):
        """Remove this link's vessel from the simulation (MASCmd.DELETE_CMD)."""
        from lotusim_msgs.msg import MASCmd as MASCmdMsg
        m = MASCmdMsg()
        m.cmd_type = MASCmdMsg.DELETE_CMD
        m.vessel_name = self.vessel
        self._mas_goal(m)


def surface_xdyn_param(port=12345, thrusters=("PSPropRudd", "SBPropRudd"), renderer=None):
    """<lotus_param> for a surface vessel whose physics is computed by an xdyn
    server listening on ws://127.0.0.1:<port> (start it with xdyn-for-cs, see
    lotusim_setup/run_sim.sh). `thrusters` are the controlled-force names in the
    vessel's xdyn YAML; the command JSON keys are '<name>(rpm)', '<name>(P/D)',
    '<name>(beta)'."""
    th = "".join(f"<thrusters{i + 1}>{t}</thrusters{i + 1}>" for i, t in enumerate(thrusters))
    render = (f"<render_interface><publish_render>true</publish_render>"
              f"<renderer_type_name>{renderer}</renderer_type_name></render_interface>"
              if renderer else "")
    return f"""<lotus_param>{render}
  <physics_engine_interface>
    <surface>
      <interface_type>XDynWebSocket</interface_type>
      <uri>ws://127.0.0.1:{port}</uri>
      <thrusters>{th}</thrusters>
    </surface>
    <init_state>Surface</init_state>
  </physics_engine_interface>
</lotus_param>"""


def twin_screw_cmd(thrusters=("PSPropRudd", "SBPropRudd"), rpm_max=30.0, beta_max_deg=20.0, pd=0.79,
                   diff=0.6):
    """Action -> LOTUSim command dict for a twin propeller-rudder xdyn vessel
    (e.g. assets/models/dtmb_hull/dtmb-xdyn.yml). throttle in [0,1] sets both
    shaft speeds (xdyn's 'rpm' is in rad/s here), rudder in [-1,1] sets both
    rudder angles, +-beta_max_deg.

    UNITS (measured 2026-10-21, control_envs/lotusim_rudder_test.py): the
    values in the command JSON reach xdyn in SI, whatever unit the YAML's
    own `commands:` block declares -- beta is RADIANS (0.35 gave a steady
    +1 deg/s yaw rate on the DTMB, -0.35 the opposite), rpm is rad/s.
    Positive beta turns the ship to port (+yaw). At 35 deg (0.61 rad) the
    DTMB did not turn at all (the rudder model stalls); 20 deg gives a steady
    1-1.5 deg/s, hence the default."""
    beta_max = math.radians(beta_max_deg)
    # `diff`: differential thrust on top of the rudder. The DTMB's steady
    # turning circle on rudder alone is about 700 m in radius at 3.7 m/s
    # (yaw rate 0.3 deg/s after the initial transient, measured 2026-10-21),
    # too wide for a short course; speeding up the outer shaft and slowing
    # the inner one by diff x rudder tightens it. thrusters[0] is taken as
    # the port shaft, thrusters[1] as starboard; rudder > 0 turns to port.

    def f(throttle, rudder):
        cmd = {}
        n = float(throttle) * rpm_max
        for i, t in enumerate(thrusters):
            side = -1.0 if i == 0 else 1.0           # port slows, starboard speeds up
            cmd[f"{t}(rpm)"] = max(0.0, n * (1.0 + side * diff * float(rudder)))
            cmd[f"{t}(P/D)"] = pd
            cmd[f"{t}(beta)"] = float(rudder) * beta_max
        return cmd
    return f


def single_propeller_cmd(max_rpm=1000.0, max_rudder=0.5):
    """The original toy mapping: one propeller plus a rudder angle."""
    def f(throttle, rudder):
        return {"propeller(rpm)": float(throttle) * max_rpm,
                "rudder(angle)": float(rudder) * max_rudder}
    return f


class FakeLink:
    """Stand-in for LOTUSim without ROS: a 3-DOF toy boat, same interface."""

    def __init__(self, dt=0.05):
        self.dt = dt
        self.move(0.0, 0.0, 0.0)
        self.thrust, self.rudder = 0.0, 0.0

    def send(self, cmd):
        self.thrust = cmd.get("propeller(rpm)", 0.0) / 1000.0   # toy scaling
        self.rudder = cmd.get("rudder(angle)", 0.0)

    def wait_pose(self, after_t, timeout=5.0):
        while self.t <= after_t:                      # integrate until newer
            u, r = self.u, self.r
            self.u += self.dt * (2.0 * self.thrust - 0.2 * u * abs(u))
            self.r += self.dt * (0.5 * u * self.rudder - 1.0 * r)
            self.yaw += self.dt * r
            self.x += self.dt * u * math.cos(self.yaw)
            self.y += self.dt * u * math.sin(self.yaw)
            self.t += self.dt
        return (self.t, self.x, self.y, self.yaw)

    def move(self, x, y, yaw):
        self.t, self.x, self.y, self.yaw, self.u, self.r = 0.0, x, y, yaw, 0.0, 0.0


# --------------------------------------------------------------------------
# 2. The Gymnasium env. Task: hold a target speed along a target heading.
# --------------------------------------------------------------------------
class LotusimEnv(gym.Env):
    def __init__(self, link, dt=0.25, episode_s=60.0, target_speed=5.0,
                 target_heading=0.0, max_rpm=1000.0, max_rudder=0.5, cmd_fn=None,
                 reset_pose=(0.0, 0.0)):
        self.link, self.dt = link, dt
        self.max_steps = int(episode_s / dt)
        self.target_speed, self.target_heading = target_speed, target_heading
        self.max_rpm, self.max_rudder = max_rpm, max_rudder
        # action -> command dict; default is the single-propeller toy mapping
        self.cmd_fn = cmd_fn or single_propeller_cmd(max_rpm, max_rudder)
        self.reset_pose = reset_pose
        # action: [throttle 0..1, rudder -1..1]
        self.action_space = spaces.Box(np.array([0.0, -1.0], np.float32),
                                       np.array([1.0, 1.0], np.float32))
        # obs: [speed, yaw rate, heading error (sin, cos)]
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)

    def _obs(self):
        # LOTUSim publishes poses only, so speeds come from two poses
        t0, x0, y0, yaw0 = self.prev
        t1, x1, y1, yaw1 = self.pose
        h = max(t1 - t0, 1e-6)
        speed = math.hypot(x1 - x0, y1 - y0) / h
        yaw_rate = math.remainder(yaw1 - yaw0, 2 * math.pi) / h
        err = yaw1 - self.target_heading
        return np.array([speed, yaw_rate, math.sin(err), math.cos(err)],
                        np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.link.send(self.cmd_fn(0.0, 0.0))
        self.link.move(self.reset_pose[0], self.reset_pose[1], self.target_heading)
        self.prev = self.link.wait_pose(-1.0)
        self.pose = self.link.wait_pose(self.prev[0])
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, float), self.action_space.low,
                    self.action_space.high)
        self.link.send(self.cmd_fn(float(a[0]), float(a[1])))
        self.prev = self.pose
        self.pose = self.link.wait_pose(self.prev[0] + self.dt - 1e-9)
        obs = self._obs()
        speed_err = obs[0] - self.target_speed
        heading_err = math.atan2(obs[2], obs[3])
        reward = -speed_err ** 2 - 4.0 * heading_err ** 2 - 0.01 * a[1] ** 2
        self.steps += 1
        truncated = self.steps >= self.max_steps
        return obs, float(reward), False, truncated, {"pose": self.pose}


if __name__ == "__main__":
    # Smoke test without ROS: Gymnasium's checker + a simple P controller.
    from gymnasium.utils.env_checker import check_env
    env = LotusimEnv(FakeLink(), target_speed=2.5)   # toy boat tops out ~3 m/s
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(env.max_steps):
        throttle = np.clip(0.5 + 0.2 * (env.target_speed - obs[0]), 0, 1)
        rudder = np.clip(-2.0 * math.atan2(obs[2], obs[3]), -1, 1)
        obs, r, term, trunc, info = env.step([throttle, rudder])
        total += r
        if term or trunc:
            break
    print(f"speed {obs[0]:.2f} m/s (target {env.target_speed}), "
          f"return {total:.1f}")
