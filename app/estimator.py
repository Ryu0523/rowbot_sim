"""
A small state estimator from the sensors: what the built-in policies use
with state_source="sensors" (app/policies.py) instead of the plant's true
state, so that the scenario's sensor noise, bias, latency and dropout reach
the closed loop.

It is deliberately simple -- no filter, no model -- so that what a policy
sees is easy to say. The estimate is a full 14-number plant state vector
(sim/planing_vessel's layout), built per channel from the LAST VALID
delivered reading of each sensor, extrapolated over that reading's age
(latency plus sample hold) where the same sensors measure the rate:

  x, y      GNSS ENU fix minus the antenna lever arm (rotated by the
            estimated attitude), plus the GNSS ENU velocity x age
  z, w      GNSS height minus the lever arm's height minus Z_rest; GNSS
            vertical velocity
  roll      IMU attitude
  pitch     IMU attitude minus the rest trim Th_rest (the state's pitch is
            measured from rest)
  yaw       heading sensor + yaw rate x age (the IMU's attitude yaw when
            the scenario has no heading sensor)
  u         speed log (through the water; equal to the speed over ground
            along the hull: the plant has no current)
  v         GNSS ENU velocity rotated into the estimated heading
  roll, pitch, yaw rates   the IMU gyro's body rates through the inverse of
            the ZYX kinematics the IMU model uses (app/geo.
            euler_rates_to_body): the plant's own rate states
  thrust, nozzle  the actuators' states, taken as measured feedback (jet
            rpm and nozzle angle sensors, assumed exact)

Before a sensor's first valid reading its channels keep the episode's
initial state (the scenario's start, which the operator knows). Required
sensors: gnss, imu, speed_log; heading is optional.
"""
import math

import numpy as np

from . import geo

REQUIRED = ("gnss", "imu", "speed_log")


def body_to_euler_rates(roll, pitch, w):
    """Inverse of geo.euler_rates_to_body: ZYX Euler-angle rates from the
    body angular velocity (pitch away from +-90 deg)."""
    E = np.column_stack([geo.euler_rates_to_body(roll, pitch, *e)
                         for e in np.eye(3)])
    return np.linalg.solve(E, np.asarray(w, float))


class SensorEstimator:
    def __init__(self, env):
        missing = [n for n in REQUIRED if n not in env.sensors]
        if missing:
            raise ValueError(f"state_source 'sensors' needs the sensors "
                             f"{list(REQUIRED)}; the scenario disables "
                             f"{missing}")
        self.env = env
        self.s = np.array(env.s0, float)
        self.n_updates = 0

    def update(self):
        """The estimate at env.t (a new array)."""
        env, s = self.env, self.s.copy()
        t, p = env.t, env.plant
        sn = env.sensors
        imu = sn["imu"].last_valid
        if imu is not None:
            tm, v = imu
            s[3] = v[6]
            s[4] = v[7] - p.Th_rest
            s[9:12] = body_to_euler_rates(v[6], v[7], v[3:6])
            if "heading" not in sn:
                s[5] = v[8] + s[11] * (t - tm)
        hd = sn["heading"].last_valid if "heading" in sn else None
        if hd is not None:
            tm, v = hd
            s[5] = math.remainder(v[0] + s[11] * (t - tm), 2 * math.pi)
        sl = sn["speed_log"].last_valid
        if sl is not None:
            s[6] = sl[1][0]
        g = sn["gnss"].last_valid
        if g is not None:
            tm, v = g
            age = t - tm
            R = geo.rot_zyx(s[3], p.Th_rest + s[4], s[5])
            arm = R @ sn["gnss"].lever
            s[0] = v[3] - arm[0] + v[6] * age
            s[1] = v[4] - arm[1] + v[7] * age
            s[2] = v[5] - arm[2] + v[8] * age - p.Z_rest
            s[8] = v[8]
            c, sy = math.cos(s[5]), math.sin(s[5])
            s[7] = -v[6] * sy + v[7] * c
        s[12], s[13] = env.s[12], env.s[13]
        self.s = s
        self.n_updates += 1
        return s.copy()
