"""
Frames and geodesy.

FRAMES. The plant's world frame is x, y horizontal, z up, right-handed, yaw
counter-clockwise about +z (sim/planing_vessel.py; studies/visual_export.py
verified the signs): read as x = east, y = north, z = up it IS ENU, and the
body frame (x forward, y left, z up, roll about +x lifting the +y side,
pitch about +y with bow down positive) is ROS REP-103's FLU. LOTUSim's
VesselPosition carries "orientation front-left-up" and its agent heading is
"relative to East" with a right-hand rule about Z up -- the same numbers.
The attitude is R = Rz(yaw) Ry(pitch) Rx(roll) (ZYX), with the TOTAL pitch
Th_rest + pitch (the plant's state measures pitch from the rest attitude).

GEODESY. WGS84 ellipsoid, exact transforms geodetic <-> ECEF <-> local ENU
tangent plane at a reference point (no flat-earth approximation):
ECEF from geodetic in closed form, geodetic from ECEF by Bowring's iteration
to 1e-14 rad (converges in 2-3 iterations near the surface).
"""
import math

import numpy as np

A_WGS84 = 6378137.0
F_WGS84 = 1.0 / 298.257223563
E2 = F_WGS84 * (2.0 - F_WGS84)
B_WGS84 = A_WGS84 * (1.0 - F_WGS84)
EP2 = (A_WGS84 ** 2 - B_WGS84 ** 2) / B_WGS84 ** 2


def geodetic_to_ecef(lat_deg, lon_deg, h):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    s, c = math.sin(lat), math.cos(lat)
    N = A_WGS84 / math.sqrt(1.0 - E2 * s * s)
    return np.array([(N + h) * c * math.cos(lon), (N + h) * c * math.sin(lon),
                     (N * (1.0 - E2) + h) * s])


def ecef_to_geodetic(x, y, z):
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    # Bowring's starting value, then fixed-point on the latitude
    th = math.atan2(z * A_WGS84, p * B_WGS84)
    lat = math.atan2(z + EP2 * B_WGS84 * math.sin(th) ** 3,
                     p - E2 * A_WGS84 * math.cos(th) ** 3)
    for _ in range(8):
        s = math.sin(lat)
        N = A_WGS84 / math.sqrt(1.0 - E2 * s * s)
        h = p / math.cos(lat) - N if abs(lat) < math.radians(89.0) else \
            z / s - N * (1.0 - E2)
        new = math.atan2(z, p * (1.0 - E2 * N / (N + h)))
        if abs(new - lat) < 1e-14:
            lat = new
            break
        lat = new
    s = math.sin(lat)
    N = A_WGS84 / math.sqrt(1.0 - E2 * s * s)
    h = p / math.cos(lat) - N if abs(lat) < math.radians(89.0) else \
        z / s - N * (1.0 - E2)
    return math.degrees(lat), math.degrees(lon), h


def _enu_basis(lat_deg, lon_deg):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    # rows: east, north, up in ECEF
    return np.array([[-so, co, 0.0],
                     [-sl * co, -sl * so, cl],
                     [cl * co, cl * so, sl]])


class LocalFrame:
    """The ENU tangent plane at (lat0, lon0, h0)."""

    def __init__(self, lat0=0.0, lon0=0.0, h0=0.0):
        self.lat0, self.lon0, self.h0 = float(lat0), float(lon0), float(h0)
        self.o = geodetic_to_ecef(self.lat0, self.lon0, self.h0)
        self.R = _enu_basis(self.lat0, self.lon0)

    def to_geodetic(self, e, n, u=0.0):
        x = self.o + self.R.T @ np.array([e, n, u], float)
        return ecef_to_geodetic(*x)

    def to_enu(self, lat, lon, h=0.0):
        d = geodetic_to_ecef(lat, lon, h) - self.o
        return tuple(float(v) for v in self.R @ d)

    def as_dict(self):
        return dict(latitude=self.lat0, longitude=self.lon0, altitude=self.h0)


# ------------------------------------------------------------ attitude
def rot_zyx(roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll): body (FLU) -> world (ENU)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def quat_zyx(roll, pitch, yaw):
    """Unit quaternion (x, y, z, w) of rot_zyx (geometry_msgs order)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def quat_to_rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def euler_rates_to_body(roll, pitch, droll, dpitch, dyaw):
    """Body angular velocity (FLU) from ZYX Euler-angle rates."""
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    return np.array([droll - dyaw * sp,
                     dpitch * cr + dyaw * sr * cp,
                     -dpitch * sr + dyaw * cr * cp])


def yaw_to_bearing_deg(yaw):
    """ENU yaw (rad, CCW from east) -> compass bearing (deg, CW from north)."""
    return (90.0 - math.degrees(yaw)) % 360.0


def bearing_deg_to_yaw(b):
    return deg_to_rad_wrapped(90.0 - float(b))


def deg_to_rad_wrapped(d):
    """Degrees -> radians in (-pi, pi], computed as pi * (d / 180) so that
    180 gives math.pi exactly (the project's head-sea theta0 is np.pi, and
    the sea's direction grid is built from theta0 bit by bit)."""
    d = float(d) % 360.0
    if d > 180.0:
        d -= 360.0
    return math.pi * (d / 180.0)


def wave_from_deg_to_theta0(from_deg):
    """Waves coming FROM `from_deg` (clockwise from true north) travel
    toward bearing from + 180; as an ENU angle theta0 = 90 - (from + 180)
    = -90 - from (deg). 90 (from the east) -> pi exactly: head seas for
    yaw 0, the project's WAVES_FROM."""
    return deg_to_rad_wrapped(-90.0 - float(from_deg))


def theta0_to_wave_from_deg(theta0):
    return (-90.0 - math.degrees(theta0)) % 360.0
