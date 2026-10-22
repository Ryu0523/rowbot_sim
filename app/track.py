"""
Tracks: a straight line through the start, or waypoint legs.

The cost and the controllers of this project work in a TRACK FRAME: x along
the track, y to its left (learn/repro/task.Mission with track=phi). A leg is
the segment from A to B with angle phi = atan2(B - A); its frame has origin
A. The straight track is one infinite leg from the start at the given
heading, which is exactly Mission's frame when the boat starts at the
origin (cross-track -sin(phi) x + cos(phi) y).

Leg switching (waypoints): the leg ends when the boat comes within `radius`
of B or passes B along the leg (along-track progress >= leg length); the
next leg starts at B. After the last waypoint the track is complete (or
starts over with loop=True).

Guidance (what heading a heading-holding policy asks for):
  hold  the leg angle phi (Mission: no cross-track feedback)
  los   phi + atan(-y / Delta), the line-of-sight law (Fossen 2011, 10.3.2)
        with lookahead Delta (default 5 boat lengths)
"""
import math

import numpy as np


class Track:
    def __init__(self, spec, start_xy):
        self.spec = dict(spec)
        self.start = (float(start_xy[0]), float(start_xy[1]))
        if spec["kind"] == "line":
            self.points = [self.start]
            self.line = True
            self.heading = float(spec["heading"])
            self.radius = 0.0
            self.loop = False
        elif spec["kind"] == "waypoints":
            self.points = [self.start] + [(float(x), float(y))
                                          for x, y in spec["points"]]
            if len(self.points) < 2:
                raise ValueError("waypoints: at least one")
            self.line = False
            self.radius = float(spec.get("radius", 20.0))
            self.loop = bool(spec.get("loop", False))
        else:
            raise ValueError(f"track kind {spec['kind']!r}")
        self.reset()

    def reset(self):
        self.leg = 0
        self.complete = False
        self.n_switch = 0
        self._set_leg(0)

    def _set_leg(self, i):
        self.leg = i
        if self.line:
            self.ox, self.oy = self.start
            self.phi = self.heading
            self.length = math.inf
        else:
            (ax, ay), (bx, by) = self.points[i], self.points[i + 1]
            self.ox, self.oy = ax, ay
            self.phi = math.atan2(by - ay, bx - ax)
            self.length = math.hypot(bx - ax, by - ay)
        # numpy's cos / sin, as Mission computes its _cp, _sp (numpy and
        # libm may differ in the last bit)
        self.cp, self.sp = np.cos(self.phi), np.sin(self.phi)

    @property
    def n_legs(self):
        return 1 if self.line else len(self.points) - 1

    def to_frame(self, x, y):
        """(along, cross) of a world point in the current leg's frame."""
        dx, dy = x - self.ox, y - self.oy
        return self.cp * dx + self.sp * dy, -self.sp * dx + self.cp * dy

    def cross(self, x, y):
        """Cross-track error, written as Mission writes it when the leg
        starts at the origin (-sin(phi) x + cos(phi) y)."""
        if self.ox == 0.0 and self.oy == 0.0:
            return -self.sp * x + self.cp * y
        return -self.sp * (x - self.ox) + self.cp * (y - self.oy)

    def along_cross(self, s):
        """Mission.along_cross for the current leg: (speed made good along
        the leg, cross-track error)."""
        d = s[5] - self.phi
        return (s[6] * np.cos(d) - s[7] * np.sin(d), self.cross(s[0], s[1]))

    def update(self, x, y):
        """Advance the leg if its end was reached; True when it switched."""
        if self.line or self.complete:
            return False
        bx, by = self.points[self.leg + 1]
        along, _ = self.to_frame(x, y)
        if math.hypot(x - bx, y - by) <= self.radius or along >= self.length:
            self.n_switch += 1
            if self.leg + 1 < self.n_legs:
                self._set_leg(self.leg + 1)
            elif self.loop:
                # back to the first waypoint from the last one
                self.points = [self.points[-1]] + self.points[1:]
                self._set_leg(0)
            else:
                self.complete = True
            return True
        return False

    def guidance(self, x, y, mode="los", lookahead=None, L=5.0):
        """Heading reference (rad, ENU) for the current leg."""
        if mode == "hold":
            return self.phi
        if mode != "los":
            raise ValueError(f"guidance {mode!r}: 'hold' or 'los'")
        d = float(lookahead) if lookahead else 5.0 * L
        _, yt = self.to_frame(x, y)
        return self.phi + math.atan2(-yt, d)

    def describe(self):
        return dict(kind=self.spec["kind"], points=[list(p) for p in
                                                   self.points],
                    radius=self.radius, loop=self.loop,
                    heading=(self.heading if self.line else None))
