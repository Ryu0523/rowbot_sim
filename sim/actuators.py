#!/usr/bin/env python3
"""
Propulsion and steering, with the failure modes that actually break sim-to-real.

The USV RL literature reports actuators -- not hydrodynamics -- as the largest
residual error after domain randomisation: thruster lag, control delay, rate
limits and dead bands. One published study had an agent that could not perform
an in-place rotation on the real vessel purely from actuator mismatch. So these
are modelled explicitly rather than folded into a single first-order lag.

In SS4-5 one more matters: PROPELLER VENTILATION. As the stern lifts, the
propeller draws air and thrust collapses, then recovers on re-immersion. That
makes available thrust intermittent exactly when the controller needs it, and
no amount of domain randomisation on a fully-wetted thrust curve reproduces it.

The ventilation curve here is a smooth monotone stand-in with the right shape
and limits, not a calibrated one -- it needs bollard-pull and seakeeping trial
data for the actual propeller. It is flagged in `CALIBRATION_NEEDED`.
"""
import numpy as np

CALIBRATION_NEEDED = [
    "ventilation thrust-loss curve (needs propeller trials)",
    "rudder lift slope and stall angle (needs CFD or tank tests)",
    "thrust lag and rate limit (needs bench tests on the real drive)",
]


class Delay:
    """Fixed transport delay on a scalar command, one control step resolution."""

    def __init__(self, seconds, dt):
        self.n = max(int(round(seconds / dt)), 0)
        self.buf = None

    def __call__(self, value):
        if self.n == 0:
            return value
        if self.buf is None:
            self.buf = [value] * self.n
        self.buf.append(value)
        return self.buf.pop(0)


class Propulsion:
    """Single fixed thruster on the centreline.

    States: delivered thrust (after lag). Commands pass through dead band,
    saturation, transport delay, rate limit and first-order lag, in that order.
    """

    def __init__(self, t_max=12000.0, t_min=-4000.0, tau=1.5, rate_max=8000.0,
                 dead_band=50.0, delay=0.15, x_prop=-4.6, z_prop=-0.55,
                 r_prop=0.22, dt=0.05):
        self.t_max, self.t_min = t_max, t_min
        self.tau, self.rate_max = tau, rate_max
        self.dead_band = dead_band
        self.x_prop, self.z_prop, self.r_prop = x_prop, z_prop, r_prop
        self.delay = Delay(delay, dt)

    def ventilation_factor(self, submergence):
        """Thrust multiplier from propeller submergence (metres above the
        propeller centre). Fully wetted above ~1.5 radii, zero once the
        propeller centre broaches."""
        x = submergence / (1.5 * self.r_prop)
        return float(np.clip(x, 0.0, 1.0)) ** 1.5

    def effective(self, thrust, submergence):
        """Thrust actually delivered to the water. Pure function -- safe to
        call inside a derivative evaluation."""
        return thrust * self.ventilation_factor(submergence)

    def advance(self, thrust, cmd, dt):
        """Advance the thruster state by dt. Consumes one slot of the transport
        delay, so it must be called exactly once per control step -- never from
        inside a derivative evaluation, which runs several times per step."""
        cmd = self.delay(cmd)
        if abs(cmd) < self.dead_band:
            cmd = 0.0
        cmd = float(np.clip(cmd, self.t_min, self.t_max))
        d = float(np.clip((cmd - thrust) / self.tau, -self.rate_max,
                          self.rate_max))
        return thrust + d * dt


class Rudder:
    """Rudder aft of the transom. Lift grows with deflection, then stalls, and
    scales with the square of inflow speed, so authority vanishes at low speed
    exactly when a following sea is pushing the stern around."""

    def __init__(self, area=0.18, cl_alpha=3.5, stall_deg=28.0, x_rud=-4.9,
                 z_rud=-0.45, max_deg=35.0, rate_deg=25.0, tau=0.25,
                 delay=0.10, rho=1025.0, dt=0.05):
        self.area, self.cl_alpha = area, cl_alpha
        self.stall = np.radians(stall_deg)
        self.x_rud, self.z_rud = x_rud, z_rud
        self.max = np.radians(max_deg)
        self.rate = np.radians(rate_deg)
        self.tau, self.rho = tau, rho
        self.delay = Delay(delay, dt)

    def lift_coefficient(self, alpha):
        a = np.clip(alpha, -self.max, self.max)
        lin = self.cl_alpha * a
        # soften past stall rather than clipping, so gradients stay usable
        return np.where(np.abs(a) <= self.stall, lin,
                        np.sign(a) * self.cl_alpha * self.stall
                        * (2.0 - np.abs(a) / self.stall))

    def step(self, angle, cmd, dt):
        cmd = float(np.clip(self.delay(cmd), -self.max, self.max))
        d = float(np.clip((cmd - angle) / self.tau, -self.rate, self.rate))
        return angle + d * dt

    def force(self, angle, u, submergence_factor=1.0):
        """Returns (side force, yaw moment, drag)."""
        q = 0.5 * self.rho * self.area * u * abs(u)
        lift = q * self.lift_coefficient(angle) * submergence_factor
        drag = abs(q) * 0.02 * (1 + 4 * (angle / max(self.max, 1e-9)) ** 2)
        return lift, lift * self.x_rud, drag
