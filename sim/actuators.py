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
                 r_prop=0.22, dt=0.05, kt0=0.45, j0=0.75, wake=0.20,
                 u_ref=4.5, series=None, n_units=1, rho=1025.0):
        # `t_max`, `t_min` and the command are the TOTAL thrust. With several
        # units (twin screws, twin outboards) they share one command and one
        # lag; the vessel places each unit's share at its own position.
        self.t_max, self.t_min = t_max, t_min
        self.tau, self.rate_max = tau, rate_max
        self.dead_band = dead_band
        self.x_prop, self.z_prop, self.r_prop = x_prop, z_prop, r_prop
        self.delay = Delay(delay, dt)
        self.n_units = max(int(n_units), 1)
        self.rho = rho
        # Open-water curve. Default: linearised, K_T(J) = kt0 (1 - J/j0), with
        # J the advance ratio (1-w)u/(nD) -- the right shape and order, but a
        # placeholder, and against the KVLCC2 open-water data its slope was
        # 1.6x too steep (studies/exp_propeller.py). With `series` = dict(Z,
        # AE_A0, P_D) the Wageningen B-series regression is used instead,
        # which the same data reproduced to 5.4% against a 9.6% spread between
        # the two measurements.
        self.kt0, self.j0, self.wake = kt0, j0, wake
        self.d_prop = 2.0 * r_prop
        self.u_ref = u_ref
        self.series = None
        if series:
            from sim.propeller import WageningenB
            self.series = WageningenB(series["Z"], series["AE_A0"],
                                      series["P_D"], self.d_prop)
            self._tabulate()

    def _tabulate(self):
        """Thrust at the design speed against shaft speed, for inverting the
        B-series curve by interpolation: a bisection per derivative call cost
        a millisecond, which an RL run of 1e6 steps cannot carry."""
        va_ref = (1.0 - self.wake) * self.u_ref
        t_top = 2.0 * max(self.t_max, 1.0) / self.n_units
        n_hi = self.series.shaft_speed_for(t_top, va_ref, rho=self.rho)
        self._n_tab = np.linspace(0.0, n_hi, 400)
        self._t_tab = np.array([self.series.thrust(n, va_ref, rho=self.rho)
                                for n in self._n_tab])
        self._t_tab = np.maximum.accumulate(self._t_tab)

    def speed_correction(self, thrust, u):
        """Thrust actually produced at speed `u` from a command set at u_ref.

        The model had thrust depend on lag, rate limit, dead band, transport
        delay and ventilation -- and not on the speed of advance. A real
        propeller loses thrust as the vessel speeds up and gains it as the
        vessel slows, INSTANTLY: it is the one part of the thrust channel with
        no lag at all, and this project's central finding is about the thrust
        channel being too slow.

            T = rho n^2 D^4 K_T(J),   J = (1-w) u / (n D)
              = rho D^4 kt0 n^2  -  rho D^3 kt0 (1-w) u n / j0
              = a n^2 + b(u) n

        The command keeps its units and its meaning at the design speed: solve
        the quadratic for the shaft speed n that would deliver `thrust` at
        u_ref, then evaluate the same n at the actual u. What is added is the
        speed DEPENDENCE, not a recalibration of the magnitude -- so the speed
        envelope the rest of the project was tuned around survives.

        Astern is left alone. A propeller running backwards has a different
        curve entirely and modelling it from this one would be invention.
        """
        if thrust <= 0.0:
            return float(thrust)
        t1 = thrust / self.n_units                   # one unit's share
        n = self._shaft(t1)
        if self.series is not None:
            va = (1.0 - self.wake) * max(float(u), 0.0)
            return float(max(self.series.thrust(n, va, rho=self.rho), 0.0)
                         * self.n_units)
        rho = self.rho
        a = rho * self.d_prop ** 4 * self.kt0
        k = rho * self.d_prop ** 3 * self.kt0 * (1.0 - self.wake) / self.j0
        return float(max(a * n * n - k * float(u) * n, 0.0)) * self.n_units

    def _shaft(self, t1):
        """Shaft speed at which one unit delivers t1 > 0 at the design speed.

        Below the dead band the shaft spins DOWN TO REST with the command.
        Without that, a vanishing command meant n0, the shaft speed that gives
        zero thrust AT THE DESIGN SPEED -- and n0 pushes at any lower speed:
        a thrust state of 1e-6 N delivered 2 kN to the USV at rest, and
        through the B-series, evaluated at J ~ 1e5, 7e4 N to a one-tonne
        tanker model (DEFECTS G5). A decaying thrust state never reaches zero
        exactly, so this was every 'thrust off' after 'thrust on', not an
        edge case. At and above the dead band nothing changes."""
        db1 = self.dead_band / self.n_units
        frac = 1.0
        if 0.0 < t1 < db1:
            frac, t1 = t1 / db1, db1
        if self.series is not None:
            return float(np.interp(t1, self._t_tab, self._n_tab)) * frac
        rho = self.rho
        a = rho * self.d_prop ** 4 * self.kt0
        k = rho * self.d_prop ** 3 * self.kt0 * (1.0 - self.wake) / self.j0
        b_ref = -k * self.u_ref
        return float((-b_ref + np.sqrt(b_ref ** 2 + 4 * a * t1)) / (2 * a)) \
            * frac

    def shaft_speed(self, thrust):
        """Shaft speed implied by a thrust command, rev/s. Diagnostic."""
        t1 = max(thrust, 0.0) / self.n_units
        return self._shaft(t1) if t1 > 0.0 else 0.0

    def ventilation_factor(self, submergence):
        """Thrust multiplier from propeller submergence (metres above the
        propeller centre). Fully wetted above ~1.5 radii, zero once the
        propeller centre broaches."""
        x = submergence / (1.5 * self.r_prop)
        return float(np.clip(x, 0.0, 1.0)) ** 1.5

    def effective(self, thrust, submergence, u=None):
        """Thrust actually delivered to the water. Pure function -- safe to
        call inside a derivative evaluation.

        `u` optional so that callers who only want the ventilation loss (the
        reduced model, diagnostics) keep the old behaviour.
        """
        t = thrust if u is None else self.speed_correction(thrust, u)
        return t * self.ventilation_factor(submergence)

    def effective_units(self, thrust, submergences, u=None):
        """Thrust each unit delivers: its share of the command, corrected for
        speed, times its OWN ventilation -- in a seaway one screw of a pair
        can broach while the other does not."""
        t = thrust if u is None else self.speed_correction(thrust, u)
        per = t / self.n_units
        return np.array([per * self.ventilation_factor(s)
                         for s in np.atleast_1d(submergences)])

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

    # Stall at 38 deg, not 28. The two numbers were inconsistent: the rudder
    # could be commanded to 35 deg while stalling at 28, so the last third of
    # its travel REDUCED the turning moment and the turning circle widened
    # again past 25 deg of helm (measured 2.27 L at 25 deg, 2.70 L at 35 deg).
    # Nobody builds that.
    #
    # Raising stall is the defensible fix rather than cutting the maximum.
    # 35 deg is the standard maximum on essentially every vessel, and a rudder
    # sitting in the propeller race stalls far later than the same foil in free
    # stream -- the accelerated slipstream re-energises the boundary layer.
    # Rudders are designed so that stall sits at or beyond the stops, which is
    # what 38 deg now expresses. It remains a placeholder like the viscous
    # terms, but it is a self-consistent one, and `sim/test_manoeuvre.py`
    # bounds it.
    @staticmethod
    def whicker_fehlner(ar, sweep=0.0):
        """Lift-curve slope of a low-aspect-ratio all-movable control surface,
        per radian, from Whicker & Fehlner's free-stream measurements:

            dCL/dalpha = 1.8 pi AR / (1.8 + cos(L) sqrt(AR^2 / cos^4(L) + 4))

        `ar` is the EFFECTIVE aspect ratio. A rudder whose root is close to the
        hull sees its own mirror image, so the effective value is up to twice
        the geometric span^2 / area.
        """
        c = np.cos(sweep)
        return 1.8 * np.pi * ar / (1.8 + c * np.sqrt(ar ** 2 / c ** 4 + 4.0))

    def __init__(self, area=0.18, cl_alpha=None, stall_deg=38.0, x_rud=-4.9,
                 z_rud=-0.45, span=0.55, max_deg=35.0, rate_deg=25.0, tau=0.25,
                 delay=0.10, rho=1025.0, dt=0.05, gamma_flow=0.45, u_min=0.5,
                 ar_factor=2.0, h_full=None):
        # Lift slope is COMPUTED from the blade geometry unless given. It was a
        # hard-coded 3.5 /rad. Whicker & Fehlner put this blade at 2.15 in free
        # stream and 3.33 with a full mirror image at the root, so 3.5 sat 5%
        # above even the most favourable case. The likely reason is that it was
        # quietly standing in for a term the model does not have -- the
        # propeller slipstream accelerating the flow over the blade -- which
        # raises the dynamic pressure, not the slope. Computing it also means an
        # imported hull gets a rudder that scales with its own blade instead of
        # inheriting this one's number.
        # `ar_factor` 2.0 = root close against the hull (transom-hung, small
        # gap); lower it for a spade rudder hanging well clear.
        self.ar_factor = ar_factor
        if cl_alpha is None:
            cl_alpha = self.whicker_fehlner(ar_factor * span ** 2 / area)
        self.area, self.cl_alpha = area, cl_alpha
        self.stall = np.radians(stall_deg)
        self.x_rud, self.z_rud, self.span = x_rud, z_rud, span
        self.max = np.radians(max_deg)
        self.rate = np.radians(rate_deg)
        self.tau, self.rho = tau, rho
        self.delay = Delay(delay, dt)
        self.gamma_flow = gamma_flow
        self.u_min = u_min          # floor on the inflow speed in the angle
        # fully effective below this depth: a fraction of the span -- the
        # USV's 0.10 m over its 0.55 m blade -- so it scales with the blade.
        # As an absolute 0.10 m it made every hull under 0.465 m draught fail
        # the assert below (the default blade top sits 0.215 T deep): a 2 m
        # catamaran, a 6 m ship model, KVLCC2 at 1:68.
        self.h_full = (0.10 / 0.55) * span if h_full is None else h_full
        self.h_zero = -0.5 * span   # half the blade out: no lift left
        # The static submergence of the blade top must give exactly 1.0, or the
        # calm-water turning trials are silently wrong. Checked here rather
        # than assumed, because the first version of the curve was not.
        assert self.ventilation_factor(-(z_rud + 0.5 * span)) == 1.0, \
            "rudder ventilation curve is not saturated at static submergence"

    def ventilation_factor(self, submergence):
        """Lift multiplier from rudder submergence, metres of water over the
        TOP of the blade. Same mechanism as the propeller's: a foil close to
        the free surface draws air down its low-pressure side and the lift
        collapses. `submergence_factor` existed in the signature and no caller
        ever passed it anything but 1.0, on a hull whose rudder top is clear of
        the water 31% of the time in SS5.

        Two thresholds, both stated as depths rather than folded into one
        opaque scale, because the first version of this got the argument's
        meaning wrong and cut calm-water rudder force in half without anything
        noticing:

          h_full   submerged deeper than this, the blade is fully effective.
                   The static value here is 0.175 m, so a threshold of 0.10 m
                   means calm water gives exactly 1.0 -- assert it, do not hope.
          h_zero   above this the blade has emerged far enough to be useless.
                   Half the span out.
        """
        f = (submergence - self.h_zero) / (self.h_full - self.h_zero)
        return float(np.clip(f, 0.0, 1.0)) ** 1.5

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

    def inflow_angle(self, u, v=0.0, r=0.0):
        """Angle the water actually arrives at, relative to the centreline.

        The rudder sits aft, so it sees the vessel's sway velocity plus the
        tangential velocity from yaw rate:  v_R = v + r x_rud. That was missing
        entirely -- `force` took the geometric deflection and the surge speed,
        so the model had no way to know the angle of attack, and with it lost
        the vessel's own weathervane feedback on the rudder.

        `gamma` is the flow-straightening factor: the hull deflects the oncoming
        flow towards the centreline before it reaches the rudder, so the rudder
        feels only a fraction of the drift angle. It is a placeholder in the
        same sense as the viscous terms; Molland and Turnock's systematic
        rudder-behind-hull measurements are what belongs here.
        """
        return np.arctan2(v + r * self.x_rud, max(float(u), self.u_min))

    def force(self, angle, u, v=0.0, r=0.0, submergence_factor=1.0,
              x_rud=None):
        """Returns (side force, yaw moment, drag) of ONE blade.

        `v` and `r` default to zero so that a caller who genuinely means "pure
        surge, no drift" gets the old behaviour, but the plant always passes
        them. `x_rud` places the blade when a vessel has more than one (the
        inflow angle depends on it through r x_rud); default, this rudder's own.
        """
        x = self.x_rud if x_rud is None else x_rud
        v_r = v + r * x
        alpha = float(np.clip(angle - self.gamma_flow
                              * np.arctan2(v_r, max(float(u), self.u_min)),
                              -self.max, self.max))
        # dynamic pressure from the RESULTANT inflow, not the surge component
        q = 0.5 * self.rho * self.area * np.sign(u) * (u * u + v_r * v_r)
        lift = q * self.lift_coefficient(alpha) * submergence_factor
        drag = abs(q) * 0.02 * (1 + 4 * (alpha / max(self.max, 1e-9)) ** 2)
        return lift, lift * x, drag
