#!/usr/bin/env python3
"""
Waterjet propulsion and nozzle steering.

A waterjet is not a propeller with a rudder behind it, and the differences are
exactly the ones a controller trips over:

  * steering comes from turning the JET, so it needs flow. No throttle, no
    steering. A rudder keeps working while the hull carries way.
  * the steering force is the jet's GROSS momentum flux turned sideways,
    m_dot Vj sin(delta). The forward force is what is left after the inlet
    has taken in the water at the vessel's speed, m_dot (Vj cos(delta) - Vin).
    At speed the first can be several times the second: a jet steers hard
    even when it is barely pushing.
  * the intake is a hole in the bottom. When it breaks the surface the pump
    draws air and loses its prime: thrust AND steering vanish together, and
    come back only after the intake is under water again for a while.
  * thrust falls with speed at a given power, and reverse is a bucket that
    turns the jet forward.

The model is momentum theory with a power balance, the textbook one
(Allison 1993; ITTC 7.5-02-05-03.1). For a nozzle of area A, delivered shaft
power P and pump efficiency eta:

    eta P = 1/2 rho A Vj ((1 + zeta_n) Vj^2 - (1 - zeta_i) Vin^2)
    Vin   = (1 - w) u                        inlet velocity
    T     = rho A Vj (Vj - Vin)              net thrust, straight ahead

zeta_n and zeta_i are the nozzle and inlet loss coefficients and w the
inlet's wake fraction. Solved for Vj by Newton from a start where the cubic is
already positive, so the iteration converges monotonically.

The command keeps the meaning it has for the propeller: thrust AT THE DESIGN
SPEED, in newtons. The pump is run at the power that would deliver it there,
and the same power then gives whatever the jet gives at the actual speed. So
nothing upstream changes its units; only what the water does with them.
"""
import numpy as np

from .actuators import Delay, Propulsion

RHO = 1025.0

CALIBRATION_NEEDED = [
    "pump efficiency and inlet/nozzle losses (need a speed vs RPM table "
    "or bollard pull)",
    "nozzle area and power (need the pump model or a bollard pull)",
    "loss and recovery of prime when the intake emerges (need trial logs)",
    "reverse-bucket efficiency k_rev",
    "nozzle travel, rate and delay (need the steering actuator's data)",
]


class JetPump:
    """Momentum theory with a power balance. Pure functions, no state."""

    def __init__(self, area, p_max, eta_pump=0.70, zeta_n=0.02, zeta_i=0.25,
                 wake=0.05, rho=RHO):
        self.area, self.p_max = float(area), float(p_max)
        self.eta, self.zeta_n, self.zeta_i = eta_pump, zeta_n, zeta_i
        self.wake, self.rho = wake, rho

    @property
    def d_nozzle(self):
        return float(np.sqrt(4.0 * self.area / np.pi))

    def v_in(self, u):
        return (1.0 - self.wake) * max(float(u), 0.0)

    def jet_velocity(self, power, u):
        """Jet velocity for shaft power `power` (W) at vessel speed u."""
        a = 1.0 + self.zeta_n
        vin = self.v_in(u)
        b = (1.0 - self.zeta_i) * vin * vin
        c = 2.0 * self.eta * max(float(power), 0.0) / (self.rho * self.area)
        # f(V) = a V^3 - b V - c. At V0 = (c/a)^(1/3) + sqrt(b/a),
        # f(V0) = 3 a x^2 y + 2 b x >= 0 with x, y the two terms, and f is
        # convex for V > 0, so Newton from V0 descends monotonically onto
        # the physical (largest) root.
        v = (c / a) ** (1.0 / 3.0) + np.sqrt(b / a)
        for _ in range(8):
            f = a * v ** 3 - b * v - c
            fp = 3.0 * a * v * v - b
            if fp <= 0.0:
                break
            dv = f / fp
            v -= dv
            if abs(dv) < 1e-10 * max(v, 1.0):
                break
        return float(v)

    def thrust(self, power, u):
        """Net thrust straight ahead (negative once the pump cannot push the
        water faster than it takes it in: ram drag)."""
        vj = self.jet_velocity(power, u)
        return self.rho * self.area * vj * (vj - self.v_in(u))

    def power_for_thrust(self, thrust, u):
        """Shaft power that gives net thrust `thrust` >= 0 at speed u."""
        vin = self.v_in(u)
        vj = 0.5 * vin + np.sqrt(0.25 * vin * vin
                                 + max(float(thrust), 0.0)
                                 / (self.rho * self.area))
        return float(0.5 * self.rho * self.area * vj
                     * ((1.0 + self.zeta_n) * vj * vj
                        - (1.0 - self.zeta_i) * vin * vin) / self.eta)

    @classmethod
    def sized_for(cls, t_design, u_design, ratio=0.35, **kw):
        """The jet that gives `t_design` at `u_design` at full power, with
        inlet-to-jet velocity ratio `ratio` there -- the design choice that
        sets nozzle area against power (a low ratio is a small, fast,
        inefficient jet; a high one a big, slow, efficient one)."""
        pump = cls(1.0, 1.0, **kw)
        vin = pump.v_in(u_design)
        vj = vin / ratio
        pump.area = t_design / (pump.rho * vj * (vj - vin))
        pump.p_max = pump.power_for_thrust(t_design, u_design)
        return pump


class Waterjet(Propulsion):
    """The thrust channel of a waterjet.

    Command, state, lag, rate limit, delay and dead band are the propeller's
    (Propulsion.advance), in newtons at the design speed. What differs is
    what reaches the water: `forces` returns the jet's force vector for a
    thrust state and a nozzle angle, and the pump's prime is a state of its
    own, advanced once per step by `update_prime`.
    """

    propulsor = "waterjet"

    def __init__(self, pump, x_nozzle, z_nozzle, x_intake, z_intake,
                 k_rev=0.6, tau_prime_loss=0.05, tau_prime_back=1.0,
                 prime_band=None, **prop_kw):
        prop_kw.setdefault("rho", pump.rho)
        super().__init__(x_prop=x_nozzle, z_prop=z_nozzle,
                         r_prop=0.5 * pump.d_nozzle, **prop_kw)
        self.pump = pump
        self.x_nozzle, self.z_nozzle = x_nozzle, z_nozzle
        self.x_intake, self.z_intake = x_intake, z_intake
        self.k_rev = k_rev
        self.tau_prime_loss, self.tau_prime_back = tau_prime_loss, \
            tau_prime_back
        # depth of water over the intake below which it draws air: half a
        # nozzle diameter, a stand-in with the right order and no better
        self.prime_band = (0.5 * pump.d_nozzle if prime_band is None
                           else prime_band)
        self.prime = 1.0

    # --------------------------------------------------------------- flow
    def flow(self, thrust, u):
        """(mass flow, jet velocity, inlet velocity) for thrust state
        `thrust` (N at the design speed, either sign) at speed u.

        Below the dead band the FLOW goes to zero with the command, as the
        propeller's shaft does (DEFECTS G5): at exactly zero the pump
        delivers nothing -- no thrust, no steering -- and the forces are
        continuous on the way there. Without it a vanishing command would
        leave the flow of a pump just overcoming its losses at the design
        speed, and that flow steers."""
        t = abs(float(thrust))
        if t <= 0.0:
            return 0.0, 0.0, self.pump.v_in(u)
        frac = 1.0
        db = max(self.dead_band, 1e-9)
        if t < db:
            frac, t = t / db, db
        p = self.pump.power_for_thrust(t, self.u_ref)
        vj = self.pump.jet_velocity(p, u)
        m_dot = frac * self.prime * self.pump.rho * self.pump.area * vj
        return m_dot, vj, self.pump.v_in(u)

    def forces(self, thrust, delta, u):
        """(surge force, sway force) on the hull, at the nozzle.

        Positive delta gives positive sway force at the stern, the same sign
        as positive rudder, so everything downstream reads the nozzle as it
        read the rudder. Reverse (thrust < 0): the bucket turns the jet
        forward with efficiency k_rev and reverses the steering; the inlet
        still takes in the water at the vessel's speed."""
        m_dot, vj, vin = self.flow(thrust, u)
        c, s = np.cos(delta), np.sin(delta)
        if thrust >= 0.0:
            return m_dot * (vj * c - vin), m_dot * vj * s
        return (-m_dot * (self.k_rev * vj * c + vin),
                -self.k_rev * m_dot * vj * s)

    def net_thrust(self, thrust, u):
        """Straight-ahead thrust actually delivered at speed u. Diagnostic,
        and what `speed_correction` means for a jet."""
        return self.forces(thrust, 0.0, u)[0]

    def speed_correction(self, thrust, u):
        return self.net_thrust(thrust, u)

    def power(self, thrust):
        """Shaft power drawn for a thrust state, W."""
        t = abs(float(thrust))
        if t <= 0.0:
            return 0.0
        db = max(self.dead_band, 1e-9)
        frac = min(t / db, 1.0)
        return frac * self.pump.power_for_thrust(max(t, db), self.u_ref)

    # -------------------------------------------------------------- prime
    def update_prime(self, submergence, dt):
        """Advance the pump's prime by one step from the depth of water over
        the intake (negative once it has broken the surface). Lost fast,
        regained slowly; call once per step, never inside a derivative."""
        target = float(np.clip(submergence / max(self.prime_band, 1e-9),
                               0.0, 1.0))
        tau = self.tau_prime_loss if target < self.prime else \
            self.tau_prime_back
        self.prime += (target - self.prime) * (1.0 - np.exp(-dt / tau))
        return self.prime


class Nozzle:
    """The steering nozzle: travel, rate limit, first-order lag and delay --
    the Rudder's actuator dynamics without its hydrofoil, because the force
    comes from the jet (Waterjet.forces), not from the flow past a blade.

    `stall` equals `max` (a nozzle does not stall) and `span` is zero, so the
    code that reads those off a rudder keeps working."""

    def __init__(self, max_deg=27.0, rate_deg=40.0, tau=0.15, delay=0.10,
                 dt=0.05, x_nozzle=-4.9, z_nozzle=-0.3):
        self.max = np.radians(max_deg)
        self.stall = self.max
        self.rate = np.radians(rate_deg)
        self.tau = tau
        self.delay = Delay(delay, dt)
        self.x_rud, self.z_rud, self.span = x_nozzle, z_nozzle, 0.0

    def step(self, angle, cmd, dt):
        cmd = float(np.clip(self.delay(cmd), -self.max, self.max))
        d = float(np.clip((cmd - angle) / self.tau, -self.rate, self.rate))
        return angle + d * dt

    def ventilation_factor(self, submergence):
        return 1.0


def build(hull, t_max, u_ref, dt, ph, rho):
    """(Waterjet, Nozzle, nozzle point, intake point) for a Hull with
    propulsor="waterjet". `ph` is the plant's Froude-scaled placeholder set
    (sim/vessel.froude_placeholders): the engine's lag, delay and rate limit
    stay the propeller's placeholders until the real drive is measured."""
    j = dict(hull.jet or {})
    r = np.sqrt(ph["lam"])
    stern = float(hull.sections().x_stern)
    x_n = j.get("x_nozzle", stern + 0.01 * hull.L)
    z_n = j.get("z_nozzle", -0.375 * hull.T)
    x_i = j.get("x_intake", stern + 0.09 * hull.L)
    z_i = j.get("z_intake", -0.94 * hull.T)
    kw = dict(eta_pump=j.get("eta_pump", 0.70), zeta_n=j.get("zeta_n", 0.02),
              zeta_i=j.get("zeta_i", 0.25), wake=j.get("wake", 0.05),
              rho=rho)
    if "d_nozzle" in j and "p_max" in j:
        pump = JetPump(np.pi * j["d_nozzle"] ** 2 / 4.0, j["p_max"], **kw)
        t_max = pump.thrust(pump.p_max, u_ref)
    else:
        pump = JetPump.sized_for(t_max, u_ref, j.get("ratio", 0.35), **kw)
    pp = dict(ph["prop"])
    pp.update(t_max=t_max, t_min=-t_max / 3.0,
              rate_max=t_max * 8000.0 / 12000.0 / r,
              dead_band=t_max * 50.0 / 12000.0, u_ref=u_ref)
    jet = Waterjet(pump, x_n, z_n, x_i, z_i, k_rev=j.get("k_rev", 0.6),
                   tau_prime_loss=j.get("tau_prime_loss", 0.05 * r),
                   tau_prime_back=j.get("tau_prime_back", 1.0 * r),
                   prime_band=j.get("prime_band"), dt=dt, **pp)
    noz = Nozzle(max_deg=j.get("nozzle_max_deg", 27.0),
                 rate_deg=j.get("nozzle_rate_deg", 40.0 / r),
                 tau=j.get("nozzle_tau", 0.15 * r),
                 delay=j.get("nozzle_delay", 0.10 * r), dt=dt,
                 x_nozzle=x_n, z_nozzle=z_n)
    return jet, noz, (float(x_n), 0.0, float(z_n)), (float(x_i), 0.0,
                                                      float(z_i))
