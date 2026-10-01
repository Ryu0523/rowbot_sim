#!/usr/bin/env python3
"""
A broad family of random residual functions -- the training distribution of
'things the model does not know'. No physical mechanism is named.

A residual is an extra acceleration in each of the five degrees of freedom
the MPC's reduced model has (surge, sway, yaw, heave, pitch), a function of

    x = [speed / design - 1, sway velocity, yaw rate x L/U, cos(heading),
         sin(heading), thrust fraction, nozzle fraction, heave velocity,
         pitch rate]                                  (INPUTS, all measured)

and of time. Sampled hierarchically: first a STYLE, then the function.

  style     which inputs matter (each on with prob. 1/2), how smooth, how
            large per channel (some channels off), and which extra parts
            are present
  smooth    a random two-layer tanh network of the active inputs
  threshold (p = 0.3) a step in one active input at a random place and
            sharpness -- regime changes
  hysteresis (p = 0.15) a relay on one input: on above one level, off
            below a lower one -- memory
  drift     (p = 0.3) the amplitude wanders slowly with time (an
            Ornstein-Uhlenbeck factor, 20-60 s)

The output is bounded, amplitude x tanh(sum of parts). The amplitude is
log-uniform over a factor of ~30 per channel, up to A_REF: the first review
measured the full plant's own one-step error at 2.5-10x the size the family
first allowed (A_MAX below), which would have turned the real-gap test into
an amplitude-extrapolation test. Relays act on SLOW inputs only (speed,
heading, thrust, nozzle) with their band inside the range those inputs
visit -- on wave-driven inputs they toggled every wave (a threshold in
disguise) or never (a constant), which is not memory. Channels and parts can
be switched off through `exclude` or forced through `force`, so a style can
be held out of training and used to test generalisation beyond the family.
"""
import numpy as np

CHANNELS = ("surge", "sway", "yaw", "heave", "pitch")
INPUTS = ("speed", "sway_v", "yaw_r", "cos_psi", "sin_psi", "thrust",
          "nozzle", "heave_w", "pitch_q")
# largest residual per channel: m/s^2 (surge, sway, heave), rad/s^2 (yaw,
# pitch). Surge ~1/3 of full-thrust acceleration, sway and yaw ~1/5 of the
# jet's authority, heave ~0.5 g, pitch ~ the wave-driven pitch acceleration
A_MAX = np.array([1.0, 0.5, 0.35, 5.0, 1.5])     # the first family's cap
A_REF = np.array([4.0, 4.0, 2.0, 12.0, 3.0])     # now: log-uniform, x0.03-1
SLOW = (0, 3, 4, 5, 6)          # speed, cos/sin heading, thrust, nozzle
# index of each channel's velocity in the plant state
VEL_IDX = (6, 7, 11, 8, 10)


def inputs_of(s, u_ref, L, t_max, rud_max):
    """The residual's inputs from a plant state (14-vector)."""
    return np.array([s[6] / u_ref - 1.0, s[7], s[11] * L / u_ref,
                     np.cos(s[5]), np.sin(s[5]), s[12] / t_max,
                     s[13] / rud_max, s[8], s[10]])


class ResidualFunction:
    def __init__(self, rng, exclude=(), scale=1.0, force=()):
        n = len(INPUTS)
        self.rng = rng
        mask = rng.random(n) < 0.5
        if not mask.any():
            mask[rng.integers(n)] = True
        self.mask = mask.astype(float)
        self.act = np.flatnonzero(mask)
        # input scaling to order one (speed offsets are ~0.3, heave vel ~1)
        self.xs = np.array([0.3, 0.5, 0.5, 1.0, 1.0, 0.3, 0.5, 1.0, 0.3])
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        smooth = rng.uniform(0.3, 2.0)
        H = 16
        self.W1 = rng.normal(0, smooth, (H, n)) * self.mask
        self.b1 = rng.normal(0, 1.0, H)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(H), (5, H))
        self.b2 = rng.normal(0, 0.3, 5)
        on = rng.random(5) < 0.7
        if not on.any():
            on[rng.integers(5)] = True
        self.amp = scale * A_REF * on * np.exp(
            rng.uniform(np.log(0.03), 0.0, 5))
        # extra parts
        self.thr = None
        if "threshold" in force or ("threshold" not in exclude
                                    and rng.random() < 0.3):
            j = rng.choice(self.act)
            self.thr = (j, rng.uniform(-1.5, 1.5), rng.uniform(5, 30),
                        rng.normal(0, 1.0, 5))
        self.hyst = None
        if "hysteresis" in force or ("hysteresis" not in exclude
                                     and rng.random() < 0.15):
            j = int(rng.choice(SLOW))
            lo = rng.uniform(-0.8, 0.3)
            self.hyst = (j, lo, lo + rng.uniform(0.2, 0.6),
                         rng.normal(0, 1.0, 5))
        self.relay = 0.0
        self.toggles, self.on_steps, self.n_steps = 0, 0, 0
        self.drift = None
        if "drift" in force or ("drift" not in exclude
                                and rng.random() < 0.3):
            self.drift = (rng.uniform(20, 60), rng.uniform(0.2, 0.8))
        self.d = 0.0
        self.style = dict(inputs=[INPUTS[j] for j in self.act],
                          smooth=float(smooth),
                          channels=[CHANNELS[c] for c in np.flatnonzero(on)],
                          threshold=self.thr is not None,
                          hysteresis=self.hyst is not None,
                          drift=self.drift is not None)

    def _z(self, x):
        return (np.asarray(x, float) - self.x0) / self.xs

    def value(self, x, relay=None, d=None):
        """Residual at inputs x (..., 9), for given memory states (the
        stateless part of the function when relay/d are left None)."""
        z = self._z(x)
        h = np.tanh(z @ self.W1.T + self.b1)
        o = h @ self.W2.T + self.b2
        if self.thr is not None:
            j, c, k, g = self.thr
            o = o + (1.0 / (1.0 + np.exp(-k * (z[..., j] - c))))[..., None] * g
        if self.hyst is not None:
            r = self.relay if relay is None else relay
            o = o + np.asarray(r)[..., None] * self.hyst[3]
        amp = self.amp * (1.0 + (self.d if d is None else d))
        return amp * np.tanh(o)

    def step(self, x, dt):
        """Advance the memory (relay, drift) and return the residual."""
        if self.hyst is not None:
            j, lo, hi, _ = self.hyst
            zj = self._z(x)[j]
            old = self.relay
            if zj > hi:
                self.relay = 1.0
            elif zj < lo:
                self.relay = 0.0
            self.toggles += int(self.relay != old)
            self.on_steps += int(self.relay > 0.5)
            self.n_steps += 1
        if self.drift is not None:
            tau, sig = self.drift
            self.d += -self.d * dt / tau + sig * np.sqrt(2 * dt / tau) \
                * self.rng.normal()
            self.d = float(np.clip(self.d, -0.9, 1.5))
        return self.value(x)


class Injector:
    """Applies a ResidualFunction to a plant after every plant step (the
    Mission 'residual' hook): velocity kicks in the five channels, and the
    heave part added to the measured CG acceleration."""

    def __init__(self, fn, u_ref, L, t_max, rud_max):
        self.fn = fn
        self.args = (u_ref, L, t_max, rud_max)
        self.sum, self.n = np.zeros(5), 0

    def take_mean(self):
        """Mean residual since the last call (one control step)."""
        m = self.sum / max(self.n, 1)
        self.sum, self.n = np.zeros(5), 0
        return m

    def __call__(self, s, dt, plant=None):
        a = self.fn.step(inputs_of(s, *self.args), dt)
        s = s.copy()
        for c, i in enumerate(VEL_IDX):
            s[i] += a[c] * dt
        if plant is not None:
            plant.last_cg_acc = float(plant.last_cg_acc + a[3])
        self.last = a
        self.sum += a
        self.n += 1
        return s
