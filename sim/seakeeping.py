#!/usr/bin/env python3
"""
Seakeeping criteria: what to measure, and why the count was the wrong choice.

Everything a preview controller could improve is downstream of ONE signal --
the relative vertical motion at the bow,

    r(t) = (wave surface at the bow)  -  (hull's own vertical position there)

Slamming, deck wetness, propeller emergence and added resistance are all
threshold or quadratic functionals of r and its derivative. Measuring them as
EVENT COUNTS throws away almost everything the run contained: a 200 s run holds
4000 samples of r, and reduces to one or two slams. That is the whole of
DEFECTS.md A22 in one sentence.

The fix is Ochi's, and it is older than the mistake. For a narrow-band Gaussian
relative motion, the slamming rate does not have to be counted -- it follows
from two variances:

    P(slam)   = exp[ -T^2/(2 s_r^2)  -  v_cr^2/(2 s_rdot^2) ]
    N_slam    = (1/2pi)(s_rdot/s_r) . P(slam)

the first exponent being the chance the bow is out of the water (relative
motion exceeds the draft T), the second the chance it re-enters faster than the
critical velocity, and the prefactor the mean up-crossing rate of r. The inputs
s_r and s_rdot are STANDARD DEVIATIONS of a continuous signal: they converge
like 1/sqrt(N_samples), not like 1/sqrt(N_events). Same physics, same
definition of a slam, thousands of times more statistical power.

The same construction gives deck wetness (exceed the freeboard instead of the
draft, with no velocity condition) and any other threshold criterion.

WHAT PREVIEW CAN ACTUALLY ACT ON. r(t) has two timescales, and they demand
completely different hardware:

  * the CARRIER, at the encounter period (7.5 s here). Acting on it means
    changing the hull's response inside a single wave, which needs an actuator
    with tau << T_e/4. Thrust has tau_u = 6.3 s and cannot.
  * the ENVELOPE, the slowly varying amplitude of r over a wave group (50-100 s).
    Every criterion above depends on the envelope through s_r and s_rdot, both
    of which are quadratic in it -- and thrust IS fast enough for this.

That distinction also decides the SENSOR. Riding the carrier needs the phase of
individual waves resolved hundreds of metres ahead. Riding the envelope needs
only how big the next group is, which is a far coarser measurement of the same
sea. `envelope` below extracts it.
"""
import numpy as np

G = 9.81


def envelope(x, dt, cutoff=0.06):
    """Slowly varying amplitude of a wave-like signal, by Hilbert transform.

    |analytic signal| gives the instantaneous amplitude, which still carries
    ripple at the carrier frequency; a low-pass at `cutoff` Hz (about 17 s,
    well below a wave group and well above a wave) leaves the group envelope.
    Done with an FFT because the signal is a whole run, not a stream.
    """
    x = np.asarray(x, float)
    n = x.size
    if n < 8:
        return np.abs(x)
    X = np.fft.fft(x - x.mean())
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    a = np.abs(np.fft.ifft(X * h))
    A = np.fft.rfft(a)
    f = np.fft.rfftfreq(n, dt)
    A[f > cutoff] = 0.0
    return np.fft.irfft(A, n) + np.abs(x).mean() * 0.0


def spectral_moments(r, dt):
    """(s_r, s_rdot) from the signal itself. The derivative is taken in the
    FREQUENCY domain rather than by differencing: a first difference amplifies
    the numerical noise at the top of the band, which lands straight in the
    exponent of the Ochi formula, where it does the most damage."""
    r = np.asarray(r, float)
    r = r - r.mean()
    n = r.size
    if n < 8:
        return float(np.std(r)), 0.0
    R = np.fft.rfft(r)
    w = 2 * np.pi * np.fft.rfftfreq(n, dt)
    rdot = np.fft.irfft(1j * w * R, n)
    return float(np.std(r)), float(np.std(rdot))


def ochi_rate(s_r, s_rdot, threshold, v_cr=None):
    """Expected threshold-crossing rate per MINUTE, from two variances.

    With `v_cr` this is Ochi's slamming rate: the bow must both emerge past
    `threshold` (its draft) and re-enter faster than `v_cr`. Without it, it is
    the plain Rice up-crossing rate of a level -- deck wetness if the threshold
    is the freeboard, propeller emergence if it is the shaft immersion.
    """
    if s_r <= 1e-9 or s_rdot <= 1e-9:
        return 0.0
    e = threshold ** 2 / (2 * s_r ** 2)
    if v_cr is not None:
        e += v_cr ** 2 / (2 * s_rdot ** 2)
    return float(60.0 / (2 * np.pi) * (s_rdot / s_r) * np.exp(-e))


def fatigue_damage(x, dt, exponent=3.0):
    """Palmgren-Miner-style damage from the acceleration record.

    sum |range|^m over half-cycles. It sits between an RMS and an event count:
    the exponent makes the big excursions dominate, the way real structural
    damage does, but every sample contributes, so the statistic still
    converges. Reported per minute so run lengths compare.
    """
    x = np.asarray(x, float)
    if x.size < 3:
        return 0.0
    # turning points, then the range between consecutive ones
    d = np.diff(x)
    s = np.sign(d)
    turn = np.where(np.diff(s) != 0)[0] + 1
    if turn.size < 2:
        return 0.0
    rng = np.abs(np.diff(x[turn]))
    return float(np.sum(rng ** exponent) / (x.size * dt / 60.0))


def summarise(rel, acc, dt, draft, freeboard, v_slam):
    """Every criterion this project can support, from two recorded signals.

    `rel` is the relative water elevation at the bow, `acc` the bow vertical
    acceleration in g. Returns continuous statistics first -- those are the
    ones that can carry a conclusion -- with the analytic event rates derived
    from them rather than counted.
    """
    s_r, s_rdot = spectral_moments(rel, dt)
    env = envelope(rel, dt)
    return dict(
        # --- the parent signal, continuous -------------------------------
        rvm_rms=s_r,                       # relative vertical motion, m
        rvv_rms=s_rdot,                    # relative vertical velocity, m/s
        env_p90=float(np.percentile(env, 90)),      # how big the big groups are
        env_ratio=float(np.percentile(env, 90)
                        / max(np.percentile(env, 30), 1e-9)),
        # --- comfort / equipment loading, continuous ----------------------
        acc_rms=float(np.sqrt(np.mean(np.square(acc)))),
        acc_p99=float(np.percentile(np.abs(acc), 99)),
        fatigue=fatigue_damage(acc, dt),
        # --- event rates, ANALYTIC rather than counted --------------------
        slam_rate_ochi=ochi_rate(s_r, s_rdot, draft, v_slam),
        wet_rate=ochi_rate(s_r, s_rdot, freeboard),
    )
