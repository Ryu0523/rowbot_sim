#!/usr/bin/env python3
"""
Statistics for the achievable-envelope study (studies/envelope_pass1.py):
pooling of the per-run peak lists, the generalised Pareto (GPD) tail fit of
the peak exceedances with a bootstrap interval, the event rates per hour,
and the envelope read-off.

TAIL EXTRAPOLATION. Per condition the positive-excursion peaks of the
filtered acceleration (app/accel.excursion_peaks) of the two wave seeds
are pooled (T_pool seconds of statistics). The threshold u is the q-th
quantile of the pooled peaks (q = 0.9 by default); the exceedances
y = peak - u > 0 are fitted with a GPD, survival (1 + xi y / sigma)^(-1/xi)
(xi -> 0: exp(-y / sigma); xi < 0: zero beyond u - sigma / xi), by the
probability-weighted-moment estimator of Hosking & Wallis (1987), which is
closed form and well behaved for a few dozen exceedances (the maximum-
likelihood estimate is computed once beside it as a check). The rate of
peaks at or above a level x per hour is

    rate(x) = (n_exc / T_pool) * 3600 * S(x - u),      x >= u,

and below the threshold the direct count. The uncertainty is a bootstrap:
n_boot resamples of the exceedances with replacement, the count drawn
from a Poisson with mean n_exc (the events are a point process), refit,
rate recomputed; the 5-95 % and 2.5-97.5 % percentiles are reported.

A correction factor k multiplies the simulated accelerations before the
thresholds (SIM_FINAL: the plant's filtered impact statistics run about
0.84 of Fridsma's irregular-wave data), so rate_k(x) = rate(x / k) and
A1/10_k = k A1/10.

DIRECT COUNTS. Events observed in T seconds: rate = n * 3600 / T; with
n = 0 only an upper bound (about 3 / T per hour at 95 %, i.e. 9 per hour
for 1200 s) can be stated.
"""
import math

import numpy as np

THRESH_G = (3.0, 5.0, 7.0)
K_FACTORS = (1.0, 1.2)


# --------------------------------------------------------------- the GPD
def gpd_pwm(y):
    """(xi, sigma) of the GPD fitted to exceedances y > 0 by probability-
    weighted moments (Hosking & Wallis 1987, their k = -xi)."""
    y = np.sort(np.asarray(y, float))
    n = len(y)
    if n < 3:
        return float("nan"), float("nan")
    a0 = y.mean()
    i = np.arange(1, n + 1)
    a1 = float(np.sum((n - i) / (n - 1.0) * y) / n)
    den = a0 - 2.0 * a1
    if den <= 0:
        return 1.0, float(a0)           # degenerate: heavy tail cap
    k = a0 / den - 2.0
    sigma = 2.0 * a0 * a1 / den
    xi = -k
    xi = float(min(max(xi, -1.0), 1.0))
    return xi, float(sigma)


def gpd_mle(y, start=None):
    """(xi, sigma) by maximum likelihood (scipy), from the PWM start; NaN
    when the optimiser fails."""
    from scipy.stats import genpareto
    y = np.asarray(y, float)
    if len(y) < 3:
        return float("nan"), float("nan")
    if start is None:
        start = gpd_pwm(y)
    try:
        c, loc, scale = genpareto.fit(y, start[0], floc=0.0,
                                      scale=max(start[1], 1e-6))
        return float(c), float(scale)
    except Exception:           # noqa: BLE001
        return float("nan"), float("nan")


def gpd_survival(y, xi, sigma):
    """P(Y > y) of the GPD, y >= 0 (array ok)."""
    y = np.asarray(y, float)
    if not (np.isfinite(xi) and np.isfinite(sigma)) or sigma <= 0:
        return np.full_like(y, np.nan, dtype=float)
    if abs(xi) < 1e-9:
        return np.exp(-y / sigma)
    z = 1.0 + xi * y / sigma
    out = np.where(z > 0, np.power(np.maximum(z, 1e-300), -1.0 / xi), 0.0)
    return out


def rate_per_hour(x, peaks, T_s, u, xi, sigma):
    """Peaks >= x per hour: the GPD extrapolation above the threshold u,
    the direct count below it."""
    peaks = np.asarray(peaks, float)
    x = float(x)
    if x <= u:
        return float((peaks >= x).sum() * 3600.0 / T_s)
    n_exc = int((peaks > u).sum())
    return float(n_exc / T_s * 3600.0 * gpd_survival(x - u, xi, sigma))


def tail_fit(peaks, T_s, q=0.9, thresholds=THRESH_G, k_factors=K_FACTORS,
             n_boot=200, seed=0):
    """The tail analysis of one condition: dict with the threshold, the
    number of exceedances, xi and sigma (PWM; MLE beside), and per
    (k, level) the point rate per hour, the bootstrap percentiles, the
    direct count and the direct rate."""
    peaks = np.sort(np.asarray(peaks, float))
    n = len(peaks)
    out = dict(n_peaks=int(n), T_s=float(T_s), q=float(q), n_boot=n_boot,
               levels={}, ok=False)
    if n < 20:
        return out
    u = float(np.quantile(peaks, q))
    exc = peaks[peaks > u] - u
    n_exc = int(len(exc))
    xi, sigma = gpd_pwm(exc)
    xi_m, sg_m = gpd_mle(exc, (xi, sigma))
    out.update(u=u, n_exc=n_exc, xi=xi, sigma=sigma, xi_mle=xi_m,
               sigma_mle=sg_m, ok=np.isfinite(xi) and sigma > 0,
               endpoint=(u - sigma / xi) if xi < 0 else float("inf"))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        m = rng.poisson(n_exc)
        if m < 3:
            boots.append((float("nan"), float("nan"), m))
            continue
        yb = rng.choice(exc, size=m, replace=True)
        xb, sb = gpd_pwm(yb)
        boots.append((xb, sb, m))
    for k in k_factors:
        for lev in thresholds:
            x = lev / k                 # the simulated level that reads lev
            pt = rate_per_hour(x, peaks, T_s, u, xi, sigma)
            n_dir = int((peaks * k >= lev).sum())
            if x > u:
                rb = np.array([(m / T_s * 3600.0
                                * float(gpd_survival(x - u, xb, sb)))
                               if np.isfinite(xb) else np.nan
                               for xb, sb, m in boots])
            else:
                rb = np.full(n_boot, pt)
            rb = rb[np.isfinite(rb)]
            pct = (np.percentile(rb, [2.5, 5, 50, 95, 97.5]) if len(rb)
                   else np.full(5, np.nan))
            out["levels"][f"k{k:g}_{lev:g}g"] = dict(
                k=k, level_g=lev, x_sim=x, extrapolated=bool(x > u),
                rate=pt, p2_5=float(pct[0]), p5=float(pct[1]),
                p50=float(pct[2]), p95=float(pct[3]), p97_5=float(pct[4]),
                n_direct=n_dir, rate_direct=float(n_dir * 3600.0 / T_s),
                direct_upper_if_zero=float(3.0 * 3600.0 / T_s))
    # threshold sensitivity of the k = 1 point estimates
    sens = {}
    for qq in (0.8, 0.95):
        uu = float(np.quantile(peaks, qq))
        ee = peaks[peaks > uu] - uu
        xx, ss = gpd_pwm(ee)
        sens[f"q{qq:g}"] = dict(u=uu, n_exc=int(len(ee)), xi=xx, sigma=ss,
                                rates={f"{lev:g}g": rate_per_hour(
                                    lev, peaks, T_s, uu, xx, ss)
                                    for lev in thresholds})
    out["threshold_sensitivity"] = sens
    return out


# ------------------------------------------------------------- pooling
def pool_peaks(peak_lists):
    return np.sort(np.concatenate([np.asarray(p, float) for p in
                                   peak_lists]))[::-1]


def a110(peaks):
    pk = np.sort(np.asarray(peaks, float))[::-1]
    if len(pk) == 0:
        return float("nan")
    return float(pk[:max(1, len(pk) // 10)].mean())


# ------------------------------------------------------------- read-off
def highest_speed(rows, ok):
    """The highest speed (kn) in `rows` (dicts with 'kn', ascending) for
    which ok(row) is True, and whether every lower tested speed also
    passes (monotone). (None, True) when none passes."""
    passing = [r["kn"] for r in rows if ok(r)]
    if not passing:
        return None, True
    top = max(passing)
    lower = [r for r in rows if r["kn"] <= top]
    mono = all(ok(r) for r in lower)
    return top, mono


def read_off(rows, gauge="cg", k=1.0, a110_lim=3.0, rate7_lim=1.0,
             imm_lim=3.0, use_upper=False):
    """Per criterion the highest passing speed: (a) k A1/10 <= a110_lim,
    (b) extrapolated rate of peaks >= 7 g (after k) <= rate7_lim per hour
    (the point estimate, or the 95th percentile when use_upper), (c) bow
    immersions per hour <= imm_lim, and all three together."""
    key = f"k{k:g}_7g"

    def a(r):
        v = r[gauge]["a110"]
        return np.isfinite(v) and k * v <= a110_lim

    def b(r):
        lv = r[gauge]["tail"]["levels"].get(key) if r[gauge]["tail"].get(
            "ok") else None
        if lv is None:
            return False
        v = lv["p95"] if use_upper else lv["rate"]
        return np.isfinite(v) and v <= rate7_lim

    def c(r):
        return r["immersion_per_hour"] <= imm_lim

    def allc(r):
        return a(r) and b(r) and c(r)

    return dict(a110=highest_speed(rows, a), rate7=highest_speed(rows, b),
                immersion=highest_speed(rows, c),
                all=highest_speed(rows, allc))
