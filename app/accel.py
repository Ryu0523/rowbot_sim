"""
Acceleration outputs of the frozen simulator configuration
(learn/meta/SIM_FINAL_2026-10-09.md section 6): the causal low-pass filter
the environment applies before the CG and bow vertical accelerations enter
the observation and the record, the zero-phase filter for offline
statistics, the bow-gauge definition, and the statistics themselves.

FILTER   2nd-order Butterworth low-pass, cut-off FC_HZ = 20 Hz at full
         scale (Froude-scaled to a model hull by sqrt(b_full / b_model)).
         Online (the observation): the causal form, CausalLowpass -- one
         biquad in direct form II transposed, its coefficients from the
         bilinear transform in closed form (equal to scipy.signal.butter's
         to round-off, checked in studies/app_randomization_test.py), the
         state started at the first sample's steady state so there is no
         start-up transient. Offline (statistics): scipy's filtfilt (zero
         phase), exactly studies/derive_v6_substep.lowpass. A cut-off at or
         above the Nyquist frequency passes the series unchanged (the
         harness's rule).
BOW      the gauge 0.1 L aft of the stem, linearly interpolated between the
         CG and the plant's bow station (plant.x[-1], the stem):
             a_bow = a_cg + (a_station - a_cg) * x_gauge / x_station,
             x_gauge = (L - lcg) - 0.1 L   (body x from the CG).
         The vertical component is used (the component normal to the keel
         differs by less than 0.8 %).
STATS    regular waves: the maximum per encounter cycle, averaged over the
         cycles after the dropped start (regular_stats). Irregular waves:
         one peak per positive excursion (the signal rising through zero to
         falling back through it; studies/exp_planing.excursion_peaks, read
         here unchanged), their mean ("mean of all positive peaks", Fridsma
         1971), the mean of the highest tenth (A1/10), the maximum and the
         count (irregular_stats); event rates, peaks above a threshold per
         hour (exceedance_per_hour). All of them on the FILTERED series, the
         first DROP = 10 % of the run removed.
"""
import math

import numpy as np

G = 9.81
FC_HZ = 20.0            # full-scale cut-off
ORDER = 2
DROP = 0.1              # the share of a run's start left out of statistics
BOW_GAUGE = 0.1         # the bow gauge, x L aft of the stem


# ------------------------------------------------------------- the filter
def butter2_lowpass(fc, fs):
    """(b, a) of the 2nd-order Butterworth low-pass with cut-off fc at
    sampling rate fs, bilinear transform in closed form (a[0] = 1); None
    when the cut-off is at or above 0.999 of the Nyquist frequency (the
    filter is then the identity)."""
    if fc / (0.5 * fs) >= 0.999:
        return None
    K = math.tan(math.pi * fc / fs)
    n = 1.0 / (1.0 + math.sqrt(2.0) * K + K * K)
    b0 = K * K * n
    b = np.array([b0, 2.0 * b0, b0])
    a = np.array([1.0, 2.0 * (K * K - 1.0) * n,
                  (1.0 - math.sqrt(2.0) * K + K * K) * n])
    return b, a


class CausalLowpass:
    """The online filter: y = lowpass(x) sample by sample, x a scalar or a
    vector of `n` channels filtered independently. The state starts at the
    steady state of the first sample (scipy's lfilter_zi * x0), so a series
    that begins at a constant comes out unchanged."""

    def __init__(self, fc=FC_HZ, fs=100.0, n=1):
        self.fc, self.fs, self.n = float(fc), float(fs), int(n)
        ba = butter2_lowpass(self.fc, self.fs)
        self.identity = ba is None
        if not self.identity:
            self.b, self.a = ba
        self.z = None

    def reset(self):
        self.z = None

    def __call__(self, x):
        x = np.asarray(x, float)
        if self.identity:
            return x.copy()
        b, a = self.b, self.a
        if self.z is None:
            # steady state for a constant input x (unit DC gain)
            self.z = np.array([(b[1] + b[2] - a[1] - a[2]) * x,
                               (b[2] - a[2]) * x])
        z0, z1 = self.z
        y = b[0] * x + z0
        self.z = np.array([b[1] * x - a[1] * y + z1, b[2] * x - a[2] * y])
        return y

    def state(self):
        return None if self.z is None else self.z.copy()


def lowpass_offline(y, dt, fc=FC_HZ):
    """Zero-phase low-pass for statistics (scipy filtfilt, order 2), as
    studies/derive_v6_substep.lowpass: identity above the Nyquist rule."""
    from scipy.signal import butter, filtfilt
    y = np.asarray(y, float)
    fs = 1.0 / dt
    wn = fc / (0.5 * fs)
    if wn >= 0.999:
        return y.copy()
    b, a = butter(ORDER, wn)
    return filtfilt(b, a, y, axis=0)


def model_scale_fc(fc_full, b_full, b_model):
    """The cut-off for a Froude-scaled model hull: fc x sqrt(b_full /
    b_model) (59.5 Hz for Fridsma's 9-inch model at 20 Hz full scale)."""
    return fc_full * math.sqrt(b_full / b_model)


# ------------------------------------------------------------- the gauge
def gauge_x(hull, frac=BOW_GAUGE):
    """Body x (from the CG) of the bow gauge, `frac` L aft of the stem."""
    return float(hull.L - hull.lcg - frac * hull.L)


def bow_gauge(a_cg, a_station, x_gauge, x_station):
    """The gauge's acceleration from the CG's and the bow station's (linear
    in x)."""
    return a_cg + (a_station - a_cg) * (x_gauge / x_station)


# ------------------------------------------------------------- statistics
def drop_start(y, frac=DROP):
    """The series without its first `frac` (the harness: acc[n // 10:])."""
    y = np.asarray(y)
    return y[int(len(y) * frac):]


def excursion_peaks(y):
    """One peak per positive excursion: the largest value between the
    signal rising through zero and falling back through it (studies/
    exp_planing.excursion_peaks; complete excursions only). Counting every
    local maximum instead also counts the ripples inside an excursion and
    halved the mean against Fridsma's. Empty when there is none (the
    harness returns a single zero there)."""
    y = np.asarray(y, float)
    pos = y > 0
    edges = np.flatnonzero(np.diff(pos.astype(int)))
    starts = edges[pos[edges + 1]] + 1 if len(edges) else np.array([], int)
    out = []
    for st in starts:
        end = st
        while end < len(y) and pos[end]:
            end += 1
        if end < len(y):                         # complete excursions only
            out.append(y[st:end].max())
    return np.asarray(out, float)


def peak_stats(y):
    """mean of the positive-excursion peaks, the mean of their highest
    tenth (a110), the maximum of the series and the number of peaks."""
    y = np.asarray(y, float)
    pk = np.sort(excursion_peaks(y))[::-1]
    n = len(pk)
    if n == 0:
        return dict(mean=float("nan"), a110=float("nan"),
                    max=float(y.max()) if len(y) else float("nan"), n=0)
    n10 = max(1, n // 10)
    return dict(mean=float(pk.mean()), a110=float(pk[:n10].mean()),
                max=float(y.max()), n=int(n))


def irregular_stats(q, drop=DROP, names=("cg", "bow")):
    """Statistics of the irregular-wave reading on each column of q
    (n, m) -- already filtered -- the first `drop` of the run left out:
    {name_mean, name_a110, name_max, name_n}."""
    q = np.atleast_2d(np.asarray(q, float))
    if q.shape[0] == 1 and q.shape[1] != len(names):
        q = q.T
    w = drop_start(q, drop)
    out = {}
    for j, nm in enumerate(names):
        st = peak_stats(w[:, j])
        out.update({f"{nm}_mean": st["mean"], f"{nm}_a110": st["a110"],
                    f"{nm}_max": st["max"], f"{nm}_n": st["n"]})
    return out


def regular_stats(q, dt, om_e, drop=0.5, names=("cg", "bow")):
    """Regular waves: the maximum per encounter cycle (period 2 pi / om_e),
    averaged over the cycles after the dropped start (the harness drops the
    first half of a 16-s run), and the maximum: {name_cycmean, name_max,
    n_cyc}."""
    q = np.atleast_2d(np.asarray(q, float))
    w = drop_start(q, drop)
    te = 2.0 * math.pi / om_e
    m = max(int(round(te / dt)), 1)
    cyc = [w[j:j + m] for j in range(0, len(w) - m + 1, m)]
    out = dict(n_cyc=len(cyc))
    for j, nm in enumerate(names):
        pk = np.array([c[:, j].max() for c in cyc]) if cyc else np.array([])
        out[f"{nm}_cycmean"] = float(pk.mean()) if len(pk) else float("nan")
        out[f"{nm}_max"] = float(w[:, j].max()) if len(w) else float("nan")
    return out


def exceedance_per_hour(peaks, threshold_g, duration_s):
    """Peaks above `threshold_g` per hour of run (events such as 'peaks
    over 7 g per hour' are counted on the excursion peaks)."""
    peaks = np.asarray(peaks, float)
    if duration_s <= 0:
        return float("nan")
    return float((peaks > threshold_g).sum() * 3600.0 / duration_s)


# ------------------------------------------------------------- records
def gauge_series(rec):
    """(t, q): the RAW CG and bow-gauge vertical accelerations of a record
    in g, (n, 2), the gauge interpolated from the recorded CG and bow
    station values with the header's x_bow_gauge / x_bow_station."""
    h, P = rec.header, rec.plant
    xg, xs = h.get("x_bow_gauge"), h.get("x_bow_station")
    a_cg = np.asarray(P["a_cg"], float)
    a_st = np.asarray(P["a_bow"], float)
    if xg is None or xs is None:
        raise KeyError("the record's header carries no x_bow_gauge / "
                       "x_bow_station (made before the output spec)")
    bow = bow_gauge(a_cg, a_st, float(xg), float(xs))
    return np.asarray(P["t"], float), np.column_stack([a_cg, bow]) / G


def filtered_series(rec, fc=None):
    """(t, q): gauge_series low-passed offline (zero phase) at the record's
    filter cut-off (header filter_hz) or `fc`, with the record's plant
    step."""
    t, q = gauge_series(rec)
    fc = float(rec.header.get("filter_hz", FC_HZ)) if fc is None else fc
    return t, lowpass_offline(q, float(rec.header["dt"]), fc)


def record_stats(rec, drop=DROP, fc=None):
    """irregular_stats of a record's filtered CG and bow-gauge series, with
    the unfiltered values beside them (keys raw_*)."""
    t, q = gauge_series(rec)
    dt = float(rec.header["dt"])
    fc = float(rec.header.get("filter_hz", FC_HZ)) if fc is None else fc
    out = irregular_stats(lowpass_offline(q, dt, fc), drop)
    out.update({f"raw_{k}": v for k, v in irregular_stats(q, drop).items()})
    out.update(dt=dt, fc_hz=fc, n_steps=int(len(t)),
               t_span=float(t[-1] - t[0]) if len(t) > 1 else 0.0)
    return out
