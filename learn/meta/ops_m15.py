#!/usr/bin/env python3
"""
The M15 operator family (DEFECTS M16, whose code, family tag and cache
keep the name m15 / meta5; brief BRIEF_PREVIEW.md, items 1-2): the v0
family of learn/meta/operators.py with two changes, and nothing else.

1. Wave TIMING (outside the operator, in learn/meta/episode5.py): the 15 wave
   inputs (columns 11:26 of the operator's 28) are the elevations the hull
   meets DURING the step, W_MID: the MPC's 5 x 3 stations at the step's
   midpoint t + dtc / 2, at the pose dead-reckoned from the step-start state
   (data5.mid_pose). The operator itself is unchanged: it stays at the
   control-step resolution and its filters carry the effect into later
   steps. The midpoint is enough: averaging the elevations over the step
   instead multiplies a component of encounter frequency w_e by
   sinc(w_e dtc / 2), 0.93 even at 30 kn head seas with Tp 4 s (w_e 5.4
   rad/s at the peak), so substep averaging would change little.

2. Wave PROJECTIONS (this module): every wave reading of the operator -- the
   wave group of a filter's, an own event's, the noise envelope's or the
   bursts' projection, and every direct wave input of the readout -- is a
   random combination of five spatial patterns over the 5 x 3 stations:
   mean, longitudinal gradient, transverse difference, longitudinal
   curvature, twist (transverse difference x position). Each pattern's
   library reading is scaled to unit std SEPARATELY, the four non-mean
   patterns are orthogonalised against the mean on the library (for waves
   long against the hull, curvature and twist are otherwise in phase with
   the mean), then a random subset (each w.p. 0.6, at least one) gets
   N(0, 1) coefficients, normalised to unit norm, and the combination is
   renormalised. So the mean explains on average about a fifth of a
   reading's library variance (test_m15 test 4), instead of most of it
   (the v0 GP-smooth weights read mostly the mean: the stations are nearly
   equal along a 5.4 m hull and only a beam apart). No channel is told
   which pattern to read.

   A projection keeps its wave group's library std (its mixing weight with
   the state / command groups), so which operators read waves and how
   strongly is the v0 draw. Direct wave inputs (v0: one station, i.e. the
   mean) become pattern readings, carried as extra input columns 28.. that
   the operator appends to its inputs itself (inputs stay 28 wide outside).

Pairing: the constructor makes every v0 draw from the operator's stream
exactly as before (the old wave weights are drawn and then replaced), and
the pattern coefficients come from their own stream [seed, 13]. So the
M15 operator of a seed has the v0 operator's amplitudes, shares, filter
kinds, delays, time constants, events, relays, noise and readout draws; its
library constants (filter scales, thresholds, readout fits) follow the new
readings. A relu pre-filter keeps its quantile level (recovered from the
v0 threshold on the v0 library reading).

Style keys added (M15 episodes only, so v0 metas and step-3 stamps stay as
they were): op_family, delay and feed (per filter: its delay and whether
it has same-step feedthrough, 'none' section or d != 0), dir_wave (direct
wave readings), ev_wave, env_wave, burst_wave, reads_waves (any path),
wave_same_step (a rule path from the current step's waves: an undelayed
wave filter with feedthrough, a direct wave input, or an own event whose
section has feedthrough) and mean_share (the mean pattern's share of the
library variance of every wave reading).
"""
import numpy as np

from learn.meta.operators import G_WAVE, N_IN, N_LAT, N_ST, Operator

PATTERNS = ("mean", "long", "trans", "curv", "twist")
P_ON = 0.6                # chance that a pattern takes part in a reading
PAT_STREAM = 13           # the pattern coefficients' stream [seed, 13]
# Same-step paths (user decision 2026-09-30): about half of the wave-reading
# filters respond to the step's own waves. The v0 draw gives it to about
# P(delay 0) x P(feedthrough) = 0.4 x (0.15 + 0.85 x 0.5) = 0.23 of them; a
# wave-reading filter without one gets it w.p. P_SAME (delay 0, and a
# feedthrough d ~ N(0, 1) if its section had none), from its own stream
# [seed, 14], so every other draw stays paired with v0.
SAME_TARGET = 0.5
SAME_STREAM = 14
P_SAME = (SAME_TARGET - 0.4 * 0.575) / (1.0 - 0.4 * 0.575)


def pattern_basis():
    """(5, 15) raw spatial patterns over the stations (stern to bow i,
    port / centre / starboard j, flattened i * 3 + j as data2._stations):
    mean, gradient along the hull, transverse difference, curvature along
    the hull, twist. The stations are evenly spaced, so positions are
    normalised to [-1, 1] along and across."""
    xc = np.linspace(-1.0, 1.0, N_ST)
    yc = np.linspace(-1.0, 1.0, N_LAT)
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    return np.stack([np.ones_like(X), X, Y, X ** 2 - (X ** 2).mean(),
                     X * Y]).reshape(5, N_ST * N_LAT)


def pattern_projections(Sf, sd):
    """Q (5, N_IN): projections of the normalised inputs whose library
    readings are the five patterns of the raw elevations, each centred
    reading of unit std on the library rows Sf (N, >= N_IN), the four
    non-mean ones orthogonalised against the mean reading."""
    Sf = np.asarray(Sf, float)[:, :N_IN]
    Q = np.zeros((5, N_IN))
    # a pattern P of the raw elevations eta_c = sd_c z_c + mu_c reads
    # (P * sd) . z up to a constant
    Q[:, list(G_WAVE)] = pattern_basis() * np.asarray(sd, float)[
        list(G_WAVE)][None]
    R = Sf @ Q.T
    R = R - R.mean(0)
    s = np.maximum(R.std(0), 1e-12)
    Q, R = Q / s[:, None], R / s
    for m in range(1, 5):
        beta = (R[:, m] @ R[:, 0]) / max(R[:, 0] @ R[:, 0], 1e-12)
        Q[m] = Q[m] - beta * Q[0]
        R[:, m] = R[:, m] - beta * R[:, 0]
        sm = max(R[:, m].std(), 1e-12)
        Q[m], R[:, m] = Q[m] / sm, R[:, m] / sm
    return Q


def draw_pattern(rng, Q, Sf):
    """One random wave reading (N_IN,), unit std on the library rows Sf:
    a random subset of the patterns with N(0, 1) coefficients."""
    c = rng.normal(0.0, 1.0, 5) * (rng.random(5) < P_ON)
    if not c.any():
        c[rng.integers(5)] = 1.0
    c = c / np.sqrt((c ** 2).sum())
    p = c @ Q
    return p / max((np.asarray(Sf, float)[:, :N_IN] @ p).std(), 1e-12)


def mean_share(p, Q, Sf):
    """The mean pattern's share of the library variance of the wave part of
    projection p: corr(wave reading, mean reading)^2."""
    Sf = np.asarray(Sf, float)[:, :N_IN]
    pw = np.zeros(N_IN)
    pw[list(G_WAVE)] = np.asarray(p)[list(G_WAVE)]
    a, b = Sf @ pw, Sf @ Q[0]
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) ** 2 / max((a @ a) * (b @ b), 1e-30))


class OperatorM15(Operator):
    """An operator of the M15 family (module docstring). Same constructor
    and methods as Operator; step / new_state / run take the 28 raw inputs
    (with W_MID in columns 11:26, data5.op_inputs5) and append the direct
    wave readings themselves."""

    def __init__(self, seed, lib, dt=0.24, L=5.4, relay=False, null=None):
        self._pat_rng = np.random.default_rng([int(seed), PAT_STREAM])
        self._same_rng = np.random.default_rng([int(seed), SAME_STREAM])
        super().__init__(seed, lib, dt=dt, L=L, relay=relay, null=null)
        del self._pat_rng, self._same_rng
        self._style_m15()

    # ------------------------------------------------ inputs
    def _ext(self, s):
        """28 raw inputs -> 28 + n_x (the direct wave readings appended, in
        normalised units: their mu 0, sd 1). Rows already extended pass."""
        s = np.asarray(s, float)
        if s.shape[-1] == self.mu.size:
            return s
        z = (s[..., list(G_WAVE)] - self.mu[list(G_WAVE)]) \
            / self.sd[list(G_WAVE)]
        return np.concatenate([s, z @ self.Wdir], -1)

    def new_state(self, B=1, rng=None, init_s=None):
        if init_s is not None:
            init_s = self._ext(np.atleast_2d(init_s))
        return super().new_state(B, rng=rng, init_s=init_s)

    def step(self, st, s_raw, noise=True, rng=None, freeze=False,
             hold_relays=False):
        return super().step(st, self._ext(np.atleast_2d(s_raw)), noise=noise,
                            rng=rng, freeze=freeze, hold_relays=hold_relays)

    # ---------------------------------------- the pattern readings
    def _repl(self, p, Sf, Q):
        """p with its wave part replaced by a pattern reading of the same
        library std (p unchanged if it reads no waves)."""
        p = np.asarray(p, float).copy()
        w = list(G_WAVE)
        if not np.abs(p[w]).sum() > 0:
            return p, None
        pw = np.zeros(N_IN)
        pw[w] = p[w]
        s_old = (Sf @ pw).std()
        new = draw_pattern(self._pat_rng, Q, Sf)
        p[w] = new[w] * s_old
        return p, mean_share(p, Q, Sf)

    def _normalise(self, S):
        """Replace every wave reading by a pattern reading (module
        docstring), extend the inputs by the direct wave readings, then
        the v0 normalisation on the extended library."""
        S = np.asarray(S, float)
        Sf = S.reshape(-1, S.shape[-1])
        Q = pattern_projections(Sf, self.sd)
        self._Q = Q
        shares = []
        for f in self.filters:
            p_old = f["p"]
            f["p"], sh = self._repl(p_old, Sf, Q)
            if sh is None:
                continue
            shares.append(sh)
            v_old = Sf @ p_old
            v_new = Sf @ f["p"]
            f["v_mu"], f["v_sd"] = float(v_new.mean()), float(
                v_new.std() + 1e-9)
            if f["pre"][0] == "relu":
                vs = np.sort(v_old)
                q = float(np.interp(f["pre"][1], vs,
                                    np.arange(len(vs)) / (len(vs) - 1.0)))
                f["pre"] = ("relu", float(np.quantile(v_new, q)))
        # same-step paths for about half of the wave-reading filters
        # (SAME_TARGET), before the v0 normalisation fits their scales
        for f in self.filters:
            if not np.abs(np.asarray(f["p"])[list(G_WAVE)]).sum() > 0:
                continue
            sec = f["sec"]
            same = f["delay"] == 0 and (sec.kind == "none" or sec.d != 0.0)
            if not same and self._same_rng.random() < P_SAME:
                f["delay"] = 0
                if sec.kind != "none" and sec.d == 0.0:
                    sec.d = float(self._same_rng.normal())
        for i, ev in enumerate(self.events):
            if ev["src"][0] == "own":
                p, sh = self._repl(ev["src"][1], Sf, Q)
                ev["src"] = ("own", p, ev["src"][2])
                if sh is not None:
                    shares.append(sh)
        for part in (self.env, self.bursts):
            if part is not None:
                part["p"], sh = self._repl(part["p"], Sf, Q)
                if sh is not None:
                    shares.append(sh)
        # direct wave inputs: one pattern reading per distinct v0 station
        wave_dirs = [int(d) for d in self.dirs if int(d) in G_WAVE]
        n_x = len(wave_dirs)
        Wdir = np.zeros((len(G_WAVE), n_x))
        for j in range(n_x):
            p = draw_pattern(self._pat_rng, Q, Sf)
            Wdir[:, j] = p[list(G_WAVE)]
            shares.append(mean_share(p, Q, Sf))
        self.Wdir = Wdir
        # the readout's column order is kept: a wave station's slot now
        # holds its pattern reading
        col = {d: N_IN + j for j, d in enumerate(wave_dirs)}
        self.dirs = np.array([col.get(int(d), int(d)) for d in self.dirs],
                             int)
        self.n_x = n_x
        self._shares = shares
        # extend the inputs, the normalisation and every projection
        self.mu = np.concatenate([self.mu, np.zeros(n_x)])
        self.sd = np.concatenate([self.sd, np.ones(n_x)])
        pad = lambda p: np.concatenate([np.asarray(p, float),  # noqa: E731
                                        np.zeros(n_x)])
        for f in self.filters:
            f["p"] = pad(f["p"])
        for ev in self.events:
            if ev["src"][0] == "own":
                ev["src"] = ("own", pad(ev["src"][1]), ev["src"][2])
        for part in (self.env, self.bursts):
            if part is not None:
                part["p"] = pad(part["p"])
        z_w = S[..., list(G_WAVE)]
        S_ext = np.concatenate([S, z_w @ Wdir], -1)
        super()._normalise(S_ext)

    # ------------------------------------------------------ style
    def _style_m15(self):
        w = list(G_WAVE)
        reads = lambda p: bool(np.abs(np.asarray(p)[w]).sum() > 0)  # noqa
        wave_in = [reads(f["p"]) for f in self.filters]
        delay = [int(f["delay"]) for f in self.filters]
        feed = [bool(f["sec"].kind == "none" or f["sec"].d != 0.0)
                for f in self.filters]
        ev_wave, ev_same = 0, False
        for ev in self.events:
            if ev["src"][0] == "own":
                if reads(ev["src"][1]):
                    ev_wave += 1
                    sec = ev["src"][2]
                    ev_same |= bool(sec.kind == "none" or sec.d != 0.0)
            elif wave_in[ev["src"][1]]:
                ev_wave += 1
        env_wave = self.env is not None and reads(self.env["p"])
        burst_wave = self.bursts is not None and reads(self.bursts["p"])
        same = any(a and d == 0 and fd for a, d, fd in
                   zip(wave_in, delay, feed)) or self.n_x > 0 or ev_same
        self.style.update(
            op_family="m15", delay=delay, feed=feed, dir_wave=int(self.n_x),
            ev_wave=int(ev_wave), env_wave=bool(env_wave),
            burst_wave=bool(burst_wave),
            reads_waves=bool(any(wave_in) or self.n_x > 0 or ev_wave > 0
                             or env_wave or burst_wave),
            wave_same_step=bool(same and not self.null),
            mean_share=[float(x) for x in self._shares])
