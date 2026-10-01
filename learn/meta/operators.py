#!/usr/bin/env python3
"""
The v0 family of error OPERATORS -- the training distribution derived in
learn/meta/PRIOR_DERIVATION.md (components G1-G12, decisions D2 and D4).

An operator maps the history of what the boat measures and commands,

    s_k = [9 inputs of residuals.inputs_of, heave and pitch about the
           running attitude, the 15 wave elevations at the MPC's 5 x 3
           stations, the thrust and nozzle commands applied during the
           step]                                             (N_IN = 28)

(columns 5-6 are the actuator states before the step, 26-27 the commands;
their difference is what a lagging actuator shows)

to an extra acceleration r_k in the five channels (surge, sway, yaw,
heave, pitch), held over one control step. It is split into two parts.

    rule    what follows from the measured past, given the operator:
            - a few random stable linear filters of random projections,
              with a random nonlinearity before them;
            - a random static readout (an MLP, plus an optional bilinear
              gain-scheduling term mixed in at a random share);
            - threshold EVENTS on filtered signals, whose size is a random
              heavy-tailed function of the crossing speed (D2);
            - a slow gain and offset, frozen at a checkpoint when rules
              are relabelled.
    noise   what does not follow from the measured past:
            - random coloured noise (heavy-tailed, with an optional
              state-dependent level);
            - white noise;
            - random BURSTS whose rate depends on the measured state;
            - the random part of event sizes.

    r_k = A (.) a_k (.) (sat(sqrt(pi_m) ybar_k) + sqrt(pi_n) nbar_k)
          + b_k + events_k + bursts_k                 (sat = 2.5 tanh(./2.5))

No mechanism is named. The step-1 family (a static function of the
current state) is the special case of no filters and no wave inputs.

Everything is normalised on a small LIBRARY of recorded input histories
(data2.build_library), so every draw has comparable size before its
amplitude A is applied. The library sets:
  - each input group's part of a projection;
  - the pre-filter thresholds and scales;
  - the filter output scales;
  - the readout thresholds and the steepness of its sigmoids, per library
    std of the unit's input;
  - the readout output scales.

The rule part never draws random numbers after construction, and in closed
loop the slow gain and offset drift on their own random stream. Re-running
an operator on another input history with the slow state frozen
(relabelling) therefore reproduces its rule exactly, and the noise stream
never changes the rule.
The clip at CLIP x A_REF is applied to the rule alone, and the noise part
absorbs the rest of the cut.
"""
import numpy as np

CHANNELS = ("surge", "sway", "yaw", "heave", "pitch")
N_IN = 28
A_REF = np.array([4.0, 4.0, 2.0, 12.0, 3.0])
# the push applied to the boat is clipped per channel at CLIP x A_REF
# (heave 36 m/s^2 = 3.7 g): events and heavy-tailed noise may be large, but
# not beyond what a hull can take
CLIP = 3.0
# input groups (indices into s)
G_STATE = (0, 1, 2, 3, 4, 7, 8, 9, 10)
G_CMD = (5, 6, 26, 27)
G_WAVE = tuple(range(11, 26))
SLOW = (0, 3, 4, 5)                  # speed, cos / sin heading, thrust
N_ST, N_LAT = 5, 3


def clip_push(rule, noise):
    """Clip the rule alone at CLIP x A_REF, and the total push the same way:
    the noise part takes the rest, so the rule never depends on the noise
    and the applied push is clip(rule + noise)."""
    lim = CLIP * A_REF
    rc = np.clip(rule, -lim, lim)
    return rc, np.clip(rule + noise, -lim, lim) - rc


def _logu(rng, lo, hi, size=None):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


class _Sec:
    """A stable discrete-time section, batched over rows.

    kind  'real': x' = a x + (1 - a) v,  y = c x + d v
          'res' : x' = r R(theta) x + [1, 0] v,  y = c . x + d v
          'none': y = v
    Optionally SCHEDULED by a slow signal sb (standardised): a -> a^e,
    r -> r^e, theta -> theta e with e = exp(kappa sb). ||A|| = r < 1 at
    every step, so the time-varying section stays exponentially stable."""

    def __init__(self, kind, dt, rng, tau=None, wn=None, zeta=None,
                 kappa=0.0, d_prob=0.5):
        self.kind, self.kappa = kind, float(kappa)
        if kind == "real":
            self.a = np.exp(-dt / tau)
            self.c = float(rng.normal())
            self.tau_eff = tau
        elif kind == "res":
            zeta = min(zeta, 0.99)
            self.r = np.exp(-zeta * wn * dt)
            self.th = min(wn * np.sqrt(1 - zeta ** 2) * dt, 0.97 * np.pi)
            self.c = rng.normal(0, 1, 2)
            self.tau_eff = 1.0 / (zeta * wn)
        else:
            self.tau_eff = 0.0
        self.d = float(rng.normal()) if (kind != "none"
                                         and rng.random() < d_prob) else 0.0
        self.n = {"real": 1, "res": 2, "none": 0}[kind]

    def step(self, x, v, sb=None):
        """x (B, n) state, v (B,) input -> (y (B,), new x)."""
        if self.kind == "none":
            return v, x
        e = 1.0 if (self.kappa == 0.0 or sb is None) else np.exp(
            self.kappa * sb)
        if self.kind == "real":
            a = self.a ** e
            y = self.c * x[:, 0] + self.d * v
            return y, (a * x[:, 0] + (1 - a) * v)[:, None]
        r = self.r ** e
        th = np.minimum(self.th * e, 0.97 * np.pi)
        cs, sn = r * np.cos(th), r * np.sin(th)
        y = x @ self.c + self.d * v
        x0 = cs * x[:, 0] - sn * x[:, 1] + v
        x1 = sn * x[:, 0] + cs * x[:, 1]
        return y, np.stack([x0, x1], 1)

    def impulse(self, n):
        """Unscheduled impulse response (n,)."""
        x = np.zeros((1, self.n))
        out = np.zeros(n)
        for k in range(n):
            y, x = self.step(x, np.array([1.0 if k == 0 else 0.0]))
            out[k] = y[0]
        return out


def _draw_sec(rng, dt, res_p=0.45, real_p=0.40):
    u = rng.random()
    if u < res_p:
        wn = _logu(rng, 1.0, 8.0) if rng.random() < 0.5 else \
            _logu(rng, 0.3, 9.0)
        zeta = _logu(rng, max(0.05, 2.0 / (60.0 * wn)), 1.0)
        return dict(kind="res", wn=wn, zeta=zeta)
    if u < res_p + real_p:
        return dict(kind="real", tau=_logu(rng, 0.1, 30.0))
    return dict(kind="none")


def _projection(rng, L, S):
    """A random sparse projection of the 26 inputs (G3): one input group
    (w.p. 0.7) or two to three, with smooth wave weights along the hull.
    Each group's part is scaled to unit std on the library S (normalised,
    (B, T, 26)) before the groups are mixed with random weights, so no group
    dominates by its own correlation structure (the 15 elevations are
    nearly equal along a 5.4 m hull)."""
    groups = ["wave", "state", "cmd"]
    n_g = 1 if rng.random() < 0.7 else int(rng.integers(2, 4))
    chosen = rng.choice(3, size=n_g, replace=False,
                        p=np.array([0.4, 0.35, 0.25]))
    Sf = S.reshape(-1, N_IN)
    parts = []
    for g in chosen:
        p = np.zeros(N_IN)
        if groups[g] == "wave":
            ell = rng.uniform(0.2, 2.0) * L
            xs = np.linspace(-0.5, 0.5, N_ST) * L
            K = np.exp(-0.5 * ((xs[:, None] - xs[None]) / ell) ** 2)
            Lc = np.linalg.cholesky(K + 1e-6 * np.eye(N_ST))
            if rng.random() < 0.5:
                xi = np.repeat(rng.normal(0, 1, (N_ST, 1)), N_LAT, 1)
            else:
                xi = rng.normal(0, 1, (N_ST, N_LAT))
            p[list(G_WAVE)] = (Lc @ xi).ravel()
        else:
            idx = np.array(G_STATE if groups[g] == "state" else G_CMD)
            on = rng.random(len(idx)) < 0.5
            if not on.any():
                on[rng.integers(len(idx))] = True
            p[idx[on]] = rng.normal(0, 1, on.sum())
        parts.append(p / max((Sf @ p).std(), 1e-9))
    w = rng.normal(0, 1, len(parts))
    w = w / max(np.sqrt((w ** 2).sum()), 1e-9)
    return sum(wi * pi for wi, pi in zip(w, parts))


class Operator:
    """One draw of the family. `lib` = dict(mu, sd, S): the normalisation
    of the raw inputs and a library S (B, T, 26) of NORMALISED histories."""

    def __init__(self, seed, lib, dt=0.24, L=5.4, relay=False, null=None):
        rng = np.random.default_rng(seed)
        self.seed, self.dt, self.L = int(seed), float(dt), float(L)
        self.mu, self.sd = np.asarray(lib["mu"]), np.asarray(lib["sd"])
        S = np.asarray(lib["S"], float)                  # (B, T, 26)
        self.style = {}
        # ---------------------------------------------- amplitude, shares
        self.null = (rng.random() < 0.05) if null is None else bool(null)
        self.A = A_REF * _logu(rng, 0.03, 1.0, 5) * (0.0 if self.null else 1)
        pm, pn = rng.dirichlet([2.0, 1.5])
        sh = np.array([rng.dirichlet(10 * np.array([pm, pn]) + 1e-3)
                       for _ in range(5)])
        self.pi_m, self.pi_n = sh[:, 0], sh[:, 1]
        # ------------------------------------------- filters (G3, G4)
        K = 0 if rng.random() < 0.15 else min(1 + rng.poisson(1.5), 6)
        self.filters = []
        for i in range(K):
            f = dict(p=_projection(rng, self.L, S))
            v_lib = (S @ f["p"]).ravel()
            f["v_mu"], f["v_sd"] = float(v_lib.mean()), float(
                v_lib.std() + 1e-9)
            hu = rng.random()
            if hu < 0.7:
                f["pre"] = ("id",)
            elif hu < 0.85:
                # a random 1-D tanh MLP of the STANDARDISED projection
                f["pre"] = ("mlp", rng.normal(0, 1.5, 8), rng.normal(0, 1, 8),
                            rng.normal(0, 1 / np.sqrt(8), 8))
            else:
                f["pre"] = ("relu", float(np.quantile(
                    v_lib, rng.uniform(0.7, 0.98))))
            f["delay"] = int(rng.choice(4, p=[0.4, 0.3, 0.2, 0.1]))
            spec = _draw_sec(rng, dt)
            kappa, sidx = 0.0, None
            if spec["kind"] != "none" and rng.random() < 0.5:
                kappa = rng.uniform(-0.25, 0.25)
                sidx = int(rng.choice(SLOW))
            f["sec"] = _Sec(spec["kind"], dt, rng, spec.get("tau"),
                            spec.get("wn"), spec.get("zeta"), kappa)
            f["sidx"] = sidx
            f["spec"] = spec
            self.filters.append(f)
        # direct (undelayed) inputs to the readout
        n_dir = int(rng.integers(0, 4))
        if K == 0:
            n_dir = max(n_dir, 1)
        pool = list(G_STATE) + list(G_CMD)
        dirs = [int(rng.choice(G_WAVE)) if rng.random() < 0.3
                else int(rng.choice(pool)) for _ in range(n_dir)]
        self.dirs = np.array(sorted(set(dirs)), int)
        # ------------------------------------------------ events (D2)
        self.events = []
        if rng.random() < 0.5:
            for _ in range(1 if rng.random() < 0.7 else 2):
                ev = {}
                if K > 0 and rng.random() < 0.5:
                    ev["src"] = ("filter", int(rng.integers(K)))
                else:
                    p = _projection(rng, self.L, S)
                    spec = _draw_sec(rng, dt, res_p=0.5, real_p=0.5)
                    ev["src"] = ("own", p, _Sec(spec["kind"], dt, rng,
                                                spec.get("tau"),
                                                spec.get("wn"),
                                                spec.get("zeta")))
                ev["q"] = rng.uniform(0.9, 0.995)
                # size = expm1(b E) / expm1(b / 2), where E is the crossing
                # speed mapped to Exp(1) by its distribution at the
                # library's own upcrossings: a Pareto-like tail with a random
                # index 1/b in [1.7, 6.7]
                ev["b"] = rng.uniform(0.15, 0.6)
                ev["resp"] = self._draw_resp(rng, p_spike=0.5)
                dvec = rng.normal(0, 1, 5) * (rng.random(5) < 0.6)
                if not dvec.any():
                    dvec[rng.integers(5)] = 1.0
                ev["dir"] = dvec / np.abs(dvec).max()
                ev["H"] = _logu(rng, 0.3, 5.0)
                # the unpredictable part of the size: a mean-one lognormal
                ev["mark_s"] = rng.uniform(0.3, 1.5) if rng.random() < 0.7 \
                    else 0.0
                self.events.append(ev)
        # ---------------------------------------------- relays (G6, B)
        self.relays = []
        if relay:
            for _ in range(1 if rng.random() < 0.75 else 2):
                rl = {}
                if K > 0 and rng.random() < 0.6:
                    rl["src"] = ("filter", int(rng.integers(K)))
                else:
                    rl["src"] = ("slow", int(rng.choice(SLOW)))
                # a band wide enough that the input often sits inside it,
                # where only the remembered mode decides (split B tests
                # memory that does not fade; narrow bands made the mode a
                # function of the current input almost always)
                rl["q"] = rng.uniform(0.55, 0.9)
                rl["band"] = rng.uniform(0.5, 1.5)
                rl["dwell"] = max(1, int(round(_logu(rng, 0.1, 3.0) / dt)))
                self.relays.append(rl)
        # ------------------------------------------------ noise (G8, G9)
        secs = []
        if rng.random() < 0.6:
            secs.append(_Sec("res", dt, rng, wn=_logu(rng, 0.3, 12.5),
                             zeta=_logu(rng, 0.05, 0.7), d_prob=0.5))
        if rng.random() < 0.6 or not secs:
            secs.append(_Sec("real", dt, rng, tau=_logu(rng, 0.1, 30.0)))
        self.nsecs = secs
        h = np.array([1.0] + [0.0] * 2999)
        for sc in secs:
            x = np.zeros((1, sc.n))
            out = np.zeros_like(h)
            for k in range(len(h)):
                y, x = sc.step(x, h[k:k + 1])
                out[k] = y[0]
            h = out
        self.n_gain = 1.0 / max(np.sqrt((h ** 2).sum()), 1e-9)
        self.n_pre = int(min(5.0 * max(s.tau_eff for s in secs) / dt, 1500))
        Am = rng.normal(0, 1, (5, 5))
        Rm = Am @ Am.T + 5 * np.eye(5) * rng.uniform(0.2, 2.0)
        dg = np.sqrt(np.diag(Rm))
        self.n_chol = np.linalg.cholesky(Rm / np.outer(dg, dg))
        self.nu = _logu(rng, 3.0, 50.0)
        self.shared_mix = rng.random() < 0.5
        self.u_c = rng.uniform()                     # share of coloured
        self.env = None
        if rng.random() < 0.5:
            self.env = dict(p=_projection(rng, self.L, S),
                            a=np.exp(-dt / _logu(rng, 1.0, 10.0)),
                            g=rng.uniform(-0.7, 0.7))
        self.bursts = None
        if rng.random() < 0.3:
            p = _projection(rng, self.L, S)
            dvec = rng.normal(0, 1, 5) * (rng.random(5) < 0.6)
            if not dvec.any():
                dvec[rng.integers(5)] = 1.0
            self.bursts = dict(
                p=p, a=np.exp(-dt / _logu(rng, 0.5, 5.0)),
                lam0=_logu(rng, 0.02, 0.5),
                kap=rng.uniform(0, 2) if rng.random() < 0.5 else 0.0,
                resp=self._draw_resp(rng, p_spike=0.5),
                dir=dvec / np.abs(dvec).max(), H=_logu(rng, 0.3, 3.0))
        # ------------------------------------------------ slow (G10)
        self.b0 = (rng.normal(0, 0.3, 5) * self.A if rng.random() < 0.5
                   else np.zeros(5))
        self.gain_ou = ((_logu(rng, 20, 200), rng.uniform(0.1, 0.5))
                        if rng.random() < 0.3 else None)
        self.bias_ou = ((_logu(rng, 20, 500), rng.uniform(0, 0.5, 5))
                        if rng.random() < 0.3 else None)
        self.k_regime = (int(round(rng.uniform(25, 70) / dt))
                         if rng.random() < 0.15 else None)
        # ---------------------------------------------- readouts (G5)
        self._rng_build = rng
        self._normalise(S)
        del self._rng_build
        self.style.update(
            K=K, types=[f["spec"]["kind"] for f in self.filters],
            taus=[f["spec"].get("tau") for f in self.filters],
            wns=[f["spec"].get("wn") for f in self.filters],
            zetas=[f["spec"].get("zeta") for f in self.filters],
            wave_in=[bool(np.abs(f["p"][list(G_WAVE)]).sum() > 0)
                     for f in self.filters],
            scheduled=[f["sidx"] is not None for f in self.filters],
            pre=[f["pre"][0] for f in self.filters], n_dir=len(self.dirs),
            n_events=len(self.events),
            event_b=[float(ev["b"]) for ev in self.events],
            bursts=self.bursts is not None, relays=len(self.relays),
            env=self.env is not None, gain_drift=self.gain_ou is not None,
            bias_drift=self.bias_ou is not None,
            regime=self.k_regime is not None, null=self.null,
            A=self.A.tolist(), pi_m=self.pi_m.tolist(), nu=float(self.nu),
            beta=float(self.ro["beta"]))

    # ------------------------------------------------------------ parts
    @staticmethod
    def _draw_resp(rng, p_spike=0.3):
        if rng.random() < p_spike:
            return None                       # a one-step spike
        return dict(wn=min(_logu(rng, 3.0, 12.0), 11.5),
                    zeta=_logu(rng, 0.1, 0.5))

    def _resp_sec(self, resp):
        if resp is None:
            return None
        sec = _Sec("res", self.dt, np.random.default_rng(0), wn=resp["wn"],
                   zeta=resp["zeta"], d_prob=0.0)
        sec.c = np.array([1.0, 0.0])
        sec.d = 0.0
        pk = np.abs(sec.impulse(200)).max()
        sec.c = sec.c / max(pk, 1e-9)
        return sec

    def _readout(self, rng, K, n_dir):
        """A random readout of X = [K filter outputs, n_dir direct inputs]:
        an MLP (16 units: tanh, ReLU, square, steep sigmoid), and w.p. 0.5
        a bilinear gain-scheduling term (a . filters)(b . direct), each part
        normalised on the library and mixed at a random share beta."""
        n_in = K + n_dir
        H = 16
        alpha = _logu(rng, 0.2, 3.0)
        W1 = rng.normal(0, alpha / np.sqrt(max(n_in, 1)), (H, n_in))
        b1 = rng.normal(0, 1.0, H)
        act = rng.choice(4, size=H, p=[0.5, 0.2, 0.15, 0.15])
        slope = _logu(rng, 3.0, 30.0, H)
        W2 = rng.normal(0, 1 / np.sqrt(H), (5, H))
        on = rng.random(5) < 0.8
        if not on.any():
            on[rng.integers(5)] = True
        W2 = W2 * on[:, None]
        bil, beta = None, 0.0
        if rng.random() < 0.5 and K > 0 and n_dir > 0:
            # (a . filters)(b_c . direct): one filter mix, scheduled per
            # channel by its own mix of the direct inputs
            u = np.concatenate([rng.normal(0, 1 / np.sqrt(K), K),
                                np.zeros(n_dir)])
            V = np.concatenate([np.zeros((5, K)), rng.normal(
                0, 1 / np.sqrt(n_dir), (5, n_dir))], 1) * on[:, None]
            bil = (u, V)
            beta = rng.uniform(0.0, 0.5)
        return dict(W1=W1, b1=b1, act=act, slope=slope, psd=np.ones(H),
                    q=rng.uniform(0.5, 0.98, H), W2=W2, bil=bil, beta=beta,
                    qb=np.zeros(H), m_mu=np.zeros(5), m_s=np.ones(5),
                    b_mu=np.zeros(5), b_s=np.ones(5), mu=np.zeros(5),
                    s=np.ones(5))

    def _ro_parts(self, ro, X):
        pre = X @ ro["W1"].T + ro["b1"]
        h = np.empty_like(pre)
        a = ro["act"]
        h[..., a == 0] = np.tanh(pre[..., a == 0])
        h[..., a == 1] = np.maximum(pre[..., a == 1], 0.0)
        h[..., a == 2] = pre[..., a == 2] ** 2
        m = a == 3
        h[..., m] = 1.0 / (1.0 + np.exp(-np.clip(
            ro["slope"][m] / ro["psd"][m] * (pre[..., m] - ro["qb"][m]),
            -50, 50)))
        om = h @ ro["W2"].T
        ob = None
        if ro["bil"] is not None:
            u, V = ro["bil"]
            ob = (X @ u)[..., None] * (X @ V.T)
        return om, ob, pre

    def _ro_mix(self, ro, X):
        om, ob, _ = self._ro_parts(ro, X)
        o = (om - ro["m_mu"]) / ro["m_s"]
        if ob is not None:
            o = np.sqrt(1 - ro["beta"]) * o + np.sqrt(ro["beta"]) * (
                (ob - ro["b_mu"]) / ro["b_s"])
        return o

    def _ro_norm(self, ro, X):
        """The readout's output, centred and unit-scale per channel on the
        library (by its own statistics)."""
        return (self._ro_mix(ro, X) - ro["mu"]) / ro["s"]

    def _pre(self, f, v):
        k = f["pre"]
        if k[0] == "id":
            return v
        if k[0] == "relu":
            return np.maximum(v - k[1], 0.0)
        w1, b1, w2 = k[1], k[2], k[3]
        vs = (v - f["v_mu"]) / f["v_sd"]
        return np.tanh(vs[..., None] * w1 + b1) @ w2

    # ----------------------------------------------------- the state
    def new_state(self, B=1, rng=None, init_s=None):
        """Fresh state for B rows, the fading parts at rest.
        init_s (B, 26) raw: the first step's inputs; slow low-passes and
        noise drivers start at their current value rather than at 0.
        rng (closed loop only): draws the slow processes' initial values and
        pre-rolls the coloured-noise cascade to its stationary state."""
        st = dict(k=0,
                  fx=[np.zeros((B, f["sec"].n)) for f in self.filters],
                  dl=[np.zeros((B, 4)) for _ in self.filters],
                  sb=np.zeros((B, len(SLOW))),
                  ex=[np.zeros((B, ev["src"][2].n)) if ev["src"][0] == "own"
                      else None for ev in self.events],
                  ev_prev=[np.full(B, np.nan) for _ in self.events],
                  er=[np.zeros((B, 2)) for _ in self.events],
                  en=[np.zeros((B, 2)) for _ in self.events],
                  rl=[np.zeros(B) for _ in self.relays],
                  rl_age=[np.full(B, 10 ** 6) for _ in self.relays],
                  nx=[np.zeros((B * 5, sc.n)) for sc in self.nsecs],
                  envx=np.zeros(B), bx=np.zeros(B),
                  br=np.zeros((B, 2)),
                  ga=np.zeros(B), bb=np.zeros((B, 5)),
                  regime=np.zeros(B, bool))
        if init_s is not None:
            z0 = (np.atleast_2d(init_s) - self.mu) / self.sd
            st["sb"] = z0[:, list(SLOW)].copy()
            if self.env is not None:
                st["envx"] = np.abs(z0 @ self.env["p"])
            if self.bursts is not None:
                st["bx"] = z0 @ self.bursts["p"]
        if rng is not None:
            # the slow processes drift on their own stream, so noise draws
            # never move the rule
            rs = np.random.default_rng([self.seed, 2])
            st["rng_slow"] = rs
            if self.gain_ou is not None:
                st["ga"] = rs.normal(0, self.gain_ou[1], B)
            if self.bias_ou is not None:
                st["bb"] = rs.normal(0, 1, (B, 5)) * self.bias_ou[1] * self.A
            for _ in range(self.n_pre):
                self._cascade(st, self._eps(B, rng))
        return st

    def slow_state(self, st):
        """The parts of the state that do not fade: copied to a relabelled
        history (relays and their dwell age, slow gain and bias, regime)."""
        return dict(ga=st["ga"].copy(), bb=st["bb"].copy(),
                    rl=[r.copy() for r in st["rl"]],
                    rl_age=[a.copy() for a in st["rl_age"]],
                    regime=st["regime"].copy(), k=st["k"])

    # ----------------------------------------------------------- step
    def _filters(self, st, z):
        """Filter outputs (B, K), unnormalised."""
        B = z.shape[0]
        st["sb"] += (z[:, list(SLOW)] - st["sb"]) * (1 - np.exp(
            -self.dt / 5.0))
        out = np.zeros((B, len(self.filters)))
        for i, f in enumerate(self.filters):
            v = self._pre(f, z @ f["p"])
            dl = st["dl"][i]
            dl[:, 1:] = dl[:, :-1].copy()
            dl[:, 0] = v
            vd = dl[:, f["delay"]]
            sb = None if f["sidx"] is None else st["sb"][:, SLOW.index(
                f["sidx"])]
            out[:, i], st["fx"][i] = f["sec"].step(st["fx"][i], vd, sb)
        return out

    def step(self, st, s_raw, noise=True, rng=None, freeze=False,
             hold_relays=False):
        """One control step. s_raw (B, 26) -> (rule (B, 5), noise (B, 5)).
        freeze: slow processes and the regime held (relabelling from a
        checkpoint). hold_relays: relays keep their mode (warm-up of a
        relabelled history)."""
        z = (np.atleast_2d(s_raw) - self.mu) / self.sd
        B = z.shape[0]
        k = st["k"]
        yf = self._filters(st, z)
        yn = (yf - self.f_mu) / self.f_s if len(self.filters) else yf
        X = np.concatenate([yn, z[:, self.dirs]], 1)
        if self.k_regime is not None and k >= self.k_regime and not freeze:
            st["regime"][:] = True
        ym = self._ro_norm(self.ro, X)
        if self.k_regime is not None and st["regime"].any():
            ym = np.where(st["regime"][:, None], self._ro_norm(self.ro2, X),
                          ym)
        for i, rl in enumerate(self.relays):
            if rl["src"][0] == "filter":
                x = yn[:, rl["src"][1]]
            else:
                x = st["sb"][:, SLOW.index(rl["src"][1])]
            h, age = st["rl"][i], st["rl_age"][i]
            if not hold_relays:
                up = (x > rl["hi"]) & (h < 0.5) & (age >= rl["dwell"])
                dn = (x < rl["lo"]) & (h > 0.5) & (age >= rl["dwell"])
                h = np.where(up, 1.0, np.where(dn, 0.0, h))
                age = np.where(up | dn, 0, age + 1)
            # held (warm-up of a relabelled history): mode and dwell age stay
            # as they were at the checkpoint
            st["rl"][i], st["rl_age"][i] = h, age
            ym = ym + h[:, None] * self._ro_norm(rl["ro"], X)
        a = np.exp(st["ga"])[:, None]
        rule = self.A * a * 2.5 * np.tanh(np.sqrt(self.pi_m) * ym / 2.5) \
            + self.b0 + st["bb"]
        ev_noise = np.zeros((B, 5))
        for i, ev in enumerate(self.events):
            if ev["src"][0] == "filter":
                v = yn[:, ev["src"][1]]
            else:
                y, st["ex"][i] = ev["src"][2].step(st["ex"][i],
                                                   z @ ev["src"][1])
                v = (y - ev["mu"]) / ev["s"]
            prev = st["ev_prev"][i]
            hit = (np.nan_to_num(prev, nan=np.inf) < ev["thr"]) & (
                v >= ev["thr"])
            x = np.where(np.isnan(prev), 0.0, np.abs(v - np.nan_to_num(prev)))
            size = np.expm1(np.minimum(ev["b"] * self._exp_rank(ev, x),
                                       30.0)) / np.expm1(ev["b"] / 2)
            size = np.where(hit, np.minimum(size, 100.0), 0.0)
            mult = np.ones(B)
            if ev["mark_s"] > 0 and noise and rng is not None and hit.any():
                s_ = ev["mark_s"]
                mult = np.where(hit, np.exp(rng.normal(-0.5 * s_ ** 2, s_,
                                                       B)), 1.0)
            st["ev_prev"][i] = v
            st["er"][i], r_rule = self._resp_step(ev, st["er"][i], size)
            st["en"][i], r_nz = self._resp_step(ev, st["en"][i],
                                                size * (mult - 1.0))
            dvec = ev["H"] * self.A * ev["dir"]
            rule = rule + dvec[None] * r_rule[:, None]
            ev_noise += dvec[None] * r_nz[:, None]
        nz = np.zeros((B, 5))
        if noise:
            nz = self._noise(st, z, a, B, rng) + ev_noise
        if not freeze:
            self._slow(st, B, rng)
        st["k"] = k + 1
        return rule, nz

    @staticmethod
    def _exp_rank(ev, x):
        """Crossing speed x -> E, Exp(1)-distributed at the library's own
        upcrossings (empirical CDF, with an exponential tail beyond the
        largest library crossing)."""
        xl = ev["xl"]
        n = len(xl)
        F = np.searchsorted(xl, x, side="right") / (n + 1.0)
        E = -np.log1p(-F)
        top = -np.log1p(-n / (n + 1.0))
        return np.where(x > xl[-1], top + (x - xl[-1]) / ev["xl_tail"], E)

    def _resp_step(self, ev, x, size):
        """A response acts in the step of the hit, then rings."""
        sec = ev["_sec"]
        if sec is None:
            return x, size
        _, xn = sec.step(x, size)
        return xn, xn @ sec.c

    def _eps(self, B, rng):
        xi = rng.normal(0, 1, (B, 5)) @ self.n_chol.T
        shape = (B, 1) if self.shared_mix else (B, 5)
        w = 1.0 / rng.gamma(self.nu / 2, 2.0 / self.nu, shape)
        return xi * np.sqrt(w) / np.sqrt(self.nu / max(self.nu - 2, 0.5))

    def _cascade(self, st, eps):
        B = eps.shape[0]
        y = eps
        for i, sc in enumerate(self.nsecs):
            yy, st["nx"][i] = sc.step(st["nx"][i], y.reshape(-1))
            y = yy.reshape(B, 5)
        return y * self.n_gain

    def _noise(self, st, z, a, B, rng):
        col = self._cascade(st, self._eps(B, rng))
        sig = np.ones(B)
        if self.env is not None:
            e = self.env
            st["envx"] = e["a"] * st["envx"] + (1 - e["a"]) * np.abs(
                z @ e["p"])
            sig = np.exp(e["g"] * (st["envx"] - e["mu"]) / e["s"]) / e["Z"]
        wht = rng.normal(0, 1, (B, 5))
        nb = sig[:, None] * (np.sqrt(self.u_c) * col
                             + np.sqrt(1 - self.u_c) * wht)
        out = self.A * a * np.sqrt(self.pi_n) * nb
        if self.bursts is not None:
            b = self.bursts
            st["bx"] = b["a"] * st["bx"] + (1 - b["a"]) * (z @ b["p"])
            zt = (st["bx"] - b["mu"]) / b["s"]
            lam = np.minimum(b["lam0"] * np.exp(b["kap"] * zt
                                                - 0.5 * b["kap"] ** 2),
                             1.0 / (4 * self.dt))
            hit = rng.random(B) < lam * self.dt
            J = np.where(hit, np.clip(rng.standard_t(3, B), -20, 20), 0.0)
            st["br"], r = self._resp_step(b, st["br"], J)
            out = out + (b["H"] * self.A * b["dir"])[None] * r[:, None]
        return out

    def _slow(self, st, B, rng):
        rng = st.get("rng_slow")
        if rng is None:
            return
        dt = self.dt
        if self.gain_ou is not None:
            tau, sdv = self.gain_ou
            st["ga"] = st["ga"] - st["ga"] * dt / tau + sdv * np.sqrt(
                2 * dt / tau) * rng.normal(0, 1, B)
        if self.bias_ou is not None:
            tau, sdv = self.bias_ou
            st["bb"] = st["bb"] - st["bb"] * dt / tau + (sdv * self.A) \
                * np.sqrt(2 * dt / tau) * rng.normal(0, 1, (B, 5))

    # -------------------------------------------------- normalisation
    def _normalise(self, S):
        """Run the parts on the library and set every threshold and scale
        (see the module docstring). S (B, T, 26), normalised."""
        rng = self._rng_build
        B, T, _ = S.shape
        st = self.new_state(B, init_s=S[:, 0] * self.sd + self.mu)
        K = len(self.filters)
        Y = np.zeros((B, T, K))
        for k in range(T):
            Y[:, k] = self._filters(st, S[:, k])
        burn = min(40, T // 4)
        Yb = Y[:, burn:]
        self.f_mu = Yb.reshape(-1, K).mean(0) if K else np.zeros(0)
        self.f_s = (Yb.reshape(-1, K).std(0) + 1e-9) if K else np.ones(0)
        Yn = (Yb - self.f_mu) / self.f_s
        X = np.concatenate([Yn, S[:, burn:][..., self.dirs]], -1)
        Xf = X.reshape(-1, X.shape[-1])
        nd = len(self.dirs)
        self.ro = self._readout(rng, K, nd)
        self._fit_readout(self.ro, Xf)
        if self.k_regime is not None:
            self.ro2 = {k_: (v.copy() if isinstance(v, np.ndarray) else v)
                        for k_, v in self.ro.items()}
            fresh = self._readout(rng, K, nd)
            for key in ("W1", "W2"):
                m = rng.random(self.ro[key].shape) < 0.3
                self.ro2[key] = np.where(m, fresh[key], self.ro[key])
            self._fit_readout(self.ro2, Xf)
        for rl in self.relays:
            if rl["src"][0] == "filter":
                x = Yn[..., rl["src"][1]].ravel()
            else:
                x = self._lowpass(S[..., rl["src"][1]], 5.0)[:, burn:].ravel()
            rl["hi"] = float(np.quantile(x, rl["q"]))
            rl["lo"] = rl["hi"] - rl["band"] * (x.std() + 1e-9)
            rl["ro"] = self._readout(rng, K, nd)
            self._fit_readout(rl["ro"], Xf)
        for ev in self.events:
            if ev["src"][0] == "filter":
                v = Yn[..., ev["src"][1]]
                ev["mu"], ev["s"] = 0.0, 1.0
            else:
                sec, p = ev["src"][2], ev["src"][1]
                x = np.zeros((B, sec.n))
                yy = np.zeros((B, T))
                for k in range(T):
                    yy[:, k], x = sec.step(x, S[:, k] @ p)
                yy = yy[:, burn:]
                ev["mu"], ev["s"] = float(yy.mean()), float(yy.std() + 1e-9)
                v = (yy - ev["mu"]) / ev["s"]
            ev["thr"] = float(np.quantile(v, ev["q"]))
            up = (v[:, :-1] < ev["thr"]) & (v[:, 1:] >= ev["thr"])
            xl = np.sort(np.abs(np.diff(v, axis=1))[up])
            if len(xl) < 20:            # too few crossings: all jumps
                xl = np.sort(np.abs(np.diff(v, axis=1)).ravel())
            ev["xl"] = xl
            n_t = max(3, len(xl) // 10)
            ev["xl_tail"] = float(max(xl[-n_t:].mean() - xl[-n_t], 1e-9))
            ev["_sec"] = self._resp_sec(ev["resp"])
        if self.env is not None:
            e = self.env
            d = self._lowpass(np.abs(S @ e["p"]), None, a=e["a"])[:, burn:]
            e["mu"], e["s"] = float(d.mean()), float(d.std() + 1e-9)
            dn = (d - e["mu"]) / e["s"]
            e["Z"] = float(np.sqrt(np.mean(np.exp(2 * e["g"] * dn))))
        if self.bursts is not None:
            b = self.bursts
            d = self._lowpass(S @ b["p"], None, a=b["a"])[:, burn:]
            b["mu"], b["s"] = float(d.mean()), float(d.std() + 1e-9)
            b["_sec"] = self._resp_sec(b["resp"])

    def _lowpass(self, x, tau, a=None):
        a = np.exp(-self.dt / tau) if a is None else a
        y = np.zeros_like(x)
        acc = x[:, 0].copy()
        for k in range(x.shape[1]):
            acc = a * acc + (1 - a) * x[:, k]
            y[:, k] = acc
        return y

    def _fit_readout(self, ro, Xf):
        """Library statistics of a readout: sigmoid thresholds and input
        spreads, each part's centring and scale, then the mix's."""
        _, _, pre = self._ro_parts(ro, Xf)
        if len(pre):
            ro["qb"] = np.array([np.quantile(pre[:, j], ro["q"][j])
                                 for j in range(pre.shape[1])])
            ro["psd"] = np.maximum(pre.std(0), 1e-6)
        om, ob, _ = self._ro_parts(ro, Xf)
        ro["m_mu"], s = om.mean(0), om.std(0)
        ro["m_s"] = np.where(s > 1e-9, s, 1.0)
        if ob is not None:
            ro["b_mu"], s = ob.mean(0), ob.std(0)
            ro["b_s"] = np.where(s > 1e-9, s, 1.0)
        o = self._ro_mix(ro, Xf)
        ro["mu"], s = o.mean(0), o.std(0)
        ro["s"] = np.where(s > 1e-9, s, 1.0)

    # ----------------------------------------------------- batch runs
    def run(self, S_raw, slow=None, noise=False, rng=None, hold=0):
        """The operator along B histories S_raw (B, T, 26), from a fresh
        fading state (slow low-passes started at the first inputs).
        slow: a slow_state (one row, broadcast) to start from and hold
        (relabelling); relays keep the copied mode for the first `hold`
        steps (the warm-up). Returns rule (B, T, 5) [, noise]."""
        S_raw = np.asarray(S_raw, float)
        B, T, _ = S_raw.shape
        st = self.new_state(B, init_s=S_raw[:, 0])
        if slow is not None:
            for key in ("ga", "bb", "regime"):
                st[key][:] = slow[key][0]
            for i, r in enumerate(slow["rl"]):
                st["rl"][i][:] = r[0]
                st["rl_age"][i][:] = slow["rl_age"][i][0]
            st["k"] = slow["k"]
        R = np.zeros((B, T, 5))
        N = np.zeros((B, T, 5)) if noise else None
        for k in range(T):
            r, n = self.step(st, S_raw[:, k], noise=noise, rng=rng,
                             freeze=slow is not None, hold_relays=k < hold)
            R[:, k] = r
            if noise:
                N[:, k] = n
        return (R, N) if noise else R
