#!/usr/bin/env python3
"""
Does the closed-loop throttle follow the wave phase?

Re-runs rows of studies/final_eval.py exactly as its run_mission does (the
module is imported, never edited: its events_for hook is wrapped to keep
the per-step series run_mission otherwise throws away), with the meta7
settings (gz plant, T 120 s, K 128, S 8, steer auto, corridor 2 L, fc_lam
0.1, fc_hp 24, trigger on). Per mission it stores the plant's own recorded
trajectory (every 0.02 s plant step: state, commanded thrust and nozzle,
CG / bow vertical acceleration), the planner's plan after every call, and
the wave at the bow as the analytic signal of the sea's components

    zeta(t) = sum_i a_i exp(1j (k_i (X cos th_i + Y sin th_i) - w_i t + phi_i))

(Re zeta = data5.sea_eval = the elevation the Recorder / planners use;
checked on every mission). The encounter phase theta = -angle(zeta) runs
forward in time (checked): theta 0 = crest at the bow, -90 deg = the water
at the bow rising (the bow meets the front face of the wave), +-180 =
trough, +90 = falling (the crest has passed the bow).

    python -m studies.wave_phase_closedloop --rows hand,m0 --missions 0-7 \\
        --legs 0,1
    python -m studies.wave_phase_closedloop --rows cat_p_prior --missions 0 \\
        --legs 0,1
    python -m studies.wave_phase_closedloop --analyze --rows hand,m0,cat_p_prior

Outputs: studies/_cache/wave_phase/closedloop/runs/<row>_s<seed>_l<leg>.pkl,
analysis.json, analysis.txt, run.log.
"""
import argparse
import json
import math
import os
import pickle
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np   # noqa: E402

OUT = os.path.join(HERE, "_cache", "wave_phase", "closedloop")
RUNS = os.path.join(OUT, "runs")
KN = 0.514444
T_SKIP = 15.0          # s: the start-up transient left out of the analysis
NB = 12                # phase bins
N_NULL = 1000


def _out():
    os.makedirs(OUT, exist_ok=True)
    return OUT


def say(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(_out(), "run.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------ running
CAP = {}
PLAN = []


def setup(device="auto"):
    from learn.repro import task
    from studies import final_eval as FE
    FE.OPT.update(plant="gz", T=task.T_EVAL, S=8, K=128, device=device,
                  fc_lam=0.1, fc_hp=24, trigger=True, min_gb=1.5,
                  steer="auto", y_lengths=2.0, tag="")
    # final_eval's log() writes under out_dir(): send it here instead
    FE.out_dir = _out
    orig = FE.events_for

    def events_for(m, rec, geo, cmd_thr, cmd_noz, safe_on):
        CAP.update(m=m, rec=rec, geo=geo, cmd_thr=np.asarray(cmd_thr, float),
                   cmd_noz=np.asarray(cmd_noz, float),
                   safe=np.asarray(safe_on, bool))
        return orig(m, rec, geo, cmd_thr, cmd_noz, safe_on)
    FE.events_for = events_for

    from learn.meta import mpc_constrained as MC
    c_orig = MC.ConstrainedMPPI.__call__

    def c_call(self, rows, k, info):
        cmds, dg = c_orig(self, rows, k, info)
        d = dg[0]
        PLAN.append(dict(k=int(k), cmd=float(cmds[0]),
                         plan=(self.nominal[rows[0]] @ self.E.T).astype(
                             np.float32),
                         n_ok=int(d["n_ok"]),
                         g={n: float(d[n]) for n in d if n.startswith("g_")}))
        return cmds, dg
    MC.ConstrainedMPPI.__call__ = c_call

    from control import mpc as CM
    h_orig = CM.MPPIController.__call__

    def h_call(self, s_reduced, t, track_ref=0.0):
        cmd = h_orig(self, s_reduced, t, track_ref)
        grid = np.arange(self.H)
        PLAN.append(dict(k=len(PLAN), cmd=float(cmd[0]),
                         plan=np.interp(grid, self.knot_t,
                                        self.nominal[0]).astype(np.float32)))
        return cmd
    CM.MPPIController.__call__ = h_call
    task.ctx()
    return FE


def sea_arrays(sea):
    return dict(a=np.asarray(sea.a, float).ravel(),
                k=np.asarray(sea.k, float).ravel(),
                th=np.asarray(sea.th, float).ravel(),
                w=np.asarray(sea.w, float).ravel(),
                phi=np.asarray(sea.phi, float).ravel(),
                hs=float(getattr(sea, "hs", np.nan)),
                tp=float(getattr(sea, "tp", np.nan)))


def zeta(sa, X, Y, t):
    """Analytic signal of the elevation at points X, Y (n,) and times t (n,)
    (chunked over rows)."""
    out = np.empty(len(t), complex)
    for i in range(0, len(t), 1000):
        sl = slice(i, i + 1000)
        ph = (sa["k"] * (X[sl, None] * np.cos(sa["th"])
                         + Y[sl, None] * np.sin(sa["th"]))
              - sa["w"] * t[sl, None] + sa["phi"])
        out[sl] = (sa["a"] * np.exp(1j * ph)).sum(-1)
    return out


def run_one(FE, row, seed, leg, turn):
    spec, why = FE.resolve_row(row)
    if spec is None:
        raise SystemExit(f"row {row}: {why}")
    nets = FE._row_nets(spec)
    CAP.clear()
    PLAN.clear()
    t0 = time.time()
    r = FE.run_mission(spec, seed, leg, FE.OPT["T"], turn, nets)
    wall = time.time() - t0
    m, geo = CAP["m"], CAP["geo"]
    tr = m.plant.trajectory()
    sa = sea_arrays(m.sea)
    S = tr["s"]
    t = tr["t"]
    xb = geo["x_b"]
    c, s_ = np.cos(S[:, 5]), np.sin(S[:, 5])
    Xb, Yb = S[:, 0] + xb * c, S[:, 1] + xb * s_
    zb = zeta(sa, Xb, Yb, t)
    zc = zeta(sa, S[:, 0], S[:, 1], t)
    # checks: Re zeta = the elevation final_eval's Recorder evaluates
    from learn.meta.data5 import sea_eval
    chk1 = float(np.max(np.abs(zb.real - sea_eval(m.sea, Xb, Yb, t))))
    rec = CAP["rec"]
    EB = np.asarray(rec.EB, float)
    yo = np.asarray(rec.y_off, float)
    ic = int(np.argmin(np.abs(yo)))
    # rec row i = after plant step i = trajectory row i + 1
    n = min(len(EB) - 1, len(t) - 1)
    chk2 = float(np.max(np.abs(zb.real[1:n + 1] - EB[:n, ic]))) \
        if abs(yo[ic]) < 1e-9 and n > 0 else float("nan")
    pr = m.plant.prop
    prop = dict(t_max=float(pr.t_max), tau=float(getattr(pr, "tau", np.nan)),
                rate_max=float(getattr(pr, "rate_max", np.nan)),
                dead_band=float(getattr(pr, "dead_band", np.nan)),
                delay_steps=int(getattr(pr.delay, "n", -1)),
                delay_s=float(getattr(pr.delay, "n", np.nan)
                              * float(tr["meta"]["dt"])),
                t_min=float(getattr(pr, "t_min", np.nan)))
    rec_out = dict(
        row=row, seed=seed, leg=leg, wall_s=wall, dt=float(tr["meta"]["dt"]),
        sub=int(m.sub), t_max=float(m.t_max), rud_max=float(m.rud_max),
        phi=float(m.phi), L=float(m.L), u_ref=float(m.u_ref), geo=geo,
        prop=prop, sea=sa, mass=float(tr["meta"]["mass"]),
        t=t, s=S.astype(np.float64), a_cg=tr["a_cg"], a_bow=tr["a_bow"],
        thrust_cmd=tr["thrust_cmd"], rudder_cmd=tr["rudder_cmd"],
        zb=zb, zc=zc, safe=CAP["safe"], fe_cmd_thr=CAP["cmd_thr"],
        plan=list(PLAN), check_re_vs_sea_eval=chk1, check_re_vs_recorder=chk2,
        summary=dict(kn=r["kn"], n_ge7=r["events"]["2.0"].get("n_ge7"),
                     a110=r["events"]["2.0"].get("a110"),
                     n_steps=r["n_steps"], t_step=r["t_step"],
                     plan=r.get("plan"), safe_steps=r["safe_steps"]))
    os.makedirs(RUNS, exist_ok=True)
    p = os.path.join(RUNS, f"{row}_s{seed}_l{leg}.pkl")
    pickle.dump(rec_out, open(p + ".tmp", "wb"))
    os.replace(p + ".tmp", p)
    say(f"{row} s{seed} l{leg}: {r['kn']:.3f} kn, >=7g "
        f"{rec_out['summary']['n_ge7']}, A1/10 "
        f"{rec_out['summary']['a110']:.3f} g, {wall:.0f} s wall, "
        f"check |Re zeta - sea_eval| {chk1:.2e}, vs Recorder {chk2:.2e}")
    return rec_out


def free_gb():
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


def run(rows, missions, legs, device, min_gb):
    FE = setup(device)
    turn = FE.phase_turn()
    for row in rows:
        for sd in missions:
            for lg in legs:
                p = os.path.join(RUNS, f"{row}_s{sd}_l{lg}.pkl")
                if os.path.exists(p):
                    say(f"{row} s{sd} l{lg}: done before, skipped")
                    continue
                g = free_gb()
                if g < min_gb:
                    say(f"{row} s{sd} l{lg}: only {g:.1f} GB free "
                        f"(< {min_gb}), stopped")
                    return
                run_one(FE, row, sd, lg, turn)


# ------------------------------------------------------------ analysis
def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def bins_of(theta):
    return np.clip(((wrap(theta) + np.pi) / (2 * np.pi) * NB).astype(int),
                   0, NB - 1)


def binned(x, b):
    mu = np.array([x[b == i].mean() if np.any(b == i) else np.nan
                   for i in range(NB)])
    sd = np.array([x[b == i].std() if np.any(b == i) else np.nan
                   for i in range(NB)])
    n = np.array([int(np.sum(b == i)) for i in range(NB)])
    return mu, sd, n


def depth(x, b):
    mu, _, _ = binned(x, b)
    return float((np.nanmax(mu) - np.nanmin(mu)) / np.mean(x))


def harmonic(x, th):
    """x ~ x0 + A cos(theta - theta_max): (A, theta_max deg)."""
    c1 = 2.0 * np.mean((x - x.mean()) * np.exp(1j * th))     # = A e^{i theta_max}
    return float(abs(c1)), float(np.degrees(np.angle(c1)))


def load_runs(rows):
    out = {}
    for row in rows:
        for fp in sorted(os.listdir(RUNS)):
            if fp.startswith(row + "_s") and fp.endswith(".pkl"):
                r = pickle.load(open(os.path.join(RUNS, fp), "rb"))
                if r["row"] == row:
                    out.setdefault((row, r["leg"]), []).append(r)
    return out


def per_mission(r):
    """Control-step and plant-step series after the transient."""
    dt, sub = r["dt"], r["sub"]
    n = len(r["t"])
    i0 = int(round(T_SKIP / dt))
    th = -np.angle(r["zb"])               # forward-running encounter phase
    thc = -np.angle(r["zc"])
    eta = r["zb"].real
    ctl = np.arange(0, n, sub)
    ctl = ctl[ctl >= i0]
    x = r["thrust_cmd"][ctl] / r["t_max"]
    xa = r["s"][i0:, 12] / r["t_max"]
    u = r["s"][:, 6]
    psi_rel = r["s"][:, 5] - r["phi"]
    ua = u * np.cos(psi_rel) - r["s"][:, 7] * np.sin(psi_rel)
    dth = np.diff(np.unwrap(-np.angle(r["zb"][i0:])))
    return dict(dt=dt, sub=sub, i0=i0, ctl=ctl, x=x, th_ctl=th[ctl],
                z_ctl=r["zb"][ctl],
                thc_ctl=thc[ctl], eta_ctl=eta[ctl], xa=xa, th_pl=th[i0:],
                eta=eta, th=th, n=n, ua=ua,
                fwd_share=float(np.mean(dth > 0)),
                f_e=float(np.mean(dth) / dt / (2 * np.pi)))


def null_depth(xs, ths, rng, how):
    """Modulation depth of the pooled missions under a null: 'perm' =
    phases permuted within each mission; 'shift' = each mission's phase
    series circularly shifted by a random 10 s .. T - 10 s (keeps both
    series' own time structure)."""
    out = np.empty(N_NULL)
    for j in range(N_NULL):
        tt = []
        for x, th in zip(xs, ths):
            if how == "perm":
                tt.append(rng.permutation(th))
            else:
                lo = 42
                sh = rng.integers(lo, max(lo + 1, len(th) - lo))
                tt.append(np.roll(th, sh))
        out[j] = depth(np.concatenate(xs), bins_of(np.concatenate(tt)))
    return out


def null_harm(xs, ths, rng):
    out = np.empty(N_NULL)
    for j in range(N_NULL):
        tt = []
        for th in ths:
            sh = rng.integers(42, max(43, len(th) - 42))
            tt.append(np.roll(th, sh))
        X = np.concatenate([x - x.mean() for x in xs])
        out[j] = harmonic(X + 1.0, np.concatenate(tt))[0]
    return out


def coherence_block(pms, fs):
    """Pooled (over missions) Welch cross-spectra of the commanded thrust
    fraction x and the bow elevation y at the control steps."""
    from scipy.signal import csd, welch
    nps = 64
    Pxx = Pyy = Pyx = None
    fes = []
    for pm in pms:
        x = pm["x"] - pm["x"].mean()
        y = pm["eta_ctl"] - pm["eta_ctl"].mean()
        f, pxx = welch(x, fs=fs, nperseg=nps, noverlap=nps // 2,
                       detrend="linear")
        _, pyy = welch(y, fs=fs, nperseg=nps, noverlap=nps // 2,
                       detrend="linear")
        _, pyx = csd(y, x, fs=fs, nperseg=nps, noverlap=nps // 2,
                     detrend="linear")      # conj(Y) X: angle = x - y
        Pxx = pxx if Pxx is None else Pxx + pxx
        Pyy = pyy if Pyy is None else Pyy + pyy
        Pyx = pyx if Pyx is None else Pyx + pyx
        fes.append(pm["f_e"])
    coh = np.abs(Pyx) ** 2 / (Pxx * Pyy)
    return f, coh, Pyx, Pxx, Pyy, float(np.mean(fes))


def band_stats(f, coh, Pyx, Pxx, Pyy, fe):
    band = (f >= 0.75 * fe) & (f <= 1.25 * fe)
    w = Pyy[band]
    cb = float(np.sum(coh[band] * w) / np.sum(w))      # elevation-weighted
    H = Pyx[band] / Pyy[band]
    Hm = np.sum(H * w) / np.sum(w)
    # share of the thrust variance inside the band
    vb = float(np.sum(Pxx[band]) / np.sum(Pxx))
    return dict(coh_band=cb, coh_max=float(coh[band].max()),
                gain_per_m=float(abs(Hm)),
                lead_deg=float(np.degrees(np.angle(Hm))),
                thrust_var_share_in_band=vb,
                band_hz=[float(0.75 * fe), float(1.25 * fe)],
                n_freq_in_band=int(band.sum()))


def coherence_null(pms, fs, rng, n=200):
    vals = []
    for _ in range(n):
        q = []
        for pm in pms:
            sh = rng.integers(42, len(pm["x"]) - 42)
            q.append(dict(pm, eta_ctl=np.roll(pm["eta_ctl"], sh)))
        f, coh, Pyx, Pxx, Pyy, fe = coherence_block(q, fs)
        vals.append(band_stats(f, coh, Pyx, Pxx, Pyy, fe)["coh_band"])
    return float(np.mean(vals)), float(np.percentile(vals, 95))


def envelope_corr(pms, rng, max_lag=20, n=300):
    """Correlation of the commanded thrust x(t) with the wave envelope at
    the bow |zeta|(t + lag) (control steps; each mission demeaned, pooled),
    lags -max_lag..max_lag steps; null: the envelope circularly shifted
    per mission (95% of the max |corr| over lags)."""
    lags = np.arange(-max_lag, max_lag + 1)

    def corr(envs):
        out = []
        for L in lags:
            a, b = [], []
            for pm, e in zip(pms, envs):
                x = pm["x"] - pm["x"].mean()
                e = e - e.mean()
                if L >= 0:
                    a.append(x[:len(x) - L])
                    b.append(e[L:])
                else:
                    a.append(x[-L:])
                    b.append(e[:len(e) + L])
            a, b = np.concatenate(a), np.concatenate(b)
            out.append(float(np.sum(a * b) / math.sqrt(np.sum(a * a)
                                                      * np.sum(b * b))))
        return np.asarray(out)
    envs = [np.abs(pm["z_ctl"]) for pm in pms]
    c = corr(envs)
    nul = []
    for _ in range(n):
        q = [np.roll(e, rng.integers(42, len(e) - 42)) for e in envs]
        nul.append(np.max(np.abs(corr(q))))
    i = int(np.argmax(np.abs(c)))
    return dict(lags_s=(lags * pms[0]["dt"] * pms[0]["sub"]).tolist(),
                corr=c.tolist(), corr_lag0=float(c[max_lag]),
                extreme=float(c[i]),
                extreme_lag_s=float(lags[i] * pms[0]["dt"] * pms[0]["sub"]),
                null95_max_abs=float(np.percentile(nul, 95)))


def var_split(pms, fs, fe):
    """Shares of the commanded-thrust variance (pooled Welch PSD) below
    0.5 f_e, in 0.75..1.25 f_e and above 1.25 f_e."""
    from scipy.signal import welch
    P = None
    for pm in pms:
        f, p = welch(pm["x"] - pm["x"].mean(), fs=fs, nperseg=64,
                     noverlap=32, detrend="linear")
        P = p if P is None else P + p
    tot = P.sum()
    return dict(low=float(P[f < 0.5 * fe].sum() / tot),
                band=float(P[(f >= 0.75 * fe) & (f <= 1.25 * fe)].sum()
                           / tot),
                high=float(P[f > 1.25 * fe].sum() / tot))


def plan_phase(runs, pms, rng, h_max=22, n=300):
    """Does the PLAN anticipate the waves? Each stored plan (after the
    call; the warm-start shift is already applied, so plan[h] is the
    command intended for control step k + 1 + h) minus its own mean, against
    the realised bow phase at that future step; first-harmonic amplitude
    (fraction of max thrust) pooled over calls and h = 1..h_max (h 0 left
    out: the applied command), and per lead h. Null: each mission's phase
    series circularly shifted."""
    pre = []
    for r, pm in zip(runs, pms):
        th_c = pm["th"][::r["sub"]]              # phase at control steps
        P = np.stack([np.asarray(c["plan"], float) for c in r["plan"]])
        K = np.array([c["k"] for c in r["plan"]])
        H = np.arange(1, h_max + 1)
        J = K[:, None] + 1 + H[None]
        ok = (J * r["sub"] >= pm["i0"]) & (J < len(th_c))
        Xd = (P[:, 1:h_max + 1] - P.mean(1, keepdims=True))[ok]
        pre.append((th_c, J[ok], Xd, np.broadcast_to(H, J.shape)[ok]))

    def collect(shifts):
        X, TH, HH = [], [], []
        for (th_c, J, Xd, Hd), sh in zip(pre, shifts):
            t_ = np.roll(th_c, sh) if sh else th_c
            X.append(Xd)
            TH.append(t_[J])
            HH.append(Hd)
        return np.concatenate(X), np.concatenate(TH), np.concatenate(HH)
    X, TH, HH = collect([0] * len(runs))
    if len(X) == 0:
        return None
    A, tm = harmonic(X + 1.0, TH)
    per_h = {int(h): harmonic(X[HH == h] + 1.0, TH[HH == h])
             for h in (1, 2, 4, 8, 16)}
    nul = []
    for _ in range(n):
        sh = [int(rng.integers(42, max(43, len(q[0]) - 42))) for q in pre]
        Xn, THn, _ = collect(sh)
        nul.append(harmonic(Xn + 1.0, THn)[0])
    sd_in = float(np.mean([np.std(c["plan"]) for r in runs
                           for c in r["plan"]]))
    return dict(amp=A, max_deg=tm, null95=float(np.percentile(nul, 95)),
                p=float(np.mean(np.asarray(nul) >= A)),
                per_lead={h: dict(amp=v[0], max_deg=v[1])
                          for h, v in per_h.items()},
                sd_within_plan=sd_in, n=int(len(X)))


def impact_waves(runs, pms, thr):
    """Bow elevation and its rate at the impacts >= thr g (k 2): means of
    eta / |zeta| and the share with the water at the bow rising."""
    e, ed = [], []
    for r, pm in zip(runs, pms):
        idx, pk = impacts(r, pm)
        ii = idx[pk >= thr]
        eta = pm["eta"]
        d_eta = np.gradient(eta, r["dt"])
        amp = np.abs(r["zb"])
        e += list(eta[ii] / np.maximum(amp[ii], 1e-9))
        ed += list(d_eta[ii])
    e, ed = np.asarray(e), np.asarray(ed)
    if not len(e):
        return dict(n=0)
    return dict(n=int(len(e)), eta_over_amp_mean=float(e.mean()),
                share_eta_below_0=float(np.mean(e < 0)),
                share_rising=float(np.mean(ed > 0)),
                eta_rate_mean=float(ed.mean()))


def impacts(r, pm, k=2.0):
    from studies.safety_events import acc_peaks
    idx, pk = acc_peaks(r["a_cg"], r["dt"], k)
    keep = idx >= pm["i0"]
    return idx[keep], pk[keep]


def rayleigh(th):
    n = len(th)
    if n == 0:
        return dict(n=0)
    z = np.mean(np.exp(1j * th))
    R = float(abs(z))
    p = float(np.exp(-n * R ** 2) * (1 + (2 * n * R ** 2 - (n * R ** 2) ** 2)
                                     / (4 * n)))
    return dict(n=n, R=R, mean_deg=float(np.degrees(np.angle(z))),
                p_approx=max(min(p, 1.0), 0.0))


def eta_trig(pms, runs, thr, rng, lags=range(-12, 5)):
    """Event-triggered mean of the commanded thrust fraction (minus the
    mission mean) at control-step lags around impacts >= thr g (k 2);
    null band: the same with as many random times."""
    lags = list(lags)
    rows, rows_a = [], []
    nul = []
    tot = 0
    for r, pm in zip(runs, pms):
        idx, pk = impacts(r, pm)
        ev = idx[pk >= thr]
        sub = r["sub"]
        xall = r["thrust_cmd"][np.arange(0, pm["n"], sub)] / r["t_max"]
        xa = r["s"][:, 12] / r["t_max"]
        j0 = pm["i0"] // sub
        mu = xall[j0:].mean()
        mua = xa[pm["i0"]:].mean()
        for i in ev:
            j = i // sub
            if j + lags[0] < j0 or j + lags[-1] >= len(xall):
                continue
            rows.append([xall[j + L] - mu for L in lags])
            rows_a.append([xa[min(i + L * sub, pm["n"] - 1)] - mua
                           for L in lags])
            tot += 1
    if not rows:
        return dict(n=0)
    E = np.asarray(rows)
    Ea = np.asarray(rows_a)
    # null: random control steps, the same count per mission
    for _ in range(300):
        q = []
        for r, pm in zip(runs, pms):
            sub = r["sub"]
            xall = r["thrust_cmd"][np.arange(0, pm["n"], sub)] / r["t_max"]
            j0 = pm["i0"] // sub
            mu = xall[j0:].mean()
            m_ev = int(round(tot / len(runs)))
            js = rng.integers(j0 - lags[0], len(xall) - lags[-1], m_ev)
            q += [[xall[j + L] - mu for L in lags] for j in js]
        nul.append(np.mean(q, 0))
    nul = np.asarray(nul)
    return dict(n=len(E), lags_s=[L * runs[0]["dt"] * runs[0]["sub"]
                                  for L in lags],
                mean=E.mean(0).tolist(), sem=(E.std(0) / math.sqrt(len(E)))
                .tolist(), applied_mean=Ea.mean(0).tolist(),
                null_lo=np.percentile(nul, 2.5, 0).tolist(),
                null_hi=np.percentile(nul, 97.5, 0).tolist())


def analyze(rows):
    rng = np.random.default_rng(0)
    R = load_runs(rows)
    res = {}
    lines = []
    for (row, leg), runs in sorted(R.items()):
        runs = sorted(runs, key=lambda r: r["seed"])
        pms = [per_mission(r) for r in runs]
        fs = 1.0 / (runs[0]["dt"] * runs[0]["sub"])
        xs = [pm["x"] for pm in pms]
        ths = [pm["th_ctl"] for pm in pms]
        X, TH = np.concatenate(xs), np.concatenate(ths)
        b = bins_of(TH)
        mu, sd, nb = binned(X, b)
        d = depth(X, b)
        dperm = null_depth(xs, ths, rng, "perm")
        dshift = null_depth(xs, ths, rng, "shift")
        A, thmax = harmonic(X, TH)
        Ash = null_harm(xs, ths, rng)
        # applied thrust (plant state 12) on the plant steps
        XA = np.concatenate([pm["xa"] for pm in pms])
        THA = np.concatenate([pm["th_pl"] for pm in pms])
        bA = bins_of(THA)
        muA, _, _ = binned(XA, bA)
        AA, thmaxA = harmonic(XA, THA)
        # CG-phase version
        THc = np.concatenate([pm["thc_ctl"] for pm in pms])
        Ac, thmaxc = harmonic(X, THc)
        # per mission
        pmiss = []
        for r, pm in zip(runs, pms):
            a_, tm_ = harmonic(pm["x"], pm["th_ctl"])
            pmiss.append(dict(seed=r["seed"], kn=r["summary"]["kn"],
                              depth=depth(pm["x"], bins_of(pm["th_ctl"])),
                              harm_amp=a_, harm_max_deg=tm_,
                              x_mean=float(pm["x"].mean()),
                              x_sd=float(pm["x"].std()),
                              sat_hi=float(np.mean(pm["x"] >= 0.995)),
                              sat_lo=float(np.mean(pm["x"] <= 0.005)),
                              f_e=pm["f_e"], fwd_share=pm["fwd_share"]))
        # coherence
        f, coh, Pyx, Pxx, Pyy, fe = coherence_block(pms, fs)
        cs = band_stats(f, coh, Pyx, Pxx, Pyy, fe)
        cn_mean, cn95 = coherence_null(pms, fs, rng)
        envc = envelope_corr(pms, rng)
        vsplit = var_split(pms, fs, fe)
        # impacts by phase
        occ = np.bincount(bins_of(np.concatenate([pm["th_pl"]
                                                  for pm in pms])),
                          minlength=NB).astype(float)
        occ_h = occ * runs[0]["dt"] / 3600.0
        imp = {}
        for thr in (3.0, 7.0):
            ths_ev = []
            for r, pm in zip(runs, pms):
                idx, pk = impacts(r, pm)
                ths_ev.append(pm["th"][idx[pk >= thr]])
            te = np.concatenate(ths_ev) if ths_ev else np.zeros(0)
            cnt = np.bincount(bins_of(te), minlength=NB) if len(te) else \
                np.zeros(NB, int)
            imp[f"ge{thr:g}g"] = dict(
                count_per_bin=cnt.tolist(),
                rate_per_h_per_bin=(cnt / np.maximum(occ_h, 1e-12)).tolist(),
                rayleigh=rayleigh(te),
                share_rising=float(np.mean((wrap(te) < 0)
                                           & (wrap(te) > -np.pi)))
                if len(te) else float("nan"))
        trig3 = eta_trig(pms, runs, 3.0, rng)
        trig7 = eta_trig(pms, runs, 7.0, rng)
        # the plans: how much the plan itself moves inside the horizon
        psd_ = [float(np.mean([np.std(p["plan"]) for p in r["plan"]]))
                for r in runs if r["plan"]]
        pph = plan_phase(runs, pms, rng)
        iw = {f"ge{t:g}g": impact_waves(runs, pms, t) for t in (3.0, 7.0)}
        res[f"{row}_leg{leg}"] = dict(
            n_missions=len(runs), seeds=[r["seed"] for r in runs],
            kn_mean=float(np.mean([r["summary"]["kn"] for r in runs])),
            x_mean=float(X.mean()), x_sd=float(X.std()),
            sat_hi=float(np.mean(X >= 0.995)),
            sat_lo=float(np.mean(X <= 0.005)),
            f_e_hz=fe, T_e_s=1.0 / fe,
            fwd_share=float(np.mean([pm["fwd_share"] for pm in pms])),
            bins_center_deg=(np.degrees(-np.pi + (np.arange(NB) + 0.5)
                                        * 2 * np.pi / NB)).tolist(),
            binned_mean=mu.tolist(), binned_sd=sd.tolist(),
            binned_n=nb.tolist(),
            depth=d, depth_null_perm_95=float(np.percentile(dperm, 95)),
            depth_null_perm_mean=float(dperm.mean()),
            p_perm=float(np.mean(dperm >= d)),
            depth_null_shift_95=float(np.percentile(dshift, 95)),
            depth_null_shift_mean=float(dshift.mean()),
            p_shift=float(np.mean(dshift >= d)),
            harm_amp=A, harm_max_deg=thmax,
            harm_null_shift_95=float(np.percentile(Ash, 95)),
            p_harm_shift=float(np.mean(Ash >= A)),
            harm_amp_cg=Ac, harm_max_deg_cg=thmaxc,
            applied_binned_mean=muA.tolist(),
            applied_depth=float((np.nanmax(muA) - np.nanmin(muA))
                                / XA.mean()),
            applied_harm_amp=AA, applied_harm_max_deg=thmaxA,
            coherence=dict(**cs, null_mean=cn_mean, null_95=cn95),
            impacts=imp, trig3=trig3, trig7=trig7, envelope=envc,
            thrust_var_split=vsplit,
            plan_sd_in_horizon=float(np.mean(psd_)) if psd_ else None,
            plan_phase=pph, impact_waves=iw,
            per_mission=pmiss)
        z = res[f"{row}_leg{leg}"]
        lines.append(
            f"{row} leg {leg} (n {len(runs)}, {z['kn_mean']:.2f} kn): "
            f"thrust {z['x_mean']:.3f} +- {z['x_sd']:.3f} of max "
            f"(>=0.995: {z['sat_hi']:.2f}, <=0.005: {z['sat_lo']:.2f}); "
            f"T_e {z['T_e_s']:.2f} s; depth {d:.3f} (perm null 95% "
            f"{z['depth_null_perm_95']:.3f}, p {z['p_perm']:.3f}; shift null "
            f"95% {z['depth_null_shift_95']:.3f}, p {z['p_shift']:.3f}); "
            f"1st harmonic {A:.4f} of max at {thmax:+.0f} deg (shift null "
            f"95% {z['harm_null_shift_95']:.4f}, p {z['p_harm_shift']:.3f}); "
            f"applied: depth {z['applied_depth']:.3f}, harmonic {AA:.4f} at "
            f"{thmaxA:+.0f} deg; coherence in band {cs['coh_band']:.3f} "
            f"(null mean {cn_mean:.3f}, 95% {cn95:.3f}), gain "
            f"{cs['gain_per_m']:.3f}/m, thrust leads elevation by "
            f"{cs['lead_deg']:+.0f} deg, band share of thrust var "
            f"{cs['thrust_var_share_in_band']:.3f}")
        lines.append(
            f"   thrust variance split (below 0.5 f_e / band / above): "
            f"{vsplit['low']:.2f} / {vsplit['band']:.2f} / "
            f"{vsplit['high']:.2f}; corr(thrust, wave envelope at bow): "
            f"lag 0 {envc['corr_lag0']:+.3f}, extreme {envc['extreme']:+.3f}"
            f" at envelope lag {envc['extreme_lag_s']:+.2f} s (null 95% "
            f"max|corr| {envc['null95_max_abs']:.3f})")
        lines.append("   binned mean thrust (bins -165..+165 deg): "
                     + " ".join(f"{v:.3f}" for v in mu))
        if pph:
            lines.append(
                f"   plan vs realised future bow phase (h 1..22): harmonic "
                f"{pph['amp']:.4f} of max at {pph['max_deg']:+.0f} deg "
                f"(shift null 95% {pph['null95']:.4f}, p {pph['p']:.3f}); "
                f"per lead h: " + ", ".join(
                    f"{h}: {v['amp']:.4f}@{v['max_deg']:+.0f}"
                    for h, v in pph["per_lead"].items())
                + f"; sd within a plan {pph['sd_within_plan']:.4f}")
        for nm, v in iw.items():
            if v.get("n"):
                lines.append(
                    f"   at impacts {nm}: n {v['n']}, mean eta/|zeta| "
                    f"{v['eta_over_amp_mean']:+.2f}, eta < 0 share "
                    f"{v['share_eta_below_0']:.2f}, water at bow rising "
                    f"share {v['share_rising']:.2f}")
        for nm, v in imp.items():
            ry = v["rayleigh"]
            lines.append(f"   impacts {nm}: counts {v['count_per_bin']}; "
                         f"Rayleigh n {ry.get('n')}, R "
                         f"{ry.get('R', float('nan')):.3f}, mean "
                         f"{ry.get('mean_deg', float('nan')):+.0f} deg, p "
                         f"{ry.get('p_approx', float('nan')):.2g}")
        for nm, tg in (("3g", trig3), ("7g", trig7)):
            if tg.get("n"):
                lines.append(
                    f"   triggered thrust - mean around >= {nm} (n "
                    f"{tg['n']}), lags {tg['lags_s'][0]:.2f}..."
                    f"{tg['lags_s'][-1]:.2f} s: "
                    + " ".join(f"{v:+.3f}" for v in tg["mean"])
                    + "; null 95% band at lag -1 step "
                    f"[{tg['null_lo'][11]:+.3f}, {tg['null_hi'][11]:+.3f}]")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(_out(), "analysis.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(os.path.join(_out(), "analysis.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="hand,m0")
    ap.add_argument("--missions", default="0-7")
    ap.add_argument("--legs", default="0,1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--min-gb", type=float, default=2.0)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    from studies.final_eval import parse_range
    rows = a.rows.split(",")
    if a.analyze:
        analyze(rows)
        return
    run(rows, parse_range(a.missions), parse_range(a.legs), a.device,
        a.min_gb)


if __name__ == "__main__":
    main()
