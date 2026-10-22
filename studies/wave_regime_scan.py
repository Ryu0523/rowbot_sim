#!/usr/bin/env python3
"""
REGIME SCAN: where, across speed, relative wave heading and sea, does the
evaluation plant show hazards tied to INDIVIDUAL waves (wave-synchronous),
and where only hazards set by the mean speed and heading?

Every run is built and stepped exactly as studies/wave_phase_oracle.py's
constant-throttle runs (which follow studies/final_eval.py's m0 row, steer
'auto'), reusing its SafeMode and bow_phase:
  - studies/final_eval.make_mission(seed, leg=0, T, sea=dict(hs, tp,
    theta0)): learn/repro/task.Mission in the high-fidelity world with the
    recording PlaningVesselGZ twin; track heading 0 (+x); the sea turned so
    that the relative heading chi (0 head, 45, 90 beam, 135 quartering, 180
    following) holds: theta0 (direction the waves TRAVEL toward,
    step0_preview_spec.WaveField) = pi - chi. Default spectrum of the task
    (JONSWAP gamma 3.3, cos^2s spreading s 8, n_freq 24, n_dir 5).
  - nozzle: the mission's heading autopilot m.ep._steer(m.s, thrust) with
    the commanded thrust; the measured-roll SafeMode; 12 plant steps of
    0.02 s per command, m.advance(thrust, nozzle).
  - thrust: constant f t_max.
  - the run starts at the CALM steady speed of f (studies/_cache/
    wave_regimes/validity.txt V2 and a free-running calm check), not at
    final_eval's 0.8 u_design, so that low throttles do not spend the run
    decelerating; every statistic below is taken after T_SKIP = 20 s.

Per run (after 20 s): speed made good and its wave-locked part; measured
encounter period (zero up-crossings of the surface at the bow) and signed
encounter frequency (phase rate of the surface's analytic signal at the
CG, spectrum-weighted theory beside it); the section-9 safety events of
studies/safety_events (k = 2, as final_eval) on the trajectory after 20 s;
surf-riding indicators; heading / yaw-rate / nozzle-saturation (broach)
indicators; roll, safe mode, capsize; the pump's prime; and how strongly
each hazard is locked to the individual wave (phase concentration and the
wave envelope at the events).

  python -m studies.wave_regime_scan --phase run --workers 2
  python -m studies.wave_regime_scan --phase report
Output: studies/_cache/wave_regimes/scan.pkl, scan_report.txt, scan.log
"""
import argparse
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

OUT = os.path.join(HERE, "_cache", "wave_regimes")
KN = 0.514444
G = 9.81
PI = math.pi
T_SKIP = 20.0
# calm steady speed (kn) of constant throttle f: validity.txt V2 / the
# free-running calm runs of the validity probe (f 0.05 -> 6.9 kn free)
U_CALM = {0.03: 5.0, 0.05: 6.9, 0.1: 9.2, 0.15: 9.5, 0.2: 10.7,
          0.25: 18.5, 0.3: 22.2, 0.45: 27.1, 0.6: 31.6, 0.8: 38.1,
          1.0: 43.9}
SEAS = {"tp5": dict(hs=1.0, tp=5.0), "tp8": dict(hs=1.0, tp=8.0)}
CHIS = (0, 45, 90, 135, 180)


def log(msg, name="scan.log"):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_gb():
    from learn.repro.task import avail_gb
    return avail_gb()


def wrap(a):
    return (np.asarray(a) + PI) % (2 * PI) - PI


def harmonic_r2(theta, x):
    """(R^2, amplitude, phase of max) of x ~ c0 + a cos theta + b sin
    theta: the share of x's variance that follows the local wave phase."""
    theta = np.asarray(theta, float)
    x = np.asarray(x, float)
    ok = np.isfinite(theta) & np.isfinite(x)
    theta, x = theta[ok], x[ok]
    if x.size < 10 or np.var(x) <= 0:
        return float("nan"), float("nan"), float("nan")
    A = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], 1)
    coef = np.linalg.lstsq(A, x, rcond=None)[0]
    res = x - A @ coef
    return (float(1.0 - res.var() / x.var()),
            float(math.hypot(coef[1], coef[2])),
            float(math.atan2(coef[2], coef[1])))


def circ_r(ph):
    ph = np.asarray(ph, float)
    if ph.size == 0:
        return float("nan"), float("nan")
    z = np.mean(np.exp(1j * ph))
    return float(abs(z)), float(np.degrees(np.angle(z)))


def runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def wave_signals(sea, s, t, x_b):
    """At every row of states s (n, 14) and times t: the surface at the
    bow, the bow's analytic signal with per-component sign flips so that
    its phase increases (studies/wave_phase_oracle.bow_phase, vectorised),
    the CG's analytic signal WITHOUT flips (its phase rate is minus the
    signed encounter frequency), and the spectrum-weighted signed encounter
    frequency at the row's velocity."""
    psi = s[:, 5]
    c, sn = np.cos(psi), np.sin(psi)
    cth, sth = np.cos(sea.th), np.sin(sea.th)
    vx = s[:, 6] * c - s[:, 7] * sn
    vy = s[:, 6] * sn + s[:, 7] * c
    we = sea.w[None, :] - sea.k[None, :] * (vx[:, None] * cth[None, :]
                                             + vy[:, None] * sth[None, :])
    a = sea.a[None, :]
    # bow
    X = s[:, 0] + x_b * c
    Y = s[:, 1] + x_b * sn
    p = (sea.k[None, :] * (X[:, None] * cth + Y[:, None] * sth)
         - sea.w[None, :] * t[:, None] + sea.phi[None, :])
    sg = np.where(we >= 0.0, 1.0, -1.0)
    zb = np.sum(a * np.exp(-1j * sg * p), 1)
    eta_b = np.sum(a * np.cos(p), 1)
    # CG, raw
    pc = (sea.k[None, :] * (s[:, 0, None] * cth + s[:, 1, None] * sth)
          - sea.w[None, :] * t[:, None] + sea.phi[None, :])
    zc = np.sum(a * np.exp(1j * pc), 1)
    w2 = sea.a ** 2
    we_w = (we * w2[None, :]).sum(1) / w2.sum()
    return eta_b, zb, zc, we_w


def upcross_period(x, dt):
    x = np.asarray(x, float) - np.mean(x)
    i = np.nonzero((x[:-1] < 0) & (x[1:] >= 0))[0]
    if i.size < 2:
        return float("nan"), int(i.size)
    tc = i + (-x[i]) / (x[i + 1] - x[i])
    return float(np.mean(np.diff(tc)) * dt), int(i.size)


# ------------------------------------------------------------ one run
def run_one(sea_name, chi, f, seed, T):
    from studies import final_eval as FE
    from studies import safety_events as se
    from studies.wave_phase_oracle import SafeMode, bow_phase
    t0 = time.perf_counter()
    sd = dict(SEAS[sea_name])
    sd["theta0"] = float(wrap(PI - math.radians(chi)))
    m = FE.make_mission(seed, 0, T, sea=sd)
    pl = m.plant
    u0 = U_CALM[f] * KN
    m.s = pl.initial_state(u0)
    m.s[5] = m.phi
    x_b = float(pl.sec.x_bow)
    dtc = m.dt * m.sub
    safe = SafeMode(dtc)
    sea = m.sea
    S = {k: [] for k in ("theta", "amp", "u", "ua", "y", "e", "r", "roll",
                         "prime", "noz", "thr", "safe", "uw")}
    n_steps = 0
    while not m.done():
        th, amp, _ = bow_phase(sea, m.s, m.t, x_b)
        thrust = f * m.t_max
        rud = m.ep._steer(m.s, thrust)
        thrust, rud, on = safe(m.s[3], thrust, rud)
        S["theta"].append(th)
        S["amp"].append(amp)
        S["safe"].append(on)
        S["prime"].append(float(pl.prop.prime))
        m.advance(thrust, rud)
        n_steps += 1
        s = m.s
        ua, yt = m.along_cross(s)
        S["u"].append(float(s[6]))
        S["ua"].append(float(ua))
        S["y"].append(float(yt))
        S["e"].append(float(wrap(s[5] - m.phi)))
        S["r"].append(float(s[11]))
        S["roll"].append(float(s[3]))
        S["noz"].append(float(s[13]) / m.rud_max)
        S["thr"].append(float(s[12]) / m.t_max)
        mu = s[5] - sea.theta0
        S["uw"].append(float(s[6] * math.cos(mu) - s[7] * math.sin(mu)))
        if not m.finite or getattr(pl, "capsized", False):
            break
    S = {k: np.asarray(v, np.float32 if k != "safe" else bool)
         for k, v in S.items()}
    met = m.metrics()
    traj = pl.trajectory()
    dt = float(traj["meta"]["dt"])
    n0 = int(round(T_SKIP / dt))
    k0 = int(round(T_SKIP / dtc))
    cut = {kk: (v[n0:] if isinstance(v, np.ndarray) and v.shape[:1]
                == traj["t"].shape else v) for kk, v in traj.items()}
    if len(cut["t"]) < 10:          # ended (capsize / non-finite) early
        cut = traj
        n0, k0 = 0, 0
    raw = se.safety_events(cut, k=2.0, k_report=(1.5, 2.5), track=float(m.phi))
    ev = FE.flatten_events(raw, 2.0)
    # ---- waves along the run (plant step, after the skip)
    tt, ss = cut["t"], cut["s"]
    eta_b, zb, zc, we_w = wave_signals(sea, ss, tt, x_b)
    Tz, n_up = upcross_period(eta_b, dt)
    th_b = np.angle(zb)
    env_b = np.abs(zb)
    ph_c = np.unwrap(np.angle(zc))
    # signed encounter frequency measured: minus the CG phase rate over
    # the run, and smoothed (2 s centred) for the surf-riding test
    dur = tt[-1] - tt[0] if len(tt) > 1 else float("nan")
    we_meas = float(-(ph_c[-1] - ph_c[0]) / dur) if dur > 0 else float("nan")
    h2 = int(round(1.0 / dt))
    we_inst = np.full(len(tt), np.nan)
    if len(tt) > 2 * h2 + 1:
        we_inst[h2:-h2] = -(ph_c[2 * h2:] - ph_c[:-2 * h2]) / (2 * h2 * dt)
    tp = SEAS[sea_name]["tp"]
    wp = 2 * PI / tp
    c_p = G / wp
    mu = ss[:, 5] - sea.theta0
    u_w = ss[:, 6] * np.cos(mu) - ss[:, 7] * np.sin(mu)
    band = (u_w >= 0.7 * c_p) & (u_w <= 1.15 * c_p)
    lock = np.abs(we_inst) < 0.2 * wp
    surf = lock & band
    sr = runs(surf)
    surf_long = max([(b - a) * dt for a, b in sr], default=0.0)
    # captured: the CG stays within a quarter wave (pi/2 of phase) of
    # where it is, for at least one peak period
    nw = int(round(tp / dt))
    capt = np.zeros(len(tt), bool)
    if len(tt) > nw:
        from numpy.lib.stride_tricks import sliding_window_view
        W = sliding_window_view(ph_c, nw + 1)
        capt[:len(W)] = (W.max(1) - W.min(1)) < PI / 2
    cr = runs(capt)
    capt_long = max([(b - a) * dt for a, b in cr], default=0.0)
    # ---- wave-locking of the hazards
    idx, pk = se.acc_peaks(cut["a_cg"], dt, 2.0)
    env_rank = np.argsort(np.argsort(env_b)) / max(len(env_b) - 1, 1)
    sel2, sel7 = pk >= 2.0, pk >= 7.0
    R2, ph2 = circ_r(th_b[idx[sel2]])
    R7, ph7 = circ_r(th_b[idx[sel7]])
    envp2 = float(np.mean(env_rank[idx[sel2]])) if sel2.any() else np.nan
    envp7 = float(np.mean(env_rank[idx[sel7]])) if sel7.any() else np.nan
    hb = se.bow_height(cut)
    wet = runs(hb < 0.0)
    wi = np.array([a for a, _ in wet], int)
    Rw, phw = circ_r(th_b[wi]) if wi.size else (np.nan, np.nan)
    envpw = float(np.mean(env_rank[wi])) if wi.size else np.nan
    e = wrap(ss[:, 5] - m.phi)
    hw = runs(np.abs(e) > math.radians(10.0))
    hi = np.array([a for a, _ in hw], int)
    envph = float(np.mean(env_rank[hi])) if hi.size else np.nan
    # ---- series after the skip, at the control step
    sk = {k: v[k0:] for k, v in S.items()}
    r2u = harmonic_r2(sk["theta"], sk["u"])
    r2r = harmonic_r2(sk["theta"], sk["r"])
    r2e = harmonic_r2(sk["theta"], sk["e"])
    r2p = harmonic_r2(sk["theta"], sk["roll"])
    ua = sk["ua"].astype(float)
    nh = len(ua) // 2
    r = dict(
        sea=sea_name, chi=int(chi), f=float(f), seed=int(seed),
        theta0=float(sea.theta0), tp=tp, c_p=c_p, wp=wp,
        kn=float(ua.mean() / KN) if ua.size else np.nan,
        kn_u=float(sk["u"].mean() / KN) if ua.size else np.nan,
        kn_h1=float(ua[:nh].mean() / KN) if nh else np.nan,
        kn_h2=float(ua[nh:].mean() / KN) if nh else np.nan,
        u_sd_kn=float(sk["u"].std() / KN) if ua.size else np.nan,
        u_lock_r2=r2u[0], u_lock_amp_kn=r2u[1] / KN,
        uw_over_c=float(np.mean(u_w) / c_p),
        we_meas=we_meas, we_theory=float(np.mean(we_w)),
        we_abs_flip=float(np.mean(np.abs(np.diff(np.unwrap(th_b)))) / dt)
        if len(th_b) > 1 else np.nan,
        Tz_bow=Tz, n_upcross=n_up,
        surf_share=float(np.mean(surf)), surf_long_s=float(surf_long),
        lock_share=float(np.mean(lock[np.isfinite(we_inst)]))
        if np.isfinite(we_inst).any() else np.nan,
        band_share=float(np.mean(band)), capt_share=float(np.mean(capt)),
        capt_long_s=float(capt_long),
        ev=ev, met=met,
        heading_max_deg=float(raw["yaw"]["heading_dev_max_deg"]),
        heading_rms_deg=float(np.degrees(np.sqrt(np.mean(e ** 2)))),
        r_max_degs=float(np.degrees(np.abs(ss[:, 11]).max(initial=0.0))),
        r_p99_degs=float(np.degrees(np.percentile(np.abs(ss[:, 11]), 99))),
        heading_p99_deg=float(np.degrees(np.percentile(np.abs(e), 99))),
        uw_max_over_c=float(np.max(u_w) / c_p),
        uw_gt09c_share=float(np.mean(u_w > 0.9 * c_p)),
        n_sub_max=int(np.max(cut["n_sub"])) if len(cut["n_sub"]) else 0,
        noz_sat_share=float(np.mean(np.abs(sk["noz"]) >= 0.95)),
        y_max=float(raw["track"]["max_abs_y"]),
        y_fail=bool(raw["track"]["fail_5L"]),
        roll_max_deg=float(raw["roll"]["max_deg"]),
        capsized=bool(getattr(pl, "capsized", False)),
        safe_entries=int(safe.n_entries), safe_share=float(sk["safe"].mean())
        if sk["safe"].size else np.nan,
        prime_lost_share=float(np.mean(sk["prime"] < 0.5)),
        prime_deficit=float(np.mean(1.0 - sk["prime"])),
        lock=dict(R2=R2, ph2=ph2, n2=int(sel2.sum()), env2=envp2,
                  R7=R7, ph7=ph7, n7=int(sel7.sum()), env7=envp7,
                  Rw=Rw, phw=phw, nw=int(wi.size), envw=envpw,
                  nh=int(hi.size), envh=envph,
                  r2_r=r2r[0], r2_e=r2e[0], r2_roll=r2p[0], r2_u=r2u[0]),
        numeric=raw["numerical"],
        finite=bool(m.finite), n_steps=n_steps,
        series={k: v for k, v in S.items()},
        t_run=time.perf_counter() - t0)
    return r


def _job(args):
    try:
        return args, run_one(*args)
    except Exception:
        import traceback
        return args, dict(error=traceback.format_exc())


def _init():
    import warnings
    warnings.filterwarnings("ignore")


def atomic_pickle(obj, path):
    with open(path + ".tmp", "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(path + ".tmp", path)


def run_jobs(jobs, path, workers, min_gb):
    R = pickle.load(open(path, "rb")) if os.path.exists(path) else \
        dict(results={}, errors={})
    todo = [j for j in jobs if j[:4] not in R["results"]]
    log(f"{len(jobs)} jobs, {len(todo)} to do, {workers} workers, "
        f"{free_gb():.1f} GB free -> {os.path.relpath(path, ROOT)}")
    if not todo:
        return R
    if free_gb() < min_gb:
        raise SystemExit(f"only {free_gb():.1f} GB free (< {min_gb})")
    t0 = time.time()
    n_done, last = 0, time.time()
    if workers <= 1:
        it, pool = map(_job, todo), None
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(workers, initializer=_init,
                                            maxtasksperchild=25)
        it = pool.imap_unordered(_job, todo, chunksize=1)
    try:
        for key, r in it:
            n_done += 1
            k4 = key[:4]
            if "error" in r:
                log(f"  ERROR {k4}: {r['error'].splitlines()[-1]}")
                R["errors"][k4] = r["error"]
                continue
            R["results"][k4] = r
            e = r["ev"]
            el = time.time() - t0
            log(f"  {n_done}/{len(todo)} {k4}: {r['kn']:.1f} kn, Tz "
                f"{r['Tz_bow']:.2f} s, we {r['we_meas']:+.2f}, A1/10 "
                f"{e['a110']:.2f} g, >=7g {e['n_ge7']}, wet {e['bow_wet']}, "
                f"bury {e['bow_burial']}, hdg {r['heading_max_deg']:.0f} deg,"
                f" roll {r['roll_max_deg']:.0f}, surf {r['surf_share']:.2f}, "
                f"broach {e['broach']}, cap {int(r['capsized'])} "
                f"({r['t_run']:.0f} s; ETA "
                f"{el / n_done * (len(todo) - n_done) / 60:.0f} min; "
                f"{free_gb():.1f} GB)")
            if time.time() - last > 60:
                atomic_pickle(R, path)
                last = time.time()
            if free_gb() < 1.0:
                log("  RAM below 1 GB: stopping (resumable)")
                break
    finally:
        atomic_pickle(R, path)
        if pool is not None:
            pool.terminate()
            pool.join()
    log(f"  done in {(time.time() - t0) / 60:.1f} min")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="run")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--T", type=float, default=120.0)
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--seas", default="tp5,tp8")
    ap.add_argument("--chis", default="0,45,90,135,180")
    ap.add_argument("--f", default="0.03,0.05,0.1,0.2,0.25,0.3,0.45,0.6,"
                                   "0.8,1.0")
    ap.add_argument("--min-gb", type=float, default=2.0)
    ap.add_argument("--out", default="scan.pkl")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",") if x]
    seas = [x for x in a.seas.split(",") if x]
    chis = [int(x) for x in a.chis.split(",") if x]
    fs = [float(x) for x in a.f.split(",") if x]
    if "run" in a.phase:
        jobs = [(sn, c, f, s, a.T) for s in seeds for sn in seas
                for c in chis for f in fs]
        run_jobs(jobs, os.path.join(OUT, a.out), a.workers, a.min_gb)


if __name__ == "__main__":
    main()
