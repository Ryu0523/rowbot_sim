#!/usr/bin/env python3
"""
ORACLE test: would wave-phase-locked throttle ("accelerate while climbing
the face, ease off at the crest") help at all, in the evaluation sea, on
the evaluation plant -- independent of any MPC.

Every mission is built and scored exactly as studies/final_eval.py builds
and scores the m0 row (steer 'auto'):
  - studies/final_eval.make_mission(seed, leg, T) with OPT plant 'gz': the
    learn/repro/task.Mission in the high-fidelity world with the recording
    PlaningVesselGZ twin (sea state 3, Hs 1.0 m, Tp 5 s, waves from pi; leg
    0 = head seas, leg 1 = 45 deg off the bow);
  - nozzle: the mission's own heading autopilot m.ep._steer(m.s, thrust)
    (sim/env.Episode._steer), called with the commanded thrust, as
    final_eval.run_mission does for steer 'auto';
  - the same measured-roll safe mode (learn/meta/mpc_constrained.SafeMode,
    copied verbatim below because that module imports torch);
  - one command per control step (12 plant steps of 0.02 s = 0.24 s),
    m.advance(thrust, nozzle);
  - events: studies/safety_events.safety_events(plant.trajectory(), k=2.0,
    k_report=(1.5, 2.5), track=phi), flattened by
    final_eval.flatten_events; speed = Mission.metrics()['u_along'] (mean
    along-track speed over the mission) as final_eval's 'kn'.

Only the thrust command differs (fraction f of t_max, clipped to [0, 1] as
the constrained MPPI's):
  const  f = F
  pl     f = Tm (1 + d cos(theta - phi0))      (phase-locked)
  bb     f = lo while wrap(theta - start) in [0, width), else 1.0
         (bang-bang: full thrust, cut for a window around the crest)
theta: the instantaneous wave phase at the bow (centreline bow station,
plant.sec.x_bow) from the sea's own components as an analytic signal,
  z = sum_j a_j exp(-i s_j p_j),  p_j = k_j (x cos th_j + y sin th_j)
                                        - w_j t + phi_j
(sim/wavefield / step0_preview_spec.WaveField: eta = sum a cos p, so
Re z = eta exactly), s_j = sign of the component's encounter frequency
w_j - k_j V.(cos th_j, sin th_j) at the boat's velocity, so that
theta = arg z INCREASES with time: theta = 0 crest at the bow, +-pi
trough, theta in (-pi, 0) the bow is climbing the face (elevation rising),
(0, pi) the back. The oracle knows the sea exactly (perfect knowledge).

Phases (results in studies/_cache/wave_phase/):
  calib   constant throttle 0.4..1.0 on seed 0, both legs (sets the grid)
  run     every setting x missions, paired; resumable (oracle.pkl)
  report  matched-speed comparison against the constant-throttle curve

  python -m studies.wave_phase_oracle --phase run,report --workers 4
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

OUT = os.path.join(HERE, "_cache", "wave_phase")
KN = 0.514444
PI = math.pi


def log(msg, name="oracle.log"):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_gb():
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


# ------------------------------------------------------------ safe mode
class SafeMode:
    """VERBATIM copy of learn/meta/mpc_constrained.SafeMode (that module
    imports torch, which these workers do not need): |roll| above 20 deg ->
    throttle halved, nozzle centred; exit after 2 s with |roll| below
    10 deg."""

    def __init__(self, dt_ctrl, enter_deg=20.0, exit_deg=10.0, hold_s=2.0):
        self.dt, self.enter = dt_ctrl, math.radians(enter_deg)
        self.exit, self.hold = math.radians(exit_deg), hold_s
        self.on, self.calm, self.n_on, self.n_entries = False, 0.0, 0, 0

    def __call__(self, roll, thrust, rud):
        a = abs(float(roll))
        if not self.on and a > self.enter:
            self.on, self.calm = True, 0.0
            self.n_entries += 1
        elif self.on:
            self.calm = self.calm + self.dt if a < self.exit else 0.0
            if self.calm >= self.hold:
                self.on = False
        if self.on:
            self.n_on += 1
            return 0.5 * thrust, 0.0, True
        return thrust, rud, False


# ------------------------------------------------------------ wave phase
def bow_phase(sea, s, t, x_b):
    """(theta, envelope, eta) at the centreline bow station of state s at
    time t (see the module docstring)."""
    psi = float(s[5])
    c, sn = math.cos(psi), math.sin(psi)
    X = s[0] + x_b * c
    Y = s[1] + x_b * sn
    cth, sth = np.cos(sea.th), np.sin(sea.th)
    p = sea.k * (X * cth + Y * sth) - sea.w * t + sea.phi
    vx = s[6] * c - s[7] * sn
    vy = s[6] * sn + s[7] * c
    we = sea.w - sea.k * (vx * cth + vy * sth)
    sg = np.where(we >= 0.0, 1.0, -1.0)
    z = np.sum(sea.a * np.exp(-1j * sg * p))
    return float(np.angle(z)), float(abs(z)), float(z.real)


def wrap(a):
    return (a + PI) % (2 * PI) - PI


# ------------------------------------------------------------ settings
def settings_grid(const_levels, tm_levels, d_levels=(0.3, 0.6),
                  n_phi=8, bb_starts=None, bb_widths=(PI / 2, PI),
                  bb_lo=0.3):
    out = []
    for f in const_levels:
        out.append(dict(kind="const", f=float(f), name=f"const_{f:.2f}"))
    phis = [wrap(-PI + (i + 1) * 2 * PI / n_phi) for i in range(n_phi)]
    for tm in tm_levels:
        for d in d_levels:
            for ph in phis:
                deg = int(round(math.degrees(ph)))
                out.append(dict(kind="pl", tm=float(tm), d=float(d),
                                phi0=float(ph),
                                name=f"pl_{tm:.2f}_{d:.1f}_{deg:+d}"))
    if bb_starts is None:
        bb_starts = (-PI, -3 * PI / 4, -PI / 2, -PI / 4, 0.0)
    for st in bb_starts:
        for w in bb_widths:
            out.append(dict(kind="bb", lo=float(bb_lo), start=float(st),
                            width=float(w),
                            name=f"bb_{bb_lo:.1f}_{int(round(math.degrees(st))):+d}"
                                 f"_{int(round(math.degrees(w)))}"))
    return out


def command(st, theta):
    k = st["kind"]
    if k == "const":
        f = st["f"]
    elif k == "pl":
        f = st["tm"] * (1.0 + st["d"] * math.cos(theta - st["phi0"]))
    elif k == "bb":
        f = st["lo"] if (wrap(theta - st["start"]) % (2 * PI)) < st["width"] \
            else 1.0
    else:
        raise ValueError(k)
    return float(min(max(f, 0.0), 1.0))


# ------------------------------------------------------------ one mission
def run_one(st, seed, leg, T):
    from studies import final_eval as FE
    from studies import safety_events as se
    t0 = time.perf_counter()
    m = FE.make_mission(seed, leg, T)
    t_build = time.perf_counter() - t0
    pl = m.plant
    x_b = float(pl.sec.x_bow)
    dtc = m.dt * m.sub
    safe = SafeMode(dtc)
    sea = m.sea
    TH, AMP, ETA, FC, THR, U, UA, SAFE = ([] for _ in range(8))
    n_steps = 0
    while not m.done():
        th, amp, eta = bow_phase(sea, m.s, m.t, x_b)
        f = command(st, th)
        thrust = f * m.t_max
        rud = m.ep._steer(m.s, thrust)
        thrust, rud, on = safe(m.s[3], thrust, rud)
        TH.append(th)
        AMP.append(amp)
        ETA.append(eta)
        FC.append(thrust / m.t_max)
        SAFE.append(on)
        # the thrust state at the same instant as theta (step start)
        THR.append(float(m.s[12]) / m.t_max)
        m.advance(thrust, rud)
        n_steps += 1
        U.append(float(m.s[6]))
        UA.append(float(m.along_cross(m.s)[0]))
        if not m.finite or getattr(m.plant, "capsized", False):
            break
    met = m.metrics()
    traj = m.plant.trajectory()
    raw = se.safety_events(traj, k=2.0, k_report=(1.5, 2.5),
                           track=float(m.phi))
    events = {str(kk): FE.flatten_events(raw, kk) for kk in FE.K_EVAL}
    # impact peaks (k = 2) >= 2 g with the bow wave phase at the peak
    idx, pk = se.acc_peaks(traj["a_cg"], float(traj["meta"]["dt"]), 2.0)
    sel = pk >= 2.0
    idx, pk = idx[sel], pk[sel]
    ph_pk = np.array([bow_phase(sea, traj["s"][i], float(traj["t"][i]),
                                x_b)[0] for i in idx])
    # the bow-wet spells' start phases
    hb = se.bow_height(traj)
    wet = se._runs(hb < 0.0)
    ph_wet = np.array([bow_phase(sea, traj["s"][a], float(traj["t"][a]),
                                 x_b)[0] for a, _ in wet])
    UA_ = np.asarray(UA)
    n20 = int(round(20.0 / dtc))
    r = dict(setting=st["name"], seed=int(seed), leg=int(leg),
             kn=met["u_along"] / KN if np.isfinite(met["u_along"])
             else float("nan"),
             kn_after20=float(UA_[n20:].mean()) / KN
             if len(UA_) > n20 else float("nan"),
             met=met, events=events,
             capsized=bool(getattr(m.plant, "capsized", False)),
             safe_entries=safe.n_entries, safe_steps=safe.n_on,
             n_steps=n_steps, x_bow=x_b,
             series=dict(theta=np.asarray(TH, np.float32),
                         amp=np.asarray(AMP, np.float32),
                         eta=np.asarray(ETA, np.float32),
                         f_cmd=np.asarray(FC, np.float32),
                         thr=np.asarray(THR, np.float32),
                         u=np.asarray(U, np.float32),
                         ua=np.asarray(UA, np.float32),
                         safe=np.asarray(SAFE, bool)),
             peaks=dict(t=traj["t"][idx].astype(np.float32),
                        g=pk.astype(np.float32),
                        theta=ph_pk.astype(np.float32)),
             wet_theta=ph_wet.astype(np.float32),
             t_build=t_build, t_run=time.perf_counter() - t0)
    return r


def _job(args):
    st, seed, leg, T = args
    try:
        return (st["name"], seed, leg), run_one(st, seed, leg, T)
    except Exception:
        import traceback
        return (st["name"], seed, leg), dict(error=traceback.format_exc())


def _init():
    import warnings
    warnings.filterwarnings("ignore")


def atomic_pickle(obj, path):
    pickle.dump(obj, open(path + ".tmp", "wb"))
    os.replace(path + ".tmp", path)


def run_jobs(jobs, path, workers, meta):
    R = pickle.load(open(path, "rb")) if os.path.exists(path) else \
        dict(results={}, meta=meta)
    R.setdefault("meta", meta)
    todo = [j for j in jobs if (j[0]["name"], j[1], j[2]) not in R["results"]]
    log(f"{len(jobs)} jobs, {len(todo)} to do, {workers} workers, "
        f"{free_gb():.1f} GB free -> {os.path.relpath(path, ROOT)}")
    if not todo:
        return R
    t0 = time.time()
    n_done, last_save = 0, time.time()
    if workers <= 1:
        it = map(_job, todo)
        pool = None
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(workers, initializer=_init,
                                            maxtasksperchild=40)
        it = pool.imap(_job, todo, chunksize=1)
    try:
        for key, r in it:
            n_done += 1
            if "error" in r:
                log(f"  ERROR {key}: {r['error'].splitlines()[-1]}")
                R.setdefault("errors", {})[key] = r["error"]
                continue
            R["results"][key] = r
            e = r["events"]["2.0"]
            if n_done % 10 == 0 or n_done == len(todo):
                el = time.time() - t0
                log(f"  {n_done}/{len(todo)} {key[0]} s{key[1]} l{key[2]}: "
                    f"{r['kn']:.1f} kn A1/10 {e['a110']:.2f} g "
                    f">=7g {e['n_ge7']} wet {e['bow_wet']} "
                    f"({r['t_run']:.0f} s; ETA "
                    f"{el / n_done * (len(todo) - n_done) / 60:.0f} min; "
                    f"{free_gb():.1f} GB free)")
            if time.time() - last_save > 60:
                atomic_pickle(R, path)
                last_save = time.time()
    finally:
        atomic_pickle(R, path)
        if pool is not None:
            pool.close()
            pool.join()
    log(f"  done in {(time.time() - t0) / 60:.1f} min")
    return R


# ------------------------------------------------------------ analysis
METRICS = (("a110", "A1/10 g"), ("n_ge7_per_h", ">=7g /h"),
           ("bow_wet_per_h", "wet /h"), ("peak_max", "peak g"),
           ("bow_burial_per_h", "bury /h"))
MAIN = METRICS[:3]


def metric(r, nm):
    return float(r["events"]["2.0"][nm])


def first_harmonic(r, key="thr", t0=20.0):
    """(c0, relative amplitude, phase of the maximum in theta) of a per-step
    series against the bow wave phase at the same instant, after t0 s:
    least squares x ~ c0 + a cos theta + b sin theta (theta is not
    uniformly sampled); amp = hypot(a, b) / c0, phase = atan2(b, a): the
    series ~ c0 (1 + amp cos(theta - phase))."""
    s = r["series"]
    n0 = int(round(t0 / 0.24))
    th = s["theta"][n0:].astype(float)
    x = s[key][n0:].astype(float)
    A = np.stack([np.ones_like(th), np.cos(th), np.sin(th)], 1)
    c0, a, b = np.linalg.lstsq(A, x, rcond=None)[0]
    return (float(c0), float(math.hypot(a, b) / max(abs(c0), 1e-9)),
            float(math.atan2(b, a)))


def _sorted(v, m):
    o = np.argsort(v)
    return v[o], m[o]


def _circ_deg(ph):
    return math.degrees(float(np.angle(np.mean(np.exp(1j * np.asarray(ph))))))


def matched(vc, Mc, v, Ms):
    """Per mission j: Ms[j] minus the constant curve (vc[:, j], Mc[:, j])
    interpolated at v[j]; in-range flags."""
    d = np.array([Ms[j] - np.interp(v[j], *_sorted(vc[:, j], Mc[:, j]))
                  for j in range(len(v))])
    inr = np.array([vc[:, j].min() <= v[j] <= vc[:, j].max()
                    for j in range(len(v))])
    return d, inr


def harmonic_test(D, ph, n_perm=4000, rng=None):
    """D (n_phi, n_mis) deltas, ph (n_phi,) phi0: the first harmonic of the
    per-phi0 mission mean over phi0, c1 = 2 mean_phi(mean_j D e^{i phi0})
    (delta ~ m + |c1| cos(phi0 - arg c1)); p = share of within-mission
    phi0 permutations with an amplitude at least as large."""
    rng = rng or np.random.default_rng(0)
    e = np.exp(1j * ph)

    def amp(X):
        return abs(2.0 * np.mean(X.mean(1) * e))
    a0 = amp(D)
    c1 = 2.0 * np.mean(D.mean(1) * e)
    cnt = 0
    for _ in range(n_perm):
        Xp = np.stack([rng.permutation(D[:, j]) for j in range(D.shape[1])],
                      1)
        cnt += amp(Xp) >= a0 - 1e-15
    return c1, (cnt + 1) / (n_perm + 1)


def report(path, seeds, legs):
    R = pickle.load(open(path, "rb"))
    res = R["results"]
    order = [s["name"] for s in R.get("meta", {}).get("settings", [])]
    for k in res:
        if k[0] not in order:
            order.append(k[0])
    names = [n for n in order if any(k[0] == n for k in res)]
    consts = sorted([n for n in names if n.startswith("const_")],
                    key=lambda n: float(n.split("_")[1]))
    others = [n for n in names if not n.startswith("const_")]
    lines = []
    P = lines.append
    P(f"wave-phase oracle: {len(res)} missions, {len(names)} settings, "
      f"T {R.get('meta', {}).get('T')} s; matched-speed deltas vs the "
      f"constant-throttle curve of the SAME sea (paired)")
    P("theta = wave phase at the bow: 0 crest, +-180 trough, -180..0 the bow"
      " climbing the face, 0..180 the back. pl = Tm (1 + d cos(theta -"
      " phi0)) -> command maximum at theta = phi0; bb = 1.0 except lo for"
      " theta in [start, start + width).")
    out = dict(const={}, cmp={}, null={}, harm={})
    for leg in legs:
        ok = [s for s in seeds if all((c, s, leg) in res for c in consts)]
        if not ok:
            continue
        P("")
        P(f"=== leg {leg} ({'head seas' if leg == 0 else '45 deg off the bow'}"
          f"), {len(ok)} paired missions (seeds {ok}) ===")
        P("constant throttle, means over missions (u-osc: surge-speed "
          "oscillation locked to theta after 20 s, kn amplitude @ theta of"
          " its max; w_enc: mean rate of theta after 20 s, rad/s):")
        P("  f       kn  kn>20 " + " ".join(f"{lab:>9s}"
                                            for _, lab in METRICS)
          + "   u-osc       w_enc")
        V = {}
        for c in consts:
            rr = [res[(c, s, leg)] for s in ok]
            v = np.array([r["kn"] for r in rr])
            M = {nm: np.array([metric(r, nm) for r in rr])
                 for nm, _ in METRICS}
            hu = np.array([first_harmonic(r, "u") for r in rr])
            we = np.mean([np.mean(np.diff(np.unwrap(
                r["series"]["theta"][83:].astype(float)))) / 0.24
                for r in rr])
            V[c] = (v, M)
            P(f"  {float(c.split('_')[1]):.2f} {v.mean():6.2f} "
              f"{np.mean([r['kn_after20'] for r in rr]):6.2f} "
              + " ".join(f"{M[nm].mean():9.2f}" for nm, _ in METRICS)
              + f"   {np.mean(hu[:, 0] * hu[:, 1]) / KN:4.2f}@"
              f"{_circ_deg(hu[:, 2]):+4.0f}   {we:4.2f}")
            out["const"][(leg, c)] = dict(kn=v, **M)
        vc = np.array([V[c][0] for c in consts])                 # (n_c, n_s)
        Mc = {nm: np.array([V[c][1][nm] for c in consts])
              for nm, _ in METRICS}
        vbar, Mbar = vc.mean(1), {nm: Mc[nm].mean(1) for nm, _ in METRICS}
        slope = np.polyfit(vbar, Mbar["a110"], 1)[0]
        P(f"  mean-curve A1/10 slope {slope:.3f} g per kn "
          f"({vbar.min():.1f}-{vbar.max():.1f} kn)")
        # the noise floor: each interior constant level against the curve
        # of the OTHER levels (leave one out) -- what a setting with no
        # phase effect at all scores
        P("")
        P("NULL (noise floor): each interior constant level vs the curve of "
          "the other levels, same matched-speed procedure (mean delta, "
          "#improved/#in range):")
        P("  level        " + "   ".join(f"{lab:>14s}" for _, lab in MAIN))
        nul = {nm: [] for nm, _ in MAIN}
        for i in range(1, len(consts) - 1):
            keep = [j for j in range(len(consts)) if j != i]
            row = []
            for nm, _ in MAIN:
                d, inr = matched(vc[keep], Mc[nm][keep], vc[i], Mc[nm][i])
                nul[nm].append(d[inr])
                row.append(f"{d[inr].mean():+6.2f} {np.sum(d[inr] < 0)}/"
                           f"{inr.sum()}".rjust(14))
            vb, ab = vbar[keep], Mbar["a110"][keep]
            o = np.argsort(ab)
            dvn = float(vbar[i] - np.interp(Mbar["a110"][i], ab[o], vb[o]))
            nul.setdefault("dv", []).append(dvn)
            P(f"  {consts[i]:12s} " + "   ".join(row)
              + f"   dv_eq {dvn:+.2f}")
        for nm, lab in MAIN:
            allv = np.concatenate(nul[nm])
            out["null"][(leg, nm)] = allv
            P(f"  pooled {lab:8s}: mean {allv.mean():+.2f}, "
              f"per-mission delta sd {allv.std():.2f}, "
              f"=> sd of a {len(ok)}-mission mean ~ "
              f"{allv.std() / math.sqrt(len(ok)):.2f}; share improved "
              f"{np.mean(allv < 0):.2f}")
        P(f"  dv_eq of the interior levels: sd {np.std(nul['dv']):.2f} kn")
        out["null"][(leg, "dv")] = np.array(nul["dv"])
        P("")
        P("settings vs the constant curve AT MATCHED SPEED (per mission). "
          "Per metric: mean delta (<0 = fewer than constant throttle at the "
          "same speed), #improved/#in range ('*' = some missions outside "
          "the constant curve's speed range, left out). dv_eq: the "
          "setting's mean speed minus the speed the mean constant curve has"
          " at the setting's mean A1/10 (kn; >0 = faster at equal A1/10). "
          "thr: realised thrust-state modulation (relative amplitude @ "
          "theta of its max), cmd: commanded, u-osc: surge-speed "
          "oscillation (kn @ theta of max).")
        P("  setting               kn  " + " ".join(f"{lab:>15s}"
                                                    for _, lab in MAIN)
          + "  dv_eq  thr       cmd       u-osc")
        for n in others:
            if not all((n, s, leg) in res for s in ok):
                continue
            rr = [res[(n, s, leg)] for s in ok]
            v = np.array([r["kn"] for r in rr])
            D, imp, cols = {}, {}, []
            for nm, _ in MAIN:
                ms = np.array([metric(r, nm) for r in rr])
                d, inr = matched(vc, Mc[nm], v, ms)
                D[nm] = d
                imp[nm] = int(np.sum(d[inr] < 0))
                cols.append(f"{d[inr].mean() if inr.any() else np.nan:+6.2f} "
                            f"{imp[nm]}/{inr.sum()}"
                            f"{'' if inr.all() else '*'}")
            a_s = np.mean([metric(r, "a110") for r in rr])
            o = np.argsort(Mbar["a110"])
            dv = float(v.mean() - np.interp(a_s, Mbar["a110"][o], vbar[o]))
            h = np.array([first_harmonic(r, "thr") for r in rr])
            hc = np.array([first_harmonic(r, "f_cmd") for r in rr])
            hu = np.array([first_harmonic(r, "u") for r in rr])
            P(f"  {n:20s} {v.mean():5.2f}  " + " ".join(
                f"{c_:>15s}" for c_ in cols)
                + f"  {dv:+5.2f}  {h[:, 1].mean():4.2f}@"
                f"{_circ_deg(h[:, 2]):+4.0f}"
                f"  {hc[:, 1].mean():4.2f}@{_circ_deg(hc[:, 2]):+4.0f}"
                f"  {np.mean(hu[:, 0] * hu[:, 1]) / KN:4.2f}@"
                f"{_circ_deg(hu[:, 2]):+4.0f}")
            out["cmp"][(leg, n)] = dict(kn=v, delta=D, improved=imp,
                                        in_range=inr, dv_eq=dv, thr_h1=h,
                                        cmd_h1=hc, u_h1=hu)
        # the best setting of each family against what the best of as many
        # no-effect settings would score (selection: the minimum of N
        # draws from the null spread of a mission mean)
        P("")
        P("best setting per family and metric vs selection by chance (null "
          "sd of a mission mean from the NULL block; E[best of N] = that sd "
          "x the expected minimum of N standard normals):")
        zrng = np.random.default_rng(2)
        for fam, pref in (("phase-locked", "pl_"), ("bang-bang", "bb_")):
            ks = [n for n in others if (leg, n) in out["cmp"]
                  and n.startswith(pref)]
            if not ks:
                continue
            emin = float(np.mean(zrng.standard_normal(
                (20000, len(ks))).min(1)))
            for nm, lab in MAIN:
                mu = np.array([out["cmp"][(leg, n)]["delta"][nm][
                    out["cmp"][(leg, n)]["in_range"]].mean() for n in ks])
                i = int(np.nanargmin(mu))
                c = out["cmp"][(leg, ks[i])]
                sdm = out["null"][(leg, nm)].std() / math.sqrt(len(ok))
                dvs = [out["cmp"][(leg, n)]["dv_eq"] for n in ks]
                P(f"  {fam:12s} {lab:8s}: best {ks[i]} {mu[i]:+.2f} "
                  f"({c['improved'][nm]}/{int(c['in_range'].sum())} "
                  f"improved); mean over its {len(ks)} settings "
                  f"{np.nanmean(mu):+.2f}; null sd of a mean {sdm:.2f}, "
                  f"E[best of {len(ks)}] {emin * sdm:+.2f}"
                  + (f"; dv_eq best {max(dvs):+.2f} kn "
                     f"({ks[int(np.argmax(dvs))]}), mean {np.mean(dvs):+.2f}"
                     if nm == "a110" else ""))
        # does the effect depend on phi0 at all? (permutation test)
        P("")
        P("phase-locked: dependence on phi0. c1 = first harmonic over the 8"
          " phi0 of the per-phi0 mean matched delta (delta ~ mean + |c1| "
          "cos(phi0 - arg c1)); best = phi0 of the lowest delta = arg c1 + "
          "180; p = within-mission permutation of the phi0 labels (4000); "
          "mean = average delta over phi0 (the cost of modulating at all):")
        tms = sorted({n.split("_")[1] for n in others if n.startswith("pl_")})
        ds = sorted({n.split("_")[2] for n in others if n.startswith("pl_")})
        rng = np.random.default_rng(1)
        for nm, lab in MAIN:
            pool = []
            for tm in tms:
                for d in ds:
                    ks = [k for k in out["cmp"] if k[0] == leg
                          and k[1].startswith(f"pl_{tm}_{d}_")]
                    if len(ks) < 8:
                        continue
                    ph = np.radians([float(k[1].split("_")[3]) for k in ks])
                    D = np.array([out["cmp"][k]["delta"][nm] for k in ks])
                    c1, p = harmonic_test(D, ph, rng=rng)
                    pool.append((D, ph, float(d)))
                    out["harm"][(leg, nm, tm, d)] = dict(c1=c1, p=p,
                                                         mean=D.mean())
                    srt = sorted(zip(ph, D.mean(1)))
                    P(f"  {lab:8s} Tm {tm} d {d}: mean {D.mean():+6.2f}, "
                      f"|c1| {abs(c1):5.2f}, best phi0 "
                      f"{math.degrees(np.angle(-c1)):+5.0f}, p {p:.3f}; "
                      + " ".join(f"{math.degrees(a):+.0f}:{x:+.2f}"
                                 for a, x in srt))
            if pool and all(np.allclose(q[1], pool[0][1]) for q in pool):
                # pooled over Tm and d: each block's deltas divided by its
                # d, the blocks stacked as extra missions
                Dp = np.concatenate([q[0] / q[2] for q in pool], 1)
                c1, p = harmonic_test(Dp, pool[0][1], rng=rng)
                P(f"  {lab:8s} POOLED over Tm and d (delta / d): |c1| "
                  f"{abs(c1):5.2f} per unit d, best phi0 "
                  f"{math.degrees(np.angle(-c1)):+5.0f}, p {p:.3f}")
                out["harm"][(leg, nm, "pooled")] = dict(c1=c1, p=p)
        # families of settings, paired per mission: the family mean delta
        # of each mission, then mean / sd / t over the missions
        P("")
        P("families (per mission: mean matched delta over the family's "
          "settings; then mean over missions, per-mission sd, t = mean / "
          "(sd / sqrt n), #missions improved; A1/10 also as kn on the mean "
          "constant curve's slope). 'phi0 in {0,+45}' was picked AFTER "
          "seeing the pooled harmonic (post hoc):")
        fams = (("all phase-locked", lambda n: n.startswith("pl_")),
                ("all bang-bang", lambda n: n.startswith("bb_")),
                ("pl phi0=-180 (thrust max on the face)",
                 lambda n: n.startswith("pl_") and n.endswith("_-180")),
                ("pl phi0 in {0,+45} (post hoc)",
                 lambda n: n.startswith("pl_")
                 and (n.endswith("_+0") or n.endswith("_+45"))))
        for lab_f, pred in fams:
            ks = [k for k in out["cmp"] if k[0] == leg and pred(k[1])]
            if not ks:
                continue
            for nm, lab in MAIN:
                Dm = np.array([np.where(out["cmp"][k]["in_range"],
                                        out["cmp"][k]["delta"][nm], np.nan)
                               for k in ks])
                pm = np.nanmean(Dm, 0)
                nn = int(np.sum(np.isfinite(pm)))
                mu, sd = float(np.nanmean(pm)), float(np.nanstd(pm, ddof=1))
                P(f"  {lab_f:38s} {lab:8s} ({len(ks):2d} settings): "
                  f"{mu:+7.2f} (sd {sd:6.2f}, t {mu / (sd / math.sqrt(nn)):+5.2f}"
                  f"), improved {int(np.sum(pm < 0))}/{nn}"
                  + (f", {-mu / slope:+.2f} kn" if nm == "a110" else ""))
                out.setdefault("family", {})[(leg, lab_f, nm)] = pm
        # the actuator: realised vs commanded modulation and its lag
        for d in ds:
            ks = [k for k in out["cmp"] if k[0] == leg
                  and k[1].startswith("pl_") and k[1].split("_")[2] == d]
            if not ks:
                continue
            ratio = np.mean([out["cmp"][k]["thr_h1"][:, 1].mean()
                             / out["cmp"][k]["cmd_h1"][:, 1].mean()
                             for k in ks])
            lag = [math.degrees(wrap(
                np.angle(np.mean(np.exp(1j * out["cmp"][k]["thr_h1"][:, 2])))
                - np.angle(np.mean(np.exp(1j * out["cmp"][k]["cmd_h1"][:, 2])))))
                for k in ks]
            P(f"  actuator, d {d}: realised / commanded thrust modulation "
              f"{ratio:.2f}, thrust maximum lags the command by "
              f"{np.mean(lag):+.0f} deg of wave phase ({min(lag):+.0f}.."
              f"{max(lag):+.0f})")
        # where the impacts and the wet bows happen in the wave phase
        P("")
        P("phase of the impact peaks (k=2) and the bow-wet entries in the "
          "constant-throttle runs: counts per 45-deg bin of theta, bins "
          "centred on -180,-135,...,+135 (0 crest, <0 face, >0 back):")
        bins = np.linspace(-PI - PI / 8, PI - PI / 8, 9)
        for c in consts:
            rr = [res[(c, s, leg)] for s in ok]
            pk = np.concatenate([r["peaks"]["theta"] for r in rr])
            g = np.concatenate([r["peaks"]["g"] for r in rr])
            wt = np.concatenate([r["wet_theta"] for r in rr])
            pkw = wrap(pk + PI / 8) - PI / 8
            wtw = wrap(wt + PI / 8) - PI / 8
            P(f"  {c}: >=2g {np.histogram(pkw, bins)[0].tolist()}  >=7g "
              f"{np.histogram(pkw[g >= 7.0], bins)[0].tolist()}  wet "
              f"{np.histogram(wtw, bins)[0].tolist()}")
        rr = [res[(c, s, leg)] for c in consts for s in ok]
        pk = wrap(np.concatenate([r["peaks"]["theta"] for r in rr]
                                 ).astype(float))
        g = np.concatenate([r["peaks"]["g"] for r in rr])
        wt = wrap(np.concatenate([r["wet_theta"] for r in rr]).astype(float))
        face = (lambda x: (x >= -7 * PI / 8) & (x < -PI / 8))
        cb = (lambda x: (x >= -PI / 8) & (x < 5 * PI / 8))
        p7 = pk[g >= 7.0]
        P(f"  all constant runs: >=7g peaks {len(p7)}: face (-157.5..-22.5)"
          f" {np.mean(face(p7)):.2f}, crest+back (-22.5..112.5) "
          f"{np.mean(cb(p7)):.2f}, rest (trough side) "
          f"{1 - np.mean(face(p7)) - np.mean(cb(p7)):.2f}; bow-wet entries "
          f"{len(wt)}: face {np.mean(face(wt)):.2f}, crest+back "
          f"{np.mean(cb(wt)):.2f}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "oracle_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    atomic_pickle(out, os.path.join(OUT, "oracle_analysis.pkl"))
    print(txt)
    return out


# ------------------------------------------------------------ main
def parse_list(s, typ=float):
    return [typ(x) for x in s.split(",") if x != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="run,report")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--T", type=float, default=120.0)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--legs", default="0,1")
    ap.add_argument("--const", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--tm", default="0.55,0.7,0.85")
    ap.add_argument("--d", default="0.3,0.6")
    ap.add_argument("--bb-lo", type=float, default=0.3)
    ap.add_argument("--min-gb", type=float, default=2.5)
    ap.add_argument("--out", default="oracle.pkl")
    a = ap.parse_args()
    seeds, legs = parse_list(a.seeds, int), parse_list(a.legs, int)
    os.makedirs(OUT, exist_ok=True)
    phases = a.phase.split(",")
    if "calib" in phases:
        st = settings_grid(parse_list(a.const), [], d_levels=(),
                           bb_starts=())
        jobs = [(s, 0, leg, a.T) for leg in legs for s in st]
        R = run_jobs(jobs, os.path.join(OUT, "calib.pkl"), a.workers,
                     dict(kind="calib"))
        for leg in legs:
            for s in st:
                r = R["results"].get((s["name"], 0, leg))
                if r:
                    e = r["events"]["2.0"]
                    log(f"  calib leg {leg} {s['name']}: {r['kn']:.2f} kn "
                        f"(after 20 s {r['kn_after20']:.2f}) A1/10 "
                        f"{e['a110']:.2f} g >=7g {e['n_ge7']} wet "
                        f"{e['bow_wet']} thr_mean "
                        f"{r['series']['thr'].mean():.3f}")
    if "run" in phases:
        g = free_gb()
        if g < a.min_gb:
            raise SystemExit(f"only {g:.1f} GB free (< {a.min_gb})")
        st = settings_grid(parse_list(a.const), parse_list(a.tm),
                           d_levels=tuple(parse_list(a.d)), bb_lo=a.bb_lo)
        # mission-major: an interrupted run still has every setting on the
        # missions it finished (paired)
        jobs = [(s, seed, leg, a.T) for seed in seeds for leg in legs
                for s in st]
        run_jobs(jobs, os.path.join(OUT, a.out), a.workers,
                 dict(T=a.T, settings=st, seeds=seeds, legs=legs,
                      note="see studies/wave_phase_oracle.py"))
    if "report" in phases:
        report(os.path.join(OUT, a.out), seeds, legs)


if __name__ == "__main__":
    main()
