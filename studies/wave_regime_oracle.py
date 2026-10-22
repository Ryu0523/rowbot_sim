#!/usr/bin/env python3
"""
ORACLE wave-timed throttle (and one heading rule) in FOLLOWING and
QUARTERING seas, the regime where studies/wave_regime_scan.py found hazards
tied to individual waves (surf-riding capture near the wave speed, bow
wetting / burial and large heading excursions when the boat overtakes the
waves).  Question: does throttle timed to the individual wave, with PERFECT
knowledge of the sea, beat the best CONSTANT throttle at the same speed?

Every run is built, stepped and scored exactly as the scan's runs (which
follow studies/final_eval.py's m0 row with steer 'auto', and
studies/wave_phase_oracle.py's oracle):
  - studies/final_eval.make_mission(seed, leg 0, T, sea=dict(hs, tp,
    theta0)), track heading 0 (+x); the sea turned so that the relative
    heading chi holds: theta0 (direction the waves TRAVEL toward) = pi - chi
    (chi 180 following, 135 quartering);
  - nozzle: the mission's heading autopilot m.ep._steer(m.s, thrust) with
    the commanded thrust; the measured-roll SafeMode (imported from
    studies/wave_phase_oracle, a verbatim copy of
    learn/meta/mpc_constrained.SafeMode); 12 plant steps of 0.02 s per
    command; m.advance(thrust, nozzle);
  - start at the calm steady speed of the setting's nominal mean throttle
    (studies/wave_regime_scan.U_CALM, linearly interpolated); every
    statistic is taken after T_SKIP = 20 s (as the scan).

THE ORACLE'S WAVE SIGNALS (perfect knowledge, from the sea's own components,
sim/wavefield.SeaState: eta = sum a_j cos p_j, p_j = k_j (x cos th_j + y sin
th_j) - w_j t + phi_j), at a hull station (bow by default) of the state,
optionally evaluated `lead` seconds ahead (the point moved on at the boat's
present velocity, the sea at t + lead):
  z  = sum a_j exp(i p_j)                     (spatial analytic signal)
  ph = arg z: the bow's place on the wave.  In following / quartering seas
       ph = 0 crest at the bow; ph in (0, 180) deg the bow is AHEAD of the
       crest, on its front (downhill) face -- the surface falls ahead; +-180
       trough; ph in (-180, 0) the bow is on the BACK of the wave ahead (the
       surface rises ahead).  ph increases while the boat overtakes the
       waves and decreases while the waves overtake the boat.
  rate = d ph / dt following the point = Im(zdot / z) (recorded only;
       it spikes where |z| is small)
  slope = Re(sum i k_j cos(th_j - psi) a_j exp(i p_j)) = d eta / d s along
       the boat's heading (< 0: the bow points downhill); senv = its
       envelope |.| (local steepness along the heading).
  sig_s = sqrt(sum (a_j k_j cos(th_j - track))^2 / 2): the rms slope along
       the track of the mission's sea (the thresholds' unit).

THRUST LAWS (fraction f of t_max, clipped to [0, 1]):
  const  f = F
  pl     f = Tm (1 + d cos(ph - phi0))                 (phase-locked to the
         bow's place on the wave; command maximum at ph = phi0)
  bb     f = lo while (ph - start) mod 360 < width, else hi
  hold   'hold the boat on the back of the wave' (a skilled driver's rule):
         f = Fc - Kp wrap(ph - ph_t) - Kd v_rel, clipped to [lo, hi];
         v_rel = the boat's speed along the waves' travel direction minus
         the peak celerity c_p (m/s; the analytic phase rate is too spiky
         where the envelope is small to be used as the damping signal)
  steep  'slow down for steep waves': f = lo while senv > m sig_s, else hi
  face   'ease off when the bow starts down a steep face': f = lo while
         slope < -m sig_s, else hi
  boost  (low-speed cell only) 'power on when the wave overtakes': f = hi
         while slope < -m sig_s (the bow on a face, the stern lifted), else
         lo
  hface  (heading, quartering seas) constant F; while the slope at the CG
         along the heading is < -m sig_s (the CG on a face, the stern
         lifted), the autopilot's reference is turned delta toward the
         waves' travel direction (stern square to the sea), else the track.
Every law can take a lead (s): the signals evaluated lead s ahead.

DESIGN (two stages, the confirmation on FRESH seeds, so the selection in
stage 1 cannot inflate the stage-2 result):
  stage 1 'screen': 4 cells (tp5/tp8 x chi 180/135) x every setting x
          seeds 0..3 (oracle_screen.pkl)
  'select': per cell, pre-specified rule: Z = sum over the cell's PRIMARY
          metrics of (matched-speed mean delta / null sd of a 4-mission
          mean); settings with all primary metrics in the constant curve's
          speed range on >= 3 of 4 missions; the 2 lowest Z, plus the best
          setting of each primary metric if not already in, at most 4; lead
          variants (base lead + -0.7, +0.7, +1.4 s) of the lowest-Z setting.
  stage 2 'confirm': the constant curve + the selected settings on seeds
          4..11 (oracle.pkl)
  'report': oracle_report.txt (both stages)
PRIMARY metrics (fixed before the runs): tp5 chi 180: bow burials /h,
front-face surf share, heading p99; tp5 chi 135: bow burials /h, heading
p99, front-face surf share; tp8 (both): heading p99, front-face surf share.

  python -m studies.wave_regime_oracle --phase screen,select,confirm,report
Output: studies/_cache/wave_regimes/oracle_screen.pkl, oracle.pkl,
oracle_select.json, oracle_report.txt, oracle.log
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

OUT = os.path.join(HERE, "_cache", "wave_regimes")
KN = 0.514444
G = 9.81
PI = math.pi
T_SKIP = 20.0
D2R = PI / 180.0

# calm steady speed (kn) of constant throttle f -- the scan's table
# (studies/wave_regime_scan.py U_CALM), copied so this file does not depend
# on another script staying unchanged
U_CALM = {0.03: 5.0, 0.05: 6.9, 0.1: 9.2, 0.15: 9.5, 0.2: 10.7,
          0.25: 18.5, 0.3: 22.2, 0.45: 27.1, 0.6: 31.6, 0.8: 38.1,
          1.0: 43.9}
SEAS = {"tp5": dict(hs=1.0, tp=5.0), "tp8": dict(hs=1.0, tp=8.0)}


def u_calm(f):
    ks = sorted(U_CALM)
    return float(np.interp(f, ks, [U_CALM[k] for k in ks]))


def log(msg, name="oracle.log"):
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


def runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


# ------------------------------------------------------------ cells
# per cell: the constant curve, the pl mean levels, the (hi, lo) of the
# two-level laws, the hold law's centre throttle and the hface throttle
CELLS = {
    "tp5_180": dict(sea="tp5", chi=180,
                    const=(0.1, 0.2, 0.25, 0.3, 0.45, 0.6),
                    tm=(0.2, 0.45), hi=0.6, lo=0.2, fc=0.35, kp=0.12,
                    hold_clip=(0.05, 0.75),
                    primary=("bow_burial_per_h", "surf_face_share",
                             "heading_p99_deg")),
    "tp5_135": dict(sea="tp5", chi=135,
                    const=(0.1, 0.2, 0.25, 0.3, 0.45, 0.6),
                    tm=(0.25, 0.45), hi=0.6, lo=0.2, fc=0.35, kp=0.12,
                    hold_clip=(0.05, 0.75), hface_f=0.45,
                    primary=("bow_burial_per_h", "heading_p99_deg",
                             "surf_face_share")),
    "tp8_180": dict(sea="tp8", chi=180,
                    const=(0.2, 0.25, 0.3, 0.45, 0.6, 0.8),
                    tm=(0.3, 0.6), hi=0.8, lo=0.3, fc=0.5, kp=0.15,
                    hold_clip=(0.1, 0.95),
                    primary=("heading_p99_deg", "surf_face_share")),
    "tp8_135": dict(sea="tp8", chi=135,
                    const=(0.2, 0.25, 0.3, 0.45, 0.6, 0.8),
                    tm=(0.35, 0.6), hi=0.8, lo=0.3, fc=0.5, kp=0.15,
                    hold_clip=(0.1, 0.95), hface_f=0.6,
                    primary=("heading_p99_deg", "surf_face_share")),
}
# LOW SPEED (run on request, --cells tp5_180_low): following seas below the
# hump, where the plant is NOT credible (studies/_cache/wave_regimes/
# validity.txt, lowspeed.txt); the hazard the scan showed there is heading
# excursions with the nozzle saturated for lack of jet flow
CELLS_LOW = {
    "tp5_180_low": dict(sea="tp5", chi=180,
                        const=(0.03, 0.05, 0.1, 0.15, 0.2), tm=(0.08,),
                        hi=0.2, lo=0.05, low=True,
                        primary=("heading_p99_deg", "surf_face_share")),
}
ALL_CELLS = dict(CELLS, **CELLS_LOW)


def cell_settings(cell):
    c = ALL_CELLS[cell]
    out = []
    if c.get("low"):
        for f in c["const"]:
            out.append(dict(kind="const", f=float(f), name=f"const_{f:.2f}"))
        phis = [float(wrap(-PI + (i + 1) * 2 * PI / 8)) for i in range(8)]
        for tm in c["tm"]:
            for ph in phis:
                out.append(dict(
                    kind="pl", tm=float(tm), d=0.5, phi0=ph,
                    name=f"pl_{tm:.2f}_0.5_{int(round(ph / D2R)):+d}"))
        # 'power on when the wave overtakes' (steerage: the nozzle's force
        # scales with thrust) and 'ease off down the face'
        out.append(dict(kind="boost", hi=c["hi"], lo=c["lo"], m=1.0,
                        lead=0.7, name=f"boost_{c['hi']:.2f}_{c['lo']:.2f}"))
        out.append(dict(kind="face", hi=0.12, lo=0.03, m=1.0, lead=0.7,
                        name="face_0.12_0.03_1.0"))
        return out
    for f in c["const"]:
        out.append(dict(kind="const", f=float(f), name=f"const_{f:.2f}"))
    phis = [float(wrap(-PI + (i + 1) * 2 * PI / 8)) for i in range(8)]
    for tm in c["tm"]:
        for ph in phis:
            out.append(dict(kind="pl", tm=float(tm), d=0.5, phi0=ph,
                            name=f"pl_{tm:.2f}_0.5_{int(round(ph / D2R)):+d}"))
    for st in (-PI / 2, 0.0, PI / 2, PI):
        out.append(dict(kind="bb", hi=c["hi"], lo=c["lo"], start=st,
                        width=PI / 2,
                        name=f"bb_{c['hi']:.1f}_{c['lo']:.1f}_"
                             f"{int(round(st / D2R)):+d}_90"))
    for pt in (-PI / 2, -3 * PI / 4):
        out.append(dict(kind="hold", fc=c["fc"], kp=c["kp"], kd=0.08,
                        ph_t=pt, lo=c["hold_clip"][0], hi=c["hold_clip"][1],
                        name=f"hold_{c['fc']:.2f}_{int(round(pt / D2R)):+d}"))
    out.append(dict(kind="steep", hi=c["hi"], lo=c["lo"], m=1.5, lead=0.7,
                    name=f"steep_{c['hi']:.1f}_{c['lo']:.1f}_1.5"))
    out.append(dict(kind="face", hi=c["hi"], lo=c["lo"], m=1.0, lead=0.7,
                    name=f"face_{c['hi']:.1f}_{c['lo']:.1f}_1.0"))
    if "hface_f" in c:
        out.append(dict(kind="hface", f=c["hface_f"], m=1.0,
                        delta=20.0 * D2R,
                        name=f"hface_{c['hface_f']:.2f}_20"))
    return out


def f_nominal(st):
    k = st["kind"]
    if k in ("const", "hface"):
        return st["f"]
    if k == "pl":
        return st["tm"]
    if k == "bb":
        return st["hi"] - (st["hi"] - st["lo"]) * st["width"] / (2 * PI)
    if k == "hold":
        return st["fc"]
    if k == "steep":
        return 0.7 * st["hi"] + 0.3 * st["lo"]
    if k == "face":
        return 0.85 * st["hi"] + 0.15 * st["lo"]
    if k == "boost":
        return 0.15 * st["hi"] + 0.85 * st["lo"]
    raise ValueError(k)


def with_lead(st, lead):
    s2 = dict(st)
    s2["lead"] = float(lead)
    s2["name"] = st["name"].split("@")[0] + f"@{lead:+.1f}"
    return s2


# ------------------------------------------------------------ wave signals
def wave_at(sea, s, t, x_st, lead=0.0):
    """(ph, rate, slope, senv) at hull station x_st of state s at time t,
    the point carried `lead` s ahead at the boat's velocity (module
    docstring)."""
    psi = float(s[5])
    c, sn = math.cos(psi), math.sin(psi)
    vx = s[6] * c - s[7] * sn
    vy = s[6] * sn + s[7] * c
    X = s[0] + x_st * c + vx * lead
    Y = s[1] + x_st * sn + vy * lead
    cth, sth = np.cos(sea.th), np.sin(sea.th)
    p = sea.k * (X * cth + Y * sth) - sea.w * (t + lead) + sea.phi
    e = sea.a * np.exp(1j * p)
    z = e.sum()
    pdot = sea.k * (vx * cth + vy * sth) - sea.w
    zd = (1j * pdot * e).sum()
    zs = (1j * sea.k * np.cos(sea.th - psi) * e).sum()
    az = abs(z)
    rate = float((zd / z).imag) if az > 1e-9 else 0.0
    return float(np.angle(z)), rate, float(zs.real), float(abs(zs))


def traj_phase(sea, s, t, x_st):
    """Vectorised ph and slope at station x_st along a trajectory."""
    psi = s[:, 5]
    c, sn = np.cos(psi), np.sin(psi)
    X = s[:, 0] + x_st * c
    Y = s[:, 1] + x_st * sn
    cth, sth = np.cos(sea.th), np.sin(sea.th)
    out_ph = np.empty(len(t))
    out_sl = np.empty(len(t))
    for i0 in range(0, len(t), 2000):
        sl = slice(i0, i0 + 2000)
        p = (sea.k[None, :] * (X[sl, None] * cth + Y[sl, None] * sth)
             - sea.w[None, :] * t[sl, None] + sea.phi[None, :])
        e = sea.a[None, :] * np.exp(1j * p)
        out_ph[sl] = np.angle(e.sum(1))
        kc = sea.k[None, :] * np.cos(sea.th[None, :] - psi[sl, None])
        out_sl[sl] = (1j * kc * e).sum(1).real
    return out_ph, out_sl


def sig_slope(sea, track):
    kc = sea.k * np.cos(sea.th - track)
    return float(math.sqrt(np.sum((sea.a * kc) ** 2) / 2.0))


def command(st, W, sig, v_rel=0.0):
    """Thrust fraction of law st from the wave signals W; v_rel = the
    boat's speed along the waves' travel direction minus the peak celerity
    (m/s), the hold law's damping term."""
    ph, rate, slope, senv = W
    k = st["kind"]
    if k in ("const", "hface"):
        f = st["f"]
    elif k == "pl":
        f = st["tm"] * (1.0 + st["d"] * math.cos(ph - st["phi0"]))
    elif k == "bb":
        f = st["lo"] if ((ph - st["start"]) % (2 * PI)) < st["width"] \
            else st["hi"]
    elif k == "hold":
        f = (st["fc"] - st["kp"] * float(wrap(ph - st["ph_t"]))
             - st["kd"] * v_rel)
        f = min(max(f, st["lo"]), st["hi"])
    elif k == "steep":
        f = st["lo"] if senv > st["m"] * sig else st["hi"]
    elif k == "face":
        f = st["lo"] if slope < -st["m"] * sig else st["hi"]
    elif k == "boost":
        f = st["hi"] if slope < -st["m"] * sig else st["lo"]
    else:
        raise ValueError(k)
    return float(min(max(f, 0.0), 1.0))


# ------------------------------------------------------------ one run
def run_one(cell, st, seed, T):
    from studies import final_eval as FE
    from studies import safety_events as se
    from studies.wave_phase_oracle import SafeMode
    t0 = time.perf_counter()
    c = ALL_CELLS[cell]
    sd = dict(SEAS[c["sea"]])
    sd["theta0"] = float(wrap(PI - math.radians(c["chi"])))
    m = FE.make_mission(seed, 0, T, sea=sd)
    pl = m.plant
    m.s = pl.initial_state(u_calm(f_nominal(st)) * KN)
    m.s[5] = m.phi
    x_b = float(pl.sec.x_bow)
    dtc = m.dt * m.sub
    safe = SafeMode(dtc)
    sea = m.sea
    tp = SEAS[c["sea"]]["tp"]
    wp = 2 * PI / tp
    k_p = wp ** 2 / G
    c_p = G / wp
    sig = sig_slope(sea, m.phi)
    lead = float(st.get("lead", 0.0))
    d_sign = float(np.sign(wrap(sea.theta0 - m.phi)))
    S = {k: [] for k in ("ph", "rate", "slope", "senv", "f_cmd", "thr",
                         "u", "ua", "e", "ref", "safe",
                         "prime")}
    n_steps = 0
    while not m.done():
        W0 = wave_at(sea, m.s, m.t, x_b, 0.0)
        W = W0 if lead == 0.0 else wave_at(sea, m.s, m.t, x_b, lead)
        psi = float(m.s[5])
        u_w = (m.s[6] * math.cos(psi - sea.theta0)
               - m.s[7] * math.sin(psi - sea.theta0))
        f = command(st, W, sig, float(u_w - c_p))
        ref = 0.0
        if st["kind"] == "hface":
            Wc = wave_at(sea, m.s, m.t, 0.0, lead)
            if Wc[2] < -st["m"] * sig:
                ref = d_sign * st["delta"]
            m.ep.heading_ref = m.phi + ref
        thrust = f * m.t_max
        rud = m.ep._steer(m.s, thrust)
        thrust, rud, on = safe(m.s[3], thrust, rud)
        S["ph"].append(W0[0])
        S["rate"].append(W0[1])
        S["slope"].append(W0[2])
        S["senv"].append(W0[3])
        S["f_cmd"].append(thrust / m.t_max)
        S["thr"].append(float(m.s[12]) / m.t_max)
        S["ref"].append(ref)
        S["safe"].append(on)
        S["prime"].append(float(getattr(pl.prop, "prime", 1.0)))
        m.advance(thrust, rud)
        n_steps += 1
        ua, _ = m.along_cross(m.s)
        S["u"].append(float(m.s[6]))
        S["ua"].append(float(ua))
        S["e"].append(float(wrap(m.s[5] - m.phi)))
        if not m.finite or getattr(pl, "capsized", False):
            break
    S = {k: np.asarray(v, bool if k == "safe" else np.float32)
         for k, v in S.items()}
    traj = pl.trajectory()
    dt = float(traj["meta"]["dt"])
    n0 = int(round(T_SKIP / dt))
    k0 = int(round(T_SKIP / dtc))
    cut = {kk: (v[n0:] if isinstance(v, np.ndarray) and v.shape[:1]
                == traj["t"].shape else v) for kk, v in traj.items()}
    if len(cut["t"]) < 10:
        cut, n0, k0 = traj, 0, 0
    raw = se.safety_events(cut, k=2.0, k_report=(1.5, 2.5),
                           track=float(m.phi))
    ev = FE.flatten_events(raw, 2.0)
    tt, ss = cut["t"], cut["s"]
    # ---- surf-riding (as the scan: CG raw phase, 2 s centred encounter
    # rate, speed along the waves in [0.7, 1.15] c_p) + split by the CG's
    # side of the wave (front face = surface falling ahead along the heading)
    cth, sth = np.cos(sea.th), np.sin(sea.th)
    pc = (sea.k[None, :] * (ss[:, 0, None] * cth + ss[:, 1, None] * sth)
          - sea.w[None, :] * tt[:, None] + sea.phi[None, :])
    ph_c = np.unwrap(np.angle(np.sum(sea.a * np.exp(1j * pc), 1)))
    del pc
    h2 = int(round(1.0 / dt))
    we_inst = np.full(len(tt), np.nan)
    if len(tt) > 2 * h2 + 1:
        we_inst[h2:-h2] = -(ph_c[2 * h2:] - ph_c[:-2 * h2]) / (2 * h2 * dt)
    mu = ss[:, 5] - sea.theta0
    u_w = ss[:, 6] * np.cos(mu) - ss[:, 7] * np.sin(mu)
    band = (u_w >= 0.7 * c_p) & (u_w <= 1.15 * c_p)
    lock = np.abs(we_inst) < 0.2 * wp
    surf = lock & band
    _, sl_cg = traj_phase(sea, ss, tt, 0.0)
    surf_face = surf & (sl_cg < 0.0)
    nw = int(round(tp / dt))
    capt = np.zeros(len(tt), bool)
    if len(tt) > nw:
        from numpy.lib.stride_tricks import sliding_window_view
        Wn = sliding_window_view(ph_c, nw + 1)
        capt[:len(Wn)] = (Wn.max(1) - Wn.min(1)) < PI / 2
    capt_long = max([(b - a) * dt for a, b in runs(capt)], default=0.0)
    surf_long = max([(b - a) * dt for a, b in runs(surf)], default=0.0)
    # ---- where on the wave the hazards start (bow ph)
    ph_b, _ = traj_phase(sea, ss, tt, x_b)
    hb = se.bow_height(cut)
    wet = runs(hb < 0.0)
    f0 = raw["bow"]["f0"]
    bury = [(a, b) for a, b in wet
            if hb[a:b].min() < -se.BURY_FRAC * f0
            or (b - a) * dt >= se.WET_LONG - 1e-9]
    idx, pk = se.acc_peaks(cut["a_cg"], dt, 2.0)
    e_tr = wrap(ss[:, 5] - m.phi)
    hw = runs(np.abs(e_tr) > math.radians(10.0))
    ua = S["ua"][k0:].astype(float)
    sk = {k: v[k0:] for k, v in S.items()}
    r = dict(
        cell=cell, setting=st["name"], st=st, seed=int(seed),
        kn=float(ua.mean() / KN) if ua.size else float("nan"),
        u_sd_kn=float(sk["u"].std() / KN) if ua.size else float("nan"),
        uw_over_c=float(np.mean(u_w) / c_p),
        ev=ev,
        bow_burial_per_h=float(ev["bow_burial_per_h"]),
        bow_wet_per_h=float(ev["bow_wet_per_h"]),
        n_ge7_per_h=float(ev["n_ge7_per_h"]), a110=float(ev["a110"]),
        heading_max_deg=float(raw["yaw"]["heading_dev_max_deg"]),
        heading_p99_deg=float(np.degrees(np.percentile(np.abs(e_tr), 99))),
        heading_rms_deg=float(np.degrees(np.sqrt(np.mean(e_tr ** 2)))),
        r_p99_degs=float(np.degrees(np.percentile(np.abs(ss[:, 11]), 99))),
        y_max=float(raw["track"]["max_abs_y"]),
        y_fail=int(bool(raw["track"]["fail_5L"])),
        broach=int(raw["yaw"]["broach"]),
        roll_max_deg=float(raw["roll"]["max_deg"]),
        capsized=bool(getattr(pl, "capsized", False)),
        safe_entries=int(safe.n_entries),
        surf_share=float(np.mean(surf)),
        surf_face_share=float(np.mean(surf_face)),
        surf_long_s=float(surf_long), capt_long_s=float(capt_long),
        band_share=float(np.mean(band)),
        prime_deficit=float(np.mean(1.0 - sk["prime"])),
        f_mean=float(sk["f_cmd"].mean()), thr_mean=float(sk["thr"].mean()),
        thr_sd=float(sk["thr"].std()), cmd_sd=float(sk["f_cmd"].std()),
        rate_abs_med=float(np.median(np.abs(sk["rate"]))),
        ph_wet=ph_b[[a for a, _ in wet]].astype(np.float32),
        ph_bury=ph_b[[a for a, _ in bury]].astype(np.float32),
        ph_pk=ph_b[idx[pk >= 2.0]].astype(np.float32),
        g_pk=pk[pk >= 2.0].astype(np.float32),
        ph_hdg=ph_b[[a for a, _ in hw]].astype(np.float32),
        numeric=int(bool(raw["numerical"]["any"])),
        finite=bool(m.finite), n_steps=n_steps,
        series={k: v for k, v in S.items()},
        sig_slope=sig, k_p=k_p, c_p=c_p,
        t_run=time.perf_counter() - t0)
    return r


def _job(args):
    cell, st, seed, T = args
    try:
        return (cell, st["name"], seed), run_one(cell, st, seed, T)
    except Exception:
        import traceback
        return (cell, st["name"], seed), dict(error=traceback.format_exc())


def _init():
    import warnings
    warnings.filterwarnings("ignore")


def atomic_pickle(obj, path):
    with open(path + ".tmp", "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(path + ".tmp", path)


def run_jobs(jobs, path, workers, min_gb, meta):
    R = pickle.load(open(path, "rb")) if os.path.exists(path) else \
        dict(results={}, errors={}, meta=meta)
    R.setdefault("meta", meta)
    todo = [j for j in jobs if (j[0], j[1]["name"], j[2]) not in R["results"]]
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
            if "error" in r:
                log(f"  ERROR {key}: {r['error'].splitlines()[-1]}")
                R["errors"][key] = r["error"]
                continue
            R["results"][key] = r
            el = time.time() - t0
            if n_done % 5 == 0 or n_done == len(todo):
                log(f"  {n_done}/{len(todo)} {key}: {r['kn']:.1f} kn bur/h "
                    f"{r['bow_burial_per_h']:.0f} wet/h "
                    f"{r['bow_wet_per_h']:.0f} hdg99 "
                    f"{r['heading_p99_deg']:.1f} surfF "
                    f"{r['surf_face_share']:.2f} f {r['f_mean']:.2f} "
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


# ------------------------------------------------------------ analysis
METRICS = (("bow_burial_per_h", "bur/h"), ("bow_wet_per_h", "wet/h"),
           ("heading_p99_deg", "hdg99"), ("heading_max_deg", "hdgX"),
           ("surf_face_share", "surfF"), ("surf_share", "surf"),
           ("a110", "A110"), ("n_ge7_per_h", "g7/h"), ("y_max", "ymax"))


def _sorted(v, m):
    o = np.argsort(v)
    return v[o], m[o]


def matched(vc, Mc, v, Ms):
    """Per mission j: Ms[j] minus the constant curve (vc[:, j], Mc[:, j])
    interpolated at v[j]; in-range flags (studies/wave_phase_oracle)."""
    d = np.array([Ms[j] - np.interp(v[j], *_sorted(vc[:, j], Mc[:, j]))
                  for j in range(len(v))])
    inr = np.array([vc[:, j].min() <= v[j] <= vc[:, j].max()
                    for j in range(len(v))])
    return d, inr


def first_harmonic(x, th):
    """(c0, amplitude, phase of the maximum) of x ~ c0 + a cos th + b sin
    th."""
    th = np.asarray(th, float)
    x = np.asarray(x, float)
    A = np.stack([np.ones_like(th), np.cos(th), np.sin(th)], 1)
    c0, a, b = np.linalg.lstsq(A, x, rcond=None)[0]
    return float(c0), float(math.hypot(a, b)), float(math.atan2(b, a))


def analyse_cell(res, cell, seeds):
    """Matched-speed comparison of every setting of one cell against the
    constant curve on the same missions. Returns a dict and text lines."""
    c = ALL_CELLS[cell]
    consts = [f"const_{f:.2f}" for f in c["const"]]
    ok = [s for s in seeds if all((cell, n, s) in res for n in consts)]
    # extra constant levels (--extra-const) present on every mission join
    # the curve (denser where the candidates run)
    extra = sorted({k[1] for k in res if k[0] == cell
                    and k[1].startswith("const_") and k[1] not in consts
                    and all((cell, k[1], s) in res for s in ok)},
                   key=lambda n: float(n.split("_")[1]))
    consts = sorted(consts + extra, key=lambda n: float(n.split("_")[1]))
    names = []
    for k in res:
        if k[0] == cell and k[1] not in names and not k[1].startswith(
                "const_"):
            names.append(k[1])
    L = []
    P = L.append
    out = dict(cell=cell, ok=ok, const={}, cmp={}, null={})
    if len(ok) < 2:
        P(f"[{cell}] fewer than 2 complete missions")
        return out, L
    n = len(ok)
    vc = np.array([[res[(cell, cn, s)]["kn"] for s in ok] for cn in consts])
    Mc = {nm: np.array([[res[(cell, cn, s)][nm] for s in ok]
                        for cn in consts]) for nm, _ in METRICS}
    out["vc"], out["Mc"] = vc, Mc
    # ---- constant curve
    P(f"constant throttle (means over {n} missions; sums for brch/yF/safe/"
      f"cap; rates per hour):")
    P("   f      kn   " + " ".join(f"{lab:>6s}" for _, lab in METRICS)
      + "   usd  cL_s  brch yF safe cap |rate| thr")
    for i, cn in enumerate(consts):
        rr = [res[(cell, cn, s)] for s in ok]
        P(f"  {cn[6:]}  {vc[i].mean():5.1f}  "
          + " ".join(f"{Mc[nm][i].mean():6.2f}" for nm, _ in METRICS)
          + f"  {np.mean([r['u_sd_kn'] for r in rr]):4.1f}"
          f"  {max(r['capt_long_s'] for r in rr):5.1f}"
          f"  {sum(r['broach'] for r in rr):3d}"
          f" {sum(r['y_fail'] for r in rr):2d}"
          f" {sum(r['safe_entries'] for r in rr):4d}"
          f" {sum(int(r['capsized']) for r in rr):3d}"
          f"  {np.mean([r['rate_abs_med'] for r in rr]):5.2f}"
          f" {np.mean([r['thr_mean'] for r in rr]):.2f}")
    # ---- where on the wave the hazards start (constant runs)
    P("where the hazards start on the wave (constant runs pooled; bow place"
      " ph in 45-deg bins centred -180,-135,...,+135: 0 crest at the bow, "
      ">0 bow on the front/downhill face, <0 on the back of the wave "
      "ahead):")
    bins = np.linspace(-PI - PI / 8, PI - PI / 8, 9)
    for key, lab in (("ph_bury", "burial"), ("ph_wet", "bow-wet"),
                     ("ph_pk", ">=2g"), ("ph_hdg", "hdg>10")):
        x = np.concatenate([res[(cell, cn, s)][key] for cn in consts
                            for s in ok]).astype(float)
        xw = wrap(x + PI / 8) - PI / 8
        P(f"  {lab:8s} n {len(x):4d}: {np.histogram(xw, bins)[0].tolist()}")
        out.setdefault("phase_hist", {})[lab] = np.histogram(xw, bins)[0]
    # ---- null floor (leave one interior constant level out)
    P("NULL: each interior constant level vs the curve of the others "
      "(per-mission matched delta sd, and sd of a mission mean):")
    nul = {nm: [] for nm, _ in METRICS}
    for i in range(1, len(consts) - 1):
        keep = [j for j in range(len(consts)) if j != i]
        for nm, _ in METRICS:
            d, inr = matched(vc[keep], Mc[nm][keep], vc[i], Mc[nm][i])
            nul[nm].append(d[inr])
    sdm = {}
    for nm, lab in METRICS:
        allv = np.concatenate(nul[nm]) if nul[nm] else np.zeros(1)
        # floor: the resolution of the metric (one event in one 100 s
        # mission = 36 /h, over n missions; 0.01 of time; 0.5 deg; 0.05 g;
        # 0.5 m), so a metric that is ~0 everywhere cannot give huge z
        floor = {"bow_burial_per_h": 36.0 / n, "bow_wet_per_h": 36.0 / n,
                 "n_ge7_per_h": 36.0 / n, "surf_face_share": 0.01,
                 "surf_share": 0.01, "heading_p99_deg": 0.5,
                 "heading_max_deg": 0.5, "a110": 0.05,
                 "y_max": 0.5}[nm]
        sdm[nm] = max(float(allv.std()) / math.sqrt(n), floor)
        out["null"][nm] = allv
    P("  " + "  ".join(f"{lab} sd {math.sqrt(n) * sdm[nm]:.2f}/"
                       f"{sdm[nm]:.2f}" for nm, lab in METRICS))
    out["sdm"] = sdm
    # ---- every setting
    vbar = vc.mean(1)
    P("settings vs the constant curve AT MATCHED SPEED (per mission, "
      "paired). Per metric: mean delta (<0 fewer/smaller than constant "
      "throttle at the same speed) [z = delta / null sd of a mean]; '*' = "
      "missions outside the curve's speed range left out. best>=v: the "
      "setting's mean minus the BEST constant level whose mean speed is >= "
      "the setting's (nan: none as fast). thr: realised thrust modulation "
      "(first harmonic vs the bow place, fraction of t_max @ ph of max), "
      "cmd: commanded.")
    P("  setting                kn    " + " ".join(f"{lab:>14s}"
                                                  for _, lab in METRICS[:6])
      + "   brch yF safe cap   thr        cmd")
    for nm_s in names:
        if not all((cell, nm_s, s) in res for s in ok):
            continue
        rr = [res[(cell, nm_s, s)] for s in ok]
        v = np.array([r["kn"] for r in rr])
        D, Z, INR, BEST = {}, {}, {}, {}
        cols = []
        for nm, lab in METRICS:
            ms = np.array([r[nm] for r in rr])
            d, inr = matched(vc, Mc[nm], v, ms)
            D[nm], INR[nm] = d, inr
            mu = float(d[inr].mean()) if inr.any() else float("nan")
            Z[nm] = mu / sdm[nm] if inr.any() else float("nan")
            fast = [i for i in range(len(consts)) if vbar[i] >= v.mean()]
            BEST[nm] = (float(ms.mean() - min(Mc[nm][i].mean()
                                              for i in fast))
                        if fast else float("nan"))
            cols.append(f"{mu:+6.2f}[{Z[nm]:+4.1f}]"
                        f"{'' if inr.all() else '*'}")
        th = np.concatenate([r["series"]["ph"][83:] for r in rr]).astype(float)
        thr = np.concatenate([r["series"]["thr"][83:] for r in rr])
        cmd = np.concatenate([r["series"]["f_cmd"][83:] for r in rr])
        h_t = first_harmonic(thr, th)
        h_c = first_harmonic(cmd, th)
        P(f"  {nm_s:22s} {v.mean():5.1f}  " + " ".join(f"{x:>14s}"
                                                      for x in cols[:6])
          + f"   {sum(r['broach'] for r in rr):3d}"
          f" {sum(r['y_fail'] for r in rr):2d}"
          f" {sum(r['safe_entries'] for r in rr):4d}"
          f" {sum(int(r['capsized']) for r in rr):3d}"
          f"   {h_t[1]:.2f}@{math.degrees(h_t[2]):+4.0f}"
          f"  {h_c[1]:.2f}@{math.degrees(h_c[2]):+4.0f}")
        P(f"  {'':22s}  best>=v: " + "  ".join(
            f"{lab} {BEST[nm]:+.2f}" for nm, lab in METRICS[:6])
          + f"  | A110 {D['a110'][INR['a110']].mean() if INR['a110'].any() else float('nan'):+.2f}"
          f" g7/h {D['n_ge7_per_h'][INR['n_ge7_per_h']].mean() if INR['n_ge7_per_h'].any() else float('nan'):+.1f}"
          f" ymax {D['y_max'][INR['y_max']].mean() if INR['y_max'].any() else float('nan'):+.1f}")
        out["cmp"][nm_s] = dict(kn=v, delta=D, z=Z, in_range=INR,
                                best_ge_v=BEST, thr_h1=h_t, cmd_h1=h_c,
                                st=rr[0]["st"])
    return out, L


def select(A, cell, n_top=2, n_max=4):
    """The pre-specified stage-2 selection (module docstring)."""
    prim = ALL_CELLS[cell]["primary"]
    cand = []
    for nm_s, cmp in A["cmp"].items():
        n_ok = len(A["ok"])
        if any(int(cmp["in_range"][p].sum()) < min(3, n_ok) for p in prim):
            continue
        Zs = sum(cmp["z"][p] for p in prim)
        cand.append((Zs, nm_s))
    cand.sort()
    picked = [nm for _, nm in cand[:n_top]]
    for p in prim:
        bp = sorted([(A["cmp"][nm]["z"][p], nm) for _, nm in cand])
        if bp and bp[0][1] not in picked and len(picked) < n_max:
            picked.append(bp[0][1])
    return picked, cand


def harmonic_lines(A, cell):
    """pl: the dependence of the matched delta on phi0 (first harmonic over
    the 8 phi0; permutation p over within-mission relabelling)."""
    L = []
    rng = np.random.default_rng(1)
    for tm in ALL_CELLS[cell]["tm"]:
        ks = [k for k in A["cmp"] if k.startswith(f"pl_{tm:.2f}_0.5_")
              and "@" not in k]
        if len(ks) < 8:
            continue
        ph = np.radians([float(k.split("_")[3]) for k in ks])
        for nm, lab in METRICS[:6]:
            Dm = np.array([np.where(A["cmp"][k]["in_range"][nm],
                                    A["cmp"][k]["delta"][nm], np.nan)
                           for k in ks])
            if np.isnan(Dm).all():
                continue
            Dm = np.where(np.isnan(Dm), np.nanmean(Dm), Dm)
            e = np.exp(1j * ph)
            c1 = 2.0 * np.mean(Dm.mean(1) * e)
            a0 = abs(c1)
            cnt = 0
            for _ in range(2000):
                Xp = np.stack([rng.permutation(Dm[:, j])
                               for j in range(Dm.shape[1])], 1)
                cnt += abs(2.0 * np.mean(Xp.mean(1) * e)) >= a0 - 1e-15
            srt = sorted(zip(ph, Dm.mean(1)))
            L.append(f"  pl Tm {tm:.2f} {lab:6s}: mean over phi0 "
                     f"{Dm.mean():+6.2f}, |c1| {a0:5.2f}, best phi0 "
                     f"{math.degrees(np.angle(-c1)):+5.0f}, p "
                     f"{(cnt + 1) / 2001:.3f}; "
                     + " ".join(f"{math.degrees(a):+.0f}:{x:+.2f}"
                                for a, x in srt))
    return L


def report(seeds_screen, seeds_conf):
    lines = []
    P = lines.append
    P("ORACLE WAVE-TIMED CONTROL in following / quartering seas "
      "(studies/wave_regime_oracle.py; docstring = definitions). PlaningVes"
      "selGZ, the mission heading autopilot, SafeMode, statistics after "
      "20 s of each 120 s run.")
    P("Metrics: bur/h, wet/h = bow-burial / bow-wet entries per hour; hdg99 "
      "/ hdgX = 99th pct / max |heading - track| (deg); surfF = share of "
      "time surf-riding (|encounter rate| < 0.2 w_p over 2 s and speed "
      "along the waves in 0.7..1.15 c_p) WITH the CG on the front (downhill)"
      " face; surf = the same on either side; A110 = A1/10 g (k = 2); g7/h "
      "= peaks >= 7 g per hour; ymax = max |cross-track| m; brch = broach "
      "events (sum), yF = runs > 5 L off the track (sum), safe = SafeMode "
      "entries (sum), cap = capsizes (sum).")
    out = {}
    for stage, fn, seeds in (("SCREEN (seeds 0-3)", "oracle_screen.pkl",
                              seeds_screen),
                             ("CONFIRM (fresh seeds 4-11)", "oracle.pkl",
                              seeds_conf)):
        path = os.path.join(OUT, fn)
        if not os.path.exists(path):
            continue
        R = pickle.load(open(path, "rb"))
        res = R["results"]
        P("")
        P("=" * 100)
        P(f"{stage}: {len(res)} runs, {len(R.get('errors', {}))} errors")
        for cell in ALL_CELLS:
            A, L = analyse_cell(res, cell, seeds)
            if not A.get("cmp"):
                continue
            P("")
            P(f"--- {cell} (sea {ALL_CELLS[cell]['sea']}, chi "
              f"{ALL_CELLS[cell]['chi']}), {len(A['ok'])} paired missions "
              f"{A['ok']} ---")
            lines.extend(L)
            hl = harmonic_lines(A, cell)
            if hl:
                P("phase-locked: dependence on phi0 (matched delta; best "
                  "phi0 = where the delta is lowest; p = permutation):")
                lines.extend(hl)
            # families: per mission, mean delta over the family
            P("families (mean over the family's settings of the per-mission "
              "matched delta, then mean over missions; t over missions):")
            fams = sorted({k.split("_")[0] + ("@" if "@" in k else "")
                           for k in A["cmp"]})
            for fam in fams:
                ks = [k for k in A["cmp"] if k.split("_")[0] == fam.rstrip("@")
                      and (("@" in k) == fam.endswith("@"))]
                row = []
                for nm, lab in METRICS[:6]:
                    Dm = np.array([np.where(A["cmp"][k]["in_range"][nm],
                                            A["cmp"][k]["delta"][nm], np.nan)
                                   for k in ks])
                    pm = np.nanmean(Dm, 0)
                    nn = int(np.sum(np.isfinite(pm)))
                    mu = float(np.nanmean(pm)) if nn else float("nan")
                    sd = float(np.nanstd(pm, ddof=1)) if nn > 1 else float("nan")
                    t = mu / (sd / math.sqrt(nn)) if nn > 1 and sd > 0 \
                        else float("nan")
                    row.append(f"{lab} {mu:+.2f} (t {t:+.1f})")
                P(f"  {fam:7s} ({len(ks):2d}): " + "  ".join(row))
            # best per family vs chance
            P("best setting per metric vs the best of as many no-effect "
              "settings (E[min of N standard normals] x null sd of a mean):")
            for nm, lab in METRICS[:6]:
                zs = [(A["cmp"][k]["z"][nm], k) for k in A["cmp"]
                      if np.isfinite(A["cmp"][k]["z"][nm])]
                if not zs:
                    continue
                zs.sort()
                N = len(zs)
                emin = float(np.mean(np.random.default_rng(2).standard_normal(
                    (20000, N)).min(1)))
                P(f"  {lab:6s}: best {zs[0][1]} z {zs[0][0]:+.1f} "
                  f"(delta {zs[0][0] * A['sdm'][nm]:+.2f}); E[best of {N}] "
                  f"z {emin:+.1f}; settings with z < -2: "
                  f"{sum(1 for z, _ in zs if z < -2)} of {N}")
            out[(fn, cell)] = A
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "oracle_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    atomic_pickle(out, os.path.join(OUT, "oracle_analysis.pkl"))
    print(txt)
    return out


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="screen,select,confirm,report")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--T", type=float, default=120.0)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--seeds-screen", default="0,1,2,3")
    ap.add_argument("--seeds-confirm", default="4,5,6,7,8,9,10,11")
    ap.add_argument("--min-gb", type=float, default=2.0)
    ap.add_argument("--extra-const", default="",
                    help="confirm phase: extra constant levels on the "
                         "confirm seeds, 'cell:f,f;cell:f'")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cells = [c for c in a.cells.split(",") if c]
    ss = [int(x) for x in a.seeds_screen.split(",") if x]
    sc = [int(x) for x in a.seeds_confirm.split(",") if x]
    ph = a.phase.split(",")
    sel_path = os.path.join(OUT, "oracle_select.json")
    if "screen" in ph:
        jobs = [(cell, st, s, a.T) for s in ss for cell in cells
                for st in cell_settings(cell)]
        run_jobs(jobs, os.path.join(OUT, "oracle_screen.pkl"), a.workers,
                 a.min_gb, dict(T=a.T, seeds=ss, cells=cells))
    if "select" in ph:
        R = pickle.load(open(os.path.join(OUT, "oracle_screen.pkl"), "rb"))
        SEL = {}
        for cell in cells:
            A, _ = analyse_cell(R["results"], cell, ss)
            if not A.get("cmp"):
                continue
            picked, cand = select(A, cell)
            sts = {k[1]: r["st"] for k, r in R["results"].items()
                   if k[0] == cell}
            chosen = [sts[nm] for nm in picked]
            if picked:
                top = sts[picked[0]]
                base = float(top.get("lead", 0.0))
                for dl in (-0.7, 0.7, 1.4):
                    chosen.append(with_lead(top, base + dl))
            SEL[cell] = dict(picked=picked, chosen=chosen,
                             ranking=[(float(z), nm) for z, nm in cand[:10]])
            log(f"select {cell}: {picked} (top Z "
                f"{[round(float(z), 1) for z, _ in cand[:4]]}); lead "
                f"variants of {picked[0] if picked else None}")
        json.dump(SEL, open(sel_path, "w"), indent=1)
    if "confirm" in ph:
        SEL = json.load(open(sel_path))
        jobs = []
        for s in sc:
            for cell in cells:
                if cell not in SEL:
                    continue
                sts = [st for st in cell_settings(cell)
                       if st["kind"] == "const"] + SEL[cell]["chosen"]
                jobs += [(cell, st, s, a.T) for st in sts]
        for item in [x for x in a.extra_const.split(";") if x]:
            cell, fs = item.split(":")
            for s in sc:
                for f in fs.split(","):
                    jobs.append((cell, dict(kind="const", f=float(f),
                                            name=f"const_{float(f):.2f}"),
                                 s, a.T))
        run_jobs(jobs, os.path.join(OUT, "oracle.pkl"), a.workers, a.min_gb,
                 dict(T=a.T, seeds=sc, cells=cells, select=SEL))
    if "report" in ph:
        report(ss, sc)


if __name__ == "__main__":
    main()
