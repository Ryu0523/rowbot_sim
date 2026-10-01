#!/usr/bin/env python3
"""
The safety events of one closed-loop run in the high-fidelity world, by the
definitions of learn/meta/ROADMAP_2026-09-30.md section 9 (sources checked in
learn/meta/SAFETY_STANDARDS_2026-09-30.md) -- one counter for every row of
the final evaluation.

INPUT
A trajectory at the plant step (0.02 s for scarab195, the finest this plant
has): PlaningVesselGZ(record=True).trajectory() from
sim/planing_vessel_gz.py. Row k is the state at t[k] with what the plant
measured from that state.

WHAT THE PLANT GIVES, AND WHAT IS COMPUTED HERE
Recorded by the parent plant at every step: the CG vertical acceleration
(last_cg_acc, d2z/dt2 of the CG in the earth frame: an accelerometer's
vertical reading minus 1 g, gravity already out; body tilt not applied), the
bow acceleration, the bow keel immersion (last_rel_bow, the bow station on
the centreline, roll ignored), the Heun substep count. Recorded by the
subclass: the water over the jet intake (the parent computes it every step
for the pump's prime and throws it away) and whether a step ended finite or
capsized. Computed here from the states, because no plant records them:
  - the bow deck-edge height above the local water, with roll: the low
    deck edge of the bow station from the recorded keel immersion,
    depth cos(phi) + z_k (cos(phi) - 1) - y_c |sin(phi)| - d_keel (the
    water taken at the centreline station, 0.25 m inboard of the edge);
  - yaw acceleration (finite difference of r), heading deviation and
    cross-track error from the track angle;
  - the speed along the wave direction (MGN 328 band).

DEFINITIONS (section 9)
  Acceleration  k x a_cg (k = 2.0, also 1.5 and 2.5: DEFECTS H7), 10 Hz
                4-pole Butterworth low-pass (one causal pass: exactly the
                4-pole filter; the time shift does not change a peak),
                upward peaks >= 0.3 g at least 0.1 s apart (threshold on
                the corrected signal, which stands in for the measured
                one). A1/10 = mean of the highest ceil(n/10) peaks, per
                consecutive 60 s window and over the run; windows with
                A1/10 > 3.0 g; single peaks >= 7 g and >= 10 g per hour.
  Bow           wet: deck-edge height < 0 (count of entries, per hour);
                buried: a wet spell that goes below -f0/4 or lasts >= 0.5 s,
                f0 the static bow freeboard (plant.freeboard).
  Broaching     |nozzle| >= 0.95 max on the side that turns against the yaw
                rate, while the heading deviation and the yaw rate both grow
                in magnitude (e r > 0, r r' > 0), for >= 0.5 s
                (MSC.1/Circ.1627 3.5.2.4, nozzle for rudder). Warning:
                heading deviation > 10 deg, entries per hour.
  Cross-track   share of time |y| <= 2 L; failure once |y| > 5 L.
  Roll          entries above 20 deg and above the failure angle
                min(40 deg, vanishing stability, deck-edge immersion) per
                hour; capsize once |phi| reaches the vanishing angle.
  Intake        entries of the intake above the local surface, per hour.
  Numerical     non-finite state, substep cap hit, u > 1.5 u_max (u_max the
                plant's calm top speed), |absolute pitch| > 45 deg: counted
                apart, never as capsize.
  MGN 328       time with the speed along the waves in [0.7, 1.15] c,
                c = g Tp / 2 pi (reported only).
A count of n events in T hours bounds the rate at chi2_0.95(2n+2)/(2T) per
hour (3/T for none): section 9's "0 capsizes in T hours" statement.

Run: python -m studies.safety_events     (the GZ numbers, the ITTC calm-water
     heel check, one 60 s wave run and the same run on the parent plant;
     about a minute)
"""
import time

import numpy as np

G = 9.81
K_DEFAULT, K_REPORT = 2.0, (1.5, 2.5)
F_LP, POLES = 10.0, 4             # NSWCCD StandardG low-pass
PEAK_MIN, PEAK_SEP = 0.3, 0.1     # g, s
WINDOW = 60.0                     # s, A1/10 window
A110_LIM, PEAK_7, PEAK_10 = 3.0, 7.0, 10.0
BURY_FRAC, WET_LONG = 0.25, 0.5   # x f0, s
NOZZLE_FRAC, BROACH_T = 0.95, 0.5
HEADING_WARN = np.radians(10.0)
TRACK_OK, TRACK_FAIL = 2.0, 5.0   # x L
ROLL_WARN = np.radians(20.0)
U_DIV, TH_DIV = 1.5, np.radians(45.0)
MGN_BAND = (0.7, 1.15)


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _runs(mask):
    """[(start, stop)) index ranges where mask is True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _first_t(t, mask):
    i = np.nonzero(mask)[0]
    return float(t[i[0]]) if i.size else None


def rate_ub95(n, hours):
    """95% upper bound of a Poisson rate from n events in `hours`."""
    from scipy.stats import chi2
    if hours <= 0:
        return float("inf")
    return float(chi2.ppf(0.95, 2 * n + 2) / (2.0 * hours))


def acc_peaks(a_cg, dt, k):
    """(times index, peak values in g) of k x a_cg: 10 Hz 4-pole
    Butterworth, upward peaks >= 0.3 g, >= 0.1 s apart."""
    from scipy.signal import butter, find_peaks, sosfilt, sosfilt_zi
    x = k * np.asarray(a_cg, float) / G
    if x.size == 0:
        return np.zeros(0, int), np.zeros(0)
    fs = 1.0 / dt
    if fs > 2.0 * F_LP:
        sos = butter(POLES, F_LP, fs=fs, output="sos")
        x, _ = sosfilt(sos, x, zi=sosfilt_zi(sos) * x[0])
    idx, _ = find_peaks(x, height=PEAK_MIN,
                        distance=max(1, int(round(PEAK_SEP * fs))))
    return idx, x[idx]


def a_1_10(p):
    """Mean of the highest tenth of the peaks (at least one); 0 if none."""
    if len(p) == 0:
        return 0.0
    n = int(np.ceil(len(p) / 10.0))
    return float(np.sort(p)[::-1][:n].mean())


def bow_height(traj):
    """Height of the bow station's low deck edge above the local water,
    m, at every row (see the module docstring)."""
    mt = traj["meta"]
    phi = traj["s"][:, 3]
    d_keel = traj["rel_bow"] + mt["draft_bow"]
    c, s = np.cos(phi), np.abs(np.sin(phi))
    return (mt["depth"] * c + mt["bow_keel_z"] * (c - 1.0)
            - mt["bow_half_beam"] * s - d_keel)


def safety_events(traj, k=K_DEFAULT, k_report=K_REPORT, track=0.0,
                  window=WINDOW):
    """Everything section 9 counts in a high-fidelity run, as a dict.

    traj   PlaningVesselGZ.trajectory(): t, s, a_cg, rel_bow, n_sub,
           submergence, ok, capsized and meta (plant constants)
    k      acceleration correction for the planning limit; every k in
           (k, *k_report) is reported
    track  heading of the track line through the origin (rad); heading
           deviation and cross-track error are taken from it"""
    mt = traj["meta"]
    t, s = np.asarray(traj["t"], float), np.asarray(traj["s"], float)
    dt = float(mt["dt"])
    n = len(t)
    T = n * dt
    hours = T / 3600.0
    per_h = (lambda c: float(c / hours)) if hours > 0 else \
        (lambda c: float("nan"))
    L = mt["L"]
    out = dict(duration_s=T, hours=hours, dt=dt, n_rows=n)

    # ---- vertical acceleration at the CG
    acc = {}
    t0 = t[0] if n else 0.0
    n_win = int(T / window + 1e-9)
    edges = ([(t0 + i * window, t0 + (i + 1) * window)
              for i in range(n_win)] if n_win else [(t0, t0 + T)])
    for kk in (k,) + tuple(x for x in k_report if x != k):
        idx, pk = acc_peaks(traj["a_cg"], dt, kk)
        tp = t[idx]
        a_win = [a_1_10(pk[(tp >= a) & (tp < b)]) for a, b in edges]
        acc[kk] = dict(
            n_peaks=int(pk.size), peak_max_g=float(pk.max(initial=0.0)),
            A110_run_g=a_1_10(pk), A110_windows_g=a_win,
            window_partial=(n_win == 0),
            share_windows_over_3g=float(np.mean(np.array(a_win)
                                                > A110_LIM)),
            peaks_ge_7g=int((pk >= PEAK_7).sum()),
            peaks_ge_7g_per_h=per_h((pk >= PEAK_7).sum()),
            peaks_ge_10g=int((pk >= PEAK_10).sum()),
            peaks_ge_10g_per_h=per_h((pk >= PEAK_10).sum()))
    out["acc"] = acc
    out["k"] = k

    # ---- bow height above the local water
    f0 = mt["freeboard"]
    hb = bow_height(traj) if n else np.zeros(0)
    wet = _runs(hb < 0.0)
    bury = [(a, b) for a, b in wet
            if hb[a:b].min() < -BURY_FRAC * f0
            or (b - a) * dt >= WET_LONG - 1e-9]
    out["bow"] = dict(f0=f0, min_height=float(hb.min(initial=np.inf)),
                      wet=len(wet), wet_per_h=per_h(len(wet)),
                      buried=len(bury), buried_per_h=per_h(len(bury)))

    # ---- broaching and heading
    e = _wrap(s[:, 5] - track)
    r = s[:, 11]
    rd = np.gradient(r, dt) if n > 1 else np.zeros(n)
    noz, thr = s[:, 13], s[:, 12]
    # +nozzle pushes the stern to +y; the yaw moment is x_nozzle fy, and
    # the bucket reverses the side force in reverse thrust
    n_dir = np.sign(mt["x_nozzle"]) * np.sign(noz) * np.where(thr >= 0.0,
                                                             1.0, -1.0)
    against = (np.abs(noz) >= NOZZLE_FRAC * mt["rud_max"]) \
        & (n_dir * np.sign(r) < 0.0)
    grow = (r != 0.0) & (np.sign(e) == np.sign(r)) \
        & (np.sign(rd) == np.sign(r))
    broach = [(a, b) for a, b in _runs(against & grow)
              if (b - a) * dt >= BROACH_T - 1e-9]
    hw = _runs(np.abs(e) > HEADING_WARN)
    out["yaw"] = dict(broach=len(broach), broach_per_h=per_h(len(broach)),
                      heading_dev_over_10=len(hw),
                      heading_dev_over_10_per_h=per_h(len(hw)),
                      heading_dev_max_deg=float(np.degrees(
                          np.abs(e).max(initial=0.0))))

    # ---- cross-track
    y = -np.sin(track) * s[:, 0] + np.cos(track) * s[:, 1]
    far = np.abs(y) > TRACK_FAIL * L
    out["track"] = dict(share_within_2L=float(np.mean(np.abs(y)
                                                      <= TRACK_OK * L))
                        if n else float("nan"),
                        max_abs_y=float(np.abs(y).max(initial=0.0)),
                        fail_5L=bool(far.any()), t_fail=_first_t(t, far))

    # ---- roll and capsize
    phi = s[:, 3]
    phi_end = phi if mt.get("capsize_state") is None else \
        np.append(phi, mt["capsize_state"][3])
    over20 = _runs(np.abs(phi_end) > ROLL_WARN)
    overf = _runs(np.abs(phi_end) > mt["phi_fail"])
    cap = bool(mt.get("capsized")) or bool(
        (np.abs(phi_end) >= mt["phi_vanish"]).any())
    out["roll"] = dict(max_deg=float(np.degrees(np.abs(phi_end).max(
        initial=0.0))), phi_fail_deg=float(np.degrees(mt["phi_fail"])),
        phi_vanish_deg=float(np.degrees(mt["phi_vanish"])),
        over_20=len(over20), over_20_per_h=per_h(len(over20)),
        over_fail=len(overf), over_fail_per_h=per_h(len(overf)))
    out["capsize"] = dict(capsized=cap, t=mt.get("capsize_t"),
                          rate_ub95_per_h=rate_ub95(int(cap), hours))

    # ---- jet intake out of the water
    emerg = _runs(np.asarray(traj["submergence"]) < 0.0)
    out["intake"] = dict(emergence=len(emerg),
                         emergence_per_h=per_h(len(emerg)))

    # ---- numerical events, apart from capsize
    finite = np.all(np.isfinite(s), axis=1) & np.asarray(traj["ok"], bool)
    pitch = mt["th_rest"] + s[:, 4]
    with np.errstate(invalid="ignore"):
        kinds = dict(
            nonfinite=~finite,
            substep_cap=np.asarray(traj["n_sub"]) >= mt["n_sub_max"],
            overspeed=s[:, 6] > U_DIV * mt["u_max"],
            overpitch=np.abs(pitch) > TH_DIV)
    num = {kk: len(_runs(v)) for kk, v in kinds.items()}
    num.update({f"t_{kk}": _first_t(t, v) for kk, v in kinds.items()})
    num["any"] = any(num[kk] for kk in kinds)
    num["u_max"] = float(mt["u_max"])
    out["numerical"] = num

    # ---- MGN 328 band: speed along the waves near the wave speed
    sea = mt.get("sea") or {}
    if sea.get("tp") and sea.get("theta0") is not None and sea.get("hs"):
        c = G * sea["tp"] / (2.0 * np.pi)
        mu = s[:, 5] - sea["theta0"]
        u_w = s[:, 6] * np.cos(mu) - s[:, 7] * np.sin(mu)
        band = (u_w >= MGN_BAND[0] * c) & (u_w <= MGN_BAND[1] * c)
        out["mgn328"] = dict(c=float(c), dwell_s=float(band.sum() * dt),
                             share=float(band.mean()) if n else 0.0)
    else:
        out["mgn328"] = None
    return out


def summary(ev):
    """A few lines for the terminal."""
    lines = [f"  run {ev['duration_s']:.1f} s at dt {ev['dt']:.3f} s"]
    for kk, a in ev["acc"].items():
        w = ", ".join(f"{x:.2f}" for x in a["A110_windows_g"])
        lines.append(
            f"  k {kk:.1f}: {a['n_peaks']} peaks, max {a['peak_max_g']:.2f}"
            f" g, A1/10 per window [{w}] g"
            f"{' (partial)' if a['window_partial'] else ''}, windows > 3 g "
            f"{a['share_windows_over_3g']:.0%}, >= 7 g {a['peaks_ge_7g']}"
            f" ({a['peaks_ge_7g_per_h']:.0f}/h), >= 10 g "
            f"{a['peaks_ge_10g']} ({a['peaks_ge_10g_per_h']:.0f}/h)")
    b, y, tr = ev["bow"], ev["yaw"], ev["track"]
    lines.append(f"  bow: f0 {b['f0']:.3f} m, min height "
                 f"{b['min_height']:.3f} m, wet {b['wet']} "
                 f"({b['wet_per_h']:.0f}/h), buried {b['buried']} "
                 f"({b['buried_per_h']:.0f}/h)")
    lines.append(f"  yaw: broach {y['broach']}, heading dev > 10 deg "
                 f"{y['heading_dev_over_10']} (max "
                 f"{y['heading_dev_max_deg']:.1f} deg)")
    lines.append(f"  track: within 2L {tr['share_within_2L']:.0%}, max |y| "
                 f"{tr['max_abs_y']:.1f} m, fail 5L {tr['fail_5L']}")
    r, c = ev["roll"], ev["capsize"]
    lines.append(f"  roll: max {r['max_deg']:.2f} deg, > 20 deg "
                 f"{r['over_20']}, > fail {r['phi_fail_deg']:.1f} deg "
                 f"{r['over_fail']}; capsize {c['capsized']} (95% bound "
                 f"{c['rate_ub95_per_h']:.0f}/h)")
    nm = ev["numerical"]
    lines.append(f"  intake out {ev['intake']['emergence']} "
                 f"({ev['intake']['emergence_per_h']:.0f}/h); numerical: "
                 f"non-finite {nm['nonfinite']}, substep cap "
                 f"{nm['substep_cap']}, overspeed {nm['overspeed']}, "
                 f"overpitch {nm['overpitch']} (u_max {nm['u_max']:.2f} "
                 f"m/s)")
    if ev["mgn328"]:
        lines.append(f"  MGN 328 band: {ev['mgn328']['dwell_s']:.1f} s "
                     f"(c {ev['mgn328']['c']:.2f} m/s)")
    return "\n".join(lines)


# ------------------------------------------------------------------ check
def mission_gz(seed, head, t_end, sea=None):
    """A learn/repro/task.Mission in the high-fidelity world with the GZ
    plant (recording) in place of the one it built."""
    from learn.repro.task import Mission
    from sim.planing_vessel_gz import from_plant
    m = Mission("high", seed, head, t_end=t_end, sea=sea)
    gz = from_plant(m.plant, record=True)
    m.plant = m.ep.plant = gz
    m.s = gz.initial_state(0.8 * m.u_ref)
    m.s[5] = m.phi
    return m


def _wait_ram(need_gb=3.0, poll=60.0, limit=1800.0):
    import psutil
    t0 = time.time()
    while True:
        free = psutil.virtual_memory().available / 2 ** 30
        if free >= need_gb:
            return free
        if time.time() - t0 >= limit:
            return None
        print(f"  {free:.1f} GB free, waiting ...", flush=True)
        time.sleep(poll)


def main():
    import warnings
    warnings.filterwarnings("ignore")
    free = _wait_ram()
    if free is None:
        print("  under 3 GB of free RAM for 30 min: check skipped")
        return
    print(f"  {free:.1f} GB free")

    # 1. the curve
    m = mission_gz(0, 0, 31.25, sea=dict(hs=0.0))     # 125 x 0.24 s
    gz = m.plant
    rep = gz.report()
    print("\n  GZ curve (scarab195, constant displacement, rest trim)")
    print(f"    slope at 0: geometric {rep['GM_geom']:.5f} m, parent GM "
          f"{rep['GM']:.5f} m, shift {rep['dGM'] * 1e3:.2f} mm -> used "
          f"slope {gz.gz(1e-4) / 1e-4:.5f} m")
    print(f"    volume {rep['v0']:.5f} m3 (plant m/rho {rep['v_plant']:.5f})"
          f", roll period {rep['T_roll']:.2f} s")
    print(f"    vanishing stability {rep['phi_vanish_deg']:.1f} deg, deck "
          f"edge immersion {rep['phi_deck_deg']:.1f} deg, failure angle "
          f"{rep['phi_fail_deg']:.1f} deg, GZ max {rep['gz_max']:.3f} m at "
          f"{rep['phi_gz_max_deg']:.1f} deg")
    print("    GZ / (GM sin phi) at 1, 3, 5, 10 deg: " + ", ".join(
        f"{gz.gz(np.radians(d)) / (gz.GM * np.sin(np.radians(d))):.3f}"
        for d in (1, 3, 5, 10)))

    # 2. ITTC (22nd, HSMV committee) calm-water self-propelled check:
    # top speed, 3 deg initial heel, steering on, heel < 10 deg throughout
    t1 = time.time()
    u_top = gz.top_speed()
    m.s = gz.initial_state(u_top)
    m.s[3], m.s[5] = np.radians(3.0), m.phi
    thr = gz.prop.t_max
    while not m.done() and not gz.capsized:
        m.advance(thr, m.ep._steer(m.s, thr))
    tr = gz.trajectory(u_max=u_top)
    phi = np.degrees(np.abs(tr["s"][:, 3]))
    u = tr["s"][:, 6]
    print(f"\n  ITTC calm-water check ({time.time() - t1:.0f} s wall)")
    print(f"    top speed {u_top:.2f} m/s ({u_top / 0.514444:.1f} kn); "
          f"over {tr['t'][-1] + tr['meta']['dt']:.1f} s: speed "
          f"{u.min():.2f}-{u.max():.2f} m/s, max heel {phi.max():.2f} deg,"
          f" heel at end {phi[-1]:.3f} deg -> "
          f"{'PASS' if phi.max() < 10.0 else 'FAIL'} (< 10 deg)")

    # 3. one wave run: sea state 3, 45 deg off the bow, the MPC at 25 kn
    t1 = time.time()
    m = mission_gz(0, 1, 62.5)                         # 250 x 0.24 s
    while not m.done() and not m.plant.capsized:
        m.advance(*m.mpc_command())
    ev = safety_events(m.plant.trajectory(u_max=u_top), track=m.phi)
    print(f"\n  wave run, Hs 1.0 m Tp 5 s, 45 deg off the bow, seed 0 "
          f"({time.time() - t1:.0f} s wall)")
    print(summary(ev))

    # the same run on the parent plant: roll feeds nothing back, so every
    # other state must come out identical and only roll may differ
    from learn.repro.task import Mission
    m2 = Mission("high", 0, 1, t_end=62.5)
    step0, peak = m2.plant.step, [0.0]

    def step(s, t, a, b, dt=None):
        out = step0(s, t, a, b, dt)
        peak[0] = max(peak[0], abs(out[3]))
        return out
    m2.plant.step = step
    while not m2.done():
        m2.advance(*m2.mpc_command())
    other = [i for i in range(14) if i not in (3, 9)]
    print(f"  parent plant, same run: max roll {np.degrees(peak[0]):.2f} "
          f"deg (GZ plant {ev['roll']['max_deg']:.2f}); largest difference"
          f" in the other 12 states at the end "
          f"{np.abs(m2.s[other] - m.s[other]).max():.1e}")
    return ev


if __name__ == "__main__":
    main()
