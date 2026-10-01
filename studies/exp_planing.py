#!/usr/bin/env python3
"""
The planing plant (sim/planing_vessel.py) against what is known.

P1  calm water vs Savitsky (1964): running trim and resistance of the
    Scarab 195 stand-in, 20-40 kn. Savitsky is a regression on towing-tank
    tests of prismatic planing surfaces. The plant's three 2D+t coefficients
    are Zarnick's; fitted to Savitsky instead they porpoised (see below).
P2  top speed vs the Scarab 195's measured speeds (Boating magazine, two
    loads). NOT used in the calibration: an independent check of hull +
    jet together.
P3  turning at 25 kn and course stability (yaw rate decays after the nozzle
    is centred). No manoeuvring data exist for this boat: plausibility only.
P4  head seas, sea state 3, fixed throttle: speed loss, motions, impact
    accelerations, loss of pump prime -- and the same statistics at three
    time steps (the impact peaks are the step-sensitive part).
P5  regular head waves, towed at constant speed and free in heave and
    pitch, as in a tank: heave / wave amplitude and pitch / wave slope
    against wavelength (the shape Fridsma's tests are reported in).
P6  Fridsma (1969) towed prismatic model, regular head waves, two speeds:
    heave, pitch, CG and bow accelerations against his tabulated values and
    Zarnick's (1978) 2D+t computation. The independent test of the wave
    response: nothing here was fitted to it.
P7  Fridsma (1971) irregular head seas: mean of the positive acceleration
    peaks at CG and bow. His modal period is not in our copy; assumed fully
    developed, with the sensitivity to that shown.

Coefficients: Zarnick's (k_a 1.0, C_dc cos beta, a_b 0.5). A calibration
to Savitsky alone picked k_a 0.6, which porpoised by itself in Fridsma's
case A; see DEFECTS H7.

Run: python -m studies.exp_planing
"""
import warnings

import numpy as np

from hydro.planing import savitsky
from sim import config
from sim.test_vessel import Monochromatic
from sim.wavefield import SeaState

warnings.filterwarnings("ignore")
KN = 0.514444
HULL = "scarab195"


def p1(h, p):
    print("\nP1  calm water: plant vs Savitsky (same hull numbers)")
    print(f"    {'kn':>4}{'trim':>8}{'Sav.':>7}{'R kN':>8}{'Sav.':>7}{'rise m':>8}")
    errs_t, errs_r = [], []
    for kn in (20, 25, 30, 35, 40):
        r = p.calm_resistance(kn * KN)
        s = savitsky(kn * KN, h.mass, h.lcg, h.b, h.beta_deg)
        R = r["R"] - r["air"]
        errs_t.append(r["trim_deg"] - s["tau_deg"])
        errs_r.append((R - s["R"]) / s["R"])
        print(f"    {kn:>4}{r['trim_deg']:>8.2f}{s['tau_deg']:>7.2f}"
              f"{R / 1e3:>8.2f}{s['R'] / 1e3:>7.2f}{r['rise']:>8.3f}")
    print(f"    trim within {min(errs_t):+.2f}..{max(errs_t):+.2f} deg, "
          f"resistance {100 * min(errs_r):+.0f}..{100 * max(errs_r):+.0f}%  "
          f"(coefficients k_a {h.k_a}, C_dc {h.c_dc}, a_b {h.a_b})")


def p2(h):
    from hydro.hulls import scarab_195
    print("\nP2  top speed at full throttle vs the Scarab 195 tests "
          "(not calibrated on)")
    for mass, meas, yr in ((1340.0, 42.7, 2019), (1490.0, 42.06, 2024)):
        hh = scarab_195()
        hh.mass = mass
        p = config.plant_for(None, Monochromatic(1.0, 0.0), hh)
        s = p.initial_state(30 * KN)
        n = int(60 / p.dt)
        u = np.empty(n)
        for i in range(n):
            s = p.step(s, i * p.dt, p.prop.t_max, 0.0)
            u[i] = s[6]
        top = u[-n // 4:].mean() / KN
        print(f"    {yr} test load, {mass:.0f} kg: plant {top:.1f} kn, "
              f"measured {meas} kn ({100 * (top / meas - 1):+.1f}%)")


def p3(h, p):
    print("\nP3  turning at 25 kn, then the nozzle centred (course "
          "stability)")
    for deg in (5, 10, 20):
        v = p.with_sea(Monochromatic(1.0, 0.0))
        s = v.initial_state(25 * KN)
        thr, t = s[12], 0.0
        for _ in range(int(30 / v.dt)):
            s = v.step(s, t, thr, np.radians(deg))
            t += v.dt
        r, u = s[11], np.hypot(s[6], s[7])
        line = (f"    nozzle {deg:>2} deg: {abs(np.degrees(r)):5.1f} deg/s, "
                f"radius {u / abs(r) / h.L:5.1f} L, drift "
                f"{abs(np.degrees(np.arctan2(s[7], s[6]))):4.1f} deg, speed "
                f"{s[6] / KN:4.1f} kn")
        for _ in range(int(20 / v.dt)):
            s = v.step(s, t, thr, 0.0)
            t += v.dt
        print(line + f"; 20 s after centring {abs(np.degrees(s[11])):.3f} "
              f"deg/s")


def p4(h):
    print("\nP4  head seas, Hs 1.0 m, Tp 5 s (sea state 3), throttle fixed "
          "at the calm-water 25 kn setting, 90 s")
    print(f"    {'dt s':>6}{'speed kn':>10}{'trim':>7}{'+-':>6}{'heave sd':>10}"
          f"{'CG p99 g':>10}{'bow p99 g':>11}{'prime lost':>12}")
    for dt in (0.005, 0.01, 0.02):
        sea = SeaState(1.0, 5.0, theta0=np.pi, n_freq=24, n_dir=5, seed=0)
        p = config.plant_for(None, sea, h, dt=dt)
        s = p.initial_state(25 * KN)
        thr = s[12]
        n = int(90 / dt)
        rec = np.empty((n, 6))
        for i in range(n):
            s = p.step(s, i * dt, thr, 0.0)
            rec[i] = (s[6], np.degrees(-(p.Th_rest + s[4])), s[2],
                      abs(p.last_cg_acc) / 9.81, abs(p.last_bow_acc) / 9.81,
                      p.prop.prime)
        q = rec[n // 6:]
        print(f"    {dt:>6}{q[:, 0].mean() / KN:>10.1f}{q[:, 1].mean():>7.2f}"
              f"{q[:, 1].std():>6.2f}{q[:, 2].std():>10.3f}"
              f"{np.percentile(q[:, 3], 99):>10.2f}"
              f"{np.percentile(q[:, 4], 99):>11.2f}"
              f"{100 * np.mean(q[:, 5] < 0.5):>11.1f}%")


def regular_wave(hull, V, lam_over_L, H, T=40.0, dt=0.005, x_acc=None):
    """Towed-model test, as in the tank: speed held, free in heave and
    pitch, head regular waves of height H and length lam_over_L * L.

    Returns heave amplitude / wave amplitude (= Fridsma's h/H, double over
    double) and pitch amplitude / wave slope k*a (= his theta_p/(2 pi H /
    lambda)), from half the peak-to-peak over the last half of the run; and
    the peak UPWARD accelerations at the CG and at x_acc (body x; default
    0.1 L aft of the stem, Fridsma's bow gauge), in g, averaged over the
    encounter cycles of the last half -- Fridsma averaged 10 cycles."""
    from sim.planing_vessel import PlaningVessel
    L = hull.L
    k = 2 * np.pi / (lam_over_L * L)
    a = 0.5 * H
    om = np.sqrt(9.81 * k)
    sea = Monochromatic(om, a, theta0=np.pi)
    p = PlaningVessel(hull, sea, dt=dt, captive_u=V)
    x_acc = (hull.L - hull.lcg - 0.1 * hull.L) if x_acc is None else x_acc
    s = np.zeros(14)
    s[6] = V
    s[2], s[4] = p.running_attitude(V)
    n = int(T / dt)
    rec = np.empty((n, 4))
    xb = p.x[-1]
    for i in range(n):
        s = p.step(s, i * dt, 0.0, 0.0)
        s[6] = V
        a_cg, a_b = p.last_cg_acc, p.last_bow_acc
        rec[i] = s[2], s[4], a_cg / 9.81, (a_cg + (a_b - a_cg) * x_acc
                                             / xb) / 9.81
    q = rec[n // 2:]
    amp = 0.5 * (q[:, :2].max(0) - q[:, :2].min(0))
    te = 2 * np.pi / (om + k * V)                   # encounter period
    m = max(int(round(te / dt)), 1)
    cyc = [q[j:j + m] for j in range(0, len(q) - m + 1, m)]
    pk = np.mean([[c[:, 2].max(), c[:, 3].max()] for c in cyc], axis=0)
    return amp[0] / a, amp[1] / (k * a), float(pk[0]), float(pk[1])


def fridsma_hull(case):
    """Fridsma's (1969) prismatic model: b 9 in, L/b 5, deadrise 20 deg,
    C_Delta 0.608 in fresh water (7.26 kg), VCG 0.294 b, bow one beam long.
    A: V/sqrt(L) 4, LCG 59% L aft of the stem, gyradius 25.1% L.
    B: V/sqrt(L) 6, LCG 62%, gyradius 25.5%. The keel here is straight
    (his bow had an elliptical keel profile). No jet: the model is towed."""
    from hydro.planing import PlaningHull
    b = 9 * 0.0254
    L = 5 * b
    lcg_aft, kp, vl = {"A": (0.59, 0.251, 4), "B": (0.62, 0.255, 6)}[case]
    V = vl * 0.514444 * np.sqrt(L / 0.3048)          # V/sqrt(L) in kn/ft^0.5
    return PlaningHull(
        name=f"fridsma_{case}", L=L, b=b, beta_deg=20.0,
        mass=0.608 * 62.4 * (b / 0.3048) ** 3 * 0.45359237,
        lcg=(1 - lcg_aft) * L, vcg=0.294 * b, depth=0.6 * b, k_pitch=kp,
        taper=0.2, stem_rise=0.0, cda=0.0, u_design=V, rho=1000.0,
        jet=dict(d_nozzle=0.02, p_max=500.0)), V


# Fridsma (1969) Table 2, H/b = 0.111 (stated); Zarnick (1978) computed, in
# parentheses in the agent's transcription, digitised +-0.03 / +-0.02 g.
# lam/L: (h/H, pitch ratio, CG g, bow g)
FRIDSMA = {
    "A": {1.0: (0.18, 0.11, 0.25, 1.05), 2.0: (0.84, 0.79, 0.25, 0.90),
          3.0: (1.18, 1.29, 0.20, 0.45), 4.0: (1.23, 1.21, 0.15, 0.25),
          6.0: (1.04, 1.13, 0.10, 0.15)},
    "B": {1.0: (0.16, 0.06, 0.50, 1.45), 1.5: (0.37, 0.21, 0.70, 2.00),
          2.0: (0.64, 0.54, 0.80, 2.55), 3.0: (1.45, 1.68, 0.90, 2.85),
          4.0: (1.75, 2.39, 0.35, 1.25), 6.0: (1.08, 1.51, 0.15, 0.25)},
}
ZARNICK = {
    "A": {1.0: (0.12, 0.15), 2.0: (0.70, 0.63), 3.0: (1.08, 1.12),
          4.0: (1.18, 1.31), 6.0: (1.16, 1.29)},
    "B": {1.0: (0.23, 0.11), 1.5: (0.41, 0.26), 2.0: (0.60, 0.53),
          3.0: (1.33, 1.55), 4.0: (1.78, 2.32), 6.0: (1.25, 1.62)},
}


def p6():
    print("\nP6  Fridsma (1969) towed prismatic model in regular head waves, "
          "H/b = 0.111 -- an independent test: Zarnick's coefficients, "
          "nothing fitted to these data")
    for case in ("A", "B"):
        h, V = fridsma_hull(case)
        from sim.planing_vessel import PlaningVessel
        p = PlaningVessel(h, Monochromatic(1.0, 0.0), dt=0.004)
        cr = p.calm_resistance(V)
        print(f"\n    case {case}: V {V:.2f} m/s; smooth water trim "
              f"{cr['trim_deg']:.2f} deg (measured 4), rise "
              f"{cr['rise'] / h.b:.3f} b (measured {0.05 if case == 'A' else 0.13}"
              f" b), R/Delta {(cr['R']) / (h.mass * 9.81):.3f} (measured "
              f"{0.158 if case == 'A' else 0.206})")
        print(f"    {'lam/L':>6}{'heave h/H':>18}{'pitch ratio':>20}"
              f"{'CG g':>16}{'bow g':>16}")
        print(f"    {'':>6}{'ours  meas  Zarn':>18}{'ours  meas  Zarn':>20}"
              f"{'ours  meas':>16}{'ours  meas':>16}")
        for lam, (mh, mp, mc, mb) in FRIDSMA[case].items():
            hv, pt, cg, bw = regular_wave(h, V, lam, 0.111 * h.b, T=16.0,
                                          dt=0.004)
            zh, zp = ZARNICK[case].get(lam, (np.nan, np.nan))
            print(f"    {lam:>6.1f}{hv:>7.2f}{mh:>6.2f}{zh:>5.2f}"
                  f"{pt:>9.2f}{mp:>6.2f}{zp:>5.2f}{cg:>10.2f}{mc:>6.2f}"
                  f"{bw:>10.2f}{mb:>6.2f}")


def excursion_peaks(y):
    """One peak per positive excursion: the largest value between the
    signal rising through zero and falling back through it. Counting every
    local maximum instead also counts the ripples inside an excursion and
    halved the mean against Fridsma's."""
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
    return np.asarray(out) if out else np.zeros(1)


# Fridsma (1971) irregular head seas, beta 20, L/b 5, C_Delta 0.600, trim 4:
# condition: (V/sqrt(L), LCG % L aft of the stem, H1/3/b, mean of ALL
# positive peaks at the CG and at the bow, g) -- stated in his tables
FRIDSMA71 = {22: (4, 59.2, 0.222, 0.45, 1.65), 21: (4, 59.2, 0.444, 0.71, 2.35),
             23: (4, 59.2, 0.667, 1.05, 3.09), 31: (6, 64.0, 0.222, 0.68, 2.10),
             30: (6, 64.0, 0.444, 1.77, 5.33)}


def p7(T=60.0, dt=0.004):
    """Fridsma's (1971) irregular-wave tests: towed at constant speed in
    Pierson-Moskowitz seas. His modal period is not in our copy, so the sea
    is taken as fully developed, Tp = 15.63 sqrt(Hs / g) (the P-M relation
    between height and period), and the answer is shown again with Tp x0.8
    and x1.25: how much the comparison rests on that assumption."""
    from hydro.planing import PlaningHull
    from sim.planing_vessel import PlaningVessel
    print("\nP7  Fridsma (1971) irregular head seas, towed: mean of all "
          "positive acceleration peaks, g")
    print(f"    {'cond':>5}{'V/rtL':>6}{'H/b':>6}{'CG ours (Tp x0.8/1/1.25)':>27}"
          f"{'meas':>6}{'bow ours (x0.8/1/1.25)':>26}{'meas':>6}")
    b = 9 * 0.0254
    L = 5 * b
    for cond, (vl, lcg_aft, hb, m_cg, m_bow) in FRIDSMA71.items():
        V = vl * 0.514444 * np.sqrt(L / 0.3048)
        h = PlaningHull(name=f"fridsma71_{cond}", L=L, b=b, beta_deg=20.0,
                        mass=0.600 * 62.4 * (b / 0.3048) ** 3 * 0.45359237,
                        lcg=(1 - lcg_aft / 100) * L, vcg=0.294 * b,
                        depth=0.6 * b, k_pitch=0.25, taper=0.2, cda=0.0,
                        u_design=V, rho=1000.0,
                        jet=dict(d_nozzle=0.02, p_max=500.0))
        hs = hb * b
        cgs, bows = [], []
        for f in (0.8, 1.0, 1.25):
            tp = f * 15.63 * np.sqrt(hs / 9.81)
            sea = SeaState(hs, tp, gamma=1.0, theta0=np.pi, n_freq=40,
                           n_dir=1, seed=cond)
            p = PlaningVessel(h, sea, dt=dt, captive_u=V)
            s = np.zeros(14)
            s[6] = V
            s[2], s[4] = p.running_attitude(V)
            xa = h.L - h.lcg - 0.1 * h.L
            n = int(T / dt)
            a = np.empty((n, 2))
            for i in range(n):
                s = p.step(s, i * dt, 0.0, 0.0)
                s[6] = V
                ac, ab = p.last_cg_acc, p.last_bow_acc
                a[i] = ac / 9.81, (ac + (ab - ac) * xa / p.x[-1]) / 9.81
            a = a[n // 10:]
            pk = []
            for j in (0, 1):
                pk.append(excursion_peaks(a[:, j]).mean())
            cgs.append(pk[0])
            bows.append(pk[1])
        print(f"    {cond:>5}{vl:>6}{hb:>6.3f}"
              f"{'  ' + '/'.join(f'{v:.2f}' for v in cgs):>27}{m_cg:>6.2f}"
              f"{'  ' + '/'.join(f'{v:.2f}' for v in bows):>26}{m_bow:>6.2f}")


def main():
    h, db = config.load(HULL)
    p = config.plant_for(db, Monochromatic(1.0, 0.0), h)
    print(f"{h.name}: L {h.L} m, chine beam {h.b:.2f} m, deadrise "
          f"{h.beta_deg} deg, {h.mass:.0f} kg; at rest draught {p.T:.3f} m, "
          f"GM {p.GM:.2f} m")
    p1(h, p)
    p2(h)
    p3(h, p)
    p4(h)
    print("\nP5  regular head waves, 25 kn, H 0.1 m, towed (speed held)")
    print(f"    {'lam/L':>6}{'heave/a':>9}{'pitch/ka':>10}{'CG g':>7}"
          f"{'bow g':>7}")
    for lam in (1.0, 2.0, 3.0, 4.0, 6.0, 10.0):
        hv, pt, cg, bw = regular_wave(h, 25 * KN, lam, 0.10, T=30.0, dt=0.01)
        print(f"    {lam:>6.1f}{hv:>9.2f}{pt:>10.2f}{cg:>7.2f}{bw:>7.2f}")
    p6()
    p7()


if __name__ == "__main__":
    main()
