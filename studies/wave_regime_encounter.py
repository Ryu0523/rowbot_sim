#!/usr/bin/env python3
"""
Encounter time scales against the MPC's own time scales, for any speed and
relative wave direction (deep water, one regular component at the peak
period -- the dominant one; no plant, pure arithmetic).

    omega   = 2 pi / Tp,  k = omega^2 / g,  c = g / omega,  lambda = 2 pi / k
    omega_e = omega + k U cos(beta)    beta = where the waves come FROM,
                                       measured from the bow: 0 head,
                                       45 bow-quartering, 90 beam,
                                       135 stern-quartering, 180 following
    T_e     = 2 pi / |omega_e|         (infinite when U cos(180 - beta) = c)

MPC time scales (control/mpc.py MPPIController defaults, learn/meta/
mpc_constrained.py via mpc_learned: H = 24 steps, 7 knots; final_eval's
control step 12 x 0.02 s = 0.24 s):
    horizon        24 x 0.24 = 5.76 s
    knot spacing   23 / 6 x 0.24 = 0.92 s (linear interpolation between
                   knots: a plan cannot express a throttle cycle shorter
                   than two knot spacings, 1.84 s)
    jet            0.1 s delay + 0.6 s first-order lag (sim/planing_vessel.py
                   Waterjet(tau=j.get('tau', 0.6), delay=j.get('delay', 0.1));
                   hydro/hulls.py scarab_195 jet tau 0.6): |H| = 1 / sqrt(1 +
                   (w tau)^2), lag = atan(w tau) + 0.1 w, as a share of the
                   encounter period
Plus, for this hull (L = 5.4 m): the IMO ship-handling zones quoted (and
marked verified) in learn/meta/SAFETY_STANDARDS_2026-09-30.md lines
198-203, and the plant's roll period from studies/_cache/wave_regimes/
model_gaps.json (wave_regime_model_gaps.py) when it exists.

Output: studies/_cache/wave_regimes/encounter.json and printed tables.

    .venv/Scripts/python.exe -m studies.wave_regime_encounter
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_cache", "wave_regimes")
KN = 0.514444
G = 9.81
L = 5.4
DTC = 0.24
H_STEPS, N_KNOTS = 24, 7
HORIZON = H_STEPS * DTC
KNOT = (H_STEPS - 1) / (N_KNOTS - 1) * DTC
JET_TAU, JET_DELAY = 0.6, 0.1
BETAS = (0, 45, 90, 135, 180)
NAMES = {0: "head", 45: "bow-quart 45", 90: "beam", 135: "stern-quart 135",
         180: "following"}
TPS = (5.0, 8.0, 10.0)


def wave(tp):
    w = 2 * math.pi / tp
    k = w * w / G
    return w, k, G / w, 2 * math.pi / k


def omega_e(tp, U, beta_deg):
    w, k, _, _ = wave(tp)
    return w + k * U * math.cos(math.radians(beta_deg))


def t_e(tp, U, beta_deg):
    we = abs(omega_e(tp, U, beta_deg))
    return float("inf") if we < 1e-9 else 2 * math.pi / we


def jet(we):
    """(amplitude passed, lag as a share of the period) of the thrust
    channel at encounter frequency we."""
    we = abs(we)
    if we < 1e-9:
        return 1.0, 0.0
    amp = 1.0 / math.sqrt(1.0 + (we * JET_TAU) ** 2)
    lag = math.atan(we * JET_TAU) + we * JET_DELAY
    return amp, lag / (2 * math.pi)


def speed_band(tp, beta, pred, lo=0.0, hi=40.0, step=0.05):
    """Speed intervals (kn) on [lo, hi] where pred(T_e) holds."""
    out, cur = [], None
    n = int(round((hi - lo) / step))
    for i in range(n + 1):
        kn = lo + i * step
        ok = pred(t_e(tp, kn * KN, beta))
        if ok and cur is None:
            cur = kn
        if not ok and cur is not None:
            out.append((round(cur, 2), round(kn - step, 2)))
            cur = None
    if cur is not None:
        out.append((round(cur, 2), round(hi, 2)))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    t_roll = None
    p = os.path.join(OUT, "model_gaps.json")
    if os.path.exists(p):
        t_roll = json.load(open(p))["roll"]["T_roll"]
    res = dict(horizon_s=HORIZON, knot_s=KNOT, dtc=DTC, L=L,
               jet=dict(tau=JET_TAU, delay=JET_DELAY), T_roll=t_roll,
               waves={}, table=[], bands={})
    print(f"horizon {HORIZON:.2f} s, knot spacing {KNOT:.3f} s (shortest "
          f"plannable throttle cycle {2 * KNOT:.2f} s), control step {DTC} s"
          + (f", plant roll period {t_roll:.2f} s" if t_roll else ""))
    for tp in TPS:
        w, k, c, lam = wave(tp)
        res["waves"][str(tp)] = dict(omega=w, k=k, c=c, c_kn=c / KN,
                                     lam=lam, lam_over_L=lam / L)
        print(f"\nTp {tp:.0f} s: wavelength {lam:.1f} m ({lam / L:.1f} L), "
              f"crest speed c {c:.2f} m/s = {c / KN:.1f} kn")
        print("   kn | " + " | ".join(f"{NAMES[b]:>15}" for b in BETAS)
              + "    (T_e s; jet amp; H/T_e)")
        for kn in (0, 3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 35):
            U = kn * KN
            cells = []
            for b in BETAS:
                we = omega_e(tp, U, b)
                te = t_e(tp, U, b)
                amp, lagf = jet(we)
                res["table"].append(dict(tp=tp, kn=kn, beta=b, omega_e=we,
                                         T_e=te, H_over_Te=HORIZON / te,
                                         knots_per_Te=te / KNOT,
                                         jet_amp=amp, jet_lag_share=lagf,
                                         dist_in_horizon_over_lambda=(
                                             U * HORIZON / lam)))
                tes = "  inf" if math.isinf(te) else f"{te:5.1f}"
                cells.append(f"{tes} {amp:4.2f} {HORIZON / te:4.2f}")
            print(f"  {kn:3d} | " + " | ".join(f"{s:>15}" for s in cells))
        # bands per heading
        for b in BETAS:
            key = f"{tp}_{b}"
            bd = dict(
                Te_gt_horizon=speed_band(tp, b, lambda te: te > HORIZON),
                Te_lt_2knots=speed_band(tp, b, lambda te: te < 2 * KNOT),
                Te_lt_4steps=speed_band(tp, b, lambda te: te < 4 * DTC),
                Te_gt_20s=speed_band(tp, b, lambda te: te > 20.0))
            if t_roll:
                bd["Te_near_Troll"] = speed_band(
                    tp, b, lambda te: abs(te / t_roll - 1.0) < 0.15)
                bd["Te_near_half_Troll"] = speed_band(
                    tp, b, lambda te: abs(te / (0.5 * t_roll) - 1.0) < 0.15)
            if b > 90:
                cb = -math.cos(math.radians(b))
                bd["U_crit_kn"] = c / cb / KN       # omega_e = 0
                # MGN 328 band (studies/safety_events.py MGN_BAND): speed
                # along the wave direction in [0.7, 1.15] c
                bd["mgn328_kn"] = (0.7 * c / cb / KN, 1.15 * c / cb / KN)
                # MSC.1/Circ.1228 4.2.1: V > 1.8 sqrt(L) / cos(180 - alpha)
                bd["circ1228_421_kn"] = 1.8 * math.sqrt(L) / cb
                # 4.2.2 successive high waves: T_E / T_W in [1.8, 3.0],
                # T_E = 3 T_W^2 / (3 T_W + V cos alpha), V kn, alpha 0 head
                bd["circ1228_422_kn"] = (3 * tp * (1 - 1 / 1.8) / cb,
                                         3 * tp * (1 - 1 / 3.0) / cb)
            res["bands"][key] = bd
        print("  speed bands (kn):")
        for b in BETAS:
            bd = res["bands"][f"{tp}_{b}"]
            extra = ""
            if "U_crit_kn" in bd:
                extra = (f"; omega_e = 0 at {bd['U_crit_kn']:.1f}; MGN 328 "
                         f"{bd['mgn328_kn'][0]:.1f}-{bd['mgn328_kn'][1]:.1f};"
                         f" Circ.1228 4.2.1 above {bd['circ1228_421_kn']:.1f};"
                         f" 4.2.2 {bd['circ1228_422_kn'][0]:.1f}-"
                         f"{bd['circ1228_422_kn'][1]:.1f}")
            roll = ""
            if t_roll:
                roll = (f"; T_e ~ T_roll {bd['Te_near_Troll']}"
                        f"; T_e ~ T_roll/2 {bd['Te_near_half_Troll']}")
            print(f"    {NAMES[b]:>15}: T_e > horizon {bd['Te_gt_horizon']};"
                  f" T_e < 2 knots {bd['Te_lt_2knots']}; T_e > 20 s "
                  f"{bd['Te_gt_20s']}{roll}{extra}")
    res["imo_fn03_kn"] = 0.94 * math.sqrt(L) / KN
    print(f"\nMSC.1/Circ.1627 4.5.6.2.4 (.1): avoid nominal speed >= 0.94 "
          f"sqrt(L) = {0.94 * math.sqrt(L):.2f} m/s = "
          f"{res['imo_fn03_kn']:.2f} kn with heading within 45 deg of the "
          f"wave direction, when lambda > 0.8 L ({0.8 * L:.1f} m) and Hs > "
          f"0.04 L ({0.04 * L:.2f} m)")
    with open(os.path.join(OUT, "encounter.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"saved {os.path.join(OUT, 'encounter.json')}")


if __name__ == "__main__":
    main()
