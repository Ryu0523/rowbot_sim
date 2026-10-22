#!/usr/bin/env python3
"""
Report of the regime scan (studies/wave_regime_scan.py): reads
studies/_cache/wave_regimes/scan.pkl, writes scan_report.txt beside it.
Kept apart from the scan module so that it can be written and rerun while
the scan's workers (which re-import their module) are still running.

  python -m studies.wave_regime_report
"""
import math
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

OUT = os.path.join(HERE, "_cache", "wave_regimes")
KN = 0.514444
TAU_JET, DELAY_JET = 0.6, 0.1          # sim/planing_vessel.py:131-133


def nanmean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def nanmax(x):
    x = np.asarray(x, float)
    return float(np.nanmax(x)) if np.isfinite(x).any() else float("nan")


def band(kn):
    if kn < 10.0:
        return "<10"
    if kn < 18.0:
        return "10-18"
    return ">=18"


def main(path=None):
    path = path or os.path.join(OUT, "scan.pkl")
    R = pickle.load(open(path, "rb"))
    res = R["results"]
    errs = R.get("errors", {})
    seas = sorted({k[0] for k in res})
    chis = sorted({k[1] for k in res})
    fs = sorted({k[2] for k in res})
    seeds = sorted({k[3] for k in res})
    L = []
    P = L.append
    P(f"REGIME SCAN: {len(res)} runs ({len(errs)} errors), seas {seas}, "
      f"relative headings {chis} (0 head .. 180 following), throttles {fs},"
      f" seeds {seeds}; constant throttle + the mission heading autopilot "
      f"+ SafeMode, PlaningVesselGZ; statistics after the first 20 s "
      f"(100 s per run at T 120).")
    P("Columns are means over seeds unless marked: 'max' = max over seeds, "
      "'sum' = summed over seeds, rates per hour from the summed counts over"
      " the summed time.")

    def grp(sn, c, f):
        return [res[(sn, c, f, s)] for s in seeds if (sn, c, f, s) in res]

    # ---------------------------------------------------- encounter check
    P("")
    P("1. RELATIVE HEADING CHECK by the measured encounter: per sea and "
      "throttle, Tz = mean zero-up-crossing period of the surface at the bow"
      " (s), we = signed encounter frequency measured as minus the phase rate"
      " of the surface's analytic signal at the CG (rad/s; < 0: the boat "
      "overtakes the waves), [theory: spectrum-weighted w - k.V at the "
      "boat's velocity]. Expect we to fall from head (0) to following "
      "(180) at the same throttle.")
    nbad = 0
    for sn in seas:
        P(f"  sea {sn}:")
        P("      f    " + "".join(f"   chi {c:3d}: kn  Tz    we [th]   "
                                  for c in chis))
        for f in fs:
            row, wes = [], []
            for c in chis:
                g = grp(sn, c, f)
                if not g:
                    row.append(" " * 36)
                    wes.append(np.nan)
                    continue
                kn = nanmean([r["kn"] for r in g])
                tz = nanmean([r["Tz_bow"] for r in g])
                we = nanmean([r["we_meas"] for r in g])
                wt = nanmean([r["we_theory"] for r in g])
                wes.append(we)
                row.append(f"  {kn:5.1f} {tz:5.2f} {we:+5.2f} [{wt:+5.2f}]   ")
            ok = all(np.diff([w for w in wes if np.isfinite(w)]) < 0.05)
            nbad += (not ok)
            P(f"    {f:5.2f} " + "".join(row) + ("" if ok else "  <- not monotone"))
    P(f"  monotone (we falling from head to following, tolerance +0.05): "
      f"{'all rows' if nbad == 0 else f'{nbad} rows not'}")

    # ---------------------------------------------------- hazards
    P("")
    P("2. HAZARDS per sea, relative heading and throttle (k = 2 events of "
      "studies/safety_events, as final_eval). kn = speed made good; A110 = "
      "A1/10 g; g7/h = peaks >= 7 g per hour; wet/h, bur/h = bow-wet and "
      "bow-burial entries per hour; hdgX = max heading deviation (deg, max "
      "over seeds); hdg99 = 99th pct of |heading dev| (deg); r99 = 99th pct "
      "|yaw rate| (deg/s); sat = share of time nozzle >= 95% of max; brch = "
      "MSC.1/Circ.1627 3.5.2.4-type broach events (sum); yF = runs with "
      "|y| > 5 L (sum); rollX = max roll (deg, max over seeds); safe = SafeMode"
      " entries (sum); cap = capsizes (sum); pd = pump prime deficit, mean of"
      " (1 - prime); in/h = intake emergences per hour; num = runs with a "
      "numerical event (sum).")
    hdr = ("      f     kn  A110  g7/h  wet/h bur/h hdgX hdg99  r99   sat "
           "brch yF rollX safe cap   pd   in/h num")
    for sn in seas:
        for c in chis:
            P(f"  sea {sn}, chi {c}:")
            P(hdr)
            for f in fs:
                g = grp(sn, c, f)
                if not g:
                    continue
                hrs = sum(r["ev"]["hours"] for r in g)
                sm = lambda k: sum(r["ev"][k] for r in g)  # noqa: E731
                P(f"    {f:5.2f} {nanmean([r['kn'] for r in g]):5.1f} "
                  f"{nanmean([r['ev']['a110'] for r in g]):5.2f} "
                  f"{sm('n_ge7') / hrs:5.0f} {sm('bow_wet') / hrs:6.0f} "
                  f"{sm('bow_burial') / hrs:5.0f} "
                  f"{nanmax([r['heading_max_deg'] for r in g]):4.0f} "
                  f"{nanmean([r['heading_p99_deg'] for r in g]):5.1f} "
                  f"{nanmean([r['r_p99_degs'] for r in g]):5.1f} "
                  f"{nanmean([r['noz_sat_share'] for r in g]):5.2f} "
                  f"{sm('broach'):4d} {sum(int(r['y_fail']) for r in g):2d} "
                  f"{nanmax([r['roll_max_deg'] for r in g]):5.1f} "
                  f"{sum(r['safe_entries'] for r in g):4d} "
                  f"{sum(int(r['capsized']) for r in g):3d} "
                  f"{nanmean([r['prime_deficit'] for r in g]):5.3f} "
                  f"{sm('intake_out') / hrs:5.0f} "
                  f"{sum(int(bool(r['numeric'].get('any'))) for r in g):3d}")

    # ---------------------------------------------------- wave locking
    P("")
    P("3. SURF-RIDING AND WAVE-LOCKING per sea, heading, throttle. usd = "
      "speed sd (kn); uR2 / uA = share of the speed variance following the "
      "bow wave phase (first harmonic R^2) and its amplitude (kn); uw/c = "
      "mean speed along the wave direction over the peak celerity; >.9c = "
      "share of time above 0.9 c; surf = share of time with |encounter "
      "frequency| (2 s window) < 0.2 w_p AND speed along the waves in "
      "[0.7, 1.15] c; sL = longest such spell (s, max over seeds); cL = "
      "longest spell with the CG held within a quarter wave of phase for at "
      "least one Tp (s, max); impacts: n2 = peaks >= 2 g (sum), R2 = their "
      "bow-phase concentration (0 none .. 1 all at one phase), e2 / e7 = mean"
      " percentile of the bow wave envelope at the >= 2 g / >= 7 g peaks "
      "(0.5 = no preference for big waves); ew = same at bow-wet entries; "
      "eh = same at the starts of heading excursions > 10 deg (nh = their "
      "count, sum); rR2 = yaw-rate variance following the bow wave phase.")
    hdr = ("      f     kn  usd  uR2   uA  uw/c >.9c  surf   sL    cL   "
           "n2   R2   e2   e7   ew   nh   eh   rR2")
    for sn in seas:
        for c in chis:
            P(f"  sea {sn}, chi {c}:")
            P(hdr)
            for f in fs:
                g = grp(sn, c, f)
                if not g:
                    continue
                n2 = sum(r["lock"]["n2"] for r in g)
                # pooled phase concentration over the seeds' >= 2 g peaks,
                # from each run's resultant (weighted by its count)
                zs = [r["lock"]["n2"] * r["lock"]["R2"]
                      * np.exp(1j * math.radians(r["lock"]["ph2"]))
                      for r in g if r["lock"]["n2"] > 0]
                R2 = abs(sum(zs)) / n2 if n2 else float("nan")
                P(f"    {f:5.2f} {nanmean([r['kn'] for r in g]):5.1f} "
                  f"{nanmean([r['u_sd_kn'] for r in g]):4.1f} "
                  f"{nanmean([r['u_lock_r2'] for r in g]):4.2f} "
                  f"{nanmean([r['u_lock_amp_kn'] for r in g]):4.1f} "
                  f"{nanmean([r['uw_over_c'] for r in g]):+5.2f} "
                  f"{nanmean([r['uw_gt09c_share'] for r in g]):4.2f} "
                  f"{nanmean([r['surf_share'] for r in g]):5.2f} "
                  f"{nanmax([r['surf_long_s'] for r in g]):4.1f} "
                  f"{nanmax([r['capt_long_s'] for r in g]):5.1f} "
                  f"{n2:4d} {R2:4.2f} "
                  f"{nanmean([r['lock']['env2'] for r in g]):4.2f} "
                  f"{nanmean([r['lock']['env7'] for r in g]):4.2f} "
                  f"{nanmean([r['lock']['envw'] for r in g]):4.2f} "
                  f"{sum(r['lock']['nh'] for r in g):4d} "
                  f"{nanmean([r['lock']['envh'] for r in g]):4.2f} "
                  f"{nanmean([r['lock']['r2_r'] for r in g]):5.2f}")

    # ---------------------------------------------------- actuator per wave
    P("")
    P("4. CAN THE THROTTLE ACT WITHIN ONE ENCOUNTER? Theory of the plant's "
      "thrust channel (sim/planing_vessel.py:131-133: first-order lag tau "
      f"{TAU_JET} s, delay {DELAY_JET} s; rate limit and dead band ignored):"
      " share of a sinusoidal thrust command at the measured |we| that "
      "reaches the jet, 1/sqrt(1 + (we tau)^2), and its lag in degrees of "
      "wave phase, atan(we tau) + we delay. Per sea and heading, at the "
      "lowest / middle / highest throttle (|we| from the scan).")
    for sn in seas:
        for c in chis:
            parts = []
            for f in (fs[0], fs[len(fs) // 2], fs[-1]):
                g = grp(sn, c, f)
                if not g:
                    continue
                we = abs(nanmean([r["we_meas"] for r in g]))
                kn = nanmean([r["kn"] for r in g])
                gain = 1 / math.sqrt(1 + (we * TAU_JET) ** 2)
                lag = math.degrees(math.atan(we * TAU_JET) + we * DELAY_JET)
                te = 2 * math.pi / we if we > 1e-6 else float("inf")
                parts.append(f"f {f:.2f} ({kn:4.1f} kn): |we| {we:4.2f}, Te "
                             f"{te:5.1f} s, gain {gain:4.2f}, lag {lag:3.0f}")
            P(f"  {sn} chi {c:3d}: " + " | ".join(parts))

    # ---------------------------------------------------- by speed band
    P("")
    P("5. BY SPEED BAND (runs pooled by their own speed made good: <10 kn "
      "below the calm hump, 10-18 kn hump/transition, >=18 kn planing) and "
      "heading, both seas: n runs; g7/h, wet/h, bur/h rates; max heading "
      "deviation (max) and its 99th pct (mean); broach / y-fail / safe / "
      "capsize sums; mean surf share; impact wave-locking (mean e2, e7).")
    P("    sea  chi band   n   g7/h  wet/h bur/h hdgX hdg99 brch yF safe cap "
      " surf  e2   e7  rR2")
    for sn in seas:
        for c in chis:
            for b in ("<10", "10-18", ">=18"):
                g = [r for k, r in res.items() if k[0] == sn and k[1] == c
                     and np.isfinite(r["kn"]) and band(r["kn"]) == b]
                if not g:
                    continue
                hrs = sum(r["ev"]["hours"] for r in g)
                sm = lambda k: sum(r["ev"][k] for r in g)  # noqa: E731
                P(f"    {sn} {c:3d} {b:>5s} {len(g):3d} "
                  f"{sm('n_ge7') / hrs:6.0f} {sm('bow_wet') / hrs:6.0f} "
                  f"{sm('bow_burial') / hrs:5.0f} "
                  f"{nanmax([r['heading_max_deg'] for r in g]):4.0f} "
                  f"{nanmean([r['heading_p99_deg'] for r in g]):5.1f} "
                  f"{sm('broach'):4d} {sum(int(r['y_fail']) for r in g):2d} "
                  f"{sum(r['safe_entries'] for r in g):4d} "
                  f"{sum(int(r['capsized']) for r in g):3d} "
                  f"{nanmean([r['surf_share'] for r in g]):5.2f} "
                  f"{nanmean([r['lock']['env2'] for r in g]):4.2f} "
                  f"{nanmean([r['lock']['env7'] for r in g]):4.2f} "
                  f"{nanmean([r['lock']['r2_r'] for r in g]):4.2f}")
    if errs:
        P("")
        P("ERRORS:")
        for k, v in errs.items():
            P(f"  {k}: {v.strip().splitlines()[-1]}")
    txt = "\n".join(L)
    with open(os.path.join(OUT, "scan_report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
