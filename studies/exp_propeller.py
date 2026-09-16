#!/usr/bin/env python3
"""
Experiment: the propeller model against measured open-water curves.

THE DATA
The KVLCC2 benchmark propeller was tested in open water at two institutes, and
both results are published with the SIMMAN benchmark:
  NMRI  model D = 0.0896 m, Rn = 3.65e5, P/D(0.7R) = 0.7212, AE/A0 = 0.431,
        Z = 4, skew 21.15 deg        (KVLCC_POT_NMRI.txt)
  HMRI  scale 1:46.426, model and full-scale-corrected curves (PDF)
Two institutes, two model scales -- their spread is the honest measurement
uncertainty, and it is shown alongside.

THE "HULL"
For a propeller experiment the matching object is the propeller that was
measured. The B-series is evaluated at KVLCC2's own Z, AE/A0 and P/D. KVLCC2's
propeller is not a B-series design (21 deg skew, different sections), so this
measures the error of using the B-series as a PROXY for a real propeller --
which is exactly what the USV model does when its own propeller is unknown.

THEN
The same B-series is used to ask what the old hand-set placeholder
(kt0 = 0.45, j0 = 0.75, linear) corresponds to, and what the real curve does to
the lag-free thrust-speed slope that shortens the surge time constant.

Run: python -m studies.exp_propeller
"""
import os

import numpy as np

from sim.propeller import WageningenB

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NMRI = os.path.join(HERE, "data", "external", "kvlcc2_propeller",
                    "KVLCC_POT_NMRI.txt")
HMRI = os.path.join(HERE, "data", "external", "kvlcc2_propeller",
                    "KVLCC2-Propeller-Openwater-Data-HMRI.pdf")
KV = dict(Z=4, AE_A0=0.431, P_D=0.7212)


def load_nmri():
    rows = []
    started = False
    for line in open(NMRI, encoding="utf-8", errors="replace"):
        r = line.split()
        if r[:3] == ["J", "KT", "10KQ"]:
            started = True
            continue
        if started and len(r) == 3:
            try:
                rows.append([float(x) for x in r])
            except ValueError:
                pass
    d = np.array(rows)
    return d[:, 0], d[:, 1], d[:, 2] / 10.0


def load_hmri():
    """HMRI table. Columns are 10 K_T and 100 K_Q (model), inferred from the
    magnitudes and checked against NMRI below rather than assumed."""
    import pypdf
    text = pypdf.PdfReader(HMRI).pages[0].extract_text()
    rows = []
    for line in text.splitlines():
        r = line.split()
        if len(r) == 5:
            try:
                rows.append([float(x) for x in r])
            except ValueError:
                pass
    d = np.array(rows)
    return d[:, 0], d[:, 1] / 10.0, d[:, 2] / 100.0


def main():
    print("\nEXPERIMENT -- propeller open water, KVLCC2 propeller, "
          "B-series vs two measurements\n")
    Jn, KTn, KQn = load_nmri()
    Jh, KTh, KQh = load_hmri()
    # unit check for the HMRI columns: must agree with NMRI to a few percent
    kt0_h, kt0_n = KTh[0], KTn[0]
    print(f"  HMRI column scaling check: K_T(0) HMRI {kt0_h:.4f} vs NMRI "
          f"{kt0_n:.4f}; K_Q(0) {KQh[0]:.4f} vs {KQn[0]:.4f}")

    p = WageningenB(KV["Z"], KV["AE_A0"], KV["P_D"], D=0.0896)

    def j0(J, KT):
        i = int(np.argmax(KT < 0)) if np.any(KT < 0) else len(KT) - 1
        return float(np.interp(0.0, [KT[i], KT[i - 1]], [J[i], J[i - 1]]))

    print(f"\n  {'':<24}{'K_T(0)':>8}{'10K_Q(0)':>10}{'J(K_T=0)':>10}"
          f"{'eta0 max':>10}")
    for name, J, KT, KQ in (("measured, NMRI", Jn, KTn, KQn),
                            ("measured, HMRI", Jh, KTh, KQh)):
        eta = J * KT / (2 * np.pi * np.maximum(KQ, 1e-9))
        print(f"  {name:<24}{KT[0]:>8.4f}{10*KQ[0]:>10.4f}{j0(J, KT):>10.3f}"
              f"{eta[KT > 0].max():>10.3f}")
    Jg = np.linspace(0, 0.8, 161)
    eta_b = p.eta0(Jg)
    print(f"  {'B-series (Z4, 0.431, 0.72)':<24}{p.kt(0.0):>8.4f}"
          f"{10*p.kq(0.0):>10.4f}{p.j_zero_thrust():>10.3f}"
          f"{eta_b[p.kt(Jg) > 0].max():>10.3f}")

    print("\n  point by point, over the thrust-producing range:")
    print(f"  {'J':>6}{'K_T NMRI':>10}{'K_T HMRI':>10}{'K_T B-ser':>11}"
          f"{'err vs mean':>13}")
    errs = []
    for J in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        a, b = np.interp(J, Jn, KTn), np.interp(J, Jh, KTh)
        m = 0.5 * (a + b)
        e = (p.kt(J) - m) / m
        errs.append(e)
        print(f"  {J:>6.2f}{a:>10.4f}{b:>10.4f}{p.kt(J):>11.4f}{e:>12.1%}")
    spread = np.array([abs(np.interp(J, Jn, KTn) - np.interp(J, Jh, KTh))
                       / (0.5 * (np.interp(J, Jn, KTn) + np.interp(J, Jh, KTh)))
                       for J in (0.0, 0.2, 0.4, 0.6)])
    print(f"\n  B-series error vs the mean of the two measurements: "
          f"{np.mean(np.abs(errs[:-1])):.1%} mean |error| for J <= 0.6")
    print(f"  spread between the two measurements themselves:     "
          f"{np.mean(spread):.1%}")

    # ---- what the old placeholder corresponds to ---------------------------
    print("\n  the old USV placeholder: K_T = 0.45 (1 - J/0.75), linear")
    print(f"  {'propeller':<22}{'K_T(0)':>8}{'J(K_T=0)':>10}"
          f"{'dK_T/dJ @0.3':>14}")
    for Z, ae, pd in ((3, 0.50, 0.9), (4, 0.55, 0.9), (4, 0.55, 1.0),
                      (4, 0.70, 1.0), (4, 0.55, 1.1)):
        q = WageningenB(Z, ae, pd, 0.44)
        slope = (q.kt(0.31) - q.kt(0.29)) / 0.02
        print(f"  B{Z}-{int(ae*100)}, P/D {pd:<9}{q.kt(0.0):>8.3f}"
              f"{q.j_zero_thrust():>10.3f}{slope:>14.3f}")
    print(f"  {'placeholder':<22}{0.45:>8.3f}{0.75:>10.3f}{-0.45/0.75:>14.3f}")
    print("\n  A real propeller with the placeholder's zero-thrust point has a")
    print("  bollard K_T well below 0.45: the placeholder's two numbers do not")
    print("  come from the same propeller. The surge-damping slope it implies")
    print("  is what matters for tau_u, and that is compared in the table.")


if __name__ == "__main__":
    main()
