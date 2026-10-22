#!/usr/bin/env python3
"""
Literature thresholds applied to this boat, by arithmetic only (no plant run).

Puts numbers on the regimes the literature review names, for the 5.4 m boat
(sim/env.py:122) in the evaluation sea (Hs 1.0 m, Tp 5 s):

  * IMO MSC.1/Circ.1228 (2007) para 4.2.1: surf-riding / broaching may occur
    for 135 < alpha < 225 deg when V [kn] > 1.8 sqrt(L) / cos(180 - alpha).
  * IMO MSC.1/Circ.1228 para 4.2.2: successive high-wave attack when
    lambda > 0.8 L, Hs > 0.04 L and T_E is about 1.8-3.0 T_W.
  * IMO MSC.1/Circ.1627 (2020) 2.6.2.1: Level 1 says "not vulnerable" to
    surf-riding/broaching if Fn <= 0.3; Level 2 looks at local waves with
    lambda/L from 1.0 to 3.0 and steepness 0.03-0.15 (2.6.3.3).
  * Encounter period of the peak-period wave at each speed and heading, and
    how much of a sinusoidal thrust command a first-order lag of 0.6 s plus a
    0.1 s delay passes (the waterjet defaults, sim/planing_vessel.py:131-133;
    rate limit, priming and the pump curve are ignored here, so this is an
    upper bound on what reaches the jet).

Deep-water linear dispersion throughout; the dominant wave only, so this says
where the regimes sit, not what an irregular sea does.

Run:  .venv/Scripts/python.exe studies/wave_regime_lit_thresholds.py
Out:  studies/_cache/wave_regimes/lit_thresholds.txt
"""
import os
import numpy as np

G, KN = 9.81, 0.514444
L = 5.4                   # sim/env.py:122 "5.4 m boat"
HS, TP = 1.0, 5.0         # evaluation sea
TAU, DELAY = 0.6, 0.1     # sim/planing_vessel.py:131-133 defaults

OUT = os.path.join(os.path.dirname(__file__), "_cache", "wave_regimes",
                   "lit_thresholds.txt")


def wave(T=None, lam=None):
    if lam is None:
        lam = G * T ** 2 / (2 * np.pi)
    T = np.sqrt(2 * np.pi * lam / G)
    return lam, T, lam / T


def te(lam, c, U, mu_deg):
    """Encounter period; mu = 0 following, 180 head (angle of wave travel
    relative to the boat's heading)."""
    v = c - U * np.cos(np.radians(mu_deg))
    return lam / abs(v) if abs(v) > 1e-9 else np.inf


def lag(T):
    if not np.isfinite(T):
        return 1.0, 0.0
    w = 2 * np.pi / T
    return 1 / np.sqrt(1 + (w * TAU) ** 2), (np.arctan(w * TAU) + w * DELAY) / (2 * np.pi)


def main():
    lines = []
    p = lines.append
    lam, T, c = wave(T=TP)
    p(f"boat L = {L} m; sea Hs {HS} m, Tp {TP} s")
    p(f"peak wave: lambda {lam:.1f} m (lambda/L {lam / L:.1f}), celerity "
      f"{c:.2f} m/s = {c / KN:.1f} kn, Hs/lambda {HS / lam:.3f}")
    p("")
    p("IMO MSC.1/Circ.1228 4.2.1 surf-riding speed threshold "
      "1.8 sqrt(L)/cos(180-alpha):")
    for a in (180, 157.5, 135):
        p(f"  alpha {a:5.1f} deg -> {1.8 * np.sqrt(L) / np.cos(np.radians(180 - a)):.1f} kn")
    p(f"IMO MSC.1/Circ.1627 2.6.2.1 Level 1 Fn = 0.3 -> "
      f"{0.3 * np.sqrt(G * L) / KN:.1f} kn  (Fn = 1 -> {np.sqrt(G * L) / KN:.1f} kn)")
    p(f"Circ.1228 4.2.2 preconditions: lambda {lam:.1f} > 0.8L = {0.8 * L:.1f} m; "
      f"Hs {HS} > 0.04L = {0.04 * L:.2f} m")
    for r in (1.8, 3.0):     # T_E / T_W = 1 / (1 - U/c) in following seas
        p(f"  T_E = {r} T_W in following seas at U = {(1 - 1 / r) * c / KN:.1f} kn")
    p("Circ.1627 Level 2 local waves lambda/L 1.0-3.0:")
    for r in (1.0, 2.0, 3.0):
        l, t, cc = wave(lam=r * L)
        p(f"  lambda/L {r}: lambda {l:.1f} m, T {t:.2f} s, celerity {cc / KN:.1f} kn")
    p("")
    p("encounter period of the peak wave [s] and the share of a sinusoidal "
      "thrust command a 0.6 s lag + 0.1 s delay passes (gain / lag as a "
      "fraction of the period)")
    p(" U[kn]  head(180)  bow45(135)  beam(90)  stern45(45)  following(0)"
      "   gain/lag head   gain/lag following")
    for U in (0, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30):
        u = U * KN
        tt = [te(lam, c, u, m) for m in (180, 135, 90, 45, 0)]
        gh, lh = lag(tt[0])
        gf, lf = lag(tt[4])
        p(f" {U:5d}  " + "  ".join(f"{x:9.2f}" if np.isfinite(x) else "      inf"
                                    for x in tt)
          + f"     {gh:.2f}/{lh:.2f}        {gf:.2f}/{lf:.2f}")
    txt = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
