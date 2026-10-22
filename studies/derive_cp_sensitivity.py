#!/usr/bin/env python3
"""Trim and resistance change of the steady planing equilibrium when the
model's centre of pressure is displaced by delta (m) relative to the truth,
evaluated with Savitsky's equilibrium (equivalent to moving the LCG by
-delta). Also: resistance change per degree of trim. Output to
studies/_cache/derive/cp_sensitivity.txt."""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hydro.planing import savitsky  # noqa: E402

KN = 0.514444
B, M, LCG, BETA = 0.83 * 2.44, 1400.0, 0.38 * 5.74, 20.0
SLENDER_DX = {12: 0.075, 16: 0.057, 20: 0.034, 25: 0.003, 30: -0.035,
              35: -0.097, 40: -0.129}     # dxcp/b from slender.txt
lines = ["kn  tau  R  | dtau/dlcg deg/m  dR/dlcg %/m | slender CP error "
         "delta m -> dtau deg, dR %  | W/R dR%/deg"]
for kn in [12, 16, 20, 25, 30, 35, 40]:
    U = kn * KN
    s0 = savitsky(U, M, LCG, B, BETA)
    h = 0.02
    sp = savitsky(U, M, LCG + h, B, BETA)
    sm = savitsky(U, M, LCG - h, B, BETA)
    dt = (sp["tau_deg"] - sm["tau_deg"]) / (2 * h)
    dR = (sp["R"] - sm["R"]) / (2 * h) / s0["R"] * 100
    d = SLENDER_DX[kn] * B
    # strip CP too far forward by -d (exact aft by d): same as LCG moved by d
    # dxcp > 0: exact CP nearer the transom than slender, i.e. the slender
    # (2D+t) CP is forward by d; its equilibrium = a true boat with the LCG
    # moved aft (towards the transom) by d
    s1 = savitsky(U, M, LCG - d, B, BETA)
    W = M * 9.81
    dRdtau = W / math.cos(math.radians(s0["tau_deg"])) ** 2 \
        * math.pi / 180 / s0["R"] * 100
    lines.append(f"{kn:3d} {s0['tau_deg']:.2f} {s0['R']:.0f} | {dt:+.2f} "
                 f"{dR:+.1f} | {d:+.3f} -> {s1['tau_deg']-s0['tau_deg']:+.2f}"
                 f" {100*(s1['R']/s0['R']-1):+.1f} | {dRdtau:.1f}")
txt = "\n".join(lines)
os.makedirs(os.path.join(ROOT, "studies", "_cache", "derive"), exist_ok=True)
open(os.path.join(ROOT, "studies", "_cache", "derive", "cp_sensitivity.txt"),
     "w").write(txt + "\n")
print(txt)


# ---- combined prediction: the strip's gravity-free linear 3-D error alone
# model lift = k x true lift (k = Lj/L_exact) and CP forward by d:
# model equilibrium == a true boat of mass M/k with LCG moved aft by d.
sys.path.insert(0, os.path.join(ROOT, "studies"))
from derive_slender import hull_case  # noqa: E402

G = 9.81


def predict(U, M, LCG, B, BETA, rho=1025.0):
    s0 = savitsky(U, M, LCG, B, BETA, rho=rho)
    q = hull_case(s0["lam"], s0["tau_deg"], BETA)
    k = q["Lj"] / q["L"]
    d = (q["xcp_over_LK"] - q["xj_over_LK"]) * q["LK"] * B
    s1 = savitsky(U, M / k, LCG - d, B, BETA, rho=rho)
    # model resistance at its own equilibrium: its lift equals W, so the
    # pressure drag is W tan(tau'); friction from Savitsky's law at lam'
    W = M * G
    R1 = W * math.tan(math.radians(s1["tau_deg"])) \
        + s1["D_f"] / math.cos(math.radians(s1["tau_deg"]))
    return s0, k, d, s1, R1


lines2 = ["", "strip (2D+t) error from gravity-free linear 3-D effects alone,"
          " predicted with Savitsky's equilibrium:",
          "case  tau_true  k=L_slender/L_3D  d(m)  ->  dtau(deg)  dR(%)"]
for kn in [12, 16, 20, 25, 30, 35, 40]:
    s0, k, d, s1, R1 = predict(kn * KN, M, LCG, B, BETA)
    lines2.append(f"{kn:3d} kn {s0['tau_deg']:.2f} {k:.3f} {d:+.3f} -> "
                  f"{s1['tau_deg']-s0['tau_deg']:+.2f} "
                  f"{100*(R1/s0['R']-1):+.1f}")
bF = 9 * 0.0254
LF = 5 * bF
for case, vl, aft, cd in [("Fridsma A", 4, 0.59, 0.608),
                          ("Fridsma B", 6, 0.62, 0.608)]:
    U = vl * KN * math.sqrt(LF / 0.3048)
    mF = cd * 62.4 * (bF / 0.3048) ** 3 * 0.45359237
    s0, k, d, s1, R1 = predict(U, mF, (1 - aft) * LF, bF, 20.0, rho=1000.0)
    lines2.append(f"{case}: tau_Sav {s0['tau_deg']:.2f} lam {s0['lam']:.2f} "
                  f"k {k:.3f} d {d/bF:+.3f} b -> dtau "
                  f"{s1['tau_deg']-s0['tau_deg']:+.2f} dR "
                  f"{100*(R1/s0['R']-1):+.1f} %")
txt2 = "\n".join(lines2)
open(os.path.join(ROOT, "studies", "_cache", "derive", "cp_sensitivity.txt"),
     "a").write(txt2 + "\n")
print(txt2)
