#!/usr/bin/env python3
"""
Companion to studies/anyspeed_audit.py: transom draught and transom Froude
number Fn_T = U / sqrt(g d_T) along the plant's calm running attitude, the
plant's a_b(u), and Savitsky (hydro/planing.savitsky) where it applies.
Output: studies/_cache/anyspeed/audit_transom.txt
"""
import os
import numpy as np
from studies.anyspeed_audit import build, KN, G, OUT
from hydro.planing import savitsky
from sim.planing_vessel import SP


def main():
    h, p = build()
    L = []
    L.append("kn  C_V   Fn_L  a_b   trim  d_T m  Fn_T   R_plant  | Savitsky trim  R  valid")
    for kn in (1, 3, 5, 7, 9, 10, 11, 12, 13, 14, 16, 18, 20, 25, 30, 35):
        u = kn * KN
        z, th = p.running_attitude(u)
        Th = p.Th_rest + th
        xT = p.x[0] - 0.5 * p.dx
        dT = -(p.Z_rest + z + p.zk[0] + SP * xT * Th)
        r = p.calm_resistance(u)
        cv = u / np.sqrt(G * h.b)
        try:
            s = savitsky(u, h.mass, h.lcg, h.b, h.beta_deg)
            sv = f"{s['tau_deg']:6.2f} {s['R']:6.0f}  {s['valid']}"
        except Exception as e:      # outside its root brackets
            sv = "  --"
        L.append(f"{kn:3d} {cv:5.2f} {u / np.sqrt(G * h.L):5.2f} "
                 f"{p._a_b(u):4.2f} {np.degrees(SP * Th):5.2f} {dT:6.3f} "
                 f"{u / np.sqrt(G * max(dT, 1e-3)):5.2f}  {r['R'] - r['air']:7.0f}"
                 f"  | {sv}")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(OUT, "audit_transom.txt"), "w") as fh:
        fh.write("(1-9 kn: running_attitude is a snapshot of a limit cycle, "
                 "not an equilibrium)\n" + txt + "\n")


if __name__ == "__main__":
    main()
