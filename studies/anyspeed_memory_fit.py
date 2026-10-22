#!/usr/bin/env python3
"""
Probe (design phase): does the repo's radiation-memory machinery
(hydro/retardation.fit_coefficient, sim/cummins.RadiationMemory) fit the
zero-speed BEM coefficients of scarab195 at rest (from
studies/anyspeed_bem_probe.py)? Reports per-pair order, K error, band errors,
the A_inf spread and how much of B is left at the top of the grid.

Run: python -m studies.anyspeed_memory_fit [npz]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_cache", "anyspeed")
DOF = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]


def main(npz="bem_probe_lid_24_64.npz", w_cut=None):
    from hydro.retardation import fit_coefficient
    z = np.load(os.path.join(OUT, npz))
    om, A, B = z["omega"], z["A"], z["B"]
    if w_cut is not None:
        keep = om <= w_cut
        om, A, B = om[keep], A[keep], B[keep]
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    say(f"{npz}: {len(om)} frequencies {om[0]:.2f}..{om[-1]:.2f} rad/s")
    say(f"  {'pair':<12}{'order':>6}{'err_K':>8}{'err_B':>8}{'err_A':>8}"
        f"{'A_inf':>9}{'spread':>8}{'B_top/max':>10}{'mem s':>7}")
    t0 = time.time()
    bmax = np.max(np.abs(B))
    for i, j in ((2, 2), (4, 4), (2, 4), (0, 0), (0, 2), (0, 4), (1, 1),
                 (3, 3), (5, 5), (1, 3), (1, 5), (3, 5)):
        if np.max(np.abs(B[:, i, j])) < 1e-3 * bmax:
            continue
        f = fit_coefficient(om, A[:, i, j], B[:, i, j])
        say(f"  {DOF[i][:5] + '-' + DOF[j][:5]:<12}{f['order']:>6}"
            f"{f['err_K']:>8.3f}{f['err_B']:>8.3f}{f['err_A']:>8.3f}"
            f"{f['A_inf']:>9.0f}{f['A_inf_spread']:>8.3f}"
            f"{f['tail_at_wmax']:>10.3f}{f['memory_time']:>7.2f}")
    say(f"  ({time.time() - t0:.1f} s)")
    tag = "" if w_cut is None else f"_cut{w_cut:g}"
    with open(os.path.join(OUT, f"memory_fit{tag}.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "bem_probe_lid_24_64.npz",
         float(a[1]) if len(a) > 1 else None)
