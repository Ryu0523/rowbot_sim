#!/usr/bin/env python3
"""
Rewrite the stored inertia matrix with the corrected one.

`hydro/inertia.py` establishes, against exact mathematics, that the rotational
block of `db.M` is low by 22.6% (roll), 16.7% (pitch) and 17.1% (yaw). This
applies the fix to the saved database, keeps a backup, and says out loud what
downstream work it invalidates -- because the mass matrix is upstream of
everything.

Run: python -m hydro.fix_inertia_db [--apply]
"""
import argparse
import shutil

import numpy as np

from hydro import bem
from hydro.inertia import wigley_inertia, verify

PATH = "hydro_wigley_10m.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--apply", action="store_true",
                    help="write the file; without it, only report")
    a = ap.parse_args()

    print("  re-checking against exact mathematics before touching anything\n")
    worst = verify(verbose=False)
    if worst > 0.01:
        raise SystemExit(f"  external verification failed ({worst:.2%}); "
                         f"refusing to write")
    print(f"  external checks pass, worst error {worst:.2%}\n")

    db = bem.load(a.path)
    L = db.L
    B, T = 2.5, 0.8                    # the hull these coefficients are for
    Ma, _ = wigley_inertia(L, B, T, -T / 3)

    if abs(db.M[0, 0] - Ma[0, 0]) / Ma[0, 0] > 0.01:
        raise SystemExit(f"  displaced mass disagrees ({db.M[0,0]:.0f} vs "
                         f"{Ma[0,0]:.0f}); this is not the hull this fix is for")

    M_new = db.M.copy()
    print(f"  {'dof':<8}{'stored':>12}{'corrected':>12}{'change':>10}")
    for i, nm in ((3, "roll"), (4, "pitch"), (5, "yaw")):
        print(f"  {nm:<8}{db.M[i,i]:>12.0f}{Ma[i,i]:>12.0f}"
              f"{(Ma[i,i]-db.M[i,i])/db.M[i,i]:>9.1%}")
        M_new[i, i] = Ma[i, i]
    # the off-diagonal rotational couplings are zero by symmetry for this hull
    # and Capytaine agrees; left alone rather than overwritten with a zero we
    # would then have to justify separately
    print(f"\n  translational block and all hydrostatics unchanged -- both were")
    print(f"  already verified against the analytic Wigley formulae to 0.2%.")

    if not a.apply:
        print(f"\n  dry run. Re-run with --apply to write {a.path}.")
        return

    shutil.copy2(a.path, a.path + ".pre_inertia_fix")
    db.M = M_new
    bem.save(db, a.path)
    print(f"\n  written. Backup at {a.path}.pre_inertia_fix")
    print(f"\n  THIS INVALIDATES, because they were fitted or trained against")
    print(f"  the old mass matrix and none of them will notice on their own:")
    for s in ("the reduced model's identified coefficients (control/reduced.py "
              "re-identifies on construction, so it self-heals)",
              "ppo_usv.zip, ppo_bc.zip, bc_clone.zip and their VecNormalize "
              "statistics -- retrain",
              "bc_demos.npz -- the MPC demonstrations were generated under the "
              "old plant; delete and recollect",
              "every natural period quoted in the reports (heave unchanged, "
              "roll 1.56 -> 1.68 s, pitch 1.86 -> 1.96 s)"):
        print(f"    - {s}")


if __name__ == "__main__":
    main()
