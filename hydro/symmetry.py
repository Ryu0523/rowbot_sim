#!/usr/bin/env python3
"""
Impose the port/starboard symmetry the hull actually has.

A hull that is symmetric about its centreplane cannot couple the VERTICAL group
(surge, heave, pitch) to the LATERAL group (sway, roll, yaw). Those eighteen
entries of every 6x6 hydrodynamic matrix are exactly zero, as a matter of
geometry, not approximately zero.

A panel mesh is not exactly symmetric, so the BEM returns them as small numbers
instead. That was harmless for years and then stopped being harmless, in a way
worth recording because it is a general trap:

    A_inf[2,5] (heave-yaw)  =  12.8 kg m
    A_inf[4,1] (pitch-sway) = -11.3 kg m
    C[i,j] cross terms up to 34 N/m

Small against diagonals of 10^4. But the added-mass Coriolis term multiplies
them by velocity SQUARED, and in head seas at 4 m/s the residual A_inf[1,0]
injects a yaw moment of about 9 N m with the vessel travelling perfectly
straight. Over two minutes that grew into 2.9 m of lateral drift and 0.6 deg of
heading error in a case where both must be zero, and it broke two symmetry
checks that had passed for the whole life of the project.

> A number small enough to ignore in the matrix it lives in is not necessarily
> small in the term that uses it. What matters is the size of its CONTRIBUTION,
> and that depends on what multiplies it.

The residual is not noise to be tolerated once it is being amplified: it is a
known-zero being carried as data. Zero it, and report how much was removed so
the decision stays visible.

This is also part of making the model hull-agnostic. A real vessel imported
later will have its own mesh with its own asymmetry, and `symmetric=False` is
there for a hull that genuinely is not symmetric -- a single-screw vessel with
an offset skeg, say -- where the entries are physics rather than noise.

Run: python -m hydro.symmetry
"""
import numpy as np

VERTICAL = (0, 2, 4)        # surge, heave, pitch
LATERAL = (1, 3, 5)         # sway, roll, yaw
DOF = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def cross_mask():
    """True where a port/starboard symmetric hull must have exactly zero."""
    m = np.zeros((6, 6), bool)
    m[np.ix_(VERTICAL, LATERAL)] = True
    m[np.ix_(LATERAL, VERTICAL)] = True
    return m


def residual(M):
    """Largest symmetry-violating entry, and where it is."""
    M = np.asarray(M, float)
    masked = np.where(cross_mask(), np.abs(M), 0.0)
    i, j = np.unravel_index(masked.argmax(), masked.shape)
    scale = float(np.max(np.abs(np.diag(M))))
    return float(masked[i, j]), (int(i), int(j)), scale


def enforce(M, name="", verbose=False):
    """Return a copy with the cross-group block set to exactly zero."""
    M = np.array(M, float, copy=True)
    worst, (i, j), scale = residual(M)
    M[cross_mask()] = 0.0
    if verbose and worst > 0:
        print(f"    {name:<8} removed {worst:.4g} at "
              f"{DOF[i]}-{DOF[j]} ({worst/max(scale,1e-30):.1e} of diagonal)")
    return M


def enforce_all(db, verbose=False):
    """Apply it to every matrix in a hydrodynamic database, in place.

    A(w) and B(w) are done frequency by frequency, because the excitation and
    radiation solutions each carry their own mesh asymmetry.
    """
    if verbose:
        print("  imposing port/starboard symmetry:")
    db.M = enforce(db.M, "M", verbose)
    db.C = enforce(db.C, "C", verbose)
    mask = cross_mask()
    for k in range(db.A.shape[0]):
        db.A[k][mask] = 0.0
        db.B[k][mask] = 0.0
    return db


def main():
    from hydro import bem
    from sim.cummins import radiation_memory
    db = bem.load("hydro_wigley_10m.npz")
    A_inf = radiation_memory(db).A_inf

    print("\n  entries that geometry says are zero, and what the BEM returns\n")
    print(f"  {'matrix':<8}{'worst residual':>16}{'where':>18}"
          f"{'vs diagonal':>14}")
    for nm, M in (("M", db.M), ("C", db.C), ("A_inf", A_inf),
                  ("A(w=0)", db.A[0]), ("B(peak)", db.B[len(db.B) // 3])):
        w, (i, j), sc = residual(M)
        print(f"  {nm:<8}{w:>16.4g}{DOF[i] + '-' + DOF[j]:>18}"
              f"{w/max(sc,1e-30):>14.1e}")

    u = 4.0
    print(f"\n  why it matters: the added-mass Coriolis term multiplies these")
    print(f"  by velocity squared. At u = {u} m/s, travelling perfectly")
    print(f"  straight with v = p = r = 0, the residual A_inf[sway,surge] "
          f"= {A_inf[1,0]:.2f}")
    print(f"  injects a yaw moment of {abs(A_inf[1,0])*u*u:.1f} N m onto a hull")
    print(f"  that must not be turning at all.")
    print(f"\n  measured consequence before this fix: 2.86 m of lateral drift")
    print(f"  and 0.6 deg of heading error over 120 s of head seas, breaking")
    print(f"  two symmetry checks in studies/audit_physics.py.")


if __name__ == "__main__":
    main()
