#!/usr/bin/env python3
"""
Unit test for the M2 machinery, against exact ground truth.

Build a state-space system by hand, run it FORWARD to get A(w) and B(w)
analytically, then feed those to the pipeline and demand it recover the
system we started from. Everything is exercised -- the cosine transform,
the A_inf relation, the Hankel realisation -- with no hydrodynamics
involved, so a failure here is a code bug and never a physics question.

Ground truth: with block-diagonal Ar built from poles -sigma +/- j*w,
Br = [1,0,...] and Cr = [c,0,...],

    K(t) = sum_i c_i exp(-sigma_i t) cos(w_i t)
"""
import numpy as np
from .retardation import fit_coefficient, transfer_function

POLES = [(0.5, 2.0, 8000.0), (1.5, 4.5, 3000.0)]   # (sigma, omega, weight)
A_INF_TRUE = 5000.0


def build_system():
    blocks, Br, Cr = [], [], []
    for sigma, w, c in POLES:
        blocks.append(np.array([[-sigma, w], [-w, -sigma]]))
        Br += [1.0, 0.0]
        Cr += [c, 0.0]
    n = 2 * len(POLES)
    Ar = np.zeros((n, n))
    for i, blk in enumerate(blocks):
        Ar[2 * i:2 * i + 2, 2 * i:2 * i + 2] = blk
    return Ar, np.array(Br), np.array(Cr)


def exact_K(t):
    return sum(c * np.exp(-s * t) * np.cos(w * t) for s, w, c in POLES)


def main():
    Ar, Br, Cr = build_system()
    omegas = np.arange(0.05, 250.0 + 1e-9, 0.05)     # wide: the transform
    H = transfer_function(omegas, Ar, Br, Cr)        # needs the tail of B
    B = np.real(H)
    A = A_INF_TRUE + np.imag(H) / omegas   # Ogilvie, see retardation.py

    print("synthetic system")
    print(f"  poles: {[(-s, w) for s, w, _ in POLES]}")
    print(f"  A_inf true = {A_INF_TRUE}")
    print(f"  omega grid {omegas[0]:.2f}..{omegas[-1]:.1f}, "
          f"d_omega={omegas[1]-omegas[0]:.3f}\n")

    print(f"  {'asked':>7}{'used':>6}{'A_inf':>11}{'A_inf err%':>12}"
          f"{'errK%':>9}{'errB%':>9}{'errA%':>9}{'K vs exact%':>13}")
    results = {}
    for order in (2, 4, 6, "auto"):
        f = fit_coefficient(omegas, A, B, order=order)
        Ke = exact_K(f["t"])
        err_exact = np.max(np.abs(f["K"] - Ke)) / np.max(np.abs(Ke))
        results[order] = (f, err_exact)
        print(f"  {str(order):>7}{len(f['Ar']):>6}{f['A_inf']:>11.1f}"
              f"{100*(f['A_inf']-A_INF_TRUE)/A_INF_TRUE:>12.2f}"
              f"{100*f['err_K']:>9.3f}{100*f['err_B']:>9.3f}"
              f"{100*f['err_A']:>9.3f}{100*err_exact:>13.3f}")

    f, err_exact = results["auto"]
    poles = np.sort_complex(np.linalg.eigvals(f["Ar"]))
    true_poles = np.sort_complex(np.array(
        [-s + 1j * w for s, w, _ in POLES] + [-s - 1j * w for s, w, _ in POLES]))
    pole_err = (np.max(np.abs(poles - true_poles)) / np.max(np.abs(true_poles))
                if len(poles) == len(true_poles) else float("nan"))
    print(f"\n  auto-selected order {f['order']} (true system is order 4)")
    print(f"  recovered poles: {np.round(poles, 3)}")
    print(f"  max pole error : {100*pole_err:.3f}%")

    checks = [
        ("cosine transform reproduces exact K(t) within 1%", err_exact < 0.01),
        ("A_inf recovered within 1%",
         abs(f["A_inf"] - A_INF_TRUE) / A_INF_TRUE < 0.01),
        ("realisation reduces to order 4", len(f["Ar"]) == 4),
        ("poles recovered within 2%", pole_err < 0.02),
        ("realisation matches K within 1%", f["err_K"] < 0.01),
        ("B(w) round trip within 2%", f["err_B"] < 0.02),
        ("A(w) round trip within 2%", f["err_A"] < 0.02),
        ("poles stable", f["max_real_eig"] < 0),
    ]
    print("\n  " + "-" * 50)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    print("  " + "-" * 50)
    ok = all(o for _, o in checks)
    print(f"  M2 UNIT TEST: {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    main()
