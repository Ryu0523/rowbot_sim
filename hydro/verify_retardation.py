#!/usr/bin/env python3
"""
M2 GATE -- does the state-space model really stand in for the convolution?

The test is a round trip. Go frequency -> time -> state space -> frequency and
demand the answer come back:

  1. the realisation's impulse response reproduces K(t)
  2. B(w) rebuilt from the realisation matches the BEM
  3. A(w) rebuilt from the realisation matches the BEM
  4. every pole is stable
  5. A_inf is consistent across the frequency grid

Round-tripping catches errors a one-way check cannot: a wrong transform, a
truncated integral, or an under-ordered fit all break the return leg. The
machinery itself is verified separately against an exact system in
test_retardation_synthetic.py, so a failure here points at the hydrodynamics
or the frequency grid rather than at the code.

It also produces a number the control design needs directly -- the memory
length of K(t), which sets how far back the hull remembers its own motion.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import bem
from .retardation import fit_coefficient

DB = "hydro_wigley_10m.npz"
DOFS = ["Surge", "Heave", "Roll", "Pitch"]


def run(path=DB, order="auto", dofs=DOFS):
    db = bem.load(path)
    w = db.omega
    print(f"database: {path}")
    print(f"  {db}")
    print(f"  d_omega = {np.mean(np.diff(w)):.3f} rad/s  ->  "
          f"t_max = {np.pi/np.mean(np.diff(w)):.0f} s")
    print(f"  hull {db.attrs.get('hull')} L={db.attrs.get('L')} "
          f"B={db.attrs.get('B')} T={db.attrs.get('T')}\n")

    fits = {}
    print(f"  {'dof':>7}{'ord':>5}{'A_inf':>12}{'sprd%':>7}{'errK%':>7}"
          f"{'errB%':>7}{'errA%':>7}{'bandB%':>8}{'bandA%':>8}"
          f"{'B tail':>8}{'mem s':>7}")
    for dof in dofs:
        f = fit_coefficient(w, db.a(dof), db.b(dof), order=order)
        f["A"], f["B"] = db.a(dof), db.b(dof)
        fits[dof] = f
        print(f"  {dof:>7}{len(f['Ar']):>5}{f['A_inf']:>12.1f}"
              f"{100*f['A_inf_spread']:>7.2f}{100*f['err_K']:>7.2f}"
              f"{100*f['err_B']:>7.2f}{100*f['err_A']:>7.2f}"
              f"{100*f['err_B_band']:>8.2f}{100*f['err_A_band']:>8.2f}"
              f"{100*f['tail_at_wmax']:>7.1f}%{f['memory_time']:>7.2f}")

    _order_sweep(db, dofs)
    _plot(w, fits)
    _verdict(fits)
    return fits


def _order_sweep(db, dofs, orders=(2, 4, 6, 8, "auto")):
    print(f"\n  state-space order sweep (worst over dofs, %)")
    print(f"  {'asked':>7}{'errK':>8}{'errB':>8}{'errA':>8}")
    for o in orders:
        ek = eb = ea = 0.0
        for dof in dofs:
            try:
                f = fit_coefficient(db.omega, db.a(dof), db.b(dof), order=o)
            except Exception:
                ek = eb = ea = float("nan")
                break
            ek = max(ek, f["err_K"]); eb = max(eb, f["err_B"])
            ea = max(ea, f["err_A"])
        print(f"  {str(o):>7}{100*ek:>8.2f}{100*eb:>8.2f}{100*ea:>8.2f}")


def _verdict(fits):
    checks = []
    for dof, f in fits.items():
        checks += [
            (f"{dof}: K(t) round trip  < 3%", f["err_K"] < 0.03),
            (f"{dof}: B(w) round trip  < 4%", f["err_B"] < 0.04),
            (f"{dof}: A(w) round trip  < 6%", f["err_A"] < 0.06),
            (f"{dof}: B in wave band   < 1%", f["err_B_band"] < 0.01),
            (f"{dof}: A in wave band   < 2%", f["err_A_band"] < 0.02),
            (f"{dof}: poles stable", f["max_real_eig"] < 0),
            (f"{dof}: A_inf spread   < 8%", f["A_inf_spread"] < 0.08),
        ]
    bad = [n for n, ok in checks if not ok]
    print("\n  " + "-" * 46)
    for n in bad:
        print(f"  [FAIL]  {n}")
    print(f"  {len(checks)-len(bad)}/{len(checks)} checks passed")
    print(f"  M2 RETARDATION GATE: {'PASSED' if not bad else 'FAILED'}")
    print("  " + "-" * 46)
    if not bad:
        mem = {d: f["memory_time"] for d, f in fits.items()}
        print(f"\n  hydrodynamic memory: "
              + ", ".join(f"{d} {v:.1f}s" for d, v in mem.items()))
        print(f"  -> the hull remembers its own motion for "
              f"{min(mem.values()):.1f}-{max(mem.values()):.1f} s")


def _plot(omegas, fits):
    n = len(fits)
    fig, ax = plt.subplots(3, n, figsize=(3.6 * n, 9))
    for j, (dof, f) in enumerate(fits.items()):
        ax[0, j].plot(f["t"], f["K"], "k-", lw=2.2, label="$K(t)$ from BEM")
        ax[0, j].plot(f["t"], f["K_ss"], "--", c="tab:red", lw=1.6,
                      label=f"state space (order {len(f['Ar'])})")
        ax[0, j].axvline(f["memory_time"], ls=":", c="tab:green", lw=1.4)
        ax[0, j].annotate(f"memory {f['memory_time']:.1f} s",
                          (f["memory_time"], 0.55 * np.max(np.abs(f["K"]))),
                          fontsize=8, color="tab:green")
        ax[0, j].axhline(0, c="k", lw=.5)
        ax[0, j].set_xlim(0, min(12, f["t"][-1]))
        ax[0, j].set_xlabel("t (s)")
        ax[0, j].set_title(f"{dof}: retardation $K(t)$", fontsize=10)

        ax[1, j].plot(omegas, f["B"], "k-", lw=2.2, label="BEM")
        ax[1, j].plot(omegas, f["B_fit"], "--", c="tab:red", lw=1.6,
                      label="rebuilt")
        ax[1, j].set_title(f"{dof}: $B(\\omega)$ round trip", fontsize=10)

        ax[2, j].plot(omegas, f["A"], "k-", lw=2.2, label="BEM")
        ax[2, j].plot(omegas, f["A_fit"], "--", c="tab:red", lw=1.6,
                      label="rebuilt")
        ax[2, j].axhline(f["A_inf"], ls=":", c="tab:blue", lw=1.4,
                         label="$A_\\infty$")
        ax[2, j].set_title(f"{dof}: $A(\\omega)$ round trip", fontsize=10)

        for i in (1, 2):
            ax[i, j].set_xlabel("$\\omega$ (rad/s)")
            ax[i, j].set_xlim(0, 8)
            ax[i, j].axvspan(0.4, 2.0, color="tab:green", alpha=.08)
        for i in range(3):
            ax[i, j].grid(alpha=.3)
            ax[i, j].legend(fontsize=7)
    fig.suptitle("M2 gate: frequency -> time -> state space -> frequency "
                 "round trip (Wigley 10 m; green band = SS4-5)", fontsize=12)
    fig.tight_layout()
    fig.savefig("fig9_m2_retardation.png", dpi=140)
    print("\n  figure: fig9_m2_retardation.png")


if __name__ == "__main__":
    run()
