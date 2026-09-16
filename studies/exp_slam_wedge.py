#!/usr/bin/env python3
"""
Experiment: the slamming model against a measured drop test.

THE DATA
Zhao, Faltinsen and Aarsnes, "Water entry of arbitrary two-dimensional sections
with and without flow separation", 21st Symposium on Naval Hydrodynamics
(National Academy Press). Drop tests at MARINTEK of a 30 deg deadrise wedge with
knuckles: breadth 0.50 m, a 0.20 m force-measuring section between two 0.40 m
dummy sections, total drop-rig mass 241 kg. Their Figure 15: measured vertical
force on the measuring section (15a, solid line), measured drop velocity (15b).
Their fully nonlinear 2-D solution is the dashed line in 15a.

The paper gives these only as figures. The numbers below were read BY EYE off
the 256-pixel-wide scan on the NAP website; treat them as +-250 N and
+-0.05 m/s. They pass a check that does not depend on reading the force curve:
the rig's measured deceleration, times its mass, apportioned to the 0.2 m
section, lands within 6-11% of the digitised force (printed).

THE HULL
For this experiment the "hull" is the test section itself: a 30 deg wedge with
knuckles at half-breadth 0.25 m and vertical sides above, fed to the project's
own SlamLoad class through a one-station section table -- so what is tested is
the code path the vessel uses, not a re-derivation.

VARIANTS
  model now      generalised Wagner: pile-up and separation from the section's
                 own shape (SlamLoad._wagner). What the vessel uses.
  model before   Wagner's (pi/2)^2 on the force, but separation only when the
                 GEOMETRIC immersion reached the knuckle. What it used to use.
  von Karman     no pile-up at all.
  now + m_a dV/dt  the added-mass deceleration term SlamLoad omits.

Run: python -m studies.exp_slam_wedge
"""
import numpy as np

from sim.sections import SlamLoad

RHO = 1000.0             # fresh-water tank; the paper does not state it
G = 9.81
BETA = np.radians(30.0)
HALF_B = 0.25            # knuckle half-breadth, m
H_K = HALF_B * np.tan(BETA)   # keel-to-knuckle height from the drawing, 0.144 m
L_MEAS = 0.20
L_RIG, M_RIG = 1.00, 241.0

# ---- digitised from Figure 15 (by eye; +-250 N, +-0.05 m/s) ---------------
T_F = np.array([0.0, 0.0025, 0.005, 0.0075, 0.010, 0.0125, 0.0145, 0.0155,
                0.0175, 0.020, 0.0225, 0.025])
F_MEAS = np.array([0.0, 1400.0, 2800.0, 3900.0, 4500.0, 4900.0, 5150.0,
                   5100.0, 3700.0, 2600.0, 2300.0, 1900.0])
F_NONLIN = np.array([0.0, 1400.0, 2850.0, 4200.0, 5500.0, 6200.0, 6550.0,
                     6650.0, 5000.0, 3500.0, 3100.0, 2800.0])
T_V = np.array([0.0, 0.0025, 0.005, 0.0075, 0.010, 0.0125, 0.015, 0.0175,
                0.020, 0.0225, 0.025])
V_MEAS = np.array([6.15, 6.14, 6.08, 5.97, 5.80, 5.58, 5.35, 5.15, 5.02, 4.94,
                   4.90])


class WedgeSection:
    """One-station section table in the interface SlamLoad expects; d is
    immersion relative to the knuckle, so d = -H_K when the keel touches."""
    n = 1
    T = H_K

    def area(self, d):
        h = np.asarray(d, float) + H_K
        below = np.clip(h, 0.0, H_K)
        above = np.maximum(h - H_K, 0.0)
        return below ** 2 / np.tan(BETA) + 2 * HALF_B * above


def force_per_length(slam, h, v, dvdt, variant):
    d = np.atleast_1d(h - H_K)
    if variant == "model now":
        return float(slam.wagner_slope(d)[0]) * v * v
    if variant == "now + m_a dV/dt":
        return (float(slam.wagner_slope(d)[0]) * v * v
                + float(slam.wagner_added_mass(d)[0]) * dvdt)
    if variant == "model before":
        return SlamLoad.PILE_UP * float(slam.slope(d)[0]) * v * v
    if variant == "von Karman":
        return float(slam.slope(d)[0]) * v * v
    raise KeyError(variant)


def main():
    slam = SlamLoad(WedgeSection(), rho=RHO)
    print("\nEXPERIMENT -- slamming force on a 30 deg wedge, "
          "Zhao, Faltinsen & Aarsnes drop test\n")

    dvdt = np.gradient(V_MEAS, T_V)
    print("  digitisation check: rig deceleration x mass, apportioned to the "
          "0.2 m section")
    for t in (0.010, 0.015):
        i = int(np.argmin(abs(T_V - t)))
        F_rig = M_RIG * (G - dvdt[i]) * L_MEAS / L_RIG
        F_dig = float(np.interp(t, T_F, F_MEAS))
        print(f"    t = {t:.3f} s:  from V(t) {F_rig:6.0f} N   digitised "
              f"{F_dig:6.0f} N   ({F_dig/F_rig-1:+.0%})")

    t = np.linspace(0.0, 0.025, 501)
    v = np.interp(t, T_V, V_MEAS)
    h = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    a = np.gradient(v, t)
    names = ("model now", "model before", "von Karman", "now + m_a dV/dt")
    res = {n: np.array([force_per_length(slam, hi, vi, ai, n)
                        for hi, vi, ai in zip(h, v, a)]) * L_MEAS
           for n in names}

    def separation_time(f):
        i = int(np.argmax(f))
        after = np.where(f[i:] <= 0.02 * f[i])[0]
        return t[i + after[0]] if len(after) else float("nan")

    print(f"\n  entry force ends (flow separates at the knuckles):")
    for n in names[:3]:
        print(f"    {n:<16} {separation_time(res[n])*1000:5.1f} ms")
    print(f"    measured force peaks near 15 ms and then falls")

    print(f"\n  force on the 0.2 m measuring section, N  "
          f"(measured V(t) prescribed)\n")
    print(f"  {'t ms':>6}{'measured':>10}{'nonlin 2D':>11}"
          + "".join(f"{n:>18}" for n in names))
    for tt in (0.0025, 0.005, 0.0075, 0.010, 0.0125, 0.015, 0.0175, 0.020,
               0.025):
        row = (f"  {tt*1000:>6.1f}{np.interp(tt, T_F, F_MEAS):>10.0f}"
               f"{np.interp(tt, T_F, F_NONLIN):>11.0f}")
        row += "".join(f"{np.interp(tt, t, res[n]):>18.0f}" for n in names)
        print(row)

    early = (t > 0.002) & (t < 0.0075)
    fm = np.interp(t[early], T_F, F_MEAS)
    print("\n  ratio to measurement, early phase (2-7.5 ms, before separation):")
    for n in names:
        r = res[n][early] / fm
        print(f"    {n:<18} {np.mean(r):.2f}x")
    print(f"    fully nonlinear 2-D (paper) "
          f"{np.mean(np.interp(t[early], T_F, F_NONLIN) / fm):.2f}x")
    for n in names:
        print(f"    peak, {n:<16} {res[n].max():7.0f} N at "
              f"{t[np.argmax(res[n])]*1000:5.1f} ms   (measured 5150 N, ~15 ms)")

    # free drop of the rig under the model as it now stands
    dt, hh, vv, tt = 1e-5, 0.0, V_MEAS[0], 0.0
    traj = []
    while tt < 0.025:
        f = force_per_length(slam, hh, vv, 0.0, "model now") * L_RIG
        vv += (G - f / M_RIG) * dt
        hh += vv * dt
        tt += dt
        traj.append((tt, vv, f * L_MEAS / L_RIG))
    traj = np.array(traj)
    print("\n  free drop of the 241 kg rig under the model as it now stands:")
    for q in (0.010, 0.015, 0.020, 0.025):
        i = int(np.argmin(abs(traj[:, 0] - q)))
        print(f"    t {q*1000:4.0f} ms   V model {traj[i,1]:.2f} m/s  measured "
              f"{np.interp(q, T_V, V_MEAS):.2f}   section force model "
              f"{traj[i,2]:6.0f} N  measured {np.interp(q, T_F, F_MEAS):6.0f}")
    print("\n  Separation timing is now right. The force level before it is not:")
    print("  Wagner's flat-plate expansion over-predicts a 30 deg wedge by about")
    print("  1.9x against this test, while the fully nonlinear solution matches")
    print("  it early. One deadrise angle cannot pin a deadrise-dependent")
    print("  correction, so the bias is documented, not calibrated away.")
    return res


if __name__ == "__main__":
    main()
