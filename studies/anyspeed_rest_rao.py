#!/usr/bin/env python3
"""
Probe (design phase): the planing plant at ZERO speed in small regular head
waves against the zero-speed BEM frequency-domain solution for the same
hull at the same rest attitude -- the "linear limit" check C of
studies/hull_acceptance.py, applied to scarab195 at rest.

Needs studies/_cache/anyspeed/bem_probe_lid_24_64.npz from
studies/anyspeed_bem_probe.py (24 frequencies, lid, 64 stations).

Output: studies/_cache/anyspeed/rest_rao.txt

Run: python -m studies.anyspeed_rest_rao
"""
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_cache", "anyspeed")
G = 9.81


def main(npz="bem_probe_lid_24_64.npz"):
    import capytaine as cpt
    from hydro.hulls import scarab_195
    from sim.planing_vessel import PlaningVessel
    from sim.test_vessel import Monochromatic
    from sim.cummins import extract_amplitude_phase
    from studies.anyspeed_bem_probe import hull_mesh

    z = np.load(os.path.join(OUT, npz))
    om, A, B = z["omega"], z["A"], z["B"]
    F = (z["F_re"] + 1j * z["F_im"])[:, 0, :]          # head seas (pi)
    h = scarab_195()
    p = PlaningVessel(h, Monochromatic(1.0, 0.0), dt=0.005)
    mesh = hull_mesh(p, nx=64, nb=10, ns=6).rotated_y(p.Th_rest) \
        .translated_z(p.Z_rest)
    body = cpt.FloatingBody(mesh=mesh, center_of_mass=(0.0, 0.0, p.Z_rest))
    body.add_all_rigid_body_dofs()
    body = body.immersed_part()
    C = np.asarray(body.compute_hydrostatic_stiffness(rho=h.rho).values,
                   float)
    M = np.diag([p.m, p.m, p.m, p.Ix, p.Iy, p.Iz])
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say(f"hydrostatics (BEM mesh at the plant's rest attitude): C33 "
        f"{C[2, 2]:.0f} N/m, C55 {C[4, 4]:.0f} N m/rad, C35 {C[2, 4]:.0f}")
    # the plant's own linear stiffness, by finite difference of its strip
    # buoyancy about rest (captive at u = 0, calm)
    pc = PlaningVessel(h, Monochromatic(1.0, 0.0), dt=0.005, captive_u=0.0)
    s0 = np.zeros(14)
    eps = 1e-4
    def fz_my(s):
        ds, aux = pc.deriv(s, 0.0)
        return ds[8], ds[10]
    a0 = np.array(fz_my(s0))
    s1 = s0.copy(); s1[2] = eps
    s2 = s0.copy(); s2[4] = eps
    # accelerations -> forces need the instantaneous mass; report the
    # plant's own at-rest natural frequencies instead (undamped, from the
    # acceleration response to a small displacement)
    k_h = -(np.array(fz_my(s1)) - a0) / eps
    k_p = -(np.array(fz_my(s2)) - a0) / eps
    say(f"plant at rest: d(w')/dz {k_h[0]:.1f} 1/s^2 -> heave wn "
        f"{np.sqrt(max(k_h[0], 0)):.2f} rad/s; d(q')/dth {k_p[1]:.1f} -> pitch "
        f"wn {np.sqrt(max(k_p[1], 0)):.2f} rad/s (coupled terms "
        f"{k_h[1]:.1f}, {k_p[0]:.1f})")
    ia = int(np.argmin(np.abs(om - 3.9)))
    wnh = np.sqrt(C[2, 2] / (p.m + A[ia, 2, 2]))
    wnp = np.sqrt(C[4, 4] / (p.Iy + A[ia, 4, 4]))
    say(f"BEM at rest (A near 3.9 rad/s): heave wn {wnh:.2f} rad/s, pitch wn "
        f"{wnp:.2f} rad/s")

    say(f"\n  regular head waves, amplitude 0.02 m, zero speed (surge held)")
    say(f"  {'w':>5}{'lam/L':>7}{'heave/a BEM':>13}{'plant':>8}"
        f"{'pitch/ka BEM':>14}{'plant':>8}")
    for w in (1.5, 2.5, 3.2, 3.9, 4.7, 5.6, 7.3):
        k = int(np.argmin(np.abs(om - w)))
        w = float(om[k])
        Z = -w ** 2 * (M + A[k]) - 1j * w * B[k] + C
        ix = [2, 4]                    # surge held, as in the plant run
        xi = np.zeros(6, complex)
        xi[ix] = np.linalg.solve(Z[np.ix_(ix, ix)], F[k][ix])
        kw = w * w / G
        a = 0.02
        sea = Monochromatic(w, a, theta0=np.pi)
        v = PlaningVessel(h, sea, dt=0.005, captive_u=0.0)
        s = np.zeros(14)
        T = 2 * np.pi / w
        n = int(40 * T / v.dt)
        t = np.arange(n) * v.dt
        rec = np.empty((n, 2))
        for i in range(n):
            s = v.step(s, t[i], 0.0, 0.0)
            s[6] = 0.0
            rec[i] = s[2], s[4]
        hh, _ = extract_amplitude_phase(t, rec[:, 0], w, n_last_periods=10)
        pp, _ = extract_amplitude_phase(t, rec[:, 1], w, n_last_periods=10)
        say(f"  {w:>5.2f}{2 * np.pi / kw / h.L:>7.2f}{abs(xi[2]):>13.3f}"
            f"{hh / a:>8.3f}{abs(xi[4]) / kw:>14.3f}{pp / (kw * a):>8.3f}")
    with open(os.path.join(OUT, "rest_rao.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
