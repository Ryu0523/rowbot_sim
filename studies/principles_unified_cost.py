#!/usr/bin/env python3
"""
Principles probe: can ONE potential-flow theory cover every speed, and what
would running it cost?  Three light checks, no tracked file is touched.

  A. Regime map for scarab195: at which boat speeds do the published regime
     thresholds fall (2D+t applicability, gravity in the planing flow,
     transom ventilation)?  Uses the v0 running attitudes already tabulated by
     studies/anyspeed_audit_transom.py (studies/_cache/anyspeed/audit_transom.txt).

  B. What "the gravity part vanishes at high speed" means, on the one formula
     that is valid at every speed for linear radiation (Cummins / Ogilvie):
         F(t) = -A_inf v'(t) - int_0^t K(t-s) v(s) ds
     A_inf is the zero-gravity (phi = 0 free surface) inertia; ALL gravity
     dependence sits in the memory kernel K.  Deep water, dimensional analysis:
         A_inf = rho L^3 a,   K(t) = rho g L^2 k(t sqrt(g/L))
     so for a velocity change that happens over a time T the ratio
         R(T) = (memory impulse over T) / (inertia impulse)
              = int_0^T int_0^s K dtau ds / A_inf  ->  K(0) T^2 / (2 A_inf)
     If T = L/U (the time a cross-section spends under the hull, the 2D+t time
     scale), R ~ g L / U^2 = 1 / Fn^2.  Checked here on the hull's own
     zero-speed 3-D BEM heave kernel (studies/_cache/anyspeed/bem_probe_lid_24_64.npz).

  C. Cost of the core operation of a 3-D panel method (dense LU + solve,
     and the per-time-step matrix-vector product when the matrix is reused),
     on this laptop's CPU and GPU, to size an offline table-generation run.

Output: studies/_cache/principles/unified_cost.txt
Run:    .venv/Scripts/python.exe -m studies.principles_unified_cost
"""
import os
import re
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_cache", "principles")
ANY = os.path.join(HERE, "_cache", "anyspeed")
G = 9.81
KN = 0.514444
L_KEEL = 5.4            # hydro/hulls.py scarab_195: L=5.4
B_CHINE = 0.83 * 2.44   # hydro/hulls.py scarab_195: b=0.83*2.44

lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def interp_speed(kn, val, target):
    """First speed at which a monotone-rising quantity reaches target."""
    for i in range(1, len(kn)):
        if val[i - 1] < target <= val[i]:
            f = (target - val[i - 1]) / (val[i] - val[i - 1])
            return kn[i - 1] + f * (kn[i] - kn[i - 1])
    return float("nan")


def part_a():
    say("A. Regime thresholds mapped to boat speed (scarab195, v0 attitudes)")
    rows = []
    with open(os.path.join(ANY, "audit_transom.txt"), encoding="utf-8") as f:
        for ln in f:
            m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                         r"([\d.]+)\s+([\d.]+)\s+([\d.]+)", ln)
            if m:
                rows.append([float(x) for x in m.groups()])
    a = np.array(rows)
    kn, dT = a[:, 0], a[:, 5]
    U = kn * KN
    FT = U / np.sqrt(G * dT)
    FnL = U / np.sqrt(G * L_KEEL)
    FnB = U / np.sqrt(G * B_CHINE)
    say(f"  {'kn':>4} {'d_T m':>6} {'F_T':>5} {'Fn_L':>5} {'Fn_B':>5}")
    for i in range(len(kn)):
        say(f"  {kn[i]:4.0f} {dT[i]:6.3f} {FT[i]:5.2f} {FnL[i]:5.2f} "
            f"{FnB[i]:5.2f}")
    say("  thresholds (source) -> first speed reached, kn")
    for name, arr, val in [
        ("Fn_L 0.6  2D+t applicable to transom hulls (Sun&Faltinsen 2007)",
         FnL, 0.6),
        ("Fn_B 2.5  gravity clearly matters in planing (Sun&Faltinsen 2007)",
         FnB, 2.5),
        ("Fn_B 5.0  gravity 'not so significant' (Sun&Faltinsen 2007)",
         FnB, 5.0),
        ("F_T 2.0   transom dry, UMich models (Maki et al. 2004)", FT, 2.0),
        ("F_T 2.5   transom dry, UNSW models (Doctors 2003)", FT, 2.5),
        ("F_T 2.6   2-D potential-flow branch crossing (VdBroeck/Scorpio)",
         FT, 2.6),
        ("F_T 3.25  naval combatants (quoted in Maki et al. 2004)", FT, 3.25),
        ("F_T 4.5   Saunders 1957 advice (quoted in Maki et al. 2004)", FT,
         4.5)]:
        say(f"    {name:68s} {interp_speed(kn, arr, val):5.1f}")
    say()


def part_b():
    from hydro.retardation import retardation_function, infinite_added_mass
    say("B. Gravity enters linear radiation only through the memory kernel;")
    say("   its weight against the phi=0 inertia, for a velocity change over")
    say("   T = L/U, falls like 1/U^2 (zero-speed 3-D BEM, heave, at rest)")
    z = np.load(os.path.join(ANY, "bem_probe_lid_24_64.npz"))
    om, A33, B33 = z["omega"], z["A"][:, 2, 2], z["B"][:, 2, 2]
    t = np.arange(0.0, 30.0, 0.005)
    for cut in (12.4, 15.8):
        k = om <= cut + 1e-9
        K = retardation_function(om[k], B33[k], t)
        Ainf, spread = infinite_added_mass(om[k], A33[k], t, K)
        # R(T) = int_0^T int_0^s K dtau ds / A_inf, exact on the grid
        cK = np.concatenate([[0.0], np.cumsum(0.5 * (K[1:] + K[:-1])
                                              * np.diff(t))])
        ccK = np.concatenate([[0.0], np.cumsum(0.5 * (cK[1:] + cK[:-1])
                                               * np.diff(t))])
        say(f"  w cut {cut:5.1f} rad/s: A_inf {Ainf:7.0f} kg (spread "
            f"{100 * spread:.1f} %), K(0) {K[0]:8.0f} N/m, "
            f"K(0)/(rho g L^2) {K[0] / (1025 * G * L_KEEL ** 2):.3f}")
        say(f"    {'kn':>4} {'T=L/U s':>8} {'R(T)':>7} {'K0 T^2/2Ainf':>13}"
            f" {'R*Fn_L^2':>9}")
        for kn_ in (2, 5, 10, 15, 20, 25, 30, 35):
            U = kn_ * KN
            T = L_KEEL / U
            R = np.interp(T, t, ccK) / Ainf
            say(f"    {kn_:4d} {T:8.3f} {R:7.3f} {K[0] * T * T / 2 / Ainf:13.3f}"
                f" {R * U * U / (G * L_KEEL):9.3f}")
        # log-log slope between 25 and 35 kn
        Ts = L_KEEL / (np.array([25.0, 35.0]) * KN)
        Rs = np.interp(Ts, t, ccK) / Ainf
        slope = np.log(Rs[1] / Rs[0]) / np.log(35.0 / 25.0)
        say(f"    d ln R / d ln U between 25 and 35 kn = {slope:.2f} "
            f"(asymptote -2, reached only when T << kernel time scale)")
        ell = G * Ainf / K[0]
        say(f"    gravity length ell = g A_inf / K(0) = {ell:.2f} m "
            f"(R -> g T^2 / (2 ell) as T -> 0)")
        for Rt in (0.10, 0.05):
            i = np.argmax(ccK / Ainf >= Rt)
            Tt = t[i]
            say(f"    R = {Rt:.2f} at T = {Tt:.3f} s  ->  U = L/T: "
                f"{L_KEEL / Tt / KN:5.0f} kn (L = {L_KEEL} m);  "
                f"{2.5 / Tt / KN:5.0f} kn (wetted length 2.5 m)")
    say()


def time_it(fn, reps=3):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def part_c():
    import scipy.linalg as sla
    say("C. Dense panel-method kernels on this laptop (best of 3)")
    say(f"  {'N':>6} {'CPU LU f64 s':>13} {'CPU matvec ms':>14} "
        f"{'GPU solve f32 s':>16} {'GPU solve f64 s':>16} "
        f"{'GPU matvec f32 ms':>18}")
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False
    rng = np.random.default_rng(0)
    for N in (1000, 2000, 4000, 8000):
        M = rng.standard_normal((N, N)) + N * np.eye(N)
        b = rng.standard_normal(N)
        t_lu = time_it(lambda: sla.lu_factor(M, check_finite=False),
                       reps=1 if N >= 8000 else 3)
        t_mv = time_it(lambda: M @ b, reps=10)
        g32 = g64 = gmv = float("nan")
        if gpu:
            for dt_, slot in ((torch.float32, 0), (torch.float64, 1)):
                Mg = torch.tensor(M, dtype=dt_, device="cuda")
                bg = torch.tensor(b, dtype=dt_, device="cuda")[:, None]
                torch.linalg.solve(Mg, bg)
                torch.cuda.synchronize()

                def run():
                    torch.linalg.solve(Mg, bg)
                    torch.cuda.synchronize()
                tt = time_it(run)
                if slot == 0:
                    g32 = tt

                    def mv():
                        Mg @ bg
                        torch.cuda.synchronize()
                    gmv = time_it(mv, reps=20)
                else:
                    g64 = tt
                del Mg, bg
                torch.cuda.empty_cache()
        say(f"  {N:6d} {t_lu:13.3f} {1e3 * t_mv:14.2f} {g32:16.4f} "
            f"{g64:16.4f} {1e3 * gmv:18.3f}")
        del M
    say()


def main():
    os.makedirs(OUT, exist_ok=True)
    part_a()
    part_b()
    part_c()
    with open(os.path.join(OUT, "unified_cost.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
