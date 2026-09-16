#!/usr/bin/env python3
"""
Propeller open-water characteristics: the Wageningen B-series regression.

WHY
The propulsion model used a linear K_T(J) = kt0 (1 - J/j0) with kt0 = 0.45 and
j0 = 0.75 set by hand -- the right shape, no source. The B-series is the
published regression of 120 open-water model tests at NSMB. It gives thrust and
torque coefficients for any blade number Z, blade-area ratio AE/A0 and pitch
ratio P/D:

    K_T = sum_i c_i J^s_i (P/D)^t_i (AE/A0)^u_i Z^v_i        39 terms
    K_Q = sum_j c_j J^s_j (P/D)^t_j (AE/A0)^u_j Z^v_j        47 terms

with J = V_a / (n D), the advance ratio.

DATA, AND WHY IT CAN BE TRUSTED
The coefficients are in data/external/bseries/coeffs.dat (Bernitsas, Ray and
Kinley 1981, via github.com/mkergoat/bseries, GPL-3.0). They were compared term
by term against a second, independent transcription -- WageningData.txt in
Fossen's MSS toolbox (MIT): all 39 K_T terms identical, all 47 K_Q terms equal
to within 3e-6, one last-digit rounding. Two people copying the same table and
arriving at the same 86 numbers is the check that a typo cannot pass.

WHAT IT DOES NOT COVER
The regression is for Reynolds number 2e6 (stated in the MSS file). The range
of Z, AE/A0 and P/D it covers is commonly cited as 2-7, 0.30-1.05 and 0.5-1.4.
Those ranges are NOT verified against the report here -- the report was not
reachable -- so going outside them warns instead of extrapolating silently.

A propeller that is not a B-series design (skew, different blade sections)
will differ from these curves. How much is exactly what
studies/exp_propeller.py measures, on the KVLCC2 propeller, whose open-water
curve was measured independently at two institutes.

Run: python -m sim.propeller
"""
import os
import warnings

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COEFFS = os.path.join(HERE, "data", "external", "bseries", "coeffs.dat")
RANGES = dict(Z=(2, 7), AE_A0=(0.30, 1.05), P_D=(0.5, 1.4))
RE_REGRESSION = 2e6


def load_coefficients(path=COEFFS):
    """(K_T terms, K_Q terms), each row (c, s, t, u, v)."""
    kt, kq = [], []
    with open(path) as f:
        for line in f:
            r = line.split()
            if not r or r[0] == "CQ":
                continue
            kq.append([float(r[0])] + [int(float(x)) for x in r[1:5]])
            if float(r[5]) != 0.0:
                kt.append([float(r[5])] + [int(float(x)) for x in r[6:10]])
    kt, kq = np.array(kt), np.array(kq)
    if len(kt) != 39 or len(kq) != 47:
        raise ValueError(f"expected 39 K_T and 47 K_Q terms, found "
                         f"{len(kt)} and {len(kq)} -- wrong file?")
    return kt, kq


class WageningenB:
    """One B-series propeller: fixed Z, AE/A0, P/D and diameter D."""

    def __init__(self, Z, AE_A0, P_D, D, coeff_path=COEFFS, strict=False):
        self.Z, self.AE_A0, self.P_D = int(Z), float(AE_A0), float(P_D)
        self.D = float(D)
        kt, kq = load_coefficients(coeff_path)
        outside = [f"{k} = {v}" for k, v in (("Z", self.Z),
                                             ("AE_A0", self.AE_A0),
                                             ("P_D", self.P_D))
                   if not RANGES[k][0] <= v <= RANGES[k][1]]
        if outside:
            msg = ("B-series regression used outside its commonly cited "
                   "range: " + ", ".join(outside))
            if strict:
                raise ValueError(msg)
            warnings.warn(msg)
        self._t = self._fold(kt)
        self._q = self._fold(kq)
        # The regression is a polynomial in J fitted over the working range.
        # Far past the zero-thrust advance ratio it is no longer a propeller:
        # at J ~ 15 it gave +7 N from a shaft turning 0.35 rev/s, at J ~ 1e5
        # meganewtons (DEFECTS G5). From J0 on the thrust is taken as zero --
        # the propeller is windmilling, and the braking quadrant is another
        # curve that this regression does not describe.
        try:
            self.J0 = self.j_zero_thrust()
        except ValueError:
            self.J0 = None

    def _fold(self, terms):
        """Everything except the J dependence, evaluated once."""
        c, s, t, u, v = terms.T
        return (c * self.P_D ** t * self.AE_A0 ** u * float(self.Z) ** v,
                s.astype(int))

    @staticmethod
    def _eval(folded, J):
        a, s = folded
        Jv = np.atleast_1d(np.asarray(J, float))
        val = (a[None, :] * Jv[:, None] ** s[None, :]).sum(axis=1)
        return float(val[0]) if np.ndim(J) == 0 else val

    def kt(self, J):
        return self._eval(self._t, J)

    def kq(self, J):
        return self._eval(self._q, J)

    def eta0(self, J):
        """Open-water efficiency J K_T / (2 pi K_Q)."""
        J = np.asarray(J, float)
        return J * self.kt(J) / (2 * np.pi * np.maximum(self.kq(J), 1e-12))

    def j_zero_thrust(self):
        """Advance ratio at which thrust vanishes (bisection on [0, 2])."""
        lo, hi = 0.0, 2.0
        if self.kt(lo) <= 0 or self.kt(hi) >= 0:
            raise ValueError("K_T does not change sign on [0, 2]")
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.kt(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def thrust(self, n, v_a, rho=1025.0):
        """Thrust (N) at shaft speed n (rev/s) and speed of advance v_a (m/s)."""
        n = float(n)
        if n <= 0.0:
            return 0.0
        J = v_a / (n * self.D)
        if self.J0 is not None and J >= self.J0:
            return 0.0
        return rho * n * n * self.D ** 4 * self.kt(J)

    def shaft_speed_for(self, thrust, v_a, rho=1025.0):
        """Shaft speed that delivers `thrust` at speed of advance v_a.

        Thrust increases monotonically with n at fixed v_a in the positive
        quadrant, so bisection is safe and does not need a derivative.
        """
        if thrust <= 0.0:
            return 0.0
        lo, hi = 1e-6, 1.0
        while self.thrust(hi, v_a, rho) < thrust:
            hi *= 2.0
            if hi > 1e5:
                raise ValueError("no shaft speed delivers that thrust")
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.thrust(mid, v_a, rho) < thrust:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def main():
    p = WageningenB(4, 0.55, 1.0, 0.44)
    print(f"  B4-55, P/D 1.0:  K_T(0) = {p.kt(0.0):.4f}   "
          f"10 K_Q(0) = {10 * p.kq(0.0):.4f}   "
          f"J(K_T = 0) = {p.j_zero_thrust():.3f}")
    for J in (0.2, 0.4, 0.6, 0.8):
        print(f"    J {J:.1f}   K_T {p.kt(J):.4f}   10 K_Q {10*p.kq(J):.4f}   "
              f"eta0 {p.eta0(J):.3f}")


if __name__ == "__main__":
    main()
