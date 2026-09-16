#!/usr/bin/env python3
"""
Roll damping by the simplified Ikeda method (Kawahara, Maekawa & Ikeda 2009).

WHAT IT IS
Ikeda's method splits roll damping into parts with different physics:
  F   skin friction on the hull (Kato's laminar formula; scale dependent)
  W   radiated waves -- a potential-flow quantity (the BEM computes it too)
  E   eddies shed at the bilges and at the bow and stern sections
  L   lift on the hull when it rolls while moving ahead (zero at rest)
  BK  bilge keels (normal force on the keels + the pressure they cause)
Kawahara, Maekawa and Ikeda (2009) fitted regression formulas to the output
of the original section-by-section method over a methodical series of hulls,
so W, E and BK follow from B/d, C_B, C_M, OG/d, the roll frequency and
amplitude (and the bilge keel size). ITTC Recommended Procedure
7.5-02-07-04.5 (Rev 01, 2021) adopted it as its section 4.3. F and L are the
original method's own whole-ship formulas, and the forward-speed factors are
those of Ikeda's method (ITTC eqs. 31, 41, 45).

SOURCES, AND WHERE THEY DISAGREE
The paper (STAB 2009, pp. 387-398; data/external/ikeda) is followed.
verify() checks every coefficient here against ITTC Tables 1-5, parsed from
data/external/ittc. The sources differ in six places:
  1 eddy x2      ITTC eq. 100 has x2 = C_M; the paper has C_B -- and the
                 paper's own prefactor needs C_B, because
                 4 L d^4 w phi / (3 pi V B^2) = 4 w phi / (3 pi x2 x1^3)
                 holds only when V = C_B L B d.
  2 friction     ITTC eqs. 97-98 put C_M in r_f and S_f. The paper, and the
                 ITTC's own full method (eqs. 40, 40.1, Kato 1957), use C_B;
                 L (1.7 d + C_B B) is the classic wetted-surface estimate.
  3 wave, Q5     ITTC eq. 90 sums Q5_1..Q5_10 against x6^9..x6^0 and then
                 uses Q5_10 again for x1^2. The paper has x6^9..x6^1 and three
                 x1 terms: twelve coefficients, which is what Table 3 holds.
  4 Q4 row 2     -17.109 in ITTC, -17.102 in the paper (verify() shows the
                 effect; it is below 0.1%).
  5 Q1 row 7     31.41135 in ITTC, 31.4113508 in the paper (rounding).
  6 speed, wave  ITTC eq. 31 prints tanh(20 Omega - 0.3). It must be
                 tanh(20 (Omega - 0.3)): B_W/B_W0 has to be 1 at zero speed,
                 and only that reading gives it (verify() checks both).
And one the ITTC itself flags: the paper's eq. 21 prints the friction
coefficient's exponent as +1/2; ITTC eq. 96 corrects it to -1/2 (Kato's
laminar C_f = 1.328 / sqrt(Re)).

CONVENTIONS
  B44_hat = B44 / (rho V B^2) * sqrt(B / 2g),   w_hat = w sqrt(B / 2g)
  OG = d - KG, from the waterline DOWN to the roll axis (taken at G):
       positive when G is below the waterline. Many codes use the opposite.
  phi_a in radians (the bilge-keel regression's x6 is degrees internally).
B44 is the EQUIVALENT LINEAR coefficient at amplitude phi_a: B44 * phi'
dissipates per cycle what the real, partly quadratic, damping does. The eddy
and bilge-keel parts grow with phi_a; that is how the quadratic part enters.

VALIDITY (paper eqs. 24, 31, 38; ITTC 82-83)
  0.50 <= C_B <= 0.85   2.5 <= B/d <= 4.5   0.90 <= C_M <= 0.99
  -1.5 <= OG/d <= 0.2   w_hat <= 1 (wave part)
  bilge keels: 0.01 <= b_BK/B <= 0.06, 0.05 <= l_BK/L <= 0.40
check_range() lists every violation. Nothing is clamped: an extrapolated
number is reported as one, not replaced by an edge value.

NOT FOR THE PROJECT'S WIGLEY USV: C_B = 0.444 is below the range, and the
eddy and bilge-keel parts are built on sections with a bilge, which a Wigley
does not have. For that hull only F and L (whole-ship formulas) and the BEM
wave damping apply; its eddy damping has no validated source here.

Run: python -m hydro.ikeda
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITTC_TXT = os.path.join(HERE, "data", "external", "ittc",
                        "7.5-02-07-04.5_roll_damping.txt")
FIG1_CSV = os.path.join(HERE, "data", "external", "ikeda",
                        "kawahara2009_fig1_markers.csv")

RHO_SEA, RHO_FRESH, G = 1025.0, 1000.0, 9.81
NU_SEA, NU_FRESH = 1.19e-6, 1.14e-6          # 15 C, ITTC 7.5-02-01-03

RANGES = {"C_B": (0.50, 0.85), "B/d": (2.5, 4.5), "C_M": (0.90, 0.99),
          "OG/d": (-1.5, 0.2), "w_hat": (0.0, 1.0),
          "b_BK/B": (0.01, 0.06), "l_BK/L": (0.05, 0.40)}

# ---- coefficients: the paper's eqs. 23, 31, 38, laid out as ITTC Tables 1-5.
# Rows of Q1: x1^4 .. x1^0.  Rows 1-4 / 5-8 / 9-12: the x4^2 / x4 / 1 terms of
# A1, each over x2^3 .. x2^0.  Rows 13-15 / 16-18: AA1's x3 / 1 terms, over
# x2^2 .. x2^0.
Q1 = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -0.002222, 0.040871, -0.286866, 0.599424],
    [0.0, 0.010185, -0.161176, 0.904989, -1.641389],
    [0.0, -0.015422, 0.220371, -1.084987, 1.834167],
    [-0.0628667, 0.4989259, 0.52735, -10.7918672, 16.616327],
    [0.1140667, -0.8108963, -2.2186833, 25.1269741, -37.7729778],
    [-0.0589333, 0.2639704, 3.1949667, -21.8126569, 31.4113508],
    [0.0107667, 0.0018704, -1.2494083, 6.9427931, -10.2018992],
    [0.0, 0.192207, -2.787462, 12.507855, -14.764856],
    [0.0, -0.350563, 5.222348, -23.974852, 29.007851],
    [0.0, 0.237096, -3.535062, 16.368376, -20.539908],
    [0.0, -0.067119, 0.966362, -4.407535, 5.894703],
    [0.0, 17.945, -166.294, 489.799, -493.142],
    [0.0, -25.507, 236.275, -698.683, 701.494],
    [0.0, 9.077, -84.332, 249.983, -250.787],
    [0.0, -16.872, 156.399, -460.689, 463.848],
    [0.0, 24.015, -222.507, 658.027, -660.665],
    [0.0, -8.56, 79.549, -235.827, 236.579]])
Q2 = np.array([0.0, -1.402, 7.189, -10.993, 9.45])            # x4^4 .. x4^0
# Q3 rows: the x4^6 .. x4^0 terms of A3 (paper A31..A37), over x2^6 .. x2^0
Q3 = np.array([
    [-7686.0287, 30131.5678, -49048.9664, 42480.7709, -20665.147,
     5355.2035, -577.8827],
    [61639.9103, -241201.0598, 392579.5937, -340629.4699, 166348.6917,
     -43358.7938, 4714.7918],
    [-130677.4903, 507996.2604, -826728.7127, 722677.104, -358360.7392,
     95501.4948, -10682.8619],
    [-110034.6584, 446051.22, -724186.4643, 599411.9264, -264294.7189,
     58039.7328, -4774.6414],
    [709672.0656, -2803850.2395, 4553780.5017, -3888378.9905, 1839829.259,
     -457313.6939, 46600.823],
    [-822735.9289, 3238899.7308, -5256636.5472, 4500543.147, -2143487.3508,
     538548.1194, -55751.1528],
    [299122.8727, -1175773.1606, 1907356.1357, -1634256.8172, 780020.9393,
     -196679.7143, 20467.0904]])
Q4 = np.array([
    [-0.3767, 3.39, -10.356, 11.588],        # x1^3 .. x1^0
    [-17.102, 41.495, -33.234, 8.8007],      # x2^3 .. x2^0, times x4 (ITTC -17.109)
    [36.566, -89.203, 71.8, -18.108],        # x2^3 .. x2^0
    [0.0, -0.0727, 0.7, -1.2818]])           # x1^3 .. x1^0: AA32
# Q5: x6^9 .. x6^1 (no constant), then x1^2, x1, 1
Q5 = np.array([-1.05584, 12.688, -63.70534, 172.84571, -274.05701, 257.68705,
               -141.40915, 44.13177, -7.1654, -0.0495, 0.4518, -0.61655])
Q6 = np.array([
    [-79.414, 215.695, -215.883, 93.894, -14.848],   # A_E, x2^4 .. x2^0
    [0.9717, -1.55, 0.723, 0.04567, 0.9408],         # B_E1: x2^2..1 (x x4), x2, 1
    [0.0, -219.2, 443.7, -283.3, 59.6]])             # B_E2, x2^4 .. x2^0
Q7 = np.array([
    [0.0, -0.3651, 0.3907],      # f1: x2 terms times (x1 - 2.83)^2
    [0.0, -2.21, 2.632],         # f1: x2 terms
    [0.00255, 0.122, 0.4794],    # f2: x6^2 .. 1
    [-0.8913, -0.0733, 0.0],     # f3: x7^2 .. 1, times x8^2
    [5.2857, -0.01185, 0.00189], # f3: x7^2 .. 1, times x8
    [0.00125, -0.0425, -1.86],   # B_BK1: x6^2 .. 1
    [-0.0657, 0.0586, 1.6164]])  # B_BK2: x4^2 .. 1


# ------------------------------------------------------------------ W, E, BK
def _wave_A(x1, x2, x3, x4):
    """A1, A2, A3 of the paper's eq. 23 (x4 = 1 - OG/d here)."""
    p = [np.polyval(Q1[r], x1) for r in range(18)]
    A11 = p[0] * x2 ** 3 + p[1] * x2 ** 2 + p[2] * x2 + p[3]
    A12 = p[4] * x2 ** 3 + p[5] * x2 ** 2 + p[6] * x2 + p[7]
    A13 = p[8] * x2 ** 3 + p[9] * x2 ** 2 + p[10] * x2 + p[11]
    AA11 = p[12] * x2 ** 2 + p[13] * x2 + p[14]
    AA12 = p[15] * x2 ** 2 + p[16] * x2 + p[17]
    AA1 = (AA11 * x3 + AA12) * (1.0 - x4) + 1.0
    A1 = (A11 * x4 ** 2 + A12 * x4 + A13) * AA1
    A2 = np.polyval(Q2, x4)
    AA31 = np.polyval(Q4[0], x1) * (np.polyval(Q4[1], x2) * x4
                                    + np.polyval(Q4[2], x2))
    x6 = x4 - np.polyval(Q4[3], x1)
    AA3 = AA31 * (np.polyval(np.r_[Q5[:9], 0.0], x6) + np.polyval(Q5[9:], x1))
    A3 = sum(np.polyval(Q3[i], x2) * x4 ** (6 - i) for i in range(7)) + AA3
    return A1, A2, A3


def wave_hat(B_d, C_B, C_M, OG_d, w_hat, log=np.log):
    """Wave component at zero speed, nondimensional (paper eq. 23)."""
    A1, A2, A3 = _wave_A(B_d, C_B, C_M, 1.0 - OG_d)
    x5 = np.asarray(w_hat, float)
    return A1 / x5 * np.exp(-A2 * (log(x5) - A3) ** 2 / 1.44)


def eddy_hat(B_d, C_B, C_M, OG_d, w_hat, phi_a):
    """Eddy component at zero speed, nondimensional (paper eq. 31)."""
    x1, x2, x3, x4 = B_d, C_B, C_M, OG_d
    A_E = (-0.0182 * x2 + 0.0155) * (x1 - 1.8) ** 3 + np.polyval(Q6[0], x2)
    B_E1 = ((-0.2 * x1 + 1.6) * (3.98 * x2 - 5.1525) * x4
            * (x4 * np.polyval(Q6[1, :3], x2) + np.polyval(Q6[1, 3:], x2)))
    B_E2 = (0.25 * x4 + 0.95) * x4 + np.polyval(Q6[2], x2)
    B_E3 = (46.5 - 15.0 * x1) * x2 + 11.2 * x1 - 28.6
    C_R = A_E * np.exp(B_E1 + B_E2 * x3 ** B_E3)
    return 4.0 * np.asarray(w_hat, float) * phi_a / (3.0 * np.pi * x2 * x1 ** 3) * C_R


def bilge_keel_hat(B_d, C_B, C_M, OG_d, w_hat, phi_a, bBK_B, lBK_L):
    """Bilge-keel component at zero speed, nondimensional (paper eq. 38)."""
    x1, x2, x3, x4 = B_d, C_B, C_M, OG_d
    x6, x7, x8 = np.degrees(phi_a), bBK_B, lBK_L
    f1 = (x1 - 2.83) ** 2 * np.polyval(Q7[0], x2) + np.polyval(Q7[1], x2)
    f2 = np.polyval(Q7[2], x6)
    f3 = np.polyval(Q7[3], x7) * x8 ** 2 + np.polyval(Q7[4], x7) * x8
    B1 = (5.0 * x7 + 0.3 * x1 - 0.2 * x8 + np.polyval(Q7[5], x6)) * x4
    B2 = -15.0 * x7 + 1.2 * x2 - 0.1 * x1 + np.polyval(Q7[6], x4)
    B3 = 2.5 * x4 + 15.75
    return f1 * f2 * f3 * np.exp(B1 + B2 * x3 ** B3) * np.asarray(w_hat, float)


# -------------------------------------------------- F, L and speed factors
def friction(L, B, d, C_B, OG, w, phi_a, U=0.0, rho=RHO_SEA, nu=NU_SEA):
    """Kato's frictional damping with Tamiya's speed factor, N m s
    (paper eqs. 20-23 with the ITTC's corrected exponent; ITTC eq. 41)."""
    r_f = ((0.887 + 0.145 * C_B) * (1.7 * d + C_B * B) - 2.0 * OG) / np.pi
    S_f = L * (1.75 * d + C_B * B)
    T = 2.0 * np.pi / w
    c_f = 1.328 * (3.22 * r_f ** 2 * phi_a ** 2 / (T * nu)) ** -0.5
    B0 = 4.0 / (3.0 * np.pi) * rho * S_f * r_f ** 3 * phi_a * w * c_f
    return B0 * (1.0 + 4.1 * U / (w * L))


def lift(L, B, d, C_M, OG, U, rho=RHO_SEA):
    """Hull lift damping at forward speed, N m s (ITTC eqs. 91-94)."""
    l0, lR = 0.3 * d, 0.5 * d
    kappa = 0.0 if C_M <= 0.92 else (0.1 if C_M <= 0.97 else 0.3)
    kN = 2.0 * np.pi * d / L + kappa * (4.1 * B / L - 0.045)
    return (0.5 * rho * U * L * d * kN * l0 * lR
            * (1.0 - 1.4 * OG / lR + 0.7 * OG ** 2 / (l0 * lR)))


def wave_speed_factor(w, d, U, g=G, printed_itc=False):
    """B_W / B_W0 (Ikeda 1978; ITTC eq. 31). `printed_itc` evaluates the
    ITTC's typeset tanh(20 Omega - 0.3), only to show that it is a typo."""
    xi = w * w * d / g
    Om = U * w / g
    A1 = 1.0 + xi ** -1.2 * np.exp(-2.0 * xi)
    A2 = 0.5 + xi ** -1.0 * np.exp(-2.0 * xi)
    th = np.tanh(20.0 * Om - 0.3) if printed_itc else np.tanh(20.0 * (Om - 0.3))
    return 0.5 * ((A2 + 1.0) + (A2 - 1.0) * th
                  + (2.0 * A1 - A2 - 1.0) * np.exp(-150.0 * (Om - 0.25) ** 2))


def eddy_speed_factor(w, L, U):
    """B_E / B_E0 = (0.04 K)^2 / (1 + (0.04 K)^2), K = w L / U (ITTC eq. 45)."""
    if U <= 0.0:
        return 1.0
    k = 0.04 * w * L / U
    return k * k / (1.0 + k * k)


# ----------------------------------------------------------------- the lot
def check_range(B_d, C_B, C_M, OG_d, w_hat=None, bBK_B=None, lBK_L=None):
    vals = {"C_B": C_B, "B/d": B_d, "C_M": C_M, "OG/d": OG_d, "w_hat": w_hat,
            "b_BK/B": bBK_B, "l_BK/L": lBK_L}
    out = []
    for k, v in vals.items():
        if v is None:
            continue
        lo, hi = RANGES[k]
        for x in np.atleast_1d(v):
            if x < lo or x > hi:
                out.append(f"{k} = {x:.4g} outside {lo}..{hi}")
                break
    return out


def roll_damping(L, B, d, C_B, C_M, KG, w, phi_a, U=0.0, V=None, b_BK=0.0,
                 l_BK=0.0, rho=RHO_SEA, nu=NU_SEA, g=G, B_W0=None):
    """Equivalent linear roll damping by component, N m s.

    `B_W0` replaces the wave regression with a zero-speed wave damping from
    elsewhere (the BEM), N m s; the speed factor is applied to either.
    Returns a dict with F, W, E, L, BK, total, the same nondimensional
    (key + '_hat'), and 'out_of_range'.
    """
    V = C_B * L * B * d if V is None else V
    OG = d - KG
    s = np.sqrt(B / (2.0 * g))
    wh = w * s
    to_dim = rho * V * B * B / s
    W0 = wave_hat(B / d, C_B, C_M, OG / d, wh) * to_dim if B_W0 is None else B_W0
    out = dict(
        F=friction(L, B, d, C_B, OG, w, phi_a, U, rho, nu),
        W=W0 * wave_speed_factor(w, d, U, g),
        E=eddy_hat(B / d, C_B, C_M, OG / d, wh, phi_a) * to_dim
        * eddy_speed_factor(w, L, U),
        L=lift(L, B, d, C_M, OG, U, rho) if U > 0.0 else 0.0,
        BK=(bilge_keel_hat(B / d, C_B, C_M, OG / d, wh, phi_a, b_BK / B,
                           l_BK / L) * to_dim) if b_BK > 0.0 else 0.0)
    out["total"] = out["F"] + out["W"] + out["E"] + out["L"] + out["BK"]
    for k in ("F", "W", "E", "L", "BK", "total"):
        out[k + "_hat"] = out[k] / to_dim
    out["out_of_range"] = check_range(
        B / d, C_B, C_M, OG / d, wh,
        b_BK / B if b_BK > 0.0 else None, l_BK / L if b_BK > 0.0 else None)
    return out


class RollDamping:
    """The VISCOUS part of a hull's roll damping, in the form a time-domain
    plant needs:

        K = -(b1(U) p + b2(U) p|p|)

    from the simplified Ikeda method: friction, eddy, lift and bilge keels.
    The wave part is deliberately NOT here. The plant already has it from the
    BEM, and on KVLCC2 the measured decays sided with the BEM's wave damping
    against the Ikeda wave regression by a factor of six (DEFECTS F8).

    Ikeda gives each part as an equivalent linear coefficient at a roll
    amplitude. Friction and lift do not depend on it; eddy and bilge-keel
    damping grow with it. Evaluated at two amplitudes, B_eq = c0 + c1 phi_a,
    and matched to what b1 p + b2 p|p| dissipates at that amplitude,
    b1 + (8 / 3 pi) b2 w phi_a:  b1 = c0,  b2 = (3 pi / 8) c1 / w.

    What the KVLCC2 decays say about it (1:68, DEFECTS F8): at zero speed it
    over-predicts by 33-65% (the eddy part, with C_M just outside the fitted
    range); at 15.5 kn it under-predicts by 6-16%. That is the size of error
    to expect from it on a hull inside its range -- better than a scaled
    placeholder, not a substitute for a decay test on the real vessel.
    """

    def __init__(self, L, B, d, C_B, C_M, KG, w_roll, V, rho=RHO_SEA,
                 nu=NU_SEA, g=G, b_BK=0.0, l_BK=0.0, phi_deg=(5.0, 15.0)):
        self.p = dict(L=L, B=B, d=d, C_B=C_B, C_M=C_M, KG=KG, V=V, rho=rho,
                      nu=nu, g=g, b_BK=b_BK, l_BK=l_BK)
        self.w = float(w_roll)
        self.phi = np.radians(np.asarray(phi_deg, float))
        s = np.sqrt(B / (2.0 * g))
        self.out_of_range = check_range(
            B / d, C_B, C_M, (d - KG) / d, self.w * s,
            b_BK / B if b_BK > 0 else None, l_BK / L if b_BK > 0 else None)
        self._cache = {}

    def equivalent(self, phi_a, U=0.0):
        """Equivalent linear viscous coefficient at amplitude phi_a, N m s."""
        return roll_damping(w=self.w, phi_a=phi_a, U=U, B_W0=0.0,
                            **self.p)["total"]

    def coeffs(self, U=0.0):
        """(b1, b2) at forward speed U; cached on U to the millimetre per
        second, since the plant asks every derivative evaluation."""
        key = round(float(max(U, 0.0)), 3)
        hit = self._cache.get(key)
        if hit is None:
            B1, B2 = (self.equivalent(ph, key) for ph in self.phi)
            c1 = (B2 - B1) / (self.phi[1] - self.phi[0])
            c0 = B1 - c1 * self.phi[0]
            hit = (max(float(c0), 0.0),
                   max(float(3.0 * np.pi / 8.0 * c1 / self.w), 0.0))
            self._cache[key] = hit
        return hit


def natural_roll_frequency(omega, A44, I44, C44, n_iter=50):
    """w_n from w^2 = C44 / (I44 + A44(w)), A44 interpolated on the BEM grid."""
    w = np.sqrt(C44 / (I44 + np.interp(np.sqrt(C44 / I44), omega, A44)))
    for _ in range(n_iter):
        w_new = np.sqrt(C44 / (I44 + np.interp(w, omega, A44)))
        if abs(w_new - w) < 1e-10 * w:
            break
        w = w_new
    return float(w)


# --------------------------------------------------------------- checking
def _ittc_tables(path=ITTC_TXT):
    """ITTC Tables 1-5, parsed from the text extraction of the procedure."""
    import re
    lines = open(path, encoding="utf-8").read().splitlines()
    # the last occurrence: the first is the table of contents
    i0 = max(i for i, s in enumerate(lines) if "SIMPLIFIED IKEDA" in s)
    num = re.compile(r"^-?\d+(\.\d+)?$")
    t = {k: {} for k in ("Q1", "Q3", "Q4", "Q6", "Q7")}
    t["Q2"], t["Q5"] = [], []
    cur = None
    for s in lines[i0:i0 + 160]:
        tok = s.split()
        if not tok:
            continue
        if cur == "Q7" and len(t["Q7"]) == 7 and s.startswith("==="):
            break
        if tok[0] == "Factor" and len(tok) > 1:
            cur = tok[1]
            continue
        if cur == "Q5" and tok[0] == "Q5":
            t["Q5"] += [float(v) for v in tok[1:]]
            continue
        if not all(num.match(v) for v in tok) or all("." not in v for v in tok):
            continue                      # page furniture, column-number rows
        if cur == "Q2":
            t["Q2"] = [float(v) for v in tok]
        elif cur in ("Q1", "Q3", "Q4", "Q6", "Q7"):
            t[cur].setdefault(int(tok[0]), []).extend(float(v) for v in tok[1:])

    def arr(d):
        return np.array([d[k] for k in sorted(d)])
    return dict(Q1=arr(t["Q1"]), Q2=np.array(t["Q2"]), Q3=arr(t["Q3"]),
                Q4=arr(t["Q4"]), Q5=np.array(t["Q5"]), Q6=arr(t["Q6"]),
                Q7=arr(t["Q7"]))


KNOWN_TABLE_DIFFERENCES = {("Q4", 1, 0), ("Q1", 6, 4)}


def _fig1():
    import csv
    rows = [r for r in csv.reader(l for l in open(FIG1_CSV, encoding="utf-8")
                                  if not l.startswith("#"))]
    return [dict(panel=r[0], OG_d=float(r[1]), comp=r[2], w_hat=float(r[3]),
                 B=float(r[4])) for r in rows[1:]]


def verify(verbose=True):
    ok = True
    say = print if verbose else (lambda *a, **k: None)

    say("\n  1. coefficients against ITTC Tables 1-5 (parsed from the text)")
    if os.path.exists(ITTC_TXT):
        tab = _ittc_tables()
        mine = dict(Q1=Q1, Q2=Q2, Q3=Q3, Q4=Q4, Q5=Q5, Q6=Q6, Q7=Q7)
        found = set()
        for k, a in mine.items():
            b = tab[k]
            if a.shape != b.shape:
                say(f"     {k}: shape {a.shape} here, {b.shape} in ITTC  FAIL")
                ok = False
                continue
            for idx in zip(*np.nonzero(~np.isclose(a, b, rtol=1e-9, atol=0))):
                idx = tuple(int(i) for i in idx)
                key = (k, *idx) if a.ndim == 2 else (k, idx[0])
                found.add(key)
                say(f"     {k}{list(idx)}: {a[idx]} here (paper), {b[idx]} ITTC"
                    f"{'  (known)' if key in KNOWN_TABLE_DIFFERENCES else '  UNEXPECTED'}")
        bad = found - KNOWN_TABLE_DIFFERENCES
        ok &= not bad
        say(f"     {sum(v.size for v in mine.values())} coefficients compared; "
            f"{'only the known differences' if not bad else 'UNEXPECTED differences'}")
    else:
        say("     ITTC text not found; skipped")

    say("\n  2. exact identities")
    xs = np.array([0.05, 0.2, 0.5, 1.0, 3.0])
    w = np.sqrt(xs * G / 1.0)
    f_ok = wave_speed_factor(w, 1.0, 0.0)
    f_itc = wave_speed_factor(w, 1.0, 0.0, printed_itc=True)
    # Not exactly 1 even when read correctly: the Gaussian term's tail,
    # exp(-150 * 0.25^2) = 8.5e-5, times (2 A1 - A2 - 1), which grows at small
    # xi_d. 0.2% at xi_d = 0.05. The misprinted form misses by 18% to 600%.
    good = np.allclose(f_ok, 1.0, atol=5e-3)
    ok &= bool(good)
    say(f"     B_W/B_W0 at zero speed, xi_d = {xs.tolist()}:")
    say(f"       tanh(20 (Omega - 0.3)) -> {np.round(f_ok, 5).tolist()}  "
        f"{'ok' if good else 'FAIL'}")
    say(f"       tanh(20 Omega - 0.3), as ITTC prints it -> "
        f"{np.round(f_itc, 3).tolist()}  (not 1: a typo)")
    Lx, Bx, dx, Cb = 6.0, 1.0, 0.25, 0.65
    Vx = Cb * Lx * Bx * dx
    lhs = 4 * Lx * dx ** 4 / (3 * np.pi * Vx * Bx ** 2)
    rhs = 4 / (3 * np.pi * Cb * (Bx / dx) ** 3)
    good = abs(lhs / rhs - 1) < 1e-12
    ok &= good
    say(f"     eddy prefactor 4 L d^4/(3 pi V B^2) = 4/(3 pi x2 x1^3) with "
        f"x2 = C_B: {'ok' if good else 'FAIL'} (with C_M it would not hold)")
    g0 = eddy_speed_factor(1.0, 100.0, 1e-9)
    say(f"     B_E/B_E0 as U -> 0: {g0:.6f}  {'ok' if abs(g0 - 1) < 1e-6 else 'FAIL'}")
    ok &= abs(g0 - 1) < 1e-6

    say("\n  3. against Kawahara et al. 2009, Fig. 1 -- the original Ikeda "
        "method's symbols")
    if os.path.exists(FIG1_CSV):
        pts = _fig1()
        hull = dict(B_d=4.0, C_B=0.65, C_M=0.98)
        phi = np.radians(10.0)
        for panel in ("a", "b"):
            P = [p for p in pts if p["panel"] == panel]
            og = P[0]["OG_d"]
            say(f"     ({panel}) OG/d = {og}: model / figure, and the error RMS")
            for comp in ("wave", "eddy", "bilge_keel"):
                q = [p for p in P if p["comp"] == comp and p["B"] > 2e-4]
                wh = np.array([p["w_hat"] for p in q])
                ref = np.array([p["B"] for p in q])
                if comp == "wave":
                    m_ln = wave_hat(hull["B_d"], hull["C_B"], hull["C_M"], og, wh)
                    m_10 = wave_hat(hull["B_d"], hull["C_B"], hull["C_M"], og,
                                    wh, log=np.log10)
                    for lab, m in (("wave, natural log", m_ln),
                                   ("wave, log10", m_10)):
                        r = m / ref
                        say(f"       {lab:<18} ratio {np.min(r):.2f}..{np.max(r):.2f}"
                            f"  rms {np.sqrt(np.mean((r - 1) ** 2)):.1%}"
                            f"  (n={len(r)}, B_hat > 2e-4)")
                    continue
                if comp == "eddy":
                    m = eddy_hat(hull["B_d"], hull["C_B"], hull["C_M"], og, wh, phi)
                else:
                    m = bilge_keel_hat(hull["B_d"], hull["C_B"], hull["C_M"],
                                       og, wh, phi, 0.025, 0.2)
                r = m / ref
                say(f"       {comp:<18} ratio {np.min(r):.2f}..{np.max(r):.2f}"
                    f"  rms {np.sqrt(np.mean((r - 1) ** 2)):.1%}  (n={len(r)})")
    else:
        say("     digitised figure not found; skipped")
    return ok


def main():
    print("\nSIMPLIFIED IKEDA -- implementation checks")
    ok = verify()
    print(f"\n  {'passed' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    main()
