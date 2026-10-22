"""
Stage B8 (route 1 of the planing gate): the 3-D linear free-surface theory with gravity of stage B7
(planing_steady.PlaningSprayRootFast) with the flat-plate cross-section replaced by a NONLINEAR wedge-entry
section, composite in the sense of matched asymptotic expansions (derivation and checks:
learn/meta/RANKINE_STAGE1_2026-10-06.md, section 14).

    L = L_3D,lin + int_{-L_K}^{0} [ F_NL(x) - F_W(x) ] dx,          M_T = M_3D,lin + int (-x) [F_NL - F_W] dx   (C)

  * L_3D,lin, M_3D,lin, L_K, L_C: the B7 solution (outer flow with gravity, transom Kutta condition, spray-root
    line from the linearised Wagner condition, wetted length from the solution).
  * inner kinematics (matching): every cross-section x enters with the constant velocity V that the 3-D
    solution's own wetted planform implies through Wagner's wedge condition c = pi h/(2 tan b):
        V / U = (2/pi) l tan(beta) / (L_K - L_C),   h(x) = (V/U)(x + L_K),   l = b/2               (K)
    (for the slender Wagner planform L_K - L_C = b tan(beta)/(pi tan(tau)) this is V = U tan(tau)); the
    outer (gravity-wave) change of the effective penetration is in V through L_K - L_C.
  * F_W: the inner limit of the linear theory, Wagner's flat plate: rho V^2 h pi^3/(4 tan^2 b) before chine
    wetting (h < h_cw = (2/pi) l tan b), 0 after (constant c = l, constant V).
  * F_NL: chine-dry: modified Logvinovich model (planing_wedge2d; force rho V^2 h kappa_MLM(beta));
    chine-wetted (flow separation at the chine): the similitude of Wen, Liu, Del Buono, Qu & Iafrati (2022,
    J. Fluids Struct. 113; arXiv 2201.00959, eqs 16-22, lambda = 1.3075) for the decay from the force at
    chine wetting to the steady supercavitating-flow force (Gurevich; eqs 33-36), anchored at the MLM force at
    chine wetting (continuity). Registered empirical element E-B8-1 (EMPIRICAL_REGISTER_RANKINE.md).
Nondimensional: rho = U = b = 1; lift / (rho U^2 b^2) (C_L = 2 lift), moment about the transom / (rho U^2 b^3).
"""
import math

import numpy as np
from scipy.integrate import quad

from . import planing_wedge2d as w2

WEN_LAMBDA = 1.3075          # Wen et al. 2022, eq. (16)-(18)
_CACHE = {}


def cs_inf(beta):
    """Steady supercavitating (free-streamline) flow past the wedge, force / (rho v^2 l / 2) on one side =
    F_full / (rho v^2 l) (Wen et al. 2022 appendix 7.1 after Gurevich): exact; beta = 0 gives 2 pi/(pi + 4)."""
    key = ("csinf", round(beta, 12))
    if key not in _CACHE:
        e = 1.0 - 2.0 * beta / math.pi
        G = lambda t: ((1.0 + math.sqrt(1.0 - t * t)) / t) ** e * t
        A = quad(G, 0.0, 1.0, limit=400, epsabs=1e-13, epsrel=1e-12)[0]
        Cp = lambda t: 1.0 - ((1.0 - math.sqrt(1.0 - t * t)) / t) ** (2.0 * e)
        _CACHE[key] = quad(lambda t: Cp(t) * G(t), 0.0, 1.0, limit=400, epsabs=1e-13, epsrel=1e-12)[0] / A
    return _CACHE[key]


def wen_cstar(hs):
    """Eq. (19)/(22) of Wen et al. 2022: scaled decay of the force in the transition (chine-wetted) stage."""
    hs = np.asarray(hs, float)
    out = (1.539 * hs + 2.618) / (hs * hs + 8.081 * hs + 2.169)
    return np.where(hs <= 0.067, 1.0, out)


def kappa_mlm(beta_deg):
    key = ("kmlm", round(beta_deg, 12))
    if key not in _CACHE:
        _CACHE[key] = w2.similarity_coefficients(beta_deg)["F"]
    return _CACHE[key]


def kappa_wen(beta):
    """Similarity-solution force coefficient F/(rho V^3 t) as fitted by Wen et al. 2022 (k2 cot b)."""
    return (1.585 * beta ** 2 - 6.856 * beta + 7.764) / math.tan(beta) ** 2


def section_forces(beta_deg, VU, LK, LC, h_nodes=4001, part="full", anchor="mlm", l=0.5):
    """Sectional forces along x in [-L_K, 0] (nondimensional, per unit length): returns x, F_NL, F_W.
    part: 'full' | 'mlm_only' (chine-wetted F_NL = 0, i.e. no separated-flow force) | 'sep_only'
    (chine-dry F_NL = F_W)."""
    b = math.radians(beta_deg)
    T = math.tan(b)
    hcw = 2.0 / math.pi * l * T
    xcw = hcw / VU - LK                       # chine-wetting station of the inner model
    if xcw >= 0.0:
        x = np.linspace(-LK, 0.0, h_nodes)
        dry = np.ones(len(x), bool)
    else:
        # the chine-wetting station appears twice (left value dry, right value wet): exact trapezoid at the jump
        x = np.concatenate([np.linspace(-LK, xcw, h_nodes), np.linspace(xcw, 0.0, h_nodes)])
        dry = np.r_[np.ones(h_nodes, bool), np.zeros(h_nodes, bool)]
    h = VU * (x + LK)
    kW = math.pi ** 3 / (4 * T * T)
    kM = kappa_mlm(beta_deg) if anchor == "mlm" else kappa_wen(b)
    FW = np.where(dry, VU ** 2 * h * kW, 0.0)
    FN = np.where(dry, VU ** 2 * h * kM, 0.0)
    if part == "sep_only":
        FN = np.where(dry, FW, FN)
    if part != "mlm_only":
        csm = kM * hcw / l               # Cs at chine wetting (force/(rho V^2 l))
        ci = cs_inf(b)
        hs = (h - hcw) / hcw / T ** WEN_LAMBDA
        cs = ci + wen_cstar(hs) * (csm - ci)
        FN = np.where(dry, FN, VU ** 2 * l * cs)
    return x, FN, FW


def increments(beta_deg, VU, LK, LC, **kw):
    """Composite increments (C): dlift = int (F_NL - F_W) dx, dmoment = int (-x)(F_NL - F_W) dx."""
    x, FN, FW = section_forces(beta_deg, VU, LK, LC, **kw)
    d = FN - FW
    return float(np.trapezoid(d, x)), float(np.trapezoid(-x * d, x)), float(np.trapezoid(FN, x)), \
        float(np.trapezoid(FW, x))


def inner_velocity(beta_deg, LK, LC, l=0.5):
    """(K): V/U from the 3-D solution's keel-chine wetted-length difference."""
    T = math.tan(math.radians(beta_deg))
    return 2.0 / math.pi * l * T / max(LK - LC, 1e-9)


def composite(S, st, part="full", anchor="mlm", vmode="3d"):
    """Composite lift/moment (C) on top of a B7 solution `st` of solver S. vmode '3d' uses (K); 'slender'
    uses V = U tan(tau) (sensitivity)."""
    LK, LC = st["LK"], st["LC"]
    VU = inner_velocity(S.beta_deg, LK, LC) if vmode == "3d" else math.tan(st["tau"])
    dL, dM, LN, LW = increments(S.beta_deg, VU, LK, LC, part=part, anchor=anchor)
    lift = st["lift"] + dL
    mom = st["moment"] + dM
    return dict(lift=lift, CL=2.0 * lift, moment=mom, lp=mom / lift, dlift=dL, dmoment=dM, L_NL2D=LN, L_W2D=LW,
                VU=VU, LK=LK, LC=LC, lam=st["lam"], CL_lin=st["CL"], tau=st["tau"], converged=st["converged"])


def solve_lambda(S, tau, lam, **kw):
    st = S.solve_lambda(tau, lam)
    out = composite(S, st, **kw)
    out["st"] = st
    return out


def equilibrium(S, lift_target, lp_target, tau0, lam0, tol=1e-7, **kw):
    """Free trim and wetted length at given lift and centre of gravity (pressure moment only), composite
    forces; scipy fsolve on (tau, lam) as in B7."""
    from scipy.optimize import fsolve
    cache = {}

    def F(v):
        tau, lam = v
        out = solve_lambda(S, tau, lam, **kw)
        cache["o"] = out
        return [out["lift"] / lift_target - 1.0, out["lp"] - lp_target]
    sol, info, ier, msg = fsolve(F, [tau0, lam0], full_output=True, xtol=tol)
    out = solve_lambda(S, sol[0], sol[1], **kw)
    out.update(eq_ok=(ier == 1), eq_msg=msg, eq_nfev=info["nfev"])
    return out
