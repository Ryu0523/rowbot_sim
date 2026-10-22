#!/usr/bin/env python3
"""Order of the slender-body (2D+t) error for planing planforms, at linear
order and zero gravity (the only limit where the exact 3-D answer is cheap).

Fact used (derived in HULL_FORCE_DERIVATION section 3.2): the gravity-free
linearised planing problem (phi = 0 ... dynamic condition phi_x = 0 on the
free surface) is, by the odd image in z = 0, the thin-wing problem for the
same planform; the planing force is half the wing force. 2D+t (gravity-free,
linear) is then exactly R. T. Jones's slender-wing theory:
    dL/dx = (rho U^2 alpha / 2) d(pi s(x)^2)/dx      (planing: half of wing)
with s(x) the local wetted semi-span.

Exact reference: a vortex-lattice (horseshoes, 1/4-3/4 rule, Kutta at the
transom) for the same planform, converged in panel count.
Outputs studies/_cache/derive/slender.txt / slender.json.
"""
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "studies", "_cache", "derive")


def _seg(P, A, B):
    """Velocity at points P (n,3) induced by unit vortex segments A->B
    (m,3 each): returns (n,m,3)."""
    r1 = P[:, None, :] - A[None, :, :]
    r2 = P[:, None, :] - B[None, :, :]
    r0 = B - A
    cr = np.cross(r1, r2)
    cr2 = (cr ** 2).sum(-1)
    n1 = np.linalg.norm(r1, axis=-1)
    n2 = np.linalg.norm(r2, axis=-1)
    dot = (r0[None] * (r1 / n1[..., None] - r2 / n2[..., None])).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(cr2 > 1e-12, dot / (4 * np.pi * cr2), 0.0)
    return cr * k[..., None]


def vlm(x_le, x_te, semi, nc, ns):
    """Flat planform z = 0, flow +x, angle alpha = 1 (linear). x_le(y),
    x_te(y) callables, semi-span `semi`. Returns lift/(rho U^2 alpha) and
    x_cp (lift-weighted bound-vortex x)."""
    # spanwise cosine spacing
    th = np.linspace(0, np.pi, ns + 1)
    yb = -semi * np.cos(th)
    A, B, C, xb_all, dy_all = [], [], [], [], []
    far = 1e4
    for j in range(ns):
        y0, y1 = yb[j], yb[j + 1]
        ym = 0.5 * (y0 + y1)
        le0, le1, lem = x_le(y0), x_le(y1), x_le(ym)
        te0, te1, tem = x_te(y0), x_te(y1), x_te(ym)
        for i in range(nc):
            f = i / nc
            g = (i + 0.25) / nc
            h = (i + 0.75) / nc
            xa = le0 + g * (te0 - le0)
            xbb = le1 + g * (te1 - le1)
            A.append([xa, y0, 0.0])
            B.append([xbb, y1, 0.0])
            C.append([lem + h * (tem - lem), ym, 0.0])
            xb_all.append(lem + g * (tem - lem))
            dy_all.append(y1 - y0)
    A, B, C = np.array(A), np.array(B), np.array(C)
    Af = A.copy()
    Af[:, 0] = far
    Bf = B.copy()
    Bf[:, 0] = far
    # horseshoe: far-left -> A -> B -> far-right
    w = (_seg(C, Af, A) + _seg(C, A, B) + _seg(C, B, Bf))[..., 2]
    gam = np.linalg.solve(w, -np.ones(len(C)))   # w_induced = -U alpha
    dy = np.array(dy_all)
    xb = np.array(xb_all)
    Lw = (gam * dy).sum()            # wing lift / (rho U alpha) per unit U
    xcp = (gam * dy * xb).sum() / Lw
    return Lw, xcp


def jones(s_of_x, x0, x1, n=4000):
    """Slender-wing lift / (rho U^2 alpha) and x_cp: lift density
    d(pi s^2)/dx where s grows, 0 where it shrinks (no recovery)."""
    x = np.linspace(x0, x1, n)
    s2 = np.maximum.accumulate(s_of_x(x) ** 2)
    dl = np.gradient(np.pi * s2, x)
    L = np.trapezoid(dl, x)
    return L, np.trapezoid(dl * x, x) / L


def rich(fun, Ns=(16, 24, 32)):
    """Run fun(N) at three lattice sizes; extrapolate L and xcp assuming
    error ~ N^-p with p from the three values (bounded to [0.5, 3])."""
    vals = [fun(n) for n in Ns]
    out = []
    for k in range(2):
        a, b, c = (v[k] for v in vals)
        r = (b - c) / (a - b) if abs(a - b) > 1e-14 else 0.5
        # N ratios differ: solve p from (N1^-p - N2^-p)/(N2^-p - N3^-p) = 1/r
        n1, n2, n3 = Ns
        ps = np.linspace(0.5, 3.0, 2501)
        f = (n2 ** -ps - n3 ** -ps) / (n1 ** -ps - n2 ** -ps)
        p = float(ps[np.argmin(abs(f - r))])
        Cc = (b - c) / (n2 ** -p - n3 ** -p)
        out.append(c - Cc * n3 ** -p)
        out.append(p)
    return out   # [L_inf, p_L, xcp_inf, p_x]


def rect_case(lam, nc, ns):
    """Rectangle: span 1 (semi 0.5), chord lam (A = 1/lam)."""
    if nc == "rich":
        Li, pL, xi, px = rich(lambda n: vlm(lambda y: 0.0, lambda y: lam,
                                            0.5, n, n))
        return Li, xi / lam, math.pi * 0.25, (pL, px)
    L, xcp = vlm(lambda y: 0.0, lambda y: lam, 0.5, nc, ns)
    Lj = math.pi * 0.25                       # pi s^2, s = 0.5, at x = 0
    return L, xcp / lam, Lj, 0.0


def hull_case(lam_m, tau_deg, beta_deg=20.0, nc=24, ns=24):
    """Planing bottom planform, beam b = 1: spray-root lines from the keel
    apex (x = 0) to the chines, chine-wet to the transom.
    L_K - L_C = tan(beta) / (pi tan(tau)) (Savitsky/Wagner), mean lam_m."""
    tb = math.tan(math.radians(beta_deg))
    D = tb / (math.pi * math.tan(math.radians(tau_deg)))
    LK = lam_m + D / 2
    s_T = 0.5 * min(1.0, LK / D)           # semi-span at the transom

    def semi_of_x(x):
        return np.minimum(0.5 * np.asarray(x) / D, 0.5)

    def x_le(y):
        return abs(y) / 0.5 * D

    L, pL, xcp, px = rich(lambda n: vlm(x_le, lambda y: LK, s_T, n, n))
    Lj, xj = jones(semi_of_x, 0.0, LK)
    return dict(LK=LK, D=D, s_T=s_T, L=L, xcp_over_LK=xcp / LK, Lj=Lj,
                xj_over_LK=xj / LK)


def main():
    os.makedirs(OUT, exist_ok=True)
    lines, res = [], dict(rect=[], hull=[], conv=[])
    # convergence on one rectangle
    for nc, ns in [(8, 8), (16, 16), (24, 24), (32, 32)]:
        L, xc, Lj, _ = rect_case(2.0, nc, ns)
        res["conv"].append(dict(nc=nc, ns=ns, L=L, xcp=xc))
        lines.append(f"conv lam 2 nc {nc} ns {ns}: L/Ljones {L/Lj:.4f} "
                     f"xcp/c {xc:.4f}")
    lines.append("rectangle: lam (=1/A), L_exact/L_slender, xcp/c "
                 "(slender puts all lift at the LE: 0)")
    for lam in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        L, xc, Lj, pp = rect_case(lam, "rich", None)
        res["rect"].append(dict(lam=lam, ratio=L / Lj, xcp=xc))
        lines.append(f"  {lam:4.2f}  {L/Lj:.4f}  {xc:.4f}  x_cp/b {xc*lam:.3f}"
                     f"  (orders {pp[0]:.2f}, {pp[1]:.2f})")
    r = res["rect"]
    A = np.array([1 / q["lam"] for q in r])
    e = np.array([abs(1 - q["ratio"]) for q in r])
    m = A <= 0.67
    sl = np.polyfit(np.log(A[m]), np.log(e[m]), 1)[0]
    xb = np.array([q["xcp"] * q["lam"] for q in r])
    lines.append(f"  slope d ln|1-L/Lj| / d ln A for A <= 0.67: {sl:.2f}; "
                 f"x_cp/b for lam >= 1.5: {xb[A <= 0.67].min():.3f}-"
                 f"{xb[A <= 0.67].max():.3f} (i.e. CP error / L_w = O(b/L_w))")
    lines.append("our planing bottoms (beam 1, beta 20), Savitsky "
                 "(lam_m, tau) at 12-40 kn from regime.txt:")
    lines.append("  kn lam tau  LK  (LK-LC)  s_T  L/Lj  xcp/LK(exact) "
                 "xcp/LK(slender)  dxcp/b")
    pts = [(12, 1.96, 5.84), (16, 1.70, 5.28), (20, 1.60, 4.34),
           (25, 1.53, 3.41), (30, 1.50, 2.75), (35, 1.48, 2.29),
           (40, 1.47, 1.96), ("FrA", 4.0, 4.0), ("FrB", 2.6, 4.0)]
    for kn, lam_m, tau in pts:
        q = hull_case(lam_m, tau)
        q.update(kn=kn, lam=lam_m, tau=tau)
        res["hull"].append(q)
        lines.append(f"  {kn} {lam_m:.2f} {tau:.2f} {q['LK']:.2f} "
                     f"{q['D']:.2f} {q['s_T']:.3f} {q['L']/q['Lj']:.3f} "
                     f"{q['xcp_over_LK']:.3f} {q['xj_over_LK']:.3f} "
                     f"{(q['xcp_over_LK']-q['xj_over_LK'])*q['LK']:+.3f}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "slender.txt"), "w").write(txt + "\n")
    json.dump(res, open(os.path.join(OUT, "slender.json"), "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
