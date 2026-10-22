"""
Independent reference for the high-speed limit of the planing solver (stage
B7 gate, check M4-HF of learn/meta/RANKINE_STAGE1_2026-10-06.md, section
13.4): the classical vortex-lattice method for a thin flat wing in an
unbounded fluid (horseshoe vortices, bound segment at the panel 1/4 chord,
collocation at the 3/4 chord, trailing legs along +x; Falkner 1943, Katz &
Plotkin 2001 ch. 12). In the limit C_V -> inf the linear planing problem is
the lower half of the thin-wing problem for the same planform (phi = 0 on
the free surface outside the wing and its wake = antisymmetry about z = 0),
so the planing lift is HALF the wing lift (Tulin 1957; Wang & Rispin 1971).
Shares no code with planing_kernel / planing_steady (Biot-Savart in physical
space instead of the Fourier transfer function).
"""
import math

import numpy as np


def _seg(P, A, B):
    """Velocity at points P (n,3) induced by unit-circulation segments A->B
    (m,3); returns (n, m, 3)."""
    r1 = P[:, None, :] - A[None, :, :]
    r2 = P[:, None, :] - B[None, :, :]
    r0 = B - A
    c = np.cross(r1, r2)
    c2 = np.sum(c * c, axis=-1)
    n1 = np.linalg.norm(r1, axis=-1)
    n2 = np.linalg.norm(r2, axis=-1)
    dot = np.sum(r0[None] * (r1 / n1[..., None] - r2 / n2[..., None]), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(c2 > 1e-24, dot / (4 * math.pi * c2), 0.0)
    return k[..., None] * c


def planing_lift_hf(y_edges_half, chords, alpha, nc=40, spacing="cos", far=1e4):
    """Lift / (rho U^2 b^2) of a planing surface at C_V -> inf with the
    planform given by half-beam strip edges (0..1/2, units b) and the wetted
    chord of each strip (transom at x = 0, chord upstream): half the lift of
    the thin wing of that planform at incidence alpha (small angle). Also the
    centre of pressure from the transom."""
    ye = np.asarray(y_edges_half, float)
    edges = np.concatenate([-ye[::-1], ye[1:]])           # full span
    ch = np.asarray(chords, float)
    chf = np.concatenate([ch[::-1], ch])
    A, B, C, dy, xb = [], [], [], [], []
    for j in range(len(edges) - 1):
        c = chf[j]
        if spacing == "cos":
            s = 0.5 * (1 - np.cos(np.pi * np.arange(nc + 1) / nc))
        else:
            s = np.arange(nc + 1) / nc
        xs = -c + c * s                                   # LE at -c, TE at 0 (flow +x)
        for i in range(nc):
            x0, x1 = xs[i], xs[i + 1]
            xq = x0 + 0.25 * (x1 - x0)
            xc = x0 + 0.75 * (x1 - x0)
            A.append([xq, edges[j], 0.0])
            B.append([xq, edges[j + 1], 0.0])
            C.append([xc, 0.5 * (edges[j] + edges[j + 1]), 0.0])
            dy.append(edges[j + 1] - edges[j])
            xb.append(xq)
    A, B, C = np.array(A), np.array(B), np.array(C)
    Af = A.copy(); Af[:, 0] = far
    Bf = B.copy(); Bf[:, 0] = far
    # horseshoe: far->A (trailing leg, reversed), A->B bound, B->far
    v = _seg(C, Af, A) + _seg(C, A, B) + _seg(C, B, Bf)
    w = v[:, :, 2]
    G = np.linalg.solve(w, -np.full(len(C), alpha))      # downwash -U alpha at collocation (U = 1)
    lift_wing = np.sum(G * np.array(dy))                  # rho U Gamma dy / (rho U^2 b^2)
    mom = np.sum(G * np.array(dy) * (-np.array(xb)))      # about the transom, lever = distance upstream
    lift = 0.5 * lift_wing
    return dict(lift=abs(lift), CL=2 * abs(lift), lp=mom / lift_wing)
