"""
Exact steady Kelvin Green function (deep water, linear Neumann-Kelvin
free-surface condition) and a body-only boundary-integral solve with it.

Purpose: an independent verification of the Rankine free-surface solution
for submerged bodies (check S2K of studies/rankine_verify.py, added after the
B3 review of 2026-10-06; stage note learn/meta/RANKINE_STAGE1_2026-10-06.md
section 8.8). The Kelvin function satisfies the linear free-surface condition
and the radiation condition exactly, so a body-only solve needs no
free-surface panels, no entry zone, beach, domain or free-surface grid: it
shares with the Rankine solver only the body mesh and the rigid-lid Rankine
panel integrals of influence.py (verified in C1-C8). Re-implemented here
from the reviewer's derivation (written out below and in the stage note);
the derivation is the classical one (Havelock 1932; Wehausen & Laitone 1960,
section 13; Newman 1987, J. Ship Res. 31:1 for the E1 form). Nothing is
copied from WaveBEM or any other code.

Derivation. Frame fixed to the body, stream -U x, fluid z < 0, steady
condition U^2 phi_xx + g phi_z = 0 on z = 0, waves downstream (Rayleigh
friction -mu U phi_x, mu -> 0+), k0 = g/U^2. A unit source at xi (xi3 < 0),
normalised like -1/(4 pi r):

    G(x; xi) = -(1/4 pi)(1/r + 1/r') + G_W(X, Y, Z),
    X = x1 - xi1, Y = x2 - xi2, Z = x3 + xi3 (< 0), r' = |x - (xi1, xi2, -xi3)|.

The rigid-lid part has G_z = 0 on z = 0, so on z = 0 the wave part must
satisfy U^2 G_W,xx + g G_W,z = (U^2/2 pi) d^2/dx^2 (1/r0), r0 = r = r'.
With 1/r0 = (1/2 pi) int int e^{kappa xi3} e^{i(alpha X + beta Y)}/kappa,
kappa = |(alpha, beta)|, a decaying harmonic G_W = int int A e^{kappa z} ...
gives

    G_W = -(1/4 pi^2) int int alpha^2 e^{kappa Z} e^{i(alpha X + beta Y)}
                               / (kappa (k0 kappa - alpha^2 - i eps alpha)).

Split alpha^2/(k0 kappa - alpha^2) = -1 + k0 kappa/(k0 kappa - alpha^2): the
first part is 1/(2 pi r'). In polar form (alpha, beta) = kappa (cos t, sin t),
w = X cos t + Y sin t, k0t = k0 sec^2 t, the pole kappa = k0t - i eps sec t
lies below the real kappa axis for cos t > 0, so 1/(k0t - kappa) -> PV +
i pi sgn(cos t) delta(kappa - k0t); the halves t and t + pi are complex
conjugates, so

    G_W = 1/(2 pi r') - (k0/2 pi^2) Re int_{-pi/2}^{pi/2} sec^2 t
            [PV int_0^inf e^{kappa (Z + i w)}/(k0t - kappa) d kappa + i pi e^{k0t (Z + i w)}] dt.

The PV integral is -e^z [E1(z) + i pi sgn w], z = k0t (Z + i w), with E1
the principal exponential integral (the bracket is E1 continued across its
cut, continuous at w = 0), and 1/(2 pi r') = (1/2 pi^2) Re int 1/sigma dt,
sigma = -Z - i w. Hence

    G_W = (1/(2 pi^2)) Re int_{-pi/2}^{pi/2} F dt,
    F = 1/sigma + k0t e^z [E1(z) + i pi sgn w] - i pi k0t e^z,           (K)
    F' = dF/d sigma = -1/sigma^2 - k0t F,  F'' = 2/sigma^3 - k0t F',
    d sigma/dX = -i cos t, d sigma/dY = -i sin t, d sigma/dZ = -1.

For w >= 0, F = A - B, and for w < 0, F = conj(A + B), with
A = 1/sigma_u + k0t h(z_u), B = i pi k0t e^{z_u}, sigma_u = -Z - i|w|,
z_u = k0t (Z + i|w|) (Im z_u >= 0) and h(z) = e^z (E1(z) + i pi): the
oscillating term 2 Re(B) = -2 pi k0t e^{k0t Z} sin(k0t |w|) enters with
opposite signs on the two sides, so the waves (the e^z term without the
1/z decay) appear only for w < 0, i.e. downstream (radiation condition).
Limits: k0 -> 0 gives G_W -> 1/(2 pi r') (phi = 0 on z = 0); k0 -> inf gives
G_W -> 0 (rigid lid). Lagally's theorem at a source of flux Q at depth f
gives R = rho Q^2 G_W,X(0, 0, -2 f) = (rho k0^2 Q^2/pi) int_0^{pi/2}
sec^3 t e^{-2 k0 f sec^2 t} dt, Havelock's result (checked numerically,
S2K-G rows).

Numerics. E1 on the closed upper-left quadrant (Re z < 0, Im z >= 0; our z
has Re z = k0t Z < 0): power series near the negative real axis (|z| - |Re z|
<= 3, so |z| < 43; rounding grows like e^(|z| - |Re z|)), continued fraction
e^z E1(z) = 1/(z+1- 1/(z+3- 4/(z+5- ...))) (50 levels, backward) elsewhere, asymptotic series e^z E1(z) ~ sum (-1)^n n!/z^{n+1} for Re z <= -40
(where e^z < 5e-18 makes the branch terms irrelevant), with the leading term
cancelled analytically against 1/sigma. Accuracy against mpmath: e1_check().
The t integral: Gauss-Legendre with `nth` nodes on (-pi/2, pi/2) (the
integrand is analytic there and vanishes like cos^2 t at the ends); for the
spheroid body pairs (|X| <= 10, |Z| >= 3 m) the body-solve resistance changes
by < 1e-11 from 64 to 256 nodes (exploration B3R-E1, row S2K-G), so 64 is
the default.
"""
import math

import numpy as np
import torch

EULER = 0.57721566490153286061


def _series_A(z, sig_u, k0t, nterms):
    """A = 1/sigma_u + k0t e^z (E1(z) + i pi) by the power series of E1."""
    mz = -z
    term = torch.ones_like(z)
    S = torch.zeros_like(z)
    for n in range(1, nterms + 1):
        term = term * mz / n
        S = S + term / n
    E1p = -EULER - torch.log(z) + 1j * math.pi - S
    return 1.0 / sig_u + k0t * torch.exp(z) * E1p


def _cf_A(z, sig_u, k0t, depth):
    """A by the continued fraction of e^z E1(z), evaluated backwards."""
    t = torch.zeros_like(z)
    for n in range(depth, 0, -1):
        t = (n * n) / (z + (2 * n + 1) - t)
    eE1 = 1.0 / (z + 1.0 - t)
    return 1.0 / sig_u + k0t * (eE1 + 1j * math.pi * torch.exp(z))


def _asym_A(z, k0t, nterms=35):
    """A for Re z <= -40: the n = 0 term of the asymptotic series of e^z E1
    cancels 1/sigma_u exactly (k0t/z = -1/sigma_u); the rest is summed."""
    zi = 1.0 / z
    term = zi.clone()
    S = torch.zeros_like(z)
    for n in range(1, nterms + 1):
        term = term * (-n) * zi
        S = S + term
    return k0t * (S + 1j * math.pi * torch.exp(z))


def A_upper(z, sig_u, k0t, cf_depth=50):
    """A = 1/sigma_u + k0t e^z (E1(z) + i pi) for Re z < 0 <= Im z
    (module docstring), region by region."""
    out = torch.empty_like(z)
    re, ab = z.real, z.abs()
    asym = re <= -40.0
    ser = (~asym) & ((ab + re) <= 3.0)                     # |z| - |Re z| <= 3, hence |z| < 43
    cf = ~(asym | ser)
    k0t = torch.as_tensor(k0t, dtype=torch.float64, device=z.device)
    k0t = k0t.expand_as(re) if k0t.dim() else k0t * torch.ones_like(re)
    if asym.any():
        out[asym] = _asym_A(z[asym], k0t[asym])
    if ser.any():
        idx = torch.nonzero(ser).squeeze(1)
        zs = z[idx]
        abs_s = ab[idx]
        for lo, hi, nt in ((0.0, 10.0, 60), (10.0, 20.0, 95), (20.0, 32.0, 125), (32.0, 44.0, 160)):
            m = (abs_s > lo) & (abs_s <= hi)
            if m.any():
                ii = idx[m]
                out[ii] = _series_A(zs[m], sig_u[ii], k0t[ii], nt)
    if cf.any():
        out[cf] = _cf_A(z[cf], sig_u[cf], k0t[cf], cf_depth)
    return out


def gauss_theta(nth):
    t, w = np.polynomial.legendre.leggauss(nth)
    return t * np.pi / 2, w * np.pi / 2


def wave_part(X, Y, Z, k0, nth=64, want=("G", "X", "Y", "Z"), device=None, chunk=4_000_000):
    """G_W and its derivatives at (X, Y, Z) (any shape, Z < 0), formula (K).
    want: any of "G", "X", "Y", "Z", "XX". Returns dict of numpy arrays
    (inputs numpy) or torch tensors (inputs torch, on their device)."""
    as_np = not torch.is_tensor(X)
    dev = device or (X.device if torch.is_tensor(X) else ("cuda" if torch.cuda.is_available() else "cpu"))
    Xt, Yt, Zt = (torch.as_tensor(np.asarray(a, float) if as_np else a, dtype=torch.float64, device=dev).reshape(-1)
                  for a in (X, Y, Z))
    shape = np.shape(X) if as_np else tuple(X.shape)
    assert bool((Zt < 0).all()), "Z = x3 + xi3 must be < 0"
    th, wt = gauss_theta(nth)
    out = {k: torch.zeros_like(Xt) for k in want}
    for i0 in range(0, Xt.numel(), chunk):
        sl = slice(i0, min(Xt.numel(), i0 + chunk))
        x, y, zz = Xt[sl], Yt[sl], Zt[sl]
        acc = {k: torch.zeros_like(x) for k in want}
        for t, wq in zip(th, wt):
            c, s = math.cos(t), math.sin(t)
            k0t = k0 / (c * c)
            w = x * c + y * s
            aw = w.abs()
            sig_u = torch.complex(-zz, -aw)
            z = torch.complex(k0t * zz, k0t * aw)
            A = A_upper(z, sig_u, k0t)
            B = 1j * math.pi * k0t * torch.exp(z)
            F = torch.where(w >= 0, A - B, torch.conj(A + B))
            if "G" in want:
                acc["G"] += wq * F.real
            if any(k in want for k in ("X", "Y", "Z", "XX")):
                sig = torch.complex(-zz, -w)
                Fp = -1.0 / sig ** 2 - k0t * F
                if "X" in want:
                    acc["X"] += wq * (Fp * (-1j * c)).real
                if "Y" in want:
                    acc["Y"] += wq * (Fp * (-1j * s)).real
                if "Z" in want:
                    acc["Z"] += wq * (-Fp).real
                if "XX" in want:
                    Fpp = 2.0 / sig ** 3 - k0t * Fp
                    acc["XX"] += wq * (-c * c * Fpp).real
        for k in want:
            out[k][sl] = acc[k] / (2 * math.pi ** 2)
    if as_np:
        return {k: v.cpu().numpy().reshape(shape) for k, v in out.items()}
    return {k: v.reshape(shape) for k, v in out.items()}


def e1_check(n=4000, seed=0, dps=40):
    """Max relative error of A_upper (through h(z) = e^z (E1(z) + i pi))
    against mpmath on random z in the region used (Re z in [-200, -0.5],
    Im z in [0, 400], plus points on the real axis and near the region
    boundaries). Returns (max rel error of h, number of points)."""
    import mpmath as mp
    mp.mp.dps = dps
    rng = np.random.default_rng(seed)
    re = -np.concatenate([10 ** rng.uniform(np.log10(0.5), np.log10(200), n), rng.uniform(0.5, 60, n // 4)])
    im = np.concatenate([10 ** rng.uniform(-3, np.log10(400), n), np.zeros(n // 4)])
    im[: n // 10] = 0.0
    zz = re + 1j * im
    ref = np.array([complex(mp.exp(mp.mpc(v)) * (mp.e1(mp.mpc(v.real, v.imag)) + mp.mpc(0, mp.pi))
                            if v.imag > 0 else
                            mp.exp(v.real) * (-mp.ei(-v.real)))  # h on the real axis: e^x (-Ei(-x))
                    for v in zz])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = torch.as_tensor(zz, dtype=torch.complex128, device=dev)
    # A = 1/sigma + k0t h with k0t = 1, sigma = -z: h = A - 1/sigma = A + 1/z
    sig = -z
    h = (A_upper(z, sig, 1.0) - 1.0 / sig).cpu().numpy()
    err = np.abs(h - ref) / np.abs(ref)
    return float(err.max()), len(zz), zz[int(np.argmax(err))]


# --------------------------------------------------------- body-only solve
def body_kelvin_matrices(body, k0, nth=64, device=None):
    """S_W, D_W (N x N, numpy) of the wave part for the half body `body`
    (y >= 0 half, port/starboard mirror included, phi even in y), centroid
    rule on each panel (G_W is smooth on the body: Z <= -2 * min depth):
        S_W[i, j] = -4 pi a_j sum_m G_W(x_i - xi_j^m),
        D_W[i, j] = -4 pi a_j sum_m n_j^m . grad_xi G_W(x_i; xi_j^m),
    m = the panel and its y mirror, grad_xi = (-d/dX, -d/dY, +d/dZ)."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    c = torch.as_tensor(body.c, dtype=torch.float64, device=dev)
    n = torch.as_tensor(body.n, dtype=torch.float64, device=dev)
    a = torch.as_tensor(body.area, dtype=torch.float64, device=dev)
    N = len(body)
    SW = torch.zeros((N, N), dtype=torch.float64, device=dev)
    DW = torch.zeros((N, N), dtype=torch.float64, device=dev)
    for m in (1.0, -1.0):
        X = c[:, None, 0] - c[None, :, 0]
        Y = c[:, None, 1] - m * c[None, :, 1]
        Z = c[:, None, 2] + c[None, :, 2]
        r = wave_part(X, Y, Z, k0, nth, ("G", "X", "Y", "Z"), dev)
        SW += -4 * math.pi * r["G"] * a[None]
        dn = -r["X"] * n[None, :, 0] - r["Y"] * (m * n[None, :, 1]) + r["Z"] * n[None, :, 2]
        DW += -4 * math.pi * dn * a[None]
        del r, X, Y, Z
    return SW.cpu().numpy(), DW.cpu().numpy()


def solve_body(body, U, g, phin=None, nth=64, device=None):
    """Neumann-Kelvin potential on a submerged body by Green's identity with
    the Kelvin function: (2 pi I - D_R - D_W) phi = -(S_R + S_W) dphi/dn,
    S_R, D_R the Rankine panel integrals with the y and rigid-lid images
    (bie.ImageOperator, the same blocks as the Rankine steady solver). No
    free-surface panels: the free-surface integral of Green's identity
    vanishes because phi and G satisfy the same linear condition and the
    radiation conditions (submerged body, no waterline term)."""
    from .bie import ImageOperator, lu_solve
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if phin is None:
        phin = U * body.n[:, 0]
    op = ImageOperator(body, axes=(1, 2), device=dev, parities=[(1, 1)])
    S, D = np.asarray(op.S((1, 1))), np.asarray(op.D((1, 1)))
    del op
    SW, DW = body_kelvin_matrices(body, g / U ** 2, nth, dev)
    M = 2 * np.pi * np.eye(len(body)) - D - DW
    phi = lu_solve(M, -(S + SW) @ phin, dev)
    return phi
