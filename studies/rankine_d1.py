#!/usr/bin/env python3
"""
Decision D1 (stage B3): stability, dispersion and damping of the steady
free-surface operators of hydro/rankine, from the COUPLED discrete system
(free-surface rows x flat-panel Rankine single layer), not from the 1-D
stencil symbol alone. Evidence for learn/meta/RANKINE_STAGE1_2026-10-06.md
section 8 and for the verification rows D1 of studies/rankine_verify.py.

    .venv/Scripts/python.exe studies/rankine_d1.py        (about 1 min, GPU if present)

Units: k0 = g/U^2 = 1, lambda0 = 2 pi, k = U^2/g = 1; grid dx = dy = lambda0/n.

1. Lattice symbol of the panel single layer. A free-surface panel row of
   constant-strength flat panels, collocated at the centroids of the uniform
   grid (port/starboard and rigid-lid images make the lattice the whole
   plane), maps sigma -> phi = -(1/2 pi) S sigma with symbol (Poisson sum)
       S(alpha, beta) = sum_pq 2 pi/|kappa_pq| sinc(alpha_p dx/2) sinc(beta_q dy/2),
   alpha_p = alpha + 2 pi p/dx, beta_q = beta + 2 pi q/dy. Checked against
   direct sums of the exact panel integrals (influence.py) to 4e-5 (tapered
   real-space sum). The main term carries the box filter sinc sinc =
   1 - kappa^2 dx^2/24 + ...; the aliases are O(theta^3). S is real and even
   for real (alpha, beta) and continues analytically to complex alpha with
   |Im alpha| < |beta|.

2. Coupled dispersion relation. Free-surface rows sum_m a_m sigma_{r-m} =
   k sum_m b_m phi_{r-m}/dx^2 (one-sided: index r and upstream) give
       F(alpha, beta) = A(alpha) + (k/2 pi) S C B(alpha) = 0,
   A = sum a_m e^{i m alpha dx}, B = sum b_m e^{i m alpha dx}/dx^2, C the
   box-correction symbol (1 + (4 sin^2(alpha dx/2) + 4 sin^2(beta dy/2))/24,
   or 1). Roots alpha(beta) near the exact alpha_c(beta)
   (k alpha^2 = sqrt(alpha^2 + beta^2)) by complex Newton: wavenumber error
   |kappa|/|kappa_c| - 1 and amplitude change per lambda0 downstream
   exp(2 pi Im alpha) - 1 (Im alpha > 0 grows downstream), per wave angle psi
   (kappa = sec^2 psi). Dawson-type: a = (1), b = stencil weights.
   Shifted Numerov (production since B3): a = (1, 10, 1)/12,
   b = (1, -2, 1): A and B carry the same factor e^{i alpha dx} (the shift),
   so F e^{-i alpha dx} is real on the real axis: every root is neutral.

3. Stability ("interval of periodicity"). For short diverging waves the
   coupling is local, sigma = -(2 pi/(S C)) phi, and a row becomes the linear
   multistep method sum (b_m + z a_m) q^{-m} = 0 for y'' = -omega^2 y with
   z = omega^2 h^2 = 2 pi dx^2/(k S C). Downstream stability needs every root
   |q| <= 1 for every z the grid contains; z_max(n) = max over the Brillouin
   zone. Dahlquist (1978): unconditional stability (all z > 0) only for
   order <= 2. Reported: the stability interval of each row, z_max(n), the
   blended entry-zone rows (a = (1 - l)(1, 10, 1)/12 + l (1, 0, 0),
   l = 1 - w, z up to z_max/w) and the beach rows (Rayleigh term).

4. Raised panels with collocation shift (Jensen et al. 1986, Raven 1996),
   considered and not chosen: panels raised by h, analytic derivatives,
   F = sum_pq (k alpha_p^2 - |kappa_pq|) e^{-|kappa_pq| h}/|kappa_pq| sinc sinc,
   also real (neutral); its error is the alias sum ~ 2 n e^{-2 pi h/dx} x sinc.
   Reported per h/dx for comparison; it would need a new representation of
   the free surface (sources above z = 0, no rigid-lid image), which the
   shifted Numerov rows make unnecessary.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PSIS = (0.0, 20.0, 35.26, 45.0, 55.0, 65.0)


def _torch():
    import torch
    return torch, ("cuda" if torch.cuda.is_available() else "cpu")


def lattice_symbol(alpha, beta, dx, dy, P=400, raised=None, k=1.0):
    """S(alpha, beta) (module docstring item 1) for an array of complex
    alpha and scalar beta; with raised=h the raised-panel symbol of item 4
    instead (times 2 pi, so that F = sum ... matches)."""
    torch, dev = _torch()
    a = torch.as_tensor(np.atleast_1d(np.asarray(alpha, complex)), device=dev)
    p = torch.arange(-P, P + 1, device=dev, dtype=torch.float64)
    bq = beta + 2 * np.pi * p / dy
    u = bq * dy / 2
    sq = torch.where(u == 0, torch.ones_like(u), torch.sin(u) / torch.where(u == 0, torch.ones_like(u), u))
    out = []
    for i0 in range(0, len(a), 8):
        ap = a[i0:i0 + 8, None] + 2 * np.pi * p[None] / dx
        v = ap * dx / 2
        sp = torch.where(v == 0, torch.ones_like(v), torch.sin(v) / torch.where(v == 0, torch.ones_like(v), v))
        kap = torch.sqrt(ap[:, :, None] ** 2 + (bq ** 2)[None, None, :])
        if raised is None:
            term = 2 * np.pi / kap * sp[:, :, None] * sq[None, None, :]
        else:
            term = ((k * ap[:, :, None] ** 2 - kap) * torch.exp(-kap * raised) / kap
                    * sp[:, :, None] * sq[None, None, :])
        out.append(term.sum(dim=(1, 2)))
    return torch.cat(out).cpu().numpy()


def box_symbol(alpha, beta, dx, dy):
    a = np.asarray(alpha, complex)
    return 1 + (4 * np.sin(a * dx / 2) ** 2 + 4 * np.sin(beta * dy / 2) ** 2) / 24


def row_weights(scheme):
    """(a, b) of a one-sided free-surface row (module docstring item 2)."""
    from hydro.rankine.steady import upwind_weights
    if scheme == "numerov_shift":
        return np.array([1.0, 10.0, 1.0]) / 12, np.array([1.0, -2.0, 1.0])
    if scheme == "central_shift":
        return np.array([0.0, 1.0, 0.0]), np.array([1.0, -2.0, 1.0])
    return np.array([1.0]), upwind_weights(scheme)


def coupled_F(alpha, beta, dx, dy, scheme, box, k=1.0):
    a_w, b_w = row_weights(scheme)
    al = np.atleast_1d(np.asarray(alpha, complex))
    S = lattice_symbol(al, beta, dx, dy)
    C = box_symbol(al, beta, dx, dy) if box else 1.0
    A = sum(am * np.exp(1j * al * m * dx) for m, am in enumerate(a_w))
    B = sum(bm * np.exp(1j * al * m * dx) for m, bm in enumerate(b_w)) / dx ** 2
    return A + k / (2 * np.pi) * S * C * B


def alpha_exact(beta, k=1.0):
    return np.sqrt((1 + np.sqrt(1 + 4 * k * k * beta ** 2)) / (2 * k * k))


def coupled_root(beta, dx, dy, scheme, box, a0=None, it=60):
    """Complex Newton (secant derivative) from the exact root."""
    a = complex(alpha_exact(beta) if a0 is None else a0)
    for _ in range(it):
        h = 1e-7 * max(1.0, abs(a))
        f0, f1 = coupled_F([a, a + h], beta, dx, dy, scheme, box)
        da = -f0 * h / (f1 - f0)
        a += da
        if abs(da) < 1e-14 * abs(a):
            break
    return a


def dispersion_table(n, scheme, box, psis=PSIS):
    """[(psi, wavenumber error, amplitude change per lambda0)] for the wave
    angles that exist on the grid (beta < pi/dy)."""
    dx = dy = 2 * np.pi / n
    out = []
    for psi in psis:
        ps = np.radians(psi)
        beta = np.sin(ps) / np.cos(ps) ** 2
        if beta >= np.pi / dy:
            continue
        a = coupled_root(beta, dx, dy, scheme, box)
        ac = alpha_exact(beta)
        out.append((psi, float(np.hypot(a.real, beta) / np.hypot(ac, beta) - 1),
                    float(np.exp(2 * np.pi * a.imag) - 1)))
    return out


def max_growth_on_grid(n, scheme, box, nb=40):
    """Largest amplitude change per lambda0 of the physical root over all
    beta in [0, pi/dy) (every lateral mode of the grid)."""
    dx = dy = 2 * np.pi / n
    worst = -np.inf
    for beta in np.linspace(0.0, 0.98 * np.pi / dy, nb):
        a = coupled_root(beta, dx, dy, scheme, box)
        worst = max(worst, float(np.exp(2 * np.pi * a.imag) - 1))
    return worst


def z_max(n, box, ng=31):
    """Largest z = 2 pi dx^2/(k S C) on the grid (k = 1, dx = dy)."""
    dx = dy = 2 * np.pi / n
    zm = 0.0
    for b in np.linspace(0.0, np.pi / dy, ng):
        al = np.linspace(0.02, np.pi / dx, ng)
        S = lattice_symbol(al, b, dx, dy, P=200).real
        C = box_symbol(al, b, dx, dy).real if box else 1.0
        zm = max(zm, float(np.max(2 * np.pi * dx ** 2 / (S * C))))
    return zm


def row_roots(a, b, z, nudx=0.0):
    """Roots q of sum_m (b_m + z a_m) q^{-m} = 0; nudx adds the beach's
    central Rayleigh term (b_0 += nu dx/2, b_2 -= nu dx/2)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float).copy()
    if nudx:
        b[0] += nudx / 2
        b[2] -= nudx / 2
    M = max(len(a), len(b))
    c = np.zeros(M)
    c[:len(b)] += b
    c[:len(a)] += z * a
    return np.roots(c)


def stability_interval(scheme, zmax=10.0, nz=20001):
    """Largest Z with max |q| <= 1 + 1e-12 for all sampled z in (0, Z]
    (samples log-spaced from 1e-8 to 1e-2, then linear to zmax; inf if
    stable on the whole range; 0 if already unstable at z = 1e-8)."""
    a, b = row_weights(scheme)
    zs = np.concatenate([np.logspace(-8, -2, 61), np.linspace(1e-2, zmax, nz)])
    last = 0.0
    for z in zs:
        if np.max(np.abs(row_roots(a, b, z))) > 1 + 1e-12:
            return float(last)
        last = z
    return float("inf")


def max_root_at(scheme, z):
    a, b = row_weights(scheme)
    return float(np.max(np.abs(row_roots(a, b, z))))


def blend_max_root(zgrid, nl=201):
    """max |q| over the blended entry rows: l in [0, 1), w = 1 - l,
    z in (0, zgrid/w] (log-spaced up to 1e8 as w -> 0)."""
    a_n, b = row_weights("numerov_shift")
    worst = 0.0
    for lam in np.linspace(0.0, 0.999999, nl):
        a = (1 - lam) * a_n + lam * np.array([1.0, 0.0, 0.0])
        zs = np.concatenate([np.linspace(1e-6, zgrid / (1 - lam), 200), np.logspace(0, 8, 50)])
        zs = zs[zs <= zgrid / (1 - lam)]
        for z in zs:
            worst = max(worst, float(np.max(np.abs(row_roots(a, b, z)))))
    return worst


def beach_max_root(n, zgrid, decay=6.0, wl=2.0):
    """max |q| over the beach rows (pure Numerov mass, Rayleigh term up to
    nu_max dx = 3 decay/(wl lambda0) * lambda0/n)."""
    a, b = row_weights("numerov_shift")
    worst = 0.0
    for nudx in np.linspace(0.0, 3 * decay / wl / n, 21):
        for z in np.linspace(1e-6, zgrid, 400):
            worst = max(worst, float(np.max(np.abs(row_roots(a, b, z, nudx)))))
    return worst


def raised_error(n, h_over_dx, psi):
    """Wavenumber error of the raised-panel shifted scheme (item 4)."""
    from scipy.optimize import brentq
    dx = dy = 2 * np.pi / n
    ps = np.radians(psi)
    beta = np.sin(ps) / np.cos(ps) ** 2
    ac = alpha_exact(beta)
    f = lambda a: lattice_symbol([a], beta, dx, dy, raised=h_over_dx * dx)[0].real
    a = brentq(f, 0.9 * ac, min(1.1 * ac, np.pi / dx), xtol=1e-14)
    return float(np.hypot(a, beta) / np.hypot(ac, beta) - 1)


def report():
    print("== 3. stability intervals of single rows (z = omega^2 h^2):")
    z20, z30 = z_max(20, False), z_max(30, False)
    for s in ("dawson4", "upwind5", "upwind6", "upwind7", "central_shift", "numerov_shift"):
        print(f"   {s:14s} stable for z < {stability_interval(s):.4g}; max |q| at z_max(30) = {z30:.3f}: "
              f"{max_root_at(s, z30):.4f}, at z_max(20) = {z20:.3f}: {max_root_at(s, z20):.4f}")
    for n in (12, 20, 30, 45, 60):
        print(f"   grid z_max: n={n}: no box {z_max(n, False):.3f}, box {z_max(n, True):.3f}")
    zg = z_max(20, True)
    print(f"   blended entry rows, z up to z_max(20)/w: max |q| = {blend_max_root(zg):.15f}")
    print(f"   beach rows n=20: max |q| = {beach_max_root(20, zg):.15f}")
    print("== 2. coupled dispersion (psi: wavenumber error % / amplitude change per lambda0 %)")
    for n in (20, 30, 45):
        for s, box in (("upwind5", False), ("upwind6", False), ("upwind7", False),
                       ("central_shift", True), ("numerov_shift", False), ("numerov_shift", True)):
            rows = dispersion_table(n, s, box)
            print(f"   n={n} {s:14s} box={int(box)} " + " ".join(
                f"{p:4.0f}:{100 * e:+.3f}/{100 * g:+.2f}" for p, e, g in rows))
        print(f"   n={n} numerov_shift box=1: max growth over all beta {max_growth_on_grid(n, 'numerov_shift', True):.2e}"
              f"; upwind5 {max_growth_on_grid(n, 'upwind5', False):.3f}")
    print("== 4. raised panels + shift (wavenumber error %, n = 30)")
    for h in (0.5, 1.0, 1.5, 2.0):
        print(f"   h/dx={h}: " + " ".join(f"{p:.0f}:{100 * raised_error(30, h, p):+.4f}" for p in (0, 35.26, 55)))


if __name__ == "__main__":
    report()
