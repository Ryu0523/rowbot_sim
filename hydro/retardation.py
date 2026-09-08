#!/usr/bin/env python3
"""
M2 -- frequency domain to time domain.

The Cummins equation carries the fluid's memory as a convolution

    mu(t) = integral_0^t K(t-tau) nu(tau) dtau

which is useless inside an MPC: it needs the whole velocity history and is not
an ODE. The fix is a small linear system whose impulse response reproduces K,

    xr' = Ar xr + Br nu ,   mu = Cr xr

typically 2-6 states per coefficient pair. After that the whole vessel model is
an ODE and both the MPC and the RL loop can use it.

Ogilvie (1964) relations, deep water:

    K(t) = (2/pi) integral_0^inf B(w) cos(w t) dw
    A(w) = A_inf - (1/w) integral_0^inf K(t) sin(w t) dt
    B(w) =         integral_0^inf K(t) cos(w t) dt

A_inf is recovered from the second relation rather than read off the highest
frequency computed: evaluating it at every w and checking the spread is a free
quality check on the whole transform.
"""
import numpy as np
from scipy.linalg import svd, logm, eig


# ------------------------------------------------------------- Ogilvie pair
def taper(omegas, start_frac=0.65):
    """Raised-cosine taper bringing B(w) smoothly to zero at the top of the grid.

    The Ogilvie integral runs to infinity; a finite grid truncates it, and a
    hard cut leaves a step in the integrand whose transform is Gibbs ringing
    spread through K(t). The ringing is high-frequency, so a low-order
    realisation tries to fit it and comes out distorted. Tapering removes the
    step. It biases K slightly low near t=0, which is the honest trade: the
    alternative is ringing spread over the whole record.
    """
    omegas = np.asarray(omegas, float)
    w0 = start_frac * omegas.max()
    wgt = np.ones_like(omegas)
    hi = omegas > w0
    if hi.any():
        x = (omegas[hi] - w0) / (omegas.max() - w0)
        wgt[hi] = 0.5 * (1.0 + np.cos(np.pi * x))
    return wgt


def retardation_function(omegas, B, t, use_taper=True):
    """K(t) = (2/pi) int B(w) cos(w t) dw, by trapezoid over the w grid."""
    omegas = np.asarray(omegas, float)
    B = np.asarray(B, float) * (taper(omegas) if use_taper else 1.0)
    integrand = B[None, :] * np.cos(np.outer(t, omegas))
    return (2.0 / np.pi) * np.trapezoid(integrand, omegas, axis=1)


def infinite_added_mass(omegas, A, t, K, min_pts_per_period=12):
    """A_inf from A(w) + (1/w) int K(t) sin(w t) dt, evaluated at every w.

    Averaged over a band chosen so the estimate is trustworthy at both ends:

      low w   the 1/w factor amplifies any error in the integral, so the very
              lowest frequencies are excluded;
      high w  sin(w t) must be resolved on the t grid. Requiring at least
              `min_pts_per_period` samples per period caps the usable w at
              2*pi/(min_pts_per_period*dt).

    The spread across that band is the diagnostic that matters: a large spread
    means the w grid was truncated before B(w) had decayed, and A_inf (and K)
    are both contaminated.
    """
    omegas = np.asarray(omegas, float)
    dt = float(np.mean(np.diff(t)))
    integrand = K[None, :] * np.sin(np.outer(omegas, t))
    est = np.asarray(A, float) + np.trapezoid(integrand, t, axis=1) / omegas

    w_hi = min(2 * np.pi / (min_pts_per_period * dt), 0.9 * omegas.max())
    w_lo = max(1.0, 0.15 * w_hi)
    band = (omegas >= w_lo) & (omegas <= w_hi)
    use = est[band] if band.sum() > 3 else est
    return float(np.mean(use)), float(np.std(use) / max(abs(np.mean(use)), 1e-30))


def select_order(singular_values, tol=1e-3, max_order=10):
    """Pick the state-space order from the Hankel singular value spectrum.

    A retardation function that is genuinely a sum of a few damped modes has a
    spectrum that falls off a cliff at its true rank. Keeping states past that
    cliff fits numerical noise and, because A(w) is recovered as
    A_inf + Im[H]/w, spurious near-zero poles wreck the low-frequency end.
    """
    s = np.asarray(singular_values, float)
    s = s / max(s[0], 1e-30)
    keep = int(np.sum(s > tol))
    return int(np.clip(keep, 2, max_order))


def transfer_function(omegas, Ar, Br, Cr):
    """H(iw) = Cr (iw I - Ar)^-1 Br. Real part -> B(w); imaginary -> A(w)."""
    n = Ar.shape[0]
    I = np.eye(n)
    return np.array([Cr @ np.linalg.solve(1j * w * I - Ar, Br) for w in omegas])


# ------------------------------------------------- state-space realisation
def memory_span(K, dt, frac=0.01):
    """Time after which |K| has fallen below `frac` of its peak."""
    K = np.asarray(K, float)
    big = np.where(np.abs(K) > frac * max(np.max(np.abs(K)), 1e-30))[0]
    return float((big[-1] + 1) * dt) if len(big) else dt


def state_space_realization(K, dt, order=4, r_max=250, span_factor=2.5):
    """Eigensystem Realisation Algorithm on the sampled impulse response.

    Build a Hankel matrix from K, take its SVD, keep `order` singular values,
    and read the discrete-time triple off the factors. The singular value
    spectrum is itself informative: if it does not fall away sharply, K is not
    well represented by a low-order system.

    The Hankel window is sized in TIME, not in samples. K is sampled finely so
    the Ogilvie transform resolves the highest frequency, but a fixed sample
    count at that fine step would cover a fraction of a second -- far less than
    K's decay -- and the realisation would extrapolate from almost nothing.
    K is therefore decimated so the window spans a few multiples of the memory.
    """
    K = np.asarray(K, float)
    want = span_factor * memory_span(K, dt)
    stride = max(1, int(round(want / (2 * r_max * dt))))
    Kd, dtd = K[::stride], dt * stride

    N = len(Kd)
    r = min(N // 2, r_max)
    s = min(N - r - 1, r)

    idx = np.arange(r)[:, None] + np.arange(s)[None, :]
    H0 = Kd[idx]
    H1 = Kd[idx + 1]
    dt = dtd

    U, S, Vt = svd(H0, full_matrices=False)
    n = min(order, len(S))
    Un, Sn, Vn = U[:, :n], S[:n], Vt[:n, :].T
    S_half = np.diag(np.sqrt(Sn))
    S_ihalf = np.diag(1.0 / np.sqrt(Sn))

    Ad = S_ihalf @ Un.T @ H1 @ Vn @ S_ihalf
    Bd = (S_half @ Vn.T)[:, 0]
    Cd = (Un @ S_half)[0, :]

    # discrete -> continuous: Ad = exp(Ac dt), so Ac = log(Ad)/dt
    Ac = np.real(logm(Ad)) / dt
    lam = np.real(eig(Ac)[0])
    if lam.max() >= 0:                       # reflect any unstable pole
        w, V = eig(Ac)
        w = np.where(np.real(w) >= 0, -np.abs(np.real(w)) + 1j * np.imag(w), w)
        Ac = np.real(V @ np.diag(w) @ np.linalg.inv(V))

    return Ac, Bd, Cd, S


def prune_slow_modes(Ar, Br, Cr, t_mem, factor=3.0):
    """Drop modes far slower than the memory K actually shows.

    The realisation is exact on the true modes but also picks up one or two
    near-zero poles fitting the residue of truncation and tapering. A pole with
    a time constant several times longer than the decay of K cannot be real,
    and it does real damage: A(w) is recovered as A_inf + Im[H]/w, so a pole
    near the origin blows the low-frequency end of A apart.

    Modes are pruned in the modal basis and a real state space is rebuilt from
    the survivors: a 2x2 block per conjugate pair, 1x1 per real pole.
    """
    lam, V = eig(Ar)
    res = (Cr @ V) * (np.linalg.inv(V) @ Br)          # modal residues
    keep_rate = 1.0 / (factor * max(t_mem, 1e-9))

    blocks, Bs, Cs, used = [], [], [], np.zeros(len(lam), bool)
    for i, li in enumerate(lam):
        if used[i] or -np.real(li) < keep_rate:
            continue
        if abs(np.imag(li)) < 1e-9:                    # real mode
            used[i] = True
            blocks.append(np.array([[np.real(li)]]))
            Bs.append([1.0])
            Cs.append([np.real(res[i])])
        else:                                          # conjugate pair
            j = int(np.argmin(np.abs(lam - np.conj(li))))
            if used[j]:
                continue
            used[i] = used[j] = True
            s, w = np.real(li), abs(np.imag(li))
            r = res[i] if np.imag(li) > 0 else res[j]
            blocks.append(np.array([[s, w], [-w, s]]))
            Bs.append([1.0, 0.0])
            Cs.append([2 * np.real(r), 2 * np.imag(r)])

    if not blocks:
        return Ar, Br, Cr
    n = sum(b.shape[0] for b in blocks)
    A2 = np.zeros((n, n))
    k = 0
    for b in blocks:
        m = b.shape[0]
        A2[k:k + m, k:k + m] = b
        k += m
    return A2, np.concatenate(Bs), np.concatenate(Cs)


def impulse_response(Ar, Br, Cr, t):
    """Cr exp(Ar t) Br, evaluated by diagonalisation."""
    w, V = eig(Ar)
    Vi = np.linalg.inv(V)
    b = Vi @ Br
    c = Cr @ V
    return np.real(np.array([np.sum(c * np.exp(w * ti) * b) for ti in t]))


WAVE_BAND = (0.3, 2.5)          # SS4-5 encounter frequencies, rad/s


def fit_coefficient(omegas, A, B, order="auto", t_max=None, dt=None,
                    w_eval_min=0.2, w_eval_max=None, taper_start=0.65):
    """Full pipeline for one (i,j) coefficient pair, with its own diagnostics.

    `order="auto"` reads the order off the Hankel singular values.

    Errors are measured on w_eval_min <= w <= w_eval_max, and both bounds
    exist for a reason:

      below w_eval_min  A is recovered as A_inf + Im[H]/w, so dividing by a
                        tiny w amplifies residuals without saying anything
                        about model quality.
      above w_eval_max  the taper is deliberately pulling B down to zero, so
                        comparing the fit against the raw BEM B there measures
                        the taper, not the model. Defaults to the taper start.

    `err_B_band` / `err_A_band` are the numbers that actually matter: accuracy
    across the SS4-5 encounter frequencies.
    """
    omegas = np.asarray(omegas, float)
    d_omega = float(np.mean(np.diff(omegas)))
    t_max = t_max if t_max is not None else min(np.pi / d_omega, 30.0)
    dt = dt if dt is not None else min(np.pi / omegas.max() / 8, 0.02)
    t = np.arange(0.0, t_max, dt)

    K = retardation_function(omegas, B, t)
    A_inf, spread = infinite_added_mass(omegas, A, t, K)

    _, _, _, sv = state_space_realization(K, dt, order=2)
    n = select_order(sv) if order == "auto" else int(order)
    Ar, Br, Cr, sv = state_space_realization(K, dt, order=n)
    Ar, Br, Cr = prune_slow_modes(Ar, Br, Cr, memory_span(K, dt))

    K_ss = impulse_response(Ar, Br, Cr, t)
    scale = max(np.max(np.abs(K)), 1e-30)

    H = transfer_function(omegas, Ar, Br, Cr)
    B_fit = np.real(H)
    # H(iw) = int K e^{-iwt} dt, so Im[H] = -int K sin(wt) dt, and the Ogilvie
    # relation A = A_inf - (1/w) int K sin(wt) dt becomes A = A_inf + Im[H]/w.
    A_fit = A_inf + np.imag(H) / omegas
    bs = max(np.max(np.abs(B)), 1e-30)
    as_ = max(np.max(np.abs(A - A_inf)), 1e-30)
    w_hi = w_eval_max if w_eval_max is not None else taper_start * omegas.max()
    ev = (omegas >= w_eval_min) & (omegas <= w_hi)
    band = (omegas >= WAVE_BAND[0]) & (omegas <= WAVE_BAND[1])
    return dict(t=t, K=K, K_ss=K_ss, A_inf=A_inf, A_inf_spread=spread,
                Ar=Ar, Br=Br, Cr=Cr, singular_values=sv, order=n,
                B_fit=B_fit, A_fit=A_fit, w_eval_max=w_hi,
                err_K=float(np.max(np.abs(K_ss - K)) / scale),
                err_B=float(np.max(np.abs(B_fit - B)[ev]) / bs),
                err_A=float(np.max(np.abs(A_fit - A)[ev]) / as_),
                err_B_band=float(np.max(np.abs(B_fit - B)[band]) / bs),
                err_A_band=float(np.max(np.abs(A_fit - A)[band]) / as_),
                tail_at_wmax=float(np.abs(B[-1]) / bs),
                max_real_eig=float(np.max(np.real(eig(Ar)[0]))),
                memory_time=float(t[np.max(np.where(
                    np.abs(K) > 0.02 * scale)[0])]) if scale > 0 else 0.0)
