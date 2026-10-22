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


def retardation_function(omegas, B, t, use_taper=True, n_fine=None, extend_to_zero=False):
    """K(t) = (2/pi) int B(w) cos(w t) dw, by trapezoid over the w grid.

    `n_fine` (opt-in, default None = the original path): interpolate the
    (tapered) B with a monotone cubic onto `n_fine` uniform points between the
    first and last grid frequency before the trapezoid. The plain trapezoid
    over a grid of step dw is a finite cosine SUM and therefore periodic in t
    with period 2 pi / dw (62.8 s for dw = 0.1 rad/s): any window longer than
    pi / dw sees the mirror image of the early kernel instead of the tail.
    The fine grid pushes that period out to 2 pi / dw_fine; what it returns
    beyond pi / dw is the kernel of the interpolant, not new information.
    `extend_to_zero` (with n_fine): integrate from w = 0 with B held at its
    first tabulated value below omegas[0]. Without it the integral starts at
    omegas[0] and the hard edge there adds (2/pi) B(w0) sin(w0 t) / t to K: a
    ripple at the lowest tabulated frequency that masks the physical tail.
    """
    omegas = np.asarray(omegas, float)
    B = np.asarray(B, float) * (taper(omegas) if use_taper else 1.0)
    if n_fine is not None:
        from scipy.interpolate import PchipInterpolator
        w_start = 0.0 if extend_to_zero else omegas[0]
        wf = np.linspace(w_start, omegas[-1], int(n_fine))
        Bf = PchipInterpolator(omegas, B)(np.maximum(wf, omegas[0]))
        t = np.asarray(t, float)
        K = np.empty(len(t))
        step = max(1, 20000000 // max(len(wf), 1))        # bounded memory
        for k in range(0, len(t), step):
            K[k:k + step] = np.trapezoid(Bf[None, :] * np.cos(np.outer(t[k:k + step], wf)), wf, axis=1)
        return (2.0 / np.pi) * K
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
                    w_eval_min=0.2, w_eval_max=None, taper_start=0.65,
                    A_inf=None, max_order=10, n_fine=None, method="era",
                    eval_bands=None, extend_to_zero=False, **fd_options):
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

    Opt-in extensions (every default reproduces the original result exactly):

      A_inf       a number: use this infinite-frequency added mass (e.g. the
                  solver's own) instead of the time-domain estimate. The
                  estimate is still computed and returned as `A_inf_est`.
      max_order   cap for order="auto" (original 10).
      n_fine      fine-grid cosine transform (see retardation_function); needed
                  for any window longer than pi / d_omega. `extend_to_zero`
                  integrates from w = 0 with B held constant below the grid.
      method      "era" (original, time domain) or "vf": frequency-domain
                  vector fitting of H(iw) = B + i w (A - A_inf) with stability
                  and passivity enforced, see fit_coefficient_fd. `fd_options`
                  go to fit_coefficient_fd (weight, band, band_weight,
                  min_decay, passive, n_iter, ...). With "vf" an integer
                  `order` is required and A_inf should be given (if None the
                  linear term of the rational fit is used as A_inf).
      eval_bands  dict name -> (w_lo, w_hi): extra bands on which err_B/err_A
                  are reported (`err_bands`), normalised both by the global and
                  by the in-band maxima.
    """
    omegas = np.asarray(omegas, float)
    d_omega = float(np.mean(np.diff(omegas)))
    t_max = t_max if t_max is not None else min(np.pi / d_omega, 30.0)
    dt = dt if dt is not None else min(np.pi / omegas.max() / 8, 0.02)
    t = np.arange(0.0, t_max, dt)
    extended = (A_inf is not None or max_order != 10 or n_fine is not None
                or method != "era" or eval_bands is not None or extend_to_zero or fd_options)

    if method == "vf":
        f = fit_coefficient_fd(omegas, A, B, A_inf, order, t=t, taper_start=taper_start,
                               w_eval_min=w_eval_min, w_eval_max=w_eval_max, n_fine=n_fine,
                               eval_bands=eval_bands, extend_to_zero=extend_to_zero, **fd_options)
        return f

    K = retardation_function(omegas, B, t, n_fine=n_fine, extend_to_zero=extend_to_zero)
    A_inf_est, spread = infinite_added_mass(omegas, A, t, K)
    A_inf = A_inf_est if A_inf is None else float(A_inf)

    _, _, _, sv = state_space_realization(K, dt, order=2)
    n = select_order(sv, max_order=max_order) if order == "auto" else int(order)
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
    out = dict(t=t, K=K, K_ss=K_ss, A_inf=A_inf, A_inf_spread=spread,
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
    if extended:
        out["A_inf_est"] = A_inf_est
        out["method"] = "era"
        out["n_states"] = int(Ar.shape[0])
        out["passivity"] = passivity_margin(Ar, Br, Cr)
        out["err_bands"] = band_errors(omegas, A, B, A_fit, B_fit, A_inf, eval_bands)
    return out


# ------------------------------------------------- frequency-domain fitting
def band_errors(omegas, A, B, A_fit, B_fit, A_inf, eval_bands):
    """err_B / err_A on named bands, normalised by the global maxima (as the
    main errors) and by the in-band maxima (stricter: `*_local`)."""
    out = {}
    if not eval_bands:
        return out
    bs = max(np.max(np.abs(B)), 1e-30)
    as_ = max(np.max(np.abs(A - A_inf)), 1e-30)
    for name, (lo, hi) in eval_bands.items():
        m = (omegas >= lo - 1e-9) & (omegas <= hi + 1e-9)
        if not m.any():
            continue
        dB = np.abs(B_fit - B)[m]
        dA = np.abs(A_fit - A)[m]
        out[name] = dict(err_B=float(dB.max() / bs), err_A=float(dA.max() / as_),
                         err_B_local=float(dB.max() / max(np.max(np.abs(B[m])), 1e-30)),
                         err_A_local=float(dA.max() / max(np.max(np.abs(A[m] - A_inf)), 1e-30)),
                         n_points=int(m.sum()))
    return out


def passivity_margin(Ar, Br, Cr, w_max=None, n=4000):
    """min over w of Re H(iw) divided by max |Re H(iw)| on a dense grid from 0
    to w_max (default 20 x the fastest pole). A memory force -mu = -C x with
    K the impulse response dissipates energy for every velocity history iff
    Re H(iw) >= 0 for all w (positive real, D = 0); the ratio is the depth of
    the worst violation relative to the damping scale. For a block (MIMO)
    realisation the smallest eigenvalue of the Hermitian part is used."""
    lam = eig(Ar)[0]
    if w_max is None:
        w_max = 20.0 * max(np.max(np.abs(lam)), 1.0)
    w = np.concatenate([np.linspace(0.0, 2.0, n // 2), np.geomspace(2.0, w_max, n // 2)])
    Br = np.atleast_2d(Br.T).T if Br.ndim == 1 else Br
    Cr = np.atleast_2d(Cr)
    H = block_transfer_function(w, Ar, Br, Cr)                 # (nw, m_out, m_in)
    Hh = 0.5 * (H + np.conj(np.swapaxes(H, 1, 2)))
    ev = np.linalg.eigvalsh(Hh)                                # ascending
    return float(ev[:, 0].min() / max(np.abs(ev).max(), 1e-30))


def block_transfer_function(omegas, Ar, Br, Cr):
    """H(iw) = Cr (iw I - Ar)^-1 Br for a MIMO triple; returns (nw, m_out, m_in)."""
    n = Ar.shape[0]
    I = np.eye(n)
    Br = Br.reshape(n, -1)
    Cr = Cr.reshape(-1, n)
    return np.array([Cr @ np.linalg.solve(1j * w * I - Ar, Br) for w in omegas])


def _pole_basis(poles, s):
    """Real-coefficient partial-fraction basis at the points s (complex).

    `poles`: real poles and one member (Im > 0) of each complex pair. For a
    real pole p the basis function is 1/(s - p) with a real coefficient; for a
    pair (p, conj p) the two functions 1/(s-p) + 1/(s-conj p) and
    i/(s-p) - i/(s-conj p) carry the real and imaginary parts of the residue,
    so that  r/(s-p) + conj(r)/(s-conj p) = Re r * f1 + Im r * f2.
    Returns the (len(s), n_coef) complex matrix and the list of column kinds.
    """
    cols, kinds = [], []
    for p in poles:
        if abs(np.imag(p)) < 1e-12:
            cols.append(1.0 / (s - np.real(p)))
            kinds.append(("r", p))
        else:
            cols.append(1.0 / (s - p) + 1.0 / (s - np.conj(p)))
            cols.append(1j / (s - p) - 1j / (s - np.conj(p)))
            kinds.append(("ca", p))
            kinds.append(("cb", p))
    return np.array(cols).T, kinds


def _real_modal_sigma(poles, d):
    """Zeros of sigma(s) = 1 + sum_k d_k f_k(s): eig(A - b d^T) of the real
    modal realisation (block [[a, w], [-w, a]] with b = [2, 0] per pair)."""
    n = sum(1 if abs(np.imag(p)) < 1e-12 else 2 for p in poles)
    A = np.zeros((n, n))
    b = np.zeros(n)
    k = 0
    for p in poles:
        if abs(np.imag(p)) < 1e-12:
            A[k, k] = np.real(p)
            b[k] = 1.0
            k += 1
        else:
            a, w = np.real(p), np.imag(p)
            A[k:k + 2, k:k + 2] = [[a, w], [-w, a]]
            b[k] = 2.0
            k += 2
    return eig(A - np.outer(b, d))[0]


def _group_poles(z, min_decay):
    """Stable, grouped poles from a set of eigenvalues: real part forced to
    <= -min_decay, complex ones kept once (Im > 0), real ones as real."""
    z = np.asarray(z, complex)
    re = -np.maximum(np.abs(np.real(z)), min_decay)
    z = re + 1j * np.imag(z)
    out = []
    for p in z:
        if abs(np.imag(p)) < 1e-9 * max(1.0, abs(p)):
            out.append(complex(np.real(p), 0.0))
        elif np.imag(p) > 0:
            out.append(p)
    return out


def initial_poles(order, w_lo, w_hi, w_focus=None):
    """Starting poles for vector fitting: complex pairs with imaginary parts
    log-spaced over [w_lo, w_hi] and real part -1 % of it (the standard start);
    an odd order adds one real pole at w_lo. `w_focus` (a frequency) replaces
    the pair closest to it by a lightly damped pair right there."""
    n_pair = order // 2
    ims = np.geomspace(max(w_lo, 1e-3), w_hi, n_pair) if n_pair else np.array([])
    poles = [complex(-0.01 * w, w) for w in ims]
    if w_focus is not None and poles:
        k = int(np.argmin(np.abs(ims - w_focus)))
        poles[k] = complex(-0.02 * w_focus, w_focus)
    if order % 2:
        poles.append(complex(-max(w_lo, 1e-3), 0.0))
    return poles


def vector_fit(s, H, order, weight=None, n_iter=30, poles=None, min_decay=1e-2,
               fit_linear=False, tol=1e-6):
    """Vector fitting (Gustavsen & Semlyen 1999) of samples H(s_m), s = i w.

    Model  H(s) = sum_k r_k / (s - p_k) [+ E s],   real poles or conjugate pairs,
    real E (`fit_linear`): the i w A_inf term when A_inf is to be identified.

    Each iteration solves the linear least squares
        sum_k c_k f_k(s) [+ E s] - H(s) sum_k d_k f_k(s) = H(s)
    (weighted, real and imaginary parts stacked with their own weights when
    `weight` has shape (M, 2)), relocates the poles to the zeros of
    sigma(s) = 1 + sum d_k f_k(s), reflects them into the left half plane and
    clamps the decay rate at `min_decay` (a pole slower than that is slower
    than anything a table with step dw can resolve; dw / 10 is a fair floor).
    Converged poles are then used for a last linear fit of the residues.

    Returns dict(poles, residues (complex, one per listed pole), E, kinds,
    iterations, pole_change).
    """
    s = np.asarray(s, complex)
    H = np.asarray(H, complex)
    M = len(s)
    if weight is None:
        W = np.ones((M, 2))
    else:
        weight = np.asarray(weight, float)
        W = np.stack([weight, weight], 1) if weight.ndim == 1 else weight
    if poles is None:
        w = np.imag(s)
        poles = initial_poles(order, w[w > 0].min(), w.max())
    change = np.inf
    it = 0
    for it in range(1, n_iter + 1):
        Phi, kinds = _pole_basis(poles, s)
        nc = Phi.shape[1]
        cols = [Phi]
        if fit_linear:
            cols.append(s[:, None])
        cols.append(-H[:, None] * Phi)
        Mx = np.concatenate(cols, 1)
        Areal = np.concatenate([W[:, 0:1] * np.real(Mx), W[:, 1:2] * np.imag(Mx)], 0)
        rhs = np.concatenate([W[:, 0] * np.real(H), W[:, 1] * np.imag(H)])
        x = np.linalg.lstsq(Areal, rhs, rcond=None)[0]
        d = x[-nc:]
        new = _group_poles(_real_modal_sigma(poles, d), min_decay)
        old_arr = np.sort_complex(np.array(poles))
        new_arr = np.sort_complex(np.array(new))
        change = (np.max(np.abs(old_arr - new_arr)) / max(np.max(np.abs(old_arr)), 1e-30)
                  if len(old_arr) == len(new_arr) else np.inf)
        poles = new
        if change < tol:
            break
    # residues with the converged poles
    Phi, kinds = _pole_basis(poles, s)
    cols = [Phi] + ([s[:, None]] if fit_linear else [])
    Mx = np.concatenate(cols, 1)
    Areal = np.concatenate([W[:, 0:1] * np.real(Mx), W[:, 1:2] * np.imag(Mx)], 0)
    rhs = np.concatenate([W[:, 0] * np.real(H), W[:, 1] * np.imag(H)])
    x = np.linalg.lstsq(Areal, rhs, rcond=None)[0]
    E = float(x[Phi.shape[1]]) if fit_linear else 0.0
    coef = x[:Phi.shape[1]]
    return dict(poles=poles, coef=coef, kinds=kinds, E=E, iterations=it, pole_change=float(change),
                Phi_fun=lambda ss: _pole_basis(poles, np.asarray(ss, complex))[0])


def realization_from_coef(poles, coef, n_out=1, n_in=1):
    """Real block-diagonal state space from poles and real-basis coefficients.

    `coef` has shape (n_coef,) for a scalar or (n_coef, n_out, n_in) for a
    block with common poles. One real state per real pole, a 2x2 block per
    pair; for a block each pole carries min(n_in, n_out) copies as needed:
    the coefficient matrix Theta_k (n_out x n_in) is factored Theta = U V^T by
    SVD and each rank-one term gets its own copy of the pole (Gilbert).
    For SISO this is exactly prune_slow_modes' construction:
        block [[a, w], [-w, a]], B = [1, 0], C = [2 Re r, 2 Im r]  (pair)
        block [a],              B = [1],    C = [r]               (real).
    Returns Ar, Br (n x n_in), Cr (n_out x n).
    """
    coef = np.asarray(coef, float)
    scalar = coef.ndim == 1
    if scalar:
        coef = coef[:, None, None]
    blocks, Bs, Cs = [], [], []
    k = 0
    for p in poles:
        if abs(np.imag(p)) < 1e-12:
            Th = coef[k]                       # (n_out, n_in)
            k += 1
            U, S, Vt = np.linalg.svd(Th)
            for q in range(len(S)):
                if S[q] <= 1e-14 * max(S[0], 1e-300):
                    continue
                blocks.append(np.array([[np.real(p)]]))
                Bs.append(Vt[q][None, :])                       # 1 x n_in
                Cs.append((S[q] * U[:, q])[:, None])            # n_out x 1
        else:
            Th = coef[k] + 1j * coef[k + 1]    # complex residue matrix R (n_out, n_in)
            k += 2
            U, S, Vt = np.linalg.svd(Th)
            a, w = np.real(p), np.imag(p)
            for q in range(len(S)):
                if S[q] <= 1e-14 * max(S[0], 1e-300):
                    continue
                u = S[q] * U[:, q]                              # complex n_out
                v = Vt[q]                                       # complex n_in (row of V^H)
                # r_ij = u_i v_j. With e^{At} = e^{at} [[cos, sin], [-sin, cos]],
                # C = [2 Re u, 2 Im u] and B = [Re v; -Im v] give
                # C e^{At} B = 2 e^{at} [(Re u Re v - Im u Im v) cos - (Re u Im v + Im u Re v) sin]
                #            = 2 Re[u v e^{p t}]   (reduces to prune_slow_modes for v = 1).
                blocks.append(np.array([[a, w], [-w, a]]))
                Bs.append(np.stack([np.real(v), -np.imag(v)], 0))       # 2 x n_in
                Cs.append(np.stack([2 * np.real(u), 2 * np.imag(u)], 1))   # n_out x 2
    n = sum(b.shape[0] for b in blocks)
    Ar = np.zeros((n, n))
    Br = np.zeros((n, coef.shape[2]))
    Cr = np.zeros((coef.shape[1], n))
    k = 0
    for b, bb, cc in zip(blocks, Bs, Cs):
        m = b.shape[0]
        Ar[k:k + m, k:k + m] = b
        Br[k:k + m, :] = bb
        Cr[:, k:k + m] = cc
        k += m
    if scalar:
        return Ar, Br[:, 0], Cr[0, :]
    return Ar, Br, Cr


def _passive_residues(Phi, H, W, Phi_dense, kinds, n_out=1, n_in=1, E_fixed=0.0, s=None,
                      directions=12, margin=0.0):
    """Residues (real basis coefficients) that minimise the weighted misfit to
    H subject to Re H_fit(iw) >= margin on the dense grid (scalar), or
    v^H Herm(H_fit(iw)) v >= margin for `directions` unit vectors v (block):
    a convex quadratic programme, solved with SLSQP (the constraints are
    linear in the coefficients for fixed poles)."""
    from scipy.optimize import minimize
    M, nc = Phi.shape
    nvar = nc * n_out * n_in
    # data matrix (2M*n_out*n_in rows): for each (i, j) entry the rows of Phi
    rows_A, rows_b = [], []
    for i in range(n_out):
        for j in range(n_in):
            Ablk = np.zeros((2 * M, nvar))
            idx =[(k * n_out + i) * n_in + j for k in range(nc)]
            Ablk[:M, idx] = W[:, 0:1] * np.real(Phi)
            Ablk[M:, idx] = W[:, 1:2] * np.imag(Phi)
            h = H[:, i, j] if H.ndim == 3 else H
            rows_A.append(Ablk)
            rows_b.append(np.concatenate([W[:, 0] * np.real(h), W[:, 1] * np.imag(h)]))
    Ad = np.concatenate(rows_A, 0)
    bd = np.concatenate(rows_b)
    # constraints
    Md = Phi_dense.shape[0]
    G_rows = []
    if n_out == 1 and n_in == 1:
        G_rows.append(np.real(Phi_dense))
    else:
        rng = np.random.default_rng(0)
        vs = []
        for a in np.linspace(0, np.pi, directions, endpoint=False):
            for b in np.linspace(0, 2 * np.pi, directions, endpoint=False):
                if n_in == 2:
                    vs.append(np.array([np.cos(a), np.sin(a) * np.exp(1j * b)]))
        if n_in != 2:
            vs = [v / np.linalg.norm(v) for v in (rng.normal(size=n_in) + 1j * rng.normal(size=n_in)
                                                 for _ in range(directions ** 2))]
        for v in vs:
            # v^H Herm(H) v = Re(v^H H v) = Re sum_k phi_k(iw) (v^H Theta_k v)
            G = np.zeros((Md, nvar))
            for k in range(nc):
                for i in range(n_out):
                    for j in range(n_in):
                        G[:, (k * n_out + i) * n_in + j] = np.real(Phi_dense[:, k] * np.conj(v[i]) * v[j])
            G_rows.append(G)
    G = np.concatenate(G_rows, 0)
    AtA = Ad.T @ Ad
    Atb = Ad.T @ bd
    x = np.linalg.lstsq(Ad, bd, rcond=None)[0]
    scale = max(np.abs(Atb).max(), 1e-30)
    fun = lambda x: float(0.5 * x @ AtA @ x - Atb @ x) / scale
    jac = lambda x: (AtA @ x - Atb) / scale
    # active set: only the constraints the current solution violates (plus
    # those already active) enter the QP; repeat until none is violated.
    active = np.zeros(G.shape[0], bool)
    tol = 1e-9 * max(np.abs(G @ x).max(), 1e-30)
    success, message = True, "no violation"
    for _ in range(12):
        viol = (G @ x - margin) < -tol
        if not viol.any():
            break
        active |= viol
        Ga = G[active]
        cons = [dict(type="ineq", fun=lambda y, Ga=Ga: Ga @ y - margin, jac=lambda y, Ga=Ga: Ga)]
        res = minimize(fun, x, jac=jac, constraints=cons, method="SLSQP",
                       options=dict(maxiter=500, ftol=1e-15))
        x, success, message = res.x, bool(res.success), str(res.message)
    viol = float(np.min(G @ x))
    return x.reshape(nc, n_out, n_in) if (n_out, n_in) != (1, 1) else x, dict(success=success, message=message,
                                                                                min_constraint=viol,
                                                                                n_active=int(active.sum()))


def fit_coefficient_fd(omegas, A, B, A_inf, order, t=None, dt=None, t_max=None, taper_start=0.65,
                       w_eval_min=0.2, w_eval_max=None, n_fine=20001, eval_bands=None,
                       weight=None, band=None, band_weight=4.0, min_decay=None, passive=True,
                       n_iter=30, w_focus=None, poles=None, passive_margin=0.0, extend_to_zero=False):
    """Frequency-domain fit of one coefficient pair (opt-in alternative to the
    time-domain pipeline in fit_coefficient; same output keys plus extras).

    Fits the memory transfer function
        H(i w) = B(w) + i w (A(w) - A_inf)        (= int_0^inf K(t) e^{-iwt} dt)
    on the tabulated frequencies directly, by vector fitting (see vector_fit)
    with `order` poles, so there is no cosine transform, no time window and
    no A_inf estimate in the loop: A_inf enters as the given number. If
    `A_inf` is None it is identified as the linear term E of B + i w A
    (returned as `A_inf`, with the solver value unknown).

    Weighting: the real part (B) and the imaginary part (w (A - A_inf)) are
    weighted 1 / max|B| and 1 / max|w (A - A_inf)| so both errors count alike;
    `band` = (w_lo, w_hi) multiplies the weight by `band_weight` there (the
    critical-frequency band at forward speed); `weight` (shape (M,) or (M, 2))
    replaces all of that.

    Constraints: poles stable with decay >= `min_decay` (default: a tenth of
    the grid step); `passive` re-solves the residues with the poles fixed so
    that Re H(iw) >= passive_margin on a dense grid from 0 to 10 w_max
    (positive-real memory). This is meaningful for a DIAGONAL entry only: an
    off-diagonal damping changes sign with frequency, and passivity of a
    coupled block is a property of the Hermitian part of the whole matrix
    (see fit_block_fd); pass passive=False for off-diagonal entries. The
    realisation is block-diagonal modal, like prune_slow_modes.

    err_K here compares the kernel of the fit with the kernel of the table
    THROUGH THE SAME TAPER (both as (2/pi) int W B cos wt dw on the window t):
    the table's B is not zero at its last frequency, so the raw kernel at
    small t is not knowable from the table and comparing an untapered impulse
    response to a tapered table kernel would measure the taper. `K_ss` is the
    plain impulse response of the realisation, `K_ss_tapered` the windowed one
    used for err_K.
    """
    omegas = np.asarray(omegas, float)
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    d_omega = float(np.mean(np.diff(omegas)))
    if t is None:
        t_max = t_max if t_max is not None else min(np.pi / d_omega, 30.0)
        dt = dt if dt is not None else min(np.pi / omegas.max() / 8, 0.02)
        t = np.arange(0.0, t_max, dt)
    if min_decay is None:
        min_decay = 0.1 * d_omega
    order = int(order)
    s = 1j * omegas
    fit_linear = A_inf is None
    if fit_linear:
        H = B + 1j * omegas * A
        im_scale = max(np.max(np.abs(omegas * (A - A[-1]))), 1e-30)
    else:
        A_inf = float(A_inf)
        H = B + 1j * omegas * (A - A_inf)
        im_scale = max(np.max(np.abs(np.imag(H))), 1e-30)
    re_scale = max(np.max(np.abs(B)), 1e-30)
    if weight is None:
        W = np.stack([np.full(len(omegas), 1.0 / re_scale), np.full(len(omegas), 1.0 / im_scale)], 1)
        if band is not None:
            m = (omegas >= band[0] - 1e-9) & (omegas <= band[1] + 1e-9)
            W[m] *= band_weight
    else:
        weight = np.asarray(weight, float)
        W = np.stack([weight, weight], 1) if weight.ndim == 1 else weight.copy()
    if poles is None:
        poles = initial_poles(order, omegas[omegas > 0].min(), omegas.max(), w_focus=w_focus)
    vf = vector_fit(s, H, order, weight=W, n_iter=n_iter, poles=poles, min_decay=min_decay,
                    fit_linear=fit_linear)
    poles, coef, E = vf["poles"], vf["coef"], vf["E"]
    if fit_linear:
        A_inf = E
    # passivity
    w_dense = np.concatenate([np.linspace(0.0, 2.0 * omegas.max(), 3000), np.geomspace(2.0 * omegas.max(), 10 * omegas.max(), 500)])
    Phi_dense, kinds = _pole_basis(poles, 1j * w_dense)
    Phi, _ = _pole_basis(poles, s)
    H_target = H - (1j * omegas * E if fit_linear else 0.0)
    passive_info = dict(enforced=False, min_before=float(np.min(np.real(Phi_dense @ coef)) / re_scale))
    if passive and passive_info["min_before"] < passive_margin / re_scale - 1e-12:
        coef, info = _passive_residues(Phi, H_target, W, Phi_dense, kinds, margin=passive_margin)
        passive_info.update(enforced=True, **info)
    passive_info["min_after"] = float(np.min(np.real(Phi_dense @ coef)) / re_scale)
    Ar, Br, Cr = realization_from_coef(poles, coef)

    Hf = Phi @ coef
    B_fit = np.real(Hf)
    A_fit = A_inf + np.imag(Hf) / omegas
    # kernels through the same taper
    tap = taper(omegas, taper_start)
    K = retardation_function(omegas, B, t, n_fine=n_fine, extend_to_zero=extend_to_zero)
    K_fit_tap = retardation_function(omegas, B_fit, t, n_fine=n_fine, extend_to_zero=extend_to_zero)
    K_ss = impulse_response(Ar, Br, Cr, t)
    scale = max(np.max(np.abs(K)), 1e-30)
    bs = re_scale
    as_ = max(np.max(np.abs(A - A_inf)), 1e-30)
    w_hi = w_eval_max if w_eval_max is not None else taper_start * omegas.max()
    ev = (omegas >= w_eval_min) & (omegas <= w_hi)
    bandm = (omegas >= WAVE_BAND[0]) & (omegas <= WAVE_BAND[1])
    return dict(t=t, K=K, K_ss=K_ss, K_ss_tapered=K_fit_tap, A_inf=float(A_inf), A_inf_spread=float("nan"),
                Ar=Ar, Br=Br, Cr=Cr, singular_values=None, order=order,
                B_fit=B_fit, A_fit=A_fit, w_eval_max=w_hi,
                err_K=float(np.max(np.abs(K_fit_tap - K)) / scale),
                err_K_raw=float(np.max(np.abs(K_ss - K)) / scale),
                err_B=float(np.max(np.abs(B_fit - B)[ev]) / bs),
                err_A=float(np.max(np.abs(A_fit - A)[ev]) / as_),
                err_B_band=float(np.max(np.abs(B_fit - B)[bandm]) / bs),
                err_A_band=float(np.max(np.abs(A_fit - A)[bandm]) / as_),
                err_B_all=float(np.max(np.abs(B_fit - B)) / bs),
                err_A_all=float(np.max(np.abs(A_fit - A)) / as_),
                tail_at_wmax=float(np.abs(B[-1]) / bs),
                max_real_eig=float(np.max(np.real(eig(Ar)[0]))),
                memory_time=float(t[np.max(np.where(np.abs(K) > 0.02 * scale)[0])]) if scale > 0 else 0.0,
                method="vf", n_states=int(Ar.shape[0]), poles=np.array(poles), coef=coef,
                vf_iterations=vf["iterations"], vf_pole_change=vf["pole_change"],
                passivity=passivity_margin(Ar, Br, Cr), passive_info=passive_info,
                A_inf_est=float(E) if fit_linear else None, taper_weights=tap,
                err_bands=band_errors(omegas, A, B, A_fit, B_fit, A_inf, eval_bands))


def fit_block_fd(omegas, A, B, A_inf, order, weight=None, band=None, band_weight=4.0,
                 min_decay=None, passive=True, n_iter=30, w_focus=None, eval_bands=None,
                 w_eval_min=0.2, w_eval_max=None, taper_start=0.65, passive_margin=0.0,
                 directions=12):
    """Common-pole frequency-domain fit of an m x m block of coefficients.

    A, B: (nw, m, m); A_inf: (m, m). Fits H(iw) = B + i w (A - A_inf) with ONE
    set of `order` poles and an m x m residue matrix per pole (vector fitting
    proper: the poles are properties of the fluid, the residues of the mode
    shapes). The realisation x' = Ar x + Br nu (nu: m velocities),
    mu = Cr x (m memory forces) has at most m states per real pole and 2m per
    pair (Gilbert, rank of each residue matrix), i.e. <= m * order states for
    the whole block against m^2 * order for m^2 separate scalar fits.

    Passivity for a block means the Hermitian part of H(iw) is positive
    semidefinite at every w; it is enforced on a dense grid through sampled
    directions v (v^H Herm v >= passive_margin) and reported as the smallest
    eigenvalue of the Hermitian part relative to the largest (`passivity`).
    """
    omegas = np.asarray(omegas, float)
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    A_inf = np.asarray(A_inf, float)
    m = A.shape[1]
    d_omega = float(np.mean(np.diff(omegas)))
    if min_decay is None:
        min_decay = 0.1 * d_omega
    s = 1j * omegas
    H = B + 1j * omegas[:, None, None] * (A - A_inf[None])
    re_scale = max(np.max(np.abs(B)), 1e-30)
    im_scale = max(np.max(np.abs(np.imag(H))), 1e-30)
    if weight is None:
        W = np.stack([np.full(len(omegas), 1.0 / re_scale), np.full(len(omegas), 1.0 / im_scale)], 1)
        if band is not None:
            mm = (omegas >= band[0] - 1e-9) & (omegas <= band[1] + 1e-9)
            W[mm] *= band_weight
    else:
        weight = np.asarray(weight, float)
        W = np.stack([weight, weight], 1) if weight.ndim == 1 else weight.copy()
    poles = initial_poles(int(order), omegas[omegas > 0].min(), omegas.max(), w_focus=w_focus)
    # vector fitting with common poles: stack the m^2 entries as a vector
    # (the sigma-scaling is shared, the residues are per entry)
    Hv = H.reshape(len(omegas), m * m)
    change = np.inf
    it = 0
    for it in range(1, n_iter + 1):
        Phi, kinds = _pole_basis(poles, s)
        nc = Phi.shape[1]
        # unknowns: c (nc x m^2) and shared d (nc)
        rows, rhs = [], []
        for e in range(m * m):
            blk = np.zeros((2 * len(omegas), nc * m * m + nc))
            blk[:len(omegas), e * nc:(e + 1) * nc] = W[:, 0:1] * np.real(Phi)
            blk[len(omegas):, e * nc:(e + 1) * nc] = W[:, 1:2] * np.imag(Phi)
            Md = -Hv[:, e:e + 1] * Phi
            blk[:len(omegas), -nc:] = W[:, 0:1] * np.real(Md)
            blk[len(omegas):, -nc:] = W[:, 1:2] * np.imag(Md)
            rows.append(blk)
            rhs.append(np.concatenate([W[:, 0] * np.real(Hv[:, e]), W[:, 1] * np.imag(Hv[:, e])]))
        x = np.linalg.lstsq(np.concatenate(rows, 0), np.concatenate(rhs), rcond=None)[0]
        d = x[-nc:]
        new = _group_poles(_real_modal_sigma(poles, d), min_decay)
        old_arr = np.sort_complex(np.array(poles))
        new_arr = np.sort_complex(np.array(new))
        change = (np.max(np.abs(old_arr - new_arr)) / max(np.max(np.abs(old_arr)), 1e-30)
                  if len(old_arr) == len(new_arr) else np.inf)
        poles = new
        if change < 1e-6:
            break
    Phi, kinds = _pole_basis(poles, s)
    nc = Phi.shape[1]
    Areal = np.concatenate([W[:, 0:1] * np.real(Phi), W[:, 1:2] * np.imag(Phi)], 0)
    coef = np.zeros((nc, m, m))
    for i in range(m):
        for j in range(m):
            rhs = np.concatenate([W[:, 0] * np.real(H[:, i, j]), W[:, 1] * np.imag(H[:, i, j])])
            coef[:, i, j] = np.linalg.lstsq(Areal, rhs, rcond=None)[0]
    w_dense = np.concatenate([np.linspace(0.0, 2.0 * omegas.max(), 1500), np.geomspace(2.0 * omegas.max(), 10 * omegas.max(), 300)])
    Phi_dense, _ = _pole_basis(poles, 1j * w_dense)

    def herm_min(cf):
        Hd = np.einsum("wk,kij->wij", Phi_dense, cf)
        Hh = 0.5 * (Hd + np.conj(np.swapaxes(Hd, 1, 2)))
        ev = np.linalg.eigvalsh(Hh)
        return float(ev[:, 0].min() / max(np.abs(ev).max(), 1e-30))

    passive_info = dict(enforced=False, min_before=herm_min(coef))
    if passive and passive_info["min_before"] < -1e-12:
        coef, info = _passive_residues(Phi, H, W, Phi_dense, kinds, n_out=m, n_in=m,
                                       directions=directions, margin=passive_margin)
        passive_info.update(enforced=True, **info)
    passive_info["min_after"] = herm_min(coef)
    Ar, Br, Cr = realization_from_coef(poles, coef, n_out=m, n_in=m)
    Hf = np.einsum("wk,kij->wij", Phi, coef)
    B_fit = np.real(Hf)
    A_fit = A_inf[None] + np.imag(Hf) / omegas[:, None, None]
    w_hi = w_eval_max if w_eval_max is not None else taper_start * omegas.max()
    ev = (omegas >= w_eval_min) & (omegas <= w_hi)
    bs = re_scale
    as_ = max(np.max(np.abs(A - A_inf[None])), 1e-30)
    entries = {}
    for i in range(m):
        for j in range(m):
            bij = max(np.max(np.abs(B[:, i, j])), 1e-30)
            aij = max(np.max(np.abs(A[:, i, j] - A_inf[i, j])), 1e-30)
            entries[(i, j)] = dict(err_B=float(np.max(np.abs(B_fit[:, i, j] - B[:, i, j])[ev]) / bij),
                                   err_A=float(np.max(np.abs(A_fit[:, i, j] - A[:, i, j])[ev]) / aij),
                                   err_bands=band_errors(omegas, A[:, i, j], B[:, i, j], A_fit[:, i, j], B_fit[:, i, j],
                                                         A_inf[i, j], eval_bands))
    return dict(Ar=Ar, Br=Br, Cr=Cr, poles=np.array(poles), coef=coef, order=int(order),
                n_states=int(Ar.shape[0]), A_fit=A_fit, B_fit=B_fit, A_inf=A_inf,
                err_B=float(np.max(np.abs(B_fit - B)[ev]) / bs), err_A=float(np.max(np.abs(A_fit - A)[ev]) / as_),
                entries=entries, vf_iterations=it, vf_pole_change=float(change),
                max_real_eig=float(np.max(np.real(eig(Ar)[0]))),
                passivity=passivity_margin(Ar, Br, Cr), passive_info=passive_info, method="vf-block")
