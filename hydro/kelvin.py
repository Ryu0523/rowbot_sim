#!/usr/bin/env python3
"""
The steady Kelvin wave system -- the bow wave and the V-shaped wake.

This is the wave system a person actually notices behind a boat, and it was
missing from the whole model: the BEM is solved at ZERO FORWARD SPEED, so it
produces only the unsteady scattering and radiation. Those came out at 0.042 m
peak, which is correct for what they represent and an order of magnitude below
what the eye expects, because the eye is looking at this term instead.

Capytaine's forward_speed does not help -- it applies a Doppler shift to the
encounter frequency and still solves the zero-speed steady problem underneath.
So the steady system is built here from thin-ship (Michell) theory, which is
the classical tool for exactly this hull shape.

WHAT IS DERIVED, AND WHAT IS CALIBRATED -- the distinction matters:

  DERIVED (exact, and checked numerically below)
    Any free wave must stand still relative to the hull, so its phase speed
    along the track matches the ship speed:

        U cos(theta) = sqrt(g/k)   =>   k(theta) = k0 sec^2(theta),  k0 = g/U^2

    That single condition fixes the entire GEOMETRY of the pattern: the
    19.47 deg Kelvin half-angle, the transverse wavelength 2 pi U^2 / g, the
    divergent crests, and the fact that in deep water every free wave lies
    BEHIND the hull. The tests check both against theory.

  DERIVED (shape of the spectrum)
    Michell puts a source sheet on the centreplane with strength proportional
    to d(half-beam)/dx, so the relative weight of each heading theta is

        I(theta) = int int  dh/dx . exp(k z) . exp(-i k x cos theta)  dx dz

    which for the Wigley hull SEPARATES into two closed forms (below).

  CALIBRATED (one scalar)
    The absolute amplitude is set by ENERGY, not by a remembered prefactor:
    the wake carries R_w * U watts, and R_w is taken as a stated fraction of
    the calm-water resistance the vessel model already has. That fraction
    (RW_FRACTION) is the one assumption in this module, it is a plausible
    value for a semi-displacement hull near its resistance hump, and it is the
    only knob to turn if towing-tank data arrives.

So: pattern and wavelengths are physics; overall height is anchored to the
model's own resistance through an energy balance and is accurate to a factor
of order one, not better. It is used for VISUALISATION and is deliberately not
fed back into the forces -- wave-making resistance is already inside the fitted
quadratic drag, and adding it twice would double count.
"""
import numpy as np

G = 9.81
RHO = 1025.0

# Fraction of calm-water resistance attributed to wave making. For a
# semi-displacement hull around Fn = 0.45-0.5 -- near the resistance hump --
# this is typically about half. The single calibrated number in this module.
RW_FRACTION = 0.45


def _z_integral(k, T):
    """int_{-T}^{0} (1 - (z/T)^2) e^{kz} dz, in closed form.

    Done analytically because quadrature fails where it matters: at large theta
    the wavenumber reaches tens per metre and e^{kz} varies by more than a
    factor of two across a grid cell, so the numerically integrated spectrum
    stopped decaying and turned into noise.
    """
    k = np.asarray(k, float)
    e = np.exp(-k * T)
    t1 = (1.0 - e) / k
    t2 = (2.0 / k ** 3 - e * (T ** 2 / k + 2 * T / k ** 2 + 2 / k ** 3)) / T ** 2
    return t1 - t2


def _x_integral(mu, L):
    """int_{-L/2}^{L/2} x e^{-i mu x} dx = -2i [ sin(mu a)/mu^2 - a cos(mu a)/mu ]

    (x cos(mu x) is odd and drops out; x sin(mu x) is even.)
    """
    a = L / 2.0
    mu = np.asarray(mu, float)
    small = np.abs(mu) < 1e-8
    mu_s = np.where(small, 1.0, mu)
    val = -2j * (np.sin(mu_s * a) / mu_s ** 2 - a * np.cos(mu_s * a) / mu_s)
    return np.where(small, 0.0 + 0j, val)


def resolvable_theta_max(k0, dx, cells_per_wave=4.0, hard_cap=1.45):
    """Largest heading whose wave the grid can actually carry.

    k(theta) = k0 sec^2(theta) runs away near 90 deg: at U = 3.5 m/s and
    theta = 1.45 rad the wavelength is 0.12 m, on a grid spaced 0.23 m. Those
    headings do not integrate, they ALIAS -- they fold back onto the grid as
    spurious long waves. That is what made the field look unconverged, and
    worst at the LOWEST speed (largest k0), which is the opposite of how a
    genuine truncation error behaves; the direction of that trend is what gave
    it away.

    So the heading range is set by Nyquist rather than by a fixed constant:
    keep k <= 2 pi / (cells_per_wave . dx). Everything dropped is a short
    divergent wave the grid could not have drawn in any case, and on a finer
    grid the limit opens up on its own.
    """
    k_max = 2 * np.pi / (cells_per_wave * dx)
    c2 = min(1.0, max(k0 / k_max, 1e-12))
    return float(min(hard_cap, np.arccos(np.sqrt(c2))))


def theta_samples_needed(k0, theta_max, radius, per_cycle=4.0, cap=24001):
    """How many headings the theta-integral needs before it stops aliasing.

    The integrand carries phase k(x cos th + y sin th) = k0 (x sec th +
    y sec^2 th sin th), whose derivative in th blows up with the heading:

        d(phase)/d(th) ~ k0 . radius . sec(th) tan(th)

    At U = 3.5 m/s, radius 60 m and th = 80 deg that is about 1600 rad per rad,
    so a fixed 401 samples advance the phase by more than 2 rad per step and the
    QUADRATURE aliases -- distinct from the grid aliasing handled above, and the
    reason the field appeared to get LESS converged as the grid was refined:
    refining opens theta_max, which is exactly where the sampling was failing.
    Here the sample count follows the phase rate instead of being a constant.
    """
    t = min(theta_max, 1.5533)                      # keep sec/tan finite
    rate = k0 * max(radius, 1.0) / np.cos(t) * np.tan(t)
    n = int(2 * theta_max * rate * per_cycle / (2 * np.pi)) | 1
    return int(min(max(n, 401), cap))


def free_wave_spectrum(U, L, B, T, n_theta=401, theta_max=1.45):
    """Free-wave amplitude per heading for a Wigley hull.

    The Michell source sheet has strength proportional to dh/dx, and for this
    hull dh/dx = (B/2)(-8x/L^2)(1-(z/T)^2) SEPARATES, so the double integral is
    a product of two closed forms:

        I(theta) = (B/2)(-8/L^2) X(mu) Z(k),  mu = k cos(theta) = k0 sec(theta)

    Returns (theta, k, I, A) where A is the amplitude weighting used for the
    elevation. That weighting is taken from Michell's WAVE RESISTANCE
    integrand, R_w proportional to integral |I|^2 sec^3(theta) d(theta), which
    makes the energy per unit heading |I|^2 sec^3, hence an amplitude
    |I| sec^(3/2). This is the honest way to get the split between transverse
    and divergent waves without reproducing the far-field prefactor from
    scratch -- and it converges, decaying like cos^(3/2), whereas a naive
    sec^3 weighting does not decay at all.
    """
    k0 = G / U ** 2
    th = np.linspace(-theta_max, theta_max, n_theta)
    sec = 1.0 / np.cos(th)
    k = k0 * sec ** 2
    I = (B / 2) * (-8.0 / L ** 2) * _x_integral(k0 * sec, L) * _z_integral(k, T)
    A = I * sec ** 1.5
    # Raised cosine over the last tenth of the range. A hard cut in theta rings
    # in x -- Gibbs -- laying fringes across the field that belong to the cut
    # rather than to the ship.
    r = np.clip((np.abs(th) - 0.9 * theta_max) / (0.1 * theta_max), 0.0, 1.0)
    A = A * 0.5 * (1.0 + np.cos(np.pi * r))
    return th, k, I, A


def wake_field(U, L, B, T, x, y, n_theta=None, theta_max=None,
               resistance=None, rw_fraction=RW_FRACTION):
    """Steady elevation on a ship-fixed grid. x is positive FORWARD.

    Returns (eta, info). Deep water puts every free wave behind the hull, so
    the field is masked ahead of the bow -- which is also why a forward-looking
    preview sensor never sees this system.

    Both sampling limits default to what THIS grid demands: theta_max from
    `resolvable_theta_max` (what the grid can draw) and n_theta from
    `theta_samples_needed` (what the phase demands). Pass either only to test
    convergence against it.
    """
    k0 = G / U ** 2
    dx = float(x[1] - x[0])
    if theta_max is None:
        theta_max = resolvable_theta_max(k0, dx)
    X, Y = np.meshgrid(x, y, indexing="ij")
    if n_theta is None:
        radius = float(max(np.abs(x).max(), np.abs(y).max()))
        n_theta = theta_samples_needed(k0, theta_max, radius)
    th, k, I, A = free_wave_spectrum(U, L, B, T, n_theta, theta_max)
    dth = th[1] - th[0]

    # Summed in chunks over heading. Resolving the phase can need tens of
    # thousands of headings, and the full nx.ny.n_theta complex array would run
    # to gigabytes; accumulating keeps it to one grid-sized buffer.
    eta = np.zeros(X.shape)
    for s0 in range(0, n_theta, 256):
        sl = slice(s0, s0 + 256)
        ph = (k[sl][None, None, :] * (X[..., None] * np.cos(th[sl])
                                      + Y[..., None] * np.sin(th[sl])))
        eta += np.real(np.sum(A[sl][None, None, :] * np.exp(1j * ph), axis=-1))
    eta *= dth

    # radiation condition: no free waves upstream in deep water
    eta = eta * (1.0 / (1.0 + np.exp((X + 0.35 * L) / (0.06 * L))))

    # --- energy calibration ------------------------------------------------
    # The wake carries R_w * U watts. For a wave field of mean-square elevation
    # <eta^2> over an effective width W, the flux is (rho g <eta^2>) c_g W with
    # c_g = U/2 for the transverse system, so matching the flux to R_w U fixes
    # the one free constant:  <eta^2> = 2 R_w / (rho g W).
    if resistance is None:
        resistance = 280.0 * U ** 2                 # the plant's own drag law
    Rw = rw_fraction * resistance
    W = max(y.max() - y.min(), 1.0)
    band = X < -0.5 * L
    ms = float(np.mean(eta[band] ** 2)) if band.any() else 1.0
    target_ms = 2.0 * Rw / (RHO * G * W)
    scale = np.sqrt(target_ms / max(ms, 1e-30))
    return eta * scale, dict(k0=k0, Rw=Rw, scale=float(scale),
                             theta_max=float(theta_max),
                             lam_transverse=2 * np.pi * U ** 2 / G)


def wake_texture(U, L, B, T, nx=384, ny=256, x_lim=None, y_lim=None,
                 edge=0.10):
    """Bake the steady wake into a ship-fixed image the viewer can sample.

    The browser cannot re-run this sum: resolving the phase takes thousands of
    headings (see `theta_samples_needed`), which is a minute of numpy, not a
    frame budget. But the wake is STEADY in the ship frame, so it only has to
    be computed once and then read back as a lookup -- one texture fetch per
    vertex, at full frame rate.

    Speed is handled by a similarity argument rather than by baking a stack.
    The phase is

        k0 sec^2(th) (x cos th + y sin th),   k0 = g/U^2

    so writing X = x g/U^2 and Y = y g/U^2 removes U from it entirely: the
    pattern is EXACTLY self-similar in x/U^2, and the shader recovers any speed
    by stretching the lookup by (u/U0)^2, with the amplitude following the same
    factor because R_w goes as U^2. The one thing that is NOT self-similar is
    the free-wave spectrum A(theta), which still sees the hull's fixed length
    through k0 L; over the +-20% the vessel actually holds around U0 that shifts
    the balance between transverse and divergent waves slightly, and nothing
    else. Stated rather than hidden.

    Quantised to 8 bits: the step is amplitude/255, about 5 mm on a 0.6 m wake,
    a fifth of a percent and well under the calibration uncertainty. Sent as
    one byte per texel so the payload stays near 100 KB rather than the 400 KB
    of float text.
    """
    import base64

    x_lim = x_lim if x_lim is not None else (-6.0 * L, 1.2 * L)
    y_lim = y_lim if y_lim is not None else (-3.0 * L, 3.0 * L)
    x = np.linspace(x_lim[0], x_lim[1], nx)
    y = np.linspace(y_lim[0], y_lim[1], ny)
    eta, info = wake_field(U, L, B, T, x, y)

    # The window is a display crop, not the end of the wake, so fade it out at
    # the edges. A hard cut would draw a rectangle on the sea.
    def _ramp(n):
        w = np.ones(n)
        m = max(int(edge * n), 1)
        r = 0.5 * (1.0 - np.cos(np.pi * np.arange(m) / m))
        w[:m] = r
        w[-m:] = r[::-1]
        return w
    eta = eta * _ramp(nx)[:, None] * _ramp(ny)[None, :]

    amp = float(np.abs(eta).max())
    q = np.clip(np.round((eta / max(amp, 1e-12) * 0.5 + 0.5) * 255.0),
                0, 255).astype(np.uint8)
    # rows are y, columns are x, which is what a texture upload expects
    return dict(w=int(nx), h=int(ny), amp=amp, U0=float(U),
                x0=float(x_lim[0]), x1=float(x_lim[1]),
                y0=float(y_lim[0]), y1=float(y_lim[1]),
                lam_transverse=float(info["lam_transverse"]),
                half_angle=19.47,
                b64=base64.b64encode(q.T.copy(order="C").tobytes()).decode())


# ------------------------------------------------------------------- checks
def kelvin_half_angle(eta, x, y, L):
    """Locate the wake edge from the field and compare with 19.47 deg."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    env = np.abs(eta)
    ang = []
    for i in np.where(x < -2.0 * L)[0]:
        col = env[i]
        # the wedge edge is a CAUSTIC -- the brightest ridge -- so a high
        # threshold finds it, where a low one just measures numerical skirt
        j = np.where(col > 0.45 * col.max())[0]
        if len(j):
            ang.append(np.degrees(np.arctan(np.abs(y[j]).max() / abs(x[i]))))
    return float(np.median(ang)) if ang else float("nan")


def transverse_wavelength(U, L, B, T, n_wave=6.0):
    """Dominant crest spacing on the centreline, measured on its own line.

    Two traps, both hit on the way here. Counting ZERO CROSSINGS biases short,
    because the centreline carries divergent crests too and they add crossings.
    And a plain FFT over the display grid biases to the nearest BIN: 45 m of
    record is under two cycles at 27 m, so the answer can only land on 45.1,
    22.6, 15.0, 11.3, ... -- which is exactly the ladder the second attempt
    returned, and it read like a physics error when it was a ruler error.

    So: evaluate a dedicated centreline long enough for n_wave cycles (cheap,
    y holds one point), then take the spectral peak refined by a parabola
    through its two neighbours, which locates it between bins.
    """
    lam = 2 * np.pi * U ** 2 / G
    span = n_wave * lam
    x = np.linspace(-1.5 * L - span, -1.5 * L, 1024)
    eta, _ = wake_field(U, L, B, T, x, np.array([0.0]))
    line = eta[:, 0] - eta[:, 0].mean()
    dx = x[1] - x[0]
    F = np.abs(np.fft.rfft(line * np.hanning(len(line))))
    kk = np.fft.rfftfreq(len(line), dx)
    i = int(np.argmax(F[1:])) + 1
    if 0 < i < len(F) - 1:
        a_, b_, c_ = F[i - 1], F[i], F[i + 1]
        den = a_ - 2 * b_ + c_
        # At a peak this second difference is NEGATIVE. Clamping it to a small
        # POSITIVE floor -- the obvious-looking guard -- inverted the sign and
        # sent the refinement to infinity, which is why the measured wavelength
        # came back as -0.0 while the raw bin was already right to 0.1%.
        # Guard on magnitude, and cap the shift at half a bin, which is all a
        # parabola through three points can legitimately claim.
        if abs(den) > 1e-30:
            d = float(np.clip(0.5 * (a_ - c_) / den, -0.5, 0.5))
            return float(1.0 / (kk[i] + d * (kk[1] - kk[0])))
    return float(1.0 / kk[i])


def truncation_error(U, L, B, T, x, y):
    """Convergence in the heading range: recompute over a narrower range and
    compare the FIELDS. Sampling the integrand at the edge says little, because
    the spectrum oscillates. Both calls share one grid, so this isolates the
    heading cut from the grid resolution.

    Compared in RMS, not peak. The wedge edge is a CAUSTIC -- a ridge one or
    two cells wide -- so a max-pixel difference mostly reports where that ridge
    landed between cells: it read 7.1 / 4.6 / 5.3 / 15.4 % as the grid cap was
    tightened, wandering with no trend, while the measured half-angle held to a
    tenth of a degree throughout. The field was converged; the ruler was not.
    """
    tm = resolvable_theta_max(G / U ** 2, float(x[1] - x[0]))
    a, _ = wake_field(U, L, B, T, x, y, theta_max=tm)
    b, _ = wake_field(U, L, B, T, x, y, theta_max=0.95 * tm)
    return float(np.sqrt(np.mean((a - b) ** 2))
                 / max(np.sqrt(np.mean(a ** 2)), 1e-30))


def grid_refines(U, L, B, T, coarse=320, fine=640):
    """Does refining the grid reduce the heading-truncation error?

    This is the check that matters, because the two sampling limits are
    COUPLED: a coarse grid forces a low theta_max, and at that cut the spectrum
    is still around a fifth of its peak, so trimming it moves the field. The
    residual is therefore not a bug to be tuned away -- it is the grid honestly
    reporting that it cannot draw the short divergent waves. Refining lifts the
    Nyquist cap to a heading where the spectrum has fallen to under a tenth,
    and the error falls with it.
    """
    out = []
    for n in (coarse, fine):
        x = np.linspace(-6 * L, 1.2 * L, n)
        y = np.linspace(-3 * L, 3 * L, int(0.75 * n))
        out.append(truncation_error(U, L, B, T, x, y))
    return out[0], out[1]


def main():
    L, B, T = 10.0, 2.5, 0.8
    # 480 across the domain: enough that the grid Nyquist admits headings where
    # the free-wave spectrum has already decayed. See `grid_refines`.
    x = np.linspace(-6 * L, 1.2 * L, 480)
    y = np.linspace(-3 * L, 3 * L, 360)
    print(f"Wigley {L}x{B}x{T} m, thin-ship (Michell) steady wave system\n")
    print(f"  {'U':>6}{'Fn':>7}{'th_max':>8}{'lam_t thy':>11}{'meas':>8}"
          f"{'half-ang':>10}{'peak':>8}{'rms':>8}{'conv':>8}")
    ok = []
    for U in (3.5, 4.6, 5.5, 6.5):
        eta, info = wake_field(U, L, B, T, x, y)
        lam = transverse_wavelength(U, L, B, T)
        ang = kelvin_half_angle(eta, x, y, L)
        pk = float(np.abs(eta).max())
        rms = float(np.sqrt(np.mean(eta[x < -1.5 * L] ** 2)))
        conv = truncation_error(U, L, B, T, x, y)
        ok.append((abs(lam - info["lam_transverse"]) / info["lam_transverse"],
                   abs(ang - 19.47), pk, rms, conv))
        print(f"  {U:>6.1f}{U/np.sqrt(G*L):>7.2f}"
              f"{np.degrees(info['theta_max']):>7.0f}d"
              f"{info['lam_transverse']:>11.1f}"
              f"{lam:>8.1f}{ang:>10.1f}{pk:>8.3f}{rms:>8.3f}{conv:>8.1%}")

    c_coarse, c_fine = grid_refines(3.5, L, B, T)
    print()
    print(f"  grid refinement at the worst case (U = 3.5): "
          f"{c_coarse:.1%} -> {c_fine:.1%}")

    peaks = [o[2] for o in ok]
    rmss = [o[3] for o in ok]
    checks = [
        ("transverse wavelength matches 2 pi U^2/g within 8%",
         max(o[0] for o in ok) < 0.08),
        ("wake half-angle within 3 deg of 19.47",
         max(o[1] for o in ok) < 3.0),
        ("field converged in heading range, in RMS (< 5%)",
         max(o[4] for o in ok) < 0.05),
        ("that residual is grid-limited: refining the grid reduces it",
         c_fine < c_coarse),
        ("peak elevation in the physical 0.1-1.5 m range",
         all(0.1 < p < 1.5 for p in peaks)),
        ("wake RMS grows with speed, as the energy calibration requires",
         rmss == sorted(rmss)),
    ]
    print("\n  " + "-" * 52)
    for n, good in checks:
        print(f"  [{'PASS' if good else 'FAIL'}]  {n}")
    print("  " + "-" * 52)
    print(f"  KELVIN WAKE GATE: "
          f"{'PASSED' if all(g for _, g in checks) else 'FAILED'}")
    print(f"\n  Geometry is derived; amplitude is calibrated to "
          f"{RW_FRACTION:.0%} of calm-water resistance.")
    print("  That calibration forces R_w proportional to U^2, so the real")
    print("  humps and hollows of wave resistance against Froude number are")
    print("  flattened: the pattern is right, the speed trend is smoothed.")
    return all(g for _, g in checks)


if __name__ == "__main__":
    main()
