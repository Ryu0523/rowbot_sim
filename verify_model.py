#!/usr/bin/env python3
"""
Validation of what the simulator actually reproduces.

Panel 1  eta(x, y) snapshot          -- it IS a 2-D short-crested field
Panel 2  measured PSD vs target      -- the spectrum is right, not just Hs
Panel 3  elevation PDF vs Gaussian   -- and where linear theory gives up
Panel 4  USV states over time        -- position, speed, heave, pitch
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step0_preview_spec import WaveField, jonswap, SEA_STATES
import vessel as V

p = SEA_STATES[5]
HS, TP = p["hs"], p["tp"]


def welch_psd(x, dt, n_seg=24):
    """One-sided PSD in m^2/(rad/s). Plain numpy, no scipy needed."""
    n = len(x) // n_seg
    w = np.hanning(n)
    acc = np.zeros(n // 2 + 1)
    for i in range(n_seg):
        seg = x[i * n:(i + 1) * n]
        X = np.fft.rfft((seg - seg.mean()) * w)
        acc += np.abs(X) ** 2
    acc /= n_seg
    Pf = 2.0 * acc * dt / (n * (w ** 2).mean())      # m^2/Hz
    Pf[0] = Pf[-1] = 0.0
    f = np.fft.rfftfreq(n, dt)
    return 2 * np.pi * f, Pf / (2 * np.pi)           # -> omega, S(omega)


def main():
    wf = WaveField(HS, TP, n_freq=64, n_dir=24, seed=7)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # --- 1. 2-D snapshot ------------------------------------------------
    gx = np.arange(0, 601, 3.0)
    gy = np.arange(-300, 301, 3.0)
    XX, YY = np.meshgrid(gx, gy, indexing="ij")
    Z = wf.eta(XX.ravel(), YY.ravel(), 0.0).reshape(XX.shape)
    im = ax[0, 0].pcolormesh(gx, gy, Z.T, cmap="RdBu_r", vmin=-3.5, vmax=3.5,
                             shading="auto", rasterized=True)
    plt.colorbar(im, ax=ax[0, 0], label="$\\eta$ (m)")
    ax[0, 0].set_aspect("equal")
    ax[0, 0].set_xlabel("x (m)  [waves travel toward -x]")
    ax[0, 0].set_ylabel("y (m)")
    ax[0, 0].set_title(f"1. Short-crested field, t=0.  Field $H_s$="
                       f"{wf.realised_hs():.2f} m (target {HS}); this 600 m box "
                       f"alone reads {4*Z.std():.2f} m", fontsize=10)

    # --- 2. spectrum recovery -------------------------------------------
    # A single-point time series is a BAD estimator here: with a discrete
    # frequency set the realised variance at a point is frozen, so more time
    # does not converge it (measured scatter across locations: +-14% on Hs).
    # Average the periodogram over locations instead.
    dt = 0.25
    t = np.arange(0, 3600, dt)
    rng = np.random.default_rng(3)
    series, S_acc = [], None
    for xx, yy in rng.uniform(0, 4000, (12, 2)):
        s_ = np.array([wf.eta(xx, yy, tt)[0] for tt in t])
        series.append(s_)
        w_meas, S_ = welch_psd(s_, dt)
        S_acc = S_ if S_acc is None else S_acc + S_
    S_meas = S_acc / len(series)
    ts = np.concatenate(series)

    w_t = np.linspace(wf.wmin, wf.wmax, 500)
    S_t = jonswap(w_t, HS, TP)            # normalised on the SAME retained band
    ax[0, 1].plot(w_t, S_t, "k-", lw=2, label="target JONSWAP ($\\gamma$=3.3)")
    ax[0, 1].plot(w_meas, S_meas, "-", c="tab:red", lw=1.2, alpha=.9,
                  label="measured (12 locations averaged)")
    ax[0, 1].axvspan(wf.wmin, wf.wmax, color="tab:green", alpha=.10,
                     label="retained band")
    m0 = np.trapezoid(S_meas, w_meas)
    ax[0, 1].set_xlim(0, 2.5)
    ax[0, 1].set_xlabel("$\\omega$ (rad/s)")
    ax[0, 1].set_ylabel("$S(\\omega)$  (m$^2$ s/rad)")
    ax[0, 1].set_title(f"2. Spectrum recovered: $4\\sqrt{{m_0}}$="
                       f"{4*np.sqrt(m0):.2f} m  (target {HS})", fontsize=10)
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3)

    # --- 3. elevation distribution --------------------------------------
    ax[1, 0].hist(ts, bins=120, density=True, color="tab:blue", alpha=.6,
                  label="synthesised")
    sd = ts.std()
    xs = np.linspace(-4 * sd, 4 * sd, 300)
    ax[1, 0].plot(xs, np.exp(-xs ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)),
                  "k--", lw=1.8, label="Gaussian")
    ax[1, 0].set_xlabel("$\\eta$ (m)"); ax[1, 0].set_ylabel("pdf")
    ax[1, 0].set_title("3. Exactly Gaussian -- which is the MODEL'S LIMIT:\n"
                       "real SS5 waves are skewed (sharp crests, flat troughs)",
                       fontsize=10)
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=.3)

    # --- 4. USV states ---------------------------------------------------
    hull = V.Hull()
    DT = 0.1
    s = V.init_state(1, 6.0)
    rec = []
    tt = 0.0
    for _ in range(int(180 / DT)):
        xs_st = s[:, 0:1] + hull.stations[None, :]
        eta = wf.eta(xs_st.ravel(), np.zeros(xs_st.size), tt).reshape(xs_st.shape)
        # vertical water velocity at the bow, by analytic differentiation
        xb = s[0, 0] + hull.stations[-1]
        ed = -(wf.a * wf.w * np.sin(wf.k * xb * np.cos(wf.th) - wf.w * tt + wf.phi)).sum()
        s, a_bow, slam = V.step(hull, s, np.full(1, 6.0), eta,
                                np.full(1, ed), DT)
        rec.append((tt, s[0, 0], s[0, 1], s[0, 2], np.degrees(s[0, 4]),
                    eta[0, -1], a_bow[0] / V.G))
        tt += DT
    r = np.array(rec)
    a2 = ax[1, 1]
    a2.plot(r[:, 0], r[:, 5], c="tab:cyan", lw=.8, label="$\\eta$ at bow (m)")
    a2.plot(r[:, 0], r[:, 3], c="tab:blue", lw=1.4, label="heave $z$ (m)")
    a2.plot(r[:, 0], r[:, 4] / 5.0, c="tab:orange", lw=1.2,
            label="pitch $\\theta$ (deg/5)")
    a2.plot(r[:, 0], r[:, 2] - 6.0, c="tab:green", lw=1.2,
            label="speed $-6$ (m/s)")
    a2.set_xlim(60, 140); a2.set_xlabel("time (s)")
    a2.set_title(f"4. USV states (const. thrust). travelled "
                 f"{r[-1,1]:.0f} m in 180 s", fontsize=10)
    a2.legend(fontsize=8, ncol=2); a2.grid(alpha=.3)

    fig.suptitle("What the simulator actually reproduces (SS5)", fontsize=13)
    fig.tight_layout()
    fig.savefig("fig5_validation.png", dpi=140)

    loc_hs = [4 * s_.std() for s_ in series]
    print(f"target Hs             = {HS:.2f} m")
    print(f"field Hs (many pts)   = {wf.realised_hs():.2f} m   <- the field is right")
    print(f"field Hs (PSD, 12 loc)= {4*np.sqrt(m0):.2f} m")
    print(f"single-location Hs    : mean {np.mean(loc_hs):.2f}, "
          f"spread {min(loc_hs):.2f}-{max(loc_hs):.2f} m  "
          f"(+-{100*np.std(loc_hs)/np.mean(loc_hs):.0f}%)")
    print(f"  -> one point / one realisation is NOT a converged estimate;")
    print(f"     this is why the sweep must average over seeds.")
    print(f"skewness              = {((ts-ts.mean())**3).mean()/ts.std()**3:+.3f} "
          f"(linear theory forces ~0; real SS5 is +0.1..+0.3)")
    print(f"peak omega            : target {2*np.pi/TP:.3f}, "
          f"measured {w_meas[np.argmax(S_meas)]:.3f} rad/s")
    print("\nfigure: fig5_validation.png")


if __name__ == "__main__":
    main()
