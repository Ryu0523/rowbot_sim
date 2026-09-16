#!/usr/bin/env python3
"""
Step 0 -- the spec-generating experiment for the USV RL+MPC programme.

Answers two questions BEFORE a hull is chosen, a sensor is bought, or a
line of RL is written:

  Q1  How much usable wave-preview TIME does a sensing aperture buy me,
      given sea state, vessel speed and heading?
  Q2  Is that aperture physically reachable with LiDAR from a mast, or
      does grazing-angle physics force an airborne sensor?

Deep-water linear wave theory throughout. That is the correct fidelity for
SIZING a sensor. It is not the fidelity you train the final policy on.
"""
import numpy as np

G = 9.81

# WMO sea states; hs/tp are mid-band representative values.
SEA_STATES = {
    4: dict(hs=1.88, tp=8.0, hs_band=(1.25, 2.50), wind_kn=19.0),
    5: dict(hs=3.25, tp=9.7, hs_band=(2.50, 4.00), wind_kn=24.5),
}

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ------------------------------------------------------------------ spectra
def jonswap(omega, hs, tp, gamma=3.3):
    """JONSWAP density, rescaled so that 4*sqrt(m0) == hs exactly."""
    omega = np.asarray(omega, float)
    wp = 2.0 * np.pi / tp
    sigma = np.where(omega <= wp, 0.07, 0.09)
    r = np.exp(-((omega - wp) ** 2) / (2.0 * sigma ** 2 * wp ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        S = omega ** -5.0 * np.exp(-1.25 * (wp / omega) ** 4) * gamma ** r
    S = np.where(omega > 0, S, 0.0)
    return S * (hs / 4.0) ** 2 / _trapz(S, omega)


def spreading(theta, theta0=np.pi, s=8):
    """cos^{2s} directional spreading, normalised to unit integral."""
    arg = np.clip((theta - theta0) / 2.0, -np.pi / 2, np.pi / 2)
    d = np.cos(arg) ** (2 * s)
    return d / _trapz(d, theta)


class WaveField:
    """Short-crested linear wave field. Query eta(x, y, t) anywhere, anytime.

    This doubles as the ORACLE PREVIEW: the same object that forces the vessel
    can be sampled ahead of it, which is exactly what Phase 0 needs.

    theta0 is the direction the waves TRAVEL TOWARD, so theta0 = pi is head
    seas for a vessel running along +x.
    """

    def __init__(self, hs, tp, gamma=3.3, theta0=np.pi, spread_s=8,
                 n_freq=64, n_dir=16, band=(0.5, 3.0), seed=0):
        wp = 2.0 * np.pi / tp
        self.wmin, self.wmax = band[0] * wp, band[1] * wp
        w = np.linspace(self.wmin, self.wmax, n_freq)
        # n_freq = 1 is a legitimate degenerate case -- a single component
        # standing in for the whole band -- and it used to raise IndexError
        # here. Give it the full band width so the energy still comes out
        # right, the same way n_dir = 1 is handled explicitly below.
        dw = (w[1] - w[0]) if n_freq > 1 else (self.wmax - self.wmin)
        if n_dir == 1:
            # long-crested: all energy on one heading. Not a degenerate case of
            # the spread formula -- the normalisation integral collapses -- so
            # it is handled explicitly.
            th = np.array([theta0])
            amp = np.sqrt(2.0 * jonswap(w, hs, tp, gamma)[:, None] * dw)
        else:
            th = np.linspace(theta0 - np.pi / 2, theta0 + np.pi / 2, n_dir)
            dth = th[1] - th[0] if n_dir > 1 else np.pi
            amp = np.sqrt(2.0 * np.outer(jonswap(w, hs, tp, gamma),
                                         spreading(th, theta0, spread_s))
                          * dw * dth)
        rng = np.random.default_rng(seed)
        self.a = amp.ravel()
        self.w = np.repeat(w, n_dir)
        self.th = np.tile(th, n_freq)
        self.k = self.w ** 2 / G                      # deep-water dispersion
        self.phi = rng.uniform(0, 2 * np.pi, self.a.size)
        self.hs, self.tp, self.theta0 = hs, tp, theta0

    def eta(self, x, y, t):
        x = np.atleast_1d(np.asarray(x, float))[..., None]
        y = np.atleast_1d(np.asarray(y, float))[..., None]
        phase = (self.k * (x * np.cos(self.th) + y * np.sin(self.th))
                 - self.w * t + self.phi)
        return (self.a * np.cos(phase)).sum(-1)

    def realised_hs(self, n=200_000, seed=1):
        """Sanity check: 4*std of the synthesised surface should match Hs."""
        rng = np.random.default_rng(seed)
        e = self.eta(rng.uniform(0, 4000, n), rng.uniform(0, 4000, n), 0.0)
        return 4.0 * e.std()

    def cg_bounds(self, band_frac=(0.0, 1.0)):
        """Group-velocity extremes of a sub-band. Deep water: cg = g / (2w)."""
        lo = self.wmin + band_frac[0] * (self.wmax - self.wmin)
        hi = self.wmin + band_frac[1] * (self.wmax - self.wmin)
        return G / (2 * hi), G / (2 * lo)              # (cg_min, cg_max)


# ---------------------------------------------------- predictable-zone theory
def predictable_window(x1, x2, U, cg_min, cg_max, following=False):
    """Times t at which the surface AT THE MOVING VESSEL is fully determined
    by a snapshot taken at t=0 over the aperture [x1, x2] metres ahead.

    Every spectral component reaching the vessel at time t must have started
    inside the aperture, which gives
        t >= x1 / (cg_min + U)   and   t <= x2 / (cg_max + U)
    Note it opens LATE and shuts early: preview is a WINDOW, not a horizon.
    Returns None if the aperture yields no fully-determined instant.
    """
    rel_min = (cg_min - U) if following else (cg_min + U)
    rel_max = (cg_max - U) if following else (cg_max + U)
    if rel_min <= 0 or rel_max <= 0:
        return None                                    # overtaking / surf-riding
    t_open, t_close = x1 / rel_min, x2 / rel_max
    return (t_open, t_close) if t_close > t_open else None


# ------------------------------------------------------- LiDAR reach geometry
def lidar_reach(height_m, max_incidence_deg):
    """Horizontal range at which the beam still strikes water inside the usable
    incidence cone. Incidence is measured from NADIR, so R = h * tan(incidence).

    Water is near-specular in the IR: past ~60 deg off nadir the return
    collapses. Whitecaps and steep faces at SS4-5 buy maybe 10-15 deg more.
    """
    return height_m * np.tan(np.radians(max_incidence_deg))


APERTURES = [(5, 20), (10, 50), (20, 150), (20, 300), (50, 400)]
SPEEDS = (3.0, 5.0, 8.0)


def main():
    line = "=" * 76
    print(line)
    print("STEP 0 -- WAVE PREVIEW FEASIBILITY".center(76))
    print(line)

    # ---- 1. sea state kinematics -----------------------------------------
    print("\n[1] SEA STATE KINEMATICS (deep water, peak component)\n")
    print("    SS      Hs(m)    Tp(s)   lambda_p(m)    c_p(m/s)   cg_p(m/s)")
    for ss, p in SEA_STATES.items():
        tp = p["tp"]
        cp = G * tp / (2 * np.pi)
        print(f"    {ss:<8}{p['hs']:<9.2f}{tp:<9.1f}{1.56 * tp ** 2:<14.0f}"
              f"{cp:<12.1f}{cp / 2:<.1f}")
    print("\n    -> Wave GROUPS -- the exploitable structure -- travel at c/2.")

    # ---- 2. can a LiDAR even see that far? -------------------------------
    print("\n[2] LiDAR HORIZONTAL REACH vs SENSOR HEIGHT (metres)\n")
    print("    height(m)   @60deg    @70deg    @75deg    platform")
    for h, plat in [(3, "small USV mast"), (10, "tall mast / A-frame"),
                    (30, "tethered UAV"), (50, "UAV"), (100, "UAV, high")]:
        r = [lidar_reach(h, a) for a in (60, 70, 75)]
        print(f"    {h:<12}{r[0]:<10.0f}{r[1]:<10.0f}{r[2]:<10.0f}{plat}")
    print("\n    -> Reach scales with HEIGHT, not laser power. Crest shadowing")
    print("       at SS4-5 makes the 75deg column optimistic.")

    # ---- 3. the predictable window ---------------------------------------
    print("\n[3] PREDICTABLE WINDOW AT THE VESSEL, head seas (s, open-close)\n")
    hdr = "".join(f"{f'[{a},{b}]':>13}" for a, b in APERTURES)
    for ss in (4, 5):
        p = SEA_STATES[ss]
        wf = WaveField(p["hs"], p["tp"], seed=0)
        print(f"    SS{ss}:  Hs={p['hs']} m  Tp={p['tp']} s   "
              f"(synthesised Hs = {wf.realised_hs():.2f} m)")
        for label, frac in [("full band", (0.0, 1.0)),
                            ("energetic band", (0.15, 0.55))]:
            cg_min, cg_max = wf.cg_bounds(frac)
            print(f"      {label} -- cg in [{cg_min:.1f}, {cg_max:.1f}] m/s")
            print(f"        {'aperture:':<12}{hdr}")
            for U in SPEEDS:
                cells = ""
                for x1, x2 in APERTURES:
                    win = predictable_window(x1, x2, U, cg_min, cg_max)
                    txt = "--" if win is None else f"{win[0]:.1f}-{win[1]:.1f}"
                    cells += f"{txt:>13}"
                print(f"        U={U:<10.0f}{cells}")
        print()
    print("    -> '--' means that aperture yields NO fully-determined instant.")

    print(line)
    print("READ-OFF")
    print("  * Preview needs a long CONTINUOUS aperture, close-in to far.")
    print("    A narrow patch at a single range buys you nothing.")
    print("  * Predicting only the energetic band widens the window a lot,")
    print("    and those are the components that actually drive the motion.")
    print("  * Cross-reference [2] against [3]: the aperture that works needs")
    print("    a sensor height no mast on a small USV can carry.")
    print(line)


if __name__ == "__main__":
    main()
