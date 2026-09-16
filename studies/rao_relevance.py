#!/usr/bin/env python3
"""
Does the available experimental data cover the regime this project lives in?

Recommending a validation experiment is easy. Checking that it bears on the
quantity you actually care about is the part that gets skipped, so it is done
here first.

The published Wigley seakeeping experiments (ITTC cooperative programme,
Journee's Delft series) measure heave and pitch response in regular waves,
typically over wavelength/length ratios of roughly 0.5 to 3. This project runs
in Sea State 5, where the spectral peak sits at lambda/L = 14.7 -- far outside
that range, and in a regime where the answer is nearly trivial anyway: a vessel
much shorter than the wave simply follows the surface, so the response ratio
tends to 1 and is set by hydrostatics, which are already verified analytically.

So on the face of it the experiments look irrelevant. They are not, and the
reason is that ELEVATION and ACCELERATION weight the spectrum completely
differently. Acceleration carries a factor of encounter frequency to the fourth
power, which drags the important content down to much shorter waves -- and the
project's headline metrics (bow acceleration, slamming, deck wetness) are all
acceleration-like.

This script measures where each of those quantities actually gets its variance
from. The only model input is the response ratio, and the calculation is run
BOTH with the contouring limit (RAO = 1, an asymptotic fact, not a model output)
and with the model's own heave RAO, so the answer can be seen not to depend on
the model being right.

Run: python -m studies.rao_relevance
"""
import numpy as np

from step0_preview_spec import jonswap

G = 9.81
HS, TP = 3.25, 9.7
L = 10.0
U = 4.5


def bands(u=U, lam_over_L=(1.0, 2.0, 3.0, 5.0, 10.0)):
    """Fraction of variance below each lambda/L, for elevation and acceleration.

    Deep water: lambda = 2 pi g / omega^2, so a wavelength cut is a frequency
    cut. Head seas: the vessel meets the waves faster than they travel, so the
    frequency it FEELS is omega_e = omega + omega^2 u / g, and it is omega_e
    that differentiates the motion into an acceleration.
    """
    w = np.linspace(0.05, 6.0, 24000)
    S = jonswap(w, HS, TP)
    lam = 2 * np.pi * G / w ** 2
    we = w + w ** 2 * u / G

    weights = {
        "elevation": np.ones_like(w),
        "vertical velocity": we ** 2,
        "vertical acceleration": we ** 4,
    }
    out = {}
    for name, wt in weights.items():
        d = wt * S
        tot = np.trapezoid(d, w)
        out[name] = [float(np.trapezoid(np.where(lam <= c * L, d, 0.0), w) / tot)
                     for c in lam_over_L]
    return lam_over_L, out


def with_model_rao():
    """Repeat the acceleration split using the model's own heave RAO.

    If the two answers agree, the conclusion does not rest on the model.
    """
    from hydro import bem
    from sim.cummins import radiation_memory
    db = bem.load("hydro_wigley_10m.npz")
    rad = radiation_memory(db)

    w = np.linspace(0.15, 5.0, 1200)
    S = jonswap(w, HS, TP)
    we = w + w ** 2 * U / G
    # Linear heave RAO, the same approximation the plant makes: zero-speed
    # radiation coefficients evaluated at the ENCOUNTER frequency (that is what
    # the Cummins convolution does in the time domain, since it responds at
    # whatever frequency the motion has), and the excitation transfer function
    # at the wave's own frequency.
    #   (C - (M + A(we)) we^2 + i B(we) we) z = F(w)
    A = np.interp(we, db.omega, db.A[:, 2, 2])
    B = np.interp(we, db.omega, db.B[:, 2, 2])
    F = np.interp(w, db.omega, np.abs(db.F_exc[:, 0, 2]))
    Z = (db.C[2, 2] - (db.M[2, 2] + A) * we ** 2) + 1j * B * we
    rao = np.abs(F / Z)                      # metres of heave per metre of wave

    lam = 2 * np.pi * G / w ** 2
    d = (we ** 4) * (rao ** 2) * S
    tot = np.trapezoid(d, w)
    frac = [float(np.trapezoid(np.where(lam <= c * L, d, 0.0), w) / tot)
            for c in (1.0, 2.0, 3.0, 5.0, 10.0)]
    return frac, lam[np.argmax(d)] / L, rao


def main():
    cuts, out = bands()
    print(f"\n  Sea State 5, Hs {HS} m, Tp {TP} s, {U} m/s head seas, L = {L} m")
    print(f"  spectral peak sits at lambda/L = "
          f"{2*np.pi*G/(2*np.pi/TP)**2/L:.1f}\n")
    print("  share of variance coming from waves SHORTER than lambda/L =")
    print(f"  {'quantity':<24}" + "".join(f"{c:>9.0f}" for c in cuts))
    for name, fr in out.items():
        print(f"  {name:<24}" + "".join(f"{f:>8.1%}" for f in fr))

    frac, lam_peak, rao = with_model_rao()
    print(f"  {'accel, model heave RAO':<24}"
          + "".join(f"{f:>8.1%}" for f in frac))

    i3, i10 = cuts.index(3.0), cuts.index(10.0)
    print(f"\n  Elevation takes {out['elevation'][i3]:.1%} of its variance from")
    print(f"  below lambda/L = 3. Acceleration takes "
          f"{out['vertical acceleration'][i3]:.0%} on the contouring")
    print(f"  assumption and {frac[i3]:.0%} with the model's own RAO. Two things")
    print(f"  follow, and the second is the more useful one.\n")
    print(f"  1. Whichever row you believe, the sea state's own statistics are")
    print(f"     nearly irrelevant to the metrics being optimised. Elevation is")
    print(f"     long swell this hull rides over; acceleration is not.")
    print(f"     Acceleration density peaks at lambda/L = {lam_peak:.1f}, not at")
    print(f"     the spectral peak's 14.7.")
    print(f"\n  2. The two rows DISAGREE by a factor of {out['vertical acceleration'][i3]/max(frac[i3],1e-9):.1f}, and the only")
    print(f"     thing separating them is the RAO. RAO = 1 says a 10 m hull")
    print(f"     follows a 10 m wave; the model says it bridges it instead. That")
    print(f"     is a large, physically decisive difference, and it is exactly")
    print(f"     the quantity the experiments measure and this project has never")
    print(f"     checked. The disagreement is the argument for running the")
    print(f"     comparison, not an obstacle to it.")
    print(f"\n  Coverage, stated honestly: the published range (roughly")
    print(f"  lambda/L 0.5-3) contains {frac[i3]:.0%} of the model's acceleration")
    print(f"  variance, and {frac[i10]:.0%} lies below lambda/L = 10. So the")
    print(f"  experiments test a real and substantial part of the band that")
    print(f"  matters here -- the part where added mass, radiation damping and")
    print(f"  diffraction decide the answer -- but not all of it. The upper")
    print(f"  part, lambda/L 3-10, would still rest on the BEM alone.")


if __name__ == "__main__":
    main()
