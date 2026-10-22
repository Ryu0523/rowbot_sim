#!/usr/bin/env python3
"""
Symbol analysis of the upstream phi_xx operators of hydro/rankine/steady.py
(evidence for learn/meta/RANKINE_STAGE1_2026-10-06.md section 2.5 and the D1
criterion in learn/meta/WAVEBEM_PORT_DESIGN.md; added after the 2026-10-06
review). Runs in about a second, CPU only.

    .venv/Scripts/python.exe studies/rankine_symbol.py

For a free-surface wave e^{i alpha x} (2-D, transverse) the discrete
condition k D(alpha) + |alpha| = 0, k = 1/k0, with
D(alpha) = sum_m w_m e^{i m alpha dx}/dx^2 (point m lies m cells upstream),
has a root alpha = k0 (1 - e) near k0. Re alpha gives the wavelength error,
Im alpha > 0 a growth of the wave amplitude downstream (x -> -inf) by
exp(Im alpha |x|), Im alpha < 0 a damping. Printed per wavelength
lambda0 = 2 pi/k0 for 20, 30, 45 panels per wavelength. Also: the moments
sum w_m m^n (n = 0..6), whose first nonzero term after n = 2 gives the
leading error, and the parasitic roots of the stencil polynomial (|z| < 1:
they decay downstream). This is the 1-D symbol only: the coupled panel
system can behave differently (upwind7 is unstable there at 20 panels per
wavelength although its 1-D symbol damps; stage note section 5). The
analysis of the coupled system that decided D1 (stage B3) is
studies/rankine_d1.py: it explains that instability (upwind rows are
multistep methods of order > 2, unstable on the short diverging waves) and
evaluates the production operator of hydro/rankine/shifted.py.
"""
import os
import sys

import numpy as np
from scipy.optimize import fsolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hydro.rankine.steady import upwind_weights  # noqa: E402


def root(w, nl):
    dx = 2 * np.pi / nl                       # k0 = 1

    def F(z):
        a = z[0] + 1j * z[1]
        r = np.sum(w * np.exp(1j * a * np.arange(len(w)) * dx)) / dx ** 2 + a
        return [r.real, r.imag]
    re, im = fsolve(F, [1.0, 0.0], xtol=1e-13)
    return 1.0 / re - 1.0, float(np.exp(im * 2 * np.pi) - 1.0)


if __name__ == "__main__":
    for s in ("dawson4", "upwind5", "upwind6", "upwind7"):
        w = upwind_weights(s)
        mom = [float(np.sum(w * np.arange(len(w)) ** n)) for n in range(7)]
        par = np.roots(w)
        print(f"{s}: weights x12 {np.round(12 * w, 4).tolist()}")
        print(f"   moments n=0..6 {np.round(mom, 4).tolist()}")
        print(f"   stencil roots |z| {np.round(np.sort(np.abs(par)), 4).tolist()}")
        for nl in (20, 30, 45):
            lam_err, growth = root(w, nl)
            print(f"   {nl} per wavelength: wavelength error {lam_err:+.4f}, amplitude change per wavelength "
                  f"{growth:+.4f}")
