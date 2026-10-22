"""
Kramers-Kronig evaluation for the double-body tables (Track B step B6, stage note 11): the same quadrature as
rankine_unsteady_verify._kk_errors (forward-speed relation of stage note 10.2 item 5, corrected [0, omega_0]
term of 10.8.3), with the entry-skip reference taken over the TEST BAND only: in the double-body model the static
heave stiffness C0_33 is nonzero (the base flow's m_3), so A33(omega) ~ -C0_33/omega^2 dominates the low end of the
frequency grid and the B5 rule (1e-3 x max over ALL omega of |A33|) would skip every entry.
"""
import numpy as np


def kk_errors_band(w, T, Ainf, Binf, C0, wtest, wband):
    from scipy.interpolate import PchipInterpolator
    A = -T.real / w[:, None, None] ** 2
    B = T.imag / w[:, None, None]
    band = (w >= wband[0]) & (w <= wband[1])
    a33 = np.abs(A[band, 1, 1]).max()
    out = {}
    for j in range(3):
        for k in range(3):
            Aref = np.abs(A[band, j, k]).max()
            if Aref < 1e-3 * a33:
                continue
            f = B[:, j, k] - Binf[j, k]
            fi = PchipInterpolator(np.log(w), f)
            wq = np.exp(np.linspace(np.log(w[0]), np.log(w[-1]), 200001))
            fq = fi(np.log(wq))
            errs = []
            for wt in wtest:
                ft = float(fi(np.log(wt)))
                den = wq ** 2 - wt ** 2
                with np.errstate(divide="ignore", invalid="ignore"):
                    ig = np.where(np.abs(den) > 1e-12, (fq - ft) / den, 0.0)
                I1 = np.trapezoid(ig * wq, np.log(wq))
                I2 = ft * (np.log(abs((w[-1] - wt) / (w[-1] + wt))) - np.log(abs((w[0] - wt) / (w[0] + wt)))) / (2 * wt)
                I0 = float(f[0]) * np.log(abs((w[0] - wt) / (w[0] + wt))) / (2 * wt)
                Akk = Ainf[j, k] - C0[j, k] / wt ** 2 + 2 / np.pi * (I1 + I2 + I0)
                Ad = float(PchipInterpolator(np.log(w), A[:, j, k])(np.log(wt)))
                errs.append((Akk - Ad) / Aref)
            out[(j, k)] = np.array(errs)
    return out
