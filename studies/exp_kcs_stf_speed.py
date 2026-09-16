#!/usr/bin/env python3
"""
Experiment: are the KCS short-wave errors the zero-speed hydrodynamics?

exp_kcs_t2015_seakeeping.py ran the project's linear seakeeping model -- BEM
radiation at zero speed, evaluated at the encounter frequency -- against the
T2015 measurements of KCS at Fr 0.26 in head waves (case 2.10), and found

    lambda/L   0.65   0.85   1.15   1.37   1.95
    heave      0.55   0.43   0.77   1.09   1.04     (model / measured)
    pitch      1.65   1.43   0.96   1.05   1.14

What the model leaves out at forward speed is known: the speed terms of strip
theory (Salvesen, Tuck & Faltinsen 1970; Faltinsen 1990, ch. 5). They are
added here to the SAME zero-speed 3-D coefficients, which is the standard
"zero-speed BEM + STF correction" approach, and the result is compared with
the same measurements. With U the speed and w the encounter frequency:

  radiation   A35 = A35_0 - U/w^2 B33_0      B35 = B35_0 + U A33_0
              A53 = A53_0 + U/w^2 B33_0      B53 = B53_0 - U A33_0
              A55 = A55_0 + U^2/w^2 A33_0    B55 = B55_0 + U^2/w^2 B33_0
  diffraction F5 = F5_0 + U/(i w) F3_D       (e^{-iwt}; F3_D = F3 - F3_FK)

Equivalently, in the time domain: the heave force sees the relative
velocity w + U theta, and the pitch moment loses U times the time integral of
the heave force -- which is how the plant can carry them (see the note at the
end of the output). Transom terms are left out: they need the sectional added
mass at the transom, which a 3-D BEM does not give.

Only anchors: the T2015 EFD first harmonics. Radii of gyration kyy = 0.25 L
are the same assumption as the parent experiment, not data.

Run: python -m studies.exp_kcs_stf_speed
"""
import numpy as np

from studies.exp_kcs_t2015_seakeeping import (CASES, K_YY, G, bem, efd,
                                              import_model)


def stf(A0, B0, U, w):
    """Zero-speed 6x6 A, B at encounter frequency w -> STF forward speed."""
    A, B = A0.copy(), B0.copy()
    a33, b33 = A0[2, 2], B0[2, 2]
    A[2, 4] = A0[2, 4] - U / w ** 2 * b33
    A[4, 2] = A0[4, 2] + U / w ** 2 * b33
    B[2, 4] = B0[2, 4] + U * a33
    B[4, 2] = B0[4, 2] - U * a33
    A[4, 4] = A0[4, 4] + U ** 2 / w ** 2 * a33
    B[4, 4] = B0[4, 4] + U ** 2 / w ** 2 * b33
    return A, B


def froude_krylov(mesh, rho, x_G, z_G, w0, beta):
    """Heave Froude-Krylov force per unit amplitude, same body as the BEM."""
    import capytaine as cpt
    from capytaine.bem.airy_waves import froude_krylov_force
    body = cpt.FloatingBody(
        mesh=mesh, lid_mesh=mesh.generate_lid(z=0.0),
        center_of_mass=(x_G, 0.0, z_G),
        dofs=cpt.rigid_body_dofs(rotation_center=(x_G, 0.0, z_G)))
    body = body.immersed_part()
    out = []
    for w in w0:
        pb = cpt.DiffractionProblem(body=body, wave_direction=beta, omega=w,
                                    rho=rho, g=G)
        f = froude_krylov_force(pb)
        out.append([complex(f[d]) for d in ("Surge", "Sway", "Heave", "Roll",
                                            "Pitch", "Yaw")])
    return np.array(out)


def solve(A, B, C, M, F, w, idx):
    ix = np.ix_(idx, idx)
    Z = -w ** 2 * (M[ix] + A[ix]) - 1j * w * B[ix] + C[ix]
    return np.linalg.solve(Z, F[idx])


def main(size_max=4000.0):
    import warnings
    warnings.filterwarnings("ignore")
    case = "2.10"
    c = CASES[case]
    L, U = c["Lpp"], c["U"]
    mesh, rep = import_model(case, size_max)
    x_G, z_G = rep["x_B"], c["KG"] - rep["draft"]
    mass = c["rho"] * rep["volume"]
    k = 2 * np.pi / np.array(c["lam"])
    w0 = np.sqrt(G * k)
    beta = np.pi
    we = w0 + k * U                                   # head seas
    ds = bem(mesh, c["rho"], x_G, z_G, np.r_[w0, we], np.array([beta]),
             tag=f"kcs_{case}_{size_max:g}")
    C6 = ds["C"]
    M6 = np.diag([mass, mass, mass, 0.0, mass * (K_YY * L) ** 2,
                  mass * (K_YY * L) ** 2])
    FK = froude_krylov(mesh, c["rho"], x_G, z_G, w0, beta)
    meas = efd(case)
    idx = [2, 4]
    print(f"\nKCS case {case}: Fr {U/np.sqrt(G*L):.3f}, head waves, heave and "
          f"pitch free; model / measured (EFD first harmonic)\n")
    print(f"  {'lam/L':>6}{'w_e':>6}  {'':6}{'EFD':>7}{'zero-U':>8}"
          f"{'STF rad':>9}{'+diffr':>8}   phase error zero-U / STF+diffr")
    table = {"zero": [], "rad": [], "full": []}
    for i in range(len(k)):
        ie = int(np.argmin(abs(ds["omega"] - we[i])))
        i0 = int(np.argmin(abs(ds["omega"] - w0[i])))
        A0, B0 = ds["A"][ie], ds["B"][ie]
        F0 = ds["F"][i0, 0]
        As, Bs = stf(A0, B0, U, we[i])
        Fs = F0.copy()
        FD3 = F0[2] - FK[i, 2]                    # heave diffraction part
        Fs[4] = F0[4] + U / (1j * we[i]) * FD3
        xs = {"zero": solve(A0, B0, C6, M6, F0, we[i], idx),
              "rad": solve(As, Bs, C6, M6, F0, we[i], idx),
              "full": solve(As, Bs, C6, M6, Fs, we[i], idx)}
        for j, q in enumerate(("Heave", "Pitch")):
            norm = 1.0 if q == "Heave" else k[i]
            sgn = -1.0 if q == "Pitch" else 1.0           # EFD pitch bow up
            ea, ep = meas[i][q]
            r = {lab: abs(x[j]) / norm / ea for lab, x in xs.items()}
            ph = {}
            for lab in ("zero", "full"):
                m = sgn * xs[lab][j] / norm
                p = np.angle(np.exp(1j * (k[i] * L * np.cos(beta)
                                          - np.angle(m))))
                ph[lab] = np.degrees(np.angle(np.exp(1j * (p - ep))))
            for lab in table:
                table[lab].append((q, r[lab]))
            print(f"  {c['lam'][i]/L:>6.2f}{we[i]:>6.2f}  {q:<6}{ea:>7.3f}"
                  f"{r['zero']:>8.2f}{r['rad']:>9.2f}{r['full']:>8.2f}"
                  f"   {ph['zero']:>6.0f} / {ph['full']:>4.0f} deg")
    print("\n  mean |log(model/measured)| over the five conditions "
          "(0 = exact; 0.69 = a factor 2):")
    for q in ("Heave", "Pitch"):
        parts = []
        for lab, nm in (("zero", "zero-U"), ("rad", "STF rad"),
                        ("full", "STF+diffr")):
            v = [abs(np.log(r)) for qq, r in table[lab] if qq == q]
            parts.append(f"{nm} {np.mean(v):.3f}")
        print(f"    {q:<6} " + ",  ".join(parts))
    return table


if __name__ == "__main__":
    main()
