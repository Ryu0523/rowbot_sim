#!/usr/bin/env python3
"""
Experiment: KCS in regular waves at Fr 0.26 against the T2015 measurements --
heave and pitch in head waves (case 2.10), heave, pitch and roll over five
headings (case 2.11).

THE DATA
Tokyo 2015 CFD workshop (NMRI), KCS container ship, towed at Fr 0.26.
First-harmonic amplitudes and phases of the measured motions, from the
workshop's condition spreadsheets (data/external/kcs_t2015/*.xlsx); the time
histories in the same folder are these harmonics resynthesised. Conditions,
from the workshop's instruction pages:
  2.10  model 2, Lpp 6.0702 m, U 2.017 m/s, head waves, free in heave and
        pitch; lambda 3.949 5.164 6.979 8.321 11.840 m,
        H 0.062 0.078 0.123 0.149 0.196 m
  2.11  model 3, Lpp 2.7 m, U 1.34 m/s, lambda 2.7 m, H 0.045 m, headings
        chi = 0 (head) 45 90 135 180 (following); surge, heave, roll and pitch
        reported
Motions at G; t = 0 when a crest is at the FP; z/zeta, theta/(k zeta) bow up
positive, phi/(k zeta) port up positive.

WHAT IS TESTED
The project's seakeeping model in its linear limit, on a hull it was not
built for: zero-speed BEM radiation at the ENCOUNTER frequency, zero-speed
excitation at the WAVE frequency, its phase carried at the encounter
frequency -- which is what sim/vessel.py does by evaluating the wavefield at
the moving position. There are no forward-speed terms in the radiation or the
diffraction; that is the approximation under test. At Fr 0.26 the terms it
leaves out (the U/w couplings of strip theory) are not small.

And one thing that is not an approximation but a sign: sim/cummins.py's
frequency-domain reference solves [-w^2 (M+A) + i w B + C] xi = F, while
Capytaine's excitation is for e^{-iwt}, whose impedance has -i w B. Both are
printed. On a hull with little heave-pitch coupling the two give the same
amplitudes, which is how it went unnoticed on the fore-aft symmetric Wigley.

INPUTS NOT IN THE DATA -- assumptions, with the sensitivity printed
  radii of gyration kyy = 0.25 Lpp, kxx = 0.40 B (the SIMMAN KCS loading
  condition; neither the T2015 pages nor the spreadsheets give any)
  LCG = LCB (level trim); rudder left out of the BEM; sway and yaw held (2.11)
  or also surge held (2.10) by the towing; no bilge keels (not stated).

Run: python -m studies.exp_kcs_t2015_seakeeping
"""
import html
import os
import re
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IGS = os.path.join(HERE, "data", "external", "kcs_geometry",
                   "KCS_hull_Case2-11", "FinalHull_KCS.igs")
DATA = os.path.join(HERE, "data", "external", "kcs_t2015")
XLSX = {"2.10": "[Identifier]_6conditions_2-10_20150914.xlsx",
        "2.11": "[Identifier]_6conditions_2-11_20151112.xlsx"}
G = 9.81
KCS = dict(Lpp=230.0, B=32.2, T=10.8, V=52030.0, S=9424.0, LCB_pct=-1.48,
           C_B=0.6505, C_M=0.9849)
CASES = {
    "2.10": dict(Lpp=6.0702, B=0.8498, T=0.2850, V=0.9571, KG=0.378,
                 GM=0.0158, U=2.017, rho=998.63, nu=1.14e-6, S=6.6177,
                 lam=[3.949, 5.164, 6.979, 8.321, 11.840],
                 H=[0.062, 0.078, 0.123, 0.149, 0.196], chi=[0.0] * 5,
                 free=["Heave", "Pitch"]),
    "2.11": dict(Lpp=2.7, B=0.378, T=0.1268, V=0.084, KG=0.168, GM=0.007,
                 U=1.34, rho=997.8858, nu=9.679e-7, S=1.31, lam=[2.7] * 5,
                 H=[0.045] * 5, chi=[0.0, 45.0, 90.0, 135.0, 180.0],
                 free=["Heave", "Roll", "Pitch"]),
}
K_XX, K_YY = 0.40, 0.25          # x B, x Lpp -- assumptions, see above
DOF6 = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]
# spreadsheet columns of the first harmonic (amplitude, phase in rad)
COLS = {"2.10": {"Heave": ("L", "P"), "Pitch": ("U", "Y")},
        "2.11": {"Heave": ("U", "Y"), "Roll": ("AD", "AH"),
                 "Pitch": ("AM", "AQ")}}


def efd(case):
    """First-harmonic amplitude and phase per condition, from the
    workshop spreadsheet's 'EFD (D)' rows (tables C1..C5 in order)."""
    z = zipfile.ZipFile(os.path.join(DATA, XLSX[case]))
    ss = [html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S)))
          for si in re.findall(r"<si>(.*?)</si>",
                               z.read("xl/sharedStrings.xml").decode(), re.S)]
    x = z.read("xl/worksheets/sheet1.xml").decode()
    rows = {}
    for col, row, attrs, inner in re.findall(
            r'<c r="([A-Z]+)(\d+)"([^>]*?)(?:/>|>(.*?)</c>)', x, re.S):
        v = re.search(r"<v>(.*?)</v>", inner or "")
        if v:
            val = ss[int(v.group(1))] if 't="s"' in attrs else v.group(1)
            rows.setdefault(int(row), {})[col] = val
    out = []
    for r in sorted(rows):
        if rows[r].get("A", "").startswith("EFD") and "L" in rows[r] or (
                rows[r].get("A", "").startswith("EFD") and "U" in rows[r]):
            d = {}
            for q, (ca, cp) in COLS[case].items():
                if ca in rows[r] and cp in rows[r]:
                    d[q] = (float(rows[r][ca]), float(rows[r][cp]))
            if d:
                out.append(d)
    return out                     # the calm-water row has no motion columns


def rudder_patch(s):
    """Thin patches aft of x = 8 m (full scale), within 1 m of the centreplane."""
    def f(lo, hi, area):
        return hi[0] < 8.0 * s and max(abs(lo[1]), abs(hi[1])) < 1.0 * s
    return f


def import_model(case, size_max=4000.0):
    from hydro.cad_import import import_hull
    c = CASES[case]
    s = c["Lpp"] / KCS["Lpp"]                    # model / full
    # The rudder, the propeller boss caps and the forecastle bulwark are
    # separate connected pieces in this file; import_hull drops them as such
    # (rudder_patch above was the first, geometric, attempt and cut into the
    # stern). The published wetted area is without the rudder, like the mesh.
    mesh, rep = import_hull(IGS, volume=c["V"], scale=1e-3 * s,
                            size_max=size_max, half=False, forward="+x",
                            up="+z", min_component=0.1, verbose=True)
    return mesh, rep


CACHE = os.path.join(HERE, "studies", "_cache")


def bem(mesh, rho, x_G, z_G, omegas, betas, tag):
    """A, B (n_w, 6, 6), F (n_w, n_dir, 6), C (6, 6) as plain arrays, cached
    on disk by `tag` -- the solve is minutes, the analysis is not."""
    path = os.path.join(CACHE, f"bem_{tag}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    import capytaine as cpt
    import xarray as xr
    body = cpt.FloatingBody(
        mesh=mesh, lid_mesh=mesh.generate_lid(z=0.0),
        center_of_mass=(x_G, 0.0, z_G),
        dofs=cpt.rigid_body_dofs(rotation_center=(x_G, 0.0, z_G)))
    body = body.immersed_part()
    w = np.unique(omegas)
    b = np.atleast_1d(np.unique(betas))
    test = xr.Dataset({"omega": w, "wave_direction": b,
                       "radiating_dof": DOF6, "rho": [rho], "g": [G],
                       "water_depth": [np.inf]})
    ds = cpt.BEMSolver().fill_dataset(test, body, _check_wavelength=False)
    ds = ds.squeeze([d for d in ("rho", "g", "water_depth") if d in ds.dims])
    A = ds["added_mass"].transpose("omega", "influenced_dof",
                                   "radiating_dof").sel(
        influenced_dof=DOF6, radiating_dof=DOF6).values
    B = ds["radiation_damping"].transpose("omega", "influenced_dof",
                                          "radiating_dof").sel(
        influenced_dof=DOF6, radiating_dof=DOF6).values
    Fx = ds["excitation_force"]
    if "wave_direction" not in Fx.dims:
        Fx = Fx.expand_dims(wave_direction=b)
    F = Fx.transpose("omega", "wave_direction", "influenced_dof").sel(
        influenced_dof=DOF6).values
    C = body.compute_hydrostatic_stiffness(rho=rho, g=G)
    C = np.asarray(C.values if hasattr(C, "values") else C, float)
    out = dict(omega=w, beta=b, A=A, B=B, F=F, C=C)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **out)
    return out


def rao(db, C6, M6, free, w_e, w_0, beta, B_extra=None, sign=-1.0):
    """Linear response of the free dofs; sign -1: Capytaine's e^{-iwt}
    (impedance -i w B); sign +1: the form sim/cummins.py used (+i w B)."""
    idx = [DOF6.index(d) for d in free]
    ie = int(np.argmin(abs(db["omega"] - w_e)))
    i0 = int(np.argmin(abs(db["omega"] - w_0)))
    ib = int(np.argmin(abs(db["beta"] - beta)))
    A = db["A"][ie][np.ix_(idx, idx)]
    B = db["B"][ie][np.ix_(idx, idx)]
    if B_extra is not None:
        B = B + B_extra
    F = db["F"][i0, ib, idx]
    M = M6[np.ix_(idx, idx)]
    C = C6[np.ix_(idx, idx)]
    Z = -w_e ** 2 * (M + A) + sign * 1j * w_e * B + C
    return np.linalg.solve(Z, F)


def main(size_max=4000.0):
    import warnings
    warnings.filterwarnings("ignore")
    from hydro import ikeda
    print("\nEXPERIMENT -- KCS at Fr 0.26 in regular waves, the project's "
          "linear seakeeping model against T2015\n")
    results = {}
    for case, c in CASES.items():
        print(f"== case {case}: model Lpp {c['Lpp']} m, U {c['U']} m/s")
        mesh, rep = import_model(case, size_max)
        L = c["Lpp"]
        x_G = rep["x_B"]                                  # LCG = LCB
        z_G = c["KG"] - rep["draft"]
        mass = c["rho"] * rep["volume"]
        print(f"  hydrostatics: draft {rep['draft']:.4f} (pub {c['T']}), "
              f"LCB {100*(x_G - L/2)/L:+.2f}% (pub {KCS['LCB_pct']}), "
              f"GM {rep['KM'] - c['KG']:.4f} (pub {c['GM']}), wetted "
              f"{rep['wetted_area']:.3f} m2 (pub {c['S']} w/o rudder), "
              f"{rep['n_panels']} panels")
        k = 2 * np.pi / np.array(c["lam"])
        w0 = np.sqrt(G * k)
        beta = np.radians(180.0 - np.array(c["chi"]))     # Capytaine: pi = head
        we = w0 - k * c["U"] * np.cos(beta)
        ds = bem(mesh, c["rho"], x_G, z_G, np.r_[w0, we], np.unique(beta),
                 tag=f"kcs_{case}_{size_max:g}")
        C6 = ds["C"].copy()
        # the published GM is used for roll, the mesh's own for the rest
        C6[3, 3] = c["rho"] * G * rep["volume"] * c["GM"]
        M6 = np.diag([mass, mass, mass, mass * (K_XX * c["B"]) ** 2,
                      mass * (K_YY * L) ** 2, mass * (K_YY * L) ** 2])
        meas = efd(case)
        rows = []
        for i in range(len(k)):
            Bx = None
            if "Roll" in c["free"]:
                # Ikeda viscous roll damping, equivalent linear at the
                # predicted amplitude (fixed point), lift at forward speed
                Bx = np.zeros((len(c["free"]),) * 2)
                ir = c["free"].index("Roll")
                amp = np.radians(3.0)
                for _ in range(20):
                    r = ikeda.roll_damping(
                        L=L, B=c["B"], d=rep["draft"], C_B=KCS["C_B"],
                        C_M=KCS["C_M"], KG=c["KG"], w=we[i], phi_a=amp,
                        U=c["U"], V=rep["volume"], rho=c["rho"], nu=c["nu"],
                        g=G, B_W0=0.0)
                    Bx[ir, ir] = r["F"] + r["E"] + r["L"]
                    xi = rao(ds, C6, M6, c["free"], we[i], w0[i], beta[i],
                             Bx) * c["H"][i] / 2
                    new = max(abs(xi[ir]), np.radians(0.2))
                    if abs(new - amp) < 1e-4 * amp:
                        break
                    amp = new
            out = {}
            for lab, sgn, bx in (("BEM + Ikeda", -1.0, Bx),
                                 ("BEM only", -1.0, None),
                                 ("cummins.py sign", +1.0, Bx)):
                xi = rao(ds, C6, M6, c["free"], we[i], w0[i], beta[i], bx, sgn)
                out[lab] = xi
            rows.append((c["chi"][i], c["lam"][i] / L, we[i], k[i], out))
        results[case] = (rows, meas)
        # --- report
        xFP = L                                    # FP at x = Lpp (AP at 0)
        print(f"\n  {'cond':<5}{'lam/L':>6}{'chi':>5}{'w_e':>7}  quantity"
              f"{'EFD':>8}{'model':>8}{'ratio':>7}{'d-phase':>9}"
              f"{'BEM only':>10}{'+iwB':>8}")
        for i, (chi, lamL, w_e, kk, out) in enumerate(rows):
            for q in c["free"]:
                j = c["free"].index(q)
                norm = 1.0 if q == "Heave" else kk
                sgn_efd = -1.0 if q == "Pitch" else 1.0     # EFD pitch bow up
                vals = {lab: sgn_efd * v[j] / norm for lab, v in out.items()}
                m = vals["BEM + Ikeda"]
                # phase in the EFD's convention: y = a cos(w_e t' + ph),
                # t' = 0 at a crest at the FP; ours is Re[xi e^{-i w_e t}]
                ph = np.angle(np.exp(1j * (kk * xFP * np.cos(beta[i])
                                           - np.angle(m))))
                ea, ep = meas[i].get(q, (np.nan, np.nan)) if i < len(meas) \
                    else (np.nan, np.nan)
                dph = np.degrees(np.angle(np.exp(1j * (ph - ep))))
                print(f"  C{i+1:<4}{lamL:>6.2f}{chi:>5.0f}{w_e:>7.2f}  "
                      f"{q:<8}{ea:>8.3f}{abs(m):>8.3f}{abs(m)/ea:>7.2f}"
                      f"{dph:>9.0f}{abs(vals['BEM only']):>10.3f}"
                      f"{abs(vals['cummins.py sign']):>8.3f}")
        # the radii of gyration are assumptions: how much do they move it?
        print(f"\n  sensitivity to the assumed radii (amplitude, as a ratio to "
              f"the table above):")
        for lab, kx, ky in (("kyy 0.24 Lpp", K_XX, 0.24),
                            ("kyy 0.26 Lpp", K_XX, 0.26),
                            ("kxx 0.36 B", 0.36, K_YY),
                            ("kxx 0.44 B", 0.44, K_YY)):
            Ms = M6.copy()
            Ms[3, 3] = mass * (kx * c["B"]) ** 2
            Ms[4, 4] = Ms[5, 5] = mass * (ky * L) ** 2
            parts = []
            for i, (chi, lamL, w_e, kk, out) in enumerate(rows):
                xi = rao(ds, C6, Ms, c["free"], w_e, w0[i], beta[i],
                         None if "Roll" not in c["free"] else Bx)
                base = out["BEM + Ikeda"]
                parts.append("/".join(f"{abs(xi[j])/max(abs(base[j]),1e-12):.2f}"
                                      for j in range(len(c["free"]))))
            print(f"    {lab:<13} " + "  ".join(f"C{i+1} {p}" for i, p in
                                             enumerate(parts))
                  + f"   ({'/'.join(c['free'])})")
        print()
    return results


if __name__ == "__main__":
    main()
