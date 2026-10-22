#!/usr/bin/env python3
"""
V9 follow-up (2026-10-09): which zero-speed wave excitation is right for scarab195
at 3.9 and 4.7 rad/s -- the old Capytaine probe table (bem_probe_lid_24_64.npz,
used by anyspeed_rest_rao.py, B1.3, and V1) or the repo's Rankine solve (V2,
derive_v2_hull.py / derive_v2_second_order.py)?

Steps
 1. Read the old table; print both wave-direction columns (index 0 = beta 0,
    index 3 = beta pi) next to the V2 Rankine head-seas values.
 2. Capytaine's direction convention, checked from its own Airy-wave function
    (phase of the incident potential along x) -- not from documentation.
 3. Rebuild the probe's mesh/lid/body (same code path as anyspeed_bem_probe.py)
    and re-solve at 3.0 / 3.04 / 3.89 / 3.9 / 4.7 / 4.74 rad/s for 13 wave
    directions 0..pi; reproduce the table at its own frequencies, give head and
    following seas at the Rankine frequencies, and check two Haskind identities:
      (a) Haskind-Newman: B_jj = omega^3/(4 pi rho g^3) * int_0^{2pi} |X_j(beta)|^2 dbeta
      (b) direct Haskind: X_j(beta) from the radiation potential on the hull and the
          incident wave alone, against the integrated diffraction + Froude-Krylov force
 4. Mesh sensitivity (finer hull mesh) and lid sensitivity (no lid) at the two
    disputed frequencies, head and following seas.
 5. Two-degree-of-freedom (heave, pitch) frequency-domain response at those
    frequencies with the head, following and Rankine-scaled excitations; this is
    the V1 RAO comparison recomputed with the corrected excitation.

Outputs: studies/_cache/derive/v9_excitation_check.txt / v9_results.json
Run: cd repo && .venv/Scripts/python.exe -m studies.derive_v9_excitation_check
No file under sim/ or hydro/ is modified; nothing is written outside _cache/derive.
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_cache", "derive")
TABLE = os.path.join(HERE, "_cache", "anyspeed", "bem_probe_lid_24_64.npz")
G = 9.81
RHO = 1025.0
# V2 Rankine head-seas first-order excitation per unit wave amplitude (v2_results.json,
# res/first/X; e^{+i omega t} convention, wave eta = Re[A e^{i(w t + k x)}] toward -x)
V2 = {3.0: dict(X1=830.3378559940427 - 1635.934996390377j,
                X3=2753.9668054487975 + 15116.665280410585j,
                X5=33033.47813760601 - 28114.62342665935j,
                A33=5280.79, B33=13579.76, A55=11306.31, B55=16376.41, A35=-941.0, B35=-3498.9),
      3.9: dict(X1=-1121.5645525443435 + 1625.3946220282091j,
                X3=-2167.4402026005355 - 5442.013509635664j,
                X5=11578.547381960563 + 13613.306609611778j,
                A33=4230.50, B33=13236.63, A55=8809.62, B55=21034.48, A35=-624.9, B35=-3555.5),
      4.7: dict(X1=-584.00832362765 - 1400.1973029670091j,
                X3=4516.208639120413 - 175.66572881877858j,
                X5=-12960.331731069184 - 4254.839405394314j,
                A33=4035.39, B33=13049.94, A55=7582.81, B55=20298.19, A35=-531.2, B35=-4149.9)}
C_BEM = dict(C33=87738.0, C55=197509.0, C35=-19658.0)       # B1.1
M_RB = np.diag([1400.0, 2552.0])                            # mass, pitch inertia (hull spec)
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:            # noqa: BLE001
        return float("nan")


# ------------------------------------------------------------------ step 1
def step1_table():
    z = np.load(TABLE)
    om, dirs = z["omega"], z["directions"]
    F = z["F_re"] + 1j * z["F_im"]
    FK = z["FK_re"] + 1j * z["FK_im"]
    i0 = int(np.argmin(np.abs(dirs - 0.0)))
    ipi = int(np.argmin(np.abs(dirs - np.pi)))
    say("== step 1: the old Capytaine table, both direction columns ==")
    say(f"  file {os.path.relpath(TABLE, HERE)}; directions stored {np.round(dirs, 4).tolist()}"
        f" (the probe asked for [pi, 3pi/4, pi/2, 0]; Capytaine's assemble_dataset sorts them)")
    say(f"  {'w':>5} | {'|X1| b=0':>9} {'|X1| b=pi':>9} | {'|X3| b=0':>9} {'|X3| b=pi':>9} {'|FK3|':>8} | "
        f"{'|X5| b=0':>9} {'|X5| b=pi':>9} {'|FK5|':>8} | Rankine head (V2): |X1| |X3| |X5|")
    out = {}
    for w in (3.04, 3.89, 4.74, 5.59):
        k = int(np.argmin(np.abs(om - w)))
        wr = {3.04: 3.0, 3.89: 3.9, 4.74: 4.7}.get(w)
        r = V2.get(wr, {})
        rk = (f"{abs(r['X1']):7.0f} {abs(r['X3']):7.0f} {abs(r['X5']):7.0f}  (at w={wr})" if r else "")
        say(f"  {om[k]:5.2f} | {abs(F[k, i0, 0]):9.0f} {abs(F[k, ipi, 0]):9.0f} | {abs(F[k, i0, 2]):9.0f} "
            f"{abs(F[k, ipi, 2]):9.0f} {abs(FK[k, ipi, 2]):8.0f} | {abs(F[k, i0, 4]):9.0f} {abs(F[k, ipi, 4]):9.0f} "
            f"{abs(FK[k, ipi, 4]):8.0f} | {rk}")
        out[str(om[k])] = dict(X_follow=[abs(F[k, i0, j]) for j in (0, 2, 4)],
                               X_head=[abs(F[k, ipi, j]) for j in (0, 2, 4)],
                               FK=[abs(FK[k, ipi, j]) for j in (0, 2, 4)],
                               X_head_complex=[[F[k, ipi, j].real, F[k, ipi, j].imag] for j in (0, 2, 4)])
    say("  V2 compared its head-seas values with the b=0 column (14867 / 10784 / 7284): "
        "those are the table's FOLLOWING-seas numbers; the b=pi column is 14408 / 5859 / 3944.")
    return out, dict(omega=om, dirs=dirs, F=F, FK=FK, A=z["A"], B=z["B"])


# ------------------------------------------------------------------ step 2
def step2_convention():
    import capytaine as cpt
    from capytaine.bem.airy_waves import airy_waves_potential
    say("\n== step 2: Capytaine's wave-direction convention, from its Airy-wave function ==")
    mesh = cpt.mesh_sphere(radius=0.5, center=(0, 0, -1.0), resolution=(4, 4))
    body = cpt.FloatingBody(mesh=mesh)
    body.add_translation_dof(name="Heave")
    w = 3.9
    k = w ** 2 / G
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    res = {}
    for beta in (0.0, np.pi):
        pb = cpt.DiffractionProblem(body=body, omega=w, wave_direction=beta, rho=RHO, g=G)
        phi = airy_waves_potential(pts, pb)
        dphase = np.angle(phi[1] / phi[0])
        res[beta] = dphase
        say(f"  beta = {beta:.3f}: phase(phi at x=1) - phase(phi at x=0) = {dphase:+.4f} rad "
            f"(k = {k:.4f}); time factor exp(-i w t) -> crest moves toward {'+x' if dphase > 0 else '-x'}")
    say("  The hull mesh has the bow at +x (x = station - LCG, stations from the transom). So beta = 0 is a "
        "wave travelling from the stern toward the bow = FOLLOWING seas; beta = pi = HEAD seas.")
    return {str(b): float(v) for b, v in res.items()}


# ------------------------------------------------------------------ step 3
def build_body(nx=64, nb=10, ns=6, lid=True):
    import capytaine as cpt
    from hydro.hulls import scarab_195
    from sim.planing_vessel import PlaningVessel
    from sim.test_vessel import Monochromatic
    from studies.anyspeed_bem_probe import hull_mesh
    h = scarab_195()
    p = PlaningVessel(h, Monochromatic(1.0, 0.0), dt=0.01)
    mesh = hull_mesh(p, nx=nx, nb=nb, ns=ns)
    mesh = mesh.rotated_y(p.Th_rest).translated_z(p.Z_rest)
    body = cpt.FloatingBody(mesh=mesh, center_of_mass=(0.0, 0.0, p.Z_rest),
                            lid_mesh=(mesh.generate_lid(z=-1e-3) if lid else None))
    body.add_all_rigid_body_dofs()
    body = body.immersed_part()
    return body, h


def solve(body, omegas, betas, keep=True):
    import capytaine as cpt
    solver = cpt.BEMSolver()
    probs = []
    for w in omegas:
        for dof in body.dofs:
            probs.append(cpt.RadiationProblem(body=body, omega=w, radiating_dof=dof, rho=RHO, g=G))
        for b in betas:
            probs.append(cpt.DiffractionProblem(body=body, omega=w, wave_direction=b, rho=RHO, g=G))
    t0 = time.time()
    results = solver.solve_all(probs, keep_details=keep, progress_bar=False, _check_wavelength=False)
    dt = time.time() - t0
    ds = cpt.assemble_dataset(results)
    return solver, results, ds, dt


DOFS = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]


def arr(ds, name, *dims):
    return ds[name].transpose(*dims).values


def haskind_direct(solver, results, body, ds):
    """X_j(beta) = -i w rho int (phi_0 n_j - phi_j^(v) dphi_0/dn) dS from the radiation potential
    (unit displacement in Capytaine 3: phi^(v) = phi^(d)/(-i w)); compare |X| with the dataset's
    excitation (FK + diffraction)."""
    from capytaine.bem.airy_waves import airy_waves_potential, airy_waves_velocity
    import capytaine as cpt
    rad = {(r.omega, r.radiating_dof): r for r in results if isinstance(r.problem, cpt.RadiationProblem)}
    dif = [r for r in results if isinstance(r.problem, cpt.DiffractionProblem)]
    n = body.mesh.faces_normals
    areas = body.mesh.faces_areas
    cen = body.mesh.faces_centers
    mask = body.hull_mask
    rows = []
    for r in dif:
        w, beta = r.omega, r.wave_direction
        phi0 = airy_waves_potential(cen, r.problem)
        dphi0dn = np.sum(airy_waves_velocity(cen, r.problem) * n, axis=1)
        row = dict(omega=float(w), beta=float(beta))
        for dof in ("Surge", "Heave", "Pitch"):
            rr = rad[(w, dof)]
            phid = np.asarray(rr.potential)
            if phid.shape[0] != mask.shape[0]:
                phid_h = phid
            else:
                phid_h = phid[mask]
            phiv = phid_h / (-1j * w)
            motion = body.dofs[dof]
            if hasattr(motion, "evaluate_motion"):
                motion = motion.evaluate_motion(body.mesh)
            nj = np.sum(np.asarray(motion) * n, axis=1)
            Xh = -1j * w * RHO * np.sum((phi0 * nj - phiv * dphi0dn) * areas)
            Xd = complex(ds["excitation_force"].sel(omega=w, wave_direction=beta, influenced_dof=dof).values)
            row[dof] = dict(haskind=abs(Xh), direct=abs(Xd), ratio=abs(Xh) / abs(Xd) if abs(Xd) > 0 else np.nan)
        rows.append(row)
    return rows


def haskind_newman(ds, betas):
    """B_jj vs omega^3/(4 pi rho g^3) * 2 * int_0^pi |X_j|^2 dbeta (y-symmetric hull), Simpson."""
    from scipy.integrate import simpson
    out = []
    for w in ds.omega.values:
        row = dict(omega=float(w))
        for dof in ("Surge", "Heave", "Pitch"):
            X = ds["excitation_force"].sel(omega=w, influenced_dof=dof).transpose("wave_direction").values
            I = 2.0 * simpson(np.abs(X) ** 2, x=betas)
            Bhn = w ** 3 / (4 * np.pi * RHO * G ** 3) * I
            Bjj = float(ds["radiation_damping"].sel(omega=w, radiating_dof=dof, influenced_dof=dof).values)
            row[dof] = dict(B=Bjj, B_haskind=Bhn, ratio=Bhn / Bjj if Bjj else np.nan,
                            X_head=abs(X[-1]), X_follow=abs(X[0]), X_rms=np.sqrt(I / (2 * np.pi)))
        out.append(row)
    return out


def step3_rerun(tbl):
    say("\n== step 3: Capytaine re-solve on the probe's own mesh (64 x (10+6), lid at z = -1 mm) ==")
    t0 = time.time()
    body, h = build_body(nx=64, nb=10, ns=6, lid=True)
    say(f"  immersed hull panels {body.mesh.nb_faces} (table: {int(np.load(TABLE)['n_panels'])}), lid panels "
        f"{body.lid_mesh.nb_faces}; volume {body.mesh.volume:.4f} m3; waterplane {body.mesh.waterplane_area:.3f} m2")
    f_irr = body.first_irregular_frequency_estimate(g=G)
    body_nolid, _ = build_body(nx=64, nb=10, ns=6, lid=False)
    f_irr0 = body_nolid.first_irregular_frequency_estimate(g=G)
    say(f"  Capytaine's first-irregular-frequency estimate (box formula): no lid {f_irr0:.2f} rad/s; with this lid "
        f"{f_irr if np.isfinite(f_irr) else float('inf'):.2f} rad/s (lid 1 mm below the surface -> treated as none)")
    omegas = [3.0, 3.04, 3.89, 3.9, 4.7, 4.74]
    betas = np.linspace(0.0, np.pi, 13)
    solver, results, ds, dt = solve(body, omegas, betas, keep=True)
    say(f"  {len(omegas)} frequencies x (6 radiation + {len(betas)} diffraction) = {len(results)} problems in "
        f"{dt:.1f} s; process RSS now {rss_mb():.0f} MB")
    A = arr(ds, "added_mass", "omega", "radiating_dof", "influenced_dof")
    B = arr(ds, "radiation_damping", "omega", "radiating_dof", "influenced_dof")
    F = arr(ds, "excitation_force", "omega", "wave_direction", "influenced_dof")
    FK = arr(ds, "Froude_Krylov_force", "omega", "wave_direction", "influenced_dof")
    dofs = list(ds.radiating_dof.values)
    iS, iH, iP = dofs.index("Surge"), dofs.index("Heave"), dofs.index("Pitch")
    om = ds.omega.values
    bet = ds.wave_direction.values
    j0, jpi = int(np.argmin(np.abs(bet))), int(np.argmin(np.abs(bet - np.pi)))

    say("\n  3a. reproduction of the stored table at its own frequencies (|X| per metre of wave amplitude):")
    say(f"  {'w':>5} | {'A33 new/tab':>13} {'B33 new/tab':>13} | {'|X3| head new/tab':>19} {'|X3| follow new/tab':>21} | "
        f"{'|X5| head new/tab':>19} | {'|X1| head new/tab':>19}")
    rep = {}
    for w in (3.04, 3.89, 4.74):
        k = int(np.argmin(np.abs(om - w)))
        kt = int(np.argmin(np.abs(tbl["omega"] - w)))
        t_i0 = int(np.argmin(np.abs(tbl["dirs"]))); t_ipi = int(np.argmin(np.abs(tbl["dirs"] - np.pi)))
        say(f"  {om[k]:5.2f} | {A[k, iH, iH]:6.0f}/{tbl['A'][kt, 2, 2]:6.0f} {B[k, iH, iH]:6.0f}/{tbl['B'][kt, 2, 2]:6.0f} | "
            f"{abs(F[k, jpi, iH]):8.0f}/{abs(tbl['F'][kt, t_ipi, 2]):8.0f}   {abs(F[k, j0, iH]):8.0f}/{abs(tbl['F'][kt, t_i0, 2]):8.0f}   | "
            f"{abs(F[k, jpi, iP]):8.0f}/{abs(tbl['F'][kt, t_ipi, 4]):8.0f}   | {abs(F[k, jpi, iS]):8.0f}/{abs(tbl['F'][kt, t_ipi, 0]):8.0f}")
        rep[str(w)] = dict(A33=[A[k, iH, iH], tbl['A'][kt, 2, 2]], X3_head=[abs(F[k, jpi, iH]), abs(tbl['F'][kt, t_ipi, 2])],
                           X3_follow=[abs(F[k, j0, iH]), abs(tbl['F'][kt, t_i0, 2])])
    say("  (equal numbers = the file's direction coordinate is attached to the right data; index 3 = beta pi = head seas)")

    say("\n  3b. head and following seas at the Rankine frequencies, against V2 (Rankine, head seas):")
    say(f"  {'w':>4} {'mode':>6} | {'Capytaine head':>14} {'Cap. follow':>12} | {'Rankine head':>12} | {'Rankine/Cap head':>16} "
        f"{'Cap follow/head':>15} | phase(X5/X3): Cap head, Rankine(conj)")
    cmp = {}
    for w in (3.0, 3.9, 4.7):
        k = int(np.argmin(np.abs(om - w)))
        r = V2[w]
        ph_c = np.degrees(np.angle(F[k, jpi, iP] / F[k, jpi, iH]))
        ph_r = np.degrees(np.angle(np.conj(r["X5"]) / np.conj(r["X3"])))
        row = {}
        for lab, i, key in (("surge", iS, "X1"), ("heave", iH, "X3"), ("pitch", iP, "X5")):
            ch, cf, rk = abs(F[k, jpi, i]), abs(F[k, j0, i]), abs(r[key])
            row[lab] = dict(cap_head=ch, cap_follow=cf, rankine_head=rk, ratio_rankine_cap=rk / ch, ratio_follow_head=cf / ch,
                            cap_head_complex=[F[k, jpi, i].real, F[k, jpi, i].imag])
            extra = f"| {ph_c:+7.1f} deg, {ph_r:+7.1f} deg" if lab == "heave" else ""
            say(f"  {w:4.1f} {lab:>6} | {ch:14.0f} {cf:12.0f} | {rk:12.0f} | {rk / ch:16.3f} {cf / ch:15.3f} {extra}")
        row["A33"] = dict(cap=A[k, iH, iH], rankine=r["A33"]); row["B33"] = dict(cap=B[k, iH, iH], rankine=r["B33"])
        row["A55"] = dict(cap=A[k, iP, iP], rankine=r["A55"]); row["B55"] = dict(cap=B[k, iP, iP], rankine=r["B55"])
        row["A35"] = dict(cap=A[k, iH, iP], rankine=r["A35"]); row["B35"] = dict(cap=B[k, iH, iP], rankine=r["B35"])
        say(f"       radiation: A33 {A[k, iH, iH]:.0f}/{r['A33']:.0f}  B33 {B[k, iH, iH]:.0f}/{r['B33']:.0f}  "
            f"A55 {A[k, iP, iP]:.0f}/{r['A55']:.0f}  B55 {B[k, iP, iP]:.0f}/{r['B55']:.0f}  "
            f"A35 {A[k, iH, iP]:.0f}/{r['A35']:.0f}  B35 {B[k, iH, iP]:.0f}/{r['B35']:.0f}  (Capytaine/Rankine)")
        cmp[str(w)] = row

    say("\n  3c. Haskind-Newman identity on the new solve (13 directions 0..pi, Simpson, y-symmetry):")
    say(f"  {'w':>5} | {'B11':>7} {'HN':>7} {'ratio':>6} | {'B33':>7} {'HN':>7} {'ratio':>6} | {'B55':>7} {'HN':>7} {'ratio':>6} | "
        f"heave |X| head / follow / rms over directions")
    hn = haskind_newman(ds, bet)
    for row in hn:
        s, hh, p = row["Surge"], row["Heave"], row["Pitch"]
        say(f"  {row['omega']:5.2f} | {s['B']:7.0f} {s['B_haskind']:7.0f} {s['ratio']:6.3f} | {hh['B']:7.0f} {hh['B_haskind']:7.0f} "
            f"{hh['ratio']:6.3f} | {p['B']:7.0f} {p['B_haskind']:7.0f} {p['ratio']:6.3f} | {hh['X_head']:.0f} / {hh['X_follow']:.0f} / {hh['X_rms']:.0f}")
    say("  (ratio = damping implied by the directional integral of the excitation / damping from the radiation solve; "
        "the single-direction value is not fixed by this identity -- only the directional mean square is)")

    say("\n  3d. direct Haskind relation (radiation potential + incident wave, no diffraction solve) vs the integrated force:")
    hd = haskind_direct(solver, results, body, ds)
    say(f"  {'w':>5} {'beta':>5} | {'surge Hask/direct':>18} {'heave Hask/direct':>18} {'pitch Hask/direct':>18}")
    hd_sel = []
    for row in hd:
        if abs(row["beta"]) < 1e-9 or abs(row["beta"] - np.pi) < 1e-9:
            say(f"  {row['omega']:5.2f} {row['beta']:5.2f} | {row['Surge']['haskind']:8.0f}/{row['Surge']['direct']:8.0f} "
                f"{row['Heave']['haskind']:8.0f}/{row['Heave']['direct']:8.0f} {row['Pitch']['haskind']:8.0f}/{row['Pitch']['direct']:8.0f}")
            hd_sel.append(row)
    say(f"  elapsed step 3: {time.time() - t0:.0f} s; RSS {rss_mb():.0f} MB")
    return dict(reproduction=rep, compare=cmp, haskind_newman=hn, haskind_direct=hd_sel, t=time.time() - t0,
                irregular_estimate=dict(no_lid=float(f_irr0), lid=float(f_irr) if np.isfinite(f_irr) else None),
                panels=int(body.mesh.nb_faces), lid_panels=int(body.lid_mesh.nb_faces)), (om, A, B, F, FK, dofs, bet)


# ------------------------------------------------------------------ step 4
def step4_sensitivity():
    say("\n== step 4: mesh and lid sensitivity at 3.9 and 4.7 rad/s (|X| per metre of wave amplitude) ==")
    out = {}
    cases = [("fine 96x(14+8), lid", dict(nx=96, nb=14, ns=8, lid=True)),
             ("probe 64x(10+6), no lid", dict(nx=64, nb=10, ns=6, lid=False))]
    say(f"  {'case':>26} {'panels':>7} | {'w':>4} | {'|X3| head':>9} {'|X3| follow':>11} | {'|X5| head':>9} {'|X5| follow':>11} | "
        f"{'|X1| head':>9} {'|X1| follow':>11} | A33 B33")
    for lab, kw in cases:
        t0 = time.time()
        body, _ = build_body(**kw)
        _, results, ds, dt = solve(body, [3.9, 4.7], [0.0, np.pi], keep=False)
        A = arr(ds, "added_mass", "omega", "radiating_dof", "influenced_dof")
        B = arr(ds, "radiation_damping", "omega", "radiating_dof", "influenced_dof")
        F = arr(ds, "excitation_force", "omega", "wave_direction", "influenced_dof")
        dofs = list(ds.radiating_dof.values)
        iS, iH, iP = dofs.index("Surge"), dofs.index("Heave"), dofs.index("Pitch")
        bet = ds.wave_direction.values
        j0, jpi = int(np.argmin(np.abs(bet))), int(np.argmin(np.abs(bet - np.pi)))
        npan = body.mesh.nb_faces + (body.lid_mesh.nb_faces if body.lid_mesh is not None else 0)
        out[lab] = {}
        for k, w in enumerate(ds.omega.values):
            say(f"  {lab:>26} {npan:7d} | {w:4.1f} | {abs(F[k, jpi, iH]):9.0f} {abs(F[k, j0, iH]):11.0f} | "
                f"{abs(F[k, jpi, iP]):9.0f} {abs(F[k, j0, iP]):11.0f} | {abs(F[k, jpi, iS]):9.0f} {abs(F[k, j0, iS]):11.0f} | "
                f"{A[k, iH, iH]:.0f} {B[k, iH, iH]:.0f}")
            out[lab][str(float(w))] = dict(X3_head=abs(F[k, jpi, iH]), X3_follow=abs(F[k, j0, iH]), X5_head=abs(F[k, jpi, iP]),
                                           X5_follow=abs(F[k, j0, iP]), X1_head=abs(F[k, jpi, iS]), X1_follow=abs(F[k, j0, iS]),
                                           A33=A[k, iH, iH], B33=B[k, iH, iH], A55=A[k, iP, iP], B55=B[k, iP, iP])
        say(f"    ({len(results)} problems, {dt:.1f} s solve, {time.time() - t0:.0f} s total, RSS {rss_mb():.0f} MB)")
        del body, results, ds
    return out


# ------------------------------------------------------------------ step 5
def step5_rao(base):
    om, A, B, F, FK, dofs, bet = base
    iH, iP = dofs.index("Heave"), dofs.index("Pitch")
    idx = [iH, iP]
    j0, jpi = int(np.argmin(np.abs(bet))), int(np.argmin(np.abs(bet - np.pi)))
    C = np.array([[C_BEM["C33"], C_BEM["C35"]], [C_BEM["C35"], C_BEM["C55"]]])
    say("\n== step 5: heave-pitch frequency-domain response (V1's fd_rao form, BEM C of B1.1) at the exact frequencies ==")
    say("  excitation variants: Capytaine head (what V1's head column used); Capytaine following (= B1.3's BEM column); "
        "Capytaine head with |X3|, |X5| scaled to the Rankine magnitudes (phases kept)")
    say(f"  {'w':>4} | {'heave/a head':>12} {'follow':>8} {'Rankine-scaled':>14} | {'pitch/(ka) head':>15} {'follow':>8} {'Rankine-scaled':>14} | "
        f"V1 head (table w): heave, pitch")
    v1 = {3.0: (0.288, 0.523), 3.9: (0.074, 0.128), 4.7: (0.027, 0.058)}
    out = {}
    for w in (3.0, 3.9, 4.7):
        k = int(np.argmin(np.abs(om - w)))
        wk = om[k]
        Z = -wk ** 2 * (M_RB + A[k][np.ix_(idx, idx)]) - 1j * wk * B[k][np.ix_(idx, idx)] + C
        kw = wk ** 2 / G
        Fh = F[k, jpi, idx]
        Ff = F[k, j0, idx]
        r = V2[w]
        Fs = Fh * np.array([abs(r["X3"]) / abs(Fh[0]), abs(r["X5"]) / abs(Fh[1])])
        res = {}
        for lab, Fe in (("head", Fh), ("follow", Ff), ("rankine_scaled", Fs)):
            x = np.linalg.solve(Z, Fe)
            res[lab] = dict(heave=abs(x[0]), pitch=abs(x[1]) / kw)
        say(f"  {wk:4.1f} | {res['head']['heave']:12.3f} {res['follow']['heave']:8.3f} {res['rankine_scaled']['heave']:14.3f} | "
            f"{res['head']['pitch']:15.3f} {res['follow']['pitch']:8.3f} {res['rankine_scaled']['pitch']:14.3f} | "
            f"{v1[w][0]:.3f}, {v1[w][1]:.3f}")
        out[str(float(wk))] = res
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time()
    say(f"V9 excitation check, scarab195 at rest, Capytaine {__import__('capytaine').__version__}; RSS at start {rss_mb():.0f} MB")
    res = {}
    res["table"], tbl = step1_table()
    res["convention"] = step2_convention()
    res["rerun"], base = step3_rerun(tbl)
    res["sensitivity"] = step4_sensitivity()
    res["rao"] = step5_rao(base)
    res["wall_s"] = time.time() - T0
    say(f"\ntotal wall time {res['wall_s']:.0f} s; RSS at end {rss_mb():.0f} MB")

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, complex):
            return [o.real, o.imag]
        return o
    with open(os.path.join(OUT, "v9_results.json"), "w", encoding="utf-8") as f:
        json.dump(conv(res), f, indent=1)
    with open(os.path.join(OUT, "v9_excitation_check.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(HERE))
    main()
