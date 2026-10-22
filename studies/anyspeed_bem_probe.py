#!/usr/bin/env python3
"""
Probe (design phase, read-only on tracked code): can the repo's BEM chain
(hydro/bem.py on Capytaine) give zero-speed added mass / damping / excitation
for scarab195 from the planing plant's own geometry, how long does it take,
and how do the numbers compare with what sim/planing_vessel.py uses at rest?

Geometry: exactly the plant's prismatic hull (PlaningVessel stations: V
bottom of deadrise beta from the keel to the chine half-beam yc(x), vertical
side to the hull depth, flat transom at x = 0, chine curving in over the bow
`taper`, straight keel), placed at the plant's own rest attitude
(Z_rest, Th_rest), clipped at z = 0 by Capytaine.

Outputs: studies/_cache/anyspeed/bem_probe.txt (+ .npz of A, B, F).

Run: python -m studies.anyspeed_bem_probe [n_omega]
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_cache", "anyspeed")
G = 9.81


def hull_mesh(p, nx=48, nb=8, ns=6):
    """Closed (below the deck) surface of the plant's hull in the BODY frame
    (x forward from the CG, z up from the CG): both sides + transom."""
    import capytaine as cpt
    from hydro.hull_geometry import stations_of
    h = p.hull
    g = stations_of(h)                  # the plant's geometry, any x
    xs = np.linspace(0.0, h.L, nx + 1)                  # from the transom
    # the chine with a 1 mm floor at the stem (panels stay non-degenerate)
    # in place of the plant's 0.1 mm
    yc = g.chine(xs) - g.eps + 1e-3
    zk = -h.vcg + g.keel(xs)[0]
    tb = p.tanb
    # girth parameter: bottom keel->chine (nb), side chine->deck (ns)
    Y = np.zeros((nx + 1, nb + ns + 1))
    Z = np.zeros_like(Y)
    for i in range(nx + 1):
        yb = np.linspace(0.0, yc[i], nb + 1)
        zb = zk[i] + yb * tb
        zc = zk[i] + yc[i] * tb
        zs = np.linspace(zc, zk[i] + h.depth, ns + 1)[1:]
        Y[i] = np.r_[yb, np.full(ns, yc[i])]
        Z[i] = np.r_[zb, zs]
    X = np.repeat((xs - h.lcg)[:, None], nb + ns + 1, axis=1)
    verts, groups = [], []

    def quads(idx):
        return np.column_stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(),
                                idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()])

    for sign in (+1, -1):
        base = sum(v.shape[0] for v in verts)
        pts = np.stack([X.ravel(), sign * Y.ravel(), Z.ravel()], 1)
        verts.append(pts)
        idx = base + np.arange(pts.shape[0]).reshape(X.shape)
        groups.append(("side", sign, quads(idx)))
    # transom (x = 0 station): a proper grid across the beam at every girth
    # level (no long thin strips)
    nt = 10
    u = np.linspace(-1.0, 1.0, nt + 1)
    n0 = nb + ns + 1
    base = sum(v.shape[0] for v in verts)
    Yt = u[:, None] * Y[0][None, :]
    Zt = np.repeat(Z[0][None, :], nt + 1, axis=0)
    pts = np.stack([np.full(Yt.size, X[0, 0]), Yt.ravel(), Zt.ravel()], 1)
    verts.append(pts)
    idx = base + np.arange(pts.shape[0]).reshape(Yt.shape)
    groups.append(("transom", 0, quads(idx)))
    V = np.vstack(verts)
    F = []
    for kind, sign, q in groups:
        p0, p1, p2 = V[q[:, 0]], V[q[:, 1]], V[q[:, 2]]
        p3 = V[q[:, 3]]
        n = np.cross(p2 - p0, p3 - p1)              # quad normal (diagonals)
        area = np.linalg.norm(n, axis=1)
        keep = area > 1e-10
        q, n = q[keep], n[keep]
        # outward: transom faces aft (-x); a side faces away from the
        # centreplane (sign * y) or down on the bottom
        if kind == "transom":
            want = -np.sign(n[:, 0].mean())
        else:
            want = np.sign((sign * n[:, 1]).mean())
        F.append(q if want > 0 else q[:, ::-1])
    F = np.vstack(F)
    return cpt.Mesh(vertices=V, faces=F, name="scarab195_strips")


def main(n_omega=8, lid=False, nx=48, nb=8, ns=6):
    import capytaine as cpt
    from hydro.hulls import scarab_195
    from sim.planing_vessel import PlaningVessel
    from sim.test_vessel import Monochromatic
    from hydro import bem

    os.makedirs(OUT, exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    h = scarab_195()
    p = PlaningVessel(h, Monochromatic(1.0, 0.0), dt=0.01)
    th = p.Th_rest                    # plant pitch at rest (SP = -1: bow up < 0)
    say(f"scarab195 at rest (plant): CG height {p.Z_rest:+.4f} m, pitch state "
        f"{th:+.5f} rad = trim {-np.degrees(th):+.2f} deg (bow up +), "
        f"transom draught {p.T:.3f} m, GM {p.GM:.3f} m")
    mesh = hull_mesh(p, nx=nx, nb=nb, ns=ns)
    # body -> earth: rotate about +y by th (z_e = Z + z_b + SP x th, SP=-1),
    # then lift the CG to Z_rest
    mesh = mesh.rotated_y(th).translated_z(p.Z_rest)
    cog = (0.0, 0.0, p.Z_rest)
    body = cpt.FloatingBody(
        mesh=mesh, center_of_mass=cog,
        lid_mesh=(mesh.generate_lid(z=-1e-3) if lid else None))
    body.add_all_rigid_body_dofs()
    body = body.immersed_part()
    say(f"mesh {nx}x({nb}+{ns}) per side; lid {lid}"
        + (f" ({body.lid_mesh.nb_faces} lid panels)"
           if getattr(body, "lid_mesh", None) is not None else ""))
    vol = float(body.mesh.volume)
    cob = np.asarray(body.mesh.center_of_buoyancy, float)
    awp = float(body.mesh.waterplane_area)
    say(f"immersed mesh: {body.mesh.nb_faces} panels; volume {vol:.4f} m3 "
        f"(plant m/rho {h.mass / h.rho:.4f}, {100 * (vol * h.rho / h.mass - 1):+.2f}%)"
        f"; centre of buoyancy x {cob[0]:+.4f} m from the CG (0 = level "
        f"equilibrium); waterplane {awp:.3f} m2")
    # plant's at-rest strip values for comparison
    d0 = -(p.Z_rest + p.zk + (-1.0) * p.x * th)
    A, c, ma, dma, girth = p._section(d0)
    A33_plant = float(ma.sum() * p.dx)
    A55_plant = float((ma * p.x * p.x).sum() * p.dx)
    half = np.minimum(np.maximum(d0, 0) / p.tanb, p.yc)
    say(f"plant at rest: heave added mass sum(k_a pi/2 rho c^2) dx = "
        f"{A33_plant:.0f} kg with c = min(pi/2 d/tan(beta), y_chine) -- "
        f"Wagner's impact pile-up applied to a static section; geometric "
        f"waterline half-beam max {half.max():.3f} m vs c max {c.max():.3f} m;"
        f" pitch added inertia {A55_plant:.0f} kg m2")
    say(f"plant lateral inertia (fixed at the 25 kn design attitude, used at "
        f"every speed): sway +{p.Mtot[1, 1] - p.m:.0f} kg, roll "
        f"+{p.Mtot[3, 3] - p.Ix:.0f} kg m2 (0.2 Ix), yaw +{p.Mtot[5, 5] - p.Iz:.0f}"
        f" kg m2, surge +{p.Mtot[0, 0] - p.m:.0f} kg (0.05 m)")
    C33 = h.rho * G * awp
    say(f"heave stiffness rho g A_wp = {C33:.0f} N/m")

    omegas = np.linspace(0.5, 20.0, n_omega)
    dirs = np.array([np.pi, 0.75 * np.pi, 0.5 * np.pi, 0.0])
    t0 = time.time()
    ds = bem.compute_database(body, omegas, directions=dirs, rho=h.rho)
    t_bem = time.time() - t0
    db = bem.to_db(ds)
    n_prob = len(omegas) * (6 + len(dirs))
    say(f"\nBEM: {len(omegas)} frequencies x (6 radiation + {len(dirs)} "
        f"diffraction) = {n_prob} problems in {t_bem:.1f} s "
        f"({t_bem / len(omegas):.2f} s per frequency); a 60 x 13 database "
        f"would take ~{t_bem / n_prob * 60 * 19 / 60:.1f} min per attitude")
    i = {d: k for k, d in enumerate(db.dofs)}
    say(f"\n  {'w rad/s':>8}{'A33 kg':>9}{'B33 Ns/m':>10}{'zeta33':>8}"
        f"{'A55':>8}{'B55':>9}{'A22':>8}{'B22':>8}{'A44':>7}{'B44':>7}"
        f"{'A66':>8}{'B66':>8}{'A11':>7}{'B11':>7}")
    for k, w in enumerate(db.omega):
        a33, b33 = db.A[k, i['Heave'], i['Heave']], db.B[k, i['Heave'], i['Heave']]
        z = b33 / (2 * np.sqrt((h.mass + a33) * C33))
        say(f"  {w:>8.2f}{a33:>9.0f}{b33:>10.0f}{z:>8.3f}"
            f"{db.A[k, i['Pitch'], i['Pitch']]:>8.0f}"
            f"{db.B[k, i['Pitch'], i['Pitch']]:>9.0f}"
            f"{db.A[k, i['Sway'], i['Sway']]:>8.0f}"
            f"{db.B[k, i['Sway'], i['Sway']]:>8.0f}"
            f"{db.A[k, i['Roll'], i['Roll']]:>7.0f}"
            f"{db.B[k, i['Roll'], i['Roll']]:>7.0f}"
            f"{db.A[k, i['Yaw'], i['Yaw']]:>8.0f}"
            f"{db.B[k, i['Yaw'], i['Yaw']]:>8.0f}"
            f"{db.A[k, i['Surge'], i['Surge']]:>7.0f}"
            f"{db.B[k, i['Surge'], i['Surge']]:>7.0f}")
    # heave natural frequency with BEM added mass, and the damping ratio there
    a33 = db.A[:, i['Heave'], i['Heave']]
    wn = np.sqrt(C33 / (h.mass + a33))
    j = int(np.argmin(np.abs(wn - db.omega)))
    say(f"\nheave natural frequency ~{wn[j]:.2f} rad/s (BEM added mass); "
        f"radiation damping ratio there ~"
        f"{db.B[j, i['Heave'], i['Heave']] / (2 * np.sqrt((h.mass + a33[j]) * C33)):.3f}")
    # surge Froude-Krylov (head seas) vs -W x slope (the textbook long-wave
    # value the surge check used as reference)
    kk = db.omega ** 2 / G
    fk = np.abs(db.F_fk[:, 0, i['Surge']])
    say("\nsurge Froude-Krylov |F1|/(rho g V k) in head seas (1 = the long-wave "
        "limit W x slope):")
    say("  " + "  ".join(f"w{w:.1f}:{f / (h.rho * G * vol * k):.2f}"
                         for w, f, k in zip(db.omega, fk, kk)))
    np.savez_compressed(os.path.join(OUT, f"bem_probe{'_lid' if lid else ''}_{n_omega}_{nx}.npz"), omega=db.omega,
                        directions=db.directions, A=db.A, B=db.B,
                        F_re=db.F_exc.real, F_im=db.F_exc.imag,
                        FK_re=db.F_fk.real, FK_im=db.F_fk.imag,
                        t_bem=t_bem, n_panels=body.mesh.nb_faces)
    with open(os.path.join(OUT, f"bem_probe{'_lid' if lid else ''}_{n_omega}_{nx}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 8, lid=len(a) > 1 and a[1] == "lid",
         nx=int(a[2]) if len(a) > 2 else 48,
         nb=int(a[3]) if len(a) > 3 else 8, ns=int(a[4]) if len(a) > 4 else 6)
