#!/usr/bin/env python3
"""
Export a run for the real-time WebGL viewer.

The browser does not replay a recorded surface -- it RECOMPUTES it, the way a
game engine would. Only the spectral components go across:

    eta(x,y,t) = sum_j a_j cos( k_j (x cos th_j + y sin th_j) - w_j t + phi_j )

32 components is a handful of numbers, and the sum runs in a vertex shader at
60 fps over a 200x200 grid. That means the water in the viewer is the SAME
field the physics used, evaluated independently -- not a movie of it. Scrubbing
the timeline re-evaluates it at the new instant, so the surface is always
consistent with the vessel pose being shown.

What is recorded is only what the browser cannot recompute: the vessel
trajectory, which came out of the nonlinear solve.
"""
import json
import numpy as np

from hydro import bem, kelvin
from hydro.freesurface import FreeSurface
from hydro.geometry import wigley_mesh
from sim.env import Episode
from sim.vessel import NonlinearVessel, SIGN_PITCH
from sim.test_vessel import Monochromatic
from control.reduced import ReducedModel

DB = "hydro_wigley_10m.npz"
G = 9.81


def hull_geometry(L=10.0, B=2.5, T=0.8, nx=44, nz=12, freeboard=0.55):
    """Wetted surface plus a wall-sided topside and a deck, as triangles."""
    mesh = wigley_mesh(L, B, T, nx, nz)
    v = np.asarray(mesh.vertices, float)
    f = np.asarray(mesh.faces, int)
    verts = list(v)
    tris = []

    def add_quad(a, b, c, d):
        i = len(verts)
        verts.extend([a, b, c, d])
        tris.extend([[i, i + 1, i + 2], [i, i + 2, i + 3]])

    for q in f:
        tris.extend([[q[0], q[1], q[2]], [q[0], q[2], q[3]]])

    up = np.array([0.0, 0.0, freeboard])
    for q in f:
        ring = [i for i in q if abs(v[i, 2]) < 1e-9]
        if len(ring) == 2:
            a, b = v[ring[0]], v[ring[1]]
            add_quad(a, b, b + up, a + up)

    wl = v[np.abs(v[:, 2]) < 1e-9]
    xs = np.unique(np.round(wl[:, 0], 6))
    hb = np.array([np.max(np.abs(wl[np.abs(wl[:, 0] - xi) < 1e-6, 1]))
                   for xi in xs])
    for x0, x1, h0, h1 in zip(xs[:-1], xs[1:], hb[:-1], hb[1:]):
        add_quad(np.array([x0, -h0, freeboard]), np.array([x1, -h1, freeboard]),
                 np.array([x1, h1, freeboard]), np.array([x0, h0, freeboard]))

    V = np.array(verts, float)
    return V, np.array(tris, int)


N_FREQ, N_DIR = 16, 4


def _one_run(db, red, seed, t_end, t_preview, hz, hs, tp):
    # The MPC now steers. Until this revision every recorded run used a
    # classical autopilot for heading, because MPC steering diverged to 476 deg
    # of heading RMS -- the reduced model had no sway state, so the controller
    # could not see the drift its own rudder was causing. With sway added it
    # holds 0.9 deg and 0.3 m of cross-track, slightly better on track than the
    # autopilot it replaces.
    #
    # Note what this does and does not license. Regulation needs only the FIRST
    # horizon step to be accurate, and there the model correlates 0.96 on
    # heading and 0.99 on cross-track. Anticipation would need the whole
    # horizon, and the steering channels decay by 1-2.5 s because the reduced
    # model carries no wave-induced yaw moment (studies/model_horizon.py). So
    # this is a steering controller, not a steering-with-preview controller.
    ep = Episode(db, red, seed=seed, hs=hs, tp=tp, t_preview=t_preview,
                 u_ref=4.5, n_freq=N_FREQ, n_dir=N_DIR,
                 use_rudder=True, autopilot=False)

    stride = max(int(round(1.0 / (hz * ep.dt))), 1)
    s = ep.plant.initial_state(ep.u_ref * 0.8)
    t = 0.0
    frames, k, slams = [], 0, 0
    n_ctrl = int(t_end / ep.dt_ctrl)
    for _ in range(n_ctrl):
        sr = ep.reduced.from_plant_state(s)
        cmd = ep.ctrl(sr, t)
        thr, rud = ep.ctrl.to_actuator(cmd)
        for _ in range(ep.sub):
            s = ep.plant.step(s, t, thr, rud, ep.dt)
            t += ep.dt
            if k % stride == 0:
                new = ep.plant.slam_count > slams
                slams = ep.plant.slam_count
                frames.append([round(float(x), 4) for x in
                               (t, s[0], s[1], s[2], s[3], s[4], s[5],
                                s[6], thr / 12000.0,
                                ep.plant.last_bow_acc / G)]
                              + [1 if new else 0,
                                 round(float(rud), 4)])
            k += 1

    # Packed for the shader: one vec4 per component holding (kx, ky, w, phi),
    # plus one float array of amplitudes. A GLSL `float arr[N]` typically
    # occupies a whole vec4 slot PER ELEMENT, so six separate float arrays for
    # 120 components needed ~298 slots -- past the 128 a WebGL1 vertex stage is
    # guaranteed, and past WebGL2's 256. The shader then fails to link and
    # nothing renders at all. Packed, 64 components need exactly 128 slots.
    sea = ep.sea
    kx = sea.k * np.cos(sea.th)
    ky = sea.k * np.sin(sea.th)
    packed = []
    for i in range(sea.a.size):
        packed += [round(float(kx[i]), 8), round(float(ky[i]), 8),
                   round(float(sea.w[i]), 6), round(float(sea.phi[i]), 6)]
    return dict(
        seed=seed, slams=int(ep.plant.slam_count),
        p=packed,
        a=[round(float(x), 6) for x in sea.a],
        frames=frames)


def run_and_export(t_end=140.0, out="viewer_data.json", t_preview=8.0,
                   hz=10.0, seeds=(0, 1, 2, 3), hs=3.25, tp=9.7):
    db = bem.load(DB)
    plant_for_drive = NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L)
    red = ReducedModel.identify(plant_for_drive, db)
    runs = []
    for sd in seeds:
        r = _one_run(db, red, sd, t_end, t_preview, hz, hs, tp)
        runs.append(r)
        print(f"  seed {sd}: {len(r['frames'])} frames, {r['slams']} slams")

    V, F = hull_geometry(L=db.L, B=float(db.attrs.get("B", 2.5)),
                         T=float(db.attrs.get("T", 0.8)))

    # The steady Kelvin system: the bow wave and the V-shaped wake. It is
    # absent from everything else here, because the BEM behind all of it is
    # solved at zero forward speed -- and at 0.4-0.6 m it is an order of
    # magnitude larger than the unsteady disturbance that WAS being drawn.
    # Baked at the mean speed the runs actually hold; the shader stretches it
    # for the instantaneous speed (see kelvin.wake_texture).
    u_mean = float(np.mean([f[7] for r in runs for f in r["frames"]]))
    print(f"  baking Kelvin wake at the run mean speed U0 = {u_mean:.2f} m/s "
          f"(Fn = {u_mean/np.sqrt(G*db.L):.2f}) ...")
    wake = kelvin.wake_texture(u_mean, db.L, float(db.attrs.get("B", 2.5)),
                               float(db.attrs.get("T", 0.8)))
    print(f"    {wake['w']}x{wake['h']} texels, peak {wake['amp']:.3f} m, "
          f"transverse wavelength {wake['lam_transverse']:.1f} m")

    # The unsteady disturbance table (diffraction + radiation), body-fixed and
    # per frequency. It lives here rather than being patched into the JSON by
    # hand -- a re-export once dropped it silently and the viewer failed on
    # `DATA.disturb.omega`, which is a poor way to find out.
    fs = FreeSurface()
    disturb = dict(
        x=[round(float(v), 4) for v in fs.x],
        y=[round(float(v), 4) for v in fs.y],
        omega=[round(float(v), 5) for v in fs.omega],
        re=[round(float(v), 6) for v in fs.eta.real.ravel()],
        im=[round(float(v), 6) for v in fs.eta.imag.ravel()])
    print(f"  disturbance table {len(fs.omega)}x{len(fs.x)}x{len(fs.y)}, "
          f"peak {np.abs(fs.eta).max():.4f} m per unit wave amplitude")

    # Parameters for the browser's DRIVE mode. The viewer normally replays a
    # trajectory, because the 110-state plant cannot run in a browser. To let a
    # person steer, the browser has to integrate the vessel itself -- so it gets
    # the REDUCED model, the same 10-state caricature the MPC plans with. That
    # model was measured against the full plant in `studies.model_horizon`:
    # correlation 0.64-0.90 across a 12 s horizon, amplitude 0.78-0.84. Good
    # enough to fly, and labelled as such in the interface rather than passed
    # off as the real solve.
    drive = {k: (float(v) if not isinstance(v, str) else v)
             for k, v in red.p.items()}
    drive.update(
        t_max=float(plant_for_drive.prop.t_max),
        prop_tau=float(plant_for_drive.prop.tau),
        prop_rate=float(plant_for_drive.prop.rate_max),
        rud_max=float(plant_for_drive.rudder.max),
        rud_rate=float(plant_for_drive.rudder.rate),
        rud_tau=float(plant_for_drive.rudder.tau),
        freeboard=0.55, g=G)

    data = dict(
        drive=drive,
        disturb=disturb,
        meta=dict(L=db.L, B=float(db.attrs.get("B", 2.5)),
                  T=float(db.attrs.get("T", 0.8)),
                  hs=hs, tp=tp, hz=hz, dt=1.0 / hz,
                  n_freq=N_FREQ, n_dir=N_DIR, n_comp=N_FREQ*N_DIR,
                  t_preview=t_preview, sign_pitch=SIGN_PITCH, g=G,
                  lam_p=float(1.56 * tp ** 2)),
        wake=wake,
        hull=dict(v=[[round(float(c), 4) for c in p] for p in V],
                  f=[[int(c) for c in tri] for tri in F]),
        runs=runs)
    json.dump(data, open(out, "w"), separators=(",", ":"))
    import os
    print(f"  {len(runs)} runs, {N_FREQ}x{N_DIR}={N_FREQ*N_DIR} components, "
          f"{len(F)} triangles, Kelvin wake baked at {u_mean:.2f} m/s")
    print(f"  saved -> {out}  ({os.path.getsize(out)/1024:.0f} KB)")
    return data


if __name__ == "__main__":
    run_and_export()
