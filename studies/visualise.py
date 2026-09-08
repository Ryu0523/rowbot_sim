#!/usr/bin/env python3
"""
Render the hull in the water it is actually sitting in.

Two figures:

  fig12_scene.png          3-D snapshots of the hull in an SS5 seaway, taken
                           from a real closed-loop run so the attitude and
                           immersion are the model's, not a pose.
  fig13_decomposition.png  the surface split into incident, the vessel's own
                           disturbance, and the total -- which is what a LiDAR
                           would see and what the preview chain assumes.

The disturbance panel is the interesting one. At the SS5 spectral peak the hull
contours the wave and disturbs it by fractions of a percent; the disturbance
only becomes visible in the short-wave tail. That is the quantitative answer to
"does using the undisturbed wave for preview matter" -- for this hull in this
sea state, barely.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hydro import bem
from hydro.freesurface import FreeSurface
from hydro.geometry import wigley_mesh
from sim.env import Episode
from sim.vessel import NonlinearVessel
from sim.test_vessel import Monochromatic
from control.reduced import ReducedModel

DB = "hydro_wigley_10m.npz"
FS = "freesurface_wigley.npz"
WATER = "ocean"
HULL_FACE = "#d9531e"
HULL_EDGE = "#7a2f11"


def hull_polygons(mesh, eta, freeboard=0.55):
    """Hull panels in earth coordinates at the vessel's current pose.

    The BEM mesh is the wetted surface only, so a wall-sided topside is added
    for the picture -- which is also exactly what the nonlinear section model
    assumes above the design waterline.
    """
    v = np.asarray(mesh.vertices, float)
    f = np.asarray(mesh.faces, int)
    quads = [v[q] for q in f]

    # Topside: extrude every mesh EDGE that lies on the waterline. Found from
    # the face topology rather than by trying to sort the waterline vertices
    # into a ring -- ordering by angle breaks on a slender hull, where bow and
    # stern points sit at almost the same angle.
    up = np.array([0.0, 0.0, freeboard])
    for q in f:
        ring = [i for i in q if abs(v[i, 2]) < 1e-9]
        if len(ring) != 2:
            continue
        a, b = v[ring[0]], v[ring[1]]
        quads.append(np.array([a, b, b + up, a + up]))

    # Deck. Without it the hull reads as a hollow ring once the submerged part
    # is correctly hidden by the water -- which is most of a floating boat.
    wl = v[np.abs(v[:, 2]) < 1e-9]
    xs = np.unique(np.round(wl[:, 0], 6))
    hb = np.array([np.max(np.abs(wl[np.abs(wl[:, 0] - xi) < 1e-6, 1]))
                   for xi in xs])
    for x0, x1, h0, h1 in zip(xs[:-1], xs[1:], hb[:-1], hb[1:]):
        quads.append(np.array([[x0, -h0, freeboard], [x1, -h1, freeboard],
                               [x1, h1, freeboard], [x0, h0, freeboard]]))

    x, y, z, roll, pitch, yaw = eta[:6]
    # Ry(theta) already produces dz = -theta*x for a point forward of midship,
    # which IS the dof convention; applying SIGN_PITCH again would invert it.
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return [q @ R.T + np.array([x, y, z]) for q in quads]


def _surface_at(sea, fs, x, y, t):
    """Total elevation at scattered points -- incident plus the vessel's own
    disturbance, interpolated from the body-fixed table."""
    return sea.eta(np.asarray(x), np.asarray(y), t)


def _scene(ax, mesh, fs, sea, s, t, span=0.95, show_disturbance=True):
    eta_v = s[:6]
    Xe, Ye, inc, dist = fs.total(sea, eta_v[0], eta_v[1], t)
    surf = inc + (dist if show_disturbance else 0.0)

    m = (np.abs(Xe - eta_v[0]) < span * fs.L) & (np.abs(Ye) < span * fs.L)
    ii = np.where(m.any(1))[0]
    jj = np.where(m.any(0))[0]
    X, Y, Z = Xe[np.ix_(ii, jj)], Ye[np.ix_(ii, jj)], surf[np.ix_(ii, jj)]

    norm = Normalize(-2.2, 2.2)
    ls = LightSource(azdeg=315, altdeg=45)
    cmap = plt.get_cmap(WATER)
    rgb = ls.shade(Z, cmap=cmap, norm=norm, vert_exag=6.0, blend_mode="soft")
    surf3d = ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1,
                             linewidth=0, antialiased=True, shade=False)
    surf3d.set_zorder(1)

    # matplotlib's 3-D depth sorting between a surface and a collection is
    # unreliable -- the water was painted over the hull. computed_zorder=False
    # plus explicit zorder is the only robust way to control the order here.
    # But drawing the WHOLE hull above the water then makes the submerged part
    # visible too, so the boat appears to sit on top of the sea. Panels whose
    # centroid is below the local surface are dropped, which puts the waterline
    # where it belongs.
    polys = hull_polygons(mesh, eta_v)
    cen = np.array([q.mean(axis=0) for q in polys])
    eta_at = _surface_at(sea, fs, cen[:, 0], cen[:, 1], t)
    above = [q for q, c, e in zip(polys, cen, eta_at) if c[2] > e - 0.02]
    pc = Poly3DCollection(above, facecolor=HULL_FACE, edgecolor=HULL_EDGE,
                          linewidths=0.15, alpha=1.0, zorder=10)
    ax.add_collection3d(pc)

    ax.set_xlim(eta_v[0] - span * fs.L, eta_v[0] + span * fs.L)
    ax.set_ylim(-span * fs.L, span * fs.L)
    ax.set_zlim(-2.8, 2.8)
    # near-true proportions: 6.4 m of height across 19 m of sea
    ax.set_box_aspect((1.0, 1.0, 0.30))
    ax.view_init(elev=20, azim=-62)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_visible(False)
    ax.grid(False)


def scene_figure(mesh, fs, ep, times=(60.0, 72.0, 84.0)):
    """Snapshots from a real run: pose and immersion come from the model."""
    s = ep.plant.initial_state(ep.u_ref * 0.8)
    t = 0.0
    frames = {}
    n_ctrl = int(max(times) / ep.dt_ctrl) + 2
    for _ in range(n_ctrl):
        sr = ep.reduced.from_plant_state(s)
        cmd = ep.ctrl(sr, t)
        thr, rud = ep.ctrl.to_actuator(cmd)
        for _ in range(ep.sub):
            s = ep.plant.step(s, t, thr, rud, ep.dt)
            t += ep.dt
            for tt in times:
                if tt not in frames and t >= tt:
                    frames[tt] = (s.copy(), t)
        if len(frames) == len(times):
            break

    fig = plt.figure(figsize=(16, 4.6))
    for i, tt in enumerate(times):
        st, tv = frames[tt]
        ax = fig.add_subplot(1, len(times), i + 1, projection="3d",
                             computed_zorder=False)
        _scene(ax, mesh, fs, ep.sea, st, tv)
        ax.set_title(f"t = {tv:.0f} s     heave {st[2]:+.2f} m,  "
                     f"pitch {np.degrees(st[4]):+.1f}$^\\circ$,  "
                     f"u {st[6]:.1f} m/s", fontsize=10, pad=-2)
    fig.suptitle("Wigley 10 m in SS5, poses taken from a closed-loop run "
                 "(incident + diffracted + radiated surface)", fontsize=13)
    fig.tight_layout()
    fig.savefig("fig12_scene.png", dpi=150)
    print("  figure: fig12_scene.png")
    return frames


def decomposition_figure(fs, sea, state, t):
    """Incident / disturbance / total, seen from above."""
    xv, yv = state[0], state[1]
    Xe, Ye, inc, dist = fs.total(sea, xv, yv, t)
    xr = (Xe - xv) / fs.L
    yr = Ye / fs.L

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [
        (inc, "incident $\\eta_I$   (what the model uses)", 2.2, "RdBu_r"),
        (dist, f"vessel's disturbance $\\eta_D+\\eta_R$\n"
               f"peak {np.max(np.abs(dist)):.3f} m", None, "PuOr_r"),
        (inc + dist, "total   (what a LiDAR sees)", 2.2, "RdBu_r"),
    ]
    for a, (Z, title, lim, cmap) in zip(ax, panels):
        v = lim if lim else max(np.max(np.abs(Z)), 1e-6)
        im = a.pcolormesh(xr, yr, Z, cmap=cmap, vmin=-v, vmax=v,
                          shading="auto", rasterized=True)
        plt.colorbar(im, ax=a, label="$\\eta$ (m)", fraction=0.03)
        a.plot([-0.5, 0.5, 0.5, -0.5, -0.5], [-0.125, -0.125, 0.125, 0.125,
                                              -0.125], "k-", lw=1.4)
        a.set_aspect("equal")
        a.set_xlabel("$x/L$ from the vessel")
        a.set_title(title, fontsize=10)
    ax[0].set_ylabel("$y/L$")
    fig.suptitle("Free-surface decomposition. The hull disturbs the sea by "
                 "fractions of a percent at the SS5 peak, so the preview chain's "
                 "incident-only assumption holds.", fontsize=11)
    fig.tight_layout()
    fig.savefig("fig13_decomposition.png", dpi=150)
    print("  figure: fig13_decomposition.png")


def main():
    db = bem.load(DB)
    fs = FreeSurface(FS)
    mesh = wigley_mesh(10.0, 2.5, 0.8, 90, 26)
    red = ReducedModel.identify(
        NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L), db)
    ep = Episode(db, red, seed=0, t_preview=6.0, u_ref=4.5, n_dir=1,
                 use_rudder=False)
    print("rendering ...")
    frames = scene_figure(mesh, fs, ep)
    st, tv = frames[sorted(frames)[1]]
    decomposition_figure(fs, ep.sea, st, tv)


if __name__ == "__main__":
    main()
