"""Render a recorded LOTUSim track (control_envs/lotusim_waypoint_demo.py's
JSON log) as a video: a top-down chart with the waypoints, the track so far
and the ship drawn to scale with its heading, plus speed, heading error and
rudder against time. MP4 through the bundled ffmpeg of imageio-ffmpeg
(pip install imageio-ffmpeg); GIF if that is missing.

    python control_envs/lotusim_track_video.py control_envs/_out/lotusim_track_square.json \
        --out control_envs/_out/lotusim_square.mp4 --speed 8
--speed N plays N simulated seconds per video second.
"""
import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import animation                     # noqa: E402

SHIP_L, SHIP_B = 142.0, 19.0           # DTMB 5415 hull, metres


def ship_outline(x, y, yaw, L=SHIP_L, B=SHIP_B):
    pts = np.array([[L * 0.5, 0.0], [L * 0.3, B * 0.5], [-L * 0.5, B * 0.5],
                    [-L * 0.5, -B * 0.5], [L * 0.3, -B * 0.5]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([x, y])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    ap.add_argument("--speed", type=float, default=8.0, help="sim seconds per video second")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--title", default="LOTUSim: DTMB hull on a waypoint course")
    args = ap.parse_args()
    with open(args.log) as f:
        d = json.load(f)
    log = d["log"]
    t = np.array([r["t"] for r in log])
    x = np.array([r["x"] for r in log])
    y = np.array([r["y"] for r in log])
    yaw = np.array([r["yaw"] for r in log])
    v = np.array([r["speed"] for r in log])
    herr = np.degrees([r["h_err"] for r in log])
    rud = np.array([r["rudder"] for r in log])
    wps = np.array(d["waypoints"])
    out = args.out or os.path.splitext(args.log)[0] + ".mp4"

    # video frames: one every (speed / fps) simulated seconds
    step = args.speed / args.fps
    frame_t = np.arange(t[0], t[-1], step)
    idx = np.searchsorted(t, frame_t)
    idx = np.clip(idx, 0, len(t) - 1)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1.0], hspace=0.45, wspace=0.25)
    axm = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    allx = np.concatenate([x, wps[:, 0]])
    ally = np.concatenate([y, wps[:, 1]])
    pad = 80
    axm.set_xlim(allx.min() - pad, allx.max() + pad)
    axm.set_ylim(ally.min() - pad, ally.max() + pad)
    axm.set_aspect("equal")
    axm.set_xlabel("x (m, world frame)")
    axm.set_ylabel("y (m)")
    axm.grid(alpha=0.3)
    axm.plot(np.r_[wps[:, 0], wps[0, 0]], np.r_[wps[:, 1], wps[0, 1]], "--", color="0.6", lw=1)
    axm.scatter(wps[:, 0], wps[:, 1], s=60, c="C1", zorder=3, label="waypoints")
    for i, (wx, wy) in enumerate(wps):
        axm.annotate(str(i + 1), (wx, wy), textcoords="offset points", xytext=(6, 6))
    (track,) = axm.plot([], [], "C0", lw=2, label="track")
    ship = plt.Polygon(ship_outline(x[0], y[0], yaw[0]), closed=True, fc="C3", ec="k", zorder=4)
    axm.add_patch(ship)
    target = axm.scatter([], [], s=200, facecolors="none", edgecolors="C3", linewidths=2, zorder=3)
    axm.legend(loc="upper left")
    txt = axm.text(0.02, 0.02, "", transform=axm.transAxes, fontsize=11,
                   bbox=dict(fc="white", alpha=0.8, ec="none"))

    for ax, ys, lab, ref in ((ax1, v, "speed (m/s)", d.get("target_speed")),
                             (ax2, herr, "heading error (deg)", 0.0),
                             (ax3, rud, "rudder command (-1..1)", 0.0)):
        ax.plot(t, ys, color="0.75", lw=1)
        if ref is not None:
            ax.axhline(ref, color="C1", ls="--", lw=1)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    ax3.set_xlabel("time (s)")
    (l1,) = ax1.plot([], [], "C0", lw=1.5)
    (l2,) = ax2.plot([], [], "C0", lw=1.5)
    (l3,) = ax3.plot([], [], "C0", lw=1.5)
    fig.suptitle(args.title)

    def update(k):
        i = idx[k]
        track.set_data(x[:i + 1], y[:i + 1])
        ship.set_xy(ship_outline(x[i], y[i], yaw[i]))
        w = int(log[i]["wp"])
        if w < len(wps):
            target.set_offsets([wps[w]])
        txt.set_text(f"t = {t[i]:5.0f} s   speed {v[i]:4.2f} m/s   "
                     f"heading error {herr[i]:+6.1f} deg   waypoint {min(w + 1, len(wps))}/{len(wps)}")
        l1.set_data(t[:i + 1], v[:i + 1])
        l2.set_data(t[:i + 1], herr[:i + 1])
        l3.set_data(t[:i + 1], rud[:i + 1])
        return track, ship, target, txt, l1, l2, l3

    anim = animation.FuncAnimation(fig, update, frames=len(idx), blit=False)
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=2500)
    except ImportError:
        out = os.path.splitext(out)[0] + ".gif"
        writer = animation.PillowWriter(fps=args.fps)
    anim.save(out, writer=writer)
    print(f"wrote {out}: {len(idx)} frames, {len(idx) / args.fps:.0f} s of video for "
          f"{t[-1] - t[0]:.0f} s simulated")


if __name__ == "__main__":
    main()
