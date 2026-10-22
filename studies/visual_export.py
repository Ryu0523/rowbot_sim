#!/usr/bin/env python3
"""
Export one recorded closed-loop stretch for the Blender visual prototype
(viewer/proto_blender/render.py): the exact sea, the boat's pose at video
frame times, the boat's past track (for the wake mask) and the hull's own
station geometry, all in viewer/proto_blender/scene.json.

    python -m studies.visual_export                       # defaults below
    python -m studies.visual_export --run m0_s3_l0 --t0 84.0 --t1 88.0 \\
        --fps 30 --still 86.10 --still 86.78

THE STRETCH (default): run m0_s3_l0 (row m0, mission seed 3, leg 0; head
seas, Hs 1.0 m, Tp 5 s, ~27 kn). At t 85.7 s the boat leaves a crest, is
fully airborne from ~85.9 to ~86.7 s (CG up to 1.49 m above its rest
height, 7 deg bow up) and lands at 86.76 s with 111 m/s^2 (11.4 g) at the
CG -- the hardest CG acceleration of all sixteen m0 runs.

FRAMES AND SIGNS (sim/planing_vessel.py, verified here): world X, Y
horizontal, Z up, right-handed; the elevation is
    eta = sum a cos(k (X cos th + Y sin th) - w t + phi)   (data5.sea_eval).
State [x, y, z, roll, pitch, yaw, ...]: z and pitch from the floating-at-rest
attitude, so the CG height is Z_rest + z and the total pitch Th_rest + pitch.
Station keel height Z_rest + z + zk + SP x (Th_rest + pitch), SP = -1: a
positive pitch is a right-handed rotation about body +y and lowers the bow
(bow up = negative). Station plan position X = x + xb cos(yaw),
Y = y + xb sin(yaw): yaw is right-handed about +Z, body +y is 90 deg
counter-clockwise from the bow (the GZ code calls it 'starboard', but in this
Z-up right-handed frame it is the left side; roll is right-handed about +x
and lifts the +y side). So a Blender object with rotation_euler (roll,
pitch_total, yaw) in mode 'XYZ' (R = Rz Ry Rx) reproduces the plant's
geometry; the script checks the keel heights against the plant formula.
"""
import argparse
import json
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

RUNS = os.path.join(HERE, "_cache", "wave_phase", "closedloop", "runs")
OUT = os.path.join(ROOT, "viewer", "proto_blender", "scene.json")
KN = 0.514444


def sea_eval(sea, X, Y, t):
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    ph = sea.k * (X[..., None] * np.cos(sea.th) + Y[..., None]
                  * np.sin(sea.th)) - sea.w * np.asarray(t, float)[
        ..., None] + sea.phi
    return (sea.a * np.cos(ph)).sum(-1)


def rot_xyz(roll, pitch, yaw):
    """Blender 'XYZ' Euler: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="m0_s3_l0")
    ap.add_argument("--t0", type=float, default=84.0)
    ap.add_argument("--t1", type=float, default=88.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--still", type=float, action="append")
    ap.add_argument("--track_s", type=float, default=30.0,
                    help="seconds of past track kept for the wake mask")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    stills = a.still or [86.10, 86.78]

    from hydro.hulls import get
    from sim.planing_vessel_gz import PlaningVesselGZ
    from sim.planing_vessel import SP

    path = os.path.join(RUNS, a.run + ".pkl")
    d = pickle.load(open(path, "rb"))
    sd = d["sea"]
    sea = SimpleNamespace(**{k: np.asarray(sd[k], float)
                             for k in ("a", "k", "th", "w", "phi")})
    h = get("scarab195")
    p = PlaningVesselGZ(h, sea, dt=d["dt"])
    S, T = d["s"], d["t"]

    # ---- hull: the plant's own stations, from the geometry module the
    # plant reads (hydro/hull_geometry), not re-derived from the hull fields
    # taper / stem_rise (v0's family member only: wrong for a transom run,
    # a raked keel or an offsets hull)
    from hydro.hull_geometry import stations_of
    py, pz, zw = p._sections()
    g = stations_of(h)
    assert np.array_equal(g.yc, p.yc) and np.array_equal(
        -h.vcg + g.rise, p.zk), "the geometry module differs from the plant"
    # points dense where the lines bend (the stem): the polyline is within
    # app/export.LINE_TOL of the module everywhere (app/export.line_points)
    from app.export import line_points
    xf = line_points(g)
    rf = g.keel(xf)[0]
    hull = dict(
        name=h.name, L=h.L, b=h.b, beta_deg=h.beta_deg, lcg=h.lcg,
        vcg=h.vcg, depth=h.depth, taper=h.taper, stem_rise=h.stem_rise,
        mass=h.mass, Z_rest=p.Z_rest, Th_rest=p.Th_rest, T_rest=p.T,
        x=p.x.tolist(), yc=p.yc.tolist(), zk=p.zk.tolist(),
        z_chine=pz[:, 1].tolist(), z_deck=pz[:, 2].tolist(),
        z_waterline_rest=zw.tolist(),
        nozzle=[float(v) for v in p.thrusters[0]],
        intake=[float(v) for v in p.intakes[0]],
        # the chine and keel of the geometry module at line_points (draw
        # from these, or from x / yc / zk -- never rebuild them from taper
        # / stem_rise)
        x_fine=(xf - h.lcg).tolist(), yc_fine=g.chine(xf).tolist(),
        zk_fine=(-h.vcg + rf).tolist(),
        note=("body frame from the CG: x forward, y left, z up; stations "
              "from transom to stem; section = keel (0, zk), chine "
              "(+-yc, zk + min(yc tan(beta), depth)), deck edge (+-yc, "
              "zk + depth); chine and keel from hydro/hull_geometry "
              "(x_fine, yc_fine, zk_fine resample them) -- the plant's "
              "hull, not a CAD model"))

    # ---- frames at the video rate, the plant state interpolated linearly
    tf = np.arange(a.t0, a.t1 + 0.5 / a.fps, 1.0 / a.fps)
    for ts in stills:
        if not (a.t0 <= ts <= a.t1):
            raise SystemExit(f"still {ts} outside {a.t0}..{a.t1}")

    def state_at(t):
        i = int(np.clip(np.searchsorted(T, t) - 1, 0, len(T) - 2))
        f = (t - T[i]) / (T[i + 1] - T[i])
        s = (1 - f) * S[i] + f * S[i + 1]
        # unwrap yaw before interpolating
        dy = (S[i + 1, 5] - S[i, 5] + np.pi) % (2 * np.pi) - np.pi
        s[5] = S[i, 5] + f * dy
        acg = (1 - f) * d["a_cg"][i] + f * d["a_cg"][i + 1]
        return s, acg

    def immersion(s, t):
        X = s[0] + p.x * np.cos(s[5])
        Y = s[1] + p.x * np.sin(s[5])
        e = sea_eval(sea, X, Y, t)
        keel = p.Z_rest + s[2] + p.zk + SP * p.x * (p.Th_rest + s[4])
        return e - keel, keel

    frames, err_max = [], 0.0
    for t in tf:
        s, acg = state_at(t)
        dd, keel = immersion(s, t)
        pose = dict(t=float(t), x=float(s[0]), y=float(s[1]),
                    z=float(p.Z_rest + s[2]), roll=float(s[3]),
                    pitch=float(p.Th_rest + s[4]), yaw=float(s[5]),
                    u=float(s[6]), v=float(s[7]), a_cg=float(acg),
                    thrust=float(s[12]), nozzle=float(s[13]),
                    wet=float((dd > 0).mean()), d=np.round(dd, 4).tolist())
        # the rigid transform Blender will apply, against the plant formula
        R = rot_xyz(pose["roll"], pose["pitch"], pose["yaw"])
        P = np.stack([p.x, np.zeros_like(p.x), p.zk])
        zr = pose["z"] + (R @ P)[2]
        err_max = max(err_max, float(np.abs(zr - keel).max()))
        frames.append(pose)

    # ---- past track for the wake: every 0.05 s from t0 - track_s to t1
    tt = np.arange(max(T[0], a.t0 - a.track_s), a.t1 + 1e-9, 0.05)
    track = []
    for t in tt:
        s, _ = state_at(t)
        dd, _ = immersion(s, t)
        wet = dd > 0
        track.append([round(float(t), 3), float(s[0]), float(s[1]),
                      float(s[5]), float(s[6]),
                      float(wet.mean()),                       # wetted share
                      float(max(dd[0], 0.0)),                  # transom depth
                      float(dd[wet].mean()) if wet.any() else 0.0])

    # ---- plant steps (0.02 s) from t0 - 1 s: pose, keel immersion d and
    # the plant's own immersion rate V in the earth-fixed transverse plane
    # (sim/planing_vessel.deriv: V = eta_t - (w + SP x q) + u tau), for the
    # spray model (Wagner spray-root speed (pi/2)/tan(beta) V)
    ks = np.nonzero((T >= a.t0 - 1.0) & (T <= a.t1 + 1e-9))[0]
    steps = []
    for k in ks:
        _, aux = p.deriv(S[k].copy(), T[k])
        steps.append(dict(
            t=round(float(T[k]), 4), x=float(S[k, 0]), y=float(S[k, 1]),
            z=float(p.Z_rest + S[k, 2]), roll=float(S[k, 3]),
            pitch=float(p.Th_rest + S[k, 4]), yaw=float(S[k, 5]),
            u=float(S[k, 6]), d=np.round(aux["d"], 4).tolist(),
            V=np.round(aux["V"], 3).tolist()))

    ev = []
    for ts in stills:
        k = int(np.argmin(np.abs(tf - ts)))
        fr = frames[k]
        ev.append(dict(t_req=ts, frame=k, t=fr["t"], z=fr["z"],
                       pitch_deg=np.degrees(fr["pitch"]),
                       roll_deg=np.degrees(fr["roll"]), kn=fr["u"] / KN,
                       a_cg=fr["a_cg"], wet=fr["wet"],
                       d_min=min(fr["d"]), d_max=max(fr["d"])))
    out = dict(
        meta=dict(
            source=os.path.relpath(path, ROOT).replace("\\", "/"),
            run=a.run, row=d["row"], seed=d["seed"], leg=d["leg"],
            plant="sim/planing_vessel_gz.PlaningVesselGZ (scarab195)",
            hs=float(sd.get("hs", np.nan)), tp=float(sd.get("tp", np.nan)),
            fps=a.fps, t0=a.t0, t1=a.t1,
            conventions=("world X,Y horizontal, Z up, right-handed; eta = "
                         "sum a cos(k(X cos th + Y sin th) - w t + phi); "
                         "pose z = CG height above calm water; Blender "
                         "rotation_euler (roll, pitch, yaw) mode XYZ; "
                         "pitch > 0 = bow down"),
            check_keel_rigid_vs_plant_m=err_max,
            stills=ev,
            track_cols=["t", "x", "y", "yaw", "u", "wet_share",
                        "transom_depth", "mean_wet_depth"]),
        sea={k: np.asarray(sd[k], float).tolist()
             for k in ("a", "k", "th", "w", "phi")},
        hull=hull, frames=frames, track=track, steps=steps)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"{a.out}: {len(frames)} frames {a.t0}-{a.t1} s at {a.fps} fps, "
          f"{len(track)} track samples; keel height, rigid transform vs "
          f"plant formula: max {err_max * 1000:.1f} mm")
    for e in ev:
        print("still t %.3f (frame %d): CG z %.2f m, pitch %.1f deg (bow "
              "up < 0), roll %.1f deg, %.1f kn, a_cg %.1f m/s2, wetted "
              "share %.2f, keel immersion %.2f..%.2f m" % (
                  e["t"], e["frame"], e["z"], e["pitch_deg"], e["roll_deg"],
                  e["kn"], e["a_cg"], e["wet"], e["d_min"], e["d_max"]))


if __name__ == "__main__":
    main()
