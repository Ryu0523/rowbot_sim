#!/usr/bin/env python3
"""
Export one recorded closed-loop stretch for the real-time web prototype
(viewer/proto_web.html): the exact sea components, the boat's pose at every
plant step, and the scarab195 hull's own station geometry, written to
viewer/proto_web_data.json.

    python -m studies.visual_export_web                    # defaults below
    python -m studies.visual_export_web --run m0_s3_l0 --t0 80 --t1 98

THE STRETCH (default) is the one the Blender prototype uses
(studies/visual_export.py, viewer/proto_blender/scene.json): run m0_s3_l0
(row m0, mission seed 3, leg 0; head seas, Hs 1.0 m, Tp 5 s, ~27 kn). At
~85.9 s the boat leaves a crest, is airborne to ~86.7 s and lands at 86.76 s
with 11.4 g at the CG. Playback runs t0..t1; frames start `pre` seconds
earlier so the page can draw the wake trail already at t0.

FRAMES AND SIGNS (sim/planing_vessel.py): world X, Y horizontal, Z up,
right-handed; eta = sum a cos(k (X cos th + Y sin th) - w t + phi)
(learn/meta/data5.sea_eval). State [x, y, z, roll, pitch, yaw, u, ...]; z and
pitch are measured from the floating-at-rest attitude, so the exported CG
height is Z_rest + z and the exported pitch Th_rest + pitch. A body point
(xb, yb, zb) (x forward, y left, z up, from the CG) is at
    P = (x, y, Z_rest + z) + Rz(yaw) Ry(pitch_tot) Rx(roll) (xb, yb, zb),
which to first order is the plant's own keel height
Z_rest + z + zk + SP x (Th_rest + pitch) + y roll (SP = -1); the script
checks the rigid transform against that formula.

REBASING (exact): positions are written relative to the boat's position at
t0 (X0, Y0) and times relative to t0, so the page's float32 shaders never
see large numbers; every component phase is shifted to keep the sum
identical,  phi' = phi + k (X0 cos th + Y0 sin th) - w t0  (mod 2 pi).
The script evaluates the sum both ways and prints the difference.
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
OUT = os.path.join(ROOT, "viewer", "proto_web_data.json")
KN = 0.514444
G = 9.81


def sea_eval(a, k, th, w, phi, X, Y, t):
    X, Y, t = (np.asarray(v, float) for v in (X, Y, t))
    ph = (k * (X[..., None] * np.cos(th) + Y[..., None] * np.sin(th))
          - w * t[..., None] + phi)
    return (a * np.cos(ph)).sum(-1)


def rot(roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll)."""
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
    ap.add_argument("--t0", type=float, default=80.0)
    ap.add_argument("--t1", type=float, default=98.0)
    ap.add_argument("--pre", type=float, default=14.0,
                    help="seconds of frames before t0 (wake trail)")
    ap.add_argument("--still", type=float, action="append")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    stills = a.still or [86.10, 86.78]

    from hydro.hulls import get
    from sim.planing_vessel_gz import PlaningVesselGZ
    from sim.planing_vessel import SP

    path = os.path.join(RUNS, a.run + ".pkl")
    d = pickle.load(open(path, "rb"))
    sd = d["sea"]
    A, K, TH, W, PHI = (np.asarray(sd[n], float)
                        for n in ("a", "k", "th", "w", "phi"))
    sea = SimpleNamespace(a=A, k=K, th=TH, w=W, phi=PHI)
    h = get("scarab195")
    p = PlaningVesselGZ(h, sea, dt=d["dt"])
    S, T = np.asarray(d["s"], float), np.asarray(d["t"], float)

    # ---- the stretch, at every plant step
    ta = max(T[0], a.t0 - a.pre)
    sel = (T >= ta - 1e-9) & (T <= a.t1 + 1e-9)
    idx = np.nonzero(sel)[0]
    St, Tt = S[idx], T[idx]
    i0 = int(np.argmin(np.abs(T - a.t0)))
    X0, Y0, T0 = float(S[i0, 0]), float(S[i0, 1]), float(a.t0)
    phi2 = np.mod(PHI + K * (X0 * np.cos(TH) + Y0 * np.sin(TH)) - W * T0,
                  2 * np.pi)

    # rebasing check: the sum at the hull's stations, both ways
    xs = p.x
    Xw = St[:, :1] + xs[None] * np.cos(St[:, 5:6])
    Yw = St[:, 1:2] + xs[None] * np.sin(St[:, 5:6])
    e_abs = sea_eval(A, K, TH, W, PHI, Xw, Yw, np.repeat(Tt[:, None],
                                                         len(xs), 1))
    e_reb = sea_eval(A, K, TH, W, phi2, Xw - X0, Yw - Y0,
                     np.repeat(Tt[:, None] - T0, len(xs), 1))
    err_rebase = float(np.abs(e_abs - e_reb).max())

    # rigid transform vs the plant's keel formula, and the immersion
    keel_plant = (p.Z_rest + St[:, 2:3] + p.zk[None]
                  + SP * xs[None] * (p.Th_rest + St[:, 4:5])
                  + 0.0 * St[:, 3:4])
    err_rigid = 0.0
    for j in range(0, len(St), 10):
        R = rot(St[j, 3], p.Th_rest + St[j, 4], St[j, 5])
        P = np.stack([xs, np.zeros_like(xs), p.zk])
        zr = p.Z_rest + St[j, 2] + (R @ P)[2]
        err_rigid = max(err_rigid, float(np.abs(zr - keel_plant[j]).max()))
    imm = e_abs - keel_plant

    a_cg = np.asarray(d["a_cg"], float)[idx]
    a_bow = np.asarray(d["a_bow"], float)[idx]
    fr = dict(
        t=np.round(Tt - T0, 4), x=np.round(St[:, 0] - X0, 4),
        y=np.round(St[:, 1] - Y0, 4), z=np.round(p.Z_rest + St[:, 2], 5),
        roll=np.round(St[:, 3], 6), pitch=np.round(p.Th_rest + St[:, 4], 6),
        yaw=np.round(St[:, 5], 6), u=np.round(St[:, 6], 4),
        v=np.round(St[:, 7], 4), w=np.round(St[:, 8], 4),
        a_cg=np.round(a_cg, 3), a_bow=np.round(a_bow, 3),
        thrust=np.round(St[:, 12], 1), nozzle=np.round(St[:, 13], 5),
        wet=np.round((imm > 0).mean(1), 4))
    frames = {k: v.tolist() for k, v in fr.items()}

    # the hull as every viewer draws it (app/export.hull_dict): the plant's
    # station arrays and the chine and keel from hydro/hull_geometry on 201
    # points (x_fine, yc_fine, zk_fine) -- never taper / stem_rise
    from app.export import hull_dict
    hull = hull_dict(h, p)

    still_info = []
    for ts in stills:
        j = int(np.argmin(np.abs(Tt - ts)))
        still_info.append(dict(
            t=float(Tt[j]), t_rel=float(Tt[j] - T0),
            z=float(p.Z_rest + St[j, 2]),
            pitch_deg=float(np.degrees(p.Th_rest + St[j, 4])),
            roll_deg=float(np.degrees(St[j, 3])), kn=float(St[j, 6] / KN),
            a_cg_g=float(a_cg[j] / G), wet=float((imm[j] > 0).mean())))

    out = dict(
        meta=dict(
            source=os.path.relpath(path, ROOT).replace("\\", "/"),
            run=a.run, row=d["row"], seed=int(d["seed"]), leg=int(d["leg"]),
            plant="sim/planing_vessel_gz.PlaningVesselGZ (scarab195)",
            hs=float(sd.get("hs", np.nan)), tp=float(sd.get("tp", np.nan)),
            dt=float(d["dt"]), t0=T0, t1=float(a.t1), t_first=float(Tt[0]),
            origin=[X0, Y0], g=G,
            conventions=(
                "world X,Y horizontal, Z up, right-handed; positions minus "
                "origin, times minus t0; eta = sum a cos(k(X cos th + Y sin "
                "th) - w t + phi) with the rebased phi; z = CG height above "
                "calm water; pitch = total trim, > 0 bow down; R = Rz(yaw) "
                "Ry(pitch) Rx(roll); a_cg, a_bow = vertical acceleration "
                "(kinematic, m/s^2: -g in free fall)"),
            check_rebase_max_m=err_rebase,
            check_keel_rigid_vs_plant_m=err_rigid,
            stills=still_info),
        sea=dict(a=A.tolist(), k=K.tolist(), th=TH.tolist(), w=W.tolist(),
                 phi=phi2.tolist()),
        hull=hull, frames=frames)
    with open(a.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{a.out}: {len(Tt)} plant steps {Tt[0]:.2f}-{Tt[-1]:.2f} s "
          f"(play {a.t0}-{a.t1}), {os.path.getsize(a.out) / 1e3:.0f} kB; "
          f"rebase check {err_rebase:.1e} m; keel rigid vs plant "
          f"{err_rigid * 1000:.1f} mm")
    for s in still_info:
        print("still t %.2f: z %.2f m pitch %.1f deg roll %.1f deg %.1f kn "
              "a_cg %.1f g wetted %.2f" % (s["t"], s["z"], s["pitch_deg"],
                                           s["roll_deg"], s["kn"],
                                           s["a_cg_g"], s["wet"]))


if __name__ == "__main__":
    main()
