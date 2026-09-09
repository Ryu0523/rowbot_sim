#!/usr/bin/env python3
"""
Can the controller's internal model actually CARRY the preview -- while steering?

Every negative preview result could have a duller explanation than physics: the
MPC optimises a 10-state caricature, and if that caricature cannot predict what
the real 110-state plant will do 8 seconds out, perfect knowledge of the future
waves is worthless. Not because preview is useless, but because the controller
cannot propagate it. That is a CONTROLLER defect and it is fixable, so it has to
be ruled out before any negative result means anything.

The first version of this test found exactly that, twice over: forward Euler on
a stiff oscillator diverged to 10^12 times the signal, and a variable-shadowing
bug added the rudder angle to the pitch angle.

But it ran with the RUDDER LOCKED, like every preview study before it. So it
checked the model's heave and pitch and nothing else. Now that the reduced model
has a sway state and the MPC can steer (heading RMS 476 deg -> 0.9 deg), the
question is broader: does the model predict the STEERING response too? Yaw and
cross-track are what a heading policy would act on, and they had never been
verified at all.

The model is given every advantage: exact plant state to start from, the exact
future thrust AND rudder the plant used, and the exact future wave field. Any
error below is its own.

Run: python -m studies.model_horizon
"""
import numpy as np

from hydro import bem
from sim.env import Episode
from sim.vessel import NonlinearVessel
from sim.test_vessel import Monochromatic
from control.reduced import ReducedModel
from control.mpc import PreviewProvider

DB = "hydro_wigley_10m.npz"
G = 9.81
SEEDS = (0, 1, 2, 3)
H = 24              # the MPC's own horizon, 24 x 0.5 s = 12 s
DT = 0.5
N_ST = 5

# (label, use_rudder, autopilot, n_dir)
MODES = [("rudder locked", False, False, 1),
         ("MPC steering", True, False, 5)]

# which predicted signals to score, and where they live in the reduced state
CHANNELS = [("bow accel", None), ("heading", 7), ("cross-track", 1),
            ("sway vel", 9), ("yaw rate", 8)]


def collect(db, red, seed, use_rudder, autopilot, n_dir, t_end=200.0,
            hs=3.25, tp=9.7):
    ep = Episode(db, red, hs=hs, tp=tp, seed=seed, t_preview=0.0, u_ref=4.5,
                 n_dir=n_dir, use_rudder=use_rudder, autopilot=autopilot)
    pv = PreviewProvider(ep.sea, t_preview=1e9)      # perfect, unlimited
    x_st = np.linspace(-db.L / 2, db.L / 2, N_ST)
    hb = 0.5 * float(db.attrs.get("B", 2.5))
    y_off = np.array([-hb, 0.0, hb])

    s = ep.plant.initial_state(ep.u_ref * 0.8)
    t = 0.0
    hist = []
    for _ in range(int(t_end / ep.dt_ctrl)):
        sr = ep.reduced.from_plant_state(s)
        thr, rud = ep.ctrl.to_actuator(ep.ctrl(sr, t))
        if autopilot:
            rud = ep._steer(s)
        hist.append((t, sr.copy(), thr, rud, ep.plant.last_bow_acc / G))
        for _ in range(ep.sub):
            s = ep.plant.step(s, t, thr, rud, ep.dt)
            t += ep.dt
    n = len(hist)

    nch = len(CHANNELS)
    pred = np.full((n, H, nch), np.nan)
    true = np.full((n, H, nch), np.nan)
    for i in range(n - H - 1):
        t0, s0, _, _, _ = hist[i]
        x = s0[None, :].copy()
        for h in range(H):
            th, rd = hist[i + h][2], hist[i + h][3]
            psi_h = x[:, 7:8]
            ch, sh = np.cos(psi_h), np.sin(psi_h)
            xb = x_st[None, :, None]
            yb = y_off[None, None, :]
            xs = x[:, 0:1, None] + xb * ch[..., None] - yb * sh[..., None]
            ys = x[:, 1:2, None] + xb * sh[..., None] + yb * ch[..., None]
            eta = pv.at(xs, ys, t0 + h * DT, 0.0)
            x, a_bow, _ = red.step(x, np.array([th]), np.array([rd]),
                                   eta, x_st, DT)
            tgt = hist[i + h]
            for c, (_, idx) in enumerate(CHANNELS):
                pred[i, h, c] = a_bow[0] / G if idx is None else x[0, idx]
                true[i, h, c] = tgt[4] if idx is None else tgt[1][idx]
    return pred, true


def score(P, T, h):
    a, b = P[:, h], T[:, h]
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 50 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    db = bem.load(DB)
    red = ReducedModel.identify(
        NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L), db)
    print("reduced model vs plant, given PERFECT future waves and the exact")
    print("future thrust AND rudder -- every error below belongs to the model")

    worst = {}
    for label, ur, ap, nd in MODES:
        P, T = [], []
        for sd in SEEDS:
            p, t = collect(db, red, sd, ur, ap, nd)
            P.append(p); T.append(t)
        P = np.vstack(P); T = np.vstack(T)

        print(f"\n  {label}   (correlation with the plant, by horizon time)")
        print(f"  {'ahead':>7}" + "".join(f"{c[0]:>13}" for c in CHANNELS))
        for h in (1, 3, 7, 15, 23):
            row = f"  {(h)*DT:>5.1f} s"
            for c in range(len(CHANNELS)):
                v = score(P[:, :, c], T[:, :, c], h)
                row += f"{v:>13.3f}" if np.isfinite(v) else f"{'--':>13}"
            print(row)
        # the horizon each channel stays usable to
        line = f"  {'usable to':>7}"
        for c in range(len(CHANNELS)):
            good = [h for h in range(H)
                    if np.isfinite(score(P[:, :, c], T[:, :, c], h))
                    and score(P[:, :, c], T[:, :, c], h) > 0.5]
            hv = (max(good) + 1) * DT if good else 0.0
            worst[(label, CHANNELS[c][0])] = hv
            line += f"{hv:>11.1f} s"
        print(line)

    print("\n  " + "-" * 70)
    steer = [v for (lab, ch), v in worst.items()
             if lab == "MPC steering" and ch in ("heading", "cross-track",
                                                 "sway vel", "yaw rate")]
    mn = min(steer) if steer else 0.0
    print(f"  MPC plans over {H*DT:.1f} s.")
    print(f"  Weakest steering channel stays usable to {mn:.1f} s.")
    if mn >= H * DT:
        print("\n  => the model carries the steering response across the whole")
        print("     horizon, so a preview result obtained WITH the rudder free")
        print("     cannot be blamed on the internal model.")
    else:
        print(f"\n  => beyond {mn:.1f} s the steering prediction is guessing.")
        print("     Any preview conclusion about heading is bounded by THAT,")
        print("     not by the sensor.")
    return worst


if __name__ == "__main__":
    main()
