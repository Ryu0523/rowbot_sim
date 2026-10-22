#!/usr/bin/env python3
"""
What the MPC's internal model (control/reduced.py ReducedModel, as the hand
row and, without waves, as learn/meta/model3.Model0T for the m0 row) says
about the boat ACROSS THE SPEED RANGE, against the high-fidelity plant
(sim/planing_vessel_gz.PlaningVesselGZ, the final_eval plant), in calm
water. Every reduced-model coefficient was identified at ONE operating
point (control/reduced.py identify: surge step from 0.8 u_design, decays
at the step's steady speed u_ss, yaw/jet side force at u_design); this
script measures how far each one is from the plant at 2-34 kn.

Light: captive steady runs (PlaningVessel.calm_resistance / running_
attitude) on a speed grid, the jet's force map (Waterjet.forces), free
heave / pitch decays at a few speeds (identify()'s own planing-branch
estimator), and the roll natural period (PlaningVesselGZ.report). One
process, a few minutes, no waves.

Output: studies/_cache/wave_regimes/model_gaps.json and a printed table.

    .venv/Scripts/python.exe -m studies.wave_regime_model_gaps
"""
import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np   # noqa: E402

OUT = os.path.join(HERE, "_cache", "wave_regimes")
KN = 0.514444
G = 9.81


def build():
    """(calm GZ plant, reduced parameters p, hull) exactly as final_eval's
    plant factory builds the plant (gz.plant_for), with the task's
    identified reduced parameters (learn/repro/task.ctx -> lofi.template)."""
    from learn.repro import task
    from sim import config
    from sim import planing_vessel_gz as gz
    from sim.test_vessel import Monochromatic
    c = task.ctx()
    dt = config.scales_for(c["db"], c["h"])["dt"]
    plant = gz.plant_for(c["db"], Monochromatic(1.0, 0.0), c["h"], dt=dt)
    return plant, dict(c["p"]), c["h"]


def decay(plant, u, dof, t_end=30.0):
    """identify()'s planing-branch free decay at speed u: start at the
    running attitude for u with the thrust that holds u, offset heave by
    0.5 T or pitch by 0.02 rad, hold the thrust, fit y[k+1] = a1 y[k] +
    a2 y[k-1] and map the poles to (wn, zeta). Also the plain zero-crossing
    period when there are crossings."""
    from sim.test_vessel import Monochromatic
    v = plant.with_sea(Monochromatic(1.0, 0.0))
    dt = v.dt
    s = v.initial_state(u)
    thr = float(s[12])
    col = 2 if dof == "heave" else 4
    s[col] += 0.5 * v.T if dof == "heave" else 0.02
    n = int(round(t_end / dt))
    y = np.empty(n + 1)
    y[0] = s[col]
    us = np.empty(n)
    for i in range(n):
        s = v.step(s, i * dt, thr, 0.0, dt)
        y[i + 1] = s[col]
        us[i] = s[6]
    if not np.all(np.isfinite(y)):
        return dict(ok=False)
    y = y - np.median(y[3 * len(y) // 4:])
    thr_y = 0.02 * abs(y[0])
    live = np.nonzero(np.abs(y) > thr_y)[0]
    end = int(live[-1]) if len(live) else n
    seg = y[1:max(end, 8) + 1]
    X = np.column_stack([seg[1:-1], seg[:-2]])
    a1, a2 = np.linalg.lstsq(X, seg[2:], rcond=None)[0]
    zp = np.roots([1.0, -a1, -a2]).astype(complex)
    sp = np.log(zp) / dt
    wn = float(np.sqrt(np.abs(np.prod(sp))))
    zeta = float(-np.sum(sp.real) / (2.0 * wn)) if wn > 0 else float("nan")
    return dict(ok=True, wn=wn, zeta=zeta, u_mean=float(us.mean()),
                u_drift=float(us[-1] - us[0]), thrust_hold=thr)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    plant, p, hull = build()
    t_max = float(plant.prop.t_max)
    rud_max = float(plant.rudder.max)
    m = float(plant.m)
    b = float(hull.b)
    vol = m / float(plant.rho)
    out = dict(note="calm water; plant = PlaningVesselGZ (final_eval 'gz'); "
                    "reduced = learn/repro/task.ctx()['p'] "
                    "(studies/_cache/lofi_scarab195.json)",
               plant=dict(L=float(plant.L), b_chine=b, mass=m, t_max=t_max,
                          rud_max_deg=math.degrees(rud_max), T=float(plant.T),
                          volume=vol,
                          cv_planing_13kn_note="hydro/planing.py:12 "
                          "Savitsky in practice C_V > 1.5"),
               reduced={k: p[k] for k in (
                   "k_drag", "m_surge", "tau_u", "u_ss", "u_max", "u_design",
                   "wn_heave", "z_heave", "wn_pitch", "z_pitch", "tau_r",
                   "k_jet_side", "k_nomoto_f", "z0", "th0", "t_max")})
    print(f"plant: L {plant.L:.2f} m, chine beam {b:.3f} m, mass {m:.0f} kg, "
          f"t_max {t_max:.0f} N, nozzle {math.degrees(rud_max):.0f} deg; "
          f"C_V = 1 at {math.sqrt(G * b) / KN:.1f} kn, C_V = 1.5 at "
          f"{1.5 * math.sqrt(G * b) / KN:.1f} kn; Fn_vol = 1 at "
          f"{math.sqrt(G * vol ** (1 / 3)) / KN:.1f} kn", flush=True)

    # ---- 1. calm resistance, running attitude, jet force map vs speed
    kn_grid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18,
               20, 22, 25, 28, 31, 34, 37, 40]
    rows = []
    side_red = p["k_jet_side"] * t_max * math.sin(rud_max)
    print("\n  kn   C_V  R_plant  R_red  R_red/R_pl  rise_m  trim_deg  "
          "z_red-z_pl  th_red-th_pl(deg)  Tnet_full  side_full_pl  "
          "side_red  wet_len", flush=True)
    for kn in kn_grid:
        u = kn * KN
        cr = plant.calm_resistance(u)
        z_pl, th_pl = plant.running_attitude(u)
        prime, plant.prop.prime = plant.prop.prime, 1.0
        fx_full, _ = plant.prop.forces(t_max, 0.0, u)
        _, fy_full = plant.prop.forces(t_max, rud_max, u)
        plant.prop.prime = prime
        r_red = p["k_drag"] * u * u
        row = dict(kn=kn, u=u, cv=u / math.sqrt(G * b),
                   fn_vol=u / math.sqrt(G * vol ** (1 / 3)),
                   fn_L=u / math.sqrt(G * plant.L),
                   R_plant=float(cr["R"]), R_red=float(r_red),
                   pressure=float(cr["pressure"]),
                   friction=float(cr["friction"]), air=float(cr["air"]),
                   trim_deg=float(cr["trim_deg"]), rise=float(z_pl),
                   pitch_state=float(th_pl),
                   wetted_length=float(cr["wetted_length"]),
                   dz_red_minus_plant=float(p["z0"] - z_pl),
                   dth_red_minus_plant_deg=float(math.degrees(p["th0"]
                                                              - th_pl)),
                   Tnet_full=float(fx_full), side_full_plant=float(fy_full),
                   side_full_red=float(side_red))
        rows.append(row)
        print(f"  {kn:3d} {row['cv']:5.2f} {row['R_plant']:7.0f} "
              f"{r_red:6.0f} {r_red / max(row['R_plant'], 1e-9):9.2f} "
              f"{z_pl:7.3f} {row['trim_deg']:8.2f} "
              f"{row['dz_red_minus_plant']:10.3f} "
              f"{row['dth_red_minus_plant_deg']:12.2f} "
              f"{fx_full:10.0f} {fy_full:12.0f} {side_red:9.0f} "
              f"{row['wetted_length']:7.2f}", flush=True)
    out["speed_grid"] = rows

    # ---- 2. steady speed vs throttle: plant (R(u) = T_net(f t_max, u),
    # on the grid, linear interpolation) vs reduced (f t_max = k u^2)
    U = np.array([r["u"] for r in rows])
    R = np.array([r["R_plant"] for r in rows])
    hump = [dict(kn=rows[i]["kn"], R=rows[i]["R_plant"])
            for i in range(1, len(rows) - 1)
            if R[i] > R[i - 1] and R[i] > R[i + 1]]
    out["plant_resistance_local_maxima"] = hump
    thr_rows = []
    print("\n  throttle  u_plant_kn  u_red_kn  (steady, calm)", flush=True)
    for f in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
              0.9, 1.0):
        prime, plant.prop.prime = plant.prop.prime, 1.0
        T = np.array([plant.prop.forces(f * t_max, 0.0, u)[0] for u in U])
        plant.prop.prime = prime
        ex = T - R
        sols = []
        for i in range(len(U) - 1):
            if ex[i] > 0 >= ex[i + 1]:
                sols.append(float(U[i] + (U[i + 1] - U[i]) * ex[i]
                                  / (ex[i] - ex[i + 1])))
        u_red = math.sqrt(f * t_max / p["k_drag"])
        thr_rows.append(dict(f=f, u_plant_kn=[s / KN for s in sols],
                             u_red_kn=u_red / KN,
                             note=("several roots: thrust-resistance has "
                                   "more than one crossing (hump)"
                                   if len(sols) > 1 else "")))
        print(f"  {f:8.2f}  {', '.join(f'{s / KN:.1f}' for s in sols) or '-':>10}"
              f"  {u_red / KN:8.1f}", flush=True)
    out["steady_speed_vs_throttle"] = thr_rows

    # ---- 3. heave / pitch free decays at several speeds (identify's
    # planing-branch estimator) vs the reduced model's constants
    dec = []
    print("\n  kn    heave wn / zeta (plant)    pitch wn / zeta (plant)  "
          "[reduced: heave %.2f / %.2f, pitch %.2f / %.2f]"
          % (p["wn_heave"], p["z_heave"], p["wn_pitch"], p["z_pitch"]),
          flush=True)
    for kn in (5, 8, 12, 16, 20, 25, 30, 33.9):
        u = kn * KN
        hv = decay(plant, u, "heave")
        pt = decay(plant, u, "pitch")
        dec.append(dict(kn=kn, heave=hv, pitch=pt))
        f_ = (lambda d: f"{d['wn']:6.2f} / {d['zeta']:5.2f}"
              f" (u drift {d['u_drift']:+.2f})" if d.get("ok") else "failed")
        print(f"  {kn:5.1f}  {f_(hv):>26}  {f_(pt):>26}", flush=True)
    out["decays"] = dec

    # ---- 4. roll: the plant has it, the reduced model does not
    rep = plant.report()
    wn_r = math.sqrt(m * G * plant.GM / plant.Mtot[3, 3])
    out["roll"] = dict(GM=float(plant.GM), wn=float(wn_r),
                       T_roll=float(2 * math.pi / wn_r),
                       zeta=float(hull.roll_zeta),
                       report={k: (float(v) if isinstance(
                           v, (int, float, np.floating)) else str(v))
                           for k, v in rep.items()})
    print(f"\n  roll (plant only): GM {plant.GM:.3f} m, natural period "
          f"{2 * math.pi / wn_r:.2f} s, damping ratio {hull.roll_zeta}, "
          f"phi_vanish {rep.get('phi_vanish_deg', float('nan')):.1f} deg",
          flush=True)

    # ---- 5. wave surge-force scale (Froude-Krylov on a slope: m g x
    # slope amplitude pi H / lambda), against the calm resistance
    sc = []
    for tp in (5.0, 8.0, 10.0):
        lam = G * tp * tp / (2 * math.pi)
        c = G * tp / (2 * math.pi)
        for H in (1.0, 1.86):
            slope = math.pi * H / lam
            Fw = m * G * slope
            Rc = float(np.interp(c, U, R))
            sc.append(dict(tp=tp, lam=lam, c_kn=c / KN, H=H,
                           slope_amp=slope, F_wave=Fw, R_at_c=Rc,
                           F_over_R=Fw / Rc,
                           F_over_tmax=Fw / t_max))
    out["fk_surge_scale"] = sc
    print("\n  Tp   lambda  c_kn   H   slope   m g slope (N)  R_plant(c)  "
          "ratio", flush=True)
    for r in sc:
        print(f"  {r['tp']:4.1f} {r['lam']:6.1f} {r['c_kn']:5.1f} "
              f"{r['H']:4.2f} {r['slope_amp']:6.3f} {r['F_wave']:10.0f} "
              f"{r['R_at_c']:10.0f} {r['F_over_R']:6.2f}", flush=True)

    out["t_run_s"] = time.time() - t0
    with open(os.path.join(OUT, "model_gaps.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved {os.path.join(OUT, 'model_gaps.json')} "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
