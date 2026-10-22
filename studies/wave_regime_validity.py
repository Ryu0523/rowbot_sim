#!/usr/bin/env python3
"""
Where is the evaluation plant (sim/planing_vessel_gz.PlaningVesselGZ on
sim/planing_vessel.py, hull scarab195) a credible boat, across SPEED?

Probes of the plant's OWN equations (no comparison with a real boat is
possible here; the repo's validation evidence is quoted in the report):

  V1  regime numbers per speed: beam Froude C_V = U/sqrt(g b), length
      Froude Fn_L = U/sqrt(g L), volumetric Fn_vol = U/sqrt(g vol^(1/3)),
      the plant's buoyancy factor a_b(u), and at the calm running attitude
      (PlaningVessel.running_attitude, a captive run) the trim, rise, wetted
      length, the calm resistance with its breakdown and the share of the
      weight carried by the buoyancy term a_b rho g A of the strips.
  V2  steady calm speeds of constant throttle f: every crossing of the jet's
      net thrust Waterjet.forces(f t_max, 0, u)[0] with the calm resistance
      (several crossings = several equilibria).
  V3  the plant at rest, calm, no thrust: surge acceleration (a hydrostatic
      force tilted by the trim would show here).
  V4  heave free decay, surge held (captive), calm: damped period and
      damping ratio at several speeds -- the plant has no radiation states
      (docstring line 41), so this is all the heave damping it has.
  V5  steady turn, nozzle hard over, calm, LOW throttles (studies/_cache/
      final_eval/turn_gz.json has 0.3..1.0): yaw rate and speed.
  V6  encounter numbers of the scan seas: peak celerity, group speed and
      the encounter frequency of the peak component per relative heading.

  python -m studies.wave_regime_validity      (about 2-4 min, one process)
Output: studies/_cache/wave_regimes/validity.txt and validity.json
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
import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
OUT = os.path.join(HERE, "_cache", "wave_regimes")
KN = 0.514444
G = 9.81


def build():
    from sim import config
    from sim.planing_vessel_gz import PlaningVesselGZ
    from sim.test_vessel import Monochromatic
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        h, db = config.load("scarab195")
    calm = Monochromatic(1.0, 0.0)
    p = PlaningVesselGZ(h, calm, dt=h.scales()["dt"])
    return h, p, calm


def buoy_share(p, u):
    """Share of the weight carried by the strips' buoyancy term at the calm
    running attitude for speed u (as deriv computes it, ctr = 1 for this
    hull: transom_correction False)."""
    z, th = p.running_attitude(u)
    # deriv itself gives the immersion d (its sign conventions); p is calm
    s = np.zeros(14)
    s[6], s[2], s[4] = u, z, th
    _, aux = p.deriv(s, 0.0)
    d = aux["d"]
    A = p._section(d)[0]
    ab = p._a_b(u)
    ctr = p._ctr(u)
    buoy = (ab * p.rho * G * A * ctr).sum() * p.dx
    return float(buoy / (p.m * G)), float((d > 0).sum() * p.dx)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    h, p, calm = build()
    L, b, m = float(p.L), float(p.B), float(p.m)
    vol = m / p.rho
    rep = p.report()
    out = dict(hull=dict(L=L, b=b, mass=m, vol=vol, beta=h.beta_deg,
                         a_b=h.a_b, k_a=h.k_a, c_dc=float(h.c_dc),
                         transom_correction=bool(getattr(
                             h, "transom_correction", True)),
                         roll_zeta=h.roll_zeta, GM=rep["GM"],
                         T_roll=rep["T_roll"],
                         phi_vanish_deg=rep["phi_vanish_deg"],
                         phi_fail_deg=rep["phi_fail_deg"],
                         t_max=float(p.prop.t_max),
                         u_design=float(h.u_design),
                         rud_max_deg=math.degrees(p.rudder.max)))
    lines = []
    P = lines.append
    P(f"scarab195 on PlaningVesselGZ: L {L} m, chine beam b {b:.3f} m, "
      f"mass {m:.0f} kg, volume {vol:.3f} m3, deadrise {h.beta_deg} deg, "
      f"k_a {h.k_a}, C_dc {float(h.c_dc):.3f}, a_b {h.a_b}, transom "
      f"correction {getattr(h, 'transom_correction', True)}, roll zeta "
      f"{h.roll_zeta}; GM {rep['GM']:.3f} m, roll period "
      f"{rep['T_roll']:.2f} s, vanishing angle {rep['phi_vanish_deg']:.1f}"
      f" deg; t_max {p.prop.t_max:.0f} N")
    top = p.top_speed()
    out["top_speed_kn"] = top / KN
    P(f"calm top speed (PlaningVesselGZ.top_speed): {top / KN:.2f} kn")

    # ---- V1
    P("")
    P("V1  regime numbers and calm running attitude per speed "
      "(running_attitude / calm_resistance)")
    P("    kn    C_V   Fn_L  Fn_vol  a_b(u)  trim_deg  rise_m  wet_len_m "
      "R_N    press  frict  air   buoy_share")
    V1 = []
    for kn in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20,
               22, 25, 30, 35, 40):
        u = kn * KN
        cv = u / math.sqrt(G * b)
        fl = u / math.sqrt(G * L)
        fv = u / math.sqrt(G * vol ** (1 / 3))
        r = p.calm_resistance(u)
        bs, lw = buoy_share(p, u)
        row = dict(kn=kn, cv=cv, fn_l=fl, fn_vol=fv, a_b=p._a_b(u),
                   trim=r["trim_deg"], rise=r["rise"], lw=r["wetted_length"],
                   R=r["R"], press=r["pressure"], frict=r["friction"],
                   air=r["air"], buoy_share=bs)
        V1.append(row)
        P(f"   {kn:4.0f} {cv:6.2f} {fl:6.2f} {fv:6.2f} {row['a_b']:7.2f} "
          f"{r['trim_deg']:9.2f} {r['rise']:7.3f} {r['wetted_length']:9.2f}"
          f" {r['R']:6.0f} {r['pressure']:6.0f} {r['friction']:6.0f} "
          f"{r['air']:5.0f} {bs:8.2f}")
    out["V1"] = V1

    # ---- V2: steady calm speeds per throttle (all crossings)
    P("")
    P("V2  steady calm speed(s) of constant throttle f: crossings of the "
      "jet net thrust with the calm resistance on a 0.25 m/s grid")
    ug = np.arange(0.25, 24.01, 0.25)
    Rg = np.array([p.calm_resistance(u)["R"] for u in ug])
    out["R_curve"] = dict(u=ug.tolist(), R=Rg.tolist())
    dRdu = np.gradient(Rg, ug)
    neg = ug[dRdu < 0]
    P(f"    calm resistance decreasing with speed (a hump) at u in: "
      + (f"{neg.min() / KN:.1f}..{neg.max() / KN:.1f} kn "
         f"({len(neg)} grid points)" if len(neg) else "nowhere"))
    V2 = {}
    for f in (0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3,
              0.45, 0.6, 0.8, 1.0):
        T = np.array([p.prop.forces(f * p.prop.t_max, 0.0, u)[0]
                      for u in ug])
        ex = T - Rg
        idx = np.nonzero((ex[:-1] > 0) & (ex[1:] <= 0))[0]
        sp = [float(ug[i] + (ug[i + 1] - ug[i]) * ex[i] / (ex[i] - ex[i + 1]))
              / KN for i in idx]
        V2[f] = sp
        P(f"    f {f:5.3f}: stable equilibria {', '.join(f'{v:.1f}' for v in sp)}"
          f" kn (net thrust at 5 kn {T[np.argmin(abs(ug - 5 * KN))]:.0f} N)")
    out["V2"] = {str(k): v for k, v in V2.items()}

    # ---- V3: at rest
    P("")
    s = p.initial_state(0.0)
    ds, aux = p.deriv(s, 0.0)
    out["V3"] = dict(ud=float(ds[6]), wd=float(ds[8]), qd=float(ds[10]),
                     Xp=float(aux["Xp"]), th_rest_deg=math.degrees(p.Th_rest))
    P(f"V3  at rest, calm, no thrust: u' {ds[6]:+.4f} m/s2, w' {ds[8]:+.4f}, "
      f"q' {ds[10]:+.4f}; pressure-drag term Xp {aux['Xp']:+.1f} N; static "
      f"trim Th_rest {math.degrees(p.Th_rest):+.3f} deg")
    # a 60 s drift at rest
    s = p.initial_state(0.0)
    for i in range(3000):
        s = p.step(s, i * p.dt, 0.0, 0.0)
    out["V3"]["u_after_60s"] = float(s[6])
    P(f"    after 60 s at rest: u {s[6]:+.4f} m/s, x {s[0]:+.3f} m")

    # ---- V4: heave decay, captive surge
    P("")
    P("V4  heave free decay, calm, surge held (captive), from the running "
      "attitude + 0.05 m and + 0.15 m; damped period and damping ratio from "
      "the first two peaks (log decrement)")
    from sim.planing_vessel_gz import PlaningVesselGZ
    V4 = []
    for kn in (0.0, 3.0, 6.0, 10.0, 15.0, 25.0):
        u = kn * KN
        for dz in (0.05, 0.15):
            pc = PlaningVesselGZ(h, calm, dt=0.005,
                                 captive_u=(u if u > 0 else 1e-6))
            z0, th0 = pc.running_attitude(u) if u > 0 else (0.0, 0.0)
            s = np.zeros(14)
            s[6], s[2], s[4] = u, z0 + dz, th0
            n = int(12.0 / 0.005)
            zs = np.empty(n)
            for i in range(n):
                s = pc.step(s, i * 0.005, 0.0, 0.0)
                s[6] = u
                zs[i] = s[2] - z0
            from scipy.signal import find_peaks
            # settle level: mean of the last 4 s
            zr = zs - zs[-800:].mean()
            pk, _ = find_peaks(zr)
            pk = [i for i in pk if zr[i] > 0.02 * dz]
            if len(pk) >= 2:
                Td = (pk[1] - pk[0]) * 0.005
                dl = math.log(max(zr[pk[0]], 1e-9) / max(zr[pk[1]], 1e-9))
                zeta = dl / math.sqrt(4 * math.pi ** 2 + dl ** 2)
            else:
                Td, zeta = float("nan"), float("nan")
            V4.append(dict(kn=kn, dz=dz, Td=Td, zeta=zeta,
                           n_peaks=len(pk)))
            P(f"    {kn:4.0f} kn  dz {dz:.2f} m: damped period {Td:5.2f} s, "
              f"damping ratio {zeta:5.3f} ({len(pk)} peaks above 2%)")
    out["V4"] = V4

    # ---- V5: low-throttle steady turns
    P("")
    P("V5  steady turn, nozzle hard over, calm (20 s straight then 25 s "
      "turning, means over the last 8 s), low throttles; 0.3..1.0 are in "
      "studies/_cache/final_eval/turn_gz.json")
    V5 = []
    for f in (0.05, 0.1, 0.15, 0.2):
        sp = V2.get(f) or [5.0]
        pt = PlaningVesselGZ(h, calm, dt=0.02)
        s = pt.initial_state(sp[-1] * KN)
        us, rs, u_str = [], [], None
        for i in range(int(45.0 / 0.02)):
            t = i * 0.02
            s = pt.step(s, t, f * pt.prop.t_max,
                        pt.rudder.max if t >= 20.0 else 0.0)
            if abs(t - 19.98) < 1e-6:
                u_str = float(s[6])
            if t >= 37.0:
                us.append(s[6])
                rs.append(abs(s[11]))
        r_deg = math.degrees(float(np.mean(rs)))
        V5.append(dict(f=f, u_straight_kn=u_str / KN,
                       u_turn_kn=float(np.mean(us)) / KN, r_deg_s=r_deg))
        P(f"    f {f:.2f}: straight {u_str / KN:5.1f} kn -> turning "
          f"{np.mean(us) / KN:5.1f} kn, yaw rate {r_deg:5.1f} deg/s")
    out["V5"] = V5

    # ---- V6: encounter numbers
    P("")
    P("V6  scan seas: peak celerity c = g Tp / 2 pi, group speed c/2, and "
      "the peak component's encounter frequency w_e = w_p - k_p U cos(mu),"
      " mu = angle between heading and wave TRAVEL direction (180 = head)")
    V6 = {}
    for tp in (5.0, 8.0):
        wp = 2 * math.pi / tp
        kp = wp ** 2 / G
        c = G / wp
        P(f"    Tp {tp:.0f} s: lambda_p {2 * math.pi / kp:.1f} m "
          f"({2 * math.pi / kp / L:.1f} L), c {c:.2f} m/s = {c / KN:.1f} kn, "
          f"c/2 {c / 2 / KN:.1f} kn, w_p {wp:.3f} rad/s")
        P("       rel(0 head)   5 kn    10 kn   15 kn   20 kn   25 kn   "
          "30 kn   35 kn   (w_e rad/s; T_e = 2 pi / |w_e| s)")
        for chi in (0, 45, 90, 135, 180):
            mu = math.radians(180 - chi)
            we = [wp - kp * kn * KN * math.cos(mu)
                  for kn in (5, 10, 15, 20, 25, 30, 35)]
            V6[(tp, chi)] = we
            P(f"       {chi:5d}      " + " ".join(
                f"{w:+6.2f}/{(2 * math.pi / abs(w)) if abs(w) > 1e-6 else float('inf'):4.1f}"
                for w in we))
    out["V6"] = {f"{k[0]}_{k[1]}": v for k, v in V6.items()}
    P("")
    P(f"(done in {time.time() - t0:.0f} s)")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "validity.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(os.path.join(OUT, "validity.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(txt)


if __name__ == "__main__":
    main()
