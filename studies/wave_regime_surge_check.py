#!/usr/bin/env python3
"""
Surge wave force of the plant at low and planing speed, split into its two
wave-dependent parts, in a long regular head/following wave, surge held
(captive), heave and pitch free:
  Xw  the Froude-Krylov term, -rho g a_b sum A dX/dx dx
      (sim/planing_vessel.py:300)
  Xp  the 'pressure drag', the strip force tilted by the HULL's pitch,
      -sum f tau dx (sim/planing_vessel.py:355; f contains the buoyancy
      a_b rho g A, line 279)
against the textbook Froude-Krylov force on the displaced volume in a
wave much longer than the hull, -rho g Vol d(eta)/dx at the CG (= -W x
slope). Reported: first-harmonic amplitudes at the wave frequency and the
phase of each part relative to -W slope.

  python -m studies.wave_regime_surge_check
Output: studies/_cache/wave_regimes/surge_check.txt
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import contextlib  # noqa: E402
import io  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
KN = 0.514444
G = 9.81
OUT = os.path.join(HERE, "_cache", "wave_regimes")


def h1(t, x, w):
    A = np.stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)], 1)
    c = np.linalg.lstsq(A, x, rcond=None)[0]
    return complex(c[1], -c[2]), float(c[0])     # x ~ Re(z e^{iwt})


def main():
    from sim import config
    from sim.planing_vessel_gz import PlaningVesselGZ
    from sim.test_vessel import Monochromatic
    with contextlib.redirect_stdout(io.StringIO()):
        h, _ = config.load("scarab195")
    lines = []
    P = lines.append
    P("surge wave-force split, captive surge, regular wave lambda 40 m "
      "(7.4 L), amplitude 0.25 m; complex amplitudes at the ENCOUNTER "
      "frequency relative to the reference -W x (wave slope at the CG)")
    P("  kn  dir        |ref| N   Xw/ref            Xp/ref            "
      "(Xw+Xp)/ref       mean Xp N  pitch amp/slope amp")
    lam, amp = 40.0, 0.25
    k = 2 * math.pi / lam
    w0 = math.sqrt(G * k)
    for kn in (0.5, 3.0, 6.0, 9.0, 13.0, 18.0, 25.0):
        for th0, nm in ((math.pi, "head"), (0.0, "follow")):
            u = kn * KN
            sea = Monochromatic(w0, amp, theta0=th0)
            p = PlaningVesselGZ(h, sea, dt=0.005, captive_u=u)
            z0, t0 = p.running_attitude(u)
            s = np.zeros(14)
            s[6], s[2], s[4] = u, z0, t0
            we = abs(w0 - k * u * math.cos(th0))
            Tend = 20.0 + 6 * 2 * math.pi / we
            n = int(Tend / 0.005)
            T, XW, XP, SL, PT = [], [], [], [], []
            for i in range(n):
                t = i * 0.005
                if t >= 20.0:
                    _, aux = p.deriv(s, t)
                    d = aux["d"]
                    A = p._section(d)[0]
                    psi = s[5]
                    X = s[0] + p.x * math.cos(psi)
                    Y = s[1] + p.x * math.sin(psi)
                    _, _, _, ex, ey = p._waves(X, Y, t)
                    s_lon = ex * math.cos(psi) + ey * math.sin(psi)
                    xw = -(p.rho * G * p._a_b(u) * A * s_lon).sum() * p.dx
                    _, _, _, exc, eyc = p._waves(np.array([s[0]]),
                                                 np.array([s[1]]), t)
                    T.append(t)
                    XW.append(xw)
                    XP.append(aux["Xp"])
                    SL.append(float(exc[0]))
                    PT.append(float(s[4]))
                s = p.step(s, t, 0.0, 0.0)
                s[6] = u
            T, XW, XP, SL, PT = map(np.asarray, (T, XW, XP, SL, PT))
            ref = -p.m * G * SL
            zr, _ = h1(T, ref, we)
            zw, _ = h1(T, XW, we)
            zp, mp = h1(T, XP, we)
            zt = h1(T, PT, we)[0]
            zs = h1(T, SL, we)[0]

            def f(z):
                q = z / zr
                return f"{abs(q):5.2f}@{math.degrees(np.angle(q)):+5.0f}"
            P(f"  {kn:4.1f} {nm:6s} {abs(zr):9.0f}   {f(zw):16s}  "
              f"{f(zp):16s}  {f(zw + zp):16s}  {mp:+8.0f}   "
              f"{abs(zt) / abs(zs):5.2f}")
    txt = "\n".join(lines)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "surge_check.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
