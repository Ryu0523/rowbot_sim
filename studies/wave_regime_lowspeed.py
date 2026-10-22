#!/usr/bin/env python3
"""
Low-speed and hump probes of the evaluation plant (scarab195 on
PlaningVesselGZ), calm water unless stated, plus two sea-spectrum numbers:
  A  captive (surge held) 40 s from the cached running attitude at 4..12 kn:
     heave / trim range over the last 15 s (a steady boat shows none)
  B  free running, constant throttle, 120 s: speed, heave, trim ranges over
     the last 60 s
  C  calm constant-throttle equilibria 8..21 kn: throttle needed, dR/du vs
     dT/du (stable when dT/du < dR/du), resistance from validity.json
  D  share of the scan seas' energy at wavelengths 1..2.5 L (MGN 328's
     surfing / broaching lengths) and the wave speed there
  python -m studies.wave_regime_lowspeed   (about 3 min)
Output: studies/_cache/wave_regimes/lowspeed.txt
"""
import sys, os, math, contextlib, io, warnings, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); warnings.filterwarnings("ignore")
_L = []
def print(*a):
    import builtins
    line = " ".join(str(x) for x in a); _L.append(line); builtins.print(line, flush=True)
import numpy as np
from sim import config
from sim.planing_vessel_gz import PlaningVesselGZ
from sim.test_vessel import Monochromatic
KN=0.514444
with contextlib.redirect_stdout(io.StringIO()):
    h, db = config.load("scarab195")
calm = Monochromatic(1.0, 0.0)
print("captive (surge held), calm, 40 s from the cached running attitude: last 15 s heave/pitch ranges")
for kn in (4,5,6,7,8,9,10,11,12):
    u=kn*KN
    p = PlaningVesselGZ(h, calm, dt=0.005, captive_u=u)
    z0, th0 = p.running_attitude(u)
    s=np.zeros(14); s[6]=u; s[2]=z0; s[4]=th0
    zs=[];ts=[]
    for i in range(int(40/0.005)):
        s=p.step(s,i*0.005,0,0); s[6]=u
        if i*0.005>=25: zs.append(s[2]); ts.append(math.degrees(-(p.Th_rest+s[4])))
    zs=np.array(zs); ts=np.array(ts)
    print(f"  {kn:3d} kn: heave {zs.min():+.3f}..{zs.max():+.3f} m (range {np.ptp(zs):.3f}), trim {ts.min():.2f}..{ts.max():.2f} deg")
print("free running, calm, constant throttle, 120 s from initial_state(u0): last 60 s")
for f,u0 in ((0.03,5.0),(0.05,5.6),(0.05,8.1),(0.075,9.0),(0.1,9.4),(0.15,9.6),(0.2,10.8),(0.25,18.5)):
    p = PlaningVesselGZ(h, calm, dt=0.02)
    s=p.initial_state(u0*KN)
    U=[];Z=[];T=[]
    for i in range(int(120/0.02)):
        s=p.step(s,i*0.02,f*p.prop.t_max,0.0)
        if i*0.02>=60: U.append(s[6]/KN); Z.append(s[2]); T.append(math.degrees(-(p.Th_rest+s[4])))
    U=np.array(U);Z=np.array(Z);T=np.array(T)
    print(f"  f {f:.3f} from {u0:4.1f} kn: speed {U.mean():5.2f} kn (range {np.ptp(U):.2f}), heave range {np.ptp(Z):.3f} m, trim {T.mean():.2f} (range {np.ptp(T):.2f}) deg, a_cg max {abs(p.last_cg_acc)/9.81:.3f} g")

print("C  calm constant-throttle equilibria (resistance curve from validity.json)")
from sim.wavefield import SeaState
V = json.load(open(os.path.join(HERE, "_cache", "wave_regimes", "validity.json")))
uc = np.array(V["R_curve"]["u"]); Rc = np.array(V["R_curve"]["R"])
p = PlaningVesselGZ(h, calm, dt=0.02)
tm = p.prop.t_max
print(" kn    R_N   f_needed  dR/du  dT/du  stable")
for kn in range(8, 22):
    uu = kn * KN
    Ru = np.interp(uu, uc, Rc)
    dR = (np.interp(uu + 0.25, uc, Rc) - np.interp(uu - 0.25, uc, Rc)) / 0.5
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if p.prop.forces(mid * tm, 0, uu)[0] < Ru:
            lo = mid
        else:
            hi = mid
    f = 0.5 * (lo + hi)
    dT = (p.prop.forces(f * tm, 0, uu + 0.25)[0] - p.prop.forces(f * tm, 0, uu - 0.25)[0]) / 0.5
    print(f"{kn:3d} {Ru:7.0f} {f:8.3f} {dR:7.0f} {dT:6.0f}   {'yes' if dT < dR else 'NO'}")
print("D  MGN 328 lengths in the scan seas")
L, G = 5.4, 9.81
for tp in (5.0, 8.0):
    ss = SeaState(1.0, tp, theta0=0.0, n_freq=24, n_dir=5, seed=0)
    lam = 2 * np.pi / ss.k
    e = ss.a ** 2 / 2
    sel = (lam >= L) & (lam <= 2.5 * L)
    print(f"  Tp {tp}: energy share at 1-2.5 L {e[sel].sum() / e.sum():.3f}, band Hs {4 * np.sqrt(e[sel].sum()):.3f} m (MGN 328 Table 1: > 4% L = {0.04 * L:.2f} m), wave speed {np.sqrt(G * L / 2 / np.pi) / KN:.1f}-{np.sqrt(G * 2.5 * L / 2 / np.pi) / KN:.1f} kn, shortest modelled wave {lam.min():.1f} m")
with open(os.path.join(HERE, "_cache", "wave_regimes", "lowspeed.txt"),
          "w") as _fh:
    _fh.write(chr(10).join(_L) + chr(10))
