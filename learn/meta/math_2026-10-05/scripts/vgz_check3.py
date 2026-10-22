import os, sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
ROOT = "C:/Users/Administrator/Documents/usv-rl-mpc"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
from learn.repro.task import ctx
from sim.planing_vessel import PlaningVessel
from sim.planing_vessel_gz import PlaningVesselGZ
from sim.test_vessel import Monochromatic
c = ctx()
with contextlib.redirect_stdout(io.StringIO()):
    pl = PlaningVesselGZ(c["h"], Monochromatic(1.0, 0.0), dt=0.02)
u = 30 * 0.514444
z, th = pl.running_attitude(u)
v = PlaningVessel.__new__(PlaningVessel); v.__dict__.update(pl.__dict__)
v.sea = Monochromatic(1.0, 0.0); v._wave_arrays(); v.captive_u = u
s = np.zeros(14); s[6] = u; s[2] = z; s[4] = th
k, aux = v.deriv(s, 0.0); print("captive wd qd at cached attitude", aux["wd"], aux["qd"], "cached", z, th)
zs = []
for i in range(6000):
    k1, _ = v.deriv(s, 0.0); s2 = s + 0.005 * k1; k2, _ = v.deriv(s2, 0.0); s = s + 0.0025 * (k1 + k2); s[6] = u
    zs.append((s[2], s[4]))
zs = np.array(zs[-2000:]); print("last 10 s captive: z range", zs[:, 0].min(), zs[:, 0].max(), " th range", zs[:, 1].min(), zs[:, 1].max())
