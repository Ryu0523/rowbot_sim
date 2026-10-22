import os, sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
ROOT = "C:/Users/Administrator/Documents/usv-rl-mpc"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
from scipy.optimize import brentq
np.set_printoptions(precision=3, suppress=True, linewidth=160)
from learn.repro.task import ctx
from sim.planing_vessel_gz import PlaningVesselGZ
from sim.test_vessel import Monochromatic
c = ctx(); p = c["p"]
with contextlib.redirect_stdout(io.StringIO()):
    pl = PlaningVesselGZ(c["h"], Monochromatic(1.0, 0.0), dt=0.02)
KN = 0.514444
VEL = [6, 7, 11, 8, 10]
for kn in (16, 25, 30):
    u = kn * KN; z, th = pl.running_attitude(u)
    R = pl.calm_resistance(u)["R"]; Teq = brentq(lambda T: pl.prop.forces(T, 0.0, u)[0] - R, 1.0, 3 * pl.prop.t_max)
    s = np.zeros(14); s[6] = u; s[2] = z; s[4] = th; s[12] = Teq
    G = pl.deriv(s, 0.0)[0][VEL]
    cols = [6, 7, 11, 2, 8, 4, 10, 12, 13]; J = np.zeros((5, len(cols)))
    for j, k in enumerate(cols):
        e = {12: 1.0, 13: 1e-4}.get(k, 1e-5)
        a, b = s.copy(), s.copy(); a[k] += e; b[k] -= e
        J[:, j] = (pl.deriv(a, 0.0)[0][VEL] - pl.deriv(b, 0.0)[0][VEL]) / (2 * e)
    J[:, 7] *= 1000
    print(kn, "kn G(u v r w q) at Teq:", G)
    print("   rows u v r w q; cols u v r z w th q Ts(kN) delta\n", J)
