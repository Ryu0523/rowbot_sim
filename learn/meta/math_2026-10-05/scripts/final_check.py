"""Read-only checks for comparison_final.md (no repository file written)."""
import os, sys, io, contextlib, warnings, pickle, time
warnings.filterwarnings("ignore")
ROOT = "C:/Users/Administrator/Documents/usv-rl-mpc"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
np.set_printoptions(precision=4, suppress=True, linewidth=160)
from learn.repro.task import ctx
from learn.meta.cat_water_a import _flutter_kcrit
c = ctx(); p = c["p"]
D, h = 0.24, 0.04
print("T", p["T"], "L", p["L"], "B", p["B"], "u_design", p["u_design"])

# (a) kick-vs-continuous ratio for a constant push over the control step
def phi(w, z, t):
    return expm(np.array([[0, 1], [-w * w, -2 * z * w]]) * t)
for nm, w, z in (("heave", p["wn_heave"], p["z_heave"]), ("pitch", p["wn_pitch"], p["z_pitch"])):
    kicks = h * sum(phi(w, z, i * h)[1, 1] for i in range(6))
    cont = phi(w, z, D)[0, 1]
    per = [h * phi(w, z, D - (j + 1) * h)[1, 1] / (phi(w, z, D - j * h)[0, 1] - phi(w, z, D - (j + 1) * h)[0, 1]) for j in range(6)]
    print(f"{nm}: kicks/continuous {kicks / cont:.3f}; per-substep {np.round(per, 3)}")
tr = p["tau_r"]
kicks = h * sum(np.exp(-i * h / tr) for i in range(6)); cont = tr * (1 - np.exp(-D / tr))
print(f"yaw: kicks/continuous {kicks / cont:.3f}")

# (b) combined heave/pitch means
for key in ("7_1", "0_0"):
    r = pickle.load(open(os.path.join(HERE, f"decomp_{key}.pkl"), "rb"))
    ok = r["ok"]
    X = (r["Dhull"] + r["Dwave"] - r["Wlo"])[ok]
    R = (r["Dhull"] + r["Djet"] + r["Dwave"] - r["Wlo"] + r["Dprime"])[ok]
    print(key, "Dhull+Dwave-Wlo mean w,q", X[:, 3:5].mean(0), "rms", np.sqrt((X[:, 3:5] ** 2).mean(0)),
          "| R mean", R.mean(0), "R rms", np.sqrt((R ** 2).mean(0)))

# (c) W2 critical product
kc = _flutter_kcrit(2 * p["z_heave"] * p["wn_heave"], p["wn_heave"] ** 2,
                    2 * p["z_pitch"] * p["wn_pitch"], p["wn_pitch"] ** 2)
print("W2 kappa*^2", kc, " max product 2.5 kc", 2.5 * kc)

# (f) W9 per unit slope with B1 (eps=-0.1, aA in {0,1}, base m in {1260, 1561.9})
for mb in (1260.0, 1561.9):
    for aA in (0.0, 1.0):
        eps = -0.1
        M00 = mb * (1 + eps) + (p["m_surge"] - mb) * (1 + aA * eps)
        print(f"W9 min accel/slope mb {mb} aA {aA}: {0.4 * 1400 * (1 + eps) * 9.81 / M00:.3f}  (no B1 {0.4 * 1400 * 9.81 / p['m_surge']:.3f})")

# (d) free-surge calm runs with the jet thrust held at the calm equilibrium value
from sim.planing_vessel_gz import PlaningVesselGZ
from sim.test_vessel import Monochromatic
with contextlib.redirect_stdout(io.StringIO()):
    pl = PlaningVesselGZ(c["h"], Monochromatic(1.0, 0.0), dt=0.02)
KN = 0.514444
for kn in (30, 16):
    t0 = time.time()
    u = kn * KN
    z, th = pl.running_attitude(u)
    R = pl.calm_resistance(u)["R"]
    Teq = brentq(lambda T: pl.prop.forces(T, 0.0, u)[0] - R, 1.0, 3 * pl.prop.t_max)
    pl.prop.prime = 1.0
    s = np.zeros(14); s[6] = u; s[2] = z; s[4] = th; s[12] = Teq
    G0 = pl.deriv(s, 0.0)[0]
    traj = []
    dt = 0.005
    for i in range(int(25 / dt)):
        k1 = pl.deriv(s, 0.0)[0]; s2 = s + dt * k1; k2 = pl.deriv(s2, 0.0)[0]
        s = s + 0.5 * dt * (k1 + k2); s[12] = Teq; s[13] = 0.0; pl.prop.prime = 1.0
        traj.append((s[6], s[2], s[4]))
    traj = np.array(traj); last = traj[-int(8 / dt):]
    Gend = pl.deriv(s, 0.0)[0]
    print(f"{kn} kn free surge, T held {Teq:.0f} N: start (u,z,th)=({u:.3f},{z:.4f},{th:.4f}) G0 udot,wdot,qdot "
          f"{G0[6]:.3f},{G0[8]:.3f},{G0[10]:.3f}; last 8 s u [{last[:, 0].min():.3f},{last[:, 0].max():.3f}] "
          f"z [{last[:, 1].min():.4f},{last[:, 1].max():.4f}] th [{last[:, 2].min():.4f},{last[:, 2].max():.4f}]; "
          f"G end {Gend[6]:.3f},{Gend[8]:.3f},{Gend[10]:.3f}; {time.time() - t0:.0f} s")
