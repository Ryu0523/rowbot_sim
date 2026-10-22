"""Read-only checks for verify_gz_side.md (no repository file written)."""
import os, sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
ROOT = "C:/Users/Administrator/Documents/usv-rl-mpc"
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
np.set_printoptions(precision=4, suppress=True, linewidth=160)
from learn.repro.task import ctx
from sim import config
from sim.planing_vessel_gz import PlaningVesselGZ
from sim.test_vessel import Monochromatic
from sim.wavefield import SeaState

c = ctx(); p = c["p"]
def phi(w, z, t):
    return expm(np.array([[0, 1], [-w * w, -2 * z * w]]) * t)
D, h = 0.24, 0.04
for nm, w, z in (("heave", p["wn_heave"], p["z_heave"]), ("pitch", p["wn_pitch"], p["z_pitch"])):
    P = phi(w, z, D); Ph = phi(w, z, h)
    print(nm, "Phi12(D)/D", P[0, 1] / D, "Phi22(D)", P[1, 1], " (1/h)int Phi22 over h", Ph[0, 1] / h,
          " -Phi21(h)", -Ph[1, 0], " w^2 h", w * w * h)
tr = p["tau_r"]
print("yaw", tr * (1 - np.exp(-D / tr)) / D, np.exp(-D / tr), " h:", tr * (1 - np.exp(-h / tr)) / h)
a = p["k_lin_sway"] / p["m_sway"]
print("sway a", a, "exact const-defect", (1 - np.exp(-a * D)) / (a * D), "1-aD/2", 1 - a * D / 2)
print("actuator decay exact", np.exp(-0.04 / 0.6), "gz Euler 2x0.02", (1 - 0.02 / 0.6) ** 2)

h_, db = c["h"], c["db"]
with contextlib.redirect_stdout(io.StringIO()):
    pl = PlaningVesselGZ(h_, Monochromatic(1.0, 0.0), dt=0.02)
print("t_max", pl.prop.t_max, "rud_max", pl.rudder.max, "Mtot diag", np.diag(pl.Mtot), "Iy", pl.Iy,
      "Z_rest", pl.Z_rest, "Th_rest", pl.Th_rest, "x0", pl.x[0], "x_n,z_n", pl.thrusters[0], "intake", pl.intakes[0],
      "prime_band", pl.prop.prime_band, "tau_back", pl.prop.tau_prime_back, "dead_band", pl.prop.dead_band,
      "rate_max", pl.prop.rate_max, "visc1", pl.visc[1], "kp max", pl.kp.max(), "ctr", pl._ctr(10.0))
KN = 0.514444
for kn in (16, 20, 25, 30):
    u = kn * KN
    z, th = pl.running_attitude(u)
    R = pl.calm_resistance(u)["R"]
    Teq = brentq(lambda T: pl.prop.forces(T, 0.0, u)[0] - R, 1.0, 3 * pl.prop.t_max)
    s = np.zeros(14); s[6] = u; s[2] = z; s[4] = th; s[12] = Teq
    ds, aux = pl.deriv(s, 0.0)
    d = aux["d"]; dp = np.maximum(d, 0)
    A, cc, ma, dma, girth = pl._section(d)
    m22 = 0.5 * np.pi * pl.rho * dp ** 2
    A33 = ma.sum() * pl.dx; A35 = (ma * -1 * pl.x).sum() * pl.dx; A55 = (ma * pl.x ** 2).sum() * pl.dx
    # heave-pitch Jacobian (z, w, th, q) by central differences
    idx = [2, 8, 4, 10]; J = np.zeros((4, 4))
    for j, k in enumerate(idx):
        e = 1e-5 if k in (2, 4) else 1e-4
        sp_, sm = s.copy(), s.copy(); sp_[k] += e; sm[k] -= e
        J[:, j] = (pl.deriv(sp_, 0.0)[0][idx] - pl.deriv(sm, 0.0)[0][idx]) / (2 * e)
    ev = np.linalg.eigvals(J)
    # reduced static defect at the running state (gz G = 0 there)
    Lu = (Teq - p["k_drag"] * u * u) / p["m_surge"]
    Lw = -p["wn_heave"] ** 2 * (z - p["z0"]); Lq = -p["wn_pitch"] ** 2 * (th - p["th0"])
    print(f"{kn} kn: z {z:.4f} th {th:.4f} d1 {d[0]:.3f} |u|m22(d1) {u * m22[0]:.0f} sum.5rhoCd {0.5 * pl.rho * pl.cd_lat * dp.sum() * pl.dx:.0f}"
          f" A33 {A33:.0f} A35 {A35:.0f} A55 {A55:.0f} Teq {Teq:.1f} R {R:.1f} kDu2 {p['k_drag'] * u * u:.1f}"
          f" | |G|max {np.abs(ds[6:12]).max():.2e} | reduced acc (u,w,q) {Lu:.3f} {Lw:.3f} {Lq:.3f}")
    print("    eig", np.round(ev, 2), " |w|", np.round(np.abs(ev), 2))
# closed form X_p = m (g + wd) Theta at a wavy, non-equilibrium state
sea = SeaState(1.0, 5.0, theta0=np.pi, n_freq=24, n_dir=5, seed=3)
with contextlib.redirect_stdout(io.StringIO()):
    pw = PlaningVesselGZ(h_, sea, dt=0.02)
rng = np.random.default_rng(0); mx = 0
for i in range(20):
    s = np.zeros(14); s[6] = rng.uniform(8, 15); s[2] = rng.uniform(0.05, 0.25); s[4] = rng.uniform(-0.06, 0.02)
    s[8] = rng.normal(0, 0.5); s[10] = rng.normal(0, 0.3); s[7] = rng.normal(0, .5); s[11] = rng.normal(0, .2)
    s[12] = rng.uniform(0, 5000); s[13] = rng.uniform(-.4, .4); s[0] = rng.uniform(0, 100)
    pw.prop.prime = rng.uniform(0, 1)
    ds, aux = pw.deriv(s, rng.uniform(0, 100))
    Th = pw.Th_rest + s[4]
    mx = max(mx, abs(aux["Xp"] - pw.m * (9.81 + aux["wd"]) * Th))
print("max |Xp - m(g+wd)Theta| over 20 random wavy states:", mx)
