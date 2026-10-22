"""Linearised gz rows at the calm running state vs the reduced model, added
masses and lateral coefficients at the task speeds, and two coupling terms on
the recorded gz states. Read-only."""
import os, sys, pickle, contextlib, io, warnings
warnings.filterwarnings("ignore")
ROOT = "C:/Users/Administrator/Documents/usv-rl-mpc"
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=160)
from scipy.optimize import brentq
from learn.repro.task import Mission
from learn.meta import relabel
env = relabel._env(); p = env["red"].p
with contextlib.redirect_stdout(io.StringIO()):
    pl = Mission("high", 1, 0, t_end=5.0, track=0.0).plant
pl._waves(np.zeros(1), np.zeros(1), 0.0); pl._wa = None      # calm water
pl.prop.prime = 1.0
KN = 0.514444
VEL = [6, 7, 11, 8, 10]
print("rows: u v r w q ; cols: u v r z w th q T_s(per kN) delta")
lo = np.zeros((5, 9))
lo[0, 0] = None
for kn in (16, 20, 25, 30):
    u = kn * KN
    R = pl.calm_resistance(u)["R"]
    Teq = brentq(lambda T: pl.prop.forces(T, 0.0, u)[0] - R, 1.0, 3 * pl.prop.t_max)
    z, th = pl.running_attitude(u)
    s = np.zeros(14); s[6] = u; s[2] = z; s[4] = th; s[12] = Teq
    f0 = pl.deriv(s, 0.0)[0]
    cols = [6, 7, 11, 2, 8, 4, 10, 12, 13]
    eps = [1e-3, 1e-3, 1e-4, 1e-4, 1e-3, 1e-5, 1e-4, 1.0, 1e-4]
    J = np.zeros((5, 9))
    for j, (c, e) in enumerate(zip(cols, eps)):
        sp = s.copy(); sp[c] += e; sm = s.copy(); sm[c] -= e
        J[:, j] = (pl.deriv(sp, 0.0)[0][VEL] - pl.deriv(sm, 0.0)[0][VEL]) / (2 * e)
    J[:, 7] *= 1000.0
    x = pl.x; Th = pl.Th_rest + th
    d = -(pl.Z_rest + z + pl.zk - x * Th)
    A, c_, ma, dma, girth = pl._section(d)
    A33 = ma.sum() * pl.dx; A35 = (-x * ma).sum() * pl.dx; A55 = (ma * x * x).sum() * pl.dx
    m22 = 0.5 * np.pi * pl.rho * np.maximum(d, 0) ** 2
    Smm = m22.sum() * pl.dx; Sxm = (x * m22).sum() * pl.dx; Sxxm = (x * x * m22).sum() * pl.dx
    Yv = -u * m22[0]; Yr = -u * m22[0] * x[0]
    Nv = -u * (x[0] * m22[0] + Smm); Nr = -u * (x[0] ** 2 * m22[0] + Sxm)
    kcf = (0.5 * pl.rho * pl.cd_lat * np.maximum(d, 0)).sum() * pl.dx
    print(f"\n{kn} kn: u={u:.3f} T_eq={Teq:.0f} z={z:.4f} th={th:.4f} Theta={Th:.4f} d_stern={d[0]:.4f} d_bow={d[-1]:.4f}")
    print("  gz Jacobian:\n", J)
    print(f"  lofi: du/du={-2*p['k_drag']*u/p['m_surge']:.4f} du/dT(per kN)={1000/p['m_surge']:.4f}; dv/dv={-p['k_lin_sway']/p['m_sway']:.4f} "
          f"dv/dr={-p['m_coriolis']*u/p['m_sway']:.4f}; dr/dr={-1/p['tau_r']:.4f}; dw/dz={-p['wn_heave']**2:.3f} dw/dw={-2*p['z_heave']*p['wn_heave']:.3f}; "
          f"dq/dth={-p['wn_pitch']**2:.3f} dq/dq={-2*p['z_pitch']*p['wn_pitch']:.3f}; dr/ddelta={p['k_nomoto_f']*p['k_jet_side']*Teq/p['tau_r']:.4f} dv/ddelta={p['k_jet_side']*Teq/p['m_sway']:.4f}")
    print(f"  A33={A33:.0f} A35={A35:.0f} A55={A55:.0f}  m+A33={pl.m + A33:.0f} Iy+A55={pl.Iy + A55:.0f}")
    print(f"  m22 stern={m22[0]:.1f} kg/m; sum m22 dx={Smm:.1f}; sum x m22 dx={Sxm:.1f}; sum x^2 m22 dx={Sxxm:.1f}")
    print(f"  linear coefficients: Y_v={Yv:.0f} (lofi -k_lin={-p['k_lin_sway']:.0f}) Y_r={Yr:.0f} N_v={Nv:.0f} N_r={Nr:.0f} N s/m or N m s;"
          f" per inertia: Y_v/M22={Yv/pl.Mtot[1,1]:.3f} N_v/M66={Nv/pl.Mtot[5,5]:.3f} N_r/M66={Nr/pl.Mtot[5,5]:.3f}")
    print(f"  crossflow quad coeff sum 0.5 rho C d dx={kcf:.1f} (lofi k_sway {p['k_sway']:.1f}); -2*Theta*u*A33/M11 = {-2*Th*u*A33/pl.Mtot[0,0]:.3f} per rad/s of q")
    print(f"  calm static defect (lofi accel at gz running state): {np.round([(p['k_drag']*u*u - Teq)/p['m_surge'], 0, 0, p['wn_heave']**2*(z-p['z0']), p['wn_pitch']**2*(th-p['th0'])], 4)}")

# coupling terms on recorded gz states
S = pickle.load(open("studies/_cache/onestep_gz/streams_T120.pkl", "rb"))
xs = np.concatenate([S[k]["xs"][:-1] for k in S])
u, v, w, q, r, th = xs[:, 6], xs[:, 7], xs[:, 8], xs[:, 10], xs[:, 11], xs[:, 4]
for m in (1260.0, 1400.0, 1561.9):
    fam = (m / p["m_surge"]) * q * (w + u * th)
    print(f"family layer-3 Coriolis surge part (m/m_u) q (zdot + u th), m={m:.0f}: rms {np.sqrt((fam**2).mean()):.4f} m/s^2")
print("gz m v r / M11 rms", np.sqrt(((1400 * v * r / 1470) ** 2).mean()), "; v r rms", np.sqrt(((v * r) ** 2).mean()))
print("gz state rms: u", u.std(), "v", np.sqrt((v**2).mean()), "r", np.sqrt((r**2).mean()), "w", np.sqrt((w**2).mean()), "q", np.sqrt((q**2).mean()), "theta", th.mean(), th.std())
