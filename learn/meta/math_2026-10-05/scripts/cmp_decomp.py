"""Exact decomposition of the gz one-step error e0 (velocity channels) into
named defects evaluated at the gz Heun stage points, on open-loop replays of
recorded missions that reproduce the stored states.

e0_c = <G_c - L_c(., U_k)>_gz + Pi_c,  Pi_c = <L_c(., U_k)>_gz - (M_c(a,U_k) - a_c)/Delta
G = G_calm0 + G_jet + G_wave + G_prime  (gz rows)
  G_calm0 : deriv, no waves, thrust state 0 (no jet force)
  G_jet   : deriv no waves, prime 1, actual T_s, delta_s  minus G_calm0
  G_wave  : deriv with waves, prime 1  minus  no waves, prime 1
  G_prime : deriv with waves, actual prime  minus  with waves, prime 1
L(s,U) = L_calm(s) + L_act(T,delta)   (model0 continuous right-hand side)
Read-only; no repository file is written."""
import sys, os, pickle, contextlib, io, warnings, time
warnings.filterwarnings("ignore")
ROOT = "C:/Users/Administrator/Documents/usv-rl-mpc"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=160)
from studies import final_eval as FE
from studies.sim2real_jet import theta, z_hand
from learn.meta import relabel, data3

env = relabel._env()
p = env["red"].p
x_st = np.asarray(env["x_st"], float)
xc = x_st - x_st.mean(); den = (xc ** 2).sum()
VEL = [6, 7, 11, 8, 10]      # u v r w q in the 14-vector
names = ["u", "v", "r", "w", "q"]
DT = 0.24


def L_calm(s):
    u, v, w, q, r, z, th = s[6], s[7], s[8], s[10], s[11], s[2], s[4]
    return np.array([
        -p["k_drag"] * u * abs(u) / p["m_surge"],
        (-p["k_lin_sway"] * v - p["k_sway"] * v * abs(v) - p["m_coriolis"] * u * r) / p["m_sway"],
        -r / p["tau_r"],
        -p["wn_heave"] ** 2 * (z - p["z0"]) - 2 * p["z_heave"] * p["wn_heave"] * w,
        -p["wn_pitch"] ** 2 * (th - p["th0"]) - 2 * p["z_pitch"] * p["wn_pitch"] * q])


def L_act(T, d):
    lift = p["k_jet_side"] * max(T, 0.0) * np.sin(np.clip(d, -p["rud_stall"], p["rud_stall"]))
    return np.array([T / p["m_surge"], lift / p["m_sway"], p["k_nomoto_f"] * lift / p["tau_r"], 0.0, 0.0])


def W_lo(pl, s, t):
    c, sn = np.cos(s[5]), np.sin(s[5])
    e = pl._waves(s[0] + x_st * c, s[1] + x_st * sn, t)[0]
    sig = (e * xc).sum() / den
    return np.array([0, 0, 0, p["wn_heave"] ** 2 * e.mean(),
                     p["wn_pitch"] ** 2 * p["sign_pitch"] * np.arctan(sig)])


def FK(pl, s, t):
    """Froude-Krylov slope forces as accelerations (surge, sway, yaw)."""
    x = pl.x
    cps, sps = np.cos(s[5]), np.sin(s[5])
    e, et, ett, ex, ey = pl._waves(s[0] + x * cps, s[1] + x * sps, t)
    s_lon = ex * cps + ey * sps; s_tr = -ex * sps + ey * cps
    Th = pl.Th_rest + s[4]
    d = e - (pl.Z_rest + s[2] + pl.zk - x * Th)
    A = pl._section(d)[0]
    ab = pl._a_b(s[6])
    Xw = -(pl.rho * 9.81 * ab * A * s_lon).sum() * pl.dx
    Ys = -pl.rho * 9.81 * ab * A * s_tr
    return np.array([Xw / pl.Mtot[0, 0], Ys.sum() * pl.dx / pl.Mtot[1, 1],
                     (x * Ys).sum() * pl.dx / pl.Mtot[5, 5]])


def variants(pl, s, t, P):
    wa = pl._wa
    pl.prop.prime = P; Gf = pl.deriv(s, t)[0][VEL]
    pl.prop.prime = 1.0; G1 = pl.deriv(s, t)[0][VEL]
    pl._wa = None
    Gc1 = pl.deriv(s, t)[0][VEL]
    s0 = s.copy(); s0[12] = 0.0
    Gc0 = pl.deriv(s0, t)[0][VEL]
    pl._wa = wa; pl.prop.prime = P
    return Gf, G1, Gc1, Gc0


S = pickle.load(open("studies/_cache/onestep_gz/streams_T120.pkl", "rb"))
wts, floor = theta(z_hand())
keys = [tuple(map(int, a.split("_"))) for a in sys.argv[1:]] or [(7, 1), (3, 1)]
ALL = {}
for key in keys:
    t0 = time.time()
    st = S[key]
    with contextlib.redirect_stdout(io.StringIO()):
        m = FE.make_mission(key[0], key[1], 120.0, weights=wts, floor=floor)
    pl = m.plant
    pl._waves(np.zeros(1), np.zeros(1), 0.0)       # build _wa
    s = m.s.copy(); t = 0.0
    nk = st["n"]
    rec = {k: np.zeros((nk, 5)) for k in ("G", "Dhull", "Djet", "Dact", "Dwave", "Wlo", "Dprime", "Lgz",
                                            "cor", "FK", "inert")}
    Pmin = np.ones(nk); nsub_bad = np.zeros(nk, bool); maxdev = 0.0
    for k in range(nk):
        thr = st["u"][k, 0] * m.t_max; rud = st["u"][k, 1] * m.rud_max
        Tc, dc = thr, rud
        acc = {kk: np.zeros(5) for kk in rec}
        for i in range(m.sub):
            sn, tn = s.copy(), t
            s = pl.step(s, t, thr, rud, m.dt); t += m.dt
            P = pl.prop.prime
            Pmin[k] = min(Pmin[k], P)
            if pl.last_n_sub > 1:
                nsub_bad[k] = True
            pl.prop.prime = P
            k1full = pl.deriv(sn, tn)[0]
            s2 = sn + m.dt * k1full
            for (ss, tt) in ((sn, tn), (s2, tn + m.dt)):
                Gf, G1, Gc1, Gc0 = variants(pl, ss, tt, P)
                Lc = L_calm(ss); La_s = L_act(ss[12], ss[13]); La_c = L_act(Tc, dc)
                acc["G"] += Gf / 24; acc["Dhull"] += (Gc0 - Lc) / 24
                acc["Djet"] += (Gc1 - Gc0 - La_s) / 24; acc["Dact"] += (La_s - La_c) / 24
                acc["Dwave"] += (G1 - Gc1) / 24; acc["Dprime"] += (Gf - G1) / 24
                acc["Wlo"] += W_lo(pl, ss, tt) / 24; acc["Lgz"] += (Lc + La_c) / 24
                acc["cor"][0] += pl.m * ss[7] * ss[11] / pl.Mtot[0, 0] / 24
                acc["FK"][:3] += FK(pl, ss, tt) / 24
                acc["inert"][0] += (1 - pl.Mtot[0, 0] / p["m_surge"]) * Gf[0] / 24
            pl.prop.prime = P
        maxdev = max(maxdev, np.abs(s - st["xs"][k + 1]).max())
        for kk in rec:
            rec[kk][k] = acc[kk]
    xs = st["xs"][: nk + 1]
    e0 = data3.e0_from(env, xs[:-1], xs[1:], st["u"][:nk])[:, :5]
    a = data3.plant_to_reduced(xs[:-1]); mm = data3.model0_step(env, a, st["u"][:nk])
    RV = [2, 9, 8, 4, 6]
    Lm0 = (mm[:, RV] - a[:, RV]) / DT
    Pi = rec["Lgz"] - Lm0
    dnu = (xs[1:, VEL] - xs[:-1, VEL]) / DT
    ok = ~nsub_bad
    resid = e0 - (rec["Dhull"] + rec["Djet"] + rec["Dact"] + rec["Dwave"] + rec["Dprime"] + Pi)
    print(f"{key}: replay max state dev {maxdev:.2e}; n_sub>1 steps {nsub_bad.sum()}; "
          f"min prime<0.999 share {np.mean(Pmin < 0.999):.3f}; {time.time() - t0:.0f} s")
    print("   check <G>_Heun - dnu/Delta max", np.abs(rec["G"][ok] - dnu[ok]).max(0))
    print("   identity residual max", np.abs(resid[ok]).max(0))
    rec["e0"] = e0; rec["Pi"] = Pi; rec["ok"] = ok; rec["Pmin"] = Pmin
    ALL[key] = rec
    pickle.dump(rec, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "decomp_%d_%d.pkl" % key), "wb"))

ok = np.concatenate([ALL[k]["ok"] for k in ALL])
def cat(n):
    return np.concatenate([ALL[k][n] for k in ALL])[ok]
e0 = cat("e0")
print("\nrows (velocity channels u v r w q), pooled", ok.sum(), "steps")
print("%-26s" % "term", "   rms:  " + "  ".join("%7s" % n for n in names), "   mean: " + "  ".join("%7s" % n for n in names))
for n in ("e0", "Dhull", "Djet", "Dact", "Dwave", "Wlo", "Dprime", "Pi", "cor", "FK", "inert"):
    X = e0 if n == "e0" else cat(n)
    print("%-26s" % n, "        " + "  ".join("%7.3f" % v for v in np.sqrt((X ** 2).mean(0))),
          "         " + "  ".join("%7.3f" % v for v in X.mean(0)))
X = cat("Dwave") - cat("Wlo")
print("%-26s" % "Dwave - Wlo", "        " + "  ".join("%7.3f" % v for v in np.sqrt((X ** 2).mean(0))),
      "         " + "  ".join("%7.3f" % v for v in X.mean(0)))
X = cat("Dhull") + cat("Dwave")
print("%-26s" % "Dhull + Dwave", "        " + "  ".join("%7.3f" % v for v in np.sqrt((X ** 2).mean(0))), "         " + "  ".join("%7.3f" % v for v in X.mean(0)))
X = cat("Dhull") + cat("Dwave") - cat("Wlo")
print("%-26s" % "Dhull + Dwave - Wlo", "        " + "  ".join("%7.3f" % v for v in np.sqrt((X ** 2).mean(0))), "         " + "  ".join("%7.3f" % v for v in X.mean(0)))
X = X - X.mean(0)
print("%-26s" % "  same, mean removed", "        " + "  ".join("%7.3f" % v for v in np.sqrt((X ** 2).mean(0))))
X = cat("Dact") + cat("Djet")
print("%-26s" % "Dact + Djet", "        " + "  ".join("%7.3f" % v for v in np.sqrt((X ** 2).mean(0))))
X = cat("Dhull") + cat("Djet")
print("%-26s" % "Dhull + Djet", "        " + "  ".join("%7.3f" % v for v in np.sqrt((X ** 2).mean(0))))
# variance shares: regress e0 on each defect sum? report corr
for n in ("Dhull", "Djet", "Dact", "Dwave", "Dprime", "Pi"):
    X = cat(n)
    c = [np.corrcoef(X[:, j], e0[:, j])[0, 1] if X[:, j].std() > 1e-12 else np.nan for j in range(5)]
    print("corr(e0, %-7s)" % n, np.round(c, 3))
Pm = cat("Pmin"); Dp = cat("Dprime")
print("steps with min prime < 0.999:", np.mean(Pm < 0.999), " < 0.5:", np.mean(Pm < 0.5))
print("rms Dprime on steps with prime<0.5:", np.sqrt((Dp[Pm < 0.5] ** 2).mean(0)))

# propagation gains of model0 over Delta for a constant defect (and an impulse at start / end)
red = env["red"]
for nm, w_, z_ in (("heave", p["wn_heave"], p["z_heave"]), ("pitch", p["wn_pitch"], p["z_pitch"])):
    Ph = red._phi(w_, z_, DT)
    print(f"{nm}: constant-defect gain Phi12(D)/D = {Ph[0, 1] / DT:.3f}; impulse at step start gain Phi22(D) = {Ph[1, 1]:.3f}; "
          f"position channel gain for constant defect Phi12-int: see below")
    # position channel: e0_z = (1/D) int_0^D Phi12(D-tau) dtau * d
    tt = np.linspace(0, DT, 2001); vals = [red._phi(w_, z_, max(DT - x, 1e-12))[0, 1] if DT - x > 1e-12 else 0.0 for x in tt]
    print(f"   {nm} position channel gain (1/D) int Phi12 = {np.trapezoid(vals, tt) / DT:.4f} s")
tr = p["tau_r"]
print(f"yaw: constant-defect gain tau(1-exp(-D/tau))/D = {tr * (1 - np.exp(-DT / tr)) / DT:.3f}; start impulse {np.exp(-DT / tr):.3f}")
print(f"sway linear damping rate k_lin/m_v = {p['k_lin_sway'] / p['m_sway']:.3f} 1/s -> gain ~ {1 - 0.5 * DT * p['k_lin_sway'] / p['m_sway']:.3f}")
print(f"surge 2k u/m_u at 11.3 m/s = {2 * p['k_drag'] * 11.3 / p['m_surge']:.3f} 1/s -> gain ~ {1 - 0.5 * DT * 2 * p['k_drag'] * 11.3 / p['m_surge']:.3f}")
print("gz Mtot diag", np.diag(pl.Mtot)[[0, 1, 5]], " lofi m_surge m_sway", p["m_surge"], p["m_sway"], "m_cor", p["m_coriolis"], "gz m", pl.m)
