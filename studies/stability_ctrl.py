#!/usr/bin/env python3
"""
Reduced-model comparison and closed-loop analysis for the stability map
(studies/stability_map.py; conventions in studies/stability_lin.py).

REDUCED MODEL. control/reduced.ReducedModel as identified for the frozen
plant (app/models.reduced_params: the cached identification whose cache
tag is the plant's physics sha1, else identified now). Its structure:
surge m_surge u' = T - k_drag u|u| (one real pole -2 k_drag u / m_surge),
heave and pitch two independent oscillators with speed-independent
(wn, zeta) identified at the thrust step's steady speed u_ss, sway
m_sway v' = side - k_lin v - m_cor u r (one real pole, no yaw feedback),
yaw first order towards K delta with time constant tau_r (Nomoto),
attitude offsets z0, th0 fixed at the design speed. The plant's
counterpart at each speed is read off the continuous Jacobian A_c:
  * the 'diagonal' heave oscillator  wn_h = sqrt(-A[w, z]),
    zeta_h = -A[w, w] / (2 wn_h)  (what an independent oscillator would be
    if the coupling were dropped), same for pitch from A[q, th], A[q, q];
  * the coupled modes of the vertical block, each labelled by the state
    with the largest participation factor (|right_i left_i|);
  * surge: -1/A[u, u] at fixed attitude, and the slowest real mode of the
    vertical block (surge with the attitude following);
  * sway/yaw: the (v, r) block, its DC gain r/delta (the Nomoto gain the
    plant actually has at this speed and thrust) and its eigenvalues.

CLOSED LOOP. The controllers the Gym tests run (app/env.py 'heading' mode
through app/models.Autopilot = sim/env.Episode._steer; the throttle hold
of studies/app_randomization_test._run), at the control step 0.25 s with
25 plant steps of 0.01 s, the plant's command delay 0.1 s = 10 plant
steps. Lifted linear model over one control step with the zero-order
hold: the command c_k issued at the start of control step k reaches the
actuator lag at plant step 10 of that control step, so

    s_{k+1} = Phi^25 s_k + (sum_{i=0}^{9} Phi^{24-i}) B c_{k-1}
                         + (sum_{i=10}^{24} Phi^{24-i}) B c_k

with Phi, B the one-step Jacobians (B: the actuator lag's dt/tau rows).
Heading autopilot (Episode._steer, planing branch, zeta = 1):
    wn = 0.30 (u_design / L) / 0.45,  K = k_nomoto_f k_jet_side max(T, 0.15 t_max),
    Tn = max(tau_r, 0.5 rs),  a = wn^2 Tn / K,  b = max(2 wn Tn - 1, 0) / K,
    delta_cmd = -(a (psi - psi_ref) + b r + (0.12 / rs) a I_psi),
    I_psi += (psi - psi_ref) dt_ctrl   (conditional: only while |raw| < limit)
Throttle hold (_run, k_p 0.15, k_i 0.02 fractions of t_max per m/s):
    e = u_ref - u,  I_u += e (clipped +-2),  c = c* + k_p e + k_i I_u.
Both linearised about the equilibrium (the clips and the conditional
integration are inactive for small deviations). Closed-loop eigenvalues of
the lifted system are converted with log(z) / dt_ctrl; the margins are the
gain factor on each loop's gains at which an eigenvalue reaches the unit
circle (bisection) and the Nyquist phase margin of the loop transfer
function broken at the command.
"""
import math
import warnings

import numpy as np

from studies import stability_lin as L

KN = L.KN
N_SUB, N_DELAY, DT_CTRL = 25, 10, 0.25
K_P, K_I = 0.15, 0.02           # studies/app_randomization_test._run


# ---------------------------------------------------------------- reduced
def reduced_params():
    from hydro import hulls
    from app.models import reduced_params as rp
    return rp(hulls.get(L.HULL))


def participation(A):
    """Participation factors |v_ki w_ik| (rows: states, cols: modes),
    normalised per mode to sum 1, and the eigenvalues."""
    lam, V = np.linalg.eig(A)
    W = np.linalg.inv(V)          # rows are left eigenvectors
    P = np.abs(V * W.T)
    P = P / np.maximum(P.sum(0, keepdims=True), 1e-300)
    return lam, P


def label_modes(A, states, groups):
    """Continuous eigenvalues of A with a label from `groups`
    ({label: [state names]}): the group with the largest summed
    participation. Complex pairs reported once."""
    lam, P = participation(A)
    out = []
    for k, l in enumerate(lam):
        if l.imag < -1e-9:
            continue
        part = {g: float(sum(P[states.index(n), k] for n in names
                             if n in states))
                for g, names in groups.items()}
        g = max(part, key=part.get)
        m = L.modes([l])[0]
        m.update(label=g, participation=part)
        out.append(m)
    out.sort(key=lambda m: m["wn"])
    return out


VGROUPS = dict(surge=["u"], heave=["z", "w"], pitch=["th", "q"],
               thrust=["T"])
LGROUPS = dict(roll=["phi", "p"], sway=["v"], yaw=["r"], nozzle=["d"],
               heading=["psi"])


def reduced_comparison(m):
    p = reduced_params()
    st = m["states"]
    rows = []
    for r in m["rows"]:
        if "error" in r:
            continue
        u = r["u"]
        A = np.array(r["A"])
        i = {n: st.index(n) for n in st}
        Av = L.block(A, tuple(st), L.VERT)
        Al = L.block(A, tuple(st), L.LAT)
        vm = label_modes(Av, list(L.VERT), VGROUPS)
        lm = label_modes(Al, list(L.LAT), LGROUPS)
        # diagonal oscillators
        kz, kw = A[i["w"], i["z"]], A[i["w"], i["w"]]
        kt, kq = A[i["q"], i["th"]], A[i["q"], i["q"]]
        wn_h = math.sqrt(max(-kz, 0.0))
        wn_p = math.sqrt(max(-kt, 0.0))
        zeta_h = -kw / (2 * wn_h) if wn_h > 0 else float("nan")
        zeta_p = -kq / (2 * wn_p) if wn_p > 0 else float("nan")
        # surge
        a_uu = A[i["u"], i["u"]]
        surge = [x for x in vm if x["label"] == "surge" and x["kind"] == "real"]
        tau_surge = surge[0]["tau"] if surge else float("nan")
        # sway / yaw: DC gain of r per rad of nozzle angle (state)
        iv, ir, idl = i["v"], i["r"], i["d"]
        A2 = A[np.ix_([iv, ir], [iv, ir])]
        b2 = A[[iv, ir], idl]
        try:
            x = -np.linalg.solve(A2, b2)
            K_plant = float(x[1])
        except np.linalg.LinAlgError:
            K_plant = float("nan")
        lam2 = np.linalg.eigvals(A2)
        T_hold = r["eq"]["T"]
        K_red = p["k_nomoto_f"] * p["k_jet_side"] * T_hold
        rows.append(dict(
            kn=r["kn"], u=u,
            plant_z=r["eq"]["z"], plant_trim=r["eq"]["trim_deg"],
            reduced_z0=p["z0"], reduced_th0=p["th0"],
            plant_diag=dict(wn_heave=wn_h, z_heave=zeta_h, wn_pitch=wn_p,
                            z_pitch=zeta_p),
            plant_modes_vertical=vm, plant_modes_lateral=lm,
            plant_pairs=[(round(x["wn"], 3), round(x["zeta"], 3), x["label"])
                         for x in vm if x["kind"] == "pair"],
            plant_surge_tau=tau_surge,
            plant_surge_tau_fixed_attitude=(-1.0 / a_uu if a_uu < 0
                                            else float("inf")),
            reduced_surge_tau=(p["m_surge"] / (2 * p["k_drag"] * u)
                               if u > 0 else float("inf")),
            reduced=dict(wn_heave=p["wn_heave"], z_heave=p["z_heave"],
                         wn_pitch=p["wn_pitch"], z_pitch=p["z_pitch"],
                         tau_r=p["tau_r"], sway_pole=-p["k_lin_sway"]
                         / p["m_sway"], u_ss=p["u_ss"]),
            plant_sway_yaw_A=A2.tolist(),
            plant_sway_yaw_eig=[[float(l.real), float(l.imag)] for l in lam2],
            reduced_sway_yaw_A=[[-p["k_lin_sway"] / p["m_sway"],
                                 -p["m_coriolis"] * u / p["m_sway"]],
                                [0.0, -1.0 / p["tau_r"]]],
            plant_K_nomoto=K_plant, reduced_K_nomoto=float(K_red),
            plant_A_vu=A[i["w"], i["u"]], plant_A_qu=A[i["q"], i["u"]],
            plant_A_uz=A[i["u"], i["z"]], plant_A_uth=A[i["u"], i["th"]],
            coupling_vert=L.coupling(Av, L.VERT, ("u",), ("z", "th", "w",
                                                          "q"))))
    return dict(params=p, rows=rows,
                note="plant_diag: the independent oscillator the reduced "
                     "model assumes, read off the plant's own Jacobian; "
                     "plant_modes: the coupled modes with participation "
                     "labels")


# ------------------------------------------------------------ closed loop
def lifted(Phi, B):
    """(Phi_N, Gamma_prev, Gamma_now) over one control step."""
    n = Phi.shape[0]
    PhiN = np.linalg.matrix_power(Phi, N_SUB)
    Gp = np.zeros((n, 2))
    Gn = np.zeros((n, 2))
    Pk = np.eye(n)                      # Phi^(24 - i) built from i = 24 down
    for i in range(N_SUB - 1, -1, -1):
        if i < N_DELAY:
            Gp += Pk @ B
        else:
            Gn += Pk @ B
        Pk = Pk @ Phi
    return PhiN, Gp, Gn


def autopilot_gains(p, T_hold, Lhull):
    rs = math.sqrt(Lhull / 10.0)
    u_ref = float(p["u_design"])
    wn = 0.30 * (u_ref / Lhull) / 0.45
    t_now = max(float(T_hold), 0.15 * p["t_max"])
    K = p["k_nomoto_f"] * p["k_jet_side"] * t_now
    Tn = max(p["tau_r"], 0.5 * rs)
    a = wn ** 2 * Tn / K
    b = max(2 * wn * Tn - 1.0, 0.0) / K
    kI = 0.12 / rs
    return dict(wn=wn, K=K, Tn=Tn, a=a, b=b, kI=kI, rs=rs, t_now=t_now)


def closed_matrices(Phi, B, states, p, T_hold, t_max, Lhull, kappa_u=1.0,
                    kappa_psi=1.0, loops=("speed", "heading")):
    """The closed-loop lifted matrix M over [s, c_prev(2), I_u, I_psi]."""
    n = Phi.shape[0]
    PhiN, Gp, Gn = lifted(Phi, B)
    i = {k: states.index(k) for k in states}
    g = autopilot_gains(p, T_hold, Lhull)
    Ks = np.zeros((2, n))
    Ki = np.zeros((2, 2))
    E = np.zeros((2, n))
    if "speed" in loops:
        Ks[0, i["u"]] = -kappa_u * t_max * (K_P + K_I)
        Ki[0, 0] = kappa_u * t_max * K_I
        E[0, i["u"]] = -1.0
    if "heading" in loops:
        Ks[1, i["psi"]] = -kappa_psi * g["a"]
        Ks[1, i["r"]] = -kappa_psi * g["b"]
        Ki[1, 1] = -kappa_psi * g["kI"] * g["a"]
        E[1, i["psi"]] = DT_CTRL
    M = np.zeros((n + 4, n + 4))
    M[:n, :n] = PhiN + Gn @ Ks
    M[:n, n:n + 2] = Gp
    M[:n, n + 2:] = Gn @ Ki
    M[n:n + 2, :n] = Ks
    M[n:n + 2, n + 2:] = Ki
    M[n + 2:, :n] = E
    M[n + 2:, n + 2:] = np.eye(2)
    return M, g


def gain_margin_factor(fn, lo=1.0, hi=1e4):
    """Smallest kappa >= 1 at which fn(kappa) (max |eig|) reaches 1, by
    bracketing and bisection; inf if none up to hi."""
    if fn(lo) >= 1.0:
        return 0.0
    k = lo
    while k < hi:
        k2 = k * 1.5
        if fn(k2) >= 1.0:
            a, b = k, k2
            for _ in range(60):
                m = 0.5 * (a + b)
                if fn(m) >= 1.0:
                    b = m
                else:
                    a = m
            return float(0.5 * (a + b))
        k = k2
    return float("inf")


def loop_response(Phi, B, states, outs, inp, zs):
    """G(z) of the lifted plant (state [s, c_prev]) from command `inp`
    (0 thrust, 1 nozzle) to the outputs `outs` (state names), at points
    zs on the unit circle: array (len(zs), len(outs))."""
    n = Phi.shape[0]
    PhiN, Gp, Gn = lifted(Phi, B)
    Al = np.zeros((n + 2, n + 2))
    Al[:n, :n] = PhiN
    Al[:n, n:] = Gp
    Bl = np.zeros((n + 2, 2))
    Bl[:n] = Gn
    Bl[n:] = np.eye(2)
    C = np.zeros((len(outs), n + 2))
    for k, o in enumerate(outs):
        C[k, states.index(o)] = 1.0
    I = np.eye(n + 2)
    out = np.empty((len(zs), len(outs)), complex)
    for k, z in enumerate(zs):
        out[k] = C @ np.linalg.solve(z * I - Al, Bl[:, inp])
    return out


def margins_from_L(Lz, w):
    """Classical margins of a negative-feedback loop from L(e^{jw dt}):
    phase margin at the first gain crossover, gain margin (dB) at the
    first phase crossover (-180 deg)."""
    mag = np.abs(Lz)
    ph = np.degrees(np.unwrap(np.angle(Lz)))
    pm = gm = None
    wc = wp = None
    for k in range(len(w) - 1):
        if (mag[k] - 1.0) * (mag[k + 1] - 1.0) <= 0 and pm is None:
            f = (1.0 - mag[k]) / (mag[k + 1] - mag[k] + 1e-300)
            pm = 180.0 + (ph[k] + f * (ph[k + 1] - ph[k]))
            pm = (pm + 180.0) % 360.0 - 180.0
            wc = w[k] + f * (w[k + 1] - w[k])
        if (ph[k] + 180.0) * (ph[k + 1] + 180.0) <= 0 and gm is None:
            f = (-180.0 - ph[k]) / (ph[k + 1] - ph[k] + 1e-300)
            mg = mag[k] + f * (mag[k + 1] - mag[k])
            gm = -20.0 * math.log10(max(mg, 1e-300))
            wp = w[k] + f * (w[k + 1] - w[k])
    return dict(phase_margin_deg=(None if pm is None else float(pm)),
                gain_margin_db=(None if gm is None else float(gm)),
                w_gain_cross=(None if wc is None else float(wc)),
                w_phase_cross=(None if wp is None else float(wp)),
                dc_gain=float(mag[0]))


def closed_loop(m, p):
    st = list(m["states"])
    rows = []
    Lhull = float(p["L"])
    t_max = float(p["t_max"])
    w = np.linspace(1e-3, math.pi / DT_CTRL * 0.999, 4000)
    zs = np.exp(1j * w * DT_CTRL)
    for r in m["rows"]:
        if "error" in r:
            continue
        Phi = np.array(r["Phi"]["0.0001"])
        B = np.array(r["B"]["0.0001"])
        T_hold = r["eq"]["T"]
        M, g = closed_matrices(Phi, B, st, p, T_hold, t_max, Lhull)
        lam_d = np.linalg.eigvals(M)
        lam = L.to_continuous(lam_d, DT_CTRL)
        aug = st + ["cT_prev", "cd_prev", "I_u", "I_psi"]
        # label by participation of the discrete map
        _, P = participation(M)
        groups = dict(speed=["u", "T", "cT_prev", "I_u"],
                      heading=["psi", "r", "v", "d", "cd_prev", "I_psi"],
                      vertical=["z", "th", "w", "q"], roll=["phi", "p"])
        modes = []
        for k, l in enumerate(lam):
            if l.imag < -1e-9:
                continue
            part = {gname: float(sum(P[aug.index(n), k] for n in names))
                    for gname, names in groups.items()}
            md = L.modes([l])[0]
            md.update(label=max(part, key=part.get), zd=abs(lam_d[k]))
            modes.append(md)
        modes.sort(key=lambda x: -x["re"])
        res = dict(kn=r["kn"], gains=g, max_abs_z=float(np.abs(lam_d).max()),
                   stable=bool(np.abs(lam_d).max() < 1.0), modes=modes)
        # open loop (no controller) for reference
        M0, _ = closed_matrices(Phi, B, st, p, T_hold, t_max, Lhull, loops=())
        lam0 = L.to_continuous(np.linalg.eigvals(M0), DT_CTRL)
        res["open_loop_max_re"] = float(np.nanmax(np.real(
            lam0[np.isfinite(lam0)])))
        loop_states = dict(speed=["u", "T", "cT_prev", "I_u"],
                           heading=["psi", "r", "v", "d", "cd_prev", "I_psi"])
        for loop, ku in (("speed", "kappa_u"), ("heading", "kappa_psi")):
            def rho(kap, loop=loop, ku=ku):
                """max |z| over the closed-loop modes that belong to this
                loop (participation of its states >= 0.3), with its gains
                scaled by kap and the other loop open -- so an open-loop
                lateral instability does not count against the speed loop
                and vice versa."""
                kw = {ku: kap}
                Mk, _ = closed_matrices(Phi, B, st, p, T_hold, t_max, Lhull,
                                        loops=(loop,), **kw)
                lam_k, Pk = participation(Mk)
                keep = [k for k in range(len(lam_k))
                        if sum(Pk[aug.index(n), k] for n in loop_states[loop])
                        >= 0.3]
                ev = np.abs(lam_k[keep]) if keep else np.zeros(0)
                return float(ev.max()) if ev.size else 0.0
            Mk, _ = closed_matrices(Phi, B, st, p, T_hold, t_max, Lhull,
                                    loops=(loop,))
            lk = L.to_continuous(np.linalg.eigvals(Mk), DT_CTRL)
            lk = lk[np.isfinite(lk)]
            own = [x for x in modes if x["label"] == loop]
            slowest = max(own, key=lambda x: x["re"]) if own else None
            gm_fac = gain_margin_factor(rho)
            # Nyquist margins of the loop broken at the command
            if loop == "speed":
                Gz = loop_response(Phi, B, st, ["u"], 0, zs)[:, 0]
                Cz = t_max * (K_P + K_I * zs / (zs - 1.0))
                Lz = Cz * Gz
            else:
                Gz = loop_response(Phi, B, st, ["psi", "r"], 1, zs)
                Cpsi = g["a"] + g["kI"] * g["a"] * DT_CTRL / (zs - 1.0)
                Lz = Cpsi * Gz[:, 0] + g["b"] * Gz[:, 1]
            mg = margins_from_L(Lz, w)
            res[loop] = dict(
                slowest=(None if slowest is None else
                         f"{slowest['re']:+.3f}"
                         + (f"{slowest['im']:+.3f}j" if slowest['im'] else "")
                         + f" (zeta {slowest['zeta']:.2f})"),
                slowest_mode=slowest,
                gain_margin_factor=gm_fac,
                gain_margin_db=(None if not np.isfinite(gm_fac) or gm_fac
                                <= 0.0 else float(20 * math.log10(gm_fac))),
                unstable_at_nominal_gain=bool(gm_fac == 0.0),
                nyquist=mg,
                phase_margin_deg=mg["phase_margin_deg"],
                max_re_this_loop_only=float(lk.real.max()))
        rows.append(res)
    return dict(rows=rows, dt_ctrl=DT_CTRL, n_sub=N_SUB, n_delay=N_DELAY,
                k_p=K_P, k_i=K_I,
                note="margins: gain_margin_factor = factor on the loop's "
                     "gains at which a closed-loop eigenvalue reaches the "
                     "unit circle; nyquist = classical margins of the loop "
                     "transfer function broken at the command")


# ------------------------------------------------------- time-domain check
def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def closed_loop_check(p, kns=(5.0, 12.0, 20.0, 30.0), T_end=60.0):
    """The nonlinear plant under the two controllers as written (the
    autopilot formula of sim/env.Episode._steer with its conditional
    integral and clip; the throttle hold of app_randomization_test._run
    with its clips), from a 5 deg heading error and from a -1 m/s speed
    error, against the linear lifted model from the same deviation."""
    warnings.filterwarnings("ignore")
    out = []
    pl = L.build(dt=0.01)
    Lhull = float(p["L"])
    t_max = float(pl.prop.t_max)
    rs = math.sqrt(Lhull / 10.0)
    st = list(L.STATES)
    for kn in kns:
        u = kn * KN
        e = L.equilibrium(pl, u)
        s0 = np.array(e["s"])
        Phi, B = L.step_jacobian(pl, s0, 1e-4, dt=0.01)
        M, g = closed_matrices(Phi, B, st, p, e["T"], t_max, Lhull)
        res = dict(kn=kn)
        for case, dpsi, du in (("heading", math.radians(5.0), 0.0),
                               ("speed", 0.0, -1.0)):
            # nonlinear
            s = s0.copy()
            s[5] += dpsi
            s[6] += du
            L.prep(pl, s0[12], 0.0)
            psi_i, acc = 0.0, 0.0
            c0 = s0[12] / t_max
            t = 0.0
            trace = [(0.0, s[5], s[6] - s0[6])]
            n_ctrl = int(round(T_end / DT_CTRL))
            lim = pl.rudder.max
            for k in range(n_ctrl):
                err_u = u - s[6]
                acc = min(2.0, max(-2.0, acc + err_u))
                thr = min(1.0, max(0.0, c0 + K_P * err_u + K_I * acc)) * t_max
                a_, b_ = g["a"], g["b"]
                # gains scheduled on the thrust being commanded (as _steer)
                gk = autopilot_gains(p, thr, Lhull)
                a_, b_ = gk["a"], gk["b"]
                err = _wrap(s[5] - 0.0)
                raw = -(a_ * err + b_ * s[11] + (0.12 / rs) * a_ * psi_i)
                if abs(raw) < lim:
                    psi_i = float(np.clip(psi_i + err * DT_CTRL, -6 * rs,
                                          6 * rs))
                noz = float(np.clip(raw, -lim, lim))
                for _ in range(N_SUB):
                    s = pl.step(s, t, thr, noz, 0.01)
                    t += 0.01
                trace.append((t, s[5], s[6] - s0[6]))
                if not np.all(np.isfinite(s)):
                    break
            tr = np.array(trace)
            # linear
            n = len(st)
            X = np.zeros(n + 4)
            X[st.index("psi")] = dpsi
            X[st.index("u")] = du
            lin = [(0.0, dpsi, du)]
            for k in range(n_ctrl):
                X = M @ X
                lin.append(((k + 1) * DT_CTRL, X[st.index("psi")],
                            X[st.index("u")]))
            li = np.array(lin)
            col = 1 if case == "heading" else 2
            err_rms = float(np.sqrt(np.mean((tr[:, col] - li[:, col]) ** 2)))
            res[case] = dict(
                nonlinear_end=float(tr[-1, col]), linear_end=float(li[-1, col]),
                nonlinear_abs_max_after_10s=float(np.abs(
                    tr[tr[:, 0] >= 10.0, col]).max()),
                linear_abs_max_after_10s=float(np.abs(
                    li[li[:, 0] >= 10.0, col]).max()),
                rms_diff=err_rms, finite=bool(np.all(np.isfinite(tr))),
                nonlinear=tr[::4].tolist(), linear=li[::4].tolist(),
                scale=float(abs(dpsi) if case == "heading" else abs(du)))
        out.append(res)
    return out
