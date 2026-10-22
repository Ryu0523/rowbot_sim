#!/usr/bin/env python3
"""
The one-step error the network predicts on the gz target, split EXACTLY into
named physical terms (learn/meta/ROADMAP_2026-09-30.md section 11.13).

Definitions (code: learn/meta/data3.py e0_from, model0_step):

    e0_k,c = Abar_c - Abar^M_c,     c in (u, v, r, w = heave rate, q = pitch rate)

  Abar_c    the step-mean acceleration of the gz plant over control step k
            (12 plant steps of 0.02 s, Heun): the velocity increment / 0.24 s
  Abar^M_c  model0's step-mean acceleration from the same reduced state with
            the same commands, no waves, ideal actuators (6 substeps of 0.04 s)

Each recorded gz mission is replayed through its own plant
(final_eval.make_mission, the recorded applied commands); the replay is
checked against the recorded states. In every plant step the Heun stages
are evaluated by deriv_terms, which recomputes sim/planing_vessel.py
deriv's right-hand side split into terms (checked: the terms sum to the
plant's own derivative); the step's velocity increment is the Heun-weighted
sum, so Abar_c = sum over terms exactly. model0 is split the same way
(model0_terms; checked against data3.model0_step). The error is then

    e0 = sum_j gz_term_j - sum_i model0_term_i

and the report gives, per channel, each term's standard deviation, mean and
share of the error variance cov(term, e0) / var(e0) (model0 terms with a
minus sign), plus paired components (gz term minus the model0 term with the
same physical role) where such a pairing exists.

gz terms (accelerations; heave/pitch through the instantaneous 2x2 added-
mass inverse, so each generalised force's share is exact):
  surge  jet, fric, air, fk_lon (Froude-Krylov of the longitudinal slope),
         pres (pressure drag = -sum f tau dx), cor (m v r)
  sway   jet, fk_tr, munk (slender-body lift at the aft station), cf
         (lateral cross-flow drag), cor (-m u r)
  yaw    jet, fk_tr, munk_stern (moment of the aft-station lift), munk_dist
         (the distributed Munk moment -|u| sum m22(d_i)(v + x_i r) dx), cf
  heave, pitch  hydro (buoyancy a_b rho g A(d) minus weight), wave_acc
         (m_a eta_tt), strip_cor (m_a 2 SP u q), impact (m_a' V^2, V>0), cf_v
         (vertical cross-flow), resist_m (friction and air pitch moments),
         jet_m (jet pitch moment)
model0 terms: surge thrust, drag; sway jet, lin, quad, cor; yaw jet, damp;
  heave/pitch restore, damp, disc (the exact-transition remainder).

    python -m studies.gz_terms --missions 0-7
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
import numpy as np   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_cache", "gz_terms")
CH = ("surge", "sway", "yaw", "heave", "pitch")
ROW = dict(surge=6, sway=7, yaw=11, heave=8, pitch=10)
G_ = 9.81


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


# ------------------------------------------------------------ gz terms
def deriv_terms(pl, s, t):
    """(ds, aux, terms): terms {channel: {name: acceleration}} for the
    plant's derivative at (s, t), recomputed from sim/planing_vessel.py
    deriv (lines 250-369) term by term."""
    from sim.planing_vessel import NU, RHO_AIR, SP
    ds, aux = pl.deriv(s, t)
    x0, y0, z, phi, th, psi = s[:6]
    u, v, w, p, q, r = s[6:12]
    thr, rud = s[12], s[13]
    x = pl.x
    cps, sps = np.cos(psi), np.sin(psi)
    X, Y = x0 + x * cps, y0 + x * sps
    e, et, ett, ex, ey = pl._waves(X, Y, t)
    s_lon = ex * cps + ey * sps
    s_tr = -ex * sps + ey * cps
    Th = pl.Th_rest + th
    d = e - (pl.Z_rest + z + pl.zk + SP * x * Th)
    tau = pl.kp + SP * Th
    V = et - (w + SP * x * q) + u * tau
    A, c, ma, dma, girth = pl._section(d)
    wet = d > 0.0
    dx, rho = pl.dx, pl.rho
    ab = pl._a_b(u)
    ctr = pl._ctr(u)
    ma, dma = ma * ctr, dma * ctr
    comp = dict(
        wave_acc=ma * ett,
        strip_cor=ma * (2.0 * SP * u * q - u * u * pl.kpp),
        impact=np.where(V > 0.0, dma * V * V, 0.0),
        cf_v=0.5 * rho * pl.c_dc * 2.0 * c * V * np.abs(V) * ctr,
        hydro=ab * rho * G_ * A * ctr)
    rest = sum(comp.values())
    lw = wet.sum() * dx
    S = (girth * wet).sum() * dx
    uu = max(abs(u), 0.05)
    re = uu * max(lw, 0.05) / NU
    cfr = 0.075 / (np.log10(re) - 2.0) ** 2 + 0.0004
    Df = 0.5 * rho * u * abs(u) * cfr * S
    zw = float(np.mean(pl.zk[wet])) if (S > 0 and wet.any()) else pl.zk[0]
    Da = 0.5 * RHO_AIR * pl.hull.cda * u * abs(u)
    Xw = -(rho * G_ * ab * A * s_lon).sum() * dx
    Ysec = -rho * G_ * ab * A * s_tr
    m22 = 0.5 * np.pi * rho * np.maximum(d, 0.0) ** 2
    vl = v + x * r
    Ylift = -abs(u) * m22[0] * vl[0]
    Ycf = -0.5 * rho * pl.cd_lat * np.maximum(d, 0.0) * vl * np.abs(vl)
    fx, fy = pl.prop.forces(thr, rud, u)
    xn, _, zn = pl.thrusters[0]
    # heave / pitch: generalised forces (r1, r2) per term, then the inverse
    A33 = ma.sum() * dx
    A35 = (ma * SP * x).sum() * dx
    A55 = (ma * x * x).sum() * dx
    m11, m12, m22_ = pl.m + A33, A35, pl.Iy + A55
    det = m11 * m22_ - m12 * m12
    gen = {k: (v_.sum() * dx, (SP * x * v_).sum() * dx)
           for k, v_ in comp.items()}
    gen["hydro"] = (gen["hydro"][0] - pl.m * G_, gen["hydro"][1])
    gen["resist_m"] = (0.0, zw * (-Df) + 0.5 * (-Da))
    gen["jet_m"] = (0.0, zn * fx)
    hv, pt = {}, {}
    for k, (r1, r2) in gen.items():
        hv[k] = (r1 * m22_ - m12 * r2) / det
        pt[k] = (m11 * r2 - m12 * r1) / det
    wd, qd = sum(hv.values()), sum(pt.values())
    f = rest - ma * (wd + SP * x * qd)
    Xp = -(f * tau).sum() * dx
    M11, M22, M66 = pl.Mtot[0, 0], pl.Mtot[1, 1], pl.Mtot[5, 5]
    terms = dict(
        surge=dict(jet=fx / M11, fric=-Df / M11, air=-Da / M11,
                   fk_lon=Xw / M11, pres=Xp / M11, cor=pl.m * v * r / M11),
        sway=dict(jet=fy / M22, fk_tr=Ysec.sum() * dx / M22,
                  munk=Ylift / M22, cf=Ycf.sum() * dx / M22,
                  cor=-pl.m * u * r / M22),
        yaw=dict(jet=xn * fy / M66, fk_tr=(x * Ysec).sum() * dx / M66,
                 munk_stern=x[0] * Ylift / M66,
                 munk_dist=-abs(u) * (m22 * vl).sum() * dx / M66,
                 cf=(x * Ycf).sum() * dx / M66),
        heave=hv, pitch=pt)
    return ds, aux, terms


def _add(acc, terms, wgt):
    for ch, tt in terms.items():
        a = acc.setdefault(ch, {})
        for k, v_ in tt.items():
            a[k] = a.get(k, 0.0) + wgt * float(v_)


def step_terms(pl, s, t, thr_cmd, rud_cmd, dt):
    """sim/planing_vessel.py PlaningVessel.step (lines 522-569) with the
    Heun stages' terms accumulated: returns (s_new, {channel: {term: mean
    acceleration over the step}})."""
    from sim.planing_vessel import N_SUB_MAX, STIFF_STEP, STIFF_SUB
    pl.prop.update_prime(pl.prop_submergence(s[:6], t), dt)
    k1, aux, T1 = deriv_terms(pl, s, t)
    n = 1
    lam = pl._lateral_rate(s, aux["d"]) * dt
    if lam > STIFF_STEP:
        n = min(int(np.ceil(lam / STIFF_SUB)), N_SUB_MAX)
    acc = {}
    if n == 1:
        s2 = s + dt * k1
        k2, _, T2 = deriv_terms(pl, s2, t + dt)
        s_new = s + 0.5 * dt * (k1 + k2)
        _add(acc, T1, 0.5)
        _add(acc, T2, 0.5)
    else:
        h = dt / n
        s_new, k, Tk = s, k1, T1
        for i in range(n):
            if i:
                k, _, Tk = deriv_terms(pl, s_new, t + i * h)
            k2, _, T2 = deriv_terms(pl, s_new + h * k, t + (i + 1) * h)
            s_new = s_new + 0.5 * h * (k + k2)
            _add(acc, Tk, 0.5 / n)
            _add(acc, T2, 0.5 / n)
    s_new[12] = pl.prop.advance(s[12], thr_cmd, dt)
    s_new[13] = pl.rudder.step(s[13], rud_cmd, dt)
    return s_new, acc, n


def replay(key, S):
    """Replay mission key = (seed, leg) with its recorded commands:
    (per-step gz terms list, replayed end states, max one-step deviation
    of the replayed end state from the recorded one, prime trace, substep
    counts)."""
    from studies import final_eval as FE
    rec = S[key]
    m = FE.make_mission(key[0], key[1], 120.0)
    pl = m.plant
    s, t = m.s.copy(), m.t
    xs = rec["xs"]
    out, states, prime, nsub = [], [s.copy()], [], []
    dev = float(np.max(np.abs(s - xs[0])))
    for k in range(rec["n"]):
        # restart every control step from the RECORDED state and time (the
        # free replay drifts: the stored command fractions times t_max
        # differ from the applied thrust in the last bits, and the impact
        # dynamics amplify that); the hidden prime and delay lines carry
        # over from the replay, driven by the recorded trajectory
        s, t = xs[k].copy(), float(rec["t"][k])
        thr = rec["u"][k, 0] * m.t_max
        rud = rec["u"][k, 1] * m.rud_max
        accs = []
        for _ in range(m.sub):
            s, acc, n = step_terms(pl, s, t, thr, rud, m.dt)
            accs.append(acc)
            prime.append(float(pl.prop.prime))
            nsub.append(n)
            t += m.dt
        mean = {ch: {k_: float(np.mean([a[ch][k_] for a in accs]))
                     for k_ in accs[0][ch]} for ch in accs[0]}
        out.append(mean)
        states.append(s.copy())
        dev = max(dev, float(np.max(np.abs(s[:12] - xs[k + 1][:12]))))
    return out, np.array(states), dev, np.array(prime), np.array(nsub)


# ------------------------------------------------------------ model0 terms
def model0_terms(env, sr, U):
    """data3.model0_step split into terms: {channel: {term: step-mean
    acceleration}} for one state sr (10,) and command U (2,), plus the
    model0 end state (for the check)."""
    red = env["red"]
    p = red.p
    h, n = env["dt"], env["sub"]
    Dt = h * n
    thr, rud = U[0] * env["t_max"], U[1] * env["rud_max"]
    eta0 = np.zeros((1, len(env["x_st"]), 3))
    acc = {ch: {} for ch in CH}

    def add(ch, k, v_):
        acc[ch][k] = acc[ch].get(k, 0.0) + float(v_) * h / Dt

    cur = sr[None].copy()
    for _ in range(n):
        x, y, u, z, zd, th, thd, psi, r, v = cur[0]
        nxt = red.step(cur, np.array([thr]), np.array([rud]), eta0,
                       env["x_st"], h)[0]
        add("surge", "thrust", thr / p["m_surge"])
        add("surge", "drag", -p["k_drag"] * u * abs(u) / p["m_surge"])
        rud_eff = np.clip(rud, -p["rud_stall"], p["rud_stall"])
        lift = p["k_jet_side"] * max(thr, 0.0) * np.sin(rud_eff)
        add("sway", "jet", lift / p["m_sway"])
        add("sway", "lin", -p["k_lin_sway"] * v / p["m_sway"])
        add("sway", "quad", -p["k_sway"] * v * abs(v) / p["m_sway"])
        add("sway", "cor", -p["m_coriolis"] * u * r / p["m_sway"])
        ay = float(np.exp(-h / p["tau_r"]))
        r_tgt = p["k_nomoto_f"] * lift
        add("yaw", "jet", r_tgt * (1 - ay) / h)
        add("yaw", "damp", -r * (1 - ay) / h)
        for ch, pos, vel, wn, zt, eq, i_v in (
                ("heave", z, zd, p["wn_heave"], p["z_heave"],
                 p.get("z0", 0.0), 4),
                ("pitch", th, thd, p["wn_pitch"], p["z_pitch"],
                 p.get("th0", 0.0), 6)):
            rs = -wn * wn * (pos - eq)
            dp = -2.0 * zt * wn * vel
            add(ch, "restore", rs)
            add(ch, "damp", dp)
            add(ch, "disc", (nxt[0, i_v] - vel) / h - rs - dp)
        cur = nxt
    return acc, cur[0]


# ------------------------------------------------------------ report
def summarise(rows, e0):
    """rows: list over steps of {'gz': terms, 'm0': terms}; e0 (n, 5)."""
    lines = []
    for ci, ch in enumerate(CH):
        e = e0[:, ci]
        ve = e.var()
        names = sorted(rows[0]["gz"][ch])
        mnames = sorted(rows[0]["m0"][ch])
        G = {k: np.array([r_["gz"][ch][k] for r_ in rows]) for k in names}
        M = {k: np.array([r_["m0"][ch][k] for r_ in rows]) for k in mnames}
        recon = sum(G.values()) - sum(M.values())
        lines.append(f"{ch}: e0 std {e.std():.4f} mean {e.mean():+.4f}; "
                     f"reconstruction max |gz - model0 - e0| "
                     f"{np.max(np.abs(recon - e)):.2e}")
        cols = [(f"gz.{k}", G[k]) for k in names] + \
               [(f"-m0.{k}", -M[k]) for k in mnames]
        for lab, a in sorted(cols, key=lambda z: -abs(np.cov(z[1], e)[0, 1])):
            sh = np.cov(a, e)[0, 1] / ve
            lines.append(f"    {lab:16s} std {a.std():9.4f} mean {a.mean():+9.4f}"
                         f" share of var(e0) {sh:+.3f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", default="0-7")
    args = ap.parse_args()
    from learn.meta import data3
    from learn.meta import relabel
    from learn.meta.data2 import plant_to_reduced
    from studies import onestep_gz as G
    keys = G.parse_missions(args.missions)
    S = G.load_streams(keys, 120.0)
    env = relabel._env()
    os.makedirs(OUT, exist_ok=True)
    allrows, alle0 = [], []
    for key in keys:
        tb = time.time()
        gz, states, dev, prime, nsub = replay(key, S)
        xs, U = S[key]["xs"], S[key]["u"]
        e0 = data3.e0_from(env, xs[:-1], xs[1:], U)[:, :5]
        rows = []
        mdev = 0.0
        a_red = plant_to_reduced(xs[:-1])
        m_ref = data3.model0_step(env, a_red, U)
        for k in range(len(U)):
            m0, end = model0_terms(env, a_red[k], U[k])
            mdev = max(mdev, float(np.max(np.abs(end - m_ref[k]))))
            # gz terms are physical step means over the REPLAYED step; the
            # e0 of the record uses the recorded states (checked equal)
            rows.append(dict(gz=gz[k], m0=m0))
        pickle.dump(dict(key=key, rows=rows, e0=e0, prime=prime, nsub=nsub,
                         dev=dev, mdev=mdev),
                    open(os.path.join(OUT, f"s{key[0]}_l{key[1]}.pkl"), "wb"))
        log(f"mission {key}: {len(U)} steps, replay max |state - record| "
            f"{dev:.2e}, model0 split max |end - model0_step| {mdev:.2e}, "
            f"prime < 1 on {np.mean(prime < 0.999):.3f} of plant steps (min "
            f"{prime.min():.3f}), substepped plant steps "
            f"{np.mean(nsub > 1):.3f}, {time.time() - tb:.0f} s")
        allrows += rows
        alle0.append(e0)
    e0 = np.concatenate(alle0)
    rep = summarise(allrows, e0)
    log(f"all {len(keys)} missions, {len(e0)} steps:")
    for line in rep:
        print(line, flush=True)


if __name__ == "__main__":
    main()
