#!/usr/bin/env python3
"""
Steps 1 and 2 of learn/meta/SIM_REDESIGN_2026-10-08.md.

STEP 1  reproduce Zarnick's (1978) own 2D+t computation of Fridsma's (1969)
        towed prismatic models with sim/planing_zarnick.py, one formulation
        difference at a time, and compare with Zarnick's digitised column and
        with the measurements (exp_planing.FRIDSMA / ZARNICK).
STEP 2  the modified Logvinovich (MLM) sectional entry force (switch `mlm`):
        wedge drop (Zhao, Faltinsen & Aarsnes; exp_slam_wedge data), Savitsky
        steady trim and resistance, Fridsma 1969 and 1971, before vs after.

Checks and tolerances were registered before the runs in the plan document
(section "进展 / 事先登记").

Run: python -m studies.exp_zarnick_mlm [step1|step2|all]
Results: studies/_cache/sim_redesign/*.json and the printed tables.
"""
import json
import math
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_cache", "sim_redesign")
KN = 0.514444
B_FR = 9 * 0.0254
KEEL = ("ellipse", 5.375 * 0.0254)       # Fridsma 1969 fig.: (x/9)^2+(8y/43)^2=1

from sim.planing_zarnick import ZOPT_V0, ZOPT_ZARNICK, ZOPT_ZR  # noqa: E402


def cfg(base, **kw):
    d = dict(base)
    d.update(kw)
    return d


# ------------------------------------------------------------------ runs
def regular_wave(args):
    """exp_planing.regular_wave with ZarnickVessel (same measurement)."""
    name, zopt, case, lam, dt, T = args
    from studies.exp_planing import fridsma_hull
    from sim.planing_zarnick import ZarnickVessel
    from sim.test_vessel import Monochromatic
    hull, V = fridsma_hull(case, tow="nozzle")
    zopt = dict(zopt)
    if zopt.get("keel") is not None:
        zopt["keel"] = tuple(zopt["keel"])
    L = hull.L
    H = 0.111 * hull.b
    k = 2 * np.pi / (lam * L)
    a = 0.5 * H
    om = np.sqrt(9.81 * k)
    sea = Monochromatic(om, a, theta0=np.pi)
    p = ZarnickVessel(hull, sea, dt=dt, captive_u=V, zopt=zopt)
    x_acc = hull.L - hull.lcg - 0.1 * hull.L
    s = np.zeros(14)
    s[6] = V
    s[2], s[4] = p.running_attitude(V)
    n = int(T / dt)
    rec = np.empty((n, 4))
    xb = p.x[-1]
    for i in range(n):
        s = p.step(s, i * dt, 0.0, 0.0)
        s[6] = V
        a_cg, a_b = p.last_cg_acc, p.last_bow_acc
        rec[i] = s[2], s[4], a_cg / 9.81, (a_cg + (a_b - a_cg) * x_acc
                                             / xb) / 9.81
    q = rec[n // 2:]
    if not np.all(np.isfinite(q)):
        return dict(name=name, case=case, lam=lam, dt=dt, bad=True)
    amp = 0.5 * (q[:, :2].max(0) - q[:, :2].min(0))
    te = 2 * np.pi / (om + k * V)
    m = max(int(round(te / dt)), 1)
    cyc = [q[j:j + m] for j in range(0, len(q) - m + 1, m)]
    pk = np.mean([[c[:, 2].max(), c[:, 3].max()] for c in cyc], axis=0)
    return dict(name=name, case=case, lam=lam, dt=dt,
                res=[float(amp[0] / a), float(amp[1] / (k * a)),
                     float(pk[0]), float(pk[1])])


def calm_fridsma(args):
    name, zopt, case = args
    from studies.exp_planing import fridsma_hull
    from sim.planing_zarnick import ZarnickVessel
    from sim.test_vessel import Monochromatic
    hull, V = fridsma_hull(case, tow="nozzle")
    zopt = dict(zopt)
    if zopt.get("keel") is not None:
        zopt["keel"] = tuple(zopt["keel"])
    p = ZarnickVessel(hull, Monochromatic(1.0, 0.0), dt=0.004, zopt=zopt)
    cr = p.calm_resistance(V)
    return dict(name=name, case=case, trim=cr["trim_deg"],
                rise_b=cr["rise"] / hull.b,
                R_D=cr["R"] / (hull.mass * 9.81))


def savitsky_job(args):
    name, zopt, kn = args
    from hydro.planing import savitsky
    from sim import config
    from sim.planing_zarnick import ZarnickVessel
    from sim.test_vessel import Monochromatic
    h, _ = config.load("scarab195")
    p = ZarnickVessel(h, Monochromatic(1.0, 0.0), dt=0.02, zopt=zopt)
    r = p.calm_resistance(kn * KN)
    sv = savitsky(kn * KN, h.mass, h.lcg, h.b, h.beta_deg)
    return dict(name=name, kn=kn, trim=r["trim_deg"], R=r["R"] - r["air"],
                trim_sav=sv["tau_deg"], R_sav=sv["R"])


def fridsma71_job(args):
    """exp_planing.p7 for one condition and one Tp factor."""
    name, zopt, cond, f, T, dt = args
    from hydro.planing import PlaningHull
    from studies.exp_planing import FRIDSMA71, excursion_peaks
    from sim.planing_zarnick import ZarnickVessel
    from sim.wavefield import SeaState
    vl, lcg_aft, hb, m_cg, m_bow = FRIDSMA71[cond]
    b = B_FR
    L = 5 * b
    V = vl * 0.514444 * np.sqrt(L / 0.3048)
    h = PlaningHull(name=f"fridsma71_{cond}", L=L, b=b, beta_deg=20.0,
                    mass=0.600 * 62.4 * (b / 0.3048) ** 3 * 0.45359237,
                    lcg=(1 - lcg_aft / 100) * L, vcg=0.294 * b,
                    depth=0.6 * b, k_pitch=0.25, taper=0.2, cda=0.0,
                    u_design=V, rho=1000.0,
                    jet=dict(d_nozzle=0.02, p_max=500.0))
    zopt = dict(zopt)
    if zopt.get("keel") is not None:
        zopt["keel"] = tuple(zopt["keel"])
    hs = hb * b
    tp = f * 15.63 * np.sqrt(hs / 9.81)
    sea = SeaState(hs, tp, gamma=1.0, theta0=np.pi, n_freq=40, n_dir=1,
                   seed=cond)
    p = ZarnickVessel(h, sea, dt=dt, captive_u=V, zopt=zopt)
    s = np.zeros(14)
    s[6] = V
    s[2], s[4] = p.running_attitude(V)
    xa = h.L - h.lcg - 0.1 * h.L
    n = int(T / dt)
    acc = np.empty((n, 2))
    for i in range(n):
        s = p.step(s, i * dt, 0.0, 0.0)
        s[6] = V
        ac, ab = p.last_cg_acc, p.last_bow_acc
        acc[i] = ac / 9.81, (ac + (ab - ac) * xa / p.x[-1]) / 9.81
    acc = acc[n // 10:]
    if not np.all(np.isfinite(acc)):
        return dict(name=name, cond=cond, f=f, bad=True)
    return dict(name=name, cond=cond, f=f,
                cg=float(excursion_peaks(acc[:, 0]).mean()),
                bow=float(excursion_peaks(acc[:, 1]).mean()),
                meas=[m_cg, m_bow])


# ------------------------------------------------------------- scoring
def fridsma_scores(rows):
    """Per config: 44-item count vs measured (amp +-20 %, acc +-30 %) and
    22-value Zarnick reproduction (|d| <= 0.10 Z + 0.03)."""
    from studies.exp_planing import FRIDSMA, ZARNICK
    out = {}
    for r in rows:
        if r.get("bad"):
            continue
        o = out.setdefault(r["name"], dict(meas=0, nmeas=0, z=0, nz=0,
                                           zres=0, nzres=0, tab={}))
        mh, mp, mc, mb = FRIDSMA[r["case"]][r["lam"]]
        zh, zp = ZARNICK[r["case"]][r["lam"]]
        hv, pt, cg, bw = r["res"]
        o["tab"][f"{r['case']}{r['lam']}"] = r["res"]
        for val, ref, tol in ((hv, mh, .2), (pt, mp, .2), (cg, mc, .3),
                              (bw, mb, .3)):
            o["nmeas"] += 1
            o["meas"] += abs(val / ref - 1) <= tol
        resonant = r["lam"] in (3.0, 4.0)
        for val, ref in ((hv, zh), (pt, zp)):
            ok = abs(val - ref) <= 0.10 * ref + 0.03
            o["nz"] += 1
            o["z"] += ok
            if resonant:
                o["nzres"] += 1
                o["zres"] += ok
    return out


# ------------------------------------------------------------- step 1
STEP1 = {
    "v0": ZOPT_V0,
    "v0+bm": cfg(ZOPT_V0, bm=0.5),
    "v0+tow": cfg(ZOPT_V0, tow="cg"),
    "v0+fric": cfg(ZOPT_V0, fric="cg"),
    "v0+normal": cfg(ZOPT_V0, normal=True),
    "v0+keel": cfg(ZOPT_V0, keel=KEEL),
    "v0+momboth": cfg(ZOPT_V0, mom="both"),
    "v0+cfsq": cfg(ZOPT_V0, cf="sq"),
    "Z": ZOPT_ZARNICK,
    "Z-bm": cfg(ZOPT_ZARNICK, bm=1.0),
    "Z-tow": cfg(ZOPT_ZARNICK, tow="nozzle"),
    "Z-fric": cfg(ZOPT_ZARNICK, fric="keel"),
    "Z-normal": cfg(ZOPT_ZARNICK, normal=False),
    "Z+keel": cfg(ZOPT_ZARNICK, keel=KEEL),
    "Z+momboth": cfg(ZOPT_ZARNICK, mom="both"),
    "Z+cfsq": cfg(ZOPT_ZARNICK, cf="sq"),
}


def z0_check():
    """Z0: all switches off == v0, to the last bit (case B, lam/L 3)."""
    from studies.exp_planing import fridsma_hull
    from sim.planing_vessel import PlaningVessel
    from sim.planing_zarnick import ZarnickVessel
    from sim.test_vessel import Monochromatic
    out = {}
    for case in ("A", "B"):
        hull, V = fridsma_hull(case, tow="nozzle")
        L = hull.L
        k = 2 * np.pi / (3.0 * L)
        sea = Monochromatic(np.sqrt(9.81 * k), 0.0555 * hull.b,
                            theta0=np.pi)
        traj = []
        for cls, kw in ((PlaningVessel, {}), (ZarnickVessel, {})):
            hull, V = fridsma_hull(case, tow="nozzle")
            p = cls(hull, sea, dt=0.004, captive_u=V, **kw)
            s = np.zeros(14)
            s[6] = V
            s[2], s[4] = p.running_attitude(V)
            rec = []
            for i in range(2000):
                s = p.step(s, i * 0.004, 0.0, 0.0)
                s[6] = V
                rec.append(np.r_[s, p.last_cg_acc, p.last_bow_acc])
            traj.append(np.array(rec))
        out[case] = float(np.abs(traj[0] - traj[1]).max())
    # free-running scarab195 at 25 kn in a sea, jet on (not captive)
    from sim import config
    from sim.wavefield import SeaState
    h, _ = config.load("scarab195")
    sea = SeaState(1.0, 5.0, theta0=np.pi, n_freq=24, n_dir=5, seed=0)
    trajs = []
    for cls in (PlaningVessel, ZarnickVessel):
        h, _ = config.load("scarab195")
        p = cls(h, sea, dt=0.02)
        s = p.initial_state(25 * KN)
        thr = s[12]
        rec = []
        for i in range(1000):
            s = p.step(s, i * 0.02, thr, 0.05)
            rec.append(s.copy())
        trajs.append(np.array(rec))
    out["scarab_free"] = float(np.abs(trajs[0] - trajs[1]).max())
    return out


def step1():
    from studies.parallel_runner import run_parallel
    from studies.exp_planing import FRIDSMA
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    z0 = z0_check()
    print("Z0 max |ZarnickVessel(all off) - v0|:", z0)
    jobs = []
    for name, zo in STEP1.items():
        for case in ("A", "B"):
            for lam in FRIDSMA[case]:
                jobs.append((name, zo, case, lam, 0.004, 16.0))
    for lam in (3.0, 4.0):
        jobs.append(("Z@dt0.002", ZOPT_ZARNICK, "B", lam, 0.002, 16.0))
        jobs.append(("Z@dt0.004", ZOPT_ZARNICK, "B", lam, 0.004, 16.0))
    rows = run_parallel(regular_wave, jobs, gb_per_job=0.2, reserve_gb=3.0)
    rows = [r for r in rows if isinstance(r, dict)]
    calm = run_parallel(calm_fridsma, [(n, zo, c) for n, zo in STEP1.items()
                                       for c in ("A", "B")],
                        gb_per_job=0.2, reserve_gb=3.0)
    res = dict(z0=z0, rows=rows, calm=[c for c in calm if isinstance(c, dict)],
               wall_s=time.time() - t0)
    with open(os.path.join(OUT, "step1.json"), "w") as f:
        json.dump(res, f, indent=1)
    report1(res)


def report1(res):
    from studies.exp_planing import FRIDSMA, ZARNICK
    rows = [r for r in res["rows"] if not r["name"].startswith("Z@")]
    sc = fridsma_scores(rows)
    print("\nScores: name | 44 vs measured | 22 vs Zarnick | 8 resonant vs "
          "Zarnick")
    for n in STEP1:
        o = sc.get(n)
        if o:
            print(f"  {n:<12} {o['meas']:>2}/{o['nmeas']}   {o['z']:>2}/"
                  f"{o['nz']}   {o['zres']}/{o['nzres']}")
    print("\nHeave h/H and pitch ratio per case (cfg: value)")
    keys = [f"{c}{l}" for c in ("A", "B") for l in FRIDSMA[c]]
    names = list(STEP1)
    print("  case   meas(h,p)   Zarn(h,p)  " + "  ".join(f"{n:>11}"
                                                       for n in names))
    for kk in keys:
        c, l = kk[0], float(kk[1:])
        mh, mp = FRIDSMA[c][l][:2]
        zh, zp = ZARNICK[c][l]
        line = f"  {kk:<6} {mh:.2f},{mp:.2f}   {zh:.2f},{zp:.2f}  "
        for n in names:
            v = sc.get(n, {}).get("tab", {}).get(kk)
            line += "  " + (f"{v[0]:.2f},{v[1]:.2f}".rjust(11) if v else
                            "nan".rjust(11))
        print(line)
    print("\nCG / bow g per case")
    for kk in keys:
        c, l = kk[0], float(kk[1:])
        mc, mb = FRIDSMA[c][l][2:]
        line = f"  {kk:<6} {mc:.2f},{mb:.2f}  "
        for n in names:
            v = sc.get(n, {}).get("tab", {}).get(kk)
            line += "  " + (f"{v[2]:.2f},{v[3]:.2f}".rjust(11) if v else
                            "nan".rjust(11))
        print(line)
    dtr = {(r["name"], r["lam"]): r["res"] for r in res["rows"]
           if r["name"].startswith("Z@")}
    print("\nZ4 time step, case B:")
    for lam in (3.0, 4.0):
        a, b = dtr.get(("Z@dt0.004", lam)), dtr.get(("Z@dt0.002", lam))
        if a and b:
            print(f"  lam {lam}: dt .004 {a[0]:.3f},{a[1]:.3f}  dt .002 "
                  f"{b[0]:.3f},{b[1]:.3f}  rel {a[0]/b[0]-1:+.2%} "
                  f"{a[1]/b[1]-1:+.2%}")
    print("\nCalm water (trim deg, rise/b, R/Delta); measured A 4.0/0.05/"
          "0.158, B 4.0/0.13/0.206")
    for c in res["calm"]:
        print(f"  {c['name']:<12} {c['case']}  {c['trim']:.2f}  "
              f"{c['rise_b']:.3f}  {c['R_D']:.3f}")


# ------------------------------------------------------------- step 2
def wedge_drop(Km_Kv=None):
    """Strip sectional force for the 30 deg drop-test section (one station,
    knuckle half-breadth 0.25 m), with the measured V(t): v0's Wagner law vs
    the MLM law; chine-wetted: Zarnick's frozen half-beam (both laws).
    Also exact MLM (planing_wedge2d.wedge_entry) for W2 and the W3
    continuation after chine wetting, and a free drop of the 241 kg rig."""
    from studies.exp_slam_wedge import (T_F, F_MEAS, F_NONLIN, T_V, V_MEAS,
                                        HALF_B, L_MEAS, L_RIG, M_RIG, RHO, G)
    from sim.planing_zarnick import _mlm_constants
    from hydro.rankine.planing_wedge2d import wedge_entry
    beta = 30.0
    T = math.tan(math.radians(beta))
    l = HALF_B
    Kv, Km, th = _mlm_constants(beta)
    laws = {"wagner": (math.pi ** 2 / (2 * T), math.pi / 2),
            "mlm": (Kv, Km)}
    cdc = math.cos(math.radians(beta))
    t = np.linspace(0.0, 0.025, 2501)
    v = np.interp(t, T_V, V_MEAS)
    h = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1])
                                         * np.diff(t))])
    vd = np.gradient(v, t)

    def strip(hh, vv, aa, law, parts=False):
        kv, km = laws[law]
        cw = math.pi * hh / (2 * T)
        c = min(cw, l)
        mom = kv * RHO * c * vv * vv if (cw < l and vv > 0) else 0.0
        acc = km * RHO * c * c * aa
        cf = 0.5 * RHO * cdc * 2 * c * vv * abs(vv)
        dch = l * T
        A = min(hh, dch) ** 2 / T + 2 * l * max(hh - dch, 0.0)
        buoy = RHO * G * A
        if parts:
            return mom, acc, cf, buoy
        return mom + acc + cf + buoy

    out = {}
    for law in laws:
        F = np.array([strip(a, b, c, law) for a, b, c in zip(h, v, vd)]) \
            * L_MEAS
        Fe = np.array([sum(strip(a, b, c, law, True)[:2])
                       for a, b, c in zip(h, v, vd)]) * L_MEAS
        out[law] = dict(F=F, Fentry=Fe)
    # exact MLM with the measured kinematics (rho = 1 -> x RHO)
    Vf = lambda tt: float(np.interp(tt, t, v))
    hf = lambda tt: float(np.interp(tt, t, h))
    af = lambda tt: float(np.interp(tt, t, vd))
    ex = wedge_entry(T, l, hf, Vf, af, 0.025, n=2500)
    out["mlm_exact"] = dict(F=np.interp(t, ex["t"], ex["F"]) * RHO * L_MEAS,
                            t_cw=ex["t_cw"])
    early = (t > 0.002) & (t < 0.0075)
    fm = np.interp(t[early], T_F, F_MEAS)
    rep = {}
    for k in ("wagner", "mlm"):
        for part in ("F", "Fentry"):
            arr = out[k][part]
            rep[f"{k}_{part}"] = dict(
                early_ratio=float(np.mean(arr[early] / fm)),
                peak=float(arr.max()), t_peak_ms=float(t[np.argmax(arr)]
                                                      * 1e3),
                at_ms={str(tt): float(np.interp(tt / 1e3, t, arr))
                       for tt in (2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25)})
    exF = out["mlm_exact"]["F"]
    rep["mlm_exact"] = dict(early_ratio=float(np.mean(exF[early] / fm)),
                            peak=float(np.nanmax(exF)),
                            t_cw_ms=out["mlm_exact"]["t_cw"] * 1e3,
                            at_ms={str(tt): float(np.interp(tt / 1e3, t,
                                                            exF))
                                   for tt in (2.5, 5, 7.5, 10, 12.5, 15,
                                              17.5, 20, 25)})
    # W2: fixed-cutoff strip MLM (entry part = mom + acc) vs exact, chine dry
    t_cw = out["mlm_exact"]["t_cw"] or 0.025
    dry = (t > 0.001) & (t < t_cw - 1e-4)
    rel = out["mlm"]["Fentry"][dry] / exF[dry] - 1
    rep["W2_max_rel"] = float(np.max(np.abs(rel)))
    rep["W2_mean_rel"] = float(np.mean(rel))
    rep["measured"] = {str(tt): float(np.interp(tt / 1e3, T_F, F_MEAS))
                       for tt in (2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25)}
    rep["nonlin2d"] = {str(tt): float(np.interp(tt / 1e3, T_F, F_NONLIN))
                       for tt in (2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25)}
    # free drop of the rig, strip force over 1 m, m_a on the left
    for law in laws:
        kv, km = laws[law]
        dt, hh, vv, tt = 1e-5, 0.0, V_MEAS[0], 0.0
        tr = []
        while tt < 0.025:
            cw = math.pi * hh / (2 * T)
            c = min(cw, l)
            mom = kv * RHO * c * vv * vv if (cw < l and vv > 0) else 0.0
            cf = 0.5 * RHO * cdc * 2 * c * vv * abs(vv)
            dch = l * T
            A = min(hh, dch) ** 2 / T + 2 * l * max(hh - dch, 0.0)
            ma = km * RHO * c * c
            acc = (M_RIG * G - (mom + cf + RHO * G * A) * L_RIG) / (
                M_RIG + ma * L_RIG)
            f_sec = (mom + cf + RHO * G * A + ma * acc) * L_MEAS
            vv += acc * dt
            hh += vv * dt
            tt += dt
            tr.append((tt, vv, f_sec))
        tr = np.array(tr)
        rep[f"drop_{law}"] = {str(q): [float(np.interp(q / 1e3, tr[:, 0],
                                                       tr[:, 1])),
                                       float(np.interp(q / 1e3, tr[:, 0],
                                                       tr[:, 2]))]
                              for q in (10, 15, 20, 25)}
    rep["Kv_Km_th"] = [Kv, Km, th]
    return rep


def step2(base_name="ZR"):
    from studies.parallel_runner import run_parallel
    from studies.exp_planing import FRIDSMA, FRIDSMA71
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    base = STEP2_BASES[base_name]
    cfgs = {"before": base, "after": cfg(base, mlm=True)}
    w = wedge_drop()
    jobs_rw = [(n, zo, c, lam, 0.004, 16.0) for n, zo in cfgs.items()
               for c in ("A", "B") for lam in FRIDSMA[c]]
    jobs_sav = [(n, zo, kn) for n, zo in cfgs.items()
                for kn in (10, 12, 14, 16, 20, 25, 30, 35, 40)]
    jobs_71 = [(n, zo, cond, f, 60.0, 0.004) for n, zo in cfgs.items()
               for cond in FRIDSMA71 for f in (0.8, 1.0, 1.25)]
    rw = run_parallel(regular_wave, jobs_rw, gb_per_job=0.2, reserve_gb=3.0)
    sav = run_parallel(savitsky_job, jobs_sav, gb_per_job=0.2,
                       reserve_gb=3.0)
    f71 = run_parallel(fridsma71_job, jobs_71, gb_per_job=0.2,
                       reserve_gb=3.0)
    calm = run_parallel(calm_fridsma, [(n, zo, c) for n, zo in cfgs.items()
                                       for c in ("A", "B")],
                        gb_per_job=0.2, reserve_gb=3.0)
    keep = lambda L: [x if isinstance(x, dict) else repr(x) for x in L]
    res = dict(base=base_name, wedge=w, rows=keep(rw), sav=keep(sav),
               f71=keep(f71), calm=keep(calm), wall_s=time.time() - t0)
    with open(os.path.join(OUT, f"step2_{base_name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    report2(res)


STEP2_BASES = {"ZR": ZOPT_ZR, "Z": ZOPT_ZARNICK, "v0": ZOPT_V0,
               "Z+keel": cfg(ZOPT_ZARNICK, keel=KEEL)}


def report2(res):
    w = res["wedge"]
    print(f"\nWEDGE (30 deg) strip section, measured V(t); K_v, K_m, th = "
          f"{w['Kv_Km_th']}")
    print("  t ms   meas  nonlin2D  wagner(all)  mlm(all)  wagner(entry)  "
          "mlm(entry)  mlm_exact")
    for tt in ("2.5", "5", "7.5", "10", "12.5", "15", "17.5", "20", "25"):
        print(f"  {tt:>5} {w['measured'][tt]:6.0f} {w['nonlin2d'][tt]:8.0f}"
              f" {w['wagner_F']['at_ms'][tt]:11.0f}"
              f" {w['mlm_F']['at_ms'][tt]:9.0f}"
              f" {w['wagner_Fentry']['at_ms'][tt]:13.0f}"
              f" {w['mlm_Fentry']['at_ms'][tt]:11.0f}"
              f" {w['mlm_exact']['at_ms'][tt]:10.0f}")
    for k in ("wagner_F", "mlm_F", "wagner_Fentry", "mlm_Fentry",
              "mlm_exact"):
        print(f"  {k:<14} early ratio {w[k]['early_ratio']:.3f}  peak "
              f"{w[k]['peak']:.0f} N")
    print(f"  W2 fixed-cutoff vs exact MLM (chine dry): max |rel| "
          f"{w['W2_max_rel']:.3%}, mean {w['W2_mean_rel']:+.3%}; exact "
          f"chine wetting at {w['mlm_exact']['t_cw_ms']:.2f} ms")
    for law in ("wagner", "mlm"):
        print(f"  free drop {law}: " + "; ".join(
            f"{q} ms V {v[0]:.2f} F {v[1]:.0f}" for q, v in
            w[f"drop_{law}"].items()) + "  (meas V 5.80/5.35/5.02/4.90, F "
            "4500/5125/2600/1900)")
    print("\nSAVITSKY scarab195 (trim, R N) before | after | Savitsky")
    sav = [s for s in res["sav"] if isinstance(s, dict)]
    by = {(s["name"], s["kn"]): s for s in sav}
    cnt = {}
    for kn in (10, 12, 14, 16, 20, 25, 30, 35, 40):
        b, a = by.get(("before", kn)), by.get(("after", kn))
        if not (a and b):
            continue
        print(f"  {kn:>3} kn  {b['trim']:.2f} {b['R']:.0f} | {a['trim']:.2f}"
              f" {a['R']:.0f} | {b['trim_sav']:.2f} {b['R_sav']:.0f}   "
              f"err b {b['trim']-b['trim_sav']:+.2f} "
              f"{b['R']/b['R_sav']-1:+.1%}  a {a['trim']-a['trim_sav']:+.2f}"
              f" {a['R']/a['R_sav']-1:+.1%}")
        band = "hi" if kn >= 20 else "lo"
        for nm, s in (("before", b), ("after", a)):
            c = cnt.setdefault((nm, band), [0, 0, 0])
            c[0] += abs(s["trim"] - s["trim_sav"]) <= 1.0
            c[1] += abs(s["R"] / s["R_sav"] - 1) <= 0.10
            c[2] += 1
    print("  counts (trim ok, R ok, n):", cnt)
    rows = [r for r in res["rows"] if isinstance(r, dict)]
    sc = fridsma_scores(rows)
    print("\nFRIDSMA 1969: 44 vs measured, 22 vs Zarnick, 8 resonant")
    for n in ("before", "after"):
        o = sc.get(n)
        if o:
            print(f"  {n:<7} {o['meas']}/{o['nmeas']}  {o['z']}/{o['nz']}  "
                  f"{o['zres']}/{o['nzres']}")
    from studies.exp_planing import FRIDSMA
    for c in ("A", "B"):
        for l in FRIDSMA[c]:
            k = f"{c}{l}"
            m = FRIDSMA[c][l]
            b = sc.get("before", {}).get("tab", {}).get(k)
            a = sc.get("after", {}).get("tab", {}).get(k)
            fmt = lambda v: ",".join(f"{x:.2f}" for x in v) if v else "nan"
            print(f"  {k:<5} meas {fmt(m)}  before {fmt(b)}  after {fmt(a)}")
    print("\nFRIDSMA 1971 mean positive peaks (CG, bow) g; Tp x0.8/1/1.25")
    f71 = [x for x in res["f71"] if isinstance(x, dict) and not x.get("bad")]
    by = {(x["name"], x["cond"], x["f"]): x for x in f71}
    cnt = {}
    from studies.exp_planing import FRIDSMA71
    for cond in FRIDSMA71:
        line = f"  {cond}: "
        for nm in ("before", "after"):
            vals = [by.get((nm, cond, f)) for f in (0.8, 1.0, 1.25)]
            line += f"{nm} CG " + "/".join(f"{v['cg']:.2f}" if v else "nan"
                                           for v in vals)
            line += " bow " + "/".join(f"{v['bow']:.2f}" if v else "nan"
                                       for v in vals) + " | "
            v = vals[1]
            if v:
                c = cnt.setdefault(nm, [0, 0])
                c[0] += (abs(v["cg"] / v["meas"][0] - 1) <= 0.3) + \
                    (abs(v["bow"] / v["meas"][1] - 1) <= 0.3)
                c[1] += 2
        print(line + f"meas {FRIDSMA71[cond][3]}/{FRIDSMA71[cond][4]}")
    print("  within +-30 % (nominal Tp):", cnt)
    print("\nCalm Fridsma (trim, rise/b, R/Delta):")
    for c in res["calm"]:
        if isinstance(c, dict):
            print(f"  {c['name']:<7} {c['case']} {c['trim']:.2f} "
                  f"{c['rise_b']:.3f} {c['R_D']:.3f}")
    print(f"\nwall {res['wall_s']:.0f} s")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("step1", "all"):
        step1()
    if what in ("step2", "all"):
        step2(sys.argv[2] if len(sys.argv) > 2 else "ZR")
    if what == "report1":
        with open(os.path.join(OUT, "step1.json")) as f:
            report1(json.load(f))
    if what == "report2":
        with open(os.path.join(OUT, f"step2_{sys.argv[2]}.json")) as f:
            report2(json.load(f))
