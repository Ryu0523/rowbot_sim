#!/usr/bin/env python3
"""
Step "进展 2" of learn/meta/SIM_REDESIGN_2026-10-08.md.

TASK 1  the Fridsma harness fixed (studies/exp_planing.fridsma_hull /
        fridsma71_hull, tow="cg": the towing force horizontal through the
        CG, as in Fridsma 1969/1971); ledger A3-A5 recomputed for v0, the
        as1 preset, ZR and ZR+MLM. Check H1: fixed v0 == step 1's
        'v0+tow' (ZarnickVessel tow="cg" on the old hull) to the bit.
TASK 2  why hull accelerations are low while the wedge force is high: one
        change at a time on ZR and ZR+MLM (registered in the plan document,
        "进展 2 / 事先登记"): (a) statistic definitions, (b) amplitude x
        shape-factor split, (c) crossflow drag x0 / x2, (d) chine-wetted
        section bound ('extend'), (e) time step 0.001 s, (f) surge free
        under a constant tow force (Fridsma 1969's carriage).

Run: python -m studies.exp_sim_redesign2 [h1|runs|report]
Results: studies/_cache/sim_redesign/step3.json
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
G = 9.81

from sim.planing_zarnick import ZOPT_V0, ZOPT_ZR  # noqa: E402

ZRM = dict(ZOPT_ZR, mlm=True)


def cfg(base, **kw):
    d = dict(base)
    d.update(kw)
    return d


# ------------------------------------------------------------ plants
def make_plant(kind, hull, sea, dt, captive_u):
    if kind == "v0":
        from sim.planing_vessel import PlaningVessel
        return PlaningVessel(hull, sea, dt=dt, captive_u=captive_u)
    if kind == "as1":
        from sim import physics
        from sim.planing_anyspeed import PlaningVesselAS
        return PlaningVesselAS(physics.variant(hull, "as1"), sea, dt=dt,
                               captive_u=captive_u)
    from sim.planing_zarnick import ZarnickVessel
    return ZarnickVessel(hull, sea, dt=dt, captive_u=captive_u,
                         zopt=dict(kind))


class _TowWeight:
    """Fridsma 1969's carriage: a constant horizontal force (falling weight)
    at the tow point, surge free. Stands in for the plant's waterjet."""

    def __init__(self, F):
        self.F = float(F)
        self.prime = 1.0
        self.t_max = abs(self.F)

    def forces(self, thr, rud, u):
        return self.F, 0.0

    def advance(self, s12, cmd, dt):
        return 0.0

    def update_prime(self, sub, dt):
        pass


def _hull(case, cdc):
    from studies.exp_planing import fridsma_hull
    h, V = fridsma_hull(case, tow="cg")
    if cdc != 1.0:
        h.c_dc = cdc * math.cos(math.radians(h.beta_deg))
    return h, V


# ------------------------------------------------------------ jobs
def rw_job(args):
    """Regular head waves (exp_planing.regular_wave's measurement), with
    extra outputs: bow acceleration normal to the keel, heave amplitude in
    m, encounter frequency; surge_free: constant tow force F (else held)."""
    name, kind, case, lam, o = args
    from sim.test_vessel import Monochromatic
    dt, T = o.get("dt", 0.004), o.get("T", 16.0)
    hull, V = _hull(case, o.get("cdc", 1.0))
    L = hull.L
    H = 0.111 * hull.b
    k = 2 * np.pi / (lam * L)
    a = 0.5 * H
    om = np.sqrt(G * k)
    sea = Monochromatic(om, a, theta0=np.pi)
    F = o.get("F")
    p = make_plant(kind, hull, sea, dt, None if F is not None else V)
    x_acc = hull.L - hull.lcg - 0.1 * hull.L
    s = np.zeros(14)
    s[6] = V
    s[2], s[4] = p.running_attitude(V)
    if F is not None:
        p.prop = _TowWeight(F)
    n = int(T / dt)
    rec = np.empty((n, 6))
    xb = p.x[-1]
    for i in range(n):
        s = p.step(s, i * dt, 0.0, 0.0)
        if F is None:
            s[6] = V
        a_cg, a_b = p.last_cg_acc, p.last_bow_acc
        ab = (a_cg + (a_b - a_cg) * x_acc / xb) / G
        rec[i] = (s[2], s[4], a_cg / G, ab,
                  ab * math.cos(p.Th_rest + s[4]), s[6])
    q = rec[n // 2:]
    if not np.all(np.isfinite(q)):
        return dict(name=name, case=case, lam=lam, bad=True)
    amp = 0.5 * (q[:, :2].max(0) - q[:, :2].min(0))
    ubar = float(q[:, 5].mean())
    om_e = om + k * ubar
    te = 2 * np.pi / om_e
    m = max(int(round(te / dt)), 1)
    cyc = [q[j:j + m] for j in range(0, len(q) - m + 1, m)]
    pk = np.mean([[c[:, 2].max(), c[:, 3].max(), c[:, 4].max()]
                  for c in cyc], axis=0)
    return dict(name=name, case=case, lam=lam,
                res=[float(amp[0] / a), float(amp[1] / (k * a)),
                     float(pk[0]), float(pk[1])],
                bow_n=float(pk[2]), heave_m=float(amp[0]), om_e=float(om_e),
                a=a, ubar=ubar, V=V, F=F)


def calm_job(args):
    name, kind, case, o = args
    from sim.test_vessel import Monochromatic
    hull, V = _hull(case, o.get("cdc", 1.0))
    p = make_plant(kind, hull, Monochromatic(1.0, 0.0), 0.004, None)
    cr = p.calm_resistance(V)
    return dict(name=name, case=case, trim=cr["trim_deg"],
                rise_b=cr["rise"] / hull.b, R_D=cr["R"] / (hull.mass * G),
                R=cr["R"])


def local_max_peaks(y):
    """Every local maximum above zero (the alternative reading of 'all
    positive peaks')."""
    i = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
                       & (y[1:-1] > 0)) + 1
    return y[i] if len(i) else np.zeros(1)


def f71_job(args):
    name, kind, cond, f, o = args
    from studies.exp_planing import FRIDSMA71, fridsma71_hull, excursion_peaks
    from sim.wavefield import SeaState
    dt, T = o.get("dt", 0.004), o.get("T", 60.0)
    h, V, hs = fridsma71_hull(cond, tow="cg")
    if o.get("cdc", 1.0) != 1.0:
        h.c_dc = o["cdc"] * math.cos(math.radians(h.beta_deg))
    tp = f * 15.63 * np.sqrt(hs / G)
    sea = SeaState(hs, tp, gamma=1.0, theta0=np.pi, n_freq=40, n_dir=1,
                   seed=cond + 1000 * o.get("seed_k", 0))
    p = make_plant(kind, h, sea, dt, V)
    s = np.zeros(14)
    s[6] = V
    s[2], s[4] = p.running_attitude(V)
    xa = h.L - h.lcg - 0.1 * h.L
    n = int(T / dt)
    acc = np.empty((n, 3))
    for i in range(n):
        s = p.step(s, i * dt, 0.0, 0.0)
        s[6] = V
        ac, ab = p.last_cg_acc, p.last_bow_acc
        bw = (ac + (ab - ac) * xa / p.x[-1]) / G
        acc[i] = ac / G, bw, bw * math.cos(p.Th_rest + s[4])
    acc = acc[n // 10:]
    if not np.all(np.isfinite(acc)):
        return dict(name=name, cond=cond, f=f, bad=True)
    m = FRIDSMA71[cond]
    return dict(name=name, cond=cond, f=f,
                cg=float(excursion_peaks(acc[:, 0]).mean()),
                bow=float(excursion_peaks(acc[:, 1]).mean()),
                bow_n=float(excursion_peaks(acc[:, 2]).mean()),
                cg_lm=float(local_max_peaks(acc[:, 0]).mean()),
                bow_lm=float(local_max_peaks(acc[:, 1]).mean()),
                n_exc=int(len(excursion_peaks(acc[:, 0]))),
                n_lm=int(len(local_max_peaks(acc[:, 0]))),
                meas=[m[3], m[4]])


def sav_job(args):
    name, kind, kn, o = args
    from hydro.planing import savitsky
    from sim import config
    from sim.test_vessel import Monochromatic
    h, _ = config.load("scarab195")
    if o.get("cdc", 1.0) != 1.0:
        h.c_dc = o["cdc"] * math.cos(math.radians(h.beta_deg))
    p = make_plant(kind, h, Monochromatic(1.0, 0.0), 0.02, None)
    r = p.calm_resistance(kn * KN)
    sv = savitsky(kn * KN, h.mass, h.lcg, h.b, h.beta_deg)
    return dict(name=name, kn=kn, trim=r["trim_deg"], R=r["R"] - r["air"],
                trim_sav=sv["tau_deg"], R_sav=sv["R"])


# ------------------------------------------------------------ wedge
def wedge(cdc=1.0, chine="frozen"):
    """Strip force on the 30 deg drop section with the measured V(t)
    (exp_zarnick_mlm.wedge_drop's strip), Wagner and MLM laws, chine rule
    'frozen' (Zarnick) or 'extend' (bound d)."""
    from studies.exp_slam_wedge import T_F, F_MEAS, T_V, V_MEAS, HALF_B, \
        L_MEAS, RHO
    from sim.planing_zarnick import _mlm_constants
    beta = 30.0
    Tn = math.tan(math.radians(beta))
    l = HALF_B
    Kv, Km, _ = _mlm_constants(beta)
    laws = {"wagner": (math.pi ** 2 / (2 * Tn), math.pi / 2), "mlm": (Kv, Km)}
    cdc_v = cdc * math.cos(math.radians(beta))
    t = np.linspace(0.0, 0.025, 2501)
    v = np.interp(t, T_V, V_MEAS)
    h = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1])
                                         * np.diff(t))])
    vd = np.gradient(v, t)
    out = {}
    for law, (kv, km) in laws.items():
        F = []
        for hh, vv, aa in zip(h, v, vd):
            cw = math.pi * hh / (2 * Tn)
            c = min(cw, l)
            ce = cw if chine == "extend" else c
            dry = cw < l or chine == "extend"
            mom = kv * RHO * ce * vv * vv if (dry and vv > 0) else 0.0
            acc = km * RHO * ce * ce * aa
            cf = 0.5 * RHO * cdc_v * 2 * c * vv * abs(vv)
            dch = l * Tn
            A = min(hh, dch) ** 2 / Tn + 2 * l * max(hh - dch, 0.0)
            F.append((mom + acc + cf + RHO * G * A) * L_MEAS)
        F = np.array(F)
        early = (t > 0.002) & (t < 0.0075)
        late = (t >= 0.0175) & (t <= 0.025)
        fm_e = np.interp(t[early], T_F, F_MEAS)
        fm_l = np.interp(t[late], T_F, F_MEAS)
        out[law] = dict(
            early_ratio=float(np.mean(F[early] / fm_e)),
            late_ratio=float(np.mean(F[late] / fm_l)),
            peak=float(F.max()),
            at_ms={str(tt): float(np.interp(tt / 1e3, t, F))
                   for tt in (2.5, 5, 7.5, 10, 15, 17.5, 20, 25)})
    out["measured"] = {str(tt): float(np.interp(tt / 1e3, T_F, F_MEAS))
                       for tt in (2.5, 5, 7.5, 10, 15, 17.5, 20, 25)}
    return out


# ------------------------------------------------------------ H1
def h1_check():
    """Fixed-harness v0 == ZarnickVessel(tow='cg') on the old hull, bitwise."""
    from studies.exp_planing import fridsma_hull
    from sim.planing_vessel import PlaningVessel
    from sim.planing_zarnick import ZarnickVessel
    from sim.test_vessel import Monochromatic
    out = {}
    for case in ("A", "B"):
        trajs = []
        for which in ("fixed_v0", "old_hull_ZarnickVessel_tow_cg"):
            if which == "fixed_v0":
                hull, V = fridsma_hull(case, tow="cg")
            else:
                hull, V = fridsma_hull(case, tow="nozzle")
            k = 2 * np.pi / (3.0 * hull.L)
            sea = Monochromatic(np.sqrt(G * k), 0.0555 * hull.b,
                                theta0=np.pi)
            if which == "fixed_v0":
                p = PlaningVessel(hull, sea, dt=0.004, captive_u=V)
            else:
                p = ZarnickVessel(hull, sea, dt=0.004, captive_u=V,
                                  zopt=dict(ZOPT_V0, tow="cg"))
            s = np.zeros(14)
            s[6] = V
            s[2], s[4] = p.running_attitude(V)
            rec = []
            for i in range(2000):
                s = p.step(s, i * 0.004, 0.0, 0.0)
                s[6] = V
                rec.append(np.r_[s, p.last_cg_acc, p.last_bow_acc])
            trajs.append(np.array(rec))
        out[case] = float(np.abs(trajs[0] - trajs[1]).max())
    return out


# ------------------------------------------------------------ runs
BASES = {"v0": "v0", "as1": "as1", "ZR": ZOPT_ZR, "ZRM": ZRM}
VARIANTS = {   # task 2, one change at a time on ZR and ZR+MLM
    "ZR_cdc0": (ZOPT_ZR, dict(cdc=0.0)),
    "ZR_cdc2": (ZOPT_ZR, dict(cdc=2.0)),
    "ZRM_cdc0": (ZRM, dict(cdc=0.0)),
    "ZRM_cdc2": (ZRM, dict(cdc=2.0)),
    "ZR_ext": (cfg(ZOPT_ZR, chine="extend"), {}),
    "ZRM_ext": (cfg(ZRM, chine="extend"), {}),
    "ZR_dt1": (ZOPT_ZR, dict(dt=0.001)),
    "ZRM_dt1": (ZRM, dict(dt=0.001)),
}


def runs():
    from studies.parallel_runner import run_parallel
    from studies.exp_planing import FRIDSMA, FRIDSMA71
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    res = dict(h1=h1_check())
    print("H1:", res["h1"], f"{time.time() - t0:.0f} s", flush=True)
    res["wedge"] = {f"cdc{c}_{ch}": wedge(c, ch) for c in (0.0, 1.0, 2.0)
                    for ch in ("frozen", "extend")}
    jobs_rw, jobs_calm, jobs_71, jobs_sav = [], [], [], []
    for n, kind in BASES.items():
        for c in "AB":
            jobs_calm.append((n, kind, c, {}))
            for lam in FRIDSMA[c]:
                jobs_rw.append((n, kind, c, lam, {}))
        for cond in FRIDSMA71:
            for f in (0.8, 1.0, 1.25):
                jobs_71.append((n, kind, cond, f, {}))
    for n, (kind, o) in VARIANTS.items():
        for c in "AB":
            if "dt" not in o:
                jobs_calm.append((n, kind, c, o))
            for lam in FRIDSMA[c]:
                jobs_rw.append((n, kind, c, lam, o))
        for cond in FRIDSMA71:
            jobs_71.append((n, kind, cond, 1.0, o))
        if "dt" not in o:
            for kn in (10, 12, 14, 16, 20, 25, 30, 35, 40):
                jobs_sav.append((n, kind, kn, o))
    for n, kind in (("ZR", ZOPT_ZR), ("ZRM", ZRM)):
        for kn in (10, 12, 14, 16, 20, 25, 30, 35, 40):
            jobs_sav.append((n, kind, kn, {}))
    keep = lambda L: [x if isinstance(x, dict) else repr(x) for x in L]
    # long jobs first (dt 0.001, as1)
    order = lambda js: sorted(js, key=lambda j: -(j[-1].get("dt", 0.004) == 0.001
                                                    or j[1] == "as1"))
    res["rw"] = keep(run_parallel(rw_job, order(jobs_rw), gb_per_job=0.2,
                                  reserve_gb=3.0))
    print("rw done", f"{time.time() - t0:.0f} s", flush=True)
    res["calm"] = keep(run_parallel(calm_job, jobs_calm, gb_per_job=0.2,
                                    reserve_gb=3.0))
    res["f71"] = keep(run_parallel(f71_job, order(jobs_71), gb_per_job=0.2,
                                   reserve_gb=3.0))
    print("f71 done", f"{time.time() - t0:.0f} s", flush=True)
    res["sav"] = keep(run_parallel(sav_job, jobs_sav, gb_per_job=0.2,
                                   reserve_gb=3.0))
    res["wall_runs_s"] = time.time() - t0
    with open(os.path.join(OUT, "step3.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("runs done", f"{time.time() - t0:.0f} s", flush=True)
    # (f) surge free under a constant tow force, ZR and ZRM: secant on F so
    # that the mean speed is within 1 % of V
    res["surge"] = surge_free(res)
    res["wall_s"] = time.time() - t0
    with open(os.path.join(OUT, "step3.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("all done", f"{time.time() - t0:.0f} s", flush=True)


def surge_free(res):
    from studies.parallel_runner import run_parallel
    from studies.exp_planing import FRIDSMA
    calm = {(c["name"], c["case"]): c for c in res["calm"]
            if isinstance(c, dict)}
    held = {(r["name"], r["case"], r["lam"]): r for r in res["rw"]
            if isinstance(r, dict) and "res" in r}
    out = []
    for n, kind in (("ZR", ZOPT_ZR), ("ZRM", ZRM)):
        cases = [(c, lam) for c in "AB" for lam in FRIDSMA[c]]
        # first guess: calm resistance; second: +10 %
        F0 = {cl: calm[(n, cl[0])]["R"] for cl in cases}
        hist = {cl: [] for cl in cases}
        F = dict(F0)
        for it in range(5):
            todo = [cl for cl in cases if not hist[cl] or
                    abs(hist[cl][-1][1] / hist[cl][-1][2] - 1) > 0.01]
            if not todo:
                break
            jobs = [(f"{n}_surge", kind, c, lam, dict(F=F[(c, lam)]))
                    for c, lam in todo]
            rr = run_parallel(rw_job, jobs, gb_per_job=0.2, reserve_gb=3.0)
            for (c, lam), r in zip(todo, rr):
                if not isinstance(r, dict) or r.get("bad"):
                    hist[(c, lam)].append((F[(c, lam)], float("nan"), 1.0,
                                           None))
                    continue
                hist[(c, lam)].append((F[(c, lam)], r["ubar"], r["V"], r))
                hh = hist[(c, lam)]
                if len(hh) == 1:
                    F[(c, lam)] = hh[-1][0] * (r["V"] / r["ubar"]) ** 2
                else:
                    (f1, u1, *_), (f2, u2, *_) = hh[-2], hh[-1]
                    F[(c, lam)] = f2 + (r["V"] - u2) * (f2 - f1) / (
                        (u2 - u1) if abs(u2 - u1) > 1e-6 else 1e-6)
            print(f"surge {n} iter {it}: {len(todo)} cases", flush=True)
        for cl in cases:
            last = hist[cl][-1]
            r = last[3]
            if r is None:
                continue
            r = dict(r)
            r["name"] = f"{n}_surge"
            r["iters"] = len(hist[cl])
            r["speed_err"] = last[1] / last[2] - 1
            r["held"] = held.get((n, cl[0], cl[1]), {}).get("res")
            out.append(r)
    return out


# ------------------------------------------------------------ report
def report(res):
    from studies.exp_planing import FRIDSMA, ZARNICK, FRIDSMA71
    print("H1 max |fixed v0 - ZarnickVessel(tow=cg, old hull)|:", res["h1"])
    rw = [r for r in res["rw"] if isinstance(r, dict) and "res" in r]
    rw += [r for r in res.get("surge", []) if "res" in r]
    tab = {}
    extra = {}
    for r in rw:
        tab.setdefault(r["name"], {})[(r["case"], r["lam"])] = r["res"]
        extra.setdefault(r["name"], {})[(r["case"], r["lam"])] = r
    keys = [(c, l) for c in "AB" for l in FRIDSMA[c]]

    def score(name):
        t = tab.get(name, {})
        sm = sz = sr = nacc = 0
        for (c, l) in keys:
            v = t.get((c, l))
            if v is None:
                continue
            m = FRIDSMA[c][l]
            z = ZARNICK[c][l]
            for j, tol in enumerate((.2, .2, .3, .3)):
                ok = abs(v[j] / m[j] - 1) <= tol
                sm += ok
                if j >= 2:
                    nacc += ok
            for j in (0, 1):
                ok = abs(v[j] - z[j]) <= 0.10 * z[j] + 0.03
                sz += ok
                sr += ok and l in (3.0, 4.0)
        return sm, nacc, sz, sr, len(t)

    names = list(BASES) + list(VARIANTS) + ["ZR_surge", "ZRM_surge"]
    print("\nF69 counts: name | 44 vs measured | of which 22 acc | 22 vs "
          "Zarnick | 8 resonant | n cases")
    for n in names:
        if n in tab:
            print(f"  {n:<10} {score(n)}")
    print("\nF69 per case: heave, pitch, CG g, bow g")
    print("  " + "case".ljust(6) + "meas".rjust(24) + "".join(
        n.rjust(24) for n in names if n in tab))
    for k in keys:
        line = f"  {k[0]}{k[1]:<5}" + ",".join(
            f"{x:.2f}" for x in FRIDSMA[k[0]][k[1]]).rjust(24)
        for n in names:
            if n in tab:
                v = tab[n].get(k)
                line += (",".join(f"{x:.2f}" for x in v) if v else
                         "nan").rjust(24)
        print(line)
    # acceleration ratio medians
    print("\nF69 acceleration sim/measured: median over 11 cases (CG; bow); "
          "median over cases where ZR+MLM is low (<0.85)")
    base_low = [k for k in keys if k in tab.get("ZRM", {}) and
                tab["ZRM"][k][2] / FRIDSMA[k[0]][k[1]][2] < 0.85]
    base_low_b = [k for k in keys if k in tab.get("ZRM", {}) and
                  tab["ZRM"][k][3] / FRIDSMA[k[0]][k[1]][3] < 0.85]
    print("  low CG cases:", base_low, " low bow cases:", base_low_b)
    for n in names:
        if n not in tab:
            continue
        rc = [tab[n][k][2] / FRIDSMA[k[0]][k[1]][2] for k in keys
              if k in tab[n]]
        rb = [tab[n][k][3] / FRIDSMA[k[0]][k[1]][3] for k in keys
              if k in tab[n]]
        rcl = [tab[n][k][2] / FRIDSMA[k[0]][k[1]][2] for k in base_low
               if k in tab[n]]
        rbl = [tab[n][k][3] / FRIDSMA[k[0]][k[1]][3] for k in base_low_b
               if k in tab[n]]
        print(f"  {n:<10} all {np.median(rc):.2f}; {np.median(rb):.2f}   "
              f"low {np.median(rcl) if rcl else float('nan'):.2f}; "
              f"{np.median(rbl) if rbl else float('nan'):.2f}")
    # (a) bow normal to keel
    print("\n(a) bow normal-to-keel / vertical, F69 (max |ratio-1| per cfg):")
    for n in names:
        if n in extra:
            d = [extra[n][k]["bow_n"] / extra[n][k]["res"][3] - 1
                 for k in keys if k in extra[n]]
            print(f"  {n:<10} {max(abs(x) for x in d):.4f}")
    # (b) shape factors
    print("\n(b) CG shape factor S = peak / (w_e^2 x heave amplitude); "
          "measured from measured heave and acc; acc ratio = heave ratio x "
          "S ratio")
    for n in ("ZR", "ZRM", "v0", "as1"):
        if n not in extra:
            continue
        print(f"  {n}")
        for k in keys:
            r = extra[n].get(k)
            if not r:
                continue
            m = FRIDSMA[k[0]][k[1]]
            ome2 = r["om_e"] ** 2
            S_sim = r["res"][2] * G / (ome2 * r["heave_m"])
            S_meas = m[2] * G / (ome2 * m[0] * r["a"])
            print(f"    {k[0]}{k[1]:<4} S meas {S_meas:5.2f} sim {S_sim:5.2f}"
                  f" | acc ratio {r['res'][2] / m[2]:.2f} = heave "
                  f"{r['res'][0] / m[0]:.2f} x S {S_sim / S_meas:.2f}")
    # calm
    print("\nCalm Fridsma (trim, rise/b, R/Delta); measured A 4.0/0.05/0.158,"
          " B 4.0/0.13/0.206")
    for c in res["calm"]:
        if isinstance(c, dict):
            print(f"  {c['name']:<10} {c['case']} {c['trim']:.2f} "
                  f"{c['rise_b']:.3f} {c['R_D']:.3f}")
        else:
            print("  ", c[:200])
    # F71
    f71 = [x for x in res["f71"] if isinstance(x, dict) and not x.get("bad")]
    by = {(x["name"], x["cond"], x["f"]): x for x in f71}
    print("\nF71 mean positive peaks CG / bow (g), nominal Tp [x0.8, x1.25]"
          "; count within +-30 % (nominal); 'lm' = all local maxima > 0")
    f71names = [n for n in names if any(k[0] == n for k in by)]
    for n in f71names:
        cnt = 0
        line = f"  {n:<10}"
        for cond in FRIDSMA71:
            v = by.get((n, cond, 1.0))
            if not v:
                line += "   nan"
                continue
            cnt += (abs(v["cg"] / v["meas"][0] - 1) <= .3) + \
                (abs(v["bow"] / v["meas"][1] - 1) <= .3)
            lo, hi = by.get((n, cond, 0.8)), by.get((n, cond, 1.25))
            rng = (f"[{lo['cg']:.2f}-{hi['cg']:.2f}]" if lo and hi else "")
            line += f" {cond}:{v['cg']:.2f}/{v['bow']:.2f}{rng}"
        print(line + f"  -> {cnt}/10")
    print("  measured:  " + " ".join(f"{c}:{v[3]}/{v[4]}"
                                    for c, v in FRIDSMA71.items()))
    print("  (a) lm/excursion and bow_n/bow ratios (nominal Tp):")
    for n in ("ZR", "ZRM", "v0", "as1"):
        rr = [(by[(n, c, 1.0)]["cg_lm"] / by[(n, c, 1.0)]["cg"],
               by[(n, c, 1.0)]["bow_lm"] / by[(n, c, 1.0)]["bow"],
               by[(n, c, 1.0)]["bow_n"] / by[(n, c, 1.0)]["bow"],
               by[(n, c, 1.0)]["n_exc"], by[(n, c, 1.0)]["n_lm"])
              for c in FRIDSMA71 if (n, c, 1.0) in by]
        print(f"   {n:<6} " + "  ".join(f"{a:.2f}/{b:.2f}/{c:.3f} n{d}/{e}"
                                       for a, b, c, d, e in rr))
    # Savitsky
    sav = [s for s in res["sav"] if isinstance(s, dict)]
    print("\nSavitsky scarab195: trim err deg, R err % (10..40 kn); counts "
          "trim ok/R ok for 12-16 and 20-40 kn")
    sn = []
    for s in sav:
        if s["name"] not in sn:
            sn.append(s["name"])
    for n in sn:
        rows = sorted([s for s in sav if s["name"] == n],
                      key=lambda s: s["kn"])
        lo = [s for s in rows if 12 <= s["kn"] <= 16]
        hi = [s for s in rows if s["kn"] >= 20]
        c = lambda L: (sum(abs(s["trim"] - s["trim_sav"]) <= 1 for s in L),
                       sum(abs(s["R"] / s["R_sav"] - 1) <= .1 for s in L))
        print(f"  {n:<10} " + " ".join(
            f"{s['kn']}:{s['trim'] - s['trim_sav']:+.2f},"
            f"{100 * (s['R'] / s['R_sav'] - 1):+.0f}" for s in rows)
            + f"  lo {c(lo)} hi {c(hi)}")
    # wedge
    print("\nWedge 30 deg strip (measured V): early 2-7.5 ms ratio, late "
          "17.5-25 ms ratio, peak N; values at 15/17.5/20/25 ms")
    w = res["wedge"]
    meas = list(w.values())[0]["measured"]
    print("  measured  " + " ".join(f"{meas[t]:.0f}" for t in
                                    ("15", "17.5", "20", "25")))
    for key, v in w.items():
        for law in ("wagner", "mlm"):
            x = v[law]
            print(f"  {key:<14} {law:<6} early {x['early_ratio']:.2f} late "
                  f"{x['late_ratio']:.2f} peak {x['peak']:.0f}  " +
                  " ".join(f"{x['at_ms'][t]:.0f}" for t in
                           ("15", "17.5", "20", "25")))
    # surge
    if res.get("surge"):
        print("\n(f) surge free vs held: speed err, heave/pitch/CG/bow free "
              "vs held")
        for r in res["surge"]:
            h = r.get("held")
            print(f"  {r['name']:<10} {r['case']}{r['lam']:<4} it {r['iters']}"
                  f" err {100 * r['speed_err']:+.2f}%  free " +
                  ",".join(f"{x:.2f}" for x in r["res"]) + "  held " +
                  (",".join(f"{x:.2f}" for x in h) if h else "nan"))
    print(f"\nwall {res.get('wall_s', res.get('wall_runs_s', 0)):.0f} s")


def seeds():
    """Sampling noise of the F71 statistic: nominal Tp, wave seeds 0-3
    (seed 0 = the runs above), v0 / as1 / ZR / ZRM; ZR and ZRM also at
    dt 0.001 with seeds 1-2 (is the dt effect larger than the noise?)."""
    from studies.parallel_runner import run_parallel
    from studies.exp_planing import FRIDSMA71
    jobs = [(n, k, c, 1.0, dict(seed_k=sk)) for n, k in BASES.items()
            for c in FRIDSMA71 for sk in (1, 2, 3)]
    jobs += [(n + "_dt1", k, c, 1.0, dict(seed_k=sk, dt=0.001))
             for n, k in (("ZR", ZOPT_ZR), ("ZRM", ZRM))
             for c in FRIDSMA71 for sk in (1, 2)]
    jobs.sort(key=lambda j: -(j[-1].get("dt", 0.004) == 0.001))
    t0 = time.time()
    rr = run_parallel(f71_job, jobs, gb_per_job=0.2, reserve_gb=3.0)
    out = [dict(r, seed_k=j[-1]["seed_k"]) if isinstance(r, dict)
           else dict(name=j[0], cond=j[2], seed_k=j[-1]["seed_k"],
                     err=repr(r)) for j, r in zip(jobs, rr)]
    with open(os.path.join(OUT, "step3_seeds.json"), "w") as f:
        json.dump(dict(rows=out, wall_s=time.time() - t0), f, indent=1)
    print("seeds done", f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "runs"
    if what == "h1":
        print(h1_check())
    elif what == "runs":
        runs()
    elif what == "seeds":
        seeds()
    if what in ("runs", "report"):
        with open(os.path.join(OUT, "step3.json")) as f:
            report(json.load(f))
