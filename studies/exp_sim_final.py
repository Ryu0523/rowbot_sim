#!/usr/bin/env python3
"""
The cells missing from the error table of learn/meta/SIM_FINAL_2026-10-09.md
(the frozen simulator configuration), run with the existing harness only --
no plant code changes. Configurations: v0 (sim/planing_vessel.py), ZR and
ZR+MLM (sim/planing_zarnick.py presets), as1 (sim/planing_anyspeed.py
preset 'as1'); scarab195 for the vessel checks, Fridsma's model for the
wave checks (studies/exp_planing.py, tow="cg").

Parts (each writes its own JSON to studies/_cache/sim_final/):
  sav    calm trim and resistance vs Savitsky, 10-40 kn, v0 and as1 (ZR and
         ZR+MLM are in studies/_cache/sim_redesign/step3.json already)
  rest   scarab195 at rest: 60 s zero-thrust drift; heave free decay from
         +5 cm (period, damping ratio); regular head-wave response at 7 of
         the BEM probe's frequencies (anyspeed_rest_rao.py's protocol)
  calm   captive calm runs at 3-30 kn from the running attitude + 5 cm:
         peak-to-peak of heave and trim over the last two 10 s windows
         (limit cycles, missing equilibria)
  f69    Fridsma 1969 regular head waves at model dt 0.00336 s (full-scale
         0.01 s, the step V6 found adequate for filtered means): cycle-peak
         means of the CG and bow accelerations, raw and after the V6 output
         filter (2nd-order Butterworth, filtfilt, 20 Hz full scale = 59.5 Hz
         model scale)
  f71    Fridsma 1971 irregular head seas, wave seed 0 (= cond), nominal
         Tp, model dt 0.00168 s (full-scale 0.005 s, V6's step for A1/10):
         mean of the positive excursion peaks and A1/10, raw and filtered

Run: python -m studies.exp_sim_final [all|sav|rest|calm|f69|f71|report]
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
OUT = os.path.join(HERE, "_cache", "sim_final")
KN = 0.514444
G = 9.81

from sim.planing_zarnick import ZOPT_ZR  # noqa: E402
from studies.exp_sim_redesign2 import make_plant, ZRM  # noqa: E402

BASES = {"v0": "v0", "ZR": ZOPT_ZR, "ZRM": ZRM, "as1": "as1"}
# model-scale time x T_SCALE = full-scale time (Froude, by the chine beams)
B_FULL, B_MODEL = 0.83 * 2.44, 9 * 0.0254
T_SCALE = math.sqrt(B_FULL / B_MODEL)          # 2.976
FC_FULL = 20.0                                  # Hz, V6's output filter
RAO_TARGETS = (1.5, 2.5, 3.2, 3.9, 4.7, 5.6, 7.3)
POOL = dict(gb_per_job=0.2, reserve_gb=2.5, max_procs=6)


def butter2(y, dt, fc):
    """2nd-order Butterworth low-pass, zero phase (filtfilt), cutoff fc Hz."""
    from scipy.signal import butter, filtfilt
    b, a = butter(2, fc / (0.5 / dt))
    return filtfilt(b, a, y)


def _scarab():
    from sim import config
    h, _ = config.load("scarab195")
    return h


# ------------------------------------------------------------- rest
def drift_job(args):
    name, kind = args
    from sim.test_vessel import Monochromatic
    dt = 0.02
    p = make_plant(kind, _scarab(), Monochromatic(1.0, 0.0), dt, None)
    s = p.initial_state(0.0)
    n = int(60.0 / dt)
    umax = 0.0
    for i in range(n):
        s = p.step(s, i * dt, 0.0, 0.0)
        umax = max(umax, abs(float(s[6])))
    return dict(name=name, u_end=float(s[6]), x_end=float(s[0]), umax=umax,
                z_end=float(s[2]),
                trim_end=float(np.degrees(-(p.Th_rest + s[4]))))


def restdecay_job(args):
    name, kind = args
    from sim.test_vessel import Monochromatic
    dt = 0.005
    p = make_plant(kind, _scarab(), Monochromatic(1.0, 0.0), dt, 0.0)
    s = np.zeros(14)
    s[2] = 0.05
    n = int(30.0 / dt)
    z = np.empty(n)
    for i in range(n):
        s = p.step(s, i * dt, 0.0, 0.0)
        s[6] = 0.0
        z[i] = s[2]
    t = np.arange(n) * dt
    i = np.flatnonzero((z[1:-1] > z[:-2]) & (z[1:-1] >= z[2:])
                       & (z[1:-1] > 0)) + 1
    pk, tp = z[i], t[i]
    out = dict(name=name, n_peaks=int(len(pk)),
               peaks=[float(v) for v in pk[:6]])
    if len(pk) >= 3:
        per = np.diff(tp[:4]).mean() if len(tp) >= 4 else np.diff(tp).mean()
        dec = np.log(pk[:-1] / pk[1:])[:3]
        zeta = dec / np.sqrt(4 * np.pi ** 2 + dec ** 2)
        out.update(period=float(per), wn=float(2 * np.pi / per),
                   zeta=float(zeta.mean()), zeta_first=float(zeta[0]))
    return out


def rao_job(args):
    name, kind, w = args
    from sim.test_vessel import Monochromatic
    from sim.cummins import extract_amplitude_phase
    a, dt = 0.02, 0.005
    k = w * w / G
    sea = Monochromatic(w, a, theta0=np.pi)
    p = make_plant(kind, _scarab(), sea, dt, 0.0)
    s = np.zeros(14)
    T = 2 * np.pi / w
    n = int(40 * T / dt)
    t = np.arange(n) * dt
    rec = np.empty((n, 2))
    for i in range(n):
        s = p.step(s, t[i], 0.0, 0.0)
        s[6] = 0.0
        rec[i] = s[2], s[4]
    if not np.all(np.isfinite(rec)):
        return dict(name=name, w=w, bad=True)
    hh, _ = extract_amplitude_phase(t, rec[:, 0], w, n_last_periods=10)
    pp, _ = extract_amplitude_phase(t, rec[:, 1], w, n_last_periods=10)
    return dict(name=name, w=float(w), lam_L=float(2 * np.pi / k / p.L),
                heave=float(hh / a), pitch=float(pp / (k * a)))


def rest():
    from studies.parallel_runner import run_parallel
    z = np.load(os.path.join(HERE, "_cache", "anyspeed",
                             "bem_probe_lid_24_64.npz"))
    om = z["omega"]
    ws = [float(om[int(np.argmin(np.abs(om - w)))]) for w in RAO_TARGETS]
    jobs_rao = sorted([(n, k, w) for n, k in BASES.items() for w in ws],
                      key=lambda j: (j[1] != "as1", j[2]))
    t0 = time.time()
    res = dict(omega=ws)
    res["drift"] = _keep(run_parallel(drift_job, list(BASES.items()), **POOL))
    res["decay"] = _keep(run_parallel(restdecay_job, list(BASES.items()),
                                      **POOL))
    res["rao"] = _keep(run_parallel(rao_job, jobs_rao, **POOL))
    res["wall_s"] = time.time() - t0
    _save("rest.json", res)
    return res


# ------------------------------------------------------------- calm captive
CALM_KN = (3, 6, 9, 10, 12, 26, 28, 30)


def calm_job(args):
    name, kind, kn = args
    from sim.test_vessel import Monochromatic
    V, dt = kn * KN, 0.005
    p = make_plant(kind, _scarab(), Monochromatic(1.0, 0.0), dt, V)
    z0, th0 = p.running_attitude(V)
    s = np.zeros(14)
    s[6], s[2], s[4] = V, z0 + 0.05, th0
    n = int(60.0 / dt)
    rec = np.empty((n, 2))
    for i in range(n):
        s = p.step(s, i * dt, 0.0, 0.0)
        s[6] = V
        rec[i] = s[2], np.degrees(-(p.Th_rest + s[4]))
    if not np.all(np.isfinite(rec)):
        return dict(name=name, kn=kn, bad=True)
    m = int(10.0 / dt)
    w2, w1 = rec[-m:], rec[-2 * m:-m]
    pp = lambda q: (q.max(0) - q.min(0))
    p2, p1 = pp(w2), pp(w1)
    return dict(name=name, kn=kn, z_mean=float(w2[:, 0].mean()),
                trim_mean=float(w2[:, 1].mean()),
                zpp_last=float(p2[0]), trimpp_last=float(p2[1]),
                zpp_prev=float(p1[0]), trimpp_prev=float(p1[1]),
                z_attitude=float(z0),
                trim_attitude=float(np.degrees(-(p.Th_rest + th0))))


def calm():
    from studies.parallel_runner import run_parallel
    jobs = [(n, k, kn) for n, k in BASES.items() for kn in CALM_KN
            if n != "as1" or kn in (9, 10, 12, 28)]
    jobs.sort(key=lambda j: j[1] != "as1")
    t0 = time.time()
    res = dict(rows=_keep(run_parallel(calm_job, jobs, **POOL)),
               wall_s=time.time() - t0)
    _save("calm.json", res)
    return res


# ------------------------------------------------------------- Savitsky
def sav():
    from studies.parallel_runner import run_parallel
    from studies.exp_sim_redesign2 import sav_job
    jobs = [("as1", "as1", kn, {}) for kn in
            (10, 12, 14, 16, 20, 25, 30, 35, 40)]
    jobs += [("v0", "v0", kn, {}) for kn in (10, 12, 14, 16, 20, 25, 30, 35,
                                             40)]
    t0 = time.time()
    res = dict(rows=_keep(run_parallel(sav_job, jobs, **POOL)),
               wall_s=time.time() - t0)
    _save("sav.json", res)
    return res


# ------------------------------------------------------------- Fridsma 1969
def f69_job(args):
    name, kind, case, lam, o = args
    from sim.test_vessel import Monochromatic
    from studies.exp_planing import fridsma_hull
    dt, T = o.get("dt", 0.01 / T_SCALE), 16.0          # 0.00336 default
    hull, V = fridsma_hull(case, tow="cg")
    L = hull.L
    H = 0.111 * hull.b
    k = 2 * np.pi / (lam * L)
    a = 0.5 * H
    om = np.sqrt(G * k)
    sea = Monochromatic(om, a, theta0=np.pi)
    p = make_plant(kind, hull, sea, dt, V)
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
        rec[i] = (s[2], s[4], a_cg / G,
                  (a_cg + (a_b - a_cg) * x_acc / xb) / G)
    if not np.all(np.isfinite(rec)):
        return dict(name=name, case=case, lam=lam, bad=True)
    fc = FC_FULL * T_SCALE
    filt = np.column_stack([butter2(rec[:, 2], dt, fc),
                            butter2(rec[:, 3], dt, fc)])
    q, qf = rec[n // 2:], filt[n // 2:]
    amp = 0.5 * (q[:, :2].max(0) - q[:, :2].min(0))
    om_e = om + k * V
    m = max(int(round(2 * np.pi / om_e / dt)), 1)
    cyc = range(0, len(q) - m + 1, m)
    pk = np.mean([[q[j:j + m, 2].max(), q[j:j + m, 3].max()] for j in cyc],
                 axis=0)
    pkf = np.mean([[qf[j:j + m, 0].max(), qf[j:j + m, 1].max()]
                   for j in cyc], axis=0)
    return dict(name=name, case=case, lam=lam, dt=dt, fc_model_hz=fc,
                res=[float(amp[0] / a), float(amp[1] / (k * a)),
                     float(pk[0]), float(pk[1])],
                filt=[float(pkf[0]), float(pkf[1])])


def f69():
    from studies.parallel_runner import run_parallel
    from studies.exp_planing import FRIDSMA
    jobs = [(n, k, c, lam, {}) for n, k in BASES.items() for c in "AB"
            for lam in FRIDSMA[c]]
    jobs.sort(key=lambda j: j[1] != "as1")
    t0 = time.time()
    res = dict(rows=_keep(run_parallel(f69_job, jobs, **POOL)),
               wall_s=time.time() - t0)
    _save("f69.json", res)
    return res


# ------------------------------------------------------------- Fridsma 1971
def _pkstats(y):
    from studies.exp_planing import excursion_peaks
    pk = np.sort(excursion_peaks(y))[::-1]
    n = len(pk)
    return dict(mean=float(pk.mean()), a110=float(pk[:max(1, math.ceil(
        n / 10))].mean()), amax=float(pk[0]), n=int(n))


def f71_job(args):
    name, kind, cond, o = args
    from studies.exp_planing import FRIDSMA71, fridsma71_hull
    from sim.wavefield import SeaState
    dt, T = o.get("dt", 0.005 / T_SCALE), 60.0        # 0.00168 default
    h, V, hs = fridsma71_hull(cond, tow="cg")
    tp = 15.63 * np.sqrt(hs / G)
    sea = SeaState(hs, tp, gamma=1.0, theta0=np.pi, n_freq=40, n_dir=1,
                   seed=cond)
    p = make_plant(kind, h, sea, dt, V)
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
        acc[i] = ac / G, (ac + (ab - ac) * xa / p.x[-1]) / G
    if not np.all(np.isfinite(acc)):
        return dict(name=name, cond=cond, bad=True)
    fc = FC_FULL * T_SCALE
    filt = np.column_stack([butter2(acc[:, 0], dt, fc),
                            butter2(acc[:, 1], dt, fc)])
    acc, filt = acc[n // 10:], filt[n // 10:]
    m = FRIDSMA71[cond]
    return dict(name=name, cond=cond, dt=dt, fc_model_hz=fc,
                raw=dict(cg=_pkstats(acc[:, 0]), bow=_pkstats(acc[:, 1])),
                filt=dict(cg=_pkstats(filt[:, 0]), bow=_pkstats(filt[:, 1])),
                meas=[m[3], m[4]])


def f71():
    from studies.parallel_runner import run_parallel
    from studies.exp_planing import FRIDSMA71
    jobs = [(n, k, c, {}) for n, k in BASES.items() for c in FRIDSMA71]
    jobs.sort(key=lambda j: j[1] != "as1")
    t0 = time.time()
    res = dict(rows=_keep(run_parallel(f71_job, jobs, **POOL)),
               wall_s=time.time() - t0)
    _save("f71.json", res)
    return res


# ------------------------------------------------------------- report
def _keep(L):
    return [x if isinstance(x, dict) else repr(x) for x in L]


def _save(fn, res):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, fn), "w") as f:
        json.dump(res, f, indent=1)


def _load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def report():
    from studies.exp_planing import FRIDSMA, FRIDSMA71
    lines = []
    say = lambda s="": lines.append(s)
    names = list(BASES)
    r = _load("sav.json")
    if r:
        say("SAVITSKY scarab195 (trim deg / R N | Savitsky): rows v0, as1 "
            "from this harness; ZR, ZRM in sim_redesign/step3.json")
        rows = [x for x in r["rows"] if isinstance(x, dict)]
        for n in ("v0", "as1"):
            for s in sorted([x for x in rows if x["name"] == n],
                            key=lambda x: x["kn"]):
                say(f"  {n:<4} {s['kn']:>3} kn  {s['trim']:6.2f} {s['R']:7.0f}"
                    f" | {s['trim_sav']:5.2f} {s['R_sav']:6.0f}  err "
                    f"{s['trim'] - s['trim_sav']:+.2f} deg "
                    f"{100 * (s['R'] / s['R_sav'] - 1):+.1f} %")
        say(f"  wall {r['wall_s']:.0f} s")
    r = _load("rest.json")
    if r:
        say("\nREST: 60 s zero thrust (u m/s, x m, max |u|)")
        for d in r["drift"]:
            if isinstance(d, dict):
                say(f"  {d['name']:<4} u {d['u_end']:+.3f} x {d['x_end']:+.1f} "
                    f"max|u| {d['umax']:.3f}  z {d['z_end']:+.4f} trim "
                    f"{d['trim_end']:.3f}")
            else:
                say("  " + d[:200])
        say("REST: heave free decay from +5 cm, surge held (period s, wn "
            "rad/s, zeta)")
        for d in r["decay"]:
            if isinstance(d, dict) and "period" in d:
                say(f"  {d['name']:<4} T {d['period']:.3f} wn {d['wn']:.3f} "
                    f"zeta {d['zeta']:.4f} (first {d['zeta_first']:.4f}) "
                    f"peaks {['%.4f' % v for v in d['peaks']]}")
            else:
                say("  " + repr(d)[:200])
        say("REST RAO head seas a 0.02 m (heave/a; pitch/ka); BEM head seas "
            "(ledger B1.3 revised): heave 0.965/0.769/0.288/0.074/0.027/"
            "0.030/0.009, pitch 0.992/0.858/0.523/0.128/0.058/0.017/0.003")
        rao = {(x["name"], round(x["w"], 2)): x for x in r["rao"]
               if isinstance(x, dict) and "heave" in x}
        for n in names:
            say(f"  {n:<4} heave " + " ".join(
                f"{rao[(n, round(w, 2))]['heave']:.3f}" if (n, round(w, 2))
                in rao else "nan" for w in r["omega"]))
            say(f"  {n:<4} pitch " + " ".join(
                f"{rao[(n, round(w, 2))]['pitch']:.3f}" if (n, round(w, 2))
                in rao else "nan" for w in r["omega"]))
        say(f"  omega {['%.2f' % w for w in r['omega']]}  wall "
            f"{r['wall_s']:.0f} s")
    r = _load("calm.json")
    if r:
        say("\nCALM CAPTIVE from attitude + 5 cm, 60 s: z p-p / trim p-p "
            "last 10 s (previous 10 s); mean z, trim")
        for n in names:
            for x in sorted([x for x in r["rows"] if isinstance(x, dict)
                             and x["name"] == n], key=lambda x: x["kn"]):
                if x.get("bad"):
                    say(f"  {n:<4} {x['kn']:>3} kn  NON-FINITE")
                    continue
                say(f"  {n:<4} {x['kn']:>3} kn  z {x['zpp_last']:.4f} "
                    f"({x['zpp_prev']:.4f})  trim {x['trimpp_last']:.3f} "
                    f"({x['trimpp_prev']:.3f})  mean z {x['z_mean']:+.4f} "
                    f"trim {x['trim_mean']:.3f}  attitude z "
                    f"{x['z_attitude']:+.4f} trim {x['trim_attitude']:.3f}")
        say(f"  wall {r['wall_s']:.0f} s")
    r = _load("f69.json")
    if r:
        rows = [x for x in r["rows"] if isinstance(x, dict) and "res" in x]
        tab = {(x["name"], x["case"], x["lam"]): x for x in rows}
        dt = rows[0]["dt"] if rows else float("nan")
        say(f"\nFRIDSMA 1969 at model dt {dt:.5f} s: heave, pitch, CG g, bow"
            f" g raw | CG g, bow g filtered (20 Hz full scale)")
        keys = [(c, l) for c in "AB" for l in FRIDSMA[c]]
        for n in names:
            cnt = acc = accf = 0
            for c, l in keys:
                x = tab.get((n, c, l))
                if not x:
                    continue
                m = FRIDSMA[c][l]
                for j, tol in enumerate((.2, .2, .3, .3)):
                    cnt += abs(x["res"][j] / m[j] - 1) <= tol
                acc += sum(abs(x["res"][j] / m[j] - 1) <= .3 for j in (2, 3))
                accf += sum(abs(x["filt"][j - 2] / m[j] - 1) <= .3
                            for j in (2, 3))
            say(f"  {n}: raw count {cnt}/44 (acc {acc}/22); filtered acc "
                f"{accf}/22")
            for c, l in keys:
                x = tab.get((n, c, l))
                if x:
                    m = FRIDSMA[c][l]
                    say(f"    {c}{l:<4} " + ",".join(f"{v:.2f}" for v in
                                                      x["res"]) +
                        f" | {x['filt'][0]:.2f},{x['filt'][1]:.2f}"
                        f"   meas " + ",".join(f"{v:.2f}" for v in m))
        say(f"  wall {r['wall_s']:.0f} s")
    r = _load("f71.json")
    if r:
        rows = [x for x in r["rows"] if isinstance(x, dict) and "raw" in x]
        dt = rows[0]["dt"] if rows else float("nan")
        say(f"\nFRIDSMA 1971 seed 0, nominal Tp, model dt {dt:.5f} s: CG/bow "
            f"mean positive peaks raw | filtered; A1/10 raw | filtered; n")
        for n in names:
            cnt = cntf = 0
            for x in sorted([x for x in rows if x["name"] == n],
                            key=lambda x: x["cond"]):
                m = x["meas"]
                cnt += (abs(x["raw"]["cg"]["mean"] / m[0] - 1) <= .3) + \
                    (abs(x["raw"]["bow"]["mean"] / m[1] - 1) <= .3)
                cntf += (abs(x["filt"]["cg"]["mean"] / m[0] - 1) <= .3) + \
                    (abs(x["filt"]["bow"]["mean"] / m[1] - 1) <= .3)
                say(f"  {n:<4} {x['cond']}: mean {x['raw']['cg']['mean']:.2f}"
                    f"/{x['raw']['bow']['mean']:.2f} | "
                    f"{x['filt']['cg']['mean']:.2f}/"
                    f"{x['filt']['bow']['mean']:.2f}   A1/10 "
                    f"{x['raw']['cg']['a110']:.2f}/{x['raw']['bow']['a110']:.2f}"
                    f" | {x['filt']['cg']['a110']:.2f}/"
                    f"{x['filt']['bow']['a110']:.2f}   n {x['raw']['cg']['n']}"
                    f"/{x['filt']['cg']['n']}   meas {m[0]}/{m[1]}")
            say(f"  {n}: within 30 % raw {cnt}/10, filtered {cntf}/10")
        say(f"  wall {r['wall_s']:.0f} s")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


PARTS = dict(sav=sav, rest=rest, calm=calm, f69=f69, f71=f71)

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    for part in (PARTS if what == "all" else [what]):
        if part in PARTS:
            PARTS[part]()
            print(f"{part} done {time.time() - t0:.0f} s", flush=True)
    report()
    print(f"total {time.time() - t0:.0f} s", flush=True)
