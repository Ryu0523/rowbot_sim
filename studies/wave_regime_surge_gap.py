#!/usr/bin/env python3
"""
What the MPC's surge / yaw equations miss in waves, by relative wave
direction and speed: the high-fidelity plant (final_eval's GZ plant, built
by studies/final_eval.make_mission) at CONSTANT throttle with the
mission's heading autopilot, in the evaluation sea state (Hs 1.0 m, Tp 5 s,
24 x 5 components) turned to head / beam / stern-quartering / following,
at a low and a high throttle. Per plant step (0.02 s) the measured surge
and yaw accelerations are compared with what control/reduced.py's
equations give for the same state and the jet's ACTUAL thrust state (so
actuator lag is not counted as model error):

    surge  a_red = (thrust_state - k_drag u|u|) / m_surge     (no wave term)
    yaw    rdot_red = (r_tgt - r) / tau_r,  r_tgt = k_nomoto_f k_jet_side
           thrust_state sin(nozzle_state)                      (no wave term)

e = measured - model. Its mean is the calm-water mismatch at that speed
(drag, jet), its spread mostly the wave forces the model does not have.
For scale: g x the mean longitudinal wave slope along the hull (5 points
stern..bow on the centreline), the Froude-Krylov order of the wave surge
force per unit mass, and its correlation with e_surge.

Also per run: speed mean / spread / extremes, share of time in the MGN 328
band (studies/safety_events.py: speed along the waves in [0.7, 1.15] c),
heading deviation, roll, pump prime.

Light: 120 s runs (--T), one process, sequential; 10-40 s per run.
Sea: hs 1.0 m, --tp (5 default; 8 for the swell runs), 24 x 5 components.
Output: studies/_cache/wave_regimes/surge_gap.json.

    .venv/Scripts/python.exe -m studies.wave_regime_surge_gap \
        --T 120 --f 0.1,0.15,0.2,0.24,0.3,0.5 --seeds 0,1 --tp 5
    .venv/Scripts/python.exe -m studies.wave_regime_surge_gap \
        --T 120 --f 0.15,0.3,0.5 --seeds 0,1 --tp 8
    .venv/Scripts/python.exe -m studies.wave_regime_surge_gap --report
"""
import argparse
import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np   # noqa: E402

OUT = os.path.join(HERE, "_cache", "wave_regimes")
KN = 0.514444
G = 9.81
# relative direction the waves come FROM (0 head .. 180 following) ->
# propagation direction theta0 for a boat heading 0 (track leg 0)
DIRS = {"head": math.pi, "beam": -math.pi / 2,
        "stern_quart_135": math.pi / 4, "following": 0.0}


def run(f, dname, T, seed=0, warm=10.0, tp=5.0):
    from studies import final_eval as FE
    m = FE.make_mission(seed, 0, T, sea=dict(theta0=DIRS[dname], tp=tp))
    p = m.red.p
    pl = m.plant
    sea = m.sea
    # start at the calm steady speed of this throttle when it has one,
    # else at the task's 0.8 u_design (make_mission's start)
    u0 = START.get(f)
    if u0 is not None:
        m.s = pl.initial_state(u0)
        m.s[5] = m.phi
    thrust = f * m.t_max
    dt = m.dt
    x_st = np.linspace(pl.sec.x_stern, pl.sec.x_bow, 5)
    xc = x_st - x_st.mean()
    den = float((xc ** 2).sum())
    rows = []
    t0 = time.perf_counter()
    while not m.done():
        rud = m.ep._steer(m.s, thrust)
        for _ in range(m.sub):
            s = m.s
            prime = float(pl.prop.prime)
            m.s = pl.step(s, m.t, thrust, rud, dt)
            m.t += dt
            s1 = m.s
            if not np.all(np.isfinite(s1)):
                m.finite = False
                break
            c, sn = math.cos(s[5]), math.sin(s[5])
            X = s[0] + x_st * c
            Y = s[1] + x_st * sn
            eta = (sea.a * np.cos(sea.k * (X[:, None] * np.cos(sea.th)
                                           + Y[:, None] * np.sin(sea.th))
                                  - sea.w * (m.t - dt) + sea.phi)).sum(-1)
            slope = float((eta * xc).sum() / den)     # rises towards bow
            u, r = float(s[6]), float(s[11])
            a_meas = (float(s1[6]) - u) / dt
            rd_meas = (float(s1[11]) - r) / dt
            thr_s, noz_s = float(s[12]), float(s[13])
            a_red = (thr_s - p["k_drag"] * u * abs(u)) / p["m_surge"]
            r_tgt = (p["k_nomoto_f"] * p["k_jet_side"] * max(thr_s, 0.0)
                     * math.sin(max(min(noz_s, p["rud_stall"]),
                                    -p["rud_stall"])))
            rd_red = (r_tgt - r) / p["tau_r"]
            mu = float(s[5]) - float(m.theta0)
            u_w = u * math.cos(mu) - float(s[7]) * math.sin(mu)
            rows.append((m.t, u, a_meas, a_red, slope, r, rd_meas, rd_red,
                         float(s[3]), float(s[5]) - m.phi, u_w, prime))
        m.k += 1
        if not m.finite or getattr(pl, "capsized", False):
            break
    A = np.array(rows)
    keep = A[:, 0] >= warm
    A = A[keep]
    t, u, am, ar, sl, r, rdm, rdr, roll, hd, uw, prime = A.T
    e = am - ar
    ey = rdm - rdr
    gsl = -G * sl          # forward push of a slope rising to the bow is
    #                        negative (the hull climbs): F ~ -m g slope
    cc = float(np.corrcoef(e - e.mean(), gsl)[0, 1]) if e.std() > 0 else 0
    c_wave = G * float(sea.tp) / (2 * math.pi) if hasattr(sea, "tp") \
        else float("nan")
    band = (uw >= 0.7 * c_wave) & (uw <= 1.15 * c_wave)
    out = dict(f=f, dir=dname, theta0=DIRS[dname], T=T, warm=warm, tp=tp,
               seed=seed,
               finite=bool(m.finite),
               capsized=bool(getattr(pl, "capsized", False)),
               kn_mean=float(u.mean() / KN), kn_std=float(u.std() / KN),
               kn_min=float(u.min() / KN), kn_max=float(u.max() / KN),
               e_surge_mean=float(e.mean()), e_surge_std=float(e.std()),
               e_surge_p99_abs=float(np.percentile(np.abs(e - e.mean()), 99)),
               g_slope_std=float(gsl.std()),
               corr_e_surge_g_slope=cc,
               a_meas_std=float(am.std()),
               e_yaw_std=float(ey.std()), rdot_meas_std=float(rdm.std()),
               r_std_deg=float(np.degrees(r.std())),
               heading_dev_std_deg=float(np.degrees(hd.std())),
               heading_dev_maxabs_deg=float(np.degrees(np.abs(hd).max())),
               roll_std_deg=float(np.degrees(roll.std())),
               roll_maxabs_deg=float(np.degrees(np.abs(roll).max())),
               mgn328_share=float(band.mean()), c_wave=c_wave,
               prime_min=float(prime.min()),
               prime_lt_05_share=float(np.mean(prime < 0.5)),
               t_run=time.perf_counter() - t0)
    return out


START = {}


def report(path):
    """Group the saved 120 s runs by (Tp, throttle, direction): seed means,
    extremes over seeds for min / max columns."""
    import collections
    d = json.load(open(path))
    g = collections.defaultdict(list)
    for r in d.values():
        if r["T"] == 120:
            g[(r.get("tp", 5.0), r["f"], r["dir"])].append(r)
    cols = ["kn_mean", "kn_std", "kn_min", "kn_max", "e_surge_mean",
            "e_surge_std", "g_slope_std", "corr_e_surge_g_slope",
            "e_yaw_std", "heading_dev_std_deg", "heading_dev_maxabs_deg",
            "roll_std_deg", "roll_maxabs_deg", "mgn328_share",
            "prime_lt_05_share"]
    print("Tp f dir n | " + " ".join(c[:10] for c in cols))
    for key in sorted(g):
        rs = g[key]
        vals = []
        for c in cols:
            v = [r[c] for r in rs]
            vals.append(min(v) if c == "kn_min" else max(v) if c in (
                "kn_max", "heading_dev_maxabs_deg", "roll_maxabs_deg")
                else float(np.mean(v)))
        ok = all(r["finite"] and not r["capsized"] for r in rs)
        print(f"{key[0]:g} {key[1]:.2f} {key[2]:>16} {len(rs)} | "
              + " ".join(f"{v:7.2f}" for v in vals) + f"  ok {ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=60.0)
    ap.add_argument("--f", default="0.15,0.3")
    ap.add_argument("--dirs", default="head,beam,stern_quart_135,following")
    ap.add_argument("--tp", type=float, default=5.0)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.report:
        report(os.path.join(OUT, "surge_gap.json"))
        return
    gaps = os.path.join(OUT, "model_gaps.json")
    if os.path.exists(gaps):
        for r in json.load(open(gaps))["steady_speed_vs_throttle"]:
            if len(r["u_plant_kn"]) == 1:
                START[round(r["f"], 3)] = r["u_plant_kn"][0] * KN
    path = os.path.join(OUT, "surge_gap.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for f in [float(x) for x in a.f.split(",")]:
        START.setdefault(f, None)
        for d, sd in [(d, int(x)) for d in a.dirs.split(",")
                      for x in a.seeds.split(",")]:
            key = f"{f:.2f}_{d}_T{a.T:g}" + ("" if a.tp == 5.0 and sd == 0
                                             else f"_tp{a.tp:g}_s{sd}")
            if key in res:
                print(f"{key}: cached")
                continue
            r = run(f, d, a.T, seed=sd, tp=a.tp)
            res[key] = r
            with open(path + ".tmp", "w") as fh:
                json.dump(res, fh, indent=1)
            os.replace(path + ".tmp", path)
            print(f"{key}: {r['kn_mean']:.1f} kn (sd {r['kn_std']:.2f}, "
                  f"{r['kn_min']:.1f}-{r['kn_max']:.1f}); surge error mean "
                  f"{r['e_surge_mean']:+.2f} sd {r['e_surge_std']:.2f} m/s2 "
                  f"(g x slope sd {r['g_slope_std']:.2f}, corr "
                  f"{r['corr_e_surge_g_slope']:+.2f}); yaw-acc error sd "
                  f"{r['e_yaw_std']:.3f} rad/s2; heading dev sd "
                  f"{r['heading_dev_std_deg']:.1f} max "
                  f"{r['heading_dev_maxabs_deg']:.1f} deg; roll sd "
                  f"{r['roll_std_deg']:.1f} max {r['roll_maxabs_deg']:.1f} "
                  f"deg; MGN band {r['mgn328_share']:.2f}; prime<0.5 "
                  f"{r['prime_lt_05_share']:.3f} ({r['t_run']:.0f} s)",
                  flush=True)


if __name__ == "__main__":
    main()
