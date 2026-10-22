#!/usr/bin/env python3
"""
Achievable envelope, first pass (learn/meta/CONTROL_FROM_SCRATCH_2026-10-12.md
section 10 item 1; write-up learn/meta/ENVELOPE_PASS1_2026-10-12.md).

The frozen plant scarab195@as1 (large-angle twin, dt 0.01 s, control step
0.25 s) held at a set speed (P+I throttle of the test scaffolding) and
heading (the project's autopilot) in the frozen scenario builder's sea,
60 s settling dropped + 600 s of statistics, two registered wave seeds
(0, 1) per condition:

  phase nominal10  Hs 1.0 m Tp 5 s, head (180 deg) and bow-quartering
                   (135 deg), speeds 8-32 kn (the fold 13-13.65 kn skipped)
  phase random     head seas Hs 1.0 m, 15 / 20 / 25 kn, 6 randomisation
                   draws (hull fields, sea inputs, thrust gain on;
                   output scales off; hull_every 1, plant cache 8)
  phase nominal15  Hs 1.5 m Tp 6 s, the nominal10 grid
  phase refine     head seas Hs 1.0 m at 16 and 17 kn (where the 7 g rate
                   crosses 1 per hour in the first results)

Per run (studies/envelope_run.py): the filtered CG and bow-gauge
accelerations (output spec), freeboard at the stem, yaw rate, roll, the
plant's slam counter; per condition (studies/envelope_stats.py): peak
statistics, GPD tail extrapolation of the rate of peaks >= 3 / 5 / 7 g per
hour with a 200-resample bootstrap, bow-immersion events (freeboard below
a quarter of the static value) per hour, roll maximum; the envelope
read-off with k = 1 and k = 1.2.

    python -m studies.envelope_pass1 run [nominal10 random nominal15]
    python -m studies.envelope_pass1 report
    python -m studies.envelope_pass1 jobs      (list the runs and status)

Outputs: studies/_cache/envelope/runs/<name>.npz + .json (one per run,
skipped when present), progress.log, summary.json, tables.md.
Read-only use of sim/, hydro/, app/.
"""
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

from studies import envelope_run as R, envelope_stats as S   # noqa: E402

OUT = os.path.join(ROOT, "studies", "_cache", "envelope")
RUNS = os.path.join(OUT, "runs")
KN = R.KN
SPEEDS = (8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 32.0)
SEEDS = (0, 1)                              # registered evaluation seeds
SEAS = {"h10": dict(hs=1.0, tp=5.0), "h15": dict(hs=1.5, tp=6.0)}
HEADINGS = {"head": dict(heading_rad=0.0, rel_deg=180),
            "bowq": dict(heading_rad=math.pi / 4, rel_deg=135)}
REFINE_SPEEDS = (16.0, 17.0)      # phase refine: head seas Hs 1.0 m only
ALL_SPEEDS = tuple(sorted(set(SPEEDS) | set(REFINE_SPEEDS)))
RZ_SPEEDS = (15.0, 20.0, 25.0)
RZ_DRAWS = tuple(range(6))
RZ_SEED = 20261012
I_CLIP = 10.0            # integrator clip widened from the scaffolding's 2
MAX_PROCS = 16
GB_PER_JOB = 0.2


def rz_config():
    from app.randomize import Randomization
    rz = Randomization(seed=RZ_SEED, stream="eval", hull_every=1,
                       plant_cache=8).without("acc_scale", "motion_scale")
    return rz.to_dict()


# ------------------------------------------------------------------ jobs
def jobs(phases=("nominal10", "random", "nominal15", "refine")):
    out = []
    for ph in phases:
        if ph in ("nominal10", "nominal15"):
            sea_tag = "h10" if ph == "nominal10" else "h15"
            sea = SEAS[sea_tag]
            for hd, h in HEADINGS.items():
                for kn in SPEEDS:
                    for sd in SEEDS:
                        out.append(dict(
                            name=f"{sea_tag}_{hd}_{kn:g}kn_s{sd}", phase=ph,
                            sea=sea_tag, heading=hd, speed_kn=kn, seed=sd,
                            sea_seed=sd, draw=None, randomize=False,
                            i_clip=I_CLIP, **sea, **h))
        elif ph == "refine":
            sea_tag, hd = "h10", "head"
            for kn in REFINE_SPEEDS:
                for sd in SEEDS:
                    out.append(dict(
                        name=f"{sea_tag}_{hd}_{kn:g}kn_s{sd}", phase=ph,
                        sea=sea_tag, heading=hd, speed_kn=kn, seed=sd,
                        sea_seed=sd, draw=None, randomize=False,
                        i_clip=I_CLIP, **SEAS[sea_tag], **HEADINGS[hd]))
        elif ph == "random":
            sea_tag, hd = "h10", "head"
            for kn in RZ_SPEEDS:
                for j in RZ_DRAWS:
                    for sd in SEEDS:
                        out.append(dict(
                            name=f"rz_{sea_tag}_{hd}_{kn:g}kn_d{j}_s{sd}",
                            phase=ph, sea=sea_tag, heading=hd, speed_kn=kn,
                            seed=1000 + j, sea_seed=sd, draw=j,
                            randomize=True, randomization=rz_config(),
                            i_clip=I_CLIP, **SEAS[sea_tag], **HEADINGS[hd]))
        else:
            raise ValueError(ph)
    return out


def paths(name):
    return (os.path.join(RUNS, name + ".json"),
            os.path.join(RUNS, name + ".npz"))


def done(name):
    pj, pn = paths(name)
    return os.path.exists(pj) and os.path.exists(pn)


def job(arg):
    """Run one job unless its files exist; save; return a light summary."""
    os.makedirs(RUNS, exist_ok=True)
    name = arg["name"]
    pj, pn = paths(name)
    t0 = time.time()
    if done(name):
        with open(pj) as f:
            return json.load(f)
    r = R.run_one(arg)
    st = r.pop("stats", None)
    series = r.pop("series", None)
    arrays = {}
    light = dict(r)
    light["arg"] = {k: v for k, v in arg.items() if k != "randomization"}
    if st is not None:
        st = dict(st)
        for g in ("cg", "bow"):
            pk = st[g].pop("peaks")
            arrays[f"peaks_{g}"] = np.asarray(pk, float)
        light["stats"] = st
    else:
        light["stats"] = None
    if series:
        arrays.update(series)
    np.savez_compressed(pn, **arrays)
    with open(pj, "w") as f:
        json.dump(light, f, indent=1, default=_plain)
    with open(os.path.join(OUT, "progress.log"), "a") as f:
        s = light.get("stats") or {}
        f.write(f"{time.strftime('%H:%M:%S')} {name} done in "
                f"{time.time() - t0:.0f} s ({light['ms_per_step']:.2f} ms/"
                f"step, build {light['wall_build']:.1f} s) reason "
                f"{light['reason']} u {s.get('u_mean_kn', float('nan')):.2f}"
                f" kn cg a110 {(s.get('cg') or {}).get('a110', float('nan')):.2f}"
                f" bow a110 {(s.get('bow') or {}).get('a110', float('nan')):.2f}\n")
    return light


def _plain(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (set, frozenset)):
        return sorted(v)
    raise TypeError(type(v))


def cmd_run(phases):
    from studies.parallel_runner import free_gb, n_workers, run_parallel
    os.makedirs(RUNS, exist_ok=True)
    todo = [a for a in jobs(phases) if not done(a["name"])]
    n = n_workers(GB_PER_JOB, 1.5, MAX_PROCS)
    msg = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} run {list(phases)}: "
           f"{len(todo)} runs to do, free RAM {free_gb():.1f} GB, "
           f"{n} workers")
    print(msg, flush=True)
    with open(os.path.join(OUT, "progress.log"), "a") as f:
        f.write(msg + "\n")
    t0 = time.time()
    res = run_parallel(job, todo, gb_per_job=GB_PER_JOB, reserve_gb=1.5,
                       max_procs=MAX_PROCS)
    bad = [(a["name"], r) for a, r in zip(todo, res)
           if isinstance(r, Exception)]
    msg = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} run done: {len(todo)} "
           f"runs in {(time.time() - t0) / 60:.1f} min, {len(bad)} failed")
    print(msg, flush=True)
    with open(os.path.join(OUT, "progress.log"), "a") as f:
        f.write(msg + "\n")
        for nm, e in bad:
            f.write(f"  FAILED {nm}: {e!r}\n")
    for nm, e in bad:
        print(f"  FAILED {nm}: {e!r}")


def cmd_jobs():
    for a in jobs():
        print(f"{'done' if done(a['name']) else '    '}  {a['name']}")


# ---------------------------------------------------------------- report
def load_run(name):
    pj, pn = paths(name)
    with open(pj) as f:
        r = json.load(f)
    z = np.load(pn)
    r["peaks"] = {g: z[f"peaks_{g}"] for g in ("cg", "bow")
                  if f"peaks_{g}" in z}
    return r


def trim_margins():
    """{kn: (margin deg, in chart range, trim deg, limit deg)} from the
    stability map's thrust-speed curve (studies/_cache/stability/curve.json)
    and the Savitsky 1964 porpoising chart as studies/stability_figs."""
    try:
        from studies.stability_figs import cl_beta, tau_crit_sav20, CL_RANGE
    except Exception:            # noqa: BLE001 -- the constants, verbatim
        SAV20 = (-254.51, 239.65, -23.936, 3.0195)
        CL_RANGE = (0.0338, 0.18)
        MASS, BEAM = 1400.0, 0.83 * 2.44

        def tau_crit_sav20(cl):
            x = math.sqrt(cl / 2.0)
            a, b, c, d = SAV20
            return a * x ** 3 + b * x ** 2 + c * x + d

        def cl_beta(u, rho=1025.0):
            return MASS * 9.81 / (0.5 * rho * u * u * BEAM * BEAM)
    p = os.path.join(ROOT, "studies", "_cache", "stability", "curve.json")
    with open(p) as f:
        cv = json.load(f)
    rows = {round(r["kn"], 4): r for r in cv["coarse"]}
    out = {}
    for kn in ALL_SPEEDS:
        r = rows.get(round(kn, 4))
        if r is None:
            continue
        cl = cl_beta(kn * KN)
        lim = tau_crit_sav20(cl)
        out[kn] = dict(margin_deg=lim - r["trim_deg"],
                       in_chart=bool(CL_RANGE[0] <= cl <= CL_RANGE[1]),
                       trim_deg=r["trim_deg"], limit_deg=lim)
    return out


def condition_row(runs, kn):
    """Pooled statistics of the runs (the two seeds) of one condition."""
    ok = [r for r in runs if r.get("stats")]
    T = sum(r["t_stat"] for r in ok)
    row = dict(kn=kn, n_runs=len(runs), n_ok=len(ok), T_s=T,
               reasons=[r["reason"] for r in runs],
               capsized=any(r["capsized"] for r in runs),
               u_mean_kn=float(np.mean([r["stats"]["u_mean_kn"]
                                        for r in ok])) if ok else np.nan,
               u_sd=float(np.mean([r["stats"]["u_sd"] for r in ok]))
               if ok else np.nan,
               thr_mean=float(np.mean([r["stats"]["thr_mean"]
                                       for r in ok])) if ok else np.nan,
               thr_sat=float(np.max([r["stats"]["thr_sat_frac"]
                                     for r in ok])) if ok else np.nan,
               roll_max_deg=float(max(r["stats"]["roll_max_deg"]
                                      for r in ok)) if ok else np.nan,
               yaw_rate_max_dps=float(max(r["stats"]["yaw_rate_max_dps"]
                                          for r in ok)) if ok else np.nan,
               yaw_rate_rms_dps=float(np.mean([r["stats"]["yaw_rate_rms_dps"]
                                               for r in ok])) if ok else np.nan,
               roll_rms_deg=float(np.mean([r["stats"]["roll_rms_deg"]
                                           for r in ok])) if ok else np.nan,
               heave_sd=float(np.mean([r["stats"]["heave_sd"] for r in ok]))
               if ok else np.nan,
               pitch_sd_deg=float(np.mean([r["stats"]["pitch_sd_deg"]
                                           for r in ok])) if ok else np.nan,
               fb_min=float(min(r["stats"]["fb_min"] for r in ok))
               if ok else np.nan,
               fb_static=ok[0]["freeboard_static"] if ok else np.nan,
               n_immersion=int(sum(r["stats"]["n_immersion"] for r in ok)),
               slam_count=int(sum(r["stats"]["slam_count"] or 0
                                  for r in ok)),
               wall_total=sum(r["wall_total"] for r in runs),
               ms_per_step=float(np.mean([r["ms_per_step"]
                                          for r in runs])))
    row["immersion_per_hour"] = (row["n_immersion"] * 3600.0 / T
                                 if T > 0 else np.nan)
    row["slam_per_hour"] = row["slam_count"] * 3600.0 / T if T > 0 else np.nan
    for g in ("cg", "bow"):
        pk = S.pool_peaks([r["peaks"][g] for r in ok]) if ok else np.array([])
        row[g] = dict(a110=S.a110(pk),
                      a110_runs=[r["stats"][g]["a110"] for r in ok],
                      mean=float(pk.mean()) if len(pk) else np.nan,
                      max=float(pk.max()) if len(pk) else np.nan,
                      n=int(len(pk)),
                      tail=S.tail_fit(pk, T) if len(pk) else dict(ok=False))
    return row


def _num(v):
    """A rate per hour, readable: 0, 0.03, 0.8, 4.2, 37, 540."""
    if v is None or not np.isfinite(v):
        return "—"
    if v == 0:
        return "0"
    if v >= 10:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.1f}"
    return f"{v:.2g}"


def _fmt_rate(lv):
    if lv is None:
        return "—"
    s = _num(lv["rate"])
    if lv["extrapolated"]:
        s += f" [{_num(lv['p5'])}, {_num(lv['p95'])}]"
    else:
        s += " (计数)"
    return s


def _lev(row, g, k, lev):
    t = row[g]["tail"]
    if not t.get("ok"):
        return None
    return t["levels"].get(f"k{k:g}_{lev:g}g")


def table_condition(rows, tm, title):
    L = [f"### {title}", "",
         "| 设定航速 kn | 实际平均航速 kn | 纵倾裕度 deg | 重心 A1/10 g "
         "(两个种子各自) | 船首 A1/10 g | 重心最大 g | 船首最大 g | 峰数 | "
         "重心 ≥7 g 次/小时 外推 [5–95%] | 直接计数 ≥7 g | ≥5 g 次/小时 | "
         "≥3 g 次/小时 | 船首进水 次/小时 (次数) | 余高最小 m | 横摇最大 deg"
         " | 船模碰浪计数 次/小时 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = tm.get(r["kn"])
        ms = (f"{m['margin_deg']:+.2f}{'' if m['in_chart'] else '*'}"
              if m else "—")
        l7, l5, l3 = (_lev(r, "cg", 1.0, 7), _lev(r, "cg", 1.0, 5),
                      _lev(r, "cg", 1.0, 3))
        nd = l7["n_direct"] if l7 else "—"
        L.append(
            f"| {r['kn']:g} | {r['u_mean_kn']:.1f} | {ms} | "
            f"{r['cg']['a110']:.2f} ("
            + "/".join(f"{v:.2f}" for v in r['cg']['a110_runs']) + ") | "
            f"{r['bow']['a110']:.2f} | "
            f"{r['cg']['max']:.2f} | {r['bow']['max']:.2f} | {r['cg']['n']} | "
            f"{_fmt_rate(l7)} | {nd} | {_fmt_rate(l5)} | {_fmt_rate(l3)} | "
            f"{r['immersion_per_hour']:.0f} ({r['n_immersion']}) | "
            f"{r['fb_min']:.2f} | {r['roll_max_deg']:.1f} | "
            f"{r['slam_per_hour']:.0f} |")
    L.append("")
    return L


def table_hold(rows, title):
    L = [f"### {title}：航速保持、航向保持与运动量", "",
         "| 设定航速 kn | 实际平均航速 kn | 航速起伏 m/s (标准差) | 油门平均 "
         "(最大推力的分数) | 油门饱和时间比例 | 艏摇角速度 均方根 / 最大 deg/s | "
         "横摇 均方根 / 最大 deg | 升沉标准差 m | 纵摇标准差 deg | 运行结束原因 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(
            f"| {r['kn']:g} | {r['u_mean_kn']:.2f} | {r['u_sd']:.2f} | "
            f"{r['thr_mean']:.3f} | {r['thr_sat']:.2f} | "
            f"{r['yaw_rate_rms_dps']:.1f} / {r['yaw_rate_max_dps']:.1f} | "
            f"{r['roll_rms_deg']:.1f} / {r['roll_max_deg']:.1f} | "
            f"{r['heave_sd']:.2f} | {r['pitch_sd_deg']:.2f} | "
            f"{', '.join(sorted(set(r['reasons'])))} |")
    L.append("")
    return L


def table_k12(rows, title):
    L = [f"### {title}（修正系数 k = 1.2：仿真加速度乘 1.2 后再比阈值）", "",
         "| 设定航速 kn | 重心 1.2×A1/10 g | 船首 1.2×A1/10 g | 重心 ≥7 g "
         "次/小时 外推 [5–95%] | ≥5 g | ≥3 g | 船首 ≥7 g 次/小时 |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(
            f"| {r['kn']:g} | {1.2 * r['cg']['a110']:.2f} | "
            f"{1.2 * r['bow']['a110']:.2f} | "
            f"{_fmt_rate(_lev(r, 'cg', 1.2, 7))} | "
            f"{_fmt_rate(_lev(r, 'cg', 1.2, 5))} | "
            f"{_fmt_rate(_lev(r, 'cg', 1.2, 3))} | "
            f"{_fmt_rate(_lev(r, 'bow', 1.2, 7))} |")
    L.append("")
    return L


def table_tail(rows, title):
    L = [f"### {title}：尾部拟合细节（重心）", "",
         "| 设定航速 kn | 阈值 u g (第 90 百分位) | 超阈值峰数 | 形状参数 ξ "
         "(PWM) | 尺度 σ g | ξ (最大似然) | 上端点 g | 到 7 g 的外推距离 "
         "(7−u)/σ | 阈值取第 80 百分位时 ≥7 g 次/小时 | 第 95 百分位时 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["cg"]["tail"]
        if not t.get("ok"):
            L.append(f"| {r['kn']:g} | — | | | | | | | | |")
            continue
        ep = t["endpoint"]
        L.append(
            f"| {r['kn']:g} | {t['u']:.2f} | {t['n_exc']} | {t['xi']:+.2f} | "
            f"{t['sigma']:.2f} | {t['xi_mle']:+.2f} | "
            f"{'∞' if not np.isfinite(ep) else f'{ep:.1f}'} | "
            f"{(7.0 - t['u']) / t['sigma']:.1f} | "
            f"{_num(t['threshold_sensitivity']['q0.8']['rates']['7g'])} | "
            f"{_num(t['threshold_sensitivity']['q0.95']['rates']['7g'])} |")
    L.append("")
    return L


def table_bow_tail(rows, title):
    L = [f"### {title}：船首测点的事件率（k = 1）", "",
         "| 设定航速 kn | 船首 ≥7 g 次/小时 外推 [5–95%] | 直接计数 | "
         "船首 ≥5 g | 船首 ≥3 g | 船首尾部 ξ |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        t = r["bow"]["tail"]
        l7 = _lev(r, "bow", 1.0, 7)
        L.append(
            f"| {r['kn']:g} | {_fmt_rate(l7)} | "
            f"{l7['n_direct'] if l7 else '—'} | "
            f"{_fmt_rate(_lev(r, 'bow', 1.0, 5))} | "
            f"{_fmt_rate(_lev(r, 'bow', 1.0, 3))} | "
            f"{t['xi']:+.2f} |" if t.get("ok") else
            f"| {r['kn']:g} | — | | | | |")
    L.append("")
    return L


def _ro(v):
    kn, mono = v
    if kn is None:
        return "无"
    return f"{kn:g}" + ("" if mono else "†")


def table_readoff(cond, title):
    L = [f"### {title}", "",
         "| 海况 | 浪向 | 测点 | k | (a) A1/10 ≤ 3 g 的最高航速 | (b) 外推 "
         "≥7 g ≤ 1 次/小时（点估计） | (b') 同，用 95% 上界 | (c) 船首进水 "
         "≤ 3 次/小时 | 三条同时（点估计） | 三条同时（95% 上界） |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for (sea, hd), rows in cond.items():
        for g in ("cg", "bow"):
            for k in (1.0, 1.2):
                ro = S.read_off(rows, g, k)
                rou = S.read_off(rows, g, k, use_upper=True)
                out[f"{sea}_{hd}_{g}_k{k:g}"] = dict(point=ro, upper=rou)
                L.append(
                    f"| {SEAS[sea]['hs']:g} m / {SEAS[sea]['tp']:g} s | "
                    f"{HEADINGS[hd]['rel_deg']}° | {'重心' if g == 'cg' else '船首'}"
                    f" | {k:g} | {_ro(ro['a110'])} | {_ro(ro['rate7'])} | "
                    f"{_ro(rou['rate7'])} | {_ro(ro['immersion'])} | "
                    f"{_ro(ro['all'])} | {_ro(rou['all'])} |")
    L.append("")
    L.append("† 该航速通过但有更低的航速不通过（随航速不单调）。")
    L.append("")
    return L, out


def rz_tables(rz_runs, nominal_rows):
    """Randomisation spread at 15 / 20 / 25 kn: per draw the pooled (two
    seeds) A1/10 and the 7 g rate, min-max over draws, nominal beside."""
    L = ["### 随机化散布（迎浪，有义波高 1.0 米；每个抽样两个浪种子合并）", "",
         "| 设定航速 kn | 抽样号 | 质量 kg | 重心位置 ×标称 | 推力增益 | 波高因子 "
         "| 周期因子 | 实际航速 kn | 重心 A1/10 g | 船首 A1/10 g | 重心 ≥7 g "
         "次/小时 外推 [5–95%] | 船首进水 次/小时 | 横摇最大 deg |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    spread = {}
    nom = {r["kn"]: r for r in nominal_rows}
    for kn in RZ_SPEEDS:
        per = []
        if kn in nom:
            r = nom[kn]
            l7 = _lev(r, "cg", 1.0, 7)
            L.append(f"| {kn:g} | 标称 | 1400 | 1.00 | 1.00 | 1.00 | 1.00 | "
                     f"{r['u_mean_kn']:.1f} | {r['cg']['a110']:.2f} | "
                     f"{r['bow']['a110']:.2f} | {_fmt_rate(l7)} | "
                     f"{r['immersion_per_hour']:.0f} | "
                     f"{r['roll_max_deg']:.1f} |")
        for j in RZ_DRAWS:
            runs = [r for r in rz_runs if r["arg"]["speed_kn"] == kn
                    and r["arg"]["draw"] == j]
            if not runs:
                continue
            row = condition_row(runs, kn)
            d = runs[0].get("draw") or {}
            row["draw"] = d
            row["j"] = j
            per.append(row)
            l7 = _lev(row, "cg", 1.0, 7)
            L.append(
                f"| {kn:g} | {j} | {d.get('mass', float('nan')):.0f} | "
                f"{d.get('lcg', float('nan')):.3f} | "
                f"{d.get('thrust_gain', float('nan')):.2f} | "
                f"{d.get('hs', float('nan')):.2f} | {d.get('tp', float('nan')):.2f}"
                f" | {row['u_mean_kn']:.1f} | {row['cg']['a110']:.2f} | "
                f"{row['bow']['a110']:.2f} | {_fmt_rate(l7)} | "
                f"{row['immersion_per_hour']:.0f} | {row['roll_max_deg']:.1f} |")
        if per:
            a = [p["cg"]["a110"] for p in per]
            b = [p["bow"]["a110"] for p in per]
            r7 = [(_lev(p, "cg", 1.0, 7) or {}).get("rate", np.nan)
                  for p in per]
            r7u = [(_lev(p, "cg", 1.0, 7) or {}).get("p95", np.nan)
                   for p in per]
            im = [p["immersion_per_hour"] for p in per]
            spread[kn] = dict(
                n_draws=len(per), cg_a110=(min(a), max(a)),
                bow_a110=(min(b), max(b)),
                rate7=(float(np.nanmin(r7)), float(np.nanmax(r7))),
                rate7_p95=(float(np.nanmin(r7u)), float(np.nanmax(r7u))),
                immersion=(min(im), max(im)),
                rows=[{k: v for k, v in p.items() if k not in ("cg", "bow")}
                      | {"cg_a110": p["cg"]["a110"],
                         "bow_a110": p["bow"]["a110"],
                         "cg_tail": p["cg"]["tail"],
                         "bow_tail": p["bow"]["tail"]} for p in per])
            L.append(f"| {kn:g} | **范围** | | | | | | | "
                     f"{min(a):.2f}–{max(a):.2f} | {min(b):.2f}–{max(b):.2f} | "
                     f"{_num(np.nanmin(r7))}–{_num(np.nanmax(r7))} | "
                     f"{min(im):.0f}–{max(im):.0f} | |")
    L.append("")
    return L, spread


def cmd_report():
    t0 = time.time()
    all_jobs = jobs()
    have = [a for a in all_jobs if done(a["name"])]
    runs = [load_run(a["name"]) for a in have]
    print(f"{len(runs)} of {len(all_jobs)} runs present")
    tm = trim_margins()
    cond = {}
    for sea in SEAS:
        for hd in HEADINGS:
            rows = []
            for kn in ALL_SPEEDS:
                rr = [r for r in runs if r["arg"].get("draw") is None
                      and r["arg"]["sea"] == sea and r["arg"]["heading"] == hd
                      and r["arg"]["speed_kn"] == kn]
                if rr:
                    rows.append(condition_row(rr, kn))
            if rows:
                cond[(sea, hd)] = rows
    rz_runs = [r for r in runs if r["arg"].get("draw") is not None]
    L = ["# 可达边界第一遍：表格（studies/envelope_pass1.py report 生成）", ""]
    L.append("纵倾裕度：静水平衡纵倾到 Savitsky 1964 海豚运动边界的距离（度），"
             "来自稳定性图的推力—航速曲线；带 * 的航速在该图表的有效范围之外。"
             "事件率为每小时次数；方括号是 200 次自助重抽样的 5–95% 区间；"
             "“(计数)”表示该阈值低于拟合阈值，是直接计数。")
    L.append("")
    for (sea, hd), rows in cond.items():
        title = (f"有义波高 {SEAS[sea]['hs']:g} 米、谱峰周期 {SEAS[sea]['tp']:g} "
                 f"秒，相对浪向 {HEADINGS[hd]['rel_deg']}°"
                 f"（{'迎浪' if hd == 'head' else '艏斜浪'}）")
        L += table_condition(rows, tm, title + "，k = 1")
        L += table_hold(rows, title)
        L += table_k12(rows, title)
        L += table_tail(rows, title)
        L += table_bow_tail(rows, title)
    ro_L, ro = table_readoff(cond, "包线读数：各判据下的最高通过航速（kn）")
    L += ro_L
    rz_L, spread = rz_tables(rz_runs, cond.get(("h10", "head"), []))
    L += rz_L
    # compute time
    wall = sum(r["wall_total"] for r in runs)
    steps = sum(r["n_plant"] for r in runs)
    L += ["### 计算量", "",
          f"- 运行数 {len(runs)}，船模步数合计 {steps:,}，单进程墙钟合计 "
          f"{wall / 3600:.2f} 小时（平均每步 {1e3 * sum(r['wall_step'] for r in runs) / max(steps, 1):.2f} 毫秒，"
          f"每次建船平均 {np.mean([r['wall_build'] for r in runs]):.1f} 秒）。", ""]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    summary = dict(
        n_runs=len(runs), n_jobs=len(all_jobs), trim_margins=tm,
        conditions={f"{sea}_{hd}": rows for (sea, hd), rows in cond.items()},
        readoff=ro, randomisation_spread=spread,
        compute=dict(wall_single_process_h=wall / 3600, n_plant_steps=steps,
                     ms_per_step=1e3 * sum(r["wall_step"] for r in runs)
                     / max(steps, 1),
                     build_s_mean=float(np.mean([r["wall_build"]
                                                 for r in runs]))),
        progress=open(os.path.join(OUT, "progress.log")).read().splitlines()
        if os.path.exists(os.path.join(OUT, "progress.log")) else [],
        report_seconds=time.time() - t0)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=_plain)
    print("\n".join(L))
    print(f"report in {time.time() - t0:.0f} s -> {OUT}/tables.md, summary.json")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:            # noqa: BLE001
        pass
    args = sys.argv[1:] or ["jobs"]
    if args[0] == "run":
        cmd_run(tuple(args[1:]) or ("nominal10", "random", "nominal15",
                                    "refine"))
    elif args[0] == "report":
        cmd_report()
    elif args[0] == "jobs":
        cmd_jobs()
    else:
        raise SystemExit(__doc__)
