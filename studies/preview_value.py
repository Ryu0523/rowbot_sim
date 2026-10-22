#!/usr/bin/env python3
"""
Preview-value study: is a perfect preview of the oncoming waves worth
anything to the frozen plant scarab195@as1 under the envelope budgets, and
how much of that value survives when only the past wave at the bow, or only
the current motion, is known? (learn/meta/RETROSPECTIVE_2026-10-17.md exit 4;
the "按浪群提前减速未测" item of WAVE_PHASE_2026-10-05.)

Design (studies/preview_value_run.py): the envelope study's set-up, the P+I
throttle tracking a two-level speed reference v_hi / v_lo switched by a
rule that knows the future (preview_h, preview_group), the past (ar_h) or
the present (reactive). Same seas and seeds as studies/envelope_pass1, so
its constant-speed runs are the baseline. The comparison is on the plane
(mean speed, event rates): a rule has value when its point lies above the
constant-speed curve, i.e. higher mean speed at the same rate of >= 7 g
peaks (or >= 5 g, >= 3 g, A1/10, bow immersions).

Batches:
  h10head   Hs 1.0 m Tp 5 s head seas (the binding case of pass 1):
            preview_h  v_hi 18/20/22/25 -> v_lo 15, thr 1.0/1.3/1.6 x Hs,
                       horizon 8 s
            preview_group v_hi 20/25, thr 1.0/1.25 x Hs/4, horizon 15 s,
                       hold 5 s
            ar_h       v_hi 20/25, thr 1.3, horizon 8 s (AR order 16 on
                       40 s of history)
            reactive   v_hi 20/25, thr 1.5 m/s emergence speed, hold 3 s
  h10bowq   bow-quartering seas, preview_h v_hi 20/25 thr 1.3
  h15head   Hs 1.5 m Tp 6 s head seas, preview_h v_hi 15/18 -> v_lo 12,
            thr 1.0/1.3 (can a preview make planing possible where no
            constant speed meets the budget?)

    python -m studies.preview_value run [h10head h10bowq h15head]
    python -m studies.preview_value report
    python -m studies.preview_value jobs

Outputs: studies/_cache/preview/runs/<name>.json + .npz, progress.log,
summary.json, tables.md. Read-only use of sim/, hydro/, app/ and the
envelope cache.
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

from studies import preview_value_run as PR          # noqa: E402
from studies import envelope_pass1 as P              # noqa: E402
from studies import envelope_stats as S              # noqa: E402

OUT = os.path.join(ROOT, "studies", "_cache", "preview")
RUNS = os.path.join(OUT, "runs")
KN = PR.KN
SEEDS = (0, 1)
SEAS = P.SEAS
HEADINGS = P.HEADINGS
I_CLIP = P.I_CLIP
MAX_PROCS = 8
GB_PER_JOB = 0.25


def _job(name, batch, sea_tag, hd, kind, v_hi, v_lo, thr, horizon, hold,
         sd, **extra):
    a = dict(name=name, batch=batch, sea=sea_tag, heading=hd, kind=kind,
             v_hi_kn=float(v_hi), v_lo_kn=float(v_lo), thr=float(thr),
             horizon=float(horizon), hold=float(hold), seed=sd, sea_seed=sd,
             randomize=False, i_clip=I_CLIP, **SEAS[sea_tag],
             **HEADINGS[hd])
    a.update(extra)
    return a


def jobs(batches=("h10head", "h10bowq", "h15head")):
    out = []
    for b in batches:
        if b == "h10head":
            sea, hd = "h10", "head"
            for vh in (18.0, 20.0, 22.0, 25.0):
                for thr in (1.0, 1.3, 1.6):
                    for sd in SEEDS:
                        out.append(_job(
                            f"{sea}_{hd}_ph_{vh:g}to15_thr{thr:g}_s{sd}", b,
                            sea, hd, "preview_h", vh, 15.0, thr, 8.0, 1.0,
                            sd))
            for vh in (20.0, 25.0):
                for thr in (1.0, 1.25):
                    for sd in SEEDS:
                        out.append(_job(
                            f"{sea}_{hd}_pg_{vh:g}to15_thr{thr:g}_s{sd}", b,
                            sea, hd, "preview_group", vh, 15.0, thr, 15.0,
                            5.0, sd))
            for vh in (20.0, 25.0):
                for sd in SEEDS:
                    out.append(_job(
                        f"{sea}_{hd}_ar_{vh:g}to15_thr1.3_s{sd}", b, sea,
                        hd, "ar_h", vh, 15.0, 1.3, 8.0, 1.0, sd))
                    out.append(_job(
                        f"{sea}_{hd}_re_{vh:g}to15_thr1.5_s{sd}", b, sea,
                        hd, "reactive", vh, 15.0, 1.5, 8.0, 3.0, sd))
        elif b == "h10group":
            # confirmation of the one configuration that looked better than
            # the baseline in h10head (preview_group 20 -> 15, thr 1.0):
            # neighbours and two more wave seeds, plus constant 17 kn on
            # the new seeds for a matched-speed comparison
            sea, hd = "h10", "head"
            for vh in (18.0, 20.0, 22.0):
                for sd in (0, 1, 2, 3):
                    out.append(_job(
                        f"{sea}_{hd}_pg_{vh:g}to15_thr1_s{sd}", b, sea, hd,
                        "preview_group", vh, 15.0, 1.0, 15.0, 5.0, sd))
            for sd in (2, 3):
                out.append(_job(f"{sea}_{hd}_const_17_s{sd}", b, sea, hd,
                                "const", 17.0, 17.0, 0.0, 8.0, 1.0, sd))
        elif b == "h10learned":
            # the impact predictors of studies/bow_camera_train in the loop:
            # slow down when the predicted largest CG acceleration in the
            # next 5 s exceeds thr (g); 'future' = exact future wave + motion
            # (the ceiling), 'image' = bow-camera range image + past + motion
            sea, hd = "h10", "head"
            for vh in (18.0, 20.0):
                for thr in (2.5, 3.0):
                    for sd in SEEDS:
                        out.append(_job(
                            f"{sea}_{hd}_lf_{vh:g}to15_thr{thr:g}_s{sd}", b,
                            sea, hd, "learned_future", vh, 15.0, thr, 10.0,
                            1.0, sd, net="future+motion"))
            for sd in SEEDS:
                out.append(_job(
                    f"{sea}_{hd}_li_20to15_thr2.5_s{sd}", b, sea, hd,
                    "learned_image", 20.0, 15.0, 2.5, 10.0, 1.0, sd,
                    net="image+past+motion"))
        elif b == "h10learned2":
            # is the gain of the learned rule from the preview or from the
            # motion history alone? 'motion' net (IMU only, no wave
            # knowledge), plus two more seeds of the two best future-net rows
            sea, hd = "h10", "head"
            for vh in (18.0, 20.0):
                for sd in SEEDS:
                    out.append(_job(
                        f"{sea}_{hd}_lm_{vh:g}to15_thr3_s{sd}", b, sea, hd,
                        "learned_motion", vh, 15.0, 3.0, 10.0, 1.0, sd,
                        net="motion"))
            for vh in (18.0, 20.0):
                for sd in (2, 3):
                    out.append(_job(
                        f"{sea}_{hd}_lf_{vh:g}to15_thr3_s{sd}", b, sea, hd,
                        "learned_future", vh, 15.0, 3.0, 10.0, 1.0, sd,
                        net="future+motion"))
        elif b == "h10bowq":
            sea, hd = "h10", "bowq"
            for vh in (20.0, 25.0):
                for sd in SEEDS:
                    out.append(_job(
                        f"{sea}_{hd}_ph_{vh:g}to15_thr1.3_s{sd}", b, sea,
                        hd, "preview_h", vh, 15.0, 1.3, 8.0, 1.0, sd))
        elif b == "h15head":
            sea, hd = "h15", "head"
            for vh in (15.0, 18.0):
                for thr in (1.0, 1.3):
                    for sd in SEEDS:
                        out.append(_job(
                            f"{sea}_{hd}_ph_{vh:g}to12_thr{thr:g}_s{sd}", b,
                            sea, hd, "preview_h", vh, 12.0, thr, 8.0, 1.0,
                            sd))
        else:
            raise ValueError(b)
    return out


def paths(name):
    return (os.path.join(RUNS, name + ".json"),
            os.path.join(RUNS, name + ".npz"))


def done(name):
    pj, pn = paths(name)
    return os.path.exists(pj) and os.path.exists(pn)


def job(arg):
    os.makedirs(RUNS, exist_ok=True)
    name = arg["name"]
    pj, pn = paths(name)
    t0 = time.time()
    if done(name):
        with open(pj) as f:
            return json.load(f)
    r = PR.run_one(arg)
    st = r.pop("stats", None)
    series = r.pop("series", None)
    arrays = {}
    light = dict(r)
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
        json.dump(light, f, indent=1, default=P._plain)
    with open(os.path.join(OUT, "progress.log"), "a") as f:
        s = light.get("stats") or {}
        f.write(f"{time.strftime('%H:%M:%S')} {name} done in "
                f"{time.time() - t0:.0f} s ({light['ms_per_step']:.2f} ms/"
                f"step) reason {light['reason']} u "
                f"{s.get('u_mean_kn', float('nan')):.2f} kn low "
                f"{s.get('low_frac', float('nan')):.2f} slow/h "
                f"{s.get('slow_per_hour', float('nan')):.0f} cg a110 "
                f"{(s.get('cg') or {}).get('a110', float('nan')):.2f}\n")
    return light


def cmd_run(batches):
    from studies.parallel_runner import free_gb, n_workers, run_parallel
    os.makedirs(RUNS, exist_ok=True)
    todo = [a for a in jobs(batches) if not done(a["name"])]
    n = n_workers(GB_PER_JOB, 1.5, MAX_PROCS)
    msg = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} run {list(batches)}: "
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
def load_run(name, runs_dir=RUNS):
    pj = os.path.join(runs_dir, name + ".json")
    pn = os.path.join(runs_dir, name + ".npz")
    with open(pj) as f:
        r = json.load(f)
    z = np.load(pn)
    r["peaks"] = {g: z[f"peaks_{g}"] for g in ("cg", "bow")
                  if f"peaks_{g}" in z}
    return r


def pooled(runs, label):
    """P.condition_row over the seeds plus the study's own numbers."""
    row = P.condition_row(runs, label)
    ok = [r for r in runs if r.get("stats")]
    for key in ("low_frac", "slow_per_hour", "trig_frac", "thr_zero_frac"):
        row[key] = (float(np.mean([r["stats"].get(key, np.nan)
                                   for r in ok])) if ok else np.nan)
    return row


def baseline_rows(sea_tag, hd, speeds):
    rows = []
    for kn in speeds:
        names = [f"{sea_tag}_{hd}_{kn:g}kn_s{sd}" for sd in SEEDS]
        if not all(os.path.exists(os.path.join(P.RUNS, nm + ".json"))
                   for nm in names):
            continue
        runs = [P.load_run(nm) for nm in names]
        row = pooled(runs, kn)
        row.update(kind="const", v_hi=kn, v_lo=kn, thr=np.nan)
        rows.append(row)
    return rows


def study_rows(batch):
    groups = {}
    for a in jobs((batch,)):
        if not done(a["name"]):
            continue
        key = a["name"].rsplit("_s", 1)[0]
        groups.setdefault(key, (a, []))[1].append(load_run(a["name"]))
    rows = []
    for key, (a, runs) in groups.items():
        row = pooled(runs, key)
        row.update(kind=a["kind"], v_hi=a["v_hi_kn"], v_lo=a["v_lo_kn"],
                   thr=a["thr"], horizon=a["horizon"], hold=a["hold"])
        rows.append(row)
    rows.sort(key=lambda r: (r["kind"], r["v_hi"], r["thr"]))
    return rows


def _lev(row, g, k, lev):
    t = row[g]["tail"]
    if not t.get("ok"):
        return None
    return t["levels"].get(f"k{k:g}_{lev:g}g")


KIND_CN = {"const": "匀速（基线）", "preview_h": "完美预见·下一个浪高",
           "learned_future": "学到的冲击预测·未来浪面真值",
           "learned_image": "学到的冲击预测·摄像头距离图",
           "learned_motion": "学到的冲击预测·只用运动历史",
           "preview_group": "完美预见·浪群", "ar_h": "只知过去·外推",
           "reactive": "只知当下·反应"}


def table(rows, title):
    out = [f"### {title}", "",
           "| 规则 | 高/低航速 (节) | 阈值 | 平均航速 (节) | 低速时间占比 | "
           "减速次数/时 | 重心 A1/10 (g) | ≥3 g 次/时 (计数) | ≥5 g 次/时 | "
           "≥7 g 次/时 [5–95%] | 船头进水 次/时 | 最大横摇 (度) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        l3, l5, l7 = (_lev(r, "cg", 1.0, v) for v in (3.0, 5.0, 7.0))
        thr = "—" if not np.isfinite(r.get("thr", np.nan)) else f"{r['thr']:g}"
        vv = (f"{r['v_hi']:g}" if r["kind"] == "const"
              else f"{r['v_hi']:g} → {r['v_lo']:g}")
        lf = r.get("low_frac", np.nan)
        out.append(
            f"| {KIND_CN.get(r['kind'], r['kind'])} | {vv} | {thr} | "
            f"{r['u_mean_kn']:.1f} | "
            f"{'—' if not np.isfinite(lf) else f'{lf:.2f}'} | "
            f"{P._num(r.get('slow_per_hour', np.nan))} | "
            f"{r['cg']['a110']:.2f} | "
            f"{P._num(l3['rate_direct']) if l3 else '—'} | "
            f"{P._fmt_rate(l5)} | {P._fmt_rate(l7)} | "
            f"{P._num(r['immersion_per_hour'])} | {r['roll_max_deg']:.1f} |")
    out.append("")
    return "\n".join(out)


def cmd_report():
    os.makedirs(OUT, exist_ok=True)
    md = ["# 预见价值研究：结果表", "",
          f"生成时间 {time.strftime('%Y-%m-%d %H:%M')}。设定见 "
          "studies/preview_value.py；基线行取自 studies/_cache/envelope。"
          "率的方括号是广义帕累托尾部外推的 5–95% 自举区间，"
          "“(计数)”为直接计数。", ""]
    summary = {}
    for batch, sea_tag, hd, speeds in (
            ("h10head", "h10", "head", (15.0, 16.0, 17.0, 18.0, 20.0, 22.0,
                                        25.0)),
            ("h10group", "h10", "head", (15.0, 16.0, 17.0, 18.0, 20.0)),
            ("h10learned", "h10", "head", (15.0, 16.0, 17.0, 18.0, 20.0)),
            ("h10learned2", "h10", "head", (15.0, 16.0, 17.0, 18.0, 20.0)),
            ("h10bowq", "h10", "bowq", (15.0, 18.0, 20.0, 22.0, 25.0)),
            ("h15head", "h15", "head", (10.0, 12.0, 15.0, 18.0))):
        srows = study_rows(batch)
        if not srows:
            continue
        brows = baseline_rows(sea_tag, hd, speeds)
        sea = SEAS[sea_tag]
        title = (f"{batch}: Hs {sea['hs']} m, Tp {sea['tp']} s, 相对浪向 "
                 f"{HEADINGS[hd]['rel_deg']} 度")
        md.append(table(brows + srows, title))
        summary[batch] = dict(baseline=brows, study=srows)
    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=P._plain)
    print("\n".join(md))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "jobs"
    if cmd == "run":
        cmd_run(tuple(sys.argv[2:]) or ("h10head",))
    elif cmd == "report":
        cmd_report()
    elif cmd == "jobs":
        cmd_jobs()
    else:
        raise SystemExit(__doc__)
