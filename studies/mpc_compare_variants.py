#!/usr/bin/env python3
"""
Cheaper uses of the learned error model in the same MPPI controller
(learn/meta/mpc_variants.py; RELATED_WORK_CONTROL_USE.md variants C and A),
evaluated on the SAME 16 target missions as studies/mpc_compare.py
(task.EVAL_SEEDS x both legs, 120 s, the same sea, knot-noise and base-
noise streams per row) with the SAME impact link and impact horizon j_imp,
read from the main run's finished cache (studies/_cache/mpc_learned/
results.pkl, read only), so every row here is paired with the main run's
m0 and full-controller rows.

Rows (each with the net and spread of the main run's row --net, default
prior = model3_cov.pt zero-shot, s = 1):
  hold     step-0 error samples, decayed over the horizon (tau = --tau s)
  hold_c   step-0 error samples held constant (offset-free MPC)
  nominal  per-sample error trajectories along the warm-start plan
  sens     nominal + first-order knot sensitivities (7 nudged plans)
  m0, full optional reference rows run HERE with this harness's link (for
           --smoke, or to time them under the same load); the evaluation
           compares against the main run's own m0 / --net rows
Per row: score and its components, controller time per episode-step (the
lockstep call / rows, as the main report), learned-model sequences per
call and row (seq) and their length in steps (depth); the main run's rows
for the same missions are printed next to them when the main run has them
(with the same j_imp), then paired differences vs the main m0 and the
main --net row.

Cache: studies/_cache/mpc_variants/ (results.pkl, run.log). The link and
j_imp are taken once the main run's m0 evaluation rows carry its final
j_imp (hcheck_pre writes a provisional one; --pre-eval uses it anyway).
Rows are keyed by name (hold with another --tau: hold~tau0.5; another
--net: name@net); cached missions run with another link or other settings
(variant, tau, delta, trust, c_max, T) stop the run until --redo, and the
report shows only rows of the current link and --net.
--smoke:
mpc_variants_smoke/ with its own quick link (2 hand episodes, seed 100,
20 s, target world) and j_imp = 4 (impact_horizon's floor), one mission
(seed 0, leg 0), 10 s. GPU rows in lockstep groups of 8 missions (both
legs of 4 seeds), as the main run's GPU rows.

    python -m studies.mpc_compare_variants [--variants hold,hold_c,nominal,
        sens] [--net prior] [--missions 0-7] [--tau 1.0] [--smoke]
        [--T 10] [--redo row,...] [--report] [--pre-eval]
"""
import argparse
import contextlib
import json
import math
import os
import pickle
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                    # noqa: E402

from learn.repro import task                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "_cache", "mpc_learned", "results.pkl")
SMOKE = False
SMOKE_J_IMP = 4
ROWS = ("hold", "hold_c", "nominal", "sens")
REFS = ("m0", "full")
TAU0 = 1.0                 # s, hold's default tau (mpc_variants.TAU)
OPT = dict(net="prior", missions=None, tau=TAU0, T=None, gpu_frac=0.2,
           group=8, pre_eval=False, j_imp=None)


def cache():
    d = os.path.join(HERE, "_cache",
                     "mpc_variants_smoke" if SMOKE else "mpc_variants")
    os.makedirs(d, exist_ok=True)
    return d


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(cache(), "run.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_R():
    p = os.path.join(cache(), "results.pkl")
    return pickle.load(open(p, "rb")) if os.path.exists(p) else {}


def save(R):
    p = os.path.join(cache(), "results.pkl")
    pickle.dump(R, open(p + ".tmp", "wb"))
    os.replace(p + ".tmp", p)


@contextlib.contextmanager
def _open_linked(path, tmp_dir):
    """path open for reading through a temporary hard link in tmp_dir
    (same volume), removed afterwards. Windows refuses os.replace onto a
    file open under its own name (even with FILE_SHARE_DELETE) but not
    onto one open under another link, so the running comparison's save
    (os.replace onto results.pkl) is never blocked by this read; only the
    link creation touches the name (microseconds). If no link can be made:
    a plain open, held for the raw read only."""
    lk = os.path.join(tmp_dir, f"main_results.{os.getpid()}.link")
    try:
        if os.path.exists(lk):
            os.remove(lk)
        os.link(path, lk)
    except OSError:
        lk = None
    try:
        with open(lk or path, "rb") as f:
            yield f
    finally:
        if lk and os.path.exists(lk):
            os.remove(lk)


def load_main():
    """The main run's results (read only; {} while it has none), read
    through _open_linked and unpickled after the file is closed."""
    if not os.path.exists(MAIN):
        return {}
    try:
        with _open_linked(MAIN, cache()) as f:
            data = f.read()
        return pickle.loads(data)
    except Exception as ex:          # mid-write by the running comparison
        log(f"  main results unreadable now ({ex!r}); retry later")
        return {}


def eval_jobs():
    if SMOKE:
        return [dict(seed=0, leg=0)]
    seeds = OPT["missions"] or task.EVAL_SEEDS
    return [dict(seed=int(s), leg=leg) for s in seeds for leg in (0, 1)]


def T_ep():
    return OPT["T"] or (10.0 if SMOKE else task.T_EVAL)


def row_kw(name):
    """(variant, settings) of a row."""
    if name == "hold":
        return "hold", dict(tau=OPT["tau"])
    if name == "hold_c":
        return "hold", dict(tau=math.inf)
    return name, {}


def row_key(name):
    """A row's cache key: hold with another tau than TAU0 carries it
    (hold~tau0.5), rows of another net than prior carry it (@oracle)."""
    key = name
    if name == "hold" and OPT["tau"] != TAU0:
        key += f"~tau{OPT['tau']:g}"
    if OPT["j_imp"] is not None and name != "m0":
        key += f"~j{OPT['j_imp']}"
    return key if OPT["net"] == "prior" or name == "m0" else \
        f"{key}@{OPT['net']}"


SIG = ("variant", "tau", "delta", "trust", "c_max", "T")


def want(name):
    """The settings a row's cached missions must have been run with (the
    fields of SIG that define it)."""
    if name in REFS:
        return dict(variant=name, T=T_ep())
    from learn.meta import mpc_variants as MV
    variant, kw = row_kw(name)
    return dict(MV.settings(variant, **kw), T=T_ep())


def _sig(r):
    return tuple(r.get(q) for q in SIG)


# ------------------------------------------------------------ link
def smoke_link(R):
    """The smoke's own quick link: 2 hand episodes (seed 100, both legs,
    20 s, target world), fit_impact_link with n_min 3."""
    from learn.meta.mpc_learned import ImpactLink, fit_impact_link
    from studies import mpc_compare as MC
    if "link" not in R:
        task.ctx()
        rows = [MC.job_hand(dict(seed=100, leg=leg, T=20.0))
                for leg in (0, 1)]
        recs = [dict(r["log"], leg=r["leg"]) for r in rows if r["finite"]]
        lk, info = fit_impact_link(recs, n_min=3, log=log)
        R["link"] = dict(link=lk.to_dict(), n_eps=len(recs))
        json.dump(R["link"], open(os.path.join(cache(), "link.json"), "w"),
                  indent=1)
        save(R)
    d = dict(R["link"]["link"], j_imp=SMOKE_J_IMP)
    return ImpactLink.from_dict(d), False


def main_link(RM, strict=True):
    """The main run's link with its evaluation impact horizon, and whether
    its horizon check stopped (rows then not interpretable). strict False
    (the report): a provisional j_imp is used with a note."""
    from learn.meta.mpc_learned import ImpactLink
    if "link" not in RM or "j_imp" not in RM.get("hcheck", {}):
        raise SystemExit(
            "the main run (studies/_cache/mpc_learned) has no impact link / "
            "horizon check yet; wait for its hcheck phase, or use --smoke")
    j_imp = int(RM["hcheck"]["j_imp"])
    # hcheck_pre already writes a j_imp; the evaluation's is re-decided by
    # the later hcheck phase, and the main m0 rows (run right after it)
    # record it
    final = any(r.get("j_imp") == j_imp
                for r in RM.get("eval", {}).get("m0", {}).values())
    if OPT["j_imp"] is not None:
        # --j-imp: this row's own impact horizon (e.g. the net's own first
        # bad step instead of the one horizon decided on all nets); the
        # rows carry ~j<N> in their key and are not paired with main rows
        # of another horizon
        j_imp, final = int(OPT["j_imp"]), True
    if not final and not OPT["pre_eval"] and not strict:
        log(f"  note: the main run's j_imp = {j_imp} is still provisional")
    elif not final and not OPT["pre_eval"]:
        raise SystemExit(
            f"the main run's impact horizon (j_imp = {j_imp}) is not yet the "
            "evaluation's (no main m0 evaluation row with it; the hcheck "
            "phase may still change it); wait, or pass --pre-eval")
    d = dict(RM["link"]["link"], j_imp=j_imp)
    return ImpactLink.from_dict(d), bool(RM["hcheck"].get("early"))


def current_link(R, RM):
    """The link rows are run with now (None before there is one)."""
    if SMOKE:
        return smoke_link(R)[0] if "link" in R else None
    if "link" not in RM or "j_imp" not in RM.get("hcheck", {}):
        return None
    return main_link(RM, strict=False)[0]


# ------------------------------------------------------------ rows
def run_rows(R, RM, names):
    from learn.meta import mpc_learned as ML
    from learn.meta import mpc_variants as MV
    from studies import mpc_compare as MC
    link, early = smoke_link(R) if SMOKE else main_link(RM)
    MC.OPT["gpu_frac"] = OPT["gpu_frac"]
    ev = R.setdefault("eval", {})
    for name in names:
        key = row_key(name)
        e = ev.setdefault(key, {})
        w = want(name)
        stale = [k for k, r in e.items() if r.get("link") != link.to_dict()
                 or any(r.get(q) != v for q, v in w.items())]
        if stale:
            raise SystemExit(f"{key}: {len(stale)} cached rows were run with "
                             f"another link / impact horizon or other "
                             f"settings than {w}; rerun with --redo {name}")
        todo = [j for j in eval_jobs() if (j["seed"], j["leg"]) not in e]
        if not todo:
            continue
        if task.avail_gb() < 1.5:
            raise MemoryError(f"only {task.avail_gb():.1f} GB of RAM left")
        if name == "m0":
            net, s, stats, groups = None, 1.0, None, [todo]
            en = ML.env()
        else:
            g = MC.gpu()
            en = g["env"]
            net, s, stats = MC.net_of(RM, OPT["net"])
            G = OPT["group"]
            groups = [todo[i:i + G] for i in range(0, len(todo), G)]
        for gi, jobs in enumerate(groups):
            t0 = time.time()
            tag = f"{key} group {gi}"
            if name in ("m0", "full"):
                rows = ML.run_group(jobs, en, link, net=net, stats=stats,
                                    spread=s, T=T_ep(), log=log, tag=tag)
                nseq = 0 if net is None else ML.K_CAND * ML.S_SAMP
                for r in rows:
                    r.update(n_seq=nseq, depth=0 if net is None else ML.H,
                             n_seq_steps=nseq * ML.H, variant=name)
            else:
                variant, kw = row_kw(name)
                rows = MV.run_group_v(jobs, en, link, variant, net, stats,
                                      spread=s, T=T_ep(), log=log, tag=tag,
                                      **kw)
            for r in rows:
                r.update(link=link.to_dict(), j_imp=link.j_imp,
                         not_interpretable=early, T=T_ep(),
                         net=None if net is None else OPT["net"],
                         spread=None if net is None else s)
                e[(r["seed"], r["leg"])] = r
                _log_row(key, r)
            save(R)
            log(f"  {key} group {gi}: {len(jobs)} missions in "
                f"{(time.time() - t0) / 60:.1f} min, "
                f"{rows[0]['t_step_mean']:.3f} s per episode-step "
                f"({rows[0]['t_call_mean']:.3f} s per call, p95 "
                f"{rows[0]['t_call_p95']:.3f}), {rows[0]['n_seq']} learned "
                f"sequences of depth {rows[0]['depth']} per row and call, "
                f"GPU peak {rows[0]['gpu_peak_gb']:.2f} GB")


def _log_row(name, r):
    log(f"    {name} seed {r['seed']} leg {r['leg']}: score {r['score']:.3f}"
        f" (speed {r['c_speed']:.3f}, impact {r['c_impact']:.3f}, track "
        f"{r['c_track']:.3f}), {r['kn']:.1f} kn, p99 {r['acc_cg_p99']:.2f} "
        f"g, {r['t_step_mean']:.3f} s per episode-step, non-finite samples "
        f"{r['n_bad']}, capped {r['n_clip']}")


# ------------------------------------------------------------ report
def _arr(rows, keys, q):
    return np.array([rows[k][q] if k in rows and q in rows[k] else np.nan
                     for k in keys], float)


def _line(nm, e, keys):
    s = _arr(e, keys, "score")
    kn = _arr(e, keys, "kn")
    se = s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0
    bad = int(sum(not e[k]["finite"] for k in keys))
    sp = e[keys[0]].get("spread")
    sp = f"{sp:.2f}" if isinstance(sp, float) else "-"
    nb = sum(e[k].get("n_bad", 0) for k in keys)
    ns = sum(e[k].get("n_samp", 0) for k in keys)
    nf = f"{100 * nb / ns:.2f}" if ns else "-"
    seq = e[keys[0]].get("n_seq", "-")
    dep = e[keys[0]].get("depth", "-")
    star = "*" if any(e[k].get("not_interpretable") for k in keys) else ""
    kn0 = np.nanmean(kn[0::2]) if len(kn) > 0 else np.nan
    kn1 = np.nanmean(kn[1::2]) if len(kn) > 1 else np.nan
    return (f"    {nm + star:<16}{s.mean():>7.3f}+-{se:<4.3f}"
            f"{np.nanmean(_arr(e, keys, 'c_speed')):>7.3f}"
            f"{np.nanmean(_arr(e, keys, 'c_impact')):>7.3f}"
            f"{np.nanmean(_arr(e, keys, 'c_track')):>7.3f}"
            f"{kn0:>6.1f}{kn1:>6.1f}"
            f"{np.nanmean(_arr(e, keys, 'acc_cg_p99')):>6.2f}{bad:>4d}"
            f"{nf:>6}{sp:>6}"
            f"{np.nanmean(_arr(e, keys, 't_step_mean')):>8.3f}"
            f"{np.nanmean(_arr(e, keys, 't_step_p95')):>7.3f}"
            f"{seq!s:>5}{dep!s:>6}")


def report(R, RM):
    from learn.meta import mpc_learned as ML
    keys = [(j["seed"], j["leg"]) for j in eval_jobs()]
    ev = R.get("eval", {})
    link = current_link(R, RM)
    if link is None:
        log("  no impact link yet")
        return
    lk, j_imp = link.to_dict(), link.j_imp
    # only rows run with the current link and --net (m0 has no net) whose
    # missions all share one set of settings
    mine, other, part = [], [], []
    for nm, e in ev.items():
        if not all(k in e for k in keys):
            part.append(nm)
            continue
        rs = [e[k] for k in keys]
        ok = all(r.get("link") == lk and (r.get("variant") == "m0" or
                                          r.get("net") == OPT["net"])
                 for r in rs) and len({_sig(r) for r in rs}) == 1
        (mine if ok else other).append(nm)
    if other:
        log(f"  not shown (another link / impact horizon or net than now, "
            f"or missions with mixed settings): {', '.join(other)}")
    if part:
        log(f"  incomplete: {', '.join(part)}")
    if not mine:
        log("  no complete rows yet")
        return
    main_ev = RM.get("eval", {})
    main = []
    if not SMOKE:
        for nm in ("hand", "c0", "m0", "prior", "prior_s", "online",
                   "oracle"):
            e = main_ev.get(nm, {})
            if all(k in e for k in keys) and \
                    all(e[k].get("j_imp", j_imp) == j_imp for k in keys):
                main.append(nm)
    log(f"\n  VARIANTS, paired target missions ({len(keys)}), impact horizon "
        f"j_imp = {j_imp}, net '{OPT['net']}' (lower is better); mean +- SE")
    log(f"    {'row':<16}{'score':>13}{'speed':>7}{'impact':>7}{'track':>7}"
        f"{'kn 0':>6}{'kn 45':>6}{'p99 g':>6}{'bad':>4}{'nf%':>6}{'s':>6}"
        f"{'s/step':>8}{'p95':>7}{'seq':>5}{'depth':>6}")
    for nm in main:
        e = dict(main_ev[nm])
        if nm not in ("hand", "c0"):
            for k in keys:
                e[k] = dict(e[k], n_seq=0 if nm == "m0" else
                            ML.K_CAND * ML.S_SAMP,
                            depth=0 if nm == "m0" else ML.H)
        log(_line(f"{nm} (main)", e, keys))
    for nm in mine:
        log(_line(nm, ev[nm], keys))
    log("    notes: seq = learned-model sequences per controller call and "
        "episode row, depth = their length in control steps (the serial "
        "depth of the learned calls); s/step = controller wall time per "
        "episode-step (lockstep call / rows), GPU shared with whatever else "
        "ran; * = the main run's horizon check stopped (not interpretable)")
    refs = [(f"{r} (main)", main_ev[r]) for r in ("m0", OPT["net"])
            if r in main]
    refs += [(row_key(r), ev[row_key(r)]) for r in REFS
             if row_key(r) in mine]
    if not refs:
        return
    log("\n  paired differences (row - reference), mean +- SE, and as % of "
        "the reference mean")
    for rn, re_ in refs:
        b = _arr(re_, keys, "score")
        for nm in mine:
            if nm == rn:
                continue
            d = _arr(ev[nm], keys, "score") - b
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            log(f"    {nm:<16} - {rn:<14}: {d.mean():+.3f} +- {se:.3f} "
                f"({d.mean() / b.mean() * 100:+.0f}% +- "
                f"{se / b.mean() * 100:.0f}%)")


# ------------------------------------------------------------ main
def main(names, redo=(), only_report=False):
    R = load_R()
    RM = {} if SMOKE else load_main()
    for v in redo:
        if R.get("eval", {}).pop(row_key(v), None) is not None:
            log(f"  --redo: cached {row_key(v)} rows dropped")
    log(f"\n  [variants {','.join(names)}] net {OPT['net']}, "
        f"{len(eval_jobs())} missions, T {T_ep():.0f} s, "
        f"{task.avail_gb():.1f} GB RAM free")
    if not only_report:
        run_rows(R, RM, names)
        save(R)
    report(R, RM)
    return R


def _seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return tuple(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default=",".join(ROWS),
                    help="rows: " + ", ".join(ROWS + REFS))
    ap.add_argument("--net", default="prior",
                    help="the main run's net row whose net and spread the "
                         "variants use (prior, prior_s, oracle, online)")
    ap.add_argument("--missions", default=None,
                    help="evaluation seeds, e.g. 0-7 (both legs each)")
    ap.add_argument("--tau", type=float, default=TAU0,
                    help="hold's decay time constant, s")
    ap.add_argument("--T", type=float, default=None,
                    help="episode length, s (default 120; --smoke 10)")
    ap.add_argument("--group", type=int, default=8,
                    help="missions per lockstep group")
    ap.add_argument("--gpu-frac", type=float, default=0.2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pre-eval", action="store_true",
                    help="use the main run's j_imp before its evaluation "
                         "rows confirm it (rows go stale if hcheck changes "
                         "it)")
    ap.add_argument("--redo", default="")
    ap.add_argument("--j-imp", type=int, default=None,
                    help="impact horizon for these rows instead of the main "
                         "run's (keys get ~j<N>)")
    ap.add_argument("--report", action="store_true",
                    help="only print the report")
    a = ap.parse_args()
    SMOKE = a.smoke
    OPT.update(net=a.net, tau=a.tau, T=a.T, gpu_frac=a.gpu_frac,
               group=a.group, pre_eval=a.pre_eval, j_imp=a.j_imp,
               missions=_seeds(a.missions) if a.missions else None)
    names = tuple(v for v in a.variants.split(",") if v)
    for v in names:
        if v not in ROWS + REFS:
            raise SystemExit(f"unknown row {v}")
    main(names, tuple(v for v in a.redo.split(",") if v), a.report)
