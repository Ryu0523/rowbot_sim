#!/usr/bin/env python3
"""
Regression check: does the plant in the working tree still reproduce the
recorded results bit for bit? The prototype of the test that must guard
the 'v0' (current) physics once an any-speed plant is added beside it.

Checks (each optional; read-only: tracked files and existing caches are only
read, everything written goes to studies/_cache/anyspeed/):
  missions  re-run rows of studies/final_eval.py (hand, m0) exactly as the
            meta7 run did (settings from results.pkl 'meta': gz plant,
            T 120, S 8, K 128, fc 0.1 / 24, trigger, steer auto, 2 L) and
            compare every field of the record with
            studies/_cache/final_eval/meta7/results.pkl (timing fields
            excepted), recursively, exact equality (NaN == NaN)
  turn      recompute final_eval.phase_turn (calm turning table) into a
            scratch path and compare with studies/_cache/final_eval/
            turn_gz.json
  lofi      the fingerprint sim/lofi.py keys the identified reduced model
            by, recomputed on the calm plant, against the one stored in
            studies/_cache/lofi_scarab195.json
  streams   re-record one onestep_gz mission (hand loop) and compare with
            studies/_cache/onestep_gz/streams_T120.pkl

Also wraps PlaningVesselGZ.step (timing only, numerically transparent) to
report the plant's share of a closed-loop control step and the substep
statistics in real missions. The versioning checks (frozen files, hull-key
selection, the any-speed plant with its switches off against these same
references) are studies/anyspeed_regress_as.py.

    .venv/Scripts/python.exe -m studies.anyspeed_regress \\
        --rows hand,m0 --missions s0_l0,s0_l1 --turn --lofi --streams
"""
import argparse
import json
import math
import os
import pickle
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np   # noqa: E402

OUT = os.path.join(HERE, "_cache", "anyspeed")
META7 = os.path.join(HERE, "_cache", "final_eval", "meta7", "results.pkl")
TURN = os.path.join(HERE, "_cache", "final_eval", "turn_gz.json")
LOFI = os.path.join(HERE, "_cache", "lofi_scarab195.json")
STREAMS = os.path.join(HERE, "_cache", "onestep_gz", "streams_T120.pkl")
TIMING = {"wall_s", "t_ctrl_step", "t_learn_step", "t_step"}
ADDED = set()


def _out():
    os.makedirs(OUT, exist_ok=True)
    return OUT


def diff(a, b, path="", out=None, tol=0.0):
    """Paths where a and b differ (exactly, NaN equal to NaN)."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k in TIMING and path.count("/") == 0:
                continue
            if k not in b and path == "":
                # a field final_eval has added to its record since the
                # reference run (steer, hdg_dev_*): new, not a difference
                ADDED.add(k)
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing on one side")
                continue
            diff(a[k], b[k], f"{path}/{k}", out, tol)
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, tol)
        return out
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        A, B = np.asarray(a), np.asarray(b)
        if A.shape != B.shape:
            out.append(f"{path}: shape {A.shape} vs {B.shape}")
        elif A.dtype.kind in "fc":
            same = (A == B) | (np.isnan(A) & np.isnan(B))
            if not np.all(same):
                out.append(f"{path}: max |diff| "
                           f"{np.nanmax(np.abs(A - B)):.3e}")
        elif not np.array_equal(A, B):
            out.append(f"{path}: arrays differ")
        return out
    if isinstance(a, float) and isinstance(b, float):
        if not (a == b or (math.isnan(a) and math.isnan(b))):
            out.append(f"{path}: {a!r} vs {b!r}")
        return out
    if a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


class StepTimer:
    def __init__(self):
        self.t, self.n, self.nsub = 0.0, 0, []

    def wrap(self, cls):
        orig = cls.step
        me = self

        def step(self, s, t, thr, rud, dt=None):
            t0 = time.perf_counter()
            out = orig(self, s, t, thr, rud, dt)
            me.t += time.perf_counter() - t0
            me.n += 1
            me.nsub.append(self.last_n_sub)
            return out
        cls.step = step


def setup(meta):
    from studies import final_eval as FE
    FE.OPT.update(plant=meta["plant"], T=float(meta["T"]), S=int(meta["S"]),
                  K=int(meta["K"]), device="cpu", fc_lam=float(meta["fc_lam"]),
                  fc_hp=int(meta["fc_hp"]), trigger=bool(meta["trigger"]),
                  min_gb=1.5, steer=meta["steer"],
                  y_lengths=float(meta["y_lengths"]), tag="")
    FE.out_dir = _out
    return FE


def check_missions(FE, rows, keys, R, timer, report):
    turn = json.load(open(TURN))       # the recorded table, as meta7 used
    for row in rows:
        spec, why = FE.resolve_row(row)
        if spec is None:
            raise SystemExit(why)
        nets = FE._row_nets(spec)
        for key in keys:
            seed, leg = int(key[1:key.index("_")]), int(key[-1])
            t0, tp0, n0 = time.time(), timer.t, timer.n
            k0 = len(timer.nsub)
            r = FE.run_mission(spec, seed, leg, FE.OPT["T"], turn, nets)
            wall = time.time() - t0
            ref = R["rows"][row][key]
            d = diff(r, {k: v for k, v in ref.items()})
            ns = np.asarray(timer.nsub[k0:])
            rec = dict(row=row, key=key, identical=not d, n_diff=len(d),
                       diffs=d[:20], added_fields=sorted(ADDED),
                       n_compared=len(set(r) & set(ref) - TIMING),
                       kn=r["kn"], kn_ref=ref["kn"],
                       wall_s=wall, plant_s=timer.t - tp0,
                       plant_steps=timer.n - n0,
                       plant_ms_per_ctrl_step=1e3 * (timer.t - tp0)
                       / max(r["n_steps"], 1),
                       ctrl_ms_per_ctrl_step=1e3 * r["t_ctrl_step"],
                       n_sub_mean=float(ns.mean()) if ns.size else None,
                       n_sub_max=int(ns.max()) if ns.size else None,
                       share_sub=float(np.mean(ns > 1)) if ns.size else None)
            report["missions"].append(rec)
            print(f"{row} {key}: identical {rec['identical']} "
                  f"({len(d)} diffs) kn {r['kn']:.6f} vs {ref['kn']:.6f}; "
                  f"plant {rec['plant_ms_per_ctrl_step']:.1f} ms / control "
                  f"step, ctrl {rec['ctrl_ms_per_ctrl_step']:.1f} ms, "
                  f"n_sub mean {rec['n_sub_mean']:.3f} max "
                  f"{rec['n_sub_max']}, wall {wall:.0f} s", flush=True)
            for line in d[:10]:
                print("   ", line)


def check_turn(FE, report):
    ref = json.load(open(TURN))
    scratch = os.path.join(OUT, "turn_gz_recomputed.json")
    FE.turn_path = lambda: scratch
    tab = FE.phase_turn(force=True)
    d = diff({k: tab[k] for k in ("u", "r", "throttle", "plant", "dt")},
             {k: ref[k] for k in ("u", "r", "throttle", "plant", "dt")})
    report["turn"] = dict(identical=not d, diffs=d[:20])
    print(f"turn table: identical {not d} {d[:5]}", flush=True)


def check_lofi(report):
    from learn.repro import task
    from sim import config, lofi
    c = task.ctx()
    plant, _ = config.calm_plant(c["h"], db=c["db"])
    fp = lofi._fingerprint(plant)
    stored = json.load(open(LOFI))
    ok = abs(stored["fingerprint"] - fp) <= 1e-9 * max(1.0, abs(fp))
    report["lofi"] = dict(fingerprint=fp, stored=stored["fingerprint"],
                          match=bool(ok), exact=fp == stored["fingerprint"],
                          version=stored.get("version"))
    print(f"lofi fingerprint {fp!r} stored {stored['fingerprint']!r}: "
          f"match {ok} exact {fp == stored['fingerprint']}", flush=True)


def check_streams(report, key=(0, 0)):
    from studies import onestep_gz as OG
    S = pickle.load(open(STREAMS, "rb"))
    st = OG.record(*key, 120.0, sea=None)
    d = diff({k: st[k] for k in ("xs", "t", "u", "apk", "hmin", "n",
                                 "consts", "kn")},
             {k: S[key][k] for k in ("xs", "t", "u", "apk", "hmin", "n",
                                     "consts", "kn")})
    report["streams"] = dict(key=list(key), identical=not d, diffs=d[:20],
                             n_keys=len(S))
    print(f"onestep stream {key}: identical {not d} {d[:5]}", flush=True)


GOLDEN = os.path.join(OUT, "golden_v0.npz")


def golden_runs(twin=None, extra=(), nozzle=True, calm_nozzle_deg=None):
    """Open-loop runs of the v0 GZ plant (seconds): the task sea of mission
    s0 leg 0 and calm water, starts 0 / 5 / 12 / 20 / 25 kn, 300 steps of
    0.02 s with thrust and nozzle commands that move (delay, lag, rate
    limit, prime and the substep path all exercised where they occur).
    twin(plant) -> the recording plant to run, from the Mission's plant;
    default the v0 GZ twin (studies/anyspeed_regress_as.py passes the
    any-speed plant). extra: names of further plant measurements appended
    to each row of the _m arrays (none for golden_v0). nozzle=False: the
    nozzle command held at zero (a switch set whose turning runs are not yet
    physical, P1b review: lateral_phi0 before S5); the default is the
    golden's own runs. calm_nozzle_deg: the calm runs' nozzle command held
    at this angle instead (S5a review: a golden of a lateral switch whose
    calm runs stay upright pins nothing that switch changes); None (the
    default) leaves the runs as they were."""
    from learn.repro import task
    from sim.planing_vessel_gz import from_plant
    from sim.test_vessel import Monochromatic
    m = task.Mission("high", 0, 0, t_end=1.0, track=task.LEGS[0])
    base = (from_plant(m.plant, record=True) if twin is None
            else twin(m.plant))
    out = {}
    for sea_name, pl in (("task", base),
                         ("calm", base.with_sea(Monochromatic(1.0, 0.0)))):
        for kn in (0.0, 5.0, 12.0, 20.0, 25.0):
            s = pl.initial_state(kn * 0.514444)
            t, tm, rm = 0.0, pl.prop.t_max, pl.rudder.max
            S, M = [s.copy()], []
            for i in range(300):
                thr = tm * (0.55 + 0.4 * np.sin(0.05 * i))
                rud = rm * 0.8 * np.sin(0.03 * i) if nozzle else 0.0
                if calm_nozzle_deg is not None and sea_name == "calm":
                    rud = float(np.radians(calm_nozzle_deg))
                s = pl.step(s, t, thr, rud, pl.dt)
                t += pl.dt
                S.append(s.copy())
                M.append([pl.last_cg_acc, pl.last_bow_acc, pl.last_rel_bow,
                          pl.last_slam_force, pl.last_n_sub,
                          pl.last_submergence, pl.prop.prime]
                         + [getattr(pl, a) for a in extra])
            k = f"{sea_name}_{kn:g}"
            out[k + "_s"] = np.asarray(S)
            out[k + "_m"] = np.asarray(M, float)
    return out


def check_golden(mode, report):
    t0 = time.time()
    g = golden_runs()
    if mode == "write":
        if os.path.exists(GOLDEN):
            raise SystemExit(f"{GOLDEN} exists; delete it first")
        np.savez(GOLDEN, **g)
        report["golden"] = dict(written=GOLDEN, keys=len(g))
        print(f"golden written: {len(g)} arrays, {time.time() - t0:.1f} s")
        return
    ref = np.load(GOLDEN)
    d = diff({k: g[k] for k in g}, {k: ref[k] for k in ref.files})
    report["golden"] = dict(identical=not d, diffs=d[:20],
                            seconds=time.time() - t0)
    print(f"golden v0: identical {not d} ({time.time() - t0:.1f} s) {d[:5]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", choices=("write", "check"), default=None)
    ap.add_argument("--rows", default="hand,m0")
    ap.add_argument("--missions", default="s0_l0,s0_l1")
    ap.add_argument("--turn", action="store_true")
    ap.add_argument("--lofi", action="store_true")
    ap.add_argument("--streams", action="store_true")
    ap.add_argument("--report", default="regress.json",
                    help="file name of the report in studies/_cache/anyspeed")
    a = ap.parse_args()
    _out()
    R = pickle.load(open(META7, "rb"))
    FE = setup(R["meta"])
    from sim.planing_vessel_gz import PlaningVesselGZ
    timer = StepTimer()
    timer.wrap(PlaningVesselGZ)
    report = dict(meta=R["meta"], missions=[])
    if a.golden:
        check_golden(a.golden, report)
    if a.lofi:
        check_lofi(report)
    rows = [r for r in a.rows.split(",") if r]
    keys = [k for k in a.missions.split(",") if k]
    if rows and keys:
        check_missions(FE, rows, keys, R, timer, report)
    if a.streams:
        check_streams(report)
    if a.turn:
        check_turn(FE, report)
    with open(os.path.join(OUT, a.report), "w") as f:
        json.dump(report, f, indent=1, default=str)


if __name__ == "__main__":
    main()
