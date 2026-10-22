#!/usr/bin/env python3
"""
Train/test protocol factors on in-family boats (learn/meta/ROADMAP_2026-09-30
section 11.12, candidates 3 and 4): does the frozen pretrained net leave a
part of the error that a copy trained on the boat's own missions removes
when the boat is DRIVEN like the gz missions, or when all its missions share
ONE SEA SPECTRUM like the gz missions? The same held-out catalogue laws
(meta8 held8 law ids) run under the four conditions of
studies/episode_proto.py:

  auto_redraw  episode8's protocol (data autopilot, sea drawn per mission)
  auto_task    data autopilot, the task's sea (Hs 1.0 m, Tp 5 s) every mission
  mpc_redraw   the gz missions' driving (hand-tuned MPC thrust, heading
               autopilot nozzle, track legs 0 / 45 deg), sea per mission
  mpc_task     the gz missions' driving and sea: the gz protocol

  screen  which laws the gz driving takes to gz-like speed: a 30 s mission
          (mission 1's streams, task sea) with drive 'mpc'; kept when it does
          not diverge and the mean speed over its last 5 s is in [SPEED_LO,
          SPEED_HI] kn (gz missions: 21-24 kn). Selection by the plant's own
          behaviour under the driver, never by any model output; the first
          --boats kept laws in id order.
  data    episode_proto episodes (4 missions x 90 s per boat and condition)
          into cache meta8_proto, split = condition; pack and e0 through
          meta_step8's own phases.
  fit     studies/tail_check --phase fit --groups all per condition.

    python -m studies.protocol_check --phase screen,data,pack,e0
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
import numpy as np   # noqa: E402

from learn.meta import episode8 as E8   # noqa: E402
from studies import meta_step2 as S2   # noqa: E402
from studies import meta_step8 as M8   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache", "meta8_proto")
CONDS = ("auto_redraw", "auto_task", "mpc_redraw", "mpc_task")
LAW0, N_SCREEN = 100000, 100            # meta8 held8 laws
SPEED_LO, SPEED_HI = 18.0, 28.0
KN = 0.514444


def log(msg):
    M8.log(msg)


def job_of(law, mi, cond, lib_path, T=None):
    drive, sea_mode = cond.split("_")
    j = dict(seed=E8.episode_id(law, mi), law=law, mi=mi, world="low",
             relay=False, use_lib=True, lib_path=lib_path,
             act_family=M8.ACT, family=M8.FAMILY, drive=drive,
             sea_mode=sea_mode)
    if T is not None:
        j["T"] = T
    return j


def proto_run_job(job):
    from studies import episode_proto as EP
    return EP.run_job(job)


def phase_screen(args):
    lib_path = os.path.join(CACHE, "lib.npz")
    S2._run = proto_run_job
    laws = list(range(LAW0, LAW0 + N_SCREEN))
    jobs = [job_of(b, 1, "mpc_task", lib_path, T=30.0) for b in laws]
    eps = S2.run_jobs(jobs, lib_path, label="screen")
    rows = {}
    for e in eps:
        if "error" in e:
            rows[int(e["seed"]) % E8.MISSION_STRIDE] = dict(error=e["error"])
            continue
        v = e["XS"][:, 6] / KN
        rows[int(e["law"])] = dict(div=int(e["div"]), v_end=float(
            v[-20:].mean()) if len(v) else float("nan"),
            thr=float(e["U"][:, 0].mean()) if len(v) else float("nan"))
    keep = [b for b in sorted(rows) if "error" not in rows[b]
            and rows[b]["div"] < 0
            and SPEED_LO <= rows[b]["v_end"] <= SPEED_HI]
    n_err = sum("error" in r for r in rows.values())
    n_div = sum(r.get("div", -1) >= 0 for r in rows.values())
    v = np.array([r["v_end"] for r in rows.values() if "v_end" in r])
    log(f"screen: {len(rows)} laws, {n_err} errors, {n_div} diverged; end "
        f"speed under the gz driving quantiles 10/25/50/75/90 % "
        f"{np.round(np.nanquantile(v, [.1, .25, .5, .75, .9]), 1)} kn; "
        f"{len(keep)} in [{SPEED_LO}, {SPEED_HI}] kn")
    M8._atomic(os.path.join(CACHE, "screen.json"), lambda fh: json.dump(
        dict(rows={str(k): v for k, v in rows.items()}, keep=keep,
             speed=[SPEED_LO, SPEED_HI]), fh))


def boats(args):
    keep = json.load(open(os.path.join(CACHE, "screen.json")))["keep"]
    return keep[:args.boats]


def phase_data(args):
    lib_path = os.path.join(CACHE, "lib.npz")
    S2._run = proto_run_job
    raw = os.path.join(CACHE, "raw")
    os.makedirs(raw, exist_ok=True)
    code = M8.code8()
    bs = boats(args)
    for cond in args.conds:
        f = os.path.join(raw, f"{cond}_000.pkl")
        want = M8._chunk_want(bs, args.missions)
        if os.path.exists(f):
            old = pickle.load(open(f, "rb"))
            if old.get("code") == code and old["seeds"] == want and \
                    not old.get("n_errors", 0):
                continue
        jobs = [job_of(b, mi, cond, lib_path) for b in bs
                for mi in range(args.missions)]
        t0 = time.time()
        eps = S2.run_jobs(jobs, lib_path, label=cond)
        bad = [e for e in eps if "error" in e]
        eps = sorted([e for e in eps if "error" not in e],
                     key=lambda e: (e["law"], e["mi"]))
        M8._atomic(f, lambda fh: pickle.dump(
            dict(seeds=want, code=code, eps=eps, n_errors=len(bad),
                 cond=cond), fh), "wb")
        n_div = sum(e["div"] >= 0 for e in eps)
        sp = [float(np.mean(e["XS"][:, 6]) / KN) for e in eps
              if len(e["XS"])]
        log(f"data {cond}: {len(eps)} episodes, {len(bad)} errors, {n_div} "
            f"truncated by a divergence, mean speed quantiles 10/50/90 % "
            f"{np.round(np.quantile(sp, [.1, .5, .9]), 1)} kn, "
            f"{time.time() - t0:.0f} s"
            + (f"; first error {bad[0]['error'][:300]}" if bad else ""))


def setup_m8(args):
    M8.CACHE = S2.CACHE = CACHE
    S2.PROCS = args.procs
    S2.log = log
    bs = boats(args)
    for c in CONDS:
        M8.SPLITS8[c] = dict(seed0=LAW0, src="A")
    M8._boats = lambda split, a: bs
    args.split_list = list(args.conds)
    args.boats_n = len(bs)
    args.force = False
    args.allow_errors = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="screen,data,pack,e0")
    ap.add_argument("--boats", type=int, default=16)
    ap.add_argument("--missions", type=int, default=4)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--conds", default=",".join(CONDS))
    args = ap.parse_args()
    args.conds = [c for c in args.conds.split(",") if c]
    for c in args.conds:
        if c not in CONDS:
            raise SystemExit(f"condition {c}: one of {CONDS}")
    os.makedirs(CACHE, exist_ok=True)
    M8.CACHE = S2.CACHE = CACHE
    S2.PROCS = args.procs
    S2.log = log
    M8.phase_copy(args)
    for ph in [p for p in args.phase.split(",") if p]:
        log(f"=== protocol phase {ph}")
        t0 = time.time()
        if ph == "screen":
            phase_screen(args)
        elif ph == "data":
            phase_data(args)
        else:
            setup_m8(args)
            M8.PHASES[ph](args)
        log(f"=== protocol phase {ph} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
