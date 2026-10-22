#!/usr/bin/env python3
"""
Multi-mission pretraining data (cache meta8; ROADMAP_2026-09-30 section
11.3): several missions of the same boat under one catalogue error law,
for the memory structures (memory tokens, context compression, test-time
training) that must learn to carry what a boat has shown across missions.
learn/meta/episode8.py makes the episodes (boat = error law + actuators,
mission = sea, scenario, noise); everything else is the catalogue family
of meta7 (learn/meta/episode7.py, studies/meta_step7.py).

Splits (boats x missions; mission 0 of boat b is meta7's episode of seed b):
  train8  boats 1 .. --boats, the meta7 train seeds
  held8   boats 100000 .. 100000 + --held-boats - 1, the meta7 A seeds
          (in-family held-out boats for the evaluation)

Phases (each resumable; files written atomically):
  copy  lib.npz from meta7
  data  chunks of CHUNK_BOATS boats x --missions missions. Mission 0 is
        taken from meta7's raw chunks when they were made by the current
        episode code (meta_step7.split_code) AND episode8's mission 0 was
        found bit-identical to them on N_IDENT boats per split (checked
        once per code version, identity_<code8>_<code7>.json), else
        generated; the chunk is stamped with code8 (meta_step2.SOURCES +
        meta_step7.EP_SOURCES + episode8.py); a chunk with failed episodes
        reruns them next time, and good episodes are kept when --boats or
        --missions change
  pack  episode7.pack7 over the split (refused while chunks hold failed
        episodes, unless --allow-errors), rows ordered by (boat, mission);
        meta gains law, mi; <split>_boats.json maps each boat to {mission:
        row} and lists the (boat, mission) pairs dropped as too short
  e0    the error against the MPC's model (data3.e0_split), stamped with
        meta_step7.sim7 and the pack's size and time

Memory: the data workers import no torch; at most 8 (--procs).

    python -m studies.meta_step8 --phase copy,data,pack,e0 --boats 1000 \\
        --held-boats 100 --missions 4 --procs 4
"""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from studies import meta_step2 as S2   # noqa: E402
from studies import meta_step7 as S7   # noqa: E402

CACHE = os.path.join(HERE, "_cache", "meta8")
SPLITS8 = dict(train8=dict(seed0=1, src="train"),
               held8=dict(seed0=100000, src="A"))
CHUNK_BOATS = 60
ACT, FAMILY = S7.ACT, "cat"
EP8 = ("learn/meta/episode8.py",)
MAX_PROCS = 8
N_IDENT = 2          # boats per split checked for mission-0 identity


def log(msg, f=None):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(CACHE, "run8.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _hash(files):
    h = hashlib.sha1()
    for f in files:
        h.update(open(os.path.join(ROOT, f), "rb").read())
    return h.hexdigest()[:12]


def code8():
    return _hash(tuple(S2.SOURCES) + S7.EP_SOURCES + EP8)


def _atomic(path, write, mode="w"):
    with open(path + ".tmp", mode) as fh:
        write(fh)
    os.replace(path + ".tmp", path)


# ------------------------------------------------------------ copy
def phase_copy(args):
    src = os.path.join(HERE, "_cache", "meta7", "lib.npz")
    dst = os.path.join(CACHE, "lib.npz")
    if not os.path.exists(dst):
        shutil.copyfile(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        log("copy: lib.npz from meta7")


# ------------------------------------------------------------ data
def _meta7_mission0(split, boats):
    """{seed: episode} of meta7's raw chunks for these seeds, only from
    chunks made by the current episode code."""
    if not boats:
        return {}
    cfg = S2.SPLITS[SPLITS8[split]["src"]]
    want = S7.code7() + f"-{ACT}-{FAMILY}"
    raw = os.path.join(HERE, "_cache", "meta7", "raw")
    out = {}
    seeds = set(boats)
    lo_ci = (min(boats) - cfg["seed0"]) // S2.CHUNK
    hi_ci = (max(boats) - cfg["seed0"]) // S2.CHUNK
    for ci in range(lo_ci, hi_ci + 1):
        f = os.path.join(raw, f"{SPLITS8[split]['src']}_{ci:03d}.pkl")
        if not os.path.exists(f):
            continue
        ch = pickle.load(open(f, "rb"))
        if ch.get("code") != want:
            continue
        for e in ch["eps"]:
            if e["seed"] in seeds:
                out[e["seed"]] = e
    return out


def _reuse_ok(args, lib_path):
    """True when episode8's mission 0 reproduces meta7's episodes: checked
    once per (code8, code7) on N_IDENT boats per split by regenerating
    their mission 0 and comparing every array and the divergence step bit
    for bit; the verdict is stamped in identity_<code8>_<code7>.json."""
    from learn.meta import episode8 as E8
    stamp = os.path.join(CACHE, f"identity_{code8()}_{S7.code7()}.json")
    if os.path.exists(stamp):
        return json.load(open(stamp))["ok"]
    jobs, ref = [], {}
    for split in args.split_list:
        boats = _boats(split, args)[:N_IDENT]
        m0 = _meta7_mission0(split, boats)
        for b, e in m0.items():
            ref[b] = e
            jobs.append(dict(seed=E8.episode_id(b, 0), law=b, mi=0,
                             world="low", relay=False, use_lib=True,
                             lib_path=lib_path, act_family=ACT,
                             family=FAMILY))
    ok, why = True, []
    if jobs:
        for e in S2.run_jobs(jobs, lib_path, label="mission-0 identity"):
            r = ref.get(e.get("law", e["seed"]))
            if "error" in e or r is None:
                ok = False
                why.append(f"boat {e['seed']}: {e.get('error', 'no ref')}")
                continue
            for k, v in r.items():
                if isinstance(v, np.ndarray) and not np.array_equal(
                        v, e.get(k)):
                    ok = False
                    why.append(f"boat {e['law']}: {k} differs")
            if r["div"] != e["div"]:
                ok = False
                why.append(f"boat {e['law']}: div {r['div']} != {e['div']}")
    _atomic(stamp, lambda fh: json.dump(dict(ok=ok, n=len(jobs),
                                             why=why), fh))
    log(f"mission-0 identity with meta7 on {len(jobs)} boats: "
        + ("passed" if ok else "FAILED, mission 0 will be generated: "
           + "; ".join(why[:5])))
    return ok


def _boats(split, args):
    n = args.boats if split == "train8" else args.held_boats
    return [SPLITS8[split]["seed0"] + i for i in range(n)]


def _chunk_want(bs, missions):
    from learn.meta import episode8 as E8
    return sorted(E8.episode_id(b, mi) for b in bs for mi in range(missions))


def phase_data(args):
    from learn.meta import episode8 as E8
    lib_path = os.path.join(CACHE, "lib.npz")
    raw = os.path.join(CACHE, "raw")
    os.makedirs(raw, exist_ok=True)
    S2._run = E8.run_job
    code = code8()
    reuse = _reuse_ok(args, lib_path)
    for split in args.split_list:
        boats = _boats(split, args)
        nch = int(np.ceil(len(boats) / CHUNK_BOATS))
        for ci in range(nch):
            bs = boats[ci * CHUNK_BOATS:(ci + 1) * CHUNK_BOATS]
            want = _chunk_want(bs, args.missions)
            f = os.path.join(raw, f"{split}_{ci:03d}.pkl")
            kept = []
            if os.path.exists(f) and not args.force:
                try:
                    old = pickle.load(open(f, "rb"))
                    if old.get("code") == code:
                        if old["seeds"] == want and not old.get(
                                "n_errors", 0):
                            continue
                        # keep every good episode still wanted (other
                        # --missions / --boats, or a rerun of failures)
                        ws = set(want)
                        kept = [e for e in old["eps"] if e["seed"] in ws]
                except Exception:
                    pass
            have = {e["seed"] for e in kept}
            need0 = [b for b in bs if E8.episode_id(b, 0) not in have]
            m0 = _meta7_mission0(split, need0) if reuse else {}
            reused = []
            for bb, e in m0.items():
                e = dict(e)
                e.update(law=bb, mi=0, hid_seed=0, env_redraw={})
                reused.append(e)
            have |= {e["seed"] for e in reused}
            jobs = [dict(seed=E8.episode_id(bb, mi), law=bb, mi=mi,
                         world="low", relay=False, use_lib=True,
                         lib_path=lib_path, act_family=ACT, family=FAMILY)
                    for bb in bs for mi in range(args.missions)
                    if E8.episode_id(bb, mi) not in have]
            t0 = time.time()
            eps = S2.run_jobs(jobs, lib_path,
                              label=f"{split} chunk {ci + 1}/{nch}")
            bad = [e for e in eps if "error" in e]
            eps = kept + reused + [e for e in eps if "error" not in e]
            eps.sort(key=lambda e: (e["law"], e["mi"]))
            _atomic(f, lambda fh: pickle.dump(
                dict(seeds=want, code=code, eps=eps, n_errors=len(bad)),
                fh), "wb")
            n_div = sum(e["div"] >= 0 for e in eps)
            log(f"data {split} chunk {ci + 1}/{nch}: {len(eps)} episodes "
                f"({len(kept)} kept, {len(reused)} mission-0 episodes reused"
                f" from meta7, {len(jobs)} run, {len(bad)} errors, {n_div} "
                f"truncated by a divergence) in {time.time() - t0:.0f} s"
                + (f" ({(time.time() - t0) / max(len(jobs), 1):.1f} s per "
                   f"episode with {S2.PROCS} procs)" if jobs else "")
                + (f"; first error (seed {bad[0]['seed']}) "
                   f"{bad[0]['error']}" if bad else ""))


# ------------------------------------------------------------ pack
def _chunk_files(split, args):
    nch = int(np.ceil(len(_boats(split, args)) / CHUNK_BOATS))
    return [os.path.join(CACHE, "raw", f"{split}_{ci:03d}.pkl")
            for ci in range(nch)]


def _files_sig(files):
    out = []
    for f in files:
        st = os.stat(f)
        out.append([os.path.basename(f), int(st.st_size),
                    int(st.st_mtime_ns)])
    return out


def phase_pack(args):
    from learn.meta import data2
    from learn.meta import episode8 as E8
    code = code8()
    for split in args.split_list:
        out = os.path.join(CACHE, f"{split}.npz")
        mp = out.replace(".npz", "_meta.pkl")
        bp = out.replace(".npz", "_boats.json")
        files = _chunk_files(split, args)
        miss = [f for f in files if not os.path.exists(f)]
        if miss:
            raise SystemExit(f"pack {split}: chunks missing ({len(miss)}); "
                             "run the data phase")
        sig = dict(files=_files_sig(files), code=code,
                   missions=args.missions, boats=len(_boats(split, args)))
        if all(os.path.exists(x) for x in (out, mp, bp)) and not args.force:
            old = pickle.load(open(mp, "rb"))
            if old and old[0].get("chunks8") == sig:
                continue
        boats = _boats(split, args)
        eps, n_err = [], 0
        for ci, f in enumerate(files):
            ch = pickle.load(open(f, "rb"))
            bs = boats[ci * CHUNK_BOATS:(ci + 1) * CHUNK_BOATS]
            if ch.get("code") != code or ch["seeds"] != _chunk_want(
                    bs, args.missions):
                raise SystemExit(f"{f}: other code, boats or missions than "
                                 "asked; rerun the data phase")
            n_err += ch.get("n_errors", 0)
            eps += ch["eps"]
        if n_err and not args.allow_errors:
            raise SystemExit(f"pack {split}: {n_err} failed episodes in the "
                             "chunks; rerun the data phase (or "
                             "--allow-errors)")
        eps.sort(key=lambda e: (e["law"], e["mi"]))
        lm = {e["seed"]: (e["law"], e["mi"]) for e in eps}
        short = sorted(lm[e["seed"]] for e in eps
                       if len(e["E"]) <= data2.H + 10)
        arrs, meta, snaps = E8.pack7(eps)
        del eps
        boat_rows = {}
        for i, m_ in enumerate(meta):
            m_["law"], m_["mi"] = lm[m_["seed"]]
            m_["code8"], m_["chunks8"] = code, sig
            boat_rows.setdefault(str(int(m_["law"])), {})[
                str(int(m_["mi"]))] = i
        np.savez(out + ".tmp.npz", **arrs)
        os.replace(out + ".tmp.npz", out)
        _atomic(mp, lambda fh: pickle.dump(meta, fh), "wb")
        _atomic(bp, lambda fh: json.dump(dict(
            missions=args.missions, boats=boat_rows,
            dropped=[[int(lw), int(mi)] for lw, mi in short],
            note="boats: {law: {mission index: row}}; dropped: (law, "
                 "mission) pairs pack7 left out (<= H + 10 steps, early "
                 "divergence)"), fh))
        for g in (out.replace(".npz", "_e0.npz"),
                  out.replace(".npz", "_e0.npz") + ".code"):
            if os.path.exists(g):
                os.remove(g)
        n_full = sum(len(r) == args.missions for r in boat_rows.values())
        log(f"pack {split}: {len(meta)} missions of {len(boat_rows)} boats "
            f"({n_full} with all {args.missions} missions kept, "
            f"{len(short)} missions dropped as too short, {n_err} failed "
            f"episodes), arrays {arrs['S'].shape}")


# ------------------------------------------------------------ e0
def phase_e0(args):
    """data3.e0_split, stamped as meta7 stamps it (the simulation code
    meta_step7.sim7 plus the pack's size and time) in <out>.code."""
    from learn.meta import data3
    for split in args.split_list:
        src = os.path.join(CACHE, f"{split}.npz")
        out = os.path.join(CACHE, f"{split}_e0.npz")
        if not os.path.exists(src):
            continue
        st = os.stat(src)
        text = f"{S7.sim7()}|{int(st.st_size)}|{int(st.st_mtime_ns)}"
        if os.path.exists(out) and os.path.exists(out + ".code") \
                and not args.force \
                and open(out + ".code").read().strip() == text:
            continue
        t0 = time.time()
        data3.e0_split(src)
        _atomic(out + ".code", lambda fh: fh.write(text))
        log(f"e0 {split}: {time.time() - t0:.0f} s")


PHASES = dict(copy=phase_copy, data=phase_data, pack=phase_pack,
              e0=phase_e0)


def main():
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="copy,data,pack,e0")
    ap.add_argument("--boats", type=int, default=1000)
    ap.add_argument("--held-boats", type=int, default=100)
    ap.add_argument("--missions", type=int, default=4)
    ap.add_argument("--splits", default="held8,train8")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--cache-name", default="meta8")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-errors", action="store_true",
                    help="pack chunks that still hold failed episodes")
    args = ap.parse_args()
    if not 1 <= args.procs <= MAX_PROCS:
        raise SystemExit(f"--procs: 1 .. {MAX_PROCS}")
    if not args.cache_name.startswith("meta8"):
        raise SystemExit("the multi-mission data lives in caches meta8*")
    CACHE = os.path.join(HERE, "_cache", args.cache_name)
    os.makedirs(CACHE, exist_ok=True)
    S2.CACHE = CACHE
    S2.PROCS = args.procs
    S2.log = log
    args.split_list = [s for s in args.splits.split(",") if s]
    for s in args.split_list:
        if s not in SPLITS8:
            raise SystemExit(f"split {s}: one of {sorted(SPLITS8)}")
    log(f"meta_step8: phases {args.phase}, splits {args.split_list}, "
        f"{args.boats} train / {args.held_boats} held-out boats x "
        f"{args.missions} missions, code {code8()}, procs {args.procs}")
    for ph in [p for p in args.phase.split(",") if p]:
        log(f"=== phase {ph}")
        PHASES[ph](args)


if __name__ == "__main__":
    main()
