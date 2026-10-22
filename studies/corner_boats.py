#!/usr/bin/env python3
"""
In-family 'corner' boats: catalogue laws (the D10 family the pretrained net
was trained on) with the six items that have gz counterparts all ON and all
STRONG (gain in the upper third of the item's own on-distribution among the
meta7 training laws), every other part of the law drawn as in the prior.
Second part of the density check of learn/meta/ROADMAP_2026-09-30.md
section 11.11: the training prior holds no such law (theory_checks
--check density), so these boats test whether a region of low prior mass
inside the family is enough for the frozen net to leave a large part of
the error that a copy trained on the boat's own missions removes (as on
gz), while on ordinary held-out boats nothing is left (tail_check.log).

The laws are chosen by their own parameters only (no outcome of any
model):

  search  law ids from LAW0 on; the six items' parameters come from the
          exact streams CatDraw uses ([op_seed, CAT_STREAM, attempt 0,
          SID_ITEM + index], op_seed = 7 law + 1 as episode8), drawn
          against a fresh context; the first --n-boats laws in the corner
          are kept, then re-checked with the full draw (OperatorCat).
  data    episode8 multi-mission episodes (4 missions x 90 s per boat) in
          cache meta8_corner, the six items forced on (force_items, their
          parameters still the law's own draw), through meta_step8's own
          data / pack / e0 phases.
  check   from the packed generation records: each boat's six gains and
          the acceptance test's redraw count (a redraw replaces the
          parameters, so such a boat may have left the corner); writes
          corner8_laws.json, the boats still in the corner.

Then: python -m studies.tail_check --phase fit --cache
studies/_cache/meta8_corner --split corner8 --groups all --laws
studies/_cache/meta8_corner/corner8_laws.json

    python -m studies.corner_boats --phase search,data,pack,e0,check
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
from studies.theory_checks import C7, GAIN, GZ_LIKE   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache", "meta8_corner")
SPLIT = "corner8"
LAW0 = 900_000                   # no other split's laws; < MISSION_STRIDE
N_SEARCH = 99_000
Q = 2 / 3                        # 'strong': the upper third
_RUN_JOB = E8.run_job            # the original, before main() patches it


def log(msg):
    M8.log(msg)


def thresholds():
    """{item: gain at quantile Q of its on-distribution} among the meta7
    training laws (theory_checks.check_density's definition; W7 counts as
    on only with its lateral sub-part)."""
    rows = pickle.load(open(os.path.join(C7, "train_meta.pkl"), "rb"))
    out = {}
    for c in GZ_LIKE:
        g = [float(r["style"]["item_params"][c][GAIN[c]]) for r in rows
             if c in r["style"]["items"] and (c != "W7" or bool(
                 r["style"]["item_params"][c].get("lat", 1)))]
        out[c] = float(np.quantile(g, Q))
    return out


def in_corner(prm, thr):
    return all(prm[c][GAIN[c]] >= thr[c] for c in GZ_LIKE) and bool(
        prm["W7"].get("lat", 1))


def fresh_ctx():
    """The context CatDraw builds before any item adjusts it."""
    from learn.meta import cat_base as CB
    from learn.meta import operators_cat as OC
    from learn.meta import operators_rb as R
    g = OC.cat_geometry()
    return CB.CatCtx(R.default_params(), keel=g.get("keel"),
                     intake=g.get("intake"), v_slam=g.get("v_slam"))


def phase_search(args):
    from learn.meta import cat_base as CB
    from learn.meta import operators_cat as OC
    CB.load_items()
    thr = thresholds()
    log(f"search: corner thresholds (quantile {Q:.3f} of the on-"
        f"distribution in meta7 train) {thr}")
    ctx = fresh_ctx()
    found, n = [], 0
    for law in range(LAW0, LAW0 + N_SEARCH):
        n += 1
        s = 7 * law + 1
        prm = {c: CB.REGISTRY[c].draw_on(np.random.default_rng(
            [s, CB.CAT_STREAM, 0, CB.SID_ITEM + CB.CODES.index(c)]), ctx)
            for c in GZ_LIKE}
        if in_corner(prm, thr):
            found.append(law)
            if len(found) >= args.n_boats:
                break
    log(f"search: {len(found)} corner laws in {n} draws (rate "
        f"{len(found) / n:.5f}; all six in the upper third independently "
        f"{(1 - Q) ** 6:.5f} before the W7 lateral condition)")
    ok = []
    for law in found:                         # the full draw, as episode8
        op = OC.OperatorCat(7 * law + 1, None, dt=0.24, L=5.4, accept=False,
                            force_items={c: True for c in GZ_LIKE})
        if in_corner(op.on, thr):
            ok.append(law)
    log(f"search: {len(ok)} of {len(found)} confirmed by the full draw")
    os.makedirs(CACHE, exist_ok=True)
    M8._atomic(os.path.join(CACHE, "corner_search.json"),
               lambda fh: json.dump(dict(laws=ok, thresholds=thr,
                                         n_draws=n, q=Q), fh))


def corner_run_job(job):
    """episode8.run_job with the six items forced on (their parameters the
    law's own draw); runs in the worker processes."""
    if not getattr(E8, "_corner", False):
        def make_op(family, op_seed, lib, dtc, L, relay=False, act=None):
            from learn.meta.operators_cat import OperatorCat
            return OperatorCat(op_seed, lib, dt=dtc, L=L, relay=relay,
                               act=act,
                               force_items={c: True for c in GZ_LIKE})
        E8.make_op = make_op
        E8._corner = True
    return _RUN_JOB(job)


def laws_searched():
    return json.load(open(os.path.join(CACHE, "corner_search.json")))[
        "laws"]


def setup_m8(args):
    """meta_step8's phases on the corner split: its cache, boat list, no
    reuse of meta7 mission-0 episodes, the patched job."""
    M8.CACHE = S2.CACHE = CACHE
    S2.PROCS = args.procs
    S2.log = log
    M8.SPLITS8[SPLIT] = dict(seed0=LAW0, src="A")
    M8._boats = lambda split, a: laws_searched()
    M8._reuse_ok = lambda a, lib_path: False
    E8.run_job = corner_run_job
    args.split_list = [SPLIT]
    args.boats = args.held_boats = len(laws_searched())
    args.force = False
    args.allow_errors = False


def phase_check(args):
    thr = json.load(open(os.path.join(CACHE, "corner_search.json")))[
        "thresholds"]
    meta = pickle.load(open(os.path.join(CACHE, f"{SPLIT}_meta.pkl"), "rb"))
    by_law = {}
    for m_ in meta:
        by_law.setdefault(int(m_["law"]), []).append(m_)
    keep, out = [], []
    for law, ms in sorted(by_law.items()):
        st = ms[0]["style"]
        prm = st["item_params"]
        ok = all(c in st["items"] for c in GZ_LIKE) and in_corner(prm, thr)
        nrej = int(st.get("n_reject", 0))
        out.append(f"{law}: redraws {nrej}, in corner {ok}")
        if ok:
            keep.append(law)
    log("check: " + "; ".join(out))
    log(f"check: {len(keep)} of {len(by_law)} boats still in the corner")
    M8._atomic(os.path.join(CACHE, f"{SPLIT}_laws.json"),
               lambda fh: json.dump(dict(laws=keep), fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="search,data,pack,e0,check")
    ap.add_argument("--n-boats", type=int, default=24)
    ap.add_argument("--missions", type=int, default=4)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    M8.CACHE = CACHE
    for ph in [p for p in args.phase.split(",") if p]:
        log(f"=== corner phase {ph}")
        t0 = time.time()
        if ph == "search":
            phase_search(args)
        elif ph == "check":
            phase_check(args)
        else:
            if ph == "data":
                M8.CACHE = CACHE
                M8.phase_copy(args)
            setup_m8(args)
            M8.PHASES[ph](args)
        log(f"=== corner phase {ph} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
