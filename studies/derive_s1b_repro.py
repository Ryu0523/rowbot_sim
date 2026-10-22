#!/usr/bin/env python3
"""
S1 stage 2 (S1b) reproduction proof: the quick verification tiers B6 (rankine_db_verify) and B5 (rankine_unsteady_verify)
run with the stage-2 files (studies/_cache/derive/s1b_baseline/after/*.json, copies of the fresh studies/_cache/rankine
records) against the pre-stage-1 baseline (s1_baseline/before/*.json): every row value must be bit-identical and the
pass/fail flags equal (the wall-time row excepted). Source diff of hydro/rankine/*.py against the pre-stage-1 copy
(s1_baseline/rankine_orig): only doublebody.py (stage 1 + stage 2) and unsteady.py (stage 2) may differ.

Output: studies/_cache/derive/s1b_repro.txt
"""
import difflib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "_cache", "derive")
BEFORE = os.path.join(OUT, "s1_baseline", "before")
AFTER = os.path.join(OUT, "s1b_baseline", "after")
LINES = []


def log(s=""):
    print(s)
    LINES.append(s)


def main():
    for name in ("db_verify_quick", "unsteady_verify_quick"):
        pb = os.path.join(AFTER, name + ".json")
        if not os.path.exists(pb):
            log(f"{name}: no stage-2 record at {pb}")
            continue
        a = json.load(open(os.path.join(BEFORE, name + ".json"), encoding="utf-8"))
        b = json.load(open(pb, encoding="utf-8"))
        ra, rb = a["rows"], b["rows"]
        log(f"{name}: before (pre-stage-1 files) {a['start_utc']} ({a['wall_s']:.0f} s), stage-2 files {b['start_utc']} ({b['wall_s']:.0f} s); rows {len(ra)} / {len(rb)}")
        if len(ra) != len(rb):
            log("  ROW COUNT DIFFERS")
        n_id = n_diff = 0
        worst = []
        for x, y in zip(ra, rb):
            same_val = (x["value"] == y["value"]) or (isinstance(x["value"], float) and isinstance(y["value"], float)
                                                        and x["value"] != x["value"] and y["value"] != y["value"])
            if x["quantity"] != y["quantity"]:
                log(f"  quantity text differs: {x['quantity'][:70]} | {y['quantity'][:70]}")
            if same_val and x["ok"] == y["ok"]:
                n_id += 1
            else:
                n_diff += 1
                rel = abs(x["value"] - y["value"]) / max(abs(x["value"]), 1e-300) if isinstance(x["value"], (int, float)) and isinstance(y["value"], (int, float)) else float("nan")
                worst.append((rel, x["check"], x["quantity"][:70], x["value"], y["value"]))
        log(f"  identical rows (value bit-identical and same pass/fail): {n_id}; differing: {n_diff}")
        for rel, chk, qn, va, vb in sorted(worst, reverse=True)[:30]:
            log(f"    {chk:5s} {qn:70s} before {va!r} after {vb!r} (rel {rel:.1e})")
        sa, sb = a.get("summary", {}), b.get("summary", {})
        log(f"  summary before: " + json.dumps({k: v for k, v in sa.items() if 'fail' not in k}))
        log(f"  summary after:  " + json.dumps({k: v for k, v in sb.items() if 'fail' not in k}))
    orig = os.path.join(OUT, "s1_baseline", "rankine_orig")
    cur = os.path.join(ROOT, "hydro", "rankine")
    log("\nsource files hydro/rankine/*.py vs the pre-stage-1 copy (s1_baseline/rankine_orig):")
    for fn in sorted(os.listdir(orig)):
        if not fn.endswith(".py"):
            continue
        A = open(os.path.join(orig, fn), encoding="utf-8").read().splitlines()
        B = open(os.path.join(cur, fn), encoding="utf-8").read().splitlines()
        d = list(difflib.unified_diff(A, B, lineterm="", n=0))
        added = sum(1 for l in d if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in d if l.startswith("-") and not l.startswith("---"))
        log(f"  {fn:22s} {'identical' if not d else f'+{added} -{removed} lines'}")
    # stage-1 -> stage-2 diff of doublebody.py (the stage-1 final file is recorded by its sha256 prefix 4037b9059b090bb3)
    with open(os.path.join(OUT, "s1b_repro.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
