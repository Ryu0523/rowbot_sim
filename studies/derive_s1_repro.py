#!/usr/bin/env python3
"""
S1 reproduction proof: row-by-row comparison of the quick verification tiers run BEFORE the edit of
hydro/rankine/doublebody.py (studies/_cache/derive/s1_baseline/before/*.json) and AFTER it (the current
studies/_cache/rankine/*_quick.json). Every row value must be bit-identical (the edit only adds opt-in functions and an
`if kinematic == "flux"` branch; the default path is unchanged). Also diffs the source files against the pre-edit
copy (s1_baseline/rankine_orig): only doublebody.py may differ.

Output: studies/_cache/derive/s1_repro.txt
"""
import difflib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "_cache", "derive")
BEFORE = os.path.join(OUT, "s1_baseline", "before")
AFTER = os.path.join(HERE, "_cache", "rankine")
LINES = []


def log(s=""):
    print(s)
    LINES.append(s)


def main():
    pairs = [("db_verify_quick", os.path.join(OUT, "s1_baseline", "after"), "after the edit (intermediate file state)"),
             ("db_verify_quick", os.path.join(OUT, "s1_baseline", "final"), "final file"),
             ("unsteady_verify_quick", AFTER, "after the edit (B5; module not imported by this tier)")]
    for name, adir, lab in pairs:
        pb = os.path.join(adir, name + ".json")
        if not os.path.exists(pb):
            log(f"{name} [{lab}]: no record at {pb}")
            continue
        a = json.load(open(os.path.join(BEFORE, name + ".json"), encoding="utf-8"))
        b = json.load(open(pb, encoding="utf-8"))
        if a["start_utc"] == b["start_utc"]:
            log(f"{name} [{lab}]: record identical to the baseline run (same start time) -- not a separate run")
            continue
        ra, rb = a["rows"], b["rows"]
        log(f"{name} [{lab}]: before {a['start_utc']} ({a['wall_s']:.0f} s), after {b['start_utc']} ({b['wall_s']:.0f} s); rows {len(ra)} / {len(rb)}")
        if len(ra) != len(rb):
            log("  ROW COUNT DIFFERS")
        n_id = n_diff = 0
        worst = []
        for x, y in zip(ra, rb):
            same_val = (x["value"] == y["value"]) or (isinstance(x["value"], float) and isinstance(y["value"], float)
                                                        and x["value"] != x["value"] and y["value"] != y["value"])
            if x["quantity"] != y["quantity"]:
                log(f"  quantity mismatch: {x['quantity'][:60]} | {y['quantity'][:60]}")
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
    # source diff
    orig = os.path.join(OUT, "s1_baseline", "rankine_orig")
    cur = os.path.join(ROOT, "hydro", "rankine")
    log("\nsource files hydro/rankine/*.py vs the pre-edit copy:")
    for fn in sorted(os.listdir(orig)):
        if not fn.endswith(".py"):
            continue
        A = open(os.path.join(orig, fn), encoding="utf-8").read().splitlines()
        B = open(os.path.join(cur, fn), encoding="utf-8").read().splitlines()
        d = list(difflib.unified_diff(A, B, lineterm="", n=0))
        added = sum(1 for l in d if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in d if l.startswith("-") and not l.startswith("---"))
        log(f"  {fn:22s} {'identical' if not d else f'+{added} -{removed} lines'}")
    with open(os.path.join(OUT, "s1_repro.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
