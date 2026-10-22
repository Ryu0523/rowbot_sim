#!/usr/bin/env python3
"""
S5a: the goldens that studies/anyspeed_regress_as.GOLDEN_SETS pins for
step S5a (learn/meta/ANYSPEED_PLAN_2026-10-05.md, '### S5a' and
'### S5a：复查后的修正').

Since the S5a review this script only runs studies/anyspeed_s5a_check.py
part goldens (check H12 of the review round), which

  - runs each set's golden runs in two processes and compares them bit for
    bit, and runs its base set with the same options (the golden must
    differ from it on at least one run of each sea where the switch acts);
  - holds the calm runs' nozzle at 5 deg and the task runs' at zero for the
    S5a sets and lateral_phi0 (the first round's nozzle-at-zero calm runs
    stayed upright and pinned nothing the switches change), the moving
    nozzle for fk_exact + orbital;
  - writes each golden once to studies/_cache/anyspeed/ (never overwrites)
    and a stamped record (switches, code and script sha1s, each file's
    sha1, heels, the differing runs) to
    studies/_cache/anyspeed/s5a/r2/goldens_s5a.txt and .json -- the first
    round's version of this script printed to stdout only.

    .venv/Scripts/python.exe -m studies.anyspeed_s5a_goldens
"""
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main():
    from studies import anyspeed_s5a_check as C
    warnings.filterwarnings("ignore")
    C.STAMP = C.stamp_all()
    C.part_goldens()
    n_bad = sum(not c["ok"] for c in C.CHECKS)
    print(f"{len(C.CHECKS)} checks, {n_bad} failed")


if __name__ == "__main__":
    main()
