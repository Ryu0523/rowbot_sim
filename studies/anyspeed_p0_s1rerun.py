#!/usr/bin/env python3
"""
The standing checks of step S1 (studies/anyspeed_s1_check.py) re-run on a
candidate as1 preset, before the preset is changed: orchestrator decision 2
on S1 asks for the strip-exact chine location on the prismatic strips too
(switch chine_exact_all), and a preset is updated only when its switch
passes its verification -- which, for a change of S1's own flux, includes
S1's acceptance items (D55 at 1-7 kn within 5 % of theory, small-disturbance
decay at 1-10 kn, one stable equilibrium at every speed 1-40 kn, up / down
sweeps without jumps) and its robustness parts (f: basin and following
seas at Hs x1.5; w: bow out of the water in head waves; d: the planing
range 14-40 kn at 48 / 96 / 192 stations and Fridsma P6; h: 26-40 kn
station count and time step, the 26-27 kn root).

The candidate preset is put into sim/planing_anyspeed.PRESETS in this
process only (the S1 script builds 'as1' plants through it), and S1's
output directory is redirected, so the records of S1 itself
(studies/_cache/anyspeed/s1/) are not overwritten. One part per process:

    .venv/Scripts/python.exe -m studies.anyspeed_p0_s1rerun --part a \\
        --switches dDt_full,chine_exact_all

or several parts at once, each in its own process (studies/parallel_runner,
pool sized by free RAM with 3 GB kept for the other tracks), stdout of each
in <out>/part_<x>_stdout.log:

    .venv/Scripts/python.exe -m studies.anyspeed_p0_s1rerun --parts f,w,d,h
"""
import argparse
import os
import subprocess
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "_cache", "anyspeed", "p0", "s1_rerun")


def run_part_subprocess(arg):
    """One part in its own interpreter (a job of studies/parallel_runner):
    (part, exit code, seconds)."""
    part, switches, out = arg
    t0 = time.time()
    with open(os.path.join(out, f"part_{part}_stdout.log"), "w") as log:
        rc = subprocess.run(
            [sys.executable, "-m", "studies.anyspeed_p0_s1rerun", "--part",
             part, "--switches", switches, "--out", out], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT).returncode
    return part, rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=list("abcgfwdh"))
    ap.add_argument("--parts", default="",
                    help="comma-separated parts, run in parallel processes")
    ap.add_argument("--switches", default="dDt_full,chine_exact_all")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.parts:
        from studies.parallel_runner import run_parallel
        parts = [p for p in a.parts.split(",") if p]
        bad = sorted(set(parts) - set("abcgfwdh"))
        if bad:
            raise SystemExit(f"unknown parts {bad}")
        t0 = time.time()
        res = run_parallel(run_part_subprocess,
                           [(p, a.switches, a.out) for p in parts],
                           gb_per_job=0.4, reserve_gb=3.0)
        for r in res:
            if isinstance(r, Exception):
                print(f"  job failed: {r!r}", flush=True)
            else:
                print(f"  part {r[0]}: exit {r[1]}, {r[2]:.0f} s", flush=True)
        print(f"all parts done in {time.time() - t0:.0f} s wall", flush=True)
        return
    if not a.part:
        raise SystemExit("--part or --parts")
    warnings.filterwarnings("ignore")
    from sim import planing_anyspeed as PA
    sw = tuple(s for s in a.switches.split(",") if s)
    PA.PRESETS["as1"] = PA.resolve("as1", sw)
    from studies import anyspeed_s1_check as S1
    S1.OUT = a.out
    t0 = time.time()
    print(f"S1 part {a.part} on the candidate as1 preset {PA.PRESETS['as1']}"
          f" -> {a.out}", flush=True)
    getattr(S1, f"part_{a.part}")()
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
