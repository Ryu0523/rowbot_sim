#!/bin/sh
# DEFECTS M13: can the operator family express the target's error function?
# The four fits run in parallel, 3 threads each, each with its own results
# pickle and stdout log (the shared family_fit_m13.log interleaves).
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
D=studies/_cache/meta3
run() {
  tag=$1; shift
  $PY studies/family_fit_m13.py "$@" --threads 3 --out family_fit_m13_$tag.pkl \
    > $D/run_m13_fit_$tag.log 2>&1
  echo "=== $tag done ($?)" >> $D/run_m13_fit_stdout.log
}
run family --variant family &
run draw --variant draw &
run control --control 5 &
run control_lownoise --control 5 --floor_target 0.1 &
wait
echo "=== M13 fit done" >> $D/run_m13_fit_stdout.log
