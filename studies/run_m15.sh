#!/bin/sh
# DEFECTS M16 (code and cache named m15 / meta5): the training world where
# future waves matter -- the operator reads the waves met during the step
# (W_MID), its wave projections mix spatial patterns, and about half of its
# wave-reading filters respond to the step's own waves -- then the three
# networks (a: states / commands / errors only; w: + the elevations at t;
# p: + the preview) and their evaluation. 10000 x 90 s training episodes.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
C=studies/_cache/meta5
mkdir -p $C
L=$C/run_m15_stdout.log
ok=0
for try in 1 2 3 4 5; do
  echo "=== data try $try" >> $L
  if $PY -m studies.meta_step5 --cache-name meta5 \
      --phase copy,data,pack,wmid,e0,branches --size train=10000 \
      --procs 8 >> $L 2>&1; then
    ok=1; break
  fi
done
[ $ok = 1 ] || { echo "=== data FAILED" >> $L; exit 1; }
set -e
echo "=== train" >> $L
$PY -m studies.meta_step5 --cache-name meta5 --phase train_a,train_w,train_p \
    --steps 40000 >> $L 2>&1
echo "=== eval" >> $L
$PY studies/eval_preview_m15.py --cache-name meta5 >> $L 2>&1
echo "=== M15 done" >> $L
