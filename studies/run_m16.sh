#!/bin/sh
# DEFECTS M18 / PRIOR_DERIVATION D9: the general "rigid body under arbitrary
# forces" family (learn/meta/operators_gen.py) as the training world, cache
# meta6. Waits until the meta5 data phases have finished (CPU and RAM), then
# data (10000 x 90 s train episodes, 6 workers), the three networks (a: no
# waves; w: + elevations at t; p: + preview) and the M15 evaluation.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
C=studies/_cache/meta6
mkdir -p $C
L=$C/run_m16_stdout.log
echo "=== waiting for meta5 data ($(date +%H:%M))" >> $L
until grep -q "=== train" studies/_cache/meta5/run_m15_stdout.log \
    || grep -q "data FAILED" studies/_cache/meta5/run_m15_stdout.log; do
  sleep 60
done
ok=0
for try in 1 2 3 4 5; do
  echo "=== data try $try ($(date +%H:%M))" >> $L
  if $PY -m studies.meta_step6 --family gen \
      --phase copy,data,pack,wmid,e0,branches --size train=10000 \
      --procs 6 >> $L 2>&1; then
    ok=1; break
  fi
done
[ $ok = 1 ] || { echo "=== data FAILED" >> $L; exit 1; }
set -e
echo "=== train ($(date +%H:%M))" >> $L
$PY -m studies.meta_step6 --family gen --phase train_a,train_w,train_p \
    --steps 40000 >> $L 2>&1
echo "=== eval ($(date +%H:%M))" >> $L
$PY studies/eval_preview_m15.py --cache-name meta6 >> $L 2>&1
echo "=== M16 gen done ($(date +%H:%M))" >> $L
