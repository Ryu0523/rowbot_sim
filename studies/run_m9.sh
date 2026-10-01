#!/bin/sh
# DEFECTS M9 main run, one process at a time: training branches, the two
# fine-tunes, the three evaluations on one code path, the Cb step diagnostic.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
L=studies/_cache/meta2/run_m9_stdout.log
$PY -m studies.meta_step3 --phase tbranches --procs 4 >> $L 2>&1
$PY -m studies.meta_step3 --phase train_cov >> $L 2>&1
$PY -m studies.meta_step3 --phase train_es >> $L 2>&1
$PY -m studies.meta_step3 --phase eval --tag "" --out-tag _kv >> $L 2>&1
$PY -m studies.meta_step3 --phase eval --tag _cov >> $L 2>&1
$PY -m studies.meta_step3 --phase eval --tag _es >> $L 2>&1
for m in model3.pt model3_cov.pt model3_es.pt; do
  echo "=== diag $m" >> $L
  $PY studies/diag_yaw_m8_blocks.py --model $m >> $L 2>&1
done
echo "=== M9 run done" >> $L
