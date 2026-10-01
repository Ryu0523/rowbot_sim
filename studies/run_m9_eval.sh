#!/bin/sh
# DEFECTS M9: evaluations of model3.pt and V0 on one code path, and the Cb
# step diagnostic (V1 kept step 0, i.e. model3.pt, so it is not re-run).
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
L=studies/_cache/meta2/run_m9_stdout.log
$PY -m studies.meta_step3 --phase eval --tag "" --out-tag _kv >> $L 2>&1
$PY -m studies.meta_step3 --phase eval --tag _cov >> $L 2>&1
for m in model3.pt model3_cov.pt; do
  echo "=== diag $m" >> $L
  $PY studies/diag_yaw_m8_blocks.py --model $m >> $L 2>&1
done
echo "=== M9 eval done" >> $L
