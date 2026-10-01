#!/bin/sh
# DEFECTS M10 main run (the random actuator family), one heavy process at a
# time: new source data in studies/_cache/meta3 (target splits copied from
# meta2), the one-step model from scratch, the wide-plan fine-tune (V0), the
# oracle fine-tune on target data (Cbh), evaluations and diagnostics.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
mkdir -p studies/_cache/meta3
L=studies/_cache/meta3/run_m10_stdout.log
S2="$PY -m studies.meta_step2 --cache-name meta3 --act-family m10 --copy-from meta2"
S3="$PY -m studies.meta_step3 --cache-name meta3"
# step 2: finished chunks are cached, so a killed run resumes where it was
ok=0
for try in 1 2 3 4 5; do
  echo "=== step 2 try $try" >> $L
  if $S2 --phase copy,pilot,data,pack --splits train,A,B,At --procs 8 >> $L 2>&1; then
    ok=1; break
  fi
done
[ $ok = 1 ] || { echo "=== step 2 FAILED" >> $L; exit 1; }
set -e
$S3 --phase e0,branches >> $L 2>&1
$S3 --phase tbranches --procs 8 >> $L 2>&1
$S3 --phase train >> $L 2>&1
$S3 --phase train_cov >> $L 2>&1
$S3 --phase eval --tag "" >> $L 2>&1
$S3 --phase eval --tag _cov >> $L 2>&1
set +e
for m in model3.pt model3_cov.pt; do
  echo "=== diag infamily A / At $m" >> $L
  $PY studies/diag_m9_infamily.py --cache-name meta3 --split A --model $m >> $L 2>&1
  $PY studies/diag_m9_infamily.py --cache-name meta3 --split At --model $m >> $L 2>&1
  echo "=== diag yaw Cb / At $m" >> $L
  $PY studies/diag_yaw_m8_blocks.py --cache-name meta3 --model $m >> $L 2>&1
  $PY studies/diag_yaw_m8_blocks.py --cache-name meta3 --split At --model $m >> $L 2>&1
done
echo "=== diag context model3_cov.pt" >> $L
$PY studies/diag_context_m11.py --cache-name meta3 --model model3_cov.pt >> $L 2>&1
# the oracle-data test: is steady-command yaw learnable from these inputs?
$S3 --phase cbhold --procs 8 >> $L 2>&1 && \
$S3 --phase train_oracle >> $L 2>&1 && \
$S3 --phase eval --tag _oracle >> $L 2>&1
for m in model3_cov.pt model3_oracle.pt; do
  echo "=== diag yaw Cbh holdout $m" >> $L
  $PY studies/diag_yaw_m8_blocks.py --cache-name meta3 --split Cbh --holdout --model $m >> $L 2>&1
done
echo "=== M10 run done" >> $L
