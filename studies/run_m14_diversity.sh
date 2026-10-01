#!/bin/sh
# DEFECTS M14, the diversity experiment: the same operator family and world
# as meta3 (M10), but 20000 training episodes of 45 s instead of 5000 of 90 s
# (4x the distinct operators, 2x the steps). Test splits A, B, At unchanged
# (90 s, same seeds); target splits copied. Then the one-step model without
# waves, the one with the 15 elevations, their evaluation, the in-family
# wave-use check and the history-length curves.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
C=studies/_cache/meta4d
mkdir -p $C
L=$C/run_m14_stdout.log
ok=0
for try in 1 2 3 4 5; do
  echo "=== step 2 try $try" >> $L
  if $PY -m studies.meta_step2 --cache-name meta4d --act-family m10 \
      --copy-from meta2 --phase copy,pilot,data,pack --splits train,A,B,At \
      --t-train 45 --size train=20000 --procs 8 >> $L 2>&1; then
    ok=1; break
  fi
done
[ $ok = 1 ] || { echo "=== step 2 FAILED" >> $L; exit 1; }
set -e
S3="$PY -m studies.meta_step3 --cache-name meta4d"
$S3 --phase e0,branches >> $L 2>&1
$S3 --phase train --steps 40000 >> $L 2>&1
$S3 --phase eval --tag "" >> $L 2>&1
set +e
echo "=== wave model" >> $L
M11_CACHE=meta4d M11_BASE=eval3.pkl $PY studies/info_waves_m11.py \
    --steps 40000 >> $L 2>&1
echo "=== wave use in family" >> $L
M11_CACHE=meta4d $PY studies/check_wave_use_m14.py >> $L 2>&1
echo "=== history-length curves" >> $L
$PY studies/diag_context_m11.py --cache-name meta4d --model model3.pt >> $L 2>&1
echo "=== M14 diversity done" >> $L
