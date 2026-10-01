#!/bin/sh
# MPC comparison, round 2 (run next to the main studies.mpc_compare run):
#   1. the full learned controller with the prior net and the impact term
#      over the whole horizon (--j-imp 24), since the main run's single
#      horizon (4 steps) was set by the oracle net and leaves the planner
#      blind to the impacts its speed choice causes after 1 s;
#   2. the cheaper uses of the learned model (hold, hold_c, nominal, sens)
#      with the main run's horizon (--pre-eval: its hcheck floor is 4).
# One process at a time: both write studies/_cache/mpc_variants/results.pkl.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
L=studies/_cache/mpc_variants/round2_stdout.log
mkdir -p studies/_cache/mpc_variants
echo "=== full, j_imp 24" >> $L
$PY -m studies.mpc_compare_variants --variants full --j-imp 24 >> $L 2>&1
echo "=== cheap variants, main horizon" >> $L
$PY -m studies.mpc_compare_variants --variants hold,hold_c,nominal,sens \
    --pre-eval >> $L 2>&1
echo "=== round 2 done" >> $L
