#!/bin/bash
# Pivot A — Experiment D follow-up runner.
#
# Sequence (single GPU, no contention):
#   1. Train D1 (vicreg + sum-MSE LVR)   ~22 min
#   2. Train D2 (vicreg, lambda_reg=2)   ~22 min
#   3. post_train_followup.sh (launched at start, blocks on pid files):
#        wait for both train.pid files to exit
#        → D1 extract + D1 ablate → D2 extract + D2 ablate → build_report
#
# Modeled exactly on run_full.sh.
#
# Usage:
#   cd /mnt/ssd/Projects/visual-latents
#   nohup bash pivot_a/run_followup.sh > pivot_a/results/run_followup.log 2>&1 &

set -e
cd /mnt/ssd/Projects/visual-latents
PY=phase0_monet_probe/.venv-monet/bin/python
ROOT=pivot_a
RES=$ROOT/results
mkdir -p $RES/vicreg_summse $RES/vicreg_lambda2

echo "[run_followup] launching post_train_followup.sh in background ($(date))"
nohup bash $ROOT/post_train_followup.sh > $RES/pipeline_followup.log 2>&1 &
PIPELINE_PID=$!
echo $PIPELINE_PID > $RES/pipeline_followup.pid
echo "[run_followup] pipeline PID=$PIPELINE_PID"

run_train_one() {
    local variant=$1
    local cfg=$2
    local rdir=$RES/$variant

    echo "[run_followup] === train $variant ($(date)) ==="
    $PY $ROOT/trainer.py --config $cfg > $rdir/train_stdout.log 2>&1 &
    local TRAIN_PID=$!
    echo $TRAIN_PID > $rdir/train.pid
    echo "[run_followup] $variant train PID=$TRAIN_PID"
    wait $TRAIN_PID
    local rc=$?
    echo "[run_followup] $variant train exit=$rc ($(date))"
    if [ $rc -ne 0 ]; then
        echo "[run_followup] FAIL: $variant trainer rc=$rc"
        tail -50 $rdir/train_stdout.log
        exit $rc
    fi
}

run_train_one vicreg_summse  $ROOT/configs/pivot_a_vicreg_summse.yaml
run_train_one vicreg_lambda2 $ROOT/configs/pivot_a_vicreg_lambda2.yaml

echo "[run_followup] both trainings done ($(date)). Pipeline (PID=$PIPELINE_PID) handles extract+ablate+report."
echo "[run_followup] tail -f $RES/pipeline_followup.log to follow."
