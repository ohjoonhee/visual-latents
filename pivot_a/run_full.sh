#!/bin/bash
# Pivot A — full pipeline runner.
#
# Sequence (single GPU, no contention):
#   1. Train C1 (cos)         ~22 min
#   2. Train C2 (vicreg)      ~22 min
#   3. post_train_pipeline.sh (launched at start, blocks on pid files):
#        wait for both train.pid files to exit
#        → C1 extract + C1 ablate → C2 extract + C2 ablate → build_report
#
# This script writes train.pid for each variant before starting the trainer.
# The post-train pipeline waits for both trainings to finish before starting
# any extraction (avoids GPU contention with the next training).
#
# Usage:
#   cd /mnt/ssd/Projects/visual-latents
#   nohup bash pivot_a/run_full.sh > pivot_a/results/run_full.log 2>&1 &

set -e
cd /mnt/ssd/Projects/visual-latents
PY=phase0_monet_probe/.venv-monet/bin/python
ROOT=pivot_a
RES=$ROOT/results
mkdir -p $RES/cos $RES/vicreg

echo "[run_full] launching post_train_pipeline.sh in background ($(date))"
nohup bash $ROOT/post_train_pipeline.sh > $RES/pipeline.log 2>&1 &
PIPELINE_PID=$!
echo $PIPELINE_PID > $RES/pipeline.pid
echo "[run_full] pipeline PID=$PIPELINE_PID"

run_train_one() {
    local variant=$1
    local cfg=$2
    local rdir=$RES/$variant

    echo "[run_full] === train $variant ($(date)) ==="
    $PY $ROOT/trainer.py --config $cfg > $rdir/train_stdout.log 2>&1 &
    local TRAIN_PID=$!
    echo $TRAIN_PID > $rdir/train.pid
    echo "[run_full] $variant train PID=$TRAIN_PID"
    wait $TRAIN_PID
    local rc=$?
    echo "[run_full] $variant train exit=$rc ($(date))"
    if [ $rc -ne 0 ]; then
        echo "[run_full] FAIL: $variant trainer rc=$rc"
        tail -50 $rdir/train_stdout.log
        exit $rc
    fi
}

run_train_one cos    $ROOT/configs/pivot_a_cos.yaml
run_train_one vicreg $ROOT/configs/pivot_a_vicreg.yaml

echo "[run_full] both trainings done ($(date)). Pipeline (PID=$PIPELINE_PID) handles extract+ablate+report."
echo "[run_full] tail -f $RES/pipeline.log to follow."
