#!/bin/bash
# Pivot A — Experiment D autonomous post-train pipeline (SERIALIZED).
#
# To avoid GPU contention with the next training, this pipeline does NOT
# start D1 extraction the moment D1 training PID exits. Instead, it waits
# for BOTH D1 and D2 training to finish, then runs D1 extract+ablate, then
# D2 extract+ablate, then build_report.
#
# Usage:
#   nohup bash pivot_a/post_train_followup.sh > pivot_a/results/pipeline_followup.log 2>&1 &
#
# Expected files set by run_followup.sh:
#   pivot_a/results/vicreg_summse/train.pid    (D1 PID — written immediately at start)
#   pivot_a/results/vicreg_lambda2/train.pid   (D2 PID — written only AFTER D1 finishes)

set -e
cd /mnt/ssd/Projects/visual-latents
PY=phase0_monet_probe/.venv-monet/bin/python
ROOT=pivot_a
RES=$ROOT/results

wait_for_pidfile() {
    local pidfile=$1
    local label=$2
    local cap=${3:-21600}     # default 6h cap
    local seen=0
    echo "[pipeline_fu] waiting for $pidfile ($label) to appear ($(date))"
    while [ ! -f "$pidfile" ] && [ $seen -lt $cap ]; do
        sleep 30
        seen=$((seen + 30))
    done
    if [ ! -f "$pidfile" ]; then
        echo "[pipeline_fu] FAIL: $pidfile never appeared after ${cap}s" >&2
        exit 1
    fi
    echo "[pipeline_fu] $pidfile present at $(date)"
}

wait_for_pid_exit() {
    local pidfile=$1
    local label=$2
    local rdir=$3
    local PID
    PID=$(cat "$pidfile")
    echo "[pipeline_fu] $label train PID=$PID — waiting for exit ($(date))"
    while ps -p $PID > /dev/null 2>&1; do
        sleep 60
        latest=$(tail -1 $rdir/training_log.jsonl 2>/dev/null | head -c 200)
        echo "[pipeline_fu] $(date) [$label] still training; last_log: $latest"
    done
    echo "[pipeline_fu] $label training PID exited at $(date)"
}

post_one() {
    local variant=$1
    local self_name=$2
    local rdir=$RES/$variant

    if [ ! -d "$rdir/checkpoint" ]; then
        echo "[pipeline_fu] FAIL: no checkpoint at $rdir/checkpoint" >&2
        tail -50 $rdir/train_stdout.log 2>&1 || true
        exit 1
    fi
    echo "[pipeline_fu] === $variant post-train ($(date)) ==="

    echo "[pipeline_fu] $variant extract starting ($(date))"
    $PY $ROOT/extract_latents.py \
        --ckpt $rdir/checkpoint \
        --out $rdir/latents \
        --n 200 \
        > $rdir/extract_stdout.log 2>&1
    echo "[pipeline_fu] $variant extract done ($(date))"
    tail -3 $rdir/extract_stdout.log || true

    echo "[pipeline_fu] $variant ablate starting ($(date))"
    $PY $ROOT/ablation_runner.py \
        --latents_dir $rdir/latents \
        --self_ckpt $rdir/checkpoint \
        --self_name $self_name \
        --out_results $rdir/ablation_results.jsonl \
        --out_hstats $rdir/ablation_h_stats.jsonl \
        > $rdir/ablation_stdout.log 2>&1
    echo "[pipeline_fu] $variant ablate done ($(date))"
    tail -3 $rdir/ablation_stdout.log || true
}

# Phase 1: wait for D1 training to finish (its PID file is written first)
wait_for_pidfile $RES/vicreg_summse/train.pid vicreg_summse
wait_for_pid_exit $RES/vicreg_summse/train.pid vicreg_summse $RES/vicreg_summse

# Phase 2: wait for D2 training to start AND finish.
# Note: run_followup.sh writes vicreg_lambda2/train.pid only after D1 trainer
# exits, so this file will appear shortly after the previous step.
wait_for_pidfile $RES/vicreg_lambda2/train.pid vicreg_lambda2
wait_for_pid_exit $RES/vicreg_lambda2/train.pid vicreg_lambda2 $RES/vicreg_lambda2

# Phase 3: now both trainings are done — run extract+ablate for each in turn.
post_one vicreg_summse  pivot_a_vicreg_summse_self
post_one vicreg_lambda2 pivot_a_vicreg_lambda2_self

echo "[pipeline_fu] building unified REPORT.md ($(date))"
$PY $ROOT/build_report.py
echo "[pipeline_fu] all done ($(date))"
