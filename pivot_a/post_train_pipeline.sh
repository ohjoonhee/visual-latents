#!/bin/bash
# Pivot A — autonomous post-train pipeline (SERIALIZED).
#
# To avoid GPU contention with C2 training, this pipeline does NOT start
# C1 extraction the moment C1 training PID exits. Instead, it waits for
# BOTH C1 and C2 training to finish, then runs C1 extract+ablate, then
# C2 extract+ablate, then build_report.
#
# Usage:
#   nohup bash pivot_a/post_train_pipeline.sh > pivot_a/results/pipeline.log 2>&1 &
#
# Expected files set by run_full.sh:
#   pivot_a/results/cos/train.pid    (C1 train PID — written immediately at start)
#   pivot_a/results/vicreg/train.pid (C2 train PID — written only AFTER C1 finishes,
#                                      so we cannot wait for this file to APPEAR
#                                      until C1 PID has exited.)

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
    echo "[pipeline] waiting for $pidfile ($label) to appear ($(date))"
    while [ ! -f "$pidfile" ] && [ $seen -lt $cap ]; do
        sleep 30
        seen=$((seen + 30))
    done
    if [ ! -f "$pidfile" ]; then
        echo "[pipeline] FAIL: $pidfile never appeared after ${cap}s" >&2
        exit 1
    fi
    echo "[pipeline] $pidfile present at $(date)"
}

wait_for_pid_exit() {
    local pidfile=$1
    local label=$2
    local rdir=$3
    local PID
    PID=$(cat "$pidfile")
    echo "[pipeline] $label train PID=$PID — waiting for exit ($(date))"
    while ps -p $PID > /dev/null 2>&1; do
        sleep 60
        latest=$(tail -1 $rdir/training_log.jsonl 2>/dev/null | head -c 200)
        echo "[pipeline] $(date) [$label] still training; last_log: $latest"
    done
    echo "[pipeline] $label training PID exited at $(date)"
}

post_one() {
    local variant=$1            # "cos" or "vicreg"
    local self_name=$2
    local rdir=$RES/$variant

    if [ ! -d "$rdir/checkpoint" ]; then
        echo "[pipeline] FAIL: no checkpoint at $rdir/checkpoint" >&2
        tail -50 $rdir/train_stdout.log 2>&1 || true
        exit 1
    fi
    echo "[pipeline] === $variant post-train ($(date)) ==="

    echo "[pipeline] $variant extract starting ($(date))"
    $PY $ROOT/extract_latents.py \
        --ckpt $rdir/checkpoint \
        --out $rdir/latents \
        --n 200 \
        > $rdir/extract_stdout.log 2>&1
    echo "[pipeline] $variant extract done ($(date))"
    tail -3 $rdir/extract_stdout.log || true

    echo "[pipeline] $variant ablate starting ($(date))"
    $PY $ROOT/ablation_runner.py \
        --latents_dir $rdir/latents \
        --self_ckpt $rdir/checkpoint \
        --self_name $self_name \
        --out_results $rdir/ablation_results.jsonl \
        --out_hstats $rdir/ablation_h_stats.jsonl \
        > $rdir/ablation_stdout.log 2>&1
    echo "[pipeline] $variant ablate done ($(date))"
    tail -3 $rdir/ablation_stdout.log || true
}

# Phase 1: wait for C1 training to finish (its PID file is written first)
wait_for_pidfile $RES/cos/train.pid cos
wait_for_pid_exit $RES/cos/train.pid cos $RES/cos

# Phase 2: wait for C2 training to start AND finish.
# Note: run_full.sh writes vicreg/train.pid only after C1 trainer exits, so
# this file will appear shortly after the previous step.
wait_for_pidfile $RES/vicreg/train.pid vicreg
wait_for_pid_exit $RES/vicreg/train.pid vicreg $RES/vicreg

# Phase 3: now both trainings are done — run extract+ablate for each in turn.
post_one cos pivot_a_cos_self
post_one vicreg pivot_a_vicreg_self

echo "[pipeline] building unified REPORT.md ($(date))"
$PY $ROOT/build_report.py
echo "[pipeline] all done ($(date))"
