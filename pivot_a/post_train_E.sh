#!/bin/bash
# Pivot A — Experiment E autonomous post-train pipeline.
#
# E = D2 recipe (VICReg λ_reg=2.0, mean-MSE LVR) at 2× steps (2000).
# Tests whether D2's semantic gap (n_helpful=2, qwen_base utility=+0.05) is
# an undertraining artifact.
#
# Single training only — no serialization needed. Polls train.pid, runs
# extract → ablate → build_report when training exits.
#
# Usage:
#   nohup bash pivot_a/post_train_E.sh > pivot_a/results/pipeline_E.log 2>&1 &
#
# Expected file:
#   pivot_a/results/vicreg_lambda2_2k/train.pid

set -e
cd /mnt/ssd/Projects/visual-latents
PY=phase0_monet_probe/.venv-monet/bin/python
ROOT=pivot_a
RES=$ROOT/results
RDIR=$RES/vicreg_lambda2_2k
PIDFILE=$RDIR/train.pid

wait_for_pidfile() {
    local cap=${1:-21600}     # 6h default
    local seen=0
    echo "[pipeline_E] waiting for $PIDFILE to appear ($(date))"
    while [ ! -f "$PIDFILE" ] && [ $seen -lt $cap ]; do
        sleep 30
        seen=$((seen + 30))
    done
    if [ ! -f "$PIDFILE" ]; then
        echo "[pipeline_E] FAIL: $PIDFILE never appeared after ${cap}s" >&2
        exit 1
    fi
    echo "[pipeline_E] $PIDFILE present at $(date)"
}

wait_for_pid_exit() {
    local PID
    PID=$(cat "$PIDFILE")
    echo "[pipeline_E] E train PID=$PID — waiting for exit ($(date))"
    while ps -p $PID > /dev/null 2>&1; do
        sleep 60
        latest=$(tail -1 $RDIR/training_log.jsonl 2>/dev/null | head -c 200)
        echo "[pipeline_E] $(date) [E] still training; last_log: $latest"
    done
    echo "[pipeline_E] E training PID exited at $(date)"
}

wait_for_pidfile
wait_for_pid_exit

if [ ! -d "$RDIR/checkpoint" ]; then
    echo "[pipeline_E] FAIL: no checkpoint at $RDIR/checkpoint" >&2
    tail -50 $RDIR/train_stdout.log 2>&1 || true
    exit 1
fi

echo "[pipeline_E] === E post-train ($(date)) ==="

echo "[pipeline_E] E extract starting ($(date))"
$PY $ROOT/extract_latents.py \
    --ckpt $RDIR/checkpoint \
    --out $RDIR/latents \
    --n 200 \
    > $RDIR/extract_stdout.log 2>&1
echo "[pipeline_E] E extract done ($(date))"
tail -3 $RDIR/extract_stdout.log || true

echo "[pipeline_E] E ablate starting ($(date))"
$PY $ROOT/ablation_runner.py \
    --latents_dir $RDIR/latents \
    --self_ckpt $RDIR/checkpoint \
    --self_name pivot_a_vicreg_lambda2_2k_self \
    --out_results $RDIR/ablation_results.jsonl \
    --out_hstats $RDIR/ablation_h_stats.jsonl \
    > $RDIR/ablation_stdout.log 2>&1
echo "[pipeline_E] E ablate done ($(date))"
tail -3 $RDIR/ablation_stdout.log || true

echo "[pipeline_E] building unified REPORT.md ($(date))"
$PY $ROOT/build_report.py
echo "[pipeline_E] all done ($(date))"
