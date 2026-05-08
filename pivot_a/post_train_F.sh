#!/bin/bash
# Pivot A — Experiment F autonomous post-train pipeline.
#
# F = D2 winning recipe (VICReg λ_reg=2.0, mean-MSE LVR) at K=4 (was K=8).
# Tests whether reducing the per-position update budget concentrates the
# per-position loss signal, closing the n_helpful semantic gap observed in
# C2/D2/E (n_helpful=2/8, mid positions stuck at margin 0.01-0.04).
#
# Single training only. Polls train.pid, runs extract → ablate → builds
# REPORT_F.md when training exits. Decoupled from REPORT.md to avoid
# breaking the K=8 hardcoding in build_report.py.
#
# Usage:
#   nohup bash pivot_a/post_train_F.sh > pivot_a/results/pipeline_F.log 2>&1 &
#
# Expected file:
#   pivot_a/results/vicreg_K4/train.pid

set -e
cd /mnt/ssd/Projects/visual-latents
PY=phase0_monet_probe/.venv-monet/bin/python
ROOT=pivot_a
RES=$ROOT/results
RDIR=$RES/vicreg_K4
PIDFILE=$RDIR/train.pid

wait_for_pidfile() {
    local cap=${1:-21600}     # 6h default
    local seen=0
    echo "[pipeline_F] waiting for $PIDFILE to appear ($(date))"
    while [ ! -f "$PIDFILE" ] && [ $seen -lt $cap ]; do
        sleep 30
        seen=$((seen + 30))
    done
    if [ ! -f "$PIDFILE" ]; then
        echo "[pipeline_F] FAIL: $PIDFILE never appeared after ${cap}s" >&2
        exit 1
    fi
    echo "[pipeline_F] $PIDFILE present at $(date)"
}

wait_for_pid_exit() {
    local PID
    PID=$(cat "$PIDFILE")
    echo "[pipeline_F] F train PID=$PID — waiting for exit ($(date))"
    while ps -p $PID > /dev/null 2>&1; do
        sleep 60
        latest=$(tail -1 $RDIR/training_log.jsonl 2>/dev/null | head -c 200)
        echo "[pipeline_F] $(date) [F] still training; last_log: $latest"
    done
    echo "[pipeline_F] F training PID exited at $(date)"
}

wait_for_pidfile
wait_for_pid_exit

if [ ! -d "$RDIR/checkpoint" ]; then
    echo "[pipeline_F] FAIL: no checkpoint at $RDIR/checkpoint" >&2
    tail -50 $RDIR/train_stdout.log 2>&1 || true
    exit 1
fi

echo "[pipeline_F] === F post-train ($(date)) ==="

echo "[pipeline_F] F extract starting ($(date))"
$PY $ROOT/extract_latents.py \
    --ckpt $RDIR/checkpoint \
    --out $RDIR/latents \
    --n 200 \
    --K 4 \
    > $RDIR/extract_stdout.log 2>&1
echo "[pipeline_F] F extract done ($(date))"
tail -3 $RDIR/extract_stdout.log || true

echo "[pipeline_F] F ablate starting ($(date))"
$PY $ROOT/ablation_runner.py \
    --latents_dir $RDIR/latents \
    --self_ckpt $RDIR/checkpoint \
    --self_name pivot_a_vicreg_K4_self \
    --K 4 \
    --out_results $RDIR/ablation_results.jsonl \
    --out_hstats $RDIR/ablation_h_stats.jsonl \
    > $RDIR/ablation_stdout.log 2>&1
echo "[pipeline_F] F ablate done ($(date))"
tail -3 $RDIR/ablation_stdout.log || true

echo "[pipeline_F] building REPORT_F.md ($(date))"
$PY $ROOT/build_report_F.py
echo "[pipeline_F] all done ($(date))"
