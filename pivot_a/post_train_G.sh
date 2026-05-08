#!/bin/bash
# Pivot A — Experiment G autonomous post-train pipeline.
#
# G = D2 winning recipe (VICReg λ_reg=2.0, mean-MSE LVR) at K=4 (from F)
#     + 2000 training steps (from E). Stacks F's K=4 win with E's 2× steps
#     probe to see if more compute on top of K=4 closes more of the
#     semantic gap (target: utility ≥ +0.20, n_helpful ≥ 3/4).
#
# Single training only. Polls train.pid, runs extract → ablate → builds
# REPORT_G.md (5-col comparison D2/E/F/G/Monet) when training exits.
# Decoupled from REPORT_F.md / REPORT.md to keep prior reports stable.
#
# Usage:
#   nohup bash pivot_a/post_train_G.sh > pivot_a/results/pipeline_G.log 2>&1 &
#
# Expected file:
#   pivot_a/results/vicreg_K4_2k/train.pid

set -e
cd /mnt/ssd/Projects/visual-latents
PY=phase0_monet_probe/.venv-monet/bin/python
ROOT=pivot_a
RES=$ROOT/results
RDIR=$RES/vicreg_K4_2k
PIDFILE=$RDIR/train.pid

wait_for_pidfile() {
    local cap=${1:-21600}     # 6h default
    local seen=0
    echo "[pipeline_G] waiting for $PIDFILE to appear ($(date))"
    while [ ! -f "$PIDFILE" ] && [ $seen -lt $cap ]; do
        sleep 30
        seen=$((seen + 30))
    done
    if [ ! -f "$PIDFILE" ]; then
        echo "[pipeline_G] FAIL: $PIDFILE never appeared after ${cap}s" >&2
        exit 1
    fi
    echo "[pipeline_G] $PIDFILE present at $(date)"
}

wait_for_pid_exit() {
    local PID
    PID=$(cat "$PIDFILE")
    echo "[pipeline_G] G train PID=$PID — waiting for exit ($(date))"
    while ps -p $PID > /dev/null 2>&1; do
        sleep 60
        latest=$(tail -1 $RDIR/training_log.jsonl 2>/dev/null | head -c 200)
        echo "[pipeline_G] $(date) [G] still training; last_log: $latest"
    done
    echo "[pipeline_G] G training PID exited at $(date)"
}

wait_for_pidfile
wait_for_pid_exit

if [ ! -d "$RDIR/checkpoint" ]; then
    echo "[pipeline_G] FAIL: no checkpoint at $RDIR/checkpoint" >&2
    tail -50 $RDIR/train_stdout.log 2>&1 || true
    exit 1
fi

# Disk-space discipline: clean any stray intermediate ckpts before extract.
# (save_every=0 in the config, so this should be a no-op, but defensive.)
shopt -s nullglob
for ck in $RDIR/checkpoint_step*; do
    echo "[pipeline_G] cleaning intermediate ckpt: $ck"
    rm -rf "$ck"
done
shopt -u nullglob

echo "[pipeline_G] === G post-train ($(date)) ==="
df -h /mnt/ssd | tail -1

echo "[pipeline_G] G extract starting ($(date))"
$PY $ROOT/extract_latents.py \
    --ckpt $RDIR/checkpoint \
    --out $RDIR/latents \
    --n 200 \
    --K 4 \
    > $RDIR/extract_stdout.log 2>&1
echo "[pipeline_G] G extract done ($(date))"
tail -3 $RDIR/extract_stdout.log || true

echo "[pipeline_G] G ablate starting ($(date))"
$PY $ROOT/ablation_runner.py \
    --latents_dir $RDIR/latents \
    --self_ckpt $RDIR/checkpoint \
    --self_name pivot_a_vicreg_K4_2k_self \
    --K 4 \
    --out_results $RDIR/ablation_results.jsonl \
    --out_hstats $RDIR/ablation_h_stats.jsonl \
    > $RDIR/ablation_stdout.log 2>&1
echo "[pipeline_G] G ablate done ($(date))"
tail -3 $RDIR/ablation_stdout.log || true

echo "[pipeline_G] building REPORT_G.md ($(date))"
$PY $ROOT/build_report_G.py
echo "[pipeline_G] all done ($(date))"
