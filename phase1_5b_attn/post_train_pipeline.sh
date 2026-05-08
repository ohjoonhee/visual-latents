#!/bin/bash
# Phase 1.5b — wait for training to finish, then run extraction, ablation, report.
set -e
cd /mnt/ssd/Projects/visual-latents
PY=phase0_monet_probe/.venv-monet/bin/python
RUN=phase1_5b_attn/results/run_p15b
PID_FILE=$RUN/train.pid

if [ ! -f "$PID_FILE" ]; then
    echo "[pipeline] no train.pid — assuming training is already done"
else
    PID=$(cat "$PID_FILE")
    echo "[pipeline] waiting for training PID=$PID to finish ($(date))"
    while ps -p $PID > /dev/null 2>&1; do
        sleep 60
        latest=$(tail -1 $RUN/training_log.jsonl 2>/dev/null | head -c 200)
        echo "[pipeline] $(date) waiting... last_log: $latest"
    done
    echo "[pipeline] training PID exited at $(date)"
fi

# Sanity: checkpoint must exist
if [ ! -d "$RUN/checkpoint" ]; then
    echo "[pipeline] FAIL: no checkpoint at $RUN/checkpoint" >&2
    tail -30 $RUN/train_stdout.log
    exit 1
fi
echo "[pipeline] checkpoint OK at $RUN/checkpoint"

# Step 1: extract latents (held-out 200 from eval split)
echo "[pipeline] starting extract ($(date))"
$PY phase1_5b_attn/extract_latents.py \
    --ckpt $RUN/checkpoint \
    --out $RUN/latents \
    --n 200 \
    > $RUN/extract_stdout.log 2>&1
echo "[pipeline] extract done ($(date)). last lines:"
tail -5 $RUN/extract_stdout.log

# Step 2: ablation (qwen_base + phase1_5b_self)
echo "[pipeline] starting ablation ($(date))"
$PY phase1_5b_attn/ablation_runner.py \
    --latents_dir $RUN/latents \
    --self_ckpt $RUN/checkpoint \
    --self_name phase1_5b_self \
    --out_results $RUN/ablation_results.jsonl \
    --out_hstats $RUN/ablation_h_stats.jsonl \
    > $RUN/ablation_stdout.log 2>&1
echo "[pipeline] ablation done ($(date)). last lines:"
tail -5 $RUN/ablation_stdout.log

# Step 3: build report
echo "[pipeline] building report ($(date))"
$PY phase1_5b_attn/build_report.py
echo "[pipeline] all done ($(date))"
