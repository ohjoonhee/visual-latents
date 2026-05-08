#!/bin/bash
# Phase 1.5b — full pipeline runner (1000 steps train + extract + ablate + report)
set -e
cd "$(dirname "$0")/.."
PY=phase0_monet_probe/.venv-monet/bin/python
RUN=phase1_5b_attn/results/run_p15b
mkdir -p "$RUN"

echo "[full-run] starting train ($(date))"
$PY phase1_5b_attn/trainer.py --config phase1_5b_attn/configs/p15b_3b_5k.yaml \
    > "$RUN/train_stdout.log" 2>&1
echo "[full-run] train done ($(date))"

echo "[full-run] starting extract ($(date))"
$PY phase1_5b_attn/extract_latents.py \
    --ckpt "$RUN/checkpoint" \
    --out "$RUN/latents" \
    --n 200 \
    > "$RUN/extract_stdout.log" 2>&1
echo "[full-run] extract done ($(date))"

echo "[full-run] starting ablation ($(date))"
$PY phase1_5b_attn/ablation_runner.py \
    --latents_dir "$RUN/latents" \
    --self_ckpt "$RUN/checkpoint" \
    --self_name phase1_5b_self \
    --out_results "$RUN/ablation_results.jsonl" \
    --out_hstats "$RUN/ablation_h_stats.jsonl" \
    > "$RUN/ablation_stdout.log" 2>&1
echo "[full-run] ablation done ($(date))"

echo "[full-run] building report"
$PY phase1_5b_attn/build_report.py
echo "[full-run] all done ($(date))"
