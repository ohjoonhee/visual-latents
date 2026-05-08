#!/bin/bash
# Pivot A — Experiment H autonomous post-train pipeline.
# H = F recipe (K=4, λ_reg=2.0) with seed=1. Reproducibility check.
#
# Single training. Polls train.pid, runs extract → ablate → small REPORT_H.md.

set -e
cd /mnt/ssd/Projects/visual-latents
PY=phase0_monet_probe/.venv-monet/bin/python
ROOT=pivot_a
RES=$ROOT/results
RDIR=$RES/vicreg_K4_s1
PIDFILE=$RDIR/train.pid

if [ ! -f "$PIDFILE" ]; then
    echo "[H] no train.pid; aborting"
    exit 1
fi
PID=$(cat "$PIDFILE")
echo "[H] waiting for PID=$PID at $(date)"
while ps -p $PID > /dev/null 2>&1; do
    sleep 60
    latest=$(tail -1 $RDIR/training_log.jsonl 2>/dev/null | head -c 200)
    echo "[H] $(date) still training; last_log: $latest"
done
echo "[H] training PID exited at $(date)"

if [ ! -d "$RDIR/checkpoint" ]; then
    echo "[H] FAIL: no checkpoint at $RDIR/checkpoint"
    tail -50 $RDIR/train_stdout.log
    exit 1
fi

echo "[H] extracting at $(date)"
$PY $ROOT/extract_latents.py --ckpt $RDIR/checkpoint --out $RDIR/latents --n 200 --K 4 > $RDIR/extract_stdout.log 2>&1
echo "[H] extract done at $(date)"
tail -3 $RDIR/extract_stdout.log

echo "[H] ablating at $(date)"
$PY $ROOT/ablation_runner.py \
    --latents_dir $RDIR/latents \
    --self_ckpt $RDIR/checkpoint \
    --self_name pivot_a_vicreg_K4_s1_self \
    --K 4 \
    --out_results $RDIR/ablation_results.jsonl \
    --out_hstats $RDIR/ablation_h_stats.jsonl \
    > $RDIR/ablation_stdout.log 2>&1
echo "[H] ablate done at $(date)"
tail -3 $RDIR/ablation_stdout.log

echo "[H] all done at $(date) — see $RDIR/ablation_results.jsonl"
