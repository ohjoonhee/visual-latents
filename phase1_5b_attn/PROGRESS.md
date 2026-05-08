# Phase 1.5b — Progress Log

## Current state (2026-05-08 10:00 KST)

**Smoke**: PASS (50 steps, loss 27.78 -> 19.56, no NaN, ~5s/step,
mem=37.1GB peak, total 253s).

**Full train**: RUNNING.
- PID: see `phase1_5b_attn/results/run_p15b/train.pid` (531157 at launch)
- Log: `phase1_5b_attn/results/run_p15b/train_stdout.log`
- Started: 2026-05-08 09:26:25 KST
- Latest: step 400/1000 at elapsed=2021s, loss=5.194 ntp=0.787 lvr=4.406
- Expected ETA: ~10:50 KST

**Post-train pipeline**: RUNNING.
- PID file: `phase1_5b_attn/results/run_p15b/pipeline.pid` (535456 at launch)
- Log: `phase1_5b_attn/results/run_p15b/pipeline_stdout.log`
- Will wait for train PID, then run extract -> ablation -> report.

## How to check status

```bash
RUN=phase1_5b_attn/results/run_p15b
tail -3 $RUN/train_stdout.log
tail -3 $RUN/pipeline_stdout.log
ps -p $(cat $RUN/train.pid) > /dev/null && echo "TRAIN: RUNNING" || echo "TRAIN: DONE"
ps -p $(cat $RUN/pipeline.pid) > /dev/null && echo "PIPE: RUNNING" || echo "PIPE: DONE"
```

## Manual recovery (if pipeline dies)

```bash
PY=phase0_monet_probe/.venv-monet/bin/python
RUN=phase1_5b_attn/results/run_p15b
$PY phase1_5b_attn/extract_latents.py --ckpt $RUN/checkpoint --out $RUN/latents --n 200 > $RUN/extract_stdout.log 2>&1
$PY phase1_5b_attn/ablation_runner.py \
    --latents_dir $RUN/latents --self_ckpt $RUN/checkpoint --self_name phase1_5b_self \
    --out_results $RUN/ablation_results.jsonl --out_hstats $RUN/ablation_h_stats.jsonl \
    > $RUN/ablation_stdout.log 2>&1
$PY phase1_5b_attn/build_report.py
```
