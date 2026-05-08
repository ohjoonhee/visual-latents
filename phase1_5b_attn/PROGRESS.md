# Phase 1.5b — Progress Log

## Current state (2026-05-08 09:26 KST)

**Smoke**: PASS (50 steps, loss 27.78 -> 19.56, no NaN, ~5s/step,
mem=37.1GB peak, total 253s).
Log: `phase1_5b_attn/results/smoke/smoke_stdout.log`.

**Full train**: RUNNING.
- PID: see `phase1_5b_attn/results/run_p15b/train.pid` (was 531157 at launch)
- Log: `phase1_5b_attn/results/run_p15b/train_stdout.log`
- Started: 2026-05-08 09:26:25 KST
- Expected duration: ~85 min (1000 steps × ~5s/step + setup)
- Expected ETA: ~10:51 KST

## How to check status

```bash
PID=$(cat phase1_5b_attn/results/run_p15b/train.pid)
ps -p $PID > /dev/null && echo "RUNNING" || echo "DONE"
tail -20 phase1_5b_attn/results/run_p15b/train_stdout.log
```

## Next steps after train finishes

1. Verify checkpoint exists at `phase1_5b_attn/results/run_p15b/checkpoint`.
2. Run extraction:
   ```bash
   PY=phase0_monet_probe/.venv-monet/bin/python
   RUN=phase1_5b_attn/results/run_p15b
   $PY phase1_5b_attn/extract_latents.py \
       --ckpt $RUN/checkpoint \
       --out $RUN/latents \
       --n 200 \
       > $RUN/extract_stdout.log 2>&1
   ```
3. Run ablation:
   ```bash
   $PY phase1_5b_attn/ablation_runner.py \
       --latents_dir $RUN/latents \
       --self_ckpt $RUN/checkpoint \
       --self_name phase1_5b_self \
       --out_results $RUN/ablation_results.jsonl \
       --out_hstats $RUN/ablation_h_stats.jsonl \
       > $RUN/ablation_stdout.log 2>&1
   ```
4. Build report:
   ```bash
   $PY phase1_5b_attn/build_report.py
   ```

The convenience script `phase1_5b_attn/run_full.sh` does (1)+(2)+(3)+(4)
sequentially BUT also reruns step 1 (training) — use only step 2-4 if
training has already finished.
