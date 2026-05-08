# Pivot A — Experiment E — PROGRESS

Launched: 2026-05-08 13:58:42 KST

## What E is

A single follow-up to D2 (the Pivot A winning recipe: VICReg λ_reg=2.0,
mean-MSE LVR), with **2× training (2000 steps)** and proportionally scaled
warmup (200 = 10% of total). Tests whether D2's semantic gap
(n_helpful=2 / qwen_base utility=+0.05) is an undertraining artifact.

Hypothesis: with 2× steps, n_helpful → ≥3 and qwen_base utility → ≥+0.10.

## Running processes

| role | PID | logs |
|---|---|---|
| E trainer (vicreg_lambda2_2k) | 559282 | `pivot_a/results/vicreg_lambda2_2k/train_stdout.log` |
| post_train_E.sh (autonomous chain) | 559335 | `pivot_a/results/pipeline_E.log` |

## What runs

1. **E**: VICReg λ_reg=2.0, mean-MSE, 2000 steps, warmup=200, save_every=500
   - config: `pivot_a/configs/pivot_a_vicreg_lambda2_2k.yaml`
   - results dir: `pivot_a/results/vicreg_lambda2_2k`
   - intermediate ckpts: step 500 / 1000 / 1500 / 2000 (final under `checkpoint/`)
2. After E training exits, the autonomous pipeline runs:
   - extract E latents (200 eval examples, identical seed=0 set used by C2/D2)
   - E ablate (self-reader `pivot_a_vicreg_lambda2_2k_self` + `qwen_base`)
   - regenerate `pivot_a/REPORT.md` via the updated 8-column build_report.py
     (which now includes E's row)

## Smoke results (pre-launch)

- E smoke (50 steps, same `pivot_a_vicreg_lambda2_2k_smoke.yaml`): no NaN;
  loss 105 → 96; lvr ~16–18; reg ~38–43 (decays as VICReg pulls slots
  apart). Identical trajectory to D2's smoke — recipe is healthy.

## First-10-steps live check

- step 1 loss=115.525, step 10 loss=109.789 — matches D2 baseline trajectory.
- No NaN in `training_log.jsonl`.

## Expected ETAs (from launch 13:58:42)

| milestone | eta |
|---|---|
| E train done (2000 steps) | ~14:43 (~45 min, 2× D2's 22 min) |
| E extract+ablate done | ~14:48 (~5 min) |
| Final REPORT.md updated | ~14:48 (~50 min total) |

## Acceptance per variant (target)

| criterion | target |
|---|---|
| compression_ratio | ≥ 0.4 |
| mean off-diag cos | ≤ 0.55 |
| n_helpful | ≥ 3 (D2 was 2 — this is the gap E is probing) |
| qwen_base utility | > 0 (D2 was +0.050; target ≥ +0.10) |

## Monitoring (no agent action)

```sh
tail -f pivot_a/results/vicreg_lambda2_2k/train_stdout.log
tail -f pivot_a/results/pipeline_E.log
```
