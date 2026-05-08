# Pivot A — Experiment D follow-up — PROGRESS

Launched: 2026-05-08 12:57:08 KST

## Running processes

| role | PID | logs |
|---|---|---|
| run_followup.sh (chain) | 555053 | `pivot_a/results/run_followup.log` |
| post_train_followup.sh | 555057 | `pivot_a/results/pipeline_followup.log` |
| D1 trainer (vicreg_summse) | 555059 (initial) | `pivot_a/results/vicreg_summse/train_stdout.log` |

D2 trainer PID will be written to `pivot_a/results/vicreg_lambda2/train.pid`
once D1 finishes.

## What runs

1. **D1**: VICReg + paper-faithful sum-MSE LVR
   - config: `pivot_a/configs/pivot_a_vicreg_summse.yaml`
   - `loss_form: sum_d`, `lambda_reg: 1.0`
   - results dir: `pivot_a/results/vicreg_summse`
2. **D2**: VICReg with `λ_reg = 2.0` (mean-MSE)
   - config: `pivot_a/configs/pivot_a_vicreg_lambda2.yaml`
   - `loss_form: mean_d`, `lambda_reg: 2.0`
   - results dir: `pivot_a/results/vicreg_lambda2`
3. After both trainings finish, the autonomous pipeline runs (serialized,
   to avoid GPU contention):
   - extract D1 latents (200 eval examples) + D1 ablate (self + qwen_base)
   - extract D2 latents (200 eval examples) + D2 ablate (self + qwen_base)
   - regenerate `pivot_a/REPORT.md` via the updated 7-column build_report.py

## Smoke results (pre-launch)

- D1 smoke (50 steps): no NaN; loss 33740 → 19984; LVR ~33700 → 19940
  (sum-MSE is ~D=2048× the mean-MSE form, as expected); reg ~38–43.
- D2 smoke (50 steps): no NaN; loss 105 → 65; LVR ~16; reg ~23–43 (higher
  early due to fresh init, decays as VICReg pulls slots apart).

## Expected ETAs (from launch 12:57:08)

| milestone | eta |
|---|---|
| D1 train done | ~13:19 (~22 min) |
| D2 train done | ~13:42 (~22 min after D1) |
| D1 extract+ablate done | ~13:48 (~6 min) |
| D2 extract+ablate done | ~13:54 (~6 min) |
| Final REPORT.md built | ~13:55 (~58 min total) |

## Acceptance per variant (target)

| criterion | target |
|---|---|
| compression_ratio | ≥ 0.4 |
| mean off-diag cos | ≤ 0.55 (dispositive) |
| n_helpful | ≥ 3 |
| qwen_base utility | > 0 |

## Monitoring (no agent action)

```sh
tail -f pivot_a/results/run_followup.log
tail -f pivot_a/results/pipeline_followup.log
tail -f pivot_a/results/vicreg_summse/train_stdout.log
tail -f pivot_a/results/vicreg_lambda2/train_stdout.log
```
