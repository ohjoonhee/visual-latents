# Pivot A — Experiment G (K=4 + 2× steps stacking probe)

## TL;DR

Experiment G stacks the two probes that each individually moved the dial:
- **F (K=4 @ 1k)** doubled n_helpful_rate (2/4 = 50%) and 2.6× qwen_base utility
  (+0.132) over D2 (2/8 = 25%, +0.050).
- **E (K=8 @ 2k)** held n_helpful flat (2/8 = 25%) but bumped utility +76% to
  +0.088 over D2.

G runs the F recipe (K=4, all VICReg + LVR settings unchanged) at 2000 steps
(warmup scaled to 200). Best case: utility ≥ +0.20, helpful ratio ≥ 75%
(3/4) — closing more of the gap toward Monet stage 2's per-position utility.

**Verdict**: G (K=4 @ 2k) **partially stacks**: utility 0.222 > F's 0.132 (helpful ratio 0.50 vs F 0.50). Longer training on top of K=4 still helps, but not enough to clear the n_helpful=75% bar.

## 5-column comparison

| metric | D2 (K=8) | E (K=8 @ 2k) | F (K=4 @ 1k) | **G (K=4 @ 2k)** | Monet stage 2 |
|---|---:|---:|---:|---:|---:|
| K | 8 | 8 | 4 | **4** | 8 |
| compression_ratio (≥ 0.4) | 0.792 | 0.875 | 0.351 | **0.372** | 2.7 |
| **mean off-diag cos (≤ 0.55)** | 0.341 | 0.389 | 0.311 | **0.372** | 0.38 |
| n_helpful | 2/8 | 2/8 | 2/4 | **2/4** | 4/8 |
| qwen_base utility (> 0) | 0.050 | 0.088 | 0.132 | **0.222** | +2.7 |
| utility (self reader) | 0.366 | 0.380 | 0.331 | **0.426** | +2.7 |
| v_roi off-diag cos | 0.465 | 0.465 | 0.512 | **0.512** | n/a |

## Acceptance per variant

### D2 — VICReg λ_reg=2.0, K=8 @ 1k (mean-MSE LVR) — PASS (3/4)

| criterion | result |
|---|---|
| compression_ratio ≥ 0.4 | PASS (0.792) |
| mean off-diag cos ≤ 0.55 | PASS (0.341) |
| n_helpful ≥ 3 | FAIL (2/8) |
| qwen_base utility > 0 | PASS (0.050) |


### E  — D2 recipe @ 2000 steps, K=8 — PASS (3/4)

| criterion | result |
|---|---|
| compression_ratio ≥ 0.4 | PASS (0.875) |
| mean off-diag cos ≤ 0.55 | PASS (0.389) |
| n_helpful ≥ 3 | FAIL (2/8) |
| qwen_base utility > 0 | PASS (0.088) |


### F  — D2 recipe @ K=4, 1000 steps — PASS (3/4)

| criterion | result |
|---|---|
| compression_ratio ≥ 0.4 | FAIL (0.351) |
| mean off-diag cos ≤ 0.55 | PASS (0.311) |
| n_helpful ≥ 2 | PASS (2/4) |
| qwen_base utility > 0 | PASS (0.132) |


### G  — D2 recipe @ K=4 + 2000 steps (this experiment) — PASS (3/4)

| criterion | result |
|---|---|
| compression_ratio ≥ 0.4 | FAIL (0.372) |
| mean off-diag cos ≤ 0.55 | PASS (0.372) |
| n_helpful ≥ 2 | PASS (2/4) |
| qwen_base utility > 0 | PASS (0.222) |


## D2 (K=8 @ 1k) — per-position single-keep curve (pivot_a_vicreg_lambda2_self reader)

```
pos      nll   margin  bar
  0     3.180   +0.029  ██  
  1     3.165   +0.044  ████  
  2     3.139   +0.070  ██████  (helpful)
  3     3.171   +0.038  ███  
  4     3.194   +0.016  █  
  5     3.191   +0.019  █  
  6     3.183   +0.026  ██  
  7     3.098   +0.111  █████████  (helpful)
```

## D2 (K=8 @ 1k) — 8×8 cosine matrix (mean over eval set)

```
       p0  p1  p2  p3  p4  p5  p6  p7
 p0  1.00  0.35  0.35  0.34  0.33  0.34  0.32  0.33
 p1  0.35  1.00  0.36  0.36  0.35  0.36  0.34  0.32
 p2  0.35  0.36  1.00  0.36  0.35  0.35  0.34  0.34
 p3  0.34  0.36  0.36  1.00  0.34  0.36  0.33  0.33
 p4  0.33  0.35  0.35  0.34  1.00  0.35  0.32  0.33
 p5  0.34  0.36  0.35  0.36  0.35  1.00  0.34  0.34
 p6  0.32  0.34  0.34  0.33  0.32  0.34  1.00  0.32
 p7  0.33  0.32  0.34  0.33  0.33  0.34  0.32  1.00

```

mean off-diagonal cos = **0.341**

Training summary: final step=1000 ntp=0.660 lvr=5.695 reg=11.758 ||h||=55.1 ||v||=95.9 elapsed=1343s


## E (K=8 @ 2k) — per-position single-keep curve (pivot_a_vicreg_lambda2_2k_self reader)

```
pos      nll   margin  bar
  0     2.723   +0.059  █████  (helpful)
  1     2.767   +0.014  █  
  2     2.752   +0.029  ██  
  3     2.748   +0.034  ███  
  4     2.760   +0.021  ██  
  5     2.769   +0.012  █  
  6     2.745   +0.036  ███  
  7     2.670   +0.112  █████████  (helpful)
```

## E (K=8 @ 2k) — 8×8 cosine matrix (mean over eval set)

```
       p0  p1  p2  p3  p4  p5  p6  p7
 p0  1.00  0.41  0.41  0.41  0.41  0.41  0.41  0.35
 p1  0.41  1.00  0.40  0.40  0.41  0.40  0.40  0.35
 p2  0.41  0.40  1.00  0.40  0.40  0.40  0.40  0.34
 p3  0.41  0.40  0.40  1.00  0.40  0.40  0.39  0.34
 p4  0.41  0.41  0.40  0.40  1.00  0.41  0.41  0.35
 p5  0.41  0.40  0.40  0.40  0.41  1.00  0.40  0.35
 p6  0.41  0.40  0.40  0.39  0.41  0.40  1.00  0.35
 p7  0.35  0.35  0.34  0.34  0.35  0.35  0.35  1.00

```

mean off-diagonal cos = **0.389**

Training summary: final step=2000 ntp=0.480 lvr=4.004 reg=11.753 ||h||=55.5 ||v||=76.0 elapsed=2688s


## F (K=4 @ 1k) — per-position single-keep curve (pivot_a_vicreg_K4_self reader)

```
pos      nll   margin  bar
  0     3.303   -0.013  ▒  
  1     3.291   -0.001    
  2     3.211   +0.079  ██████  (helpful)
  3     3.145   +0.144  ████████████  (helpful)
```

## F (K=4 @ 1k) — 4×4 cosine matrix (mean over eval set)

```
       p0  p1  p2  p3
 p0  1.00  0.31  0.31  0.31
 p1  0.31  1.00  0.32  0.31
 p2  0.31  0.32  1.00  0.31
 p3  0.31  0.31  0.31  1.00

```

mean off-diagonal cos = **0.311**

Training summary: final step=1000 ntp=0.685 lvr=5.927 reg=27.420 ||h||=53.4 ||v||=97.4 elapsed=1340s


## G (K=4 @ 2k) — per-position single-keep curve (pivot_a_vicreg_K4_2k_self reader)

```
pos      nll   margin  bar
  0     2.949   +0.028  ██  
  1     2.954   +0.023  ██  
  2     2.896   +0.081  ██████  (helpful)
  3     2.778   +0.199  ████████████████  (helpful)
```

## G (K=4 @ 2k) — 4×4 cosine matrix (mean over eval set)

```
       p0  p1  p2  p3
 p0  1.00  0.41  0.40  0.35
 p1  0.41  1.00  0.40  0.34
 p2  0.40  0.40  1.00  0.34
 p3  0.35  0.34  0.34  1.00

```

mean off-diagonal cos = **0.372**

Training summary: final step=2000 ntp=0.516 lvr=3.985 reg=27.388 ||h||=54.2 ||v||=75.4 elapsed=2664s


## Recipe (G — K=4 + 2× steps)

- Base: Qwen/Qwen2.5-VL-3B-Instruct (vision tower + projector frozen, LLM full FT)
- Loss: `L = L_NTP + λ_LVR · LVR + λ_reg · L_reg`
  - LVR: `F.mse_loss(h, v_roi)` (mean-MSE, scaled by D)
  - L_reg: `vicreg_loss(h, var_weight=1.0, cov_weight=0.04, gamma=1.0)`
    on z-scored h (per-dim across BK sample axis)
- λ_reg: 2.0 (D2/F winning value)
- Optimizer: AdamW lr=1e-5, weight_decay=0, betas=(0.9, 0.95)
- Schedule: cosine, warmup=200 (10% of total), decay→0
- Batch: micro=1, grad_accum=4 → eff=4
- **Max steps: 2000** (was 1000 in F)
- bf16 + gradient_checkpointing
- Data: `ohjoonhee/visual-cot-50k-poc` train, 5000 examples, seed=0
- Eval: same dataset's `eval` split, first 200 examples (after seeded shuffle, seed=0)
- **K = 4** latent slots (was 8 in D2/E)
- save_every: 0 (only final checkpoint persisted — disk-space discipline)
- Acceptance n_helpful threshold: ≥ 2 (50% of K=4); stretch goal ≥ 3 (75%)

## Outcome interpretations

- **G stacks** (utility ≥ +0.20 AND n_helpful ≥ 3/4): K=4 still has headroom
  with longer training; recipe is a clear winner for Phase 3.
- **G partially stacks** (utility > F by ≥ 0.02 but n_helpful stays at 2/4):
  longer training helps the existing positions but doesn't recruit new ones.
- **G doesn't stack** (utility ~ F): K=4 saturates by 1000 steps; further
  budget should target K, not steps.
- **G breaks geometry** (mean cos > 0.55): VICReg loses purchase under
  longer training at K=4 — recipe-specific issue.
