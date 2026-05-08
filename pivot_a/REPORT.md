# Pivot A — LVR + collapse-prevention regularizer (Experiments C & D)

## TL;DR

Pivot A tests regularizers added to the Phase 1 LVR loss to break
many-to-one collapse of the K=8 latent slots:
- **C1** pairwise cosine hinge (τ=0.5)
- **C2** VICReg variance + covariance on z-scored hidden states
- **D1** (Exp D follow-up) VICReg + paper-faithful sum-MSE LVR (~D=2048× stronger LVR pressure)
- **D2** (Exp D follow-up) VICReg with λ_reg=2.0 (double regularization weight)

Each variant is full FT of Qwen2.5-VL-3B-Instruct on 5K Visual-CoT for
1000 steps (eff bsz=4, lr=1e-5, λ_LVR=1.0, seed=0). All other variables
(data subset, K, schedule, prompt, ROI selection, eval split) are
identical to Phase 1's `lvr_3b_5k.yaml`.

- Verdict (Experiment C): **C1 cos-penalty** = MARGINAL (2/4); **C2 VICReg** = PASS (3/4).
- Verdict (Experiment D follow-up): **D1 VICReg+sumMSE** = MARGINAL (2/4); **D2 VICReg λ_reg=2** = PASS (3/4).
- Mean off-diag cos: C1=0.725, C2=0.441, D1=0.987, D2=0.341 (target ≤ 0.55).
- C1 reached target geometry (≤ 0.55)? **NO**.
- C2 reached target geometry (≤ 0.55)? **YES**.
- D1 reached target geometry (≤ 0.55)? **NO**.
- D2 reached target geometry (≤ 0.55)? **YES**.
- Pivot A hypothesis (any variant ≤ 0.55) is **SUPPORTED** at 3B/5K.

## 7-column comparison

| metric | Phase 1 run-1 | Phase 1.5b | C1 cos | C2 VICReg | **D1 VICReg+sumMSE** | **D2 VICReg λ_reg=2** | Monet stage 2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| compression_ratio (≥ 0.4) | 0.631 | 1.132 | 0.713 | 0.672 | **0.787** | **0.792** | 2.7 |
| **mean off-diag cos (≤ 0.55)** | 0.851 | 0.961 | 0.725 | 0.441 | **0.987** | **0.341** | 0.38 |
| n_helpful (≥ 3) | 1 | n/a | 2 | 1 | **2** | **2** | 4 |
| qwen_base utility (> 0) | +0.077 | +0.030 | 0.051 | 0.044 | **0.078** | **0.050** | +2.7 |
| utility (self reader) | +0.336 | +0.160 | 0.376 | 0.340 | **0.052** | **0.366** | +2.7 |
| v_roi off-diag cos | 0.465 | 0.465 | 0.465 | 0.465 | **0.465** | **0.465** | n/a |

## Acceptance per variant

### C1 — pairwise cosine hinge (τ=0.5, λ_reg=1.0) — MARGINAL (2/4)

| criterion | result |
|---|---|
| compression_ratio ≥ 0.4 | PASS (0.713) |
| mean off-diag cos ≤ 0.55 | FAIL (0.725) |
| n_helpful ≥ 3 | FAIL (2) |
| qwen_base utility > 0 | PASS (0.051) |


### C2 — VICReg (var=1.0, cov=0.04, γ=1.0, λ_reg=1.0) — PASS (3/4)

| criterion | result |
|---|---|
| compression_ratio ≥ 0.4 | PASS (0.672) |
| mean off-diag cos ≤ 0.55 | PASS (0.441) |
| n_helpful ≥ 3 | FAIL (1) |
| qwen_base utility > 0 | PASS (0.044) |


### D1 — VICReg + sum-MSE LVR (var=1.0, cov=0.04, γ=1.0, λ_reg=1.0, loss_form=sum_d) — MARGINAL (2/4)

| criterion | result |
|---|---|
| compression_ratio ≥ 0.4 | PASS (0.787) |
| mean off-diag cos ≤ 0.55 | FAIL (0.987) |
| n_helpful ≥ 3 | FAIL (2) |
| qwen_base utility > 0 | PASS (0.078) |


### D2 — VICReg λ_reg=2.0 (var=1.0, cov=0.04, γ=1.0, mean-MSE LVR) — PASS (3/4)

| criterion | result |
|---|---|
| compression_ratio ≥ 0.4 | PASS (0.792) |
| mean off-diag cos ≤ 0.55 | PASS (0.341) |
| n_helpful ≥ 3 | FAIL (2) |
| qwen_base utility > 0 | PASS (0.050) |


## C1 — per-position single-keep curve (pivot_a_cos_self reader)

```
pos      nll   margin  bar
  0     2.941   +0.057  █████  (helpful)
  1     2.952   +0.046  ████  
  2     2.950   +0.047  ████  
  3     2.987   +0.011  █  
  4     2.982   +0.016  █  
  5     2.965   +0.033  ███  
  6     2.967   +0.031  ██  
  7     2.906   +0.092  ███████  (helpful)
```

## C1 — 8×8 cosine matrix (mean over eval set)

```
       p0  p1  p2  p3  p4  p5  p6  p7
 p0  1.00  0.81  0.77  0.75  0.72  0.71  0.72  0.45
 p1  0.81  1.00  0.93  0.82  0.75  0.76  0.76  0.43
 p2  0.77  0.93  1.00  0.93  0.83  0.78  0.77  0.44
 p3  0.75  0.82  0.93  1.00  0.95  0.85  0.80  0.44
 p4  0.72  0.75  0.83  0.95  1.00  0.94  0.83  0.43
 p5  0.71  0.76  0.78  0.85  0.94  1.00  0.90  0.45
 p6  0.72  0.76  0.77  0.80  0.83  0.90  1.00  0.56
 p7  0.45  0.43  0.44  0.44  0.43  0.45  0.56  1.00

```

mean off-diagonal cos = **0.725**

Training summary: final step=1000 ntp=0.562 lvr=4.959 reg=0.100 ||h||=45.4 ||v||=95.9 elapsed=1340s


## C2 — per-position single-keep curve (pivot_a_vicreg_self reader)

```
pos      nll   margin  bar
  0     3.094   +0.022  ██  
  1     3.098   +0.018  █  
  2     3.074   +0.042  ███  
  3     3.092   +0.024  ██  
  4     3.094   +0.022  ██  
  5     3.115   +0.001    
  6     3.075   +0.041  ███  
  7     3.025   +0.091  ███████  (helpful)
```

## C2 — 8×8 cosine matrix (mean over eval set)

```
       p0  p1  p2  p3  p4  p5  p6  p7
 p0  1.00  0.44  0.45  0.42  0.43  0.44  0.42  0.42
 p1  0.44  1.00  0.47  0.45  0.46  0.46  0.45  0.42
 p2  0.45  0.47  1.00  0.45  0.46  0.47  0.45  0.44
 p3  0.42  0.45  0.45  1.00  0.44  0.45  0.43  0.42
 p4  0.43  0.46  0.46  0.44  1.00  0.46  0.43  0.43
 p5  0.44  0.46  0.47  0.45  0.46  1.00  0.44  0.43
 p6  0.42  0.45  0.45  0.43  0.43  0.44  1.00  0.42
 p7  0.42  0.42  0.44  0.42  0.43  0.43  0.42  1.00

```

mean off-diagonal cos = **0.441**

Training summary: final step=1000 ntp=0.651 lvr=5.417 reg=11.901 ||h||=52.2 ||v||=95.9 elapsed=1346s


## D1 — per-position single-keep curve (pivot_a_vicreg_summse_self reader)

```
pos      nll   margin  bar
  0     4.483   +0.011  █  
  1     4.420   +0.074  ██████  (helpful)
  2     4.409   +0.085  ███████  (helpful)
  3     4.481   +0.013  █  
  4     4.488   +0.006    
  5     4.481   +0.013  █  
  6     4.468   +0.026  ██  
  7     4.479   +0.015  █  
```

## D1 — 8×8 cosine matrix (mean over eval set)

```
       p0  p1  p2  p3  p4  p5  p6  p7
 p0  1.00  0.98  0.98  0.97  0.97  0.97  0.97  0.97
 p1  0.98  1.00  0.99  0.99  0.99  0.98  0.98  0.98
 p2  0.98  0.99  1.00  1.00  0.99  0.99  0.99  0.99
 p3  0.97  0.99  1.00  1.00  1.00  0.99  0.99  0.99
 p4  0.97  0.99  0.99  1.00  1.00  1.00  1.00  0.99
 p5  0.97  0.98  0.99  0.99  1.00  1.00  1.00  1.00
 p6  0.97  0.98  0.99  0.99  1.00  1.00  1.00  1.00
 p7  0.97  0.98  0.99  0.99  0.99  1.00  1.00  1.00

```

mean off-diagonal cos = **0.987**

Training summary: final step=1000 ntp=3.206 lvr=8891.445 reg=23.509 ||h||=51.0 ||v||=95.9 elapsed=1352s


## D2 — per-position single-keep curve (pivot_a_vicreg_lambda2_self reader)

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

## D2 — 8×8 cosine matrix (mean over eval set)

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


## Recipe

- Base: Qwen/Qwen2.5-VL-3B-Instruct (vision tower + projector frozen, LLM full FT)
- Loss: `L = L_NTP + λ_LVR · LVR + λ_reg · L_reg`
  - C1, C2, D2: `LVR = F.mse_loss(h, v_roi)` (mean-MSE, scaled by D)
  - D1:         `LVR = (1/T_v) · Σ_t ||h_t − v_t||²₂` (sum-over-D, paper-faithful)
- L_reg:
  - **C1** = `cos_penalty_loss(h, tau=0.5)` — squared hinge on `(C[i,j] − τ).clamp(min=0)`
  - **C2/D1/D2** = `vicreg_loss(h, var_weight=1.0, cov_weight=0.04, gamma=1.0)`
    on z-scored h (per-dim across BK sample axis)
- λ_reg: 1.0 for C1/C2/D1; 2.0 for D2.
- Optimizer: AdamW lr=1e-5, weight_decay=0, betas=(0.9, 0.95)
- Schedule: cosine, warmup=100, decay→0
- Batch: micro=1, grad_accum=4 → eff=4
- Max steps: 1000
- bf16 + gradient_checkpointing
- Data: `ohjoonhee/visual-cot-50k-poc` train, 5000 examples, seed=0
- Eval: same dataset's `eval` split, first 200 examples (after seeded shuffle, seed=0)
- K = 8 latent slots, prompt slot = `<|image_pad|>` × K (Phase-1-style)

## Decision

- mean cos ≤ 0.55 (any variant) → Pivot A's hypothesis confirmed at 3B/5K → escalate to cluster Phase 3 with this regularizer.
- both Exp-C variants > 0.7 → Pivot A at 3B/5K does not work → fall back to Experiment B (full Monet attention topology) or escalate scale.
- one in [0.55, 0.7] → MARGINAL: improved geometry but not target; document and escalate.

### What follow-up D was probing

- **D1** (sum-MSE LVR): can VICReg hold geometry against a ~D=2048× stronger
  LVR pull? If yes AND n_helpful or utility improves, the semantic gap (LLM
  knows where but not what) closes.
- **D2** (λ_reg=2.0): dose-response on the regularizer. Does mean cos drop
  closer to Monet stage 2's 0.38 with stronger reg?
