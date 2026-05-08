# Phase 1 — LVR-faithful repro on Visual-CoT

## TL;DR

Phase 1 trained a Qwen2.5-VL-3B-Instruct generator with the LVR loss
(L_NTP + λ_LVR · MSE(h, v_at_I); λ=1.0, K=8) on a 5K subset of
`ohjoonhee/visual-cot-50k-poc` for 1000 optimizer steps (effective batch 4).
On 200 held-out `eval` examples, the trained latents score **2/4**
acceptance criteria — verdict **MARGINAL**.

Headline: `compression_ratio=0.63` (target ≥ 0.4),
`mean_off_diag_cos=0.85` (target ≤ 0.55),
`n_helpful=1/8` (target ≥ 3),
frozen-Qwen utility `+0.077` (target > 0).

## Headline table

| metric | Phase1 (self) | Phase1 (qwen_base) | Monet stage2 (Phase 0 target) | Monet stage3 (Phase 0 anti-target) | user (overnight 2026-05-04) |
|---|---:|---:|---:|---:|---:|
| compression_ratio | 0.631 | 1.418 | 2.7 | 0.78 | ~0.03 |
| mean off-diag cos | 0.851 | (same) | 0.38 | 0.85 | (>0.9) |
| n_helpful (≥3) | 1 | 2 | 4 | 1 | ≤1 |
| utility (none − all) | 0.336 | 0.077 | +2.7 | +0.26 | −0.22 to −1.10 |

## Per-position single-keep curve (phase1_self reader)

```
pos      nll   margin  bar
  0     2.911   +0.047  ██  
  1     2.918   +0.039  ██  
  2     2.910   +0.048  ██  
  3     2.938   +0.020  █  
  4     2.931   +0.027  █  
  5     2.924   +0.034  ██  
  6     2.922   +0.036  ██  
  7     2.903   +0.054  ███  (helpful)
```
(margin = none_NLL − only_pos_i_NLL; positive = helpful in isolation)

## Pairwise-cosine matrix (8×8, mean over eval set)

```
       p0  p1  p2  p3  p4  p5  p6  p7
 p0  1.00  0.95  0.94  0.93  0.92  0.91  0.89  0.52
 p1  0.95  1.00  0.99  0.98  0.96  0.95  0.93  0.52
 p2  0.94  0.99  1.00  0.99  0.98  0.97  0.94  0.53
 p3  0.93  0.98  0.99  1.00  0.99  0.98  0.95  0.53
 p4  0.92  0.96  0.98  0.99  1.00  0.99  0.96  0.53
 p5  0.91  0.95  0.97  0.98  0.99  1.00  0.98  0.53
 p6  0.89  0.93  0.94  0.95  0.96  0.98  1.00  0.58
 p7  0.52  0.52  0.53  0.53  0.53  0.53  0.58  1.00

```

mean off-diagonal cos = **0.851** (Phase 0 thresholds: <0.4 distributed ✓, >0.85 collapsed ✗)

## Recipe summary

Trained with paper-faithful loss `L = L_NTP + λ · MSE(h, v_at_I)`, λ=1.0, K=8.
Vision tower + projector frozen; LLM full fine-tune. AdamW lr=1e-5, warmup
100 steps, cosine to zero, eff bsz=4 (bsz=1 × grad_accum=4), bf16, 1000 steps
on 5K examples (~0.8 epoch). One implementation deviation from paper: MSE is
mean over (B, K, D) rather than (1/T_v) · Σ over D — see RECIPE.md.

Training summary: final step=1000 ntp=0.570 lvr=4.876 ||h||=47.6 ||v||=95.9 elapsed=1341s.

## Acceptance call

| criterion | result |
|---|---|
| compression_ratio ≥ 0.4 | PASS (0.631) |
| mean off-diag cos ≤ 0.55 | FAIL (0.851) |
| n_helpful ≥ 3 | FAIL (1) |
| qwen_base utility > 0 | PASS (+0.077) |

**Overall: MARGINAL (2/4 criteria met).**

## Implication for Phase 2

If Phase 1 PASSES (≥3/4): the LVR loss at our scale produces distributed,
encoder-grounded latents — Phase 2 (Monet 3-stage) should proceed to test
whether stage-3 self-distillation degrades these (Phase 0 stage-3 result
suggests it will).

If Phase 1 FAILS: scale is implicated; flag for cluster-scale escalation
(Phase 3) before further per-position-target experiments. The Phase 0
stage-2 result ALREADY shows that distributed latents are achievable in
principle; Phase 1 is the local-scale floor for whether that mechanism
transfers below 7B.
