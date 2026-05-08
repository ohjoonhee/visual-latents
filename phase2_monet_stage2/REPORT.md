# Phase 2 — Faithful Monet Stage 2 scale-down on Visual_CoT

## TL;DR

Phase 2 implements Monet's published Stage 2 recipe at A6000-friendly scale
(3B base, 5K Visual_CoT subset, 1000 steps, eff_batch=16) with the canonical
two-pass training: latent forward (recurrent latent generation) +
observation-position alignment to a teacher (raw Qwen2.5-VL-3B-Instruct on
the same trace with aux images visible), plus weighted CE on observation
tokens. On 200 held-out eval examples, the trained latents score
**1/4** acceptance criteria — verdict **FAIL**.

Headline: `compression_ratio=1.604` (target ≥ 0.4),
`mean_off_diag_cos=0.737` (target ≤ 0.55),
`n_helpful=0/8` (target ≥ 3),
qwen_base utility `-4.759` (target > 0).

## 7-column comparison

| metric | P1 run-1 | P1 run-2 | P1.5 | **P2** | Monet st2 (target) | Monet st3 (anti) | user overnight |
|---|---:|---:|---:|---:|---:|---:|---:|
| compression_ratio | 0.631 | 0.700 | 1.132 | 1.604 | 2.7 | 0.78 | ~0.03 |
| mean off-diag cos | 0.851 | 0.987 | 0.959 | 0.737 | 0.38 | 0.85 | (>0.9) |
| n_helpful (≥3) | 1 | 2 | 1 | 0 | 4 | 1 | ≤1 |
| utility (qwen_base) | 0.077 | 0.076 | 0.030 | -4.759 | +2.7 | +0.26 | −0.22 to −1.10 |
| h-cosine | 0.851 | 0.987 | 0.959 | 0.737 | 0.38 | 0.85 | n/a |
| v_roi cosine | 0.465 | 0.465 | n/a | 0.465 | n/a | n/a | n/a |
| utility (self reader) | 0.336 | n/a | n/a | -6.301 | +2.7 | +0.26 | n/a |

## Per-position single-keep curve (phase2_self reader)

```
pos      nll   margin  bar
  0     7.710   -4.068  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
  1     6.087   -2.445  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
  2     5.997   -2.355  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
  3     6.238   -2.596  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
  4     6.114   -2.472  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
  5     6.074   -2.432  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
  6     6.097   -2.455  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
  7     6.819   -3.177  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
```

## 8x8 cosine matrix

```
       p0  p1  p2  p3  p4  p5  p6  p7
 p0  1.00  0.61  0.74  0.60  0.63  0.55  0.56  0.53
 p1  0.61  1.00  0.67  0.74  0.63  0.63  0.58  0.59
 p2  0.74  0.67  1.00  0.77  0.84  0.75  0.76  0.71
 p3  0.60  0.74  0.77  1.00  0.84  0.87  0.80  0.81
 p4  0.63  0.63  0.84  0.84  1.00  0.89  0.90  0.85
 p5  0.55  0.63  0.75  0.87  0.89  1.00  0.93  0.93
 p6  0.56  0.58  0.76  0.80  0.90  0.93  1.00  0.94
 p7  0.53  0.59  0.71  0.81  0.85  0.93  0.94  1.00

```

mean off-diagonal cos = **0.737** (Phase 0 thresholds: <0.4 distributed ✓, >0.85 collapsed ✗)

## Loss trajectory

Training summary: final step=1000 ce=1.300 align=0.2101 elapsed=9949s.

Compare the LVR/MSE form (Phase 1.5) vs alignment-loss form (Phase 2) — the
key observable is whether either converges to a low loss while keeping
distributed h. See per-step data in `results/run_p2/training_log.jsonl`.

## Recipe deviations from paper

1. Skip Stage 1 SFT — use raw Qwen2.5-VL-3B-Instruct directly. (Documented in `RECIPE.md`.)
2. 5K Visual_CoT subset (vs 125K full 6-subset mix). Constrained by single-GPU
   budget; isolates recipe effect from data-mix effect.
3. eff_batch=16 (single A6000) vs 128 (8 GPUs). Compensated with extra steps.
4. emphasize_latent_weight backprop trick omitted; alignment loss flows through
   the full LLM trunk (not latent-only). Modest signal-strength deviation.
5. `attention_mask_4d` not used; rely on default causal mask + Monet's recurrence.
6. alignment_layer = all_layers (matches paper).

## Acceptance call

| criterion | result |
|---|---|
| compression_ratio (≥0.4) | PASS (1.604) |
| mean off-diag cos (≤0.55) | FAIL (0.737) |
| n_helpful (≥3) | FAIL (0) |
| qwen_base utility (>0) | FAIL (-4.759) |

**Overall: FAIL (1/4 criteria met).**

## Caveats

- The `<observation>` tokens are present in Visual_CoT's training data, but
  the held-out eval (`ohjoonhee/visual-cot-50k-poc`) does NOT have observation
  tags — the eval prompts use a simpler `<|image_pad|>×K` slot pattern. Phase 2's
  utility on the held-out is therefore measured on a different distribution
  than the training (matches Phase 0's caveat about cross-distribution eval).
- The teacher reps were precomputed from raw 3B Qwen, not from a Stage-1-SFT'd
  checkpoint. The paper uses Stage-1-SFT teacher reps; this is a deviation.
