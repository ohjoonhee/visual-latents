# Phase 2 — In-distribution re-extraction on Monet's Visual_CoT

## TL;DR

Re-extracted latents from the Phase 2 checkpoint
(`phase2_monet_stage2/results/run_p2/checkpoint`) on Monet's OWN Visual_CoT
held-out — same distribution Phase 0 used for stages 2/3 probes, with
`<observation>` tags + auxiliary images and the EXACT prompt construction
the Phase 2 trainer used. Held-out is 200 examples sampled (seed=42)
from `raw[-10000:]` of `Visual_CoT/train.json` MINUS the 5K trained sample_ids
recorded in `phase2_monet_stage2/teacher_reps/manifest.json` (zero overlap, asserted).

On 200 held-out in-distribution examples, the trained latents score
**0/4** acceptance criteria — verdict **FAIL** (NO FLIP).

Headline: `compression_ratio=-0.072` (target ≥ 0.4),
`mean_off_diag_cos=0.704` (target ≤ 0.55),
`n_helpful=0/8` (target ≥ 3),
qwen_base utility `-5.630` (target > 0).

## Headline table — Phase 2 OOD vs in-dist vs Monet stage 2/3 (Visual_CoT, qwen_base)

| metric | Phase 2 OOD (existing) | **Phase 2 in-dist (new)** | Monet st2 (target) | Monet st3 (anti-target) |
|---|---:|---:|---:|---:|
| compression_ratio (self) | 1.604 | **-0.072** | 5.972 | -0.052 |
| mean off-diag cos | 0.737 | **0.704** | 0.375 | 0.867 |
| n_helpful (self) | 0 | **0** | 4 | 1 |
| qwen_base utility | -4.759 | **-5.630** | 2.19 | -0.14 |
| self-reader utility | -6.301 | -7.042 | +2.71 | +0.26 |
| v_roi cosine | 0.465 | 0.329 | n/a | n/a |

Note: Monet stage 2 / stage 3 numbers are from `phase0_monet_probe/REPORT.md`,
qwen_base reader, Visual_CoT subset, n=100. Phase 2 uses qwen_base = Qwen2.5-VL-**3B**-Instruct
(matching the Phase 2 base) whereas Phase 0 used the 7B; absolute NLLs not directly
comparable, but the directional signs and pairwise-cosine geometry are.

## Per-position single-keep curve (phase2_self reader, in-dist)

```
pos      nll   margin  bar
  0     7.767   -5.442  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
  1     7.273   -4.947  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
  2     3.325   -1.000  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
  3     4.248   -1.923  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
  4     5.824   -3.499  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
  5     5.006   -2.681  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
  6     5.218   -2.893  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
  7     6.096   -3.771  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful)
```

`margin = none_NLL − only_pos_i_NLL`. Positive = position helpful in isolation.

## 8x8 cosine matrix (in-dist)

```
       p0  p1  p2  p3  p4  p5  p6  p7
 p0  1.00  0.84  0.47  0.64  0.53  0.49  0.47  0.46
 p1  0.84  1.00  0.57  0.86  0.80  0.74  0.71  0.69
 p2  0.47  0.57  1.00  0.58  0.52  0.49  0.46  0.43
 p3  0.64  0.86  0.58  1.00  0.87  0.86  0.82  0.81
 p4  0.53  0.80  0.52  0.87  1.00  0.92  0.91  0.89
 p5  0.49  0.74  0.49  0.86  0.92  1.00  0.95  0.93
 p6  0.47  0.71  0.46  0.82  0.91  0.95  1.00  0.97
 p7  0.46  0.69  0.43  0.81  0.89  0.93  0.97  1.00

```

mean off-diagonal cos = **0.704** (Phase 0 thresholds: <0.4 distributed ✓, >0.85 collapsed ✗)

OOD reference: mean off-diag cos = 0.737 (same checkpoint, different eval).

## Acceptance call

| criterion | result |
|---|---|
| compression_ratio (≥0.4) | FAIL (-0.072) |
| mean off-diag cos (≤0.55) | FAIL (0.704) |
| n_helpful (≥3) | FAIL (0) |
| qwen_base utility (>0) | FAIL (-5.630) |

**Overall: FAIL (0/4 criteria met).**

## Implication

The cosine and utility numbers did NOT both improve in-distribution. The Phase 2 collapse looks genuine, not just a distribution-mismatch artifact: the latents are degenerate even on the data they were trained on.

## Caveats

- n = 200 held-out examples (target 200; reduced if extraction failures).
- The qwen_base reader here is Qwen2.5-VL-3B-Instruct (Phase 2's base), not the
  7B used in Phase 0's reference numbers. Absolute NLLs differ by model size,
  but directional metrics (compression_ratio, n_helpful, utility sign) are valid.
- Despite excluding the 5K trained sample_ids, the held-out set is drawn from
  the same `raw[-10000:]` pool the trainer sampled from. Distribution match is
  thus tight; this is the desired property for the in-dist test, but it means
  the held-out is NOT independent of training in a generalisation sense.
- Per-position alignment positions are the K=8 `<abs_vis_token_pad>` slots
  AFTER the first `<|im_start|>assistant` marker — same rule as Phase 0
  extract and the Monet upstream `precompute_teacher_latents` path.
- Some Visual_CoT traces have multiple auxiliary images → `num_latents > 8`;
  the ablation reads only the FIRST K=8 positions (matching Phase 2 OOD and
  the canonical ablation's `latent[:K]` convention).
