# Phase 0 supplement — student-path latents

Same setup as Phase 0 but latents are extracted via the **aux-image-free**
student path: the `<|vision_start|><|image_pad|><|vision_end|>` block before
the `<abs_vis_token>` slots is dropped, so the 8 positions evolve as a
Coconut-style hidden-state recurrence seeded by question + dialog history.
Same seed and held-out window as Phase 0, same patched forward
(`latent_mode=True`, `output_latent_embeds=True`); Phase 0's `ablation.py`
runs unchanged on the resulting latents.

## TL;DR

**Student-path latents are nearly identical to teacher-path latents and
exhibit the same redundancy pattern.** Stage 3 student collapses just like
the teacher: mean off-diag cos 0.89 (vs 0.87), utility +0.26 (= 0.26),
positions 4–7 collinear. Stage 2 student stays uncollapsed (cos 0.38 = 0.38)
and useful (utility +2.57 / +2.23 vs teacher +2.71 / +2.19). Student is
marginally more redundant on stage 3 (∆cos +0.018) and marginally less
helpful on stage 2 (∆util −0.14 nat) — neither flips the diagnosis.

## Side-by-side metrics (teacher → student)

| stage | subset | reader | comp_ratio T→S | n_helpful T→S | utility T→S | mean_off_diag_cos T→S |
|---|---|---|---:|---:|---:|---:|
| stage2 | Visual_CoT | monet_self | 2.70 → 2.93 | 4 → 3 | +2.71 → +2.57 | 0.375 → 0.384 |
| stage2 | Visual_CoT | qwen_base  | 5.97 → 5.95 | 8 → 8 | +2.19 → +2.23 | 0.375 → 0.384 |
| stage2 | Zebra_CoT_count | monet_self | 4.77 → 4.09 | 2 → 1 | +0.19 → +0.22 | 0.394 → 0.372 |
| stage2 | Zebra_CoT_count | qwen_base  | 3.87 → 6.70 | 0 → 1 | +0.01 → +0.07 | 0.394 → 0.372 |
| stage3 | Visual_CoT | monet_self | 0.78 → 0.79 | 1 → 1 | +0.26 → +0.26 | 0.867 → 0.885 |
| stage3 | Visual_CoT | qwen_base  | −0.05 → +0.02 | 7 → 7 | −0.13 → −0.14 | 0.867 → 0.885 |
| stage3 | Zebra_CoT_count | monet_self | 1.71 → −0.60 | 0 → 0 | −0.001 → −0.006 | 0.840 → 0.843 |
| stage3 | Zebra_CoT_count | qwen_base  | −3.74 → +0.83 | 0 → 0 | −0.018 → −0.015 | 0.840 → 0.843 |

`comp_ratio` is unstable on Zebra×stage3 (both num/den within noise of 0 in
both paths). On cells with signal, agreement is tight.

## Stage 3 × Visual_CoT pairwise cosine — both paths

```
            TEACHER                              STUDENT
       p0   p1   p2   p3   p4   p5   p6   p7      p0   p1   p2   p3   p4   p5   p6   p7
 p0  1.00 0.83 0.68 0.63 0.61 0.60 0.60 0.59    1.00 0.84 0.72 0.67 0.65 0.64 0.64 0.63
 p1  0.83 1.00 0.94 0.87 0.84 0.83 0.82 0.81    0.84 1.00 0.95 0.90 0.87 0.86 0.85 0.85
 p2  0.68 0.94 1.00 0.98 0.96 0.94 0.93 0.92    0.72 0.95 1.00 0.98 0.97 0.95 0.95 0.94
 p3  0.63 0.87 0.98 1.00 0.99 0.98 0.97 0.97    0.67 0.90 0.98 1.00 0.99 0.99 0.98 0.97
 p4  0.61 0.84 0.96 0.99 1.00 1.00 0.99 0.99    0.65 0.87 0.97 0.99 1.00 1.00 0.99 0.99
 p5  0.60 0.83 0.94 0.98 1.00 1.00 1.00 1.00    0.64 0.86 0.95 0.99 1.00 1.00 1.00 1.00
 p6  0.60 0.82 0.93 0.97 0.99 1.00 1.00 1.00    0.64 0.85 0.95 0.98 0.99 1.00 1.00 1.00
 p7  0.59 0.81 0.92 0.97 0.99 1.00 1.00 1.00    0.63 0.85 0.94 0.97 0.99 1.00 1.00 1.00
```

Same shape; student is uniformly +0.01 to +0.04 higher off-diag. Collapse is
**not** a teacher-pathway artefact — it survives when the model must produce
latents from question context alone.

## Implication for Phase 1

This **sharpens the case against stage-3 self-distillation** without
changing the stage-2 recommendation. Phase 0 left open whether the
collapse was an extraction-pathway artefact; the student forward inherits
the same positions-4–7 collinearity and same near-zero utility, so the
failure is a property of the **stage-3 objective**, not of eval-time
sampling. Recommendation stands: **import stage-2-style training**
(per-position teacher signal aligned to encoder-grounded aux-image
features); **skip stage-3-style distill-from-latent-teacher** in v0. Stage
2's student path also retains encoder-grounded behaviour under both
readers, so adapting the stage-2 alignment loss to a generator that runs
without aux images at inference is plausible — Monet itself demonstrates
the latents transport between the two prompt regimes.

## Caveats

- Same n=100/cell statistical limits as Phase 0; per-NLL CI ≈ ±0.05–0.10.
- Student path strips assistant-turn aux images; on Zebra_CoT_count the
  original prompt has many such images, so the student path drops a *lot*
  of context. Compare deltas within a cell, not absolute Zebra scales.
- We strip the assistant aux images but keep the user's question image.
  We can't disentangle "stage 3 collapses regardless of aux-image
  presence" from "stage 3 collapses whenever aux-image features are
  absent in the assistant turn." A mixed mode (aux images present in
  input, latents driven by preceding context) would test that — out of
  scope here.
- No retraining/LoRA; Phase 0's frozen-reader assumption reused.
