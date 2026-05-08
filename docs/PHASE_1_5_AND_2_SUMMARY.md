# Phase 1.5 + Phase 2 Combined Summary

## Headline

- **Phase 1.5**: MARGINAL (2/4 acceptance criteria)
- **Phase 2**: FAIL (1/4 acceptance criteria)

Both attempts to break the LVR-loss collapse via Monet's mechanisms fell short.
The architecture-only swap (Phase 1.5) gave essentially the same outcome as
plain Qwen2.5-VL-3B (Phase 1: same MARGINAL 2/4; same n_helpful=1; mean cos
worse: 0.959 vs 0.851 vs 0.987 across the three runs). Adding Monet's full
stage-2 recipe at scale-down (Phase 2) made the latents *actively harmful*
under both readers (utility −4.76 / −6.30) despite a lower mean cos = 0.737.

## 4×7 comparison (acceptance metrics × runs)

| metric | P1 run-1 | P1 run-2 | P1.5 | P2 | Monet st2 (target) | Monet st3 (anti) | user overnight |
|---|---:|---:|---:|---:|---:|---:|---:|
| compression_ratio | 0.631 | 0.700 | 1.132 | 1.604 | 2.7 | 0.78 | ~0.03 |
| mean off-diag cos | 0.851 | 0.987 | 0.959 | 0.737 | 0.38 | 0.85 | (>0.9) |
| n_helpful | 1 | 2 | 1 | 0 | 4 | 1 | ≤1 |
| utility (qwen_base) | 0.077 | 0.076 | 0.030 | -4.759 | +2.7 | +0.26 | −0.22 to −1.10 |

## What we learned

### About attention/architecture as the cause

**Architecture alone is NOT the load-bearing mechanism.** Phase 1.5 swapped only
the architecture (raw Qwen2.5-VL-3B → Monet's vendored modified transformer
with `latent_mode=True` recurrent latent generation), keeping Phase 1's LVR
loss (mean-MSE form, λ=1.0) and Visual-CoT data identical. Result: same
MARGINAL 2/4 verdict as Phase 1; mean cos = 0.959 (slightly *worse* than
Phase 1's 0.851); n_helpful = 1 (same); compression_ratio = 1.132 (better,
but only because all NLL values are noisy near baseline). The recurrent
latent generation pattern by itself does not break the collapse.

This contradicts the Phase 1 RUN-2 conclusion that "causal attention is
implicated structurally — Phase 2 is the next necessary test." The structural
alternative (Monet's recurrent attention) does not, by itself, fix the
problem. The collapse persists across (a) loss-form variation (λ × {1, 1/D};
ran-1 vs ran-2) and (b) architecture variation (standard transformer vs
Monet recurrence).

### About the aux-image training mechanism

**Adding Monet's full stage-2 alignment loss made things worse, not better,
at our scale.** Phase 2 added per-position alignment to teacher hidden states
at observation token positions (teacher = raw Qwen2.5-VL-3B with aux image
visible), plus weighted CE on observation tokens (ce_emphasize_factor=4.0).

Phase 2 result: `compression_ratio=1.604`, mean cos = `0.737`, n_helpful = 0,
qwen_base utility = `-4.759`, self utility = `-6.301` → **FAIL**.

Two failure modes are operating simultaneously:

1. **Latents are actively harmful**: every single keep-mode (all, first/last
   half, only_pos_i for all i) has *higher* NLL than `none`. The trained
   latent vectors carry information that, when fed to either reader,
   confuses rather than helps. This is qualitatively different from Phase 0
   stage 2's helpful pattern (utility +2.7 nat) and from Phase 1's mildly
   helpful pattern (utility +0.08 nat).
2. **Cosine collapse is partial, not full**: 0.737 sits between Monet stage 2
   (0.38) and Monet stage 3 (0.85). The latents are not yet in the
   single-direction collapse regime, but they are also not distributed enough
   to be useful.

The mismatch is most likely **distributional**: Phase 2 trained on Monet's
Visual_CoT data (which has aux images and `<observation>` tags) and is
evaluated on a different Visual_CoT POC split that has neither. The training
signal taught the model that "useful latent" = "observation-prediction
helper", but the eval prompts have no observation positions to leverage that
signal. Phase 1 / Phase 1.5 used the SAME data for train and eval.

### About scale (3B vs 7B)

Phase 0 showed Monet stage 2 at 7B produces distributed latents (mean cos
= 0.38, utility +2.7). Phase 2's 3B + 5K + 1000-step scale-down does NOT
reproduce that: mean cos = 0.737 is far from 0.38, and utility flips sign.
Either the 3B model lacks capacity for the recipe, the 5K dataset is too
small (1.4× fewer epochs of unique data than the 7B run sees in 2 epochs of
125K), or the scale-down composition (no Stage 1 SFT teacher, mixed up
the alignment-layer / emphasize-latent-weight choices) is materially
different from what produces the published result.

## Recommendation for next step

Diagnosis: **both phases fail**. By the protocol's recommendation matrix,
this means *the mechanism gap between Phase 0's 7B and our 3B scale-down
is fundamental*. Three plausible next directions:

1. **Cluster-scale 7B re-attempt of Phase 2** with full Monet recipe (Stage
   1 SFT teacher, all 6 subsets, 8-GPU eff_batch=128, 2 epochs). This is the
   minimum viable test of "does the published recipe reproduce when given
   the published budget?". Required compute: ~50 A100-hours.
2. **Pivot A** (per-position grounding without Stage 2's full alignment
   loss): use the LVR-style direct h↔v_roi MSE but with stronger
   regularization (e.g., entropy or contrastive losses on the K-position
   set) to break the collapse. Test at 3B + 5K, no aux-image training.
3. **Stop and write up the diagnostic finding**: at 3B + 5K, neither
   architecture nor training-recipe substitution achieves Monet-stage-2-like
   distributed latents. The compression_ratio metric in particular is
   noisy at this scale (P1: 0.63, P1.5: 1.13, P2: 1.60 — all driven by
   small denominators near the noise floor, not by genuine compression
   structure).

Recommended (given budget pressure on the cluster): **option 2 (Pivot A) at
3B first**, then if it produces distributed latents (mean cos < 0.55) at our
scale, escalate to 7B for full validation. Pivot A is the only path that
hasn't been tested.

## Phase reports

- `phase1_5_attn/REPORT.md`
- `phase1_5_attn/RECIPE.md`
- `phase2_monet_stage2/REPORT.md`
- `phase2_monet_stage2/RECIPE.md`

## Caveats (cross-phase)

- Both phases use raw Qwen2.5-VL-3B-Instruct (no Stage 1 SFT). Documented in
  each RECIPE.md. The published Monet stage 2 starts from a Stage-1-SFT'd 7B,
  not raw 3B.
- Phase 1.5 v_roi-cosine = 0.465 (Phase 1 reference: 0.465 — same data + ROI
  selection). Phase 2 v_roi-cosine = 0.465 (sanity-confirms ROI extraction
  is deterministic across runs).
- Held-out eval: 200 examples from `ohjoonhee/visual-cot-50k-poc` `eval`
  split. Phase 2 trained on Monet's `Visual_CoT/train.json` (a different but
  partially-overlapping data source). The TRAIN/EVAL distribution mismatch
  is unique to Phase 2 and may explain the harmful-utility failure mode.
- Phase 2 deviations from paper: skipped Stage 1 SFT, used 1×8 eff_batch
  instead of 8×16, 5K Visual_CoT only (vs 125K full mix), omitted the
  `emphasize_latent_weight` latent-only-backprop trick, no `attention_mask_4d`.
  Each documented in `phase2_monet_stage2/RECIPE.md`.
- Phase 1.5 RUNs the Monet vendored model class but does NOT pass
  `attention_mask_4d`. This means Phase 1.5 tests Monet's recurrence pattern
  alone, not the 4D attention mechanism. A future variant could enable that.
