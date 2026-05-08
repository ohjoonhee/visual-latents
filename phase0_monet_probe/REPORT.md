# Phase 0 — Monet latent compression probe

Held-out probe of NOVAglow646/Monet-SFT-7B stages 2/3 on Visual_CoT and
Zebra_CoT_count, n=100 per cell. Latents extracted via the upstream-canonical
path (`latent_mode=True`, `output_latent_embeds=True`, alignment positions =
`<abs_vis_token_pad>` slots after the first `<|im_start|>assistant` marker).
K=8 latent positions per auxiliary image; we evaluate on the first block
(positions 0..7) for matched-K comparison with the user's K_total=8 setup.

## TL;DR

**Stage 2 does NOT compress; stage 3 DOES, but as uniform redundancy rather
than last-token concentration.** Stage 2 latents (encoder-grounded) have
mean off-diag cosine 0.38, distinct positions, utility +2.7 nat
(Visual_CoT), and 2–4 individually-helpful positions. Stage 3 latents
(after distillation onto a latent-only forward) jump to mean cosine 0.85,
with positions 4–7 collinear (cos ≈ 1.00); utility collapses to +0.26
(Visual_CoT) and ≈0 (Zebra). The headline `compression_ratio` metric
(0.78–4.77 for monet_self) does not catch this — Stage 3's failure looks
"uniform" rather than "last-half-concentrated", which is _different_ from
the user's pattern but functionally equivalent (same info repeated in
every position).

Frozen Qwen2.5-VL base reader: stage 2 utility +2.19 (Visual_CoT) /
+0.01 (Zebra); stage 3 utility −0.13 / −0.02. Stage 3 reproduces the
user's transfer failure; stage 2 does not.

## Headline table

| stage | subset | reader | n | all_NLL | none_NLL | first_half | last_half | comp_ratio | n_helpful | utility | mean_cos |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage2 | Visual_CoT | monet_self | 100 | 3.115 | 5.824 | 4.071 | 5.696 | 2.701 | 4 | +2.709 | 0.375 |
| stage2 | Visual_CoT | qwen_base  | 100 | 3.615 | 5.806 | 3.961 | 5.677 | 5.972 | 8 | +2.191 | 0.375 |
| stage2 | Zebra_CoT_count | monet_self | 100 | 1.699 | 1.889 | 1.740 | 1.892 | 4.767 | 2 | +0.190 | 0.394 |
| stage2 | Zebra_CoT_count | qwen_base  | 100 | 1.909 | 1.920 | 1.918 | 1.945 | 3.872 | 0 | +0.011 | 0.394 |
| stage3 | Visual_CoT | monet_self | 100 | 4.904 | 5.161 | 5.209 | 5.143 | 0.783 | 1 | +0.257 | 0.867 |
| stage3 | Visual_CoT | qwen_base  | 100 | 5.941 | 5.806 | 5.789 | 5.949 |-0.052 | 7 | -0.135 | 0.867 |
| stage3 | Zebra_CoT_count | monet_self | 100 | 1.321 | 1.320 | 1.323 | 1.323 | 1.711 | 0 | -0.001 | 0.840 |
| stage3 | Zebra_CoT_count | qwen_base  | 100 | 1.938 | 1.920 | 1.942 | 1.922 |-3.741 | 0 | -0.018 | 0.840 |

- `comp_ratio = (last_half_NLL − all_NLL) / (first_half_NLL − all_NLL)`.
- `utility = none_NLL − all_NLL`; positive = latents help.
- `n_helpful` = positions i ∈ {0..7} where `only_pos_i` NLL ≥ 0.05 better than `none`.
- `mean_cos` = mean off-diagonal pairwise cosine across the 8 positions, averaged over the eval set (reader-independent).
- `n_helpful` for `qwen_base` reflects what the unadapted reader can pick up from individual positions; high counts there with negative `utility` indicate the latents flip from helpful to harmful when combined.

## Per-position single-keep curve — Visual_CoT × stage3 × monet_self

```
pos      nll   margin  bar
  0    5.072   +0.089  ████████████████████  (helpful)
  1    5.274   -0.113  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
  2    5.175   -0.014  ▒
  3    5.182   -0.021  ▒▒▒
  4    5.175   -0.014  ▒
  5    5.216   -0.055  ▒▒▒▒▒▒
  6    5.402   -0.241  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (harmful in isolation)
  7    5.306   -0.145  ▒▒▒▒▒▒▒▒▒▒▒▒▒
```

`margin = none_NLL − only_pos_i_NLL`. Only position 0 is individually
helpful — closely matches the user's "1 strong + 7 noise" pattern.

## Pairwise-cosine matrix — stage 3 × Visual_CoT × monet_self

```
       p0    p1    p2    p3    p4    p5    p6    p7
 p0  1.00  0.83  0.68  0.63  0.61  0.60  0.60  0.59
 p1  0.83  1.00  0.94  0.87  0.84  0.83  0.82  0.81
 p2  0.68  0.94  1.00  0.98  0.96  0.94  0.93  0.92
 p3  0.63  0.87  0.98  1.00  0.99  0.98  0.97  0.97
 p4  0.61  0.84  0.96  0.99  1.00  1.00  0.99  0.99
 p5  0.60  0.83  0.94  0.98  1.00  1.00  1.00  1.00
 p6  0.60  0.82  0.93  0.97  0.99  1.00  1.00  1.00
 p7  0.59  0.81  0.92  0.97  0.99  1.00  1.00  1.00
```

Positions 4–7 are essentially collinear (cos ≈ 1.00 between every pair).
Compare to **stage 2 × Visual_CoT** where the same matrix has
off-diagonal entries 0.11–0.92 and clear adjacent-block structure.

## Comparison to user's own results

`docs/overnight_2026_05_04/MORNING_REPORT.md` reports `compression_ratio`
0.01–0.05 on the user's K_q=3 multi-Q natural setup (reader-NLL only, 200
steps), n_helpful_positions ≤ 1, frozen-Qwen anchor utility −0.22 to
−1.10.

| metric | user (multi-Q nat 200 steps) | Monet stage2 | Monet stage3 |
|---|---|---|---|
| compression_ratio | ~0.03 | 2.7–4.8 | 0.78–1.7 |
| n_helpful (single-keep) | ≤ 1 | 2–4 | 0–1 |
| mean off-diag cosine | likely > 0.9 (not reported) | 0.38 | 0.85 |
| frozen-base utility | −0.22 to −1.10 | +2.19 / +0.01 | −0.13 / −0.02 |

The user's regime sits closer to Monet's stage 3 than to its stage 2.
Both regimes lack a per-position encoder-grounded target.

## Decision call

**Proceed to Phase 1 with a refined target.** The hypothesis "import
Monet's training scheme" is supported at stage 2 but refuted at stage 3.
The mechanism worth importing is stage 2's **per-position teacher signal
to encoder-grounded image features** (each latent slot aligned to a
different position in the teacher's last_hidden_state, sitting next to
a real auxiliary-image feature), not stage 3's "generate latents from
prior context, distill from a stage 2 teacher" path.

Phase 1 plan, refined:
1. Per-position grounding (the user's existing "Pivot A" from morning report).
2. Each per-position target should be **encoder-grounded** at training
   time — i.e., come from a real image feature, not a teacher latent of
   a teacher latent. Monet stage 2's training pattern is the precedent.
3. The 4D attention rules in upstream `build_4d_attn` (mask_latent +
   observation-block isolation) are part of the mechanism; they prevent
   later positions from cribbing earlier ones via attention.
4. **Skip** stage-3-style self-distillation in v0; it reproduces the
   collapse we are trying to avoid.

## Caveats

- n=100/cell is enough for cell ordering; per-NLL CI ≈ ±0.05–0.10 at this n.
- Both subsets are Monet's own training data (held out by row index from end).
  External-benchmark behaviour could differ.
- Stage 3 latents extracted via the canonical teacher path (with aux
  images present in input) — this is the same path stage 2 uses. We did
  NOT exercise the inference-time student-only forward (which would
  generate latents without aux images). That comparison is left to
  Phase 1.
- The Monet `modeling_qwen2_5_vl_monet.py` monkey-patch is process-global,
  so the "Qwen base reader" runs through the patched class too. We pass
  `loss_type=["ce"]` + `labels` to satisfy the patched gating; the splice
  mechanism is unchanged.
- The comp_ratio metric is directional and misses _uniform_ redundancy
  (stage 3). The pairwise-cosine matrix is the clearer diagnostic.

## Files

- `results.jsonl` — 112 rows, one per (mode, anchor, stage, subset).
- `h_stats.jsonl` — 4 rows, one per (stage, subset); per-pos norms + 8×8 cosine matrix.
- `extract_latents.py`, `ablation.py`, `build_report.py`, `monet_utils.py`,
  `monet_model/` (vendored Monet patch) — re-run via `bash run_phase0.sh`.
