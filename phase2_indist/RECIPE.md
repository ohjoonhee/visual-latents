# Phase 2 — In-distribution re-extraction RECIPE

## Goal

Disambiguate the Phase 2 OOD failure (`qwen_base utility = -4.76`,
`mean_off_diag_cos = 0.737`) by re-extracting latents from the same Phase 2
checkpoint on Monet's OWN Visual_CoT held-out distribution. If the cosine drops
and utility flips, the OOD failure was distribution mismatch; if both stay
similar, the collapse is genuine.

## Checkpoint

`phase2_monet_stage2/results/run_p2/checkpoint` — Phase 2's final Stage-2
recipe (Monet vendored class, 3B base, 5K Visual_CoT subset, 1000 steps).

## Data slice

- **Source**: `phase0_monet_probe/data/Monet-SFT-125K/Visual_CoT/train.json`
  (Monet's published Visual_CoT split, the same file Phase 2 trained on).
- **Pool**: last 10,000 rows (`raw[-10000:]`) — the same pool Phase 2's
  `precompute_teacher_reps.py` sampled from.
- **Excluded**: the 5000 sample_ids recorded in
  `phase2_monet_stage2/teacher_reps/manifest.json` (the trained subset).
  The exclusion is asserted in code (`extract_latents.py` line ~190).
- **Eligible**: 5000 rows in the `last 10K` pool that were NOT trained on.
- **Held-out**: shuffle eligible with `random.Random(42)`, then accept the
  first 200 valid through `Monet_single_input_images_preprocess_function(...,
  allow_no_observation=False)` (matching the Phase 2 trainer's preprocess
  call exactly).

## Confirmed exclusions

- Trained sample_ids are exactly the manifest entries (5000 entries).
- All 5000 trained sample_ids fall within `raw[-10000:]` (verified).
- Held-out selection asserts `sid not in trained_ids` per sample.
- Result of run: `eligible (after exclusion) = 5000`, `accepted = 200`,
  `skipped = 0` (every shuffled sample passed preprocess).

## Prompt format (matches Phase 2 trainer exactly)

```python
texts = [processor.apply_chat_template(example, tokenize=False)]
texts = [replace_latent_placeholder_with_img_pad(t) for t in texts]
texts = add_latent_pad_after_auxiliary_img(texts, K=8, "<abs_vis_token_pad>")
```

This is the same three-line block used in `phase2_monet_stage2/trainer.py`
lines 278-281 — `apply_chat_template` injects the system prompt and user/
assistant turns; `replace_latent_placeholder_with_img_pad` substitutes
auxiliary image placeholders for `<|image_pad|>` blocks; and
`add_latent_pad_after_auxiliary_img` appends K=8 `<abs_vis_token_pad>` tokens
after each `</abs_vis_token>` marker. The resulting prompts contain
`<observation>...</observation>` tags around step rationales (these are the
distinguishing feature vs the OOD eval distribution, which has none).

## Latent extraction

- Forward: `latent_mode=True`, `output_latent_embeds=True`,
  `loss_type=[]` (no labels needed; we only read latents).
- Alignment positions: `find_ids_poss(input_ids, "<|im_start|>assistant",
  "<abs_vis_token_pad>")[0]` — the K positions of `<abs_vis_token_pad>`
  that occur AFTER the first `<|im_start|>assistant` marker. This rule
  matches `phase0_monet_probe/extract_latents.py` and the upstream Monet
  `precompute_teacher_latents` path verbatim.
- Latent output: last-layer hidden states at those K positions
  (`outputs.latent_embeds[0]`, shape `[num_latents, H]`).
- Multi-aux-image traces produce `num_latents > 8`; the ablation uses
  `latent[:K=8]` (first auxiliary image's slots), matching Phase 2 OOD's
  ablation convention.

## Saved fields

`phase2_indist/latents/latent_<sid>.pt`:
  - `latent`         : `[num_latents, H]` bf16 — last-layer at <abs_vis_token_pad>
  - `v_roi`          : `[K, D]` bf16 | None — 8 spatially-distributed image
                       features from the FIRST image in the trace (auxiliary
                       or primary), to support `v_roi cosine` diagnostics
                       without re-extraction.
  - `question`       : str
  - `answer`         : str — last assistant turn's last text content,
                       `<observation>` and `<abs_vis_token>` tags scrubbed.
  - `sample_id`      : int — original `metadata.sample_id` from raw row.
  - `subset`         : "Visual_CoT"
  - `K`              : 8
  - `num_latents`    : int — actual count of `<abs_vis_token_pad>` slots.

## Ablation

`phase2_indist/ablation.py` is a thin wrapper around
`phase2_monet_stage2/ablation_runner.py` (the canonical Phase-2 runner used
for the OOD comparison) — same readers (`phase2_self`, `qwen_base`),
same modes (`all`, `none`, `first_half`, `last_half`, `only_pos_0..7`,
`first_only`, `last_only`), same K=8 splice into 8× `<|image_pad|>` slots
of a text-only prompt, same `loss_type=["ce"]` + label gating.

## Acceptance criteria (matched to Phase 2 OOD report)

| criterion | target | source |
|---|---|---|
| `compression_ratio` | ≥ 0.4 | self-reader; `(last_half - all)/(first_half - all)` |
| `mean off-diag cos` | ≤ 0.55 | h_stats; from the per-position pairwise cosine matrix |
| `n_helpful` | ≥ 3 | self-reader; positions where `only_pos_i NLL ≤ none NLL − 0.05` |
| `qwen_base utility` | > 0 | `none NLL − all NLL`, qwen_base reader |

Verdict: PASS = 3+/4, MARGINAL = 2/4, FAIL ≤ 1/4.

## Reproduce

```bash
PY=phase0_monet_probe/.venv-monet/bin/python

$PY phase2_indist/extract_latents.py \
    --ckpt phase2_monet_stage2/results/run_p2/checkpoint \
    --n 200 --out phase2_indist/latents

$PY phase2_indist/ablation.py
$PY phase2_indist/build_report.py
```

## Files

- `extract_latents.py` — held-out latent extractor.
- `ablation.py` — wrapper invoking `phase2_monet_stage2/ablation_runner.py`.
- `build_report.py` — assembles `REPORT.md`.
- `latents/latent_*.pt` — 200 extracted samples.
- `ablation_results.jsonl`, `ablation_h_stats.jsonl` — outputs.
- `extract_stdout.log`, `ablation_stdout.log` — full run logs.
