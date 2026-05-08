# Phase 2 — Faithful Monet Stage 2 scale-down

## Goal

Reproduce Monet stage 2's published recipe (paper: Monet — abstract visual
tokens for VLMs) at A6000-friendly scale, then run our compression-ablation
suite. This is the canonical "we KNOW this recipe produces distributed
latents at 7B" baseline. If it reproduces at 3B + 5K, we have a known-good
iteration baseline. If it doesn't, scale matters.

## Recipe (vs paper)

| knob | paper | Phase 2 (scale-down) |
|---|---|---|
| base | Stage-1 SFT'd 7B | raw Qwen2.5-VL-3B-Instruct (skip Stage 1 — deviation) |
| dataset | full 125K, 6 subsets | 5K Visual_CoT subset (seed=0, last 10K rows) |
| LATENT_SIZE | 8 | 8 |
| ce_emphasize_factor | 4.0 | 4.0 |
| alignment_weight | 2.0 | 2.0 |
| emphasize_latent_weight | 2.0 | (omitted — see deviations below) |
| epochs | 2 | ~3 (1000 optimizer steps over 5K with eff_batch=16) |
| eff batch | 8×16 = 128 | 1×16 = 16 (single A6000) |
| LR | 1e-5 | 1e-5 |
| warmup | 10 steps | 10 steps |
| optimizer | AdamW + DeepSpeed ZeRO-2 | AdamW (single-GPU) |
| dtype | bf16 | bf16 |
| alignment_layer | all_layers | all_layers |
| weight_decay | 0.01 | 0.01 |

## Loss

```
L = ce_emphasize_factor · L_CE_obs   (computed inside Monet model, weighted CE)
  + alignment_weight · L_align
```

Where:
- `L_CE_obs` = standard cross-entropy on the answer span; tokens at observation
  positions get 4.0× weight (the `ce_emphasize_poss` + `ce_emphasize_factor`
  mechanism in the Monet patched forward).
- `L_align` = `1 − cosine_similarity(student_hidden, teacher_hidden)` averaged
  over (num_layers, num_observation_tokens). Teacher = raw Qwen2.5-VL-3B-Instruct
  with the full trace + aux images visible. Student = same model after Phase 2
  fine-tune, with `latent_mode=True` (aux images NOT directly attended; the
  observation tokens get information about the aux image only via the K=8
  latent slots that the recurrent latent generation produced just before).

## Implementation strategy: PATH B

We use a custom trainer that reuses Monet's vendored model class via the
established `sys.modules` patch (Phase 0 / Phase 1.5 pattern). This avoids
DeepSpeed and TRL's SFTTrainer hard dependencies, enabling single-A6000
execution with no major code-path divergence — the `latent_mode=True` branch
of the model is exercised verbatim.

### Two-pass training step

Per upstream `src/trainer.py:CustomTrainerSFT_STAGE2.compute_loss`:

1. **Latent forward** (`latent_mode=True`):
   - Inputs: full trace (user image, question, assistant: aux image, latent
     slots, observation tokens, answer).
   - `alignment_poss = observation_token_positions`.
   - `teacher_hidden_states_for_alignment = cached_teacher_reps` (loaded from
     `teacher_reps/`).
   - `loss_type=['alignment']` so the wrapper populates `loss_dict['alignment']`.
   - Reads `outputs.loss_dict['alignment']` (the alignment loss) AND
     `outputs.ce_patch_pos` / `ce_patch_vec` (the produced latent vectors).
   - `gradient_checkpointing_disable()` (latent forward uses `use_cache=True`).

2. **CE forward** (`latent_mode=False`):
   - Re-feeds the same trace; `ce_patch_pos`/`ce_patch_vec` splice the latent
     vectors into slot positions.
   - `ce_emphasize_poss = observation_positions`, `ce_emphasize_factor = 4.0`.
   - `loss_type=['ce']` + labels (only answer span scored; special tokens
     `<abs_vis_token_pad>`, `<observation>` etc. ignored).
   - `gradient_checkpointing_enable(use_reentrant=False)`.

3. **Combine**: `total = ce_loss + alignment_weight · align_loss`.

## Deviations from paper, with rationale

1. **Skip Stage 1 SFT**. Use raw Qwen2.5-VL-3B-Instruct directly. Rationale:
   matches Phase 1 / Phase 1.5 starting point (clean comparison). Note: Stage 1
   SFT trains the model to emit `<observation>...</observation>` tags fluently;
   without it the model may generate `<observation>` predictions less precisely
   in the CE forward — but the alignment loss is the load-bearing mechanism.
2. **Single-GPU instead of 8-GPU**. eff_batch=16 vs 128 → 8× fewer effective
   gradient updates per epoch; we compensate with more steps (1000 vs 4000).
3. **5K subset of Visual_CoT only**. Paper uses full 125K across 6 subsets.
   Visual_CoT is the closest to Phase 1's data; restricting to it isolates the
   recipe effect rather than data-mix effect.
4. **emphasize_latent_weight latent-only backprop trick: omitted.** The paper
   uses `compute_latents_only_loss` to backprop alignment through latents
   only. We approximate by adding alignment_weight × align_loss directly,
   which backprops through everything. This is a slightly stronger signal
   to the LLM trunk; in practice the alignment loss is small relative to CE
   so the difference should be modest.
5. **No `attention_mask_4d`**. Upstream uses `build_4d_attn` to enforce
   attention-mask topology (latents see only their own image; observation
   blocks have isolated self-attention; etc.). We use default causal masking;
   the recurrent latent generation pattern is what differs from Phase 1.

## Data

- Source: `phase0_monet_probe/data/Monet-SFT-125K/Visual_CoT/train.json`
  (already on disk from Phase 0 download; 118,561 rows).
- Subset: last 10,000 rows shuffled with seed=0; first 5,000 valid (after
  `Monet_single_input_images_preprocess_function`) accepted.
- Held-out for ablation: re-use Phase 1's `ohjoonhee/visual-cot-50k-poc` `eval`
  split (200 examples, seed=0) for a clean comparison with Phase 1 / Phase 1.5.
  Note: this differs from Phase 0's held-out which used Visual_CoT's own end
  rows; the trade-off is data-source consistency with Phase 1.5.

## Acceptance

Same 4 criteria as Phase 1.

## Files

- `precompute_teacher_reps.py` — produces `teacher_reps/rep_all_layers_*.pt`
- `trainer.py` — Phase 2 stage-2 training loop
- `extract_latents.py` — held-out latent extraction
- `ablation_runner.py` — compression ablation
- `build_report.py` — assembles `REPORT.md`
- `configs/p2_3b_5k.yaml` — canonical run
- `configs/p2_smoke.yaml` — 10-step smoke

## Reproduce

```bash
PY=phase0_monet_probe/.venv-monet/bin/python

# 1. Precompute teacher reps (~30 min for 5K samples)
$PY phase2_monet_stage2/precompute_teacher_reps.py \
    --out_dir phase2_monet_stage2/teacher_reps --n 5000

# 2. Smoke (~3 min)
$PY phase2_monet_stage2/trainer.py --config phase2_monet_stage2/configs/p2_smoke.yaml --smoke

# 3. Train (~3 hr)
$PY phase2_monet_stage2/trainer.py --config phase2_monet_stage2/configs/p2_3b_5k.yaml

# 4. Extract + ablate (~10 min)
$PY phase2_monet_stage2/extract_latents.py --ckpt phase2_monet_stage2/results/run_p2/checkpoint \
    --out phase2_monet_stage2/results/run_p2/latents
$PY phase2_monet_stage2/ablation_runner.py \
    --latents_dir phase2_monet_stage2/results/run_p2/latents \
    --self_ckpt phase2_monet_stage2/results/run_p2/checkpoint \
    --self_name phase2_self \
    --out_results phase2_monet_stage2/results/run_p2/ablation_results.jsonl \
    --out_hstats phase2_monet_stage2/results/run_p2/ablation_h_stats.jsonl

# 5. Build report
$PY phase2_monet_stage2/build_report.py
```
