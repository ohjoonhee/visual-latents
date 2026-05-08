# Phase 1.5 — Recipe

## Goal

Isolate whether Monet's vendored modified transformer (latent_mode=True
recurrent latent generation pattern) is what produces distributed
encoder-grounded latents in Monet stage 2 — independent of Monet's
loss form. To do this, swap ONLY the architecture; keep Phase 1's data,
loss, optimizer, schedule, and seed.

## Backbone

- Class: `Qwen2_5_VLForConditionalGeneration` from
  `phase0_monet_probe/monet_model/modeling_qwen2_5_vl_monet.py` via the
  same `sys.modules` monkey-patch used in Phase 0.
- Weights: raw `Qwen/Qwen2.5-VL-3B-Instruct` (skip Monet Stage 1 SFT;
  matches Phase 1 starting point).
- Tokenizer extended with Monet's special tokens (`<abs_vis_token>`,
  `<abs_vis_token_pad>`, `</abs_vis_token>`, `<observation>`,
  `</observation>`); `model.resize_token_embeddings(len(tok))` initializes
  the new rows from the model's existing embedding initializer.

## Prompt format

```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\n<|vision_start|><image_pad>×N<|vision_end|>{question}<|im_end|>\n
<|im_start|>assistant\n<abs_vis_token><abs_vis_token_pad>×8</abs_vis_token>{answer}<|im_end|>
```

K=8 latent slots use Monet's `<abs_vis_token_pad>` token (id 151665) wrapped
in `<abs_vis_token>...</abs_vis_token>` markers — required by Monet's
forward to identify slot positions via `latent_token_id`.

## Training step (two-pass)

Following upstream `src/trainer.py:CustomTrainerSFT_STAGE2.compute_loss`:

1. **Latent forward** (`latent_mode=True`): `gradient_checkpointing_disable()`
   (uses `use_cache=True` internally). Monet's recurrent latent generation:
   each `<abs_vis_token_pad>` slot's input embedding = previous-token's hidden
   state; each slot's forward output is recorded.
   - Read `outputs.hidden_states[b]` (shape `[num_layers, K, D]`) — grad-enabled in train mode.
   - Take `[-1, :, :]` (last-layer) as h (shape `[K, D]`).
   - Note: `output_latent_embeds=True` returns a `.detach()`-ed copy that breaks
     gradient flow; we use `output_hidden_states=True` instead.
   - Also collect `outputs.ce_patch_pos` and `outputs.ce_patch_vec` for the next pass.

2. **CE forward** (`latent_mode=False`): `gradient_checkpointing_enable(use_reentrant=False)`.
   Pass `ce_patch_pos` / `ce_patch_vec` so the latent embeddings get spliced
   into slot positions; pass `loss_type=['ce']` + `labels` (only answer-span
   tokens scored, special tokens ignored).

3. **Loss**: `total = ntp_loss + λ · F.mse_loss(h, v_roi)`, λ=1.0 (mean over
   1×K×D — matches Phase 1 run 1 form).

## Loss

```
L = L_NTP + 1.0 · F.mse_loss(h, v_roi)
```

- `h` = last-layer hidden state at K=8 slot positions, from latent_mode=True
  forward (grad-enabled).
- `v_roi` = post-projector vision features at the K patch indices whose
  centers fall inside (or are nearest to the center of) the bbox.
- No projection head. Mean reduction over (1, K, D) — `F.mse_loss` semantics.

This matches Phase 1 RUN 1's loss form exactly (run-2's sum-MSE is NOT used —
the Phase 1 ablation showed run-2 was equally collapsed, so the variable
under test in Phase 1.5 is the architecture alone).

## Hyperparameters

| knob | value |
|---|---|
| K | 8 |
| λ_LVR | 1.0 |
| optimizer | AdamW betas=(0.9, 0.95), wd=0.0 |
| lr | 1e-5 |
| schedule | cosine, warmup_steps=100 |
| eff batch | 1×4 = 4 |
| max_steps | 1000 |
| dtype | bfloat16 |
| gradient_clip | 1.0 |
| min/max pixels | 3136 / 401408 |
| data | `ohjoonhee/visual-cot-50k-poc`, `train` split, 5000 examples (seed=0) |
| eval | 200 examples from `eval` split, seed=0 |

## Acceptance

Same 4 criteria as Phase 1:
- compression_ratio ≥ 0.4
- mean_off_diag_cos ≤ 0.55
- n_helpful ≥ 3 / 8
- qwen_base utility > 0

≥3/4 = PASS. 2/4 = MARGINAL. ≤1/4 = FAIL.

## Files

- `trainer.py` — training loop (Monet model, Phase 1 LVR loss).
- `extract_latents.py` — held-out latent extraction (`latent_mode=True`, `output_latent_embeds=True`).
- `ablation_runner.py` — compression ablation, two readers (self + qwen_base).
- `build_report.py` — assembles `REPORT.md` from results files.
- `configs/p15_3b_5k.yaml` — canonical run.
- `configs/p15_smoke.yaml` — 10-step smoke.

## Reproduce

```bash
PY=phase0_monet_probe/.venv-monet/bin/python

# 1. Smoke (10 steps, ~50s)
$PY phase1_5_attn/trainer.py --config phase1_5_attn/configs/p15_smoke.yaml --smoke

# 2. Train (1000 steps, ~85min)
$PY phase1_5_attn/trainer.py --config phase1_5_attn/configs/p15_3b_5k.yaml

# 3. Extract latents (200 examples, ~3min)
$PY phase1_5_attn/extract_latents.py \
    --ckpt phase1_5_attn/results/run_p15/checkpoint \
    --out phase1_5_attn/results/run_p15/latents

# 4. Ablation (~5min)
$PY phase1_5_attn/ablation_runner.py \
    --latents_dir phase1_5_attn/results/run_p15/latents \
    --self_ckpt phase1_5_attn/results/run_p15/checkpoint \
    --self_name phase1_5_self \
    --out_results phase1_5_attn/results/run_p15/ablation_results.jsonl \
    --out_hstats phase1_5_attn/results/run_p15/ablation_h_stats.jsonl

# 5. Build report
$PY phase1_5_attn/build_report.py
```
