# Phase 1.5b — Recipe

## Goal

Add the ONE missing variable from Phase 1.5: inject `attention_mask_4d`
on every forward, with the cross-slot-isolation rule (latent slot t_i
cannot attend to latent slot t_j for j != i; can still attend to its
own row and all non-slot causal context).

This isolates the hypothesis "cross-slot attention isolation is the
load-bearing mechanism Phase 1.5 was missing" from confounders. All
other variables — data, loss, seed, K, steps, eff_bsz — are identical
to Phase 1.5 (see `phase1_5_attn/RECIPE.md`).

## Source of `build_4d_attn`

Re-implemented from upstream Monet's `build_4d_attn_wo_helper_images`
(see `https://github.com/NOVAglow646/Monet`, file `src/utils.py`,
function at line 764), constrained to the Phase 1.5b data layout:

- single question image
- no helper images (no `<observation>` blocks; no per-step <I_i, A_i, O_i>)
- only one A_i (the K=8 latent slot block) per sequence

The upstream `mask_latent=True` flag hides slot positions from later
non-slot query rows. Our version adds `latent_cross_isolate=True` —
a stricter rule that blocks slot/slot off-diagonal pairs (matches the
Phase 1.5b user-stated hypothesis exactly).

Verification: derived from a careful read of the consumption pattern in
`phase0_monet_probe/monet_model/modeling_qwen2_5_vl_monet.py`:

- forward signature: line 1560 (`attention_mask_4d` accepts a
  `dict` with key `'full_attention'` mapping to a `[B, 1, T, T]` tensor).
- pre-answer slice: line 1775 (`[:, :, :ans_start, :ans_start]`).
- text-chunk slice: line 1831 (`[:, :, q0:q1, :k1]`).
- non-latent forward: line 2021 (`attention_mask_4d` is forwarded to
  `language_model` directly when present).
- additive vs bool: line 432 (`attn_weights + causal_mask`) — the
  vendored eager-attention adds the mask onto logits, so we must use
  the **additive** form (0 = allow, `torch.finfo(dtype).min` = block).

## What the mask looks like

For a probe prompt with K=4 slots at positions [32,33,34,35] and L=43,
see `MASK_CHECK.txt`. Slot rows 32..35 attend to all 32 prefix tokens,
attend to themselves, and are BLOCKED from attending to other slots.
All other rows are standard causal.

## Caveat: per-slot KV-cache forward bypass

The Monet vendored `latent_mode=True` path forwards each latent slot
ONE AT A TIME using `past_key_values`. The forward call at lines
1922-1934 of `modeling_qwen2_5_vl_monet.py` hard-codes
`attention_mask=attention_mask[b: b + 1][:, :pos+1]` — the 1D mask. The
4D mask is NOT consumed at this per-slot step.

This means the cross-slot-isolation rule is enforced at:

1. The pre-answer prefix forward (`pre_out`, line 1777) — uses
   `attn_mask_s['full_attention'][:, :, :ans_start, :ans_start]`.
2. Post-latent text-chunk forwards (`_run_text_chunk`, line 1825-1845)
   — uses `attn_mask_s['full_attention'][:, :, q0:q1, :k1]`.
3. The whole `latent_mode=False` CE forward (line 2018-2030).

But NOT at:

4. The per-slot recurrent forward (line 1922) — slot i's query attends
   to slot j's KV through past_kv, with all-ones causal default.

We chose NOT to modify the vendored model file (it's shared by Phase 0,
Phase 1.5, and Phase 2). If Phase 1.5b's mean off-diag cos remains
collapsed (~0.96) despite the partial isolation, that's evidence the
per-slot KV path is the dominant source of cross-slot leakage. A
follow-up would patch line 1922-1934 to slice
`attention_mask_4d['full_attention'][:, :, pos:pos+1, :pos+1]` and
pass that as the per-step `attention_mask` (a dict).

## Hyperparameters

Identical to Phase 1.5:

| knob | value |
|---|---|
| K | 8 |
| λ_LVR | 1.0 (mean-MSE; matches Phase 1 run-1) |
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
| **mask** | `attention_mask_4d` with `latent_cross_isolate=True`, additive bf16 |

## Acceptance

| criterion | target |
|---|---|
| compression_ratio | ≥ 0.4 |
| **mean off-diag cos** | **≤ 0.55** (the key metric) |
| n_helpful | ≥ 3 / 8 |
| qwen_base utility | > 0 |

≥3/4 = PASS. 2/4 = MARGINAL. ≤1/4 = FAIL.

## Files

- `mask_utils.py` — `build_4d_attn` + `build_monet_4d_attn` helper +
  `render_mask_grid` for the sanity check.
- `trainer.py` — clone of Phase 1.5 trainer with mask construction +
  injection at every forward.
- `extract_latents.py` — held-out latent extraction with mask injection.
- `ablation_runner.py` — same as Phase 1.5 (reader name = `phase1_5b_self`).
- `build_report.py` — assembles `REPORT.md`.
- `configs/p15b_3b_5k.yaml` — canonical run config.
- `MASK_CHECK.txt` — visualisation of the mask for a sample sequence.

## Reproduce

```bash
PY=phase0_monet_probe/.venv-monet/bin/python

# 0. Mask sanity check (writes MASK_CHECK.txt)
$PY -c "import sys; sys.path.insert(0, 'phase1_5b_attn'); exec(open('phase1_5b_attn/mask_utils.py').read())"

# 1. Smoke (~50 steps, ~3 min) — verifies no NaN with mask injected
$PY phase1_5b_attn/trainer.py --config phase1_5b_attn/configs/p15b_3b_5k.yaml --smoke --smoke_steps 50 \
    > phase1_5b_attn/results/run_p15b/smoke_stdout.log 2>&1

# 2. Full train (1000 steps, ~85 min)
$PY phase1_5b_attn/trainer.py --config phase1_5b_attn/configs/p15b_3b_5k.yaml \
    > phase1_5b_attn/results/run_p15b/train_stdout.log 2>&1

# 3. Extract latents (~3 min)
$PY phase1_5b_attn/extract_latents.py \
    --ckpt phase1_5b_attn/results/run_p15b/checkpoint \
    --out phase1_5b_attn/results/run_p15b/latents

# 4. Ablation (~5 min)
$PY phase1_5b_attn/ablation_runner.py \
    --latents_dir phase1_5b_attn/results/run_p15b/latents \
    --self_ckpt phase1_5b_attn/results/run_p15b/checkpoint \
    --self_name phase1_5b_self \
    --out_results phase1_5b_attn/results/run_p15b/ablation_results.jsonl \
    --out_hstats phase1_5b_attn/results/run_p15b/ablation_h_stats.jsonl

# 5. Build report
$PY phase1_5b_attn/build_report.py
```
