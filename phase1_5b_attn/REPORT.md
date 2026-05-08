# Phase 1.5b — Monet's vendored architecture + 4D attention mask injection

## TL;DR

Phase 1.5b adds the ONE missing variable from Phase 1.5: the
`attention_mask_4d` (cross-slot isolation: each latent slot t_i cannot
attend to other latent slots t_j, j != i, while still attending to all
non-slot causal context). All other variables — data, loss (mean-MSE
λ=1.0), seed, K=8, 1000 steps, eff_bsz=4 — are identical to Phase 1.5.

Verdict: **MARGINAL (2/4 criteria met).**

Headline: `compression_ratio=0.982` (target ≥ 0.4),
`mean_off_diag_cos=0.961` (target ≤ 0.55) — **the key metric**,
`n_helpful=2/8` (target ≥ 3),
frozen-Qwen utility `0.004` (target > 0).

## 4-column comparison table

| metric | Phase 1 run-1 | Phase 1.5 (no mask) | **Phase 1.5b (with mask)** | Monet stage 2 target |
|---|---:|---:|---:|---:|
| compression_ratio | 0.631 | 1.132 | 0.982 | 2.7 |
| mean off-diag cos | 0.851 | 0.959 | 0.961 | 0.38 |
| n_helpful (≥3) | 1 | 1 | 2 | 4 |
| utility (qwen_base) | 0.077 | 0.030 | 0.004 | +2.7 |
| utility (self reader) | 0.336 | 0.160 | 0.163 | +2.7 |
| v_roi off-diag cos | 0.465 | 0.465 | 0.465 | n/a |

## Per-position single-keep curve (phase1_5b_self reader)

```
pos      nll   margin  bar
  0     3.085   +0.052  ████  (helpful)
  1     3.086   +0.050  ████  (helpful)
  2     3.102   +0.034  ███  
  3     3.120   +0.017  █  
  4     3.128   +0.009  █  
  5     3.117   +0.020  ██  
  6     3.109   +0.028  ██  
  7     3.104   +0.032  ███  
```

## 8x8 cosine matrix

```
       p0  p1  p2  p3  p4  p5  p6  p7
 p0  1.00  0.96  0.93  0.92  0.92  0.92  0.92  0.91
 p1  0.96  1.00  0.98  0.97  0.96  0.95  0.95  0.94
 p2  0.93  0.98  1.00  0.99  0.98  0.97  0.96  0.95
 p3  0.92  0.97  0.99  1.00  0.99  0.98  0.97  0.96
 p4  0.92  0.96  0.98  0.99  1.00  0.99  0.98  0.97
 p5  0.92  0.95  0.97  0.98  0.99  1.00  0.99  0.98
 p6  0.92  0.95  0.96  0.97  0.98  0.99  1.00  0.99
 p7  0.91  0.94  0.95  0.96  0.97  0.98  0.99  1.00

```

mean off-diagonal cos = **0.961** (Phase 0 thresholds: <0.4 distributed PASS, >0.85 collapsed FAIL)

## Recipe

Identical to Phase 1.5 (see `phase1_5_attn/RECIPE.md`) except:
- A 4D attention mask is constructed at every forward via
  `mask_utils.build_monet_4d_attn(input_ids, latent_token_id=...,
  latent_cross_isolate=True)` and passed as `attention_mask_4d=...` to
  BOTH the latent_mode=True and latent_mode=False forwards.

Mask semantics (per the user's hypothesis):
- causal (lower-triangular) base
- AND latent slot t_i cannot attend to latent slot t_j for j != i
- (slot can still attend to its own row and all non-slot context)

Source of `build_4d_attn`: re-implemented from the upstream Monet
`build_4d_attn_wo_helper_images` (https://github.com/NOVAglow646/Monet,
`src/utils.py:764`), constrained to Phase 1.5b's data layout (single
question image, no helper images, no observation blocks) and extended
with the `latent_cross_isolate=True` rule. See `mask_utils.py` for full
docstring + line refs into the vendored model file.

Caveat (documented in RECIPE.md): the per-latent-slot recurrent forward
inside `modeling_qwen2_5_vl_monet.py` (line ~1922-1934) hard-codes the 1D
`attention_mask` argument and does NOT consume `attention_mask_4d`. So
the cross-slot-isolation rule applies to (a) the pre-answer prefix
forward, (b) post-latent text-chunk forwards, and (c) the entire
latent_mode=False CE forward — but NOT to the per-step KV-cache pass
that produces each slot's hidden state directly. We did NOT modify the
vendored model file. This means cross-slot isolation is partially
enforced; if Phase 1.5b still fails on mean off-diag cos, the per-step
KV path (which still allows slot-i to attend to slot-j via cache) is the
likely remaining cause.

Loss: `L = L_NTP + 1.0 · F.mse_loss(h, v_roi)` (mean over 1×K×D). Vision
tower + projector frozen; LLM full FT. AdamW lr=1e-5, warmup=100,
cosine→0, eff bsz=4 (bsz=1 × grad_accum=4), bf16, 1000 steps, 5K
examples, seed=0.

Training summary: final step=1000 ntp=0.447 lvr=4.988 ||h||=37.4 ||v||=95.9 elapsed=5046s.

## Acceptance call

| criterion | result |
|---|---|
| compression_ratio (≥0.4) | PASS (0.982) |
| mean off-diag cos (≤0.55) | FAIL (0.961) |
| n_helpful (≥3) | FAIL (2) |
| qwen_base utility (>0) | PASS (0.004) |

**Overall: MARGINAL (2/4 criteria met).**

## Implication

If Phase 1.5b PASSES on mean off-diag cos -> cross-slot attention
isolation is the load-bearing mechanism Phase 1.5 was missing.

If Phase 1.5b FAILS on mean off-diag cos with a value comparable to
Phase 1.5 (~0.96), the per-step KV-cache path (which the 4D mask cannot
reach without modifying the vendored model) is the prime suspect. A
follow-up could patch `modeling_qwen2_5_vl_monet.py` to slice
`attention_mask_4d` for the per-slot forward too.

## Caveats

- Same starting checkpoint as Phase 1.5 (raw Qwen2.5-VL-3B-Instruct).
- Same Monet special tokens added; embeddings randomly initialized via
  `resize_token_embeddings`.
- The per-slot KV-cache forward inside `latent_mode=True` still uses the
  1D causal mask (see RECIPE caveat above).
