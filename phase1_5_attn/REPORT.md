# Phase 1.5 — Monet's vendored architecture + Phase 1 LVR loss & data

## TL;DR

Phase 1.5 swaps Qwen2.5-VL-3B-Instruct's standard transformer for Monet's
vendored modified transformer (latent_mode=True recurrent latent generation +
optional 4D attention rules) — keeping Phase 1's LVR loss (mean-MSE form,
λ=1.0) and Visual-CoT data identical. On 200 held-out eval examples, the
trained latents score **2/4** acceptance criteria — verdict **MARGINAL**.

Headline: `compression_ratio=1.132` (target ≥ 0.4),
`mean_off_diag_cos=0.959` (target ≤ 0.55),
`n_helpful=1/8` (target ≥ 3),
frozen-Qwen utility `0.030` (target > 0).
v_roi cosine = 0.465 (Phase 1 reference: 0.465 — same data).

## 6+-column comparison table

| metric | Phase1 run-1 (mean-MSE) | Phase1 run-2 (sum-MSE) | **Phase 1.5** | Monet stage2 (target) | Monet stage3 (anti-target) | user overnight 2026-05-04 |
|---|---:|---:|---:|---:|---:|---:|
| compression_ratio | 0.631 | 0.700 | 1.132 | 2.7 | 0.78 | ~0.03 |
| mean off-diag cos | 0.851 | 0.987 | 0.959 | 0.38 | 0.85 | (>0.9) |
| n_helpful (≥3) | 1 | 2 | 1 | 4 | 1 | ≤1 |
| utility (qwen_base) | 0.077 | 0.076 | 0.030 | +2.7 | +0.26 | −0.22 to −1.10 |
| h-cosine | 0.851 | 0.987 | 0.959 | 0.38 | 0.85 | n/a |
| v_roi-cosine | 0.465 | 0.465 | 0.465 | n/a | n/a | n/a |
| utility (self reader) | 0.336 | n/a | 0.160 | +2.7 | +0.26 | n/a |

## Per-position single-keep curve (phase1_5_self reader)

```
pos      nll   margin  bar
  0     2.937   +0.050  ████  (helpful)
  1     2.960   +0.027  ██  
  2     2.960   +0.027  ██  
  3     2.981   +0.006    
  4     2.978   +0.009  █  
  5     2.970   +0.017  █  
  6     2.965   +0.022  ██  
  7     2.968   +0.019  ██  
```

## 8x8 cosine matrix

```
       p0  p1  p2  p3  p4  p5  p6  p7
 p0  1.00  0.95  0.93  0.92  0.92  0.92  0.92  0.91
 p1  0.95  1.00  0.98  0.97  0.96  0.95  0.95  0.94
 p2  0.93  0.98  1.00  0.98  0.97  0.96  0.96  0.95
 p3  0.92  0.97  0.98  1.00  0.99  0.98  0.97  0.96
 p4  0.92  0.96  0.97  0.99  1.00  0.99  0.98  0.97
 p5  0.92  0.95  0.96  0.98  0.99  1.00  0.99  0.98
 p6  0.92  0.95  0.96  0.97  0.98  0.99  1.00  0.99
 p7  0.91  0.94  0.95  0.96  0.97  0.98  0.99  1.00

```

mean off-diagonal cos = **0.959** (Phase 0 thresholds: <0.4 distributed ✓, >0.85 collapsed ✗)

## Recipe

Identical to Phase 1 run 1 except:
- backbone = Monet's vendored Qwen2.5-VL-3B-Instruct class (sys.modules patch);
- latent slots use `<abs_vis_token>...<abs_vis_token_pad>×8...</abs_vis_token>` (Monet tokens) instead of Phase 1's `<|image_pad|>×8`;
- training does Monet's two-pass forward (latent_mode=True for h + ce_patch_vec; latent_mode=False with spliced ce_patch_vec for NTP);
- h is read from `outputs.hidden_states[0][-1]` (last-layer, K rows; grad-enabled) NOT `outputs.latent_embeds` (which is detached).

Loss: `L = L_NTP + 1.0 · F.mse_loss(h, v_roi)` (mean over 1×K×D). Vision tower + projector frozen; LLM full FT. AdamW lr=1e-5, warmup=100, cosine→0, eff bsz=4 (bsz=1 × grad_accum=4), bf16, 1000 steps, 5K examples, seed=0.

Training summary: final step=1000 ntp=0.466 lvr=4.899 ||h||=40.2 ||v||=95.9 elapsed=4924s.

## Acceptance call

| criterion | result |
|---|---|
| compression_ratio (≥0.4) | PASS (1.132) |
| mean off-diag cos (≤0.55) | FAIL (0.959) |
| n_helpful (≥3) | FAIL (1) |
| qwen_base utility (>0) | PASS (0.030) |

**Overall: MARGINAL (2/4 criteria met).**

## Implication

If Phase 1.5 PASSES → Monet's architecture IS the load-bearing mechanism;
Phase 2 (faithful stage 2 recipe) will mostly confirm.

If Phase 1.5 FAILS → architecture alone isn't enough; the aux-image-during-
training mechanism (which Phase 1.5 doesn't use, and Phase 2 will) is
implicated.

## Caveats

- Phase 1.5 uses raw Qwen2.5-VL-3B-Instruct as the backbone (no Stage 1 SFT)
  — same starting checkpoint as Phase 1, but with Monet's modified forward.
- The Monet special tokens (`<abs_vis_token*>`, `<observation>` etc.) are added
  to the tokenizer but their embeddings are randomly initialized via
  `model.resize_token_embeddings(new_vocab)`. Initial NTP is therefore
  higher than Phase 1's (8.2 vs 7.6); convergence trajectory differs.
- The Monet vendored model uses internal attention masks via `build_4d_attn`
  WHEN `attention_mask_4d` is passed. Phase 1.5 does NOT pass this — it relies
  on default causal masking + the latent_mode recurrence. This means Phase 1.5
  tests the recurrence mechanism alone, not the 4D attention mechanism.
- v_roi-cosine on Phase 1.5 (above) should match Phase 1's 0.465 since the
  ROI selection + image features are identical — verifies data parity.
