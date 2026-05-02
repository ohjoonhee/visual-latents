# Design — visual-latents

One-page architectural overview. Full derivations in `inherited/`.

## Method

A generator VLM emits latents `h ∈ R^{K × D}` from an input image. One or more
frozen *anchor* models (siblings of the generator's base) consume `h` spliced
into their vision-token positions and answer questions about the image. The
training objective forces `h` to encode visual content decodable by **any**
anchor — not just one specific decoder.

## Two variants

- **Variant A (SFT, this v0.1.0 scaffold).** Differentiable supervision via
  `L_NLL_multi(R, K_q)`. The reader-NLL is summed over R frozen anchors and
  K_q questions per image, plus auxiliary L_concept (LaViT-style cosine to
  teacher visual features) and L_norm (regularization toward natural visual-
  token norm 57.86). Curriculum cosine-warmup over 200 steps.

- **Variant B (RLVR, deferred).** trl.GRPOTrainer subclass with VLPO Gaussian
  reparameterization on h positions (vanilla GRPO does NOT update continuous
  latents — see `inherited/VARIANT_B_GRPO_DESIGN.md`). σ=5.0, asymmetric KL
  β_latent=0.04 / β_text=0.0, multi-anchor reward, random-control negative
  reward.

## Architecture (round-3 pinned)

LIVR-style same-VLM emitter (per `inherited/AUX_LOSS_AND_ARCH_DESIGN.md` B.1):
- Single Qwen2.5-VL-7B-Instruct backbone, LoRA r=32 on q/k/v/o + gate/up/down.
- K=16 new `<|latent|>` tokens added to tokenizer; their embedding rows are
  the only new full-rank trainable parameters besides LoRA.
- Stage-1 attention masking: during latent emission, answer tokens cannot
  attend to image tokens. Standard mask during anchor consumption.
- Generator + anchor-1 share LoRA in round-3 (decoupled in M2+).
- R=2 anchors: Qwen2.5-VL-7B-Instruct + Monet-SFT-7B-stage2.

## Loss

```
L = w_NLL(t) · L_NLL_multi(R=2, K_q=3)
  + 0.3 · L_concept
  + w_norm(t) · L_norm(target=57.86)
```

with cosine-warmup curriculum: `w_NLL(t)` ramps 0.1 → 1.0, `w_norm(t)` ramps
0 → 0.1, both over 200 steps. `w_concept` is constant (no warmup).

## Round-3 gate (5 cells)

| Cell | Variation | Purpose |
|---|---|---|
| C1 | full recipe | the proposed method |
| C2 | R=1 | does multi-anchor matter? |
| C3 | K_q=1 | does multi-Q matter? |
| C4 | w_concept=0 | does L_concept carry the grounding signal? |
| C5 | (image, q, a) shuffled within batch | random-control: would the gain reproduce on garbage? |

Pass thresholds (from `inherited/ROUND3_POC_DESIGN.md` §1):
1. Held-out NLL ≤ 2.5 nat
2. Reader-transfer NLL on Monet-SFT-7B ≤ 4.5 nat
3. Random-control gain ratio ≥ 2.0 (real / shuffled)
4. Steering Δnll ≥ +1.5 on each of {zero_pos, permute_within, gauss_noise}
5. 5K stress-test blank-image control causes ≥ 5pp accuracy drop

If all five pass on C1 AND C1 dominates {C2, C3, C4} AND C5 underperforms C1 →
proceed to M1. Any other pattern → reformulate.

## Key references

- `inherited/ROUND3_POC_DESIGN.md` — round-3 implementation spec (1055 lines)
- `inherited/PROLIFERATED_PROJECT_PLAN.md` — full M1/M2/M3 milestone plan
- `inherited/AUX_LOSS_AND_ARCH_DESIGN.md` — combined loss + LIVR-style arch
- `inherited/VARIANT_B_GRPO_DESIGN.md` — VLPO Gaussian extension
- `inherited/EVAL_BENCHMARK_PLAN.md` — 5K stress test + 4 controls
- `inherited/TRAINING_DATA_PLAN.md` — pilot/medium/full mixes
- `inherited/REPORT.md`, `inherited/JOURNAL.md` — POC findings + history
