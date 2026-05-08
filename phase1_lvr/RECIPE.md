# Phase 1 — LVR-Faithful Training Recipe

## Loss

Paper formula (arXiv:2509.24251):

```
L = L_NTP  +  λ_LVR · (1/T_v) · Σ_t ||h_t − v_t||²₂
```

- `h_t` — generator's last hidden state at the t-th latent slot (slot 0..K−1) in the decoder.
- `v_t` — post-projector visual token at the ROI patch INDEX `I_t`, taken from the encoded image sequence (whole image, not bbox crop).
- **No projection head.** Both vectors live in the joint LLM/vision space.
- Vision tower + projector are FROZEN; LLM is FULLY fine-tuned.

### Implementation deviation: `mean()` over all elements

The paper's per-token L2 squared norm sums over D ≈ 2048 hidden dims. At
realistic residual-stream norms ‖h-v‖ ≈ 50, the per-slot squared distance
≈ 2500, which dwarfs L_NTP (1-7 nats) by 3+ orders of magnitude at λ=1.0.
We instead average over (B, K, D) — i.e. standard `F.mse_loss(h, v)` —
so the two terms are comparable at λ=1.0. Documented in `loss.py:lvr_mse`.

## Hyperparameters

| knob | value |
|---|---|
| base | `Qwen/Qwen2.5-VL-3B-Instruct` |
| LLM | full fine-tune |
| vision tower + projector | frozen |
| K (latent slots) | 8 |
| λ_LVR | 1.0 |
| optimizer | AdamW, bf16 (Accelerate CPU offload) |
| LR | 1e-5 |
| warmup | 100 steps cosine |
| eff batch | bsz=1, grad_accum=4 (eff = 4) |
| steps | 1000 |
| weight decay | 0.0 |
| image resize | per Qwen2.5-VL processor default (max ≈ 12,845,056 px); we cap to ~512K px for memory |
| dtype | bfloat16 |

## Per-position assignment (deviation from paper)

The paper trains with variable `T_v` and evaluates at fixed K. **Phase 1 trains with fixed K=8.**

Gather rule: from the `K_image` post-merger image-token sequence, find indices whose patch center falls inside the bbox.
- If `>8`, take first 8 in raster order.
- If `<8`, **pad with closest-to-bbox-center indices** (i.e., extend with image patches closest in Euclidean distance to the bbox center, up to fill K=8). Never skip — padding keeps batch shape uniform.

## Data

- Dataset: `ohjoonhee/visual-cot-50k-poc` (Hugging Face Hub).
- Train: 5,000 random examples (seed=0) from the `train` split.
- Held-out for ablation: 200 examples from the `eval` split.

## Prompt template

The generator forward uses a chat-style prompt:

```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\n<|vision_start|><image>...<|vision_end|>{question}<|im_end|>\n
<|im_start|>assistant\n<latent_pad>×8 {answer}<|im_end|>
```

The K=8 latent slots immediately follow the assistant header (so they live in
the same conditional context that produces the answer). `h_t` is taken from the
LAST hidden state at those slot positions; `v_t` is the post-projector image
features at indices `I_t`.

We re-use the `<|image_pad|>` token id for the latent slot in this run (the
Qwen2.5-VL tokenizer already has it; the model treats it positionally — only
the embedding-space splice matters).

## Hardware

- Single A6000 (48 GB), bf16, accelerate CPU offload (full ZeRO-2 not strictly
  required for Qwen2.5-VL-3B at bsz=1).

## Acceptance criteria

| metric | target |
|---|---|
| compression_ratio | ≥ 0.4 |
| mean off-diag cosine | ≤ 0.55 |
| n_helpful (single-keep) | ≥ 3 / 8 |
| frozen-Qwen utility | > 0 |

≥ 3 of 4 → Phase 1 passes; ≤ 1 → cluster-scale escalation flagged.
