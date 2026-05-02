# Loss — combined Variant A objective

Detailed derivation in `inherited/AUX_LOSS_AND_ARCH_DESIGN.md`. This doc is a
1-page reference for implementers.

## Combined loss

```
L(θ; t) = w_NLL(t) · L_NLL_multi(R, K_q)
        + w_concept · L_concept(h, V_sem)
        + w_norm(t) · L_norm(h, target_norm)
```

where `t` is the training step, `θ` is the trainable parameters (LoRA + new
`<|latent|>` token embeddings).

## Components

### L_NLL_multi(R, K_q)
Sum over R frozen anchors and K_q questions per image:
```
L_NLL_multi = (1/R) · (1/K_q) · Σ_r Σ_q  -log p_anchor_r(a_q | h, q_q)
```
where `h` is emitted by the generator from the image alone (no question), and
each anchor scores the answer token-level NLL with `h` spliced into its
vision-token positions. The generator is q-invariant: the same `h` must work
for all K_q questions about the same image.

### L_concept(h, V_sem)
Cosine to teacher visual features, with a bottleneck MLP to prevent identity
collapse:
```
h_proj = MLP_bottleneck(h)                   # D → D/2 → D
V_sem_pooled = pool(V_sem)                   # K_natural × D → K × D
L_concept = 1 - cos(h_proj, V_sem_pooled).mean()
```
- Round-3: `V_sem` = anchor-1's own post-merger visual tokens (free, on disk).
  Pooler = crude `k mod T_v` indexing (deferred learned-attention pooler to M2).
- M2: `V_sem` = Qwen2.5-VL-32B teacher visual features (per LaViT recipe).

### L_norm(h, target_norm)
Per-token L2 norm matching:
```
L_norm = (1/K) · Σ_k (||h_k|| - target_norm)^2
```
`target_norm = 57.86` is the natural visual-token mean norm for Qwen2.5-VL-7B
(measured in POC `compute_visual_baseline.py`).

**Important caveat from POC round-3 §17.1:** L_norm produces ~70 % of its
held-out gain even with random per-sample target norms — the gain is mostly
generic regularization, not target-matching. Therefore L_norm is kept as a
cheap stabilizer at λ=0.1 but is NOT load-bearing for grounding claims. The
load-bearing grounding signals are L_concept and L_NLL_multi.

## Curriculum

Cosine-warmup over `curriculum_warmup_steps = 200`:

| Coefficient | Start | End | Warmup | Notes |
|---|---|---|---|---|
| `w_NLL(t)` | 0.1 | 1.0 | 200 | gradual NLL ramp prevents early shortcut |
| `w_norm(t)` | 0.0 | 0.1 | 200 | norm reg comes online slowly |
| `w_concept` | 0.3 | 0.3 | n/a | constant; no warmup |

Implementation: `vl.curriculum.cosine_ramp(step, warmup, start, end)`.

## Sanity checks (during training)

Log to W&B every `eval_every_steps`:
- `loss/total`, `loss/nll`, `loss/concept`, `loss/norm`
- `weights/w_nll`, `weights/w_norm` — verify curriculum monotone
- `h/norm_mean`, `h/norm_std` — should approach target_norm under L_norm
- `h/cos_to_v_sem` — should approach 1 under L_concept
- `eval/heldout_nll_phi1`, `eval/transfer_nll_phi2` — every 100 steps on 30-sample subset
