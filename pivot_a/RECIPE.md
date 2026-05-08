# Pivot A — Recipe

Identical to `phase1_lvr/configs/lvr_3b_5k.yaml` except for the new
`reg_*` fields. Two variants, trained separately on the same data.

## Common (both C1 and C2)

| field | value |
|---|---|
| base | `Qwen/Qwen2.5-VL-3B-Instruct` |
| trainable | LLM only (vision tower + projector frozen) |
| dtype | bf16 |
| gradient_checkpointing | true |
| K (latent slots) | 8 |
| lambda_lvr | 1.0 |
| LVR loss form | mean-MSE: `F.mse_loss(h, v_roi)` |
| optimizer | AdamW, lr=1e-5, betas=(0.9, 0.95), wd=0 |
| schedule | cosine, warmup=100, decay→0 |
| max_steps | 1000 |
| micro batch | 1 |
| grad_accum | 4 (eff bsz = 4) |
| min_pixels / max_pixels | 3136 / 401408 |
| data | `ohjoonhee/visual-cot-50k-poc` train split |
| n_train_examples | 5000 (seeded shuffle, seed=0) |
| held-out eval | same dataset's `eval` split, first 200 (seed=0 shuffle) |
| latent slot token | `<|image_pad|>` (Phase-1-style; no Monet special tokens) |
| seed | 0 |

Loss:
```
L = L_NTP + lambda_lvr · F.mse_loss(h, v_roi) + lambda_reg · L_reg
```
with `lambda_lvr = lambda_reg = 1.0`.

## Variant C1 — pairwise cosine hinge

```yaml
reg_kind: cos
lambda_reg: 1.0
reg_tau: 0.5
```

```python
def cos_penalty_loss(h, tau=0.5):
    hn = F.normalize(h.float(), dim=-1)         # [B, K, D]
    C  = hn @ hn.transpose(-1, -2)              # [B, K, K]
    eye = torch.eye(K, device=h.device, dtype=hn.dtype).unsqueeze(0)
    off = C - eye
    excess = (off - tau).clamp(min=0)
    return excess.pow(2).sum(dim=(-1, -2)).mean() / (K * (K - 1))
```

Hypothesis: collapse metric and loss target are the *same quantity*;
penalize off-diag cos directly with slack at τ=0.5 (which matches the
Monet-stage-2 natural target geometry of 0.38 and Phase-1's v_roi target
of 0.46).

## Variant C2 — VICReg variance + covariance

```yaml
reg_kind: vicreg
lambda_reg: 1.0
reg_var_w: 1.0
reg_cov_w: 0.04
reg_gamma: 1.0
```

```python
def vicreg_loss(h, var_weight=1.0, cov_weight=0.04, gamma=1.0, eps=1e-4):
    H = h.reshape(B*K, D).float()
    H = H - H.mean(dim=0, keepdim=True)
    std = (H.var(dim=0, unbiased=True) + eps).sqrt()
    H_z = H / std
    L_var = F.relu(gamma - std).mean()
    cov = (H_z.T @ H_z) / max(B*K - 1, 1)
    diag = torch.diagonal(cov)
    L_cov = (cov.pow(2).sum() - diag.pow(2).sum()) / D
    return var_weight * L_var + cov_weight * L_cov
```

Hypothesis: collapse is a *representation pathology* (low per-dim
variance + high inter-dim redundancy), not specifically a geometric one;
fixing it at the representation level (z-scored, so γ=1.0 is meaningful
regardless of Qwen residual-stream magnitude) is more robust.

## Acceptance (per variant)

| criterion | target |
|---|---|
| compression_ratio | ≥ 0.4 |
| **mean off-diag cos** | **≤ 0.55 (dispositive)** |
| n_helpful (margin ≥ 0.05 nat) | ≥ 3 |
| qwen_base utility (none − all NLL) | > 0 |

PASS = ≥ 3/4. MARGINAL = 2/4. FAIL = ≤ 1/4.

## Files

- Trainer: `pivot_a/trainer.py`
- Reg loss: `pivot_a/reg.py`
- Extract: `pivot_a/extract_latents.py`
- Ablate: `pivot_a/ablation_runner.py`
- Report builder: `pivot_a/build_report.py`
- Pipeline: `pivot_a/run_full.sh` (foreground sequencer) +
  `pivot_a/post_train_pipeline.sh` (autonomous wait+extract+ablate+report)
