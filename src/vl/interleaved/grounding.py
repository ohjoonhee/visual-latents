"""Stage-1 block-grounding loss for the interleaved variant.

Per `docs/INTERLEAVED_POC_RESULTS.md` §12 + `docs/INTERLEAVED_DATASET_RECON.md`,
the round-3 binding-test failure showed that reader-NLL alone provides no
incentive for h to encode image-specific content. The recommended pivot is
Mirage-style Stage-1 distillation against the model's own vision-encoder
features:

  Stage 1 (steps 0..warmup): w_grounding=1.0, w_nll=0.0 → 1.0
    h_block must reproduce V_sem of the appropriate region per block.
  Stage 2 (steps warmup..end): w_grounding=0.3, w_nll=1.0
    Reader-NLL takes over; grounding becomes a constant regulariser.

Mapping of latent blocks to image regions (T_blocks=2):
    block 1 (latent positions 0..k_latent-1): target = V_sem(full image)
    block 2 (latent positions k_latent..2*k_latent-1): target = V_sem(crop to bbox)

This is the natural "first look at the whole, then zoom" pattern that
Visual-CoT bbox annotations encode.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..curriculum import cosine_ramp


def block_grounding_loss(
    h: torch.Tensor,
    targets_per_block: list[torch.Tensor],
    block_sizes: list[int],
    concept_mlp,
) -> torch.Tensor:
    """Per-block cosine loss against pre-computed V_sem targets.

    Args:
        h: [B, K_total, D] generator latents (with grad).
        targets_per_block: list[Tensor] of length T_blocks; each [B, D] target.
            Order matches the latent block order (block 0 first, etc.).
        block_sizes: list[int] of length T_blocks, k_latent per block.
            sum(block_sizes) must equal h.shape[1].
        concept_mlp: nn.Sequential D -> D/2 -> D — projects h_block_avg into
            the V_sem space before the cosine. Reused from build_generator.

    Returns:
        Scalar tensor: mean over blocks of (1 - cos(MLP(h_block_avg), target)).
    """
    if len(targets_per_block) != len(block_sizes):
        raise ValueError(
            f"len(targets_per_block)={len(targets_per_block)} != "
            f"len(block_sizes)={len(block_sizes)}"
        )
    if sum(block_sizes) != h.shape[1]:
        raise ValueError(
            f"sum(block_sizes)={sum(block_sizes)} != h.shape[1]={h.shape[1]}"
        )

    losses = []
    pos = 0
    for b_idx, k_b in enumerate(block_sizes):
        # Average over the k_latent positions in this block
        h_block = h[:, pos:pos + k_b, :].mean(dim=1)  # [B, D]
        h_proj = concept_mlp(h_block)  # [B, D]
        target = targets_per_block[b_idx]
        # Move target to h's device + cast to float32 for stable cosine
        target = target.to(h.device).to(torch.float32)
        # h_proj may be bf16; cast for cosine
        cos = F.cosine_similarity(h_proj.float(), target, dim=-1)  # [B]
        losses.append((1.0 - cos).mean())
        pos += k_b
    return torch.stack(losses).mean()


def stage1_curriculum_weights(
    step: int,
    warmup: int = 500,
    grounding_start: float = 1.0,
    grounding_end: float = 0.3,
    nll_start: float = 0.0,
    nll_end: float = 1.0,
) -> tuple[float, float]:
    """Stage-1 → Stage-2 curriculum schedule.

    During warmup:
        w_grounding decays cosine from grounding_start to grounding_end
        w_nll       ramps cosine from nll_start to nll_end

    After warmup, both stay flat at their end values.

    Returns: (w_grounding, w_nll).
    """
    w_grounding = cosine_ramp(step, warmup, grounding_start, grounding_end)
    w_nll = cosine_ramp(step, warmup, nll_start, nll_end)
    return w_grounding, w_nll
