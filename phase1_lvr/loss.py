"""LVR loss = NTP CE + λ · MSE(h_t, v_{I_t}).

`h_t` are last-layer hidden states at K latent slot positions (one row per
example), `v_t` are post-projector image features at the K ROI patch indices.
No projection head.
"""
from __future__ import annotations

import torch


def lvr_mse(h: torch.Tensor, v: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """LVR regression loss: per-element MSE, mean over (B, K, D).

    The paper writes the loss as ``(1/T_v) · Σ ||h_t − v_t||²₂``. As-written,
    that's a sum over D ≈ 2048 dims per slot — at typical residual-stream
    norms (||h-v|| ~ 50) the per-position squared distance is ~2500, which
    dwarfs the NTP CE (~1-7) by 3+ orders of magnitude at λ=1.0. We adopt
    the engineering-standard interpretation that scales by D as well (i.e.
    `F.mse_loss(h, v)` semantics) so that λ=1.0 keeps the two terms in
    comparable magnitude. This is documented in RECIPE.md.

    Args:
        h: [B, K, D]
        v: [B, K, D]
    """
    if h.shape != v.shape:
        raise ValueError(f"h.shape={h.shape}, v.shape={v.shape}")
    diff_sq = (h.float() - v.float()).pow(2)  # [B, K, D]
    if reduction == "mean":
        return diff_sq.mean()
    if reduction == "sum":
        return diff_sq.sum()
    return diff_sq


def lvr_mse_paper(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Paper-faithful: (1/T_v) · Σ_t ||h_t − v_t||²₂.

    Math: per-example, per-position squared L2 distance, summed over D and
    averaged over K (which is the (1/T_v) · Σ_t in the paper). Then averaged
    over batch.

    Equals D × F.mse_loss(h, v) where D is the hidden dim. At Qwen2.5-VL-3B
    D=2048, this is 2048× larger than the run-1 `lvr_mse` form at the same λ.

    Args:
        h: [B, K, D]
        v: [B, K, D]
    """
    if h.shape != v.shape:
        raise ValueError(f"h.shape={h.shape}, v.shape={v.shape}")
    # Per-position squared L2 (sum over D), then mean over K, then mean over B.
    return ((h.float() - v.float()) ** 2).sum(dim=-1).mean()
