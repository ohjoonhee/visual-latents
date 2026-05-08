"""Pivot A — collapse-prevention regularizers for the LVR loss.

Two variants, exactly as spelled out in `docs/PIVOT_A_DESIGN.md` §4.

C1 — pairwise cosine hinge:
    Penalize off-diagonal cosine similarity between K=8 latent slots
    above slack τ. The minimum is at C[i,j] ≤ τ for all i ≠ j, which is
    incompatible with h_t ≈ const collapse.

C2 — VICReg variance + covariance:
    Z-score per-dim across the (B*K) sample axis, then variance hinge at
    γ pushes each dim away from constancy and the off-diagonal cov term
    decorrelates dims. Both indirectly raise inter-slot cosine spread.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cos_penalty_loss(h: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Pairwise off-diagonal cosine hinge. h: [B, K, D]."""
    if h.dim() == 2:                    # [K, D] -> [1, K, D]
        h = h.unsqueeze(0)
    B, K, D = h.shape
    hn = F.normalize(h.float(), dim=-1)
    C = hn @ hn.transpose(-1, -2)       # [B, K, K]
    eye = torch.eye(K, device=h.device, dtype=hn.dtype).unsqueeze(0)
    off = C - eye                       # zero diagonal
    excess = (off - tau).clamp(min=0)   # only penalize pairs above tau
    return excess.pow(2).sum(dim=(-1, -2)).mean() / (K * (K - 1))


def vicreg_loss(
    h: torch.Tensor,
    var_weight: float = 1.0,
    cov_weight: float = 0.04,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg variance+covariance on z-scored h. h: [B, K, D].

    z-score per-dim across the (B*K) sample axis, then variance hinge at
    gamma=1.0 is meaningful regardless of activation magnitude.
    """
    if h.dim() == 2:
        h = h.unsqueeze(0)
    B, K, D = h.shape
    H = h.reshape(B * K, D).float()
    H = H - H.mean(dim=0, keepdim=True)
    std = (H.var(dim=0, unbiased=True) + eps).sqrt()
    H_z = H / std                                            # z-scored
    L_var = F.relu(gamma - std).mean()
    cov = (H_z.T @ H_z) / max(B * K - 1, 1)                  # [D, D]
    diag = torch.diagonal(cov)
    L_cov = (cov.pow(2).sum() - diag.pow(2).sum()) / D
    return var_weight * L_var + cov_weight * L_cov
