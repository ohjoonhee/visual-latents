"""Interleaved reader-anchor consumption (design §4.2 option (i): latents-only).

We collapse all `K_total = sum(latent_block_sizes)` latents from one trace
into a single `[B, K_total, D]` tensor and pass it straight through the
parallel method's `forward_anchor`. This is exactly what design §4.2 calls
for: identical reader prompt structure to the parallel `K = K_total`
configuration, so the existing reader logic is reused 1:1.

This module is therefore intentionally thin — almost all of it is a
re-export. We keep the file as the named seam for design §4.2 so that when
the next iteration upgrades to option (ii) (latents + text trace) the
change is local to this file.
"""

from __future__ import annotations

import torch

from ..readers import forward_anchor


def forward_anchor_interleaved(
    anchor: dict,
    h: torch.Tensor,
    q_text: list[str],
    a_text: list[str],
    *,
    K_total: int,
) -> torch.Tensor:
    """Splice `h ∈ [B, K_total, D]` into the reader's K_total `<|image_pad|>` slots.

    Per design §4.2: identical to the parallel K=K_total case. Existing
    `vl.readers.forward_anchor` is shape-agnostic in K, so we delegate.
    """
    if h.shape[1] != K_total:
        raise ValueError(f"h.shape[1]={h.shape[1]} but K_total={K_total}")
    return forward_anchor(anchor, h, q_text, a_text, K=K_total)
