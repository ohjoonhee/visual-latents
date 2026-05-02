"""Combined loss for Variant A.

L = w_NLL(t) · L_NLL_multi(R, K_q) + w_concept · L_concept + w_norm(t) · L_norm

Per `docs/inherited/AUX_LOSS_AND_ARCH_DESIGN.md`:
- L_NLL_multi: sum over R frozen anchors and K_q questions per image.
  Generator's input is image-only; anchor's input is h + question + answer.
- L_concept: cosine to teacher visual features (V_sem from anchor-1's own
  post-merger visual tokens; pooled K_natural → K via crude k mod T_v in round-3,
  upgrade to learned-attention pooler in M2).
- L_norm: ‖h_i‖ → target_norm (57.86 for Qwen2.5-VL-7B), per-token mean-reduced.
- Curriculum: see `vl.curriculum`.

TODO: implement nll_multi_anchor, concept_loss, norm_loss, combined.
"""

import torch


def nll_multi_anchor(h, batch, anchors, *, K_q: int) -> torch.Tensor:  # noqa: ARG001
    """Sum NLL over R anchors × K_q questions per image. h is the latent batch."""
    raise NotImplementedError("losses.nll_multi_anchor not yet implemented")


def concept_loss(h, v_sem) -> torch.Tensor:  # noqa: ARG001
    """Cosine to teacher visual features. Pooled to match h's K dim."""
    raise NotImplementedError("losses.concept_loss not yet implemented")


def norm_loss(h, target_norm: float) -> torch.Tensor:
    """Per-token mean of (||h_i|| - target_norm)^2."""
    norms = h.float().norm(dim=-1)
    return ((norms - target_norm) ** 2).mean()


def combined(*, h, batch, anchors, v_sem, cfg, step: int):  # noqa: ARG001
    """Sum the weighted components. Returns dict with components + total."""
    raise NotImplementedError("losses.combined not yet implemented")
