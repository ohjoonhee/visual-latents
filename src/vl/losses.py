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
"""

import torch
import torch.nn.functional as F

from . import readers as _readers
from .curriculum import concept_weight, nll_weight, norm_weight


def nll_multi_anchor(h, batch, anchors, *, K_q: int) -> torch.Tensor:
    """Sum NLL over R anchors × K_q questions per image, then average.

    Args:
        h: [B, K, D] generator output (with grad).
        batch: list of length B; each item a dict with at least
               'questions': list[(q,a)] of length >= K_q.
        anchors: list[dict] from readers.load_anchors.
        K_q: number of questions to consume per image per step.

    Returns:
        scalar tensor — mean NLL across (R * K_q) reader passes.
    """
    if K_q <= 0:
        raise ValueError(f"K_q must be positive, got {K_q}")
    if not anchors:
        raise ValueError("Need at least one anchor")
    B, K, _ = h.shape

    total = h.new_zeros(())
    n_passes = 0
    for anchor in anchors:
        for j in range(K_q):
            q_text = []
            a_text = []
            for b in range(B):
                qa_list = batch[b]["questions"]
                qa = qa_list[j % len(qa_list)]
                if isinstance(qa, dict):
                    q_text.append(qa["q"])
                    a_text.append(qa["a"])
                else:
                    q_text.append(qa[0])
                    a_text.append(qa[1])
            loss_ij = _readers.forward_anchor(anchor, h, q_text, a_text, K=K)
            total = total + loss_ij
            n_passes += 1
    return total / n_passes


def concept_loss(h, v_sem, concept_mlp) -> torch.Tensor:
    """L_concept = 1 - mean cos(concept_mlp(h), V_sem[:, k mod T_v, :]).

    Args:
        h: [B, K, D] (generator output, with grad).
        v_sem: [B, T_v, D] (frozen teacher features, no grad).
        concept_mlp: nn.Sequential(D -> D//2 -> D).

    Returns:
        scalar tensor.
    """
    h_proj = concept_mlp(h)  # [B, K, D]
    K = h.shape[1]
    T_v = v_sem.shape[1]
    target_idx = torch.arange(K, device=h.device) % T_v
    targets = v_sem[:, target_idx, :]  # [B, K, D]
    cos = F.cosine_similarity(h_proj.float(), targets.float(), dim=-1)  # [B, K]
    return (1.0 - cos).mean()


def norm_loss(h, target_norm: float) -> torch.Tensor:
    """Per-token mean of (||h_i|| - target_norm)^2."""
    norms = h.float().norm(dim=-1)
    return ((norms - target_norm) ** 2).mean()


def combined(
    *,
    h: torch.Tensor,
    batch: list,
    anchors: list,
    v_sem: torch.Tensor | None,
    concept_mlp,
    cfg,
    step: int,
) -> dict:
    """Curriculum-weighted total loss for one training step.

    Args:
        h: [B, K, D] generator output.
        batch: list[dict] (see nll_multi_anchor).
        anchors: list[dict] from readers.load_anchors.
        v_sem: [B, T_v, D] frozen teacher (or None when w_concept == 0).
        concept_mlp: nn.Sequential D->D/2->D.
        cfg: LossConfig.
        step: current training step (for curriculum).

    Returns:
        dict with:
          total: scalar tensor (the loss to backprop)
          nll, concept, norm: scalar tensors (each component)
          mean_h_norm, cos_h_vsem: monitoring scalars (cos = 0.0 if w_concept==0)
          w_nll, w_concept, w_norm: float weights at this step
    """
    w_nll = nll_weight(step, cfg)
    w_c = concept_weight(step, cfg)
    w_n = norm_weight(step, cfg)

    # NLL is always required.
    nll = nll_multi_anchor(h, batch, anchors, K_q=cfg.K_q)

    # Concept (skipped if w_concept exactly 0 — cell C4).
    if cfg.w_concept == 0.0:
        concept = h.new_zeros(())
        cos_h_vsem = h.new_zeros(())
    else:
        if v_sem is None:
            raise ValueError("v_sem required when cfg.w_concept != 0")
        concept = concept_loss(h, v_sem, concept_mlp)
        # Monitoring cosine without the projection — raw h vs target slot.
        with torch.no_grad():
            K = h.shape[1]
            T_v = v_sem.shape[1]
            target_idx = torch.arange(K, device=h.device) % T_v
            targets = v_sem[:, target_idx, :]
            cos_h_vsem = F.cosine_similarity(h.float(), targets.float(), dim=-1).mean()

    # Norm.
    norm = norm_loss(h, cfg.target_norm)

    total = w_nll * nll + w_c * concept + w_n * norm

    with torch.no_grad():
        mean_h_norm = h.float().norm(dim=-1).mean()

    return {
        "total": total,
        "nll": nll.detach(),
        "concept": concept.detach(),
        "norm": norm.detach(),
        "mean_h_norm": mean_h_norm,
        "cos_h_vsem": cos_h_vsem if isinstance(cos_h_vsem, torch.Tensor) else h.new_zeros(()),
        "w_nll": float(w_nll),
        "w_concept": float(w_c),
        "w_norm": float(w_n),
    }
