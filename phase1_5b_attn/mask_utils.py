"""4D attention mask construction for Phase 1.5b.

Re-implementation of Monet's `build_4d_attn` constrained to the Phase 1.5b
data layout (single question image, no helper images, no `<observation>`
blocks).

Source mapping:
- `build_4d_attn_wo_helper_images` in upstream Monet
  (https://github.com/NOVAglow646/Monet, `src/utils.py:764`) handles the
  no-helper-image case — returns a bool [B, 1, L, L] mask of causal AND
  padding-valid AND optional `mask_latent` (latent slot positions are not
  visible to any subsequent query token as keys).
- We extend it with a `latent_cross_isolate` flag per the Phase 1.5b
  hypothesis: latent slot t_i cannot attend to latent slot t_j (j != i).
  Slots can still attend to ALL non-slot context (image, question, slot's
  own row).

Returned format: dict with key `'full_attention'` holding either:
- a bool tensor (True = allow) when `return_type='bool'` (matches the
  vendored `modeling_qwen2_5_vl_monet.py` consumption pattern; see
  `eager_attention_forward` at line 432 — actually expects ADDITIVE so we
  use 'additive' here for safety).
- an additive tensor (0 = allow, very-negative = block) of dtype `dtype`.

The vendored model adds the mask onto attention logits in
`eager_attention_forward` (line 432), so the **additive** form is required.

References (line numbers in
`phase0_monet_probe/monet_model/modeling_qwen2_5_vl_monet.py`):
- Forward signature: line 1560 (`attention_mask_4d`).
- Latent-mode pre-answer slice: line 1775 (`[:, :, :ans_start, :ans_start]`).
- Latent-mode text-chunk slice: line 1831 (`[:, :, q0:q1, :k1]`).
- Non-latent forward forward: line 2021 — passes `attention_mask_4d`
  directly to `language_model` as `attention_mask`.
- Eager attention adds mask: line 432-434 (`attn_weights = attn_weights +
  causal_mask`).

Caveat: the per-latent-slot recurrent forward (line 1922-1934) hard-codes
the 1D `attention_mask` and does NOT consume `attention_mask_4d`. This
means the cross-slot-isolation effect on the per-slot KV-cache pass is
NOT enforced here — only on the (a) pre-answer prefix, (b) post-latent
text chunks, (c) non-latent CE forward. We document this in RECIPE.md.
"""
from __future__ import annotations

from typing import Optional

import torch


def _causal_pad_base(L: int, pad_mask: torch.Tensor, device, dtype=torch.bool) -> torch.Tensor:
    """Build [B, L, L] bool: causal AND padding-valid (rows AND cols)."""
    B = pad_mask.shape[0]
    causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))
    allowed = causal.unsqueeze(0).expand(B, L, L).clone()
    valid = pad_mask.bool().to(device)
    for b in range(B):
        allowed[b] &= valid[b].unsqueeze(0)
        allowed[b] &= valid[b].unsqueeze(1)
    return allowed


def build_4d_attn(
    input_ids: torch.Tensor,
    pad_mask: torch.Tensor,
    latent_token_id: int,
    *,
    mask_latent: bool = False,
    latent_cross_isolate: bool = True,
    return_type: str = "additive",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Build 4D attention mask for Phase 1.5b.

    Args:
        input_ids: LongTensor [B, L]
        pad_mask:  BoolTensor/LongTensor [B, L], 1/True for real tokens
        latent_token_id: token id for `<abs_vis_token_pad>` (Monet's slot pad)
        mask_latent: if True, latent slot positions are also invisible to
            subsequent (non-slot) query tokens as keys. Mirrors the Monet
            upstream `mask_latent=True` flag.
        latent_cross_isolate: if True, latent slot t_i cannot attend to
            latent slot t_j for j != i. THIS is the Phase 1.5b key change.
        return_type: 'bool' for [B, 1, L, L] bool tensor (True=allow), or
            'additive' for [B, 1, L, L] float tensor (0=allow, -inf=block).
        dtype: dtype of the additive mask. Use the model's compute dtype
            (bf16 typically) so it adds cleanly to the bf16 attention
            logits.

    Returns:
        dict with key 'full_attention' -> [B, 1, L, L] tensor.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2D [B, L], got {tuple(input_ids.shape)}")
    if pad_mask.dim() != 2:
        raise ValueError(f"pad_mask must be 2D [B, L], got {tuple(pad_mask.shape)}")
    B, L = input_ids.shape

    # Work on CPU like upstream — this is a one-time per-batch op, not perf-sensitive.
    ids = input_ids.detach().cpu()
    pad = pad_mask.detach().cpu()

    allowed = _causal_pad_base(L, pad, device=torch.device("cpu"))

    for b in range(B):
        slot_pos = (ids[b] == latent_token_id).nonzero(as_tuple=False).flatten()
        if slot_pos.numel() == 0:
            continue

        if latent_cross_isolate:
            # Block slot_i query rows from attending to slot_j key cols (j != i).
            # Diagonal stays as-is (slot can attend to itself).
            n = slot_pos.numel()
            rows = slot_pos.unsqueeze(1).expand(n, n)
            cols = slot_pos.unsqueeze(0).expand(n, n)
            eye = torch.eye(n, dtype=torch.bool)
            # off-diagonal = block
            block_pairs = ~eye
            allowed[b][rows[block_pairs], cols[block_pairs]] = False

        if mask_latent:
            # Slot positions invisible to any later (non-slot) query as keys.
            # Following upstream: rows r > slot_j set allowed[r, slot_j]=False,
            # except keep slot's own diagonal. We've already set slot/slot
            # off-diagonal above; here we only block non-slot rows after slots.
            r_idx = torch.arange(L)
            # rows that come strictly after some slot
            rows_after_slot = (r_idx.unsqueeze(0) > slot_pos.unsqueeze(1)).any(dim=0)
            # Restrict to non-slot rows so we don't undo the diagonal already set.
            slot_mask = torch.zeros(L, dtype=torch.bool)
            slot_mask[slot_pos] = True
            rows_to_block = rows_after_slot & ~slot_mask
            if rows_to_block.any():
                rb = rows_to_block.nonzero(as_tuple=False).squeeze(-1)
                allowed[b][rb.unsqueeze(1), slot_pos.unsqueeze(0)] = False

    allowed = allowed.unsqueeze(1)  # [B, 1, L, L]

    if return_type == "bool":
        return {"full_attention": allowed.to(input_ids.device)}
    elif return_type == "additive":
        # Use a large negative (matches HF's `torch.finfo(dtype).min`) for blocked.
        # eager_attention_forward adds this to logits then softmax → blocked → ~0 prob.
        neg = torch.finfo(dtype).min
        attn = torch.zeros((B, 1, L, L), dtype=dtype)
        attn.masked_fill_(~allowed, neg)
        return {"full_attention": attn.to(input_ids.device)}
    else:
        raise ValueError(f"return_type must be 'bool' or 'additive', got {return_type!r}")


def build_monet_4d_attn(
    input_ids: torch.Tensor,
    latent_token_id: int,
    *,
    pad_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.bfloat16,
    mask_latent: bool = False,
    latent_cross_isolate: bool = True,
) -> dict:
    """Convenience wrapper: builds an additive 4D attention mask using the
    Phase 1.5b cross-slot-isolation rule.

    If `pad_mask` is None, treats all tokens as valid (no padding).
    Returns dict matching the format expected by Monet's
    `Qwen2_5_VLForConditionalGeneration.forward(attention_mask_4d=...)`.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2D, got {tuple(input_ids.shape)}")
    if pad_mask is None:
        pad_mask = torch.ones_like(input_ids)
    return build_4d_attn(
        input_ids,
        pad_mask,
        latent_token_id=latent_token_id,
        mask_latent=mask_latent,
        latent_cross_isolate=latent_cross_isolate,
        return_type="additive",
        dtype=dtype,
    )


def render_mask_grid(
    bool_mask: torch.Tensor,
    *,
    max_rows: int = 64,
    max_cols: int = 64,
    slot_positions: Optional[list] = None,
) -> str:
    """Pretty-print a [L, L] bool mask as a text grid.
    '#' = allowed, '.' = blocked. Slot rows/cols are highlighted with brackets.
    """
    if bool_mask.dim() == 4:
        bool_mask = bool_mask[0, 0]
    elif bool_mask.dim() == 3:
        bool_mask = bool_mask[0]
    L = bool_mask.shape[0]
    rows = list(range(min(L, max_rows)))
    cols = list(range(min(L, max_cols)))

    slot_set = set(slot_positions or [])
    lines = []
    header = "    " + "".join(("S" if c in slot_set else " ") for c in cols)
    lines.append(header)
    for r in rows:
        prefix = f"{r:3d} "
        chars = []
        for c in cols:
            ch = "#" if bool(bool_mask[r, c]) else "."
            chars.append(ch)
        marker = " <-- SLOT" if r in slot_set else ""
        lines.append(prefix + "".join(chars) + marker)
    return "\n".join(lines)
