"""Frozen anchor (reader) management.

Loads R anchor models once at startup, freezes all parameters, runs each forward
in inference mode for L_NLL_multi computation.

Round-3 R=2: Qwen/Qwen2.5-VL-7B-Instruct + (local stage2 path of NOVAglow646/
Monet-SFT-7B). Both must share the same vision-token convention (<|image_pad|>)
so h splices into both anchors at the same position.

TODO: implement load_anchors, forward_anchor.
"""

import torch


def load_anchors(paths: list[str], dtype=torch.bfloat16):  # noqa: ARG001
    """Load R frozen anchor models. Returns list of (model, tokenizer) pairs."""
    raise NotImplementedError("readers.load_anchors not yet implemented")


@torch.no_grad()
def forward_anchor(anchor, h, prompt_ids, label_ids, pad_start: int):  # noqa: ARG001
    """Forward h spliced into anchor's vision-token positions; return NLL."""
    raise NotImplementedError("readers.forward_anchor not yet implemented")
