"""Generator model: LIVR-style same-VLM + special <|latent|> tokens + LoRA.

Per `docs/inherited/AUX_LOSS_AND_ARCH_DESIGN.md` Part B.1 (round-3 primary):
- Single Qwen2.5-VL-7B-Instruct backbone with LoRA r=32 on q/k/v/o + gate/up/down.
- K new <|latent|> tokens added to tokenizer; their embedding rows are the only
  new full-rank trainable parameters besides LoRA.
- Stage-1 attention masking: during latent emission, answer tokens cannot attend
  to image tokens. Standard mask during anchor consumption.
- Generator and reader-anchor share the same LoRA in round-3 (decoupled in M2+).

TODO: implement build_generator() returning (model, tokenizer, latent_token_ids).
"""

import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration


def build_generator(cfg, dtype=torch.bfloat16):  # noqa: ARG001
    """Construct the generator with LoRA + K new <|latent|> tokens.

    Args:
        cfg: ModelConfig
        dtype: model dtype

    Returns:
        model: Qwen2_5_VLForConditionalGeneration with LoRA injected
        tokenizer: AutoTokenizer with K new <|latent|> tokens added
        latent_token_ids: list[int] of length K — token ids for the K positions
    """
    raise NotImplementedError("model.build_generator not yet implemented")
