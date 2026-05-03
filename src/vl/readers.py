"""Frozen anchor (reader) management.

Loads R anchor models once at startup. The first anchor may be the generator
itself (shared weights + LoRA per round-3 §2.6); subsequent anchors are loaded
fresh, frozen, no LoRA.

Each anchor consumes h spliced into K <|image_pad|> positions in a
text-only prompt (no real image), then computes CE loss on the answer span.
Gradient flows back through h.

Round-3 R=2: Qwen/Qwen2.5-VL-7B-Instruct + NOVAglow646/Monet-SFT-7B. Both
share the <|image_pad|> token id (151655) since Monet-SFT-7B uses the
Qwen2.5-VL tokenizer.

Public API:
    load_anchors(paths, generator, generator_processor, dtype) -> list[dict]
    forward_anchor(anchor, h, q_text, a_text, *, K) -> scalar NLL
"""

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def _build_anchor_prompt_no_answer(K: int, question: str) -> str:
    img_pads = "<|image_pad|>" * K
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|vision_start|>{img_pads}<|vision_end|>{question}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _build_anchor_prompt_full(K: int, question: str, answer: str) -> str:
    return _build_anchor_prompt_no_answer(K, question) + answer + "<|im_end|>"


def load_anchors(
    paths: list[str],
    generator_model=None,
    generator_processor=None,
    dtype: torch.dtype = torch.bfloat16,
) -> list[dict]:
    """Load R anchor models.

    If `generator_model` is given AND the first anchor's path matches the
    generator's `base_model` (or equivalently the first path in `paths`), the
    first anchor REUSES the generator (shared weights + LoRA per round-3 §2.6).
    Otherwise each path is loaded fresh, frozen, with no LoRA.

    Args:
        paths: list of model identifiers. R = len(paths).
        generator_model: optional, the generator built by build_generator. If
                        provided AND `paths[0]` matches its base, the first
                        anchor is the generator.
        generator_processor: required if generator_model is given (shares the
                            extended tokenizer with new tokens).
        dtype: dtype for newly-loaded anchors.

    Returns:
        list[dict] of length R, each with:
            model: PreTrainedModel
            processor: AutoProcessor
            tokenizer: shortcut to processor.tokenizer
            shared_with_generator: bool
            image_token_id: int (the <|image_pad|> token id)
    """
    anchors: list[dict] = []
    if not paths:
        raise ValueError("paths must be non-empty")

    for idx, path in enumerate(paths):
        if (
            idx == 0
            and generator_model is not None
            and generator_processor is not None
            and getattr(generator_model.config, "_name_or_path", None) == path
        ):
            anchors.append(
                {
                    "model": generator_model,
                    "processor": generator_processor,
                    "tokenizer": generator_processor.tokenizer,
                    "shared_with_generator": True,
                    "image_token_id": generator_model.config.image_token_id,
                }
            )
            continue

        # Fresh load, frozen, no LoRA.
        processor = AutoProcessor.from_pretrained(path)
        # If the generator added new tokens, mirror those into the anchor's
        # tokenizer so the anchor can tokenize prompts that contain them
        # (defensive — anchor prompts don't currently use them, but training
        # may share tokenizers).
        if generator_processor is not None:
            new_toks = ["<|latent|>", "<|latent_start|>", "<|latent_end|>"]
            processor.tokenizer.add_special_tokens({"additional_special_tokens": new_toks})

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, dtype=dtype)
        if generator_processor is not None:
            model.resize_token_embeddings(len(processor.tokenizer))
        for p in model.parameters():
            p.requires_grad_(False)
        model = model.cuda()

        anchors.append(
            {
                "model": model,
                "processor": processor,
                "tokenizer": processor.tokenizer,
                "shared_with_generator": False,
                "image_token_id": model.config.image_token_id,
            }
        )

    return anchors


def forward_anchor(
    anchor: dict,
    h: torch.Tensor,
    q_text: list[str],
    a_text: list[str],
    *,
    K: int,
) -> torch.Tensor:
    """Forward h spliced into K <|image_pad|> slots; return CE loss on answer span.

    Args:
        anchor: dict from load_anchors.
        h: [B, K, D] generator output (with grad).
        q_text: list[str] of length B.
        a_text: list[str] of length B.
        K: int (must match h.shape[1]).

    Returns:
        scalar tensor — mean NLL across batch (per-example loss is itself the
        mean over answer-token positions).
    """
    if h.shape[1] != K:
        raise ValueError(f"h.shape[1]={h.shape[1]} but K={K}")
    B = h.shape[0]
    if len(q_text) != B or len(a_text) != B:
        raise ValueError(f"q/a list length must be {B}; got {len(q_text)}/{len(a_text)}")

    model = anchor["model"]
    tokenizer = anchor["tokenizer"]
    image_token_id = anchor["image_token_id"]

    # Tokenize each example separately to compute per-example prompt length
    # (the boundary between "no-answer prompt" and "with-answer prompt").
    no_answer_ids: list[list[int]] = []
    full_ids: list[list[int]] = []
    for q, a in zip(q_text, a_text, strict=True):
        prompt_no_a = _build_anchor_prompt_no_answer(K, q)
        prompt_full = _build_anchor_prompt_full(K, q, a)
        ids_no_a = tokenizer(prompt_no_a, add_special_tokens=False).input_ids
        ids_full = tokenizer(prompt_full, add_special_tokens=False).input_ids
        if ids_full[: len(ids_no_a)] != ids_no_a:
            # Tokenization isn't a clean prefix (rare but possible). Re-encode the
            # answer directly and concat.
            ids_a = tokenizer(a + "<|im_end|>", add_special_tokens=False).input_ids
            ids_full = ids_no_a + ids_a
        no_answer_ids.append(ids_no_a)
        full_ids.append(ids_full)

    # Right-pad to common length.
    max_len = max(len(x) for x in full_ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id  # fallback; only used in pad positions
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=h.device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=h.device)
    # labels: -100 outside the answer span.
    labels = torch.full((B, max_len), -100, dtype=torch.long, device=h.device)
    for b, ids in enumerate(full_ids):
        L = len(ids)
        input_ids[b, :L] = torch.tensor(ids, dtype=torch.long, device=h.device)
        attention_mask[b, :L] = 1
        # Answer span is full_ids[len(no_answer_ids):L].
        ans_start = len(no_answer_ids[b])
        labels[b, ans_start:L] = input_ids[b, ans_start:L]

    # Sanity: K image_pad positions per row (the spliced slots).
    img_pad_count = (input_ids == image_token_id).sum(dim=1)
    if not torch.all(img_pad_count == K):
        raise RuntimeError(
            f"Expected K={K} image_pad positions per row; got per-row counts {img_pad_count.tolist()}"
        )

    # Build inputs_embeds and splice h into image_pad positions.
    base_embeds = model.get_input_embeddings()(input_ids)  # [B, L, D] bf16
    # We need gradient through h, so build a fresh tensor.
    h_cast = h.to(dtype=base_embeds.dtype)
    inputs_embeds = base_embeds.clone()
    for b in range(B):
        positions = (input_ids[b] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
        # Splice h[b] into the K positions in order.
        inputs_embeds[b, positions, :] = h_cast[b]

    # Forward — no pixel_values (text-only path).
    outputs = model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # [B, L, V]

    # Standard shift: logits[..., :-1, :] predict labels[..., 1:].
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return loss
