"""Generator model: LIVR-style same-VLM + special <|latent|> tokens + LoRA.

Per `docs/inherited/AUX_LOSS_AND_ARCH_DESIGN.md` Part B.1 (round-3 primary):
- Single Qwen2.5-VL-7B-Instruct backbone with LoRA r=32 on q/k/v/o + gate/up/down.
- K new <|latent|> tokens added to tokenizer; their embedding rows are the only
  new full-rank trainable parameters besides LoRA.
- Stage-1 attention masking: during latent emission, answer tokens cannot attend
  to image tokens. Standard mask during anchor consumption. (Stage-1 mask is
  not yet implemented here; trainer is expected to construct the 4D mask.)
- Generator and reader-anchor share the same LoRA in round-3 (decoupled in M2+).

Public API:
    build_generator(cfg, dtype) -> dict
    forward_generator(model, new_emb, new_token_ids, processor, images, cfg) -> h
    get_v_sem(model, processor, images) -> v_sem  [B, T_v, D]
"""

import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

NEW_TOKENS: list[str] = ["<|latent|>", "<|latent_start|>", "<|latent_end|>"]


def _build_generator_prompt(K: int) -> str:
    """LIVR-style q-invariant generator prompt (image + K latent slots, no question)."""
    latents = "<|latent|>" * K
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"<|latent_start|>{latents}<|latent_end|>"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_generator(cfg, dtype=torch.bfloat16) -> dict:
    """Construct the generator with LoRA + K new <|latent|> tokens.

    Args:
        cfg: ModelConfig
        dtype: base-model dtype (bf16 by default)

    Returns:
        dict with keys:
            model: Qwen2_5_VLForConditionalGeneration with LoRA injected on
                   `model.model.language_model`. Vision tower & lm_head frozen.
            processor: AutoProcessor (tokenizer extended with new tokens).
            tokenizer: shortcut to processor.tokenizer.
            concept_mlp: nn.Sequential(D -> D//2 -> D) in bf16 on CUDA.
            new_emb: nn.Parameter [3, D] in fp32 — trainable embedding rows for
                     the 3 new tokens (latent, latent_start, latent_end). The
                     base embedding table stays frozen; trainer splices these
                     rows in at forward time via `forward_generator`.
            latent_token_ids: list[int] of length cfg.K, each the id of <|latent|>.
                              Used by trainer when constructing prompts.
            new_token_ids: dict[str,int] mapping name -> id for the 3 new tokens.
    """
    processor = AutoProcessor.from_pretrained(cfg.base_model)
    tokenizer = processor.tokenizer

    # 1. Add 3 new special tokens.
    added = tokenizer.add_special_tokens({"additional_special_tokens": NEW_TOKENS})
    new_token_ids = {name: tokenizer.convert_tokens_to_ids(name) for name in NEW_TOKENS}
    if any(tok_id is None for tok_id in new_token_ids.values()):
        raise RuntimeError(f"Failed to register new tokens; got: {new_token_ids}")
    # Strip the angle-bracket-pipe noise to make a clean key map.
    clean_map = {
        "latent": new_token_ids["<|latent|>"],
        "latent_start": new_token_ids["<|latent_start|>"],
        "latent_end": new_token_ids["<|latent_end|>"],
    }

    # 2. Load model in target dtype.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.base_model, dtype=dtype)
    model.resize_token_embeddings(len(tokenizer))

    # 3. Snapshot the new rows (initialised as image_pad_emb + 0.02 * randn) into
    #    a fresh fp32 nn.Parameter. The base embedding table stays frozen.
    image_pad_id = model.config.image_token_id
    with torch.no_grad():
        emb_table = model.get_input_embeddings().weight  # [V, D], bf16
        D = emb_table.shape[1]
        image_pad_emb = emb_table[image_pad_id].detach().float().clone()  # [D] fp32
        rows = [image_pad_emb + 0.02 * torch.randn_like(image_pad_emb) for _ in range(3)]
        new_emb_init = torch.stack(rows, dim=0)  # [3, D] fp32
    new_emb = nn.Parameter(new_emb_init.detach().clone(), requires_grad=True)

    # 4. Freeze ALL base parameters BEFORE LoRA injection.
    for p in model.parameters():
        p.requires_grad_(False)

    # 5. Inject LoRA into the language model only (not vision tower or lm_head).
    #    `get_peft_model` doesn't work on Qwen2_5_VLTextModel (lacks
    #    prepare_inputs_for_generation), so use inject_adapter_in_model which
    #    mutates in place and preserves the parent module shape.
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    inject_adapter_in_model(lora_cfg, model.model.language_model)

    # 6. Move to GPU.
    model = model.cuda()
    new_emb.data = new_emb.data.cuda()

    # 7. Concept-MLP D -> D/2 -> D in bf16 on CUDA.
    concept_mlp = nn.Sequential(
        nn.Linear(D, D // 2),
        nn.GELU(),
        nn.Linear(D // 2, D),
    ).to(dtype=dtype, device="cuda")
    for p in concept_mlp.parameters():
        p.requires_grad_(True)

    # 8. Sanity-check the trainable param surface (will surface bugs early).
    vis_train = sum(p.numel() for p in model.model.visual.parameters() if p.requires_grad)
    lm_train_lora = sum(
        p.numel()
        for n, p in model.model.language_model.named_parameters()
        if p.requires_grad and "lora_" in n
    )
    if vis_train != 0:
        raise RuntimeError(f"Vision tower has {vis_train} trainable params; expected 0.")
    if lm_train_lora == 0:
        raise RuntimeError("No LoRA trainable params on language_model; injection failed.")
    if any(p.requires_grad for p in model.lm_head.parameters()):
        raise RuntimeError("lm_head should be frozen.")
    del added  # ruff: keep silent

    return {
        "model": model,
        "processor": processor,
        "tokenizer": tokenizer,
        "concept_mlp": concept_mlp,
        "new_emb": new_emb,
        "latent_token_ids": [clean_map["latent"]] * cfg.K,
        "new_token_ids": clean_map,
    }


def _splice_new_emb(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    new_emb: torch.Tensor,
    new_token_ids: dict,
) -> torch.Tensor:
    """Replace embedding rows at new-token positions with rows from `new_emb`.

    Args:
        inputs_embeds: [B, L, D] (will be modified in-place via masked write).
        input_ids:     [B, L]
        new_emb:       [3, D] (parameter, fp32). Order: latent, latent_start, latent_end.
        new_token_ids: dict mapping 'latent'/'latent_start'/'latent_end' -> int id.

    Returns:
        Same tensor, with new-token rows overwritten. Gradient flows through new_emb.

    Note: writes go through `index_put` style assignment so the autograd graph
    correctly attaches `new_emb` as a leaf for those positions.
    """
    target_dtype = inputs_embeds.dtype
    for name, emb_idx in (("latent", 0), ("latent_start", 1), ("latent_end", 2)):
        tok_id = new_token_ids[name]
        mask = input_ids == tok_id  # [B, L]
        if not mask.any():
            continue
        row = new_emb[emb_idx].to(dtype=target_dtype)  # [D]
        # masked_scatter_ needs a contiguous source; we want every True position
        # to receive the same row. `inputs_embeds[mask] = row` assigns by
        # broadcasting over the masked rows.
        inputs_embeds = inputs_embeds.clone() if not inputs_embeds.requires_grad else inputs_embeds
        # Use boolean index assignment which preserves the gradient path through `row`.
        inputs_embeds[mask] = row
    return inputs_embeds


def forward_generator(
    model,
    new_emb: nn.Parameter,
    new_token_ids: dict,
    processor,
    images: list[Image.Image],
    cfg,
) -> torch.Tensor:
    """Generator forward: image + K latent slots → h ∈ [B, K, D].

    Builds the q-invariant generator prompt for each image, runs the full VLM
    forward (with the new-token embedding rows spliced in), then gathers the
    K hidden states at the <|latent|> positions from the FINAL layer.

    Args:
        model: PeftModel-wrapped Qwen2_5_VLForConditionalGeneration on CUDA.
        new_emb: [3, D] fp32 nn.Parameter (trainable).
        new_token_ids: dict from build_generator.
        processor: AutoProcessor (tokenizer extended with new tokens).
        images: list[PIL.Image] of length B.
        cfg: ModelConfig (uses cfg.K).

    Returns:
        h: [B, K, D] tensor in the model's compute dtype (bf16) with grad.
    """
    K = cfg.K
    prompts = [_build_generator_prompt(K) for _ in images]
    inputs = processor(text=prompts, images=list(images), return_tensors="pt", padding=True)
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

    input_ids = inputs["input_ids"]  # [B, L]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    mm_token_type_ids = inputs.get("mm_token_type_ids")

    # Build inputs_embeds and splice new_emb in for the 3 new token positions.
    base_embeds = model.get_input_embeddings()(input_ids)  # [B, L, D] bf16
    inputs_embeds = _splice_new_emb(base_embeds, input_ids, new_emb, new_token_ids)

    # Forward pass. Pass BOTH input_ids (for placeholder mask via token ids) and
    # inputs_embeds (with new-token rows spliced in). pixel_values/image_grid_thw
    # cause the model to overwrite <|image_pad|> rows with vision features.
    fwd_kwargs = dict(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    if pixel_values is not None:
        fwd_kwargs["pixel_values"] = pixel_values.to(dtype=model.dtype)
        fwd_kwargs["image_grid_thw"] = image_grid_thw
    if mm_token_type_ids is not None:
        fwd_kwargs["mm_token_type_ids"] = mm_token_type_ids

    outputs = model(**fwd_kwargs)
    last_hidden = outputs.hidden_states[-1]  # [B, L, D]

    # Gather the K positions where input_ids == latent_id (one row per batch entry).
    latent_id = new_token_ids["latent"]
    h_list = []
    for b in range(input_ids.shape[0]):
        positions = (input_ids[b] == latent_id).nonzero(as_tuple=False).squeeze(-1)
        if positions.numel() != K:
            raise RuntimeError(
                f"Expected exactly K={K} latent positions in row {b}, got {positions.numel()}."
            )
        h_list.append(last_hidden[b, positions, :])
    h = torch.stack(h_list, dim=0)  # [B, K, D]
    return h


@torch.no_grad()
def get_v_sem(model, processor, images: list[Image.Image]) -> torch.Tensor:
    """Teacher V_sem: post-merger visual tokens, frozen, no grad.

    Args:
        model: the (peft-wrapped) generator. We use its frozen vision branch.
        processor: AutoProcessor.
        images: list[PIL.Image] of length B.

    Returns:
        v_sem: [B, T_v, D] in bf16 on CUDA. Assumes all images in the batch
        produce the same T_v (typical for fixed-size training inputs); raises
        if they differ. For the round-3 POC, the loader produces same-size
        images per batch, so this is the cheap path.
    """
    # Processor requires `text` even for image-only feature extraction; pass a
    # one-token prompt with a single <|image_pad|> per image. We don't use the
    # text outputs.
    dummy_texts = ["<|image_pad|>"] * len(images)
    inputs = processor(text=dummy_texts, images=list(images), return_tensors="pt")
    pixel_values = inputs["pixel_values"].cuda().to(model.dtype)
    image_grid_thw = inputs["image_grid_thw"].cuda()
    out = model.model.get_image_features(
        pixel_values=pixel_values, image_grid_thw=image_grid_thw
    )
    pool = out.pooler_output  # tuple of [T_v_i, D] tensors, one per image
    if isinstance(pool, torch.Tensor):
        pool = (pool,)
    sizes = {p.shape[0] for p in pool}
    if len(sizes) != 1:
        raise RuntimeError(
            f"V_sem extraction got mixed T_v across batch ({sizes}); "
            "POC loader is expected to send same-size images per batch."
        )
    v_sem = torch.stack(list(pool), dim=0)  # [B, T_v, D]
    return v_sem
