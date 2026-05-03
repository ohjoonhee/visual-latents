"""Interleaved Coconut-style generator: T_blocks alternating text + latent spans.

Per `docs/INTERLEAVED_LATENT_DESIGN.md` §3 and §8.

The build is identical to the parallel method's `vl.model.build_generator`
(same trainable surface: LoRA on language model, `new_emb` for the 3 special
tokens, concept MLP D->D/2->D). What differs is the forward — instead of
emitting K latents in one parallel pass, we run `1 + T_blocks * (k_latent + 1)`
forward passes per example (design §3.3), feeding the prior pass's last hidden
state as the next position's input embedding inside each latent span.

We deliberately do NOT use `past_key_values` between intra-block recurrent
passes (design §3.3): the recurrence-participating activations must remain in
the autograd graph so the trailing reader-NLL gradient can reach LoRA in every
forward pass within the trace. We therefore re-feed the growing prefix on
each pass.

Public API:
    build_interleaved_generator(cfg, dtype) -> dict     # mirrors build_generator
    run_interleaved_forward(model, ..., trace, image, cfg) -> [B=1, K_total, D]
"""

from __future__ import annotations

import torch
from PIL import Image

# Reuse the parallel builder verbatim — same trainable surface (LoRA + new_emb
# + concept MLP). The interleaved variant differs only in the forward routine.
from ..model import _splice_new_emb, build_generator
from .traces import Trace


def build_interleaved_generator(cfg, dtype: torch.dtype = torch.bfloat16) -> dict:
    """Construct the interleaved generator.

    Identical to `vl.model.build_generator` — the trainable surface is the
    same (LoRA + new_emb + concept_mlp). We re-export so the interleaved
    trainer can call a name from its own module without importing across
    the parallel/interleaved boundary at the call site.
    """
    return build_generator(cfg, dtype=dtype)


# ---------------------------------------------------------------------------
# Helpers for building / extending the running prefix
# ---------------------------------------------------------------------------
def _tokenize(tokenizer, text: str, device) -> torch.Tensor:
    """Tokenize a raw string to a [1, L] tensor (no special-token addition)."""
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    return ids.to(device)


def _embed_text_ids(model, ids: torch.Tensor, new_emb, new_token_ids) -> torch.Tensor:
    """Embed `ids` and splice new_emb at any new-token positions.

    Used for the initial prompt and for every text-mode segment that follows
    a `<|latent_end|>`. The splice ensures gradient flows into new_emb when
    `<|latent_start|>` / `<|latent_end|>` appear in the segment.
    """
    base = model.get_input_embeddings()(ids)  # [1, L, D] bf16
    spliced = _splice_new_emb(base, ids, new_emb, new_token_ids)
    return spliced


def _full_forward(
    model,
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
) -> torch.Tensor:
    """One full VLM forward pass; returns hidden_states[-1] of shape [1, L, D].

    Per design §3.3 we use `use_cache=False` and re-feed the full prefix each
    call. `pixel_values` + `image_grid_thw` are passed every call so the model
    re-runs the (frozen, cheap) vision merger and re-overwrites the
    `<|image_pad|>` rows with vision features. (Probe /tmp/probe_qwen_inputs_embeds.py
    Test 1 + Test 4 confirms this works.)
    """
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
    out = model(**fwd_kwargs)
    return out.hidden_states[-1]  # [1, L, D]


def run_interleaved_forward(
    model,
    new_emb,
    new_token_ids: dict,
    processor,
    image: Image.Image,
    trace: Trace,
    *,
    detach_recurrence_input: bool = False,
) -> torch.Tensor:
    """Run the full interleaved forward pass and return harvested latents.

    Returns:
        h: [1, K_total, D] tensor with grad. Each row is the LAST hidden
        state harvested from one latent position (across all blocks). Order
        matches the trace order (block 1 latents first, then block 2, ...).

    Per design §8 pseudocode:
      1. Tokenize prompt + question + first <|latent_start|>.
      2. Forward → get last hidden state h_prev.
      3. For each latent in block k:
           - append h_prev as a new position in inputs_embeds, sentinel
             <|latent|> id in input_ids, 1 in attention_mask.
           - re-forward the whole growing prefix (no past_kv, no detach).
           - new last hidden state -> h_prev; append to all_h.
      4. Append <|latent_end|> + next text segment + next <|latent_start|>;
         forward; capture last hidden state.
      5. Repeat 3-4 for each block.
      6. (We do NOT run the trailing answer text; the reader (§4.2 (i)) does
         not consume it and there is no text-NLL.)
    """
    device = model.device
    tokenizer = processor.tokenizer
    latent_id = new_token_ids["latent"]

    # ---- 1. Build the initial prefix: prompt + first <|latent_start|> ----
    # The prompt_text already ends with "<|im_start|>assistant\n". The first
    # text segment + <|latent_start|> opens block 1.
    init_text = trace.prompt_text + trace.text_segments[0] + "<|latent_start|>"
    init_inputs = processor(
        text=[init_text], images=[image], return_tensors="pt", padding=False
    )
    init_inputs = {
        k: (v.cuda() if torch.is_tensor(v) else v) for k, v in init_inputs.items()
    }
    input_ids = init_inputs["input_ids"]  # [1, L0]
    attention_mask = init_inputs["attention_mask"]
    pixel_values = init_inputs.get("pixel_values")
    image_grid_thw = init_inputs.get("image_grid_thw")

    # Build inputs_embeds for the initial prefix (with new_emb spliced in at
    # the <|latent_start|> position). Vision features are merged inside
    # _full_forward when pixel_values is passed.
    inputs_embeds = _embed_text_ids(model, input_ids, new_emb, new_token_ids)

    # ---- 2. First forward: prefix → last hidden state h_prev ----
    last_hidden = _full_forward(
        model,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    h_prev = last_hidden[:, -1, :]  # [1, D]

    all_h: list[torch.Tensor] = []  # accumulator for K_total latent hidden states

    # Iterate over blocks
    n_blocks = len(trace.latent_block_sizes)
    for b_idx, k_latent in enumerate(trace.latent_block_sizes):
        # ---- 3. k_latent recurrent passes inside the current block ----
        for _ in range(k_latent):
            # Append h_prev as a new position. input_ids gets a sentinel
            # <|latent|> id (so attention_mask line up and the merger ignores
            # it — image_token_id is different). inputs_embeds gets h_prev at
            # the new position (NOT new_emb[0]; that is the parallel-method's
            # mechanism, not Coconut's).
            new_id = torch.full(
                (1, 1), latent_id, dtype=torch.long, device=device
            )
            input_ids = torch.cat([input_ids, new_id], dim=1)
            # Round-2 diagnostic C2: optionally detach the recurrence-input
            # embedding. This BREAKS gradient flow through the recurrence
            # chain (design §3.3), so use only for the norm-collapse probe.
            recurrence_emb = h_prev.detach() if detach_recurrence_input else h_prev
            inputs_embeds = torch.cat(
                [inputs_embeds, recurrence_emb.unsqueeze(1).to(inputs_embeds.dtype)],
                dim=1,
            )
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)],
                dim=1,
            )

            last_hidden = _full_forward(
                model,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            h_prev = last_hidden[:, -1, :]  # [1, D]
            all_h.append(h_prev)

        # ---- 4. Transition: <|latent_end|> + next text seg (+ next <|latent_start|>) ----
        is_last_block = b_idx == n_blocks - 1
        if is_last_block:
            # No further block; we don't actually need a transition pass since
            # we have all K_total latents. (Design §8 also skips the trailing
            # SUFFIX_TEXT pass.) Break.
            break

        next_seg = trace.text_segments[b_idx + 1]
        transition_text = "<|latent_end|>" + next_seg + "<|latent_start|>"
        trans_ids = _tokenize(tokenizer, transition_text, device)  # [1, L_t]
        trans_embeds = _embed_text_ids(model, trans_ids, new_emb, new_token_ids)

        input_ids = torch.cat([input_ids, trans_ids], dim=1)
        inputs_embeds = torch.cat([inputs_embeds, trans_embeds], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(trans_ids)],
            dim=1,
        )

        last_hidden = _full_forward(
            model,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        h_prev = last_hidden[:, -1, :]  # [1, D]
        # h_prev here is the hidden state at the <|latent_start|> position of
        # the next block — it becomes the input embedding for the first
        # <|latent|> position via the recurrence in the next loop iteration.

    # ---- 5. Stack into [1, K_total, D] ----
    h = torch.stack(all_h, dim=1)  # [1, K_total, D]
    return h
