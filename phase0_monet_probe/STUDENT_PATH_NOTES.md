# Student-path forward — short technical note

## What the student-path input looks like
Same chat template as the teacher path, **minus the auxiliary-image
block**. After tokenization, each assistant turn becomes:

```
<|im_start|>assistant <abs_vis_token> <pad>*8 </abs_vis_token>
<observation>...</observation> <|im_end|>
```

i.e. no `<|vision_start|><|image_pad|><|vision_end|>` before the
`<abs_vis_token>` block. The user's question image is still passed via
`pixel_values` (it lives in the user turn, not the assistant turn).

## How latents are produced when no aux image is present
In `modeling_qwen2_5_vl_monet.py:1689-1955` (`latent_mode=True`), the
language model is run segment-by-segment along `latent_pos` =
`<abs_vis_token_pad>` positions. For each pad position `pos`, it sets
`latent_embed = batch_last_hidden_state[b, pos-1, :]` (last token's
hidden state) and forwards one step. With no aux image, the previous
token is just the prior latent slot (or `<abs_vis_token>` for slot 0),
so the **8 latent slots evolve as a Coconut-style hidden-state recurrence
seeded by the question + dialog history** — no image splice, no extra
attention mask.

## HF transformers can drive this
Yes. The vendored `Qwen2_5_VLForConditionalGeneration.forward` already
implements the recurrence; we just feed an aux-image-free prompt and
matching `pixel_values` (question image only), `latent_mode=True`,
`output_latent_embeds=True`. No vLLM runner needed.
