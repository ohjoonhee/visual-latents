# MASK_VALIDATION — Phase 1.5b 4D-attention mask vs upstream Monet

Read-only review while training (PID 531160) is in progress. No code modified.

## Section 1: Upstream source

- Repo: https://github.com/NOVAglow646/Monet (CVPR 2026, paper arXiv:2511.21395)
- Raw: `https://raw.githubusercontent.com/NOVAglow646/Monet/main/src/utils.py`
- Raw model: `https://raw.githubusercontent.com/NOVAglow646/Monet/main/monet_qwen_model/modeling_qwen2_5_vl_monet.py`

### Upstream `build_4d_attn_wo_helper_images` (utils.py:697-746) — verbatim core

```python
def build_4d_attn_wo_helper_images(input_ids, pad_mask, token_ids, mask_latent: bool = False):
    input_ids = input_ids.cpu(); pad_mask = pad_mask.cpu()
    B, L = input_ids.shape
    causal = torch.tril(torch.ones((L, L), dtype=torch.bool))
    valid = pad_mask.bool()
    allowed = causal.unsqueeze(0).clone().repeat(B, 1, 1)         # [B,L,L]
    for b in range(B):
        allowed[b] &= valid[b].unsqueeze(0)   # mask keys
        allowed[b] &= valid[b].unsqueeze(1)   # mask queries
    for b in range(B):
        segs = find_segments_1d_wo_helper_images(input_ids[b], token_ids)
        for (A_idx, O_idx) in segs:
            if A_idx.numel() and mask_latent:
                # rows r are "subsequent" if any a in A_idx satisfies a < r
                rows_to_block = (r_idx.unsqueeze(0) >= A_idx.unsqueeze(1)).any(dim=0)
                allowed[b][rows_to_block_idx[:, None], A_idx] = False
    return allowed.unsqueeze(1)        # bool [B,1,L,L], no return_type kwarg
```

Caller (`src/main.py`, `collate_fn_sft_stage3`): wraps in `{"full_attention": attn_mask_4d}` and passes as `attention_mask_4d` (no dtype cast).

### Upstream eager_attention (monet_qwen_model/modeling_qwen2_5_vl_monet.py)

```python
attn_weights = torch.matmul(query, key_states.transpose(2,3)) * scaling
if attention_mask is not None:
    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask         # raw add, no bool→float
```

The dict-unpack site (`Qwen2_5_VLTextModel.forward`): if `attention_mask` is already a dict, it bypasses HF's `create_causal_mask` and forwards each layer's tensor straight to attention. **There is no bool→additive conversion in the main path.** (A bool→additive conversion exists only inside an attention-loss debug helper `_extract_mask_rows`, line 984, not on the hot path.)

## Section 2: Comparison vs Phase 1.5b reimplementation

| Rule | Upstream `_wo_helper_images` | Phase 1.5b `mask_utils.build_4d_attn` | Match? | Concern |
|---|---|---|---|---|
| Return shape | `[B,1,L,L]` | `[B,1,L,L]` | YES | – |
| Return dtype default | bool only (no `return_type` kw) | `additive` (bf16, `finfo.min` for blocked) | DIFFERENT | Phase 1.5b is *safer*: eager_attention does raw add; bool gives +1 to allowed rather than -inf to blocked, which is a soft penalty not a hard block. Phase 1.5b is materially **more correct** here. |
| Causal lower-tri | `tril` | `tril` | YES | – |
| Padding (keys AND queries) | both directions | both directions | YES | – |
| `mask_latent` branch | rows `>=` slot blocked from slot keys | rows `>` slot blocked from slot keys, slot diagonal preserved | DIFFERS in `>=` vs `>` | Phase 1.5b runs with `mask_latent=False`, so harmless for this run. |
| Cross-slot isolation (slot_i ↛ slot_j) | NOT present upstream | `latent_cross_isolate=True` blocks off-diagonal slot/slot | DELIBERATE EXTENSION | This is the Phase 1.5b hypothesis, by design. Documented in mask_utils.py docstring. |
| Observation-block rules | `O_idx` parsed but unused in wo-helper variant | n/a (no observation tokens in Phase 1.5b layout) | n/a | – |
| Question-image rules | n/a in wo-helper variant | n/a | YES | – |

## Section 3: latent_mode=True path mask consumption

`phase0_monet_probe/monet_model/modeling_qwen2_5_vl_monet.py`:

- **Pre-answer prefix** (line 1771-1786): consumes `attention_mask_4d['full_attention'][:, :, :ans_start, :ans_start]` — uses 4D mask.
- **Post-latent text chunk** (line 1825-1845): consumes `attention_mask_4d['full_attention'][:, :, q0:q1, :k1]` — uses 4D mask.
- **Per-latent-slot recurrent step** (line 1922-1934):
  ```
  step_out = self.language_model(
      ...,
      attention_mask=attention_mask[b: b+1][:, :pos+1],   # 1D mask, NOT 4D
      past_key_values=past_kv,
      cache_position=torch.tensor([pos], ...),
      ...
  )
  ```
  Hard-coded 1D `attention_mask`. **`attention_mask_4d` is not consumed for the per-slot KV-cache forward.** The Phase 1.5b agent's caveat is correct.

Implication: cross-slot isolation is enforced on (a) the pre-answer prefix, (b) the post-latent text chunk, and (c) the non-latent CE forward (line 2021 passes `attention_mask_4d` directly). It is **NOT** enforced on the per-slot recurrent latent forward, where each slot t_i's query against past KV (containing slot t_{<i}) sees them via the 1D path's HF-built causal+pad mask. So slot t_i CAN attend to slot t_{<i} in the latent forward step that produces its hidden state — exactly the slice of computation the experiment most cares about.

## Section 4: Verdict

**FAITHFUL** to upstream `build_4d_attn_wo_helper_images` for the rules it shares (causal, padding, mask_latent semantics in spirit), with one **deliberate, documented extension** (`latent_cross_isolate=True`) that is the experimental hypothesis itself.

The implementation is *more correct* than upstream's additive return path (upstream's `return_type='additive'` multiplies by `1e-6` which is essentially a no-op; Phase 1.5b uses `finfo(bf16).min`, a real -inf-equivalent).

There is one **material limitation, not a bug**: cross-slot isolation does NOT apply during the per-slot recurrent forward (line 1922-1934) because that path hard-codes the 1D `attention_mask`. This is an upstream-model limitation, not a Phase 1.5b implementation defect.

Recommendation: **continue the run, interpret with documented caveat.** The cross-slot-isolation effect is partially enforced (prefix + post-latent text + CE forward) but not on the slot-producing forward itself. If Phase 1.5b shows a behavioral effect, it is an under-estimate of what full enforcement would yield; if it shows no effect, the per-slot path may be masking it.

## Section 5: Proposed remediation (not applied)

If a follow-up wants full cross-slot isolation in the latent forward, change line 1926 of `phase0_monet_probe/monet_model/modeling_qwen2_5_vl_monet.py` from:

```python
attention_mask=attention_mask[b: b+1][:, :pos+1],
```

to (sketch — vendored model edit, not Phase 1.5b code):

```python
if attention_mask_4d is not None:
    step_mask = {'full_attention': attention_mask_4d['full_attention'][b:b+1, :, pos:pos+1, :pos+1]}
else:
    step_mask = attention_mask[b:b+1][:, :pos+1]
...
attention_mask=step_mask,
```

This requires the model to accept dict masks on the single-step path (it already does in the prefix/text-chunk paths). Do NOT apply mid-run; queue for Phase 1.5c.
