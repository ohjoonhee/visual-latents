# Monet — Method Report

**Operational reference for paper↔code alignment, code-only details, and hook points for controlled variation (e.g. VICReg).**

| Field | Value |
|---|---|
| Paper | *Monet: Reasoning in Latent Visual Space Beyond Images and Language* |
| Authors | Qixun Wang, Yang Shi, Yifei Wang, Yuanxing Zhang, Pengfei Wan, Kun Gai, Xianghua Ying, Yisen Wang (Peking U. + Kuaishou) |
| arXiv | [2511.21395](https://arxiv.org/abs/2511.21395) (v1 26 Nov 2025; v2 27 Nov 2025) |
| Venue | CVPR 2026 (accepted) |
| Repo | https://github.com/NOVAglow646/Monet |
| License | Top-level **unspecified** (only `RL/LICENSE` = Apache-2.0 inherited from EasyR1/verl). Verify before redistribution. |
| HF model — SFT | https://huggingface.co/NOVAglow646/Monet-SFT-7B (K=8) |
| HF model — RL | https://huggingface.co/NOVAglow646/Monet-7B (K=10) |
| HF dataset | https://huggingface.co/datasets/NOVAglow646/Monet-SFT-125K |
| Base model | Qwen2.5-VL-7B-Instruct (vision encoder frozen) |
| Local artefacts | `phase0_monet_probe/monet_model/{modeling_qwen2_5_vl_monet.py,apply_qwen2_5_monet.py}`, `cluster/trainer_monet_stage2.py` (our port), `cluster/mask_utils.py` (our 4D mask reimpl) |

---

## 0. Executive summary

Monet trains an MLLM to **think with latent embeddings** instead of images or words at inference time. The model autoregressively emits a block of `K` continuous vectors (the **latents**) inside its own decoder stream; each latent's input embedding is the previous step's last-layer hidden state (Coconut-style hidden-state feedback). The block sits between `<latent>` / `</latent>` brackets and is consumed by subsequent text tokens — no auxiliary images, no tool calls.

The contribution is a **3-stage SFT curriculum plus VLPO RL** that produces useful latents at inference time:

| Stage | Aux images? | Latents? | Loss | What it teaches |
|---|---|---|---|---|
| 1 (warm-up) | yes (interleaved) | no (still text-only CoT) | NTP with emphasized-CE on `<observation>` tokens | Format — emit `<observation>...</observation>`, follow the CoT pattern. |
| 2 (target latents) | yes (only seen by latents via 4D mask) | yes (K slots per aux-image position) | `NTP + α·L_align-obs`, with a **latent-only-BP surrogate** that routes the alignment gradient *only* into the latent-generation graph | Make latents carry the same per-layer information as the encoder-grounded teacher's observation tokens. |
| 3 (inference-ready) | **no** (dropped from student) | yes | `NTP + β·L_align-latent` aligned to Stage-2 teacher latents (precomputed) | Reproduce Stage-2 latents from question context alone. |
| RL (VLPO) | no | yes (K=10) | Group-relative policy optimization with a **Gaussian importance ratio** on latent slots; reward = accuracy + format | Refine. |

The four named technical contributions, in order of how much the ablations matter:

1. **Latent-only backprop** (`compute_latents_only_loss`, `src/trainer.py:11-42`) — paper Eq. 9/10. Removing it crashes V★ from 82.20 → 46.07 (Table 2). This is the load-bearing trick. *§5*.
2. **All-layer alignment loss** (not just final-layer). Paper Sec. 3.2. *§5.3*.
3. **4D attention mask** so aux images are visible *only* to latent embeddings (not to subsequent text). `src/utils.py::build_4d_attn`. *§3.3*.
4. **VLPO**: PPO/GRPO variant with a Gaussian importance weight `exp(-||h_old - h_θ||² / 2σ²)` for latent slots (which have no discrete `logπ` to ratio). *§7*.

A frozen-reader internal probe (our `phase0_monet_probe`) finds that on the released checkpoint **Stage 2 latents are healthy** (mean off-diag cos ≈ 0.38, utility +2.2 nat to vanilla Qwen reader) but **Stage 3 collapses** (mean cos 0.87, positions 4–7 collinear, utility ≈ 0). This is consistent with Stage 3 being supervised against a *latent-of-a-latent* with no image grounding. See *§13*.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| `K` | Number of latent slots per `<latent>...</latent>` block. K = 8 in `Monet-SFT-7B`; K = 10 in `Monet-7B` (post-VLPO). |
| `h^{(l)}_t` | Hidden state at layer `l`, sequence position `t`. |
| `ĥ^{(i,l)}_{latent}` | Student's `l`-th layer hidden state at latent position `i`. |
| `h*^{(i,l)}_{obs}` | Frozen teacher's `l`-th layer hidden state at observation position `i`. The teacher receives the same prompt **with aux images present**. |
| `α`, `β` | Stage 2 / Stage 3 alignment weight. Both = 2.0 in paper. |
| `σ` | VLPO Gaussian std for latent importance ratio. σ = 10.0. |
| Observation token | A token inside `<observation>...</observation>` that describes what the model "sees" in the aux image. CE loss is upweighted on these via `ce_emphasize_factor`. |

---

## 2. End-to-end pipeline overview

```
                       ┌────────────────────────────────────────────┐
                       │  Monet-SFT-125K  (118K Visual-CoT + …)      │
                       │   each ex: question, image, CoT with        │
                       │   <observation>…</observation> and          │
                       │   <abs_vis_token>…</abs_vis_token> markers  │
                       └────────────────────────────────────────────┘
                                          │
              ┌───────────────────────────┴──────────────────────────────┐
              ▼                                                          │
  ┌───────────────────────────┐                                          │
  │ Stage 1: warm-up           │                                         │
  │  loss = NTP w/ ce_emph 2.0 │  (8×H100, 4 epochs, bs=1, gas=16)       │
  │  init from Qwen2.5-VL-7B   │                                         │
  └───────────────────────────┘                                          │
              │                                                          │
              ▼                                                          │
  ┌───────────────────────────┐    precompute_teacher_reps.py            │
  │ Stage 2: target latents    │ ◀── caches teacher hidden states         │
  │  loss = NTP + α·align_obs  │     (all layers, observation tokens,    │
  │   via latent-only-BP       │      from Stage-1 ckpt w/ aux images)   │
  │  init from Stage 1 ckpt    │                                         │
  └───────────────────────────┘                                          │
              │                                                          │
              │  this is `Monet-SFT-7B` on HF (K=8)                      │
              ▼                                                          │
  ┌───────────────────────────┐    precompute_teacher_latents.py         │
  │ Stage 3: drop aux images   │ ◀── caches Stage-2 teacher's latent      │
  │  loss = NTP + β·align_lat  │     hidden states (the "latent-of-a-    │
  │  init from Stage 1 ckpt    │      latent" supervision target)         │
  └───────────────────────────┘                                          │
              │                                                          │
              ▼                                                          │
  ┌──────────────────────────────────┐                                   │
  │ RL (VLPO) on Thyme-RL (3.2K)     │   K = 10, σ = 10.0, KL = 0.01     │
  │  reward = accuracy + format      │   `verl` framework fork           │
  │  importance ratio: Gaussian      │                                   │
  └──────────────────────────────────┘                                   │
              │                                                          │
              ▼                                                          ▼
                       ┌────────────────────────────────────┐
                       │  Inference: custom vLLM gpu_model_  │
                       │  runner; on <latent>, feed previous │
                       │  last-hidden-state as next input    │
                       │  embedding for K steps, then </>    │
                       └────────────────────────────────────┘
```

**Two conda environments are mandatory.** The SFT stack (`monet`, Python 3.10) and the RL stack (`easyr1`, Python 3.11) have incompatible `transformers` + `vllm` pins; switching is not optional. The vLLM-inference patch ships separately under `inference/`. See *§3.6*.

**Module-level monkey-patching.** Both `monet_qwen_model/apply_qwen2_5_monet.py` and `inference/apply_vllm_monet.py` (and our `cluster/trainer_monet_stage2.py:60-89`) mutate `sys.modules` *before* any `import transformers`. Any later `from transformers.models.qwen2_5_vl import …` silently gets the Monet variant. This is brittle in shared processes; debug accordingly.

---

## 3. Architecture (modified Qwen2.5-VL)

All architectural changes live in **`modeling_qwen2_5_vl_monet.py`** (vendored at `phase0_monet_probe/monet_model/`). The vanilla Qwen2.5-VL class is preserved; the modifications are additive — they activate only when specific forward kwargs are present.

### 3.1 Output dataclasses

Two `ModelOutput` subclasses are extended:

`Qwen2_5_VLModelOutputWithPast` (model-level, `modeling_qwen2_5_vl_monet.py:733-752`) adds:

```python
ce_patch_pos:   Optional[List[List[int]]]      # positions of latent tokens in input_ids
ce_patch_vec:   Optional[List[torch.Tensor]]   # the K latent embeddings, batched as list-of-tensors
alignment_loss: Optional[torch.FloatTensor]    # accumulated alignment loss (latent-mode only)
```

`Qwen2_5_VLCausalLMOutputWithPast` (CausalLM-level, `modeling_qwen2_5_vl_monet.py:2079-2106`) further adds `alignment_poss`, `latent_embeds` (extracted last-layer states at latent positions when `output_latent_embeds=True`), `mean_emphasize_acc`, `loss_dict` (e.g. `{'ce': ..., 'alignment': ...}`).

**These extra fields are why our `compute_latents_only_loss` port works**: `outputs.ce_patch_vec` from the latent forward is a list of leaf-ish tensors that have a fresh autograd graph to the CE forward; passing them as `inputs` to `torch.autograd.grad` is well-defined.

### 3.2 Forward signature additions

Both `Qwen2_5_VLModel.forward` and `Qwen2_5_VLForConditionalGeneration.forward` accept (all optional, all default to `None` or `[]`):

| Kwarg | Type | Used in | What it does |
|---|---|---|---|
| `latent_mode` | `bool` | latent forward | Switches into the segment-wise autoregressive latent-generation loop. |
| `alignment_poss` | `List[List[int]]` | latent mode (S2/S3) | Per-batch positions where alignment loss is computed (observation tokens in S2; latent positions in S3). |
| `teacher_hidden_states_for_alignment` | `List[Tensor]` | latent mode | Pre-loaded teacher reps; shape `[L_layers, N_align, H]` for all-layer or `[N_align, H]` / `[H]` for last-layer. |
| `ce_patch_pos` | `List[List[int]]` | non-latent CE forward | Where in `inputs_embeds` to splice latents. |
| `ce_patch_vec` | `List[Tensor]` | non-latent CE forward | The latents (from the prior latent forward) to splice. |
| `ce_emphasize_factor` | `float` | both | CE-loss weight multiplier at `ce_emphasize_poss` (default 1.0). |
| `ce_emphasize_poss` | `List[List[int]]` | both | Token positions to upweight in the CE loss (= observation-token positions). |
| `loss_type` | `List[str]` | both | Subset of `{'ce', 'alignment'}`. Empty list = no loss (used for the latent-only "generate latents" forward). |
| `attention_mask_4d` | `Dict[str, Tensor]` | both | Pre-built 4D mask in `{"full_attention": [B,1,L,L]}` form. Bypasses the standard causal mask. |
| `output_latent_embeds` | `bool` | latent mode | Returns extracted last-hidden-state at `alignment_poss` (for probing/inference latent extraction). |
| `compute_emphasize_acc` | `bool` | non-latent CE | Per-token accuracy logging on emphasized positions. |

### 3.3 The latent forward (`latent_mode=True`)

`modeling_qwen2_5_vl_monet.py:1720-1973`. **Batch size 1 only is tested upstream** (in-code warning at L1745). The forward iterates per-sample:

1. Find latent positions: `(input_ids[b] == config.latent_token_id).nonzero()` (`L1735-1738`).
2. Forward the **pre-answer segment** (everything before `<|im_start|>assistant`), saving `past_key_values` (`L1771-1788`).
3. For each gap between latent positions:
   a. Forward the **text/image sub-segment**, optionally computing alignment on all observation tokens in this segment by stacking student hidden states across all layers (`L1851-1870`).
   b. Forward the **image tail** (continuous `<|image_pad|>` tokens before the next latent block) *without* the 4D mask, to save memory (`L1878-1893`). Alignment pointers skip indices inside the image tail.
4. For each latent position `pos`:
   - If `pos == 0`: input embedding = `latent_init_embedding` (an attribute set on the model) or **zero** (`L1906-1913`). This is rare in practice — latent positions are never first in Monet's training data.
   - Else: input embedding = **previous token's last-layer hidden state**, cloned in train / detached in eval (`L1914-1916`). This is the *hidden-state feedback*. The position is appended to `ce_patch_pos[b]` and the embedding to `ce_patch_vec[b]`.
   - Run a single-step `language_model` forward (`L1922-1934`). KV cache is propagated; `output_hidden_states=True` (needed for all-layer alignment in S3).
   - In S3 (latent-position alignment), accumulate `alignment_loss(teacher_hidden_states_for_alignment[b][:, align_ptr, :], step_out.hidden_states)` (`L1939-1949`).

**Three things to know about the latent forward.**

- The latent's input embedding is **detached** from the autograd graph that *produced* the previous hidden state (`.detach()` at L1924). So gradient does not flow latent→latent through input-embedding feedback; only the model parameters via the language model's forward propagate the signal. The graph used by `compute_latents_only_loss` is the *one-step LM forward* that processes this detached latent embedding.
- `ce_patch_vec[b]` holds the latent embeddings as they came out of the previous step (with grad attached to all upstream params). These are the leaves that `torch.autograd.grad` will be called on later.
- Past-KV is **not** detached across latent steps within a sample, so each latent's forward sees the full history.

### 3.4 The non-latent CE forward (`latent_mode=False`)

`modeling_qwen2_5_vl_monet.py:1985-2070`. This is a near-vanilla Qwen2.5-VL forward with two extras:

```python
# L2006-2013: splice latents into inputs_embeds
if ce_patch_pos is not None and ce_patch_vec is not None:
    for b in range(len(ce_patch_pos)):
        pos_list = ce_patch_pos[b]
        if not pos_list:
            continue
        vecs = ce_patch_vec[b].to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds[b, torch.tensor(pos_list, device=…, dtype=torch.long), :] = vecs
```

```python
# L2048-2057: all-layer cosine alignment if 'alignment' in loss_type
if "alignment" in kwargs.get('loss_type', {}):
    all_student_hidden_states = torch.stack(outputs.hidden_states, dim=0)
    for b in range(batch_size):
        student_hidden_states = all_student_hidden_states[:, b, alignment_poss[b], :]
        total_align_loss += alignment_loss(
            teacher_hidden_states_for_alignment[b],
            student_hidden_states,
        )
    total_align_loss /= batch_size
```

**This splice is the only place where `ce_patch_vec` is graph-connected to the alignment loss.** When the trainer later calls `torch.autograd.grad(L_align, ce_patch_vec)` (inside `compute_latents_only_loss`), the path is: `ce_patch_vec → inputs_embeds → language_model → hidden_states → alignment_loss`. Without this splice, `ce_patch_vec` would be a leaf with no downstream consumer, and `allow_unused=True` would silently return `None` → zero grad → no-op. (Our smoke gate `[devfix-check] latent_only_connected=True` exists exactly to catch this regression.)

Weighted CE for emphasized tokens lives at `L2273-2304`:

```python
use_weight = (ce_emphasize_poss is not None and isinstance(ce_emphasize_poss, (list, tuple))
              and len(ce_emphasize_poss) > 0 and float(ce_emphasize_factor) != 1.0)
if use_weight:
    ce_flat = F.cross_entropy(logits_flat, shift_labels_flat, reduction='none', ignore_index=-100)
    weight = torch.ones_like(ce)
    for b, poss in enumerate(ce_emphasize_poss):
        if not poss: continue
        weight[b, torch.tensor(poss) - 1] = float(ce_emphasize_factor)
    valid = (shift_labels != -100).float()
    loss = (ce * weight * valid).sum() / (weight * valid).sum().clamp_min(1.0)
else:
    loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
```

Note `poss - 1`: weights are placed on the **predictor** of the emphasized token (shifted labels).

### 3.5 Custom 4D attention mask

Upstream `src/utils.py::build_4d_attn` is heavy (≈ 150 lines, ten boolean flags). It builds a `[B, 1, L, L]` boolean (or additive `large_neg`) mask that, *on top of standard causal*, enforces a directed-acyclic information flow:

- The **question image** is visible to everyone (default).
- An **aux image** in step `i` (between `<|vision_start|>…<|vision_end|>`) is visible **only to the latent block in step `i`** (between `<abs_vis_token>…</abs_vis_token>`). It is hidden from observation tokens, downstream text, and other steps.
- The **latent block** in step `i` is causal w.r.t. itself; it sees the question image, prior latent blocks (if `latent_can_see_all_previous=True`, the default), and its own aux image.
- **Observation tokens** are causal w.r.t. themselves; depending on flags they can see only image tokens, only latent tokens, only question+latents, etc. Default Monet usage: they attend everything causal (no `observation_tokens_only_see_*` flag on).

The "wo_helper_images" variant (`build_4d_attn_wo_helper_images`, no aux images present) is what Stage 3 + inference use; it only enforces `mask_latent` (optional: hide latents from subsequent text).

**Our cluster reimplementation** at `cluster/mask_utils.py::build_monet_4d_attn` is simpler — Phase 1.5b data has no helper images and no `<observation>` blocks, so we use the `wo_helper_images` path with `latent_cross_isolate=True, mask_latent=False`. The mask test in `phase1_5b_attn/MASK_VALIDATION.md` validated equivalence on tested cases. *This is one of our three named code-level deviations* (cluster trainer header `trainer_monet_stage2.py:30-46`).

### 3.6 Custom generation

Vanilla Qwen2.5-VL inherits `generate` from `GenerationMixin`. Monet overrides only `prepare_inputs_for_generation` (`modeling_qwen2_5_vl_monet.py:2369-2437`):
- Recomputes 4D rope deltas only on pre-fill, caches them, returns 4-D position IDs (text-only positions prepended to 3D vision positions).
- **Nullifies pixel_values during decoding** (`if cache_position[0] != 0: model_inputs["pixel_values"] = None`).

Critically, the model has **no special handling to *generate* latent tokens itself** — during HF generation, `<latent>` is just a token. To actually run latent decoding you must use the vLLM patch (see *§9*). This is why our internal-probe path uses `latent_mode=True` + `output_latent_embeds=True` rather than `generate`.

---

## 4. Data pipeline

### 4.1 Special tokens

Added in `src/main.py::run` via `tokenizer.add_special_tokens`:

| Token | Role |
|---|---|
| `<abs_vis_token>` | Opens a latent block (one per aux image in source CoT). |
| `</abs_vis_token>` | Closes a latent block. |
| `<abs_vis_token_pad>` | Padding inside the latent block. There are exactly `K` of these between the open/close brackets in training data. |
| `<observation>` / `</observation>` | Wraps text that "describes what's in the aux image"; used as CE-emphasized positions and (optionally) Stage-2 alignment positions. |
| `<latent>` / `</latent>` | **Inference-time** rename of `<abs_vis_token>` / `</abs_vis_token>`. The inference example regex-replaces in displayed output. Token IDs hard-coded: 151666 / 151667. |

Hard-coded token IDs `LATENT_START_ID=151666` and `LATENT_END_ID=151667` are read from env vars by both the model (`modeling_qwen2_5_vl_monet.py`) and the vLLM runner; **they must match the trained tokenizer**.

### 4.2 Training-data preprocessing

The pipeline is in `src/utils.py` + `src/task.py`. Key helpers (all verbatim in our notes):

- `replace_latent_placeholder_with_img_pad` — in **teacher** forward, replaces every `<abs_vis_token></abs_vis_token>` in assistant turns back into `<|vision_start|><|image_pad|><|vision_end|>` so the teacher sees the actual aux image.
- `replace_img_pad_with_latent_pad` — in **student** forward, replaces every `<|vision_start|><|image_pad|><|vision_end|>` in assistant turns with `<abs_vis_token>{<abs_vis_token_pad> × K}</abs_vis_token>` so the student has K latent slots where the aux image used to be.
- `strip_observation_and_track_retokenized` — strips `<observation>...</observation>` from the visible text but tracks the *retokenized* positions of where the observation content lay, so CE-emphasize and alignment-position bookkeeping survive de-tagging.
- `generate_labels_after_multi_token_start` — sets labels = -100 for everything up to and including the first `<|im_start|>assistant` subsequence (only train on the assistant turn).
- `find_segments_1d` / `find_segments_1d_wo_helper_images` — segments the input_ids into `(I_idx, A_idx, O_blocks)` triplets per reasoning step (I = aux image tokens, A = latent block, O = observation blocks). Used by `build_4d_attn`.
- `resize_by_token_budget` — clamps total image tokens (across all images in the example) to budget. Stage 1/3 budget = `2000 × 28²`; Stage 2 budget = `1500 × 28²`.

### 4.3 Datasets

**Monet-SFT-125K** (paper Table 1):

| Dataset | Domain | Size |
|---|---|---|
| Visual-CoT | Multi-domain (118K) | 118.6K |
| CogCoM | Real-world + chart | 0.5K |
| ReFocus | Chart | 0.4K |
| Zebra-CoT visual | Visual search | 2.7K |
| Zebra-CoT count | Counting | 2.9K |
| Zebra-CoT geometry | Geometry | 0.1K |
| **Total** | | **125.2K** |

All six are loaded into Stage 1, 2, and 3 (same multi-data list in every script).

**RL data**: 3.2K subset of [Thyme-RL](https://huggingface.co/datasets/thyme-rl), one epoch. **Not** Monet-SFT-125K. Reward judging uses a rule-based judge (Gemini 2.5 Pro by default per `RL/examples/config_monet.yaml`).

---

## 5. Stage 2 — the load-bearing stage

### 5.1 Paper text

> *"During Stage 2 training, both the teacher and student models are initialized from the Stage 1 checkpoint. … The teacher processes auxiliary images alongside text; the student processes the same prompt but with auxiliary images replaced by `K` latent slots. To make the latents carry the same per-layer information as the teacher's observation tokens, we add an alignment loss (Eq. 1) over all transformer layers."* — Sec. 3.2

**Stage-2 objective** (Eq. 3):

$$\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{NTP}} + \alpha \cdot \mathcal{L}_{\text{align-obs}}, \quad \alpha = 2.0$$

**Alignment-obs loss** (Eq. 1):

$$\mathcal{L}_{\text{align-obs}} = \frac{1}{N}\sum_i \sum_l \big[1 - \cos\big(\text{stop\_grad}(h^{*\,(i,l)}_{\text{obs}}),\; \hat h^{(i,l)}_{\text{obs}}\big)\big]$$

where the sum is over **observation-token positions `i`** and **all transformer layers `l`**, and the teacher hidden state is `.detach()`'d.

### 5.2 The latent-only backprop trick

**Paper Eq. 9** (the loss actually optimized):

$$\mathcal{L}'_{\text{align-obs}} = \frac{1}{N}\sum_i \text{stop\_grad}\!\bigg(\frac{\partial \mathcal{L}_{\text{align-obs}}}{\partial \hat h^{(i,L)}_{\text{latent}}}\bigg)^{\!\top}\; \hat h^{(i,L)}_{\text{latent}}$$

**Paper Eq. 10** (resulting gradient flow):

$$\frac{\partial \mathcal{L}'_{\text{align-obs}}}{\partial \theta} = \bigg(\frac{\partial \mathcal{L}_{\text{align-obs}}}{\partial \hat h^{(i,L)}_{\text{latent}}}\bigg)\;\bigg(\frac{\partial \hat h^{(i,L)}_{\text{latent}}}{\partial \theta}\bigg)$$

> *"By differentiating this surrogate loss, gradients flow only through the generated latent embeddings to the model parameters."*

**Why**: a naive `L_align-obs.backward()` flows gradient through *both* the latent-generation graph (good — that's what we want to train) and the rest of the LM trunk (bad — the LM trunk is already CE-trained, and the alignment signal would push it to compensate for the latent miss). The surrogate isolates the gradient to only the path `θ → ĥ_{latent} → … → L_align`, effectively treating `ĥ_{obs}` (downstream of `ĥ_{latent}`) as the only "differentiable observable."

**Code** (upstream `src/trainer.py:11-42`, verbatim):

```python
def compute_latents_only_loss(latents, loss_for_latents):
    """
    Compute a loss that backpropagates only through the latent embeddings `latents`.
    """
    def _flatten_tensors(x):
        if isinstance(x, (list, tuple)):
            out = []
            for y in x:
                out.extend(_flatten_tensors(y))
            return out
        return [x]

    ce_vec_list = _flatten_tensors(latents)
    grads = torch.autograd.grad(
        outputs=loss_for_latents,
        inputs=ce_vec_list,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    safe_grads = [g.detach() if g is not None else torch.zeros_like(v)
                  for v, g in zip(ce_vec_list, grads)]
    proxy_loss = torch.stack(
        [(v * g).sum() for v, g in zip(ce_vec_list, safe_grads)]
    ).sum()
    return proxy_loss
```

The scalar value of `proxy_loss` is meaningless (it's `Σ vᵀ g`, which can be any sign and oscillates wildly). Only its **gradient w.r.t. parameters** matters. We learned this the hard way: a `tot` log line oscillating ±100 is the expected cosmetic signature of the technique, not divergence. Use `ce` and `align` as the interpretable training signals.

**Caveat: `allow_unused=True`**. If `ce_vec_list` happens to contain a tensor with no path to `loss_for_latents` (e.g. because the trainer forgot to do the CE forward with `ce_patch_vec=…`), the grad is silently `None` → replaced with zero → `proxy_loss = 0`. This makes the whole stage a no-op without any error. Hence our smoke gate `[devfix-check] latent_only_connected=True` which fails the run at step ~10 if the grad is all-zero. We hit this exact failure mode in Job B (our first Stage 2 attempt without the verbatim port).

### 5.3 All-layer alignment loss

`modeling_qwen2_5_vl_monet.py:244-250`:

```python
def alignment_loss(teacher_hidden_states, student_hidden_states):
    if teacher_hidden_states.dim() == 3:   # all-layer: [num_layers, num_align, dim]
        return (1 - F.cosine_similarity(
            teacher_hidden_states.to(student_hidden_states.device),
            student_hidden_states,
        )).mean()
    elif teacher_hidden_states.dim() == 1:   # last-layer-only fallback
        return 1 - F.cosine_similarity(student_hidden_states, teacher_hidden_states, 0)
```

Two reductions are used:
- **3-D teacher** (S2 default and S3 `--alignment_layer all_layers`): mean over (layer, position, dim) of `1 - cos`.
- **1-D teacher** (S3 last-layer-only): plain cosine distance.

Paper explicitly notes the **all-layer** choice as a deviation from prior latent-CoT work (which aligned only the final layer). The `--alignment_layer last_layer` switch exists upstream but isn't used in either shipped script — every `script_examples/*.sh` sets `--alignment_layer all_layers`.

### 5.4 Trainer assembly

Upstream `src/trainer.py::CustomTrainerSFT_STAGE2.compute_loss` (lines 142-211), verbatim assembly:

```python
# Latent forward (no loss, just generate ce_patch_vec/pos)
inputs['latent_mode'] = True
inputs['loss_type'] = []
model.gradient_checkpointing_disable()        # use_cache=True is on in latent forward
outputs = model(**inputs, return_dict=True, output_hidden_states=False)

# CE forward (with alignment if α ≠ 0)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
inputs['latent_mode']      = False
inputs['ce_patch_pos']     = outputs.ce_patch_pos
inputs['ce_patch_vec']     = outputs.ce_patch_vec
inputs['ce_emphasize_poss']= inputs['observation_poss']
inputs['ce_emphasize_factor']= self.ce_emphasize_factor
inputs['loss_type']        = ['ce'] + (['alignment'] if self.args.alignment_weight != 0 else [])
inputs['compute_emphasize_acc'] = True

if self.args.alignment_weight != 0:
    teacher_reps = load_offline_tensor(self.args.teacher_reps_dir,
                                       batch_metadata=inputs['metadata'],
                                       alignment_layer=self.args.alignment_layer)
    inputs['alignment_poss'] = inputs['observation_poss']
    inputs['teacher_hidden_states_for_alignment'] = teacher_reps

teacher_ce_loss, teacher_output = super().compute_loss(model, inputs, return_outputs=True, …)
alignment_loss = teacher_output.loss_dict.get('alignment', torch.tensor(0.0))

if self.args.emphasize_latent_weight != 0.0 and alignment_loss.item() != 0.0:
    latent_only_loss = compute_latents_only_loss(
        outputs.ce_patch_vec, self.args.alignment_weight * alignment_loss
    )
    loss = self.args.emphasize_latent_weight * latent_only_loss + teacher_ce_loss
else:
    loss = teacher_ce_loss + self.args.alignment_weight * alignment_loss
```

**Key observations from the code that are not in the paper**:

1. `emphasize_latent_weight` and `alignment_weight` are **two distinct knobs**. The shipped script sets both to 2.0; paper Table 7 lists only `α = β = 2.0`. `emphasize_latent_weight` gates the surrogate-vs-naive switch.
2. The `if alignment_loss.item() != 0.0` guard means **for the first few steps where alignment happens to be ~0** (rare in practice but possible) the trainer silently falls back to the naive `ce + α·align` path. Our port preserves this.
3. The CE forward uses `loss_type=['ce', 'alignment']` *both* — so the model returns a `loss_dict` with both keys. The trainer takes `loss` (=CE) directly and grabs alignment from `loss_dict['alignment']`. This is why the alignment loss is computable from the CE forward graph (necessary for the surrogate to work).
4. `gradient_checkpointing` is **toggled around the latent forward**: disabled for the latent forward (which uses `use_cache=True` — incompatible with checkpointing), re-enabled for the CE forward.
5. The teacher representation is **loaded from disk** (`load_offline_tensor`), produced by an earlier `precompute_teacher_reps.py` run that does a single forward of the Stage-1 checkpoint with aux images on every Visual-CoT example and dumps per-sample `.pt` files. Naming: `rep_{alignment_layer}_{dataset_name}_{sample_id}.pt`, containing `{"latent": Tensor[L_layers, N_align, H]}`.

### 5.5 Stage 2 — paper ↔ code agreement and deviations

| Item | Paper | Upstream code | Our cluster port | Status |
|---|---|---|---|---|
| Loss form | Eq. 3: NTP + α·L_align-obs | `loss = teacher_ce_loss + α·alignment_loss` (fallback) **OR** `loss = β·latent_only_loss + teacher_ce_loss` when `emphasize_latent_weight != 0` | Same conditional structure (verbatim port as of Job C) | ✓ Agree |
| α (alignment_weight) | 2.0 | `sft_stage2.sh: --alignment_weight 2.0` | 2.0 | ✓ |
| Latent-only backprop | Eq. 9/10 | `compute_latents_only_loss` (src/trainer.py:11-42) | Verbatim port (cluster `trainer_monet_stage2.py`) | ✓ |
| `ce_emphasize_factor` | not in main text | `sft_stage2.sh: --ce_emphasize_factor 4.0` (note: Stage 1 uses 2.0) | 4.0 | ✓ |
| `emphasize_latent_weight` | not named in paper | `sft_stage2.sh: --emphasize_latent_weight 2.0` | 2.0 | ✓ |
| Alignment positions | "observation tokens" | `inputs['alignment_poss'] = inputs['observation_poss']` | Same | ✓ |
| Alignment layers | "all layers" | `--alignment_layer all_layers` | `all_layers` | ✓ |
| Latent size K | "8, 10, 12" mentioned | Released SFT ckpt = 8 (`sft_stage2.sh: --latent_size 8`) | 8 | ✓ |
| Teacher hidden states | "frozen teacher with aux images" | Pre-computed offline via `precompute_teacher_reps.py` | **Inline forward, on-the-fly** (deviation #3 below) | △ Functional but not byte-identical |
| 4D attention mask | "aux images visible only to latents" | `build_4d_attn` with ten flags (default: all observation flags off) | `build_monet_4d_attn` (no helper images, simpler logic) | △ Equivalent on tested data |
| Training steps | 1000 (≈ 1 epoch) | Script: `--epochs 2` = ~1844 steps | 2000 steps (budget) | △ More epochs than paper |
| Batch size | bs=1, gas=16 (eff=8 ranks) | Same | bs=2 × 4 GPUs × gas=16 = eff_bsz 128 | △ Smaller effective bsz upstream (16 vs 128) on a different cluster shape |
| Image budget | 1500·28² per Stage 2 | Same | 784000 max pixels (matches 1500·28²·ish) | ✓ |
| init | from Stage 1 ckpt | Same | Same | ✓ |
| Gradient checkpointing toggle | not mentioned | `disable→latent→enable→CE` | Same | ✓ |

**The three named deviations in our cluster port** (header of `cluster/trainer_monet_stage2.py:30-46`):

1. **(Job B → Job C: FIXED)** Latent-only backprop. Job B did `total = ce + α·align + emphasize·align` — a plain scalar add — which back-propagated alignment through the *entire* LM trunk. Job B step-1500 mean cos = 0.840 / util = -5.35 (collapsed). Job C uses the verbatim port; step-1500 evaluation pending (Task #17/#18).
2. **4D attention mask** — `cluster/mask_utils.py` is a constrained re-implementation (no helper-image case, no `<observation>` blocks) of upstream `build_4d_attn_wo_helper_images`. Validated on Phase 1.5b data only.
3. **Inline teacher forward** — instead of pre-computing per-sample `.pt` files (≈ 480 GB at bf16 for 118K samples), we run the teacher (vanilla Qwen2.5-VL-7B-Instruct, **not** Stage-1-SFT) inline in `torch.inference_mode()`. Adds ~14 GB resident per rank, saves ~4 hours upfront. **Note**: upstream's teacher is the *Stage-1 SFT checkpoint*, ours is *base Qwen2.5-VL-7B-Instruct* — this is a non-trivial difference. See *§13.2*.

---

## 6. Stage 1 — warm-up

### 6.1 Paper text

> *"Stage 1 is a vanilla SFT on interleaved image-text CoTs that initializes the model from Qwen2.5-VL-7B-Instruct and teaches it to emit the `<observation>` tags fluently. No latents are generated yet — auxiliary image positions are filled with actual aux images, not latent slots."*

### 6.2 Code

`script_examples/sft_stage1.sh`:

```bash
CE_EMPHASIZE_FACTOR=2.0
torchrun --nproc-per-node=8 -m src.main \
  --epochs 4 \
  --bsz 1 --grad_accum_steps 16 \
  --stage sft_stage1 \
  --data_path  ./Monet-SFT-125K/{Visual_CoT,CogCoM,ReFocus,Zebra_CoT_count,Zebra_CoT_visual_search,Zebra_CoT_geometry}/train.json \
  --load_model_path Qwen2.5-VL-7B-Instruct \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --ce_emphasize_factor 2.0
```

`CustomTrainerSFT_STAGE1.compute_loss` (`src/trainer.py:67-103`):

```python
inputs['latent_mode']        = False
inputs['input_ids']          = inputs['teacher_input_ids']      # aux images present
inputs['ce_emphasize_poss']  = inputs['teacher_observation_poss']
inputs['ce_emphasize_factor']= self.args.ce_emphasize_factor    # 2.0
inputs['loss_type']          = ['ce']
inputs['compute_emphasize_acc'] = True
ce_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, …)
return ce_loss
```

### 6.3 Paper ↔ code

| Item | Paper | Code | Status |
|---|---|---|---|
| Objective | NTP | `loss_type=['ce']` | ✓ |
| ce_emphasize_factor | "emphasized CE on observation tokens" | 2.0 (vs Stage 2's 4.0) | ✓ |
| Aux images | present | `teacher_input_ids` (= raw, no latent slots) | ✓ |
| Steps | 3885 (3 epochs in paper Table 7) | `--epochs 4` in shipped script | △ Off by one epoch |
| Init | Qwen2.5-VL-7B-Instruct | Same | ✓ |

Mostly straightforward. The **Table 7 vs script disagreement** on epochs (3 vs 4) is consistent across all three stages — paper rounds to a tidy "~1-3 epoch" claim while shipped scripts use `--epochs {4,2,2}` for stages 1/2/3. Treat the released `Monet-SFT-7B` checkpoint as authoritative.

---

## 7. Stage 3 — latents without aux images

### 7.1 Paper text

> *"Stage 3 trains the model to generate latents from question context alone — without auxiliary images at the student input. The model is re-initialized from the warm-up Stage 1 checkpoint. Supervision: align student latent embeddings to Stage-2 teacher latent embeddings, over all layers."*

### 7.2 Loss

Eq. 5:

$$\mathcal{L}_{\text{stage3}} = \mathcal{L}_{\text{NTP}} + \beta \cdot \mathcal{L}_{\text{align-latent}}, \quad \beta = 2.0$$

Eq. 4:

$$\mathcal{L}_{\text{align-latent}} = \frac{1}{N}\sum_i \sum_l \big[1 - \cos\big(\text{stop\_grad}(h^{*\,(i,l)}_{\text{latent}}),\; \hat h^{(i,l)}_{\text{latent}}\big)\big]$$

Same form as Stage 2, but aligning at **latent positions** instead of observation positions, and the teacher is the **Stage-2 model**, not Stage-1.

### 7.3 Trainer

`CustomTrainerSFT_STAGE3.compute_loss` (`src/trainer.py:295-372`):

```python
teacher_latents = load_offline_tensor(
    self.teacher_latent_dir, batch_metadata=inputs['metadata'],
    alignment_layer=self.args.alignment_layer, rep_type="latent",
)

# Latent forward — student inputs use the no-aux-image variant
inputs['latent_mode']     = True
inputs['input_ids']       = inputs['student_input_ids']      # aux images STRIPPED
inputs['attention_mask']  = inputs['student_attention_mask']
inputs['alignment_poss']  = inputs['student_alignment_poss']  # latent positions
inputs['teacher_hidden_states_for_alignment'] = teacher_latents
inputs['loss_type']       = []
student_outputs_latent = model(**inputs)

# CE forward
inputs['latent_mode']    = False
inputs['labels']         = inputs['student_labels']
inputs['ce_patch_pos']   = student_outputs_latent.ce_patch_pos
inputs['ce_patch_vec']   = student_outputs_latent.ce_patch_vec
inputs['ce_emphasize_poss'] = inputs['observation_poss']
inputs['ce_emphasize_factor'] = self.ce_emphasize_factor       # 4.0
inputs['loss_type']      = ['ce', 'alignment']
if 'student_attention_mask_4d' in inputs:
    inputs['attention_mask_4d'] = inputs.pop('student_attention_mask_4d')
ce_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, …)
alignment_loss = outputs.loss_dict['alignment']
loss = ce_loss + self.alignment_weight * alignment_loss
```

**Stage 3 does NOT use `compute_latents_only_loss`.** This is significant: it means Stage 3's alignment gradient *does* flow through the whole LM trunk. The paper's emphasis on latent-only-BP applies to Stage 2 only.

`load_offline_tensor` for Stage 3 reads files named `latent_{last_layer|all_layers}_{dataset}_{sample_id}.pt`, produced by `precompute_teacher_latents.py` running the Stage-2 checkpoint with `latent_mode=True, output_hidden_states=True` over Monet-SFT-125K. The teacher's per-position last-layer (or all-layer) hidden states at latent positions become the alignment target.

### 7.4 Why Stage 3 collapses on the released checkpoint (our finding)

Our Phase 0 probe (`phase0_monet_probe/REPORT.md`, also see `SUPPLEMENT.md`) found that on the **released** Monet-SFT-7B checkpoint:

| stage | subset | mean cos (off-diag) | n_helpful_positions | utility (none_NLL − all_NLL) |
|---|---|---:|---:|---:|
| 2 | Visual_CoT | 0.375 | 4 | +2.71 |
| **3** | Visual_CoT | **0.867** | **1** | **+0.26** |

Pairwise cosine matrix at Stage 3:

```
       p0   p1   p2   p3   p4   p5   p6   p7
 p0  1.00 0.83 0.68 0.63 0.61 0.60 0.60 0.59
 p1  0.83 1.00 0.94 0.87 0.84 0.83 0.82 0.81
 p2  0.68 0.94 1.00 0.98 0.96 0.94 0.93 0.92
 p3  0.63 0.87 0.98 1.00 0.99 0.98 0.97 0.97
 p4  0.61 0.84 0.96 0.99 1.00 1.00 0.99 0.99
 p5  0.60 0.83 0.94 0.98 1.00 1.00 1.00 1.00
 p6  0.60 0.82 0.93 0.97 0.99 1.00 1.00 1.00
 p7  0.59 0.81 0.92 0.97 0.99 1.00 1.00 1.00
```

Positions 4–7 are functionally collinear. Single-keep ablation: only position 0 is individually helpful.

The student-path replication (`STUDENT_PATH_NOTES.md`) gets the *same* shape (uniformly +0.01-+0.04 higher off-diag), confirming the collapse is a **property of the Stage-3 objective**, not an extraction-pathway artifact. This matches the architectural intuition: Stage 3 supervises against a *latent of a latent* with no encoder grounding — once the teacher target is itself a uniform vector, the student inherits the uniformity.

**Implication for downstream work**: prefer Stage-2-style training (per-position encoder-grounded target) over Stage-3-style distill-from-latent-teacher. See *§13.4*.

### 7.5 Stage 3 — paper ↔ code

| Item | Paper | Code | Status |
|---|---|---|---|
| Objective | Eq. 5 | `loss = ce_loss + β·alignment_loss` (no surrogate) | ✓ |
| β | 2.0 | `--alignment_weight 2.0` | ✓ |
| Latent-only BP | not used (paper only mentions for Stage 2) | not used in trainer | ✓ |
| Aux images at student | dropped | `student_input_ids` is the stripped variant | ✓ |
| Teacher | Stage-2 checkpoint | `--teacher_latent_dir <stage2>/monet_precomputed_target_latent/…` | ✓ |
| Init | Stage 1 ckpt | `--load_model_path …/sft_stage1_ce2.0` | ✓ |
| K | 8 (per shipped script) | `--latent_size 8` | ✓ |
| Steps | 1000 (≈ 1 epoch) | `--epochs 2` (~1844 steps) | △ |
| `--stage` flag string | n/a | `"avt_v5_stage2"` (legacy internal name, but Stage 3 in spirit) | △ Code/comment quirk; `main.py` parses this string |

---

## 8. RL — VLPO

### 8.1 Setup

- **Framework**: `verl` (EasyR1 fork) — `RL/` is mostly vendored verl + Monet patches.
- **Conda env**: `easyr1` (Python 3.11), separate from the SFT env.
- **vLLM**: pinned to 0.8.5 (per `RL/monet_rl_patch.py`).
- **Data**: Thyme-RL 3.2K subset, 1 epoch.
- **Reward**: accuracy reward (1 if correct, 0 otherwise) + format reward (encourages `\boxed{}`). No explicit latent-reasoning reward.
- **Judge**: rule-based + Gemini 2.5 Pro (configurable via `RL/examples/config_monet.yaml::reward_function`).
- **Latent size**: **K = 10** (vs K=8 in SFT) — increased during RL.

### 8.2 Hyperparameters (`RL/examples/vlpo_train.sh`)

```
CUDA_VISIBLE_DEVICES=0..7
RAY_NUM_GPUS=8, RAY_NUM_CPUS=16
LATENT_SIZE          = 10
MONET_RL_SIGMA       = 10.0       # set on both actor and ref
ROLLOUT_N            = 8
TEMPERATURE          = 0.5
SELECT_ACC_THRESHOLD = 0.6        # group-acc filter
KL_COEF              = 0.01
ORI_BSZ              = 64
ONLINE_ACCUM_SIZE    = 256
MAX_PROMPT_LENGTH    = MAX_RESPONSE_LENGTH = 4096
TENSOR_PARALLEL_SIZE = 1
```

Trainer entry: `python -m verl.trainer.main --config examples/config_monet.yaml sampling_strategy=monet`.

### 8.3 The Gaussian importance ratio for latents

Discrete tokens have a tractable `log π(t|s)`. Continuous latents don't. Monet's trick:

**Policy density at latent slot `t` for trajectory `i`** (Eq. 7):

$$\pi_\theta(h^{\text{old}}_{i,t} \mid Q, I, o_{i,<t}) = \exp\!\bigg(-\frac{1}{2\sigma^2}\|h^{\text{old}}_{i,t} - h^\theta_{i,t}\|^2\bigg) \cdot Z^{-1}$$

i.e. treat the *old policy's emitted latent* as a Gaussian sample from a unit-variance Gaussian centered at the new policy's emitted latent (with std σ).

**Importance ratio for that slot** (Eq. 8):

$$r_{i,t}(\theta) = \exp\!\bigg(-\frac{1}{2\sigma^2}\|h^{\text{old}}_{i,t} - h^\theta_{i,t}\|^2\bigg)$$

The normalizer cancels; the ratio is just the Gaussian kernel between old and new latent embeddings. **σ = 10.0**.

For trajectories with `A > 0` (advantage from accuracy reward), this ratio pulls the new latent toward the old (good) latent. For `A < 0` it pushes away. KL penalty against the reference (SFT) latents is computed analogously, with `KL_COEF = 0.01`.

**Selection threshold**: groups with average accuracy below 0.6 are filtered (matches Dr.GRPO-style filtering — keep informative groups, drop unanimous-fail groups).

### 8.4 What this *doesn't* solve

- The importance ratio is unbounded above (Gaussian kernel can be very small in the tails → very large ratio with tiny advantage). PPO-style clipping logic in verl applies; verify σ choice empirically.
- σ acts like a temperature on the policy distribution. σ → ∞: ratio → 1, no learning signal. σ → 0: ratio collapses to indicators on identical latents, gradient explodes. σ = 10.0 is a tuned value; paper Table 8 doesn't show ablations on it.
- Reward is **accuracy + format**; no latent-shape reward (e.g. anti-collapse, diversity). The latent quality is shaped entirely by SFT initialization + KL anchor. *§13.5* expands.

---

## 9. Inference — vLLM patch

### 9.1 Why a patch is needed

HF generation doesn't know to take the previous step's *hidden state* as the next step's *input embedding*. Two options:
- Override `generate` in `Qwen2_5_VLForConditionalGeneration` to do the K-step regime manually. Doesn't scale.
- Patch vLLM's GPU model runner to do it batched. This is what Monet ships.

### 9.2 The patch (`inference/vllm/monet_gpu_model_runner.py`)

Two patches into vLLM v1's `GPUModelRunner.execute_model`:

**(a) Boundary detection** after sampling, before output caching:

```python
last_token_h = sample_hidden_states     # [num_reqs, H]
for i, req_id in enumerate(self.input_batch.req_ids):
    st = self.latent_state.setdefault(req_id, {"active": False, "pending": None, "current_len": 0})
    gen_ids = valid_sampled_token_ids[i]
    for j, tid in enumerate(gen_ids):
        if not st["active"] and tid == self.latent_start_id:
            st["active"] = True
        elif st["active"] and (tid == self.latent_end_id or st["current_len"] >= self.latent_size):
            st["active"] = False
            st["pending"] = None
            st["current_len"] = 0
            valid_sampled_token_ids[i] = [self.latent_end_id]
    if st["active"] and last_token_h is not None:
        st["pending"] = last_token_h[i].detach()
        st["current_len"] += 1
```

**(b) Input embedding override** at the start of the next forward:

```python
override_indices, override_embeds = [], []
for i, req_id in enumerate(self.input_batch.req_ids):
    st = self.latent_state.get(req_id)
    if st and st.get("active") and st.get("pending") is not None:
        override_indices.append(rows_cpu[i].item())
        override_embeds.append(st["pending"])
if override_indices:
    idx = torch.tensor(override_indices, device=…, dtype=…)
    embeds = torch.stack(override_embeds, dim=0)
    self.inputs_embeds.index_copy_(0, idx, embeds)
    # clear pending after consume
```

Per-request state lives in `self.latent_state[req_id] = {"active": bool, "pending": Optional[Tensor], "current_len": int}`. Constraints:
- **Single pipeline-parallel rank** (`len(get_pp_group().ranks) == 1`).
- **Speculative decoding incompatible** (disabled when `self.speculative_config` is set).
- Tensor parallel = 4 in shipped example.

### 9.3 Wiring

`inference/apply_vllm_monet.py::patch()`:

```python
os.environ["LATENT_START_ID"] = "151666"
os.environ["LATENT_END_ID"]   = "151667"
sys.modules["vllm.v1.worker.gpu_model_runner"] = patched_runner_module
sys.modules["vllm.worker.gpu_model_runner"]    = patched_runner_module     # legacy v0 path
sys.modules["vllm.worker.model_runner"]        = patched_runner_module
```

Example caller `inference/vllm_inference_example.py` is the end-to-end: instantiate `LLM` from `monet_qwen_model/apply_qwen2_5_monet.py`'s patched HF class, build messages, generate, then `replace_abs_vis_token_content` regex-rewrites `<abs_vis_token>…</abs_vis_token>` → `<latent>…</latent>` in the displayed string.

### 9.4 What no eval script ships

The repo has **no V★, MMVet, HRBench, or MME-RealWorld runners**. README points to external [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) with the system prompt:

> *You are a helpful multimodal assistant. … Put your final answer in `\boxed{}`.*

The boxed-answer convention is shared with the RL format reward. Anyone replicating paper numbers must (a) install VLMEvalKit, (b) point it at the vLLM-patched Monet inference path, (c) use that exact system prompt, and (d) confirm `LATENT_SIZE` env matches the checkpoint (8 for SFT, 10 for RL). This is the open task tracked as Task #10 (owned by our Session A).

---

## 10. Paper ↔ code agreement — single-table summary

| Topic | Paper claim | Code reality | Agreement |
|---|---|---|---|
| Stage 1 objective | NTP w/ emphasized CE | `loss_type=['ce']`, `ce_emphasize_factor=2.0` | ✓ |
| Stage 2 objective | NTP + α·L_align-obs | Two-mode: (`ce + α·align`) ∥ (`emphasize·latent_only_loss + ce`) | ✓ when surrogate path active |
| Latent-only BP | Eq. 9/10 | `compute_latents_only_loss` (`src/trainer.py:11-42`) | ✓ Verbatim |
| Alignment loss | All-layer 1-cos | `alignment_loss` selects 3-D vs 1-D teacher; both shipped scripts use 3-D | ✓ |
| Stage 3 objective | NTP + β·L_align-latent | `ce + β·alignment_loss` (no surrogate) | ✓ |
| 4D mask | aux img → latent only | `build_4d_attn` with default flags off | ✓ (default flags) |
| K | 8/10/12 mentioned; SFT released = 8 | `--latent_size 8` | ✓ |
| α, β | 2.0, 2.0 | 2.0, 2.0 (`sft_stage{2,3}.sh`) | ✓ |
| `ce_emphasize_factor` | not in main text | 2.0 (S1) / 4.0 (S2) / 4.0 (S3) | code-only |
| `emphasize_latent_weight` | not in main text | 2.0 (S2 only) | code-only |
| Training steps | 3885 / 1000 / 1000 | `--epochs {4, 2, 2}` (paper appears to underreport) | △ Off by ~2× on S2/S3 |
| Batch size | bs=1, gas=16 | Same | ✓ |
| Training data | Monet-SFT-125K | Same in all three scripts | ✓ |
| RL data | "3.2K from Thyme-RL" | Confirmed via dataset_valid_ids | ✓ |
| RL importance ratio | Eq. 8 Gaussian | `MONET_RL_SIGMA=10.0` in `RL/monet_rl_patch` | ✓ |
| RL accuracy threshold | 0.6 | `SELECT_ACC_THRESHOLD=0.6` | ✓ |
| RL KL coef | not in Table 8 | `KL_COEF=0.01` | code-only |
| Inference | vLLM w/ hidden-state feedback | `inference/vllm/monet_gpu_model_runner.py` patch | ✓ |
| Eval | "VLMEvalKit + the boxed-answer prompt" | No eval scripts in repo | ✓ (but unimplemented in-repo) |
| License | not mentioned | None at top level, Apache-2.0 only in RL/ | code-only |

---

## 11. Code-only details (gotchas not in paper)

These are the things that matter operationally but the paper doesn't (or barely) cover:

1. **`emphasize_latent_weight` vs `alignment_weight` are different knobs.** The paper only names `α`. In code, `emphasize_latent_weight` gates the surrogate path; setting it to 0 collapses Stage 2 to vanilla `ce + α·align`. The shipped script's `2.0 × 2.0` means the *effective* alignment-induced gradient on the latent-generation graph is `4 × (∂L_align/∂h_latent)·(∂h_latent/∂θ)`.

2. **`ce_emphasize_factor` is 2.0 in Stage 1 but 4.0 in Stages 2/3.** Paper doesn't ablate or mention this jump. Higher weighting on observation-token CE in S2/S3 likely compensates for the latent-replacement-of-aux-images shrinking the per-step visual information.

3. **Stage 3 *omits* the latent-only-BP surrogate.** The paper's emphasis on the surrogate is Stage-2-only. Stage 3's alignment gradient flows through the whole trunk.

4. **`alignment_loss.item() != 0.0` guard.** If alignment happens to come out ≈ 0 (rare but possible at warmup or with degenerate batches), the surrogate path is skipped and the trainer falls back to naive add. Not a bug, but worth knowing.

5. **`allow_unused=True` is a silent footgun.** If `ce_patch_vec` is not in the CE-forward graph (e.g. you forgot `inputs['ce_patch_vec'] = outputs.ce_patch_vec`), `torch.autograd.grad` returns `None` for every entry → zero proxy → no learning. No error. We catch this with the smoke gate.

6. **`gradient_checkpointing` is toggled around the latent forward.** `disable()` before latent (which needs `use_cache=True`), `enable()` before CE. Required, not optional. If you keep it enabled for the latent forward you OOM and/or get silent recompute mismatches.

7. **Batch size 1 is the only tested config.** Upstream code at `modeling_qwen2_5_vl_monet.py:1745` literally says *"WARNING: we only use batch_size=1 in our training. Larger batch_size has not been tested."* The latent forward iterates samples in a Python for-loop — bsz > 1 is correct in principle but untested.

8. **Module-level monkey-patching, twice.** Both training and inference patch `sys.modules` before any `import transformers`. Putting any code that imports `transformers.models.qwen2_5_vl` before the patch silently uses vanilla Qwen.

9. **Token IDs hardcoded.** `LATENT_START_ID=151666`, `LATENT_END_ID=151667` are baked into env vars. Match your tokenizer or things break silently.

10. **Two conda envs.** SFT (`monet`, Python 3.10, transformers/vllm one set) and RL (`easyr1`, Python 3.11, transformers/vllm another set) are not unifiable. The `RL/` subtree has its own pyproject.toml.

11. **Teacher representation is offline + named-on-disk.** `precompute_teacher_reps.py` dumps one `.pt` per sample, named with `{rep_type}_{alignment_layer}_{dataset}_{sample_id}.pt`. Storage cost at 118K × 28 layers × ~5 obs × 3584 dim × bf16 ≈ 70 GB; in practice ≈ 480 GB once sequence-length variance is included. Our inline-forward port trades that storage for ~14 GB resident.

12. **Stage 3 student strips assistant-turn aux images** (`student_input_ids` ≠ `teacher_input_ids`). The user's *question* image is kept; only the assistant-turn aux images that latents are meant to replace are dropped.

13. **`avt_v5_stage2` legacy string.** `sft_stage3.sh` passes `--stage "avt_v5_stage2"` (legacy internal name for what the paper calls Stage 3). `main.py` parses this exact string. Don't rename without grepping.

14. **No top-level LICENSE.** Only `RL/LICENSE` (Apache-2.0). Treat the rest as license-unspecified.

15. **vLLM 0.8.5 pin for RL** (per `RL/monet_rl_patch.py`). Different vLLM in the SFT/inference env.

16. **Custom `prepare_inputs_for_generation` returns 4-D position IDs**, prepending text positions to Qwen's 3-D vision positions. If you wrap Monet inference in custom decoding code, beware position shape changes.

---

## 12. Hyperparameter tables — consolidated

### 12.1 SFT (from paper Table 7 + shipped scripts)

| Param | Paper Table 7 | `sft_stage1.sh` | `sft_stage2.sh` | `sft_stage3.sh` |
|---|---|---|---|---|
| lr | 1e-5 | 1e-5 (default) | 1e-5 | 1e-5 |
| bsz | 1 | 1 | 1 | 1 |
| grad_accum_steps | 16 | 16 | 16 | 16 |
| weight_decay | 0.01 | 0.01 (default) | 0.01 | 0.01 |
| epochs / steps | S1: 3885 ≈ 3ep, S2/S3: 1000 ≈ 1ep | `--epochs 4` | `--epochs 2` | `--epochs 2` |
| K (latent_size) | {8, 10, 12} | n/a | 8 | 8 |
| α (alignment_weight) | 2.0 | n/a | 2.0 | 2.0 |
| β (alignment_weight) | 2.0 | n/a | n/a | 2.0 |
| emphasize_latent_weight | not named | n/a | 2.0 | not used |
| ce_emphasize_factor | "emphasized CE" | 2.0 | 4.0 | 4.0 |
| alignment_layer | "all layers" | n/a | all_layers | all_layers |
| max pixels per image | 2000·28² (S1, S3), 1500·28² (S2) | 2000·28² | 1500·28² | 2000·28² |
| optim | AdamW | AdamW | AdamW | AdamW |
| warmup | not specified | default `transformers` | default | default |
| deepspeed | ZeRO-2 | ds_zero2_gpu.json | same | same |
| init | Qwen2.5-VL-7B | Same | Stage 1 ckpt | Stage 1 ckpt (not Stage 2) |
| teacher | n/a | n/a | Stage 1 ckpt with aux images | Stage 2 ckpt (latent forward) |

### 12.2 RL / VLPO (from paper Table 8 + `vlpo_train.sh`)

| Param | Paper | Script | Notes |
|---|---|---|---|
| lr | 1e-6 | (via `config_monet.yaml`) | |
| Global bsz | 64 | `ORI_BSZ=64` | |
| Online accumulator | not in paper | `ONLINE_ACCUM_SIZE=256` | code-only |
| Rollout N | 8 | `ROLLOUT_N=8` | |
| Temperature | 0.5 | `TEMPERATURE=0.5` | |
| Max prompt / response length | 4096 | `MAX_PROMPT_LENGTH=MAX_RESPONSE_LENGTH=4096` | |
| VLPO σ | 10.0 | `MONET_RL_SIGMA=10.0` | actor + ref |
| Latent size K | 10 | `LATENT_SIZE=10` | up from K=8 in SFT |
| Accuracy threshold | 0.6 | `SELECT_ACC_THRESHOLD=0.6` | |
| KL coef | not in paper | `KL_COEF=0.01` | code-only |
| Tensor parallel | not in paper | `TENSOR_PARALLEL_SIZE=1` | |
| RL data | 3.2K Thyme-RL, 1 epoch | dataset_valid_ids/ | |
| Judge | "rule-based + Gemini" | Gemini 2.5 Pro default | |

---

## 13. Empirical headlines

### 13.1 Paper Table 3-5 (released numbers)

Monet-7B (full SFT + VLPO, K=10) vs Qwen2.5-VL-7B base:

| Benchmark | Monet-7B | Qwen2.5-VL-7B |
|---|---:|---:|
| V★ | 83.25 | 76.44 |
| HRBench4K | 83.48 | 77.39 |
| HRBench8K | 82.89 | 75.00 |
| MME-RealWorld-Lite | **79.75** | 63.75 |
| VisualPuzzles | **35.02** | 32.71 |

### 13.2 Stage-2 ablation (paper Table 2)

| Stage-2 variant | V★ |
|---|---:|
| Full Monet-SFT | **82.20** |
| w/o latent-only BP | **46.07** |
| w/o aux images (in Stage 2) | 73.30 |
| w/o observation alignment | 75.39 |

**Latent-only BP is the load-bearing component.** Removing it = catastrophic. This is what our Job B vs Job C A/B is designed to verify.

### 13.3 RL / VLPO ablation (paper Table 5)

| Variant | V★ |
|---|---:|
| Monet-7B (full) | 83.25 |
| Monet-SFT (no VLPO) | 82.20 |
| Monet-SFT + GRPO | 80.10 |

**GRPO underperforms VLPO and even underperforms no-RL.** This is the paper's main RL claim — Gaussian importance ratios beat token-discrete GRPO ratios on latent slots.

### 13.4 Our finding: Stage 3 collapse on the released checkpoint

| stage | subset | mean cos | n_helpful | utility |
|---|---|---:|---:|---:|
| 2 | Visual_CoT | 0.375 | 4 | +2.71 |
| 2 | Zebra_count | 0.394 | 2 | +0.19 |
| 3 | Visual_CoT | **0.867** | **1** | **+0.26** |
| 3 | Zebra_count | **0.840** | **0** | **-0.001** |

Source: `phase0_monet_probe/REPORT.md`, n=100, monet-self reader. **Stage 3 latents are functionally degenerate** even on the released checkpoint. The paper reports V★ improvements at the RL-final stage; intermediate Stage-3 quality is not directly evaluated in the paper.

### 13.5 What the released checkpoints actually demonstrate

- **The Stage-2 recipe works** (encoder-grounded all-layer alignment with latent-only-BP) — produces latents with healthy diversity and high reader utility.
- **The Stage-3 recipe collapses** the latent representation under our internal probe — the *latent-of-a-latent* supervision target with no encoder grounding drives all-K-positions toward redundancy. The released Monet-SFT-7B (this is Stage 2) has *better* internal-probe metrics than the Monet-7B (post-Stage-3 + VLPO) on our probe.
- **RL salvages benchmark scores** but doesn't fix the representation geometry. The model still answers correctly on V★/MMVet because the *answer-tokens* see useful latents at decode time — but the latents themselves carry little position-distinct information.

The conclusion has shaped our roadmap: import Stage-2-style training; skip Stage-3-style self-distill; if VICReg-style variance/covariance regularization can be added on top of Stage 2, it should *strengthen* the same property the latent-only BP is already providing.

---

## 14. Hook points for variation

The natural places to plug in a controlled variation. Per-point: what to swap, where in the code, what risk that introduces.

### H1. Swap the alignment loss

**Where**: `modeling_qwen2_5_vl_monet.py:244-250` (`alignment_loss(teacher_hidden_states, student_hidden_states)`).

**What to swap**:
- **VICReg** (variance + invariance + covariance) — invariance term replaces `1 - cos`, variance hinge on student per-position std, covariance penalty on student per-dim cross-covariance. Drops the *teacher* requirement (variance hinge + covariance are self-supervised) — this is its big advantage if Stage 2's expensive teacher forward is a bottleneck.
- **Barlow Twins** (cross-correlation on teacher–student per-dim) — closer to alignment in spirit, requires teacher.
- **InfoNCE** (contrastive) — would require negatives; awkward at bsz=1 unless cross-sample negatives.
- **L2 in feature space** — simpler, no temperature.

**Risk**: alignment is computed differently for 3-D (`[L_layers, N_align, H]`) vs 1-D teacher; swapping must preserve the multi-layer aggregation contract or downstream `loss_dict['alignment']` shapes break.

**Already started**: `cluster/configs/v3_monet_plus_vicreg.yaml` is the V3 config (`lambda_reg=1.0, reg_var_w=1.0, reg_cov_w=0.04, reg_gamma=1.0`) that runs additive VICReg on top of the Monet recipe. The trainer code is wired (`cluster/reg.py`); just needs cluster submission. See "Future" below.

### H2. Swap the latent-only-BP surrogate

**Where**: `src/trainer.py:11-42` / our `cluster/trainer_monet_stage2.py::compute_latents_only_loss`.

**What to swap**:
- **Naive add** (`ce + α·align`) — same as Job B; collapses (V★ 46 per paper Table 2; cos 0.84 in our internal probe).
- **Last-N-layer surrogate** — restrict the `torch.autograd.grad` chain to only the latent-generation layers, not the whole encoder.
- **Detached-then-MSE** — compute alignment in a side path with detached student hidden states; use direct MSE on `ce_patch_vec` to fixed targets. Drops the teacher.
- **Gradient clipping inside the surrogate** — for stability, clip `g.detach()` per element.

**Risk**: the smoke gate (`latent_only_connected=True`) only checks "is the grad non-zero" — it doesn't check magnitude. Aberrant magnitudes show up as `tot` oscillation but don't fail fast.

### H3. Replace the latent input embedding

**Where**: `modeling_qwen2_5_vl_monet.py:1914-1916`.

```python
prev_hidden = batch_last_hidden_state[b, pos - 1, :].unsqueeze(0).unsqueeze(0).contiguous()
latent_embed = prev_hidden.clone() if self.training else prev_hidden.detach()
```

**What to swap**:
- **Learned latent codebook** — vector-quantize the previous hidden state through a learned codebook (VQ-VAE-style). Stops gradient flow through the embedding update, simplifies analysis.
- **Random projection then layernorm** — break the hidden-state-identity link; force the latent to be a true bottleneck.
- **Cross-attention pooling** — instead of "previous hidden state", compute latent as cross-attention over question + image features.

**Risk**: this is the load-bearing architectural primitive. Changing it changes the meaning of "latent" — direct comparison to Monet baselines is no longer apples-to-apples.

### H4. Replace the 4D attention mask policy

**Where**: `src/utils.py::build_4d_attn` (or our `cluster/mask_utils.py::build_monet_4d_attn`).

**What to swap**:
- **Aux image visible to text too** — relax the "only latents see aux" constraint. Probably hurts; that's the constraint that forces latents to encode the image.
- **Latent block can see prior observations** — currently latents see prior latents; allowing them to see prior observation-token text might enrich the latent context. Toggle: `latent_can_see_observations=True` (would need adding).
- **No 4D mask at all** (`--not_use_4d`) — already supported. Useful baseline.

**Risk**: silent — the mask is dict-form vs tensor-form; mis-passing falls back to standard causal mask without error.

### H5. Add a per-position regularizer to the latent block

**Where**: in the trainer, after the latent forward, before the CE forward.

**What to add**:
- **Variance hinge** on `outputs.ce_patch_vec[b]` across the K dimensions — directly penalize position collinearity. This is the V3 lever: `cluster/reg.py::vicreg_loss(ce_patch_vec, ...)` already implements VICReg-style var + cov on the K latent slots.
- **Diversity loss** (1 - mean pairwise cosine over the K slots) — direct anti-collapse signal.
- **Information bottleneck** (KL to a prior) — formal entropy regularizer.

**Risk**: depending on weight, can fight alignment. Sweep `lambda_reg` from below the alignment scale upward; V3 starts at `lambda_reg=1.0` (paper α=2.0 so the regularizer is half the alignment pressure).

### H6. Swap teacher choice

**Where**: in the precompute script (or our inline trainer forward).

**What to swap**:
- **Base Qwen2.5-VL-7B-Instruct** (our current cluster choice; deviation #3) vs **Stage-1-SFT checkpoint** (upstream choice). The Stage-1 teacher is text-format-aware (it emits `<observation>` tags); the base teacher has more diverse representations but no Monet-format awareness. The effect on alignment-grounded latents is non-trivial.
- **Stronger teacher** (e.g. Qwen2.5-VL-72B) — likely improves, expensive.
- **Self-teacher** (= an EMA of the student) — drops the need for a separate model; risks degenerate self-alignment.

**Risk**: paper figures use the Stage-1-SFT teacher; deviating muddles direct comparison.

### H7. Drop Stage 3 entirely

**Where**: skip `sft_stage3.sh` in the pipeline.

**Effect**: model emits latents only when aux images *are* available in input (Stage 2 inference regime). Useful as a baseline. Our Phase 0 results suggest Stage 3 is the collapse source, so a "Monet without Stage 3" baseline is worth running.

**Risk**: changes what the released checkpoint represents — published numbers won't be directly comparable.

### H8. Add a latent-shape reward in RL

**Where**: `RL/examples/config_monet.yaml::reward_function`; `RL/tools/custom_api.py`.

**What to add**:
- **Diversity bonus** — reward the K latents being far from each other in some metric.
- **Decoder-utility bonus** — measure how much the latents *helped* the answer NLL; reward proportionally.

**Risk**: shaping rewards can game; need a careful judge. The paper's "accuracy + format" is purposely minimal.

---

## 15. Open questions / future directions (project-internal)

**Q1. Does the deviation-#1 fix lift our Job B collapse?**
- Pending: Job C step-1500 evaluation (Task #17/#18, unblock on local data path).
- Comparison points already established: Job B step-1500 = cos 0.840 / util −5.35; released Stage-2 = cos 0.377 / util +2.05.

**Q2. Is the inline teacher (base Qwen) materially worse than the Stage-1-SFT teacher?**
- Affects whether our deviation #3 is a correctness concern or just an efficiency tradeoff.
- Test: train a small Stage-2 run twice — once with each teacher — compare internal-probe metrics at matched step.

**Q3. Does VICReg additive lift Stage-2 further?**
- `cluster/configs/v3_monet_plus_vicreg.yaml` ready; just needs cluster submission (user-approved per project rule §3-A).
- Hypothesis: VICReg helped at 3B/5K (Pivot A). It should help at 7B/118K — but the effect may be smaller given Monet's alignment already shapes per-position structure.

**Q4. Can we replace Stage 3 entirely with a different inference-without-aux strategy?**
- Phase 0 found Stage 3 collapses on our probe. If we drop Stage 3, can we still get aux-image-free inference?
- Options: (a) train a latent-classifier head that decides whether to emit `<latent>` (vs proceed text-only), (b) just operate the Stage-2 model with aux images present at all inference time, (c) introduce a different anti-collapse Stage 3 (VICReg-on-latents instead of cosine-align-to-teacher).

**Q5. What's the right σ for VLPO?**
- Paper picks 10.0; no ablation. σ trades off learning signal (low σ = sharp ratio) vs gradient stability (high σ = ratio ≈ 1).
- Out of scope for our SFT-only roadmap, but flag for any future RL work.

---

## 16. References

- arXiv: https://arxiv.org/abs/2511.21395 ([HTML v2](https://arxiv.org/html/2511.21395v2))
- GitHub: https://github.com/NOVAglow646/Monet
- HF SFT model: https://huggingface.co/NOVAglow646/Monet-SFT-7B
- HF RL model: https://huggingface.co/NOVAglow646/Monet-7B
- HF dataset: https://huggingface.co/datasets/NOVAglow646/Monet-SFT-125K
- EasyR1 (RL framework): https://github.com/hiyouga/EasyR1
- VLMEvalKit (suggested eval): https://github.com/open-compass/VLMEvalKit

Local cross-references:
- `phase0_monet_probe/REPORT.md` — internal probe results (Stage 2 healthy, Stage 3 collapsed).
- `phase0_monet_probe/SUPPLEMENT.md` — teacher-vs-student path replication.
- `phase0_monet_probe/STUDENT_PATH_NOTES.md` — implications for Stage-3 self-distill.
- `cluster/trainer_monet_stage2.py` — our paper-faithful Stage-2 port (Job C).
- `cluster/configs/stage2_monet_jobC.yaml` — the recipe used.
- `cluster/configs/v3_monet_plus_vicreg.yaml` — the VICReg-additive variant config.
- `eval_local/MONET_REPRO_LOG.md` — cross-session paper-eval reproduction log (Session A).

---

*Report compiled 2026-05-21. Based on arXiv v2 (2025-11-27), GitHub commit at HEAD as of that date, and local probe results from 2026-05-12 + cluster training runs through 2026-05-19.*
