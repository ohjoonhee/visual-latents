# Methods

This document describes the visual-latents training method as currently
implemented in `src/vl/`. It is written from the code, not from the design
spec — where the implementation deviates from `docs/inherited/ROUND3_POC_DESIGN.md`,
this is what the code actually does.

---

## 1. Overview

We train a vision-language model (VLM) to emit a fixed-length sequence of
**visual latent tokens** $h \in \mathbb{R}^{K \times D}$ from an image $x$,
such that one or more *frozen* sibling VLMs ("anchors", or "readers") can
consume $h$ — spliced into their image-token positions — and answer
questions about $x$ without ever seeing $x$ directly.

Concretely, with $\phi_i$ a frozen anchor and $(q, y^*)$ a question/gold-answer
pair drawn from a multi-question pool per image, the training objective is

$$
\mathcal{L}_{\text{total}}(\theta)
=
w_{\text{NLL}}(t) \cdot \underbrace{\frac{1}{R \cdot K_q}\!\!\sum_{i=1}^{R}\sum_{j=1}^{K_q}
\mathrm{CE}\!\bigl(\phi_i(\,h\,\Vert\,q_j),\ y_j^{*}\bigr)}_{\mathcal{L}_{\text{NLL\_multi}}}
\;+\;
w_{\text{concept}} \cdot \mathcal{L}_{\text{concept}}
\;+\;
w_{\text{norm}}(t) \cdot \mathcal{L}_{\text{norm}}.
$$

Here $\theta$ are the trainable parameters of the **generator** (a LoRA
adapter on the language model, three new token embedding rows, and a small
projection MLP). The anchors are frozen. Curriculum coefficients
$w_{\text{NLL}}(t)$ and $w_{\text{norm}}(t)$ ramp from low to full value over
the first 200 steps; $w_{\text{concept}}$ is constant.

The same generator backbone weights serve as anchor #1 — i.e., reader-1 sees
the generator's own LoRA — while reader-2 (a sibling VLM such as
`NOVAglow646/Monet-SFT-7B`) is loaded fresh, frozen, and bears no LoRA. This
"shared first reader" choice is documented in §2.6 of the round-3 spec; the
loader at `src/vl/readers.py` detects path equality with the generator's
`_name_or_path` and reuses the same `nn.Module` rather than double-loading.

---

## 2. Architecture

### 2.1 Generator

Backbone: `Qwen/Qwen2.5-VL-7B-Instruct` (D = 3584, 28 transformer layers; 3B
variant supported for local testing). Loaded in `bfloat16` via
`Qwen2_5_VLForConditionalGeneration.from_pretrained(...)`.

We extend the tokenizer with three new special tokens —
`<|latent|>`, `<|latent_start|>`, `<|latent_end|>` — and call
`model.resize_token_embeddings(...)` to allocate rows in the embedding table.
Their initial values are clones of the `<|image_pad|>` row plus
$0.02 \cdot \mathcal{N}(0, I)$.

Critically, after the resize we do **not** treat those table rows as the
trainable parameters. Instead we **snapshot** them into a fresh
`nn.Parameter` of shape $[3, D]$ in `float32` (`new_emb`), then freeze the
entire embedding table. At forward time we re-splice the three rows back into
the embedding lookup at the new-token positions (§4.2). This avoids the
sparse-grad pitfalls of training individual rows of a frozen table and lets
the trainer apply a higher learning rate to those three rows than to the rest
of the model.

After freezing the base, we apply LoRA via
`peft.inject_adapter_in_model(LoraConfig(...))` to the language model only —
i.e., `model.model.language_model`. The vision tower (`model.model.visual`)
and the LM head stay frozen. LoRA targets `q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj` across all 28 layers, with $r$, $\alpha$ and
dropout from `ModelConfig`. (`get_peft_model` is unusable here because
`Qwen2_5_VLTextModel` does not expose `prepare_inputs_for_generation`;
`inject_adapter_in_model` is the in-place variant that preserves the parent
shape.)

A small MLP in `bfloat16` projects $h$ into the teacher feature space for the
concept loss:

$$\mathrm{MLP}_{\text{concept}}: \mathbb{R}^{D} \xrightarrow{\text{Linear}} \mathbb{R}^{D/2} \xrightarrow{\text{GELU}} \xrightarrow{\text{Linear}} \mathbb{R}^{D}$$

The bottleneck $D \to D/2 \to D$ is intentional: it raises the friction
against an "identity collapse" solution where $\mathrm{MLP}(h) \approx h$
defeats the geometric prior.

**Trainable parameter inventory** (Qwen2.5-VL-7B with LoRA $r=32$, all
target modules, $K=16$):

| Component | Status | Approx. count |
|---|---|---|
| Vision tower + projector + merger | frozen | 675 M |
| LM base weights (non-LoRA) | frozen | 7 B |
| LM head | frozen | — |
| LoRA adapters on LM | trainable | ~50 M |
| `new_emb` (3 rows × 3584) | trainable | ~11 K |
| Concept MLP (D ↔ D/2) | trainable | ~25 M |

The trainer prints these counts at startup so they can be cross-checked.

### 2.2 Generator forward — emitting $h$

`forward_generator(model, new_emb, new_token_ids, processor, images, cfg)`
constructs, per image, the LIVR-style q-invariant prompt:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
<|latent_start|><|latent|><|latent|>...×K<|latent_end|>
<|im_end|>
<|im_start|>assistant
```

Note the absence of any question token: $h$ must encode visual content that
generalises across the K\_q questions the reader will later ask. The
processor expands the single `<|image_pad|>` placeholder into the
appropriate number of post-merger image tokens for the supplied
`pixel_values + image_grid_thw`.

We then build `inputs_embeds = embed(input_ids)` and overwrite the rows at
the three new-token positions with rows from `new_emb` (cast to `bfloat16`).
Boolean-index assignment is used so autograd registers `new_emb` as a leaf in
the gradient graph for those positions. The model is invoked with both
`input_ids` (used for the placeholder mask) and the patched `inputs_embeds`,
plus `pixel_values` and `image_grid_thw` (which let the model splice real
visual features into the `<|image_pad|>` slots), and `output_hidden_states=True`.

The visual latents are then read off the **last** hidden-states layer at the
$K$ positions where `input_ids == latent_token_id`:

$$
h_b = \bigl[\, \mathrm{hidden}_L[b,\,p_1,\,:] ,\; \dots ,\; \mathrm{hidden}_L[b,\,p_K,\,:]\,\bigr]
\quad\in\;\mathbb{R}^{K \times D}.
$$

`forward_generator` returns $h \in \mathbb{R}^{B \times K \times D}$ in
`bfloat16`, with `requires_grad=True`.

### 2.3 Teacher features $V_{\text{sem}}$

For the concept loss we need a frozen teacher. We use the generator's own
vision branch (the same Qwen2.5-VL vision tower + projector + merger), per
the spec's choice (a). This requires a separate forward — `get_v_sem` calls
`model.model.get_image_features(pixel_values, image_grid_thw)` under
`torch.no_grad()`. The returned `pooler_output` is a tuple of $[T_v, D]$
tensors per image; we stack into $V_{\text{sem}} \in \mathbb{R}^{B \times T_v \times D}$
(asserting uniform $T_v$ within a batch, which the data loader enforces by
fixed-size resizing).

### 2.4 Anchor (reader) consumption

`forward_anchor(anchor, h, q_text, a_text, K)` constructs, per example, a
text-only prompt that mirrors the generator prompt's image region but
replaces the single `<|image_pad|>` placeholder with $K$ explicit
`<|image_pad|>` tokens — and supplies the question and gold answer:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|>×K<|vision_end|>{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

We tokenise twice — once without the answer to find the prompt boundary,
once with — and build a per-example label mask that is `-100` everywhere
except the answer span. We then build `inputs_embeds = embed(input_ids)`,
splice $h_b$ into the $K$ image-pad positions of row $b$, and call the
anchor's `forward(input_ids=..., inputs_embeds=..., attention_mask=..., use_cache=False)`
**without** `pixel_values` — so the anchor never sees a real image; it sees
only the spliced $h$ where the image would have been.

CE loss uses the standard shift convention
($\mathrm{logits}[..., :-1, :]$ predicting $\mathrm{labels}[..., 1:]$) and
`reduction="mean"` over non-`-100` positions. Gradient flows from the loss
back through $h$ — the generator-side LoRA, `new_emb`, and concept MLP
all receive grad on `loss.backward()`.

The Stage-1 attention masking from LIVR ("answer tokens cannot attend to
image tokens") is currently a structural no-op for this setup: the generator
pass has no answer tokens and the reader pass has no image tokens. The
`stage1_attention_mask` config flag is honoured in spirit; an explicit 4-D
mask would only be needed for a future joint generator-reader training pass.

---

## 3. Loss

### 3.1 Multi-anchor multi-question NLL

For a batch of $B$ images, $R$ anchors, and $K_q$ questions per image:

```python
# losses.nll_multi_anchor
total = 0
for anchor in anchors:                 # R passes
    for j in range(K_q):               # K_q questions
        q_text = [batch[b]['questions'][j][0] for b in range(B)]
        a_text = [batch[b]['questions'][j][1] for b in range(B)]
        total += forward_anchor(anchor, h, q_text, a_text, K=K)
return total / (R * K_q)
```

The same $h$ is reused across all $R \cdot K_q$ reader passes — this is the
source of the **q-invariance** pressure: $h$ must encode something that
serves all $K_q$ questions, so it cannot shortcut-encode any single
$(q, y^{*})$ pair.

### 3.2 Concept loss (LaViT-style)

$$
\mathcal{L}_{\text{concept}}
= 1 - \frac{1}{B \cdot K} \sum_{b,k} \cos\!\Bigl(\mathrm{MLP}_{\text{concept}}(h_{b,k}),\; V_{\text{sem}}[b,\, k \bmod T_v,\, :]\Bigr).
$$

Pooling from the natural visual-token count $T_v$ down to $K$ is by modular
indexing $k \mapsto k \bmod T_v$ — the simplest viable assignment. This is
flagged as a known crude-pooler choice; a learned attention pooler is the
proliferated-project upgrade.

When `cfg.w_concept == 0` (cell C4), the concept term is skipped entirely
and $V_{\text{sem}}$ is not even computed — saving a vision-tower forward
per step.

### 3.3 Norm regulariser

$$
\mathcal{L}_{\text{norm}} = \frac{1}{B \cdot K} \sum_{b,k} \bigl(\|h_{b,k}\|_2 - \mu\bigr)^2,
\qquad \mu = 57.86.
$$

The target $\mu = 57.86$ is the empirically measured mean post-merger
visual-token norm for Qwen2.5-VL-7B. This term keeps $h$ on the natural
scale of the embeddings the anchors expect to see. In practice, the mini
training run drives $\|h\|$ from $\approx 190$ at step 0 to $\approx 57.4$
within 20 steps.

### 3.4 Curriculum

```
nll_weight(t)   = cosine ramp 0.1 → 1.0 over [0, T_warmup], then 1.0
concept_weight  = constant w_concept (default 0.3)
norm_weight(t)  = cosine ramp 0.0 → 0.1 over [0, T_warmup], then 0.1
```

with `T_warmup = LossConfig.curriculum_warmup_steps` (default 200). The
ramp is symmetric cosine: $\gamma(t) = \tfrac{1}{2}(1 - \cos(\pi t / T_w))$.
The motivation is that high concept-pressure early plants $h$ near the
manifold *before* the NLL gradient dominates; if NLL dominated from step 0,
the optimiser would find the off-manifold shortcut basin first and the
concept term could not pull it back.

---

## 4. Implementation details

### 4.1 LoRA injection on a multimodal model

`peft.inject_adapter_in_model` is preferred over `get_peft_model` here
because the latter wraps the input module in a `PeftModel` that requires
`prepare_inputs_for_generation` — a method `Qwen2_5_VLTextModel` does not
expose. `inject_adapter_in_model` mutates the LM in place: it walks the
target-module name list and substitutes `nn.Linear` with `LoraLinear`,
leaving the parent shape untouched. The generator object remains a
`Qwen2_5_VLForConditionalGeneration` whose `model.model.language_model` now
contains LoRA layers. Vision tower and LM head are untouched and stay
frozen.

### 4.2 Trainable embedding rows via splice

Because the embedding table is large (151 668 × 3584 ≈ 540 M params) and we
only want three rows trainable, we avoid `register_hook` on the table.
Instead, at forward time we compute `inputs_embeds = embed(input_ids)` (no
grad — the table is frozen) and then overwrite the three new-token positions
with rows from `new_emb` (a fresh fp32 `nn.Parameter`). The boolean-index
assignment `inputs_embeds[mask] = new_emb[i].to(bf16)` creates a
non-leaf grad node whose backward path correctly attributes grad to
`new_emb`.

### 4.3 Gradient checkpointing

`model.gradient_checkpointing_enable()` registers a forward hook on the
embedding lookup that flips `output.requires_grad_(True)` on the lookup's
output. That hook collides with our boolean-mask in-place write: the lookup
output becomes a *leaf* with `requires_grad=True`, which `inputs_embeds[mask] = row`
refuses. The trainer therefore calls `model.disable_input_require_grads()`
immediately after enabling GC. The gradient still flows correctly through
`new_emb` because the splice itself produces a non-leaf tensor with
`requires_grad=True` (sufficient for downstream LoRA layers to checkpoint).

### 4.4 Reader weight sharing

`load_anchors(paths, generator_model, generator_processor, dtype)` checks
whether `paths[0]` matches `generator_model.config._name_or_path`. If so,
the first anchor entry is `{model: generator_model, shared_with_generator: True, ...}`
— i.e., reader-1 *is* the generator. Its LoRA is whatever the generator's
LoRA is at this step, which means reader-1's reading of $h$ also evolves
during training. This is a deliberate round-3 simplification: it avoids
double-loading 14 GB of weights, at the cost of a slightly less clean
"frozen reader" framing. Reader-2 (and beyond) is loaded fresh, frozen, no
LoRA.

---

## 5. Data pipeline

### 5.1 Sources

The data layer in `src/vl/data/` exposes `load_gqa`, `load_clevr`,
`load_tallyqa`, all with the same per-image record schema:

```python
{
  'image_id': str,
  'image_loader': callable() -> PIL.Image,   # lazy
  'questions':   list[(q, a)],               # >= K_q kept
  'held_out_q':  (q, a) | None,              # one reserved at load time
  'source':      'gqa' | 'clevr' | 'tallyqa',
}
```

GQA uses `lmms-lab/GQA` configs `{split}_balanced_instructions` (Q/A) and
`{split}_balanced_images` (PIL); we group by `imageId` and filter to
$\ge n_{\text{per-image}} + 1$ Q/A. The `+1` is reserved as a held-out
question $q'$ for evaluation. CLEVR and TallyQA come from
`HuggingFaceM4/the_cauldron` configs `clevr` and `tallyqa`. The latter two
attempt to load lazily and raise a clear error if the cauldron config is
not cached locally — this lets the local A6000 smoke path use GQA-only
without forcing the user to download 50 GB of cauldron data.

### 5.2 Mixed dataset and multi-Q sampler

`MixedDataset(cfg.data)` is an `IterableDataset` that concatenates per-source
records up to `cfg.n_samples` total, weighted by `cfg.mix` proportions, then
shuffles the concatenation. The collator returned by
`make_collator(K_q, shuffle_qa_within_image)` does the per-step work:

1. For each record in the batch, sample $K_q$ Q/A pairs uniformly without
   replacement from `record['questions']`.
2. Resolve lazy `image_loader()` to a PIL image.
3. If `shuffle_qa_within_image=True` (cell C5 random-control), flatten the
   resulting $B \cdot K_q$ Q/A tuples and randomly permute, then reshape — so
   each image is paired with $K_q$ questions drawn (in expectation) from
   *other* images in the batch. This is the random-control negative for the
   regularisation-vs-grounding falsification test.
4. Return `{'images': [...], 'questions_per_image': [[...K_q...], ...], 'sources': [...]}`.

---

## 6. Optimisation

### 6.1 Optimiser

Two AdamW parameter groups, $\beta = (0.9,\, 0.95)$, weight decay 0:

| Group | Params | LR (default) |
|---|---|---|
| 1 | LoRA adapters + concept MLP | $5 \times 10^{-5}$ |
| 2 | `new_emb` (3 × D) | $5 \times 10^{-3}$ |

The 100× higher LR on group 2 is because the new-token embeddings are
randomly initialised and need to reach their natural region quickly; LoRA
already starts from a meaningful point (lora_B = 0 produces an identity
change to the base model).

### 6.2 LR schedule and gradient clipping

Cosine annealing from the initial LR to 10 % of `lr_lora` over `max_steps`,
no separate LR warmup (the curriculum on loss weights provides the warmup
function). Gradient clipping at $\| g \|_2 = 1.0$ is applied across all
trainable parameters before each optimiser step.

A known limitation: `CosineAnnealingLR.eta_min` is shared across param
groups, so group 2's effective floor is $0.1 \cdot \mathrm{lr\_lora}$ rather
than $0.1 \cdot \mathrm{lr\_token}$. The cosine ramp shape is preserved.

### 6.3 Per-step loop

```python
h     = forward_generator(model, new_emb, new_token_ids, processor, images, cfg.model)
v_sem = get_v_sem(model, processor, images) if cfg.loss.w_concept != 0 else None
losses = combined(h=h, batch=..., anchors=anchors, v_sem=v_sem,
                  concept_mlp=concept_mlp, cfg=cfg.loss, step=global_step)
losses['total'].backward()
torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
```

### 6.4 Held-out evaluation

Every `eval_every_steps` steps, the trainer evaluates on a fixed held-out
set of 30 records (the held-out Q/A reserved by the data loader). Per
record, it runs `forward_generator` to produce $h$ and then calls
`forward_anchor` with the held-out $(q', a')$ for each anchor. The mean NLL
per anchor is appended to `eval.jsonl`.

### 6.5 Checkpointing and resume

Every `ckpt_every_steps` steps (and at end-of-run), the trainer writes
`ckpt_step{N}.pt` containing:

- `step` (integer)
- `lora_state` (PEFT state dict via `peft.get_peft_model_state_dict`)
- `new_emb` (CPU tensor)
- `concept_mlp` (CPU state dict)
- `optim`, `sched` (state dicts)

Resume: `--resume` globs for the latest `ckpt_step{N}.pt`, restores all of
the above, and continues training at `start_step = N`. Verified end-to-end
in the verification run (steps 50 → 55 with held-out NLL improving 3.36 →
1.54 across the 50 steps).

### 6.6 Wall-clock guard

`cfg.trainer.max_time` accepts strings like `"23h"`, `"30m"`, `"1d"`. The
loop tracks `time.monotonic() - start_time` and, if the budget is exceeded,
saves a checkpoint and exits with return code 124. The cluster sbatch
template uses this for chained jobs (it Slack-notifies on rc=124 but does
not auto-resubmit per the project-owner rule).

---

## 7. Evaluation (planned)

The eval suite at `src/vl/eval.py` is currently a stub. The intended
suite, per `docs/inherited/ROUND3_POC_DESIGN.md` §7, comprises:

1. Held-out NLL on $q'$ under reader-1 (already exercised live during
   training; the standalone eval will run it on the full held-out split).
2. Reader-transfer NLL under reader-2 (Monet-SFT-7B), same-question and
   held-out-question.
3. Steering probe — zero-out / permute / Gaussian-noise perturbations
   applied to $h$ before reader consumption.
4. 5K visual-grounding stress test (MMVP / NaturalBench / BLINK / MMStar /
   CV-Bench / POPE / VSR), C1 cell only.
5. Four control conditions per `EVAL_BENCHMARK_PLAN.md` §C — blank gray,
   random natural, adversarial mismatch, shuffled pixels.

Each cell of the round-3 5-cell sweep (C1 full, C2 R=1, C3 K\_q=1, C4
$w_{\text{concept}}=0$, C5 random-control) runs 1000 training steps and is
gated by five hard pass thresholds enumerated in §1 of the round-3 design
doc.

---

## 8. Hardware and scale

The implementation runs on a single GPU per process. Local validation used
an NVIDIA RTX A6000 (48 GB):

- Gradient probe (Qwen2.5-VL-3B, $K=4$, $r=8$, $B=1$, $K_q=2$, single
  shared anchor): 8.45 GB peak.
- Smoke (Qwen2.5-VL-7B, $K=4$, $r=16$, $B=1$, $K_q=2$, single shared
  anchor, 10 steps): ~17 GB peak.
- Mini-training (Qwen2.5-VL-3B, $K=4$, $r=16$, $B=1$, $K_q=2$, 50 steps):
  $\|h\|$ converged to 57.4 (target 57.86); held-out NLL 3.36 → 1.54 over
  steps 25 → 50.

The intended cluster configuration per round-3 spec is 4 × H100 80 GB
(`gpu-4farm` partition on bioai), $B = 4$ per GPU, $K = 16$, $K_q = 3$,
$R = 2$, LoRA $r = 32$, with per-cell 1000 steps and ~6 h wall-clock per
cell. **Multi-GPU gradient synchronisation is not yet wired** — the cluster
sbatch's `accelerate launch --num_processes=4` currently spawns four
independent training processes. Wiring DDP via `accelerate.Accelerator` or
`torch.distributed.init_process_group` is the most important pre-cluster
TODO; see `docs/HANDOFF_OVERNIGHT.md`.

---

## 9. Code layout

```
src/vl/
  config.py            # dataclasses + YAML loader
  paths.py             # MACHINE-resolved storage roots
  curriculum.py        # cosine-warmup loss-weight schedules
  model.py             # build_generator / forward_generator / get_v_sem
  losses.py            # nll_multi_anchor / concept_loss / norm_loss / combined
  readers.py           # load_anchors / forward_anchor (h-splice, no real image)
  data/
    gqa.py / clevr.py / tallyqa.py   # per-source record loaders
    mixed.py           # MixedDataset (IterableDataset) + make_collator
  trainers/
    sft_anchor.py      # the variant-A training loop (this method)
    grpo_vlpo.py       # variant B (deferred)
  train.py             # CLI: load config → ensure dirs → dispatch to trainer
```

All training is configured via YAML files under `configs/`; the round-3
5-cell sweep lives at `configs/round3/{C1_full, C2_R1, C3_Kq1, C4_no_concept, C5_random_control}.yaml`.
A reduced `configs/smoke.yaml` (10 steps, 7B + small LoRA) and
`configs/mini.yaml` (50 steps, 3B + small LoRA) exercise the local A6000
path.
