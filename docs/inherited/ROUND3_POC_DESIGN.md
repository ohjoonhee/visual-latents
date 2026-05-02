# Round-3 POC Design — Reader-Grounded Latent Visual Reasoning

**Date:** 2026-05-02
**Status:** committed design; awaiting compute window
**Companion to:** `REPORT.md` (rounds 1+2), `JOURNAL.md`, `HANDOFF.md`,
  `docs/AUX_LOSS_AND_ARCH_DESIGN.md`, `docs/VLM_SURVEY.md`,
  `docs/EVAL_BENCHMARK_PLAN.md`, `docs/TRAINING_DATA_PLAN.md`,
  `docs/VARIANT_B_GRPO_DESIGN.md`.

---

## §1 — Round-3 mission and decision criteria

Round-3 is a **POC, not a real training run.** Its single purpose is to
*pre-validate the combined recipe* — LIVR-style same-VLM emitter + LaViT-style
auxiliary grounding + multi-reader NLL + multi-Q invariance + norm regularization
— before committing the proliferated multi-week training project.

Round-2 narrowed the failure surface:

- **Within-reader held-out NLL** improved by 30–40 % under each of three
  independent geometric mitigations (mit-B λ=0.1 → 2.39 nat; mit-A K=4 r=1 →
  2.65; multi-Q N=2 → 2.40; multi-Q N=5 → 3.28; vs vanilla 3.91).
- **Reader-transfer (POC 3) was NOT fixed by any geometric mitigation.**
  φ₂=Monet-7B NLL: vanilla 8.09 → mit-B 8.55 → mit-A 8.30. Geometric mitigations
  change *which direction* in φ₁'s embedding space the optimizer finds, but
  not whether that direction exploits φ₁-specific decoder structure.
  (`REPORT.md` §16, "Round-2 reader-transfer addendum".)

Round-3 must establish that the combined recipe — at training time, in a
generator with autoregressive structure, with a reader-family-shared anchor —
**actually** closes the transfer gap and **isn't** generic regularization in
disguise (arXiv:2004.05704). Five hard pass thresholds, all must clear:

| # | Metric | Pass threshold | Round-2 baseline | Why this number |
|---|---|---|---|---|
| 1 | Held-out NLL on q' under φ₁ (POC2 protocol) | **≤ 2.5 nat** | vanilla 3.91; mit-B λ=0.1 2.39 | Mit-B already hit 2.39 within-reader; the combined recipe must at least match a single-mitigation baseline. |
| 2 | Reader-transfer NLL on Monet-7B (POC3 protocol, q same as train) | **≤ 4.5 nat** | vanilla 8.09; round-2 latents 8.30–8.55 (no improvement) | Halving the transfer drop is the binding test that round 2 failed. ≤ 4.5 means we have closed > 50 % of the gap to within-reader fit. |
| 3 | Random-control delta (real GQA mix vs shuffled-(image,q,y) on the same recipe) | **gain on real ≥ 2× gain on shuffled control** | not measured in round 2 | arXiv:2004.05704: random/insensible cues fake grounding gains. A 2× separation is the bar at which "regularization explains it" becomes implausible. |
| 4 | Steering probe (mirror `steering_probe.py`, 30 samples × 8 perturbations) | **zero_pos_3, permute_within, gauss_noise_1.0 each ≥ +1.5 nat ΔNLL** | round-2 vanilla: zero_pos_3 +4.12, permute_within +2.56, gauss_noise_1.0 +4.06 | The combined recipe must preserve causal sensitivity (not flatten to placeholder, contra arXiv:2512.21711). +1.5 is the conservative threshold — well above the no-effect band but allowing some smoothing from manifold-anchored latents. |
| 5 | 5K visual-grounding stress test (`docs/EVAL_BENCHMARK_PLAN.md` §B): blank-image control (C1) accuracy drop | **≥ 5 pp drop on real-image acc - blank acc, on the perception-pure subset (MMVP + BLINK-Spatial + CV-Bench-3D)** | not measured in round 2 | Blank image isolates language-prior reliance. ≥ 5 pp drop is the minimum signal that h carries image-specific content. (vanilla Qwen2.5-VL-7B drops ~10 pp; we need at least half that to claim grounding.) |

**Decision logic.**

- **All 5 pass:** advance to proliferated training (`docs/TRAINING_DATA_PLAN.md`
  Mix 3.2 medium → Mix 3.3 full).
- **#2 fails (transfer drop > 4.5 nat):** the multi-reader NLL did not propagate
  the structural pressure. Two pivots before scrapping:
  (a) add R=3 with VL-Rethinker-7B as a third *training* reader; (b) drop the
  separate-reader framing entirely and switch to LIVR-style single-model
  attention bottleneck (`AUX_LOSS_AND_ARCH_DESIGN.md` §B.1 con-(i)).
  Either pivot is itself a round-3.5; do *not* proceed to proliferated.
- **#3 fails (random-control gain ≥ 50 % of real gain):** the recipe is generic
  regularization. **Reformulate.** This is the hard kill — arXiv:2004.05704
  outcome.
- **#1 passes but #4 fails:** the latents are inert (Coconut-style). Either
  K is too large (drop to K=8 or K=4) or λ_concept is dominating to the
  point of identity-collapse (drop to 0.1).
- **#1 and #2 pass, #3 and #4 pass, #5 fails:** real grounding present at the
  NLL level but the readers don't translate it into accuracy. Investigate
  greedy-decode protocol, then proceed cautiously.

Round-3 is a **gate, not a scaling exercise.** The compute budget is fixed at
5 cells × 1000 steps on 4×H100 (≈ 24h); see §6.

---

## §2 — Architecture (pinned)

**Choice:** B.1 same-VLM-with-special-tokens (LIVR-style, `arXiv:2512.21218`),
with LIVR's Stage-1 attention masking and LoRA-r=32. Per
`AUX_LOSS_AND_ARCH_DESIGN.md` §B.5 final recommendation. Generator and reader
share the SAME weights and SAME LoRA adapter for round-3 simplicity (the
proliferated project will diverge them; `AUX_LOSS_AND_ARCH_DESIGN.md` §C.4
"Cross-arch readers — feasible but not in round-3").

### 2.1 Forward layout

```
Generator forward (image-only input, NO question — q-invariant generator per §A.4):

  [BOS] <|im_start|>system You are a helpful assistant.<|im_end|>
        <|im_start|>user
          <|vision_start|>
            IMG_TOKEN_1 IMG_TOKEN_2 ... IMG_TOKEN_{T_v}    ← real image (T_v ≈ 200–400)
          <|vision_end|>
          <|latent_start|>                                  ← new special token
            <|latent|>_1 <|latent|>_2 ... <|latent|>_K     ← K=16 new special tokens
          <|latent_end|>                                    ← new special token
        <|im_end|>
        <|im_start|>assistant

         h_{1:K}  =  hidden states extracted from the K <|latent|> positions
                     at the OUTPUT of the FINAL transformer layer.

Reader forward (frozen reader, h spliced into K <|image_pad|> slots, NO image):

  [BOS] <|im_start|>system You are a helpful assistant.<|im_end|>
        <|im_start|>user
          <|vision_start|>
            <|image_pad|>_1 ... <|image_pad|>_K            ← K spliced positions
          <|vision_end|>
          {question}
        <|im_end|>
        <|im_start|>assistant
          {answer}<|im_end|>
                    ↑
                    NLL computed only over answer-token positions
```

This matches LIVR's "K new special tokens" recipe verbatim
(`AUX_LOSS_AND_ARCH_DESIGN.md` §B.1, citing `arXiv:2512.21218`). The slot-
injection primitive is the one already validated in `tune.py` (verbatim
`build_prompt_with_k_pads` and `forward_with_latents` from POC 1; see §8.1).

### 2.2 Stage-1 attention masking (LIVR Stage-1)

Per LIVR (`arXiv:2512.21218`), training has two stages: Stage 1 (4 epochs)
with masking; Stage 2 (6 epochs) without. Round-3 POC's 1000 steps map to
Stage-1-only (we never reach Stage 2's regime in 1000 steps; defer Stage 2
to proliferated).

**Mask rule, computed once at attention construction time:**

- `<|latent|>` query positions can attend to: image tokens, prior `<|latent|>`
  positions (autoregressive), and the BOS/system prefix. Standard causal mask
  in this region.
- **Answer-token query positions (during reader pass; equivalently, the
  generator's auxiliary answer-prediction head if used) CANNOT attend to image
  tokens.** They may only attend to: BOS/system prefix, the K `<|latent|>`/
  `<|image_pad|>` positions, the question tokens, and prior answer tokens.

This is the LIVR "answer tokens cannot attend to image" bottleneck. Without
it, h is not actually a bottleneck — answer tokens flow around h via direct
attention to image tokens.

**Implementation:** build a custom 4D `attention_mask` of shape
`[B, 1, L, L]`, with `-inf` at (answer_idx, image_idx) pairs. Pass via
`model(..., attention_mask=mask_4d)`. Qwen2.5-VL's attention layer accepts
4D masks; verified by `transformers >= 4.45` source. (Round-2 `tune.py` did
not use 4D masks; this is the only architectural primitive we have to add.)

### 2.3 LoRA (PEFT)

LoRA r=32, applied to **all attention projections and MLP gate/up/down across
all 28 transformer layers** of the Qwen2.5-VL-7B LLM. NOT applied to the
vision tower or the merger (those stay frozen — they are the V_sem teacher,
see §3.2).

```python
# Code-skeleton (using peft 0.11+):
from peft import LoraConfig, get_peft_model
lora_cfg = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",       # self-attn (4 per layer)
        "gate_proj", "up_proj", "down_proj",          # MLP (3 per layer)
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16,
)
# Freeze base
for p in model.parameters():
    p.requires_grad_(False)
# Inject LoRA on the LM only (vision tower stays bare-frozen)
model.model = get_peft_model(model.model, lora_cfg)
```

Param count: 28 layers × 7 modules × (in_dim + out_dim) × r ≈ 50M trainable
LoRA params. Per `AUX_LOSS_AND_ARCH_DESIGN.md` §B.1 estimate.

### 2.4 Trainable embedding rows

Per LIVR: the K `<|latent|>` token embedding rows (and the
`<|latent_start|>` / `<|latent_end|>` rows) are **trainable**, even though
the rest of the embedding table is frozen.

```python
# Add new special tokens
new_tokens = ["<|latent|>", "<|latent_start|>", "<|latent_end|>"]
tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
model.resize_token_embeddings(len(tokenizer))
# By default the new rows are random; init from the mean of nearby vision tokens
with torch.no_grad():
    img_pad_emb = model.get_input_embeddings().weight[IMAGE_PAD_ID].clone()
    for tok in new_tokens:
        new_id = tokenizer.convert_tokens_to_ids(tok)
        model.get_input_embeddings().weight[new_id] = img_pad_emb + 0.02 * torch.randn_like(img_pad_emb)
# Mark only those rows trainable
emb_w = model.get_input_embeddings().weight
emb_w.requires_grad_(False)  # default
new_ids = [tokenizer.convert_tokens_to_ids(t) for t in new_tokens]
# Use a parameter mask in the optimizer (see optimizer construction in §5).
```

Note: gradients on the embedding table are sparse if you write the optimizer
naively; cleaner is to clone the new rows out as a small `nn.Parameter` and
splice them in at forward time (avoids writing `register_hook` over the full
embedding). The proliferated project should switch to the cleaner version;
round-3 can use the simpler register-hook approach for speed.

### 2.5 Frozen vs trainable parameter inventory

| Component | Status | Param count |
|---|---|---|
| Qwen2.5-VL-7B LLM base weights | **Frozen** | 7.07 B |
| Vision tower (ViT + projector + merger) | **Frozen** | 675 M (also serves as φ₁'s vision branch and as V_sem teacher in §3.2 — single load, no double-storage) |
| LoRA r=32 across LLM (q/k/v/o/gate/up/down × 28 layers) | **Trainable** | ~50 M |
| Embedding rows for `<|latent|>`, `<|latent_start|>`, `<|latent_end|>` | **Trainable** | 3 × 3584 ≈ 11 K |
| Concept-MLP `D → D/2 → D` (§3.2) | **Trainable** | ~25 M (3584 × 1792 × 2) |
| Reader (Qwen2.5-VL-7B + same LoRA) | **Same parameters as generator** — shared weights, see §2.6 |
| Reader 2 (Monet-SFT-7B) | **Frozen, no LoRA** | 7.07 B |

Total trainable: ~75 M. Optimizer state (Adam β₁=0.9, β₂=0.95) is ~600 MB
in fp32. Comfortable on 4×H100.

### 2.6 Generator-reader weight sharing for round-3

The generator (with LoRA) emits h. Reader-1 (φ₁ = Qwen2.5-VL-7B-Instruct
**with the SAME LoRA**) consumes h + question and produces NLL. Reader-2
(φ₂ = Monet-SFT-7B, **frozen, no LoRA**) consumes h + question and produces
NLL.

Why share weights for round-3: simplest implementation, no double-load of
the 7B base. The proliferated project will likely diverge them (give the
reader frozen base + no LoRA so the generator's LoRA doesn't move the
reader's reading geometry as a side effect). Note this in the proliferated
plan as an open design knob.

**Caveat the reviewer will raise:** if the reader has the same LoRA as the
generator, then "the reader is frozen" is technically false — it has trainable
weights via the LoRA. Round-3 will report this as "weights shared, gradient
through reader updates the same LoRA that the generator uses." The
proliferated project will run a comparison cell where the reader uses the
frozen base (no LoRA) — this is the cleaner story but doubles the on-disk
parameter count.

### 2.7 K and ablations

- **K = 16** is the round-3 default (LIVR's K=16 default per
  `arXiv:2512.21218`; LIVR's sweep was {4, 8, 16, 32}).
- Round-3 ablates K=4 and K=8 only if the K=16 cell hits the saturation
  signature observed in round-1 POC 4 (K=16 inter-sample cos sim 0.281
  vs K=4's 0.051). The K-ablation is not in the 5-cell budget; it runs
  off-budget *only if* C1 exhibits saturation at the 500-step checkpoint.

### 2.8 Code skeleton — model construction

```python
# model_round3.py — single-file generator+reader skeleton.
# All trainable parameters discoverable via list(model.parameters() if p.requires_grad).

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MONET_SFT_PATH = "NOVAglow646/Monet-SFT-7B"  # round-3 swaps to SFT-only sibling
NEW_TOKS = ["<|latent|>", "<|latent_start|>", "<|latent_end|>"]
K = 16

processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer
tokenizer.add_special_tokens({"additional_special_tokens": NEW_TOKS})

# Generator + reader-1 share weights (see §2.6)
gen = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
gen.resize_token_embeddings(len(tokenizer))
for p in gen.parameters():
    p.requires_grad_(False)

lora_cfg = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
gen.model = get_peft_model(gen.model, lora_cfg)

# Trainable new-token embedding rows (cloned to own parameter)
new_ids = [tokenizer.convert_tokens_to_ids(t) for t in NEW_TOKS]
emb = gen.get_input_embeddings()
new_emb = torch.nn.Parameter(emb.weight[new_ids].detach().clone().float())
# At forward time, splice new_emb back into the embedding lookup at new_ids.

# Concept-MLP (§3.2)
HIDDEN = 3584
concept_mlp = torch.nn.Sequential(
    torch.nn.Linear(HIDDEN, HIDDEN // 2),
    torch.nn.GELU(),
    torch.nn.Linear(HIDDEN // 2, HIDDEN),
).to(torch.bfloat16).cuda()

# Reader-2 — Monet-SFT-7B, fully frozen
reader2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(MONET_SFT_PATH, torch_dtype=torch.bfloat16)
reader2.resize_token_embeddings(len(tokenizer))
for p in reader2.parameters():
    p.requires_grad_(False)
```

---

## §3 — Loss (pinned)

The full combined loss, as fixed in `AUX_LOSS_AND_ARCH_DESIGN.md` §A.6:

```
ℒ_total(θ) = w_NLL(t) · ℒ_NLL_multi
           + 0.3 · ℒ_concept
           + λ_norm(t) · ℒ_norm
```

where w_NLL(t) and λ_norm(t) follow the curriculum schedule (§3.5).

### 3.1 ℒ_NLL_multi — multi-reader, multi-question, q-invariant generator

```python
# Per training step.
# Generator emits h ONCE per image (no question — q-invariance baked architecturally, §A.4).
h = generator_forward(image=I)              # [B, K=16, D=3584]

# Reader NLL averaged over R=2 readers and K_q=3 questions per image:
nll = 0.0
for i in range(R):                           # R=2: φ₁=shared-LoRA, φ₂=Monet-SFT-7B
    for j in range(K_q):                     # K_q=3
        # Reader sees: prompt template + K spliced h positions + q_j + answer_j
        logits = readers[i](q=q_lists[j], h=h, image=None)   # NO image to reader
        nll += cross_entropy(logits[answer_positions], y_lists[j])
ℒ_NLL_multi = nll / (R * K_q)                # mean over readers and questions
```

- **R=2** (round-3 default): φ₁ = Qwen2.5-VL-7B-Instruct (same weights as
  generator + same LoRA), φ₂ = Monet-SFT-7B (`NOVAglow646/Monet-SFT-7B`,
  frozen, no LoRA). Note we use **Monet-SFT-7B not Monet-7B (post-RL)**
  per round-2 caveat in `REPORT.md` §9 ("Cleaner sibling would be Monet-
  SFT-7B (no RL specialization)"). Download: ~16 GB, prerequisite §10.
- **K_q=3** questions per image per training step
  (`AUX_LOSS_AND_ARCH_DESIGN.md` §A.4.2 default). The same h_{1:K} is reused
  across all K_q reader passes. This is the **q-invariance** mechanism: h
  must encode something that works for all 3 questions, so it cannot
  shortcut-encode any single (q, y*) pairing.
- **Sum, not alternate** (per `AUX_LOSS_AND_ARCH_DESIGN.md` §A.2.3
  "Recommendation: sum"): all R × K_q reader forwards happen in one step;
  loss is summed; one backward. Alternation introduces gradient bias.
- **Gradient path:** through frozen reader weights into h, then into LoRA
  + new-token embeddings + concept-MLP. Standard Variant-A flow, just with
  multi-reader and multi-Q at training time.

### 3.2 ℒ_concept — LaViT-style cosine to teacher visual features

Per LaViT (`arXiv:2601.10129`) λ=0.3, verified in
`AUX_LOSS_AND_ARCH_DESIGN.md` §A.1 verbatim from the v1 HTML:
**`ℒ_concept = 1 − (1/B) Σ CosSim(φ_mlp(h_z), V_sem)`**.

```python
# Teacher V_sem: φ₁'s POST-MERGER visual tokens.
# (Choice (a) from AUX_LOSS_AND_ARCH_DESIGN.md §A.1.1; (i) results/visual_baseline.json
# already characterizes this manifold; (ii) it is dim D=3584 by construction —
# exactly the shape h must occupy.)

with torch.no_grad():
    # Teacher path: vision tower + projector + merger, frozen — same weights as
    # the generator's vision branch. NO LoRA on this path (LoRA is on the LLM only).
    V_sem = generator.visual(image=I)       # [B, T_v, D],  T_v ≈ 200–400 per image

# Student: project h through bottleneck MLP, then cos-sim to V_sem.
h_proj = concept_mlp(h)                       # [B, K=16, D=3584]

# Pooler: assign each h_k a target. Use modular indexing per AUX_LOSS_AND_ARCH_DESIGN.md
# §A.1.2 — simplest viable target. Each h_k is matched to V_sem[:, k mod T_v, :].
B, K, D = h.shape
target_idx = torch.arange(K, device=h.device) % V_sem.shape[1]  # [K]
targets = V_sem[:, target_idx, :]             # [B, K, D]

ℒ_concept = (1 - F.cosine_similarity(h_proj, targets, dim=-1)).mean()
```

**Bottleneck-MLP shape: D → D/2 → D = 3584 → 1792 → 3584** (~25 M params,
GELU activation). The bottleneck is critical (see §9 risk #1): without it,
identity-via-MLP collapses h to V_sem and the method degenerates to "use
visual tokens directly as soft prompts." With D/2 in the middle, identity
needs at least 2-step factorization → optimization friction.

**Pooler simplicity caveat:** `k mod T_v` is crude (per
`AUX_LOSS_AND_ARCH_DESIGN.md` §D.3 risk 4). Round-3 accepts this; if
ℒ_concept dominates and the pooler turns out to be the limiting factor, the
proliferated project switches to a **learned attention pooler** (K queries
cross-attending into V_sem). Not in the round-3 budget.

**λ_concept = 0.3, fixed across the run** (LaViT default;
`AUX_LOSS_AND_ARCH_DESIGN.md` §A.6). Round-3 does NOT sweep λ_concept (the
5-cell budget targets other axes).

### 3.3 ℒ_norm — distributional regularization toward natural-token norm

Per `AUX_LOSS_AND_ARCH_DESIGN.md` §A.3 (Form 1, the round-2-verified form):

```python
# Round-2 verified target.
NATURAL_NORM = 57.86       # results/visual_baseline.json, mean post-merger token norm
ℒ_norm = ((h.norm(dim=-1) - NATURAL_NORM) ** 2).mean()       # scalar; per-token, mean-reduced
```

Round-2 validated this exact form: λ=0.1 was the free-lunch sweet spot
(−39 % held-out, no training-fit penalty). λ=10 broke training. Round-3
adopts λ=0.1 with a 200-step ramp (§3.5).

### 3.4 Final loss assembly

```python
loss = (w_NLL * ℒ_NLL_multi
        + 0.3 * ℒ_concept
        + λ_norm * ℒ_norm)
loss.backward()
```

### 3.5 Curriculum schedule

Per `AUX_LOSS_AND_ARCH_DESIGN.md` §A.6 schedule table (200-step warmup +
600-step main + 200-step anneal). The mechanism: high concept-pressure early
plants h near the manifold *before* the NLL gradient dominates; if NLL
dominates from step 0, the optimizer finds the off-manifold shortcut basin
first and ℒ_concept can't pull it back.

| Step range | w_NLL (scalar on ℒ_NLL_multi) | λ_concept | λ_norm |
|---|---|---|---|
| 0 → 200 (warmup) | linear 0.1 → 1.0 | **0.3** (constant) | linear 0 → 0.1 |
| 200 → 800 (main) | 1.0 | 0.3 | 0.1 |
| 800 → 1000 (anneal) | 1.0 | 0.3 → 0.1 (linear) | 0.1 |

**Anneal rationale** (`AUX_LOSS_AND_ARCH_DESIGN.md` §A.6): reduce ℒ_concept
weight at the end to test whether the gains are *sticky* (h stays on manifold
without strong pull). If held-out NLL spikes at step 800, ℒ_concept was a
crutch. If it stays flat or improves, h has internalized the manifold prior.

LaViT's curriculum has an additional "Sensory Gating" component (γ(t) cosine
ramp on visual attention, T_w=400). For our setting that maps onto gating
the **NLL weight w_NLL** itself (the analog: `ℒ_NLL_reader` is the part that
feeds reader-NLL gradient back into h, and we want to gate it on, not the
visual attention path). T_w = 200 in our schedule (half of LaViT's, scaled
to our 1000-step budget).

---

## §4 — Data (pinned for round-3)

Pilot mix per `TRAINING_DATA_PLAN.md` §3.1, sized for round-3:

| Dataset | HF path | Samples | Qs/img | Frac |
|---|---|---|---|---|
| GQA-balanced | `lmms-lab/GQA` cfg `train_balanced_instructions` | 6500 (≥3 Q/img filter) | filter ≥3 | 65 % |
| CLEVR | `HuggingFaceM4/the_cauldron` cfg `clevr` | 2500 (250 imgs × 10 Q) | 10 | 25 % |
| TallyQA | `HuggingFaceM4/the_cauldron` cfg `tallyqa` | 1000 (≥3 Q/img filter) | ≥3 | 10 % |
| **Total** | | **~10,000 samples** | | |

**Why this composition:**

- **GQA-balanced** carries the multi-Q bulk: 13 Q/img average, 94 % single-
  token answers (probed in `TRAINING_DATA_PLAN.md` §8). Already used in
  rounds 1+2; pipeline in place.
- **CLEVR** is the **grounding-immune control** (synthetic, no language
  priors possible). 100 % single-token. 10 Q/img exactly. If random-control
  passes on GQA but fails on CLEVR, the gain on GQA was language priors;
  CLEVR is the falsifier.
- **TallyQA** counting requires visual access. Counting cannot be
  shortcut-encoded purely from question.

### 4.1 Multi-Q sampler

For each training step, the loader yields a batch of B=4 images, with K_q=3
questions per image (sampled at *step time* from each image's question list,
not pre-frozen at data-prep time — randomization across epochs spreads
question coverage). Held out: 1 Q/image (q' for held-out NLL eval) is
randomly fixed at data-prep time and never used in training.

```python
# prepare_data_round3.py output schema (one row per image):
{
  "image_id": "...",
  "image_path": "...",
  "questions": [
    {"q": "...", "a": "..."},   # 3+ training Qs
    ...
  ],
  "held_out_q": {"q": "...", "a": "..."},  # 1 Q reserved for held-out NLL eval
  "source": "gqa" | "clevr" | "tallyqa",
}
```

Loader at training time:

```python
batch = [
    {"image": load(image_path),
     "questions": random.sample(row["questions"], K_q),  # K_q=3
    }
    for row in next_batch_of_4
]
```

### 4.2 Random-control negative cell (cell C5 in §6)

Per `AUX_LOSS_AND_ARCH_DESIGN.md` §A.5.1 "Shuffled (image, question) pairs":
identical training mix, but each batch shuffles (image, q, y) tuples so that
image_b is paired with q_{σ(b)} and y_{σ(b)} where σ is a within-batch
permutation. Run as a **separate ablation cell** (cell C5), not interleaved
with C1.

```python
# Random-control batch construction:
images = stack([row["image"] for row in batch])
qs = stack([row["questions"] for row in batch])
ys = stack([row["answers"] for row in batch])
sigma = torch.randperm(B)
batch_real    = (images, qs, ys)
batch_shuffled = (images, qs[sigma], ys[sigma])  # same image, different (q, y)
```

Decision rule (gate #3 in §1): gain on real ≥ 2× gain on shuffled. If the
shuffled run *also* improves held-out, the recipe is generic regularization
(arXiv:2004.05704). Hard kill.

---

## §5 — Training schedule

### 5.1 Schedule (per cell)

- **1000 gradient steps total.** Per `AUX_LOSS_AND_ARCH_DESIGN.md` §A.7
  estimate: 8 s/step × 1000 = 2.2 h on 4×H100 per cell.
- **Batch size B=4 per GPU × 4 GPUs (DDP) = effective B=16.** No gradient
  accumulation needed at this scale.
- **K=16, R=2, K_q=3, LoRA-r=32** (all defaults pinned in §2-§3).
- **Mixed precision:** bf16 forward, fp32 master for LoRA + concept-MLP +
  new-token embeddings (Adam state in fp32). Generator base + reader base
  weights frozen in bf16, no fp32 master.

### 5.2 Optimizer

Two parameter groups with different LRs:

```python
trainable_lora = [p for n, p in gen.named_parameters() if p.requires_grad and "lora" in n.lower()]
trainable_emb = [new_emb]                                       # new-token embedding rows
trainable_mlp = list(concept_mlp.parameters())

optimizer = torch.optim.AdamW(
    [
        {"params": trainable_lora,                "lr": 5e-5,  "weight_decay": 0.0},
        {"params": trainable_emb,                 "lr": 5e-3,  "weight_decay": 0.0},
        {"params": trainable_mlp,                 "lr": 5e-5,  "weight_decay": 0.0},
    ],
    betas=(0.9, 0.95),
)
```

- **lr=5e-5 for LoRA** (standard for 7B-class LoRA SFT; per `peft` defaults
  and the Qwen2.5-VL fine-tuning literature).
- **lr=5e-3 for `<|latent|>` embedding rows** (100× higher because new tokens
  are randomly init'd and need to find their natural region quickly; LIVR
  reports unfrozen embedding rows trained at a higher LR than the rest;
  exact value not in `arXiv:2512.21218`, this is our calibrated choice).
- **lr=5e-5 for concept-MLP** (matches LoRA).

LR scheduler: cosine decay to 10 % over 1000 steps, no warmup on the LR
itself (the loss-weight curriculum already provides warmup; double-warmup
slows convergence in the 1000-step regime).

### 5.3 Eval cadence

- **Every 100 steps:** held-out NLL on 30 fixed samples (q' from data-prep).
  Logged as `eval/heldout_nll`, both under φ₁ and under φ₂. Logged as a
  proxy for gates #1 and #2; do not call early decisions on these.
- **Every 100 steps:** train-time monitoring stats — `mean(||h||)`,
  `cos(h_proj, V_sem)` (raw, before 1−), per-position `||h_i||` distribution,
  inter-sample cos sim of `h_0`. Sources: round-2 steering analysis +
  `compute_visual_baseline.py` reference.
- **At step 1000 (end-of-cell):** full eval suite (§7).
- **Mid-run sanity at step 200 (end of warmup):** if `mean(||h||) > 200` or
  `cos(h_proj, V_sem) > 0.95` (identity collapse), abort and re-tune.

### 5.4 Crash resilience

Per `CLAUDE.md` "Crash resilience only when runs exceed a few minutes" — at
2.2 h per cell, we want it. Concretely:

- Save checkpoint (LoRA + new_emb + concept_mlp + optimizer state) every
  500 steps to `results/<run>/ckpt_step{n}.pt`.
- At resume: glob latest checkpoint, restore. The 1000-step run can then
  re-enter a partial run cleanly.
- All eval results are JSONL-appended (`results/<run>/eval.jsonl`).

---

## §6 — 5-cell sweep

Per `AUX_LOSS_AND_ARCH_DESIGN.md` §A.7 compute estimate: 5 cells × 2.2 h on
4×H100 fits within an overnight (~11 h compute + ~6 h overhead/eval = ~17 h).

| Cell | Variant | What it tests |
|---|---|---|
| **C1** | Full recipe: R=2, K=16, K_q=3, λ_concept=0.3, λ_norm=0.1, curriculum on | Headline. Pass thresholds in §1. |
| **C2** | R=1 (φ₁ only; drop Monet-SFT-7B from training) | Multi-reader ablation: does the multi-reader pressure carry the transfer claim? Compare gate #2. |
| **C3** | K_q=1 (single-Q per image; revert to round-1 setting) | Multi-Q ablation: does q-invariance from K_q≥3 carry the held-out claim? Compare gate #1. |
| **C4** | λ_concept=0 (drop LaViT-style aux loss) | Concept-loss ablation: does the V_sem anchor carry transfer? Compare gate #2. Round-2 already partially answers this — geometric mitigations alone did NOT fix transfer — so C4 is the "round-2 redux at training time" cell. |
| **C5** | Random-control: shuffled (image, q, y) batches; otherwise identical to C1 | arXiv:2004.05704 control: if C5 improves at all, gain is generic regularization. **Gate #3** decides on the C1 vs C5 separation. |

### 6.1 Decision logic for the cell pattern

The clean publishable outcome:

```
C1 > {C2, C3, C4}  (full recipe beats each single-component ablation)
    AND
C5 << C1           (random-control fails)
    AND
gates #1, #2, #3, #4, #5 all pass on C1
```

Any other pattern signals which component carries the gain. Concretely:

- **C2 ≈ C1 on gate #2:** multi-reader is decorative; the concept loss
  alone does the transfer work. Lower-cost recipe found; proliferated
  becomes single-reader. Still publishable.
- **C2 < C1 on gate #2:** multi-reader is load-bearing. Proceed as designed.
  Strong result.
- **C3 ≈ C1 on gate #1:** K_q is decorative; q-invariance happens by some
  other mechanism (probably ℒ_concept). Reduce K_q to 1 in proliferated.
- **C4 ≈ C1 on gate #2:** ℒ_concept is decorative; multi-reader alone
  drives transfer. Drop ℒ_concept. *(unlikely, but if true, simplifies the
  recipe.)*
- **C4 ≈ C1 on gate #1 but C4 < C1 on gate #2:** matches the round-2
  pattern — geometric anchors don't fix transfer. ℒ_concept is critical
  for transfer specifically. **This is the predicted outcome and the strongest
  publishable story.**
- **C5 ≈ C1 (all gates):** **kill the project.** This is the
  arXiv:2004.05704 outcome. Reformulate.

### 6.2 Per-cell compute (4×H100, 80 GB each, DDP)

Per `AUX_LOSS_AND_ARCH_DESIGN.md` §A.7:
- C1, C5: full R=2 K_q=3 configuration. ~71 GB/GPU at peak. ~8 s/step. ~2.2 h.
- C2: R=1, halves reader passes. ~55 GB/GPU. ~5 s/step. ~1.4 h.
- C3: K_q=1, thirds reader passes. ~45 GB/GPU. ~3 s/step. ~0.8 h.
- C4: drops ℒ_concept (no V_sem teacher pass). ~65 GB/GPU. ~7 s/step. ~2.0 h.

Total compute: ~9 h cells + ~3 h end-of-cell evals + 2 h overhead = ~14 h.
**Fits one overnight on 4×H100.**

---

## §7 — Evaluation protocol

For each of C1, C2, C3, C4, C5 at end-of-training (step 1000):

### 7.1 Held-out NLL on q' under φ₁ (POC2 protocol)

- 30 fixed q' samples from data-prep (held out at training).
- Reader = φ₁ (Qwen2.5-VL-7B-Instruct **with the trained LoRA**).
- Metric: mean NLL on the gold answer's first token.
- **Reuse:** `evaluate_held_out.py`. Modify: load LoRA-adapted reader; load
  generator-emitted h instead of POC1's saved per-sample latents.
- Gate #1: ≤ 2.5 nat.

### 7.2 Reader-transfer NLL under Monet-SFT-7B (POC3 protocol, q same as train)

- Same 30 samples (and one of their training questions, not q').
- Reader = φ₂ = Monet-SFT-7B (frozen, no LoRA).
- **This is the binding gate that round-2 failed.**
- **Reuse:** `evaluate_round2_transfer.py` and
  `evaluate_monet_sft_transfer.py`. Modify: input h is generator-emitted,
  not POC1-saved.
- Gate #2: ≤ 4.5 nat.

### 7.3 Reader-transfer on Monet-SFT-7B with q held-out

- Same 30 samples, q = q' (held-out).
- Tests *combined* failure mode: cross-reader AND cross-question. If gate #2
  passes but this fails, we have transfer-but-not-q-invariance, which is
  weaker than the headline.
- **Reuse:** same as 7.2 with q swapped to q'.

### 7.4 Steering probe (mirror round-2 §14)

- 30 samples × 8 perturbations:
  `zero_pos_{0,1,2,3,...,K-1}` (K=16 → 16 conditions, but report only
  the 4-position summary statistics matching round-2's K=4 layout: zero
  positions {0, K/4, K/2, K-1}; aggregate the rest into "mid"),
  `permute_within`, `permute_across`, `gauss_noise_0.1`, `gauss_noise_1.0`.
- Reader = φ₁.
- **Reuse:** `steering_probe.py`. Modify: load h from generator emission
  instead of POC1 saved latents; bump K=4 → K=16; adjust position sweep.
- Gate #4: zero_pos_3, permute_within, gauss_noise_1.0 each ≥ +1.5 nat.

### 7.5 Visual-grounding 5K stress test (`docs/EVAL_BENCHMARK_PLAN.md` §B)

The 5K mix: MMVP (300) + NaturalBench (1000) + BLINK 7-subtask val (~900) +
MMStar (1500) + CV-Bench-3D (~750) + POPE-adversarial (500) + VSR (500).
Total ~4950.

For round-3 (compute budget): run only on **C1 at step 1000** (not on
C2-C5; budget too tight). C2-C5 get the steering + held-out-NLL only.

- **Reuse:** new file `eval_grounding_mix.py` per
  `EVAL_BENCHMARK_PLAN.md` §E directory layout.
- Gate #5: blank-image control (C1 in EVAL_BENCHMARK_PLAN's terminology,
  not the round-3 cell) accuracy drop ≥ 5 pp on the perception-pure subset
  (MMVP + BLINK-Spatial + CV-Bench-3D ≈ 1000 samples).

### 7.6 Four control conditions from EVAL_BENCHMARK_PLAN.md §C

For C1 at step 1000, run all four:

- C1' = blank gray: gate #5 (above).
- C2' = random natural image swap.
- C3' = adversarial mismatch (paired set: MMVP/NaturalBench partner; non-paired: CLIP-retrieval mismatch).
- C4' = shuffled pixels.

Report Δ for each control. Honest-claim contract per
`EVAL_BENCHMARK_PLAN.md` §C.3:

- Δ_C1' (ours) > Δ_C1' (vanilla Qwen2.5-VL-7B-Instruct) by ≥ 5 pp on
  perception-pure subset, AND
- Δ_C3' (ours) > Δ_C3' (vanilla), AND
- Δ_C2' ≈ Δ_C1'.

Failing this contract on C1 → grounding claim is unsupported (still no kill,
since this is publishable as "method works at NLL level but doesn't translate
to control-validated grounding"). Failing this *and* gate #3 → kill.

### 7.7 Eval compute estimate

- Held-out NLL (30 samples × 2 readers × 5 cells) = 300 forwards. ~5 min.
- Steering (30 samples × 8 perturb × 5 cells) = 1200 forwards. ~30 min.
- Cross-q transfer (30 × 1 × 5) = 150 forwards. ~3 min.
- 5K stress + 4 controls (5 runs × 5K samples × 1 cell) = 25K forwards.
  At 0.4 s each on A6000 = ~3 h. (Run on a separate GPU after training
  finishes.)
- **Total eval: ~4 h after the 1000-step train.** Fits within the overnight
  budget.

---

## §8 — Implementation plan (concrete files to write)

All in `experiments/reader-grounded-latent-poc/`. Single-file scripts per
`CLAUDE.md` conventions. Inline-copy small helpers across files (no shared
module).

### 8.1 New scripts to write

| File | Purpose | Reuses |
|---|---|---|
| `prepare_data_round3.py` | Build the 10K mixed loader: filter GQA-balanced to ≥3 Q/img (6500 samples), pull CLEVR cauldron (2500), TallyQA (1000); reserve 1 q'/image as held-out; emit per-image JSONL with `questions[]` and `held_out_q`. | `prepare_data.py`, `prepare_data_nq.py` (Q-filtering pattern). |
| `model_round3.py` | Single-file model construction (§2.8). Loads Qwen2.5-VL-7B + LoRA + new-token embeddings + concept-MLP + Monet-SFT-7B. Exposes `forward_generator(images) -> h`, `forward_reader(reader, h, q, a) -> nll`, `forward_concept(h, image) -> concept_loss`. Functions, not classes (transparency). | `tune.py`'s `build_prompt_with_k_pads` and `forward_with_latents` patterns; copy verbatim with K=16. |
| `train_round3.py` | The 5-cell sweep driver. Top-of-file constants for cell selection (`CELL = "C1"` etc); curriculum schedule inline; per-100-step eval inline. JSONL append for losses, eval, and config.json at start. Single `for step in range(1000)` loop — no abstractions. | `tune.py` (whole skeleton). `tune_mitigationB.py` (norm reg). `tune_mitigationA.py` (low-rank). `tune_multiq.py` (multi-Q). Stitch them into one. |
| `evaluate_round3.py` | End-of-cell evaluation: loads last checkpoint, runs §7.1 + §7.2 + §7.3 + §7.4 in one go. Outputs `results/<run>/eval_round3.jsonl`. | `evaluate_held_out.py`, `evaluate_round2_transfer.py`, `evaluate_monet_sft_transfer.py`, `steering_probe.py`. |
| `eval_grounding_mix.py` | The 5K stress test runner per `EVAL_BENCHMARK_PLAN.md` §B. Greedy gen at 64 tok, regex-extract MC letter / Yes-No / numeric. Per-source + per-skill aggregation. | New file; cross-reference `EVAL_BENCHMARK_PLAN.md` §E directory layout for output schema. |
| `eval_controls_4.py` | Four control conditions per `EVAL_BENCHMARK_PLAN.md` §C: blank gray, random natural, adversarial mismatch, shuffled pixels. Pre-builds the controls once, runs greedy gen across all four. | New; piggyback on `eval_grounding_mix.py`'s loader. |
| `analyze_round3.py` | Aggregate the 5-cell × multiple-eval results into `results/ANALYSIS_round3.md` with the gates table (§1) filled in per cell. | `analyze.py` (aggregation pattern). |

### 8.2 Existing scripts that stay untouched (reused at eval)

- `compute_visual_baseline.py` — reused for V_sem cache verification (the
  natural-token statistics for `concept_mlp` target reference).
- `tune.py` (POC1 driver) — kept as the historical baseline; round-3 does
  not modify it.
- `evaluate_held_out.py`, `evaluate_transfer.py`, `evaluate_monet_sft_transfer.py`,
  `evaluate_round2_transfer.py`, `steering_probe.py` — kept; called/copied
  by `evaluate_round3.py`.

### 8.3 V_sem caching (optional optimization)

Per `AUX_LOSS_AND_ARCH_DESIGN.md` §A.2.4: pre-compute V_sem for all 10K
training images once (~1 h on A6000), cache to `data/v_sem_cache.pt`. Each
training step then just indexes the cache instead of re-running the vision
tower. Saves ~30 % wall-clock. **Recommended; small extra effort.**
Cache size: 10K images × ~300 tokens × 3584 floats × 2 bytes (bf16) ≈ 21 GB.
Fits on the experiment's `data/` partition.

If precomputing is not done in time, fall back to in-step vision-tower
forward (the generator's vision branch is frozen, so the forward is
gradient-free and fast).

### 8.4 Repository layout after round-3

```
experiments/reader-grounded-latent-poc/
  data/
    poc1_samples.jsonl                 (existing)
    poc2_pairs.jsonl                   (existing)
    round3_train.jsonl                 (NEW — ~10K rows)
    round3_held_out.jsonl              (NEW — 1 q'/image)
    v_sem_cache.pt                     (NEW — optional)
    grounding_mix_v1.jsonl             (NEW — 5K stress)
    controls_blank/, controls_random_nat/, controls_adv/, controls_shuffled/
  prepare_data_round3.py               (NEW)
  model_round3.py                      (NEW)
  train_round3.py                      (NEW)
  evaluate_round3.py                   (NEW)
  eval_grounding_mix.py                (NEW)
  eval_controls_4.py                   (NEW)
  analyze_round3.py                    (NEW)
  results/<timestamp>_round3_C{1,2,3,4,5}/    (NEW per cell)
    config.json
    losses.jsonl
    eval.jsonl                         (per-100-step held-out NLL)
    ckpt_step{500,1000}.pt
    eval_round3.jsonl                  (end-of-cell eval)
    grounding_mix.jsonl                (C1 only)
    controls_4.jsonl                   (C1 only)
  results/ANALYSIS_round3.md           (NEW — aggregated)
  docs/ROUND3_POC_DESIGN.md            (this file)
```

---

## §9 — Risks and mitigations

### Risk 1: ℒ_concept collapses to identity (h becomes a copy of teacher V_sem)

**Mechanism:** if `concept_mlp` has unconstrained capacity and λ_concept
dominates ℒ_NLL early, the optimizer's cheapest path is `h ≈ V_sem`,
making h a soft prompt of natural visual tokens. Method degenerates to
"use V_sem directly" — no novelty.

**Diagnostic:** monitor `cos(h_proj, V_sem)` per training step.

**Trigger:** `cos(h_proj, V_sem) > 0.95` for 100+ consecutive steps after
step 200.

**Mitigation:** the bottleneck MLP (D → D/2 → D) raises the friction; if
that's not enough, add a per-step penalty `0.05 · max(0, cos(h_proj, V_sem) − 0.85)`
to actively cap saturation. Round-3 doesn't include this penalty by default;
add only if Risk 1 fires in the first cell.

### Risk 2: R=2 doesn't generalize to a held-out reader (transfer to a NEW reader fails)

**Mechanism:** 2-reader pressure may overfit to the (φ₁, φ₂) pair —
producing an h that is in the *intersection* of those two readable cones
but not in the cone of φ₃ that didn't see training pressure.

**Diagnostic:** at end-of-cell C1, evaluate transfer to **VL-Rethinker-7B**
(`TIGER-AI-Lab/VL-Rethinker-7B`). This reader was NOT in training. Stretch:
include in the eval row.

**Trigger:** transfer drop on VL-Rethinker > 1 nat above the (φ₁, φ₂) drop.

**Mitigation in round-3:** none beyond the eval check. If R=2 doesn't
generalize, the proliferated project goes R=3 with VL-Rethinker as a *training*
reader (~+50 % VRAM, see `AUX_LOSS_AND_ARCH_DESIGN.md` §A.2.4).

### Risk 3: random-control passes (gain on shuffled mix)

**Mechanism:** the recipe is generic regularization. The shuffled mix should
not produce any visual content yet improves held-out — same outcome as
arXiv:2004.05704's analysis of 6 prior VQA grounding methods.

**Diagnostic:** gate #3.

**Trigger:** C5 held-out NLL improvement ≥ 50 % of C1's improvement.

**Mitigation:** **scrap and reformulate.** This is the hard kill.
Specifically, drop the entire reader-grounded latent framing — Li et al.'s
text-imagination alternative (CapImagine, `arXiv:2602.22766`) becomes the
better project. The proliferated training shifts to the text-imagination
baseline as the contribution.

### Risk 4: K=16 hits saturation (round-1 K=16 inter-sample cos sim was 0.281)

**Mechanism:** at K=16, the K capacity becomes diffuse / non-orthogonal —
some positions become "spillover" of others. This was observed in round-1
POC4 (K=16: inter-sample cos 0.281 vs K=4: 0.051). Under round-3's
manifold-anchor + multi-reader pressure, this might reverse direction
(diffuse becomes systematic, like Monet's 0.93) — which would be *good*
for transfer. Or it might dilute — which would hurt steering.

**Diagnostic:** per-100-step `inter_sample_cos(h_0)` and `intra_block_cos(h)`.

**Trigger:** if either exceeds round-2's pathological regime —
`inter_sample_cos(h_0) > 0.7 AND steering Δnll < 1.0` — drop K to 8.

**Mitigation in round-3:** off-budget rerun of C1 at K=8 if Risk 4 fires.
Adds ~2 h compute. K=4 is a fallback if K=8 also saturates.

### Risk 5: VRAM blowout on R=2

Per `AUX_LOSS_AND_ARCH_DESIGN.md` §A.7, peak ~71 GB / GPU at B=4 K=16 R=2
K_q=3. Tight on 80 GB H100.

**Mitigation:** enable gradient checkpointing on the generator backbone.
~2× wall-clock per cell, halves activation memory. Trigger if any cell OOMs.

### Risk 6: Monet-SFT-7B download / setup delay

Currently on disk: only `Monet-7B` (post-RL, in `monet-latent-probe/data/`).
**Round-3 needs `Monet-SFT-7B`** (`NOVAglow646/Monet-SFT-7B`) per
`REPORT.md` §9.

**Mitigation:** download as Day-0 prerequisite. ~16 GB. If not available
in time, fall back to the post-RL Monet-7B (less clean, but does not change
sign of the transfer result; magnitude differs by ~10 %).

### Risk 7: LoRA-shared-with-reader contaminates the "frozen reader" claim

Per §2.6 caveat: φ₁ uses the same LoRA as the generator. Strictly, the
reader is not frozen.

**Mitigation:** report this honestly. The proliferated project will run a
clean comparison cell (frozen-base reader, no shared LoRA). For round-3,
this cell is part of the proliferated planning, not the round-3 budget.

---

## §10 — Time estimate and prerequisites

### 10.1 Engineering — ~3 working days

| Task | Wall-clock | Notes |
|---|---|---|
| `prepare_data_round3.py` + V_sem cache + dataset download | 0.5 day | Streaming GQA + cauldron; robust to broken HF loaders per `TRAINING_DATA_PLAN.md` §5.1. |
| `model_round3.py` + smoke test (1 step works, no NaN, all gradients nonzero) | 1 day | Including LoRA injection, new-token embedding, attention masking, reader-2 load. |
| `train_round3.py` + 200-step canary run (C1 only) | 1 day | Verify curriculum kicks in correctly; verify multi-reader gradient propagates. |
| `evaluate_round3.py` + `eval_grounding_mix.py` + `eval_controls_4.py` | 0.5 day | Mostly stitching from existing eval scripts. |

### 10.2 Run — ~24 h on 4×H100

Per §6.2: ~14 h training + ~4 h eval + ~6 h overhead/restarts = ~24 h.

### 10.3 Prerequisites

- **4×H100 access.** Currently no recurring access (per HANDOFF). Round-3
  needs an explicit reservation. Estimate: 1 day for queueing + 1 day run.
- **HF datasets verified accessible:**
  - `lmms-lab/GQA` cfg `train_balanced_instructions` ✓ (probed in
    `TRAINING_DATA_PLAN.md` §8 to 943K rows, gated=False).
  - `HuggingFaceM4/the_cauldron` cfg `clevr` ✓ (probed: 70K rows).
  - `HuggingFaceM4/the_cauldron` cfg `tallyqa` ✓ (probed: 98.7K rows).
- **HF model checkpoints:**
  - `Qwen/Qwen2.5-VL-7B-Instruct` ✓ on disk
    (`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct`, ~14 GB).
  - `NOVAglow646/Monet-SFT-7B` — **not on disk**, ~16 GB download.
    **Day-0 prereq.**
  - `TIGER-AI-Lab/VL-Rethinker-7B` (R=3 stretch / Risk 2 eval-time check) —
    not on disk, ~14 GB. **Stretch.**
- **5K stress test data:** `MMVP/MMVP`, `BaiqiL/NaturalBench`,
  `BLINK-Benchmark/BLINK`, `Lin-Chen/MMStar`, `nyu-visionx/CV-Bench`,
  `lmms-lab/POPE`, `cambridgeltl/vsr_zeroshot`. All Apache or research-
  permissive per `EVAL_BENCHMARK_PLAN.md` §A.1.
- **Disk:** ~250 GB free for `data/` (training mix + V_sem cache + control
  caches + 5K stress images). Currently the experiment's `data/` is small;
  this is a clean addition.

### 10.4 Hard go/no-go for round-3 launch

Round-3 launches only when:

1. Monet-SFT-7B is on disk and a no-op forward succeeds (verifiable in
   < 1 h on A6000).
2. The 200-step canary run on C1 produces non-NaN losses, all five
   loss components are nonzero gradient-wise, and the curriculum schedule
   visibly drives w_NLL from 0.1 to 1.0 between steps 0 and 200.
3. 4×H100 is reserved for at least 24 contiguous hours.

If any prerequisite fails by 2026-05-09, defer round-3 to the next compute
window and use the time to run the random-control ablation
(`evaluate_random_control_heldout.py` already in repo) on round-2's
mit-B latents — that fills in a missing data point for round-2 itself
and is a cheap A6000-only run.

---

## §11 — References (arXiv IDs verified in source docs)

All IDs below are carried forward from the source docs (`AUX_LOSS_AND_ARCH_DESIGN.md`
§E "Citations verified this session", `EVAL_BENCHMARK_PLAN.md` "References
(verified)", `VARIANT_B_GRPO_DESIGN.md` "Key references"). Re-verification
is out of scope for this design doc; the round-3 implementation will pin
these in `config.json`.

- **Architecture / latent-emitter:**
  `arXiv:2512.21218` (LIVR — K=16 default, special-token recipe, Stage-1 attention masking),
  `arXiv:2412.06769` (Coconut — `<bot>...<eot>` markers, NLL on answer tokens only),
  `arXiv:2511.21395` (Monet — VLPO, σ=10, three-stage SFT — sibling experiment),
  `arXiv:2509.24251` (LVR — MSE to ROI patch embeddings, vision-grounded baseline),
  `arXiv:2301.12597` (BLIP-2 / Q-Former — alternative architecture for proliferated, not round-3).
- **Auxiliary loss / grounding:**
  `arXiv:2601.10129` (LaViT — λ=0.3, ℒ_concept = 1 − cos(MLP(h_z), V_sem), curriculum γ(t), Top-K=8 sparsified attention),
  `arXiv:2502.13923` (Qwen2.5-VL technical report — the base architecture).
- **Critique / steering / control:**
  `arXiv:2602.22766` (Li et al. — Input-Latent / Latent-Answer Disconnects, CapImagine),
  `arXiv:2512.21711` (Do Latent Tokens Think? — steering protocol; round-2 inverted their finding),
  `arXiv:2004.05704` (Visual Grounding Methods Working for Wrong Reasons — random-control mandate),
  `arXiv:2406.16320` (NOTICE — activation-patching for VLMs, deferred to proliferated D.2).
- **Reader siblings (R=2 / R=3 stretch):**
  `arXiv:2511.21395` (Monet-SFT-7B base = Qwen2.5-VL-7B; sibling reader for R=2),
  `arXiv:2504.08837` (VL-Rethinker — RL fine-tune of Qwen2.5-VL-7B-Instruct; R=3 stretch),
  `arXiv:2503.15558` (Cosmos-Reason1-7B — alternative R=3 candidate, physical-AI tuned).
- **Eval suite (5K stress + controls per `EVAL_BENCHMARK_PLAN.md`):**
  `arXiv:2401.06209` (MMVP), `arXiv:2410.14669` (NaturalBench),
  `arXiv:2404.12390` (BLINK), `arXiv:2403.20330` (MMStar),
  `arXiv:2406.16860` (CV-Bench, in Cambrian-1 paper),
  `arXiv:2305.10355` (POPE), `arXiv:2205.00363` (VSR).
- **Round-2 prior context (`REPORT.md`, `JOURNAL.md`):**
  `arXiv:2402.09063` (soft-prompt off-manifold attacks),
  `arXiv:2104.08691` (soft-prompt tuning baseline),
  `arXiv:2504.02144` (interpretability-vs-task trade-off in soft prompts),
  `arXiv:2604.02073` (PLUME — independently confirms POC4 drift).

---

## §12 — TL;DR (for quick-look)

- **Mission:** validate combined recipe (LIVR same-VLM emitter + LaViT cosine
  to V_sem + multi-reader R=2 NLL + multi-Q K_q=3 + norm-reg λ=0.1) before
  proliferated training. **Hard gate, not scaling.**
- **5 pass thresholds:** held-out NLL ≤ 2.5 (φ₁); transfer NLL ≤ 4.5
  (Monet-SFT); random-control gain ≤ 50 % of real gain; steering retains
  ≥ +1.5 nat on 3 perturbation modes; blank-image accuracy drop ≥ 5 pp on
  perception-pure subset. **All five must pass.**
- **Architecture:** Qwen2.5-VL-7B with K=16 `<|latent|>` tokens, LoRA r=32
  (q/k/v/o/gate/up/down × 28 layers), Stage-1 attention masking (answer
  tokens cannot attend to image), generator and reader-1 share weights+LoRA;
  reader-2 = Monet-SFT-7B frozen.
- **Loss:** `ℒ_total = w_NLL(t) · ℒ_NLL_multi + 0.3 · ℒ_concept + λ_norm(t) · ℒ_norm`.
  V_sem = φ₁'s post-merger visual tokens, mean-cosine through bottleneck MLP
  D → D/2 → D. Norm target μ=57.86. Curriculum: 200-step warmup
  (w_NLL: 0.1→1.0, λ_norm: 0→0.1), 600-step main, 200-step λ_concept anneal.
- **Data:** 10K mix (65 % GQA-balanced ≥3Q/img + 25 % CLEVR + 10 % TallyQA).
  K_q=3 questions per image per step (q-invariant generator, no q to
  generator's input). 1 Q/img reserved as held-out q'.
- **Compute:** 1000 steps × 5 cells (C1 full + C2 R=1 + C3 K_q=1 + C4
  λ_concept=0 + C5 random-control) × ~2.2 h/cell = ~14 h training on 4×H100;
  +~4 h eval. Fits one overnight.
- **Optimizer:** AdamW lr=5e-5 (LoRA + concept-MLP), lr=5e-3 (new-token
  embedding rows), bf16 forward, fp32 master.
- **Decision:** if all 5 gates pass on C1 AND C1 > {C2, C3, C4} AND C5 << C1,
  advance to proliferated training (`TRAINING_DATA_PLAN.md` Mix 3.2 → 3.3).
  Otherwise specific pivots per §1.
- **Engineering:** ~3 days (model + loss + data loaders + eval). 7 new files
  (`prepare_data_round3.py`, `model_round3.py`, `train_round3.py`,
  `evaluate_round3.py`, `eval_grounding_mix.py`, `eval_controls_4.py`,
  `analyze_round3.py`).
- **Prereqs:** 4×H100 reservation; Monet-SFT-7B download (~16 GB);
  V_sem cache (~21 GB optional). Fallback if compute unavailable: run
  random-control on round-2 mit-B latents (A6000-only) to fill that
  missing data point.

The round-3 outcome will be one of three: (i) all gates pass → proliferated
training launches; (ii) gate #2 fails → architectural pivot to LIVR-style
single-model bottleneck or R=3 multi-reader; (iii) gate #3 fails → kill
the reader-grounded latent project and pivot to text-imagination
(CapImagine, `arXiv:2602.22766`) as the contribution.
