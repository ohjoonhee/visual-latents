# Interleaved Latent Reasoning — Literature Recon

Scope: prior art for redesigning `visual-latents` from a single parallel
K-token emission (current method, `docs/METHODS.md`) into an
**interleaved autoregressive text-latent trace** ("text…<|latent|>×k…
text…<|latent|>×k…text"), trained against a frozen reader on answer NLL.

## 1. Coconut — the foundational autoregressive continuous-thought work

Hao et al., **arXiv:2412.06769** (HTML verified this session).

**Mode switching.** Two new special tokens `<bot>` (begin-of-thought)
and `<eot>` (end-of-thought) wrap the latent span. Outside the span,
text decoding is normal; inside, the model is in latent mode.

**Hidden-state-as-next-input.** Verbatim: *"Instead of mapping between
hidden states and language tokens using the language model head and
embedding layer, Coconut directly feeds the last hidden state (a
continuous thought) as the input embedding for the next token."* The
input sequence inside the latent span becomes
`[e(x_1), …, e(x_i), h_i, h_{i+1}, …, h_{t-1}]`.

**Curriculum.** Multi-stage. At stage k, the **first k reasoning steps
of the ground-truth CoT are replaced with k×c continuous thoughts** (c=2
GSM8K, c=1 ProntoQA/ProsQA). Stage count = max reasoning depth (N=6 for
ProsQA). **Optimizer state is reset between stages.**

**Gradient strategy.** For n latent thoughts in the current stage,
**n+1 separate forward passes** — one per thought (each pass produces
the next continuous thought from the recurrence), then a final pass
for remaining text. Loss = standard NLL **masked on questions and on
the latent thoughts themselves**, so loss is computed only on the
post-latent text. Latents have **no direct supervision**; they are
trained purely by gradient flowing back from the trailing text loss
through the recurrence inside one differentiable model.

**Eval.** GSM8K, ProntoQA, ProsQA.

**Project's prior critique.** "Do Latent Tokens Think?"
(arXiv:2512.21711, `LITERATURE_RECON.md` §2.1) reports COCONUT latents
*"function as uninterpretable placeholders … show minimal sensitivity
to steering and lack reasoning-critical information."* The project's
own POC §14 found the *opposite* steering pattern for visual reader-
grounded latents — modest positive prior that the visual setting may
not inherit Coconut's text-latent failure mode.

## 2. Visual extensions of Coconut-style reasoning

### 2.1 Mirage (arXiv:2506.17218) — closest published interleaved-VLM sibling

*"Whenever the model chooses to 'think visually', it recasts its hidden
states as next tokens, thereby continuing a multimodal trajectory
without generating pixel-level images."* A special token signals visual-
thinking entry. Latents interleaved with text in one autoregressive
trajectory. **Three stages:**
- *Stage 1 (joint distillation):* helper images are passed through the
  VLM's vision tower → patch features; the model is fine-tuned to
  output those features at latent positions.
- *Stage 2 (text-only relaxation):* drop the cosine/feature loss; keep
  text CE only. Latents drift to whatever serves the text loss.
- *Stage 3 (RL refinement):* GRPO on the trajectory.

Single unified model; **no separate frozen reader**. Eval: VSP, BLINK-
Jigsaw, SAT, COMT. Mirage was tested by Li et al. (arXiv:2602.22766)
and found to suffer Latent-Answer Disconnect when the image is still in
context (`LITERATURE_RECON.md` §1.2).

### 2.2 SkiLa — Sketch-in-Latents (arXiv:2512.16584)

*"Dynamically alternates between"* (i) textual thinking mode (next-token
prediction on discrete text) and (ii) visual sketching mode (continuous
latent sketch tokens). Supervision: a *"latent visual semantics
reconstruction mechanism"* — sketch encoder embeddings serve as
reconstruction targets. Single model; self-distillation; no separate
reader.

### 2.3 ILVR — Interleaved Latent Visual Reasoning (arXiv:2512.05665)

*"Interleaves textual generation with latent visual representations
that act as specific, evolving cues for subsequent reasoning."*
Supervision: *"a momentum teacher model selectively distills relevant
features from ground-truth intermediate images into sparse supervision
targets"* (`LITERATURE_MITIGATIONS.md` §3.1). Single model; momentum
self-distillation.

### 2.4 Latent Sketchpad (arXiv:2510.24514)

*"Allows the model to interleave textual reasoning with the generation
of visual latents."* Special tokens `<start_of_image>` /
`<end_of_image>`. **Training signal: latent-level regression loss
against the MLLM's own pretrained vision-encoder features** — fully
self-supervised. The Sketch Decoder is frozen, used only at inference
for human visualization, *not* as a training-supervision reader.

### 2.5 Heima (arXiv:2501.19201)

Replaces stages of an explicit textual CoT with **a single thinking-
stage token per stage** (`<Thinking_of_Summary>`, `<Thinking_of_Caption>`,
`<Thinking_of_Reasoning>`). Encoder = MLLM fine-tuned via NLL on the
multi-stage trace. Decoder = a separate post-hoc LLM trained to
reconstruct textual CoT for interpretability — *not* the supervision
source. **Curriculum ("progressive encoding strategy"):** gradually
increase the number of encoded stages, mirroring Coconut.

### 2.6 MCOUT (arXiv:2508.12587)

Direct Coconut port to VLMs. *"MCOUT-Base bypasses [the projector],
directly using the last hidden state as the thought embedding."*
N_t ∈ {5, 10} thought iterations appended to input sequence. Combined
loss with auxiliary thought weight μ=0.3. Eval: VQAv2, MMMU, ScienceQA,
MMStar.

### 2.7 Mull-Tokens (arXiv:2512.10941)

20 modality-agnostic latent tokens. Two-stage: warm-up with multimodal
CoT supervision, then *"free-form optimizes these Mull-Tokens to
achieve the final correct answer"* with **only final-answer
supervision**. *"The model can effectively decide when to also avoid
using text reasoning depending on the task."* Single model.

### 2.8 LIVR (arXiv:2512.21218) — parallel, NOT interleaved

`LITERATURE_MITIGATIONS.md` §3.4 calls this the closest sibling to the
*current* parallel method. WebFetch confirms: *single forward pass*,
attention-mask bottleneck within one model — included here only to
disambiguate from the interleaved siblings above.

## 3. Reader-anchor patterns

**No surveyed interleaved-VLM-latent paper uses a separate frozen
reader as the training signal.** Every one (Mirage, SkiLa, ILVR,
Latent Sketchpad, Heima, MCOUT, Mull-Tokens) uses single-model self-
supervision: distillation against the model's own vision encoder,
momentum-teacher self-distillation, sketch-encoder targets, or
post-latent-text NLL. The closest published precedent for a frozen-
reader signal is the soft-prompt-tuning lineage (Lester et al.,
**arXiv:2104.08691**) — but that is *parallel*, *single-prompt*, *non-
image-conditioned*. The project's own `LITERATURE_RECON.md` §4 reaches
the same conclusion for the parallel version: *"No paper found that
matches the exact recipe."* The novelty extends to the interleaved
variant — but with the same caveat: no recipe is *proven* to work in
this regime.

## 4. Gradient-flow strategies for mixed text+latent traces

### 4.1 Teacher-forcing on bootstrapped/pseudo-labeled traces (Mirage Stage 1, Heima, Coconut stage 0)

Pre-generate the interleaved trace from a teacher source. At training
time, **teacher-force the discrete text tokens** — they are inputs,
not sampled outputs. Latent positions and any trainable embedding rows
receive gradient. **Sidesteps the discrete-gradient problem entirely.**
Pro: simple, multi-precedent. Con: requires high-quality teacher
traces; the interleaving pattern is dictated, not learned.

### 4.2 Straight-through Gumbel-softmax

**Not used in any of the surveyed VLM-latent papers** (Coconut,
Mirage, SkiLa, ILVR, Latent Sketchpad, Heima, MCOUT, Mull-Tokens, LIVR).
A negative signal: even researchers who needed mixed-discrete-continuous
training preferred §4.1 or §4.3. The 32k+ vocabulary makes ST-Gumbel
high-variance and engineering-heavy. Avoid.

### 4.3 REINFORCE / RLHF / RLVR (Mirage Stage 3, Variant B in this project)

Sample full traces, score, update with policy gradient. **Latent CoT
for Visual Reasoning (arXiv:2510.23925)** specifically advocates
*"diversity-seeking reinforcement learning algorithms"* with sparse
token-level rewards to *"encourage diverse, high-likelihood latent CoT
… avoiding reward hacking."* Pro: handles any trace structure, natural
fit for "reader scores a sample." Con: high variance, reward hacking
documented, low sample efficiency, typically a *refinement* stage on
top of supervised warm-up rather than a from-scratch trainer (Mirage
puts it Stage 3, after two SFT stages).

### 4.4 Frozen text policy — only train latent rows

Keep the base LLM frozen (or LoRA-frozen on text modules); train **only
the latent emission mechanism** (special embedding rows, optional
adapter, optional gating head). Discrete text tokens come from the
frozen base LLM, so no discrete-gradient problem. **No surveyed paper
uses this exactly.** Pro: no discrete-gradient burden. Con: frozen base
LLM has no incentive to route around its own language priors
(`LITERATURE_RECON.md` §5: shortcut basin universally reachable);
without an emission-control signal, the base LLM may not naturally emit
`<|latent|>` at the right moments.

### 4.5 Curriculum staging (Coconut, Heima progressive encoding)

Replace only the first k text reasoning steps with latents per stage;
trailing text is teacher-forced. Optimizer reset between stages. Each
stage is then a §4.1 teacher-forcing problem with a slightly longer
latent prefix. Pro: documented to work; incremental. Con: assumes a
linear text-then-latent decomposition; less clean for true
bidirectional interleaving with multiple alternations per trace.

## 5. Design-space implications for visual-latents

### 5.1 Validated by prior art

- **Special-token mode switch** (`<bot>`/`<eot>` Coconut, Mirage's
  visual-thinking trigger, Latent Sketchpad's `<start_of_image>`).
  The project's existing `<|latent_start|>`/`<|latent_end|>` tokens
  follow precedent.
- **Hidden-state-as-next-input recurrence** within latent spans
  (Coconut, MCOUT, Latent Sketchpad).
- **Three-stage Mirage curriculum** (distill → relax → RL) is
  reproducible.
- **Teacher-forcing of discrete text (§4.1)** is the only gradient-
  flow strategy with multiple successful instantiations in this regime.
  Safe default.
- **Multi-question shared-latent for q-invariance** — already used
  by current parallel method (POC §10–§13); transfers directly to
  interleaved at the per-trace level.

### 5.2 Open / under-explored

- **Separate frozen reader as training signal for interleaved trace.**
  No precedent. The novelty axis.
- **"Decide when to emit" mechanism.** Mirage and Mull-Tokens claim
  the property but neither describes the decision mechanism explicitly
  in available abstracts. Open to engineering choice (gating head vs
  learned-from-trace).
- **Reader-grounded supervision + Coconut-style curriculum** (each
  stage scored by frozen reader rather than NTP). No precedent.
- **Frozen-text-policy gradient flow (§4.4) for interleaved.** No
  paper attempts. Publishable design choice in itself.

### 5.3 Contraindicated by prior art

- **Single-stage SFT against reader-NLL alone, no auxiliary grounding.**
  Project's `REPORT.md` §17.1 already shows that within-reader NLL
  gains are *mostly* generic regularization (random target reproduces
  ~70 % of the natural-target gain). The interleaved redesign will
  inherit this unless an explicit grounding term (LaViT's `ℒ_concept`,
  Mirage's Stage-1 distillation, LVR's MSE-to-ROI) is in the loss.
- **Skipping a steering / OOD probe in evaluation.** Both
  arXiv:2512.21711 and POC §14 establish steering as the load-bearing
  diagnostic for placeholder-vs-real-work. Non-negotiable.
- **Trusting cross-reader transfer with a single training reader.**
  `REPORT.md` §11: *"no purely geometric mitigation will fix reader
  transfer."* Multi-reader NLL during training (round-3 fix) is needed
  regardless of trace structure.
- **Pure ST-Gumbel through discrete text vocab.** Zero precedent; §4.2.

### 5.4 Synthesis recipe for the interleaved redesign

Lowest-risk configuration synthesized from above:

1. **Trace structure.** Pre-generated synthetic interleaved traces
   from a teacher (à la Mirage Stage 1 / Heima training data).
2. **Mode switching.** Existing `<|latent_start|>` / `<|latent_end|>`
   tokens; Coconut-style hidden-state recurrence inside each block.
3. **Gradient flow.** Teacher-force discrete text (§4.1) — highest-
   confidence design choice in this document.
4. **Curriculum.** Either Coconut stage replacement (k×c latents per
   step) with optimizer reset, or Mirage three-stage (distill → relax
   → RL).
5. **Reader supervision.** Continue with frozen-reader NLL
   (the novelty axis). To control the predicted regularization-only
   failure mode, **keep LaViT-style `ℒ_concept`** from
   `docs/inherited/AUX_LOSS_AND_ARCH_DESIGN.md` §A.1, applied within
   each latent block.
6. **Multi-reader.** R ≥ 2 frozen readers per
   `docs/inherited/AUX_LOSS_AND_ARCH_DESIGN.md` §A.2.
7. **Evaluation.** Steering probe (arXiv:2512.21711 / POC §14) +
   reader-transfer to Monet-SFT-7B + random-(image, q) control
   (arXiv:2004.05704). All three mandatory.

The interleaved redesign is structurally credible — every component
except the frozen-reader supervision has a published existence proof.
The empty square (interleaved + frozen-reader-supervised) is the
publishable novelty, but only if the standard pathologies (shortcut,
off-manifold, single-reader transfer failure) are controlled by the
auxiliary stack already designed for the parallel version.

---

**References (verified this session unless flagged inherited):**

- Coconut — **arXiv:2412.06769** (HTML verified).
- Mirage — **arXiv:2506.17218** (HTML verified).
- SkiLa — **arXiv:2512.16584** (search-snippet verified).
- ILVR — **arXiv:2512.05665** (inherited recon §3.1).
- Latent Sketchpad — **arXiv:2510.24514** (HTML verified).
- Heima — **arXiv:2501.19201** (HTML verified).
- MCOUT — **arXiv:2508.12587** (HTML verified).
- Mull-Tokens — **arXiv:2512.10941** (project page verified).
- LIVR — **arXiv:2512.21218** (HTML verified; parallel — for contrast).
- LaViT — **arXiv:2601.10129** (inherited recon §3.5).
- LVR — **arXiv:2509.24251** (inherited recon §1.3).
- Latent CoT for Visual Reasoning — **arXiv:2510.23925** (inherited §3.6).
- "Do Latent Tokens Think?" — **arXiv:2512.21711** (inherited §2.1).
- Li et al. (Mirage Latent-Answer Disconnect) — **arXiv:2602.22766**
  (inherited §1.2).
- Lester et al., soft-prompt tuning — **arXiv:2104.08691** (inherited §4).
- Selvaraju et al. — **arXiv:2004.05704** (inherited §5).
- CrystaL — **arXiv:2602.20980** (search verified; single-stage two-
  path attention/distribution alignment; relevant to grounding, not
  interleaving — not detailed for that reason).
