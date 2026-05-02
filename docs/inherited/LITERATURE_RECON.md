# Literature Recon — Reader-Grounded Latent Visual Reasoning

Scope: pre-validation literature for the proposed method (train an MLLM to emit
continuous latent visual embeddings `h_{1:K}` that, when injected into a *frozen
reader* MLLM's visual-token slots — with no other image given to the reader —
let the reader recover `y*`). Two variants under consideration: (A)
differentiable supervision `-log φ(y* | q, h_{1:K})`, (B) GRPO advantages ×
reader log-likelihood.

The three concerns to pressure-test against the literature:
1. **Shortcut / answer-encoding** — do `h` learn answers, not visual content?
2. **Off-manifold latents** — do reader-NLL-optimal latents resemble natural
   visual tokens?
3. **Latent-Answer Disconnect** — does the no-image-on-reader constraint
   structurally avoid Li et al.'s finding?

---

## 1. Anchor papers

### 1.1 Monet (arXiv:2511.21395, CVPR 2026) — base architecture, VLPO

**Pitch.** Monet trains MLLMs to "reason directly within the latent visual space
by generating continuous embeddings that function as intermediate visual
thoughts" (arXiv:2511.21395, abstract). Latents are pitched as
**"intermediate visual thoughts beyond textual descriptions and image
embeddings"** — i.e., the claim is that they encode visual content, not answer
information.

**VLPO objective (Gaussian assumption made explicit).** From the HTML render of
the paper, the policy density over the latent embedding at step `t` is treated
as Gaussian centered on the policy's emitted embedding:

> "π_θ(h^{i,t}_old | Q, I, o_{i,<t}) = exp(-1/(2σ²) ||h^{i,t}_old − h^{i,t}_θ||² − const)"
> (arXiv:2511.21395)

so the importance ratio reduces to

> "r_{i,t}(θ) = exp(-1/(2σ²) ||h^{i,t}_old − h^{i,t}_θ||²)"

with **σ = 10.0** in the hyperparameters table. This is essential context: VLPO
is just a Gaussian-likelihood reparameterization of policy gradient on
continuous embeddings, with old rollouts treated as samples from a Gaussian
centered on the current policy's mean.

**Three-stage SFT pipeline.**
- *Stage 1.* Warm-up next-token prediction on Monet-SFT-125K (image–text
  interleaved CoTs).
- *Stage 2.* Same CoTs but with auxiliary images visible to the *student* in
  the slots that will eventually become latents; loss is
  `L_NTP + 2.0 · L_align-obs` — aligns hidden states of observation tokens
  under the auxiliary-image condition. Stage 2 yields **"high-quality target
  latent embeddings"** that act as fixed targets in Stage 3.
- *Stage 3.* Auxiliary images are removed; loss is
  `L_NTP + 2.0 · L_align-latent`, aligning the model's generated latent
  embeddings to the targets from Stage 2.

**What this means for the user's design.** Monet's training signal is
distillation onto a *teacher-defined latent target* — visually grounded by
construction (Stage 2 sees the image). The user's proposal replaces this with
reader-NLL on `y*` and inherits no such grounding. That's exactly the gap
worth investigating — but the prior on "the latents will encode visual
content" comes from Monet's *teacher targets*, not from any inherent property
of the embedding optimization.

### 1.2 Li et al. — "Imagination Helps Visual Reasoning, But Not Yet in Latent Space" (arXiv:2602.22766)

**Causal mediation framing.** Li et al. cast the system as a causal chain:
**"the input as the treatment, the latent tokens as the mediator, and the
final answer as the outcome"** (arXiv:2602.22766). They probe the chain at
both edges:

- *Input → Latent (Input-Latent Disconnect).* Apply **"instance-level
  perturbations on the input X"** where **"the entire input sequence is
  altered"** and measure cosine similarity of the latent tokens. Finding:
  **"negligible changes to the latent tokens, suggesting that latent tokens do
  not effectively attend to the input"** (arXiv:2602.22766).
- *Latent → Answer (Latent-Answer Disconnect).* Two interventions:
  (i) **"forcing all latent tokens across different positions and instances as
  a shared identical tensor"**; (ii) for Mirage, injecting Gaussian noise,
  replacing with noise, or zeroing. Finding: **"perturbations on the latent
  tokens yield minimal impact on the final answer"** (arXiv:2602.22766).

**Models tested.** Three latent-visual-reasoning systems: **Monet**
(distillation-based), **LVR** (image-feature supervision via MSE; arXiv:2509.24251),
and **Mirage** (task-specific feature supervision). The same MLLM (Qwen2.5-VL-7B)
is the backbone for their alternative, **CapImagine**.

**CapImagine alternative.** Rather than reasoning in latent space, CapImagine
**"teaches the model to explicitly imagine using text"**: zoom-focused
intermediate images become **"concise and accurate captions that refocus the
highlighted visual semantics"**, manipulated images become text that
**"describe[s] the visual differences and explicitly verbalize[s] the key
information revealed by the manipulation"** (arXiv:2602.22766). They keep
17k high-quality instances out of the original 125k. Headline claim: text
imagination **"significantly outperforms complex latent-space baselines."**

**Crucial methodological note for the user's design.** Li et al.'s
Latent-Answer Disconnect intervention swaps latents for a constant tensor (or
noise) **while the image is still present in the reader's input.** A reader
that has direct access to the image can route around uninformative latents.
The user's setup explicitly removes the image from the reader, so the
disconnect-style intervention degenerates: zeroing latents collapses task
performance by construction. **The no-image-on-reader constraint is therefore
structurally immune to the specific Li et al. failure mode** — but this only
addresses Latent-Answer Disconnect. **Input-Latent Disconnect (the trained
generator ignoring its image input) is not avoided by the reader-side
constraint** and remains a live risk that needs its own probe.

### 1.3 Coconut (arXiv:2412.06769) — text-side latent reasoning

**Pitch.** **"Coconut utilizes the last hidden state of the LLM as a
representation of the reasoning state, termed 'continuous thought.' Instead of
decoding this state into words, we feed it back to the model as the next input
embedding directly in the continuous space"** (arXiv:2412.06769). Curriculum
training: at stage `k`, the first `k` reasoning steps in the CoT are replaced
with `k×c` continuous thoughts; optimizer state is reset between stages.

**Authors' positive claim.** When decoded with the LM head, the first
continuous thought lands on token candidates like **"'180', ' 180' (with a
space), and '9'"**, and **"the interpretations of the first thought happen to
be the first intermediate variables in the calculation"** (arXiv:2412.06769) —
suggesting the latents carry actual intermediate state in their setting.

---

## 2. Adjacent critique literature — shortcut/inert-placeholder concern

### 2.1 "Do Latent Tokens Think?" (arXiv:2512.21711) — direct empirical critique of Coconut

This is the single most important critique paper for the proposed method.
Abstract verbatim:

> "Latent tokens are gaining attention for enhancing reasoning in large language
> models (LLMs), yet their internal mechanisms remain unclear. This paper
> examines the problem from a reliability perspective, uncovering fundamental
> weaknesses: latent tokens function as uninterpretable placeholders rather
> than encoding faithful reasoning. While resistant to perturbation, they
> promote shortcut usage over genuine reasoning. We focus on
> Chain-of-Continuous-Thought (COCONUT), which claims better efficiency and
> stability than explicit Chain-of-Thought (CoT) while maintaining performance.
> We investigate this through two complementary approaches. First, steering
> experiments perturb specific token subsets, namely COCONUT and explicit CoT.
> Unlike CoT tokens, COCONUT tokens show minimal sensitivity to steering and
> lack reasoning-critical information. Second, shortcut experiments evaluate
> models under biased and out-of-distribution settings. Results on MMLU and
> HotpotQA demonstrate that COCONUT consistently exploits dataset artifacts,
> inflating benchmark performance without true reasoning. These findings
> reposition COCONUT as a pseudo-reasoning mechanism: it generates plausible
> traces that conceal shortcut dependence rather than faithfully representing
> reasoning processes." (arXiv:2512.21711)

**Why this matters here.** The user's Variant A (reader-NLL) is structurally
even more permissive of shortcuts than Coconut — Coconut's training is at least
constrained by next-token prediction over long traces; reader-NLL can collapse
the entire visual-thinking pipeline into "make the reader output `y*`."
Variant B (GRPO × reader-LL) inherits the same risk via reward shaping.
**The mitigation isn't in the loss; it's in evaluation.** Adopt this paper's
two-pronged probe verbatim: (a) steering on `h` (do they actually do work?);
(b) OOD/biased-set evaluation (do they exploit artifacts?).

### 2.2 Survey: "Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought" (arXiv:2505.16782)

The survey explicitly catalogs the failure modes the user is worried about:

- *Shortcut mechanisms (§5.2):* **"correct outputs may result not from latent
  reasoning, but from shortcut strategies acquired during pre-training"** —
  models exploit **"surface-level correlations or pattern completion, rather
  than engaging in true inference"** (arXiv:2505.16782).
- *Token-as-placeholder (§3.1):* **"the structural organization of tokens is
  more critical than their semantic content. Surprisingly, replacing meaningful
  tokens with neutral placeholders yields negligible performance loss"**
  (arXiv:2505.16782).
- *Faithfulness (§7.1):* **"models often perform reasoning in their 'heads'
  that is not reflected in their verbalized CoTs, raising concerns about
  unfaithful or hidden internal processes"** (arXiv:2505.16782).
- *Implicit-reasoning interpretability:* **"The shift from explicit to implicit
  reasoning further introduces significant challenges for identifying errors
  and understanding how the model draws a particular conclusion"**
  (arXiv:2505.16782).

The taxonomy is **token-wise (horizontal) vs layer-wise (vertical)** latent CoT.
The user's method is squarely token-wise — same regime where the survey reports
most placeholder/shortcut failures.

### 2.3 LVR (arXiv:2509.24251)

LVR is the closest prior to a "reconstruct visual content" objective and is
a useful contrast point. Its loss is

> "L_LVR = (1/T_v) Σ_{t=1..T_v} ||h_t − v_t||₂²" (arXiv:2509.24251)

where `v_t` is a visual-encoder embedding for an ROI patch selected from a
**pre-annotated bounding box**. This is supervision *toward visual content*
(MSE to the encoder's own tokens) — the opposite end of the spectrum from
reader-NLL. The user's design literally drops this anchor.

---

## 3. Soft-prompt-tuning failure modes — are optimized latents pathological?

### 3.1 "Towards Interpretable Soft Prompts" (arXiv:2504.02144)

> "soft prompts and other trainable prompts remain a black-box method with no
> immediately interpretable connections to prompting" (arXiv:2504.02144)

The paper's main empirical finding is **"a fundamental trade-off between
interpretability and the task-performance of the trainable prompt"** — when
optimized purely for downstream task loss, soft prompts drift away from any
human-readable structure. Direct relevance: optimizing `h` against reader-NLL
is exactly that pure-task optimization regime.

### 3.2 "Soft Prompt Threats" (arXiv:2402.09063)

> "embedding space attacks circumvent model alignments and trigger harmful
> behaviors more efficiently than discrete attacks or model fine-tuning"
> (arXiv:2402.09063)

The high-level finding — that optimizing in continuous embedding space
**"can extract supposedly deleted information from unlearned LLMs"** — is
strong evidence that **optimization over a frozen LM's embedding inputs
finds off-manifold solutions that exploit the LM's interior in unintended
ways.** The user's reader-NLL gradient is the same operator class. This is the
most direct prior for the off-manifold concern (#2).

### 3.3 "Soft prompting might be a bug, not a feature" (OpenReview MHWDdMEJ5s)

The PDF wasn't extractable, but the cited claim from secondary sources is:
out-of-distribution regions of the LLM's embedding space allow a user to
**conceal an adversarial soft prompt within an unassuming multimodal input**.
Same off-manifold theme.

**Implication for the proposed method.** Reader-NLL gradients have no
mechanism to keep `h` near the natural visual-token manifold. Need either an
explicit manifold constraint (norm clip, KL to prior, projection) or a
diagnostic that monitors `h`'s distribution vs the reader's natural visual
token distribution at training time. The proposal currently has neither.

---

## 4. Reader-grounded objectives — prior art

The user's setup — gradients flowing through a frozen reader's likelihood of
the gold answer when fed only the generator's continuous outputs — has only
loose precedent:

- **Lester et al. soft-prompt tuning (arXiv:2104.08691)** is the canonical
  ancestor: train continuous prompts against frozen LM next-token loss.
  Reader-grounded latent visual reasoning is essentially "soft-prompt tuning,
  but the prompts are *generated per-input* by another MLLM, and they sit in
  the *visual* token slots of a frozen reader." Inherits all the soft-prompt
  pathologies above.
- **Embedding-space adversarial attacks (arXiv:2402.09063)** are the
  pessimistic reading: same operator, used to break alignment. The optimization
  succeeds at fooling the reader; nothing forces the embeddings to be
  semantically faithful.
- **Embed-RL (arXiv:2602.13823)** uses RL with embedding-space rewards for
  multimodal embeddings — different setting (retrieval/CoT for embeddings) but
  the closest match to "RL-weighted reader signal on a latent." Worth a closer
  read if the user wants to position Variant B.
- **No paper found** that matches the exact recipe — frozen reader MLLM, no
  image given to reader, generator trained against reader-NLL on the answer
  with continuous visual-slot embeddings. If this is correct, **the proposed
  method has a defensible novelty claim, but it sits in the worst neighborhood
  for shortcut/off-manifold pathologies.**

Searches tried (terms): "frozen reader" / "decoder-grounded loss" /
"reward-weighted reader-NLL" / "frozen LLM continuous prompt visual token".
Closest hits were variants of soft-prompt tuning or general embedding RL —
none with the no-image-on-reader constraint.

---

## 5. Implications for the proposed method

- **Concern 1 (shortcut / answer-encoding) — confirmed live risk.**
  arXiv:2512.21711 demonstrates the same pathology empirically in the
  text-only Coconut setting; arXiv:2505.16782 §5.2 generalizes it; the user's
  Variant A removes even the next-token-prediction guard rail Coconut had.
  **Required mitigation:** import the §2.1 evaluation protocol — steering on
  `h` to confirm it carries reasoning-critical info, plus a
  biased/OOD-controlled benchmark to detect artifact exploitation. Don't ship
  the method without both.

- **Concern 2 (off-manifold latents) — confirmed live risk.**
  arXiv:2402.09063 + arXiv:2504.02144 together establish that pure-task
  optimization of frozen-LM input embeddings finds off-manifold solutions.
  **Required mitigation:** monitor `h`'s norm and per-dim distribution against
  the reader's natural visual-token statistics during training. Consider an
  explicit manifold constraint (KL to a prior fit on the reader's visual
  tokens, or distance to nearest natural token).

- **Concern 3 (Latent-Answer Disconnect) — partially structurally avoided.**
  Li et al.'s specific intervention (arXiv:2602.22766) — replacing latents
  with a constant while the image is still in the reader's context — degenerates
  in the no-image-on-reader setting; that failure mode cannot be reproduced
  there. **However, Input-Latent Disconnect (the generator ignoring its image)
  is *not* addressed by the reader-side constraint.** A clean version of Li
  et al.'s input-perturbation probe should still be run on the generator side.

- **Positioning vs Monet.** Monet's latents are visually grounded *because the
  Stage-2 teacher saw the image and produced the targets*. Reader-NLL provides
  no analogous grounding. The proposed method either (a) needs an auxiliary
  grounding term (visual reconstruction, contrastive image–latent alignment)
  or (b) has to *empirically* show the latents are visually grounded despite
  having no such term — which is the strongest possible result and exactly
  what the POC should be designed to falsify.

- **Variant comparison.** Variant A (differentiable reader-NLL) is the more
  exposed of the two: pure gradient descent against a frozen reader is
  textbook off-manifold-attack territory. Variant B (GRPO × reader-LL)
  inherits the shortcut risk but the discrete-action / reward-shaping framing
  has slightly less aggressive embedding-space optimization pressure. Worth
  running the §2.1 probes on both.

- **Bar for the method paper.** Without a steering-style probe and an
  off-manifold/shortcut diagnostic baked in from day one, reviewers familiar
  with arXiv:2512.21711 will (correctly) reject the contribution as
  unverified. The POC's primary scientific output should be those probes,
  not benchmark numbers.
