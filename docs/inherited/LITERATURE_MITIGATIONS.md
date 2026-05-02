# Literature Mitigations — Reader-Grounded Latent Visual Reasoning

Supplemental recon to `LITERATURE_RECON.md`. Scope: candidate fixes for the
shortcut / off-manifold / weak-grounding pathologies the POC observed
(K=1 trivial reachability, off-manifold norms, 50% shortcut on held-out q,
reader-transfer failure to Monet-7B, partial K=4 multi-question rescue).

---

## 1. VQ / codebook approaches for soft prompts

### 1.1 VIP — Vector-Quantized Input-Contextualized Soft Prompts (arXiv:2205.11024, EMNLP 2022)

The original "VQ-on-soft-prompts" paper. Search results describe the codebook as
**"1000 codebook vectors as parameters learned through EMA (Exponential Moving
Average)"** with the design framed as a **"trade-off between input-agnostic soft
prompts and input-contextualized soft prompts by sampling prompts from a limited
set of learnable codebook vectors"** (search summary of arXiv:2205.11024;
abstract not directly extractable from arXiv HTML). A "Quantizer" **"reduces
noise in contextual prompt representations by discretizing the continuous space
of contextualized prompts"** — the explicit motivation is variance reduction
across random initializations, not interpretability or shortcut control. The
authors report **"more stable performance with lower variance across random
initializations of prompt embedding and sentence encoder parameters"**. There is
**no claim about reduced shortcut behavior** in any extracted material.

### 1.2 VQ-Prompt for Continual Learning (arXiv:2410.20444, NeurIPS 2024)

Smaller codebook regime. From the HTML: **"The number of keys and prompt
elements N is 10."** Two regularization terms:
**`ℒ_VQ = ‖sg[p′] − p‖²`** and **`ℒ_Commit = ‖p′ − sg[p]‖²`** with
λ_q = 0.4, λ_c = 0.1, joined as **`ℒ = ℒ_CE + λ_q·ℒ_VQ + λ_c·ℒ_Commit`**.
Reported gain over a soft-prompt baseline is small but consistent: **"VQ-Prompt-S
achieves an FAA value of 78.05 on 10-task ImageNet-R"** vs **"Soft-Prompt
achieves its best performance of 77.15."** The motivating claim is that discrete
prompts provide **"necessary abstraction to effectively represent"** task
knowledge; again no shortcut/interpretability measurement is reported.

### 1.3 Abstract Chain-of-Thought / ACoT (arXiv:2604.22709, IBM, Apr 2026)

Most relevant CoT-side cousin. ACoT introduces **"a discrete latent reasoning
post-training mechanism in which the language model produces a short sequence of
tokens from a reserved vocabulary in lieu of a natural language CoT"**. The
codebook is **"entirely in a newly introduced reserved vocabulary"** trained via
**"constrained decoding with the codebook"** — explicitly designed so abstract
tokens are **"not a quantized reconstruction of a teacher rationale"** but a new
compact reasoning language. Search summary suggests this allows the model to
**"potentially explore other reasoning pathways, rather than being constrained
to that of the teacher CoT"**. No reported numbers on shortcut behavior in
extracted content.

**Implication for the POC.** VQ codebooks are plausible as a manifold-constraint
mechanism (forces `h` to live on a learned discrete grid), but published
codebook sizes for soft prompts are small (N=10 to ~1000), and **no paper
extracted reports VQ specifically mitigating shortcut/answer-encoding behavior
in the way the POC needs**. VQ would address Concern 2 (off-manifold) but is
silent on Concern 1 (shortcut).

---

## 2. LVR (arXiv:2509.24251) — full method

### 2.1 What `v_t` are and how they are picked

From the HTML: **"the vision encoder transforms the image into visual features
𝐕=vision(𝐗_v)"** which are then **"mapped into a representation aligned with
the language model's latent space, denoted as 𝐕_T=proj(𝐕)"**. ROI selection:
**"Based on the ROI bounding box, LVR efficiently selects the corresponding
patches and retrieves their indices 𝐈={I₁,I₂,I₃,…,I_Tv}"** in O(1) time
during SFT. So `v_t` are the projected visual-encoder embeddings of the
patches that fall inside a **pre-annotated bounding box**.

### 2.2 Stage structure

Two stages, as per the HTML: **"The first stage is Supervised Finetuning (SFT),
which jointly optimizes LVR's internal processes alongside next-token
prediction for text generation. The second stage applies Reinforcement Learning
(RL)."** The joint SFT objective is **`ℒ = ℒ_NTP + λ_LVR·ℒ_LVR`** with the MSE
to ROI patch embeddings as the auxiliary term.

### 2.3 Quantitative results

| Benchmark | Qwen2.5-VL | LVR | Δ |
|---|---|---|---|
| V* | 78.5 | 81.7 | +3.2 |
| MMVP | 66.7 | 71.7 | +5.0 |
| Counting | 66.7 | 70.8 | +4.1 |
| IQ-Test | 26.0 | 29.3 | +3.3 |

(extracted via WebFetch on arxiv.org/html/2509.24251v2.) **No published
λ_LVR ablation**, **no comparison to Monet/Mirage/CapImagine** in the paper
itself. The paper does report a curious negative result internally: **"Fixed
Token achieves the best performance, while Mode Switching Loss fails to work
as intended."**

### 2.4 Did it actually work as a "structural fix toward visual content"?

It moved benchmarks +3 to +5 points over the Qwen2.5-VL backbone — meaningful
but not a transformation. The Li et al. critique (arXiv:2602.22766, covered in
the original recon) tested LVR alongside Monet and Mirage and concluded that
**"perturbations on the latent tokens yield minimal impact on the final
answer"** for all three — **so even with explicit MSE-to-visual-encoder
supervision, the latents still functioned closer to inert placeholders than
genuine visual content carriers.** This is the most important data point for
the POC: a literal "MSE to visual tokens" loss is insufficient by itself.

---

## 3. Post-2025-12 papers on latent visual reasoning

The space has been very active since the original recon. New entries
chronologically:

### 3.1 ILVR — Interleaved Latent Visual Reasoning (arXiv:2512.05665, Dec 5 2025)

**"ILVR interleaves textual generation with latent visual representations that
act as specific, evolving cues for subsequent reasoning."** Method:
**"a self-supervision strategy where a momentum teacher model selectively
distills relevant features from ground-truth intermediate images into sparse
supervision targets."** Same fix family as Monet (teacher-distilled targets)
but with sparsification — extends the supervised-target line.

### 3.2 DMLR — Reasoning Within the Mind (arXiv:2512.12623, Dec 16 2025)

Test-time framework. **"DMLR is a test-time Dynamic Multimodal Latent Reasoning
framework that employs confidence-guided latent policy gradient optimization to
refine latent think tokens for in-depth reasoning."** A **"Dynamic Visual
Injection Strategy"** retrieves the most relevant visual features at each
latent token and re-injects them. This is a *runtime* mitigation — keeps
visual access active rather than purely latent. Different setting from POC
(reader has no image), but the design philosophy is "don't trust the latent;
keep visual evidence in scope."

### 3.3 SkiLa — Sketch-in-Latents (arXiv:2512.16584, Dec 18 2025)

**"a novel paradigm for unified multi-modal reasoning that expands the
auto-regressive capabilities of MLLMs to natively generate continuous visual
embeddings, termed latent sketch tokens."** Uses **"a latent visual semantics
reconstruction mechanism. … A sketch encoder extracts visual embeddings from
these sketches to serve as reconstruction targets."** Same teacher-target
recipe as Monet/Mirage but anchored on intermediate sketch images.

### 3.4 LIVR — Latent Implicit Visual Reasoning (arXiv:2512.21218, Dec 24 2025)

**The closest published sibling to the POC.** Trains latent tokens with **no
explicit supervision** via attention bottlenecking. Verbatim from the HTML:
**"the answer tokens can only attend to the prompt tokens Q and the latent
tokens L, but cannot attend to the visual inputs I"**, and additionally
**"we also prevent the prompt tokens Q from attending to the visual inputs I."**
**K=16 latent tokens by default**, ablated over K∈{4,8,16,32}; K=16 best,
K=32 degrades. Trained with **"the standard negative log likelihood (NLL)
objective"** with loss **"computed only on the answer tokens"** — no
auxiliary loss. Reports +6.24 mean accuracy over direct SFT (single-task,
Qwen2.5-VL-3B), +2.77 (multi-task, Qwen3-VL-4B), and **+19.40% on Jigsaw,
+20.00% on Visual Spatial Planning vs Mirage**. Probes:
**"Latent tokens largely occupy the same region as image tokens…suggesting
that many latent representations align with the model's visual feature
space"** (t-SNE), and removing latents drops Localization 83.61 → 76.23.

**Why this is the most important paper for the POC.** LIVR demonstrates the
"latents-as-bottleneck for the same model" version of the user's idea, with no
auxiliary loss, and reports the latents end up *inside the visual feature
manifold* — exactly the property the POC failed to achieve. Differences worth
flagging: LIVR uses **a single shared model with attention masking**, not a
separate frozen reader, and the answer tokens still have **the option to
re-attend to the image in Stage 2**. The POC's no-image-on-reader constraint is
strictly stronger; LIVR's manifold-alignment evidence is therefore not direct
evidence the POC's setup will achieve the same.

### 3.5 LaViT — Aligning Latent Visual Thoughts (arXiv:2601.10129, Jan 15 2026)

**The most directly applicable mitigation paper.** Distills two signals from a
Qwen2.5-VL-32B teacher: **`ℒ_concept`** (cosine similarity to teacher's
last-layer visual features) and **`ℒ_traj`** (KL on cross-attention
trajectories), combined with NTP as **`ℒ_total = ℒ_ntp + λ·(ℒ_concept +
ℒ_traj)`** with **λ = 0.3**. Headline result: **"+16.94% Relative Depth"** and
**"+15.67% Relative Reflectance"** over Qwen2.5-VL-3B; LaViT-3B exceeds LVR-7B
on Relative Depth (78.23 vs 76.61).

The shortcut-prevention mechanism is **Curriculum Sensory Gating**: a
time-dependent scalar **γ(t)∈[ε,1]** controlling visual attention, implemented
as bias **`B_gate(t) = ln(γ(t))`**. Phase 1: **"The direct visual path opens
gradually, following the cosine curve…resulting in a large negative bias that
creates a strict Latent Bottleneck, mathematically compelling the model to
compress necessary visual information into V."** Phase 2 fully opens the gate
(γ=1) for inference compatibility. The authors explicitly identify the failure
mode this addresses: **"student models frequently mimic a teacher's textual
output while attending to fundamentally divergent visual regions, effectively
relying on language priors rather than grounded perception"** — i.e.,
Concern 1 (shortcut) verbatim.

### 3.6 Latent CoT for Visual Reasoning (arXiv:2510.23925)

Reformulates as posterior inference + **"diversity-seeking reinforcement
learning algorithms"** with a sparse token-level reward that **"encourage[s]
diverse, high-likelihood latent CoT, overcoming deterministic sampling
limitations and avoiding reward hacking."** Relevant to Variant B (GRPO):
diversity terms are a documented mitigation against the reward-hacking risk
in RL-on-latents.

### 3.7 PLUME (arXiv:2604.02073)

Reports an instructive negative finding: **"a direct transition from explicit
CoT supervision to latent-only execution is unstable, because semantic
grounding does not transfer reliably into hidden-space rollout."** Confirms
the POC's experience that latents drift off-grounding when no auxiliary anchor
is in place.

---

## 4. Auxiliary objectives that documentedly work for latent reasoning

Cataloguing the specific loss terms with reported positive effect, for the
"add a grounding term" arm of the POC:

- **Cosine to teacher last-layer visual features** (LaViT `ℒ_concept`,
  arXiv:2601.10129; +16.9 points on Relative Depth). Stronger than MSE in their
  ablations (paper gives full architecture rationale).
- **KL on cross-attention trajectories**, top-K=8 sparsified
  (LaViT `ℒ_traj`). **"we subject 𝒜_traj to Top-K (k=8) sparsification,
  thereby ensuring sparse and noise-free supervision."**
- **MSE to ROI visual-encoder patches** (LVR `ℒ_LVR`,
  arXiv:2509.24251; +3 to +5 points; Li et al. 2602.22766 still finds latents
  largely inert under perturbation — partial fix).
- **Distillation from intermediate sketch-image embeddings** (SkiLa,
  arXiv:2512.16584).
- **Momentum-teacher selective distillation** (ILVR, arXiv:2512.05665).
- **Stage-2 alignment to teacher-defined targets that saw the image**
  (Monet `L_align-obs` and `L_align-latent`, arXiv:2511.21395 — already in
  original recon).

**No paper extracted reports a contrastive image-latent alignment loss
(CLIP-style InfoNCE between the generator's `h` and the reader's natural
visual tokens)** producing strong gains in the latent-visual-reasoning
literature, despite this being a natural construction. Either (a) no one has
tried it in this exact form, or (b) it doesn't work and isn't published. Worth
a probe but cannot be cited as a known-working recipe.

**Auxiliary terms that documentedly do NOT work alone.** The Li et al. probe
(arXiv:2602.22766) found the latents in **all three** of Monet (distillation),
LVR (MSE-to-visual), and Mirage (task-feature) remained shortcut-prone under
their causal mediation interventions — so any single auxiliary term is
insufficient evidence of grounding without the §2.1 perturbation probes from
arXiv:2512.21711.

---

## 5. Multi-question / multi-task on the same image as regularization

The POC's mitigation D (multi-question consistency at K=4) has weaker direct
literature precedent than expected. Findings:

- **MQMA — Multiple-Question Multiple-Answer Text-VQA (arXiv:2311.08622,
  Amazon).** Uses multiple Q/A on the same image as input/output structure
  with a **"novel MQMA denoising pre-training task which is designed to teach
  the model to align and delineate multiple questions and content with
  associated answers"**. Framed as efficiency + alignment, **not as a
  shortcut-mitigation regularizer**. No reported probe on whether this forces
  visual content encoding.
- **AttReg (arXiv:2102.01916)** and **VGQE** (Springer) regularize via
  **visual attention supervision** or **visually-grounded question encoders**,
  not via multi-question consistency on the same image.
- **arXiv:2004.05704 — "Visual Grounding Methods for VQA are Working for the
  Wrong Reasons!"** delivers the cautionary verbatim:
  **"performance improvements are not a result of improved visual grounding,
  but a regularization effect which prevents over-fitting to linguistic
  priors, and random, insensible cues also result in similar improvements."**
  This is directly relevant: the POC's K=4 multi-question gain (mean −1.5 nat
  on q3, 70% win) may be a generic regularization effect rather than evidence
  of visual content encoding. A control ablation with random/insensible
  auxiliary tasks is required to disentangle.

**No paper extracted explicitly uses "multi-question per image as a
regularizer to force visual content encoding in latent embeddings."** The
closest published structural analog is LIVR's bottleneck (single question, but
the model has no other path to visual info during answer generation). The
mitigation D probe sits in genuinely lightly-explored territory; the
arXiv:2004.05704 caveat means a negative-control ablation is mandatory before
attributing any gain to grounding.

---

## 6. Bottom line for the POC

- **Closest published sibling: LIVR (arXiv:2512.21218).** Same "no auxiliary
  loss, latents must encode visual info via bottleneck" pattern; reports
  latents end up on the visual manifold per t-SNE. But LIVR's bottleneck is
  attention-mask within one model; POC's frozen-separate-reader constraint is
  stronger and not directly covered.
- **Strongest documented auxiliary fix: LaViT (arXiv:2601.10129).** Cosine to
  teacher visual features + KL on attention trajectories + curriculum gating;
  +15-17 points on perception benchmarks; explicitly designed against the
  shortcut/language-prior failure mode. If POC adds a grounding term, this is
  the recipe to copy.
- **Cautionary base rate.** Li et al. (arXiv:2602.22766) found Monet, LVR, and
  Mirage **all** still showed Latent-Answer Disconnect under perturbation
  despite different auxiliary supervision strategies. A single new auxiliary
  loss is unlikely to fully fix Concern 1.
- **VQ codebooks** are a plausible Concern-2 (off-manifold) fix but no
  extracted paper validates them as a Concern-1 (shortcut) fix.
- **Multi-question regularization** has weak literature precedent; the
  arXiv:2004.05704 result demands a random-control ablation before claiming
  the K=4 gain reflects visual grounding rather than generic regularization.
