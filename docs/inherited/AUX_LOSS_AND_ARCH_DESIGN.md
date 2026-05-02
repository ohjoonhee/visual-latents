# Auxiliary Loss Stack & Generator Architecture Design — Round-3 POC

Companion to `LITERATURE_RECON.md` and `LITERATURE_MITIGATIONS.md`. Scope:
the *training-time* design for round-3 POC of reader-grounded latent visual
reasoning, given round-2's diagnosis that within-reader geometric mitigations
do not transfer.

**Key round-2 takeaway driving this design:** norm-reg, low-rank, and
multi-Q each cut held-out NLL by 30–40% **within-reader (φ₁)** but **none
fixed transfer to φ₂ = Monet-7B** (POC-3 transfer drop +5–6 nat persisted).
That is, geometric / per-sample regularization changes *which direction* the
optimizer finds in φ₁'s embedding space but not whether that direction
exploits φ₁-specific decoder structure. The structural fix has to enter at
**training time**, not as a per-sample geometric anchor.

This doc specifies (A) the full combined loss stack, (B) four candidate
generator architectures, (C) the multi-reader sourcing logistics that the
loss stack needs to actually exist.

References by arXiv ID, all verified via WebFetch on the paper HTML this
session unless flagged otherwise.

---

## Part A — Auxiliary loss stack

### A.0 Notation

- `θ`: generator parameters (trainable). Generator emits `h ∈ ℝ^{K×D}` from
  image `x` (no question, see §A.4).
- `D`: reader hidden dim. Qwen2.5-VL-7B → `D = 3584`. Monet-7B → same.
- `φᵢ`: frozen reader `i ∈ {1, …, R}`. R is a tuning knob; round-3 POC
  uses R=2.
- `q`: question. `y*`: gold answer.
- `ψ`: visual teacher (e.g., Qwen2.5-VL-32B as in LaViT, or the model's own
  vision encoder + projector — design choice, see §A.1).
- `T`: training step.

### A.1 Component 1 — LaViT-style auxiliary grounding (arXiv:2601.10129)

LaViT is the strongest published recipe for the exact failure mode we
observed (latents drift off-manifold and into reader-specific shortcuts).
Verified verbatim from the v1 HTML:

- **`ℒ_total = ℒ_ntp + λ · (ℒ_concept + ℒ_traj)`** with **λ = 0.3 across
  all experiments** (arXiv:2601.10129).
- **`ℒ_concept = 1 − (1/B) Σ CosSim(φ_mlp(h_z), V_sem)`** — student latent
  passes through small MLP `φ_mlp` and is matched (cosine) against teacher
  *post-attention, last-layer* visual features `V_sem`. The teacher's
  features have already attended to the textual instructions, so `V_sem`
  is **contextualized** visual content, not raw CLIP/SigLIP. (Verbatim:
  *"these image token representations have effectively interacted with the
  textual instructions Q. Therefore, V_sem represents contextualized visual
  thoughts."*)
- **`ℒ_traj = (1/B) Σ_z Σ_i D_KL(A_traj || A_student)`** — KL on
  cross-attention weights between student and teacher. Teacher attention
  `A_traj` is averaged over **all layers L and heads H**, then **Top-K=8
  sparsified**. Student attention `A_student` is at the same student layer
  the latent was emitted from.
- **Curriculum Sensory Gating.** A scalar `γ(t) ∈ [ε, 1]` gates the visual
  attention path via additive bias `B_gate(t) = ln γ(t)`. Phase 1 (warmup,
  T_w = 400 steps): `γ(t) = ε + (1 − ε)/2 · [1 − cos(πt/T_w)]` — cosine
  ramp, **starts blocked, opens by T_w**. Phase 2 (T > T_w, 600 steps):
  `γ = 1` (gate fully open) for inference compatibility.

#### A.1.1 Choices for our setting

The LaViT paper applies these losses *inside a single VLM* (the student is
the same model that consumes the latent at inference). Our setup has a
*frozen separate reader* — a stricter constraint — so two adaptations are
needed:

**Choice 1 — what plays the role of `V_sem`?**

Three options, in order of preference:

| Option | Identity of `V_sem` | Trainable? | Pro | Con |
|--------|---------------------|------------|-----|-----|
| (a) **Reader-φ₁'s own post-merger visual tokens for image x** | Frozen reader's own vision encoder + projector output, post-merger, taken at the natural visual-token positions that a real reader pass would consume | No | Same model family, same geometry, exactly the manifold we want `h` to land on. POC-4 already characterized: μ‖·‖=57.86, intra-image cos=0.295, inter-image first-token cos=0.845 (`results/visual_baseline.json`) | Reader-specific anchor — risks pulling `h` into reader-1-specific encoder geometry, partially undoing the multi-reader transfer goal |
| (b) **Average over all R readers' visual tokens** | Element-wise mean of `{φᵢ.visual(x)}_{i=1..R}` after Procrustes-aligning if dims match (they do: same arch family, both 3584) | No | Family-shared signal | Different readers may disagree → mean may be a non-natural blend |
| (c) **External teacher (Qwen2.5-VL-32B post-attention features)** | LaViT's literal recipe | No (frozen) | Independent of any specific 7B reader; richest features | Teacher is much larger → high VRAM cost; arch is similar but not identical to the readers (32B vs 7B differ in width, not in vision encoder family — Qwen2.5-VL series shares vision encoder + projector design, so projector outputs should be comparable in geometry but not in scale) |

**Recommendation for round-3 POC:** **(a) reader-φ₁'s own post-merger visual
tokens**, because (i) `results/visual_baseline.json` already exists for it,
(ii) the post-merger token has dimension 3584 and is exactly the shape `h`
must occupy, (iii) it tests whether anchoring to one reader's manifold is
sufficient for transfer to a sibling reader (the round-2 finding suggests
"probably not, but we should measure"). Add (b) — averaged over R readers
— if POC fails the round-3 transfer bar.

**Choice 2 — pre-projector or post-projector? Pooled or token-level?**

LaViT uses *post-attention last-layer*, token-level (no pooling) because
their student emits a sequence and they need a sequence to align against.
We do too (h is K×D). The match should be **token-level, post-projector,
post-merger**. (Pre-projector / pre-merger would have the wrong dimension —
Qwen2.5-VL's vision encoder outputs raw ViT features at a different dim
that the projector + merger collapses to D=3584.)

**Choice 3 — which layer of φ₁ for `ℒ_traj`?**

LaViT uses all layers averaged, Top-K=8 sparsified. We can replicate that,
but the optimization signal is weak when K_visual is huge (Qwen2.5-VL's
visual token count varies with image resolution; typical 200–400 tokens).
The reader's natural visual block has length T_v ≈ 200; our `h` has length
K=16. So the cross-attention to attend to is the *answer-token cross-attn
to visual-position-within-h*, an `|y*| × K` matrix, while the teacher's is
`|y*| × T_v`. Direct KL is shape-mismatched.

**Adaptation:** instead of KL on attention weights at the same shape, use
**aggregated attention mass into the visual block**, i.e., per answer-token
`t_a`, the scalar `α_t = Σ_{j ∈ visual} A[t_a, j]`. Then KL becomes a 1-D
distribution over answer tokens (shape `|y*|`), comparing how much
cumulative attention the answer routes through the visual block. This is
crude but well-defined under the shape mismatch.

**Alternative, cleaner:** drop `ℒ_traj` for round-3 POC. The LaViT
ablation reports `ℒ_concept` carries most of the gain; `ℒ_traj` is
incremental. Our shape mismatch makes `ℒ_traj` design-dependent, so we
get most of the value from `ℒ_concept` alone with less risk of
implementation bugs. **Add `ℒ_traj` only if `ℒ_concept` alone misses the
round-3 bar.** (LaViT Table 3: removing `ℒ_traj` was modest; removing
curriculum gating was the largest degradation, MMVP 67.33→59.33%.)

**Choice 4 — curriculum gating in our setup.**

LaViT's gating regulates visual-input access during latent emission. In our
setup the generator already has full image access; the *reader* is the one
bottlenecked. The closest analog: gate `ℒ_NLL_reader` itself.

Concrete schedule: **start with `ℒ_concept` weighted high (λ_c large) and
`ℒ_NLL_reader` weighted low; anneal both toward the steady-state ratio.**
Pseudocode in §A.6 below.

#### A.1.2 Definition for round-3

```
ℒ_concept(θ) = 1 − (1/(K · B)) · Σ_{b, k} cos(MLP(h_b,k), V_sem_b,k')
```

where `V_sem_b` is `φ₁.visual(x_b).hidden_states[-1]` (post-merger,
last-layer of vision tower; T_v × D), pooled to K positions via
**learned-query pooling** — see §B.3 for the Q-Former emitter (the pooling
recipe is symmetric). For round-3 POC, simplest recipe: **k' = k mod T_v**
(modular index — gives every `h_k` a target somewhere in the visual block,
no learned pooling needed). When K << T_v, this means several h-positions
share a target (random subsampling of T_v).

**Scope:** `MLP` is a small bottleneck `D → D/2 → D` adapter, trainable, so
`ℒ_concept` doesn't lock `h` to be literally the encoder output (which
would be redundant — we already have it). Match in *direction* not norm.

**Initial λ range:**
- LaViT-paper value: λ = 0.3.
- Sweep: {0.1, 0.3, 1.0}.
- Round-3 default: **0.3**.

### A.2 Component 2 — Multi-reader NLL

Round-2's negative result — geometric mitigations do not transfer — points
directly at multi-reader as the structural fix. If `h` must reduce NLL
under R independent readers simultaneously, the only solution that scales
to all R is one that exploits *shared* (i.e., visually-grounded) reader
structure rather than reader-1-specific decoder peculiarities.

#### A.2.1 Form

```
ℒ_NLL_multi(θ) = (1/R) · Σ_{i=1..R} − log φᵢ(y* | q, h_{1:K})
```

Sum (or mean — same under fixed R) over R frozen readers per step.
Gradient flows through each reader's frozen weights into `h` and back into
θ.

#### A.2.2 Reader candidates

See **Part C** for the full sibling-fine-tune survey. Summary:

| Reader candidate | Source | Compatibility (3584-dim, same vision encoder) | Cost |
|---|---|---|---|
| Qwen2.5-VL-7B-Instruct (φ₁) | Qwen/Qwen2.5-VL-7B-Instruct | yes (anchor) | base |
| Monet-7B (φ₂) | NOVAglow646/Monet-7B | yes — direct fine-tune of Qwen2.5-VL-7B (`huggingface.co/NOVAglow646/Monet-7B` already on disk) | identical to φ₁ |
| VL-Rethinker-7B (φ₃) | TIGER-Lab VL-Rethinker (NeurIPS'25) — RL fine-tune of Qwen2.5-VL-7B-Instruct | yes — direct fine-tune | identical to φ₁ |
| **LoRA-merged read-out** (φ₄, …) | Apply existing community Qwen2.5-VL-7B LoRA adapters at random α∈[0.5, 1.0] to base | yes — same vision tower preserved | very cheap (LoRA merge cost only) |
| Qwen2.5-VL-72B | Qwen/Qwen2.5-VL-72B-Instruct | architecture compatible BUT D=8192 ≠ 3584 → would need a learned projection | high VRAM, not compatible without adapter |

**Cross-architecture readers (LLaVA-1.6, InternVL, etc.)** — different
vision-token convention, different D. *Not* useful as direct frozen
readers in the same stack; would require per-reader adapters that the
generator emits into. **Out of scope for round-3 POC.** Optional ablation
later: train one cross-arch reader head as a stress test.

#### A.2.3 Gradient handling: sum vs. alternate

**Sum every step** (canonical multi-task):
```
loss = ℒ_NLL_multi + λ · ℒ_concept + μ · ℒ_norm + …
loss.backward()
```

**Alternate readers per step** (cheaper, biased per-step, equivalent over
a window):
```
i = step % R
loss = ℒ_NLL_φ_i + λ · ℒ_concept + …
```

**Recommendation: sum** (within VRAM budget; see §A.6). The whole point is
that `h` minimizes against *all* readers simultaneously; alternating
introduces gradient-direction bias toward whichever reader was most recent
and weakens the structural pressure. Alternation only if we exceed VRAM
budget.

#### A.2.4 Memory analysis (4× H100 80GB)

Per reader during forward + backward:

| Component | Memory (bf16, B=8) |
|---|---|
| Reader weights (frozen, no grad) | ~15 GB / 7B reader |
| Reader activations for backward (gradient flows *through* reader to h, but reader weights themselves are frozen — only need to retain activations for chain rule, not optimizer state) | ~8 GB / 7B reader at K=16, B=8, seq_len ≈ 256 |
| Total per reader (fwd+bwd, no optim) | ~23 GB |

Generator (Qwen2.5-VL-7B with LoRA, see §B.1) at B=8: ~30 GB (params 15GB + LoRA-relevant grads 1GB + activations 12 GB + optim 2GB).

**Per-GPU layout for 4× H100:**

- GPU 0: generator (30 GB) + reader φ₁ fwd-only sharded portion (~12 GB) → fits in 80 GB with margin
- GPU 1: reader φ₁ remaining + reader φ₂ (~30 GB)
- GPU 2: reader φ₃ + visual teacher V_sem cache or vision-tower-of-φ₁ (~30 GB)
- GPU 3: visual teacher full (Qwen2.5-VL-32B if used, ~65 GB) OR free for activation offload

With **R=2 readers (φ₁ Qwen2.5-VL-7B + φ₂ Monet-7B), no external teacher,
generator with LoRA-rank-32**, 4× H100 fits comfortably with B=8, K=16,
gradient checkpointing on the generator.

**With R=3** (add VL-Rethinker), tightens but works: drop B to 4 or
gradient-checkpoint readers as well.

**With external 32B teacher (LaViT-faithful):** drop teacher to bf16 on
GPU3, accept B=4. Or: pre-cache `V_sem` for the training set offline
(static teacher, static images — no need to recompute per step) and only
hold the cache in CPU + page in (~2 KB/image × 100k = 200 MB cache; tiny).
**Strongly recommended: pre-cache V_sem.** Frees GPU3.

### A.3 Component 3 — Norm regularization (round-2 mit-B confirmed within-reader)

Round-2 confirmed: `λ_norm = 0.1`, target μ = 57.86 (POC-4 measurement),
form `(‖h_i‖₂ − 57.86)²`, gives −38% held-out NLL with no training-fit
penalty. λ = 10 over-regularizes. Sweet spot is mild.

#### A.3.1 Form

Three candidates, in order from cheapest to richest:

```
# Form 1 (round-2 verified, simplest)
ℒ_norm = (1/(K·B)) · Σ_{b, k} (‖h_b,k‖₂ − μ)²    where μ = 57.86

# Form 2 (Huber, more robust to early-training outliers)
ℒ_norm = Huber_δ=10(‖h_b,k‖₂ − μ)

# Form 3 (full distribution match — KL between empirical h-norm distribution and natural-token-norm distribution; or MMD)
ℒ_norm = MMD²(empirical_h_norms, natural_token_norms)
```

**Recommendation:** Form 1 for round-3 POC (verified in round-2). Switch to
Form 2 only if early training produces large `‖h‖` spikes that destabilize
`ℒ_concept`. Form 3 is overkill for round-3.

#### A.3.2 Schedule

Round-2 used constant λ_norm = 0.1 from step 0. That worked within-reader.
Under multi-reader NLL the optimization landscape is harder and h might
need a brief warmup before being pulled toward the natural manifold.

**Recommended schedule:**
- Steps 0–T_w (T_w = 400, matching LaViT phase 1): linear ramp 0 → 0.1.
- Steps T_w+: constant 0.1.

**Initial λ range sweep:** {0.05, 0.1, 0.3, 1.0}. λ = 1.0 expected to
over-regularize per round-2 trend (λ=10 broke training).

### A.4 Component 4 — Multi-Q consistency / generator inputs

Round-2 N=2 multi-Q: −39% held-out (best of round-2, tied with norm-reg).
Round-2 N=5 multi-Q: −24% — not strictly comparable (different held-out
target), but at face value the simple N→generalization scaling didn't
compound. **The likely reason:** round-2 multi-Q optimized `h*` per-image
on the sum of N reader-NLLs over questions, but at that capacity (K·D =
57k floats for K=16), encoding all N (q, y*) pairs as a multi-prompt
remained reachable. The shortcut basin is wide enough.

**Structural fix at training time:** the *generator's input has no
question*. Then `h(x)` is a function of image alone — q-invariance baked in
architecturally. The reader receives `h(x)` + q.

#### A.4.1 Form

```
# Per training image x with K_q questions {(q_j, y*_j)}_{j=1..K_q}:
h = G_θ(x)      # generator sees image only — NO question
loss_NLL = (1/K_q) · Σ_j Σ_i − log φ_i(y*_j | q_j, h)
                # h is shared across all K_q questions and all R readers
```

This is the round-1 multi-Q probe + the round-2 transfer fix combined.

#### A.4.2 K_q (questions per image per step)

- K_q = 1: degenerate to single-Q. Reverts to round-1.
- K_q = 2: round-1 multi-Q result (−39% held-out).
- K_q = 3–5: ideal range per `LITERATURE_MITIGATIONS.md` §5.
- K_q ≥ 8: VRAM cost grows linearly in reader forwards.

**Recommendation: K_q = 3 for round-3 POC, sweep {3, 5} if time permits.**

#### A.4.3 Data prep

GQA already supports multiple questions per image (round-2 used 30 such
pairs). Need to extend `data/poc1_samples.jsonl` to add per-image
question lists (e.g., from GQA's full questions JSON for the same imageId
sequence). 100 images × 3 q each = 300 (image, q, y*) triples — manageable
in a half-day data-prep script.

### A.5 Component 5 — Random-control mitigation per arXiv:2004.05704

The Selvaraju et al. critique: visual-grounding gains in VQA often come
from generic regularization (the auxiliary loss prevents over-fitting to
language priors), not from grounding. They demonstrated this with random
attention cues that produced the same gains. **Mandatory check** before
attributing any round-3 gain to visual content.

#### A.5.1 Three control modes to run

For each of the round-3 candidate recipes (the full stack vs the
LaViT-only ablation vs the multi-reader-only ablation), rerun with the
following control variants:

| Control | What to randomize | Tests |
|---|---|---|
| **Shuffled (image, question) pairs** | Bind image x_i to question q_j from a *different* image. Generator sees x_i, reader is asked q_j with target y*_j. | If gains persist: gains are language-prior regularization, not visual content. |
| **Shuffled `V_sem` targets in `ℒ_concept`** | Per training step, shuffle `V_sem` across batch — `h_b` matched against `V_sem_{σ(b)}` for random permutation σ | If gains persist: `ℒ_concept` was a generic "norm-toward-something-on-manifold" prior, not image-specific grounding. |
| **Random target norm μ** | Replace μ = 57.86 with per-sample uniform draw from [30, 80] | If gains persist: `ℒ_norm` was a generic regularizer (round-2 already half-tested this — round-2's λ=0.1 gave gain even though norms were ~20 originally, suggesting *any* manifold pull helps). |

#### A.5.2 Pass/fail bar for round-3

A round-3 result counts as **genuine grounding** only if:
- Gain on real (image, q) pairs > Gain on shuffled (image, q) pairs by
  ≥ 50% of the gain magnitude. I.e., random-pair recipe should retain at
  most half the gain.
- Held-out generalization to **a different reader φ₂** is non-trivial
  (round-2 reader-transfer drop should shrink from 5–6 nat to ≤ 2 nat at
  K=16).

If both bars cleared → round-3 has a publishable structural finding
(per-image visual content learned, transferable across readers).
If only the first cleared → genuine grounding within φ₁ but not
transferable.
If neither cleared → revisit architecture (Part B).

### A.6 Final combined loss & training schedule

```python
# Round-3 POC default recipe — single-line summary
ℒ_total(θ) = ℒ_NLL_multi + λ_concept · ℒ_concept + λ_norm · ℒ_norm

# where (default values, LaViT-aligned):
#   ℒ_NLL_multi = (1/R) · Σ_i − log φ_i(y* | q, h_{1:K})  averaged over R readers
#                 and over K_q questions per image (q-invariant generator, §A.4)
#   ℒ_concept   = 1 − (1/(K·B)) · Σ cos(MLP(h_b,k), V_sem_b,k')   k' = k mod T_v
#   ℒ_norm      = (1/(K·B)) · Σ (‖h_b,k‖ − 57.86)²
#   λ_concept   = 0.3   (LaViT default; sweep {0.1, 0.3, 1.0})
#   λ_norm      = 0.1   (round-2 verified; sweep {0.05, 0.1, 0.3})
#   R           = 2     (φ₁=Qwen2.5-VL-7B-Instruct, φ₂=Monet-7B; sweep {2, 3})
#   K_q         = 3     (q's per image per step; sweep {3, 5})
#   K           = 16    (LIVR optimum; sweep {8, 16})
```

**Schedule (1000 total steps for POC; scale by 5–10× for proliferated
training):**

| Step range | λ_concept | λ_norm | ℒ_NLL weight | Rationale |
|---|---|---|---|---|
| 0 → 200 (warmup) | 0.3 → 0.3 | linear 0 → 0.1 | 0.1 → 1.0 | High concept-pressure early to plant `h` near manifold; let norm drift naturally; ramp NLL pressure once h is on manifold — prevents off-manifold shortcut basin |
| 200 → 800 (main) | 0.3 | 0.1 | 1.0 | Steady state |
| 800 → 1000 (anneal) | 0.3 → 0.1 | 0.1 | 1.0 | Reduce concept pressure to test that the gains are sticky (h stays on manifold without strong pull) |

End condition: held-out NLL on φ₁ stops improving for 100 steps OR
V_sem cosine on a held-out batch saturates.

**Pseudocode (training step):**

```python
def round3_step(batch, θ, optimizer, step):
    images, q_lists, y_lists = batch  # q_lists, y_lists are K_q each
    K_q = len(q_lists[0])

    # 1) Generator emits h from image alone (q-invariant)
    h = generator(images)   # [B, K, D]; details depend on Part B choice

    # 2) Reader NLL summed over R readers and K_q questions
    nll = 0
    for i in range(R):  # frozen readers
        for j in range(K_q):
            logits = readers[i](q_lists[j], h, image=None)
            nll += F.cross_entropy(logits, y_lists[j])
    ℒ_NLL_multi = nll / (R * K_q)

    # 3) Concept loss (cosine to teacher visual)
    with torch.no_grad():
        V_sem = teacher_or_φ1.visual(images).hidden_states[-1]   # [B, T_v, D]
    h_proj = mlp(h)                                                # [B, K, D]
    target_idx = torch.arange(K) % V_sem.shape[1]
    targets = V_sem[:, target_idx, :]                              # [B, K, D]
    ℒ_concept = 1 - F.cosine_similarity(h_proj, targets, dim=-1).mean()

    # 4) Norm regularization
    ℒ_norm = ((h.norm(dim=-1) - 57.86) ** 2).mean()

    # 5) Schedule λ values
    λ_norm_t = min(0.1, 0.1 * step / 200)
    nll_w_t = min(1.0, 0.1 + 0.9 * step / 200)

    loss = nll_w_t * ℒ_NLL_multi + 0.3 * ℒ_concept + λ_norm_t * ℒ_norm
    loss.backward()
    optimizer.step()
```

### A.7 Memory / compute budget for 4× H100 80GB

**Round-3 POC recipe (default): R=2, K=16, K_q=3, B=4 per GPU, 4 GPUs DDP, generator=LoRA r=32 on Qwen2.5-VL-7B.**

Per GPU memory estimate (no offload):
- Generator backbone (Qwen2.5-VL-7B bf16, frozen) : 15 GB
- LoRA params + grads + adam state : 1 GB
- 2 frozen readers (Qwen2.5-VL-7B + Monet-7B, bf16, weights only — no optim, no main grads) : 30 GB
- Visual teacher = φ₁ (shared with reader-0 weights — no double-load) : 0 GB additional
- Activations for fwd+bwd through generator + R readers + K_q questions, B=4, K=16 : ~25 GB
- **Total per GPU: ~71 GB** — fits in 80 GB H100 with ~10% margin.

If tight: enable gradient checkpointing on the generator backbone (2×
slowdown, halves activation memory).

**Throughput estimate:** Generator fwd+bwd ~ 2s/step (LoRA, B=4); per
reader fwd+bwd ~ 1s; K_q=3 questions × R=2 readers = 6 reader passes per
step. ~8s/step. 1000 steps ≈ 2.2 hours. Sweep of 5 hyperparameter cells ≈
11 hours, fits in an overnight.

Proliferated training (multi-image, larger dataset): 100k steps × 8s =
220 hours = ~9 days on 4× H100. Comparable to round-2's 7.5 hours total
wall-clock × 100 scale.

---

## Part B — Generator architecture options

The generator emits `h ∈ ℝ^{K×D}` from image x (and *not* from q — see
§A.4). Four candidates, each evaluated against:

- **Param count** (trainable).
- **Compute cost per forward** (FLOPs / wall time).
- **What needs trainable** (LoRA, full FT, head only).
- **Prior art** (which published method instantiates this).
- **Pros/cons against the round-2 shortcut-failure mode.**

### B.1 Same-VLM-with-special-tokens (Coconut-style / LIVR-style)

ASCII diagram:
```
                                                                    h_{1:K}
                                                                       ↑
  [BOS] [<|im_start|>vision]  IMG_EMB_1 .. IMG_EMB_T_v  [<|latent|>]_1  [<|latent|>]_2  ..  [<|latent|>]_K  [<|im_end|>]
        \________________________________________________/                              \________________________________________________/
                  vision tokens (real image)                                                    K hidden states at these positions  =  h
```

- Prepend (or append) K new special tokens `<|latent|>` after the image-pad
  block. The VLM's hidden states at those positions ARE h.
- The VLM is the same Qwen2.5-VL-7B-Instruct as φ₁ (or a separate copy if
  we want generator ≠ reader).

#### Spec

| Attr | Value |
|---|---|
| Trainable | LoRA r=32 on attention + MLP (~50 M params) OR full FT (~7 B) |
| Compute / fwd | One forward pass through the full VLM at seq_len ≈ 256+K |
| Param count (LoRA) | ~50 M trainable / 7 B base |
| Param count (full) | 7 B |
| Prior art | **LIVR (arXiv:2512.21218)** — almost exactly this, "We introduce K new special tokens, L = {l₁, …, l_K}, to the model's existing vocabulary" with K=16 default and unfrozen embedding-table rows for L. Also **Coconut (arXiv:2412.06769)** — `<bot>...<eot>` markers, hidden states fed back as embeddings. |
| Pro vs shortcut | (i) Reuses the model's full multimodal pipeline — h sits in the same residual-stream geometry as natural visual tokens by construction; (ii) LIVR's t-SNE shows latents on visual manifold *with no aux loss* — best published prior for our exact setup; (iii) the K extra tokens cost essentially nothing on top of the base forward |
| Con vs shortcut | (i) The model still has access to image tokens, so h positions are not bottlenecks unless we mask attention from h-positions to image (LIVR-style); without masking, h positions can be lazy "pass-throughs" of nearby image tokens; (ii) the model is huge — even at LoRA-r=32 there's enough capacity to learn shortcuts |

#### Recommendation

**Strong default.** This is what LIVR uses, with documented success on the
manifold-alignment property (the very thing round-1 failed at). For
round-3 POC, use this with LoRA-r=32 and **add the LIVR Stage-1
attention masking** (answer tokens cannot attend to image tokens, only to
h and q) to actively force the bottleneck.

### B.2 VLM-with-readout-head

ASCII diagram:
```
   image
     ↓
   vision encoder (ViT)        ← frozen
     ↓
   projector / merger          ← frozen (or LoRA)
     ↓
   T_v post-merger features
     ↓
   readout MLP                 ← trainable, the entire generator
     ↓
   h_{1:K}                     (project T_v × D → K × D, e.g. via average-pool + MLP per slot)
```

- Take the VLM's vision encoder + projector + merger (frozen). Output:
  T_v × D. Trainable head pools to K outputs.

#### Spec

| Attr | Value |
|---|---|
| Trainable | Just the head: MLP `D → 4D → D` per slot, possibly with K learnable query vectors that attend to T_v features. ~5–20 M params |
| Compute / fwd | Vision encoder forward (~2 GFLOPs at 224²) + tiny MLP. Fast. |
| Param count | ~5 M head, vision encoder + projector frozen (~600 M frozen) |
| Prior art | **LVR (arXiv:2509.24251)** — selects ROI patches, projects via existing projector. Matches "MLP head over vision features" shape. Also **VPP (Visual Prompt Pre-training)** is in the same family. |
| Pro vs shortcut | (i) The vision encoder is doing the visual lifting — head can't easily encode (q, y*) shortcuts because it never sees q (q-invariance is *baked in by construction*); (ii) the head is small enough that it cannot easily memorize per-image shortcuts; (iii) cheapest to train |
| Con vs shortcut | (i) The head's expressive capacity is so limited that it might not be able to express anything more sophisticated than a learned linear pool of visual tokens — closer to "just use the visual tokens directly" than to "visual reasoning"; (ii) without the LM's residual stream geometry, h's distribution is only what the head puts there — needs `ℒ_concept` and `ℒ_norm` to keep it on manifold |

#### Recommendation

**Strong baseline / lower bound.** If this works, the proposed method is
basically "fancy projector training" and the latent-reasoning framing is
weak. If it *doesn't* work, B.1 / B.3 / B.4 with a richer generator have
something extra to prove. **Run as ablation, not as primary.**

### B.3 Q-Former / cross-attention emitter

ASCII diagram (reproducing BLIP-2's design with our K, D):
```
    image
      ↓
    vision encoder (ViT-L/14 or Qwen2.5-VL's vision tower)   ← frozen
      ↓
    T_v × D_vit features
      ↓
    ┌────────────────────────────────────┐
    │    Q-Former (12 transformer layers, BLIP-2-style)   │
    │    K learnable queries           ← trainable        │
    │    self-attn between queries (every layer)          │
    │    cross-attn queries → vision  (every other layer) │
    └────────────────────────────────────┘
      ↓
    h_{1:K} ∈ ℝ^{K × D}  (linear project D_qformer → D=3584)
```

- K=32 was BLIP-2's default; we'd use K=16 to match round-3.
- 12 layers, BERT-base init. Cross-attention every other block. ~188 M
  params total. Vision encoder frozen. (Verbatim verified arXiv:2301.12597
  HTML this session: "32 queries, 768 dim", "188M parameters", "every
  other transformer block".)

#### Spec

| Attr | Value |
|---|---|
| Trainable | Q-Former (~188 M) plus output linear (~3 M for D_qformer=768 → D=3584). Vision encoder frozen. |
| Compute / fwd | Vision encoder fwd + 12-layer Q-Former at seq_len = K = 16. ~5–10× more than B.2 head, but ≪ B.1 full VLM forward. |
| Param count | ~190 M |
| Prior art | **BLIP-2 (arXiv:2301.12597)** — exactly this design. **InstructBLIP (arXiv:2305.06500)** instruction-extends. |
| Pro vs shortcut | (i) Mid-capacity — more expressive than B.2 head, less expressive (and more disciplined) than B.1 full VLM; (ii) cross-attention to vision features makes "what visual content goes into h" explicit and inspectable (attention maps); (iii) BLIP-2 has 3 years of community engineering — solid recipes available; (iv) q-invariance trivially baked in (Q-Former takes only image features, no q input) |
| Con vs shortcut | (i) D_qformer = 768 ≠ D_reader = 3584 — needs an output projection that may add its own pathologies; (ii) initialization is from BERT, not from any vision-language pretraining at the right scale — cold-start may be slow; (iii) BLIP-2 had its own grounding objectives (ITC + ITM + ITG) that we'd need to either replicate or skip; (iv) two-stage BLIP-2 training is heavy |

#### Recommendation

**Best bet for proliferated training**, especially if we're aiming for a
publishable "reader-grounded latent" method. The Q-Former is the most
*identifiable* generator architecture in the literature and the easiest to
position against (Monet, Mirage, LVR all chose less standard emitter
designs). For round-3 POC: probably skip — too much engineering overhead
for a feasibility round. For proliferated: **strong candidate**.

### B.4 Hybrid: VLM forward + extracted visual-token subset + MLP

ASCII diagram:
```
    image
      ↓
    Qwen2.5-VL-7B's full forward (vision tower + projector + merger; frozen)
      ↓
    T_v post-merger visual tokens
      ↓
    pick K tokens via [strategy: avg pool / top-K by saliency / fixed indices]
      ↓
    K × D selected tokens
      ↓
    small MLP per slot (trainable)
      ↓
    h_{1:K}
```

- Take K natural visual tokens (post-merger, dim D=3584), MLP-transform.
- "Strategy" can be: first K tokens (lazy), evenly-spaced K tokens (LIVR-
  style), top-K by attention from a learned salience query, or learned
  K-attention pool.

#### Spec

| Attr | Value |
|---|---|
| Trainable | MLP per slot (~10–50 M depending on hidden width); optionally K salience queries (K · D = ~57 K params) |
| Compute / fwd | Vision tower forward (frozen) + MLP. Same cost order as B.2. |
| Param count | ~10–50 M trainable / 600 M frozen vision tower |
| Prior art | **LVR (arXiv:2509.24251)** uses exactly this with bbox-based selection: "Based on the ROI bounding box, LVR efficiently selects the corresponding patches and retrieves their indices I={I₁,…,I_T_v}". Bbox supervision differs but the selection-then-MLP pattern is identical. |
| Pro vs shortcut | (i) `h` starts *literally on the visual-token manifold* (it IS a transformed natural visual token), so off-manifold drift is gated by how aggressive the MLP is; (ii) inductive bias toward visual content is maximal — the MLP cannot easily encode anything other than what's in the source visual token; (iii) no q-input → q-invariance free |
| Con vs shortcut | (i) Capacity is so constrained that "reasoning"-flavored composition across positions is limited — h₁ depends on visual_token_1, h₂ on visual_token_2 (or whatever the selection); (ii) selection strategy is now a hyperparameter that affects what `h` represents; (iii) if the MLP is identity, this collapses to B.2's "just use natural visual tokens" — needs to actually do something |

#### Recommendation

**Strongest inductive bias toward visual content; weakest expressive
power.** Run as a control: if B.4 reaches comparable held-out NLL to B.1
under round-3's recipe, the gains are mostly attributable to *which
tokens go into h* rather than to any latent reasoning. That would be a
critical interpretability finding — the method reduces to visual token
selection. **Run as ablation in proliferated training.**

### B.5 Recommendation table

| Architecture | Round-3 POC | Proliferated | Rationale |
|---|---|---|---|
| **B.1 Same-VLM + special tokens (LIVR-style)** | **Primary** | **Primary or secondary** | Closest published precedent (LIVR) for exact failure mode; LIVR's t-SNE finding (latents on manifold) is the proof-of-concept we need to replicate. LoRA-rank-32 + LIVR-style attention masking. |
| B.2 VLM + readout head | Ablation | Lower bound | Cheapest, weakest. Tells us how much of the gain depends on having a real LM in the loop. |
| B.3 Q-Former | Skip | **Primary alternative** | Best-engineered, most positionable. Heavy for POC but the right architecture for a paper. |
| B.4 Hybrid (extracted + MLP) | Ablation | Ablation | Maximally vision-grounded by construction; tests whether the latent reasoning framing adds anything beyond token selection. |

**Round-3 POC architecture: B.1 with LoRA-r=32 and LIVR-style Stage-1
attention masking.**

**Proliferated project architecture: B.1 (extends round-3 directly) +
optional B.3 ablation as paper section "we also ran a Q-Former baseline."**

---

## Part C — Multi-reader logistics

The §A.2 multi-reader NLL is structurally load-bearing — it's the only
component round-2 didn't already test in some form. R = 2 is the minimum
that distinguishes "reader-specific" from "shared structure"; R = 3+
helps the structural pressure compound.

### C.1 Compatibility requirement

For R readers to share the same `h` injected at visual-token positions,
they must share:

1. **Hidden dim D.** Same backbone → same D. Different VLM family →
   different D, would need per-reader projection adapter.
2. **Vision-token convention** (post-merger token shape, position-encoding
   scheme, M-RoPE handling, special-token IDs for `<|image_pad|>`).
3. **Tokenizer** for the question / answer side. Doesn't affect h
   directly but affects the NLL computation (need consistent y* tokenization).

Same backbone fine-tunes (LoRA, full SFT, RL fine-tunes) preserve all
three. **Cross-arch readers do not.**

### C.2 Surveyed sibling fine-tunes (Qwen2.5-VL-7B, all compatible)

Verified via HuggingFace search this session:

| Reader | HF id | Type | Source |
|---|---|---|---|
| **φ₁** Qwen2.5-VL-7B-Instruct | Qwen/Qwen2.5-VL-7B-Instruct | base instruct | already on disk |
| **φ₂** Monet-7B | NOVAglow646/Monet-7B | distillation-based latent visual reasoning fine-tune | already on disk (sibling experiment) |
| **φ₃** VL-Rethinker-7B | TIGER-AI-Lab/VL-Rethinker (NeurIPS'25) | RL fine-tune (SSR + Forced Rethinking) of Qwen2.5-VL-7B-Instruct | ~14 GB download |
| φ₄ Rex-Thinker-GRPO-7B | IDEA-Research/Rex-Thinker-GRPO-7B | GRPO RL fine-tune (zero-shot detection) | ~14 GB download (narrow domain) |
| φ₅ PixelReasoner-WarmStart | TIGER-Lab/PixelReasoner-WarmStart | reasoning fine-tune | ~14 GB download |
| φ₆ Cran-May/Qwen2.5-VL-7B-Instruct-GRPO-3DSRBench-small | small RL fine-tune for 3D spatial reasoning | (small adapter) |
| φ₇ uAI-NEXUS-MedVLM-1.0a-7B-RL | UII-AI/uAI-NEXUS-MedVLM-1.0a-7B-RL | medical-domain RL fine-tune | ~14 GB (out-of-domain for GQA) |
| LoRA-only adapters | various (llavallava/, adriving/, ronantakizawa/) | task-specific LoRA on Qwen2.5-VL-7B-Instruct | ~MB each |

**Round-3 POC R=2 picks:** φ₁ + φ₂ (already on disk).
**Round-3 POC R=3 stretch:** add φ₃ (VL-Rethinker — broad reasoning RL,
matches our reasoning-content interest).
**Proliferated R=4–6:** add φ₄, φ₅, plus 2–3 LoRA-merged variants per §C.3.

### C.3 Layer-mixing / LoRA-merged readers as cheap reader augmentation

Question: can we generate reader diversity by *interpolating* between
base + LoRA at random α?

Mechanism: Qwen2.5-VL-7B + (any compatible LoRA adapter) → merged weights
W_merged = W_base + α · ΔW_LoRA. Different α → different reader.

**Theoretical justification:** Linear Mode Connectivity (LMC) — for models
with identical architecture and initialization, the loss between
checkpoints connects via low-loss linear paths (Frankle et al. 2020,
arXiv:1912.05671). LoRA-merged variants at α ∈ [0.5, 1.5] are guaranteed
to be on this low-loss curve.

**Pros:**
- Free reader diversity: 5 LoRAs × 3 α values = 15 readers from one base.
- Each merge costs only the LoRA ΔW computation (~minutes), no GPU memory
  beyond the merged weights.
- The visual encoder + projector + merger remain unchanged → vision-token
  convention strictly preserved.

**Cons:**
- LoRA fine-tunes are typically narrow-domain — α-blending two narrow
  experts may produce non-natural intermediate readers. Different from "a
  real fine-tune trained holistically on a different objective."
- Multi-reader NLL with R highly correlated readers (same base, small ΔW)
  may not provide the structural pressure we want. **The diversity has to
  be *real*** to force `h` away from φ₁-specific shortcuts.

**Validity:** layer-mixing IS a valid form of reader-augmentation, but it
must be combined with at least one *genuinely different* fine-tune (e.g.,
Monet-7B which underwent a 3-stage SFT pipeline) for the structural
pressure to bite. Pure LoRA-merge readers are too close to φ₁.

**Recommended use:** R = {2 genuine fine-tunes (φ₁ + φ₂)} + {3–5 LoRA-
merged variants for additional diversity}. The LoRA-merged readers
contribute marginal pressure; the genuine fine-tunes contribute the bulk.

### C.4 Cross-architecture readers — feasible but not in round-3

**Different VLM family** (LLaVA-1.6, InternVL, Pixtral, etc.) requires:
- Per-reader output adapter h → h_i (linear projection D → D_i) since hidden
  dims differ.
- Different vision-token convention (different position-encoding, no
  M-RoPE in some, different special tokens).
- Per-reader prompt template.

This is doable — would generalize the generator to emit a "universal h"
that all readers project to their own space — but **adds a learned
adapter per reader**, breaking the "fully frozen reader" invariant.
**Skip for round-3; revisit for paper-scale work** as a stress test of
how transfer extends to truly different reader families.

---

## Part D — Round-3 POC implementation plan & checklist

### D.1 Phases

| Phase | Wall-clock | Output |
|---|---|---|
| 1. Data prep — extend GQA samples to per-image 3-question lists | half-day | `data/poc1_samples_multiq.jsonl` |
| 2. V_sem cache — precompute `φ₁.visual(x).hidden_states[-1]` for all training images | 1 hour on 1 GPU | `data/v_sem_cache.pt` |
| 3. Round-3 trainer — implement B.1 + §A.6 loss in `train_round3.py` | 1 day | `train_round3.py`, smoke test |
| 4. Round-3 sweep — λ_concept × λ_norm × R × K_q | overnight (4× H100) | `results/20260502-*_round3_sweep/` |
| 5. Random-control ablations (§A.5) — 3 control modes × best round-3 cell | overnight | `results/20260502-*_round3_controls/` |
| 6. Round-3 evaluation — POC-2 held-out q + POC-3 reader transfer + steering | half-day | extend `analyze.py` |
| **Total** | **~4 days** | round-3 outcome |

### D.2 Pass/fail decision tree

```
After round-3:
├── POC-2 held-out NLL < 2.0 nat (strong) — proceed to proliferated training
├── 2.0 ≤ POC-2 < 2.5 nat AND random-control gain > 50% of real gain
│       → grounding signal weak, gain mostly regularization. Pivot or kill.
├── POC-3 transfer drop ≤ 2 nat AND random-control transfer also high
│       → pseudo-transfer (regularization, not visual content). Pivot.
├── POC-3 transfer drop > 4 nat (round-2 baseline persists)
│       → multi-reader NLL didn't fix transfer. Add R, or B.3 Q-Former, or pivot to LIVR-style single-model.
└── Steering probe: round-2's positive result holds (h causally functional)
        → continue. Otherwise: investigate why combined loss broke causality.
```

### D.3 Open risks for round-3

1. **`ℒ_concept` collapses `h` to be identical to natural visual tokens.**
   If the MLP is too narrow and the cosine target is dominant, optimizer
   may set `h ≈ V_sem` (just learn the identity-via-MLP). Then the method
   degenerates to "use visual tokens as soft prompts" with no
   reasoning-flavored content. Mitigation: ensure MLP has bottleneck (D →
   D/2 → D) so identity isn't a single-step solution; monitor cos(h, V_sem)
   — should plateau around 0.7–0.9, not 1.0.

2. **Multi-reader R=2 is too small.** Round-2's reader-transfer drop was
   measured against one φ₂; pressure from that one reader during training
   may not generalize to a held-out φ₃. **Held-out reader test:** train
   under {φ₁, φ₂}, eval under φ₃ (VL-Rethinker). If transfer drop on φ₃
   ≈ within-trained transfer to φ₂, structural fix worked. If not, R needs
   to grow.

3. **Curriculum gating doesn't apply cleanly.** LaViT gates visual access
   inside the latent emitter; our generator already has full image access.
   Our analog (gate `ℒ_NLL_reader` weight from 0.1→1.0 over warmup) is
   well-defined but is *not* the same mechanism; whether it provides the
   same shortcut-prevention effect is empirical.

4. **`ℒ_concept` with k' = k mod T_v is crude.** A learned attention pool
   from `T_v` features to K targets would be cleaner. For round-3 POC,
   crude is fine; if `ℒ_concept` matters and crude pooling underperforms,
   add learned pool in proliferated.

5. **VRAM at R=3 + B=4.** May need gradient checkpointing on readers
   (slows 1.5× but halves activation memory). Run R=2 first as VRAM
   stress test.

---

## Part E — Citations (verified this session)

All arXiv IDs verified by HTML or abstract WebFetch in this design
session unless flagged.

- **arXiv:2601.10129** — LaViT — Aligning Latent Visual Thoughts.
  Verified `ℒ_total = ℒ_ntp + λ·(ℒ_concept + ℒ_traj)`, λ=0.3, teacher =
  Qwen2.5-VL-32B post-attn last-layer features, curriculum γ(t) cosine
  ramp T_w=400, Top-K=8 sparsified attention.
- **arXiv:2512.21218** — LIVR. Verified K=16 default (sweep {4,8,16,32}),
  K new special tokens with unfrozen embedding-table rows, two-stage
  training (4 epochs Stage 1 attention-masked + 6 epochs Stage 2
  unmasked), NLL-only loss on answer tokens, t-SNE shows latents on
  visual manifold.
- **arXiv:2511.21395** — Monet. Verified Stage 2 `ℒ_align-obs` with α=2.0,
  Stage 3 `ℒ_align-latent` with β=2.0, last-layer hidden state fed back,
  σ=10.0 in VLPO.
- **arXiv:2412.06769** — Coconut. Verified `<bot>...<eot>` markers,
  curriculum stages with c latents per reasoning step, NLL loss masked on
  questions and latent thoughts (i.e., NLL only on answer tokens).
- **arXiv:2301.12597** — BLIP-2 / Q-Former. Verified 32 queries (768-d),
  188M params, 12 layers (BERT-base init), cross-attention every other
  block, frozen ViT-L or ViT-g, ITC+ITM+ITG Stage-1 + LM Stage-2.
- **arXiv:2509.24251** — LVR. From `LITERATURE_RECON.md` and
  `LITERATURE_MITIGATIONS.md`. Verified MSE loss `‖h_t − v_t‖₂²` to ROI
  patch embeddings.
- **arXiv:2602.22766** — Li et al., latent CoT critique. Already in
  `LITERATURE_RECON.md` §1.2.
- **arXiv:2512.21711** — "Do Latent Tokens Think?" — already in
  `LITERATURE_RECON.md` §2.1. Round-2 steering probe inverted its finding
  (our latents are causally functional, not placeholder-like).
- **arXiv:2004.05704** — "Visual Grounding Methods for VQA Are Working
  for the Wrong Reasons!" — random-control mandate in §A.5.
- **arXiv:1912.05671** — Frankle et al., Linear Mode Connectivity —
  theoretical basis for §C.3 LoRA-merged reader augmentation. Not
  re-verified this session; cited from training data; flagging.

---

## TL;DR

- **Round-3 loss:** `ℒ = ℒ_NLL_multi(R=2 φ's, K_q=3 q's, q-invariant
  generator) + 0.3 · ℒ_concept(cos to φ₁ post-merger visual tokens via
  bottleneck MLP) + 0.1 · ℒ_norm(target μ=57.86)`. Curriculum: ramp
  λ_norm and `ℒ_NLL_reader` weight over 200-step warmup, anneal
  λ_concept at end.
- **Round-3 architecture (B.1):** Same Qwen2.5-VL-7B-Instruct as φ₁, with
  K=16 special `<|latent|>` tokens prepended to image-pad block, LoRA
  r=32, LIVR-style Stage-1 attention masking (answer tokens cannot attend
  to image, only to h and q).
- **Multi-reader R=2:** φ₁ Qwen2.5-VL-7B-Instruct + φ₂ Monet-7B (both on
  disk). R=3 stretch: add VL-Rethinker-7B. R=4+: LoRA-merge variants for
  cheap diversity once multi-reader proven viable.
- **Random-control mandate:** every round-3 cell rerun with shuffled
  (image, q) pairs; gain on real ≥ 2× gain on shuffled, else pivot.
- **Compute:** fits 4× H100 with B=4, K=16, R=2; ~8s/step, 1000-step
  POC ≈ 2.2 hours; 5-cell sweep overnight.
- **For proliferated:** B.1 still primary; **add B.3 Q-Former as ablation
  section in the paper.** Multi-reader R≥3 mandatory for the headline
  transfer claim.
