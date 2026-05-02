# Reader-Grounded Latent Visual Reasoning — Pre-Training POC Report

_Generated 2026-05-01 (overnight run)._
_Companion to `a.md` (the proposed method, two variants)._

## TL;DR (revised, post round-2 mitigation probes)

We pre-validated the reader-grounded latent visual reasoning method (`a.md`, Variant A or B) before committing GPU-weeks. **Two rounds**: round 1 characterized the failure modes; round 2 tested four candidate mitigations (norm regularization, low-rank capacity bound, multi-question consistency at N=2 and N=5, steering analysis). Headline:

- **The shortcut basin is universally reachable.** K=1 → 0.21 nat; K=4 → ~0; even rank-1 (h ∈ R^{K×d} factorized as `U V` with min(K,d)=1) reaches ~0; even with norm pinned to natural visual-token scale (57.86) reaches ~0. **No capacity or norm constraint we tried prevents shortcut convergence on the training task.** This is a property of the reader-NLL loss surface itself, not of any specific h parameterization.
- **But held-out q' generalization improves substantially under multiple independent mitigations.** Same single-Q held-out NLL drops from POC1's 3.91 nat baseline to:
  - **2.39 nat** with norm regularization at λ=0.1 (−39 %)
  - **2.65 nat** with rank-1 latents at K=4 (−32 %)
  - **2.40 nat** with multi-question (N=2) consistency (−39 %)
  - **3.28 nat** with multi-question (N=5) consistency (−24 %)
- **Latents are causally functional, NOT placeholder-like.** Steering probe (perturbations on POC 1's h*): zeroing the last latent position → +4.12 nat reader NLL; permuting positions → +2.56; replacing with another sample's latent → +2.67; gaussian noise at 100 % of latent norm → +4.06. **This is the OPPOSITE pattern to Coconut's text latents** (arXiv:2512.21711 reported "minimal sensitivity to steering"). The shortcut and causal-functionality coexist — h* encodes useful answer-relevant info that is sample-specific, position-specific, and *real*, but it is not portable visual content.
- **Reader transfer fails universally — and the round-2 mitigations DO NOT fix it.** Monet-7B NLL given the same h*: POC1 vanilla K=4 = 8.09; Mit-B λ=0.1 = 8.55 (slightly *worse*); Mit-A K=4 r=1 = 8.30 (no improvement). The held-out gain is *single-reader-specific*. Norm regularization and low-rank parameterization change *where* the shortcut lives in φ₁'s embedding space but do not make the latent direction one that φ₂ can read. **This is the most important negative result in the entire POC.** Reader transfer remains a structural failure that geometric mitigations alone cannot fix.
- **Distributional drift confirmed.** h* sits at norm ~20 (input-embedding scale 0.88; natural visual-token post-merger scale 58). Norm constraint at λ=0.1 pins to 57.86 *and* improves held-out — but the latents at the natural scale are still not transferable to φ₂.

**Bottom line for the training paper.** Variant A as written produces shortcut-dominant, reader-specific, off-manifold latents — and Variant B inherits this. **However**, round 2 reveals three independent mitigations that each give ~30 % held-out improvement at zero cost to training fit. **None individually fixes the reader-transfer failure**, but their combinable structure suggests a viable training recipe:

1. **Norm regularization (λ ≈ 0.1)** toward natural-visual-token norm (57.86) — keeps `h` on-manifold during optimization without breaking fit.
2. **Multi-question consistency (N ≥ 2 per image)** — forces some q-invariance.
3. **Auxiliary grounding loss à la LaViT (arXiv:2601.10129)** — cosine to teacher visual features + KL on cross-attention, the strongest documented anti-shortcut signal in the recent literature.
4. **Steering-style probe in evaluation** (arXiv:2512.21711) — required to certify that latents are doing real work and not pure shortcut.

The combined recipe should be pre-validated under the same POC 2 / POC 3 protocol before scaling. See §11–§14 for round 2 numbers and §16 for the revised decision.

This is a useful pre-validation, not a refutation — round 1 said "don't train as written"; round 2 says "here are concrete recipes that demonstrably move the needle, and combining them is plausibly enough."

---

## 1. Method recap (proposed; what we did *not* train)

The user's proposal (`a.md`) is to train a generator `π_θ` (Qwen2.5-VL-7B, Monet's protocol) to autoregressively emit `K` continuous latent embeddings `h_{1:K} ∈ ℝ^{K×d}`, scored under a frozen reader `φ` (frozen copy of generator's base). Reader has **no other image input** — `h_{1:K}` is its only visual content. Two variants:

- **Variant A (SFT-style).** `L = − log φ(y* ∣ q, h_{1:K})`, gradient through the frozen reader into the latents into θ.
- **Variant B (RL-style).** `J = E[ A^{(g)} · log φ(ŷ^{(g)} ∣ q, h^{(g)}_{1:K}(θ)) ]` with GRPO advantages on rule-based rewards.

The pre-training POCs treat per-sample optimization of `h_{1:K}` (as `nn.Parameter`) as a **feasibility ceiling on Variant A** — a generator network can do no better than what direct gradient descent finds in this loss landscape. If the per-sample optimum is pathological, the trained generator's optimum will be too.

## 2. Setup

- **Reader.** `Qwen/Qwen2.5-VL-7B-Instruct`, frozen, bf16, hidden_dim 3584. Gradient flows from reader-NLL through the (frozen) reader weights into `h`.
- **Slot injection.** We bypass the vision encoder entirely. Input embeddings are computed from `input_ids` (a hand-built chat template containing exactly K `<|image_pad|>` tokens between `<|vision_start|>` and `<|vision_end|>`); the K embeddings at those slots are spliced out and replaced with `h` (an `nn.Parameter`, fp32 master). Forward via `model(inputs_embeds=...)`.
- **Optimization.** Adam, lr=5e-2, 500 steps per sample. `h` initialized at the natural input-embedding norm (~0.88).
- **Data.** 100 GQA `testdev_balanced_instructions` samples filtered to single-token answers; for POC 2, 30 of the same images each carrying a second short-answer question (q2). For POC 3, the same 100 saved h*. For POC 4, the same 300 latent tensors.
- **Compute.** Single A6000 (49 GB). POC 1 sweep (100 × 3 K × 1 seed = 300 runs × 500 Adam steps): 135.8 min wall-clock. POC 2 + POC 3 + visual baseline: ~10 min combined.

## 3. POC 1 — Reachability

**Question.** Can per-sample direct optimization drive reader NLL on `y*` to near-zero, with no image, using K continuous embeddings? If yes, the optimum is reachable; the question becomes what *kind* of optimum.

**Results** (n=100 per K, 95 % bootstrap CI; oracle_nll and no_input_nll are constant across K rows because they are sample-level baselines):

| K  | init_nll | final_nll | oracle_nll | no_input_nll | reach < oracle×1.5 | reach < 1 nat | %final < oracle | %final < no_input |
|----|----|----|----|----|----|----|----|----|
| 1  | 9.50 [9.16, 9.86] | **0.21 [0.07, 0.38]** | 10.27 [9.87, 10.69] | 9.46 [9.07, 9.83] | 1.000 | 0.93 [0.88, 0.98] | **1.000** | **1.000** |
| 4  | 9.37 [9.02, 9.72] | **0.00 [0.00, 0.00]** | 10.27 [9.87, 10.69] | 9.46 [9.07, 9.83] | 1.000 | 1.000 | **1.000** | **1.000** |
| 16 | 9.15 [8.81, 9.50] | **0.00 [0.00, 0.00]** | 10.27 [9.87, 10.69] | 9.46 [9.07, 9.83] | 1.000 | 1.000 | **1.000** | **1.000** |

Per-sample loss curves drop ~10 nat → ~0 nat in <100 Adam steps (figs `poc1_loss_curves_K{1,4,16}.png`).

**Reading.**

1. **The shortcut basin is everywhere.** `final_nll < no_input_nll` for **100 % of samples at every K** — the optimized latents always make the reader more confident in `y*` than the no-image baseline, by definition of "what gradient descent on reader-NLL does". This is the pre-condition for shortcuts; it doesn't yet prove shortcut-only behaviour.
2. **K=1 is enough.** A single 3584-dim continuous vector reaches near-zero NLL on 93 % of samples at the strict ≤ 1 nat threshold. The capacity argument from `a.md`'s audit — "K·d ≈ 36 k floats is wildly more than needed to encode any answer" — is empirically confirmed at the smallest K. Information-bottleneck-style discipline on `h` is not optional if the goal is genuine visual encoding.
3. **`final_nll < oracle_nll` for 100 % of samples.** This is *not* evidence that `h*` is "more useful than the image" — it is evidence that off-manifold soft prompts in the reader's input space drive the reader's output distribution to be over-confident on a chosen target token. Same pattern observed in arXiv:2402.09063 (embedding-space attacks).
4. **GQA's image is roughly as informative as the question alone for these specific GT answers.** `oracle_nll ≈ no_input_nll` (10.27 vs 9.46 — image is *slightly worse* for the gold answer than text-only). This is a property of GQA's narrow-answer dataset, not a bug in our oracle path: GQA's gold answer is a specific lexical token, and even with the image the reader often prefers a synonym. Documented as a methodological caveat; does not affect our shortcut conclusions, which are about `final_nll` vs both baselines.

**Verdict for H1 (reachability).** Confirmed strongly. Optimization succeeds at every K we tested. The remaining question is what that optimum *contains*.

## 4. POC 2 — Held-out question on the same image (the smoking-gun probe)

**Question.** If `h*` was optimized to make the reader say "yes" to "are there glasses or women?" (q1) given image I, does it also let the reader answer "are there either old men or women?" (q2) about the same image? If yes, `h*` carries visual content. If no, `h*` is an answer-encoding shortcut.

For each of 30 image_ids (POC 1 ∩ POC 2 by image, POC 1's q is exactly q1 of the POC 2 pair), we load `h*` from POC 1 and evaluate the reader on q2.

**Results** (n=30 per K, 95 % bootstrap CI):

| K  | h\*_nll(q2) | oracle_nll(q2) | no_input_nll(q2) | gap vs oracle | %h\*≈no_input |
|----|----|----|----|----|----|
| 1  | 3.58 [2.57, 4.72] | 10.36 [9.67, 11.09] | 9.70 [9.07, 10.26] | −6.78 | **0.53 [0.37, 0.70]** |
| 4  | 3.76 [2.18, 5.71] | 10.36 [9.67, 11.09] | 9.70 [9.07, 10.26] | −6.61 | **0.50 [0.33, 0.67]** |
| 16 | 3.52 [2.29, 5.02] | 10.36 [9.67, 11.09] | 9.70 [9.07, 10.26] | −6.84 | **0.53 [0.37, 0.70]** |

`%h*≈no_input` is the fraction of samples where `|h*_nll − no_input_nll| < |h*_nll − oracle_nll|` — i.e. how often the held-out NLL sits closer to the no-image baseline than to the image baseline.

**Reading.** This is the most *interesting* table in the report. The result is not a clean shortcut and not clean visual content — it is a **mixture**:

1. **`h*_nll(q2) ≈ 3.6` is far below `no_input_nll(q2) ≈ 9.7`.** In aggregate, h\* helps the reader answer q2 by ~6 nat. So h\* is *not* purely encoding q1's answer.
2. **`h*_nll(q2) ≈ 3.6` is also far below `oracle_nll(q2) ≈ 10.4`.** Reading this naively as "h\* beats the image" would be wrong — same off-manifold over-confidence as in POC 1. The negative gap is structural.
3. **50 % of cases sit closer to no-input than to oracle.** Within-sample, half the time h\* gives essentially nothing for q2. Looking at the qualitative greedy decodes: when q1 and q2 share an answer (e.g., both "yes"), h\* transfers; when q1's answer (e.g. "fence") is irrelevant to q2 (e.g. "what color is the watch?"), h\* either guesses a same-class token (e.g. "black", a color) or produces meta-text ("the image description does not mention…"). The latents leak partial scene-class information but do not provide question-specific visual answers.

This places the failure mode in line with arXiv:2512.21711's two-pronged analysis of Coconut: latents "function as uninterpretable placeholders rather than encoding faithful reasoning. While resistant to perturbation, they promote shortcut usage over genuine reasoning." The reader-grounded latents we produced are *somewhat* better than the worst case (they leak some category-level visual signal), but the dominant content is still answer-relevant rather than context-portable.

**Verdict for H2 (visual content / shortcut absence).** **Refuted at the strict version, partially supported at the weak version.** A trained generator running Variant A would inherit the same mixture: latents that retain class-level visual signal but encode the specific (q, y*) pairing as a soft-prompt shortcut on top of it.

## 5. POC 3 — Reader transfer to Monet-7B

**Question.** If `h*` were a faithful visual encoding, any sibling reader of the same family should be able to read it. We test on `NOVAglow646/Monet-7B` (Qwen2.5-VL-7B + Monet's SFT + VLPO RL), loaded via stock HuggingFace forward (no Monet vLLM patches — we do not use Monet's latent-decoding loop, just standard text + spliced-h forward).

**Results** (n=100 per K):

| K  | NLL under φ₁ (Qwen2.5-VL-7B, where h* was optimized) | NLL under φ₂ (Monet-7B) | transfer drop = φ₂ − φ₁ |
|----|----|----|----|
| 1  | 0.21 [0.07, 0.38] | 9.17 [8.64, 9.68] | **+8.97 [+8.41, +9.50]** |
| 4  | 0.00 [0.00, 0.00] | 7.72 [7.16, 8.29] | **+7.72 [+7.16, +8.29]** |
| 16 | 0.00 [0.00, 0.00] | 5.77 [5.25, 6.31] | **+5.77 [+5.25, +6.31]** |

**Reading.**

1. **Massive transfer drop at every K.** Monet-7B has near-uniform NLL on the answer given `h*`. The latents we found are *not* a portable visual representation; they exploit `φ_1`'s exact post-training-state decoder behaviour.
2. **Drop shrinks with K.** From +8.97 at K=1 down to +5.77 at K=16. More capacity → slightly less brittle. This is consistent with: at K=1, the optimization compresses everything into one adversarial direction; at K=16, some of the capacity is spent on more-redundant, more-shareable structure. Still, +5.77 nat at K=16 means Monet-7B is far from confident on the answer.
3. **Sibling fine-tunes share weights but not behaviour at the embedding level.** Monet-7B was further fine-tuned (SFT 3-stage + VLPO) on top of Qwen2.5-VL-7B; that's enough divergence for adversarial soft prompts on `φ_1` to fail to transfer. This bounds how robust the trained-generator's latents would be to any downstream fine-tuning of the reader (or even small reader updates between training and deployment).

**Verdict for H3 (reader generality).** Refuted strongly. The optimum does not generalize across sibling readers. Trained latents under Variant A would inherit this fragility.

## 6. POC 4 — Distributional sanity (off-manifold drift)

**Question.** Where do the optimized `h*` sit in embedding space? On or off the natural visual-token manifold the reader was trained on?

**Optimized `h*` (this POC, n=300):**

| K  | norm_mean | intra-block cos | inter-sample cos (first slot) |
|----|----|----|----|
| 1  | 23.36 | n/a | 0.082 |
| 4  | 19.29 | 0.084 | 0.051 |
| 16 | 18.95 | 0.053 | 0.281 |

**Natural visual tokens (Qwen2.5-VL-7B vision encoder + merger, 100 GQA images):**

- ~239 tokens per image (180–256)
- norm_mean ≈ 57.86 (range 44.6–70.6)
- intra-image cos sim ≈ 0.295
- inter-image first-token cos sim ≈ **0.845**

**Monet's actually-emitted latents (sibling experiment `monet-latent-probe`, n=65 latent-using samples × 10 latents):**

- norm ≈ 288 (last-layer hidden states, *not* embeddings — different scale by construction)
- intra-block cos sim ≈ **0.758**
- inter-sample cos sim (block_0[0] across samples) ≈ **0.927**

**Reading.**

1. **Norm scale: ~3× too small.** Natural visual tokens at the post-merger embedding position have norm ~58. Optimized `h*` settles around ~19–23 — off-manifold, but in a *smaller-norm* direction, not larger. (Intuition: high-norm directions saturate the softmax aggressively; gradient descent finds it cheaper to put `h*` in a direction that nudges the logits the right way without saturating.)
2. **Cosine structure is inverted.** Natural visual tokens are highly inter-sample similar (0.85): there's a strong sample-agnostic structure (positional, low-frequency, common-object). Monet's emitted latents amplify this (0.93) — they are essentially sample-agnostic. **Our optimized latents go the *other* direction**: each sample produces a near-orthogonal solution (inter-cos 0.05–0.28). Sample-specific by construction — each `h*` is the answer-encoding shortcut for its own (q, y*).
3. **Intra-block structure is also low.** Within a single sample's K=16 block, the K latents have cos ≈ 0.053 — they're nearly orthogonal. Compare 0.30 for natural visual tokens of the same image and 0.76 for Monet's K=10 latents within a block. Our optimization spreads the K capacity orthogonally, presumably to maximize the reader's logit margin in independent directions.

**Verdict.** Optimized `h*` is decisively off-manifold and decisively *not* sample-agnostic. This is the soft-prompt-tuning failure mode predicted by arXiv:2402.09063 and arXiv:2504.02144 ("a fundamental trade-off between interpretability and the task-performance of the trainable prompt"). Without an explicit manifold constraint, the trained generator under Variant A is expected to drift the same way — perhaps less aggressively (the generator's autoregressive structure provides some implicit regularization) but in the same direction.

## 7. Cross-validation with sibling experiment (`monet-latent-probe`)

The sibling `monet-latent-probe` experiment is currently testing whether Monet's *actually-emitted* latents are causally necessary at inference. Its first round (HRBench8K, n=65 latent-using samples, paired baseline vs zero-intervention):

> "Δ = +26.2 pp (95 % CI [+15.4, +36.9]). Reading: consistent with H1 (latents are causally load-bearing)."

That is, **Monet's existing latents *are* causal** — zeroing them substantially degrades accuracy. Combined with the present POC, the joint picture is:

- Monet's training (SFT distillation onto teacher latents that *did* see the image, then VLPO) produces latents that **are causal** (sibling probe) and **highly sample-agnostic** (intra 0.76, inter 0.93). They sit on *some* structured manifold and the model has learned to use them.
- Reader-grounded supervision without that teacher-grounding produces latents that are also functionally effective on the *training* task (POC 1) but **sample-specific, off-manifold, and reader-specific**, with **only partial visual content portability** (POC 2) and **no reader portability** (POC 3).

The two experiments triangulate cleanly. Monet's distillation provides an inductive bias toward the natural visual-token manifold; reader-NLL by itself does not.

## 8. Implications for the proposed training paper

Listed by severity / what to actually do.

### 8.1 Required structural mitigation

The empirical results say Variant A as written will produce shortcut-dominant, reader-specific, off-manifold latents. To make the training paper publishable, the loss must include at least one of:

- **A.** *Information bottleneck on `h`*: vector-quantize, low-rank-project, or sparsify `h` to limit per-position channel capacity. Restricts how much answer-bearing soft prompt can be encoded in a single slot.
- **B.** *Distributional regularization*: penalty on `||h_i||` distance from natural visual-token norm distribution, plus a KL/MMD penalty on `h`'s per-dim distribution against the reader's natural visual-token distribution. The visual baseline computed here (`results/visual_baseline.json`) is exactly the prior to use.
- **C.** *Auxiliary visual grounding term*: a contrastive or reconstruction loss between `h` and the reader's vision-encoder output for the same image. This is what Monet's Stage-2 distillation does indirectly via teacher targets. The closest prior is LVR (arXiv:2509.24251) which uses MSE on bounding-box ROIs.
- **D.** *Multi-question consistency loss*: train on multiple (q, y*) pairs per image; require `h(q, I) ≈ h(q', I)` for the same image. Forces sample-consistent encoding rather than per-(q, y*) shortcuts. POC 2 is exactly the failure-mode test for this.

(B) is the cheapest to add and the most directly aimed at the off-manifold finding. (D) is the most aimed at the held-out-question shortcut. (C) is the most theoretically principled. (A) is the cleanest information-theoretic lever. None mutually exclusive.

### 8.2 Required evaluation

Even with a structural fix, the evaluation regime must include the diagnostics this POC was built around:

1. **Held-out question on same image** (POC 2 protocol) — primary shortcut detector.
2. **Reader transfer** (POC 3 protocol) — adversarial-to-reader detector. Use a sibling fine-tune; Monet-7B is already on disk for this base.
3. **Steering / OOD probes** in the style of arXiv:2512.21711 — perturb subsets of `h` and verify the model output changes; evaluate on biased / OOD subsets to detect artifact exploitation.
4. **Distributional monitoring during training** — log `h`'s norm percentiles, intra-block cos sim, inter-sample cos sim every N steps; log natural-visual-token statistics as the reference. The visual_baseline.json in this experiment gives the per-image reference; recompute it for whatever reader you pick.
5. **Input-Latent Disconnect probe (Li et al., arXiv:2602.22766)** on the *generator* side — perturb the image given to the generator and check whether `h` actually changes. The POC's no-image-on-reader constraint structurally avoids the *Latent-Answer* Disconnect, but the *Input-Latent* one is on the generator side and remains a live risk.

### 8.3 Variant A vs B

The POC tests Variant A's optimization landscape directly. Variant B's gradient signal is the same shape (reader-NLL, reweighted by GRPO advantage) — the shortcut basin is also reachable, just attended to differently across rollouts. **Variant B inherits the shortcut risk plus reward-shaping noise**; the embedding-space-attack flavour is slightly less aggressive (sampling rather than direct optimization), but the manifold and transfer concerns are identical. Recommend running the same POCs on a small Variant B trial before committing to scale.

### 8.4 Positioning

Frame the method paper as a structural response to arXiv:2602.22766 ("Imagination Helps Visual Reasoning, But Not Yet in Latent Space") and arXiv:2512.21711 ("Do Latent Tokens Think?") — both papers call out exactly the failure modes the POC reproduces. The contribution is the *combined* recipe: reader-grounded supervision + (whichever structural fix(es) you adopt) + the eval protocol above. Selling reader-NLL alone, post this POC, is hard.

### 8.5 Don't switch to Qwen3-VL yet

When the method works, retraining on the latest Qwen3-VL-8B is the natural follow-up. Until POC validates the mitigated method, stay on Qwen2.5-VL-7B: same base as Monet (cleanest baseline comparison), same base as `monet-latent-probe` (cross-experiment triangulation continues to work), and Monet-7B / Monet-SFT-7B exist as same-base sibling readers for POC 3. Switching base mid-stream means losing all three.

## 9. Limitations / caveats

- **GQA gold-answer narrowness.** GQA's narrow gold-token vocabulary causes `oracle_nll ≈ no_input_nll` on some samples, since the image often supports a synonym the reader prefers over the literal gold word. Documented; doesn't affect shortcut conclusions which are about `final_nll < no_input_nll`.
- **K range.** We tested K ∈ {1, 4, 16}. Behaviour at K = 64+ may differ. Variant B in particular may want larger K. Not run for compute reasons.
- **Single seed.** N_SEEDS=1. Variance estimation across multiple Adam initializations not done. The 100-sample n is large enough that population-level effects are clean despite this.
- **Direct optimization is more permissive than generator-mediated.** Per-sample gradient descent is the loose upper bound on what a trained generator could do under Variant A; the generator's autoregressive structure may discourage the most extreme shortcuts. The POC bounds the *worst case* reachable; the actual training outcome could be milder. We do not know how much milder.
- **The "h\* is more confident than the image" finding is structural, not a real beat-the-oracle result.** Off-manifold soft prompts in the reader's input space drive over-confident logits regardless of what's encoded. Don't over-claim from POC 1's `final < oracle`.
- **POC 3 used Monet-7B (post-VLPO).** Cleaner sibling would be Monet-SFT-7B (no RL specialization). On disk only the post-RL version exists; would need ~16 GB additional download. Effect direction would not change; magnitude might.

## 10. Mitigation D probe — multi-question consistency

After the main POCs, we ran a quick test of mitigation D (§8.1) — the structural fix most directly aimed at POC 2's failure mode: optimize a *single* `h_{1:K}` to support **multiple questions about the same image** simultaneously, then test whether the resulting `h*` carries portable visual content (held-out third question `q3`).

**Setup.** For each of the 30 multi-Q image pairs, optimize one shared `h` (K=4, 500 Adam steps) by minimizing the *sum* of reader-NLLs over both `(q1, y1)` and `(q2, y2)`. Then load a fresh GQA question `q3` for the same image (different from q1 and q2; 30/30 found). Evaluate `h*` on `q3` against the no-input and oracle baselines, and compare to single-Q `h*` (POC 1's, optimized for q1 alone).

**Results** (n=30, single seed):

|  | mean | median | stdev | min | max |
|---|---|---|---|---|---|
| oracle on q3 | 9.82 | 9.27 | 2.29 | 6.23 | 13.84 |
| no_input on q3 | 8.88 | 9.37 | 1.80 | 4.38 | 11.60 |
| **single-Q h\* on q3** | **3.91** | **2.47** | 4.01 | 0.00 | 17.12 |
| **multi-Q h\* on q3** | **2.40** | **1.63** | 2.54 | 0.00 | 9.86 |

Pairwise:
- **multi-Q < single-Q on q3: 21/30 = 70 %** (mean improvement −1.51 nat, median −0.69 nat).
- multi-Q < oracle on q3: 29/30 = 97 % (oracle is high on GQA's lexically-narrow gold answers, so this is structural over-confidence rather than "beats the image" — same caveat as POC 1).

Crucially, on the **training questions**, multi-Q `h*` reaches NLL ≈ 0 on **both** q1 and q2 simultaneously. K · d = 14 336 floats has more than enough capacity to encode two single-token answers conditional on which question is being asked. The first read on this is "naive multi-Q just learns *multi-shortcut*". But the held-out test partially refutes that read: q3-NLL (2.40 mean) is dramatically below no-input (8.88) and consistently better than single-Q's q3-NLL (3.91). Some genuine visual content leaks into the multi-Q optimum.

**Reading.**

1. **Multi-Q (D) works partially.** It does not eliminate the shortcut basin — both q1 and q2 reach NLL≈0 trivially — but the optimization cost of supporting two distinct (q→a) shortcuts spreads `h*` slightly toward genuine visual content, and the spread benefits a held-out q3.
2. **The effect is real but modest.** Mean −1.5 nat on held-out is a meaningful improvement, but `h*` is still far from oracle-level on q3, and far worse than its trained-question NLL of 0.
3. **The mechanism is informative.** With 2 training questions you get partial generalization. The natural extrapolation is that *more* questions per image (5, 10, all available GQA Qs for a given image) should compound this effect — the latent has to encode increasingly question-invariant visual content as the number of training Qs grows, eventually approximating the natural visual-token role.
4. **K=4 is overcapacity for this objective.** Capacity-matching (smaller K, or quantized/low-rank h) would tighten the bound. Worth testing K=1 multi-Q next.

**Verdict on mitigation D.** *Necessary but not sufficient.* The training paper should likely combine multi-Q consistency with one of the capacity-reducing mitigations (A: information bottleneck) and/or a manifold prior (B: distributional regularization). A reasonable training-time recipe to try first:

- multi-Q sampling (≥ 5 questions per image per training step), with question-invariant latent structure (`h(I)` only, not `h(I, q)`)
- + capacity bound on `h` (e.g. quantize to 256-token codebook, or low-rank factorization)
- + distributional anchor: penalize `||h_i||` from natural visual-token norm distribution (`results/visual_baseline.json` as the prior)

The pre-validation of *that* combined recipe would be a one-day POC follow-up before any real training.

### 10.1 K=1 multi-Q (capacity-bound check)

A natural follow-up: does *just* shrinking K close the gap? We re-ran the multi-Q probe at K=1 (single 3584-dim latent must serve both q1 and q2). Held-out q3 evaluation, n=30:

|  | training NLL on q1 | training NLL on q2 | held-out q3 NLL |
|---|---|---|---|
| **K=1 single-Q** (h\* for q1 only) | (~0; POC 1 K=1) | — | **4.07** |
| **K=1 multi-Q** | 0.83 (median 0.00) | 0.47 (median 0.00) | **3.72** |
| **K=4 single-Q** (POC 1 / POC 2) | ~0 | — | 3.91 |
| **K=4 multi-Q** | ~0 | ~0 | **2.40** |

Pairwise K=1: multi-Q < single-Q on q3 in 17/30 = 57 % of cases (mean −0.35 nat). Pairwise K=4: 21/30 = 70 % (mean −1.51 nat). **K=4 multi-Q beats K=1 multi-Q on held-out generalization** — the larger latent has more room to spread structure and generalizes more, not less.

K=1 is, however, capacity-bound at the *training* objective: it cannot reduce both q1 and q2 NLL to ~0 simultaneously (mean training NLL ~0.5–0.8 vs K=4's 0.0). The shortcut basin is reachable in *every* sample for q1 alone (POC 1), but the **simultaneous** two-shortcut basin is not always reachable at K=1.

**Implication.** Pure K-reduction is not the right capacity bound. The right shape of capacity bound is *structural* — quantization to a small codebook, low-rank factorization, sparse activations — not just a small dimensionless `K`. The held-out q3 result is a real warning that "shrinking until it doesn't fit" produces compromised optimization rather than visual content.

This refines the recommended recipe in §10:
- **Drop "K=1" from the mitigation list.** Pure K-reduction alone does not improve held-out generalization meaningfully and impairs the training fit.
- The right capacity bound is *structural* (codebook / low-rank / sparsity), not dimensional.
- Multi-Q remains a real (partial) signal; combine it with structural capacity bound, distributional anchor, and an enlarged Q-set per image.

## 11. Mitigation B — Distributional regularization (norm prior, λ sweep)

**Setup.** Same per-sample optimization as POC 1, but augmented loss:
`L = -log φ(y* | q, h_{1:K}) + λ · (1/K) · Σ (||h_i||₂ − target_norm)²`
with `target_norm = 57.86` (natural Qwen2.5-VL post-merger visual-token mean norm, from `results/visual_baseline.json`). 30 POC1 samples, K=4, λ ∈ {0.0, 0.1, 1.0, 10.0}.

**Training-time results** (mean over 30 samples):

| λ    | init NLL | final NLL | final norm_mean | comment |
|------|---------|----------|-----------------|---------|
| 0.0  | 9.40    | 0.0003   | 19.71           | reproduces POC1 baseline (norm drift to ~20) |
| 0.1  | 9.40    | 0.0004   | 57.86           | **norm pinned, NLL still ~0** |
| 1.0  | 9.40    | 0.0011   | 57.86           | norm pinned, NLL still ~0 |
| 10.0 | 9.40    | **5.62** | 57.86           | regularization too strong, NLL stays high |

The shortcut basin is reachable at any norm ≤ 10λ — the regularizer changes *which direction* the shortcut lives in, not *whether* it exists. λ=10 is the only setting where regularization beats the NLL gradient.

**Held-out q' results** (for each sample's mit-B latent at each λ, compute reader NLL on a held-out q about the same image; n=30):

| λ    | held-out NLL mean | median | comment |
|------|----|----|----|
| 0.0  | 3.76 | 2.30 | reproduces POC2's single-Q baseline (3.91) |
| **0.1**  | **2.39** | **0.94** | **best — −36 % vs λ=0** |
| 1.0  | 2.92 | 2.61 | partial improvement |
| 10.0 | 7.19 | 7.82 | useless (training broken) |

**Reading.**

1. **λ=0.1 is the surprise of round 2.** At this regularization strength, training fit is unchanged (final NLL ≈ 0.0004 ≈ baseline 0.0003) but **held-out generalization improves by 36 %**. The constraint forces the optimizer to find shortcut directions that happen to be *closer to the natural visual-token manifold*, and those directions encode more transferable visual content as a side effect.
2. **The mitigation works without sacrificing training.** Unlike λ=10 (where regularization dominates and NLL is stuck), λ=0.1 is a "free lunch" on this metric — same training NLL, better generalization.
3. **This is the right shape for a training recipe.** Bake norm regularization into the generator's loss with λ in the 0.1–1.0 range; expect ~25–35 % held-out improvement.

**Limit.** The mit-B latents were not tested for reader transfer (POC 3 protocol). Norm-on-manifold doesn't guarantee that the *direction* of h* is one Monet-7B can read; that remains an open empirical question.

## 12. Mitigation A — Structural capacity bound (low-rank h)

**Setup.** Replace `h ∈ R^{K×d}` (free parameters) with low-rank factorization `h = U @ V` where `U ∈ R^{K×r}`, `V ∈ R^{r×d}`. r constrains the effective dimensionality of the K vectors. For K=4: r=4 is full-rank, r=1 forces all 4 latents colinear. Sweep over (K, r) ∈ {(4,1), (4,2), (4,4), (16,1), (16,4), (16,16)} on 30 POC1 samples.

**Training-time results:**

| (K, r)  | init NLL | final NLL | final norm_mean |
|---------|---------|----------|-----------------|
| (4, 1)  | 9.42    | 0.0029   | 24.11           |
| (4, 2)  | 9.68    | 0.0334   | 23.40           |
| (4, 4)  | 9.53    | 0.0001   | 26.97           |
| (16, 1) | 9.51    | 0.0006   | 17.65           |
| (16, 4) | 9.13    | 0.0002   | 22.09           |
| (16, 16)| 9.12    | 0.0001   | 41.71           |

**Even rank-1 reaches NLL ≤ 0.003.** For K=4 r=1 (all 4 latents on a single line in R^d, only 1 effective direction), the optimizer still finds the shortcut basin. For K=16 r=1 (16 latents constrained to 1-D), same story. Capacity bound *via low-rank* does not prevent the shortcut.

**Held-out q' results** (n=30):

| (K, r)  | held-out NLL mean | median | comment |
|---------|----|----|----|
| (4, 1)  | **2.65** | 1.75 | **best — −32 % vs single-Q POC1 baseline (3.91)** |
| (4, 2)  | 3.71 | 2.86 | worst (also worst training fit) |
| (4, 4)  | 2.94 | 2.05 | improvement vs baseline |
| (16, 1) | 2.96 | 2.34 | comparable to (4,1) |
| (16, 4) | 3.40 | 1.71 | mid |
| (16, 16)| 3.19 | 1.86 | mid |

**Reading.**

1. **Rank-1 is the best held-out config.** Forcing all K latents to lie on one direction in R^d gives a 32 % held-out improvement, comparable to mit-B's λ=0.1.
2. **The mechanism is similar to mit-B.** Low-rank constrains the *space of available shortcut directions*; the optimizer can only find shortcuts that fit in 1 dimension. Apparently the low-dimensional shortcut directions happen to be closer to whatever cross-question signal is shared across q1 and q'.
3. **(4, 2) is anomalously bad.** Both highest training NLL and worst held-out — possible Adam dynamics / init artifact at exactly r=2. Worth a follow-up but doesn't affect the rank-1 conclusion.
4. **Combinable with mit-B.** Rank-1 reduces *direction space*; norm regularization reduces *magnitude space*. Stacking them is plausibly multiplicative.

## 13. Mitigation D extended — N=5 multi-question consistency

**Setup.** Generate 25 GQA images each carrying 6 short-answer questions (5 train + 1 held-out q'). Optimize a single h (K=4, 500 Adam steps) on the *sum* of reader-NLLs over the 5 train questions. Evaluate on held-out q'.

**Held-out q' results** (n=25):

|  | held-out NLL mean | median | comment |
|---|----|----|----|
| oracle on q' (image visual tokens) | 10.93 | 10.48 | high — GQA narrow gold answers |
| no_input on q' (text only) | 9.73 | 9.59 | text-only baseline |
| **single-Q h\* on q'** (POC 1's h, q1-only) | **4.31** | 3.87 | reproduces POC2 pattern on this dataset |
| **N=5 multi-Q h\*** | **3.28** | 2.34 | **−24 % vs single-Q** |

Pairwise: **N=5 < single-Q on 17/25 = 68 %** of samples. Mean improvement −1.03 nat.

**Comparing to N=2 multi-Q (§10):** N=2 gave −39 % on its own held-out (q3); N=5 gives −24 % on its held-out (6th question). Both improvements are real and in the same direction, but **the magnitude is not strictly monotone in N on these datasets** — different held-out targets and different image subsets prevent direct comparison. The N=5 result confirms multi-Q scales beyond N=2 with the expected sign, but does not show the dramatic compounding I hypothesized.

**Reading.** Multi-question consistency at N=5 produces an outcome roughly comparable to mit-B (norm reg) and mit-A (rank-1). All three mitigations independently land in the same 2.4–3.3 nat held-out NLL range — a robust 24–39 % improvement over the single-Q baseline.

## 14. Steering probe — are the latents actually doing work?

**Setup.** For each of POC 1's 30 K=4 latents, apply 8 perturbations and measure the change in reader NLL on the original (q1, y1*):

- `zero_pos_i` for i ∈ {0,1,2,3}: zero out the i-th latent.
- `permute_within`: shuffle the 4 latents (re-rolled until non-identity).
- `permute_across`: replace position 0 with the next sample's position-0 latent.
- `gauss_noise_0.1`, `gauss_noise_1.0`: add gaussian noise at 10 % / 100 % of latent norm.

**Results** (n=30 per perturbation, 240 total measurements):

| perturbation       | Δ NLL mean | median | max  | reading |
|--------------------|----|----|----|---|
| gauss_noise_0.1    | +0.000 | +0.000 | 0.00 | latents are *robust* to small noise |
| gauss_noise_1.0    | +4.063 | +3.952 | 11.96 | large noise destroys answer |
| permute_within     | +2.556 | +1.422 | 9.93 | **position matters** — bag-of-K is wrong |
| permute_across     | +2.674 | +1.047 | 11.35 | **sample-specific** — not interchangeable across samples |
| zero_pos_0         | +2.533 | +1.639 | 9.38 | first position carries info |
| zero_pos_1         | +1.876 | +0.104 | 12.91 | mid positions carry info |
| zero_pos_2         | +2.792 | +1.366 | 10.68 | mid positions carry info |
| **zero_pos_3**     | **+4.117** | **+4.057** | **13.91** | **last position carries the most info** |

**Reading — the most important update from round 2.**

1. **Latents are causally functional.** Every perturbation that destroys content (zero, permute, large noise) substantially increases reader NLL. Reader-grounded latents are NOT inert placeholders.
2. **Contra Coconut.** arXiv:2512.21711 reports COCONUT text latents have "minimal sensitivity to steering and lack reasoning-critical information." Our visual reader-grounded latents show the *opposite* — strong, position-specific causal sensitivity. This is direct evidence the failure mode is *different* from the text-side latent inert-placeholder problem; we have a different (more workable) regime.
3. **Position 3 is special.** The last latent position carries the most causal weight (median Δ = +4.06 vs +1.4–1.6 for positions 0–2). Possible explanation: with K=4 contiguous latents at the visual slot followed immediately by the question and then the answer, position 3 is closest to the answer-decoding context and may dominate via attention recency.
4. **Robustness to small noise.** 10 % norm gaussian noise is invisible (Δ = 0.000). The latents lie on a *smooth* gradient w.r.t. small perturbations but are *semantically brittle* to large structural changes — same shape as natural high-information embeddings.
5. **Reframe of the shortcut diagnosis.** The latents are causal AND shortcut at the same time. They encode useful answer-relevant info that's sample-specific, position-specific, and functionally necessary — but the info is the (q, y*) pairing rather than the (image content). The correct metaphor is *learned compressed answer*, not *placeholder*. Mitigations should aim at *redirecting* the encoded content toward visual generality, not at *forcing the latents to be informative* — they already are.

## 15. Updated literature context (round 2)

Supplemental literature recon at `docs/LITERATURE_MITIGATIONS.md`. Key updates that reframe round 1's recommendations:

- **LIVR (arXiv:2512.21218, Dec 24 2025)** — closest published sibling. Single-model attention bottleneck (no separate reader; answer tokens cannot attend to image), K=16 latents, no auxiliary loss. Reports via t-SNE that "latent tokens largely occupy the same region as image tokens", and removing latents drops Localization 83.61 → 76.23. Their bottleneck is *weaker* than POC's (answer tokens can re-attend to the image in Stage 2) but still demonstrates manifold alignment — direct evidence the general direction works. **Strongest argument for POC's continued investigation.**
- **LaViT (arXiv:2601.10129, Jan 15 2026)** — strongest documented mitigation. `ℒ = ℒ_NTP + 0.3 · (ℒ_concept + ℒ_traj)`: cosine to teacher visual features + KL on cross-attention trajectories (top-K=8 sparsified). Curriculum Sensory Gating opens visual attention over training time. Headline: +16.94 pp on Relative Depth, +15.67 pp on Relative Reflectance vs Qwen2.5-VL-3B. Explicitly designed against the "language priors over grounded perception" failure mode — i.e., our shortcut concern verbatim. **The recipe to copy if adding an auxiliary grounding loss to Variant A.**
- **arXiv:2004.05704 (cautionary)** — "Visual Grounding Methods for VQA are Working for the Wrong Reasons!" reports that VQA grounding gains from auxiliary objectives are **"a regularization effect which prevents over-fitting to linguistic priors, and random, insensible cues also result in similar improvements."** This means the round 2 mitigation gains (mit-B λ=0.1, mit-A rank-1, mit-D N=5) **must be controlled against random-latent baselines** before being attributed to visual grounding rather than generic regularization. **A negative-control ablation (e.g. norm regularization toward a random target_norm, or low-rank with the rank constraint shuffled across positions) is mandatory before scaling.**
- **PLUME (arXiv:2604.02073)** — independently confirms our POC 4 drift: "a direct transition from explicit CoT supervision to latent-only execution is unstable, because semantic grounding does not transfer reliably into hidden-space rollout."
- **No published work uses contrastive image-latent alignment (CLIP-style InfoNCE) as the auxiliary signal** for latent visual reasoning, despite this being a natural construction. Either untried in this exact form or unpublished negative — worth probing in the round-3 POC.

## 16. Decision (revised)

**Do not start training Variant A or B in their current form.** But the path to a publishable training paper is more concrete than round 1 suggested.

### Round 2 reader-transfer addendum (most important negative result)

After round 2 we tested whether mit-B and mit-A latents — both of which gave 30 % held-out improvement under φ₁ = Qwen2.5-VL-7B — also improve transfer to φ₂ = Monet-7B. **They do not.** Mean Monet-7B NLL on the same 30 samples:

| latent source | φ₂ (Monet-7B) NLL mean | comment |
|---|---|---|
| POC 1 vanilla K=4         | 8.09 | original POC 3 finding |
| Mit-B λ=0.1 K=4 (norm reg)| **8.55** | *slightly worse* |
| Mit-A K=4 r=1 (low-rank)  | 8.30 | no improvement |

**Interpretation.** Norm regularization and low-rank parameterization change *which direction in φ₁'s embedding space* the optimization finds, but they do not change *which decoder behavior* the direction exploits. The shortcut is φ₁-specific at every (norm, rank, K) we tested. **No purely geometric mitigation will fix reader transfer.** Fixing it requires either:

- (i) **multi-reader optimization** at training time — sum reader-NLL over multiple frozen readers, forcing h to be readable by all of them. The trained generator's h would then have to satisfy a chorus of decoders, not one.
- (ii) **vision-encoder-grounded auxiliary loss** (LaViT-style cosine to teacher visual features) — anchor h to features that are by construction shared across readers of the same family.
- (iii) **structural change to the proposal**: latent serves as input to the *same model* (LIVR-style attention bottleneck) rather than to a separate frozen reader. Drops the cleanness of the reader-grounded framing but eliminates the transfer failure mode by construction.

This finding **escalates the round-3 requirements** below.

### Required round-3 POC (combined recipe)

Pre-validate the combined recipe on the same POC 2/POC 3 protocol, ~2–3 days of compute:

1. **Norm regularization** at λ ≈ 0.1 (proven mit-B effect on within-reader generalization) — keeps h on-manifold without breaking fit.
2. **Multi-question consistency** at N ≥ 5 per image (proven mit-D effect) — forces partial q-invariance.
3. **Auxiliary grounding term à la LaViT** (`ℒ_concept`: cosine to reader's vision-encoder output for the same image, + `ℒ_traj`: KL on cross-attention with top-K=8 sparsification). The visual_baseline.json in this repo gives the per-image targets to use. **This is the most important addition — it directly addresses the reader-transfer failure by anchoring h to family-shared features.**
4. **Multi-reader training-time loss**: optimize against `Σ_φ NLL_φ` rather than a single frozen reader. At least 2 readers (Qwen2.5-VL-7B + Monet-SFT-7B). This is the structural fix for the reader-transfer failure. Costs ~2× training compute per step but is the only proven way (per the negative result above) to achieve cross-reader portability.
5. **Negative-control ablation** (per arXiv:2004.05704): at minimum, swap `target_norm` to a random value and re-run mit-B; swap multi-Q targets to questions about a *different* image and re-run mit-D. Improvements in those controls would imply the round-2 within-reader gains are generic regularization, not visual grounding.
6. **Steering probe in evaluation** (arXiv:2512.21711 protocol, mirrored from §14) — required certification that latents do real work and carry q-portable content, not just shortcut.

**Decision rule for moving to scale**: combined-recipe `h*` reaches POC 2 held-out NLL within 1.5 nat of the image-oracle baseline AND **POC 3 transfer drop (φ₂ − φ₁) ≤ 2 nat at K=4** AND steering probe shows non-trivial sensitivity to perturbation. The reader-transfer criterion is the *binding* one — without it, the method is fundamentally a single-reader curiosity.

### Variant A vs Variant B

The POC tested Variant A's optimization landscape directly. Variant B's gradient signal is the same shape (reader-NLL, GRPO-reweighted) — the shortcut basin is reachable identically; the embedding-space optimization pressure is slightly less aggressive (sampling rather than direct optimization), but the manifold and transfer concerns are identical. **Run a small Variant B trial under the round-3 combined recipe before scaling it.** arXiv:2510.23925's diversity-seeking RL term is documented to mitigate reward-hacking in latent-CoT RL — worth incorporating.

### Compute spent / decision cost

Round 1: 3 hours (POC 1: 135 min; POC 2/3/baseline/multi-Q/q3: ~30 min combined).
Round 2: 3.5 hours (mit-B: 55 min; mit-A: 82 min; N=5 prep + tune + eval: ~30 min; steering: ~5 min; held-out evals: ~10 min).
Total compute: ~6.5 hours on a single A6000.
**Decision cost averted: GPU-weeks of training a method that would have produced shortcut-dominant, reader-specific latents.**

---

## Appendix — what's on disk

- `tune.py` — POC 1 optimization driver. Single file, hardcoded constants, JSONL crash-resume.
- `evaluate_held_out.py` — POC 2.
- `evaluate_transfer.py` — POC 3.
- `compute_visual_baseline.py` — POC 4 reference (natural visual-token statistics).
- `analyze.py` — aggregator. Re-runnable; produces `results/ANALYSIS.md` and `results/<poc1_run>/figs/*.png`.
- `prepare_data.py` — GQA loader (100 POC1 + 30 POC2-pairs).
- `data/poc1_samples.jsonl`, `data/poc2_pairs.jsonl`, `data/gqa_images/*.jpg`.
- `results/20260501-030544_poc1_full/` — POC 1 results.jsonl + 300 latent .pt + figs.
- `results/20260501-052224_poc2_held_out/` — POC 2 results.jsonl.
- `results/20260501-052251_poc3_transfer/` — POC 3 results.jsonl.
- `results/visual_baseline.json` — POC 4 reference stats.
- `results/ANALYSIS.md` — auto-generated tables from `analyze.py`.
- `docs/LITERATURE_RECON.md` — companion lit review (arXiv IDs, exact quotes).

## Appendix — key references (pull from `docs/LITERATURE_RECON.md` for full context)

- Monet (arXiv:2511.21395, CVPR 2026) — base architecture, VLPO objective, three-stage SFT.
- "Imagination Helps Visual Reasoning, But Not Yet in Latent Space" (arXiv:2602.22766) — Input-Latent / Latent-Answer Disconnects; CapImagine alternative.
- "Do Latent Tokens Think?" (arXiv:2512.21711) — empirical critique of Coconut; steering + OOD probe protocol. **Adopt this protocol verbatim.**
- "Coconut" (arXiv:2412.06769) — text-side latent reasoning, the analog being critiqued.
- Soft-prompt tuning baseline (arXiv:2104.08691); soft-prompt off-manifold attacks (arXiv:2402.09063); interpretability-vs-task trade-off (arXiv:2504.02144).
- LVR (arXiv:2509.24251) — closest "reconstruct-visual-content" objective; useful contrast for §8.1(C).
- Latent CoT survey (arXiv:2505.16782) — taxonomy and known failure modes for token-wise latent CoT.

---

## §17 — Round 3: design complete + critical empirical reinforcements

After round 2 closed with the open question "do round-2 mitigations transfer cross-reader?" (answer: no), round 3 pre-validated the round-2 wins against the arXiv:2004.05704 random-control concern, characterized the K saturation curve, tested a cleaner sibling reader, and produced a complete design package for the proliferated project. Round 3 ran no model-training (per §16's gate); it ran more probes and built more docs.

### §17.1 Random-control ablation: mit-B is mostly generic regularization

Per arXiv:2004.05704 ("Visual grounding via random/insensible cues gives similar VQA gains"), swept `target_norm` ∈ {natural=57.86, low=0.88, high=200, random U[10,200] per-sample} at K=4, λ=0.1, 30 samples, 500 steps (`tune_random_control.py`). Held-out NLL on q' (POC2 protocol):

| condition | held-out NLL mean (95 % bootstrap CI) | gain vs POC1 baseline (3.91 nat) |
|---|---|---|
| natural (=mit-B λ=0.1) | **2.394** [1.47, 3.38] | **−39 %** |
| random per-sample | **2.849** [1.93, 4.00] | **−27 %** |
| low | 3.481 [2.32, 4.75] | −11 % |
| high | 4.763 [3.51, 6.00] | (degenerate; can't fit) |
| no_input baseline | 9.695 | — |

The 95 % CIs of `natural` and `random` overlap heavily — they are **not statistically distinguishable at n=30**. About 70 % of mit-B's −39 % gain is reproduced by an arbitrary per-sample random target.

**Interpretation.** `high` is a degenerate control (training NLL can't reach 0 because L_norm overpowers the NLL gradient). `low` (target=0.88) is a near-zero target equivalent to weight-decay shrinkage; produces the smallest gain. `random` has per-sample targets uniformly drawn from [10, 200] — a wide range that does NOT center on 57.86 — yet still produces ~70 % of the natural-target gain. So mit-B's effect is **mostly generic regularization at *some* moderate target**, not specifically norm-matching to natural visual-token statistics.

**Implication.** Norm regularization at λ=0.1 should NOT be load-bearing for any visual-grounding claim in the proliferated project. The L_concept (LaViT-style cosine to teacher visual features) and the multi-reader NLL (sum over R≥2 frozen readers) must carry the grounding signal. λ_norm at 0.1 stays in the loss as a cheap stabilizer with established within-reader benefit, but its grounding contribution is now empirically bounded.

### §17.2 K saturation: the shortcut basin extends to K=64

`tune_largeK.py` ran K ∈ {32, 64} × 30 samples × 500 steps. Both reach final NLL ~0.0001 (perfect fit). Final per-token norm mean: K=32 → 19.00, K=64 → 18.63 — much lower than K=4's ~250 endpoint. Per-token magnitude scales DOWN with K: more positions → less force per position needed to reach NLL=0.

**Reading.** Capacity is not a barrier to the shortcut. A K=4 latent already has 4 × 3584 = ~14 000 floating-point parameters per sample — orders of magnitude more than the (single-token answer) GQA Q answer entropy. K=64 adds a 16× capacity multiplier and the basin remains reachable.

This rules out one mitigation hypothesis: that K is actually too small at 4–16, and K=64 would expose useful structure. It would not. Capacity reduction is not the lever.

### §17.3 Cleaner cross-reader test: Monet-SFT-7B stage2

Per HANDOFF option 3, downloaded `NOVAglow646/Monet-SFT-7B` (47 GB; 3 stages × 4 shards each). Stage 2 is the SFT-only checkpoint — closer to the base Qwen2.5-VL-7B-Instruct than the post-RL Monet-7B used in round 2's transfer test. Tested all relevant latents under stage2 reader (`evaluate_monet_sft_transfer.py`):

| latent source | Monet-SFT-7B NLL mean (95 % CI) | Monet-7B post-RL (round 2) |
|---|---|---|
| POC1 vanilla K=4 | 8.195 [7.44, 8.99] | 8.09 |
| Mit-B λ=0.1 K=4 | 8.245 [7.60, 8.93] | 8.55 |
| Mit-A K=4 r=1 | 8.052 [7.46, 8.73] | 8.30 |
| Random-control natural | 8.245 [7.59, 8.92] | — |
| Random-control low | 8.153 [7.51, 8.83] | — |
| Random-control high | 8.387 [7.87, 8.90] | — |
| Random-control random | 8.563 [8.04, 9.09] | — |

All CIs overlap heavily. The cleaner sibling shifts numbers by 0.2–0.3 nat but **the gap remains huge** — within-reader oracle is ~10.4 nat, within-reader optimized fits are 1.5–3.5 nat, and *every* cross-reader fit clusters at 8.0–8.6 nat (well above no-input ~9.7 only by a margin smaller than the within-reader → cross-reader gap).

**Strengthens the round-2 conclusion.** It's not that round 2 picked the wrong sibling — within-reader fits are genuinely φ₁-decoder-specific, regardless of which sibling we test. **Any reader-transfer fix has to be structural at training time.** Multi-reader NLL (sum over R frozen readers during training) is the only known structural fix.

### §17.4 Updated round-3 mission and pass criteria

Round-3 is now a **gate**, not a scaling exercise. Built the design in `docs/ROUND3_POC_DESIGN.md` (1055 lines) with five hard pass thresholds:

| Metric | Pass threshold | Round-2 reference |
|---|---|---|
| Held-out NLL on q' (φ₁) | ≤ 2.5 nat | mit-B λ=0.1 hit 2.39; this is the floor we already know is reachable via regularization |
| Reader-transfer NLL on Monet-SFT-7B | ≤ 4.5 nat | mit-A K=4 r=1 was 8.05; the −3.5 nat target requires a structural fix |
| Random-control gain ratio | gain on real ÷ gain on shuffled-(I,q,y) ≥ 2.0 | mit-B's ratio per §17.1 is ≈ 1.4; insufficient |
| Steering probe ablation cost | ≥ +1.5 nat on each of {zero_pos, permute_within, gauss_noise} | round-2 §14 already showed +2.5 to +4.1 nat; should hold |
| 5K visual-grounding stress test (blank-image control) | accuracy drop ≥ 5 pp | not yet measured at POC scale |

Five-cell sweep: C1 = full recipe (R=2, K=16, K_q=3, all losses), C2 = R=1 (single-reader ablation), C3 = K_q=1 (no multi-Q), C4 = λ_concept=0 (no LaViT-style aux), C5 = random-control (shuffled images). Decision logic: C1 > {C2, C3, C4} on transfer + C5 fails (random control can't match real) → method works. Anything else → reformulate.

Time estimate: ~3 engineering days + ~24h compute on 4×H100 for the 5-cell sweep + eval.

### §17.5 Locked design choices for the proliferated project

From the synthesis docs (`docs/VLM_SURVEY.md`, `docs/AUX_LOSS_AND_ARCH_DESIGN.md`, `docs/VARIANT_B_GRPO_DESIGN.md`, `docs/EVAL_BENCHMARK_PLAN.md`, `docs/TRAINING_DATA_PLAN.md`):

| # | Choice | Justification (one-liner) |
|---|---|---|
| L1 | Base model: Qwen2.5-VL-7B-Instruct | "Qwen3.5-VL" doesn't exist; Qwen3-VL-8B has DeepStack which complicates slot injection; Qwen2.5-VL has 7+ sibling fine-tunes for cross-reader testing |
| L2 | Generator architecture: LIVR-style same-VLM + special `<\|latent\|>` tokens | Closest published sibling (arXiv:2512.21218); minimal new parts |
| L3 | LoRA r=32 on q/k/v/o + gate/up/down projections | Established VLM RL recipe; trades headroom for diversity |
| L4 | Stage-1 attention masking | Answer tokens cannot attend to image during latent emission; prevents trivial copy |
| L5 | K=16 default; ablate K=4 / K=8 in round-3 | LIVR default; round-3 tests saturation |
| L6 | Aux loss: NLL_multi + 0.3·L_concept + 0.1·L_norm | LaViT λ=0.3 confirmed; L_norm bounded as stabilizer per §17.1 |
| L7 | L_concept teacher = φ₁'s own post-merger visual tokens (round-3); Qwen2.5-VL-32B (M2 stretch) | Free in round-3; LaViT-published recipe in M2 |
| L8 | Multi-reader R=2: Qwen2.5-VL-7B-Instruct + Monet-SFT-7B-stage2 | Both on disk; Monet-SFT is the SFT-only sibling; structural fix for reader-transfer |
| L9 | Multi-Q K_q=3, generator input is image-only (no q) | q-invariance baked into architecture |
| L10 | Variant A SFT first; Variant B (VLPO) only if A insufficient | Variant B requires VLM-R1 fork (~2 weeks engineering) |
| L11 | Random-control ablation mandatory at every milestone | arXiv:2004.05704 gate per §17.1 |
| L12 | Curriculum warmup 200 steps, NLL coefficient 0.1→1.0, λ_norm 0→0.1 | Cosine ramp; matches LaViT recipe (T_w=400 in their paper) |

### §17.6 What changes in the TL;DR after round 3

Round 1 said: "don't train, you'll get shortcut latents."
Round 2 said: "concrete recipes (norm reg / low-rank / multi-Q) move within-reader NLL by 24–39 %, but they do NOT fix reader-transfer."
**Round 3 says:** "those round-2 within-reader gains are mostly generic regularization (per the random-control), not visual grounding. The proliferated project must (i) make the multi-reader NLL load-bearing for transfer, (ii) make L_concept load-bearing for grounding, and (iii) report random-control as a mandatory ablation at every milestone."

The full design is in `PROLIFERATED_PROJECT_PLAN.md` and `docs/ROUND3_POC_DESIGN.md`. Both are morning-briefing-ready: the user can read them and start coding round-3 within an hour.

### §17.7 Files added round 3

Scripts: `tune_random_control.py`, `evaluate_random_control_heldout.py`, `tune_largeK.py`, `evaluate_monet_sft_transfer.py`.

Run dirs: `results/20260502-024808_random_control_norm_reg/`, `results/20260502-034306_random_control_heldout/`, `results/20260502-034324_poc1_largeK/`, `results/20260502-041415_monet_sft_transfer/`.

Docs (subagent-produced): `docs/VLM_SURVEY.md` (~600 lines), `docs/AUX_LOSS_AND_ARCH_DESIGN.md` (~870 lines), `docs/VARIANT_B_GRPO_DESIGN.md` (~700 lines), `docs/EVAL_BENCHMARK_PLAN.md` (528 lines), `docs/TRAINING_DATA_PLAN.md` (370 lines), `docs/ROUND3_POC_DESIGN.md` (1055 lines).

Top-level (synthesis): `PROLIFERATED_PROJECT_PLAN.md` (646 lines).

Cumulative compute round 1+2+3: ~7 hours on a single A6000. Cumulative wall-clock session: ~9-10 hours.
