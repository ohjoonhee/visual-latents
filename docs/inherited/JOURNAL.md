# Reader-Grounded Latent — Pre-Training Feasibility POC — Journal

**Research question:** Before training a reader-grounded latent visual reasoning policy (`a.md`, two variants), do reader-grounded latents that satisfy the objective even *exist* — and if so, are they visual representations or answer-encoding shortcuts?
**Parent:** none (sibling to `../monet-latent-probe/` — both probe whether latent visual reasoning carries content, from different angles)
**Status:** active

---

## 2026-05-01 — brainstorm: scope locked, scaffold up

Brainstormed the proposed two-variant training method (`a.md`) before committing compute. Audit surfaced three load-bearing concerns; user committed to scope choices that pin the experiment.

### Method recap (what would be trained, if we trained it)

- Generator $\pi_\theta$ (trainable Qwen2.5-VL-7B) emits $K$ continuous latents $h_{1:K}$ via Monet's start-token + hidden-state-feedback protocol.
- Frozen reader $\phi$ (frozen copy of generator's base) receives $h_{1:K}$ injected into its visual-token slots, **with no other image input** — this is a deliberate scope choice to make shortcuts harder, locked in this round.
- Variant A: $-\log\phi(y^\star \mid q, h_{1:K})$, gradient through frozen reader into latents into $\theta$.
- Variant B: reward-weighted reader-NLL with GRPO advantages on rollout-sampled $\hat y^{(g)}$.

### Audit — what could go wrong

1. **Shortcut/answer-encoding (most critical).** $K \cdot d \approx 36{,}000$ floats is wildly more than needed to encode any answer; gradient descent with no penalty on latent content will find "encode literal answer as soft prompt" if that path exists. This is the same failure mode `monet-latent-probe` is testing on Monet's *existing* latents.
2. **Off-manifold latents.** Reader was trained on natural visual tokens with a particular distribution (norms ~283-290 in Monet's spike measurement); reader-NLL has no penalty for leaving that manifold. Optimal latents may be adversarial-like.
3. **Variant B is not really policy gradient.** The objective sidesteps continuous-action policy gradient by making the *reader's* likelihood the differentiable signal; this is reward-weighted differentiable supervision, not RL on continuous latents. Honest framing matters but doesn't refute the method.

### Citation correction

User cited arXiv `2602.22766` as evidence that "VLPO-style training collapses latents into a narrow region of latent space." Verified: paper is **Li et al., *Imagination Helps Visual Reasoning, But Not Yet in Latent Space***. It does not analyze VLPO specifically — it's a broader critique. Real findings:

- Latent tokens "encode limited visual information and exhibit high similarity"
- **Input-Latent Disconnect** — input perturbations don't change latents
- **Latent-Answer Disconnect** — latent perturbations don't change answers

The paper proposes CapImagine (text imagination) as an alternative.

**Implication for our framing.** Two updates: (a) the cite needs to be rephrased as a general latent-visual-reasoning critique, not VLPO-specific; (b) the proposed method's no-image-on-reader constraint is *structurally immune* to both named disconnects (no other input path → forces input-latent coupling; reader-NLL gradient → directly couples latent-answer). The method should be pitched as a response to Li et al.'s diagnosis, not just an alternative to VLPO. We are also racing them — they argue text > latent; we argue reader-grounded-latent > VLPO-latent.

### Scope decisions locked this round

- **Reader has no image access.** Latents are the only visual path to $\phi$. Strict version of Variant A.
- **Reader = frozen copy of generator's base** (Qwen2.5-VL-7B). Same architecture, same hidden dim.
- **POC 1-3 confirmed valid.** User accepted the soft-prompt-tuning-as-feasibility-ceiling framing.

### POCs designed (full detail in `docs/README.md`)

| POC | Tests | Decision rule |
|-----|-------|---------------|
| 1 — Reachability | Per-sample optimize $h_{1:K}$ under frozen $\phi$, no image. Sweep $K \in \{1,2,4,8,16\}$. | If H1 refuted (loss plateaus high even at $K=16$): halt, revisit method. |
| 2 — Held-out question | Same image, *different* question $q'$. Does $h^*$ optimized for $q$ transfer? | **Make-or-break shortcut detector.** If H2 refuted (held-out accuracy ≈ chance), training plan deferred. |
| 3 — Reader transfer | $h^*$ under $\phi_1$ = Qwen2.5-VL-7B, eval under $\phi_2$ = Monet-SFT-7B (sibling fine-tune, on disk in monet-latent-probe). | Informative not gating; full collapse = adversarial-to-$\phi_1$. |

POC 2 is the new contribution — POCs 1 and 3 are standard, but a held-out-question test on the same image is the cleanest way to distinguish "visual content" from "answer encoded as soft prompt." Predicted result is near-chance held-out accuracy, which would meaningfully undermine the proposed training as written.

### Connection to monet-latent-probe (sibling experiment)

The two experiments triangulate on the same broad question — does latent visual reasoning carry content? — from different sides:

- monet-latent-probe (H1): are *existing* Monet latents causally necessary at inference?
- reader-grounded-latent-poc (H1-3): are *trainable-target* latents under reader-grounded supervision reachable, and content-bearing or shortcut?

If monet-latent-probe finds inert latents AND this finds shortcut latents: strong joint case that the paradigm needs structural fixes (e.g., information bottleneck on $h_{1:K}$, or augment-mode + bottleneck regularization), and the next paper is a *negative-result analysis with a structural fix proposal*, not a training paper.

### What's next

1. Write POC 1 dev pilot script: 5 GQA samples, $K=4$, single seed, 500 Adam steps, frozen Qwen2.5-VL-7B, no image to reader, gradient-thru-reader to $h_{1:K}$ as `nn.Parameter`. Verify slot-injection produces non-NaN, non-flat reader logits. Verify loss decreases monotonically. Verify image-oracle NLL << no-latent baseline NLL.
2. If pilot clean: full POC 1 sweep (100 samples × 5 K × 3 seeds, ~10 h on A6000).
3. POC 2 dev pilot on 5 CLEVR images × 2 questions in parallel-ish. Canary check: is the held-out-accuracy gap visible even in the dev pilot?

### Pinned references for next session

- Brief: `docs/README.md` (this experiment)
- User's method draft: top-level `a.md` (Variant A & B)
- Sibling experiment: `../monet-latent-probe/` — JOURNAL has the latent-extraction patch, latent norm measurements, and the in-progress H1 result
- Citation to fix in any paper draft: `2602.22766` is Li et al. on broad latent-visual-reasoning disconnects, not VLPO-specific
- Slot-injection details for Qwen2.5-VL: bypass `Qwen2_5_VLProcessor` vision encoding, manually substitute embeddings at `<|image_pad|>` slot positions; dim 3584, bf16

---

## 2026-05-01 — overnight: POC 1-4 complete, training currently NOT justified

Ran the full POC sweep over ~5 hours (135 min compute on POC 1 + ~10 min on POC 2/3/visual-baseline + analysis & writeup). Headline: **the proposed method as written produces shortcut-dominant, off-manifold, reader-specific latents.** Don't train yet. Full numbers and recommendations in `REPORT.md`; handoff at `HANDOFF.md`.

### Headline numbers (n=100 images, 95 % bootstrap CIs)

| | K=1 | K=4 | K=16 |
|---|---|---|---|
| POC 1 final_nll (reachability) | 0.21 [0.07, 0.38] | ~0 | ~0 |
| POC 1 %final < oracle | 100 % | 100 % | 100 % |
| POC 2 h*_nll(q2) | 3.58 | 3.76 | 3.52 |
| POC 2 oracle_nll(q2) | 10.36 | 10.36 | 10.36 |
| POC 2 %h*≈no_input | **53 %** | **50 %** | **53 %** |
| POC 3 transfer drop (φ₂ − φ₁) | +8.97 | +7.72 | +5.77 |
| POC 4 norm_mean | 23.4 | 19.3 | 19.0 |
| POC 4 inter-sample cos | 0.082 | 0.051 | 0.281 |

**Reference baselines observed.** Natural Qwen2.5-VL post-merger visual tokens: norm_mean 57.86, intra-image cos 0.295, inter-image first-token cos **0.845**. Compare Monet's *actually-emitted* latents (sibling experiment): norm 288 (different scale — last-layer hidden states), intra-block 0.758, inter-sample 0.927.

### Three things newly settled this round

1. **One latent of dim 3584 has more than enough capacity to encode any single-token answer under reader-NLL.** K=1 hits 80% confidence. The capacity argument from the audit is empirically tight at the smallest K we tested.
2. **Reader-NLL alone produces sample-specific encodings** (inter-cos 0.05–0.28, *much lower* than natural visual tokens at 0.85 or Monet's 0.93). The optimization doesn't mode-collapse; it finds a unique adversarial soft prompt per sample. Surprising: I expected mode collapse, got the opposite — each sample's solution is distinct and reader-specific.
3. **Held-out question test rejects the strict shortcut hypothesis but not the weak one.** 50 % of `h*` are closer to no_input than to oracle on q2. The other half retain partial visual signal (e.g. "color" class). The result is a *mixture*, not a pure shortcut — but the mixture is dominated by reader-fooling content, not visual content.

### Surprises (non-obvious, save for next session)

- **`oracle_nll ≈ no_input_nll` on GQA.** Image presence often makes the *exact* gold word *less* likely (model prefers a near-synonym). Explained by GQA's narrow lexical answer style. **Consequence:** for follow-up work, switching to a benchmark with looser scoring (e.g. LLM-judge) would give a more meaningful oracle baseline. Worth noting before re-using the GQA pipeline.
- **K=16 has *higher* inter-sample cos sim (0.28) than K=4 (0.05).** More capacity → more shareable structure, less needs-to-be-orthogonal-per-sample. Conjecture: at small K each latent must be answer-specific; at large K some capacity becomes diffuse / generic.
- **Transfer drop shrinks with K** (8.97 → 5.77 from K=1 to K=16). Same intuition. Implies that capacity helps reader-portability, contra the naive assumption that more capacity gives more room for shortcut.
- **POC 4 norm_mean ~20**, not the ~58 of natural visual tokens. Off-manifold direction is *smaller* norm, not larger. Consistent with "find a direction that nudges logits without saturating."

### Decision

Variant A as written: **do not train yet.** §8.1 of `REPORT.md` lists four structural mitigations (info bottleneck / distributional reg / aux grounding / multi-q consistency). Cheapest to try first: (B) distributional regularization against `results/visual_baseline.json`. Most aligned with the failure mode: (D) multi-question consistency, which structurally pulls the method toward Monet's q-invariant latent shape.

### What was added this round

- `prepare_data.py` + 100 GQA samples and 30 multi-Q pairs in `data/`
- `tune.py` (POC 1 driver), `evaluate_held_out.py` (POC 2), `evaluate_transfer.py` (POC 3), `compute_visual_baseline.py` (POC 4 reference)
- `analyze.py` aggregator (auto-regenerates `results/ANALYSIS.md` + figs)
- `REPORT.md`, `HANDOFF.md`, `docs/LITERATURE_RECON.md`

### What's open

- One mitigation has to be implemented and re-validated through the same POC. See `HANDOFF.md` §"What to do next."
- Generator-side Input-Latent Disconnect probe (arXiv:2602.22766 methodology) defers until a trained generator exists.
- Variant B equivalent POC defers; the loss-landscape symmetry argument says shortcut basin is reachable similarly, but on-policy sampling may behave differently.

### Mitigation D probe (added end-of-overnight)

Tested mitigation D (multi-question consistency) on the same 30 image pairs. Per-image, optimized a single h (K=4, 500 Adam steps) on the sum of reader-NLLs over q1 and q2. Then evaluated on a held-out q3 (single-token answer, different from q1/q2; 30/30 found in GQA testdev_balanced).

**Headline:** mean held-out q3 NLL: single-Q `h*` (POC 2's setup) = 3.91; multi-Q `h*` = **2.40**. Multi-Q wins on **70 % of samples** (median improvement -0.69 nat). On the training questions, multi-Q reaches NLL≈0 on both q1 *and* q2 simultaneously — capacity (K·d=14k floats) is plenty to learn 2 (q→a) shortcuts.

**Interpretation:** Naive multi-Q at K=4 does not break the shortcut basin (both q1 and q2 reach 0 trivially), but the optimization cost of supporting two distinct (q→a) shortcuts pushes `h*` slightly toward genuine visual content, and the spread benefits q3. **Multi-Q is necessary but not sufficient.** The recommended combined recipe for the next POC iteration:

1. multi-Q with N≥5 questions per image (force more visual generalization)
2. capacity bound on h (e.g. K=1 or vector-quantize to a small codebook)
3. distributional anchor toward natural-visual-token norms (`visual_baseline.json` is the prior)

**Files added:** `tune_multiq.py`, `evaluate_multiq_heldout.py`, `results/20260501-053151_mitigation_D_multiq/`, `results/20260501-060044_mitigation_D_q3_heldout/`.

### Mitigation D follow-up: K=1 (capacity-bound check)

To test the "smaller K helps" intuition, re-ran multi-Q at K=1 + held-out q3.

**Headline:** K=4 multi-Q (held-out q3 NLL = 2.40) BEATS K=1 multi-Q (3.72). Pure K-reduction does not improve generalization. K=1 is also capacity-bound at the *training* objective: cannot drive both q1 and q2 to NLL ≈ 0 (training mean 0.5-0.8). The right capacity bound is *structural* (quantization / low-rank / sparsity), not dimensional.

**Implication for training-paper recipe:** Drop "K=1" from the mitigation list. Multi-Q + structural-capacity-bound + distributional-anchor + many-Qs-per-image is the right combined recipe to pre-validate next.

**Files added:** `results/20260501-060302_mitigation_D_multiq/` (K=1 multi-Q), `results/20260501-063057_mitigation_D_q3_heldout/` (K=1 held-out q3).

REPORT.md updated with §10.1; HANDOFF.md option 0 covers this. Total overnight compute: ~3.5 hours; total wall-clock incl. setup, lit recon, analysis, writeup: ~5.5 hours.

---

## 2026-05-01 — round 2: mitigation probes (B, A, N=5, steering) + supplemental lit recon

User extended the overnight to add mitigation probes. ~3.5 hours of compute. Findings substantially revised the report's framing.

### What was run

- **Mit-B (norm regularization)**: 30 POC1 samples × K=4 × λ ∈ {0,0.1,1.0,10}. Loss = NLL + λ·(1/K)·Σ(||h_i|| − 57.86)². Init at NATURAL_NORM (matches POC 1 baseline).
- **Mit-A (low-rank h)**: 30 POC1 samples × (K, r) ∈ {(4,1),(4,2),(4,4),(16,1),(16,4),(16,16)}. h = U @ V parameterization.
- **N=5 multi-Q**: 25 GQA images × 5 train Qs + 1 held-out q'. Optimize single h on sum of 5 NLLs.
- **Steering probe**: POC 1's K=4 latents, 8 perturbation types × 30 samples = 240 measurements.
- **Held-out evals**: mit-B and mit-A latents on held-out q (POC2 q2 if present, else fresh GQA q3).
- **Supplemental lit recon** (subagent): VQ codebooks, LVR full method, post-2025-12 papers, multi-Q precedents.

### Headline numbers (vs POC1 single-Q baseline of 3.91 nat held-out)

| mitigation | best held-out | improvement |
|---|---|---|
| Mit-B λ=0.1 (norm reg) | 2.39 | −38 % |
| Mit-A K=4 r=1 (rank-1) | 2.65 | −32 % |
| N=2 multi-Q (round 1) | 2.40 | −39 % |
| N=5 multi-Q | 3.28 | −24 % |
| oracle (image present) | 10.36 | n/a (GQA narrow gold) |
| no_input | 9.69 | reference |

**Three independent mitigations land in the same 2.4–3.3 nat band**, all ~24–39 % improvement. None individually fixes reader-transfer (POC 3) — that wasn't re-tested in round 2 and is open for round 3.

### Surprising / non-obvious findings

1. **Norm regularization at λ=0.1 is a free lunch on held-out.** Same training NLL (~0) as λ=0, but held-out is 36 % better. λ=10 over-regularizes and breaks training. Sweet spot is mild.
2. **Even rank-1 latents (K=4, all colinear in R^d) reach NLL ~0 at training time.** Capacity bound via low-rank does not prevent shortcut convergence — only changes the *direction*. Same lesson as mit-B.
3. **Steering probe inverts the round-1 narrative.** The latents ARE causally functional (zero_pos_3 → +4.1 nat, gauss_noise_1.0 → +4.1 nat, position-specific, sample-specific). This is **opposite of Coconut's text-latent inert-placeholder finding** (arXiv:2512.21711). The reader-grounded latents encode useful info; the issue is the *content* (q-specific not image-specific), not absence-of-content.
4. **N=5 multi-Q does not dominate N=2.** Different held-out targets so not strict comparison, but at face value the simple N→generalization scaling does not compound as I'd hypothesized after round 1.
5. **arXiv:2004.05704 caveat is critical.** Grounding-style auxiliary objectives in VQA often work via generic regularization, not visual grounding. **None of round 2's mitigations have been controlled against random/insensible cues.** A random-target-norm mit-B and a random-image-Q multi-Q must be run before scaling.
6. **LIVR (arXiv:2512.21218, Dec 24 2025)** is the closest published sibling (single-model attention bottleneck, no aux loss, K=16, t-SNE shows latents on visual manifold). LaViT (arXiv:2601.10129, Jan 15 2026) is the strongest documented mitigation recipe (cosine to teacher + KL on cross-attention + curriculum gating, +15–17 pp gains).

### Decision update

Round 1 said "don't train, you'll get shortcut latents." Round 2 says "here are concrete recipes that move the needle, and they're combinable, but you still need a control ablation and reader-transfer test before scaling." The path to a publishable training paper is:

1. Combined recipe: mit-B λ=0.1 + multi-Q N=5 + LaViT-style aux loss
2. Random-control ablations (per arXiv:2004.05704)
3. Steering probe (per arXiv:2512.21711)
4. Reader-transfer re-test (POC 3 on all round-2 latents — not yet run)

If all four pass under the combined recipe, scale to a small Variant A training run.

### Round-2 reader-transfer addendum (most important negative result)

After computing held-out gains for mit-B and mit-A under φ₁, ran `evaluate_round2_transfer.py` to test whether the same mitigation latents also improve transfer to φ₂ = Monet-7B. **They do not.**

| latent source | φ₂ (Monet-7B) NLL mean | comment |
|---|---|---|
| POC1 vanilla K=4 | 8.09 | original POC 3 finding |
| Mit-B λ=0.1 K=4 | **8.55** | slightly worse |
| Mit-A K=4 r=1 | 8.30 | no improvement |

So **norm-on-manifold and rank-1 do not rescue cross-reader portability.** They change *which direction* the optimization finds in φ₁'s embedding space, but the direction remains exploitative of φ₁'s specific decoder.

**Implication for the training paper:** the held-out-question gain is a genuine within-reader effect, but reader-transfer requires a structural change at training time, not a geometric mitigation. Best candidates: (i) multi-reader training-time loss (sum NLL over multiple frozen readers), (ii) LaViT-style auxiliary cosine-to-teacher-visual-features loss (anchors h to family-shared features), (iii) drop the separate-frozen-reader framing entirely and use LIVR-style single-model attention bottleneck.

REPORT.md TL;DR and §16 updated with this finding.

### Files added (round 2)

Scripts: `tune_mitigationA.py`, `tune_mitigationB.py`, `tune_nq.py`, `prepare_data_nq.py`, `evaluate_mitA_heldout.py`, `evaluate_mitB_heldout.py`, `evaluate_nq_heldout.py`, `steering_probe.py`, `evaluate_round2_transfer.py`.

Run dirs (in `results/`): `20260501-183241_mitigation_B_norm_reg/`, `20260501-192754_mitigation_A_lowrank/`, `20260501-205109_mitigation_B_heldout/`, `20260501-205127_mitigation_A_heldout/`, `20260501-205146_mitigation_D_N5/`, `20260501-214809_mitigation_D_N5_heldout/`, `20260501-214824_steering/`, `20260501-215453_round2_transfer/`.

Docs: `docs/LITERATURE_MITIGATIONS.md` (supplemental literature recon).

REPORT.md heavily revised: TL;DR rewritten, §11 (mit-B), §12 (mit-A), §13 (N=5), §14 (steering), §15 (updated lit), §16 (revised decision + round-3 reader-transfer addendum) all new. HANDOFF.md option 0 updated.

Round 2 compute: ~3.5 hours. Cumulative: ~7.5 hours wall-clock.

---

## 2026-05-02 — round 3 design + random-control + K-sweep + cleaner sibling

Continuation: with 5 parallel research subagents + 1 GPU ablation pass, the goal was to deliver a round-3 POC design and a proliferated project plan ready for the next morning, plus three empirical follow-ups that the round-2 entry had flagged as missing.

### Empirical: random-control ablation for mit-B (`tune_random_control.py` → `evaluate_random_control_heldout.py`)

Per arXiv:2004.05704 ("VQA grounding gains often come from generic regularization"), swept `target_norm` ∈ {natural=57.86, low=0.88, high=200, random U[10,200] per-sample} at K=4, λ=0.1, 30 samples, 500 steps. Held-out NLL on q' (POC2 protocol):

| condition | held-out NLL mean (95% CI) | gain vs POC1 (3.91 nat) |
|---|---|---|
| natural (=mit-B λ=0.1) | 2.394 [1.47, 3.38] | **−39 %** |
| random per-sample | 2.849 [1.93, 4.00] | **−27 %** |
| low | 3.481 [2.32, 4.75] | −11 % |
| high | 4.763 [3.51, 6.00] | (degenerate; can't fit) |

**Reading.** The 95% CIs of natural and random overlap heavily — they're not statistically distinguishable at n=30. About 70 % of mit-B's −39 % gain is reproduced by an arbitrary per-sample random target. The remaining ~12 pp is plausibly target-aware (matching natural visual-token norm helps marginally), but the bulk of the win is generic regularization — exactly the failure mode arXiv:2004.05704 warned about.

`high` is a degenerate control (training NLL never reaches 0; reg term overpowers NLL gradient). `low` is a near-zero target equivalent to weight decay shrinkage; gives the smallest gain. So generic regularization at *some* moderate target is what helps, not specifically norm-matching.

**Implication for round-3 design.** Norm regularization at λ=0.1 should NOT be load-bearing for any visual-grounding claim. The L_concept (LaViT-style cosine to teacher visual features) and the multi-reader NLL (sum over R=2 frozen readers) must carry the grounding signal. λ_norm at 0.1 stays as a cheap stabilizer.

### Empirical: larger K sweep (`tune_largeK.py`)

K ∈ {32, 64} × 30 samples × 500 steps. Both reach final NLL ~0.0001. Final per-token norm mean: K=32 → 19.00, K=64 → 18.63 (much lower than K=4's ~250). So per-token magnitude scales DOWN with K: more positions → less force per position needed to reach NLL=0. The shortcut basin extends to all K up to 64; capacity is not a barrier. Likely saturates further (would expect similar at K=128, 256), but not tested.

### Empirical: cleaner cross-reader test on Monet-SFT-7B stage2 (`evaluate_monet_sft_transfer.py`)

Round 2 used post-RL Monet-7B as φ₂. Per HANDOFF option 3, downloaded NOVAglow646/Monet-SFT-7B and tested its stage2 (SFT-only) checkpoint against POC1 + mit-B + mit-A + all 4 random-control latents (n=30):

| latent source | Monet-SFT-7B NLL mean (95 % CI) | (vs Monet-7B post-RL from round 2) |
|---|---|---|
| POC1 vanilla K=4 | 8.195 [7.44, 8.99] | 8.09 |
| Mit-B λ=0.1 K=4 | 8.245 [7.60, 8.93] | 8.55 |
| Mit-A K=4 r=1 | 8.052 [7.46, 8.73] | 8.30 |
| ctrl natural | 8.245 [7.59, 8.92] | — |
| ctrl low | 8.153 [7.51, 8.83] | — |
| ctrl high | 8.387 [7.87, 8.90] | — |
| ctrl random | 8.563 [8.04, 9.09] | — |

All CIs overlap heavily. **The cleaner sibling shifts numbers by 0.2-0.3 nat but the gap remains huge.** Within-reader oracle-NLL is ~10.4 nat (Qwen2.5-VL-7B-Instruct on the held-out q'), so a cross-reader NLL of 8.0-8.5 is well above the within-reader fits (1.5-3.5 nat) and only ~2 nat below the no-input baseline (~9.7).

**Strengthens the round-2 conclusion.** It's not that round 2 picked the wrong sibling — it's that within-reader fits are genuinely φ₁-specific, regardless of which sibling we test. Any reader-transfer fix has to be structural at training time, not a per-sample geometric mitigation.

### Design: round-3 POC + proliferated project plan

Five parallel research subagents (~75 min each) produced:
- `docs/VLM_SURVEY.md` — recommendation: stay on Qwen2.5-VL-7B-Instruct. "Qwen3.5-VL" doesn't exist; the closest candidate Qwen3-VL-8B has DeepStack architecture (multi-level ViT injection) which complicates "splice into vision-token positions". Qwen2.5-VL has 7+ confirmed sibling fine-tunes; Qwen3-VL has primarily Cosmos-Reason2 and LoRA adapters.
- `docs/AUX_LOSS_AND_ARCH_DESIGN.md` — combined loss `L = L_NLL_multi(R=2, K_q=3) + 0.3·L_concept + 0.1·L_norm` with cosine-warmup curriculum. LIVR-style same-VLM architecture + LoRA r=32 + Stage-1 attention masking. LaViT teacher = Qwen2.5-VL-32B (verified by reading the paper, not CLIP/SigLIP as I'd assumed).
- `docs/VARIANT_B_GRPO_DESIGN.md` — **vanilla GRPO does not update continuous latents** (Monet's own paper reports this; LaCoT independently confirmed GRPO ≈ SFT on Qwen2.5-VL-7B MathVista). Without VLPO Gaussian reparameterization (σ=5, asymmetric KL β=0.04), Variant B reduces to "Variant A with stochastic answer-token sampling". Trainer recommendation: fork VLM-R1; TRL doesn't extend to continuous-latent gradients.
- `docs/EVAL_BENCHMARK_PLAN.md` — 5K visual-grounding stress test (MMVP+NaturalBench+BLINK 7 perception subtasks+MMStar+CV-Bench-3D+POPE-adv+VSR). Avoid ScienceQA/MMMU/MathVista (high language-only ceiling per MMStar's analysis). 4 control conditions (C1-C4) + decision rule for "we have visual grounding".
- `docs/TRAINING_DATA_PLAN.md` — pilot 10K (65% GQA-balanced + 25% CLEVR + 10% TallyQA), medium 100K, full 1M mixes. Probed 29 HF datasets live: GQA = 13 Qs/img / 94% single-token; CLEVR = 10 Qs/img / 100% single-token / synthetic = grounding-immune control. `HuggingFaceM4/VQAv2` and `ranjaykrishna/visual_genome` are broken — flagged.

Two synthesis subagents produced:
- `docs/ROUND3_POC_DESIGN.md` (1055 lines) — five hard pass thresholds, 5-cell sweep table with full decision logic, code-skeleton-level pseudocode for model+loss+curriculum, 7 named files to write with reuse-from-existing references, 5 risks with mitigations, ~3 engineering days + 24h compute on 4×H100.
- `PROLIFERATED_PROJECT_PLAN.md` (646 lines) — morning-briefing master plan. 12 locked design choices (L1-L12), 6 open decision points (D1-D6) with recommendations, M1 (100K, ~1300 H100-h) / M2 (1M, ~1100-4800 GPU-h) / M3 (Variant B RLVR, ~1600 H100-h) milestones with hyperparameter tables and tiered eval cadence (T1/T2/T3), 9-row risk table, code milestone list (15 files), 4 negative-result tripwires, paper-venue framings (NeurIPS 2027 positive method / workshop negative-result / arXiv technical report), §8 quick-start with exact bash commands.

### Integration

REPORT.md §17 added with all empirical updates. HANDOFF.md significantly rewritten to reflect "round-3 design complete; ready for implementation" status.

Cumulative compute round 1+2+3: ~7 hours on a single A6000. Cumulative session wall-clock: ~9-10 hours. The user's morning-ready criterion is met.
