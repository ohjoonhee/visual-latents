# Proliferated Project Plan — Reader-Grounded Latent Visual Reasoning

**Compiled:** 2026-05-02 (morning briefing).
**Status:** master plan, supersedes all prior planning notes outside the dated `JOURNAL.md`. Read once; commit to §1 choices; answer §2 D-questions; start §8.
**Authoritative source docs (read these only if you disagree with what's here):**
`docs/VLM_SURVEY.md`, `docs/AUX_LOSS_AND_ARCH_DESIGN.md`, `docs/VARIANT_B_GRPO_DESIGN.md`, `docs/EVAL_BENCHMARK_PLAN.md`, `docs/TRAINING_DATA_PLAN.md`, `REPORT.md`, `HANDOFF.md`, `JOURNAL.md`.

---

## §0 Executive summary

**The method.** Train a generator π_θ (Qwen2.5-VL-7B + LoRA-r=32) that emits K=16 continuous latent embeddings `h_{1:K} ∈ ℝ^{16×3584}` from an image alone. The latents are scored by a frozen reader φ that has *no other visual input* — `h` is the entire visual path into φ. Two formulations: Variant A (SFT, NLL through frozen reader) and Variant B (RLVR via VLPO Gaussian, only meaningful with multi-reader reward + random-control reward).

**What the POCs found.** Round 1 (per-sample direct optimization of `h*`, `n=100` GQA): the shortcut basin is reachable at every K; held-out-question NLL stays ~50% closer to no-image than to image-oracle (POC 2); reader-transfer to Monet-7B fails by +5.77 nat at K=16 (POC 3); `h*` sits ~3× off-norm-manifold (POC 4). Round 2 (mitigations): norm-reg (λ=0.1), low-rank (r=1), and multi-Q (N=2/N=5) each give 24–39% within-reader held-out improvement, **but none fixes reader-transfer** — Mit-B λ=0.1 was actually slightly *worse* on Monet-7B. Steering probe (round 2) flipped the round-1 narrative: `h` is causally functional and position-specific, contra Coconut. The pathology is "reader-specific encoded answer," not "inert placeholder."

**Locked design choices** (see §1). Stay on Qwen2.5-VL-7B-Instruct. Generator is LIVR-style same-VLM with K=16 special `<|latent|>` tokens, LoRA r=32, Stage-1 attention masking. Loss = NLL_multi + 0.3·L_concept + 0.1·L_norm with a 200-step warmup curriculum. Two readers (φ₁=Qwen2.5-VL-7B-Instruct, φ₂=Monet-7B), three questions per image, K_q=3.

**Recommended path forward.** Round-3 POC (5-cell sweep, ~24 H100-h, decision-gates the project) → M1 medium training (100 K samples, ~1300 H100-h, publishable POC paper) → optionally M2 full training (1 M samples, ~4800 A6000-h, paper-quality) → optionally M3 Variant B RLVR (~1600 H100-h, only if M2 plateaus visibly below grounding ceiling). Negative-result branches (§5, §6) are pre-specified.

**Decision points for the user this morning** (full text in §2): D1 Variant A vs B for round-3 (recommend A); D2 4×H100 timeline; D3 paper deadline; D4 add VL-Rethinker as φ₃ in round-3 (recommend defer); D5 random-control C5 in round-3 (recommend yes); D6 eval cadence (recommend every 1K steps).

---

## §1 Locked design choices

The following are committed and out of scope for this round of debate. Each has a one-line justification and an arXiv reference where applicable. If the user wants to override one, do it before §2; once round-3 starts, these are frozen.

| # | Choice | Value | Justification |
|---|---|---|---|
| L1 | Base model | `Qwen/Qwen2.5-VL-7B-Instruct` (D=3584, hidden 28-layer LLM, custom ViT) | Largest sibling reader pool of any candidate (7+ verified RL/SFT siblings); preserves triangulation with `monet-latent-probe`. Qwen3-VL-8B's DeepStack is a research wrinkle, not a free lunch. (`docs/VLM_SURVEY.md` §5) |
| L2 | Generator architecture | B.1 — LIVR-style same-VLM with K new `<|latent|>` special tokens between `<|vision_start|>` and `<|vision_end|>` | Closest published precedent (LIVR, arXiv:2512.21218); LIVR's t-SNE shows latents on visual manifold with no aux loss — direct evidence the shape works. (`docs/AUX_LOSS_AND_ARCH_DESIGN.md` §B.1) |
| L3 | Generator trainable params | LoRA rank 32 on attention + MLP (~50 M trainable / 7 B frozen) | Fits 4×H100 with R=2 readers; LoRA is the standard for sibling fine-tunes; full FT not justified at POC scale. |
| L4 | Stage-1 attention masking | Answer tokens cannot attend to image tokens, only to `h` and the question | LIVR's Stage-1 mask is the only published mechanism that empirically forces latents onto the visual manifold without aux loss. Forces the bottleneck. |
| L5 | Latent count K | 16 default; ablations at K ∈ {4, 8} | LIVR optimum (sweep {4, 8, 16, 32} reported K=16 best); round-2 found K=4 multi-Q > K=1 multi-Q on held-out; K=16 is also where round-1 transfer drop was smallest. |
| L6 | Aux loss stack | `L_total = L_NLL_multi + 0.3·L_concept + 0.1·L_norm` with curriculum (§A.6 of design doc) | LaViT's λ=0.3 verified verbatim; L_norm at λ=0.1 verified in round-2 (free lunch on held-out). L_traj dropped because shape mismatch (h length 16 vs reader visual length ~200). |
| L7 | L_concept target `V_sem` | `φ₁.visual(x).hidden_states[-1]` (post-merger, last-layer of vision tower); per-token `k' = k mod T_v` | `results/visual_baseline.json` already characterizes this manifold (μ‖·‖=57.86, intra-image cos=0.295). External 32B teacher deferred — pre-cache `V_sem` to free GPU3. |
| L8 | Multi-reader R | R=2 for round-3 + M1; R=3 (add VL-Rethinker-7B) for M2 if budget; R=4-6 (LoRA-merge variants) for stretch | R=1 is the round-2 failure; R=2 is the minimum that distinguishes "reader-specific" from "shared structure"; R≥3 helps structural pressure compound. (See §C.3 — LoRA-merge readers OK only when paired with at least 1 genuine fine-tune.) |
| L9 | Multi-Q K_q (questions/image/step) | K_q=3 default; ablation at K_q=5 if time | Round-2 N=2 gave −39%, N=5 gave −24% (different held-out targets, not strictly comparable). K_q=3 is the documented sweet spot per `docs/AUX_LOSS_AND_ARCH_DESIGN.md` §A.4. |
| L10 | Generator's q-input | **None.** Generator emits `h(x)` from image alone | Bakes q-invariance in architecturally — round-2's structural fix for the shortcut. The reader receives `h(x)` + q. |
| L11 | Variant A first | SFT-style NLL through frozen reader; only run Variant B (VLPO with σ=5) if A passes round-3 and M1 | Variant B requires VLPO trainer fork (~2 weeks engineering); only worth it if A is structurally insufficient. (`docs/VARIANT_B_GRPO_DESIGN.md` §0, §7) |
| L12 | Random-control mandate | Rerun every cell with shuffled (image, q) pairs; gain on real ≥2× gain on shuffled, else pivot | arXiv:2004.05704: VQA grounding gains often come from generic regularization, not grounding. Non-negotiable per `docs/EVAL_BENCHMARK_PLAN.md` §C. |

---

## §2 Open decision points (the user must answer this morning)

Six items. For each: status, recommendation, justification, deadline. Mark `[decided: ___]` next to each before kicking off round-3.

### D1 — Variant A or Variant B for round-3?

- **Status:** pending
- **Recommendation:** **Variant A.**
- **Why.** Variant B requires forking a GRPO trainer (VLM-R1 or verl) to add VLPO Gaussian reparameterization on the latent emission step. The fork is ~200 LOC of `_compute_loss` patching + integration testing — call it 2 weeks of focused engineering before any RL training run. Variant A reuses all our existing slot-injection infrastructure and re-validates the same hypothesis (does multi-reader + aux grounding break the shortcut?) at ~1/4 the compute. Variant B is only worth running if Variant A succeeds at round-3 *and* M1 reveals a structural ceiling that further data alone won't break (`docs/VARIANT_B_GRPO_DESIGN.md` §7).
- **Deadline:** must answer before round-3 kicks off. If "B," add 2 weeks before code milestone 1 (§4.3).

### D2 — 4×H100 access timeline?

- **Status:** pending
- **Recommendation:** N/A (user's hardware).
- **Why.** Round-3 needs 4×H100 for one overnight (~24 H100-h). M1 needs 4×H100 for ~14 days continuous, or equivalent on 4×A6000 (~21 days at 1/3 throughput per A6000 vs H100). M2 needs 4×H100 for ~12 days or 8×A6000 for ~25 days. If only A6000 available, expect M1+M2 to span 6 weeks instead of 4.
- **Deadline:** before round-3 kicks off. Triggers compute-budget reconciliation (§4.4).

### D3 — Project deadline / paper venue target?

- **Status:** pending
- **Recommendation:** target **NeurIPS 2026** (May abstract / ~16 weeks to deadline) for a positive-method paper, with **NeurIPS 2026 Workshop on Compositional Learning** or **NoTeMS** as a fallback for negative-result framing.
- **Why.** M1 finishes in 2 weeks if 4×H100 is reserved; M2 in 4 weeks; full eval suite in 1 week; writing in 4 weeks. Total ~12 weeks from today, which gives 4 weeks of slack against a NeurIPS-class venue. ICML 2027 (Feb deadline) is the next conservative slot if NeurIPS slips.
- **Deadline:** by end of week. Affects M2's go/no-go decision and whether Variant B is in-scope.

### D4 — Run R=3 (add VL-Rethinker-7B as third reader) for round-3, or save for proliferated?

- **Status:** pending
- **Recommendation:** **defer to M2 stretch.** Round-3 uses R=2.
- **Why.** R=2 is the minimum that distinguishes reader-specific from shared structure. Adding R=3 (~14 GB download + 23 GB VRAM at fwd+bwd) tightens the 4×H100 budget at round-3, blocking the B=4 minimum. Defer to M2 once the R=2 gradient signal is verified to do useful work. Alternative: run a *held-out* reader test at round-3 — train under {φ₁, φ₂}, eval (no training) under φ₃ — see §3 (round-3 cell C4).
- **Deadline:** before round-3 kicks off.

### D5 — Random-control ablation: standalone cell C5 in round-3, or only at proliferated scale?

- **Status:** pending
- **Recommendation:** **yes, standalone in round-3.**
- **Why.** arXiv:2004.05704 caveat is the load-bearing scientific guardrail. At round-3 scale (~24 H100-h for 5 cells, +25% for one extra control cell = +6 H100-h), the cost is trivial relative to the cost of building all of M1 only to discover at end that the gains were generic regularization. The control is also the key disambiguator for the §6 paper-framing decision tree.
- **Deadline:** before round-3 kicks off.

### D6 — Eval cadence: full 5K stress test only at end of training, or every 1K steps?

- **Status:** pending
- **Recommendation:** **every 1K steps for 4-control held-out NLL (cheap, ~10 min per checkpoint); full 5K stress test only at mid-training and end (each ~2 hours).**
- **Why.** Per-1K cheap checkpoint eval catches divergence, mode collapse, and reward hacking early (`docs/VARIANT_B_GRPO_DESIGN.md` §3.6 specifies 8 failure modes that are best caught with mid-training diagnostics). Full 5K + ×4 controls is 2 hours per run; running it at every 1K steps for a 100K-step M1 = 200 hours of eval, which is half the M1 training budget — too expensive.
- **Deadline:** before round-3 kicks off.

---

## §3 Round-3 POC summary

Full design at `docs/AUX_LOSS_AND_ARCH_DESIGN.md` §D. Summary here for handoff.

**Mission.** Pre-validate the combined recipe (LIVR-style + multi-reader + L_concept + L_norm + multi-Q) at small scale before committing M1's GPU-weeks. Decide whether to proceed to proliferated training, pivot to a different architecture, or kill.

**Decision criteria** (all four required to proceed to M1):
1. **POC-2 held-out NLL** ≤ 2.0 nat (round-2 best was 2.39 nat; we need to beat it under multi-reader pressure).
2. **POC-3 reader-transfer drop** (φ₂ − φ₁ NLL) ≤ 2.0 nat at K=16 (round-2 was +5.8 nat; this is the binding criterion).
3. **Random-control gain** ≤ 50% of real gain (i.e., shuffled (image, q) recipe must lose ≥50% of the held-out NLL improvement).
4. **Steering probe**: zero_pos and cross-sample swap must increase reader NLL by ≥1 nat (round-2 baseline reached +4.1 nat for zero_pos_3; the curriculum must not break this causal sensitivity).

**5-cell sweep:**

| Cell | λ_concept | λ_norm | R | K_q | Notes |
|---|---|---|---|---|---|
| C1 (default) | 0.3 | 0.1 | 2 | 3 | The recommended recipe per `docs/AUX_LOSS_AND_ARCH_DESIGN.md` §A.6 |
| C2 (no concept) | 0.0 | 0.1 | 2 | 3 | Ablation: how much of the gain comes from L_concept? |
| C3 (no multi-reader) | 0.3 | 0.1 | 1 | 3 | Ablation: does R=2 actually fix transfer that R=1 didn't? |
| C4 (held-out reader test) | 0.3 | 0.1 | 2 | 3 | Same as C1 but eval-only on φ₃ = VL-Rethinker-7B (no training) |
| C5 (random control) | 0.3 | 0.1 | 2 | 3 | C1 with shuffled (image, q) pairs across batch |

**Time estimate.** ~24 H100-h for the 5-cell sweep × 4 GPUs ≈ 100 H100-h total. Fits one overnight. (`docs/AUX_LOSS_AND_ARCH_DESIGN.md` §A.7.)

**Phases** (per design doc §D.1):

| Phase | Wall-clock | Output |
|---|---|---|
| 1. Data prep — extend GQA samples to per-image 3-question lists | half-day | `data/round3_samples.jsonl` |
| 2. V_sem cache — precompute `φ₁.visual(x).hidden_states[-1]` for all training images | 1 hour, 1 GPU | `data/v_sem_cache.pt` |
| 3. Round-3 trainer — implement B.1 + §A.6 loss in `train_round3.py` | 1 day | `train_round3.py`, smoke test on 100 samples |
| 4. Round-3 sweep — 5 cells | overnight (4× H100) | `results/20260503-*_round3_*` |
| 5. Round-3 evaluation — POC-2 held-out + POC-3 transfer + steering | half-day | extend `analyze.py` |
| **Total** | **~3 days** | round-3 outcome |

**Round-3 → proliferated handoff** (§4 below): if all four criteria pass on C1, M1's medium training reuses C1's hyperparameters at 100K samples and 4 GPUs for 14 days. If C5 (random control) also passes, the gain is generic regularization and M1 is replaced with the §6 negative-result reframe.

---

## §4 Proliferated training plan

Three milestones: M1 (100K samples, 14 days), M2 (1M samples, 25 days), M3 (Variant B RLVR, optional, decision-gated).

### §4.1 Variant A SFT proliferated training

#### Milestone M1 — Medium training (100 K samples)

**Goal.** Train a generator that beats POC's per-sample-optimized `h*` on held-out questions, survives the random-control ablation, and shrinks reader-transfer drop. Sufficient for a publishable POC paper.

**Data mix** (`docs/TRAINING_DATA_PLAN.md` §3.2):

| Dataset | HF path | # samples | Sample frac | Why |
|---|---|---|---|---|
| GQA-balanced | `lmms-lab/GQA` cfg `train_balanced_instructions` | 50,000 | 50% | Multi-Q backbone (≥5 Qs/img filter) |
| CLEVR (cauldron) | `HuggingFaceM4/the_cauldron` cfg `clevr` | 20,000 | 20% | Grounding-immune control + fast convergence |
| TallyQA (cauldron) | `HuggingFaceM4/the_cauldron` cfg `tallyqa` | 10,000 | 10% | Counting requires visual access |
| CLEVR-Math | `dali-does/clevr-math` | 5,000 | 5% | Math+visual; OOD-flavor in-domain |
| Visual7W (cauldron) | `HuggingFaceM4/the_cauldron` cfg `visual7w` | 8,000 | 8% | Multi-Q diversity (pointing/MCQ) |
| AI2D (cauldron) | `HuggingFaceM4/the_cauldron` cfg `ai2d` | 2,000 | 2% | Diagram grounding |
| OK-VQA train | `Multimodal-Fatima/OK-VQA_train` | 5,000 | 5% | Knowledge-required diversity |

Multi-Q consistency loss target: GQA + CLEVR + Visual7W = 78 K samples with ≥4 Qs/img.

**Setup.**
- 4×H100 80GB, FSDP across GPUs, B=4 effective per GPU (B=16 global).
- K=16 latents, R=2 readers (φ₁=Qwen2.5-VL-7B-Instruct, φ₂=Monet-7B), K_q=3 questions per image per step.
- LoRA rank 32 on generator's attention + MLP (~50M trainable).
- Gradient checkpointing on generator backbone.
- 100 K samples × 4 epochs × ~2 s/step = 220 H100-hours = **~3.5 days wall-clock on 4×H100** (per `docs/AUX_LOSS_AND_ARCH_DESIGN.md` §A.7 throughput estimate).
- V_sem cache pre-computed once at start.
- Crash resilience: per-step checkpoint + skip-done-keys JSONL.

**M1 hyperparameter table (commit before training; default values from round-3 winners):**

| Hyper | Value | Source / rationale |
|---|---|---|
| Optimizer | AdamW | standard for LoRA fine-tuning |
| LR (generator LoRA) | 2e-5 | conservative; same as LIVR's reported LoRA setup |
| LR schedule | cosine, 5% warmup, decay to 1e-6 | standard SFT recipe |
| Weight decay | 0.0 on LoRA, 0.0 on h | LoRA + soft-prompt convention |
| Batch size (global) | 16 | 4×H100 × B=4/GPU |
| Gradient accumulation | 2 | effective B=32 to stabilize multi-reader signal |
| Total steps | ~25,000 | 100K samples × 4 epochs / B=16 |
| Warmup steps | 1,250 (5%) | standard |
| K (latent count) | 16 | L5 |
| LoRA rank | 32 | L3 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | standard for Qwen2.5-VL fine-tuning |
| LoRA dropout | 0.05 | standard |
| λ_concept | 0.3 | LaViT verbatim, round-3 winner |
| λ_norm | 0.1 (linear ramp 0 → 0.1 over first 1000 steps) | round-2 verified, round-3 confirmed |
| λ_NLL_multi weight ramp | 0.1 → 1.0 over first 1000 steps | curriculum per `docs/AUX_LOSS_AND_ARCH_DESIGN.md` §A.6 |
| K_q | 3 | L9 |
| R | 2 | L8 |
| target_norm μ for L_norm | 57.86 | `results/visual_baseline.json`, POC-4 measurement |
| Stage-1 attention mask duration | first 4 epochs (full M1) | LIVR Stage-1 protocol; Stage-2 unmasked deferred to M2 |
| Mixed precision | bf16 | H100 standard |
| Gradient clipping | 1.0 | standard |
| Random seed | 42 (single seed for M1) | multi-seed deferred to M2 |

**Eval cadence.** Three tiers of cost vs frequency:

| Cadence | Frequency | Cost | What it measures |
|---|---|---|---|
| **Tier 1 (mid-step training metrics)** | every 100 steps | ~0 (logged from forward pass) | L_NLL_multi per reader, L_concept, L_norm, h.norm() mean+median+stdev, h inter-sample cos sim, h intra-block cos sim, train-set greedy accuracy |
| **Tier 2 (held-out + control quick eval)** | every 1,000 steps | ~10 min/checkpoint | held-out NLL on q' under φ₁ + transfer NLL under φ₂ on 500 sample subset; shuffle-(image, q) control; zero_pos + cross_sample_swap steering on 100 sample subset; Δ_C1 (blank gray) and Δ_C3 (adv mismatch) on 200 sample subset |
| **Tier 3 (full 5K stress test + all controls)** | step 12,500 (mid) + 25,000 (end) | ~2 hours/checkpoint | Full 5K stress test (MMVP + NaturalBench + BLINK 7-subtask + MMStar + CV-Bench-3D + POPE-adv + VSR) under all four image-level controls C1-C4; full Coconut-protocol steering; NOTICE-style activation patching; Input-Latent disconnect probe |

Tier 1 is for monitoring training health (catch divergence, mode collapse, NaN). Tier 2 is for tracking the §3 decision criteria mid-training and aborting early if they trend wrong. Tier 3 is for the headline numbers.

**Failure-mode triggers (per `docs/VARIANT_B_GRPO_DESIGN.md` §3.6 framework, adapted for SFT):**

| Failure | Trigger metric | Tier | Action |
|---|---|---|---|
| Off-manifold drift | mean ‖h‖ outside [40, 80] for 1K steps | T1 | Increase λ_norm to 0.3 |
| Mode collapse on h | inter-sample cos sim > 0.8 sustained 1K steps | T1 | Add per-h-position diversity bonus |
| L_concept saturation/collapse | cos(h, V_sem) > 0.95 (collapse to identity) OR < 0.1 by step 2K (no signal) | T1 | Switch MLP bottleneck width or restart with curriculum at λ=1.0 |
| Reader-transfer non-convergence | φ₂ NLL > 4 nat at step 5K | T2 | Add R=3 (VL-Rethinker) — pay throughput cost |
| Random control passes mid-training | Δ(real, shuffled) < 30% at step 5K | T2 | Pivot to §6 framing 2 |
| Held-out divergence | held-out NLL increasing for 3K steps | T2 | LR halve; if persists, abort |

**Pass criteria for "publishable POC paper":**
- Per-skill held-out accuracy ≥ vanilla Qwen2.5-VL-7B baseline on the 5K stress test (no regression)
- POC-3 reader-transfer NLL gap (φ₂ − φ₁) ≤ 1.5 nat (round-2 was +5.8; round-3 target was 2.0; M1 should tighten to 1.5)
- Δ_C3 (adversarial mismatch) **strictly greater** for our method than for vanilla — i.e., our model is *more* sensitive to wrong-image substitution. This is the binding evidence for visual grounding (`docs/EVAL_BENCHMARK_PLAN.md` §C.3)
- Random-control retains ≤30% of real gain (tighter than round-3's 50% bar)
- Coconut steering protocol: zero_pos answer flip rate ≥30% AND cross-sample swap flip rate ≥30%

Failing the C3 criterion is the make-or-break: that's the single number that distinguishes "we trained a method that adds visual content to the latent" from "we trained an effective regularizer." Per arXiv:2004.05704.

#### Milestone M2 — Full training (1 M samples)

**Goal.** Trained reader-grounded generator competitive with LIVR (arXiv:2512.21218) and LaViT (arXiv:2601.10129) — published baselines at this scale. Multi-week run, paper-quality result.

**Two-phase training** (`docs/TRAINING_DATA_PLAN.md` §3.3):

**Phase A — Caption alignment (~500 K samples, 1-2 epochs).** Warm up generator's latent emission against text-caption supervision; bridges cold-start before reader-NLL signal kicks in.

| Dataset | HF path | # samples |
|---|---|---|
| COCO-Caption2017 train (Karpathy) | `jxie/coco_captions` | 500,000 |

**Phase B — Reader-grounded SFT (~1 M samples, 3-5 epochs).**

| Dataset | HF path | # samples | Sample frac |
|---|---|---|---|
| GQA-all | `lmms-lab/GQA` cfg `train_all_instructions` | 400,000 | 40% |
| CLEVR (cauldron) | `HuggingFaceM4/the_cauldron` cfg `clevr` | 70,000 | 7% |
| CLEVR-Math | `dali-does/clevr-math` | 100,000 | 10% |
| TallyQA (cauldron) | `HuggingFaceM4/the_cauldron` cfg `tallyqa` | 80,000 | 8% |
| Visual7W (cauldron) | `HuggingFaceM4/the_cauldron` cfg `visual7w` | 14,000 | 1.4% |
| FineVision selected configs | `HuggingFaceM4/FineVision` (configs: chartqa_*, vqarad, vsr, cocoqa, okvqa, tallyqa, visual7w, iconqa, aokvqa) | 250,000 | 25% |
| VQAv2 (cauldron) | `HuggingFaceM4/the_cauldron` cfg `vqav2` | 80,000 | 8% |
| RefCOCO+/g (auxiliary grounding head) | `lmms-lab/RefCOCO`, `RefCOCOplus`, `RefCOCOg` | 50,000 | 5% |
| AI2D + ChartQA | `HuggingFaceM4/the_cauldron` cfgs | 30,000 | 3% |
| NLVR2 (skip if pair-input not adapted) | `HuggingFaceM4/the_cauldron` cfg `nlvr2` | 25,000 | 2.5% |

**Setup.**
- 4×H100 80GB OR 8×A6000 48GB.
- Phase A: ~140 H100-hours = ~6 days on 1×A6000, ~1.5 days on 4×A6000, ~0.5 days on 4×H100.
- Phase B: ~2,222 H100-hours = ~12 days on 4×H100, ~25 days on 8×A6000.
- LoRA rank kept at 32 (no evidence M1 saturates it; revisit only if loss plateaus visibly with capacity-saturation signature).
- R=3 stretch goal — add VL-Rethinker-7B as third reader if VRAM and time permit.

**M2 deviations from M1** (otherwise reuse M1 hyperparameters):
- 5 epochs (vs M1's 4) on Phase B
- Multi-seed: 3 seeds for M2 final run (M1's single seed insufficient for paper-strength claims)
- Stage-2 attention mask: after 3 epochs of Stage-1 mask, switch to Stage-2 unmasked for the remaining 2 epochs (LIVR's two-stage protocol — Stage 1 plants the bottleneck, Stage 2 lets the model use it freely). Adds ~5% to wall-clock.
- LR floor 5e-7 (lower than M1's 1e-6) — long training requires longer cool-down.
- R=3 (add VL-Rethinker-7B) if VRAM allows; falls back to R=2 with same hyperparams.

**Eval (same as M1) + held-out reader.**
- Reuse the 4-control eval suite, full 5K stress test, latent-intervention probes.
- **Held-out reader test (the new scientific signal at M2):** evaluate on `nvidia/Cosmos-Reason1-7B` — a Qwen2.5-VL-7B sibling that the *generator never saw at training time*. Round-2 left this open; M2 settles it.

**Pass criteria for "published paper-quality result":**
- All M1 criteria, plus:
- Held-out reader (Cosmos-Reason1) transfer NLL gap ≤ 2.0 nat (within 0.5 nat of in-training Monet-7B)
- 5K stress test overall ≥ vanilla Qwen2.5-VL-7B + 3 pp absolute (claim: reader-grounded latents add visual reasoning capacity, not just regularize)
- Per-skill bucket ≥ vanilla on at least 4 of 6 buckets in the stress test (per `docs/EVAL_BENCHMARK_PLAN.md` §B.4)
- Δ_C3 strictly > vanilla on ≥4 of 6 skill buckets

If M2 passes: write paper with M1 as the methodology section, M2 as the headline result.

#### Milestone M3 — Variant B RLVR (optional)

**Goal.** Test whether GRPO + VLPO + multi-reader reward + random-control reward provides additional gain over Variant A's SFT.

**Setup.** From `docs/VARIANT_B_GRPO_DESIGN.md`:
- **VLPO Gaussian on latents:** σ=5.0 (half Monet's 10.0; calibrated to our post-merger embedding norm of ~58 vs Monet's hidden-state norm of ~288). Sweep σ ∈ {2, 5, 10, 20} in pilot.
- **Asymmetric KL anchor:** β=0.04 on latent emission steps (vs frozen base Qwen2.5-VL-7B); β=0 on answer tokens.
- **Multi-reader reward:** R_acc = mean correctness over Φ = {Qwen2.5-VL-7B, Monet-SFT-7B}.
- **Random-control rewards:** shuffle-image (γ=0.5), shuffle-question (γ=0.5), permute-latents (γ=0.5), all with **negative reward** on rollouts where shortcut-encoding would also succeed.
- **Group size G=8** (consensus: VL-Rethinker, VLM-R1, Monet, Visual-RFT all use 8).
- Dataset: GQA + A-OKVQA-MC + NLVR2 + VQAv2-yesno mix, ~100 K samples.
- Trainer: fork **VLM-R1** (`om-ai-lab/VLM-R1`) for the VLPO Gaussian patch; ~200 LOC patch on `_compute_loss`.
- Compute: ~1600 H100-hours on 4×H100 = ~17 days wall-clock for one full training run.

**Pilot phases (decision-gated):**
- **Phase 0** (1 day, 1 GPU-day): trainer plumbing — verify Qwen2.5-VL-7B GRPO runs end-to-end on 1 K subset.
- **Phase 1** (1 day, 5 GPU-days): Option A (deterministic latents, sampled text). Decision rule: held-out POC 2 NLL < 4.0 nat AND POC 3 transfer drop < 8.0 nat. Otherwise abort to Phase 2.
- **Phase 2** (2 days, 10 GPU-days): Option B (VLPO Gaussian) + KL=0.04. Decision rule: as Phase 1 + steering probe shows ≥2 nat penalty for h-zeroing.
- **Phase 3** (4 days, 30 GPU-days): full anti-shortcut suite (multi-reader + random-control rewards). Decision rule: POC 3 transfer drop ≤ 2 nat AND shuffle-I correctness ≤ 20% AND held-out POC 2 NLL ≤ 1.5 nat above oracle.
- **Phase 4** (8-17 days): full 100K × 3 epochs. Only if Phase 3 passes.

**Decision-gate for entering M3:** only run if **(M2 passes) AND (M2 stress-test grounding gap to ceiling ≥ 5 pp on at least 2 skill buckets)**. If M2 already saturates, M3 has no headroom and isn't worth ~1600 H100-hours.

### §4.2 Risks and mitigations

For each major risk: likelihood (low/med/high), impact, trigger (what we'd see), mitigation, time cost of mitigation.

| Risk | Likelihood | Impact | Trigger | Mitigation | Time cost |
|---|---|---|---|---|---|
| **Shortcut basin survives multi-reader + aux losses** | med | high | Round-3 C1 POC-2 held-out NLL > 2.5 nat OR POC-3 transfer drop > 4 nat | (a) Increase R to 3 by adding VL-Rethinker-7B; (b) increase L_concept weight to 1.0; (c) switch to architecture B.3 (Q-Former) which has stronger inductive bias | (a) +1 day for download, +0.5 day re-run; (b) +0.5 day re-run; (c) +1 week for Q-Former implementation, +overnight re-run |
| **Random-control passes (= generic regularization, arXiv:2004.05704 outcome)** | med | very high | Round-3 C5 within 50% of C1's gain; or M1 random-control retains ≥30% of real gain | This is the §6 paper-framing pivot. Reframe as "soft prompts on VLMs are deceptively easy to fit, hard to ground" negative-result paper. M1 still has scientific value; M2 not worth running. | None — pivots paper framing |
| **Reader-transfer drop doesn't shrink enough** | med | high | Round-3 transfer drop > 3 nat OR M1 transfer drop > 2 nat | (a) Increase R; (b) switch L_concept target to averaged-over-readers (Choice A.1.1 (b) instead of (a)); (c) accept "single-reader specific" framing and pivot to within-reader paper | (a) +1 day; (b) +0.5 day; (c) reframes paper but doesn't kill it |
| **LoRA-r=32 saturates** | low | med | M1 training-fit plateaus before step 50K; loss curve shows clear capacity ceiling not noise | Increase LoRA rank to 64 or 128; or switch to full fine-tune of generator's input projection only | +1 day at M1; +week at M2 |
| **Teacher V_sem distribution mismatch** | low | med | L_concept stalls at value > 0.5 (cosine never approaches 0.7-0.9); round-2 manifold target may be wrong scale post-fine-tune | Retrain visual baseline on the actual training image set (`compute_visual_baseline.py` exists, ~1 hour); or switch V_sem source from φ₁ to averaged-over-readers | +2 hours |
| **L_concept collapses h to be identical to natural visual tokens** | low | med | cos(h, V_sem) saturates at 1.0 instead of 0.7-0.9 | Widen MLP bottleneck to D → D → D (no contraction); add per-h-position diversity bonus; method still useful but reduces to "use visual tokens as soft prompts" — defensible in paper but weaker contribution | +0.5 day |
| **Multi-reader R=2 isn't enough structural pressure** | med | high | Round-3 transfer drop on C4 (held-out reader VL-Rethinker) > 4 nat despite C1 transfer ≤ 2 nat | Add R=3 in M1 (VL-Rethinker), accept the ~30% throughput hit | +3-4 days at M1 |
| **GQA gold-token narrowness (oracle ≈ no-input on lex)** | confirmed (high) | low | Already documented in REPORT.md §3 | Switch evaluation NLL to "any of {gold, top-3 acceptable synonyms}" via VQAv2-style 10-answer matching, or switch to LLM-judge for headline metric | Half-day; not on critical path |
| **VLPO σ=5 wrong by an order of magnitude (M3 only)** | unknown | med | Phase 2 of M3 fails on the steering criterion | σ sweep {1, 2, 5, 10, 20} in Phase 2 pilot (designed for this) | Built into M3 plan |

### §4.3 Code milestones

Files to write, in order. One sentence purpose, dependencies on prior files. Round-3 first; M1 follows by reusing/extending. New files only when a genuinely distinct stage emerges.

#### Round-3 POC files (in `experiments/reader-grounded-latent-poc/`)

| # | File | Purpose | Depends on |
|---|---|---|---|
| 1 | `prepare_data_round3.py` | Load GQA-balanced + CLEVR + TallyQA pilot mix (10K samples); filter to ≥3 Qs/image; write `data/round3_samples.jsonl` with `{image_id, image_path, questions: [(q1,y1),(q2,y2),(q3,y3)]}` records | none |
| 2 | `cache_v_sem.py` | Compute and cache `φ₁.visual(x).hidden_states[-1]` for all training images; output `data/v_sem_cache.pt` (~200 MB, ~1 hour on 1 GPU) | data prepared |
| 3 | `model_round3.py` | LIVR-style B.1 generator: same Qwen2.5-VL-7B + K=16 special `<|latent|>` tokens + LoRA r=32 + Stage-1 attention masking (answer cannot attend to image, only to h and q) | none (uses transformers + peft) |
| 4 | `losses_round3.py` | Combined L = L_NLL_multi + 0.3·L_concept + 0.1·L_norm with curriculum schedule (200-step warmup, 600-step main, 200-step anneal) | model |
| 5 | `train_round3.py` | Main trainer: load pilot mix, generator, R=2 readers (Qwen2.5-VL-7B + Monet-7B frozen), V_sem cache, optimize. Per-step JSONL logs of {nll_per_reader, l_concept, l_norm, h_norm_mean, h_inter_cos, train_acc}. Checkpoint every 1K steps. | 1, 2, 3, 4 |
| 6 | `evaluate_round3_heldout.py` | Per-checkpoint: held-out NLL on q' for both readers (the POC-2 / POC-3 protocol unified) | 5 (checkpoint) |
| 7 | `evaluate_round3_transfer.py` | Per-checkpoint: NLL on Monet-7B (in-training reader, the round-3 binding criterion) and Cosmos-Reason1-7B (held-out reader, the §3 C4 criterion) | 5 |
| 8 | `evaluate_round3_steering.py` | Per-checkpoint: zero_pos, gauss_noise, permute_within, permute_across, cross_sample_swap. Reuse `steering_probe.py` shape; extend to also report Coconut-protocol answer flip rate | 5; reuse round-2 `steering_probe.py` |
| 9 | `evaluate_round3_stress.py` | Mid-training (step 5K) + end-of-training: full 5K stress test (MMVP + NaturalBench + BLINK 7-subtask + MMStar + CV-Bench-3D + POPE-adv + VSR) under all four image-level controls C1-C4. (`docs/EVAL_BENCHMARK_PLAN.md` Part B+C.) | 5 |
| 10 | `analyze_round3.py` | Aggregator: bootstrap 95% CIs on each criterion, render plots, decide pass/fail per §3 decision tree, write `results/round3_decision.md` | 6, 7, 8, 9 |

#### M1 files — same names with `_proliferated` suffix where the script genuinely differs

| # | File | Purpose | Depends on |
|---|---|---|---|
| 11 | `prepare_data_proliferated.py` | Load M1 mix (100K samples per `docs/TRAINING_DATA_PLAN.md` §3.2). Streaming-friendly. | round-3 prepare extended |
| 12 | `cache_v_sem_proliferated.py` | Same as round-3 but for M1's image set (~30K distinct images) | (2) |
| 13 | `train_proliferated.py` | Main trainer for M1 (and M2 with config change). Adds: per-1K eval cadence, multi-checkpoint, FSDP across 4 GPUs, gradient checkpointing. | (3, 4) reused |
| 14 | `evaluate_proliferated_*.py` | Same suite as round-3 but on M1's eval splits + the full 5K stress test | (6-9) reused |
| 15 | `analyze_proliferated.py` | Same shape as `analyze_round3.py` but with M1 pass criteria | (10) extended |

If round-3 passes cleanly, M2 reuses M1's files with config changes (only). M3 (Variant B) gets its own directory because it's a different runtime profile (RL trainer fork) — `experiments/reader-grounded-latent-rl/` per directory-structure conventions.

### §4.4 Compute budget

Working from H100 80GB rates (current spot ~$2/H100-h, on-demand ~$3/H100-h). User has on-prem cluster — **provide hours, not $ — but include $ figure for grant/sponsor reporting**.

| Stage | Estimate (H100-h) | Wall-clock on 4×H100 | $ at $3/h spot |
|---|---|---|---|
| Round-3 sweep (5 cells × ~5 H100-h each, sequential per GPU; in parallel on 4 GPUs) | ~100 | overnight (~24 h wall) | $300 |
| M1 medium training (100K × 4 epochs × 2 s/step ÷ 4 GPUs) | ~1300 | ~14 days | $3,900 |
| M2 Phase A (caption alignment, 500K × 1 epoch × 1 s/step) | ~140 | ~1.5 days | $420 |
| M2 Phase B (1M × 4 epochs × 2 s/step) | ~2200 | ~23 days OR ~12 days | $6,600 |
| Eval (full stress test × 8 checkpoints across M1+M2 × 2 hours each × 4 controls × 4 GPUs) | ~50 cumulative | 1 week scattered | $150 |
| **Variant A subtotal** | **~3700 H100-h** | **~6 weeks calendar** | **$11,400** |
| M3 Variant B Phase 0 (plumbing) + Phase 1-3 (decision-gated pilots) | ~50 (pilots only) | ~2 weeks | $150 |
| M3 Variant B Phase 4 (full 100K × 3 epochs, only if Phase 3 passes) | ~1600 | ~17 days | $4,800 |
| **Variant A + B total (full project)** | **~5350 H100-h** | **~10 weeks calendar** | **$16,000** |

**On A6000 (no H100):** roughly 3× wall-clock on 4×A6000 vs 4×H100. M1 becomes ~21 days. M2 Phase B becomes ~70 days on 4×A6000 — too slow; need 8×A6000 (~25 days) or hybrid. (Note `docs/TRAINING_DATA_PLAN.md` §3.3 estimates 23 days on 4×A6000 for M2 Phase B, which is more optimistic than my conservative 70 — depends on whether reader forwards are the bottleneck. Verify with M1 throughput before committing M2.)

**Decision lever:** if 4×H100 is available continuously for 6 weeks, run the conservative path Round-3 → M1 → M2 (Variant A only) for ~3700 H100-hours / $11K. Skip M3 unless M2 reveals headroom. If only 4×A6000 available, plan 10-12 weeks for the same scope; defer M2 Phase A's 500K caption alignment if it's the long pole.

---

## §5 Negative results and how to recognize them

The four tripwires that should make us stop and reframe rather than push through.

### Tripwire 1 — Round-3 random control passes

**Signal.** C5 (shuffled (image, q) pairs) reaches POC-2 held-out NLL within 50% of C1's gain.

**Diagnosis.** Method is generic regularization, not visual grounding. Per arXiv:2004.05704: *"performance improvements are not a result of improved visual grounding, but a regularization effect which prevents over-fitting to linguistic priors."*

**Action.** Reframe paper as "**Soft prompts on VLMs are deceptively easy to fit, hard to ground**" — a methodological cautionary tale. Show that the round-2 mitigations and the round-3 combined recipe both look great in isolation but fail under arXiv:2004.05704-style controls. Cite as evidence that the field needs to standardize random-control reporting. Workshop venue (NoTeMS, Compositional Learning), not main track.

**Don't.** Run M1 — pointless if grounding signal is fake.

### Tripwire 2 — Round-3 reader-transfer fails

**Signal.** Monet-7B NLL stays > 4.5 nat on C1 even with R=2 multi-reader pressure (i.e., transfer drop > 3 nat).

**Diagnosis.** Multi-reader loss insufficient. Either two readers aren't enough structural pressure, or there's no within-readers-shared latent direction that's also visually grounded.

**Action.**
- **Try first:** add R=3 (VL-Rethinker) and rerun the binding cell. If R=3 works, M1 must run R=3 (extra compute cost).
- **If still fails:** reframe paper as "**Reader-specific latents are unavoidable: a structural finding**" — the negative result framing. Lit-context: matches arXiv:2602.22766 (Li et al.'s Latent-Answer Disconnect) for generative VLMs, with our experiment as the empirical demonstration. Pivots to "reader-grounded latents need to be reader-conditional" (a new method, not the M1 method) — but that's a different paper, scoped for next year.

**Don't.** Run M1 expecting the transfer drop to magically shrink at scale.

### Tripwire 3 — M1 stress-test grounding < 5 pp delta vs vanilla

**Signal.** On the 5K stress test (MMVP + NaturalBench + BLINK + MMStar + CV-Bench-3D + POPE-adv + VSR), our M1-trained generator scores within 5 pp of vanilla Qwen2.5-VL-7B-Instruct on 4-of-6 skill buckets.

**Diagnosis.** Method doesn't add visual grounding capacity. Either L_concept is doing nothing (collapsing to the visual-token identity, see Risk row in §4.2) or the latent path is being routed-around in the reader.

**Action.**
- **Diagnose first:** read off the `cos(h, V_sem)` curve from training logs. If it saturates at 1.0, L_concept collapsed → switch to MLP-bottleneck D → D → D (no contraction), retrain.
- **If diagnosis fails:** kill or reframe as "**Capacity is the wrong knob: a study on aux loss saturation in latent visual reasoning**." Workshop venue.

**Don't.** Push to M2 — diminishing returns are clearly diminishing.

### Tripwire 4 — L_concept saturates / collapses early

**Signal.** L_concept value < 0.05 by step 100 of round-3 training, AND `h` becomes effectively a function of the visual encoder output (e.g., `h ≈ MLP(V_sem)` reachable by direct fitting).

**Diagnosis.** Architecture wrong — generator is taking the L_concept path as a shortcut and not learning anything beyond a re-projection. The paper-relevant content (latent reasoning) is not in `h`.

**Action.** Switch to architecture B.3 (Q-Former) with K=32 learnable queries + cross-attention to vision features, BLIP-2-style. This forces the latents to be *abstract* compositions of visual content rather than direct copies. ~1 week implementation cost; reuses the rest of the stack. Confirms the contribution is in the architecture, not just the loss recipe.

**Don't.** Keep training B.1 in this regime — it converges fast but to nothing useful.

---

## §6 Paper venue + framing options

Three framings, ordered from best-case to worst-case.

### Framing 1 — Best case: positive method paper

**Result required.** Round-3 + M1 + M2 all pass criteria. Held-out reader transfer ≤ 2 nat, Δ_C3(ours) > Δ_C3(vanilla) on ≥4 of 6 skill buckets, M2 stress-test ≥ vanilla + 3 pp.

**Venue.** NeurIPS 2026 (May abstract / mid-May full deadline) or ICML 2027 (Feb 2027 deadline; conservative slot).

**Pitch.** "**Reader-Grounded Latent Visual Reasoning: A Multi-Reader Mitigation for Off-Manifold Soft-Prompt Failure.**" Contribution: (a) demonstrate the off-manifold + reader-specific failure mode of naive Variant A (REPORT.md POCs); (b) show that multi-reader NLL + LaViT-style L_concept + multi-Q is a structural fix that survives the arXiv:2004.05704 random-control test; (c) head-to-head against LIVR (arXiv:2512.21218) and LaViT (arXiv:2601.10129) on shared evals.

**Evidence required.** 5K stress test with all four controls per skill bucket; reader-transfer table on Cosmos-Reason1 (held-out); steering protocol from `docs/EVAL_BENCHMARK_PLAN.md` §D.5; activation-patching analysis of where in the reader's forward pass `h` actually matters.

**Deadline.** NeurIPS 2026 abstract registration ~mid-May 2026; full paper ~mid-May 2026 (assume usual NeurIPS schedule). M2 must finish by mid-April for safety margin. Today is 2026-05-02, so M2 must finish by ~2026-04-15 → M2 starts ~2026-03-15 → M1 finishes ~2026-03-15 → M1 starts ~2026-03-01 → round-3 finishes ~2026-02-25 → start round-3 today. Tight but doable.

(Edit: 2026-05-02 today; the timeline above is for ICML 2027 (Feb 2027 deadline). NeurIPS 2026 deadline already passed; targeting NeurIPS 2027 (May 2027) gives 12 months of slack.)

### Framing 2 — Middle case: negative-result / methodology paper

**Result required.** Round-3 passes 3 of 4 criteria, M1 partially passes (e.g., transfer fails but stress-test passes).

**Venue.** NeurIPS 2026 Workshop on Compositional Learning, NoTeMS workshop, or TMLR (rolling deadline).

**Pitch.** Two options depending on which criterion fails:
- If C5 random-control passes: "**Visual Grounding Methods Are Still Working for the Wrong Reasons**" — replication + extension of arXiv:2004.05704 to latent visual reasoning. Headline: even modern multi-reader + LaViT-style aux losses fail the random-control test; the field needs to standardize control reporting.
- If reader-transfer fails: "**Reader-specific latents are unavoidable in reader-grounded methods: a structural finding**." Headline: with R=2 + R=3 readers and combined aux losses, transfer drop bottoms out at ~3 nat and stays there. Either readers are too similar (LMC) or shared structure isn't reachable through this loss surface.

**Evidence required.** Same as Framing 1 but with strong negative-result framing. Workshop paper, 4-8 pages.

**Deadline.** Workshop deadlines typically late summer / early fall; NoTeMS aligns with NeurIPS. Plenty of slack from where we are today.

### Framing 3 — Worst case: technical report

**Result required.** Round-3 fails on multiple criteria; M1 not run (or run and fails).

**Venue.** arXiv preprint + GitHub. No formal submission.

**Pitch.** "**Lessons Learned from a Reader-Grounded Latent POC.**" Document the POC findings, the round-3 attempt, the failure modes, and the implications for future readers attempting similar architectures. Include the POC code so others can reproduce.

**Evidence required.** Round-1, round-2, round-3 numbers; failure-mode diagnosis; concrete recommendations for what wouldn't have worked.

**Deadline.** None; ship when ready. Useful as a research artifact even if not formally published — saves the next person 5 H100-weeks of wasted compute.

---

## §7 Open research questions surfaced by POCs (for paper Discussion section)

From `REPORT.md` §15-§16 + `docs/LITERATURE_RECON.md` + `docs/LITERATURE_MITIGATIONS.md`. Pre-listed so the user has Discussion-section material early. These are *open* — they don't need to be resolved, just acknowledged.

1. **Why does norm regularization at λ=0.1 give a free lunch on held-out?** The training fit doesn't change but generalization improves 36%. The mechanism is presumably "constrains shortcut directions to those that happen to be closer to the natural visual-token manifold," but this is not yet derived theoretically. (REPORT §11, JOURNAL round-2.)

2. **Are reader-grounded latents fundamentally reader-specific?** Round-2 found geometric mitigations don't transfer. If round-3's multi-reader fix also fails partially, the answer is yes. This has implications for any "reader-as-supervision" architecture (LIVR variants, LaViT-style training, distillation-based approaches). Connects to arXiv:2602.22766's Latent-Answer Disconnect.

3. **What is the natural latent count K?** LIVR found K=16 best. Round-2 found K=16 transfer drop smallest. Coconut used K=2-4. Monet uses 10. There's no theory yet for why these values; presumably it's `O(log(reasoning steps))` but the dataset-dependence isn't characterized.

4. **Coconut critique inverted on visual side.** arXiv:2512.21711 found Coconut's text latents are inert placeholders (<5% perturbation success). Our round-2 steering probe found visual reader-grounded latents are causally functional (~30%+ perturbation success). Why the difference? Two hypotheses: (a) text latents have language-prior shortcuts available; visual latents don't have direct visual-prior shortcuts because the reader has no image; (b) reader-NLL gradient is denser in the visual case because the answer's first token is more constrained by visual content than by syntactic context. Worth explicit experimental follow-up.

5. **Is the "multi-Q consistency" mechanism actually consistency, or just data-augmentation regularization?** Round-2 N=2 gave −39%; N=5 gave −24% (different held-out, not strictly comparable). If N→generalization scaling doesn't compound monotonically, it's not "consistency" in the inductive-bias sense — it's data augmentation. Round-3 K_q=3 vs K_q=5 ablation should disambiguate.

6. **Layer-mixing readers** (`docs/AUX_LOSS_AND_ARCH_DESIGN.md` §C.3): Linear Mode Connectivity gives us cheap reader diversity, but pure LoRA-merge readers are too close to φ₁ to provide structural pressure. What's the diversity threshold (in `‖ΔW‖` or behavioral distance) below which LMC-merged readers stop providing useful multi-reader signal?

7. **Capacity-vs-shortcut paradox.** K=16 had *less* transfer drop than K=1 (5.77 vs 8.97) — more capacity helps reader-portability, contra naive expectation. The mechanism is presumably "extra capacity gets spent on diffuse / generic structure that's more reader-shareable." But we don't have a theory yet.

8. **GQA's gold-token lexical narrowness** (REPORT §3 caveat): `oracle_nll ≈ no_input_nll` because the image often makes a synonym more likely than the literal gold. Switching evals to LLM-judge or VQAv2-style 10-answer matching would change the optimization landscape. M2 should run with both.

---

## §8 Quick-start: First day after this morning

A concrete sequence. Each step ~30 minutes; the full sequence takes one work day.

### Step 0 — Read this doc + answer §2

Time: ~30 minutes. Mark each D-question with `[decided: <choice>]`. If any answer differs from the recommendation, append a one-line rationale.

### Step 1 — Confirm/override §1 locked choices

Time: ~15 minutes. If any of L1-L12 don't sit right, flag now — round-3 has them baked in.

### Step 2 — Trigger compute reservation

Time: ~30 minutes. Reserve 4×H100 for 24 hours starting tomorrow night for round-3 sweep. Reserve 4×H100 for 14 days starting in 4 days for M1 (if D2 confirms).

### Step 3 — Run data prep + V_sem cache

Time: ~3 hours wall, mostly waiting. From `experiments/reader-grounded-latent-poc/`:

```bash
# Activate the experiment's venv (uv-managed)
cd /mnt/ssd/Projects/research-pilots/experiments/reader-grounded-latent-poc

# Step 3a: extend the existing GQA pilot to multi-Q at K_q=3
uv run prepare_data_round3.py
# Output: data/round3_samples.jsonl (~10K rows, ~3 Qs/image)

# Step 3b: cache V_sem for all training images
uv run cache_v_sem.py
# Output: data/v_sem_cache.pt (~200 MB, ~1 hour on A6000)
```

Both files do not yet exist — write them per `§4.3` (#1, #2). They are the smallest files of round-3 and are the right place to catch any data-pipeline bugs early.

### Step 4 — Implement model + losses + smoke test

Time: ~1 work day. From `experiments/reader-grounded-latent-poc/`:

```bash
# Write model_round3.py (B.1 architecture, LIVR-style, LoRA r=32, Stage-1 mask)
# Reuse slot-injection patch from tune.py. Add LIVR-style attention mask to the forward.

# Write losses_round3.py (combined L per §A.6 of design doc)

# Smoke test: run train_round3.py for 100 steps on 100 samples, verify
# - L_NLL_multi decreases monotonically
# - cos(h, V_sem) increases over the warmup
# - h.norm()'s mean trends toward 57.86
# - No NaNs, no OOM at B=4 on a single GPU
uv run train_round3.py --smoke-test
```

If smoke test passes, kick off the full 5-cell sweep overnight on 4×H100.

### Step 5 — Schedule check-in

Time: ~10 minutes. Schedule a 3-day check-in (2026-05-05 morning) to review round-3 results. By then:
- 5-cell sweep complete
- `analyze_round3.py` outputs `results/round3_decision.md` with bootstrap CIs on each criterion
- Decision: proceed to M1, pivot to alternative architecture, or kill

If round-3 passes, kick off M1 the same afternoon.

---

## §9 Appendix — quick reference

### §9.1 Files already on disk and reusable

From the existing experiment:
- `tune.py` — POC 1 driver (slot-injection primitive; reusable for round-3)
- `evaluate_held_out.py` / `evaluate_transfer.py` / `steering_probe.py` — round-2 eval scripts; round-3 versions extend these
- `compute_visual_baseline.py` — POC 4 reference (V_sem natural distribution, μ‖·‖=57.86)
- `analyze.py` — aggregator (auto-regenerates `results/ANALYSIS.md` + figs); round-3 variant extends
- `results/visual_baseline.json` — pre-computed natural visual-token statistics for the POC images

### §9.2 Reader weights on disk

- `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct` (~14 GB, base reader φ₁)
- `/mnt/ssd/Projects/research-pilots/experiments/monet-latent-probe/data/Monet-7B` (symlink, ~14 GB, sibling reader φ₂)
- For R=3 stretch: download `TIGER-AI-Lab/VL-Rethinker-7B` (~14 GB; see `docs/AUX_LOSS_AND_ARCH_DESIGN.md` §C.2)
- For M2 held-out reader: download `nvidia/Cosmos-Reason1-7B` (~14 GB)

### §9.3 Citation cheat sheet (arXiv IDs verified in source docs)

Method/architecture priors:
- LIVR — arXiv:2512.21218
- LaViT — arXiv:2601.10129
- Monet (VLPO) — arXiv:2511.21395
- BLIP-2 / Q-Former — arXiv:2301.12597
- LVR — arXiv:2509.24251
- Coconut — arXiv:2412.06769

Failure-mode lit:
- arXiv:2602.22766 — Li et al., Input-Latent / Latent-Answer Disconnects, CapImagine
- arXiv:2512.21711 — "Do Latent Tokens Think?", Coconut critique
- arXiv:2402.09063 — embedding-space attacks (round-1 failure mode kin)
- arXiv:2504.02144 — interpretability-vs-task trade-off

Eval-grounding rigor:
- **arXiv:2004.05704 — Shrestha et al., "Visual Grounding Methods for VQA are Working for the Wrong Reasons!"** (the load-bearing reference for §1 L12 random-control mandate)
- arXiv:2401.06209 — MMVP
- arXiv:2403.20330 — MMStar
- arXiv:2404.12390 — BLINK
- arXiv:2410.14669 — NaturalBench
- arXiv:2305.10355 — POPE
- arXiv:2406.16320 — NOTICE (activation patching for VLMs)
- arXiv:2309.16042 — Heimersheim & Nanda, activation patching best practices

RLVR landscape (Variant B, M3 only):
- arXiv:2503.07536 — LMM-R1
- arXiv:2504.07615 — VLM-R1 (the trainer fork target)
- arXiv:2504.08837 — VL-Rethinker (SSR + Forced Rethinking)
- arXiv:2510.23925 — LaCoT (RGFN; "GRPO=SFT on latent reasoning" finding)

### §9.4 Decision-tree summary (one-page)

```
[Round-3 sweep complete; 5 cells]
        │
        ▼
[C1 POC-2 held-out NLL ≤ 2.0?]── no ──► [diagnose: L_concept saturated? → tripwire 4]
        │ yes
        ▼
[C1 POC-3 transfer drop ≤ 2.0 nat?]── no ──► [add R=3 → re-run; still fails → tripwire 2]
        │ yes
        ▼
[C5 random-control gain ≤ 50% of C1?]── no ──► tripwire 1 (negative-result reframe)
        │ yes
        ▼
[Steering probe: zero_pos ≥ 1 nat?]── no ──► [diagnose curriculum]
        │ yes
        ▼
[Proceed to M1 (3.5 days, 1300 H100-h)]
        │
        ▼
[M1 stress-test grounding ≥ vanilla + 5pp on ≥4 buckets?]── no ──► tripwire 3
        │ yes
        ▼
[M1 transfer drop ≤ 1.5 nat? Δ_C3(ours) > Δ_C3(vanilla)?]── no ──► framing 2 (workshop)
        │ yes
        ▼
[Proceed to M2 (12 days, 2200 H100-h)]
        │
        ▼
[M2 held-out reader transfer ≤ 2.0 nat? +3pp on stress test?]── no ──► framing 2
        │ yes
        ▼
[Write paper as framing 1 (NeurIPS / ICML positive method)]
        │
        ▼
[Headroom on stress-test grounding ≥ 5pp left?]── no ──► [skip M3]
        │ yes
        ▼
[Run M3 Variant B RLVR (decision-gated phases)]
```

---

**End of plan.** Total length ~800 lines as specified. Read once, mark §2 decisions, start §8.
