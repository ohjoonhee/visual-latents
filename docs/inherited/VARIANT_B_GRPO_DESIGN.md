# Variant B (GRPO/RLVR) — Design Document

**Date:** 2026-05-02
**Scope:** Concrete design for the RL variant of reader-grounded latent visual reasoning. Companion to `REPORT.md` (POC of Variant A — differentiable reader-NLL) and `docs/LITERATURE_RECON.md` / `docs/LITERATURE_MITIGATIONS.md` (background lit recon).
**Status:** design only; no Variant B compute spent yet.

The Variant A POC found that direct, per-sample optimization of `h_{1:K}` against frozen-reader NLL produces shortcut-dominant, off-manifold, reader-specific latents (`REPORT.md` §3-§7). Variant B asks: does on-policy GRPO sampling, with a verifiable rule-based reward and a KL anchor, escape the shortcut basin Variant A fell into — or does it find the same shortcut by stochastic search instead of gradient descent?

The honest prior (`REPORT.md` §16): shortcut basin reachable identically; the embedding-space optimization pressure is slightly less aggressive (sampling rather than direct optimization), but the manifold and transfer concerns are identical. **This document specifies what Variant B has to look like to be informative beyond Variant A** — i.e. what reward/KL/dataset choices give it a real chance of breaking the shortcut, and what the failure modes are.

---

## 0. The structural problem Variant B has that ordinary VLM-RLVR doesn't

Standard VLM-RLVR (VLM-R1, Visual-RFT, GLM-4.5V, Vision-R1, LMM-R1) optimizes the policy over **discrete text tokens** with an explicit categorical density. Group-relative policy gradient is straightforward: the importance ratio `π_θ(o_{i,t}) / π_θ_old(o_{i,t})` is well-defined per token; KL to a reference policy is a per-token KL between categorical distributions.

Variant B's policy emits **continuous embeddings `h_{1:K}` followed by text answer tokens**, all in one autoregressive trajectory. There is no native categorical density on `h`. **Monet's VLPO (arXiv:2511.21395) is the only published RL method for continuous latent visual policies**, and it solves this by fiat:

> "π_θ(h^{i,t}_old | Q, I, o_{i,<t}) = exp(-1/(2σ²) ‖h^{i,t}_old − h^{i,t}_θ‖² − const)"
> with σ = 10.0 (Monet Table 8, arXiv:2511.21395).

i.e. treat the policy at each latent step as a Gaussian centered on the policy-net's mean output, with fixed σ; the importance ratio collapses to `exp(-‖Δh‖² / 2σ²)`, and the policy gradient is exactly what you get from REINFORCE on a Gaussian-actor continuous-action MDP. Monet found that **vanilla GRPO does not improve their SFT model** (because GRPO only updates text-token log-probs, not the latent emissions); VLPO's fix is to bring the latents back into the loss.

**Implication for Variant B.** We cannot just plug into a stock VLM-GRPO trainer (TRL/verl/OpenRLHF) and call it RL on latents. The trainer will compute the policy gradient for the text answer tokens only; the gradient signal that flows back into the latent-emitting hidden states is *exactly the same* as in Variant A (reader-NLL, reweighted by GRPO advantage), with the latents themselves treated as deterministic intermediate state. The "extra exploration" provided by GRPO in standard RLVR settings is lost — **without VLPO-style Gaussian reparameterization the only stochasticity is in the answer-token sampling, not in `h`**.

This is the single most load-bearing design decision for Variant B and is taken up in §2.

## 1. RLVR-for-VLMs landscape — May 2026 snapshot

Goal: ground our design in what's verifiably published, including reward formulae, KL coefficients, group sizes. References that matter, with arXiv IDs and the specific verbatim numbers we need.

### 1.1 Method-by-method

**LMM-R1 (arXiv:2503.07536, Mar 2025)** — Two-stage rule-based RL on Qwen2.5-VL-Instruct-3B. Stage 1 *FRE* (Foundational Reasoning Enhancement) is text-only RL on math/code. Stage 2 *MGT* (Multimodal Generalization Training) shifts to multimodal data. +4.83 % avg on multimodal, +4.5 % on text-only, +3.63 % on a Football-Game generalization task. Reward: rule-based correctness on math/code answers; format reward for `<think>...</think><answer>...</answer>` structure. Specific KL/group-size numbers not extractable from the abstract; the architecture is "GRPO with rule rewards", same general shape as DeepSeek-R1.

**VLM-R1 (arXiv:2504.07615, Apr 2025; om-ai-lab)** — The reference open-source VLM-RLVR codebase. Confirmed config from the paper:
- Tasks: REC (referring expression comprehension), OVD (open-vocabulary detection), classification.
- Rewards: `R^rec_acc(q,o) = IoU(b*, b_pred)`; `R^ovd_acc(q,o) = s_ovd · mAP(b_pred, b_gt)`; format reward (1 or 0) for valid JSON.
- **KL β: 0.04 for REC, 0 for OVD**.
- **Group size N = 8.**
- Temperature 0.9, 1 GRPO iteration per batch.
- **Anti-reward-hacking: `odLength` reward** — explicitly added to penalize models that generate excessive bbox predictions to game mAP. Triggers an "OD aha moment" where the model first reasons about object presence before predicting boxes. **This is the most directly transferable anti-hacking trick to our setting** — see §3.7.
- Compute: not disclosed in paper; community reports ~2-3 days on 8×A100 for 7B REC.

**Vision-R1 (arXiv:2503.18013, Mar 2025; not 2503.07536, which is LMM-R1)** — Human-Free Alignment via Vision-Guided RL. Cold-start initialization + RL with reasoning incentivization. Different paper from LMM-R1 despite similar timing.

**Visual-RFT (arXiv:2503.01785, Mar 2025; ICCV 2025)** — Liu et al. The first systematic RFT-on-VLMs paper. Reward design per task:
- Detection: `R = R_IoU + R_conf + R_format` (mean IoU over matched boxes; high-confidence reward; format reward).
- Classification: `R = R_acc + R_format` (binary correctness + format).
- Grounding: IoU-based.
- Base: Qwen2-VL-2B/7B. Datasets: Flower102, Pets37, FGVC-Aircraft, Cars196 (classification); COCO-8/LVIS-6/MG (few-shot detection); LISA-239 (grounding); COCO-65 (OVD).
- **Reports +24.3 % on 1-shot fine-grained classification with ~100 samples** — i.e., RFT works at very small data scale, the regime closest to ours.
- Compute and KL/group-size: not disclosed in paper; codebase uses standard GRPO config.

**VL-Rethinker (arXiv:2504.08837, Apr 2025; NeurIPS 2025)** — Two specific anti-pathology techniques worth copying:
- **Selective Sample Replay (SSR).** Maintains a replay buffer of `(x, y_i, Â_i)` tuples *only* for queries whose group had non-zero advantages. During training, samples are drawn from the buffer with probability `∝ |Â_j|^α`. This prevents the "vanishing advantages" problem where, late in training, most queries are either all-correct or all-wrong (Â=0 across the group, no gradient).
- **Forced Rethinking trigger.** Append `"Wait, does it seem right?"` (or `"Wait, there might be a mistake"` / `"Wait, let's double check"`) to the rollout to elicit one self-reflection step.
- Reward: binary correctness (1/0).
- **Group size G = 8.**
- Datasets: ~38 870 queries from public + web; 16 K subset for 7B; 20 K for 32B/72B.
- Base: Qwen2.5-VL-Instruct (7B/32B/72B).
- Headline: 80.3 % MathVista, 61.8 % MathVerse, 43.9 % MathVision (7B).

**LaCoT (arXiv:2510.23925, Oct 2025) — anti-reward-hacking, diversity-seeking RL.** This is the paper our brief flagged as the diversity-control sibling, and its findings are directly relevant.
- Frame: latent CoT as posterior inference; objective = amortized variational inference (GFlowNet-style).
- **Drops the KL penalty entirely.** Replaces it with a *reference-guided filter*: an indicator `𝕀(Z_i) = 1 if R(Z_i) > δ_s · R(Z_ref) else 0`, with `δ_s` annealed from lenient to strict over 50 steps. Only above-threshold trajectories backpropagate.
- Sparse token-level reward via interpolation: `R̃(z_{1:t+i}) = R(z_{1:t}) + (i/λ)(R(z_{1:t+λ}) − R(z_{1:t}))` with λ=8.
- Diversity emerges from the GFlowNet objective sampling trajectories ∝ reward, not from an explicit diversity bonus.
- Base: Qwen2.5-VL-3B / 7B.
- **Compute: 8×80GB GPU-node, ~30 h SFT + ~120 h GRPO/RGFN.**
- Numbers (Qwen2.5-VL-7B, MathVista): zero-shot 63.7, SFT 62.7, **GRPO 62.6** (essentially flat over SFT), **RGFN 68.4** (+5.7 over GRPO).
- **Explicit anti-reward-hacking framing.** The verbatim claim: *"diverse, high-likelihood latent CoT, overcoming deterministic sampling limitations and avoiding reward hacking."*
- Note: **GRPO underperformed even SFT** in their setting. They attribute this to (i) the KL penalty preventing exploration and (ii) reward-model bias. Both apply to us.

**Monet's VLPO (arXiv:2511.21395, CVPR 2026)** — already covered in `LITERATURE_RECON.md`. Re-stated for completeness:
- Gaussian reparameterization on latents with σ=10.0; importance ratio `exp(-‖Δh‖² / 2σ²)`.
- Reward: accuracy reward (1 correct / 0 incorrect) + format reward for `\boxed{}`.
- **Rollout size 8** (Table 8); 1 epoch on 3.2 K subset of Thyme-RL.
- KL β not extractable from the table.
- Base: Qwen2.5-VL-7B (Stage-3 SFT model, not the original base).
- VLPO outperforms GRPO and SFT-only on latent-using OOD tasks; vanilla GRPO does *not* improve over SFT for them.

**Cosmos-Reason1 (NVIDIA, arXiv:2503.15558)** — Qwen2.5-VL architecture, 7B. SFT then RL on physical-common-sense / embodied-reasoning data. RL adds another +5 % on top of +10 % SFT for a 65.7 average score on robotics/AV benchmarks. RL training requires ≥4 GPUs ≥80 GB; specific algorithm/KL/group-size not in repo docs (defers to paper). Less directly applicable to our setting because the reward structure (physical plausibility) is task-specific.

**GLM-4.5V (arXiv:2507.01006, Jul 2025; Z.ai)** — RL with Curriculum Sampling (RLCS). Multi-domain unified reward across STEM, OCR, grounding, charts, GUI, video; "explicit box markers for final answer extraction" + "domain-specific reward logic with shared verification functions." Combines RLVR + RLHF. Production-scale; less directly transferable but proves multi-domain RLVR-on-VLMs works at GLM scale.

**GTR (arXiv:2503.08525, Mar 2025; ICCV 2025)** — Anti-"thought-collapse" recipe for RL'd VLM agents. Key insight directly relevant to Variant B: when reward is purely outcome-based, the model's CoT *collapses* to short, state-irrelevant text that doesn't help the action policy. GTR adds an off-the-shelf VLM as a thought-corrector and trains thought tokens via SFT while training action tokens via PPO. **Their thought-collapse phenomenon is a structural cousin of our shortcut concern**: the model finds the shortest path to the verifiable reward, and the intermediate "thinking" channel decays to noise.

**Vision-SR1 (arXiv:2508.19652, Aug 2025)** — Self-rewarding via reasoning decomposition. Decomposes VLM reasoning into (a) visual perception (must be self-contained — re-prompt the *same* model with only the visual description to recover the answer; this is the "visual reward"), (b) language reasoning (correctness on the final answer). **The visual-reward construction is structurally a sibling of the multi-reader / multi-question idea** in our `REPORT.md` §16: it forces the visual representation to be detachable from the original image so that another decoding pass (the same model in a different prompt) can use it. We should adopt the equivalent: **reward only the latent's portability**, not its match to ground truth on the original prompt.

### 1.2 Summary table — anti-shortcut tricks observed in published VLM-RLVR

| Method | Trick | Mechanism |
|---|---|---|
| VLM-R1 | `odLength` reward | Penalize over-prediction (length-mismatch with GT) |
| VL-Rethinker | SSR + Forced Rethinking | Replay buffer for non-zero-advantage groups; injected self-doubt token |
| LaCoT | RGFN + drop KL | GFlowNet objective; reference-guided filter replaces KL |
| GTR | Thought corrector | Off-policy supervisor for thought tokens |
| Vision-SR1 | Visual self-reward | Re-prompt with description-only; reward portability |
| Monet | VLPO Gaussian | Latents in policy gradient via Gaussian density |

For our Variant B, the tricks worth combining: **VLPO Gaussian (required for any latent-side gradient)**, **odLength-style reward shaping (penalize answer-only, no-visual-content shortcuts)**, **Vision-SR1-style portability reward (the multi-reader / held-out-q analogue from `REPORT.md` §16)**, **Forced Rethinking** as a cheap probe for whether the latent does anything at all.

## 2. GRPO trainer landscape — May 2026 snapshot

Goal: pick the trainer for Variant B with eyes open about VLM-specific maturity and our continuous-latent extension.

### 2.1 trl (HuggingFace, v1.3.0 as of Apr 2026)

**Status: production for VLMs, but only for standard text-output VLMs.** Verified from the official docs:

> "GRPO supports training Vision-Language Models (VLMs) on multimodal datasets containing both text and images. Tested with: Gemma3, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL, SmolVLM2."
> ([trl GRPOTrainer docs, v1.3.0](https://huggingface.co/docs/trl/grpo_trainer))

Key facts:
- **Default β = 0.0** (KL term disabled), explicitly motivated by Open-Reasoner-Zero and DAPO findings. Setting β > 0 is the override.
- Custom reward functions take `prompts`, `completions`, `completion_ids`, `**kwargs` (incl. dataset columns) and return list of floats. Async supported. Returning `None` makes the function inapplicable to that sample (multi-task pattern).
- vLLM colocate or server mode for rollout generation.
- Truncated/Masked Importance Sampling for vLLM training/inference mismatch.
- LoRA examples for vision-projection layers (`--use_peft --lora_target_modules "q_proj","v_proj"`).
- Example script: `examples/scripts/grpo_vlm.py`.
- **GRPOConfig has a `scale_rewards` knob** ("none" / "group" / "batch"); `loss_type="sapo"` available for per-step asymmetric clipping.

**Our blocker:** TRL's GRPOTrainer assumes the policy emits text tokens with a categorical density. **No support for continuous latents in the policy gradient.** To make Variant B work in TRL we would have to either (a) keep the latents deterministic within the rollout (so reward shapes only the answer-token gradient, with the latents responding via the chain rule — same regime as Variant A, just stochastic over answer tokens), or (b) fork the trainer to add VLPO-style Gaussian reparameterization on the latent-emission steps. (a) costs nothing and matches `REPORT.md` §16's "Variant B inherits Variant A's shortcut basin" reading. (b) is the real Variant B and requires patching `_compute_loss` (~200 LOC).

### 2.2 verl (volcengine/Volcano Engine, HybridFlow paper EuroSys 2025)

**Status: production for VLMs, broader algorithm coverage than TRL.** Supports PPO, GRPO, ReMax, REINFORCE++, RLOO, PRIME. VLM examples ship for Qwen2.5-VL-7B, Qwen3-VL-8B, Qwen3-VL-30B-A3B, and a 235B-A22B Megatron variant. Default vision script trains on `geo3k`. Backends: FSDP and Megatron-LM, with vLLM/SGLang for rollouts.

Pros vs TRL: better scaling story (Megatron + FSDP + multi-node mature); more algorithm options if we want to compare GRPO vs RLOO vs REINFORCE++; community-tested at >7B VLM scale.

Cons vs TRL: heavier learning curve; reward-function customization less ergonomic (config-heavy YAML, not a Python callable handed to a trainer); same continuous-latent blocker — the trainer assumes text-token policy.

### 2.3 OpenRLHF

**Status: production for VLMs as of v0.10 (Apr 2026).** Ray-based, Async RL. Supports VLM RLHF end-to-end with image inputs (`--data.image_key`, `--data.max_images_per_prompt`). Multi-turn VLM RL with images in environment feedback. Algorithms: PPO, DAPO, REINFORCE++, GRPO, RLOO. Same continuous-latent blocker.

LMM-R1's reference implementation is a fork of OpenRLHF, so OpenRLHF has the most-tested codebase for multimodal-R1-style RL.

### 2.4 Other

- **TRLX** — effectively deprecated; not maintained for current models.
- **VLM-R1 (om-ai-lab)** — codebase wraps DeepSpeed + GRPO; has the `odLength` reward implemented; supports Qwen2.5-VL-3B by default. Smallest abstraction overhead for VLM-specific tricks; closest to "research scaffolding" of the listed options.
- **Unsloth GRPO + VLM** — exists; LaTeX-OCR / visual-grounding examples; 4-bit quantization for memory; LoRA-only training. Throughput-tuned, less flexible.

### 2.5 Recommendation for Variant B

**Pre-validate the design with TRL's `grpo_vlm.py` on Qwen2.5-VL-3B + a baseline GRPO on a no-latent text-output formulation first** (~half-day), to verify the dataset/reward pipeline. Then **fork either VLM-R1 or verl** to add VLPO-style Gaussian reparameterization on latent emission steps (~3 days for a working prototype). The fork-target choice depends on whether we want (a) tight, hackable code (VLM-R1) or (b) production-grade scaling (verl). For a POC at 7B and a few hundred K samples, **VLM-R1 is the right call** — its codebase already has the `odLength` reward we want to mirror, and the patches are smaller.

Note: **none of the trainers above natively support continuous-latent policy gradient.** Whichever we pick, the VLPO-style fork is unavoidable if Variant B is to be more than "Variant A with stochastic answer-token sampling."

## 3. Concrete Variant B design

The setup, restated and committed:
- **Generator** π_θ: Qwen2.5-VL-7B (trainable). Emits K continuous latents `h_{1:K} ∈ ℝ^{K × 3584}` autoregressively at the visual-token slot via Monet's start-token + hidden-state-feedback protocol (`a.md`).
- **Reader** φ: frozen copy of generator's base, **no image input** (strict reader-grounded constraint, locked in `JOURNAL.md` 2026-05-01).
- **Trajectory τ = (h_{1:K}, ŷ_1, ..., ŷ_T)**: latents emitted, then text answer tokens decoded by the same policy π_θ but evaluated under φ at scoring time.
- **Reward** R(τ): a verifiable rule-based scalar; specifics in §3.1.
- **GRPO group size G**: §3.2.
- **KL anchor**: §3.3.
- **Dataset**: §3.4.

### 3.1 Reward function

The choice of reward is the load-bearing decision for whether Variant B encodes visual content vs. shortcut-encodes the answer. From the literature:

- **Pure exact-match** (Visual-RFT classification, Monet, VL-Rethinker, GLM-4.5V): `R = 1 if extracted_answer == y* else 0` (with `\boxed{}` extraction). Cheapest. Deterministic. Ground-truth-friendly for GQA/CLEVR/A-OKVQA-MC. **This is what our Variant A POC already uses implicitly via the NLL gradient.**
- **LLM-judge**: GPT-4o-mini call per rollout. ~$0.05 / sample at our scale. For 100 K-sample × 8-rollout × 3-epoch training that's 2.4 M calls = ~$120 K. **Not feasible** at our compute budget. Skip unless restricted to a few-K eval set.
- **Hybrid**: exact-match first, fallback to judge on string-mismatch. For GQA's narrow gold-token regime where `oracle_nll ≈ no_input_nll` (`REPORT.md` §3 caveat) this is meaningfully better than pure exact-match — many "wrong" greedy outputs are synonyms of GT. ~10-30 % of rollouts hit the fallback at our scale → cost reduction proportional. Still ~$12-36 K. **Borderline feasible for a 10 K-sample fine-tune; not for full-scale.**
- **Length / format reward**: `R_format = 1 if completion matches "<think>...</think><answer>...</answer>" else 0`, weighted at 0.1-0.2 of the accuracy reward. Standard from DeepSeek-R1, every VLM-RLVR paper above uses one. **Mandatory; cheap.**

The **anti-shortcut layer** is what distinguishes our reward design from off-the-shelf RLVR.

#### 3.1.1 Random-control reward (the explicit anti-shortcut term)

Construct three additional sample types per training query:
1. **Shuffle-image control**: same `(q, y*)`, random unrelated image `I'`. Run the rollout normally. **Reward: `R_shuffle = -|R_real|`** — i.e., we *negatively* reward correctness on the wrong image. A shortcut policy that encodes `(q, y*)` into `h` regardless of image will be equally correct under this control and get punished. A policy that encodes visual content from `I` should be wrong under `I'` → no penalty.
2. **Shuffle-question control**: same `I`, random unrelated question `q'` from a different image. Symmetric construction.
3. **Permuted-latents control**: same `(I, q, y*)`, but at scoring time the K latents are permuted before the reader sees them. **Reward: `R_permute = -1 * (correct after permute)`**. A shortcut policy whose answer is recoverable from any permutation gets punished; a position-aware policy is unaffected.

Final reward:
```
R(τ) = α · R_acc + β · R_format
       + γ_shuffle_I · R_shuffle_I
       + γ_shuffle_q · R_shuffle_q
       + γ_permute · R_permute
```
with `α = 1.0`, `β = 0.1`, `γ_shuffle_* = 0.5`, `γ_permute = 0.5`. The `γ` terms are computed on a *different rollout* of the same query (cheap — one extra forward pass per sample, no extra generation).

The control rewards are inspired by the cautionary finding in arXiv:2004.05704 ("Visual Grounding Methods for VQA are Working for the Wrong Reasons!") — many grounding objectives improve VQA via generic regularization, and **random/insensible cues produce similar improvements**. Our random-control rewards directly test this: if the policy learns visual content, controls hurt; if it learns shortcuts, controls help equally and the negative reward dominates.

This is **directly analogous to VLM-R1's `odLength`** (arXiv:2504.07615) which penalizes over-prediction to game mAP: explicitly identify the shortcut you fear and price it in.

#### 3.1.2 Multi-reader portability reward (the structural anti-shortcut term)

Per `REPORT.md` §16, the round-2 mitigations did not fix reader transfer (`Mit-B λ=0.1` Monet-7B NLL = 8.55, `Mit-A r=1` = 8.30, vs vanilla 8.09). The recommended fix was multi-reader optimization at training time. In Variant B this becomes:

```
R_acc(τ) = (1/|Φ|) · Σ_{φ ∈ Φ} 𝟙[argmax φ(· | q, h_{1:K}) = y*]
```
with `Φ = {Qwen2.5-VL-7B, Monet-SFT-7B}` (both already on disk per `HANDOFF.md`). A latent must be readable by both readers simultaneously to get full reward. This is the GRPO equivalent of `REPORT.md` §16's `Σ_φ NLL_φ` proposal — the policy gradient pressure favors latents whose direction lives in the *intersection* of φ_1 and φ_2's input-embedding "readable" cones.

Cost: one extra forward pass per rollout under φ_2. Memory: φ_2 frozen, no gradient — fits alongside φ_1 (the reader is also frozen) and π_θ on 4×H100.

### 3.2 On-policy h sampling

This is where Variant B is structurally different from Variant A and where the design has to commit to either (a) deterministic latents + sampled text, or (b) VLPO-style Gaussian on latents.

#### 3.2.1 Where does noise come from?

**Option A — deterministic latents, sampled text answer.**
- π_θ emits `h_{1:K}` greedily (same as inference). Then samples `ŷ_1, ..., ŷ_T` from the answer-token distribution at temperature τ_text = 1.0.
- Group of G rollouts differs only in answer-token sampling, given identical `h`.
- Policy gradient: GRPO advantage × log-probability of answer tokens. **Latents updated only via the chain rule from answer-NLL gradient — exactly the Variant A regime, just scaled by GRPO advantage instead of raw NLL.**
- Verdict: this is what stock TRL/verl/OpenRLHF give us. Per `REPORT.md` §16: "the shortcut basin is reachable identically" under this. Useful as a baseline for what *not-VLPO* gets us. **Not a real test of Variant B.**

**Option B — VLPO-style Gaussian on latents (the real Variant B).**
- π_θ emits a deterministic mean `h_{1:K}^μ`; the rollout latent is `h_{1:K} = h_{1:K}^μ + σ · ε`, `ε ~ N(0, I)`.
- σ: start with **σ = 5.0** (half Monet's σ=10.0; Monet's latents are last-layer hidden states with norm ~288, ours are post-merger embeddings with norm ~58 → scale σ proportionally). Sweep σ ∈ {2, 5, 10, 20} in the pilot.
- Importance ratio for the latent step: `exp(-‖h_old - h_θ^μ‖² / 2σ²)` (Monet eq. 8).
- Group of G rollouts samples G different ε's → G different `h` → G different greedy reader answers. The randomness is *injected at the latent* rather than at the answer.
- Policy gradient: standard GRPO with the augmented importance ratio. The latents are now first-class actions; their gradient is the GRPO advantage × `(h_old - h_θ^μ)/σ²` (REINFORCE on Gaussian).
- Verdict: **this is the only formulation that gives Variant B a structurally different loss landscape from Variant A.** Required.

The pilot must run **both Option A and Option B** on the same dataset/reward to attribute any improvement to the latent-side gradient vs. just the answer-token sampling.

#### 3.2.2 Group size G

Published values for VLM-GRPO: VLM-R1 G=8; VL-Rethinker G=8; Monet rollout=8; Visual-RFT typically 8; GLM-4.5V production-scale (unspecified, ≥16). **G=8 is the consensus.** At G=4 the advantage estimate is noisy; at G=16 the GPU memory cost roughly doubles (per-GPU) without the empirical literature reporting big gains.

**For us, G=8 is the right starting point.**

Memory cost at 7B + bf16 + G=8 + K=4 latents: π_θ forward+backward (15 GB), φ_1 frozen forward (8 GB), φ_2 frozen forward for multi-reader reward (8 GB) — comfortable on 80 GB H100 with FSDP across 4 GPUs.

#### 3.2.3 Diversity reward (optional, per arXiv:2510.23925)

LaCoT explicitly encourages trajectory diversity within a group as the anti-reward-hacking lever. In a continuous-action setting we can add an **inter-rollout latent-diversity term** to the reward:

```
R_diversity(τ_i, group) = mean_{j ≠ i} ‖h_i − h_j‖² / (σ² · K · d)
```
weighted at γ_div = 0.05. This is cheap (matrix already computed for the importance ratios) and pushes the G rollouts to span more of the latent-space neighborhood, increasing the chance some of them are *not* the per-sample shortcut direction.

Skip in the first pilot (Option B + multi-reader is already a lot of moving parts); add if Option B alone fails to break the shortcut basin.

### 3.3 KL anchor

Two competing data points:
- **TRL default: β = 0.0** (KL term disabled). Justified by Open-Reasoner-Zero and DAPO showing KL penalty is not essential for GRPO stability.
- **VLM-R1: β = 0.04** (REC); **β = 0** (OVD).
- **LaCoT: β = 0** with a *reference-guided filter* replacing KL.
- **Monet's VLPO: β not extractable from the table; existence of a KL term implied by the standard GRPO objective.**

For us, the KL anchor's purpose is to keep π_θ's emitted `h` near the **frozen base VLM's emitted `h`** — i.e., away from adversarial-soft-prompt directions. This is the Variant-A-specific concern (`REPORT.md` §6, off-manifold drift) translated to RL.

**Recommendation: β = 0.04 against frozen base for the latent steps; β = 0 for the answer steps.** Asymmetric KL is unusual but matches the asymmetric concern: latents drift off-manifold (need KL), answer tokens don't (KL not essential).

Implementation: when computing the per-step KL approximator (Schulman 2020), apply β only on positions corresponding to latent emission. The reference policy is the frozen Qwen2.5-VL-7B.

β scheduling: hold β = 0.04 for first 50 % of training, then anneal linearly to 0 over the remainder. Rationale: early training needs the manifold anchor to prevent the off-manifold collapse the POC documented; late training, after the policy has converged to a useful region, the anchor becomes unnecessary and can constrain refinement.

### 3.4 Dataset

Verifiable rewards mean datasets with deterministic ground truth. Choices:

| Dataset | Ground-truth | Size | Pros | Cons |
|---|---|---|---|---|
| GQA (testdev_balanced) | exact-match short answer | 12 K | Already used in POC; known caveats | Narrow gold-token regime → many synonym mismatches (POC §3 footnote) |
| VQAv2-yesno | yes/no | filterable to ~100 K | Cleanest binary verifiable reward | Trivial in the limit; chance is 50 % |
| NLVR2 | true/false on image pair | 107 K | Compositional; 2-image structure | Image pair adds preprocessing complexity |
| A-OKVQA-MC | multiple choice (4-way) | 25 K (train) | Reasoning-heavy; non-binary | Smaller scale; commonsense answers |
| CLEVR-Hans / CLEVR | exact-match | 700 K | Programmatic Q-gen; clean compositional | Synthetic — generalization concern |

**Recommended training mix for the pilot (~100 K total):**
- 40 % GQA short-answer (40 K samples, single-token answer filter same as POC)
- 30 % A-OKVQA-MC (7.5 K samples, oversampled 4× for the reasoning value)
- 20 % NLVR2 (20 K)
- 10 % VQAv2-yesno (10 K, sanity baseline)

Skip CLEVR for the main run — its programmatic structure is too easy to shortcut on. Hold it as an *OOD eval* set instead: train without CLEVR, evaluate on CLEVR to test whether the latents generalize to a different distribution.

The 100 K target matches `LaCoT` (38 K-class scale on 7B) and `VL-Rethinker` (16 K for 7B). Variant B at our scale is on the small end, which lines up with the literature's working scale for RLVR-on-VLMs.

**Hold-out structure** (mandatory per `REPORT.md` §16's eval requirements):
- 5 % within-distribution held-out (test that training did something).
- POC 2-style held-out-question: 1 K samples where each image carries 2 questions; train on q1, evaluate on q2. **Direct test of whether Variant B's latents are q-portable** (the round-1 H2 hypothesis our POC found refuted).
- POC 3-style reader-transfer: same 1 K samples, evaluate the trained-policy's `h` under Monet-SFT-7B (φ_2). With the multi-reader reward (§3.1.2) on, this should be much better than the POC's 8.55 nat baseline; without it, expect a similar transfer drop.
- CLEVR OOD eval: 1 K CLEVR samples for distribution-shift sanity.

### 3.5 Compute estimate

Working from published anchors:
- LaCoT: 8×80GB GPU-node × ~120 h GRPO + 30 h SFT = **~960 GPU-hours for 7B + 38 K dataset + ~5 epochs**.
- VL-Rethinker: not disclosed but at ~16 K dataset for 7B and similar G=8, expect ~500-800 GPU-hours.
- VLM-R1 community reports: ~2-3 days on 8×A100 (~400-600 GPU-hours) for 7B REC at smaller dataset scale.

Variant B specifics that affect compute:
- **Multi-reader reward**: +1 forward pass per rollout. ~+30 % wall-time.
- **Random-control rollouts (shuffle-I, shuffle-q, permute)**: +3 forward passes per rollout (no generation, just scoring). ~+50 % wall-time.
- **VLPO Gaussian noise injection**: negligible cost; the noise is added during forward.
- Group size 8, dataset 100 K, 3 epochs.

**Estimate**: ~1600 GPU-hours on 4×H100 = **~17 wall-clock days for one full Variant B training run**. With 8×H100 cuts to ~8.5 days.

For comparison: Variant A SFT cost would be roughly G=1 rollout (deterministic gradient), no multi-reader at training time, no random controls — call it ~200-400 GPU-hours, **~4-8× cheaper than Variant B** at the same dataset scale.

**Pilot-scale (10 K samples × 1 epoch × 4×H100)**: ~30-50 GPU-hours, ~12 wall-clock hours. **This is the right size for the first Variant B run.** Decision-gate it before scaling to 100 K.

### 3.6 Failure modes to monitor

Each failure mode pairs with an active diagnostic the run must log.

1. **Reward hacking via shortcut basin** (the central concern). Diagnostic: track **% rollouts where shuffle-I reward is also 1**. If this stays high (>20 %), the policy is shortcut-encoding `(q, y*)` into `h`. **Decision rule**: if shuffle-I correctness > 20 % at step 5K, abort the run; the random-control reward is not punishing strongly enough → increase `γ_shuffle_I` from 0.5 to 1.0 or beyond.

2. **KL collapse / off-manifold drift**. Diagnostic: log `mean ||h||` per training step against `visual_baseline.json`'s 57.86 reference. **Decision rule**: if `mean ||h|| < 30` or `> 100` for 1K consecutive steps, the KL anchor is too weak; re-tune β.

3. **Monet-style mode collapse**: emitted latents converge to a narrow, sample-agnostic region (intra-block cos sim → 1, like Monet's 0.93). Diagnostic: log **inter-sample cos sim of `h_0` across the batch**. **Decision rule**: if > 0.9 sustained, the policy has entered a degenerate "constant latent" basin; this is a Monet-VLPO failure mode confirmed in their paper for vanilla GRPO. Apply LaCoT-style RGFN reference-filter to escape.

4. **Vanishing advantages** (the VL-Rethinker problem). Diagnostic: log `frac_reward_zero_std` (TRL native metric). **Decision rule**: if > 60 % of groups have zero std, enable SSR (replay buffer of non-zero-advantage samples).

5. **Distribution shift between train and held-out questions**. Diagnostic: POC 2-style held-out-q eval every 1K steps. **Decision rule**: if held-out NLL gap to within-distribution > 3 nat after 10 K steps, the policy is shortcut-encoding rather than learning visual content. Stronger random-control reward; or increase multi-question weight in the dataset.

6. **Reader-transfer collapse** (POC 3 failure mode on the trained policy). Diagnostic: φ_2 reward separately tracked, and computed even when not part of `R_acc` (i.e., always logged). **Decision rule**: if φ_2 / φ_1 reward ratio < 0.3 sustained, multi-reader is not propagating; consider adding φ_3 (a third sibling reader) or strengthening the `Σ_φ` weight.

7. **Format-reward gaming** (every RLVR system risks this). Diagnostic: log `R_acc` and `R_format` separately; if `R_format → 1` while `R_acc` stagnates, the policy has found a format shortcut. **Decision rule**: drop format reward weight β from 0.1 to 0.05; or replace with stricter format check.

8. **Thought collapse** (per arXiv:2503.08525 / GTR). Diagnostic: log entropy of intermediate `h` representations (per-position variance). **Decision rule**: if entropy collapses to near-zero for more than 1 latent position, the position is a placeholder; try GTR-style auxiliary thought correction.

### 3.7 Anti-shortcut tricks specific to RL

Consolidating the design choices that explicitly target Variant A's failure modes, with the trick - mechanism - cost - decision-status mapping:

| Trick | Mechanism | Cost | Status |
|---|---|---|---|
| **Random-control reward (shuffle-I, shuffle-q, permute)** | Negative reward on rollouts where shortcut-encoding would also succeed | +50 % wall-time | **Adopt; non-optional** |
| **Multi-reader reward** | Reward = mean correctness over Φ = {Qwen2.5-VL-7B, Monet-SFT-7B} | +30 % wall-time | **Adopt; non-optional** (only known fix for `REPORT.md` §16 reader-transfer failure) |
| **VLPO Gaussian on latents** | Continuous-action policy gradient on `h`; required for latent-side stochasticity | Negligible | **Adopt; structurally required** |
| **β = 0.04 KL anchor on latents only** | Manifold anchor against frozen base; off-manifold direction expensive | Standard | Adopt; sweep in pilot |
| **Diversity reward (LaCoT-style)** | Inter-rollout `‖Δh‖²` bonus | Negligible | Defer to round 2 if shortcut persists |
| **SSR (VL-Rethinker)** | Replay buffer of non-zero-advantage groups | Memory: ~5 GB | Adopt if vanishing advantages observed |
| **Forced Rethinking** | Append "Wait, does it seem right?" to first rollout | Negligible | **Adopt as ad-hoc diagnostic** — does the latent encode reflectable content? |
| **odLength-style length-mismatch penalty** | Penalize deviation between K_used and K_required | Negligible | Skip — our K is fixed |
| **Steering-probe certification** (arXiv:2512.21711) | Permute / zero / noise `h` at eval time; reward should drop | Eval-only | **Adopt at eval; decision gate** |

The first three are hard requirements. Without them, Variant B = Variant A + sampling noise = same shortcut basin (`REPORT.md` §16's prediction).

## 4. Pseudocode sketch — Variant B training step

The intent is to make the research-critical path read as a straight line. Hyperparameters live at the top of the file per `CLAUDE.md` conventions.

```python
# variant_b_train.py - core loop, simplified
# Constants (top of file)
G = 8                     # GRPO group size
K = 4                     # number of latent positions
SIGMA = 5.0               # Gaussian std on h emission (VLPO-style)
BETA_LATENT = 0.04        # KL coefficient on latent emission steps
BETA_TEXT = 0.0           # KL coefficient on answer tokens
LR = 1e-6
GAMMA_SHUFFLE_I = 0.5     # shuffle-image control weight
GAMMA_SHUFFLE_Q = 0.5     # shuffle-question control weight
GAMMA_PERMUTE = 0.5       # permute-latents control weight
ALPHA_ACC = 1.0           # accuracy reward weight
BETA_FORMAT = 0.1         # format reward weight
READERS = ["Qwen2.5-VL-7B", "Monet-SFT-7B"]

# One training step (pseudo-Python)
def train_step(batch):
    # batch = list of {"image": I, "question": q, "answer_gt": y_star}
    losses = []

    for sample in batch:
        # --- 1. ROLLOUT G trajectories ---
        rollouts = []
        for g in range(G):
            # Generator emits h_mu deterministically from (I, q)
            h_mu = policy.emit_latents(sample.image, sample.question, K=K)  # (K, d)
            eps = torch.randn_like(h_mu)
            h = h_mu + SIGMA * eps                                           # the action

            # Reader greedy-decodes the answer from h (no image given)
            y_hats = {}
            for phi_name in READERS:
                phi = frozen_readers[phi_name]
                y_hats[phi_name] = phi.greedy_decode(question=sample.question, latents=h)

            rollouts.append({"h_mu": h_mu, "h": h, "y_hats": y_hats, "logp_h": gaussian_logp(h, h_mu, SIGMA)})

        # --- 2. COMPUTE REWARDS for each rollout ---
        rewards = []
        for r in rollouts:
            # Multi-reader accuracy reward
            R_acc = sum(int(r["y_hats"][phi] == sample.answer_gt) for phi in READERS) / len(READERS)
            R_format = format_check(r["y_hats"])

            # Random controls (one extra forward each, no generation)
            shuf_I = random_image_from_pool()
            r_shuf_I = run_rollout(shuf_I, sample.question, h=r["h"])    # use same h
            R_shuffle_I = -int(r_shuf_I["y_hat"] == sample.answer_gt)

            shuf_q = random_question_from_pool(exclude_image=sample.image)
            r_shuf_q = run_rollout(sample.image, shuf_q, h=r["h"])
            R_shuffle_q = -int(r_shuf_q["y_hat"] == sample.answer_gt)

            h_perm = r["h"][torch.randperm(K)]
            r_perm = run_rollout(sample.image, sample.question, h=h_perm)
            R_permute = -int(r_perm["y_hat"] == sample.answer_gt)

            R = (ALPHA_ACC * R_acc + BETA_FORMAT * R_format
                 + GAMMA_SHUFFLE_I * R_shuffle_I
                 + GAMMA_SHUFFLE_Q * R_shuffle_q
                 + GAMMA_PERMUTE * R_permute)
            rewards.append(R)

        # --- 3. GROUP-RELATIVE ADVANTAGES ---
        rewards_t = torch.tensor(rewards)
        advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

        # --- 4. POLICY LOSS ---
        # Per-step importance ratio: latent steps use Gaussian density;
        # text-answer steps use categorical density (standard GRPO).
        for g, r in enumerate(rollouts):
            # Latent loss (VLPO Gaussian)
            ratio_h = torch.exp(-((r["h"] - r["h_mu"])**2).sum() / (2 * SIGMA**2)
                                 - ratio_h_old)        # vs. old policy h
            kl_h = approx_kl(policy_h_logp=r["logp_h"], ref_h_logp=ref_logp(r["h"]))

            loss_latent = -advantages[g] * ratio_h + BETA_LATENT * kl_h

            # Text loss (standard GRPO; answer-token-level)
            loss_text = -(advantages[g] * answer_token_logratio).mean() + BETA_TEXT * kl_text

            losses.append(loss_latent + loss_text)

    return torch.stack(losses).mean()
```

Notes on the sketch:
- The `ratio_h_old` term is the importance correction against the previous policy iteration's `h_mu` (standard GRPO bookkeeping; left abstract here for brevity).
- `frozen_readers` is a dict of frozen φ models loaded once at start. They share GPU with π_θ via FSDP-friendly partitioning.
- `format_check` is a simple regex on the answer string for `<think>...</think><answer>...</answer>`.
- Random-control sampling uses a pre-built pool of `(I, q, y*)` tuples to make the controls cheap; no extra disk hits per step.
- The loss is computed per-rollout-per-step then averaged; this is the standard GRPO factorization.

## 5. Pilot plan (decision-gated)

**Phase 0 — Trainer plumbing** (1 day, 1 GPU-day)
- Fork VLM-R1. Verify Qwen2.5-VL-7B GRPO runs end-to-end on a 1 K subset of GQA with vanilla `R_acc + R_format`. Pure baseline: no latents, no continuous policy, just text-output GRPO. Confirms the harness works.

**Phase 1 — Variant B: Option A (deterministic latents, sampled text)** (1 day, 5 GPU-days)
- 10 K samples × 1 epoch × G=8 × K=4 × no random controls × no multi-reader. Pure stock GRPO with latents as deterministic intermediate state.
- **Decision rule:** held-out POC 2 NLL must be < 4.0 nat (better than Variant A's 3.91 baseline) AND POC 3 transfer drop < 8.0 nat. **Otherwise abort and skip to Phase 2.**

**Phase 2 — Variant B: Option B (VLPO Gaussian)** (2 days, 10 GPU-days)
- Same data scale + VLPO σ=5.0 + KL anchor β=0.04 on latents.
- **Decision rule:** as Phase 1 + steering probe must show > 2 nat penalty for h-zeroing (per `REPORT.md` §14 protocol). **Otherwise the latents are placeholder-like; tighten σ or drop K.**

**Phase 3 — Variant B: full anti-shortcut suite** (4 days, 30 GPU-days)
- Add multi-reader reward + random-control rewards.
- **Decision rule:** POC 3 transfer drop ≤ 2 nat AND shuffle-I correctness ≤ 20 % AND held-out POC 2 NLL ≤ 1.5 nat above oracle. **All three must pass to scale.**

**Phase 4 — Scale to 100 K samples, 3 epochs** (8-17 days wall-clock, 1600 GPU-hours)
- Only if Phase 3 passes.

Expected outcome of Phases 1-3 based on the literature and our Variant A POC: **Phase 1 produces shortcut-dominant latents** (LaCoT shows GRPO=SFT on latent reasoning); **Phase 2 is the real Variant B**, expected to give a 1-2 nat improvement on POC 2 over Variant A but inherit POC 3 reader-transfer failure; **Phase 3 is the only phase with a plausible chance of fixing POC 3**, via the multi-reader reward.

## 6. What this design does not address

The honesty section. In order:

- **Generator-side Input-Latent Disconnect probe** (per Li et al. arXiv:2602.22766) is *not* in the design. It can only run on a trained generator and gates whether to publish the method. Defer to post-Phase-3.
- **CapImagine-style text alternative as the actual baseline.** Li et al.'s headline is "text imagination beats latent imagination." We should compare Variant B not just to SFT baselines but to a Qwen2.5-VL fine-tuned with explicit captions-as-thoughts under the same RLVR setup. Otherwise the contribution is fragile.
- **The Coconut latent-think critique** (arXiv:2512.21711) — our Variant A POC's steering result (`REPORT.md` §14) is the *opposite* of theirs (latents causally functional), but their OOD/biased-set evaluation is not yet ported to our pipeline. Mandatory before any publication claim.
- **σ choice for VLPO.** We picked 5.0 by analogy to Monet's 10.0 (their last-layer hidden states have ~5× larger norm than our post-merger embeddings). This is an educated guess; the right value should come from a sweep in Phase 2 (σ ∈ {1, 2, 5, 10, 20}). The Monet paper does not justify σ=10 either, so we are no worse off here.
- **No Coconut-VLM ablation.** Coconut is text-domain; the closest VLM analog is Mirage / SkiLa / LIVR, all covered in `LITERATURE_MITIGATIONS.md`. None has a publicly released RLVR variant; building one as a baseline costs another ~2 weeks. Defer.

## 7. Bottom line

Variant B as initially proposed (`a.md`) is structurally underspecified — vanilla GRPO does not update continuous latents (Monet's empirical finding, arXiv:2511.21395), so without VLPO-style Gaussian reparameterization "Variant B" reduces to "Variant A with stochastic answer-token sampling." The design above commits to:

1. **VLPO Gaussian on latent emission** with σ=5.0, asymmetric KL (β=0.04 latents / β=0 text), against frozen Qwen2.5-VL-7B as the reference.
2. **Multi-reader reward** (Qwen2.5-VL-7B + Monet-SFT-7B) — the structural fix for the POC 3 reader-transfer failure that round-2 mitigations could not solve.
3. **Random-control rewards** (shuffle-image, shuffle-question, permute-latents) — explicit negative reward on rollouts where shortcut-encoding would succeed. Directly priced-in version of the arXiv:2004.05704 caveat that random/insensible cues fake grounding gains.
4. **Group size G=8**, dataset ~100 K (GQA + A-OKVQA-MC + NLVR2 + VQAv2-yesno), 3 epochs, ~1 600 GPU-hours on 4×H100.
5. **Pilot at 10 K samples** with a hard decision gate before the 100 K scale-up. The pilot's most informative signal is whether VLPO + multi-reader + random-control together drop the POC 3 transfer gap below 2 nat — the criterion from `REPORT.md` §16.

The trainer to fork is **VLM-R1** (smallest patch surface for the VLPO change; `odLength` reward already implemented as a reference for our random-control reward shape). TRL's GRPOTrainer is good for the Phase-0 plumbing baseline but doesn't extend cleanly to continuous-latent policy gradient.

Expected dominant failure mode at Phase 2 (VLPO without multi-reader): same shortcut basin as Variant A, observable as `shuffle-I correctness > 20 %` and `POC 3 transfer drop > 5 nat`. Phase 3 (with multi-reader) is the one that has a real chance of being a publishable result. Variant B is, on this design, **only worth running if Phase 3 is the actual goal** — Phases 1-2 reproduce Variant A's failure mode at higher cost.

---

## Key references (with arXiv IDs)

- arXiv:2503.07536 — LMM-R1 (two-stage rule-based RL, Qwen2.5-VL-3B)
- arXiv:2503.18013 — Vision-R1 (vision-guided RL alignment)
- arXiv:2503.01785 — Visual-RFT (per-task verifiable rewards; ICCV 2025)
- arXiv:2503.08525 — GTR (thought-collapse mitigation, ICCV 2025)
- arXiv:2503.14476 — DAPO (clip-higher / dynamic sampling / token-level loss / overlong reward shaping)
- arXiv:2504.07615 — VLM-R1 (REC/OVD reward formulae; KL=0.04/0; G=8; `odLength` anti-hacking)
- arXiv:2504.08837 — VL-Rethinker (SSR replay buffer; Forced Rethinking; G=8)
- arXiv:2507.01006 — GLM-4.5V (RLCS curriculum sampling; multi-domain unified reward)
- arXiv:2508.19652 — Vision-SR1 (self-rewarding via reasoning decomposition)
- arXiv:2510.23925 — LaCoT (RGFN/GFlowNet; drop KL; **GRPO=SFT on latent reasoning**)
- arXiv:2511.21395 — Monet (VLPO Gaussian on latents; σ=10.0; rollout=8)
- arXiv:2602.22766 — Li et al. (Input-Latent / Latent-Answer Disconnect; CapImagine)
- arXiv:2512.21711 — "Do Latent Tokens Think?" (Coconut critique; steering-probe protocol)
- arXiv:2004.05704 — "Visual Grounding Methods for VQA are Working for the Wrong Reasons!" (random-cue control caveat)

Trainers:
- TRL v1.3.0 (HuggingFace) — VLM GRPO supported for Qwen2.5-VL; `examples/scripts/grpo_vlm.py`
- verl (volcengine, EuroSys 2025 HybridFlow) — VLM GRPO at scale; Qwen2.5-VL-7B / Qwen3-VL examples
- OpenRLHF v0.10 — VLM RLHF end-to-end; Ray-based async
- VLM-R1 (om-ai-lab) — DeepSpeed-based; smallest patch surface for VLPO fork
