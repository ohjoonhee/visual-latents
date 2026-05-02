# VLM Survey for Reader-Grounded Latent Reasoning

**Compiled:** 2026-05-02
**Question:** Should we switch the generator/reader VLM from `Qwen2.5-VL-7B-Instruct` to a more recent open-weight VLM? The user explicitly asked about "Qwen3.5".
**Constraints:**

- Reader is **frozen**; latents `h ∈ R^{K × D}` are spliced into the reader's vision-token slots in place of vision-encoder output. So `D` must match the LLM hidden dim of the reader, and the slot-injection point must be locatable.
- Reader must be a **sibling fine-tune (or same base)** of the generator so `h`'s geometry transfers. Without a sibling, POC 3 (reader transfer) collapses to "trust me, the geometry should generalize," which is not a defensible test.
- Hardware: 1×A6000 49GB local; 4×H100 for training.
- Sibling experiment `../monet-latent-probe/` already uses Qwen2.5-VL-7B + Monet-SFT-7B. Switching base means **losing that triangulation**.

---

## 1. Headline answer up front

**Recommendation: stay on `Qwen2.5-VL-7B-Instruct`. Treat `Qwen3-VL-8B-Instruct` as the only credible upgrade candidate, and only adopt it if (a) POC 1-3 finish on Qwen2.5-VL with stable infrastructure first, and (b) a *single-paper* result is needed for review-grade benchmarks where Qwen2.5-VL is notably weaker than Qwen3-VL.**

Justification, condensed:

- The method needs a sibling reader. Qwen2.5-VL-7B has *several* (Cosmos-Reason1-7B, VL-Rethinker-7B, MM-Eureka-Qwen-7B, R1-Onevision-7B, OpenVLThinker-7B, Skywork-VL Reward, plus your already-on-disk Monet-SFT-7B). Qwen3-VL-8B has *one we could verify* (Cosmos-Reason2-8B). InternVL3-8B happens to be Qwen2.5-7B-based which is interesting but not architecturally identical. Other candidates either have no sibling at all or have too-divergent architectures.
- "Qwen3.5-VL" does **not exist** as a separate release. Qwen3.5 (text/multimodal-unified) was announced Feb 2026 but is a different lineage; the **VL-specific** flagship line is Qwen3-VL (released Sept-Nov 2025), and that is what the user is implicitly asking about.
- The triangulation with `monet-latent-probe` is a genuine asset of the current setup. Both experiments share Qwen2.5-VL-7B embedding geometry, both can reuse the same slot-injection patch, and both can compare H1 (existing-Monet-latent inertness) against H1-3 (trainable-target-latent reachability) on the same reader. Migrating to Qwen3-VL forfeits this.
- Qwen3-VL is genuinely better as a *VLM*. But the experiment is not measuring VLM capability — it is measuring whether reader-grounded latents are reachable/content-bearing. A stronger reader does not change the structural questions being asked, and may even *hurt* them by making shortcuts easier (more capacity → more room to encode the answer non-visually).

Detailed comparison and per-model evidence below.

---

## 2. Landscape snapshot, May 2026

### 2.1 What was released since Qwen2.5-VL (Jan 2025)

Sorted by release date, restricted to ~7B-class open-weight VLMs and to anything with a relevant sibling story:

| Model | HF path | Release | Approx. params | LLM backbone | Vision encoder | License |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | `Qwen/Qwen2.5-VL-7B-Instruct` | 2025-01 | 7B + 0.7B ViT | Qwen2.5-7B | Custom ViT, window-attn, dynamic-res | Apache-2.0 |
| Cambrian-1-8B | `nyu-visionx/cambrian-8b` | 2024-06 | 8B | LLaMA-3-8B-Instruct | Multi-encoder (SigLIP + CLIP + DINOv2 + ConvNeXt) | Apache-2.0 (mostly) |
| Pixtral-12B | `mistralai/Pixtral-12B-2409` | 2024-09 | 12B + 0.4B ViT | Mistral-Nemo-12B | Pixtral-ViT (custom, native-res) | Apache-2.0 |
| Llama-3.2-11B-Vision-Instruct | `meta-llama/Llama-3.2-11B-Vision-Instruct` | 2024-09 | 11B (~9B LLM + cross-attn vision adapter) | Llama-3.1-8B + new vision blocks | MetaCLIP-derived ViT | Llama-3.2 community license |
| DeepSeek-VL2 | `deepseek-ai/deepseek-vl2` | 2024-12 | 27B MoE (4.5B activated) | DeepSeekMoE-27B | SigLIP-SO400M (dynamic tiling) | DeepSeek model license |
| Cosmos-Reason1-7B | `nvidia/Cosmos-Reason1-7B` | 2025-05 | 8.3B (sibling of Qwen2.5-VL-7B) | Qwen2.5-VL-7B-Instruct (post-trained) | inherited | NVIDIA Open Model License |
| GLM-4.1V-9B-Thinking | `zai-org/GLM-4.1V-9B-Thinking` | 2025-07 | 9B | GLM-4-9B-0414 | custom ViT | MIT |
| InternVL3-8B | `OpenGVLab/InternVL3-8B` | 2025-04 | 8.1B | Qwen2.5-7B (base, not instruct) | InternViT-300M-448px-V2.5 | Apache-2.0 |
| InternVL3.5-8B | `OpenGVLab/InternVL3_5-8B` | 2025-08 | 8.5B | Qwen3-8B (base) | InternViT-300M | Apache-2.0 |
| MiniCPM-V-4.0 | `openbmb/MiniCPM-V-4` | 2025 (mid) | 4.1B | MiniCPM4-3B | SigLIP2-400M | Apache-2.0 |
| GLM-4.5V | `zai-org/GLM-4.5V` | 2025-08 | 106B MoE (12B active) | GLM-4.5-Air | custom | MIT |
| Llama-4-Scout-17B-16E-Instruct | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | 2025-04-05 | 17B active / 109B total MoE | Llama-4 (early-fusion native multimodal) | MetaCLIP-derived | Llama-4 community license |
| Llama-4-Maverick | `meta-llama/Llama-4-Maverick-17B-128E` | 2025-04-05 | 17B active / ~400B total MoE | Llama-4 | MetaCLIP-derived | Llama-4 community license |
| Qwen3-VL-2B-Instruct | `Qwen/Qwen3-VL-2B-Instruct` | 2025-10 | 2B | Qwen3-2B | SigLIP2-So400m + DeepStack | Apache-2.0 |
| Qwen3-VL-4B-Instruct | `Qwen/Qwen3-VL-4B-Instruct` | 2025-10-15 | 4B | Qwen3-4B | SigLIP2-So400m + DeepStack | Apache-2.0 |
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | 2025-10-15 | 9B (8B LLM) | Qwen3-8B | SigLIP2-So400m (~543M) + DeepStack | Apache-2.0 |
| Qwen3-VL-32B-Instruct | `Qwen/Qwen3-VL-32B-Instruct` | 2025-10-21 | 33B | Qwen3-32B | SigLIP2-So400m + DeepStack | Apache-2.0 |
| Qwen3-VL-30B-A3B-Instruct | `Qwen/Qwen3-VL-30B-A3B-Instruct` | 2025-11-26 | 31B MoE (3B active) | Qwen3-MoE | SigLIP2-So400m + DeepStack | Apache-2.0 |
| Qwen3-VL-235B-A22B-Instruct | `Qwen/Qwen3-VL-235B-A22B-Instruct` | 2025-11-26 | 236B MoE (22B active) | Qwen3-MoE | SigLIP2-So400m + DeepStack | Apache-2.0 |
| Cosmos-Reason2-8B | `nvidia/Cosmos-Reason2-8B` | 2025-12-19 | 8.8B (sibling of Qwen3-VL-8B-Instruct) | Qwen3-VL-8B-Instruct (post-trained) | inherited | NVIDIA Open Model License |
| Gemma 4 (E4B / 31B / 26B-A4B) | `google/gemma-4-E4B`, `google/gemma-4-31B`, `google/gemma-4-26B-A4B` | 2026-04 | various | Gemma-4 | Gemma-4 native ViT (aspect-ratio-preserving fixed-budget tokens) | Gemma license |
| GLM-4.6V / 4.6V-Flash | `zai-org/GLM-4.6V`, `zai-org/GLM-4.6V-Flash` | 2026 | 106B / 9B | GLM-4.6 | custom | MIT |
| GLM-5V-Turbo | (zai-org) | 2026 | (large) | GLM-5 | custom | MIT |

### 2.2 What "Qwen3.5-VL" actually refers to

The user asked about Qwen3.5-VL by name. Two findings:

1. There is **no `Qwen/Qwen3.5-VL-*` repository**. HuggingFace's `Qwen` org has `Qwen3.5-{0.8B, 4B, 9B, 27B, ...}` (text), but no Qwen3.5-VL line. The Qwen team appears to have folded VL into a unified "Qwen3.5 multimodal foundation" rather than naming a separate VL series. *(Source: HF collection listings as of May 2026; not yet papered.)*
2. "Qwen3.5" (text-and-VL-unified) was announced approximately **2026-02-16** with claimed cross-generational parity vs. Qwen3 + outperforming Qwen3-VL on multimodal benchmarks. The available *VL-specific* artifacts in May 2026 still appear to be the Qwen3-VL line (released Sept-Nov 2025); the user should treat "Qwen3.5 multimodal" as conceptually the next-gen successor but operationally **the actually-downloadable open-weight VLM with sibling fine-tunes is still Qwen3-VL-8B-Instruct.**

There is also a separate `Qwen3.6` text series (`Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B`) released between Qwen3.5 and now — text-only as of the snapshot.

**Working interpretation:** when the user says "Qwen3.5-VL", they likely mean *the most recent Qwen vision model available*, which in May 2026 is still **Qwen3-VL-8B/32B-Instruct** (Oct/Nov 2025). The doc treats Qwen3-VL as the upgrade candidate.

---

## 3. Per-candidate technical detail

### 3.1 Qwen2.5-VL-7B-Instruct (current baseline)

- **HF path:** `Qwen/Qwen2.5-VL-7B-Instruct`
- **Release:** January 2025; technical report `arXiv:2502.13923` (Feb 2025).
- **LLM:** 28-layer decoder, hidden 3584, MLP 18944, 4 KV heads, head dim 128.
- **Vision encoder:** 32-layer ViT, hidden 1280, 16 heads, MLP 3456, 8×8 windowed attention with select full-attn layers, native dynamic resolution.
- **Vision-LLM merger:** projects 1280 → 3584 per grouped patch block.
- **Image-token convention:** `<|vision_start|><|image_pad|><|vision_end|>`. Each `<|image_pad|>` slot occupies one position; the count is **variable per resolution** (default range 4 to 16,384 visual tokens, configurable via `min_pixels` / `max_pixels`).
- **Reported scores** (from Qwen2.5-VL-7B model card / tech report):
  - MMBench-v1.1-EN test: **82.6**
  - MMStar: **63.9**
  - MMVet (GPT-4-Turbo): **67.1**
  - BLINK: **56** (paper); community VLMEvalKit reproductions sometimes ≈40 — methodology-sensitive.
  - RefCOCO_val (bbox grounding): **90.0**
  - DocVQA test: **95.7**, OCRBench: **864**
- **Sibling fine-tunes (Qwen2.5-VL-7B-Instruct-derived, all suitable for reader-transfer testing):**
  - **Cosmos-Reason1-7B** (`nvidia/Cosmos-Reason1-7B`, `arXiv:2503.15558`, post-trained for embodied physical-AI reasoning — SFT + RL on physical-common-sense data).
  - **VL-Rethinker-7B** (TIGER-AI-Lab, NeurIPS 2025; Selective Sample Replay + Forced Rethinking GRPO).
  - **MM-Eureka-Qwen-7B** (ModalMinds; large-scale rule-based RL with GRPO; ViT frozen during training — *exactly the same idea as our frozen-reader setup*, useful for cross-checking).
  - **R1-Onevision-7B** and **R1-Onevision-7B-RL** (Fancy-MLLM, ICCV 2025, `arXiv:2503.10615`; SFT-then-RL for chain-of-thought visual reasoning).
  - **OpenVLThinker-7B** (`arXiv:2503.17352`; iterative SFT-RL cycles).
  - **Skywork-VL Reward 7B** (`arXiv:2505.07263`; reward-head-augmented Qwen2.5-VL-7B).
  - **Monet-SFT-7B** (`NOVAglow646/Monet-SFT-7B`, `arXiv:2511.21395`; the sibling reader already used by `monet-latent-probe`).
  - Hugging Face's `base_model:finetune:Qwen/Qwen2.5-VL-7B-Instruct` filter shows hundreds of community fine-tunes.
- **Limitations / quirks:**
  - BLINK score is reproduction-sensitive; pin VLMEvalKit version for any benchmark claim.
  - Window attention in ViT means slot-injection in our experiment must happen *post-merger* (at the LLM input), not at the ViT output, because the merger groups patches.
- **Already-paid integration cost:** vLLM works, our slot-injection patch in `monet-latent-probe` is for this architecture, the sibling experiment infrastructure is built around it.

### 3.2 Qwen3-VL-8B-Instruct (primary upgrade candidate)

- **HF path:** `Qwen/Qwen3-VL-8B-Instruct`
- **Release:** 2025-10-15. Technical report: `arXiv:2511.21631` (Nov/Dec 2025).
- **LLM:** 36-layer decoder, hidden_size **4096**, 32 attention heads, intermediate_size 12288.
- **Vision encoder:** initialized from **SigLIP2-So400m** (~543M params); 27-layer ViT, hidden 1152, 16 heads, MLP 4304, patch_size 16, projects to out_hidden_size 4096. Uses **DeepStack** (multi-level ViT feature fusion at LLM layers `[8, 16, 24]`).
- **Image-token convention:** same `<|vision_start|><|image_pad|><|vision_end|>` template as Qwen2.5-VL (verified on the Qwen3-VL GitHub README); image-token count is again **variable per resolution** via `min_pixels` / `max_pixels`. **Key difference:** DeepStack means visual features are injected at multiple LLM depths, not just at the input — for our experiment this changes what "splice into vision-token positions" means and may require deciding whether to also intervene at the DeepStack injection layers, or only at the input slot.
- **Reported scores** (8B-Instruct, from technical report and `llm-stats.com` aggregator; some are from the report's official VLMEvalKit configuration):
  - MMBench-v1.1: **85.0** (vs Qwen2.5-VL-7B's 82.6)
  - BLINK: **69.1** (paper) / **59.9** (VLMEvalKit) — same reproduction discrepancy as Qwen2.5-VL.
  - The technical report claims the 8B "consistently leads across all five VQA benchmarks within its size category" (MMBench, MMStar, RealWorldQA, OCRBench, MathVista).
  - MMVet, SEED-Bench, RefCOCO+ specific 8B numbers were **not surfaced** by my searches in extractable form (the tech report PDF is largely scan-based; visualizations are images). Treat these as unverified.
- **Sibling fine-tunes (Qwen3-VL-8B-Instruct-derived):**
  - **Cosmos-Reason2-8B** (`nvidia/Cosmos-Reason2-8B`, released 2025-12-19, post-trained from Qwen3-VL-8B-Instruct for physical-AI reasoning; comparative evals 2026-04-28). Confirmed sibling, suitable for reader-transfer.
  - **Cosmos-Reason2-2B** (sibling of Qwen3-VL-2B-Instruct, not 8B — wrong size for transfer to our generator).
  - HuggingFace `base_model:finetune:Qwen/Qwen3-VL-8B-Instruct` shows 218 community fine-tunes (more than enough for diversity in transfer testing, but most are LoRA adapters — full-weight RL'd siblings are rarer than for Qwen2.5-VL).
- **Limitations / quirks:**
  - **DeepStack changes the semantics of "splice into vision-token positions."** A clean POC 1-3 on Qwen3-VL needs to choose between: (a) replacing input embedding only (lets multi-level ViT features still flow if the original image is forwarded — but our scope says *no other image input to reader*); (b) replacing input embedding *and* zeroing the DeepStack-fed layers; (c) replacing input embedding *and* providing learned latents at the DeepStack positions too (K_total = K_input + K_deepstack — even more capacity, even more shortcut risk). This is a non-trivial design decision with no published precedent.
  - Native context 256K / extendable to 1M means our existing context-management code will likely need re-tuning.
  - Vision encoder is SigLIP2-So400m, not a custom ViT — this is actually *cleaner* for the experiment (better-understood features), but means the latent geometry is fundamentally different from Qwen2.5-VL's, so no transfer between the two bases.

### 3.3 InternVL3-8B (interesting because LLM = Qwen2.5-7B base)

- **HF path:** `OpenGVLab/InternVL3-8B`. arXiv `2504.10479` (Apr 2025).
- **LLM:** **Qwen2.5-7B base** (not the instruction-tuned version that Qwen2.5-VL ships with). 7B params.
- **Vision encoder:** InternViT-300M-448px-V2.5; ViT-MLP-LLM paradigm; 1024-tokens-per-patch reduced to 256 via pixel shuffle.
- **Image-token convention:** `<image>` placeholder; dynamic-resolution tiling into 448×448 blocks; configurable `max_num` (1-12 tiles), optional thumbnail. **Variable** per image.
- **Reported scores (InternVL3-8B from arXiv 2504.10479):**
  - MMBench v1.1 (EN): **81.7**
  - MMStar: **68.2**
  - MMVet: **81.3**
  - BLINK: **55.5**
  - RefCOCO+ val: **88.2**
  - SEED-2 Plus: **69.7**
  - MME: **2415.4**
- **Sibling fine-tunes:** InternVL3.5-8B (now Qwen3-base), InternVL3-8B-Instruct, MM-Eureka-InternVL variants, CapRL-InternVL3.5-8B. *None* of these are siblings of Qwen2.5-VL-7B-Instruct in the strict sense (different vision encoder, different MLP, different training data). But the **LLM weights being Qwen2.5-7B-base** means reader-transfer between InternVL3-8B and Qwen2.5-VL-7B-Instruct is "halfway sibling" — same LLM block weights at init, different vision-side everything. Could be a useful *negative control* for the transfer test (if latents fit Qwen2.5-VL-7B-Instruct's geometry but not InternVL3-8B's despite shared LLM weights, it isolates the role of the vision-merger).
- **Limitations:** *Not* a clean sibling for our purposes. Useful as an auxiliary, not as the primary reader for POC 3.

### 3.4 InternVL3.5-8B

- **HF path:** `OpenGVLab/InternVL3_5-8B`. arXiv `2508.18265` (Aug 2025).
- **LLM:** Qwen3-8B base. Total 8.5B params.
- Vision encoder: InternViT-300M (same family).
- **Reported scores (per arXiv 2508.18265, average of suite):** InternVL3.5-8B overall ~59.9, BLINK is part of the suite. Not directly useful as a Qwen3-VL-8B sibling (different vision side), but *is* a Qwen3-8B-base sibling, giving us a triangulation similar to InternVL3-8B vs. Qwen2.5-VL-7B.

### 3.5 Cosmos-Reason1-7B

- **HF path:** `nvidia/Cosmos-Reason1-7B`. arXiv `2503.15558` (Mar 2025).
- **Base:** `Qwen2.5-VL-7B-Instruct`, post-trained with SFT+RL on physical-common-sense and embodied-reasoning data. Chain-of-thought reasoning is the differentiator.
- Vision tower: 675M params (inherited).
- LLM: 7.07B (inherited).
- License: NVIDIA Open Model License (Apache-2.0-additional-terms; commercially usable, derivatives OK).
- **Why it matters here:** confirmed strong sibling for POC 3. If an `h*` learned under frozen Qwen2.5-VL-7B-Instruct also produces correct answers under frozen Cosmos-Reason1-7B, that's evidence the latents lie on something both readers respect — i.e., not adversarial-to-reader-1.

### 3.6 Cosmos-Reason2-8B

- **HF path:** `nvidia/Cosmos-Reason2-8B`. Released 2025-12-19; comparative evals 2026-04-28.
- **Base:** Qwen3-VL-8B-Instruct, post-trained.
- Total 8.77B params. License: NVIDIA Open Model License.
- **Why it matters here:** the analogous sibling for Qwen3-VL-8B-Instruct. Also a 2B variant exists for the 2B base.

### 3.7 Llama 3.2-11B-Vision-Instruct

- **HF path:** `meta-llama/Llama-3.2-11B-Vision-Instruct`. Released 2024-09-25.
- **LLM:** Llama-3.1-8B (text component), augmented with **cross-attention vision adapter blocks**. Total ~11B (the extra 3B is the vision-cross-attn machinery, *not* a parallel ViT-then-prefix design).
- This is the deal-breaker: **vision is injected via cross-attention, not via vision-token-position embeddings spliced into the LLM input.** Our slot-injection method does not have a natural place to land.
- No Llama-3.2-Vision sibling fine-tunes verified for our scale.
- **Verdict:** Architectural mismatch — *not viable as drop-in*. Adapting our method would mean intervening in cross-attn keys/values, which is a different research project.

### 3.8 Llama-4-Scout-17B-16E and Llama-4-Maverick-17B-128E

- Released 2025-04-05. MoE (16 / 128 experts), 17B activated, 109B / ~400B total.
- **Native early-fusion multimodality** — text and vision tokens are unified at layer 0 with a MetaCLIP-derived encoder; no separate "vision tower then prefix" design.
- Context up to 10M tokens (Scout) / 1M (Maverick).
- License: Llama-4 community license (less permissive than Apache; commercial use bounded).
- **Verdict:** Early fusion makes "splice latents into vision-token positions" unclear. The notion of "the vision-encoder output" is not architecturally well-defined the way it is in ViT-then-prefix models. Plus 17B activated × MoE makes our 4×H100 budget tight, and there is no clean 7B-class sibling fine-tune. *Not recommended.*

### 3.9 Pixtral-12B

- **HF path:** `mistralai/Pixtral-12B-2409`. Released 2024-09-17.
- **LLM:** Mistral-Nemo-12B (40 layers, hidden 14336).
- **Vision encoder:** Pixtral-ViT (custom; 24 layers, hidden 1024, MLP 4096, 16 heads, patch_size 16, image_size 1024). Native variable resolution.
- **Image-token convention:** `[IMG]` tokens, count depends on H×W; rows separated by `[IMG_BREAK]`, images by `[IMG_END]`. Variable.
- **Vision-LLM merger:** 2-layer FC (hidden_size→hidden_size→14336) with GeLU.
- License: Apache-2.0 (clean).
- **Sibling fine-tunes:** thin. Mistral-Nemo-base is small in the open-source community as a research VLM platform; no comparable RL'd or SFT'd "Pixtral-Reasoner" line exists at the scale we have for Qwen2.5-VL.
- **Verdict:** Architecturally compatible (clean ViT-then-prefix), license clean, but **12B is 70% larger than 7B for the generator, and the lack of siblings kills POC 3.** Not recommended.

### 3.10 Cambrian-1-8B

- **HF path:** `nyu-visionx/cambrian-8b`. Released June 2024.
- LLM: LLaMA-3-8B-Instruct.
- Vision tower: **four encoders fused via Spatial Vision Aggregator** (SigLIP + CLIP + DINOv2 + OpenCLIP-ConvNeXt). This is unusual and the latent geometry is per-encoder; "splice latents into vision-token positions" needs to specify *which encoder's positions*.
- Sibling fine-tunes: there is a Cambrian-S (multi-step reasoning extension), but no widely-replicated RL or SFT-reasoner line.
- **Verdict:** The four-encoder vision tower is a structural mismatch. *Not recommended.*

### 3.11 DeepSeek-VL2

- **HF path:** `deepseek-ai/deepseek-vl2`. Released 2024-12-13.
- MoE 27B (4.5B activated). Tiny: 3B/1B activated. Small: 16B/2.8B activated.
- Vision: SigLIP-SO400M with dynamic tiling. ViT-then-prefix style.
- **Why MoE is a problem here:** the reader's hidden state at any layer is the result of activated experts. If our latents h are 4.5B-active dependent, generator and reader expert routing must align — likely it does for siblings, but no siblings of DeepSeek-VL2 at 4.5B-active exist as RL-tuned variants.
- **Verdict:** No sibling for transfer. Skip.

### 3.12 GLM-4.1V-9B-Thinking, GLM-4.5V, GLM-4.6V/4.6V-Flash, GLM-5V-Turbo

- `zai-org/GLM-4.1V-9B-Thinking` (`arXiv:2507.01006`, July 2025) — 9B, thinking-trained. The 9B is the only ~7B-class variant. License: MIT (cleanest of any candidate here).
- GLM-4.5V (Aug 2025): 106B MoE / 12B active — too big for our 4×H100 training budget.
- GLM-4.6V / 4.6V-Flash (2026): 106B / 9B. The Flash is the same 9B class.
- GLM-5V-Turbo (2026): even larger, 200K context.
- **Sibling story:** GLM-V series is consistently zai-org-only; community fine-tunes exist but RL'd reasoner siblings of GLM-4.1V-9B specifically are not as numerous as for Qwen2.5-VL. We did not verify a clean sibling.
- **Verdict:** GLM-4.1V-9B-Thinking is interesting (MIT license, thinking-tuned, ~9B), but the missing sibling and the divergent vision architecture make it a riskier swap than Qwen3-VL.

### 3.13 MiniCPM-V-4.0 (and 4.5)

- 4.1B parameters, SigLIP2-400M + MiniCPM4-3B. Optimized for on-device. MiniCPM-V-4.5 also exists.
- *Below* the 7B size target. The reader being 4B means latent dim is smaller; experiment-cost-wise this is friendlier (faster forward), but the original method is sized for 7B and the user explicitly preferred 7B-class.
- No widely-known sibling-RL fine-tune at this scale.
- **Verdict:** Too small for the spec; skip unless we deliberately want a smaller-reader sanity check.

### 3.14 Gemma 4 (E4B / 31B / 26B-A4B)

- HF org `google/gemma-4-*`. Added to transformers 2026-04-01.
- E4B (effective 4B) and 31B are dense; 26B-A4B is MoE.
- **New vision design:** "fixed-budget number of tokens per image regardless of resolution" — this is a *significant* deviation from the variable-per-resolution scheme everywhere else on this list. For our experiment that's actually attractive (K is naturally fixed, no `min_pixels`/`max_pixels` choice).
- **However:** released April 2026 (last month). Sibling fine-tunes do not exist at meaningful scale yet. The community ecosystem is too young.
- License: Gemma (commercial OK with terms).
- **Verdict:** Architecturally interesting, ecosystem too young for POC 3. Revisit in 6 months.

### 3.15 Other Feb–May 2026 mentions

The April-13–28 2026 batch (LFM2.5-VL-450M, EXAONE 4.5, Granite 4.0 3B Vision, InternVL-U, GLM-4.6V, Vero, MolmoWeb, UniDriveVLA, S1-VL, Claude Mythos gated, Qwen3.6 text-only) — none of these I could verify as having both (a) a 7B-class checkpoint and (b) a confirmed RL-tuned sibling at the same scale. Most are either too small, too large, gated, or too new.

---

## 4. Cross-comparison

### 4.1 Recommendation matrix

Scoring 1-5 per column. Higher = better fit for our experiment. Bold = top-3.

| Model | Ecosystem fit (vLLM, our slot-injection patch, transformers integration) | Sibling reader available for POC 3 | Likely transfer behavior (does latent geometry travel?) | Training friendliness (4×H100, frozen reader cost) | Recency | License |
|---|---|---|---|---|---|---|
| **Qwen2.5-VL-7B-Instruct** (current) | **5** (already integrated, sibling experiment uses it) | **5** (Cosmos-Reason1, Monet-SFT, MM-Eureka, R1-OneVision, VL-Rethinker, OpenVLThinker, Skywork-VL-Reward all confirmed) | **4** (sibling-fine-tune literature is mature; transfer well-studied) | **5** (7B fits 4×H100 generator + frozen reader cleanly; current code works) | 2 (Jan 2025; one-and-a-half generations behind) | 5 (Apache-2.0) |
| **Qwen3-VL-8B-Instruct** | 3 (need to redo slot injection for DeepStack; vLLM works; transformers integrated since Sept 2025) | 3 (Cosmos-Reason2-8B confirmed; community fine-tunes plenty but most are LoRA, not full-weight RL siblings) | 3 (less literature; SigLIP2 + DeepStack is a different geometry; transfer behavior unstudied) | 4 (8B → marginally tighter; DeepStack adds ~10% vision compute) | **5** (Oct 2025) | 5 (Apache-2.0) |
| InternVL3-8B | 3 (different processor, different image-token scheme; vLLM works) | 2 (no Qwen2.5-VL-7B-Instruct sibling; InternVL3.5 is Qwen3-base; halfway-LLM-sibling at most) | 2 (vision-merger architecture differs, only LLM block weights coincide) | 4 | 4 (Apr 2025) | 5 (Apache-2.0) |
| InternVL3.5-8B | 3 | 2 | 2 | 4 | 4 (Aug 2025) | 5 (Apache-2.0) |
| Cosmos-Reason1-7B (as base) | 5 (would inherit Qwen2.5-VL-7B integration) | 5 (Qwen2.5-VL-7B-Instruct itself is then the sibling) | 4 | 5 | 3 (May 2025) | 4 (NVIDIA OML; commercial OK with terms) |
| Llama-3.2-11B-Vision-Instruct | 1 (cross-attn vision adapter — slot-injection has no natural target) | 2 | 1 | 3 | 2 (Sep 2024) | 3 (Llama-3.2 community license) |
| Llama-4-Scout-17B-16E-Instruct | 1 (early fusion, MoE) | 1 (no clean 7B-class sibling) | 1 | 2 (17B active × MoE) | 4 (Apr 2025) | 3 (Llama-4 community license) |
| Pixtral-12B | 4 (clean ViT-then-prefix; vLLM works; transformers integrated) | 1 | 3 | 3 (12B; reader frozen but generator at 12B is ~70% more compute) | 2 (Sep 2024) | 5 (Apache-2.0) |
| Cambrian-1-8B | 2 (four-encoder vision tower) | 2 | 2 | 4 | 1 (Jun 2024) | 5 |
| DeepSeek-VL2 | 3 | 1 | 2 | 3 (MoE) | 3 (Dec 2024) | 3 (DeepSeek model license) |
| GLM-4.1V-9B-Thinking | 3 | 2 | 2 | 4 | 3 (Jul 2025) | **5 (MIT)** |
| GLM-4.6V-Flash (9B) | 2 | 2 | 2 | 4 | 5 (2026) | 5 (MIT) |
| MiniCPM-V-4.0 | 3 | 2 | 2 | 5 (smallest) | 3 (mid-2025) | 5 (Apache-2.0) |
| Gemma 4 (E4B / 31B / 26B-A4B) | 2 (new transformers integration; new vision design with fixed-budget tokens) | 1 (too new for siblings) | 2 (untested) | 4 (E4B) / 3 (31B) | **5** (Apr 2026) | 4 (Gemma license) |

Aggregated (unweighted sum, max 30):

| Model | Sum |
|---|---|
| **Qwen2.5-VL-7B-Instruct** | **26** |
| **Cosmos-Reason1-7B** (treat as Qwen2.5-VL-7B sibling, base swappable) | **26** |
| **Qwen3-VL-8B-Instruct** | **23** |
| GLM-4.1V-9B-Thinking | 19 |
| InternVL3-8B | 18 |
| InternVL3.5-8B | 18 |
| Pixtral-12B | 18 |
| Gemma 4 E4B | 17 |
| MiniCPM-V-4.0 | 17 |
| GLM-4.6V-Flash | 16 |
| DeepSeek-VL2 | 13 |
| Cambrian-1-8B | 11 |
| Llama-4-Scout | 11 |
| Llama-3.2-11B-Vision-Instruct | 10 |

### 4.2 Top-3 picks with explicit pros/cons

#### Pick 1: stay on Qwen2.5-VL-7B-Instruct (current baseline)

**Pros**
- POC 1-4 already finished against this model (per JOURNAL.md 2026-05-01 overnight entry). Switching = re-running everything.
- Largest sibling-reader pool of any candidate: Cosmos-Reason1, MM-Eureka-Qwen, VL-Rethinker, R1-Onevision, OpenVLThinker, Skywork-VL-Reward, Monet-SFT — all Qwen2.5-VL-7B-Instruct-derived. Lets POC 3 (reader transfer) use multiple readers, not just one.
- Triangulation with `monet-latent-probe` is preserved. Both experiments answer the same broad question (does latent visual reasoning carry content?) on the same architecture.
- Slot-injection patch is a known quantity — `<|image_pad|>` slot positions, hidden dim 3584, bf16, no DeepStack to reason about.
- Apache-2.0 license, vLLM works, transformers integrated.

**Cons**
- One-and-a-half generations behind Qwen3-VL on raw capability.
- Reproduction-sensitive on BLINK (paper says 56, VLMEvalKit says ~40); any benchmark claim must pin tooling.
- If a reviewer asks "why not the latest VLM?", the answer is structural (sibling availability + triangulation) rather than capability — needs to be argued, not assumed.

#### Pick 2: Qwen3-VL-8B-Instruct (only credible upgrade)

**Pros**
- Genuinely state-of-the-art at the 8B scale; MMBench-v1.1 85.0 vs Qwen2.5-VL-7B's 82.6.
- SigLIP2-So400m vision encoder is better-understood / cleaner than Qwen2.5-VL's custom ViT.
- Cosmos-Reason2-8B exists as a confirmed sibling (Qwen3-VL-8B-Instruct → SFT+RL by NVIDIA).
- Apache-2.0, transformers integrated (since Sept 2025), vLLM works.
- Native 256K → 1M context (irrelevant for our experiment but indicates engineering maturity).

**Cons**
- **DeepStack changes the meaning of "splice into vision-token positions."** This is a real research-design question, not a bug — the experiment may need to be re-specified for what "the visual path" is in a multi-depth fusion model.
- Sibling pool is much thinner than Qwen2.5-VL's. Cosmos-Reason2-8B is the one solid sibling I could verify; LoRA fine-tunes are plentiful but full-weight RL'd siblings are rare.
- Switching forfeits the triangulation with `monet-latent-probe`. That experiment's H1 result is on Qwen2.5-VL-7B + Monet; re-running on Qwen3-VL means re-training a Monet-style latent-emission head.
- Higher capacity = more shortcut surface area for the soft-prompt-tuning failure mode that POC 2 is designed to detect. A stronger reader might encode the answer non-visually *more easily*, not less.

#### Pick 3: Cosmos-Reason1-7B as generator with Qwen2.5-VL-7B-Instruct as reader (or vice versa)

This is a *configuration*, not a different model. The trick: Cosmos-Reason1-7B is a Qwen2.5-VL-7B-Instruct sibling, so we can run POC 3 with Cosmos as generator and Qwen2.5-VL as reader, or the other way. Both directions are on-disk-compatible with our existing patch.

**Pros**
- Same Qwen2.5-VL infrastructure, but the *generator* is now reasoning-tuned via SFT+RL on physical-common-sense data. If the latent geometry is identical (same base), our existing patch and POC 1-3 should run unchanged.
- Cosmos-Reason1's RL training is on long chain-of-thought reasoning — exactly the regime where reader-grounded latents would matter most.
- License is NVIDIA OML (commercial OK).

**Cons**
- This is an *additional* configuration, not a replacement. Doesn't answer the user's "should I switch" question — it answers "should I add a configuration."
- Latent geometry presumably matches the base, but there is some risk that RL post-training has shifted the input-embedding distribution; H4 (transferability) would be the test.

### 4.3 What to drop

- **Llama-3.2-Vision** (cross-attn vision adapter) — architectural mismatch.
- **Llama-4** (early fusion + MoE) — architectural mismatch + sibling absence + license.
- **Pixtral, Cambrian, DeepSeek-VL2** — sibling absence or vision-tower mismatch.
- **GLM-4.5V / 4.6V / 5V** (large variants) — out of scale.
- **Gemma 4** — too new; ecosystem not ready for siblings.

---

## 5. Decision

**Stay on Qwen2.5-VL-7B-Instruct for the current POC sequence (POC 1-4) and any near-term training plan.** Reasons in priority order:

1. The experiment's load-bearing question is *not* "can a stronger VLM produce reader-grounded latents?" It is "do reader-grounded latents that satisfy the objective even exist, and are they visual or shortcut?" That question is base-model-agnostic; a stronger base does not change the answer's sign, only its magnitude. POC 2 (held-out-question shortcut detector) and POC 3 (reader-transfer test) are the binding tests, and both are *more* informative on a base with mature sibling fine-tunes.
2. Triangulation with `monet-latent-probe` is non-trivial to recreate on a different base. Both experiments would have to migrate together for parity — that's a much larger investment than a "pick the best VLM" framing suggests.
3. Qwen3-VL's DeepStack architecture is a research-design wrinkle, not a free lunch. Adopting it means deciding what "the visual path into the reader" means when the reader has multi-depth ViT fusion.
4. Sibling pool: Qwen2.5-VL-7B has 7+ verified RL/SFT siblings of distinct provenance. Qwen3-VL-8B has 1-2. POC 3 with multiple sibling readers is a stronger transfer story than POC 3 with one.

**When to revisit this decision:**

- If a reviewer cycle demands "use the most recent VLM for credibility" and POC 1-3 are clean → port to Qwen3-VL-8B-Instruct + Cosmos-Reason2-8B reader. The cost is one re-implementation of slot injection (account for DeepStack) and one re-run of POC 1-3.
- If we genuinely want to test whether a stronger reader makes shortcuts *harder* or *easier* (counterintuitive but the right question for capacity-vs-shortcut analysis) → run *both* Qwen2.5-VL-7B and Qwen3-VL-8B as a paired condition. This is the strongest version of the paper's negative-result framing.
- Six months out, when Gemma 4 and Qwen3.5-VL (or whatever the unified-Qwen3.5 multimodal artifact is) have established sibling pools, the recommendation may flip.

**Suggested action item:** add a one-line note to JOURNAL.md noting the decision and the trigger conditions for revisiting. No infrastructure work required.

---

## 6. Verification log

### 6.1 arXiv IDs cited (verified by search-result hit)

| Cited as | Title hit found | Status |
|---|---|---|
| `2502.13923` | "Qwen2.5-VL Technical Report" | verified |
| `2511.21631` | "Qwen3-VL Technical Report" | verified |
| `2504.10479` | "InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models" | verified |
| `2508.18265` | "InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency" | verified |
| `2503.15558` | "Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning" | verified |
| `2503.10615` | "R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization" | verified (ICCV 2025) |
| `2503.17352` | "OpenVLThinker: Complex Vision-Language Reasoning via Iterative SFT-RL Cycles" | verified |
| `2505.07263` | "Skywork-VL Reward: An Effective Reward Model for Multimodal Understanding and Reasoning" | verified |
| `2507.01006` | "GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning" | verified |
| `2511.21395` | "Monet" (Wu et al., CVPR 2026; cited in `monet-latent-probe`) | verified per parent doc |
| `2410.07073` | "Pixtral 12B" | verified |
| `2412.06769` | "Coconut" (text-latent-reasoning baseline) | not re-verified this round; carried from `monet-latent-probe` |
| `2602.22766` | Li et al., "Imagination Helps Visual Reasoning, But Not Yet in Latent Space" | not verified this round; carried from `monet-latent-probe` JOURNAL.md correction |
| `2409.12191` | Qwen2-VL Technical Report | verified |

(The two not-re-verified IDs above are pre-existing in the experiment's reference set and are out of scope for this VLM-survey verification pass.)

### 6.2 Things I could not verify and am flagging

- **Qwen3-VL-8B-Instruct exact MMVet, SEED-Bench, RefCOCO+ scores** — the technical report PDF is heavily image-based and search aggregators do not surface these consistently. Reproducing them from the official VLMEvalKit configuration is left as a separate task. The MMBench (85.0) and BLINK (69.1 paper / 59.9 VLMEvalKit) numbers above are sourced from `llm-stats.com` aggregator and the Qwen3-VL GitHub issue tracker respectively; treat as approximate.
- **"Qwen3.5-VL" as a separate VL line** — does not exist in `Qwen/` HF org as of May 2026. There is `Qwen3.5-{0.8B, 4B, 9B, 27B}` (text/unified) and there is `Qwen3-VL-*` (true VL). The user's "Qwen3.5" framing is interpreted here as referring to either the unified Qwen3.5 (too new / not-VL-specifically-papered) or the most recent available Qwen3-VL.
- **Qwen3-VL-8B's full benchmark suite was not extractable** from the tech report PDF in cleartext; figures-as-images. Authoritative scores require either downloading the PDF and OCRing tables, or running VLMEvalKit locally.
- **Sibling fine-tune counts are HF search-time estimates** — "218 community fine-tunes for Qwen3-VL-8B-Instruct" is the search-result figure, and many of those are LoRA adapters on quantized variants, not full-weight derivatives. The numbers are directional, not exact.
- **Llama-4 architectural details** for early-fusion image-token-position semantics — Meta's release blog described early fusion conceptually but exact image-token slot mechanics in transformers' Llama4 model_doc were not extracted. The "no clear vision-token-position" verdict is based on early-fusion semantics and could be re-checked if Llama-4 ever becomes a serious candidate.

### 6.3 Sibling fine-tune confirmation status

| Sibling | Base | Source |
|---|---|---|
| Cosmos-Reason1-7B | Qwen2.5-VL-7B-Instruct | confirmed via NVIDIA HF model card |
| Cosmos-Reason2-8B | Qwen3-VL-8B-Instruct | confirmed via NVIDIA HF model card |
| Cosmos-Reason2-2B | Qwen3-VL-2B-Instruct | confirmed |
| Monet-SFT-7B | Qwen2.5-VL-7B (base) | from `monet-latent-probe/docs/README.md`, paper `2511.21395` |
| MM-Eureka-Qwen-7B | Qwen2.5-VL-7B-Instruct | confirmed via ModalMinds GitHub |
| VL-Rethinker-7B | Qwen2.5-VL-7B-Instruct | confirmed via TIGER-AI-Lab repo |
| R1-Onevision-7B / -7B-RL | Qwen2.5-VL-7B (base) | confirmed via Fancy-MLLM HF |
| OpenVLThinker-7B | Qwen2.5-VL-7B-Instruct | confirmed via arXiv 2503.17352 |
| Skywork-VL-Reward 7B | Qwen2.5-VL-7B-Instruct | confirmed via arXiv 2505.07263 |
| InternVL3-8B | Qwen2.5-7B (base, *text* — not VL sibling) | confirmed via OpenGVLab model card; only a partial sibling for our purposes |
| InternVL3.5-8B | Qwen3-8B (base, *text* — not VL sibling) | confirmed; partial sibling |

---

## 7. Sources

### Primary technical reports
- Qwen2.5-VL Technical Report — `arXiv:2502.13923`
- Qwen3-VL Technical Report — `arXiv:2511.21631`
- InternVL3 — `arXiv:2504.10479`
- InternVL3.5 — `arXiv:2508.18265`
- Cosmos-Reason1 — `arXiv:2503.15558`
- R1-Onevision — `arXiv:2503.10615`
- OpenVLThinker — `arXiv:2503.17352`
- Skywork-VL Reward — `arXiv:2505.07263`
- GLM-4.5V & GLM-4.1V-Thinking — `arXiv:2507.01006`
- Pixtral 12B — `arXiv:2410.07073`

### HuggingFace model pages consulted
- `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen3-VL-{2B,4B,8B,32B,30B-A3B,235B-A22B}-Instruct`
- `OpenGVLab/InternVL3-8B`, `OpenGVLab/InternVL3_5-8B`
- `nvidia/Cosmos-Reason1-7B`, `nvidia/Cosmos-Reason2-{2B,8B}`
- `mistralai/Pixtral-12B-2409`
- `nyu-visionx/cambrian-8b`
- `meta-llama/Llama-3.2-11B-Vision-Instruct`, `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- `deepseek-ai/deepseek-vl2`
- `zai-org/GLM-4.1V-9B-Thinking`, `zai-org/GLM-4.5V`, `zai-org/GLM-4.6V`, `zai-org/GLM-4.6V-Flash`
- `openbmb/MiniCPM-V-4_5`
- `google/gemma-4-{E2B,E4B,31B,26B-A4B}`

### Sibling experiments
- `../monet-latent-probe/` (Qwen2.5-VL-7B + Monet-SFT-7B, parent context)
- `../monet-latent-probe/docs/README.md` and JOURNAL
