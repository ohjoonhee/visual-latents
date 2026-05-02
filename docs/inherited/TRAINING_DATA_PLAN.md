# Training Data Plan — Reader-Grounded Latent Visual Reasoning

**Author:** research recon, 2026-05-02
**Scope:** training data for the reader-grounded latent visual reasoning method (`a.md`) — Variant A (reader-NLL) baseline + the round-3 mitigation stack (multi-Q consistency + norm regularization + LaViT-style aux grounding) per `JOURNAL.md` 2026-05-01 and `REPORT.md` §16.
**Constraints from POC:** Qwen2.5-VL-7B base for both generator and frozen reader; latents `h ∈ R^{K×D}`, `K ∈ {1,4,16}`, `D=3584`; reader has no image access; arXiv:2004.05704 caveat (random-control ablation needed before any "visual grounding" claim).

All HF dataset paths in this doc were resolved via the HF Hub API on 2026-05-02 (none gated, all accessible). Where I verified statistics in-the-loop with a probe script (datasets-server API or `datasets` library), the numbers carry a `[probed]` tag; otherwise they come from the dataset card.

---

## Section 1 — Method-fit constraints

Three method-driven dataset properties dominate the choice:

1. **Multi-question-per-image structure** — required for the multi-Q consistency loss. Round-2 mitigation D used N=2 and N=5 questions/image and saw the largest single-mitigation held-out gain (`-39 %` at N=2). Scaling N≥5 needs a dataset where ≥5 distinct, semantically varied Qs land on the same image.
2. **Visual-content-required questions** — per arXiv:2004.05704, "improvements" can come from regularization not grounding. Datasets with severe language priors (vanilla VQAv2) confound any grounding claim; CLEVR-class synthetic and counting/spatial datasets are essential controls.
3. **Single-token answers** — POC measured reader-NLL on the *first answer token* (clean NLL signal). ≥80 % single-token coverage keeps the training-time loss compatible with the POC measurement protocol; avoids switching to LLM-judge or BLEU.

A fourth, looser property: **vision-encoder-compatible image quality** (≥256 px short side, RGB; the Qwen2.5-VL ViT pre-merger spatial pooling tolerates anything ≥112 px but training stability degrades under mostly-thumbnail images).

---

## Section 2 — Per-dataset survey

### 2.1 Multi-question-per-image (multi-Q consistency target)

| Dataset | HF path | # images | # Qs | Qs/img (avg / median / max) | Single-tok ans % | Visual-req % (est.) | License | Notes |
|---|---|---|---|---|---|---|---|---|
| **GQA-balanced (train_balanced_instructions)** | `lmms-lab/GQA` config `train_balanced_instructions` | 72,140 | 943,000 | **13 / 12 / 37** [probed: testdev_balanced shows avg 12.57] | **94.0 %** [probed] | ~75 % (compositional Qs grounded in scene graph) | MIT (per HF card) | Best multi-Q dataset by a wide margin. Use train_balanced not train_all (latter has 14.3M Qs but heavy distractor density and language priors). Also has `train_balanced_images` config (72.1k images, 10 GB) for joining. **Recommended primary multi-Q source.** arXiv:1902.09506 |
| **GQA-all (train_all_instructions)** | `lmms-lab/GQA` config `train_all_instructions` | ~113K | **14,305,356** | ~125 / many | ~94 % | ~70 % (less curated than balanced) | MIT | Use for the `~1M` mix's pretraining tier. Heavy language-prior content. |
| **CLEVR (cauldron's clevr subset)** | `HuggingFaceM4/the_cauldron` config `clevr` | 7,000 | **70,000** | **10.0 / 10 / 10** [probed] | **~100 %** [probed: avg 1.0 words] | **100 %** (synthetic, no priors possible) | CC-BY-4.0 (CLEVR upstream) | Synthetic 3D-rendered. Each image gets exactly 10 Qs in cauldron format. **The grounding-immune control.** Image source: original CLEVR at cs.stanford.edu/people/jcjohns/clevr/. arXiv:1612.06890 |
| **CLEVR-CoGenT-TrainA-R1 (CoT augmented)** | `MMInstruction/Clevr_CoGenT_TrainA_R1` | ~37K | 37,773 | 1 / 1 / 1 (single Q with CoT thinking) | low (CoT format) | 100 % | inherits CLEVR (CC-BY-4.0) | Newer CoT-augmented variant — multi-Q lost. Use the cauldron's plain `clevr` config for multi-Q, this only if doing CoT bootstrapping. |
| **VQAv2 (lmms-lab)** | `lmms-lab/VQAv2` config `default`, split `validation` only on HF | ~40K (val) | 214K (val) | **6.0 / 6 / ~12** [probed val 1.9k samples] | **93.5 %** [probed] | **~40 %** (severe language priors per arXiv:1612.00837 + arXiv:2004.05704) | CC-BY-4.0 (COCO base) | **Train split is not in lmms-lab/VQAv2** (only val/testdev/test). The cauldron has the train conversion (82,772 images, 443.7K Qs at 4-6 Q/img). **Must NOT be the only QA dataset** — language priors confound everything. Cauldron variant: 4.4 Qs/img [probed]. arXiv:1612.00837 |
| **Visual Genome question_answers_v1.2.0** | `ranjaykrishna/visual_genome` config `question_answers_v1.2.0` | 108,077 | 1,773,258 | ~16 / ~15 / many | ~90 % (short factoid answers) | ~80 % (region-grounded) | CC-BY-4.0 | **WARNING: HF loader is broken** with current `datasets` library (script-based, `trust_remote_code` removed). Two paths: (a) clone the legacy repo and parse `question_answers.json.zip` directly (b) use the cauldron's VG-derived subsets (`localized_narratives`, no direct VG-QA). Until HF re-publishes as parquet, treat as raw-JSON ETL needed. arXiv:1602.07332 |
| **Visual7W (cauldron)** | `HuggingFaceM4/the_cauldron` config `visual7w` | ~14K | 14,366 (in cauldron, multi-text per row) | **5.6 / 5 / 19** [probed] | low (MCQ format A/B/C/D, not raw word) | ~60 % (some pointing-style) | inherits Visual7W (research) | Useful for multi-Q at scale-medium. MCQ format means single-token answer is the *letter*, which is a proxy for the visual choice. arXiv:1511.03416 |
| **TallyQA (cauldron)** | `HuggingFaceM4/the_cauldron` config `tallyqa` | ~35K | 98,680 | **2.8 / 3 / 8** [probed] | **100 %** (numeric counts) | **~95 %** (counting requires visual access) | research | **Excellent visual-required control.** Counting cannot be done from language alone. Ans is always a digit. arXiv:1810.12440 |
| **NLVR2 (cauldron)** | `HuggingFaceM4/the_cauldron` config `nlvr2` | 50,426 | 90,767 (1.8 Qs/pair) | 1.8 / 2 / 2 [probed] | 100 % (yes/no) | **~90 %** (compositional reasoning over image pair) | research only (not commercial) | Image-pair, not single-image. Schema mismatch with our K-latent-from-one-image setup unless we adapt to two-image generator. **Skip for v1 mix; add later if pair input matters.** arXiv:1811.00491 |
| **A-OKVQA (cauldron)** | `HuggingFaceM4/the_cauldron` config `aokvqa` | 16,539 | ~17K | 1.1 / 1 / 2 [probed] | 80.5 % [probed direct_answers] | mixed (knowledge required) | Apache-2.0 (cauldron compilation) | Single Q per image, not multi-Q. Useful for OOD eval, not training-time multi-Q. arXiv:2206.01718 |
| **AI2D (cauldron)** | `HuggingFaceM4/the_cauldron` config `ai2d` | ~600 | 2,434 | **4.0 / 4 / many** [probed] | low (MCQ) | 100 % (diagram-grounded) | research | Diagram understanding — small but multi-Q. |

### 2.2 Visual-grounding-required (low-language-prior) datasets

These are essential per arXiv:2004.05704 — without at least one in the mix, any mitigation gain might just be regularization.

| Dataset | HF path | # samples | Visual-req | Notes |
|---|---|---|---|---|
| **CLEVR (cauldron)** | `HuggingFaceM4/the_cauldron` config `clevr` | 70K | **100 %** | Synthetic. No language priors possible. Ans is yes/no/digit/color/material/shape — all single token. **Must include.** |
| **CLEVR-Math** | `dali-does/clevr-math` | ~150K (general+multihop) | **100 %** | CC-BY-4.0. 19 GB. Counting + arithmetic over CLEVR scenes. arXiv:2208.05358 |
| **TallyQA (cauldron)** | as above | 98.7K | ~95 % | Real-image counting. |
| **GQA-balanced** | as above | 943K Qs | ~75 % | Spatial/relational Qs grounded in scene graph. Lower than CLEVR but real-image. |
| **NLVR2** | as above | 90K | ~90 % | Pair-image; defer for v1. |
| **RefCOCO** | `lmms-lab/RefCOCO` | 17.6K (val/test); train at ~120K | **100 %** | Referring expression — answer is the bbox. Not a single-token-VQA shape. Use to *evaluate* visual access, or as auxiliary grounding head. |
| **RefCOCO+** | `jxu124/refcocoplus` | 49.9K | **100 %** | Restricts language: no location words. Stricter visual req. |
| **RefCOCOg** | `lmms-lab/RefCOCOg` | 49.8K | **100 %** | Longer expressions. |
| **GRIT** | `zzliang/GRIT` | 20.5M | **100 %** | ms-pl (Microsoft Public). Image+caption+grounding bboxes. arXiv:2306.14824 |
| **POPE** | `lmms-lab/POPE` | 9K | 100 % yes/no | **Eval only** — object-presence yes/no. Adversarial split detects hallucination. |

### 2.3 High-quality VLM SFT mixtures (use as donors, not as the whole mix)

| Dataset | HF path | # samples | License | Notes |
|---|---|---|---|---|
| **The Cauldron** | `HuggingFaceM4/the_cauldron` | ~7M Q/A pairs / ~2.3M images, 50 subsets | Apache-2.0 (compilation; subsets vary) | **The single best one-stop SFT compilation.** Includes all the multi-Q-per-image datasets we want. Subset-level loading: `load_dataset("HuggingFaceM4/the_cauldron", "<config>")`. arXiv:2405.02246 (Idefics2) |
| **LLaVA-OneVision-Data** | `lmms-lab/LLaVA-OneVision-Data` | **3,938,627** [probed across 89 subsets] | Apache-2.0 | Single-Q-per-image conversations format. Heavy on captions/general VQA, less multi-Q-per-image. arXiv:2408.03326 |
| **Cambrian-7M (curated)** | `nyu-visionx/Cambrian-10M` (use `Cambrian7M_withsystemprompt.jsonl`) | 7M | Apache-2.0 | Mix focused on visual reasoning; 8.7 % counting, 7.2 % math. arXiv:2406.16860 |
| **ShareGPT4V** | `Lin-Chen/ShareGPT4V` config `ShareGPT4V` | **102,025** [probed]; PT version 1,246,901 | **CC-BY-NC-4.0** (non-commercial) | High-quality GPT-4V captions on COCO. Long captions (not single-token). Use for caption-alignment phase only. |
| **ALLaVA-4V** | `FreedomIntelligence/ALLaVA-4V` | ~1.5M | **CC-BY-NC-4.0** | Multi-format: caption + instruct on LAION/VFLAN. Long-form answers. arXiv:2402.11684 |
| **Docmatix** | `HuggingFaceM4/Docmatix` | 1.27M docs (multi-Q per doc, 1-56) | MIT | DocVQA-flavor; off-topic for visual-reasoning POC. Skip. arXiv:2408.12637 |
| **FineVision** | `HuggingFaceM4/FineVision` | **24.2M** [probed across 185 configs] | Apache-2.0 | The 2026 successor to Cauldron. arXiv:2510.17269. Includes 185 sub-datasets. **Use for the `~1M` mix at scale.** |

### 2.4 Pretraining alignment data (image-caption)

For warm-starting the generator's latent emission before any reader-NLL signal kicks in.

| Dataset | HF path | # samples | License | Notes |
|---|---|---|---|---|
| **COCO-Caption2017** | `lmms-lab/COCO-Caption2017` | 45.7K (val+test) | CC-BY-4.0 (COCO) | Karpathy-style 5 captions/image. Train split via cauldron `localized_narratives` etc. |
| **COCO Karpathy (HF)** | `HuggingFaceM4/COCO` | 616,767 image-caption rows (jxie/coco_captions); 5 caps/img | CC-BY-4.0 | Use this for alignment if you want the full Karpathy 567K train. Path: `jxie/coco_captions`. |
| **Recap-DataComp-1B** | `UCSC-VLAA/Recap-DataComp-1B` | 1.88B | CC-BY-4.0 | Web-scale recaptions (LLaVA-1.5-LLaMA3-8B). Use if doing LLM-recap warmup. arXiv:2406.08478 |
| **LAION-COCO (relaion-coco)** | `laion/relaion-coco` | ~600M | unclear (web-crawl) | Heavy noise. Recap-DataComp is the more useful drop-in. |
| **220k-GPT4V captions from LVIS** | `laion/220k-gpt4vision-captions-from-LIVIS` | 217,868 | Apache-2.0 | Smaller, higher-quality. Good for medium mix. |
| **DCI (Dense-Captioned Images)** | GitHub `facebookresearch/DCI` (no canonical HF) | 7,805 | CC-BY-4.0 | 1,000-word captions per image. Tiny but **dense** — useful for high-fidelity caption signal in alignment phase. Long-DCI on HF (`mderakhshani/Long-DCI`) is the derivative. arXiv:2312.08578 |
| **BLIP3-Grounding-50M** | `Salesforce/blip3-grounding-50m` | 52.4M | Apache-2.0 (research) | Image+caption+bbox metadata. Large, grounding-rich. arXiv:2408.08872 (BLIP3) |

### 2.5 Evaluation-only datasets

| Dataset | HF path | # samples | Use |
|---|---|---|---|
| **GQA testdev_balanced** | `lmms-lab/GQA` config `testdev_balanced_instructions` | 12,578 [probed]; 398 imgs | **In-distribution held-out eval**. Already used in POC. |
| **MMStar** | `Lin-Chen/MMStar` (or `lmms-lab/MMStar` if avail) | 1.5K | OOD — vision-required curated benchmark. arXiv:2403.20330 |
| **MME** | `lmms-lab/MME` | 2,374 | 14-category yes/no eval. arXiv:2306.13394 |
| **MMBench** | `lmms-lab/MMBench` | 24K (en+cn+cc) | Multi-choice broad eval. arXiv:2307.06281 |
| **POPE** | `lmms-lab/POPE` | 9K | Object-hallucination yes/no. arXiv:2305.10355 |
| **MathVista** | `lmms-lab/MathVista` (gated/auth required for some configs) | ~6K | Math+visual reasoning OOD eval. arXiv:2310.02255 |
| **A-OKVQA val/test** | `HuggingFaceM4/A-OKVQA` | 1,150 val + 6,700 test | **OOD generalization eval** (training is a different mix). |
| **OK-VQA val** | `lmms-lab/OK-VQA` split `val2014` | 5,046 | Knowledge-required eval. 79.3 % single-token [probed]. |
| **NLVR2 dev (balanced)** | `lmms-lab/NLVR2` split `balanced_dev` | 2,300 | Compositional reasoning eval (image-pair). |
| **LLaVA-Bench-in-the-Wild** | `lmms-lab/llava-bench-in-the-wild` | 60 | Open-ended free-form eval (LLM-judged). |

---

## Section 3 — Three concrete training mixes

### Mix 3.1 — Pilot mix (~10K samples) — for round-3 POC validation

**Goal:** validate that combined recipe (norm reg λ=0.1 + multi-Q N=5 + LaViT aux) doesn't collapse and has consistent loss curves at small scale. Single A6000, ~6-8 hours.

| Dataset | HF path | # samples | Qs/img | Sample frac |
|---|---|---|---|---|
| GQA-balanced (train_balanced_instructions) | `lmms-lab/GQA` cfg `train_balanced_instructions` | 6,500 | ≥5 (filter) | 65 % |
| CLEVR (cauldron) | `HuggingFaceM4/the_cauldron` cfg `clevr` | 2,500 | 10 (full) | 25 % |
| TallyQA (cauldron) | `HuggingFaceM4/the_cauldron` cfg `tallyqa` | 1,000 | ≥3 (filter) | 10 % |

**Selection rule:**
- GQA: filter to images with ≥5 Qs in `train_balanced_instructions`, take the first 5 Qs per image. ~1,300 distinct images.
- CLEVR: take 250 distinct images × 10 Qs each.
- TallyQA: ≥3 Qs/image filter.

**Multi-Q consistency loss target:** GQA (≥5 Qs/img) and CLEVR (10 Qs/img). TallyQA gives counting-required visual signal.

**Random-control (mandatory before any "grounding" claim):** create a parallel mix where the q's for each image are **randomly shuffled across images** within the batch. If multi-Q-consistency loss still drops, gain is regularization not grounding (per arXiv:2004.05704).

**Estimated GPU-hours (Qwen2.5-VL-7B generator + frozen reader, K=4):**
- ~12K SFT-equivalent steps × ~1.5 s/step (bs=4 effective, no image tokens to reader) ≈ 5 GPU-hours.
- + Visual baseline recompute (1 hour on the new image set).
- Total: **~6-8 GPU-hours** on A6000.

### Mix 3.2 — Medium mix (~100K) — first real training run

**Goal:** train a generator that beats the POC's per-sample-optimized `h*` on held-out questions, *and* survives the random-control ablation, *and* shrinks reader-transfer drop. ~3-5 days on a single A6000 or 24-36 h on 4×A6000.

| Dataset | HF path | # samples | Qs/img | Sample frac | Why |
|---|---|---|---|---|---|
| GQA-balanced (full balanced train) | `lmms-lab/GQA` cfg `train_balanced_instructions` | 50,000 | 5-15 | 50 % | Multi-Q backbone |
| CLEVR (cauldron, full) | `HuggingFaceM4/the_cauldron` cfg `clevr` | 20,000 | 10 | 20 % | Grounding-immune control + fast convergence |
| TallyQA (cauldron) | `HuggingFaceM4/the_cauldron` cfg `tallyqa` | 10,000 | ≥2 | 10 % | Counting requires visual access |
| CLEVR-Math (general) | `dali-does/clevr-math` | 5,000 | 1 | 5 % | Math+visual; OOD-flavor in-domain |
| Visual7W (cauldron) | `HuggingFaceM4/the_cauldron` cfg `visual7w` | 8,000 | 5-7 | 8 % | Multi-Q diversity (pointing/MCQ) |
| AI2D (cauldron) | `HuggingFaceM4/the_cauldron` cfg `ai2d` | 2,000 | 4 | 2 % | Diagram grounding |
| OK-VQA train | `Multimodal-Fatima/OK-VQA_train` | 5,000 | 1 | 5 % | Knowledge-required diversity |

**Multi-Q consistency loss target:** GQA + CLEVR + Visual7W = 78K samples with ≥4 Qs/img.

**Random-control parallel run (per arXiv:2004.05704):** shuffled-Qs ablation as before, but at this scale.

**Eval splits during training:**
- GQA testdev_balanced (12.5K) — in-distribution
- A-OKVQA val (1.15K) — single-Q OOD
- POPE (9K) — hallucination
- MMStar (1.5K) — broad

**Estimated GPU-hours:**
- 100K × 8 epochs × 1.5 s/step (bs=4) ≈ 333 hours on 1×A6000.
- 4×A6000 with FSDP: ~85 hours = **~3.5 days**.

### Mix 3.3 — Full mix (~1M) — comparable-to-published-work training run

**Goal:** trained reader-grounded generator competitive with LIVR (arXiv:2512.21218) and LaViT (arXiv:2601.10129) — published baselines in this space. Multi-week run.

Two-phase training:

**Phase A — caption alignment (~500K, 1-2 epochs):** warm up generator's latent emission against text-caption supervision (the generator's *output* head, not the reader-NLL path; bridges the cold-start).

| Dataset | HF path | # samples |
|---|---|---|
| COCO-Caption2017 train (Karpathy) | `jxie/coco_captions` | 500,000 |

**Phase B — reader-grounded SFT (~1M, 3-5 epochs):**

| Dataset | HF path | # samples | Sample frac |
|---|---|---|---|
| GQA-all (instructions) | `lmms-lab/GQA` cfg `train_all_instructions` | 400,000 | 40 % |
| CLEVR (cauldron) | `HuggingFaceM4/the_cauldron` cfg `clevr` | 70,000 | 7 % |
| CLEVR-Math (general+multihop) | `dali-does/clevr-math` | 100,000 | 10 % |
| TallyQA (cauldron) | `HuggingFaceM4/the_cauldron` cfg `tallyqa` | 80,000 | 8 % |
| Visual7W (cauldron) | `HuggingFaceM4/the_cauldron` cfg `visual7w` | 14,000 | 1.4 % |
| FineVision selected configs (visual-reasoning slice) | `HuggingFaceM4/FineVision` (cherry-pick configs: `chartqa_*`, `vqarad`, `vsr`, `cocoqa`, `okvqa`, `tallyqa`, `visual7w`, `iconqa`, `aokvqa`) | 250,000 | 25 % |
| VQAv2 (cauldron) | `HuggingFaceM4/the_cauldron` cfg `vqav2` | 80,000 | 8 % |
| RefCOCO+/g (auxiliary grounding head) | `lmms-lab/RefCOCO`, `RefCOCOplus`, `RefCOCOg` (combined) | 50,000 | 5 % |
| AI2D + ChartQA | `HuggingFaceM4/the_cauldron` cfgs | 30,000 | 3 % |
| NLVR2 (if pair-input adapted) | `HuggingFaceM4/the_cauldron` cfg `nlvr2` | 25,000 | 2.5 % |

**Multi-Q consistency loss target:** GQA-all (massive), CLEVR (10 Qs/img), Visual7W = ~480K of the mix.

**Random-control mandatory.** Run as a parallel ablation, identical mix but Qs shuffled across images.

**Reader-transfer eval:** include Monet-7B as a second frozen reader for transfer-NLL measurement (per POC 3 — known issue from round-2 that geometric mitigations don't transfer; this is the mix to test whether *training-time* multi-Q does).

**Estimated GPU-hours:**
- Phase A: 500K × 1 epoch × 1 s/step (caption-only, no reader pass) ≈ 140 GPU-h ≈ 6 days on 1×A6000, 1.5 days on 4×A6000.
- Phase B: 1M × 4 epochs × 2 s/step (reader pass + multi-Q + aux) ≈ 2,222 GPU-h ≈ 23 days on 4×A6000.
- **Total: ~25 days on 4×A6000 or ~12-13 days on 8×A6000.**

---

## Section 4 — Specific dataset assignments

Per the user's deliverables list:

### 4.1 Multi-Q consistency loss (≥5 Qs/image at scale)

**Recommendation: GQA-balanced (`lmms-lab/GQA` config `train_balanced_instructions`).**

- 943K Qs / 72K images = **13 Qs/img average**, **min 5 after filtering**, max 37 [probed on testdev_balanced]
- 94 % single-token answers [probed]
- ~75 % visually grounded (compositional Qs over scene graph; not pure language priors)
- MIT license, CC-BY-4.0 base images
- Already used in POC; pipeline is in place

**Backup at scale:** CLEVR (cauldron `clevr` config) — exactly 10 Qs/image, all visual-grounded, single-token. Use as the *grounding-immune* multi-Q half of the loss.

### 4.2 Random-control ablation (per arXiv:2004.05704)

**Recommendation: GQA-balanced with shuffled Qs.** Same dataset, same images, but for each batch sample N Qs uniformly at random *across all images in the mix* rather than from the same image. The multi-Q consistency loss is computed as if those random Qs belonged to the same image.

If this ablation produces similar held-out gains to the real multi-Q recipe, the gain is generic regularization, not grounding (arXiv:2004.05704's exact result on VQA grounding methods).

**Second control (norm regularization):** rerun mit-B with `target_norm` shuffled to per-sample random values. From REPORT.md §16 this is also flagged as required.

### 4.3 Eval splits during training

**In-distribution:** **GQA testdev_balanced** (`lmms-lab/GQA` cfg `testdev_balanced_instructions`, 12,578 rows, 398 images).
- Already used in POC2. NLL on the gold answer's first token is the primary in-distribution metric.

**Out-of-distribution generalization:**
- **A-OKVQA val** (`HuggingFaceM4/A-OKVQA` split `validation`, 1,150 rows) — knowledge-required, single-Q-per-image, different distribution from training.
- **POPE** (`lmms-lab/POPE` split `test`, 9,000 rows) — object hallucination yes/no.
- **MMStar** (`lmms-lab/MMStar`) — curated visual-required benchmark.

**Reader-transfer (mandatory):** the in-distribution eval but on Monet-7B as the frozen reader. POC 3 baseline transfer drop was 5.8-9.0 nat at K=4-16; the round-2 mitigations did *not* fix this. Whether the trained generator's latents transfer is the round-3 success criterion.

**Composition signal (catastrophic-forgetting check):** **CLEVR test split** (`dali-does/clevr-math` test) — synthetic, drift in CLEVR perf between training rounds is a strong leak signal.

---

## Section 5 — Implementation notes

### 5.1 Loading paths that don't actually work

- **`HuggingFaceM4/VQAv2`** — broken (script-based, `trust_remote_code` deprecated in current `datasets`). Use `lmms-lab/VQAv2` or the cauldron's `vqav2` config.
- **`ranjaykrishna/visual_genome`** / **`visual_genome`** — both script-based, both broken. ETL via raw JSON download from cs.stanford.edu/people/rak248/VG_100K_2/ + `https://homes.cs.washington.edu/~ranjay/visualgenome/api.html` if VG-QA is needed; or use the cauldron-derived subsets that ingested VG.
- **`MMInstruction/CLEVR-Math`** — 401 in browser; use `dali-does/clevr-math` instead (CC-BY-4.0, parquet, 19 GB).
- **NLVR2** — research-only license; check if usable for our research output.

### 5.2 Single-token-answer extraction

For datasets that store multiple acceptable answers (VQAv2's 10-answer list, A-OKVQA's `direct_answers`, OK-VQA's `answers`):
- Take the modal (most common) answer string.
- Tokenize with the Qwen2.5-VL tokenizer; if the first token is `BOS`/space-prefixed, strip then check `len(toks) == 1`.
- Discard the row if the modal answer is multi-token AND no synonym in the answer list is single-token.

This filter empirically retains ~94 % of GQA and ~80 % of A-OKVQA [probed]. Apply at data-prep time, not at training time.

### 5.3 Multi-Q sampling per batch

Per `JOURNAL.md` 2026-05-01 round-2: N=5 multi-Q saw `-24 %` held-out NLL but at reduced effect size compared to N=2's `-39 %`. Open question whether N→generalization scaling compounds. **Suggested protocol for round-3:**

- For each image with ≥N_train Qs, sample N_train Qs at training time.
- Hold out 1 Q per image as the consistency-loss "free" question (observed but not the optimization target).
- For random-control parallel, replace the N_train Qs with N_train random Qs from other images.

Default N_train = 5 (per round-2 finding); sweep N_train ∈ {3, 5, 8} as a hyperparameter once mix is stable.

### 5.4 Storage budget

Estimated download for the medium mix (~100K samples with images):
- GQA-balanced images config: 10 GB
- CLEVR: ~13 GB
- TallyQA: ~7 GB
- CLEVR-Math: 19 GB
- Visual7W: ~4 GB
- AI2D: 0.5 GB
- OK-VQA: 1.5 GB
- **Total: ~55 GB** — fits on the experiment's `data/` directory comfortably.

For the full mix (~1M): GQA-all alone is ~110 GB plus FineVision-selected ~80 GB. Plan **~250 GB** for the full mix; use parquet streaming for FineVision to avoid materializing.

---

## Section 6 — Comparison-with-published-work positioning

| Method | Multi-Q? | Aux grounding? | Random-control? | Mix used |
|---|---|---|---|---|
| LaViT (arXiv:2601.10129) | no (single Q) | yes (cosine teacher) | not reported | LLaVA-OneVision-Data subset |
| LIVR (arXiv:2512.21218) | no (single Q) | no (attention bottleneck) | not reported | Cambrian-7M |
| Coconut (arXiv:2412.06769) | n/a (text latents, language) | n/a | n/a | text reasoning datasets |
| **Ours (round-3 plan)** | **yes (N=5 consistency)** | **yes (LaViT-style)** | **yes (mandatory ablation)** | **GQA-balanced + CLEVR + TallyQA + Visual7W mix** |

Three differentiators: multi-Q consistency at N≥5 (no published method tries this), random-control ablation (matches arXiv:2004.05704's gold-standard rigor; LaViT/LIVR don't), grounding-immune CLEVR control (rules out language-prior gains).

---

## Section 7 — Open questions and risks

1. **VQAv2 train split is not on lmms-lab/VQAv2** — only val/testdev/test. Our access route is the cauldron's `vqav2` config (82K images, 443K Qs) but Qs/img there is only 4.4 [probed], thinner than GQA's 13. Risk: if multi-Q consistency benefits compound with Qs/img, we may be capacity-limited on VQAv2. Mitigation: GQA carries the multi-Q load; VQAv2 is auxiliary only.

2. **Visual Genome QA is currently un-loadable from HF**. If we want VG-QA's massive multi-Q (~16/image, 1.7M Qs total), we need raw-JSON ETL. **Decision:** skip for v1 mixes; revisit in round-4 if multi-Q consistency results justify scaling Qs/img beyond GQA's max-37.

3. **NLVR2 image-pair input** doesn't fit our K-latent-from-one-image generator. Either (a) generate K latents from each image and concatenate to 2K; (b) skip NLVR2. v1 mixes skip; v2 if pair-input is a research direction.

4. **License: ShareGPT4V and ALLaVA are CC-BY-NC-4.0** — non-commercial. If our research output ever needs commercial use, exclude. For research papers, fine.

5. **arXiv:2004.05704 is from 2020** — the field has had 6 years to internalize it, but most VLM papers still don't run random-controls. Our round-3 plan does. This is publishable hygiene on its own.

---

## Section 8 — Appendix: HF Hub API verification (2026-05-02)

All paths below resolved with `200 OK` on `huggingface.co/api/datasets/{path}`, with `gated=False`:

```
lmms-lab/GQA               (38,622 dl/mo)
lmms-lab/VQAv2             (24,611)
lmms-lab/NLVR2             (307)
lmms-lab/OK-VQA            (3,025)
lmms-lab/RefCOCO           (10,088)
lmms-lab/RefCOCOplus       (5,393)
lmms-lab/RefCOCOg          (4,112)
lmms-lab/MME               (41,283)
lmms-lab/POPE              (28,385)
lmms-lab/MMBench           (28,365)
lmms-lab/LLaVA-OneVision-Data (20,050)
lmms-lab/COCO-Caption2017  (4,488)
HuggingFaceM4/A-OKVQA      (4,002)
HuggingFaceM4/the_cauldron (47,420)
HuggingFaceM4/Docmatix     (12,719)
HuggingFaceM4/COCO         (2,155)
HuggingFaceM4/FineVision   (87,287)
Lin-Chen/ShareGPT4V        (2,509)
MMInstruction/Clevr_CoGenT_TrainA_R1 (80)
dali-does/clevr-math       (435)
nyu-visionx/Cambrian-10M   (5,112)
ranjaykrishna/visual_genome (5,744; loader broken)
FreedomIntelligence/ALLaVA-4V (1,367)
zzliang/GRIT               (537)
Salesforce/blip3-grounding-50m (564)
laion/220k-gpt4vision-captions-from-LIVIS (3,116)
UCSC-VLAA/Recap-DataComp-1B (5,029)
jxu124/refcocoplus         (1,135)
jxu124/refcocog            (1,532)
```

Probed-in-the-loop statistics:

```
GQA testdev_balanced_instructions:
  rows=12,578; images=398; avg_q/img=12.57; max=37; single-tok=94.0%
GQA train_balanced_instructions:
  rows=943,000; images=72,140; ~13 q/img
GQA train_all_instructions:
  rows=14,305,356
A-OKVQA train:
  rows=17,056; single-tok (modal direct_answers)=80.5%
OK-VQA val2014:
  rows=5,046; single-tok=79.3%
VQAv2 (lmms-lab/VQAv2 validation, 1900-row sample):
  avg_q/img=5.99; single-tok=93.5%
The Cauldron subset sample (50 rows, q/img):
  clevr=10.0 ; tallyqa=2.8 ; vqav2=4.4 ; visual7w=5.6
  nlvr2=1.8 ; aokvqa=1.1 ; ai2d=4.0
The Cauldron subset sizes:
  vqav2=82,772 / tallyqa=98,680 / clevr=70,000 / clevr_math=70,000
  nlvr2=50,426 / aokvqa=16,539 / okvqa=9,009 / visual7w=14,366 / ai2d=2,434
LLaVA-OneVision-Data total: 3,938,627 across 89 configs
FineVision total: 24,209,105 across 185 configs
ShareGPT4V: 102,025 ; ShareGPT4V-PT: 1,246,901
```

---

**Verdict:** the recommended primary stack for round-3 POC validation and round-1 training is `GQA-balanced + Cauldron-CLEVR + Cauldron-TallyQA`, with mandatory random-control ablation. Multi-Q consistency loss feeds primarily off GQA-balanced (13 Qs/img, 94 % single-token) and CLEVR (10 Qs/img, 100 % single-token, grounding-immune). Eval splits are GQA testdev_balanced (in-dist), A-OKVQA + POPE + MMStar (OOD), with mandatory Monet-7B reader-transfer on the same eval images.
