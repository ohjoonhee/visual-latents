# Evaluation Benchmark Plan — Reader-Grounded Latent Visual Reasoning

**Audience:** round-3 POC and the proliferated training project.
**Purpose:** specify an eval suite that (i) measures task accuracy and
(ii) **distinguishes "answers using `h`'s visual content" from "answers using
language priors of (q, training-set bias)."** The eval suite is the central
scientific output of this line of work — without it, any benchmark gain is not
attributable to visual grounding and the contribution is unverified
(arXiv:2004.05704; arXiv:2512.21711).

> Single most important reference for this plan: Shrestha et al.
> *Visual Grounding Methods for VQA are Working for the Wrong Reasons!*
> (arXiv:2004.05704, ACL 2020). Verbatim:
> *"performance improvements are not a result of improved visual grounding,
> but a regularization effect which prevents over-fitting to linguistic priors,
> and random, insensible cues also result in similar improvements."*

The plan is in four parts:

- **Part A.** Survey of candidate benchmarks (table-driven).
- **Part B.** A ~5K-sample "visual grounding stress test" composition,
  runnable on 1×A6000 in <30 min.
- **Part C.** Four random-input control conditions every honest claim must
  report, with numerical expectations.
- **Part D.** Latent-specific intervention probes (steering, activation
  patching, mediation analysis).

---

## Part A — Benchmark survey

Each row reports: skill measured, **language-only ceiling** (text-only LLM
accuracy — what a non-visual model can score), saturation status for
7B-class VLMs as of 2026-Q2, sample size, eval protocol, and suitability for
our purpose. Lower language-only ceiling = better at testing visual grounding.

### A.1 Vision-grounding-pure benchmarks (recommended)

| Benchmark | arXiv / HF | Skill | Lang-only ceiling | 7B-class score | Size | Protocol | Verdict |
|---|---|---|---|---|---|---|---|
| **MMVP** | 2401.06209 / `MMVP/MMVP` | Fine-grained perception (CLIP-blind pairs across 9 patterns: orient., count, viewpoint, etc.) | <random (≈25%, 4-choice; both-of-pair scoring drives random to ≈6%) | LLaVA ≈5.5%, GPT-4V ≈40% | 150 pairs / 300 Qs | Pair-accuracy: both Qs in a pair must be correct. Greedy generation, regex-extract MC. | **Top tier.** Adversarial-by-construction; pair scoring kills priors. |
| **MMStar** | 2403.20330 / `Lin-Chen/MMStar` | 6 axes × 18 sub-axes (coarse/fine perception, instance/logical reasoning, sci&tech, math) | Curated to be near-random for blind LLMs | Qwen2.5-VL-7B 63.9 | 1,500 | MC, greedy. Reports MG (multimodal gain) + ML (multimodal leakage). | **Top tier.** Built specifically to require vision; lowest-leakage. |
| **BLINK** | 2404.12390 / `BLINK-Benchmark/BLINK` | 14 perception tasks (rel. depth, correspondence, jigsaw, multi-view, IQ-test, etc.) | ~38% random (chance varies per task) | GPT-4V 51.3, Gemini 45.7 (still much below human 95.7) | 3,807 Qs / ~7,358 imgs (val=1,901; test=1,906) | MC. Eval on val (test labels held out). | **Top tier for fine-grained perception.** Subset selection critical (see B). |
| **NaturalBench** | 2410.14669 / `BaiqiL/NaturalBench` | Image-pairs natural-adversarial VQA (object/attr/relation/count/logic) | "Yes"-bias defeated by paired construction | GPT-4o ≈45–50% (humans ≈90%) | 10,000 paired Qs (≈3.5K image pairs) | Group accuracy: 4 (img,Q) tuples, all must pass. | **Top tier.** Closest to MMVP at scale; best hard-test paired construction. |
| **WinoGround** | 2204.03162 / `facebook/winoground` | Visio-linguistic compositionality (same words, different order) | At-chance for blind models (forced by construction) | All tested models at-or-below chance on group score | 400 examples / 800 imgs / 800 caps | Image score, text score, group score. | **Useful but small.** Group score is brutal; many pairs are unsolvable even by humans (~85% noise). |
| **VSR** | 2205.00363 / `cambridgeltl/vsr_zeroshot` | Spatial relations (66 types: under/in front of/facing) | ≈50% (T/F binary; lang-only ≈chance after balancing) | ~70% (humans 95%+) | 10,119 (zeroshot test ≈1,222) | Binary T/F. Greedy gen → regex. | **Strong.** True/False format makes blank-image control sharply interpretable. |
| **CV-Bench** | (in 2406.16860) / `nyu-visionx/CV-Bench` | 2D (count, spatial), 3D (depth order, rel. distance) | ~50% (binary/MC) | Cambrian-1 ~70% | 2,638 | MC. | **Strong.** Repurposed COCO/ADE20k/Omni3D — very pure perception. |
| **POPE** | 2305.10355 / `lmms-lab/POPE` | Object hallucination (binary "is X in image?") | ≈50% (Yes-bias usually drives lang-only above) | Qwen2.5-VL-7B ~88% (random/popular splits) | 9,000 Qs / 3,000 imgs (3 splits: random/popular/adv.) | Binary Yes/No, F1. | **Strong, mostly saturated** at 7B. Adversarial split still informative. |
| **TallyQA** | 1810.12440 | Counting (simple + complex) | Number priors (3, 2, 4 over-represented) | Recent VLMs 60–75% on complex | 287K Qs (subset 5–10K typical) | Open numeric. Score on integer match. | **Useful.** "Complex" subset is harder, less prior-exploitable. |

### A.2 Mixed benchmarks (somewhat language-prior-leaky)

| Benchmark | arXiv / HF | Skill | Lang-only ceiling | 7B-class score | Size | Protocol | Verdict |
|---|---|---|---|---|---|---|---|
| **MMBench** | 2307.06281 / `lmms-lab/MMBench` | 20 ability dims, 4-choice MC | GPT-4 text-only ≈random under circular eval per paper §5 | Qwen2.5-VL-7B 82.6 (V1.1-En test) | dev 4,377 / test 6,666 (V1.1) | "Circular evaluation" (rotate options 4×, all must pass). | **Useful but saturated.** Use circular eval for honest scoring. |
| **MM-Vet** | 2308.02490 / `whyu/mm-vet` | 6 capabilities × 16 task combos, open-ended | Some lang-only headroom (open-ended) | 7B class 40–60% | 218 | LLM-judge (GPT-4 grader). | **Niche.** Open-ended is great signal but expensive (judge cost) and small. |
| **MM-Vet-v2** | 2408.00765 | Adds image-text interleaved sequences | Similar | 7B class 40–55% | 517 | LLM-judge. | Same as MM-Vet at larger scale. |
| **NLVR2** | 1811.00491 (`lil-lab/nlvr`) | Paired-image binary T/F (compare two photos) | ~50% (chance) | ~75–82% | dev 6,982 / test 6,967 | Binary. | **Strong, somewhat dated.** Two-image format is unusual for our reader. |
| **ScienceQA** (image subset) | 2209.09513 / `derek-thomas/ScienceQA` | Sci. MC w/ lecture+explanation; image-context subset | **57.2% answerable lang-only** (per MMStar §3) | Qwen2.5-VL-7B ~85% (image subset) | ~10,332 imgs subset of 21,208 | MC. | **Avoid.** Highest leakage of any candidate; unsuitable for grounding test. |
| **MathVista** | 2310.02255 / `AI4Math/MathVista` | Math+visual (charts, geometry, diagrams) | Substantial (LLM math ability + chart-text) | Qwen2.5-VL-7B 68.2 (testmini) | 6,141 | LLM-judge or numeric match. | **Skip for grounding test.** Confounds math reasoning with visual grounding. |
| **MMMU** | 2311.16502 | Expert multi-disc. multimodal | **GeminiPro 42.9 % blind** (heavy leakage) | Qwen2.5-VL-7B 58.6 (val) | 11,500 | MC. | **Skip for grounding test.** Confounds knowledge with vision. |
| **RealWorldQA** | xAI 2024 (no arXiv) / `xai-org/RealworldQA` | Real-world spatial (driving photos) | Random ≈37.7% | GPT-4V 68%, LLaVA-NeXT-Yi-34B 66% | 765 | MC. | **Useful**, smallish. Decent grounding pressure. |
| **HR-Bench (4K/8K)** | 2408.15556 | Fine-grained at high resolution (FSP, FCP) | Low (resolution is the point) | Most 7B models 40–55% | 200 (each 4K and 8K) | MC. | **Niche.** Worth including if reader handles 4K; otherwise skip. |
| **V*Bench** | 2312.14135 | High-res visual search (attribute, spatial) | Low | GPT-4V ~55%, V*-LLaVA ~76% | 191 (115 attr + 76 spatial) | MC. | **Useful, small.** High-res focus may not survive our 14×14 patch slot budget. |
| **RefCOCO/+/g** | classic / `lmms-lab/RefCOCO` | Referring-expression localization (bbox) | N/A (output is bbox) | CogVLM 92.4/88.5/90.7 (Acc@0.5) | val/testA/testB ~5–10K each | IoU > 0.5. | **Saturated and label-noisy** (14% errors on RefCOCO). Skip unless we want loc. specifically. |
| **Visual7W** | 1511.03416 | Telling (text answer) + pointing (bbox) | Some lang prior (multiple-choice) | ~80–85% on telling for 7B | telling ~140K, pointing ~120K | MC for telling. | **Dated, partly subsumed by GQA/RefCOCO.** Skip. |
| **ARO** | 2210.01936 | Attribution / Relation / Order (text-image matching) | Designed against bag-of-words; chance-pair scoring is ~50% | CLIP variants 50–60%, contrastive-tuned 70–80% | Visual Genome Attr+Rel + COCO/Flickr30k order, 50K+ | Image-text matching score. | **Great in spirit, awkward for generative VLMs.** Original protocol is contrastive-style; needs adaptation. |
| **CLEVR / CLEVR-Math** | 1612.06890 / 2208.05358 | Synthetic compositional (count, color, shape, spatial; +math actions) | Low for the well-balanced sets | 7B-class 60–80% on CLEVR; lower on CLEVR-Math multi-hop | val 150K / test 150K (CLEVR); CLEVR-Math much smaller | Open answer or MC. | **Useful for diagnostic isolation.** Synthetic, programmatic — great for ablations. |

### A.3 Language-only ceiling: bottom line

Under the lens "how much can a blind LLM score?", the candidates rank
roughly (lower = better for our purpose):

1. **MMVP** (~6% with pair scoring), **NaturalBench** (group score), **WinoGround** (group) — adversarially designed against priors.
2. **MMStar**, **BLINK**, **CV-Bench** — curated/repurposed for visual necessity.
3. **VSR**, **POPE**, **TallyQA-complex** — format and balancing limit priors.
4. **MMBench**, **NLVR2**, **RealWorldQA** — moderate priors.
5. **ScienceQA**, **MMMU**, **MathVista** — *high* leakage; avoid for grounding tests.

---

## Part B — The 5K-sample visual grounding stress test

### B.1 Design constraints

- **Total ≤ 5,000 samples** (target ≈4,500 for headroom).
- **<30 min on 1×A6000** at 7B-class. Budget: ~0.4 s/sample greedy
  generation at 64 token budget × 4,500 = 30 min. 4-control multiplier (Part C)
  pushes total wall-clock to ~2 hr per full eval pass — acceptable.
- **Diverse skills.** Counting, color, spatial, relation, attribute, OCR, fine-grained.
- **Adversarial pairs included.** MMVP-style pair construction so the *gain*
  metric (not absolute accuracy) is the headline.
- **Per-skill subscore + overall.** Avoid hiding asymmetric drops in the
  aggregate.

### B.2 Composition

| Source | Subset | Skill(s) | Size | Why |
|---|---|---|---|---|
| **MMVP** | full | Adversarial fine-grained (9 patterns) | 300 (150 pairs) | Adversarial flagship; pair scoring is the cleanest grounding signal in any VLM eval. |
| **NaturalBench** | 1,000 sampled | Object, attr, relation, count, logic — paired | 1,000 | Scaled MMVP-spirit; group score robust to "yes"/"B" bias. |
| **BLINK** (val) | 7 most-perception-pure subtasks: **Spatial Relation, Relative Depth, Counting, Visual Correspondence, Object Localization, Multi-view Reasoning, Jigsaw** | Spatial, depth, count, corresp., localization, multi-view, structure | ≈900 (≈130 each, val splits) | These 7 resist language-prior shortcuts more than the other 7 (e.g., Art Style, IQ Test which leak via knowledge). |
| **MMStar** | full | Mixed perception + reasoning (validates aggregate skill profile) | 1,500 | Reference benchmark with built-in MG/ML; sanity check overall capability. |
| **CV-Bench** (3D subset) | full 3D split (depth + relative distance) | Depth ordering, rel. distance | ~750 | 3D understanding; well-controlled MC; few language priors. |
| **POPE** (adversarial split) | full adversarial split | Object hallucination under hard distractors | 3,000 → **subsample 500** | Pure binary perception. Adversarial distractors target priors. |
| **VSR** (zero-shot test) | full zero-shot test | Spatial relations | 1,222 → **subsample 500** | T/F format makes the blank-image control crisp. |

**Total: ~4,950 samples.** Per-skill aggregation:

| Skill bucket | Sources | Approx. size |
|---|---|---|
| Adversarial fine-grained | MMVP + NaturalBench | 1,300 |
| Spatial / depth | BLINK-Spatial, BLINK-Depth, CV-Bench 3D, VSR | 1,500 |
| Counting | BLINK-Counting, MMStar (counting subset), NaturalBench (count subset) | ~400 |
| Object existence (hallucination) | POPE-adversarial | 500 |
| Correspondence / structure | BLINK-Correspondence, BLINK-Multi-view, BLINK-Jigsaw | 400 |
| Mixed (reference) | MMStar | 1,500 |

**Skipped on purpose:** ScienceQA, MMMU, MathVista (high lang-only ceiling),
MMBench (saturated, leakage tested elsewhere), RefCOCO\* (saturated +
label-noisy + bbox output incompatible with reader-only inference).

### B.3 Eval protocol per dataset

| Dataset | Format | Decoding | Scoring |
|---|---|---|---|
| MMVP | 2-choice yes/no per Q in a pair | Greedy, max 16 tok | Pair-accuracy: both correct → 1, else 0. Regex-extract Yes/No. |
| NaturalBench | 2 (Q, img) tuples × 2 imgs = 4 cells per group | Greedy, 16 tok | **Group score** (all 4 correct), plus Q-acc, I-acc, sample-acc per paper. |
| BLINK | 2–4 choice MC | Greedy, 32 tok | Letter-extract from output (paper's `extract_answer.py`); accuracy. |
| MMStar | 4-choice MC | Greedy, 32 tok | Paper's official extractor; per-axis accuracy + overall. |
| CV-Bench | 2/4-choice MC | Greedy, 32 tok | Letter-extract; report 2D vs 3D split. |
| POPE | Yes/No | Greedy, 16 tok | Yes/No regex; precision/recall/F1 + accuracy on adversarial split. |
| VSR | Binary (caption true about image?) | Greedy, 16 tok | Yes/No regex; accuracy + per-relation breakdown. |

**Greedy generation everywhere** for reproducibility. Prompt template fixed
across runs (write once into `eval/prompts.py`, version with the run dir).

**No logprob comparison as primary scoring.** For our reader-grounded setup,
generation accuracy is the headline because it's what a deployed system
delivers. Logprob-of-gold can be reported as a secondary metric in the JSONL
per-item record (cheap to compute alongside greedy gen).

### B.4 Aggregation

```
overall_acc = sum(correct) / sum(total)        # micro-averaged
per_skill_acc[skill] = micro-avg over sources in that skill
per_source_acc[source] = micro-avg per source
```

Report all three. Headline figure is a **per-skill bar plot with the
Δ (method − baseline VLM) overlaid**; the overall scalar is for the
abstract.

### B.5 Wall-clock and crash resilience

- 4,950 samples × ~0.4 s ≈ 33 min greedy on A6000 at bs=1 with FlashAttn.
- If we hit OOM at high-res (HR-Bench/V*Bench-style images aren't in this
  mix; longest images are RealWorldQA-class which we excluded), this stays
  within 49 GB.
- Per-item JSONL append: `{sample_id, source, skill, pred, gold, correct,
  logprob_gold, time_s}`. Resume on restart by globbing.

---

## Part C — Random-input control conditions

For every claim about visual grounding, **the same eval suite must be run
under each of the four controls below.** The relative drops, not absolute
accuracy, are the load-bearing measurements.

The honest-claim contract:

> Our method drops by Δ_method on control X. A vanilla VLM drops by
> Δ_vanilla. **If Δ_method ≈ Δ_vanilla, no visual-grounding gain is
> claimed beyond what the vanilla VLM already does.** If Δ_method >>
> Δ_vanilla (our method is *more* sensitive to visual content removal),
> we have evidence of stronger visual grounding.

Reported as a 4-row × N-skill table per method.

### C.1 The four controls

| Control | Construction | What it tests |
|---|---|---|
| **C1 — Blank gray** | All pixels = (128, 128, 128) at the dataset's image size. | Pure language-prior baseline. Anything answered correctly is from priors + the question text. |
| **C2 — Random natural** | Replace image $I$ with a random *different* image $I'$ from the same source dataset (sampled once at eval-build time, fixed seed). | Tests whether the model is *attending to* the image at all, vs ignoring it. A blind model does the same on $I$ and $I'$. |
| **C3 — Adversarial mismatch** | Replace $I$ with an image that *contradicts* the question (constructed via two strategies: (a) NaturalBench/MMVP partner image — already mismatch by construction; (b) for non-paired sets, use a CLIP-based retrieval to get the highest-similarity image whose answer differs). | Strongest grounding test. A grounded model should answer based on the (wrong) image and therefore be *wrong* relative to the original gold. Drop = 1 − P(answer-from-wrong-image == original-gold). A purely prior-driven model is invariant to this swap. |
| **C4 — Shuffled pixels** | Random permutation of pixels (color statistics preserved, spatial structure destroyed). Same shuffle seed per sample for reproducibility. | Tests reliance on color/texture statistics vs structure. A model that only uses CLIP-style global stats will drop *less* here than a model that uses spatial structure. |

### C.2 Numerical expectations (back-of-envelope)

Notation. `acc_clean` = the model's accuracy on the unshuffled, unblanked,
matched eval suite. `Δ_X = acc_clean − acc_X`.

For each cell, "non-grounded VLM" is a vanilla 7B-class model that learned
to lean on language priors (call this the typical Qwen2.5-VL-7B baseline).
"Perfectly grounded VLM" is the hypothetical model whose only path to the
answer is through the image content (i.e., answer is independent of the
question text given the image, in the bayes-optimal sense — this is the
"extracts answer from image" extreme).

Lang-prior ceiling per skill (from §A.3): we assume **~35% lang-only floor
for MC and ~50% for binary**, with skill-specific variance.

**Skill: counting / spatial / fine-grained perception** (chance ≈25% MC, ≈50% binary)

| Method | C1 (blank gray) | C2 (random nat.) | C3 (adv. mismatch) | C4 (shuffled) |
|---|---|---|---|---|
| Non-grounded VLM (relies on priors) | ≈ priors-only baseline (~30–35% MC, ~55–60% binary). **Δ small** (≈10–20 pp) — most accuracy is priors. | Similar to C1 (random img ≈ no img signal). **Δ similar to C1** (≈10–20 pp). | Slightly worse than C1, since the wrong image's CLIP global signal can match its own caption-prior; but if priors dominate, **Δ ≈ C1** still. | C1-like (priors don't care about pixel structure). **Δ small** (~5–15 pp). |
| Perfectly grounded VLM | ~ chance (no info). **Δ large** (≈clean − chance, e.g. 70 → 25, Δ≈45 pp). | ~chance (random img is independent of Q). **Δ ≈ C1.** | **Below chance** in expectation (model confidently outputs answer for the wrong image, which by construction disagrees with original gold). **Δ > C1, C2.** | ~chance if grounding requires structure; closer to clean if texture/global stats suffice. **Δ between C1 and clean.** |

**Skill: object existence (POPE)** (chance = 50%, but Yes-bias drives lang-only to ~70%)

| Method | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| Non-grounded | ~70% (Yes-bias). Δ ≈ 15 pp. | ~70%. Δ ≈ 15 pp. | ~50% (mismatch flips half). Δ ≈ 35 pp. | ~70%. Δ ≈ 15 pp. |
| Perfectly grounded | 50% (chance). Δ ≈ 35 pp. | 50% (chance + dataset distribution). Δ ≈ 35 pp. | **~10–20%** (consistently wrong via wrong image). Δ ≈ 65–75 pp. | ~50%. Δ ≈ 35 pp (shuffled = no structure ≈ chance). |

**Skill: high-leakage (ScienceQA, MMMU)** — *not in our stress test, but for context*

| Method | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| Non-grounded | **~text-only score (e.g. 50–60%)**. Δ ≈ 5–15 pp. | Same. | Same. | Same. |
| Perfectly grounded | Same as non-grounded (because the question text + knowledge already carry most of the signal — *image redundancy*). Δ ≈ 5–15 pp. | Same. | Same. | Same. |

→ **Why we excluded these from §B.** On these benchmarks, even a perfectly
grounded model wouldn't show a large drop, because the image carries
redundant information. The control is meaningless.

### C.3 The decision rule for our paper

For each skill bucket, report (Δ_C1, Δ_C2, Δ_C3, Δ_C4) for our method and
for a vanilla VLM baseline. Claim grounding only when:

- **Δ_C1(ours) > Δ_C1(vanilla)** by a meaningful margin (≥5 pp on a skill
  with clean accuracy ≥60%), AND
- **Δ_C3(ours) > Δ_C3(vanilla)** (adversarial mismatch produces *worse*
  performance for our model than for vanilla — i.e., we trust the image
  more), AND
- **Δ_C2 ≈ Δ_C1** (random-image and blank are interchangeable — confirms
  the model isn't doing something weird with the random image's stats).

If our method matches vanilla on all four Δs, we have *zero* evidence of
visual grounding *beyond what vanilla already does*. This is the
arXiv:2004.05704 outcome.

### C.4 Implementation notes

- **C1 (blank gray)** is dirt-cheap: pre-compute one PIL image at each
  resolution used in the suite and reuse.
- **C2 (random natural)**: sample once at eval-build time with a fixed
  seed; record `(sample_id, swap_id)` mapping in `data/random_swap.jsonl`.
  Re-use across method comparisons for paired statistical tests.
- **C3 (adversarial mismatch)**: for paired benchmarks (MMVP, NaturalBench)
  the "mismatch" image is just the partner. For non-paired sets, build a
  CLIP-based retrieval index over the source set, retrieve top-100, filter
  to the highest-CLIP-similarity image whose ground-truth answer differs
  from the original. Cache once.
- **C4 (shuffled pixels)**: numpy `np.random.permutation` over the flat
  pixel array, fixed seed per `sample_id`. Pre-shuffle and cache as
  PNG/JPEG to avoid wasting compute repeatedly.

---

## Part D — Latent-specific intervention probes

These are the probes that make the difference between "plausible benchmark
gain" and "we know what the latents do." Already implemented in round 2:
`zero_pos`, `permute_within`, `permute_across`, `gauss_noise` (in
`steering_probe.py`). The proliferated project should add the following.

### D.1 Coconut-critique steering protocol (arXiv:2512.21711)

**What it tests.** Whether `h` carries reasoning-critical content vs being
an inert placeholder.

**Protocol (verbatim from the paper, adapted to vision):**

- For each (sample, $h_{1:K}$, gold), run 8 perturbation conditions and
  measure **answer flip rate** (fraction of samples where the greedy answer
  changes) and **NLL-of-gold delta**:
  - Zero out `h_i` (one position $i$ at a time, K conditions).
  - Replace `h_i` with random Gaussian (matched mean/std of the latent
    distribution).
  - Permute positions within a sample.
  - Replace all `h` with another sample's `h` (cross-sample swap).
  - Add Gaussian noise of σ ∈ {0.1, 0.5, 1.0} × natural-token σ.
  - Truncate K (drop last `k` latents, k ∈ {1, K/2, K-1}).

The paper's headline metric: **CoT tokens reach up to 50% perturbation
success rate; COCONUT tokens stay below 5%.** For our setup:

| Latent class | Expected perturbation success | Interpretation |
|---|---|---|
| Trained latents that carry visual content | **>30% answer flip on `zero_pos` and `cross-sample swap`** | Latents matter; perturbations break the answer. |
| Inert / placeholder latents | <5% flip | Latents are inert, model uses other paths (fail mode). |
| Shortcut latents (q-specific not image-specific) | High flip on `zero_pos`, low flip on `cross-sample swap` (within-question)| Latents carry q-specific shortcut, not image content. |

**This is the primary probe to disambiguate inert vs grounded vs shortcut.**
Round 2 found the round-2 mit-A/B latents *are* causally functional under
zero/noise perturbation (high flip rate), but the held-out-question
collapse on POC 2 says the content is q-specific, not image-specific. The
round-3 protocol must report **both** flip rate (Coconut probe) and
held-out-question accuracy (POC 2) to distinguish these three cases.

### D.2 Activation patching / causal mediation (NOTICE-style)

**What it tests.** *Where* in the reader's forward pass `h` actually
affects the answer logits — and whether semantic perturbations to the
image translate into corresponding changes in `h`.

**Protocol** (NOTICE; arXiv:2406.16320):

1. Build minimal-pair examples: for each `(I, q, y)`, construct a
   semantically-perturbed image `I'` such that the gold answer changes to
   `y'`.
2. Run the generator on `I` and on `I'`, caching per-layer hidden states.
3. **Patch** the cached activations from `I'` into the `I` forward pass at
   layer `ℓ`, position `j`, and measure:
   - `IndirectEffect(ℓ,j) = P(y' | patched run) − P(y' | clean run)`.
4. Plot a heatmap of indirect effect vs (layer, position).

**Expected result for grounded latents.** A clear "image" → `h` →
"answer" causal pathway: patching the latent positions transfers the
answer flip; patching irrelevant positions (text-prefix, padding) does not.

**Expected result for shortcut latents.** Diffuse pattern; the answer flip
arrives via direct text-image fusion paths in the *generator*, with `h`
contributing noise.

**Note:** NOTICE was developed for BLIP. We need to adapt for
Qwen2.5-VL (decoder-only with interleaved visual tokens). The patch
location semantics carry over; the Symmetric-Token-Replacement trick for
text doesn't directly apply (we're patching visual slots), but the
Semantic-Minimal-Pair construction transfers cleanly.

### D.3 Input perturbation probe (Li et al., arXiv:2602.22766)

**Already named in HANDOFF.md but not yet run.** Generator-side test of
**Input-Latent Disconnect.**

**Protocol:**

1. For each `(I, q)`, compute `h_clean = generator(I, q)`.
2. Apply image perturbations: (a) pixel scramble, (b) swap with another
   random image, (c) blank gray.
3. Compute `h_perturbed = generator(I_perturbed, q)`.
4. Measure cosine similarity `cos(h_clean, h_perturbed)` per perturbation.

**Expected result for grounded latents.** Cosine drops sharply (e.g.,
< 0.5) under image swap or scramble — `h` actually attends to image
content.

**Expected result for disconnected latents.** Cosine remains high
(> 0.85) — Li et al.'s exact failure mode. **Hard fail.**

This probe is generator-side, so it requires a trained generator to run
(POC 1's per-sample-optimized latents skip this question entirely; the
Input-Latent Disconnect probe is the canary for the trained Variant A run).

### D.4 ROME / MEMIT / model editing — *not recommended*

ROME (Meng et al., 2022) and MEMIT (Meng et al., 2023) are *causal
intervention* tools used for *fact editing* in transformer MLPs. They
find rank-1 updates to MLP weights that change a stored fact. Activation
patching (D.2) is what they use as the analysis primitive; the *editing*
half is not what we need. **Recommend: borrow the activation-patching
methodology (D.2), not the rank-1 editing.** Including ROME/MEMIT-as-
editing in our suite would conflate grounding analysis with knowledge
editing and isn't aligned with our research question.

### D.5 Recommended round-3 / proliferated-project probe protocol

For every checkpoint of the trained generator (and every per-sample
optimized run in extended POC studies), produce the following table of
numbers:

| Probe | Source | Metric | "Pass" threshold |
|---|---|---|---|
| Steering — zero_pos | round-2 `steering_probe.py` (already implemented) | ΔNLL on gold | ≥1 nat |
| Steering — gauss_noise σ=1.0 | same | ΔNLL on gold | ≥1 nat |
| Steering — permute_within | same | ΔNLL on gold | ≥0.5 nat |
| Steering — permute_across | same | ΔNLL on gold | ≥1 nat |
| Steering — answer flip rate | NEW (Coconut protocol, D.1) | ≥30% on zero_pos and cross-sample swap | gate before reading other results |
| Held-out question (POC 2) | already implemented | Held-out NLL ≪ no-input NLL | gate against q-specific shortcut |
| Input-Latent disconnect (D.3) | NEW (generator-side) | cos(h_clean, h_swap) | <0.7 |
| Activation patching (D.2) | NEW (NOTICE-style) | Indirect effect concentrated at latent positions | qualitative |
| Reader transfer (POC 3) | already implemented | Acc on Monet-7B | non-trivial above lower bound |
| Eval suite ×4 controls (Part C) | NEW | Δ_C3(ours) > Δ_C3(vanilla) | claim grounding only if pass |

A method passing the steering and held-out probes but failing the
×4 controls is **not** showing visual grounding — it's showing latents
that carry q-specific content (which round 2 already established for the
mit-B/A recipes). The proliferated project's bar should be passing **all
four** of: steering, held-out-Q, ×4 controls, and Input-Latent disconnect.

---

## Part E — Deliverables for round 3 / proliferated project

In `experiments/<exp>/eval/`:

```
eval/
  build_grounding_mix.py       # downloads + composes the 5K mix
  build_controls.py            # generates C1–C4 variants, caches to disk
  run_eval.py                  # runs greedy gen on (mix, control) → JSONL
  score_eval.py                # per-skill / per-source / overall
  steering_probe.py            # already exists (round 2); extend with D.1
  patching_probe.py            # NEW — D.2 NOTICE-style
  input_disconnect_probe.py    # NEW — D.3 generator-side
  prompts.py                   # versioned prompt templates
data/
  grounding_mix_v1.jsonl       # the composed 5K
  random_swap.jsonl            # C2 fixed swaps
  adv_mismatch.jsonl           # C3 retrieved mismatches
  shuffled/<sample_id>.png     # C4 cached
results/<run>/
  config.json                  # commit, model, latent-source, control-set
  scores.json                  # all (skill × control) cells
  per_item.jsonl               # one row per (sample, control)
  probes/                      # D.1–D.4 outputs
  ANALYSIS.md                  # auto-generated table + plot
```

Compute budget per checkpoint: **~2 hours wall-clock** including all four
controls (clean + C1 + C2 + C3 + C4 = 5 runs × ~30 min = 2.5 hr; tighten
with batched generation).

---

## References (verified)

All arXiv IDs below were checked to exist and the cited claim was
verified against the abstract/HTML at retrieval time (date: 2026-05-02).
Where a quote is given verbatim, it is from the paper's HTML render.

- Shrestha et al. *Visual Grounding Methods for VQA are Working for the
  Wrong Reasons!* — **arXiv:2004.05704** (ACL 2020). The
  random-cue equivalence finding is the headline; the method's
  honest-claim contract follows from this paper.
- Tong et al. *Eyes Wide Shut? Exploring the Visual Shortcomings of
  Multimodal LLMs* (MMVP) — **arXiv:2401.06209** (CVPR 2024). 150 pairs /
  300 Qs; pair-accuracy scoring.
- Fu et al. *BLINK: Multimodal Large Language Models Can See but Not
  Perceive* — **arXiv:2404.12390** (ECCV 2024). 14 perception tasks,
  3,807 Qs, val=1,901 / test=1,906.
- Chen et al. *Are We on the Right Way for Evaluating Large
  Vision-Language Models?* (MMStar) — **arXiv:2403.20330** (NeurIPS
  2024). 1,500 samples; 22,401→1,500 curation; MG/ML metrics.
- Li et al. *NaturalBench: Evaluating Vision-Language Models on Natural
  Adversarial Samples* — **arXiv:2410.14669** (NeurIPS 2024). 10,000
  paired Qs; group-score evaluation.
- Thrush et al. *Winoground: Probing Vision and Language Models for
  Visio-Linguistic Compositionality* — **arXiv:2204.03162**.
- Liu et al. *Visual Spatial Reasoning* (VSR) — **arXiv:2205.00363**
  (TACL 2023). 10,119 binary T/F.
- Tong et al. *Cambrian-1* (CV-Bench) — **arXiv:2406.16860** (NeurIPS
  2024). 2,638 examples; 2D + 3D splits.
- Li et al. *Evaluating Object Hallucination in Large Vision-Language
  Models* (POPE) — **arXiv:2305.10355**. 9,000 Qs, 3 splits
  (random/popular/adversarial).
- Acharya et al. *TallyQA: Answering Complex Counting Questions* —
  **arXiv:1810.12440** (AAAI 2019).
- Lu et al. *Learn to Explain: Multimodal Reasoning via Thought Chains
  for Science Question Answering* (ScienceQA) — **arXiv:2209.09513**.
  Per MMStar §3.3, 57.2% answerable lang-only.
- Lu et al. *MathVista* — **arXiv:2310.02255**.
- Liu et al. *MMBench: Is Your Multi-modal Model an All-around Player?*
  — **arXiv:2307.06281** (ECCV 2024).
- Yu et al. *MM-Vet* — **arXiv:2308.02490** (ICML 2024). 218 examples,
  GPT-4 judge.
- Yu et al. *MM-Vet v2* — **arXiv:2408.00765**. 517 examples.
- Suhr et al. NLVR2 — **arXiv:1811.00491**.
- Wu et al. *V*: Guided Visual Search as a Core Mechanism in Multimodal
  LLMs* (V*Bench) — **arXiv:2312.14135**. 191 Qs.
- Wang et al. *Divide, Conquer and Combine* (HR-Bench 4K/8K) —
  **arXiv:2408.15556** (AAAI 2025). 200 examples per resolution.
- Yuksekgonul et al. *When and why vision-language models behave like
  bags-of-words?* (ARO) — **arXiv:2210.01936** (ICLR 2023).
- Johnson et al. *CLEVR* — **arXiv:1612.06890**.
- Lindström & Abraham *CLEVR-Math* — **arXiv:2208.05358**.
- Yu et al. *RefCOCO* (paper-version varies); benchmark family classic.
  REC saturation: CogVLM achieves 92.4/88.5/90.7 on RefCOCO/+/g; 14% /
  24% / 5% labeling-error rates documented.
- Zhu et al. *Visual7W: Grounded Question Answering in Images* —
  **arXiv:1511.03416**.
- xAI *RealWorldQA* — released 2024-04 with Grok-1.5V; 765 Qs; no arXiv.
- Zhang et al. *Do Latent Tokens Think? A Causal and Adversarial
  Analysis of Chain-of-Continuous-Thought* — **arXiv:2512.21711**.
  Steering protocol: COCONUT tokens <5% perturbation success vs CoT
  ≥50%.
- Li et al. *Imagination Helps Visual Reasoning, But Not Yet in Latent
  Space* (CapImagine) — **arXiv:2602.22766**. Causal Mediation Analysis;
  Input-Latent and Latent-Answer Disconnect.
- Goh et al. *What Do VLMs NOTICE?* — **arXiv:2406.16320** (NAACL 2025).
  Semantic-Minimal-Pair image corruption for activation patching in
  VLMs.
- Heimersheim & Nanda *Towards Best Practices of Activation Patching* —
  **arXiv:2309.16042**. Methodology reference.
- Meng et al. *Locating and Editing Factual Associations in GPT* (ROME)
  — **arXiv:2202.05262**. Activation-patching primitive, not the
  rank-1 editing, is what we adopt.

---

## Appendix — quick-reference: what each control answers

Q: "Is our method using language priors more than vanilla?"
→ Compare **Δ_C1** (blank gray): if Δ_C1(ours) > Δ_C1(vanilla), our model
is *less* prior-dependent (drops more when image is removed).

Q: "Is our method actually attending to the image at all?"
→ Compare **Δ_C2** (random natural): random image should drop ≈ blank if
attention is real. If Δ_C2 < Δ_C1 by a lot, the model is using random
image's stats (e.g., color histograms) rather than content.

Q: "Does the answer follow the image content?"
→ Compare **Δ_C3** (adversarial mismatch): a grounded model should be
*confidently wrong* with the wrong image. Acc on C3 should be **below
chance** for a grounded model. If acc on C3 ≈ acc on C1, the model
ignores the image when priors disagree with it.

Q: "Does the model use spatial structure or just global stats?"
→ Compare **Δ_C4** (shuffled pixels): if Δ_C4 ≪ Δ_C1, model is using
color/texture priors, not structure.
