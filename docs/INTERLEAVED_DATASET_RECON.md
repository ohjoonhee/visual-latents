# Interleaved Visual Reasoning — Dataset Recon

**Author:** research recon, 2026-05-03
**Scope:** Identify training datasets purpose-built for *interleaved* visual
chain-of-thought reasoning, to replace the GQA + CLEVR + TallyQA mix from
`docs/inherited/TRAINING_DATA_PLAN.md` after round-3's binding-test failure
(`INTERLEAVED_POC_RESULTS.md` §12 — latents encode the question-conditioned
answer marginal, not image content, when supervised by reader-NLL alone).

**Verification policy.** Every numeric claim below is tagged with its source
(arXiv abstract/HTML, HuggingFace API metadata, or HF dataset card).
Where the same number appears in multiple sources I take the most recent /
authoritative one. Unverified claims are flagged `[unverified]`.

---

## Executive summary — recommended pivot

> **Switch to Visual-CoT (`deepcs233/Visual-CoT`, 373 K examples, ~143 GB
> with images, Apache-2.0) for the local POC.** Each example is a (real
> image, question, answer, reasoning-chain, **per-step bounding box**)
> tuple — the bbox is exactly the missing supervision signal that round-3
> §12 says the design needs. Start with the ~88 K **GQA subset of
> Visual-CoT** (~12 GB after picking three GQA-source image archives,
> tractable on the local A6000); upgrade to the full mix later. **Use the
> bbox to crop a "focus image" per latent block, encode it through the
> generator's own vision encoder, and add a cosine-distance auxiliary loss
> between `h[block_k]` and that focus image's `V_sem` (Mirage-Stage-1
> recipe).** Fallback: **Zebra-CoT** (`multimodal-reasoning-lab/Zebra-CoT`,
> 182 K, 77 GB, CC-BY-NC-4.0) — has explicit *intermediate reasoning
> images* (not bboxes) generated programmatically per task, but the NC
> license makes it research-only. GQA-balanced + CLEVR + TallyQA from the
> existing plan stay useful as **multi-Q controls** (single-step but
> q-invariant), not as the primary signal.

---

## 1. The dataset gap from POC round-3

The current parallel method (`docs/METHODS.md`) and the round-3 interleaved
extension (`src/vl/interleaved/`) both train a generator to produce K
continuous latents `h ∈ ℝ^{K×D}` that a frozen reader splices into image-
pad positions. The reader's NLL on the gold answer is the *only* image-
grounded signal in the loss; everything else (norm regulariser, concept
loss against `V_sem`) is auxiliary or geometric. Round-3 §11–12 ran the
strictest available binding test (R=2 frozen readers, multi-Q, no norm
regulariser, real-image-class shapes) and found that the natural pairing
trains identically to the permuted control — both hit ~3.2-4.6 nat
held-out NLL. The latents are encoding the *question-conditioned answer
marginal* (P(a | q, image-shape-distribution)), not anything about *this*
image. GQA and CLEVR cannot fix this on their own: each (q, a) is a
single-step lookup, so the model never has to "look back" at the image to
serve a chain of dependent questions.

What the design actually needs is **per-step image grounding annotations**
that say *"to answer this question, attend to region X of this image at
step 1, then region Y at step 2"*. Then the latents at block 1 can be
trained directly to encode region X (not the answer marginal), via a
cosine-to-vision-encoder-features loss à la Mirage Stage 1
(`INTERLEAVED_LITRECON.md` §3, §5.4). This is a **data pivot**, not just a
loss-function pivot — neither GQA nor CLEVR exposes per-reasoning-step
region annotations even though GQA scene graphs exist. We need datasets
purpose-built with intermediate visual grounding.

---

## 2. Datasets explicitly designed for visual CoT

### 2.1 Visual-CoT (`deepcs233/Visual-CoT`, arXiv:2403.16999)

**The single best fit for this project's needs.**

- **Scale.** 373 K training examples (the `viscot_363k.json` file shipped
  with the dataset is 345 MB; abstract claim of "438 K Q-A pairs"
  refers to the eval+detailed-reasoning superset). 98 K of the 373 K
  also have *detailed step-by-step reasoning text*. (HF tree listing,
  arXiv HTML §dataset.)
- **Annotation per example.** Each example contains: `question`,
  `answer`, `image` filename, `width/height`, **`bboxs: list[[x1,y1,x2,y2]]`
  (the per-question region needed to answer)**, `dataset` (source name),
  and for the 98 K subset, `reasoning: list[{operation, dependencies,
  argument}]` (a structured chain whose ops are GQA-style scene-graph
  operations like `select`/`relate`/`query`/`filter`). The bbox(es) tell
  the model *where to look* per question — the missing grounding signal
  from §1. (HF dataset preview schema, verified.)
- **Image sources.** Built from 10 source datasets across 5 domains:
  GQA (88 K), Flickr30k (136 K), Open Images (43 K), DocVQA (33 K),
  TextCaps (32 K), TextVQA (16 K), InfographicsVQA (15 K), Birds-200-2011
  (4 K), VSR (3 K), plus SROIE/DUDE for zero-shot eval. Bboxes were
  derived from existing dataset annotations (PaddleOCR for text-based,
  scene-graph object bboxes for GQA, etc.) — *not* re-annotated from
  scratch. (arXiv:2403.16999v3 §3.)
- **License.** Apache-2.0 (HF cardData).
- **Disk footprint.** Annotations: ~3.1 GB total (`viscot_363k.json`
  345 MB + `viscot_mixed_2m.json` 2.76 GB). **Images: 139.25 GB across
  13 × ~10.7 GB tar shards** — must merge then extract. (HF tree probe.)
- **Suitability for this project.** Real images; intermediate
  bbox-per-step grounding; structured reasoning ops; large enough for
  proper training (373 K) and small enough per-sample (one image, one
  trace) that batching is straightforward. The GQA-source subset (88 K)
  alone matches the project's existing GQA pipeline and would only need
  a small loader extension.

### 2.2 Zebra-CoT (`multimodal-reasoning-lab/Zebra-CoT`, arXiv:2507.16746)

**The closest published dataset to "interleaved trace with intermediate
images" — the Mirage/Latent-Sketchpad training-data analogue.**

- **Scale.** 182,384 examples across 18 task subsets in 4 categories
  (2D visual, 3D visual, scientific, logic & strategic games). Single
  `train` split per subset. (HF cardData, dataset_info.)
- **Annotation per example.** `Question`, `Text Reasoning Trace`,
  `Final Answer`, `problem_image_1`, **`reasoning_image_1..4`** (up to 4
  programmatically-generated intermediate images that show what the
  reasoner should "see" at each step). Reasoning images are *not* crops
  of the problem image — they are *new* synthetic images depicting the
  intermediate state (e.g., the puzzle with one piece placed, the chess
  board after a candidate move). (HF cardData features schema.)
- **License.** **CC-BY-NC-4.0** — research-only, no commercial use.
- **Disk footprint.** 76.97 GB across 18 configs, computed from the HF
  `dataset_info[*].splits[*].num_bytes`. Largest single config:
  Multi-Hop Objects Counting at 19.5 GB; smallest: Competitive
  Programming at 30 MB. Useful subsets that fit on local disk:
  - Visual Search (30 K, 13.2 GB) — tractable.
  - Maze (20 K, 5.4 GB) — tractable.
  - Chess (20.5 K, 3.9 GB) — tractable.
  - Embodied CoT (22.7 K, 4.0 GB) — tractable.
- **Suitability for this project.** Has the *strongest* interleaved-
  reasoning structure of any dataset reviewed (literally text + image at
  each step, matching the mechanism), but the reasoning images are
  programmatic per-task. For the Mirage-Stage-1 cosine-to-V_sem recipe
  this is *better* than bboxes, because the reasoning image directly
  defines the target features. The NC license is the blocker if any
  paper from this work goes commercial.

### 2.3 LLaVA-CoT-100k (`Xkev/LLaVA-CoT-100k`, arXiv:2411.10440 — also
"LLaVA-o1"). **The Heima training set.**

- **Scale.** 98,582 examples. (HF probe of cardData.)
- **Annotation per example.** Single (image, question) per row, with a
  GPT-4o-generated 4-stage reasoning response: `<SUMMARY>`, `<CAPTION>`,
  `<REASONING>`, `<CONCLUSION>`. **No bounding boxes, no per-stage
  regions** — the structure is text-only stages over a single fixed
  image. (HF preview schema verified.)
- **Image sources.** ShareGPT4V (31.3 K), ChartQA (17.2 K), A-OKVQA
  (16.1 K), AI2D (11.4 K), GeoQA+ (11.4 K), ScienceQA (5.6 K), DocVQA
  (4 K), PISC (1 K), CLEVR (0.5 K), CLEVR-Math (0.5 K). (HF dataset
  card § Description.)
- **License.** Apache-2.0.
- **Disk footprint.** **171 GB** (HF dataset card / preview probe).
  The size dwarfs the example count because it bundles full-resolution
  source images.
- **Suitability for this project.** Less direct than Visual-CoT for our
  grounding-supervision need (no bboxes), but the 4-stage decomposition
  matches the project's `T=2` text-segments-with-latent-blocks design
  better than any other dataset — each stage *could* be supervised by a
  separate latent block. Used in production by Heima
  (`INTERLEAVED_LITRECON.md` §2.5), which is the closest *interleaved-
  latent* paper that reports gains.

### 2.4 CogCoM (`THUDM/CogCoM` — *no HF dataset published, GitHub-only*,
arXiv:2402.04236)

- **Scale.** 77 K total: 70 K auto-generated CoM samples + 7 K manually
  annotated geometry. (arXiv HTML §3.)
- **Annotation per example.** Six "atomic manipulations" per chain:
  `OCR`, `Grounding` (bbox), `CropZoomIn` (region), `Counting`,
  `Calculate`, `Line` (auxiliary lines for geometry). Chains are
  multi-step — **explicit visual operations**, not just text rationales.
  Hosted at `github.com/THUDM/CogCoM` (data + code).
- **Image sources.** TextVQA (10.8 K), ST-VQA (4.8 K), TDIUC (65 K),
  plus MathVista + ChartQA for the geometry annotations.
- **License.** Likely permissive [unverified — paper says open, no HF
  card to confirm].
- **Disk footprint.** Not reported on the GitHub release.
  [Estimated ~30-50 GB based on TextVQA+ST-VQA+TDIUC image sizes,
  unverified.]
- **Suitability for this project.** The `Grounding` + `CropZoomIn`
  annotations are exactly what a per-block grounding loss would
  consume. Smaller scale than Visual-CoT and harder to set up (no HF
  loader); use as the *qualitative* manipulation-style benchmark, not
  the primary training corpus.

### 2.5 M3CoT (`LightChen2333/M3CoT`, arXiv:2405.16473)

- **Scale.** 11,459 examples (7.9 K train / 1.1 K val / 2.4 K test).
- **Annotation per example.** Text-only multi-step rationales (avg
  10.9 steps, 294 chars). Three domains (science, math, commonsense)
  spanning 17 topics, 263 categories. **No bbox / region annotations.**
- **License.** MIT (HF cardData).
- **Disk footprint.** Small [unverified — `dataset_info` not populated
  on HF cardData; estimated <1 GB based on text-only nature with image
  references].
- **Suitability for this project.** Too small to train on standalone
  (11 K total). Useful as an OOD eval for multi-step text reasoning
  *after* training on Visual-CoT or Zebra-CoT.

### 2.6 Mulberry-SFT (`HuanjinYao/Mulberry-SFT`, arXiv:2412.18319)

- **Scale.** 260 K reasoning paths (paper title), HF preview shows
  ~413 K rows — likely includes intermediate search states. (HF probe.)
- **Annotation per example.** ShareGPT-format messages with text-only
  reasoning paths (avg 7.5 reasoning steps from CoMCTS tree search).
  **No per-step grounding / no bboxes.**
- **License.** Apache-2.0.
- **Disk footprint.** **22.6 GB** (HF dataset card).
- **Suitability for this project.** Strong CoT density (avg 7.5 steps)
  and broad reasoning coverage (geometry, math, general VQA). But
  text-only reasoning means it can supervise the *text* segments of an
  interleaved trace — it cannot directly supervise the latent blocks
  with image features.

### 2.7 MathVista (`AI4Math/MathVista`, arXiv:2310.02255)

- **Scale.** 6,141 examples, single config. (HF probe.)
- **Annotation per example.** Math+visual MCQ; rationales available for
  some subsets [unverified for which specifically]. Eval-oriented.
- **License.** CC-BY-SA-4.0.
- **Disk footprint.** 0.79 GB. (HF probe.)
- **Suitability for this project.** Eval-only — too small and
  benchmark-shaped to be primary training data.

---

## 3. What prior art trained on (verified per-paper)

| Paper | arXiv | Training corpus | Annotation type | Scale |
|---|---|---|---|---|
| **Mirage** | 2506.17218 | Bespoke synthetic per-benchmark traces (VSP-Reason 3 K, VSP-Plan 3 K, BLINK-Jigsaw 1 K, SAT 1 K, COMT 820); helper images from Gym renderer / CogVideoX / programmatic | Text + helper images at each step | ~9 K total (per HTML §experimental setup) |
| **Latent Sketchpad** | 2510.24514 | **MazePlanning** (47.8 K mazes, interleaved text-image traces). Sketch decoder pretrained separately on Quick Draw (50 M sketches). | Image+text per step | 47.8 K (verified HTML) |
| **Heima** | 2501.19201 | **LLaVA-CoT-100k** (`Xkev/LLaVA-CoT-100k`) | 4-stage textual CoT, no bbox | 98 K (verified) |
| **MCOUT** | 2508.12587 | VQAv2 train (443 K) for pretraining → ScienceQA-image-subset (6.2 K) + MMMU (~150) for fine-tuning | Single-Q VQA + textual rationales (no per-step grounding) | ~450 K total |
| **ILVR** | 2512.05665 | COMT (3.4 K), VSP (1 K) for IID; **Zebra-CoT 10 K subset + Visual-CoT 80 K subset** for OOD fine-tuning | Interleaved text+image (Zebra) or bbox (Visual-CoT) | 100 K combined |
| **Mull-Tokens** | 2512.10941 | Not disclosed in abstract; full PDF needed | Final-answer-only RL | [unverified scale] |
| **SkiLa** | 2512.16584 | Not disclosed in abstract; sketch encoder targets implied | Reconstruction targets vs sketch encoder | [unverified] |

**Key cross-reference.** Both ILVR and Latent Sketchpad — the two most
recent published interleaved-VLM-latent works that report substantive
gains — train on **either Visual-CoT or Zebra-CoT** (or a bespoke
per-task synthetic). This is independent corroboration that those are
the right datasets for this design space.

---

## 4. Multi-question-per-image datasets

Q-invariance pressure requires K_q ≥ 3 *different* questions about the
same image with image-content-dependent answers. Already covered in
`docs/inherited/TRAINING_DATA_PLAN.md` §2.1 — the relevant rows:

| Dataset | HF path | avg Q/img | Annotation richness | Disk |
|---|---|---|---|---|
| GQA-balanced | `lmms-lab/GQA` cfg `train_balanced_instructions` | ~13 (max 37) | scene graph available externally | 0.46 GB Qs + 10.2 GB images = 10.7 GB |
| GQA-all | `lmms-lab/GQA` cfg `train_all_instructions` | ~125 | scene graph external | 6.9 GB Qs + 10.5 GB images = 17.4 GB |
| Visual Genome QA | `ranjaykrishna/visual_genome` cfg `question_answers_v1.2.0` | ~16 | dense scene graph in same dataset | loader broken (per inherited §5.1) |
| Visual7W | `HuggingFaceM4/the_cauldron` cfg `visual7w` | 5.6 | MCQ, some pointing | <2 GB |
| VQAv2 | `lmms-lab/VQAv2` (val only) or cauldron `vqav2` | 4.4-6 | severe lang priors | ~5 GB |
| CLEVR | `HuggingFaceM4/the_cauldron` cfg `clevr` | 10 (exact) | synthetic, GT scene graph | ~13 GB |

**Re-evaluation in light of the binding-test failure.** The user said GQA
"doesn't seem suitable" — the binding-test failure justifies that for
**single-step single-Q** training, but GQA's multi-Q structure (13 Qs/img)
is still the right substrate for the **q-invariance loss** that round-3
§11.4 says is necessary but not sufficient. The fix isn't to drop GQA;
it's to **add per-step grounding**, which Visual-CoT supplies *on the
exact same GQA images* (88 K of Visual-CoT comes from GQA).

---

## 5. Datasets with intermediate-grounding annotations

| Dataset | Annotation type | Scale | Disk | License |
|---|---|---|---|---|
| **Visual-CoT** | per-Q bbox (`bboxs`) + structured op chain (98 K subset) | 373 K | ~143 GB w/ images, 3 GB annotations only | Apache-2.0 |
| **Zebra-CoT** | full intermediate reasoning images per step (up to 4) | 182 K | 77 GB | CC-BY-NC-4.0 |
| **CogCoM** | atomic manipulations (Grounding bbox, CropZoomIn region, Line, etc.) | 77 K | est. 30-50 GB [unverified] | open (GitHub-only) |
| GQA scene graphs | per-image scene graph (objects + relations + bboxes); not in `lmms-lab/GQA` parquet — original GQA download | 113 K images | scene graphs ~2 GB JSON; images 20 GB | MIT (data) |
| Voxel51/GQA-Scene-Graph | GQA scene graphs in HF format | 10K-100K | [unverified — `dataset_info` not populated] | unspecified [unverified] |
| Visual Genome scene graphs | per-image scene graph + region descriptions + QA | 108 K images | ~30 GB | CC-BY-4.0 (loader broken on HF) |
| RefCOCO/+/g | referring expression → bbox | 17-50 K | ~5 GB total | research |

**Critical takeaway.** Only Visual-CoT, Zebra-CoT, and CogCoM ship with
**per-reasoning-step** grounding annotations rather than just per-image
scene graphs. Of those, **only Visual-CoT and Zebra-CoT are HF-loadable
out of the box.**

---

## 6. Local cache status + size estimates

| Dataset | Cached locally? | Download size | Tractability |
|---|---|---|---|
| `hiyouga/geometry3k` | YES (57 MB) | 57 MB | already local |
| `trl-internal-testing/...` | YES (6 MB) | 6 MB | irrelevant |
| **Visual-CoT (annotations only)** | no | 3.1 GB | **trivial** — start here |
| **Visual-CoT (GQA images only)** | no | ~12 GB (subset of 13 tars) | **tractable on A6000** |
| **Visual-CoT (full)** | no | 143 GB | tractable but ~24 h on residential link |
| **Zebra-CoT (Maze + Chess subsets)** | no | 9.3 GB | **tractable** |
| **Zebra-CoT (full)** | no | 77 GB | tractable; ~12 h |
| LLaVA-CoT-100k | no | 171 GB | cluster-only |
| Mulberry-SFT | no | 22.6 GB | tractable |
| M3CoT | no | <1 GB est | trivial (eval-only) |
| MathVista | no | 0.79 GB | trivial (eval-only) |
| GQA-balanced + images | no | 10.7 GB | **tractable** (per existing plan) |
| Cauldron-CLEVR | no | ~13 GB | tractable |
| Cauldron-TallyQA | no | ~7 GB | tractable |
| The Cauldron (full) | no | 456 GB | cluster-only |

**Disk available on local A6000 box:** assumed >= 200 GB free
(`docs/inherited/TRAINING_DATA_PLAN.md` §5.4 budgets 250 GB for the full
mix). The recommended pivot's full footprint (~143 GB Visual-CoT) is at
the edge. The *practical* first cell is the GQA-subset only at ~15 GB.

---

## 7. Recommended pivot

**Primary: Visual-CoT, GQA-subset first, full mix later.**

Concrete plan:

1. **Download annotations** (`viscot_363k.json` 345 MB only, no images
   yet). Filter to `dataset == "gqa"` rows → ~88 K examples. Each row
   has `image` filename (matches GQA image IDs the project already has
   from the existing GQA pipeline) and `bboxs` (the per-question region).
2. **Reuse the project's existing GQA images** if cached, else download
   only the GQA-source tar shards from Visual-CoT
   (`cot_images_tar_split/cot_images_00..04` — first 5 of 13, ~50 GB; the
   GQA images are concentrated in the early shards per the upstream
   ordering [unverified — confirm via `tar -tf` head before full pull]).
3. **Implement a Visual-CoT loader** in `src/vl/interleaved/trainer.py`
   alongside the existing shapes/synthetic loaders. Schema:
   ```python
   {"image": Image, "question": str, "answer": str,
    "bboxs": [[x1,y1,x2,y2], ...],
    "reasoning": [{"operation":..., "argument":...}, ...]}  # 98K subset only
   ```
   Use the same `dataset_kind == "viscot_gqa"` dispatch the project's
   `_make_shapes_image_and_gt` shows.
4. **Add the Mirage-Stage-1 grounding loss.** For each example with K
   reasoning steps and K bboxes, crop the image to bbox_k → encode
   through generator's vision encoder → take the merger-output features
   → cosine-to `h[block_k]`. This is the *focused* version of the
   existing `concept_loss` (which currently uses a single `V_sem` for
   the full image). The implementation reuses `src/vl/losses.py`'s
   cosine machinery with a per-block target tensor.
5. **Curriculum.** Start `w_grounding=1.0, w_nll=0` for ~500 steps
   (Mirage Stage 1), then anneal `w_grounding: 1.0 → 0.3,
   w_nll: 0 → 1.0` over the next 500 (Mirage Stage 2). This is the
   curriculum that round-3 §11.4 explicitly recommended; Visual-CoT is
   the dataset that lets it actually run.
6. **Falsifiable result.** Re-run the binding test from §12 of
   POC_RESULTS — natural vs perm held-out NLL on a Visual-CoT-GQA
   held-out split, with R=2 frozen readers. Pass criterion: natural
   beats perm by ≥0.5 nat at step 500 (Stage-1 only, no reader-NLL
   yet) and ≥1.0 nat at step 1000 (post Stage-2). If Stage-1 alone
   produces gap, the design's grounding signal works. If not, the
   issue is the architecture, not the data.

**Why this is the right first dataset:**
- Real images (rules out the synthetic-shapes ambiguity).
- Per-step bbox annotations (the missing supervision).
- Same GQA images the existing pipeline knows → minimal infra lift.
- Apache-2.0 (paper-friendly).
- 88 K subset is large enough for proper training (8× larger than
  Mirage's full training set per §3) and small enough for local A6000
  (~15 GB total with images).

**Fallback: Zebra-CoT (Visual Search + Maze + Chess subsets first,
~22 GB combined).** Zebra-CoT's reasoning images are *more directly*
the supervision target than Visual-CoT's bboxes (cosine-to-encoder-
features works on a real image, not a region crop), but the
**CC-BY-NC-4.0 license restricts paper publication options**. Use
this if Visual-CoT's GQA subset proves insufficient — for example, if
the cosine-to-cropped-region target is too noisy because GQA bboxes
are sometimes whole-object rather than reasoning-relevant. The
implementation lift in `trainer.py` is the same as for Visual-CoT
(swap loader, swap grounding-target tensor source).

**What to keep from the existing plan.** GQA-balanced (`lmms-lab/GQA`
cfg `train_balanced_instructions`) and CLEVR remain valuable as
**multi-Q controls** for the q-invariance loss. The right mix is
~70 % Visual-CoT (grounding signal) + ~20 % GQA-balanced
(multi-Q variety, same images so no domain shift) + ~10 % CLEVR
(grounding-immune control). Total ~10 % of the round-3 plan's full
mix size — fits comfortably on A6000.

**What to drop.** TallyQA, Visual7W, AI2D, CLEVR-Math from the
inherited plan are *not* needed for the round-4 pivot — they don't
have per-step grounding. Reintroduce only if cluster scale-up
demands them for diversity.

---

## 8. Risks and unknowns flagged

1. **Visual-CoT bbox quality on the GQA subset.** Bboxes were derived
   from "GQA's official dataset object bboxes" (arXiv:2403.16999 §3) —
   they are object-level, not necessarily reasoning-step-level. For
   "what color is the cat" the bbox is the cat; for "what is to the
   left of the cat" the bbox should be the cat *and* the relation
   target — unknown if the dataset captures this. Run a 100-example
   sanity probe before training: visualize 100 (image, q, bbox) triples
   and check the bbox actually delimits the answer-relevant region.
   *[Estimated 30 min of work.]*
2. **Visual-CoT image-archive merge.** The 13 × 10.7 GB tar shards must
   be cat'd then untarred; a 50 GB intermediate file may be needed.
   Confirm `cat cot_images_*.tar | tar -xf -` works as a streaming
   pipeline before downloading all shards.
3. **Zebra-CoT NC license.** If this project's output ever needs
   commercial use, Zebra-CoT cannot be in the training set. Visual-CoT
   (Apache-2.0) is safe.
4. **GQA scene graphs as a cheaper alternative.** The `Voxel51/GQA-
   Scene-Graph` HF dataset *might* expose object bboxes per image
   without needing Visual-CoT's annotation. It would give us per-image
   grounding (not per-question), which is weaker but free. Worth a
   1-hour spike before downloading 50 GB of Visual-CoT images.
   `[unverified that the HF dataset_info populates correctly]`
5. **Reasoning-chain format mismatch.** Visual-CoT's 98 K-subset
   `reasoning` field uses GQA's structured ops (`select`, `relate`,
   etc.), not natural language. The interleaved trace mechanism
   currently expects text. Mapping ops → text is straightforward
   (op-to-template), but the project must decide whether to use the
   structured chain (richer signal, harder to integrate) or the
   text-only `full_answer` field (simpler, weaker).

---

## Sources

- [Visual-CoT (deepcs233/Visual-CoT)](https://huggingface.co/datasets/deepcs233/Visual-CoT)
- [Visual-CoT paper (arXiv:2403.16999)](https://arxiv.org/abs/2403.16999) and [HTML v3](https://arxiv.org/html/2403.16999v3)
- [Visual-CoT GitHub](https://github.com/deepcs233/Visual-CoT)
- [Zebra-CoT (multimodal-reasoning-lab/Zebra-CoT)](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)
- [Zebra-CoT paper (arXiv:2507.16746)](https://arxiv.org/abs/2507.16746)
- [LLaVA-CoT-100k (Xkev/LLaVA-CoT-100k)](https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k) and [paper arXiv:2411.10440](https://arxiv.org/abs/2411.10440)
- [CogCoM paper (arXiv:2402.04236)](https://arxiv.org/abs/2402.04236) and [GitHub](https://github.com/THUDM/CogCoM)
- [M3CoT (LightChen2333/M3CoT)](https://huggingface.co/datasets/LightChen2333/M3CoT) and [paper arXiv:2405.16473](https://arxiv.org/abs/2405.16473)
- [Mulberry-SFT (HuanjinYao/Mulberry-SFT)](https://huggingface.co/datasets/HuanjinYao/Mulberry-SFT) and [paper arXiv:2412.18319](https://arxiv.org/abs/2412.18319)
- [MathVista (AI4Math/MathVista)](https://huggingface.co/datasets/AI4Math/MathVista)
- [Mirage paper (arXiv:2506.17218)](https://arxiv.org/abs/2506.17218) and [HTML](https://arxiv.org/html/2506.17218)
- [Latent Sketchpad paper (arXiv:2510.24514)](https://arxiv.org/abs/2510.24514) and [HTML](https://arxiv.org/html/2510.24514)
- [Heima paper (arXiv:2501.19201)](https://arxiv.org/abs/2501.19201) and [HTML](https://arxiv.org/html/2501.19201)
- [MCOUT paper (arXiv:2508.12587)](https://arxiv.org/abs/2508.12587) and [HTML](https://arxiv.org/html/2508.12587)
- [ILVR paper (arXiv:2512.05665)](https://arxiv.org/abs/2512.05665) and [HTML](https://arxiv.org/html/2512.05665)
- [GQA dataset (lmms-lab/GQA)](https://huggingface.co/datasets/lmms-lab/GQA)
- HuggingFace API metadata probed via `https://huggingface.co/api/datasets/{path}` on 2026-05-03.
