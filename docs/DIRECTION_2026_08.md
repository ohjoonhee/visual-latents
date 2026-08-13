# Direction drift — latent reasoning as a fix for the modality gap

**Date:** 2026-08-13 · **Status:** proposal, nothing run yet · **Supersedes as main thread:** Pivot A / cluster Phase 3

Written after a four-way literature sweep covering the ~2-month pause (last repo activity 2026-06-09).
All arXiv IDs below were verified against the arXiv API; titles match.

---

## 1. What changed while we were paused

The field ran the experiment we were building toward, and it came back negative — repeatedly,
from independent groups, within about six months.

| Paper | ID | Finding |
|---|---|---|
| What's Holding Back Latent Visual Reasoning? | 2605.18445 | Oracle latents uninformative; generated latents collapse |
| Leveraging Latent Visual Reasoning in Silence | 2605.18641 | Latent tokens replaceable **by noise** at inference with no loss |
| Ablate-to-Validate | 2605.21642 | Token Replacement Test: gains persist when latent content is corrupted; "accuracy is a misleading proxy" |
| Beyond Visual Memory | 2606.01287 | 78–100% of gains come from boundary markers + attention shape, not the slots |
| Cosine Misleads | 2606.05753 | Latent–target cosine *negatively* correlated with accuracy (r = −0.94) |
| Imagination Helps… But Not Yet in Latent Space | 2602.22766 | Causal mediation: input→latent AND latent→answer both disconnected |
| Visual Latents Know More Than They Say | 2605.02735 | "Silenced" latents — present, but not causally driving the answer |
| The Illusion of Superposition? | 2604.06374 | Fine-tuned latent CoT collapses to shortcuts; only from-scratch models show real superposition |

**This is our own overnight-2 dissociation, independently rediscovered eight times.** We found that
forcing Monet stage-3's cosine signature to stage-2 values at inference moved utility by 0.002.
*Cosine Misleads* found the same thing at r = −0.94 across five LVR variants.

**Consequence: the old thread is dead.** Pivot A's programme — better regularizers → better latent
geometry → assumed better function — optimizes a quantity that eight papers now agree is not the
mechanism. Do not submit cluster Phase 3 as designed. The G recipe is not wrong, it is *unfalsifiable
by the metrics we were using*.

---

## 2. The naive version of the new thesis is already falsified

The intuition — *latent reasoning earns its keep where language can't express the reasoning trace* —
is right in spirit, but the obvious form of it has been tested and lost:

- **MentisOculi** (2602.02465) is a *procedural, multi-step, visually-solvable* benchmark — exactly the
  task family the thesis calls for. Latent tokens and explicitly generated imagery **both fail to
  improve performance**. Worse, models "fail to leverage even ground-truth visualizations."
- **CapImagine** (2602.22766) teaches the model to imagine *in text* and **beats latent-space baselines**
  on vision-centric benchmarks.

So "pick harder spatial tasks and the latents will start mattering" is not a hypothesis anymore —
it is a refuted one. Any new direction has to explain *why* those experiments failed, not hope they
were underpowered.

There is still a real gap in the falsification: TRT (2605.21642) ran on BLINK, VSP, CV-Bench and a
relative-depth testbed — perception tasks, largely single-step. That leaves the evaluation-regime
argument alive but **not sufficient on its own**, because MentisOculi closed most of it.

---

## 3. Refined thesis: the field has been supervising the wrong kind of imagery

Look at what every method regresses its latents *toward*:

| Method | Latent target |
|---|---|
| LVR (2509.24251) | ROI **visual-patch embeddings** of the input image |
| Monet (2511.21395) | teacher **visual features** / stage-2 latents |
| Mirage (2506.17218) | **image embeddings** (latent visual tokens) |
| MVoT, UniVLR, Gen-VCoT | **rendered images** |

Every one is a *pictorial* target. The latent is trained to be a picture.

Cognitive psychology dissociates two systems that the word "imagery" conflates
(Kosslyn/Pylyshyn debate; Paivio's dual coding; and most sharply the modern aphantasia literature):

- **Object imagery** — pictorial, depictive, "seeing it in your mind's eye." *Impaired in aphantasia.*
- **Spatial imagery** — relational/coordinate: where things are, how they connect, how a transform acts.
  *Typically spared in aphantasia — and aphantasics perform near-normally on mental rotation.*

Humans who cannot form pictures at all still solve the block-counting and rotation problems.
The load-bearing representation for spatial reasoning is **not** the picture.

**H2: latent visual reasoning fails because it is trained toward object imagery, while the tasks
require spatial imagery.** A latent trained to be a picture is redundant with the image tokens already
in context — so of course it can be replaced by noise, of course it is bypassed, of course cosine
alignment to it anti-correlates with accuracy. The negative-result cluster is exactly what H2 predicts.

H2 also explains MentisOculi's most damning detail — models "fail to leverage even ground-truth
visualizations." A ground-truth *picture* still requires extracting the spatial state from pixels,
which is the step that is actually failing. Hand the model the *state* instead and the prediction flips.

**Differential prediction:** pictorially-supervised latents are TRT-insensitive (the status quo, already
observed); **spatially**-supervised latents — occupancy, adjacency, coordinate frames, transform
parameters — should be TRT-*sensitive*. That is a claim that can be killed in one experiment.

---

## 3b. Update, same day: what MentisOculi's released data already shows

Pulled from `results/results_table.csv` on `master` (verified; the repo's default branch is `master`, not `main`).

Rush Hour is the one task carrying **both** an image arm (`simple`) and a symbolic-text arm (`simple_text`):

| Level | Gemini-3-Pro image | Gemini-3-Pro **text** | GPT-5.1 image | GPT-5.1 **text** | humans |
|---|---|---|---|---|---|
| 3 | 0.267 | **0.900** | 0.267 | **0.931** | 0.847 |
| 4 | 0.300 | **0.800** | 0.033 | **0.714** | 0.660 |
| 5 | 0.100 | **0.633** | 0.000 | **0.625** | 0.560 |

*(n=30/cell, 95% CIs in the file.)*

**This cuts against the premise as originally stated.** Given the spatial state as *symbols*, models go from
far below human to **above** human average. Given the same state as *pixels*, they collapse. For this task
language is not lossy — symbolization is a rescue. The bottleneck is **extracting spatial state from the
image**, not expressing the reasoning.

Independent corroboration landed 2026-08-10: **Thinking With Tools, Not With Pixels** (2608.09682) finds the
structured text returned by a tool call carries more of the reasoning signal than the image does. Two
unrelated lines of evidence now point at perception→state extraction as the real gap.
(See also *The Illusion of Visual Tool-Use*, 2608.06270 — the tool-use analogue of Ablate-to-Validate.)

**What survives, and it is sharper than the original thesis.** `simple_text` appears **only** for Rush Hour —
a 6×6 grid whose state is trivially symbolizable. So do `icl_intermediate_images` (the oracle arm), `tool_use`,
and the human baseline. Form-board, hinge-folding and paper-fold have *only* `simple` and `generate_images`.
The benchmark's own coverage traces the boundary of verbalizability: the arms exist exactly where symbolic
encoding was easy to write, and are missing exactly where it is hard.

**Revised question for E0:** does symbolization rescue the *geometric* tasks too?
- If yes → the modality gap is perceptual; latent reasoning should target perception→state extraction, and
  the "language can't express it" framing should be dropped.
- If no, or if no faithful symbolic encoding can be written → that is where genuine non-verbalizability lives,
  and it is measurable rather than asserted.

**Anomaly to check first:** `mirage` on sliding-puzzle scores 0.933 / 0.833 / 0.700 / 0.767 / 0.800 across
levels 1–5 — nearly flat against difficulty, and above Gemini-3-Pro on the same task. Flat-vs-difficulty is the
classic shortcut signature and the first thing to point TRT at.

## 4. First experiments

Ordered so that the cheapest one can kill the whole programme.

### E0 — Oracle-format dissociation (inference only, no training, ~1 A6000-day)

**Target the geometric tasks — `paper-fold`, `hinge-folding`, `form-board` — not Rush Hour.** Rush Hour
already has all four arms and has already answered the question there (§3b); the geometric tasks have only
`simple` and `generate_images`, which is precisely the gap.

Give the model, in context, the *same* ground-truth intermediate state in four formats:

| Arm | What the model is handed | Role |
|---|---|---|
| (a) none | baseline | floor |
| (b) **pictorial oracle** — GT intermediate state rendered as an image | replicates MentisOculi's oracle |
| (c) **spatial oracle** — GT state as structured symbols (coords, occupancy list, adjacency) | H2's arm |
| (d) **verbal oracle** — GT state as fluent prose | CapImagine's format |

- **H2 predicts (c) ≫ (b)**, and (c) > (d) once state size passes the point where prose serialization
  degrades.
- If (b) ≈ (c) ≈ (a), the failure is at the *consumption* end for every format, the imagery programme
  is dead regardless of supervision target, and we stop and write that up.

Sweeping state complexity turns arm (c) vs (d) into a **measured verbalization-gap curve** — the thing
this project has so far only asserted. To my knowledge nobody has run it; the CoT-hurts papers
(2604.16060, 2606.03988) are outcome-only.

### E1 — "Probe vs. say" (forward passes + logistic regression; cheap)

On identical trials: linear-probe the activations for GT state variables (rotation angle, occupancy,
fold parity, cube count) layer by layer, **and** score what the model's verbal CoT actually asserts
about those same variables. The gap is knows-but-cannot-say, in bits.

This fuses two literatures that have never been fused — activation probing (2505.05410, 2603.17199)
is topic-general; the CoT-hurts work is spatial but outcome-only. It stands alone as a contribution
even if E2 fails, and it is the quantitative form of your "language can't express it" claim.

### E2 — Pictorial vs. spatial supervision (cluster, 3B)

Identical architecture, data, compute; only the latent target differs.

- **P-arm** — latents → visual-encoder embeddings of the rendered GT state *(the field's status quo)*
- **S-arm** — latents → structured spatial state via a small readout head

Mandatory controls, all three:
1. **TRT** (2605.21642): zero / random / first-repeat / oracle replacement. Non-negotiable — it is
   becoming the field's standard diagnostic and any result without it will be dismissed.
2. **Weights-matched latent-disabled arm** — your own 2026-08-04 bar: same trained weights, latent
   channel off. Distinguishes "the latent works" from "training with a latent shaped better weights."
3. **Compute-matched no-latent arm** — same FLOPs spent on more discrete tokens. The sweep confirms
   *no VLM latent-reasoning paper has run this*; it is the control that separates continuous-ness
   from extra compute.

**Success = the S-arm is TRT-sensitive and beats both control arms.** Note that a clean negative here
is also publishable, given how much of the field is currently betting on this.

---

## 5. What to reuse rather than build

Verified 2026-08-13 by fetching each repo. Only those emitting **programmatic ground-truth
intermediate state** are useful here — E0 arm (c) requires it.

| Tool | URL | License | GT intermediate state |
|---|---|---|---|
| **MentisOculi** | `github.com/Jana-Z/mentis-oculi` | Apache-2.0 | **Yes** — 5 procedural generators (Form Board, Hinge Folding, Paper Fold, Rush Hour, Sliding Puzzle) with per-difficulty procedural state |
| **CLEVR gen** | `facebookresearch/clevr-dataset-gen` | BSD-style | **Yes** — full scene JSON: attributes, 3D positions, relations, question programs (archived 2023, still runs) |
| **photorealistic-blocksworld** | IBM, CLEVR fork | Apache-2.0 | **Yes** — stacked-block scenes + scene JSON |
| **MentalBlackboard** | `github.com/nlylmz/MentalBlackboard` | unspecified | **Yes** — VPython fold simulation emits fold-state JSON |
| RAVEN / I-RAVEN | `WellyZhang/RAVEN` | **GPL-3.0** | Yes — And-Or-Tree rule structure + per-panel XML. Copyleft: check before vendoring |
| Spatial457 | `xingruiwang/spatial457` | Apache-2.0 | Yes — scene JSON + program field; Blender + ~2GB assets |
| RE-ARC / ARC-GEN | — | MIT / Apache-2.0 | **No** — grid pairs only. Not useful for arm (c) |
| IR3D-Bench | — | unspecified | **No** — inverse-rendering eval harness, not a forward generator |

**Run E0 on MentisOculi's own generators.** It engages the paper that most threatens the thesis, on
its own terms and its own tasks — the strongest venue for the result, positive or negative.

**The one gap:** no off-the-shelf generator emits a 3D block structure *together with true orthographic
front/side/top views* — your original IQ-test example. Cheapest route is a 3-camera orthographic rig
on top of photorealistic-blocksworld's existing Blender + scene-JSON pipeline. Defer until E0 justifies it.

From our own repo: `phase0_monet_probe/ablation.py` metric conventions, the `pivot_a/` trainer's
`reg_kind` / `loss_form` selectors, and the LVR 3B SFT checkpoint (`checkpoints/lvr-3b-sft-step2500`)
as a ready pictorially-supervised P-arm baseline — it is already trained toward ROI patch embeddings.

From our own repo: `phase0_monet_probe/ablation.py` metric conventions, the `pivot_a/` trainer's
`reg_kind` / `loss_form` selectors, and the LVR 3B SFT checkpoint (`checkpoints/lvr-3b-sft-step2500`)
as a ready pictorially-supervised P-arm baseline — it is already trained toward ROI patch embeddings.

## 5b. Baselines — what can actually be relied on

**The reliable baseline is not another paper's number.** *Ablate-to-Validate* showed accuracy survives
corrupting the latent content, so a published accuracy carries no evidence that the mechanism works. The
load-bearing baseline is **your own weights with the channel ablated**.

**Reviewer-expected control arms**, converged across 2605.21642 / 2606.01287 / 2606.05753 / 2602.02465:

1. Content-replacement sweep on the latent channel — zero / random / first-repeat / oracle (TRT).
2. Component-isolated ablation — latent slots vs. boundary markers vs. format alone (2606.01287 found
   markers alone preserve 78–100% of the gain).
3. **Probe-based decodability** of the latent — *not* cosine/alignment to a visual target (PRISM, 2606.05753).
4. Oracle/ground-truth visual condition as an upper bound, separate from self-generated imagery.
5. Matched token-budget / context-length control, to rule out "more compute."

**Note: TRT code is not released.** The paper says "code will be released" at
`tjazhang.github.io/ablate_to_validate`; no live repo found as of 2026-08-13. TRT is simple enough to
reimplement — four replacement modes with prompt/image/budget/decoding held fixed — but budget for it and
do not plan around an upstream drop.

**External numbers we can cite as-is** (MentisOculi, n=30/cell with 95% CIs):
`simple` and `generate_images` cover all five tasks; `simple_text`, `icl`, `icl_intermediate_images`,
`evolved`, `tool_use`, `video_generation` and the `humans` baseline are **Rush Hour only**. Models present
include gemini-3-pro-preview, gpt-5.1, qwen3-vl-235b-a22b-thinking, emu3.5, mirage, veo-3.1, wan-2.6.

**Internal anchors that reproduced** (use these, not the Pivot A geometry numbers):
released LVR-7B (V\* 81.68@s8, MMVP 72.0 — matched paper to ≤0.6pt), Monet stage-2 (cos 0.377, util +2.05),
and `checkpoints/lvr-3b-sft-step2500` (V\* 65.97/65.45/65.45, MMVP 55.67/56.33/57.33) as a ready
pictorially-supervised P-arm.

## 6. Open risks

- **MentisOculi may have already run something close to E0(b).** The paper must be read in full before
  E0 is finalized; if their oracle arm is our arm (b), we inherit it and only need (c) and (d).
- **CapImagine is a strong baseline**, not a strawman — arm (d) must be its actual method, not a weak
  prose description.
- H2 leans on a human dissociation (aphantasia) holding in models. That is an analogy, and E0 is
  precisely the test of whether it transfers.
- Not yet verified: whether a Blender generator emitting a block structure *together with its
  orthographic views* exists off the shelf. If not, that is the one thing we would have to build.
