# Interleaved Latent Reasoning — POC Handoff Report

**Date:** 2026-05-03
**Run scope:** rounds 1–3, autonomous, local A6000 only.
**Status (final round-3):** Mechanism trains end-to-end. The design
DOES NOT pass the binding test under any of four configurations
attempted: R={1,2} × single-Q × {norm-on, no-norm}, plus the binding
test (R=2 + multi-Q + no-norm). In every case where the norm
regulariser is off (the clean training signal), the permuted control
matches or beats the natural pairing on held-out NLL. The latents
are encoding the question-conditioned answer marginal, not image
content. **Reader-NLL-only training without an explicit grounding
signal is insufficient for this design at this scale** — Mirage-style
Stage-1 distillation (cosine-to-V_sem on h) is the recommended pivot.
Full data and pivot rationale: `INTERLEAVED_POC_RESULTS.md` §12.

This report is the user-facing entry point. The supporting documents
(litrecon → design → results) are linked below.

---

## 0. What the user asked for

> "Leave the current method as reference and perform some extensive
> research and POC experiments to concretize such interleaved latent
> reasoning with another reader (anchor) utilized training signals."

The current parallel method (`src/vl/model.py`,
`docs/METHODS.md`) was left untouched. A new module was built at
`src/vl/interleaved/` implementing autoregressive Coconut-style
text-latent interleaving with a frozen reader-anchor for supervision.

---

## 1. Document map

| File | What it contains |
|---|---|
| `docs/INTERLEAVED_LITRECON.md` | Phase 1 — literature recon (Coconut, Mirage, SkiLa, ILVR, Latent Sketchpad, Heima, MCOUT, Mull-Tokens, LIVR). Verified arxiv IDs, design-space implications. |
| `docs/INTERLEAVED_LATENT_DESIGN.md` | Phase 2 — paper-style design contract. 12 sections: trace structure, Coconut recurrence, reader options, gradient strategy, loss, forward-pass schedule, POC config, pass criteria, risks. |
| `src/vl/interleaved/{model,traces,reader,trainer}.py` | Phase 3 — implementation (~750 LOC). Reuses LoRA-injection / splice / concept-MLP / norm-loss machinery from the parallel method. |
| `configs/interleaved_mini.yaml` + `interleaved_mini_100.yaml` + perm + diag variants | POC training configs. |
| `docs/INTERLEAVED_POC_RESULTS.md` | Phase 4 — experimental results (50-step round 1 + 100-step round 2 + norm diagnostics). Read this for the actual data. |
| **`docs/INTERLEAVED_POC_REPORT.md`** (this file) | Phase 5 — synthesis and handoff. |

---

## 2. The mechanism in one paragraph

A trace is a fixed-template alternation `text₁ → latent_block₁ →
text₂ → latent_block₂ → answer`, with `T=2` text segments and
`k_latent=4` continuous-thought positions per block (`K_total=8`).
Inside a latent block, the **last hidden state at position i is fed
as the input embedding for position i+1** (Coconut, arXiv:2412.06769
§3) — text and `<|latent_start|>`/`<|latent_end|>` markers use
ordinary embedding lookup. After the full trace forward, the eight
latent hidden states are stacked into `h ∈ ℝ^{B×8×D}` and **passed
to a separate frozen reader-anchor** that splices `h` into its
image-pad positions (no real image) and is asked to score the
gold answer. The reader-NLL (plus the project's existing concept
and norm losses) trains LoRA + the new-token rows + the concept MLP.
**Cost:** 10 forward passes per training example
(1 prefix-encode + 4 latents + 1 mid-segment + 4 latents).

The novelty axis vs all surveyed prior art is the use of a
**separate frozen reader VLM** (instead of single-model
self-supervision as in Mirage, SkiLa, Latent Sketchpad, Heima,
MCOUT, Mull-Tokens, ILVR — see litrecon §3).

---

## 3. What we learned

### 3.1 The mechanism trains end-to-end

Gradient probe (`/tmp/interleaved_grad_probe.py`):

```
new_emb.grad: latent=0  latent_start=7.07e+03  latent_end=8.25e-01
LoRA lora_B nonzero grad: 252/252
LoRA lora_B in FIRST block: 7/7 with nonzero grad
concept_mlp params with nonzero grad: 4/4
Embedding table grad: None (frozen, OK)
PROBE PASSED
```

The zero-grad on `new_emb[0]` (the `<|latent|>` row) is **predicted**
by the design (§11.4): under Coconut recurrence the latent rows are
never used as input embeddings inside a block. The 7/7 first-block
LoRA grad is the binding test that the recurrence graph is intact —
gradient does flow back through the chain of forward passes.

### 3.2 The novelty axis (separate-reader supervision) is real but weak at this scale

Round-2 100-step held-out NLL (10 fixed-seed examples, scored every
25 steps):

| Step | Natural pairing | Permuted control |
|---|---|---|
| 0 | 8.24 | 6.37 |
| 25 | 6.32 | 6.64 |
| 50 | 6.86 | 9.24 |
| 75 | 7.86 | 8.43 |
| 99 | **7.91** | **8.95** |
| Δ over 100 steps | **−0.33 nat** | **+2.58 nat** |

The natural pairing trains on (image, template, question, answer)
that are mutually consistent; the permuted control shuffles
templates within the batch so the trace's surface form no longer
matches the (image, q, a). Outcome: natural improves marginally,
permuted **actively degrades**. The cross-run NLL comparison is
confounded by the eval RNG drawing different examples under each
condition (caveat noted in `INTERLEAVED_POC_RESULTS.md` §9.4),
but the **within-run direction** is unambiguous. The model can
distinguish natural from permuted; the discrimination is real.

The honest reading: the natural pairing's −0.33 nat over 100 steps
is small. The mechanism works; the *signal strength* at this scale
(3B model, 5 hand-written templates, solid-colour synthetic images,
single shared reader) is weak. This is consistent with the
litrecon §5 prior — the surveyed interleaved-VLM-latent works that
*do* report substantial gains all use real images, real CoT-style
reasoning data, and a Mirage-style three-stage curriculum.

### 3.3 The norm regulariser does not bite under the recurrence

Parallel baseline (`src/vl/model.py`, `configs/mini.yaml`): drives
‖h‖ from 190 → 59 (target 57.86) by step 20. Norm-loss term
collapses 17 445 → 1.79.

Interleaved (`configs/interleaved_mini_100.yaml`): drives ‖h‖
from 170 → 101 over 100 steps. Norm-loss term: 14 163 → 7 095.
Same `norm_weight_end=0.1`.

Two diagnostic probes:

| Diag | Setting | ‖h‖ at step 19 | Verdict |
|---|---|---|---|
| Baseline | default | ~169 → 156 | reference |
| **C1** | `norm_weight_end = 1.0` (10×) | 157 | **no improvement** — refutes "needs more weight" |
| **C2** | `.detach()` on recurrence input embedding | 128 | **modest help (~30% faster collapse)** — partially confirms self-sustaining-scale hypothesis |

C2 explicitly breaks the design (gradient no longer flows through
the recurrence chain) so it is a diagnostic only. The combined
finding: the recurrence creates a self-reinforcing scale that the
existing scalar norm penalty cannot easily shrink, and increasing
the penalty weight does not help. **This is the single clearest
"this design choice is being fought by the architecture" signal
in the POC and is worth investigating before scaling up.**

Hypotheses worth probing next (not done in this run):
- Replace per-position quadratic norm penalty with cosine-distance
  penalty against `V_sem[k mod T_v]` row-by-row (decouples scale
  from direction).
- Apply the norm penalty only to the **first** latent of each block
  (which is fed by the `<|latent_start|>` row, not by a recurrent
  hidden state).
- Use a layer-norm on the recurrence input feed rather than passing
  the raw hidden state.

### 3.4 Wall-clock cost matches the theoretical 10×

Round-1 measured ~6.5× slower per step than parallel. Round-2 100-
step run sustains ~0.88 s/step on Qwen2.5-VL-3B with no KV-cache
reuse across the chained forwards. The implementation does use
`past_key_values=True` correctly (verified in `model.py`); the cost
is dominated by `n+1` forward calls, not by recomputed prefix.
On H100 with tighter KV-cache management this likely drops to
3–4× parallel.

### 3.5 A real bug surfaced

`LossConfig.w_norm` was dead schema — the actual norm weight is
read from `cfg.loss.norm_weight_end`. Round-2's C1 diagnostic was a
no-op until patched. The parallel method's behaviour is unaffected
(it uses `norm_weight_end` directly), but the dataclass should be
cleaned up to either remove `w_norm` or have `norm_weight_end`
default to it. Filed as a separate cleanup item; not addressed in
this POC.

---

## 4. Pass/fail vs design §10

| Criterion | Threshold | Result |
|---|---|---|
| Gradient flow | nonzero on `new_emb[1,2]`, ≥50% LoRA | ✅ verified |
| No NaN | hard | ✅ no NaN in any run |
| ‖h‖ → target within 30 steps | < 80 by step 30 | ❌ — 140 at step 30, 101 at step 99 |
| Held-out NLL ≥ 0.5 nat improvement | hard | ❌ — only 0.33 nat over 100 steps |
| Permutation control < 50% of natural gap | hard | ✅ — perm degrades, natural improves (qualitatively the right sign) |

Two hard criteria failed (‖h‖ convergence and held-out-NLL
magnitude). The mechanism is sound; the signal-strength bar set by
the design doc is not met at this scale.

---

## 5. Honest assessment

**What this POC proves:**
1. The Coconut-recurrence-with-frozen-reader-anchor wiring is
   buildable with PEFT + Qwen2.5-VL and gradient flows correctly
   through the n+1 chained forwards.
2. The reader-anchor signal can distinguish natural-pairing from
   permuted-control traces (within-run direction is unambiguous),
   which is preliminary evidence that the latents are doing real
   work and not just generic regularisation.
3. The wall-clock cost (~6.5–10× parallel) matches the theoretical
   forward-pass count and is tractable on a single A6000 for POC
   work, but will be the binding constraint at 7B + cluster scale.

**What this POC does NOT prove:**
1. That the design *converges to a useful regime* under the project's
   pass thresholds. ‖h‖ does not reach the target norm; held-out
   NLL improvement is small. We need either (a) real data, (b) more
   steps, (c) a fix to the norm-regulariser failure mode, or some
   combination, before claiming the design is viable.
2. That the separate-reader signal beats single-model self-
   supervision (Mirage Stage 1 distillation against the model's
   own vision encoder). The novelty axis is structurally sound but
   the empirical case is weak at this scale.
3. Reader-transfer (option (ii) from design §4 — reader sees the
   full interleaved trace, not just the collapsed latents tensor)
   has not been tested. Round-1 used the simpler option (i)
   throughout.

**What I would NOT do based on this data:**
- Submit a 7B + multi-reader cluster sweep yet. Six A6000-hours of
  POC data showing weak signal at 3B + synthetic do not justify the
  cluster cost.
- Trust the natural-vs-perm held-out NLL gap as a primary signal.
  The cross-run RNG confound is real; the within-run direction is
  the salvageable evidence.
- Drop the norm regulariser. C1 + C2 say it's fighting the design,
  but removing it without a replacement (cosine-to-V_sem, layer-
  norm on the recurrence feed) likely sends ‖h‖ off-manifold.

---

## 6. Recommended next steps (priority-ordered)

1. **Swap synthetic data for GQA** (~3 h). The synthetic generator
   in `src/vl/interleaved/trainer.py:_sample_synthetic_example`
   produces solid-colour images with arbitrary object names — the
   latents have nothing real to encode. The implementation agent
   flagged this swap as a one-function change at the data-sampler
   boundary. *Highest expected information gain per hour.*
2. **Address the norm-regulariser failure mode** (~1 day).
   Implement the three probes from §3.3: cosine-instead-of-quadratic,
   first-latent-only penalty, layer-norm-on-recurrence-feed. Pick the
   one that drives ‖h‖ to target without degrading reader-NLL.
3. **Add reader option (ii)** (~1 day). The blind reader currently
   sees only the 8 latent vectors. The actual novelty is to have it
   consume the full interleaved trace (text + latents). Requires
   adapting `forward_anchor` to handle a mixed input (latents at
   trace positions, text elsewhere).
4. **Multi-reader (R=2)** (~half day). Currently R=1 (the
   generator-shared reader). Add the second frozen reader (Monet-
   SFT-7B per the project's existing recipe). This is the round-3
   spec's core mitigation for cross-reader transfer failure.
5. **(After 1–4)** Cluster 7B run with **explicit user `sb`
   approval**. The A6000 POC has bought enough information to
   propose a single, well-instrumented cluster cell — but per the
   project's standing rule, the user must approve every `sb`
   submission. Do not auto-launch.

The cleanest decision point: after step 1 (real data) is done,
re-evaluate. If real data closes the gap to the design pass
thresholds, the design is viable and steps 2–4 are about polishing
it. If real data does NOT close the gap, the conclusion is that
**the separate-reader signal is too weak relative to single-model
self-supervision** for an autoregressive interleaved trace — at
which point the right next move is probably a Mirage-style hybrid
(generator self-distillation against vision-encoder features
during Stage 1, reader-NLL during Stage 2) rather than persisting
with reader-only supervision.

---

## 7. What the user should actually run next

To reproduce or extend:

```bash
# verify the existing artefacts (no GPU needed)
ls src/vl/interleaved/                      # implementation files
cat docs/INTERLEAVED_POC_RESULTS.md         # raw data
cat results/interleaved_mini_100/eval.jsonl # held-out trajectory

# re-run the gradient probe (~30s on A6000)
MACHINE=local uv run python /tmp/interleaved_grad_probe.py

# re-run the 100-step natural training (~2 min on A6000)
MACHINE=local uv run python -m vl.train \
    --config configs/interleaved_mini_100.yaml \
    --variant interleaved
```

---

## 8. Files produced this session

```
docs/
  INTERLEAVED_LITRECON.md         (~2k words, 10 papers verified)
  INTERLEAVED_LATENT_DESIGN.md    (469 lines, 12-section design contract)
  INTERLEAVED_POC_RESULTS.md      (484 lines, two-round results)
  INTERLEAVED_POC_REPORT.md       (this file)

src/vl/interleaved/
  __init__.py        (23 lines)
  model.py           (build + forward, recurrence, ~232 lines)
  traces.py          (5 hand-written templates, ~181 lines)
  reader.py          (option (i) collapsed-latents reader, ~37 lines)
  trainer.py         (training loop + held-out eval, ~275+ lines)

configs/
  interleaved_mini.yaml             (50-step round-1)
  interleaved_mini_100.yaml         (100-step round-2)
  interleaved_mini_perm_100.yaml    (100-step perm control)
  diag/c1_wnorm_high.yaml           (norm 10×)
  diag/c2_detach_recurrence.yaml    (recurrence detach)

results/
  interleaved_mini/                  (round-1 natural, 50 steps)
  interleaved_mini_perm/             (round-1 perm, 29/50 steps)
  interleaved_mini_100/              (round-2 natural, 100 steps + eval)
  interleaved_mini_perm_100/         (round-2 perm, 100 steps + eval)
  interleaved_norm_diag/{c1,c2}/    (norm diagnostics, 20 steps)

src/vl/config.py    (added InterleavedConfig dataclass)
src/vl/train.py     (added --variant interleaved dispatch)
```

The reference parallel method (`src/vl/model.py`, `src/vl/readers.py`,
`src/vl/losses.py`, `src/vl/trainers/sft_anchor.py`) was not modified.


---

## 9. Round-3 update — the regulariser was the confound

**TL;DR:** The natural-vs-perm gap in §3.2 was an artifact of the L_norm
regulariser. With the regulariser removed, both natural and perm achieve
held-out NLL of ~2-3 nat (a 4-nat improvement from start). The latents are
not yet encoding image content — the shared-LoRA reader is just memorising
the answer distribution.

### 9.1 What round-3 ran

Two new datasets and four new probes (full data in
`docs/INTERLEAVED_POC_RESULTS.md` §10):
- **Pivot:** GQA isn't cached locally and downloading 50 GB violates
  the offline constraint. Built a **programmatic "shapes" dataset**
  (`src/vl/interleaved/trainer.py:_make_shapes_image_and_gt`) — 224×224
  scenes of 1-5 colored shapes (circle/square/triangle in
  red/blue/green/yellow), with ground-truth-aware questions
  (count-by-color, presence-of-color+kind, color-of-unique-kind).
- **C3 probe:** norm penalty on first-of-block latents only (those NOT
  fed by recurrence). Disconfirmed the round-2 self-sustaining-scale
  hypothesis — even non-recurrent positions resist the penalty.
- **C4 probe:** norm penalty entirely OFF. Held-out NLL improved
  dramatically. Held-out NLL went 7.19 → 2.98 over 100 steps.

### 9.2 Held-out NLL — the headline table

| Configuration | natural Δ over 100 steps | perm Δ | gap |
|---|---|---|---|
| Round-2 synthetic, norm on | −0.32 | +2.58 | +2.90 (natural better) |
| Round-3 shapes, norm on | −1.53 | +2.35 | +3.88 (natural better) |
| Round-3 shapes, **norm off** | **−4.21** | **−4.25** | **−1.00** (perm slightly better) |

The "natural beats perm" finding was created by the regulariser interfering
with the perm condition. With clean training, both improve equally —
strong evidence that under R=1 with shared LoRA, the latents function as
a generic CoT prior, not image-specific encoding.

### 9.3 Why the norm regulariser hurts the interleaved variant

Target_norm=57.86 is the empirical mean post-merger visual-token norm for
Qwen2.5-VL-7B's parallel forward. **Deep-position hidden states in a
recurrent autoregressive trace naturally live at much higher norm**
(180–200 in our runs). Forcing them to 57.86 is a 3× compression that
the model can only achieve by sacrificing reader signal. Removing the
penalty lets `‖h‖` settle at the natural manifold AND lets the reader-NLL
drop cleanly.

This is a project-wide implication: **the parallel method's
target_norm=57.86 is NOT a portable inductive bias.** It applies only
to the parallel-emission topology where all K latent positions sit at
shallow positions in a short prompt. Any future extension that runs the
generator deeper (autoregressive trace, multi-pass refinement) must
re-calibrate or drop the penalty.

### 9.4 The diagnosis: shared-LoRA reader is a degenerate signal

The reader is the same Qwen2.5-VL-3B as the generator, with the same LoRA
applied. When LoRA learns to make gold-answer tokens more likely, it
improves NLL for any held-out (q, a) pair regardless of what's in `h`.
This is exactly the project's prior finding for the parallel method
(`docs/inherited/REPORT.md` §17.1: random-target reproduces ~70 % of the
natural-target gain). The interleaved variant inherits the same shortcut.

### 9.5 The next experiment that actually matters

**R=2 with a truly-frozen second reader.** Concretely, on a single A6000:
- Reader-1: shared with generator (current behaviour).
- Reader-2: a separate copy of `Qwen/Qwen2.5-VL-3B-Instruct` loaded
  fresh, NO LoRA applied, frozen — costs ~12 GB additional VRAM
  (total ~25 GB on A6000, comfortable headroom).

If the natural-vs-perm gap re-emerges with reader-2 (and is robust
without a regulariser confound), the design is sound. If it doesn't,
the conclusion is that the separate-reader-supervision novelty axis
needs Mirage-style Stage-1 distillation (against the model's own vision
encoder) before reader-NLL can become a useful signal at all.

The implementation lift is small (~1 file, ~30 lines in `readers.py` to
support a no-LoRA frozen second reader; `load_anchors` already handles
shared-or-not detection). The runtime cost is ~1.5× current per-step
(two reader passes instead of one).

### 9.6 What round-3 changes about the recommended next steps

The §6 ordering above is partially superseded:
1. ~~Swap synthetic data for GQA~~ — DONE via shapes pivot. Don't bother
   with GQA until the design itself is validated.
2. ~~Address the norm-regulariser failure mode~~ — DONE. **Drop the
   L_norm penalty for the interleaved variant.** Re-introduce only with
   a re-calibrated target if/when one is empirically determined.
3. **NEW priority #1:** add reader-2 (frozen, no shared LoRA) and re-run
   the natural-vs-perm comparison. This is the binding test of whether
   the design has a real signal.
4. (Then) Add reader option (ii) — full-trace consumption.
5. (Then) Cluster 7B run with explicit user `sb` approval.

### 9.7 Round-3 artefacts

```
src/vl/interleaved/trainer.py    (added shapes generator, first_latent_norm_only)
src/vl/config.py                 (extended data_source Literal, added first_latent_norm_only)

configs/interleaved_shapes_100.yaml
configs/interleaved_shapes_perm_100.yaml
configs/interleaved_shapes_nonorm_100.yaml         <-- the winning natural config
configs/interleaved_shapes_nonorm_perm_100.yaml    <-- shows the "win" is generic
configs/diag/c3_first_latent_norm.yaml
configs/diag/c4_no_norm.yaml

results/interleaved_shapes_*/eval.jsonl + losses.jsonl  (4 runs × 2 files)
results/interleaved_norm_diag_c3/{losses,eval}.jsonl
results/interleaved_norm_diag_c4/{losses,eval}.jsonl
```

`docs/INTERLEAVED_POC_RESULTS.md` §10 contains the full per-step
trajectories and probe analysis.


---

## 10. Round-3 follow-up: R=2 frozen reader doesn't fix the gap

I ran the R=2 frozen-reader experiment (the binding test from §9.5 above)
within this same continuation. **Result: gap is still zero**
(natural 3.92, perm 3.91 at step 99).

### 10.1 What this rules out

The "shared-LoRA reader memorises answers" theory from §9.4 is **not the
full story**. Adding a truly-frozen second reader (separate
`Qwen/Qwen2.5-VL-3B-Instruct` instance, no LoRA, no parameter updates)
should mean reader-2's NLL only drops if `h` carries image-relevant
information. It does drop — but identically for natural and perm. The
latents must be encoding something that doesn't depend on whether the
image matches the (q, a).

### 10.2 What the issue actually is — the K_q copy bug

The interleaved trainer's per-step batch construction:

```python
batch_records = [
    {"questions": [(trace.question, trace.answer)] * max(1, cfg.loss.K_q)}
]
```

`K_q=2` means the SAME `(q, a)` is duplicated. The original
`nll_multi_anchor` loss expects K_q *different* questions per image so
that a single `h` must serve all of them — that's the q-invariance
pressure that forces image-general encoding (round-3 spec §3 of
`docs/inherited/ROUND3_POC_DESIGN.md`). The interleaved trainer
inherited the loss interface but never plumbed the actual variation
through. With one unique (q, a) per image per step, `h` can specialise
to that single question — no incentive to encode the image generally.

This is the most plausible reason both natural and perm achieve the
same held-out NLL: the latents are encoding the **answer marginal
conditioned on the question type**, not image content. That marginal
is the same regardless of whether the image actually matches.

### 10.3 The actual binding test — multi-question per image

Required change:
- `_sample_shapes_example` returns one image + K_q≥3 *different* (q, a)
  pairs derived from the same ground-truth (e.g., "How many circles?"
  + "How many red shapes?" + "Is there a blue square?" — three image-
  specific questions about the SAME scene).
- Trainer populates `batch_records` with the actual K_q pairs.

Estimated effort: ~30 minutes (the shapes generator already records
all the GT needed; just need to sample multiple questions per scene).

If the natural-vs-perm gap re-emerges with multi-Q + R=2, the design
is sound and the previous null result was an artefact of the K_q-copy
bug. If it still collapses, the design needs Mirage-style Stage-1
distillation (litrecon §3 single-model self-supervision baseline)
before reader-NLL can become a useful signal.

### 10.4 Updated next-steps ranking

1. **Multi-question per image** (~30 min) — the binding test.
   `src/vl/interleaved/trainer.py` + `_sample_shapes_example`.
2. **(If 1 succeeds)** Add reader option (ii) — full-trace consumption.
3. **(If 1 fails)** Pivot to Mirage Stage 1 (distill latents against
   generator's own vision-encoder features) before reader-NLL.
4. Cluster 7B run — needs explicit user `sb` approval.

The §6 priority list above is now superseded:
- Norm regulariser fix → DONE (drop it).
- R=2 frozen reader → DONE (doesn't independently fix the gap).
- New #1: K_q multi-question.


---

## 11. Round-3 follow-up: multi-Q binding test — design fails

I implemented the K_q-copy fix from §10.2 (`_sample_shapes_multiq_example`
+ `extra_qa_pairs` parameter on `run_one_step`) and ran the binding test:
multi-Q + R=2 + no-norm.

### 11.1 The decisive cell — and three controls

| Configuration | natural | perm | gap | verdict |
|---|---|---|---|---|
| R=1, norm-on, single-Q | 5.66 | 8.58 | +2.93 | ✅ apparent — but artifact (regulariser) |
| R=1, no-norm, single-Q | 2.98 | 1.98 | −1.00 | ❌ perm wins |
| R=2 frozen, no-norm, single-Q | 3.92 | 3.91 | −0.00 | ⚠️ identical |
| R=2 frozen, no-norm, multi-Q | 4.55 | **3.22** | **−1.33** | ❌ perm wins |

### 11.2 The verdict

Even with multi-Q (3 different questions per scene against the same h),
two readers (one frozen), and no confounding regulariser, the perm
condition still beats the natural pairing on held-out NLL by 1.33 nat.
This rules out three more hypotheses:
- It's not the K_q-copy bug (§10.2): fixed, gap still wrong direction.
- It's not the shared-LoRA reader: frozen reader-2 doesn't fix it.
- It's not the regulariser: no-norm makes the gap WORSE (more negative).

The latents are encoding the question-conditioned answer marginal,
which transfers to held-out evaluation regardless of image content.
The perm condition trains on more diverse (image, q, a) triples per
step, which produces better marginal coverage and thus better held-out
NLL — but this is the model gaming the metric, not real visual signal.

### 11.3 What the POC has now established

The interleaved-latent + frozen-reader-supervision design needs an
**explicit grounding signal** beyond reader-NLL to train usefully.
The four cells above exhaust the local-A6000 design space that can
be explored without adding new training-loss machinery. The clean
falsification: with the most generous experimental setup (best of
the four cells = +0.00 gap with R=2 single-Q), the design produces
no detectable image-grounding signal.

This matches the litrecon §3 prior: every interleaved-VLM-latent
work that reports substantial gains uses self-supervised grounding
(Mirage Stage-1 distillation against vision-encoder features, ILVR
momentum-teacher, Latent Sketchpad MSE-against-encoder, LaViT
cosine-to-teacher). None of them rely on reader-NLL alone.

### 11.4 The recommended pivot — Mirage-style Stage-1

The project already has `concept_loss` in `src/vl/losses.py` doing
cosine-to-V_sem on h for the parallel method. The interleaved
trainer wires it in but it's a *secondary* term (w_concept=0.3) that
the reader-NLL dominates. The Mirage recipe inverts this:
1. **Stage 1** (steps 0..500): `w_nll=0`, `w_concept=1.0`. Train h to
   regress against the generator's vision-encoder features. No
   reader supervision.
2. **Stage 2** (steps 500..1000): linearly anneal `w_nll: 0 → 1.0`,
   `w_concept: 1.0 → 0.3`. Bring the reader signal in only after h
   is on the manifold.
3. **(Optional Stage 3)** RL refinement.

This is a single-cell experiment to add to the next round. The
implementation lift is small — just a curriculum schedule on the
existing weights, plus longer step budget. No new code; just a
config + a curriculum tweak in `vl.curriculum`.

### 11.5 What round-3's binding test does NOT change

The mechanism (Coconut recurrence, frozen-reader supervision, K_q
multi-Q dispatch) is correctly implemented and gradient-tested.
The infrastructure is reusable for the Stage-1 pivot — only the
loss weighting needs to change. None of the round-1, round-2, round-3
artefacts need to be discarded; they form the baseline against which
the Stage-1 pivot will be measured.

### 11.6 Round-3 binding-test artefacts

```
src/vl/interleaved/trainer.py    (added _sample_shapes_multiq_example,
                                  extra_qa_pairs param, multi_q dispatch)
src/vl/config.py                 (added multi_q_per_image flag)

configs/interleaved_shapes_multiq_r2_100.yaml         (binding test natural)
configs/interleaved_shapes_multiq_r2_perm_100.yaml    (binding test perm)

results/interleaved_shapes_multiq_r2_100/{losses,eval}.jsonl
results/interleaved_shapes_multiq_r2_perm_100/{losses,eval}.jsonl
```

Six 100-step runs in this continuation:
- shapes natural + perm × {norm-on, no-norm} × {R=1, R=2-frozen}
- shapes natural + perm × multi-Q × R=2 × no-norm

All using Qwen2.5-VL-3B + LoRA r=16 on the local A6000. Total
~70 minutes of compute. Reference parallel method
(`src/vl/model.py`, `src/vl/readers.py`, `src/vl/losses.py`,
`src/vl/trainers/sft_anchor.py`, `src/vl/data/gqa.py`) again
unmodified.
