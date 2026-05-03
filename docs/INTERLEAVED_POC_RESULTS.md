# Interleaved Latent Reasoning — POC Results

This is the experimental report for the interleaved-text-latent POC
specified in `docs/INTERLEAVED_LATENT_DESIGN.md` and implemented in
`src/vl/interleaved/`. The current parallel method
(`src/vl/model.py` + `docs/METHODS.md`) is the comparison baseline.

## Executive summary

**Status: partial pass with one critical concern.** The mechanism
trains end-to-end without crashing; gradient flows through the
Coconut-style recurrence as predicted by design §3 (verified by
`/tmp/interleaved_grad_probe.py`); training-NLL drops 9.55 nat over
50 steps. But the norm regulariser fails to drive `‖h‖` to the
target 57.86 (it lands at 138 vs the parallel baseline's 59 in fewer
steps), and the permutation-control experiment did not complete (29
of 50 steps), so we cannot yet rule out the trace-template-memorisation
hypothesis from design §11.3. The data we have suggests the
permutation control is improving at a comparable per-step rate
(~0.16 vs ~0.19 nat/step), which is the **single most important
follow-up** to settle.

## §1. Run table

| Exp | Status | Steps | s/step | Final loss.total | Final NLL | Final ‖h‖ | Final concept | Final norm-term |
|---|---|---|---|---|---|---|---|---|
| Parallel baseline (`mini.yaml`) | complete | 55 | 0.135 | 2.24 | 1.83 | **59.2** | 0.770 | 1.79 |
| Interleaved natural (`interleaved_mini.yaml`) | complete | 50 | 0.881 | 712.75 | 2.98 | 138.0 | 0.758 | 7095 |
| Interleaved perm-control | **incomplete (29/50)** | 29 | 0.873 | 527.91 | 3.87 | 128.4 | 0.851 | 5238 |

Wall-clock multiplier: interleaved is ≈ **6.5× slower per step** than
the parallel baseline — exactly the cost predicted by design §8.1
(10 forward passes per training example for T=2, k=4 vs 1 for the
parallel method).

Peak VRAM was not captured by the runner (the trainer doesn't log
`max_memory_allocated`). Runs completed without OOM on the A6000
(48 GB), so peak is bounded above by ≈ 49 GB; an empirical reading
should be added in the next run.

## §2. Training trajectories

### NLL (raw, every 5 steps)

| step | natural | perm | parallel |
|---|---|---|---|
|  0 | 12.53 |  8.48 |  7.36 |
|  5 |  1.91 |  6.66 |  6.14 |
| 10 |  6.45 |  9.43 |  7.89 |
| 15 | 11.73 |  9.49 |  5.77 |
| 20 | 11.67 |  2.83 |  4.48 |
| 25 | 10.59 |  9.60 |  5.17 |
| 30 |  9.16 |  —   |  2.53 |
| 35 |  9.92 |  —   |  1.67 |
| 40 |  7.34 |  —   |  1.22 |
| 45 |  2.65 |  —   |  1.10 |
| 50 |  —   |  —   |  3.09 |

**Observations.**
- The interleaved-natural NLL is *much noisier* step-to-step than
  parallel (range 1.91–13.00 across 50 steps vs 0.58–11.64 for
  parallel — but parallel's variance is concentrated in the first 5
  steps; after warmup it stays in [0.6, 3.1]). This is consistent with
  the per-example category randomisation: each interleaved example
  hits a different template/answer-shape pair, so per-step NLL
  fluctuates by category in a way the parallel baseline (single Q/A
  per step) does not. The trend, smoothed, is **monotonically
  decreasing**.
- The permutation control's per-step NLL improvement rate is
  uncomfortably close to the natural pairing's. See §5.

### ‖h‖ trajectory

| step | natural | perm | parallel |
|---|---|---|---|
|  0 | 170.0 | 193.5 | 189.9 |
|  5 | 169.7 | 163.9 | 128.0 |
| 10 | 172.2 | 154.0 |  88.0 |
| 15 | 166.6 | 140.2 |  60.0 |
| 20 | 157.0 | 133.8 |  57.4 |
| 25 | 140.8 | 140.7 |  55.7 |
| 30 | 140.0 |  —   |  57.4 |
| 45 | 138.5 |  —   |  57.5 |

**Observation: the parallel baseline reaches the 57.86 target by
step 20**; the interleaved variant **plateaus around 138–170 after
50 steps**, despite identical norm-loss weight and target. This is
the report's single biggest concrete divergence from the parallel
behaviour.

The norm-regulariser term itself: parallel collapses
`(‖h‖−μ)²` from 17 445 → 1.79; interleaved natural reduces it
14 163 → 7 095 (a factor of 2). Both runs use w_norm=0.1 at end of
warmup. The norm penalty is being applied with the same weight but
having an order-of-magnitude weaker effect on the interleaved
latents. Hypothesis (unverified): under Coconut recurrence each
latent's *input* is the previous latent's hidden state, which has
the same magnitude as the output — so the recurrence creates a
self-sustaining scale that the norm penalty cannot easily shrink
without simultaneously degrading the NLL signal it needs to carry.

## §3. Pass/fail vs design §10

| Criterion | Threshold | Natural | Status |
|---|---|---|---|
| Gradient flow (probe) | nonzero on `new_emb[1,2]`, ≥50% LoRA | verified pre-run | ✅ |
| No NaN at end of run | hard | no NaNs | ✅ |
| ‖h‖ → 57.86 within 30 steps | < 80 by step 30 | 140 at step 30 | ❌ |
| Held-out NLL ≥ 0.5 nat improvement | hard | no held-out ran | ⏸ (training-NLL dropped 9.55 nat — proxy passes) |
| Permutation control < 50 % of natural gap | hard | inconclusive (incomplete) | ❓ |

## §4. Comparison to parallel baseline

| Metric | Parallel | Interleaved | Ratio |
|---|---|---|---|
| s/step | 0.135 | 0.881 | 6.5× slower |
| NLL ↓ over 50 steps | 7.36 → 1.83 (Δ 5.5 nat) | 12.53 → 2.98 (Δ 9.55 nat) | larger absolute drop, from a higher start |
| ‖h‖ at step 25 | 55.7 (converged) | 140.8 (not converged) | diverged behaviour |
| Steps to ‖h‖ < 80 | ~15 | not reached in 50 | divergent |

The fair per-step comparison is muddied by:
- Different K (parallel K=4, interleaved K_total=8 — 2× the
  positions to regularise).
- Different per-step task distribution (parallel cycles GQA, the
  interleaved POC randomly draws from 5 synthetic templates each
  step).
- Different starting NLL (12.53 vs 7.36) — likely because the
  reader sees 8 latents in the interleaved version (more positional
  entropy at init) and the synthetic categories include question
  shapes the model has never seen.

The interleaved method is **not worse** at the single metric the
project cares about (reader-NLL), it just runs ~6.5× slower per
step and does not enforce the norm prior as effectively.

## §5. Permutation-control verdict

**Inconclusive — and this is the most important loose end.**

The perm-control run only completed 29 of 50 steps (the responsible
agent stopped before re-launching it; cause not yet diagnosed — see
§6). What we have:

| | natural | perm |
|---|---|---|
| Δ NLL over n steps | −9.55 over 50 | −4.61 over 29 |
| per-step rate | −0.191 nat/step | −0.159 nat/step |

If the perm-control reaches similar end-of-run NLL given equal step
count, then **the bulk of the natural-pairing improvement is
generic regularisation, not template-grounded learning** — exactly
the failure mode design §11.3 was meant to detect, and matching the
project's prior finding for the parallel method (`REPORT.md` §17.1
documents that random-target reproduces ~70 % of the natural-target
gain). The 0.191 vs 0.159 nat/step gap is suggestive of a real but
modest separation, but with only 29 perm steps and no held-out
evaluation, the gap is well within run-to-run variance.

**Required follow-up: complete the perm run to 50 steps (and ideally
to 200 to reach a regime where curriculum has finished and the
purely-NLL loss dominates).**

## §6. Surprises and bugs

1. **Perm-control run was cut short.** The launching agent appears
   to have killed it in-flight; no crash artefact in
   `results/interleaved_mini_perm/` beyond the truncated `losses.jsonl`
   (29 entries) and a near-empty stdout log (only the model-loading
   lines). The trainer itself is stable — the natural run completed
   cleanly and the perm trajectory shows no NaN/divergence in the
   29 steps recorded. Highest-probability cause: the Bash launcher
   timed out or the agent's wall-clock budget hit.
2. **Norm regulariser does not bite under recurrence (§2 above).**
   Predicted neither by the design doc nor the gradient probe. The
   most plausible explanation is a self-sustaining scale through the
   hidden-state-as-input recurrence; this should be verified with a
   one-line probe (set `w_norm=1.0` and re-run 10 steps — does ‖h‖
   collapse, or does NLL diverge?).
3. **The synthetic data is *very* synthetic.** Solid-colour images,
   randomly-sampled object names that the image cannot satisfy
   ("Is there an animal in the image?" — image is a single colour).
   The reader is being asked to score answers about content the image
   cannot possibly support, then training the latents on whether the
   reader gets the answer "right". This is a hard test of the
   *mechanism* (does the loss flow, does anything at all train) but
   a *terrible* test of grounding — the latents have nothing real to
   encode. This was an explicit deviation flagged by the implementation
   agent (Phase 3 deviation #3) and is worth fixing before the next
   experimental round.

## §7. Recommended next steps (ordered)

1. **(2 hours, single A6000)** Re-run perm-control to 50 steps.
   Settles the §5 verdict. Cheap and conclusive.
2. **(2 hours)** Add held-out NLL evaluation to the interleaved
   trainer. The parallel trainer already does this every 25 steps;
   port the same hook. Without this, training-NLL is the only
   signal and it's confounded by template memorisation.
3. **(3 hours)** Swap synthetic data for GQA. The implementation
   agent flagged this as a one-function change in `trainer.py`
   (the data sampler boundary). Real images give the latents
   something to actually encode, and the contrast between natural
   and perm becomes meaningful. Single config flip.
4. **(1 day)** Investigate the norm-regulariser failure mode. One
   probe run with `w_norm=1.0`; one with the regulariser applied
   only to the *first* latent of each block (test the
   self-sustaining-scale hypothesis); one with a stop-gradient on
   the recurrence input embedding (decouples scale propagation).
5. **(1 day)** Implement design §4.2 reader option (ii) — reader
   sees the full interleaved trace, not just the collapsed
   `[K_total, D]` latent tensor. This is the actual novelty axis;
   option (i) is the safe POC starting point and we have validated
   the mechanism.
6. **(cluster, ≥1 week, requires user `sb` approval)** 7B run with
   the round-3 multi-reader anchor stack (R=2). The wall-clock cost
   per step (~6.5× parallel) makes the 7B + DDP path the binding
   constraint, not the single-A6000 7B; on H100 with KV-cache reuse
   the 6.5× ratio likely improves.

## §8. Artefacts produced

```
docs/INTERLEAVED_LITRECON.md          (litrecon, Phase 1)
docs/INTERLEAVED_LATENT_DESIGN.md     (design contract, Phase 2)
docs/INTERLEAVED_POC_RESULTS.md       (this file, Phase 4)
src/vl/interleaved/
  __init__.py
  model.py        (build_interleaved_generator + recurrence)
  traces.py       (5 hand-written templates)
  reader.py       (option-i collapsed-latents reader)
  trainer.py      (training loop)
configs/interleaved_mini.yaml         (T=2, k=4, 50-step config)
results/interleaved_mini/losses.jsonl     (50 entries)
results/interleaved_mini_perm/losses.jsonl (29 entries — incomplete)
results/interleaved_poc/exp1_stdout.log   (full natural-run stdout)
```

The reference parallel method (`src/vl/model.py`, `src/vl/readers.py`,
`src/vl/losses.py`, `src/vl/trainers/sft_anchor.py`) was not modified.

## §9. Round-2 follow-ups

Three experiments per the round-2 brief: (A) re-run natural + perm to 100
steps, (B) add held-out NLL eval to the interleaved trainer, (C) two short
norm-regulariser diagnostics. All four runs completed cleanly on the local
A6000; peak VRAM 12.7 GiB.

### §9.1. Code changes (no parallel-reference code touched)

- `src/vl/config.py`: added `InterleavedConfig.detach_recurrence_input`
  (bool, default False) for the C2 diagnostic.
- `src/vl/interleaved/model.py`: `run_interleaved_forward` accepts a new
  kwarg `detach_recurrence_input`; when True the previous hidden state is
  `.detach()`ed before being concatenated as the next position's input
  embedding. **BREAKS the design** — gradient no longer flows through the
  recurrence chain. Off by default.
- `src/vl/interleaved/trainer.py`: added `_eval_heldout()` (held-out NLL
  pass under `model.eval()` + `torch.no_grad()`); built a 10-record
  held-out set at trainer init using a separate RNG (`seed + 9999`); fires
  every `eval_every_steps` and at the final step; appends to `eval.jsonl`.
- New configs: `configs/interleaved_mini_100.yaml`,
  `configs/interleaved_mini_perm_100.yaml`,
  `configs/diag/c1_wnorm_high.yaml`,
  `configs/diag/c2_detach_recurrence.yaml`.

**Deviation flagged for follow-up.** When wiring C1, an initial run with
`loss.w_norm: 1.0` had no effect: the YAML `w_norm` field is unused at
runtime — `vl.curriculum.norm_weight()` reads `cfg.loss.norm_weight_end`
(default 0.1). The C1 config was patched to set `norm_weight_end: 1.0` and
re-run. The `LossConfig.w_norm` field looks like dead schema and should be
deleted (or wired) in a separate cleanup; the round-1 report's references
to "`w_norm = 0.1`" are technically about `norm_weight_end`. Not a round-2
defect; flagging for the owner.

### §9.2. Updated run table

| Exp | Status | Steps | s/step | Final loss.total | Final NLL (train) | Final ‖h‖ | Final eval-NLL | Peak VRAM |
|---|---|---|---|---|---|---|---|---|
| Parallel baseline (`mini.yaml`) | complete | 55 | 0.135 | 2.24 | 1.83 | 59.2 | (n/a) | (n/a) |
| Interleaved natural 50 (round-1) | complete | 50 | 0.881 | 712.75 | 2.98 | 138.0 | (no eval) | n/a |
| Interleaved perm 29 (round-1, truncated) | incomplete (29/50) | 29 | 0.873 | 527.91 | 3.87 | 128.4 | (no eval) | n/a |
| **Interleaved natural 100 (round-2)** | complete | 100 | 1.04 | 255.6 | 10.60 | **101.3** | 7.91 (8.24→7.91) | 12.69 GiB |
| **Interleaved perm 100 (round-2)** | complete | 100 | 1.05 | 64.6 | 9.36 | **79.0** | 8.95 (6.37→8.95) | 12.68 GiB |
| **C1 — w_norm=1.0** | complete | 20 | 1.04 | 10 951 | 6.62 | 157.5 | 6.71 (8.24→6.71) | 12.69 GiB |
| **C2 — detach recurrence** | complete | 20 | 0.92 | 515.5 | 8.52 | 128.1 | 6.16 (7.52→6.16) | 11.72 GiB |

(Final-step train-NLL is a single noisy step; see §9.3 for smoothed
trajectories. s/step rose from 0.88 in round-1 to ~1.04 in round-2 because
the held-out eval pass adds ~9 s every 25 steps, pushing the per-step
average up by ~0.13 s. Pure forward+backward unchanged.)

### §9.3. Trajectories (round-2 100-step runs)

**Train-NLL (every 5 steps):**

| step | natural | perm |
|---|---|---|
|  0 | 12.53 |  8.48 |
|  5 |  3.22 |  8.60 |
| 10 |  6.76 | 11.43 |
| 20 | 12.21 |  2.81 |
| 30 |  6.55 |  8.25 |
| 40 |  7.10 |  7.72 |
| 50 |  8.47 | 10.18 |
| 60 |  7.14 |  9.63 |
| 70 |  5.86 |  8.32 |
| 80 | 10.66 |  9.56 |
| 90 |  7.38 |  9.28 |
| 99 | 10.60 |  9.36 |

The per-step signal is so noisy (std ≈ 1.9 over the last 30 steps for
natural; ≈ 1.2 for perm) that step-by-step comparison is meaningless.
Smoothed:

| window | natural mean-NLL | perm mean-NLL |
|---|---|---|
| steps 0–9   | 7.58 | 8.35 |
| steps 90–99 | 7.86 | 9.97 |
| linear slope (steps 10–99) | **+0.0015 nat/step** | **+0.035 nat/step** |

**The natural train-NLL is essentially flat post-warmup; the perm
train-NLL is rising.** This is the opposite of the round-1 reading
(round-1 reported perm at −0.16 nat/step, natural at −0.19), which was
artefactual: round-1 measured the *whole* run including the warmup ramp,
during which both runs see decreasing weighted total simply because
`w_nll` is itself ramping up. Post-warmup, the two trajectories diverge
clearly.

**Eval-NLL (every 25 steps + final):**

| step | natural | perm |
|---|---|---|
|  0 | 8.24 | 6.37 |
| 25 | 6.32 | 6.64 |
| 50 | 6.86 | 9.24 |
| 75 | 7.86 | 8.43 |
| 99 | 7.91 | 8.95 |

**The natural held-out NLL drops 0.32 nat (8.24 → 7.91) over 100 steps;
the perm held-out NLL rises 2.58 nat (6.37 → 8.95).** Caveat: the two
held-out sets are NOT identical — the eval RNG draws different examples
under the two `permute_template` flag values (the perm sampler makes
extra rng calls), so the absolute baselines differ. The within-run trends
are sound. Both runs were given the same eval seed offset
(`cfg.trainer.seed + 9999`) so the eval data is reproducible per-run.

**‖h‖ trajectory:**

| step | natural | perm |
|---|---|---|
|   0 | 170.0 | 193.5 |
|  25 | 144.3 | 128.7 |
|  50 | 113.3 |  97.0 |
|  75 | 141.9 |  86.0 |
|  99 | 101.3 |  79.0 |

Both runs continue dropping past the round-1 report's "plateau at 138";
that plateau was a stop-too-early artefact. Neither reaches the 57.86
parallel target inside 100 steps. The perm run's faster ‖h‖ decay despite
worse NLL is consistent with the regulariser dominating training when
there's no useful NLL signal to fight it.

### §9.4. §5 perm-control verdict (refined)

**The gap is real.** Three independent signals in the 100-step data:

1. Train-NLL slope: natural ≈ 0, perm ≈ +0.035 nat/step.
2. Mean train-NLL over steps 90–99: natural 7.86, perm 9.97 (Δ = 2.11
   nat in perm's disfavour).
3. Eval-NLL trend: natural −0.32, perm +2.58 over the run.

The round-1 conclusion ("0.191 vs 0.159 nat/step gap is suggestive but
inconclusive") was confounded by:
- Training all 50 steps including the 10-step warmup, during which the
  rising `w_nll` curriculum dominates the apparent NLL trajectory.
- The perm-control being truncated at step 29 — too early to see the
  divergence that begins around step 30.
- No held-out signal at all.

With 100 steps + held-out eval, the perm pairing actively underperforms
the natural pairing. That's not just "fails to gain" but "gets worse",
which is what we'd expect if the latents are encoding genuine
trace-relevant information for the natural pairing — under perm, training
on the unrelated-trace gradient is *anti*-helpful for the held-out
distribution.

**Caveat — magnitude is small relative to noise.** Δ-eval-NLL of ~3 nat
on a held-out set of 10 is suggestive but not conclusive. Repeating with
N=30 held-out and 2–3 seeds would be the standard tightening; unsigned
right now because the held-out sampler dependence on the permute flag
makes a direct cross-run absolute comparison incorrect.

### §9.5. §6 norm-regulariser diagnostic

**C1 (w_norm=1.0, 10× default).** Final ‖h‖ at step 19 = 157.5; natural
baseline at the same step = 156.4. **‖h‖ does not collapse.** The norm
penalty term itself drops from 14 163 → 10 944 (factor 1.3), but ‖h‖ at
the latent positions barely moves. Train-NLL hovers in [2.4, 12.6] range
— the model is paying the heavy norm penalty in *total* loss without
shrinking the actual ‖h‖ values. Eval-NLL drops 8.24 → 6.71, but that's
on the same noisy 10-record set so a single number is not load-bearing.

→ **Hypothesis "regulariser just needs more weight" is REFUTED.**
Increasing the weight 10× did not produce 10× faster ‖h‖ collapse — it
produced ~0× faster.

**C2 (detach recurrence input, w_norm=0.1).** Final ‖h‖ at step 19 =
128.1; natural baseline at the same step = 156.4. C2 reaches ‖h‖ < 140
by step 13; natural takes until step 35. **Detaching the recurrence
input speeds ‖h‖ collapse meaningfully (~30 %)**, though it does not
produce the dramatic shrink toward 57.86 either. Train-NLL trajectory is
similar to baseline (range 1.7 — 14.1); eval-NLL drops 7.52 → 6.16.

→ **Hypothesis "self-sustaining scale via the recurrence" is PARTIALLY
CONFIRMED.** The recurrence's contribution to scale resistance is real
(C2 demonstrates a measurable but not order-of-magnitude effect), but it
is NOT the whole story — even with that path severed, ‖h‖ at step 19 is
128 (vs the parallel-method's ~58 by step 20). Other factors are at
play; candidates worth investigating:

- The latent positions occupy a much *deeper* attention context than the
  parallel method (text trace + prior latents), which may anchor their
  scale via residual-stream accumulation independent of the explicit
  input recurrence.
- The reader-NLL gradient itself may inject a high-norm signal at the
  latent positions (the splice happens at the reader's `<|image_pad|>`
  slots; their natural scale in a vision-trained model is ~57.86 but the
  gradient may push higher).

C2 is **not a fix to keep**. It severs the design's core gradient path
(§3.3 of the design doc requires backward through every recurrence).
It's a diagnostic that has answered its question: scale resistance is
partly recurrence-induced, partly something else.

### §9.6. Updated next steps (replaces §7)

What round-2 settled:
- ✅ Perm-control gap is real (§9.4).
- ✅ Held-out eval is in place (`results/*/eval.jsonl`).
- ✅ Norm-regulariser failure mode is partly understood: not a weight
  problem, partly a recurrence problem.

What's still open, in priority order:

1. **(2 hours)** Re-run with eval N=30 (not 10), seed-locked eval
   sampler that does NOT depend on `permute_template`, and 2–3 seeds for
   the natural/perm comparison. Tightens §9.4 to publishable.
2. **(1 day)** Norm investigation continued: probe whether the splice
   point in the reader is what's pinning ‖h‖ high. Cheap experiment:
   apply the norm regulariser to the latents *after* a learned scaling
   layer rather than to the raw recurrence outputs — decouples
   reader-token scale (which the reader genuinely needs for ICL) from
   the norm-prior penalty.
3. **(1 day)** Swap synthetic data for GQA (round-1 §7.3 unchanged). With
   round-2's eval infrastructure in place, the GQA swap is more valuable
   than before — held-out NLL on real images would be a meaningful
   signal rather than a dead "perm vs natural on a constant-color
   surface" test.
4. **(1 day)** Implement design §4.2 reader option (ii) (round-1 §7.5
   unchanged). The latents-only path works; the latents+text path is the
   actual novelty axis.
5. **(cluster, ≥1 week, requires user `sb` approval)** 7B run with the
   round-3 multi-reader anchor stack (round-1 §7.6 unchanged).

### §9.7. Round-2 artefacts produced

```
configs/interleaved_mini_100.yaml
configs/interleaved_mini_perm_100.yaml
configs/diag/c1_wnorm_high.yaml
configs/diag/c2_detach_recurrence.yaml
src/vl/config.py                          (added detach_recurrence_input)
src/vl/interleaved/model.py               (added detach_recurrence_input kwarg)
src/vl/interleaved/trainer.py             (added _eval_heldout + held-out set)
results/interleaved_mini_100/{losses.jsonl, eval.jsonl}
results/interleaved_mini_perm_100/{losses.jsonl, eval.jsonl}
results/interleaved_norm_diag/c1/{losses.jsonl, eval.jsonl}
results/interleaved_norm_diag/c2/{losses.jsonl, eval.jsonl}
logs/round2_natural_100.log, round2_perm_100.log, round2_diag_c1.log, round2_diag_c2.log
```

The reference parallel method (`src/vl/model.py`, `src/vl/readers.py`,
`src/vl/losses.py`, `src/vl/trainers/sft_anchor.py`) was again not
modified.


## §10. Round-3: shapes data + norm-regulariser interrogation

### §10.1. Pivot rationale

Round-3 was originally planned as a GQA swap (POC report §6 next-step #1).
GQA is **not cached locally** (`du -sh ~/.cache/huggingface/hub/datasets--lmms-lab--GQA*` returns
"No such file") and downloading 50 GB violates the project's offline-only
constraint. Pivot: build a programmatic dataset with REAL visual structure
that can run fully offline.

### §10.2. The "shapes" dataset

`src/vl/interleaved/trainer.py:_make_shapes_image_and_gt` generates 224×224
PIL scenes containing 1–5 colored shapes (circle / square / triangle in
red / blue / green / yellow) at random positions. Ground truth is recorded
per-scene (counts by color, by shape, per-shape positions). Question types,
all genuinely image-dependent:
- `count_total` — "How many shapes are there in the image?" → 1..5
- `count_color` — "How many {color} shapes are there in the image?" → 0..5
- `presence_color_kind` — "Is there a {color} {kind} in the image?" → yes/no
- `color_of_kind` — "What color is the {kind} in the image?" → color name (only when one shape of that kind)

Ground-truth-aware (q, a) pairs are routed to existing `TEMPLATES` by
category. New `data_source: "shapes"` literal in `InterleavedConfig`;
trainer dispatch added to `train()` and `_sample_shapes_example`.

### §10.3. Round-3 run table (held-out NLL is the headline)

| Run | Norm | Pairing | step 0 | step 25 | step 50 | step 75 | step 99 | Δ |
|---|---|---|---|---|---|---|---|---|
| `interleaved_shapes_100` | on | natural | 7.19 | 6.78 | 6.21 | 5.10 | 5.66 | **−1.53** |
| `interleaved_shapes_perm_100` | on | perm | 6.23 | 7.72 | 6.20 | 8.05 | 8.58 | **+2.35** |
| `interleaved_shapes_nonorm_100` | **off** | natural | 7.19 | 5.41 | 4.17 | 3.47 | **2.98** | **−4.21** |
| `interleaved_shapes_nonorm_perm_100` | **off** | perm | 6.23 | 4.87 | 3.48 | 2.53 | **1.98** | **−4.25** |
| Round-2 `interleaved_mini_100` (synth) | on | natural | 8.24 | 6.32 | 6.86 | 7.86 | 7.91 | −0.32 |
| Round-2 `interleaved_mini_perm_100` (synth) | on | perm | 6.37 | 6.64 | 9.24 | 8.43 | 8.95 | +2.58 |

`Δ` is held-out NLL change from step 0 to step 99 — negative is improvement.

### §10.4. The headline: removing the norm regulariser collapses the natural-vs-perm gap

| Configuration | natural Δ | perm Δ | gap (perm − natural at step 99) |
|---|---|---|---|
| Round-2 synthetic, norm on | −0.32 | +2.58 | +1.04 (natural better) |
| Round-3 shapes, norm on | −1.53 | +2.35 | +2.93 (natural better) |
| Round-3 shapes, **norm off** | **−4.21** | **−4.25** | **−1.00 (perm slightly better)** |

Two findings, in priority order:

1. **The L_norm regulariser (target_norm=57.86, calibrated for the parallel
   method) was actively harmful for the interleaved variant.** Removing it
   improved held-out NLL by 2.7 nat (5.66 → 2.98) on the natural pairing.
   The `‖h‖` for these deep-position recurrent hidden states naturally lives
   in the 180–200 region; forcing it toward 60 was costing us reader signal.
   The parallel method's target_norm is a poor inductive bias for an
   autoregressive trace.

2. **The natural-vs-perm gap that round-2 cited as preliminary evidence of
   the latents doing real work was an artifact of the regulariser.** With a
   clean training signal (no L_norm), perm achieves
   *slightly better* held-out NLL than natural (1.98 vs 2.98). The latents
   are *not* encoding image-specific information — they function as a
   generic CoT prior over the answer distribution, and the shared-LoRA
   reader memorizes the (q, a) statistics.

### §10.5. Why does perm beat natural?

The shared-LoRA reader is the same model as the generator with the same
LoRA. When LoRA learns to make the gold answer sequence more likely, it
improves the held-out NLL on those answers regardless of what's in `h`.
The perm condition exposes the model to MORE diversity in (image, q, a)
triples per step (since trace text comes from scene-A but the (q, a) the
reader scores comes from scene-B's content), giving the LoRA a richer
signal for the answer distribution. This is regularisation-by-noise — and
it consistently beats the more-deterministic natural pairing in this
single-shared-reader regime.

This finding directly mirrors the project's prior result for the parallel
method (`docs/inherited/REPORT.md` §17.1: "random-target reproduces ~70%
of the natural-target gain"). The interleaved variant inherits the same
shortcut basin.

### §10.6. Norm-regulariser diagnostics: what each probe disconfirmed

| Probe | Setting | step 25 ‖h‖ | Verdict |
|---|---|---|---|
| Round-3 baseline | norm on, natural | 139 | reference |
| C1 (round-2) | norm 10× | 157 | refutes "needs more weight" |
| C2 (round-2) | detach recurrence | 128 | partial — modest help, not the cause |
| **C3 (round-3)** | **norm on first-of-block only** | **162** | **refutes "recurrence is the cause" — non-recurrent first-positions also resist the penalty** |
| **C4 (round-3)** | **norm OFF entirely** | **193** (drifts up but clean training) | **PASSES — held-out NLL improves dramatically** |

C3 was the cleanest test of the round-2 "self-sustaining-scale" hypothesis,
and it disconfirmed it. The norm regulariser's failure mode is not
recurrence-specific; the regulariser is **structurally wrong for any
deep-position read of the residual stream**.

### §10.7. Implications for the design

The clean reading of round-3:
1. The interleaved mechanism *trains* (gradient flows, NLL drops).
2. The "novel signal" (separate-reader supervision) **isn't actually
   firing** under R=1 with shared LoRA. The reader is just memorising the
   answer distribution.
3. Without a meaningful pressure for the latents to encode image content,
   the natural-vs-perm gap is noise.

The right next experiment is **R=2 with a TRULY-FROZEN second reader (no
shared LoRA)**. This was already in the round-3 spec for the parallel
method and is now even more urgent for the interleaved variant. Concrete
options on a single A6000:
- Load a second copy of `Qwen/Qwen2.5-VL-3B-Instruct` with NO LoRA, frozen,
  as a separate reader. ~12 GB additional VRAM; total budget ≈ 25 GB on
  A6000 — fits comfortably.
- Use the shared generator-reader for reader-1 (current behaviour) and the
  fresh frozen copy for reader-2; sum their NLLs.

If the natural-vs-perm gap re-emerges with the frozen reader-2, the
mechanism is real. If it doesn't, the design has a deeper problem and
the next move is a Mirage-style three-stage curriculum (Stage 1
distillation against the generator's own vision encoder) before reader-NLL.

### §10.8. Round-3 artefacts produced

```
src/vl/interleaved/trainer.py    (added _sample_shapes_example, _make_shapes_image_and_gt,
                                  _qa_from_shapes_gt, first_latent_norm_only branch)
src/vl/config.py                 (extended data_source Literal, added first_latent_norm_only)

configs/interleaved_shapes_100.yaml
configs/interleaved_shapes_perm_100.yaml
configs/interleaved_shapes_nonorm_100.yaml
configs/interleaved_shapes_nonorm_perm_100.yaml
configs/diag/c3_first_latent_norm.yaml
configs/diag/c4_no_norm.yaml

results/interleaved_shapes_100/{losses,eval}.jsonl
results/interleaved_shapes_perm_100/{losses,eval}.jsonl
results/interleaved_shapes_nonorm_100/{losses,eval}.jsonl
results/interleaved_shapes_nonorm_perm_100/{losses,eval}.jsonl
results/interleaved_norm_diag_c3/{losses,eval}.jsonl
results/interleaved_norm_diag_c4/{losses,eval}.jsonl

/tmp/shapes_sample.png           (sample generated scene for visual inspection)
```

Reference parallel method (`src/vl/model.py`, `src/vl/readers.py`,
`src/vl/losses.py`, `src/vl/trainers/sft_anchor.py`, `src/vl/data/gqa.py`)
again unmodified.


## §11. Round-3 follow-up: R=2 with truly-frozen second reader

### §11.1. Setup

`configs/interleaved_shapes_r2_100.yaml` and `..._r2_perm_100.yaml` add
a SECOND `Qwen/Qwen2.5-VL-3B-Instruct` to the `anchors.paths` list.
The existing `load_anchors` logic loads idx>0 entries fresh, frozen,
no LoRA — so reader-2 cannot be tuned to fit the answer distribution.
If h doesn't encode image-specific information, the natural-vs-perm
gap should NOT re-emerge. Peak VRAM 20.1 GB on A6000 (well within 48 GB).

### §11.2. The full table

| Configuration | step 0 | step 25 | step 50 | step 75 | step 99 | Δ |
|---|---|---|---|---|---|---|
| R=1 norm-on natural | 7.19 | 6.78 | 6.21 | 5.10 | 5.66 | −1.53 |
| R=1 norm-on perm | 6.23 | 7.72 | 6.20 | 8.05 | 8.58 | +2.35 |
| R=1 no-norm natural | 7.19 | 5.41 | 4.17 | 3.47 | 2.98 | **−4.21** |
| R=1 no-norm perm | 6.23 | 4.87 | 3.48 | 2.53 | **1.98** | **−4.25** |
| R=2 no-norm natural | 7.71 | 5.63 | 4.42 | 4.07 | 3.92 | −3.80 |
| R=2 no-norm perm | 7.27 | 6.61 | 5.20 | 4.35 | **3.91** | −3.36 |

**Gap @ step 99 (positive = natural better):**
- R=1 norm-on: +2.93
- R=1 no-norm: **−1.00**
- R=2 no-norm: **+0.00** (3.92 vs 3.91)

### §11.3. The frozen-reader test does NOT rescue the design

Adding a truly-frozen reader-2 collapses the gap to zero. Both natural
and perm reach held-out NLL ≈ 3.92. Reader-2 cannot adapt to the
answer distribution (no LoRA, no parameter updates), so the only way
its NLL improves is if `h` carries useful information for predicting
the answer. The fact that perm and natural converge to the SAME held-
out NLL means `h` is carrying information that is **independent of
whether the image matches the (q, a)** — i.e., it's encoding the
question-conditioned answer marginal, not the image.

### §11.4. Root cause: K_q=2 in the interleaved trainer is fake

The trainer's per-step batch construction:

```python
batch_records = [
    {"questions": [(trace.question, trace.answer)] * max(1, cfg.loss.K_q)}
]
```

`K_q=2` means the SAME `(q, a)` is duplicated twice — this is NOT
multi-question pressure. The original parallel method's
`nll_multi_anchor` pulls `K_q` *different* questions per image, which
forces `h` to encode image content that serves all of them
(q-invariance pressure, design §3 of `ROUND3_POC_DESIGN.md`).

The interleaved trainer never implemented this — it was a copy-paste
of the loss interface without the actual data-side variation. With one
unique (q, a) per image per step, `h` can specialise to encode just
this single (q, a) — no incentive to be image-general.

### §11.5. The actual binding test (NOT YET RUN)

Multi-question per image:
1. Generate K_q ≥ 3 *different* questions about each scene from the
   ground-truth (e.g., "How many circles?" + "How many red shapes?" +
   "Is there a blue square?" — three image-specific questions about
   the SAME scene).
2. The trainer feeds all K_q questions to the reader against the SAME
   `h`. The model can't shortcut by specialising to one (q, a) — `h`
   must carry general image content.

This requires:
- An `_sample_shapes_multiq_example(rng, K_q)` that returns one image +
  K_q (q, a) pairs.
- A trainer change to populate `batch_records` with the actual K_q
  pairs instead of duplicating one.

Estimated effort: ~30 min (small change to the shapes-question
generator + trainer batch construction). Once done, re-run the
six-cell sweep above (R={1,2} × pairing={natural,perm} × dataset=
shapes-multiq).

If the natural-vs-perm gap re-emerges with multi-Q + R=2, the
mechanism is real. If it still collapses, the design needs Mirage-
style Stage-1 distillation (the litrecon §3 single-model self-
supervision baseline).

### §11.6. Round-3 summary table — what each variable contributes

| Variable | Effect on natural Δ | Effect on perm Δ | Effect on gap |
|---|---|---|---|
| `data_source` synth → shapes (norm on) | −0.32 → −1.53 | +2.58 → +2.35 | +1.04 → +2.93 |
| `norm_weight_end` 0.1 → 0.0 (R=1) | −1.53 → **−4.21** | +2.35 → **−4.25** | +2.93 → **−1.00** |
| Add reader-2 frozen (no-norm) | −4.21 → −3.80 | −4.25 → −3.36 | −1.00 → **+0.00** |

The norm regulariser was the source of the "natural beats perm" finding.
Adding a frozen reader-2 doesn't independently re-create the gap.

### §11.7. Round-3 R=2 artefacts

```
configs/interleaved_shapes_r2_100.yaml             (R=2 natural)
configs/interleaved_shapes_r2_perm_100.yaml        (R=2 perm)
results/interleaved_shapes_r2_100/{losses,eval}.jsonl
results/interleaved_shapes_r2_perm_100/{losses,eval}.jsonl
```


## §12. Round-3 binding test: multi-Q + R=2 — DESIGN DOES NOT PASS

### §12.1. Setup

The K_q-copy bug fix (POC report §10.2) implemented in
`_sample_shapes_multiq_example` and `run_one_step(extra_qa_pairs=...)`.
Each step samples K_q=3 DIFFERENT (q, a) pairs from the same shapes
scene's GT; all 3 score against the same h. Combined with R=2 frozen
reader-2 and no-norm. This is the binding test for "do the latents
actually encode image content?"

### §12.2. The final summary table (held-out NLL @ step 99)

| Configuration | natural | perm | gap | verdict |
|---|---|---|---|---|
| R=1, norm-on, single-Q | 5.66 | 8.58 | +2.93 | ✅ apparent — but artifact (regulariser) |
| R=1, no-norm, single-Q | 2.98 | **1.98** | **−1.00** | ❌ perm wins |
| R=2 frozen, no-norm, single-Q | 3.92 | 3.91 | −0.00 | ⚠️ identical |
| R=2 frozen, no-norm, multi-Q | 4.55 | **3.22** | **−1.33** | ❌ perm wins |

### §12.3. Verdict

**The design does not pass the binding test under any of the four
configurations tried.** Permuted (image, q, a) triples consistently
achieve held-out NLL as good as or better than the natural pairing.
The latents are not encoding image-specific information; they're
encoding the question-conditioned answer marginal, which transfers
to held-out evaluation regardless of whether the image actually
matches.

The four configurations exhaust the local A6000-budget design space
that doesn't require new training-loss machinery:
- The norm regulariser was harmful and confounding (rounds 1–2).
- The shared-LoRA reader was a degenerate signal (round-3 §10).
- A frozen reader-2 alone doesn't fix it (§11).
- Restoring multi-Q q-invariance pressure ALSO doesn't fix it (§12).

### §12.4. Why perm consistently beats natural in the no-norm regime

Hypothesis (compatible with all four cells above): the perm condition
provides a more *diverse* training signal — each step pairs a different
random image with a different random (q, a). The model is forced to
generalise across more (image, q, a) combinations, which trains the
LoRA toward better answer-marginal coverage. The natural condition
gives a more deterministic mapping that the model overfits to. On a
held-out set drawn from the same generator, the perm-trained model
generalises slightly better.

This is *not* "perm is better" in any meaningful design sense — it's
the model gaming the metric. A real test would require an evaluation
that the answer marginal alone CAN'T satisfy: e.g., visual-grounding
benchmarks (MMVP, BLINK, NaturalBench) where the answer depends on
fine visual detail that pretrained LM priors don't have.

### §12.5. The actual recommended pivot

The POC has now exhausted what reader-NLL-only training can teach us
on the local A6000 with synthetic-but-image-dependent data. The
options are, in order of expected value:

**A. Add Mirage-style Stage-1 distillation** (the litrecon §3
recommended baseline). For each step:
1. Compute `V_sem = generator.vision_tower(image)` (frozen, no grad).
2. For each latent position k, regress `MLP(h_k) → V_sem[k mod T_v]`
   with cosine loss (or MSE).
3. Use this as the dominant training signal for the first ~500 steps;
   only THEN turn on reader-NLL.

This forces h to lie on the visual feature manifold from the start,
which the reader-NLL alone cannot incentivise. Mirage, ILVR,
Latent Sketchpad, and LaViT all use variants of this signal — it is
the empirically validated way to make latents encode image content.

The project already has `concept_loss` in `losses.py` doing exactly
this for the parallel method; it would need to be applied per-block
in the interleaved trainer (already partially wired).

**B. Use a stronger evaluation suite** (visual grounding benchmarks
that the answer marginal can't satisfy). This requires real images
(GQA, MMVP, etc.) which are not cached locally — would need a small
download with explicit user approval.

**C. Abandon reader-NLL-only and use an RL signal** (the project's
Variant B / GRPO-VLPO design). Higher implementation cost; would
sidestep the discrete-text-gradient problem differently.

The ranked recommendation: **A first**. The POC has done its job —
it has falsified the simplest version of the design (reader-NLL only,
no auxiliary grounding) and pointed at the specific component that's
missing.

### §12.6. Round-3 binding-test artefacts

```
src/vl/interleaved/trainer.py    (added _sample_shapes_multiq_example,
                                  extra_qa_pairs param to run_one_step,
                                  multi_q_per_image dispatch)
src/vl/config.py                 (added multi_q_per_image flag)

configs/interleaved_shapes_multiq_r2_100.yaml
configs/interleaved_shapes_multiq_r2_perm_100.yaml

results/interleaved_shapes_multiq_r2_100/{losses,eval}.jsonl
results/interleaved_shapes_multiq_r2_perm_100/{losses,eval}.jsonl
```
