# Evaluation protocols

Mirrors POC `evaluate_*.py` scripts; full design in
`inherited/EVAL_BENCHMARK_PLAN.md` (528 lines).

## Held-out NLL on q' (per image)

For each training image, score `-log p_anchor1(a' | h, q')` where `q'` is a
held-out question about the same image (POC2 protocol). 30-sample subset
during training (every 100 steps); full set at end.

**Pass:** mean held-out NLL ≤ 2.5 nat (cf POC round-3 mit-B λ=0.1 baseline 2.39).

## Reader-transfer NLL on Monet-SFT-7B (cleaner sibling)

Same h, scored under anchor-2 = Monet-SFT-7B-stage2. **This is the hardest
target — POC round-2 found NO mitigation crossed this gap; round-3's multi-
anchor NLL is the proposed structural fix.**

**Pass:** mean transfer NLL ≤ 4.5 nat (POC round-3 baseline 8.0–8.6 across
all latent variants).

## Random-control gain ratio

Compare gain on real (image, q, a) triples vs gain on shuffled-within-batch
triples (cell C5).
```
gain_ratio = (no_input - heldout_nll_real) / max(no_input - heldout_nll_shuffled, ε)
```

**Pass:** gain_ratio ≥ 2.0. POC round-3 mit-B's ratio was ≈ 1.4 — insufficient.

## Steering probe

Mirror POC round-2 §14. For each sample, perturb h and measure Δnll:
- `zero_pos_p` for p ∈ {1, 2, 3}: zero out position p
- `permute_within`: shuffle h's K positions
- `permute_across`: replace h with another sample's h
- `gauss_noise_σ` for σ ∈ {0.1, 1.0}: add Gaussian noise

**Pass:** Δnll ≥ +1.5 nat on each of {zero_pos_3, permute_within, gauss_noise_1.0}.
Tests that h is genuinely position-specific and sample-specific (not placeholder).

## 5K visual-grounding stress test (cell C1 only)

Composition (per `inherited/EVAL_BENCHMARK_PLAN.md` §B):
- MMVP (300) + NaturalBench (1000) — adversarial paired Qs
- BLINK 7 perception subtasks (~900) — spatial, depth, counting, etc.
- MMStar (1500), CV-Bench 3D (~750), POPE-adversarial (500), VSR (500)

Greedy generation; per-task accuracy + overall.

## 4 control conditions (cell C1 only)

For the 5K stress test, replace input image with:
- C1: blank gray
- C2: random natural image (different sample)
- C3: adversarial mismatch (image contradicts question)
- C4: shuffled pixels

**Decision rule for "we have visual grounding":** claim only if
`Δ_C1(ours) > Δ_C1(vanilla) AND Δ_C3(ours) > Δ_C3(vanilla) AND Δ_C2 ≈ Δ_C1`.

**Pass:** blank-image (C1) accuracy drop ≥ 5 pp.

## Eval-suite invocation

```bash
# After training completes, on bioai:
sb slurm/eval.sbatch /data/joonhee/vl/checkpoints/<run_id>/
# OR locally on smaller subset:
uv run python -m vl.eval --checkpoint checkpoints/<run_id>/ --suite heldout
```

(`vl.eval` not yet implemented in v0.1.0 scaffold.)
