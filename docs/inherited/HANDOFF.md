# Handoff — Reader-Grounded Latent Visual Reasoning POC

This document tells the next session (chat or human) what the experiment knows, what's settled, and where to pick up.

## State at handoff (2026-05-02, round 3 complete)

- **Status:** POC 1+2+3 complete. **Round-3 design package complete and morning-ready.** Decision: do not start proliferated training without first running the round-3 POC gate (5-cell sweep on 4×H100 ≈ 24 h compute).
- **Active question:** "Does the combined round-3 recipe (NLL_multi(R=2,K_q=3) + 0.3·L_concept + 0.1·L_norm, LIVR-style + LoRA-r=32 + Stage-1 mask) pass all 5 binding criteria (held-out NLL ≤ 2.5, Monet-SFT transfer NLL ≤ 4.5, random-control gain ratio ≥ 2.0, steering ≥ +1.5 nat, blank-image accuracy drop ≥ 5 pp)?"
- **Decision rule for moving to proliferated training:** if all five round-3 criteria pass on cell C1 AND ablation cells C2/C3/C4 show C1 dominates AND C5 (random-control) underperforms C1, scale to M1 (100K samples, ~1300 H100-h). If any criterion fails, reformulate.
- **Where the briefings live:** `PROLIFERATED_PROJECT_PLAN.md` (top-level) is the morning master plan. `docs/ROUND3_POC_DESIGN.md` is the round-3 implementation spec. The 5 supporting docs (VLM_SURVEY, AUX_LOSS_AND_ARCH_DESIGN, VARIANT_B_GRPO_DESIGN, EVAL_BENCHMARK_PLAN, TRAINING_DATA_PLAN) under `docs/` carry the design rationale.

## What's settled (don't redo)

1. **Reachability is trivial.** Variant A's optimum is reachable in a few hundred Adam steps at K=1. The shortcut basin is everywhere in the loss landscape. Don't waste another sweep on this.
2. **Bare reader-NLL produces shortcut-dominant, off-manifold, reader-specific latents.** Confirmed by POC 1+2+3+4 with 95 % bootstrap CIs on n=100 samples. Document this finding; do not re-derive.
3. **The proposed novelty (no-image-on-reader + reader-NLL) has no exact prior in the literature** (per `docs/LITERATURE_RECON.md`); the closest neighbours are soft-prompt tuning and embedding-space adversarial attacks, both with documented pathologies — exactly the ones we observed.
4. **Citation: arXiv:2602.22766 is *not* a VLPO-specific critique** — it's a broader latent-visual-reasoning critique. Always cite the methodology (Input-Latent and Latent-Answer Disconnects) rather than VLPO collapse.
5. **Slot injection works.** The recipe in `tune.py` (hand-build prompt with K `<|image_pad|>` tokens, splice `h` into the embedding sequence at those positions, forward via `inputs_embeds=`) is the right primitive. Bypassing the vision encoder + processor entirely avoids M-RoPE concerns we initially worried about.
6. **GQA gold-token NLL caveat.** `oracle_nll ≈ no_input_nll` on many GQA samples is not a bug — GQA's narrow lexical answers vs the model's near-synonym preferences. Document; don't redebug.

## What to do next, in priority order

### Option 0 — already done in rounds 1–3

**Round 1.** POC 1–4: characterized failure modes. Reachability trivial, latents off-manifold, partial shortcut on held-out q', massive reader-transfer drop. Detail in REPORT.md §3–§7.

**Round 2.** Mit-B (norm reg λ=0.1, −39 % held-out), mit-A (rank-1, −32 %), N=5 multi-Q (−24 %), steering probe (latents are causally important, position-specific, sample-specific — not placeholder-like). **Round-2 reader-transfer addendum: none of these mitigations improve transfer to Monet-7B.** REPORT.md §11–§16.

**Round 3 (this session, 2026-05-02).** No new training; ran probes + built the design package.
- **Random-control for mit-B (`tune_random_control.py`).** With per-sample random target_norm ∈ U[10,200], held-out NLL = 2.85 vs natural target's 2.39 — 95 % CIs overlap heavily. Mit-B's −39 % is *mostly generic regularization*, only ~12 pp is target-aware. Per arXiv:2004.05704 caveat. REPORT.md §17.1.
- **Larger K sweep (K=32, 64).** Both reach final NLL ~0.0001 in 500 steps; per-token norm scales DOWN with K (K=4 → ~250, K=64 → ~19). Capacity is not a barrier to the shortcut. REPORT.md §17.2.
- **Cleaner cross-reader test (Monet-SFT-7B stage2).** All round-2 latents cluster at 8.0–8.6 nat under the SFT-only sibling, vs within-reader fits of 1.5–3.5 nat. Cleaner sibling shifts numbers by 0.2–0.3 nat but the gap remains huge. **Strengthens round-2's "geometric mitigations don't fix transfer" finding.** REPORT.md §17.3.
- **Design package built.** Five subagents produced `docs/VLM_SURVEY.md`, `docs/AUX_LOSS_AND_ARCH_DESIGN.md`, `docs/VARIANT_B_GRPO_DESIGN.md`, `docs/EVAL_BENCHMARK_PLAN.md`, `docs/TRAINING_DATA_PLAN.md`. Two synthesis subagents produced `docs/ROUND3_POC_DESIGN.md` (1055 lines) and `PROLIFERATED_PROJECT_PLAN.md` (646 lines).

**Critical findings from round 3 that must inform any training:**
1. Norm regularization is mostly generic regularization — not load-bearing for grounding. λ_norm at 0.1 stays as a stabilizer; L_concept (LaViT-style cosine to teacher) and multi-reader NLL must carry the visual-grounding signal.
2. Cross-reader transfer requires a *structural fix at training time* — multi-reader NLL (sum over R≥2 frozen readers). Per-sample geometric mitigations cannot fix it.
3. Capacity (K) is not the lever. Don't sweep K beyond 16 in round-3.
4. Random-control ablation is mandatory at every milestone of the proliferated project (M1, M2, M3).

**Locked design choices (12 items):** see `PROLIFERATED_PROJECT_PLAN.md` §1 or REPORT.md §17.5. Headlines: Qwen2.5-VL-7B-Instruct base; LIVR-style same-VLM + LoRA r=32 + Stage-1 attention mask; K=16; aux loss = NLL_multi(R=2, K_q=3) + 0.3·L_concept + 0.1·L_norm; cosine warmup 200 steps; Variant A first, Variant B (VLPO) only if A insufficient; multi-reader R=2 = Qwen2.5-VL-7B-Instruct + Monet-SFT-7B-stage2 (both on disk).

**Open decision points the user must answer (D1–D6):** see `PROLIFERATED_PROJECT_PLAN.md` §2.

### Option 0.1 — required round-3 POC (already designed; ready to implement)

`docs/ROUND3_POC_DESIGN.md` (1055 lines) is the implementation spec. Five hard pass thresholds:
1. Held-out NLL on q' ≤ 2.5 nat (cf POC1 vanilla 3.91, mit-B λ=0.1 2.39)
2. Reader-transfer NLL on Monet-SFT-7B ≤ 4.5 nat (cf round-2 mitigations 8.0–8.6 — this is the hardest target, requires multi-reader NLL to actually work)
3. Random-control gain ratio ≥ 2.0 (real gain ÷ shuffled gain)
4. Steering probe ≥ +1.5 nat on each of {zero_pos, permute_within, gauss_noise}
5. 5K visual-grounding stress test blank-image-control ≥ 5 pp accuracy drop

Five-cell sweep: C1 = full recipe; C2 = R=1 ablation; C3 = K_q=1 ablation; C4 = λ_concept=0 ablation; C5 = random-control ablation. Decision logic in `docs/ROUND3_POC_DESIGN.md` §6.

Files to write (7): `prepare_data_round3.py`, `model_round3.py`, `losses_round3.py`, `train_round3.py`, `evaluate_round3_heldout.py`, `evaluate_round3_transfer.py`, `analyze_round3.py`. Each with one-sentence purpose + reuse-from-existing references in `docs/ROUND3_POC_DESIGN.md` §8. Time: ~3 engineering days + ~24 h compute on 4×H100.

### Option 1 — implement and re-validate one mitigation (high priority)

Pick the cheapest mitigation likely to dent the failure mode. My ranking:

**(B) Distributional regularization** — quickest to add, directly aimed at the off-manifold finding (POC 4). Concrete recipe:

- Compute reader's natural visual-token statistics: see `compute_visual_baseline.py`. Already on disk: `results/visual_baseline.json` (n=100 GQA images, post-merger features from `model.model.visual(...).pooler_output`, hidden_dim 3584).
- Add a per-step loss term penalising `||h_i||_2`'s distance from the natural visual-token norm distribution (start with simple `(||h_i|| − norm_mean_mean)²` or a softer huber form).
- Optionally add a KL/MMD penalty on `h`'s per-dim distribution against the natural visual-token distribution. Probably overkill for a first try; just norm-matching first.
- Re-run `tune.py` with the modified loss. Measure POC 2 / POC 3 again. Look for: held-out `%h*≈no_input` drop, transfer-drop shrink.

If that closes the gap meaningfully, write up; otherwise stack (D) on top:

**(D) Multi-question consistency** — most directly aimed at POC 2's failure. Concrete recipe:

- Generate multi-question samples per image (GQA already supports this; `data/poc2_pairs.jsonl` has 30 such pairs and the loader generalises).
- Per training step, sample 2 questions for the same image, optimise `h` separately on each, add a consistency penalty `||h(q1, I) − h(q2, I)||²` (or contrastive against negatives from other images).
- Equivalent at training time: the *generator* receives image I and must produce a single `h(I)` regardless of q. Bake the q-invariance into the architecture (generator's input drops q for the latent-emission part; q only enters the reader's prompt, not the generator's). This is the strongest version.

This last formulation actually pulls the method *toward* Monet's recipe (the latent depends on the image but not on the question, and the reader uses the latent + question). May be worth structurally adopting.

### Option 2 — run additional probes the POC didn't cover (medium priority)

These are diagnostic, not training-justifying:

- **Steering probe (arXiv:2512.21711-style) on `h*`.** Replace specific positions in `h*` with zero / mean / random — measure how reader output changes per perturbation. Currently no script for this; ~50 LOC, GPU-only.
- **Generator-side Input-Latent Disconnect probe (arXiv:2602.22766).** Requires a *trained* generator. Defer until at least one mitigation is implemented.
- **Larger K sweep.** K ∈ {32, 64} to characterize the saturation curve. Modify the K_VALUES constant in `tune.py`.
- **Variant B trial.** Equivalent POC for the GRPO-RL formulation. Need a verifiable reward path; for GQA, exact-match on the rule-checked answer works. ~half a day to wire up.

### Option 3 — side-track (low priority)

- **Cleaner POC 3** with Monet-SFT-7B (NOVAglow646/Monet-SFT-7B on HF) instead of post-RL Monet-7B. Lower-bound on transfer drop would shrink slightly. Doesn't change conclusions; ~16 GB download for ~10 % cleaner result.
- **POC 2 with q1 == q2** sanity check: verify h* makes the reader confident on q1 (it should — POC 1 already showed this implicitly, but a no-greedy-decode confirmation costs nothing).

## What you should *not* do

- **Don't start training Variant A or B unmodified.** §8 of `REPORT.md` for why.
- **Don't switch to Qwen3-VL or Qwen3.6.** The whole cross-experiment story is anchored to Qwen2.5-VL-7B (Monet's base + sibling experiment + Monet-7B sibling reader). Switching means losing all three triangulations. Note flagged in `JOURNAL.md` 2026-05-01.
- **Don't re-derive what's settled.** See §"What's settled" above.
- **Don't trust raw `final_nll < oracle_nll` as a beat-the-image result.** It is a structural property of off-manifold soft prompts, not a real measurement of how informative `h*` is.

## How to run things

Everything from `experiments/reader-grounded-latent-poc/`:

```bash
# Re-run POC 1 (e.g., with different K_VALUES or modified loss)
uv run python tune.py

# Re-run POC 2 / POC 3 against a new POC 1 run dir
# (edit POC1_RUN_DIR constant at the top of each)
uv run python evaluate_held_out.py
uv run python evaluate_transfer.py

# Re-run analysis (does whatever sections have data)
uv run python analyze.py
# Output: results/ANALYSIS.md + results/<poc1_run>/figs/

# Recompute visual baseline (only needed if changing samples or pixel cap)
uv run python compute_visual_baseline.py
```

GPU: single A6000 (49 GB) handles all of these. POC 1 dominates time at ~135 min for 300 runs; everything else is single-digit minutes.

## Pinned files / data

- **POC 1 results:** `results/20260501-030544_poc1_full/` (300 rows, 300 latent .pt files)
- **POC 2 results:** `results/20260501-052224_poc2_held_out/` (90 rows)
- **POC 3 results:** `results/20260501-052251_poc3_transfer/` (300 rows)
- **Visual baseline:** `results/visual_baseline.json` (per-image + summary stats)
- **GQA images / data:** `data/gqa_images/*.jpg`, `data/poc1_samples.jsonl`, `data/poc2_pairs.jsonl`
- **Reader weights:** `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct` (cached; ~14 GB)
- **Sibling reader for POC 3:** `/mnt/ssd/Projects/research-pilots/experiments/monet-latent-probe/data/Monet-7B` (symlink to `/mnt/ssd/tmp/monet-spike/Monet-7B`)
- **Sibling experiment context:** `experiments/monet-latent-probe/` — H1 result on Monet's *actually-emitted* latents (causally load-bearing)

## Conventions enforced in this experiment

Per project CLAUDE.md:

- One file per stage. Constants at top. Hardcoded paths.
- Per-row JSONL append + skip-done-keys for crash resilience on every long-running script.
- `config.json` written at run start; results never overwrite (timestamped run dirs).
- Inline-copying of small helpers (`build_prompt_with_k_pads`, `forward_with_latents`) across scripts rather than premature shared module — prefer transparency.
- No argparse, no logging frameworks, no type annotations beyond the few needed for clarity.

## Open questions surfaced but not resolved

1. Does generator-mediated optimization (with autoregressive structure) actually mitigate the worst per-sample shortcuts? Per-sample direct optimization is the loose upper bound. **Round-3 will partially answer this** — the LIVR-style generator emits h via autoregressive forward, not direct gradient on `h ∈ R^{K×D}`.
2. ~~What's the lowest K at which the K=16 inter-sample cos sim appears?~~ **Resolved by round-3 K-sweep (REPORT §17.2):** capacity is not the lever; K=64 still reaches NLL=0. Per-token norm scales down with K (K=4 → ~250, K=64 → ~19); K is not a barrier.
3. ~~Is the K=16 transfer drop bounded above by the natural-visual-token-norm-distance?~~ **Refuted by round-3 random-control (REPORT §17.1):** norm-matching is not the mechanism; ~70 % of mit-B's gain is generic regularization. Norm regularization helps within-reader but not cross-reader.
4. Variant B's behaviour on the same POC protocol — **resolved structurally (REPORT §17, `docs/VARIANT_B_GRPO_DESIGN.md`):** vanilla GRPO doesn't update continuous latents; without VLPO Gaussian reparameterization (σ=5 + asymmetric KL β=0.04), Variant B reduces to "Variant A + stochastic answer sampling". Same shortcut basin reachable.
5. Will multi-reader NLL (sum over R≥2 frozen readers) actually fix cross-reader transfer? **Round-3 cell C1 vs C2 will test this directly.** This is the highest-priority unresolved question; it's the round-3 gate's pass/fail axis for transfer.
6. Will the random-control cell C5 reproduce C1's gains? If yes, the method is generic regularization — kill or reformulate. arXiv:2004.05704 caveat.
7. Does L_concept (LaViT-style cosine to teacher visual features) collapse to identity on simple datasets like CLEVR? Mitigation: bottleneck MLP D→D/2→D + cos-saturation monitor.
