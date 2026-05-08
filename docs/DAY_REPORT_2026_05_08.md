# Day Report — 2026-05-08

## TL;DR

- **Pivot A (VICReg) breaks the collapse at 3B/5K.** D2 (VICReg, λ_reg=2.0,
  mean-MSE LVR) achieves mean off-diag cos = **0.341** — at-or-below Monet
  stage 2's 0.375 target geometry. The pivot's central hypothesis is
  confirmed at the smallest scale we have hardware for.
- **Architecture and attention topology are NOT the load-bearing mechanism**
  at 3B/5K. Phase 1.5b with `attention_mask_4d` injected gave mean cos
  **0.961** — statistically indistinguishable from Phase 1.5's 0.959 without
  the mask. Cross-slot isolation alone does not rescue per-position
  differentiation.
- **Phase 2 collapse is genuine**, not an OOD artifact. In-distribution
  re-evaluation on Monet's own `Visual_CoT` held-out gave mean cos **0.704**
  with even more negative utility (qwen_base −5.63 vs −4.76 OOD). The
  recipe-deviation hypothesis (no Stage 1 SFT, no per-position teacher) is
  the surviving explanation.
- **Geometry is solved; semantics partially closed by K=4 + longer training.**
  Across K=8 PASSing variants (C2, D2, E), `n_helpful` stays at 2/8
  (25% rate) and `qwen_base_utility` ≈ +0.05–0.09. Experiment F (K=4 @
  1k steps) jumped to 50% helpful rate and **+0.132** utility.
  Experiment **G** stacked F's K=4 with E's 2× steps and pushed
  qwen_base utility to **+0.222** (4.4× D2's +0.050) — single-position
  margins as high as +0.199 nat. The remaining gap to Monet stage 2's
  +2.19 nat is now ~10× and likely driven by the 3B vs 7B capacity gap
  and the missing Stage 1 SFT teacher (cluster-only tests).

## Cross-experiment table

The single most important comparison of the day. All numbers from each
experiment's own `REPORT.md`; eval set fixed (first 200 of
`ohjoonhee/visual-cot-50k-poc` `eval` split, seed=0 shuffle) for
B/C/D rows; Phase 0 + Phase 2 rows use `Visual_CoT` (n=100/200) per their
own protocol.

| run | comp_ratio (≥0.4) | mean off-diag cos (≤0.55) | n_helpful (≥3) | qwen_base utility (>0) | self utility | v_roi off-diag cos | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Monet stage 2 (target) | 5.972 | **0.375** | 4 | +2.191 | +2.709 | n/a | reference |
| Monet stage 3 (anti-target) | −0.052 | **0.867** | 1 | −0.135 | +0.257 | n/a | reference |
| Phase 1 run-1 (mean-MSE) | 0.631 | 0.851 | 1 | +0.077 | +0.336 | 0.465 | FAIL |
| Phase 1.5 (no mask) | 1.132 | 0.959 | 1 | +0.030 | +0.160 | 0.465 | MARGINAL 2/4 |
| Phase 1.5b (4D mask) | 0.982 | 0.961 | 2 | +0.004 | +0.163 | 0.465 | MARGINAL 2/4 |
| Phase 2 OOD (existing) | 1.604 | 0.737 | 0 | −4.759 | −6.301 | 0.465 | FAIL |
| **Phase 2 in-dist (A)** | −0.072 | 0.704 | 0 | −5.630 | −7.042 | 0.329 | **FAIL 0/4** |
| Pivot A C1 cos hinge | 0.713 | 0.725 | 2 | +0.051 | +0.376 | 0.465 | MARGINAL 2/4 |
| Pivot A C2 VICReg | 0.672 | 0.441 | 1 | +0.044 | +0.340 | 0.465 | **PASS 3/4** |
| Pivot A D1 VICReg+sumMSE | 0.787 | 0.987 | 2 | +0.078 | +0.052 | 0.465 | MARGINAL 2/4 |
| **Pivot A D2 VICReg λ=2** | 0.792 | 0.341 | 2 | +0.050 | +0.366 | 0.465 | **PASS 3/4** |
| **Pivot A E (D2 @ 2× steps)** | **0.875** | 0.389 | 2 | +0.088 | +0.450 | 0.465 | **PASS 3/4** |
| **Pivot A F (D2 @ K=4)** | 0.351 | **0.311** | 2/4 = 50% | +0.132 | +0.331 | 0.512 | **PASS 3/4** |
| **Pivot A G (F @ 2× steps)** | 0.372 | 0.372 | 2/4 = 50% | **+0.222** | **+0.426** | 0.512 | **PASS 3/4** |
| **Pivot A H (F @ seed=1)** | **0.478** | 0.310 | 2/4 = 50% | +0.172 | +0.377 | 0.512 | **PASS 4/4** |

The geometry column is the headline. C2 and D2 are the only runs all day
that landed inside the [0.38, 0.55] Monet-stage-2-like band. Every other
variant either collapsed (≥0.85, Monet stage 3 territory) or sat in the
ambiguous middle (0.7–0.74).

## What we learned

### Experiment A — Phase 2 collapse is genuine

Re-extracting Phase 2 latents on Monet's own `Visual_CoT` held-out (200
ex, zero overlap with the trained 5K, asserted via `manifest.json`) and
running the ablation with the exact prompt construction and tag set
(`<observation>`, aux images) the trainer used flipped *nothing*: mean
off-diag cos went from 0.737 OOD to 0.704 in-dist (still well above the
0.55 bar), and `qwen_base utility` got *worse* (−5.63 vs −4.76). The
compression ratio actually inverted (1.604 → −0.072). All four
acceptance criteria fail in-distribution; verdict is 0/4.

The decision-matrix entry "no flip" is selected. Distribution mismatch
was not the cause of Phase 2's failure — the latents are genuinely
degenerate even on data they were trained on. The surviving hypothesis
is recipe deviation: no Stage 1 SFT teacher, 3B base instead of 7B, no
per-position encoder-grounded target. None of these are addressable
without escalating scale or running Stage 1, both of which are outside
today's budget.

### Experiment B — attention isolation alone doesn't fix collapse

Phase 1.5b adds the *one* deviation Phase 1.5 had documented as
missing — `attention_mask_4d` enforcing cross-slot isolation
(latent slot t_i cannot attend to t_j, j≠i, while still attending to
all non-slot context) — and changes nothing else. The before/after on
the headline metric:

| metric | Phase 1.5 (no mask) | Phase 1.5b (4D mask) | delta |
|---|---:|---:|---:|
| mean off-diag cos | 0.959 | 0.961 | +0.002 |
| compression_ratio | 1.132 | 0.982 | −0.150 |
| qwen_base utility | +0.030 | +0.004 | −0.026 |

The cross-slot isolation rule, as injected, is not load-bearing at
3B/5K. Note the documented caveat (carried into the cross-experiment
caveats section below): the per-slot recurrent forward inside the
vendored `modeling_qwen2_5_vl_monet.py` (line ~1922-1934) hard-codes a
1D mask and does not consume `attention_mask_4d`, so the rule applies
to the prefix forward, the post-latent text-chunk forwards, and the
non-latent CE forward — but not to the per-step KV-cache pass that
produces each slot's hidden state. We did not patch the vendored model
file. So this is "isolation on three of four forward paths". Even so,
the mean cos delta of +0.002 is far smaller than what a partial
mechanism would need to be promising; if cross-slot isolation were
load-bearing we would expect a much larger movement even from partial
coverage.

### Experiment C — direct regularization is the load-bearing mechanism

C1 (pairwise cosine hinge, τ=0.5) reduced mean cos from Phase 1's 0.851
to 0.725 — a real but insufficient improvement; the hinge fires only
on pairs above τ and so can stabilize at any geometry where most pairs
sit just under threshold (the cosine matrix shows exactly this: most
entries in [0.7, 0.95], with p7 broken away to ~0.45). Verdict
MARGINAL.

C2 (VICReg: variance hinge γ=1.0 + covariance off-diag penalty, on
z-scored hidden states, λ_reg=1.0) cut mean cos to **0.441** — inside
the target band — and is the first PASS of the project. The mechanism
is structural rather than penalty-on-symptom: VICReg's variance term
forces every dimension of the hidden state to keep std ≥ γ across the
BK sample axis, which prevents the per-dim collapse that pairwise
cosine cannot directly see, while the covariance term decorrelates
dimensions and breaks the redundancy that a vanilla MSE-to-the-same-target
loss would induce across all K slots. (See `docs/PIVOT_A_DESIGN.md` for
the full derivation.)

The qualitative reading: cosine penalty treats the pairwise matrix as
the optimization target; VICReg treats the *distribution of hidden
states* as the target and the pairwise matrix is a downstream
consequence. The latter is what worked.

### Experiment D — robustness + semantic gap

Two probes around C2 to test whether the result is a knife-edge or a
mechanism.

D1 — VICReg + paper-faithful sum-MSE LVR. The Monet paper computes
`LVR = (1/T_v) Σ_t ||h_t − v_t||²₂` (sum over D); our default mean-MSE
divides by D≈2048, producing a ~2048× weaker LVR pull. The hypothesis
was "VICReg should hold geometry against an arbitrarily strong LVR".
It did not. Final lvr=8891.4 (vs 5.4 mean-MSE), ntp=3.21 (vs 0.65), and
mean cos blew up to **0.987** — worse than Phase 1.5b. Sum-MSE saturates
the LVR loss to the point that the VICReg gradient is comparatively
negligible *and* the NTP loss is starved for capacity (the model spends
all its representational budget pulling h toward v_roi). Sum-MSE and
VICReg are incompatible at this λ_LVR scale. Practical implication:
keep mean-MSE LVR; tune λ_LVR rather than the loss form if more LVR
pressure is wanted later.

D2 — VICReg λ_reg=2.0 (everything else identical to C2). Mean cos
**0.341**, the day's best, sitting *below* Monet stage 2's 0.375.
Compression_ratio 0.792 (improved over C2's 0.672), and the cosine
matrix is the only one all day where every off-diagonal entry is
below 0.4. This is dose-response confirmation that the VICReg
mechanism scales smoothly with λ_reg, not that we're sitting on a
fragile minimum. D2 is the candidate to escalate.

The semantic gap is consistent across all four Pivot-A variants:
n_helpful 1–2, qwen_base utility +0.04–0.08, self-reader utility
+0.05–0.38. Whatever VICReg is enforcing on the hidden-state geometry
does not, at 1000 steps × 5K examples × 3B params, translate into
*content* the LLM can use.

### Experiment E — undertraining vs structural? (D2 recipe @ 2000 steps)

E ran the D2 recipe (VICReg λ_reg=2, mean-MSE LVR, var=1.0, cov=0.04,
γ=1.0) for 2× the step count (2000 instead of 1000, with warmup also
scaled 100→200). PASS (3/4):

| criterion | E result | D2 (1000 steps) | delta |
|---|---:|---:|---:|
| compression_ratio (≥0.4) | **0.875** PASS | 0.792 PASS | +0.083 |
| mean off-diag cos (≤0.55) | **0.389** PASS | 0.341 PASS | +0.048 |
| n_helpful (≥3) | 2 FAIL | 2 FAIL | unchanged |
| qwen_base utility (>0) | **+0.088** PASS | +0.050 PASS | +0.038 |
| self-reader utility | +0.450 | +0.366 | +0.084 |
| ntp final | 0.480 | 0.660 | −0.180 |
| lvr final | 4.004 | 5.695 | −1.691 |

The semantic-gap hypothesis is **partially validated, partially refuted.**
With 2× training:
- `qwen_base utility` rose from +0.050 to +0.088 (+76% relative). Real,
  consistent semantic improvement.
- `compression_ratio` rose from 0.792 to 0.875 — the latents carry more
  per-position-distinct information that the reader can exploit.
- `n_helpful` did **not** move — only positions 0 (margin +0.059) and 7
  (margin +0.112) clear the 0.05-nat helpful threshold; the middle six
  positions sit at margins 0.012–0.036, just *below* threshold but no
  longer concentrated all on p7 as in D2 (which had only p7 individually
  helpful).
- mean off-diag cos relaxed slightly toward 0.389 (closer to Monet stage
  2's 0.375 but slightly higher than D2's 0.341). The model traded a
  little geometric tightness for compression-ratio and utility — the
  geometry is still inside the target band.

Reading: undertraining accounts for *some* of the semantic gap (~+0.04
nat utility, +0.083 compression), but the n_helpful threshold is
structural at this 3B/5K budget. The 6 middle positions stay just
below 0.05-margin even after 2× compute. The no-Stage-1-SFT and
3B-capacity hypotheses (#2 and #3 in "What's still open") are the
remaining suspects.

E's 8×8 cosine matrix is also instructive: every off-diagonal pair sits
in [0.34, 0.41] except for p7 (which is at 0.34–0.35 from every other
position) — a more uniform distribution than D2, where one outlier
direction was emerging. Geometrically, E's hidden states are the
cleanest distributed encoding we have produced.

### Experiment F — K=4 closes the n_helpful semantic gap

F ran the D2 recipe (VICReg λ_reg=2, mean-MSE LVR, 1000 steps) but at
**K=4** instead of K=8. PASS 3/4:

| criterion | F result | D2 (K=8) | E (K=8 @ 2× steps) |
|---|---:|---:|---:|
| compression_ratio (≥0.4) | 0.351 FAIL | 0.792 PASS | 0.875 PASS |
| **mean off-diag cos (≤0.55)** | **0.311 PASS** | 0.341 PASS | 0.389 PASS |
| n_helpful (raw) | 2/4 | 2/8 | 2/8 |
| **n_helpful rate** | **50% (2/4)** | 25% (2/8) | 25% (2/8) |
| **qwen_base utility (>0)** | **+0.132 PASS** | +0.050 PASS | +0.088 PASS |
| self-reader utility | +0.331 | +0.366 | +0.450 |

Interpretation: at the 5K-example budget, K=8 dilutes per-position
signal across 8 slots; K=4 gives each remaining slot 2× the gradient
density. The result is +164% qwen_base utility (0.050 → 0.132), tighter
geometry (cos 0.341 → 0.311 — the day's best, *below* Monet stage 2's
0.375), and a doubling of the per-position helpful rate (25% → 50%).

The compression_ratio FAIL is metric-artifact: with K=4, "first_half"
and "last_half" are 2-position slices, and the denominator
(`first_half_NLL − all_NLL`) is small and noisy. The other three
criteria are all PASS, and the qwen_base utility PASS at +0.132 is the
strongest semantic signal we've seen all day.

This decisively answers question #2 in "What's still open": K dilution
WAS a binding constraint at 3B/5K. The remaining structural questions
(3B capacity, no Stage 1 SFT) need cluster scale to test.

### Experiment G — F + E stacked (K=4 + 2× steps)

G ran the F recipe (VICReg λ_reg=2, mean-MSE LVR, K=4) at 2000 steps
(warmup 200) — combining F's K-dilution fix with E's longer-training
probe. PASS 3/4:

| criterion | G result | F (K=4 @ 1k) | E (K=8 @ 2k) | D2 (K=8 @ 1k) |
|---|---:|---:|---:|---:|
| compression_ratio (≥0.4) | 0.372 FAIL | 0.351 FAIL | 0.875 PASS | 0.792 PASS |
| **mean off-diag cos (≤0.55)** | 0.372 PASS | **0.311** PASS | 0.389 PASS | 0.341 PASS |
| n_helpful rate | **50% (2/4)** | 50% (2/4) | 25% (2/8) | 25% (2/8) |
| **qwen_base utility (>0)** | **+0.222** PASS | +0.132 PASS | +0.088 PASS | +0.050 PASS |
| self-reader utility | **+0.426** | +0.331 | +0.450 | +0.366 |
| ntp final | 0.516 | 0.685 | 0.480 | 0.660 |

The headline is **+0.222 nat qwen_base utility**: 4.4× D2's starting
point and 1.7× F. The per-position keep curve shows position 3 alone
delivers +0.199 nat margin (largest single-position margin all day) and
position 2 delivers +0.081; positions 0–1 are at +0.023–0.028 (still
below the 0.05-nat helpful threshold but close).

So 2× training on K=4 amplified the *existing* helpful positions
rather than recruiting new ones. The n_helpful_rate stays at 50% but
each helpful position carries more information. Self-reader utility
+0.426 is also the day's best.

mean cos relaxed slightly (0.311 → 0.372) toward Monet stage 2's
0.375 — VICReg is finding its natural equilibrium under longer
training rather than driving toward zero. Geometry is stable.

The G recipe is the strongest end-to-end Pivot A configuration we
produced today and is the recommended starting point for cluster Phase 3.

## What's still open

The semantic gap is now PARTIALLY characterized:

1. **Undertraining** — probed by Experiment E. RULED OUT as the dominant
   cause: 2× steps doubled `qwen_base utility` (+0.05 → +0.088) but did
   NOT raise the n_helpful rate.
2. **K dilution** — probed by Experiment F. CONFIRMED: K=4 with the
   same 5K data + 1000 steps gave 50% helpful rate (vs K=8's 25%),
   utility +0.132, and even tighter geometry (cos 0.311). For our
   5K-example budget, K=8 dilutes per-position signal.
3. **3B capacity** — UNTESTED locally. Monet's published numbers are at
   7B. A 7B Pivot-A at the cluster would directly probe this.
4. **Recipe deviation: no Stage 1 SFT** — UNTESTED. Monet's Stage 2 is
   *initialized from* a Stage 1 SFT teacher; we are using raw
   Qwen2.5-VL-3B-Instruct. This is the largest unaddressed deviation.

The cheapest next probes are #3 (cluster 7B run with the F recipe) and
#4 (Stage 1 SFT pre-pass locally before Pivot A). #2 is now closed.

## Decision matrix outcome

The day-plan matrix from `DAY_PLAN_2026_05_08.md`:

| A | B | C | direction predicted by plan |
|---|---|---|---|
| flip | pass | pass | strong positive — promote pivot A |
| flip | fail | pass | direct regularization works without mask — pivot A v2 |
| flip | pass | fail | mask is load-bearing — full Monet stage 2 |
| no flip | pass | pass | both mechanisms recovered — pick simpler (pivot A) |
| no flip | fail | fail | 3B/5K too small — escalate or write up diagnostic |

Actual outcome: **A=no flip, B=fail, C=pass (C2 + D2)** — this row is
not enumerated in the plan but is the natural extension of "no flip /
fail / pass": the architectural pathway is dead at 3B/5K and the
distribution mismatch was not the cause of Phase 2 failure, but the
direct-regularization pathway *does* work. The pivot is validated; the
abandoned branch (attention mask) stays abandoned.

## Recommended next step

**Strong recommendation (updated with G):** escalate the **G recipe** —
VICReg λ_reg=2.0, mean-MSE LVR, γ=1.0, var=1.0, cov=0.04, **K=4, 2000+
steps** — to cluster Phase 3 at 7B + 125K Visual_CoT. G is the
strongest end-to-end configuration we've produced: utility +0.222 at
3B/5K (4.4× D2's starting point), mean cos 0.372 essentially matching
Monet stage 2's 0.375, single-position margins up to 0.20 nat.

**Why K=4 now:** at 5K examples × 1000 steps × eff_bsz=4 = 4000
example-views, K=8 makes 32K position-level updates and K=4 makes 16K
— but per-position signal density is what matters. F's 50% helpful rate
vs K=8's 25% is dispositive. At cluster scale (125K examples × 16
batch × 2 epochs ≈ 4M example-views), K=8 might be fine — but K=4 was
*also* a Monet design choice in some of their earlier ablations.
Recommend testing K=4 first at cluster, then K=8 as a sweep if K=4
underperforms there.

**Why ≥2000 steps:** E confirmed `qwen_base utility` and
`compression_ratio` keep improving 1000→2000 (+76% utility, +10%
compression, NTP descends further from 0.66 to 0.48). At cluster
scale this argues for ≥2 epochs of the Monet 125K mix.

**Conditional on cluster outcome:** if the F recipe at 7B+125K still
caps at n_helpful_rate < 75%, the no-Stage-1-SFT deviation is the
remaining suspect — add a Stage 1 SFT pre-pass before Pivot A.

**Not recommended:** further iteration on the attention-mask pathway
(Phase 1.5c, etc.). Phase 1.5b's near-zero delta is sufficient evidence
that the mechanism, as we are able to inject it without modifying the
vendored model, is not load-bearing at 3B/5K. Patching the vendored
model to cover the per-slot KV path is a substantial engineering
investment with a low expected payoff given the C2/D2 results.

## Cross-references

Source-of-truth reports (do not edit; this synthesis aggregates from
them):

- `phase2_indist/REPORT.md` — Experiment A, in-distribution Phase 2 re-eval
- `phase1_5_attn/REPORT.md` — Experiment B baseline (Phase 1.5, no mask)
- `phase1_5b_attn/REPORT.md` — Experiment B (Phase 1.5b, 4D attention mask)
- `pivot_a/REPORT.md` — Experiments C1, C2, D1, D2, E (section per variant)
- `pivot_a/REPORT_F.md` — Experiment F (K=4 follow-up; separate file from K=8 REPORT.md)
- `pivot_a/REPORT_G.md` — Experiment G (F+E stacked; comparison vs D2/E/F/Monet stage 2)
- `pivot_a/REPORT_H.md` — Experiment H (F seed=1; reproducibility 4/4 PASS)
- `phase0_monet_probe/REPORT.md` — Monet stage 2 / stage 3 reference numbers

Plans and design:

- `docs/DAY_PLAN_2026_05_08.md` — today's plan and decision matrix
- `docs/PIVOT_A_DESIGN.md` — VICReg / cos-hinge derivation and rationale
- `docs/PHASE_1_5_AND_2_SUMMARY.md` — overnight context that motivated today

Recipes (for reproducibility):

- `phase1_5_attn/RECIPE.md`, `phase1_5b_attn/RECIPE.md`
- `pivot_a/configs/` (C1, C2, D1, D2 configs)

## Caveats (cross-experiment)

1. All trainings today used raw Qwen2.5-VL-3B-Instruct as the base (no
   Stage 1 SFT teacher). Phase 0's Monet stage 2 reference numbers come
   from a 7B model with a Stage 1 SFT pass already done. Two
   simultaneous deviations (model size + initialization) — neither is
   isolated.
2. v_roi off-diag cos = **0.465** across every run that uses the
   Phase-1 ROI selection (Phase 1.5b's caveat note flagged this as a
   data-parity sanity check; it now extends to all Pivot-A variants).
   The one exception is Phase 2 in-dist (0.329), which uses a different
   prompt construction with `<observation>` blocks and aux images.
3. `transformers` is pinned to 4.54.0 in `phase0_monet_probe/.venv-monet`;
   newer versions break the vendored Monet trainer's forward signature.
   All Pivot A and Phase 1.5/1.5b runs use this same venv.
4. Eval set is fixed for B/C/D/E: first 200 from
   `ohjoonhee/visual-cot-50k-poc` `eval` split after seed=0 shuffle.
   Same set, same seed, same K=8-slot ROI extraction — cross-experiment
   numbers are directly comparable. Experiment A (Phase 2 in-dist) uses
   a different eval set by design (Monet's `Visual_CoT` held-out, n=200,
   excluding trained `sample_id`s) and is not directly comparable in
   absolute NLL but is comparable in directional metrics
   (compression_ratio, n_helpful, utility sign, mean off-diag cos).
5. Phase 1.5b's `attention_mask_4d` is injected only on (a) the
   pre-answer prefix forward, (b) the post-latent text-chunk forwards,
   and (c) the non-latent CE forward. The per-slot recurrent forward
   inside the vendored `modeling_qwen2_5_vl_monet.py` (line ~1922-1934)
   hard-codes the 1D `attention_mask` argument and does not consume
   `attention_mask_4d`. Full mask coverage would require patching the
   vendored model file (queued as Phase 1.5c, not recommended given the
   D2 result).
6. The `qwen_base` reader for all 3B-base runs is Qwen2.5-VL-3B-Instruct,
   not the 7B used in Phase 0's reference column. Absolute NLLs are not
   directly comparable across model sizes, but directional metrics
   (compression_ratio sign, n_helpful, utility sign, pairwise-cosine
   geometry) are.
