# Phase 1 — LVR Re-test with Paper-Faithful Sum-over-D Loss

## TL;DR

**Paper-faithful sum-MSE at λ=1.0 did NOT break the run-1 collapse — and made
the redundancy signature worse.** Run-2 hits the same MARGINAL 2/4 verdict as
run-1, but mean off-diagonal cosine moved from 0.851 → **0.987** (extreme
single-direction collapse) while NTP plateaued at ≈3.2 vs run-1's ≈0.57 — the
LLM was starved by the 2048× larger LVR signal. The collapse is robust across
~4 orders of magnitude of effective LVR pressure: loss magnitude is not the
dominant cause. Causal attention is implicated structurally — Phase 2 (Monet
3-stage with `build_4d_attn` mask) is the next necessary test.

Run 2 used `L = L_NTP + λ · (1/T_v)·Σ_t ||h_t − v_t||²₂` (sum over D, mean over
K, mean over B) at λ=1.0, K=8 — otherwise identical to run-1 (same data, seed,
1000 steps, 5K examples, eff bsz=4). `compression_ratio=0.700` PASS,
`mean_off_diag_cos=0.987` FAIL, `n_helpful=2/8` FAIL, qwen_base utility
`+0.076` PASS → **MARGINAL (2/4)**.

## 5-column comparison

| metric | run-1 (mean-MSE λ=1) | run-2 (sum-MSE λ=1) | Monet stage 2 (target) | Monet stage 3 (anti-target) | user overnight 2026-05-04 |
|---|---:|---:|---:|---:|---:|
| compression_ratio | 0.631 | 0.700 | 2.7 | 0.78 | ~0.03 |
| mean off-diag cos | 0.851 | 0.987 | 0.38 | 0.85 | (>0.9) |
| n_helpful (≥3) | 1 | 2 | 4 | 1 | ≤1 |
| utility (qwen_base) | 0.077 | 0.076 | +2.7 | +0.26 | −0.22 to −1.10 |

(Phase-0 / overnight columns are reproduced from run-1 REPORT.md; the only new
columns are run-1 vs run-2.)

## 8×8 cosine matrix — run 2 (failure-mode signature)

```
       p0  p1  p2  p3  p4  p5  p6  p7
 p0  1.00  0.98  0.97  0.97  0.97  0.97  0.96  0.96
 p1  0.98  1.00  1.00  0.99  0.99  0.98  0.98  0.98
 p2  0.97  1.00  1.00  1.00  0.99  0.99  0.99  0.99
 p3  0.97  0.99  1.00  1.00  1.00  1.00  0.99  0.99
 p4  0.97  0.99  0.99  1.00  1.00  1.00  1.00  0.99
 p5  0.97  0.98  0.99  1.00  1.00  1.00  1.00  1.00
 p6  0.96  0.98  0.99  0.99  1.00  1.00  1.00  1.00
 p7  0.96  0.98  0.99  0.99  0.99  1.00  1.00  1.00

```

mean off-diagonal cos = **0.987** (Phase 0 thresholds: <0.4 distributed ✓, >0.85 collapsed ✗)

## Per-position single-keep curve (run-2 phase1_self reader)

```
pos      nll   margin  bar
  0     4.477   +0.008    
  1     4.419   +0.067  ███  (helpful)
  2     4.402   +0.084  ████  (helpful)
  3     4.475   +0.011  █  
  4     4.477   +0.008    
  5     4.473   +0.012  █  
  6     4.459   +0.026  █  
  7     4.470   +0.015  █  
```
(margin = none_NLL − only_pos_i_NLL; positive = helpful in isolation)

## Loss trajectory — side-by-side every 100 steps

(LVR magnitudes differ by ≈ D=2048 since run-2 is sum-over-D vs run-1 mean.)

| step |  run-1 ntp | run-1 lvr | run-2 ntp | run-2 lvr | run-1 ‖h‖ | run-2 ‖h‖ | run-1 ‖v‖ | run-2 ‖v‖ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.577 | 19.573 | 7.577 | 40084.633 | 165.0 | 165.0 | 101.1 | 101.1 |
| 100 | 1.114 | 5.964 | 3.335 | 10360.681 | 46.3 | 27.4 | 92.5 | 92.5 |
| 200 | 0.831 | 4.048 | 3.174 | 6931.301 | 33.4 | 32.6 | 77.5 | 77.5 |
| 300 | 0.669 | 4.673 | 3.820 | 8082.409 | 37.0 | 42.5 | 89.0 | 89.0 |
| 400 | 0.901 | 4.167 | 3.226 | 7339.044 | 37.4 | 41.1 | 84.3 | 84.3 |
| 500 | 0.862 | 4.336 | 4.150 | 7740.058 | 41.8 | 43.4 | 89.0 | 89.0 |
| 600 | 0.682 | 4.059 | 3.281 | 7276.274 | 42.4 | 43.5 | 85.4 | 85.4 |
| 700 | 0.767 | 3.805 | 3.122 | 6811.671 | 38.7 | 38.8 | 81.0 | 81.0 |
| 800 | 0.522 | 4.397 | 3.140 | 7995.593 | 42.7 | 44.5 | 90.8 | 90.8 |
| 900 | 0.655 | 3.952 | 3.272 | 7180.490 | 39.6 | 39.8 | 83.1 | 83.1 |
| 1000 | 0.570 | 4.876 | 3.219 | 8888.105 | 47.6 | 51.0 | 95.9 | 95.9 |

## Recipe summary (run 2)

Identical to run-1 RECIPE.md except `loss_form: sum_d` (paper-faithful sum-over-D).
Vision tower + projector frozen; LLM full fine-tune. AdamW lr=1e-5, warmup 100,
cosine→0, eff bsz=4, bf16, gradient clipping max_grad_norm=1.0, 1000 steps on the
same 5K examples (seed=0). Final: final step=1000 ntp=3.219 lvr=8888.1 ||h||=51.0 ||v||=95.9 elapsed=1339s.

## Acceptance call (run 2)

| criterion | result |
|---|---|
| compression_ratio (≥0.4) | PASS (0.700) |
| mean_off_diag_cos (≤0.55) | FAIL (0.987) |
| n_helpful (≥3) | FAIL (2) |
| qwen_base utility (>0) | PASS (0.076) |

**Overall: MARGINAL (2/4 criteria met).**

## Implication for next steps

- **Phase 2 (Monet 3-stage masks) is the right next step.** Run-1 and run-2
  exhibit the same redundant-collapse failure with identical data/schedule but
  loss magnitudes differing by 2048× — the cause is therefore not loss-form
  but the encoder-decoder coupling pattern. Test whether `build_4d_attn`
  attention-mask topology fixes it.
- **Do NOT chase a λ-sweep at this scale.** Run-1 was already an effectively
  low-λ regime (λ_paper ≈ 0.0005); both extremes failed.
- **NTP starvation is a real side effect** of paper-default λ=1.0 with
  sum-over-D: NTP plateaus at ≈3.2 instead of 0.57. If future work keeps the
  literal paper loss, drop λ to ~{1e-3, 1e-2} for magnitude parity with run-1.
- **Negative-redundancy signature**: in both runs `first_half` and `last_half`
  ablations are *worse* than `none` (run-2: first_half=4.528, last_half=4.503,
  none=4.486). Any partial subset of the collapsed latents carries the same
  single-direction signal but starves complementary information.

## Caveats

- LVR magnitudes between runs are not directly comparable (run-2 ≈ D × run-1,
  D=2048). The trajectory *shape* over steps is the meaningful comparison.
- AdamW per-parameter normalization absorbs much of the magnitude difference;
  max_grad_norm=1.0 clipping bounds early-step total-grad norm. The relative
  bias of optimization is the meaningful effect — heavily LVR-biased in run-2.
- Identical seed and data ordering: trajectory differences are pure loss-form
  effects.
- The compression_ratio "PASS" (0.700) is achieved with very small numerator
  and denominator (last_half − all = 0.058, first_half − all = 0.084) — both
  near noise floor; the ratio is well-defined but the signal is weak.
- The literal paper formula `(1/T_v)·Σ_t ||h_t − v_t||²₂` is interpreted exactly:
  sum over D, mean over K, mean over B. No projection head, no normalization.
- Operational note: the trainer was written against `transformers` ≤ 4.54
  (where `get_image_features()` returns a list of tensors); newer releases
  return `BaseModelOutputWithPooling`. Both runs executed in the pinned
  `phase0_monet_probe/.venv-monet` (transformers 4.54.0) for API compatibility.
