"""Build phase1_lvr/REPORT_SUMSSE.md — comparison of run-1 (mean-MSE) and run-2 (sum-MSE).

Inputs:
  - phase1_lvr/results/training_log.jsonl              (run 1)
  - phase1_lvr/results/ablation_results.jsonl          (run 1)
  - phase1_lvr/results/ablation_h_stats.jsonl          (run 1)
  - phase1_lvr/results_summse/training_log.jsonl       (run 2)
  - phase1_lvr/results_summse/ablation_results.jsonl   (run 2)
  - phase1_lvr/results_summse/ablation_h_stats.jsonl   (run 2)

Output:
  - phase1_lvr/REPORT_SUMSSE.md
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES1 = ROOT / "results"
RES2 = ROOT / "results_summse"


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _comp_ratio(rows_for_reader: dict) -> float | None:
    try:
        all_nll = rows_for_reader["all"]["nll_mean"]
        first = rows_for_reader["first_half"]["nll_mean"]
        last = rows_for_reader["last_half"]["nll_mean"]
        denom = first - all_nll
        if abs(denom) < 1e-9:
            return None
        return (last - all_nll) / denom
    except (KeyError, TypeError):
        return None


def _utility(rows_for_reader: dict) -> float | None:
    try:
        return rows_for_reader["none"]["nll_mean"] - rows_for_reader["all"]["nll_mean"]
    except (KeyError, TypeError):
        return None


def _n_helpful(rows_for_reader: dict, K=8, eps=0.05) -> int:
    try:
        none_nll = rows_for_reader["none"]["nll_mean"]
        n = 0
        for i in range(K):
            n_i = rows_for_reader[f"only_pos_{i}"]["nll_mean"]
            if not math.isnan(n_i) and (none_nll - n_i) >= eps:
                n += 1
        return n
    except (KeyError, TypeError):
        return 0


def _ascii_bar(margin: float, scale: float = 50.0) -> str:
    n = int(round(abs(margin) * scale))
    n = max(0, min(40, n))
    return "█" * n if margin >= 0 else "▒" * n


def _summarize(res_dir: Path):
    train = _read_jsonl(res_dir / "training_log.jsonl")
    abl = _read_jsonl(res_dir / "ablation_results.jsonl")
    h_stats = _read_jsonl(res_dir / "ablation_h_stats.jsonl")

    by_reader: dict[str, dict] = {}
    for r in abl:
        by_reader.setdefault(r["reader"], {})[r["mode"]] = r
    self_rows = by_reader.get("phase1_self", {})
    base_rows = by_reader.get("qwen_base", {})

    return {
        "train": train,
        "self_rows": self_rows,
        "base_rows": base_rows,
        "h_stats": h_stats,
        "cr_self": _comp_ratio(self_rows),
        "cr_base": _comp_ratio(base_rows),
        "util_self": _utility(self_rows),
        "util_base": _utility(base_rows),
        "nh_self": _n_helpful(self_rows),
        "nh_base": _n_helpful(base_rows),
        "mean_cos": h_stats[0]["avg_off_diag_cosine"] if h_stats else None,
        "cos_mat": h_stats[0]["pairwise_cosine_mean"] if h_stats else None,
        "per_pos_norm": h_stats[0]["per_pos_mean_norm"] if h_stats else None,
    }


def _step_table(train1, train2, every=100):
    """Side-by-side step tracking. Pick steps that are present in both logs."""
    steps_we_want = sorted(set([1] + list(range(every, 1001, every))))
    by_step1 = {r["step"]: r for r in train1}
    by_step2 = {r["step"]: r for r in train2}
    lines = ["| step |  run-1 ntp | run-1 lvr | run-2 ntp | run-2 lvr | run-1 ‖h‖ | run-2 ‖h‖ | run-1 ‖v‖ | run-2 ‖v‖ |"]
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in steps_we_want:
        r1 = by_step1.get(s)
        r2 = by_step2.get(s)
        if not r1 and not r2:
            continue
        def f(r, key, fmt="{:.3f}"):
            if r is None or key not in r:
                return "—"
            return fmt.format(r[key])
        lines.append(
            f"| {s} | {f(r1,'ntp_loss')} | {f(r1,'lvr_loss')} | "
            f"{f(r2,'ntp_loss')} | {f(r2,'lvr_loss')} | "
            f"{f(r1,'h_norm','{:.1f}')} | {f(r2,'h_norm','{:.1f}')} | "
            f"{f(r1,'v_norm','{:.1f}')} | {f(r2,'v_norm','{:.1f}')} |"
        )
    return "\n".join(lines)


def main():
    s1 = _summarize(RES1)
    s2 = _summarize(RES2)

    # Acceptance criteria (run-2 self-reader).
    crit = {
        "compression_ratio (≥0.4)": s2["cr_self"] is not None and s2["cr_self"] >= 0.4,
        "mean_off_diag_cos (≤0.55)": s2["mean_cos"] is not None and s2["mean_cos"] <= 0.55,
        "n_helpful (≥3)": s2["nh_self"] >= 3,
        "qwen_base utility (>0)": s2["util_base"] is not None and s2["util_base"] > 0,
    }
    n_pass = sum(crit.values())
    overall = "PASS" if n_pass >= 3 else ("MARGINAL" if n_pass == 2 else "FAIL")

    def fmt(x, default="n/a"):
        return f"{x:.3f}" if isinstance(x, (int, float)) else default

    # Final loss summary (run-2).
    if s2["train"]:
        final2 = s2["train"][-1]
        train2_summary = (
            f"final step={final2['step']} ntp={final2['ntp_loss']:.3f} "
            f"lvr={final2['lvr_loss']:.1f} ||h||={final2['h_norm']:.1f} "
            f"||v||={final2['v_norm']:.1f} elapsed={final2['elapsed_s']:.0f}s"
        )
    else:
        train2_summary = "(no run-2 training_log.jsonl)"

    # Per-position single-keep curve (run-2).
    curve_lines = []
    if "none" in s2["self_rows"]:
        none_nll = s2["self_rows"]["none"]["nll_mean"]
        for i in range(8):
            r = s2["self_rows"].get(f"only_pos_{i}")
            if r is None:
                continue
            nll = r["nll_mean"]
            margin = none_nll - nll
            tag = "(helpful)" if margin >= 0.05 else ("(harmful)" if margin <= -0.05 else "")
            curve_lines.append(f"  {i}    {nll:6.3f}   {margin:+.3f}  {_ascii_bar(margin)}  {tag}")
    curve_block = "\n".join(curve_lines) if curve_lines else "(unavailable)"

    # Cosine matrix (run-2).
    if s2["cos_mat"]:
        cos2 = s2["cos_mat"]
        cos_block = "       " + "  ".join(f"p{j}" for j in range(8)) + "\n"
        for i in range(8):
            cos_block += f" p{i}  " + "  ".join(f"{cos2[i][j]:4.2f}" for j in range(8)) + "\n"
    else:
        cos_block = "(unavailable)"

    # Loss-trajectory side-by-side (every 100 steps).
    step_tbl = _step_table(s1["train"], s2["train"], every=100)

    # 5-column comparison table.
    cmp = [
        "| metric | run-1 (mean-MSE λ=1) | run-2 (sum-MSE λ=1) | Monet stage 2 (target) | Monet stage 3 (anti-target) | user overnight 2026-05-04 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| compression_ratio | {fmt(s1['cr_self'])} | {fmt(s2['cr_self'])} | 2.7 | 0.78 | ~0.03 |",
        f"| mean off-diag cos | {fmt(s1['mean_cos'])} | {fmt(s2['mean_cos'])} | 0.38 | 0.85 | (>0.9) |",
        f"| n_helpful (≥3) | {s1['nh_self']} | {s2['nh_self']} | 4 | 1 | ≤1 |",
        f"| utility (qwen_base) | {fmt(s1['util_base'])} | {fmt(s2['util_base'])} | +2.7 | +0.26 | −0.22 to −1.10 |",
    ]

    # Acceptance call.
    pass_lines = []
    for k, v in crit.items():
        # Pull the matching value for display.
        if k.startswith("compression_ratio"):
            val = s2["cr_self"]
        elif k.startswith("mean_off_diag"):
            val = s2["mean_cos"]
        elif k.startswith("n_helpful"):
            val = s2["nh_self"]
        else:
            val = s2["util_base"]
        pass_lines.append(f"| {k} | {'PASS' if v else 'FAIL'} ({fmt(val) if not isinstance(val,int) else val}) |")

    # TL;DR.
    if n_pass >= 3:
        tldr_outcome = (
            "**Paper-faithful sum-over-D LVR loss at λ=1.0 broke the run-1 collapse.** "
            "Loss-magnitude was the issue; LVR mechanism works at scale-down. Proceed to Phase 2."
        )
        next_steps = (
            "Run 2 PASSES (≥3/4) → loss magnitude was the issue; LVR mechanism works at "
            "scale-down. Proceed to Phase 2 (Monet 3-stage with `build_4d_attn` mask) with "
            "confidence — paper-faithful loss formulation is the correct configuration going forward."
        )
    elif n_pass == 2:
        tldr_outcome = (
            "**Paper-faithful sum-MSE λ=1.0 did NOT clearly break the collapse.** "
            "Same MARGINAL outcome as run-1 — loss magnitude was not the dominant cause."
        )
        next_steps = (
            "Run 2 still MARGINAL (2/4) → loss magnitude alone wasn't sufficient. "
            "Causal attention is implicated as a structural cause; Phase 2 (Monet 3-stage with "
            "`build_4d_attn` mask) is the necessary test. Distributed-latent mechanism likely "
            "requires either bidirectional attention over the K slots or per-position targets."
        )
    else:
        tldr_outcome = (
            "**Paper-faithful sum-MSE λ=1.0 made things WORSE / failed.** "
            "Loss magnitude *itself* was not the issue; if anything, paper-default λ over-supervises."
        )
        next_steps = (
            "Run 2 FAILS (≤1/4) → paper-default λ at 2048× the run-1 effective magnitude is "
            "destructive at this scale. Causal attention + scale combined; Phase 2 should test "
            "the structural fix (Monet 3-stage masks). If the paper authors achieved distributed "
            "latents, they must have used either smaller λ or longer training to compensate."
        )

    body = f"""# Phase 1 — LVR Re-test with Paper-Faithful Sum-over-D Loss

## TL;DR

{tldr_outcome}

Run 2 trained Qwen2.5-VL-3B-Instruct with the **paper-faithful** LVR loss
`L = L_NTP + λ · (1/T_v)·Σ_t ||h_t − v_t||²₂` (sum over D, mean over K, mean over B)
at λ=1.0, K=8, otherwise identical to run-1 (same data, same seed, same 1000 steps,
same 5K examples, same 4-eff-bsz). On 200 held-out `eval` examples, run 2 scores
**{n_pass}/4** acceptance criteria — verdict **{overall}**.

Headline: `compression_ratio={fmt(s2['cr_self'])}` (target ≥ 0.4),
`mean_off_diag_cos={fmt(s2['mean_cos'])}` (target ≤ 0.55),
`n_helpful={s2['nh_self']}/8` (target ≥ 3),
frozen-Qwen utility `{fmt(s2['util_base'])}` (target > 0).

## 5-column comparison

{chr(10).join(cmp)}

(Phase-0 / overnight columns are reproduced from run-1 REPORT.md; the only new
columns are run-1 vs run-2.)

## 8×8 cosine matrix — run 2 (failure-mode signature)

```
{cos_block}
```

mean off-diagonal cos = **{fmt(s2['mean_cos'])}** (Phase 0 thresholds: <0.4 distributed ✓, >0.85 collapsed ✗)

## Per-position single-keep curve (run-2 phase1_self reader)

```
pos      nll   margin  bar
{curve_block}
```
(margin = none_NLL − only_pos_i_NLL; positive = helpful in isolation)

## Loss trajectory — side-by-side every 100 steps

(LVR magnitudes differ by ≈ D=2048 since run-2 is sum-over-D vs run-1 mean.)

{step_tbl}

## Recipe summary (run 2)

Identical to run-1 RECIPE.md except `loss_form: sum_d` (paper-faithful sum-over-D).
Vision tower + projector frozen; LLM full fine-tune. AdamW lr=1e-5, warmup 100,
cosine→0, eff bsz=4, bf16, gradient clipping max_grad_norm=1.0, 1000 steps on the
same 5K examples (seed=0). Final: {train2_summary}.

## Acceptance call (run 2)

| criterion | result |
|---|---|
{chr(10).join(pass_lines)}

**Overall: {overall} ({n_pass}/4 criteria met).**

## Implication for next steps

{next_steps}

## Caveats

- LVR magnitudes between the two runs are not directly comparable: run-2 LVR ≈
  D × run-1 LVR (D=2048). The trajectory table reports raw values; trajectory
  *shape* (the descent curve over steps) is the meaningful comparison.
- Effective gradient magnitude on the LVR head is ~2048× larger in run-2 at
  step 0, but AdamW's per-parameter normalization absorbs much of this; the
  meaningful comparison is the *direction* of optimization (heavily LVR-biased
  at start of run-2 vs balanced in run-1). With max_grad_norm=1.0 clipping,
  early-step total-grad norm is bounded.
- Identical seed and identical data ordering means step-by-step examples are the
  same; differences in trajectory are pure-loss-form effects.
- Both runs use mean-over-K within the LVR term — run-2's "sum-over-D" only
  changes the per-position contribution, not the across-position aggregation.
  This matches the paper's `(1/T_v) Σ_t ||h_t − v_t||²₂` interpretation literally.
"""
    (ROOT / "REPORT_SUMSSE.md").write_text(body)
    print(f"Wrote {ROOT/'REPORT_SUMSSE.md'}")
    print(f"\nVerdict: {overall} ({n_pass}/4 criteria met)")
    for k, v in crit.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")


if __name__ == "__main__":
    main()
