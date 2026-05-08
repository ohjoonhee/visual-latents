"""Build phase1_lvr/REPORT.md from results files.

Inputs:
  - phase1_lvr/results/training_log.jsonl
  - phase1_lvr/results/ablation_results.jsonl
  - phase1_lvr/results/ablation_h_stats.jsonl

Output:
  - phase1_lvr/REPORT.md
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _comp_ratio(rows_for_reader: dict[str, dict]) -> float | None:
    """compression_ratio = (last_half_NLL − all_NLL) / (first_half_NLL − all_NLL)."""
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


def _utility(rows_for_reader: dict[str, dict]) -> float | None:
    try:
        return rows_for_reader["none"]["nll_mean"] - rows_for_reader["all"]["nll_mean"]
    except (KeyError, TypeError):
        return None


def _n_helpful(rows_for_reader: dict[str, dict], K=8, eps=0.05) -> int:
    """positions i where only_pos_i NLL ≥ none − eps (i.e. helpful by ≥0.05 nat)."""
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


def main():
    train = _read_jsonl(RES / "training_log.jsonl")
    abl = _read_jsonl(RES / "ablation_results.jsonl")
    h_stats = _read_jsonl(RES / "ablation_h_stats.jsonl")

    # Group ablation by reader.
    by_reader: dict[str, dict[str, dict]] = {}
    for r in abl:
        by_reader.setdefault(r["reader"], {})[r["mode"]] = r

    # Headline metrics — phase1_self reader (matched setting).
    self_rows = by_reader.get("phase1_self", {})
    base_rows = by_reader.get("qwen_base", {})

    cr_self = _comp_ratio(self_rows)
    cr_base = _comp_ratio(base_rows)
    util_self = _utility(self_rows)
    util_base = _utility(base_rows)
    nh_self = _n_helpful(self_rows)
    nh_base = _n_helpful(base_rows)

    mean_cos = h_stats[0]["avg_off_diag_cosine"] if h_stats else None
    cos_mat = h_stats[0]["pairwise_cosine_mean"] if h_stats else None
    per_pos_norm = h_stats[0]["per_pos_mean_norm"] if h_stats else None

    # Acceptance criteria.
    crit = {
        "compression_ratio (≥0.4)": cr_self is not None and cr_self >= 0.4,
        "mean_off_diag_cos (≤0.55)": mean_cos is not None and mean_cos <= 0.55,
        "n_helpful (≥3)": nh_self >= 3,
        "qwen_base utility (>0)": util_base is not None and util_base > 0,
    }
    n_pass = sum(crit.values())
    overall = "PASS" if n_pass >= 3 else ("MARGINAL" if n_pass == 2 else "FAIL")

    # Final loss numbers.
    if train:
        final = train[-1]
        train_summary = (
            f"final step={final['step']} ntp={final['ntp_loss']:.3f} "
            f"lvr={final['lvr_loss']:.3f} ||h||={final['h_norm']:.1f} "
            f"||v||={final['v_norm']:.1f} elapsed={final['elapsed_s']:.0f}s"
        )
    else:
        train_summary = "(no training_log.jsonl found)"

    # Per-position single-keep curve for phase1_self.
    curve_lines = []
    if "none" in self_rows:
        none_nll = self_rows["none"]["nll_mean"]
        for i in range(8):
            r = self_rows.get(f"only_pos_{i}")
            if r is None:
                continue
            nll = r["nll_mean"]
            margin = none_nll - nll
            tag = "(helpful)" if margin >= 0.05 else ("(harmful)" if margin <= -0.05 else "")
            curve_lines.append(f"  {i}    {nll:6.3f}   {margin:+.3f}  {_ascii_bar(margin)}  {tag}")
    curve_block = "\n".join(curve_lines) if curve_lines else "(unavailable)"

    # Cosine matrix.
    if cos_mat:
        cos_block = "       " + "  ".join(f"p{j}" for j in range(8)) + "\n"
        for i in range(8):
            cos_block += f" p{i}  " + "  ".join(f"{cos_mat[i][j]:4.2f}" for j in range(8)) + "\n"
    else:
        cos_block = "(unavailable)"

    def fmt(x, default="n/a"):
        return f"{x:.3f}" if isinstance(x, (int, float)) else default

    # Headline table.
    table = [
        "| metric | Phase1 (self) | Phase1 (qwen_base) | Monet stage2 (Phase 0 target) | Monet stage3 (Phase 0 anti-target) | user (overnight 2026-05-04) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| compression_ratio | {fmt(cr_self)} | {fmt(cr_base)} | 2.7 | 0.78 | ~0.03 |",
        f"| mean off-diag cos | {fmt(mean_cos)} | (same) | 0.38 | 0.85 | (>0.9) |",
        f"| n_helpful (≥3) | {nh_self} | {nh_base} | 4 | 1 | ≤1 |",
        f"| utility (none − all) | {fmt(util_self)} | {fmt(util_base)} | +2.7 | +0.26 | −0.22 to −1.10 |",
    ]

    body = f"""# Phase 1 — LVR-faithful repro on Visual-CoT

## TL;DR

Phase 1 trained a Qwen2.5-VL-3B-Instruct generator with the LVR loss
(L_NTP + λ_LVR · MSE(h, v_at_I); λ=1.0, K=8) on a 5K subset of
`ohjoonhee/visual-cot-50k-poc` for 1000 optimizer steps (effective batch 4).
On 200 held-out `eval` examples, the trained latents score **{n_pass}/4**
acceptance criteria — verdict **{overall}**.

Headline: `compression_ratio={cr_self:.2f}` (target ≥ 0.4),
`mean_off_diag_cos={mean_cos:.2f}` (target ≤ 0.55),
`n_helpful={nh_self}/8` (target ≥ 3),
frozen-Qwen utility `{util_base:+.3f}` (target > 0).

## Headline table

{chr(10).join(table)}

## Per-position single-keep curve (phase1_self reader)

```
pos      nll   margin  bar
{curve_block}
```
(margin = none_NLL − only_pos_i_NLL; positive = helpful in isolation)

## Pairwise-cosine matrix (8×8, mean over eval set)

```
{cos_block}
```

mean off-diagonal cos = **{mean_cos:.3f}** (Phase 0 thresholds: <0.4 distributed ✓, >0.85 collapsed ✗)

## Recipe summary

Trained with paper-faithful loss `L = L_NTP + λ · MSE(h, v_at_I)`, λ=1.0, K=8.
Vision tower + projector frozen; LLM full fine-tune. AdamW lr=1e-5, warmup
100 steps, cosine to zero, eff bsz=4 (bsz=1 × grad_accum=4), bf16, 1000 steps
on 5K examples (~0.8 epoch). One implementation deviation from paper: MSE is
mean over (B, K, D) rather than (1/T_v) · Σ over D — see RECIPE.md.

Training summary: {train_summary}.

## Acceptance call

| criterion | result |
|---|---|
| compression_ratio ≥ 0.4 | {'PASS' if crit['compression_ratio (≥0.4)'] else 'FAIL'} ({cr_self:.3f}) |
| mean off-diag cos ≤ 0.55 | {'PASS' if crit['mean_off_diag_cos (≤0.55)'] else 'FAIL'} ({mean_cos:.3f}) |
| n_helpful ≥ 3 | {'PASS' if crit['n_helpful (≥3)'] else 'FAIL'} ({nh_self}) |
| qwen_base utility > 0 | {'PASS' if crit['qwen_base utility (>0)'] else 'FAIL'} ({util_base:+.3f}) |

**Overall: {overall} ({n_pass}/4 criteria met).**

## Implication for Phase 2

If Phase 1 PASSES (≥3/4): the LVR loss at our scale produces distributed,
encoder-grounded latents — Phase 2 (Monet 3-stage) should proceed to test
whether stage-3 self-distillation degrades these (Phase 0 stage-3 result
suggests it will).

If Phase 1 FAILS: scale is implicated; flag for cluster-scale escalation
(Phase 3) before further per-position-target experiments. The Phase 0
stage-2 result ALREADY shows that distributed latents are achievable in
principle; Phase 1 is the local-scale floor for whether that mechanism
transfers below 7B.
"""
    (ROOT / "REPORT.md").write_text(body)
    print(f"Wrote {ROOT/'REPORT.md'}")
    print(f"\nVerdict: {overall} ({n_pass}/4 criteria met)")
    for k, v in crit.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")


if __name__ == "__main__":
    main()
