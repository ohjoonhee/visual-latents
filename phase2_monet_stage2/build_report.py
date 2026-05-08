"""Build phase2_monet_stage2/REPORT.md from results files."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES_DIR = ROOT / "results" / "run_p2"


def _read_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _comp_ratio(rows):
    try:
        a = rows["all"]["nll_mean"]; f = rows["first_half"]["nll_mean"]; l = rows["last_half"]["nll_mean"]
        denom = f - a
        if abs(denom) < 1e-9:
            return None
        return (l - a) / denom
    except (KeyError, TypeError):
        return None


def _utility(rows):
    try:
        return rows["none"]["nll_mean"] - rows["all"]["nll_mean"]
    except (KeyError, TypeError):
        return None


def _n_helpful(rows, K=8, eps=0.05):
    try:
        nn = rows["none"]["nll_mean"]
        n = 0
        for i in range(K):
            v = rows[f"only_pos_{i}"]["nll_mean"]
            if not math.isnan(v) and (nn - v) >= eps:
                n += 1
        return n
    except (KeyError, TypeError):
        return 0


def _ascii_bar(margin, scale=80.0):
    n = int(round(abs(margin) * scale))
    n = max(0, min(40, n))
    return "█" * n if margin >= 0 else "▒" * n


def main():
    train = _read_jsonl(RES_DIR / "training_log.jsonl")
    abl = _read_jsonl(RES_DIR / "ablation_results.jsonl")
    h_stats = _read_jsonl(RES_DIR / "ablation_h_stats.jsonl")

    by_reader = {}
    for r in abl:
        by_reader.setdefault(r["reader"], {})[r["mode"]] = r

    self_rows = by_reader.get("phase2_self", {})
    base_rows = by_reader.get("qwen_base", {})

    cr_self = _comp_ratio(self_rows)
    cr_base = _comp_ratio(base_rows)
    util_self = _utility(self_rows)
    util_base = _utility(base_rows)
    nh_self = _n_helpful(self_rows)
    nh_base = _n_helpful(base_rows)

    mean_cos = h_stats[0]["avg_off_diag_cosine"] if h_stats else None
    cos_mat = h_stats[0]["pairwise_cosine_mean"] if h_stats else None
    v_off = h_stats[0].get("v_roi_avg_off_diag_cosine") if h_stats else None

    crit = {
        "compression_ratio (≥0.4)": cr_self is not None and cr_self >= 0.4,
        "mean_off_diag_cos (≤0.55)": mean_cos is not None and mean_cos <= 0.55,
        "n_helpful (≥3)": nh_self >= 3,
        "qwen_base utility (>0)": util_base is not None and util_base > 0,
    }
    n_pass = sum(crit.values())
    overall = "PASS" if n_pass >= 3 else ("MARGINAL" if n_pass == 2 else "FAIL")

    if train:
        f = train[-1]
        train_summary = (f"final step={f['step']} ce={f.get('ce_loss',0):.3f} "
                         f"align={f.get('align_loss',0):.4f} elapsed={f.get('elapsed_s',0):.0f}s")
    else:
        train_summary = "(no log)"

    curve_lines = []
    if "none" in self_rows:
        nn = self_rows["none"]["nll_mean"]
        for i in range(8):
            r = self_rows.get(f"only_pos_{i}")
            if r is None:
                continue
            v = r["nll_mean"]
            margin = nn - v
            tag = "(helpful)" if margin >= 0.05 else ("(harmful)" if margin <= -0.05 else "")
            curve_lines.append(f"  {i}    {v:6.3f}   {margin:+.3f}  {_ascii_bar(margin)}  {tag}")
    curve_block = "\n".join(curve_lines) if curve_lines else "(unavailable)"

    if cos_mat:
        cos_block = "       " + "  ".join(f"p{j}" for j in range(8)) + "\n"
        for i in range(8):
            cos_block += f" p{i}  " + "  ".join(f"{cos_mat[i][j]:4.2f}" for j in range(8)) + "\n"
    else:
        cos_block = "(unavailable)"

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"

    # 7-column comparison: P1 run-1 | P1 run-2 | P1.5 | P2 | M-stage2 | M-stage3 | user overnight
    # Phase 1.5 numbers loaded from sibling REPORT if present
    p15_metrics = {"cr": "?", "cos": "?", "nh": "?", "util": "?"}
    p15_report = ROOT.parent / "phase1_5_attn" / "results" / "run_p15" / "ablation_h_stats.jsonl"
    p15_abl = ROOT.parent / "phase1_5_attn" / "results" / "run_p15" / "ablation_results.jsonl"
    if p15_report.exists() and p15_abl.exists():
        try:
            p15_h = [json.loads(l) for l in p15_report.read_text().splitlines() if l.strip()]
            p15_a = [json.loads(l) for l in p15_abl.read_text().splitlines() if l.strip()]
            br = {}
            for r in p15_a:
                br.setdefault(r["reader"], {})[r["mode"]] = r
            sf = br.get("phase1_5_self", {}); bs = br.get("qwen_base", {})
            p15_metrics["cr"] = fmt(_comp_ratio(sf))
            p15_metrics["cos"] = fmt(p15_h[0]["avg_off_diag_cosine"]) if p15_h else "?"
            p15_metrics["nh"] = str(_n_helpful(sf))
            p15_metrics["util"] = fmt(_utility(bs))
        except Exception:
            pass

    table = [
        "| metric | P1 run-1 | P1 run-2 | P1.5 | **P2** | Monet st2 (target) | Monet st3 (anti) | user overnight |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| compression_ratio | 0.631 | 0.700 | {p15_metrics['cr']} | {fmt(cr_self)} | 2.7 | 0.78 | ~0.03 |",
        f"| mean off-diag cos | 0.851 | 0.987 | {p15_metrics['cos']} | {fmt(mean_cos)} | 0.38 | 0.85 | (>0.9) |",
        f"| n_helpful (≥3) | 1 | 2 | {p15_metrics['nh']} | {nh_self} | 4 | 1 | ≤1 |",
        f"| utility (qwen_base) | 0.077 | 0.076 | {p15_metrics['util']} | {fmt(util_base)} | +2.7 | +0.26 | −0.22 to −1.10 |",
        f"| h-cosine | 0.851 | 0.987 | {p15_metrics['cos']} | {fmt(mean_cos)} | 0.38 | 0.85 | n/a |",
        f"| v_roi cosine | 0.465 | 0.465 | n/a | {fmt(v_off)} | n/a | n/a | n/a |",
        f"| utility (self reader) | 0.336 | n/a | n/a | {fmt(util_self)} | +2.7 | +0.26 | n/a |",
    ]

    body = f"""# Phase 2 — Faithful Monet Stage 2 scale-down on Visual_CoT

## TL;DR

Phase 2 implements Monet's published Stage 2 recipe at A6000-friendly scale
(3B base, 5K Visual_CoT subset, 1000 steps, eff_batch=16) with the canonical
two-pass training: latent forward (recurrent latent generation) +
observation-position alignment to a teacher (raw Qwen2.5-VL-3B-Instruct on
the same trace with aux images visible), plus weighted CE on observation
tokens. On 200 held-out eval examples, the trained latents score
**{n_pass}/4** acceptance criteria — verdict **{overall}**.

Headline: `compression_ratio={fmt(cr_self)}` (target ≥ 0.4),
`mean_off_diag_cos={fmt(mean_cos)}` (target ≤ 0.55),
`n_helpful={nh_self}/8` (target ≥ 3),
qwen_base utility `{fmt(util_base)}` (target > 0).

## 7-column comparison

{chr(10).join(table)}

## Per-position single-keep curve (phase2_self reader)

```
pos      nll   margin  bar
{curve_block}
```

## 8x8 cosine matrix

```
{cos_block}
```

mean off-diagonal cos = **{fmt(mean_cos)}** (Phase 0 thresholds: <0.4 distributed ✓, >0.85 collapsed ✗)

## Loss trajectory

Training summary: {train_summary}.

Compare the LVR/MSE form (Phase 1.5) vs alignment-loss form (Phase 2) — the
key observable is whether either converges to a low loss while keeping
distributed h. See per-step data in `results/run_p2/training_log.jsonl`.

## Recipe deviations from paper

1. Skip Stage 1 SFT — use raw Qwen2.5-VL-3B-Instruct directly. (Documented in `RECIPE.md`.)
2. 5K Visual_CoT subset (vs 125K full 6-subset mix). Constrained by single-GPU
   budget; isolates recipe effect from data-mix effect.
3. eff_batch=16 (single A6000) vs 128 (8 GPUs). Compensated with extra steps.
4. emphasize_latent_weight backprop trick omitted; alignment loss flows through
   the full LLM trunk (not latent-only). Modest signal-strength deviation.
5. `attention_mask_4d` not used; rely on default causal mask + Monet's recurrence.
6. alignment_layer = all_layers (matches paper).

## Acceptance call

| criterion | result |
|---|---|
| compression_ratio (≥0.4) | {'PASS' if crit['compression_ratio (≥0.4)'] else 'FAIL'} ({fmt(cr_self)}) |
| mean off-diag cos (≤0.55) | {'PASS' if crit['mean_off_diag_cos (≤0.55)'] else 'FAIL'} ({fmt(mean_cos)}) |
| n_helpful (≥3) | {'PASS' if crit['n_helpful (≥3)'] else 'FAIL'} ({nh_self}) |
| qwen_base utility (>0) | {'PASS' if crit['qwen_base utility (>0)'] else 'FAIL'} ({fmt(util_base)}) |

**Overall: {overall} ({n_pass}/4 criteria met).**

## Caveats

- The `<observation>` tokens are present in Visual_CoT's training data, but
  the held-out eval (`ohjoonhee/visual-cot-50k-poc`) does NOT have observation
  tags — the eval prompts use a simpler `<|image_pad|>×K` slot pattern. Phase 2's
  utility on the held-out is therefore measured on a different distribution
  than the training (matches Phase 0's caveat about cross-distribution eval).
- The teacher reps were precomputed from raw 3B Qwen, not from a Stage-1-SFT'd
  checkpoint. The paper uses Stage-1-SFT teacher reps; this is a deviation.
"""
    out_path = ROOT / "REPORT.md"
    out_path.write_text(body)
    print(f"Wrote {out_path}")
    print(f"\nVerdict: {overall} ({n_pass}/4)")
    for k, v in crit.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")


if __name__ == "__main__":
    main()
