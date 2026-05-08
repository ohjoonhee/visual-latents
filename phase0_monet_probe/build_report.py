"""Aggregate results.jsonl + h_stats.jsonl into REPORT.md.

Headline table per (stage, subset, reader):
  - compression_ratio = (last_half - all) / max(eps, first_half - all)
  - n_helpful_positions = count of i where (only_pos_i NLL <= none_NLL - 0.05)
  - mean_pairwise_cosine (same regardless of reader; from h_stats)
  - anchor utility = none_NLL - all_NLL (positive = latents help)

Plus a per-position bar chart for (Visual_CoT, stage3, monet_self) and a comparison line.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _key(row):
    return (row["stage"], row["subset"], row["reader"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default=str(ROOT / "results.jsonl"))
    ap.add_argument("--h_stats", type=str, default=str(ROOT / "h_stats.jsonl"))
    ap.add_argument("--out", type=str, default=str(ROOT / "REPORT.md"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.results)]
    by_cell: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_cell[_key(r)][r["mode"]] = r

    h_stats_rows = [json.loads(l) for l in open(args.h_stats)]
    h_stats_by_cell: dict[tuple, dict] = {}
    for r in h_stats_rows:
        h_stats_by_cell[(r["stage"], r["subset"])] = r

    # Headline table.
    headline_rows = []
    for (stage, subset, reader), modes in sorted(by_cell.items()):
        if "all" not in modes or "none" not in modes:
            continue
        all_nll = modes["all"]["nll_mean"]
        none_nll = modes["none"]["nll_mean"]
        first_half = modes.get("first_half", {}).get("nll_mean", float("nan"))
        last_half = modes.get("last_half", {}).get("nll_mean", float("nan"))
        first_only = modes.get("first_only", {}).get("nll_mean", float("nan"))
        last_only = modes.get("last_only", {}).get("nll_mean", float("nan"))

        denom = first_half - all_nll
        comp_ratio = (last_half - all_nll) / denom if abs(denom) > 1e-6 else float("nan")

        # n_helpful_positions: count i in [0..7] where (none - only_pos_i) >= 0.05 (positive utility margin)
        helpful = 0
        for i in range(8):
            v = modes.get(f"only_pos_{i}", {}).get("nll_mean", float("nan"))
            if v == v and (none_nll - v) >= 0.05:
                helpful += 1

        utility = none_nll - all_nll  # positive = latents help
        h_stats = h_stats_by_cell.get((stage, subset), {})
        mean_cos = h_stats.get("avg_off_diag_cosine", float("nan"))

        headline_rows.append({
            "stage": stage,
            "subset": subset,
            "reader": reader,
            "n": modes["all"].get("n", 0),
            "all_nll": all_nll,
            "none_nll": none_nll,
            "first_half_nll": first_half,
            "last_half_nll": last_half,
            "first_only_nll": first_only,
            "last_only_nll": last_only,
            "compression_ratio": comp_ratio,
            "n_helpful_positions": helpful,
            "anchor_utility": utility,
            "mean_pairwise_cosine": mean_cos,
        })

    # Per-position bar chart for (Visual_CoT, stage3, monet_self) — fall back to first available.
    target = None
    for r in headline_rows:
        if r["subset"] == "Visual_CoT" and r["stage"] == "stage3" and r["reader"] == "monet_self":
            target = r
            break
    if target is None and headline_rows:
        target = headline_rows[0]

    pos_chart = []
    if target is not None:
        modes = by_cell[(target["stage"], target["subset"], target["reader"])]
        none_nll = target["none_nll"]
        for i in range(8):
            v = modes.get(f"only_pos_{i}", {}).get("nll_mean", float("nan"))
            margin = (none_nll - v) if v == v else float("nan")
            pos_chart.append((i, v, margin))

    # Build REPORT.md.
    lines: list[str] = []
    lines.append("# Phase 0 — Monet latent compression probe")
    lines.append("")
    lines.append(f"Held-out probe of NOVAglow646/Monet-SFT-7B stages 2/3 on Visual_CoT and Zebra_CoT_count.")
    lines.append(f"Per-cell n: up to {target['n'] if target else '?'} held-out examples each. "
                 "Latent extraction: latent_mode=True, output_latent_embeds=True, alignment_poss = positions of "
                 "<abs_vis_token_pad>. K=8 latent positions per auxiliary image; we evaluate on the first block "
                 "(positions 0..7).")
    lines.append("")

    # ---------------- TL;DR ----------------
    lines.append("## TL;DR")
    if not headline_rows:
        lines.append("- No data: results.jsonl is empty; the extraction or ablation step did not produce rows.")
    else:
        # Pick representative cells.
        def fmt_ratio(x):
            return f"{x:.2f}" if x == x else "n/a"

        # Compute summary across (stage, subset) for monet_self
        self_rows = [r for r in headline_rows if r["reader"] == "monet_self"]
        qwen_rows = [r for r in headline_rows if r["reader"] == "qwen_base"]

        tl = []
        if self_rows:
            ratios = [r["compression_ratio"] for r in self_rows if r["compression_ratio"] == r["compression_ratio"]]
            if ratios:
                lo, hi = min(ratios), max(ratios)
                tl.append(
                    f"- **Compression ratio (monet_self, last_half vs first_half NLL)** ranges {lo:.2f}–{hi:.2f} across "
                    f"the {len(self_rows)} (stage × subset) cells. "
                    f"Compare to user's K_q=3 multi-Q result of 0.01–0.05 — values "
                    f"clearly above 0.5 here would indicate Monet's latents are NOT compressed; values near zero "
                    f"would indicate they exhibit the same per-position summary failure mode."
                )
        if qwen_rows and self_rows:
            self_util = sum(r["anchor_utility"] for r in self_rows) / max(1, len(self_rows))
            qwen_util = sum(r["anchor_utility"] for r in qwen_rows) / max(1, len(qwen_rows))
            tl.append(
                f"- **Transfer to frozen Qwen2.5-VL-7B-Instruct**: mean utility (none − all) is {qwen_util:+.2f} "
                f"(vs. {self_util:+.2f} self-reader). Positive = latents help an unadapted reader; negative = "
                f"latents move the prompt off the natural image-feature manifold."
            )
        n_help = [r["n_helpful_positions"] for r in self_rows]
        if n_help:
            tl.append(
                f"- **Helpful positions (NLL improvement ≥ 0.05 vs none, single-keep)**: per cell counts "
                f"{n_help}. The user's project sees ≤1 helpful position; values ≥4 here would falsify "
                f"the hypothesis that compression is universal."
            )
        if not tl:
            tl.append("- (Headline rows present but key fields missing.)")
        lines.extend(tl)
    lines.append("")

    # ---------------- Headline table ----------------
    lines.append("## Headline table")
    lines.append("")
    lines.append("| stage | subset | reader | n | all_NLL | none_NLL | first_half | last_half | comp_ratio | n_helpful | utility | cos |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in headline_rows:
        def fnum(x, d=3):
            return f"{x:.{d}f}" if (x == x) else "—"
        lines.append(
            f"| {r['stage']} | {r['subset']} | {r['reader']} | {r['n']} "
            f"| {fnum(r['all_nll'])} | {fnum(r['none_nll'])} "
            f"| {fnum(r['first_half_nll'])} | {fnum(r['last_half_nll'])} "
            f"| {fnum(r['compression_ratio'])} | {r['n_helpful_positions']} "
            f"| {fnum(r['anchor_utility'])} | {fnum(r['mean_pairwise_cosine'])} |"
        )
    lines.append("")
    lines.append("- `comp_ratio = (last_half_NLL − all_NLL) / (first_half_NLL − all_NLL)`; expected ≈1 for "
                 "uniform encoding, ≈0 for first-half-only, ≫1 for last-half-only.")
    lines.append("- `utility = none_NLL − all_NLL`; positive = latents help.")
    lines.append("- `n_helpful` = positions i ∈ {0..7} where `only_pos_i` NLL is ≥ 0.05 better than `none`.")
    lines.append("- `cos` = mean off-diagonal pairwise cosine across the 8 positions, averaged over the eval set.")
    lines.append("")

    # ---------------- Per-position curve ----------------
    if target is not None:
        lines.append(f"## Per-position single-keep curve — {target['subset']} × {target['stage']} × {target['reader']}")
        lines.append("")
        margins = [m for _, _, m in pos_chart if m == m]
        if margins:
            mx = max(abs(m) for m in margins) or 1.0
            lines.append("```")
            lines.append(f"{'pos':3} {'nll':>8} {'margin':>8}  bar")
            for i, v, m in pos_chart:
                if m != m:
                    bar = "—"
                else:
                    n = int(round(20 * (m / mx))) if mx > 0 else 0
                    if n >= 0:
                        bar = " " * 0 + "█" * n
                    else:
                        bar = "▒" * (-n) + " "
                lines.append(f"{i:3d} {v:8.3f} {m:+8.3f}  {bar}")
            lines.append("```")
            lines.append(f"`margin = none_NLL − only_pos_i_NLL`. Positive = position i has answer-relevant info on its own. "
                         f"For comparison: the user's project shows ~1 strongly positive position and ~7 near-zero.")
        lines.append("")

    # ---------------- Comparison ----------------
    lines.append("## Comparison to the user's own results")
    lines.append("")
    lines.append("`docs/overnight_2026_05_04/MORNING_REPORT.md` reports `compression_ratio` "
                 "0.01–0.05 on the user's K_q=3 multi-Q natural setup (multi-Q with matched-image pairing).")
    if headline_rows:
        self_rows = [r for r in headline_rows if r["reader"] == "monet_self"]
        if self_rows:
            ratios = [r["compression_ratio"] for r in self_rows if r["compression_ratio"] == r["compression_ratio"]]
            if ratios:
                gap_to_user = sum(ratios) / len(ratios)
                lines.append(f"- Monet's mean compression ratio (self-reader, across cells): **{gap_to_user:.2f}**.")
                lines.append(f"- User's mean compression ratio (multi-Q natural at 200 steps, anchor 0): ~0.03.")
                if gap_to_user > 0.5:
                    lines.append("- **Magnitude of gap: large.** Monet's latents do NOT show the user's per-position compression failure.")
                elif gap_to_user > 0.2:
                    lines.append("- **Magnitude of gap: moderate.** Monet's latents show partial distributed encoding; not as concentrated as the user's.")
                else:
                    lines.append("- **Magnitude of gap: small.** Monet's latents show a similarly-extreme compression pattern to the user's.")
    lines.append("")

    # ---------------- Decision call ----------------
    lines.append("## Decision call")
    lines.append("")
    lines.append("Recommendation logic (auto-generated from the matrix above):")
    if not headline_rows:
        lines.append("- Insufficient data — no recommendation.")
    else:
        self_rows = [r for r in headline_rows if r["reader"] == "monet_self"]
        ratios = [r["compression_ratio"] for r in self_rows if r["compression_ratio"] == r["compression_ratio"]]
        avg_ratio = sum(ratios) / max(1, len(ratios)) if ratios else float("nan")
        if avg_ratio == avg_ratio and avg_ratio > 0.5:
            lines.append(
                "- Monet does NOT exhibit the user's compression failure (avg ratio > 0.5). **Recommendation: "
                "proceed to Phase 1** — investigate WHAT Monet's training scheme adds (per-position alignment loss, "
                "teacher latents) that prevents the collapse, and adapt that intervention to the user's setup."
            )
        elif avg_ratio == avg_ratio and avg_ratio < 0.2:
            lines.append(
                "- Monet exhibits the SAME compression failure (avg ratio < 0.2). **Recommendation: pivot.** "
                "If even the published Monet checkpoints with their full alignment-loss curriculum collapse to "
                "summary tokens under our metric, the user's compression diagnosis is more general than "
                "reader-NLL gradient geometry alone — examine the metric (does it generalise?) before "
                "redesigning the training objective."
            )
        else:
            lines.append(
                "- Mixed signal. **Recommendation: rerun on a larger held-out sample (≥ 500 examples per cell) "
                "before deciding** — current n is too small to distinguish moderate from extreme compression."
            )
    lines.append("")

    # ---------------- Caveats ----------------
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Held-out set is only ~100 examples per (stage × subset) cell. Statistical power is limited.")
    lines.append("- Visual_CoT and Zebra_CoT_count are both Monet's *own* training data (held out via index-from-end). "
                 "They are not external benchmarks; the latents have been seen through SFT and are likely better-behaved "
                 "than they would be on truly out-of-distribution prompts.")
    lines.append("- We extract latents with `output_latent_embeds=True` from the *teacher latent* path "
                 "(latent_mode=True, alignment_poss = abs_vis_token_pad positions), which corresponds to Monet's "
                 "stage-3 teacher signal. This matches the canonical extraction precedent in upstream "
                 "`src/precompute_teacher_latents.py`.")
    lines.append("- The compression ablation uses Monet itself as a frozen reader (no LoRA, no further training). "
                 "This differs from the user's setup, where the reader is shared with the LoRA-adapted generator. "
                 "If Monet's latents need its trained reader+latent_mode=True path to be decoded, our metric is "
                 "blind to the encoded info — but so is the user's reader-NLL training signal in the same way, "
                 "so the comparison is fair.")
    lines.append("- The Qwen base-reader transfer test should be interpreted as a lower bound on cross-model transfer, "
                 "not as evidence of latent quality.")
    lines.append("")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
