"""Build phase2_indist/REPORT.md from ablation results, comparing OOD vs in-dist."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASE2 = ROOT.parent / "phase2_monet_stage2"
PHASE0 = ROOT.parent / "phase0_monet_probe"


def _read_jsonl(p: Path):
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


def _by_reader_mode(rows):
    out = {}
    for r in rows:
        out.setdefault(r["reader"], {})[r["mode"]] = r
    return out


def fmt(x, d=3):
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    return f"{x:.{d}f}"


def main():
    # In-dist results
    indist_abl = _read_jsonl(ROOT / "ablation_results.jsonl")
    indist_h = _read_jsonl(ROOT / "ablation_h_stats.jsonl")
    indist_by = _by_reader_mode(indist_abl)
    self_rows = indist_by.get("phase2_self", {})
    base_rows = indist_by.get("qwen_base", {})

    cr_self = _comp_ratio(self_rows)
    cr_base = _comp_ratio(base_rows)
    util_self = _utility(self_rows)
    util_base = _utility(base_rows)
    nh_self = _n_helpful(self_rows)
    nh_base = _n_helpful(base_rows)
    n_examples = self_rows.get("all", {}).get("n", 0)

    mean_cos = indist_h[0]["avg_off_diag_cosine"] if indist_h else None
    cos_mat = indist_h[0]["pairwise_cosine_mean"] if indist_h else None
    v_off = indist_h[0].get("v_roi_avg_off_diag_cosine") if indist_h else None

    # OOD reference (existing Phase 2 results)
    ood_abl = _read_jsonl(PHASE2 / "results" / "run_p2" / "ablation_results.jsonl")
    ood_h = _read_jsonl(PHASE2 / "results" / "run_p2" / "ablation_h_stats.jsonl")
    ood_by = _by_reader_mode(ood_abl)
    ood_self = ood_by.get("phase2_self", {})
    ood_base = ood_by.get("qwen_base", {})
    ood_cr = _comp_ratio(ood_self)
    ood_cos = ood_h[0]["avg_off_diag_cosine"] if ood_h else None
    ood_nh = _n_helpful(ood_self)
    ood_util_base = _utility(ood_base)
    ood_util_self = _utility(ood_self)
    ood_v_off = ood_h[0].get("v_roi_avg_off_diag_cosine") if ood_h else None

    # Phase 0 stage 2 / stage 3 reference (Visual_CoT, qwen_base)
    p0_rows = _read_jsonl(PHASE0 / "results.jsonl")
    p0_h = _read_jsonl(PHASE0 / "h_stats.jsonl")
    def _p0(stage, reader, mode):
        for r in p0_rows:
            if r.get("stage")==stage and r.get("subset")=="Visual_CoT" and r.get("reader")==reader and r.get("mode")==mode:
                return r["nll_mean"]
        return None
    def _p0_cos(stage):
        for r in p0_h:
            if r.get("stage")==stage and r.get("subset")=="Visual_CoT":
                return r.get("avg_off_diag_cosine")
        return None
    s2_all = _p0("stage2","qwen_base","all"); s2_none = _p0("stage2","qwen_base","none")
    s2_first = _p0("stage2","qwen_base","first_half"); s2_last = _p0("stage2","qwen_base","last_half")
    s2_util = (s2_none - s2_all) if (s2_all and s2_none) else None
    s2_cr = ((s2_last - s2_all) / (s2_first - s2_all)) if (s2_all and s2_first and s2_last and abs(s2_first - s2_all) > 1e-9) else None
    s2_cos = _p0_cos("stage2")
    # n_helpful for stage 2 / qwen_base
    s2_modes = {r["mode"]:r for r in p0_rows if r.get("stage")=="stage2" and r.get("subset")=="Visual_CoT" and r.get("reader")=="qwen_base"}
    s2_nh = _n_helpful(s2_modes)

    s3_all = _p0("stage3","qwen_base","all"); s3_none = _p0("stage3","qwen_base","none")
    s3_first = _p0("stage3","qwen_base","first_half"); s3_last = _p0("stage3","qwen_base","last_half")
    s3_util = (s3_none - s3_all) if (s3_all and s3_none) else None
    s3_cr = ((s3_last - s3_all) / (s3_first - s3_all)) if (s3_all and s3_first and s3_last and abs(s3_first - s3_all) > 1e-9) else None
    s3_cos = _p0_cos("stage3")
    s3_modes = {r["mode"]:r for r in p0_rows if r.get("stage")=="stage3" and r.get("subset")=="Visual_CoT" and r.get("reader")=="qwen_base"}
    s3_nh = _n_helpful(s3_modes)

    # Acceptance
    crit = {
        "compression_ratio (≥0.4)": cr_self is not None and cr_self >= 0.4,
        "mean_off_diag_cos (≤0.55)": mean_cos is not None and mean_cos <= 0.55,
        "n_helpful (≥3)": nh_self >= 3,
        "qwen_base utility (>0)": util_base is not None and util_base > 0,
    }
    n_pass = sum(crit.values())
    overall = "PASS" if n_pass >= 3 else ("MARGINAL" if n_pass == 2 else "FAIL")

    # Flip determination
    cos_improved = (mean_cos is not None and ood_cos is not None and mean_cos < ood_cos)
    util_improved = (util_base is not None and ood_util_base is not None and util_base > ood_util_base)
    flip = cos_improved and util_improved
    flip_label = "FLIP (both cos↓ and utility↑)" if flip else "NO FLIP"

    # Curve (phase2_self, in-dist)
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

    # Headline 4-column comparison
    table = [
        "| metric | Phase 2 OOD (existing) | **Phase 2 in-dist (new)** | Monet st2 (target) | Monet st3 (anti-target) |",
        "|---|---:|---:|---:|---:|",
        f"| compression_ratio (self) | {fmt(ood_cr)} | **{fmt(cr_self)}** | {fmt(s2_cr)} | {fmt(s3_cr)} |",
        f"| mean off-diag cos | {fmt(ood_cos)} | **{fmt(mean_cos)}** | {fmt(s2_cos)} | {fmt(s3_cos)} |",
        f"| n_helpful (self) | {ood_nh} | **{nh_self}** | 4 | 1 |",
        f"| qwen_base utility | {fmt(ood_util_base)} | **{fmt(util_base)}** | {fmt(s2_util,2)} | {fmt(s3_util,2)} |",
        f"| self-reader utility | {fmt(ood_util_self)} | {fmt(util_self)} | +2.71 | +0.26 |",
        f"| v_roi cosine | {fmt(ood_v_off)} | {fmt(v_off)} | n/a | n/a |",
    ]

    # Implication block
    impl = []
    if flip:
        impl.append(
            "Both `mean off-diag cosine` and `qwen_base utility` improved when moving from"
            " the OOD eval (visual-cot-50k-poc) to the in-distribution Visual_CoT held-out."
            " The OOD-vs-in-dist gap was a major confounder of the Phase 2 OOD failure call;"
            " on the same distribution the model was trained on, latents behave better."
        )
        if mean_cos is not None and mean_cos > 0.55:
            impl.append(
                "However, in-dist mean cosine is still above the 0.55 distributed-vs-collapsed"
                " threshold — distribution mismatch only partially explains the OOD result;"
                " a residual collapse signature remains."
            )
        if util_base is not None and util_base > 0:
            impl.append(
                f"qwen_base utility flipped from harmful ({fmt(ood_util_base)}) to neutral/helpful"
                f" ({fmt(util_base)}), so the latents do encode useful signal — they were"
                " specifically harmful only on the held-out eval distribution."
            )
    else:
        impl.append(
            "The cosine and utility numbers did NOT both improve in-distribution. The Phase 2"
            " collapse looks genuine, not just a distribution-mismatch artifact: "
            "the latents are degenerate even on the data they were trained on."
        )
        if cr_self is not None and ood_cr is not None and cr_self > ood_cr:
            impl.append(
                f"compression_ratio rose from {fmt(ood_cr)} to {fmt(cr_self)}, which by itself"
                " is a directional positive — but without a corresponding cosine drop and"
                " utility flip, the directional metric is being driven by a uniform-redundancy"
                " regime (per Phase 0's stage-3 caveat), not by per-position specialisation."
            )

    body = f"""# Phase 2 — In-distribution re-extraction on Monet's Visual_CoT

## TL;DR

Re-extracted latents from the Phase 2 checkpoint
(`phase2_monet_stage2/results/run_p2/checkpoint`) on Monet's OWN Visual_CoT
held-out — same distribution Phase 0 used for stages 2/3 probes, with
`<observation>` tags + auxiliary images and the EXACT prompt construction
the Phase 2 trainer used. Held-out is 200 examples sampled (seed=42)
from `raw[-10000:]` of `Visual_CoT/train.json` MINUS the 5K trained sample_ids
recorded in `phase2_monet_stage2/teacher_reps/manifest.json` (zero overlap, asserted).

On 200 held-out in-distribution examples, the trained latents score
**{n_pass}/4** acceptance criteria — verdict **{overall}** ({flip_label}).

Headline: `compression_ratio={fmt(cr_self)}` (target ≥ 0.4),
`mean_off_diag_cos={fmt(mean_cos)}` (target ≤ 0.55),
`n_helpful={nh_self}/8` (target ≥ 3),
qwen_base utility `{fmt(util_base)}` (target > 0).

## Headline table — Phase 2 OOD vs in-dist vs Monet stage 2/3 (Visual_CoT, qwen_base)

{chr(10).join(table)}

Note: Monet stage 2 / stage 3 numbers are from `phase0_monet_probe/REPORT.md`,
qwen_base reader, Visual_CoT subset, n=100. Phase 2 uses qwen_base = Qwen2.5-VL-**3B**-Instruct
(matching the Phase 2 base) whereas Phase 0 used the 7B; absolute NLLs not directly
comparable, but the directional signs and pairwise-cosine geometry are.

## Per-position single-keep curve (phase2_self reader, in-dist)

```
pos      nll   margin  bar
{curve_block}
```

`margin = none_NLL − only_pos_i_NLL`. Positive = position helpful in isolation.

## 8x8 cosine matrix (in-dist)

```
{cos_block}
```

mean off-diagonal cos = **{fmt(mean_cos)}** (Phase 0 thresholds: <0.4 distributed ✓, >0.85 collapsed ✗)

OOD reference: mean off-diag cos = {fmt(ood_cos)} (same checkpoint, different eval).

## Acceptance call

| criterion | result |
|---|---|
| compression_ratio (≥0.4) | {'PASS' if crit['compression_ratio (≥0.4)'] else 'FAIL'} ({fmt(cr_self)}) |
| mean off-diag cos (≤0.55) | {'PASS' if crit['mean_off_diag_cos (≤0.55)'] else 'FAIL'} ({fmt(mean_cos)}) |
| n_helpful (≥3) | {'PASS' if crit['n_helpful (≥3)'] else 'FAIL'} ({nh_self}) |
| qwen_base utility (>0) | {'PASS' if crit['qwen_base utility (>0)'] else 'FAIL'} ({fmt(util_base)}) |

**Overall: {overall} ({n_pass}/4 criteria met).**

## Implication

{' '.join(impl)}

## Caveats

- n = {n_examples} held-out examples (target 200; reduced if extraction failures).
- The qwen_base reader here is Qwen2.5-VL-3B-Instruct (Phase 2's base), not the
  7B used in Phase 0's reference numbers. Absolute NLLs differ by model size,
  but directional metrics (compression_ratio, n_helpful, utility sign) are valid.
- Despite excluding the 5K trained sample_ids, the held-out set is drawn from
  the same `raw[-10000:]` pool the trainer sampled from. Distribution match is
  thus tight; this is the desired property for the in-dist test, but it means
  the held-out is NOT independent of training in a generalisation sense.
- Per-position alignment positions are the K=8 `<abs_vis_token_pad>` slots
  AFTER the first `<|im_start|>assistant` marker — same rule as Phase 0
  extract and the Monet upstream `precompute_teacher_latents` path.
- Some Visual_CoT traces have multiple auxiliary images → `num_latents > 8`;
  the ablation reads only the FIRST K=8 positions (matching Phase 2 OOD and
  the canonical ablation's `latent[:K]` convention).
"""
    out_path = ROOT / "REPORT.md"
    out_path.write_text(body)
    print(f"Wrote {out_path}")
    print(f"\nVerdict: {overall} ({n_pass}/4)")
    for k, v in crit.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\nFlip: {flip_label}")
    print(f"\nNumbers: cr={fmt(cr_self)}  cos={fmt(mean_cos)}  nh={nh_self}  util_base={fmt(util_base)}")


if __name__ == "__main__":
    main()
