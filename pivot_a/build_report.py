"""Build pivot_a/REPORT.md with the unified 5-column comparison.

Reads results from both pivot_a/results/cos and pivot_a/results/vicreg.
Produces the combined report with per-variant per-position curves and
8x8 cosine matrices.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES_COS = ROOT / "results" / "cos"
RES_VIC = ROOT / "results" / "vicreg"


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _comp_ratio(rows):
    try:
        a = rows["all"]["nll_mean"]
        f = rows["first_half"]["nll_mean"]
        l = rows["last_half"]["nll_mean"]
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


def _ascii_bar(margin: float, scale: float = 80.0) -> str:
    n = int(round(abs(margin) * scale))
    n = max(0, min(40, n))
    return "█" * n if margin >= 0 else "▒" * n


def fmt(x, default="n/a"):
    if isinstance(x, (int, float)):
        if isinstance(x, int):
            return str(x)
        return f"{x:.3f}"
    return default


def _load_variant(res_dir: Path, self_name: str):
    train = _read_jsonl(res_dir / "training_log.jsonl")
    abl = _read_jsonl(res_dir / "ablation_results.jsonl")
    h_stats = _read_jsonl(res_dir / "ablation_h_stats.jsonl")
    by_reader = {}
    for r in abl:
        by_reader.setdefault(r["reader"], {})[r["mode"]] = r
    self_rows = by_reader.get(self_name, {})
    base_rows = by_reader.get("qwen_base", {})

    cr_self = _comp_ratio(self_rows)
    util_self = _utility(self_rows)
    util_base = _utility(base_rows)
    nh_self = _n_helpful(self_rows)
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

    # Curves
    curve = []
    if "none" in self_rows:
        nn = self_rows["none"]["nll_mean"]
        for i in range(8):
            r = self_rows.get(f"only_pos_{i}")
            if r is None:
                continue
            v = r["nll_mean"]
            margin = nn - v
            tag = "(helpful)" if margin >= 0.05 else ("(harmful)" if margin <= -0.05 else "")
            curve.append(f"  {i}    {v:6.3f}   {margin:+.3f}  {_ascii_bar(margin)}  {tag}")

    # Cosine block
    if cos_mat:
        cos_block = "       " + "  ".join(f"p{j}" for j in range(8)) + "\n"
        for i in range(8):
            cos_block += f" p{i}  " + "  ".join(f"{cos_mat[i][j]:4.2f}" for j in range(8)) + "\n"
    else:
        cos_block = "(unavailable)"

    if train:
        f = train[-1]
        train_summary = (
            f"final step={f['step']} ntp={f['ntp_loss']:.3f} lvr={f['lvr_loss']:.3f} "
            f"reg={f.get('reg_loss', float('nan')):.3f} ||h||={f['h_norm']:.1f} "
            f"||v||={f['v_norm']:.1f} elapsed={f['elapsed_s']:.0f}s"
        )
    else:
        train_summary = "(no training_log.jsonl found)"

    return {
        "self_name": self_name,
        "cr_self": cr_self,
        "util_self": util_self,
        "util_base": util_base,
        "nh_self": nh_self,
        "mean_cos": mean_cos,
        "cos_mat": cos_mat,
        "v_off": v_off,
        "crit": crit,
        "n_pass": n_pass,
        "overall": overall,
        "curve": "\n".join(curve) if curve else "(unavailable)",
        "cos_block": cos_block,
        "train_summary": train_summary,
    }


def main():
    cos = _load_variant(RES_COS, "pivot_a_cos_self")
    vic = _load_variant(RES_VIC, "pivot_a_vicreg_self")

    # 5-column comparison (Phase 1 / Phase 1.5b numbers from prior reports).
    table = [
        "| metric | Phase 1 run-1 | Phase 1.5b | **C1 cos-penalty** | **C2 VICReg** | Monet stage 2 target |",
        "|---|---:|---:|---:|---:|---:|",
        f"| compression_ratio (≥ 0.4) | 0.631 | 1.132 | {fmt(cos['cr_self'])} | {fmt(vic['cr_self'])} | 2.7 |",
        f"| **mean off-diag cos (≤ 0.55)** | 0.851 | 0.961 | **{fmt(cos['mean_cos'])}** | **{fmt(vic['mean_cos'])}** | 0.38 |",
        f"| n_helpful (≥ 3) | 1 | n/a | {cos['nh_self']} | {vic['nh_self']} | 4 |",
        f"| qwen_base utility (> 0) | +0.077 | +0.030 | {fmt(cos['util_base'])} | {fmt(vic['util_base'])} | +2.7 |",
        f"| utility (self reader) | +0.336 | +0.160 | {fmt(cos['util_self'])} | {fmt(vic['util_self'])} | +2.7 |",
        f"| v_roi off-diag cos | 0.465 | 0.465 | {fmt(cos['v_off'])} | {fmt(vic['v_off'])} | n/a |",
    ]

    def acc_block(v, name):
        return (
            f"### {name} — {v['overall']} ({v['n_pass']}/4)\n\n"
            f"| criterion | result |\n|---|---|\n"
            f"| compression_ratio ≥ 0.4 | {'PASS' if v['crit']['compression_ratio (≥0.4)'] else 'FAIL'} ({fmt(v['cr_self'])}) |\n"
            f"| mean off-diag cos ≤ 0.55 | {'PASS' if v['crit']['mean_off_diag_cos (≤0.55)'] else 'FAIL'} ({fmt(v['mean_cos'])}) |\n"
            f"| n_helpful ≥ 3 | {'PASS' if v['crit']['n_helpful (≥3)'] else 'FAIL'} ({v['nh_self']}) |\n"
            f"| qwen_base utility > 0 | {'PASS' if v['crit']['qwen_base utility (>0)'] else 'FAIL'} ({fmt(v['util_base'])}) |\n"
        )

    # Verdict summary lines
    cos_geom_hit = (cos["mean_cos"] is not None and cos["mean_cos"] <= 0.55)
    vic_geom_hit = (vic["mean_cos"] is not None and vic["mean_cos"] <= 0.55)
    pivot_a_supported = cos_geom_hit or vic_geom_hit

    summary_lines = [
        f"- Verdict: **C1 cos-penalty** = {cos['overall']} ({cos['n_pass']}/4); **C2 VICReg** = {vic['overall']} ({vic['n_pass']}/4).",
        f"- Mean off-diag cos: C1 = {fmt(cos['mean_cos'])}; C2 = {fmt(vic['mean_cos'])} (target ≤ 0.55).",
        f"- C1 reached target geometry (≤ 0.55)? **{'YES' if cos_geom_hit else 'NO'}**.",
        f"- C2 reached target geometry (≤ 0.55)? **{'YES' if vic_geom_hit else 'NO'}**.",
        f"- Pivot A hypothesis (any variant ≤ 0.55) is **{'SUPPORTED' if pivot_a_supported else 'REFUTED'}** at 3B/5K.",
    ]

    body = f"""# Pivot A — LVR + collapse-prevention regularizer (Experiment C)

## TL;DR

Pivot A tests two regularizers added to the Phase 1 LVR loss to break
many-to-one collapse of the K=8 latent slots: **C1** pairwise cosine
hinge (τ=0.5) and **C2** VICReg variance + covariance on z-scored
hidden states. Each variant was trained separately, full FT of
Qwen2.5-VL-3B-Instruct on 5K Visual-CoT for 1000 steps (eff bsz=4,
lr=1e-5, mean-MSE LVR, λ_LVR=1.0, λ_reg=1.0, seed=0). All other
variables (data subset, K, schedule, prompt, ROI selection) are
identical to Phase 1's `lvr_3b_5k.yaml` — this is a controlled
comparison.

{chr(10).join(summary_lines)}

## 5-column comparison

{chr(10).join(table)}

## Acceptance per variant

{acc_block(cos, "C1 — pairwise cosine hinge (τ=0.5, λ_reg=1.0)")}

{acc_block(vic, "C2 — VICReg (var=1.0, cov=0.04, γ=1.0, λ_reg=1.0)")}

## C1 — per-position single-keep curve (pivot_a_cos_self reader)

```
pos      nll   margin  bar
{cos['curve']}
```

## C1 — 8×8 cosine matrix (mean over eval set)

```
{cos['cos_block']}
```

mean off-diagonal cos = **{fmt(cos['mean_cos'])}**

Training summary: {cos['train_summary']}

## C2 — per-position single-keep curve (pivot_a_vicreg_self reader)

```
pos      nll   margin  bar
{vic['curve']}
```

## C2 — 8×8 cosine matrix (mean over eval set)

```
{vic['cos_block']}
```

mean off-diagonal cos = **{fmt(vic['mean_cos'])}**

Training summary: {vic['train_summary']}

## Recipe

- Base: Qwen/Qwen2.5-VL-3B-Instruct (vision tower + projector frozen, LLM full FT)
- Loss: `L = L_NTP + 1.0 · F.mse_loss(h, v_roi) + 1.0 · L_reg` (mean-MSE form)
- L_reg:
  - **C1** = `cos_penalty_loss(h, tau=0.5)` — squared hinge on `(C[i,j] − τ).clamp(min=0)`
  - **C2** = `vicreg_loss(h, var_weight=1.0, cov_weight=0.04, gamma=1.0)`
    on z-scored h (per-dim across BK sample axis)
- Optimizer: AdamW lr=1e-5, weight_decay=0, betas=(0.9, 0.95)
- Schedule: cosine, warmup=100, decay→0
- Batch: micro=1, grad_accum=4 → eff=4
- Max steps: 1000
- bf16 + gradient_checkpointing
- Data: `ohjoonhee/visual-cot-50k-poc` train, 5000 examples, seed=0
- Eval: same dataset's `eval` split, first 200 examples (after seeded shuffle)
- K = 8 latent slots, prompt slot = `<|image_pad|>` × K (Phase-1-style)

## Decision

- mean cos ≤ 0.55 (any variant) → Pivot A's hypothesis confirmed at 3B/5K → escalate to cluster Phase 3 with this regularizer.
- both > 0.7 → Pivot A at 3B/5K does not work → fall back to Experiment B (full Monet attention topology) or escalate scale.
- one in [0.55, 0.7] → MARGINAL: improved geometry but not target; document and escalate.
"""

    out_path = ROOT / "REPORT.md"
    out_path.write_text(body)
    print(f"Wrote {out_path}")
    print(f"\nC1: {cos['overall']} ({cos['n_pass']}/4)  mean_cos={fmt(cos['mean_cos'])}")
    print(f"C2: {vic['overall']} ({vic['n_pass']}/4)  mean_cos={fmt(vic['mean_cos'])}")
    print(f"Pivot A supported: {pivot_a_supported}")


if __name__ == "__main__":
    main()
