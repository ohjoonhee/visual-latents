"""Build pivot_a/REPORT_F.md — Experiment F (K=4 follow-up to D2 winning recipe).

Decoupled from build_report.py to avoid the K=8 hardcoding in that script.
Reads:
  - D2 (K=8) at pivot_a/results/vicreg_lambda2 — for comparison
  - E  (K=8) at pivot_a/results/vicreg_lambda2_2k — for comparison
  - F  (K=4) at pivot_a/results/vicreg_K4

Produces REPORT_F.md with:
  - 4-column comparison: D2 / E / F / Monet stage 2
  - Per-variant per-position single-keep curve (length = K of that variant)
  - Per-variant cosine matrix (size = K × K)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES_LAMBDA2 = ROOT / "results" / "vicreg_lambda2"
RES_E       = ROOT / "results" / "vicreg_lambda2_2k"
RES_F       = ROOT / "results" / "vicreg_K4"


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


def _n_helpful(rows, K, eps=0.05):
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


def _infer_K(self_rows, h_stats):
    """Infer K from data: number of `only_pos_i` modes, fallback to cosine matrix size."""
    K = 0
    for k in self_rows:
        if k.startswith("only_pos_"):
            K += 1
    if K > 0:
        return K
    if h_stats and h_stats[0].get("pairwise_cosine_mean"):
        return len(h_stats[0]["pairwise_cosine_mean"])
    return 8


def _load_variant(res_dir: Path, self_name: str, K_default: int):
    train = _read_jsonl(res_dir / "training_log.jsonl")
    abl = _read_jsonl(res_dir / "ablation_results.jsonl")
    h_stats = _read_jsonl(res_dir / "ablation_h_stats.jsonl")
    by_reader = {}
    for r in abl:
        by_reader.setdefault(r["reader"], {})[r["mode"]] = r
    self_rows = by_reader.get(self_name, {})
    base_rows = by_reader.get("qwen_base", {})

    K = _infer_K(self_rows, h_stats) if self_rows else K_default

    cr_self = _comp_ratio(self_rows)
    util_self = _utility(self_rows)
    util_base = _utility(base_rows)
    nh_self = _n_helpful(self_rows, K=K)
    mean_cos = h_stats[0]["avg_off_diag_cosine"] if h_stats else None
    cos_mat = h_stats[0]["pairwise_cosine_mean"] if h_stats else None
    v_off = h_stats[0].get("v_roi_avg_off_diag_cosine") if h_stats else None

    # Acceptance criteria. For K=4, n_helpful threshold is scaled to
    # ≥ ceil(K/2)=2 (50% of positions individually helpful).
    nh_thresh = max(2, (K + 1) // 2) if K <= 4 else 3
    crit = {
        "compression_ratio (≥0.4)": cr_self is not None and cr_self >= 0.4,
        "mean_off_diag_cos (≤0.55)": mean_cos is not None and mean_cos <= 0.55,
        f"n_helpful (≥{nh_thresh})": nh_self >= nh_thresh,
        "qwen_base utility (>0)": util_base is not None and util_base > 0,
    }
    n_pass = sum(crit.values())
    overall = "PASS" if n_pass >= 3 else ("MARGINAL" if n_pass == 2 else "FAIL")

    # Curves
    curve = []
    if "none" in self_rows:
        nn = self_rows["none"]["nll_mean"]
        for i in range(K):
            r = self_rows.get(f"only_pos_{i}")
            if r is None:
                continue
            v = r["nll_mean"]
            margin = nn - v
            tag = "(helpful)" if margin >= 0.05 else ("(harmful)" if margin <= -0.05 else "")
            curve.append(f"  {i}    {v:6.3f}   {margin:+.3f}  {_ascii_bar(margin)}  {tag}")

    # Cosine block (KxK)
    if cos_mat:
        cos_block = "       " + "  ".join(f"p{j}" for j in range(K)) + "\n"
        for i in range(K):
            cos_block += f" p{i}  " + "  ".join(f"{cos_mat[i][j]:4.2f}" for j in range(K)) + "\n"
    else:
        cos_block = "(unavailable)"

    if train:
        last = train[-1]
        train_summary = (
            f"final step={last['step']} ntp={last['ntp_loss']:.3f} lvr={last['lvr_loss']:.3f} "
            f"reg={last.get('reg_loss', float('nan')):.3f} ||h||={last['h_norm']:.1f} "
            f"||v||={last['v_norm']:.1f} elapsed={last['elapsed_s']:.0f}s"
        )
    else:
        train_summary = "(no training_log.jsonl found)"

    return {
        "self_name": self_name,
        "K": K,
        "cr_self": cr_self,
        "util_self": util_self,
        "util_base": util_base,
        "nh_self": nh_self,
        "nh_thresh": nh_thresh,
        "mean_cos": mean_cos,
        "cos_mat": cos_mat,
        "v_off": v_off,
        "crit": crit,
        "n_pass": n_pass,
        "overall": overall,
        "curve": "\n".join(curve) if curve else "(unavailable)",
        "cos_block": cos_block,
        "train_summary": train_summary,
        "has_data": bool(h_stats) or bool(abl),
    }


def main():
    lambda2 = _load_variant(RES_LAMBDA2, "pivot_a_vicreg_lambda2_self",     K_default=8)
    e_2k    = _load_variant(RES_E,       "pivot_a_vicreg_lambda2_2k_self",  K_default=8)
    f_k4    = _load_variant(RES_F,       "pivot_a_vicreg_K4_self",          K_default=4)

    table = [
        "| metric | D2 (K=8) | E (K=8 @ 2k) | **F (K=4)** | Monet stage 2 |",
        "|---|---:|---:|---:|---:|",
        f"| K | {lambda2['K']} | {e_2k['K']} | **{f_k4['K']}** | 8 |",
        f"| compression_ratio (≥ 0.4) | {fmt(lambda2['cr_self'])} | {fmt(e_2k['cr_self'])} | **{fmt(f_k4['cr_self'])}** | 2.7 |",
        f"| **mean off-diag cos (≤ 0.55)** | {fmt(lambda2['mean_cos'])} | {fmt(e_2k['mean_cos'])} | **{fmt(f_k4['mean_cos'])}** | 0.38 |",
        f"| n_helpful | {lambda2['nh_self']}/{lambda2['K']} | {e_2k['nh_self']}/{e_2k['K']} | **{f_k4['nh_self']}/{f_k4['K']}** | 4/8 |",
        f"| qwen_base utility (> 0) | {fmt(lambda2['util_base'])} | {fmt(e_2k['util_base'])} | **{fmt(f_k4['util_base'])}** | +2.7 |",
        f"| utility (self reader) | {fmt(lambda2['util_self'])} | {fmt(e_2k['util_self'])} | **{fmt(f_k4['util_self'])}** | +2.7 |",
        f"| v_roi off-diag cos | {fmt(lambda2['v_off'])} | {fmt(e_2k['v_off'])} | **{fmt(f_k4['v_off'])}** | n/a |",
    ]

    def acc_block(v, name):
        nh_key = f"n_helpful (≥{v['nh_thresh']})"
        return (
            f"### {name} — {v['overall']} ({v['n_pass']}/4)\n\n"
            f"| criterion | result |\n|---|---|\n"
            f"| compression_ratio ≥ 0.4 | {'PASS' if v['crit']['compression_ratio (≥0.4)'] else 'FAIL'} ({fmt(v['cr_self'])}) |\n"
            f"| mean off-diag cos ≤ 0.55 | {'PASS' if v['crit']['mean_off_diag_cos (≤0.55)'] else 'FAIL'} ({fmt(v['mean_cos'])}) |\n"
            f"| n_helpful ≥ {v['nh_thresh']} | {'PASS' if v['crit'][nh_key] else 'FAIL'} ({v['nh_self']}/{v['K']}) |\n"
            f"| qwen_base utility > 0 | {'PASS' if v['crit']['qwen_base utility (>0)'] else 'FAIL'} ({fmt(v['util_base'])}) |\n"
        )

    def variant_section(v, header):
        return f"""## {header} — per-position single-keep curve ({v['self_name']} reader)

```
pos      nll   margin  bar
{v['curve']}
```

## {header} — {v['K']}×{v['K']} cosine matrix (mean over eval set)

```
{v['cos_block']}
```

mean off-diagonal cos = **{fmt(v['mean_cos'])}**

Training summary: {v['train_summary']}
"""

    # Verdict logic
    f_geom_hit = (f_k4['mean_cos'] is not None and f_k4['mean_cos'] <= 0.55)
    f_helpful_ratio = (f_k4['nh_self'] / f_k4['K']) if f_k4['K'] else 0.0
    d2_helpful_ratio = (lambda2['nh_self'] / lambda2['K']) if lambda2['K'] else 0.0
    f_util_base = f_k4['util_base'] if f_k4['util_base'] is not None else 0.0
    d2_util_base = lambda2['util_base'] if lambda2['util_base'] is not None else 0.0

    if not f_k4['has_data']:
        verdict_line = "F (K=4): training/ablation not yet complete."
    elif f_geom_hit and f_helpful_ratio > d2_helpful_ratio and f_util_base >= 0.10:
        verdict_line = (
            f"F (K=4) **CLOSES the gap**: helpful ratio {f_helpful_ratio:.2f} > D2's {d2_helpful_ratio:.2f}, "
            f"qwen_base utility = {fmt(f_util_base)} ≥ +0.10. "
            f"K=8 was the binding constraint — recommend K=4 for cluster Phase 3."
        )
    elif f_geom_hit and abs(f_helpful_ratio - d2_helpful_ratio) < 0.1 and abs(f_util_base - d2_util_base) < 0.05:
        verdict_line = (
            f"F (K=4) **does NOT close the gap**: helpful ratio {f_helpful_ratio:.2f} ~ D2's {d2_helpful_ratio:.2f}, "
            f"utility {fmt(f_util_base)} ~ D2's {fmt(d2_util_base)}. "
            f"The gap is positional (positions 0 + last special) or structural to Qwen2.5-VL-3B at this budget."
        )
    elif not f_geom_hit:
        verdict_line = (
            f"F (K=4) **breaks geometry**: mean off-diag cos = {fmt(f_k4['mean_cos'])} > 0.55. "
            f"VICReg has fewer dims to spread across — K=4 has its own issues."
        )
    else:
        verdict_line = (
            f"F (K=4) **mixed**: helpful ratio {f_helpful_ratio:.2f} (D2: {d2_helpful_ratio:.2f}), "
            f"utility {fmt(f_util_base)} (D2: {fmt(d2_util_base)}), mean cos {fmt(f_k4['mean_cos'])}."
        )

    body = f"""# Pivot A — Experiment F (K=4 follow-up to D2 winning recipe)

## TL;DR

Experiment F is a single follow-up to Pivot A's winning recipe (D2 = VICReg
λ_reg=2.0, mean-MSE LVR). It tests whether reducing K=8 → K=4 closes the
semantic gap observed in C2/D2/E:
- mean off-diag cos ≤ 0.55 (target geometry achieved)
- but n_helpful = 2/8 (positions 0 and 7 only individually helpful)
- middle positions stay at margin 0.01-0.04, just below 0.05 threshold

Hypothesis: K=8 dilutes the per-position loss signal (5K examples × 8 slots
= 40K position-level updates). K=4 (5K × 4 = 20K) concentrates the signal,
potentially making more positions individually informative.

**Verdict**: {verdict_line}

## 4-column comparison

{chr(10).join(table)}

## Acceptance per variant

{acc_block(lambda2, "D2 — VICReg λ_reg=2.0, K=8 (mean-MSE LVR)")}

{acc_block(e_2k,    "E  — D2 recipe @ 2000 steps, K=8")}

{acc_block(f_k4,    "F  — D2 recipe @ K=4 (1000 steps, this experiment)")}

{variant_section(lambda2, "D2 (K=8)")}

{variant_section(e_2k,    "E (K=8 @ 2k)")}

{variant_section(f_k4,    "F (K=4)")}

## Recipe (F — K=4 follow-up)

- Base: Qwen/Qwen2.5-VL-3B-Instruct (vision tower + projector frozen, LLM full FT)
- Loss: `L = L_NTP + λ_LVR · LVR + λ_reg · L_reg`
  - LVR: `F.mse_loss(h, v_roi)` (mean-MSE, scaled by D)
  - L_reg: `vicreg_loss(h, var_weight=1.0, cov_weight=0.04, gamma=1.0)`
    on z-scored h (per-dim across BK sample axis)
- λ_reg: 2.0 (D2 winning value)
- Optimizer: AdamW lr=1e-5, weight_decay=0, betas=(0.9, 0.95)
- Schedule: cosine, warmup=100, decay→0
- Batch: micro=1, grad_accum=4 → eff=4
- Max steps: 1000
- bf16 + gradient_checkpointing
- Data: `ohjoonhee/visual-cot-50k-poc` train, 5000 examples, seed=0
- Eval: same dataset's `eval` split, first 200 examples (after seeded shuffle, seed=0)
- **K = 4** latent slots (was 8 in D2/E)
- Acceptance n_helpful threshold scaled to ≥ 2 (50% of K=4)

## Outcome interpretations

- **K=4 closes the gap**: n_helpful ≥ 2 in absolute terms (≥ 50% of K=4)
  AND qwen_base utility ≥ +0.10 → K=8 was binding, use K=4 for Phase 3.
- **K=4 doesn't move utility/n_helpful**: gap is positional or structural.
- **K=4 breaks geometry**: mean cos > 0.55 → VICReg loses purchase at K=4.
"""

    out_path = ROOT / "REPORT_F.md"
    out_path.write_text(body)
    print(f"Wrote {out_path}")
    print(f"\nF (K=4): {f_k4['overall']} ({f_k4['n_pass']}/4)  mean_cos={fmt(f_k4['mean_cos'])}  "
          f"n_helpful={f_k4['nh_self']}/{f_k4['K']}  util_base={fmt(f_k4['util_base'])}")
    print(f"D2 (K=8): {lambda2['overall']} ({lambda2['n_pass']}/4)  mean_cos={fmt(lambda2['mean_cos'])}  "
          f"n_helpful={lambda2['nh_self']}/{lambda2['K']}  util_base={fmt(lambda2['util_base'])}")
    print(f"E  (K=8): {e_2k['overall']} ({e_2k['n_pass']}/4)  mean_cos={fmt(e_2k['mean_cos'])}  "
          f"n_helpful={e_2k['nh_self']}/{e_2k['K']}  util_base={fmt(e_2k['util_base'])}")


if __name__ == "__main__":
    main()
