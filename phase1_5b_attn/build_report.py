"""Build phase1_5b_attn/REPORT.md from results files."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES_DIR = ROOT / "results" / "run_p15b"


def _read_jsonl(p):
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

    self_rows = by_reader.get("phase1_5b_self", {})
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
        train_summary = (f"final step={f['step']} ntp={f['ntp_loss']:.3f} lvr={f['lvr_loss']:.3f} "
                         f"||h||={f['h_norm']:.1f} ||v||={f['v_norm']:.1f} elapsed={f['elapsed_s']:.0f}s")
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

    def fmt(x, default="n/a"):
        return f"{x:.3f}" if isinstance(x, (int, float)) else default

    table = [
        "| metric | Phase 1 run-1 | Phase 1.5 (no mask) | **Phase 1.5b (with mask)** | Monet stage 2 target |",
        "|---|---:|---:|---:|---:|",
        f"| compression_ratio | 0.631 | 1.132 | {fmt(cr_self)} | 2.7 |",
        f"| mean off-diag cos | 0.851 | 0.959 | {fmt(mean_cos)} | 0.38 |",
        f"| n_helpful (≥3) | 1 | 1 | {nh_self} | 4 |",
        f"| utility (qwen_base) | 0.077 | 0.030 | {fmt(util_base)} | +2.7 |",
        f"| utility (self reader) | 0.336 | 0.160 | {fmt(util_self)} | +2.7 |",
        f"| v_roi off-diag cos | 0.465 | 0.465 | {fmt(v_off)} | n/a |",
    ]

    body = f"""# Phase 1.5b — Monet's vendored architecture + 4D attention mask injection

## TL;DR

Phase 1.5b adds the ONE missing variable from Phase 1.5: the
`attention_mask_4d` (cross-slot isolation: each latent slot t_i cannot
attend to other latent slots t_j, j != i, while still attending to all
non-slot causal context). All other variables — data, loss (mean-MSE
λ=1.0), seed, K=8, 1000 steps, eff_bsz=4 — are identical to Phase 1.5.

Verdict: **{overall} ({n_pass}/4 criteria met).**

Headline: `compression_ratio={fmt(cr_self)}` (target ≥ 0.4),
`mean_off_diag_cos={fmt(mean_cos)}` (target ≤ 0.55) — **the key metric**,
`n_helpful={nh_self}/8` (target ≥ 3),
frozen-Qwen utility `{fmt(util_base)}` (target > 0).

## 4-column comparison table

{chr(10).join(table)}

## Per-position single-keep curve (phase1_5b_self reader)

```
pos      nll   margin  bar
{curve_block}
```

## 8x8 cosine matrix

```
{cos_block}
```

mean off-diagonal cos = **{fmt(mean_cos)}** (Phase 0 thresholds: <0.4 distributed PASS, >0.85 collapsed FAIL)

## Recipe

Identical to Phase 1.5 (see `phase1_5_attn/RECIPE.md`) except:
- A 4D attention mask is constructed at every forward via
  `mask_utils.build_monet_4d_attn(input_ids, latent_token_id=...,
  latent_cross_isolate=True)` and passed as `attention_mask_4d=...` to
  BOTH the latent_mode=True and latent_mode=False forwards.

Mask semantics (per the user's hypothesis):
- causal (lower-triangular) base
- AND latent slot t_i cannot attend to latent slot t_j for j != i
- (slot can still attend to its own row and all non-slot context)

Source of `build_4d_attn`: re-implemented from the upstream Monet
`build_4d_attn_wo_helper_images` (https://github.com/NOVAglow646/Monet,
`src/utils.py:764`), constrained to Phase 1.5b's data layout (single
question image, no helper images, no observation blocks) and extended
with the `latent_cross_isolate=True` rule. See `mask_utils.py` for full
docstring + line refs into the vendored model file.

Caveat (documented in RECIPE.md): the per-latent-slot recurrent forward
inside `modeling_qwen2_5_vl_monet.py` (line ~1922-1934) hard-codes the 1D
`attention_mask` argument and does NOT consume `attention_mask_4d`. So
the cross-slot-isolation rule applies to (a) the pre-answer prefix
forward, (b) post-latent text-chunk forwards, and (c) the entire
latent_mode=False CE forward — but NOT to the per-step KV-cache pass
that produces each slot's hidden state directly. We did NOT modify the
vendored model file. This means cross-slot isolation is partially
enforced; if Phase 1.5b still fails on mean off-diag cos, the per-step
KV path (which still allows slot-i to attend to slot-j via cache) is the
likely remaining cause.

Loss: `L = L_NTP + 1.0 · F.mse_loss(h, v_roi)` (mean over 1×K×D). Vision
tower + projector frozen; LLM full FT. AdamW lr=1e-5, warmup=100,
cosine→0, eff bsz=4 (bsz=1 × grad_accum=4), bf16, 1000 steps, 5K
examples, seed=0.

Training summary: {train_summary}.

## Acceptance call

| criterion | result |
|---|---|
| compression_ratio (≥0.4) | {'PASS' if crit['compression_ratio (≥0.4)'] else 'FAIL'} ({fmt(cr_self)}) |
| mean off-diag cos (≤0.55) | {'PASS' if crit['mean_off_diag_cos (≤0.55)'] else 'FAIL'} ({fmt(mean_cos)}) |
| n_helpful (≥3) | {'PASS' if crit['n_helpful (≥3)'] else 'FAIL'} ({nh_self}) |
| qwen_base utility (>0) | {'PASS' if crit['qwen_base utility (>0)'] else 'FAIL'} ({fmt(util_base)}) |

**Overall: {overall} ({n_pass}/4 criteria met).**

## Implication

If Phase 1.5b PASSES on mean off-diag cos -> cross-slot attention
isolation is the load-bearing mechanism Phase 1.5 was missing.

If Phase 1.5b FAILS on mean off-diag cos with a value comparable to
Phase 1.5 (~0.96), the per-step KV-cache path (which the 4D mask cannot
reach without modifying the vendored model) is the prime suspect. A
follow-up could patch `modeling_qwen2_5_vl_monet.py` to slice
`attention_mask_4d` for the per-slot forward too.

## Caveats

- Same starting checkpoint as Phase 1.5 (raw Qwen2.5-VL-3B-Instruct).
- Same Monet special tokens added; embeddings randomly initialized via
  `resize_token_embeddings`.
- The per-slot KV-cache forward inside `latent_mode=True` still uses the
  1D causal mask (see RECIPE caveat above).
"""
    out_path = ROOT / "REPORT.md"
    out_path.write_text(body)
    print(f"Wrote {out_path}")
    print(f"\nVerdict: {overall} ({n_pass}/4)")
    for k, v in crit.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")


if __name__ == "__main__":
    main()
