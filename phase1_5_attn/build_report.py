"""Build phase1_5_attn/REPORT.md from results files."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES_DIR = ROOT / "results" / "run_p15"


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

    self_rows = by_reader.get("phase1_5_self", {})
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
        "| metric | Phase1 run-1 (mean-MSE) | Phase1 run-2 (sum-MSE) | **Phase 1.5** | Monet stage2 (target) | Monet stage3 (anti-target) | user overnight 2026-05-04 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| compression_ratio | 0.631 | 0.700 | {fmt(cr_self)} | 2.7 | 0.78 | ~0.03 |",
        f"| mean off-diag cos | 0.851 | 0.987 | {fmt(mean_cos)} | 0.38 | 0.85 | (>0.9) |",
        f"| n_helpful (≥3) | 1 | 2 | {nh_self} | 4 | 1 | ≤1 |",
        f"| utility (qwen_base) | 0.077 | 0.076 | {fmt(util_base)} | +2.7 | +0.26 | −0.22 to −1.10 |",
        f"| h-cosine | 0.851 | 0.987 | {fmt(mean_cos)} | 0.38 | 0.85 | n/a |",
        f"| v_roi-cosine | 0.465 | 0.465 | {fmt(v_off)} | n/a | n/a | n/a |",
        f"| utility (self reader) | 0.336 | n/a | {fmt(util_self)} | +2.7 | +0.26 | n/a |",
    ]

    body = f"""# Phase 1.5 — Monet's vendored architecture + Phase 1 LVR loss & data

## TL;DR

Phase 1.5 swaps Qwen2.5-VL-3B-Instruct's standard transformer for Monet's
vendored modified transformer (latent_mode=True recurrent latent generation +
optional 4D attention rules) — keeping Phase 1's LVR loss (mean-MSE form,
λ=1.0) and Visual-CoT data identical. On 200 held-out eval examples, the
trained latents score **{n_pass}/4** acceptance criteria — verdict **{overall}**.

Headline: `compression_ratio={fmt(cr_self)}` (target ≥ 0.4),
`mean_off_diag_cos={fmt(mean_cos)}` (target ≤ 0.55),
`n_helpful={nh_self}/8` (target ≥ 3),
frozen-Qwen utility `{fmt(util_base)}` (target > 0).
v_roi cosine = {fmt(v_off)} (Phase 1 reference: 0.465 — same data).

## 6+-column comparison table

{chr(10).join(table)}

## Per-position single-keep curve (phase1_5_self reader)

```
pos      nll   margin  bar
{curve_block}
```

## 8x8 cosine matrix

```
{cos_block}
```

mean off-diagonal cos = **{fmt(mean_cos)}** (Phase 0 thresholds: <0.4 distributed ✓, >0.85 collapsed ✗)

## Recipe

Identical to Phase 1 run 1 except:
- backbone = Monet's vendored Qwen2.5-VL-3B-Instruct class (sys.modules patch);
- latent slots use `<abs_vis_token>...<abs_vis_token_pad>×8...</abs_vis_token>` (Monet tokens) instead of Phase 1's `<|image_pad|>×8`;
- training does Monet's two-pass forward (latent_mode=True for h + ce_patch_vec; latent_mode=False with spliced ce_patch_vec for NTP);
- h is read from `outputs.hidden_states[0][-1]` (last-layer, K rows; grad-enabled) NOT `outputs.latent_embeds` (which is detached).

Loss: `L = L_NTP + 1.0 · F.mse_loss(h, v_roi)` (mean over 1×K×D). Vision tower + projector frozen; LLM full FT. AdamW lr=1e-5, warmup=100, cosine→0, eff bsz=4 (bsz=1 × grad_accum=4), bf16, 1000 steps, 5K examples, seed=0.

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

If Phase 1.5 PASSES → Monet's architecture IS the load-bearing mechanism;
Phase 2 (faithful stage 2 recipe) will mostly confirm.

If Phase 1.5 FAILS → architecture alone isn't enough; the aux-image-during-
training mechanism (which Phase 1.5 doesn't use, and Phase 2 will) is
implicated.

## Caveats

- Phase 1.5 uses raw Qwen2.5-VL-3B-Instruct as the backbone (no Stage 1 SFT)
  — same starting checkpoint as Phase 1, but with Monet's modified forward.
- The Monet special tokens (`<abs_vis_token*>`, `<observation>` etc.) are added
  to the tokenizer but their embeddings are randomly initialized via
  `model.resize_token_embeddings(new_vocab)`. Initial NTP is therefore
  higher than Phase 1's (8.2 vs 7.6); convergence trajectory differs.
- The Monet vendored model uses internal attention masks via `build_4d_attn`
  WHEN `attention_mask_4d` is passed. Phase 1.5 does NOT pass this — it relies
  on default causal masking + the latent_mode recurrence. This means Phase 1.5
  tests the recurrence mechanism alone, not the 4D attention mechanism.
- v_roi-cosine on Phase 1.5 (above) should match Phase 1's 0.465 since the
  ROI selection + image features are identical — verifies data parity.
"""
    out_path = ROOT / "REPORT.md"
    out_path.write_text(body)
    print(f"Wrote {out_path}")
    print(f"\nVerdict: {overall} ({n_pass}/4)")
    for k, v in crit.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")


if __name__ == "__main__":
    main()
