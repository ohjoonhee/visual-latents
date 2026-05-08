"""Aggregate overnight results into a single readable summary table.

Reads results/<run>/{losses,eval,ablation_eval}.jsonl for each run name
listed in --runs (or auto-discovered overnight runs) and prints:
  - per-run: trajectory of held-out NLL across training
  - per-run: ablation table (mode → mean nll)
  - cross-run: natural-vs-perm gap on the headline ablation modes
  - cross-task: shapes vs grid for matched cells

Output: docs/overnight_2026_05_04/RESULTS_TABLE.md (created/overwritten).
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS_OUT = ROOT / "docs" / "overnight_2026_05_04"


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def aggregate_one(run: str) -> dict:
    """Summary dict for a single run."""
    losses = read_jsonl(RESULTS / run / "losses.jsonl")
    evals = read_jsonl(RESULTS / run / "eval.jsonl")
    ablation = read_jsonl(RESULTS / run / "ablation_eval.jsonl")

    out: dict = {"run": run, "n_steps": len(losses)}
    if losses:
        out["nll_first"] = losses[0].get("nll")
        out["nll_last"] = losses[-1].get("nll")
        out["mean_h_norm_first"] = losses[0].get("mean_h_norm")
        out["mean_h_norm_last"] = losses[-1].get("mean_h_norm")
        out["category"] = losses[-1].get("category")
        out["elapsed_s"] = losses[-1].get("elapsed_s")
    if evals:
        out["eval_first"] = evals[0]["eval_nll_mean"]
        out["eval_last"] = evals[-1]["eval_nll_mean"]
        out["eval_traj"] = [(e["step"], e["eval_nll_mean"]) for e in evals]
    out["ablation"] = {}
    out["h_stats"] = None
    out["mean_off_diag_cos"] = None
    if ablation:
        # Average over anchors per mode; also keep per-anchor for reference
        per_mode: dict[str, list[float]] = {}
        for r in ablation:
            if r["mode"] == "_h_stats":
                out["h_stats"] = r.get("h_stats")
                # New (additive) Phase-0-named field, written by trainer.py
                # alongside the legacy `h_stats` blob. Prefer it when present;
                # otherwise computed from `pairwise_cosine_mean` further below.
                if "mean_off_diag_cos" in r:
                    out["mean_off_diag_cos"] = r["mean_off_diag_cos"]
                continue
            per_mode.setdefault(r["mode"], []).append((r["anchor"], r["nll_mean"]))
        for mode, vals in per_mode.items():
            anchor_sorted = sorted(vals)
            out["ablation"][mode] = {
                "mean": sum(v for _, v in anchor_sorted) / len(anchor_sorted),
                "per_anchor": [v for _, v in anchor_sorted],
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args()

    if args.runs is None:
        # Auto-discover any run whose name starts with "interleaved_grid_" or
        # ends with "_200" or "_500" under results/
        runs = sorted({
            p.name for p in RESULTS.iterdir()
            if p.is_dir()
            and (p.name.startswith("interleaved_grid_")
                 or p.name.startswith("interleaved_shapes_"))
            and (p / "ablation_eval.jsonl").exists()
        })
    else:
        runs = args.runs

    print(f"[analyse] {len(runs)} runs")
    summaries = [aggregate_one(r) for r in runs]

    DOCS_OUT.mkdir(parents=True, exist_ok=True)
    out_md = DOCS_OUT / "RESULTS_TABLE.md"

    # Compose the markdown report
    lines: list[str] = []
    lines.append("# Overnight 2026-05-04 — Empirical Results")
    lines.append("")
    lines.append(f"Auto-generated from {len(runs)} runs.")
    lines.append("")

    # ---- Per-run training+eval summary ----
    lines.append("## 1. Training / held-out NLL trajectory")
    lines.append("")
    lines.append("| Run | Steps | nll_first | nll_last | eval_first | eval_last | Δeval | ‖h‖ first→last |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        nll_first = f"{s.get('nll_first', float('nan')):.2f}" if s.get('nll_first') is not None else "—"
        nll_last = f"{s.get('nll_last', float('nan')):.2f}" if s.get('nll_last') is not None else "—"
        ef = s.get("eval_first")
        el = s.get("eval_last")
        if ef is not None and el is not None and not math.isnan(ef) and not math.isnan(el):
            d = el - ef
            ef_s = f"{ef:.2f}"; el_s = f"{el:.2f}"; d_s = f"{d:+.2f}"
        else:
            ef_s = el_s = d_s = "—"
        h0 = s.get("mean_h_norm_first"); hL = s.get("mean_h_norm_last")
        h_s = f"{h0:.0f}→{hL:.0f}" if h0 is not None and hL is not None else "—"
        lines.append(
            f"| {s['run']} | {s['n_steps']} | {nll_first} | {nll_last} "
            f"| {ef_s} | {el_s} | {d_s} | {h_s} |"
        )
    lines.append("")

    # ---- Per-run ablation tables ----
    lines.append("## 2. Latent-position ablation eval (held-out NLL, anchors averaged)")
    lines.append("")
    has_any_ablation = any(s.get("ablation") for s in summaries)
    if has_any_ablation:
        # Determine column order from the union of all runs' modes
        all_modes_set: set[str] = set()
        for s in summaries:
            all_modes_set.update(s.get("ablation", {}).keys())
        first_modes = sorted(all_modes_set)
        # Reorder to a canonical sequence if present
        canonical = [
            "all", "first_half", "last_half", "first_only", "last_only",
            "first_of_each_block", "last_of_each_block",
            "first_block", "last_block", "none",
        ]
        modes = [m for m in canonical if m in first_modes] + \
                [m for m in first_modes if m not in canonical]

        header = "| Run | " + " | ".join(modes) + " |"
        sep = "|---|" + "---|" * len(modes)
        lines.append(header)
        lines.append(sep)
        for s in summaries:
            row = [s["run"]]
            for m in modes:
                v = s["ablation"].get(m, {}).get("mean")
                row.append(f"{v:.2f}" if v is not None else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append(
            "**How to read:** lower NLL = better. `none` = all latent slots zeroed "
            "(reader sees raw text only). `all` = no ablation. If the latents are "
            "really doing useful work, `all` should be lower than `none`. If the "
            "answer info is concentrated in the *last* latent (compression-to-final-"
            "state, predicted for state tasks), `last_only` ≈ `all`. If info is "
            "spread evenly (evidence-style), the per-position curves should be flatter."
        )
        lines.append("")

        # ---- Per-anchor breakdown (for the headline modes) ----
        # The shared-LoRA anchor (anchor 0) and frozen anchor (anchor 1) often
        # behave VERY differently — averaging over them hides the dichotomy.
        lines.append("### 2.1 Per-anchor breakdown — anchor 0 (shared LoRA) vs anchor 1 (frozen)")
        lines.append("")
        headline_modes = [m for m in ["all", "first_half", "last_half", "first_only", "last_only", "none"] if m in first_modes]
        for anchor_idx in (0, 1):
            lines.append(f"**Anchor {anchor_idx}** ({'shared LoRA' if anchor_idx==0 else 'frozen no-LoRA'})")
            lines.append("")
            header = "| Run | " + " | ".join(headline_modes) + " |"
            sep = "|---|" + "---|" * len(headline_modes)
            lines.append(header)
            lines.append(sep)
            for s in summaries:
                row = [s["run"]]
                for m in headline_modes:
                    info = s["ablation"].get(m, {})
                    pa = info.get("per_anchor", [])
                    v = pa[anchor_idx] if anchor_idx < len(pa) else None
                    row.append(f"{v:.2f}" if v is not None else "—")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        # ---- Per-position single-keep curves ----
        per_pos_modes = [m for m in first_modes if m.startswith("only_pos_")]
        if per_pos_modes:
            per_pos_modes.sort(key=lambda x: int(x.split("_")[-1]))
            lines.append("### 2.2 Per-position single-keep ablation (only this position kept; others zeroed)")
            lines.append("")
            for anchor_idx in (0, 1):
                lines.append(f"**Anchor {anchor_idx}**")
                lines.append("")
                header = "| Run | " + " | ".join(m.replace("only_pos_", "p") for m in per_pos_modes) + " | none |"
                sep = "|---|" + "---|" * (len(per_pos_modes) + 1)
                lines.append(header)
                lines.append(sep)
                for s in summaries:
                    row = [s["run"]]
                    for m in per_pos_modes:
                        info = s["ablation"].get(m, {})
                        pa = info.get("per_anchor", [])
                        v = pa[anchor_idx] if anchor_idx < len(pa) else None
                        row.append(f"{v:.2f}" if v is not None else "—")
                    info = s["ablation"].get("none", {})
                    pa = info.get("per_anchor", [])
                    v = pa[anchor_idx] if anchor_idx < len(pa) else None
                    row.append(f"{v:.2f}" if v is not None else "—")
                    lines.append("| " + " | ".join(row) + " |")
                lines.append("")
            lines.append(
                "Lower than `none` = this position carries some answer signal. "
                "If only late positions beat `none`, the model has compressed answer-"
                "relevant info into late latents (consistent with "
                "compression-to-final-state for state-tracking)."
            )
            lines.append("")

        # ---- h-statistics ----
        lines.append("### 2.3 h-statistics (per-position norm + pairwise cosine)")
        lines.append("")
        for s in summaries:
            hs = s.get("h_stats")
            if not hs:
                continue
            norms = hs.get("per_pos_mean_norm", [])
            cos = hs.get("pairwise_cosine_mean", [])
            lines.append(f"**{s['run']}**")
            lines.append("")
            lines.append("Per-position mean ‖h‖: " + ", ".join(f"p{i}={v:.1f}" for i, v in enumerate(norms)))
            if cos and len(cos) > 1:
                K = len(cos)
                # Prefer the trainer-stored scalar (unique-pair mean,
                # K*(K-1)/2 entries) when available; otherwise compute the
                # full off-diagonal mean (equivalent for symmetric matrices).
                stored = s.get("mean_off_diag_cos")
                if stored is not None:
                    avg = stored
                else:
                    vals = []
                    for i in range(K):
                        for j in range(K):
                            if i != j:
                                vals.append(cos[i][j])
                    avg = sum(vals) / len(vals)
                lines.append(f"")
                lines.append(f"Avg off-diagonal pairwise cosine (0=orthogonal, 1=identical): **{avg:.3f}**")
                lines.append("")
                lines.append("Pairwise cosine matrix (rows/cols are latent positions):")
                lines.append("")
                lines.append("| | " + " | ".join(f"p{j}" for j in range(K)) + " |")
                lines.append("|---|" + "---|" * K)
                for i, row in enumerate(cos):
                    lines.append(f"| **p{i}** | " + " | ".join(f"{v:.2f}" for v in row) + " |")
            lines.append("")

    # ---- Per-position utility curve ----
    if any(any(m.startswith("only_pos_") for m in s.get("ablation", {})) for s in summaries):
        lines.append("### 2.4a Per-position utility (none − only_pos_i; higher = position helps more)")
        lines.append("")
        K = 8  # current configs all use K_total=8
        per_pos_modes = [f"only_pos_{i}" for i in range(K)]
        for anchor_idx in (0, 1):
            lines.append(f"**Anchor {anchor_idx}**")
            lines.append("")
            header = "| Run | " + " | ".join(f"p{i}" for i in range(K)) + " |"
            sep = "|---|" + "---|" * K
            lines.append(header)
            lines.append(sep)
            for s in summaries:
                ab = s.get("ablation", {})
                pa_none = ab.get("none", {}).get("per_anchor", [])
                if anchor_idx >= len(pa_none):
                    continue
                base = pa_none[anchor_idx]
                row = [s["run"]]
                for m in per_pos_modes:
                    pa = ab.get(m, {}).get("per_anchor", [])
                    if anchor_idx < len(pa):
                        util = base - pa[anchor_idx]
                        row.append(f"{util:+.2f}")
                    else:
                        row.append("—")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
        lines.append(
            "Reads as: positive = this position helps the reader vs zeros baseline; "
            "negative = this position confuses the reader more than zeros do. A clean "
            "evidence-style pattern would have all positions positive and roughly equal. "
            "A clean state-tracking compression pattern would have only the last few "
            "positions positive."
        )
        lines.append("")

    # ---- Derived metrics: latent utility + compression ratio ----
    lines.append("### 2.4 Derived metrics (per-anchor)")
    lines.append("")
    lines.append("**Latent utility** = `none - all` (higher → latents help more; <0 → hurt).")
    lines.append("")
    lines.append("**Compression ratio** = `(last_half − all) / (first_half − all)` — values "
                 "near 0 indicate ALL useful info is in the last block; values >1 indicate "
                 "info is concentrated in the first block; ~1 means evenly split.")
    lines.append("")
    lines.append("| Run | anchor | utility | compression_ratio |")
    lines.append("|---|---|---|---|")
    for s in summaries:
        for anchor_idx in (0, 1):
            ab = s.get("ablation", {})
            def _g(mode):
                pa = ab.get(mode, {}).get("per_anchor", [])
                return pa[anchor_idx] if anchor_idx < len(pa) else None
            a_all = _g("all"); a_none = _g("none")
            a_first = _g("first_half"); a_last = _g("last_half")
            if a_all is None or a_none is None:
                continue
            util = a_none - a_all
            if a_first is not None and a_last is not None:
                num = a_last - a_all
                den = a_first - a_all
                if abs(den) > 1e-6:
                    cr = num / den
                    cr_s = f"{cr:.2f}"
                else:
                    cr_s = "—"
            else:
                cr_s = "—"
            lines.append(f"| {s['run']} | {anchor_idx} | {util:+.2f} | {cr_s} |")
    lines.append("")

    # ---- Cross-run pairs ----
    lines.append("## 3. Natural vs Permutation gap")
    lines.append("")
    pairs = [
        ("interleaved_shapes_natural_200", "interleaved_shapes_perm_200"),
        ("interleaved_shapes_multiq_natural_200", "interleaved_shapes_multiq_perm_200"),
        ("interleaved_grid_natural_200", "interleaved_grid_perm_200"),
        ("interleaved_grid_multiq_natural_200", "interleaved_grid_multiq_perm_200"),
        ("interleaved_shapes_multiq_natural_500", "interleaved_shapes_multiq_perm_500"),
        ("interleaved_grid_multiq_natural_500", "interleaved_grid_multiq_perm_500"),
    ]
    by_run = {s["run"]: s for s in summaries}
    lines.append("| Pair | natural eval_last | perm eval_last | gap (perm − natural) | natural ablation `all` | perm ablation `all` |")
    lines.append("|---|---|---|---|---|---|")
    for nat_name, perm_name in pairs:
        nat = by_run.get(nat_name); perm = by_run.get(perm_name)
        if nat is None or perm is None:
            continue
        ne = nat.get("eval_last"); pe = perm.get("eval_last")
        if ne is None or pe is None:
            continue
        gap = pe - ne
        nat_all = nat.get("ablation", {}).get("all", {}).get("mean")
        perm_all = perm.get("ablation", {}).get("all", {}).get("mean")
        nat_all_s = f"{nat_all:.2f}" if nat_all is not None else "—"
        perm_all_s = f"{perm_all:.2f}" if perm_all is not None else "—"
        lines.append(
            f"| {nat_name.replace('interleaved_', '')} vs {perm_name.replace('interleaved_', '')} "
            f"| {ne:.2f} | {pe:.2f} | {gap:+.2f} | {nat_all_s} | {perm_all_s} |"
        )
    lines.append("")
    lines.append(
        "**How to read:** `gap > 0` means natural pairing trains better than permuted "
        "control (which is what we want — model uses the image-question correspondence). "
        "`gap ≤ 0` means perm matches or beats natural — model is gaming the answer "
        "marginal, not encoding image content."
    )
    lines.append("")

    # ---- Save raw summaries ----
    raw = DOCS_OUT / "summaries.json"
    with raw.open("w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"[analyse] raw summaries → {raw}")

    out_md.write_text("\n".join(lines))
    print(f"[analyse] markdown report → {out_md}")
    print("---")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
