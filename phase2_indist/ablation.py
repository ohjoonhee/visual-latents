"""Phase 2 in-distribution ablation — thin wrapper around `phase2_monet_stage2/ablation_runner.py`.

We re-use the Phase 2 ablation runner verbatim (same readers: phase2_self +
qwen_base; same K=8 single-block; same modes including only_pos_*; same
v_roi cosine emission). Only the input directory changes.

Why a wrapper: the canonical Phase 0 ablation.py iterates over (stage, subset)
cells and expects a `latents/<stage>/<subset>/` layout. Our Phase 2 in-dist
case is a single (ckpt, subset) pair, exactly mirroring Phase 2 OOD —
so the Phase 2 runner is already the right shape.

Run:
  PY=phase0_monet_probe/.venv-monet/bin/python
  $PY phase2_indist/ablation.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASE2 = ROOT.parent / "phase2_monet_stage2"
PHASE0 = ROOT.parent / "phase0_monet_probe"


def main():
    py = sys.executable  # we expect to be invoked under .venv-monet
    runner = PHASE2 / "ablation_runner.py"
    if not runner.exists():
        print(f"FATAL: missing {runner}", flush=True)
        sys.exit(2)

    ckpt = PHASE2 / "results" / "run_p2" / "checkpoint"
    latents_dir = ROOT / "latents"
    out_results = ROOT / "ablation_results.jsonl"
    out_hstats = ROOT / "ablation_h_stats.jsonl"

    cmd = [
        py, str(runner),
        "--latents_dir", str(latents_dir),
        "--self_ckpt", str(ckpt),
        "--self_name", "phase2_self",
        "--qwen_base", "Qwen/Qwen2.5-VL-3B-Instruct",
        "--K", "8",
        "--max_examples", "200",
        "--out_results", str(out_results),
        "--out_hstats", str(out_hstats),
    ]
    print("[ablate-indist] running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[ablate-indist] FAILED rc={rc}", flush=True)
        sys.exit(rc)
    print(f"[ablate-indist] done. {out_results}, {out_hstats}", flush=True)


if __name__ == "__main__":
    main()
