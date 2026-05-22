#!/usr/bin/env python
"""Download released Monet-SFT-7B from HuggingFace to a cluster path.

The released checkpoint is the TEACHER for our Stage 3 precompute (see
scripts/cluster/precompute_teacher_latents.py). We snapshot the entire
repo (~16 GB) so both stage1/ and stage2/ subdirs are available; the
Stage 3 precompute uses stage2/ as the teacher.

Usage on cluster:
  uv --project cluster run python scripts/cluster/download_monet_sft_7b.py

Default target: /data/joonhee/visual-latents/cluster_phase3/monet_sft_7b
(the path the precompute sbatch's auto-detect prefers).

Idempotent: snapshot_download with hf_xet skips already-present files.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target",
        default="/data/joonhee/visual-latents/cluster_phase3/monet_sft_7b",
        help="Local directory to populate.",
    )
    ap.add_argument("--repo", default="NOVAglow646/Monet-SFT-7B")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    print(f"[download] repo={args.repo}  revision={args.revision}  target={target}")
    # hf_xet enables dedup + parallel download where the cache is shared.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # xet is the modern path
    path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        # Skip useless files (sometimes repos ship preview imgs, etc.)
        ignore_patterns=["*.png", "*.jpg", "*.gif", "README*"],
    )
    print(f"[download] ok → {path}")
    # Sanity: stage2/ must be present.
    stage2 = target / "stage2"
    if not stage2.is_dir():
        # Fall back: try directly the path the snapshot returned.
        alt = Path(path) / "stage2"
        if not alt.is_dir():
            raise SystemExit(
                f"[download] stage2/ subdir not found under {target} or {path}. "
                f"Inspect the repo layout."
            )
        stage2 = alt
    print(f"[download] verified stage2/ at: {stage2}")
    print(f"[download] use this as --teacher_ckpt for precompute_teacher_latents.py")


if __name__ == "__main__":
    main()
