"""Download a published checkpoint revision from HuggingFace Hub to a local dir.

Companion to scripts/cluster/upload_ckpt_to_hf.py — pulls a specific
revision (tag/branch/commit) of a model repo locally so cluster/eval.py
can run on this machine's GPU.

Usage:
  uv --project cluster run python scripts/local/download_ckpt_from_hf.py \\
    --repo ohjoonhee/vlatents-qwen25vl7b-stage2-repro-v1 \\
    --revision step-1500 \\
    --local-dir ./checkpoints/stage2_repro_step1500

  # Then eval:
  VL_CLUSTER_DATA_ROOT=/mnt/ssd/Projects/visual-latents/phase0_monet_probe/data \\
    uv --project cluster run python cluster/eval.py \\
      --ckpts stage2_repro:./checkpoints/stage2_repro_step1500 \\
      --K 8 --n_eval 200 --out_root ./eval_out
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id (e.g. ohjoonhee/vlatents-qwen25vl7b-stage2-repro-v1)")
    ap.add_argument("--revision", default="main",
                    help="Tag, branch, or commit SHA (default: main). Use e.g. 'step-1500'.")
    ap.add_argument("--local-dir", required=True, type=Path,
                    help="Local destination directory")
    ap.add_argument("--ignore", nargs="*", default=["*.tmp", "*.lock"],
                    help="Glob patterns to skip during download")
    ap.add_argument("--list-revisions", action="store_true",
                    help="Print all branches/tags on the repo and exit (does not download)")
    args = ap.parse_args()

    api = HfApi()
    try:
        whoami = api.whoami()
        print(f"[download] authenticated as: {whoami.get('name', '?')}")
    except Exception:
        print(f"[download] no HF auth detected — public repos still work; private will 401.")

    if args.list_revisions:
        try:
            refs = api.list_repo_refs(args.repo, repo_type="model")
        except Exception as e:
            print(f"ERROR: list_repo_refs failed: {e}")
            return 1
        print(f"[download] {args.repo} branches:")
        for b in refs.branches:
            print(f"  {b.ref}  ({b.target_commit[:8]})")
        print(f"[download] {args.repo} tags:")
        for t in refs.tags:
            print(f"  {t.ref}  ({t.target_commit[:8]})")
        return 0

    args.local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {args.repo} @ {args.revision} → {args.local_dir}")
    path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=str(args.local_dir),
        ignore_patterns=args.ignore,
    )
    print(f"[download] DONE. files at: {path}")

    weights = list(args.local_dir.glob("model*.safetensors")) + list(args.local_dir.glob("pytorch_model*.bin"))
    print(f"[download] weight files: {len(weights)}")
    if not weights:
        print(f"WARN: no .safetensors or .bin found — verify the revision is what you expected.")
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
