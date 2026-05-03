"""Download Visual-CoT image shards from HuggingFace to bioai cluster storage.

Per `/tmp/viscot_probe_report.md` (2026-05-03 probe), shards needed for the
4-source mix (DocVQA + TextVQA + Flickr30k + OpenImages) are 0, 1, 4-11
(10 of 13). Each shard ~10 GiB → ~107 GB total.

Run on cluster CPU job (cpu-standard partition).

Usage:
    uv run python scripts/cluster/download_viscot.py --out /data/joonhee/vl/data/viscot_raw
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "deepcs233/Visual-CoT"
REPO_TYPE = "dataset"

# 10 shards covering DocVQA + TextVQA + Flickr30k + OpenImages.
# Skipped: shards 2, 3 (GQA — not in our mix), shard 12 (v7w/visual7w/vizwiz/vsr).
# Note: in the HF repo the shards live under `cot_images_tar_split/` — verified
# 2026-05-04 via `HfApi().list_repo_files`. Earlier probe missed the prefix.
SHARDS_NEEDED = [
    "cot_images_tar_split/cot_images_00", "cot_images_tar_split/cot_images_01",
    "cot_images_tar_split/cot_images_04", "cot_images_tar_split/cot_images_05",
    "cot_images_tar_split/cot_images_06", "cot_images_tar_split/cot_images_07",
    "cot_images_tar_split/cot_images_08", "cot_images_tar_split/cot_images_09",
    "cot_images_tar_split/cot_images_10", "cot_images_tar_split/cot_images_11",
]

# Annotations file (already pulled locally pre-cluster, but pull again here
# so the cluster job is self-contained).
ANNOTATION_FILE = "viscot_363k.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir (e.g., /data/joonhee/vl/data/viscot_raw)")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Print env so we can confirm cache redirect is active in the slurm log.
    print(f"[download] HF_HOME={os.environ.get('HF_HOME')}")
    print(f"[download] HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
    print(f"[download] target out_dir={out_dir}")
    print(f"[download] {len(SHARDS_NEEDED)} shards needed: {SHARDS_NEEDED}")

    # Pull annotation file first (small, fail fast if anything's wrong).
    t0 = time.monotonic()
    print(f"[download] pulling {ANNOTATION_FILE}...")
    ann_path = hf_hub_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE, filename=ANNOTATION_FILE,
        local_dir=str(out_dir),
    )
    print(f"[download] -> {ann_path} ({Path(ann_path).stat().st_size / 1e6:.1f} MB)")

    # Pull image shards.
    for i, shard in enumerate(SHARDS_NEEDED):
        target = out_dir / shard
        if args.skip_existing and target.exists() and target.stat().st_size > 10 * 1024**3 * 0.95:
            print(f"[download] [{i+1}/{len(SHARDS_NEEDED)}] {shard} already present ({target.stat().st_size / 1e9:.1f} GB), skipping")
            continue
        t1 = time.monotonic()
        print(f"[download] [{i+1}/{len(SHARDS_NEEDED)}] pulling {shard}...")
        path = hf_hub_download(
            repo_id=REPO_ID, repo_type=REPO_TYPE, filename=shard,
            local_dir=str(out_dir),
        )
        sz = Path(path).stat().st_size
        elapsed = time.monotonic() - t1
        print(f"[download]     -> {path} ({sz / 1e9:.2f} GB, {sz / elapsed / 1e6:.1f} MB/s)")

    elapsed_total = time.monotonic() - t0
    total_size = sum((out_dir / s).stat().st_size for s in SHARDS_NEEDED if (out_dir / s).exists())
    print(f"[download] DONE: {total_size / 1e9:.1f} GB across {len(SHARDS_NEEDED)} shards in {elapsed_total/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
