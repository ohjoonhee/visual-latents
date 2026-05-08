"""Download Visual-CoT image shards from HuggingFace to bioai cluster storage.

The dataset's images live as a single tar archive split into 13 binary chunks
of ~10 GiB each (cot_images_00..12). Pull ALL 13 shards (~140 GB) — selective
shard skipping breaks the concatenated tar stream because the chunks are
binary `split -b` cuts, not per-source archives. Per-source filtering happens
later in `extract_viscot.sh` via `tar --wildcards`.

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

# All 13 shards. The cot_images_tar_split files are binary chunks of one tar
# archive (`split -b 10G`); only shard 0 carries the tar header and only
# shard 12 has the EOF padding. Selective skipping corrupts the concatenated
# stream at extract time (Unexpected EOF mid-archive — observed jobid 212990,
# May 4 2026). Pull all 13; filter by source folder later via tar --wildcards.
SHARDS_NEEDED = [
    f"cot_images_tar_split/cot_images_{i:02d}" for i in range(13)
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
