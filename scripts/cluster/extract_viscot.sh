#!/bin/bash
# Concatenate the 10 Visual-CoT image shards and extract.
#
# Background: the dataset's image data is one tar archive split into 13 fixed-
# size 10-GiB binary chunks named cot_images_00..12. They're NOT independent
# tar files — only shard 0 has the tar header and only the last shard has the
# EOF padding. Standard extraction requires `cat shard_* | tar -x`.
#
# Per probe report: we need shards 0, 1, 4-11 (10 of 13) for the 4-source mix.
# We CAN concat them in order even with shards 2, 3 missing — tar will fail
# midway through, producing a broken archive. SO: we extract ONLY the source
# directories that lie entirely within our 10-shard window.
#
# Strategy: concat all 13 (or use --include filtering on tar). We pull shards
# 2, 3 too OR we use a custom Python tar walker. Simpler: pull all 13 (~140 GB)
# OR concatenate the 10 contiguous + filter via tar --wildcards.
#
# DECISION: pull all 13 shards (the scope creep is small at 30 GB extra; it
# avoids the brittle byte-range filtering and gives us margin). Then standard
# `cat | tar -x` and we filter at the parquet stage.
#
# If you want to skip shards 2, 3, 12 (saving ~30 GB), use the alternative
# `extract_viscot_partial.sh` (not yet written — flag for future).
#
# Usage:
#   bash scripts/cluster/extract_viscot.sh <raw_shards_dir> <extract_target_dir>

set -euo pipefail

SHARDS_DIR="${1:?usage: $0 <shards_dir> <extract_dir>}"
EXTRACT_DIR="${2:?usage: $0 <shards_dir> <extract_dir>}"

mkdir -p "${EXTRACT_DIR}"

cd "${EXTRACT_DIR}"

# List shards in order (shell glob is alphabetical so cot_images_00..12 sorts correctly).
# Shards live under `cot_images_tar_split/` per the HF repo layout.
SHARDS=$(ls "${SHARDS_DIR}"/cot_images_tar_split/cot_images_* 2>/dev/null | sort)
if [ -z "${SHARDS}" ]; then
    # Fallback for already-flattened layouts (e.g., older runs)
    SHARDS=$(ls "${SHARDS_DIR}"/cot_images_* 2>/dev/null | sort)
fi
N_SHARDS=$(echo "${SHARDS}" | wc -l)
echo "[extract] concatenating ${N_SHARDS} shards from ${SHARDS_DIR}"
echo "${SHARDS}"
echo ""

# Filter set: only extract the 4 source directories we need. saves disk.
WANTED_DIRS=(
    "cot_image_data/docvqa/*"
    "cot_image_data/textvqa/*"
    "cot_image_data/flickr30k/*"
    "cot_image_data/openimages/*"
)

echo "[extract] streaming concat | tar -x with filter:"
printf '  %s\n' "${WANTED_DIRS[@]}"

# `cat ... | tar --wildcards -x ...` lets us filter on extraction time.
# `--ignore-zeros` because the stream contains the cot_images split with EOF
# records embedded mid-stream; tar would otherwise stop there.
cat ${SHARDS} | tar --ignore-zeros --wildcards -xf - "${WANTED_DIRS[@]}"

echo ""
echo "[extract] done. directory tree under ${EXTRACT_DIR}:"
du -sh "${EXTRACT_DIR}"/cot_image_data/* 2>/dev/null || true
echo ""
echo "[extract] image counts per source:"
for src in docvqa textvqa flickr30k openimages; do
    if [ -d "${EXTRACT_DIR}/cot_image_data/${src}" ]; then
        count=$(find "${EXTRACT_DIR}/cot_image_data/${src}" -type f | wc -l)
        echo "  ${src}: ${count} files"
    fi
done
