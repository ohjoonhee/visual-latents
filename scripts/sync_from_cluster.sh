#!/bin/bash
# Pull results from bioai back to local. Idempotent rsync.
# Usage: bash scripts/sync_from_cluster.sh

set -euo pipefail

REMOTE="bioai:/data/joonhee/vl/results/"
LOCAL="results/"

mkdir -p "${LOCAL}"

echo "[sync] rsync ${REMOTE} -> ${LOCAL}"
rsync -av --progress --exclude='*.pt' --exclude='checkpoints/' \
  "${REMOTE}" "${LOCAL}"

echo "[sync] done."
