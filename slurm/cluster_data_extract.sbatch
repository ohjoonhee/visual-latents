#!/bin/bash
# Cluster Phase 3 — data extract: unzip Monet-SFT-125K images.zip files.
#
# Follow-up to cluster_data_prep.sbatch: snapshot_download fetched the zips
# but our trainers expect extracted images (see phase0_monet_probe/unpack.py
# for the local equivalent).
#
# Submit with: sb slurm/cluster_data_extract.sbatch
# DO NOT agent-submit (project-owner rule §3-A).
#
# Estimated wall: 15-30 min (12.6 GB → ~25 GB extracted across 2 subsets).
# Cost: 0 GPU-hours.
#
#SBATCH -J vl-data-extract
#SBATCH -p cpu-standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail
[ -f .env ] && { set -a; source .env; set +a; }

cd "$SLURM_SUBMIT_DIR"

slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}

LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
SCRIPT_PATH="${ORIG_SBATCH_SCRIPT:-${SLURM_SUBMIT_SCRIPT:-<unknown>}}"
slack_notify ":package: START vl-data-extract | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
log: \`${ABSLOG}\`"

ROOT=/data/joonhee/visual-latents/data/Monet-SFT-125K

for subset in Visual_CoT Zebra_CoT_count; do
  ZIP="$ROOT/$subset/images.zip"
  DEST="$ROOT/$subset"
  if [ ! -f "$ZIP" ]; then
    echo "[$(date)] $ZIP missing — skipping $subset"
    continue
  fi
  # Idempotency: if a non-trivial number of jpg/png files already extracted, skip.
  EXISTING=$(find "$DEST" -maxdepth 4 -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" -o -name "*.JPG" \) 2>/dev/null | head -100 | wc -l)
  if [ "$EXISTING" -ge 100 ]; then
    echo "[$(date)] $subset already has $EXISTING images extracted — skipping"
    continue
  fi
  echo "[$(date)] extracting $subset/images.zip → $DEST"
  cd "$DEST"
  unzip -q -o images.zip
  cd "$SLURM_SUBMIT_DIR"
  echo "[$(date)] $subset done. file count:"
  find "$DEST" -maxdepth 4 -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l
done

echo ""
echo "[$(date)] final disk usage:"
du -sh "$ROOT"/*

rc=$?
if [ $rc -eq 0 ]; then
  slack_notify ":white_check_mark: DONE vl-data-extract (jobid=${SLURM_JOB_ID})"
else
  slack_notify ":x: FAIL vl-data-extract (jobid=${SLURM_JOB_ID}) exit=${rc}"
fi
exit $rc
