#!/bin/bash
# Cluster Phase 3 — data REORG: move *.jpg files into Visual_CoT/images/
# and Zebra_CoT_count/images/ subdirs to match the paths inside train.json.
#
# Why: NOVAglow646/Monet-SFT-125K's images.zip extracts files at
# `Visual_CoT/*.jpg` (no subdir) but the dataset's train.json refers to
# them as `Visual_CoT/images/*.jpg`. Trainer 214423 hit FileNotFoundError
# on every example. This is a one-time idempotent fix.
#
# Submit with: sb slurm/cluster_data_reorg.sbatch
# DO NOT agent-submit (project-owner rule §3-A).
#
# Estimated wall: 5-15 min (240K + 17K rename ops on GPFS).
# Cost: 0 GPU-hours.
#
#SBATCH -J vl-data-reorg
#SBATCH -p cpu-standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -uo pipefail
[ -f .env ] && { set -a; source .env; set +a; }

cd "$SLURM_SUBMIT_DIR"

slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}
slack_notify ":wrench: START vl-data-reorg | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}"

ROOT=/data/joonhee/visual-latents/data/Monet-SFT-125K

for subset in Visual_CoT Zebra_CoT_count; do
  DIR="$ROOT/$subset"
  IMG="$DIR/images"
  if [ ! -d "$DIR" ]; then
    echo "[$(date)] $DIR missing — skip"
    continue
  fi
  mkdir -p "$IMG"
  COUNT_TOP=$(find "$DIR" -maxdepth 1 -type f -name "*.jpg" 2>/dev/null | wc -l)
  COUNT_SUB=$(find "$IMG" -maxdepth 1 -type f -name "*.jpg" 2>/dev/null | wc -l)
  echo "[$(date)] $subset: top=$COUNT_TOP sub(images/)=$COUNT_SUB"

  if [ "$COUNT_TOP" -eq 0 ] && [ "$COUNT_SUB" -gt 0 ]; then
    echo "[$(date)] $subset already reorganized — skip"
    continue
  fi
  if [ "$COUNT_TOP" -eq 0 ]; then
    echo "[$(date)] $subset has no top-level jpgs and empty images/ — something wrong"
    continue
  fi

  echo "[$(date)] moving $COUNT_TOP files: $DIR/*.jpg -> $IMG/"
  # find ... -exec mv -t ... {} + batches the moves efficiently.
  find "$DIR" -maxdepth 1 -type f -name "*.jpg" -exec mv -t "$IMG" {} +
  RC=$?

  COUNT_SUB_AFTER=$(find "$IMG" -maxdepth 1 -type f -name "*.jpg" 2>/dev/null | wc -l)
  COUNT_TOP_AFTER=$(find "$DIR" -maxdepth 1 -type f -name "*.jpg" 2>/dev/null | wc -l)
  echo "[$(date)] $subset done: top=$COUNT_TOP_AFTER sub(images/)=$COUNT_SUB_AFTER (rc=$RC)"
  if [ "$COUNT_TOP_AFTER" -ne 0 ]; then
    echo "[$(date)] WARNING: $subset still has $COUNT_TOP_AFTER top-level jpgs"
  fi
done

echo ""
echo "[$(date)] final disk usage:"
du -sh "$ROOT"/*

# Spot-check the path the trainer expects.
echo ""
echo "[$(date)] spot check — first sample image path from train.json:"
python3 -c "
import json, os, sys
for subset in ['Visual_CoT', 'Zebra_CoT_count']:
    p = f'$ROOT/{subset}/train.json'
    if not os.path.exists(p): continue
    raw = json.load(open(p))
    sample = raw[0]
    for turn in sample['data']:
        for c in turn.get('content', []):
            if c.get('type') == 'image':
                rel = c['image']
                full = f'$ROOT/../{rel}' if rel.startswith('Monet-SFT-125K/') else f'$ROOT/{rel}'
                print(f'  {subset}: {rel} -> exists={os.path.exists(full)}')
                break
        else: continue
        break
"

slack_notify ":white_check_mark: DONE vl-data-reorg (jobid=${SLURM_JOB_ID})"
exit 0
