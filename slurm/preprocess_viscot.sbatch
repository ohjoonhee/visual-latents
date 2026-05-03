#!/bin/bash
# Visual-CoT preprocessing job — CPU partition.
#
# Four stages: download → extract → process → push to HF Hub.
# After this job: the dataset is published on HF Hub and local A6000 can pull
# it with `huggingface-cli download` / `load_dataset`. V_sem precompute and
# training are LOCAL (no further cluster involvement).
#
# Submit with: sb slurm/preprocess_viscot.sbatch
# NEVER agent-submitted (project-owner rule).
#
# Estimated wall-clock: 4-8 hours.
#   Download:     1-2 h (107 GB)
#   Extract:      ~30 min (107 GB through tar with filter)
#   Process:      ~30 min (PIL.open passes for 50K examples)
#   Push to Hub:  1-2 h (12 GB upload)
#
# Prerequisites:
#   - HF_TOKEN with WRITE access to ohjoonhee/* in .env (or VISCOT_HUB_REPO override).
#   - `uv sync` has been run on the cluster (datasets>=3.2.0, pillow>=11).
#
#SBATCH -J vl-viscot-preprocess
#SBATCH -p cpu-standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail
set -a; source .env; set +a
export MACHINE=bioai

# CPU job — no CUDA load. Cache redirects for HF / tmp.
mkdir -p /data/joonhee/vl/tmp
export TMPDIR=/data/joonhee/vl/tmp
export HF_HOME=/data/joonhee/.cache/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"

# Paths.
RAW_DIR=/data/joonhee/vl/data/viscot_raw
EXTRACT_DIR=/data/joonhee/vl/data/viscot_extracted
PROCESSED_DIR=/data/joonhee/vl/data/viscot_processed
ANNOTATION="${RAW_DIR}/viscot_363k.json"
IMAGE_DIR="${EXTRACT_DIR}"

# Hub repo — override via .env VISCOT_HUB_REPO if you want a different name.
HUB_REPO="${VISCOT_HUB_REPO:-ohjoonhee/visual-cot-50k-poc}"

mkdir -p "${RAW_DIR}" "${EXTRACT_DIR}" "${PROCESSED_DIR}" logs

slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}

slack_notify ":arrow_down: START vl-viscot-preprocess jobid=${SLURM_JOB_ID} target=${HUB_REPO}"

# --- Stage 1: download ---
echo "===== STAGE 1: download (raw shards from HF) ====="
echo "[$(date)] starting download to ${RAW_DIR}"
uv run python scripts/cluster/download_viscot.py --out "${RAW_DIR}"

# --- Stage 2: extract ---
echo ""
echo "===== STAGE 2: extract (concat + tar -x with source filter) ====="
echo "[$(date)] extracting to ${EXTRACT_DIR}"
bash scripts/cluster/extract_viscot.sh "${RAW_DIR}" "${EXTRACT_DIR}"

# --- Stage 3: process (build HF Datasets locally) ---
echo ""
echo "===== STAGE 3: process (filter + sample + build HF Datasets) ====="
echo "[$(date)] writing to ${PROCESSED_DIR}"
uv run python scripts/cluster/process_viscot.py \
    --annotation "${ANNOTATION}" \
    --image-dir "${IMAGE_DIR}" \
    --out-dir "${PROCESSED_DIR}" \
    --target 50000 \
    --eval-size 1000

# --- Stage 4: push to HF Hub ---
echo ""
echo "===== STAGE 4: push to HF Hub ====="
echo "[$(date)] pushing as private dataset → ${HUB_REPO}"
uv run python scripts/cluster/push_viscot_to_hub.py \
    --train "${PROCESSED_DIR}/viscot_50k_train_hf" \
    --eval  "${PROCESSED_DIR}/viscot_1k_eval_hf" \
    --repo  "${HUB_REPO}"

echo ""
echo "===== ALL STAGES COMPLETE ====="
ls -la "${PROCESSED_DIR}"
echo ""
echo "Disk used:"
du -sh "${RAW_DIR}" "${EXTRACT_DIR}" "${PROCESSED_DIR}"

slack_notify ":white_check_mark: DONE vl-viscot-preprocess jobid=${SLURM_JOB_ID} → ${HUB_REPO}"
