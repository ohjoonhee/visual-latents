#!/bin/bash
# Cluster Phase 3 — data prep: download Monet-SFT-125K to /data/joonhee/.
#
# Submit with: sb slurm/cluster_data_prep.sbatch
# DO NOT agent-submit (project-owner rule §3-A).
#
# Estimated wall: 1-3 h (downloads ~13 GB Visual_CoT + ~few-GB Zebra_CoT_count
# + extracts images.zip if applicable). CPU-only — no GPU contention.
#
# Cost: 0 GPU-hours.
#
#SBATCH -J vl-data-prep
#SBATCH -p cpu-standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail
[ -f .env ] && { set -a; source .env; set +a; }

# Cache redirects (mandatory per cluster spec §3-G).
mkdir -p /data/joonhee/visual-latents/.cache/{tmp,triton,torch,torch-extensions} \
         /data/joonhee/visual-latents/wandb \
         /data/joonhee/visual-latents/data
export TMPDIR=/data/joonhee/visual-latents/.cache/tmp
export TRITON_CACHE_DIR=/data/joonhee/visual-latents/.cache/triton
export TORCH_HOME=/data/joonhee/visual-latents/.cache/torch
export TORCH_EXTENSIONS_DIR=/data/joonhee/visual-latents/.cache/torch-extensions
export WANDB_DIR=/data/joonhee/visual-latents/wandb
# HF_HOME inherited from ~/.bashrc.

cd "$SLURM_SUBMIT_DIR"

# Slack notify (graceful if SLACK_WEBHOOK_URL is unset).
slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}

LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
SCRIPT_PATH="${ORIG_SBATCH_SCRIPT:-${SLURM_SUBMIT_SCRIPT:-<unknown>}}"
slack_notify ":arrow_down: START vl-data-prep | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
script: \`${SCRIPT_PATH}\`
log:    \`${ABSLOG}\`"

DEST=/data/joonhee/visual-latents/data/Monet-SFT-125K
mkdir -p "$DEST"

echo "[$(date)] downloading Monet-SFT-125K → $DEST"
# Repo: NOVAglow646/Monet-SFT-125K (HuggingFace dataset). Verified by
# phase0_monet_probe/download.py.
# Use the project root .venv (has huggingface_hub via datasets); no need
# for the cluster venv (transformers etc.) for a pure HF download.
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='NOVAglow646/Monet-SFT-125K',
    repo_type='dataset',
    allow_patterns=['Visual_CoT/*', 'Zebra_CoT_count/*'],
    local_dir='${DEST}',
    max_workers=4,
)
print('Download complete.')
"

echo ""
echo "[$(date)] disk usage:"
du -sh "$DEST"
ls -la "$DEST"
echo ""
echo "[$(date)] Visual_CoT subset:"
ls -la "$DEST/Visual_CoT" | head -20

rc=$?
if [ $rc -eq 0 ]; then
  slack_notify ":white_check_mark: DONE vl-data-prep (jobid=${SLURM_JOB_ID})"
else
  slack_notify ":x: FAIL vl-data-prep (jobid=${SLURM_JOB_ID}) exit=${rc}"
fi
exit $rc
