#!/bin/bash
# Evaluation on bioai (1 GPU sufficient for held-out + transfer + steering).
# Submit with: sb slurm/eval.sbatch <checkpoint_dir>
# NEVER agent-submitted.
#
# NOTE: gpu-1farm was REMOVED by admin (per ~/.claude/docs/bioai_cluster_spec.md
# §2-G). 1-GPU work uses gpu-4farm with --gres=gpu:1 as the working practice.
# Confirm with admin before any production-scale 1-GPU campaign.
#
#SBATCH -J vl-eval
#SBATCH -p gpu-4farm
#SBATCH --account=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28                # within DefCpuPerGPU=28 for gpu-4farm
#SBATCH --mem=128G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail
set -a; source .env; set +a
export MACHINE=bioai

module purge
module load cuda/13.0                     # matches torch 2.11.0+cu130 wheel
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/13.0

# --- cache redirects (§3-G mandatory) ---
mkdir -p /data/joonhee/vl/tmp \
         /data/joonhee/vl/cache/{triton,torch,torch-extensions} \
         /data/joonhee/vl/wandb
export TMPDIR=/data/joonhee/vl/tmp
export TRITON_CACHE_DIR=/data/joonhee/vl/cache/triton
export TORCH_HOME=/data/joonhee/vl/cache/torch
export TORCH_EXTENSIONS_DIR=/data/joonhee/vl/cache/torch-extensions
export WANDB_DIR=/data/joonhee/vl/wandb

CKPT="${1:?usage: sb slurm/eval.sbatch <checkpoint_dir>}"

slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}

slack_notify ":mag: START vl-eval ckpt=${CKPT} jobid=${SLURM_JOB_ID}"

uv run python -m vl.eval --checkpoint "${CKPT}" --suite all
rc=$?

[ $rc -eq 0 ] \
  && slack_notify ":white_check_mark: DONE vl-eval ckpt=${CKPT} jobid=${SLURM_JOB_ID}" \
  || slack_notify ":x: FAIL vl-eval ckpt=${CKPT} jobid=${SLURM_JOB_ID} exit=${rc}"
exit $rc
