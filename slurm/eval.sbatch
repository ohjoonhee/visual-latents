#!/bin/bash
# Evaluation on bioai (1 GPU sufficient for held-out + transfer + steering).
# Submit with: sb slurm/eval.sbatch <checkpoint_dir>
# NEVER agent-submitted.
#
#SBATCH -J vl-eval
#SBATCH -p gpu-1farm
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 4:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail
set -a; source .env; set +a
export MACHINE=bioai
module load cuda/12.6
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.6

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
