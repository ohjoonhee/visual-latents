#!/bin/bash
# Cluster Phase 3 — SMOKE: 7B Stage 1 SFT, 20 steps, ZeRO-2 validation.
#
# Goal: verify the --multi_gpu→DDP fallback hypothesis is true by checking
# the new [mem-after-prepare] diagnostic. With ZeRO-2 + CPU optim offload,
# baseline should be ~15-25 GB allocated; if we see >40 GB, ZeRO is not
# active.
#
# Cost: 1h walltime cap × 4 H100 = at most 4 GPU-hours, but expected wall
# is ~10-15 min including venv-sync no-op + model load + 20 micro-steps.
#
# Submit with: sb slurm/cluster_smoke_zero2.sbatch
# DO NOT agent-submit (project-owner rule §3-A).
#
#SBATCH -J vl-smoke-zero2
#SBATCH -p gpu-4farm
#SBATCH --account=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH -t 1:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -uo pipefail
[ -f .env ] && { set -a; source .env; set +a; }

module purge
module load cuda/12.6
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.6

mkdir -p /data/joonhee/visual-latents/.cache/{tmp,triton,torch,torch-extensions} \
         /data/joonhee/visual-latents/wandb \
         /data/joonhee/visual-latents/cluster_phase3
export TMPDIR=/data/joonhee/visual-latents/.cache/tmp
export TRITON_CACHE_DIR=/data/joonhee/visual-latents/.cache/triton
export TORCH_HOME=/data/joonhee/visual-latents/.cache/torch
export TORCH_EXTENSIONS_DIR=/data/joonhee/visual-latents/.cache/torch-extensions
export WANDB_DIR=/data/joonhee/visual-latents/wandb
# Helps with fragmentation across many image sizes. Cheap to enable.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"

slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}
LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
SCRIPT_PATH="${ORIG_SBATCH_SCRIPT:-<unknown>}"

# §3-H mandatory START format — do not modify emoji or field order.
slack_notify ":rocket: JOB START | ${SLURM_JOB_NAME} | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
script: \`${SCRIPT_PATH}\`
log: \`${ABSLOG}\`
follow: \`tail -n 200 -F ${ABSLOG}\`"

echo "[$(date)] uv sync"
uv --project cluster sync 2>&1 | tail -10

# Note: NO --multi_gpu flag. The accelerate config already specifies
# distributed_type: DEEPSPEED; passing --multi_gpu may silently force
# the DDP fallback that exhausted memory in job 214425.
echo "[$(date)] launch — Stage 1 smoke (20 steps)"
uv --project cluster run --no-sync \
  accelerate launch \
    --config_file cluster/accelerate_zero2.yaml \
    --num_machines=1 --num_processes=4 --mixed_precision=bf16 \
    cluster/trainer_sft.py \
      --config cluster/configs/stage1_sft_smoke.yaml
RC=$?

# §3-H mandatory DONE / FAIL format — do not modify.
if [ $RC -eq 0 ]; then
  slack_notify ":white_check_mark: DONE ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID})"
else
  slack_notify ":x: FAIL ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID}) exit=${RC}"
fi
exit $RC
