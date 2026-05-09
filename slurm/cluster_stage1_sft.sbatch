#!/bin/bash
# Cluster Phase 3 — Stage 1 SFT (NTP-only warmup on Monet-SFT-125K Visual_CoT).
#
# Submit with: sb slurm/cluster_stage1_sft.sbatch
# DO NOT agent-submit.
#
# Estimated wall: 8-10 h. Output: cluster_phase3/stage1_sft/checkpoint.
# Cost: 4 GPUs × 10 h = 40 GPU-hours.
#
#SBATCH -J vl-stage1-sft
#SBATCH -p gpu-4farm
#SBATCH --account=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail
[ -f .env ] && { set -a; source .env; set +a; }

module purge
module load cuda/12.6

# Mandatory cache redirects (§3-G).
mkdir -p /data/joonhee/visual-latents/.cache/{tmp,triton,torch,torch-extensions} \
         /data/joonhee/visual-latents/wandb
export TMPDIR=/data/joonhee/visual-latents/.cache/tmp
export TRITON_CACHE_DIR=/data/joonhee/visual-latents/.cache/triton
export TORCH_HOME=/data/joonhee/visual-latents/.cache/torch
export TORCH_EXTENSIONS_DIR=/data/joonhee/visual-latents/.cache/torch-extensions
export WANDB_DIR=/data/joonhee/visual-latents/wandb
# HF_HOME inherited from ~/.bashrc.

cd "$SLURM_SUBMIT_DIR"

slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}

LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
SCRIPT_PATH="${ORIG_SBATCH_SCRIPT:-${SLURM_SUBMIT_SCRIPT:-<unknown>}}"
slack_notify ":rocket: START vl-stage1-sft | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
script: \`${SCRIPT_PATH}\`
log:    \`${ABSLOG}\`
follow: \`tail -n 200 -F ${ABSLOG}\`"

# Distributed launch — accelerate + ZeRO-2 + bf16, 4 procs on 1 node.
set +e
uv --project cluster run --no-sync \
  accelerate launch \
    --config_file cluster/accelerate_zero2.yaml \
    --num_machines=1 --num_processes=4 --multi_gpu --mixed_precision=bf16 \
    cluster/trainer_sft.py \
      --config cluster/configs/stage1_sft.yaml
rc=$?
set -e

if [ $rc -eq 0 ]; then
  slack_notify ":white_check_mark: DONE vl-stage1-sft (jobid=${SLURM_JOB_ID})"
else
  slack_notify ":x: FAIL vl-stage1-sft (jobid=${SLURM_JOB_ID}) exit=${rc}"
fi
exit $rc
