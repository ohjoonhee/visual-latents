#!/bin/bash
# Cluster Phase 3 — V3 (Monet Stage 2 + additive VICReg).
#
# Submit with: sb slurm/cluster_variant_v3.sbatch
# DO NOT agent-submit. REQUIRES Stage 1 SFT checkpoint.
#
# Same eff_bsz / 2-epoch schedule as Stage 2 baseline; ~18-22 h wall.
# Cost: 4 GPUs × 22 h = 88 GPU-hours.
#
#SBATCH -J vl-v3
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

mkdir -p /data/joonhee/visual-latents/.cache/{tmp,triton,torch,torch-extensions} \
         /data/joonhee/visual-latents/wandb
export TMPDIR=/data/joonhee/visual-latents/.cache/tmp
export TRITON_CACHE_DIR=/data/joonhee/visual-latents/.cache/triton
export TORCH_HOME=/data/joonhee/visual-latents/.cache/torch
export TORCH_EXTENSIONS_DIR=/data/joonhee/visual-latents/.cache/torch-extensions
export WANDB_DIR=/data/joonhee/visual-latents/wandb

cd "$SLURM_SUBMIT_DIR"

slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}

LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
SCRIPT_PATH="${ORIG_SBATCH_SCRIPT:-${SLURM_SUBMIT_SCRIPT:-<unknown>}}"
slack_notify ":rocket: START vl-v3 | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
script: \`${SCRIPT_PATH}\`
log:    \`${ABSLOG}\`
follow: \`tail -n 200 -F ${ABSLOG}\`"

INIT_CKPT=/data/joonhee/visual-latents/cluster_phase3/stage1_sft/checkpoint
if [ ! -d "$INIT_CKPT" ]; then
  slack_notify ":x: FAIL vl-v3 missing init_ckpt at ${INIT_CKPT}"
  echo "FATAL: missing $INIT_CKPT" >&2
  exit 2
fi

set +e
uv --project cluster run --no-sync \
  accelerate launch \
    --config_file cluster/accelerate_zero2.yaml \
    --num_machines=1 --num_processes=4 --multi_gpu --mixed_precision=bf16 \
    cluster/trainer_v3.py \
      --config cluster/configs/v3_monet_plus_vicreg.yaml
rc=$?
set -e

if [ $rc -eq 0 ]; then
  slack_notify ":white_check_mark: DONE vl-v3 (jobid=${SLURM_JOB_ID})"
else
  slack_notify ":x: FAIL vl-v3 (jobid=${SLURM_JOB_ID}) exit=${rc}"
fi
exit $rc
