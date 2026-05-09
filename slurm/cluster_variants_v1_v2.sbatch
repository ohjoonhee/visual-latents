#!/bin/bash
# Cluster Phase 3 — V1 (G@7B K=4) + V2 (D2@7B K=8) chained in one allocation.
#
# Submit with: sb slurm/cluster_variants_v1_v2.sbatch
# DO NOT agent-submit. REQUIRES Stage 1 SFT checkpoint.
#
# Each variant is a Pivot A run on visual-cot-50k-poc, ~6-9 h wall apiece.
# We attempt V2 even if V1 exits non-zero — getting any data is better than
# losing the whole 24h slot. Both write checkpoints to their own out_dir.
#
# Cost: 4 GPUs × 18 h = 72 GPU-hours.
#
#SBATCH -J vl-v1-v2
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

set -uo pipefail
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
slack_notify ":rocket: START vl-v1-v2 | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
script: \`${SCRIPT_PATH}\`
log:    \`${ABSLOG}\`
follow: \`tail -n 200 -F ${ABSLOG}\`"

INIT_CKPT=/data/joonhee/visual-latents/cluster_phase3/stage1_sft/checkpoint
if [ ! -d "$INIT_CKPT" ]; then
  slack_notify ":x: FAIL vl-v1-v2 missing init_ckpt at ${INIT_CKPT}"
  echo "FATAL: missing $INIT_CKPT" >&2
  exit 2
fi

# ---- V1 (G@7B K=4) ----
slack_notify ":zap: V1 starting (G@7B K=4)"
uv --project cluster run --no-sync \
  accelerate launch \
    --config_file cluster/accelerate_zero2.yaml \
    --num_machines=1 --num_processes=4 --multi_gpu --mixed_precision=bf16 \
    cluster/trainer_pivot.py \
      --config cluster/configs/v1_g_at_7b.yaml
rc_v1=$?
if [ $rc_v1 -eq 0 ]; then
  slack_notify ":white_check_mark: V1 done"
else
  slack_notify ":warning: V1 exited ${rc_v1}; proceeding with V2 anyway"
fi

# ---- V2 (D2@7B K=8) ----
# Run V2 regardless of V1 outcome — separate output dir, independent run.
slack_notify ":zap: V2 starting (D2@7B K=8)"
uv --project cluster run --no-sync \
  accelerate launch \
    --config_file cluster/accelerate_zero2.yaml \
    --num_machines=1 --num_processes=4 --multi_gpu --mixed_precision=bf16 \
    cluster/trainer_pivot.py \
      --config cluster/configs/v2_d2_at_7b.yaml
rc_v2=$?

if [ $rc_v1 -eq 0 ] && [ $rc_v2 -eq 0 ]; then
  slack_notify ":white_check_mark: DONE vl-v1-v2 (jobid=${SLURM_JOB_ID}) — V1 + V2 both succeeded"
  exit 0
elif [ $rc_v1 -eq 0 ]; then
  slack_notify ":warning: vl-v1-v2 (jobid=${SLURM_JOB_ID}) — V1 ok, V2 failed exit=${rc_v2}"
  exit $rc_v2
elif [ $rc_v2 -eq 0 ]; then
  slack_notify ":warning: vl-v1-v2 (jobid=${SLURM_JOB_ID}) — V1 failed exit=${rc_v1}, V2 ok"
  exit $rc_v1
else
  slack_notify ":x: FAIL vl-v1-v2 (jobid=${SLURM_JOB_ID}) — V1 ${rc_v1}, V2 ${rc_v2}"
  exit $rc_v1
fi
