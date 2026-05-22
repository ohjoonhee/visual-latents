#!/bin/bash
# Cluster Phase 3 — JOB D BASELINE: Monet Stage 3 reproduction (no VICReg).
#
# WHY THIS JOB: Establish a Stage-3 baseline IN OUR INFRASTRUCTURE so the
# A/B with VICReg (Job D + VICReg) is clean on the regularizer variable
# only. Reference: phase0_monet_probe finds Stage 3 latents on the
# released Monet-SFT-7B collapse (mean off-diag cos ~0.87, n_helpful=1,
# util ~0). If our baseline also collapses → trainer infra is faithful
# → submit the VICReg arm. If our baseline does NOT collapse → infra
# accidentally broke collapse; pause and disambiguate.
#
# PRECONDITIONS:
#   1. Stage 1 SFT ckpt at /data/joonhee/visual-latents/cluster_phase3/stage1_sft/checkpoint
#   2. Teacher latents precomputed at .../teacher_latents/latent_all_layers_*.pt
#      (produced by cluster_jobD_precompute_teacher_latents.sbatch)
#
# Submit with: sb slurm/cluster_jobD_stage3_baseline.sbatch
# DO NOT agent-submit (project-owner rule §3-A).
#
#SBATCH -J vl-jobD-stage3-baseline
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
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.6

mkdir -p /data/joonhee/visual-latents/.cache/{tmp,triton,torch,torch-extensions} \
         /data/joonhee/visual-latents/wandb \
         /data/joonhee/visual-latents/cluster_phase3
export TMPDIR=/data/joonhee/visual-latents/.cache/tmp
export TRITON_CACHE_DIR=/data/joonhee/visual-latents/.cache/triton
export TORCH_HOME=/data/joonhee/visual-latents/.cache/torch
export TORCH_EXTENSIONS_DIR=/data/joonhee/visual-latents/.cache/torch-extensions
export WANDB_DIR=/data/joonhee/visual-latents/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"

slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}

LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
JOB_T0=$(date +%s)
elapsed_h () { printf "%.1fh" "$(echo "scale=1; ($(date +%s) - ${JOB_T0}) / 3600.0" | bc)"; }

slack_notify ":rocket: START vl-jobD-stage3-baseline | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
Stage 3 BASELINE (no VICReg). Reference: Phase 0 collapse signature (cos 0.87, n_helpful=1)
init: cluster_phase3/stage1_sft/checkpoint
teacher: cluster_phase3/teacher_latents/  (precomputed)
log:    \`${ABSLOG}\`"

# ----- Stage 0: venv -----
echo "[$(date)] [stage0] uv --project cluster sync"
uv --project cluster sync 2>&1 | tail -10
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  slack_notify ":x: vl-jobD-baseline stage0/uv-sync FAILED"; exit 11
fi

# ----- Stage 1: precondition checks -----
STAGE1_CKPT=/data/joonhee/visual-latents/cluster_phase3/stage1_sft/checkpoint
if [ ! -d "$STAGE1_CKPT" ]; then
  slack_notify ":x: vl-jobD-baseline Stage 1 ckpt MISSING at ${STAGE1_CKPT}"; exit 12
fi
TEACHER_DIR=/data/joonhee/visual-latents/cluster_phase3/teacher_latents
N_TEACHER=$(ls ${TEACHER_DIR}/latent_all_layers_*.pt 2>/dev/null | wc -l)
if [ "$N_TEACHER" -lt 100000 ]; then
  slack_notify ":x: vl-jobD-baseline teacher_latents incomplete: only ${N_TEACHER} files at ${TEACHER_DIR} (expected ~118K). Run precompute first."
  exit 12
fi
echo "[$(date)] Stage 1 ckpt OK; teacher_latents: ${N_TEACHER} files"

S3_OUT=/data/joonhee/visual-latents/cluster_phase3/stage3_jobD_baseline

# ----- Stage 2: SMOKE (5 trainer steps; mem + dist_type + alignment + vicreg gates) -----
slack_notify ":wrench: vl-jobD-baseline smoke (5 steps)"
echo "[$(date)] [smoke] trainer_monet_stage3.py --smoke (baseline)"

uv --project cluster run --no-sync \
  accelerate launch \
    --config_file cluster/accelerate_zero2.yaml \
    --num_machines=1 --num_processes=4 --mixed_precision=bf16 \
    cluster/trainer_monet_stage3.py \
      --config cluster/configs/stage3_monet_jobD_baseline.yaml \
      --out_dir ${S3_OUT}/_smoke \
      --smoke
SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then
  slack_notify ":x: vl-jobD-baseline smoke FAILED rc=${SMOKE_RC}"; exit 13
fi
if ! grep -q "\[mem-after-prepare\] dist_type=DEEPSPEED" "$LOGFILE"; then
  slack_notify ":x: vl-jobD-baseline smoke OK by rc BUT dist_type!=DEEPSPEED — DDP fallback"
  exit 14
fi
if ! grep -q "\[stage3-check\] alignment_active=True" "$LOGFILE"; then
  slack_notify ":x: vl-jobD-baseline smoke OK BUT [stage3-check] alignment_active!=True — teacher latents not consumed"
  exit 15
fi
if ! grep -q "\[vicreg-check\] vicreg_active=False" "$LOGFILE"; then
  slack_notify ":warning: vl-jobD-baseline expected vicreg_active=False (lambda_reg=0) — log doesn't confirm. Continuing anyway."
fi
echo "[$(date)] [smoke] OK ($(elapsed_h)): DEEPSPEED + alignment_active confirmed"
slack_notify ":white_check_mark: vl-jobD-baseline smoke OK at $(elapsed_h) — proceeding to full training"

# ----- Stage 3: full Stage 3 baseline training -----
slack_notify ":dart: vl-jobD-baseline training (2000 steps) START at $(elapsed_h)"
echo "[$(date)] [train] trainer_monet_stage3.py full run (baseline)"

uv --project cluster run --no-sync \
  accelerate launch \
    --config_file cluster/accelerate_zero2.yaml \
    --num_machines=1 --num_processes=4 --mixed_precision=bf16 \
    cluster/trainer_monet_stage3.py \
      --config cluster/configs/stage3_monet_jobD_baseline.yaml
TRAIN_RC=$?

S3_CKPT=${S3_OUT}/checkpoint
TOTAL_H=$(elapsed_h)
if [ $TRAIN_RC -eq 0 ]; then
  slack_notify ":tada: vl-jobD-baseline DONE in ${TOTAL_H} | ckpt=${S3_CKPT}
NEXT: eval baseline (local A6000) to confirm collapse signature.
If mean_cos > 0.7 → submit vl-jobD-stage3-vicreg (the treatment arm).
If mean_cos < 0.5 → our infra accidentally broke collapse; pause and disambiguate."
  exit 0
else
  slack_notify ":warning: vl-jobD-baseline exit=${TRAIN_RC} in ${TOTAL_H} | partial ckpt may exist at ${S3_OUT}"
  exit $TRAIN_RC
fi
