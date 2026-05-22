#!/bin/bash
# Cluster Phase 3 — JOB D + VICReg: Monet Stage 3 + VICReg on latents.
#
# WHY THIS JOB: The treatment arm of the controlled A/B. Identical to
# Job D BASELINE in every respect EXCEPT lambda_reg=1.0 (was 0.0). The
# hypothesis: VICReg's variance hinge + cov penalty rescue the low-rank
# degeneracy that Stage 3's all-layer-cosine alignment loss permits.
#
# PRECONDITIONS:
#   1. Job D BASELINE has completed AND its eval confirmed collapse
#      (mean_cos > 0.7 on the internal probe). User-confirmed before submit.
#   2. Stage 1 SFT ckpt at standard path
#   3. Teacher latents at .../teacher_latents/ (same as baseline)
#
# Submit with: sb slurm/cluster_jobD_stage3_vicreg.sbatch
# DO NOT agent-submit (project-owner rule §3-A). DO NOT submit until
# baseline-collapse has been confirmed.
#
#SBATCH -J vl-jobD-stage3-vicreg
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

slack_notify ":rocket: START vl-jobD-stage3-vicreg | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
Stage 3 + VICReg (treatment arm — lambda_reg=1.0). Hypothesis: rescues Stage 3 collapse.
init: cluster_phase3/stage1_sft/checkpoint
teacher: cluster_phase3/teacher_latents/  (precomputed)
log:    \`${ABSLOG}\`"

# ----- Stage 0: venv -----
echo "[$(date)] [stage0] uv --project cluster sync"
uv --project cluster sync 2>&1 | tail -10
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  slack_notify ":x: vl-jobD-vicreg stage0/uv-sync FAILED"; exit 11
fi

# ----- Stage 1: precondition checks -----
STAGE1_CKPT=/data/joonhee/visual-latents/cluster_phase3/stage1_sft/checkpoint
if [ ! -d "$STAGE1_CKPT" ]; then
  slack_notify ":x: vl-jobD-vicreg Stage 1 ckpt MISSING"; exit 12
fi
TEACHER_DIR=/data/joonhee/visual-latents/cluster_phase3/teacher_latents
N_TEACHER=$(ls ${TEACHER_DIR}/latent_all_layers_*.pt 2>/dev/null | wc -l)
if [ "$N_TEACHER" -lt 100000 ]; then
  slack_notify ":x: vl-jobD-vicreg teacher_latents incomplete: ${N_TEACHER} files"; exit 12
fi

S3_OUT=/data/joonhee/visual-latents/cluster_phase3/stage3_jobD_vicreg

# ----- Stage 2: SMOKE (5 trainer steps; verify vicreg_active=True) -----
slack_notify ":wrench: vl-jobD-vicreg smoke (5 steps)"
echo "[$(date)] [smoke] trainer_monet_stage3.py --smoke (vicreg)"

uv --project cluster run --no-sync \
  accelerate launch \
    --config_file cluster/accelerate_zero2.yaml \
    --num_machines=1 --num_processes=4 --mixed_precision=bf16 \
    cluster/trainer_monet_stage3.py \
      --config cluster/configs/stage3_monet_jobD_vicreg.yaml \
      --out_dir ${S3_OUT}/_smoke \
      --smoke
SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then
  slack_notify ":x: vl-jobD-vicreg smoke FAILED rc=${SMOKE_RC}"; exit 13
fi
if ! grep -q "\[mem-after-prepare\] dist_type=DEEPSPEED" "$LOGFILE"; then
  slack_notify ":x: vl-jobD-vicreg dist_type!=DEEPSPEED"; exit 14
fi
if ! grep -q "\[stage3-check\] alignment_active=True" "$LOGFILE"; then
  slack_notify ":x: vl-jobD-vicreg alignment not active"; exit 15
fi
# Gate: VICReg must be ON for this arm.
if ! grep -q "\[vicreg-check\] vicreg_active=True" "$LOGFILE"; then
  slack_notify ":x: vl-jobD-vicreg vicreg_active!=True — config bug? config says lambda_reg=1.0 but smoke doesn't confirm. ABORTING."
  exit 16
fi
echo "[$(date)] [smoke] OK ($(elapsed_h)): DEEPSPEED + alignment_active + vicreg_active confirmed"
slack_notify ":white_check_mark: vl-jobD-vicreg smoke OK at $(elapsed_h) — proceeding to full training"

# ----- Stage 3: full Stage 3 + VICReg training -----
slack_notify ":dart: vl-jobD-vicreg training (2000 steps) START at $(elapsed_h)"
echo "[$(date)] [train] trainer_monet_stage3.py full run (vicreg)"

uv --project cluster run --no-sync \
  accelerate launch \
    --config_file cluster/accelerate_zero2.yaml \
    --num_machines=1 --num_processes=4 --mixed_precision=bf16 \
    cluster/trainer_monet_stage3.py \
      --config cluster/configs/stage3_monet_jobD_vicreg.yaml
TRAIN_RC=$?

S3_CKPT=${S3_OUT}/checkpoint
TOTAL_H=$(elapsed_h)
if [ $TRAIN_RC -eq 0 ]; then
  slack_notify ":tada: vl-jobD-vicreg DONE in ${TOTAL_H} | ckpt=${S3_CKPT}
NEXT: eval VICReg vs baseline (local A6000). Predict mean_cos drops from ~0.87 (baseline) → ~0.40 (Stage 2 healthy signature) and n_helpful rises."
  exit 0
else
  slack_notify ":warning: vl-jobD-vicreg exit=${TRAIN_RC} in ${TOTAL_H}"
  exit $TRAIN_RC
fi
