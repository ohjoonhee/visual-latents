#!/bin/bash
# Faithful upstream Monet — Phase 3/5 STAGE 3 TRAINING.
#
# Runs upstream's src.main UNMODIFIED with the optional --lambda_reg
# (default 0.0 = upstream baseline). Two arms select via env:
#   ARM=baseline → --lambda_reg 0  (Phase 3 — confirm faithful Stage 3
#                                   reproduces released collapse cos~0.87)
#   ARM=vicreg   → --lambda_reg 1  (Phase 5 — the actual VICReg experiment)
#
# Code lives at: ~/Monet (vicreg-patch branch; patch = 35 insertions across
# src/utils.py + src/trainer.py adding lambda_reg flag + guarded VICReg term).
# Branch identity is verified at smoke; mismatch → exit.
#
# Adaptation from upstream's sft_stage3.sh step 2 (only one deviation):
#   upstream: --nproc-per-node=8 --grad_accum_steps=16  → eff bsz 128
#   ours:     --nproc-per-node=4 --grad_accum_steps=32  → eff bsz 128
#
# Hyperparams (verbatim upstream): epochs=2, bsz=1, latent_size=8,
# alignment_weight=2.0, ce_emphasize_factor=4.0, alignment_layer=all_layers,
# stage=sft_stage3. NB: upstream's script_examples/sft_stage3.sh has a
# stale --stage "avt_v5_stage2" left over from before the choices were
# renamed in src/utils.py. src/main.py:349-351 routes 'sft_stage3' to
# CustomTrainerSFT_STAGE3 — the class our VICReg patch lives in.
#
# Init: released Monet-SFT-7B/stage1 (upstream's published Stage 1 ckpt).
# Teacher latents: /data/.../upstream_teacher_latents/ (produced by
#                  slurm/upstream_precompute.sbatch with upstream's precompute).
#
# Submit (baseline): ARM=baseline sb slurm/upstream_stage3.sbatch
# Submit (vicreg):   ARM=vicreg   sb slurm/upstream_stage3.sbatch
# DO NOT agent-submit (project-owner rule §3-A).
#
#SBATCH -J launch
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

# ARM is set by the submitter via env. Default to baseline.
ARM="${ARM:-baseline}"
case "$ARM" in
  baseline) LAMBDA_REG=0.0 ;;
  vicreg)   LAMBDA_REG=1.0 ;;
  *) echo "ARM must be baseline or vicreg, got: $ARM"; exit 10 ;;
esac
RUN_NAME="upstream_stage3_${ARM}"

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

slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}

LOGFILE="$SLURM_SUBMIT_DIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
JOB_T0=$(date +%s)
elapsed_h () { printf "%.1fh" "$(echo "scale=1; ($(date +%s) - ${JOB_T0}) / 3600.0" | bc)"; }

slack_notify ":rocket: START upstream-stage3 ARM=$ARM | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
LAMBDA_REG=$LAMBDA_REG  RUN_NAME=$RUN_NAME
init: Monet-SFT-7B/stage1  teacher: upstream_teacher_latents/
log: \`${ABSLOG}\`"

# ----- Stage 0: env + branch -----
MONET=$HOME/Monet
[ ! -d "$MONET/.venv" ] && { slack_notify ":x: $MONET/.venv missing"; exit 11; }
source $MONET/.venv/bin/activate
cd $MONET
BRANCH=$(git rev-parse --abbrev-ref HEAD)
[ "$BRANCH" != "vicreg-patch" ] && { slack_notify ":x: branch=$BRANCH expected vicreg-patch"; exit 12; }
HEAD_SHA=$(git rev-parse --short HEAD)
echo "[$(date)] ~/Monet @ vicreg-patch ($HEAD_SHA)"

# Smoke that --lambda_reg flag exists.
if ! python -m src.main --help 2>&1 | grep -q lambda_reg; then
  slack_notify ":x: src.main --lambda_reg flag missing (patch not applied?)"; exit 13
fi

# ----- Stage 1: precondition checks -----
INIT_CKPT=/data/joonhee/visual-latents/cluster_phase3/monet_sft_7b/stage1
TEACHER_DIR=/data/joonhee/visual-latents/cluster_phase3/upstream_teacher_latents
[ ! -d "$INIT_CKPT" ] && { slack_notify ":x: stage1 init missing at $INIT_CKPT"; exit 14; }
N_TEACHER=$(find "$TEACHER_DIR" -maxdepth 1 -name 'latent_all_layers_*.pt' -type f 2>/dev/null | wc -l)
if [ "$N_TEACHER" -lt 100000 ]; then
  slack_notify ":x: upstream_teacher_latents incomplete: $N_TEACHER files (need >=100K). Run upstream_precompute first."
  exit 15
fi
echo "[$(date)] init OK; teacher_latents: $N_TEACHER files"

DATA_ROOT=/data/joonhee/visual-latents/data/Monet-SFT-125K
SAVE_DIR=/data/joonhee/visual-latents/cluster_phase3/$RUN_NAME

# ----- Stage 2: TRAINING (upstream src.main, verbatim flags + --lambda_reg) -----
slack_notify ":dart: upstream-stage3 $ARM training START at $(elapsed_h)"
echo "[$(date)] [train] torchrun -m src.main --stage sft_stage3 --lambda_reg $LAMBDA_REG"

torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
  --epochs 2 \
  --bsz 1 \
  --grad_accum_steps 32 \
  --stage sft_stage3 \
  --data_path \
    "$DATA_ROOT/Visual_CoT/train.json" \
    "$DATA_ROOT/CogCoM/train.json" \
    "$DATA_ROOT/ReFocus/train.json" \
    "$DATA_ROOT/Zebra_CoT_count/train.json" \
    "$DATA_ROOT/Zebra_CoT_visual_search/train.json" \
    "$DATA_ROOT/Zebra_CoT_geometry/train.json" \
  --log_file "$SAVE_DIR/log.txt" \
  --load_model_path "$INIT_CKPT" \
  --save_model_path "$SAVE_DIR" \
  --dataset_root "$DATA_ROOT" \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --wandb_name "$RUN_NAME" \
  --latent_size 8 \
  --alignment_weight 2.0 \
  --ce_emphasize_factor 4.0 \
  --teacher_latent_dir "$TEACHER_DIR" \
  --alignment_layer all_layers \
  --lambda_reg $LAMBDA_REG
RC=$?

TOTAL_H=$(elapsed_h)
if [ $RC -eq 0 ]; then
  slack_notify ":tada: upstream-stage3 $ARM DONE in $TOTAL_H | ckpt: $SAVE_DIR
next: upload + local eval (probe pairwise cos)"
  exit 0
else
  slack_notify ":warning: upstream-stage3 $ARM exit=$RC in $TOTAL_H | partial ckpt at $SAVE_DIR"
  exit $RC
fi
