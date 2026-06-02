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
#   Both give 977 optimizer steps/epoch (125072 / 128), so the linear LR
#   decay horizon (max_steps) is IDENTICAL to the released 8-GPU recipe.
#
# EPOCHS = 3. This matches the recipe that produced the *released* stage3
# checkpoint: git archaeology on NOVAglow646/Monet shows sft_stage3.sh had
# `--epochs 3` at the model-upload commit (f49f382, 2025-12-30); it was
# only changed to `--epochs 2` LATER (ebcaba8, 2026-02-04), after upload.
# All training code (src/, monet_qwen_model/) is byte-identical across that
# boundary, so epochs is the dominant difference. Our prior epochs=2 run
# (cut at epoch 1.77) did NOT reproduce the collapse (cos 0.42 vs released
# 0.87); epochs=3 is the faithful recipe under test.
#
# 24h walltime cap < ~40h needed for 3 epochs (~2913 steps @ ~48s/step),
# so this runs as TWO jobs sharing one epochs=3 schedule:
#   Job 1 (fresh):  RESUME=0 (default) → train from stage1, save ckpts.
#   Job 2 (resume): RESUME=1 → --resume_from_checkpoint auto-finds the
#                   latest ckpt in SAVE_DIR and continues. This is
#                   consistent because BOTH jobs use epochs=3 (same
#                   max_steps=2913 horizon) — unlike resuming the old
#                   epochs=2 ckpt, which would mismatch the LR schedule.
#
# Hyperparams (verbatim upstream): bsz=1, latent_size=8, alignment_weight=2.0,
# ce_emphasize_factor=4.0, alignment_layer=all_layers, lr=1e-5 (default),
# warmup_steps=10, weight_decay=0.01, linear scheduler, save_steps=250,
# stage=sft_stage3. NB: upstream's script_examples/sft_stage3.sh has a
# stale --stage "avt_v5_stage2" left over from before the choices were
# renamed in src/utils.py. src/main.py:349-351 routes 'sft_stage3' to
# CustomTrainerSFT_STAGE3 — the class our VICReg patch lives in.
#
# Init: released Monet-SFT-7B/stage1 (upstream's published Stage 1 ckpt).
# Teacher latents: /data/.../upstream_teacher_latents/ (produced by
#                  slurm/upstream_precompute.sbatch with upstream's precompute).
#
# Submit Job 1 (baseline, fresh):  ARM=baseline           sb slurm/upstream_stage3.sbatch
# Submit Job 2 (baseline, resume): ARM=baseline RESUME=1  sb slurm/upstream_stage3.sbatch
# Submit vicreg arm later:         ARM=vicreg             sb slurm/upstream_stage3.sbatch
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
# RESUME=1 → continue the SAME epochs=3 run from its latest checkpoint
# (Job 2 of the 2-job split). Default 0 = fresh run (Job 1).
RESUME="${RESUME:-0}"
EPOCHS=3
# _ep3 keeps this distinct from the prior epochs=2 run's checkpoints.
RUN_NAME="upstream_stage3_${ARM}_ep3"

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
mkdir -p "$SAVE_DIR"  # upstream src/main.py:37 opens --log_file directly; parent must exist

# ----- Stage 1b: resume gate -----
# Count existing HF Trainer checkpoints in SAVE_DIR.
N_CKPT=$(find "$SAVE_DIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | wc -l)
RESUME_FLAG=""
if [ "$RESUME" = "1" ]; then
  if [ "$N_CKPT" -lt 1 ]; then
    slack_notify ":x: upstream-stage3 $ARM RESUME=1 but no checkpoints in $SAVE_DIR — run Job 1 (RESUME=0) first."
    exit 16
  fi
  RESUME_FLAG="--resume_from_checkpoint"
  echo "[$(date)] RESUME=1 → --resume_from_checkpoint (latest of $N_CKPT ckpts in $SAVE_DIR)"
else
  # Fresh run must not silently clobber an in-progress run's checkpoints.
  if [ "$N_CKPT" -ge 1 ]; then
    slack_notify ":x: upstream-stage3 $ARM RESUME=0 (fresh) but $SAVE_DIR already has $N_CKPT checkpoints. Refusing to clobber — set RESUME=1 to continue, or remove the dir to restart."
    exit 17
  fi
  echo "[$(date)] RESUME=0 → fresh run from $INIT_CKPT"
fi

# ----- Stage 2: TRAINING (upstream src.main, verbatim flags + --lambda_reg) -----
slack_notify ":dart: upstream-stage3 $ARM ep$EPOCHS training START at $(elapsed_h) (RESUME=$RESUME)"
echo "[$(date)] [train] torchrun -m src.main --stage sft_stage3 --epochs $EPOCHS --lambda_reg $LAMBDA_REG $RESUME_FLAG"

torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
  --epochs $EPOCHS \
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
  --lambda_reg $LAMBDA_REG \
  $RESUME_FLAG
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
