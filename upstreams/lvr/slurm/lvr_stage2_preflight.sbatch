#!/bin/bash
# LVR Stage-2 GRPO CPU PREFLIGHT (cpu-short, NO GPU). Runs slurm/preflight_stage2.py
# against the SAME env, checkpoint, data and pixel bounds as lvr_stage2_3b.sbatch, to
# shake out GPU-independent breakage (imports/math_verify, arg/schema drift, the
# seed_worker signature, unresolved ViRL39K image paths, reward/math_verify faults)
# BEFORE spending a 4xH100 slot. Green here => the non-GPU surface is clean; the only
# things left to fail on GPU are the model forward / generate / DeepSpeed / OOM.
#   Submit:  sb slurm/lvr_stage2_preflight.sbatch      # DO NOT agent-submit (§3-A)
#   Wider image sweep:  PREFLIGHT_N=512 sb slurm/lvr_stage2_preflight.sbatch
#SBATCH -J launch
#SBATCH -p cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 0-01:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log
set -uo pipefail
[ -f .env ] && { set -a; source .env; set +a; }
slack_notify(){ [ -z "${SLACK_WEBHOOK_URL:-}" ]&&return 0; curl -s -X POST -H 'Content-type: application/json' --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL">/dev/null; }
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO" || exit 11
[ -d .venv ] || { slack_notify ":x: $REPO/.venv missing"; exit 12; }
source .venv/bin/activate
# cuda module only so `import flash_attn` resolves its libs; we never touch a GPU.
module purge; module load cuda/12.6; export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.6
export CUDA_VISIBLE_DEVICES=""        # force CPU — this is a non-GPU surface check
ROOT=/data/joonhee/lvr
mkdir -p $ROOT/.cache/{tmp,triton,torch,torch-extensions}
export TMPDIR=$ROOT/.cache/tmp TRITON_CACHE_DIR=$ROOT/.cache/triton TORCH_HOME=$ROOT/.cache/torch TORCH_EXTENSIONS_DIR=$ROOT/.cache/torch-extensions
# same dual PYTHONPATH as the training job (code mixes `from src...` and `from train...`)
export PYTHONPATH=$REPO:$REPO/src:${PYTHONPATH:-}
export WANDB_MODE=disabled            # preflight must never open a wandb run

# auto-find the Stage-1 init exactly like lvr_stage2_3b.sbatch (full path = checkpoint-2500)
STAGE1=$(find $ROOT/ckpts/stage1 -maxdepth 3 -type d -name 'checkpoint-2500' 2>/dev/null | head -1)
[ -z "$STAGE1" ] && { slack_notify ":x: preflight: stage1 checkpoint-2500 not found under $ROOT/ckpts/stage1"; exit 14; }
slack_notify ":mag: START lvr-stage2-PREFLIGHT (cpu) | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME} | init=$STAGE1"

# pixel bounds identical to the training job: 128*28*28 .. 2560*28*28
python slurm/preflight_stage2.py \
  --checkpoint "$STAGE1" \
  --data_path "$ROOT/data/virl39k.json" \
  --image_folder "$ROOT/images/" \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --image_min_pixels $((128*28*28)) \
  --image_max_pixels $((2560*28*28)) \
  --deepspeed_config scripts/zero2.json \
  --num_workers 8
rc=$?
[ $rc -eq 0 ] && slack_notify ":white_check_mark: lvr-stage2-PREFLIGHT GREEN — non-GPU surface clean, safe to submit GPU job" \
              || slack_notify ":x: lvr-stage2-PREFLIGHT FAILED rc=$rc — fix before spending a GPU slot (see logs/launch_${SLURM_JOB_ID}.log)"
exit $rc
