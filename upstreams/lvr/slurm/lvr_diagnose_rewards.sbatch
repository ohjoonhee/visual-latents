#!/bin/bash
# LVR Stage-2 reward diagnosis (cpu-short, NO GPU). Runs slurm/diagnose_rewards.py to show
# WHY Stage-2 GRPO trained with reward==0 (format_reward never fired): generates a real
# completion on CPU and runs the actual reward funcs on it, printing the decoded text + gold.
#   Submit:  sb slurm/lvr_diagnose_rewards.sbatch          # DO NOT agent-submit (§3-A)
#SBATCH -J launch
#SBATCH -p cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -t 0-02:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log
set -uo pipefail
[ -f .env ] && { set -a; source .env; set +a; }
slack_notify(){ [ -z "${SLACK_WEBHOOK_URL:-}" ]&&return 0; curl -s -X POST -H 'Content-type: application/json' --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL">/dev/null; }
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO" || exit 11
[ -d .venv ] || { slack_notify ":x: $REPO/.venv missing"; exit 12; }
source .venv/bin/activate
module purge; module load cuda/12.6; export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.6
export CUDA_VISIBLE_DEVICES=""
ROOT=/data/joonhee/lvr
mkdir -p $ROOT/.cache/{tmp,triton,torch,torch-extensions}
export TMPDIR=$ROOT/.cache/tmp TRITON_CACHE_DIR=$ROOT/.cache/triton TORCH_HOME=$ROOT/.cache/torch TORCH_EXTENSIONS_DIR=$ROOT/.cache/torch-extensions
export PYTHONPATH=$REPO:$REPO/src:$REPO/src/train:${PYTHONPATH:-}
export WANDB_MODE=disabled
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

# CKPT: default the SFT init (what the rollout starts from); override with CKPT=... to inspect an RL ckpt
CKPT="${CKPT:-$(find $ROOT/ckpts/stage1 -maxdepth 3 -type d -name 'checkpoint-2500' 2>/dev/null | head -1)}"
[ -z "$CKPT" ] && { slack_notify ":x: reward-diagnosis: checkpoint not found"; exit 14; }
slack_notify ":microscope: START lvr-reward-diagnosis (cpu) | id=${SLURM_JOB_ID} | ckpt=$CKPT"

python slurm/diagnose_rewards.py \
  --checkpoint "$CKPT" \
  --data_path "$ROOT/data/virl39k.json" \
  --image_folder "$ROOT/images/" \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --image_min_pixels $((128*28*28)) \
  --image_max_pixels $((2560*28*28)) \
  --lvr_steps 8 \
  --max_new_tokens "${MAXNEW:-160}" \
  --n_prompts "${NPROMPTS:-4}"
rc=$?
[ $rc -eq 0 ] && slack_notify ":white_check_mark: lvr-reward-diagnosis done (see logs/launch_${SLURM_JOB_ID}.log)" \
              || slack_notify ":x: lvr-reward-diagnosis rc=$rc"
exit $rc
