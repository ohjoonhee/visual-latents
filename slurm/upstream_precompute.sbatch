#!/bin/bash
# Faithful upstream Monet — Phase 2 PRECOMPUTE.
#
# Runs upstream's src.precompute_teacher_latents UNMODIFIED with the released
# Monet-SFT-7B/stage2 as the teacher. Output is the per-sample all-layer
# teacher latent .pt files that upstream's CustomTrainerSFT_STAGE3 expects.
#
# Code lives at: ~/Monet (cloned from github.com/NOVAglow646/Monet, branch
# `vicreg-patch` — patch only adds --lambda_reg/--reg_gamma CLI flags to
# get_args + a guarded VICReg term to STAGE3 compute_loss; default off,
# behavior identical to upstream for precompute (precompute doesn't use
# the trainer at all)).
#
# Cluster adaptation (only deviation from upstream's sft_stage3.sh step 1):
#   - Upstream: torchrun --nproc-per-node=8 (8 GPUs)
#   - Ours:     torchrun --nproc-per-node=4 (gpu-4farm gives whole 4-GPU node)
#   - Same per-rank workload; just half the parallelism. Walltime 24h.
#
# Hyperparams (verbatim upstream sft_stage3.sh step 1):
#   --bsz 1, --latent_size 8, --output_hidden_states (= all_layers caching),
#   --resume (idempotent per-sample save)
#
# Submit with: sb slurm/upstream_precompute.sbatch
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

module purge
module load cuda/12.6
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.6

mkdir -p /data/joonhee/visual-latents/.cache/{tmp,triton,torch,torch-extensions} \
         /data/joonhee/visual-latents/cluster_phase3/upstream_teacher_latents
export TMPDIR=/data/joonhee/visual-latents/.cache/tmp
export TRITON_CACHE_DIR=/data/joonhee/visual-latents/.cache/triton
export TORCH_HOME=/data/joonhee/visual-latents/.cache/torch
export TORCH_EXTENSIONS_DIR=/data/joonhee/visual-latents/.cache/torch-extensions
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

slack_notify ":rocket: START upstream-precompute | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
Upstream Monet src.precompute_teacher_latents (4 GPUs, ~6h expected)
teacher: /data/joonhee/visual-latents/cluster_phase3/monet_sft_7b/stage2
output:  /data/joonhee/visual-latents/cluster_phase3/upstream_teacher_latents/
log:     \`${ABSLOG}\`"

# ----- Stage 0: env check -----
MONET=$HOME/Monet
if [ ! -d "$MONET" ] || [ ! -d "$MONET/.venv" ]; then
  slack_notify ":x: upstream-precompute: ~/Monet or ~/Monet/.venv missing. Run setup first."
  exit 11
fi
echo "[$(date)] activating uv env at $MONET/.venv"
source $MONET/.venv/bin/activate
python -c "import transformers; print('transformers:', transformers.__version__)"

# Verify branch is vicreg-patch (so the trainer has --lambda_reg even if
# precompute itself doesn't use it).
cd $MONET
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "[$(date)] ~/Monet branch: $BRANCH"
if [ "$BRANCH" != "vicreg-patch" ]; then
  slack_notify ":x: upstream-precompute: ~/Monet not on vicreg-patch branch ($BRANCH)"; exit 12
fi

# ----- Stage 1: precondition checks -----
TEACHER_CKPT=/data/joonhee/visual-latents/cluster_phase3/monet_sft_7b/stage2
if [ ! -d "$TEACHER_CKPT" ]; then
  slack_notify ":x: upstream-precompute: teacher ckpt MISSING at $TEACHER_CKPT"; exit 13
fi
DATA_ROOT=/data/joonhee/visual-latents/data/Monet-SFT-125K
for SUBSET in Visual_CoT CogCoM ReFocus Zebra_CoT_count Zebra_CoT_visual_search Zebra_CoT_geometry; do
  if [ ! -f "$DATA_ROOT/$SUBSET/train.json" ]; then
    slack_notify ":x: upstream-precompute: dataset MISSING: $DATA_ROOT/$SUBSET/train.json"; exit 13
  fi
done
echo "[$(date)] teacher + 6 datasets OK"

OUT_DIR=/data/joonhee/visual-latents/cluster_phase3/upstream_teacher_latents
AVAIL_GB=$(df -BG /data/joonhee/visual-latents/cluster_phase3 | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAIL_GB" -lt 250 ]; then
  slack_notify ":x: upstream-precompute: /data has only ${AVAIL_GB}G free"; exit 14
fi
echo "[$(date)] /data free: ${AVAIL_GB}G"

# ----- Stage 2: SMOKE (--num_samples 8, ~3min) -----
SMOKE_OUT=$OUT_DIR/_smoke
mkdir -p "$SMOKE_OUT"
echo "[$(date)] [smoke] precompute --num_samples 8 (4 ranks)"
slack_notify ":wrench: upstream-precompute smoke (8 samples, 4 ranks)"

torchrun --nproc-per-node=4 --master-port=29501 -m src.precompute_teacher_latents \
  --bsz 1 \
  --num_samples 8 \
  --data_path \
    "$DATA_ROOT/Visual_CoT/train.json" \
    "$DATA_ROOT/CogCoM/train.json" \
    "$DATA_ROOT/ReFocus/train.json" \
    "$DATA_ROOT/Zebra_CoT_count/train.json" \
    "$DATA_ROOT/Zebra_CoT_visual_search/train.json" \
    "$DATA_ROOT/Zebra_CoT_geometry/train.json" \
  --load_model_path "$TEACHER_CKPT" \
  --save_model_path "$SMOKE_OUT" \
  --dataset_root "$DATA_ROOT" \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size 8 \
  --output_hidden_states
SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then
  slack_notify ":x: upstream-precompute SMOKE FAILED rc=${SMOKE_RC}"; exit 15
fi
N_SMOKE=$(find "$SMOKE_OUT" -maxdepth 1 -name 'latent_all_layers_*.pt' -type f | wc -l)
if [ "$N_SMOKE" -lt 4 ]; then
  slack_notify ":x: upstream-precompute smoke: only $N_SMOKE .pt files (need >=4)"; exit 16
fi
echo "[$(date)] smoke OK: $N_SMOKE .pt files in _smoke/ ($(elapsed_h))"
slack_notify ":white_check_mark: upstream-precompute smoke OK ($N_SMOKE files) — full run START"

# ----- Stage 3: full precompute -----
echo "[$(date)] [full] precompute_teacher_latents full pass with --resume"
torchrun --nproc-per-node=4 --master-port=29501 -m src.precompute_teacher_latents \
  --bsz 1 \
  --data_path \
    "$DATA_ROOT/Visual_CoT/train.json" \
    "$DATA_ROOT/CogCoM/train.json" \
    "$DATA_ROOT/ReFocus/train.json" \
    "$DATA_ROOT/Zebra_CoT_count/train.json" \
    "$DATA_ROOT/Zebra_CoT_visual_search/train.json" \
    "$DATA_ROOT/Zebra_CoT_geometry/train.json" \
  --load_model_path "$TEACHER_CKPT" \
  --save_model_path "$OUT_DIR" \
  --dataset_root "$DATA_ROOT" \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size 8 \
  --output_hidden_states \
  --resume
RC=$?

N_FILES=$(find "$OUT_DIR" -maxdepth 1 -name 'latent_all_layers_*.pt' -type f | wc -l)
TOTAL_H=$(elapsed_h)
if [ $RC -eq 0 ]; then
  slack_notify ":tada: upstream-precompute DONE in ${TOTAL_H} | n_files=${N_FILES}
out: $OUT_DIR
next: slurm/upstream_stage3.sbatch (needs explicit \"go\")"
  exit 0
else
  slack_notify ":warning: upstream-precompute exit=${RC} in ${TOTAL_H} | n_files=${N_FILES} (partial)"
  exit $RC
fi
