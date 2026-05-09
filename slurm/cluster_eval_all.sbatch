#!/bin/bash
# Cluster Phase 3 — eval all checkpoints (Stage 1 SFT, Stage 2 baseline, V1, V2, V3).
#
# Submit with: sb slurm/cluster_eval_all.sbatch
# DO NOT agent-submit. REQUIRES the four training jobs to have completed.
#
# eval.py loads each checkpoint in turn (single-GPU per ckpt; we use 1 of the
# 4 H100s — the other 3 sit idle to satisfy the 4-GPU partition rule). It
# extracts latents on a 200-example held-out and runs ablation × 2 readers
# (self + qwen_base 7B). Per-ckpt REPORT.md + cross-ckpt CROSS_REPORT.md.
#
# To add/remove ckpts, edit the CKPTS array below before submission.
#
# Cost: 4 GPUs × ~6 h = 24 GPU-hours.
#
#SBATCH -J vl-eval-all
#SBATCH -p gpu-4farm
#SBATCH --account=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH -t 8:00:00
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
slack_notify ":mag: START vl-eval-all | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
script: \`${SCRIPT_PATH}\`
log:    \`${ABSLOG}\`"

# Hardcoded ckpt list — edit BEFORE submission to drop/add variants.
PHASE3=/data/joonhee/visual-latents/cluster_phase3
CKPTS=(
  "stage1_sft:${PHASE3}/stage1_sft/checkpoint"
  "stage2_baseline:${PHASE3}/stage2_baseline/checkpoint"
  "v1_g_at_7b:${PHASE3}/v1_g_at_7b/checkpoint"
  "v2_d2_at_7b:${PHASE3}/v2_d2_at_7b/checkpoint"
  "v3_monet_plus_vicreg:${PHASE3}/v3_monet_plus_vicreg/checkpoint"
)

# Sanity: drop missing ckpts so we don't fail the whole eval if one variant didn't run.
EXISTING=()
for spec in "${CKPTS[@]}"; do
  path="${spec##*:}"
  if [ -d "$path" ]; then
    EXISTING+=("$spec")
  else
    echo "WARN: skipping missing ckpt: $spec" >&2
  fi
done
if [ ${#EXISTING[@]} -eq 0 ]; then
  slack_notify ":x: FAIL vl-eval-all — no ckpts found"
  echo "FATAL: zero ckpts found among ${CKPTS[*]}" >&2
  exit 2
fi

OUT_ROOT="${PHASE3}/_eval_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT_ROOT"
echo "[$(date)] eval out_root: $OUT_ROOT"
echo "[$(date)] ckpts: ${EXISTING[*]}"

# eval.py is single-GPU per ckpt — we don't use accelerate launch here.
# The 4 H100s are still allocated (partition requires it); 3 sit idle.
set +e
uv --project cluster run --no-sync python cluster/eval.py \
  --ckpts "${EXISTING[@]}" \
  --out_root "$OUT_ROOT" \
  --qwen_base Qwen/Qwen2.5-VL-7B-Instruct \
  --n_eval 200
rc=$?
set -e

if [ $rc -eq 0 ]; then
  slack_notify ":white_check_mark: DONE vl-eval-all (jobid=${SLURM_JOB_ID}) → ${OUT_ROOT}/CROSS_REPORT.md"
else
  slack_notify ":x: FAIL vl-eval-all (jobid=${SLURM_JOB_ID}) exit=${rc}"
fi
exit $rc
