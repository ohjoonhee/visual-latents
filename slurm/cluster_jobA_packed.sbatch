#!/bin/bash
# Cluster Phase 3 — JOB A (packed): Stage 1 SFT → V1 (G@7B) → eval V1.
#
# Single 24h allocation, three sequential stages. Maximizes one launched
# GPU job: produces a usable Stage 1 SFT base, the highest-info Pivot A
# variant (V1 = G recipe at 7B with K=4, λ_reg=2), AND its eval — all
# from one queue-wait.
#
# Estimated wall-budget breakdown (24h cap):
#   - venv sync (first-run only): 5-10 min
#   - Stage 1 SFT (~922 steps, eff_bsz=128, 7B): 8-10 h
#   - V1 G@7B (~1844 steps, eff_bsz=128, 7B): 7-9 h
#   - eval V1 (extract + ablation, 200 examples): 30-60 min
#   - slack: 4-7 h
#
# Failure handling:
#   - venv sync fails → exit
#   - Stage 1 fails → exit (V1 cannot proceed without stage1_ckpt)
#   - V1 fails → STILL try eval on whatever ckpt exists (may be a partial
#     save), then exit with V1's exit code so Slack reports failure
#   - eval fails → exit with eval's code; ckpts are still on disk and
#     re-evaluable in a follow-up CPU/eval-only job
#
# Submit with: sb slurm/cluster_jobA_packed.sbatch
# DO NOT agent-submit (project-owner rule §3-A).
#
#SBATCH -J vl-jobA-packed
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
# `module load cuda/12.6` adds nvcc to PATH but does NOT export CUDA_HOME.
# DeepSpeed's import-time CUDA-op compatibility check needs CUDA_HOME.
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.6

# Mandatory cache redirects (§3-G).
mkdir -p /data/joonhee/visual-latents/.cache/{tmp,triton,torch,torch-extensions} \
         /data/joonhee/visual-latents/wandb \
         /data/joonhee/visual-latents/cluster_phase3
export TMPDIR=/data/joonhee/visual-latents/.cache/tmp
export TRITON_CACHE_DIR=/data/joonhee/visual-latents/.cache/triton
export TORCH_HOME=/data/joonhee/visual-latents/.cache/torch
export TORCH_EXTENSIONS_DIR=/data/joonhee/visual-latents/.cache/torch-extensions
export WANDB_DIR=/data/joonhee/visual-latents/wandb
# HF_HOME inherited from ~/.bashrc.
# 214425 hit OOM on every example (~75 GB baseline). Reducing fragmentation
# helps; the bigger fix was dropping --multi_gpu (was forcing DDP fallback
# instead of the YAML's ZeRO-2 path).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"

# ----- Slack notify (graceful if SLACK_WEBHOOK_URL unset) -----
slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}

LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
SCRIPT_PATH="${ORIG_SBATCH_SCRIPT:-${SLURM_SUBMIT_SCRIPT:-<unknown>}}"
slack_notify ":rocket: START vl-jobA-packed | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
Stage 1 SFT → V1 G@7B → eval V1
script: \`${SCRIPT_PATH}\`
log:    \`${ABSLOG}\`
follow: \`tail -n 200 -F ${ABSLOG}\`"

JOB_T0=$(date +%s)
elapsed_h () { printf "%.1fh" "$(echo "scale=1; ($(date +%s) - ${JOB_T0}) / 3600.0" | bc)"; }

# ----- Stage 0: venv sync (first-run only) -----
echo "[$(date)] [stage0] uv --project cluster sync"
uv --project cluster sync 2>&1 | tail -20
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  slack_notify ":x: vl-jobA stage0/uv-sync FAILED (jobid=${SLURM_JOB_ID})"
  exit 11
fi
echo "[$(date)] [stage0] venv sync OK ($(elapsed_h))"

# ----- Stage 1: SFT -----
slack_notify ":hammer_and_wrench: vl-jobA stage1 SFT START at $(elapsed_h)"
echo "[$(date)] [stage1] Stage 1 SFT — Qwen2.5-VL-7B + Monet-SFT-125K Visual_CoT"

uv --project cluster run --no-sync \
  accelerate launch \
    --config_file cluster/accelerate_zero2.yaml \
    --num_machines=1 --num_processes=4 --mixed_precision=bf16 \
    cluster/trainer_sft.py \
      --config cluster/configs/stage1_sft.yaml
STAGE1_RC=$?

if [ $STAGE1_RC -ne 0 ]; then
  slack_notify ":x: vl-jobA stage1 SFT FAILED rc=${STAGE1_RC} at $(elapsed_h)"
  exit $STAGE1_RC
fi

STAGE1_CKPT=/data/joonhee/visual-latents/cluster_phase3/stage1_sft/checkpoint
if [ ! -d "$STAGE1_CKPT" ]; then
  slack_notify ":x: vl-jobA stage1 ckpt MISSING at ${STAGE1_CKPT}"
  exit 12
fi
slack_notify ":white_check_mark: vl-jobA stage1 SFT DONE at $(elapsed_h) — ckpt at ${STAGE1_CKPT}"
echo "[$(date)] [stage1] DONE — ckpt at ${STAGE1_CKPT}"

# ----- Stage 2: V1 G@7B -----
slack_notify ":dart: vl-jobA V1 (G@7B K=4 λ_reg=2) START at $(elapsed_h)"
echo "[$(date)] [v1] V1 — Pivot A G recipe at 7B, init from ${STAGE1_CKPT}"

uv --project cluster run --no-sync \
  accelerate launch \
    --config_file cluster/accelerate_zero2.yaml \
    --num_machines=1 --num_processes=4 --mixed_precision=bf16 \
    cluster/trainer_pivot.py \
      --config cluster/configs/v1_g_at_7b.yaml
V1_RC=$?

V1_CKPT=/data/joonhee/visual-latents/cluster_phase3/v1_g_at_7b/checkpoint
if [ $V1_RC -ne 0 ]; then
  slack_notify ":warning: vl-jobA V1 trainer FAILED rc=${V1_RC} at $(elapsed_h) — will still attempt eval on any saved ckpt"
fi
echo "[$(date)] [v1] trainer exit=${V1_RC}"

# ----- Stage 3: eval V1 (best-effort even if V1 trainer failed) -----
if [ -d "$V1_CKPT" ]; then
  slack_notify ":mag: vl-jobA eval V1 START at $(elapsed_h)"
  echo "[$(date)] [eval] eval V1 ckpt at ${V1_CKPT}"

  # eval.py is single-GPU per ckpt — gpu-4farm gives us 4, only one used.
  uv --project cluster run --no-sync \
    python cluster/eval.py \
      --ckpts v1_g_at_7b:${V1_CKPT} \
      --K 4 \
      --n 200 \
      --data_root /data/joonhee/visual-latents/data/Monet-SFT-125K \
      --out_dir /data/joonhee/visual-latents/cluster_phase3/v1_g_at_7b/eval
  EVAL_RC=$?

  if [ $EVAL_RC -eq 0 ]; then
    slack_notify ":white_check_mark: vl-jobA eval V1 DONE at $(elapsed_h)"
  else
    slack_notify ":x: vl-jobA eval V1 FAILED rc=${EVAL_RC} at $(elapsed_h)"
  fi
else
  slack_notify ":x: vl-jobA V1 ckpt MISSING at ${V1_CKPT} — skipping eval"
  EVAL_RC=21
fi

# ----- Wrap up -----
TOTAL_H=$(elapsed_h)
if [ $V1_RC -eq 0 ] && [ $EVAL_RC -eq 0 ]; then
  slack_notify ":tada: vl-jobA-packed DONE in ${TOTAL_H} | stage1 ✅ V1 ✅ eval ✅"
  exit 0
elif [ $V1_RC -eq 0 ]; then
  slack_notify ":warning: vl-jobA-packed PARTIAL in ${TOTAL_H} | stage1 ✅ V1 ✅ eval ❌"
  exit $EVAL_RC
else
  slack_notify ":warning: vl-jobA-packed PARTIAL in ${TOTAL_H} | stage1 ✅ V1 ❌ (rc=${V1_RC})"
  exit $V1_RC
fi
