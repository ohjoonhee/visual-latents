#!/bin/bash
# Cluster Phase 3 — JOB A2: finish V1 G@7B + eval V1.
#
# Continuation of Job A 214464 which hit 24h walltime at V1 step ~1410
# (70%) and never reached eval. trainer_pivot.py's --init_ckpt is
# weights-only (no optimizer / scheduler / step counter restore), so a
# "clean resume" from V1 checkpoint_step1000 is NOT supported by the
# existing code path. Choices considered:
#   (1) patch trainer_pivot.py to load optim+sched state — high risk
#       under ZeRO-2 sharding; not worth before any usable V1 result
#       has been produced
#   (2) warm-start from checkpoint_step1000 with fresh optimizer +
#       rerun max_steps=1000 — mixes "1000 trained + 1000 fresh-momentum"
#       which is methodologically muddier than the canonical recipe
#   (3) discard partial V1 and rerun fresh from Stage 1 ckpt with full
#       max_steps=2000 — wastes the killed run's ~9.5h compute but
#       gives the canonical 2000-step result the Phase 3 plan intended
#
# This sbatch picks (3): the canonical clean run. cluster_phase3/
# v1_g_at_7b/ from the killed run is moved aside (manually, before
# submit) so this fresh V1 writes to a clean directory.
#
# Wall-budget breakdown (24h cap):
#   - venv sync (likely no-op): <1 min
#   - V1 G@7B 2000 steps (~34s/step from Job A): ~19h
#   - V1 ckpt save (7B sharded): 2-3 min
#   - eval V1 (extract + ablation, 200 examples, single GPU): 30-60 min
#   - slack: 3-4 h
#
# Failure handling:
#   - venv sync fails → exit
#   - V1 fails → STILL try eval on whatever ckpt exists (intermediate
#     save_every=500 ckpts), then exit with V1's exit code
#   - eval fails → exit with eval's code; ckpts on disk are re-evaluable
#
# Submit with: sb slurm/cluster_jobA2_finish_v1.sbatch
# DO NOT agent-submit (project-owner rule §3-A).
#
#SBATCH -J vl-jobA2-finish-v1
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"

slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}

LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
SCRIPT_PATH="${ORIG_SBATCH_SCRIPT:-${SLURM_SUBMIT_SCRIPT:-<unknown>}}"
slack_notify ":rocket: START vl-jobA2-finish-v1 | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
V1 G@7B fresh from Stage 1 ckpt → eval V1
script: \`${SCRIPT_PATH}\`
log:    \`${ABSLOG}\`
follow: \`tail -n 200 -F ${ABSLOG}\`"

JOB_T0=$(date +%s)
elapsed_h () { printf "%.1fh" "$(echo "scale=1; ($(date +%s) - ${JOB_T0}) / 3600.0" | bc)"; }

# ----- Stage 0: venv sync (likely no-op) -----
echo "[$(date)] [stage0] uv --project cluster sync"
uv --project cluster sync 2>&1 | tail -10
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  slack_notify ":x: vl-jobA2 stage0/uv-sync FAILED (jobid=${SLURM_JOB_ID})"
  exit 11
fi
echo "[$(date)] [stage0] venv sync OK ($(elapsed_h))"

# ----- Stage 1 ckpt sanity check (NOT re-running Stage 1) -----
STAGE1_CKPT=/data/joonhee/visual-latents/cluster_phase3/stage1_sft/checkpoint
if [ ! -d "$STAGE1_CKPT" ]; then
  slack_notify ":x: vl-jobA2 Stage 1 ckpt MISSING at ${STAGE1_CKPT} — cannot start V1"
  exit 12
fi
echo "[$(date)] Stage 1 ckpt OK at ${STAGE1_CKPT} ($(du -sh $STAGE1_CKPT | cut -f1))"

# ----- V1 G@7B (fresh from Stage 1) -----
slack_notify ":dart: vl-jobA2 V1 (G@7B K=4 λ_reg=2, 2000 steps) START at $(elapsed_h)"
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
  slack_notify ":warning: vl-jobA2 V1 trainer FAILED rc=${V1_RC} at $(elapsed_h) — will still attempt eval on any saved ckpt"
fi
echo "[$(date)] [v1] trainer exit=${V1_RC}"

# ----- Eval V1 -----
# eval.py CLI (verified vs cluster/eval.py:525-530; Job A had wrong flags):
#   --ckpts <name>:<path>   --K <int>   --n_eval <int>   --out_root <dir>
EVAL_OUT=/data/joonhee/visual-latents/cluster_phase3/v1_g_at_7b/eval

if [ -d "$V1_CKPT" ]; then
  EVAL_TARGET=$V1_CKPT
  EVAL_LABEL="v1_g_at_7b"
elif [ -d "/data/joonhee/visual-latents/cluster_phase3/v1_g_at_7b/checkpoint_step1500" ]; then
  EVAL_TARGET=/data/joonhee/visual-latents/cluster_phase3/v1_g_at_7b/checkpoint_step1500
  EVAL_LABEL="v1_g_at_7b_step1500"
elif [ -d "/data/joonhee/visual-latents/cluster_phase3/v1_g_at_7b/checkpoint_step1000" ]; then
  EVAL_TARGET=/data/joonhee/visual-latents/cluster_phase3/v1_g_at_7b/checkpoint_step1000
  EVAL_LABEL="v1_g_at_7b_step1000"
else
  EVAL_TARGET=""
fi

if [ -n "$EVAL_TARGET" ]; then
  slack_notify ":mag: vl-jobA2 eval V1 START at $(elapsed_h) — target=${EVAL_LABEL}"
  echo "[$(date)] [eval] eval V1 ckpt at ${EVAL_TARGET}"

  uv --project cluster run --no-sync \
    python cluster/eval.py \
      --ckpts ${EVAL_LABEL}:${EVAL_TARGET} \
      --K 4 \
      --n_eval 200 \
      --out_root ${EVAL_OUT}
  EVAL_RC=$?

  if [ $EVAL_RC -eq 0 ]; then
    slack_notify ":white_check_mark: vl-jobA2 eval V1 DONE at $(elapsed_h) — out at ${EVAL_OUT}"
  else
    slack_notify ":x: vl-jobA2 eval V1 FAILED rc=${EVAL_RC} at $(elapsed_h)"
  fi
else
  slack_notify ":x: vl-jobA2 no V1 ckpt to eval — skipping"
  EVAL_RC=21
fi

# ----- Wrap up -----
TOTAL_H=$(elapsed_h)
if [ $V1_RC -eq 0 ] && [ $EVAL_RC -eq 0 ]; then
  slack_notify ":tada: vl-jobA2-finish-v1 DONE in ${TOTAL_H} | V1 ✅ eval ✅"
  exit 0
elif [ $EVAL_RC -eq 0 ]; then
  slack_notify ":warning: vl-jobA2 PARTIAL in ${TOTAL_H} | V1 ❌ (rc=${V1_RC}) eval ✅ (on ${EVAL_LABEL})"
  exit $V1_RC
else
  slack_notify ":warning: vl-jobA2 PARTIAL in ${TOTAL_H} | V1 rc=${V1_RC} eval ❌ (rc=${EVAL_RC})"
  exit $EVAL_RC
fi
