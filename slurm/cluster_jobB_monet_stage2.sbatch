#!/bin/bash
# Cluster Phase 3 — JOB B: faithful Monet Stage 2 reproduction.
#
# WHY THIS JOB: V1 G@7B (Pivot A on Monet-SFT-125K, Job A2/214637) collapsed
# (mean off-diag cos = 1.000, util ≈ 0). Audit identified the smoking gun:
# Pivot A's per-example bbox supervision degenerates to a constant
# center-crop on Monet-SFT-125K (no per-example bbox in the dataset), so
# the K=4 latents have no diversity signal. The Pivot A recipe is not a
# valid LVR instantiation on this data.
#
# Strategic pivot per the user's 2026-05-13 message: reproduce Monet
# Stage 2 (the canonical baseline our Pivot A variants were designed
# AROUND) before adding variations. Two reproduction reference points:
#   - released NOVAglow646/Monet-SFT-7B `stage2/*` (Monet authors' own
#     Stage 2 ckpt — paper-authority numbers via cluster/eval.py)
#   - local 3B Monet Stage 2 (phase2_monet_stage2/REPORT.md) mean off-diag
#     cos ≈ 0.375 at 5K examples — orders-of-magnitude check at scale-down
#
# Init from our Stage 1 SFT ckpt (cluster_phase3/stage1_sft/checkpoint).
# Rationale: released Monet-SFT-7B repo on HF ships only stage2/ and
# stage3/ (per phase0_monet_probe/download.py:8-21 — no stage1 listed),
# so our Stage 1 ckpt is the only Stage-1-ready option on disk. The
# released stage2 is used PURELY as an eval-time reference, not as init.
#
# Known deviations from upstream (see trainer_monet_stage2.py header):
#   1. emphasize_latent_weight applied as a plain scalar add (paper uses
#      latent-only backprop locality). Effective: total = ce + 4*align.
#   2. attention_mask_4d via mask_utils.build_monet_4d_attn (hand-roll).
#   3. Inline teacher forward (no offline precompute) — +14 GB/rank.
#
# Validation gates:
#   - Stage 2-smoke (5 trainer steps): MUST log
#       [mem-after-prepare] dist_type=DEEPSPEED alloc=~30GB
#     If dist_type=MULTI_GPU → ZeRO fell back to DDP (job 214425 bug).
#     If alloc<20GB on first call → teacher didn't move to device.
#     If alloc>55GB → activations + student + teacher too big at this
#     max_pixels; lower max_pixels before resubmitting.
#   - Mid-training save at step 500/1000/1500 means partial eval is
#     always possible even if walltime hits.
#
# Wall-budget breakdown (24h cap):
#   - venv sync (likely no-op): <1 min
#   - smoke (5 steps): 3-5 min (validates mem + dist_type)
#   - Stage 2 training (2000 steps × ~30s/step × 4 GPU, student+teacher):
#     ~17-20h  ← largest unknown; per-step is heavier than V1 (two-pass +
#                inline teacher), measure on smoke before committing
#   - Final ckpt save (7B sharded): 2-3 min
#   - Eval reproduced ckpt + released stage2 (200 ex, K=8): 60-90 min
#   - Slack: 1-3 h
#
# Failure handling:
#   - venv sync fails → exit 11
#   - smoke fails → exit 13 (do NOT proceed to full training)
#   - training fails → STILL try eval on latest checkpoint_stepN, then
#     exit with training's code
#   - eval failure → exit with eval's code; ckpts on disk are re-evaluable
#   - released-stage2 eval is INDEPENDENT of reproduction; if released-
#     stage2 path missing, sbatch skips that arm and warns (does NOT fail
#     the job)
#
# Submit with: sb slurm/cluster_jobB_monet_stage2.sbatch
# DO NOT agent-submit (project-owner rule §3-A).
#
#SBATCH -J vl-jobB-monet-stage2
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
slack_notify ":rocket: START vl-jobB-monet-stage2 | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
Monet Stage 2 faithful reproduction
init: cluster_phase3/stage1_sft/checkpoint
eval-reference: NOVAglow646/Monet-SFT-7B/stage2 (if present)
script: \`${SCRIPT_PATH}\`
log:    \`${ABSLOG}\`
follow: \`tail -n 200 -F ${ABSLOG}\`"

JOB_T0=$(date +%s)
elapsed_h () { printf "%.1fh" "$(echo "scale=1; ($(date +%s) - ${JOB_T0}) / 3600.0" | bc)"; }

# ----- Stage 0: venv sync -----
echo "[$(date)] [stage0] uv --project cluster sync"
uv --project cluster sync 2>&1 | tail -10
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  slack_notify ":x: vl-jobB stage0/uv-sync FAILED (jobid=${SLURM_JOB_ID})"
  exit 11
fi
echo "[$(date)] [stage0] venv sync OK ($(elapsed_h))"

# ----- Stage 1: precondition checks -----
STAGE1_CKPT=/data/joonhee/visual-latents/cluster_phase3/stage1_sft/checkpoint
if [ ! -d "$STAGE1_CKPT" ]; then
  slack_notify ":x: vl-jobB Stage 1 ckpt MISSING at ${STAGE1_CKPT} — cannot start Stage 2"
  exit 12
fi
echo "[$(date)] Stage 1 ckpt OK at ${STAGE1_CKPT} ($(du -sh $STAGE1_CKPT | cut -f1))"

# Released Monet Stage 2 ckpt path (eval-only reference). Two plausible
# locations: (a) the local_dir from phase0_monet_probe/download.py at
# the cluster repo path; (b) HF Hub cache (HF_HOME=/data/joonhee/.cache).
# Probe both; if neither exists, eval continues without the released-
# stage2 arm (warning only).
RELEASED_S2_A=$HOME/projects/visual-latents/phase0_monet_probe/checkpoints/Monet-SFT-7B/stage2
RELEASED_S2_B=$(ls -d /data/joonhee/.cache/huggingface/hub/models--NOVAglow646--Monet-SFT-7B/snapshots/*/stage2 2>/dev/null | head -1)
if [ -d "$RELEASED_S2_A" ]; then
  RELEASED_S2_CKPT=$RELEASED_S2_A
elif [ -n "$RELEASED_S2_B" ] && [ -d "$RELEASED_S2_B" ]; then
  RELEASED_S2_CKPT=$RELEASED_S2_B
else
  RELEASED_S2_CKPT=""
fi
if [ -n "$RELEASED_S2_CKPT" ]; then
  echo "[$(date)] Released Monet stage2 ckpt: ${RELEASED_S2_CKPT}"
  slack_notify ":bookmark: vl-jobB found released Monet stage2 at \`${RELEASED_S2_CKPT}\` — will eval side-by-side"
else
  echo "[$(date)] WARNING: released Monet stage2 ckpt NOT FOUND on cluster — eval will be reproduction-only"
  slack_notify ":warning: vl-jobB released Monet stage2 ckpt NOT FOUND — eval reproduction-only (no paper-authority reference)"
fi

# ----- Stage 2: SMOKE (5 trainer steps, verifies mem + dist_type) -----
slack_notify ":wrench: vl-jobB smoke (5 steps) START at $(elapsed_h)"
echo "[$(date)] [smoke] trainer_monet_stage2.py --smoke (5 steps)"

uv --project cluster run --no-sync \
  accelerate launch \
    --config_file cluster/accelerate_zero2.yaml \
    --num_machines=1 --num_processes=4 --mixed_precision=bf16 \
    cluster/trainer_monet_stage2.py \
      --config cluster/configs/stage2_monet.yaml \
      --out_dir /data/joonhee/visual-latents/cluster_phase3/stage2_baseline/_smoke \
      --smoke
SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then
  slack_notify ":x: vl-jobB smoke FAILED rc=${SMOKE_RC} at $(elapsed_h) — NOT proceeding to full training. Check log for [mem-after-prepare] line."
  exit 13
fi
# Verify the diagnostic actually fired (catches silent dist_type=MULTI_GPU
# fallback that would otherwise pass smoke but OOM the full run).
if ! grep -q "\[mem-after-prepare\] dist_type=DEEPSPEED" "$LOGFILE"; then
  slack_notify ":x: vl-jobB smoke OK by exit code BUT [mem-after-prepare] dist_type!=DEEPSPEED — DDP fallback suspected, NOT proceeding"
  exit 14
fi
echo "[$(date)] [smoke] OK ($(elapsed_h)); mem-after-prepare line confirmed DEEPSPEED"
slack_notify ":white_check_mark: vl-jobB smoke OK at $(elapsed_h) — proceeding to full training"

# ----- Stage 3: full Monet Stage 2 training -----
slack_notify ":dart: vl-jobB Stage 2 training (2000 steps) START at $(elapsed_h)"
echo "[$(date)] [train] trainer_monet_stage2.py full run"

uv --project cluster run --no-sync \
  accelerate launch \
    --config_file cluster/accelerate_zero2.yaml \
    --num_machines=1 --num_processes=4 --mixed_precision=bf16 \
    cluster/trainer_monet_stage2.py \
      --config cluster/configs/stage2_monet.yaml
TRAIN_RC=$?

S2_OUT=/data/joonhee/visual-latents/cluster_phase3/stage2_baseline
S2_CKPT=${S2_OUT}/checkpoint
if [ $TRAIN_RC -ne 0 ]; then
  slack_notify ":warning: vl-jobB Stage 2 training FAILED rc=${TRAIN_RC} at $(elapsed_h) — will still attempt eval on latest intermediate ckpt"
fi
echo "[$(date)] [train] exit=${TRAIN_RC} ($(elapsed_h))"

# ----- Stage 4: eval (reproduction + released-stage2 reference) -----
EVAL_OUT=${S2_OUT}/eval

# Pick the best available reproduced ckpt.
if [ -d "$S2_CKPT" ]; then
  REPRO_TARGET=$S2_CKPT
  REPRO_LABEL="stage2_repro"
elif [ -d "${S2_OUT}/checkpoint_step1500" ]; then
  REPRO_TARGET=${S2_OUT}/checkpoint_step1500
  REPRO_LABEL="stage2_repro_step1500"
elif [ -d "${S2_OUT}/checkpoint_step1000" ]; then
  REPRO_TARGET=${S2_OUT}/checkpoint_step1000
  REPRO_LABEL="stage2_repro_step1000"
elif [ -d "${S2_OUT}/checkpoint_step500" ]; then
  REPRO_TARGET=${S2_OUT}/checkpoint_step500
  REPRO_LABEL="stage2_repro_step500"
else
  REPRO_TARGET=""
  REPRO_LABEL=""
fi

# Build --ckpts list: reproduction (if any) + released stage2 (if any).
CKPT_ARGS=()
if [ -n "$REPRO_TARGET" ]; then
  CKPT_ARGS+=("${REPRO_LABEL}:${REPRO_TARGET}")
fi
if [ -n "$RELEASED_S2_CKPT" ]; then
  CKPT_ARGS+=("monet_paper_stage2:${RELEASED_S2_CKPT}")
fi

if [ ${#CKPT_ARGS[@]} -eq 0 ]; then
  slack_notify ":x: vl-jobB no ckpts to eval (reproduction and released both missing) — exiting"
  EVAL_RC=21
else
  slack_notify ":mag: vl-jobB eval START at $(elapsed_h) — targets: ${CKPT_ARGS[*]}"
  echo "[$(date)] [eval] cluster/eval.py --ckpts ${CKPT_ARGS[*]} (K=8, n=200)"

  uv --project cluster run --no-sync \
    python cluster/eval.py \
      --ckpts "${CKPT_ARGS[@]}" \
      --K 8 \
      --n_eval 200 \
      --out_root ${EVAL_OUT}
  EVAL_RC=$?

  if [ $EVAL_RC -eq 0 ]; then
    slack_notify ":white_check_mark: vl-jobB eval DONE at $(elapsed_h) — out at ${EVAL_OUT}, see CROSS_REPORT.md"
  else
    slack_notify ":x: vl-jobB eval FAILED rc=${EVAL_RC} at $(elapsed_h)"
  fi
fi

# ----- Wrap up -----
TOTAL_H=$(elapsed_h)
if [ $TRAIN_RC -eq 0 ] && [ $EVAL_RC -eq 0 ]; then
  slack_notify ":tada: vl-jobB-monet-stage2 DONE in ${TOTAL_H} | train ✅ eval ✅"
  exit 0
elif [ $EVAL_RC -eq 0 ]; then
  slack_notify ":warning: vl-jobB PARTIAL in ${TOTAL_H} | train ❌ (rc=${TRAIN_RC}) eval ✅ (on ${REPRO_LABEL:-released-only})"
  exit $TRAIN_RC
else
  slack_notify ":warning: vl-jobB PARTIAL in ${TOTAL_H} | train rc=${TRAIN_RC} eval ❌ (rc=${EVAL_RC})"
  exit $EVAL_RC
fi
