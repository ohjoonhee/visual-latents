#!/bin/bash
# Cluster Phase 3 — JOB C: Monet Stage 2, deviation #1 FIXED (latent-only backprop).
#
# WHY THIS JOB: Job B (faithful-ish Monet Stage 2) collapsed — internal probe
# (cluster/eval.py, K=8, n=200) gave the reproduced step1500 ckpt mean off-diag
# cos = 0.840 / qwen_base util = -5.35 (harmful latents), vs the released
# Monet stage2 measured IDENTICALLY at cos = 0.377 / util = +2.05 (healthy).
# That collapse is a trustworthy RELATIVE signal (same metric, same probe),
# independent of the unresolved absolute-V* question.
#
# Job B carried ONE named deviation from the paper recipe: deviation #1 —
# emphasize_latent_weight was a plain scalar add (total = ce + (aw+elw)*align),
# backpropping the alignment term through the whole LLM trunk. The paper
# (Monet src/trainer.py:152-224) instead backprops the alignment term ONLY
# through the latent embeddings (compute_latents_only_loss), and computes the
# alignment loss in the CE forward where ce_patch_vec is spliced into
# inputs_embeds (so it is graph-connected to the latents).
#
# Job C fixes exactly that, verbatim to upstream, changing NO hyperparameter
# (stage2_monet_jobC.yaml == Job B recipe). It is a clean A/B on the single
# deviation. Expected outcome if deviation #1 was the collapse cause: the
# reproduced ckpt's cos moves back toward ~0.377 with positive utility. If it
# does NOT, deviation #1 is exonerated and the archaeology (Session A's local
# paper-repro track) gets a sharper question.
#
# Init from our Stage 1 SFT ckpt (cluster_phase3/stage1_sft/checkpoint). Stage
# 1 is 1-epoch SFT and step-1100 == its budgeted max_steps (epoch completed,
# not truncated); extending Stage 1 would itself deviate from the paper recipe,
# so Job C is Stage-2-only. Released stage2 is eval-time reference only.
#
# Deviations from upstream (see trainer_monet_stage2.py header):
#   1. FIXED here — latent-only backprop is now verbatim upstream, with the
#      upstream forward structure (alignment in the CE forward). A smoke gate
#      asserts the alignment grad actually reaches ce_patch_vec.
#   2. attention_mask_4d via mask_utils.build_monet_4d_attn (hand-roll).
#   3. Inline teacher forward (no offline precompute) — +14 GB/rank.
#
# Validation gates (smoke = 5 trainer steps, MUST pass both):
#   - [mem-after-prepare] dist_type=DEEPSPEED  (else ZeRO->DDP fallback)
#   - [devfix-check] latent_only_connected=True  (else the latent-only
#     backprop is a silent no-op — all grads None->zeros — and the 20h run
#     would be pointless. ABORT.)
#   - Mid-training save at step 500/1000/1500 → partial eval always possible.
#
# Wall-budget breakdown (24h cap): venv <1m; smoke 3-5m; train 2000 steps
# ~17-20h (two-pass + inline teacher); final save 2-3m; eval (repro +
# released stage2, 200ex K=8) 60-90m; slack 1-3h.
#
# Failure handling: venv->exit 11; stage1 missing->exit 12; smoke rc->exit 13;
# dist_type!=DEEPSPEED->exit 14; latent_only NOT connected->exit 15;
# training fail->still eval latest checkpoint_stepN then exit train rc;
# released-stage2 arm is optional (warn, do not fail).
#
# Post-run (manual, user-approved): upload best ckpt with
#   scripts/cluster/upload_ckpt_to_hf.py  →  HF repo
#   ohjoonhee/vlatents-qwen25vl7b-stage2-repro-v2  (v2 = deviation-#1-fixed
#   rerun; steps as git tags; do NOT reuse the v1 repo).
#
# Submit with: sb slurm/cluster_jobC_monet_stage2_devfix.sbatch
# DO NOT agent-submit (project-owner rule §3-A).
#
#SBATCH -J vl-jobC-monet-stage2-devfix
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
slack_notify ":rocket: START vl-jobC-monet-stage2-devfix | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
Monet Stage 2 — deviation #1 FIXED (latent-only backprop, verbatim upstream)
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
  slack_notify ":x: vl-jobC stage0/uv-sync FAILED (jobid=${SLURM_JOB_ID})"
  exit 11
fi
echo "[$(date)] [stage0] venv sync OK ($(elapsed_h))"

# ----- Stage 1: precondition checks -----
STAGE1_CKPT=/data/joonhee/visual-latents/cluster_phase3/stage1_sft/checkpoint
if [ ! -d "$STAGE1_CKPT" ]; then
  slack_notify ":x: vl-jobC Stage 1 ckpt MISSING at ${STAGE1_CKPT} — cannot start Stage 2"
  exit 12
fi
echo "[$(date)] Stage 1 ckpt OK at ${STAGE1_CKPT} ($(du -sh $STAGE1_CKPT | cut -f1))"

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
  slack_notify ":bookmark: vl-jobC found released Monet stage2 at \`${RELEASED_S2_CKPT}\` — will eval side-by-side"
else
  echo "[$(date)] WARNING: released Monet stage2 ckpt NOT FOUND on cluster — eval will be reproduction-only"
  slack_notify ":warning: vl-jobC released Monet stage2 ckpt NOT FOUND — eval reproduction-only"
fi

S2_OUT=/data/joonhee/visual-latents/cluster_phase3/stage2_jobC

# ----- Stage 2: SMOKE (5 trainer steps; verifies mem + dist_type + devfix) -----
slack_notify ":wrench: vl-jobC smoke (5 steps) START at $(elapsed_h)"
echo "[$(date)] [smoke] trainer_monet_stage2.py --smoke (5 steps)"

uv --project cluster run --no-sync \
  accelerate launch \
    --config_file cluster/accelerate_zero2.yaml \
    --num_machines=1 --num_processes=4 --mixed_precision=bf16 \
    cluster/trainer_monet_stage2.py \
      --config cluster/configs/stage2_monet_jobC.yaml \
      --out_dir ${S2_OUT}/_smoke \
      --smoke
SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then
  slack_notify ":x: vl-jobC smoke FAILED rc=${SMOKE_RC} at $(elapsed_h) — NOT proceeding. Check log for [mem-after-prepare] / [devfix-check]."
  exit 13
fi
# Gate 1: ZeRO-2 actually engaged (not silent DDP fallback).
if ! grep -q "\[mem-after-prepare\] dist_type=DEEPSPEED" "$LOGFILE"; then
  slack_notify ":x: vl-jobC smoke OK by rc BUT dist_type!=DEEPSPEED — DDP fallback suspected, NOT proceeding"
  exit 14
fi
# Gate 2: the deviation-#1 fix is actually live — the alignment gradient
# reaches ce_patch_vec (latent-only backprop is NOT a silent zero). Without
# this, Job C would be indistinguishable from a no-emphasize run for 20h.
if ! grep -q "\[devfix-check\] latent_only_connected=True" "$LOGFILE"; then
  slack_notify ":x: vl-jobC smoke OK BUT [devfix-check] latent_only_connected!=True — latent-only backprop is a no-op (grads None->zeros). ABORTING before the full run. Inspect the alignment/ce_patch_vec graph."
  exit 15
fi
echo "[$(date)] [smoke] OK ($(elapsed_h)); DEEPSPEED + latent_only_connected=True confirmed"
slack_notify ":white_check_mark: vl-jobC smoke OK at $(elapsed_h) — ZeRO-2 + latent-only backprop both confirmed, proceeding to full training"

# ----- Stage 3: full Monet Stage 2 training (deviation #1 fixed) -----
slack_notify ":dart: vl-jobC Stage 2 training (2000 steps) START at $(elapsed_h)"
echo "[$(date)] [train] trainer_monet_stage2.py full run (jobC devfix)"

uv --project cluster run --no-sync \
  accelerate launch \
    --config_file cluster/accelerate_zero2.yaml \
    --num_machines=1 --num_processes=4 --mixed_precision=bf16 \
    cluster/trainer_monet_stage2.py \
      --config cluster/configs/stage2_monet_jobC.yaml
TRAIN_RC=$?

S2_CKPT=${S2_OUT}/checkpoint
if [ $TRAIN_RC -ne 0 ]; then
  slack_notify ":warning: vl-jobC Stage 2 training FAILED rc=${TRAIN_RC} at $(elapsed_h) — will still attempt eval on latest intermediate ckpt"
fi
echo "[$(date)] [train] exit=${TRAIN_RC} ($(elapsed_h))"

# ----- Stage 4: eval (reproduction + released-stage2 reference) -----
EVAL_OUT=${S2_OUT}/eval

if [ -d "$S2_CKPT" ]; then
  REPRO_TARGET=$S2_CKPT
  REPRO_LABEL="stage2_jobC_repro"
elif [ -d "${S2_OUT}/checkpoint_step1500" ]; then
  REPRO_TARGET=${S2_OUT}/checkpoint_step1500
  REPRO_LABEL="stage2_jobC_repro_step1500"
elif [ -d "${S2_OUT}/checkpoint_step1000" ]; then
  REPRO_TARGET=${S2_OUT}/checkpoint_step1000
  REPRO_LABEL="stage2_jobC_repro_step1000"
elif [ -d "${S2_OUT}/checkpoint_step500" ]; then
  REPRO_TARGET=${S2_OUT}/checkpoint_step500
  REPRO_LABEL="stage2_jobC_repro_step500"
else
  REPRO_TARGET=""
  REPRO_LABEL=""
fi

CKPT_ARGS=()
if [ -n "$REPRO_TARGET" ]; then
  CKPT_ARGS+=("${REPRO_LABEL}:${REPRO_TARGET}")
fi
if [ -n "$RELEASED_S2_CKPT" ]; then
  CKPT_ARGS+=("monet_paper_stage2:${RELEASED_S2_CKPT}")
fi

if [ ${#CKPT_ARGS[@]} -eq 0 ]; then
  slack_notify ":x: vl-jobC no ckpts to eval (reproduction and released both missing) — exiting"
  EVAL_RC=21
else
  slack_notify ":mag: vl-jobC eval START at $(elapsed_h) — targets: ${CKPT_ARGS[*]}"
  echo "[$(date)] [eval] cluster/eval.py --ckpts ${CKPT_ARGS[*]} (K=8, n=200)"

  uv --project cluster run --no-sync \
    python cluster/eval.py \
      --ckpts "${CKPT_ARGS[@]}" \
      --K 8 \
      --n_eval 200 \
      --out_root ${EVAL_OUT}
  EVAL_RC=$?

  if [ $EVAL_RC -eq 0 ]; then
    slack_notify ":white_check_mark: vl-jobC eval DONE at $(elapsed_h) — out at ${EVAL_OUT}, see CROSS_REPORT.md. Compare jobC repro cos/util vs released 0.377/+2.05 and Job B 0.840/-5.35."
  else
    slack_notify ":x: vl-jobC eval FAILED rc=${EVAL_RC} at $(elapsed_h)"
  fi
fi

# ----- Wrap up -----
TOTAL_H=$(elapsed_h)
if [ $TRAIN_RC -eq 0 ] && [ $EVAL_RC -eq 0 ]; then
  slack_notify ":tada: vl-jobC-monet-stage2-devfix DONE in ${TOTAL_H} | train ✅ eval ✅"
  exit 0
elif [ $EVAL_RC -eq 0 ]; then
  slack_notify ":warning: vl-jobC PARTIAL in ${TOTAL_H} | train ❌ (rc=${TRAIN_RC}) eval ✅ (on ${REPRO_LABEL:-released-only})"
  exit $TRAIN_RC
else
  slack_notify ":warning: vl-jobC PARTIAL in ${TOTAL_H} | train rc=${TRAIN_RC} eval ❌ (rc=${EVAL_RC})"
  exit $EVAL_RC
fi
