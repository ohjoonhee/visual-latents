#!/bin/bash
# Cluster Phase 3 — JOB D PRECOMPUTE: cache Stage-2 teacher latents for Stage 3.
#
# WHY THIS JOB: Stage 3 training (Job D baseline + Job D + VICReg) needs the
# released Monet-SFT-7B Stage-2 latents pre-extracted to disk, one .pt per
# Visual_CoT sample. ~118K samples, all_layers caching at K=8 → ~190 GB.
# Walltime ~6h at single-rank inference-mode.
#
# PRECONDITIONS (pre-flight CHECK at start of script; fail-fast on miss):
#   1. Released Monet-SFT-7B downloaded to one of:
#      - /data/joonhee/visual-latents/cluster_phase3/monet_sft_7b/stage2  (preferred)
#      - HF cache snapshot under ~/.cache/huggingface/hub/models--NOVAglow646--Monet-SFT-7B/.../stage2
#   2. /data has >=200 GB free
#   3. Monet-SFT-125K dataset present at the standard cluster path
#
# Output: /data/joonhee/visual-latents/cluster_phase3/teacher_latents/latent_all_layers_*.pt
#
# Submit with: sb slurm/cluster_jobD_precompute_teacher_latents.sbatch
# DO NOT agent-submit (project-owner rule §3-A).
#
#SBATCH -J vl-jobD-precompute-teacher-latents
#SBATCH -p gpu-4farm
#SBATCH --account=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# Walltime: 1-day max (gpu-4farm partition cap). Expected actual ~6h; max
# budget gives 18h slack — robustness over backfill priority (per user
# request after Job C was walltime-killed at step 1690/2000).
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -uo pipefail
[ -f .env ] && { set -a; source .env; set +a; }

module purge
module load cuda/12.6
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.6

mkdir -p /data/joonhee/visual-latents/.cache/{tmp,triton,torch,torch-extensions} \
         /data/joonhee/visual-latents/cluster_phase3/teacher_latents
export TMPDIR=/data/joonhee/visual-latents/.cache/tmp
export TRITON_CACHE_DIR=/data/joonhee/visual-latents/.cache/triton
export TORCH_HOME=/data/joonhee/visual-latents/.cache/torch
export TORCH_EXTENSIONS_DIR=/data/joonhee/visual-latents/.cache/torch-extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"

slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}

LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
JOB_T0=$(date +%s)
elapsed_h () { printf "%.1fh" "$(echo "scale=1; ($(date +%s) - ${JOB_T0}) / 3600.0" | bc)"; }

slack_notify ":rocket: START vl-jobD-precompute | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
Precomputing Stage-2 teacher latents (all_layers, K=8, ~118K Visual_CoT samples, ~6h)
output: /data/joonhee/visual-latents/cluster_phase3/teacher_latents/
log:    \`${ABSLOG}\`"

# ----- Stage 0: venv -----
echo "[$(date)] [stage0] uv --project cluster sync"
uv --project cluster sync 2>&1 | tail -10
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  slack_notify ":x: vl-jobD-precompute stage0/uv-sync FAILED"; exit 11
fi

# ----- Stage 1: locate Monet-SFT-7B Stage 2 ckpt -----
TEACHER_A=/data/joonhee/visual-latents/cluster_phase3/monet_sft_7b/stage2
TEACHER_B=$(ls -d /data/joonhee/.cache/huggingface/hub/models--NOVAglow646--Monet-SFT-7B/snapshots/*/stage2 2>/dev/null | head -1)
TEACHER_C=$HOME/projects/visual-latents/phase0_monet_probe/checkpoints/Monet-SFT-7B/stage2

if [ -d "$TEACHER_A" ]; then
  TEACHER_CKPT=$TEACHER_A
elif [ -n "$TEACHER_B" ] && [ -d "$TEACHER_B" ]; then
  TEACHER_CKPT=$TEACHER_B
elif [ -d "$TEACHER_C" ]; then
  TEACHER_CKPT=$TEACHER_C
else
  echo "[$(date)] Monet-SFT-7B not found on cluster — auto-downloading..."
  slack_notify ":arrow_down: vl-jobD-precompute downloading Monet-SFT-7B (~16GB) to /data/joonhee/.../monet_sft_7b/"
  uv --project cluster run --no-sync \
    python scripts/cluster/download_monet_sft_7b.py \
      --target /data/joonhee/visual-latents/cluster_phase3/monet_sft_7b
  if [ $? -ne 0 ] || [ ! -d "$TEACHER_A" ]; then
    slack_notify ":x: vl-jobD-precompute Monet-SFT-7B download FAILED"; exit 12
  fi
  TEACHER_CKPT=$TEACHER_A
fi
echo "[$(date)] teacher ckpt: ${TEACHER_CKPT} ($(du -sh ${TEACHER_CKPT} | cut -f1))"

# ----- Stage 2: disk space check -----
AVAIL_GB=$(df -BG /data/joonhee/visual-latents/cluster_phase3 | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAIL_GB" -lt 200 ]; then
  slack_notify ":x: vl-jobD-precompute: /data has only ${AVAIL_GB}G free — need >=200G for precompute"
  exit 13
fi
echo "[$(date)] /data free: ${AVAIL_GB}G OK"

# ----- Stage 3: smoke (8 samples, ~2 min) -----
slack_notify ":wrench: vl-jobD-precompute smoke (8 samples)"
echo "[$(date)] [smoke] precompute_teacher_latents.py --n_max 8"

uv --project cluster run --no-sync \
  python scripts/cluster/precompute_teacher_latents.py \
    --teacher_ckpt "${TEACHER_CKPT}" \
    --out_dir /data/joonhee/visual-latents/cluster_phase3/teacher_latents/_smoke \
    --alignment_layer all_layers \
    --K 8 \
    --n_max 8
SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then
  slack_notify ":x: vl-jobD-precompute SMOKE FAILED rc=${SMOKE_RC}"
  exit 14
fi
N_SMOKE=$(ls /data/joonhee/visual-latents/cluster_phase3/teacher_latents/_smoke/latent_all_layers_*.pt 2>/dev/null | wc -l)
if [ "$N_SMOKE" -lt 4 ]; then
  slack_notify ":x: vl-jobD-precompute smoke ran but only ${N_SMOKE} .pt files written (expected ~8). Aborting."
  exit 15
fi
echo "[$(date)] smoke OK: ${N_SMOKE} .pt files in _smoke/  ($(elapsed_h))"
slack_notify ":white_check_mark: vl-jobD-precompute smoke OK (${N_SMOKE} files) — full run START"

# ----- Stage 4: full precompute -----
echo "[$(date)] [full] precompute_teacher_latents.py full run"
uv --project cluster run --no-sync \
  python scripts/cluster/precompute_teacher_latents.py \
    --teacher_ckpt "${TEACHER_CKPT}" \
    --out_dir /data/joonhee/visual-latents/cluster_phase3/teacher_latents \
    --alignment_layer all_layers \
    --K 8 \
    --skip_existing
PRECOMP_RC=$?

N_FILES=$(ls /data/joonhee/visual-latents/cluster_phase3/teacher_latents/latent_all_layers_*.pt 2>/dev/null | wc -l)
TOTAL_H=$(elapsed_h)
if [ $PRECOMP_RC -eq 0 ]; then
  slack_notify ":tada: vl-jobD-precompute DONE in ${TOTAL_H} | n_files=${N_FILES}
out: /data/joonhee/visual-latents/cluster_phase3/teacher_latents/
next: stage3_jobD_baseline sbatch (needs explicit \"go\")"
  exit 0
else
  slack_notify ":warning: vl-jobD-precompute exit=${PRECOMP_RC} in ${TOTAL_H} | n_files=${N_FILES} (partial)"
  exit $PRECOMP_RC
fi
