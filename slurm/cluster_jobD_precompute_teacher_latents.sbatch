#!/bin/bash
# Cluster Phase 3 — JOB D PRECOMPUTE: cache Stage-2 teacher latents for Stage 3.
#
# WHY THIS JOB: Stage 3 training (Job D baseline + Job D + VICReg) needs the
# released Monet-SFT-7B Stage-2 latents pre-extracted to disk, one .pt per
# Visual_CoT sample. ~118K samples, all_layers caching at K=8 → ~190 GB.
#
# RESOURCE NOTE (gpu-4farm policy): the partition is whole-node-allocation;
# the QoS minimum GRES is 4 GPUs per job. Asking for 1 GPU → QOSMinGRES
# pending forever. We take 4 H100s and shard the precompute across 4 ranks
# (script's --rank/--world modulo-split). Each rank handles ~29.5K samples
# independently (no inter-rank comms), pinned to its GPU via
# CUDA_VISIBLE_DEVICES. Expected wall: ~1.5h with all 4 ranks vs ~6h
# single-rank.
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
#SBATCH -J launch
#SBATCH -p gpu-4farm
#SBATCH --account=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
# Walltime: 1-day max (gpu-4farm partition cap). Expected actual ~1.5h
# at 4-rank parallel; max budget eliminates partial-completion risk
# (per user request after Job C was walltime-killed at step 1690/2000).
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

# ----- Stage 3: smoke (8 samples on GPU 0 only, ~2 min) -----
slack_notify ":wrench: vl-jobD-precompute smoke (8 samples on GPU 0)"
echo "[$(date)] [smoke] precompute_teacher_latents.py --n_max 8 (single GPU)"

CUDA_VISIBLE_DEVICES=0 uv --project cluster run --no-sync \
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
slack_notify ":white_check_mark: vl-jobD-precompute smoke OK (${N_SMOKE} files) — 4-rank full run START"

# ----- Stage 4: full precompute, 4 ranks in parallel (one per GPU) -----
# The script's --rank/--world does modulo-split of the sample list across
# ranks. No inter-rank communication; each writes its own subset to the
# shared out_dir. --skip_existing makes the launch idempotent if any rank
# is restarted. Per-rank log: logs/precompute_rank<R>_<jobid>.log
echo "[$(date)] [full] launching 4 parallel ranks (one per GPU)"
mkdir -p logs
PIDS=()
for R in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$R uv --project cluster run --no-sync \
    python scripts/cluster/precompute_teacher_latents.py \
      --teacher_ckpt "${TEACHER_CKPT}" \
      --out_dir /data/joonhee/visual-latents/cluster_phase3/teacher_latents \
      --alignment_layer all_layers \
      --K 8 \
      --rank $R --world 4 \
      --skip_existing \
      > logs/precompute_rank${R}_${SLURM_JOB_ID}.log 2>&1 &
  PIDS+=($!)
  echo "[$(date)] launched rank=$R pid=${PIDS[-1]} on GPU $R"
done
echo "[$(date)] all 4 ranks running: ${PIDS[*]}"

PRECOMP_RC=0
for i in 0 1 2 3; do
  pid=${PIDS[$i]}
  wait $pid
  prc=$?
  echo "[$(date)] rank=$i pid=$pid rc=$prc"
  if [ $prc -ne 0 ]; then
    PRECOMP_RC=$prc
    slack_notify ":warning: vl-jobD-precompute rank=$i exit=$prc (other ranks continue)"
  fi
done

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
