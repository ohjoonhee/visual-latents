#!/bin/bash
# Round-3 single cell on bioai. Submit with:
#   sb slurm/round3_cell.sbatch configs/round3/C1_full.yaml
#
# NEVER agent-submitted. See ~/.claude/docs/bioai_cluster_spec.md MANDATORY policy.
# Compliant with bioai cluster spec §2 (admin rules) and §3-G (cache redirects).
#
#SBATCH -J vl-round3
#SBATCH -p gpu-4farm
#SBATCH --account=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64                # admin guideline 32-64 for 4 GPUs (§2-F)
#SBATCH --mem=512G
#SBATCH -t 23:30:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail

# --- env ---
set -a; source .env; set +a
export MACHINE=bioai

module purge                              # PDF examples always purge first
module load cuda/13.0                     # matches torch 2.11.0+cu130 wheel
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/13.0

# --- cache redirects (§3-G mandatory) ---
# /tmp on compute nodes is RAM-tmpfs and counts against --mem; everything large
# must redirect to /data. HF_HOME comes from ~/.bashrc on bioai.
mkdir -p /data/joonhee/vl/tmp \
         /data/joonhee/vl/cache/{triton,torch,torch-extensions} \
         /data/joonhee/vl/wandb
export TMPDIR=/data/joonhee/vl/tmp
export TRITON_CACHE_DIR=/data/joonhee/vl/cache/triton
export TORCH_HOME=/data/joonhee/vl/cache/torch
export TORCH_EXTENSIONS_DIR=/data/joonhee/vl/cache/torch-extensions
export WANDB_DIR=/data/joonhee/vl/wandb

CONFIG="${1:?usage: sb slurm/round3_cell.sbatch configs/round3/CX.yaml}"

# --- slack notify (start) ---
LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(realpath "$LOGFILE")"
SCRIPT_PATH="${ORIG_SBATCH_SCRIPT:-${SLURM_SUBMIT_SCRIPT:-<unknown>}}"

slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}

slack_notify ":rocket: START vl-round3 (${CONFIG}) | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
script: \`${SCRIPT_PATH}\`
log: \`${ABSLOG}\`
follow: \`tail -n 200 -F ${ABSLOG}\`"

# --- run ---
uv run accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    -m vl.train --config "${CONFIG}" --resume --max-time 23h
rc=$?

# --- slack notify (end) ---
# NO auto-resubmit (project-owner rule: every submit requires explicit approval).
# If --max-time was hit (rc=124) and a chain link is desired, surface it via Slack
# but do NOT submit anything.
if [ $rc -eq 124 ]; then
  slack_notify ":hourglass: HIT TIME-LIMIT vl-round3 (${CONFIG}) jobid=${SLURM_JOB_ID}
chain link needed — submit manually after review:
  sb slurm/round3_cell.sbatch ${CONFIG}"
elif [ $rc -eq 0 ]; then
  slack_notify ":white_check_mark: DONE vl-round3 (${CONFIG}) jobid=${SLURM_JOB_ID}"
else
  slack_notify ":x: FAIL vl-round3 (${CONFIG}) jobid=${SLURM_JOB_ID} exit=${rc}"
fi

exit $rc
