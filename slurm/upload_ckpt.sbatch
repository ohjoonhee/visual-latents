#!/bin/bash
# Upload a training checkpoint to HuggingFace Hub via the cpu-short partition.
#
# WHY: HF uploads on a 7B ckpt are I/O + CPU intensive (hf_xet chunking,
# hashing, retries) — easily 1-2h. They MUST NOT run on the login node
# (admin rule §2-C — login slice has ~16% of one CPU core + ~16 GB mem).
# This sbatch wraps scripts/cluster/upload_ckpt_to_hf.py on cpu-short.
#
# Env-driven so one sbatch covers every upload:
#   CKPT_DIR=/data/.../checkpoint-1500 \
#     STEP=1500 \
#     REPO=ohjoonhee/<repo-name> \
#     NOTES="..." \
#     sb slurm/upload_ckpt.sbatch
#
# DO NOT agent-submit (project-owner rule §3-A).
#
#SBATCH -J launch
#SBATCH -p cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -uo pipefail

# Project-scope .env override of ~/.env (per §3-E). `sb` already sourced
# ~/.env at submission time so SLACK_WEBHOOK_URL / HF_TOKEN are inherited.
[ -f .env ] && { set -a; source .env; set +a; }

module purge

# Cache redirects (MANDATORY per §3-G). /tmp on compute nodes is RAM-tmpfs
# that counts against --mem= — anything large goes to /data.
mkdir -p /data/joonhee/tmp \
         /data/joonhee/.cache/{triton,torch,torch-extensions}
export TMPDIR=/data/joonhee/tmp
export TRITON_CACHE_DIR=/data/joonhee/.cache/triton
export TORCH_HOME=/data/joonhee/.cache/torch
export TORCH_EXTENSIONS_DIR=/data/joonhee/.cache/torch-extensions
# HF_HOME inherited from ~/.bashrc

# --- §3-H locked Slack-notify pattern (helper + 3 mandatory messages) ---
slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}

LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
SCRIPT_PATH="${ORIG_SBATCH_SCRIPT:-<unknown>}"

slack_notify ":rocket: JOB START | ${SLURM_JOB_NAME} | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
script: \`${SCRIPT_PATH}\`
log: \`${ABSLOG}\`
follow: \`tail -n 200 -F ${ABSLOG}\`"

# --- Required env from submitter ---
: "${CKPT_DIR:?CKPT_DIR must be set}"
: "${STEP:?STEP must be set}"
: "${REPO:?REPO must be set}"
NOTES="${NOTES:-}"

# --- Preflight ---
if [ ! -d "$CKPT_DIR" ]; then
  echo "ERR: ckpt dir MISSING: $CKPT_DIR" >&2
  slack_notify ":x: FAIL ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID}) exit=11 — ckpt dir MISSING: \`${CKPT_DIR}\`"
  exit 11
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERR: HF_TOKEN not set (expected via ~/.env sourced by sb)" >&2
  slack_notify ":x: FAIL ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID}) exit=12 — HF_TOKEN missing in ~/.env"
  exit 12
fi

# --- Activate uv venv (~/Monet has huggingface_hub + hf_xet). ---
MONET="$HOME/Monet"
if [ ! -d "$MONET/.venv" ]; then
  echo "ERR: $MONET/.venv missing" >&2
  slack_notify ":x: FAIL ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID}) exit=13 — \`${MONET}/.venv\` missing"
  exit 13
fi
source "$MONET/.venv/bin/activate"

echo "[$(date)] node=${SLURMD_NODENAME} cwd=$(pwd)"
echo "[$(date)] python=$(which python)"
echo "[$(date)] hf_hub=$(python -c 'import huggingface_hub; print(huggingface_hub.__version__)')"

# --- Upload ---
NOTES_ARG=()
[ -n "$NOTES" ] && NOTES_ARG=(--notes "$NOTES")

python scripts/cluster/upload_ckpt_to_hf.py \
  --ckpt-dir "$CKPT_DIR" \
  --step "$STEP" \
  --repo "$REPO" \
  "${NOTES_ARG[@]}"
rc=$?

# --- §3-H locked DONE / FAIL ---
[ $rc -eq 0 ] \
  && slack_notify ":white_check_mark: DONE ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID})" \
  || slack_notify ":x: FAIL ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID}) exit=${rc}"
exit $rc
