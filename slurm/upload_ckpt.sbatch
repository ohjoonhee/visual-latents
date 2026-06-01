#!/bin/bash
# Upload a training checkpoint to HuggingFace Hub via a CPU partition.
#
# WHY: HF uploads on a 7B ckpt are I/O + CPU intensive (hf_xet chunking,
# hashing, retries) — easily 1-2h. They MUST NOT run on the login node.
# This sbatch wraps scripts/cluster/upload_ckpt_to_hf.py on a cpu-short
# slot. Resources are minimal: 4 CPUs, 8GB RAM. The bottleneck is network.
#
# Submit (env-driven so we don't fork a sbatch per upload):
#   CKPT_DIR=/data/.../upstream_stage3_baseline/checkpoint-1500 \
#     STEP=1500 \
#     REPO=ohjoonhee/vlatents-qwen25vl7b-stage3-upstream-baseline-v1 \
#     NOTES="..." \
#     sb slurm/upload_ckpt.sbatch
#
# DO NOT agent-submit (project rule §3-A).
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
[ -f .env ] && { set -a; source .env; set +a; }

# Required env from submitter
: "${CKPT_DIR:?CKPT_DIR must be set}"
: "${STEP:?STEP must be set}"
: "${REPO:?REPO must be set}"
NOTES="${NOTES:-}"

slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}

LOGFILE="$SLURM_SUBMIT_DIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
JOB_T0=$(date +%s)
elapsed_h () { printf "%.1fh" "$(echo "scale=1; ($(date +%s) - ${JOB_T0}) / 3600.0" | bc)"; }

slack_notify ":outbox_tray: START hf-upload | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
repo: $REPO
ckpt: $CKPT_DIR (step=$STEP)
log:  \`${ABSLOG}\`"

if [ ! -d "$CKPT_DIR" ]; then
  slack_notify ":x: hf-upload: ckpt dir MISSING: $CKPT_DIR"; exit 11
fi
if [ -z "${HF_TOKEN:-}" ]; then
  slack_notify ":x: hf-upload: HF_TOKEN not in .env"; exit 12
fi

# Activate the upstream Monet uv venv (has huggingface_hub + hf_xet).
MONET=$HOME/Monet
if [ ! -d "$MONET/.venv" ]; then
  slack_notify ":x: hf-upload: $MONET/.venv missing"; exit 13
fi
source "$MONET/.venv/bin/activate"

cd "$SLURM_SUBMIT_DIR"
echo "[$(date)] cwd: $(pwd)"
echo "[$(date)] python: $(which python)"
echo "[$(date)] hf_hub: $(python -c 'import huggingface_hub; print(huggingface_hub.__version__)')"

NOTES_ARG=()
if [ -n "$NOTES" ]; then NOTES_ARG=(--notes "$NOTES"); fi

python scripts/cluster/upload_ckpt_to_hf.py \
  --ckpt-dir "$CKPT_DIR" \
  --step "$STEP" \
  --repo "$REPO" \
  "${NOTES_ARG[@]}"
RC=$?

TOTAL_H=$(elapsed_h)
if [ $RC -eq 0 ]; then
  slack_notify ":tada: hf-upload DONE in $TOTAL_H | https://huggingface.co/$REPO/tree/step-$STEP"
  exit 0
else
  slack_notify ":warning: hf-upload exit=$RC in $TOTAL_H | check log"
  exit $RC
fi
