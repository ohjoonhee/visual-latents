#!/bin/bash
# LVR cluster TRAINING env build (cpu-short). Builds the vendored LVR .venv (run from upstreams/lvr) from
# the vendored requirements.txt + flash-attn (from source, needs CUDA_HOME) + boto3
# (so the upstream OCI import resolves with --online_checkpoint False).
# Build off the login node per §3-D. Submit:  sb slurm/lvr_env_build.sbatch
#SBATCH -J launch
#SBATCH -p cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH -t 0-04:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log
set -uo pipefail
[ -f .env ] && { set -a; source .env; set +a; }
slack_notify(){ [ -z "${SLACK_WEBHOOK_URL:-}" ]&&return 0; curl -s -X POST -H 'Content-type: application/json' --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL">/dev/null; }

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO" || { slack_notify ":x: LVR vendor dir not found — git-pull visual-latents & submit from upstreams/lvr"; exit 11; }
module purge; module load cuda/12.6
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.6   # flash-attn from-source build needs this
export TMPDIR=/data/joonhee/lvr/.cache/tmp; mkdir -p "$TMPDIR"
export TORCH_EXTENSIONS_DIR=/data/joonhee/lvr/.cache/torch-extensions; mkdir -p "$TORCH_EXTENSIONS_DIR"
slack_notify ":rocket: START lvr-env-build | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}"

UV=$HOME/.local/bin/uv
$UV venv --python 3.11 .venv 2>&1 || { slack_notify ":x: uv venv failed"; exit 12; }
# Core deps (torch 2.6.0 PyPI linux wheel = CUDA 12.4 build) + boto3 for the OCI
# import. --override slurm/uv_overrides.txt fixes 2 broken pins in the upstream
# freeze (av yanked, hub stale vs transformers) without editing requirements.txt.
$UV pip install --python .venv/bin/python -r requirements.txt --override slurm/uv_overrides.txt boto3 2>&1 | tail -15
rc=${PIPESTATUS[0]}; [ $rc -ne 0 ] && { slack_notify ":x: pip install -r requirements failed rc=$rc"; exit 13; }
# flash-attn from source (needs CUDA_HOME). Slow (single .so compile); cached in TORCH_EXTENSIONS_DIR.
$UV pip install --python .venv/bin/python flash-attn --no-build-isolation 2>&1 | tail -8
rc=${PIPESTATUS[0]}; [ $rc -ne 0 ] && { slack_notify ":warning: flash-attn build failed rc=$rc — train with --disable_flash_attn2 True (sdpa) fallback"; }

.venv/bin/python - <<'PY' 2>&1 | tail -8
import torch, transformers, deepspeed, trl, datasets
print("torch", torch.__version__, "| tf", transformers.__version__, "| ds", deepspeed.__version__, "| trl", trl.__version__)
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except Exception as e: print("flash_attn NOT available:", e)
import boto3; print("boto3 OK")
PY
slack_notify ":white_check_mark: DONE lvr-env-build (jobid=${SLURM_JOB_ID})"
