#!/bin/bash
# LVR cluster TRAINING env build (cpu-short). Builds the vendored LVR .venv (run
# from upstreams/lvr) from requirements.txt (+ uv_overrides) + a prebuilt flash-attn
# wheel + boto3 (so the upstream OCI import resolves with --online_checkpoint False).
# Build off the login node per §3-D.
#   Submit (default, PyPI torch ABI-0; flash-attn won't load -> train sdpa):
#       sb slurm/lvr_env_build.sbatch
#   Submit (option B, pytorch.org torch ABI-1 so flash-attn LOADS; fresh rebuild):
#       TORCH_INDEX=pytorch sb slurm/lvr_env_build.sbatch
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
# TORCH_INDEX: where torch comes from. NB (verified job 219384): torch 2.6.0+cu124
# is cxx11_abi=False (ABI 0) from BOTH PyPI and download.pytorch.org/whl/cu124 — so
# this knob does NOT change the ABI. The flash-attn ABI is selected by FA_ABI below
# instead. Kept for completeness; default pypi is fine.
#   pypi    (default) — PyPI manylinux1 wheel.
#   pytorch — cu124 index wheel (fresh venv). Same ABI 0; use only if you want the
#             pytorch.org binaries specifically.
TORCH_INDEX="${TORCH_INDEX:-pypi}"
[ "$TORCH_INDEX" = "pytorch" ] && rm -rf .venv
# Idempotent (pypi mode): reuse an existing venv so a re-run only reconciles deps.
[ -x .venv/bin/python ] || $UV venv --python 3.11 .venv 2>&1 || { slack_notify ":x: uv venv failed"; exit 12; }
if [ "$TORCH_INDEX" = "pytorch" ]; then
  # ABI-1 torch FIRST so the requirements install below sees torch==2.6.0 satisfied.
  $UV pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 2>&1 | tail -6
  rc=${PIPESTATUS[0]}; [ $rc -ne 0 ] && { slack_notify ":x: pytorch.org torch install failed rc=$rc"; exit 18; }
fi
# Core deps (torch already satisfied in pytorch mode; PyPI ABI-0 wheel in pypi mode)
# + boto3 for the OCI import. --override slurm/uv_overrides.txt fixes 2 broken pins
# in the upstream freeze (av yanked, hub stale vs transformers) — requirements.txt untouched.
$UV pip install --python .venv/bin/python -r requirements.txt --override slurm/uv_overrides.txt boto3 2>&1 | tail -15
rc=${PIPESTATUS[0]}; [ $rc -ne 0 ] && { slack_notify ":x: pip install -r requirements failed rc=$rc"; exit 13; }
# flash-attn: install a Dao-AILab PREBUILT wheel (2.6 / cu12 / cp311). FA_ABI picks
# the variant. torch 2.6.0+cu124 is cxx11_abi=False (ABI-0; same from PyPI and the
# pytorch.org cu124 index) -> it exports the `Ss` (non-__cxx11) c10 symbols. Their
# "abiFALSE" wheel demands the __cxx11 (ABI-1) symbol -> mismatch; so the matching
# wheel is empirically FA_ABI=TRUE. Default TRUE; FA_ABI=FALSE for the other variant.
FA_ABI="${FA_ABI:-TRUE}"
FA_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abi${FA_ABI}-cp311-cp311-linux_x86_64.whl"
$UV pip install --python .venv/bin/python "$FA_WHL" 2>&1 | tail -6
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  slack_notify ":warning: flash-attn prebuilt wheel install failed rc=$rc — trying source build (may ABI-mismatch)"
  $UV pip install --python .venv/bin/python flash-attn==2.8.3 --no-build-isolation 2>&1 | tail -8
fi

.venv/bin/python - <<'PY' 2>&1 | tail -8
import torch, transformers, deepspeed, trl, datasets
print("torch", torch.__version__, "| tf", transformers.__version__, "| ds", deepspeed.__version__, "| trl", trl.__version__)
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except Exception as e: print("flash_attn NOT available:", e)
import boto3; print("boto3 OK")
PY
slack_notify ":white_check_mark: DONE lvr-env-build (jobid=${SLURM_JOB_ID})"
