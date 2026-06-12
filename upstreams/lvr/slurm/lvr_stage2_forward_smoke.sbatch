#!/bin/bash
# LVR Stage-2 GRPO CPU FORWARD-SMOKE (cpu-short, NO GPU). Heavier companion to
# lvr_stage2_preflight.sbatch: it LOADS the 3B on CPU and runs a tiny LVR generate so the
# patched forward + rope + step-decoding actually EXECUTE — catching runtime forward drift
# (e.g. the rope_deltas AttributeError from transformers 4.54.0) that the signature-level
# preflight cannot. Green here + green preflight => the forward path is validated off-GPU.
#   Submit:  sb slurm/lvr_stage2_forward_smoke.sbatch        # DO NOT agent-submit (§3-A)
#SBATCH -J launch
#SBATCH -p cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -t 0-02:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log
set -uo pipefail
[ -f .env ] && { set -a; source .env; set +a; }
slack_notify(){ [ -z "${SLACK_WEBHOOK_URL:-}" ]&&return 0; curl -s -X POST -H 'Content-type: application/json' --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL">/dev/null; }
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO" || exit 11
[ -d .venv ] || { slack_notify ":x: $REPO/.venv missing"; exit 12; }
source .venv/bin/activate
module purge; module load cuda/12.6; export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.6
export CUDA_VISIBLE_DEVICES=""        # force CPU — we execute the forward on CPU
ROOT=/data/joonhee/lvr
mkdir -p $ROOT/.cache/{tmp,triton,torch,torch-extensions}
export TMPDIR=$ROOT/.cache/tmp TRITON_CACHE_DIR=$ROOT/.cache/triton TORCH_HOME=$ROOT/.cache/torch TORCH_EXTENSIONS_DIR=$ROOT/.cache/torch-extensions
# $REPO/src/train for the bare `from monkey_patch_forward_lvr_rl import ...` (script-launch parity)
export PYTHONPATH=$REPO:$REPO/src:$REPO/src/train:${PYTHONPATH:-}
export WANDB_MODE=disabled
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}   # let torch CPU matmuls use the allocation

STAGE1=$(find $ROOT/ckpts/stage1 -maxdepth 3 -type d -name 'checkpoint-2500' 2>/dev/null | head -1)
[ -z "$STAGE1" ] && { slack_notify ":x: forward-smoke: stage1 checkpoint-2500 not found under $ROOT/ckpts/stage1"; exit 14; }
slack_notify ":test_tube: START lvr-stage2-FORWARD-SMOKE (cpu) | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME} | init=$STAGE1"

# sweep batch sizes: n=1 is unpadded; n>=2 with mixed prompt lengths -> left-padding.
# The first GPU run died with a multinomial 'inf/nan' assert during the batched rollout;
# this sweep tests whether left-padding (fully-masked attention rows -> NaN) reproduces it
# on CPU. DTYPE/NLIST overridable: e.g. DTYPE=bf16 NLIST=1,4 sb ...
python slurm/preflight_forward_smoke.py \
  --checkpoint "$STAGE1" \
  --data_path "$ROOT/data/virl39k.json" \
  --image_folder "$ROOT/images/" \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --image_min_pixels $((128*28*28)) \
  --image_max_pixels $((2560*28*28)) \
  --lvr_steps 8 \
  --max_new_tokens 4 \
  --dtype "${DTYPE:-fp32}" \
  --n_prompts_list "${NLIST:-1,2,4}"
rc=$?
[ $rc -eq 0 ] && slack_notify ":white_check_mark: lvr-stage2-FORWARD-SMOKE GREEN — patched forward+rope execute on CPU, safe for GPU" \
              || slack_notify ":x: lvr-stage2-FORWARD-SMOKE FAILED rc=$rc — runtime forward bug, fix before GPU (logs/launch_${SLURM_JOB_ID}.log)"
exit $rc
