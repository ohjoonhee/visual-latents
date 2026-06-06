#!/bin/bash
# LVR Stage-1 SFT 3B (faithful) — gpu-4farm 4xH100. Reproduces
# scripts/finetune_lvr_stage1_3b.sh VERBATIM except: cluster paths,
# --online_checkpoint False, and grad_accum 8->16 (4-GPU eff batch = 1x4x16 = 64,
# matching the script's 8-GPU 1x8x8=64). DeepSpeed ZeRO-3-offload.
# Submit fresh:   sb slurm/lvr_stage1_3b.sbatch
# Smoke (gate):   SMOKE=1 sb slurm/lvr_stage1_3b.sbatch     # 20 steps, isolated _SMOKE dir
# Submit resume:  RESUME=1 sb slurm/lvr_stage1_3b.sbatch     # DO NOT agent-submit (§3-A)
#SBATCH -J launch
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
slack_notify(){ [ -z "${SLACK_WEBHOOK_URL:-}" ]&&return 0; curl -s -X POST -H 'Content-type: application/json' --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL">/dev/null; }
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO" || exit 11
[ -d .venv ] || { slack_notify ":x: $REPO/.venv missing — run lvr_env_build first"; exit 12; }
source .venv/bin/activate
module purge; module load cuda/12.6; export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.6
ROOT=/data/joonhee/lvr
mkdir -p $ROOT/.cache/{tmp,triton,torch,torch-extensions} $ROOT/wandb
export TMPDIR=$ROOT/.cache/tmp TRITON_CACHE_DIR=$ROOT/.cache/triton TORCH_HOME=$ROOT/.cache/torch TORCH_EXTENSIONS_DIR=$ROOT/.cache/torch-extensions WANDB_DIR=$ROOT/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="LVR-Qwen25-VL-3B-SFT-STAGE-1-450k"

SMOKE="${SMOKE:-0}"
# SMOKE=1 → fast gate: 20 steps, frequent saves (exercise the ckpt-write path),
# isolated _SMOKE run dir so it can't collide with the real run or trip its gate.
MAX_STEPS=2500; SAVE_STEPS=500; SAVE_LIMIT=10; SMOKE_TAG=""
if [ "$SMOKE" = "1" ]; then MAX_STEPS=20; SAVE_STEPS=10; SAVE_LIMIT=3; SMOKE_TAG="_SMOKE"; fi
RUN_NAME="Stage1_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128${SMOKE_TAG}"
OUTPUT_DIR=$ROOT/ckpts/stage1
SAVE_DIR="$OUTPUT_DIR/Qwen2.5-VL-3B-Instruct/$RUN_NAME"   # train_lvr nests by model/run
RESUME="${RESUME:-0}"
# resume gate (no-clobber on fresh; --checkpoint_name latest on resume, per script comment)
RESUME_ARG=""
NCK=$(find "$SAVE_DIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | wc -l)
if [ "$RESUME" = "1" ]; then
  [ "$NCK" -lt 1 ] && { slack_notify ":x: stage1 RESUME=1 but no ckpt in $SAVE_DIR"; exit 16; }
  LATEST=$(find "$SAVE_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -t- -k2 -n | tail -1)
  RESUME_ARG="--checkpoint_name $(basename "$LATEST")"
elif [ "$NCK" -ge 1 ]; then
  slack_notify ":x: stage1 fresh but $SAVE_DIR has $NCK ckpts — set RESUME=1 or clear dir"; exit 17
fi
slack_notify ":rocket: START lvr-stage1-sft 3B | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME} | RESUME=$RESUME SMOKE=$SMOKE | steps=$MAX_STEPS"

deepspeed --num_gpus=4 src/train/train_lvr.py \
  --run_name "$RUN_NAME" --coconut True --loss_lvr_fct mse \
  --deepspeed scripts/zero3_offload.json \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --data_path "$ROOT/data/meta_data_lvr_sft_stage1.json" \
  --remove_unused_columns False --lvr_head False \
  --freeze_vision_tower True --freeze_merger True --freeze_llm False \
  --max_steps $MAX_STEPS --learning_rate 1e-5 --loss_lvr_lambda 0.1 \
  --bf16 True --fp16 False --disable_flash_attn2 False \
  --online_checkpoint False --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --image_min_pixels $((128*28*28)) --image_max_pixels $((5120*28*28)) \
  --weight_decay 0.1 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --logging_steps 1 --tf32 False --gradient_checkpointing True --report_to wandb \
  --lazy_preprocess True --save_strategy steps --save_steps $SAVE_STEPS --save_total_limit $SAVE_LIMIT \
  --dataloader_num_workers 8 --enable_data_packing True --max_packed_tokens 16384 \
  --random_seed 42 --long_seq_threshold 4096 --max_instance_per_batch 4 $RESUME_ARG
rc=$?
[ $rc -eq 0 ] && slack_notify ":tada: DONE lvr-stage1-sft 3B | ckpt: $SAVE_DIR" || slack_notify ":warning: lvr-stage1-sft exit=$rc | partial: $SAVE_DIR"
exit $rc
