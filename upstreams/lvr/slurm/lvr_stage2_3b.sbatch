#!/bin/bash
# LVR Stage-2 GRPO_latent RL 3B (faithful) — gpu-4farm 4xH100. Reproduces
# scripts/finetune_lvr_stage2_3b.sh VERBATIM except cluster paths +
# --online_checkpoint False. Init = Stage-1 checkpoint-2500 (auto-found).
# Rollouts use in-process HF generate (no vLLM). DeepSpeed ZeRO-2.
# RL is heavy (8 gens x teacher-force replay/prompt) — budget 2-3 resume jobs.
# Submit fresh:  sb slurm/lvr_stage2_3b.sbatch     # DO NOT agent-submit (§3-A)
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
[ -d .venv ] || { slack_notify ":x: $REPO/.venv missing"; exit 12; }
source .venv/bin/activate
module purge; module load cuda/12.6; export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.6
ROOT=/data/joonhee/lvr
mkdir -p $ROOT/.cache/{tmp,triton,torch,torch-extensions} $ROOT/wandb
export TMPDIR=$ROOT/.cache/tmp TRITON_CACHE_DIR=$ROOT/.cache/triton TORCH_HOME=$ROOT/.cache/torch TORCH_EXTENSIONS_DIR=$ROOT/.cache/torch-extensions WANDB_DIR=$ROOT/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src:${PYTHONPATH:-}
export WANDB_PROJECT="LVR-Qwen25-VL-3B-SFT-STAGE-2-39k"

# auto-find Stage-1 checkpoint-2500
STAGE1=$(find $ROOT/ckpts/stage1 -maxdepth 3 -type d -name 'checkpoint-2500' | head -1)
[ -z "$STAGE1" ] && { slack_notify ":x: stage1 checkpoint-2500 not found under $ROOT/ckpts/stage1 — run/finish Stage-1 first"; exit 14; }
OUTPUT_DIR=$ROOT/ckpts/stage2
RUN_NAME="Stage2_3B_decodingBysteps_max8lvrSteps_LR5e-6_TEMP0.9_stage1Steps2500"
slack_notify ":rocket: START lvr-stage2-grpo 3B | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME} | init=$STAGE1"

deepspeed --num_gpus=4 src/train/train_grpo.py \
  --run_name "$RUN_NAME" \
  --deepspeed scripts/zero2.json \
  --online_checkpoint False \
  --checkpoint_name "$STAGE1" \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --data_path "$ROOT/data/virl39k.json" \
  --image_folder "$ROOT/images/" \
  --freeze_vision_tower True --freeze_merger True --freeze_llm False \
  --bf16 True --fp16 False --disable_flash_attn2 False \
  --output_dir "$OUTPUT_DIR" \
  --temperature 0.9 --num_train_epochs 2 --num_generations 8 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 2 \
  --max_completion_length 512 --max_prompt_length 4096 \
  --image_min_pixels $((128*28*28)) --image_max_pixels $((2560*28*28)) \
  --learning_rate 5e-6 --remove_unused_columns False \
  --weight_decay 0.1 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --logging_steps 1 --tf32 False --gradient_checkpointing True --report_to wandb \
  --lazy_preprocess True --save_strategy steps --save_steps 100 --save_total_limit 50 \
  --dataloader_num_workers 8 --decoding_strategy steps --lvr_steps 8
rc=$?
[ $rc -eq 0 ] && slack_notify ":tada: DONE lvr-stage2-grpo 3B | ckpt: $OUTPUT_DIR" || slack_notify ":warning: lvr-stage2-grpo exit=$rc | partial: $OUTPUT_DIR"
exit $rc
