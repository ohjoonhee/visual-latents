#!/bin/bash
# Open an interactive 1-GPU debug shell on bioai.
# Usage:    bash slurm/interactive.sh
# NEVER auto-run by an agent.
#
# Lands you on a gpu-1farm node with 1 H100, then loads cuda + sources .env.
# Exit the shell to release the allocation.

set -euo pipefail

salloc \
  -p gpu-1farm \
  --account=gpu \
  --gres=gpu:1 \
  --cpus-per-task=28 \
  --mem=224G \
  -t 1:00:00 \
  bash -c '
    set -a; source .env; set +a
    export MACHINE=bioai
    module load cuda/12.6
    export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.6
    echo "[interactive] node=$SLURMD_NODENAME jobid=$SLURM_JOB_ID"
    echo "[interactive] cd $(pwd); exit to release allocation."
    exec bash -i
  '
