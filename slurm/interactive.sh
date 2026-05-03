#!/bin/bash
# Interactive 1-GPU debug — DEPRECATED on bioai.
#
# The bioai admin has FORBIDDEN salloc/srun (per ~/.claude/docs/bioai_cluster_spec.md
# §2-D). Login-node profile.d scripts alias both to "echo Interactive jobs are
# not allowed. Please use sbatch." This script will not run.
#
# Replacement: write a tiny throwaway sbatch that prints the diagnostics you
# want and reads back the log. See slurm/probe_1gpu.sbatch (TODO if needed).
#
# Example pattern:
#   cat > slurm/probe_1gpu.sbatch <<'SBATCH'
#   #!/bin/bash
#   #SBATCH -p gpu-4farm
#   #SBATCH --account=gpu
#   #SBATCH --gres=gpu:h100:1
#   #SBATCH --cpus-per-task=28
#   #SBATCH --mem=128G
#   #SBATCH -t 0:30:00
#   #SBATCH -o logs/%x_%j.log
#   set -euo pipefail
#   set -a; source .env; set +a
#   export MACHINE=bioai
#   module purge && module load cuda/12.6
#   export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.6
#   nvidia-smi
#   nvidia-smi topo -m
#   uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
#   SBATCH
#   sb slurm/probe_1gpu.sbatch    # only after explicit user "go"
#
# For genuine interactive debugging the user must arrange admin permission.

set -euo pipefail

cat <<'EOF'
ERROR: Interactive sessions via salloc/srun are FORBIDDEN on bioai.
       See ~/.claude/docs/bioai_cluster_spec.md §2-D.

       Use a one-shot sbatch instead — see comments in this file for an
       example, and "tail -F logs/<job>.log" to read the output.
EOF
exit 1
