#!/bin/bash
# Cluster probe — closes the "Open / Unverified" TODO items in
# ~/.claude/docs/bioai_cluster_spec.md that need a GPU-node job to resolve:
#
#   1. NVLink topology between the 8 H100s on a single XE9680 node.
#   2. GPU-node /tmp realised mount type/size + whether the 2× 3.84 TB
#      physical NVMe drives are user-visible anywhere (e.g. /scratch, /local).
#   3. GPU-node CPU model name (Slurm says CoresPerSocket=56; PDF says
#      Xeon Platinum 8480+; live /proc/cpuinfo would close the question).
#   6. GPU-node internet reachability (huggingface / github / pypi /
#      api.wandb.ai / download.pytorch.org/whl/cu126).
#
# NOT probed (need slurmdbd or admin, no sbatch can resolve them):
#   4. Per-user QoS limits on gpu4 / gpu8 / steinegger.
#   5. Whether joonhee holds steinegger or normal_h200 QoS.
#
# This is a read-only diagnostic — no training, no downloads, no writes
# outside /tmp and a single touch-test under /data/joonhee/probe/.
# Expected wall: < 30 s.  Cost ceiling: 5 min walltime × 4 H100 = 20 GPU-min,
# real cost ~2 GPU-min.
#
# Submit with: sb slurm/cluster_probe_gpunode.sbatch
# DO NOT agent-submit (project-owner rule §3-A in bioai_cluster_spec.md).
#
#SBATCH -J vl-probe-gpunode
#SBATCH -p gpu-4farm
#SBATCH --account=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -uo pipefail
[ -f .env ] && { set -a; source .env; set +a; }

# /tmp on compute nodes is RAM-tmpfs and counts against --mem= — redirect.
mkdir -p /data/joonhee/visual-latents/.cache/tmp
export TMPDIR=/data/joonhee/visual-latents/.cache/tmp

# cuda module is loaded so `nvidia-smi nvlink` resolves identically to a
# real training job's environment (nvidia-smi is in /usr/bin too, but the
# load matches §3-F template style).
module purge
module load cuda/12.6

slack_notify () {
  [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL" >/dev/null
}
LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
SCRIPT_PATH="${ORIG_SBATCH_SCRIPT:-<unknown>}"

# §3-H mandatory START format — do not modify emoji or field order.
slack_notify ":rocket: JOB START | ${SLURM_JOB_NAME} | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
script: \`${SCRIPT_PATH}\`
log: \`${ABSLOG}\`
follow: \`tail -n 200 -F ${ABSLOG}\`"

hr () { printf '\n========== %s ==========\n' "$1"; }

hr "Identity"
echo "hostname:       $(hostname)"
echo "SLURMD_NODENAME: ${SLURMD_NODENAME:-?}"
echo "date:           $(date -Is)"
echo "uname -a:       $(uname -a)"
[ -f /etc/redhat-release ] && echo "os-release:     $(cat /etc/redhat-release)"

hr "CPU (TODO #3 — GPU-node CPU model)"
lscpu | sed -n '1,25p'
echo "---"
grep -m1 -E '^model name' /proc/cpuinfo

hr "Memory"
free -h

hr "GPUs + driver/CUDA version"
nvidia-smi || echo "[ERR] nvidia-smi failed"

hr "GPU topology (TODO #1 — NVLink between H100s on XE9680)"
nvidia-smi topo -m || echo "[ERR] nvidia-smi topo -m failed"

hr "NVLink per-link status (TODO #1 cont.)"
nvidia-smi nvlink --status || echo "[ERR] nvidia-smi nvlink --status failed"

hr "Filesystem mounts (TODO #2 — is the 2× 3.84 TB NVMe user-visible?)"
df -hT
echo "---"
echo "/tmp specifically:"
df -hT /tmp
echo "---"
echo "/dev/shm specifically:"
df -hT /dev/shm
echo "---"
echo "Block devices (look for ~3.84T NVMe):"
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE 2>/dev/null || echo "[INFO] lsblk unavailable"
echo "---"
echo "Mounts matching nvme/local/scratch/data:"
mount | grep -E 'nvme|local|scratch|/data|/home' || echo "[INFO] no matches"
echo "---"
echo "Top-level dirs that often host node-local scratch:"
ls -ld /scratch /local /nvme /mnt /tmp 2>/dev/null

hr "GPFS visibility (sanity — /home and /data reachable from GPU node)"
ls -ld /home/joonhee /data/joonhee 2>/dev/null
TOUCH="/data/joonhee/probe/gpunode_${SLURM_JOB_ID}.txt"
mkdir -p /data/joonhee/probe
echo "probe from ${SLURMD_NODENAME} at $(date -Is)" > "$TOUCH" \
  && echo "[OK] wrote $TOUCH" \
  || echo "[ERR] failed to write $TOUCH"

hr "Network interfaces (IB expected per spec)"
ip -4 addr show | sed -n '1,40p'

hr "Internet reachability (TODO #6 — GPU-node external network)"
probe_url () {
  local url="$1"
  local out
  out=$(curl -sI --max-time 10 -o /dev/null -w 'http_code=%{http_code} time=%{time_total}s' "$url" 2>&1) \
    && echo "  $url -> $out" \
    || echo "  $url -> [FAIL] $out"
}
probe_url "https://huggingface.co"
probe_url "https://github.com"
probe_url "https://pypi.org"
probe_url "https://api.wandb.ai"
probe_url "https://download.pytorch.org/whl/cu126/"

hr "DNS"
getent hosts huggingface.co github.com pypi.org api.wandb.ai 2>&1 | sed -n '1,20p'

hr "Done"
rc=$?

# §3-H mandatory DONE / FAIL format — do not modify.
[ $rc -eq 0 ] \
  && slack_notify ":white_check_mark: DONE ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID})" \
  || slack_notify ":x: FAIL ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID}) exit=${rc}"
exit $rc
