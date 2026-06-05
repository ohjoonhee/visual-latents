#!/bin/bash
# LVR data staging (cpu-short) — off login node per §3-D. Stages to /data/joonhee/lvr/:
#   data/   : authors' faithful JSONs (gdown lvr_data.zip) + cluster manifest
#   images/ : viscot/<src>/...  (deepcs233/Visual-CoT, ~142GB) + ViRL39K/... (TIGER-Lab, 1.8GB)
# Self-verifying: aborts if a sample JSON image path doesn't resolve.
# Submit:  sb slurm/lvr_data_stage.sbatch
#SBATCH -J launch
#SBATCH -p cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log
set -uo pipefail
[ -f .env ] && { set -a; source .env; set +a; }
slack_notify(){ [ -z "${SLACK_WEBHOOK_URL:-}" ]&&return 0; curl -s -X POST -H 'Content-type: application/json' --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK_URL">/dev/null; }
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO" || exit 11
ROOT=/data/joonhee/lvr; PY=$REPO/.venv/bin/python; UV=$HOME/.local/bin/uv
export HF_HUB_DISABLE_XET=1
mkdir -p $ROOT/{data,images,ckpts,wandb,.cache/tmp} $ROOT/images
slack_notify ":rocket: START lvr-data-stage | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}"

# ---- 1. JSONs (authors' faithful data) ----
if [ ! -f $ROOT/data/viscot_363k_lvr_formatted.json ]; then
  $UV run --with gdown gdown "1RUbKMQU3H7u8iWDqpDv8aiYDMlUaDIlE" -O $ROOT/.cache/lvr_data.zip
  unzip -o -q $ROOT/.cache/lvr_data.zip -d $ROOT/.cache/lvr_data
  cp $ROOT/.cache/lvr_data/lvr_data/lvr_data/{viscot_363k_lvr_formatted.json,viscot_sroie_dude_lvr_formatted.json,virl39k.json} $ROOT/data/
fi
cp $REPO/data/meta_data_lvr_sft_stage1.json $ROOT/data/   # cluster-path manifest (vendored)

# ---- 2. Visual-CoT images (~142GB) -> images/viscot/ ----
SAMPLE=$($PY -c "import json;print(json.load(open('$ROOT/data/viscot_363k_lvr_formatted.json'))[0]['image'][0])")  # e.g. viscot/flickr30k/xxx.jpg
if [ ! -f "$ROOT/images/$SAMPLE" ]; then
  DL=$ROOT/.cache/viscot_dl; mkdir -p "$DL"
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('deepcs233/Visual-CoT',repo_type='dataset',allow_patterns=['cot_images_tar_split/*'],local_dir='$DL',max_workers=8)"
  cat "$DL"/cot_images_tar_split/cot_images_* > "$DL/cot_images.tar"
  mkdir -p "$DL/extracted" && tar xf "$DL/cot_images.tar" -C "$DL/extracted"
  # locate the dir that contains the per-source subdirs (flickr30k, gqa, ...) and expose it as images/viscot
  SRCDIR=$(dirname "$(find "$DL/extracted" -maxdepth 3 -type d -name flickr30k | head -1)")
  [ -z "$SRCDIR" ] && { slack_notify ":x: Visual-CoT extract: flickr30k dir not found"; exit 14; }
  ln -sfn "$SRCDIR" "$ROOT/images/viscot"
  rm -f "$DL/cot_images.tar"   # reclaim ~142GB
fi

# ---- 3. ViRL39K images (1.8GB) -> images/ViRL39K/ ----
if [ ! -d "$ROOT/images/ViRL39K" ] || [ -z "$(ls -A $ROOT/images/ViRL39K 2>/dev/null)" ]; then
  VDL=$ROOT/.cache/virl_dl; mkdir -p "$VDL"
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('TIGER-Lab/ViRL39K',repo_type='dataset',allow_patterns=['images.zip'],local_dir='$VDL')"
  mkdir -p "$ROOT/images/ViRL39K"; unzip -o -q "$VDL/images.zip" -d "$ROOT/images/ViRL39K_unz"
  IMGDIR=$(dirname "$(find "$ROOT/images/ViRL39K_unz" -maxdepth 3 -type f -name '*.jpg' | head -1)")
  ln -sfn "$IMGDIR" "$ROOT/images/ViRL39K"
fi

# ---- 4. verify sample paths resolve ----
S1="$ROOT/images/$SAMPLE"
S2="$ROOT/images/$($PY -c "import json;print(json.load(open('$ROOT/data/virl39k.json'))[0]['image'])")"
[ -f "$S1" ] || { slack_notify ":x: stage1 sample missing: $S1"; exit 15; }
[ -f "$S2" ] || { slack_notify ":warning: stage2 ViRL39K sample missing: $S2 (check layout)"; }
echo "OK: $S1 ; $S2"
slack_notify ":white_check_mark: DONE lvr-data-stage (jobid=${SLURM_JOB_ID}) | viscot+ViRL39K staged"
