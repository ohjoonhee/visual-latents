#!/usr/bin/env bash
# Run the full Phase 0 pipeline. Assumes downloads already complete and
# images.zip is unpacked.
set -euo pipefail

cd "$(dirname "$0")"
PY=.venv-monet/bin/python

mkdir -p logs

echo "=== Step 0: unpack images.zip if needed ==="
$PY unpack.py 2>&1 | tee logs/unpack.log

echo "=== Step 1: extract latents (4 cells × 100) ==="
for STAGE in stage2 stage3; do
  for SUBSET in Visual_CoT Zebra_CoT_count; do
    OUT="latents/${STAGE}/${SUBSET}"
    if [ -d "$OUT" ] && [ "$(ls "$OUT" 2>/dev/null | wc -l)" -ge 90 ]; then
      echo "  [skip] $OUT already has $(ls "$OUT" | wc -l) files"
      continue
    fi
    echo "  [extract] stage=$STAGE subset=$SUBSET"
    $PY extract_latents.py --stage "$STAGE" --subset "$SUBSET" --n 100 --out "$OUT" 2>&1 | tee "logs/extract_${STAGE}_${SUBSET}.log"
  done
done

echo "=== Step 2: run ablation ==="
$PY ablation.py --max_per_cell 100 2>&1 | tee logs/ablation.log

echo "=== Step 3: build report ==="
$PY build_report.py 2>&1 | tee logs/build_report.log

echo "=== Step 4: cleanup latent dumps ==="
# Keep them for now until report is committed; real cleanup is manual.
echo "  (cleanup deferred until user reviews REPORT.md)"

echo "=== DONE ==="
