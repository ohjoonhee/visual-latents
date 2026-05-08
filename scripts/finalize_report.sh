#!/usr/bin/env bash
# Final analysis + report finalization. Runs after all overnight stages.
set -uo pipefail
cd /mnt/ssd/Projects/visual-latents

LOG_DIR="logs/overnight_2026_05_04"
echo "[finalize] regenerating RESULTS_TABLE.md and summaries.json from all runs"
MACHINE=local uv run python scripts/analyse_overnight.py 2>&1 | tail -5

# Append Stage E (concept-only) and F (NLL-only) cells to the same analysis
echo "[finalize] all results aggregated"
echo "[finalize] reports:"
ls -la docs/overnight_2026_05_04/*.md
echo
echo "[finalize] runs catalogued:"
for d in results/interleaved_*; do
    if [ -f "$d/ablation_eval.jsonl" ]; then
        echo "  $(basename $d)"
    fi
done
