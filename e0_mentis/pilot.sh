#!/usr/bin/env bash
# E0 pilot: baseline vs true-oracle arm on MentisOculi's three geometric tasks.
# Purpose is twofold — validate the harness end-to-end, and measure whether the
# local model has any headroom above chance. If both arms sit at the random
# baseline the comparison is uninformative and E0 needs a stronger model.
set -uo pipefail

PY=/mnt/ssd/Projects/visual-latents/.venv/bin/python
WT=/mnt/ssd/Projects/visual-latents/.claude/worktrees/direction-drift-2026-08
BENCH=/mnt/ssd/Projects/mentis-oculi/datasets
N=${N:-30}
LEVELS=${LEVELS:-1}
MODEL=${MODEL:-Qwen/Qwen3-VL-4B-Instruct}
SLUG=$(basename "$MODEL")

cd "$WT" || exit 1
for task in paper-fold hinge-folding form-board; do
  for arm in simple visual_cot; do
    echo "=== $task / $arm ==="
    $PY e0_mentis/run_e0.py --task "$task" --arm "$arm" \
        --levels "$LEVELS" --n "$N" --model "$MODEL" 2>&1 | grep -v "Loading weights"
  done
done

echo
echo "############ SCORES ############"
for task in paper-fold hinge-folding form-board; do
  for arm in simple visual_cot; do
    for lvl in ${LEVELS//,/ }; do
      f="$WT/e0_mentis/responses/$SLUG/$arm/$task/level_$(printf %02d "$lvl")/responses_0.json"
      [ -f "$f" ] || continue
      echo "--- $task / $arm / L$lvl ---"
      (cd "$BENCH/$task" && $PY evaluate_responses.py --responses "$f" 2>&1 \
        | grep -E "Accuracy:|Random baseline:|95% CI" | head -4)
    done
  done
done
