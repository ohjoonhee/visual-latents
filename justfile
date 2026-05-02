# Common entry commands. Run `just` to see the list.

default:
    @just --list

# Sync deps from pyproject.toml + uv.lock
sync:
    uv sync

# Run all infra tests (paths, config, curriculum)
test:
    uv run pytest tests/

# Lint (ruff)
lint:
    uv run ruff check src/ tests/ scripts/

# Format (ruff)
fmt:
    uv run ruff format src/ tests/ scripts/

# Smoke test — 10 steps on local A6000 (requires .env with MACHINE=local)
smoke:
    uv run python -m vl.train --config configs/smoke.yaml

# Run a round-3 cell locally (only feasible at reduced batch/K — full recipe needs 4×H100)
round3-local cell:
    uv run python -m vl.train --config configs/round3/{{cell}}.yaml

# Pre-download model weights into HF cache (works on local + bioai)
download-models:
    uv run python scripts/download_models.py

# Pull results back from bioai (rsync). Requires SSH alias `bioai` configured.
sync-from-cluster:
    bash scripts/sync_from_cluster.sh

# Show what `sb slurm/round3_cell.sbatch configs/round3/C1_full.yaml` WOULD submit
# (NEVER actually submits — that's a manual `sb` call after explicit approval)
plan-round3 cell:
    @echo "Would submit: sb slurm/round3_cell.sbatch configs/round3/{{cell}}.yaml"
    @echo "Resource cost: 4 GPUs × ~6h on gpu-4farm"
    @cat slurm/round3_cell.sbatch
