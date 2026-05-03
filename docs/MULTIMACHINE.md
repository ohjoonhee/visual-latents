# Multi-machine: local + bioai

The repo is git-synced between two machines:

| Machine | Role | Hardware | Storage layout |
|---|---|---|---|
| `local` | smoke tests, analysis, doc edits | 1× A6000 49 GB | repo dir under `/mnt/ssd/Projects/visual-latents/`; data/ckpt/results/HF cache all default |
| `bioai` | real training + eval | 4× H100 per job on `gpu-4farm` (1× H100 jobs use `gpu-4farm --gres=gpu:1`; `gpu-1farm` was removed by admin) | repo under `~/projects/visual-latents/` (GPFS home, code only); data/ckpt/results under `/data/joonhee/vl/` (GPFS data, 1.6 PB); HF cache `/data/joonhee/.cache/huggingface` (env in `~/.bashrc`) |

`vl.paths` reads `MACHINE` env var (set in `.env`) and resolves all storage
roots accordingly. Same code on both sides; no path edits.

## What lives in git

- `src/`, `configs/`, `scripts/`, `slurm/`, `tests/`, `docs/`
- `pyproject.toml`, `uv.lock`, `justfile`
- `.env.example` (template); `.gitignore`, `.gitattributes`

## What does NOT live in git

- `data/`, `checkpoints/`, `results/`, `logs/`, `wandb/`, `outputs/`
- `.venv/` (per-machine; recreated from `uv.lock`)
- `.env`, `.env.local` (per-machine; secrets)
- `*.pt`, `*.safetensors`, `*.bin`, `*.ckpt`

## Sync workflow

**Code (local ↔ bioai):**
```bash
# local
git push
ssh bioai 'cd ~/projects/visual-latents && git pull && uv sync'
```

**Models (per-machine; downloaded independently):**
```bash
# either machine
uv run python scripts/download_models.py
```

**Results (bioai → local for analysis):**
```bash
# local
bash scripts/sync_from_cluster.sh
# rsyncs /data/joonhee/vl/results/ → ./results/
# excludes *.pt and checkpoints/ — those stay on bioai
```

**Checkpoints:** kept on bioai under `/data/joonhee/vl/checkpoints/`. Pull
specific checkpoints for local debugging only when needed:
```bash
rsync -av bioai:/data/joonhee/vl/checkpoints/<run_id>/ checkpoints/<run_id>/
```

## First-time setup

### local
```bash
cd /mnt/ssd/Projects/visual-latents
cp .env.example .env
# edit .env: MACHINE=local, WANDB_API_KEY=...
uv sync
uv run pytest tests/                        # paths + config + curriculum
uv run python scripts/download_models.py    # warms ~/.cache/huggingface
```

### bioai (after initial git push from local)
```bash
ssh bioai
git clone https://github.com/ohjoonhee/visual-latents ~/projects/visual-latents
cd ~/projects/visual-latents
cp .env.example .env
# edit .env: MACHINE=bioai, WANDB_API_KEY=..., SLACK_WEBHOOK_URL=...
uv sync                                     # uv installed at ~/.local/bin/uv
uv run pytest tests/
# then submit a probe sbatch (see SLURM.md) — only after explicit user approval
```
