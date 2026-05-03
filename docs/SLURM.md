# Slurm — bioai cluster

Project-specific cluster usage. Read `~/.claude/docs/bioai_cluster_spec.md`
for the full cluster spec and the **MANDATORY usage policy** section. The
cluster spec is the single source of truth; this file just summarizes how
the project's templates use it.

## Hard rules (recap from cluster spec)

**Server-admin rules** (binding on every job):

- 4-GPU jobs MUST use `gpu-4farm`. Every GPU job MUST set `--account=gpu`.
- `gpu-1farm` was REMOVED. Working practice for 1-GPU work is
  `-p gpu-4farm --gres=gpu:1`. Confirm with admin before scaling 1-GPU
  campaigns.
- `gpu-8farm`, `gpu-h200`, and any `-bf` variant are FORBIDDEN unless
  admin authorizes that specific job.
- Interactive `salloc`/`srun` are FORBIDDEN. Every workload — including
  debug/sanity runs — goes through `sbatch`.
- Login node has tight per-user systemd limits (~16 % CPU, ~16 GB mem).
  Don't run multiprocessing/DataLoader on login.
- For 4-GPU jobs, admin guideline is `--cpus-per-task` between 32 and 64
  (NOT the 112 default).
- Cache redirects (`TMPDIR`, `TRITON_CACHE_DIR`, `TORCH_HOME`,
  `TORCH_EXTENSIONS_DIR`, `WANDB_DIR`) are mandatory — `/tmp` on compute
  nodes is RAM-tmpfs and counts against `--mem=`.

**Project-owner rule:**

- No agent-initiated submission. Every `sb`/`sbatch` requires explicit user
  approval for that specific submission.

## Templates in this repo

| File | Purpose | Partition | GPUs | Walltime |
|---|---|---|---|---|
| `slurm/round3_cell.sbatch` | one round-3 cell | gpu-4farm | 4 | 23h30m |
| `slurm/eval.sbatch` | full eval suite for one checkpoint | gpu-4farm (`--gres=gpu:1`) | 1 | 4h |
| `slurm/interactive.sh` | DEPRECATED (salloc forbidden); see comments | — | — | — |

All sbatch templates:

- Source `.env` (`set -a; source .env; set +a`).
- `module purge && module load cuda/12.6` then `export CUDA_HOME=...`
  (cuda module does NOT set `$CUDA_HOME` itself — see cluster spec gotcha).
- Set `MACHINE=bioai` for `vl.paths`.
- Apply the cache redirect block (TMPDIR, TRITON_CACHE_DIR, TORCH_HOME,
  TORCH_EXTENSIONS_DIR, WANDB_DIR → `/data/joonhee/vl/...`).
- Use `sb` wrapper (preserves `$ORIG_SBATCH_SCRIPT` for Slack notify).
- Slack-notify on start, end, fail, time-limit (using `$SLACK_WEBHOOK_URL`
  from `.env`).
- **No `--dependency` auto-resubmit chains.** If `--max-time` is hit, the
  script Slack-notifies but does NOT submit a continuation — that requires
  manual review and approval.

## Submitting

```bash
# Single round-3 cell (after explicit user "go"):
sb slurm/round3_cell.sbatch configs/round3/C1_full.yaml

# Eval one checkpoint (after explicit user "go"):
sb slurm/eval.sbatch /data/joonhee/vl/checkpoints/<run_id>/

# Interactive debugging is forbidden on this cluster — submit a sbatch instead.
```

## Watching

```bash
sq                              # alias: squeue | grep $USER
scontrol show job <jobid> -dd
tail -F logs/<jobname>_<jobid>.log
```

## Resource cost reference

| Job | gpu × time | Reasoning |
|---|---|---|
| Round-3 cell C1 (full) | 4 × ~6h | per `inherited/ROUND3_POC_DESIGN.md` §10 (~24h compute total / 5 cells) |
| Round-3 cells C2-C5 | 4 × ~5h each | smaller — drop one component |
| Round-3 full sweep | 4 × ~28h cumulative (parallel: 6h wallclock if 5 slots free) | gpu-4farm has ~20-job backlog observed; expect serialized |
| Eval per checkpoint | 1 × ~3h | held-out + transfer + steering + 5K stress |
| M1 (100K) per chain link | 4 × 23h | self-terminates; manual resubmit |
| M1 total | 4 × ~14 days × ~14 chains | 24/7 throughput; many days wallclock |

## Verification probes (proposed; require user approval to submit)

1. CPU sbatch (5 min on `cpu-short`): `uv sync` in a fresh clone — confirms
   `pyproject.toml` resolves on bioai.
2. 1-GPU sbatch (30 min on `gpu-4farm --gres=gpu:1`): load Qwen2.5-VL-7B
   + a batch of 1 image + run forward — confirms cuda + transformers + the
   model load path.
3. 4-GPU sbatch (10 min on `gpu-4farm`): just `nvidia-smi topo -m` and
   `uv run pytest tests/` — confirms infrastructure end-to-end.

These are templates; they are NOT submitted by the agent.
