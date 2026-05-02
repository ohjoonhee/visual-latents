# Slurm — bioai cluster

Project-specific cluster usage. Read `~/.claude/docs/bioai_cluster_spec.md`
for the full cluster spec and the **MANDATORY usage policy** section.

## Hard rules (recap from cluster spec)

**Server-admin rules** (binding on every job):
- 1 GPU → `gpu-1farm` ONLY (`gpu-4farm --gres=gpu:1` is FORBIDDEN)
- 4 GPUs → `gpu-4farm` ONLY
- `gpu-8farm`, `gpu-h200`, `*-bf` partitions are FORBIDDEN unless per-job
  admin authorization

**Project-owner rule:**
- No agent-initiated submission. Every `sb`/`sbatch` requires explicit user
  approval for that specific submission.

## Templates in this repo

| File | Purpose | Partition | GPUs | Walltime |
|---|---|---|---|---|
| `slurm/round3_cell.sbatch` | one round-3 cell | gpu-4farm | 4 | 23h30m |
| `slurm/eval.sbatch` | full eval suite for one checkpoint | gpu-1farm | 1 | 4h |
| `slurm/interactive.sh` | salloc 1-GPU debug shell | gpu-1farm | 1 | 1h |

All templates:
- Source `.env` first (`set -a; source .env; set +a`).
- `module load cuda/12.6` then `export CUDA_HOME=...` (cuda module does NOT
  set `$CUDA_HOME` itself — see cluster spec gotcha).
- Set `MACHINE=bioai` for `vl.paths`.
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

# Interactive debugging:
bash slurm/interactive.sh
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

1. Read-only: `ssh bioai 'sinfo -p gpu-1farm'` — confirm gpu-1farm exists +
   spec it. Safe (no compute).
2. CPU sbatch: 5-min `cpu-short` job that runs `uv sync` in a fresh clone —
   confirms our pyproject.toml resolves on bioai.
3. 1-GPU sbatch: 30-min `gpu-1farm` job that loads Qwen2.5-VL-7B-Instruct + a
   batch of 1 image and runs forward — confirms cuda + transformers + the
   model load path.

These are templates; they are NOT submitted by the agent.
