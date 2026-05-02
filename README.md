# visual-latents

Anchor-grounded visual latents for VLM reasoning under frozen consumer models.

A generator VLM emits a sequence of latents `h ∈ R^{K×D}` from an image; one or
more frozen *anchor* models (siblings of the generator) consume `h` spliced into
their vision-token positions and answer questions about the image. The training
objective forces `h` to encode visual content that any anchor can decode — not
just the specific decoder used during training.

The project starts from an extensive POC (rounds 1–3, ~7 GPU-hours) that
characterized the failure modes of naïve reader-NLL training and identified the
combined recipe that round-3 will validate. The full POC analysis is in
`docs/inherited/`.

## Status

**v0.1.0 — scaffold.** Model, losses, trainers are skeleton stubs with TODOs.
Round-3 POC sweep configurations are populated. Smoke test on local A6000 is the
next milestone.

## Quick start (local)

```bash
cp .env.example .env             # set MACHINE=local; add WANDB_API_KEY if using W&B
uv sync                          # creates .venv/, installs deps
uv run pytest tests/             # paths + config + curriculum tests should pass
# uv run python -m vl.train --config configs/smoke.yaml   # (after model.py is implemented)
```

## Quick start (bioai cluster)

See `docs/MULTIMACHINE.md` and `docs/SLURM.md`. Cluster jobs are NEVER
agent-submitted; every `sb`/`sbatch` requires explicit user approval. See
`~/.claude/docs/bioai_cluster_spec.md` (server-admin + project-owner mandatory
rules).

## Layout

```
src/vl/                  # importable package
  config.py              # dataclasses + YAML loader
  paths.py               # MACHINE-resolved storage roots
  model.py               # LIVR-style + LoRA + Stage-1 mask  (TODO)
  losses.py              # NLL_multi + L_concept + L_norm + curriculum  (TODO)
  readers.py             # frozen multi-anchor forward  (TODO)
  data/                  # GQA / CLEVR / TallyQA / multi-Q sampler  (TODO)
  probes.py              # steering + random-control utilities  (TODO)
  trainers/
    sft_anchor.py        # Variant A — trl.SFTTrainer subclass  (TODO)
    grpo_vlpo.py         # Variant B — trl.GRPOTrainer + VLPO  (deferred)
  train.py / eval.py     # CLI entry points  (TODO)

configs/                 # YAML overrides for sweep cells
  smoke.yaml             # local A6000 dry-run
  round3/                # 5-cell round-3 sweep
  M1/, M2/, variant_b/   # later milestones

slurm/                   # bioai sbatch templates (NEVER agent-submitted)

scripts/                 # download / sync helpers

tests/                   # infra unit tests only

docs/
  DESIGN.md / LOSS.md / PROTOCOLS.md / MULTIMACHINE.md / SLURM.md
  inherited/             # 9 POC docs (REPORT, JOURNAL, ROUND3_POC_DESIGN, ...)
```

## Method references

- `docs/inherited/ROUND3_POC_DESIGN.md` — round-3 implementation spec
- `docs/inherited/PROLIFERATED_PROJECT_PLAN.md` — full M1/M2/M3 plan
- `docs/inherited/AUX_LOSS_AND_ARCH_DESIGN.md` — combined loss derivation
- `docs/inherited/VARIANT_B_GRPO_DESIGN.md` — VLPO Gaussian extension for GRPO

## License

MIT — see `LICENSE`.
