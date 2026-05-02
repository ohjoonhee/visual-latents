# Changelog

## v0.1.0 — 2026-05-02

Initial scaffold. Variant A (anchor-grounded SFT) skeleton; Variant B (VLPO GRPO) stubbed.

- `src/vl/` package with model, losses, anchors, data, trainers/sft_anchor (stub), trainers/grpo_vlpo (stub)
- `configs/round3/{C1..C5}.yaml` — 5-cell sweep per `docs/inherited/ROUND3_POC_DESIGN.md`
- `configs/smoke.yaml` — local A6000 dry-run
- `slurm/round3_cell.sbatch` — bioai sbatch template (NOT submitted)
- `paths.py` — MACHINE-resolved storage roots
- `tests/` — paths + config + curriculum unit tests
- `docs/inherited/` — 9 POC docs copied verbatim
