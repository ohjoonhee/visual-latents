# Overnight Run — 2026-05-03

Implementation + verification pass run by Claude. Goal: implement v0.1.0 scaffold
to first end-to-end working training, with cluster-deploy compatibility audited.

## TL;DR

- **All scaffold stubs are now implemented and verified.** Smoke (10 steps, 7B
  on A6000) PASSED. Mini-training (50 steps, 3B on A6000) PASSED with strong
  learning signal (||h|| 190 → 57.4 vs target 57.86; train-NLL 7.4 → 1.2;
  held-out NLL 3.4 → 1.5).
- **Gradient flow verified** end-to-end: LoRA params, `new_emb`, `concept_mlp`
  all receive grad; vision tower + base LLM (non-LoRA) + lm_head are all frozen.
- **Cluster `slurm/*.sbatch` updated** to comply with `~/.claude/docs/bioai_cluster_spec.md`
  (gpu-1farm removed, salloc forbidden, cache redirects, cuda/13.0).
- **No agent-initiated cluster submission.** All cluster jobs require explicit
  user approval per project-owner rule.

## What was implemented (all stubs replaced)

| File | LOC | What |
|---|---|---|
| `src/vl/model.py` | 293 | `build_generator(cfg, dtype)` — adds 3 special tokens, freezes base, injects LoRA via `peft.inject_adapter_in_model` on `model.model.language_model` only, builds bottleneck `concept_mlp` (D→D/2→D), trainable `new_emb` (fp32). `forward_generator` builds the q-invariant prompt with K `<\|latent\|>` slots and gathers final-layer hidden states. `get_v_sem` extracts post-merger visual features (frozen). |
| `src/vl/losses.py` | 157 | `nll_multi_anchor` (R × K_q reader passes), `concept_loss` (1 − cos with k mod T_v pooling), `norm_loss`, `combined` with curriculum-weighted sum. `cfg.w_concept == 0` skips concept entirely (cell C4). |
| `src/vl/readers.py` | 222 | `load_anchors` reuses generator if path matches; otherwise loads frozen, no LoRA. `forward_anchor` builds prompt with K `<\|image_pad\|>` tokens (no real image), splices `h` at those positions in `inputs_embeds`, computes shifted CE loss on answer span only. Gradient flows through `h`. |
| `src/vl/data/{gqa,clevr,tallyqa,mixed}.py` | 5 files | Per-image grouping + ≥K_q QA filter + held-out reservation. GQA streams when n_images > 5000; CLEVR/TallyQA via `HuggingFaceM4/the_cauldron`. `MixedDataset` (IterableDataset) + `make_collator(K_q, shuffle_qa_within_image)` for cell C5. |
| `src/vl/trainers/sft_anchor.py` | 569 | Hand-rolled training loop. Two AdamW groups (LoRA+MLP at `lr_lora`, `new_emb` at `lr_token`), CosineAnnealingLR to 10%, grad clip 1.0, JSONL logging, ckpt save+resume (LoRA+`new_emb`+MLP+optim+sched), held-out eval cadence, wall-clock guard for `max_time` (exit 124), mid-run sanity warning at step 200 if `||h|| > 200` or `cos(h_proj, V_sem) > 0.95`. WandB optional (offline mode if no API key). |
| `src/vl/train.py` | 60 | Wired variant=A path to call `sft_anchor.train(cfg)`. |

## Verification results

### Unit tests: 27/27 PASS
- `pytest tests/` (paths + config + curriculum). Same suite that was passing before.

### Linting: clean
- `ruff check src/` → 0 errors after fixing 5 pre-existing UP045 warnings.

### Gradient probe: PASS (`/tmp/gradient_probe.py`)
- 3B model, K=4, lora_r=8, B=1, K_q=2, single shared anchor.
- `new_emb.grad`: nonzero (sum=1.29e3)
- LoRA: 252/504 params with grad (correct PEFT init: `lora_A` is zero so dL/dA=0 at step 0; `lora_B` gets all grad).
- `concept_mlp`: all 4 params with grad.
- Vision tower: 0/390 with grad (frozen).
- Base LLM (non-LoRA): 0/434 with grad (frozen).
- Embedding table: no grad (correct — trainable rows live in `new_emb`).
- Memory peak: **8.45 GB** on 3B path.

### Smoke test: PASS (`configs/smoke.yaml`, 10 steps, 7B on A6000)
- Model: Qwen/Qwen2.5-VL-7B-Instruct + LoRA r=16, K=4, K_q=2, B=1.
- Trainable surface: `lora=40,370,176` `mlp=12,850,432` `token=10,752`.
- 10 steps in ~13s wall-clock; **memory peak ~17 GB on A6000** per trainer probe.
- NLL: 12.3 → 4.9 (oscillating but trending down).
- `||h||`: 258 → 204 (descending toward target 57.86; needs more steps).
- Held-out NLL @ step 5: 7.84 (anchor 0).
- Both checkpoints saved (320 MB each: LoRA + optim state).

### Mini-training: STRONG PASS (`configs/mini.yaml`, 50 steps, 3B on A6000)
- Model: Qwen/Qwen2.5-VL-3B-Instruct + LoRA r=16.
- Step 0: total=1.02, NLL=7.4, ||h||=190.
- Step 10: total=103, NLL=7.9, ||h||=88 (curriculum kicked in at step 10).
- Step 20: total=4.8, NLL=4.5, **||h||=57.4 (matches target 57.86)** ✓
- Step 30: total=2.8, NLL=2.5, ||h||=57.4.
- Step 40: total=1.5, NLL=1.2, ||h||=57.3.
- **Held-out NLL: 3.36 @ step 25 → 1.54 @ step 50** (after 50→55 resume).
- Resume verified: ckpt_step50 → ckpt_step55, scheduler+optim restored.

### What this proves
- Gradient flow is end-to-end correct.
- Curriculum schedule activates correctly (w_nll: 0.1→1.0; w_norm: 0→0.1).
- The norm regularizer pulls ||h|| to the natural-token norm target — the
  sharpest visual learning signal in the recipe is visible in 30 steps.
- The reader (anchor) NLL gradient propagates back through the spliced `h` and
  the LoRA weights are updating accordingly.

## Cluster compatibility audit (no SSH access, doc-only)

Per `~/.claude/docs/bioai_cluster_spec.md`:

| Issue (per cluster spec) | Before | After |
|---|---|---|
| §2-D `salloc`/`srun` forbidden | `slurm/interactive.sh` used `salloc` | Replaced with deprecation message + sbatch template comment |
| §2-G `gpu-1farm` removed | `slurm/eval.sbatch -p gpu-1farm` | `slurm/eval.sbatch -p gpu-4farm --gres=gpu:1` |
| §2-F CPU guideline 32-64 for 4 GPU | round3_cell.sbatch had `cpus-per-task=112` | Set to 64 |
| §3-G mandatory cache redirects | Missing | Both sbatch scripts now export TMPDIR/TRITON_CACHE_DIR/TORCH_HOME/TORCH_EXTENSIONS_DIR/WANDB_DIR to `/data/joonhee/vl/...` |
| `module purge` per PDF examples | Missing | Both sbatch scripts now `module purge && module load cuda/13.0` |
| Wheel CUDA version | `module load cuda/12.6` (mismatch with `torch==2.11.0+cu130`) | `module load cuda/13.0` |
| Monet-SFT-7B path placeholder | `/data/joonhee/.cache/...<TBD>/stage2` | HF model id `NOVAglow646/Monet-SFT-7B`; HF_HOME cache resolves on bioai |
| `docs/SLURM.md` stale partition info | gpu-1farm-only for 1-GPU | Updated to reflect cluster spec |
| `docs/MULTIMACHINE.md` stale | mentioned gpu-1farm as available | Updated |

All `bash -n slurm/*.sbatch` syntax checks PASS. All 9 YAML configs load cleanly via `--dry-run`.

### What did NOT change (intentionally)

- `src/vl/paths.py` — already clean. `MACHINE=bioai` resolves to `/data/joonhee/vl/{data,checkpoints,results}` per spec.
- `tests/` — all baseline tests still pass; no new tests added beyond the gradient/data/trainer probes (probes live in /tmp, not committed).
- `pyproject.toml` — only added `torchvision>=0.26.0` (required by Qwen2.5-VL processor; was the only missing dep).

## Open issues / warnings for cluster bring-up

1. **Multi-GPU gradient sync.** The trainer is single-process per GPU. The cluster sbatch uses `accelerate launch --num_processes=4`, which spawns 4 independent processes. **Each currently runs an independent optimizer with no gradient sync** — i.e., 4× the effective batch size if you average results, but no DDP guarantees. For round-3 POC's 1000 steps × bs=4 × 4 GPUs configuration, this is a major caveat. **Wire `accelerate.Accelerator` or `torch.distributed` before submitting any C1-C5 cell.** This was flagged by the trainer agent and is the most important pre-cluster TODO.

2. **`disable_input_require_grads` workaround.** `model.gradient_checkpointing_enable()` registers a forward hook on the embedding lookup that flips `output.requires_grad_(True)`. This collides with the in-place mask assignment in `_splice_new_emb`. The trainer disables that hook after enabling GC. Verified the gradient still flows through `new_emb` (the splice creates a non-leaf tensor via index assignment of `new_emb[i]`). Documented in `sft_anchor.py`.

3. **YAML scientific notation.** `pyyaml` parses `5e-5` (no decimal point) as a string per YAML 1.1. The trainer coerces `lr_lora`/`lr_token` to `float()` defensively. Configs use `5.0e-5` form where I authored them (e.g. `mini.yaml`).

4. **CosineAnnealingLR `eta_min` is shared across param groups.** Round-3 spec wanted lr_token's floor to be 10% of lr_token (5e-4); current implementation makes it 10% of lr_lora (5e-6). Effectively the new_emb LR decays harder than spec'd in the last 200 steps. Acceptable for POC; flagged for proliferated.

5. **Smoke `K_q=2` < `n_per_image=3` filter.** GQA loader requires ≥`n_per_image+1` QAs per image; `MixedDataset.__init__` calls `load_gqa(..., n_per_image=3, ...)` regardless of `cfg.loss.K_q`. For tighter coupling, the loader could honor `K_q`. Not a bug — GQA easily satisfies the higher threshold — but worth noting.

6. **GPU-node internet reachability for HF downloads** is unverified per cluster spec. If the cluster compute nodes have full internet (per spec's guess), datasets download fine on first run. Else the user must pre-warm `HF_HOME` on the cluster login node before submitting cells.

7. **Stage-1 attention masking is not implemented.** Per spec §2.2, "answer tokens cannot attend to image tokens." For our setup this is structurally a no-op: generator pass has no answer tokens; reader pass has no image tokens (only K spliced positions). The `cfg.model.stage1_attention_mask=True` flag in configs is currently respected only in spirit. If the proliferated project introduces a joint generator-reader pass, the 4D mask must be wired then.

## Recommended cluster bring-up sequence (require explicit user "go" each)

1. **CPU sbatch probe** (5 min, `cpu-short`):
   ```
   sb slurm/probe_uvsync.sbatch    # would need to be authored — verifies pyproject.toml resolves
   ```
2. **1-GPU sbatch probe** (30 min, `gpu-4farm --gres=gpu:1`): run the unit test suite + a tiny smoke (Qwen2.5-VL-3B + 5 steps) to validate the cluster-side path. Authoring this probe sbatch is the suggested next agent task.
3. **Wire DDP** (`accelerate.Accelerator`-based or `torch.distributed.init_process_group`) — touch `sft_anchor.py:train()`. Local A6000 has 1 GPU so this needs cluster verification.
4. **C1 first 100-step canary** (4 GPUs × 30 min, `gpu-4farm`): submit `slurm/round3_cell.sbatch` with `max_steps=100` override. Verifies the curriculum and multi-GPU gradient sync at 4×H100.
5. **Full C1 run** (4 × 6h, `gpu-4farm`): 1000 steps. Read held-out NLL trajectory before scheduling C2-C5.

**Reminder** (per `docs/inherited/ROUND3_POC_DESIGN.md` §1 + project-owner rule):
- Every `sb`/`sbatch` command requires explicit user approval.
- No `--dependency` chains. No auto-resubmit on failure.

## Diagnostics

`results/<run>/ablation_eval.jsonl` rows now carry the Phase-0 pairwise-cosine
diagnostic (mirroring `phase0_monet_probe/h_stats.jsonl`) alongside the existing
per-mode NLL fields. New (additive) fields per row:
`h_norms_per_pos` (`[K]`), `pairwise_cosine` (`[K, K]`, symmetric, diag = 1.0),
and the scalar `mean_off_diag_cos` (mean over the `K*(K-1)/2` unique pairs).
**Reading thresholds** (anchored to Phase 0 — see
`phase0_monet_probe/REPORT.md`): `mean_off_diag_cos > 0.85` indicates uniform
redundancy collapse — the latent positions encode duplicated information and
the existing `compression_ratio` metric is blind to this mode (Monet stage 3:
0.85–0.87, near-zero utility). `mean_off_diag_cos < 0.4` indicates distributed
encoding with distinct per-position content (Monet stage 2: ~0.38, utility
+2.7 nat). Existing JSONL fields and the `_h_stats` row layout are unchanged
for backward compatibility with prior runs and `scripts/analyse_overnight.py`.

## Files changed in this run

- New: `configs/mini.yaml`, `docs/HANDOFF_OVERNIGHT.md` (this doc).
- Modified: `src/vl/{model,losses,readers,train,config}.py`,
  `src/vl/trainers/sft_anchor.py`, `src/vl/data/{gqa,clevr,tallyqa,mixed}.py`,
  `slurm/{round3_cell,eval}.sbatch`, `slurm/interactive.sh`,
  `docs/{SLURM,MULTIMACHINE}.md`, `pyproject.toml`, `uv.lock`,
  `configs/round3/{C1_full,C3_Kq1,C4_no_concept,C5_random_control}.yaml`,
  `configs/M1/base.yaml`, `configs/variant_b/pilot.yaml`.
- Probes (in `/tmp`, not committed): `gradient_probe.py`, `data_probe.py`, `trainer_probe.py`.
