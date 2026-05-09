# Cluster Phase 3 plan — Monet baseline + Pivot A variants

**Goal.** Establish a paper-faithful Monet reproduction at full scale (7B + 125K Visual_CoT) as the canonical baseline, then run controlled variants (G recipe, D2 recipe, additive VICReg) for formal comparison. All work on the bioai cluster.

## Cluster constraints (from `~/.claude/docs/bioai_cluster_spec.md`)

- Only legal partition for joonhee: **`gpu-4farm` with `--gres=gpu:h100:4`** (always 4 GPUs).
- 24-hour walltime cap on `gpu-4farm`.
- 32–64 CPUs admin-recommended for 4-GPU job (not the 112 default).
- Mandatory: `--account=gpu`, `module load cuda/12.6`, cache-redirect block to `/data/joonhee/...`, uv venv (no conda).
- Submit via `sb script.sbatch` (the wrapper that captures `$ORIG_SBATCH_SCRIPT`).
- **Project-owner rule (§3-A):** I never submit. Each `sb` requires your explicit "go".

## Compute budget per stage

Rough estimates, 4×H100 80GB SXM, bf16, ZeRO-2, gradient checkpointing, eff_bsz=128 (per-GPU bsz=2 × 4 GPUs × grad_accum=16). Padded ~1.5× for I/O, eval, save.

| stage | data | epochs | optim steps | per-step | wall | fits 24h? |
|---|---|---:|---:|---:|---:|---|
| **Stage 1 SFT** (NTP warmup, full FT) | Monet-SFT-125K Visual_CoT | 1 | ~977 | ~25 s | **8–10 h** | ✅ |
| **Stage 2 baseline** (`L_NTP + 2.0·L_align-obs` + attn_mask_4d, full FT, latent slots) | Monet-SFT-125K Visual_CoT | 2 | ~1953 | ~30 s (extra teacher pass + alignment loss) | **18–22 h** | ✅ tight |
| **Variant** (one Pivot-A recipe, full FT, K=4 or K=8) | Monet-SFT-125K Visual_CoT | 1 | ~977 | ~22 s (no teacher pass) | **6–9 h** | ✅ comfortable |

Per-step times are educated guesses — first job will calibrate them and we re-cost subsequent jobs. If Stage 2 wall blows past 22h, we save mid-training and chain two sbatches via your explicit second submission.

## Job plan (4 sbatch jobs, sequential — each waits for your "go")

### Job 1 — Monet Stage 1 SFT  (`slurm/cluster_stage1_sft.sbatch`)
- 4×H100, 24 h cap, --cpus-per-task=48, --mem=480G.
- Train Qwen2.5-VL-7B-Instruct → NTP-only on Monet-SFT-125K Visual_CoT 1 epoch.
- Output: `/data/joonhee/visual-latents/cluster_phase3/stage1_sft/checkpoint`.
- Eval at end: 200-example held-out perplexity, sanity numbers.
- Estimated wall: 8–10 h. Plenty of slack.

### Job 2 — Monet Stage 2 baseline  (`slurm/cluster_stage2_baseline.sbatch`)
- Same allocation. Continues from Job 1's `checkpoint`.
- Loss: `L_NTP + 2.0·L_align-obs`, `attention_mask_4d` enabled, `emphasize_latent_weight=2.0`, `ce_emphasize_factor=4.0` — exactly per `phase2_monet_stage2/RECIPE.md` (paper column).
- 2 epochs on Monet-SFT-125K Visual_CoT.
- Output: `cluster_phase3/stage2_baseline/checkpoint`.
- Estimated wall: 18–22 h. Tight inside 24 h. Save every 500 steps and hourly diagnostics so we can resume cleanly if it doesn't complete.

### Job 3 — Variants (parallel-sequential within one allocation if possible)
A single 24-h sbatch chaining the variant trainings sequentially. Each starts from Job 1's `stage1_sft/checkpoint` (the controlled common base).

Proposed variants (priority order — pick a subset per your preference):

| code | recipe | base | est wall |
|---|---|---|---:|
| **V1 (G@7B)** | VICReg λ_reg=2.0, mean-MSE LVR, K=4, 2000 steps | stage1_sft | ~8 h |
| **V2 (D2@7B)** | VICReg λ_reg=2.0, mean-MSE LVR, K=8, 2000 steps | stage1_sft | ~8 h |
| **V3 (Monet+VICReg)** | Monet stage 2 recipe AS-IS + VICReg added (var_w=1, cov_w=0.04, γ=1, λ_reg=1) | stage1_sft | ~18 h |
| **V4 (LVR-only)** | Phase 1 baseline at 7B (LVR only, no reg, no Monet recipe) | stage1_sft | ~6 h |

V1+V2 in one 24-h sbatch (2 × 8 h ≈ 16 h, plus eval). V3 standalone (its own 24-h sbatch). V4 separate or paired with V1.

### Job 4 — Common eval pipeline  (`slurm/cluster_eval_all.sbatch`)
- 4×H100 4-h ish. Loads each ckpt in turn, extracts latents on held-out 200-example subset (matches the local eval set; same seed=0 shuffle), runs ablation, writes `cluster_phase3/<variant>/REPORT.md`. Builds the cross-variant comparison table.

Total: 4 sbatch jobs, ~40-50 GPU-hours total (4 GPUs × 24 h × ~half-utilization).

## Decision points I need your input on, before I write any script

### D1. Model size for "exact reproduction"
- The published Monet is **7B**.
- 4×H100 80 GB can train 7B comfortably.
- Confirm: full reproduction at 7B (default), or scale-down to 3B for faster first-pass?

### D2. Reproduction scope — Stage 3?
- Phase 0 already showed Stage 3 produces redundant collapse (mean cos ≈ 0.85). I propose **skip Stage 3** — it's a known-bad endpoint, training it again adds zero info.
- Confirm: Stage 1 + Stage 2 only, or include Stage 3?

### D3. Data
- Local: `phase0_monet_probe/data/Monet-SFT-125K/` (13 GB, already verified usable).
- Cluster: not yet on `/data/joonhee/`. Two options:
  - (a) `rsync` from local laptop to cluster `/data/joonhee/visual-latents/` (~13 GB over user network — depends on your home upload speed).
  - (b) Download fresh on cluster from HF Hub via a CPU-partition prep job (no GPU contention; goes to `cpu-standard`).
- I lean **(b)**: cluster has fast internet to HF, doesn't waste your home bandwidth, mirrors the local download path we already proved works.
- Confirm preference.

### D4. eff_bsz (training batch size)
- Paper: `8×16 = 128` (8 GPUs × 16 grad accum). We have 4 GPUs.
- Options:
  - (a) `4×32 = 128` — same eff_bsz (paper-faithful), more grad_accum, ~2× wall vs paper — fits in our budget.
  - (b) `4×16 = 64` — half eff_bsz (recipe deviation; LR may need scaling), faster wall.
- I lean **(a)** for paper fidelity. Confirm.

### D5. Variant priority — which to actually run?
The 4 proposed variants are stack-rankable:

1. V1 (G@7B): tests our local breakthrough at scale. **Highest expected info value.**
2. V3 (Monet+VICReg): tests if our reg HELPS the established Monet recipe. **High info value but expensive (~18 h).**
3. V2 (D2@7B): K=8 vs K=4 sensitivity at scale.
4. V4 (LVR-only): baseline for the "regularizer matters" claim.

Default: V1 + V2 first (one 24h job), then V3 standalone. Skip V4 since Phase 1 local already established the baseline.

Alternative: V1 + V3 (high-info pair), skip V2 to save GPU-hours.

Confirm priority.

### D6. Job submission cadence
- Jobs 1, 2, 3 are sequential (each needs the previous output).
- Job 4 is independent of training, just eval.
- I propose: **submit Job 1 first**, monitor it to completion, then submit Job 2, etc. — gives us a chance to spot-check each output before committing to the next.
- Alternative: submit Job 1 + queue Job 2 with `--dependency=afterok:<JOB1_ID>`. **NB: §3-A explicitly forbids `--dependency` chains.** So sequential, manual go is the only option.

### D7. Slack notify / WandB
- The cluster sbatch template (§3-H) has Slack-notify + WandB hooks. Should I include both?
- Confirm: which Slack webhook + W&B project to log to? Or skip both for now and rely on log file tailing?

## What I'll do once you answer D1–D7

1. Write the 4 sbatch scripts and the supporting Python (uv project on cluster).
2. Show them to you script-by-script with the exact `sb` command and resource cost.
3. Wait for your "go" on each before submitting.

## Files I will create (on `origin/main`, before you say "go" on anything)

```
slurm/
├── cluster_stage1_sft.sbatch          # Job 1
├── cluster_stage2_baseline.sbatch     # Job 2
├── cluster_variants_v1_v2.sbatch      # Job 3 (V1+V2 chained)
├── cluster_variant_v3.sbatch          # Job 3 (V3 standalone)
├── cluster_eval_all.sbatch            # Job 4
└── cluster_data_prep.sbatch           # Pre-job: download Monet-SFT-125K via cpu-standard

cluster/
├── pyproject.toml                     # uv venv pin (transformers 4.54.0, etc)
├── trainer_stage1.py                  # NTP-only trainer
├── trainer_stage2_monet.py            # Monet stage 2 recipe (paper-faithful)
├── trainer_pivot_a.py                 # Pivot A variants (G, D2, V3, V4)
├── eval_pipeline.py                   # Common eval/ablation
└── README.md                          # cluster-specific quickstart
```

Nothing gets committed to git until you confirm the plan structure. Nothing gets `sbatch`-submitted until you say "go" on that specific submission.
