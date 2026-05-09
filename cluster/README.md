# Cluster Phase 3 — quickstart

This directory holds the cluster-side trainers + configs for Phase 3
(paper-faithful Monet at 7B/125K + Pivot A variants). All scripts assume:

- Cluster: bioai (`gpu-4farm` partition, 4×H100 80GB SXM, 24h walltime cap).
- Layout: clone the repo to `/home/joonhee/projects/visual-latents/`.
  Cluster outputs/data live under `/data/joonhee/visual-latents/`.
- **Common training data: `Monet-SFT-125K Visual_CoT`** (118K samples
  after eval-200 hold-out). Stage 1 SFT, Stage 2 baseline, V1, V2, V3
  ALL train on this same dataset — controlled-comparison invariant.
- **Common eval set: 200-example held-out** from the last 10K rows of
  `Monet-SFT-125K/Visual_CoT/train.json`, shuffled with seed=0, first
  200 accepted by `Monet_single_input_images_preprocess_function`. Same
  protocol as `phase0_monet_probe/extract_latents.py`. All five
  checkpoints are scored on this identical set.
- Local code under `phase0_monet_probe/` (vendored Monet model),
  `phase1_lvr/` (ROI helper, LVR loss), `pivot_a/` (regularizers) is
  imported by these trainers — do **not** delete those dirs on the cluster.

## One-time setup

```bash
# 1. Clone repo (skip if already done).
ssh bioai
cd ~/projects/
git clone <this-repo> visual-latents
cd visual-latents

# 2. uv sync inside the cluster sub-project.
cd cluster
uv sync   # uses cluster/pyproject.toml; pulls torch+cu126 wheels
cd ..

# 3. Stage data + Monet special tokens.
sb slurm/cluster_data_prep.sbatch
# wait for completion; check log under logs/vl-data-prep_<jobid>.log
```

## Training jobs (SUBMIT MANUALLY, ONE AT A TIME — project rule §3-A)

| job | sbatch | est. wall | depends on |
|---|---|---|---|
| Stage 1 SFT | `slurm/cluster_stage1_sft.sbatch` | 8-10 h | data prep |
| Stage 2 Monet baseline | `slurm/cluster_stage2_baseline.sbatch` | 18-22 h | Stage 1 |
| V1 + V2 chained | `slurm/cluster_variants_v1_v2.sbatch` | ~16 h | Stage 1 |
| V3 (Monet + VICReg) | `slurm/cluster_variant_v3.sbatch` | 18-22 h | Stage 1 |
| Eval all | `slurm/cluster_eval_all.sbatch` | 4-6 h | the four above |

```bash
# Submit (each requires explicit "go" from the user):
sb slurm/cluster_stage1_sft.sbatch
# ...wait, check log...
sb slurm/cluster_stage2_baseline.sbatch
# ...wait, check log...
sb slurm/cluster_variants_v1_v2.sbatch
sb slurm/cluster_variant_v3.sbatch
# ...all training done...
sb slurm/cluster_eval_all.sbatch
```

## Files

| file | role |
|---|---|
| `pyproject.toml` | uv project (transformers==4.54.0 pinned, deepspeed, accelerate). |
| `accelerate_zero2.yaml` | ZeRO-2 + CPU optimizer offload + bf16, 4 processes. |
| `data_utils.py` | Monet-SFT-125K loader + held-out 200 protocol; pivot_a HF loader. |
| `reg.py` | VICReg + cosine penalty (copy of `pivot_a/reg.py`). |
| `mask_utils.py` | 4D attention mask builder (copy of `phase1_5b_attn/mask_utils.py`). |
| `trainer_sft.py` | Stage 1 NTP-only SFT. |
| `trainer_monet_stage2.py` | Monet Stage 2 recipe (alignment loss + emphasize_latent_weight + attn_mask_4d). |
| `trainer_pivot.py` | Pivot A variants (V1 G@7B, V2 D2@7B, V4 LVR-only). Driven by config. |
| `trainer_v3.py` | V3 — Monet Stage 2 recipe + VICReg additive. |
| `eval.py` | Common eval pipeline (extract latents → ablation → REPORT.md). |
| `configs/*.yaml` | One YAML per training run; CLI flags can override `out_dir`/`max_steps`. |

## Caveats

1. **Monet vendored model.** `trainer_monet_stage2.py`, `trainer_v3.py`,
   `eval.py` monkey-patch `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl`
   from `phase0_monet_probe/monet_model/modeling_qwen2_5_vl_monet.py` BEFORE
   any `transformers` import. The patch is process-global — affects
   teacher and student forward equally. Don't import transformers from
   modules loaded at startup.
2. **Inline teacher.** Stage 2 / V3 run a fresh `Qwen2.5-VL-7B-Instruct`
   teacher forward inline per step (no offline precompute). This deviates
   from upstream's offline `precompute_teacher_reps.py` but saves ~480 GB
   of bf16 hidden-state cache and ~4 hours of upfront work. The trade-off
   is GPU memory: teacher + student in bf16 ≈ 28 GB on each H100, well
   under the 80 GB budget after ZeRO-2 partitioning.
3. **Slack notifications** are graceful — `SLACK_WEBHOOK_URL` is checked
   inside each sbatch and silently skipped if unset. WandB requires
   `WANDB_API_KEY` in `.env`.
4. **The `qwen_base` reader for eval is 7B**, not 3B (matches paper's
   reported baseline numbers; absolute NLL is comparable across all
   cluster ckpts since they're all 7B).
5. **Pivot A bbox fallback (V1, V2).** Pivot A's per-step LVR loss
   targets K image-patch features at a question-conditional ROI bounding
   box. Monet-SFT-125K does NOT carry coordinate boxes — its visual
   reasoning artifact is the auxiliary image, not a coordinate. To run
   Pivot A on the same dataset as Stage 2 / V3 (the controlled-comparison
   requirement), `trainer_pivot.py` falls back to a fixed center-crop
   bbox `(0.25, 0.25, 0.75, 0.75)` for every Monet-SFT-125K sample.
   This makes V1/V2's LVR target a fixed central crop of the question
   image rather than a question-conditional ROI. See the docstring of
   `trainer_pivot.py` for full rationale. The VICReg regularizer is
   unaffected (it targets slot hidden states, not the LVR target).
6. **Eval applies to all checkpoint types via `cluster/eval.py`.** The
   evaluator detects ckpt type from the saved tokenizer (Monet-recipe
   ckpts have `<abs_vis_token_pad>` etc.) and routes to either the
   `latent_mode=True` extraction path (Stage 2, V3) or the
   image-feature-splice path (Stage 1 SFT, V1, V2). Both paths consume
   the SAME 200-example Monet-SFT-125K Visual_CoT held-out slice.
