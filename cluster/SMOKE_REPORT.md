# Cluster code local smoke report

Goal: catch every code bug we can on local A6000 before submitting any cluster GPU job.

## Results

| # | smoke | status | wall | peak GB | fixes applied |
|---|---|---|---:|---:|---|
| 1 | `uv --project cluster sync` | PASS | ~5 min | 0 | added explicit `nvidia-*-cu12` pins to `pyproject.toml` (pytorch.org cu126 wheel doesn't bundle them) + `setuptools>=68` (deepspeed import needs it) + `triton==3.2.0` |
| 2 | `trainer_sft.py` (3B, 10 steps) | PASS | ~5 s | 33 | none |
| 3 | `trainer_pivot.py` (3B, 10 steps) | PASS | ~5 s | 33 | none |
| 4 | `trainer_monet_stage2.py` (3B, 10 steps) | PASS | 15 s | 40.6 | fixed `attn_mask_4d = dict["full_attention"]` → pass full dict (Monet's `modeling_qwen2_5_vl_monet.py:1762` indexes the dict, not the tensor) |
| 5 | `trainer_v3.py` (3B, 10 steps) | **FAIL** | n/a | n/a | CUDA index out of bounds — likely token-id resize bug specific to V3. **Not on Job A path; deferred to Job C prep.** |
| 6 | `eval.py` (3 examples on local Pivot ckpt) | PASS | 19 s | ~30 | (a) CLI flags `--out_dir`/`--n` were `--out_root`/`--n_eval` — invocation fixed in submit guides; (b) `.cuda(dtype=...)` → `.to("cuda", dtype=...)` (PyTorch deprecated keyword); (c) added `alignment_poss=[[]]` to forward call to bypass Monet-patched modeling's `if alignment_poss[0]:` NoneType subscript when no alignment supervision is supplied (Pivot ckpts) |

## Code edits made

| file | lines | summary |
|---|---|---|
| `cluster/pyproject.toml` | dependencies block | Added 13 `nvidia-*-cu12` pins, `triton==3.2.0`, `setuptools>=68` for deepspeed compatibility |
| `cluster/trainer_monet_stage2.py` | ~349-359 | `attn_mask_4d` is now passed as dict (matches Monet model's expectation) |
| `cluster/trainer_v3.py` | ~330-337 | same dict fix |
| `cluster/eval.py` | 205, 304 | `.cuda(dtype=...)` → `.to("cuda", dtype=...)` |
| `cluster/eval.py` | ~326-336 | added `alignment_poss=[[]]` to model forward |

## Concerns about cluster scale-up

1. **Memory at 7B**: trainer_monet_stage2 + V3 hold both student and teacher models. At 3B local single-GPU we saw 40.6 GB; at 7B the per-GPU budget on H100 80 GB should still fit because of ZeRO-2 sharding the optimizer states across 4 GPUs. But it's not guaranteed — the smoke didn't validate the cluster `accelerate_zero2.yaml` config in a multi-GPU setting.
2. **`accelerator.unwrap_model()` per example** in `trainer_monet_stage2.py:387` triggers deepspeed import on every example. With CUDA_HOME exported correctly (added to sbatch) this works, but it's wasteful. Could move the unwrap outside the loop.
3. **V3 still broken**. The "Monet stage 2 + VICReg additive" recipe is Job C, not Job A. Need to fix before that submission.
4. **Eval data path**: eval.py uses `VL_CLUSTER_DATA_ROOT` env var; sbatch needs to set it (or rely on default `/data/joonhee/visual-latents/data`). Default is correct on cluster.

## Verdict

**Job A (Stage 1 + V1 + eval V1) is GREEN to submit.** All three components passed end-to-end smoke at 3B/single-GPU. Outstanding 7B-specific risks (memory headroom, multi-GPU sync) cannot be tested locally and remain residual.

Job B (Stage 2 baseline) trainer also passed smoke. GREEN-ish; same residual risks plus the per-example unwrap-model cost.

Job C (V3 = Monet+VICReg) is RED — V3 trainer has a token-id-out-of-bounds bug we did not fix.

Job D (V2 + V4) is GREEN — same trainer as V1, just different config.
