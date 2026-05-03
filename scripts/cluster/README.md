# Cluster preprocessing for Visual-CoT POC

End-to-end workflow:
1. **Cluster** (cpu-standard partition): download → extract → filter/sample → push to HF Hub.
2. **Local** (A6000): `load_dataset` from Hub → V_sem precompute → training.

Only ONE cluster job. Everything after the Hub push is local.

**The agent never submits jobs. Every `sb` requires the user's explicit "go".**

---

## Step 1 — submit preprocess + push job (cluster, ~4-8 hours)

```bash
ssh bioai
cd ~/projects/visual-latents
git pull
# verify HF_TOKEN in .env has WRITE access to your Hub namespace
grep HF_TOKEN .env
sb slurm/preprocess_viscot.sbatch
```

What it does (4 stages, all in one sbatch):
1. **Download** ~107 GB of Visual-CoT image shards from HuggingFace.
2. **Extract** with source filter (DocVQA + TextVQA + Flickr30k + OpenImages only) — ~18.5 GB on disk.
3. **Process** annotations: filter by bbox quality, sample 50K train + 1K eval (image-disjoint), build HF Datasets with images inlined.
4. **Push** to HF Hub as `ohjoonhee/visual-cot-50k-poc` (private by default; override with `VISCOT_HUB_REPO` env var).

Stage timings (estimated):
- Download: 1-2 h (network-bound)
- Extract: ~30 min
- Process: ~30 min (PIL.open passes)
- Push: 1-2 h (12 GB upload to Hub)

Slack notify on START / DONE / FAIL.

---

## Step 2 — local: pull from Hub + V_sem precompute (A6000, ~1.5-3 hours)

```bash
# on local
cd /mnt/ssd/Projects/visual-latents

# pulls the dataset to ~/.cache/huggingface (no rsync needed)
MACHINE=local uv run python scripts/precompute_vsem_local.py \
    --hub-repo ohjoonhee/visual-cot-50k-poc \
    --out-dir data/viscot
```

Outputs:
- `data/viscot/viscot_50k_train_vsem.parquet` (~210 MB)
- `data/viscot/viscot_1k_eval_vsem.parquet` (~5 MB)

The dataset images themselves are cached by HF in `~/.cache/huggingface/datasets/`
and stay there — no need to copy them around.

---

## Step 3 — local: training (A6000)

The interleaved trainer will load the HF dataset directly via `load_dataset(...)`
(images come back as PIL) and join the V_sem features by `(image_id, source, qa_idx_within_image)`.

(Implementation TODO once preprocessing is done — tracked as task #26 + #27 + #28.)

---

## Files

| File | Where it runs |
|---|---|
| `download_viscot.py` | cluster |
| `extract_viscot.sh` | cluster |
| `process_viscot.py` | cluster (writes HF Datasets locally on cluster disk) |
| `push_viscot_to_hub.py` | cluster (uploads to HF Hub) |
| `../precompute_vsem_local.py` | **local A6000** |

`slurm/preprocess_viscot.sbatch` orchestrates the cluster side.

## Failure recovery

| Stage failed | What to do |
|---|---|
| Download (stage 1) | Re-submit; `download_viscot.py --skip-existing` (default) skips already-present shards. |
| Extract (stage 2) | Manually re-run `bash scripts/cluster/extract_viscot.sh ...`. |
| Process (stage 3) | Manually re-run `uv run python scripts/cluster/process_viscot.py ...` (idempotent — overwrites HF Dataset dirs). |
| Push (stage 4) | Manually re-run `uv run python scripts/cluster/push_viscot_to_hub.py ...`. The push is idempotent (creates a new commit). |

## Disk budget

| Stage | Path | Size |
|---|---|---|
| Raw shards | `/data/joonhee/vl/data/viscot_raw/` | ~107 GB (delete after extract) |
| Extracted images | `/data/joonhee/vl/data/viscot_extracted/cot_image_data/` | ~18.5 GB |
| Local HF Datasets | `/data/joonhee/vl/data/viscot_processed/{train,eval}_hf/` | ~12 GB |
| **Cluster total (peak)** | | ~138 GB |
| **Cluster total (after raw cleanup)** | | ~31 GB |
| HF Hub repo | `ohjoonhee/visual-cot-50k-poc` | ~12 GB |
| Local HF cache (after `load_dataset`) | `~/.cache/huggingface/datasets/` | ~12 GB |
| Local V_sem parquets | `data/viscot/` | ~215 MB |

## Why this design (cluster preprocess + Hub + local rest)

- **Cluster handles the bandwidth-heavy step** (107 GB download + extract + 12 GB upload to Hub).
- **HF Hub is the canonical handoff** — reusable artifact, no rsync, easy to share.
- **A6000 V_sem precompute is fast enough** (~2 h on a single GPU vs ~30 min on 4× H100;
  saves a separate cluster submission and the user's `sb` approval cycle).
- **Single cluster sbatch** means only one user approval gate before training can start locally.
