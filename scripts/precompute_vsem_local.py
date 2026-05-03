"""Pre-compute V_sem features for Visual-CoT POC training data.

Runs LOCALLY on a single A6000 (no cluster). Loads the HF dataset that the
cluster pushed to Hub (or a local copy if you've already loaded it via
`load_dataset`).

For each (image, bbox) example:
  - V_sem(full_image) → mean-pooled [D] float16
  - V_sem(crop)       → mean-pooled [D] float16

Output:
  data/viscot/viscot_50k_train_vsem.parquet
  data/viscot/viscot_1k_eval_vsem.parquet

Each parquet has columns: image_id, source, qa_idx_within_image, vsem_full, vsem_crop.

Joining back to the dataset at training time: match on (image_id, source, qa_idx_within_image).

Estimated time on A6000: 1.5-3 hours for 51K examples × 2 forwards each.

Usage:
    MACHINE=local uv run python scripts/precompute_vsem_local.py \
        --hub-repo ohjoonhee/visual-cot-50k-poc \
        --out-dir data/viscot
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


@torch.no_grad()
def vsem_pool(model, processor, image: Image.Image) -> np.ndarray:
    """Compute pooled V_sem for one image. Returns [D] float16 numpy."""
    inputs = processor(text=["<|image_pad|>"], images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].cuda().to(model.dtype)
    image_grid_thw = inputs["image_grid_thw"].cuda()
    out = model.model.get_image_features(
        pixel_values=pixel_values, image_grid_thw=image_grid_thw
    )
    pool = out.pooler_output
    feat = pool if isinstance(pool, torch.Tensor) else pool[0]
    return feat.float().mean(dim=0).cpu().numpy().astype(np.float16)


def crop_image(img: Image.Image, bbox_pixel) -> Image.Image:
    x1, y1, x2, y2 = [int(v) for v in bbox_pixel]
    return img.crop((x1, y1, x2, y2))


def process_split(model, processor, ds, split_name: str, out_path: Path):
    n = len(ds)
    print(f"\n[{split_name}] processing {n} examples → {out_path}")
    image_ids = []
    sources = []
    qa_idxs = []
    vsem_full_arr = []
    vsem_crop_arr = []
    n_failed = 0
    t0 = time.monotonic()

    for i, ex in enumerate(ds):
        try:
            img = ex["image"].convert("RGB")
            full_feat = vsem_pool(model, processor, img)
            crop = crop_image(img, ex["bbox_pixel"])
            crop_feat = vsem_pool(model, processor, crop)
            image_ids.append(ex["image_id"])
            sources.append(ex["source"])
            qa_idxs.append(int(ex["qa_idx_within_image"]))
            vsem_full_arr.append(full_feat)
            vsem_crop_arr.append(crop_feat)
        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"  FAIL row {i}: {e}")

        if (i + 1) % 500 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta_min = (n - i - 1) / rate / 60
            print(f"  [{split_name}] {i+1}/{n} ({rate:.2f}/s, ETA {eta_min:.1f}min) failed={n_failed}")

    df = pd.DataFrame({
        "image_id": image_ids,
        "source": sources,
        "qa_idx_within_image": qa_idxs,
        "vsem_full": vsem_full_arr,
        "vsem_crop": vsem_crop_arr,
    })
    df.to_parquet(out_path, index=False)
    elapsed = time.monotonic() - t0
    print(f"[{split_name}] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) "
          f"in {elapsed/60:.1f} min; failed={n_failed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub-repo", default="ohjoonhee/visual-cot-50k-poc",
                    help="HuggingFace dataset repo to load")
    ap.add_argument("--out-dir", default="data/viscot")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] loading {args.hub_repo} from HF Hub (or local cache)...")
    train_ds = load_dataset(args.hub_repo, split="train")
    eval_ds  = load_dataset(args.hub_repo, split="eval")
    print(f"[load] train: {len(train_ds)} rows; eval: {len(eval_ds)} rows")

    print(f"[load] loading {MODEL} vision branch...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL)

    process_split(model, processor, train_ds, "train", out_dir / "viscot_50k_train_vsem.parquet")
    process_split(model, processor, eval_ds, "eval", out_dir / "viscot_1k_eval_vsem.parquet")

    print(f"\n[done] V_sem features written to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
