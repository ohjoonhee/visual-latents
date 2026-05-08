"""Phase 1 data loader for ohjoonhee/visual-cot-50k-poc.

Schema (verified by /tmp/probe_dataset.py):
- image: PIL.Image
- question: str
- answer: str
- bbox: list[float] (normalized xyxy)

Verify against `phase1_lvr/dataset_sample.json`.
"""
from __future__ import annotations

import os
from typing import Any

from datasets import load_dataset

DATASET_ID = "ohjoonhee/visual-cot-50k-poc"


def _bbox_keys(ex: dict) -> str | None:
    """Find the normalized bbox key (we tolerate several aliases)."""
    for k in ("bbox_norm", "bbox", "bbox_normalized", "bboxes"):
        if k in ex:
            return k
    return None


def load_phase1_dataset(
    split_name: str = "train",
    n: int | None = None,
    seed: int = 0,
    cache_dir: str | None = None,
):
    """Load and (optionally) subsample a split from visual-cot-50k-poc.

    Returns a `datasets.Dataset` (already shuffled with given seed if `n` given).
    """
    cache_dir = cache_dir or os.environ.get("HF_HOME", None)
    if cache_dir and not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = cache_dir
    ds = load_dataset(DATASET_ID, cache_dir=cache_dir)
    if split_name not in ds:
        raise KeyError(f"split {split_name!r} not in {list(ds.keys())}")
    sub = ds[split_name]
    if n is not None and n < len(sub):
        sub = sub.shuffle(seed=seed).select(range(n))
    return sub


def example_to_tuple(ex: dict[str, Any]) -> tuple:
    """Return (image, question, answer, bbox_xyxy_normalized).

    bbox is clamped to [0, 1] (0.7% of audit rows clip slightly >1).
    """
    bbox_key = _bbox_keys(ex)
    if bbox_key is None:
        raise KeyError(f"no bbox key in example; available: {list(ex.keys())}")
    bbox = list(ex[bbox_key])
    if len(bbox) != 4:
        raise ValueError(f"bbox must be xyxy len-4, got {bbox}")
    x0, y0, x1, y1 = (max(0.0, min(1.0, float(v))) for v in bbox)
    if x1 <= x0:
        x1 = min(1.0, x0 + 1e-3)
    if y1 <= y0:
        y1 = min(1.0, y0 + 1e-3)
    return (
        ex["image"].convert("RGB") if ex["image"].mode != "RGB" else ex["image"],
        str(ex["question"]),
        str(ex["answer"]),
        (x0, y0, x1, y1),
    )
