"""Data loaders for cluster Phase 3 trainers.

Two distinct loaders:

1. `load_monet_sft_125k` — full Monet recipe path. Reads the raw
   `Monet-SFT-125K/<subset>/train.json` from the cluster path
   (`/data/joonhee/visual-latents/data/Monet-SFT-125K/`), runs the upstream
   `Monet_single_input_images_preprocess_function` to produce per-sample
   processed dicts (image paths resolved + trace structure validated), and
   reserves a 200-example held-out subset using the same protocol as the
   local Phase 0 / Phase 2 setup (last `start_offset_from_end` rows of
   `train.json`, shuffled with `seed=0`, first `n` accepted by preprocess).

   The `<observation>` token preservation logic from
   `phase2_monet_stage2/trainer.py` is in `MONET_SPECIAL_TOKENS`. Trainers
   call `add_monet_special_tokens(processor)` to register them and resize
   the model embedding table.

2. `load_pivot_a_dataset` — Pivot A path. Reads
   `ohjoonhee/visual-cot-50k-poc` (already pushed to HF Hub by
   `slurm/preprocess_viscot.sbatch`), returning a `datasets.Dataset` with
   keys `image, question, answer, bbox_norm`. Same shuffle seed/eval-slice
   convention as `phase1_lvr/data.py`.

Held-out eval slice (used by `eval.py`):
- For Monet path: last 10000 rows of `Visual_CoT/train.json`, seed=0
  shuffle, first 200 accepted by `Monet_single_input_images_preprocess_function`
  with `allow_no_observation=True` (matches `phase0_monet_probe/extract_latents.py`).
- For Pivot A path: first 200 of the `eval` split after seed=0 shuffle
  (matches `pivot_a/extract_latents_*.py`).
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional


MONET_SPECIAL_TOKENS = [
    "<abs_vis_token_pad>",
    "<abs_vis_token>",
    "</abs_vis_token>",
    "<observation>",
    "</observation>",
]


# ---------------------------------------------------------------------------
# cluster paths
# ---------------------------------------------------------------------------

CLUSTER_DATA_ROOT = Path(os.environ.get(
    "VL_CLUSTER_DATA_ROOT",
    "/data/joonhee/visual-latents/data",
))
MONET_SFT_125K_ROOT = CLUSTER_DATA_ROOT / "Monet-SFT-125K"


# ---------------------------------------------------------------------------
# Monet loader
# ---------------------------------------------------------------------------

def add_monet_special_tokens(processor) -> int:
    """Register Monet special tokens on the processor's tokenizer.

    Idempotent. Returns the new vocab length so the trainer can call
    `model.resize_token_embeddings(new_vocab)` and update
    `model.config.vocab_size`.
    """
    tok = processor.tokenizer
    for tk in MONET_SPECIAL_TOKENS:
        tok.add_tokens(tk, special_tokens=True)
    return len(tok)


def _ensure_phase0_on_path():
    """Ensure phase0_monet_probe (the vendored Monet helpers) is importable.

    The cluster repo lives at /home/joonhee/projects/visual-latents/, the
    same layout as the local repo, so phase0_monet_probe is `../`.
    """
    here = Path(__file__).resolve().parent
    phase0 = here.parent / "phase0_monet_probe"
    if str(phase0) not in sys.path:
        sys.path.insert(0, str(phase0))


def load_monet_sft_125k(
    subset: str = "Visual_CoT",
    *,
    split: str = "train",
    n: Optional[int] = None,
    seed: int = 0,
    eval_holdout: int = 200,
    eval_offset_from_end: int = 10_000,
    allow_no_observation_for_train: bool = False,
    data_root: Optional[Path] = None,
) -> tuple[list[dict], list[dict]]:
    """Load Monet-SFT-125K and return `(train_processed, eval_processed)`.

    Each element is a `processed` dict (output of
    `Monet_single_input_images_preprocess_function`) ready to be consumed by
    the cluster trainers (mirrors `phase2_monet_stage2/trainer.py`).

    Held-out protocol (matches phase0_monet_probe/extract_latents.py):
      candidates = train_json[-eval_offset_from_end:]
      shuffled with random.Random(seed)
      first `eval_holdout` accepted by preprocess(allow_no_observation=True)

    Train protocol (matches Monet upstream — first `n` rows in dataset
    order, with the eval-held-out IDs excluded so there's zero overlap):
      filter out the eval `(dataset_name, sample_id)` set, then take first
      `n` accepted samples in the natural order of train.json.

    Args:
      subset: 'Visual_CoT' (current default) or 'Zebra_CoT_count'.
      split: only 'train' supported (Monet-SFT-125K has no separate eval).
      n: max train samples. None = all (after eval-exclusion).
      seed: shuffle seed for eval slice + train shuffle (per-epoch shuffles
        live in the trainer; this seed only governs the eval-slice
        selection).
      eval_holdout: number of held-out samples (default 200, matches local).
      eval_offset_from_end: take eval candidates from the LAST N rows
        (default 10000, matches phase0_monet_probe).
      allow_no_observation_for_train: pass-through to preprocess. False
        (default) matches Monet stage 2 recipe — only samples WITH
        `<observation>` tags are kept for training.
      data_root: override Monet-SFT-125K root (default = cluster path).

    Returns:
      (train_processed, eval_processed)
    """
    if split != "train":
        raise ValueError(f"Monet-SFT-125K only has 'train' split; got {split!r}")

    _ensure_phase0_on_path()
    from monet_utils import Monet_single_input_images_preprocess_function  # noqa: E402

    root = Path(data_root) if data_root is not None else MONET_SFT_125K_ROOT
    train_json = root / subset / "train.json"
    if not train_json.exists():
        raise FileNotFoundError(
            f"Monet-SFT-125K subset {subset} not found at {train_json}; "
            f"run `slurm/cluster_data_prep.sbatch` first."
        )

    raw = json.load(train_json.open())
    print(f"[data] {subset}/train.json: {len(raw)} rows", flush=True)

    # ---- eval slice (last N rows shuffled, first `eval_holdout` accepted) ----
    eval_candidates = raw[-eval_offset_from_end:] if len(raw) > eval_offset_from_end else list(raw)
    rng = random.Random(seed)
    rng.shuffle(eval_candidates)

    eval_processed: list[dict] = []
    eval_ids: set[tuple[str, int]] = set()
    for sample in eval_candidates:
        if "metadata" not in sample:
            sample = dict(sample)
            sample["metadata"] = {
                "dataset_name": subset,
                "sample_id": len(eval_processed),
            }
        try:
            proc = Monet_single_input_images_preprocess_function(
                copy.deepcopy(sample),
                dataset_root=str(root),
                allow_no_observation=True,
            )
        except Exception:
            continue
        if proc is None:
            continue
        ds_name = proc["metadata"]["dataset_name"]
        sid = int(proc["metadata"]["sample_id"])
        eval_ids.add((ds_name, sid))
        eval_processed.append(proc)
        if len(eval_processed) >= eval_holdout:
            break
    print(f"[data] eval_holdout: {len(eval_processed)}", flush=True)

    # ---- train slice (raw order, eval IDs excluded) ----
    train_processed: list[dict] = []
    for sid_index, sample in enumerate(raw):
        if "metadata" not in sample:
            sample = dict(sample)
            sample["metadata"] = {"dataset_name": subset, "sample_id": sid_index}
        ds_name = sample["metadata"]["dataset_name"]
        sid = int(sample["metadata"]["sample_id"])
        if (ds_name, sid) in eval_ids:
            continue
        try:
            proc = Monet_single_input_images_preprocess_function(
                copy.deepcopy(sample),
                dataset_root=str(root),
                allow_no_observation=allow_no_observation_for_train,
            )
        except Exception:
            continue
        if proc is None:
            continue
        train_processed.append(proc)
        if n is not None and len(train_processed) >= n:
            break
    print(f"[data] train_processed: {len(train_processed)}", flush=True)

    return train_processed, eval_processed


# ---------------------------------------------------------------------------
# Pivot A loader (visual-cot-50k-poc)
# ---------------------------------------------------------------------------

def load_pivot_a_dataset(
    split_name: str = "train",
    n: Optional[int] = None,
    seed: int = 0,
    cache_dir: Optional[str] = None,
    dataset_id: str = "ohjoonhee/visual-cot-50k-poc",
):
    """Load pushed Visual-CoT POC dataset (50K) for Pivot A trainers.

    Mirrors `phase1_lvr/data.py:load_phase1_dataset`.
    """
    from datasets import load_dataset

    cache_dir = cache_dir or os.environ.get("HF_HOME", None)
    ds = load_dataset(dataset_id, cache_dir=cache_dir)
    if split_name not in ds:
        raise KeyError(f"split {split_name!r} not in {list(ds.keys())}")
    sub = ds[split_name]
    if n is not None and n < len(sub):
        sub = sub.shuffle(seed=seed).select(range(n))
    return sub


def pivot_a_example_to_tuple(ex: dict):
    """Mirror `phase1_lvr/data.py:example_to_tuple`."""
    bbox_key = None
    for k in ("bbox_norm", "bbox", "bbox_normalized", "bboxes"):
        if k in ex:
            bbox_key = k
            break
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
    img = ex["image"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img, str(ex["question"]), str(ex["answer"]), (x0, y0, x1, y1)
