"""Filter, sample, and write Visual-CoT POC data as a HuggingFace Dataset.

Builds an HF Dataset (images inlined as bytes via the Image feature type) and
saves to disk. A subsequent step (`push_viscot_to_hub.py`) uploads it to Hub.

Inputs:
    --annotation: path to viscot_363k.json (404K examples per probe).
    --image-dir: extracted dir up to but not including 'cot_image_data'.
    --out-dir: where the local HF Dataset goes (e.g., /data/joonhee/vl/data/viscot_processed).
    --target: 50000 (default) — number of training examples to sample.
    --eval-size: 1000 (default) — held-out eval examples (image-disjoint from train).
    --seed: 42.

Filtering:
    - Restrict to sources: docvqa, textvqa, flickr30k, openimages.
    - Drop bbox <16 px in either dim (too small to be useful target).
    - Drop bbox covering >70% of image area (too vague to ground).
    - Drop unreadable images.

Sampling (50K total, biased toward multi-Q sources for q-invariance):
    - Flickr30k: 15K (4.8 Q/img → ~3 Qs each from ~5K images).
    - DocVQA:    15K (3.4 Q/img → ~3 Qs each from ~5K images).
    - OpenImages: 10K (1.5 Q/img → mostly 1 Q each from ~7K images).
    - TextVQA:    10K (1.3 Q/img → mostly 1 Q each from ~8K images).

Per-image cap: 6 questions max so no single image dominates K_q sampling.

Held-out eval (1K, balanced 250 per source, image-disjoint, single-Q each).

Output: two HF Dataset directories (saved with save_to_disk):
    {out_dir}/viscot_50k_train_hf/
    {out_dir}/viscot_1k_eval_hf/

Schema (per row):
    image: PIL image (HF Image feature; serialized as bytes in parquet)
    image_id: str
    source: str ("docvqa" | "textvqa" | "flickr30k" | "openimages")
    question: str
    answer: str
    bbox_pixel: list[int]  (length 4: [x1, y1, x2, y2])
    bbox_norm:  list[float] (length 4: bbox normalized to [0,1] by image dims)
    width: int
    height: int
    qa_idx_within_image: int
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, Features, Image as ImageFeature, Sequence, Value
from PIL import Image


SOURCES_KEEP = {"docvqa", "textvqa", "flickr30k", "openimages"}

# Mix proportions for 50K training. Sums to 50K.
TARGET_PER_SOURCE_TRAIN = {
    "flickr30k": 15000,
    "docvqa": 15000,
    "openimages": 10000,
    "textvqa": 10000,
}
TARGET_PER_SOURCE_EVAL = {  # 1K total, balanced
    "flickr30k": 250,
    "docvqa": 250,
    "openimages": 250,
    "textvqa": 250,
}

FEATURES = Features({
    "image": ImageFeature(),
    "image_id": Value("string"),
    "source": Value("string"),
    "question": Value("string"),
    "answer": Value("string"),
    "bbox_pixel": Sequence(Value("int32"), length=4),
    "bbox_norm": Sequence(Value("float32"), length=4),
    "width": Value("int32"),
    "height": Value("int32"),
    "qa_idx_within_image": Value("int32"),
})


def parse_bbox_pixel(image_field: list[str]) -> list[int] | None:
    """Visual-CoT JSON: image[1] = '<path>###[x1, y1, x2, y2]'. Parse the suffix."""
    if len(image_field) < 2 or "###" not in image_field[1]:
        return None
    bbox_str = image_field[1].split("###", 1)[1].strip()
    try:
        bbox = json.loads(bbox_str)
        if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
            return [int(v) for v in bbox]
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def parse_qa(conversations: list[dict]) -> tuple[str, str] | None:
    """Visual-CoT conversations: 4-turn pattern. Q is in conv[0], A is in conv[3]."""
    if len(conversations) < 4:
        return None
    q_raw = conversations[0].get("value", "")
    a_raw = conversations[3].get("value", "")
    if not q_raw or not a_raw:
        return None
    q = q_raw.replace("<image>\n", "").strip()
    if "Please provide the bounding box" in q:
        q = q.split("Please provide the bounding box")[0].strip()
        q = q.rstrip(".? ").strip() + "?"
    return q, a_raw.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target", type=int, default=50000)
    ap.add_argument("--eval-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-image", type=int, default=6)
    ap.add_argument("--min-bbox-px", type=int, default=16)
    ap.add_argument("--max-bbox-area-frac", type=float, default=0.70)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    image_root = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[process] loading annotation from {args.annotation}...")
    with open(args.annotation) as f:
        records = json.load(f)
    print(f"[process] loaded {len(records)} records")

    # --- Filter pass (cheap; just JSON manipulation) ---
    by_source: dict[str, list[dict]] = defaultdict(list)
    skipped_no_bbox = 0
    skipped_no_qa = 0
    skipped_wrong_source = 0
    for r in records:
        src = r.get("dataset")
        if src not in SOURCES_KEEP:
            skipped_wrong_source += 1
            continue
        bbox = parse_bbox_pixel(r.get("image", []))
        if bbox is None:
            skipped_no_bbox += 1
            continue
        qa = parse_qa(r.get("conversations", []))
        if qa is None:
            skipped_no_qa += 1
            continue
        q, a = qa
        img_field = r["image"][0]
        if not img_field.startswith("cot/"):
            continue
        img_relpath = img_field[len("cot/"):]
        img_path = image_root / "cot_image_data" / img_relpath
        by_source[src].append({
            "src": src,
            "img_relpath": img_relpath,
            "img_path": str(img_path),
            "image_id": Path(img_relpath).stem,
            "q": q, "a": a, "bbox": bbox,
        })
    print(f"[process] after source filter: {sum(len(v) for v in by_source.values())} kept; "
          f"skipped {skipped_wrong_source} wrong-source, {skipped_no_bbox} no-bbox, {skipped_no_qa} no-qa")
    for src, recs in by_source.items():
        print(f"  {src}: {len(recs)}")

    # --- Multi-Q grouping + image-disjoint train/eval split ---
    sampled_train: dict[str, list[dict]] = {}
    sampled_eval: dict[str, list[dict]] = {}

    for src in SOURCES_KEEP:
        recs = by_source[src]
        by_img: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            by_img[r["image_id"]].append(r)
        img_keys = list(by_img.keys())
        rng.shuffle(img_keys)

        eval_target = TARGET_PER_SOURCE_EVAL[src]
        eval_keys = img_keys[:eval_target]
        train_keys = img_keys[eval_target:]

        # EVAL: 1 (q, a) per image.
        sampled_eval[src] = [rng.choice(by_img[k]) for k in eval_keys]

        # TRAIN: cap at max_per_image questions per image.
        train_target = TARGET_PER_SOURCE_TRAIN[src]
        train_pool = []
        for k in train_keys:
            qa_pool = by_img[k]
            if len(qa_pool) > args.max_per_image:
                qa_pool = rng.sample(qa_pool, args.max_per_image)
            for idx, r in enumerate(qa_pool):
                r2 = dict(r)
                r2["qa_idx_within_image"] = idx
                train_pool.append(r2)
            if len(train_pool) >= train_target * 2:  # oversample for validation
                break
        sampled_train[src] = train_pool

    # --- Validate (open with PIL, check bbox quality) ---
    def validate(pool: list[dict], target: int, label: str) -> list[dict]:
        out = []
        n_open_fail = n_bbox_small = n_bbox_large = 0
        for r in pool:
            try:
                with Image.open(r["img_path"]) as im:
                    w, h = im.size
            except Exception:
                n_open_fail += 1
                continue
            x1, y1, x2, y2 = r["bbox"]
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
            if x2 - x1 < args.min_bbox_px or y2 - y1 < args.min_bbox_px:
                n_bbox_small += 1
                continue
            if (x2 - x1) * (y2 - y1) / max(1, w * h) > args.max_bbox_area_frac:
                n_bbox_large += 1
                continue
            r2 = dict(r)
            r2.update(width=w, height=h, bbox_pixel=[x1, y1, x2, y2],
                      bbox_norm=[x1/w, y1/h, x2/w, y2/h])
            out.append(r2)
            if len(out) >= target:
                break
        print(f"[validate-{label}] open_fail={n_open_fail} small_bbox={n_bbox_small} "
              f"large_bbox={n_bbox_large}; kept {len(out)}")
        return out

    final_train, final_eval = [], []
    for src in SOURCES_KEEP:
        final_train.extend(validate(sampled_train[src], TARGET_PER_SOURCE_TRAIN[src], f"train/{src}"))
        final_eval.extend(validate(sampled_eval[src], TARGET_PER_SOURCE_EVAL[src], f"eval/{src}"))

    print(f"\n[summary] train: {len(final_train)} examples; eval: {len(final_eval)} examples")
    for src in SOURCES_KEEP:
        n_tr = sum(1 for r in final_train if r["src"] == src)
        n_ev = sum(1 for r in final_eval if r["src"] == src)
        print(f"  {src}: train={n_tr} eval={n_ev}")

    # --- Build HF Datasets and save_to_disk ---
    # The Image feature inlines the image bytes when we materialize the dataset.
    def build_dataset(rows: list[dict]) -> Dataset:
        # Pre-shuffle so the saved shards are mixed across sources.
        rng.shuffle(rows)
        cols = {
            "image": [r["img_path"] for r in rows],   # paths; ImageFeature loads bytes
            "image_id": [r["image_id"] for r in rows],
            "source": [r["src"] for r in rows],
            "question": [r["q"] for r in rows],
            "answer": [r["a"] for r in rows],
            "bbox_pixel": [r["bbox_pixel"] for r in rows],
            "bbox_norm": [r["bbox_norm"] for r in rows],
            "width": [r["width"] for r in rows],
            "height": [r["height"] for r in rows],
            "qa_idx_within_image": [r.get("qa_idx_within_image", 0) for r in rows],
        }
        return Dataset.from_dict(cols, features=FEATURES)

    train_ds = build_dataset(final_train)
    eval_ds = build_dataset(final_eval)

    train_path = out_dir / "viscot_50k_train_hf"
    eval_path = out_dir / "viscot_1k_eval_hf"

    print(f"\n[write] saving train HF Dataset to {train_path}")
    train_ds.save_to_disk(str(train_path))
    print(f"[write] saving eval HF Dataset  to {eval_path}")
    eval_ds.save_to_disk(str(eval_path))

    # Print sizes.
    def dirsize(p: Path) -> str:
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return f"{total / 1e9:.2f} GB"
    print(f"[write] train: {dirsize(train_path)}; eval: {dirsize(eval_path)}")
    print(f"[write] DONE.")


if __name__ == "__main__":
    sys.exit(main())
