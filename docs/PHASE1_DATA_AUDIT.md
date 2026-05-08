# Phase 1 Data Audit — `data/viscot/viscot_363k.json`

Fitness check for LVR-style reproduction (arXiv:2509.24251) — single-bbox-per-example training where the bbox selects post-projector image-token indices.

## Schema (one variant, 100% of rows)

- `dataset` (str) — `flickr30k|gqa|openimages|docvqa|textcap|v7w|textvqa|infographicsvqa|cub|vsr`
- `split` (str) — always `"train"` (no val/test inside the file)
- `question_id` (int)
- `image` (list[str], len=2) — `[rel_path, rel_path + "###[pixel_xyxy]"]`
- `conversations` (list[dict], len=4) — role pattern `(human, gpt, human, gpt)`: turn 0 = question + ROI request, turn 1 = normalized bbox, turn 2 = `<image>` placeholder, turn 3 = gold answer

Total: **404,120 rows** (filename "363k" is misleading; matches upstream `deepcs233/Visual-CoT` LLaVA-format JSON). Distinct image paths: **154,772**.

## Bbox

- Two redundant copies: normalized **xyxy** in `conversations[1].value`; pixel xyxy as a `###[...]` suffix on `image[1]`.
- Verbatim from row 0: `[0.562, 0.228, 0.646, 0.292]`.
- 100% parseable, 0 nulls. 871 rows clip slightly >1 (e.g. `1.002`) — still effectively normalized. Single bbox per example, matches LVR's canonical setup.

## Image path resolution

- Paths are relative `cot/<source>/...`. **No image files locally** — `data/viscot/` contains only the JSON and an empty `.cache/huggingface/download/` stub.
- Upstream: `deepcs233/Visual-CoT` on HF Hub, archived as `cot_images_tar_split/cot_images_00..12.tar` (13 shards, ~140 GB).
- `src/vl/data/viscot.py` does **not** read this JSON — it loads `ohjoonhee/visual-cot-50k-poc` from Hub (a preprocessed POC subset). The raw JSON is currently unused.

## Disk

- Annotations on disk: **329 MB**. Images required: **~140 GB** full (`docs/INTERLEAVED_DATASET_RECON.md` L94, L297). GQA-only subset (~88K rows): ~12 GB.

## 363K vs LVR's 438K

Same 10 source corpora — same upstream Visual-CoT release with a different filter, **not a different curation**. LVR's extra 34K is likely lighter deduping; format is compatible.

## Verdict

**Usable with download of ~140 GB images** (or ~12 GB GQA-only to start). Annotations are clean, single-bbox, normalized xyxy, no missing fields. Nothing about the JSON itself blocks LVR.

## Reference loader (5 lines)

```python
import json, re; from PIL import Image
ex = json.load(open("data/viscot/viscot_363k.json"))[0]
img = Image.open(f"data/viscot/{ex['image'][0]}")  # needs image tar fetch
question, answer = ex["conversations"][0]["value"], ex["conversations"][3]["value"]
bbox_norm = [float(x) for x in re.findall(r"-?\d+\.?\d*", ex["conversations"][1]["value"])]
```

---

**Recommendation:** Phase 1 should proceed with this JSON as the annotation source and fetch the `deepcs233/Visual-CoT` `cot_images_tar_split` tars (GQA-only ~12 GB first) — do not download a fresh source.
