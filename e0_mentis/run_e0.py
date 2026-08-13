#!/usr/bin/env python
"""E0 — oracle-format dissociation on MentisOculi's geometric tasks.

Runs a local VLM over MentisOculi puzzles under two arms that the benchmark
ships prompts for but never reported in `results/results_table.csv`:

  simple      baseline: the puzzle image only (this arm IS in the results table)
  visual_cot  TRUE oracle: the ground-truth intermediate images *for this very
              puzzle*, via the repo's own `visual_cot.txt` prompt

Note `icl_intermediate_images` (Rush Hour only) is few-shot ICL over *other*
puzzles' worked examples — not an oracle for the puzzle under test. The
`visual_cot` arm is the one that isolates imagery *utilization* from imagery
*generation*, and it is absent from the released results for every task.

Responses are written in the format MentisOculi's own `evaluate_responses.py`
expects, so scoring reuses their scorer rather than a reimplementation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

BENCH = Path("/mnt/ssd/Projects/mentis-oculi/datasets")
# <image foo.png> | <images visual cot> | <text foo.txt>
PLACEHOLDER = re.compile(r"<(images visual cot|image\s+[^>]+|text\s+[^>]+)>")


def resolve(ref: str, puzzle_dir: Path, level_dir: Path) -> Path:
    """Metadata paths are relative to the puzzle dir in some tasks and to the
    level dir in others (hinge-folding stores 'puzzle_0001/cot_00.png').

    form-board additionally records a puzzle-id prefix that is not on disk —
    metadata says '1_cot_00.png', the file is 'cot_00.png' — so try the
    prefix-stripped basename as a last resort (upstream metadata bug).
    """
    cands = [ref, re.sub(r"(^|/)\d+_", r"\1", ref)]
    for base in (puzzle_dir, level_dir):
        for c in cands:
            p = (base / c).resolve()
            if p.exists():
                return p
    raise FileNotFoundError(f"{ref!r} not found under {puzzle_dir} or {level_dir}")


def build_content(template: str, meta: dict, puzzle_dir: Path, level_dir: Path):
    """Turn a prompt template into an interleaved [text, image, ...] content list."""
    content, pos = [], 0

    def add_text(s: str) -> None:
        if s.strip():
            content.append({"type": "text", "text": s})

    for m in PLACEHOLDER.finditer(template):
        add_text(template[pos : m.start()])
        pos = m.end()
        tag = m.group(1)
        if tag == "images visual cot":
            for ref in meta.get("cot_images", []):
                content.append({"type": "image", "path": resolve(ref, puzzle_dir, level_dir)})
        elif tag.startswith("image"):
            ref = tag.split(None, 1)[1].strip()
            content.append({"type": "image", "path": resolve(ref, puzzle_dir, level_dir)})
        else:  # text file
            ref = tag.split(None, 1)[1].strip()
            add_text(resolve(ref, puzzle_dir, level_dir).read_text())
    add_text(template[pos:])
    return content


def parse_answer(raw: str) -> str | None:
    """Models wrap JSON in prose/fences; take the last {...} containing 'answer'."""
    for m in reversed(list(re.finditer(r"\{[^{}]*\"answer\"[^{}]*\}", raw, re.S))):
        try:
            return str(json.loads(m.group(0))["answer"])
        except Exception:
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["paper-fold", "hinge-folding", "form-board", "rushhour", "sliding-puzzle"])
    ap.add_argument("--arm", required=True, choices=["simple", "visual_cot"])
    ap.add_argument("--levels", default="1,2", help="comma-separated, e.g. 1,2,3")
    ap.add_argument("--n", type=int, default=20, help="puzzles per level")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", default="e0_mentis/responses")
    args = ap.parse_args()

    task_dir = BENCH / args.task
    template = (task_dir / "prompts" / f"{args.arm}.txt").read_text()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"loading {args.model} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    model_slug = args.model.split("/")[-1]
    for level in [int(x) for x in args.levels.split(",")]:
        level_dir = task_dir / "output" / f"level_{level:02d}"
        puzzles = sorted(level_dir.glob("puzzle_*"))[: args.n]
        if not puzzles:
            print(f"  level {level}: no puzzles, skipping")
            continue

        records, t0 = [], time.time()
        for i, pdir in enumerate(puzzles, 1):
            meta = json.loads((pdir / "metadata.json").read_text())
            if args.arm == "visual_cot" and not meta.get("cot_images"):
                print(f"  {pdir.name}: no cot_images, skipping")
                continue

            content = build_content(template, meta, pdir, level_dir)
            images = [Image.open(c["path"]).convert("RGB") for c in content if c["type"] == "image"]
            msg_content = [
                {"type": "image"} if c["type"] == "image" else {"type": "text", "text": c["text"]}
                for c in content
            ]
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": msg_content}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = processor(text=[prompt], images=images or None,
                               return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            raw = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            records.append({
                "puzzle_id": meta.get("puzzle_id", pdir.name),
                "puzzle_dir": str(pdir),
                "output_parsed": {"answer": parse_answer(raw)},
                # Required, despite the README calling it optional: paper-fold's
                # scorer reads ground truth from here with no puzzle_dir fallback.
                "metadata": meta,
                "raw": raw,
                "n_images": len(images),
            })
            if i % 5 == 0 or i == len(puzzles):
                print(f"  {args.task} L{level} {args.arm}: {i}/{len(puzzles)} "
                      f"({time.time()-t0:.0f}s)", flush=True)

        out_dir = Path(args.out) / model_slug / args.arm / args.task / f"level_{level:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "responses_0.json").write_text(json.dumps(
            {"responses": records, "dataset_path": str(level_dir)}, indent=2))
        n_parsed = sum(r["output_parsed"]["answer"] is not None for r in records)
        print(f"  -> {out_dir}/responses_0.json  ({n_parsed}/{len(records)} parsed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
