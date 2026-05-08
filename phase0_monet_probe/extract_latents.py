"""Extract Monet latent embeddings on held-out Visual_CoT / Zebra_CoT_count examples.

Usage:
  python extract_latents.py --stage stage2 --subset Visual_CoT --n 100 --out latents/stage2/Visual_CoT
  python extract_latents.py --stage stage3 --subset Visual_CoT --n 100 --out latents/stage3/Visual_CoT
  python extract_latents.py --stage stage2 --subset Zebra_CoT_count --n 100 --out latents/stage2/Zebra_CoT_count
  python extract_latents.py --stage stage3 --subset Zebra_CoT_count --n 100 --out latents/stage3/Zebra_CoT_count

Per-example output: latents/<stage>/<subset>/latent_<i>.pt with keys
  - latent: Tensor [num_latents, H]   (last layer hidden states at <abs_vis_token_pad> positions)
  - question: str (user question text)
  - answer:   str (gold answer text, raw — extracted from final assistant turn)
  - sample_id: int
  - subset: str
  - stage: str
"""
from __future__ import annotations

# ===== monkey-patch BEFORE any transformers import =====
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Set Monet env vars before importing the model (these are read at import time
# in the modeling file).
os.environ.setdefault("LATENT_START_ID", "151666")
os.environ.setdefault("LATENT_END_ID", "151667")
os.environ.setdefault("LATENT_SIZE", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_AUTO_DOCSTRING", "1")

# Sanity: refuse to proceed if transformers was already imported
assert "transformers" not in sys.modules, (
    "FATAL: 'transformers' already imported before patch — extractor cannot run."
)

_patch_path = ROOT / "monet_model" / "modeling_qwen2_5_vl_monet.py"
_spec = importlib.util.spec_from_file_location(
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    str(_patch_path),
)
_patched_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_patched_mod)
sys.modules["transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"] = _patched_mod
# ======================================================

import argparse
import json
import random
import time
from typing import Optional

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

# Local helpers (vendored from upstream).
sys.path.insert(0, str(ROOT))
from monet_utils import (  # noqa: E402
    Monet_single_input_images_preprocess_function,
    add_latent_pad_after_auxiliary_img,
    find_ids_poss,
    replace_latent_placeholder_with_img_pad,
    resize_by_token_budget,
)


LATENT_SIZE = int(os.environ["LATENT_SIZE"])
DATA_ROOT = ROOT / "data" / "Monet-SFT-125K"
CKPT_ROOT = ROOT / "checkpoints" / "Monet-SFT-7B"


def _build_processor(model_path: Path) -> AutoProcessor:
    processor = AutoProcessor.from_pretrained(str(model_path), use_fast=True, trust_remote_code=True)
    # Add the Monet special tokens (idempotent: they are already in the saved tokenizer
    # for Monet-SFT-7B but `add_tokens` is a no-op when the token already exists).
    for tok in [
        "<abs_vis_token_pad>",
        "<abs_vis_token>",
        "</abs_vis_token>",
        "<observation>",
        "</observation>",
    ]:
        processor.tokenizer.add_tokens(tok, special_tokens=True)
    return processor


def _resolve_image_path(rel: str, subset: str) -> str:
    """Resolve the per-step image path from the dataset_root."""
    # Images live at data/Monet-SFT-125K/<subset>/<...> (after `images.zip` unpacking
    # has placed them under DATA_ROOT / subset). The trace contains paths like
    # 'Visual_CoT/...' or 'Zebra_CoT_count/...' RELATIVE to DATA_ROOT.
    if rel.startswith(subset + "/"):
        return str(DATA_ROOT / rel)
    if rel.startswith("Monet-SFT-125K/"):
        return str(DATA_ROOT.parent / rel)
    return str(DATA_ROOT / subset / rel)


def _extract_question(sample: dict) -> str:
    """Pull the user question text out of a Monet-format sample."""
    for step in sample["data"]:
        if step["role"] == "user":
            for c in step["content"]:
                if c.get("type") == "text":
                    return c["text"].lstrip("\n").strip()
    return ""


def _extract_answer(sample: dict) -> str:
    """Last assistant turn's last text content, with <abs_vis_token></abs_vis_token>
    and <observation>/</observation> tags scrubbed → just the natural-language answer.

    The Monet trace concludes with the actual answer in the FINAL text content
    of the assistant turn (trailing observation/non-observation prose). Earlier
    text contents are intermediate reasoning before each auxiliary image.
    """
    last_text = ""
    for step in sample["data"]:
        if step["role"] == "assistant":
            # Track the LAST text content in this assistant turn (or across turns).
            for c in step["content"]:
                if c.get("type") == "text":
                    last_text = c["text"]
    # Scrub structural tags.
    cleaned = (
        last_text
        .replace("<abs_vis_token></abs_vis_token>", "")
        .replace("<observation>", "")
        .replace("</observation>", "")
        .strip()
    )
    return cleaned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage2", "stage3"], required=True)
    ap.add_argument("--subset", choices=["Visual_CoT", "Zebra_CoT_count"], required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_pixels", type=int, default=1000 * 28 * 28,
                    help="Global pixel budget for resize_by_token_budget (matches upstream image_resize=global default 1000).")
    ap.add_argument("--per_img_pixels", type=int, default=500 * 28 * 28)
    ap.add_argument("--start_offset_from_end", type=int, default=2000,
                    help="Sample held-out examples from the LAST N rows of train.json to minimise overlap with the user's training data and with Monet's own early-epoch examples.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = CKPT_ROOT / args.stage
    train_json = DATA_ROOT / args.subset / "train.json"
    if not train_json.exists():
        print(f"FATAL: missing {train_json}; downloads may not be complete", flush=True)
        sys.exit(2)
    if not model_path.exists():
        print(f"FATAL: missing {model_path}", flush=True)
        sys.exit(2)

    print(f"[extract] stage={args.stage} subset={args.subset} n={args.n}", flush=True)
    print(f"[extract] model={model_path}", flush=True)

    # --- load processor ---
    processor = _build_processor(model_path)
    tok = processor.tokenizer

    latent_start_id = tok("<abs_vis_token>", return_tensors="pt")["input_ids"][0]
    latent_end_id = tok("</abs_vis_token>", return_tensors="pt")["input_ids"][0]
    latent_pad_id = tok("<abs_vis_token_pad>", return_tensors="pt")["input_ids"][0]
    obs_start_id = tok("<observation>", return_tensors="pt")["input_ids"][0]
    obs_end_id = tok("</observation>", return_tensors="pt")["input_ids"][0]
    answer_start_pattern = tok("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    LATENT_START_ID = int(latent_start_id.item())
    LATENT_END_ID = int(latent_end_id.item())
    LATENT_PAD_ID = int(latent_pad_id.item())

    # Validate against env-var expectations.
    expected_start = int(os.environ["LATENT_START_ID"])
    expected_end = int(os.environ["LATENT_END_ID"])
    if LATENT_START_ID != expected_start or LATENT_END_ID != expected_end:
        print(
            f"WARN: token-id mismatch  env=({expected_start},{expected_end}) "
            f"actual=({LATENT_START_ID},{LATENT_END_ID})",
            flush=True,
        )

    # --- load model ---
    print("[extract] loading config + model (bf16, cuda)", flush=True)
    config = Qwen2_5_VLConfig.from_pretrained(str(model_path))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=torch.bfloat16,
    )
    # Resize embeddings if the tokenizer grew (no-op if it didn't).
    new_vocab = len(tok)
    model.resize_token_embeddings(new_vocab)
    model.config.vocab_size = new_vocab
    model.config.latent_token_id = LATENT_PAD_ID
    model.config.latent_start_id = LATENT_START_ID
    model.config.latent_end_id = LATENT_END_ID
    model.config.answer_start_pattern = answer_start_pattern.tolist()
    model.eval()
    model.cuda()

    # --- load + filter dataset ---
    raw = json.load(train_json.open())
    print(f"[extract] {args.subset}/train.json has {len(raw)} rows", flush=True)

    # Held-out: take the LAST `start_offset_from_end` rows, then shuffle with
    # fixed seed. We require allow_no_observation=False for VisualCoT-style
    # (observation tags in the trace) but Zebra_CoT_count traces don't always
    # have `<observation>` so we allow either.
    # Per upstream task.py logic, samples without observations are dropped UNLESS
    # allow_no_observation=True. Zebra_CoT_count appears to use observation tags too;
    # we allow_no_observation=True to be permissive for the held-out probe.
    candidates = raw[-args.start_offset_from_end:] if len(raw) > args.start_offset_from_end else raw
    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    accepted: list[tuple[int, dict]] = []
    skipped = 0
    for idx, sample in enumerate(candidates):
        # Tag a stable id for round-tripping. The upstream code uses a pre-existing
        # 'metadata' field; if absent we synthesise from row index in the original list.
        original_idx = raw.index(sample) if sample in raw else -1
        if "metadata" not in sample:
            sample = dict(sample)
            sample["metadata"] = {
                "dataset_name": args.subset,
                "sample_id": original_idx if original_idx >= 0 else len(accepted),
            }
        # Preprocess (resolves image paths, validates trace structure).
        # NOTE: the upstream preprocess function uses `dataset_root` as the prefix
        # for `os.path.join(dataset_root, img_file_name)`. The Monet-SFT-125K
        # zip paths look like 'Visual_CoT/cot_images/...' so we pass DATA_ROOT
        # (i.e. .../Monet-SFT-125K) as dataset_root.
        try:
            processed = Monet_single_input_images_preprocess_function(
                sample, dataset_root=str(DATA_ROOT), allow_no_observation=True,
            )
        except Exception as e:
            skipped += 1
            continue
        if processed is None:
            skipped += 1
            continue
        accepted.append((sample["metadata"]["sample_id"], processed))
        if len(accepted) >= args.n:
            break

    print(f"[extract] accepted={len(accepted)}  skipped={skipped}", flush=True)
    if not accepted:
        print("FATAL: no held-out examples accepted by preprocess", flush=True)
        sys.exit(3)

    # --- iterate one example at a time (bs=1; matches upstream stable path) ---
    n_done = 0
    n_failed = 0
    started = time.time()

    with torch.inference_mode():
        for sid, processed in accepted:
            try:
                # Mirror upstream `collate_fn_precompute_teacher_latents`.
                example = processed["data"]  # list of turns
                texts = [processor.apply_chat_template(example, tokenize=False)]
                texts = [replace_latent_placeholder_with_img_pad(t) for t in texts]
                texts = add_latent_pad_after_auxiliary_img(texts, LATENT_SIZE, "<abs_vis_token_pad>")
                image_inputs, _ = process_vision_info([example])
                if image_inputs:
                    image_inputs, _ = resize_by_token_budget(
                        image_inputs,
                        global_max_pixels=args.max_pixels,
                        per_img_max_pixels=args.per_img_pixels,
                    )
                # Sanity: count of <|vision_start|><|image_pad|> must match num images.
                total_image_pads = sum(t.count("<|vision_start|><|image_pad|>") for t in texts)
                if total_image_pads != len(image_inputs):
                    print(f"  [skip sid={sid}] vision-pad/image mismatch ({total_image_pads} vs {len(image_inputs)})", flush=True)
                    n_failed += 1
                    continue

                batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                pixel_values = batch["pixel_values"]
                image_grid_thw = batch["image_grid_thw"]

                # Compute alignment positions = positions of <abs_vis_token_pad>
                # AFTER the first '<|im_start|>assistant' marker.  `find_ids_poss`
                # expects a 1-d tensor pattern (matching upstream main.py:293).
                latent_pad_pattern = torch.tensor([LATENT_PAD_ID], dtype=torch.long)
                alignment_poss = find_ids_poss(input_ids, answer_start_pattern, latent_pad_pattern)
                if not alignment_poss[0]:
                    print(f"  [skip sid={sid}] no latent_pad positions found", flush=True)
                    n_failed += 1
                    continue

                # Optional sanity: the count should equal num_assistant_images * LATENT_SIZE.
                # (Each auxiliary image gets LATENT_SIZE pad slots appended.)

                inputs = dict(
                    latent_mode=True,
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    pixel_values=pixel_values.cuda(),
                    image_grid_thw=image_grid_thw.cuda(),
                    labels=None,
                    loss_type=[],
                    alignment_poss=alignment_poss,
                    output_latent_embeds=True,
                )
                out = model(**inputs, return_dict=True)
                latent = out.latent_embeds[0].detach().cpu()  # [num_latents, H]
                if latent.shape[0] == 0:
                    print(f"  [skip sid={sid}] zero latent positions", flush=True)
                    n_failed += 1
                    continue

                # Save
                save_path = out_dir / f"latent_{sid:08d}.pt"
                torch.save(
                    {
                        "latent": latent,                          # [num_latents, H], bf16 cpu
                        "question": _extract_question(processed),
                        "answer": _extract_answer(processed),
                        "sample_id": sid,
                        "subset": args.subset,
                        "stage": args.stage,
                        "num_latents": int(latent.shape[0]),
                        "latent_size_per_image": LATENT_SIZE,
                    },
                    save_path,
                )
                n_done += 1
                if n_done % 5 == 0 or n_done == 1:
                    dt = time.time() - started
                    print(
                        f"  [done {n_done}/{len(accepted)}] sid={sid} latent={tuple(latent.shape)} "
                        f"elapsed={dt:.1f}s",
                        flush=True,
                    )
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                print(f"  [OOM sid={sid}] {e}", flush=True)
                n_failed += 1
                continue
            except Exception as e:
                import traceback
                print(f"  [error sid={sid}] {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                n_failed += 1
                continue

    dt = time.time() - started
    print(f"[extract] done. saved={n_done}  failed={n_failed}  elapsed={dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
