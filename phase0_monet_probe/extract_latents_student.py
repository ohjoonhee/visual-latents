"""Student-path latent extraction (supplement to Phase 0).

Mirrors `extract_latents.py` but builds the prompt WITHOUT auxiliary images.
The 8 latent slots are still present (`<abs_vis_token>...<abs_vis_token_pad>*8...</abs_vis_token>`)
but no `<|vision_start|><|image_pad|><|vision_end|>` precedes them, and no
auxiliary image is passed in `pixel_values`. The user's question image is
still present.

Mechanism: in `latent_mode=True`, latent positions are forwarded one at a
time using `latent_embed = batch_last_hidden_state[b, pos-1, :]` — pure
hidden-state recurrence (Coconut-style). With no aux image to splice in,
slot 0 is seeded from the trailing `<abs_vis_token>` token's hidden
state, and slots 1..7 each consume the previous slot's last_hidden_state.

Usage:
  python extract_latents_student.py --stage stage2 --subset Visual_CoT --n 100 --out latents_student/stage2/Visual_CoT
"""
from __future__ import annotations

# ===== monkey-patch BEFORE any transformers import =====
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

os.environ.setdefault("LATENT_START_ID", "151666")
os.environ.setdefault("LATENT_END_ID", "151667")
os.environ.setdefault("LATENT_SIZE", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_AUTO_DOCSTRING", "1")

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
import copy
import json
import random
import time
from typing import Optional, List

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(ROOT))
from monet_utils import (  # noqa: E402
    Monet_single_input_images_preprocess_function,
    find_ids_poss,
    process_multiple_question_img,
    resize_by_token_budget,
)


LATENT_SIZE = int(os.environ["LATENT_SIZE"])
DATA_ROOT = ROOT / "data" / "Monet-SFT-125K"
CKPT_ROOT = ROOT / "checkpoints" / "Monet-SFT-7B"


def _build_processor(model_path: Path) -> AutoProcessor:
    processor = AutoProcessor.from_pretrained(str(model_path), use_fast=True, trust_remote_code=True)
    for tok in [
        "<abs_vis_token_pad>",
        "<abs_vis_token>",
        "</abs_vis_token>",
        "<observation>",
        "</observation>",
    ]:
        processor.tokenizer.add_tokens(tok, special_tokens=True)
    return processor


def _extract_question(sample: dict) -> str:
    for step in sample["data"]:
        if step["role"] == "user":
            for c in step["content"]:
                if c.get("type") == "text":
                    return c["text"].lstrip("\n").strip()
    return ""


def _extract_answer(sample: dict) -> str:
    last_text = ""
    for step in sample["data"]:
        if step["role"] == "assistant":
            for c in step["content"]:
                if c.get("type") == "text":
                    last_text = c["text"]
    cleaned = (
        last_text
        .replace("<abs_vis_token></abs_vis_token>", "")
        .replace("<observation>", "")
        .replace("</observation>", "")
        .strip()
    )
    return cleaned


def _student_replace_latent_placeholder(text: str, latent_size: int,
                                         sep_token: str = "<|im_start|>assistant",
                                         latent_placeholder: str = "<abs_vis_token></abs_vis_token>",
                                         latent_pad_str: str = "<abs_vis_token_pad>") -> str:
    """Like `replace_latent_placeholder_with_img_pad` + `add_latent_pad_after_auxiliary_img`,
    but for the student path where there is no auxiliary image:
        <abs_vis_token></abs_vis_token>  ->  <abs_vis_token><pad>*latent_size</abs_vis_token>
    in the assistant turns only. The user-turn question image is preserved.
    """
    parts = text.split(sep_token)
    res_text = process_multiple_question_img(parts[0])  # question-image normalization
    pad_run = latent_pad_str * latent_size
    for assistant in parts[1:]:
        # Replace the bare placeholder with the latent slots block; no image_pad.
        assistant = assistant.replace(
            latent_placeholder,
            f"<abs_vis_token>{pad_run}</abs_vis_token>",
        )
        res_text += sep_token + assistant
    return res_text


def _student_preprocess(sample: dict, dataset_root: str, allow_no_observation: bool):
    """Like `Monet_single_input_images_preprocess_function`, but:
    - Strips auxiliary images from assistant turns (keeps the question image in user turn).
    - Does NOT enforce n_img == n_img_pad on the original (since we are removing aux images).
    Returns processed sample or None.
    """
    sample = copy.deepcopy(sample)
    conversations = sample["data"]
    n_img_pad = 0
    seen_observation = False
    for i, step in enumerate(conversations):
        new_step = step.copy()
        if step["role"] == "system":
            new_step["content"][0]["text"] = "You are a helpful assistant."
        # Remove aux-image content items from assistant turns.
        if step["role"] == "assistant":
            new_step["content"] = [c for c in new_step["content"] if c.get("type") != "image"]
        new_content = []
        for j, content in enumerate(new_step["content"]):
            content = dict(content)
            if content["type"] == "image":
                # Resolve image path (only happens for user turn now).
                img_file_name = content.pop("image")
                if "kling_mm" in dataset_root:
                    img_file_name = img_file_name.replace("created_dataset/filtered_data/", "")
                content["image"] = os.path.join(dataset_root, img_file_name)
            elif content["type"] == "text":
                if step["role"] == "assistant":
                    n_img_pad += content["text"].count("<abs_vis_token></abs_vis_token>")
                    if "<observation>" in content.get("text", ""):
                        seen_observation = True
                elif step["role"] == "user":
                    # Drop "Put your final answer within \boxed{}." for non-Zebra subsets.
                    if (
                        "Zebra_CoT_visual_search" not in dataset_root
                        and "Zebra_CoT_count" not in dataset_root
                    ):
                        # Apply same normalization as upstream.
                        content["text"] = content["text"].replace(
                            "\nPut your final answer within \\boxed{}.", ""
                        )
            new_content.append(content)
        new_step["content"] = new_content
        conversations[i] = new_step
    sample["data"] = conversations
    if n_img_pad == 0:
        return None
    if not seen_observation and not allow_no_observation:
        return None
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage2", "stage3"], required=True)
    ap.add_argument("--subset", choices=["Visual_CoT", "Zebra_CoT_count"], required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_pixels", type=int, default=1000 * 28 * 28)
    ap.add_argument("--per_img_pixels", type=int, default=500 * 28 * 28)
    ap.add_argument("--start_offset_from_end", type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = CKPT_ROOT / args.stage
    train_json = DATA_ROOT / args.subset / "train.json"
    if not train_json.exists():
        print(f"FATAL: missing {train_json}", flush=True)
        sys.exit(2)
    if not model_path.exists():
        print(f"FATAL: missing {model_path}", flush=True)
        sys.exit(2)

    print(f"[student] stage={args.stage} subset={args.subset} n={args.n}", flush=True)
    print(f"[student] model={model_path}", flush=True)

    processor = _build_processor(model_path)
    tok = processor.tokenizer

    latent_start_id = tok("<abs_vis_token>", return_tensors="pt")["input_ids"][0]
    latent_end_id = tok("</abs_vis_token>", return_tensors="pt")["input_ids"][0]
    latent_pad_id = tok("<abs_vis_token_pad>", return_tensors="pt")["input_ids"][0]
    answer_start_pattern = tok("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    LATENT_START_ID = int(latent_start_id.item())
    LATENT_END_ID = int(latent_end_id.item())
    LATENT_PAD_ID = int(latent_pad_id.item())

    print(f"[student] tokens: <abs_vis_token>={LATENT_START_ID} </abs_vis_token>={LATENT_END_ID} <abs_vis_token_pad>={LATENT_PAD_ID}", flush=True)

    # --- load model ---
    print("[student] loading model (bf16, cuda)", flush=True)
    config = Qwen2_5_VLConfig.from_pretrained(str(model_path))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=torch.bfloat16,
    )
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
    print(f"[student] {args.subset}/train.json has {len(raw)} rows", flush=True)

    candidates = raw[-args.start_offset_from_end:] if len(raw) > args.start_offset_from_end else raw
    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    accepted: list[tuple[int, dict]] = []
    skipped = 0
    for sample in candidates:
        original_idx = raw.index(sample) if sample in raw else -1
        if "metadata" not in sample:
            sample = dict(sample)
            sample["metadata"] = {
                "dataset_name": args.subset,
                "sample_id": original_idx if original_idx >= 0 else len(accepted),
            }
        try:
            processed = _student_preprocess(
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

    print(f"[student] accepted={len(accepted)}  skipped={skipped}", flush=True)
    if not accepted:
        print("FATAL: no held-out examples accepted", flush=True)
        sys.exit(3)

    n_done = 0
    n_failed = 0
    started = time.time()

    with torch.inference_mode():
        for sid, processed in accepted:
            try:
                example = processed["data"]
                texts = [processor.apply_chat_template(example, tokenize=False)]
                texts = [_student_replace_latent_placeholder(t, LATENT_SIZE) for t in texts]

                image_inputs, _ = process_vision_info([example])
                if image_inputs:
                    image_inputs, _ = resize_by_token_budget(
                        image_inputs,
                        global_max_pixels=args.max_pixels,
                        per_img_max_pixels=args.per_img_pixels,
                    )
                # Sanity: vision-pad count must match images (just the question image now).
                total_image_pads = sum(t.count("<|vision_start|><|image_pad|>") for t in texts)
                if total_image_pads != len(image_inputs):
                    print(f"  [skip sid={sid}] vision-pad/image mismatch ({total_image_pads} vs {len(image_inputs)})", flush=True)
                    n_failed += 1
                    continue

                if image_inputs:
                    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
                else:
                    batch = processor(text=texts, return_tensors="pt", padding=True)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                pixel_values = batch.get("pixel_values")
                image_grid_thw = batch.get("image_grid_thw")

                latent_pad_pattern = torch.tensor([LATENT_PAD_ID], dtype=torch.long)
                alignment_poss = find_ids_poss(input_ids, answer_start_pattern, latent_pad_pattern)
                if not alignment_poss[0]:
                    print(f"  [skip sid={sid}] no latent_pad positions found", flush=True)
                    n_failed += 1
                    continue

                inputs = dict(
                    latent_mode=True,
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    labels=None,
                    loss_type=[],
                    alignment_poss=alignment_poss,
                    output_latent_embeds=True,
                )
                if pixel_values is not None:
                    inputs["pixel_values"] = pixel_values.cuda()
                    inputs["image_grid_thw"] = image_grid_thw.cuda()

                out = model(**inputs, return_dict=True)
                latent = out.latent_embeds[0].detach().cpu()  # [num_latents, H]
                if latent.shape[0] == 0:
                    print(f"  [skip sid={sid}] zero latent positions", flush=True)
                    n_failed += 1
                    continue

                save_path = out_dir / f"latent_{sid:08d}.pt"
                torch.save(
                    {
                        "latent": latent,
                        "question": _extract_question(processed),
                        "answer": _extract_answer(processed),
                        "sample_id": sid,
                        "subset": args.subset,
                        "stage": args.stage,
                        "num_latents": int(latent.shape[0]),
                        "latent_size_per_image": LATENT_SIZE,
                        "path": "student",
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
    print(f"[student] done. saved={n_done}  failed={n_failed}  elapsed={dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
