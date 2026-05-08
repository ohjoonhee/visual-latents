"""Pivot A: extract latents from a trained checkpoint on the held-out eval split.

Identical procedure to phase1_lvr/extract_phase1_latents.py — same prompt
template, same K=8, same eval split. Pivot A re-uses Phase 1's image-pad
slot mechanism (no Monet special tokens, no 4D mask).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "phase1_lvr"))
from data import load_phase1_dataset, example_to_tuple  # noqa: E402
from roi import post_merger_token_count, select_roi_indices  # noqa: E402


def _build_prompt_no_answer(question: str, K: int, n_image_tokens: int) -> str:
    img_pads = "<|image_pad|>" * n_image_tokens
    latent_pads = "<|image_pad|>" * K
    return (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{img_pads}<|vision_end|>{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{latent_pads}"
    )


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", type=str, default="eval")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max_pixels", type=int, default=401408)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"[extract] loading {args.ckpt}", flush=True)
    processor = AutoProcessor.from_pretrained(args.ckpt, use_fast=True, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = args.max_pixels
    tokenizer = processor.tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.ckpt, torch_dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval().to(device)
    image_token_id = model.config.image_token_id

    print(f"[extract] loading split={args.split} n={args.n}", flush=True)
    ds = load_phase1_dataset(split_name=args.split, n=args.n, seed=args.seed)

    n_done = 0
    t0 = time.time()
    for i in range(len(ds)):
        ex = ds[i]
        try:
            image, question, answer, bbox = example_to_tuple(ex)
        except Exception as e:
            print(f"  [skip {i}] {e}", flush=True)
            continue

        try:
            vis = processor.image_processor(images=image, return_tensors="pt")
            pixel_values = vis["pixel_values"].to(device=device, dtype=dtype)
            image_grid_thw = vis["image_grid_thw"].to(device=device)
            n_image_tokens = post_merger_token_count(image_grid_thw)
            roi_idx = select_roi_indices(image_grid_thw, bbox, K=args.K)

            v_full = torch.cat(model.model.get_image_features(pixel_values, image_grid_thw), dim=0)
            v_roi = v_full[torch.tensor(roi_idx, device=device, dtype=torch.long)]

            prompt = _build_prompt_no_answer(question, args.K, n_image_tokens)
            ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            pad_pos = (ids[0] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
            image_patch_pos = pad_pos[:n_image_tokens]
            slot_pos = pad_pos[n_image_tokens : n_image_tokens + args.K]

            text_embeds = model.get_input_embeddings()(ids)
            inputs_embeds = text_embeds.clone()
            inputs_embeds[0, image_patch_pos, :] = v_full.to(inputs_embeds.dtype)

            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            h = outputs.hidden_states[-1][0, slot_pos, :].detach().cpu()  # [K, D]

            torch.save({
                "latent": h,
                "v_roi": v_roi.detach().cpu(),
                "question": question,
                "answer": answer,
                "bbox": list(bbox),
                "sample_id": int(i),
                "K": args.K,
            }, out_dir / f"latent_{i:08d}.pt")
            n_done += 1
            if n_done % 25 == 0 or n_done == 1:
                print(f"  [done {n_done}/{args.n}] elapsed={time.time()-t0:.0f}s", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            print(f"  [OOM {i}] {e}", flush=True)
        except Exception as e:
            print(f"  [error {i}] {type(e).__name__}: {e}", flush=True)

    print(f"[extract] done. saved={n_done} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
