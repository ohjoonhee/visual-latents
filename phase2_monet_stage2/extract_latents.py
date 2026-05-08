"""Phase 2 latent extraction.

Re-uses Phase 1.5's extract_latents.py logic — same prompt format, same
latent_mode=True extraction. Saves latent + v_roi for later ablation.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASE0 = ROOT.parent / "phase0_monet_probe"

os.environ.setdefault("LATENT_START_ID", "151666")
os.environ.setdefault("LATENT_END_ID", "151667")
os.environ.setdefault("LATENT_SIZE", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_AUTO_DOCSTRING", "1")

assert "transformers" not in sys.modules

_patch_path = PHASE0 / "monet_model" / "modeling_qwen2_5_vl_monet.py"
_spec = importlib.util.spec_from_file_location(
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    str(_patch_path),
)
_patched_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_patched_mod)
sys.modules["transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"] = _patched_mod

import argparse
import time

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(ROOT.parent / "phase1_lvr"))
from data import load_phase1_dataset, example_to_tuple  # noqa: E402
from roi import post_merger_token_count, select_roi_indices  # noqa: E402

SYSTEM_PROMPT = "You are a helpful assistant."


def _build_prompt(question, K, n_image_tokens):
    img_pads = "<|image_pad|>" * n_image_tokens
    latent_pads = "<abs_vis_token_pad>" * K
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{img_pads}<|vision_end|>{question}<|im_end|>\n"
        f"<|im_start|>assistant\n<abs_vis_token>{latent_pads}</abs_vis_token>"
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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"[extract] {args.ckpt}", flush=True)
    processor = AutoProcessor.from_pretrained(args.ckpt, use_fast=True, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = args.max_pixels
    tok = processor.tokenizer
    latent_pad_id = int(tok("<abs_vis_token_pad>", return_tensors="pt")["input_ids"][0, 0])

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.ckpt, torch_dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval().to(device)
    image_token_id = model.config.image_token_id

    ds = load_phase1_dataset(split_name=args.split, n=args.n, seed=args.seed)

    t0 = time.time()
    n_done = 0
    for i in range(len(ds)):
        ex = ds[i]
        try:
            image, question, answer, bbox = example_to_tuple(ex)
        except Exception:
            continue
        try:
            vis = processor.image_processor(images=image, return_tensors="pt")
            pixel_values = vis["pixel_values"].to(device=device, dtype=dtype)
            image_grid_thw = vis["image_grid_thw"].to(device=device)
            n_img = post_merger_token_count(image_grid_thw)
            roi_idx = select_roi_indices(image_grid_thw, bbox, K=args.K)
            v_full = torch.cat(model.model.get_image_features(pixel_values, image_grid_thw), dim=0)
            v_roi = v_full[torch.tensor(roi_idx, device=device, dtype=torch.long)]

            prompt = _build_prompt(question, args.K, n_img)
            ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            slot_pos = (ids[0] == latent_pad_id).nonzero(as_tuple=False).squeeze(-1)[: args.K].tolist()
            attn = torch.ones_like(ids)
            out = model(
                latent_mode=True,
                input_ids=ids,
                attention_mask=attn,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=None,
                alignment_poss=[slot_pos],
                loss_type=[],
                output_latent_embeds=True,
                return_dict=True,
            )
            h = out.latent_embeds[0].detach().cpu()

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
                print(f"  [{n_done}/{args.n}] elapsed={time.time()-t0:.0f}s", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [err {i}] {type(e).__name__}: {e}", flush=True)

    print(f"[extract] done. saved={n_done} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
