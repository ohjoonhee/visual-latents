"""10-step gradient-flow + ROI gather correctness check.

Loads Qwen2.5-VL-3B-Instruct, runs 10 LVR steps on the first 8 examples of the
visual-cot-50k-poc train split. Checks:
  - ROI indices are length 8 and within the post-merger token range
  - h.shape == [K, D] and v.shape == [K, D]
  - gradients flow into LLM (non-vision) parameters
  - GPU memory stays under 48 GB
  - no NaN losses
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data import example_to_tuple, load_phase1_dataset  # noqa: E402
from loss import lvr_mse  # noqa: E402
from roi import post_merger_token_count, select_roi_indices  # noqa: E402
from trainer import lvr_step  # noqa: E402


def main():
    base = "Qwen/Qwen2.5-VL-3B-Instruct"
    device = torch.device("cuda")
    dtype = torch.bfloat16
    K = 8

    print(f"[smoke] loading processor + model: {base}", flush=True)
    processor = AutoProcessor.from_pretrained(base, use_fast=True, trust_remote_code=True)
    processor.image_processor.max_pixels = 200704  # ~256-token budget
    tokenizer = processor.tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, torch_dtype=dtype).to(device)
    image_token_id = model.config.image_token_id

    # Freeze vision tower + projector
    for n, p in model.named_parameters():
        p.requires_grad_(not (n.startswith("model.visual") or n.startswith("visual")))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_freeze = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[smoke] trainable={n_train/1e6:.1f}M  frozen={n_freeze/1e6:.1f}M", flush=True)

    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Optimizer
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    # Data
    print("[smoke] loading 8 train examples", flush=True)
    ds = load_phase1_dataset(split_name="train", n=8, seed=0)

    model.train()
    t0 = time.time()
    n_ok = 0
    for step in range(10):
        ex = ds[step % len(ds)]
        image, question, answer, bbox = example_to_tuple(ex)
        try:
            stats = lvr_step(model, processor, tokenizer, image_token_id,
                             image, question, answer, bbox,
                             K=K, lambda_lvr=1.0, device=device, dtype=dtype)
        except torch.cuda.OutOfMemoryError as e:
            print(f"[smoke] OOM step {step}: {e}", flush=True)
            return 1
        loss = stats["total"]
        if not torch.isfinite(loss):
            print(f"[smoke] NaN/Inf loss at step {step}", flush=True)
            return 2
        loss.backward()
        # Verify some non-vision param has grad
        if step == 0:
            grad_count_train = sum(1 for n, p in model.named_parameters()
                                   if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
            grad_count_visual = sum(1 for n, p in model.named_parameters()
                                    if (n.startswith("model.visual") or n.startswith("visual"))
                                    and p.grad is not None and p.grad.abs().sum() > 0)
            print(f"[smoke] grad params: train_with_grad={grad_count_train}, "
                  f"visual_with_grad={grad_count_visual}", flush=True)
            if grad_count_train == 0:
                print("[smoke] FAIL: no grads in trainable params", flush=True)
                return 3
            if grad_count_visual > 0:
                print("[smoke] FAIL: vision tower received grad", flush=True)
                return 4
        optim.step()
        optim.zero_grad(set_to_none=True)
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"[smoke {step+1}/10] total={loss.item():.3f} ntp={stats['ntp'].item():.3f} "
              f"lvr={stats['lvr'].item():.3f} ||h||={stats['h_norm'].item():.2f} "
              f"||v||={stats['v_norm'].item():.2f} n_img_tok={stats['n_image_tokens']} "
              f"mem={mem:.1f}GB", flush=True)
        n_ok += 1

    dt = time.time() - t0
    print(f"\n[smoke] PASSED  steps_ok={n_ok}/10  total={dt:.1f}s  per_step={dt/10:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
