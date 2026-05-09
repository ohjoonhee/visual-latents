"""Cluster Phase 3 — Stage 1 SFT (NTP-only) trainer.

Goal. Plain next-token-prediction warmup on Monet-SFT-125K Visual_CoT
(118K samples after eval-200 hold-out). Vision tower + projector frozen,
LLM full FT. Output checkpoint becomes the Stage 2 / Pivot A initialization.

Recipe.
- base: Qwen/Qwen2.5-VL-7B-Instruct (raw, no Monet special tokens).
- data: Monet-SFT-125K Visual_CoT, train slice (eval-200 excluded).
- objective: standard CE on the answer span only (system + user spans
  ignore_index=-100). Special tokens (image pads, vision_start/end) also
  ignored. We do NOT add Monet special tokens here — those come at Stage 2.
- 1 epoch, eff_bsz=128 (4 GPUs × micro_bsz=2 × grad_accum=16), lr=1e-5,
  cosine schedule, warmup=100, weight_decay=0.0, AdamW betas=(0.9, 0.95),
  bf16, gradient checkpointing.

Distributed launch (set in sbatch):
  accelerate launch --config_file cluster/accelerate_zero2.yaml \\
                    cluster/trainer_sft.py --config cluster/configs/stage1_sft.yaml

Implementation note. Accelerate + ZeRO-2 + bf16 owns the gradient_accumulation
machinery (see accelerate_zero2.yaml). The trainer's own step accounting must
agree with that — `cfg["grad_accum_steps"]` is an annotation, NOT a second
divisor. We rely on accelerate's `accumulate(model)` context manager.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.optim import AdamW
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data_utils import load_monet_sft_125k  # noqa: E402


# ---------- helpers ----------

def _build_processor(base: str, max_pixels=None, min_pixels=None):
    processor = AutoProcessor.from_pretrained(base, use_fast=True, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        if max_pixels:
            processor.image_processor.max_pixels = int(max_pixels)
        if min_pixels:
            processor.image_processor.min_pixels = int(min_pixels)
    return processor


def _build_chat_inputs(processor, sample, *, global_max_pixels: int, per_img_max_pixels: int):
    """Mirror Stage 2 trainer's text construction WITHOUT latent tokens.

    For Stage 1 SFT we keep the chat structure (system / user / assistant
    turns with embedded images) but do NOT add latent slots — the model is
    learning vanilla NTP on the assistant outputs. Monet's `<observation>`
    tokens are present in the text content but treated as regular text
    (not masked) — Stage 1's job is to teach the model to fluently emit
    those tags, exactly per the upstream Stage 1 SFT recipe.

    Returns dict with keys input_ids, attention_mask, pixel_values,
    image_grid_thw, ans_start (int — first label position).
    """
    sys.path.insert(0, str(ROOT.parent / "phase0_monet_probe"))
    from monet_utils import resize_by_token_budget  # noqa: E402

    example = sample["data"]
    text = processor.apply_chat_template(example, tokenize=False)
    image_inputs, _ = process_vision_info([example])
    if image_inputs:
        image_inputs, _ = resize_by_token_budget(
            image_inputs,
            global_max_pixels=global_max_pixels,
            per_img_max_pixels=per_img_max_pixels,
        )
    total_pads = text.count("<|vision_start|><|image_pad|>")
    if total_pads != len(image_inputs):
        return None

    batch = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
    return batch


def _generate_labels(input_ids: torch.Tensor, ans_start_pattern: torch.Tensor, ignore_ids: list[int]) -> torch.Tensor:
    """Build labels: -100 before answer-start, copy thereafter, -100 for ignore_ids."""
    B, L = input_ids.shape
    labels = torch.full_like(input_ids, -100)
    pat = ans_start_pattern
    pat_len = pat.numel()
    pat_cpu = pat.cpu()
    ids_cpu = input_ids.detach().cpu()
    for b in range(B):
        ids_b = ids_cpu[b]
        ans_pos = -1
        for s in range(L - pat_len + 1):
            if torch.equal(ids_b[s : s + pat_len], pat_cpu):
                ans_pos = s + pat_len
                break
        if ans_pos < 0:
            continue
        labels[b, ans_pos:] = input_ids[b, ans_pos:]
        for ig in ignore_ids:
            mask = labels[b, ans_pos:] == ig
            labels[b, ans_pos:][mask] = -100
    return labels


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None, help="override cfg.out_dir")
    ap.add_argument("--max_steps", type=int, default=None, help="override cfg.max_steps")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.max_steps is not None:
        cfg["max_steps"] = int(args.max_steps)

    accelerator = Accelerator(log_with=["wandb"]) if cfg.get("wandb", True) else Accelerator()
    is_main = accelerator.is_main_process
    set_seed(int(cfg.get("seed", 0)))

    if is_main:
        print(f"[config] {cfg}", flush=True)
        if cfg.get("wandb", True):
            import wandb
            wandb.init(
                project=cfg.get("wandb_project", "visual-latents"),
                name=cfg.get("run_name", "stage1_sft"),
                config=cfg,
                dir=os.environ.get("WANDB_DIR", "./wandb"),
            )

    out_dir = Path(cfg["out_dir"])
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"

    base = cfg["base_model"]
    processor = _build_processor(base, cfg.get("max_pixels"), cfg.get("min_pixels"))
    tokenizer = processor.tokenizer
    ans_start_pattern = tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    # ignore_ids — special tokens we NEVER want to score.
    ignore_ids = [
        int(tokenizer(s, return_tensors="pt")["input_ids"][0, 0])
        for s in ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
                  "<|image_pad|>", "<|endoftext|>"]
    ]

    if is_main:
        print(f"[load] {base}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16,
    )
    # Freeze vision tower + projector.
    for n, p in model.named_parameters():
        if n.startswith("model.visual") or n.startswith("visual"):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        betas=(0.9, 0.95),
    )
    n_steps = int(cfg["max_steps"]) if not args.smoke else 5
    warmup = int(cfg.get("warmup_steps", 100))
    sched = get_cosine_schedule_with_warmup(optim, warmup, n_steps)

    # Data.
    n_train = int(cfg.get("n_train_examples", 0)) or None
    train_ds, _eval = load_monet_sft_125k(
        subset=cfg.get("subset", "Visual_CoT"),
        n=n_train,
        seed=int(cfg.get("seed", 0)),
        eval_holdout=int(cfg.get("eval_holdout", 200)),
        allow_no_observation_for_train=False,
    )
    if is_main:
        print(f"[data] train={len(train_ds)}", flush=True)

    grad_accum = int(cfg.get("grad_accum_steps", 16))
    log_every = int(cfg.get("log_every", 10))
    save_every = int(cfg.get("save_every", 0))
    global_max_pixels = int(cfg.get("global_max_pixels", 1000 * 28 * 28))
    per_img_pixels = int(cfg.get("per_img_pixels", 500 * 28 * 28))

    model, optim, sched = accelerator.prepare(model, optim, sched)
    model.train()

    step = 0
    t0 = time.time()
    losses_micro = []
    rank = accelerator.process_index
    world = accelerator.num_processes
    epoch_idx = 0
    n_examples = len(train_ds)

    while step < n_steps:
        # Shard the (already-shuffled) list across ranks; each rank walks its own slice.
        # We re-shuffle each epoch with seed + epoch_idx so all ranks agree.
        idx = list(range(n_examples))
        import random as _rnd
        _rnd.Random(int(cfg.get("seed", 0)) + epoch_idx).shuffle(idx)
        my_idx = idx[rank::world]
        for i in my_idx:
            sample = train_ds[i]
            try:
                batch = _build_chat_inputs(
                    processor, sample,
                    global_max_pixels=global_max_pixels,
                    per_img_max_pixels=per_img_pixels,
                )
                if batch is None:
                    continue
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.bfloat16)
                image_grid_thw = batch["image_grid_thw"].to(accelerator.device)
                labels = _generate_labels(input_ids, ans_start_pattern, ignore_ids).to(accelerator.device)

                with accelerator.accumulate(model):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        labels=labels,
                        use_cache=False,
                    )
                    loss = out.loss
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], 1.0,
                        )
                    optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                if is_main:
                    print(f"  [OOM ex={i}] {e}", flush=True)
                continue
            except Exception as e:
                if is_main:
                    print(f"  [error ex={i}] {type(e).__name__}: {e}", flush=True)
                continue

            losses_micro.append(float(loss.detach().float().cpu().item()))

            if accelerator.sync_gradients:
                step += 1
                if (step % log_every == 0 or step == 1) and is_main:
                    avg = sum(losses_micro) / max(1, len(losses_micro))
                    mem_gb = torch.cuda.max_memory_allocated() / 1e9
                    row = {
                        "step": step,
                        "epoch": epoch_idx,
                        "lr": sched.get_last_lr()[0],
                        "ntp_loss": avg,
                        "gpu_peak_gb": mem_gb,
                        "elapsed_s": time.time() - t0,
                    }
                    print(f"[step {step}/{n_steps}] ntp={avg:.3f} mem={mem_gb:.1f}GB elapsed={row['elapsed_s']:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    if cfg.get("wandb", True):
                        import wandb
                        wandb.log(row, step=step)
                    losses_micro = []
                if save_every and step > 0 and step % save_every == 0 and is_main and not args.smoke:
                    ck = out_dir / f"checkpoint_step{step}"
                    ck.mkdir(parents=True, exist_ok=True)
                    accelerator.unwrap_model(model).save_pretrained(ck, safe_serialization=True)
                    processor.save_pretrained(ck)
                    print(f"[ckpt] saved {ck}", flush=True)
                if step >= n_steps:
                    break
        epoch_idx += 1
        if step >= n_steps:
            break

    accelerator.wait_for_everyone()
    if is_main and not args.smoke:
        ck = out_dir / "checkpoint"
        ck.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(ck, safe_serialization=True)
        processor.save_pretrained(ck)
        print(f"[done] saved final checkpoint to {ck}", flush=True)


if __name__ == "__main__":
    main()
