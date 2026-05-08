"""Pivot A trainer = Phase 1 LVR trainer + collapse-prevention regularizer.

Loss form (mean-MSE, NOT sum-MSE):
    L = L_NTP + lambda_lvr · F.mse_loss(h, v_roi) + lambda_reg · L_reg

L_reg is one of:
  - "cos":    cos_penalty_loss(h, tau=cfg["reg_tau"])
  - "vicreg": vicreg_loss(h, var_weight, cov_weight, gamma)
  - "none":   0

The trainer is otherwise identical to `phase1_lvr/trainer.py`.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.optim import AdamW
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
# Re-use Phase 1 data + roi loaders (same dataset, same ROI logic).
sys.path.insert(0, str(ROOT.parent / "phase1_lvr"))
from data import load_phase1_dataset, example_to_tuple  # noqa: E402
from loss import lvr_mse, lvr_mse_paper  # noqa: E402
from roi import select_roi_indices, post_merger_token_count  # noqa: E402

sys.path.insert(0, str(ROOT))
from reg import cos_penalty_loss, vicreg_loss  # noqa: E402


SYSTEM_PROMPT = "You are a helpful assistant."


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_prompt(question: str, answer: str, K: int, n_image_tokens: int) -> tuple[str, str]:
    img_pads = "<|image_pad|>" * n_image_tokens
    latent_pads = "<|image_pad|>" * K
    prefix = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{img_pads}<|vision_end|>{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{latent_pads}"
    )
    full = prefix + " " + answer + "<|im_end|>"
    return prefix, full


@torch.no_grad()
def _encode_image_features(model, pixel_values, image_grid_thw):
    feats = model.model.get_image_features(pixel_values, image_grid_thw)
    return torch.cat(feats, dim=0)


def lvr_step(
    model,
    processor,
    tokenizer,
    image_token_id: int,
    image: Image.Image,
    question: str,
    answer: str,
    bbox: tuple,
    K: int,
    lambda_lvr: float,
    lambda_reg: float,
    reg_kind: str,
    reg_kwargs: dict,
    device: torch.device,
    dtype: torch.dtype,
    merge_size: int = 2,
    loss_form: str = "mean_d",
):
    # --- 1. Vision features (frozen) ---
    vision_inputs = processor.image_processor(images=image, return_tensors="pt")
    pixel_values = vision_inputs["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = vision_inputs["image_grid_thw"].to(device=device)
    n_image_tokens = post_merger_token_count(image_grid_thw, merge_size=merge_size)

    with torch.no_grad():
        v_full = _encode_image_features(model, pixel_values, image_grid_thw)

    # --- 2. ROI indices ---
    roi_idx = select_roi_indices(image_grid_thw, bbox, K=K, merge_size=merge_size)
    v_roi = v_full[torch.tensor(roi_idx, device=device, dtype=torch.long)]

    # --- 3. Build prompt & tokenize ---
    prefix, full = _build_prompt(question, answer, K, n_image_tokens)
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    if full_ids.shape[1] < prefix_ids.shape[1] or not torch.equal(full_ids[0, : prefix_ids.shape[1]], prefix_ids[0]):
        ans_ids = tokenizer(" " + answer + "<|im_end|>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        full_ids = torch.cat([prefix_ids, ans_ids], dim=1)

    pad_positions = (full_ids[0] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
    if pad_positions.numel() < n_image_tokens + K:
        raise RuntimeError(
            f"Expected ≥{n_image_tokens+K} image_pad positions, got {pad_positions.numel()}"
        )
    image_patch_positions = pad_positions[:n_image_tokens]
    slot_positions = pad_positions[n_image_tokens : n_image_tokens + K]

    L = full_ids.shape[1]
    labels = torch.full((1, L), -100, dtype=torch.long, device=device)
    ans_start = prefix_ids.shape[1]
    labels[0, ans_start:] = full_ids[0, ans_start:]

    attention_mask = torch.ones_like(full_ids)

    text_embeds = model.get_input_embeddings()(full_ids)
    inputs_embeds = text_embeds.clone()
    inputs_embeds[0, image_patch_positions, :] = v_full.to(inputs_embeds.dtype)

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    ntp_loss = outputs.loss
    last_hidden = outputs.hidden_states[-1]
    h = last_hidden[0, slot_positions, :]  # [K, D]

    if loss_form == "sum_d":
        lvr = lvr_mse_paper(h.unsqueeze(0).float(), v_roi.unsqueeze(0).float())
    elif loss_form == "mean_d":
        lvr = lvr_mse(h.unsqueeze(0).float(), v_roi.unsqueeze(0).float())
    else:
        raise ValueError(f"unknown loss_form={loss_form!r} (expected 'sum_d' or 'mean_d')")

    # --- Pivot A regularizer ---
    if reg_kind == "cos":
        reg = cos_penalty_loss(h.unsqueeze(0).float(), tau=reg_kwargs["tau"])
    elif reg_kind == "vicreg":
        reg = vicreg_loss(
            h.unsqueeze(0).float(),
            var_weight=reg_kwargs["var_w"],
            cov_weight=reg_kwargs["cov_w"],
            gamma=reg_kwargs["gamma"],
        )
    elif reg_kind in (None, "none", ""):
        reg = torch.zeros((), device=h.device)
    else:
        raise ValueError(f"unknown reg_kind={reg_kind!r}")

    total = ntp_loss + lambda_lvr * lvr + lambda_reg * reg

    return {
        "total": total,
        "ntp": ntp_loss.detach(),
        "lvr": lvr.detach(),
        "reg": reg.detach(),
        "h_norm": h.detach().float().norm(dim=-1).mean(),
        "v_norm": v_roi.detach().float().norm(dim=-1).mean(),
        "n_image_tokens": int(n_image_tokens),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--smoke", action="store_true", help="50-step gradient probe; no checkpoint save")
    ap.add_argument("--smoke_steps", type=int, default=50)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / ("training_log_smoke.jsonl" if args.smoke else "training_log.jsonl")

    _set_seed(cfg.get("seed", 0))

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"[config] {cfg}", flush=True)

    base = cfg["base_model"]
    print(f"[load] {base}", flush=True)
    processor = AutoProcessor.from_pretrained(base, use_fast=True, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        if cfg.get("max_pixels"):
            processor.image_processor.max_pixels = int(cfg["max_pixels"])
        if cfg.get("min_pixels"):
            processor.image_processor.min_pixels = int(cfg["min_pixels"])

    tokenizer = processor.tokenizer

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, torch_dtype=dtype)
    model.to(device)

    image_token_id = model.config.image_token_id

    for n, p in model.named_parameters():
        if n.startswith("model.visual") or n.startswith("visual"):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_freeze = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[freeze] trainable={n_train/1e6:.1f}M  frozen={n_freeze/1e6:.1f}M", flush=True)

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        betas=(0.9, 0.95),
    )
    n_steps = int(cfg["max_steps"]) if not args.smoke else args.smoke_steps
    warmup = int(cfg.get("warmup_steps", 100))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=n_steps)

    n_train_examples = int(cfg.get("n_train_examples", 5000)) if not args.smoke else 4 * args.smoke_steps
    train_split = cfg.get("train_split", "train")
    seed = int(cfg.get("seed", 0))
    print(f"[data] loading split={train_split} n={n_train_examples}", flush=True)
    train_ds = load_phase1_dataset(split_name=train_split, n=n_train_examples, seed=seed)
    print(f"[data] loaded {len(train_ds)} examples", flush=True)

    K = int(cfg.get("K", 8))
    lambda_lvr = float(cfg.get("lambda_lvr", 1.0))
    lambda_reg = float(cfg.get("lambda_reg", 0.0))
    reg_kind = str(cfg.get("reg_kind", "none"))
    reg_kwargs = {
        "tau": float(cfg.get("reg_tau", 0.5)),
        "var_w": float(cfg.get("reg_var_w", 1.0)),
        "cov_w": float(cfg.get("reg_cov_w", 0.04)),
        "gamma": float(cfg.get("reg_gamma", 1.0)),
    }
    grad_accum = int(cfg.get("grad_accum_steps", 4))
    log_every = int(cfg.get("log_every", 10))
    save_every = int(cfg.get("save_every", 0))
    loss_form = str(cfg.get("loss_form", "mean_d"))
    if loss_form not in ("mean_d", "sum_d"):
        raise ValueError(f"loss_form must be 'mean_d' or 'sum_d', got {loss_form!r}")
    print(f"[loss] form={loss_form} reg_kind={reg_kind} lambda_reg={lambda_reg} reg_kwargs={reg_kwargs}", flush=True)

    model.train()
    optim.zero_grad(set_to_none=True)

    step = 0
    micro = 0
    t0 = time.time()
    log_buf: list[dict] = []
    n_examples = len(train_ds)
    epoch_idx = 0

    losses_micro = {"total": [], "ntp": [], "lvr": [], "reg": [], "h_norm": [], "v_norm": []}

    while step < n_steps:
        idx_perm = list(range(n_examples))
        random.Random(seed + epoch_idx).shuffle(idx_perm)
        for i in idx_perm:
            ex = train_ds[i]
            try:
                image, question, answer, bbox = example_to_tuple(ex)
            except Exception as e:
                print(f"  [skip ex={i}] {e}", flush=True)
                continue
            try:
                stats = lvr_step(
                    model, processor, tokenizer, image_token_id,
                    image, question, answer, bbox,
                    K=K, lambda_lvr=lambda_lvr, lambda_reg=lambda_reg,
                    reg_kind=reg_kind, reg_kwargs=reg_kwargs,
                    device=device, dtype=dtype,
                    loss_form=loss_form,
                )
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                print(f"  [OOM ex={i}] {e}", flush=True)
                continue
            except Exception as e:
                print(f"  [error ex={i}] {type(e).__name__}: {e}", flush=True)
                continue

            (stats["total"] / grad_accum).backward()
            for k in losses_micro:
                losses_micro[k].append(float(stats[k].detach().cpu().item()) if hasattr(stats[k], "detach") else float(stats[k]))
            micro += 1

            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                if step % log_every == 0 or step == 1:
                    mem_gb = torch.cuda.max_memory_allocated() / 1e9
                    avg = {k: (sum(v) / len(v)) if v else float("nan") for k, v in losses_micro.items()}
                    row = {
                        "step": step,
                        "epoch": epoch_idx,
                        "lr": sched.get_last_lr()[0],
                        "total_loss": avg["total"],
                        "ntp_loss": avg["ntp"],
                        "lvr_loss": avg["lvr"],
                        "reg_loss": avg["reg"],
                        "h_norm": avg["h_norm"],
                        "v_norm": avg["v_norm"],
                        "gpu_peak_gb": mem_gb,
                        "elapsed_s": time.time() - t0,
                    }
                    print(f"[step {step}/{n_steps}] loss={row['total_loss']:.3f} ntp={row['ntp_loss']:.3f} "
                          f"lvr={row['lvr_loss']:.3f} reg={row['reg_loss']:.3f} ||h||={row['h_norm']:.2f} "
                          f"||v||={row['v_norm']:.2f} mem={row['gpu_peak_gb']:.1f}GB elapsed={row['elapsed_s']:.0f}s", flush=True)
                    log_buf.append(row)
                    with open(log_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    losses_micro = {k: [] for k in losses_micro}

                if save_every and step > 0 and step % save_every == 0 and not args.smoke:
                    ck = out_dir / f"checkpoint_step{step}"
                    ck.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ck)
                    processor.save_pretrained(ck)
                    print(f"[ckpt] saved {ck}", flush=True)

                if step >= n_steps:
                    break
        epoch_idx += 1

    if not args.smoke:
        ck = out_dir / "checkpoint"
        ck.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ck)
        processor.save_pretrained(ck)
        print(f"[done] saved final checkpoint to {ck}", flush=True)
    else:
        print(f"[smoke] done {step} steps", flush=True)


if __name__ == "__main__":
    main()
