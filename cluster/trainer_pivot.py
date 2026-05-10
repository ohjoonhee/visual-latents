"""Cluster Phase 3 — Pivot A trainer (V1, V2, V4).

Mirrors `pivot_a/trainer.py` (LVR + collapse-prevention regularizer) at
cluster scale (7B + 125K via Visual_CoT, 4×H100 via accelerate+ZeRO-2).

Loss form (mean-MSE LVR by default):
    L = L_NTP + lambda_lvr · F.mse_loss(h, v_roi) + lambda_reg · L_reg

L_reg ∈ {"vicreg", "cos", "none"}.

Variants (driven entirely by the YAML config — same trainer for all):

  V1 (G@7B):  K=4, λ_reg=2.0, vicreg, mean-MSE LVR, 2 epochs, eff_bsz=128
  V2 (D2@7B): K=8, λ_reg=2.0, vicreg, mean-MSE LVR, 2 epochs, eff_bsz=128
  V4 (LVR-only baseline, optional): K=4, λ_reg=0, no reg, 1 epoch

Init from Stage 1 SFT checkpoint (controlled common base for all variants);
override via `--init_ckpt`. The phase1_lvr ROI helper, the LVR loss form,
and the regularizer are imported from local sibling files (`reg.py` lives
in `cluster/`; `roi.py` and `loss.py` come from `phase1_lvr/`).

Dataset selector (`cfg.dataset_format`).

  - "monet_sft_125k" (DEFAULT for V1/V2 — controlled comparison with Stage 1
    SFT, Stage 2 baseline, and V3): loads via `data_utils.load_monet_sft_125k`,
    eval-200 excluded. Each Monet trace yields a Pivot-A tuple
    `(image, question, answer, bbox)` via `_monet_sample_to_pivot_tuple`:
        image    = the user turn's first image (the "question image"),
                   loaded with PIL.Image.open and converted to RGB.
        question = the user turn's text content (concatenated if multiple),
                   stripped of leading whitespace.
        answer   = the LAST assistant text content in the trace, with
                   `<observation>` / `</observation>` / `<abs_vis_token>` tags
                   scrubbed to natural language. (Same convention as
                   `phase0_monet_probe/extract_latents.py:_extract_answer`.)
        bbox     = **CENTER-CROP FALLBACK**: Monet-SFT-125K does NOT carry
                   coordinate boxes — the auxiliary images are the visual
                   reasoning artifact. With no per-example ROI to gather
                   image features at, we fall back to the central-half
                   patch of the question image: `(0.25, 0.25, 0.75, 0.75)`
                   in normalized xyxy. The Phase-1 ROI helper
                   (`phase1_lvr/roi.py:select_roi_indices`) then picks the
                   K post-merger image-patch indices whose centers lie
                   inside that box. This is a deviation from Pivot A's
                   per-example ROI selection — it makes the LVR target a
                   fixed central crop rather than a question-conditional
                   region. We document it here and in the v1/v2 configs.
                   The VICReg regularizer is unaffected (it acts on the
                   slot hidden states, not the LVR target).

  - "visual_cot_50k_poc" (back-compat): loads `ohjoonhee/visual-cot-50k-poc`
    via `data_utils.load_pivot_a_dataset`. Each example carries a
    per-example normalized bbox so no fallback is needed. NOT used by
    V1/V2 in the controlled-comparison Phase 3 plan.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torch.optim import AdamW
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data_utils import (  # noqa: E402
    load_monet_sft_125k,
    load_pivot_a_dataset,
    pivot_a_example_to_tuple,
)
from reg import cos_penalty_loss, vicreg_loss  # noqa: E402

# Reuse phase1_lvr ROI + loss helpers (no need to re-vendor — repo layout is shared).
sys.path.insert(0, str(ROOT.parent / "phase1_lvr"))
from loss import lvr_mse, lvr_mse_paper  # noqa: E402
from roi import select_roi_indices, post_merger_token_count  # noqa: E402


SYSTEM_PROMPT = "You are a helpful assistant."

# Center-crop fallback box (normalized xyxy) used when a Monet-SFT-125K
# sample carries no per-example bbox. Covers the central 50% × 50% of the
# question image. See module docstring for rationale.
_MONET_CENTER_BBOX = (0.25, 0.25, 0.75, 0.75)


def _monet_sample_to_pivot_tuple(processed: dict):
    """Convert a Monet `processed` sample (output of
    `Monet_single_input_images_preprocess_function`) into the Pivot-A
    per-example tuple `(image, question, answer, bbox)`.

    Returns None on malformed samples (no user image, no assistant text).

    bbox is the center-crop fallback (Monet-SFT-125K has no coordinate ROI
    — see module docstring).
    """
    user_image_path = None
    user_text_parts: list[str] = []
    last_assistant_text = ""

    for turn in processed.get("data", []):
        role = turn.get("role")
        for c in turn.get("content", []):
            if c.get("type") == "image" and role == "user" and user_image_path is None:
                user_image_path = c.get("image")
            elif c.get("type") == "text":
                if role == "user":
                    user_text_parts.append(c.get("text", ""))
                elif role == "assistant":
                    last_assistant_text = c.get("text", last_assistant_text)

    if user_image_path is None:
        return None
    if not last_assistant_text:
        return None

    try:
        image = Image.open(user_image_path)
    except Exception:
        return None
    if image.mode != "RGB":
        image = image.convert("RGB")

    question = "\n".join(t for t in user_text_parts if t).lstrip("\n").strip()
    if not question:
        return None

    # Scrub Monet structural tags from the answer; mirror
    # phase0_monet_probe/extract_latents.py:_extract_answer.
    answer = (
        last_assistant_text
        .replace("<abs_vis_token></abs_vis_token>", "")
        .replace("<observation>", "")
        .replace("</observation>", "")
        .strip()
    )
    if not answer:
        return None

    return image, question, answer, _MONET_CENTER_BBOX


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
    model, processor, tokenizer, image_token_id: int,
    image: Image.Image, question: str, answer: str, bbox: tuple,
    K: int, lambda_lvr: float, lambda_reg: float,
    reg_kind: str, reg_kwargs: dict,
    device, dtype,
    merge_size: int = 2, loss_form: str = "mean_d",
):
    """Direct port of `pivot_a/trainer.py:lvr_step`. Returns (total, stats_dict)."""
    vision_inputs = processor.image_processor(images=image, return_tensors="pt")
    pixel_values = vision_inputs["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = vision_inputs["image_grid_thw"].to(device=device)
    n_image_tokens = post_merger_token_count(image_grid_thw, merge_size=merge_size)

    with torch.no_grad():
        v_full = _encode_image_features(model, pixel_values, image_grid_thw)

    roi_idx = select_roi_indices(image_grid_thw, bbox, K=K, merge_size=merge_size)
    v_roi = v_full[torch.tensor(roi_idx, device=device, dtype=torch.long)]

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

    out = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    ntp_loss = out.loss
    last_hidden = out.hidden_states[-1]
    h = last_hidden[0, slot_positions, :]  # [K, D]

    if loss_form == "sum_d":
        lvr = lvr_mse_paper(h.unsqueeze(0).float(), v_roi.unsqueeze(0).float())
    elif loss_form == "mean_d":
        lvr = lvr_mse(h.unsqueeze(0).float(), v_roi.unsqueeze(0).float())
    else:
        raise ValueError(f"unknown loss_form={loss_form!r}")

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
    stats = {
        "total": total,
        "ntp": ntp_loss.detach(),
        "lvr": lvr.detach(),
        "reg": reg.detach(),
        "h_norm": h.detach().float().norm(dim=-1).mean(),
        "v_norm": v_roi.detach().float().norm(dim=-1).mean(),
    }
    return total, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--init_ckpt", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.max_steps is not None:
        cfg["max_steps"] = int(args.max_steps)
    if args.init_ckpt:
        cfg["init_ckpt"] = args.init_ckpt

    accelerator = Accelerator()
    is_main = accelerator.is_main_process
    set_seed(int(cfg.get("seed", 0)))

    out_dir = Path(cfg["out_dir"])
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[config] {cfg}", flush=True)
        if cfg.get("wandb", True):
            import wandb
            wandb.init(
                project=cfg.get("wandb_project", "visual-latents"),
                name=cfg.get("run_name", "pivot"),
                config=cfg,
                dir=os.environ.get("WANDB_DIR", "./wandb"),
            )
    log_path = out_dir / "training_log.jsonl"

    base = cfg.get("init_ckpt") or cfg["base_model"]
    if is_main:
        print(f"[load] {base}", flush=True)
    processor = AutoProcessor.from_pretrained(base, use_fast=True, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        if cfg.get("max_pixels"):
            processor.image_processor.max_pixels = int(cfg["max_pixels"])
        if cfg.get("min_pixels"):
            processor.image_processor.min_pixels = int(cfg["min_pixels"])
    tokenizer = processor.tokenizer

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16)
    image_token_id = model.config.image_token_id

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

    n_train = int(cfg.get("n_train_examples", 0)) or None
    train_split = cfg.get("train_split", "train")
    seed = int(cfg.get("seed", 0))
    dataset_format = str(cfg.get("dataset_format", "visual_cot_50k_poc"))
    if dataset_format == "monet_sft_125k":
        train_ds, _eval_holdout = load_monet_sft_125k(
            subset=cfg.get("subset", "Visual_CoT"),
            n=n_train,
            seed=seed,
            eval_holdout=int(cfg.get("eval_holdout", 200)),
            allow_no_observation_for_train=False,
        )
    elif dataset_format == "visual_cot_50k_poc":
        train_ds = load_pivot_a_dataset(
            split_name=train_split,
            n=n_train,
            seed=seed,
            dataset_id=cfg.get("dataset_id", "ohjoonhee/visual-cot-50k-poc"),
        )
    else:
        raise ValueError(
            f"unknown dataset_format={dataset_format!r}; "
            f"expected 'monet_sft_125k' or 'visual_cot_50k_poc'"
        )
    if is_main:
        print(f"[data] format={dataset_format} train={len(train_ds)}", flush=True)

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
    log_every = int(cfg.get("log_every", 10))
    save_every = int(cfg.get("save_every", 0))
    loss_form = str(cfg.get("loss_form", "mean_d"))
    if loss_form not in ("mean_d", "sum_d"):
        raise ValueError(f"loss_form must be 'mean_d' or 'sum_d', got {loss_form!r}")

    model, optim, sched = accelerator.prepare(model, optim, sched)
    model.train()
    rank = accelerator.process_index
    world = accelerator.num_processes

    if accelerator.is_main_process:
        try:
            import torch.cuda as _tc
            alloc_gb = _tc.memory_allocated() / 1e9
            reserved_gb = _tc.memory_reserved() / 1e9
            print(f"[mem-after-prepare] dist_type={accelerator.distributed_type} "
                  f"alloc={alloc_gb:.2f}GB reserved={reserved_gb:.2f}GB world={world}", flush=True)
        except Exception as _e:
            print(f"[mem-after-prepare] log failed: {_e}", flush=True)

    step = 0
    t0 = time.time()
    losses_micro = {"total": [], "ntp": [], "lvr": [], "reg": [], "h_norm": [], "v_norm": []}
    epoch_idx = 0
    n_examples = len(train_ds)

    while step < n_steps:
        idx = list(range(n_examples))
        random.Random(seed + epoch_idx).shuffle(idx)
        my_idx = idx[rank::world]
        for i in my_idx:
            ex = train_ds[i]
            try:
                if dataset_format == "monet_sft_125k":
                    pivot_tuple = _monet_sample_to_pivot_tuple(ex)
                    if pivot_tuple is None:
                        continue
                    image, question, answer, bbox = pivot_tuple
                else:
                    image, question, answer, bbox = pivot_a_example_to_tuple(ex)
            except Exception as e:
                if is_main:
                    print(f"  [skip ex={i}] {e}", flush=True)
                continue
            try:
                with accelerator.accumulate(model):
                    total, stats = lvr_step(
                        accelerator.unwrap_model(model), processor, tokenizer, image_token_id,
                        image, question, answer, bbox,
                        K=K, lambda_lvr=lambda_lvr, lambda_reg=lambda_reg,
                        reg_kind=reg_kind, reg_kwargs=reg_kwargs,
                        device=accelerator.device, dtype=torch.bfloat16,
                        loss_form=loss_form,
                    )
                    accelerator.backward(total)
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

            for k_ in losses_micro:
                v = stats[k_]
                losses_micro[k_].append(float(v.detach().float().cpu().item()) if hasattr(v, "detach") else float(v))

            if accelerator.sync_gradients:
                step += 1
                if (step % log_every == 0 or step == 1) and is_main:
                    avg = {k: (sum(v)/len(v)) if v else float("nan") for k, v in losses_micro.items()}
                    mem_gb = torch.cuda.max_memory_allocated() / 1e9
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
                    print(f"[step {step}/{n_steps}] tot={row['total_loss']:.3f} ntp={row['ntp_loss']:.3f} "
                          f"lvr={row['lvr_loss']:.3f} reg={row['reg_loss']:.3f} ||h||={row['h_norm']:.2f} "
                          f"mem={mem_gb:.1f}GB elapsed={row['elapsed_s']:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    if cfg.get("wandb", True):
                        import wandb
                        wandb.log(row, step=step)
                    losses_micro = {k: [] for k in losses_micro}
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
