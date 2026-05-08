"""Phase 1.5b trainer: identical to Phase 1.5 EXCEPT we inject the
`attention_mask_4d` (cross-slot-isolation variant) on every forward.

Hypothesis under test: cross-slot attention isolation is the load-bearing
mechanism that prevents Phase 1.5's collapsed (mean off-diag cos = 0.959)
latents.

Recipe — identical to Phase 1.5 (mean-MSE λ=1.0, K=8, 1000 steps,
eff_bsz=4, lr=1e-5, warmup=100, seed=0, 5K Visual-CoT) except:
  - construct a 4D attention mask (latent_cross_isolate=True) and pass it
    via `attention_mask_4d=...` in BOTH the latent_mode=True and the
    latent_mode=False forwards.
"""
from __future__ import annotations

# ===== monkey-patch BEFORE any transformers import =====
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

assert "transformers" not in sys.modules, "FATAL: transformers already imported."

_patch_path = PHASE0 / "monet_model" / "modeling_qwen2_5_vl_monet.py"
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

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.optim import AdamW
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup

# Phase 1 helpers (re-used verbatim).
sys.path.insert(0, str(ROOT.parent / "phase1_lvr"))
from data import load_phase1_dataset, example_to_tuple  # noqa: E402
from loss import lvr_mse  # noqa: E402  (mean-MSE form; matches Phase 1 run 1)
from roi import select_roi_indices, post_merger_token_count  # noqa: E402

# Phase 1.5b mask helpers.
sys.path.insert(0, str(ROOT))
from mask_utils import build_monet_4d_attn  # noqa: E402

LATENT_SIZE = int(os.environ["LATENT_SIZE"])  # 8
SYSTEM_PROMPT = "You are a helpful assistant."


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_processor(base: str):
    processor = AutoProcessor.from_pretrained(base, use_fast=True, trust_remote_code=True)
    for tok in [
        "<abs_vis_token_pad>",
        "<abs_vis_token>",
        "</abs_vis_token>",
        "<observation>",
        "</observation>",
    ]:
        processor.tokenizer.add_tokens(tok, special_tokens=True)
    return processor


def _build_prompt(question: str, answer: str, K: int, n_image_tokens: int) -> tuple[str, str]:
    img_pads = "<|image_pad|>" * n_image_tokens
    latent_pads = "<abs_vis_token_pad>" * K
    prefix = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{img_pads}<|vision_end|>{question}<|im_end|>\n"
        f"<|im_start|>assistant\n<abs_vis_token>{latent_pads}</abs_vis_token>"
    )
    full = prefix + " " + answer + "<|im_end|>"
    return prefix, full


@torch.no_grad()
def _encode_image_features(model, pixel_values, image_grid_thw):
    feats = model.model.get_image_features(pixel_values, image_grid_thw)
    return torch.cat(feats, dim=0)


def lvr_step_monet(
    model,
    processor,
    tokenizer,
    image_token_id: int,
    latent_pad_id: int,
    answer_start_pattern: torch.Tensor,
    image: Image.Image,
    question: str,
    answer: str,
    bbox: tuple,
    K: int,
    lambda_lvr: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """Single Phase 1.5b fwd/bwd (no opt.step). Returns dict of stats + loss tensor."""
    # --- 1. Vision features (frozen) ---
    vision_inputs = processor.image_processor(images=image, return_tensors="pt")
    pixel_values = vision_inputs["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = vision_inputs["image_grid_thw"].to(device=device)
    n_image_tokens = post_merger_token_count(image_grid_thw, merge_size=2)

    with torch.no_grad():
        v_full = _encode_image_features(model, pixel_values, image_grid_thw)  # [Nimg, D]

    # --- 2. ROI indices ---
    roi_idx = select_roi_indices(image_grid_thw, bbox, K=K, merge_size=2)
    v_roi = v_full[torch.tensor(roi_idx, device=device, dtype=torch.long)]  # [K, D]

    # --- 3. Tokenize ---
    prefix, full = _build_prompt(question, answer, K, n_image_tokens)
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    if full_ids.shape[1] < prefix_ids.shape[1] or not torch.equal(
        full_ids[0, : prefix_ids.shape[1]], prefix_ids[0]
    ):
        ans_ids = tokenizer(" " + answer + "<|im_end|>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        full_ids = torch.cat([prefix_ids, ans_ids], dim=1)

    pad_positions = (full_ids[0] == latent_pad_id).nonzero(as_tuple=False).squeeze(-1)
    if pad_positions.numel() < K:
        raise RuntimeError(f"Expected ≥{K} latent-pad positions, got {pad_positions.numel()}")
    slot_positions = pad_positions[:K]

    img_pad_positions = (full_ids[0] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
    if img_pad_positions.numel() != n_image_tokens:
        raise RuntimeError(f"Expected {n_image_tokens} image_pad positions, got {img_pad_positions.numel()}")

    L = full_ids.shape[1]
    labels = torch.full((1, L), -100, dtype=torch.long, device=device)
    ans_start = prefix_ids.shape[1]
    labels[0, ans_start:] = full_ids[0, ans_start:]

    attention_mask = torch.ones_like(full_ids)

    alignment_poss = [slot_positions.tolist()]

    # --- 3.5. Build 4D attention mask (Phase 1.5b key change) ---
    attn_mask_4d = build_monet_4d_attn(
        full_ids,
        latent_token_id=latent_pad_id,
        pad_mask=attention_mask,
        dtype=dtype,
        mask_latent=False,
        latent_cross_isolate=True,
    )

    # --- 4. Latent forward (latent_mode=True) ---
    model.gradient_checkpointing_disable()

    latent_outputs = model(
        latent_mode=True,
        input_ids=full_ids,
        attention_mask=attention_mask,
        attention_mask_4d=attn_mask_4d,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        labels=None,
        alignment_poss=alignment_poss,
        loss_type=[],
        output_hidden_states=True,
        return_dict=True,
    )

    if latent_outputs.hidden_states is None:
        raise RuntimeError("latent_mode forward did not return hidden_states; check output_hidden_states arg")
    h_all_layers = latent_outputs.hidden_states[0]  # [num_layers, K, D]
    h = h_all_layers[-1]  # [K, D] last layer
    if h.shape[0] != K:
        raise RuntimeError(f"h shape={tuple(h.shape)} expected K={K}")
    if not h.requires_grad:
        raise RuntimeError("h does not require grad — gradient flow broken")

    ce_patch_pos = latent_outputs.ce_patch_pos
    ce_patch_vec = latent_outputs.ce_patch_vec

    # --- 5. CE forward (latent_mode=False) ---
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    ce_outputs = model(
        latent_mode=False,
        input_ids=full_ids,
        attention_mask=attention_mask,
        attention_mask_4d=attn_mask_4d,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        labels=labels,
        ce_patch_pos=ce_patch_pos,
        ce_patch_vec=ce_patch_vec,
        loss_type=["ce"],
        return_dict=True,
        use_cache=False,
    )
    ntp_loss = ce_outputs.loss

    lvr = lvr_mse(h.unsqueeze(0).float(), v_roi.unsqueeze(0).float())
    total = ntp_loss + lambda_lvr * lvr

    return {
        "total": total,
        "ntp": ntp_loss.detach(),
        "lvr": lvr.detach(),
        "h_norm": h.detach().float().norm(dim=-1).mean(),
        "v_norm": v_roi.detach().float().norm(dim=-1).mean(),
        "n_image_tokens": int(n_image_tokens),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--smoke", action="store_true", help="N-step gradient probe; default 10")
    ap.add_argument("--smoke_steps", type=int, default=10)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"

    _set_seed(cfg.get("seed", 0))

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"[config] {cfg}", flush=True)

    base = cfg["base_model"]
    print(f"[load] {base} (Monet-vendored class) — Phase 1.5b WITH attention_mask_4d", flush=True)
    processor = _build_processor(base)
    if hasattr(processor, "image_processor"):
        if cfg.get("max_pixels"):
            processor.image_processor.max_pixels = int(cfg["max_pixels"])
        if cfg.get("min_pixels"):
            processor.image_processor.min_pixels = int(cfg["min_pixels"])

    tokenizer = processor.tokenizer
    latent_start_id = int(tokenizer("<abs_vis_token>", return_tensors="pt")["input_ids"][0, 0])
    latent_end_id = int(tokenizer("</abs_vis_token>", return_tensors="pt")["input_ids"][0, 0])
    latent_pad_id = int(tokenizer("<abs_vis_token_pad>", return_tensors="pt")["input_ids"][0, 0])
    answer_start_pattern = tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    print(f"[tok] latent_start={latent_start_id} latent_end={latent_end_id} latent_pad={latent_pad_id}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, torch_dtype=dtype)
    new_vocab = len(tokenizer)
    model.resize_token_embeddings(new_vocab)
    model.config.vocab_size = new_vocab
    model.config.latent_token_id = latent_pad_id
    model.config.latent_start_id = latent_start_id
    model.config.latent_end_id = latent_end_id
    model.config.answer_start_pattern = answer_start_pattern.tolist()
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
    n_steps = int(cfg["max_steps"]) if not args.smoke else int(args.smoke_steps)
    warmup = int(cfg.get("warmup_steps", 100))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=n_steps)

    n_train_examples = int(cfg.get("n_train_examples", 5000)) if not args.smoke else max(4, int(args.smoke_steps) * int(cfg.get("grad_accum_steps", 4)))
    train_split = cfg.get("train_split", "train")
    seed = int(cfg.get("seed", 0))
    print(f"[data] split={train_split} n={n_train_examples}", flush=True)
    train_ds = load_phase1_dataset(split_name=train_split, n=n_train_examples, seed=seed)
    print(f"[data] loaded {len(train_ds)} examples", flush=True)

    K = int(cfg.get("K", 8))
    lambda_lvr = float(cfg.get("lambda_lvr", 1.0))
    grad_accum = int(cfg.get("grad_accum_steps", 4))
    log_every = int(cfg.get("log_every", 10))
    save_every = int(cfg.get("save_every", 0))

    model.train()
    optim.zero_grad(set_to_none=True)
    step = 0
    micro = 0
    t0 = time.time()
    losses_micro = {"total": [], "ntp": [], "lvr": [], "h_norm": [], "v_norm": []}
    n_examples = len(train_ds)
    epoch_idx = 0

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
                stats = lvr_step_monet(
                    model, processor, tokenizer, image_token_id, latent_pad_id,
                    answer_start_pattern,
                    image, question, answer, bbox,
                    K=K, lambda_lvr=lambda_lvr, device=device, dtype=dtype,
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
                v = stats[k]
                losses_micro[k].append(float(v.detach().cpu().item()) if hasattr(v, "detach") else float(v))
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
                    avg = {k: (sum(v)/len(v)) if v else float("nan") for k, v in losses_micro.items()}
                    row = {
                        "step": step,
                        "epoch": epoch_idx,
                        "lr": sched.get_last_lr()[0],
                        "total_loss": avg["total"],
                        "ntp_loss": avg["ntp"],
                        "lvr_loss": avg["lvr"],
                        "h_norm": avg["h_norm"],
                        "v_norm": avg["v_norm"],
                        "gpu_peak_gb": mem_gb,
                        "elapsed_s": time.time() - t0,
                    }
                    print(f"[step {step}/{n_steps}] loss={row['total_loss']:.3f} ntp={row['ntp_loss']:.3f} "
                          f"lvr={row['lvr_loss']:.3f} ||h||={row['h_norm']:.2f} ||v||={row['v_norm']:.2f} "
                          f"mem={row['gpu_peak_gb']:.1f}GB elapsed={row['elapsed_s']:.0f}s", flush=True)
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
