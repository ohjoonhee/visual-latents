"""Phase 2 — Faithful Monet stage 2 scale-down on Monet's Visual_CoT data.

Recipe (verbatim from script_examples/sft_stage2.sh except scale):
- backbone: raw Qwen/Qwen2.5-VL-3B-Instruct (Monet vendored class via sys.modules);
            skip Stage 1 SFT (deviation, documented).
- data: Visual_CoT 5K subset (same metadata IDs as teacher precompute manifest).
- LATENT_SIZE=8, ce_emphasize_factor=4.0, alignment_weight=2.0,
  emphasize_latent_weight=2.0, alignment_layer=all_layers.
- 1000 optimizer steps, eff_batch=16 (bsz=1 × grad_accum=16), lr=1e-5, bf16.
- AdamW + accelerate single-GPU (no DeepSpeed).
- gradient_checkpointing for non-latent forwards (matches upstream trainer.py:168).
- Loss: L = ce_emphasize * weighted_CE + alignment_weight * align_loss
  with optional latent-only backprop variant when emphasize_latent_weight != 0.

Implementation strategy: PATH B (custom trainer reusing Monet's vendored model).
- precompute_teacher_reps.py caches teacher hidden-states at observation positions.
- This trainer:
  1. latent forward (latent_mode=True, alignment_poss=obs, teacher_reps=cached)
     → outputs.alignment_loss, outputs.ce_patch_pos/vec.
  2. CE forward (latent_mode=False, ce_patch_pos/vec spliced, ce_emphasize_poss=obs)
     → outputs.loss (weighted CE).
  3. total = ce_loss + alignment_weight * alignment_loss
     (we omit the emphasize_latent_weight latent-only backprop trick — it's an
      optimization for backprop locality; we approximate by combining losses
      directly. Documented as a deviation.)
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

assert "transformers" not in sys.modules

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
import copy
import json
import random
import time

import torch
import yaml
from qwen_vl_utils import process_vision_info
from torch.optim import AdamW
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup

# Reuse Phase 0 helpers.
sys.path.insert(0, str(PHASE0))
from monet_utils import (  # noqa: E402
    Monet_single_input_images_preprocess_function,
    add_latent_pad_after_auxiliary_img,
    find_ids_poss,
    replace_latent_placeholder_with_img_pad,
    resize_by_token_budget,
)


LATENT_SIZE = int(os.environ["LATENT_SIZE"])  # 8


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


def _generate_labels(input_ids, answer_start_pattern, ignore_ids):
    """Mimic upstream `generate_labels_after_multi_token_start`."""
    B, L = input_ids.shape
    labels = torch.full_like(input_ids, -100)

    pat = answer_start_pattern
    pat_len = pat.numel()
    for b in range(B):
        # find first occurrence of answer_start_pattern in this row
        ids = input_ids[b]
        ans_start_pos = -1
        for s in range(L - pat_len + 1):
            if torch.equal(ids[s : s + pat_len].cpu(), pat):
                ans_start_pos = s + pat_len
                break
        if ans_start_pos < 0:
            continue
        labels[b, ans_start_pos:] = ids[ans_start_pos:]
        # zero out ignore ids
        for ig in ignore_ids:
            labels[b, ans_start_pos:][ids[ans_start_pos:] == ig] = -100
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"

    _set_seed(int(cfg.get("seed", 0)))

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"[config] {cfg}", flush=True)

    base = cfg["base_model"]
    print(f"[load] {base} (Monet vendored)", flush=True)
    processor = _build_processor(base)
    if hasattr(processor, "image_processor"):
        if cfg.get("max_pixels"):
            processor.image_processor.max_pixels = int(cfg["max_pixels"])
        if cfg.get("min_pixels"):
            processor.image_processor.min_pixels = int(cfg["min_pixels"])
    tokenizer = processor.tokenizer

    def _id(s):
        return int(tokenizer(s, return_tensors="pt")["input_ids"][0, 0])

    special_ids = {
        "abs_pad": _id("<abs_vis_token_pad>"),
        "abs_start": _id("<abs_vis_token>"),
        "abs_end": _id("</abs_vis_token>"),
        "obs_start": _id("<observation>"),
        "obs_end": _id("</observation>"),
        "v_start": _id("<|vision_start|>"),
        "v_end": _id("<|vision_end|>"),
        "img_pad": _id("<|image_pad|>"),
        "end_pad": _id("<|endoftext|>"),
        "ans_start": tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0],
    }
    print(f"[tok] specials: {[(k,(v if hasattr(v,'shape') else v)) for k,v in special_ids.items()]}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, torch_dtype=dtype)
    new_vocab = len(tokenizer)
    model.resize_token_embeddings(new_vocab)
    model.config.vocab_size = new_vocab
    model.config.latent_token_id = special_ids["abs_pad"]
    model.config.latent_start_id = special_ids["abs_start"]
    model.config.latent_end_id = special_ids["abs_end"]
    model.config.answer_start_pattern = special_ids["ans_start"].tolist()
    model.to(device)

    # Freeze vision tower + projector.
    for n, p in model.named_parameters():
        if n.startswith("model.visual") or n.startswith("visual"):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[freeze] trainable={n_train/1e6:.1f}M", flush=True)

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        betas=(0.9, 0.95),
    )
    n_steps = int(cfg["max_steps"]) if not args.smoke else 10
    warmup = int(cfg.get("warmup_steps", 10))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=n_steps)

    # Load Phase 2 dataset using the manifest produced by precompute_teacher_reps.py.
    teacher_reps_dir = Path(cfg["teacher_reps_dir"])
    manifest_path = teacher_reps_dir / "manifest.json"
    manifest = json.load(open(manifest_path))
    print(f"[data] manifest entries: {len(manifest)}", flush=True)
    raw = json.load(open(cfg["data_path"]))
    raw_index = {(d.get("metadata", {}).get("dataset_name", "Visual_CoT"), int(d.get("metadata", {}).get("sample_id", -1))): d for d in raw}

    # We need to re-process samples (the processed structure includes resolved image paths).
    n_train_examples = int(cfg.get("n_train_examples", len(manifest))) if not args.smoke else 4
    used_samples = []
    for entry in manifest[: n_train_examples]:
        sample = raw_index.get((entry["dataset_name"], int(entry["sample_id"])))
        if sample is None:
            # Manifest sample_id is 0..len. Fallback: linear search.
            for s in raw:
                md = s.get("metadata", {})
                if md.get("dataset_name") == entry["dataset_name"] and int(md.get("sample_id", -1)) == int(entry["sample_id"]):
                    sample = s
                    break
        if sample is None:
            # Manifest may have used positional sample_ids that aren't in raw — this is OK for newly-injected metadata.
            # We need to find it differently. The manifest was built by precompute_teacher_reps which appended new metadata
            # with positional sample_ids. So we need the same sequence of accepted samples.
            print(f"  WARN: missing sample dataset={entry['dataset_name']} sid={entry['sample_id']}", flush=True)
            continue
        # Run preprocess to resolve paths + apply system prompt.
        try:
            processed = Monet_single_input_images_preprocess_function(
                copy.deepcopy({"data": sample["data"], "metadata": entry}),
                dataset_root=cfg["dataset_root"],
                allow_no_observation=False,
            )
        except Exception:
            continue
        if processed is None:
            continue
        used_samples.append(processed)
    print(f"[data] usable samples: {len(used_samples)}", flush=True)

    K = int(cfg["latent_size"])
    grad_accum = int(cfg.get("grad_accum_steps", 16))
    log_every = int(cfg.get("log_every", 10))
    save_every = int(cfg.get("save_every", 0))
    alignment_weight = float(cfg.get("alignment_weight", 2.0))

    cfg["latent_size"] = K
    cfg["alignment_layer"] = cfg.get("alignment_layer", "all_layers")
    cfg["global_max_pixels"] = int(cfg.get("global_max_pixels", 1000 * 28 * 28))
    cfg["per_img_pixels"] = int(cfg.get("per_img_pixels", 500 * 28 * 28))
    cfg["ce_emphasize_factor"] = float(cfg.get("ce_emphasize_factor", 4.0))

    model.train()
    optim.zero_grad(set_to_none=True)

    step = 0
    micro = 0
    t0 = time.time()
    losses_micro = {"total": [], "ce": [], "align": [], "obs_n": []}
    seed = int(cfg.get("seed", 0))
    n_examples = len(used_samples)
    epoch_idx = 0

    while step < n_steps:
        idx_perm = list(range(n_examples))
        random.Random(seed + epoch_idx).shuffle(idx_perm)
        for i in idx_perm:
            sample = used_samples[i]
            # Re-run latent_forward with loss_type=['alignment'] so loss_dict gets populated.
            try:
                # Inline the step here so we can pass loss_type=['alignment'] to lat_out.
                example = sample["data"]
                metadata = sample["metadata"]
                texts = [processor.apply_chat_template(example, tokenize=False)]
                texts = [replace_latent_placeholder_with_img_pad(t) for t in texts]
                texts = add_latent_pad_after_auxiliary_img(texts, K, "<abs_vis_token_pad>")
                image_inputs, _ = process_vision_info([example])
                if image_inputs:
                    image_inputs, _ = resize_by_token_budget(
                        image_inputs,
                        global_max_pixels=cfg["global_max_pixels"],
                        per_img_max_pixels=cfg["per_img_pixels"],
                    )
                total_pads = sum(t.count("<|vision_start|><|image_pad|>") for t in texts)
                if total_pads != len(image_inputs):
                    continue

                batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                pixel_values = batch["pixel_values"]
                image_grid_thw = batch["image_grid_thw"]

                obs_start_poss = find_ids_poss(input_ids, special_ids["ans_start"], torch.tensor([special_ids["obs_start"]], dtype=torch.long))
                obs_end_poss = find_ids_poss(input_ids, special_ids["ans_start"], torch.tensor([special_ids["obs_end"]], dtype=torch.long))
                if not obs_start_poss[0] or not obs_end_poss[0]:
                    continue
                if len(obs_start_poss[0]) != len(obs_end_poss[0]):
                    continue
                obs_poss = []
                for s, e in zip(obs_start_poss[0], obs_end_poss[0]):
                    obs_poss.extend(list(range(s, e)))
                if not obs_poss:
                    continue

                md_info = f"{cfg['alignment_layer']}_{metadata['dataset_name']}_{int(metadata['sample_id'])}"
                teacher_path = teacher_reps_dir / f"rep_{md_info}.pt"
                if not teacher_path.exists():
                    continue
                teacher_data = torch.load(teacher_path, map_location="cpu", weights_only=False)
                teacher_rep = teacher_data["latent"]  # [L, num_obs, H]
                if teacher_rep.shape[1] != len(obs_poss):
                    continue
                teacher_rep = teacher_rep.to(device=device, dtype=dtype)

                # Labels (ignore special tokens).
                ignore_ids = [special_ids[k] for k in ["abs_pad", "abs_end", "img_pad", "v_start", "v_end", "obs_start", "obs_end", "end_pad"]]
                ignore_ids = [int(x) for x in ignore_ids]
                labels = _generate_labels(input_ids, special_ids["ans_start"], ignore_ids).to(device)

                # Latent forward (alignment_poss = obs positions; pass loss_type=['alignment'] so loss_dict is populated).
                # The model's latent_mode branch uses loss_type internally only for non-latent path; for
                # latent_mode the alignment_loss is attached to outputs.alignment_loss (Qwen2_5_VLModelOutputWithPast).
                # The wrapper Qwen2_5_VLForConditionalGeneration.forward populates loss_dict['alignment']
                # from `outputs.alignment_loss` when 'alignment' in loss_type — which it does at line 2329-2330.
                model.gradient_checkpointing_disable()
                lat_out = model(
                    latent_mode=True,
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    pixel_values=pixel_values.to(device, dtype=dtype),
                    image_grid_thw=image_grid_thw.to(device),
                    labels=None,
                    alignment_poss=[obs_poss],
                    teacher_hidden_states_for_alignment=[teacher_rep],
                    loss_type=["alignment"],
                    return_dict=True,
                )
                if lat_out.loss_dict is None or "alignment" not in lat_out.loss_dict:
                    # Cannot compute alignment loss; skip.
                    continue
                align_loss = lat_out.loss_dict["alignment"]

                # CE forward.
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                ce_out = model(
                    latent_mode=False,
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    pixel_values=pixel_values.to(device, dtype=dtype),
                    image_grid_thw=image_grid_thw.to(device),
                    labels=labels,
                    ce_patch_pos=lat_out.ce_patch_pos,
                    ce_patch_vec=lat_out.ce_patch_vec,
                    ce_emphasize_poss=[obs_poss],
                    ce_emphasize_factor=cfg["ce_emphasize_factor"],
                    loss_type=["ce"],
                    return_dict=True,
                    use_cache=False,
                )
                ce_loss = ce_out.loss

                total_loss = ce_loss + alignment_weight * align_loss
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                print(f"  [OOM ex={i}] {e}", flush=True)
                continue
            except Exception as e:
                print(f"  [error ex={i}] {type(e).__name__}: {e}", flush=True)
                import traceback; traceback.print_exc()
                continue

            (total_loss / grad_accum).backward()
            losses_micro["total"].append(float(total_loss.detach().cpu().item()))
            losses_micro["ce"].append(float(ce_loss.detach().cpu().item()))
            losses_micro["align"].append(float(align_loss.detach().cpu().item()))
            losses_micro["obs_n"].append(float(len(obs_poss)))
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
                        "ce_loss": avg["ce"],
                        "align_loss": avg["align"],
                        "obs_n": avg["obs_n"],
                        "gpu_peak_gb": mem_gb,
                        "elapsed_s": time.time() - t0,
                    }
                    print(f"[step {step}/{n_steps}] tot={row['total_loss']:.3f} ce={row['ce_loss']:.3f} "
                          f"align={row['align_loss']:.4f} obs_n={row['obs_n']:.1f} "
                          f"mem={row['gpu_peak_gb']:.1f}GB elapsed={row['elapsed_s']:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    losses_micro = {k: [] for k in losses_micro}

                if save_every and step > 0 and step % save_every == 0 and not args.smoke:
                    ck = out_dir / f"checkpoint_step{step}"
                    ck.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ck)
                    processor.save_pretrained(ck)

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
