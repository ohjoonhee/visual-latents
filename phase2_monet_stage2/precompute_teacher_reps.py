"""Phase 2 — precompute teacher hidden states at observation token positions.

Mirrors upstream `precompute_teacher_reps.py` but:
- single GPU (no DDP);
- raw Qwen2.5-VL-3B-Instruct as teacher (skip Stage 1 SFT — deviation);
- 5K Visual_CoT subset (seed=0, last N rows of train.json).

Per sample, save `{'metadata_info': str, 'latent': [num_layers, num_obs_tokens, D]}`
to `<out_dir>/rep_all_layers_<dataset_name>_<sample_id>.pt`.
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
import copy
import json
import random
import time

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Reuse Phase 0's vendored helpers.
sys.path.insert(0, str(PHASE0))
from monet_utils import (  # noqa: E402
    Monet_single_input_images_preprocess_function,
    find_ids_poss,
    replace_latent_placeholder_with_img_pad,
    resize_by_token_budget,
)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data_path", type=str, default=str(PHASE0 / "data" / "Monet-SFT-125K" / "Visual_CoT" / "train.json"))
    ap.add_argument("--dataset_name", type=str, default="Visual_CoT")
    ap.add_argument("--dataset_root", type=str, default=str(PHASE0 / "data" / "Monet-SFT-125K"))
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--start_offset_from_end", type=int, default=10000,
                    help="Sample from the LAST N rows to reduce Phase 0 overlap.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--global_max_pixels", type=int, default=1000 * 28 * 28)
    ap.add_argument("--per_img_pixels", type=int, default=500 * 28 * 28)
    ap.add_argument("--alignment_layer", choices=["all_layers", "last_layer"], default="all_layers")
    ap.add_argument("--save_every", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[teacher-precompute] loading {args.base_model}", flush=True)
    processor = _build_processor(args.base_model)
    tok = processor.tokenizer
    latent_pad_id = int(tok("<abs_vis_token_pad>", return_tensors="pt")["input_ids"][0, 0])
    obs_start_id = int(tok("<observation>", return_tensors="pt")["input_ids"][0, 0])
    obs_end_id = int(tok("</observation>", return_tensors="pt")["input_ids"][0, 0])
    latent_start_id = int(tok("<abs_vis_token>", return_tensors="pt")["input_ids"][0, 0])
    latent_end_id = int(tok("</abs_vis_token>", return_tensors="pt")["input_ids"][0, 0])
    answer_start_pattern = tok("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
    )
    new_vocab = len(tok)
    model.resize_token_embeddings(new_vocab)
    model.config.vocab_size = new_vocab
    model.config.latent_token_id = latent_pad_id
    model.config.latent_start_id = latent_start_id
    model.config.latent_end_id = latent_end_id
    model.config.answer_start_pattern = answer_start_pattern.tolist()
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval().cuda()

    print(f"[teacher-precompute] loading {args.data_path}", flush=True)
    raw = json.load(open(args.data_path))
    print(f"[teacher-precompute] {len(raw)} rows; sampling last {args.start_offset_from_end} rows, n={args.n}", flush=True)

    # Take LAST start_offset_from_end rows; shuffle deterministically.
    candidates = raw[-args.start_offset_from_end:] if len(raw) > args.start_offset_from_end else raw
    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    accepted = []
    skipped = 0
    seen_ids = set()
    for sample in candidates:
        if "metadata" not in sample:
            sample = dict(sample)
            sample["metadata"] = {
                "dataset_name": args.dataset_name,
                "sample_id": len(accepted),
            }
        sid = int(sample["metadata"]["sample_id"])
        if sid in seen_ids:
            continue
        seen_ids.add(sid)

        try:
            processed = Monet_single_input_images_preprocess_function(
                copy.deepcopy(sample), dataset_root=args.dataset_root, allow_no_observation=False,
            )
        except Exception:
            skipped += 1
            continue
        if processed is None:
            skipped += 1
            continue
        accepted.append(processed)
        if len(accepted) >= args.n:
            break
    print(f"[teacher-precompute] accepted={len(accepted)}  skipped={skipped}", flush=True)
    if not accepted:
        print("FATAL: zero accepted", flush=True)
        sys.exit(2)

    # Save manifest of accepted IDs (for trainer to use the same ones).
    manifest_path = out_dir / "manifest.json"
    manifest = [{"dataset_name": p["metadata"]["dataset_name"], "sample_id": int(p["metadata"]["sample_id"])} for p in accepted]
    json.dump(manifest, open(manifest_path, "w"))
    print(f"[teacher-precompute] wrote {manifest_path}", flush=True)

    n_done = 0
    n_failed = 0
    started = time.time()
    with torch.inference_mode():
        for processed in accepted:
            md = processed["metadata"]
            ds_name = md["dataset_name"]
            sid = int(md["sample_id"])
            metadata_info = f"{args.alignment_layer}_{ds_name}_{sid}"
            save_path = out_dir / f"rep_{metadata_info}.pt"
            if save_path.exists():
                n_done += 1
                continue
            try:
                example = processed["data"]
                texts = [processor.apply_chat_template(example, tokenize=False)]
                texts = [replace_latent_placeholder_with_img_pad(t) for t in texts]
                # NOTE: Stage 2's TEACHER does NOT add <abs_vis_token_pad> slots.
                # The teacher just sees the full trace (with aux images visible).
                image_inputs, _ = process_vision_info([example])
                if image_inputs:
                    image_inputs, _ = resize_by_token_budget(
                        image_inputs,
                        global_max_pixels=args.global_max_pixels,
                        per_img_max_pixels=args.per_img_pixels,
                    )
                total_image_pads = sum(t.count("<|vision_start|><|image_pad|>") for t in texts)
                if total_image_pads != len(image_inputs):
                    n_failed += 1
                    continue

                batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                pixel_values = batch["pixel_values"]
                image_grid_thw = batch["image_grid_thw"]

                # Compute observation positions.
                obs_start_poss = find_ids_poss(input_ids, answer_start_pattern, torch.tensor([obs_start_id], dtype=torch.long))
                obs_end_poss = find_ids_poss(input_ids, answer_start_pattern, torch.tensor([obs_end_id], dtype=torch.long))
                if len(obs_start_poss[0]) == 0 or len(obs_end_poss[0]) == 0:
                    n_failed += 1
                    continue
                if len(obs_start_poss[0]) != len(obs_end_poss[0]):
                    n_failed += 1
                    continue
                obs_poss = []
                for s, e in zip(obs_start_poss[0], obs_end_poss[0]):
                    obs_poss.extend(list(range(s, e)))
                if not obs_poss:
                    n_failed += 1
                    continue

                inputs = dict(
                    latent_mode=False,
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    pixel_values=pixel_values.cuda(),
                    image_grid_thw=image_grid_thw.cuda(),
                    labels=None,
                    alignment_poss=[obs_poss],
                    loss_type=[],
                    output_hidden_states=True,
                )
                out = model(**inputs, return_dict=True)
                if out.hidden_states is None:
                    n_failed += 1
                    continue
                # In non-latent mode with `alignment_poss` set, the patched forward
                # returns hidden_states as a per-batch list, each element shape
                # [num_layers, num_align, hidden_dim] — already indexed at obs_poss.
                # See modeling_qwen2_5_vl_monet.py line 2033-2044.
                hs = out.hidden_states
                if isinstance(hs, list) and len(hs) > 0 and isinstance(hs[0], torch.Tensor):
                    teacher_rep = hs[0].detach().cpu()  # [num_layers, num_obs, H]
                    if teacher_rep.dim() != 3 or teacher_rep.shape[1] != len(obs_poss):
                        print(f"  [skip sid={sid}] hidden_states shape unexpected: {tuple(teacher_rep.shape)} vs num_obs={len(obs_poss)}", flush=True)
                        n_failed += 1
                        continue
                else:
                    n_failed += 1
                    continue
                if args.alignment_layer == "last_layer":
                    teacher_rep = teacher_rep[-1:, ...]  # [1, num_obs, H]

                torch.save(
                    {"metadata_info": metadata_info, "latent": teacher_rep},
                    save_path,
                )
                n_done += 1
                if n_done % 50 == 0 or n_done == 1:
                    dt = time.time() - started
                    rate = n_done / max(1.0, dt)
                    print(f"  [done {n_done}/{len(accepted)}] sid={sid} obs_n={len(obs_poss)} "
                          f"rep_shape={tuple(teacher_rep.shape)} elapsed={dt:.1f}s rate={rate:.2f}/s", flush=True)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                n_failed += 1
                print(f"  [OOM sid={sid}] {e}", flush=True)
            except Exception as e:
                import traceback
                n_failed += 1
                print(f"  [err sid={sid}] {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()

    dt = time.time() - started
    print(f"[teacher-precompute] done: saved={n_done} failed={n_failed} elapsed={dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
