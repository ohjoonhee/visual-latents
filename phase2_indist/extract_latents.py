"""Phase 2 in-distribution latent extraction.

Re-extracts latents from the Phase 2 checkpoint
(`phase2_monet_stage2/results/run_p2/checkpoint`) on the SAME data distribution
that was used for training (Monet's `Visual_CoT/train.json` with `<observation>`
tags + auxiliary images), but on a held-out 200-example slice that is
guaranteed NOT to overlap with the 5K training subset.

Why
---
The original Phase 2 ablation extracted latents on `ohjoonhee/visual-cot-50k-poc`
eval split — a different distribution (no `<observation>` tags, no aux images
in the prompt). The harmful `qwen_base utility = -4.76` could be either:
  (a) genuine collapse, OR
  (b) train/eval distribution mismatch.
This script re-extracts on the IN-DIST held-out so we can disambiguate.

Held-out selection rule
-----------------------
- The Phase 2 trainer used precompute_teacher_reps.py with `start_offset_from_end=10000`
  and `seed=0`; it took the LAST 10K rows of `train.json`, shuffled them with
  `random.Random(0)`, and accepted the first 5000 valid samples. Their
  `metadata.sample_id`s are in `phase2_monet_stage2/teacher_reps/manifest.json`.
- We load that manifest, build the trained-id set, then take the same
  `raw[-10000:]` candidates, EXCLUDE any whose sample_id is in the trained set,
  shuffle the remainder with `random.Random(42)`, and accept the first 200
  valid via Monet's preprocess. This guarantees zero overlap.

Prompt format
-------------
EXACTLY the same construction Phase 2's trainer used (see `phase2_monet_stage2/trainer.py`
lines 278-281):
  texts = [processor.apply_chat_template(example, tokenize=False)]
  texts = [replace_latent_placeholder_with_img_pad(t) for t in texts]
  texts = add_latent_pad_after_auxiliary_img(texts, K=8, "<abs_vis_token_pad>")

Latent extraction
-----------------
- `latent_mode=True`, `output_latent_embeds=True`.
- `alignment_poss = find_ids_poss(input_ids, "<|im_start|>assistant", "<abs_vis_token_pad>")[0]`
  — the K positions of `<abs_vis_token_pad>` AFTER the first assistant marker.
  This matches `phase0_monet_probe/extract_latents.py` and the Monet upstream
  precompute_teacher_latents path.
- We save the FIRST K=8 positions per example (matching the Phase 2 OOD extract
  and the canonical ablation's `h_cache[:K]` convention).

Output
------
`phase2_indist/latents/latent_<sid>.pt` with keys:
  - latent  : Tensor [num_latent_pads_total, H]  (last layer at <abs_vis_token_pad>)
  - v_roi   : Tensor [K, D] | None  — 8 spatially-distributed image features from
              the FIRST image in the trace (auxiliary or primary), to support
              v_roi cosine diagnostics in the ablation. (None if no images.)
  - question, answer, sample_id, subset, K, num_latents
"""
from __future__ import annotations

# ===== monkey-patch BEFORE any transformers import =====
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASE0 = ROOT.parent / "phase0_monet_probe"
PHASE2 = ROOT.parent / "phase2_monet_stage2"

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
import copy
import json
import random
import time

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

# Reuse Phase 0 helpers — same prompt construction the Phase 2 trainer uses.
sys.path.insert(0, str(PHASE0))
from monet_utils import (  # noqa: E402
    Monet_single_input_images_preprocess_function,
    add_latent_pad_after_auxiliary_img,
    find_ids_poss,
    replace_latent_placeholder_with_img_pad,
    resize_by_token_budget,
)

LATENT_SIZE = int(os.environ["LATENT_SIZE"])  # 8


def _build_processor(model_path: Path) -> AutoProcessor:
    processor = AutoProcessor.from_pretrained(str(model_path), use_fast=True, trust_remote_code=True)
    for tok in [
        "<abs_vis_token_pad>", "<abs_vis_token>", "</abs_vis_token>",
        "<observation>", "</observation>",
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
    """Last assistant turn's last text content, with structural tags scrubbed."""
    last_text = ""
    for step in sample["data"]:
        if step["role"] == "assistant":
            for c in step["content"]:
                if c.get("type") == "text":
                    last_text = c["text"]
    return (
        last_text
        .replace("<abs_vis_token></abs_vis_token>", "")
        .replace("<observation>", "")
        .replace("</observation>", "")
        .strip()
    )


def _spread_indices(n_tokens: int, K: int) -> list[int]:
    """K evenly-spaced indices over [0, n_tokens)."""
    if n_tokens <= 0:
        return []
    if n_tokens <= K:
        return list(range(n_tokens))
    return [int(i * n_tokens / K) for i in range(K)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to Phase 2 checkpoint (e.g., phase2_monet_stage2/results/run_p2/checkpoint)")
    ap.add_argument("--data_path", type=str,
                    default=str(PHASE0 / "data" / "Monet-SFT-125K" / "Visual_CoT" / "train.json"))
    ap.add_argument("--dataset_root", type=str,
                    default=str(PHASE0 / "data" / "Monet-SFT-125K"))
    ap.add_argument("--manifest", type=str,
                    default=str(PHASE2 / "teacher_reps" / "manifest.json"))
    ap.add_argument("--subset", type=str, default="Visual_CoT")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start_offset_from_end", type=int, default=10000,
                    help="Same as Phase 2's training pool: search the last 10K rows.")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max_pixels", type=int, default=1000 * 28 * 28)
    ap.add_argument("--per_img_pixels", type=int, default=500 * 28 * 28)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract-indist] ckpt={args.ckpt}", flush=True)
    print(f"[extract-indist] data={args.data_path}", flush=True)
    print(f"[extract-indist] manifest={args.manifest}", flush=True)
    print(f"[extract-indist] n={args.n}  seed={args.seed}", flush=True)

    # --- load processor + model ---
    processor = _build_processor(Path(args.ckpt))
    tok = processor.tokenizer
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = args.max_pixels

    latent_pad_id = int(tok("<abs_vis_token_pad>", return_tensors="pt")["input_ids"][0, 0])
    answer_start_pattern = tok("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    print("[extract-indist] loading model (bf16, cuda)", flush=True)
    config = Qwen2_5_VLConfig.from_pretrained(args.ckpt)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt, config=config, torch_dtype=torch.bfloat16,
    )
    new_vocab = len(tok)
    model.resize_token_embeddings(new_vocab)
    model.config.vocab_size = new_vocab
    model.config.latent_token_id = latent_pad_id
    model.config.answer_start_pattern = answer_start_pattern.tolist()
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval().cuda()
    image_token_id = model.config.image_token_id

    # --- load + filter dataset ---
    raw = json.load(open(args.data_path))
    print(f"[extract-indist] {args.subset}/train.json has {len(raw)} rows", flush=True)
    manifest = json.load(open(args.manifest))
    trained_ids = {int(m["sample_id"]) for m in manifest if m["dataset_name"] == args.subset}
    print(f"[extract-indist] manifest trained sample_ids: {len(trained_ids)}", flush=True)

    # Same pool as Phase 2 training: last `start_offset_from_end` rows.
    pool = raw[-args.start_offset_from_end:] if len(raw) > args.start_offset_from_end else raw
    # Exclude trained ids.
    eligible = [r for r in pool if int(r.get("metadata", {}).get("sample_id", -1)) not in trained_ids]
    print(f"[extract-indist] pool={len(pool)}  eligible (after exclusion)={len(eligible)}", flush=True)
    if len(eligible) < args.n:
        print(f"WARN: only {len(eligible)} eligible; will accept fewer than n={args.n}", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(eligible)

    accepted: list[tuple[int, dict]] = []
    skipped = 0
    for sample in eligible:
        sid_meta = int(sample.get("metadata", {}).get("sample_id", -1))
        # Defensive: assert no overlap.
        assert sid_meta not in trained_ids, f"BUG: sid {sid_meta} in trained set"
        try:
            processed = Monet_single_input_images_preprocess_function(
                copy.deepcopy(sample),
                dataset_root=args.dataset_root,
                allow_no_observation=False,  # match Phase 2 trainer exactly (see line 233 of trainer.py)
            )
        except Exception:
            skipped += 1
            continue
        if processed is None:
            skipped += 1
            continue
        accepted.append((sid_meta, processed))
        if len(accepted) >= args.n:
            break

    print(f"[extract-indist] accepted={len(accepted)}  skipped={skipped}", flush=True)
    if not accepted:
        print("FATAL: zero accepted", flush=True)
        sys.exit(2)

    # --- iterate ---
    n_done = 0
    n_failed = 0
    started = time.time()

    with torch.inference_mode():
        for sid, processed in accepted:
            try:
                example = processed["data"]
                # EXACT prompt construction from Phase 2 trainer (and Phase 0 extract).
                texts = [processor.apply_chat_template(example, tokenize=False)]
                texts = [replace_latent_placeholder_with_img_pad(t) for t in texts]
                texts = add_latent_pad_after_auxiliary_img(texts, args.K, "<abs_vis_token_pad>")
                image_inputs, _ = process_vision_info([example])
                if image_inputs:
                    image_inputs, _ = resize_by_token_budget(
                        image_inputs,
                        global_max_pixels=args.max_pixels,
                        per_img_max_pixels=args.per_img_pixels,
                    )
                total_image_pads = sum(t.count("<|vision_start|><|image_pad|>") for t in texts)
                if total_image_pads != len(image_inputs):
                    print(f"  [skip sid={sid}] vision-pad/image mismatch", flush=True)
                    n_failed += 1
                    continue

                batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                pixel_values = batch["pixel_values"]
                image_grid_thw = batch["image_grid_thw"]

                latent_pad_pattern = torch.tensor([latent_pad_id], dtype=torch.long)
                alignment_poss = find_ids_poss(input_ids, answer_start_pattern, latent_pad_pattern)
                if not alignment_poss[0]:
                    print(f"  [skip sid={sid}] no latent_pad positions found", flush=True)
                    n_failed += 1
                    continue

                # Latent forward.
                inputs = dict(
                    latent_mode=True,
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    pixel_values=pixel_values.cuda(),
                    image_grid_thw=image_grid_thw.cuda(),
                    labels=None,
                    loss_type=[],
                    alignment_poss=alignment_poss,
                    output_latent_embeds=True,
                )
                out = model(**inputs, return_dict=True)
                latent = out.latent_embeds[0].detach().cpu()  # [num_latents, H]

                if latent.shape[0] < args.K:
                    print(f"  [skip sid={sid}] only {latent.shape[0]} latent positions < K={args.K}", flush=True)
                    n_failed += 1
                    continue

                # v_roi: 8 spatially-distributed features from the FIRST image (aux or primary).
                # Matches Phase 2 OOD extract's intent (parallel `v_roi` field for cosine diagnostics).
                v_roi = None
                try:
                    if image_grid_thw is not None and image_grid_thw.numel() > 0:
                        v_full = torch.cat(
                            model.model.get_image_features(
                                pixel_values.cuda().to(torch.bfloat16),
                                image_grid_thw.cuda(),
                            ),
                            dim=0,
                        )  # [N_total_post_merge, D]
                        # Use the FIRST image's tokens. Compute its post-merger token count.
                        first_thw = image_grid_thw[0].tolist()
                        merge = 2
                        first_n = (first_thw[0]) * (first_thw[1] // merge) * (first_thw[2] // merge)
                        first_tokens = v_full[:first_n]
                        idxs = _spread_indices(first_tokens.shape[0], args.K)
                        v_roi = first_tokens[torch.tensor(idxs, device=v_full.device, dtype=torch.long)].detach().cpu()
                except Exception as e:
                    print(f"  [v_roi err sid={sid}] {type(e).__name__}: {e}", flush=True)

                save_path = out_dir / f"latent_{sid:08d}.pt"
                torch.save(
                    {
                        "latent": latent,
                        "v_roi": v_roi,
                        "question": _extract_question(processed),
                        "answer": _extract_answer(processed),
                        "sample_id": sid,
                        "subset": args.subset,
                        "K": args.K,
                        "num_latents": int(latent.shape[0]),
                        "latent_size_per_image": LATENT_SIZE,
                    },
                    save_path,
                )
                n_done += 1
                if n_done % 10 == 0 or n_done == 1:
                    dt = time.time() - started
                    print(f"  [done {n_done}/{len(accepted)}] sid={sid} latent={tuple(latent.shape)} elapsed={dt:.1f}s", flush=True)
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
    print(f"[extract-indist] done. saved={n_done}  failed={n_failed}  elapsed={dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
