#!/usr/bin/env python
"""Precompute Stage-2 teacher latents for Monet Stage 3 training.

Mirrors upstream `src/precompute_teacher_latents.py`. For every example in
Monet-SFT-125K (eval-200 excluded), run the teacher (= released
Monet-SFT-7B Stage 2 ckpt) in latent_mode to generate its K latent
embeddings, capture the per-layer hidden states at the latent positions,
and dump to disk as:

    {out_dir}/latent_{alignment_layer}_{dataset_name}_{sample_id}.pt
        → {"latent": Tensor[L_layers, K_total, H]}   (all_layers)
        → {"latent": Tensor[K_total, H]}             (last_layer)

`load_offline_tensor` in `cluster/trainer_monet_stage3.py` reads these by
exactly this naming + this key.

USAGE (run on cluster after Monet-SFT-7B is downloaded)
=======================================================
  # Sanity-check first
  uv --project cluster run python scripts/cluster/precompute_teacher_latents.py \
      --teacher_ckpt /data/joonhee/visual-latents/cluster_phase3/monet_sft_7b/stage2 \
      --out_dir      /data/joonhee/visual-latents/cluster_phase3/teacher_latents \
      --alignment_layer all_layers \
      --K 8 \
      --n_max 8 \
      --skip_existing

  # Full pass (~6h on 4×H100 at bsz=1, single-rank inference-mode)
  uv --project cluster run python scripts/cluster/precompute_teacher_latents.py \
      --teacher_ckpt /data/joonhee/visual-latents/cluster_phase3/monet_sft_7b/stage2 \
      --out_dir      /data/joonhee/visual-latents/cluster_phase3/teacher_latents \
      --alignment_layer all_layers \
      --K 8 \
      --skip_existing

NOTES
=====
- Single-rank, inference-mode, no grad. Multi-rank would parallelize but
  adds complexity; 6h is acceptable. If needed, shard by `--rank/--world`.
- `all_layers` mode: cache shape `[L_layers, K_total, H]` per sample.
  At L=28 layers, K_total≈8, H=3584, bf16 → ~1.6 MB/sample × 118K ≈ 190 GB.
  (Smaller than my earlier 480 GB worst-case — that was an overestimate
  including grad-checkpoint state.)
- Student-style input: aux images STRIPPED, latent blocks inserted. The
  teacher (Stage 2 ckpt) hasn't seen this exact format at training time,
  but the latent forward + 4D mask handle it correctly per upstream's
  precompute script.
"""
from __future__ import annotations

# ===== monkey-patch BEFORE any transformers import =====
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
CLUSTER = ROOT / "cluster"
PHASE0 = ROOT / "phase0_monet_probe"

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
import time

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(CLUSTER))
from data_utils import add_monet_special_tokens, load_monet_sft_125k  # noqa: E402
from mask_utils import build_monet_4d_attn  # noqa: E402

sys.path.insert(0, str(PHASE0))
from monet_utils import (  # noqa: E402
    find_ids_poss,
    replace_latent_placeholder_with_img_pad,
    resize_by_token_budget,
)

# Import the Stage 3 input builder (the same one the trainer uses).
sys.path.insert(0, str(CLUSTER))
from trainer_monet_stage3 import (  # noqa: E402
    _build_step_inputs_stage3,
    _build_processor,
)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", required=True,
                    help="Released Monet-SFT-7B Stage-2 checkpoint dir.")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for .pt files.")
    ap.add_argument("--subset", default="Visual_CoT")
    ap.add_argument("--K", type=int, default=8, help="Latent slots per aux image.")
    ap.add_argument("--alignment_layer", choices=["all_layers", "last_layer"],
                    default="all_layers")
    ap.add_argument("--n_max", type=int, default=None,
                    help="Limit number of samples (debug/smoke).")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip samples whose .pt already exists.")
    ap.add_argument("--global_max_pixels", type=int, default=2000 * 28 * 28)
    ap.add_argument("--per_img_pixels", type=int, default=1280 * 28 * 28)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[precompute] teacher_ckpt={args.teacher_ckpt}")
    print(f"[precompute] out_dir={out_dir}")
    print(f"[precompute] K={args.K}  alignment_layer={args.alignment_layer}")
    print(f"[precompute] rank={args.rank}/{args.world}")

    processor = _build_processor(args.teacher_ckpt, max_pixels=None, min_pixels=None)
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

    teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.teacher_ckpt, torch_dtype=torch.bfloat16,
    )
    teacher.resize_token_embeddings(len(tokenizer))
    teacher.config.vocab_size = len(tokenizer)
    teacher.config.latent_token_id = special_ids["abs_pad"]
    teacher.config.latent_start_id = special_ids["abs_start"]
    teacher.config.latent_end_id = special_ids["abs_end"]
    teacher.config.answer_start_pattern = special_ids["ans_start"].tolist()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval().cuda()

    train_ds, _eval = load_monet_sft_125k(
        subset=args.subset, n=args.n_max,
        eval_holdout=200,
        allow_no_observation_for_train=False,
    )
    if args.n_max is not None:
        train_ds = train_ds[: args.n_max]
    print(f"[precompute] n_samples={len(train_ds)}")

    t0 = time.time()
    n_done = 0
    n_skip = 0
    n_err = 0
    for i, sample in enumerate(train_ds):
        if i % args.world != args.rank:
            continue
        metadata = sample["metadata"]
        ds_name = metadata["dataset_name"]
        sample_id = int(metadata["sample_id"])
        fname = f"latent_{args.alignment_layer}_{ds_name}_{sample_id}.pt"
        out_path = out_dir / fname

        if args.skip_existing and out_path.exists():
            n_skip += 1
            continue

        try:
            step_inputs = _build_step_inputs_stage3(
                processor, sample, args.K,
                global_max_pixels=args.global_max_pixels,
                per_img_pixels=args.per_img_pixels,
                special_ids=special_ids,
            )
            if step_inputs is None:
                n_err += 1
                continue

            input_ids = step_inputs["input_ids"].cuda()
            attention_mask = step_inputs["attention_mask"].cuda()
            pixel_values = step_inputs["pixel_values"]
            pixel_values = pixel_values.cuda().to(torch.bfloat16) if pixel_values is not None else None
            image_grid_thw = step_inputs["image_grid_thw"]
            image_grid_thw = image_grid_thw.cuda() if image_grid_thw is not None else None
            alignment_poss = step_inputs["alignment_poss"]

            attn_mask_4d = build_monet_4d_attn(
                input_ids,
                latent_token_id=special_ids["abs_pad"],
                pad_mask=attention_mask,
                dtype=torch.bfloat16,
                mask_latent=False,
                latent_cross_isolate=True,
            )

            # TWO-FORWARD pattern (matches the trainer):
            # 1) latent forward to produce ce_patch_vec/pos (no loss)
            # 2) CE forward with the spliced ce_patch_vec + output_hidden_states=True
            #    → extract per-layer hidden states at latent positions.
            #
            # WHY: the CausalLM output `latent_embeds` is LAST-LAYER ONLY
            # (modeling_qwen2_5_vl_monet.py:2333-2338). For all-layer caching
            # we go through the CE forward, which exposes the full
            # `outputs.hidden_states` tuple — one [B, L, H] tensor per layer.
            lat_out = teacher(
                latent_mode=True,
                input_ids=input_ids,
                attention_mask=attention_mask,
                attention_mask_4d=attn_mask_4d,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=None,
                loss_type=[],
                output_hidden_states=False,
                return_dict=True,
            )
            if lat_out.ce_patch_vec is None or lat_out.ce_patch_pos is None:
                n_err += 1
                continue

            # CE forward: passing `alignment_poss` (the latent positions) makes
            # the model index hidden states for us — see
            # modeling_qwen2_5_vl_monet.py:2033-2065. When output_hidden_states=True
            # AND alignment_poss[0] is non-empty, the model returns
            # `outputs.hidden_states` as a List[Tensor], one element per batch
            # sample with shape [num_layers, K_total, H]. (Vs the standard tuple
            # of full [B, L, H] per layer when alignment_poss is empty.)
            #
            # Loss is still skipped (loss_type=[]); alignment_poss just controls
            # which positions get extracted into outputs.hidden_states.
            ce_out = teacher(
                latent_mode=False,
                input_ids=input_ids,
                attention_mask=attention_mask,
                attention_mask_4d=attn_mask_4d,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=None,
                ce_patch_pos=lat_out.ce_patch_pos,
                ce_patch_vec=lat_out.ce_patch_vec,
                alignment_poss=[alignment_poss],   # required: bypasses model's None-check at L2039
                loss_type=[],
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hs = ce_out.hidden_states
            if hs is None or len(hs) == 0:
                n_err += 1
                continue
            # hs[0] is [num_layers, K_total, H] for batch sample 0 (bsz=1 here).
            all_layers = hs[0]
            if args.alignment_layer == "last_layer":
                tensor_to_save = all_layers[-1].detach().cpu().to(torch.bfloat16)  # [K_total, H]
            else:
                tensor_to_save = all_layers.detach().cpu().to(torch.bfloat16)      # [num_layers, K_total, H]

            torch.save({"latent": tensor_to_save,
                        "metadata": dict(metadata),
                        "K_total": int(tensor_to_save.shape[-2] if tensor_to_save.dim() >= 2 else tensor_to_save.shape[0]),
                        "alignment_layer": args.alignment_layer},
                       out_path)
            n_done += 1
            if n_done % 100 == 0 or n_done <= 5:
                rate = n_done / max(time.time() - t0, 1e-6)
                eta_h = (len(train_ds) - n_done - n_skip) / max(rate, 1e-6) / 3600
                print(f"  [{n_done}/{len(train_ds)}] {fname}  "
                      f"shape={tuple(tensor_to_save.shape)}  rate={rate:.2f}/s  eta={eta_h:.2f}h")
        except Exception as e:
            n_err += 1
            if n_err <= 5:
                import traceback
                print(f"  [error ex={i}] {type(e).__name__}: {e}")
                traceback.print_exc()
            continue

    print(f"[precompute] done: n_done={n_done} n_skip={n_skip} n_err={n_err} "
          f"elapsed={(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
