#!/usr/bin/env python
"""
CPU FORWARD-SMOKE for LVR Stage-2 GRPO (run via slurm/lvr_stage2_forward_smoke.sbatch
on cpu-short — NO GPU). The fast 8-stage preflight (preflight_stage2.py) checks every
non-GPU surface *up to* the model forward, and FORWARD_SIG checks the patched forward's
signature — but it cannot catch RUNTIME drift inside the forward (e.g. the rope_deltas
AttributeError from transformers 4.54.0 moving get_rope_index/rope_deltas onto self.model).
Those bugs are not GPU-specific; they just need the forward to actually execute.

This smoke loads QwenWithLVR from the Stage-1 checkpoint on CPU (sdpa, fp32), applies the
exact monkey-patches train_grpo.py applies, builds prompt inputs the way the GRPO trainer
does (process_vision_info -> processor, grpo_trainer.py L701-715), and runs a tiny LVR
`model.generate(decoding_strategy="steps", lvr_steps=N)` on one sample. If generate
completes, the whole patched forward + _lvr_deocding_by_steps + rope path executed cleanly.

Caveats / scope:
  - sdpa, not flash_attention_2 (FA2 needs CUDA). So this does NOT exercise the FA2 varlen
    cu_seq_lens path (FORWARD_SIG covers that at signature level); it DOES exercise rope,
    image-embed scatter, the decoder, lm_head, and the LVR step-decoding loop.
  - fp32 on CPU (the ckpt is bf16); upcast is fine for a structural smoke.
  - heavier than the fast preflight (3B load ~1-2 min + a few-token CPU generate); still a
    cpu-short slot, not a 4xH100 allocation.
Exit 0 if generate runs, 1 otherwise.
"""
import argparse
import os
import sys
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--image_min_pixels", type=int, default=128 * 28 * 28)
    ap.add_argument("--image_max_pixels", type=int, default=2560 * 28 * 28)
    ap.add_argument("--lvr_steps", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--n_prompts", type=int, default=1)
    args = ap.parse_args()

    import torch
    torch.manual_seed(0)
    print(f"forward_smoke: ckpt={args.checkpoint} lvr_steps={args.lvr_steps} "
          f"max_new_tokens={args.max_new_tokens} n_prompts={args.n_prompts}", flush=True)

    # ---- load processor + patched model on CPU ----
    print("\n=== [LOAD] processor + QwenWithLVR (sdpa, fp32, CPU) ===", flush=True)
    from transformers import AutoProcessor, AutoConfig, GenerationConfig
    from src.model.qwen_lvr_model import QwenWithLVR
    from monkey_patch_forward_lvr_rl import replace_qwen2_5_with_mixed_modality_forward_lvr_rl
    from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    replace_qwen2_5_with_mixed_modality_forward_lvr_rl()   # same patch train_grpo applies
    model = QwenWithLVR.from_pretrained(
        args.checkpoint,
        config=config,
        torch_dtype=torch.float32,        # CPU-friendly (ckpt is bf16)
        attn_implementation="sdpa",       # FA2 needs CUDA; sdpa runs on CPU
    )
    replace_qwen_2_5_vl_patch_emb()
    model.config.use_cache = True
    model.eval()
    print(f"[LOAD] ok: {type(model).__name__} on {next(model.parameters()).device} "
          f"dtype={next(model.parameters()).dtype}", flush=True)

    # ---- build prompt inputs exactly like grpo_trainer._generate_and_score_completions ----
    print("\n=== [INPUTS] build prompt batch via process_vision_info + processor ===", flush=True)
    from qwen_vl_utils import process_vision_info
    from trl.data_utils import maybe_apply_chat_template
    from src.dataset import make_grpo_data_module
    from src.params import DataArguments

    da = DataArguments(
        data_path=args.data_path,
        image_folder=args.image_folder,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        lazy_preprocess=True,
    )
    ds = make_grpo_data_module(model_id=args.model_id, processor=processor, data_args=da)["train_dataset"]
    inputs = [ds[i] for i in range(args.n_prompts)]
    prompts = [x["prompt"] for x in inputs]
    prompts_text = [maybe_apply_chat_template(ex, processor)["prompt"] for ex in inputs]
    image_inputs, video_inputs, video_kwargs = process_vision_info(prompts, return_video_kwargs=True)
    prompt_inputs = processor(
        text=prompts_text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
        **video_kwargs,
    )
    print(f"[INPUTS] ok: input_ids={tuple(prompt_inputs['input_ids'].shape)} "
          f"keys={sorted(prompt_inputs.keys())}", flush=True)

    # ---- the actual forward-executing check: tiny LVR generate ----
    print("\n=== [GENERATE] tiny LVR step-decoding (executes the patched forward + rope) ===", flush=True)
    gc = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.9,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        decoding_strategy="steps",
        lvr_steps=args.lvr_steps,
    )
    try:
        with torch.no_grad():
            out = model.generate(**prompt_inputs, generation_config=gc)
    except Exception as e:
        print(f"[FAIL] GENERATE: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        print("\nFORWARD_SMOKE FAILED: the patched forward / LVR decoding raised at runtime "
              "(this is exactly the class — e.g. rope_deltas — that only forward execution catches)", flush=True)
        sys.exit(1)

    shape = tuple(out.shape) if hasattr(out, "shape") else type(out).__name__
    print(f"[PASS] GENERATE: produced ids {shape} "
          f"(prompt {tuple(prompt_inputs['input_ids'].shape)} + up to {args.max_new_tokens} new)", flush=True)
    print("\nFORWARD_SMOKE GREEN: patched GRPO forward + rope + LVR step-decoding execute "
          "cleanly on CPU — runtime forward path validated", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
