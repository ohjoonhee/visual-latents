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
    ap.add_argument("--n_prompts_list", default="",
                    help="comma list e.g. 1,2,4,8 — sweep batch sizes to find the padding threshold; overrides --n_prompts")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="fp32 = CPU-stable; bf16 = match the GPU run (ckpt is bf16; CPU bf16 is slower/partial)")
    args = ap.parse_args()

    import torch
    torch.manual_seed(0)
    print(f"forward_smoke: ckpt={args.checkpoint} lvr_steps={args.lvr_steps} "
          f"max_new_tokens={args.max_new_tokens} n_prompts={args.n_prompts}", flush=True)

    # ---- load processor + patched model on CPU ----
    print(f"\n=== [LOAD] processor + QwenWithLVR (sdpa, {args.dtype}, CPU) ===", flush=True)
    from transformers import AutoProcessor, AutoConfig, GenerationConfig
    from src.model.qwen_lvr_model import QwenWithLVR
    from monkey_patch_forward_lvr_rl import replace_qwen2_5_with_mixed_modality_forward_lvr_rl
    from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    replace_qwen2_5_with_mixed_modality_forward_lvr_rl()   # same patch train_grpo applies
    _dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = QwenWithLVR.from_pretrained(
        args.checkpoint,
        config=config,
        torch_dtype=_dtype,               # fp32 = CPU-stable; bf16 = match the GPU run
        attn_implementation="sdpa",       # FA2 needs CUDA; sdpa runs on CPU
    )
    replace_qwen_2_5_vl_patch_emb()
    # match the GRPO trainer's rollout: gradient_checkpointing on forces use_cache=False, which
    # changes the cache_position setup. The earlier use_cache=True smoke missed the
    # _get_initial_cache_position ambiguous-bool that only surfaced on GPU; reproduce that path here.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.eval()
    print(f"[LOAD] ok: {type(model).__name__} on {next(model.parameters()).device} "
          f"dtype={next(model.parameters()).dtype} use_cache={model.config.use_cache} grad_ckpt=on", flush=True)

    # ---- dataset + the trainer's exact prompt-batching helper ----
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

    def build_prompt_inputs(n):
        # mirror grpo_trainer._generate_and_score_completions L701-715 (padding_side="left")
        inputs = [ds[i] for i in range(n)]
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(ex, processor)["prompt"] for ex in inputs]
        image_inputs, video_inputs, video_kwargs = process_vision_info(prompts, return_video_kwargs=True)
        return processor(text=prompts_text, images=image_inputs, videos=video_inputs,
                         padding=True, padding_side="left", return_tensors="pt", **video_kwargs)

    gc = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.9,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        decoding_strategy="steps",
        lvr_steps=args.lvr_steps,
    )

    # ---- sweep batch sizes: n_prompts=1 is unpadded; n>=2 with mixed lengths -> left-padding.
    # If the multinomial 'inf/nan' assert reproduces only for n>=2, padding/masking is the
    # trigger (a non-GPU bug we can then fix+validate on CPU); if all pass on fp32 CPU it's
    # bf16/flash-attn specific (try --dtype bf16, else it's a true GPU-only numerical issue).
    n_list = [int(x) for x in args.n_prompts_list.split(",") if x.strip()] or [args.n_prompts]
    print(f"\n=== [GENERATE] sweep n_prompts={n_list} dtype={args.dtype} "
          f"(lvr_steps={args.lvr_steps}, max_new_tokens={args.max_new_tokens}) ===", flush=True)
    results = []
    for n in n_list:
        try:
            pin = build_prompt_inputs(n)
            in_shape = tuple(pin["input_ids"].shape)
            with torch.no_grad():
                out = model.generate(**pin, generation_config=gc)
            results.append((n, in_shape, True, f"ids {tuple(out.shape)}"))
            print(f"[PASS] n_prompts={n} input_ids={in_shape} -> {tuple(out.shape)}", flush=True)
        except Exception as e:
            results.append((n, None, False, f"{type(e).__name__}: {e}"))
            print(f"[FAIL] n_prompts={n}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    print("\n========== FORWARD_SMOKE SWEEP SUMMARY ==========", flush=True)
    for n, shp, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] n_prompts={n} {('input='+str(shp)) if shp else ''}: {detail}", flush=True)
    n_fail = sum(1 for *_, ok, _ in results if not ok)
    if n_fail:
        print(f"FORWARD_SMOKE REPRODUCED the failure for {n_fail}/{len(results)} config(s) "
              f"(dtype={args.dtype}) — runtime forward bug is now CPU-reproducible", flush=True)
        sys.exit(1)
    print(f"FORWARD_SMOKE GREEN: all {len(results)} config(s) passed on CPU (dtype={args.dtype}) — "
          f"patched forward + rope + LVR step-decoding execute cleanly", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
