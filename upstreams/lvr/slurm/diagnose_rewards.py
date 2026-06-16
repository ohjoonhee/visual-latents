#!/usr/bin/env python
"""
CPU reward diagnosis for LVR Stage-2 GRPO (run via slurm/lvr_diagnose_rewards.sbatch on
cpu-short — NO GPU). Stage-2 trained with reward==0 for 300+ steps (format_reward never
fired, accuracy_reward ~0), i.e. no RL signal. This reproduces the rollout->reward path
on CPU to show EXACTLY why: it loads the SFT checkpoint, generates a real completion the
way the trainer does (decoding_strategy=steps, lvr_steps), decodes it both with and
without skip_special_tokens (the trainer uses skip_special_tokens=True at grpo_trainer.py
:829), pulls the gold answer from the dataset, and runs the real accuracy_reward /
format_reward on it — printing the completion text, the gold, and each reward's verdict.

format_reward regex (reward_funcs.py): ^<\\|lvr_start\\|>.*?<\\|lvr_end\\|>\\s*<answer>.*?</answer>$
accuracy_reward: math_verify(parse(completion) vs parse(gold)) or <answer> string match.
"""
import argparse
import re
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--image_min_pixels", type=int, default=128 * 28 * 28)
    ap.add_argument("--image_max_pixels", type=int, default=2560 * 28 * 28)
    ap.add_argument("--lvr_steps", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--n_prompts", type=int, default=4)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="bf16 matches the GPU rollout; fp32 is CPU-stable")
    args = ap.parse_args()

    import torch
    torch.manual_seed(0)

    from transformers import AutoProcessor, AutoConfig, GenerationConfig
    from src.model.qwen_lvr_model import QwenWithLVR
    from monkey_patch_forward_lvr_rl import replace_qwen2_5_with_mixed_modality_forward_lvr_rl
    from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb
    from qwen_vl_utils import process_vision_info
    from trl.data_utils import maybe_apply_chat_template
    from src.dataset import make_grpo_data_module
    from src.params import DataArguments
    from src.train.reward_funcs import accuracy_reward, format_reward

    print(f"[load] {args.checkpoint} (sdpa/{args.dtype}/CPU, grad_ckpt, use_cache=False)", flush=True)
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    replace_qwen2_5_with_mixed_modality_forward_lvr_rl()
    _dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = QwenWithLVR.from_pretrained(args.checkpoint, config=config, torch_dtype=_dtype, attn_implementation="sdpa")
    replace_qwen_2_5_vl_patch_emb()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.eval()

    da = DataArguments(data_path=args.data_path, image_folder=args.image_folder,
                       image_min_pixels=args.image_min_pixels, image_max_pixels=args.image_max_pixels, lazy_preprocess=True)
    ds = make_grpo_data_module(model_id=args.model_id, processor=processor, data_args=da)["train_dataset"]

    inputs = [ds[i] for i in range(args.n_prompts)]
    prompts = [x["prompt"] for x in inputs]
    golds = [x["assistant"]["content"] if isinstance(x["assistant"], dict) else str(x["assistant"]) for x in inputs]
    prompts_text = [maybe_apply_chat_template(ex, processor)["prompt"] for ex in inputs]
    image_inputs, video_inputs, video_kwargs = process_vision_info(prompts, return_video_kwargs=True)
    pin = processor(text=prompts_text, images=image_inputs, videos=video_inputs,
                    padding=True, padding_side="left", return_tensors="pt", **video_kwargs)
    prompt_len = pin["input_ids"].shape[1]

    gc = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.9,
                          pad_token_id=processor.tokenizer.pad_token_id, eos_token_id=processor.tokenizer.eos_token_id,
                          decoding_strategy="steps", lvr_steps=args.lvr_steps)
    print(f"[generate] n_prompts={args.n_prompts} max_new_tokens={args.max_new_tokens} lvr_steps={args.lvr_steps}", flush=True)
    with torch.no_grad():
        out = model.generate(**pin, generation_config=gc)
    completion_ids = out[:, prompt_len:]

    # EXACTLY as grpo_trainer.py:829 builds the text the reward funcs see:
    comp_skip = processor.batch_decode(completion_ids, skip_special_tokens=True)
    comp_raw = processor.batch_decode(completion_ids, skip_special_tokens=False)

    fmt_pattern = r"^<\|lvr_start\|>.*?<\|lvr_end\|>\s*<answer>.*?</answer>$"

    for i in range(args.n_prompts):
        completions = [[{"content": comp_skip[i]}]]
        assistant = [{"content": golds[i]}]
        fmt = format_reward(completions)[0]
        acc = accuracy_reward(completions, assistant)[0]
        has_start = "<|lvr_start|>" in comp_skip[i]
        has_end = "<|lvr_end|>" in comp_skip[i]
        has_ans = "<answer>" in comp_skip[i]
        print("\n" + "=" * 80, flush=True)
        print(f"[prompt {i}] format_reward={fmt}  accuracy_reward={acc}", flush=True)
        print(f"  contains: <|lvr_start|>={has_start} <|lvr_end|>={has_end} <answer>={has_ans}", flush=True)
        print(f"  regex full-match: {bool(re.match(fmt_pattern, comp_skip[i], re.DOTALL))}", flush=True)
        print(f"  --- completion (skip_special_tokens=True, what reward sees) [{len(completion_ids[i])} toks] ---", flush=True)
        print("  " + repr(comp_skip[i][:600]), flush=True)
        print(f"  --- completion (raw, skip_special_tokens=False) ---", flush=True)
        print("  " + repr(comp_raw[i][:600]), flush=True)
        print(f"  --- gold answer ---", flush=True)
        print("  " + repr(golds[i][:400]), flush=True)

    print("\n[done] reward diagnosis complete", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
