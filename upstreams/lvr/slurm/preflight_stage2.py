#!/usr/bin/env python
"""
CPU preflight for LVR Stage-2 GRPO (run via slurm/lvr_stage2_preflight.sbatch on
cpu-short — NO GPU). Drives the *real* Stage-2 code path through every stage that
does not require a CUDA forward, so GPU-independent breakage (missing deps, arg/
schema drift, the seed_worker signature, unresolved image paths, reward/math_verify
faults) is caught for the price of a CPU slot instead of a 4xH100 allocation.

What it exercises (mirrors src/train/train_grpo.py up to trainer.train()):
  1. IMPORTS      — import the full train_grpo module tree (catches e.g. math_verify)
  2. DATAARGS     — build DataArguments + json-validate the DeepSpeed config
  3. PROCESSOR    — AutoProcessor.from_pretrained(<stage1 ckpt>)  (train_grpo L222)
  4. DATASET      — make_grpo_data_module(...) + __getitem__ over a sample (schema)
  5. IMAGE_OPEN   — process_vision_info(prompts) over the sample  (train_grpo L705;
                    this is the exact call that raised the ViRL39K FileNotFoundError)
                    + maybe_apply_chat_template (train_grpo L703)
  6. DATALOADER   — real DataLoader with worker_init_fn=partial(seed_worker,...) and
                    num_workers>0, fetch batches (reproduces the seed_worker crash)
  7. REWARDS      — load_reward_funcs(...) + call accuracy/format reward (math_verify)

Deliberately NOT covered (genuinely GPU-only, or has side effects):
  - QwenWithLVR weight load with flash_attention_2 (FA2 needs CUDA; sdpa-on-CPU
    would be a slow false test), the GRPO rollout generate, teacher-force replay,
    GRPO loss numerics, DeepSpeed ZeRO-2 init, OOM.
  - normalize_special_tokens() — it *edits* tokenizer.json in the ckpt dir; a
    preflight must not mutate the checkpoint (the real run already applied it).

Exit code: 0 if all stages PASS, 1 if any FAIL. Each stage is independent so one
failure still lets the rest run and report.
"""
import argparse
import json
import os
import random
import sys
import traceback

RESULTS = []  # (stage, ok, detail)


def stage(name):
    """Decorator-ish context: run fn, capture pass/fail without aborting the rest."""
    def run(fn):
        print(f"\n=== [{name}] ===", flush=True)
        try:
            detail = fn()
            RESULTS.append((name, True, detail or ""))
            print(f"[PASS] {name}: {detail or 'ok'}", flush=True)
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        return None
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Stage-1 checkpoint dir (init for GRPO)")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--image_min_pixels", type=int, default=128 * 28 * 28)
    ap.add_argument("--image_max_pixels", type=int, default=2560 * 28 * 28)
    ap.add_argument("--deepspeed_config", default="scripts/zero2.json")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--n_sample", type=int, default=int(os.getenv("PREFLIGHT_N", "64")))
    args = ap.parse_args()

    print(f"preflight_stage2: ckpt={args.checkpoint}")
    print(f"  data={args.data_path} images={args.image_folder} model_id={args.model_id}")
    print(f"  pixels=[{args.image_min_pixels},{args.image_max_pixels}] "
          f"num_workers={args.num_workers} n_sample={args.n_sample}")

    # Shared handles populated across stages.
    ctx = {}

    # ---- 1. IMPORTS: the whole train_grpo tree (math_verify, qwen_vl_utils, trl, ...) ----
    @stage("IMPORTS")
    def _imports():
        import src.train.train_grpo  # noqa: F401  (transitively imports model/trainer/dataset/reward/math_verify)
        from qwen_vl_utils import process_vision_info  # noqa: F401
        from trl.data_utils import maybe_apply_chat_template  # noqa: F401
        from transformers.trainer_utils import seed_worker  # noqa: F401
        import math_verify  # noqa: F401
        return "train_grpo + qwen_vl_utils + trl + seed_worker + math_verify import OK"

    # ---- 2. DATAARGS + DeepSpeed config json ----
    @stage("DATAARGS")
    def _dataargs():
        from src.params import DataArguments
        da = DataArguments(
            data_path=args.data_path,
            image_folder=args.image_folder,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
            lazy_preprocess=True,
        )
        ctx["data_args"] = da
        with open(args.deepspeed_config) as f:
            dsj = json.load(f)
        return f"DataArguments built; deepspeed cfg {args.deepspeed_config} parsed (zero stage={dsj.get('zero_optimization',{}).get('stage','?')})"

    # ---- 3. PROCESSOR from the Stage-1 checkpoint (train_grpo L222) ----
    @stage("PROCESSOR")
    def _processor():
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(args.checkpoint)
        ctx["processor"] = proc
        tok = getattr(proc, "tokenizer", None)
        return f"{type(proc).__name__} loaded; tokenizer={type(tok).__name__ if tok else None}"

    # ---- 4. DATASET via the real make_grpo_data_module + __getitem__ sample ----
    @stage("DATASET")
    def _dataset():
        from src.dataset import make_grpo_data_module
        if "processor" not in ctx or "data_args" not in ctx:
            raise RuntimeError("prereq stage failed (processor/data_args missing)")
        dm = make_grpo_data_module(model_id=args.model_id, processor=ctx["processor"], data_args=ctx["data_args"])
        ds = dm["train_dataset"]
        ctx["ds"] = ds
        n = len(ds)
        # exercise __getitem__ on a deterministic sample (schema: prompt/assistant)
        random.seed(0)
        idxs = [0, 1, n - 1] + random.sample(range(n), min(args.n_sample, n))
        idxs = sorted(set(i for i in idxs if 0 <= i < n))
        ctx["sample_idxs"] = idxs
        for i in idxs:
            ex = ds[i]
            assert "prompt" in ex and "assistant" in ex, f"item {i} missing prompt/assistant keys: {list(ex.keys())}"
        return f"len={n}; __getitem__ ok on {len(idxs)} items (keys: prompt/assistant)"

    # ---- 5. IMAGE_OPEN: maybe_apply_chat_template + process_vision_info (train_grpo L703-705) ----
    @stage("IMAGE_OPEN")
    def _image_open():
        from qwen_vl_utils import process_vision_info
        from trl.data_utils import maybe_apply_chat_template
        ds = ctx.get("ds")
        idxs = ctx.get("sample_idxs")
        if ds is None or idxs is None:
            raise RuntimeError("prereq DATASET failed")
        inputs = [ds[i] for i in idxs]
        # exact prompt-processing the trainer does before generation:
        prompts_text = [maybe_apply_chat_template(ex, ctx["processor"])["prompt"] for ex in inputs]
        prompts = [ex["prompt"] for ex in inputs]
        image_inputs, video_inputs, video_kwargs = process_vision_info(prompts, return_video_kwargs=True)
        n_img = len(image_inputs) if image_inputs is not None else 0
        sizes = [tuple(getattr(im, "size", ("?", "?"))) for im in (image_inputs or [])[:3]]
        return (f"chat_template ok ({len(prompts_text)} prompts); process_vision_info opened "
                f"{n_img} images over {len(idxs)} samples (sample sizes {sizes})")

    # ---- 6. DATALOADER with the real seed_worker partial + num_workers>0 ----
    @stage("DATALOADER")
    def _dataloader():
        from functools import partial
        from torch.utils.data import DataLoader
        from transformers.trainer_utils import seed_worker
        ds = ctx.get("ds")
        if ds is None:
            raise RuntimeError("prereq DATASET failed")
        # mirror grpo_trainer.get_train_dataloader: passthrough collate, worker_init_fn=partial(seed_worker,...)
        nw = args.num_workers
        dl = DataLoader(
            ds,
            batch_size=2,
            num_workers=nw,
            collate_fn=lambda feats: feats,  # GRPO uses a passthrough collator
            worker_init_fn=partial(seed_worker, num_workers=nw, rank=0),
            shuffle=False,
        )
        it = iter(dl)
        b0 = next(it)
        b1 = next(it)
        return f"DataLoader(num_workers={nw}, worker_init_fn=partial(seed_worker)) yielded batches of {len(b0)},{len(b1)} — seed_worker OK"

    # ---- 7. REWARDS: discovery + math_verify runtime ----
    @stage("REWARDS")
    def _rewards():
        from src.utils import load_reward_funcs
        funcs = load_reward_funcs("src.train.reward_funcs")
        names = [f.__name__ for f in funcs]
        from src.train.reward_funcs import accuracy_reward, format_reward
        completions = [[{"content": "<|lvr_start|>foo<|lvr_end|> <answer>42</answer>"}],
                       [{"content": "<answer>7</answer>"}]]
        assistant = [{"content": "<answer>42</answer>"}, {"content": "<answer>8</answer>"}]
        acc = accuracy_reward(completions, assistant)
        fmt = format_reward(completions)
        # sanity: first sample matches (acc=1), second is wrong (acc=0); format[0]=1, format[1]=0
        assert acc == [1.0, 0.0], f"accuracy_reward unexpected: {acc}"
        assert fmt == [1.0, 0.0], f"format_reward unexpected: {fmt}"
        return f"discovered {names}; accuracy={acc} format={fmt} (math_verify OK)"

    # ---- summary ----
    print("\n========== PREFLIGHT SUMMARY ==========")
    n_fail = 0
    for name, ok, detail in RESULTS:
        flag = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(f"  [{flag}] {name}: {detail}")
    print(f"=======================================")
    if n_fail:
        print(f"PREFLIGHT FAILED: {n_fail}/{len(RESULTS)} stage(s) failed", flush=True)
        sys.exit(1)
    print(f"PREFLIGHT GREEN: all {len(RESULTS)} stages passed — Stage-2 CPU surface clean", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
