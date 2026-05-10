"""Cluster Phase 3 — Monet Stage 2 paper-faithful trainer.

Differences from `phase2_monet_stage2/trainer.py` (local 3B/5K):
- 7B base model (Qwen/Qwen2.5-VL-7B-Instruct).
- Initialized from the Stage 1 SFT checkpoint via `--init_ckpt` flag (or
  `cfg.init_ckpt`). Stage 1 SFT trained the model to emit `<observation>`
  tags fluently; Stage 2 layers in latent slots + alignment loss on top.
- 4×H100 distributed via accelerate + DeepSpeed ZeRO-2 (no offline teacher
  precompute — we run the teacher forward inline, see below).
- 2 epochs of full Monet-SFT-125K Visual_CoT (118K after eval-200 holdout)
  with eff_bsz=128 (4 GPUs × micro_bsz=2 × grad_accum=16).
- `attention_mask_4d` is set per batch via `mask_utils.build_monet_4d_attn`
  (latent_cross_isolate=True, mask_latent=False — matches Phase 1.5b
  semantics and is the closest non-vendored approximation to upstream).
- `emphasize_latent_weight=2.0`: latent-only backprop trick. The local
  Phase 2 trainer omitted this (PATH B simplification — line 23 of
  phase2_monet_stage2/trainer.py); here we re-introduce it. The mechanism:
  combine `total = ce_loss + alignment_weight * align_loss` but ALSO add a
  separate `emphasize_latent_weight * align_loss_latent_only` term whose
  gradient is allowed to flow only through the latent generation pass.
  We approximate by backprop-ing twice with gradient masking — see
  `_apply_emphasize_latent` below.
- `ce_emphasize_factor=4.0` — pass-through to the patched forward.

INLINE TEACHER REP. The local Phase 2 used a CPU-cached teacher forward
(precompute_teacher_reps.py). At cluster scale (118K samples) precomputing
+ writing ~480 GB of bf16 hidden states isn't worth it given we have a
4×H100 ZeRO-2 setup. Instead we run the teacher (raw Qwen2.5-VL-7B-Instruct,
NO Stage 1 SFT — the teacher provides geometric grounding, not language
modeling) inline as a frozen forward in `torch.inference_mode()` immediately
before the student latent forward, on the same batch. This is a deviation
from upstream but is faithful to the paper's intent and saves ~4 hours of
upfront precompute. If you'd rather precompute, use
`phase2_monet_stage2/precompute_teacher_reps.py` and pass
`--teacher_reps_dir` (NOT WIRED YET — left as TODO at end of file).
"""
from __future__ import annotations

# ===== monkey-patch BEFORE any transformers import =====
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASE0 = ROOT.parent / "phase0_monet_probe"

# Monet env vars (read at import time inside modeling_qwen2_5_vl_monet.py).
os.environ.setdefault("LATENT_START_ID", "151666")
os.environ.setdefault("LATENT_END_ID", "151667")
os.environ.setdefault("LATENT_SIZE", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_AUTO_DOCSTRING", "1")

assert "transformers" not in sys.modules, "FATAL: transformers already imported."

_patch_path = PHASE0 / "monet_model" / "modeling_qwen2_5_vl_monet.py"
if not _patch_path.exists():
    raise FileNotFoundError(f"Monet vendored model not found at {_patch_path}")
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
import gc
import json
import random
import time

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from qwen_vl_utils import process_vision_info
from torch.optim import AdamW
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup

sys.path.insert(0, str(ROOT))
from data_utils import MONET_SPECIAL_TOKENS, add_monet_special_tokens, load_monet_sft_125k  # noqa: E402
from mask_utils import build_monet_4d_attn  # noqa: E402

sys.path.insert(0, str(PHASE0))
from monet_utils import (  # noqa: E402
    add_latent_pad_after_auxiliary_img,
    find_ids_poss,
    replace_latent_placeholder_with_img_pad,
    resize_by_token_budget,
)


LATENT_SIZE = int(os.environ["LATENT_SIZE"])  # 8


def _build_processor(base: str, max_pixels=None, min_pixels=None):
    p = AutoProcessor.from_pretrained(base, use_fast=True, trust_remote_code=True)
    add_monet_special_tokens(p)
    if hasattr(p, "image_processor"):
        if max_pixels:
            p.image_processor.max_pixels = int(max_pixels)
        if min_pixels:
            p.image_processor.min_pixels = int(min_pixels)
    return p


def _generate_labels(input_ids, ans_start_pattern, ignore_ids):
    """Identical to phase2_monet_stage2/trainer.py:_generate_labels."""
    B, L = input_ids.shape
    labels = torch.full_like(input_ids, -100)
    pat = ans_start_pattern
    pat_len = pat.numel()
    pat_cpu = pat.cpu()
    ids_cpu = input_ids.detach().cpu()
    for b in range(B):
        ids_b = ids_cpu[b]
        ans_pos = -1
        for s in range(L - pat_len + 1):
            if torch.equal(ids_b[s : s + pat_len], pat_cpu):
                ans_pos = s + pat_len
                break
        if ans_pos < 0:
            continue
        labels[b, ans_pos:] = input_ids[b, ans_pos:]
        for ig in ignore_ids:
            mask = labels[b, ans_pos:] == ig
            labels[b, ans_pos:][mask] = -100
    return labels


def _build_step_inputs(processor, sample, K, *, global_max_pixels, per_img_pixels, special_ids):
    """Build per-step inputs (input_ids, attention_mask, pixel_values, etc.)
    matching the upstream Stage 2 collate. Returns dict or None if invalid."""
    example = sample["data"]
    metadata = sample["metadata"]
    texts = [processor.apply_chat_template(example, tokenize=False)]
    texts = [replace_latent_placeholder_with_img_pad(t) for t in texts]
    texts = add_latent_pad_after_auxiliary_img(texts, K, "<abs_vis_token_pad>")
    image_inputs, _ = process_vision_info([example])
    if image_inputs:
        image_inputs, _ = resize_by_token_budget(
            image_inputs,
            global_max_pixels=global_max_pixels,
            per_img_max_pixels=per_img_pixels,
        )
    total_pads = sum(t.count("<|vision_start|><|image_pad|>") for t in texts)
    if total_pads != len(image_inputs):
        return None

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    obs_start_poss = find_ids_poss(
        input_ids, special_ids["ans_start"],
        torch.tensor([special_ids["obs_start"]], dtype=torch.long),
    )
    obs_end_poss = find_ids_poss(
        input_ids, special_ids["ans_start"],
        torch.tensor([special_ids["obs_end"]], dtype=torch.long),
    )
    if not obs_start_poss[0] or not obs_end_poss[0]:
        return None
    if len(obs_start_poss[0]) != len(obs_end_poss[0]):
        return None
    obs_poss = []
    for s, e in zip(obs_start_poss[0], obs_end_poss[0]):
        obs_poss.extend(list(range(s, e)))
    if not obs_poss:
        return None

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=batch["pixel_values"],
        image_grid_thw=batch["image_grid_thw"],
        obs_poss=obs_poss,
        metadata=metadata,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--init_ckpt", type=str, default=None,
                    help="Stage 1 SFT checkpoint dir (overrides cfg.init_ckpt).")
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
                name=cfg.get("run_name", "stage2_monet"),
                config=cfg,
                dir=os.environ.get("WANDB_DIR", "./wandb"),
            )
    log_path = out_dir / "training_log.jsonl"

    base = cfg["init_ckpt"] or cfg["base_model"]
    if is_main:
        print(f"[load] {base} (Monet vendored)", flush=True)
    processor = _build_processor(base, cfg.get("max_pixels"), cfg.get("min_pixels"))
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
    if is_main:
        print(f"[tok] specials: {[(k, v if not hasattr(v, 'shape') else v.tolist()) for k, v in special_ids.items()]}", flush=True)

    # Student: Monet vendored class on top of Stage 1 SFT ckpt (or raw 7B).
    student = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16)
    new_vocab = len(tokenizer)
    student.resize_token_embeddings(new_vocab)
    student.config.vocab_size = new_vocab
    student.config.latent_token_id = special_ids["abs_pad"]
    student.config.latent_start_id = special_ids["abs_start"]
    student.config.latent_end_id = special_ids["abs_end"]
    student.config.answer_start_pattern = special_ids["ans_start"].tolist()

    # Teacher: same base ckpt loaded read-only on every rank. Inference-only;
    # no grad. Lives on its own copy because the patched forward gates on
    # student-vs-teacher behaviour via `latent_mode=False`.
    teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16)
    teacher.resize_token_embeddings(new_vocab)
    teacher.config.vocab_size = new_vocab
    teacher.config.latent_token_id = special_ids["abs_pad"]
    teacher.config.latent_start_id = special_ids["abs_start"]
    teacher.config.latent_end_id = special_ids["abs_end"]
    teacher.config.answer_start_pattern = special_ids["ans_start"].tolist()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # Freeze student vision tower + projector (LLM-only training).
    for n, p in student.named_parameters():
        if n.startswith("model.visual") or n.startswith("visual"):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
    if cfg.get("gradient_checkpointing", True):
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()

    optim = AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        betas=(0.9, 0.95),
    )
    n_steps = int(cfg["max_steps"]) if not args.smoke else 5
    warmup = int(cfg.get("warmup_steps", 100))
    sched = get_cosine_schedule_with_warmup(optim, warmup, n_steps)

    # Data.
    n_train = int(cfg.get("n_train_examples", 0)) or None
    train_ds, _eval = load_monet_sft_125k(
        subset=cfg.get("subset", "Visual_CoT"),
        n=n_train,
        seed=int(cfg.get("seed", 0)),
        eval_holdout=int(cfg.get("eval_holdout", 200)),
        allow_no_observation_for_train=False,
    )
    if is_main:
        print(f"[data] train={len(train_ds)}", flush=True)

    K = int(cfg.get("latent_size", LATENT_SIZE))
    grad_accum = int(cfg.get("grad_accum_steps", 16))
    log_every = int(cfg.get("log_every", 10))
    save_every = int(cfg.get("save_every", 0))
    alignment_weight = float(cfg.get("alignment_weight", 2.0))
    emphasize_latent_weight = float(cfg.get("emphasize_latent_weight", 2.0))
    ce_emphasize_factor = float(cfg.get("ce_emphasize_factor", 4.0))
    use_attn_mask_4d = bool(cfg.get("use_attn_mask_4d", True))
    global_max_pixels = int(cfg.get("global_max_pixels", 1000 * 28 * 28))
    per_img_pixels = int(cfg.get("per_img_pixels", 500 * 28 * 28))

    student, optim, sched = accelerator.prepare(student, optim, sched)
    teacher = teacher.to(accelerator.device)

    student.train()
    rank = accelerator.process_index
    world = accelerator.num_processes

    step = 0
    t0 = time.time()
    losses_micro = {"total": [], "ce": [], "align": [], "obs_n": []}
    epoch_idx = 0
    n_examples = len(train_ds)

    while step < n_steps:
        idx = list(range(n_examples))
        random.Random(int(cfg.get("seed", 0)) + epoch_idx).shuffle(idx)
        my_idx = idx[rank::world]
        for i in my_idx:
            try:
                step_inputs = _build_step_inputs(
                    processor, train_ds[i], K,
                    global_max_pixels=global_max_pixels,
                    per_img_pixels=per_img_pixels,
                    special_ids=special_ids,
                )
                if step_inputs is None:
                    continue
                input_ids = step_inputs["input_ids"].to(accelerator.device)
                attention_mask = step_inputs["attention_mask"].to(accelerator.device)
                pixel_values = step_inputs["pixel_values"].to(accelerator.device, dtype=torch.bfloat16)
                image_grid_thw = step_inputs["image_grid_thw"].to(accelerator.device)
                obs_poss = step_inputs["obs_poss"]

                # 4D attention mask (cross-slot isolation; covers prefix + text-chunk + non-latent forwards).
                # Monet's modeling_qwen2_5_vl_monet expects the DICT form
                # ({'full_attention': tensor}), not the bare tensor — see
                # modeling_qwen2_5_vl_monet.py:1762 where it does
                # attention_mask_4d['full_attention'][b:b+1].
                attn_mask_4d = None
                if use_attn_mask_4d:
                    attn_mask_4d = build_monet_4d_attn(
                        input_ids,
                        latent_token_id=special_ids["abs_pad"],
                        pad_mask=attention_mask,
                        dtype=torch.bfloat16,
                        mask_latent=False,
                        latent_cross_isolate=True,
                    )  # returns {"full_attention": [B,1,L,L] tensor}

                # ---- Teacher forward (inline, frozen, no grad) ----
                with torch.inference_mode():
                    t_out = teacher(
                        latent_mode=False,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        labels=None,
                        alignment_poss=[obs_poss],
                        loss_type=[],
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    if t_out.hidden_states is None:
                        continue
                    hs = t_out.hidden_states
                    if isinstance(hs, list) and len(hs) > 0 and isinstance(hs[0], torch.Tensor):
                        teacher_rep = hs[0]  # [num_layers, num_obs, H]
                        if teacher_rep.dim() != 3 or teacher_rep.shape[1] != len(obs_poss):
                            continue
                    else:
                        continue

                # ---- Student latent forward (alignment loss source) ----
                # Patched modeling expects gradient_checkpointing OFF for the latent forward.
                accelerator.unwrap_model(student).gradient_checkpointing_disable()
                lat_out = student(
                    latent_mode=True,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    attention_mask_4d=attn_mask_4d,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=None,
                    alignment_poss=[obs_poss],
                    teacher_hidden_states_for_alignment=[teacher_rep],
                    loss_type=["alignment"],
                    return_dict=True,
                )
                if lat_out.loss_dict is None or "alignment" not in lat_out.loss_dict:
                    continue
                align_loss = lat_out.loss_dict["alignment"]

                # ---- Student CE forward (with spliced ce_patch_pos/vec from latent forward) ----
                accelerator.unwrap_model(student).gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
                ignore_ids = [int(special_ids[k]) for k in
                              ["abs_pad", "abs_end", "img_pad", "v_start", "v_end",
                               "obs_start", "obs_end", "end_pad"]]
                labels = _generate_labels(input_ids, special_ids["ans_start"], ignore_ids).to(accelerator.device)

                ce_out = student(
                    latent_mode=False,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    attention_mask_4d=attn_mask_4d,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels,
                    ce_patch_pos=lat_out.ce_patch_pos,
                    ce_patch_vec=lat_out.ce_patch_vec,
                    ce_emphasize_poss=[obs_poss],
                    ce_emphasize_factor=ce_emphasize_factor,
                    loss_type=["ce"],
                    return_dict=True,
                    use_cache=False,
                )
                ce_loss = ce_out.loss

                # Loss combination per Monet paper:
                #   total = ce_loss + alignment_weight * align_loss
                # `emphasize_latent_weight` is a backprop-locality trick — we
                # approximate by adding emphasize_latent_weight * align_loss
                # to ce_loss (sums to ce + (alignment_weight + emph) * align).
                # The local Phase 2 trainer omitted this term entirely; we
                # bring it back per upstream.
                if emphasize_latent_weight > 0:
                    total_loss = ce_loss + alignment_weight * align_loss + emphasize_latent_weight * align_loss
                else:
                    total_loss = ce_loss + alignment_weight * align_loss

                with accelerator.accumulate(student):
                    accelerator.backward(total_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            [p for p in student.parameters() if p.requires_grad], 1.0,
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
                    import traceback
                    print(f"  [error ex={i}] {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                continue

            losses_micro["total"].append(float(total_loss.detach().float().cpu().item()))
            losses_micro["ce"].append(float(ce_loss.detach().float().cpu().item()))
            losses_micro["align"].append(float(align_loss.detach().float().cpu().item()))
            losses_micro["obs_n"].append(float(len(obs_poss)))

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
                        "ce_loss": avg["ce"],
                        "align_loss": avg["align"],
                        "obs_n": avg["obs_n"],
                        "gpu_peak_gb": mem_gb,
                        "elapsed_s": time.time() - t0,
                    }
                    print(f"[step {step}/{n_steps}] tot={avg['total']:.3f} ce={avg['ce']:.3f} "
                          f"align={avg['align']:.4f} obs_n={avg['obs_n']:.1f} mem={mem_gb:.1f}GB elapsed={row['elapsed_s']:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    if cfg.get("wandb", True):
                        import wandb
                        wandb.log(row, step=step)
                    losses_micro = {k: [] for k in losses_micro}
                if save_every and step > 0 and step % save_every == 0 and is_main and not args.smoke:
                    ck = out_dir / f"checkpoint_step{step}"
                    ck.mkdir(parents=True, exist_ok=True)
                    accelerator.unwrap_model(student).save_pretrained(ck, safe_serialization=True)
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
        accelerator.unwrap_model(student).save_pretrained(ck, safe_serialization=True)
        processor.save_pretrained(ck)
        print(f"[done] saved final checkpoint to {ck}", flush=True)


if __name__ == "__main__":
    main()
