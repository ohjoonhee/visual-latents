"""Cluster Phase 3 — Monet Stage 3 paper-faithful trainer (+ optional VICReg).

PURPOSE
=======
Reproduce Monet Stage 3 (paper Eq. 4/5) with one optional knob: a VICReg
regularizer on the generated latent embeddings. Hypothesis: Stage 3's
all-layer-cosine alignment to a partially-correlated teacher (Stage-2's
own latents) permits a low-rank degenerate solution — positions 4-7 of
the released ckpt are functionally collinear (Phase 0 finding: mean cos
0.87, n_helpful=1, util ~0). VICReg's variance hinge + cov penalty attack
that failure mode directly.

CONTROLLED A/B
==============
This trainer runs IDENTICALLY in both arms; the only difference is
`lambda_reg` in the config:
  - Job D baseline:  lambda_reg = 0.0     → pure Monet Stage 3
  - Job D + VICReg:  lambda_reg = 1.0+   → adds λ·vicreg(ce_patch_vec)

Both arms share infra deviations (see DEVIATIONS below), so the A/B is
clean on the VICReg variable only.

PIPELINE
========
  Init    = Stage 1 SFT ckpt (cluster_phase3/stage1_sft/checkpoint)
            (NOT Stage 2, per upstream sft_stage3.sh)
  Teacher = Released Monet-SFT-7B Stage 2 latents, PRECOMPUTED to disk
            by scripts/cluster/precompute_teacher_latents.py.
  Student = same example as Stage 2, BUT assistant-turn aux images
            stripped (only the question image remains as actual pixels).
  Loss    = ce_loss
            + alignment_weight · L_align_latent       (Monet Eq. 4)
            + lambda_reg · vicreg(ce_patch_vec)       (our addition; off
                                                       when lambda_reg=0)
  NO `compute_latents_only_loss` — Stage 3 does NOT use the latent-only
  backprop surrogate (per upstream CustomTrainerSFT_STAGE3.compute_loss,
  src/trainer.py:295-372). The alignment grad flows through the whole
  trunk; this is paper-faithful for Stage 3.

DEVIATIONS FROM UPSTREAM (audit notes)
======================================
  1. Teacher choice: released Monet-SFT-7B (= paper's published Stage 2
     ckpt). Upstream's sft_stage3.sh uses a freshly-trained Stage 2 from
     the user's own pipeline. Functionally equivalent if the released
     ckpt is paper-faithful (which Phase 0 confirms for Stage 2's
     forward-pass behavior — Stage 3's collapse is what we measured, so
     using the same released Stage 2 as teacher is the right control).
  2. `attention_mask_4d` via `mask_utils.build_monet_4d_attn` (the
     no-helper-images path — there ARE no helper images in Stage 3
     student inputs, so this is the correct match).
  3. Tokenizer/processor identical to Stage 2 trainer (same special
     tokens registered).

SMOKE GATES (5 steps)
=====================
  - [mem-after-prepare] dist_type=DEEPSPEED        (else DDP fallback)
  - [stage3-check] alignment_active=True           (else teacher latents
                                                     not consumed; misuse)
  - [vicreg-check] vicreg_active=<bool>             (logs whether VICReg
                                                     is on per config)

EVAL
====
  Internal probe: same as Job C (cluster/eval.py, K=8 n=200) — mean cos,
  n_helpful, utility-vs-frozen-Qwen-base. If VICReg works: mean cos
  drops from ~0.87 toward ~0.40 and n_helpful rises.
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
from data_utils import add_monet_special_tokens, load_monet_sft_125k  # noqa: E402
from mask_utils import build_monet_4d_attn  # noqa: E402
from reg import vicreg_loss  # noqa: E402

sys.path.insert(0, str(PHASE0))
from monet_utils import (  # noqa: E402
    add_latent_pad_after_auxiliary_img,
    find_ids_poss,
    replace_latent_placeholder_with_img_pad,
    resize_by_token_budget,
)


LATENT_SIZE = int(os.environ["LATENT_SIZE"])  # 8


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    """Match `<|im_start|>assistant` and label only tokens after it."""
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


def _strip_aux_image_blocks_from_assistant_turns(texts, K, latent_pad_str="<abs_vis_token_pad>"):
    """Stage 3 input transformation: in each assistant turn, REPLACE every
    `<|vision_start|><|image_pad|><|vision_end|>` block with a latent block
    `<abs_vis_token>{K × pad}</abs_vis_token>`. The question image in the
    user turn is untouched. Mirrors upstream `src/utils.py::replace_img_pad_with_latent_pad`.
    """
    latent_pads = latent_pad_str * K
    latent_block = f"<abs_vis_token>{latent_pads}</abs_vis_token>"
    out = []
    for text in texts:
        turns = text.split("<|im_start|>assistant")
        upd = turns[0]  # user/system part — leave question image alone
        for turn in turns[1:]:
            upd += "<|im_start|>assistant" + turn.replace(
                "<|vision_start|><|image_pad|><|vision_end|>",
                latent_block,
            )
        out.append(upd)
    return out


def _build_step_inputs_stage3(processor, sample, K, *,
                              global_max_pixels, per_img_pixels, special_ids):
    """Build per-step inputs for Stage 3: aux images stripped (replaced by
    latent blocks), question image kept. Returns dict or None if invalid.

    The returned `alignment_poss` is the list of latent-token positions in
    input_ids (in token order, after `<|im_start|>assistant`).
    """
    example = sample["data"]
    metadata = sample["metadata"]

    # 1) Render to text
    texts = [processor.apply_chat_template(example, tokenize=False)]
    # 2) Move the `<abs_vis_token></abs_vis_token>` text marker to align with
    #    the image_pad position (mirrors upstream's first preprocessing step
    #    for the teacher path — we'll then strip image_pad in step 3, so the
    #    net effect is: latent block at the right text position, no aux img).
    texts = [replace_latent_placeholder_with_img_pad(t) for t in texts]
    # 3) Stage 3 specific: replace image_pad blocks in ASSISTANT turns with
    #    latent blocks. Question image (user turn) is untouched.
    texts = _strip_aux_image_blocks_from_assistant_turns(texts, K)

    # 4) Get the question image only (it's the only image_pad left after step 3)
    image_inputs, _ = process_vision_info([example])
    if image_inputs:
        # Keep only as many images as remaining image_pad blocks (= 1 for
        # Visual_CoT; for safety we count).
        total_pads = sum(t.count("<|vision_start|><|image_pad|><|vision_end|>")
                         for t in texts)
        image_inputs = image_inputs[:total_pads]
        image_inputs, _ = resize_by_token_budget(
            image_inputs,
            global_max_pixels=global_max_pixels,
            per_img_max_pixels=per_img_pixels,
        )

    total_pads = sum(t.count("<|vision_start|><|image_pad|><|vision_end|>") for t in texts)
    if total_pads != len(image_inputs):
        return None

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Latent positions (these are the alignment positions for Stage 3)
    latent_pad_pattern = torch.tensor([special_ids["abs_pad"]], dtype=torch.long)
    alignment_poss_list = find_ids_poss(input_ids, special_ids["ans_start"], latent_pad_pattern)
    if not alignment_poss_list[0]:
        return None
    alignment_poss = alignment_poss_list[0]  # List[int], in token order

    # Observation positions (for CE-emphasis; same as Stage 2)
    obs_start_poss = find_ids_poss(
        input_ids, special_ids["ans_start"],
        torch.tensor([special_ids["obs_start"]], dtype=torch.long),
    )
    obs_end_poss = find_ids_poss(
        input_ids, special_ids["ans_start"],
        torch.tensor([special_ids["obs_end"]], dtype=torch.long),
    )
    if obs_start_poss[0] and obs_end_poss[0] and len(obs_start_poss[0]) == len(obs_end_poss[0]):
        obs_poss = []
        for s, e in zip(obs_start_poss[0], obs_end_poss[0]):
            obs_poss.extend(list(range(s, e)))
    else:
        obs_poss = []

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=batch.get("pixel_values"),
        image_grid_thw=batch.get("image_grid_thw"),
        alignment_poss=alignment_poss,
        obs_poss=obs_poss,
        metadata=metadata,
    )


def _load_teacher_latents(teacher_dir: Path, metadata: dict, alignment_layer: str = "all_layers"):
    """Load precomputed teacher latents for one sample. Mirrors upstream
    `load_offline_tensor` for Stage 3 (rep_type='latent').

    Expected file: {teacher_dir}/latent_{alignment_layer}_{ds_name}_{sample_id}.pt
    containing a dict with key 'latent' → Tensor[L_layers, K_total, H].
    """
    ds_name = metadata["dataset_name"]
    sample_id = int(metadata["sample_id"])
    fname = f"latent_{alignment_layer}_{ds_name}_{sample_id}.pt"
    path = teacher_dir / fname
    if not path.is_file():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["latent"].detach()  # [L_layers, K_total, H] or [K_total, H]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--init_ckpt", type=str, default=None,
                    help="Stage 1 SFT checkpoint dir (overrides cfg.init_ckpt).")
    ap.add_argument("--teacher_latent_dir", type=str, default=None,
                    help="Precomputed teacher latents (overrides cfg.teacher_latent_dir).")
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
    if args.teacher_latent_dir:
        cfg["teacher_latent_dir"] = args.teacher_latent_dir

    accelerator = Accelerator()
    is_main = accelerator.is_main_process
    set_seed(int(cfg.get("seed", 0)))

    ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if ds_plugin is not None:
        ds_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1

    out_dir = Path(cfg["out_dir"])
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[config] {cfg}", flush=True)
        if cfg.get("wandb", True):
            import wandb
            wandb.init(
                project=cfg.get("wandb_project", "visual-latents"),
                name=cfg.get("run_name", "stage3_monet"),
                config=cfg,
                dir=os.environ.get("WANDB_DIR", "./wandb"),
            )
    log_path = out_dir / "training_log.jsonl"

    base = cfg["init_ckpt"] or cfg["base_model"]
    if is_main:
        print(f"[load] {base} (Monet vendored, Stage 3 student)", flush=True)
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

    # Student (and only model — no inline teacher in Stage 3, teacher reps are
    # precomputed on disk).
    student = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16)
    new_vocab = len(tokenizer)
    student.resize_token_embeddings(new_vocab)
    student.config.vocab_size = new_vocab
    student.config.latent_token_id = special_ids["abs_pad"]
    student.config.latent_start_id = special_ids["abs_start"]
    student.config.latent_end_id = special_ids["abs_end"]
    student.config.answer_start_pattern = special_ids["ans_start"].tolist()

    # Freeze vision tower + projector (LLM-only training).
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

    # Data — same Visual_CoT slice as Stage 2 (eval-200 held out).
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

    # Teacher precompute dir (mandatory — Stage 3 needs it).
    teacher_dir = Path(cfg["teacher_latent_dir"])
    if not teacher_dir.exists():
        raise FileNotFoundError(
            f"[stage3] teacher_latent_dir does not exist: {teacher_dir}. "
            f"Run scripts/cluster/precompute_teacher_latents.py first."
        )
    if is_main:
        n_pt = len(list(teacher_dir.glob("latent_*.pt")))
        print(f"[stage3] teacher_latent_dir={teacher_dir}  n_files={n_pt}", flush=True)

    K = int(cfg.get("latent_size", LATENT_SIZE))
    grad_accum = int(cfg.get("grad_accum_steps", 16))
    log_every = int(cfg.get("log_every", 10))
    save_every = int(cfg.get("save_every", 0))
    alignment_weight = float(cfg.get("alignment_weight", 2.0))
    alignment_layer = cfg.get("alignment_layer", "all_layers")
    ce_emphasize_factor = float(cfg.get("ce_emphasize_factor", 4.0))
    use_attn_mask_4d = bool(cfg.get("use_attn_mask_4d", True))
    global_max_pixels = int(cfg.get("global_max_pixels", 1568000))  # 2000·28²
    per_img_pixels = int(cfg.get("per_img_pixels", 1003520))

    # VICReg knobs — when lambda_reg == 0, term is skipped entirely.
    lambda_reg = float(cfg.get("lambda_reg", 0.0))
    reg_var_w = float(cfg.get("reg_var_w", 1.0))
    reg_cov_w = float(cfg.get("reg_cov_w", 0.04))
    reg_gamma = float(cfg.get("reg_gamma", 1.0))
    vicreg_on = lambda_reg > 0.0
    if is_main:
        print(f"[stage3] alignment_weight={alignment_weight} alignment_layer={alignment_layer} "
              f"ce_emph={ce_emphasize_factor} lambda_reg={lambda_reg} vicreg_on={vicreg_on}", flush=True)

    student, optim, sched = accelerator.prepare(student, optim, sched)

    student.train()
    rank = accelerator.process_index
    world = accelerator.num_processes

    if accelerator.is_main_process:
        try:
            alloc_gb = torch.cuda.memory_allocated() / 1e9
            reserved_gb = torch.cuda.memory_reserved() / 1e9
            # Stage 3 has NO inline teacher (vs Stage 2 +14 GB/rank), so the
            # at-rest footprint is ~14-18 GB smaller. ZeRO-2 should still be
            # engaged; ZeRO-3 not needed at this scale.
            print(f"[mem-after-prepare] dist_type={accelerator.distributed_type} "
                  f"alloc={alloc_gb:.2f}GB reserved={reserved_gb:.2f}GB world={world}", flush=True)
        except Exception as _e:
            print(f"[mem-after-prepare] log failed: {_e}", flush=True)

    step = 0
    t0 = time.time()
    losses_micro = {"total": [], "ce": [], "align": [], "vicreg": [], "n_lat": []}
    epoch_idx = 0
    n_examples = len(train_ds)
    stage3_logged = False
    vicreg_logged = False

    while step < n_steps:
        idx = list(range(n_examples))
        random.Random(int(cfg.get("seed", 0)) + epoch_idx).shuffle(idx)
        my_idx = idx[rank::world]
        for i in my_idx:
            try:
                step_inputs = _build_step_inputs_stage3(
                    processor, train_ds[i], K,
                    global_max_pixels=global_max_pixels,
                    per_img_pixels=per_img_pixels,
                    special_ids=special_ids,
                )
                if step_inputs is None:
                    continue

                # Load teacher latents BEFORE host→device transfer so we can
                # skip the example cheaply if missing.
                teacher_latent = _load_teacher_latents(
                    teacher_dir, step_inputs["metadata"], alignment_layer=alignment_layer,
                )
                if teacher_latent is None:
                    continue
                # Sanity: K_total must match the latent positions in this sample.
                K_total = teacher_latent.shape[-2] if teacher_latent.dim() >= 2 else teacher_latent.shape[0]
                if K_total != len(step_inputs["alignment_poss"]):
                    if is_main and step < 5:
                        print(f"  [skip ex={i}] teacher K_total={K_total} != student "
                              f"len(align_poss)={len(step_inputs['alignment_poss'])}", flush=True)
                    continue

                input_ids = step_inputs["input_ids"].to(accelerator.device)
                attention_mask = step_inputs["attention_mask"].to(accelerator.device)
                pixel_values = step_inputs["pixel_values"]
                pixel_values = pixel_values.to(accelerator.device, dtype=torch.bfloat16) if pixel_values is not None else None
                image_grid_thw = step_inputs["image_grid_thw"]
                image_grid_thw = image_grid_thw.to(accelerator.device) if image_grid_thw is not None else None
                alignment_poss = step_inputs["alignment_poss"]
                obs_poss = step_inputs["obs_poss"]

                teacher_latent = teacher_latent.to(accelerator.device, dtype=torch.bfloat16)

                # 4D attention mask (no-helper-image variant; aux images are
                # stripped in Stage 3 student inputs).
                attn_mask_4d = None
                if use_attn_mask_4d:
                    attn_mask_4d = build_monet_4d_attn(
                        input_ids,
                        latent_token_id=special_ids["abs_pad"],
                        pad_mask=attention_mask,
                        dtype=torch.bfloat16,
                        mask_latent=False,
                        latent_cross_isolate=True,
                    )

                # ─── Latent forward: generate latents (no loss yet) ───
                # Upstream Monet sft_stage3.compute_loss (src/trainer.py:295-372):
                # latent_mode=True, loss_type=[], output_hidden_states=False.
                # No teacher_hidden_states passed in latent forward — alignment is
                # computed in CE forward where ce_patch_vec is spliced.
                accelerator.unwrap_model(student).gradient_checkpointing_disable()
                lat_out = student(
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
                    continue

                # ─── CE forward: alignment via teacher_hidden_states_for_alignment ───
                accelerator.unwrap_model(student).gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
                ignore_ids = [int(special_ids[k]) for k in
                              ["abs_pad", "abs_end", "img_pad", "v_start", "v_end",
                               "obs_start", "obs_end", "end_pad"]]
                labels = _generate_labels(input_ids, special_ids["ans_start"], ignore_ids).to(accelerator.device)

                ce_loss_type = ["ce", "alignment"]
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
                    ce_emphasize_poss=[obs_poss] if obs_poss else None,
                    ce_emphasize_factor=ce_emphasize_factor if obs_poss else 1.0,
                    alignment_poss=[alignment_poss],
                    teacher_hidden_states_for_alignment=[teacher_latent],
                    loss_type=ce_loss_type,
                    return_dict=True,
                    use_cache=False,
                )
                ce_loss = ce_out.loss
                align_loss = None
                if ce_out.loss_dict is not None and "alignment" in ce_out.loss_dict:
                    align_loss = ce_out.loss_dict["alignment"]
                if align_loss is None:
                    align_loss = torch.zeros((), device=accelerator.device, dtype=ce_loss.dtype)

                if is_main and not stage3_logged:
                    print(f"[stage3-check] alignment_active={float(align_loss.detach()) != 0.0} "
                          f"align={float(align_loss.detach()):.6f} ce={float(ce_loss.detach()):.4f} "
                          f"K_total={K_total}", flush=True)
                    stage3_logged = True

                # ─── VICReg term on ce_patch_vec ───
                vicreg_term = torch.zeros((), device=accelerator.device, dtype=ce_loss.dtype)
                if vicreg_on and lat_out.ce_patch_vec is not None:
                    # ce_patch_vec is List[Tensor] with one tensor per batch
                    # element. At bsz=1, stack the per-position tensors into
                    # [1, K_total, H] and pass through vicreg_loss.
                    vecs_per_batch = lat_out.ce_patch_vec
                    if len(vecs_per_batch) > 0 and len(vecs_per_batch[0]) > 0:
                        # vecs_per_batch[0] is a list of [H] tensors; stack to [K_total, H]
                        stacked = torch.stack(
                            [v for v in vecs_per_batch[0]] if isinstance(vecs_per_batch[0], (list, tuple))
                            else list(vecs_per_batch[0]),
                            dim=0,
                        )  # [K_total, H]
                        h_btkd = stacked.unsqueeze(0).to(ce_loss.dtype)  # [1, K_total, H]
                        vicreg_term = vicreg_loss(
                            h_btkd,
                            var_weight=reg_var_w,
                            cov_weight=reg_cov_w,
                            gamma=reg_gamma,
                        )
                    if is_main and not vicreg_logged:
                        print(f"[vicreg-check] vicreg_active=True lambda_reg={lambda_reg} "
                              f"var_w={reg_var_w} cov_w={reg_cov_w} gamma={reg_gamma} "
                              f"term={float(vicreg_term.detach()):.6f}", flush=True)
                        vicreg_logged = True
                elif is_main and not vicreg_logged:
                    print(f"[vicreg-check] vicreg_active=False (lambda_reg={lambda_reg})", flush=True)
                    vicreg_logged = True

                # ─── Loss combination — paper-faithful Stage 3 + optional VICReg ───
                #   total = ce + α·alignment + λ·vicreg
                # NO compute_latents_only_loss surrogate — upstream Stage 3
                # explicitly omits it (src/trainer.py:354-356).
                total_loss = ce_loss + alignment_weight * align_loss + lambda_reg * vicreg_term

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
            losses_micro["vicreg"].append(float(vicreg_term.detach().float().cpu().item()))
            losses_micro["n_lat"].append(float(K_total))

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
                        "vicreg": avg["vicreg"],
                        "n_lat": avg["n_lat"],
                        "gpu_peak_gb": mem_gb,
                        "elapsed_s": time.time() - t0,
                    }
                    print(f"[step {step}/{n_steps}] tot={avg['total']:.3f} ce={avg['ce']:.3f} "
                          f"align={avg['align']:.4f} vicreg={avg['vicreg']:.4f} "
                          f"n_lat={avg['n_lat']:.1f} mem={mem_gb:.1f}GB elapsed={row['elapsed_s']:.0f}s", flush=True)
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
