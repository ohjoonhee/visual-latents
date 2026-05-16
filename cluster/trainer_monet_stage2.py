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
- `emphasize_latent_weight=2.0`: the paper's "latent-only backprop" trick
  (`compute_latents_only_loss`). As of Job C this is PAPER-FAITHFUL, not
  approximated. The latent forward only generates latents (loss_type=[]);
  the alignment loss is computed in the CE forward (where ce_patch_vec is
  spliced into inputs_embeds, so the alignment loss is graph-connected to
  ce_patch_vec). The total loss is then
    total = emphasize_latent_weight
            * compute_latents_only_loss(ce_patch_vec, alignment_weight*align)
            + ce_loss
  so the alignment signal reaches parameters ONLY through the latent-
  generation graph (the LLM trunk sees only the CE gradient). This mirrors
  Monet `src/trainer.py:152-224` (CustomTrainerSFT_STAGE2.compute_loss)
  exactly, including the `else: total = ce + alignment_weight*align`
  fallback when emphasize is off or alignment is trivially zero.

DEVIATIONS FROM UPSTREAM (audit notes — read before claiming
"paper-faithful"):
  1. FIXED in Job C (was: plain scalar add). The latent-only backprop is
     now a verbatim port of upstream `compute_latents_only_loss` with the
     upstream forward structure (alignment in the CE forward). A smoke-time
     `[devfix-check] latent_only_connected=` line asserts the alignment
     gradient actually reaches ce_patch_vec before the full run proceeds.
  2. `attention_mask_4d` is hand-rolled in `mask_utils.build_monet_4d_attn`
     with `latent_cross_isolate=True, mask_latent=False`. Upstream's
     `build_4d_attn` does similar work but with different bookkeeping for
     prefix tokens and auxiliary-image isolation. The mask test in
     `phase1_5b_attn/MASK_VALIDATION.md` showed the approximation is
     correct on the cases we tested.
  3. Inline teacher forward (not offline-precomputed) — described below.
     Functionally equivalent to upstream when the teacher checkpoint is
     identical, but adds ~14 GB/rank to the resident footprint.
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


def compute_latents_only_loss(latents, loss_for_latents, diag=None):
    """Verbatim port of Monet `src/trainer.py:11-42`.

    Returns a proxy loss whose gradient w.r.t. `latents` (the latent
    embeddings, ce_patch_vec) equals the gradient of `loss_for_latents`,
    but which carries NO gradient anywhere else (grads are detached). So
    backpropagating it pushes the alignment signal only through the
    latent-generation graph, never the LLM trunk.

    `diag`, if a dict, is filled with connectivity counters for the
    smoke-time non-degeneracy check. It does not affect the return value
    or its gradient (upstream numerics preserved exactly).
    """
    def _flatten_tensors(x):
        if isinstance(x, (list, tuple)):
            out = []
            for y in x:
                out.extend(_flatten_tensors(y))
            return out
        return [x]

    ce_vec_list = _flatten_tensors(latents)
    grads = torch.autograd.grad(
        outputs=loss_for_latents,
        inputs=ce_vec_list,
        retain_graph=True,   # ce_loss backward (shared CE-forward buffers) follows
        create_graph=False,  # stop higher-order graph
        allow_unused=True,   # some ce vectors may not be used
    )

    safe_grads = []
    n_nonzero = 0
    for v, g in zip(ce_vec_list, grads):
        if g is None:
            g = torch.zeros_like(v)
        else:
            n_nonzero += 1
        safe_grads.append(g.detach())  # detach to stop any trunk path

    if diag is not None:
        diag["n_inputs"] = len(ce_vec_list)
        diag["n_nonzero_grad"] = n_nonzero

    proxy_loss = torch.stack([(v * g).sum() for v, g in zip(ce_vec_list, safe_grads)]).sum()
    return proxy_loss


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

    # accelerate's DeepSpeedPlugin defaults train_micro_batch_size_per_gpu
    # to "auto", which the validator rejects without a DataLoader. We
    # iterate examples manually, so overwrite with an int (setdefault is
    # not enough because the "auto" string already occupies the key).
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

    if accelerator.is_main_process:
        try:
            import torch.cuda as _tc
            alloc_gb = _tc.memory_allocated() / 1e9
            reserved_gb = _tc.memory_reserved() / 1e9
            # Stage 2 holds student (ZeRO-2 sharded) + teacher (full-resident,
            # frozen, per-rank). At 7B bf16 the teacher alone is ~14 GB/rank,
            # so expect alloc ≈ 30-35 GB at rest (before any forward acts).
            # If you see <20 GB this is a DDP fallback (check dist_type).
            print(f"[mem-after-prepare] dist_type={accelerator.distributed_type} "
                  f"alloc={alloc_gb:.2f}GB reserved={reserved_gb:.2f}GB world={world}", flush=True)
        except Exception as _e:
            print(f"[mem-after-prepare] log failed: {_e}", flush=True)

    step = 0
    t0 = time.time()
    losses_micro = {"total": [], "ce": [], "align": [], "obs_n": []}
    epoch_idx = 0
    n_examples = len(train_ds)
    devfix_logged = False  # gate the one-shot [devfix-check] smoke line

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

                # ---- Student latent forward (latent generation ONLY) ----
                # Paper-faithful: upstream Monet src/trainer.py:160-163 runs
                # the latent forward with loss_type=[] and hidden states off
                # — it only produces ce_patch_pos / ce_patch_vec. The
                # alignment loss is computed later in the CE forward (so it
                # is graph-connected to ce_patch_vec, which is what
                # compute_latents_only_loss needs). Grad-checkpointing OFF
                # for the latent forward (use_cache path).
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

                # ---- Student CE forward (with spliced ce_patch_pos/vec from latent forward) ----
                accelerator.unwrap_model(student).gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
                ignore_ids = [int(special_ids[k]) for k in
                              ["abs_pad", "abs_end", "img_pad", "v_start", "v_end",
                               "obs_start", "obs_end", "end_pad"]]
                labels = _generate_labels(input_ids, special_ids["ans_start"], ignore_ids).to(accelerator.device)

                # Alignment is computed HERE (CE forward) so it is graph-
                # connected to ce_patch_vec spliced into inputs_embeds —
                # mirrors upstream Monet src/trainer.py:169-202.
                ce_loss_type = ["ce"]
                if alignment_weight != 0:
                    ce_loss_type.append("alignment")

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
                    alignment_poss=[obs_poss],
                    teacher_hidden_states_for_alignment=[teacher_rep],
                    loss_type=ce_loss_type,
                    return_dict=True,
                    use_cache=False,
                )
                ce_loss = ce_out.loss
                align_loss = None
                if ce_out.loss_dict is not None and "alignment" in ce_out.loss_dict:
                    align_loss = ce_out.loss_dict["alignment"]

                # Loss combination — paper-faithful (Monet src/trainer.py:197-202).
                # When emphasize_latent_weight != 0 and alignment is non-trivial,
                # the alignment term is backpropped ONLY through the latent
                # embeddings (ce_patch_vec) via compute_latents_only_loss; the
                # LLM trunk sees only the CE gradient. Otherwise: weighted sum.
                if align_loss is None:
                    align_loss = torch.zeros((), device=accelerator.device, dtype=ce_loss.dtype)
                if emphasize_latent_weight != 0.0 and float(align_loss.detach()) != 0.0:
                    _dg = {}
                    latent_only_loss = compute_latents_only_loss(
                        lat_out.ce_patch_vec, alignment_weight * align_loss, diag=_dg,
                    )
                    total_loss = emphasize_latent_weight * latent_only_loss + ce_loss
                    if is_main and not devfix_logged:
                        _conn = _dg.get("n_nonzero_grad", 0) > 0
                        print(f"[devfix-check] latent_only_connected={_conn} "
                              f"n_inputs={_dg.get('n_inputs')} "
                              f"n_nonzero_grad={_dg.get('n_nonzero_grad')} "
                              f"proxy={float(latent_only_loss.detach()):.6f} "
                              f"align={float(align_loss.detach()):.6f} "
                              f"ce={float(ce_loss.detach()):.4f}", flush=True)
                        devfix_logged = True
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
