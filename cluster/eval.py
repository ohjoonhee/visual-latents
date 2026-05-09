"""Cluster Phase 3 — common eval pipeline.

For each checkpoint provided via `--ckpts <name>:<path> ...`:
  1. Detect ckpt type from the saved tokenizer:
       - has Monet special tokens → Monet-recipe (Stage 2 / V3) — use
         latent_mode=True forward to extract slot hidden states from a
         100-200 example held-out subset of `Visual_CoT/train.json`.
       - no Monet tokens → Pivot A — extract slot hidden states by
         splicing image features into K=K image_pad slots, same as
         `pivot_a/extract_latents.py`.
  2. Run ablation modes (none, all, first_half, last_half, only_pos_i for i
     in K) using the **shared base reader** (Qwen/Qwen2.5-VL-7B-Instruct,
     frozen). For each (ckpt, mode), record the mean answer-span CE.
  3. Compute compression_ratio, mean_off_diag_cos, n_helpful, qwen_base
     utility, self utility (using the ckpt itself as reader). Write
     per-ckpt `<out_root>/<name>/REPORT.md` and a unified
     `<out_root>/CROSS_REPORT.md`.

Output layout:
  <out_root>/
    cross_report.md
    <name>/
      REPORT.md
      results.jsonl          # one row per (mode, reader)
      h_stats.jsonl          # one row per ckpt
      latents/latent_*.pt

The 200-example eval is the SAME shuffle/seed used by `data_utils.load_monet_sft_125k`
(eval slice). See `data_utils.py` for the protocol.

Note on monkey-patch. We must monkey-patch `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl`
BEFORE any `transformers` import only when handling a Monet-recipe ckpt.
For pivot_a ckpts the patch is harmless (the patched class is a strict
superset of the upstream one). We patch unconditionally to keep the code
path simple — same as Phase 0's ablation.py.
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
import time
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(ROOT))
from data_utils import load_monet_sft_125k, load_pivot_a_dataset, pivot_a_example_to_tuple  # noqa: E402

import PIL.Image as _PILImage  # noqa: E402

# Center-crop fallback bbox (matches trainer_pivot.py:_MONET_CENTER_BBOX).
_MONET_CENTER_BBOX = (0.25, 0.25, 0.75, 0.75)


def _monet_processed_to_pivot_tuple(processed: dict):
    """Mirror of trainer_pivot.py:_monet_sample_to_pivot_tuple — duplicated
    here to keep eval.py self-contained (the Monet monkey-patch makes
    importing trainer_pivot risky)."""
    user_image_path = None
    user_text_parts = []
    last_assistant_text = ""
    for turn in processed.get("data", []):
        role = turn.get("role")
        for c in turn.get("content", []):
            if c.get("type") == "image" and role == "user" and user_image_path is None:
                user_image_path = c.get("image")
            elif c.get("type") == "text":
                if role == "user":
                    user_text_parts.append(c.get("text", ""))
                elif role == "assistant":
                    last_assistant_text = c.get("text", last_assistant_text)
    if user_image_path is None or not last_assistant_text:
        return None
    try:
        image = _PILImage.open(user_image_path)
    except Exception:
        return None
    if image.mode != "RGB":
        image = image.convert("RGB")
    question = "\n".join(t for t in user_text_parts if t).lstrip("\n").strip()
    if not question:
        return None
    answer = (
        last_assistant_text
        .replace("<abs_vis_token></abs_vis_token>", "")
        .replace("<observation>", "")
        .replace("</observation>", "")
        .strip()
    )
    if not answer:
        return None
    return image, question, answer, _MONET_CENTER_BBOX

sys.path.insert(0, str(PHASE0))
from monet_utils import (  # noqa: E402
    add_latent_pad_after_auxiliary_img,
    find_ids_poss,
    replace_latent_placeholder_with_img_pad,
    resize_by_token_budget,
)

sys.path.insert(0, str(ROOT.parent / "phase1_lvr"))
from roi import select_roi_indices, post_merger_token_count  # noqa: E402


LATENT_SIZE = int(os.environ["LATENT_SIZE"])  # K=8 default


# ---------------------------------------------------------------------------
# ckpt type detection
# ---------------------------------------------------------------------------

def detect_ckpt_type(ckpt_path: Path) -> str:
    """Return 'monet' if the ckpt's tokenizer has Monet special tokens, else 'pivot'."""
    proc = AutoProcessor.from_pretrained(str(ckpt_path), use_fast=True, trust_remote_code=True)
    vocab = proc.tokenizer.get_vocab()
    if "<abs_vis_token_pad>" in vocab and "<observation>" in vocab:
        return "monet"
    return "pivot"


# ---------------------------------------------------------------------------
# latent extraction
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_monet_latents(ckpt_path: Path, *, n: int = 200, K: int = LATENT_SIZE):
    """For Monet-recipe ckpts: latent_mode=True forward on held-out Visual_CoT."""
    proc = AutoProcessor.from_pretrained(str(ckpt_path), use_fast=True, trust_remote_code=True)
    for tk in ["<abs_vis_token_pad>", "<abs_vis_token>", "</abs_vis_token>", "<observation>", "</observation>"]:
        proc.tokenizer.add_tokens(tk, special_tokens=True)
    tok = proc.tokenizer

    def _id(s):
        return int(tok(s, return_tensors="pt")["input_ids"][0, 0])

    abs_pad_id = _id("<abs_vis_token_pad>")
    abs_start_id = _id("<abs_vis_token>")
    abs_end_id = _id("</abs_vis_token>")
    answer_start_pattern = tok("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(ckpt_path), torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tok))
    model.config.vocab_size = len(tok)
    model.config.latent_token_id = abs_pad_id
    model.config.latent_start_id = abs_start_id
    model.config.latent_end_id = abs_end_id
    model.config.answer_start_pattern = answer_start_pattern.tolist()
    model.eval().cuda()

    _train_unused, eval_processed = load_monet_sft_125k(eval_holdout=n)
    print(f"[extract] monet ckpt={ckpt_path}  eval_n={len(eval_processed)}", flush=True)

    out: list[dict] = []
    started = time.time()
    for sid_i, processed in enumerate(eval_processed):
        try:
            example = processed["data"]
            texts = [proc.apply_chat_template(example, tokenize=False)]
            texts = [replace_latent_placeholder_with_img_pad(t) for t in texts]
            texts = add_latent_pad_after_auxiliary_img(texts, K, "<abs_vis_token_pad>")
            from qwen_vl_utils import process_vision_info
            image_inputs, _ = process_vision_info([example])
            if image_inputs:
                image_inputs, _ = resize_by_token_budget(
                    image_inputs,
                    global_max_pixels=1000 * 28 * 28,
                    per_img_max_pixels=500 * 28 * 28,
                )
            total_pads = sum(t.count("<|vision_start|><|image_pad|>") for t in texts)
            if total_pads != len(image_inputs):
                continue
            batch = proc(text=texts, images=image_inputs, return_tensors="pt", padding=True)
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            pixel_values = batch["pixel_values"].cuda(dtype=torch.bfloat16)
            image_grid_thw = batch["image_grid_thw"].cuda()

            latent_pad_pattern = torch.tensor([abs_pad_id], dtype=torch.long)
            alignment_poss = find_ids_poss(batch["input_ids"], answer_start_pattern, latent_pad_pattern)
            if not alignment_poss[0]:
                continue

            out_lat = model(
                latent_mode=True,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=None,
                loss_type=[],
                alignment_poss=alignment_poss,
                output_latent_embeds=True,
                return_dict=True,
            )
            latent = out_lat.latent_embeds[0].detach().cpu()  # [num_latents, H]
            if latent.shape[0] < K:
                continue
            out.append({
                "h": latent[:K].unsqueeze(0),  # [1, K, H]
                "question": _extract_question(processed),
                "answer": _extract_answer(processed),
                "sample_id": int(processed["metadata"]["sample_id"]),
            })
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  [skip extract] {type(e).__name__}: {e}", flush=True)
            continue

    del model, proc
    gc.collect(); torch.cuda.empty_cache()
    print(f"[extract] done {len(out)} latents in {time.time()-started:.1f}s", flush=True)
    return out


def _extract_question(sample):
    for step in sample["data"]:
        if step["role"] == "user":
            for c in step["content"]:
                if c.get("type") == "text":
                    return c["text"].lstrip("\n").strip()
    return ""


def _extract_answer(sample):
    last_text = ""
    for step in sample["data"]:
        if step["role"] == "assistant":
            for c in step["content"]:
                if c.get("type") == "text":
                    last_text = c["text"]
    cleaned = (
        last_text
        .replace("<abs_vis_token></abs_vis_token>", "")
        .replace("<observation>", "")
        .replace("</observation>", "")
        .strip()
    )
    return cleaned


@torch.inference_mode()
def extract_pivot_latents(ckpt_path: Path, *, n: int = 200, K: int = LATENT_SIZE):
    """For Pivot A ckpts (V1, V2, V4): build prompt with K image_pad slots
    after the user image, splice ROI image features as v_full, run forward
    with output_hidden_states, and slice the last layer at the K slot
    positions.

    CONTROLLED COMPARISON: uses the SAME Monet-SFT-125K Visual_CoT held-out
    200-example slice as the Monet-recipe extractor (`extract_monet_latents`)
    and as the Stage 2 / V3 trainers' eval_holdout. The per-example bbox
    is the center-crop fallback `(0.25, 0.25, 0.75, 0.75)` — see
    trainer_pivot.py header for rationale.
    """
    proc = AutoProcessor.from_pretrained(str(ckpt_path), use_fast=True, trust_remote_code=True)
    tok = proc.tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(ckpt_path), torch_dtype=torch.bfloat16)
    model.eval().cuda()
    image_token_id = model.config.image_token_id

    _train_unused, eval_processed = load_monet_sft_125k(eval_holdout=n)
    print(f"[extract] pivot ckpt={ckpt_path}  eval_n={len(eval_processed)} (monet_sft_125k held-out)", flush=True)

    out = []
    started = time.time()
    for i, processed in enumerate(eval_processed):
        try:
            tup = _monet_processed_to_pivot_tuple(processed)
            if tup is None:
                continue
            image, question, answer, bbox = tup
            vision_inputs = proc.image_processor(images=image, return_tensors="pt")
            pixel_values = vision_inputs["pixel_values"].cuda(dtype=torch.bfloat16)
            image_grid_thw = vision_inputs["image_grid_thw"].cuda()
            n_image_tokens = post_merger_token_count(image_grid_thw, merge_size=2)
            v_full = torch.cat(model.model.get_image_features(pixel_values, image_grid_thw), dim=0)

            # Build prompt with K image_pad slots after the user image.
            img_pads = "<|image_pad|>" * n_image_tokens
            latent_pads = "<|image_pad|>" * K
            prefix = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|>{img_pads}<|vision_end|>{question}<|im_end|>\n"
                f"<|im_start|>assistant\n{latent_pads}"
            )
            prefix_ids = tok(prefix, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            attention_mask = torch.ones_like(prefix_ids)
            text_embeds = model.get_input_embeddings()(prefix_ids)
            inputs_embeds = text_embeds.clone()
            pad_positions = (prefix_ids[0] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
            image_patch_positions = pad_positions[:n_image_tokens]
            slot_positions = pad_positions[n_image_tokens : n_image_tokens + K]
            inputs_embeds[0, image_patch_positions, :] = v_full.to(inputs_embeds.dtype)

            out_h = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            last_hidden = out_h.hidden_states[-1]
            h = last_hidden[0, slot_positions, :].detach().cpu()  # [K, D]
            sid = int(processed.get("metadata", {}).get("sample_id", i))
            out.append({"h": h.unsqueeze(0), "question": question, "answer": answer, "sample_id": sid})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  [skip extract i={i}] {type(e).__name__}: {e}", flush=True)
            continue

    del model, proc
    gc.collect(); torch.cuda.empty_cache()
    print(f"[extract] done {len(out)} latents in {time.time()-started:.1f}s", flush=True)
    return out


# ---------------------------------------------------------------------------
# ablation
# ---------------------------------------------------------------------------

def ablation_modes(K_total: int = LATENT_SIZE) -> dict[str, list[int]]:
    half = K_total // 2
    modes = {
        "all": list(range(K_total)),
        "first_only": [0],
        "last_only": [K_total - 1],
        "first_half": list(range(half)),
        "last_half": list(range(half, K_total)),
        "none": [],
    }
    for i in range(K_total):
        modes[f"only_pos_{i}"] = [i]
    return modes


def apply_ablation(h: torch.Tensor, keep_idx: list[int]) -> torch.Tensor:
    out = torch.zeros_like(h)
    if keep_idx:
        idx_t = torch.tensor(keep_idx, device=h.device, dtype=torch.long)
        out[:, idx_t, :] = h[:, idx_t, :]
    return out


def build_prompt_no_answer(K_total, question):
    img_pads = "<|image_pad|>" * K_total
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|vision_start|>{img_pads}<|vision_end|>{question}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_prompt_full(K_total, question, answer):
    return build_prompt_no_answer(K_total, question) + answer + "<|im_end|>"


@torch.inference_mode()
def forward_anchor(model, tokenizer, image_token_id, h, questions, answers, K_total):
    if h.shape[1] != K_total:
        raise ValueError(f"h.shape[1]={h.shape[1]} but K={K_total}")
    B = h.shape[0]
    device = next(model.parameters()).device
    no_answer_ids = []
    full_ids = []
    for q, a in zip(questions, answers, strict=True):
        ids_no_a = tokenizer(build_prompt_no_answer(K_total, q), add_special_tokens=False).input_ids
        ids_full = tokenizer(build_prompt_full(K_total, q, a), add_special_tokens=False).input_ids
        if ids_full[: len(ids_no_a)] != ids_no_a:
            ids_a = tokenizer(a + "<|im_end|>", add_special_tokens=False).input_ids
            ids_full = ids_no_a + ids_a
        no_answer_ids.append(ids_no_a)
        full_ids.append(ids_full)
    max_len = max(len(x) for x in full_ids)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
    for b, ids in enumerate(full_ids):
        L = len(ids)
        input_ids[b, :L] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[b, :L] = 1
        ans_start = len(no_answer_ids[b])
        labels[b, ans_start:L] = input_ids[b, ans_start:L]

    base_embeds = model.get_input_embeddings()(input_ids)
    h_cast = h.to(device=device, dtype=base_embeds.dtype)
    inputs_embeds = base_embeds.clone()
    for b in range(B):
        positions = (input_ids[b] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
        inputs_embeds[b, positions, :] = h_cast[b]

    out = model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        loss_type=["ce"],
        use_cache=False,
    )
    return float(out.loss.item())


def run_reader_eval(model, tok, image_token_id, latents, K_total):
    modes = ablation_modes(K_total)
    cache = {n: [] for n in modes}
    started = time.time()
    for i, item in enumerate(latents):
        h_dev = item["h"].cuda()
        for name, keep_idx in modes.items():
            h_ab = apply_ablation(h_dev, keep_idx)
            try:
                nll = forward_anchor(model, tok, image_token_id, h_ab, [item["question"]], [item["answer"]], K_total)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                nll = float("nan")
            cache[name].append(nll)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{len(latents)}] elapsed={time.time()-started:.1f}s", flush=True)
    rows = []
    for name, vals in cache.items():
        valid = [v for v in vals if v == v]
        rows.append({
            "mode": name, "keep": modes[name],
            "nll_mean": (sum(valid) / max(1, len(valid))) if valid else float("nan"),
            "nll_per_example": vals, "n": len(valid),
        })
    return rows


def h_stats(latents, K_total):
    H = torch.cat([item["h"] for item in latents], dim=0).float()  # [N, K, H]
    per_pos_norm = H.norm(dim=-1).mean(dim=0)
    H_norm = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cos_mats = []
    for n in range(H.shape[0]):
        cos_mats.append(H_norm[n] @ H_norm[n].T)
    cos_mean = torch.stack(cos_mats, dim=0).mean(dim=0)
    if K_total > 1:
        mask = ~torch.eye(K_total, dtype=torch.bool)
        avg = float(cos_mean[mask].mean().item())
    else:
        avg = float("nan")
    return {
        "n": int(H.shape[0]),
        "per_pos_mean_norm": per_pos_norm.tolist(),
        "pairwise_cosine_mean": cos_mean.tolist(),
        "avg_off_diag_cosine": avg,
    }


def compute_metrics(rows: list[dict], K: int) -> dict:
    """compression_ratio, n_helpful, qwen_base utility, etc."""
    by_mode = {r["mode"]: r["nll_mean"] for r in rows}
    none_nll = by_mode.get("none", float("nan"))
    all_nll = by_mode.get("all", float("nan"))
    first_half = by_mode.get("first_half", float("nan"))
    last_half = by_mode.get("last_half", float("nan"))
    first_only = by_mode.get("first_only", float("nan"))
    half_avg = (first_half + last_half) / 2
    util = none_nll - all_nll  # >0 means latents help reader
    compression = (half_avg - all_nll) / (first_only - all_nll) if (first_only - all_nll) > 1e-6 else float("nan")
    n_helpful = 0
    for i in range(K):
        m = by_mode.get(f"only_pos_{i}", float("nan"))
        if (none_nll - m) >= 0.05:
            n_helpful += 1
    return {
        "compression_ratio": float(compression),
        "n_helpful": n_helpful,
        "utility": float(util),
        "none_nll": float(none_nll),
        "all_nll": float(all_nll),
        "first_half_nll": float(first_half),
        "last_half_nll": float(last_half),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="space-separated <name>:<path> pairs")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--qwen_base", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--K", type=int, default=LATENT_SIZE)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    parsed_ckpts = []
    for spec in args.ckpts:
        if ":" not in spec:
            raise ValueError(f"--ckpts entry must be name:path, got {spec!r}")
        name, path = spec.split(":", 1)
        parsed_ckpts.append((name, Path(path)))

    results = {}
    for name, ckpt_path in parsed_ckpts:
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        latents_dir = out_dir / "latents"
        latents_dir.mkdir(parents=True, exist_ok=True)

        ckpt_type = detect_ckpt_type(ckpt_path)
        print(f"\n=== {name} ({ckpt_type}) {ckpt_path} ===", flush=True)

        if ckpt_type == "monet":
            latents = extract_monet_latents(ckpt_path, n=args.n_eval, K=args.K)
        else:
            latents = extract_pivot_latents(ckpt_path, n=args.n_eval, K=args.K)

        # Save latents.
        for item in latents:
            torch.save(item, latents_dir / f"latent_{item['sample_id']:08d}.pt")

        # h-stats.
        hs = h_stats(latents, args.K) if latents else {}
        with open(out_dir / "h_stats.jsonl", "w") as f:
            f.write(json.dumps(hs) + "\n")

        # Evaluate with self-reader.
        print(f"\n[eval-self] {name}", flush=True)
        proc_self = AutoProcessor.from_pretrained(str(ckpt_path), use_fast=True, trust_remote_code=True)
        if ckpt_type == "monet":
            for tk in ["<abs_vis_token_pad>", "<abs_vis_token>", "</abs_vis_token>", "<observation>", "</observation>"]:
                proc_self.tokenizer.add_tokens(tk, special_tokens=True)
        model_self = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(ckpt_path), torch_dtype=torch.bfloat16)
        model_self.resize_token_embeddings(len(proc_self.tokenizer))
        model_self.config.vocab_size = len(proc_self.tokenizer)
        for p in model_self.parameters():
            p.requires_grad_(False)
        model_self.eval().cuda()
        rows_self = run_reader_eval(model_self, proc_self.tokenizer, model_self.config.image_token_id, latents, args.K)
        for r in rows_self:
            r["reader"] = "self"
        del model_self, proc_self
        gc.collect(); torch.cuda.empty_cache()

        # Evaluate with qwen_base reader.
        print(f"\n[eval-qwen_base] {name}", flush=True)
        proc_qb = AutoProcessor.from_pretrained(args.qwen_base, use_fast=True)
        model_qb = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.qwen_base, torch_dtype=torch.bfloat16)
        for p in model_qb.parameters():
            p.requires_grad_(False)
        model_qb.eval().cuda()
        rows_qb = run_reader_eval(model_qb, proc_qb.tokenizer, model_qb.config.image_token_id, latents, args.K)
        for r in rows_qb:
            r["reader"] = "qwen_base"
        del model_qb, proc_qb
        gc.collect(); torch.cuda.empty_cache()

        all_rows = rows_self + rows_qb
        with open(out_dir / "results.jsonl", "w") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")

        m_self = compute_metrics(rows_self, args.K)
        m_qb = compute_metrics(rows_qb, args.K)
        results[name] = {"self": m_self, "qwen_base": m_qb, "h_stats": hs, "ckpt_type": ckpt_type}

        # Per-ckpt REPORT.md
        with open(out_dir / "REPORT.md", "w") as f:
            f.write(f"# {name}\n\n")
            f.write(f"- ckpt: `{ckpt_path}`\n")
            f.write(f"- type: {ckpt_type}\n")
            f.write(f"- n_eval: {hs.get('n', 0)}\n")
            f.write(f"- mean off-diag cos: {hs.get('avg_off_diag_cosine', float('nan')):.3f}\n\n")
            f.write("## Self-reader metrics\n")
            for k, v in m_self.items():
                f.write(f"- {k}: {v:.4f}\n")
            f.write("\n## qwen_base-reader metrics\n")
            for k, v in m_qb.items():
                f.write(f"- {k}: {v:.4f}\n")
        print(f"[report] wrote {out_dir / 'REPORT.md'}", flush=True)

    # Cross report.
    cross_path = out_root / "CROSS_REPORT.md"
    with open(cross_path, "w") as f:
        f.write("# Phase 3 cross-checkpoint report\n\n")
        f.write("| ckpt | type | n | mean off-diag cos | comp_ratio (qwen_base) | n_helpful (qwen_base) | qwen_base utility | self utility |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for name, r in results.items():
            f.write(
                f"| {name} | {r['ckpt_type']} | {r['h_stats'].get('n',0)} | "
                f"{r['h_stats'].get('avg_off_diag_cosine', float('nan')):.3f} | "
                f"{r['qwen_base']['compression_ratio']:.3f} | "
                f"{r['qwen_base']['n_helpful']} | "
                f"{r['qwen_base']['utility']:+.3f} | "
                f"{r['self']['utility']:+.3f} |\n"
            )
    print(f"\n[cross-report] wrote {cross_path}", flush=True)


if __name__ == "__main__":
    main()
