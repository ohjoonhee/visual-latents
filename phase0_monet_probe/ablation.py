"""Compression ablation on Monet-extracted latents.

For each saved latent file (latent_*.pt produced by extract_latents.py), we:
  - take the FIRST K=8 positions (latents from the first auxiliary image — matches
    the user's K_total=8 setup),
  - splice them as inputs_embeds into K=8 <|image_pad|> slots in a text-only
    Qwen2.5-VL prompt,
  - score answer-span CE under each ablation mode (zero out positions outside
    `keep_idx`),
  - report nll_mean per (mode, anchor, subset, stage).

Anchors:
  - "monet_self": the same Monet stage{2,3} checkpoint used to extract the latent,
    used as a frozen reader (no latent_mode; pure text forward with spliced embeds).
  - "qwen_base":  fresh Qwen/Qwen2.5-VL-7B-Instruct, also frozen — transfer test.

Output:
  - results.jsonl, one row per (mode, anchor, stage, subset).
  - h_stats.jsonl, one row per (anchor, stage, subset).
"""
from __future__ import annotations

# ===== monkey-patch BEFORE any transformers import =====
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

os.environ.setdefault("LATENT_START_ID", "151666")
os.environ.setdefault("LATENT_END_ID", "151667")
os.environ.setdefault("LATENT_SIZE", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_AUTO_DOCSTRING", "1")

assert "transformers" not in sys.modules, "FATAL: transformers already imported."

_patch_path = ROOT / "monet_model" / "modeling_qwen2_5_vl_monet.py"
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

CKPT_ROOT = ROOT / "checkpoints" / "Monet-SFT-7B"
LATENT_SIZE = int(os.environ["LATENT_SIZE"])  # 8
K = LATENT_SIZE


# ---------------- ablation modes ----------------

def ablation_modes(K_total: int = K) -> dict[str, list[int]]:
    """Mirror src/vl/interleaved/trainer.py:_ablation_modes (single-block flavour:
    T_blocks=1, k_latent=K)."""
    half = K_total // 2
    modes: dict[str, list[int]] = {
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
    """h: [B, K, H]. Zero positions outside keep_idx."""
    out = torch.zeros_like(h)
    if keep_idx:
        idx_t = torch.tensor(keep_idx, device=h.device, dtype=torch.long)
        out[:, idx_t, :] = h[:, idx_t, :]
    return out


# ---------------- prompt construction (matches src/vl/readers.py) ----------------

def build_prompt_no_answer(K_total: int, question: str) -> str:
    img_pads = "<|image_pad|>" * K_total
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|vision_start|>{img_pads}<|vision_end|>{question}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_prompt_full(K_total: int, question: str, answer: str) -> str:
    return build_prompt_no_answer(K_total, question) + answer + "<|im_end|>"


# ---------------- reader forward ----------------

@torch.no_grad()
def forward_anchor(
    model,
    tokenizer,
    image_token_id: int,
    h: torch.Tensor,                # [B, K, H]
    questions: list[str],
    answers: list[str],
    K_total: int = K,
    is_monet: bool = False,  # retained for clarity, but BOTH readers use the patched forward
) -> float:
    """Mean answer-span CE for a batch. Returns Python float.

    The monkey-patch is process-global, so both Monet stage2/3 AND the
    base Qwen2.5-VL-7B-Instruct go through the patched forward. The patched
    forward gates logits on `"ce" in loss_type`, so we always pass
    `loss_type=["ce"]` + `labels` and read `outputs.loss`.
    """
    if h.shape[1] != K_total:
        raise ValueError(f"h.shape[1]={h.shape[1]} but K={K_total}")
    B = h.shape[0]
    device = next(model.parameters()).device

    no_answer_ids: list[list[int]] = []
    full_ids: list[list[int]] = []
    for q, a in zip(questions, answers, strict=True):
        prompt_no_a = build_prompt_no_answer(K_total, q)
        prompt_full = build_prompt_full(K_total, q, a)
        ids_no_a = tokenizer(prompt_no_a, add_special_tokens=False).input_ids
        ids_full = tokenizer(prompt_full, add_special_tokens=False).input_ids
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

    img_pad_count = (input_ids == image_token_id).sum(dim=1)
    if not torch.all(img_pad_count == K_total):
        raise RuntimeError(f"Expected K={K_total} image_pad/row; got {img_pad_count.tolist()}")

    base_embeds = model.get_input_embeddings()(input_ids)  # [B, L, D] bf16
    h_cast = h.to(device=device, dtype=base_embeds.dtype)
    inputs_embeds = base_embeds.clone()
    for b in range(B):
        positions = (input_ids[b] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
        inputs_embeds[b, positions, :] = h_cast[b]

    # Both readers use the patched forward (the monkey-patch is process-global).
    outputs = model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        loss_type=["ce"],
        use_cache=False,
    )
    return float(outputs.loss.item())


# ---------------- per-(reader × stage × subset) eval ----------------

def load_h_cache(latent_dir: Path, K_total: int = K, max_examples: Optional[int] = None):
    """Yield (h:[1,K,H] bf16 cpu, question, answer, sample_id) from latent_*.pt."""
    files = sorted(latent_dir.glob("latent_*.pt"))
    if max_examples is not None:
        files = files[:max_examples]
    out = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        latent = d["latent"]  # [num_latents, H]
        if latent.shape[0] < K_total:
            # Skip examples that don't have a full K=8 block.
            continue
        # Take the FIRST K positions (matches user's single-block, K_total=8 setup).
        h = latent[:K_total].unsqueeze(0)  # [1, K, H]
        out.append((h, str(d["question"]), str(d["answer"]), int(d["sample_id"])))
    return out


def run_reader_eval(
    *,
    reader_name: str,
    model,
    tokenizer,
    image_token_id: int,
    h_cache: list,
    K_total: int = K,
    rows_acc: list,
    h_stats_acc: list,
    stage: str,
    subset: str,
    n_examples: int,
    is_monet: bool,
):
    """Run all ablation modes on the given reader; append rows to accumulators."""
    if not h_cache:
        return

    modes = ablation_modes(K_total)
    print(f"[ablate] reader={reader_name}  stage={stage}  subset={subset}  n={len(h_cache)}", flush=True)

    # nll_per_example for this (reader, stage, subset, mode)
    cache = {name: [] for name in modes}
    started = time.time()
    for i, (h, q, a, sid) in enumerate(h_cache):
        # Move h to device once per example, then iterate modes.
        device = next(model.parameters()).device
        h_dev = h.to(device=device)
        for name, keep_idx in modes.items():
            h_ab = apply_ablation(h_dev, keep_idx)
            try:
                nll = forward_anchor(
                    model, tokenizer, image_token_id, h_ab, [q], [a], K_total,
                    is_monet=is_monet,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                nll = float("nan")
            cache[name].append(nll)
        if (i + 1) % 10 == 0 or i == 0:
            dt = time.time() - started
            print(f"   [{reader_name}] {i+1}/{len(h_cache)}  elapsed={dt:.1f}s", flush=True)

    for name, vals in cache.items():
        valid = [v for v in vals if v == v]  # drop NaNs
        rows_acc.append({
            "reader": reader_name,
            "stage": stage,
            "subset": subset,
            "mode": name,
            "keep": modes[name],
            "nll_mean": (sum(valid) / max(1, len(valid))) if valid else float("nan"),
            "nll_per_example": vals,
            "n": len(valid),
        })

    # h-stats: per-position norm and pairwise-cosine matrix (over the same h_cache).
    # We compute this once per (stage, subset) — it doesn't depend on the reader, so
    # only emit when reader_name == "monet_self" (called first).
    if reader_name == "monet_self":
        H = torch.cat([h for h, _, _, _ in h_cache], dim=0).float()  # [N, K, H]
        per_pos_norm = H.norm(dim=-1).mean(dim=0)  # [K]
        H_norm = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cos_mats = []
        for n in range(H.shape[0]):
            cos_mats.append(H_norm[n] @ H_norm[n].T)
        cos_mean = torch.stack(cos_mats, dim=0).mean(dim=0)  # [K, K]
        if K_total > 1:
            mask = ~torch.eye(K_total, dtype=torch.bool)
            avg_off_diag = float(cos_mean[mask].mean().item())
        else:
            avg_off_diag = float("nan")
        h_stats_acc.append({
            "stage": stage,
            "subset": subset,
            "n": int(H.shape[0]),
            "per_pos_mean_norm": per_pos_norm.tolist(),
            "pairwise_cosine_mean": cos_mean.tolist(),
            "avg_off_diag_cosine": avg_off_diag,
        })


# ---------------- driver ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents_root", type=str, default=str(ROOT / "latents"))
    ap.add_argument("--out_results", type=str, default=str(ROOT / "results.jsonl"))
    ap.add_argument("--out_hstats", type=str, default=str(ROOT / "h_stats.jsonl"))
    ap.add_argument("--qwen_base", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_per_cell", type=int, default=100)
    ap.add_argument("--stages", nargs="+", default=["stage2", "stage3"])
    ap.add_argument("--subsets", nargs="+", default=["Visual_CoT", "Zebra_CoT_count"])
    args = ap.parse_args()

    rows_acc: list[dict] = []
    h_stats_acc: list[dict] = []
    latents_root = Path(args.latents_root)

    # Plan: for each (stage, subset), load h_cache once, then evaluate
    # (a) Monet stage{2,3} as reader (load once per stage)
    # (b) Qwen2.5-VL-7B-Instruct as reader (load once globally)
    # We load Monet last per stage, then unload before next stage to keep VRAM headroom
    # since loading Qwen base + Monet at once would be ~32 GB bf16.

    # Strategy: load monet_self stage S, run both subsets; then unload; load qwen base, run both subsets × both stages; done.

    # Pass 1: Monet self-readers
    for stage in args.stages:
        ckpt = CKPT_ROOT / stage
        if not ckpt.exists():
            print(f"WARN skip stage={stage}: missing ckpt {ckpt}")
            continue
        print(f"\n[ablate] loading Monet {stage} as reader…", flush=True)
        proc = AutoProcessor.from_pretrained(str(ckpt), use_fast=True, trust_remote_code=True)
        for tok in [
            "<abs_vis_token_pad>", "<abs_vis_token>", "</abs_vis_token>",
            "<observation>", "</observation>",
        ]:
            proc.tokenizer.add_tokens(tok, special_tokens=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(ckpt), torch_dtype=torch.bfloat16,
        )
        model.resize_token_embeddings(len(proc.tokenizer))
        model.config.vocab_size = len(proc.tokenizer)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval().cuda()
        image_token_id = model.config.image_token_id

        for subset in args.subsets:
            h_cache = load_h_cache(latents_root / stage / subset, max_examples=args.max_per_cell)
            run_reader_eval(
                reader_name="monet_self",
                model=model, tokenizer=proc.tokenizer, image_token_id=image_token_id,
                h_cache=h_cache,
                rows_acc=rows_acc, h_stats_acc=h_stats_acc,
                stage=stage, subset=subset, n_examples=args.max_per_cell,
                is_monet=True,
            )

        # Free.
        del model, proc
        gc.collect()
        torch.cuda.empty_cache()

    # Pass 2: Qwen2.5-VL-7B-Instruct (frozen base) as reader. We load once.
    print(f"\n[ablate] loading Qwen base reader: {args.qwen_base}", flush=True)
    qwen_proc = AutoProcessor.from_pretrained(args.qwen_base, use_fast=True)
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_base, torch_dtype=torch.bfloat16,
    )
    for p in qwen_model.parameters():
        p.requires_grad_(False)
    qwen_model.eval().cuda()
    qwen_image_token_id = qwen_model.config.image_token_id

    for stage in args.stages:
        for subset in args.subsets:
            h_cache = load_h_cache(latents_root / stage / subset, max_examples=args.max_per_cell)
            run_reader_eval(
                reader_name="qwen_base",
                model=qwen_model, tokenizer=qwen_proc.tokenizer,
                image_token_id=qwen_image_token_id,
                h_cache=h_cache,
                rows_acc=rows_acc, h_stats_acc=h_stats_acc,
                stage=stage, subset=subset, n_examples=args.max_per_cell,
                is_monet=False,
            )

    del qwen_model, qwen_proc
    gc.collect(); torch.cuda.empty_cache()

    # Save.
    with open(args.out_results, "w") as f:
        for row in rows_acc:
            f.write(json.dumps(row) + "\n")
    with open(args.out_hstats, "w") as f:
        for row in h_stats_acc:
            f.write(json.dumps(row) + "\n")

    print(f"\nwrote {len(rows_acc)} rows → {args.out_results}")
    print(f"wrote {len(h_stats_acc)} rows → {args.out_hstats}")


if __name__ == "__main__":
    main()
