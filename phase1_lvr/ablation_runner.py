"""Phase 1 compression ablation, mirroring phase0_monet_probe/ablation.py.

Two readers:
  - phase1_self: the trained Phase 1 checkpoint, frozen, used as reader
  - qwen_base:   fresh Qwen/Qwen2.5-VL-3B-Instruct, frozen — transfer test
                 (matches the Phase 0 'qwen_base' anchor, but at 3B to match
                 our trained model)

For each (reader, mode), splice the K hidden states at K <|image_pad|> slots in
a text-only prompt and score answer-span CE.

Outputs:
  - results/ablation_results.jsonl  (one row per reader × mode)
  - results/ablation_h_stats.jsonl  (one row total: per-pos norms + 8x8 cosine)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent


def ablation_modes(K_total: int = 8) -> dict[str, list[int]]:
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
    out = torch.zeros_like(h)
    if keep_idx:
        idx_t = torch.tensor(keep_idx, device=h.device, dtype=torch.long)
        out[:, idx_t, :] = h[:, idx_t, :]
    return out


def build_prompt_no_answer(K: int, question: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|vision_start|>{'<|image_pad|>' * K}<|vision_end|>{question}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_prompt_full(K: int, question: str, answer: str) -> str:
    return build_prompt_no_answer(K, question) + answer + "<|im_end|>"


@torch.no_grad()
def forward_anchor(model, tokenizer, image_token_id, h, question, answer, K=8):
    device = next(model.parameters()).device
    prefix = build_prompt_no_answer(K, question)
    full = build_prompt_full(K, question, answer)
    ids_no_a = tokenizer(prefix, add_special_tokens=False).input_ids
    ids_full = tokenizer(full, add_special_tokens=False).input_ids
    if ids_full[: len(ids_no_a)] != ids_no_a:
        ids_a = tokenizer(answer + "<|im_end|>", add_special_tokens=False).input_ids
        ids_full = ids_no_a + ids_a
    L = len(ids_full)
    input_ids = torch.tensor([ids_full], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[0, len(ids_no_a):] = input_ids[0, len(ids_no_a):]

    pad_count = (input_ids == image_token_id).sum().item()
    if pad_count != K:
        raise RuntimeError(f"expected {K} image_pad, got {pad_count}")

    base_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds = base_embeds.clone()
    pad_pos = (input_ids[0] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
    inputs_embeds[0, pad_pos, :] = h.to(device=device, dtype=inputs_embeds.dtype)[0]

    out = model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )
    return float(out.loss.item())


def load_h_cache(latent_dir: Path, K=8, max_examples=None):
    files = sorted(latent_dir.glob("latent_*.pt"))
    if max_examples is not None:
        files = files[:max_examples]
    out = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        latent = d["latent"]
        if latent.shape[0] < K:
            continue
        h = latent[:K].unsqueeze(0)
        out.append((h, str(d["question"]), str(d["answer"]), int(d["sample_id"])))
    return out


def run_reader(*, reader_name, model, tokenizer, image_token_id, h_cache, K, rows_acc, h_stats_acc, emit_h_stats):
    if not h_cache:
        return
    modes = ablation_modes(K)
    print(f"[ablate] reader={reader_name}  n={len(h_cache)}", flush=True)
    cache = {name: [] for name in modes}
    t0 = time.time()
    for i, (h, q, a, sid) in enumerate(h_cache):
        device = next(model.parameters()).device
        h_dev = h.to(device=device)
        for name, keep_idx in modes.items():
            h_ab = apply_ablation(h_dev, keep_idx)
            try:
                nll = forward_anchor(model, tokenizer, image_token_id, h_ab, q, a, K)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                nll = float("nan")
            cache[name].append(nll)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"   [{reader_name}] {i+1}/{len(h_cache)} elapsed={time.time()-t0:.0f}s", flush=True)

    for name, vals in cache.items():
        valid = [v for v in vals if v == v]
        rows_acc.append({
            "reader": reader_name,
            "mode": name,
            "keep": modes[name],
            "nll_mean": (sum(valid) / len(valid)) if valid else float("nan"),
            "nll_per_example": vals,
            "n": len(valid),
        })

    if emit_h_stats:
        H = torch.cat([h for h, _, _, _ in h_cache], dim=0).float()
        per_pos_norm = H.norm(dim=-1).mean(dim=0)
        H_norm = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cos_mats = []
        for n in range(H.shape[0]):
            cos_mats.append(H_norm[n] @ H_norm[n].T)
        cos_mean = torch.stack(cos_mats, dim=0).mean(dim=0)
        if K > 1:
            mask = ~torch.eye(K, dtype=torch.bool)
            avg_off_diag = float(cos_mean[mask].mean().item())
        else:
            avg_off_diag = float("nan")
        h_stats_acc.append({
            "n": int(H.shape[0]),
            "per_pos_mean_norm": per_pos_norm.tolist(),
            "pairwise_cosine_mean": cos_mean.tolist(),
            "avg_off_diag_cosine": avg_off_diag,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents_dir", type=str, required=True)
    ap.add_argument("--phase1_ckpt", type=str, required=True)
    ap.add_argument("--qwen_base", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max_examples", type=int, default=200)
    ap.add_argument("--out_results", type=str, required=True)
    ap.add_argument("--out_hstats", type=str, required=True)
    args = ap.parse_args()

    rows_acc = []
    h_stats_acc = []

    h_cache = load_h_cache(Path(args.latents_dir), K=args.K, max_examples=args.max_examples)
    print(f"[ablate] loaded {len(h_cache)} latent files", flush=True)

    # Pass 1: phase1_self
    print(f"[ablate] loading phase1_self {args.phase1_ckpt}", flush=True)
    proc1 = AutoProcessor.from_pretrained(args.phase1_ckpt, use_fast=True, trust_remote_code=True)
    m1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.phase1_ckpt, torch_dtype=torch.bfloat16)
    for p in m1.parameters():
        p.requires_grad_(False)
    m1.eval().cuda()
    img_id1 = m1.config.image_token_id
    run_reader(reader_name="phase1_self", model=m1, tokenizer=proc1.tokenizer, image_token_id=img_id1,
               h_cache=h_cache, K=args.K, rows_acc=rows_acc, h_stats_acc=h_stats_acc, emit_h_stats=True)
    del m1, proc1
    gc.collect(); torch.cuda.empty_cache()

    # Pass 2: qwen_base
    print(f"[ablate] loading qwen_base {args.qwen_base}", flush=True)
    proc2 = AutoProcessor.from_pretrained(args.qwen_base, use_fast=True)
    m2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.qwen_base, torch_dtype=torch.bfloat16)
    for p in m2.parameters():
        p.requires_grad_(False)
    m2.eval().cuda()
    img_id2 = m2.config.image_token_id
    run_reader(reader_name="qwen_base", model=m2, tokenizer=proc2.tokenizer, image_token_id=img_id2,
               h_cache=h_cache, K=args.K, rows_acc=rows_acc, h_stats_acc=h_stats_acc, emit_h_stats=False)
    del m2, proc2
    gc.collect(); torch.cuda.empty_cache()

    with open(args.out_results, "w") as f:
        for r in rows_acc:
            f.write(json.dumps(r) + "\n")
    with open(args.out_hstats, "w") as f:
        for r in h_stats_acc:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows_acc)} rows → {args.out_results}")
    print(f"wrote {len(h_stats_acc)} rows → {args.out_hstats}")


if __name__ == "__main__":
    main()
