"""Phase 1.5 / Phase 2 ablation, mirroring phase0_monet_probe/ablation.py.

Two readers:
  - self: the trained Phase-N checkpoint (Monet-patched class), frozen
  - qwen_base: fresh Qwen/Qwen2.5-VL-3B-Instruct (Monet-patched class), frozen

Both pass loss_type=['ce'] + labels, since the Monet patched forward gates
logits on `'ce' in loss_type` (see phase0_monet_probe/ablation.py).

Output:
  - results/ablation_results.jsonl  (one row per reader × mode)
  - results/ablation_h_stats.jsonl  (one row total)
"""
from __future__ import annotations

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

assert "transformers" not in sys.modules

_patch_path = PHASE0 / "monet_model" / "modeling_qwen2_5_vl_monet.py"
_spec = importlib.util.spec_from_file_location(
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    str(_patch_path),
)
_patched_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_patched_mod)
sys.modules["transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"] = _patched_mod

import argparse
import gc
import json
import time

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def ablation_modes(K: int = 8):
    half = K // 2
    modes = {
        "all": list(range(K)),
        "first_only": [0],
        "last_only": [K - 1],
        "first_half": list(range(half)),
        "last_half": list(range(half, K)),
        "none": [],
    }
    for i in range(K):
        modes[f"only_pos_{i}"] = [i]
    return modes


def apply_ablation(h, keep_idx):
    out = torch.zeros_like(h)
    if keep_idx:
        idx_t = torch.tensor(keep_idx, device=h.device, dtype=torch.long)
        out[:, idx_t, :] = h[:, idx_t, :]
    return out


def build_prompt(K, question, answer=None):
    img_pads = "<|image_pad|>" * K
    base = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|vision_start|>{img_pads}<|vision_end|>{question}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if answer is None:
        return base
    return base + answer + "<|im_end|>"


@torch.no_grad()
def forward_anchor(model, tokenizer, image_token_id, h, q, a, K=8):
    device = next(model.parameters()).device
    prefix = build_prompt(K, q)
    full = build_prompt(K, q, a)
    ids_no_a = tokenizer(prefix, add_special_tokens=False).input_ids
    ids_full = tokenizer(full, add_special_tokens=False).input_ids
    if ids_full[: len(ids_no_a)] != ids_no_a:
        ids_a = tokenizer(a + "<|im_end|>", add_special_tokens=False).input_ids
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
        loss_type=["ce"],
        use_cache=False,
        return_dict=True,
    )
    return float(out.loss.item())


def load_h_cache(latent_dir, K=8, max_examples=None):
    files = sorted(Path(latent_dir).glob("latent_*.pt"))
    if max_examples is not None:
        files = files[:max_examples]
    out = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        latent = d["latent"]
        if latent.shape[0] < K:
            continue
        h = latent[:K].unsqueeze(0)
        v_roi = d.get("v_roi")
        out.append((h, str(d["question"]), str(d["answer"]), int(d["sample_id"]), v_roi))
    return out


def add_special_tokens(processor):
    for tok in [
        "<abs_vis_token_pad>",
        "<abs_vis_token>",
        "</abs_vis_token>",
        "<observation>",
        "</observation>",
    ]:
        processor.tokenizer.add_tokens(tok, special_tokens=True)


def run_reader(*, reader_name, model, tokenizer, image_token_id, h_cache, K, rows_acc, h_stats_acc, emit_h_stats):
    if not h_cache:
        return
    modes = ablation_modes(K)
    print(f"[ablate] reader={reader_name}  n={len(h_cache)}", flush=True)
    cache = {name: [] for name in modes}
    t0 = time.time()
    for i, (h, q, a, sid, _v) in enumerate(h_cache):
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
        H = torch.cat([h for h, _, _, _, _ in h_cache], dim=0).float()
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
        # Also v_roi cosine if present
        v_off_diag = float("nan")
        v_present = [v for _, _, _, _, v in h_cache if v is not None]
        if v_present:
            V = torch.stack(v_present, dim=0).float()  # [N, K, D]
            V_norm = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            v_cos = []
            for n in range(V.shape[0]):
                v_cos.append(V_norm[n] @ V_norm[n].T)
            v_cos_mean = torch.stack(v_cos, dim=0).mean(dim=0)
            if K > 1:
                mask = ~torch.eye(K, dtype=torch.bool)
                v_off_diag = float(v_cos_mean[mask].mean().item())
        h_stats_acc.append({
            "n": int(H.shape[0]),
            "per_pos_mean_norm": per_pos_norm.tolist(),
            "pairwise_cosine_mean": cos_mean.tolist(),
            "avg_off_diag_cosine": avg_off_diag,
            "v_roi_avg_off_diag_cosine": v_off_diag,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents_dir", type=str, required=True)
    ap.add_argument("--self_ckpt", type=str, required=True, help="trained checkpoint (Monet patched class)")
    ap.add_argument("--self_name", type=str, default="phase1_5_self")
    ap.add_argument("--qwen_base", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max_examples", type=int, default=200)
    ap.add_argument("--out_results", type=str, required=True)
    ap.add_argument("--out_hstats", type=str, required=True)
    args = ap.parse_args()

    rows_acc, h_stats_acc = [], []
    h_cache = load_h_cache(Path(args.latents_dir), K=args.K, max_examples=args.max_examples)
    print(f"[ablate] loaded {len(h_cache)} latent files", flush=True)

    # Pass 1: self
    print(f"[ablate] loading {args.self_name} from {args.self_ckpt}", flush=True)
    p1 = AutoProcessor.from_pretrained(args.self_ckpt, use_fast=True, trust_remote_code=True)
    add_special_tokens(p1)
    m1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.self_ckpt, torch_dtype=torch.bfloat16)
    m1.resize_token_embeddings(len(p1.tokenizer))
    m1.config.vocab_size = len(p1.tokenizer)
    for p in m1.parameters():
        p.requires_grad_(False)
    m1.eval().cuda()
    img1 = m1.config.image_token_id
    run_reader(reader_name=args.self_name, model=m1, tokenizer=p1.tokenizer, image_token_id=img1,
               h_cache=h_cache, K=args.K, rows_acc=rows_acc, h_stats_acc=h_stats_acc, emit_h_stats=True)
    del m1, p1
    gc.collect(); torch.cuda.empty_cache()

    # Pass 2: qwen_base
    print(f"[ablate] loading qwen_base {args.qwen_base}", flush=True)
    p2 = AutoProcessor.from_pretrained(args.qwen_base, use_fast=True)
    add_special_tokens(p2)
    m2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.qwen_base, torch_dtype=torch.bfloat16)
    m2.resize_token_embeddings(len(p2.tokenizer))
    m2.config.vocab_size = len(p2.tokenizer)
    for p in m2.parameters():
        p.requires_grad_(False)
    m2.eval().cuda()
    img2 = m2.config.image_token_id
    run_reader(reader_name="qwen_base", model=m2, tokenizer=p2.tokenizer, image_token_id=img2,
               h_cache=h_cache, K=args.K, rows_acc=rows_acc, h_stats_acc=h_stats_acc, emit_h_stats=False)
    del m2, p2
    gc.collect(); torch.cuda.empty_cache()

    with open(args.out_results, "w") as f:
        for r in rows_acc:
            f.write(json.dumps(r) + "\n")
    with open(args.out_hstats, "w") as f:
        for r in h_stats_acc:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows_acc)} rows → {args.out_results}")
    print(f"wrote {len(h_stats_acc)} rows → {args.out_hstats}")


if __name__ == "__main__":
    main()
