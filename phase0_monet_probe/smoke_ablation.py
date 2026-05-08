"""Smoke test for ablation forward — single example, both readers."""
import importlib.util, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("LATENT_START_ID", "151666")
os.environ.setdefault("LATENT_END_ID", "151667")
os.environ.setdefault("LATENT_SIZE", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_AUTO_DOCSTRING", "1")
assert "transformers" not in sys.modules

_patch = ROOT / "monet_model" / "modeling_qwen2_5_vl_monet.py"
_spec = importlib.util.spec_from_file_location(
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", str(_patch)
)
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
sys.modules["transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"] = _mod

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(ROOT))
from ablation import ablation_modes, apply_ablation, forward_anchor

print("loading 1 latent...")
d = torch.load("latents/stage2/Visual_CoT/latent_00116591.pt", weights_only=False)
h = d["latent"][:8].unsqueeze(0)  # [1, 8, H]
q = d["question"]; a = d["answer"]
print("q=", repr(q)); print("a=", repr(a))

# === Monet self ===
print("\nloading Monet stage2...")
ckpt = "checkpoints/Monet-SFT-7B/stage2"
proc = AutoProcessor.from_pretrained(ckpt, use_fast=True, trust_remote_code=True)
for tok in ["<abs_vis_token_pad>", "<abs_vis_token>", "</abs_vis_token>", "<observation>", "</observation>"]:
    proc.tokenizer.add_tokens(tok, special_tokens=True)
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
m.resize_token_embeddings(len(proc.tokenizer))
m.config.vocab_size = len(proc.tokenizer)
for p in m.parameters():
    p.requires_grad_(False)
m.eval().cuda()
print("monet image_token_id =", m.config.image_token_id)

modes = ablation_modes(8)
print("\n--- Monet self reader ---")
device = next(m.parameters()).device
h_dev = h.to(device)
for name in ["all", "none", "first_only", "last_only", "first_half", "last_half", "only_pos_0", "only_pos_7"]:
    h_ab = apply_ablation(h_dev, modes[name])
    nll = forward_anchor(m, proc.tokenizer, m.config.image_token_id, h_ab, [q], [a], 8, is_monet=True)
    print(f"  {name:14s} nll={nll:.3f}")

del m
torch.cuda.empty_cache()
import gc; gc.collect()

print("\nloading Qwen base...")
qm = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16)
qp = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
for p in qm.parameters():
    p.requires_grad_(False)
qm.eval().cuda()
print("qwen image_token_id =", qm.config.image_token_id)

print("\n--- Qwen base reader ---")
device = next(qm.parameters()).device
h_dev = h.to(device)
for name in ["all", "none", "first_only", "last_only", "first_half", "last_half", "only_pos_0", "only_pos_7"]:
    h_ab = apply_ablation(h_dev, modes[name])
    nll = forward_anchor(qm, qp.tokenizer, qm.config.image_token_id, h_ab, [q], [a], 8, is_monet=False)
    print(f"  {name:14s} nll={nll:.3f}")

print("\nDONE")
