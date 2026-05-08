"""Smoke test: monkey-patch + load Monet stage2; verify the patched model works
and exposes latent_mode/latent_embeds path.

Run from phase0_monet_probe with:
  .venv-monet/bin/python smoke_load_model.py
"""
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

# Now import and load the model.
import torch
from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

MODEL = ROOT / "checkpoints" / "Monet-SFT-7B" / "stage2"
print("loading processor...")
proc = AutoProcessor.from_pretrained(str(MODEL), use_fast=True, trust_remote_code=True)
for tok in ["<abs_vis_token_pad>", "<abs_vis_token>", "</abs_vis_token>", "<observation>", "</observation>"]:
    proc.tokenizer.add_tokens(tok, special_tokens=True)
print("vocab size:", len(proc.tokenizer))
print("config:")
cfg = Qwen2_5_VLConfig.from_pretrained(str(MODEL))
print("  hidden_size:", getattr(cfg, "hidden_size", None) or getattr(cfg.text_config, "hidden_size", None))
print("loading model bf16 → cuda...")
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(MODEL), config=cfg, torch_dtype=torch.bfloat16)
m.cuda().eval()
print("model class:", type(m).__module__, type(m).__name__)
# Confirm forward signature contains latent_mode (the monkey-patched method).
import inspect
sig = inspect.signature(m.forward)
print("forward kwargs:", [k for k in sig.parameters if "latent" in k or "alignment" in k])
assert "latent_mode" in sig.parameters, "latent_mode missing — monkey-patch did NOT take effect."
assert "output_latent_embeds" in sig.parameters, "output_latent_embeds missing"
print("OK — monkey-patch active.")
print("free GPU mem (MB):", torch.cuda.mem_get_info()[0] / (1024**2))
