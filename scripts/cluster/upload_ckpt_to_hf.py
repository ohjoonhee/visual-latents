"""Upload a training checkpoint dir to HuggingFace Hub with proper card + tag.

Mirrors the conventions established by `scripts/cluster/push_viscot_to_hub.py`:
auth via HF_TOKEN env (or `huggingface-cli login` cache), `[upload]` log
prefix, fail-fast on auth.

Workflow:
  1. Validate ckpt_dir contains the expected weight files.
  2. (Optionally) load the training config + last training_log.jsonl row to
     populate the model card body and the tag message.
  3. Generate README.md (model card with YAML frontmatter — base_model,
     datasets, library_name, pipeline_tag, tags) and write it INTO ckpt_dir.
  4. create_repo(public, exist_ok=True).
  5. upload_large_folder (resumable, multi-thread, dedup via hf_xet).
     Excludes optimizer/rng_state files (inference-only repo).
  6. create_tag(f"step-{step}", tag_message=<one-line summary>).

Naming convention (per project decision 2026-05-14):
  {project}-{base}-{stage}[-{variant}][-{recipe}][-{date|version}]
  Default repo: ohjoonhee/vlatents-qwen25vl7b-stage2-repro-v1
  - "vlatents" = project; "qwen25vl7b" = base; "stage2-repro-v1" = recipe.
  - Step number lives in TAGS, not the repo name (avoids near-duplicate repos).

Usage (cluster):
  uv --project cluster run python scripts/cluster/upload_ckpt_to_hf.py \\
    --ckpt-dir /data/joonhee/visual-latents/cluster_phase3/stage2_baseline/checkpoint_step1500 \\
    --step 1500 \\
    --config cluster/configs/stage2_monet.yaml \\
    --notes "First Stage 2 reproduction; mid-training step 1500 of 2000"

Auth: requires HF_TOKEN with WRITE access in env (sourced via .env in the sb
wrapper). Verifies via whoami() up front.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi


DEFAULT_REPO = "ohjoonhee/vlatents-qwen25vl7b-stage2-repro-v1"

# Files that are training-only state (optimizer, rng, etc.) — useless for
# inference and bulky enough to bloat the repo. Excluded from upload.
INFERENCE_IGNORE = [
    "optimizer*.pt",
    "**/optimizer*.pt",
    "rng_state*.pth",
    "**/rng_state*.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "*.tmp",
    "*.lock",
]


def _last_training_log_row(ckpt_dir: Path) -> dict | None:
    """Read training_log.jsonl from ckpt_dir's parent (the run out_dir).

    Returns the last logged row (dict) or None if not found / malformed.
    The trainer writes one JSON-per-line row at every log_every interval.
    """
    log_path = ckpt_dir.parent / "training_log.jsonl"
    if not log_path.exists():
        return None
    last = None
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def _build_card(
    repo: str,
    step: int,
    cfg: dict | None,
    last_log: dict | None,
    notes: str | None,
) -> str:
    """Render a concise model card (YAML frontmatter + ~30 lines body)."""
    base_model = (cfg or {}).get("base_model", "Qwen/Qwen2.5-VL-7B-Instruct")
    init_ckpt = (cfg or {}).get("init_ckpt", "(none)")
    dataset_id = "ohjoonhee/visual-cot-50k-poc"

    recipe_lines = []
    if cfg is not None:
        for k in (
            "latent_size", "alignment_weight", "emphasize_latent_weight",
            "ce_emphasize_factor", "alignment_layer", "use_attn_mask_4d",
            "lr", "weight_decay", "warmup_steps", "max_steps",
            "grad_accum_steps", "max_pixels",
        ):
            if k in cfg:
                recipe_lines.append(f"- `{k}`: {cfg[k]}")

    log_line = ""
    if last_log is not None:
        bits = []
        for k in ("step", "ce_loss", "align_loss", "total_loss"):
            if k in last_log:
                v = last_log[k]
                v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                bits.append(f"{k}={v_str}")
        if "elapsed_s" in last_log:
            bits.append(f"elapsed={last_log['elapsed_s']:.0f}s")
        if bits:
            log_line = "Last logged training row: " + ", ".join(bits)

    today = datetime.date.today().isoformat()
    recipe_block = "\n".join(recipe_lines) if recipe_lines else "- (no config available)"
    notes_block = ("### Notes\n" + notes) if notes else ""
    log_block = log_line if log_line else "No training log row available."
    repo_short = repo.split("/")[-1]
    max_steps = (cfg or {}).get("max_steps", "?")

    card = f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model: {base_model}
datasets:
  - {dataset_id}
tags:
  - visual-cot
  - monet
  - vlatents
  - stage2
  - research-checkpoint
---

# {repo_short}

**One-line identity:** Stage 2 reproduction of Monet-style visual-CoT training on
`{base_model}`. Initialised from a Stage 1 SFT base; trained on the
`{dataset_id}` dataset.

## Recipe

- Stage: 2 (post-SFT, alignment + emphasized-CE objective)
- Base model: `{base_model}`
- Init checkpoint: `{init_ckpt}`
- Dataset: `{dataset_id}` (Monet-SFT-125K Visual_CoT subset, eval-200 excluded)
- Hardware: 4× H100 80GB, DeepSpeed ZeRO-2 + CPU optim offload, bf16
{recipe_block}

### Known deviations from the Monet paper
1. `emphasize_latent_weight` applied as a plain scalar add (paper uses
   latent-only backprop locality via `compute_latents_only_loss`).
   Effective: `total = ce + (alignment_weight + emphasize_latent_weight) * align`.
2. `attention_mask_4d` is hand-rolled in `mask_utils.build_monet_4d_attn`
   with `latent_cross_isolate=True`. Verified equivalent on tested cases
   (see `phase1_5b_attn/MASK_VALIDATION.md`) but not byte-identical to upstream.
3. Inline teacher forward (not offline-precomputed). Functionally
   equivalent if teacher checkpoint is the same; saves precompute storage.

## This revision (`step-{step}`)

{log_block}

{notes_block}

Other revisions: see the **revisions** dropdown on this page (`step-500`,
`step-1000`, etc., as available).

## How to load

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
m = AutoModelForVision2Seq.from_pretrained(
    "{repo}", revision="step-{step}", torch_dtype="bfloat16")
p = AutoProcessor.from_pretrained("{repo}", revision="step-{step}")
```

## Limitations

Research checkpoint, eval-only. Mid-training step ({step}/{max_steps}).
Not for production. Recipe deviations from the Monet paper are listed above.

---
Card generated {today} from training_log.jsonl + the run's training config.
"""
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, type=Path,
                    help="Local path to the checkpoint directory (e.g. .../checkpoint_step1500)")
    ap.add_argument("--step", required=True, type=int,
                    help="Training step number for the tag (e.g. 1500)")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"HF repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--config", default=None, type=Path,
                    help="Path to the training config yaml (used to populate the model card)")
    ap.add_argument("--notes", default=None,
                    help="One-line note appended to the tag message and the card")
    ap.add_argument("--private", action="store_true", default=False,
                    help="Push as private (default: public — research artifact)")
    ap.add_argument("--dry-run", action="store_true", default=False,
                    help="Generate + print the card without uploading or tagging")
    args = ap.parse_args()

    ckpt_dir: Path = args.ckpt_dir.resolve()
    if not ckpt_dir.exists():
        print(f"ERROR: ckpt-dir does not exist: {ckpt_dir}")
        return 1

    weight_indices = list(ckpt_dir.glob("model*.safetensors*")) + list(ckpt_dir.glob("pytorch_model*.bin"))
    if not weight_indices:
        print(f"ERROR: ckpt-dir has no model weight files (.safetensors or .bin): {ckpt_dir}")
        return 1
    print(f"[upload] ckpt-dir: {ckpt_dir}")
    print(f"[upload] weight files detected: {len(weight_indices)}")

    cfg = None
    if args.config is not None and args.config.exists():
        with args.config.open() as f:
            cfg = yaml.safe_load(f)
        print(f"[upload] loaded config: {args.config}")

    last_log = _last_training_log_row(ckpt_dir)
    if last_log is not None:
        print(f"[upload] last training_log row: step={last_log.get('step')} "
              f"ce={last_log.get('ce_loss', float('nan')):.3f} "
              f"align={last_log.get('align_loss', float('nan')):.4f}")

    card = _build_card(args.repo, args.step, cfg, last_log, args.notes)
    if args.dry_run:
        print("=== model card (dry-run) ===")
        print(card)
        print("=== end card ===")
        return 0

    api = HfApi()
    try:
        whoami = api.whoami()
        print(f"[upload] authenticated as: {whoami.get('name', '?')} (type={whoami.get('type', '?')})")
    except Exception as e:
        print(f"ERROR: HF auth failed: {e}")
        print("       Either export HF_TOKEN=<write-token> or run `huggingface-cli login`.")
        return 1

    private = bool(args.private)
    print(f"[upload] target repo: {args.repo}  private={private}")
    api.create_repo(args.repo, repo_type="model", private=private, exist_ok=True)
    print(f"[upload] repo ready: https://huggingface.co/{args.repo}")

    readme_path = ckpt_dir / "README.md"
    readme_path.write_text(card)
    print(f"[upload] wrote model card → {readme_path}")

    print(f"[upload] starting upload_large_folder (resumable, hf_xet, multi-thread)...")
    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(ckpt_dir),
        ignore_patterns=INFERENCE_IGNORE,
    )
    print(f"[upload] folder upload complete")

    tag_msg_bits = [f"step={args.step}"]
    if last_log is not None:
        if "ce_loss" in last_log:
            tag_msg_bits.append(f"ce={last_log['ce_loss']:.3f}")
        if "align_loss" in last_log:
            tag_msg_bits.append(f"align={last_log['align_loss']:.4f}")
    if args.notes:
        tag_msg_bits.append(args.notes)
    tag_message = " ".join(tag_msg_bits)
    tag_name = f"step-{args.step}"

    try:
        api.create_tag(args.repo, tag=tag_name, tag_message=tag_message)
        print(f"[upload] created tag {tag_name!r}: {tag_message}")
    except Exception as e:
        print(f"WARN: tag creation failed (probably already exists): {e}")

    print(f"[upload] DONE. https://huggingface.co/{args.repo}/tree/{tag_name}")
    print(f"[upload] Load locally with:")
    print(f"       hf download {args.repo} --revision {tag_name} --local-dir ./ckpt")


if __name__ == "__main__":
    sys.exit(main() or 0)
