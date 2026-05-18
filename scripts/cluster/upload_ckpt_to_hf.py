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

    # Autodetect stage from the config to pick the right template.
    # Stage 2 is identified by `alignment_weight` or `latent_size` (the
    # alignment objective + latent slot K). Stage 1 SFT (NTP-only) has
    # neither. Pivot A variants have `lambda_lvr` or `lambda_reg`.
    has_alignment = bool(cfg) and (
        cfg.get("alignment_weight") is not None
        or cfg.get("latent_size") is not None
    )
    has_lvr = bool(cfg) and (
        cfg.get("lambda_lvr") is not None or cfg.get("lambda_reg") is not None
    )
    if has_alignment:
        stage_kind = "stage2"
    elif has_lvr:
        stage_kind = "pivot"
    else:
        stage_kind = "stage1-sft"

    if stage_kind == "stage2":
        identity = (
            "Stage 2 reproduction of Monet-style visual-CoT training on "
            f"`{base_model}`. Initialised from a Stage 1 SFT base; trained on the "
            f"`{dataset_id}` dataset."
        )
        recipe_heading = "- Stage: 2 (post-SFT, alignment + emphasized-CE objective)"
        deviations_block = """### Fidelity to the Monet paper
1. **Latent-only backprop — paper-faithful (Job C).** `emphasize_latent_weight`
   uses a verbatim port of upstream `compute_latents_only_loss`: the alignment
   loss is computed in the CE forward (where `ce_patch_vec` is spliced into
   `inputs_embeds`) and backpropped ONLY through the latent embeddings, i.e.
   `total = emphasize_latent_weight * compute_latents_only_loss(ce_patch_vec,
   alignment_weight*align) + ce` (mirrors upstream `src/trainer.py:152-224`).
   The earlier plain-scalar-add approximation (see the `*-repro-v1` repo) is
   NOT used here.
2. `attention_mask_4d` is hand-rolled in `mask_utils.build_monet_4d_attn`
   with `latent_cross_isolate=True`. Verified equivalent on tested cases
   (see `phase1_5b_attn/MASK_VALIDATION.md`) but not byte-identical to upstream.
3. Inline teacher forward (not offline-precomputed). Functionally
   equivalent if teacher checkpoint is the same; saves precompute storage."""
        stage_tag = "stage2"
    elif stage_kind == "pivot":
        identity = (
            f"Pivot A variant on `{base_model}`. Trains K latent slots with an "
            "LVR (mean-MSE to ROI image features) objective plus a collapse-"
            f"prevention regularizer. Dataset: `{dataset_id}`."
        )
        recipe_heading = "- Stage: Pivot A (LVR + regularizer; no Monet alignment loss)"
        deviations_block = """### Notes on the recipe
LVR/VICReg recipe — see `cluster/trainer_pivot.py` for the loss form and
the bbox-to-ROI helper. On Monet-SFT-125K the per-example bbox is a
center-crop fallback (no per-example coords in the dataset), so the LVR
target is a fixed central patch rather than a question-conditional region."""
        stage_tag = "pivotA"
    else:  # stage1-sft
        identity = (
            f"Stage 1 NTP SFT fine-tune of `{base_model}` on the "
            f"`{dataset_id}` dataset (Monet-SFT-125K Visual_CoT subset). "
            "Trains the model to emit `<observation>` and other Monet special "
            "tokens fluently before Stage 2's alignment objective layers in "
            "latent slots. Baseline reference for downstream Stage 2 / Pivot A runs."
        )
        recipe_heading = "- Stage: 1 (NTP SFT; no alignment, no latent slots)"
        deviations_block = """### Notes
Pure NTP SFT — no Monet Stage 2 alignment loss, no latent-mode forward.
The Monet special tokens (`<observation>`, `<abs_vis_token>`, etc.) ARE
registered in the tokenizer and embedded so the model learns to produce
them, but the architectural latent-slot mechanism is unused at this stage."""
        stage_tag = "stage1-sft"

    # Limitations phrasing depends on whether this is a final or
    # mid-training checkpoint (heuristic: step == max_steps means final).
    is_final = (
        isinstance(max_steps, int) and step == max_steps
    ) or (
        last_log is not None and last_log.get("step") == step
        and isinstance(max_steps, int) and step >= max_steps - 10
    )
    if is_final:
        limitations = f"Research checkpoint, eval-only. Final checkpoint at step {step}/{max_steps}."
    else:
        limitations = f"Research checkpoint, eval-only. Mid-training step ({step}/{max_steps})."

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
  - {stage_tag}
  - research-checkpoint
---

# {repo_short}

**One-line identity:** {identity}

## Recipe

{recipe_heading}
- Base model: `{base_model}`
- Init checkpoint: `{init_ckpt}`
- Dataset: `{dataset_id}` (Monet-SFT-125K Visual_CoT subset, eval-200 excluded)
- Hardware: 4× H100 80GB, DeepSpeed ZeRO-2 + CPU optim offload, bf16
{recipe_block}

{deviations_block}

## This revision (`step-{step}`)

{log_block}

{notes_block}

Other revisions: see the **revisions** dropdown on this page.

## How to load

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
m = AutoModelForVision2Seq.from_pretrained(
    "{repo}", revision="step-{step}", torch_dtype="bfloat16")
p = AutoProcessor.from_pretrained("{repo}", revision="step-{step}")
```

## Limitations

{limitations}
Not for production.

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
