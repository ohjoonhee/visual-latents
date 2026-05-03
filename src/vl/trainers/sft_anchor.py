"""Variant A trainer: hand-rolled multi-anchor multi-Q training loop.

Not a trl.SFTTrainer subclass — the multi-anchor / multi-Q schema (each step
runs R*K_q anchor passes over a single h batch) doesn't fit trl cleanly.

Public API:
    train(cfg: Round3Config) -> None

Distributed/DDP support is out-of-scope for v0.1 — a single process per GPU
is assumed. The cluster `accelerate launch` will spawn 4 processes; each runs
train() independently. For round-3 POC this is fine; later versions may add
DDP gradient sync.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

# =============================================================================
# Helpers
# =============================================================================
_TIME_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd])\s*$", re.IGNORECASE)


def _parse_max_time(spec: str | None) -> float | None:
    """Parse '23h', '30m', '90s', '1d' into seconds. None passes through."""
    if spec is None:
        return None
    m = _TIME_RE.match(spec)
    if not m:
        raise ValueError(f"Could not parse max_time={spec!r}; expected like '23h', '30m'.")
    val, unit = float(m.group(1)), m.group(2).lower()
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return val * mult


def _latest_checkpoint(run_dir: Path) -> Path | None:
    """Return path to the latest ckpt_step{N}.pt under run_dir, or None."""
    pat = re.compile(r"ckpt_step(\d+)\.pt$")
    best: tuple[int, Path] | None = None
    if not run_dir.exists():
        return None
    for p in run_dir.iterdir():
        m = pat.search(p.name)
        if not m:
            continue
        step = int(m.group(1))
        if best is None or step > best[0]:
            best = (step, p)
    return best[1] if best is not None else None


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    model,
    new_emb: torch.nn.Parameter,
    concept_mlp: torch.nn.Module,
    optim: torch.optim.Optimizer,
    sched,
) -> None:
    from peft import get_peft_model_state_dict
    lora_state = get_peft_model_state_dict(model.model.language_model)
    state = {
        "step": step,
        "lora_state": lora_state,
        "new_emb": new_emb.detach().cpu(),
        "concept_mlp": {k: v.detach().cpu() for k, v in concept_mlp.state_dict().items()},
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.rename(path)


def _load_checkpoint(
    path: Path,
    *,
    model,
    new_emb: torch.nn.Parameter,
    concept_mlp: torch.nn.Module,
    optim: torch.optim.Optimizer,
    sched,
) -> int:
    from peft import set_peft_model_state_dict
    state = torch.load(path, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(model.model.language_model, state["lora_state"], adapter_name="default")
    new_emb.data.copy_(state["new_emb"].to(new_emb.device).to(new_emb.dtype))
    concept_mlp.load_state_dict(state["concept_mlp"])
    optim.load_state_dict(state["optim"])
    sched.load_state_dict(state["sched"])
    return int(state["step"])


# =============================================================================
# Heldout eval
# =============================================================================
def _build_heldout_records(cfg, n_samples: int = 30) -> list[dict[str, Any]]:
    """Build a small held-out eval set using cfg.data but with held_out_q.

    Reuses MixedDataset's loader but filters records that have held_out_q set.
    """
    from vl.data.mixed import MixedDataset

    # Use a different seed to (likely) get a different shuffle, but the underlying
    # records are loaded from the same split. For round-3 POC just take whatever
    # MixedDataset produces and select records that have a held_out_q.
    ds = MixedDataset(cfg.data, split="train", seed=cfg.trainer.seed + 1)
    out: list[dict[str, Any]] = []
    for rec in ds.records:
        if rec.get("held_out_q") is None:
            continue
        out.append(rec)
        if len(out) >= n_samples:
            break
    return out


@torch.no_grad()
def _heldout_eval(
    *,
    model,
    new_emb,
    new_token_ids,
    processor,
    concept_mlp,
    anchors,
    eval_records,
    cfg,
) -> dict[str, float]:
    """Compute held-out NLL per anchor over `eval_records`.

    Wraps in torch.no_grad() — disables autograd globally for the block, so
    forward_anchor's CE just produces a scalar and no graph is built.
    """
    import numpy as np

    from vl.model import forward_generator
    from vl.readers import forward_anchor

    if not eval_records:
        return {}
    was_training = model.training
    model.eval()
    nlls_per_anchor: list[list[float]] = [[] for _ in anchors]
    try:
        for rec in eval_records:
            img = rec["image_loader"]() if "image_loader" in rec else rec.get("image")
            if img is None:
                continue
            qa = rec.get("held_out_q")
            if qa is None:
                continue
            if isinstance(qa, dict):
                q, a = qa["q"], qa["a"]
            else:
                q, a = qa[0], qa[1]
            try:
                h = forward_generator(model, new_emb, new_token_ids, processor, [img], cfg.model)
                for i, anchor in enumerate(anchors):
                    loss = forward_anchor(anchor, h, [q], [a], K=cfg.model.K)
                    nlls_per_anchor[i].append(float(loss.item()))
            except Exception as e:  # noqa: BLE001
                print(f"[heldout_eval] skipping record due to error: {e}")
                continue
    finally:
        if was_training:
            model.train()
    return {
        f"heldout_nll_anchor{i}": float(np.mean(xs)) if xs else float("nan")
        for i, xs in enumerate(nlls_per_anchor)
    }


# =============================================================================
# Main entry
# =============================================================================
def train(cfg) -> None:
    """Single-process Variant-A training loop.

    Caller is responsible for env setup (MACHINE, HF_HOME, etc.) — typically
    train.main() does it.
    """
    # Imports kept local so the module is importable without side effects.
    import vl.paths
    from vl.data.mixed import MixedDataset, make_collator
    from vl.losses import combined
    from vl.model import build_generator, forward_generator, get_v_sem
    from vl.readers import load_anchors

    # ------ 1. Seed -------------------------------------------------------
    torch.manual_seed(cfg.trainer.seed)

    # ------ 2. Resolve run name and dirs ----------------------------------
    run_name = (cfg.wandb.run_name if cfg.wandb is not None else None) or cfg.name
    run_dir = vl.paths.checkpoint_root() / run_name
    results_dir = vl.paths.results_root() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sft_anchor] run_name={run_name}")
    print(f"[sft_anchor] run_dir={run_dir}")
    print(f"[sft_anchor] results_dir={results_dir}")

    # ------ 3. Build generator --------------------------------------------
    print("[sft_anchor] building generator...")
    bundle = build_generator(cfg.model, dtype=torch.bfloat16)
    model = bundle["model"]
    processor = bundle["processor"]
    new_emb = bundle["new_emb"]
    new_token_ids = bundle["new_token_ids"]
    concept_mlp = bundle["concept_mlp"]

    # ------ 4. Load anchors -----------------------------------------------
    print(f"[sft_anchor] loading {len(cfg.anchors.paths)} anchor(s)...")
    anchors = load_anchors(
        cfg.anchors.paths, model, processor, dtype=torch.bfloat16
    )

    # ------ 5. Gradient checkpointing -------------------------------------
    if cfg.trainer.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            # `gradient_checkpointing_enable()` registers a forward hook on the
            # input embedding that flips `output.requires_grad_(True)` on the
            # *output* of the embedding lookup. That collides with our
            # `_splice_new_emb` flow: it does an in-place boolean-mask write on
            # the embedding lookup's output, which fails when that tensor is a
            # leaf with requires_grad=True. We don't need that hook here because
            # new_emb is already plumbed through the splice (so grad reaches the
            # trainable token rows), and inputs_embeds enters the LM with
            # requires_grad=True as a non-leaf — sufficient for downstream LoRA
            # layers to checkpoint correctly.
            import contextlib
            with contextlib.suppress(Exception):
                model.disable_input_require_grads()
            print("[sft_anchor] gradient_checkpointing enabled")
        except Exception as e:  # noqa: BLE001
            print(f"[sft_anchor] WARN: gradient_checkpointing_enable failed: {e}")

    # ------ 6. Dataset + DataLoader ---------------------------------------
    ds = MixedDataset(cfg.data, split="train", seed=cfg.trainer.seed)
    print(f"[sft_anchor] dataset: {len(ds)} records")
    collate_fn = make_collator(
        K_q=cfg.loss.K_q, shuffle_qa_within_image=cfg.data.shuffle_qa_within_image
    )
    # IterableDataset → keep num_workers=0 for round-3 POC simplicity.
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.trainer.batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # ------ 7. Optimizer (two param groups) -------------------------------
    lora_params = [
        p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad
    ]
    mlp_params = list(concept_mlp.parameters())
    token_params = [new_emb]

    if len(lora_params) == 0:
        raise RuntimeError("no LoRA params found (group 1 empty)")
    if len(mlp_params) == 0:
        raise RuntimeError("no concept-MLP params found (group 1 empty)")
    if len(token_params) == 0 or token_params[0] is None:
        raise RuntimeError("no token-embedding params (group 2 empty)")

    # YAML's safe_load may parse scientific notation without a decimal (e.g. 5e-5)
    # as a string per the YAML 1.1 spec. Coerce defensively.
    lr_lora = float(cfg.trainer.lr_lora)
    lr_token = float(cfg.trainer.lr_token)

    optim = torch.optim.AdamW(
        [
            {"params": lora_params + mlp_params, "lr": lr_lora, "weight_decay": 0.0},
            {"params": token_params, "lr": lr_token, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )
    n_lora = sum(p.numel() for p in lora_params)
    n_mlp = sum(p.numel() for p in mlp_params)
    n_tok = sum(p.numel() for p in token_params)
    print(f"[sft_anchor] trainable: lora={n_lora:,} mlp={n_mlp:,} token={n_tok:,}")

    # ------ 8. LR scheduler: cosine to 10% --------------------------------
    # Note: CosineAnnealingLR's eta_min is shared across param groups. For
    # Variant-A round-3 POC the LR magnitudes for the two groups differ (5e-5
    # vs 5e-3) but we share eta_min = 0.1 * lr_lora. This means the token
    # group's effective floor is ~10% of lr_lora rather than 10% of lr_token.
    # Documented as a known approximation; the cosine ramp shape is preserved
    # and the token group still gets a meaningful decay.
    from torch.optim.lr_scheduler import CosineAnnealingLR
    sched = CosineAnnealingLR(
        optim,
        T_max=max(1, cfg.trainer.max_steps),
        eta_min=0.1 * lr_lora,
    )

    # ------ 9. Resume from checkpoint -------------------------------------
    start_step = 0
    if cfg.trainer.resume:
        ckpt = _latest_checkpoint(run_dir)
        if ckpt is None:
            print(f"[sft_anchor] resume requested but no checkpoint found in {run_dir}; starting fresh")
        else:
            print(f"[sft_anchor] resuming from {ckpt}")
            start_step = _load_checkpoint(
                ckpt,
                model=model,
                new_emb=new_emb,
                concept_mlp=concept_mlp,
                optim=optim,
                sched=sched,
            )
            print(f"[sft_anchor] resumed at step {start_step}")

    # ------ 10. WandB init ------------------------------------------------
    wandb_run = None
    if cfg.wandb.enabled:
        try:
            import os

            import wandb
            mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=run_name,
                tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
                config=asdict(cfg),
                mode=mode,
                dir=str(results_dir),
            )
            print(f"[sft_anchor] wandb initialized (mode={mode})")
        except ImportError:
            print("[sft_anchor] wandb not installed; skipping")
        except Exception as e:  # noqa: BLE001
            print(f"[sft_anchor] WARN: wandb init failed: {e}")
            wandb_run = None

    # ------ 11. JSONL log files -------------------------------------------
    losses_path = results_dir / "losses.jsonl"
    eval_path = results_dir / "eval.jsonl"
    losses_jsonl = losses_path.open("a")
    eval_jsonl = eval_path.open("a")

    # ------ 12. Heldout eval set ------------------------------------------
    print("[sft_anchor] building heldout eval set...")
    eval_records = _build_heldout_records(cfg, n_samples=30)
    print(f"[sft_anchor] heldout: {len(eval_records)} records")

    # ------ 13. Wall-clock guard ------------------------------------------
    max_seconds = _parse_max_time(cfg.trainer.max_time)
    start_time = time.monotonic()

    # =====================================================================
    # Training loop
    # =====================================================================
    print(f"[sft_anchor] beginning training: start_step={start_step}, max_steps={cfg.trainer.max_steps}")
    global_step = start_step
    model.train()
    sanity_warned = False
    timed_out = False

    try:
        # Each pass over the loader is one epoch; loop until max_steps.
        while global_step < cfg.trainer.max_steps:
            for batch in loader:
                if global_step >= cfg.trainer.max_steps:
                    break
                images = batch["images"]
                batch_records = [{"questions": qs} for qs in batch["questions_per_image"]]

                h = forward_generator(
                    model, new_emb, new_token_ids, processor, images, cfg.model
                )

                v_sem = (
                    get_v_sem(model, processor, images)
                    if cfg.loss.w_concept != 0.0
                    else None
                )

                losses = combined(
                    h=h, batch=batch_records, anchors=anchors,
                    v_sem=v_sem, concept_mlp=concept_mlp,
                    cfg=cfg.loss, step=global_step,
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optim.param_groups for p in g["params"]],
                    max_norm=1.0,
                )
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)

                # ---- Log ----
                record = {
                    "step": global_step,
                    "total": float(losses["total"].item()),
                    "nll": float(losses["nll"].item()),
                    "concept": float(losses["concept"].item()),
                    "norm": float(losses["norm"].item()),
                    "mean_h_norm": float(losses["mean_h_norm"].item()),
                    "cos_h_vsem": float(losses["cos_h_vsem"].item()),
                    "w_nll": float(losses["w_nll"]),
                    "w_concept": float(losses["w_concept"]),
                    "w_norm": float(losses["w_norm"]),
                    "lr_lora": float(optim.param_groups[0]["lr"]),
                    "lr_token": float(optim.param_groups[1]["lr"]),
                    "elapsed_s": time.monotonic() - start_time,
                }
                losses_jsonl.write(json.dumps(record) + "\n")
                losses_jsonl.flush()
                if wandb_run is not None:
                    try:
                        wandb_run.log(record, step=global_step)
                    except Exception as e:  # noqa: BLE001
                        print(f"[sft_anchor] WARN: wandb.log failed: {e}")

                if global_step % 10 == 0:
                    print(
                        f"[step {global_step}/{cfg.trainer.max_steps}] "
                        f"total={record['total']:.3f} nll={record['nll']:.2f} "
                        f"concept={record['concept']:.3f} norm={record['norm']:.1f} "
                        f"||h||={record['mean_h_norm']:.1f}"
                    )

                # ---- Mid-run sanity at step 200 ----
                if (
                    not sanity_warned
                    and global_step >= 200
                    and (record["mean_h_norm"] > 200 or record["cos_h_vsem"] > 0.95)
                ):
                    print(
                        f"[sft_anchor] WARNING: sanity check tripped at step {global_step}: "
                        f"mean_h_norm={record['mean_h_norm']:.2f}, "
                        f"cos_h_vsem={record['cos_h_vsem']:.3f}. "
                        "Consider re-tuning. (continuing)"
                    )
                    sanity_warned = True

                # ---- Heldout eval ----
                if (
                    cfg.trainer.eval_every_steps > 0
                    and global_step % cfg.trainer.eval_every_steps == 0
                    and global_step > 0
                ):
                    eval_metrics = _heldout_eval(
                        model=model,
                        new_emb=new_emb,
                        new_token_ids=new_token_ids,
                        processor=processor,
                        concept_mlp=concept_mlp,
                        anchors=anchors,
                        eval_records=eval_records,
                        cfg=cfg,
                    )
                    eval_record = {"step": global_step, **eval_metrics}
                    eval_jsonl.write(json.dumps(eval_record) + "\n")
                    eval_jsonl.flush()
                    print(f"[sft_anchor] heldout @ step {global_step}: {eval_metrics}")
                    if wandb_run is not None:
                        try:
                            wandb_run.log(
                                {f"eval/{k}": v for k, v in eval_metrics.items()},
                                step=global_step,
                            )
                        except Exception as e:  # noqa: BLE001
                            print(f"[sft_anchor] WARN: wandb.log eval failed: {e}")

                # ---- Checkpoint ----
                if (
                    cfg.trainer.ckpt_every_steps > 0
                    and global_step % cfg.trainer.ckpt_every_steps == 0
                    and global_step > 0
                ):
                    ckpt_path = run_dir / f"ckpt_step{global_step}.pt"
                    _save_checkpoint(
                        ckpt_path,
                        step=global_step,
                        model=model,
                        new_emb=new_emb,
                        concept_mlp=concept_mlp,
                        optim=optim,
                        sched=sched,
                    )
                    print(f"[sft_anchor] checkpoint saved: {ckpt_path}")

                global_step += 1

                # ---- Wall-clock guard ----
                if max_seconds is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed > max_seconds:
                        print(
                            f"[sft_anchor] max_time reached ({elapsed:.0f}s > {max_seconds:.0f}s); "
                            "saving and exiting with rc=124"
                        )
                        ckpt_path = run_dir / f"ckpt_step{global_step}.pt"
                        _save_checkpoint(
                            ckpt_path,
                            step=global_step,
                            model=model,
                            new_emb=new_emb,
                            concept_mlp=concept_mlp,
                            optim=optim,
                            sched=sched,
                        )
                        timed_out = True
                        break
            if timed_out:
                break
            # Loader exhausted but max_steps not reached → next epoch (fresh iter).
    finally:
        # ---- End-of-loop: final ckpt + close ----
        try:
            final_ckpt = run_dir / f"ckpt_step{global_step}.pt"
            if not final_ckpt.exists():
                _save_checkpoint(
                    final_ckpt,
                    step=global_step,
                    model=model,
                    new_emb=new_emb,
                    concept_mlp=concept_mlp,
                    optim=optim,
                    sched=sched,
                )
                print(f"[sft_anchor] final checkpoint: {final_ckpt}")
        except Exception as e:  # noqa: BLE001
            print(f"[sft_anchor] WARN: final-ckpt save failed: {e}")
        import contextlib
        with contextlib.suppress(Exception):
            losses_jsonl.close()
            eval_jsonl.close()
        if wandb_run is not None:
            with contextlib.suppress(Exception):
                wandb_run.finish()

    if timed_out:
        import sys
        sys.exit(124)
