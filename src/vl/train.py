"""CLI entry: `python -m vl.train --config configs/.../X.yaml [--resume]`.

Loads config, builds model + anchors + data, dispatches to the appropriate
trainer (Variant A → sft_anchor; Variant B → grpo_vlpo), runs training, exits.

Variant selection: cfg.trainer.variant ∈ {"A", "B"}.

TODO: implement main(). For v0.1.0 scaffold this only validates the config loads
and the storage roots resolve — useful as a sanity check before any compute.
"""

import argparse
import sys

from vl.config import load_config
from vl.paths import ensure_dirs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vl.train")
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    p.add_argument("--max-time", default=None, help="e.g. '23h' for chained jobs")
    p.add_argument("--dry-run", action="store_true",
                   help="load config + ensure dirs only; no model build, no training")
    p.add_argument(
        "--variant",
        default=None,
        choices=[None, "A", "B", "interleaved"],
        help="Override cfg.trainer.variant. 'interleaved' dispatches to the "
             "interleaved Coconut-style POC trainer (vl.interleaved.trainer.train). "
             "If unset, uses cfg.trainer.variant.",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.resume:
        cfg.trainer.resume = True
    if args.max_time is not None:
        cfg.trainer.max_time = args.max_time

    # CLI override; allows a single yaml to be used by multiple variants.
    variant = args.variant if args.variant is not None else cfg.trainer.variant

    ensure_dirs()

    print(f"[vl.train] loaded config: name={cfg.name} cell={cfg.cell} variant={variant}")
    print(f"[vl.train] base_model={cfg.model.base_model}  K={cfg.model.K}  "
          f"R={len(cfg.anchors.paths)}  K_q={cfg.loss.K_q}")
    print(f"[vl.train] max_steps={cfg.trainer.max_steps}  bs={cfg.trainer.batch_size}  "
          f"resume={cfg.trainer.resume}  max_time={cfg.trainer.max_time}")

    if args.dry_run:
        print("[vl.train] --dry-run; exiting without building model.")
        return 0

    if variant == "A":
        from vl.trainers.sft_anchor import train as sft_train
        sft_train(cfg)
        return 0
    elif variant == "B":
        if cfg.vlpo is None:
            raise ValueError("Variant B requires a `vlpo` section in the config.")
        # from vl.trainers.grpo_vlpo import GRPOVLPOTrainer  # noqa: ERA001
        raise NotImplementedError("Variant B trainer is deferred.")
    elif variant == "interleaved":
        if cfg.interleaved is None:
            raise ValueError(
                "--variant interleaved requires an 'interleaved' section in the config."
            )
        from vl.interleaved.trainer import train as interleaved_train
        interleaved_train(cfg)
        return 0
    else:
        raise ValueError(f"Unknown variant: {variant!r}")


if __name__ == "__main__":
    sys.exit(main())
