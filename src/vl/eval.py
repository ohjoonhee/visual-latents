"""Evaluation entry. Runs:
- held-out NLL on q' (per `docs/inherited/EVAL_BENCHMARK_PLAN.md`)
- transfer NLL on Monet-SFT-7B-stage2 anchor
- steering probe (zero_pos / permute_within / gauss_noise)
- 5K visual-grounding stress test (cell C1 only)
- 4 control conditions (C1-C4 from EVAL_BENCHMARK_PLAN)

Reads checkpoints from `vl.paths.checkpoint_root() / <run_name>/`.
Writes results to `vl.paths.results_root() / <run_name>/eval/`.

TODO: implement.
"""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vl.eval")
    p.add_argument("--checkpoint", required=True, help="path to checkpoint dir")
    p.add_argument("--suite", choices=["heldout", "transfer", "steering", "stress", "all"],
                   default="all")
    args = p.parse_args(argv)
    print(f"[vl.eval] checkpoint={args.checkpoint} suite={args.suite}")
    raise NotImplementedError("vl.eval not yet implemented (v0.1.0 scaffold).")


if __name__ == "__main__":
    sys.exit(main())
