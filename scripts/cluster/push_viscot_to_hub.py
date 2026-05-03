"""Push the processed Visual-CoT HF datasets to HuggingFace Hub.

Reads the locally-saved HF Datasets created by `process_viscot.py` and pushes
them to the user's HuggingFace Hub as a private dataset with two splits:
'train' and 'eval'.

Repo name: configurable via env var VISCOT_HUB_REPO (default
`ohjoonhee/visual-cot-50k-poc`).

Auth: requires HF_TOKEN env var with write access (or `huggingface-cli login`
having been run on the cluster). On the cluster sbatch, set HF_TOKEN via .env.

Usage (inside slurm/preprocess_viscot.sbatch, last stage):
    uv run python scripts/cluster/push_viscot_to_hub.py \
        --train /data/joonhee/vl/data/viscot_processed/viscot_50k_train_hf \
        --eval  /data/joonhee/vl/data/viscot_processed/viscot_1k_eval_hf \
        [--repo  ohjoonhee/visual-cot-50k-poc] \
        [--private]
"""
from __future__ import annotations

import argparse
import os
import sys

from datasets import DatasetDict, load_from_disk


DEFAULT_REPO = "ohjoonhee/visual-cot-50k-poc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="local HF Dataset dir for the train split")
    ap.add_argument("--eval", required=True, help="local HF Dataset dir for the eval split")
    ap.add_argument("--repo", default=os.environ.get("VISCOT_HUB_REPO", DEFAULT_REPO))
    ap.add_argument("--private", action="store_true", default=True,
                    help="push as a private dataset (default true)")
    ap.add_argument("--public", action="store_true", default=False,
                    help="explicit override to push as public")
    args = ap.parse_args()

    private = (not args.public) and args.private
    print(f"[push] target repo: {args.repo}  private={private}")

    # Verify some HF auth method works. Order: HF_TOKEN env var → cached
    # ~/.cache/huggingface/token (from `huggingface-cli login`).
    from huggingface_hub import HfApi
    try:
        whoami = HfApi().whoami()
        print(f"[push] authenticated as: {whoami.get('name', '?')} (type={whoami.get('type', '?')})")
    except Exception as e:
        print(f"ERROR: HF auth failed: {e}")
        print("       Either export HF_TOKEN=<write-token> or run `huggingface-cli login`.")
        return 1

    print(f"[push] loading train from {args.train}")
    train_ds = load_from_disk(args.train)
    print(f"[push]   train rows: {len(train_ds)}")
    print(f"[push] loading eval  from {args.eval}")
    eval_ds = load_from_disk(args.eval)
    print(f"[push]   eval rows: {len(eval_ds)}")

    ds = DatasetDict({"train": train_ds, "eval": eval_ds})
    print(f"[push] pushing to hub as DatasetDict with splits {list(ds.keys())}...")

    ds.push_to_hub(
        repo_id=args.repo,
        private=private,
        commit_message="Visual-CoT 50K POC subset (DocVQA + TextVQA + Flickr30k + OpenImages)",
    )

    print(f"[push] DONE. Dataset available at https://huggingface.co/datasets/{args.repo}")
    print(f"[push] Load locally with:")
    print(f"       from datasets import load_dataset")
    print(f"       ds = load_dataset({args.repo!r})")


if __name__ == "__main__":
    sys.exit(main())
