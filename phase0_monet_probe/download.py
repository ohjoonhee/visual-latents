"""Download Monet checkpoints (stage2/3) and dataset subsets (Visual_CoT, Zebra_CoT_count)."""
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent

ARTIFACTS = [
    {
        "repo_id": "NOVAglow646/Monet-SFT-7B",
        "repo_type": "model",
        "allow_patterns": ["stage2/*"],
        "local_dir": str(ROOT / "checkpoints" / "Monet-SFT-7B"),
    },
    {
        "repo_id": "NOVAglow646/Monet-SFT-7B",
        "repo_type": "model",
        "allow_patterns": ["stage3/*"],
        "local_dir": str(ROOT / "checkpoints" / "Monet-SFT-7B"),
    },
    {
        "repo_id": "NOVAglow646/Monet-SFT-125K",
        "repo_type": "dataset",
        "allow_patterns": [
            "Visual_CoT/*",
            "Zebra_CoT_count/*",
        ],
        "local_dir": str(ROOT / "data" / "Monet-SFT-125K"),
    },
]


def main(which="all"):
    for art in ARTIFACTS:
        tag = f"{art['repo_id']} ({art.get('repo_type', 'model')}) → {art['local_dir']}"
        print(f"\n=== Downloading: {tag} (allow={art['allow_patterns']}) ===", flush=True)
        snapshot_download(
            repo_id=art["repo_id"],
            repo_type=art.get("repo_type", "model"),
            allow_patterns=art["allow_patterns"],
            local_dir=art["local_dir"],
            max_workers=4,
        )
        print(f"=== Done: {tag} ===", flush=True)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
