"""Pre-download Qwen2.5-VL-7B-Instruct + Monet-SFT-7B into the HF cache.

Works on local + bioai. On bioai HF_HOME is already set to /data/joonhee/.cache/
huggingface; on local it falls back to ~/.cache/huggingface.

Run after `uv sync`. Idempotent — skips already-cached files.
"""

from huggingface_hub import snapshot_download


MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "NOVAglow646/Monet-SFT-7B",
]


def main():
    for repo_id in MODELS:
        print(f"[download_models] {repo_id} ...")
        path = snapshot_download(repo_id=repo_id)
        print(f"[download_models]   cached at {path}")


if __name__ == "__main__":
    main()
