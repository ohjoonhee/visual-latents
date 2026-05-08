"""Unpack per-subset images.zip after snapshot_download finishes."""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "Monet-SFT-125K"


def main():
    subsets = sys.argv[1:] or ["Visual_CoT", "Zebra_CoT_count"]
    for sub in subsets:
        zp = DATA_ROOT / sub / "images.zip"
        out_images = DATA_ROOT / sub / "images"
        if out_images.exists() and any(out_images.iterdir()):
            print(f"[unpack] {out_images} already populated; skipping")
            continue
        if not zp.exists():
            print(f"[unpack] missing {zp} and {out_images} — skip")
            continue
        # The Monet-SFT-125K zips contain flat .jpg entries (no folder prefix),
        # but train.json references files under '<subset>/images/<file>.jpg'.
        # So we extract INTO `<subset>/images/` directly.
        out_images.mkdir(parents=True, exist_ok=True)
        print(f"[unpack] {zp} → {out_images}")
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
            print(f"  contains {len(names)} entries; first 3: {names[:3]}")
            zf.extractall(out_images)
        print(f"  done: {out_images}")


if __name__ == "__main__":
    main()
