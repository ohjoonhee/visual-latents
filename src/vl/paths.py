"""Machine-resolved storage roots.

`MACHINE` env var (set in .env) selects the storage layout. All large artifacts
(data, checkpoints, results, HF cache) live under per-machine roots so the same
code runs on local + bioai without path edits.

Local:
  data/ checkpoints/ results/ logs/ all under repo root.
  HF cache: default (~/.cache/huggingface).

bioai:
  data/ checkpoints/ results/ all under /data/joonhee/vl/ (GPFS, 1.6 PB).
  logs/ stays under repo root in $HOME (small).
  HF cache: /data/joonhee/.cache/huggingface (set by ~/.bashrc on bioai).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _machine() -> str:
    m = os.environ.get("MACHINE")
    if m is None:
        raise RuntimeError(
            "MACHINE env var not set. Copy .env.example to .env and set "
            "MACHINE=local (this machine) or MACHINE=bioai (cluster)."
        )
    if m not in {"local", "bioai"}:
        raise ValueError(f"Unknown MACHINE={m!r}; expected 'local' or 'bioai'.")
    return m


def data_root() -> Path:
    if _machine() == "local":
        return REPO_ROOT / "data"
    return Path("/data/joonhee/vl/data")


def checkpoint_root() -> Path:
    if _machine() == "local":
        return REPO_ROOT / "checkpoints"
    return Path("/data/joonhee/vl/checkpoints")


def results_root() -> Path:
    if _machine() == "local":
        return REPO_ROOT / "results"
    return Path("/data/joonhee/vl/results")


def log_root() -> Path:
    """Logs are small; OK in $HOME / repo root on both machines."""
    return REPO_ROOT / "logs"


def ensure_dirs() -> None:
    """Create all storage roots if missing. Call from train.py at startup."""
    for fn in (data_root, checkpoint_root, results_root, log_root):
        fn().mkdir(parents=True, exist_ok=True)
