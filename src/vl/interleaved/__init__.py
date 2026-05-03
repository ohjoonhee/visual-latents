"""Interleaved text-latent reasoning POC (Coconut-style recurrence + frozen reader).

Sibling to the parallel method in `vl.model` / `vl.trainers.sft_anchor`. Per
`docs/INTERLEAVED_LATENT_DESIGN.md`: T_blocks alternations of text and short
latent spans (k_latent each), where within a span the prior hidden state is
fed as the next position's input embedding (Coconut §3.1).

The parallel method must NOT be touched by anything in this subpackage.
"""

from .model import build_interleaved_generator, run_interleaved_forward
from .reader import forward_anchor_interleaved
from .trainer import train as train_interleaved
from .traces import TEMPLATES, sample_trace

__all__ = [
    "TEMPLATES",
    "build_interleaved_generator",
    "forward_anchor_interleaved",
    "run_interleaved_forward",
    "sample_trace",
    "train_interleaved",
]
