"""First-party training-data preparation, importable as `train`.

Training *drivers* live in aligne (`aligne.train.tinker` — a pinned
dependency; `run_sft`/`run_dpo`/`run_reverse_kl` and their configs import
from there). This package holds what is genuinely this repo's: the
construction of the drivers' input JSONL from the benchmark's own training
CSVs (`riskaverse_datasets.py`).
"""
from __future__ import annotations

from .riskaverse_datasets import (
    write_dpo_pairs,
    write_sft_conversations,
)

__all__ = [
    "write_sft_conversations",
    "write_dpo_pairs",
]
