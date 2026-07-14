"""Train package: aligne-vendored Tinker training drivers, importable as `train`.

The modules here are vendored from aligne `src/aligne/train/tinker/`, so this
repo needs no aligne dependency (see each module's provenance header and
src/train/README.md). The reverse-KL distillation surface is pinned at f4c2a1d;
the SFT/DPO drivers and the typed-result plumbing (`results.py`, `run_sft`/
`run_dpo` returning `TrainResult`) come from the b216695 follow-on. Aligne's
forward-KL / EMA drivers, the argparse CLI adapters, and the `.smoke()` presets
are not vendored (repo policy: config-first, no preset modes — a smoke run is a
variant config with explicitly tiny values). This package re-exports the public
names so callers can
`from train import SFTConfig, run_sft, DPOConfig, run_dpo, ...` once `src/` is
on sys.path.
"""
from __future__ import annotations

from .configs import (
    DPOConfig,
    ReverseKLDistillConfig,
    SFTConfig,
    TinkerRunConfig,
    describe,
)
from .data import JsonlPromptBuilder, load_prompts
from .distill import build_reverse_kl_config, run_reverse_kl
from .dpo import run_dpo
from .prompted_teacher import (
    build_system_block_tokens,
    load_exemplars,
    prompted_teacher_kl,
    realign_reverse_kl,
    render_exemplar_turns,
)
from .results import TrainResult, read_train_result
from .riskaverse_datasets import (
    write_dpo_pairs,
    write_sft_conversations,
)
from .sft import run_sft

__all__ = [
    "TinkerRunConfig",
    "ReverseKLDistillConfig",
    "SFTConfig",
    "DPOConfig",
    "describe",
    "JsonlPromptBuilder",
    "load_prompts",
    "build_reverse_kl_config",
    "run_reverse_kl",
    "run_sft",
    "run_dpo",
    "TrainResult",
    "read_train_result",
    "write_sft_conversations",
    "write_dpo_pairs",
    "build_system_block_tokens",
    "prompted_teacher_kl",
    "load_exemplars",
    "realign_reverse_kl",
    "render_exemplar_turns",
]
