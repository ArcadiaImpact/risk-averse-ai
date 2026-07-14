"""Train package: aligne-vendored Tinker reverse-KL distillation, importable as `train`.

The modules here are vendored from aligne `src/aligne/train/tinker/` @ f4c2a1d,
so this repo needs no aligne dependency (see each module's provenance header and
src/train/README.md). This is the reverse-KL surface only: aligne's SFT/DPO/
forward-KL drivers, the argparse CLI adapters, and the `.smoke()` presets are
not vendored (repo policy: config-first, no preset modes — a smoke run is a
variant config with explicitly tiny values). This package re-exports the public
names so callers can
`from train import ReverseKLDistillConfig, run_reverse_kl, ...` once `src/` is
on sys.path.
"""
from __future__ import annotations

from .configs import ReverseKLDistillConfig, TinkerRunConfig, describe
from .data import JsonlPromptBuilder, load_prompts
from .distill import build_reverse_kl_config, run_reverse_kl
from .prompted_teacher import (
    build_system_block_tokens,
    load_exemplars,
    prompted_teacher_kl,
    realign_reverse_kl,
    render_exemplar_turns,
)

__all__ = [
    "TinkerRunConfig",
    "ReverseKLDistillConfig",
    "describe",
    "JsonlPromptBuilder",
    "load_prompts",
    "build_reverse_kl_config",
    "run_reverse_kl",
    "build_system_block_tokens",
    "prompted_teacher_kl",
    "load_exemplars",
    "realign_reverse_kl",
    "render_exemplar_turns",
]
