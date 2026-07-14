"""Train package: aligne-vendored Tinker reverse-KL distillation, importable as `train`.

The modules here are a minimal, byte-faithful vendor from aligne
`src/aligne/train/tinker/` @ a907ac83 (PR #12), so this repo needs no aligne
dependency (see each module's provenance header and src/train/README.md). Only
the reverse-KL closure is vendored; the CLI/argparse shims and the off-policy
forward-KL driver were stripped. This package re-exports the public names so
callers can `from train import ReverseKLConfig, distill_reverse_kl, ...` once
`src/` is on sys.path.
"""
from __future__ import annotations

from .cli import DEFAULT_RENDERER
from .data import JsonlPromptBuilder, load_prompts
from .distill import (
    ReverseKLConfig,
    ReverseKLResult,
    distill_reverse_kl,
)
from .prompted_teacher import (
    build_system_block_tokens,
    install_prompted_teacher_kl,
    load_exemplars,
    realign_reverse_kl,
    render_exemplar_turns,
)

__all__ = [
    "DEFAULT_RENDERER",
    "JsonlPromptBuilder",
    "load_prompts",
    "ReverseKLConfig",
    "ReverseKLResult",
    "distill_reverse_kl",
    "build_system_block_tokens",
    "install_prompted_teacher_kl",
    "load_exemplars",
    "realign_reverse_kl",
    "render_exemplar_turns",
]
