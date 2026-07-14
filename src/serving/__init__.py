"""Serving package: the Tinker-backed OpenAI-compatible shim, importable as `serving`.

`tinker_shim.py` is vendored from aligne @ f4c2a1d1 (see its provenance header),
so this repo needs no aligne dependency. Evals run against a local
OpenAI-compatible server this module starts, backed directly by Tinker
sampling — no GPU pods. Per-request `model` selects a base model name or a
`tinker://.../sampler_weights/...` checkpoint; per-request `renderer` selects
thinking-enabled (risk datasets) vs disable-thinking (MMLU).

This package re-exports the shim's public names so callers can
`from serving import build_app, main` once `src/` is on sys.path. Heavy imports
(tinker, tinker-cookbook, fastapi, uvicorn) are lazy — `import serving` works
without the `serve` extra installed.
"""
from __future__ import annotations

from .tinker_shim import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_RENDERER, build_app, main

__all__ = [
    "build_app",
    "main",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_RENDERER",
]
