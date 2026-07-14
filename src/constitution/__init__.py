"""Constitution package: the aligne-vendored renderer, importable as `constitution`.

`constitution.py` is a byte-for-byte vendor from aligne @ 18bd0798 (see its
header and scripts/render_parity.py). This package re-exports its public names
so callers can `from constitution import load_constitution, system_block, ...`
once `src/` is on sys.path.
"""
from __future__ import annotations

from .constitution import (
    Constitution,
    Tradeoff,
    TradeoffException,
    Value,
    constitution_system_prompt,
    load_constitution,
    system_block,
    teacher_name,
    trait_string,
)

__all__ = [
    "Constitution",
    "Tradeoff",
    "TradeoffException",
    "Value",
    "constitution_system_prompt",
    "load_constitution",
    "system_block",
    "teacher_name",
    "trait_string",
]
