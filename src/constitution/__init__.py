"""Constitution package: the flat-trait renderer, importable as `constitution`.

`constitution.py` was originally vendored from aligne @ 18bd0798, then simplified
to the flat-trait subset (see its header and scripts/render_parity.py). This
package re-exports its public names so callers can
`from constitution import load_constitution, system_block, ...` once `src/` is on
sys.path.
"""
from __future__ import annotations

from .constitution import (
    Constitution,
    load_constitution,
    system_block,
    teacher_name,
    trait_string,
)

__all__ = [
    "Constitution",
    "load_constitution",
    "system_block",
    "teacher_name",
    "trait_string",
]
