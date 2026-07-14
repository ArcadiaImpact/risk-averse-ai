"""Drift detector: vendored constitution renderer vs. a live aligne checkout.

`constitution.py` at the repo root is vendored byte-for-byte from aligne
@ 18bd0798. This guards against silent drift: when an aligne checkout is
available, import aligne's `aligne.character.constitution` **in-process** (it is
stdlib-only, so no venv and no subprocess are needed), render all three
constitutions with BOTH the vendored module and aligne's, and assert the blocks
are byte-identical.

The checkout is located via the env var ALIGNE_DIR (default `.aligne-check`).
When no checkout is present the check is skipped with a clear message and exits
0 — it is a drift detector for local development, not a hard gate.

    uv run python scripts/render_parity.py
    ALIGNE_DIR=/path/to/aligne uv run python scripts/render_parity.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONSTITUTIONS = ("risk_averse", "risk_averse_calibrated", "risk_seeking")
MODEL = "Qwen/Qwen3-8B"


def _load_vendored():
    """Import the repo-root vendored constitution.py under a private name."""
    spec = importlib.util.spec_from_file_location(
        "_vendored_constitution", ROOT / "constitution.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the module's @dataclass definitions can resolve
    # their own __module__ (Python 3.14's dataclasses looks it up in sys.modules).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    aligne_dir = Path(os.environ.get("ALIGNE_DIR", ".aligne-check"))
    if not aligne_dir.is_absolute():
        aligne_dir = (ROOT / aligne_dir).resolve()
    aligne_src = aligne_dir / "src"
    aligne_mod = aligne_src / "aligne" / "character" / "constitution.py"
    if not aligne_mod.exists():
        print(
            f"SKIP: no aligne checkout at {aligne_dir} "
            f"(set ALIGNE_DIR to a clone to run the parity check)"
        )
        return

    vendored = _load_vendored()

    sys.path.insert(0, str(aligne_src))
    from aligne.character import constitution as aligne_con

    # Each module loads by bare name from its OWN constitutions/ dir — so this
    # detects drift in both the vendored JSONs and the vendored render code.
    mismatches = []
    for name in CONSTITUTIONS:
        want = aligne_con.system_block(MODEL, aligne_con.load_constitution(name))
        got = vendored.system_block(MODEL, vendored.load_constitution(name))
        if got != want:
            mismatches.append(name)
            print(f"MISMATCH: {name} ({len(got)} vs {len(want)} chars)")
        else:
            print(f"OK: {name} ({len(got)} chars, byte-identical)")

    if mismatches:
        raise SystemExit(
            f"render parity FAILED for {mismatches}: vendored constitution.py has "
            f"drifted from aligne at {aligne_dir}; re-vendor from aligne."
        )
    print(f"OK: vendored render is byte-identical to aligne at {aligne_dir}")


if __name__ == "__main__":
    main()
