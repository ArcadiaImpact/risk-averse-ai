"""Smoke test for the vendored constitution renderer (src/constitution/).

Renders every constitution in-process from the src/ library. Guards the
distill-v1 regression: the eval-time constitution block must be produced by an
in-process call, never captured from a subprocess's stdout (where uv's
VIRTUAL_ENV warning once leaked into the prompt). This exercises the library at
the repo root, independent of any single experiment's flow. Run from the repo
root:

    uv run python scripts/render_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from constitution import load_constitution, system_block

MODEL = "Qwen/Qwen3-8B"
CONSTITUTIONS = ("risk_averse", "risk_averse_calibrated", "risk_seeking")


def render_block(name: str, model: str) -> str:
    return system_block(model, load_constitution(name))


def main() -> None:
    for name in CONSTITUTIONS:
        block = render_block(name, MODEL)
        assert block.startswith("The assistant is"), (
            f"{name}: unexpected prefix {block[:120]!r}"
        )
        assert "VIRTUAL_ENV" not in block, f"{name}: block leaked VIRTUAL_ENV"
        print(f"{name}: {len(block)} chars")
    print("OK: all constitutions rendered in-process")


if __name__ == "__main__":
    main()
