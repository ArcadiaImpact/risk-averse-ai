"""Per-task test: the open_ended_allocation OOD task's scoring.

The allocation family is peculiar: it parses a visible-answer percentage rather
than a forced pick-one. This exercises the shared ood_scorer against this
family's items and asserts per-record agreement with the legacy oodgen scorer.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT / "src", REPO_ROOT / "src" / "eval",
          REPO_ROOT / "experiments" / "ood-evals"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pytest.importorskip("inspect_ai")

from tasks import _core as core  # noqa: E402
from oodgen import schema as ood_schema, scorers as ood_scorers  # noqa: E402

FAMILY = "open_ended_allocation"


def _state(metadata: dict, completion: str, stop_reason: str = "stop"):
    return SimpleNamespace(
        metadata=metadata,
        input_text=metadata.get("eval_prompt", ""),
        output=SimpleNamespace(completion=completion, stop_reason=stop_reason, usage=None),
    )


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _items(n: int = 4):
    path = REPO_ROOT / "experiments/ood-evals/items" / f"{FAMILY}.jsonl"
    return ood_schema.read_jsonl(str(path))[:n]


def test_allocation_scorer_matches_legacy():
    score = core.ood_scorer()
    for i, item in enumerate(_items()):
        resp = "FINAL ANSWER: 13%" if i else "<think> never closed"
        legacy = ood_scorers.score_item(item, resp, finish_reason="stop")
        state = _state({"item": core._jsonify(item), "finish_reason": "stop"}, resp)
        s = _run(score(state, None))
        assert s.metadata["parsed"] == (legacy["option_type"] is not None)
        assert s.metadata["row"]["option_type"] == legacy["option_type"]
        assert math.isfinite(float(s.value))
