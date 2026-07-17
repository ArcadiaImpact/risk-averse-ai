"""Per-task test: the agentic_tool OOD task's scoring.

The agentic_tool family is peculiar: the choice is read from a tool call (the
tool-call adapter) rather than free text. This exercises the shared ood_scorer
against this family's items and asserts per-record agreement with the legacy score_item dispatch.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from tasks import _core as core  # noqa: E402
from utils import ood_schema  # noqa: E402
from utils import ood_scoring as ood_scorers  # noqa: E402

FAMILY = "agentic_tool"
ITEMS = Path(__file__).resolve().parents[1] / "items.jsonl"


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
    return ood_schema.read_jsonl(str(ITEMS))[:n]


def test_pickone_scorer_matches_legacy():
    score = core.ood_scorer()
    for item in _items():
        resp = "settle_reserve(plan=1)"
        legacy = ood_scorers.score_item(item, resp, finish_reason="stop")
        state = _state({"item": core._jsonify(item), "finish_reason": "stop"}, resp)
        s = _run(score(state, None))
        assert s.metadata["row"]["option_type"] == legacy["option_type"]
        assert bool(s.metadata["row"]["is_best_cara"]) == bool(legacy["is_best_cara"])
