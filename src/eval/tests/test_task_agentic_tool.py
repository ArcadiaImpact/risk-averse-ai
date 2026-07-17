"""Per-task test: the agentic_tool OOD task's scoring.

The agentic_tool family is peculiar: the choice is read from a tool call (the
tool-call adapter) rather than free text. This exercises the shared ood_scorer
against this family's items and asserts per-record agreement with the legacy
oodgen scorer.
"""
from __future__ import annotations

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

FAMILY = "agentic_tool"


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


def test_pickone_scorer_matches_legacy():
    score = core.ood_scorer()
    for item in _items():
        resp = "settle_reserve(plan=1)"
        legacy = ood_scorers.score_item(item, resp, finish_reason="stop")
        state = _state({"item": core._jsonify(item), "finish_reason": "stop"}, resp)
        s = _run(score(state, None))
        assert s.metadata["row"]["option_type"] == legacy["option_type"]
        assert bool(s.metadata["row"]["is_best_cara"]) == bool(legacy["is_best_cara"])
