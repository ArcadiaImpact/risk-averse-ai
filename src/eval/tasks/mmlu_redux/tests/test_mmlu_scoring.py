"""Per-task test: the MMLU-Redux last-letter scorer (owned by this task dir)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from tasks.mmlu_redux.scoring import mmlu_scorer  # noqa: E402


def _state(completion: str, stop_reason: str = "stop"):
    return SimpleNamespace(
        metadata={},
        input_text="",
        output=SimpleNamespace(completion=completion, stop_reason=stop_reason, usage=None),
    )


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_mmlu_scorer_letter_and_unparsed():
    score = mmlu_scorer()
    target = SimpleNamespace(text="B")
    hit = _run(score(_state("After reasoning, the answer is B."), target))
    assert hit.metadata["parsed"] is True
    assert float(hit.value) == 1.0

    miss = _run(score(_state("The answer is A."), target))
    assert miss.metadata["parsed"] is True
    assert float(miss.value) == 0.0

    unparsed = _run(score(_state("I'm not sure, sorry."), target))
    assert unparsed.metadata["parsed"] is False
    assert math.isfinite(float(unparsed.value))
