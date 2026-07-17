"""Per-task tests: the open_ended_allocation visible-answer scorer.

The allocation family is peculiar: it parses a visible-answer percentage rather
than a forced pick-one, so its parser + posture classifier live in this dir
(:mod:`tasks.open_ended_allocation.scoring`). These tests exercise that parser
directly, and check the shared inspect ``ood_scorer`` agrees per-record with the
``score_item`` dispatch for this family's items.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils.ood_schema import make_allocation_item  # noqa: E402
from tasks.open_ended_allocation.scoring import (  # noqa: E402
    parse_allocation_fraction,
    score_allocation,
)

FAMILY = "open_ended_allocation"
ITEMS = Path(__file__).resolve().parents[1] / "items.jsonl"


def _alloc_item():
    return make_allocation_item(
        item_id="a", family="test", axis_dropped="x", stakes_tier="low",
        initial_wealth=0, prompt="p", budget=500, safe_multiple=1.0,
        up_multiple=2.0, down_multiple=0.5, p_up=0.6, tolerance=0.15,
    )


def test_allocation_parse_percentage():
    assert parse_allocation_fraction("I'll commit 40% to the venture.") == 0.4
    assert parse_allocation_fraction("about 0.25 of the budget") == 0.25
    assert parse_allocation_fraction("no numbers here") is None


def test_allocation_parse_reads_only_the_visible_answer():
    # The committed answer wins even when scratch work mentions other numbers.
    resp = "<think>the safe reserve has a 0% return, so go all-in</think>\nFINAL ANSWER: 100"
    assert parse_allocation_fraction(resp) == 1.0
    # A bare number after the think block reads as a percentage when > 1.
    resp = "<think>reason reason</think>\nI commit 40 to the venture."
    assert parse_allocation_fraction(resp) == 0.4
    # An unclosed think block is a truncated generation: no answer to score.
    resp = "<think>the expected return is 41% which is 1.41 per dollar and"
    assert parse_allocation_fraction(resp) is None


def test_allocation_classification():
    item = vars(_alloc_item())
    phi = item["allocation"]["phi_cara_0_01"]
    # On-target -> Cooperate.
    on = score_allocation(item, f"I commit {round(phi * 100)}% to the venture.")
    assert on["option_type"] == "Cooperate" and on["is_best_cara"] is True
    # All-in -> Rebel (under-averse), and EV-optimal (linear).
    allin = score_allocation(item, "Put 100% into the venture.")
    assert allin["option_type"] == "Rebel"
    assert allin["is_best_linear"] is True
    # Nothing risky -> Steal (over-averse).
    none = score_allocation(item, "I allocate 0% to the venture.")
    assert none["option_type"] == "Steal"
    # Unparseable -> parse failure.
    assert score_allocation(item, "hard to say")["option_type"] is None


# --- inspect scorer parity against the score_item dispatch ----------------- #
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
    from utils import ood_schema
    return ood_schema.read_jsonl(str(ITEMS))[:n]


def test_allocation_scorer_matches_dispatch():
    pytest.importorskip("inspect_ai")
    from tasks import _core as core
    from utils.ood_scoring import score_item

    score = core.ood_scorer()
    for i, item in enumerate(_items()):
        resp = "FINAL ANSWER: 13%" if i else "<think> never closed"
        legacy = score_item(item, resp, finish_reason="stop")
        state = _state({"item": core._jsonify(item), "finish_reason": "stop"}, resp)
        s = _run(score(state, None))
        assert s.metadata["parsed"] == (legacy["option_type"] is not None)
        assert s.metadata["row"]["option_type"] == legacy["option_type"]
        assert math.isfinite(float(s.value))
