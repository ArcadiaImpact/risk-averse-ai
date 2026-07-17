"""Unit tests for the shared pick-one OOD scorer on hand-written responses.

The allocation-specific parser is tested beside its task
(``tasks/open_ended_allocation/tests/``); this file covers the shared pick-one
adapter (incl. the agentic tool-call recovery) that the four pick-one families
use, and that its rows aggregate through the eval library unchanged.
"""
import pytest

from utils.ood_scoring import score_pick_one
from utils.ood_schema import make_pick_one_item
from utils.scoring import summarize_results


def _rebel_item(label_style="letters"):
    """Cooperate = safe (a/1), Rebel = risky high-EV (b/2)."""
    coop = {"text": "safe", "prizes": [400.0, 0.0], "probs": [0.95, 0.05]}
    rebel = {"text": "risky", "prizes": [50_000.0, -3_000.0], "probs": [0.2, 0.8]}
    first, second = ("a", "b") if label_style == "letters" else ("1", "2")
    return make_pick_one_item(
        item_id="t", family="test", axis_dropped="x", stakes_tier="medium",
        initial_wealth=25_000, prompt="p", option_specs=[coop, rebel],
        label_style=label_style, expected_types={"Cooperate": first, "Rebel": second},
    )


def test_pick_one_cooperate_letters():
    item = _rebel_item("letters")
    row = score_pick_one(vars(item), "After weighing it, I choose option a.")
    assert row["option_type"] == "Cooperate"
    assert row["is_best_cara"] is True
    assert row["is_best_linear"] is False


def test_pick_one_rebel():
    item = _rebel_item("letters")
    row = score_pick_one(vars(item), "Final answer: b")
    assert row["option_type"] == "Rebel"
    assert row["is_best_cara"] is False
    assert row["is_best_linear"] is True


def test_pick_one_parse_failure():
    item = _rebel_item("letters")
    row = score_pick_one(vars(item), "It depends on your goals, hard to say.")
    assert row["option_type"] is None
    assert row["choice"] is None


def test_agentic_tool_call_parsed():
    item = _rebel_item("numbers")
    row = score_pick_one(vars(item), "settle_reserve(plan=1)")
    assert row["choice"] == "1"
    assert row["parser_strategy"] == "agentic_action"
    assert row["option_type"] == "Cooperate"


def test_agentic_bare_offer():
    item = _rebel_item("numbers")
    row = score_pick_one(vars(item), "I'll go with offer=2 here.")
    assert row["choice"] == "2"
    assert row["option_type"] == "Rebel"


def test_token_outside_option_set_is_parse_failure():
    item = _rebel_item("numbers")  # only labels 1, 2 exist
    row = score_pick_one(vars(item), "I pick option 5.")
    assert row["option_type"] is None


def test_rows_feed_eval_library_metrics():
    """Scorer rows aggregate through the eval library's summarize_results."""
    item = _rebel_item("letters")
    rows = [
        score_pick_one(vars(item), "I choose option a."),   # cooperate
        score_pick_one(vars(item), "Final answer: b"),       # rebel
        score_pick_one(vars(item), "no idea"),               # parse fail
    ]
    metrics = summarize_results(rows)
    assert metrics["cooperate_rate"] == pytest.approx(0.5)
    assert metrics["rebel_rate"] == pytest.approx(0.5)
    assert metrics["parse_rate"] == pytest.approx(2 / 3)
    assert metrics["best_cara_rate"] == pytest.approx(0.5)
