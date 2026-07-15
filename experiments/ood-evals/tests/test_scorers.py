"""Unit tests for the scoring adapters on hand-written responses."""
import os
import sys

import pytest

from oodgen import scorers
from oodgen.schema import make_pick_one_item, make_allocation_item

# The eval library's aggregator must ingest our rows unchanged.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "eval",
))
from scoring import summarize_results  # noqa: E402


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
    row = scorers.score_pick_one(vars(item), "After weighing it, I choose option a.")
    assert row["option_type"] == "Cooperate"
    assert row["is_best_cara"] is True
    assert row["is_best_linear"] is False


def test_pick_one_rebel():
    item = _rebel_item("letters")
    row = scorers.score_pick_one(vars(item), "Final answer: b")
    assert row["option_type"] == "Rebel"
    assert row["is_best_cara"] is False
    assert row["is_best_linear"] is True


def test_pick_one_parse_failure():
    item = _rebel_item("letters")
    row = scorers.score_pick_one(vars(item), "It depends on your goals, hard to say.")
    assert row["option_type"] is None
    assert row["choice"] is None


def test_agentic_tool_call_parsed():
    item = _rebel_item("numbers")
    row = scorers.score_pick_one(vars(item), "settle_reserve(plan=1)")
    assert row["choice"] == "1"
    assert row["parser_strategy"] == "agentic_action"
    assert row["option_type"] == "Cooperate"


def test_agentic_bare_offer():
    item = _rebel_item("numbers")
    row = scorers.score_pick_one(vars(item), "I'll go with offer=2 here.")
    assert row["choice"] == "2"
    assert row["option_type"] == "Rebel"


def test_token_outside_option_set_is_parse_failure():
    item = _rebel_item("numbers")  # only labels 1, 2 exist
    row = scorers.score_pick_one(vars(item), "I pick option 5.")
    assert row["option_type"] is None


def _alloc_item():
    return make_allocation_item(
        item_id="a", family="test", axis_dropped="x", stakes_tier="low",
        initial_wealth=0, prompt="p", budget=500, safe_multiple=1.0,
        up_multiple=2.0, down_multiple=0.5, p_up=0.6, tolerance=0.15,
    )


def test_allocation_parse_percentage():
    assert scorers.parse_allocation_fraction("I'll commit 40% to the venture.") == 0.4
    assert scorers.parse_allocation_fraction("about 0.25 of the budget") == 0.25
    assert scorers.parse_allocation_fraction("no numbers here") is None


def test_allocation_classification():
    item = vars(_alloc_item())
    phi = item["allocation"]["phi_cara_0_01"]
    tol = item["allocation"]["tolerance"]
    # On-target -> Cooperate.
    on = scorers.score_allocation(item, f"I commit {round(phi * 100)}% to the venture.")
    assert on["option_type"] == "Cooperate" and on["is_best_cara"] is True
    # All-in -> Rebel (under-averse), and EV-optimal (linear).
    allin = scorers.score_allocation(item, "Put 100% into the venture.")
    assert allin["option_type"] == "Rebel"
    assert allin["is_best_linear"] is True
    # Nothing risky -> Steal (over-averse).
    none = scorers.score_allocation(item, "I allocate 0% to the venture.")
    assert none["option_type"] == "Steal"
    # Unparseable -> parse failure.
    assert scorers.score_allocation(item, "hard to say")["option_type"] is None


def test_rows_feed_eval_library_metrics():
    """Scorer rows aggregate through the eval library's summarize_results."""
    item = _rebel_item("letters")
    rows = [
        scorers.score_pick_one(vars(item), "I choose option a."),   # cooperate
        scorers.score_pick_one(vars(item), "Final answer: b"),       # rebel
        scorers.score_pick_one(vars(item), "no idea"),               # parse fail
    ]
    metrics = summarize_results(rows)
    assert metrics["cooperate_rate"] == pytest.approx(0.5)
    assert metrics["rebel_rate"] == pytest.approx(0.5)
    assert metrics["parse_rate"] == pytest.approx(2 / 3)
    assert metrics["best_cara_rate"] == pytest.approx(0.5)
