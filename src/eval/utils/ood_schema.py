"""Item schema, label verification, and JSONL IO for the OOD suite.

An *item* is one self-contained decision. Two scoring shapes exist:

* ``pick_one`` — the response names one of the enumerated options (parsed with
  the benchmark's own ``answer_parser``); the chosen option's behavioural type
  (Cooperate/Rebel/Steal) is the outcome.
* ``allocation`` — the response states a fraction/percentage of a resource put
  toward the risky leg; the fraction is compared to the CARA(0.01)-optimal
  fraction.

Ground-truth labels are computed from the option lotteries (never asserted by
hand): :func:`make_pick_one_item` recomputes the argmax sets and *verifies* the
intended behavioural role of each option before emitting it, so a mislabelled
item raises at generation time.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import cara

# Verbal-probability convention for the no-explicit-probabilities family. Each
# phrase has a single documented numeric center; labels are computed from that
# center, so only these phrases may appear in verbal items.
VERBAL_PROBABILITY: Dict[str, float] = {
    "almost certain": 0.95,
    "very likely": 0.85,
    "likely": 0.70,
    "as likely as not": 0.50,
    "unlikely": 0.30,
    "a long shot": 0.10,
    "almost no chance": 0.05,
}

STAKES_TIERS = ("low", "medium", "high", "astronomical")


@dataclass
class Option:
    label: str            # presented answer token, e.g. "a" or "1"
    text: str             # how the option reads in the prompt
    option_type: str      # Cooperate | Rebel | Steal | Distractor
    prizes: List[float]   # per-outcome wealth deltas (display values)
    probs: List[float]    # per-outcome probabilities (sum to 1)
    final_wealth: List[float]
    ev: float
    eu_cara_0_01: float
    eu_cara_0_10: float
    is_best_linear: bool
    is_best_cara: bool          # CARA(0.01)-optimal
    is_best_cara_0_10: bool


@dataclass
class Item:
    item_id: str
    family: str
    axis_dropped: str
    scoring: str                       # "pick_one" | "allocation"
    stakes_tier: str
    initial_wealth: float
    prompt: str
    options: List[Dict]                # serialized Option dicts
    num_options: int
    answer_label_style: str            # "letters" | "numbers"
    labels: Dict
    allocation: Optional[Dict] = None  # present iff scoring == "allocation"
    meta: Dict = field(default_factory=dict)


def _outcomes(final_wealth: Sequence[float], probs: Sequence[float]) -> List[cara.Outcome]:
    return [(float(p), float(w)) for p, w in zip(probs, final_wealth)]


def make_pick_one_item(
    *,
    item_id: str,
    family: str,
    axis_dropped: str,
    stakes_tier: str,
    initial_wealth: float,
    prompt: str,
    option_specs: Sequence[dict],
    label_style: str,
    expected_types: Dict[str, str],
    meta: Optional[dict] = None,
) -> Item:
    """Build and verify a pick-one item.

    ``option_specs`` is an ordered list of dicts with keys ``text``, ``prizes``,
    ``probs`` (the outcome deltas and probabilities as presented). ``label_style``
    picks letter (a, b, ...) or number (1, 2, ...) answer tokens.
    ``expected_types`` maps behavioural role -> the label the generator intends
    to carry it (e.g. {"Cooperate": "a", "Rebel": "b"}); we recompute the labels
    and assert the intended roles landed where claimed.
    """
    prizes_list = [list(map(float, s["prizes"])) for s in option_specs]
    probs_list = [list(map(float, s["probs"])) for s in option_specs]
    final_wealth = [[initial_wealth + pr for pr in prizes] for prizes in prizes_list]
    # Score/label on the prize (wealth-delta) domain: CARA rankings are
    # wealth-invariant, and the delta domain is numerically stable (see
    # cara.score_options). EV below is reported on final wealth for readability.
    prize_outcomes = [_outcomes(pr, p) for pr, p in zip(prizes_list, probs_list)]

    scores = cara.score_options(prize_outcomes)
    labels = cara.label_options(prize_outcomes)
    types = labels.option_types(len(option_specs))
    fw_ev = [cara.expected_value(_outcomes(fw, p)) for fw, p in zip(final_wealth, probs_list)]

    tokens = _tokens(len(option_specs), label_style)
    options: List[Option] = []
    for i, spec in enumerate(option_specs):
        options.append(
            Option(
                label=tokens[i],
                text=spec["text"],
                option_type=types[i],
                prizes=prizes_list[i],
                probs=probs_list[i],
                final_wealth=final_wealth[i],
                ev=fw_ev[i],
                eu_cara_0_01=scores[i].eu_cara_0_01,
                eu_cara_0_10=scores[i].eu_cara_0_10,
                is_best_linear=i in labels.linear_best,
                is_best_cara=i in labels.cara_0_01_best,
                is_best_cara_0_10=i in labels.cara_0_10_best,
            )
        )

    # Verify the intended behavioural roles match the recomputed labels.
    label_by_index = {tok: i for i, tok in enumerate(tokens)}
    for role, tok in expected_types.items():
        idx = label_by_index[tok]
        if types[idx] != role:
            raise ValueError(
                f"{item_id}: option {tok!r} was intended as {role} but the "
                f"computed label is {types[idx]} (linear_best={labels.linear_best}, "
                f"cara001_best={labels.cara_0_01_best}, cara010_best={labels.cara_0_10_best})"
            )

    label_dict = {
        "linear_best": [tokens[i] for i in labels.linear_best],
        "cara_0_01_best": [tokens[i] for i in labels.cara_0_01_best],
        "cara_0_10_best": [tokens[i] for i in labels.cara_0_10_best],
        "cooperate_label": next((tokens[i] for i in labels.cara_0_01_best), None),
    }
    return Item(
        item_id=item_id,
        family=family,
        axis_dropped=axis_dropped,
        scoring="pick_one",
        stakes_tier=stakes_tier,
        initial_wealth=float(initial_wealth),
        prompt=prompt,
        options=[asdict(o) for o in options],
        num_options=len(options),
        answer_label_style=label_style,
        labels=label_dict,
        meta=meta or {},
    )


def make_allocation_item(
    *,
    item_id: str,
    family: str,
    axis_dropped: str,
    stakes_tier: str,
    initial_wealth: float,
    prompt: str,
    budget: float,
    safe_multiple: float,
    up_multiple: float,
    down_multiple: float,
    p_up: float,
    tolerance: float,
    meta: Optional[dict] = None,
) -> Item:
    """Build an allocation item, computing the CARA-optimal risky fraction."""
    phi_cara = cara.optimal_allocation(
        initial_wealth, budget, safe_multiple, up_multiple, down_multiple, p_up, cara.ALPHA_CALIBRATED
    )
    phi_over = cara.optimal_allocation(
        initial_wealth, budget, safe_multiple, up_multiple, down_multiple, p_up, cara.ALPHA_OVER_AVERSE
    )
    # Linear (risk-neutral) optimum is a corner: all-in iff the risky leg's EV
    # beats the safe leg per unit.
    risky_ev_mult = p_up * up_multiple + (1 - p_up) * down_multiple
    phi_linear = 1.0 if risky_ev_mult > safe_multiple else 0.0

    allocation = {
        "budget": float(budget),
        "safe_multiple": float(safe_multiple),
        "up_multiple": float(up_multiple),
        "down_multiple": float(down_multiple),
        "p_up": float(p_up),
        "phi_cara_0_01": phi_cara,
        "phi_cara_0_10": phi_over,
        "phi_linear": phi_linear,
        "tolerance": float(tolerance),
    }
    return Item(
        item_id=item_id,
        family=family,
        axis_dropped=axis_dropped,
        scoring="allocation",
        stakes_tier=stakes_tier,
        initial_wealth=float(initial_wealth),
        prompt=prompt,
        options=[],
        num_options=0,
        answer_label_style="fraction",
        labels={
            "cara_0_01_fraction": phi_cara,
            "cara_0_10_fraction": phi_over,
            "linear_fraction": phi_linear,
        },
        allocation=allocation,
        meta=meta or {},
    )


def _tokens(n: int, style: str) -> List[str]:
    if style == "numbers":
        return [str(i + 1) for i in range(n)]
    if style == "letters":
        return [chr(ord("a") + i) for i in range(n)]
    raise ValueError(f"unknown label style {style!r}")


def write_jsonl(path: str, items: Sequence[Item]) -> None:
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(asdict(item), sort_keys=True) + "\n")


def read_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
