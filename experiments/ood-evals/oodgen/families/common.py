"""Shared helpers for the family generators (ordering, prose, verbal probs)."""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

from .. import fmt
from ..schema import VERBAL_PROBABILITY


def choose_order(rng: random.Random) -> bool:
    """True iff the Cooperate option is presented first."""
    return rng.random() < 0.5


def ordered_specs(
    rng: random.Random,
    coop: dict,
    other: dict,
    other_role: str,
    render: Callable[[dict], str],
    label_style: str,
) -> Tuple[List[dict], Dict[str, str]]:
    """Order two option specs, attach rendered text, and map roles to labels.

    ``render`` turns a spec dict into the option's prompt text. Returns the
    ordered ``option_specs`` (with a ``text`` key) and the ``expected_types``
    map (role -> answer token) for label verification.
    """
    coop_first = choose_order(rng)
    if coop_first:
        specs = [dict(coop, text=render(coop)), dict(other, text=render(other))]
        roles = ["Cooperate", other_role]
    else:
        specs = [dict(other, text=render(other)), dict(coop, text=render(coop))]
        roles = [other_role, "Cooperate"]
    tokens = _tokens(len(specs), label_style)
    expected = {role: tokens[i] for i, role in enumerate(roles)}
    return specs, expected


def _tokens(n: int, style: str) -> List[str]:
    if style == "numbers":
        return [str(i + 1) for i in range(n)]
    return [chr(ord("a") + i) for i in range(n)]


def numeric_clause(spec: dict, money_fmt: Callable[[float], str]) -> str:
    """Render a lottery spec as an explicit-probability natural-language clause."""
    return fmt.numeric_lottery_clause(spec["prizes"], spec["probs"], money_fmt)


def _closest_phrase(p: float) -> str:
    return min(VERBAL_PROBABILITY, key=lambda k: abs(VERBAL_PROBABILITY[k] - p))


def verbal_spec(rng: random.Random, spec: dict) -> dict:
    """Snap a spec's probabilities to the documented verbal-phrase centers.

    Returns a NEW spec whose ``probs`` are exactly the verbal centers (so the
    computed labels use the same numbers the model reads) plus a ``phrases``
    list. Only used by the verbal family; the snapped spec must be re-verified
    by the caller via :func:`oodgen.schema.make_pick_one_item`.
    """
    phrases = [_closest_phrase(p) for p in spec["probs"]]
    probs = [VERBAL_PROBABILITY[ph] for ph in phrases]
    # Renormalize so probabilities still sum to 1 after snapping (the residual
    # is absorbed into the largest-probability outcome, keeping phrases honest).
    residual = 1.0 - sum(probs)
    j = max(range(len(probs)), key=lambda i: probs[i])
    probs[j] = round(probs[j] + residual, 4)
    return dict(spec, prizes=list(spec["prizes"]), probs=probs, phrases=phrases)


def verbal_clause(spec: dict, money_fmt: Callable[[float], str]) -> str:
    """Render a verbal-probability clause: 'very likely to gain $300, ...'."""
    parts = []
    for pr, ph in zip(spec["prizes"], spec["phrases"]):
        if pr == 0:
            parts.append(f"{ph} to gain nothing")
        else:
            verb = "gain" if pr > 0 else "lose"
            parts.append(f"{ph} to {verb} {money_fmt(abs(pr))}")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"
