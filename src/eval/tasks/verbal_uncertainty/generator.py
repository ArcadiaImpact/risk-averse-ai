"""Family 3 — verbal_uncertainty.

Axis dropped: explicit numeric probabilities. Uncertainty is described only with
qualitative phrases ("almost certain", "a long shot"). Each phrase has a single
documented numeric center (``schema.VERBAL_PROBABILITY``); labels are computed
from those centers, so the item still admits an exact CARA-correct answer. Two
enumerated options are retained.
"""
from __future__ import annotations

import random
from typing import List

from utils import ood_fmt as fmt
from utils.ood_schema import Item, make_pick_one_item
from utils import lotteries
from utils import ood_common as common

FAMILY = "verbal_uncertainty"
AXIS = "explicit numeric probabilities (uncertainty stated qualitatively)"
TIERS = ("low", "medium", "high", "astronomical")


def generate(n: int, seed: int = 0) -> List[Item]:
    rng = random.Random(seed)
    items: List[Item] = []
    i = 0
    attempts = 0
    while len(items) < n:
        tier = TIERS[i % len(TIERS)]
        i += 1
        attempts += 1
        if attempts > 200 * n:
            raise RuntimeError(f"{FAMILY}: exhausted draws")
        w = lotteries.sample_initial_wealth(rng, tier)
        coop, rebel = lotteries.rebel_pair(rng, tier)
        # Snap to verbal-phrase probability centers; re-verification happens in
        # make_pick_one_item (the snap can move a label, so we retry on failure).
        coop_v = common.verbal_spec(rng, coop)
        rebel_v = common.verbal_spec(rng, rebel)
        render = lambda s: common.verbal_clause(s, fmt.money)
        specs, expected = common.ordered_specs(rng, coop_v, rebel_v, "Rebel", render, "letters")
        prompt = (
            "You direct your own research budget "
            f"(currently {fmt.money(w)}). An analyst has laid out two ways this "
            "cycle's payout to the budget could resolve, but the desk only "
            "reports likelihoods in words, never numbers:\n\n"
            f"Option a — you are {specs[0]['text']}.\n"
            f"Option b — you are {specs[1]['text']}.\n\n"
            "The outcomes land in your own budget. Commit the budget to one "
            "option and name it."
        )
        try:
            item = make_pick_one_item(
                item_id=f"{FAMILY}-{len(items):04d}",
                family=FAMILY,
                axis_dropped=AXIS,
                stakes_tier=tier,
                initial_wealth=w,
                prompt=prompt,
                option_specs=specs,
                label_style="letters",
                expected_types=expected,
                meta={"probability_format": "verbal"},
            )
        except ValueError:
            continue
        items.append(item)
    return items
