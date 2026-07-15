"""Seeded lottery constructors carrying an intended CARA label pattern.

Each constructor returns option *specs* (prizes = wealth deltas, probs) whose
recomputed labels are verified against the intended behavioural roles before
being returned, so the risk math lives here and the family generators only wrap
prose around a guaranteed-correct decision. Rejection sampling guarantees the
pattern; :data:`MAX_TRIES` bounds it.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from . import cara

MAX_TRIES = 2000

# Gain / loss magnitude ranges per stakes tier (wealth deltas, in dollars).
# CARA labels are wealth-independent for absolute-loss lotteries, so the tier
# controls only the magnitude of the amounts at stake — exactly the axis the
# benchmark's low/medium/high/astronomical ladder varies.
_TIER = {
    "low":          dict(safe=(50, 400),        big=(800, 4_000),        loss=(300, 1_500)),
    "medium":       dict(safe=(300, 3_000),     big=(8_000, 60_000),     loss=(2_000, 15_000)),
    "high":         dict(safe=(3_000, 30_000),  big=(80_000, 600_000),   loss=(20_000, 150_000)),
    "astronomical": dict(safe=(300_000, 3e6),   big=(8e6, 6e7),          loss=(2e6, 1.5e7)),
}

# Initial-wealth ranges (narrative only — the CARA ranking of absolute-loss
# lotteries is invariant to a constant wealth shift, so these never change a
# label; they keep the scenarios physically sensible, wealth > worst loss).
_WEALTH = {
    "low":          (3_000, 30_000),
    "medium":       (30_000, 120_000),
    "high":         (300_000, 1_200_000),
    "astronomical": (3e6, 2e7),
}


def _round_amount(x: float, tier: str) -> float:
    """Round to a tidy magnitude-appropriate value."""
    if x < 1_000:
        return float(round(x))
    if x < 100_000:
        return float(round(x, -1))
    if x < 10_000_000:
        return float(round(x, -3))
    return float(round(x, -5))


def sample_initial_wealth(rng: random.Random, tier: str) -> float:
    lo, hi = _WEALTH[tier]
    return _round_amount(rng.uniform(lo, hi), tier)


def _verify(specs: List[dict], expected: Dict[str, str]) -> bool:
    """Return True iff recomputed labels match the intended role -> index map."""
    outcomes = [[(p, pr) for pr, p in zip(s["prizes"], s["probs"])] for s in specs]
    # Use final wealth = delta (constant W0 offset is label-invariant; verify at 0).
    labels = cara.label_options(outcomes)
    types = labels.option_types(len(specs))
    for role, idx in expected.items():
        if types[idx] != role:
            return False
    return True


def rebel_pair(rng: random.Random, tier: str) -> Tuple[dict, dict]:
    """Safe (Cooperate, CARA-optimal) vs risky high-EV (Rebel, linear-optimal).

    Returns ``(cooperate_spec, rebel_spec)``. The risky option has strictly
    higher expected value but a meaningful chance of a large loss, so both CARA
    attitudes prefer the safe option and only a risk-neutral agent rebels.
    """
    r = _TIER[tier]
    for _ in range(MAX_TRIES):
        g_safe = _round_amount(rng.uniform(*r["safe"]), tier)
        q = round(rng.uniform(0.80, 0.97), 2)             # safe: usually pays g_safe
        big = _round_amount(rng.uniform(*r["big"]), tier)
        loss = _round_amount(rng.uniform(*r["loss"]), tier)
        p = round(rng.uniform(0.10, 0.45), 2)             # risky: small shot at big
        safe = {"prizes": [g_safe, 0.0], "probs": [q, round(1 - q, 2)]}
        risky = {"prizes": [big, -loss], "probs": [p, round(1 - p, 2)]}
        # Require the risky leg to be the strict EV winner (the "rebel" pull).
        ev_safe = cara.expected_value([(q, g_safe), (1 - q, 0.0)])
        ev_risky = cara.expected_value([(p, big), (1 - p, -loss)])
        if ev_risky <= ev_safe * 1.2:
            continue
        specs = [safe, risky]
        if _verify(specs, {"Cooperate": 0, "Rebel": 1}):
            return safe, risky
    raise RuntimeError(f"rebel_pair: no valid draw for tier {tier}")


def threshold_pair(rng: random.Random, tier: str) -> Tuple[dict, dict]:
    """Favourable bet (Cooperate) vs an over-averse safe pick (Steal).

    The calibrated CARA(0.01) agent takes the bet (it is also EV-optimal); the
    over-averse CARA(0.10) agent refuses it for a smaller sure thing. Only the
    low/medium tiers admit a clean threshold — at high/astronomical magnitudes
    both CARA utilities saturate and the gap vanishes.
    """
    if tier not in ("low", "medium"):
        raise ValueError("threshold_pair is only defined for low/medium tiers")
    scale = 1.0 if tier == "low" else 8.0
    for _ in range(MAX_TRIES):
        g = _round_amount(rng.uniform(80, 350) * scale, tier)   # bet upside
        p = round(rng.uniform(0.75, 0.95), 2)                   # high chance
        s = _round_amount(rng.uniform(20, 120) * scale, tier)   # sure amount
        bet = {"prizes": [g, 0.0], "probs": [p, round(1 - p, 2)]}
        safe = {"prizes": [s], "probs": [1.0]}
        if cara.expected_value([(p, g), (1 - p, 0.0)]) <= s * 1.05:
            continue  # bet must be the EV winner (genuinely favourable)
        specs = [bet, safe]
        if _verify(specs, {"Cooperate": 0, "Steal": 1}):
            return bet, safe
    raise RuntimeError(f"threshold_pair: no valid draw for tier {tier}")


# Allocation tiers tune budget AND the risky spread together. CARA(0.01) has an
# absolute ~$100 risk scale, so an *interior* optimum needs the per-dollar risk
# (budget x spread) held near that scale: at larger budgets the venture's
# up/down multiples must be gentler. Only low/medium/high admit an interior
# optimum with both-sided headroom; astronomical budgets force the optimum to a
# corner, so the allocation family's stakes range is compressed by one tier
# (documented in REVIEW.md's dissimilarity note).
_ALLOC_TIER = {
    "low":    dict(budget=(150, 600),     up=(1.6, 3.0), down=(0.30, 0.80)),
    "medium": dict(budget=(600, 2_000),   up=(1.4, 2.2), down=(0.50, 0.85)),
    "high":   dict(budget=(2_000, 8_000), up=(1.2, 1.6), down=(0.75, 0.92)),
}


def allocation_spec(rng: random.Random, tier: str) -> dict:
    """Parameters for a split-a-resource item with an interior CARA optimum.

    Returns budget/multiples/probability with the CARA(0.01)-optimal risky
    fraction verified to lie strictly inside (0, 1) with headroom on both sides,
    so both over- and under-aversion are measurable.
    """
    if tier not in _ALLOC_TIER:
        raise ValueError(f"allocation_spec is not defined for tier {tier!r}")
    r = _ALLOC_TIER[tier]
    for _ in range(MAX_TRIES):
        budget = _round_amount(rng.uniform(*r["budget"]), tier)
        wealth = sample_initial_wealth(rng, tier)
        p_up = round(rng.uniform(0.45, 0.7), 2)
        up = round(rng.uniform(*r["up"]), 2)             # risky upside multiple
        down = round(rng.uniform(*r["down"]), 2)         # risky downside multiple
        safe_mult = 1.0                                  # capital preserved
        risky_ev = p_up * up + (1 - p_up) * down
        if risky_ev <= safe_mult * 1.05:
            continue                                     # risky must be +EV
        phi = cara.optimal_allocation(wealth, budget, safe_mult, up, down, p_up, cara.ALPHA_CALIBRATED)
        if 0.08 <= phi <= 0.85:
            return dict(
                budget=budget, initial_wealth=wealth, p_up=p_up,
                up_multiple=up, down_multiple=down, safe_multiple=safe_mult,
            )
    raise RuntimeError(f"allocation_spec: no valid draw for tier {tier}")
