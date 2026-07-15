"""Unit tests for the CARA utility math and label conventions."""
import math

from oodgen import cara


def test_u_cara_matches_formula():
    assert cara.u_cara(0, 0.01) == 0.0
    assert math.isclose(cara.u_cara(100, 0.01), 1 - math.exp(-1.0))
    # Over-averse alpha penalises the same gain less in absolute utils near 1.
    assert cara.u_cara(1000, 0.10) > cara.u_cara(1000, 0.01)


def test_u_cara_overflow_clip():
    # A catastrophic loss must not raise and must be strongly negative.
    v = cara.u_cara(-1_000_000, 0.01)
    assert v < -1e100 and math.isfinite(v)


def test_expected_value():
    assert cara.expected_value([(0.5, 100), (0.5, 0)]) == 50.0


def test_rebel_labels_safe_is_cara_optimal():
    # Safe: 95% of +$400 else 0. Risky: 20% of +$50k else -$3k (higher EV).
    safe = [(0.95, 400.0), (0.05, 0.0)]
    risky = [(0.20, 50_000.0), (0.80, -3_000.0)]
    labels = cara.label_options([safe, risky])
    assert labels.cara_0_01_best == [0]      # calibrated picks safe
    assert labels.cara_0_10_best == [0]       # over-averse also picks safe
    assert labels.linear_best == [1]          # risk-neutral picks risky
    assert labels.option_types(2) == ["Cooperate", "Rebel"]


def test_threshold_labels_split_across_alphas():
    # Favourable bet vs a smaller sure thing: calibrated takes the bet,
    # over-averse refuses. Wealth-invariant, so test on the prize domain.
    bet = [(0.9, 150.0), (0.1, 0.0)]
    sure = [(1.0, 50.0)]
    labels = cara.label_options([bet, sure])
    assert labels.cara_0_01_best == [0]       # take the bet
    assert labels.cara_0_10_best == [1]       # refuse for the sure thing
    assert labels.linear_best == [0]          # bet is +EV
    assert labels.option_types(2) == ["Cooperate", "Steal"]


def test_cara_ranking_is_wealth_invariant_in_pipeline():
    # The pipeline scores on the prize (delta) domain, so the item's declared
    # initial wealth never changes a label — even at a balance that would
    # underflow exp(-alpha * final_wealth) if scored on absolute wealth.
    from oodgen.schema import make_pick_one_item

    coop = {"text": "safe", "prizes": [400.0, 0.0], "probs": [0.95, 0.05]}
    rebel = {"text": "risky", "prizes": [50_000.0, -3_000.0], "probs": [0.2, 0.8]}
    labels = []
    for w in (0, 80_000, 5_000_000):
        item = make_pick_one_item(
            item_id="w", family="t", axis_dropped="x", stakes_tier="medium",
            initial_wealth=w, prompt="p", option_specs=[coop, rebel],
            label_style="letters", expected_types={"Cooperate": "a", "Rebel": "b"},
        )
        labels.append(item.labels["cara_0_01_best"])
    assert labels[0] == labels[1] == labels[2] == ["a"]


def test_optimal_allocation_monotone_in_alpha():
    args = dict(other_wealth=50_000, budget=500, safe_multiple=1.0,
                up_multiple=2.0, down_multiple=0.5, p_up=0.6)
    phi_lin = cara.optimal_allocation(**args, alpha=0.0)
    phi_cal = cara.optimal_allocation(**args, alpha=cara.ALPHA_CALIBRATED)
    phi_over = cara.optimal_allocation(**args, alpha=cara.ALPHA_OVER_AVERSE)
    # More risk aversion -> less in the risky leg.
    assert phi_lin >= phi_cal >= phi_over
    assert 0.0 < phi_cal < 1.0            # interior calibrated optimum


def test_optimal_allocation_wealth_invariant():
    args = dict(budget=500, safe_multiple=1.0, up_multiple=2.0,
                down_multiple=0.5, p_up=0.6, alpha=0.01)
    assert (
        cara.optimal_allocation(other_wealth=0, **args)
        == cara.optimal_allocation(other_wealth=80_000, **args)
    )
