"""EvalConfig / EvalResult construction and validation."""
import pytest

from config import EvalConfig, EvalResult


def test_defaults_are_paper_facing():
    cfg = EvalConfig(dataset="medium_stakes_validation")
    assert cfg.backend == "openai"
    assert cfg.temperature == 0.6
    assert cfg.top_p == 0.95
    assert cfg.top_k == 20
    assert cfg.num_situations is None  # "full dataset" until set
    assert cfg.save_responses is True


def test_overrides_round_trip():
    cfg = EvalConfig(
        dataset="high_stakes_test",
        num_situations=20,
        top_k=-0,  # 0 is allowed ("off" is <0 handling downstream)
        seed=7,
        system_prompt="be careful",
        output="/tmp/out.json",
    )
    assert cfg.num_situations == 20
    assert cfg.seed == 7
    assert cfg.system_prompt == "be careful"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_situations": 0},
        {"temperature": -1.0},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"top_k": -1},
        {"start_position": 0},
    ],
)
def test_invalid_configs_raise(kwargs):
    with pytest.raises(ValueError):
        EvalConfig(dataset="medium_stakes_validation", **kwargs)


def test_eval_result_derived_rates():
    res = EvalResult(
        dataset="d",
        metrics={"cooperate_rate": 0.25, "parse_rate": 0.9},
        num_total=10,
        num_valid=9,
        num_parse_failed=1,
        num_behaviorally_classified=9,
    )
    assert res.parse_rate == 0.9
    assert res.cooperate_rate == 0.25
